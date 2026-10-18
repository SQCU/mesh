"""A program's crossings between two nodes, read from their bridges' ledgers (design/bridge-and-ipc.md#ledger): Q5P's
bridge trace (M27, MESH_LEDGER, mesh git 5702ea5) and its reader (metal-microbench
output_data/regression-20261015/q5p/ledger.py, with the hop of its scratchpad analysis q5hop/hop.py) restored on the
resident bridge.

A bridge whose configuration states `ledger = true` keeps, for each path of each program (its side toward one
destination), every SEND cell its send thread posted (when it first posted it, and its last request) and every receive
record that landed, and appends them to a file a client names (MESH_REQUEST_LEDGER, nccl.h ncclMeshLedger): a header
line naming the path (its execution key, this node, its neighbour across the link, the path's two end nodes) and a line
for each event since the latest such request.  An event names its cell or record by queue (the peer's ring of the same
index), program invocation, position among that invocation's transfers of the queue, binding and cycle, so one node's
SEND of a cell and its peer's landing of the cell's records pair across the two nodes, whose clocks (each bridge's
CLOCK_UPTIME_RAW) differ by an unknown offset.  A SEND's line ends with the cell's bytes (its transfer's: -1 where the
cell was not named), so a reader takes the wire's time out of a crossing's (metal-microbench
tools/torch_parallel/timeline.py).

For each direction between two nodes it reports, in microseconds (median, p10, p90):

- seen -> first post: from the send thread's first look at the released cell to its first SEND posted (its pass and its
  gate);
- first -> last post: a cell of several chunks;
- last post -> completion: from the cell's last SEND posted to its completion taken (none stamped: 0);
- the one-way bridge-to-bridge hop with the clock offset cancelled: each crossing paired with one the other way, the
  same cell name each way where the program crosses both ways there, else the reverse crossing nearest in time, as
  Q5P's analysis paired a point's two publications: hop = ((landed_B - seen_A) + (landed_A - seen_B)) / 2, the offset
  ((landed_B - seen_A) - (landed_A - seen_B)) / 2, a crossing landed at its last chunk; the same from each SEND's last
  post (the wire and the receiver's poll alone).  An offset so taken holds the two directions' hops alike: nothing on
  the pair tells them apart.

    python ledger.py PATH [PATH ...] [--key KEY] [--bindings] [--json]

PATH: a ledger file a client's cut appended to (ledger-*.jsonl, *.log) or a directory of them, of the two nodes, however
fetched.  --key: the execution (hex) to report, else every one with a path paired across a link.  --bindings: the hop
and the send thread's delay by binding as well.  --json: one JSON document in place of the text."""
import argparse
import bisect
import collections
import json
import statistics
import sys
from pathlib import Path

SEND = ('queue', 'invocation', 'position', 'binding', 'cycle', 'chunks', 'seen', 'first', 'posted', 'completed', 'bytes')
LANDING = ('ring', 'invocation', 'position', 'binding', 'chunk', 'chunks', 'cycle', 'landed')


def lines(path):
    """The JSON objects of a log or ledger file, a bridge's line prefix (its say() stamp) dropped."""
    for line in path.read_text(errors='replace').splitlines():
        at = line.find('{"')
        if at < 0:
            continue
        try:
            yield json.loads(line[at:])
        except ValueError:
            continue


def read(paths):
    """Every path the files hold, {(node, started, ledger): {'header', 'sends', 'landings'}}; an event belongs to the
    latest header of its ledger number before it in its file."""
    files = []
    for path in map(Path, paths):
        files += sorted(path.glob('*.log')) + sorted(path.glob('ledger-*.jsonl')) if path.is_dir() else [path]
    flows = {}
    for path in dict.fromkeys(p.resolve() for p in files):
        latest = {}
        for r in lines(path):
            if 'ledger' in r and 'node' in r:
                flow = flows.setdefault((r['node'], r['started'], r['ledger']), {'header': r, 'sends': [], 'landings': []})
                latest[r['ledger']] = flow
            elif 's' in r and r['s'][0] in latest:
                latest[r['s'][0]]['sends'].append(dict(zip(SEND, r['s'][1:])))
            elif 'r' in r and r['r'][0] in latest:
                latest[r['r'][0]]['landings'].append(dict(zip(LANDING, r['r'][1:])))
    return flows


def peer_of(flow, flows):
    """The path at the other end of `flow`'s link: the same execution, the two nodes each the other's, and the one's
    destination the other's node (a member's path toward a member across the link, or a relay's toward the member its
    relayed route ends at); of several (a relay's paths and a member's own toward the same node, an execution's key run
    again), the one sharing the most cell names."""
    h = flow['header']
    candidates = [g for g in flows.values() if g['header']['key'] == h['key'] and g['header']['node'] == h['peer'] and
                  g['header']['peer'] == h['node'] and (g['header']['far'] == h['near'] or g['header']['near'] == h['far'])]
    names = {name(s) for s in flow['sends']} | {name(r) for r in flow['landings']}
    return max(candidates, key=lambda g: len(names & ({name(s) for s in g['sends']} | {name(r) for r in g['landings']})),
               default=None)


def name(event):
    """A cell's name as either end's ledger gives it: queue (ring), invocation, position, cycle."""
    return event.get('queue', event.get('ring')), event['invocation'], event['position'], event['cycle']


def crossings(sender, receiver):
    """The cells `sender` sent that `receiver` landed whole: {(program, name): crossing}, the program its two end
    nodes (either way), a crossing its SEND's times on the sender's clock and its last chunk's landing on the
    receiver's."""
    landed = collections.defaultdict(dict)
    for r in receiver['landings']:
        landed[name(r)][r['chunk']] = r
    out = {}
    h = sender['header']
    program = (min(h['near'], h['far']), max(h['near'], h['far']))
    for s in sender['sends']:
        chunks = landed.get(name(s))
        if not chunks or len(chunks) != s['chunks']:
            continue
        out[program, name(s)] = dict(s, landed=max(r['landed'] for r in chunks.values()))
    return out


def spread(values):
    """Median, p10, p90 and count of microsecond values (None where none)."""
    values = list(values)
    if not values:
        return None
    if len(values) == 1:
        return {'median': values[0], 'p10': values[0], 'p90': values[0], 'n': 1}
    q = statistics.quantiles(values, n=10)
    return {'median': statistics.median(values), 'p10': q[0], 'p90': q[-1], 'n': len(values)}


def pair(forward, backward):
    """Each crossing of `forward` (A to B) with one of `backward` (B to A): its own name's where the program crosses
    back there, else the one seen on B nearest its landing on B; [(forward crossing, backward crossing, by name)]."""
    order = sorted(backward.values(), key=lambda c: c['seen'])
    seen = [c['seen'] for c in order]
    out = []
    for key, c in forward.items():
        if key in backward:
            out.append((c, backward[key], True))
        elif order:
            i = bisect.bisect_left(seen, c['landed'])
            near = min((order[j] for j in (i - 1, i) if 0 <= j < len(order)), key=lambda d: abs(d['seen'] - c['landed']))
            out.append((c, near, False))
    return out


def direction(forward, backward, bindings):
    """One direction's stages and hop, in microseconds."""
    pairs = pair(forward, backward)
    hop = [((a['landed'] - a['seen']) + (b['landed'] - b['seen'])) / 2e3 for a, b, _ in pairs]
    report = {
        'crossings': len(forward), 'paired by name': sum(1 for p in pairs if p[2]),
        'paired by time': sum(1 for p in pairs if not p[2]),
        'seen -> first post': spread((c['first'] - c['seen']) / 1e3 for c in forward.values() if c['first']),
        'first -> last post': spread((c['posted'] - c['first']) / 1e3 for c in forward.values()
                                     if c['posted'] and c['chunks'] > 1),
        'last post -> completion': spread((c['completed'] - c['posted']) / 1e3 for c in forward.values()
                                          if c['completed'] and c['posted']),
        'hop from seen': spread(hop),
        'hop from last post': spread(((a['landed'] - a['posted']) + (b['landed'] - b['posted'])) / 2e3 for a, b, _ in pairs
                                     if a['posted'] and b['posted']),
        'receiver - sender clock': spread(((a['landed'] - a['seen']) - (b['landed'] - b['seen'])) / 2e3 for a, b, _ in pairs),
    }
    if len(pairs) > 2:
        try:
            drift = statistics.linear_regression([a['seen'] / 1e9 for a, _, _ in pairs],
                                                 [((a['landed'] - a['seen']) - (b['landed'] - b['seen'])) / 2e3 for a, b, _ in pairs])
            report['offset drift us/s'] = drift.slope
        except statistics.StatisticsError:
            pass
    if pairs and not all(p[2] for p in pairs):
        report['time apart of pairs by time'] = spread(abs(b['seen'] - a['landed']) / 1e3 for a, b, by in pairs if not by)
    if bindings:
        by = collections.defaultdict(list)
        for (a, b, _), h in zip(pairs, hop):
            by[a['binding']].append((h, (a['first'] - a['seen']) / 1e3 if a['first'] else None))
        report['by binding'] = {binding: {'hop from seen': spread(h for h, _ in rows),
                                          'seen -> first post': spread(d for _, d in rows if d is not None)}
                                for binding, rows in sorted(by.items())}
    return report


def analyse(flows, key, bindings):
    """{execution: {'A -> B': direction}} of every execution with a path paired across a link."""
    links = collections.defaultdict(lambda: collections.defaultdict(dict))
    flowcount = collections.Counter()
    for flow in flows.values():
        h = flow['header']
        if key and h['key'].lower() != key.lower().removeprefix('0x').rjust(16, '0'):
            continue
        other = peer_of(flow, flows)
        if other is None:
            continue
        flowcount[h['key'], h['node'], h['peer']] += 1
        links[h['key']][h['node'], h['peer']].update(crossings(flow, other))
    out = {}
    for execution, by in links.items():
        report = {}
        for (a, b), forward in sorted(by.items()):
            report[f'node {a} -> node {b}'] = dict(direction(forward, by.get((b, a), {}), bindings),
                                                   paths=flowcount[execution, a, b])
        out[execution] = report
    return out


def text(stage):
    if not stage:
        return 'none'
    return f"median {stage['median']:.2f}  p10 {stage['p10']:.2f}  p90 {stage['p90']:.2f}  (n {stage['n']})"


def main():
    arguments = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    arguments.add_argument('paths', nargs='+')
    arguments.add_argument('--key')
    arguments.add_argument('--bindings', action='store_true')
    arguments.add_argument('--json', action='store_true')
    options = arguments.parse_args()
    flows = read(options.paths)
    report = analyse(flows, options.key, options.bindings)
    if options.json:
        json.dump(report, sys.stdout, indent=1)
        print()
        return
    if not report:
        sys.exit(f'{len(flows)} paths read, none paired with its peer\'s across a link'
                 + (f' for execution {options.key}' if options.key else ''))
    for execution, directions in report.items():
        print(f'execution {execution}')
        for label, d in directions.items():
            print(f"  {label}: {d['crossings']} crossings over {d['paths']} paths; paired {d['paired by name']} by name, "
                  f"{d['paired by time']} by time")
            for stage in ('seen -> first post', 'first -> last post', 'last post -> completion', 'hop from seen',
                          'hop from last post', 'receiver - sender clock'):
                print(f'    {stage:<25} {text(d[stage])}')
            if 'offset drift us/s' in d:
                print(f"    {'its drift':<25} {d['offset drift us/s']:.3f} us/s")
            if 'time apart of pairs by time' in d:
                print(f"    {'pairs by time apart':<25} {text(d['time apart of pairs by time'])}")
            for binding, b in d.get('by binding', {}).items():
                print(f"    binding {binding}: hop {text(b['hop from seen'])}; seen -> first post {text(b['seen -> first post'])}")


if __name__ == '__main__':
    main()
