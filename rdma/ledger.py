"""A prepared call's stretches from its bridges' ledgers (mesh-flow.c M27, a call run with MESH_LEDGER=1): each rank's
own work between its collectives, the evidence allocate.Balancer balances on (design/heterogeneity.md R2, R3).

A bridge's log holds, per link, its receive records' layout (trace_layout), each transfer's binding (trace_binding:
its begin, the invocations it is active, its chunks or SEND cells) and every SEND observed and receive landed
(native_trace).  Bindings are the call's crossings in the caller's order (mesh_transfers_prepare), so a program's
crossings list, each with the point it serves and the rank publishing it, names every event: `events` gives, per
invocation and point, the rank's publication (its first SEND's observation) and its landing (its last chunk's).

A call runs one or more programs (sequences of crossings), each over its invocations; `stretches` gives each of the
rank's publications' stretch, from the latest of its own events at the program's earlier points in that invocation
(the previous invocation's last where none) to the publication, in seconds an invocation, named
'<program>:<point>'.  Nothing here knows what the program computes."""
import collections
import json

RECORD = 128  # sizeof(struct prepared_receive) (mesh-flow.c), the stride of a link's receive records
SEND = 256    # sizeof(struct mesh_send) (mesh.h), the stride of a transfer's SEND cells


def events(text, crossings):
    """{(invocation, point): {'send': ns, 'receive': ns}} of one rank's bridge log, `crossings` the call's in binding
    order (each with its 'point')."""
    lines = [json.loads(line) for line in text.splitlines() if line.startswith('{"trace_') or line.startswith('{"native_trace"')]
    base = {r['trace_layout']: r['receive_base'] for r in lines if 'trace_layout' in r}
    total = {r['trace_layout']: r['invocations'] for r in lines if 'trace_layout' in r}
    cells = [(r['trace_binding'], r['cells'], r['begin'], r['active'], r['binding']) for r in lines
             if 'trace_binding' in r and r['direction'] == 0]
    rings = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in lines:
        if 'trace_binding' in r and r['direction'] == 1:
            rings[r['trace_binding']][r['ring']].append((r['begin'], r['active'], r['chunks'], r['binding']))
    record = {}
    for link, by_ring in rings.items():
        index = 0
        for ring in sorted(by_ring):
            for t in range(total[link]):
                for begin, active, chunks, binding in by_ring[ring]:
                    if begin <= t < begin + active:
                        for _ in range(chunks):
                            record[link, index] = (t, binding)
                            index += 1
    out = collections.defaultdict(dict)
    for r in lines:
        if 'native_trace' not in r:
            continue
        if r['direction'] == 0:
            for link, first, begin, active, binding in cells:
                if link == r['native_trace'] and first <= r['identity'] < first + (active + 1) * SEND and binding < len(crossings):
                    at = out[begin + (r['identity'] - first) // SEND, crossings[binding]['point']]
                    at['send'] = min(at.get('send', r['ns'][0]), r['ns'][0])
        else:
            found = record.get((r['native_trace'], (r['identity'] - base[r['native_trace']]) // RECORD))
            if found and found[1] < len(crossings):
                at = out[found[0], crossings[found[1]]['point']]
                at['receive'] = max(at.get('receive', r['ns'][1]), r['ns'][1])
    return out


def stretches(seen, programs):
    """{'<program>:<point>': [seconds an invocation]} of one rank: `seen` its events, `programs` {name: (its points in
    order, its invocations)}; each stretch ends at the rank's publication of a point."""
    out = collections.defaultdict(list)
    by = collections.defaultdict(dict)
    for (t, point), at in seen.items():
        by[t][point] = at
    previous = None
    for t in sorted(by):
        program = next((name for name, (_, invocations) in programs.items() if t in invocations), None)
        if program is None:
            continue
        order = {point: k for k, point in enumerate(programs[program][0])}
        mine = {p: at for p, at in by[t].items() if p in order}
        for point, at in mine.items():
            sent = at.get('send')
            if sent is None:
                continue
            before = [v for p, a in mine.items() if order[p] < order[point] for v in a.values() if v <= sent]
            start = max(before) if before else previous
            if start is not None:
                out[f'{program}:{point}'].append((sent - start) / 1e9)
        previous = max((v for a in mine.values() for v in a.values()), default=previous)
    return dict(out)
