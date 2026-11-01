"""A stalled program's wait-for graph across its ranks, from the tables the bridge read (nccl-mesh.c plan_record, one
file a rank: MESH_PLAN_RECORD) and each node's stream words at the stall (mesh-stat streams, its last line before a
loss).  Each stream's frontier, its first SEND not complete and its first landing not landed, is a vertex; each condition
of it that does not hold is an edge to what would make it hold: a word of the same rank (a landing's or SEND's `after`
on a channel's `sent`, `landed` or summed word; a SEND's readiness on `ready` or a landing), or, for a SEND whose
conditions hold, the peer's landing of the same units on the same cable (the hardware's credit: it completes once that
receive is posted), and for a landing whose conditions hold, the peer's SEND of its units.  A cycle is the deadlock.

python3 plan_cycles.py <dir>: <dir>/<node>/plan-*.jsonl and <dir>/<node>/streams.jsonl, a directory a node."""
import glob
import json
import sys
from pathlib import Path

import mesh_graph

LOST = 2**64 - 1


def calls(path):
    """a rank's calls in present order: each channel's peer, cable, words and lists with absolute unit positions"""
    out = []
    for line in open(path):
        record = json.loads(line)
        for ch in record['ch']:
            for land in (0, 1):
                at = ch['base'][land]
                entries = []
                for units, word, base, after in ch['lists'][land]:
                    entries.append({'start': at, 'end': at + units, 'word': word, 'base': base, 'after': after})
                    at += units
                ch['lists'][land] = entries
        out.append(record)
    return out


def stall(path):
    """the node's last stream words before any is lost: {stream: (peer, ready, sent, landed, taken0, taken1)}"""
    last = None
    for line in open(path):
        row = json.loads(line)
        if row['streams'] and all(s[4] != LOST and s[5] != LOST for s in row['streams']):
            last = row
    return {s[0]: tuple(s[1:2] + s[3:]) for s in last['streams']} if last else {}


def main(directory):
    ranks = {}
    for node in sorted(Path(directory).iterdir()):
        plans = glob.glob(str(node / 'plan-*.jsonl'))
        if not plans or not (node / 'streams.jsonl').exists():
            continue
        record = calls(plans[0])
        rank = record[0]['rank']
        channels = {}
        for call_index, call in enumerate(record):
            for ch in call['ch']:
                c = channels.setdefault(ch['h'], {'peer': ch['peer'], 'cable': ch['cable'], 'words': (ch['sent'], ch['landed'], ch['ready']),
                                                  'lists': ([], [])})
                for land in (0, 1):
                    for e in ch['lists'][land]:
                        e['call'] = call_index
                        c['lists'][land].append(e)
        streams = stall(node / 'streams.jsonl')
        # a channel's stream: the rank's channels by their `sent` word's offset, its streams by index, in the same order
        order = sorted(channels, key=lambda h: channels[h]['words'][0])
        held = sorted(streams)
        for h, t in zip(order, held):
            channels[h]['now'] = dict(zip(('peer', 'ready', 'sent', 'landed', 'left0', 'left1'), streams[t]))
            if channels[h]['now']['peer'] != channels[h]['peer']:
                raise SystemExit(f'{node.name}: stream {t} names peer {channels[h]["now"]["peer"]}, channel {h} peer {channels[h]["peer"]}')
        node_id = next(iter(channels.values()))['peer'] if False else None
        ranks[rank] = {'name': node.name, 'channels': channels}
    # each rank's node number: the peer its peers name it by
    numbers = {}
    for r, data in ranks.items():
        for c in data['channels'].values():
            numbers.setdefault(c['peer'], None)
    by_word = {}
    for r, data in ranks.items():
        for h, c in data['channels'].items():
            for kind, offset in zip(('sent', 'landed', 'ready'), c['words']):
                by_word[(r, offset)] = (kind, h)

    def value(r, offset):
        kind_h = by_word.get((r, offset))
        if not kind_h:
            return None, None
        kind, h = kind_h
        return kind_h, ranks[r]['channels'][h]['now'][kind]

    def frontier(r, h, land):
        c = ranks[r]['channels'][h]
        done = c['now']['landed' if land else 'sent']
        return next((i for i, e in enumerate(c['lists'][land]) if e['end'] > done), None)

    # a rank's node number is the peer the other ranks' channels toward it name: found from the cables
    node_of = {}
    for r, data in ranks.items():
        for h, c in data['channels'].items():
            for r2, data2 in ranks.items():
                if r2 == r:
                    continue
                for h2, c2 in data2['channels'].items():
                    if c2['cable'] == c['cable'] and c2['peer'] != c['peer']:
                        node_of[r] = c2['peer']
    rank_of = {n: r for r, n in node_of.items()}

    def paired(r, h):
        c = ranks[r]['channels'][h]
        p = rank_of.get(c['peer'])
        if p is None:
            return None, None
        for h2, c2 in ranks[p]['channels'].items():
            if c2['cable'] == c['cable']:
                return p, h2
        return p, None

    edges = {}
    for r, data in ranks.items():
        for h, c in data['channels'].items():
            for land in (0, 1):
                i = frontier(r, h, land)
                if i is None:
                    continue
                e = c['lists'][land][i]
                v = (r, h, land, i)
                waits = []
                for word, base in e['after']:
                    kind_h, now = value(r, word)
                    if kind_h is None:
                        waits.append(('word?', word, base))
                    elif now < base:
                        k, q = kind_h
                        waits.append(('after', (r, q, 0 if k == 'sent' else 1, frontier(r, q, 0 if k == 'sent' else 1)), f'{k} {now} < {base}'))
                if not land:
                    if e['word']:
                        kind_h, now = value(r, e['word'])
                        if kind_h is None:
                            waits.append(('producer', e['word'], e['base']))
                        elif now < e['base'] + (c['now']['sent'] - e['start']) + 1:
                            k, q = kind_h
                            waits.append(('ready', (r, q, 1, frontier(r, q, 1)), f'{k} {now} < {e["base"]}'))
                    elif c['now']['ready'] <= max(c['now']['sent'], e['start']):
                        waits.append(('published', None, f'ready {c["now"]["ready"]} <= {e["start"]}'))
                edges[v] = (e, waits)
    # where its own conditions hold: a landing (posted) waits on the SEND that fills it; a SEND (posted) only on its
    # landing's posting, so only where that landing's own conditions do not hold (the hardware's credit)
    for v, (e, waits) in list(edges.items()):
        r, h, land, i = v
        p, h2 = paired(r, h)
        if waits or p is None or h2 is None:
            continue
        to = (p, h2, 1 - land, frontier(p, h2, 1 - land))
        if land:
            waits.append(('data', to, ''))
        elif to in edges and edges[to][1]:
            waits.append(('credit', to, 'its landing not posted'))
    names = {r: d['name'] for r, d in ranks.items()}

    def label(v):
        if v is None or v[3] is None:
            return str(v)
        r, h, land, i = v
        c = ranks[r]['channels'][h]
        e = c['lists'][land][i]
        return (f"{names[r]} {'landing' if land else 'SEND'} {i} of channel {h} (cable {c['cable']}, peer "
                f"{names.get(rank_of.get(c['peer']), c['peer'])}), call {e['call']}, units {e['start']}-{e['end']}")
    for v, (e, waits) in sorted(edges.items()):
        print(label(v))
        for kind, to, why in waits:
            print(f'    waits ({kind} {why}) on {label(to) if isinstance(to, tuple) else to}')
    # cycles over the frontier
    graph = {v: [to for _, to, _ in waits if isinstance(to, tuple) and to in edges] for v, (_, waits) in edges.items()}
    cycles = mesh_graph.cycles(graph)
    for cycle in cycles:
        print('CYCLE:')
        for v in cycle:
            print('   ', label(v))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
