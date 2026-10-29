"""The collective solver over arbitrary topologies (the operator, 2026-10-29: "simulating arbitrary topologies of nodes and
the right solutions for them, assuming an average degree of 1.5 and a k sampled uniformly from 2 to 128").

Each sample is k nodes, k uniform in [2, 128], with round(0.75 k) random links (average degree 1.5), at bandwidths
drawn from the Thunderbolt rates. The solver runs on the largest set of them the links join, as the hourly does
(mesh_hourly.run_workload). libmesh solves it as libnccl-mesh does:
- mesh_routes, then mesh_trees_pack with no roots, eps 0 and most n (nccl-mesh.c comm_routes), then mesh_trees_check;
- mesh_compile for every collective, at sampled roots, then mesh_program_check_members;
- each rank's steps cut (mesh_program_steps_cut), and each channel's lists laid (mesh_lay).

Then each program is judged three ways:
- order: rank a's send list toward b and b's landing list from a hold the same pieces in the same order;
- progress: every rank's steps complete. A step waits on its rank's earlier steps over overlapping elements of the same
  buffer, two reads excepted (nccl-mesh.c ready: "only overlapping earlier receives create data dependencies"), and a
  receive on its matched SEND;
- result: every element of every rank's result holds exactly the contributions the collective defines, each once.

`python3 mesh_simulate.py [--samples N] [--seed S] [--library PATH]`: one line a failure, then a summary; exit 1 on any.
"""
import argparse
import ctypes as C
import random
import sys
from pathlib import Path

ALLREDUCE, BROADCAST, REDUCE, REDUCE_SCATTER, ALLGATHER = range(5)
NAMES = {ALLREDUCE: 'all-reduce', BROADCAST: 'broadcast', REDUCE: 'reduce', REDUCE_SCATTER: 'reduce-scatter',
         ALLGATHER: 'all-gather'}
SEND, REDUCE_STEP, COPY = range(3)
GRAPH = 3
ENTRIES = 4096  # etc/bridge/layout.h MESH_STREAM_ENTRIES: a list's entries one group may lay
RATES = (40e9, 80e9, 120e9)


class LinkMap(C.Structure):
    _fields_ = [('kind', C.c_uint32), ('nodes', C.c_uint32), ('links', C.c_uint32), ('link', C.POINTER(C.c_uint32 * 2)),
                ('bandwidth', C.POINTER(C.c_uint64)), ('relays', C.c_uint32), ('next', C.POINTER(C.c_uint32)),
                ('mark', C.POINTER(C.c_uint32))]


class Trees(C.Structure):
    _fields_ = [('nodes', C.c_uint32), ('count', C.c_uint32), ('root', C.POINTER(C.c_uint32)),
                ('parent', C.POINTER(C.c_uint32)), ('log_weight', C.POINTER(C.c_double))]


class Operand(C.Structure):
    _fields_ = [('type', C.c_uint32), ('element_bytes', C.c_uint32), ('elements', C.c_uint64)]


class Move(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ('op', 'peer', 'round', 'buffer', 'part', 'parts')]


class Program(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ('what', 'root', 'nodes', 'parts', 'scratch', 'segments')] + [
        ('segment', C.POINTER(C.c_uint32)), ('log_weight', C.POINTER(C.c_double)), ('first', C.POINTER(C.c_uint32)),
        ('move', C.POINTER(Move))]


class Step(C.Structure):
    _fields_ = [('op', C.c_uint32), ('peer', C.c_uint32), ('round', C.c_uint32), ('buffer', C.c_uint32),
                ('first', C.c_uint64), ('piece', Operand)]


class Cable(C.Structure):
    _fields_ = [('wire', C.c_uint64), ('bandwidth', C.c_uint64), ('node', C.c_uint32 * 2)]


class Fabric(C.Structure):
    _fields_ = [('nodes', C.c_uint32), ('cables', C.c_uint32), ('cable', C.POINTER(Cable))]


class Paths(C.Structure):
    _fields_ = [('count', C.c_uint32), ('first', C.POINTER(C.c_uint32)), ('cable', C.POINTER(C.c_uint32)),
                ('weight', C.POINTER(C.c_uint64))]


class Transfer(C.Structure):
    _fields_ = [('step', Step), ('cable', C.c_uint32), ('padding', C.c_uint32)]


class Place(C.Structure):
    _fields_ = [('step', C.c_uint32), ('send', C.c_uint32), ('entry', C.c_uint64)]


def library(path):
    lib = C.CDLL(str(path))
    lib.mesh_routes.argtypes = [C.POINTER(LinkMap), C.POINTER(C.c_uint32)]
    lib.mesh_link_map_joined.argtypes = [C.POINTER(LinkMap), C.POINTER(C.c_uint32)]
    lib.mesh_trees_pack.argtypes = [C.POINTER(LinkMap), C.POINTER(C.c_uint64), C.c_double, C.c_uint32,
                                    C.POINTER(C.c_uint32), C.POINTER(C.c_uint32), C.POINTER(C.c_double)]
    lib.mesh_trees_pack.restype = C.c_uint32
    lib.mesh_trees_check.argtypes = [C.POINTER(LinkMap), C.POINTER(Trees)]
    lib.mesh_compile_room.argtypes = [C.POINTER(Trees), C.POINTER(C.c_uint32), C.POINTER(C.c_uint32)]
    lib.mesh_compile.argtypes = [C.POINTER(Trees), C.c_uint32, C.c_uint32, C.c_int, C.POINTER(C.c_uint64),
                                 C.POINTER(Program), C.POINTER(C.c_uint32), C.POINTER(C.c_double),
                                 C.POINTER(C.c_uint32), C.POINTER(Move)]
    lib.mesh_program_check_members.argtypes = [C.POINTER(Program), Operand, C.POINTER(C.c_uint64), C.c_uint64,
                                               C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)]
    lib.mesh_program_steps_cut.argtypes = [C.POINTER(Program), C.c_uint32, Operand, C.POINTER(C.c_uint64), C.c_uint64,
                                           C.POINTER(Step)]
    lib.mesh_program_steps_cut.restype = C.c_uint32
    lib.mesh_lay.argtypes = [C.POINTER(Step), C.c_uint32, C.c_uint32, C.c_int, C.c_uint64, C.POINTER(Place),
                             C.POINTER(C.c_uint32)]
    lib.mesh_lay.restype = C.c_uint64
    lib.mesh_write_routes_default.argtypes = [C.POINTER(Fabric), C.POINTER(C.POINTER(Paths))]
    lib.mesh_write_routes_free.argtypes = [C.POINTER(Paths), C.c_uint32]
    lib.mesh_write_compile.argtypes = [C.POINTER(Fabric), C.c_uint32, C.c_uint32, C.c_uint32, C.POINTER(Paths), Operand,
                                       C.c_uint64, C.c_uint64, C.POINTER(Transfer), C.c_uint32]
    return lib


def topology(rng):
    """k nodes, round(0.75 k) links (average degree 1.5), then the largest part the links join, renumbered."""
    k = rng.randint(2, 128)
    links = [tuple(rng.sample(range(k), 2)) for _ in range(max(1, round(0.75 * k)))]
    adjacent = {v: set() for v in range(k)}
    for a, b in links:
        adjacent[a].add(b)
        adjacent[b].add(a)
    seen, parts = set(), []
    for v in range(k):
        if v in seen:
            continue
        part, todo = set(), [v]
        while todo:
            x = todo.pop()
            if x not in part:
                part.add(x)
                todo.extend(adjacent[x] - part)
        seen |= part
        parts.append(part)
    largest = sorted(max(parts, key=len))
    index = {v: i for i, v in enumerate(largest)}
    kept = [(index[a], index[b]) for a, b in links if a in index and b in index]
    return k, len(largest), kept, [rng.choice(RATES) for _ in kept]


def solve(lib, n, links, rates, rng):
    """The map, its table and its packed trees, as libnccl-mesh makes them; (map, trees, keep-alive) or a failure."""
    link = (C.c_uint32 * 2 * len(links))(*[(C.c_uint32 * 2)(a, b) for a, b in links])
    bandwidth = (C.c_uint64 * len(links))(*[int(r) for r in rates])
    mark = (C.c_uint32 * n)(*rng.sample(range(1, 1 << 30), n))
    table = (C.c_uint32 * (n * n))()
    m = LinkMap(GRAPH, n, len(links), C.cast(link, C.POINTER(C.c_uint32 * 2)), bandwidth, 0, None, mark)
    if lib.mesh_routes(C.byref(m), table):
        return 'mesh_routes failed'
    m.next = table
    apart = (C.c_uint32 * 2)()
    if lib.mesh_link_map_joined(C.byref(m), apart):
        return f'the map does not join nodes {apart[0]} and {apart[1]}'
    root, parent, weight = (C.c_uint32 * (n * n + 1))(), (C.c_uint32 * (n * n * n + 1))(), (C.c_double * (n * n + 1))()
    count = lib.mesh_trees_pack(C.byref(m), None, 0.0, n, root, parent, weight)
    if not count:
        return 'mesh_trees_pack packed no trees'
    trees = Trees(n, count, root, parent, weight)
    if lib.mesh_trees_check(C.byref(m), C.byref(trees)):
        return 'mesh_trees_check refused the packed trees'
    return m, trees, (link, bandwidth, mark, table, root, parent, weight)


def compile_program(lib, trees, n, what, root, whole):
    parts, moves = C.c_uint32(), C.c_uint32()
    lib.mesh_compile_room(C.byref(trees), C.byref(parts), C.byref(moves))
    segment, weight = (C.c_uint32 * (parts.value + 1))(), (C.c_double * (parts.value + 1))()
    first, move = (C.c_uint32 * (n + 1))(), (Move * (moves.value + 1))()
    program = Program()
    if lib.mesh_compile(C.byref(trees), what, root, whole, None, C.byref(program), segment, weight, first, move):
        return None
    return program, (segment, weight, first, move)


def segments_of(elements, count):
    return [elements // count + (s < elements % count) for s in range(count)]


def judge(lib, program, n, what, root, elements, page):
    """The failures of one compiled program: its order, its progress and its result. With a page, the operand is
    `elements` pages of halves (2 bytes), every step's piece must start and end on a page, and the result is judged a
    page an element."""
    quantum = page // 2 if page else 1
    operand = Operand(0, 2 if page else 4, elements * quantum)
    if lib.mesh_program_check_members(C.byref(program), operand, None, page, None, None):
        return ['mesh_program_check_members refused it']
    steps, cut = [], []
    for r in range(n):
        room = (program.first[r + 1] - program.first[r]) * max(1, program.parts) + 1
        made = (Step * room)()
        count = lib.mesh_program_steps_cut(C.byref(program), r, operand, None, page, made)
        cut.append((made, count))
        for i in range(count):
            if made[i].first % quantum or made[i].piece.elements % quantum:
                return [f'rank {r} step {i} is not on a page: elements {made[i].first} + {made[i].piece.elements}']
        steps.append([made[i] for i in range(count)])
        for step in steps[-1]:
            step.first //= quantum
            step.piece.elements //= quantum
    failures, match = [], {}
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            lists = []
            for rank, peer, send in ((a, b, 1), (b, a, 0)):
                own, count = cut[rank]
                made, placed = (Place * (count + 1))(), C.c_uint32()
                lib.mesh_lay(own, count, peer, send, 0, made, C.byref(placed))
                lists.append([made[i] for i in range(placed.value)])
            sends, lands = lists
            if len(sends) > ENTRIES:
                failures.append(f'{a}->{b} lays {len(sends)} entries, past the {ENTRIES} a list holds')
            if len(sends) != len(lands):
                failures.append(f'{a}->{b}: {len(sends)} SEND entries against {len(lands)} landing entries')
                continue
            for s, l in zip(sends, lands):
                sent, landing = steps[a][s.step], steps[b][l.step]
                if sent.piece.elements != landing.piece.elements or s.entry != l.entry:
                    failures.append(f'{a}->{b} entry {s.entry}: a SEND of {sent.piece.elements} elements meets a '
                                    f'landing of {landing.piece.elements} (entry {l.entry})')
                match[(b, l.step)] = (a, s.step)
    if failures:
        return failures[:4]
    # each rank's buffers: a bit a contribution, per element; buffer b >= 1 is scratch, a copy of the operand at start
    # the program's segments: the most segment a part names, plus one (mesh-plan.h mesh_program_steps_cut)
    segment = segments_of(elements, 1 + max((program.segment[p] for p in range(program.parts)), default=0))
    start = [sum(segment[:s]) for s in range(len(segment))]

    def initial(r):
        if what == BROADCAST:
            return [1 << root if r == root else 0] * elements
        if what == ALLGATHER:
            values = [0] * elements
            if r < len(segment):
                values[start[r]:start[r] + segment[r]] = [1 << r] * segment[r]
            return values
        return [1 << r] * elements
    buffers = [[initial(r) for _ in range(1 + program.scratch)] for r in range(n)]
    sent_values, done = {}, [[False] * len(steps[r]) for r in range(n)]

    def span(step):
        return step.buffer, step.first, step.first + step.piece.elements

    waits = []
    for r in range(n):
        rank = []
        for j, step in enumerate(steps[r]):
            buffer, lo, hi = span(step)
            rank.append([i for i in range(j) if steps[r][i].buffer == buffer and steps[r][i].first < hi and
                         lo < steps[r][i].first + steps[r][i].piece.elements and
                         not (step.op == SEND and steps[r][i].op == SEND)])
        waits.append(rank)
    progressed = True
    while progressed:
        progressed = False
        for r in range(n):
            for j, step in enumerate(steps[r]):
                if done[r][j] or not all(done[r][i] for i in waits[r][j]):
                    continue
                buffer, lo, hi = span(step)
                if step.op == SEND:
                    sent_values[(r, j)] = buffers[r][buffer][lo:hi]
                else:
                    source = match.get((r, j))
                    if source is None:
                        failures.append(f'rank {r} step {j} receives from {step.peer} and no SEND is laid toward it')
                        return failures
                    if source not in sent_values:
                        continue
                    piece = sent_values[source]
                    target = buffers[r][buffer]
                    for i, value in enumerate(piece):
                        if step.op == COPY:
                            target[lo + i] = value
                        else:
                            if target[lo + i] & value:
                                failures.append(f'rank {r} step {j} counts a contribution twice at element {lo + i}')
                                return failures
                            target[lo + i] |= value
                done[r][j] = progressed = True
    stuck = [(r, j, steps[r][j]) for r in range(n) for j in range(len(steps[r])) if not done[r][j]]
    if stuck:
        r, j, step = stuck[0]
        return [f'{len(stuck)} steps never run: rank {r} step {j} (op {step.op}, peer {step.peer}, round {step.round}) '
                'waits first']
    every = (1 << n) - 1
    for r in range(n):
        result = buffers[r][0]
        if what == ALLREDUCE or (what in (REDUCE,) and r == root):
            expected = [every] * elements
        elif what == BROADCAST:
            expected = [1 << root] * elements
        elif what == REDUCE_SCATTER:
            if r >= len(segment):
                continue
            result, expected = result[start[r]:start[r] + segment[r]], [every] * segment[r]
        elif what == ALLGATHER:
            expected = [1 << s for s in range(len(segment)) for _ in range(segment[s])]
        else:
            continue
        if result != expected:
            at = next(i for i, (x, y) in enumerate(zip(result, expected)) if x != y)
            failures.append(f'rank {r} element {at} holds {bin(result[at]).count("1")} contributions where '
                            f'{bin(expected[at]).count("1")} belong')
            break
    return failures


def fabric_of(n, links, rates, rng):
    """The fabric as fabric_discover makes it of real nodes' records: a cable each link, its wire a distinct identity,
    its nodes the lower first, the cables in cable_order (nodes, then wire)."""
    cables = sorted(((min(a, b), max(a, b)), rng.getrandbits(63) | 1, int(rate)) for (a, b), rate in zip(links, rates))
    cables.sort(key=lambda c: (c[0][0], c[0][1], c[1]))
    array = (Cable * len(cables))(*[Cable(wire, rate, (C.c_uint32 * 2)(*ends)) for ends, wire, rate in cables])
    return Fabric(n, len(cables), array), array


def write_calls(n, rng):
    """One group of WRITEs shaped as a real program submits them: the chain's input window (chain.operands: root 0 to
    every other rank, rows x width halves, input_frames repeats) or the LSTM's weight scatter (partition
    distribute_tensor: root 0 to each rank, its shard of a 4 x H x (K + H) weight)."""
    if rng.random() < 0.5:
        rows, width, frames = 8, rng.choice((128, 1024, 16384)), 512
        return f'chain input window, width {width}', [(0, d, rows * width) for d in range(1, n)], frames
    h = rng.choice((128, 1024, 16384))
    shard = 4 * h * 2 * h // n
    return f'lstm weight scatter, H {h}', [(0, d, shard) for d in range(1, n)], 1


def judge_writes(lib, fabric, routes, n, calls, repeats, page):
    """The failures of one group of WRITEs, every rank compiling every call as write_steps does."""
    failures, entries = [], {}
    for root, destination, count in calls:
        operand = Operand(7, 2, count)
        made = []
        for r in range(n):
            pair = routes[root * n + destination]
            needed = lib.mesh_write_compile(C.byref(fabric), r, root, destination, C.byref(pair), operand, page, 0, None, 0)
            if needed < 0:
                return [f'rank {r}: mesh_write_compile {root}->{destination} returned {needed}']
            transfers = (Transfer * (needed + 1))()
            got = lib.mesh_write_compile(C.byref(fabric), r, root, destination, C.byref(pair), operand, page, 0,
                                         transfers, needed + 1)
            if got != needed:
                return [f'rank {r}: mesh_write_compile {root}->{destination} counted {needed}, wrote {got}']
            made.append([transfers[i] for i in range(got)])
        # write_steps: a relay's SEND has an earlier receive of the same interval
        for r in range(n):
            if r == root:
                continue
            for k, t in enumerate(made[r]):
                if t.step.op == SEND and not any(u.step.op != SEND and u.step.first == t.step.first and
                                                 u.step.piece.elements == t.step.piece.elements for u in made[r][:k]):
                    return [f'{root}->{destination}: rank {r} relays elements {t.step.first} + {t.step.piece.elements} '
                            'with no matching incoming interval (write_steps)']
        # each directed cable's SENDs and receives, in each end's order
        lanes = {}
        for r in range(n):
            for k, t in enumerate(made[r]):
                key = (t.cable, r, t.step.peer) if t.step.op == SEND else (t.cable, t.step.peer, r)
                lanes.setdefault(key, ([], []))[0 if t.step.op == SEND else 1].append((r, k))
        for (cable, a, b), (sends, lands) in lanes.items():
            ends = fabric.cable[cable].node
            if sorted((a, b)) != sorted((ends[0], ends[1])):
                return [f'{root}->{destination}: a transfer {a}->{b} names cable {cable}, which joins {ends[0]} and {ends[1]}']
            if len(sends) != len(lands):
                return [f'{root}->{destination}: cable {cable} {a}->{b} carries {len(sends)} SENDs against {len(lands)} receives']
            for (r, k), (q, j) in zip(sends, lands):
                if made[r][k].step.piece.elements != made[q][j].step.piece.elements:
                    return [f'{root}->{destination}: cable {cable} {a}->{b} pairs a SEND of {made[r][k].step.piece.elements} '
                            f'elements with a receive of {made[q][j].step.piece.elements}']
            entries[(cable, a, b)] = entries.get((cable, a, b), 0) + len(sends) * repeats
        # progress and result: the root's elements are their indices; a relay's SEND waits on its receive
        held = {root: list(range(count))}
        done = {(r, k): False for r in range(n) for k in range(len(made[r]))}
        match = {}
        for (cable, a, b), (sends, lands) in lanes.items():
            for send, land in zip(sends, lands):
                match[land] = send
        sent, progressed = {}, True
        while progressed:
            progressed = False
            for (r, k), finished in done.items():
                if finished:
                    continue
                t = made[r][k].step
                lo, hi = t.first, t.first + t.piece.elements
                if t.op == SEND:
                    if r != root and not any(done[(r, i)] and made[r][i].step.op != SEND and made[r][i].step.first == lo
                                             for i in range(k)):
                        continue
                    sent[(r, k)] = held.get(r, [None] * count)[lo:hi]
                else:
                    if match.get((r, k)) not in sent:
                        continue
                    buffer = held.setdefault(r, [None] * count)
                    buffer[lo:hi] = sent[match[(r, k)]]
                done[(r, k)] = progressed = True
        if not all(done.values()):
            stuck = next(key for key, finished in done.items() if not finished)
            return [f'{root}->{destination}: {sum(not v for v in done.values())} transfers never run, rank {stuck[0]} '
                    f'transfer {stuck[1]} first']
        if held.get(destination) != list(range(count)):
            return [f'{root}->{destination}: the destination does not hold the root\'s {count} elements']
    for (cable, a, b), laid in entries.items():
        if laid > ENTRIES:
            failures.append(f'the group lays {laid} entries on cable {cable} {a}->{b}, past the {ENTRIES} a list holds')
    return failures[:3]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--library', type=Path, default=Path(__file__).resolve().parent / 'libmesh.dylib')
    parser.add_argument('--page', type=int, default=0, help='bytes; 16384 cuts as the bridge does (0: any element)')
    args = parser.parse_args(argv)
    lib, rng = library(args.library), random.Random(args.seed)
    programs = failed = 0
    for sample in range(args.samples):
        k, n, links, rates = topology(rng)
        if n < 2:
            continue
        solved = solve(lib, n, links, rates, rng)
        if isinstance(solved, str):
            failed += 1
            print(f'sample {sample}: k {k}, largest part {n} nodes, {len(links)} links: {solved}', flush=True)
            continue
        _, trees, _ = solved
        fabric, keep = fabric_of(n, links, rates, rng)
        routes = C.POINTER(Paths)()
        error = lib.mesh_write_routes_default(C.byref(fabric), C.byref(routes))
        if error:
            failed += 1
            print(f'sample {sample}: k {k}, largest part {n} nodes, {len(links)} links: mesh_write_routes_default {error}',
                  flush=True)
        else:
            shape, calls, repeats = write_calls(n, rng)
            programs += 1
            failures = judge_writes(lib, fabric, routes, n, calls, repeats, args.page or 16384)
            lib.mesh_write_routes_free(routes, n)
            if failures:
                failed += 1
                print(f'sample {sample}: k {k}, largest part {n} nodes, {len(links)} links: {shape}: {"; ".join(failures)}',
                      flush=True)
        roots = sorted({0, n - 1, rng.randrange(n)})
        forms = [(ALLREDUCE, 0, 0), (ALLREDUCE, 0, 1), (REDUCE_SCATTER, 0, 0), (ALLGATHER, 0, 0)] + \
                [(what, r, 0) for r in roots for what in (BROADCAST, REDUCE)]
        for what, root, whole in forms:
            programs += 1
            compiled = compile_program(lib, trees, n, what, root, whole)
            elements = 7 * n + rng.randint(1, 97)
            failures = ['mesh_compile refused it'] if compiled is None else judge(lib, compiled[0], n, what, root, elements,
                                                                                  args.page)
            if failures:
                failed += 1
                print(f'sample {sample}: k {k}, largest part {n} nodes, {len(links)} links, {trees.count} trees: '
                      f'{NAMES[what]}{" whole" if whole else ""} root {root}, {elements} elements: {"; ".join(failures)}',
                      flush=True)
    print(f'{programs} programs over {args.samples} topologies (seed {args.seed}, page {args.page}): {failed} failed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
