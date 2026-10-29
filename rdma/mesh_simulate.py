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
