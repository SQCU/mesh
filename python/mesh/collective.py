"""The partial tensors' collectives (design/algorithm-sources.md#collective-movement, #collectivereduce_scatter): the
verbs over a program's blocks, each block row one declared group on the full physical communicator. Movement
binds original and final strided views to exact indexed native WRITEs. Reductions bind original contiguous
fragments to native reductions, by SUM, MAX or MIN; a softmax over a split uses the native log-sum-exp merge of
partials, PARTIALS (ncclMeshRedOpCreatePartials over complete rows of [m, l, O]). Any other associative combine is a supplied
kernel declaring `combine` (Swift's `using combine:`): the terms moved to their owner by one all-to-all and combined
there by a tree of its calls.  Every member records every verb (the program is SPMD); `peers` and `owners` are ranks
of the program, `owners` each block's index to its owner."""
from functools import partial

import torch

from . import BlockSpec, Program, ShapeDtypeStruct, _native

__all__ = ['SUM', 'MAX', 'MIN', 'PARTIALS', 'send', 'recv', 'recv_like', 'broadcast', 'scatter', 'gather',
           'all_gather', 'all_to_all', 'reduce', 'reduce_scatter', 'sum_scatter',
           'all_reduce', 'all_sum', 'all_max', 'all_min', 'sync_on_remote_fill', 'sync_on_some_remote_fill']

SUM, MAX, MIN, PARTIALS = 'sum', 'max', 'min', 'partials'
send = recv = Program.copy
recv_like = Program.replicate


# design/algorithm-sources.md#collective-movement
def _rows(indices):
    rows = {}
    for index in sorted(indices):
        rows.setdefault(index[0], []).append(index)
    return list(rows.values())


# design/algorithm-sources.md#collective-movement
def _record(program, kind, members, roles, run, queue):
    members = tuple(sorted(set(members)))
    refs = [ref for role in roles.values() for refs in role for ref in refs]
    if len(members) > 1 and refs:
        physical = tuple(range(program.size))
        program._record(kind, physical, min(ref.origin[0] for ref in refs), roles, run, queue=queue, crossing=True,
                        members=physical, participants=members)


# design/algorithm-sources.md#collective-movement
def _buffer(program, dtype, numel):
    return _native.allocate(numel, dtype, program.device, program.size > 1)[:numel]


def _alias_fragments(program, task, refs):
    """AOT union cuts, each interval aliasing contiguous original operand bytes."""
    runs, cursor = [], 0
    for ref in refs:
        # Ref layout is metadata; excluded ranks do not allocate these backings.
        if ref.column_stride == 1 and ref.row_stride == ref.columns:
            runs.append((cursor, cursor + ref.rows * ref.columns, ref, ref.offset))
            cursor += ref.rows * ref.columns
        else:
            for row in range(ref.rows):
                width = ref.columns if ref.column_stride == 1 else 1
                for column in range(0, ref.columns, width):
                    start = ref.offset + row * ref.row_stride + column * ref.column_stride
                    runs.append((cursor, cursor + width, ref, start))
                    cursor += width
    cuts = program._world.union_cuts(task.index, [0, cursor, *(end for _, end, _, _ in runs)])
    pieces, index = [], 0
    for first, last in zip(cuts, cuts[1:]):
        while runs[index][1] <= first:
            index += 1
        lo, hi, ref, offset = runs[index]
        if last > hi:
            raise ValueError('common fragment crosses a local contiguous alias')
        pieces.append((ref, offset + first - lo, last - first))
    return pieces


def _alias(ref, offset, count):
    return ref.backing.flat().narrow(0, offset, count)


def _issue_fragments(program, task):
    bindings, group = task.prepared
    return (_native.fragments(bindings, group),)


def _prepare_all_gather(program, task):
    _, segments = task.run
    moves = [(ref, source, ref, destination) for source, segment in enumerate(segments) for ref in segment
             for destination in task.participants if destination != source]
    return _prepare_moves(program, task, moves)


def _prepare_moves(program, task, moves):
    """Exact original/final views; distinct compact forwarding leases per write."""
    deliveries, calls = [], []
    for source, a, destination, b in moves:
        count = source.rows * source.columns * source.dtype.itemsize
        if count != destination.rows * destination.columns * destination.dtype.itemsize:
            raise ValueError('Program movement requires equal logical byte extents')
        view = source.tensor if program.rank == a else destination.tensor if program.rank == b else _buffer(program, torch.uint8, count)
        deliveries.append((view, a, b))
        calls.append(_native.call('WRITE', count, torch.uint8, root=a, destination=b))
    group = program._world.of(task.queue, task.members)
    prepared = _native.prepare_writes(deliveries, group)
    return prepared, calls


# design/algorithm-sources.md#collective-movement
def _segments(members, row, value, owners):
    return tuple(tuple(value[i] for i in row if owners[i] == member) for member in members)


# design/algorithm-sources.md#collective-movement
def all_gather(program, value, *, peers, owners, queue=0):
    members = tuple(sorted(set(peers) | set(owners.values())))
    for row in _rows(owners):
        segments = _segments(range(program.size), row, value, owners)
        roles = {m: (mine, tuple(ref for other in segments if other is not mine for ref in other), ())
                 for m, mine in enumerate(segments) if m in members}
        _record(program, 'all_gather', members, roles, (_issue_all_gather, segments), queue)
    return value


# design/algorithm-sources.md#collective-movement
def _issue_all_gather(program, task):
    return (task.prepared[0].execute(async_op=True),)


_issue_all_gather.prepare = _prepare_all_gather


# design/algorithm-sources.md#collective-movement
def broadcast(program, value, *, root, peers, queue=0):
    return all_gather(program, value, peers=peers, owners=dict.fromkeys(value.blocks, root), queue=queue)


# design/algorithm-sources.md#collective-movement
def _exchange(program, rows, queue):
    """One ordered group of exact indexed WRITEs per row, with every rank available to forward."""
    for moves in rows:
        moves = tuple(move for move in moves if move[1] != move[3])
        for source, _, destination, _ in moves:
            destination.partial = source.partial
        members = tuple(sorted({move[1] for move in moves} | {move[3] for move in moves}))
        roles = {m: (tuple(s for s, a, _, _ in moves if a == m), tuple(d for _, _, d, b in moves if b == m), ())
                 for m in members}
        _record(program, 'all_to_all', members, roles, (_issue_exchange, moves), queue)


# design/algorithm-sources.md#collective-movement
def _issue_exchange(program, task):
    return (task.prepared[0].execute(async_op=True),)


def _prepare_exchange(program, task):
    _, moves = task.run
    return _prepare_moves(program, task, moves)


_issue_exchange.prepare = _prepare_exchange


# design/algorithm-sources.md#collective-movement
def all_to_all(program, value, *, owners, destinations, queue=0):
    _exchange(program, ([(value[i], owners[i], value[i], destinations[i]) for i in row] for row in _rows(owners)), queue)
    return value


# design/algorithm-sources.md#collective-movement
def scatter(program, value, *, root, owners, queue=0):
    return all_to_all(program, value, owners=dict.fromkeys(owners, root), destinations=owners, queue=queue)


# design/algorithm-sources.md#collective-movement
def gather(program, value, *, root, owners, queue=0):
    return all_to_all(program, value, owners=owners, destinations=dict.fromkeys(owners, root), queue=queue)


# design/algorithm-sources.md#collectivereduce_scatter
def reduce_scatter(program, value, *, peers, owners, op=SUM, queue=0):
    peers = frozenset(peers)
    if callable(op):
        return _combine(program, value, peers, owners, op, queue)
    members = tuple(sorted(peers | set(owners.values())))
    for row in _rows(owners):
        segments = _segments(range(program.size), row, value, owners)
        roles = {}
        for m, mine in enumerate(segments):
            if m not in members:
                continue
            others = tuple(ref for other in segments if other is not mine for ref in other)
            roles[m] = (others, (), mine) if m in peers else ((), mine, ())
        run = (_issue_partials, tuple(value[i] for i in row), tuple(owners[i] for i in row), peers) \
            if op == PARTIALS else (_issue_reduce_scatter, segments, op, peers)
        _record(program, 'reduce', members, roles, run, queue)
    return value


# design/algorithm-sources.md#collectivereduce_scatter
def _issue_reduce_scatter(program, task):
    return _issue_fragments(program, task)


def _prepare_reduce_scatter(program, task):
    _, segments, op, peers = task.run
    refs = tuple(ref for segment in segments for ref in segment)
    owners = {id(ref): rank for rank, segment in enumerate(segments) for ref in segment}
    empty = _buffer(program, refs[0].dtype, 0)
    bindings = []
    for ref, offset, count in _alias_fragments(program, task, refs):
        owner = owners[id(ref)]
        alias = _alias(ref, offset, count) if program.rank in peers or program.rank == owner else empty
        call = _native.call('REDUCE', count, ref.dtype, op=op, root=owner,
                            contributors=tuple(peers), recipients=(owner,))
        bindings.append((call, alias if program.rank in peers else empty,
                         alias if program.rank == owner else empty))
    return bindings, program._world.of(task.queue, task.members)


_issue_reduce_scatter.prepare = _prepare_reduce_scatter


# design/algorithm-sources.md#collectivereduce_scatter
def _issue_partials(program, task):
    return _issue_fragments(program, task)


def _prepare_partials(program, task):
    """Native PARTIALS records have contiguous float32 fields [m,l,O].

    Records may be separated by arbitrary row gaps or backing allocations.
    Configuration fragments only between complete records; it never packs a
    transposed/strided field list or emulates the numerical merge in Python.
    """
    _, refs, roots, peers = task.run
    for ref in refs:
        if (ref.dtype != torch.float32 or ref.columns < 3 or ref.column_stride != 1
                or ref.rows > 1 and ref.row_stride < ref.columns):
            raise ValueError('PARTIALS requires nonoverlapping records of contiguous float32 [m,l,O] fields')
    owners = {id(ref): root for ref, root in zip(refs, roots)}
    group = program._world.of(task.queue, task.members)
    backend = group._get_backend(torch.device(program.device))
    empty, bindings = _buffer(program, refs[0].dtype, 0), []
    for ref, offset, count in _alias_fragments(program, task, refs):
        if (offset - ref.offset) % max(ref.row_stride, ref.columns) % ref.columns or count % ref.columns:
            raise ValueError('PARTIALS common fragment must contain complete typed records')
        root = owners[id(ref)]
        receives = program.rank in task.participants if root is None else program.rank == root
        alias = _alias(ref, offset, count) if program.rank in peers or receives else empty
        call = _native.call('ALLREDUCE' if root is None else 'REDUCE', count, ref.dtype,
                            op=backend.partials_operator(ref.columns), root=0 if root is None else root,
                            contributors=tuple(peers), recipients=task.participants if root is None else (root,))
        bindings.append((call, alias if program.rank in peers else empty, alias if receives else empty))
    return bindings, group


_issue_partials.prepare = _prepare_partials


# design/algorithm-sources.md#collectivereduce_scatter
def _combine(program, value, peers, owners, combine, queue):
    terms, moves = {}, {}
    for index, owner in owners.items():
        local = value[index]
        terms[index] = [local] if owner in peers else []
        for peer in sorted(peers - {owner}):
            received = program.tensor(local.shape, dtype=local.dtype)[0, 0]
            moves.setdefault(index[0], []).append((local, peer, received, owner))
            terms[index].append(received)
    _exchange(program, moves.values(), queue)
    blocks = {}
    for index, parts in terms.items():
        while len(parts) > 1:
            parts = [program.kernel_call(combine, grid=(1, 1), in_specs=(BlockSpec(None), BlockSpec(None)),
                                         out_specs=BlockSpec(parts[i].shape, lambda i, j: (i, j)),
                                         out_shape=ShapeDtypeStruct(parts[i].shape, parts[i].dtype),
                                         peer=owners[index])(parts[i], parts[i + 1])[0, 0]
                     if i + 1 < len(parts) else parts[i] for i in range(0, len(parts), 2)]
        blocks[index] = parts[0]
    return value._with_blocks(blocks)


# design/algorithm-sources.md#collectivereduce_scatter
def reduce(program, value, *, root, peers, op=SUM, queue=0):
    return reduce_scatter(program, value, peers=peers, owners=dict.fromkeys(value.blocks, root), op=op, queue=queue)


# design/algorithm-sources.md#collectivereduce_scatter
def all_reduce(program, value, *, peers, owners=None, op=SUM, queue=0):
    peers = tuple(sorted(set(peers)))
    if callable(op):
        if owners is None:
            raise ValueError('A supplied combine is reduced to its owners, then gathered: owners')
        return all_gather(program, reduce_scatter(program, value, peers=peers, owners=owners, op=op, queue=queue),
                          peers=peers, owners=owners, queue=queue)
    for row in _rows(value.blocks):
        refs = tuple(value[i] for i in row)
        run = (_issue_partials, refs, (None,) * len(refs), peers) if op == PARTIALS else (_issue_all_reduce, refs, op)
        _record(program, 'all_reduce', peers, {m: ((), (), refs) for m in peers}, run, queue)
    return value


# design/algorithm-sources.md#collectivereduce_scatter
def _issue_all_reduce(program, task):
    return _issue_fragments(program, task)


def _prepare_all_reduce(program, task):
    _, refs, op = task.run
    empty = _buffer(program, refs[0].dtype, 0)
    bindings = []
    for ref, offset, count in _alias_fragments(program, task, refs):
        alias = _alias(ref, offset, count) if program.rank in task.participants else empty
        call = _native.call('ALLREDUCE', count, ref.dtype, op=op,
                            contributors=task.participants, recipients=task.participants)
        bindings.append((call, alias, alias))
    return bindings, program._world.of(task.queue, task.members)


_issue_all_reduce.prepare = _prepare_all_reduce


def _declaration(task):
    """The same bound counts/operators execution passes to the native library."""
    issue = task.run[0]
    if issue in (_issue_all_gather, _issue_exchange):
        return [task.prepared[1]]
    if issue in (_issue_reduce_scatter, _issue_all_reduce, _issue_partials):
        bindings, _ = task.prepared
        return [[call for call, _, _ in bindings]]
    raise TypeError('unknown Program collective declaration')


for _issue in (_issue_all_gather, _issue_exchange, _issue_reduce_scatter, _issue_all_reduce, _issue_partials):
    _issue.declaration = _declaration


sum_scatter = reduce_scatter
all_sum = all_reduce
all_max = partial(all_reduce, op=MAX)
all_min = partial(all_reduce, op=MIN)


# design/algorithm-sources.md#collectivesync_on_remote_fill
def sync_on_remote_fill(*results):
    """Waits until every result is filled, or its program failed (MeshError); where a result is never filled it waits
    forever (design/collective-goals-2026-09-14.md: slower, and it can deadlock forever).  Explicit, diegetic and never
    a default: no verb calls it."""
    programs = {}
    for result in results:
        programs.setdefault(id(result.ref.program), (result.ref.program, []))[1].append(result)
    for program, waited in programs.values():
        program._wait_results(waited)


# design/algorithm-sources.md#collectivesync_on_remote_fill
def sync_on_some_remote_fill(*results):
    """Waits until at least one of `results`, all of one program, is filled, or the program failed (MeshError): the
    filled ones, in their order (MPI_Waitsome); where none is ever filled it waits forever.  Explicit as
    sync_on_remote_fill is."""
    programs = {result.ref.program for result in results}
    if len(programs) != 1:
        raise ValueError('sync_on_some_remote_fill waits on the results of one program')
    return programs.pop()._wait_results(results, every=False)
