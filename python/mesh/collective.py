"""The partial tensors' collectives (design/algorithm-sources.md#collective-movement, #collectivereduce_scatter): the
verbs over a program's blocks, each block row one libnccl-mesh call over its members' communicator (rdma/NCCL.md), so
a row crosses once its own blocks are ready (a program runs in row order).  A crossing's kind follows from how its
consumer reads the blocks: read whole is MPI_Allgatherv (all_gather, broadcast: ncclMeshAllGatherV); moved to an owner
is MPI_Alltoallv (all_to_all, scatter, gather); contracted is MPI_Reduce_scatter (reduce_scatter, reduce:
ncclMeshReduceScatterV) or the all-reduce, by SUM, MAX or MIN; a softmax over a split is the log-sum-exp merge of its
partials, PARTIALS (the reduction operator ncclMeshRedOpCreatePartials over rows of [m, l, O]).  Any other associative combine is a supplied
kernel declaring `combine` (Swift's `using combine:`): the terms moved to their owner by one all-to-all and combined
there by a tree of its calls.  Every member records every verb (the program is SPMD); `peers` and `owners` are ranks
of the program, `owners` each block's index to its owner."""
from functools import partial

import torch

from . import BlockSpec, Program, ShapeDtypeStruct, _native

__all__ = ['SUM', 'MAX', 'MIN', 'PARTIALS', 'send', 'recv', 'recv_like', 'broadcast', 'scatter', 'gather',
           'all_gather', 'all_to_all', 'reduce', 'reduce_scatter', 'sum_scatter',
           'all_reduce', 'all_sum', 'all_max', 'all_min', 'sync_on_remote_fill']

SUM, MAX, MIN, PARTIALS = 'sum', 'max', 'min', 'partials'
send = recv = Program.copy
recv_like = Program.replicate


# design/algorithm-sources.md#collectivereduce_scatter
def _identity(op, dtype):
    if op == SUM:
        return 0
    if dtype.is_floating_point:
        return float('-inf') if op == MAX else float('inf')
    info = torch.iinfo(dtype)
    return info.min if op == MAX else info.max


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
        program._record(kind, members, min(ref.origin[0] for ref in refs), roles, run, queue=queue, crossing=True,
                        members=members)


# design/algorithm-sources.md#collective-movement
def _numel(refs):
    return sum(ref.rows * ref.columns for ref in refs)


# design/algorithm-sources.md#collective-movement
def _buffer(program, dtype, numel):
    return _native.allocate(numel, dtype, program.device, program.size > 1)


# design/algorithm-sources.md#collective-movement
def _pieces(buffer, refs):
    sizes = [ref.rows * ref.columns for ref in refs]
    return zip(refs, buffer[:sum(sizes)].split(sizes))


# design/algorithm-sources.md#collective-movement
def _pack(buffer, refs):
    for ref, piece in _pieces(buffer, refs):
        piece.view(ref.shape).copy_(ref.tensor)
    return buffer


# design/algorithm-sources.md#collective-movement
def _unpack(buffer, refs):
    for ref, piece in _pieces(buffer, refs):
        ref.tensor.copy_(piece.view(ref.shape))


# design/algorithm-sources.md#collective-movement
def _segments(members, row, value, owners):
    return tuple(tuple(value[i] for i in row if owners[i] == member) for member in members)


# design/algorithm-sources.md#collective-movement
def all_gather(program, value, *, peers, owners, queue=0):
    members = tuple(sorted(set(peers) | set(owners.values())))
    for row in _rows(owners):
        segments = _segments(members, row, value, owners)
        roles = {m: (mine, tuple(ref for other in segments if other is not mine for ref in other), ())
                 for m, mine in zip(members, segments)}
        _record(program, 'all_gather', members, roles, (_issue_all_gather, segments), queue)
    return value


# design/algorithm-sources.md#collective-movement
def _issue_all_gather(program, task):
    _, segments = task.run
    me = task.members.index(program.rank)
    whole = _buffer(program, next(ref.dtype for refs in segments for ref in refs), sum(map(_numel, segments)))
    blocks = list(whole.split([_numel(refs) for refs in segments]))
    work = _native.all_gather(blocks, _pack(blocks[me], segments[me]), program._world.of(task.queue, task.members))
    for at, (block, refs) in enumerate(zip(blocks, segments)):
        if at != me:
            _unpack(block, refs)
    return (work,)


# design/algorithm-sources.md#collective-movement
def broadcast(program, value, *, root, peers, queue=0):
    return all_gather(program, value, peers=peers, owners=dict.fromkeys(value.blocks, root), queue=queue)


# design/algorithm-sources.md#collective-movement
def _exchange(program, rows, queue):
    """Each row's moves (source, sender, destination, receiver) as one MPI_Alltoallv among their members: what a
    member sends each other one, in the moves' order, received in the same order."""
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
    _, moves = task.run
    me = program.rank
    sends = [[s for s, a, _, b in moves if a == me and b == m] for m in task.members]
    receives = [[d for _, a, d, b in moves if b == me and a == m] for m in task.members]
    dtype = moves[0][0].dtype
    sent = _pack(_buffer(program, dtype, sum(map(_numel, sends))), [ref for refs in sends for ref in refs])
    received = _buffer(program, dtype, sum(map(_numel, receives)))
    work = _native.all_to_all(received, sent, [_numel(refs) for refs in receives], [_numel(refs) for refs in sends],
                              program._world.of(task.queue, task.members))
    _unpack(received, [ref for refs in receives for ref in refs])
    return (work,)


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
        segments = _segments(members, row, value, owners)
        roles = {}
        for m, mine in zip(members, segments):
            others = tuple(ref for other in segments if other is not mine for ref in other)
            roles[m] = (others, (), mine) if m in peers else ((), mine, ())
        run = (_issue_partials, tuple(value[i] for i in row), tuple(members.index(owners[i]) for i in row), peers) \
            if op == PARTIALS else (_issue_reduce_scatter, segments, op, peers)
        _record(program, 'reduce', members, roles, run, queue)
    return value


# design/algorithm-sources.md#collectivereduce_scatter
def _issue_reduce_scatter(program, task):
    _, segments, op, peers = task.run
    me = task.members.index(program.rank)
    refs = [ref for row in segments for ref in row]
    sizes = [_numel(row) for row in segments]
    terms = _buffer(program, refs[0].dtype, sum(sizes))
    if program.rank in peers:
        _pack(terms, refs)
    else:
        terms.fill_(_identity(op, refs[0].dtype))
    out = _buffer(program, refs[0].dtype, sizes[me])
    work = _native.reduce_scatter(out, list(terms.split(sizes)), op, program._world.of(task.queue, task.members))
    _unpack(out, segments[me])
    return (work,)


# design/algorithm-sources.md#collectivereduce_scatter
def _issue_partials(program, task):
    _, refs, roots, peers = task.run
    me, works = task.members.index(program.rank), []
    for ref, root in zip(refs, roots):
        whole = _buffer(program, ref.dtype, ref.rows * ref.columns)
        if program.rank in peers:
            _pack(whole, (ref,))
        else:
            whole.zero_()
        works.append(_native.partials(whole.view(ref.shape), root, program._world.of(task.queue, task.members)))
        if root in (None, me):
            _unpack(whole, (ref,))
    return tuple(works)


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
    _, refs, op = task.run
    whole = _pack(_buffer(program, refs[0].dtype, _numel(refs)), refs)
    work = _native.all_reduce(whole, op, program._world.of(task.queue, task.members))
    _unpack(whole, refs)
    return (work,)


sum_scatter = reduce_scatter
all_sum = all_reduce
all_max = partial(all_reduce, op=MAX)
all_min = partial(all_reduce, op=MIN)


# design/algorithm-sources.md#collectivesync_on_remote_fill
def sync_on_remote_fill(*results, timeout=None):
    """Waits until every result is filled (or its program failed: MeshError), within the program's timeout or
    `timeout` seconds (TimeoutError past it).  Explicit, diegetic and never a default: no verb calls it."""
    programs = {}
    for result in results:
        programs.setdefault(id(result.ref.program), (result.ref.program, []))[1].append(result)
    for program, waited in programs.values():
        program._wait_results(waited, timeout)
