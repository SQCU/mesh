"""The partial tensors' collectives (design/algorithm-sources.md#collective-movement, #collectivereduce_scatter):
every verb of the old library over a program's blocks, on libnccl-mesh.  Movement is a block's copy along its route
(send/recv: point to point) or a broadcast rooted at its owner; a reduction by `add`, `maximum` or `minimum` is the
library's own reduce or all-reduce of the block in place (ncclReduce, ncclAllReduce: libnccl-mesh's compiled trees),
any other associative combine the old form, the terms moved to their owner and summed there by the combine's kernel.
Every member records every verb (the program is SPMD); `peers` and `owners` are ranks of the program."""
from functools import partial
from itertools import product

import torch
import torch.distributed as dist

from . import BlockSpec, Program, ShapeDtypeStruct, kernels, _native

__all__ = ['send', 'recv', 'recv_like', 'broadcast', 'scatter', 'gather',
           'all_gather', 'all_to_all', 'reduce', 'reduce_scatter', 'sum_scatter',
           'all_reduce', 'all_sum', 'all_max', 'all_min', 'sync_on_remote_fill']

send = recv = Program.copy
recv_like = Program.replicate
_OPS = {kernels.add: dist.ReduceOp.SUM, kernels.maximum: dist.ReduceOp.MAX, kernels.minimum: dist.ReduceOp.MIN}


# design/algorithm-sources.md#collectivereduce_scatter
def _identity(op, scalar):
    if op is kernels.add:
        return 0
    if scalar.is_floating_point:
        return float('-inf') if op is kernels.maximum else float('inf')
    info = torch.iinfo(scalar)
    return info.min if op is kernels.maximum else info.max


# design/algorithm-sources.md#collective-movement
def _record(program, kind, ref, members, roles, run, queue):
    members = tuple(sorted(set(members)))
    if len(members) > 1:
        program._record(kind, members, ref.origin[0], roles, run, queue=queue, crossing=True, members=members)


# design/algorithm-sources.md#collective-movement
def _issue_broadcast(program, task):
    _, ref, root = task.run
    work = _native.broadcast(ref.tensor, task.members.index(root), program._world.of(task.queue, task.members))
    if program.rank == root:
        work.wait()
        return ()
    return (work,)


# design/algorithm-sources.md#collectivereduce_scatter
def _issue_reduce(program, task):
    _, ref, root, op, identity = task.run
    tensor = ref.tensor
    if identity is not None and program.rank == root:
        tensor.fill_(identity)
    work = _native.reduce(tensor, task.members.index(root), op, program._world.of(task.queue, task.members))
    if program.rank != root:
        work.wait()
        return ()
    return (work,)


# design/algorithm-sources.md#collectivereduce_scatter
def _issue_all_reduce(program, task):
    _, ref, op = task.run
    return (_native.all_reduce(ref.tensor, op, program._world.of(task.queue, task.members)),)


# design/algorithm-sources.md#collective-movement
def _broadcast_block(program, ref, root, peers, queue):
    members = set(peers) | {root}
    roles = {rank: (((ref,), (), ()) if rank == root else ((), (ref,), ())) for rank in members}
    _record(program, 'broadcast', ref, members, roles, (_issue_broadcast, ref, root), queue)


# design/algorithm-sources.md#collective-movement
def _move(program, value, routes, queue):
    for index, sender, receivers in routes:
        for receiver in receivers:
            if receiver != sender:
                program.copy(value[index].on(sender), value[index].on(receiver), queue=queue)
    return value


# design/algorithm-sources.md#collective-movement
def broadcast(program, value, *, root, peers, queue=0):
    for index in product(*(range(n) for n in value.grid)):
        _broadcast_block(program, value[index], root, peers, queue)
    return value


# design/algorithm-sources.md#collective-movement
def scatter(program, value, *, root, owners, queue=0):
    return _move(program, value, ((index, root, (owner,)) for index, owner in owners.items()), queue)


# design/algorithm-sources.md#collective-movement
def gather(program, value, *, root, owners, queue=0):
    return _move(program, value, ((index, owner, (root,)) for index, owner in owners.items()), queue)


# design/algorithm-sources.md#collective-movement
def all_gather(program, value, *, peers, owners, queue=0):
    for index, owner in owners.items():
        _broadcast_block(program, value[index], owner, peers, queue)
    return value


# design/algorithm-sources.md#collective-movement
def all_to_all(program, value, *, owners, destinations, queue=0):
    return _move(program, value, ((index, owner, (destinations[index],)) for index, owner in owners.items()), queue)


# design/algorithm-sources.md#collectivereduce_scatter
def reduce_scatter(program, value, *, peers, owners, op=kernels.add, queue=0):
    peers = tuple(peers)
    if op in _OPS:
        for index, owner in owners.items():
            ref = value[index]
            roles = {rank: ((ref,), (), ()) for rank in peers if rank != owner}
            roles[owner] = ((), (), (ref,)) if owner in peers else ((), (ref,), ())
            _record(program, 'reduce', ref, set(peers) | {owner}, roles,
                    (_issue_reduce, ref, owner, _OPS[op], None if owner in peers else _identity(op, ref.dtype)), queue)
        return value
    blocks = {}
    for index, owner in owners.items():
        local = value[index]
        terms = [local] if owner in peers else []
        for peer in peers:
            if peer == owner:
                continue
            received = program.tensor(local.shape, dtype=local.dtype)[0, 0]
            program.copy(local.on(peer), received.on(owner), queue=queue)
            terms.append(received)
        while len(terms) > 1:
            terms = [program.kernel_call(op, grid=(1, 1), in_specs=(BlockSpec(None), BlockSpec(None)),
                                         out_specs=BlockSpec(local.shape, lambda i, j: (i, j)),
                                         out_shape=ShapeDtypeStruct(local.shape, local.dtype),
                                         peer=owner)(terms[i], terms[i + 1])[0, 0]
                     if i + 1 < len(terms) else terms[i] for i in range(0, len(terms), 2)]
        blocks[index] = terms[0]
    return value._with_blocks(blocks)


# design/algorithm-sources.md#collectivereduce_scatter
def reduce(program, value, *, root, peers, op=kernels.add, queue=0):
    owners = dict.fromkeys(product(*(range(n) for n in value.grid)), root)
    return reduce_scatter(program, value, peers=peers, owners=owners, op=op, queue=queue)


# design/algorithm-sources.md#collectivereduce_scatter
def all_reduce(program, value, *, peers, owners=None, op=kernels.add, queue=0):
    peers = tuple(peers)
    if op in _OPS:
        for index in product(*(range(n) for n in value.grid)):
            ref = value[index]
            _record(program, 'all_reduce', ref, peers, {rank: ((), (), (ref,)) for rank in peers},
                    (_issue_all_reduce, ref, _OPS[op]), queue)
        return value
    if owners is None:
        raise ValueError('A combine other than add, maximum or minimum is reduced to its owners, then gathered: owners')
    return all_gather(program, reduce_scatter(program, value, peers=peers, owners=owners, op=op, queue=queue),
                      peers=peers, owners=owners, queue=queue)


sum_scatter = reduce_scatter
all_sum = all_reduce
all_max = partial(all_reduce, op=kernels.maximum)
all_min = partial(all_reduce, op=kernels.minimum)


# design/algorithm-sources.md#collectivesync_on_remote_fill
def sync_on_remote_fill(*results, timeout=None):
    """Waits until every result is filled (or its program failed: MeshError), within the program's timeout or
    `timeout` seconds (TimeoutError past it).  Explicit, diegetic and never a default: no verb calls it."""
    programs = {}
    for result in results:
        programs.setdefault(id(result.ref.program), (result.ref.program, []))[1].append(result)
    for program, waited in programs.values():
        program._wait_results(waited, timeout)
