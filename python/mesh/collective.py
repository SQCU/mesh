from functools import partial
from itertools import product

from . import BlockSpec, Program, ShapeDtypeStruct, kernels, check

__all__ = ['send', 'recv', 'recv_like', 'broadcast', 'scatter', 'gather',
           'all_gather', 'all_to_all', 'reduce', 'reduce_scatter', 'sum_scatter',
           'all_reduce', 'all_sum', 'all_max', 'all_min', 'sync_on_remote_fill']

send = recv = Program.copy
recv_like = Program.replicate


# design/algorithm-sources.md#collective-movement
def _move(program, value, routes):
    blocks = {}
    for index, sender, receivers in routes:
        shape = tuple(min(block, size - coordinate * block)
                      for coordinate, block, size in zip(index, value.block_shape, value.shape))
        result = value[index] if program.node == sender else program.tensor(shape, dtype=value.dtype)[0, 0]
        for receiver in receivers:
            if receiver != sender:
                program.copy(result.on(sender), result.on(receiver))
        if program.node in receivers:
            blocks[index] = result
    return value._with_blocks(blocks)


# design/algorithm-sources.md#collective-movement
def broadcast(program, value, *, root, peers):
    peers = tuple(peers)
    return _move(program, value, ((index, root, peers) for index in product(*(range(n) for n in value.grid))))


# design/algorithm-sources.md#collective-movement
def scatter(program, value, *, root, owners):
    return _move(program, value, ((index, root, (owner,)) for index, owner in owners.items()))


# design/algorithm-sources.md#collective-movement
def gather(program, value, *, root, owners):
    return _move(program, value, ((index, owner, (root,)) for index, owner in owners.items()))


# design/algorithm-sources.md#collective-movement
def all_gather(program, value, *, peers, owners):
    peers = tuple(peers)
    return _move(program, value, ((index, owner, peers) for index, owner in owners.items()))


# design/algorithm-sources.md#collective-movement
def all_to_all(program, value, *, owners, destinations):
    return _move(program, value, ((index, owner, (destinations[index],)) for index, owner in owners.items()))


# design/algorithm-sources.md#collectivereduce_scatter
def reduce_scatter(program, value, *, peers, owners, op=kernels.add):
    peers = tuple(peers)
    blocks = {}
    for index, owner in owners.items():
        local = value[index]
        terms = [local] if owner in peers else []
        for peer in peers:
            if peer == owner:
                continue
            received = program.tensor(local.shape, dtype=local.dtype)[0, 0] if program.node == owner else local
            program.copy(local.on(peer), received.on(owner))
            terms.append(received)
        if program.node == owner:
            while len(terms) > 1:
                terms = [program.kernel_call(op,
                    grid=(1, 1), in_specs=(BlockSpec(None), BlockSpec(None)),
                    out_specs=BlockSpec(local.shape, lambda i, j: (i, j)),
                    out_shape=ShapeDtypeStruct(local.shape, local.dtype))(terms[i], terms[i + 1])[0, 0]
                    if i + 1 < len(terms) else terms[i] for i in range(0, len(terms), 2)]
            blocks[index] = terms[0]
    return value._with_blocks(blocks)


# design/algorithm-sources.md#collectivereduce_scatter
def reduce(program, value, *, root, peers, op=kernels.add):
    owners = dict.fromkeys(product(*(range(n) for n in value.grid)), root)
    return reduce_scatter(program, value, peers=peers, owners=owners, op=op)


# design/algorithm-sources.md#collectivereduce_scatter
def all_reduce(program, value, *, peers, owners, op=kernels.add):
    peers = tuple(peers)
    return all_gather(program, reduce_scatter(program, value, peers=peers, owners=owners, op=op),
                      peers=peers, owners=owners)


# design/algorithm-sources.md#collectivereduce_scatter
sum_scatter = reduce_scatter
all_sum = all_reduce
all_max = partial(all_reduce, op=kernels.maximum)
all_min = partial(all_reduce, op=kernels.minimum)


# design/algorithm-sources.md#collectivesync_on_remote_fill
def sync_on_remote_fill(*results):
    while not all(result.ready for result in results):
        for result in results:
            check(result.ref.program.report.code)
