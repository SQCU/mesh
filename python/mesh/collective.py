from . import BlockSpec, Program, ShapeDtypeStruct, kernels

__all__ = ['send', 'reduce_scatter', 'all_gather', 'all_reduce']

send = Program.copy


# design/algorithm-sources.md#collective
def reduce_scatter(program, value, *, peers, owners):
    peers = tuple(peers)
    blocks = {}
    for index, owner in owners.items():
        local = value[index]
        terms = [local]
        for peer in peers:
            if peer == owner:
                continue
            received = program.tensor(local.shape, dtype=local.dtype)[0, 0] if program.node == owner else local
            program.copy(local.on(peer), received.on(owner))
            terms.append(received)
        if program.node == owner:
            while len(terms) > 1:
                terms = [program.kernel_call(kernels.add,
                    grid=(1, 1), in_specs=(BlockSpec(None), BlockSpec(None)),
                    out_specs=BlockSpec(local.shape, lambda i, j: (i, j)),
                    out_shape=ShapeDtypeStruct(local.shape, local.dtype))(terms[i], terms[i + 1])[0, 0]
                    if i + 1 < len(terms) else terms[i] for i in range(0, len(terms), 2)]
            blocks[index] = terms[0]
    return value._with_blocks(blocks)


# design/algorithm-sources.md#collective
def all_gather(program, value, *, peers, owners):
    peers = tuple(peers)
    blocks = {}
    for index, owner in owners.items():
        shape = tuple(min(block, size - coordinate * block)
                      for coordinate, block, size in zip(index, value.block_shape, value.shape))
        result = value[index] if program.node == owner else program.tensor(shape, dtype=value.dtype)[0, 0]
        for peer in peers:
            if peer != owner:
                program.copy(result.on(owner), result.on(peer))
        blocks[index] = result
    return value._with_blocks(blocks)


# design/algorithm-sources.md#collective
def all_reduce(program, value, *, peers, owners):
    peers = tuple(peers)
    return all_gather(program, reduce_scatter(program, value, peers=peers, owners=owners),
                      peers=peers, owners=owners)
