import numpy as np

from .tensor import Dimension, broadcast_shape


# ../../../design/algorithm-sources.md#xonotic-neighborhood-algebra
def numerical_operands(operation, values, attributes):
    if operation in ('gather_vjp', 'take_along_axis_vjp'):
        return values[1:]
    if operation == 'expert_input_vjp':
        return values[1:]
    if operation == 'expert_weight_vjp':
        return (values[0], *values[2:])
    if operation not in ('neighborhood', 'neighborhood_vjp'):
        return values
    gram, target = attributes['gram'], attributes.get('target')
    positions = ((0, 1, 2, 3, 4) if gram else (2, 3, 4)) if target is None else (
        {0: (1, 2, 3, 4, 5), 1: (0, 2, 3, 4, 5),
         2: (0, 1, 3, 4, 5), 4: (0, 1, 2, 3, 5)}[target] if gram else
        {0: (), 1: (), 2: (3, 4, 5), 4: (2, 3, 5)}[target])
    return tuple(values[index] for index in positions)


# ../../../design/algorithm-sources.md#xonotic-expert-indexed-contractions
def expert_call(program, value, operation, values, shapes, local, peer, tile_k, tile_columns):
    import math
    from mesh import BlockSpec, ShapeDtypeStruct, kernels
    rows, inner = shapes[values[0].index]
    experts, weight_inner, hidden = shapes[values[1].index]
    if inner != weight_inner or math.prod(shapes[values[2].index]) != rows or (
            operation != 'expert_matmul' and shapes[values[3].index] != (rows, hidden)):
        raise ValueError('Expert operands require matching input, output and selected row dimensions')
    if np.dtype(values[2].dtype).kind not in 'iu':
        raise TypeError('Expert selection requires integer indices')
    if operation == 'expert_weight_vjp':
        block = (1, math.gcd(hidden, min(tile_columns, hidden)))
        base = program.tensor((experts, inner * hidden), block, value.dtype)
        if program.node == peer:
            for ref in base.blocks.values():
                program.constant(ref, np.zeros(ref.shape, dtype=value.dtype))
        destinations = matrix_view(local[values[2].index], (rows, 1))
        base_arg, selected_arg, input_arg, gradient_arg = kernels.arguments(4)
        row, column = kernels.indices()
        update = input_arg.reshape((rows, inner)).at(row, column // hidden) * gradient_arg.reshape((rows, hidden)).at(row, column % hidden)
        result = program.kernel_call(kernels.expression(kernels.indexed_add(base_arg, selected_arg, update)),
            grid=base.grid, in_specs=(BlockSpec(None),) * 4,
            out_specs=BlockSpec(block, lambda i, j: (i, j)),
            out_shape=ShapeDtypeStruct(base.shape, value.dtype), peer=peer)(
                base, destinations, local[values[0].index], local[values[3].index])
        return matrix_view(result, (experts * inner, hidden))
    backward = operation == 'expert_input_vjp'
    operand = values[3] if backward else values[0]
    output_width, contraction = (inner, hidden) if backward else (hidden, inner)
    feature_tile = min(tile_columns, output_width)
    operand_arg, weight_arg, selected_arg = kernels.arguments(3)
    row = kernels.program_id(0)
    selected = selected_arg.reshape((rows,)).at(row)
    if values[2].dtype.startswith('int'):
        selected = kernels.select(selected < 0, selected + experts, selected)
    feature = kernels.arange(feature_tile).T + kernels.program_id(1) * feature_tile
    k = kernels.arange(contraction, tile=tile_k)
    weight_at = (selected, feature, k) if backward else (selected, k, feature)
    product = operand_arg.reshape((rows, contraction)).at(row, k) * weight_arg.reshape((experts, inner, hidden)).at(*weight_at)
    block = (1, feature_tile)
    return program.kernel_call(kernels.expression(product.sum().T),
        grid=(rows, (output_width + feature_tile - 1) // feature_tile),
        in_specs=(BlockSpec(None),) * 3, out_specs=BlockSpec(block, lambda i, j: (i, j)),
        out_shape=ShapeDtypeStruct((rows, output_width), value.dtype), peer=peer)(
            local[operand.index], local[values[1].index], local[values[2].index])


# ../../../design/algorithm-sources.md#xonotic-neighborhood-algebra
def neighborhood_call(program, value, operation, values, attributes, shapes, local, peer, tile_columns, statistics):
    import math
    from mesh import BlockSpec, ShapeDtypeStruct, kernels
    gram, target = attributes['gram'], attributes.get('target')
    observers, width = shapes[values[0].index]
    neighbors = shapes[values[4].index][1]
    if np.dtype(values[3].dtype).kind not in 'iu':
        raise TypeError('Neighborhood indices must have integer dtype')
    if (shapes[values[3].index] != (observers, neighbors) or
            shapes[values[4].index] != (observers, neighbors) or
            shapes[values[1].index] != shapes[values[2].index] or
            shapes[values[2].index][1] != width or
            (target is not None and shapes[values[5].index] != (observers, width))):
        raise ValueError('Neighborhood operands require matching observer, source and feature dimensions')
    edges, feature_tile = observers * neighbors, min(tile_columns, width)
    output_shape = shapes[value.index]
    if target in (0, 1) and not gram:
        result = program.tensor(output_shape, (1, min(tile_columns, output_shape[1])), value.dtype)
        if program.node == peer:
            for ref in result.blocks.values():
                program.constant(ref, np.zeros(ref.shape, dtype=value.dtype))
        return result
    operands = numerical_operands(operation, values, attributes)
    inputs = [local[operand.index] for operand in operands]
    arguments = dict(zip((operand.index for operand in operands), kernels.arguments(len(inputs))))

    # ../../../design/algorithm-sources.md#xonotic-neighborhood-algebra
    def call(expression, bound, shape, block, dtype):
        return program.kernel_call(kernels.expression(expression),
            grid=tuple((size + tile - 1) // tile for size, tile in zip(shape, block)),
            in_specs=(BlockSpec(None),) * len(bound), out_specs=BlockSpec(block, lambda i, j: (i, j)),
            out_shape=ShapeDtypeStruct(shape, dtype), peer=peer)(*bound)

    # ../../../design/algorithm-sources.md#xonotic-neighborhood-algebra
    def source(edge):
        index = arguments[values[3].index].reshape(shapes[values[3].index]).at(edge // neighbors, edge % neighbors)
        return kernels.select(index < 0, index + shapes[values[2].index][0], index) if values[3].dtype.startswith('int') else index

    # ../../../design/algorithm-sources.md#xonotic-neighborhood-algebra
    def load(position, edge, feature):
        coordinates = (edge // neighbors, edge % neighbors) if position == 4 else (
            edge // neighbors if position in (0, 5) else source(edge), feature)
        return 1.0 * arguments[values[position].index].reshape(shapes[values[position].index]).at(*coordinates)

    # ../../../design/algorithm-sources.md#xonotic-neighborhood-algebra
    def statistic(left, right):
        key = (peer, values[left].index, values[right].index, values[3].index,
               observers, neighbors, width, feature_tile)
        if key not in statistics:
            edge, feature = kernels.program_id(0), kernels.arange(width, tile=feature_tile)
            statistic_value = (load(left, edge, feature) * load(right, edge, feature)).sum()
            statistics[key] = call(statistic_value, inputs, (edges, 1), (1, 1), np.float32)
        argument = kernels.arguments(len(inputs) + 1)[-1]
        inputs.append(statistics[key])
        return argument

    affinity = statistic(0, 1) if gram and target in (None, 2, 4) else None
    response = statistic(5, 2) if target in (0, 1, 4) else None
    row, column = kernels.indices()
    edge = kernels.program_id(0) + row if target == 4 else row
    selected = source(edge)
    valid = (selected >= 0) & (selected < shapes[values[2].index][0])
    coefficient = affinity.at(edge, 0) / math.sqrt(width) if affinity is not None else 1
    if target == 4:
        result = call(kernels.select(valid, response.at(edge, 0) * coefficient, 0), inputs,
                      (edges, 1), (1, 1), value.dtype)
        return matrix_view(result, output_shape)
    weight = load(4, edge, column)
    if target in (0, 1):
        contribution = weight * response.at(edge, 0) / math.sqrt(width) * load(1 if target == 0 else 0, edge, column)
    else:
        contribution = weight * coefficient * load(2 if target is None else 5, edge, column)
    key_edge = kernels.program_id(0) + row
    key_source = source(key_edge)
    destination = key_edge // neighbors if target in (None, 0) else key_source
    destination = kernels.select((key_source >= 0) & (key_source < shapes[values[2].index][0]), destination, 0xffffffff)
    destinations = call(destination, inputs, (edges, 1), (1, 1), np.int64)
    block = (1, min(tile_columns, output_shape[1]))
    base = program.tensor(output_shape, block, value.dtype)
    if program.node == peer:
        for ref in base.blocks.values():
            program.constant(ref, np.zeros(ref.shape, dtype=value.dtype))
    base_arg, destination_arg = kernels.arguments(len(inputs) + 2)[-2:]
    return call(kernels.indexed_add(base_arg, destination_arg, contribution), (*inputs, base, destinations),
                output_shape, block, value.dtype)


# ../../../design/algorithm-sources.md#xonotic-batched-contractions
def matrix_batch(tensor, shape, batch):
    import math
    from mesh import Tensor
    if tensor.shape == shape and batch == 0:
        return tensor
    rows, columns = shape
    if not rows or not columns:
        return tensor.program.tensor(shape, dtype=tensor.dtype)
    block = (math.gcd(rows, tensor.block_shape[0]) if tensor.grid[0] > 1 else rows,
             min(columns, tensor.block_shape[1]))
    result = object.__new__(Tensor)
    result.program, result.dtype, result.handle = tensor.program, tensor.dtype, tensor.handle
    result.shape, result.block_shape = shape, block
    result.grid = tuple((size + tile - 1) // tile for size, tile in zip(shape, block))
    result.blocks = {(i, j): tensor.region(batch * rows + i * block[0], j * block[1],
                                         min(block[0], rows - i * block[0]), min(block[1], columns - j * block[1]))
                     for i in range(result.grid[0]) for j in range(result.grid[1])}
    return result


# ../../../design/algorithm-sources.md#typed-integer-contractions
def batched_matmul(program, left, right, left_shape, right_shape, attributes, *,
                   tile_rows, tile_k, tile_columns, peer, output_dtype):
    import itertools
    import math
    from mesh import Tensor, nn
    batch_shape = broadcast_shape(left_shape[:-2], right_shape[:-2])
    sources = (matrix_view(left, (math.prod(left_shape[:-1]), left_shape[-1])),
               matrix_view(right, (math.prod(right_shape[:-1]), right_shape[-1])))
    shapes, transposes = (left_shape, right_shape), (attributes['transpose_left'], attributes['transpose_right'])
    caches, batches = ({}, {}), []
    for coordinate in itertools.product(*(range(size) for size in batch_shape)):
        operands = []
        for source, shape, transpose, cache in zip(sources, shapes, transposes, caches):
            ordinal = 0
            for size, index in zip(shape[:-2], coordinate[len(batch_shape) - len(shape) + 2:]):
                ordinal = ordinal * size + (0 if size == 1 else index)
            if ordinal not in cache:
                view = matrix_batch(source, shape[-2:], ordinal)
                cache[ordinal] = view.T if transpose else view
            operands.append(cache[ordinal])
        row_tile = math.gcd(tile_rows, operands[0].shape[0]) if math.prod(batch_shape) > 1 else tile_rows
        batches.append(nn.linear(program, *operands, tile_rows=row_tile, tile_k=tile_k,
                                 tile_columns=tile_columns, peer=peer, output_dtype=output_dtype))
    first = batches[0]
    if len(batches) == 1:
        return first
    if any(batch.shape != first.shape or batch.block_shape != first.block_shape for batch in batches):
        raise ValueError('Batched contractions require a common realized output layout')
    result = object.__new__(Tensor)
    result.program, result.dtype, result.handle = first.program, first.dtype, first.handle
    result.shape = (len(batches) * first.shape[0], first.shape[1])
    result.block_shape, result.grid = first.block_shape, (len(batches) * first.grid[0], first.grid[1])
    result.blocks = {(batch * first.grid[0] + i, j): ref
                     for batch, tensor in enumerate(batches) for (i, j), ref in tensor.blocks.items()}
    return result


# ../../../design/algorithm-sources.md#xonotic-partitioned-reshape
def matrix_view(tensor, shape):
    import math
    from mesh import Tensor, Ref
    from mesh._native import View as NativeView
    shape = tuple(shape)
    if len(shape) != 2 or min(shape) < 0 or math.prod(shape) != math.prod(tensor.shape):
        raise ValueError('Matrix view must preserve element count with two nonnegative dimensions')
    if not math.prod(shape):
        return tensor.program.tensor(shape, dtype=tensor.dtype)
    if tensor.shape == shape:
        return tensor
    if tensor.shape[::-1] == shape and 1 in shape:
        return tensor.T
    if tensor.shape[1] == 1 and tensor.shape[0] == shape[0] * shape[1] and tensor.block_shape[0] % shape[1] == 0:
        block_shape = (tensor.block_shape[0] // shape[1], shape[1])
        grid = (tensor.grid[0], 1)
        regions = tuple((coordinate, ref, (ref.shape[0] // shape[1], shape[1])) for coordinate, ref in tensor.blocks.items())
    elif tensor.grid == (1, 1) and (1 in tensor.shape or
            tensor[0, 0].view.row_stride == tensor.shape[1] * tensor[0, 0].view.column_stride):
        block_shape, grid = shape, (1, 1)
        regions = (((0, 0), tensor[0, 0], shape),)
    else:
        width = math.gcd(shape[1], tensor.shape[1], tensor.block_shape[1])
        block_shape, grid = (1, width), (shape[0], shape[1] // width)
        regions = []
        for i in range(grid[0]):
            for j in range(grid[1]):
                row, column = divmod(i * shape[1] + j * width, tensor.shape[1])
                regions.append(((i, j), tensor.region(row, column, 1, width), (1, width)))
    result = object.__new__(Tensor)
    result.program, result.dtype, result.handle = tensor.program, tensor.dtype, tensor.handle
    result.shape, result.block_shape, result.grid = shape, block_shape, grid
    result.blocks = {}
    for coordinate, ref, extent_shape in regions:
        array = ref.array.view()
        array.shape = extent_shape
        view = NativeView.from_buffer_copy(ref.view)
        view.rows, view.columns = extent_shape
        view.row_stride, view.column_stride = (stride // tensor.dtype.itemsize for stride in array.strides)
        result.blocks[coordinate] = Ref(tensor.program, view, tensor.dtype)
    return result


# ../../../design/algorithm-sources.md#xonotic-logical-indexing
def logical_coordinates(kernels, shape, block):
    import math
    row, column = kernels.indices()
    row = kernels.program_id(0) * block[0] + row
    column = kernels.program_id(1) * block[1] + column
    return tuple((row // math.prod(shape[axis + 1:-1])) % size
                 for axis, size in enumerate(shape[:-1])) + ((column,) if shape else ())


# ../../../design/algorithm-sources.md#xonotic-gather-transpose
def gather_coordinates(kernels, arguments, values, shapes, attributes, coordinates, capacity):
    mapping = tuple(tuple(part.resolve(capacity) if isinstance(part, Dimension) else part for part in item)
                    for item in attributes['mapping'])
    advanced, adjacent = attributes['advanced'], attributes['adjacent']
    axis = 0 if adjacent else len(advanced)
    advanced_axis = None if adjacent else 0
    source_axis, source_coordinates = 0, []
    for item in mapping:
        if item[0] == 'new':
            axis += 1
            continue
        if item[0] == 'index':
            if advanced_axis is None:
                advanced_axis = axis
                axis += len(advanced)
            selected = values[1 + item[1]]
            selected_shape = shapes[selected.index]
            at = tuple(0 if size == 1 else coordinates[advanced_axis + len(advanced) - len(selected_shape) + i]
                       for i, size in enumerate(selected_shape))
            selected_value = arguments[item[1]].at(*at)
            coordinate = kernels.select(selected_value < 0, selected_value + shapes[values[0].index][source_axis], selected_value) if selected.dtype.startswith('int') else selected_value
        elif item[0] == 'fixed':
            coordinate = item[1]
        else:
            coordinate = item[1] + coordinates[axis] * item[2]
            axis += 1
        source_coordinates.append(coordinate)
        source_axis += 1
    return tuple(source_coordinates)


# ../../../design/algorithm-sources.md#xonotic-take-transpose
def take_coordinates(kernels, source_shape, index_shape, index_dtype, index_value, axis, coordinates):
    index_at = tuple(0 if size == 1 else coordinate for size, coordinate in zip(index_shape, coordinates))
    selected = index_value.at(*index_at)
    if index_dtype.startswith('int'):
        selected = kernels.select(selected < 0, selected + source_shape[axis], selected)
    return tuple(selected if i == axis else 0 if size == 1 else coordinates[i]
                 for i, size in enumerate(source_shape))


# ../../../design/algorithm-sources.md#xonotic-output-liveness
def row_gather_gradient(value, operation, values, attributes, shapes):
    return operation == 'gather_vjp' and len(values) == 3 and len(shapes[values[1].index]) == 1 and (
        (len(shapes[value.index]) == 1 and tuple(attributes['mapping']) == (('index', 0),)) or
        (len(shapes[value.index]) == 2 and tuple(attributes['mapping']) == (('index', 0), ('slice', 0, 1)) and
         shapes[values[-1].index] == (shapes[values[1].index][0], shapes[value.index][1])))


# ../../../design/algorithm-sources.md#xonotic-output-liveness
def kernel_calls(program, graph, capacity, inputs, *, outputs, root_peer=None,
                 tile_rows=64, tile_k=128, tile_columns=128):
    from mesh import BlockSpec, ShapeDtypeStruct, kernels, nn
    import math

    shapes = {value.index: tuple(size.resolve(capacity) if isinstance(size, Dimension) else size for size in value.shape)
              for value, _, _, _, _ in graph.nodes}
    for value, operation, values, _, _ in graph.nodes:
        if operation == 'reshape' and math.prod(shapes[value.index]) != math.prod(shapes[values[0].index]):
            raise ValueError('Reshape must preserve element count')
    outputs = tuple(outputs)
    if any(value.graph is not graph for value in outputs):
        raise ValueError('Requested outputs must belong to the compiled graph')
    live = {value.index for value in outputs}
    nodes = []
    for node in reversed(graph.nodes):
        value, operation, values, attributes, owner = node
        if value.index not in live or value.index in inputs:
            continue
        # ../../../design/algorithm-sources.md#stable-indexed-ordering
        if operation == 'argpartition':
            width = shapes[values[0].index][attributes['axis']]
            kth = attributes['kth'].resolve(capacity) if isinstance(attributes['kth'], Dimension) else int(attributes['kth'])
            if not isinstance(kth, (int, np.integer)):
                raise TypeError('Resolved partition position must be an integer')
            if not -width <= kth < width:
                raise ValueError('Partition position is outside the sorted axis')
            attributes = dict(attributes, kth=kth + width if kth < 0 else kth)
            node = value, operation, values, attributes, owner
        nodes.append(node)
        dependencies = numerical_operands(operation, values, attributes) if math.prod(shapes[value.index]) else ()
        live.update(operand.index for operand in dependencies)
    nodes.reverse()
    constants = {value.index: data for value, data in graph.constants.values()}
    tensors = dict(inputs)
    peers = {0: program.node if root_peer is None else root_peer}
    peers.update({region['owner']: region['peer'] for region in graph.regions.values()})
    owners = {value.index: peers[owner] for value, _, _, _, owner in graph.nodes}
    replicas, statistics = {}, {}
    for value, operation, values, attributes, owner in nodes:
        peer = peers[owner]
        if value.index in tensors:
            continue
        # ../../../design/algorithm-sources.md#indexed-range-generation
        shape = shapes[value.index]
        storage_shape = (math.prod(shape[:-1]), shape[-1]) if shape else (1, 1)
        if not math.prod(shape):
            tensors[value.index] = program.tensor(storage_shape, dtype=value.dtype)
            continue
        if operation in ('reshape', 'stop_gradient'):
            tensors[value.index] = tensors[values[0].index]
            owners[value.index] = owners[values[0].index]
            continue
        if operation in ('constant', 'dimension'):
            tensor = program.tensor(storage_shape, dtype=value.dtype)
            data = constants[value.index] if operation == 'constant' else np.asarray(attributes['expression'].resolve(capacity), dtype=value.dtype)
            if program.node == peer:
                program.constant(tensor[0, 0], data.reshape(storage_shape))
            tensors[value.index] = tensor
            continue
        # ../../../design/algorithm-sources.md#indexed-range-generation
        if operation == 'arange':
            start, step = (attributes[name].resolve(capacity) if isinstance(attributes[name], Dimension) else attributes[name] for name in ('start', 'step'))
            block = (1, min(tile_columns, shape[0]))
            _, column = kernels.indices()
            ordinal = kernels.program_id(1) * block[1] + column
            if np.dtype(value.dtype).kind in 'iu':
                mask = 0xffffffffffffffff
                result = (ordinal & mask) * (int(step) & mask) + (int(start) & mask)
            else:
                result = float(start) + ordinal * float(step)
            tensors[value.index] = program.kernel_call(kernels.expression(result),
                grid=(1, (shape[0] + block[1] - 1) // block[1]), in_specs=(),
                out_specs=BlockSpec(block, lambda i, j: (i, j)),
                out_shape=ShapeDtypeStruct((1, shape[0]), value.dtype), peer=peer)()
            continue
        row_gradient = row_gather_gradient(value, operation, values, attributes, shapes)
        local = {}
        for operand in numerical_operands(operation, values, attributes):
            tensor = tensors[operand.index]
            sender = owners[operand.index]
            if sender != peer and tensor.blocks:
                key = (operand.index, peer)
                if key not in replicas:
                    transposed = tensor[0, 0].view.row_stride == 1 and tensor[0, 0].view.column_stride != 1
                    backing = tensor.T if transposed else tensor
                    replica = program.tensor(backing.shape, block_shape=backing.block_shape, dtype=backing.dtype)
                    replica = replica.T if transposed else replica
                    program.copy(tensor.on(sender), replica.on(peer))
                    replicas[key] = replica
                tensor = replicas[key]
            local[operand.index] = tensor
        shape = shapes[value.index]
        # ../../../design/algorithm-sources.md#counter-based-random-generation
        if operation == 'random_normal':
            key_size = math.prod(shapes[values[0].index])
            if key_size < 2:
                raise ValueError('Counter-based normal generation requires two key words')
            if np.dtype(values[0].dtype).kind not in 'iu':
                raise TypeError('Counter-based normal generation requires integer key words')
            block = (min(tile_rows, storage_shape[0]), min(tile_columns, storage_shape[1]))
            row, column = kernels.indices()
            ordinal = (kernels.program_id(0) * block[0] + row) * storage_shape[1] + kernels.program_id(1) * block[1] + column
            key, = kernels.arguments(1)
            key = key.reshape((key_size,))
            result = kernels.random_normal(key.at(0).astype('uint32'), key.at(1).astype('uint32'), ordinal)
            tensors[value.index] = program.kernel_call(kernels.expression(result),
                grid=tuple((size + tile - 1) // tile for size, tile in zip(storage_shape, block)),
                in_specs=(BlockSpec(None),), out_specs=BlockSpec(block, lambda i, j: (i, j)),
                out_shape=ShapeDtypeStruct(storage_shape, value.dtype), peer=peer)(local[values[0].index])
            continue
        # ../../../design/algorithm-sources.md#stable-indexed-ordering
        if operation in ('argsort', 'argpartition'):
            axis = attributes['axis']
            operand = local[values[0].index]
            argument, = kernels.arguments(1)
            if len(shape) <= 2 or axis == len(shape) - 1:
                matrix_shape = (math.prod(shape[:-1]), shape[-1])
                operand = matrix_view(operand, matrix_shape)
                sorted_axis = axis if len(shape) == 2 else 1
                block = tuple(math.gcd(min(tile, size), operand.block_shape[i] if operand.grid[i] > 1 else 0)
                              for i, (tile, size) in enumerate(zip((tile_rows, tile_columns), matrix_shape)))
                tensors[value.index] = program.kernel_call(kernels.expression(argument.argsort(axis=sorted_axis)),
                    grid=tuple((size + tile - 1) // tile for size, tile in zip(matrix_shape, block)),
                    in_specs=(BlockSpec(None),), out_specs=BlockSpec(block, lambda i, j: (i, j)),
                    out_shape=ShapeDtypeStruct(matrix_shape, value.dtype), peer=peer)(operand)
            else:
                width = shape[axis]
                retained = tuple(i for i in range(len(shape)) if i != axis)
                segments = math.prod(shape[i] for i in retained)
                segment = kernels.program_id(0)
                at = [None] * len(shape)
                for position, i in enumerate(retained):
                    at[i] = (segment // math.prod(shape[j] for j in retained[position + 1:])) % shape[i]
                at[axis] = kernels.arange(width, tile=tile_columns)
                key = argument.reshape(shape).at(*at).astype(values[0].dtype)
                block = (1, min(tile_columns, width))
                ordered = program.kernel_call(kernels.expression(key.argsort()),
                    grid=(segments, (width + block[1] - 1) // block[1]), in_specs=(BlockSpec(None),),
                    out_specs=BlockSpec(block, lambda i, j: (i, j)),
                    out_shape=ShapeDtypeStruct((segments, width), value.dtype), peer=peer)(operand)
                block = (min(tile_rows, storage_shape[0]), min(tile_columns, storage_shape[1]))
                coordinates = logical_coordinates(kernels, shape, block)
                segment = sum(coordinates[i] * math.prod(shape[j] for j in retained[position + 1:])
                              for position, i in enumerate(retained))
                result = argument.at(segment, coordinates[axis])
                tensors[value.index] = program.kernel_call(kernels.expression(result),
                    grid=tuple((size + tile - 1) // tile for size, tile in zip(storage_shape, block)),
                    in_specs=(BlockSpec(None),), out_specs=BlockSpec(block, lambda i, j: (i, j)),
                    out_shape=ShapeDtypeStruct(storage_shape, value.dtype), peer=peer)(ordered)
            continue
        if operation in ('expert_matmul', 'expert_input_vjp', 'expert_weight_vjp'):
            tensors[value.index] = expert_call(program, value, operation, values, shapes, local, peer, tile_k, tile_columns)
            continue
        if operation in ('neighborhood', 'neighborhood_vjp'):
            tensors[value.index] = neighborhood_call(program, value, operation, values, attributes, shapes, local, peer, tile_columns, statistics)
            continue
        if row_gradient:
            vector = len(shape) == 1
            output_shape = (shape[0], 1) if vector else shape
            destinations = matrix_view(local[values[1].index], (shapes[values[1].index][0], 1))
            gradient_shape = (shapes[values[-1].index][0], 1) if vector else shapes[values[-1].index]
            updates = matrix_view(local[values[-1].index], gradient_shape)
            block = (min(tile_rows, output_shape[0]), math.gcd(min(tile_columns, output_shape[1]),
                updates.block_shape[1] if updates.grid[1] > 1 else 0))
            base = program.tensor(output_shape, block_shape=block, dtype=value.dtype)
            if program.node == peer:
                for ref in base.blocks.values():
                    program.constant(ref, np.zeros(ref.shape, dtype=value.dtype))
            base_arg, index_arg, update_arg = kernels.arguments(3)
            result = program.kernel_call(kernels.expression(kernels.indexed_add(base_arg, index_arg, update_arg)),
                grid=base.grid, in_specs=(BlockSpec(None),) * 3,
                out_specs=BlockSpec(block, lambda i, j: (i, j)),
                out_shape=ShapeDtypeStruct(output_shape, value.dtype), peer=peer)(base, destinations, updates)
            tensors[value.index] = result.T if vector else result
            continue
        # ../../../design/algorithm-sources.md#xonotic-gather-transpose
        if operation in ('gather_vjp', 'take_along_axis_vjp'):
            indices = tuple(local[v.index] for v in values[1:-1])
            cotangent = local[values[-1].index]
            cotangent_shape = shapes[values[-1].index]
            updates = math.prod(cotangent_shape)
            key_block = (min(tile_rows, updates), 1)
            row, _ = kernels.indices()
            ordinal = kernels.program_id(0) * key_block[0] + row
            coordinates = tuple((ordinal // math.prod(cotangent_shape[i + 1:])) % size
                                for i, size in enumerate(cotangent_shape))
            index_values = tuple(argument.reshape(shapes[value.index])
                                 for argument, value in zip(kernels.arguments(len(indices)), values[1:-1]))
            source_at = (gather_coordinates(kernels, index_values, values, shapes, attributes, coordinates, capacity)
                if operation == 'gather_vjp' else take_coordinates(kernels, shape, shapes[values[1].index],
                    values[1].dtype, index_values[0], attributes['axis'], coordinates))
            destination = sum(coordinate * math.prod(shape[i + 1:]) for i, coordinate in enumerate(source_at))
            valid = True
            for coordinate, size in zip(source_at, shape):
                valid = kernels.select(valid, (coordinate >= 0) & (coordinate < size), False)
            destination = kernels.select(valid, destination, 0xffffffff)
            destinations = program.kernel_call(kernels.expression(destination),
                grid=((updates + key_block[0] - 1) // key_block[0], 1),
                in_specs=(BlockSpec(None),) * len(indices), out_specs=BlockSpec(key_block, lambda i, j: (i, j)),
                out_shape=ShapeDtypeStruct((updates, 1), 'int64'), peer=peer)(*indices)
            storage_shape = (math.prod(shape[:-1]), shape[-1]) if shape else (1, 1)
            block = (min(tile_rows, storage_shape[0]) * storage_shape[1], 1)
            base = program.tensor((math.prod(shape), 1), block_shape=block, dtype=value.dtype)
            if program.node == peer:
                for ref in base.blocks.values():
                    program.constant(ref, np.zeros(ref.shape, dtype=value.dtype))
            base_value, index_value, cotangent_value = kernels.arguments(3)
            row, _ = kernels.indices()
            result = program.kernel_call(kernels.expression(kernels.indexed_add(base_value, index_value,
                cotangent_value.reshape((updates,)).at(row))),
                grid=base.grid, in_specs=(BlockSpec(None),) * 3,
                out_specs=BlockSpec(block, lambda i, j: (i, j)),
                out_shape=ShapeDtypeStruct(base.shape, value.dtype), peer=peer)(base, destinations, cotangent)
            tensors[value.index] = matrix_view(result, storage_shape)
            continue
        # ../../../design/algorithm-sources.md#xonotic-row-scatter
        if operation == 'scatter_add':
            if np.dtype(values[1].dtype).kind not in 'iu':
                raise TypeError('Scatter indices must have integer dtype')
            index_shape, update_shape = (shapes[v.index] for v in values[1:])
            selected_shape = index_shape + shape[1:]
            if broadcast_shape(update_shape, selected_shape) != selected_shape:
                raise ValueError('Scatter updates must broadcast to the selected rows')
            storage_shape = (shape[0], 1) if len(shape) == 1 else (math.prod(shape[:-1]), shape[-1])
            base = matrix_view(local[values[0].index], storage_shape)
            indices, updates = (local[v.index] for v in values[1:])
            row_expansion = math.prod(shape[1:-1]) if len(shape) > 1 else 1
            update_rows = math.prod(index_shape) * row_expansion
            if row_expansion == 1 and len(index_shape) <= 1 and indices.shape in ((update_rows, 1), (1, update_rows)):
                destinations = matrix_view(indices, (update_rows, 1))
            else:
                key_block = (min(tile_rows, update_rows), 1)
                row, _ = kernels.indices()
                ordinal = kernels.program_id(0) * key_block[0] + row
                selected_ordinal = ordinal // row_expansion
                index_at = tuple((selected_ordinal // math.prod(index_shape[i + 1:])) % size
                                 for i, size in enumerate(index_shape))
                index_value, = kernels.arguments(1)
                selected = index_value.reshape(index_shape).at(*index_at)
                if values[1].dtype.startswith('int'):
                    selected = kernels.select(selected < 0, selected + shape[0], selected)
                destination = kernels.select((selected >= 0) & (selected < shape[0]),
                    selected * row_expansion + ordinal % row_expansion, 0xffffffff)
                destinations = program.kernel_call(kernels.expression(destination),
                    grid=((update_rows + key_block[0] - 1) // key_block[0], 1),
                    in_specs=(BlockSpec(None),), out_specs=BlockSpec(key_block, lambda i, j: (i, j)),
                    out_shape=ShapeDtypeStruct((update_rows, 1), 'int64'), peer=peer)(indices)
            base_value, index_value, update_value = kernels.arguments(3)
            update_matrix = (math.prod(update_shape), 1) if len(shape) == 1 else update_shape if len(update_shape) == 2 else (1, math.prod(update_shape))
            direct = len(index_shape) <= 1 and len(shape) <= 2 and len(update_shape) <= 2 and (
                updates.shape == update_matrix or (updates.shape[::-1] == update_matrix and 1 in update_matrix))
            if direct:
                updates = matrix_view(updates, update_matrix).broadcast_to((update_rows, storage_shape[1]))
            else:
                row, column = kernels.indices()
                selected_ordinal, trailing_ordinal = row // row_expansion, row % row_expansion
                coordinates = tuple((selected_ordinal // math.prod(index_shape[i + 1:])) % size
                                    for i, size in enumerate(index_shape))
                coordinates += tuple((trailing_ordinal // math.prod(shape[i + 2:-1])) % size
                                     for i, size in enumerate(shape[1:-1]))
                if len(shape) > 1:
                    coordinates += (column,)
                update_at = tuple(0 if size == 1 else coordinate for size, coordinate in
                                  zip(update_shape, coordinates[len(selected_shape) - len(update_shape):]))
                update_value = update_value.reshape(update_shape).at(*update_at)
            block = (math.gcd(min(tile_rows, base.shape[0]), base.block_shape[0] if base.grid[0] > 1 or len(shape) == 1 else 0),
                     math.gcd(min(tile_columns, base.shape[1]), base.block_shape[1] if base.grid[1] > 1 else 0))
            if direct and updates.grid[1] > 1:
                block = (block[0], math.gcd(block[1], updates.block_shape[1]))
            result = program.kernel_call(kernels.expression(kernels.indexed_add(base_value, index_value, update_value)),
                grid=tuple((size + tile - 1) // tile for size, tile in zip(base.shape, block)),
                in_specs=(BlockSpec(None),) * 3, out_specs=BlockSpec(block, lambda i, j: (i, j)),
                out_shape=ShapeDtypeStruct(base.shape, value.dtype), peer=peer)(base, destinations, updates)
            tensors[value.index] = result.T if len(shape) == 1 else result
            continue
        # ../../../design/algorithm-sources.md#shared-associative-reductions
        if operation.startswith('reduce_') and not attributes['axes']:
            operand = local[values[0].index]
            if operand.dtype == np.dtype(value.dtype):
                tensors[value.index] = operand
            else:
                argument, = kernels.arguments(1)
                result = argument.equal(0).equal(False) if operation in ('reduce_any', 'reduce_all') else argument.astype(value.dtype)
                tensors[value.index] = program.kernel_call(kernels.expression(result),
                    grid=operand.grid, in_specs=(BlockSpec(operand.block_shape, lambda i, j: (i, j)),),
                    out_specs=BlockSpec(operand.block_shape, lambda i, j: (i, j)),
                    out_shape=ShapeDtypeStruct(operand.shape, value.dtype), peer=peer)(operand)
            continue
        # ../../../design/algorithm-sources.md#shared-associative-reductions
        if operation.startswith('reduce_') and len(shapes[values[0].index]) > 2:
            operand_shape = shapes[values[0].index]
            axes = tuple(sorted(set(attributes['axes'])))
            retained = tuple(axis for axis in range(len(operand_shape)) if axis not in axes)
            matrix_shape = (math.prod(shape[:-1]), shape[-1]) if shape else (1, 1)
            width = min(tile_columns, matrix_shape[1])
            output_index = kernels.program_id(0) * matrix_shape[1] + kernels.program_id(1) * width + kernels.arange(width).T
            count = math.prod(operand_shape[axis] for axis in axes)
            reduction_index = kernels.arange(count, tile=tile_k)
            coordinates = [None] * len(operand_shape)
            for domain, ordinal in ((retained, output_index), (axes, reduction_index)):
                for position, axis in enumerate(domain):
                    coordinates[axis] = (ordinal // math.prod(operand_shape[i] for i in domain[position + 1:])) % operand_shape[axis]
            argument, = kernels.arguments(1)
            term = argument.reshape(operand_shape).at(*coordinates)
            if operation in ('reduce_max', 'reduce_min'):
                term = term.astype(values[0].dtype)
            if operation in ('reduce_sum', 'reduce_mean') and np.dtype(values[0].dtype).kind in 'iu':
                term = term & 0xffffffffffffffff
            result = getattr(term, 'sum' if operation == 'reduce_mean' else operation[7:])().T
            if operation == 'reduce_mean':
                if np.dtype(values[0].dtype).kind != 'f':
                    result = result.astype('uint64' if np.dtype(values[0].dtype).kind == 'u' else 'int64').astype('float32')
                result = result / float(count)
            reduced = program.kernel_call(kernels.expression(result),
                grid=(matrix_shape[0], (matrix_shape[1] + width - 1) // width),
                in_specs=(BlockSpec(None),), out_specs=BlockSpec((1, width), lambda i, j: (i, j)),
                out_shape=ShapeDtypeStruct(matrix_shape, value.dtype), peer=peer)(local[values[0].index])
            tensors[value.index] = reduced
            continue
        # ../../../design/algorithm-sources.md#shared-associative-reductions
        if operation.startswith('reduce_') and 1 <= len(shapes[values[0].index]) <= 2:
            operand_shape = shapes[values[0].index]
            operand = matrix_view(local[values[0].index], operand_shape if len(operand_shape) == 2 else (1, math.prod(operand_shape)))
            axes = tuple(sorted(set(attributes['axes']))) if len(operand_shape) == 2 else (1,)
            reduced_shape = tuple(1 if axis in axes else size for axis, size in enumerate(operand.shape))
            block = tuple(math.gcd(min(tile, size), operand.block_shape[axis] if operand.grid[axis] > 1 else 0)
                          for axis, (tile, size) in enumerate(zip((tile_rows, tile_rows), reduced_shape)))
            argument, = kernels.arguments(1)
            term = argument & 0xffffffffffffffff if operation in ('reduce_sum', 'reduce_mean') and operand.dtype.kind in 'iu' else argument
            result = getattr(term, 'sum' if operation == 'reduce_mean' else operation[7:])(axis=axes)
            divisor = math.prod(operand.shape[axis] for axis in axes)
            if operation == 'reduce_mean':
                if operand.dtype.kind != 'f':
                    result = result.astype('uint64' if operand.dtype.kind == 'u' else 'int64').astype('float32')
                result = result / float(divisor)
            reduced = program.kernel_call(kernels.expression(result),
                grid=tuple((size + tile - 1) // tile for size, tile in zip(reduced_shape, block)),
                in_specs=(BlockSpec(None),),
                out_specs=BlockSpec(block, lambda i, j: (i, j)),
                out_shape=ShapeDtypeStruct(reduced_shape, value.dtype), peer=peer)(operand)
            tensors[value.index] = reduced
            continue
        # ../../../design/algorithm-sources.md#xonotic-logical-indexing
        if operation in ('gather', 'take_along_axis', 'concatenate', 'transpose'):
            matrix_shape = (math.prod(shape[:-1]), shape[-1]) if shape else (1, 1)
            block = (min(tile_rows, matrix_shape[0]), min(tile_columns, matrix_shape[1]))
            coordinates = logical_coordinates(kernels, shape, block)
            arguments = tuple(argument.reshape(shapes[value.index])
                              for argument, value in zip(kernels.arguments(len(values)), values))
            if operation == 'gather':
                source_at = gather_coordinates(kernels, arguments[1:], values, shapes, attributes, coordinates, capacity)
                result = arguments[0].at(*source_at)
            elif operation == 'take_along_axis':
                source_at = take_coordinates(kernels, shapes[values[0].index], shapes[values[1].index],
                    values[1].dtype, arguments[1], attributes['axis'], coordinates)
                result = arguments[0].at(*source_at)
            elif operation == 'transpose':
                source_at = [0] * len(shape)
                for coordinate, axis in zip(coordinates, attributes['axes']):
                    source_at[axis] = coordinate
                result = arguments[0].at(*source_at)
            else:
                axis, offset, terms = attributes['axis'], 0, []
                for argument, operand in zip(arguments, values):
                    at = list(coordinates)
                    at[axis] = at[axis] - offset
                    offset += shapes[operand.index][axis]
                    terms.append((coordinates[axis] < offset, argument.at(*at)))
                result = terms[-1][1]
                for condition, value_expression in reversed(terms[:-1]):
                    result = kernels.select(condition, value_expression, result)
            operands = tuple(local[v.index] for v in values)
            tensors[value.index] = program.kernel_call(kernels.expression(result),
                grid=tuple((size + extent - 1) // extent for size, extent in zip(matrix_shape, block)),
                in_specs=(BlockSpec(None),) * len(operands),
                out_specs=BlockSpec(block, lambda i, j: (i, j)),
                out_shape=ShapeDtypeStruct(matrix_shape, value.dtype), peer=peer)(*operands)
            continue
        matrix_shapes = {v.index: shapes[v.index] if len(shapes[v.index]) == 2 else (1, math.prod(shapes[v.index])) for v in values}
        shaped = all(local[v.index].shape == matrix_shapes[v.index] or
            (1 in matrix_shapes[v.index] and local[v.index].shape[::-1] == matrix_shapes[v.index]) for v in values)
        # ../../../design/algorithm-sources.md#typed-integer-contractions
        if operation == 'matmul' and all(len(shapes[v.index]) >= 2 for v in values):
            tensors[value.index] = batched_matmul(program, *(local[v.index] for v in values),
                *(shapes[v.index] for v in values), attributes, tile_rows=tile_rows,
                tile_k=tile_k, tile_columns=tile_columns, peer=peer, output_dtype=value.dtype)
            continue
        # ../../../design/algorithm-sources.md#shared-elementary-functions
        if operation in ('add', 'subtract', 'multiply', 'divide', 'negative', 'exp', 'tanh', 'rsqrt', 'sigmoid', 'maximum', 'minimum', 'cast', 'assign', 'where', 'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal', 'bitwise_and', 'bitwise_or', 'broadcast', 'logical_not', 'logical_and', 'logical_or', 'arcsinh', 'expm1', 'log', 'log1p', 'sqrt', 'abs', 'power', 'logaddexp', 'isfinite', 'floor_divide', 'bitwise_invert'):
            args = kernels.arguments(len(values))
            direct = shaped and len(shape) <= 2
            if not direct:
                matrix_shape = (math.prod(shape[:-1]), shape[-1]) if shape else (1, 1)
                block = (min(tile_rows, matrix_shape[0]), min(tile_columns, matrix_shape[1]))
                coordinates = logical_coordinates(kernels, shape, block)
                args = tuple(argument.reshape(shapes[operand.index]).at(*(
                    0 if size == 1 else coordinate for size, coordinate in
                    zip(shapes[operand.index], coordinates[len(shape)-len(shapes[operand.index]):])))
                    for argument, operand in zip(args, values))
            if operation in ('add', 'subtract', 'multiply', 'divide'):
                left, right = args
                if value.dtype in ('int32', 'uint32', 'int64', 'uint64') and operation != 'divide':
                    left, right = left & 0xffffffffffffffff, right & 0xffffffffffffffff
                result = {'add': lambda: left + right, 'subtract': lambda: left - right,
                          'multiply': lambda: left * right, 'divide': lambda: left / right}[operation]()
            elif operation == 'floor_divide':
                left, right = args
                result = left.floor_divide(right) if any(np.dtype(operand.dtype).kind == 'f' for operand in values) else left // right
            elif operation in ('power', 'logaddexp'):
                result = getattr(args[0], operation)(args[1])
            elif operation == 'bitwise_invert':
                result = 0xffffffffffffffff - (args[0] & 0xffffffffffffffff)
            elif operation in ('cast', 'assign', 'broadcast'):
                result = args[0].astype(value.dtype) if operation == 'cast' else args[0]
            elif operation in ('maximum', 'minimum'):
                left, right = args
                result = kernels.select(left > right if operation == 'maximum' else left < right, left, right)
            elif operation == 'where':
                result = kernels.select(*args)
            elif operation in ('equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal', 'bitwise_and', 'bitwise_or'):
                left, right = args
                result = {'equal': lambda: left.equal(right), 'not_equal': lambda: kernels.select(left.equal(right), False, True),
                          'less': lambda: left < right, 'less_equal': lambda: left <= right,
                          'greater': lambda: left > right, 'greater_equal': lambda: left >= right,
                          'bitwise_and': lambda: left & right, 'bitwise_or': lambda: left | right}[operation]()
            elif operation == 'logical_not':
                result = args[0].equal(0)
            elif operation in ('logical_and', 'logical_or'):
                left, right = (argument.equal(0).equal(False) for argument in args)
                result = left & right if operation == 'logical_and' else left | right
            elif operation == 'negative':
                result = (0 - (args[0] & 0xffffffffffffffff)) if value.dtype in ('int32', 'uint32', 'int64', 'uint64') else -1 * args[0]
            elif operation == 'sigmoid':
                result = 1 / (1 + (-1 * args[0]).exp())
            else:
                result = getattr(args[0], operation)()
            if direct:
                matrix_shape = shape if len(shape) == 2 else (1, math.prod(shape))
                operands = tuple(matrix_view(local[v.index], matrix_shapes[v.index]).broadcast_to(matrix_shape) for v in values)
                tensors[value.index] = nn._pointwise(program, kernels.expression(result),
                    operands, tile_rows, peer=peer, output_dtype=value.dtype)
            else:
                tensors[value.index] = program.kernel_call(kernels.expression(result),
                    grid=tuple((size + extent - 1) // extent for size, extent in zip(matrix_shape, block)),
                    in_specs=(BlockSpec(None),) * len(values),
                    out_specs=BlockSpec(block, lambda i, j: (i, j)),
                    out_shape=ShapeDtypeStruct(matrix_shape, value.dtype), peer=peer)(*(local[v.index] for v in values))
            continue
        raise ValueError(f'No shared lowering for tensor operation {operation}')
    return tensors
