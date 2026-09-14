import numpy as np

from .tensor import Dimension, broadcast_shape


# https://docs.jax.dev/en/latest/_autosummary/jax.ops.segment_sum.html
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






# https://docs.jax.dev/en/latest/_autosummary/jax.numpy.matmul.html
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


# ../../../design/algorithm-sources.md#kernelsdot
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


# https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
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


# https://docs.jax.dev/en/latest/pallas/grid_blockspec.html
def logical_coordinates(kernels, shape, block):
    import math
    row, column = kernels.indices()
    row = kernels.program_id(0) * block[0] + row
    column = kernels.program_id(1) * block[1] + column
    return tuple((row // math.prod(shape[axis + 1:-1])) % size
                 for axis, size in enumerate(shape[:-1])) + ((column,) if shape else ())


# https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py
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


# https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py
def take_coordinates(kernels, source_shape, index_shape, index_dtype, index_value, axis, coordinates):
    index_at = tuple(0 if size == 1 else coordinate for size, coordinate in zip(index_shape, coordinates))
    selected = index_value.at(*index_at)
    if index_dtype.startswith('int'):
        selected = kernels.select(selected < 0, selected + source_shape[axis], selected)
    return tuple(selected if i == axis else 0 if size == 1 else coordinates[i]
                 for i, size in enumerate(source_shape))




# https://mlir.llvm.org/docs/Canonicalization/#globally-applied-rules
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
        nodes.append(node)
        dependencies = numerical_operands(operation, values, attributes) if math.prod(shapes[value.index]) else ()
        live.update(operand.index for operand in dependencies)
    nodes.reverse()
    constants = {value.index: data for value, data in graph.constants.values()}
    tensors = dict(inputs)
    peers = {0: program.node if root_peer is None else root_peer}
    peers.update({region['owner']: region['peer'] for region in graph.regions.values()})
    owners = {value.index: peers[owner] for value, _, _, _, owner in graph.nodes}
    for value, operation, values, attributes, owner in nodes:
        peer = peers[owner]
        if value.index in tensors:
            continue
        # ../../../design/algorithm-sources.md#programtensor
        shape = shapes[value.index]
        storage_shape = (math.prod(shape[:-1]), shape[-1]) if shape else (1, 1)
        if not math.prod(shape):
            tensors[value.index] = program.tensor(storage_shape, dtype=value.dtype)
            continue
        if operation in ('reshape', 'stop_gradient'):
            tensors[value.index] = tensors[values[0].index]
            owners[value.index] = owners[values[0].index]
            continue
        # https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
        if operation == 'transpose' and (attributes['axes'] == tuple(range(len(shape))) or
                (len(shape) == 2 and attributes['axes'] == (1, 0))):
            source = tensors[values[0].index]
            tensors[value.index] = matrix_view(source, shapes[values[0].index]).T if attributes['axes'] == (1, 0) else source
            owners[value.index] = owners[values[0].index]
            continue
        if operation in ('constant', 'dimension'):
            tensor = program.tensor(storage_shape, dtype=value.dtype)
            data = constants[value.index] if operation == 'constant' else np.asarray(attributes['expression'].resolve(capacity), dtype=value.dtype)
            if program.node == peer:
                program.constant(tensor[0, 0], data.reshape(storage_shape))
            tensors[value.index] = tensor
            continue
        local = {}
        for operand in numerical_operands(operation, values, attributes):
            tensor = tensors[operand.index]
            sender = owners[operand.index]
            # ../../../design/algorithm-sources.md#programcopy
            if sender != peer:
                tensor = program.replicate(tensor.on(sender), peer)
            local[operand.index] = tensor
        shape = shapes[value.index]
        # ../../../design/algorithm-sources.md#kernelsadd
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
        # ../../../design/algorithm-sources.md#kernelsadd
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
        # https://docs.jax.dev/en/latest/pallas/grid_blockspec.html
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
        # ../../../design/algorithm-sources.md#kernelsdot
        if operation == 'matmul' and all(len(shapes[v.index]) >= 2 for v in values):
            tensors[value.index] = batched_matmul(program, *(local[v.index] for v in values),
                *(shapes[v.index] for v in values), attributes, tile_rows=tile_rows,
                tile_k=tile_k, tile_columns=tile_columns, peer=peer, output_dtype=value.dtype)
            continue
        # ../../../design/algorithm-sources.md#kernelsexpression
        if operation in ('add', 'subtract', 'multiply', 'divide', 'negative', 'exp', 'rsqrt', 'sigmoid', 'maximum', 'minimum', 'cast', 'assign', 'where', 'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal', 'bitwise_and', 'bitwise_or', 'broadcast', 'logical_not', 'logical_and', 'logical_or', 'bitwise_invert'):
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
