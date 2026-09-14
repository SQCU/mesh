from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _Operation:
    op: int
    arity: int
    alpha: float = 1
    beta: float = 0


matmul = _Operation(6, 2)
add = _Operation(1, 2, beta=1)
multiply = _Operation(2, 2)
tanh = _Operation(3, 1)
exp = _Operation(4, 1)
row_sum = _Operation(5, 1)
rsqrt = _Operation(7, 1)
swish = _Operation(8, 1)


# design/algorithm-sources.md#single-kernel-interface
def affine(alpha=1, beta=0):
    return _Operation(0, 1, alpha, beta)


@dataclass(frozen=True)
class MetalDispatch:
    name: str
    grid: tuple
    group: tuple = (256, 1, 1)
    argument_buffer: int = 0
    argument_offset: int = 0


@dataclass(frozen=True)
class Metal:
    source: str
    dispatches: tuple
    constants: tuple = ()


@dataclass(frozen=True)
class _Expression:
    operation: str
    operands: tuple = ()
    value: object = None

    # design/algorithm-sources.md#region-expression-fusion
    def __add__(self, other):
        return _Expression('+', (self, _literal(other)))

    # design/algorithm-sources.md#region-expression-fusion
    def __radd__(self, other):
        return _literal(other) + self

    # design/algorithm-sources.md#region-expression-fusion
    def __sub__(self, other):
        return _Expression('-', (self, _literal(other)))

    # design/algorithm-sources.md#region-expression-fusion
    def __rsub__(self, other):
        return _literal(other) - self

    # design/algorithm-sources.md#region-expression-fusion
    def __mul__(self, other):
        return _Expression('*', (self, _literal(other)))

    # design/algorithm-sources.md#region-expression-fusion
    def __rmul__(self, other):
        return _literal(other) * self

    # design/algorithm-sources.md#region-expression-fusion
    def __truediv__(self, other):
        return _Expression('/', (self, _literal(other)))

    # design/algorithm-sources.md#region-expression-fusion
    def __rtruediv__(self, other):
        return _literal(other) / self

    # design/algorithm-sources.md#logical-indexed-views
    def __floordiv__(self, other):
        return _Expression('//', (self, _literal(other)))

    # design/algorithm-sources.md#logical-indexed-views
    def __rfloordiv__(self, other):
        return _literal(other) // self

    # design/algorithm-sources.md#logical-indexed-views
    def __mod__(self, other):
        return _Expression('%', (self, _literal(other)))

    # design/algorithm-sources.md#logical-indexed-views
    def __rmod__(self, other):
        return _literal(other) % self

    # design/algorithm-sources.md#indexed-expression-lowering
    def __lt__(self, other):
        return _Expression('<', (self, _literal(other)))

    # design/algorithm-sources.md#indexed-expression-lowering
    def __le__(self, other):
        return _Expression('<=', (self, _literal(other)))

    # design/algorithm-sources.md#indexed-expression-lowering
    def __gt__(self, other):
        return _Expression('>', (self, _literal(other)))

    # design/algorithm-sources.md#indexed-expression-lowering
    def __ge__(self, other):
        return _Expression('>=', (self, _literal(other)))

    # design/algorithm-sources.md#indexed-expression-lowering
    def equal(self, other):
        return _Expression('==', (self, _literal(other)))

    # design/algorithm-sources.md#indexed-expression-lowering
    def __and__(self, other):
        return _Expression('&', (self, _literal(other)))

    # design/algorithm-sources.md#indexed-expression-lowering
    def __or__(self, other):
        return _Expression('|', (self, _literal(other)))

    # design/algorithm-sources.md#logical-indexed-views
    def reshape(self, shape):
        import operator
        if self.operation != 'input':
            raise ValueError('Logical indexed views require an input reference')
        shape = tuple(operator.index(dimension) for dimension in shape)
        if any(dimension < -1 or dimension == 0 for dimension in shape) or shape.count(-1) > 1:
            raise ValueError('Logical indexed dimensions must be positive with at most one inferred axis')
        return _Expression('reshape', (self,), shape)

    # design/algorithm-sources.md#logical-indexed-views
    def at(self, row=None, column=None, *coordinates, mask=True, other=0):
        coordinates = (() if row is None else (row,)) + (() if column is None else (column,)) + coordinates
        if self.operation == 'reshape':
            if len(coordinates) != len(self.value):
                raise ValueError('Logical index rank differs from the declared shape')
            return _Expression('logical_load', tuple(map(_literal, (*coordinates, mask, other))), (self.operands[0].value, self.value))
        if self.operation != 'input' or len(coordinates) != 2:
            raise ValueError('Indexed loads require an input reference and two coordinates')
        return _Expression('load', tuple(map(_literal, (*coordinates, mask, other))), self.value)

    # design/algorithm-sources.md#shared-contraction-lowering
    def astype(self, dtype):
        dtype = np.dtype(dtype)
        if dtype.name not in ('float16', 'float32', 'int32', 'uint32', 'int64', 'uint64', 'uint8', 'bool'):
            raise ValueError('Expression casts require a supported scalar dtype')
        return _Expression('cast', (self,), dtype.str)

    # design/algorithm-sources.md#region-expression-fusion
    def sum(self, axis=1):
        axes = (0, 1) if axis is None else (axis,) if isinstance(axis, int) else tuple(axis)
        if any(value not in (-2, -1, 0, 1) for value in axes):
            raise ValueError('Expression reduction axes refer to its two-dimensional domain')
        axes = tuple(sorted(set(value % 2 for value in axes)))
        if not axes:
            return self
        if axes == (0,):
            return self.T.sum().T
        reduced = _Expression('sum', (self,))
        return reduced.T.sum() if axes == (0, 1) else reduced

    @property
    # design/algorithm-sources.md#shared-contraction-lowering
    def T(self):
        if self.operation == 'transpose':
            return self.operands[0]
        if self.operation in ('literal', 'program_id'):
            return self
        if self.operation in ('row', 'column'):
            return _Expression('column' if self.operation == 'row' else 'row')
        if self.operation == 'dot':
            return _Expression('dot', tuple(child.T for child in self.operands[::-1]), self.value)
        if self.operation in ('+', '-', '*', '/', '<', '<=', '>', '>=', '==', '&', '|', 'select', 'rsqrt', 'exp', 'tanh', 'cast', '//', '%'):
            return _Expression(self.operation, tuple(child.T for child in self.operands), self.value)
        return _Expression('transpose', (self,))

    # design/algorithm-sources.md#region-expression-fusion
    def rsqrt(self):
        return _Expression('rsqrt', (self,))

    # design/algorithm-sources.md#region-expression-fusion
    def exp(self):
        return _Expression('exp', (self,))

    # design/algorithm-sources.md#region-expression-fusion
    def tanh(self):
        return _Expression('tanh', (self,))


# design/algorithm-sources.md#region-expression-fusion
def _literal(value):
    return value if isinstance(value, _Expression) else _Expression('literal', value=value.item() if isinstance(value, np.generic) else value)


# design/algorithm-sources.md#logical-indexed-views
def _resolve_logical(node, inputs):
    import math
    children = tuple(_resolve_logical(child, inputs) for child in node.operands)
    if node.operation == 'logical_load':
        index, shape = node.value
        source = inputs[index]
        volume = math.prod(source.shape)
        if -1 in shape:
            known = math.prod(dimension for dimension in shape if dimension != -1)
            if volume % known:
                raise ValueError('Logical indexed shape cannot infer an integral dimension')
            shape = tuple(volume // known if dimension == -1 else dimension for dimension in shape)
        if math.prod(shape) != volume:
            raise ValueError('Logical indexed shape volume differs from its bound input')
        ordinal, enabled = _literal(0), children[-2]
        for dimension, coordinate in zip(shape, children[:-2]):
            ordinal = ordinal * dimension + coordinate
            enabled = select(enabled, (coordinate >= 0) & (coordinate < dimension), False)
        return _Expression('load', (ordinal // source.shape[1], ordinal % source.shape[1], enabled, children[-1]), index)
    if node.operation == 'reshape':
        raise ValueError('Logical reshapes require indexed access')
    if node.operation in ('//', '%') and all(child.operation == 'literal' for child in children):
        left, right = (child.value for child in children)
        if not isinstance(left, int) or not isinstance(right, int):
            raise ValueError('Integer quotient and remainder require integral operands')
        dtype = _expression_dtype(_Expression(node.operation, children), inputs)
        if dtype.kind == 'u':
            left, right = left % (1 << (8*dtype.itemsize)), right % (1 << (8*dtype.itemsize))
        value = _literal(left // right if node.operation == '//' else left % right)
        return value.astype(dtype) if dtype.kind == 'u' else value
    return _Expression(node.operation, children, node.value)


# design/algorithm-sources.md#static-indexed-access-specialization
def _static_value(node, inputs, rows, columns, coordinate):
    if node.operation in ('input', 'load'):
        return None
    if node.operation == 'row':
        return rows
    if node.operation == 'column':
        return columns
    if node.operation == 'index_vector':
        return np.full(columns.shape, node.value[2], dtype=np.int64) if node.value[0] == 1 else columns + node.value[2]
    if node.operation == 'program_id':
        return np.full(rows.shape, coordinate[node.value], dtype=np.int64)
    if node.operation == 'literal':
        return np.full(rows.shape, node.value, dtype=_expression_dtype(node, inputs))
    if node.operation == 'select':
        condition = _static_value(node.operands[0], inputs, rows, columns, coordinate)
        if condition is None:
            return None
        result = np.empty(rows.shape, dtype=_expression_dtype(node, inputs))
        for branch, enabled in ((node.operands[1], condition != 0), (node.operands[2], condition == 0)):
            if np.any(enabled):
                value = _static_value(branch, inputs, rows[enabled], columns[enabled], coordinate)
                if value is None:
                    return None
                result[enabled] = value
        return result
    if node.operation not in ('cast', '+', '-', '*', '/', '//', '%', '<', '<=', '>', '>=', '==', '&', '|'):
        return None
    values = tuple(_static_value(child, inputs, rows, columns, coordinate) for child in node.operands)
    if any(value is None for value in values):
        return None
    dtype = _expression_dtype(node, inputs)
    if node.operation == 'cast':
        value = values[0]
        if dtype.kind in 'iu' and value.dtype.kind == 'f':
            bits = dtype.itemsize*8
            lower, upper = (0, 2**bits) if dtype.kind == 'u' else (-2**(bits-1), 2**(bits-1))
            if np.any(~np.isfinite(value)) or np.any(value < lower) or np.any(value >= upper):
                return None
        with np.errstate(over='ignore', invalid='ignore'):
            return value.astype(dtype)
    promoted = _expression_dtype(_Expression('+', node.operands), inputs)
    if promoted.kind == 'f':
        return None
    left, right = (value.astype(promoted) for value in values)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        if node.operation in ('/', '//', '%'):
            if np.any(right == 0) or (promoted.kind == 'i' and np.any((left == np.iinfo(promoted).min) & (right == -1))):
                return None
            quotient, remainder = np.floor_divide(left, right), np.remainder(left, right)
            if node.operation == '/':
                quotient = quotient + (((left < 0) != (right < 0)) & (remainder != 0)).astype(promoted)
            return (remainder if node.operation == '%' else quotient).astype(dtype)
        if promoted.kind == 'i' and node.operation in ('+', '-', '*'):
            exact = {'+': np.add, '-': np.subtract, '*': np.multiply}[node.operation](left.astype(object), right.astype(object))
            bounds = np.iinfo(promoted)
            if np.any(exact < bounds.min) or np.any(exact > bounds.max):
                return None
        operation = {'+': np.add, '-': np.subtract, '*': np.multiply, '<': np.less,
                     '<=': np.less_equal, '>': np.greater, '>=': np.greater_equal,
                     '==': np.equal, '&': np.bitwise_and, '|': np.bitwise_or}[node.operation]
        return operation(left, right).astype(dtype)


# design/algorithm-sources.md#static-indexed-access-specialization
def _specialize_accesses(expression, inputs, output, coordinate):
    domains, widths = {}, {}

    # design/algorithm-sources.md#static-indexed-access-specialization
    def width(node):
        if node not in widths:
            children = tuple(width(child) for child in node.operands)
            if any(child is None for child in children):
                return None
            if node.operation == 'input':
                value = inputs[node.value].shape[1]
            elif node.operation == 'column':
                value = output.shape[1]
            elif node.operation == 'index_vector':
                value = node.value[0]
            elif node.operation in ('row', 'sum'):
                value = 1
            else:
                value = max(children, default=1)
                if any(child not in (1, value) for child in children):
                    return None
            widths[node] = value
        return widths[node]

    # design/algorithm-sources.md#static-indexed-access-specialization
    def visit(node, path=(), extent=output.shape[1]):
        if node.operation == 'select':
            condition, yes, no = node.operands
            visit(condition, path, extent)
            visit(yes, path + ((condition, True),), extent)
            visit(no, path + ((condition, False),), extent)
        elif node.operation == 'load':
            row, column, mask, other = node.operands
            visit(mask, path, extent)
            selected = path + ((mask, True),)
            visit(row, selected, extent)
            visit(column, selected, extent)
            visit(other, path + ((mask, False),), extent)
            if hasattr(inputs[node.value], 'blocks'):
                domains.setdefault(node, set()).add((selected, extent))
        elif node.operation == 'sum':
            visit(node.operands[0], (), width(node.operands[0]))
        else:
            for child in node.operands:
                visit(child, path, extent)

    visit(expression)
    selected = {}
    for node, uses in domains.items():
        table = inputs[node.value]
        blocks, proven = set(), True
        for path, extent in uses:
            if extent is None:
                proven = False
                break
            for start in range(0, output.shape[0]*extent, 4096):
                ordinal = np.arange(start, min(start+4096, output.shape[0]*extent), dtype=np.int64)
                rows, columns = ordinal // extent, ordinal % extent
                for predicate, polarity in path:
                    value = _static_value(predicate, inputs, rows, columns, coordinate)
                    if value is None:
                        proven = False
                        break
                    enabled = (value != 0) if polarity else (value == 0)
                    rows, columns = rows[enabled], columns[enabled]
                    if not rows.size:
                        break
                if not proven:
                    break
                if not rows.size:
                    continue
                row = _static_value(node.operands[0], inputs, rows, columns, coordinate)
                column = _static_value(node.operands[1], inputs, rows, columns, coordinate)
                if row is None or column is None or row.dtype.kind not in 'iub' or column.dtype.kind not in 'iub':
                    proven = False
                    break
                if np.any(row < 0) or np.any(column < 0) or np.any(row >= table.shape[0]) or np.any(column >= table.shape[1]):
                    proven = False
                    break
                blocks.update((int(r), int(c)) for r, c in zip(row // table.block_shape[0], column // table.block_shape[1]))
            if not proven:
                break
        if proven:
            selected[node] = tuple(sorted(blocks))
    bound, positions, replacements = list(inputs), {}, {}

    # design/algorithm-sources.md#static-indexed-access-specialization
    def rewrite(node):
        if node in replacements:
            return replacements[node]
        children = tuple(rewrite(child) for child in node.operands)
        if node not in selected:
            return _Expression(node.operation, children, node.value)
        table = inputs[node.value]
        row, column, mask, other = children
        result = other
        for block in reversed(selected[node]):
            ref = table.blocks[block]
            view = ref.view
            identity = (view.tensor, view.extent, view.offset, view.rows, view.columns, view.row_stride, view.column_stride, ref.dtype.str)
            if identity not in positions:
                positions[identity] = len(bound)
                bound.append(ref)
            first_row, first_column = (index*size for index, size in zip(block, table.block_shape))
            local_row, local_column = row-first_row, column-first_column
            inside = (local_row >= 0) & (local_row < ref.shape[0]) & (local_column >= 0) & (local_column < ref.shape[1])
            value = _Expression('load', (local_row, local_column, _literal(True), other), positions[identity])
            result = value if len(selected[node]) == 1 else select(inside, value, result)
        result = select(mask, result, other) if selected[node] else other
        replacements[node] = result
        return result

    return rewrite(expression), tuple(bound)


# design/algorithm-sources.md#indexed-expression-lowering
def indices():
    return _Expression('row'), _Expression('column')


# design/algorithm-sources.md#explicit-index-vector-domains
def arange(length, *, tile=None):
    import operator
    length = operator.index(length)
    tile = length if tile is None else operator.index(tile)
    if length <= 0 or tile <= 0 or length > np.iinfo(np.int64).max:
        raise ValueError('Index vector length and tile must be positive within the int64 domain')
    return _Expression('index_vector', value=(length, min(tile, length), 0))


# design/algorithm-sources.md#dynamic-indexed-expression-lowering
def program_id(axis):
    return _Expression('program_id', value=axis)


# design/algorithm-sources.md#indexed-expression-lowering
def select(mask, yes, no):
    return _Expression('select', tuple(map(_literal, (mask, yes, no))))


# design/algorithm-sources.md#region-expression-fusion
def arguments(count):
    return tuple(_Expression('input', value=index) for index in range(count))


# design/algorithm-sources.md#region-expression-fusion
def expression(value, *outputs):
    return _ExpressionKernel(tuple(map(_literal, (value, *outputs))))


@dataclass(frozen=True)
class _ExpressionKernel:
    values: tuple

    # design/algorithm-sources.md#dynamic-indexed-expression-lowering
    def bind_grid(self, program, grid, input_specs, output_specs):
        import itertools
        if any(value.operation == 'indexed_add' for value in self.values):
            for value, spec in zip(self.values, output_specs):
                if value.operation == 'indexed_add':
                    resolved = _resolve_logical(value, tuple(source._tensor for source in input_specs))
                    _lower_indexed_add(program, resolved, grid, input_specs, spec)
                else:
                    _ExpressionKernel((value,)).bind_grid(program, grid, input_specs, (spec,))
            return
        if any(_requires_regions(value) for value in self.values):
            _lower_region_expressions(program, self.values, grid, input_specs, output_specs)
            return
        for coordinate in itertools.product(*(range(size) for size in grid)):
            self.bind(program, tuple(spec.resolve(coordinate) for spec in input_specs),
                tuple(spec.resolve(coordinate) for spec in output_specs), coordinate)

    # design/algorithm-sources.md#region-expression-fusion
    def bind(self, program, inputs, outputs, coordinate=()):
        from . import check
        from ._native import View
        import ctypes as C
        if len(outputs) != len(self.values):
            raise ValueError('Each expression requires an output region')
        if any(ref.dtype.name not in ('float16', 'float32', 'int32', 'uint32', 'int64', 'uint64', 'uint8', 'bool') for ref in (*inputs, *outputs)):
            raise ValueError('Expression regions require supported real, integer or boolean scalars')
        for value, output in zip(self.values, outputs):
            value = _resolve_logical(value, inputs)
            value, specialized_inputs = _specialize_accesses(value, inputs, output, coordinate)
            used = {}

            # design/algorithm-sources.md#indexed-expression-lowering
            def remap(node):
                index = node.value
                if node.operation == 'program_id':
                    return _literal(coordinate[index])
                if node.operation in ('input', 'load'):
                    index = used.setdefault(index, len(used))
                return _Expression(node.operation, tuple(remap(child) for child in node.operands), index)

            expression = remap(value)
            reads = tuple(specialized_inputs[index] for index in used)
            dynamic = {}
            dynamic_inputs = {index for index, source in enumerate(reads)
                if hasattr(source, 'blocks') and not all((ref.view.tensor, ref.view.extent)
                    in program._constant_extents for ref in source.blocks.values())}
            accesses = _indexed_access_paths(expression, reads, dynamic_inputs)
            for node, paths in accesses.items():
                table = reads[node.value]
                row, column, mask, _ = node.operands
                ordinal = _Expression('block_ordinal', (row, column), (*table.block_shape, table.grid[1]))
                enabled = _literal(False)
                for path in paths:
                    condition = _literal(True)
                    for predicate, polarity in path:
                        condition = select(condition, predicate if polarity else predicate.equal(False), False)
                    enabled = select(enabled, True, condition)
                selector_value = select(enabled, ordinal, 0xffffffff)
                selector_width = output.shape[1]

                # design/algorithm-sources.md#dynamic-indexed-expression-lowering
                def selector_shape(part):
                    nonlocal selector_width
                    if part.operation == 'input':
                        selector_width = max(selector_width, reads[part.value].shape[1])
                    elif part.operation == 'index_vector':
                        selector_width = max(selector_width, part.value[0])
                    for child in part.operands:
                        selector_shape(child)

                selector_shape(selector_value)
                selected = program.tensor((output.shape[0], selector_width), dtype=np.uint32)[0, 0]
                _ExpressionKernel((selector_value,)).bind(program, reads, (selected,))
                dynamic[node] = (selected, node.value)

            flattened = tuple(ref for source in reads for ref in
                (tuple(ref for _, ref in sorted(source.blocks.items())) if hasattr(source, 'blocks') else (source,)))
            function = program.native.algebra_trace_count(program.handle)
            check(program.native.algebra_source(program.handle,
                self.source(reads, output, False, expression).encode(),
                self.source(reads, output, True, expression).encode(),
                (View * len(flattened))(*(ref.view for ref in flattened)), len(flattened), output.view))
            offsets = []
            for source in reads:
                offsets.append((offsets[-1][0] + offsets[-1][1] if offsets else 0,
                    len(source.blocks) if hasattr(source, 'blocks') else 1))
            for selected, index in dynamic.values():
                first, count = offsets[index]
                check(program.native.algebra_indexed(program.handle, function, selected.view,
                    (C.c_size_t * count)(*range(first, first + count)), count))

    # design/algorithm-sources.md#region-expression-fusion
    def source(self, inputs, output, metal, expression):
        widths, reductions = {}, []
        physical, pointers = [], {}
        for index, ref in enumerate(inputs):
            refs = tuple(ref for _, ref in sorted(ref.blocks.items())) if hasattr(ref, 'blocks') else (ref,)
            pointers[index] = tuple(range(len(physical), len(physical) + len(refs)))
            physical.extend(refs)

        # design/algorithm-sources.md#shared-contraction-lowering
        def integral(node):
            if node.operation == 'input':
                return inputs[node.value].dtype.kind != 'f'
            if node.operation == 'load':
                return inputs[node.value].dtype.kind != 'f' and integral(node.operands[3])
            if node.operation == 'select':
                return all(integral(child) for child in node.operands[1:])
            if node.operation == 'cast':
                return np.dtype(node.value).kind != 'f'
            if node.operation == 'literal':
                return isinstance(node.value, (int, bool))
            if node.operation in ('<', '<=', '>', '>=', '==', 'row', 'column', 'block_ordinal'):
                return True
            if node.operation in ('exp', 'rsqrt', 'tanh'):
                return False
            if node.operation == 'sum':
                return output.dtype.kind in 'iub'
            return all(integral(child) for child in node.operands)

        # design/algorithm-sources.md#region-expression-fusion
        def visit(node):
            if node in widths:
                return widths[node]
            sizes = tuple(visit(child) for child in node.operands)
            if node.operation in ('//', '%') and not all(integral(child) for child in node.operands):
                raise ValueError('Integer quotient and remainder require integral operands')
            if node.operation == 'input':
                ref = inputs[node.value]
                if ref.shape[0] not in (1, output.shape[0]):
                    raise ValueError('Expression input rows must broadcast to the output')
                width = ref.shape[1]
            elif node.operation == 'column':
                width = output.shape[1]
            elif node.operation == 'index_vector':
                width = node.value[0]
            elif node.operation == 'row':
                width = 1
            elif node.operation == 'sum':
                reductions.append(node)
                width = 1
            else:
                width = max(sizes, default=1)
                if any(size not in (1, width) for size in sizes):
                    raise ValueError('Expression columns must broadcast')
            widths[node] = width
            return width

        width = visit(expression)
        if width not in (1, output.shape[1]):
            raise ValueError('Expression columns do not match the output')
        names = {node: f's{index}' for index, node in enumerate(reductions)}

        # design/algorithm-sources.md#shared-scalar-load-emission
        def emit(node, column):
            # design/algorithm-sources.md#shared-scalar-load-emission
            def resolve(part, args):
                if part.operation == 'row':
                    return '((long)r)' if metal else '((int64_t)r)'
                if part.operation == 'column':
                    return f'((long)({column}))' if metal else f'((int64_t)({column}))'
                if part.operation == 'index_vector':
                    return f'((int64_t)({0 if part.value[0] == 1 else column})+{part.value[2]}ll)'
                if part.operation == 'sum':
                    return f'(({"long" if metal else "int64_t"}){names[part]})' if output.dtype.kind in 'ib' else names[part]
                ref = inputs[part.value]
                if part.operation == 'load':
                    return _indexed_load_expression(ref, pointers[part.value][0], layouts.get(part.value), args, metal)
                row_stride = ref.view.row_stride if ref.shape[0] != 1 else 0
                column_stride = ref.view.column_stride if ref.shape[1] != 1 else 0
                value = f'p{pointers[part.value][0]}[r*{row_stride}+({column})*{column_stride}]'
                return f'((float)({value}))' if ref.dtype.kind == 'f' else value

            return _emit_scalar_expression(node, inputs, metal, resolve)

        lines = ['#include <metal_stdlib>\nusing namespace metal;' if metal else '#include <stdint.h>\n#include <stdbool.h>\n#include <math.h>']
        layouts = {}
        for index, ref in enumerate(inputs):
            if not hasattr(ref, 'blocks'):
                continue
            refs = tuple(ref for _, ref in sorted(ref.blocks.items()))
            scalar = {'f2': 'half' if metal else '_Float16', 'f4': 'float',
                      'i4': 'int' if metal else 'int32_t', 'u4': 'uint' if metal else 'uint32_t',
                      'i8': 'long' if metal else 'int64_t', 'u8': 'ulong' if metal else 'uint64_t',
                      'u1': 'uchar' if metal else 'uint8_t', 'b1': 'bool'}[ref.dtype.kind + str(ref.dtype.itemsize)]
            strides = []
            for name, attr in (('rs', 'row_stride'), ('cs', 'column_stride')):
                values = tuple(getattr(r.view, attr) for r in refs)
                if len(set(values)) == 1:
                    strides.append(str(values[0]))
                else:
                    lines.append(f'{"constant ulong" if metal else "static const uint64_t"} {name}{index}[]={{'+','.join(map(str, values))+'};')
                    strides.append(f'{name}{index}[CANDIDATE]')
            layouts[index] = scalar, strides
        lines.append('kernel void mesh_expression(device const ulong *buffers [[buffer(0)]], uint r [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {' if metal else 'void mesh_expression(const uintptr_t *buffers) {')
        candidates = {pointer for index, ref in enumerate(inputs) if hasattr(ref, 'blocks') for pointer in pointers[index]}
        for index, ref in enumerate((*physical, output)):
            if index in candidates:
                continue
            scalar = ({'f2': 'half' if metal else '_Float16', 'f4': 'float', 'i4': 'int' if metal else 'int32_t',
                       'u4': 'uint' if metal else 'uint32_t', 'i8': 'long' if metal else 'int64_t',
                       'u8': 'ulong' if metal else 'uint64_t', 'u1': 'uchar' if metal else 'uint8_t', 'b1': 'bool'}[ref.dtype.kind + str(ref.dtype.itemsize)])
            qualifier = ('device ' if metal else '') + ('const ' if index < len(physical) else '')
            lines.append(f'{qualifier}{scalar} *p{index}=({qualifier}{scalar} *)buffers[{index}];')
        if not metal:
            lines.append(f'for(uint64_t r=0;r<{output.shape[0]};r++) {{')
        for node in reductions:
            name, child = names[node], node.operands[0]
            unsigned = output.dtype.kind == 'u' or (output.dtype.kind in 'ib' and integral(child))
            accumulator = ('ulong' if metal else 'uint64_t') if unsigned else ('long' if metal else 'int64_t') if output.dtype.kind in 'ib' else 'float'
            lines.append(f'{accumulator} {name}=0;')
            lines.append(f'for({"uint" if metal else "uint64_t"} k={"lane" if metal else "0"};k<{widths[child]};k+={32 if metal else 1}) {name}+={emit(child, "k")};')
            if metal and output.dtype.kind in 'iub':
                lines.append(f'uint {name}_lo=simd_sum(uint(ulong({name})&65535ul)), {name}_mid=simd_sum(uint((ulong({name})>>16)&65535ul)), {name}_hi=simd_sum(uint(ulong({name})>>32));')
                lines.append(f'{name}_mid+={name}_lo>>16; {name}_hi+={name}_mid>>16; {name}=(ulong({name}_hi)<<32)|(ulong({name}_mid&65535u)<<16)|ulong({name}_lo&65535u);')
            elif metal:
                lines.append(f'{name}=simd_sum({name});')
        lines.append(f'for({"uint" if metal else "uint64_t"} c={"lane" if metal else "0"};c<{output.shape[1]};c+={32 if metal else 1}) p{len(physical)}[r*{output.view.row_stride}+c*{output.view.column_stride}]={emit(expression, "c")};')
        lines.append('}' if metal else '}}')
        return '\n'.join(lines)


# design/algorithm-sources.md#bounded-indexed-segment-loads
def _indexed_access_paths(expression, inputs, dynamic_inputs):
    accesses = {}

    # design/algorithm-sources.md#bounded-indexed-segment-loads
    def visit(node, path=()):
        if node.operation == 'input' and hasattr(inputs[node.value], 'blocks'):
            raise ValueError('Whole-tensor inputs require indexed loads')
        if node.operation == 'select':
            condition, yes, no = node.operands
            visit(condition, path)
            visit(yes, path + ((condition, True),))
            visit(no, path + ((condition, False),))
        elif node.operation == 'load':
            row, column, mask, other = node.operands
            visit(mask, path)
            selected = path + ((mask, True),)
            visit(row, selected)
            visit(column, selected)
            visit(other, path + ((mask, False),))
            if node.value in dynamic_inputs:
                paths = accesses.setdefault(node, [])
                if not any(set(previous) <= set(selected) for previous in paths):
                    paths[:] = [previous for previous in paths if not set(selected) <= set(previous)]
                    paths.append(selected)
        else:
            for child in node.operands:
                visit(child, () if node.operation == 'sum' else path)

    visit(expression)
    return accesses


# design/algorithm-sources.md#shared-scalar-load-emission
def _indexed_load_expression(ref, pointer, layout, args, metal):
    row, column, mask, other = args
    if hasattr(ref, 'blocks'):
        block = f'(({row})/{ref.block_shape[0]}*{ref.grid[1]}+({column})/{ref.block_shape[1]})'
        scalar, strides = layout
        address = f'(({"device " if metal else ""}const {scalar} *)buffers[{pointer}+{block}])'
        value = f'{address}[(({row})%{ref.block_shape[0]})*{strides[0]}+(({column})%{ref.block_shape[1]})*{strides[1]}]'.replace('CANDIDATE', block)
    else:
        value = f'p{pointer}[({row})*{ref.view.row_stride}+({column})*{ref.view.column_stride}]'
    value = f'((float)({value}))' if ref.dtype.kind == 'f' else value
    return f'(({mask})?({value}):({other}))'


# design/algorithm-sources.md#shared-scalar-load-emission
def _emit_scalar_expression(node, inputs, metal, resolve):
    if node.operation in ('input', 'row', 'column', 'sum', 'index_vector'):
        return resolve(node, ())
    args = tuple(_emit_scalar_expression(child, inputs, metal, resolve) for child in node.operands)
    if node.operation == 'load':
        return resolve(node, args)
    if node.operation == 'block_ordinal':
        row, column = args
        rows, columns, grid_columns = node.value
        return f'(({row})/{rows}*{grid_columns}+({column})/{columns})'
    return _scalar_expression(node, args, metal,
        _expression_dtype(node, inputs) if node.operation in ('//', '%') else None)


# design/algorithm-sources.md#logical-indexed-views
def _expression_dtype(node, inputs):
    if node.operation == 'input':
        return inputs[node.value].dtype
    if node.operation == 'cast':
        return np.dtype(node.value)
    if node.operation in ('dot', 'exp', 'rsqrt', 'tanh'):
        return np.dtype('float32')
    if node.operation in ('<', '<=', '>', '>=', '=='):
        return np.dtype('bool')
    if node.operation in ('row', 'column', 'program_id', 'index_vector'):
        return np.dtype('int64')
    if node.operation == 'literal':
        return np.dtype('int32' if isinstance(node.value, bool) else 'float32' if isinstance(node.value, float) else 'uint64' if node.value > 2**63-1 else 'int64')
    if node.operation == 'sum':
        child = _expression_dtype(node.operands[0], inputs)
        return np.dtype('float32' if child.kind == 'f' else 'uint64' if child.kind == 'u' else 'int64')
    if node.operation == 'load':
        types = (inputs[node.value].dtype, _expression_dtype(node.operands[3], inputs))
    else:
        types = tuple(_expression_dtype(child, inputs) for child in (node.operands[1:] if node.operation == 'select' else node.operands))
    if any(dtype.kind == 'f' for dtype in types):
        return np.dtype('float32')
    bits = max(max(32, dtype.itemsize*8) for dtype in types)
    unsigned = any(dtype.kind == 'u' and dtype.itemsize*8 == bits for dtype in types)
    return np.dtype(('uint' if unsigned else 'int') + str(bits))


# design/algorithm-sources.md#fused-indexed-update-values
def _scalar_expression(node, args, metal, dtype=None):
    if node.operation == 'cast':
        dtype = np.dtype(node.value)
        scalar = {'f2': 'half' if metal else '_Float16', 'f4': 'float', 'i4': 'int32_t',
                  'u4': 'uint32_t', 'i8': 'int64_t', 'u8': 'uint64_t', 'u1': 'uint8_t', 'b1': 'bool'}[dtype.kind + str(dtype.itemsize)]
        value = f'(({scalar})({args[0]}))'
        return f'((float)({value}))' if dtype.kind == 'f' else value
    if node.operation == 'literal':
        if isinstance(node.value, bool):
            return '1' if node.value else '0'
        if isinstance(node.value, int):
            if node.value == -(1 << 63):
                return '(-9223372036854775807ll-1ll)'
            return str(node.value) + ('ull' if node.value > 2**63 - 1 else 'll')
        return repr(float(node.value)) + 'f'
    if node.operation == 'select':
        return f'(({args[0]})?({args[1]}):({args[2]}))'
    if node.operation in ('//', '%'):
        scalar = ('uint' if dtype.kind == 'u' else 'int') + str(dtype.itemsize*8) + '_t'
        left, right = (f'(({scalar})({arg}))' for arg in args)
        if dtype.kind == 'u':
            return f'(({left}){"/" if node.operation == "//" else "%"}({right}))'
        remainder = f'(({left})%({right}))'
        correction = f'(({remainder}!=0)&&((({left})<0)!=(({right})<0)))'
        return f'((({left})/({right}))-{correction})' if node.operation == '//' else f'({remainder}+({correction}?({right}):0))'
    if node.operation in ('+', '-', '*', '/', '<', '<=', '>', '>=', '==', '&', '|'):
        return f'({args[0]}{node.operation}{args[1]})'
    if node.operation == 'rsqrt':
        return f'rsqrt({args[0]})' if metal else f'(1.0f/sqrtf({args[0]}))'
    return f'{node.operation}{"" if metal else "f"}({args[0]})'


# design/algorithm-sources.md#segmented-indexed-add
def indexed_add(base, destinations, updates, *, mask=True):
    return _Expression('indexed_add', tuple(map(_literal, (base, destinations, updates, mask))))


# design/algorithm-sources.md#segmented-indexed-add
def _compiled_region(program, inputs, output, body, dynamic_first=None):
    import ctypes as C
    from . import check
    from ._native import View
    sources = []
    for metal in (False, True):
        scalar = {'f2': 'half' if metal else '_Float16', 'f4': 'float', 'i4': 'int32_t',
                  'u4': 'uint32_t', 'i8': 'int64_t', 'u8': 'uint64_t', 'u1': 'uint8_t', 'b1': 'bool'}
        lines = ['#include <metal_stdlib>\nusing namespace metal;\ntypedef uint uint32_t; typedef ulong uint64_t; typedef long int64_t; typedef int int32_t; typedef uchar uint8_t;' if metal else
                 '#include <stdint.h>\n#include <stdbool.h>\n#include <math.h>']
        emitted = body(metal)
        preamble, statements = emitted if isinstance(emitted, tuple) else ('', emitted)
        lines.append(preamble)
        lines.append('#define PREFIX(x) simd_prefix_exclusive_sum(x)\n#define SUM(x) simd_sum(x)\n#define BARRIER threadgroup_barrier(mem_flags::mem_device)' if metal else
                     '#define PREFIX(x) 0u\n#define SUM(x) (x)\n#define BARRIER ((void)0)')
        lines.append('kernel void mesh_expression(device const ulong *buffers [[buffer(0)]], uint r [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {' if metal else
                     f'void mesh_expression(const uintptr_t *buffers) {{ const uint32_t lane=0; for(uint64_t r=0;r<{output.shape[0]};r++) {{')
        lines.append(f'const uint32_t lanes={32 if metal else 1};')
        for index, ref in enumerate((*inputs, output)):
            if dynamic_first is not None and dynamic_first <= index < len(inputs):
                continue
            qualifier = ('device ' if metal else '') + ('const ' if index < len(inputs) else '')
            dtype = scalar[ref.dtype.kind + str(ref.dtype.itemsize)]
            lines.append(f'{qualifier}{dtype} *p{index}=({qualifier}{dtype} *)buffers[{index}];')
        lines.append(statements)
        lines.append('}' if metal else '}}')
        sources.append('\n'.join(lines))
    function = program.native.algebra_trace_count(program.handle)
    check(program.native.algebra_source(program.handle, *(source.encode() for source in sources),
        (View * len(inputs))(*(ref.view for ref in inputs)), len(inputs), output.view))
    return function


# design/algorithm-sources.md#segmented-indexed-add
def _group_ordinals(program, keys, source_block_rows, ordinal_origin=0, candidate_origin=0):
    size = sum(ref.shape[0] * ref.shape[1] for ref in keys)
    grouped = program.tensor((1, 8 * size + 1), dtype=np.uint32)[0, 0]

    # design/algorithm-sources.md#segmented-indexed-add
    def body(metal):
        out = f'p{len(keys)}'
        lines, origin = [], 0
        for index, ref in enumerate(keys):
            count = ref.shape[0] * ref.shape[1]
            lines.append(f'for(uint32_t i=lane;i<{count};i+=lanes) {{ {out}[{origin}+i]=p{index}[(i/{ref.shape[1]})*{ref.view.row_stride}+(i%{ref.shape[1]})*{ref.view.column_stride}]; {out}[{size+origin}+i]={ordinal_origin+origin}+i; }}')
            origin += count
        lines.append('BARRIER;')
        lines.append(f'for(uint32_t i=lane;i<{size};i+=lanes) {{ {out}[{4*size}+i]=0xffffffffu; {out}[{5*size}+2*i]=0; {out}[{5*size}+2*i+1]=0; }}')
        lines.append(f'''for(uint32_t shift=0;shift<32;shift+=4) {{
          uint32_t from=((shift/4)%2)*{2*size},to={2*size}-from;
          uint32_t counts[16]={{0}},offsets[16],offset=0;
          for(uint32_t i=lane;i<{size};i+=lanes)counts[({out}[from+i]>>shift)&15]++;
          for(uint32_t b=0;b<16;b++) {{ offsets[b]=offset; offset+=SUM(counts[b]); }}
          for(uint32_t tile=0;tile<{size};tile+=lanes) {{
            uint32_t i=tile+lane,valid=i<{size};
            uint32_t key=valid?{out}[from+i]:0,ordinal=valid?{out}[from+{size}+i]:0;
            for(uint32_t b=0;b<16;b++) {{
              uint32_t match=valid && ((key>>shift)&15)==b;
              uint32_t rank=PREFIX(match),total=SUM(match);
              if(match) {{ {out}[to+offsets[b]+rank]=key; {out}[to+{size}+offsets[b]+rank]=ordinal; }}
              offsets[b]+=total;
            }}
          }}
          BARRIER;
        }}
        uint32_t segments=0;
        for(uint32_t tile=0;tile<{size};tile+=lanes) {{
          uint32_t i=tile+lane,key=i<{size}?{out}[i]:0xffffffffu;
          uint32_t head=key!=0xffffffffu && (!i || key!={out}[i-1]);
          uint32_t slot=segments+PREFIX(head),total=SUM(head);
          if(head) {{ {out}[{4*size}+slot]=key; {out}[{5*size}+2*slot]=i; }}
          uint32_t inclusive=slot+head;
          if(key!=0xffffffffu && (i+1=={size} || key!={out}[i+1])){out}[{5*size}+2*(inclusive-1)+1]=i+1;
          if(i<{size}){out}[{7*size}+i]=key==0xffffffffu?0xffffffffu:{out}[{size}+i]/{source_block_rows}-{candidate_origin};
          segments+=total;
        }}
        if(!lane){out}[{8*size}]=segments;''')
        return '\n'.join(lines)

    _compiled_region(program, tuple(keys), grouped, body)
    return grouped, size


# design/algorithm-sources.md#bounded-indexed-segment-loads
def _segment_expression(node, operands, column, direct):
    row = _Expression('load', (_literal(0), _Expression('row'), _literal(True), _literal(0)), len(operands))
    feature = _Expression('column') + column
    if node.operation == 'row':
        return row
    if node.operation == 'index_vector':
        return _literal(node.value[2]) if node.value[0] == 1 else feature + node.value[2]
    if node.operation == 'column':
        return feature
    if node.operation == 'input':
        index, origin, shape = direct[node.value]
        return _Expression('load', (row - origin[0] if shape[0] != 1 else _literal(0),
            feature - origin[1] if shape[1] != 1 else _literal(0), _literal(True), _literal(0)), index)
    return _Expression(node.operation, tuple(_segment_expression(child, operands, column, direct) for child in node.operands), node.value)


# design/algorithm-sources.md#segment-selector-common-subexpressions
def _segment_selector_key(program, expression, inputs, domain):
    used = set()

    # design/algorithm-sources.md#segment-selector-common-subexpressions
    def visit(node):
        if node.operation in ('input', 'load'):
            used.add(node.value)
        for child in node.operands:
            visit(child)

    # design/algorithm-sources.md#segment-selector-common-subexpressions
    def identity(ref):
        return (ref.dtype.str, tuple(getattr(ref.view, name) for name, _ in ref.view._fields_),
                (ref.view.tensor, ref.view.extent) in program._constant_extents)

    visit(expression)
    bindings = []
    for index in sorted(used):
        source = inputs[index]
        layout = (source.shape, source.block_shape, source.grid,
                  tuple((coordinate, identity(ref)) for coordinate, ref in sorted(source.blocks.items()))) if hasattr(source, 'blocks') else identity(source)
        bindings.append((index, layout))
    return expression, tuple(bindings), tuple(identity(ref) for ref in domain)


# design/algorithm-sources.md#bounded-indexed-segment-loads
def _bind_segment_expression(program, expression, operands, ordinals, bounds, flat_bounds, direct_selector,
                             count, width, target, active_count, segment, reduction, cache=None):
    import ctypes as C
    from . import check
    if count * width > 0xffffffff:
        raise ValueError('Flattened segment selector bounds exceed the uint32 domain')
    inputs = (*operands, ordinals)
    cache = {} if cache is None else cache
    used = set()

    # design/algorithm-sources.md#bounded-indexed-segment-loads
    def dependencies(node):
        if node.operation in ('load', 'input'):
            used.add(node.value)
        for child in node.operands:
            dependencies(child)

    dependencies(expression)
    dynamic = {index for index in used if hasattr(inputs[index], 'blocks') and
        not all((ref.view.tensor, ref.view.extent) in program._constant_extents for ref in inputs[index].blocks.values())}
    selectors = []
    for node, paths in _indexed_access_paths(expression, inputs, dynamic).items():
        table = inputs[node.value]
        ordinal = _Expression('block_ordinal', node.operands[:2], (*table.block_shape, table.grid[1]))
        enabled = _literal(False)
        for path in paths:
            condition = _literal(True)
            for predicate, polarity in path:
                condition = select(condition, predicate if polarity else predicate.equal(False), False)
            enabled = select(enabled, True, condition)
        selector_value = select(enabled, ordinal, 0xffffffff)
        selector_key = (count, width, segment, _segment_selector_key(program, selector_value, inputs,
            (ordinals, bounds, flat_bounds, direct_selector, active_count)))
        if selector_key not in cache:
            selected = program.tensor((1, count * width), dtype=np.uint32)[0, 0]
            _bind_segment_expression(program, selector_value, operands,
                ordinals, bounds, flat_bounds, direct_selector, count, width, selected, active_count, segment, False, cache)
            cache[selector_key] = selected
        selectors.append((cache[selector_key], node.value))
    physical, pointers = [ordinals, bounds], {len(operands): (0,)}
    for index in sorted(used - {len(operands)}):
        source = inputs[index]
        refs = tuple(ref for _, ref in sorted(source.blocks.items())) if hasattr(source, 'blocks') else (source,)
        pointers[index] = tuple(range(len(physical), len(physical) + len(refs)))
        physical.extend(refs)

    # design/algorithm-sources.md#bounded-indexed-segment-loads
    def body(metal):
        declarations, locals, layouts = [], [], {}
        for index in sorted(used - {len(operands)}):
            source = inputs[index]
            refs = tuple(ref for _, ref in sorted(source.blocks.items())) if hasattr(source, 'blocks') else (source,)
            scalar = {'f2': 'half' if metal else '_Float16', 'f4': 'float', 'i4': 'int32_t',
                      'u4': 'uint32_t', 'i8': 'int64_t', 'u8': 'uint64_t', 'u1': 'uint8_t', 'b1': 'bool'}[source.dtype.kind + str(source.dtype.itemsize)]
            if not hasattr(source, 'blocks'):
                locals.append(f'{"device " if metal else ""}const {scalar} *p{pointers[index][0]}=({"device " if metal else ""}const {scalar} *)buffers[{pointers[index][0]}];')
                continue
            strides = []
            for name, attribute in (('rs', 'row_stride'), ('cs', 'column_stride')):
                values = tuple(getattr(ref.view, attribute) for ref in refs)
                if len(set(values)) == 1:
                    strides.append(str(values[0]))
                else:
                    declarations.append(f'{"constant" if metal else "static const"} uint64_t {name}{index}[]={{'+','.join(map(str, values))+'};')
                    strides.append(f'{name}{index}[CANDIDATE]')
            layouts[index] = scalar, strides

        # design/algorithm-sources.md#bounded-indexed-segment-loads
        def resolve(node, args):
            if node.operation == 'row':
                return '((int64_t)k)'
            if node.operation == 'column':
                return '((int64_t)c)'
            return _indexed_load_expression(inputs[node.value], pointers[node.value][0], layouts.get(node.value), args, metal)

        value = _emit_scalar_expression(expression, inputs, metal, resolve)
        output = f'p{len(physical)}'
        low, high = 'p1[0]', f'p1[{bounds.view.column_stride}]'
        if reduction:
            accumulator = 'float' if target.dtype.kind == 'f' else 'uint64_t'
            statements = f'for(uint32_t c=lane;c<{width};c+=lanes) {{ {accumulator} total=0; for(uint32_t k={low};k<{high};k++) total+={value}; {output}[c*{target.view.column_stride}]=total; }}'
        else:
            statements = f'for(uint64_t t=(uint64_t){low}*{width}+lane;t<(uint64_t){high}*{width};t+=lanes) {{ uint64_t k=t/{width},c=t%{width}; {output}[t*{target.view.column_stride}]={value}; }}'
        return '\n'.join(declarations), '\n'.join((*locals, statements))

    function = _compiled_region(program, tuple(physical), target, body, 2)
    for selected, index in selectors:
        positions = pointers[index]
        check(program.native.algebra_indexed_range(program.handle, function, selected.view, flat_bounds.view,
            (C.c_size_t * len(positions))(*positions), len(positions)))
    if reduction:
        check(program.native.algebra_active(program.handle, function, active_count.view, segment))
    else:
        for index in sorted(used - {len(operands)}):
            ref = inputs[index]
            if not hasattr(ref, 'blocks') and (ref.view.tensor, ref.view.extent) not in program._constant_extents:
                positions = pointers[index]
                check(program.native.algebra_indexed_range(program.handle, function, direct_selector.view, bounds.view,
                    (C.c_size_t * len(positions))(*positions), len(positions)))
    return function


# design/algorithm-sources.md#segmented-indexed-add
def _lower_indexed_add(program, expression, grid, input_specs, output_spec):
    import itertools
    import math
    from . import check
    if any(value.operation != 'input' for value in expression.operands[:2]):
        raise ValueError('Indexed addition takes base and destination references')
    operands = tuple(spec._tensor for spec in input_specs)
    base, destinations = (operands[value.value] for value in expression.operands[:2])
    update_value, mask = expression.operands[2:]
    output = output_spec._tensor
    size, features = destinations.shape[0], base.shape[1]
    if base.shape != output.shape or destinations.shape[1] != 1:
        raise ValueError('Indexed addition requires U×1 destinations and D×F base/output')
    if output.dtype != base.dtype or base.dtype.kind not in 'fiu':
        raise ValueError('Indexed addition requires matching real or integer base/output dtypes')
    if destinations.dtype.kind not in 'iu' or max(size, base.shape[0]) >= 0xffffffff:
        raise ValueError('Indexed addition requires integer destinations within the uint32 domain')
    # design/algorithm-sources.md#bounded-indexed-validity
    def runtime_mask(node):
        if node.operation == 'input' and any((ref.view.tensor, ref.view.extent) not in program._constant_extents
                                            for ref in operands[node.value].blocks.values()):
            return True
        return node.operation == 'load' or (node.operation == 'index_vector' and node.value[0] != 1) or any(runtime_mask(child) for child in node.operands)

    late_mask = runtime_mask(mask)
    routing_mask = _literal(True) if late_mask else mask
    value_inputs, indexed_inputs = set(), set()

    # design/algorithm-sources.md#fused-indexed-update-values
    def value_dependencies(node):
        if node.operation in ('//', '%') and any(_expression_dtype(child, operands).kind not in 'iub' for child in node.operands):
            raise ValueError('Integer quotient and remainder require integral operands')
        if node.operation == 'index_vector':
            if node.value[0] not in (1, features):
                raise ValueError('Indexed update vectors must broadcast to the feature width')
        elif node.operation == 'input':
            value_inputs.add(node.value)
        elif node.operation == 'load':
            indexed_inputs.add(node.value)
        elif node.operation not in ('literal', 'row', 'column', '+', '-', '*', '/', '<', '<=', '>', '>=', '==', '&', '|', 'select', 'rsqrt', 'exp', 'tanh', 'cast', '//', '%'):
            raise ValueError('Indexed update values require pointwise expressions; reduce or index their producer explicitly')
        for child in node.operands:
            value_dependencies(child)

    value_dependencies(update_value)
    update_inputs = set(value_inputs)
    if late_mask:
        value_dependencies(mask)
        update_value = select(mask, update_value, 0)
    value_inputs = tuple(sorted(value_inputs))
    for index in value_inputs:
        tensor = operands[index]
        if tensor.shape[0] not in (1, size) or tensor.shape[1] not in (1, features):
            raise ValueError('Indexed update operands must broadcast to U×F')
        allowed = ('fiu' if base.dtype.kind == 'f' else 'iu') if index in update_inputs else 'fiub'
        if tensor.dtype.kind not in allowed:
            raise ValueError('Indexed update operand dtype is incompatible with accumulation')
    for index in indexed_inputs:
        if operands[index].dtype.kind not in 'fiub':
            raise ValueError('Indexed update loads require scalar numerical inputs')
    dynamic_indexed = any(any((ref.view.tensor, ref.view.extent) not in program._constant_extents
                              for ref in operands[index].blocks.values()) for index in indexed_inputs)
    key_inputs = set()

    # design/algorithm-sources.md#segmented-indexed-add
    def key_dependencies(node):
        if node.operation == 'input':
            key_inputs.add(node.value)
        for child in node.operands:
            key_dependencies(child)

    key_dependencies(expression.operands[1])
    key_dependencies(routing_mask)
    chunk_rows = destinations.block_shape[0]
    for index in value_inputs:
        tensor = operands[index]
        if tensor.grid[0] > 1:
            chunk_rows = math.gcd(chunk_rows, tensor.block_shape[0])
    for index in key_inputs:
        tensor = operands[index]
        if tensor.shape not in ((size, 1), (1, 1)):
            raise ValueError('Destination and validity expressions require scalar routing rows')
        if tensor.grid[0] > 1:
            chunk_rows = math.gcd(chunk_rows, tensor.block_shape[0])
    destination = expression.operands[1]
    normalized = select(destination < 0, destination + base.shape[0], destination)
    key_expression = select(routing_mask & (normalized >= 0) & (normalized < base.shape[0]), normalized, 0xffffffff)
    chunks = []
    producers = {}
    for begin in range(0, size, chunk_rows):
        length = min(chunk_rows, size - begin)
        keys = program.tensor((length, 1), dtype=np.uint32)[0, 0]
        reads = tuple(tensor.region(begin if tensor.shape[0] != 1 else 0, 0,
            length if tensor.shape[0] != 1 else 1, 1) if index in key_inputs else tensor
            for index, tensor in enumerate(operands))
        _ExpressionKernel((key_expression,)).bind(program, reads, (keys,))
        directory, count = _group_ordinals(program, (keys,), chunk_rows, begin, begin // chunk_rows)
        partials = program.tensor((min(count, base.shape[0]), features), (1, output.block_shape[1]),
            dtype=np.float32 if base.dtype.kind == 'f' else base.dtype)
        chunks.append((directory, count, partials))
        ordinal_view = directory.slice(0, count, 1, count)
        direct_selector = directory.slice(0, 7 * count, 1, count)
        active_count = directory.slice(0, 8 * count, 1, 1)
        flat_ranges, selector_caches = {}, {}
        for (segment, panel), partial in partials.blocks.items():
            column = panel * partials.block_shape[1]
            bounds = directory.slice(0, 5 * count + 2 * segment, 1, 2)
            range_key = segment, partial.shape[1]
            if range_key not in flat_ranges:
                flat_bounds = bounds
                if partial.shape[1] != 1 and dynamic_indexed:
                    flat_bounds = program.tensor((1, 2), dtype=np.uint32)[0, 0]
                    bound_value, = arguments(1)
                    _ExpressionKernel((bound_value * partial.shape[1],)).bind(program, (bounds,), (flat_bounds,))
                flat_ranges[range_key] = flat_bounds
            bound_operands, direct = list(operands), {}
            for index in value_inputs:
                tensor = operands[index]
                source_row = begin // tensor.block_shape[0] * tensor.block_shape[0] if tensor.shape[0] != 1 else 0
                source_column = column if tensor.shape[1] != 1 else 0
                ref = tensor.region(source_row, source_column,
                    min(tensor.block_shape[0], tensor.shape[0] - source_row), partial.shape[1] if tensor.shape[1] != 1 else 1)
                direct[index] = len(bound_operands), (source_row, source_column), tensor.shape
                bound_operands.append(ref)
            bound_operands = tuple(bound_operands)
            function = _bind_segment_expression(program, _segment_expression(update_value, bound_operands, column, direct),
                bound_operands, ordinal_view, bounds, flat_ranges[range_key], direct_selector, count, partial.shape[1], partial,
                active_count, segment, True, selector_caches.setdefault(range_key, {}))
            producers[(partial.view.tensor, partial.view.extent)] = function
    stripes = {}
    for coordinate in itertools.product(*(range(length) for length in grid)):
        target = output_spec.resolve(coordinate)
        index = tuple(output_spec.index_map(*coordinate))
        row, column = (i * block for i, block in zip(index, output_spec.block_shape))
        stripes.setdefault((column, target.shape[1]), []).append((row, target))
    directories = {}
    for (column, width), regions in stripes.items():
        regions.sort(key=lambda region: region[0])
        coverage = tuple((row, target.shape[0]) for row, target in regions)
        if any(row + height > next_row for (row, height), (next_row, _) in zip(coverage, coverage[1:])):
            raise ValueError('Unique-owner indexed sums require disjoint output regions')
        if coverage not in directories:
            directories[coverage] = _routing_directory(program, chunks, coverage)
        owners, reverse_keys, reverse_ordinals, offsets = directories[coverage]
        candidates = tuple(partials.region(segment, column, 1, width)
            for _, _, partials in chunks for segment in range(partials.shape[0]))
        producer_functions = tuple(producers[(ref.view.tensor, ref.view.extent)] for ref in candidates)
        route, table = _routing_domain(program, owners, reverse_keys, reverse_ordinals, offsets, candidates, producer_functions, len(regions))
        for consumer, (row, target) in enumerate(regions):
            initial = base.region(row, column, *target.shape)

            # design/algorithm-sources.md#shared-sparse-routing-lowering
            def finish(metal, target=target, initial=initial, row=row, consumer=consumer, table=table, partial_dtype=candidates[0].dtype):
                scalar = {'f2': 'half' if metal else '_Float16', 'f4': 'float',
                    'i4': 'int32_t', 'i8': 'int64_t', 'u4': 'uint32_t',
                    'u8': 'uint64_t', 'u1': 'uint8_t'}[partial_dtype.kind + str(partial_dtype.itemsize)]
                address_slot = f'ordinal*{table.view.row_stride}'
                stride_slot = f'{address_slot}+{2*table.view.column_stride}'
                address = f'(({"device " if metal else ""}const {scalar} *)p4[{address_slot}])'
                load = f'{address}[c*p4[{stride_slot}]]'
                accumulator = 'float' if base.dtype.kind == 'f' else 'uint64_t'
                return f'''uint32_t lo=p2[{consumer}],end=p2[{consumer+1}],hi=end,key={row}+r;
                while(lo<hi) {{ uint32_t mid=lo+(hi-lo)/2; if(p0[mid]<key)lo=mid+1; else hi=mid; }}
                for(uint32_t c=lane;c<{target.shape[1]};c+=lanes) {{
                  {accumulator} total=p3[r*{initial.view.row_stride}+c*{initial.view.column_stride}];
                  for(uint32_t k=lo;k<end && p0[k]==key;k++) {{ uint32_t ordinal=p1[k]; total+={load}; }}
                  p5[r*{target.view.row_stride}+c*{target.view.column_stride}]=total;
                }}'''

            function = _compiled_region(program, (reverse_keys, reverse_ordinals, offsets, initial, table), target, finish)
            check(program.native.algebra_route_attach(program.handle, function, route, consumer))


# design/algorithm-sources.md#shared-sparse-routing-lowering
def _routing_directory(program, chunks, coverage):
    total = sum(partials.shape[0] for _, _, partials in chunks)
    metadata = program.tensor((1, 2 * total), dtype=np.uint32)[0, 0]
    keys = tuple(directory.slice(0, 4 * count, 1, partials.shape[0]) for directory, count, partials in chunks)

    # design/algorithm-sources.md#shared-sparse-routing-lowering
    def owner_source(metal):
        qualifier = 'constant' if metal else 'static const'
        arrays = f'{qualifier} uint32_t starts[]={{'+','.join(str(row) for row, _ in coverage)+'};\n'
        arrays += f'{qualifier} uint32_t ends[]={{'+','.join(str(row + height) for row, height in coverage)+'};'
        lines, origin = [], 0
        for index, ref in enumerate(keys):
            count = ref.shape[1]
            lines.append(f'''for(uint32_t i=lane;i<{count};i+=lanes) {{
              uint32_t key=p{index}[i],lo=0,hi={len(coverage)},owner=0xffffffffu;
              while(lo<hi) {{ uint32_t mid=lo+(hi-lo)/2; if(starts[mid]<=key)lo=mid+1; else hi=mid; }}
              if(lo && key<ends[lo-1])owner=lo-1;
              p{len(keys)}[{origin}+i]=owner;
              p{len(keys)}[{total+origin}+i]=owner==0xffffffffu?0xffffffffu:key;
            }}''')
            origin += count
        return arrays, '\n'.join(lines)

    _compiled_region(program, keys, metadata, owner_source)
    owners = metadata.slice(0, 0, 1, total)
    reverse, _ = _group_ordinals(program, (metadata.slice(0, total, 1, total),), 1)
    reverse_keys = reverse.slice(0, 0, 1, total)
    reverse_ordinals = reverse.slice(0, total, 1, total)
    offsets = program.tensor((1, len(coverage) + 1), dtype=np.uint32)[0, 0]

    # design/algorithm-sources.md#shared-sparse-routing-lowering
    def offset_source(metal):
        boundaries = tuple(row for row, _ in coverage) + (coverage[-1][0] + coverage[-1][1],)
        array = f'{"constant" if metal else "static const"} uint32_t boundaries[]={{'+','.join(map(str, boundaries))+'};'
        return array, f'''for(uint32_t c=lane;c<{len(boundaries)};c+=lanes) {{
          uint32_t key=boundaries[c],lo=0,hi={total};
          while(lo<hi) {{ uint32_t mid=lo+(hi-lo)/2; if(p0[mid]<key)lo=mid+1; else hi=mid; }}
          p1[c]=lo;
        }}'''

    _compiled_region(program, (reverse_keys,), offsets, offset_source)
    return owners, reverse_keys, reverse_ordinals, offsets


# design/algorithm-sources.md#shared-sparse-routing-lowering
def _routing_domain(program, owners, keys, ordinals, offsets, candidates, producers, consumers):
    import ctypes as C
    from . import Ref, check
    from ._native import View
    route = program.native.algebra_route_create(program.handle, owners.view, ordinals.view, offsets.view,
        (View * len(candidates))(*(ref.view for ref in candidates)), len(candidates), consumers)
    if not route:
        check(C.get_errno() or 12)
    check(program.native.algebra_route_hold(program.handle, route, (View * 1)(keys.view), 1))
    check(program.native.algebra_route_producers(program.handle, route,
        (C.c_size_t * len(producers))(*producers), len(producers)))
    return route, Ref(program, program.native.algebra_route_table(program.handle, route), np.uint64)


# design/algorithm-sources.md#shared-contraction-lowering
def dot(left, right, *, tile_k=128):
    if tile_k < 1:
        raise ValueError('Contraction K tiles must be positive')
    return _Expression('dot', tuple(map(_literal, (left, right))), tile_k)


# design/algorithm-sources.md#shared-contraction-lowering
def _requires_regions(node):
    return node.operation in ('dot', 'sum', 'cast', 'transpose') or any(_requires_regions(child) for child in node.operands)


# design/algorithm-sources.md#shared-contraction-lowering
def _bind_operation(program, operation, inputs, target):
    from . import check
    from ._native import View
    check(program.native.algebra_bind(program.handle, operation.op, inputs[0].view,
        inputs[1].view if len(inputs) == 2 else View(), target.view, operation.alpha, operation.beta))


class _ExpressionRegions:
    # design/algorithm-sources.md#shared-contraction-lowering
    def __init__(self, program, specs, coordinate, cache):
        self.program, self.coordinate, self.cache = program, coordinate, cache
        self.sources = tuple(spec.resolve(coordinate) for spec in specs)
        self.whole = tuple(spec.block_shape is None for spec in specs)
        self.layouts = {}

    # design/algorithm-sources.md#shared-contraction-lowering
    def layout(self, node):
        from math import gcd
        if node in self.layouts:
            return self.layouts[node]
        if node.operation == 'input':
            source = self.sources[node.value]
            shape = source.shape
            result = shape, (self.whole[node.value],) * 2, source.block_shape if self.whole[node.value] else shape
        elif node.operation == 'index_vector':
            result = (1, node.value[0]), (False, False), (1, node.value[1])
        elif node.operation in ('literal', 'program_id', 'row', 'column'):
            result = (1, 1), (False, False), (1, 1)
        elif node.operation == 'cast':
            result = self.layout(node.operands[0])
        elif node.operation == 'transpose':
            result = tuple(value[::-1] for value in self.layout(node.operands[0]))
        elif node.operation == 'dot':
            left, right = map(self.layout, node.operands)
            if left[0][1] != right[0][0]:
                raise ValueError('Contraction inner dimensions differ')
            result = (left[0][0], right[0][1]), (left[1][0], right[1][1]), (left[2][0], right[2][1])
        elif node.operation == 'sum':
            child = self.layout(node.operands[0])
            result = (child[0][0], 1), (child[1][0], False), (child[2][0], 1)
        elif node.operation in ('+', '-', '*', '/', '<', '<=', '>', '>=', '==', '&', '|', 'select', 'rsqrt', 'exp', 'tanh', 'load', '//', '%'):
            children = tuple(map(self.layout, node.operands))
            shape = tuple(max(child[0][axis] for child in children) for axis in range(2))
            if any(child[0][axis] not in (1, shape[axis]) for child in children for axis in range(2)):
                raise ValueError('Computed contraction operand shapes must broadcast')
            global_axes, steps = [], []
            for axis in range(2):
                participating = tuple(child for child in children if child[0][axis] == shape[axis])
                global_axes.append(all(child[1][axis] for child in participating))
                cuts = tuple(child[2][axis] for child in participating if child[2][axis] < child[0][axis])
                step = cuts[0] if cuts else shape[axis]
                for cut in cuts[1:]:
                    step = gcd(step, cut)
                steps.append(step)
            result = shape, tuple(global_axes), tuple(steps)
        else:
            raise ValueError('Computed contraction operands require pointwise expressions or contractions')
        self.layouts[node] = result
        return result

    # design/algorithm-sources.md#shared-contraction-lowering
    def key(self, node, origin, shape):
        used = set()

        # design/algorithm-sources.md#shared-contraction-lowering
        def visit(value):
            if value.operation in ('input', 'load'):
                used.add(value.value)
            for child in value.operands:
                visit(child)

        visit(node)
        identities = []
        for index in sorted(used):
            source = self.sources[index]
            if self.whole[index]:
                identities.append((index, id(source)))
            else:
                view = source.view
                identities.append((index, view.tensor, view.extent, view.offset, view.rows, view.columns, view.row_stride, view.column_stride))
        return node, tuple(identities), origin, shape

    # design/algorithm-sources.md#shared-contraction-lowering
    def temporary(self, shape, dtype=np.float32):
        return self.program.tensor(shape, dtype=dtype)[0, 0]

    # design/algorithm-sources.md#shared-contraction-lowering
    def panel(self, node, origin, shape):
        if node.operation == 'input':
            source = self.sources[node.value]
            return source.region(*origin, *shape) if self.whole[node.value] else source.slice(*origin, *shape)
        if node.operation == 'transpose':
            return self.panel(node.operands[0], origin[::-1], shape[::-1]).T
        if node.operation == 'sum':
            return self.reduction(node, origin[0], shape[0], _expression_dtype(node, self.sources))
        dtype = _expression_dtype(node, self.sources)
        if node.operation == 'cast' and node.operands[0].operation == 'input' and self.sources[node.operands[0].value].dtype == dtype:
            return self.panel(node.operands[0], origin, shape)
        key = ('panel', self.key(node, origin, shape))
        if key not in self.cache:
            target = self.temporary(shape, dtype)
            if node.operation == 'dot':
                self.publish(self.parts(node, origin, shape, target), target)
            else:
                self.emit(node, origin, shape, target)
            self.cache[key] = target
        return self.cache[key]

    # design/algorithm-sources.md#shared-contraction-lowering
    def parts(self, node, origin, shape, direct=None):
        from math import gcd
        key = ('parts', self.key(node, origin, shape))
        if key in self.cache:
            return self.cache[key]
        left, right = node.operands
        left_layout, right_layout = self.layout(left), self.layout(right)
        inner = left_layout[0][1]
        if inner != right_layout[0][0]:
            raise ValueError('Contraction inner dimensions differ')
        tile = min(node.value, inner)
        for layout, axis in ((left_layout, 1), (right_layout, 0)):
            if layout[2][axis] < layout[0][axis]:
                tile = gcd(tile, layout[2][axis])
        parts = []
        for start in range(0, inner, tile):
            length = min(tile, inner - start)
            left_panel = self.panel(left, (origin[0], start), (shape[0], length))
            right_panel = self.panel(right, (start, origin[1]), (length, shape[1]))
            reverse = left_panel.dtype == np.dtype('float16') and right_panel.dtype == np.dtype('float32')
            if reverse:
                destination = self.temporary(shape[::-1]).T
            else:
                destination = direct if direct is not None and tile == inner and direct.dtype == np.dtype('float32') else self.temporary(shape)
            _bind_operation(self.program, matmul, (left_panel, right_panel), destination)
            parts.append(destination)
        while len(parts) > 2:
            reduced = []
            for index in range(0, len(parts), 2):
                if index + 1 == len(parts):
                    reduced.append(parts[index])
                else:
                    destination = self.temporary(shape)
                    _bind_operation(self.program, add, parts[index:index + 2], destination)
                    reduced.append(destination)
            parts = reduced
        self.cache[key] = tuple(parts)
        return self.cache[key]

    # design/algorithm-sources.md#shared-contraction-lowering
    def reduction(self, node, row, rows, dtype, direct=None):
        key = ('sum', self.key(node, (row, 0), (rows, 1)), dtype.str)
        if key in self.cache:
            return self.cache[key]
        child = node.operands[0]
        layout = self.layout(child)
        width, tile = layout[0][1], layout[2][1]
        parts = []
        for column in range(0, width, tile):
            length = min(tile, width-column)
            target = direct if direct is not None and tile >= width and direct.dtype == dtype else self.temporary((rows, 1), dtype)
            self.emit(child, (row, column), (rows, length), target, reduce=True)
            parts.append(target)
        while len(parts) > 1:
            reduced = []
            for index in range(0, len(parts), 2):
                if index + 1 == len(parts):
                    reduced.append(parts[index])
                    continue
                target = direct if direct is not None and len(parts) == 2 and direct.dtype == dtype else self.temporary((rows, 1), dtype)
                if dtype.kind == 'f':
                    _bind_operation(self.program, add, parts[index:index+2], target)
                else:
                    left, right = arguments(2)
                    bits = 0xffffffffffffffff
                    _ExpressionKernel(((left & bits)+(right & bits),)).bind(self.program, parts[index:index+2], (target,), self.coordinate)
                reduced.append(target)
            parts = reduced
        self.cache[key] = parts[0]
        return parts[0]

    # design/algorithm-sources.md#shared-contraction-lowering
    def publish(self, parts, target):
        if len(parts) == 2:
            destination = target if target.dtype == np.dtype('float32') else self.temporary(target.shape)
            _bind_operation(self.program, add, parts, destination)
            parts = (destination,)
        if parts[0] is not target:
            _bind_operation(self.program, affine(), parts, target)

    # design/algorithm-sources.md#shared-contraction-lowering
    def emit(self, value, origin, shape, target, external=False, reduce=False):
        inputs, replacements = [], {}

        # design/algorithm-sources.md#shared-contraction-lowering
        def reference(key, refs):
            if key not in replacements:
                terms = tuple(_Expression('input', value=len(inputs) + index) for index in range(len(refs)))
                inputs.extend(refs)
                replacements[key] = terms[0] if len(terms) == 1 else terms[0] + terms[1]
            return replacements[key]

        # design/algorithm-sources.md#shared-contraction-lowering
        def lower(node, accumulation=target.dtype):
            if node.operation == 'cast':
                child = node.operands[0]
                return _Expression('cast', (lower(child, _expression_dtype(child, self.sources)),), node.value)
            if node.operation == 'sum':
                layout = self.layout(node)
                row = 0 if layout[0][0] == 1 or (external and not layout[1][0]) else origin[0]
                rows = 1 if layout[0][0] == 1 else shape[0]
                dtype = np.dtype(np.int64 if accumulation.kind in 'ib' else np.uint64 if accumulation.kind == 'u' else np.float32)
                return reference(('sum', node, row, rows, dtype.str), (self.reduction(node, row, rows, dtype),))
            if node.operation == 'dot':
                layout = self.layout(node)
                where = tuple(0 if layout[0][axis] == 1 or (external and not layout[1][axis]) else origin[axis] for axis in range(2))
                extent = tuple(1 if layout[0][axis] == 1 else shape[axis] for axis in range(2))
                return reference(('dot', node, where, extent), self.parts(node, where, extent))
            if node.operation == 'input':
                source = self.sources[node.value]
                if external and not self.whole[node.value]:
                    ref = source
                else:
                    where = tuple(0 if source.shape[axis] == 1 else origin[axis] for axis in range(2))
                    extent = tuple(1 if source.shape[axis] == 1 else shape[axis] for axis in range(2))
                    ref = self.panel(node, where, extent)
                return reference(('input', node.value), (ref,))
            if node.operation == 'transpose':
                layout = self.layout(node)
                where = tuple(0 if layout[0][axis] == 1 or (external and not layout[1][axis]) else origin[axis] for axis in range(2))
                extent = tuple(1 if layout[0][axis] == 1 else shape[axis] for axis in range(2))
                return reference(('transpose', node, where, extent), (self.panel(node, where, extent),))
            if node.operation == 'load':
                symbol = reference(('load_source', node.value), (self.sources[node.value],))
                return _Expression('load', tuple(lower(child, accumulation) for child in node.operands), symbol.value)
            if node.operation == 'index_vector':
                return node if external or node.value[0] == 1 else _Expression('index_vector', value=(shape[1], shape[1], node.value[2] + origin[1]))
            if node.operation in ('row', 'column') and not external:
                axis = 0 if node.operation == 'row' else 1
                return node + origin[axis]
            if node.operation == 'indexed_add':
                raise ValueError('Indexed addition requires an output root')
            return _Expression(node.operation, tuple(lower(child, accumulation) for child in node.operands), node.value)

        lowered = lower(value)
        if reduce:
            lowered = lowered.sum()
        _ExpressionKernel((lowered,)).bind(self.program, tuple(inputs), (target,), self.coordinate)


# design/algorithm-sources.md#shared-contraction-lowering
def _lower_region_expressions(program, expressions, grid, input_specs, output_specs):
    import itertools
    cache = {}
    for coordinate in itertools.product(*(range(length) for length in grid)):
        lowering = _ExpressionRegions(program, input_specs, coordinate, cache)

        # design/algorithm-sources.md#shared-contraction-lowering
        def specialize(node):
            if node.operation == 'program_id':
                return _literal(coordinate[node.value])
            return _Expression(node.operation, tuple(specialize(child) for child in node.operands), node.value)

        for expression, spec in zip(expressions, output_specs):
            value = _resolve_logical(specialize(expression), lowering.sources)
            target = spec.resolve(coordinate)
            origin = tuple(index * block for index, block in zip(spec.index_map(*coordinate), spec.block_shape))
            domain_shape = spec._tensor.shape
            if value.operation == 'transpose':
                value, target, origin, domain_shape = value.operands[0], target.T, origin[::-1], domain_shape[::-1]
            if value.operation == 'dot':
                layout = lowering.layout(value)
                for axis in range(2):
                    expected = domain_shape[axis] if layout[1][axis] else target.shape[axis]
                    if layout[0][axis] != expected:
                        raise ValueError('Contraction output shape differs from its operand domains')
                where = tuple(origin[axis] if layout[1][axis] else 0 for axis in range(2))
                lowering.publish(lowering.parts(value, where, target.shape, target), target)
            elif value.operation == 'sum' and target.shape[1] == 1:
                layout = lowering.layout(value)
                row = origin[0] if layout[1][0] else 0
                dtype = np.dtype(np.int64 if target.dtype.kind in 'ib' else np.uint64 if target.dtype.kind == 'u' else np.float32)
                result = lowering.reduction(value, row, target.shape[0], dtype, target)
                if result is not target:
                    symbol, = arguments(1)
                    _ExpressionKernel((symbol,)).bind(program, (result,), (target,), coordinate)
            else:
                lowering.emit(value, origin, target.shape, target, external=True)
