from dataclasses import dataclass

import numpy as np

_REDUCTIONS = ('sum', 'max', 'min', 'any', 'all')
_REAL_FUNCTIONS = ('exp', 'rsqrt', 'tanh', 'asinh', 'expm1', 'log1p', 'log', 'sqrt', 'power', 'logaddexp', 'random_normal')
_POINTWISE_FUNCTIONS = _REAL_FUNCTIONS + ('isfinite', 'abs', 'floor', 'floor_divide', 'philox')
_POINTWISE_OPERATIONS = ('+', '-', '*', '/', '<', '<=', '>', '>=', '==', '&', '|', 'select', 'cast', '//', '%',
    'maximum', 'minimum', *_POINTWISE_FUNCTIONS)


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


# design/algorithm-sources.md#static-indexed-access-specialization
@dataclass(frozen=True)
class _StaticTable:
    refs: tuple
    ordinals: tuple
    shape: tuple
    block_shape: tuple
    grid: tuple
    dtype: object


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
        if any(dimension < -1 for dimension in shape) or shape.count(-1) > 1:
            raise ValueError('Logical indexed dimensions must be nonnegative with at most one inferred axis')
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

    # design/algorithm-sources.md#stable-indexed-ordering
    def argsort(self, axis=1):
        import operator
        axis = operator.index(axis)
        if axis not in (-2, -1, 0, 1):
            raise ValueError('Ordering axis must refer to the two-dimensional expression domain')
        return _Expression('argsort', (self,)) if axis % 2 else self.T.argsort().T

    # design/algorithm-sources.md#shared-associative-reductions
    def _reduce(self, operation, axis):
        axes = (0, 1) if axis is None else (axis,) if isinstance(axis, int) else tuple(axis)
        if any(value not in (-2, -1, 0, 1) for value in axes):
            raise ValueError('Expression reduction axes refer to its two-dimensional domain')
        axes = tuple(sorted(set(value % 2 for value in axes)))
        if not axes:
            return self.equal(0).equal(False) if operation in ('any', 'all') else self
        if axes == (0,):
            return self.T._reduce(operation, 1).T
        reduced = _Expression(operation, (self,))
        return reduced.T._reduce(operation, 1) if axes == (0, 1) else reduced

    # design/algorithm-sources.md#shared-associative-reductions
    def sum(self, axis=1):
        return self._reduce('sum', axis)

    # design/algorithm-sources.md#shared-associative-reductions
    def max(self, axis=1):
        return self._reduce('max', axis)

    # design/algorithm-sources.md#shared-associative-reductions
    def min(self, axis=1):
        return self._reduce('min', axis)

    # design/algorithm-sources.md#shared-associative-reductions
    def any(self, axis=1):
        return self._reduce('any', axis)

    # design/algorithm-sources.md#shared-associative-reductions
    def all(self, axis=1):
        return self._reduce('all', axis)

    @property
    # design/algorithm-sources.md#shared-contraction-lowering
    def T(self):
        if self.operation == 'transpose':
            return self.operands[0]
        if self.operation in ('literal', 'program_id'):
            return self
        if self.operation == 'index_vector':
            return _Expression('index_vector', value=(*self.value[:3], 1 - self.value[3]))
        if self.operation in ('row', 'column'):
            return _Expression('column' if self.operation == 'row' else 'row')
        if self.operation == 'domain':
            return _Expression('domain', (self.operands[0].T,), tuple(value[::-1] for value in self.value))
        if self.operation == 'dot':
            return _Expression('dot', tuple(child.T for child in self.operands[::-1]), self.value)
        if self.operation in _POINTWISE_OPERATIONS:
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

    # design/algorithm-sources.md#shared-elementary-functions
    def asinh(self):
        return _Expression('asinh', (self,))

    # design/algorithm-sources.md#shared-elementary-functions
    def expm1(self):
        return _Expression('expm1', (self,))

    # design/algorithm-sources.md#shared-elementary-functions
    def log1p(self):
        return _Expression('log1p', (self,))

    # design/algorithm-sources.md#shared-elementary-functions
    def log(self):
        return _Expression('log', (self,))

    # design/algorithm-sources.md#shared-elementary-functions
    def sqrt(self):
        return _Expression('sqrt', (self,))

    # design/algorithm-sources.md#shared-elementary-functions
    def abs(self):
        return _Expression('abs', (self,))

    # design/algorithm-sources.md#shared-elementary-functions
    def isfinite(self):
        return _Expression('isfinite', (self,))

    # design/algorithm-sources.md#shared-elementary-functions
    def floor(self):
        return _Expression('floor', (self,))

    # design/algorithm-sources.md#shared-elementary-functions
    def power(self, other):
        return _Expression('power', (self, _literal(other)))

    # design/algorithm-sources.md#shared-elementary-functions
    def logaddexp(self, other):
        return _Expression('logaddexp', (self, _literal(other)))

    # design/algorithm-sources.md#shared-elementary-functions
    def floor_divide(self, other):
        return _Expression('floor_divide', (self, _literal(other)))

    # design/algorithm-sources.md#shared-elementary-functions
    def arcsinh(self):
        return self.asinh()

    # design/algorithm-sources.md#shared-elementary-functions
    def __abs__(self):
        return self.abs()

    # design/algorithm-sources.md#shared-elementary-functions
    def __pow__(self, other):
        return self.power(other)

    # design/algorithm-sources.md#shared-elementary-functions
    def __rpow__(self, other):
        return _literal(other).power(self)


# design/algorithm-sources.md#region-expression-fusion
def _literal(value):
    return value if isinstance(value, _Expression) else _Expression('literal', value=value.item() if isinstance(value, np.generic) else value)


# design/algorithm-sources.md#logical-indexed-views
def _resolve_logical(node, inputs, evaluate=True):
    import math
    if node.operation == 'indexed_contract':
        return node
    active = evaluate and not (node.operation == 'logical_load' and 0 in inputs[node.value[0]].shape)
    children = tuple(_resolve_logical(child, inputs, active) for child in node.operands)
    if node.operation == 'logical_load':
        index, shape = node.value
        source = inputs[index]
        volume = math.prod(source.shape)
        if -1 in shape:
            known = math.prod(dimension for dimension in shape if dimension != -1)
            if known == 0 or volume % known:
                raise ValueError('Logical indexed shape cannot infer an integral dimension')
            shape = tuple(volume // known if dimension == -1 else dimension for dimension in shape)
        if math.prod(shape) != volume:
            raise ValueError('Logical indexed shape volume differs from its bound input')
        if volume == 0:
            layout = _expression_layout(_Expression('+', children), inputs, tuple(hasattr(source, 'blocks') for source in inputs), {})
            dtype = _expression_dtype(_Expression('load', (_literal(0), _literal(0), _literal(False), children[-1]), index), inputs)
            return _Expression('domain', (children[-1].astype(dtype),), layout)
        ordinal, enabled = _literal(0), children[-2]
        for dimension, coordinate in zip(shape, children[:-2]):
            ordinal = ordinal * dimension + coordinate
            enabled = select(enabled, (coordinate >= 0) & (coordinate < dimension), False)
        return _Expression('load', (ordinal // source.shape[1], ordinal % source.shape[1], enabled, children[-1]), index)
    if node.operation == 'reshape':
        raise ValueError('Logical reshapes require indexed access')
    if evaluate and node.operation in ('//', '%') and all(child.operation == 'literal' for child in children):
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
        return np.full(rows.shape, node.value[2], dtype=np.int64) if node.value[0] == 1 else (rows if node.value[3] == 0 else columns) + node.value[2]
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
                value = node.value[0] if node.value[3] == 1 else 1
            elif node.operation == 'row' or node.operation in _REDUCTIONS:
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
        elif node.operation in _REDUCTIONS:
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
        chosen = selected[node]
        if len(chosen) > 1:
            refs = tuple(table.blocks[block] for block in chosen)
            ordinals = tuple(row*table.grid[1]+column for row, column in chosen)
            source = _StaticTable(refs, ordinals, table.shape, table.block_shape, table.grid, table.dtype)
            index = len(bound)
            bound.append(source)
            result = _Expression('load', children, index)
        elif chosen:
            block, = chosen
            ref = table.blocks[block]
            view = ref.view
            identity = (view.tensor, view.extent, view.offset, view.rows, view.columns, view.row_stride, view.column_stride, ref.dtype.str)
            if identity not in positions:
                positions[identity] = len(bound)
                bound.append(ref)
            first_row, first_column = (index*size for index, size in zip(block, table.block_shape))
            result = _Expression('load', (row-first_row, column-first_column, mask, other), positions[identity])
        else:
            result = other
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
    tile = max(1, length) if tile is None else operator.index(tile)
    if length < 0 or tile <= 0 or length > np.iinfo(np.int64).max:
        raise ValueError('Index vector length must be nonnegative and tile positive within the int64 domain')
    return _Expression('index_vector', value=(length, max(1, min(tile, length)), 0, 1))


# design/algorithm-sources.md#counter-based-random-generation
def philox4x32(counter_words, key0, key1):
    counter_words = tuple(counter_words)
    if len(counter_words) != 4:
        raise ValueError('Philox4x32 requires four counter words')
    operands = tuple(map(_literal, (*counter_words, key0, key1)))
    return tuple(_Expression('philox', operands, word) for word in range(4))


# design/algorithm-sources.md#counter-based-random-generation
def random_normal(key0, key1, ordinal):
    return _Expression('random_normal', tuple(map(_literal, (key0, key1, ordinal))))


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
        grid = tuple(grid)
        if 0 in grid:
            return
        active = tuple((value, spec) for value, spec in zip(self.values, output_specs) if spec._tensor.blocks)
        if len(active) != len(self.values):
            if active:
                _ExpressionKernel(tuple(value for value, _ in active)).bind_grid(program, grid, input_specs, tuple(spec for _, spec in active))
            return
        if any(value.operation == 'indexed_add' for value in self.values):
            for value, spec in zip(self.values, output_specs):
                if value.operation == 'indexed_add':
                    resolved = _resolve_logical(value, tuple(source._tensor for source in input_specs))
                    _lower_indexed_add(program, resolved, grid, input_specs, spec)
                else:
                    _ExpressionKernel((value,)).bind_grid(program, grid, input_specs, (spec,))
            return
        if any(_requires_regions(value) for value in self.values) or any(not spec._tensor.blocks for spec in input_specs):
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
            access_axes, reduced_accesses = _expression_access_axes(expression, reads)
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
                    elif part.operation == 'index_vector' and part.value[3] == 1:
                        selector_width = max(selector_width, part.value[0])
                    for child in part.operands:
                        selector_shape(child)

                selector_shape(selector_value)
                selected = program.tensor((output.shape[0], selector_width), dtype=np.uint32)[0, 0]
                _ExpressionKernel((selector_value,)).bind(program, reads, (selected,))
                dynamic[node] = (selected, node.value)

            flattened = tuple(ref for source in reads for ref in
                (source.refs if isinstance(source, _StaticTable) else tuple(ref for _, ref in sorted(source.blocks.items())) if hasattr(source, 'blocks') else (source,)))
            sources = tuple(self.source(reads, output, metal, expression).encode() for metal in (False, True))
            offsets = []
            for source in reads:
                offsets.append((offsets[-1][0] + offsets[-1][1] if offsets else 0,
                    len(source.refs) if isinstance(source, _StaticTable) else len(source.blocks) if hasattr(source, 'blocks') else 1))
            for begin, rows, column, columns in _source_expression_regions(program, output):
                function = program.native.algebra_trace_count(program.handle)
                check(program.native.algebra_source(program.handle, *sources,
                    (View * len(flattened))(*(ref.view for ref in flattened)), len(flattened), output.view,
                    (C.c_uint8 * len(access_axes))(*access_axes), begin, rows, column, columns))
                for node, (selected, index) in dynamic.items():
                    first, count = offsets[index]
                    selection = selected.slice(begin, 0 if node in reduced_accesses else column, rows,
                                               selected.shape[1] if node in reduced_accesses else columns)
                    check(program.native.algebra_indexed(program.handle, function, selection.view,
                        (C.c_size_t * count)(*range(first, first + count)), count))

    # design/algorithm-sources.md#shared-associative-reductions
    def source(self, inputs, output, metal, expression):
        widths, reductions = {}, []
        physical, pointers = [], {}
        for index, ref in enumerate(inputs):
            refs = ref.refs if isinstance(ref, _StaticTable) else tuple(ref for _, ref in sorted(ref.blocks.items())) if hasattr(ref, 'blocks') else (ref,)
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
            if node.operation in _REAL_FUNCTIONS:
                return False
            if node.operation in ('isfinite', 'philox'):
                return True
            if node.operation in _REDUCTIONS:
                return _reduction_dtype(node, inputs, output.dtype).kind in 'iub'
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
            elif node.operation == 'domain':
                if node.value[0][0] not in (1, output.shape[0]):
                    raise ValueError('Expression domain rows must broadcast to the output')
                width = node.value[0][1]
            elif node.operation == 'column':
                width = output.shape[1]
            elif node.operation == 'index_vector':
                if node.value[3] == 0 and node.value[0] not in (1, output.shape[0]):
                    raise ValueError('Index vector rows must broadcast to the output')
                width = node.value[0] if node.value[3] == 1 else 1
            elif node.operation == 'row':
                width = 1
            elif node.operation in _REDUCTIONS:
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
                    return f'((int64_t)({0 if part.value[0] == 1 else "r" if part.value[3] == 0 else column})+{part.value[2]}ll)'
                if part.operation in _REDUCTIONS:
                    return f'(({"long" if metal else "int64_t"}){names[part]})' if part.operation == 'sum' and _reduction_dtype(part, inputs, output.dtype).kind in 'ib' else names[part]
                ref = inputs[part.value]
                if part.operation == 'load':
                    return _indexed_load_expression(ref, pointers[part.value][0], layouts.get(part.value), args, metal)
                row_stride = ref.view.row_stride if ref.shape[0] != 1 else 0
                column_stride = ref.view.column_stride if ref.shape[1] != 1 else 0
                value = f'p{pointers[part.value][0]}[r*{row_stride}+({column})*{column_stride}]'
                return f'((float)({value}))' if ref.dtype.kind == 'f' else value

            return _emit_scalar_expression(node, inputs, metal, resolve)

        lines = ['#include <metal_stdlib>\nusing namespace metal;' if metal else '#include <stdint.h>\n#include <stdbool.h>\n#include <math.h>']
        lines.append(_scalar_helpers(metal))
        layouts = {}
        for index, ref in enumerate(inputs):
            if isinstance(ref, _StaticTable):
                name = f'mesh_static_load_{index}'
                lines.append(_static_table_source(ref, pointers[index][0], name, metal))
                layouts[index] = name
                continue
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
        lines.append(_METAL_EXPRESSION_HEAD if metal else 'void mesh_expression(const uintptr_t *buffers, const struct mesh_kernel_publication *publication) {')
        candidates = {pointer for index, ref in enumerate(inputs) if hasattr(ref, 'blocks') or isinstance(ref, _StaticTable) for pointer in pointers[index]}
        for index, ref in enumerate((*physical, output)):
            if index in candidates:
                continue
            scalar = ({'f2': 'half' if metal else '_Float16', 'f4': 'float', 'i4': 'int' if metal else 'int32_t',
                       'u4': 'uint' if metal else 'uint32_t', 'i8': 'long' if metal else 'int64_t',
                       'u8': 'ulong' if metal else 'uint64_t', 'u1': 'uchar' if metal else 'uint8_t', 'b1': 'bool'}[ref.dtype.kind + str(ref.dtype.itemsize)])
            qualifier = ('device ' if metal else '') + ('const ' if index < len(physical) else '')
            lines.append(f'{qualifier}{scalar} *p{index}=({qualifier}{scalar} *)buffers[{index}];')
        if not metal:
            lines.append(_CPU_PUBLICATION_LOOP)
        for node in reductions:
            name, child = names[node], node.operands[0]
            dtype = _reduction_dtype(node, inputs, output.dtype)
            unsigned = dtype.kind == 'u' or (node.operation == 'sum' and dtype.kind in 'ib' and integral(child))
            accumulator = (('ulong' if metal else 'uint64_t') if unsigned else ('long' if metal else 'int64_t')) if dtype.itemsize == 8 else (
                'float' if dtype.kind == 'f' else ('uint' if metal else 'uint32_t') if dtype.kind in 'ub' else ('int' if metal else 'int32_t'))
            if node.operation in ('max', 'min'):
                identity = ('-INFINITY' if node.operation == 'max' else 'INFINITY') if dtype.kind == 'f' else _scalar_expression(_literal(
                    (0 if node.operation == 'max' else 1) if dtype.kind == 'b' else
                    (np.iinfo(dtype).min if node.operation == 'max' else np.iinfo(dtype).max)), (), metal)
            else:
                identity = '1' if node.operation == 'all' else '0'
            lines.append(f'{accumulator} {name}={identity};')
            operand = emit(child, 'k')
            combine = (f'{name}+={operand}' if node.operation == 'sum' else
                f'{name}{"|" if node.operation == "any" else "&"}=(({operand})!=0)' if node.operation in ('any', 'all') else
                f'{name}=' + _scalar_expression(_Expression('maximum' if node.operation == 'max' else 'minimum'), (name, operand), metal, dtype))
            lines.append(f'for({"uint" if metal else "uint64_t"} k={"lane" if metal else "0"};k<{widths[child]};k+={32 if metal else 1}) {combine};')
            if metal and node.operation == 'sum' and dtype.kind in 'iub':
                lines.append(f'uint {name}_lo=simd_sum(uint(ulong({name})&65535ul)), {name}_mid=simd_sum(uint((ulong({name})>>16)&65535ul)), {name}_hi=simd_sum(uint(ulong({name})>>32));')
                lines.append(f'{name}_mid+={name}_lo>>16; {name}_hi+={name}_mid>>16; {name}=(ulong({name}_hi)<<32)|(ulong({name}_mid&65535u)<<16)|ulong({name}_lo&65535u);')
            elif metal and node.operation in ('max', 'min') and dtype.itemsize == 8:
                high_type = 'int' if dtype.kind == 'i' else 'uint'
                operation = 'max' if node.operation == 'max' else 'min'
                low_identity = '0u' if node.operation == 'max' else '0xffffffffu'
                lines.append(f'{high_type} {name}_hi={high_type}(ulong({name})>>32), {name}_best=simd_{operation}({name}_hi);')
                lines.append(f'uint {name}_lo=simd_{operation}({name}_hi=={name}_best?uint(ulong({name})):{low_identity});')
                lines.append(f'{name}={accumulator}((ulong(uint({name}_best))<<32)|ulong({name}_lo));')
            elif metal:
                operation = 'max' if node.operation in ('max', 'any') else 'min' if node.operation in ('min', 'all') else 'sum'
                lines.append(f'{name}=simd_{operation}({name});')
        lines.append(f'for({"ulong" if metal else "uint64_t"} c={"column_begin+lane" if metal else "part.column_begin"};c<{"column_end" if metal else "part.column_end"};c+={32 if metal else 1}) p{len(physical)}[r*{output.view.row_stride}+c*{output.view.column_stride}]={emit(expression, "c")};')
        lines.append('}' if metal else _CPU_PUBLICATION_END)
        return '\n'.join(lines)


# design/algorithm-sources.md#compiled-column-access-domains
def _expression_access_axes(expression, inputs):
    axes = [3] * len(inputs)
    reduced_accesses = set()

    # design/algorithm-sources.md#compiled-column-access-domains
    def visit(node, reduced=False):
        if node.operation == 'load':
            axes[node.value] = 0
            if reduced:
                reduced_accesses.add(node)
        if node.operation == 'input' and reduced:
            axes[node.value] &= 1
        for child in node.operands:
            visit(child, reduced or node.operation in _REDUCTIONS)

    visit(expression)
    return tuple(0 if hasattr(source, 'blocks') or isinstance(source, _StaticTable) else axes[index]
                 for index, source in enumerate(inputs)
                 for _ in (source.refs if isinstance(source, _StaticTable) else source.blocks.values() if hasattr(source, 'blocks') else (source,))), frozenset(reduced_accesses)


# design/algorithm-sources.md#compiled-column-access-domains
def _source_expression_regions(program, output):
    import math
    quantum = program.native.tensor_publication_bytes(output.view.tensor, output.view.extent) // output.dtype.itemsize
    rows, columns = output.shape
    if output._writer_error:
        return ((0, rows, 0, columns),)
    row_major = output.view.column_stride == 1
    inner, outer = (columns, rows) if row_major else (rows, columns)
    if inner > quantum and (outer == 1 or inner % quantum == 0):
        rectangles = ((i, 1, j, min(quantum, inner-j)) for i in range(outer) for j in range(0, inner, quantum))
    else:
        step = quantum // math.gcd(quantum, inner)
        rectangles = ((i, min(step, outer-i), 0, inner) for i in range(0, outer, step))
    return tuple(rectangle if row_major else (rectangle[2], rectangle[3], rectangle[0], rectangle[1])
                 for rectangle in rectangles)


# design/algorithm-sources.md#compiled-row-access-domains
def _source_row_regions(program, output):
    import math
    if output._writer_error or output.view.column_stride != 1:
        return ((0, output.shape[0]),)
    page_bytes = program.native.tensor_publication_bytes(output.view.tensor, output.view.extent)
    row_bytes = output.shape[1] * output.dtype.itemsize
    rows = page_bytes // math.gcd(page_bytes, row_bytes)
    return tuple((first, min(rows, output.shape[0]-first)) for first in range(0, output.shape[0], rows))


# design/algorithm-sources.md#compiled-column-access-domains
_METAL_EXPRESSION_HEAD = 'kernel void mesh_expression(device const ulong *buffers [[buffer(0)]], constant ulong *domain [[buffer(1)]], uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { const ulong r=domain[0]+row, column_begin=domain[1], column_end=domain[2];'


# design/algorithm-sources.md#in-operation-publication
_CPU_PUBLICATION_LOOP = 'for(uint64_t section=0;section<publication->count;section++) { const struct mesh_kernel_section part=publication->sections[section]; for(uint64_t r=part.row_begin;r<part.row_end;r++) {'
_CPU_PUBLICATION_END = '} publication->publish(publication->context,part.first,part.count); }}'


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
                visit(child, () if node.operation in _REDUCTIONS else path)

    visit(expression)
    return accesses


# design/algorithm-sources.md#static-indexed-access-specialization
def _static_table_source(table, first, name, metal):
    dtype = table.dtype
    scalar = {'f2': 'half' if metal else '_Float16', 'f4': 'float', 'i4': 'int' if metal else 'int32_t',
              'u4': 'uint' if metal else 'uint32_t', 'i8': 'long' if metal else 'int64_t',
              'u8': 'ulong' if metal else 'uint64_t', 'u1': 'uchar' if metal else 'uint8_t', 'b1': 'bool'}[dtype.kind+str(dtype.itemsize)]
    integer = 'ulong' if metal else 'uint64_t'
    constant = 'constant' if metal else 'static const'
    lines = []
    for suffix, values in (('ordinal', table.ordinals), ('rs', tuple(ref.view.row_stride for ref in table.refs)),
                           ('cs', tuple(ref.view.column_stride for ref in table.refs))):
        lines.append(f'{constant} {integer} {name}_{suffix}[]={{'+','.join(map(str, values))+'};')
    value = f'p[row%{table.block_shape[0]}*{name}_rs[low]+column%{table.block_shape[1]}*{name}_cs[low]]'
    result_type = 'float' if dtype.kind == 'f' else scalar
    lines.append(f"""// design/algorithm-sources.md#static-indexed-access-specialization
    {'inline' if metal else 'static inline'} {result_type} {name}({'device const ulong *' if metal else 'const uintptr_t *'} buffers,{integer} row,{integer} column) {{
      {integer} ordinal=row/{table.block_shape[0]}*{table.grid[1]}+column/{table.block_shape[1]},low=0,high={len(table.refs)-1};
      while(low<high) {{ {integer} middle=low+(high-low)/2;
        if({name}_ordinal[middle]<ordinal)low=middle+1;else high=middle; }}
      {'device ' if metal else ''}const {scalar} *p=({'device ' if metal else ''}const {scalar} *)buffers[{first}+low];
      return {value};
    }}""")
    return '\n'.join(lines)


# design/algorithm-sources.md#shared-scalar-load-emission
def _indexed_load_expression(ref, pointer, layout, args, metal):
    row, column, mask, other = args
    if isinstance(ref, _StaticTable):
        return f'(({mask})?{layout}(buffers,({row}),({column})):({other}))'
    if hasattr(ref, 'blocks'):
        block = f'(({row})/{ref.block_shape[0]}*{ref.grid[1]}+({column})/{ref.block_shape[1]})'
        scalar, strides = layout
        address = f'(({"device " if metal else ""}const {scalar} *)buffers[{pointer}+{block}])'
        value = f'{address}[(({row})%{ref.block_shape[0]})*{strides[0]}+(({column})%{ref.block_shape[1]})*{strides[1]}]'.replace('CANDIDATE', block)
    else:
        strides = (ref.view.row_stride, ref.view.column_stride) if layout is None else layout[1]
        value = f'p{pointer}[({row})*{strides[0]}+({column})*{strides[1]}]'
    value = f'((float)({value}))' if ref.dtype.kind == 'f' else value
    return f'(({mask})?({value}):({other}))'


# design/algorithm-sources.md#shared-scalar-load-emission
def _emit_scalar_expression(node, inputs, metal, resolve):
    if node.operation in ('philox', 'random_normal') and any(_expression_dtype(child, inputs).kind not in 'iub' for child in node.operands):
        raise ValueError('Counter-based random functions require integral keys, counters and ordinals')
    if node.operation in ('input', 'row', 'column', 'index_vector') or node.operation in _REDUCTIONS:
        return resolve(node, ())
    args = tuple(_emit_scalar_expression(child, inputs, metal, resolve) for child in node.operands)
    if node.operation == 'load':
        return resolve(node, args)
    if node.operation == 'block_ordinal':
        row, column = args
        rows, columns, grid_columns = node.value
        return f'(({row})/{rows}*{grid_columns}+({column})/{columns})'
    return _scalar_expression(node, args, metal,
        _expression_dtype(node.operands[0], inputs) if node.operation in ('abs', 'floor', 'isfinite') else
        _expression_dtype(node, inputs) if node.operation in ('//', '%', 'maximum', 'minimum', 'floor_divide') else None)


# design/algorithm-sources.md#logical-indexed-views
def _expression_dtype(node, inputs):
    if node.operation == 'input':
        return inputs[node.value].dtype
    if node.operation == 'cast':
        return np.dtype(node.value)
    if node.operation == 'dot':
        types = tuple(_expression_dtype(child, inputs) for child in node.operands)
        return np.dtype('bool') if all(dtype.kind == 'b' for dtype in types) else _expression_dtype(_Expression('*', node.operands), inputs)
    if node.operation == 'indexed_contract' or node.operation in _REAL_FUNCTIONS:
        return np.dtype('float32')
    if node.operation in ('philox', 'argsort'):
        return np.dtype('uint32')
    if node.operation in ('<', '<=', '>', '>=', '==', 'isfinite'):
        return np.dtype('bool')
    if node.operation in ('abs', 'floor', 'domain'):
        return _expression_dtype(node.operands[0], inputs)
    if node.operation in ('row', 'column', 'program_id', 'index_vector'):
        return np.dtype('int64')
    if node.operation == 'literal':
        return np.dtype('int32' if isinstance(node.value, bool) else 'float32' if isinstance(node.value, float) else 'uint64' if node.value > 2**63-1 else 'int64')
    if node.operation in _REDUCTIONS:
        if node.value is not None:
            return np.dtype(node.value)
        child = _expression_dtype(node.operands[0], inputs)
        return np.dtype('bool') if node.operation in ('any', 'all') else child if node.operation != 'sum' else np.dtype('float32' if child.kind == 'f' else 'uint64' if child.kind == 'u' else 'int64')
    if node.operation == 'load':
        types = (inputs[node.value].dtype, _expression_dtype(node.operands[3], inputs))
    else:
        types = tuple(_expression_dtype(child, inputs) for child in (node.operands[1:] if node.operation == 'select' else node.operands))
    if any(dtype.kind == 'f' for dtype in types):
        return np.dtype('float32')
    bits = max(max(32, dtype.itemsize*8) for dtype in types)
    unsigned = any(dtype.kind == 'u' and dtype.itemsize*8 == bits for dtype in types)
    return np.dtype(('uint' if unsigned else 'int') + str(bits))


# design/algorithm-sources.md#shared-associative-reductions
def _reduction_dtype(node, inputs, output):
    if node.value is not None:
        return np.dtype(node.value)
    if node.operation != 'sum':
        return _expression_dtype(node, inputs)
    return np.dtype(np.int64 if output.kind in 'ib' else np.uint64 if output.kind == 'u' else np.float32)


# design/algorithm-sources.md#counter-based-random-generation
def _random_helpers(metal):
    qualifier = 'inline' if metal else 'static inline'
    u32, u64 = ('uint', 'ulong') if metal else ('uint32_t', 'uint64_t')
    log, sqrt, sin, cos = tuple('precise::'+name if metal else name+'f' for name in ('log', 'sqrt', 'sin', 'cos'))
    return f"""typedef struct {{ {u32} word[4]; }} mesh_philox_words;
    {qualifier} mesh_philox_words mesh_philox({u32} c0,{u32} c1,{u32} c2,{u32} c3,{u32} k0,{u32} k1) {{
      mesh_philox_words result={{{{c0,c1,c2,c3}}}};
      for({u32} round=0;round<10;round++) {{
        {u64} a=({u64})result.word[0]*0xD2511F53u,b=({u64})result.word[2]*0xCD9E8D57u;
        mesh_philox_words next={{{{({u32})(b>>32)^result.word[1]^k0,({u32})b,({u32})(a>>32)^result.word[3]^k1,({u32})a}}}};
        result=next;k0+=0x9E3779B9u;k1+=0xBB67AE85u;
      }}
      return result;
    }}
    {qualifier} float mesh_random_normal({u32} k0,{u32} k1,{u64} ordinal) {{
      {u64} counter=ordinal>>1;
      mesh_philox_words bits=mesh_philox(({u32})counter,({u32})(counter>>32),0u,0u,k0,k1);
      float radius={sqrt}(-2.0f*{log}(((float)(bits.word[0]>>9)+0.5f)*0x1p-23f));
      float angle=6.283185307179586f*((float)(bits.word[1]>>9)*0x1p-23f);
      return radius*((ordinal&1u)?{sin}(angle):{cos}(angle));
    }}"""


# design/algorithm-sources.md#shared-elementary-functions
def _scalar_helpers(metal):
    qualifier = 'inline' if metal else 'static inline'
    suffix = '' if metal else 'f'
    log1p = 'mesh_log1p' if metal else 'log1pf'
    exponential = 'precise::exp' if metal else 'expf'
    remainder_function = 'precise::fmod' if metal else 'fmodf'
    helpers = [_random_helpers(metal)]
    if metal:
        helpers.append("""inline float mesh_log1p(float x) {
          float u=1.0f+x;
          return u==1.0f || x==INFINITY ? x : precise::log(u)*(x/(u-1.0f));
        }
        inline float mesh_expm1(float x) {
          return fabs(x)<0.5f ? x*(1.0f+x*(0.5f+x*(1.0f/6.0f+x*(1.0f/24.0f+x*(1.0f/120.0f+x*(1.0f/720.0f+x*(1.0f/5040.0f+x*(1.0f/40320.0f+x/362880.0f)))))))) : precise::exp(x)-1.0f;
        }
        inline float mesh_asinh(float x) {
          float a=fabs(x);
          return a<0.000244140625f ? x : copysign(a>4096.0f ? precise::log(a)+0.6931471805599453f : mesh_log1p(a+a*a/(1.0f+precise::sqrt(1.0f+a*a))),x);
        }""")
    helpers.append(f"""{qualifier} float mesh_logaddexp(float x,float y) {{
      if(x==y)return x+0.6931471805599453f;
      float d=x-y;
      return d>0.0f ? x+{log1p}({exponential}(-d)) : d<=0.0f ? y+{log1p}({exponential}(d)) : d;
    }}
    {qualifier} float mesh_floor_divide(float x,float y) {{
      float remainder={remainder_function}(x,y);
      if(y==0.0f)return x/y;
      float quotient=(x-remainder)/y;
      if(remainder!=0.0f && ((y<0.0f)!=(remainder<0.0f)))quotient-=1.0f;
      if(quotient==0.0f)return copysign{suffix}(0.0f,x/y);
      float result=floor{suffix}(quotient);
      return result+(quotient-result>0.5f ? 1.0f : 0.0f);
    }}""")
    return '\n'.join(helpers)


# design/algorithm-sources.md#fused-indexed-update-values
def _scalar_expression(node, args, metal, dtype=None):
    if node.operation in ('philox', 'random_normal'):
        u32, u64 = ('uint', 'ulong') if metal else ('uint32_t', 'uint64_t')
        types = (u32,)*6 if node.operation == 'philox' else (u32, u32, u64)
        call = ('mesh_philox' if node.operation == 'philox' else 'mesh_random_normal') + '(' + ','.join(
            f'(({scalar})({value}))' for scalar, value in zip(types, args)) + ')'
        return call + f'.word[{node.value}]' if node.operation == 'philox' else call
    if node.operation == 'isfinite':
        return f'isfinite((float)({args[0]}))' if dtype.kind == 'f' else '1'
    if node.operation == 'abs':
        if dtype.kind == 'f':
            return f'fabs{"" if metal else "f"}((float)({args[0]}))'
        if dtype.kind in 'ub':
            return args[0]
        bits = max(32, dtype.itemsize*8)
        return f'((int{bits}_t)(({args[0]})<0 ? ((uint{bits}_t)0-(uint{bits}_t)({args[0]})) : (uint{bits}_t)({args[0]})))'
    if node.operation == 'floor':
        return f'floor{"" if metal else "f"}((float)({args[0]}))' if dtype.kind == 'f' else args[0]
    if node.operation == 'floor_divide':
        return f'mesh_floor_divide((float)({args[0]}),(float)({args[1]}))' if dtype.kind == 'f' else _scalar_expression(_Expression('//'), args, metal, dtype)
    if node.operation in _REAL_FUNCTIONS and node.operation != 'rsqrt':
        name = ('mesh_' + node.operation if node.operation == 'logaddexp' or metal and node.operation in ('asinh', 'expm1', 'log1p') else
            ('pow' if node.operation == 'power' else node.operation) + ('' if metal else 'f'))
        if metal and node.operation in ('power', 'log', 'sqrt'):
            name = 'precise::' + name
        return name + '(' + ','.join(f'((float)({value}))' for value in args) + ')'
    if node.operation in ('maximum', 'minimum'):
        if dtype.kind == 'f':
            return f'{"fmax" if node.operation == "maximum" else "fmin"}{"" if metal else "f"}({args[0]},{args[1]})'
        return f'(({args[0]}){">" if node.operation == "maximum" else "<"}({args[1]})?({args[0]}):({args[1]}))'
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
        if not np.isfinite(node.value):
            return 'NAN' if np.isnan(node.value) else 'INFINITY' if node.value > 0 else '(-INFINITY)'
        return repr(float(node.value)) + 'f'
    if node.operation == 'domain':
        return args[0]
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
def _compiled_region(program, inputs, output, body, dynamic_first=None, *, row_inputs=None):
    import ctypes as C
    from . import check
    from ._native import View
    sources = []
    for metal in (False, True):
        scalar = {'f2': 'half' if metal else '_Float16', 'f4': 'float', 'i4': 'int32_t',
                  'u4': 'uint32_t', 'i8': 'int64_t', 'u8': 'uint64_t', 'u1': 'uint8_t', 'b1': 'bool'}
        lines = ['#include <metal_stdlib>\nusing namespace metal;\ntypedef uint uint32_t; typedef ulong uint64_t; typedef long int64_t; typedef int int32_t; typedef uchar uint8_t;' if metal else
                 '#include <stdint.h>\n#include <stdbool.h>\n#include <math.h>']
        lines.append(_scalar_helpers(metal))
        emitted = body(metal)
        preamble, statements = emitted if isinstance(emitted, tuple) else ('', emitted)
        lines.append(preamble)
        lines.append('#define PREFIX(x) simd_prefix_exclusive_sum(x)\n#define SUM(x) simd_sum(x)\n#define BARRIER threadgroup_barrier(mem_flags::mem_device)' if metal else
                     '#define PREFIX(x) 0u\n#define SUM(x) (x)\n#define BARRIER ((void)0)')
        lines.append(_METAL_EXPRESSION_HEAD if metal else
                     'void mesh_expression(const uintptr_t *buffers, const struct mesh_kernel_publication *publication) { const uint32_t lane=0; ' + _CPU_PUBLICATION_LOOP)
        lines.append(f'const uint32_t lanes={32 if metal else 1};')
        for index, ref in enumerate((*inputs, output)):
            if dynamic_first is not None and dynamic_first <= index < len(inputs):
                continue
            qualifier = ('device ' if metal else '') + ('const ' if index < len(inputs) else '')
            dtype = scalar[ref.dtype.kind + str(ref.dtype.itemsize)]
            lines.append(f'{qualifier}{dtype} *p{index}=({qualifier}{dtype} *)buffers[{index}];')
        lines.append(statements)
        lines.append('}' if metal else _CPU_PUBLICATION_END)
        sources.append('\n'.join(lines))
    function = program.native.algebra_trace_count(program.handle)
    domains = ((0, output.shape[0]),) if row_inputs is None else _source_row_regions(program, output)
    row_inputs = (0,) * len(inputs) if row_inputs is None else tuple(row_inputs)
    if len(row_inputs) != len(inputs):
        raise ValueError('Each compiled input requires its row access relation')
    for begin, rows in domains:
        check(program.native.algebra_source(program.handle, *(source.encode() for source in sources),
            (View * len(inputs))(*(ref.view for ref in inputs)), len(inputs), output.view,
            (C.c_uint8 * len(inputs))(*row_inputs), begin, rows, 0, output.shape[1]))
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
        return _literal(node.value[2]) if node.value[0] == 1 else (row if node.value[3] == 0 else feature) + node.value[2]
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
            # design/algorithm-sources.md#grouped-segment-reductions
            if metal and count > 1 and width < 32:
                features = max(1 << (width-1).bit_length(), 32 // (1 << (min(count, 32)-1).bit_length()))
                combine = ('total+=as_type<float>(simd_shuffle_xor(as_type<uint>(total),step));' if target.dtype.kind == 'f' else
                    'uint32_t lo=simd_shuffle_xor(uint32_t(total),step),hi=simd_shuffle_xor(uint32_t(total>>32),step); total+=(uint64_t(hi)<<32)|uint64_t(lo);')
                statements = f"""uint32_t c=lane%{features},part=lane/{features}; {accumulator} total=0;
                  if(c<{width})for(uint64_t k=(uint64_t){low}+part;k<{high};k+={32//features})total+={value};
                  for(uint32_t step={features};step<32;step*=2) {{{combine}}}
                  if(part==0&&c<{width}){output}[c*{target.view.column_stride}]=total;"""
            else:
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
            if node.value[0] not in (1, size if node.value[3] == 0 else features):
                raise ValueError('Indexed update vectors must broadcast to the update row and feature domain')
        elif node.operation == 'input':
            value_inputs.add(node.value)
        elif node.operation == 'load':
            indexed_inputs.add(node.value)
        elif node.operation not in ('literal', 'row', 'column', '+', '-', '*', '/', '<', '<=', '>', '>=', '==', '&', '|', 'select', 'cast', '//', '%', 'maximum', 'minimum') and node.operation not in _POINTWISE_FUNCTIONS:
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
    return node.operation in ('dot', 'cast', 'transpose', 'argsort', 'indexed_contract') or node.operation in _REDUCTIONS or any(_requires_regions(child) for child in node.operands)


# design/algorithm-sources.md#stable-indexed-ordering
def _indexed_interval_source(name, entries, first, metal):
    dtype = entries[0][1].dtype
    scalar = {'f2': 'half' if metal else '_Float16', 'f4': 'float', 'i4': 'int32_t', 'u4': 'uint32_t',
              'i8': 'int64_t', 'u8': 'uint64_t', 'u1': 'uint8_t', 'b1': 'bool'}[dtype.kind+str(dtype.itemsize)]
    result = 'float' if dtype.kind == 'f' else scalar
    address = 'device const ulong *' if metal else 'const uintptr_t *'
    qualifier, constant = ('inline', 'constant') if metal else ('static inline', 'static const')
    tables = []
    for suffix, values in (('begin', (start for start, ref in entries)),
                           ('end', (start+ref.shape[1] for start, ref in entries)),
                           ('rs', (ref.view.row_stride for start, ref in entries)),
                           ('cs', (ref.view.column_stride for start, ref in entries))):
        tables.append(f'{constant} uint64_t {name}_{suffix}[]={{'+','.join(map(str, values))+'};')
    load = _indexed_load_expression(entries[0][1], 'key', (scalar, (f'{name}_rs[block]', f'{name}_cs[block]')),
                                    ('row', f'index-{name}_begin[block]', 'true', '0'), metal)
    tables.append(f"""// design/algorithm-sources.md#stable-indexed-ordering
    {qualifier} {result} {name}({address} buffers,uint64_t row,uint64_t index) {{
      uint32_t low=0,high={len(entries)-1};
      while(low<high) {{ uint32_t middle=low+(high-low)/2;
        if(index>={name}_end[middle])low=middle+1;else high=middle; }}
      uint32_t block=low;
      {'device ' if metal else ''}const {scalar} *pkey=({'device ' if metal else ''}const {scalar} *)buffers[{first}+block];
      return {load};
    }}""")
    return '\n'.join(tables)


# design/algorithm-sources.md#stable-indexed-ordering
def _ordering_comparator(entries, metal):
    dtype = entries[0][1].dtype
    scalar = 'float' if dtype.kind == 'f' else 'uint64_t' if dtype.kind in 'ub' else 'int64_t'
    address = 'device const ulong *' if metal else 'const uintptr_t *'
    qualifier = 'inline' if metal else 'static inline'
    nan = 'if(isnan(x)!=isnan(y))return isnan(y);' if dtype.kind == 'f' else ''
    return _indexed_interval_source('mesh_sort_key', entries, 0, metal)+f"""
    // design/algorithm-sources.md#stable-indexed-ordering
    {qualifier} bool mesh_sort_before({address} buffers,uint64_t row,uint32_t a,uint32_t b) {{
      if(a==0xffffffffu)return false;if(b==0xffffffffu)return true;
      {scalar} x=mesh_sort_key(buffers,row,a),y=mesh_sort_key(buffers,row,b);
      {nan}
      if(x<y)return true;if(y<x)return false;return a<b;
    }}"""


# design/algorithm-sources.md#stable-indexed-ordering
def _ordering_run(program, source, first, target):
    width = source.shape[1]
    padded = 1 << (width-1).bit_length()

    # design/algorithm-sources.md#stable-indexed-ordering
    def body(metal):
        barrier = 'threadgroup_barrier(mem_flags::mem_threadgroup);' if metal else ''
        return _ordering_comparator(((first, source),), metal), f"""
        {'threadgroup ' if metal else ''}uint32_t work[{padded}];
        for(uint32_t i=lane;i<{padded};i+=lanes)work[i]=i<{width}?{first}u+i:0xffffffffu;
        {barrier}
        for(uint32_t span=2;span<={padded};span*=2) {{
          for(uint32_t step=span/2;step;step/=2) {{
            for(uint32_t i=lane;i<{padded};i+=lanes) {{ uint32_t j=i^step;
              if(j>i) {{ uint32_t a=work[i],b=work[j];
                if((i&span)?mesh_sort_before(buffers,r,a,b):mesh_sort_before(buffers,r,b,a)) {{work[i]=b;work[j]=a;}}
              }}
            }}
            {barrier}
          }}
        }}
        for(uint32_t i=lane;i<{width};i+=lanes)p1[r*{target.view.row_stride}+i*{target.view.column_stride}]=work[i];
        """
    _compiled_region(program, (source,), target, body, row_inputs=(1,))


# design/algorithm-sources.md#stable-indexed-ordering
def _ordering_merge(program, keys, left, right, diagonal, target):
    inputs = tuple(ref for _, ref in (*keys, *left, *right))
    left_width, right_width = sum(ref.shape[1] for _, ref in left), sum(ref.shape[1] for _, ref in right)

    # design/algorithm-sources.md#stable-indexed-ordering
    def body(metal):
        address = 'device const ulong *' if metal else 'const uintptr_t *'
        qualifier = 'inline' if metal else 'static inline'
        preamble = _ordering_comparator(keys, metal)+_indexed_interval_source('mesh_sort_left', left, len(keys), metal)+_indexed_interval_source('mesh_sort_right', right, len(keys)+len(left), metal)
        preamble += f"""
        // design/algorithm-sources.md#stable-indexed-ordering
        {qualifier} uint32_t mesh_sort_partition({address} buffers,uint64_t row,uint32_t diagonal,
            uint32_t alo,uint32_t ahi,uint32_t blo,uint32_t bhi) {{
          uint32_t low=diagonal>bhi?diagonal-bhi:0,high=diagonal-blo;
          if(low<alo)low=alo;if(high>ahi)high=ahi;
          while(low<high) {{ uint32_t a=low+(high-low)/2,b=diagonal-a;
            if(a<ahi&&b>blo&&mesh_sort_before(buffers,row,mesh_sort_left(buffers,row,a),mesh_sort_right(buffers,row,b-1)))low=a+1;
            else high=a;
          }}
          return low;
        }}"""
        barrier = 'threadgroup_barrier(mem_flags::mem_threadgroup);' if metal else ''
        return preamble, f"""
        {'threadgroup ' if metal else ''}uint32_t boundary[4];
        if(lane==0) {{
          boundary[0]=mesh_sort_partition(buffers,r,{diagonal}u,0,{left_width}u,0,{right_width}u);
          boundary[1]={diagonal}u-boundary[0];
          boundary[2]=mesh_sort_partition(buffers,r,{diagonal+target.shape[1]}u,0,{left_width}u,0,{right_width}u);
          boundary[3]={diagonal+target.shape[1]}u-boundary[2];
        }}
        {barrier}
        uint32_t chunk=({target.shape[1]}u+lanes-1)/lanes,start=lane*chunk,end=start+chunk;
        if(end>{target.shape[1]}u)end={target.shape[1]}u;
        if(start<end) {{
          uint32_t a=mesh_sort_partition(buffers,r,{diagonal}u+start,boundary[0],boundary[2],boundary[1],boundary[3]),b={diagonal}u+start-a;
          for(uint32_t i=start;i<end;i++) {{
            bool take_left=b==boundary[3]||(a<boundary[2]&&mesh_sort_before(buffers,r,mesh_sort_left(buffers,r,a),mesh_sort_right(buffers,r,b)));
            uint32_t value=take_left?mesh_sort_left(buffers,r,a++):mesh_sort_right(buffers,r,b++);
            p{len(inputs)}[r*{target.view.row_stride}+i*{target.view.column_stride}]=value;
          }}
        }}"""
    _compiled_region(program, inputs, target, body, row_inputs=(1,) * len(inputs))


# design/algorithm-sources.md#typed-integer-contractions
def _contraction_merge(left, right, dtype):
    if dtype.kind == 'b':
        return (left | right).astype(dtype)
    if dtype.kind in 'iu':
        return ((left & 0xffffffffffffffff) + (right & 0xffffffffffffffff)).astype(dtype)
    return left + right


# design/algorithm-sources.md#typed-integer-contractions
def _integral_contraction(program, left, right, target):
    # design/algorithm-sources.md#typed-integer-contractions
    def body(metal):
        a = f'p0[r*{left.view.row_stride}+k*{left.view.column_stride}]'
        b = f'p1[k*{right.view.row_stride}+c*{right.view.column_stride}]'
        scalar = 'uint64_t' if target.dtype.itemsize == 8 else 'uint32_t'
        term = f'(({a})&&({b}))' if target.dtype.kind == 'b' else f'(({scalar})({a}))*(({scalar})({b}))'
        operation = '|=' if target.dtype.kind == 'b' else '+='
        return f"""for(uint64_t c=lane;c<{target.shape[1]};c+=lanes) {{
          {scalar} total=0;
          for(uint64_t k=0;k<{left.shape[1]};k++) total{operation}{term};
          p2[r*{target.view.row_stride}+c*{target.view.column_stride}]=total;
        }}"""
    _compiled_region(program, (left, right), target, body, row_inputs=(1, 0))


# design/algorithm-sources.md#shared-contraction-lowering
def _bind_operation(program, operation, inputs, target):
    from . import check
    from ._native import View
    check(program.native.algebra_bind(program.handle, operation.op, inputs[0].view,
        inputs[1].view if len(inputs) == 2 else View(), target.view, operation.alpha, operation.beta))


# design/algorithm-sources.md#indexed-range-generation
def _expression_layout(node, sources, whole, layouts):
    from math import gcd
    # design/algorithm-sources.md#indexed-range-generation
    def layout(child):
        return _expression_layout(child, sources, whole, layouts)

    if node in layouts:
        return layouts[node]
    if node.operation == 'input':
        source = sources[node.value]
        shape = source.shape
        result = shape, (whole[node.value],) * 2, source.block_shape if whole[node.value] else shape
    elif node.operation == 'index_vector':
        shape = tuple(node.value[0] if axis == node.value[3] else 1 for axis in range(2))
        steps = tuple(node.value[1] if axis == node.value[3] else 1 for axis in range(2))
        result = shape, (False, False), steps
    elif node.operation in ('literal', 'program_id', 'row', 'column'):
        result = (1, 1), (False, False), (1, 1)
    elif node.operation == 'domain':
        result = node.value
    elif node.operation == 'indexed_contract':
        result = node.value[1]
    elif node.operation in ('cast', 'argsort'):
        result = layout(node.operands[0])
    elif node.operation == 'transpose':
        result = tuple(value[::-1] for value in layout(node.operands[0]))
    elif node.operation == 'dot':
        left, right = map(layout, node.operands)
        if left[0][1] != right[0][0]:
            raise ValueError('Contraction inner dimensions differ')
        result = (left[0][0], right[0][1]), (left[1][0], right[1][1]), (left[2][0], right[2][1])
    elif node.operation in _REDUCTIONS:
        child = layout(node.operands[0])
        result = (child[0][0], 1), (child[1][0], False), (child[2][0], 1)
    elif node.operation in _POINTWISE_OPERATIONS or node.operation == 'load':
        children = tuple(map(layout, node.operands))
        shape = tuple(next((child[0][axis] for child in children if child[0][axis] != 1), 1) for axis in range(2))
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
            steps.append(max(1, step))
        result = shape, tuple(global_axes), tuple(steps)
    else:
        raise ValueError('Computed contraction operands require pointwise expressions or contractions')
    layouts[node] = result
    return result


class _ExpressionRegions:
    # design/algorithm-sources.md#shared-contraction-lowering
    def __init__(self, program, specs, coordinate, cache):
        self.program, self.coordinate, self.cache = program, coordinate, cache
        self.sources = tuple(spec.resolve(coordinate) for spec in specs)
        self.whole = tuple(spec.block_shape is None for spec in specs)
        self.layouts = {}
        self.reduction_uses = {}

    # design/algorithm-sources.md#indexed-range-generation
    def layout(self, node):
        return _expression_layout(node, self.sources, self.whole, self.layouts)

    # design/algorithm-sources.md#shared-contraction-lowering
    def key(self, node, origin, shape):
        used = set()

        # design/algorithm-sources.md#shared-contraction-lowering
        def visit(value):
            if value.operation in ('input', 'load'):
                used.add(value.value)
            if value.operation == 'logical_load':
                used.add(value.value[0])
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
        if node.operation in _REDUCTIONS:
            return self.reduction(node, origin[0], shape[0], _expression_dtype(node, self.sources))[0]
        if node.operation == 'argsort':
            return self.ordering_panel(node, origin, shape)
        if node.operation == 'indexed_contract':
            return self.indexed_contraction_panel(node, origin, shape)
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

    # design/algorithm-sources.md#composable-indexed-contractions
    def indexed_contraction_panel(self, node, origin, shape, direct=None):
        key = ('indexed_contraction', self.key(node, origin, shape))
        if key not in self.cache:
            target = direct if direct is not None and direct.dtype == np.dtype('float32') else self.temporary(shape)

            # design/algorithm-sources.md#composable-indexed-contractions
            def shift(value):
                if value.operation == 'index_vector' and value.value[3] == 0 and value.value[0] != 1:
                    return _Expression('index_vector', value=(*value.value[:2], value.value[2]+origin[0], 0))
                return _Expression(value.operation, tuple(shift(child) for child in value.operands), value.value)

            product = _Expression('*', tuple(shift(child) for child in node.operands))
            transposed = _Expression('transpose', (_Expression('sum', (product,)),))
            original = _resolve_logical(node.value[0], self.sources)
            plan = _ReductionPlan.create(self.reduction_regions(original))
            if shape[1] != 1:
                self.emit(original, origin, shape, target)
                binding = None
            else:
                binding = _lower_indexed_product(self, transposed, target.T, plan)
                if not binding:
                    target, identity = self.reduction(original, origin[0], shape[0], np.dtype('float32'), target, plan)
                    binding = self.program._plan_bindings[identity]
            self.cache[key] = target, binding['id'] if binding is not None else None
        result, identity = self.cache[key]
        if identity is not None:
            self.program._plan_use(identity, result, direct)
        return result

    # design/algorithm-sources.md#stable-indexed-ordering
    def ordering_panel(self, node, origin, shape):
        child = node.operands[0]
        layout = self.layout(child)
        width = layout[0][1]
        if width > 0xffffffff:
            raise ValueError('Ordering axes must fit the uint32 ordinal domain')
        key = ('ordering', self.key(node, (origin[0], 0), (shape[0], width)))
        if key not in self.cache:
            runs = []
            step = layout[2][1]
            for backing in range(0, width, step):
                for first in range(backing, min(width, backing+step), 128):
                    columns = min(128, width-first, backing+step-first)
                    source = self.panel(child, (origin[0], first), (shape[0], columns))
                    target = self.temporary((shape[0], columns), np.uint32)
                    _ordering_run(self.program, source, first, target)
                    runs.append((((first, source),), ((0, target),)))
            while len(runs) > 1:
                merged = []
                for index in range(0, len(runs), 2):
                    if index+1 == len(runs):
                        merged.append(runs[index])
                        continue
                    (left_keys, left), (right_keys, right) = runs[index:index+2]
                    keys = left_keys+right_keys
                    count = sum(ref.shape[1] for _, ref in (*left, *right))
                    outputs = []
                    for first in range(0, count, 128):
                        target = self.temporary((shape[0], min(128, count-first)), np.uint32)
                        _ordering_merge(self.program, keys, left, right, first, target)
                        outputs.append((first, target))
                    merged.append((keys, tuple(outputs)))
                runs = merged
            self.cache[key] = runs[0][1]
        refs = self.cache[key]
        for first, ref in refs:
            if first <= origin[1] and origin[1]+shape[1] <= first+ref.shape[1]:
                return ref.slice(0, origin[1]-first, *shape)
        key = ('ordering_panel', self.key(node, origin, shape))
        if key not in self.cache:
            target = self.temporary(shape, np.uint32)
            row, column = indices()
            selected = tuple((first, ref) for first, ref in refs if first < origin[1]+shape[1] and origin[1] < first+ref.shape[1])
            table = _StaticTable(tuple(ref for _, ref in selected), tuple(first//128 for first, _ in selected),
                                 (shape[0], width), (shape[0], 128), (1, (width+127)//128), np.dtype('uint32'))
            symbol, = arguments(1)
            _ExpressionKernel((symbol.at(row, column+origin[1]),)).bind(self.program, (table,), (target,), self.coordinate)
            self.cache[key] = target
        return self.cache[key]

    # design/algorithm-sources.md#shared-contraction-lowering
    def parts(self, node, origin, shape, direct=None):
        from math import gcd
        key = ('parts', self.key(node, origin, shape))
        if key in self.cache:
            return self.cache[key]
        dtype = _expression_dtype(node, self.sources)
        left, right = node.operands
        left_layout, right_layout = self.layout(left), self.layout(right)
        inner = left_layout[0][1]
        if inner != right_layout[0][0]:
            raise ValueError('Contraction inner dimensions differ')
        if inner == 0:
            target = direct if direct is not None and direct.dtype == dtype else self.temporary(shape, dtype)
            _ExpressionKernel((_literal(0),)).bind(self.program, (), (target,))
            self.cache[key] = (target,)
            return (target,)
        tile = min(node.value, inner)
        for layout, axis in ((left_layout, 1), (right_layout, 0)):
            if layout[2][axis] < layout[0][axis]:
                tile = gcd(tile, layout[2][axis])
        operands = tuple(child.astype('float32') if dtype.kind == 'f' and _expression_dtype(child, self.sources).kind in 'iub' else child
                         for child in (left, right))
        parts = []
        for start in range(0, inner, tile):
            length = min(tile, inner - start)
            left_panel = self.panel(operands[0], (origin[0], start), (shape[0], length))
            right_panel = self.panel(operands[1], (start, origin[1]), (length, shape[1]))
            reverse = left_panel.dtype == np.dtype('float16') and right_panel.dtype == np.dtype('float32')
            if reverse:
                destination = self.temporary(shape[::-1]).T
            else:
                destination = direct if direct is not None and tile == inner and direct.dtype == dtype else self.temporary(shape, dtype)
            if dtype.kind in 'iub':
                _integral_contraction(self.program, left_panel, right_panel, destination)
            else:
                _bind_operation(self.program, matmul, (left_panel, right_panel), destination)
            parts.append(destination)
        while len(parts) > 2:
            reduced = []
            for index in range(0, len(parts), 2):
                if index + 1 == len(parts):
                    reduced.append(parts[index])
                else:
                    destination = self.temporary(shape, dtype)
                    self.publish(parts[index:index + 2], destination)
                    reduced.append(destination)
            parts = reduced
        self.cache[key] = tuple(parts)
        return self.cache[key]

    # design/algorithm-sources.md#composable-indexed-contractions
    def reduction_regions(self, node):
        layout = self.layout(node.operands[0])
        width, tile = layout[0][1], layout[2][1]
        return tuple((column, min(tile, width-column)) for column in range(0, width, tile))

    # design/algorithm-sources.md#shared-associative-reductions
    def reduction(self, node, row, rows, dtype, direct=None, plan=None):
        key = ('reduction', self.key(node, (row, 0), (rows, 1)), dtype.str)
        if key in self.cache:
            result, identity = self.cache[key]
            self.program._plan_use(identity, result, direct)
            return result, identity
        child = node.operands[0]
        layout = self.layout(child)
        width, tile = layout[0][1], layout[2][1]
        if width == 0:
            identity = (1 if node.operation == 'all' else 0) if node.operation in ('sum', 'any', 'all') else (
                (-np.inf if node.operation == 'max' else np.inf) if dtype.kind == 'f' else
                (0 if node.operation == 'max' else 1) if dtype.kind == 'b' else
                np.iinfo(dtype).min if node.operation == 'max' else np.iinfo(dtype).max)
            target = direct if direct is not None and direct.dtype == dtype else self.temporary((rows, 1), dtype)
            binding = self.program._plan_binding('reduction', (target,), 0)
            with self.program._plan_operation(binding, 'identity', (), 0, reduction=node.operation):
                _ExpressionKernel((_literal(identity),)).bind(self.program, (), (target,))
            self.program._plan_use(binding['id'], target, direct)
            self.cache[key] = target, binding['id']
            return self.cache[key]
        plan = plan if plan is not None else _ReductionPlan.create(self.reduction_regions(node))
        parts = {}
        binding = self.program._plan_binding('reduction', (), plan.root)
        for index, (column, length) in enumerate(plan.regions):
            target = direct if direct is not None and index == plan.root and direct.dtype == dtype else self.temporary((rows, 1), dtype)
            binding['slots'].append(self.program._plan_view(target))
            with self.program._plan_operation(binding, 'region', (), index,
                    origin=(row, column), shape=(rows, length), reduction=node.operation):
                self.emit(child, (row, column), (rows, length), target, reduce=node.operation)
            parts[index] = target
        for left_index, right_index, output in plan.merges:
            target = direct if direct is not None and output == plan.root and direct.dtype == dtype else self.temporary((rows, 1), dtype)
            inputs = (parts[left_index], parts[right_index])
            binding['slots'].append(self.program._plan_view(target))
            with self.program._plan_operation(binding, 'merge', (left_index, right_index), output, reduction=node.operation):
                if node.operation == 'sum' and dtype.kind == 'f':
                    _bind_operation(self.program, add, inputs, target)
                else:
                    left, right = arguments(2)
                    bits = 0xffffffffffffffff
                    merged = ((left & bits)+(right & bits) if node.operation == 'sum' else
                        left | right if node.operation == 'any' else left & right if node.operation == 'all' else
                        _Expression('maximum' if node.operation == 'max' else 'minimum', (left, right)))
                    _ExpressionKernel((merged,)).bind(self.program, inputs, (target,), self.coordinate)
            parts[output] = target
        self.cache[key] = parts[plan.root], binding['id']
        self.program._plan_use(binding['id'], parts[plan.root], direct)
        return self.cache[key]

    # design/algorithm-sources.md#shared-contraction-lowering
    def publish(self, parts, target):
        dtype = parts[0].dtype
        if dtype.kind in 'iub':
            if len(parts) == 1 and parts[0] is target:
                return
            terms = arguments(len(parts))
            value = terms[0] if len(parts) == 1 else _contraction_merge(*terms, dtype)
            _ExpressionKernel((value,)).bind(self.program, parts, (target,), self.coordinate)
            return
        if len(parts) == 2:
            destination = target if target.dtype == np.dtype('float32') else self.temporary(target.shape)
            _bind_operation(self.program, add, parts, destination)
            parts = (destination,)
        if parts[0] is not target:
            if target.dtype.kind in 'iub':
                symbol, = arguments(1)
                _ExpressionKernel((symbol,)).bind(self.program, parts, (target,), self.coordinate)
            else:
                _bind_operation(self.program, affine(), parts, target)

    # design/algorithm-sources.md#in-operation-publication
    def inline_reduction(self, node, expression, origin, shape, dtype, external):
        import ctypes as C
        from . import check
        from ._native import View
        layout = self.layout(node.operands[0])
        row = 0 if layout[0][0] == 1 or (external and not layout[1][0]) else origin[0]
        rows = 1 if layout[0][0] == 1 else shape[0]
        key = ('reduction', self.key(node, (row, 0), (rows, 1)), dtype.str)
        if (self.reduction_uses.get(key) != 1 or key in self.cache or
                layout[0][1] != shape[1] or layout[2][1] < layout[0][1] or rows != shape[0]):
            return None

        # design/algorithm-sources.md#in-operation-publication
        def references(value, refs, where_origin, mapped):
            if value == node:
                return references(node.operands[0], refs, (row, 0), False)
            if value.operation not in ('input', 'literal', 'domain', *_POINTWISE_OPERATIONS):
                return False
            if value.operation == 'input':
                identity = value.value, where_origin, mapped
                if identity not in refs:
                    source = self.sources[value.value]
                    where = tuple(0 if source.shape[axis] == 1 else where_origin[axis] for axis in range(2))
                    extent = tuple(1 if source.shape[axis] == 1 else shape[axis] for axis in range(2))
                    try:
                        refs[identity] = source if mapped and not self.whole[value.value] else self.panel(value, where, extent)
                    except ValueError:
                        return False
            return all(references(child, refs, where_origin, mapped) for child in value.operands)

        # design/algorithm-sources.md#in-operation-publication
        def pages(refs):
            result = set()
            for ref in refs.values():
                if (ref.view.tensor, ref.view.extent) in self.program._constant_extents:
                    continue
                count = C.c_size_t()
                check(self.program.native.algebra_view_pages(self.program.handle, ref.view, None, 0, C.byref(count)))
                views = (View * count.value)()
                check(self.program.native.algebra_view_pages(self.program.handle, ref.view, views, count.value, C.byref(count)))
                result.update((view.tensor, view.extent, view.offset) for view in views)
            return result

        reduced, consumed = {}, {}
        if (not references(node.operands[0], reduced, (row, 0), False) or
                not references(expression, consumed, origin, external) or not pages(consumed) <= pages(reduced)):
            return None
        return {identity[0]: ref for identity, ref in reduced.items()}

    # design/algorithm-sources.md#shared-associative-reductions
    def emit(self, value, origin, shape, target, external=False, reduce=False):
        inputs, replacements = [], {}

        # design/algorithm-sources.md#shared-contraction-lowering
        def reference(key, refs):
            if key not in replacements:
                terms = tuple(_Expression('input', value=len(inputs) + index) for index in range(len(refs)))
                inputs.extend(refs)
                replacements[key] = terms[0] if len(terms) == 1 else _contraction_merge(*terms, refs[0].dtype)
            return replacements[key]

        # design/algorithm-sources.md#shared-contraction-lowering
        def lower(node, accumulation=target.dtype):
            if node.operation == 'domain':
                if 0 in node.value[0]:
                    raise ValueError('Empty expression domains cannot produce a nonempty region')
                return lower(node.operands[0], _expression_dtype(node.operands[0], self.sources))
            if node.operation == 'cast':
                child = node.operands[0]
                return _Expression('cast', (lower(child, _expression_dtype(child, self.sources)),), node.value)
            if node.operation in _REDUCTIONS:
                layout = self.layout(node)
                row = 0 if layout[0][0] == 1 or (external and not layout[1][0]) else origin[0]
                rows = 1 if layout[0][0] == 1 else shape[0]
                dtype = _reduction_dtype(node, self.sources, accumulation)
                inline = self.inline_reduction(node, value, origin, shape, dtype, external)
                if inline is not None:
                    # design/algorithm-sources.md#in-operation-publication
                    def substitute(part):
                        if part.operation == 'input':
                            return reference(('inline_reduction', node, part.value), (inline[part.value],))
                        if part.operation == 'domain':
                            return substitute(part.operands[0])
                        return _Expression(part.operation, tuple(substitute(child) for child in part.operands), part.value)

                    child = _Expression('domain', (substitute(node.operands[0]),),
                        (shape, (False, False), shape))
                    return _Expression(node.operation, (child,), dtype.str)
                return reference(('reduction', node, row, rows, dtype.str), (self.reduction(node, row, rows, dtype)[0],))
            if node.operation == 'indexed_contract':
                if accumulation.kind != 'f':
                    return lower(_resolve_logical(node.value[0], self.sources), accumulation)
                layout = self.layout(node)
                row = 0 if layout[0][0] == 1 or (external and not layout[1][0]) else origin[0]
                rows = 1 if layout[0][0] == 1 else shape[0]
                return reference(('indexed_contract', node, row, rows),
                                 (self.indexed_contraction_panel(node, (row, 0), (rows, 1)),))
            if node.operation == 'argsort':
                layout = self.layout(node)
                row = 0 if layout[0][0] == 1 or (external and not layout[1][0]) else origin[0]
                rows = 1 if layout[0][0] == 1 else shape[0]
                columns = 1 if layout[0][1] == 1 else shape[1]
                column = 0 if layout[0][1] == 1 else origin[1]
                return reference(('argsort', node, row, column, rows, columns),
                                 (self.ordering_panel(node, (row, column), (rows, columns)),))
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
                return _Expression('load', tuple(lower(child, _expression_dtype(child, self.sources)) for child in node.operands), symbol.value)
            if node.operation == 'index_vector':
                axis = node.value[3]
                return node if external or node.value[0] == 1 else _Expression('index_vector', value=(shape[axis], shape[axis], node.value[2] + origin[axis], axis))
            if node.operation in ('row', 'column') and not external:
                axis = 0 if node.operation == 'row' else 1
                return node + origin[axis]
            if node.operation == 'indexed_add':
                raise ValueError('Indexed addition requires an output root')
            return _Expression(node.operation, tuple(lower(child, _expression_dtype(child, self.sources)) for child in node.operands), node.value)

        lowered = lower(value, _expression_dtype(value, self.sources) if reduce else target.dtype)
        if reduce:
            lowered = _Expression(reduce, (lowered,))
        _ExpressionKernel((lowered,)).bind(self.program, tuple(inputs), (target,), self.coordinate)


# design/algorithm-sources.md#selected-native-contractions
def _panel_affine(node, selected, choice):
    if node == selected:
        return choice, 0, 0
    if node.operation == 'literal' and isinstance(node.value, (int, bool)):
        return int(node.value), 0, 0
    if node.operation == 'cast' and node.operands[0].operation == 'literal' and np.dtype(node.value).kind in 'iub':
        zero = np.zeros(1, dtype=np.int64)
        value = _static_value(node, (), zero, zero, ())
        return None if value is None else (int(value[0]), 0, 0)
    if node.operation == 'index_vector':
        return node.value[2], int(node.value[3] == 0 and node.value[0] > 1), int(node.value[3] == 1 and node.value[0] > 1)
    if node.operation not in ('+', '-', '*'):
        return None
    left, right = (_panel_affine(child, selected, choice) for child in node.operands)
    if left is None or right is None:
        return None
    if node.operation == '*':
        if left[1:] == (0, 0):
            return tuple(left[0] * value for value in right)
        if right[1:] == (0, 0):
            return tuple(right[0] * value for value in left)
        return None
    return tuple(a + (b if node.operation == '+' else -b) for a, b in zip(left, right))


# design/algorithm-sources.md#selected-native-contractions
def _panel_geometry(source, ordinal, row_step, column_step, rows, columns):
    row, column = divmod(ordinal, source.shape[1])
    if min(row, column, row_step, column_step) < 0 or row >= source.shape[0]:
        return None
    if hasattr(source, 'blocks'):
        block_row, block_column = row // source.block_shape[0], column // source.block_shape[1]
        backing = source[block_row, block_column]
        row, column = row % source.block_shape[0], column % source.block_shape[1]
    else:
        backing = source
    dr, dc = divmod(row_step, source.shape[1]), divmod(column_step, source.shape[1])
    remaining = tuple(size - 1 - origin - (columns - 1) * step
                      for size, origin, step in zip(backing.shape, (row, column), dc))
    if min(remaining) < 0:
        return None
    capacity = min((space // step + 1 for space, step in zip(remaining, dr) if step), default=rows)
    if capacity < rows:
        return None
    strides = tuple(sum(step * stride for step, stride in zip(delta,
        (backing.view.row_stride, backing.view.column_stride))) or 1 for delta in (dr, dc))
    if 1 not in strides:
        return None
    offset = backing.view.offset + row * backing.view.row_stride + column * backing.view.column_stride
    return backing, offset, strides, capacity


# design/algorithm-sources.md#indexed-contraction-plans
@dataclass(frozen=True)
class _ReductionPlan:
    regions: tuple
    merges: tuple
    root: int

    # design/algorithm-sources.md#indexed-contraction-plans
    @classmethod
    def create(cls, regions):
        regions = tuple(regions)
        parts, merges = list(range(len(regions))), []
        while len(parts) > 1:
            following = []
            for index in range(0, len(parts), 2):
                if index+1 == len(parts):
                    following.append(parts[index])
                else:
                    output = len(regions)+len(merges)
                    merges.append((parts[index], parts[index+1], output))
                    following.append(output)
            parts = following
        return cls(regions, tuple(merges), parts[0] if parts else -1)


# design/algorithm-sources.md#indexed-contraction-plans
@dataclass(frozen=True)
class _IndexedProductPlan:
    segments: tuple
    storage: tuple
    operations: tuple

    # design/algorithm-sources.md#indexed-contraction-plans
    @classmethod
    def create(cls, segments, target, target_pages, reverse):
        segments = tuple((first, columns, length, tuple(prepared))
            for first, columns, length, prepared in segments)
        storage, operations, groups = [None], [], {}

        # design/algorithm-sources.md#indexed-contraction-plans
        def temporary(columns, transposed=False):
            storage.append(((columns, 1) if transposed else (1, columns), transposed))
            return len(storage)-1

        for index, (first, columns, _, _) in enumerate(segments):
            output = (0 if len(segments) == 1 and target_pages == 1 and
                target.dtype == np.dtype('float32') and not reverse else temporary(columns, reverse))
            operations.append(('contract', (), output, index))
            groups.setdefault((first, columns), []).append(output)
        results = []
        for (first, columns), parts in groups.items():
            tree = _ReductionPlan.create(parts)
            slots = dict(enumerate(parts))
            for left, right, output in tree.merges:
                destination = (0 if len(groups) == 1 and output == tree.root and
                    target.dtype == np.dtype('float32') else temporary(columns))
                operations.append(('add', (slots[left], slots[right]), destination, None))
                slots[output] = destination
            result = slots[tree.root]
            if len(groups) == 1 and result != 0:
                operations.append(('copy', (result,), 0, None))
                result = 0
            results.append((first, result))
        if len(groups) > 1:
            operations.append(('assemble', tuple(slot for _, slot in results), 0,
                tuple(first for first, _ in results)))
        return cls(segments, tuple(storage), tuple(operations))


# design/algorithm-sources.md#selected-native-contractions
def _lower_indexed_product(lowering, value, target, compiled_plan):
    import ctypes as C
    from . import Ref, check
    from ._native import View
    if value.operation != 'transpose' or value.operands[0].operation != 'sum' or target.shape[0] != 1 or target.dtype.kind != 'f':
        return False
    product = value.operands[0].operands[0]
    if product.operation != '*' or any(node.operation != 'logical_load' for node in product.operands):
        return False
    operands = product.operands
    selected, choices, vectors = None, None, set()
    for node in operands:
        source = lowering.sources[node.value[0]]
        if source.dtype.kind != 'f' or node.operands[-2] != _literal(True) or node.operands[-1] != _literal(0):
            return False
        _resolve_logical(node, lowering.sources)
        if -1 in node.value[1]:
            return False
        for coordinate, size in zip(node.operands[:-2], node.value[1]):
            pending = [coordinate]
            has_vector = False
            while pending:
                part = pending.pop()
                if part.operation == 'index_vector':
                    has_vector = True
                    if part.value[3] == 1:
                        vectors.add(part.value[:2])
                pending.extend(part.operands)
            if _panel_affine(coordinate, None, 0) is None:
                if has_vector or selected is not None and selected != coordinate:
                    return False
                pending = [coordinate]
                while pending:
                    part = pending.pop()
                    if part.operation in ('row', 'column') or part.operation == 'input' and lowering.sources[part.value].shape != (1, 1):
                        return False
                    pending.extend(part.operands)
                selected, choices = coordinate, size if choices is None else min(choices, size)
    if len(vectors) != 1 or (selected is not None and _expression_dtype(_resolve_logical(selected, lowering.sources), lowering.sources).kind not in 'iub'):
        return False
    choices = 1 if selected is None else choices
    inner, tile = next(iter(vectors))
    if inner == 0:
        return False
    feature = target.shape[1]
    flattened = []
    for node in operands:
        plans = []
        for choice in range(choices):
            flat = (0, 0, 0)
            for coordinate, size in zip(node.operands[:-2], node.value[1]):
                affine = _panel_affine(coordinate, selected, choice)
                if affine is None or min(affine) < 0 or affine[0] + affine[1]*(feature-1) + affine[2]*(inner-1) >= size:
                    return False
                flat = tuple(value * size + addition for value, addition in zip(flat, affine))
            plans.append(flat)
        flattened.append(plans)
    if any(plan[1] for plan in flattened[0]):
        operands, flattened = operands[::-1], flattened[::-1]
    if any(plan[1] for plan in flattened[0]) or len(set(flattened[0])) != 1 or selected in operands[0].operands[:-2]:
        return False
    sources = tuple(lowering.sources[node.value[0]] for node in operands)
    program = lowering.program
    quantum = program.native.algebra_publication_bytes(program.handle)
    if target.shape[1] * target.dtype.itemsize > quantum:
        return False
    segments, first = [], 0
    while first < feature:
        columns = min(feature-first, quantum // 4)
        for right in flattened[1]:
            geometry = _panel_geometry(sources[1], right[0]+first*right[1], right[1], 0, 1, 1)
            if geometry is None:
                return False
            columns = min(columns, geometry[3])
        start = 0
        while start < inner:
            length = min(tile, inner-start)
            prepared = []
            for left, right in zip(*flattened):
                a = _panel_geometry(sources[0], left[0]+start*left[2], left[2], 0, 1, 1)
                b = _panel_geometry(sources[1], right[0]+start*right[2]+first*right[1], right[2], right[1], 1, columns)
                if a is None or b is None:
                    return False
                length = min(length, a[3], b[3])
                prepared.append((a, b))
            segments.append((first, columns, length, prepared))
            start += length
        first += columns
    target_pages = C.c_size_t()
    check(program.native.algebra_view_pages(program.handle, target.view, None, 0, C.byref(target_pages)))
    reverse = sources[0].dtype == np.dtype('float16') and sources[1].dtype == np.dtype('float32')
    native_plan = _IndexedProductPlan.create(segments, target, target_pages.value, reverse)
    if selected is None and len(native_plan.operations) > max(1, len(compiled_plan.regions)+len(compiled_plan.merges)):
        return False
    selector = None
    selector_functions = ()
    if selected is not None:
        selection_key = ('selected_plan', lowering.key(_resolve_logical(selected, lowering.sources), (0, 0), (1, 1)), choices)
        if selection_key not in lowering.cache:
            selector = program.tensor((1, 1), dtype=np.uint32)[0, 0]
            plan = select((selected >= 0) & (selected < choices), selected, choices)
            selector_binding = program._plan_binding('selector', (selector,), 0)
            with program._plan_operation(selector_binding, 'selector', (), 0) as selector_entry:
                lowering.emit(_resolve_logical(plan, lowering.sources), (0, 0), (1, 1), selector)
            lowering.cache['selector_binding', selection_key] = selector_binding['id']
            lowering.cache[selection_key] = selector
        selector = lowering.cache[selection_key]
        selector_binding = program._plan_bindings[lowering.cache['selector_binding', selection_key]]
        selector_functions = selector_binding['operations'][0]['functions']
    storage = [target]
    for shape, transposed in native_plan.storage[1:]:
        ref = lowering.temporary(shape)
        storage.append(ref.T if transposed else ref)
    binding = program._plan_binding('indexed_contraction', storage, 0)
    if selector is not None:
        binding['selector'] = dict(view=program._plan_view(selector), functions=selector_functions,
            binding=selector_binding['id'])
        program._plan_use(selector_binding['id'], selector)
    for operation, inputs, output, detail in native_plan.operations:
        with program._plan_operation(binding, operation, inputs, output) as entry:
            destination = storage[output]
            if operation == 'add':
                _bind_operation(program, add, tuple(storage[index] for index in inputs), destination)
                continue
            if operation == 'copy':
                lowering.publish((storage[inputs[0]],), destination)
                continue
            if operation == 'assemble':
                results = tuple(storage[index] for index in inputs)
                entries = tuple(zip(detail, results))
                entry['offsets'] = detail

                # design/algorithm-sources.md#indexed-contraction-plans
                def assemble(metal):
                    source = _indexed_interval_source('mesh_contraction_part', entries, 0, metal)
                    return source, f"""for(uint64_t column=lane;column<{destination.shape[1]};column+=lanes)
                  p{len(results)}[r*{destination.view.row_stride}+column*{destination.view.column_stride}]=mesh_contraction_part(buffers,r,column);"""

                _compiled_region(program, results, destination, assemble, row_inputs=(1,) * len(results))
                continue
            first, columns, length, prepared = native_plan.segments[detail]
            entry['segment'] = dict(first=first, columns=columns, inner=length)
            dependencies, positions, page_sets, left_views, right_views, selected_dependencies = [], {}, {}, [], [], []

            # design/algorithm-sources.md#selected-native-contractions
            def retain(geometry, shape, transpose=False):
                backing, offset, strides, _ = geometry
                view = View.from_buffer_copy(backing.view)
                view.offset, view.rows, view.columns = offset, *shape
                view.row_stride, view.column_stride = strides
                if view.rows == 1 and view.column_stride == 1:
                    view.row_stride = view.columns
                if view.columns == 1 and view.row_stride == 1:
                    view.column_stride = view.rows
                ref = Ref(program, view, backing.dtype)
                ref = ref.T if transpose else ref
                if selected is None:
                    return ref, ()
                identity = tuple(getattr(ref.view, field) for field in
                    ('tensor', 'extent', 'offset', 'rows', 'columns', 'row_stride', 'column_stride'))
                if identity not in page_sets:
                    count = C.c_size_t()
                    check(program.native.algebra_view_pages(program.handle, ref.view, None, 0, C.byref(count)))
                    pages = (View * count.value)()
                    check(program.native.algebra_view_pages(program.handle, ref.view, pages, count.value, C.byref(count)))
                    selected_pages = []
                    for page in pages:
                        key = page.tensor, page.extent, page.offset
                        if key not in positions:
                            positions[key] = len(dependencies)
                            dependencies.append(Ref(program, page, ref.dtype))
                        selected_pages.append(positions[key])
                    page_sets[identity] = tuple(selected_pages)
                return ref, page_sets[identity]

            for a, b in prepared:
                left, left_position = retain(a, (length, 1), True)
                right, right_position = retain(b, (length, columns))
                left_views.append(left)
                right_views.append(right)
                selected_dependencies.append(tuple(dict.fromkeys((*left_position, *right_position))))
            entry['choices'] = tuple(dict(left=program._plan_view(left), right=program._plan_view(right))
                for left, right in zip(left_views, right_views))
            if selected is None:
                _bind_operation(program, matmul, (left_views[0], right_views[0]), destination)
                continue
            zero_key = ('selected_zero', length, columns, sources[1].dtype.str)
            if zero_key not in lowering.cache:
                zero = program.tensor((length, columns), dtype=sources[1].dtype)[0, 0]
                program.constant(zero, np.zeros(zero.shape, dtype=zero.dtype))
                lowering.cache[zero_key] = zero
            zero = lowering.cache[zero_key]
            zero, zero_pages = retain((zero, zero.view.offset,
                (zero.view.row_stride, zero.view.column_stride), length), zero.shape)
            _, left_pages = retain(prepared[0][0], (length, 1), True)
            left_views.append(left_views[0])
            right_views.append(zero)
            selected_dependencies.append(tuple(dict.fromkeys((*left_pages, *zero_pages))))
            function = C.c_size_t()
            check(program.native.algebra_contract_select(program.handle, selector.view,
                (View * len(left_views))(*(ref.view for ref in left_views)),
                (View * len(right_views))(*(ref.view for ref in right_views)), len(left_views),
                (View * len(dependencies))(*(ref.view for ref in dependencies)), len(dependencies),
                destination.view, 1, C.byref(function)))
            entry['choices'] = tuple(dict(left=program._plan_view(left), right=program._plan_view(right), pages=pages)
                for left, right, pages in zip(left_views, right_views, selected_dependencies))
            entry['candidates'] = tuple(program._plan_view(ref) for ref in dependencies)
            entry['parent'] = function.value
            entry['invalid_choice'] = choices
            slots = max(map(len, selected_dependencies))
            selected_dependencies = tuple(index for pages in selected_dependencies
                for index in (*pages, *((0xffffffff,) * (slots-len(pages)))))
            readiness = program.tensor((1, slots), dtype=np.uint32)[0, 0]

            # design/algorithm-sources.md#selected-native-contractions
            def readiness_source(metal):
                array = ('constant' if metal else 'static const') + ' uint32_t selected[]={' + ','.join(map(str, selected_dependencies)) + '};'
                return array, f'for(uint32_t c=lane;c<{slots};c+=lanes)p1[c]=selected[{slots}*p0[0]+c];'

            readiness_function = _compiled_region(program, (selector,), readiness, readiness_source)
            entry['readiness'] = dict(view=program._plan_view(readiness), function=readiness_function)
            check(program.native.algebra_indexed(program.handle, function.value, readiness.view,
                (C.c_size_t * len(dependencies))(*range(1, len(dependencies)+1)), len(dependencies)))
    return binding


# design/algorithm-sources.md#composable-indexed-contractions
def _preserve_indexed_contractions(node, lowering):
    children = tuple(_preserve_indexed_contractions(child, lowering) for child in node.operands)
    value = _Expression(node.operation, children, node.value)
    if node.operation in ('+', '-', '*', '//', '%', 'cast') and all(child.operation == 'literal' or
            child.operation == 'cast' and child.operands[0].operation == 'literal' for child in children):
        zero = np.zeros(1, dtype=np.int64)
        folded = _static_value(value, lowering.sources, zero, zero, lowering.coordinate)
        if folded is not None:
            literal = _literal(folded[0])
            dtype = _expression_dtype(value, lowering.sources)
            return literal if _expression_dtype(literal, ()) == dtype else literal.astype(dtype)
    if value.operation != 'sum' or value.operands[0].operation != '*':
        return value
    operands = []
    for child in value.operands[0].operands:
        while True:
            if child.operation == '*' and any(term == _literal(1.0) for term in child.operands):
                child = child.operands[1] if child.operands[0] == _literal(1.0) else child.operands[0]
            elif child.operation == 'cast' and np.dtype(child.value) == np.dtype('float32'):
                child = child.operands[0]
            else:
                break
        if child.operation != 'logical_load' or lowering.sources[child.value[0]].dtype.kind != 'f':
            return value
        operands.append(child)
    resolved = _resolve_logical(value, lowering.sources)
    layout = _expression_layout(resolved, lowering.sources, lowering.whole, {})
    return _Expression('indexed_contract', tuple(operands), (value, layout))


# design/algorithm-sources.md#shared-contraction-lowering
def _lower_region_expressions(program, expressions, grid, input_specs, output_specs):
    import itertools
    cache, requests, consumers = {}, [], {}
    for coordinate in itertools.product(*(range(length) for length in grid)):
        lowering = _ExpressionRegions(program, input_specs, coordinate, cache)

        # design/algorithm-sources.md#shared-contraction-lowering
        def specialize(node):
            if node.operation == 'program_id':
                return _literal(coordinate[node.value])
            return _Expression(node.operation, tuple(specialize(child) for child in node.operands), node.value)

        for expression, spec in zip(expressions, output_specs):
            value = _preserve_indexed_contractions(specialize(expression), lowering)
            target = spec.resolve(coordinate)
            value = _resolve_logical(value, lowering.sources)
            origin = tuple(index * block for index, block in zip(spec.index_map(*coordinate), spec.block_shape))
            domain_shape = spec._tensor.shape
            if value.operation == 'transpose':
                value, target, origin, domain_shape = value.operands[0], target.T, origin[::-1], domain_shape[::-1]
            if value.operation == 'indexed_contract' and target.dtype.kind != 'f':
                value = _resolve_logical(value.value[0], lowering.sources)
            requests.append((lowering, value, target, origin, domain_shape))

            # design/algorithm-sources.md#in-operation-publication
            def demand(node, accumulation):
                if node.operation in _REDUCTIONS:
                    layout = lowering.layout(node)
                    row = 0 if layout[0][0] == 1 or not layout[1][0] else origin[0]
                    rows = 1 if layout[0][0] == 1 else target.shape[0]
                    dtype = _reduction_dtype(node, lowering.sources, accumulation)
                    key = ('reduction', lowering.key(node, (row, 0), (rows, 1)), dtype.str)
                    consumers.setdefault(key, set()).add(len(requests)-1)
                for child in node.operands:
                    demand(child, _expression_dtype(child, lowering.sources))

            demand(value, target.dtype)
    uses = {key: len(requests) for key, requests in consumers.items()}
    for lowering, value, target, origin, domain_shape in requests:
        lowering.reduction_uses = uses
        coordinate = lowering.coordinate
        if value.operation == 'indexed_contract' and target.shape[1] == 1:
            layout = lowering.layout(value)
            row = origin[0] if layout[1][0] else 0
            result = lowering.indexed_contraction_panel(value, (row, 0), target.shape, target)
            lowering.publish((result,), target)
        elif value.operation == 'dot':
            layout = lowering.layout(value)
            for axis in range(2):
                expected = domain_shape[axis] if layout[1][axis] else target.shape[axis]
                if layout[0][axis] != expected:
                    raise ValueError('Contraction output shape differs from its operand domains')
            where = tuple(origin[axis] if layout[1][axis] else 0 for axis in range(2))
            lowering.publish(lowering.parts(value, where, target.shape, target), target)
        elif value.operation in _REDUCTIONS and target.shape[1] == 1:
            layout = lowering.layout(value)
            row = origin[0] if layout[1][0] else 0
            dtype = _reduction_dtype(value, lowering.sources, target.dtype)
            result, identity = lowering.reduction(value, row, target.shape[0], dtype, target)
            if result is not target:
                symbol, = arguments(1)
                _ExpressionKernel((symbol,)).bind(program, (result,), (target,), coordinate)
        else:
            lowering.emit(value, origin, target.shape, target, external=True)
