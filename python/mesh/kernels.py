from dataclasses import dataclass

import numpy as np

_REDUCTIONS = ('sum',)
_REAL_FUNCTIONS = ('exp', 'rsqrt')
_POINTWISE_FUNCTIONS = _REAL_FUNCTIONS
_POINTWISE_OPERATIONS = ('+', '-', '*', '/', '<', '<=', '>', '>=', '==', '&', '|', 'select', 'cast', '//', '%',
    *_POINTWISE_FUNCTIONS)




# design/algorithm-sources.md#kernelsexpression
@dataclass(frozen=True)
class _StaticTable:
    refs: tuple
    ordinals: tuple
    geometry: tuple
    page_map: tuple
    unit: int
    shape: tuple
    block_shape: tuple
    grid: tuple
    dtype: object


@dataclass(frozen=True)
class _Expression:
    operation: str
    operands: tuple = ()
    value: object = None

    # design/algorithm-sources.md#kernelsexpression
    def __add__(self, other):
        return _Expression('+', (self, _literal(other)))

    # design/algorithm-sources.md#kernelsexpression
    def __radd__(self, other):
        return _literal(other) + self

    # design/algorithm-sources.md#kernelsexpression
    def __sub__(self, other):
        return _Expression('-', (self, _literal(other)))

    # design/algorithm-sources.md#kernelsexpression
    def __rsub__(self, other):
        return _literal(other) - self

    # design/algorithm-sources.md#kernelsexpression
    def __mul__(self, other):
        return _Expression('*', (self, _literal(other)))

    # design/algorithm-sources.md#kernelsexpression
    def __rmul__(self, other):
        return _literal(other) * self

    # design/algorithm-sources.md#kernelsexpression
    def __truediv__(self, other):
        return _Expression('/', (self, _literal(other)))

    # design/algorithm-sources.md#kernelsexpression
    def __rtruediv__(self, other):
        return _literal(other) / self

    # design/algorithm-sources.md#kernelsexpression
    def __floordiv__(self, other):
        return _Expression('//', (self, _literal(other)))

    # design/algorithm-sources.md#kernelsexpression
    def __rfloordiv__(self, other):
        return _literal(other) // self

    # design/algorithm-sources.md#kernelsexpression
    def __mod__(self, other):
        return _Expression('%', (self, _literal(other)))

    # design/algorithm-sources.md#kernelsexpression
    def __rmod__(self, other):
        return _literal(other) % self

    # design/algorithm-sources.md#kernelsexpression
    def __lt__(self, other):
        return _Expression('<', (self, _literal(other)))

    # design/algorithm-sources.md#kernelsexpression
    def __le__(self, other):
        return _Expression('<=', (self, _literal(other)))

    # design/algorithm-sources.md#kernelsexpression
    def __gt__(self, other):
        return _Expression('>', (self, _literal(other)))

    # design/algorithm-sources.md#kernelsexpression
    def __ge__(self, other):
        return _Expression('>=', (self, _literal(other)))

    # design/algorithm-sources.md#kernelsexpression
    def equal(self, other):
        return _Expression('==', (self, _literal(other)))

    # design/algorithm-sources.md#kernelsexpression
    def __and__(self, other):
        return _Expression('&', (self, _literal(other)))

    # design/algorithm-sources.md#kernelsexpression
    def __or__(self, other):
        return _Expression('|', (self, _literal(other)))

    # design/algorithm-sources.md#kernelsexpression
    def at(self, row, column, *, mask=True, other=0):
        if self.operation != 'input':
            raise ValueError('Indexed loads require an input reference')
        return _Expression('load', tuple(map(_literal, (row, column, mask, other))), self.value)

    # design/algorithm-sources.md#kernelsdot
    def astype(self, dtype):
        dtype = np.dtype(dtype)
        if dtype.name not in ('float16', 'float32', 'int32', 'uint32', 'int64', 'uint64', 'uint8', 'bool'):
            raise ValueError('Expression casts require a supported scalar dtype')
        return _Expression('cast', (self,), dtype.str)


    # design/algorithm-sources.md#kernelsadd
    def sum(self, axis=1):
        axes = (0, 1) if axis is None else (axis,) if isinstance(axis, int) else tuple(axis)
        if any(value not in (-2, -1, 0, 1) for value in axes):
            raise ValueError('Expression reduction axes refer to its two-dimensional domain')
        axes = tuple(sorted(set(value % 2 for value in axes)))
        if not axes:
            return self
        if axes == (0,):
            return self.T.sum(1).T
        reduced = _Expression('sum', (self,))
        return reduced.T.sum(1) if axes == (0, 1) else reduced

    @property
    # design/algorithm-sources.md#kernelsdot
    def T(self):
        if self.operation == 'transpose':
            return self.operands[0]
        if self.operation in ('literal', 'program_id'):
            return self
        if self.operation in ('row', 'column'):
            return _Expression('column' if self.operation == 'row' else 'row')
        if self.operation == 'domain':
            return _Expression('domain', (self.operands[0].T,), tuple(value[::-1] for value in self.value))
        if self.operation == 'dot':
            return _Expression('dot', tuple(child.T for child in self.operands[::-1]), self.value)
        if self.operation in _POINTWISE_OPERATIONS:
            return _Expression(self.operation, tuple(child.T for child in self.operands), self.value)
        return _Expression('transpose', (self,))

    # design/algorithm-sources.md#kernelsexpression
    def rsqrt(self):
        return _Expression('rsqrt', (self,))

    # design/algorithm-sources.md#kernelsexpression
    def exp(self):
        return _Expression('exp', (self,))


















# design/algorithm-sources.md#kernelsexpression
def _literal(value):
    return value if isinstance(value, _Expression) else _Expression('literal', value=value.item() if isinstance(value, np.generic) else value)


# design/algorithm-sources.md#kernelsexpression
def _static_value(node, inputs, rows, columns, coordinate):
    if node.operation in ('input', 'load'):
        return None
    if node.operation == 'row':
        return rows
    if node.operation == 'column':
        return columns
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


# design/algorithm-sources.md#kernelsexpression
def _specialize_accesses(expression, inputs, output, coordinate, domain):
    from . import Ref
    from ._native import View
    domains, widths = {}, {}
    row_begin, row_count, column_begin, column_count = domain
    program = output.program
    page_bytes = program.native.algebra_page_bytes(program.handle)

    # design/algorithm-sources.md#kernelsexpression
    def width(node):
        if node not in widths:
            children = tuple(width(child) for child in node.operands)
            if any(child is None for child in children):
                return None
            if node.operation == 'input':
                value = inputs[node.value].shape[1]
            elif node.operation == 'column':
                value = output.shape[1]
            elif node.operation == 'row' or node.operation in _REDUCTIONS:
                value = 1
            else:
                value = max(children, default=1)
                if any(child not in (1, value) for child in children):
                    return None
            widths[node] = value
        return widths[node]

    # design/algorithm-sources.md#kernelsexpression
    def visit(node, path=(), extent=output.shape[1], reduced=False):
        if node.operation == 'select':
            condition, yes, no = node.operands
            visit(condition, path, extent, reduced)
            visit(yes, path + ((condition, True),), extent, reduced)
            visit(no, path + ((condition, False),), extent, reduced)
        elif node.operation == 'load':
            row, column, mask, other = node.operands
            visit(mask, path, extent, reduced)
            selected = path + ((mask, True),)
            visit(row, selected, extent, reduced)
            visit(column, selected, extent, reduced)
            visit(other, path + ((mask, False),), extent, reduced)
            if hasattr(inputs[node.value], 'blocks'):
                domains.setdefault(node, set()).add((selected, extent, reduced))
        elif node.operation in _REDUCTIONS:
            visit(node.operands[0], (), width(node.operands[0]), True)
        else:
            for child in node.operands:
                visit(child, path, extent, reduced)

    visit(expression)
    selected = {}
    for node, uses in domains.items():
        table = inputs[node.value]
        blocks, proven = {}, True
        unit = page_bytes // table.dtype.itemsize
        for path, extent, reduced in uses:
            if extent is None:
                proven = False
                break
            first_column, width_count = (0, extent) if reduced else (column_begin, column_count)
            for start in range(0, row_count*width_count, 4096):
                ordinal = np.arange(start, min(start+4096, row_count*width_count), dtype=np.int64)
                rows, columns = row_begin + ordinal // width_count, first_column + ordinal % width_count
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
                for r, c in zip(row, column):
                    r, c = int(r), int(c)
                    block = r // table.block_shape[0], c // table.block_shape[1]
                    view = table.blocks[block].view
                    address = view.offset + r % table.block_shape[0] * view.row_stride + c % table.block_shape[1] * view.column_stride
                    blocks.setdefault(block, set()).add(address // unit)
            if not proven:
                break
        if proven:
            selected[node] = tuple((block, tuple(sorted(pages))) for block, pages in sorted(blocks.items()))
    bound, replacements, origins = list(inputs), {}, {}

    # design/algorithm-sources.md#kernelsexpression
    def rewrite(node):
        if node in replacements:
            return replacements[node]
        children = tuple(rewrite(child) for child in node.operands)
        if node not in selected:
            result = _Expression(node.operation, children, node.value)
            replacements[node] = result
            origins[result] = node
            return result
        table = inputs[node.value]
        row, column, mask, other = children
        chosen = selected[node]
        if chosen:
            refs, positions, geometry, page_map = [], {}, [], []
            unit = page_bytes // table.dtype.itemsize
            for block, pages in chosen:
                view = table.blocks[block].view
                full = program.native.tensor_view(view.tensor, view.extent)
                elements = full.rows * full.columns
                base = len(page_map)
                page_map.extend([0xffffffff] * (pages[-1]-pages[0]+1))
                for page in pages:
                    key = view.tensor, view.extent, page
                    if key not in positions:
                        positions[key] = len(refs)
                        offset = page * unit
                        columns = min(unit, elements-offset)
                        refs.append(Ref(program, View(view.tensor, view.extent, offset, 1, columns, columns, 1), table.dtype))
                    page_map[base+page-pages[0]] = positions[key]
                geometry.append((view.offset, view.row_stride, view.column_stride, base, pages[0]))
            ordinals = tuple(row*table.grid[1]+column for (row, column), _ in chosen)
            source = _StaticTable(tuple(refs), ordinals, tuple(geometry), tuple(page_map), unit,
                                  table.shape, table.block_shape, table.grid, table.dtype)
            index = len(bound)
            bound.append(source)
            result = _Expression('load', children, index)
        else:
            result = other
        replacements[node] = result
        return result

    result = rewrite(expression)
    return result, tuple(bound), origins


# design/algorithm-sources.md#kernelsexpression
def indices():
    return _Expression('row'), _Expression('column')








# design/algorithm-sources.md#kernelsexpression
def program_id(axis):
    return _Expression('program_id', value=axis)


# design/algorithm-sources.md#kernelsexpression
def select(mask, yes, no):
    return _Expression('select', tuple(map(_literal, (mask, yes, no))))


# design/algorithm-sources.md#kernelsexpression
def arguments(count):
    return tuple(_Expression('input', value=index) for index in range(count))


# design/algorithm-sources.md#kernelsexpression
def expression(value, *outputs):
    return _ExpressionKernel(tuple(map(_literal, (value, *outputs))))


@dataclass(frozen=True)
class _ExpressionKernel:
    values: tuple

    # design/algorithm-sources.md#kernelsexpression
    def bind_grid(self, program, grid, input_specs, output_specs):
        import itertools
        if len(output_specs) != len(self.values):
            raise ValueError('Each expression requires an output region')
        grid = tuple(grid)
        if 0 in grid:
            return
        active = tuple((value, spec) for value, spec in zip(self.values, output_specs) if spec._tensor.blocks)
        if len(active) != len(self.values):
            if active:
                _ExpressionKernel(tuple(value for value, _ in active)).bind_grid(program, grid, input_specs, tuple(spec for _, spec in active))
            return
        if any(_requires_regions(value) for value in self.values) or any(hasattr(spec._tensor, 'blocks') and not spec._tensor.blocks for spec in input_specs):
            _lower_region_expressions(program, self.values, grid, input_specs, output_specs)
            return
        for coordinate in itertools.product(*(range(size) for size in grid)):
            self.bind(program, tuple(spec.resolve(coordinate) for spec in input_specs),
                tuple(spec.resolve(coordinate) for spec in output_specs), coordinate)

    # design/algorithm-sources.md#kernelsexpression
    def bind(self, program, inputs, outputs, coordinate=()):
        from . import check, Partial
        from ._native import View
        import ctypes as C
        if len(outputs) != len(self.values):
            raise ValueError('Each expression requires an output region')
        for output in outputs:
            output.partial = Partial.merge(tuple(ref for source in inputs
                for ref in (source.blocks.values() if hasattr(source, 'blocks') else (source,))))
        binding = program._functions.get(self)
        if binding is not None and binding is not self:
            if isinstance(binding, _ExpressionKernel):
                binding.bind(program, inputs, outputs, coordinate)
            else:
                binding(program, inputs, outputs)
            return
        if any(ref.dtype.name not in ('float16', 'float32', 'int32', 'uint32', 'int64', 'uint64', 'uint8', 'bool') for ref in (*inputs, *outputs)):
            raise ValueError('Expression regions require supported real, integer or boolean scalars')
        for value, output in zip(self.values, outputs):
            original_accesses = _indexed_access_paths(value, inputs,
                {index for index, source in enumerate(inputs) if hasattr(source, 'blocks')})
            selectors = {}
            for domain in _source_expression_regions(program, output):
                specialized, specialized_inputs, origins = _specialize_accesses(value, inputs, output, coordinate, domain)
                used, original_nodes = {}, {}

                # design/algorithm-sources.md#kernelsexpression
                def remap(node):
                    index = node.value
                    if node.operation == 'program_id':
                        return _literal(coordinate[index])
                    if node.operation in ('input', 'load'):
                        index = used.setdefault(index, len(used))
                    result = _Expression(node.operation, tuple(remap(child) for child in node.operands), index)
                    if node in origins:
                        original_nodes[result] = origins[node]
                    return result

                expression = remap(specialized)
                reads = tuple(specialized_inputs[index] for index in used)
                dynamic = {}
                access_axes, reduced_accesses = _expression_access_axes(expression, reads)
                dynamic_inputs = {index for index, source in enumerate(reads)
                    if hasattr(source, 'blocks') and not all((ref.view.tensor, ref.view.extent)
                        in program._constant_extents for ref in source.blocks.values())}
                accesses = _indexed_access_paths(expression, reads, dynamic_inputs)
                for node in accesses:
                    original = original_nodes[node]
                    if original not in selectors:
                        table = inputs[original.value]
                        row, column, mask, _ = original.operands
                        ordinal, candidates = _page_selector(program, table, row, column)
                        enabled = _literal(False)
                        for path in original_accesses[original]:
                            condition = _literal(True)
                            for predicate, polarity in path:
                                condition = select(condition, predicate if polarity else predicate.equal(False), False)
                            enabled = select(enabled, True, condition)
                        selector_value = select(enabled, ordinal, 0xffffffff)
                        selector_width = output.shape[1]

                        # design/algorithm-sources.md#kernelsexpression
                        def selector_shape(part):
                            nonlocal selector_width
                            if part.operation == 'input':
                                selector_width = max(selector_width, inputs[part.value].shape[1])
                            for child in part.operands:
                                selector_shape(child)

                        selector_shape(selector_value)
                        selected = program.tensor((output.shape[0], selector_width), dtype=np.uint32)[0, 0]
                        _ExpressionKernel((selector_value,)).bind(program, inputs, (selected,), coordinate)
                        selectors[original] = selected, candidates
                    selected, candidates = selectors[original]
                    dynamic[node] = (selected, node.value, candidates)

                flattened = tuple(ref for source in reads for ref in
                    (source.refs if isinstance(source, _StaticTable) else tuple(ref for _, ref in sorted(source.blocks.items())) if hasattr(source, 'blocks') else (source,)))
                sources = tuple(self.source(reads, output, metal, expression).encode() for metal in (False, True))
                offsets = []
                for source in reads:
                    offsets.append((offsets[-1][0] + offsets[-1][1] if offsets else 0,
                        len(source.refs) if isinstance(source, _StaticTable) else len(source.blocks) if hasattr(source, 'blocks') else 1))
                begin, rows, column, columns = domain
                function = program.native.algebra_function_count(program.handle)
                check(program.native.algebra_source(program.handle, *sources,
                    (View * len(flattened))(*(ref.view for ref in flattened)), len(flattened), output.view,
                    (C.c_uint8 * len(access_axes))(*access_axes), begin, rows, column, columns))
                for node, (selected, index, candidates) in dynamic.items():
                    first, count = offsets[index]
                    selection = selected.slice(begin, 0 if node in reduced_accesses else column, rows,
                                               selected.shape[1] if node in reduced_accesses else columns)
                    check(program.native.algebra_indexed(program.handle, function, selection.view,
                        (C.c_size_t * count)(*range(first, first + count)), count,
                        (View * len(candidates))(*candidates), len(candidates)))

    # design/algorithm-sources.md#kernelsadd
    def source(self, inputs, output, metal, expression):
        widths, reductions = {}, []
        physical, pointers = [], {}
        for index, ref in enumerate(inputs):
            refs = ref.refs if isinstance(ref, _StaticTable) else tuple(ref for _, ref in sorted(ref.blocks.items())) if hasattr(ref, 'blocks') else (ref,)
            pointers[index] = tuple(range(len(physical), len(physical) + len(refs)))
            physical.extend(refs)

        # design/algorithm-sources.md#kernelsdot
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
            if node.operation in ('<', '<=', '>', '>=', '==', 'row', 'column', 'block_ordinal', 'lookup'):
                return True
            if node.operation in _REAL_FUNCTIONS:
                return False
            if node.operation in _REDUCTIONS:
                return _reduction_dtype(node, inputs, output.dtype).kind in 'iub'
            return all(integral(child) for child in node.operands)

        # design/algorithm-sources.md#kernelsexpression
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

        # design/algorithm-sources.md#kernelsexpression
        def emit(node, column):
            # design/algorithm-sources.md#kernelsexpression
            def resolve(part, args):
                if part.operation == 'row':
                    return '((long)r)' if metal else '((int64_t)r)'
                if part.operation == 'column':
                    return f'((long)({column}))' if metal else f'((int64_t)({column}))'
                if part.operation in _REDUCTIONS:
                    return f'(({"long" if metal else "int64_t"}){names[part]})' if part.operation == 'sum' and _reduction_dtype(part, inputs, output.dtype).kind in 'ib' else names[part]
                ref = inputs[part.value]
                if part.operation == 'load':
                    return _indexed_load_expression(ref, pointers[part.value][0], layouts.get(part.value), args, metal)
                row_stride = ref.view.row_stride if ref.shape[0] != 1 else 0
                column_stride = ref.view.column_stride if ref.shape[1] != 1 else 0
                value = _memory_expression(ref.dtype, pointers[part.value][0], f'r*{row_stride}+({column})*{column_stride}', metal)
                return f'((float)({value}))' if ref.dtype.kind == 'f' else value

            return _emit_scalar_expression(node, inputs, metal, resolve)

        lines = ['#include <metal_stdlib>\nusing namespace metal;' if metal else '#include <stdint.h>\n#include <stdbool.h>\n#include <math.h>\n#include <stdatomic.h>']
        lines.append(_address_source(metal))
        lines.append(_lookup_declarations(expression, metal))
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
            strides = []
            for name, attr in (('rs', 'row_stride'), ('cs', 'column_stride')):
                values = tuple(getattr(r.view, attr) for r in refs)
                if len(set(values)) == 1:
                    strides.append(str(values[0]))
                else:
                    lines.append(f'{"constant ulong" if metal else "static const uint64_t"} {name}{index}[]={{'+','.join(map(str, values))+'};')
                    strides.append(f'{name}{index}[CANDIDATE]')
            layouts[index] = strides
        lines.append(_METAL_EXPRESSION_HEAD if metal else 'void mesh_expression(const uintptr_t *buffers, const struct mesh_kernel_publication *publication) {')
        if not metal:
            lines.append(_CPU_PUBLICATION_LOOP)
        for node in reductions:
            name, child = names[node], node.operands[0]
            dtype = _reduction_dtype(node, inputs, output.dtype)
            unsigned = dtype.kind == 'u' or (dtype.kind in 'ib' and integral(child))
            accumulator = (('ulong' if metal else 'uint64_t') if unsigned else ('long' if metal else 'int64_t')) if dtype.itemsize == 8 else (
                'float' if dtype.kind == 'f' else ('uint' if metal else 'uint32_t') if dtype.kind in 'ub' else ('int' if metal else 'int32_t'))
            lines.append(f'{accumulator} {name}=0;')
            operand = emit(child, 'k')
            combine = f'{name}+={operand}'
            lines.append(f'for({"uint" if metal else "uint64_t"} k={"lane" if metal else "0"};k<{widths[child]};k+={32 if metal else 1}) {combine};')
            if metal and dtype.kind in 'iub':
                lines.append(f'uint {name}_lo=simd_sum(uint(ulong({name})&65535ul)), {name}_mid=simd_sum(uint((ulong({name})>>16)&65535ul)), {name}_hi=simd_sum(uint(ulong({name})>>32));')
                lines.append(f'{name}_mid+={name}_lo>>16; {name}_hi+={name}_mid>>16; {name}=(ulong({name}_hi)<<32)|(ulong({name}_mid&65535u)<<16)|ulong({name}_lo&65535u);')
            elif metal:
                lines.append(f'{name}=simd_sum({name});')
        destination = _memory_expression(output.dtype, len(physical), f'r*{output.view.row_stride}+c*{output.view.column_stride}', metal)
        lines.append(f'for({"ulong" if metal else "uint64_t"} c={"column_begin+lane" if metal else "part.column_begin"};c<{"column_end" if metal else "part.column_end"};c+={32 if metal else 1}) {destination}={emit(expression, "c")};')
        lines.append('}' if metal else _CPU_PUBLICATION_END)
        return '\n'.join(lines)


# design/algorithm-sources.md#kernelsexpression
def _expression_access_axes(expression, inputs):
    axes = [3] * len(inputs)
    reduced_accesses = set()

    # design/algorithm-sources.md#kernelsexpression
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


# design/algorithm-sources.md#kernelsexpression
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


# design/algorithm-sources.md#kernelsexpression
_METAL_EXPRESSION_HEAD = 'kernel void mesh_expression(device const ulong *buffers [[buffer(0)]], constant ulong *domain [[buffer(1)]], uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { const ulong r=domain[0]+row, column_begin=domain[1], column_end=domain[2];'


# design/algorithm-sources.md#programkernel_call
_CPU_PUBLICATION_LOOP = 'for(uint64_t section=0;section<publication->count;section++) { const struct mesh_kernel_section part=publication->sections[section]; for(uint64_t r=part.row_begin;r<part.row_end;r++) {'
_CPU_PUBLICATION_END = '} publication->publish(publication->context,part.first,part.count); }}'


# design/algorithm-sources.md#kernelsexpression
def _indexed_access_paths(expression, inputs, dynamic_inputs):
    accesses = {}

    # design/algorithm-sources.md#kernelsexpression
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


# design/algorithm-sources.md#kernelsexpression
def _lookup_name(values):
    import hashlib
    return 'mesh_lookup_' + hashlib.sha256(repr(values).encode()).hexdigest()


# design/algorithm-sources.md#kernelsexpression
def _lookup_declarations(expression, metal):
    tables, pending = {}, [expression]
    while pending:
        node = pending.pop()
        if node.operation == 'lookup':
            tables[_lookup_name(node.value)] = node.value
        pending.extend(node.operands)
    return '\n'.join(f'{"constant" if metal else "static const"} uint64_t {name}[]={{' +
        ','.join(f'{value}ull' for value in values) + '};' for name, values in sorted(tables.items()))


# design/algorithm-sources.md#kernelsexpression
def _page_selector(program, table, row, column):
    import ctypes as C
    from . import check
    from ._native import View
    unit = program.native.algebra_page_bytes(program.handle) // table.dtype.itemsize
    candidates, positions, maps, geometry = [], {}, [], []
    for _, ref in sorted(table.blocks.items()):
        count = C.c_size_t()
        check(program.native.algebra_view_pages(program.handle, ref.view, None, 0, C.byref(count)))
        pages = (View * count.value)()
        check(program.native.algebra_view_pages(program.handle, ref.view, pages, count.value, C.byref(count)))
        first, last = pages[0].offset // unit, pages[-1].offset // unit
        base = len(maps)
        maps.extend([0xffffffff] * (last-first+1))
        for page in pages:
            key = page.tensor, page.extent, page.offset
            if key not in positions:
                positions[key] = len(candidates)
                candidates.append(page)
            maps[base+page.offset//unit-first] = positions[key]
        geometry.append((ref.view.offset-first*unit, ref.view.row_stride, ref.view.column_stride, base))
    block = _Expression('block_ordinal', (row, column), (*table.block_shape, table.grid[1]))
    offset, row_stride, column_stride, base = (
        _Expression('lookup', (block,), tuple(entry[index] for entry in geometry)) for index in range(4))
    address = offset + row % table.block_shape[0] * row_stride + column % table.block_shape[1] * column_stride
    return _Expression('lookup', (base + address // unit,), tuple(maps)), tuple(candidates)


# design/algorithm-sources.md#programtensor
def _memory_expression(dtype, pointer, index, metal):
    scalar = {'f2': 'half' if metal else '_Float16', 'f4': 'float',
              'i4': 'int' if metal else 'int32_t', 'u4': 'uint' if metal else 'uint32_t',
              'i8': 'long' if metal else 'int64_t', 'u8': 'ulong' if metal else 'uint64_t',
              'u1': 'uchar' if metal else 'uint8_t', 'b1': 'bool'}[dtype.kind + str(dtype.itemsize)]
    return f'(*(({"device " if metal else ""}{scalar} *)mesh_address(buffers,({pointer}),({index})*{dtype.itemsize})))'


# design/algorithm-sources.md#programtensor
def _address_source(metal):
    integer = 'ulong' if metal else 'uintptr_t'
    device = 'device ' if metal else ''
    page = '((device const uint *)d[1])[byte/d[3]]' if metal else 'atomic_load_explicit(&((const _Atomic uint32_t *)d[1])[byte/d[3]],memory_order_acquire)'
    return f"""// design/algorithm-sources.md#programtensor
    {'inline' if metal else 'static inline'} {device}{'uchar' if metal else 'unsigned char'} *mesh_address({device}const {integer} *buffers,{integer} operand,{integer} offset) {{
      {device}const {integer} *d=({device}const {integer} *)buffers[operand];
      {integer} byte=d[2]+offset,physical=({integer})({page})*d[3]+byte%d[3];
      return ({device}{'uchar' if metal else 'unsigned char'} *)((({device}const {integer} *)d[0])[physical/d[4]]+physical%d[4]);
    }}"""


# design/algorithm-sources.md#kernelsexpression
def _static_table_source(table, first, name, metal):
    dtype = table.dtype
    scalar = {'f2': 'half' if metal else '_Float16', 'f4': 'float', 'i4': 'int' if metal else 'int32_t',
              'u4': 'uint' if metal else 'uint32_t', 'i8': 'long' if metal else 'int64_t',
              'u8': 'ulong' if metal else 'uint64_t', 'u1': 'uchar' if metal else 'uint8_t', 'b1': 'bool'}[dtype.kind+str(dtype.itemsize)]
    integer = 'ulong' if metal else 'uint64_t'
    constant = 'constant' if metal else 'static const'
    lines = []
    for suffix, values in (('ordinal', table.ordinals), ('pages', table.page_map),
                           *((name, tuple(entry[index] for entry in table.geometry))
                             for index, name in enumerate(('offset', 'rs', 'cs', 'base', 'first')))):
        lines.append(f'{constant} {integer} {name}_{suffix}[]={{'+','.join(map(str, values))+'};')
    value = _memory_expression(dtype, f'{first}+page', f'address%{table.unit}', metal)
    result_type = 'float' if dtype.kind == 'f' else scalar
    lines.append(f"""// design/algorithm-sources.md#kernelsexpression
    {'inline' if metal else 'static inline'} {result_type} {name}({'device const ulong *' if metal else 'const uintptr_t *'} buffers,{integer} row,{integer} column) {{
      {integer} ordinal=row/{table.block_shape[0]}*{table.grid[1]}+column/{table.block_shape[1]},low=0,high={len(table.ordinals)-1};
      while(low<high) {{ {integer} middle=low+(high-low)/2;
        if({name}_ordinal[middle]<ordinal)low=middle+1;else high=middle; }}
      {integer} address={name}_offset[low]+row%{table.block_shape[0]}*{name}_rs[low]+column%{table.block_shape[1]}*{name}_cs[low];
      {integer} page={name}_pages[{name}_base[low]+address/{table.unit}-{name}_first[low]];
      return {value};
    }}""")
    return '\n'.join(lines)


# design/algorithm-sources.md#kernelsexpression
def _indexed_load_expression(ref, pointer, layout, args, metal):
    row, column, mask, other = args
    if isinstance(ref, _StaticTable):
        return f'(({mask})?{layout}(buffers,({row}),({column})):({other}))'
    if hasattr(ref, 'blocks'):
        block = f'(({row})/{ref.block_shape[0]}*{ref.grid[1]}+({column})/{ref.block_shape[1]})'
        strides = layout
        index = f'(({row})%{ref.block_shape[0]})*{strides[0]}+(({column})%{ref.block_shape[1]})*{strides[1]}'.replace('CANDIDATE', block)
        value = _memory_expression(ref.dtype, f'{pointer}+{block}', index, metal)
    else:
        strides = (ref.view.row_stride, ref.view.column_stride) if layout is None else layout
        value = _memory_expression(ref.dtype, pointer, f'({row})*{strides[0]}+({column})*{strides[1]}', metal)
    value = f'((float)({value}))' if ref.dtype.kind == 'f' else value
    return f'(({mask})?({value}):({other}))'


# design/algorithm-sources.md#kernelsexpression
def _emit_scalar_expression(node, inputs, metal, resolve):
    if node.operation in ('input', 'row', 'column') or node.operation in _REDUCTIONS:
        return resolve(node, ())
    args = tuple(_emit_scalar_expression(child, inputs, metal, resolve) for child in node.operands)
    if node.operation == 'load':
        return resolve(node, args)
    if node.operation == 'lookup':
        return f'{_lookup_name(node.value)}[{args[0]}]'
    if node.operation == 'block_ordinal':
        row, column = args
        rows, columns, grid_columns = node.value
        return f'(({row})/{rows}*{grid_columns}+({column})/{columns})'
    return _scalar_expression(node, args, metal,
        _expression_dtype(node, inputs) if node.operation in ('//', '%') else None)


# design/algorithm-sources.md#kernelsexpression
def _expression_dtype(node, inputs):
    if node.operation == 'input':
        return inputs[node.value].dtype
    if node.operation == 'cast':
        return np.dtype(node.value)
    if node.operation == 'dot':
        types = tuple(_expression_dtype(child, inputs) for child in node.operands)
        return np.dtype('bool') if all(dtype.kind == 'b' for dtype in types) else _expression_dtype(_Expression('*', node.operands), inputs)
    if node.operation in _REAL_FUNCTIONS:
        return np.dtype('float32')
    if node.operation == 'lookup':
        return np.dtype('uint64')
    if node.operation in ('<', '<=', '>', '>=', '=='):
        return np.dtype('bool')
    if node.operation == 'domain':
        return _expression_dtype(node.operands[0], inputs)
    if node.operation in ('row', 'column', 'program_id'):
        return np.dtype('int64')
    if node.operation == 'literal':
        return np.dtype('int32' if isinstance(node.value, bool) else 'float32' if isinstance(node.value, float) else 'uint64' if node.value > 2**63-1 else 'int64')
    if node.operation in _REDUCTIONS:
        if node.value is not None:
            return np.dtype(node.value)
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


# design/algorithm-sources.md#kernelsadd
def _reduction_dtype(node, inputs, output):
    if node.value is not None:
        return np.dtype(node.value)
    return np.dtype(np.int64 if output.kind in 'ib' else np.uint64 if output.kind == 'u' else np.float32)






# design/algorithm-sources.md#kernelsexpression
def _scalar_expression(node, args, metal, dtype=None):
    if node.operation in _REAL_FUNCTIONS and node.operation != 'rsqrt':
        name = node.operation + ('' if metal else 'f')
        return name + '(' + ','.join(f'((float)({value}))' for value in args) + ')'
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


# design/algorithm-sources.md#kernelsdot
def dot(left, right, *, tile_k=128):
    if tile_k < 1:
        raise ValueError('Contraction K tiles must be positive')
    return _Expression('dot', tuple(map(_literal, (left, right))), tile_k)


# design/algorithm-sources.md#kernelsdot
def _requires_regions(node):
    return node.operation in ('dot', 'cast', 'transpose') or node.operation in _REDUCTIONS or any(_requires_regions(child) for child in node.operands)














# design/algorithm-sources.md#programkernel_call
def _bind_contraction(program, inputs, target):
    from . import check, Partial
    binding = program._functions.get(dot)
    if binding is None:
        check(program.native.algebra_contract(program.handle, inputs[0].view,
            inputs[1].view, target.view, 1))
    else:
        binding(program, inputs, (target,))
    target.partial = Partial.merge(inputs)


# design/algorithm-sources.md#programtensor
def _expression_layout(node, sources, whole, layouts):
    from math import gcd
    # design/algorithm-sources.md#programtensor
    def layout(child):
        return _expression_layout(child, sources, whole, layouts)

    if node in layouts:
        return layouts[node]
    if node.operation == 'input':
        source = sources[node.value]
        shape = source.shape
        result = shape, (whole[node.value],) * 2, source.block_shape if whole[node.value] else shape
    elif node.operation in ('literal', 'program_id', 'row', 'column'):
        result = (1, 1), (False, False), (1, 1)
    elif node.operation == 'domain':
        result = node.value
    elif node.operation == 'cast':
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
    # design/algorithm-sources.md#kernelsdot
    def __init__(self, program, specs, coordinate, cache):
        self.program, self.coordinate, self.cache = program, coordinate, cache
        self.sources = tuple(spec.resolve(coordinate) for spec in specs)
        self.whole = tuple(spec.block_shape is None and hasattr(source, 'blocks') for spec, source in zip(specs, self.sources))
        self.layouts = {}
        self.reduction_uses = {}
        self.page_bytes = program.native.algebra_page_bytes(program.handle)

    # design/algorithm-sources.md#programtensor
    def layout(self, node):
        return _expression_layout(node, self.sources, self.whole, self.layouts)

    # design/algorithm-sources.md#kernelsdot
    def key(self, node, origin, shape):
        used = set()

        # design/algorithm-sources.md#kernelsdot
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

    # design/algorithm-sources.md#kernelsdot
    def temporary(self, shape, dtype=np.float32):
        return self.program.tensor(shape, dtype=dtype)[0, 0]

    # design/algorithm-sources.md#kernelsdot
    def panel(self, node, origin, shape):
        if node.operation == 'input':
            source = self.sources[node.value]
            return source.region(*origin, *shape) if self.whole[node.value] else source.slice(*origin, *shape)
        if node.operation == 'transpose':
            return self.panel(node.operands[0], origin[::-1], shape[::-1]).T
        if node.operation in _REDUCTIONS:
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



    # design/algorithm-sources.md#kernelsdot
    def parts(self, node, origin, shape, direct=None):
        from . import Partial
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
        boundaries = {0, inner, *range(0, inner, tile)}
        boundaries.update(cut for cut in self.page_cuts(left, 1, origin[0], shape[0]) if 0 < cut < inner)
        boundaries.update(cut for cut in self.page_cuts(right, 0, origin[1], shape[1]) if 0 < cut < inner)
        ordered = sorted(boundaries)
        contributions = tuple(object() for _ in range(len(ordered) - 1))
        required = frozenset(contributions)
        parts = []
        for start, end in zip(ordered, ordered[1:]):
            length = end - start
            left_panel = self.panel(operands[0], (origin[0], start), (shape[0], length))
            right_panel = self.panel(operands[1], (start, origin[1]), (length, shape[1]))
            reverse = left_panel.dtype == np.dtype('float16') and right_panel.dtype == np.dtype('float32')
            if reverse:
                destination = self.temporary(shape[::-1]).T
            else:
                destination = direct if direct is not None and len(ordered) == 2 and direct.dtype == dtype else self.temporary(shape, dtype)
            _bind_contraction(self.program, (left_panel, right_panel), destination)
            if len(contributions) > 1:
                destination.partial = Partial(required, frozenset((contributions[len(parts)],)))
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

    # design/algorithm-sources.md#kernelsexpression
    def page_cuts(self, value, axis, origin, count):
        shape = self.layout(value)[0]
        if shape[axis] == 1:
            return set()
        if shape[1-axis] == 1:
            origin, count = 0, 1
        if value.operation == 'input':
            source = self.sources[value.value]
            refs = ((tuple(i*b for i, b in zip(coordinate, source.block_shape)), ref)
                    for coordinate, ref in source.blocks.items()) if self.whole[value.value] else (((0, 0), source),)
            result = set()
            for base, ref in refs:
                if (ref.view.tensor, ref.view.extent) in self.program._constant_extents:
                    continue
                start = max(origin, base[1-axis])
                stop = min(origin+count, base[1-axis]+ref.shape[1-axis])
                if start >= stop:
                    continue
                result.update((base[axis], base[axis]+ref.shape[axis]))
                strides = ref.view.row_stride, ref.view.column_stride
                stride, unit = strides[axis], self.page_bytes // ref.dtype.itemsize
                if stride == 0:
                    continue
                for other in range(start-base[1-axis], stop-base[1-axis]):
                    address = ref.view.offset + other*strides[1-axis]
                    position = 0
                    while position < ref.shape[axis]:
                        remaining = unit - (address+position*stride) % unit
                        position = min(ref.shape[axis], position + (remaining+stride-1)//stride)
                        result.add(base[axis]+position)
            return result
        if value.operation == 'transpose':
            return self.page_cuts(value.operands[0], 1-axis, origin, count)
        if value.operation == 'dot':
            operand = value.operands[1 if axis == 1 else 0]
            return self.page_cuts(operand, axis, 0, self.layout(operand)[0][1-axis])
        if value.operation in _REDUCTIONS:
            return self.page_cuts(value.operands[0], axis, 0, self.layout(value.operands[0])[0][1-axis]) if axis == 0 else set()
        result = set()
        for operand in value.operands:
            if self.layout(operand)[0][axis] == shape[axis]:
                result.update(self.page_cuts(operand, axis, origin, count))
        return result

    # design/algorithm-sources.md#kernelsexpression
    def reduction_regions(self, node, row, rows):
        child = node.operands[0]
        layout = self.layout(child)
        width, tile = layout[0][1], layout[2][1]
        boundaries = {0, width, *range(0, width, tile)}

        boundaries.update(cut for cut in self.page_cuts(child, 1, row, rows) if 0 < cut < width)
        ordered = sorted(boundaries)
        return tuple((first, last-first) for first, last in zip(ordered, ordered[1:]))

    # design/algorithm-sources.md#kernelsadd
    def reduction(self, node, row, rows, dtype, direct=None):
        key = ('reduction', self.key(node, (row, 0), (rows, 1)), dtype.str)
        if key in self.cache:
            return self.cache[key]
        child = node.operands[0]
        layout = self.layout(child)
        width, tile = layout[0][1], layout[2][1]
        if width == 0:
            identity = 0
            target = direct if direct is not None and direct.dtype == dtype else self.temporary((rows, 1), dtype)
            _ExpressionKernel((_literal(identity),)).bind(self.program, (), (target,))
            self.cache[key] = target
            return self.cache[key]
        regions = self.reduction_regions(node, row, rows)
        parts = []
        for column, length in regions:
            target = direct if direct is not None and len(regions) == 1 and direct.dtype == dtype else self.temporary((rows, 1), dtype)
            self.emit(child, (row, column), (rows, length), target, reduce=node.operation)
            parts.append(target)
        while len(parts) > 1:
            following = []
            for index in range(0, len(parts), 2):
                if index + 1 == len(parts):
                    following.append(parts[index])
                    continue
                target = direct if direct is not None and len(parts) == 2 and direct.dtype == dtype else self.temporary((rows, 1), dtype)
                inputs = (parts[index], parts[index + 1])
                if dtype.kind == 'f':
                    add.bind(self.program, inputs, (target,))
                else:
                    left, right = arguments(2)
                    bits = 0xffffffffffffffff
                    merged = (left & bits)+(right & bits)
                    _ExpressionKernel((merged,)).bind(self.program, inputs, (target,), self.coordinate)
                following.append(target)
            parts = following
        self.cache[key] = parts[0]
        return self.cache[key]

    # design/algorithm-sources.md#kernelsdot
    def publish(self, parts, target):
        dtype = parts[0].dtype
        if dtype.kind in 'iub':
            if len(parts) == 1 and parts[0] is target:
                return
            terms = arguments(len(parts))
            value = terms[0] if len(parts) == 1 else (terms[0] + terms[1])
            _ExpressionKernel((value,)).bind(self.program, parts, (target,), self.coordinate)
            return
        if len(parts) == 2:
            destination = target if target.dtype == np.dtype('float32') else self.temporary(target.shape)
            add.bind(self.program, parts, (destination,))
            parts = (destination,)
        if parts[0] is not target:
            if target.dtype.kind in 'iub':
                symbol, = arguments(1)
                _ExpressionKernel((symbol,)).bind(self.program, parts, (target,), self.coordinate)
            else:
                expression(arguments(1)[0]).bind(self.program, parts, (target,))

    # design/algorithm-sources.md#programkernel_call
    def inline_reduction(self, node, expression, origin, shape, dtype, external):
        import ctypes as C
        from . import check
        from ._native import View
        layout = self.layout(node.operands[0])
        row = 0 if layout[0][0] == 1 or (external and not layout[1][0]) else origin[0]
        rows = 1 if layout[0][0] == 1 else shape[0]
        key = ('reduction', self.key(node, (row, 0), (rows, 1)), dtype.str)
        if (self.reduction_uses.get(key) != 1 or key in self.cache or
                layout[0][1] != shape[1] or len(self.reduction_regions(node, row, rows)) != 1 or rows != shape[0]):
            return None

        # design/algorithm-sources.md#programkernel_call
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

        # design/algorithm-sources.md#programkernel_call
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

    # design/algorithm-sources.md#kernelsadd
    def emit(self, value, origin, shape, target, external=False, reduce=False):
        inputs, replacements = [], {}

        # design/algorithm-sources.md#kernelsdot
        def reference(key, refs):
            if key not in replacements:
                terms = tuple(_Expression('input', value=len(inputs) + index) for index in range(len(refs)))
                inputs.extend(refs)
                replacements[key] = terms[0] if len(terms) == 1 else (terms[0] + terms[1])
            return replacements[key]

        # design/algorithm-sources.md#kernelsdot
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
                    # design/algorithm-sources.md#programkernel_call
                    def substitute(part):
                        if part.operation == 'input':
                            return reference(('inline_reduction', node, part.value), (inline[part.value],))
                        if part.operation == 'domain':
                            return substitute(part.operands[0])
                        return _Expression(part.operation, tuple(substitute(child) for child in part.operands), part.value)

                    child = _Expression('domain', (substitute(node.operands[0]),),
                        (shape, (False, False), shape))
                    return _Expression(node.operation, (child,), dtype.str)
                return reference(('reduction', node, row, rows, dtype.str), (self.reduction(node, row, rows, dtype),))
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
            if node.operation in ('row', 'column') and not external:
                axis = 0 if node.operation == 'row' else 1
                return node + origin[axis]
            return _Expression(node.operation, tuple(lower(child, _expression_dtype(child, self.sources)) for child in node.operands), node.value)

        lowered = lower(value, _expression_dtype(value, self.sources) if reduce else target.dtype)
        if reduce:
            lowered = _Expression(reduce, (lowered,))
        _ExpressionKernel((lowered,)).bind(self.program, tuple(inputs), (target,), self.coordinate)






# design/algorithm-sources.md#kernelsdot
def _lower_region_expressions(program, expressions, grid, input_specs, output_specs):
    import itertools
    cache, requests, consumers = {}, [], {}
    for coordinate in itertools.product(*(range(length) for length in grid)):
        lowering = _ExpressionRegions(program, input_specs, coordinate, cache)

        # design/algorithm-sources.md#kernelsdot
        def specialize(node):
            if node.operation == 'program_id':
                return _literal(coordinate[node.value])
            return _Expression(node.operation, tuple(specialize(child) for child in node.operands), node.value)

        for expression, spec in zip(expressions, output_specs):
            value = specialize(expression)
            target = spec.resolve(coordinate)
            origin = tuple(index * block for index, block in zip(spec.index_map(*coordinate), spec.block_shape))
            domain_shape = spec._tensor.shape
            if value.operation == 'transpose':
                value, target, origin, domain_shape = value.operands[0], target.T, origin[::-1], domain_shape[::-1]
            requests.append((lowering, value, target, origin, domain_shape))

            # design/algorithm-sources.md#programkernel_call
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
        if value.operation == 'dot':
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
            result = lowering.reduction(value, row, target.shape[0], dtype, target)
            if result is not target:
                symbol, = arguments(1)
                _ExpressionKernel((symbol,)).bind(program, (result,), (target,), coordinate)
        else:
            lowering.emit(value, origin, target.shape, target, external=True)


# design/algorithm-sources.md#programkernel_call
_left, _right = arguments(2)
add = expression(_left + _right)
swish = expression(_left / (1 + (0 - _left).exp()))
del _left, _right
