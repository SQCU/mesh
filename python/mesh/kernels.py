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

    # design/algorithm-sources.md#indexed-expression-lowering
    def at(self, row, column, *, mask=True, other=0):
        if self.operation != 'input':
            raise ValueError('Indexed loads require an input reference')
        return _Expression('load', tuple(map(_literal, (row, column, mask, other))), self.value)

    # design/algorithm-sources.md#region-expression-fusion
    def sum(self):
        return _Expression('sum', (self,))

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


# design/algorithm-sources.md#indexed-expression-lowering
def indices():
    return _Expression('row'), _Expression('column')


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
        if any(value.operation in ('indexed_add', 'dot') for value in self.values):
            for value, spec in zip(self.values, output_specs):
                if value.operation == 'indexed_add':
                    _lower_indexed_add(program, value, grid, input_specs, spec)
                elif value.operation == 'dot':
                    _lower_dot(program, value, grid, input_specs, spec)
                else:
                    _ExpressionKernel((value,)).bind_grid(program, grid, input_specs, (spec,))
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
            reads = tuple(inputs[index] for index in used)
            dynamic, accesses = {}, {}
            dynamic_inputs = {index for index, source in enumerate(reads)
                if hasattr(source, 'blocks') and not all((ref.view.tensor, ref.view.extent)
                    in program._constant_extents for ref in source.blocks.values())}

            # design/algorithm-sources.md#dynamic-indexed-expression-lowering
            def accesses_for(node, path=()):
                if node.operation == 'input' and hasattr(reads[node.value], 'blocks'):
                    raise ValueError('Whole-tensor inputs require indexed loads')
                if node.operation == 'select':
                    condition, yes, no = node.operands
                    accesses_for(condition, path)
                    accesses_for(yes, path + ((condition, True),))
                    accesses_for(no, path + ((condition, False),))
                elif node.operation == 'load':
                    row, column, mask, other = node.operands
                    accesses_for(mask, path)
                    selected_path = path + ((mask, True),)
                    accesses_for(row, selected_path)
                    accesses_for(column, selected_path)
                    accesses_for(other, path + ((mask, False),))
                    if node.value in dynamic_inputs:
                        paths = accesses.setdefault(node, set())
                        if not any(set(previous) <= set(selected_path) for previous in paths):
                            paths.difference_update(previous for previous in tuple(paths) if set(selected_path) <= set(previous))
                            paths.add(selected_path)
                else:
                    for child in node.operands:
                        accesses_for(child, () if node.operation == 'sum' else path)

            accesses_for(expression)
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

        # design/algorithm-sources.md#region-expression-fusion
        def visit(node):
            if node in widths:
                return widths[node]
            sizes = tuple(visit(child) for child in node.operands)
            if node.operation == 'input':
                ref = inputs[node.value]
                if ref.shape[0] not in (1, output.shape[0]):
                    raise ValueError('Expression input rows must broadcast to the output')
                width = ref.shape[1]
            elif node.operation == 'column':
                width = output.shape[1]
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

        # design/algorithm-sources.md#region-expression-fusion
        def emit(node, column):
            if node.operation == 'row':
                return '((long)r)' if metal else '((int64_t)r)'
            if node.operation == 'column':
                return f'((long)({column}))' if metal else f'((int64_t)({column}))'
            if node.operation == 'block_ordinal':
                row, col = (emit(child, column) for child in node.operands)
                rows, columns, grid_columns = node.value
                return f'(({row})/{rows}*{grid_columns}+({col})/{columns})'
            if node.operation == 'load':
                ref = inputs[node.value]
                row, col, mask, other = (emit(child, column) for child in node.operands)
                if hasattr(ref, 'blocks'):
                    block = f'(({row})/{ref.block_shape[0]}*{ref.grid[1]}+({col})/{ref.block_shape[1]})'
                    scalar, strides = layouts[node.value]
                    address = f'(({"device " if metal else ""}const {scalar} *)buffers[{pointers[node.value][0]}+{block}])'
                    value = f'{address}[(({row})%{ref.block_shape[0]})*{strides[0]}+(({col})%{ref.block_shape[1]})*{strides[1]}]'.replace('CANDIDATE', block)
                else:
                    value = f'p{pointers[node.value][0]}[({row})*{ref.view.row_stride}+({col})*{ref.view.column_stride}]'
                value = f'((float)({value}))' if ref.dtype.kind == 'f' else value
                return f'(({mask})?({value}):({other}))'
            if node.operation == 'input':
                ref = inputs[node.value]
                row_stride = ref.view.row_stride if ref.shape[0] != 1 else 0
                column_stride = ref.view.column_stride if ref.shape[1] != 1 else 0
                value = f'p{pointers[node.value][0]}[r*{row_stride}+({column})*{column_stride}]'
                return f'((float)({value}))' if ref.dtype.kind == 'f' else value
            if node.operation == 'sum':
                return names[node]
            return _scalar_expression(node, tuple(emit(child, column) for child in node.operands), metal)

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
            accumulator = ('long' if metal else 'int64_t') if output.dtype.kind in 'ib' else ('ulong' if metal else 'uint64_t') if output.dtype.kind == 'u' else 'float'
            lines.append(f'{accumulator} {name}=0;')
            lines.append(f'for({"uint" if metal else "uint64_t"} k={"lane" if metal else "0"};k<{widths[child]};k+={32 if metal else 1}) {name}+={emit(child, "k")};')
            if metal:
                lines.append(f'{name}=simd_sum({name});')
        lines.append(f'for({"uint" if metal else "uint64_t"} c={"lane" if metal else "0"};c<{output.shape[1]};c+={32 if metal else 1}) p{len(physical)}[r*{output.view.row_stride}+c*{output.view.column_stride}]={emit(expression, "c")};')
        lines.append('}' if metal else '}}')
        return '\n'.join(lines)


# design/algorithm-sources.md#fused-indexed-update-values
def _scalar_expression(node, args, metal):
    if node.operation == 'literal':
        if isinstance(node.value, bool):
            return '1' if node.value else '0'
        if isinstance(node.value, int):
            return str(node.value) + ('ull' if node.value > 2**63 - 1 else 'll')
        return repr(float(node.value)) + 'f'
    if node.operation == 'select':
        return f'(({args[0]})?({args[1]}):({args[2]}))'
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
    grouped = program.tensor((1, 8 * size), dtype=np.uint32)[0, 0]

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
        }}''')
        return '\n'.join(lines)

    _compiled_region(program, tuple(keys), grouped, body)
    return grouped, size


# design/algorithm-sources.md#segmented-indexed-add
def _indexed_range(program, function, selector, bounds, first, count):
    import ctypes as C
    from . import check
    check(program.native.algebra_indexed_range(program.handle, function, selector.view, bounds.view,
        (C.c_size_t * count)(*range(first, first + count)), count))


# design/algorithm-sources.md#segmented-indexed-add
def _candidate_load(refs, first, ordinal, row, column, metal):
    dtype = refs[0].dtype
    scalar = {'f2': 'half' if metal else '_Float16', 'f4': 'float', 'i4': 'int32_t',
              'u4': 'uint32_t', 'i8': 'int64_t', 'u8': 'uint64_t', 'u1': 'uint8_t'}[dtype.kind + str(dtype.itemsize)]
    strides = []
    declarations = []
    for name, attribute in (('row_steps', 'row_stride'), ('column_steps', 'column_stride')):
        values = tuple(getattr(ref.view, attribute) for ref in refs)
        if len(set(values)) == 1:
            strides.append(str(values[0]))
        else:
            declarations.append(f'{"constant" if metal else "static const"} uint64_t {name}[]={{'+','.join(map(str, values))+'};')
            strides.append(f'{name}[{ordinal}]')
    address = f'(({"device " if metal else ""}const {scalar} *)buffers[{first}+({ordinal})])'
    return '\n'.join(declarations), f'{address}[({row})*{strides[0]}+({column})*{strides[1]}]'


# design/algorithm-sources.md#segmented-indexed-add
def _lower_indexed_add(program, expression, grid, input_specs, output_spec):
    import itertools
    import math
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
    value_inputs = set()

    # design/algorithm-sources.md#fused-indexed-update-values
    def value_dependencies(node):
        if node.operation == 'input':
            value_inputs.add(node.value)
        elif node.operation not in ('literal', 'row', 'column', '+', '-', '*', '/', '<', '<=', '>', '>=', '==', '&', '|', 'select', 'rsqrt', 'exp', 'tanh'):
            raise ValueError('Indexed update values require pointwise expressions; reduce or index their producer explicitly')
        for child in node.operands:
            value_dependencies(child)

    value_dependencies(update_value)
    value_inputs = tuple(sorted(value_inputs))
    for index in value_inputs:
        tensor = operands[index]
        if tensor.shape[0] not in (1, size) or tensor.shape[1] not in (1, features):
            raise ValueError('Indexed update operands must broadcast to U×F')
        if tensor.dtype.kind not in ('fiu' if base.dtype.kind == 'f' else 'iu'):
            raise ValueError('Indexed update operand dtype is incompatible with accumulation')
    key_inputs = set()

    # design/algorithm-sources.md#segmented-indexed-add
    def key_dependencies(node):
        if node.operation == 'input':
            key_inputs.add(node.value)
        for child in node.operands:
            key_dependencies(child)

    key_dependencies(expression.operands[1])
    key_dependencies(mask)
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
    key_expression = select(mask & (normalized >= 0) & (normalized < base.shape[0]), normalized, 0xffffffff)
    chunks = []
    for begin in range(0, size, chunk_rows):
        length = min(chunk_rows, size - begin)
        keys = program.tensor((length, 1), dtype=np.uint32)[0, 0]
        reads = tuple(tensor.region(begin if tensor.shape[0] != 1 else 0, 0,
            length if tensor.shape[0] != 1 else 1, 1) if index in key_inputs else tensor
            for index, tensor in enumerate(operands))
        _ExpressionKernel((key_expression,)).bind(program, reads, (keys,))
        directory, count = _group_ordinals(program, (keys,), chunk_rows, begin, begin // chunk_rows)
        partials = program.tensor((count, features), (1, output.block_shape[1]),
            dtype=np.float32 if base.dtype.kind == 'f' else base.dtype)
        chunks.append((directory, count, partials))
        ordinal_view = directory.slice(0, count, 1, count)
        selector_view = directory.slice(0, 7 * count, 1, count)
        for (segment, panel), partial in partials.blocks.items():
            column = panel * partials.block_shape[1]
            candidates = []
            for index in value_inputs:
                tensor = operands[index]
                source_row = begin // tensor.block_shape[0] * tensor.block_shape[0] if tensor.shape[0] != 1 else 0
                source_column = column if tensor.shape[1] != 1 else 0
                source_rows = min(tensor.block_shape[0], tensor.shape[0] - source_row)
                source_columns = partial.shape[1] if tensor.shape[1] != 1 else 1
                candidates.append(tensor.region(source_row, source_column, source_rows, source_columns))
            candidates = tuple(candidates)
            bounds = directory.slice(0, 5 * count + 2 * segment, 1, 2)

            # design/algorithm-sources.md#fused-indexed-update-values
            def reduce_segment(metal, candidates=candidates, partial=partial, column=column):
                loads = {}
                for position, (index, ref) in enumerate(zip(value_inputs, candidates)):
                    tensor = operands[index]
                    _, load = _candidate_load((ref,), 2 + position, '0',
                        f'ordinal%{tensor.block_shape[0]}' if tensor.shape[0] != 1 else '0',
                        'c' if tensor.shape[1] != 1 else '0', metal)
                    loads[index] = f'((float)({load}))' if tensor.dtype.kind == 'f' else load

                # design/algorithm-sources.md#fused-indexed-update-values
                def emit_value(node):
                    if node.operation == 'input':
                        return loads[node.value]
                    if node.operation == 'row':
                        return '((int64_t)ordinal)'
                    if node.operation == 'column':
                        return f'((int64_t)({column}+c))'
                    return _scalar_expression(node, tuple(emit_value(child) for child in node.operands), metal)

                load = emit_value(update_value)
                accumulator = 'float' if base.dtype.kind == 'f' else 'uint64_t'
                return f'''for(uint32_t c=lane;c<{partial.shape[1]};c+=lanes) {{
                  {accumulator} total=0;
                  for(uint32_t k=p1[0];k<p1[1];k++) {{ uint32_t ordinal=p0[k]; total+={load}; }}
                  p{2+len(candidates)}[c*{partial.view.column_stride}]=total;
                }}'''

            function = _compiled_region(program, (ordinal_view, bounds, *candidates), partial, reduce_segment, 2)
            for position, ref in enumerate(candidates):
                if (ref.view.tensor, ref.view.extent) not in program._constant_extents:
                    _indexed_range(program, function, selector_view, bounds, 2 + position, 1)
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
            for _, count, partials in chunks for segment in range(count))
        route, table = _routing_domain(program, owners, reverse_ordinals, offsets, candidates, len(regions))
        for consumer, (row, target) in enumerate(regions):
            initial = base.region(row, column, *target.shape)

            # design/algorithm-sources.md#shared-sparse-routing-lowering
            def finish(metal, target=target, initial=initial, row=row, consumer=consumer):
                scalar = 'float' if base.dtype.kind == 'f' else {
                    'i4': 'int32_t', 'i8': 'int64_t', 'u4': 'uint32_t',
                    'u8': 'uint64_t', 'u1': 'uint8_t'}[base.dtype.kind + str(base.dtype.itemsize)]
                address = f'(({"device " if metal else ""}const {scalar} *)p4[3*ordinal])'
                load = f'{address}[c*p4[3*ordinal+2]]'
                accumulator = 'float' if base.dtype.kind == 'f' else 'uint64_t'
                return f'''uint32_t lo=p2[{consumer}],end=p2[{consumer+1}],hi=end,key={row}+r;
                while(lo<hi) {{ uint32_t mid=lo+(hi-lo)/2; if(p0[mid]<key)lo=mid+1; else hi=mid; }}
                for(uint32_t c=lane;c<{target.shape[1]};c+=lanes) {{
                  {accumulator} total=p3[r*{initial.view.row_stride}+c*{initial.view.column_stride}];
                  for(uint32_t k=lo;k<end && p0[k]==key;k++) {{ uint32_t ordinal=p1[k]; total+={load}; }}
                  p5[r*{target.view.row_stride}+c*{target.view.column_stride}]=total;
                }}'''

            function = _compiled_region(program, (reverse_keys, reverse_ordinals, offsets, initial, table), target, finish)
            from . import check
            check(program.native.algebra_route_attach(program.handle, function, route, consumer))


# design/algorithm-sources.md#shared-sparse-routing-lowering
def _routing_directory(program, chunks, coverage):
    total = sum(count for _, count, _ in chunks)
    metadata = program.tensor((1, 2 * total), dtype=np.uint32)[0, 0]
    keys = tuple(directory.slice(0, 4 * count, 1, count) for directory, count, _ in chunks)

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
        boundaries = tuple(row for row, _ in coverage) + (sum(coverage[-1]),)
        array = f'{"constant" if metal else "static const"} uint32_t boundaries[]={{'+','.join(map(str, boundaries))+'};'
        return array, f'''for(uint32_t c=lane;c<{len(boundaries)};c+=lanes) {{
          uint32_t key=boundaries[c],lo=0,hi={total};
          while(lo<hi) {{ uint32_t mid=lo+(hi-lo)/2; if(p0[mid]<key)lo=mid+1; else hi=mid; }}
          p1[c]=lo;
        }}'''

    _compiled_region(program, (reverse_keys,), offsets, offset_source)
    return owners, reverse_keys, reverse_ordinals, offsets


# design/algorithm-sources.md#shared-sparse-routing-lowering
def _routing_domain(program, owners, ordinals, offsets, candidates, consumers):
    import ctypes as C
    from . import Ref, check
    from ._native import View
    route = program.native.algebra_route_create(program.handle, owners.view, ordinals.view, offsets.view,
        (View * len(candidates))(*(ref.view for ref in candidates)), len(candidates), consumers)
    if not route:
        check(C.get_errno() or 12)
    return route, Ref(program, program.native.algebra_route_table(program.handle, route), np.uint64)


# design/algorithm-sources.md#shared-contraction-lowering
def dot(left, right, *, tile_k=128):
    if tile_k < 1:
        raise ValueError('Contraction K tiles must be positive')
    return _Expression('dot', tuple(map(_literal, (left, right))), tile_k)


# design/algorithm-sources.md#shared-contraction-lowering
def _lower_dot(program, expression, grid, input_specs, output_spec):
    import itertools
    from math import gcd
    from . import check
    from ._native import View
    if any(value.operation != 'input' for value in expression.operands):
        raise ValueError('Contraction operands require logical input references')
    specs = tuple(input_specs[value.value] for value in expression.operands)
    if all(spec.block_shape is None for spec in specs):
        left, right = (spec._tensor for spec in specs)
        if output_spec._tensor.shape != (left.shape[0], right.shape[1]):
            raise ValueError('Contraction output shape differs from its logical operands')

    # design/algorithm-sources.md#shared-contraction-lowering
    def temporary(shape):
        return program.tensor(shape, dtype=np.float32)[0, 0]

    # design/algorithm-sources.md#shared-contraction-lowering
    def bind(operation, inputs, target):
        check(program.native.algebra_bind(program.handle, operation.op, inputs[0].view,
            inputs[1].view if len(inputs) == 2 else View(), target.view, operation.alpha, operation.beta))

    for coordinate in itertools.product(*(range(length) for length in grid)):
        target = output_spec.resolve(coordinate)
        row, column = (index * block for index, block in zip(output_spec.index_map(*coordinate), output_spec.block_shape))
        left, right = (spec.resolve(coordinate) for spec in specs)
        inner = left.shape[1]
        if inner != right.shape[0] or (specs[0].block_shape is not None and left.shape[0] != target.shape[0]) or (specs[1].block_shape is not None and right.shape[1] != target.shape[1]):
            raise ValueError('Mapped contraction dimensions differ')
        tile = min(expression.value, inner)
        for spec, operand, axis in zip(specs, (left, right), (1, 0)):
            if spec.block_shape is None and operand.grid[axis] > 1:
                tile = gcd(tile, operand.block_shape[axis])
        parts = []
        for start in range(0, inner, tile):
            length = min(tile, inner - start)
            destination = target if tile == inner and target.dtype == np.dtype('float32') else temporary(target.shape)
            left_panel = left.region(row, start, target.shape[0], length) if specs[0].block_shape is None else left.slice(0, start, target.shape[0], length)
            right_panel = right.region(start, column, length, target.shape[1]) if specs[1].block_shape is None else right.slice(start, 0, length, target.shape[1])
            bind(matmul, (left_panel, right_panel), destination)
            parts.append(destination)
        while len(parts) > 1:
            reduced = []
            for index in range(0, len(parts), 2):
                if index + 1 == len(parts):
                    reduced.append(parts[index])
                    continue
                destination = target if len(parts) == 2 and target.dtype == np.dtype('float32') else temporary(target.shape)
                bind(add, parts[index:index + 2], destination)
                reduced.append(destination)
            parts = reduced
        if parts[0] is not target:
            bind(affine(), parts, target)
