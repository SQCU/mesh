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

    # design/algorithm-sources.md#region-expression-fusion
    def bind(self, program, inputs, outputs):
        from . import check
        from ._native import View
        if len(outputs) != len(self.values):
            raise ValueError('Each expression requires an output region')
        if any(ref.dtype.name not in ('float16', 'float32', 'int32', 'uint32', 'int64', 'uint64', 'uint8', 'bool') for ref in (*inputs, *outputs)):
            raise ValueError('Expression regions require supported real, integer or boolean scalars')
        for value, output in zip(self.values, outputs):
            used = {}

            # design/algorithm-sources.md#indexed-expression-lowering
            def remap(node):
                index = node.value
                if node.operation in ('input', 'load'):
                    index = used.setdefault(index, len(used))
                return _Expression(node.operation, tuple(remap(child) for child in node.operands), index)

            expression = remap(value)
            reads = tuple(inputs[index] for index in used)
            check(program.native.algebra_source(program.handle,
                self.source(reads, output, False, expression).encode(),
                self.source(reads, output, True, expression).encode(),
                (View * len(reads))(*(ref.view for ref in reads)), len(reads), output.view))

    # design/algorithm-sources.md#region-expression-fusion
    def source(self, inputs, output, metal, expression):
        widths, reductions = {}, []

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
            if node.operation == 'load':
                ref = inputs[node.value]
                row, col, mask, other = (emit(child, column) for child in node.operands)
                value = f'p{node.value}[({row})*{ref.view.row_stride}+({col})*{ref.view.column_stride}]'
                value = f'((float)({value}))' if ref.dtype.kind == 'f' else value
                return f'(({mask})?({value}):({other}))'
            if node.operation == 'input':
                ref = inputs[node.value]
                row_stride = ref.view.row_stride if ref.shape[0] != 1 else 0
                column_stride = ref.view.column_stride if ref.shape[1] != 1 else 0
                value = f'p{node.value}[r*{row_stride}+({column})*{column_stride}]'
                return f'((float)({value}))' if ref.dtype.kind == 'f' else value
            if node.operation == 'literal':
                if isinstance(node.value, bool):
                    return '1' if node.value else '0'
                if isinstance(node.value, int):
                    return str(node.value) + ('ull' if node.value > 2**63 - 1 else 'll')
                return repr(float(node.value)) + 'f'
            if node.operation == 'sum':
                return names[node]
            args = tuple(emit(child, column) for child in node.operands)
            if node.operation == 'select':
                return f'(({args[0]})?({args[1]}):({args[2]}))'
            if node.operation in ('+', '-', '*', '/', '<', '<=', '>', '>=', '==', '&', '|'):
                return f'({args[0]}{node.operation}{args[1]})'
            if node.operation == 'rsqrt':
                return f'rsqrt({args[0]})' if metal else f'(1.0f/sqrtf({args[0]}))'
            return f'{node.operation}{"" if metal else "f"}({args[0]})'

        lines = ['#include <metal_stdlib>\nusing namespace metal;' if metal else '#include <stdint.h>\n#include <stdbool.h>\n#include <math.h>']
        lines.append('kernel void mesh_expression(device const ulong *buffers [[buffer(0)]], uint r [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {' if metal else 'void mesh_expression(const uintptr_t *buffers) {')
        for index, ref in enumerate((*inputs, output)):
            scalar = ({'f2': 'half' if metal else '_Float16', 'f4': 'float', 'i4': 'int' if metal else 'int32_t',
                       'u4': 'uint' if metal else 'uint32_t', 'i8': 'long' if metal else 'int64_t',
                       'u8': 'ulong' if metal else 'uint64_t', 'u1': 'uchar' if metal else 'uint8_t', 'b1': 'bool'}[ref.dtype.kind + str(ref.dtype.itemsize)])
            qualifier = ('device ' if metal else '') + ('const ' if index < len(inputs) else '')
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
        lines.append(f'for({"uint" if metal else "uint64_t"} c={"lane" if metal else "0"};c<{output.shape[1]};c+={32 if metal else 1}) p{len(inputs)}[r*{output.view.row_stride}+c*{output.view.column_stride}]={emit(expression, "c")};')
        lines.append('}' if metal else '}}')
        return '\n'.join(lines)
