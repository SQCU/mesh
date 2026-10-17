"""Kernels a program's kernel_call takes (design/algorithm-sources.md#kernelsexpression): a supplied function
`kernel(program, inputs, outputs)` over Refs, or an expression of the inputs evaluated by torch over each output block
(no source is generated, no second executor: the values, indices and masks of Pallas's indexing).  `matmul` and `dot`
contract on the program's engines (torch_mesh.engines over its Engines operand, else torch's own product); `add`,
`maximum` and `minimum` are the combines the collectives know (libnccl-mesh's sum, max and min)."""
from dataclasses import dataclass

import numpy as np
import torch

from . import _native

_REDUCTIONS = ('sum',)
_REAL_FUNCTIONS = ('exp', 'rsqrt')
_POINTWISE_OPERATIONS = ('+', '-', '*', '/', '<', '<=', '>', '>=', '==', '&', '|', 'select', 'cast', *_REAL_FUNCTIONS)


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
        return _Expression('cast', (self,), _native.dtype(dtype))

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


# design/algorithm-sources.md#kernelsdot
def dot(left, right, *, tile_k=128):
    if tile_k < 1:
        raise ValueError('Contraction K tiles must be positive')
    return _Expression('dot', tuple(map(_literal, (left, right))), tile_k)


# design/algorithm-sources.md#programtensor
def _dense(source):
    if hasattr(source, 'blocks'):
        return source.dense()
    return source.tensor


# design/algorithm-sources.md#kernelsexpression
def _evaluate(program, node, inputs, output, coordinate):
    rows, columns = output.shape
    device = program.device
    values = {}

    # design/algorithm-sources.md#kernelsexpression
    def visit(part):
        if part in values:
            return values[part]
        operation = part.operation
        if operation == 'input':
            value = _dense(inputs[part.value])
            value = value.float() if value.dtype.is_floating_point else value
        elif operation == 'literal':
            value = torch.tensor(part.value, device=device,
                                 dtype=torch.float32 if isinstance(part.value, float) else None)
        elif operation == 'row':
            value = torch.arange(rows, device=device)[:, None]
        elif operation == 'column':
            value = torch.arange(columns, device=device)[None, :]
        elif operation == 'program_id':
            value = torch.tensor(coordinate[part.value], device=device)
        elif operation == 'load':
            source = _dense(inputs[part.value])
            row, column, mask, other = map(visit, part.operands)
            row, column, mask = torch.broadcast_tensors(row, column, mask.bool())
            row = torch.where(mask, row, 0).long().clamp(0, source.shape[0] - 1)
            column = torch.where(mask, column, 0).long().clamp(0, source.shape[1] - 1)
            loaded = source[row, column]
            loaded = loaded.float() if loaded.dtype.is_floating_point else loaded
            value = torch.where(mask, loaded, other)
        elif operation == 'dot':
            left, right = map(visit, part.operands)
            left = left.expand(rows, left.shape[1]) if left.shape[0] == 1 else left
            right = right.expand(right.shape[0], columns) if right.shape[1] == 1 else right
            value = _native.contract(None, None, left, right).float()
        elif operation == 'transpose':
            value = visit(part.operands[0]).T
        elif operation == 'sum':
            value = visit(part.operands[0]).float().sum(dim=1, keepdim=True)
        elif operation == 'cast':
            value = visit(part.operands[0]).to(part.value)
            value = value.float() if part.value.is_floating_point else value
        elif operation == 'exp':
            value = torch.exp(visit(part.operands[0]).float())
        elif operation == 'rsqrt':
            value = torch.rsqrt(visit(part.operands[0]).float())
        elif operation == 'select':
            condition, yes, no = map(visit, part.operands)
            value = torch.where(condition.bool(), yes, no)
        else:
            left, right = map(visit, part.operands)
            if operation == '/' and not (left.is_floating_point() or right.is_floating_point()):
                value = torch.div(left, right, rounding_mode='trunc')
            else:
                value = {'+': torch.add, '-': torch.sub, '*': torch.mul, '/': torch.div, '<': torch.lt,
                         '<=': torch.le, '>': torch.gt, '>=': torch.ge, '==': torch.eq,
                         '&': torch.bitwise_and, '|': torch.bitwise_or}[operation](left, right)
        values[part] = value
        return value

    return visit(node)


@dataclass(frozen=True)
class _ExpressionKernel:
    values: tuple

    # design/algorithm-sources.md#kernelsexpression
    def __call__(self, program, inputs, outputs, coordinate=()):
        if len(outputs) != len(self.values):
            raise ValueError('Each expression requires an output region')
        for value, output in zip(self.values, outputs):
            target = output.tensor
            target.copy_(_evaluate(program, value, inputs, output, coordinate).expand(target.shape))


# design/algorithm-sources.md#kernelsdot
def matmul(program, inputs, outputs):
    x, w = inputs
    output, = outputs
    key = w.key if not hasattr(w, 'blocks') and program._is_constant(w) else None
    output.tensor.copy_(program._contract(_dense(x), _dense(w), key))


# design/algorithm-sources.md#nnembedding
def gather(program, inputs, outputs):
    table, selected = inputs
    output, = outputs
    source, index = _dense(table), _dense(selected)[:, 0].long()
    index = torch.where(index < 0, index + source.shape[0], index)
    output.tensor.copy_(source.index_select(0, index))


_left, _right = arguments(2)
add = expression(_left + _right)
maximum = expression(select(_left.equal(_left), select(_right.equal(_right), select(_left >= _right, _left, _right), _right), _left))
minimum = expression(select(_left.equal(_left), select(_right.equal(_right), select(_left <= _right, _left, _right), _right), _left))
swish = expression(_left / (1 + (0 - _left).exp()))
identity = expression(_left)
del _left, _right
