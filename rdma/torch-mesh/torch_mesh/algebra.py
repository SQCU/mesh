"""Numerical graph rewrites shared by tensor frontends, before native lowering."""
import math
from functools import partial

import torch
from torch import _prims_common as types
from torch._decomp import get_decompositions
from torch._inductor.decomposition import decompositions as native_decompositions
# The provider's decomposition functions use its registered IR lowerings while
# tracing (for example, pooling's offsets declaration).
from torch._inductor import lowering as _native_lowerings
from .structured import expand as butterfly


def linear_backward(self, grad_output, weight, output_mask):
    """Linear's requested gradients as products and a row reduction.

    Flatten only the batch axes. Explicit row counts also cover vector inputs
    and empty axes without an ambiguous inferred reshape dimension.
    """
    count = math.prod(self.shape[:-1])
    rows = grad_output.reshape(count, weight.shape[0])
    inputs = self.reshape(count, self.shape[-1]) if output_mask[1] else None
    return ((rows @ (weight.conj() if weight.dtype.is_complex else weight)).reshape(self.shape) if output_mask[0] else None,
            rows.t() @ (inputs.conj() if inputs.dtype.is_complex else inputs) if output_mask[1] else None,
            rows.sum(0) if output_mask[2] else None)


def _normalized_exponential(self, dim, half_to_float, logarithmic):
    """The empty domain and result form are resolved while declaring the graph."""
    values = self.to(types.get_computation_dtype(self.dtype))
    if self.numel() == 0:
        return values if half_to_float else self
    shifted = values - values.amax(dim, keepdim=True)
    weights = shifted.exp()
    total = weights.sum(dim, keepdim=True)
    result = shifted - total.log() if logarithmic else weights / total
    return result if half_to_float else result.to(self.dtype)


def softmax(self, dim, half_to_float):
    """Expose the two scalar reductions without prescribing an operand layout."""
    return _normalized_exponential(self, dim, half_to_float, False)


def log_softmax(self, dim, half_to_float):
    return _normalized_exponential(self, dim, half_to_float, True)


def softmax_backward(grad_output, output, dim, input_dtype):
    computation = types.get_computation_dtype(output.dtype)
    grad, values = grad_output.to(computation), output.to(computation)
    return (values * (grad - (grad * values).sum(dim, keepdim=True))).to(input_dtype)


def log_softmax_backward(grad_output, output, dim, input_dtype):
    computation = types.get_computation_dtype(output.dtype)
    grad, values = grad_output.to(computation), output.to(computation)
    return (grad - values.exp() * grad.sum(dim, keepdim=True)).to(input_dtype)


def vector_norm(self, ord=2, dim=None, keepdim=False, *, dtype=None):
    """Reduce the transformed values, then apply the scalar result operation.

    Sharding may split any reduced axis. Sum/max/min summaries use the same
    collective declarations as those operations anywhere else in the graph.
    """
    computation, result_type = types.reduction_dtypes(self, types.REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT, dtype)
    if ord == 0:
        return (self != 0).sum(dim, keepdim=keepdim, dtype=result_type)
    values = self.to(computation)
    if ord == math.inf:
        return values.abs().amax(dim, keepdim=keepdim).to(result_type)
    if ord == -math.inf:
        return values.abs().amin(dim, keepdim=keepdim).to(result_type)
    if values.dtype.is_complex or ord % 2 != 0:
        values = values.abs()
    return values.pow(ord).sum(dim, keepdim=keepdim).pow(1.0 / ord).to(result_type)


def mean(self, dim=None, keepdim=False, *, dtype=None):
    """The sum and its count share the declared global reduction domain."""
    dims = types.reduction_dims(self.shape, dim or None)
    return torch.sum(self, dims, keepdim=keepdim, dtype=dtype) / math.prod(self.shape[d] for d in dims)


def moments(self, dim=None, unbiased=None, keepdim=False, *, correction=None, root=False, mean=False):
    """Centered moments as sums over the declared global reduction domain.

    The mean and centered-square sum cross partition boundaries as summaries.
    Unequal or empty local shares never change the count or the correction.
    All arithmetic remains in the computation type until the result is cast.
    """
    if isinstance(dim, bool):
        dim, unbiased = None, dim
    if isinstance(dim, int):
        dim = (dim,)
    dims = types.reduction_dims(self.shape, dim or None)
    count = math.prod(self.shape[d] for d in dims) if self.ndim else 1
    correction = types.set_correction(unbiased, correction)
    computation, result_type = types.reduction_dtypes(self, types.REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT)
    values = self.to(computation)
    average = values.sum(dims, keepdim=True) / count
    centered = values - average
    squares = centered.real.square() + centered.imag.square() if values.dtype.is_complex else centered.square()
    result = squares.sum(dims, keepdim=keepdim) / max(0, count - correction)
    if root:
        result = result.sqrt()
    result = result.to(result_type)
    if not mean:
        return result
    average = average if keepdim else average.squeeze(dims)
    return result, average.to(self.dtype)


def index_values(expression, count, device):
    """Reify shared scalar index terms into the native compiler's ordinary IR.

    The compiler fuses this arithmetic into the indexing consumer. No Python
    loop enumerates tensor positions or builds an element-sized index table.
    """
    import operator
    from functools import reduce
    index = torch.arange(count, dtype=torch.int64, device=device)
    binary = {'add': operator.add, 'mul': operator.mul, 'div': operator.floordiv,
              'mod': operator.mod, 'eq': operator.eq, 'le': operator.le,
              'lt': operator.lt, 'ge': operator.ge, 'gt': operator.gt,
              'and': operator.and_, 'or': operator.or_}
    def emit(term):
        if term == 'i':
            return index
        if isinstance(term, (int, bool)):
            return term
        operation, *arguments = term
        values = [emit(value) for value in arguments]
        if operation == 'select':
            condition, yes, no = values
            return torch.where(condition, yes, no) if isinstance(condition, torch.Tensor) else yes if condition else no
        if operation in ('min', 'max'):
            function = torch.minimum if operation == 'min' else torch.maximum
            return reduce(function, [torch.scalar_tensor(value, dtype=torch.int64, device=device)
                                     if not isinstance(value, torch.Tensor) else value for value in values])
        return reduce(binary[operation], values)
    result = emit(expression)
    return result if isinstance(result, torch.Tensor) else torch.full((count,), result, dtype=torch.int64, device=device)


def convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    """Declare the shared window relation to the existing indexed product."""
    from mesh_relation import Algebra, convolution_domain
    from .operand import TensorView, relation_view
    from .matrix import declare
    domain = convolution_domain(Algebra(), tuple(input.shape), tuple(weight.shape), stride=stride,
        padding=padding, dilation=dilation, transposed=transposed, output_padding=output_padding, groups=groups)
    source, count = input, input.numel()
    if domain['input_neutral']:
        from .host import assemble
        zero = input.new_zeros((1,))
        joined = torch.empty((count + 1,), dtype=input.dtype, device=input.device)
        destination = TensorView.of(joined)
        source = assemble([(TensorView.of(input).reshape((count,)), destination.slice(0, 0, count)),
                           (TensorView.of(zero), destination.slice(0, count, 1))])
    left = relation_view(source, domain['canonical_accesses']['input'], domain['matrix_shapes'][0])
    def transform(value, steps):
        view = TensorView.of(value)
        for operation, arguments in steps:
            view = getattr(view, operation)(arguments)
        return view
    right = transform(weight, domain['weight_steps'])
    output = torch.empty(domain['output_shape'], dtype=input.dtype, device=input.device)
    result = declare(left, right, transform(output, domain['output_steps']))
    return result if bias is None else result + bias.reshape((1, domain['output_shape'][1], *([1] * (input.ndim - 2))))


def owned_product(domain, owned, inputs, supplied, output_shape):
    """Lower an owned product domain to the existing mapped matrix operation.

    Inputs are compact views of declared semantic pages. Output factors choose
    work only; their order is mapped back through the owner's original relation.
    Every tile retains its whole reduction axis and writes disjoint result pages.
    """
    from .operand import TensorView, relation_view
    from .host import assemble
    from .matrix import declare
    algebra = owned['work'].algebra
    matrices, canonical = domain['matrix_shapes'], domain['canonical_accesses']
    if not matrices[0][-1]:
        return torch.zeros(tuple(output_shape), dtype=inputs['input'].dtype, device=inputs['input'].device)
    output = torch.empty(tuple(output_shape), dtype=inputs['input'].dtype, device=inputs['input'].device)
    owner = owned['order'].inverse((math.prod(domain['output_shape']),))
    supplied = {name: algebra.selection_relation(selection).inverse((math.prod(selection['shape']),))
                for name, selection in supplied.items()}
    plans, neutral = [], False
    def selected(shape, cuts, access):
        selection = {'shape': list(shape), 'steps': [['select', axis, runs] for axis, runs in cuts]}
        local = list(shape)
        for axis, runs in cuts:
            local[axis] = sum(count * repeats for _, count, _, repeats in runs)
        return tuple(local), algebra.selection_relation(selection).then(access)
    for piece in owned['work'].project(0, 3).products():
        groups, rows, columns = piece.cartesian()
        left_shape, left = selected(matrices[0], ((0, groups), (1, rows)), canonical['input'])
        right_shape, right = selected(matrices[1], ((0, groups), (2, columns)), canonical['weight'])
        result_shape, result = selected(matrices[2], ((0, groups), (1, rows), (2, columns)), canonical['output'])
        source = left.then(supplied['input'])
        absent = left.domain().binary(source.domain(), 'subtract')
        neutral |= not absent.empty()
        plans.append((left_shape, source, absent, right_shape, right.then(supplied['weight']),
                      result_shape, result.then(owner)))
    source = inputs['input']
    neutral_at = source.numel()
    if neutral:
        zero = source.storage.new_zeros((1,))
        joined = torch.empty((neutral_at + 1,), dtype=source.dtype, device=source.device)
        destination = TensorView.of(joined)
        source = TensorView.of(assemble([(source.reshape((neutral_at,)), destination.slice(0, 0, neutral_at)),
                                        (TensorView.of(zero), destination.slice(0, neutral_at, 1))]))
    pieces = []
    for left_shape, left, absent, right_shape, right, result_shape, result in plans:
        if not absent.empty():
            neutral_relation = algebra.map(left.shape, (neutral_at + 1,), lambda x, y: [f'{y[0]} = {neutral_at}'])
            left = left.union(neutral_relation.subtract_domain(neutral_relation.domain().binary(absent, 'subtract')))
        target = torch.empty(result_shape, dtype=output.dtype, device=output.device)
        value = declare(relation_view(source, left, left_shape),
                        relation_view(inputs['weight'], right, right_shape), TensorView.of(target))
        pieces.append((TensorView.of(value), relation_view(output, result, result_shape)))
    return assemble(pieces) if pieces else output


# These mathematical declarations run before ownership propagation. The local
# native lowering then extends them with its existing Core ATen decomposition
# table. Lowering to scalar prims before partitioning would erase the tensor
# operations whose ownership relations the distributed frontend consumes.
partition_decompositions = {
    torch.ops.aten.mean.default: mean,
    torch.ops.aten.mean.dim: mean,
    torch.ops.aten.linear_backward.default: linear_backward,
    torch.ops.torch_mesh.fwht.default: butterfly,
    torch.ops.aten._softmax.default: softmax,
    torch.ops.aten._log_softmax.default: log_softmax,
    torch.ops.aten._softmax_backward_data.default: softmax_backward,
    torch.ops.aten._log_softmax_backward_data.default: log_softmax_backward,
    torch.ops.aten.linalg_vector_norm.default: vector_norm}

for name, root, mean in (('var', False, False), ('std', True, False),
                         ('var_mean', False, True), ('std_mean', True, True)):
    for overload in ('default', 'dim', 'correction'):
        partition_decompositions[getattr(getattr(torch.ops.aten, name), overload)] = partial(moments, root=root, mean=mean)

decompositions = {**native_decompositions,
                  **get_decompositions([torch.ops.aten.baddbmm]),
                  torch.ops.aten._scaled_dot_product_attention_math_for_mps.default:
                      native_decompositions[torch.ops.aten._scaled_dot_product_attention_math.default],
                  **partition_decompositions,
                  torch.ops.aten.convolution.default: convolution}
