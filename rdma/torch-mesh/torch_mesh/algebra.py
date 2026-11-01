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


def concatenate_pages(values, dim=0):
    """Concatenation's value equation becomes shared page-placement equalities.

    Capture/autograd have already supplied its shape and derivative. This
    normalization introduces no numerical copy; consumers receive these same
    pages, with any actual contiguous-operand requirement solved by placement.
    """
    from functools import reduce
    from .host import assemble
    from .operand import TensorView
    exemplar = next((value for value in values if value.ndim != 1 or value.numel()), values[0])
    dim %= exemplar.ndim
    dtype = reduce(torch.promote_types, (value.dtype for value in values))
    selected = [value.to(dtype) for value in values if value.ndim != 1 or value.numel()]
    if not selected:
        return values[0].to(dtype)
    shape = list(exemplar.shape)
    shape[dim] = sum(value.shape[dim] for value in selected)
    # Each piece's target is a TensorView over the destination itself, as convolution's and _page_tensor's are.
    # Changed by the LSTM demonstration session (the bc2f34de form passed destination.narrow views): inside the
    # normalizer's dispatch a view has no _base, so operand.signature took the first narrowed piece (offset 0) for
    # the storage root and the concatenation's result was its first operand (lstm-service's [x, h] became x, every
    # rank, 2026-11-01). What must stay: the targets carry the destination as their storage, so the result handle is
    # the whole destination.
    destination = TensorView.of(torch.empty(shape, dtype=dtype, device=exemplar.device))
    pieces, first = [], 0
    for value in selected:
        pieces.append((value, destination.slice(dim, first, value.shape[dim])))
        first += value.shape[dim]
    return assemble(pieces)


def product_implementation(dtype):
    """The configured numerical equation family for a typed contraction."""
    from mesh_compute import assigned_implementation, implementation_assignment
    return assigned_implementation(dict(arithmetic_types=[str(dtype).removeprefix('torch.')]),
                                   implementation_assignment()['product'])


def real_components(value):
    """Complex storage is the same pages viewed as pairs of real components."""
    scalar = types.corresponding_real_dtype(value.dtype)
    return value.view(scalar).reshape((*value.shape, 2))


def complex_components(value):
    scalar = types.corresponding_complex_dtype(value.dtype)
    return value.reshape((*value.shape[:-1], 2)).view(scalar).reshape(value.shape[:-1])


def component_product(left, right):
    a, b = real_components(left), real_components(right)
    ar, ai, br, bi = a[..., 0], a[..., 1], b[..., 0], b[..., 1]
    dot = lambda x, y: (x.unsqueeze(-1) * y.unsqueeze(-3)).sum(-2, dtype=x.dtype)
    return dot(ar, br) - dot(ai, bi), dot(ar, bi) + dot(ai, br)


def component_scale(real, imag, factor):
    return real * factor.real - imag * factor.imag, real * factor.imag + imag * factor.real


def contraction(left, right, out_dtype=None):
    """One typed scalar contraction; the native compiler owns its tiling.

    Integer sums retain their declared scalar width. Complex arithmetic uses
    real page components, so the lower compiler sees ordinary typed loads,
    products and reductions and never an opaque complex executor.
    """
    dtype = left.dtype if out_dtype is None else out_dtype
    if left.dtype.is_complex:
        real, imag = component_product(left, right)
        scalar = types.corresponding_real_dtype(dtype)
        return complex_components(torch.stack((real.to(scalar), imag.to(scalar)), -1))
    values = left.unsqueeze(-1) * right.unsqueeze(-3)
    return values.sum(-2, dtype=dtype)


def matrix_product(left, right, out_dtype=None):
    if product_implementation(left.dtype) == 'metal_product':
        return NotImplemented
    return contraction(left, right, out_dtype)


def matrix_add(bias, left, right, out_dtype=None, beta=1, alpha=1):
    if product_implementation(left.dtype) == 'metal_product':
        return NotImplemented
    if left.dtype.is_complex:
        real, imag = component_scale(*component_product(left, right), alpha)
        if beta != 0:
            values = real_components(bias)
            br, bi = component_scale(values[..., 0], values[..., 1], beta)
            real, imag = real + br, imag + bi
        scalar = types.corresponding_real_dtype(left.dtype if out_dtype is None else out_dtype)
        return complex_components(torch.stack((real.to(scalar), imag.to(scalar)), -1))
    product = contraction(left, right, out_dtype)
    # beta=0 does not read bias (including its NaNs), as the source declares.
    return product * alpha if beta == 0 else product * alpha + bias * beta


def declared_linear(x, w, bias=None, act=0, x1=None):
    if product_implementation(x.dtype) == 'metal_product':
        return NotImplemented
    source = x if x1 is None else torch.cat((x, x1), -1)
    result = contraction(source, w.t())
    if bias is not None:
        result = result + bias
    return result.tanh() if act else result


def declared_linears(xs, ws, biases, acts, x1s):
    if all(product_implementation(x.dtype) == 'metal_product' for x in xs):
        return NotImplemented
    return [torch.ops.torch_mesh.linear.default(x, w, bias, act, x1)
            for x, w, bias, act, x1 in zip(xs, ws, biases, acts, x1s, strict=True)]


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
    index = torch.arange(count, dtype=torch.int64, device=device)
    binary = {'add': operator.add, 'mul': operator.mul, 'div': operator.floordiv,
              'mod': operator.mod, 'eq': operator.eq, 'le': operator.le,
              'lt': operator.lt, 'ge': operator.ge, 'gt': operator.gt,
              'and': operator.and_, 'or': operator.or_}
    from .operand import index_expression
    def select(condition, yes, no):
        return torch.where(condition, yes, no) if isinstance(condition, torch.Tensor) else yes if condition else no
    def extremum(function, left, right):
        values = [value if isinstance(value, torch.Tensor) else torch.scalar_tensor(value, dtype=torch.int64, device=device)
                  for value in (left, right)]
        return function(*values)
    result = index_expression(expression, index, dict(binary, select=select,
        min=lambda left, right: extremum(torch.minimum, left, right),
        max=lambda left, right: extremum(torch.maximum, left, right)))
    return result if isinstance(result, torch.Tensor) else torch.full((count,), result, dtype=torch.int64, device=device)


def view_dtype(self, dtype):
    """A changed scalar width declares the same backing bytes with a new type."""
    if self.dtype.itemsize == dtype.itemsize:
        return NotImplemented
    from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
    from .host import assemble
    # Ask the existing metadata dispatcher for the precise view contract and
    # its validity constraints; no payload is read and no source op is retraced.
    with disable_proxy_modes_tracing():
        described = self.view(dtype)
    unit = math.lcm(self.dtype.itemsize, dtype.itemsize)
    extent = self.untyped_storage().nbytes() // unit * unit
    source = self.as_strided((extent // self.dtype.itemsize,), (1,), 0)
    target = torch.empty((extent // dtype.itemsize,), dtype=dtype, device=self.device)
    storage = assemble(((source, target),))
    return storage.as_strided(described.shape, described.stride(), described.storage_offset())


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


# These mathematical declarations run before ownership propagation. The local
# native lowering then extends them with its existing Core ATen decomposition
# table. Lowering to scalar prims before partitioning would erase the tensor
# operations whose ownership relations the distributed frontend consumes.
partition_decompositions = {
    torch.ops.aten.mm.default: matrix_product,
    torch.ops.aten.bmm.default: matrix_product,
    torch.ops.aten.addmm.default: matrix_add,
    torch.ops.torch_mesh.linear.default: declared_linear,
    torch.ops.torch_mesh.linears.default: declared_linears,
    torch.ops.aten.view_as_real.default: real_components,
    torch.ops.aten.view_as_complex.default: complex_components,
    torch.ops.aten.mean.default: mean,
    torch.ops.aten.mean.dim: mean,
    torch.ops.aten.linear_backward.default: linear_backward,
    torch.ops.torch_mesh.fwht.default: butterfly,
    torch.ops.aten._softmax.default: softmax,
    torch.ops.aten._log_softmax.default: log_softmax,
    torch.ops.aten._softmax_backward_data.default: softmax_backward,
    torch.ops.aten._log_softmax_backward_data.default: log_softmax_backward,
    torch.ops.aten.linalg_vector_norm.default: vector_norm}

# Attention is the same declared products, masks and reductions before owner
# assignment, including the CPU capture form and grouped-query head sharing.
# No placement-specific attention executor or graph break is required.
partition_decompositions.update(get_decompositions([
    torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default]))
for operation in (torch.ops.aten.scaled_dot_product_attention.default,
                  torch.ops.aten._scaled_dot_product_attention_math.default):
    partition_decompositions[operation] = native_decompositions[operation]
partition_decompositions[torch.ops.aten._scaled_dot_product_attention_math_for_mps.default] = (
    native_decompositions[torch.ops.aten._scaled_dot_product_attention_math.default])

for name, root, mean in (('var', False, False), ('std', True, False),
                         ('var_mean', False, True), ('std_mean', True, True)):
    for overload in ('default', 'dim', 'correction'):
        partition_decompositions[getattr(getattr(torch.ops.aten, name), overload)] = partial(moments, root=root, mean=mean)

decompositions = {**native_decompositions,
    torch.ops.aten.cat.default: concatenate_pages,
                  **get_decompositions([torch.ops.aten.baddbmm]),
                  torch.ops.aten._scaled_dot_product_attention_math_for_mps.default:
                      native_decompositions[torch.ops.aten._scaled_dot_product_attention_math.default],
                  **partition_decompositions,
                  torch.ops.aten.view.dtype: view_dtype,
                  torch.ops.aten.convolution.default: convolution}
