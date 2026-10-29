"""Numerical graph rewrites shared by tensor frontends, before native lowering."""
import math

import torch
from torch import _prims_common as types
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


decompositions = {torch.ops.aten.linear_backward.default: linear_backward,
                  torch.ops.torch_mesh.fwht.default: butterfly,
                  torch.ops.aten._softmax.default: softmax,
                  torch.ops.aten._log_softmax.default: log_softmax,
                  torch.ops.aten._softmax_backward_data.default: softmax_backward,
                  torch.ops.aten._log_softmax_backward_data.default: log_softmax_backward,
                  torch.ops.aten.linalg_vector_norm.default: vector_norm}
