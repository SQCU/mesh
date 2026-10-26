"""Numerical graph rewrites shared by tensor frontends, before native lowering."""
import math

import torch


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


decompositions = {torch.ops.aten.linear_backward.default: linear_backward}
