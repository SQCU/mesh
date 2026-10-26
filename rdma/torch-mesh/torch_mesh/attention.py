"""Training attention declared as part of the complete forward/backward graph.

The native compiler assigns score and gradient storage before execution, along
with every other intermediate in the function. No attention-specific workspace
pool or mask cache owns storage while the function runs.

This has scaled_dot_product_attention's causal convention: query i sees keys
0 through i. Masks, dropout and grouped-query attention are separate operands
not exposed by this declaration. attention.metal supplies the decode kernels.
"""
import math

import torch


def attention(query, key, value, is_causal=False, scale=None):
    """Declare attention and let autograd include its gradients in the graph."""
    scale = query.size(-1) ** -0.5 if scale is None else scale
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if is_causal:
        queries = torch.arange(query.size(-2), device=query.device)
        keys = torch.arange(key.size(-2), device=key.device)
        scores = scores.masked_fill(queries[:, None] < keys[None, :], -math.inf)
    return torch.matmul(scores.softmax(-1), value)
