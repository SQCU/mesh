"""Attention for a torch program that trains it, its scores in region leases (metal-microbench docs/facts.md: torch's
MPS allocator serves a tensor of 10 MiB or more from a 1 GiB heap the GPU wires whole, and torch's SDPA on MPS, where
autograd records it, is its math form, the scores, their softmax and its gradient each such a tensor of torch's).

  attention(query, key, value, is_causal=False, scale=None)    F.scaled_dot_product_attention's (no mask, dropout or
                                                               GQA): [..., L, E] queries over [..., S, E] keys

The scores are computed scaled (one baddbmm), masked (is_causal: torch's, query i sees keys 0..i) and softmaxed in place
in a lease the backward reads, and the backward writes their gradient into another, in place through the softmax's.
Leases are taken from those of their shape and returned once the work reading them is issued: MPS runs one stream, so
a later writer runs after it, a shape's leases are as many as the attentions whose backward has not run (and one), and
a step queued behind the GPU makes none.  The output and the gradients of query, key and value are torch's tensors.
(attention.metal is decode attention, another kernel.)"""
import math

import torch

_FREE, _MASKS = {}, {}


def _lease(shape, like, dtype=None):
    free = _FREE.get((tuple(shape), dtype or like.dtype))
    if free:
        return free.pop()
    from . import empty
    return empty(tuple(shape), dtype=dtype or like.dtype, device=like.device.type)


def _returned(*leases):
    for lease in leases:
        _FREE.setdefault((tuple(lease.shape), lease.dtype), []).append(lease)


def _scores(query, key, value, is_causal, scale):
    """(the output, the softmaxed scores' lease)."""
    L, S = query.size(-2), key.size(-2)
    scores = _lease(query.shape[:-1] + (S,), query)
    flat = scores.view(-1, L, S)
    torch.baddbmm(flat, query.reshape(-1, L, query.size(-1)), key.reshape(-1, S, key.size(-1)).transpose(1, 2), beta=0,
                  alpha=scale, out=flat)
    if is_causal:
        if (L, S) not in _MASKS:
            _MASKS[L, S] = _lease((L, S), query, torch.bool).fill_(True).triu_(1)
        scores.masked_fill_(_MASKS[L, S], -math.inf)
    torch.ops.aten._softmax.out(scores, -1, False, out=scores)
    return torch.matmul(scores, value), scores


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, is_causal, scale):
        out, ctx.scores = _scores(query, key, value, is_causal, scale)
        ctx.scale = scale
        ctx.save_for_backward(query, key, value)
        return out

    @staticmethod
    def backward(ctx, grad):
        query, key, value = ctx.saved_tensors
        scores, ctx.scores = ctx.scores, None
        grad_value = torch.matmul(scores.transpose(-2, -1), grad)
        grads = torch.matmul(grad, value.transpose(-2, -1), out=_lease(scores.shape, scores))
        torch.ops.aten._softmax_backward_data.out(grads, scores, -1, scores.dtype, grad_input=grads)
        grad_query = torch.matmul(grads, key).mul_(ctx.scale)
        grad_key = torch.matmul(grads.transpose(-2, -1), query).mul_(ctx.scale)
        _returned(scores, grads)
        return grad_query, grad_key, grad_value, None, None


def attention(query, key, value, is_causal=False, scale=None):
    """F.scaled_dot_product_attention(query, key, value, is_causal=is_causal, scale=scale), its scores in region
    leases."""
    scale = query.size(-1) ** -0.5 if scale is None else scale
    if torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value)):
        return _Attention.apply(query, key, value, is_causal, scale)
    out, scores = _scores(query, key, value, is_causal, scale)
    _returned(scores)
    return out
