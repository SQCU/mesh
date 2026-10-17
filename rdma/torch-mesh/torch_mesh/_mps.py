"""Ring attention over a group's sequence shards, a function a program calls: ring(group, query, key, value).

torch's context parallelism (torch/distributed/tensor/experimental/_context_parallel/_attention.py) runs its ring
(_templated_ring_attention: the shards' keys and values rotated around the group, the blocks merged by their
logsumexp, _SDPAMerger) for CUDA's fused SDPA ops alone, each a block that returns the logsumexp the merge needs.
ring() runs torch's ring with a block of its own: CPU's flash kernel on CPU tensors, and on MPS, whose SDPA returns no
logsumexp, the block's scores rows at a time (at most SCORES live) with each row's logsumexp (mps_block).  Forward
only: with gradient, SDPA on MPS decomposes into matmul and softmax, and the ring's backward is CUDA's.  Nothing here
is registered with torch, which documents no extension point for a DTensor operator's handler.  mps_block's key_valid
masks a partition's padded keys (torch_mesh.partition's ring over capacity blocks, on either device)."""
import torch
from torch.distributed.tensor.experimental._context_parallel import _attention

aten = torch.ops.aten
SCORES = 1 << 26


def _cpu_block(query, key, value, is_causal=False, scale=None):
    return aten._scaled_dot_product_flash_attention_for_cpu(query, key, value, is_causal=is_causal, scale=scale)


def mps_block(query, key, value, is_causal=False, scale=None, key_valid=None):
    """softmax(q k^T scale) v and each query row's logsumexp, SCORES scores at a time; keys where key_valid (a bool per
    key) is False are masked (a partition's padding: torch_mesh.partition's ring)."""
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    rows, keys = query.shape[-2], key.shape[-2]
    step = max(1, SCORES // (query[..., 0, 0].numel() * keys))
    out = query.new_empty(query.shape[:-1] + value.shape[-1:])
    lse = query.new_empty(query.shape[:-1], dtype=torch.float32)
    kt, positions = key.transpose(-2, -1), torch.arange(keys, device=query.device)
    for at in range(0, rows, step):
        scores = (query[..., at:at + step, :] @ kt).float() * scale
        if is_causal:
            scores.masked_fill_(positions > torch.arange(at, min(at + step, rows), device=query.device)[:, None], float("-inf"))
        if key_valid is not None:
            scores.masked_fill_(~key_valid, float("-inf"))
        top = torch.logsumexp(scores, -1, keepdim=True)
        lse[..., at:at + step] = top.squeeze(-1)
        out[..., at:at + step, :] = torch.exp(scores - top).to(value.dtype) @ value
    return out, lse


def ring(group, query, key, value, is_causal=False, scale=None):
    """This rank's query shard's attention over every rank's keys and values, the shards along dimension 2 (the
    sequence) in the group's rank order, by torch's ring: (output, each query row's logsumexp)."""
    block = mps_block if query.device.type == "mps" else _cpu_block
    out, lse, *_ = _attention._templated_ring_attention(group, 2, block, query, key, value, is_causal=is_causal, scale=scale)
    return out, lse
