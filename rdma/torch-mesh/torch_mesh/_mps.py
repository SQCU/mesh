"""What PyTorch's parallelism APIs need of MPS tensors that torch 2.14 lacks, registered with the backend.

DeviceMesh(device_type="mps") asks the device module whether a device is already selected and, if not,
selects one: torch/distributed/device_mesh.py, DeviceMesh._setup_world_group_and_device calls
device_handle.is_initialized() and then set_device() (device_mesh.py:496, 505, 530), and torch.mps has
neither.  A process has one Metal device, in use once MPS is available.

DTensor's backward through nn.Linear: on MPS, aten::linear has a kernel of its own whose autograd formula
is aten::linear_backward (an MPS-only kernel), which DTensor has no sharding strategy for
(NotImplementedError in its sharding propagation); on CPU and CUDA linear decomposes into matmul, whose
backward DTensor shards.  Here DTensor computes linear_backward as those matmuls.

Context parallelism's SDPA: torch's context_parallel
(torch/distributed/tensor/experimental/_context_parallel/_attention.py) gives the DTensor dispatcher a
handler for CUDA's fused SDPA ops alone (custom_ops: _scaled_dot_product_flash, _efficient and
_cudnn_attention), each a block op whose second output is the logsumexp that the ring's merge
(_SDPAMerger) needs.  Without gradient, F.scaled_dot_product_attention reaches
aten._scaled_dot_product_flash_attention_for_cpu on CPU tensors and
aten._scaled_dot_product_attention_math_for_mps on MPS tensors; neither has a sharding strategy, so the
call fails.  Here each gets the handler CUDA's ops get: torch's own ring (_templated_ring_attention, its
rotation and merge) over the query's sequence shard, each block by an op that returns the output and the
logsumexp: CPU's flash kernel as it is; on MPS, whose SDPA returns no logsumexp, the block's scores rows
at a time (at most 2^26 live).  On a partitioned mesh dimension (partition.py) the ring is torch's with
each source's block at its own length.  With gradient, SDPA on MPS decomposes into matmul and softmax before the
dispatcher sees it, and CP's backward handlers are CUDA's too: forward only."""
import torch
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.experimental._context_parallel import _attention

from . import partition

aten = torch.ops.aten
SCORES = 1 << 26


def _cpu_block(query, key, value, is_causal=False, scale=None, **_):
    return aten._scaled_dot_product_flash_attention_for_cpu(query, key, value, is_causal=is_causal, scale=scale)


def _mps_block(query, key, value, is_causal=False, scale=None, **_):
    """softmax(q k^T scale) v and each query row's logsumexp, SCORES scores at a time."""
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
        top = torch.logsumexp(scores, -1, keepdim=True)
        lse[..., at:at + step] = top.squeeze(-1)
        out[..., at:at + step, :] = torch.exp(scores - top).to(value.dtype) @ value
    return out, lse


_BLOCKS = {aten._scaled_dot_product_flash_attention_for_cpu.default: _cpu_block,
           aten._scaled_dot_product_attention_math_for_mps.default: _mps_block}


def _spec(like, shape, dtype):
    stride = torch.empty(shape, device="meta").stride()
    return DTensorSpec(like.mesh, like.placements, TensorMeta(torch.Size(shape), stride, dtype))


def _ring(op_call, args, kwargs):
    """The SDPA op on DTensors sharded along the sequence (Shard(2)): torch's ring over their shards."""
    named = dict(zip((a.name for a in op_call._schema.arguments), args), **kwargs)
    query, key, value = named["query"], named["key"], named["value"]
    if (named.get("attn_mask") is not None or named.get("dropout_p", 0.0) or named.get("dropout_mask") is not None
            or named.get("enable_gqa", False)):
        raise NotImplementedError("mesh: context-parallel SDPA takes no attn_mask, dropout or enable_gqa")
    spec = query._spec
    if spec.mesh.ndim != 1 or any(t._spec.placements != (Shard(2),) for t in (query, key, value)):
        raise NotImplementedError(f"mesh: context-parallel SDPA on Shard(2) of a 1-D mesh, not {spec.placements}")
    local = (query._local_tensor, key._local_tensor, value._local_tensor)
    options, n = dict(is_causal=named.get("is_causal", False), scale=named.get("scale")), spec.mesh.size()
    if isinstance(n, partition._Chunks):
        lengths = partition._split(n, query.shape[2])[0]
        out, lse = partition._ring_attention(spec.mesh.get_group(), _BLOCKS[op_call], *local, lengths, **options)
    else:
        out, lse = _attention._templated_ring_attention(spec.mesh.get_group(), 2, _BLOCKS[op_call], *local, **options)
    out = DTensor(out, _spec(spec, query.shape[:-1] + value.shape[-1:], out.dtype), requires_grad=False)
    if op_call is aten._scaled_dot_product_attention_math_for_mps.default:
        return out, None
    return out, DTensor(lse, _spec(spec, query.shape[:-1], lse.dtype), requires_grad=False)


def _linear_backward(op_call, args, kwargs):
    """aten.linear_backward on DTensors as the matmuls it is: grad @ weight, grad^T @ input, grad summed."""
    named = dict(zip((a.name for a in op_call._schema.arguments), args), **kwargs)
    x, grad, weight, mask = named["self"], named["grad_output"], named["weight"], named["output_mask"]
    rows = grad.flatten(0, -2)
    return (grad @ weight if mask[0] else None, rows.t() @ x.flatten(0, -2) if mask[1] else None,
            rows.sum(0) if mask[2] else None)


def register():
    """torch.mps.is_initialized; DTensor's linear_backward; the SDPA handlers among those context_parallel
    installs while it is active."""
    if not hasattr(torch.mps, "is_initialized"):
        torch.mps.is_initialized = torch.backends.mps.is_available
    DTensor._op_dispatcher._custom_op_handlers[aten.linear_backward.default] = _linear_backward
    for op in _BLOCKS:
        _attention.custom_ops[op] = _ring
