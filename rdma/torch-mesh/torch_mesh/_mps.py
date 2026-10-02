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
at a time (at most 2^26 live).  On a partitioned mesh dimension (partition.py) the ring is torch's over
capacity blocks, each source's keys past their valid extents masked, the block computed as MPS's is on
either device.  Under torch.compile the handler is a graph break (_ring).  With gradient, SDPA on MPS
decomposes into matmul and softmax before the dispatcher sees it, and CP's backward handlers are CUDA's
too: forward only."""
import importlib.abc
import importlib.util
import math
import os
import sys

import torch
from torch._subclasses.fake_tensor import FakeTensor, UnsupportedOperatorException
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.experimental._context_parallel import _attention

from . import partition

aten = torch.ops.aten
SCORES = 1 << 26


def _cpu_block(query, key, value, is_causal=False, scale=None, **_):
    return aten._scaled_dot_product_flash_attention_for_cpu(query, key, value, is_causal=is_causal, scale=scale)


def _mps_block(query, key, value, is_causal=False, scale=None, key_valid=None, **_):
    """softmax(q k^T scale) v and each query row's logsumexp, SCORES scores at a time; keys where
    key_valid (a bool per key) is False are masked."""
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


_BLOCKS = {aten._scaled_dot_product_flash_attention_for_cpu.default: _cpu_block,
           aten._scaled_dot_product_attention_math_for_mps.default: _mps_block}


def _spec(like, shape, dtype):
    stride = torch.empty(shape, device="meta").stride()
    return DTensorSpec(like.mesh, like.placements, TensorMeta(torch.Size(shape), stride, dtype))


def _ring(op_call, args, kwargs):
    """The SDPA op on DTensors sharded along the sequence (Shard(2)): torch's ring over their shards.
    torch.compile's tracing runs ops on fake tensors, and the ring (its collectives, a partition's host-side
    tables) runs only on real ones: under fake tensors the op is refused as unsupported, which dynamo takes
    as a graph break, so the attention runs eagerly between compiled graphs."""
    named = dict(zip((a.name for a in op_call._schema.arguments), args), **kwargs)
    query, key, value = named["query"], named["key"], named["value"]
    if any(isinstance(t._local_tensor, FakeTensor) for t in (query, key, value)):
        raise UnsupportedOperatorException(op_call)
    if (named.get("attn_mask") is not None or named.get("dropout_p", 0.0) or named.get("dropout_mask") is not None
            or named.get("enable_gqa", False)):
        raise NotImplementedError("mesh: context-parallel SDPA takes no attn_mask, dropout or enable_gqa")
    spec = query._spec
    if spec.mesh.ndim != 1 or any(t._spec.placements != (Shard(2),) for t in (query, key, value)):
        raise NotImplementedError(f"mesh: context-parallel SDPA on Shard(2) of a 1-D mesh, not {spec.placements}")
    local = (query._local_tensor, key._local_tensor, value._local_tensor)
    options, n = dict(is_causal=named.get("is_causal", False), scale=named.get("scale")), spec.mesh.size()
    if isinstance(n, partition._Chunks):
        out, lse = partition._ring_attention(spec.mesh.get_group(), n, *local, **options)
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


def _initialize_distributed_mesh(distributed_config):
    """transformers.distributed.utils.initialize_distributed_mesh (transformers 5.18) on MPS: where it refuses MPS
    ("Tensor parallelism is not supported on MPS devices") and, creating the process group itself, would put MPS on
    CPU (_ensure_torch_distributed: "Falling back to CPU"), the group is the mesh backend's (as it gives CUDA NCCL's)
    and the mesh is on MPS."""
    if torch._C._get_accelerator().type != "mps":
        return _transformers["initialize_distributed_mesh"](distributed_config)
    names = [name for name, size in (("pp", distributed_config.pp_size), ("fsdp", distributed_config.fsdp_size),
                                     ("tp", distributed_config.tp_size)) if size > 1]
    shape = [getattr(distributed_config, f"{name}_size") for name in names]
    if not shape:
        return None, None
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="mesh", rank=int(os.environ["RANK"]),
                                             world_size=int(os.environ["WORLD_SIZE"]))
    if math.prod(shape) != torch.distributed.get_world_size():
        raise RuntimeError(f"The parallel mesh requires {math.prod(shape)} processes, but world_size is "
                           f"{torch.distributed.get_world_size()}.")
    mesh = torch.distributed.init_device_mesh("mps", tuple(shape), mesh_dim_names=tuple(names))
    if len(names) > 1:
        mesh._flatten("_".join(names))
    return torch.device("mps", 0), mesh


def _initialize_tensor_parallelism(tp_plan, tp_size=None, device_mesh=None, device_map=None):
    """transformers' deprecated initialize_tensor_parallelism on MPS: its mesh on MPS (it falls back to CPU)."""
    if torch._C._get_accelerator().type != "mps" or device_mesh is not None or tp_plan is None:
        return _transformers["initialize_tensor_parallelism"](tp_plan, tp_size, device_mesh, device_map)
    if device_map is not None:
        raise ValueError("`tp_plan` and `device_map` are mutually exclusive. Choose either one for parallelization.")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="mesh", rank=int(os.environ["RANK"]),
                                             world_size=int(os.environ["WORLD_SIZE"]))
    size = tp_size or torch.distributed.get_world_size()
    return torch.device("mps", 0), torch.distributed.init_device_mesh("mps", (size,))


_transformers = {}


def _patch_transformers(utils):
    """transformers.distributed.utils's two mesh makers replaced on its module, before the modules that import
    them by name (transformers.distributed.mixin, transformers.integrations.tensor_parallel) are loaded, or in
    them too where they already are."""
    if _transformers:
        return
    _transformers.update(initialize_distributed_mesh=utils.initialize_distributed_mesh,
                         initialize_tensor_parallelism=utils.initialize_tensor_parallelism)
    utils.initialize_distributed_mesh = _initialize_distributed_mesh
    utils.initialize_tensor_parallelism = _initialize_tensor_parallelism
    for name, attribute, value in (("transformers.distributed.mixin", "initialize_distributed_mesh", _initialize_distributed_mesh),
                                   ("transformers.integrations.tensor_parallel", "initialize_tensor_parallelism",
                                    _initialize_tensor_parallelism)):
        if name in sys.modules:
            setattr(sys.modules[name], attribute, value)


def _patch_inductor(output_code):
    """A compiled graph's collectives share command buffers while it runs (_stream.mm, batch): its wrapper is
    straight-line code, so a collective left in torch's open buffer is committed by a later one or as the graph ends."""
    from . import backend
    if backend._stream is None:
        return
    graph, batch = output_code.CompiledFxGraph, backend._stream.batch
    call = graph.__call__

    def __call__(self, inputs):
        batch(1)
        try:
            return call(self, inputs)
        finally:
            batch(-1)
    graph.__call__ = __call__


_PATCHES = {"transformers.distributed.utils": _patch_transformers, "torch._inductor.output_code": _patch_inductor}


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """The modules of _PATCHES patched as each is executed, whatever the order of a program's imports."""

    def __init__(self, pending):
        self.pending = pending

    def find_spec(self, name, path, target=None):
        patch = self.pending.pop(name, None)
        if patch is None:
            return None
        if not self.pending:
            sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            patch(module)
        spec.loader.exec_module = exec_module
        return spec


def install():
    """Each module of _PATCHES patched now where it is loaded, else as it is."""
    if any(isinstance(finder, _PatchOnImport) for finder in sys.meta_path):
        return
    pending = {}
    for name, patch in _PATCHES.items():
        if name in sys.modules:
            patch(sys.modules[name])
        else:
            pending[name] = patch
    if pending:
        sys.meta_path.insert(0, _PatchOnImport(pending))


def register():
    """torch.mps.is_initialized and set_device (one Metal device); DTensor's linear_backward; the SDPA handlers
    among those context_parallel installs while it is active; transformers' tensor parallelism on MPS."""
    if not hasattr(torch.mps, "is_initialized"):
        torch.mps.is_initialized = torch.backends.mps.is_available
    if not hasattr(torch.mps, "set_device"):
        torch.mps.set_device = lambda device: None
    install()
    DTensor._op_dispatcher._custom_op_handlers[aten.linear_backward.default] = _linear_backward
    for op in _BLOCKS:
        _attention.custom_ops[op] = _ring
