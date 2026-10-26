#!/usr/bin/env python3
# docs/measurement.md#the-torch-parallel-runs
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_mesh
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import PrepareModuleInput, parallelize_module
from torch_mesh import partition
from torch_mesh.attention import attention
from torch_mesh.partition import ColwiseParallel, RowwiseParallel, SequenceParallel
from torch_mesh.configuration import placement, bind, record
from torch_mesh import _stream
from mesh_statistics import now

DIM, HEADS, FFN, LAYERS, BATCH, SEQ = 2048, 16, 5632, 4, 2, 1024
SP = os.environ.get("SP", "1") == "1"
STEPS = int(os.environ.get("STEPS", "64"))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = DIM // HEADS
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(DIM, DIM, bias=False) for _ in range(4))

    def forward(self, x, attend=attention):
        b, s, _ = x.shape
        def heads(w):
            projected = w(x)
            return projected.view(b, s, projected.shape[-1] // self.head_dim, self.head_dim).transpose(1, 2)
        q, k, v = (heads(w) for w in (self.wq, self.wk, self.wv))
        o = attend(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).reshape(b, s, o.shape[1] * o.shape[3]))


class FeedForward(nn.Module):
    def __init__(self, width=FFN):
        super().__init__()
        self.w1, self.w3 = nn.Linear(DIM, width, bias=False), nn.Linear(DIM, width, bias=False)
        self.w2 = nn.Linear(width, DIM, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm, self.ffn_norm = nn.RMSNorm(DIM), nn.RMSNorm(DIM)
        self.attention, self.feed_forward = Attention(), FeedForward()

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        return h + self.feed_forward(self.ffn_norm(h))


def plan():
    if not SP:
        return {"attention.wq": ColwiseParallel(), "attention.wk": ColwiseParallel(), "attention.wv": ColwiseParallel(),
                "attention.wo": RowwiseParallel(), "feed_forward.w1": ColwiseParallel(), "feed_forward.w3": ColwiseParallel(),
                "feed_forward.w2": RowwiseParallel()}
    return {"attention_norm": SequenceParallel(),
            "attention": PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),)),
            "attention.wq": ColwiseParallel(), "attention.wk": ColwiseParallel(), "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),)),
            "feed_forward.w1": ColwiseParallel(), "feed_forward.w3": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1))}


# docs/measurement.md#the-torch-parallel-runs
def drawn(t):
    local = t.to_local() if isinstance(t, DTensor) else t
    held = torch_mesh.empty(local.shape, dtype=local.dtype).normal_(std=t.shape[-1] ** -0.5)
    return partition.from_local(held, t.device_mesh, t.placements, shape=t.shape, stride=t.stride()) if isinstance(t, DTensor) else held


# docs/measurement.md#the-torch-parallel-runs
def held(model):
    for module in model.modules():
        for name, p in list(module.named_parameters(recurse=False)):
            setattr(module, name, nn.Parameter(drawn(p)))
    return model


def step(model, x):
    model.zero_grad(set_to_none=True)
    out = model(x)
    out.sum().backward()
    return out


def main():
    dist.init_process_group(backend=os.environ.get("BACKEND", "mesh"))
    mesh = DeviceMesh.from_group(dist.group.WORLD, "mps", mesh_dim_names=("tp",))
    partition.attach(mesh, tp=HEADS)
    configuration = placement(mesh, 'tp',
        {'function': 'transformer_forward_backward', 'steps': STEPS, 'layers': LAYERS, 'batch': BATCH,
         'sequence': SEQ, 'width': DIM, 'heads': HEADS, 'ffn': FFN, 'sequence_parallel': SP},
        os.environ.get('MESH_OBSERVATIONS', '~/.cache/mesh/observations/tp.jsonl'))
    with torch.device("meta"):
        model = nn.Sequential(*(Block() for _ in range(LAYERS)))
        whole = torch.empty(BATCH, SEQ, DIM)
    for block in model:
        parallelize_module(block, mesh, plan(), src_data_rank=None)
    held(model)
    x = drawn(partition.distribute_tensor(whole, mesh, [Shard(1) if SP else Replicate()], src_data_rank=None))
    from torch_mesh.declarations import Function
    function = Function(lambda: step(model, x.to_local()), parameters=model.parameters())
    bind(configuration, lambda: function.declare(mesh.get_group('tp')), lambda: function.prepare(STEPS))
    started = now()
    for _ in range(STEPS):
        function()
    completion = _stream.completion()
    _stream.publish_boundary()
    record(configuration, completion, started, STEPS * BATCH * SEQ * DIM * x.element_size(),
           {'program': 'tp', 'steps': STEPS, 'parts': configuration['assignment']['parts']['tp']})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
