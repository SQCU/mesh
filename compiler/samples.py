# design/the-compiler.md
import ast
import collections
import copy
import ctypes
import dataclasses
import hashlib
import itertools
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
from multiprocessing.shared_memory import SharedMemory

import torch
from torch import nn
from torch._dynamo.utils import counters
from torch._higher_order_ops.map import map as hop_map
from torch._higher_order_ops.while_loop import while_loop
from torch._ops import HigherOrderOperator
from torch.fx.passes.dialect.common.cse_pass import CSEPass
from torch.profiler import ProfilerActivity, profile
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from torch._inductor import metrics
from torch._inductor.graph import GraphLowering
from torch._ops import OpOverload

from mesh_compiler import archive, bridge, ir, lower_coreml, lower_inductor, lower_metal
from mesh_compiler.devices import TO_COPY
from mesh_compiler.driver import compile_module, lowered
from mesh_compiler.runtime import Program

TOLERANCE = 1e-4
# design/the-compiler.md, order 4
TOLERANCES = {"coreml": 2.0 ** -6}
MPS = torch.backends.mps.is_available()
HERE = pathlib.Path(__file__).parent
EMITTED = HERE / "emitted"
WITH_EFFECTS = torch.ops.higher_order.with_effects


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 32), nn.Tanh(),
                                 nn.Linear(32, 4))

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, width=32, heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, width))

    def forward(self, x):
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + attended)
        return self.norm2(x + self.mlp(x))


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                                      nn.MaxPool2d(2), nn.Conv2d(8, 16, 3), nn.BatchNorm2d(16),
                                      nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(16, 5)
        for norm in (self.features[1], self.features[5]):
            norm.running_mean.normal_()
            norm.running_var.uniform_(0.5, 2.0)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class RunningMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("count", torch.zeros(()))
        self.register_buffer("mean", torch.zeros(8))

    def forward(self, x):
        self.count.add_(1)
        self.mean.add_((x.mean(0) - self.mean) / self.count)
        return x - self.mean


class Solve(nn.Module):
    def forward(self, a, b):
        return torch.linalg.solve(a @ a.mT + torch.eye(a.shape[-1]), b)


class QR(nn.Module):
    def forward(self, a):
        q, r = torch.linalg.qr(a)
        return {"q": q, "r": r, "back": q @ r}


class Recurrent(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(8, 16, batch_first=True)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return out, h


class ShapeDependent(nn.Module):
    def forward(self, x):
        rows = x.reshape(x.shape[0], -1)
        return rows.sum(1, keepdim=True) / x.shape[0] + torch.arange(x.shape[0]).unsqueeze(1)


class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        y = torch.cond(x.sum() > 0, lambda v: self.linear(v).relu(), lambda v: v.cos() - 1, (x,))
        return y * 2


class Loop(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, trips, x):
        def test(i, v):
            return i < trips

        def body(i, v):
            return i + 1, torch.tanh(self.linear(v))

        _, y = while_loop(test, body, (torch.zeros((), dtype=torch.int64), x))
        return y + x


class LoopWithBranch(nn.Module):
    def forward(self, x):
        def body(i, v):
            v = torch.cond(v.sum() > 0, lambda u: u - 1.0, lambda u: u * -0.5 + 0.25, (v,))
            return i + 1, v

        return while_loop(lambda i, v: i < 4, body, (torch.zeros((), dtype=torch.int64), x))[1]


class StaticLoop(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        for trip in range(4):
            x = torch.tanh(self.linear(x)) + trip
        return x


class Indexing(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.randn(10, 8))

    def forward(self, x, index):
        picked = x.gather(1, index)
        rows = self.table.index_select(0, index[:, 0])
        spread = torch.zeros_like(x).scatter_add(1, index, picked)
        return torch.where(x > 0, rows, spread) + x[index[:, 1] % 4]


class MixtureOfExperts(nn.Module):
    def __init__(self, width=16, experts=4, k=2):
        super().__init__()
        self.k = k
        self.gate = nn.Linear(width, experts)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, width))
            for _ in range(experts))

    def forward(self, x):
        scores = self.gate(x)
        top, chosen = scores.topk(self.k, dim=-1)
        weights = torch.zeros_like(scores).scatter(-1, chosen, top.softmax(-1))
        outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        return (outputs * weights.unsqueeze(-2)).sum(-1)


class Mapped(nn.Module):
    def forward(self, xs):
        return hop_map(lambda x: x.sin() + 1, xs)


class PythonBranch(nn.Module):
    def forward(self, x):
        return x.relu() if x.sum() > 0 else x.neg()


class Flipped(nn.Module):
    def forward(self, x):
        return torch.flip(x.sin(), (0,)).cos()


class Rows(nn.Module):
    def forward(self, lengths, values):
        return lengths * 2, values.sin()


class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 1024))

    def forward(self, x):
        return self.net(x)


class MutatesInput(nn.Module):
    def forward(self, x):
        x.add_(1)
        return x * 2


def round_robin(*processors):
    turn = itertools.cycle(processors)
    return lambda node: next(turn)


def halves(first, second, boundary):
    seen = itertools.count()
    return lambda node: first if next(seen) < boundary else second


def by_target(table, default):
    def placement(node):
        return next((p for key, p in table.items() if key in str(node.target)), default)
    return placement


def by_module(table, default):
    def placement(node):
        stack = [str(path) for path, _ in node.meta.get("nn_module_stack", {}).values()]
        where = " ".join(stack + [node.name])
        return next((p for key, p in table.items()
                     if key in where or key.replace(".", "_") in where), default)
    return placement


def r(*shape):
    return torch.randn(*shape)


def index(rows, columns, high):
    return torch.randint(0, high, (rows, columns))


def on(machine, devices):
    return {name: (machine, device) for name, device in devices.items()}


ONE = on(0, {"cpu": "cpu"})
TWO, THREE = on(0, {"p0": "cpu", "p1": "cpu"}), on(0, {"p0": "cpu", "p1": "cpu", "p2": "cpu"})
SHARDS = ("p0", "p1")
MIXED = on(0, {"cpu": "cpu", "gpu": "mps"})
BATCH = {"x": {0: torch.export.Dim("batch", min=2, max=64)}}
EXPERTS = {"experts.0": "p1", "experts.1": "p1", "experts.2": "p2", "experts.3": "p2"}
EXPERTS_MIXED = {"experts.0": "gpu", "experts.1": "gpu"}
ENGINE = on(0, {"cpu": "cpu", "npu": "ane"})
EXPERTS_ENGINE = {"experts.0": "npu", "experts.1": "npu"}
EXPERTS_ENGINES = {**EXPERTS_MIXED, "experts.2": "npu", "experts.3": "npu"}
POSITIVE = torch.full((2, 8), 0.5)
torch.manual_seed(0)
GATHER = [(r(4, 8), index(4, 3, 8)), (r(4, 8), index(4, 3, 8))]

os.environ.update(DIM="32", HEADS="4", FFN="48", LAYERS="2", BATCH="2", SEQ="8", FF_MULT="2", VOCAB="64",
                  EXPERTS="4", TOP_K="2", TOKENS="12", CROSSINGS="3", HIDDEN="16")
sys.path += [str(HERE.parent / "examples"), str(HERE.parent / "rdma")]
import mesh_graph
import cp
import crossing
import ep
import pl
import tp
os.environ.update(K="48", N="48")
import engines


def megatron_with(stated):
    megatron = tp.placement(SHARDS)

    def placement(node):
        value = node.meta.get("val")
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
        return stated.get((node.target, shape)) or megatron(node)
    return placement


def solved_units():
    fixed = compile_module(tp.Training().eval(), (torch.zeros(2, 8, 32),), TWO, tp.placement(SHARDS))
    times = {p: {tp.shares.kind(n): {tp.shares.units(n)[0]: 3.0 if p == "p1" else 1.0} for n in call_nodes(m.graph)}
             for p, m in fixed.modules.items()}
    units = tp.units(fixed, SHARDS, times, lambda node: 0.0, [1.0, 1.0], tp.HEADS)
    assert units.keys() == {"model_0", "model_1"} and all(a > b for a, b in units.values()), units
    return units


def shifted_units():
    units = tp.shares.shifted((3, 1), [0.5, 1.0], lambda j, held: held[j])
    assert units == (2, 2), units
    return {"model_0": units, "model_1": (1, 3)}


LENGTHS = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6])


def ragged_rows():
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), LENGTHS.cumsum(0)))
    (lengths, rows), (values, held) = tp.shares.rows(tp.shares.Ragged(torch.zeros(31, 4), offsets), (1, 3))
    assert lengths.tolist() == LENGTHS.tolist() and (rows, held) == ((0, 3, 8), (0, 8, 31)), (rows, held)
    stated = {"lengths": (0, SHARDS, rows), "values": (0, SHARDS, held)}
    return lambda node: stated.get(node.name, "p0")


BMM = torch.ops.aten.bmm.default
LM = (index(2, 6, 64), torch.arange(6).expand(2, 6), index(2, 6, 64))

CASES = [
    ("mlp", MLP, [(r(5, 16),)], {}),
    ("sequential (input named `input`)", lambda: nn.Sequential(nn.Linear(16, 8), nn.GELU()),
     [(r(5, 16),)], {"processors": TWO, "placement": lambda: round_robin("p0", "p1")}),
    ("encoder block", EncoderBlock, [(r(2, 6, 32),)], {}),
    ("conv + batchnorm eval", ConvNet, [(r(2, 3, 12, 12),)], {}),
    ("buffer mutation x3", RunningMean, [(r(4, 8),), (r(4, 8),), (r(4, 8),)], {}),
    ("linalg.solve", Solve, [(r(6, 6), r(6, 2))], {}),
    ("linalg.qr", QR, [(r(7, 4),)], {}),
    ("lstm (5 steps, traced through)", Recurrent, [(r(2, 5, 8),)], {}),
    ("python for, 4 static trips", StaticLoop, [(r(2, 8),)], {}),
    ("gather/scatter/index/where", Indexing, GATHER, {}),
    ("mixture of experts, top-2 of 4", MixtureOfExperts, [(r(2, 6, 16),), (r(2, 6, 16),)], {}),
    ("mlp | 2 processors pipeline", MLP, [(r(5, 16),)],
     {"processors": TWO, "placement": lambda: halves("p0", "p1", 4)}),
    ("encoder | 3 processors round-robin", EncoderBlock, [(r(2, 6, 32),)],
     {"processors": THREE, "placement": lambda: round_robin("p0", "p1", "p2")}),
    ("buffer mutation x3 | 2 processors", RunningMean, [(r(4, 8),), (r(4, 8),), (r(4, 8),)],
     {"processors": TWO, "placement": lambda: round_robin("p0", "p1")}),
    ("solve | 3 processors round-robin", Solve, [(r(6, 6), r(6, 2))],
     {"processors": THREE, "placement": lambda: round_robin("p0", "p1", "p2")}),
    ("python for | 2 processors", StaticLoop, [(r(2, 8),)],
     {"processors": TWO, "placement": lambda: round_robin("p0", "p1")}),
    ("gather/scatter/... | 2 processors", Indexing, GATHER,
     {"processors": TWO, "placement": lambda: round_robin("p0", "p1")}),
    ("mixture of experts | 3 processors", MixtureOfExperts, [(r(2, 6, 16),), (r(2, 6, 16),)],
     {"processors": THREE, "placement": lambda: by_module(EXPERTS, "p0"), "expect_on": EXPERTS}),
    ("mlp | all on mps", MLP, [(r(5, 16),)], {"processors": on(0, {"gpu": "mps"}), "needs_mps": True}),
    ("conv | conv on mps, rest cpu", ConvNet, [(r(2, 3, 12, 12),)],
     {"processors": MIXED, "needs_mps": True,
      "placement": lambda: by_target({"convolution": "gpu", "p_features": "gpu"}, "cpu")}),
    ("encoder | cpu/mps round-robin", EncoderBlock, [(r(2, 6, 32),)],
     {"processors": MIXED, "needs_mps": True, "placement": lambda: round_robin("cpu", "gpu")}),
    ("mixture of experts | cpu/mps", MixtureOfExperts, [(r(2, 6, 16),), (r(2, 6, 16),)],
     {"processors": MIXED, "needs_mps": True, "expect_on": EXPERTS_MIXED,
      "placement": lambda: by_module(EXPERTS_MIXED, "cpu")}),
    ("buffer mutation x3 | on mps", RunningMean, [(r(4, 8),), (r(4, 8),), (r(4, 8),)],
     {"processors": on(0, {"gpu": "mps"}), "needs_mps": True}),
    ("mlp | all on mps, input on mps", MLP, [(r(5, 16),)],
     {"processors": on(0, {"gpu": "mps"}), "needs_mps": True, "inputs_on": "mps"}),
    ("gather/scatter/... | on mps", Indexing, GATHER,
     {"processors": on(0, {"gpu": "mps"}), "needs_mps": True}),
    ("mlp | 2 processors, one holds no node", MLP, [(r(5, 16),)],
     {"processors": TWO, "placement": lambda: lambda node: "p0"}),
    ("mlp | cpu, ane", MLP, [(r(5, 16),)], {"processors": ENGINE, "placement": lambda: halves("cpu", "npu", 4)}),
    ("mixture of experts | cpu/ane", MixtureOfExperts, [(r(2, 6, 16),), (r(2, 6, 16),)],
     {"processors": ENGINE, "expect_on": EXPERTS_ENGINE, "placement": lambda: by_module(EXPERTS_ENGINE, "cpu")}),
    ("mixture of experts | cpu/mps/ane", MixtureOfExperts, [(r(2, 6, 16),), (r(2, 6, 16),)],
     {"processors": {**MIXED, **ENGINE}, "needs_mps": True, "expect_on": EXPERTS_ENGINES,
      "placement": lambda: by_module(EXPERTS_ENGINES, "cpu")}),
    ("wide mlp | all but the result on ane", Wide, [(r(256, 1024),)],
     {"processors": ENGINE, "placement": lambda: halves("npu", "cpu", 5)}),
    ("mlp | ane then mps", MLP, [(r(5, 16),)],
     {"processors": on(0, {"gpu": "mps", "npu": "ane"}), "needs_mps": True,
      "placement": lambda: halves("npu", "gpu", 4)}),
    ("mlp | mps then ane", MLP, [(r(5, 16),)],
     {"processors": on(0, {"gpu": "mps", "npu": "ane"}), "needs_mps": True,
      "placement": lambda: halves("gpu", "npu", 4)}),
    ("engines | float16 on mps, cpu, ane", lambda: engines.Engines((16, 16, 16)), [(r(8, 48).half() / 4,)],
     {"processors": on(0, {"projection": "mps", "part0": "mps", "part1": "cpu", "part2": "ane"}), "needs_mps": True,
      "tolerance": TOLERANCES["coreml"],
      "placement": lambda: engines.placement("projection", ("part0", "part1", "part2"))}),
    ("mlp dynamic batch", MLP, [(r(5, 16),), (r(11, 16),)],
     {"dynamic_shapes": BATCH, "rejected": True}),
    ("shape arithmetic dynamic batch", ShapeDependent, [(r(5, 3, 4),), (r(9, 3, 4),)],
     {"dynamic_shapes": BATCH, "rejected": True}),
    ("mlp dynamic batch | 2 processors", MLP, [(r(5, 16),), (r(11, 16),)],
     {"dynamic_shapes": BATCH, "rejected": True, "processors": TWO,
      "placement": lambda: halves("p0", "p1", 4)}),
    ("torch.cond", Branch, [(POSITIVE,)], {"rejected": True}),
    ("while_loop", Loop, [(torch.tensor(3), r(2, 8))], {"rejected": True}),
    ("cond inside while_loop", LoopWithBranch, [(POSITIVE,)], {"rejected": True}),
    ("map higher-order operator", Mapped, [(r(3, 4),)], {"rejected": True}),
    ("python branch on tensor data", PythonBranch, [(r(3, 4),)], {"rejected": True}),
    ("input mutation", MutatesInput, [(r(3, 4),)], {"rejected": True}),
    ("megatron | 2 shards", tp.Training, [(r(2, 8, 32) / 32 ** 0.5,)],
     {"processors": TWO, "sharded": True, "placement": lambda: tp.placement(SHARDS)}),
    ("megatron | units 1:3 and 3:1", tp.Training, [(r(2, 8, 32) / 32 ** 0.5,)],
     {"processors": TWO, "sharded": True,
      "placement": lambda: tp.placement(SHARDS, {"model_0": (1, 3), "model_1": (3, 1)})}),
    ("megatron | units solved, p1 slow", tp.Training, [(r(2, 8, 32) / 32 ** 0.5,)],
     {"processors": TWO, "sharded": True, "placement": lambda: tp.placement(SHARDS, solved_units())}),
    ("megatron | units 3:1 held to 1/2", tp.Training, [(r(2, 8, 32) / 32 ** 0.5,)],
     {"processors": TWO, "sharded": True, "placement": lambda: tp.placement(SHARDS, shifted_units())}),
    ("megatron | 2 shards, cpu/mps", tp.Training, [(r(2, 8, 32) / 32 ** 0.5,)],
     {"processors": on(0, {"p0": "cpu", "p1": "mps"}), "sharded": True, "needs_mps": True,
      "placement": lambda: tp.placement(SHARDS)}),
    ("megatron | sum onto the sequence", tp.Training, [(r(2, 8, 32) / 32 ** 0.5,)],
     {"processors": TWO, "sharded": True,
      "placement": lambda: megatron_with({(BMM, (2, 8, 32)): (1, SHARDS, (1, 3))})}),
    ("megatron | scores whole on p1", tp.Training, [(r(2, 8, 32) / 32 ** 0.5,)],
     {"processors": TWO, "sharded": True, "placement": lambda: megatron_with({(BMM, (8, 8, 8)): (0, ("p1",), None)})}),
    ("megatron | resharded 1:1 to 1:3", tp.Training, [(r(2, 8, 32) / 32 ** 0.5,)],
     {"processors": TWO, "sharded": True,
      "placement": lambda: megatron_with({(ATEN.sigmoid.default, (2, 8, 48)): (2, SHARDS, (1, 3))})}),
    ("flip of a sharded value | 1:2", Flipped, [(r(6, 4),)],
     {"processors": TWO, "sharded": True,
      "placement": lambda: lambda node: (0, SHARDS, (1, 2)) if node.name == "x" else "p0"}),
    ("flip of a sharded value | 3 stripes", Flipped, [(r(7, 4),)],
     {"processors": THREE, "sharded": True,
      "placement": lambda: lambda node: (0, tuple(THREE), "striped") if node.name == "x" else "p0"}),
    ("ragged rows | separators, speeds 1:3", Rows, [(LENGTHS, r(31, 4))],
     {"processors": TWO, "sharded": True, "placement": ragged_rows}),
    ("parallel layers | 2 shards", pl.Inference, [(index(2, 8, 64),)],
     {"processors": TWO, "sharded": True, "placement": lambda: pl.placement(SHARDS)}),
    ("parallel layers | units 3:1, mps", pl.Inference, [(index(2, 8, 64),)],
     {"processors": on(0, {"p0": "mps", "p1": "mps"}), "sharded": True, "needs_mps": True,
      "placement": lambda: pl.placement(SHARDS, {"blocks_0": (3, 1)})}),
    ("sequence split | 2 shards", tp.Attention, [(r(1, 8, 32) / 32 ** 0.5,)],
     {"processors": TWO, "sharded": True, "placement": lambda: cp.placement(SHARDS, None)}),
    ("sequence split | 3 shards", tp.Attention, [(r(1, 8, 32) / 32 ** 0.5,)],
     {"processors": THREE, "sharded": True, "placement": lambda: cp.placement(tuple(THREE), None)}),
    ("sequence striped | 2 shards", tp.Attention, [(r(1, 8, 32) / 32 ** 0.5,)],
     {"processors": TWO, "sharded": True, "placement": lambda: cp.placement(SHARDS)}),
    ("sequence striped | 3 shards", tp.Attention, [(r(1, 8, 32) / 32 ** 0.5,)],
     {"processors": THREE, "sharded": True, "placement": lambda: cp.placement(tuple(THREE))}),
    ("crossings | 3, each sharded again", lambda: crossing.Chain(crossing.Crossing()).half(),
     [(r(2, 16).half(), r(2, 16).half())],
    ("megatron | 2 shards, backward", tp.Training, [(r(2, 8, 32) / 32 ** 0.5,)],
     {"processors": TWO, "sharded": True, "loss": 0, "placement": lambda: tp.placement(SHARDS)}),
    ("megatron | units 1:3 and 3:1, backward", tp.Training, [(r(2, 8, 32) / 32 ** 0.5,)],
     {"processors": TWO, "sharded": True, "loss": 0,
      "placement": lambda: tp.placement(SHARDS, {"model_0": (1, 3), "model_1": (3, 1)})}),
    ("causal lm striped | 2 shards, backward", lambda: cp.Causal(6), [LM],
     {"processors": TWO, "sharded": True, "loss": 0, "placement": lambda: cp.placement(SHARDS, sequence=6)}),
    ("causal lm striped | 3 shards, backward", lambda: cp.Causal(6), [LM],
     {"processors": THREE, "sharded": True, "loss": 0, "placement": lambda: cp.placement(tuple(THREE), sequence=6)}),
     {"processors": TWO, "sharded": True, "placement": lambda: crossing.placement(SHARDS)}),
    ("experts | 2 shards", lambda: ep.MoE(2), [(r(2, 6, 32) / 32 ** 0.5,)],
     {"processors": TWO, "sharded": True, "placement": lambda: ep.placement(SHARDS, ep.MoE(2).slots)}),
    ("experts | 3 shards, one empty", lambda: ep.MoE(3), [(r(3, 4, 32) / 32 ** 0.5,)],
     {"processors": THREE, "sharded": True, "placement": lambda: ep.placement(tuple(THREE), ep.MoE(3).slots)}),
]
SAVE = {"mlp": "mlp", "python for, 4 static trips": "static_for",
        "mixture of experts | 3 processors": "moe_three_processors",
        "conv | conv on mps, rest cpu": "conv_cpu_mps"}
PATHS = ("fx direct", "inductor", "archive", "fx pass", "metal", "coreml")
FRESH_CASES = ("mlp", "python for, 4 static trips", "gather/scatter/... | 2 processors",
               "buffer mutation x3 | 2 processors",
               "solve | 3 processors round-robin", "mixture of experts | 3 processors",
               "conv | conv on mps, rest cpu", "megatron | 2 shards")
KEY_CASES = ("mlp", "mlp | 2 processors pipeline", "mixture of experts | 3 processors", "megatron | 2 shards")
OPTION_FLAGS = ("rejected", "needs_mps", "expect_on", "sharded", "not_lowered", "inputs_on", "tolerance")


def errors(reference, got):
    worst_abs = worst_rel = 0.0
    for a, b in zip(pytree.tree_leaves(reference), pytree.tree_leaves(got), strict=True):
        a, b = torch.as_tensor(a).cpu().double(), torch.as_tensor(b).cpu().double()
        if a.shape != b.shape:
            return float("inf"), float("inf")
        difference = (a - b).abs().max().item() if a.numel() else 0.0
        scale = a.abs().max().item() if a.numel() else 0.0
        worst_abs = max(worst_abs, difference)
        worst_rel = max(worst_rel, difference / max(scale, 1e-12))
    return worst_abs, worst_rel


def call_nodes(graph):
    return [n for n in graph.nodes if n.op == "call_function"]


ATEN = torch.ops.aten
CAT, SLICE, SLICE_SCATTER = ATEN.cat.default, ATEN.slice.Tensor, ATEN.slice_scatter.default
PLACED = {ir.SEND, ir.RECV, TO_COPY, CAT, SLICE, SLICE_SCATTER, ATEN.add.Tensor, ATEN.mm.default, ATEN.reshape.default,
          ATEN.new_zeros.default}
BLOCK = ("dimension", "block", "counts", "displs")


def copies(node):
    return node.target is TO_COPY and "device" in node.kwargs


def check_block(sent, arrived):
    assert [sent.meta.get(key) for key in BLOCK] == [arrived.meta.get(key) for key in BLOCK], "a channel's two ends"
    if "counts" not in arrived.meta:
        return
    dimension, block, counts, displs = (arrived.meta[key] for key in BLOCK)
    for index in block if isinstance(block, tuple) else ():
        counts, displs = counts[index], displs[index]
    block = 0 if isinstance(block, tuple) else block
    counts, displs = (counts, displs) if isinstance(counts, tuple) else ((counts,), (displs,))
    assert arrived.meta["val"].shape[dimension] == counts[block], "a block's size is not its count"
    value, readers = sent.args[0], list(arrived.users)
    if copies(value):
        value = value.args[0]
    if len(readers) == 1 and copies(readers[0]):
        arrived, readers = readers[0], list(readers[0].users)
    places = []
    for reader in readers:
        if reader.target is CAT and reader.args[1] == dimension:
            before = reader.args[0][:reader.args[0].index(arrived)]
            places.append(sum(piece.meta["val"].shape[dimension] for piece in before))
        elif reader.target is SLICE_SCATTER and reader.args[1] is arrived:
            places.append(reader.args[3])
    if not places:
        sliced = len(counts) > 1 and value.target is SLICE and value.args[1] == dimension
        places.append(value.args[2] if sliced else 0)
    assert places == [displs[block]] * len(places), "a block is not placed at its displacement"


def check_ir(scheduled, expect_on=None, sharded=False):
    exported = scheduled.exported.graph_module.graph
    assert not any("processor" in n.meta for n in exported.nodes), "the exported graph was altered"
    original = {n.name: n.target for n in call_nodes(exported)}
    where, added, sender, reader, ends, shards, replicated = {}, [], {}, {}, {}, {}, set()
    for processor, module in scheduled.modules.items():
        module.graph.lint()
        for node in module.graph.nodes:
            assert node.op in ("placeholder", "call_function", "output"), node.op
            control = isinstance(node.target, HigherOrderOperator) and node.target is not WITH_EFFECTS
            assert not control, f"control operator {node.target} in a scheduled graph"
            if node.op == "placeholder":
                assert isinstance(node.meta["val"], torch.Tensor), "a graph input that is not a tensor"
                continue
            assert node.meta["processor"] == processor, (node, processor)
            if node.op != "call_function":
                continue
            if node.name in original and (node.target is original[node.name] or sharded):
                assert node.name not in where, f"{node.name} scheduled twice"
                where[node.name] = processor
            elif "constant" in node.meta:
                assert "of" not in node.meta, "a constant's copy is also a shard"
                if node.meta["constant"] not in original:
                    added.append(node)
            elif node.meta.get("of") in original:
                assert processor not in shards.setdefault(node.meta["of"], set()), f"{node.name}: two shards on {processor}"
                shards[node.meta["of"]].add(processor)
            else:
                added.append(node)
            channel = ir.channel_of(node)
            assert channel is None or type(channel) is int, "a channel must be an integer literal"
            if channel is not None:
                side = sender if node.target is ir.SEND else reader
                assert channel not in side, "a channel with two sends or two recvs"
                replicated.add(node.meta["constant"])
                side[channel] = processor
                ends.setdefault(channel, {})[node.target] = node
            for key_, wanted in (expect_on or {}).items():
                stack = [str(path) for path, _ in node.meta.get("nn_module_stack", {}).values()]
                if any(path == key_ or path.startswith(key_ + ".") for path in stack):
                    assert processor == wanted, f"{node.name} of {key_} is on {processor}, not {wanted}"

    devices, machines, host = scheduled.devices, scheduled.machines, torch.device("cpu")
    crossing = sum(devices[sender[c]] != devices[reader[c]] if machines[sender[c]] == machines[reader[c]]
                   else (devices[sender[c]] != host) + (devices[reader[c]] != host) for c in sender)
    copied = [n for n in added if copies(n)]
    assert sender.keys() == reader.keys() == set(range(scheduled.channels)), "channel numbering"
    assert all(sender[c] != reader[c] for c in sender), "a send and its recv on one processor"
    assert len(copied) == crossing, "device copies differ from cross-device channels"
    if sharded:
        assert not where.keys() & shards.keys(), "an exported node both whole and in shards"
        assert where.keys() | shards.keys() == original.keys(), "an exported node is missing from the scheduled graphs"
        assert {n.target for n in added} <= PLACED, "a node that neither export nor a collective produced"
        sent = [end[ir.SEND].args[0] for end in ends.values()]
        values = {(sender[c], (v.args[0] if copies(v) else v).name, reader[c]) for c, v in zip(ends, sent)}
        assert len(values) == scheduled.channels, "channels differ from (value, reading processor) pairs"
        for end in ends.values():
            check_block(end[ir.SEND], end[ir.RECV])
            read = mesh_graph.reach([end[ir.SEND]], lambda n: [ends[ir.channel_of(n)][ir.SEND]] if n.target is ir.RECV
                                    else n.all_input_nodes)
            assert any(n.op == "placeholder" for n in read), "a channel carries a value no input reaches"
    else:
        assert where.keys() == original.keys(), "an exported node is missing from the scheduled graphs"
        where["output"] = scheduled.output_processor
        pairs = {(n.name, where[u.name]) for n in call_nodes(exported) for u in n.users
                 if where[u.name] != where[n.name] and n.name not in replicated and any(
                     m.op == "placeholder" for m in mesh_graph.reach([n], lambda m: m.all_input_nodes))}
        assert len(pairs) == scheduled.channels, "channels differ from (value, reading processor) pairs"
        assert len(added) == 2 * scheduled.channels + len(copied), "a node that export did not produce"
    if len(scheduled.modules) == 1:
        assert not added and not Program(scheduled).argument_queues, "1-processor program differs"

    orders = [call_nodes(m.graph) for m in scheduled.modules.values()]
    at = {node: place for place, node in enumerate(itertools.chain(*orders))}
    edges = [(at[a], at[b]) for order in orders for a, b in zip(order, order[1:])]
    edges += [(at[end[ir.SEND]], at[end[ir.RECV]]) for end in ends.values()]
    assert len(set(mesh_graph.components(len(at), edges))) == len(at), "the per-processor orders deadlock"
    after = mesh_graph.reach([n for n in at if n.target is SLICE_SCATTER], lambda n: [
        *n.users, *([ends[ir.channel_of(n)][ir.RECV]] if n.target is ir.SEND else [])])
    assert not any(n.target is BMM for n in after), "a value is interleaved before a product reads it"
    return len(original), len(copied)


def fx_pass(module):
    module = CSEPass()(module).graph_module
    module.graph.eliminate_dead_code()
    module.graph.lint()
    module.recompile()
    return module


def canonical(module):
    position = {node: index for index, node in enumerate(module.graph.nodes)}

    def literal(value):
        if isinstance(value, torch.fx.Node):
            return {"node": position[value]}
        if isinstance(value, (tuple, list)):
            return [literal(v) for v in value]
        if isinstance(value, dict):
            return {key_: literal(v) for key_, v in value.items()}
        if isinstance(value, (OpOverload, HigherOrderOperator)):
            return {"operator": str(value)}
        return value if value is None or isinstance(value, (bool, int, float)) else str(value)

    def described(value):
        if isinstance(value, torch.Tensor):
            return [str(value.dtype), list(value.shape)]
        return [described(v) for v in value] if isinstance(value, (tuple, list)) else None

    return [[node.op, literal(node.target) if node.op == "call_function" else None,
             literal(list(node.args)), literal(dict(node.kwargs)),
             None if node.op == "output" else described(node.meta.get("val")),
             node.meta.get("processor"), [node.meta.get(key) for key in BLOCK]] for node in module.graph.nodes]


def digests(modules):
    return {processor: hashlib.sha256(json.dumps(canonical(m)).encode()).hexdigest()
            for processor, m in modules.items()}


def state_of(program):
    current = {}
    for name, (key_, tensor) in program.exported.state.items():
        processor, position = program.positions.get(name, [(None, None)])[0]
        current[key_] = tensor if processor is None else program.arguments[processor][position]
    return current


def compiled_program(scheduled):
    functions = lower_inductor.lower(scheduled)
    unwrapped = torch._dynamo.eval_frame.OptimizedModule
    assert all(isinstance(f, unwrapped) for f in functions.values()), "a wrapped function"
    return Program(scheduled, functions)


def archived_program(scheduled):
    with tempfile.TemporaryDirectory() as directory:
        archive.save(directory, scheduled)
        loaded = archive.load(directory)
    assert digests(loaded.modules) == digests(scheduled.modules), "digest changed by the archive"
    return Program(loaded)


class Watched(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        made = function(*args, **(kwargs or {}))
        DISPATCHED[function.name(), any(isinstance(v, torch.Tensor) and v.device.type == "mps"
                                        for v in pytree.tree_leaves((args, kwargs, made)))] += 1
        return made


def watched(function):
    def call(*arguments):
        with Watched():
            return function(*arguments)
    return call


class Profiled(Program):
    def __init__(self, scheduled):
        super().__init__(scheduled, lower_metal.lower(scheduled))
        self.ordered = [f for p, f in self.functions.items() if hasattr(f, "feedback") and p in self.arguments]
        self.functions = {p: f if p == self.caller else watched(f) for p, f in self.functions.items()}

    def __call__(self, *args, **kwargs):
        with profile(activities=[ProfilerActivity.CPU]) as found, Watched():
            results = super().__call__(*args, **kwargs)
        PROFILED.update({event.key: event.count for event in found.key_averages()})
        on_mps = sorted(name for name, on in DISPATCHED if on)
        assert not on_mps, f"torch operators on an mps tensor around a Metal function: {on_mps}"
        for function in self.ordered:
            assert all(d.causes and d.source < d.target < len(function.vertices) for d in function.dependences)
            assert all(0 < start <= end for _, start, end in function.feedback()), function.feedback()
        return results


DISPATCHED, PROFILED, PROFILES = collections.Counter(), collections.Counter(), {}


PLANNED = []


def planned_program(scheduled):
    functions = lowered(scheduled, (lower_metal.lower, lower_coreml.lower))
    PLANNED.extend((processor, functions[processor]) for processor in scheduled.engines)
    return Program(scheduled, functions)


def variants(scheduled, with_inductor):
    passed = lambda: {p: fx_pass(copy.deepcopy(m)) for p, m in scheduled.modules.items()}
    return {
        "fx direct": lambda: Program(scheduled),
        "inductor": (lambda: compiled_program(scheduled)) if with_inductor else None,
        "archive": lambda: archived_program(scheduled),
        "fx pass": lambda: Program(dataclasses.replace(scheduled, modules=passed())),
        "metal": (lambda: Profiled(scheduled))
        if any(device.type == "mps" for device in scheduled.devices.values()) else None,
        "coreml": (lambda: planned_program(scheduled)) if scheduled.engines else None,
    }


RAISED = []
threading.excepthook = lambda args: RAISED.append(f"{args.exc_type.__name__}: {args.exc_value}")


def within(seconds, thunk):
    box = []

    def attempt():
        try:
            box.append(thunk())
        except Exception as error:
            box.append(error)

    caller = threading.Thread(target=attempt, daemon=True)
    caller.start()
    caller.join(seconds)
    return box[0] if box else TimeoutError(f"no result; a thread raised {RAISED[-1:] or 'nothing'}")


def measure(make, module, calls, inputs_on=None):
    reference, runner, worst = copy.deepcopy(module), make(), (0.0, 0.0)
    for args in calls:
        with torch.no_grad():
            expected = reference(*copy.deepcopy(args))
        got = runner(*(a.to(inputs_on) * 1 if inputs_on else a for a in copy.deepcopy(args)))
        state = state_of(runner)
        buffers = {key: value for key, value in reference.named_buffers() if key in state}
        worst = max(worst, errors((expected, buffers), (got, {key: state[key] for key in buffers})))
    return worst


def save_for_review(stem, program):
    for processor, module in program.modules.items():
        (EMITTED / f"{stem}.{processor}.ir.txt").write_text(module.print_readable(print_output=False))
    archive.save(EMITTED / f"{stem}.archive", program)


def build(name, machines=None):
    _, factory, calls, options = next(case for case in CASES if case[0] == name)
    options = {k: v for k, v in options.items() if k not in OPTION_FLAGS}
    processors = options.pop("processors", ONE)
    if machines:
        processors = {processor: (machines[bool(place)], device)
                      for place, (processor, (_, device)) in enumerate(processors.items())}
    placement = options.pop("placement", lambda: lambda node: next(iter(processors)))()
    torch.manual_seed(0)
    module = factory().eval()
    scheduled = compile_module(module, copy.deepcopy(calls[0]), processors, placement, **options)
    return tp.Eager(module, options["loss"]) if "loss" in options else module, calls, scheduled


def run_case(name, with_inductor):
    module, calls, program = build(name)
    options = next(case for case in CASES if case[0] == name)[3]
    (operators, copies), channels = (check_ir(program, options.get("expect_on"), options.get("sharded")),
                                     program.channels)
    if name in SAVE:
        save_for_review(SAVE[name], program)
    results, dynamo, breaks = {}, "", 0
    for path, make in variants(program, with_inductor).items():
        if make is None:
            results[path] = None
            continue
        counters.clear()
        DISPATCHED.clear()
        PROFILED.clear()
        results[path] = within(120, lambda: measure(make, module, calls, options.get("inputs_on")))
        if path == "metal":
            PROFILES[name] = (dict(DISPATCHED), dict(PROFILED))
        if path == "inductor":
            breaks = sum(counters["graph_break"].values())
            dynamo = (f"{counters['stats']['unique_graphs']} graphs / {len(program.modules)} processors, "
                      f"{breaks} breaks")
            BREAK_REASONS.update({k.split(chr(10))[1].strip(): v
                                  for k, v in counters["graph_break"].items()})
            torch._dynamo.reset()
    return operators, channels, copies, results, dynamo, breaks


def inductor_structure(scheduled, call):
    codes = []
    GraphLowering.save_output_code = codes.append
    metrics.reset()
    with torch._inductor.config.patch(force_disable_caches=True):
        Program(scheduled, lower_inductor.lower(scheduled))(*copy.deepcopy(call))
    GraphLowering.save_output_code = None
    torch._dynamo.reset()
    code = "\n".join(codes)
    return (metrics.generated_kernel_count, code.count("empty_strided"),
            code.count("extern_kernels."), code.count("copy_(") + code.count("aten.clone"))


def check_inductor_structure():
    differences = []
    for name, _, calls, options in CASES:
        if options.get("rejected") or "processors" in options:
            continue
        scheduled = build(name)[2]
        exported = scheduled.exported.graph_module
        every = [n.target for n in exported.graph.nodes if n.op == "placeholder"]
        unscheduled = dataclasses.replace(scheduled, modules={"cpu": exported}, inputs={"cpu": every})
        ours, theirs = inductor_structure(scheduled, calls[0]), inductor_structure(unscheduled, calls[0])
        differences += [name] if ours != theirs else []
        print(f"  {name:34} scheduled {ours}  exported {theirs}  {'same' if ours == theirs else 'DIFFERENT'}")
    print(f"  (kernels, buffers, extern kernel calls, copies); cases that differ: {differences or 'none'}")


BREAK_REASONS = collections.Counter()
NOT_LOWERED = []


def cell(result, expected=None, tolerance=TOLERANCE):
    if result is None:
        return "-", True
    if isinstance(result, lower_metal.NotLowered):
        NOT_LOWERED.append(str(result))
        return "not lowered", result.operators == expected
    if isinstance(result, Exception):
        return "FAIL", False
    ok = result[1] <= tolerance or result[0] <= TOLERANCE
    return f"{result[1]:.1e}" if ok else f"BAD {result[1]:.1e}", ok


def planned(since):
    for processor, function in PLANNED[since:]:
        plans = function.plan()
        placed = collections.Counter(f"{operation.split('.')[-1]} on {preferred} of {supported}"
                                     for plan in plans for operation, _, preferred, supported in plan
                                     if operation != "const")
        print(f"    coreml {processor}: {len(plans)} programs, {len(function.transferred)} results "
              f"transferred; {', '.join(f'{count} {name}' for name, count in placed.items())}")


def imports_of(path):
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").startswith("mesh_compiler")):
            base = (node.module or "").replace("mesh_compiler", "").strip(".")
            found |= {base} if base else {alias.name for alias in node.names}
        elif isinstance(node, ast.Import):
            found |= {a.name.split(".", 1)[1] for a in node.names if a.name.startswith("mesh_compiler.")}
    return found


def check_imports():
    direct = {p.stem: imports_of(p) for p in (HERE / "mesh_compiler").glob("*.py")}
    assert not direct["__init__"], "the package __init__ must import nothing"

    def reach(name):
        seen, todo = set(), [name]
        while todo:
            for dep in direct[todo.pop()] - seen:
                seen.add(dep)
                todo.append(dep)
        return seen

    lowerers = {m for m in direct if m.startswith("lower_")}
    compile_side = {"export", "placement", "devices", "schedule", "driver"}
    for module in sorted(direct):
        print(f"  {module:15} imports {sorted(direct[module]) or '-'}")
    for module in compile_side | {"ir", "runtime"}:
        assert not reach(module) & (lowerers | {"archive"}), f"{module} reaches a lowerer or the archive"
    for module in lowerers | {"archive"} | compile_side:
        assert "runtime" not in reach(module), f"{module} reaches the runtime"
    for module in lowerers | {"archive", "schedule", "placement", "devices", "export"}:
        assert reach(module) == {"ir"}, f"{module} imports {reach(module)}"
    assert not reach("bridge") and reach("runtime") == {"ir", "bridge"}
    for name, attributes in (("runtime", {"graph", "nodes"}), ("bridge", {"graph", "modules"})):
        tree = ast.parse((HERE / "mesh_compiler" / f"{name}.py").read_text())
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        touched = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not imported & {"networkx", "torch.fx"} and not touched & attributes, \
            f"{name} must not examine graphs"
    stages = compile_side | lowerers | {"runtime", "archive"}
    between = {(a, b) for a in stages for b in direct[a] & stages}
    assert between == {("driver", stage) for stage in compile_side - {"driver"}}, between
    for path in (HERE / "mesh_compiler").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            relative = isinstance(node, ast.ImportFrom) and node.level > 0
            source = (node.module or "") if relative else ""
            if path.stem in stages and source in stages:
                owner = sys.modules[f"mesh_compiler.{source}"]
                types = [a.name for a in node.names if isinstance(getattr(owner, a.name), type)]
                assert not types, f"{path.stem} imports the type {types} from stage {source}"
    print("  ok: every stage module imports only ir, except driver, which imports the four")
    print("      compile stages for their functions, and the runtime, which imports the bridge")
    print("      module for its channels; the runtime and the bridge module examine no graph")


class Checked(nn.Module):
    def forward(self, x):
        torch.ops.aten._assert_async.msg((x == x).all(), "not a number")
        return x.sin()


class SharedValue(nn.Module):
    def forward(self, x):
        y = x.sin()
        w = (y.cos() + y).exp()
        z = (y + 1.0) @ x
        return z, w


def check_products():
    for name, pairs in (("sequence split | 2 shards", 6), ("sequence split | 3 shards", 12),
                        ("sequence striped | 2 shards", 8), ("sequence striped | 3 shards", 18)):
        scheduled = build(name)[2]
        found = sum(n.meta.get("pairs", 0) for m in scheduled.modules.values() for n in m.graph.nodes if n.target is BMM)
        assert found == pairs, (name, found)
        print(f"  {name:34} {scheduled.channels:3d} channels, {found:3d} of {2 * len(scheduled.modules) ** 2} "
              "(query block, key block) pairs in its two products")
    print("  ok: a pair the causal mask inhibits entirely is in no product; no value is interleaved before a product")


def check_overlapping_calls():
    assert not build("encoder | 3 processors round-robin")[2].feedforward, "every processor is read by the output processor"
    torch.manual_seed(0)
    scheduled = compile_module(Checked(), (r(4),), TWO,
                               lambda node: "p0" if node.name in ("sin", "output") else "p1")
    assert scheduled.feedforward == ["p1"], "the assertion's processor has no path to the output processor"
    gate, seen, checker = threading.Event(), [], scheduled.modules["p1"]

    def late(*arguments):
        gate.wait()
        seen.append(arguments[0][0].item())
        return checker(*arguments)

    program = Program(scheduled, {**scheduled.modules, "p1": late})
    for value in (1.0, 2.0, 3.0):
        program(torch.full((4,), value))
    gate.set()
    deadline = time.time() + 20
    while len(seen) < 3 and time.time() < deadline:
        time.sleep(0.01)
    assert seen == [1.0, 2.0, 3.0], f"the late thread read {seen}"
    print(f"  feedforward processor p1, its thread held back over three calls, read inputs {seen}")
    print("  ok: each call's inputs reached the thread; other processors keep one list of arguments")


def check_shared_storage():
    torch.manual_seed(0)
    module = SharedValue()
    scheduled = compile_module(module, (r(64, 64),), TWO,
                               lambda node: "p0" if node.name in ("add_1", "mm", "output") else "p1")
    received = [n for n in scheduled.modules["p0"].graph.nodes if n.target is ir.RECV]
    assert received, "the output processor must receive"
    compiled = Program(scheduled, lower_inductor.lower(scheduled))
    worst = max(errors(module(x), compiled(x))[1] for x in (r(64, 64) for _ in range(100)))
    assert worst <= TOLERANCE, worst
    print(f"  {len(received)} same-device receives on p0; 100 runs through Inductor, worst rel err {worst:.1e}")
    print("  ok: the receiver's compiled code left the sender's storage intact")


def check_digests_across_processes():
    command = [sys.executable, str(HERE / "samples.py"), "--digests"]
    runs = [json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
            for _ in range(2)]
    here = {name: digests(build(name)[2].modules) for name in KEY_CASES}
    assert runs[0] == runs[1] == here, "digests differ between processes"
    for name, found in here.items():
        print(f"  {name:34} " + " ".join(f"{p}:{d[:12]}" for p, d in found.items()))
    print("  ok: identical in two fresh interpreter processes and in this one")


def reload_case(directory, name):
    _, factory, calls, _ = next(case for case in CASES if case[0] == name)
    torch.manual_seed(0)
    module = factory().eval()
    scheduled = archive.load(directory)
    worst = measure(lambda: Program(scheduled), module, calls)
    print(json.dumps({"abs": worst[0], "rel": worst[1], "digests": digests(scheduled.modules)}))


def check_reload_in_fresh_process():
    for name in FRESH_CASES:
        if "mps" in name and not MPS:
            continue
        program = build(name)[2]
        with tempfile.TemporaryDirectory() as directory:
            archive.save(directory, program)
            size = sum(f.stat().st_size for f in pathlib.Path(directory).iterdir())
            command = [sys.executable, str(HERE / "samples.py"), "--reload", directory, name]
            done = subprocess.run(command, capture_output=True, text=True, check=True)
        report = json.loads(done.stdout.strip().splitlines()[-1])
        assert report["rel"] <= TOLERANCE or report["abs"] <= TOLERANCE, report
        assert report["digests"] == digests(program.modules), "digest changed by reload"
        print(f"  {name:34} {len(program.modules)} archives {size:7d} bytes  rel err "
              f"{report['rel']:.1e}  digests equal")
    print("  ok: loaded and run in a fresh process, matches eager, digests equal the originals")


def between(machines, with_inductor):
    failed = False
    print(f"machines {machines}: the first processor of a case on the first, the others on the second;")
    print("max relative error against eager where the results are, `ran` where they are not:")
    for name, _, _, options in CASES:
        if len(options.get("processors", ONE)) < 2 or options.get("rejected"):
            continue
        if options.get("needs_mps") and not MPS:
            print(f"{name:34} SKIP (no mps)")
            continue
        module, calls, scheduled = build(name, machines)
        check_ir(scheduled, options.get("expect_on"), options.get("sharded"))
        cells = []
        for path, make in variants(scheduled, with_inductor).items():
            if make is None:
                continue

            def outputs(make=make):
                reference, runner, worst = copy.deepcopy(module), make(), None
                for args in calls:
                    with torch.no_grad():
                        expected = reference(*copy.deepcopy(args))
                    got = runner(*copy.deepcopy(args))
                    if got is not None:
                        worst = max(worst or (0.0, 0.0), errors(expected, got))
                return worst

            result = within(120, outputs)
            text, ok = ("ran", True) if result is None else cell(
                result, options.get("not_lowered"), tolerance=options.get("tolerance", TOLERANCES.get(path, TOLERANCE)))
            failed |= not ok
            cells.append(f"{path} {text}")
            if isinstance(result, Exception):
                print(f"    {path}: {type(result).__name__}: {' '.join(str(result).split())[:300]}")
        print(f"{name:34} {scheduled.channels:3d} channels, {len(scheduled.carried):3d} between machines  "
              + "  ".join(cells), flush=True)
    sys.exit(1 if failed else 0)


RECORD_BYTES = 304


def region(node, nodes):
    mesh, pages = bridge.mesh, 2048
    header = mesh.Header(magic=0x4d455348, version=120, pgsz=16384, block=1, rows=pages, node=node,
                         wire_pages=pages // 2, view=1, view_nodes=len(nodes))
    header.algo_off, header.view_off, header.page_off, header.data_off = (
        header.pgsz, 2 * header.pgsz, 3 * header.pgsz, 8 * header.pgsz)
    header.length = header.data_off + pages * header.pgsz
    shared = SharedMemory(create=True, size=header.length, name=f"mesh_compiler_{os.getpid()}_{node}")
    shared.buf[:ctypes.sizeof(header)] = bytes(header)
    for place, other in enumerate(nodes):
        shared.buf[header.view_off + place * RECORD_BYTES:][:4] = other.to_bytes(4, "little")
    context = mesh.Context()
    mesh.check(mesh.LIB.mesh_attach(ctypes.byref(context), f"/{shared.name}".encode()))
    return shared, context


def carry(contexts):
    mesh, wake = bridge.mesh, bridge.SYSTEM.os_sync_wake_by_address_all
    headers = [mesh.Header.from_address(context.M) for context in contexts]
    algos = [mesh.Algo.from_address(context.M + header.algo_off) for context, header in zip(contexts, headers)]
    for algo in algos:
        if algo.retired == algo.published != algo.released:
            algo.released = algo.published
            wake(ctypes.addressof(algo) + mesh.Algo.released.offset, 8, bridge.SHARED)
    if not all(algo.published != algo.retired for algo in algos) or algos[0].group != algos[1].group:
        return
    for place in range(algos[0].plans):
        states = [mesh.State.from_address(context.M + ctypes.c_uint64.from_address(
            context.M + algo.table + 8 * place).value) for context, algo in zip(contexts, algos)]
        lists = [(mesh.Hop * state.entries).from_address(context.M + state.hops)
                 for context, state in zip(contexts, states)]
        if not states[0].entries:
            continue
        source = int(lists[1][0].direction == mesh.SEND)
        (sender, sends, origin, top), (receiver, receives, target, _) = [
            (states[k], lists[k], contexts[k].M + headers[k].data_off, headers[k]) for k in (source, 1 - source)]
        completion = ctypes.c_uint64.from_address(origin - top.data_off + sends[0].count)
        supplied = ctypes.c_uint64.from_address(origin - top.data_off + sends[0].word).value
        execution = completion.value // len(sends)
        if min(supplied, sender.executions, receiver.executions) <= execution:
            continue
        there = target - headers[1 - source].data_off
        for sent, landed in zip(sends, receives, strict=True):
            assert sent.bytes == landed.bytes and landed.pages == len(receives), "the two ends' entries differ"
            assert (sent.node, landed.node) == (1 - source, source), "an entry names the wrong neighbour"
            ctypes.memmove(target + landed.page * top.pgsz, origin + sent.page * top.pgsz, sent.bytes)
            ctypes.c_uint64.from_address(there + landed.word).value ^= 1 << (landed.field + (execution & 1))
        count = ctypes.c_uint64.from_address(there + receives[0].count)
        count.value += len(receives)
        wake(ctypes.addressof(count), 8, bridge.SHARED)
        completion.value += len(sends)


def check_between_machines(with_inductor):
    nodes = [3, 5]
    shared, contexts = zip(*(region(node, nodes) for node in nodes))
    command = [sys.executable, str(HERE / "samples.py"), "--machines", "3", "5"] + ["--fast"] * (not with_inductor)
    runs = [subprocess.Popen(command, stdout=subprocess.PIPE, text=True,
                             env={**os.environ, "MESH_REGION": f"/{region_.name}", "MESH_GROUP": "1"})
            for region_ in shared]
    while any(run.poll() is None for run in runs):
        carry(contexts)
    for node, run, region_ in zip(nodes, runs, shared):
        print("".join(f"  {node}: {line}" for line in run.stdout))
        region_.unlink()
    assert not any(run.returncode for run in runs), "a machine's check failed"
    print("  ok: two processes, each with a region of the bridge's layout, their entries carried by")
    print("      this process; no bridge ran")


def main():
    if "--machines" in sys.argv:
        at = sys.argv.index("--machines")
        between([int(node) for node in sys.argv[at + 1:at + 3]], "--fast" not in sys.argv)
    if "--digests" in sys.argv:
        print(json.dumps({name: digests(build(name)[2].modules) for name in KEY_CASES}))
        return
    if "--reload" in sys.argv:
        reload_case(*sys.argv[sys.argv.index("--reload") + 1:][:2])
        return
    with_inductor = "--fast" not in sys.argv
    failed, total_breaks = False, 0
    print(f"torch {torch.__version__}  mps={MPS}  tolerance={TOLERANCE}, {TOLERANCES}")
    print("max relative error against eager, per execution path of the same scheduled IR;")
    print("nodes = call_function nodes of the exported graph; chan = channels, each one send and")
    print("one recv; copy = device copies; the scheduled graphs hold exactly nodes + 2 chan + copy,")
    print("except where a placement shards (the last cases), whose graphs hold each shard's nodes:")
    print(f"{'case':34} {'nodes':>5} {'chan':>4} {'copy':>4} " + " ".join(f"{p:>11}" for p in PATHS)
          + "  torch.compile")
    for name, _, _, options in CASES:
        rejected = options.get("rejected", False)
        if options.get("needs_mps") and not MPS:
            print(f"{name:34} SKIP (no mps)")
            continue
        started, since = time.time(), len(PLANNED)
        try:
            operators, channels, copies, results, dynamo, breaks = run_case(name, with_inductor)
        except Exception as error:
            failed |= not rejected
            detail = str(error).strip().splitlines()[0][:150]
            print(f"{name:34} {'REJECTED' if rejected else 'ERROR'}: {type(error).__name__}: {detail}")
            continue
        cells = [cell(results[path], options.get("not_lowered"),
                      tolerance=options.get("tolerance", TOLERANCES.get(path, TOLERANCE)))
                 for path in PATHS]
        total_breaks += breaks
        failed |= rejected or breaks > 0 or not all(ok for _, ok in cells)
        print(f"{name:34} {operators:5d} {channels:4d} {copies:4d} " + " ".join(f"{text:>11}" for text, _ in cells)
              + f"  {dynamo} ({time.time() - started:.0f}s)", flush=True)
        planned(since)
        for path in PATHS:
            if isinstance(results[path], Exception):
                detail = " ".join(str(results[path]).split())[:300]
                print(f"    {path}: {type(results[path]).__name__}: {detail}")
    print(f"torch.compile graph breaks over all cases: {total_breaks}")
    print(f"operators with no Metal source lowering, by processor: {NOT_LOWERED or 'none'}")
    print("torch operators during each metal path: dispatched on every processor's thread (how many on an mps")
    print("tensor), and torch.profiler's events on the caller's thread:")
    for name, (dispatched, profiled) in PROFILES.items():
        aten = {key: count for (key, _), count in dispatched.items() if key.startswith("aten::")}
        print(f"  {name:34} {sum(aten.values())} aten calls, {sum(c for (_, on), c in dispatched.items() if on)} on "
              f"an mps tensor; profiler: {dict(sorted(profiled.items()))}")
    for reason, count in BREAK_REASONS.most_common():
        print(f"  {count:5d}  {reason}")
    print("import separation (from the modules' syntax trees):")
    check_imports()
    print("digests of the scheduled GraphModules across processes:")
    check_digests_across_processes()
    if with_inductor:
        print("Inductor's generated code, scheduled against exported, single-processor cases:")
        check_inductor_structure()
    print("attention's products where the sequence is sharded (Striped Attention, section 3.1):")
    check_products()
    print("overlapping calls (a processor with no path to the output processor):")
    check_overlapping_calls()
    if with_inductor:
        print("shared storage (a value read on another processor and again by its sender):")
        check_shared_storage()
    print("torch.export.save of the scheduled programs, reloaded in a fresh process:")
    check_reload_in_fresh_process()
    print("processors on two machines, their entries carried by this process in place of the bridge:")
    check_between_machines(with_inductor)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
