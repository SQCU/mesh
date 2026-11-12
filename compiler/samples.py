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
import struct
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

from mesh_compiler import archive, bridge, ir, lower_coreml, lower_inductor, lower_metal, signature
from mesh_compiler.devices import TO_COPY
from mesh_compiler.driver import compile_module, compiled, lowered
from mesh_compiler.export import calls
from mesh_compiler.placement import views
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


class Updates(nn.Module):
    def forward(self, x, index):
        put = torch.zeros_like(x).scatter(1, index, 2.5)
        whole = torch.remainder(index, index[:, :1] + 1)
        real = torch.remainder(x, 0.75) + torch.remainder(x, x.abs() + 0.5)
        apart = x.view(2, 2, 8)[index[0, :2] % 2, :, index[1, :2]]
        back = x[index[:, 0] % 4 - 4]
        ordered, places = torch.sort(x, dim=1)
        return put, whole, real, apart, back, ordered, places


class Sums(nn.Module):
    def forward(self, x, index):
        return x.cumsum(1), index.cumsum(1), (x > 0).cumsum(0), (x * 2).cumsum(0)


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


class Twice(nn.Module):
    def forward(self, x):
        return x.sin().cos().exp()


class Reduced(nn.Module):
    def forward(self, x):
        return (x.cos() + 1).sum(0), (x.sin() + 1).sum(1)


class Rows(nn.Module):
    def forward(self, lengths, values):
        return lengths * 2, values.sin()


class Windows(nn.Module):
    def __init__(self):
        super().__init__()
        self.same, self.strided = nn.Conv2d(3, 4, 3, padding=1), nn.Conv2d(4, 4, 5, stride=2, padding=2, dilation=2)

    def forward(self, x):
        y = nn.functional.max_pool2d(self.strided(self.same(x)).relu(), 3, stride=1, padding=1)
        return nn.functional.avg_pool2d(y, 2)


class Ceiled(nn.Module):
    def forward(self, x):
        return nn.functional.max_pool2d(x.sin(), 3, stride=2, padding=1, ceil_mode=True)


class Divided(nn.Module):
    def forward(self, x):
        return nn.functional.avg_pool2d(x.sin(), 3, stride=2, padding=1, count_include_pad=False, divisor_override=5)


class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 1024))

    def forward(self, x):
        return self.net(x)


class Masked(nn.Module):
    def forward(self, x):
        keep = torch.arange(6).view(1, 6, 1) <= 3
        return torch.where(keep, x, torch.zeros((), dtype=x.dtype)) + x.sum(1, keepdim=True)


class Merged(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.randn(6, 4))

    def forward(self, x):
        return x.view(12, 4).sin() + self.table[1:5].reshape(2, 8).sum()


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
FOUR = on(0, {"p0": "cpu", "p1": "cpu", "p2": "cpu", "p3": "cpu"})
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


LM = (index(2, 6, 64), torch.arange(6).expand(2, 6), index(2, 6, 64))
BMM = torch.ops.aten.bmm.default

CASES = [
    ("mlp", MLP, [(r(5, 16),)], {}),
    ("sequential (input named `input`)", lambda: nn.Sequential(nn.Linear(16, 8), nn.GELU()),
     [(r(5, 16),)], {"processors": TWO, "placement": lambda: round_robin("p0", "p1")}),
    ("encoder block", EncoderBlock, [(r(2, 6, 32),)], {}),
    ("conv + batchnorm eval", ConvNet, [(r(2, 3, 12, 12),)], {}),
    ("buffer mutation x3", RunningMean, [(r(4, 8),), (r(4, 8),), (r(4, 8),)], {}),
    ("linalg.solve", Solve, [(r(6, 6), r(6, 2))], {}),
    ("linalg.qr", QR, [(r(7, 4),)], {}),
    ("linalg.solve | on mps", Solve, [(r(6, 6), r(6, 2))], {"processors": on(0, {"gpu": "mps"}), "needs_mps": True}),
    ("linalg.qr | on mps", QR, [(r(7, 4),)], {"processors": on(0, {"gpu": "mps"}), "needs_mps": True}),
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
    ("scatter.value/remainder/index/sort", Updates, GATHER, {}),
    ("scatter.value/remainder/... | on mps", Updates, GATHER, {"processors": on(0, {"gpu": "mps"}), "needs_mps": True}),
    ("cumsum", Sums, GATHER, {}),
    ("cumsum | on mps", Sums, GATHER, {"processors": on(0, {"gpu": "mps"}), "needs_mps": True}),
    ("where over a viewed arange | on mps", Masked, [(r(2, 6, 3),)],
     {"processors": on(0, {"gpu": "mps"}), "needs_mps": True}),
    ("where over a viewed arange | cpu of cpu/mps", Masked, [(r(2, 6, 3),)],
     {"processors": MIXED, "needs_mps": True, "placement": lambda: lambda node: "cpu"}),
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
     {"processors": TWO, "sharded": True, "collectives": {"reduce": 4}, "placement": lambda: tp.placement(SHARDS)}),
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
     {"processors": TWO, "sharded": True, "collectives": {"reduce": 4, "alltoall": 2},
      "placement": lambda: megatron_with({(ATEN.sigmoid.default, (2, 8, 48)): (2, SHARDS, (1, 3))})}),
    ("flip of a sharded value | 1:2", Flipped, [(r(6, 4),)],
     {"processors": TWO, "sharded": True,
      "placement": lambda: lambda node: (0, SHARDS, (1, 2)) if node.name == "x" else "p0"}),
    ("flip of a sharded value | 3 stripes", Flipped, [(r(7, 4),)],
     {"processors": THREE, "sharded": True,
      "placement": lambda: lambda node: (0, tuple(THREE), "striped") if node.name == "x" else "p0"}),
    ("rows | 1:2:1 to stripes and to 2:1:1", Twice, [(r(7, 4),)],
     {"processors": THREE, "sharded": True, "collectives": {"alltoall": 2},
      "placement": lambda: lambda node: {"x": (0, tuple(THREE), (1, 2, 1)), "cos": (0, tuple(THREE), "striped"),
                                         "exp": (0, tuple(THREE), (2, 1, 1))}.get(node.name, "p0")}),
    ("sums | 7 rows in 3 stripes", Reduced, [(r(7, 4),)],
     {"processors": THREE, "sharded": True, "collectives": {"reduce": 1},
      "placement": lambda: lambda node: (0, tuple(THREE), "striped") if node.name == "x" else "p0"}),
    ("view over a split dimension | copied", Merged, [(r(2, 6, 4),)],
     {"processors": TWO, "sharded": True, "copied": 2,
      "placement": lambda: lambda node: (1, SHARDS, None) if node.name == "x" else "p0"}),
    ("ragged rows | separators, speeds 1:3", Rows, [(LENGTHS, r(31, 4))],
     {"processors": TWO, "sharded": True, "placement": ragged_rows}),
    ("windows | rows 1:2:1, halos exchanged", Windows, [(r(2, 3, 23, 9),)],
     {"processors": THREE, "sharded": True, "windows": {"shards": 12, "masked": 6}, "received": 6,
      "placement": lambda: lambda node: (2, tuple(THREE), (1, 2, 1)) if node.name == "x" else "p0"}),
    ("windows | rows 1:6:1, halos padded", Windows, [(r(2, 3, 23, 9),)],
     {"processors": THREE, "sharded": True, "windows": {"shards": 12, "masked": 6, "padded": 2},
      "placement": lambda: lambda node: (2, tuple(THREE), (1, 6, 1)) if node.name == "x" else "p0"}),
    ("windows | pooling with a divisor, 3 shards", Divided, [(r(2, 3, 11, 6),)],
     {"processors": THREE, "sharded": True, "windows": {"shards": 3, "masked": 2}, "collectives": {"permute": 1},
      "placement": lambda: lambda node: (2, tuple(THREE), None) if node.name == "x" else "p0"}),
    ("windows | max pooling, ceil_mode, 3 shards", Ceiled, [(r(2, 3, 12, 6),)],
     {"processors": THREE, "sharded": True, "windows": {"shards": 3, "masked": 2},
      "placement": lambda: lambda node: (2, tuple(THREE), (1, 2, 1)) if node.name == "x" else "p0"}),
    ("windows | max pooling, ceil_mode, input on p0", Ceiled, [(r(2, 3, 12, 6),)],
     {"processors": THREE, "sharded": True, "windows": {"shards": 3, "masked": 2},
      "placement": lambda: lambda node: (2, tuple(THREE), None) if "max_pool" in node.name else "p0"}),
    ("windows | rows and columns, 2 by 2", Windows, [(r(2, 3, 23, 21),)],
     {"processors": FOUR, "sharded": True, "windows": {"shards": 16, "masked": 12}, "collectives": {"permute": 14},
      "placement": lambda: lambda node: ((2, 3), (("p0", "p1"), ("p2", "p3")), (None, (1, 2)))
      if node.name == "x" else "p0"}),
    ("windows | rows and columns, 1 by 2", Windows, [(r(2, 3, 23, 21),)],
     {"processors": TWO, "sharded": True, "windows": {"shards": 8, "masked": 6}, "collectives": {"permute": 7},
      "placement": lambda: lambda node: ((2, 3), (SHARDS,), (None, None)) if node.name == "x" else "p0"}),
    ("windows | 2 shards, input on p0", Windows, [(r(2, 3, 23, 9),)],
     {"processors": TWO, "sharded": True, "windows": {"shards": 8, "masked": 4, "views": 2},
      "placement": lambda: lambda node: (2, SHARDS, None) if node.name == "convolution" else "p0"}),
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
    ("megatron | 2 shards, backward", tp.Training, [(r(2, 8, 32) / 32 ** 0.5,)],
     {"processors": TWO, "sharded": True, "loss": 0, "placement": lambda: tp.placement(SHARDS)}),
    ("megatron | units 1:3 and 3:1, backward", tp.Training, [(r(2, 8, 32) / 32 ** 0.5,)],
     {"processors": TWO, "sharded": True, "loss": 0, "received": 8,
      "placement": lambda: tp.placement(SHARDS, {"model_0": (1, 3), "model_1": (3, 1)})}),
    ("causal lm striped | 2 shards, backward", lambda: cp.Causal(6), [LM],
     {"processors": TWO, "sharded": True, "loss": 0, "placement": lambda: cp.placement(SHARDS, sequence=6)}),
    ("causal lm striped | 3 shards, backward", lambda: cp.Causal(6), [LM],
     {"processors": THREE, "sharded": True, "loss": 0, "placement": lambda: cp.placement(tuple(THREE), sequence=6)}),
    ("crossings | 3, each sharded again", lambda: crossing.Chain(crossing.Crossing()).half(),
     [(r(2, 16).half(), r(2, 16).half())],
     {"processors": TWO, "sharded": True, "placement": lambda: crossing.placement(SHARDS)}),
    ("experts | 2 shards", lambda: ep.MoE(2), [(r(2, 6, 32) / 32 ** 0.5,)],
     {"processors": TWO, "sharded": True, "collectives": {"alltoall": 2},
      "placement": lambda: ep.placement(SHARDS, ep.MoE(2).slots)}),
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
METAL_SIGNED = ("mlp | all on mps", "conv | conv on mps, rest cpu", "megatron | 2 shards, cpu/mps")
KEY_CASES = ("mlp", "mlp | 2 processors pipeline", "mixture of experts | 3 processors", "megatron | 2 shards")
OPTION_FLAGS = ("rejected", "needs_mps", "expect_on", "sharded", "not_lowered", "inputs_on", "tolerance", "copied", "windows", "received", "collectives")


def errors(reference, got):
    worst_abs = worst_rel = 0.0
    for a, b in zip(pytree.tree_leaves(reference), pytree.tree_leaves(got), strict=True):
        a, b = torch.as_tensor(a).cpu().double(), torch.as_tensor(b).cpu().double()
        if a.shape != b.shape:
            return float("inf"), float("inf")
        apart = torch.where(a == b, 0.0, (a - b).abs())
        if apart.isnan().any() or apart.isinf().any():
            return float("inf"), float("inf")
        difference = apart.max().item() if a.numel() else 0.0
        finite = a[a.isfinite()]
        scale = finite.abs().max().item() if finite.numel() else 0.0
        worst_abs = max(worst_abs, difference)
        worst_rel = max(worst_rel, difference / max(scale, 1e-12))
    return worst_abs, worst_rel


def call_nodes(graph):
    return [n for n in graph.nodes if n.op == "call_function"]


ATEN = torch.ops.aten
CAT, SLICE, SLICE_SCATTER = ATEN.cat.default, ATEN.slice.Tensor, ATEN.slice_scatter.default
PLACED = {ir.SEND, ir.RECV, TO_COPY, CAT, SLICE, SLICE_SCATTER, ATEN.add.Tensor, ATEN.mm.default, ATEN.reshape.default,
          ATEN.new_zeros.default, ATEN.constant_pad_nd.default, ATEN.arange.start_step,
          ATEN.remainder.Scalar, ATEN.le.Scalar, ATEN.logical_and.default, ATEN.logical_not.default,
          ATEN.view.default, ATEN.expand.default, ATEN.scalar_tensor.default, ATEN.where.self, ATEN.clone.out,
          ATEN.empty.memory_format}
BLOCK = ("dimension", "block", "counts", "displs")


def copies(node):
    return node.target is TO_COPY and "device" in node.kwargs


def check_block(sent, arrived):
    assert [sent.meta.get(key) for key in BLOCK] == [arrived.meta.get(key) for key in BLOCK], "a channel's two ends"
    if "counts" not in arrived.meta:
        return
    dimension, block, counts, displs = (arrived.meta[key] for key in BLOCK)
    flat = all(type(n) is int for n in (*counts, *displs))
    assert flat or all(type(n) is int and len(array) == 2 for array in (counts, displs) for part in array
                       for n in part), "counts and displs are not integer arrays of one entry per device"
    if arrived.meta["collective"][0] == "permute":
        assert flat and len(counts) == len(displs) > max(block), "a permute's counts"
        return
    if isinstance(block, tuple):
        (given, taken), (holder, reader) = counts, block
        assert len(given) == len(displs[0]) > holder and len(taken) == len(displs[1]) > reader, "an all-to-all's counts"
        assert arrived.meta["val"].shape[dimension] <= min(given[holder], taken[reader]), "a block exceeds its counts"
        return
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
        if dimension in (value.args[0] if value.target is SLICE else value).meta.get("order", {}):
            return
        sliced = (len(counts) > 1 and value.target is SLICE and value.args[1] == dimension
                  and value.args[2] == displs[block])
        places.append(value.args[2] if sliced else 0)
    assert places == [displs[block]] * len(places), "a block is not placed at its displacement"


# design/problems-of-14b9a757-change-request.md items 3, 6 and 8
def check_views(scheduled, copied=0, received=0):
    found = into = 0
    for module in scheduled.modules.values():
        names = {n.name: n for n in module.graph.nodes}
        for node in call_nodes(module.graph):
            if node.target in (ATEN.view.default, ATEN.reshape.default):
                assert "view" in node.meta, f"{node.name} is no tensor view of its operand"
                held = node.args[0]
                found += (held.target is ATEN.clone.out and "of" not in held.meta and held.meta.get("out", [None])[0]
                          == node.meta["view"][0] == held.kwargs["out"].name
                          and held.kwargs.get("memory_format") is torch.contiguous_format)
            if "view" in node.meta:
                base, offset, sizes, strides = node.meta["view"]
                value, whole = node.meta["val"], names[base].meta["val"]
                assert "view" not in names[base].meta and sizes == tuple(value.shape), f"{node.name}: {node.meta['view']}"
                last = offset + sum((n - 1) * s for n, s in zip(sizes, strides))
                held = 1 + sum((n - 1) * s for n, s in zip(whole.shape, whole.stride()))
                assert 0 <= offset and (not all(sizes) or last < held), f"{node.name} leaves {base}"
                assert ("invariant" in node.meta) == ("state" in names[base].meta), f"{node.name}: a slice of state"
            for key in {"to", "out"} & node.meta.keys():
                buffer, offset, sizes, strides = node.meta[key]
                whole, last = names[buffer].meta["val"], offset + sum((n - 1) * s for n, s in zip(sizes, strides))
                assert (key == "to") == (node.target is ir.RECV), node.name
                assert names[buffer] in (node.kwargs.get("out"), *node.users) and len(node.users) == 1, node.name
                assert sizes == tuple(node.meta["val"].shape) and (not all(sizes) or last < whole.numel()), node.name
                into += key == "to"
        records = [[n.meta.get(key) for key in ("view", "to", "out", "invariant")] for n in module.graph.nodes]
        views([n for n in call_nodes(module.graph) if n.target is ir.RECV or copies(n)])
        assert records == [[n.meta.get(key) for key in ("view", "to", "out", "invariant")]
                           for n in module.graph.nodes], "views of the nodes on_devices rewires, and of the graph"
    assert found >= copied, f"{found} of {copied} views that fail torch's condition are of a contiguous copy"
    assert into >= received, f"{into} of {received} recvs name a view of the buffer made from them"


# design/problems-of-14b9a757-change-request.md item 4, steps 1 and 2; design/unequal-shards-change-request.md item 10
def check_windows(scheduled, expected):
    found = collections.Counter()
    for module in scheduled.modules.values():
        names = {n.name: n for n in module.graph.nodes}
        for node in call_nodes(module.graph):
            if "of" in node.meta and any(kind in str(node.target) for kind in ("convolution", "pool")):
                held = node.args[0]
                found["shards"] += 1
                found["masked"] += held.target is ATEN.where.self
                found["views"] += names[held.meta["view"][0]].op == "placeholder" if "view" in held.meta else (
                    held.op == "placeholder")
            found["padded"] += node.target is ir.SEND and node.args[0].target is ATEN.constant_pad_nd.default
    assert +found == expected, f"windowed shards {dict(+found)}, not {expected}"


# design/aot-compilation-change-request.md A4
def check_collectives(scheduled, expected=None):
    ends, kinds = {}, collections.defaultdict(set)
    for processor, module in scheduled.modules.items():
        for node in call_nodes(module.graph):
            if ir.channel_of(node) is not None:
                ends.setdefault(ir.channel_of(node), {})[node.target] = node
    sources, readers, pairs = collections.defaultdict(set), collections.defaultdict(set), collections.Counter()
    for end in ends.values():
        sent, arrived = end[ir.SEND], end[ir.RECV]
        assert sent.meta["collective"] == arrived.meta["collective"], "a channel's two ends name two collectives"
        kind, identity = sent.meta["collective"]
        assert kind not in ("alltoall", "permute") or "counts" in arrived.meta, f"the ends of {kind} carry no counts"
        value = sent.args[0].args[0] if copies(sent.args[0]) and sent.args[0].name != identity else sent.args[0]
        kinds[kind].add(identity)
        sources[kind, identity].add((sent.meta["processor"], value.name))
        readers[kind, identity].add(arrived.meta["processor"])
        pairs[kind, identity, sent.meta["processor"], arrived.meta["processor"]] += kind == "alltoall"
    assert all(len(found) == 1 for (kind, _), found in sources.items() if kind == "broadcast"), "a broadcast of two values"
    assert all(len(found) == 1 for (kind, _), found in readers.items() if kind == "reduce"), "a reduction with two roots"
    assert max(pairs.values(), default=0) <= 1, "an all-to-all with two demands for one ordered pair"
    found = {kind: len(identities) for kind, identities in kinds.items() if kind != "broadcast"}
    assert expected is None or found == expected, f"collectives {found}, not {expected}"


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
                replicated.add(node.meta["constant"])
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
        nodes = [n for m in scheduled.modules.values() for n in call_nodes(m.graph)]
        held = collections.defaultdict(set)
        for node in nodes:
            if (node.meta.get("of") in original and isinstance(node.meta["val"], torch.Tensor)
                    and not node.meta.get("order")):
                cut = [u for u in node.users if u.target is SLICE and tuple(u.args[2:]) == node.meta.get("cut")]
                held[node.meta["of"]].add(tuple((cut[0] if cut else node).meta["val"].shape))
        assert all(len(shapes) == 1 for shapes in held.values()), "shards of one node differ in shape"
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


def compiled_program(scheduled, stored=()):
    functions = lowered(scheduled, (lambda given: lower_metal.metal(given, stored), lower_inductor.lower))
    unwrapped = torch._dynamo.eval_frame.OptimizedModule
    assert all(isinstance(f, unwrapped) for p, f in functions.items()
               if scheduled.devices[p].type == "cpu"), "a wrapped function"
    assert not any(isinstance(f, torch.fx.GraphModule) for f in functions.values()), "a processor not lowered"
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
    def __init__(self, scheduled, stored=()):
        super().__init__(scheduled, lower_metal.lower(scheduled, stored))
        self.ordered = [f for p, f in self.functions.items() if hasattr(f, "feedback") and p in self.arguments]
        self.functions = {p: f if p == self.caller else watched(f) for p, f in self.functions.items()}

    def __call__(self, *args, **kwargs):
        with profile(activities=[ProfilerActivity.CPU]) as found, Watched():
            results = super().__call__(*args, **kwargs)
        PROFILED.update({event.key: event.count for event in found.key_averages()})
        on_mps = sorted(name for name, on in DISPATCHED if on)
        assert not on_mps, f"torch operators on an mps tensor around a Metal function: {on_mps}"
        for function in self.ordered:
            assert all(d.causes for d in function.dependences)
            assert all(0 < start <= end for _, start, end in function.feedback()), function.feedback()
        return results


DISPATCHED, PROFILED, PROFILES = collections.Counter(), collections.Counter(), {}


PLANNED = []


def planned_program(scheduled):
    functions = lowered(scheduled, (lower_metal.lower, lower_coreml.lower))
    PLANNED.extend((processor, functions[processor]) for processor in scheduled.engines)
    return Program(scheduled, functions)


def variants(scheduled, with_inductor, stored=()):
    passed = lambda: {p: fx_pass(copy.deepcopy(m)) for p, m in scheduled.modules.items()}
    return {
        "fx direct": lambda: Program(scheduled),
        "inductor": (lambda: compiled_program(scheduled, stored)) if with_inductor else None,
        "archive": lambda: archived_program(scheduled),
        "fx pass": lambda: Program(dataclasses.replace(scheduled, modules=passed())),
        "metal": (lambda: Profiled(scheduled, stored)) if MPS else None,
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
    check_views(program, options.get("copied", 0), options.get("received", 0))
    if "windows" in options:
        check_windows(program, options["windows"])
    check_collectives(program, options.get("collectives"))
    if name in SAVE:
        save_for_review(SAVE[name], program)
    results, dynamo, breaks = {}, "", 0
    stored = program.exported.user_inputs if options.get("inputs_on") else ()
    for path, make in variants(program, with_inductor, stored).items():
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


class Joined(nn.Module):
    def forward(self, p, q):
        return p + q, p * q


class Block(nn.Module):
    def __init__(self, between=False):
        super().__init__()
        self.a, self.b, self.join = nn.Linear(8, 8), nn.Sequential(nn.Linear(8, 8), nn.Tanh()), Joined()
        self.between = between

    def forward(self, x, y):
        left, right = self.a(x), self.a(y)
        return self.join(self.b(left), right * 2 if self.between else right)


class Normed(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm, self.a, self.join = nn.BatchNorm1d(8), nn.Linear(8, 8), Joined()

    def forward(self, x, y):
        return self.join(self.a(self.norm(x)), self.norm(y))


def doubled(x):
    return x * 2


class Indexed(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.a, self.index = nn.Linear(8, 8), index

    def forward(self, x):
        return self.a(doubled(x)[0] if self.index else doubled(x))


class Called(torch.fx.Interpreter):
    def __init__(self, module, exported):
        super().__init__(module)
        self.exported, self.programs = exported, {}

    def run_node(self, node):
        self.calling = node
        return super().run_node(node)

    def call_module(self, target, args, kwargs):
        key, written, function = self.calling.meta["function"], self.calling.meta.get("written", ()), self.fetch_attr(
            target)
        if written:
            function, args = function.function, args[:len(args) - len(written)]
        if key not in self.programs:
            assert key in self.exported and self.fetch_attr(key[0]) is function, (
                f"{self.calling.name} names no exported function")
            self.programs[key] = Program(compile_module(function, args, TWO, round_robin("p0", "p1")))
        program = self.programs[key]
        result = program(*args)
        return (result, *(program.arguments[processor][place] for name in written for processor, place in
                          program.positions[program.key_of[name]][:1])) if written else result


# design/aot-compilation-change-request.md A1 and A10
def check_calls():
    torch.manual_seed(0)
    program, args = Block().eval(), (r(2, 8), r(2, 8))
    graph, exported = calls(program, args, ("a", "b", "join"))
    nodes = [n for n in graph.graph.nodes if n.op == "call_module"]
    assert [n.target for n in nodes] == ["a", "a", "b", "join"], "a call of a function is not one node"
    assert {n.op for n in graph.graph.nodes} == {"placeholder", "call_module", "output"}, "a function was traced into"
    assert sorted(key[0] for key in exported) == ["a", "b", "join"], "a function called twice is exported twice"
    assert all(isinstance(made, ir.Exported) and call_nodes(made.graph_module.graph) for made in exported.values())
    most = mesh_graph.width(nodes, lambda n: [u for u in n.users if u.op == "call_module"])
    assert most == 2, f"{most} calls are independent at once, not the two of `a`"
    called = Called(graph, exported)
    worst = errors(program(*args), called.run(*args))[1]
    assert worst <= TOLERANCE and len(called.programs) == 3, (worst, len(called.programs))
    plain, made = calls(Indexed(False).eval(), args[:1], ("a", doubled))
    assert [(n.op, n.target, n.meta["function"][0]) for n in plain.graph.nodes if "function" in n.meta] == [
        ("call_function", doubled, doubled), ("call_module", "a", "a")] and len(made) == 2, "a function is no leaf"
    writer = Normed().train()
    graph, exported = calls(writer, args, ("norm", "a", "join"))
    reference = copy.deepcopy(writer)
    first, second = [n for n in graph.graph.nodes if "written" in n.meta]
    held = [n for n in graph.graph.nodes if n.op == "get_attr"]
    assert first.meta["written"] == tuple(n.target.removeprefix("norm.") for n in held), "the buffers a call writes"
    assert first.args[1:] == tuple(held) and [(n.op, *n.args) for n in second.args[1:]] == [
        ("call_function", first, k + 1) for k in range(len(held))], "a written buffer is no argument and no result"
    called = Called(graph, exported)
    wrote = errors(reference(*args), called.run(*args))[1]
    norm = called.programs[second.meta["function"]]
    last = max(errors(reference.norm.get_buffer(name).float(), norm.arguments[processor][place].float())[1]
               for name in second.meta["written"] for processor, place in norm.positions[norm.key_of[name]][:1])
    assert max(wrote, last) <= TOLERANCE and len(called.programs) == 3, (wrote, last)
    print(f"  a function that writes {len(held)} buffers: each is an argument and a result of its call's node, the "
          f"second call reading the first's; rel err {wrote:.1e}, of the buffers {last:.1e}")
    for block, given in ((Block(True), args), (Indexed(True), args[:1])):
        try:
            calls(block.eval(), given, ("a", "b", "join", doubled))
        except NotImplementedError as error:
            print(f"  refused: {error}")
        else:
            raise AssertionError("an operation between two compiled calls was captured as no function")
    print(f"  4 calls of 3 exported functions, {most} independent at once; each function compiled once and called in the "
          f"graph's order, rel err {worst:.1e}")
    print("  ok: the block's calls are the nodes of one graph, and nothing of the block runs outside a function")


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


# design/synchronization-signature-change-request.md items 1, 2 and 4
def check_signatures():
    for name in dict.fromkeys(KEY_CASES + FRESH_CASES):
        if "mps" in name and not MPS:
            continue
        scheduled = build(name)[2]
        made = signature.signature(scheduled, executions=2)
        states, counts = made["states"], collections.Counter()
        for (_, primitive), count in made["counts"].items():
            counts[primitive] += count
        assert made["digest"] == signature.signature(scheduled, executions=2)["digest"], "the digest is not a function"
        assert states["image"] == 2 * (scheduled.channels + len(scheduled.modules) - 1), \
            f"{name}: a wait of a channel or of a processor's arguments is not a dependence's image"
        assert not (states["false"] or states["stronger"] or made["redundant"] or made["uncovered"]
                    or made["unwaited"]), (name, states, made["redundant"], made["uncovered"], made["unwaited"])
        assert states["unmatched"] == counts["host_wait"] - states["image"], name
        print(f"  {name:34} {made['digest'][:12]}  2 executions: {dict(counts)}, {states['image']} waits each a "
              f"dependence's image, {states['unmatched']} waits of torch's copies between devices")
    print(made["space_time"])
    print("  ok: every wait of a channel or of a processor's arguments is the image of its dependence, none")
    print("      redundant, no dependence uncovered")
    for name in METAL_SIGNED * MPS:
        scheduled = build(name)[2]
        functions = lowered(scheduled, (lower_metal.lower, lower_coreml.lower))
        made = signature.signature(scheduled, functions)
        assert made["digest"] == signature.signature(scheduled, functions)["digest"], "the digest is not a function"
        assert not made["unwaited"] and not made["states"]["unmatched"], (name, made["unwaited"], made["states"])
        counts = collections.Counter()
        for (_, primitive), count in made["counts"].items():
            counts[primitive] += count
        print(f"  {name:34} {made['digest'][:12]}  lowered to Metal: {dict(counts)}; edges {made['states']['image']} images, "
              f"{made['states']['false']} with no dependence under them, {made['redundant']} redundant; "
              f"{made['uncovered']} dependences with no signal-to-wait path")
        print(made["space_time"] * (name == METAL_SIGNED[0]), end="")


# design/synchronization-signature-change-request.md items 3, 4 and 5
def playback(directory, name, executions, added=False):
    _, _, calls, _ = next(case for case in CASES if case[0] == name)
    scheduled = archive.load(directory)
    functions = lowered(scheduled, (lower_metal.lower, lower_coreml.lower))
    print(f"ready {os.getpid()}", flush=True)
    sys.stdin.readline()
    program = Program(scheduled, functions)
    for _ in range(int(executions)):
        program(*copy.deepcopy(calls[0]))
    if added:
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
        metal.MTLCreateSystemDefaultDevice.restype = objc.sel_registerName.restype = ctypes.c_void_p
        made = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
        wait = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64)(
            ("objc_msgSend", objc))
        event = made(metal.MTLCreateSystemDefaultDevice(), objc.sel_registerName(b"newSharedEvent"))
        wait(event, objc.sel_registerName(b"waitUntilSignaledValue:timeoutMS:"), 0, 0)
    print("played", flush=True)


def check_monitored_playback():
    if not MPS or subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode:
        print("  SKIP (no mps, or sudo -n is refused: the monitor runs dtrace under sudo)")
        return
    executions = 3
    denied = subprocess.run(["sudo", "-n", "dtrace", "-w", "-q", "-n", "BEGIN { exit(0); }"], capture_output=True,
                            text=True).stderr.strip().splitlines()[-1:]
    observe = ["--observe"] * any("could not enable tracing" in line for line in denied)
    if observe:
        print(f"  dtrace -w (the destructive action that stops the process) is refused here: {denied[0]}; the "
              "monitor observes without it and fails the run at the operation")
    for name, added in ((METAL_SIGNED[0], False), (METAL_SIGNED[0], True), (METAL_SIGNED[1], False)):
        scheduled, _ = compiled(build(name)[2], (lower_metal.lower, lower_coreml.lower), executions)
        with tempfile.TemporaryDirectory() as directory:
            archive.save(directory, scheduled)
            player = subprocess.Popen([sys.executable, str(HERE / "samples.py"), "--playback", directory, name,
                                       str(executions)] + ["added"] * added, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True)
            ready = player.stdout.readline().split()
            monitor = subprocess.Popen([sys.executable, str(HERE.parent / "bin" / "mesh-sync-monitor.py"), "run",
                                        str(pathlib.Path(directory) / "signature.record"), "-p", ready[1]] + observe,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            seen = [monitor.stdout.readline()]
            player.stdin.write("\n")
            player.stdin.flush()
            out, report = monitor.communicate()
            played = "played" in player.stdout.readline()
            player.kill()
            stopped = [line for line in (seen[0] + out).splitlines() if "unexpected" in line]
            verdict = report.strip().splitlines()[-1]
            assert "attached" in seen[0], (seen, report)
            assert (monitor.returncode, bool(stopped)) == (int(added), added), (monitor.returncode, stopped, report)
            assert played == (not added or bool(observe)), "the process ran past the operation that stops it"
            print(f"  {name}, {executions} executions{', then a wait its signature does not have' * added}: "
                  f"{stopped[0] if stopped else verdict}")
    print("  ok: the playback that ran to completion ran the compiler's signature; the added wait failed its run")


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
    saved = mesh_graph.from_record(scheduled.signature)
    signed = [signature.signature(scheduled)["digest"], bytes(saved.digest).hex()]
    mesh_graph.free(saved)
    print(json.dumps({"abs": worst[0], "rel": worst[1], "digests": digests(scheduled.modules), "signature": signed}))


def check_reload_in_fresh_process():
    for name in FRESH_CASES:
        if "mps" in name and not MPS:
            continue
        program, _ = compiled(build(name)[2], ())
        signed = signature.signature(program)
        with tempfile.TemporaryDirectory() as directory:
            archive.save(directory, program)
            size = sum(f.stat().st_size for f in pathlib.Path(directory).iterdir())
            command = [sys.executable, str(HERE / "samples.py"), "--reload", directory, name]
            done = subprocess.run(command, capture_output=True, text=True, check=True)
        report = json.loads(done.stdout.strip().splitlines()[-1])
        assert report["rel"] <= TOLERANCE or report["abs"] <= TOLERANCE, report
        assert report["digests"] == digests(program.modules), "digest changed by reload"
        assert report["signature"] == [signed["digest"]] * 2, "the signature's digest changed by reload"
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
    header.algo_off, header.view_off, header.page_off, header.buffer_off, header.data_off = (
        header.pgsz, 2 * header.pgsz, 3 * header.pgsz, 7 * header.pgsz, 11 * header.pgsz)
    header.length = header.data_off + pages * header.pgsz
    shared = SharedMemory(create=True, size=header.length, name=f"mesh_compiler_{os.getpid()}_{node}")
    shared.buf[:ctypes.sizeof(header)] = bytes(header)
    for place, other in enumerate(nodes):
        record, peers = header.view_off + place * RECORD_BYTES, [peer for peer in nodes if peer != other]
        shared.buf[record:record + 8] = struct.pack("II", other, len(peers))
        for cable, peer in enumerate(peers):
            shared.buf[record + 72 + 24 * cable:][:24] = struct.pack("IIIIQ", peer, 0, 0, 1, 10 ** 10)
    context = mesh.Context()
    mesh.check(mesh.LIB.mesh_attach(ctypes.byref(context), f"/{shared.name}".encode()))
    return shared, context


def carry(contexts):
    mesh, wake = bridge.mesh, ctypes.CDLL("/usr/lib/libSystem.B.dylib").os_sync_wake_by_address_all
    wake.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    headers = [mesh.Header.from_address(context.M) for context in contexts]
    algos = [mesh.Algo.from_address(context.M + header.algo_off) for context, header in zip(contexts, headers)]
    for algo in algos:
        if algo.retired == algo.published != algo.released:
            algo.released = algo.published
            wake(ctypes.addressof(algo) + mesh.Algo.released.offset, 8, 1)
    if not all(algo.published != algo.retired for algo in algos) or algos[0].group != algos[1].group:
        return
    word = lambda k, at: ctypes.c_uint64.from_address(contexts[k].M + at)
    for place in range(algos[0].plans):
        states = [mesh.State.from_address(contexts[k].M + word(k, algo.table + 8 * place).value)
                  for k, algo in enumerate(algos)]
        lists = [(mesh.Hop * state.entries).from_address(context.M + state.hops)
                 for context, state in zip(contexts, states)]
        copies, started = states[0].copies, min(state.executions for state in states)
        assert copies == states[1].copies, "the two ends' copies differ"
        for source, copy in itertools.product((0, 1), range(copies)):
            ends = [[hop for hop in hops[copy * len(hops) // copies:(copy + 1) * len(hops) // copies]
                     if hop.direction == direction] for hops, direction in (
                         (lists[source], mesh.SEND), (lists[1 - source], mesh.RECEIVE))]
            ranges = {}
            for sent, landed in zip(*ends, strict=True):
                assert (sent.bytes, sent.node, landed.node) == (landed.bytes, 1 - source, source), "the two ends' entries differ"
                ranges.setdefault(sent.word, []).append((sent, landed))
            for supplied, pairs in ranges.items():
                completion, count = word(source, pairs[0][0].count), word(1 - source, pairs[0][1].count)
                turn = completion.value // len(pairs)
                if min(word(source, supplied).value, started) <= turn * copies + copy:
                    continue
                for sent, landed in pairs:
                    assert landed.pages == len(pairs) and landed.count == pairs[0][1].count, "a range's entries differ"
                    ctypes.memmove(contexts[1 - source].M + headers[1 - source].data_off + landed.page * headers[0].pgsz
                                   + landed.offset, contexts[source].M + headers[source].data_off
                                   + sent.page * headers[0].pgsz + sent.offset, sent.bytes)
                    word(1 - source, landed.word).value ^= 1 << (landed.field + (turn & 1))
                count.value += len(pairs)
                wake(ctypes.addressof(count), 8, 1)
                assert pairs[0][0].pages == len(pairs), "a sent range's entries differ from its count"
                completion.value += len(pairs)
                wake(ctypes.addressof(completion), 8, 1)


def check_between_machines(with_inductor):
    nodes = [3, 5]
    shared, contexts = zip(*(region(node, nodes) for node in nodes))
    command = [sys.executable, str(HERE / "samples.py"), "--machines", "3", "5"] + ["--fast"] * (not with_inductor)
    runs = [subprocess.Popen(command, stdout=subprocess.PIPE, text=True,
                             env={**os.environ, "MESH_REGION": f"/{region_.name}", "MESH_GROUP": "1",
                                  "MESH_PROGRAM": "0,1"})
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
    if "--playback" in sys.argv:
        playback(*sys.argv[sys.argv.index("--playback") + 1:])
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
    print("synchronization signatures of the scheduled programs (the last one's space-time diagram):")
    check_signatures()
    print("a saved program played back under bin/mesh-sync-monitor.py (DTrace, attached to the playing process):")
    check_monitored_playback()
    print("digests of the scheduled GraphModules across processes:")
    check_digests_across_processes()
    if with_inductor:
        print("Inductor's generated code, scheduled against exported, single-processor cases:")
        check_inductor_structure()
    print("attention's products where the sequence is sharded (Striped Attention, section 3.1):")
    check_products()
    print("a block that calls several functions (the calls' graph and each function's exported graph):")
    check_calls()
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
