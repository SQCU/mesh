# Switch-side algebraic page morphisms (Pages^k -> Pages^m) mirroring rdma/mesh-call.h & rdma/mesh-reduce.h.
# Ticket T1: explicit algebraic reduction IR and effectful custom ops for AOT placement & schedule lowering.
from dataclasses import dataclass
import math
import threading

import torch
from torch import Tensor
from torch._higher_order_ops.effects import _EffectType, _register_effectful_op

# Algebraic classes (bits 7..6 of wire op byte, identical to rdma/mesh-call.h)
ALG_NONE = 0
ALG_C0_COMM = 1      # Commutative monoid: any order, any Steiner merge tree; tag = leaf weight
ALG_C1_LIFTED = 2    # Lifted commutative monoid (e.g. online softmax (m,l,o)); tag = leaf weight
ALG_C2_INTERVAL = 3  # Associative non-commutative monoid; tag = lo | (hi << 8) contiguous interval


def red_op(alg: int, op_id: int) -> int:
    return ((alg & 0x3) << 6) | (op_id & 0x3F)


def red_alg(op: int) -> int:
    return (op >> 6) & 0x3


def tag_weight(w: int) -> int:
    return int(w) & 0xFFFF


def tag_interval(lo: int, hi: int) -> int:
    return (int(lo) & 0xFF) | ((int(hi) & 0xFF) << 8)


RED_NONE = 0
RED_SUM_F32 = red_op(ALG_C0_COMM, 1)
RED_MAX_F32 = red_op(ALG_C0_COMM, 2)
RED_MIN_F32 = red_op(ALG_C0_COMM, 3)
RED_SUM_I32 = red_op(ALG_C0_COMM, 4)
RED_SUM_I64 = red_op(ALG_C0_COMM, 5)
RED_SUM_F16 = red_op(ALG_C0_COMM, 6)
RED_MAX_F16 = red_op(ALG_C0_COMM, 7)
RED_MIN_F16 = red_op(ALG_C0_COMM, 8)
RED_SUM_BF16 = red_op(ALG_C0_COMM, 9)
RED_MAX_BF16 = red_op(ALG_C0_COMM, 10)
RED_MIN_BF16 = red_op(ALG_C0_COMM, 11)
RED_ONLINE_SOFTMAX_F32 = red_op(ALG_C1_LIFTED, 1)
RED_AFFINE_SCAN_F32 = red_op(ALG_C2_INTERVAL, 1)


@dataclass(frozen=True)
class AlgebraicOp:
    name: str
    op: int
    alg_class: int
    dtype: torch.dtype
    identity: tuple        # R14 padding identity values per structural lane
    wire_amp_ideal: float  # ideal wire/logical amplification under in-switch coalescing (1.0 for C0/C1/C2)


REGISTRY = {
    RED_SUM_F32: AlgebraicOp("sum_f32", RED_SUM_F32, ALG_C0_COMM, torch.float32, (0.0,), 1.0),
    RED_MAX_F32: AlgebraicOp("max_f32", RED_MAX_F32, ALG_C0_COMM, torch.float32, (-math.inf,), 1.0),
    RED_MIN_F32: AlgebraicOp("min_f32", RED_MIN_F32, ALG_C0_COMM, torch.float32, (math.inf,), 1.0),
    RED_SUM_I32: AlgebraicOp("sum_i32", RED_SUM_I32, ALG_C0_COMM, torch.int32, (0,), 1.0),
    RED_SUM_I64: AlgebraicOp("sum_i64", RED_SUM_I64, ALG_C0_COMM, torch.int64, (0,), 1.0),
    RED_ONLINE_SOFTMAX_F32: AlgebraicOp(
        "online_softmax_f32", RED_ONLINE_SOFTMAX_F32, ALG_C1_LIFTED, torch.float32, (-math.inf, 0.0, 0.0), 1.0
    ),
    RED_AFFINE_SCAN_F32: AlgebraicOp(
        "affine_scan_f32", RED_AFFINE_SCAN_F32, ALG_C2_INTERVAL, torch.float32, (1.0, 0.0), 1.0
    ),
}


def op_for_aten(target, dtype: torch.dtype) -> int | None:
    aten = torch.ops.aten
    if target in (aten.add.Tensor, aten.sum.dim_IntList, aten.mm.default, aten.bmm.default, aten.addmm.default):
        if dtype == torch.float32:
            return RED_SUM_F32
        if dtype == torch.int32:
            return RED_SUM_I32
        if dtype == torch.int64:
            return RED_SUM_I64
    if target in (aten.maximum.default, aten.amax.default) and dtype == torch.float32:
        return RED_MAX_F32
    if target in (aten.minimum.default, aten.amin.default) and dtype == torch.float32:
        return RED_MIN_F32
    return None


def leaf_tag(op: int, rank_index: int) -> int:
    return tag_interval(rank_index, rank_index + 1) if red_alg(op) == ALG_C2_INTERVAL else tag_weight(1)


def complete_tag(op: int, count: int) -> int:
    return tag_interval(0, count) if red_alg(op) == ALG_C2_INTERVAL else tag_weight(count)


def apply_reference(op: int, parts: list[Tensor]) -> Tensor:
    """Reference semantics of algebraic page morphism over ordered list of shard tensors."""
    if len(parts) == 1:
        return parts[0]
    if op in (RED_SUM_F32, RED_SUM_I32, RED_SUM_I64):
        out = parts[0].clone()
        for p in parts[1:]:
            out = out + p
        return out
    if op == RED_MAX_F32:
        out = parts[0].clone()
        for p in parts[1:]:
            out = torch.maximum(out, p)
        return out
    if op == RED_MIN_F32:
        out = parts[0].clone()
        for p in parts[1:]:
            out = torch.minimum(out, p)
        return out
    if op == RED_ONLINE_SOFTMAX_F32:
        # Last dim is 128 floats: [m, l, o_0..o_125]
        acc = parts[0].reshape(-1, 128).clone()
        for p in parts[1:]:
            b = p.reshape(-1, 128)
            m = torch.maximum(acc[:, 0:1], b[:, 0:1])
            ea = torch.exp(acc[:, 0:1] - m)
            eb = torch.exp(b[:, 0:1] - m)
            acc[:, 0:1] = m
            acc[:, 1:] = acc[:, 1:] * ea + b[:, 1:] * eb
        return acc.reshape_as(parts[0])
    if op == RED_AFFINE_SCAN_F32:
        # Pairs (a, b) composing f_{N-1} o ... o f_0
        acc = parts[0].reshape(-1, 2).clone()
        for p in parts[1:]:
            r = p.reshape(-1, 2)
            a_new = r[:, 0] * acc[:, 0]
            b_new = r[:, 0] * acc[:, 1] + r[:, 1]
            acc[:, 0] = a_new
            acc[:, 1] = b_new
        return acc.reshape_as(parts[0])
    raise ValueError(f"unsupported algebraic op {op:#x}")


LOCAL_REDUCE = threading.local()


@torch.library.custom_op("mesh_algebra::reduce_send", mutates_args=())
def reduce_send(value: Tensor, group: int, op: int, tag: int) -> None:
    LOCAL_REDUCE.groups[group].put((tag, value))


@reduce_send.register_fake
def reduce_send_fake(value: Tensor, group: int, op: int, tag: int) -> None:
    return None


@torch.library.custom_op("mesh_algebra::reduce_recv", mutates_args=())
def reduce_recv(group: int, op: int, done_tag: int) -> Tensor:
    q = LOCAL_REDUCE.groups[group]
    if red_alg(op) == ALG_C2_INTERVAL:
        want = (done_tag >> 8) - (done_tag & 0xFF)
        items = [q.get() for _ in range(want)]
        items.sort(key=lambda pair: pair[0] & 0xFF)
        return apply_reference(op, [t for _, t in items])
    got = []
    weight = 0
    while weight < done_tag:
        t, v = q.get()
        got.append(v)
        weight += t
    return apply_reference(op, got)


_register_effectful_op(reduce_send, _EffectType.ORDERED)
_register_effectful_op(reduce_recv, _EffectType.ORDERED)
REDUCE_SEND = torch.ops.mesh_algebra.reduce_send.default
REDUCE_RECV = torch.ops.mesh_algebra.reduce_recv.default


def is_reduce_send(node) -> bool:
    return getattr(node, "target", None) is REDUCE_SEND


def is_reduce_recv(node) -> bool:
    return getattr(node, "target", None) is REDUCE_RECV


def reduce_group_of(node):
    if is_reduce_send(node):
        return node.args[1]
    return node.args[0] if is_reduce_recv(node) else None
