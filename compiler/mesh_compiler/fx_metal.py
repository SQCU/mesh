# Complete torch.export FX Graph -> Single Pre-Encoded Native Metal Command Buffer Compiler
# Lowering through libmesh-graph.dylib over zero-copy /mesh0 flat-DMA + Steiner multicast + GF(2)^6 had_int8.
# Honors both functional compilation (explicit GraphIR, liveness arena coloring, static crossing plan)
# and access by view (zero-copy MG_BUF_VIEW reshapes, zero-copy torch.frombuffer state views, zero-copy /mesh0 W[]/P[]).
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass, field
import math
import os
import pathlib
from typing import Any

import torch
import torch.fx as fx
from torch.export.graph_signature import InputKind

# ---- Constants matching rdma/mesh-graph.h ----
MG_BUF_WEIGHT = 0
MG_BUF_ARENA = 1
MG_BUF_STATE = 2
MG_BUF_VIEW = 3

MG_DTYPE_BF16 = 0
MG_DTYPE_F16 = 1
MG_DTYPE_F32 = 2
MG_DTYPE_I32 = 3
MG_DTYPE_U32 = 4

MG_WIRE_BF16 = 0
MG_WIRE_F32 = 1
MG_WIRE_GATHER = 2
MG_WIRE_HAD8 = 3
MG_WIRE_HAD8_GATHER = 4
MG_WIRE_AUTO = 5

MG_OP_EMBEDDING = 1
MG_OP_LINEAR = 2
MG_OP_LINEAR_PUBLISH = 3
MG_OP_CROSSING_SUM_NORM = 4
MG_OP_RMS_NORM = 5
MG_OP_HEAD_NORM_ROPE = 6
MG_OP_KV_SDPA_DECODE = 7
MG_OP_GATE_UP_ACT = 8
MG_OP_PLE_GATE = 9
MG_OP_ELEMENTWISE = 10
MG_OP_VOCAB_ARGMAX = 11
MG_OP_ALLREDUCE = 12
MG_OP_CONCAT = 13
MG_OP_SPEC_VERIFY_COMMIT = 14
MG_OP_WRITE_TOK_SLOT = 15

MG_EW_ADD = 0
MG_EW_MUL = 1
MG_EW_SCALE = 2
MG_EW_ADD_SCALE = 3
MG_EW_GELU = 4
MG_EW_SILU = 5
MG_EW_RELU = 6
MG_EW_COPY = 7
MG_EW_MUL_SLICE = 8
MG_EW_RES_SCALE = 9

WIRE_MODE_MAP = {
    "bf16": MG_WIRE_BF16,
    "f32": MG_WIRE_F32,
    "gather": MG_WIRE_GATHER,
    "had8": MG_WIRE_HAD8,
    "had8g": MG_WIRE_HAD8_GATHER,
    "auto": MG_WIRE_AUTO,
}


class MeshGraphOpStruct(C.Structure):
    _fields_ = [
        ("kind", C.c_uint32),
        ("dtype", C.c_uint32),
        ("in_buf", C.c_int32 * 6),
        ("out_buf", C.c_int32 * 3),
        ("u", C.c_uint32 * 8),
        ("f", C.c_float * 4),
    ]


# ---- Custom PyTorch Ops (exportable via torch.export.export AND executable eagerly in PyTorch!) ----
@torch.library.custom_op("mesh::all_reduce_sum", mutates_args=())
def _mesh_all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_mesh_all_reduce_sum.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("mesh::rms_norm", mutates_args=())
def _mesh_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * inv).to(x.dtype) * weight


@_mesh_rms_norm.register_fake
def _(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("mesh::head_norm_rope_kv", mutates_args=("k_cache", "v_cache"))
def _mesh_head_norm_rope_kv(
    qkv: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    pos: torch.Tensor,
    freqs: torch.Tensor,
    h_q: int,
    h_kv: int,
    head_dim: int,
    rope_mode: int,
    rope_base: float,
    max_seq: int,
    window: int,
    eps: float,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    q_len: int = 1,
) -> torch.Tensor:
    B_tot = qkv.shape[0]
    ql = max(1, int(q_len))
    q_raw = qkv[:, : h_q * head_dim].reshape(B_tot, h_q, head_dim)
    qf = q_raw.float()
    q_norm = (qf * torch.rsqrt(qf.pow(2).mean(-1, keepdim=True) + eps)).to(qkv.dtype) * w_q
    p0 = int(pos.item())
    half = head_dim // 2
    if rope_mode == 1:
        inv_f = torch.pow(rope_base, -torch.arange(half, device=qkv.device, dtype=torch.float32) / half)
    else:
        inv_f = 1.0 / freqs.float()

    def apply_rope_pos(t, p_val):
        theta = float(p_val) * inv_f
        c, s = torch.cos(theta), torch.sin(theta)
        tf = t.float()
        a, b = tf[..., :half], tf[..., half:]
        return torch.cat([a * c - b * s, a * s + b * c], dim=-1).to(t.dtype)

    if ql == 1:
        q_out = apply_rope_pos(q_norm, p0)
    else:
        q_list = [apply_rope_pos(q_norm[bi : bi + 1], p0 + (bi % ql)) for bi in range(B_tot)]
        q_out = torch.cat(q_list, dim=0)
    if h_kv > 0:
        k_raw = qkv[:, h_q * head_dim : (h_q + h_kv) * head_dim].reshape(B_tot, h_kv, head_dim)
        v_raw = qkv[:, (h_q + h_kv) * head_dim :].reshape(B_tot, h_kv, head_dim)
        kf = k_raw.float()
        k_norm = (kf * torch.rsqrt(kf.pow(2).mean(-1, keepdim=True) + eps)).to(qkv.dtype) * w_k
        vf = v_raw.float()
        v_norm = (vf * torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + eps)).to(qkv.dtype)
        for bi in range(B_tot):
            stream = bi // ql
            p_val = p0 + (bi % ql)
            slot = (p_val % max_seq) if window > 0 else min(p_val, max_seq - 1)
            k_cache[stream, :, slot, :] = apply_rope_pos(k_norm[bi], p_val)
            v_cache[stream, :, slot, :] = v_norm[bi]
    return q_out


@_mesh_head_norm_rope_kv.register_fake
def _(qkv, w_q, w_k, pos, freqs, h_q, h_kv, head_dim, rope_mode, rope_base, max_seq, window, eps, k_cache, v_cache, q_len=1):
    return torch.empty((qkv.shape[0], h_q, head_dim), dtype=qkv.dtype, device=qkv.device)


@torch.library.custom_op("mesh::kv_sdpa_decode", mutates_args=())
def _mesh_kv_sdpa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos: torch.Tensor,
    h_q: int,
    h_kv: int,
    head_dim: int,
    max_seq: int,
    window: int,
    scale: float,
    q_len: int = 1,
    shared_kv_b0: bool = False,
) -> torch.Tensor:
    B_tot = q.shape[0]
    ql = max(1, int(q_len))
    p0 = int(pos.item())
    outs = []
    for bi in range(B_tot):
        stream = 0 if shared_kv_b0 else (bi // ql)
        q_idx = 0 if shared_kv_b0 else (bi % ql)
        length = p0 + q_idx + 1
        start = max(0, length - window) if window > 0 else 0
        idx = [((t % max_seq) if window > 0 else t) for t in range(start, length)]
        k = k_cache[stream : stream + 1, :, idx, :].float()
        v = v_cache[stream : stream + 1, :, idx, :].float()
        if h_q != h_kv:
            rep = h_q // h_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        scores = torch.einsum("bhd,bhtd->bht", q[bi : bi + 1].float(), k) * scale
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("bht,bhtd->bhd", probs, v).to(q.dtype))
    return torch.cat(outs, dim=0).reshape(B_tot, h_q * head_dim)


@_mesh_kv_sdpa_decode.register_fake
def _(q, k_cache, v_cache, pos, h_q, h_kv, head_dim, max_seq, window, scale, q_len=1, shared_kv_b0=False):
    return torch.empty((q.shape[0], h_q * head_dim), dtype=q.dtype, device=q.device)


@torch.library.custom_op("mesh::spec_verify_commit", mutates_args=("draft_tok", "pos_buf", "anchor_pos_buf", "stats_buf"))
def _mesh_spec_verify_commit(
    targ_tok: torch.Tensor,
    draft_tok: torch.Tensor,
    targ_xn: torch.Tensor,
    pos_buf: torch.Tensor,
    anchor_pos_buf: torch.Tensor,
    stats_buf: torch.Tensor,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_acc = 0
    while n_acc < K and int(targ_tok[n_acc].item()) == int(draft_tok[n_acc + 1].item()):
        n_acc += 1
    t_star = int(targ_tok[n_acc].item())
    p0 = int(pos_buf[0].item())
    stats_buf[0, 0] = n_acc
    stats_buf[0, 1] = t_star
    for k in range(K):
        stats_buf[0, 2 + k] = int(draft_tok[k + 1].item())
    anchor_pos_buf[0] = p0 + n_acc
    pos_buf[0] = p0 + n_acc + 1
    draft_tok[0] = t_star
    out_tok = torch.tensor([t_star], dtype=torch.int32, device=targ_tok.device)
    out_hid = targ_xn[n_acc : n_acc + 1].clone()
    return out_tok, out_hid


@_mesh_spec_verify_commit.register_fake
def _(targ_tok, draft_tok, targ_xn, pos_buf, anchor_pos_buf, stats_buf, K):
    return (
        torch.empty((1,), dtype=torch.int32, device=targ_tok.device),
        torch.empty((1, targ_xn.shape[-1]), dtype=targ_xn.dtype, device=targ_xn.device),
    )


@torch.library.custom_op("mesh::write_tok_slot", mutates_args=("dst_buf",))
def _mesh_write_tok_slot(src_tok: torch.Tensor, dst_buf: torch.Tensor, slot_idx: int) -> torch.Tensor:
    dst_buf[slot_idx] = src_tok[0]
    return src_tok.clone()


@_mesh_write_tok_slot.register_fake
def _(src_tok, dst_buf, slot_idx):
    return torch.empty_like(src_tok)


@torch.library.custom_op("mesh::ple_gate", mutates_args=())
def _mesh_ple_gate(x: torch.Tensor, w_gate: torch.Tensor, pli: torch.Tensor, layer_idx: int) -> torch.Tensor:
    g = torch.nn.functional.linear(x, w_gate)
    g = torch.nn.functional.gelu(g.float(), approximate="tanh").to(x.dtype)
    return (g.float() * pli[:, layer_idx, :].float()).to(x.dtype)


@_mesh_ple_gate.register_fake
def _(x, w_gate, pli, layer_idx):
    return torch.empty((x.shape[0], w_gate.shape[0]), dtype=x.dtype, device=x.device)


@torch.library.custom_op("mesh::vocab_argmax", mutates_args=("tok_buf", "pos_buf"))
def _mesh_vocab_argmax(
    logits: torch.Tensor,
    tok_buf: torch.Tensor,
    pos_buf: torch.Tensor,
    vocab_offset: int,
    update_inplace: bool,
) -> torch.Tensor:
    idx = logits.float().argmax(dim=-1).to(torch.int32) + vocab_offset
    if update_inplace:
        tok_buf.copy_(idx)
        pos_buf.add_(1)
    return idx


@_mesh_vocab_argmax.register_fake
def _(logits, tok_buf, pos_buf, vocab_offset, update_inplace):
    return torch.empty((logits.shape[0],), dtype=torch.int32, device=logits.device)


# ---- Native library loader ----
def load_libmesh_graph(path: str | pathlib.Path | None = None) -> C.CDLL:
    if path is None:
        here = pathlib.Path(__file__).resolve().parents[2] / "rdma" / ".build" / "libmesh-graph.dylib"
        alt = pathlib.Path(__file__).resolve().parents[2] / "rdma" / "libmesh-graph.dylib"
        path = here if here.exists() else alt
    lib = C.CDLL(str(path))
    lib.mesh_graph_create.argtypes = [C.c_int, C.c_int, C.c_char_p, C.c_int]
    lib.mesh_graph_create.restype = C.c_void_p
    lib.mesh_graph_destroy.argtypes = [C.c_void_p]
    lib.mesh_graph_destroy.restype = None
    lib.mesh_graph_add_buffer.argtypes = [C.c_void_p, C.c_uint64, C.c_int, C.c_void_p, C.c_int, C.c_uint64]
    lib.mesh_graph_add_buffer.restype = C.c_int
    lib.mesh_graph_import_buffer.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
    lib.mesh_graph_import_buffer.restype = C.c_int
    lib.mesh_graph_buffer_ptr.argtypes = [C.c_void_p, C.c_int]
    lib.mesh_graph_buffer_ptr.restype = C.c_void_p
    lib.mesh_graph_add_crossing.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32, C.c_int, C.c_int]
    lib.mesh_graph_add_crossing.restype = C.c_int
    lib.mesh_graph_add_op.argtypes = [C.c_void_p, C.POINTER(MeshGraphOpStruct)]
    lib.mesh_graph_add_op.restype = C.c_int
    lib.mesh_graph_compile.argtypes = [C.c_void_p]
    lib.mesh_graph_compile.restype = C.c_int
    lib.mesh_graph_run.argtypes = [
        C.c_void_p, C.c_uint32, C.c_uint32, C.POINTER(C.c_int32), C.c_uint32, C.POINTER(C.c_double)
    ]
    lib.mesh_graph_run.restype = C.c_double
    lib.mesh_graph_submit_step.argtypes = [C.c_void_p, C.c_uint32]
    lib.mesh_graph_submit_step.restype = C.c_void_p
    lib.mesh_graph_wait_step.argtypes = [C.c_void_p]
    lib.mesh_graph_wait_step.restype = None
    lib.mesh_graph_barrier.argtypes = [C.c_void_p]
    lib.mesh_graph_barrier.restype = None
    lib.mesh_graph_node_id.argtypes = [C.c_void_p]
    lib.mesh_graph_node_id.restype = C.c_uint32
    return lib


def _torch_to_mg_dtype(dt: torch.dtype) -> int:
    if dt == torch.bfloat16:
        return MG_DTYPE_BF16
    if dt == torch.float16:
        return MG_DTYPE_F16
    if dt == torch.float32:
        return MG_DTYPE_F32
    if dt == torch.int32 or dt == torch.int64:
        return MG_DTYPE_I32
    raise ValueError(f"Unsupported dtype for mesh-graph Metal lowering: {dt}")


def _dtype_itemsize(mg_dt: int) -> int:
    return 2 if mg_dt in (MG_DTYPE_BF16, MG_DTYPE_F16) else 4


@dataclass
class _TensorVal:
    name: str
    shape: tuple[int, ...]
    mg_dtype: int
    nbytes: int
    buf_id: int = -1
    is_persistent: bool = False  # weight, state, input, output, or view of persistent
    alias_of: str | None = None
    alias_byte_off: int = 0


@dataclass
class _FusedOp:
    kind: int
    mg_dtype: int
    inputs: list[str | None]
    outputs: list[str | None]
    u: list[int] = field(default_factory=lambda: [0] * 8)
    f: list[float] = field(default_factory=lambda: [0.0] * 4)
    crossing_spec: tuple[int, int, int, int] | None = None  # (n_elems, esize, wire_mode, is_argmax)


@dataclass
class FXCompileStats:
    fx_nodes: int
    fused_ops: int
    crossings: int
    arena_bytes: int
    weight_bytes: int
    wire_mode: str


class CompiledFXMetal:
    """Handle to a torch.export FX graph lowered into ONE pre-encoded native Metal command buffer."""

    def __init__(
        self,
        lib: C.CDLL,
        handle: int,
        state_meta: dict[str, tuple[int, tuple[int, ...], torch.dtype]],
        output_meta: list[tuple[str, int, tuple[int, ...], torch.dtype]],
        kept_weights: list[torch.Tensor],
        stats: FXCompileStats,
    ):
        self._lib = lib
        self._handle = handle
        self._state_meta = state_meta
        self._output_meta = output_meta
        self._kept_weights = kept_weights  # keep zero-copy borrowed tensors alive
        self.stats = stats

    def state_view(self, name: str) -> torch.Tensor:
        """Access-by-view: zero-copy torch.Tensor view directly over the shared Metal state buffer."""
        buf_id, shape, dtype = self._state_meta[name]
        ptr = self._lib.mesh_graph_buffer_ptr(self._handle, buf_id)
        nbytes = math.prod(shape) * dtype.itemsize
        raw = (C.c_byte * nbytes).from_address(ptr)
        return torch.frombuffer(raw, dtype=dtype, count=math.prod(shape)).reshape(shape)

    def output_views(self) -> list[torch.Tensor]:
        out = []
        for _, buf_id, shape, dtype in self._output_meta:
            ptr = self._lib.mesh_graph_buffer_ptr(self._handle, buf_id)
            nbytes = math.prod(shape) * dtype.itemsize
            raw = (C.c_byte * nbytes).from_address(ptr)
            out.append(torch.frombuffer(raw, dtype=dtype, count=math.prod(shape)).reshape(shape))
        return out

    def run(self, n_steps: int = 1, warm_steps: int = 0, out_tok_stride: int = 0) -> tuple[float, float, list[int]]:
        """Execute `n_steps` full forward passes in pure C++/Metal (1 CB/step, 0 Python dispatch per op/token).
        Returns (steady_ms_per_step, total_wall_ms, generated_tokens)."""
        wall = C.c_double(0.0)
        tok_arr = None
        tok_ptr = None
        if out_tok_stride > 0:
            tok_arr = (C.c_int32 * (n_steps * out_tok_stride))()
            tok_ptr = tok_arr
        steady_ms = float(
            self._lib.mesh_graph_run(
                self._handle, int(n_steps), int(warm_steps), tok_ptr, int(out_tok_stride), C.byref(wall)
            )
        )
        toks = [int(tok_arr[i]) for i in range(n_steps * out_tok_stride)] if tok_arr is not None else []
        return steady_ms, float(wall.value), toks

    def submit_step(self, step_idx: int = 0) -> int:
        """Non-blocking single-CB commit (returns opaque CB handle for wait_step)."""
        return int(self._lib.mesh_graph_submit_step(self._handle, int(step_idx)))

    def wait_step(self, cb_handle: int):
        """Wait for a submitted CB handle to complete."""
        self._lib.mesh_graph_wait_step(C.c_void_p(cb_handle))

    def barrier(self):
        self._lib.mesh_graph_barrier(self._handle)

    def close(self):
        if self._handle:
            self._lib.mesh_graph_destroy(self._handle)
            self._handle = 0

    def __del__(self):
        self.close()


def _node_shape_dtype(node: fx.Node) -> tuple[tuple[int, ...], int]:
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return tuple(int(s) for s in val.shape), _torch_to_mg_dtype(val.dtype)
    raise RuntimeError(f"FX node {node.name} ({node.target}) missing FakeTensor metadata")


def _target_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    if hasattr(target, "name"):
        return str(target.name())
    return str(target)


def _is_op(node: fx.Node, *suffixes: str) -> bool:
    if node.op != "call_function":
        return False
    tn = _target_name(node.target)
    return any(tn == s or tn.startswith(s + ".") for s in suffixes)


def _is_linear_op(node: fx.Node) -> bool:
    return _is_op(node, "aten::linear") or (
        _is_op(node, "aten::mm", "aten::matmul")
        and isinstance(node.args[1], fx.Node)
        and _is_op(node.args[1], "aten::t", "aten::transpose")
    )


def _linear_args(node: fx.Node) -> tuple[fx.Node, fx.Node]:
    if _is_op(node, "aten::linear"):
        return node.args[0], node.args[1]
    return node.args[0], node.args[1].args[0]


def _is_view_op(node: fx.Node) -> bool:
    return _is_op(
        node,
        "aten::view",
        "aten::_unsafe_view",
        "aten::reshape",
        "aten::squeeze",
        "aten::unsqueeze",
        "aten::flatten",
        "aten::unflatten",
        "aten::alias",
        "aten::contiguous",
    )


def _single_user_chain(node: fx.Node) -> fx.Node | None:
    users = [u for u in node.users if u.op != "output"]
    return users[0] if len(users) == 1 else None


def _is_gelu_op(node: fx.Node) -> bool:
    return _is_op(node, "aten::gelu")


def _is_silu_op(node: fx.Node) -> bool:
    return _is_op(node, "aten::silu")


def _is_cast_op(node: fx.Node) -> bool:
    return _is_op(node, "aten::_to_copy", "aten::to", "prims::convert_element_type")


def _unwrap_cast(node: fx.Node) -> fx.Node:
    while isinstance(node, fx.Node) and _is_cast_op(node):
        node = node.args[0]
    return node


def _resolve_rms_norm_weight(node: fx.Node) -> tuple[fx.Node, fx.Node, float] | None:
    if _is_op(node, "mesh::rms_norm"):
        return node.args[0], node.args[1], float(node.args[2])
    if _is_op(node, "aten::rms_norm"):
        x = node.args[0]
        w = node.args[2] if len(node.args) > 2 else None
        eps = float(node.args[3]) if len(node.args) > 3 and node.args[3] is not None else 1e-6
        return x, w, eps
    return None


def compile_fx_to_metal(
    exported_program,
    *,
    world_size: int = 1,
    rank: int = 0,
    comm_id: str = "",
    wire_mode: str | None = None,
    max_decode_steps: int = 128,
    shared_buffers: dict[str, tuple[CompiledFXMetal, str]] | None = None,
    lib_path: str | pathlib.Path | None = None,
) -> CompiledFXMetal:
    """Compile a torch.export.ExportedProgram end-to-end into ONE single native Metal command buffer."""
    wm_str = wire_mode or os.environ.get("MESH_FX_WIRE", "auto")
    wm_int = WIRE_MODE_MAP[wm_str]
    lib = load_libmesh_graph(lib_path)
    handle = lib.mesh_graph_create(int(world_size), int(rank), comm_id.encode("utf-8"), int(wm_int))
    if not handle:
        raise RuntimeError("mesh_graph_create failed")

    gm: fx.GraphModule = exported_program.graph_module
    sig = exported_program.graph_signature
    sd = exported_program.state_dict
    consts = exported_program.constants

    input_kind_by_node: dict[str, tuple[InputKind, str]] = {}
    for spec in sig.input_specs:
        input_kind_by_node[spec.arg.name] = (spec.kind, spec.target or spec.arg.name)

    vals: dict[str, _TensorVal] = {}
    state_meta: dict[str, tuple[int, tuple[int, ...], torch.dtype]] = {}
    kept_weights: list[torch.Tensor] = []
    weight_bytes = 0

    # 1. Bind placeholders (weights, buffers, constants, user inputs) as explicit Metal buffers.
    nodes = list(gm.graph.nodes)
    for node in nodes:
        if node.op != "placeholder":
            continue
        shape, mg_dt = _node_shape_dtype(node)
        nbytes = math.prod(shape) * _dtype_itemsize(mg_dt)
        kind, target = input_kind_by_node[node.name]
        if kind in (InputKind.PARAMETER, InputKind.CONSTANT_TENSOR):
            t = sd[target] if target in sd else consts[target]
            t = t.detach().cpu().contiguous()
            if t.dtype == torch.int64:
                t = t.to(torch.int32)
            kept_weights.append(t)
            bid = lib.mesh_graph_add_buffer(handle, t.numel() * t.element_size(), MG_BUF_WEIGHT, C.c_void_p(t.data_ptr()), -1, 0)
            weight_bytes += t.numel() * t.element_size()
            vals[node.name] = _TensorVal(node.name, shape, mg_dt, nbytes, buf_id=bid, is_persistent=True)
        elif kind == InputKind.BUFFER:
            t = sd[target].detach().cpu().contiguous()
            if t.dtype == torch.int64:
                t = t.to(torch.int32)
            if shared_buffers and target in shared_buffers:
                src_g, src_name = shared_buffers[target]
                src_bid = src_g._state_meta[src_name][0]
                bid = lib.mesh_graph_import_buffer(handle, src_g._handle, src_bid)
            else:
                bid = lib.mesh_graph_add_buffer(handle, t.numel() * t.element_size(), MG_BUF_STATE, C.c_void_p(t.data_ptr()), -1, 0)
            vals[node.name] = _TensorVal(node.name, shape, mg_dt, nbytes, buf_id=bid, is_persistent=True)
            state_meta[target] = (bid, shape, t.dtype)
        else:
            # User input state buffer
            val_t = node.meta["val"]
            dt = torch.int32 if val_t.dtype == torch.int64 else val_t.dtype
            bid = lib.mesh_graph_add_buffer(handle, math.prod(shape) * dt.itemsize, MG_BUF_STATE, None, -1, 0)
            vals[node.name] = _TensorVal(node.name, shape, mg_dt, math.prod(shape) * dt.itemsize, buf_id=bid, is_persistent=True)
            state_meta[node.name] = (bid, shape, dt)

    # 2. Functional FX walk + Algebraic Multi-Op Fusion Pass!
    consumed: set[fx.Node] = set()
    ops: list[_FusedOp] = []
    alias_map: dict[str, str] = {}

    def resolve(n: fx.Node) -> str:
        cur = n.name
        while cur in alias_map:
            cur = alias_map[cur]
        return cur

    def ensure_val(n: fx.Node) -> _TensorVal:
        if n.name not in vals:
            sh, dt = _node_shape_dtype(n)
            vals[n.name] = _TensorVal(n.name, sh, dt, math.prod(sh) * _dtype_itemsize(dt))
        return vals[n.name]

    # Pre-scan: identify transpose(weight) nodes only used by aten.mm so they don't emit ops
    for node in nodes:
        if _is_op(node, "aten::t", "aten::transpose"):
            users = list(node.users)
            if users and all(_is_op(u, "aten::mm", "aten::matmul") and u.args[1] is node for u in users):
                consumed.add(node)

    # Track output node names so outputs become persistent state buffers
    output_nodes: list[fx.Node] = []
    for node in nodes:
        if node.op == "output":
            flat_out = node.args[0]
            if isinstance(flat_out, (tuple, list)):
                output_nodes = [x for x in flat_out if isinstance(x, fx.Node)]
            elif isinstance(flat_out, fx.Node):
                output_nodes = [flat_out]

    for idx, node in enumerate(nodes):
        if node.op in ("placeholder", "output") or node in consumed:
            continue

        # Zero-copy access-by-view for reshape/view/squeeze/unsqueeze/flatten/unflatten
        if _is_view_op(node):
            src = resolve(node.args[0])
            alias_map[node.name] = src
            sh, dt = _node_shape_dtype(node)
            vals[node.name] = _TensorVal(node.name, sh, dt, math.prod(sh) * _dtype_itemsize(dt), alias_of=src)
            continue

        # Skip pure dtype casts that are folded into elementwise/gelu/rms_norm patterns
        if _is_cast_op(node):
            # Only pass through if its single user is gelu/silu/mul/add that already operates in fp32 internally
            u = _single_user_chain(node)
            if u is not None and (_is_gelu_op(u) or _is_silu_op(u) or _is_cast_op(u) or _is_op(u, "aten::mul", "aten::add")):
                alias_map[node.name] = resolve(node.args[0])
                continue
            src = resolve(node.args[0])
            alias_map[node.name] = src
            continue

        # ---- Pattern A: Embedding [+ optional scale mul] ----
        if _is_op(node, "aten::embedding"):
            w_n, tok_n = node.args[0], node.args[1]
            scale = 1.0
            out_node = node
            u = _single_user_chain(node)
            if u is not None and _is_op(u, "aten::mul") and isinstance(u.args[1], (int, float)):
                scale = float(u.args[1])
                consumed.add(u)
                out_node = u
            ov = ensure_val(out_node)
            B = math.prod(ov.shape[:-1])
            D = ov.shape[-1]
            ops.append(_FusedOp(
                MG_OP_EMBEDDING, ov.mg_dtype,
                inputs=[resolve(tok_n), resolve(w_n)],
                outputs=[out_node.name],
                u=[B, D, 0, 0, 0, 0, 0, 0],
                f=[scale, 0.0, 0.0, 0.0],
            ))
            continue

        # ---- Pattern B: Fused Linear -> AllReduce [-> RMSNorm(w1) -> Add(res) [-> RMSNorm(w2)]] ----
        if _is_linear_op(node):
            x_n, w_n = _linear_args(node)
            u1 = _single_user_chain(node)

            # Check B1: Row-parallel Linear + mesh::all_reduce_sum!
            if u1 is not None and _is_op(u1, "mesh::all_reduce_sum", "c10d_functional::all_reduce"):
                consumed.add(u1)
                yv = ensure_val(node)
                M = math.prod(yv.shape[:-1])
                N = yv.shape[-1]
                xv = vals[resolve(x_n)]
                K = xv.shape[-1]
                cid_spec = (M * N, _dtype_itemsize(yv.mg_dtype), wm_int, 0)

                # Emit fused LINEAR_PUBLISH
                ops.append(_FusedOp(
                    MG_OP_LINEAR_PUBLISH, yv.mg_dtype,
                    inputs=[resolve(x_n), resolve(w_n)],
                    outputs=[node.name],
                    u=[M, N, K, 0, 0, 0, 0, 0],
                    crossing_spec=cid_spec,
                ))

                # Try to fuse downstream: rms_norm(s, w1, eps) -> add(res, n1) [-> rms_norm(h, w2, eps)]
                u2 = _single_user_chain(u1)
                rn1 = _resolve_rms_norm_weight(u2) if u2 is not None else None
                u3 = _single_user_chain(u2) if rn1 is not None else None
                if rn1 is not None and u3 is not None and _is_op(u3, "aten::add"):
                    _, w1_n, eps1 = rn1
                    a0, a1 = u3.args[0], u3.args[1]
                    res_n = a1 if a0 is u2 else a0
                    consumed.add(u2)
                    consumed.add(u3)
                    ensure_val(u3)
                    # Check optional second RMSNorm (pre_feedforward_layernorm!)
                    # u3 (residual h) may have 2 users: residual add later + rms_norm(h, w2, eps)
                    w2_node = None
                    w2_n = None
                    for uu in list(u3.users):
                        rn2 = _resolve_rms_norm_weight(uu)
                        if rn2 is not None and abs(rn2[2] - eps1) < 1e-9:
                            w2_node = uu
                            w2_n = rn2[1]
                            break
                    if w2_node is not None:
                        consumed.add(w2_node)
                        ensure_val(w2_node)
                        ops.append(_FusedOp(
                            MG_OP_CROSSING_SUM_NORM, yv.mg_dtype,
                            inputs=[node.name, resolve(res_n), resolve(w1_n), resolve(w2_n)],
                            outputs=[u3.name, w2_node.name],
                            u=[0, M, N, 1, 0, 0, 0, 0],
                            f=[eps1, 0.0, 0.0, 0.0],
                        ))
                    else:
                        ops.append(_FusedOp(
                            MG_OP_CROSSING_SUM_NORM, yv.mg_dtype,
                            inputs=[node.name, resolve(res_n), resolve(w1_n), None],
                            outputs=[u3.name, None],
                            u=[0, M, N, 1, 0, 0, 0, 0],
                            f=[eps1, 0.0, 0.0, 0.0],
                        ))
                else:
                    # Pure all_reduce_sum without downstream RMSNorm fusion
                    ensure_val(u1)
                    ops.append(_FusedOp(
                        MG_OP_ALLREDUCE, yv.mg_dtype,
                        inputs=[node.name, None, None, None],
                        outputs=[u1.name, None],
                        u=[0, M, N, 1, 0, 0, 0, 0],
                    ))
                continue

            # Check B2: Fused Gate+Up GeGLU/SwiGLU: mul(act(linear(x, Wg)), linear(x, Wu))
            act_node = u1
            if act_node is not None and _is_cast_op(act_node):
                act_node = _single_user_chain(act_node)
            if act_node is not None and (_is_gelu_op(act_node) or _is_silu_op(act_node)):
                act_kind = 1 if _is_silu_op(act_node) else 0
                after_act = _single_user_chain(act_node)
                if after_act is not None and _is_cast_op(after_act):
                    after_act = _single_user_chain(after_act)
                if after_act is not None and _is_op(after_act, "aten::mul"):
                    other = after_act.args[1] if _unwrap_cast(after_act.args[0]) is act_node else after_act.args[0]
                    other_u = _unwrap_cast(other)
                    if isinstance(other_u, fx.Node) and _is_linear_op(other_u):
                        ox_n, ow_n = _linear_args(other_u)
                        if resolve(ox_n) == resolve(x_n):
                            # Mark entire 4-op subgraph consumed and emit ONE fused MG_OP_GATE_UP_ACT!
                            for cn in (u1, act_node, _single_user_chain(act_node), other, other_u, after_act):
                                if isinstance(cn, fx.Node) and cn is not node:
                                    consumed.add(cn)
                            ov = ensure_val(after_act)
                            B = math.prod(ov.shape[:-1])
                            N = ov.shape[-1]
                            K = vals[resolve(x_n)].shape[-1]
                            # Scratch buffer for B>=3 path
                            scr_name = f"{after_act.name}__up_scr"
                            vals[scr_name] = _TensorVal(scr_name, ov.shape, ov.mg_dtype, ov.nbytes)
                            ops.append(_FusedOp(
                                MG_OP_GATE_UP_ACT, ov.mg_dtype,
                                inputs=[resolve(x_n), resolve(w_n), resolve(ow_n)],
                                outputs=[after_act.name, scr_name],
                                u=[B, N, K, act_kind, 0, 0, 0, 0],
                            ))
                            continue

            # Plain linear (optionally fused with downstream scale? handled as general linear + elementwise)
            yv = ensure_val(node)
            M = math.prod(yv.shape[:-1])
            N = yv.shape[-1]
            K = vals[resolve(x_n)].shape[-1]
            ops.append(_FusedOp(
                MG_OP_LINEAR, yv.mg_dtype,
                inputs=[resolve(x_n), resolve(w_n)],
                outputs=[node.name],
                u=[M, N, K, 0, 0, 0, 0, 0],
            ))
            continue

        # ---- Pattern C: RMSNorm / Fused AddNorm(p, res, w1, scalar, w2, eps) ----
        rn = _resolve_rms_norm_weight(node)
        if rn is not None:
            px_n, w1_n, eps1 = rn
            u1 = _single_user_chain(node)
            # Check AddNorm pattern: h0 = add(res, rms_norm(p, w1)) -> h = mul(h0, scalar) -> xn = rms_norm(h, w2)
            if u1 is not None and _is_op(u1, "aten::add"):
                res_n = u1.args[1] if u1.args[0] is node else u1.args[0]
                u2 = _single_user_chain(u1)
                if u2 is not None and _is_op(u2, "aten::mul") and isinstance(u2.args[1], fx.Node):
                    scal_n = u2.args[1] if u2.args[0] is u1 else u2.args[0]
                    w2_node = None
                    w2_n = None
                    for uu in list(u2.users):
                        rn2 = _resolve_rms_norm_weight(uu)
                        if rn2 is not None and abs(rn2[2] - eps1) < 1e-9:
                            w2_node = uu
                            w2_n = rn2[1]
                            break
                    if w2_node is not None:
                        consumed.add(u1)
                        consumed.add(u2)
                        consumed.add(w2_node)
                        ov = ensure_val(u2)
                        ensure_val(w2_node)
                        rows = math.prod(ov.shape[:-1])
                        dim = ov.shape[-1]
                        ops.append(_FusedOp(
                            MG_OP_RMS_NORM, ov.mg_dtype,
                            inputs=[resolve(px_n), resolve(res_n), resolve(w1_n), resolve(scal_n), resolve(w2_n)],
                            outputs=[u2.name, w2_node.name],
                            u=[rows, dim, 0, 0, 0, 0, 0, 0],
                            f=[eps1, 0.0, 0.0, 0.0],
                        ))
                        continue
            ov = ensure_val(node)
            rows = math.prod(ov.shape[:-1])
            dim = ov.shape[-1]
            ops.append(_FusedOp(
                MG_OP_RMS_NORM, ov.mg_dtype,
                inputs=[resolve(px_n), None, resolve(w1_n) if w1_n is not None else None, None, None],
                outputs=[node.name, None],
                u=[rows, dim, 0, 0, 0, 0, 0, 0],
                f=[eps1, 0.0, 0.0, 0.0],
            ))
            continue

        # ---- Pattern D: HeadNorm + RoPE + Direct KV Cache Write (supports q_len>=1 causal verify) ----
        if _is_op(node, "mesh::head_norm_rope_kv"):
            args = node.args
            (qkv_n, wq_n, wk_n, pos_n, freqs_n, h_q, h_kv, head_dim, rope_mode, rope_base, max_seq, window, eps, kc_n, vc_n) = args[:15]
            q_len = int(args[15]) if len(args) > 15 else int(node.kwargs.get("q_len", 1))
            ov = ensure_val(node)
            B = ov.shape[0]
            rope_packed = (int(rope_mode) & 0xFF) | ((max(1, q_len) & 0xFFFF) << 8)
            ops.append(_FusedOp(
                MG_OP_HEAD_NORM_ROPE, ov.mg_dtype,
                inputs=[resolve(qkv_n), resolve(wq_n), resolve(wk_n), resolve(pos_n), resolve(freqs_n)],
                outputs=[node.name, resolve(kc_n) if h_kv > 0 else None, resolve(vc_n) if h_kv > 0 else None],
                u=[B, int(h_q), int(h_kv), int(h_kv), int(head_dim), rope_packed, int(max_seq), int(window)],
                f=[float(eps), math.log2(float(rope_base)) if rope_base > 0 else 0.0, 0.0, 0.0],
            ))
            continue

        # ---- Pattern E: Fused GQA KV SDPA Decode (supports q_len>=1 causal verify & shared_kv_b0) ----
        if _is_op(node, "mesh::kv_sdpa_decode"):
            args = node.args
            (q_n, kc_n, vc_n, pos_n, h_q, h_kv, head_dim, max_seq, window, scale) = args[:10]
            q_len = int(args[10]) if len(args) > 10 else int(node.kwargs.get("q_len", 1))
            shared_b0 = bool(args[11]) if len(args) > 11 else bool(node.kwargs.get("shared_kv_b0", False))
            ov = ensure_val(node)
            B = ov.shape[0]
            ops.append(_FusedOp(
                MG_OP_KV_SDPA_DECODE, ov.mg_dtype,
                inputs=[resolve(q_n), resolve(kc_n), resolve(vc_n), resolve(pos_n)],
                outputs=[node.name],
                u=[B, int(h_q), int(h_kv), int(head_dim), int(max_seq), int(window), max(1, q_len), 1 if shared_b0 else 0],
                f=[float(scale), 0.0, 0.0, 0.0],
            ))
            continue

        # ---- Pattern F: Per-Layer Input Gate ----
        if _is_op(node, "mesh::ple_gate"):
            x_n, wg_n, pli_n, layer_idx = node.args
            ov = ensure_val(node)
            B = ov.shape[0]
            N = ov.shape[-1]
            K = vals[resolve(x_n)].shape[-1]
            pli_v = vals[resolve(pli_n)]
            stride = math.prod(pli_v.shape[1:])
            ops.append(_FusedOp(
                MG_OP_PLE_GATE, ov.mg_dtype,
                inputs=[resolve(x_n), resolve(wg_n), resolve(pli_n)],
                outputs=[node.name],
                u=[B, N, K, stride, int(layer_idx), 0, 0, 0],
            ))
            continue

        # ---- Pattern G: Vocab-Parallel Global Argmax (+ optional in-GPU tok_buf/pos_buf update!) ----
        if _is_op(node, "mesh::vocab_argmax"):
            lg_n, tok_n, pos_n, v_off, upd = node.args
            ov = ensure_val(node)
            lg_v = vals[resolve(lg_n)]
            B = lg_v.shape[0]
            V = lg_v.shape[-1]
            P = (V + 2047) // 2048
            if upd:
                # Persistent output token history buffer [max_decode_steps, B] int32
                hist_bytes = max(1, max_decode_steps) * B * 4
                hist_bid = lib.mesh_graph_add_buffer(handle, hist_bytes, MG_BUF_STATE, None, -1, 0)
                ov.buf_id = hist_bid
                ov.is_persistent = True
                state_meta[node.name] = (hist_bid, (max_decode_steps, B), torch.int32)
            # Temporary scratch buffers for row_argmax_parts & local_best
            p_name = f"{node.name}__parts"
            lb_name = f"{node.name}__lbest"
            vals[p_name] = _TensorVal(p_name, (B, P, 2), MG_DTYPE_U32, B * P * 8)
            vals[lb_name] = _TensorVal(lb_name, (B, 2), MG_DTYPE_U32, B * 8)
            ops.append(_FusedOp(
                MG_OP_VOCAB_ARGMAX, MG_DTYPE_I32,
                inputs=[resolve(lg_n), resolve(tok_n), resolve(pos_n)],
                outputs=[node.name, p_name, lb_name],
                u=[B, V, int(v_off), 0, 1 if upd else 0, 0, 0, 0],
                crossing_spec=(B, 8, MG_WIRE_GATHER, 1) if world_size > 1 else None,
            ))
            continue

        # ---- Pattern G2: In-GPU Lossless Speculative Verify + Commit ----
        if _is_op(node, "mesh::spec_verify_commit"):
            targ_n, draft_n, xn_n, pos_n, apos_n, stats_n, K_spec = node.args
            K_spec = int(K_spec)
            D = vals[resolve(xn_n)].shape[-1]
            stats_v = vals[resolve(stats_n)]
            s_stride = stats_v.shape[-1]
            # Locate the two getitem users (0 = out_anchor_tok, 1 = out_hidden)
            gi_tok, gi_hid = None, None
            for uu in list(node.users):
                if uu.op == "call_function" and "getitem" in _target_name(uu.target):
                    if int(uu.args[1]) == 0:
                        gi_tok = uu
                    elif int(uu.args[1]) == 1:
                        gi_hid = uu
            tok_name = gi_tok.name if gi_tok is not None else f"{node.name}__tok"
            hid_name = gi_hid.name if gi_hid is not None else f"{node.name}__hid"
            if gi_tok is not None:
                consumed.add(gi_tok)
                ensure_val(gi_tok)
            else:
                vals[tok_name] = _TensorVal(tok_name, (1,), MG_DTYPE_I32, 4)
            if gi_hid is not None:
                consumed.add(gi_hid)
                ensure_val(gi_hid)
            else:
                vals[hid_name] = _TensorVal(hid_name, (1, D), vals[resolve(xn_n)].mg_dtype, D * 2)
            ops.append(_FusedOp(
                MG_OP_SPEC_VERIFY_COMMIT, vals[resolve(xn_n)].mg_dtype,
                inputs=[resolve(targ_n), resolve(draft_n), resolve(xn_n), resolve(pos_n), resolve(apos_n), resolve(stats_n)],
                outputs=[tok_name, hid_name],
                u=[K_spec, D, s_stride, 0, 0, 0, 0, 0],
            ))
            continue

        # ---- Pattern G3: Write 1 draft token into draft_tok[slot_idx] ----
        if _is_op(node, "mesh::write_tok_slot"):
            src_n, dst_n, slot_idx = node.args
            src_res = resolve(src_n)
            alias_map[node.name] = src_res
            ops.append(_FusedOp(
                MG_OP_WRITE_TOK_SLOT, MG_DTYPE_I32,
                inputs=[src_res, resolve(dst_n)],
                outputs=[],
                u=[int(slot_idx), 0, 0, 0, 0, 0, 0, 0],
            ))
            continue

        # ---- Pattern G4: Concat along last dim (aten::cat([a, b], dim=-1)) ----
        if _is_op(node, "aten::cat"):
            tensors_arg = node.args[0]
            if len(tensors_arg) != 2:
                raise NotImplementedError(f"aten::cat supports 2 inputs, got {len(tensors_arg)}")
            a_n, b_n = tensors_arg[0], tensors_arg[1]
            av = vals[resolve(a_n)]
            bv = vals[resolve(b_n)]
            ov = ensure_val(node)
            M = math.prod(ov.shape[:-1])
            K0 = av.shape[-1]
            K1 = bv.shape[-1]
            ops.append(_FusedOp(
                MG_OP_CONCAT, ov.mg_dtype,
                inputs=[resolve(a_n), resolve(b_n)],
                outputs=[node.name],
                u=[M, K0, K1, 0, 0, 0, 0, 0],
            ))
            continue

        # ---- Pattern H: Standalone AllReduce ----
        if _is_op(node, "mesh::all_reduce_sum", "c10d_functional::all_reduce"):
            x_n = node.args[0]
            ov = ensure_val(node)
            rows = math.prod(ov.shape[:-1]) if len(ov.shape) > 1 else 1
            dim = ov.shape[-1]
            ops.append(_FusedOp(
                MG_OP_ALLREDUCE, ov.mg_dtype,
                inputs=[resolve(x_n), None, None, None],
                outputs=[node.name, None],
                u=[0, rows, dim, 0, 0, 0, 0, 0],
                crossing_spec=(rows * dim, _dtype_itemsize(ov.mg_dtype), wm_int, 0),
            ))
            continue

        # ---- Pattern I: Elementwise (add, mul, scale, add_scale, gelu, silu, relu, copy) ----
        if _is_op(node, "aten::add"):
            a0, a1 = node.args[0], node.args[1]
            u1 = _single_user_chain(node)
            if u1 is not None and _is_op(u1, "aten::mul") and isinstance(u1.args[1], (int, float)):
                consumed.add(u1)
                ov = ensure_val(u1)
                ops.append(_FusedOp(
                    MG_OP_ELEMENTWISE, ov.mg_dtype,
                    inputs=[resolve(a0), resolve(a1)],
                    outputs=[u1.name],
                    u=[math.prod(ov.shape), MG_EW_ADD_SCALE, 0, 0, 0, 0, 0, 0],
                    f=[float(u1.args[1]), 0.0, 0.0, 0.0],
                ))
            else:
                ov = ensure_val(node)
                ops.append(_FusedOp(
                    MG_OP_ELEMENTWISE, ov.mg_dtype,
                    inputs=[resolve(a0), resolve(a1)],
                    outputs=[node.name],
                    u=[math.prod(ov.shape), MG_EW_ADD, 0, 0, 0, 0, 0, 0],
                ))
            continue

        if _is_op(node, "aten::mul"):
            a0, a1 = node.args[0], node.args[1]
            ov = ensure_val(node)
            if isinstance(a1, (int, float)):
                ops.append(_FusedOp(
                    MG_OP_ELEMENTWISE, ov.mg_dtype,
                    inputs=[resolve(a0), None],
                    outputs=[node.name],
                    u=[math.prod(ov.shape), MG_EW_SCALE, 0, 0, 0, 0, 0, 0],
                    f=[float(a1), 0.0, 0.0, 0.0],
                ))
            elif isinstance(a0, (int, float)):
                ops.append(_FusedOp(
                    MG_OP_ELEMENTWISE, ov.mg_dtype,
                    inputs=[resolve(a1), None],
                    outputs=[node.name],
                    u=[math.prod(ov.shape), MG_EW_SCALE, 0, 0, 0, 0, 0, 0],
                    f=[float(a0), 0.0, 0.0, 0.0],
                ))
            else:
                ops.append(_FusedOp(
                    MG_OP_ELEMENTWISE, ov.mg_dtype,
                    inputs=[resolve(a0), resolve(a1)],
                    outputs=[node.name],
                    u=[math.prod(ov.shape), MG_EW_MUL, 0, 0, 0, 0, 0, 0],
                ))
            continue

        if _is_gelu_op(node) or _is_silu_op(node) or _is_op(node, "aten::relu"):
            sub = MG_EW_GELU if _is_gelu_op(node) else (MG_EW_SILU if _is_silu_op(node) else MG_EW_RELU)
            ov = ensure_val(node)
            ops.append(_FusedOp(
                MG_OP_ELEMENTWISE, ov.mg_dtype,
                inputs=[resolve(node.args[0]), None],
                outputs=[node.name],
                u=[math.prod(ov.shape), sub, 0, 0, 0, 0, 0, 0],
            ))
            continue

        if _is_op(node, "aten::clone", "aten::copy"):
            ov = ensure_val(node)
            ops.append(_FusedOp(
                MG_OP_ELEMENTWISE, ov.mg_dtype,
                inputs=[resolve(node.args[0]), None],
                outputs=[node.name],
                u=[math.prod(ov.shape), MG_EW_COPY, 0, 0, 0, 0, 0, 0],
            ))
            continue

        raise NotImplementedError(f"Unsupported FX node in mesh-graph Metal lowering: {node.name} ({node.target})")

    # 3. Assign persistent state buffers to graph outputs (so callers can read them zero-copy!)
    output_meta: list[tuple[str, int, tuple[int, ...], torch.dtype]] = []
    for on in output_nodes:
        rn_name = resolve(on)
        tv = vals[rn_name]
        val_t = on.meta["val"]
        dt = torch.int32 if val_t.dtype == torch.int64 else val_t.dtype
        if not tv.is_persistent:
            bid = lib.mesh_graph_add_buffer(handle, tv.nbytes, MG_BUF_STATE, None, -1, 0)
            tv.buf_id = bid
            tv.is_persistent = True
        output_meta.append((on.name, tv.buf_id, tv.shape, dt))

    # 4. Static Liveness Interval Coloring -> Pack all temporary activations into ONE shared Metal arena!
    first_def: dict[str, int] = {}
    last_use: dict[str, int] = {}
    for step_i, op in enumerate(ops):
        for o in op.outputs:
            if o is not None and not vals[o].is_persistent:
                first_def.setdefault(o, step_i)
                last_use[o] = max(last_use.get(o, step_i), step_i)
        for inp in op.inputs:
            if inp is not None and not vals[inp].is_persistent:
                last_use[inp] = max(last_use.get(inp, step_i), step_i)

    # Greedy interval packing (256-byte alignment for Metal buffer offsets)
    free_blocks: list[tuple[int, int]] = []  # (offset, size)
    active: list[tuple[int, int, int]] = []  # (end_step, offset, size)
    offsets: dict[str, int] = {}
    arena_peak = 0

    for name in sorted(first_def, key=lambda k: first_def[k]):
        st = first_def[name]
        en = last_use[name]
        sz = (vals[name].nbytes + 255) & ~255
        # Reclaim expired intervals
        still_active = []
        for a_en, a_off, a_sz in active:
            if a_en < st:
                free_blocks.append((a_off, a_sz))
            else:
                still_active.append((a_en, a_off, a_sz))
        active = still_active
        # Coalesce adjacent free blocks
        if len(free_blocks) > 1:
            free_blocks.sort()
            merged = [free_blocks[0]]
            for fo, fs in free_blocks[1:]:
                po, ps = merged[-1]
                if po + ps == fo:
                    merged[-1] = (po, ps + fs)
                else:
                    merged.append((fo, fs))
            free_blocks = merged
        # Best-fit search
        best_i = -1
        best_sz = 1 << 62
        for bi, (fo, fs) in enumerate(free_blocks):
            if fs >= sz and fs < best_sz:
                best_i = bi
                best_sz = fs
        if best_i >= 0:
            fo, fs = free_blocks.pop(best_i)
            offsets[name] = fo
            if fs > sz:
                free_blocks.append((fo + sz, fs - sz))
            active.append((en, fo, sz))
        else:
            offsets[name] = arena_peak
            active.append((en, arena_peak, sz))
            arena_peak += sz

    arena_bytes = max(4096, (arena_peak + 4095) & ~4095)
    arena_bid = lib.mesh_graph_add_buffer(handle, arena_bytes, MG_BUF_ARENA, None, -1, 0)
    for name, off in offsets.items():
        vals[name].buf_id = lib.mesh_graph_add_buffer(handle, vals[name].nbytes, MG_BUF_VIEW, None, arena_bid, off)

    # 5. Allocate flat-DMA crossings & emit flat C MeshGraphOpStruct tape!
    crossing_count = 0
    last_cid = -1
    for op in ops:
        if op.crossing_spec is not None and world_size > 1:
            n_el, esz, wm, is_arg = op.crossing_spec
            cid = lib.mesh_graph_add_crossing(handle, int(n_el), int(esz), int(wm), int(is_arg))
            last_cid = cid
            crossing_count += 1
            if op.kind == MG_OP_LINEAR_PUBLISH:
                op.u[3] = cid
            elif op.kind == MG_OP_VOCAB_ARGMAX:
                op.u[3] = cid
            elif op.kind == MG_OP_ALLREDUCE:
                op.u[0] = cid
        elif op.kind in (MG_OP_CROSSING_SUM_NORM, MG_OP_ALLREDUCE) and op.crossing_spec is None and world_size > 1:
            op.u[0] = last_cid

        st = MeshGraphOpStruct()
        st.kind = op.kind
        st.dtype = op.mg_dtype
        for k in range(6):
            st.in_buf[k] = vals[op.inputs[k]].buf_id if (k < len(op.inputs) and op.inputs[k] is not None) else -1
        for k in range(3):
            st.out_buf[k] = vals[op.outputs[k]].buf_id if (k < len(op.outputs) and op.outputs[k] is not None) else -1
        for k in range(8):
            st.u[k] = op.u[k]
        for k in range(4):
            st.f[k] = op.f[k]
        lib.mesh_graph_add_op(handle, C.byref(st))

    rc = lib.mesh_graph_compile(handle)
    if rc != 0:
        lib.mesh_graph_destroy(handle)
        raise RuntimeError(f"mesh_graph_compile failed with rc={rc}")

    stats = FXCompileStats(
        fx_nodes=len(nodes),
        fused_ops=len(ops),
        crossings=crossing_count,
        arena_bytes=arena_bytes,
        weight_bytes=weight_bytes,
        wire_mode=wm_str,
    )
    return CompiledFXMetal(lib, handle, state_meta, output_meta, kept_weights, stats)
