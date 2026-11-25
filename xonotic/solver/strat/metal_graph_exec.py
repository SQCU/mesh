from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import torch

MG_BUF_WEIGHT = 0
MG_BUF_ARENA = 1
MG_BUF_STATE = 2
MG_DTYPE_BF16 = 0
MG_DTYPE_F32 = 2
MG_WIRE_AUTO = 5

MG_OP_LINEAR = 2
MG_OP_RMS_NORM = 5
MG_OP_GATE_UP_ACT = 8
MG_OP_ELEMENTWISE = 10
MG_EW_ADD = 0


class MeshGraphOp(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_uint32),
        ("dtype", ctypes.c_uint32),
        ("in_ids", ctypes.c_int32 * 6),
        ("out_ids", ctypes.c_int32 * 3),
        ("u", ctypes.c_uint32 * 8),
        ("f", ctypes.c_float * 4),
    ]


_LIB = None


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    root = Path(__file__).resolve().parents[3]
    path = root / "rdma" / ".build" / "libmesh-graph.dylib"
    if not path.is_file():
        return None
    lib = ctypes.CDLL(str(path))
    lib.mesh_graph_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.mesh_graph_create.restype = ctypes.c_void_p
    lib.mesh_graph_destroy.argtypes = [ctypes.c_void_p]
    lib.mesh_graph_add_buffer.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
    ]
    lib.mesh_graph_add_buffer.restype = ctypes.c_int
    lib.mesh_graph_buffer_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.mesh_graph_buffer_ptr.restype = ctypes.c_void_p
    lib.mesh_graph_add_op.argtypes = [ctypes.c_void_p, ctypes.POINTER(MeshGraphOp)]
    lib.mesh_graph_add_op.restype = ctypes.c_int
    lib.mesh_graph_compile.argtypes = [ctypes.c_void_p]
    lib.mesh_graph_compile.restype = ctypes.c_int
    lib.mesh_graph_run.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p
    ]
    lib.mesh_graph_run.restype = ctypes.c_double
    _LIB = lib
    return lib


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    rounding = ((u32 >> 16) & 1) + 0x7FFF
    return ((u32 + rounding) >> 16).astype(np.uint16)


def _bf16_u16_to_f32(arr: np.ndarray) -> np.ndarray:
    return (arr.astype(np.uint32) << 16).view(np.float32)


class _SwiGLUBlockGraph:
    """Single-command-buffer Metal execution of: out = x + Linear_out(silu(Linear_gate(rms_norm(x))) * Linear_val(rms_norm(x)))."""

    def __init__(self, lib, rows: int, d: int, h: int):
        self.lib = lib
        self.rows, self.d, self.h = int(rows), int(d), int(h)
        self.g = lib.mesh_graph_create(1, 0, b"xonotic-policy", MG_WIRE_AUTO)
        if not self.g:
            raise RuntimeError("mesh_graph_create failed")
        n_x = self.rows * self.d * 2
        n_h = self.rows * self.h * 2
        ones = _f32_to_bf16_u16(np.ones((self.d,), dtype=np.float32))
        self.b_x = lib.mesh_graph_add_buffer(self.g, n_x, MG_BUF_STATE, None, -1, 0)
        self.b_norm_w = lib.mesh_graph_add_buffer(self.g, self.d * 2, MG_BUF_WEIGHT, ones.ctypes.data, -1, 0)
        self.b_wg = lib.mesh_graph_add_buffer(self.g, self.h * self.d * 2, MG_BUF_STATE, None, -1, 0)
        self.b_wu = lib.mesh_graph_add_buffer(self.g, self.h * self.d * 2, MG_BUF_STATE, None, -1, 0)
        self.b_wo = lib.mesh_graph_add_buffer(self.g, self.d * self.h * 2, MG_BUF_STATE, None, -1, 0)
        self.b_xn = lib.mesh_graph_add_buffer(self.g, n_x, MG_BUF_ARENA, None, -1, 0)
        self.b_hid = lib.mesh_graph_add_buffer(self.g, n_h, MG_BUF_ARENA, None, -1, 0)
        self.b_up = lib.mesh_graph_add_buffer(self.g, n_h, MG_BUF_ARENA, None, -1, 0)
        self.b_proj = lib.mesh_graph_add_buffer(self.g, n_x, MG_BUF_ARENA, None, -1, 0)
        self.b_out = lib.mesh_graph_add_buffer(self.g, n_x, MG_BUF_STATE, None, -1, 0)

        def add_op(kind, ins, outs, u=(), f=()):
            op = MeshGraphOp()
            op.kind = kind
            op.dtype = MG_DTYPE_BF16
            for i in range(6):
                op.in_ids[i] = ins[i] if i < len(ins) else -1
            for i in range(3):
                op.out_ids[i] = outs[i] if i < len(outs) else -1
            for i in range(8):
                op.u[i] = u[i] if i < len(u) else 0
            for i in range(4):
                op.f[i] = f[i] if i < len(f) else 0.0
            if lib.mesh_graph_add_op(self.g, ctypes.byref(op)) != 0:
                raise RuntimeError(f"mesh_graph_add_op failed for kind={kind}")

        add_op(MG_OP_RMS_NORM, (self.b_x, -1, self.b_norm_w), (self.b_xn,), u=(self.rows, self.d), f=(1e-6,))
        add_op(MG_OP_GATE_UP_ACT, (self.b_xn, self.b_wg, self.b_wu), (self.b_hid, self.b_up), u=(self.rows, self.h, self.d, 1))
        add_op(MG_OP_LINEAR, (self.b_hid, self.b_wo), (self.b_proj,), u=(self.rows, self.d, self.h))
        add_op(MG_OP_ELEMENTWISE, (self.b_x, self.b_proj), (self.b_out,), u=(self.rows * self.d, MG_EW_ADD))
        if lib.mesh_graph_compile(self.g) != 0:
            raise RuntimeError("mesh_graph_compile failed")
        self.step = 0
        self.weight_version = -1

    def _view(self, buf_id, shape):
        ptr = self.lib.mesh_graph_buffer_ptr(self.g, buf_id)
        arr = (ctypes.c_uint16 * int(np.prod(shape))).from_address(ptr)
        return np.ctypeslib.as_array(arr).reshape(shape)

    def sync_weights(self, wg, wu, wo, version):
        if self.weight_version == version:
            return
        np.copyto(self._view(self.b_wg, (self.h, self.d)), _f32_to_bf16_u16(wg.detach().cpu().numpy()))
        np.copyto(self._view(self.b_wu, (self.h, self.d)), _f32_to_bf16_u16(wu.detach().cpu().numpy()))
        np.copyto(self._view(self.b_wo, (self.d, self.h)), _f32_to_bf16_u16(wo.detach().cpu().numpy()))
        self.weight_version = version

    def run(self, x_np: np.ndarray) -> np.ndarray:
        np.copyto(self._view(self.b_x, (self.rows, self.d)), _f32_to_bf16_u16(x_np))
        self.lib.mesh_graph_run(self.g, 1, 0, None, 0, None)
        self.step += 1
        return _bf16_u16_to_f32(self._view(self.b_out, (self.rows, self.d)).copy())

    def close(self):
        if self.g:
            self.lib.mesh_graph_destroy(self.g)
            self.g = None


class MetalPolicyGraph:
    """Single-command-buffer Metal 4 executor cache for Wally's dense SwiGLU blocks (encode_rows & gia_uma_dov)."""

    def __init__(self):
        self.lib = _load_lib()
        self.blocks: dict[tuple[str, int, int, int], _SwiGLUBlockGraph] = {}
        self.version = 0
        self.calls = 0
        self._disabled = self.lib is None

    @property
    def available(self) -> bool:
        return not self._disabled

    def invalidate_weights(self):
        self.version += 1

    def run_swiglu(self, tag: str, x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor, wo: torch.Tensor):
        if self._disabled or x.requires_grad or torch.is_grad_enabled():
            return None
        orig_shape = x.shape
        d = orig_shape[-1]
        h = wg.shape[0]
        flat = x.detach().reshape(-1, d)
        rows = flat.shape[0]
        if rows <= 0:
            return None
        key = (tag, rows, d, h)
        block = self.blocks.get(key)
        if block is None:
            try:
                block = _SwiGLUBlockGraph(self.lib, rows, d, h)
                self.blocks[key] = block
            except Exception:
                self._disabled = True
                return None
        block.sync_weights(wg, wu, wo, self.version)
        out_np = block.run(flat.cpu().numpy().astype(np.float32, copy=False))
        self.calls += 1
        return torch.from_numpy(out_np).to(device=x.device, dtype=x.dtype).reshape(orig_shape)

    def close(self):
        for block in self.blocks.values():
            block.close()
        self.blocks.clear()


def maybe_metal_graph() -> MetalPolicyGraph | None:
    if os.environ.get("MESH_POLICY_METAL_GRAPH", "0") not in ("1", "true", "yes"):
        return None
    mg = MetalPolicyGraph()
    return mg if mg.available else None
