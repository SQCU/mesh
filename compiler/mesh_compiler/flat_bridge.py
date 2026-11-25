# Ticket T5: Direct flat page-table lowering & zero-copy runtime bridge (targeting main's libmesh.dylib aef86b43b).
# Honors both functional compilation and access-by-view: /mesh0 pages are exposed directly as torch.frombuffer views
# with page-table indices W[dst_idx] serving as both storage reference and reduction completion witness.
import ctypes as C
import pathlib
import torch

from .algebra import (
    ALG_C2_INTERVAL,
    RED_NONE,
    complete_tag,
    leaf_tag,
    red_alg,
)

MESH_PAGE_BYTES = 4096
MESH_PAGE_NONE = 0xFFFFFFFF


class MeshCtx(C.Structure):
    _fields_ = [("M", C.c_void_p), ("len", C.c_size_t), ("fd", C.c_int)]


class MeshEntryInfo(C.Structure):
    _fields_ = [("page", C.c_uint32), ("bytes", C.c_uint32), ("data", C.c_void_p)]


class MeshXfer(C.Structure):
    _fields_ = [
        ("src_page", C.c_uint32),
        ("bytes", C.c_uint32),
        ("dst_node", C.c_uint32),
        ("dst_index", C.c_uint32),
    ]


class MeshRedXfer(C.Structure):
    _fields_ = [
        ("src_page", C.c_uint32),
        ("bytes", C.c_uint32),
        ("dst_node", C.c_uint32),
        ("dst_index", C.c_uint32),
        ("op", C.c_uint8),
        ("pad", C.c_uint8),
        ("tag", C.c_uint16),
    ]


class MeshReach(C.Structure):
    _fields_ = [("node", C.c_uint32), ("padding", C.c_uint32), ("bandwidth", C.c_uint64)]


class MeshLinkMap(C.Structure):
    _fields_ = [
        ("kind", C.c_uint32),
        ("nodes", C.c_uint32),
        ("links", C.c_uint32),
        ("link", C.POINTER(C.c_uint32 * 2)),
        ("bandwidth", C.POINTER(C.c_uint64)),
    ]


def load_libmesh(path: str | pathlib.Path | None = None):
    if path is None:
        active = pathlib.Path.home() / ".cache" / "mesh" / "active-source"
        if active.exists():
            path = pathlib.Path(active.read_text().strip()) / "rdma" / "libmesh.dylib"
        else:
            path = pathlib.Path(__file__).parents[2] / "rdma" / "libmesh.dylib"
    lib = C.CDLL(str(path))
    lib.mesh_attach.argtypes = [C.POINTER(MeshCtx)]
    lib.mesh_attach.restype = C.c_int
    lib.mesh_detach.argtypes = [C.POINTER(MeshCtx)]
    lib.mesh_detach.restype = C.c_int
    lib.mesh_alloc.argtypes = [C.POINTER(MeshCtx), C.c_uint32, C.c_int, C.POINTER(C.c_uint32), C.POINTER(C.c_uint32)]
    lib.mesh_alloc.restype = C.c_int
    lib.mesh_alloc_contiguous.argtypes = [C.POINTER(MeshCtx), C.c_uint32, C.POINTER(C.c_uint32), C.POINTER(C.c_void_p)]
    lib.mesh_alloc_contiguous.restype = C.c_int
    lib.mesh_write.argtypes = [C.POINTER(MeshCtx), C.POINTER(MeshXfer), C.c_uint32]
    lib.mesh_write.restype = C.c_int
    lib.mesh_reduce_write.argtypes = [C.POINTER(MeshCtx), C.POINTER(MeshRedXfer), C.c_uint32]
    lib.mesh_reduce_write.restype = C.c_int
    lib.mesh_entry.argtypes = [C.POINTER(MeshCtx), C.c_uint32, C.POINTER(MeshEntryInfo)]
    lib.mesh_entry.restype = C.c_int
    lib.mesh_entry_red.argtypes = [C.POINTER(MeshCtx), C.c_uint32, C.POINTER(MeshEntryInfo), C.POINTER(C.c_uint16)]
    lib.mesh_entry_red.restype = C.c_int
    lib.mesh_clear_indices.argtypes = [C.POINTER(MeshCtx), C.POINTER(C.c_uint32), C.c_uint32]
    lib.mesh_clear_indices.restype = None
    lib.mesh_free_indices.argtypes = [C.POINTER(MeshCtx), C.POINTER(C.c_uint32), C.c_uint32]
    lib.mesh_free_indices.restype = None
    lib.mesh_page_ptr.argtypes = [C.POINTER(MeshCtx), C.c_uint32]
    lib.mesh_page_ptr.restype = C.c_void_p
    lib.mesh_release.argtypes = [C.POINTER(MeshCtx), C.c_uint32]
    lib.mesh_release.restype = None
    lib.mesh_nodes_reached.argtypes = [C.c_void_p, C.POINTER(MeshReach), C.c_uint32]
    lib.mesh_nodes_reached.restype = C.c_uint32
    if hasattr(lib, "mesh_map"):
        lib.mesh_map.argtypes = [C.POINTER(C.c_uint32), C.POINTER(MeshLinkMap)]
        lib.mesh_map.restype = C.c_int
    if hasattr(lib, "mesh_map_serial"):
        lib.mesh_map_serial.argtypes = [C.POINTER(C.c_uint64)]
        lib.mesh_map_serial.restype = C.c_int
    return lib


class FlatBridge:
    """Direct zero-copy Python/PyTorch interface over /mesh0 flat page-table & algebraic reduction verbs."""

    def __init__(self, lib_path: str | pathlib.Path | None = None):
        self.lib = load_libmesh(lib_path)
        self.ctx = MeshCtx()
        rc = self.lib.mesh_attach(C.byref(self.ctx))
        if rc:
            raise OSError(rc, f"mesh_attach failed: {rc}")
        # rdma/mesh.h static_assert: hdr.node @ 20, hdr.data_off @ 88, hdr.length @ 96, hdr.map @ 136
        self.node = int(C.c_uint32.from_address(self.ctx.M + 20).value)
        self.data_off = int(C.c_uint64.from_address(self.ctx.M + 88).value)
        self.length = int(C.c_uint64.from_address(self.ctx.M + 96).value)

    def map_serial(self) -> int:
        """Read live atomic hdr->map serial (offset 136); increments on any cable or membership change."""
        return int(C.c_uint64.from_address(self.ctx.M + 136).value)

    def reached(self) -> list[tuple[int, float]]:
        buf = (MeshReach * 32)()
        n = int(self.lib.mesh_nodes_reached(self.ctx.M, buf, 32))
        return [(int(buf[i].node), float(buf[i].bandwidth) / 8e9) for i in range(n)]

    def live_topology(self):
        """Construct the exact live multi-hop physical TopologyGraph directly from /mesh0 (mesh_map).
        Aggregates parallel cables per directed (u -> v) edge into GB/s rates; zero hardcoded topology."""
        from .topology import DirectedLink, TopologyGraph
        serial = self.map_serial()
        if hasattr(self.lib, "mesh_map"):
            max_n, max_l = 64, 512
            nodes_arr = (C.c_uint32 * max_n)()
            links_arr = ((C.c_uint32 * 2) * max_l)()
            bw_arr = (C.c_uint64 * max_l)()
            lmap = MeshLinkMap(0, max_n, max_l, links_arr, bw_arr)
            n = int(self.lib.mesh_map(nodes_arr, C.byref(lmap)))
            if n > 0:
                node_ids = tuple(int(nodes_arr[i]) for i in range(n))
                pair_agg: dict[tuple[int, int], list] = {}
                for k in range(int(lmap.links)):
                    u = node_ids[int(links_arr[k][0])]
                    v = node_ids[int(links_arr[k][1])]
                    # mesh_map reports cable bandwidth in bits/sec; convert to GB/s (default 10 GB/s if 0)
                    gbs = float(bw_arr[k]) / 8e9 if bw_arr[k] else 10.0
                    for a, b in ((u, v), (v, u)):
                        c_b = pair_agg.setdefault((a, b), [0, 0.0])
                        c_b[0] += 1
                        c_b[1] += gbs
                dlinks = [DirectedLink(u, v, c, r, 1.5) for (u, v), (c, r) in sorted(pair_agg.items())]
                return serial, TopologyGraph(node_ids, dlinks)
        # Fallback to direct reached links if mesh_map symbol is unavailable
        rlist = self.reached()
        node_ids = tuple(sorted({self.node, *(nid for nid, _ in rlist)}))
        dlinks = [DirectedLink(self.node, dst, 1, bw if bw > 0 else 10.0, 1.5) for dst, bw in rlist if dst != self.node]
        return serial, TopologyGraph(node_ids, dlinks)

    def arena_view(self) -> tuple[int, int, torch.Tensor]:
        """Zero-copy 16 KiB-aligned view of the /mesh0 data arena for MetalBufferBorrow / torch.frombuffer."""
        raw = self.ctx.M + self.data_off
        aligned = (raw + 16383) & ~16383
        nbytes = ((self.ctx.M + self.length - aligned) // 16384) * 16384
        buf = (C.c_byte * nbytes).from_address(aligned)
        return aligned, nbytes, torch.frombuffer(buf, dtype=torch.uint8)

    def close(self):
        if self.ctx.M:
            self.lib.mesh_detach(C.byref(self.ctx))
            self.ctx.M = None

    def alloc_contiguous_view(self, nbytes: int, dtype: torch.dtype = torch.float32, *, metal_align: bool = True):
        """Allocate contiguous /mesh0 pages and return (raw_p0, aligned_p0, raw_npages, zero-copy torch Tensor view).
        When metal_align=True, the returned address is 16 KiB-aligned for MTLDevice newBufferWithBytesNoCopy."""
        npages = (nbytes + MESH_PAGE_BYTES - 1) // MESH_PAGE_BYTES
        alloc_pages = ((npages + 3) & ~3) + (4 if metal_align else 0)
        base = C.c_void_p()
        rc = self.lib.mesh_alloc_contiguous(C.byref(self.ctx), alloc_pages, None, C.byref(base))
        if rc:
            raise OSError(rc, f"mesh_alloc_contiguous({alloc_pages}) failed: {rc}")
        raw = base.value
        aligned = (raw + 16383) & ~16383 if metal_align else raw
        raw_p0 = (raw - (self.ctx.M + self.data_off)) // MESH_PAGE_BYTES
        p0 = (aligned - (self.ctx.M + self.data_off)) // MESH_PAGE_BYTES
        buf = (C.c_byte * (npages * MESH_PAGE_BYTES)).from_address(aligned)
        tensor = torch.frombuffer(buf, dtype=dtype, count=nbytes // dtype.itemsize)
        return int(raw_p0), int(p0), int(alloc_pages), tensor

    def alloc_indices(self, count: int) -> list[int]:
        arr = (C.c_uint32 * count)()
        rc = self.lib.mesh_alloc(C.byref(self.ctx), count, 0, arr, None)
        if rc:
            raise OSError(rc, f"mesh_alloc({count}) failed: {rc}")
        return [int(arr[i]) for i in range(count)]

    def release_run(self, p0: int, npages: int):
        for k in range(npages):
            self.lib.mesh_release(C.byref(self.ctx), p0 + k)

    def free_indices(self, indices: list[int]):
        arr = (C.c_uint32 * len(indices))(*indices)
        self.lib.mesh_free_indices(C.byref(self.ctx), arr, len(indices))

    def clear_indices(self, indices: list[int]):
        arr = (C.c_uint32 * len(indices))(*indices)
        self.lib.mesh_clear_indices(C.byref(self.ctx), arr, len(indices))

    def build_reduce_xfers(self, src_p0: int, nbytes: int, dst_node: int, dst_indices: list[int], op: int, tag: int):
        npages = len(dst_indices)
        arr = (MeshRedXfer * npages)()
        for k in range(npages):
            off = k * MESH_PAGE_BYTES
            pb = min(MESH_PAGE_BYTES, nbytes - off)
            arr[k] = MeshRedXfer(src_p0 + k, pb, dst_node, dst_indices[k], op, 0, tag)
        return arr

    def reduce_write(self, xfers, count: int | None = None):
        n = len(xfers) if count is None else count
        off = 0
        while off < n:
            chunk = min(240, n - off)
            ptr = C.cast(C.addressof(xfers) + off * C.sizeof(MeshRedXfer), C.POINTER(MeshRedXfer))
            while True:
                rc = self.lib.mesh_reduce_write(C.byref(self.ctx), ptr, chunk)
                if rc == 0:
                    break
                if rc != 12:  # ENOMEM retries cleanly
                    raise OSError(rc, f"mesh_reduce_write failed: {rc}")
            off += chunk

    def wait_reduced_views(self, indices: list[int], done_tag: int, dtype: torch.dtype = torch.float32) -> list[torch.Tensor]:
        """Spin until all W[idx] reach `done_tag` and return zero-copy Tensor views directly over landed /mesh0 pages."""
        info = MeshEntryInfo()
        tag = C.c_uint16(0)
        views = []
        for idx in indices:
            while True:
                self.lib.mesh_entry_red(C.byref(self.ctx), idx, C.byref(info), C.byref(tag))
                if info.data and int(tag.value) == done_tag:
                    break
            buf = (C.c_byte * info.bytes).from_address(info.data)
            views.append(torch.frombuffer(buf, dtype=dtype, count=info.bytes // dtype.itemsize))
        return views


def compile_flat_reduce_plan(
    declarations,
    topo=None,
    rank_index: int = 0,
    world_size: int = 1,
    *,
    phase_time_s: float = 0.0,
    tiles: int = 1,
    coalesce_group: int | None = None,
):
    """Lower AOT Declarations.reductions into parametric PhaseStrategy FlatBridge descriptors."""
    from .topology import PhaseStrategy, select_phase_strategy, switch_aligned_order
    records = []
    ranks = ()
    max_bytes = 0
    first_op = RED_NONE
    for red in getattr(declarations, "reductions", ()):
        op = red.op
        if first_op == RED_NONE:
            first_op = op
        if not ranks:
            ranks = tuple(red.sources)
        max_bytes = max(max_bytes, red.bytes)
        my_tag = leaf_tag(op, rank_index)
        done = complete_tag(op, len(red.sources))
        records.append({
            "group": red.group,
            "src": red.sources[rank_index] if rank_index < len(red.sources) else red.sources[0],
            "dst": red.destination,
            "op": op,
            "alg": red_alg(op),
            "tag": my_tag,
            "done_tag": done,
            "bytes": red.bytes,
            "shared_value": red.shared_value,
            "block": red.group,
        })
    if topo is None or not records:
        return records
    if coalesce_group is not None:
        strategy = PhaseStrategy(coalesce_group, tiles, first_op)
    elif phase_time_s > 0.0:
        strategy = select_phase_strategy(topo, ranks, max_bytes, phase_time_s, first_op, tiles=tiles)
    else:
        strategy = PhaseStrategy(len(ranks), tiles, first_op)
    ordered = switch_aligned_order(records, topo, ranks=ranks, strategy=strategy)
    for rec in ordered:
        rec["coalesce_group"] = strategy.coalesce_group
        rec["tiles"] = strategy.tiles
    return ordered


class MeshPhaseResult(C.Structure):
    _fields_ = [
        ("wall_us", C.c_double),
        ("gpu_us", C.c_double),
        ("throughput", C.c_double),
        ("ok", C.c_int),
        ("max_err", C.c_double),
    ]


_KERNEL_KIND = {
    "gemm_c0": 0,
    "flash_c1": 1,
    "affine_scan_c2": 2,
    "gemm_blind": 3,
}


def load_libmesh_phase(path: str | pathlib.Path | None = None):
    if path is None:
        path = pathlib.Path(__file__).parents[2] / "rdma" / "libmesh-phase.dylib"
    lib = C.CDLL(str(path))
    lib.mesh_phase_open.argtypes = [C.c_int, C.c_int, C.c_char_p, C.c_int]
    lib.mesh_phase_open.restype = C.c_void_p
    lib.mesh_phase_close.argtypes = [C.c_void_p]
    lib.mesh_phase_close.restype = None
    lib.mesh_phase_barrier.argtypes = [C.c_void_p]
    lib.mesh_phase_barrier.restype = C.c_int
    lib.mesh_phase_node_id.argtypes = [C.c_void_p, C.c_int]
    lib.mesh_phase_node_id.restype = C.c_uint32
    lib.mesh_phase_run.argtypes = [
        C.c_void_p, C.c_int,
        C.c_uint32, C.c_uint32, C.c_uint32,
        C.c_uint32, C.c_uint32, C.c_uint8, C.c_uint16, C.c_uint16,
        C.c_int, C.c_int, C.POINTER(MeshPhaseResult),
    ]
    lib.mesh_phase_run.restype = C.c_int
    if hasattr(lib, "mesh_phase_moe_sparse"):
        lib.mesh_phase_moe_sparse.argtypes = [
            C.c_void_p, C.c_uint32, C.c_uint32, C.c_uint32, C.c_int, C.c_int, C.POINTER(MeshPhaseResult),
        ]
        lib.mesh_phase_moe_sparse.restype = C.c_int
    if hasattr(lib, "mesh_phase_kv_handoff"):
        lib.mesh_phase_kv_handoff.argtypes = [
            C.c_void_p, C.c_uint32, C.c_uint32, C.c_uint32, C.c_int, C.c_int, C.POINTER(MeshPhaseResult),
        ]
        lib.mesh_phase_kv_handoff.restype = C.c_int
    if hasattr(lib, "mesh_phase_run_hetero"):
        lib.mesh_phase_run_hetero.argtypes = [
            C.c_void_p, C.c_uint32, C.c_uint32, C.c_uint32,
            C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32,
            C.c_int, C.POINTER(MeshPhaseResult),
        ]
        lib.mesh_phase_run_hetero.restype = C.c_int
    return lib


class PhaseExecutor:
    """Zero-Python-overhead native executor for CompiledPhaseStep lists (T8).
    Calls into libmesh-phase.dylib once per ParallelCall; all GPU phase dispatches,
    mesh_reduce_write drains, and W[dst_idx] completion waits execute in C/ObjC++."""

    def __init__(self, rank: int, world_size: int, master_ip: str, port: int,
                 phase_lib_path: str | pathlib.Path | None = None):
        self.rank = rank
        self.world_size = world_size
        self.lib = load_libmesh_phase(phase_lib_path)
        self.handle = self.lib.mesh_phase_open(rank, world_size, master_ip.encode("ascii"), int(port))
        if not self.handle:
            raise OSError(f"mesh_phase_open(rank={rank}, world={world_size}, {master_ip}:{port}) failed")
        self.nodes = tuple(int(self.lib.mesh_phase_node_id(self.handle, r)) for r in range(world_size))

    def close(self):
        if self.handle:
            self.lib.mesh_phase_close(self.handle)
            self.handle = None

    def run_step(self, step, *, standalone: bool = False, blind: bool = False, reps: int = 12,
                 coalesce_override: int | None = None) -> MeshPhaseResult:
        kind = 3 if blind else _KERNEL_KIND[step.kernel]
        dims = (*step.shape, 0, 0, 0)[:3]
        g = step.strategy.coalesce_group if coalesce_override is None else int(coalesce_override)
        res = MeshPhaseResult()
        rc = self.lib.mesh_phase_run(
            self.handle, kind,
            int(dims[0]), int(dims[1]), int(dims[2]),
            int(step.strategy.tiles), int(g),
            int(step.op), int(step.my_tag), int(step.done_tag),
            1 if standalone else 0, int(reps), C.byref(res),
        )
        if rc:
            raise OSError(rc, f"mesh_phase_run({step.call_name}, kind={kind}) failed: {rc}")
        return res

    def run_moe_sparse(self, n_experts: int, expert_bytes: int, top_k: int, *,
                       dense: bool = False, reps: int = 12) -> MeshPhaseResult:
        res = MeshPhaseResult()
        rc = self.lib.mesh_phase_moe_sparse(
            self.handle, int(n_experts), int(expert_bytes), int(top_k),
            1 if dense else 0, int(reps), C.byref(res),
        )
        if rc:
            raise OSError(rc, f"mesh_phase_moe_sparse failed: {rc}")
        return res

    def run_kv_handoff(self, n_layers: int, kv_seq: int, kv_dim: int, *,
                       standalone: bool = False, reps: int = 10) -> MeshPhaseResult:
        res = MeshPhaseResult()
        rc = self.lib.mesh_phase_kv_handoff(
            self.handle, int(n_layers), int(kv_seq), int(kv_dim),
            1 if standalone else 0, int(reps), C.byref(res),
        )
        if rc:
            raise OSError(rc, f"mesh_phase_kv_handoff failed: {rc}")
        return res

    def run_hetero_step(self, m: int, k_total: int, n: int, my_share: int, total_shares: int, *,
                        slow_mult: int = 1, tiles: int = 2, coalesce_g: int = 2, reps: int = 10) -> MeshPhaseResult:
        res = MeshPhaseResult()
        rc = self.lib.mesh_phase_run_hetero(
            self.handle, int(m), int(k_total), int(n),
            int(my_share), int(total_shares), int(slow_mult),
            int(tiles), int(coalesce_g), int(reps), C.byref(res),
        )
        if rc:
            raise OSError(rc, f"mesh_phase_run_hetero failed: {rc}")
        return res
