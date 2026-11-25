# Subsumed flat-page algebraic bridge interface (replaces legacy mesh_plan_program_install / mesh_section / mesh_order).
# Exposes the exact planning(), Published, prepare() surface consumed by runtime.py and native.py directly over
# /mesh0 flat page tables, zero-copy MetalBufferBorrow views, and algebraic in-switch reductions (flat_bridge.py).
from dataclasses import dataclass
import ctypes as C
import itertools
import os
import weakref

from torch.export.graph_signature import InputKind

from .flat_bridge import FlatBridge, PhaseExecutor, compile_flat_reduce_plan
from .topology import DirectedLink, PhaseStrategy, TopologyGraph, select_phase_strategy

PUBLISHED = itertools.count(1)


@dataclass(frozen=True)
class FlatPlanning:
    bridge: FlatBridge
    nodes: tuple[int, ...]
    self_index: int
    topology: TopologyGraph
    group: int
    program_mask: int


def planning(lib_path=None) -> FlatPlanning:
    """Inspect live /mesh0 full multi-hop topology (mesh_map) and membership without copying shared state."""
    fb = FlatBridge(lib_path)
    _, topo = fb.live_topology()
    nodes = tuple(topo.nodes)
    group = int(os.environ.get("MESH_GROUP", "0"))
    mask = sum(1 << int(n) for n in os.environ["MESH_PROGRAM"].split(",")) if "MESH_PROGRAM" in os.environ else (
        sum(1 << i for i in range(len(nodes)))
    )
    self_idx = nodes.index(fb.node) if fb.node in nodes else 0
    return FlatPlanning(fb, nodes, self_idx, topo, group, mask)


class _FlatMapping:
    __slots__ = ("bridge", "allocations", "indices", "release", "__weakref__")

    def __init__(self, lib_path=None):
        self.bridge = FlatBridge(lib_path)
        self.allocations: list[tuple[int, int]] = []
        self.indices: list[list[int]] = []
        self.release = weakref.finalize(self, _retired_flat, self.bridge, self.allocations, self.indices)


def _retired_flat(fb: FlatBridge, allocations: list[tuple[int, int]], indices: list[list[int]]):
    for idx_list in indices:
        if idx_list:
            fb.free_indices(idx_list)
    for raw_p0, npages in allocations:
        fb.release_run(raw_p0, npages)
    fb.close()


class Published:
    """Zero-copy /mesh0 publication owner for native.py and runtime.py (subsumes legacy PlanBinding Published)."""

    def __init__(self, source, group: int, mapping: _FlatMapping, *, topo: TopologyGraph | None = None,
                 rank_index: int = 0, world_size: int = 1, phase_time_s: float = 0.0, tiles: int = 1,
                 coalesce_group: int | None = None):
        self.source = source
        self.group = group
        self.mapping = mapping
        self.bridge = mapping.bridge
        self.context = C.addressof(self.bridge.ctx)
        self.node = self.bridge.node
        self.arena_addr, self.arena_bytes, self.arena = self.bridge.arena_view()
        from .schedule import published_program
        self.solution = published_program(self, source)
        self.binding = None
        if self.solution is not None and hasattr(self.solution, "functions"):
            self.function = self.solution.functions
            self.planned = self.solution.planned
            self.base = C.cast(self.solution.offsets.items, C.c_void_p).value
        else:
            self.function = self.planned = self.base = None
        eff_g = coalesce_group if coalesce_group is not None else (
            source.strategy.coalesce_group if getattr(source, "strategy", None) is not None else None
        )
        eff_tiles = source.strategy.tiles if getattr(source, "strategy", None) is not None else tiles
        self.reduce_plan = (
            compile_flat_reduce_plan(
                source, topo=topo, rank_index=rank_index, world_size=world_size,
                phase_time_s=phase_time_s, tiles=eff_tiles, coalesce_group=eff_g,
            )
            if hasattr(source, "reductions") else []
        )

    def bound(self, owned, M):
        from .native import planned_buffer
        return planned_buffer(self, owned, M)

    def alloc_shard(self, nbytes: int, dtype=None):
        import torch
        dt = torch.float32 if dtype is None else dtype
        raw_p0, p0, raw_npages, view = self.bridge.alloc_contiguous_view(nbytes, dt, metal_align=True)
        self.mapping.allocations.append((raw_p0, raw_npages))
        return p0, view

    def alloc_landing(self, npages: int) -> list[int]:
        idx = self.bridge.alloc_indices(npages)
        self.mapping.indices.append(idx)
        return idx

    def reduce_write(self, xfers, count: int | None = None):
        self.bridge.reduce_write(xfers, count)

    def wait_reduced(self, indices: list[int], done_tag: int, dtype=None):
        import torch
        return self.bridge.wait_reduced_views(indices, done_tag, torch.float32 if dtype is None else dtype)


FlatPublished = Published


def prepare(scheduled, entry=None, *, topo: TopologyGraph | None = None, phase_time_s: float = 0.0,
            tiles: int = 1, coalesce_group: int | None = None, lib_path=None):
    """Prepare a scheduled program directly on the live flat-page algebraic bridge (zero legacy PlanBinding)."""
    constants = {
        name: tensor for name, (_, tensor) in scheduled.exported.state.items()
        if scheduled.exported.state_kinds[name] == InputKind.CONSTANT_TENSOR
    }
    if scheduled.plan is None and not getattr(scheduled, "reductions", ()):
        key = scheduled.output_processor if entry is None else entry
        return key, scheduled.functions[key].prepare(None, constants)
    mapping = _FlatMapping(lib_path)
    try:
        node = mapping.bridge.node
        (key,) = (
            [group for group in scheduled.functions if scheduled.machines[group] == node]
            if entry is None else (entry,)
        )
        ordered_groups = list(scheduled.functions)
        rank_index = ordered_groups.index(key)
        group_id = (int(os.environ.get("MESH_GROUP", "0")) << 32) + next(PUBLISHED)
        published = Published(
            scheduled.plan if scheduled.plan is not None else scheduled,
            group_id, mapping, topo=topo, rank_index=rank_index, world_size=len(ordered_groups),
            phase_time_s=phase_time_s, tiles=tiles, coalesce_group=coalesce_group,
        )
        return key, scheduled.functions[key].prepare(published, constants)
    except BaseException:
        mapping.release()
        raise


__all__ = [
    "FlatBridge",
    "FlatPlanning",
    "FlatPublished",
    "PhaseExecutor",
    "PhaseStrategy",
    "Published",
    "planning",
    "prepare",
    "select_phase_strategy",
]
