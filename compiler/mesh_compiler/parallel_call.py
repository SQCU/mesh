# Ticket T6: Parametric SPMD Callgraph IR for TP / CP / SP / SSM parallelism.
# Instead of eagerly inlining P duplicated rank subgraphs and P^2 bespoke transfer chains per strategy,
# every parallel layer is represented ONCE as a ParallelCall(body, shard_spec, edge) over the callgraph IR,
# and lowered parametrically into (local_ir, PhaseStrategy, ReduceDeclaration) for any rank.
from dataclasses import dataclass, field
import math

import torch

from .algebra import (
    ALG_C0_COMM,
    ALG_C1_LIFTED,
    ALG_C2_INTERVAL,
    RED_AFFINE_SCAN_F32,
    RED_NONE,
    RED_ONLINE_SOFTMAX_F32,
    RED_SUM_F32,
    complete_tag,
    leaf_tag,
    red_alg,
)
from .topology import PhaseStrategy, TopologyGraph, graph_locality_order, phase_destination, select_phase_strategy
from .codec import CodecPolicy, CodecRouteDecision, select_codec_route


def _reduce_decl_cls():
    try:
        from .schedule import ReduceDeclaration
        return ReduceDeclaration
    except Exception:
        @dataclass(frozen=True)
        class _ReduceDeclaration:
            group: int
            op: int
            done_tag: int
            sources: tuple
            destination: object
            shared_value: int
            bytes: int
            shape: tuple
            stride: tuple
        return _ReduceDeclaration


@dataclass(frozen=True)
class ShardSpec:
    """Explicit sharding along a single tensor axis with optional R7 heterogeneous share counts."""
    axis: int | None
    shares: tuple[int, ...] | None = None

    def slice_for(self, rank: int, total_extent: int, world_size: int) -> tuple[int, int]:
        if self.axis is None:
            return 0, total_extent
        if self.shares is None:
            base, rem = divmod(total_extent, world_size)
            start = rank * base + min(rank, rem)
            end = start + base + (1 if rank < rem else 0)
            return start, end
        total_w = sum(self.shares)
        lo = sum(self.shares[:rank]) * total_extent // total_w
        hi = sum(self.shares[:rank + 1]) * total_extent // total_w
        return lo, hi


@dataclass(frozen=True)
class CollectiveEdge:
    """Algebraic collective annotation on an outgoing Callgraph edge.
    op == RED_NONE -> no reduction (either local passthrough if kind == 'none', or AllGather/P2P if kind == 'allgather').
    op != RED_NONE -> in-switch algebraic page morphism (C0 SUM/MAX/MIN, C1 ONLINE_SOFTMAX, C2 AFFINE_SCAN)."""
    kind: str                  # 'none' | 'allgather' | 'reduce_scatter' | 'allreduce'
    op: int = RED_NONE
    reduce_axis: int | None = None
    output_shard_axis: int | None = None
    bytes_per_shard: int = 0
    tiles: int = 1
    codec_policy: CodecPolicy | None = None


@dataclass(frozen=True)
class ParallelCall:
    """One compiled layer body + its sharding & algebraic collective edge (never inlined P times)."""
    name: str
    kernel: str                # 'gemm_c0' | 'flash_c1' | 'affine_scan_c2' | 'elementwise' | custom body name
    shape: tuple[int, ...]     # logical global shape (e.g. (M, K, N) for GEMM, (S, D) for Flash, (L, H) for Scan)
    in_shard: ShardSpec
    out_shard: ShardSpec
    edge: CollectiveEdge
    phase_time_s: float = 0.0  # measured or roofline GPU compute time per phase (drives closed-form g*)
    body: object = None        # optional FX GraphModule compiled once for local rank execution


@dataclass(frozen=True)
class CompiledPhaseStep:
    """Flat, zero-branch execution step consumed directly by native libmesh-phase.dylib."""
    call_name: str
    kernel: str
    shape: tuple[int, ...]
    rank: int
    world_size: int
    op: int
    alg: int
    my_tag: int
    done_tag: int
    bytes_per_shard: int
    strategy: PhaseStrategy
    rank_order: tuple          # physical node order (C2 uses graph_locality_order!)
    phase_dests: tuple[int, ...]  # exact destination rank index for each phase p = 0..P-1
    shard_slice: tuple[int, int]
    shares: tuple[int, ...]
    codec_route: CodecRouteDecision | None = None


class CallgraphProgram:
    """Sequence/DAG of ParallelCalls compiled ONCE and projected parametrically onto any rank & topology."""

    def __init__(self, calls: list[ParallelCall], topo: TopologyGraph, ranks=None, *,
                 rank_speeds: list[float] | None = None, map_serial: int = 0):
        self.raw_calls = tuple(calls)
        self.topo = topo
        self.ranks = tuple(topo.nodes if ranks is None else ranks)
        self.rank_speeds = rank_speeds
        self.map_serial = map_serial
        self.calls = tuple(fuse_callgraph_edges(self.raw_calls))

    def refresh_if_changed(self, flat_bridge) -> bool:
        """Diachronic topology reload across calls: re-reads /mesh0 mesh_map ONLY when hdr->map serial changes."""
        cur = flat_bridge.map_serial()
        if cur == self.map_serial and self.map_serial != 0:
            return False
        serial, live_topo = flat_bridge.live_topology()
        self.topo = live_topo
        self.map_serial = serial
        return True

    def rebalance_from_stretches(self, total_units: int, current_shares: tuple[int, ...],
                                 gpu_stretches_us: list[float], *, grain: int = 1) -> tuple[int, ...]:
        """Ticket M6b: Online R3/R14 heterogeneous share update from measured per-rank R2 GPU stretches.
        Uses rdma/allocate.py min_max over empirical per-unit times (excluding collective waits)."""
        import pathlib
        import sys
        rdma_dir = str(pathlib.Path(__file__).parents[2] / "rdma")
        if rdma_dir not in sys.path:
            sys.path.append(rdma_dir)
        import allocate
        rates = [
            (gpu_stretches_us[r] / max(1, current_shares[r]))
            for r in range(len(current_shares))
        ]
        times = [lambda c, rate=rate: rate * c for rate in rates]
        p = len(current_shares)
        new_shares, _ = allocate.min_max(total_units, grain, [grain] * p, [total_units] * p, times)
        return tuple(int(s) for s in new_shares)

    @classmethod
    def tp_mlp_block(cls, m: int, k: int, h: int, n: int, topo: TopologyGraph, ranks=None, *,
                     tiles: int = 1, phase_time_s: float = 0.0, rank_speeds=None) -> "CallgraphProgram":
        """Canonical 2-layer Megatron TP block (Col-TP -> GeLU -> Row-TP C0 ReduceScatter) in 2 ParallelCalls.
        Edge fusion automatically verifies zero collective between Col-TP and Row-TP (in_shard matches out_shard)."""
        sh_bytes = m * (n // max(1, len(topo.nodes if ranks is None else ranks))) * 4
        c1 = ParallelCall(
            name="fc1_col_tp",
            kernel="gemm_col",
            shape=(m, k, h),
            in_shard=ShardSpec(None),
            out_shard=ShardSpec(1),
            edge=CollectiveEdge("none", RED_NONE),
        )
        c2 = ParallelCall(
            name="fc2_row_tp_c0",
            kernel="gemm_c0",
            shape=(m, h, n),
            in_shard=ShardSpec(1),
            out_shard=ShardSpec(1),
            edge=CollectiveEdge("reduce_scatter", RED_SUM_F32, reduce_axis=1, output_shard_axis=1,
                                bytes_per_shard=sh_bytes, tiles=tiles),
            phase_time_s=phase_time_s,
        )
        return cls([c1, c2], topo, ranks, rank_speeds=rank_speeds)

    @classmethod
    def flash_attention_c1(cls, seq_len: int, head_dim: int, topo: TopologyGraph, ranks=None, *,
                           tiles: int = 1, phase_time_s: float = 0.0) -> "CallgraphProgram":
        """Context-Parallel FlashAttention (C1 lifted commutative monoid (m, l, o) -> 128 f32/row wire packed)."""
        p = len(topo.nodes if ranks is None else ranks)
        sh_rows = seq_len // max(1, p)
        # 128 floats per row = 512 B/row -> 8 rows per 4 KiB page (rdma/mesh-reduce.h MESH_RED_ONLINE_SOFTMAX_F32)
        sh_bytes = sh_rows * 128 * 4
        call = ParallelCall(
            name="cp_flash_attn_c1",
            kernel="flash_c1",
            shape=(seq_len, head_dim),
            in_shard=ShardSpec(0),
            out_shard=ShardSpec(0),
            edge=CollectiveEdge("reduce_scatter", RED_ONLINE_SOFTMAX_F32, reduce_axis=0, output_shard_axis=0,
                                bytes_per_shard=sh_bytes, tiles=tiles),
            phase_time_s=phase_time_s,
        )
        return cls([call], topo, ranks)

    @classmethod
    def affine_scan_c2(cls, seq_len: int, hidden: int, topo: TopologyGraph, ranks=None, *,
                       tiles: int = 1, phase_time_s: float = 0.0) -> "CallgraphProgram":
        """Sequence-Parallel Linear Recurrence / SSM (C2 associative non-commutative interval monoid (A, b))."""
        p = len(topo.nodes if ranks is None else ranks)
        # 2 floats (a, b) per channel -> 8 B per channel, sharded across ranks along H (output_shard_axis=1)
        sh_bytes = (hidden // max(1, p)) * 2 * 4
        call = ParallelCall(
            name="sp_affine_scan_c2",
            kernel="affine_scan_c2",
            shape=(seq_len, hidden),
            in_shard=ShardSpec(0),
            out_shard=ShardSpec(1),
            edge=CollectiveEdge("reduce_scatter", RED_AFFINE_SCAN_F32, reduce_axis=0, output_shard_axis=1,
                                bytes_per_shard=sh_bytes, tiles=tiles),
            phase_time_s=phase_time_s,
        )
        return cls([call], topo, ranks)

    def lower_for_rank(self, rank: int) -> tuple[list[CompiledPhaseStep], list]:
        """Project the callgraph parametrically onto `rank`: ZERO graph duplication across strategies."""
        ReduceDecl = _reduce_decl_cls()
        steps: list[CompiledPhaseStep] = []
        decls: list = []
        p = len(self.ranks)
        for gid, call in enumerate(self.calls):
            alg = red_alg(call.edge.op)
            # C2 non-commutative intervals MUST use physical locality ordering (minimizes hop crossings of [lo,hi) merges)
            rank_order = tuple(graph_locality_order(self.topo, self.ranks)) if alg == ALG_C2_INTERVAL else self.ranks
            shares = call.in_shard.shares
            if shares is None:
                if self.rank_speeds is not None and call.edge.bytes_per_shard > 0:
                    from .shares import solve_heterogeneous_shares
                    sh, _ = solve_heterogeneous_shares(
                        p * 8, 1, self.rank_speeds, self.topo, rank_order,
                        call.edge.kind if call.edge.kind != "none" else "allreduce",
                        float(call.edge.bytes_per_shard), in_switch=(call.edge.op != RED_NONE),
                    )
                    shares = tuple(sh)
                else:
                    shares = (1,) * p
            codec_route = None
            eff_op = call.edge.op
            eff_bytes = call.edge.bytes_per_shard
            if call.edge.codec_policy is not None and call.edge.bytes_per_shard > 0:
                bottleneck_gbs = self.topo.bottleneck_rate_gbs(rank_order)
                codec_route = select_codec_route(
                    nbytes_bf16=call.edge.bytes_per_shard,
                    link_gbps=bottleneck_gbs * 8.0,
                    policy=call.edge.codec_policy,
                    slack_us=call.phase_time_s * 1e6,
                    base_op=call.edge.op,
                )
                eff_op = codec_route.wire_op
                eff_bytes = max(1, int(math.ceil(call.edge.bytes_per_shard * codec_route.wire_ratio)))
            strategy = select_phase_strategy(
                self.topo, rank_order, eff_bytes, call.phase_time_s,
                eff_op, tiles=call.edge.tiles,
            )
            phase_dests = tuple(phase_destination(rank, ph, p, strategy.coalesce_group) for ph in range(p))
            my_tag = leaf_tag(eff_op, rank) if eff_op != RED_NONE else 0
            done_tag = complete_tag(eff_op, p) if eff_op != RED_NONE else 0
            extent = call.shape[call.in_shard.axis] if (call.in_shard.axis is not None and call.in_shard.axis < len(call.shape)) else call.shape[0]
            shard_slice = ShardSpec(call.in_shard.axis, shares).slice_for(rank, extent, p)
            steps.append(CompiledPhaseStep(
                call_name=call.name,
                kernel=call.kernel,
                shape=call.shape,
                rank=rank,
                world_size=p,
                op=eff_op,
                alg=alg,
                my_tag=my_tag,
                done_tag=done_tag,
                bytes_per_shard=eff_bytes,
                strategy=strategy,
                rank_order=rank_order,
                phase_dests=phase_dests,
                shard_slice=shard_slice,
                shares=shares,
                codec_route=codec_route,
            ))
            if eff_op != RED_NONE:
                for dst_idx, dst_node in enumerate(rank_order):
                    decls.append(ReduceDecl(
                        group=gid * p + dst_idx,
                        op=eff_op,
                        done_tag=done_tag,
                        sources=rank_order,
                        destination=dst_node,
                        shared_value=gid * p + dst_idx,
                        bytes=eff_bytes,
                        shape=call.shape,
                        stride=(1,),
                    ))
        return steps, decls


def fuse_callgraph_edges(calls: tuple[ParallelCall, ...]) -> list[ParallelCall]:
    """Algebraic callgraph pass: eliminate redundant AllGather/reshard edges whenever producer.out_shard
    already satisfies consumer.in_shard (e.g. Col-TP -> Row-TP contracts along the sharded hidden axis directly)."""
    out = list(calls)
    for i in range(len(out) - 1):
        cur, nxt = out[i], out[i + 1]
        if cur.edge.kind == "allgather" and cur.out_shard == nxt.in_shard and nxt.in_shard.axis is not None:
            out[i] = ParallelCall(
                name=cur.name,
                kernel=cur.kernel,
                shape=cur.shape,
                in_shard=cur.in_shard,
                out_shard=cur.out_shard,
                edge=CollectiveEdge("none", RED_NONE),
                phase_time_s=cur.phase_time_s,
                body=cur.body,
            )
    return out
