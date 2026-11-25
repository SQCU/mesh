# Ticket G1: Pure parametric Sync vs Async Pareto 3D codec router for Mesh Callgraph IR.
# Mirrors rdma/mesh-codec.h (consolidated from metal-microbench/had_int8_metal/pareto_codec_selector.h
# and sm120-chudlets/fabric/python/sm120_act_router.py). Zero dynamic branching on the hot path.
from dataclasses import dataclass
import math

from .algebra import RED_NONE, RED_SUM_F16, RED_SUM_F32, RED_SUM_HAD_I6, RED_SUM_HAD_I8

CODEC_BF16 = 0
CODEC_HAD_INT8 = 1
CODEC_HAD_INT6_PK = 2
CODEC_HAD_INT6_ANS = 3
CODEC_CHSPLIT_ANS = 4
CODEC_MST_MEALY = 5
CODEC_RVQ4_LUT2S = 6
CODEC_RVQ4_B8_3S = 7
CODEC_KALOMAZE_27B_DRAM = 8
CODEC_KALOMAZE_27B_ADC = 9
CODEC_KALOMAZE_27B_G4 = 10

WORKLOAD_ACT = 0
WORKLOAD_KV = 1

DEEP_LAYER_KL_SCALE = 0.189


@dataclass(frozen=True)
class CodecSpec:
    codec_id: int
    name: str
    bpe_inf: float
    hdr_bytes_fixed: float
    enc_gbps_bf16: float
    dec_gbps_bf16: float
    alpha_enc_us: float
    alpha_dec_us: float
    kl_act: float
    kl_kv: float
    requires_aot: bool
    in_switch_op: int | None = None


M5_CODEC_TABLE: tuple[CodecSpec, ...] = (
    CodecSpec(CODEC_BF16, "bf16", 16.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00000, 0.00000, False, RED_SUM_F16),
    CodecSpec(CODEC_HAD_INT8, "had_int8", 8.50, 0.0, 238.3, 262.1, 3.0, 3.0, 0.00103, 0.00083, False, RED_SUM_HAD_I8),
    CodecSpec(CODEC_HAD_INT6_PK, "had_int6_pk", 6.50, 0.0, 117.4, 231.9, 3.0, 3.0, 0.00420, 0.00150, False, RED_SUM_HAD_I6),
    CodecSpec(CODEC_HAD_INT6_ANS, "had_int6_ans", 5.28, 160.0, 118.5, 83.3, 7.0, 7.0, 0.00420, 0.00150, True, None),
    CodecSpec(CODEC_CHSPLIT_ANS, "chsplit_ans", 3.55, 224.0, 83.3, 88.3, 9.0, 9.0, 0.00450, 0.00200, False, None),
    CodecSpec(CODEC_MST_MEALY, "mst_mealy", 3.20, 320.0, 53.0, 33.5, 14.0, 14.0, 0.02500, 0.00669, False, None),
    CodecSpec(CODEC_RVQ4_LUT2S, "rvq4_lut2s", 4.00, 0.0, 2.0, 551.8, 12.0, 5.0, 0.03000, 0.01176, True, None),
    CodecSpec(CODEC_RVQ4_B8_3S, "rvq4_b8_3s", 7.25, 0.0, 1.5, 175.0, 30.0, 16.0, 0.00900, 0.00263, True, None),
    CodecSpec(CODEC_KALOMAZE_27B_DRAM, "kalomaze_27b_dram", 3.75, 0.0, 2.0, 245.0, 14.0, 6.0, 0.02500, 0.00633, True, None),
    CodecSpec(CODEC_KALOMAZE_27B_ADC, "kalomaze_27b_adc", 3.75, 0.0, 2.0, 810.3, 14.0, 6.0, 0.02500, 0.00633, True, None),
    CodecSpec(CODEC_KALOMAZE_27B_G4, "kalomaze_27b_g4", 2.52, 0.0, 2.0, 810.3, 14.0, 6.0, 0.02500, 0.00633, True, None),
)


@dataclass(frozen=True)
class CodecPolicy:
    """Explicit parametric compression policy attached to a CollectiveEdge."""
    mode: str = "sync"               # "sync" | "async"
    workload: int = WORKLOAD_ACT     # WORKLOAD_ACT | WORKLOAD_KV
    max_kl: float = 0.01
    layer_idx: int | None = None
    n_layers: int | None = None
    allow_aot: bool = False
    allow_error_feedback: bool = False
    preserve_bos: bool = True
    require_in_switch_reduce: bool = False
    force_codec: int | None = None


@dataclass(frozen=True)
class CodecRouteDecision:
    codec_id: int
    codec_name: str
    mode: str
    bpe: float
    wire_ratio: float                # wire_bytes / bf16_bytes
    num_chunks: int
    preserve_bos: bool
    predicted_us: float
    wire_op: int                     # resolved in-switch op (e.g. RED_SUM_HAD_I8) or original op


def codec_time_us(
    spec: CodecSpec,
    nbytes_bf16: int,
    link_gbps: float,
    slack_us: float = 0.0,
    pipelined_rx: bool = False,
    rtt_us: float = 0.0,
) -> float:
    bf16_b = float(nbytes_bf16)
    t_enc = (spec.alpha_enc_us + (bf16_b / (spec.enc_gbps_bf16 * 1e9)) * 1e6) if spec.enc_gbps_bf16 > 0.0 else 0.0
    t_dec = (spec.alpha_dec_us + (bf16_b / (spec.dec_gbps_bf16 * 1e9)) * 1e6) if spec.dec_gbps_bf16 > 0.0 else 0.0
    wire_b = spec.hdr_bytes_fixed + bf16_b * (spec.bpe_inf / 16.0)
    link_Bps = max(link_gbps, 1.0) * 1e9 / 8.0
    t_wire = (wire_b / link_Bps) * 1e6
    exposed_enc = max(0.0, t_enc - slack_us)
    if pipelined_rx:
        return rtt_us + exposed_enc + max(t_wire, t_dec)
    return rtt_us + exposed_enc + t_wire + t_dec


def select_codec_route(
    nbytes_bf16: int,
    link_gbps: float,
    policy: CodecPolicy,
    *,
    slack_us: float = 0.0,
    base_op: int = RED_NONE,
    table: tuple[CodecSpec, ...] = M5_CODEC_TABLE,
) -> CodecRouteDecision:
    """Pure O(K) lower-envelope codec selector matching rdma/mesh-codec.h bit-for-bit."""
    if policy.mode not in ("sync", "async"):
        raise ValueError(f"mode must be 'sync' or 'async', got {policy.mode!r}")

    by_id = {c.codec_id: c for c in table}
    link_Bps = max(link_gbps, 1.0) * 1e9 / 8.0
    raw_us = (float(nbytes_bf16) / link_Bps) * 1e6
    early_sensitive = (
        policy.layer_idx is not None
        and policy.n_layers is not None
        and policy.n_layers >= 6
        and policy.layer_idx < (policy.n_layers // 3)
    )

    def _resolve_op(spec: CodecSpec) -> int:
        if base_op == RED_NONE or spec.codec_id == CODEC_BF16:
            return base_op
        if base_op in (RED_SUM_F32, RED_SUM_F16, RED_SUM_HAD_I8, RED_SUM_HAD_I6) and spec.in_switch_op is not None:
            return spec.in_switch_op
        return base_op

    if policy.force_codec is not None:
        spec = by_id[policy.force_codec]
        pipelined = policy.mode == "async"
        chunks = 4 if (pipelined and nbytes_bf16 >= 524288 and spec.codec_id != CODEC_BF16) else 1
        t = codec_time_us(spec, nbytes_bf16, link_gbps, slack_us if pipelined else 0.0, pipelined)
        return CodecRouteDecision(
            spec.codec_id, spec.name, policy.mode, spec.bpe_inf, spec.bpe_inf / 16.0,
            chunks, policy.preserve_bos and spec.codec_id != CODEC_BF16, round(t, 3), _resolve_op(spec),
        )

    if nbytes_bf16 <= 16384:
        return CodecRouteDecision(CODEC_BF16, "bf16", policy.mode, 16.0, 1.0, 1, False, round(raw_us, 3), base_op)

    if policy.workload == WORKLOAD_ACT and early_sensitive and policy.max_kl <= 0.015 and not policy.allow_error_feedback:
        return CodecRouteDecision(CODEC_BF16, "bf16", policy.mode, 16.0, 1.0, 1, False, round(raw_us, 3), base_op)

    pipelined = policy.mode == "async"
    eff_slack = slack_us if pipelined else 0.0
    kl_scale = DEEP_LAYER_KL_SCALE if (policy.workload == WORKLOAD_ACT and policy.layer_idx is not None and not early_sensitive) else 1.0

    best = by_id[CODEC_BF16]
    best_t = math.inf
    for spec in table:
        if policy.require_in_switch_reduce and spec.in_switch_op is None:
            continue
        if spec.requires_aot and not policy.allow_aot:
            continue
        kl = (spec.kl_act if policy.workload == WORKLOAD_ACT else spec.kl_kv) * kl_scale
        if not policy.allow_error_feedback and kl > policy.max_kl:
            continue
        t = codec_time_us(spec, nbytes_bf16, link_gbps, eff_slack, pipelined)
        if t < best_t:
            best_t = t
            best = spec

    chunks = 4 if (pipelined and nbytes_bf16 >= 524288 and best.codec_id != CODEC_BF16) else 1
    return CodecRouteDecision(
        best.codec_id,
        best.name,
        policy.mode,
        best.bpe_inf,
        best.bpe_inf / 16.0,
        chunks,
        policy.preserve_bos and best.codec_id != CODEC_BF16,
        round(best_t, 3),
        _resolve_op(best),
    )
