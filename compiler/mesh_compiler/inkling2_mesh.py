from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .algebra import RED_NONE, RED_SUM_F32
from .parallel_call import CallgraphProgram, CollectiveEdge, ParallelCall, ShardSpec
from .topology import DirectedLink, TopologyGraph


_FP4_E2M1_LUT = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


@dataclass(frozen=True)
class Inkling2MeshSpec:
    name: str
    model_id: str
    model_type: str
    vocab_size: int
    padded_vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    d_rel: int
    rel_extent: int
    local_layer_ids: tuple[int, ...]
    sliding_window_size: int
    sconv_kernel_size: int
    rms_norm_eps: float
    log_scaling_n_floor: int
    log_scaling_alpha: float
    qk_l2_clip_threshold: float
    rel_l2_clip_threshold: float
    n_layers_per_attn_res_block: int
    final_logit_softcapping: float
    n_routed_experts: int
    num_experts_per_tok: int
    intermediate_size: int
    mlp_gate_softcap: float
    mlp_up_softcap: float
    route_scale: float
    mxfp8_layers: tuple[int, ...]
    loop_core_start: int = 0
    loop_core_end: int = 0
    loop_num_passes: int = 1
    loop_gate_rank: int = 64
    weight_gb_est: float = 20.0
    dfly2_visit_ids: tuple[int, ...] = (41,)
    dfly2_tap_visits: tuple[int, ...] = (5, 23, 41, 48)
    dfly2_inter: int = 5632
    dfly2_block_size: int = 8
    dfly2_markov_rank: int = 256

    @property
    def is_looped(self) -> bool:
        return self.loop_num_passes > 1 and self.loop_core_end > self.loop_core_start

    @property
    def visits(self) -> list[tuple[int, int]]:
        if not self.is_looped:
            return [(i, 0) for i in range(self.num_hidden_layers)]
        core = range(self.loop_core_start, self.loop_core_end)
        v = [(i, 0) for i in range(core.start)]
        v += [(i, p) for p in range(self.loop_num_passes) for i in core]
        v += [(i, 0) for i in range(core.stop, self.num_hidden_layers)]
        return v

    @property
    def num_layer_visits(self) -> int:
        return len(self.visits)

    @property
    def global_visit_ids(self) -> list[int]:
        loc = set(self.local_layer_ids)
        return [vi for vi, (lid, _) in enumerate(self.visits) if lid not in loc]

    @classmethod
    def mini(cls) -> Inkling2MeshSpec:
        loc = tuple(i for i in range(24) if (i % 6) != 5)
        return cls(
            name="inkling-2-mini",
            model_id="thinkingmachines/Inkling-2-Mini",
            model_type="inkling2_loop",
            vocab_size=200058,
            padded_vocab_size=201024,
            hidden_size=2304,
            num_hidden_layers=24,
            num_attention_heads=24,
            num_key_value_heads=8,
            head_dim=128,
            d_rel=16,
            rel_extent=1024,
            local_layer_ids=loc,
            sliding_window_size=1024,
            sconv_kernel_size=4,
            rms_norm_eps=1e-6,
            log_scaling_n_floor=8192,
            log_scaling_alpha=0.1,
            qk_l2_clip_threshold=22.627416997969522,
            rel_l2_clip_threshold=8.0,
            n_layers_per_attn_res_block=6,
            final_logit_softcapping=30.0,
            n_routed_experts=128,
            num_experts_per_tok=8,
            intermediate_size=1152,
            mlp_gate_softcap=4.0,
            mlp_up_softcap=25.0,
            route_scale=8.0,
            mxfp8_layers=(20, 21, 22, 23),
            loop_core_start=6,
            loop_core_end=18,
            loop_num_passes=3,
            loop_gate_rank=64,
            weight_gb_est=20.2,
            dfly2_visit_ids=(41,),
            dfly2_tap_visits=(5, 23, 41, 48),
            dfly2_inter=5632,
        )

    @classmethod
    def small(cls) -> Inkling2MeshSpec:
        loc = tuple(i for i in range(48) if (i % 6) != 5)
        return cls(
            name="inkling-2-small",
            model_id="thinkingmachines/Inkling-2-Small",
            model_type="inkling2",
            vocab_size=200058,
            padded_vocab_size=201024,
            hidden_size=4608,
            num_hidden_layers=48,
            num_attention_heads=48,
            num_key_value_heads=8,
            head_dim=128,
            d_rel=16,
            rel_extent=1024,
            local_layer_ids=loc,
            sliding_window_size=1024,
            sconv_kernel_size=4,
            rms_norm_eps=1e-6,
            log_scaling_n_floor=128000,
            log_scaling_alpha=0.1,
            qk_l2_clip_threshold=22.627416997969522,
            rel_l2_clip_threshold=8.0,
            n_layers_per_attn_res_block=6,
            final_logit_softcapping=30.0,
            n_routed_experts=384,
            num_experts_per_tok=8,
            intermediate_size=2304,
            mlp_gate_softcap=4.0,
            mlp_up_softcap=25.0,
            route_scale=8.0,
            mxfp8_layers=tuple(range(40, 48)),
            loop_core_start=0,
            loop_core_end=0,
            loop_num_passes=1,
            loop_gate_rank=0,
            weight_gb_est=393.0,
            dfly2_visit_ids=(41,),
            dfly2_tap_visits=(5, 11, 17, 23, 29, 35, 41, 48),
            dfly2_inter=11264,
        )


@dataclass(frozen=True)
class ClusterNode:
    hostname: str
    chip: str
    mem_gib: float
    rel_speed: float


DEFAULT_MAC_STUDIO_CLUSTER: tuple[ClusterNode, ...] = (
    ClusterNode("m5ult2", "Apple M5 Ultra", 96.0, 1.0),
    ClusterNode("mdots-Mac-Studio", "Apple M5 Ultra", 96.0, 1.0),
    ClusterNode("mdots-Mac-Studio-2", "Apple M5 Ultra", 96.0, 1.0),
    ClusterNode("mdots-Mac-Studio-3", "Apple M5 Ultra", 96.0, 1.0),
    ClusterNode("Mac.localdomain", "Apple M5 Max", 128.0, 0.55),
)


def default_cluster_topology(n_nodes: int = 4, link_gbs: float = 10.0) -> TopologyGraph:
    nodes = tuple(range(n_nodes))
    links: list[DirectedLink] = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                links.append(DirectedLink(i, j, 1, link_gbs, 1.5))
    return TopologyGraph(nodes, links)


def plan_cluster_placement(
    spec: Inkling2MeshSpec,
    nodes: tuple[ClusterNode, ...] = DEFAULT_MAC_STUDIO_CLUSTER,
    kv_budget_gib_per_node: float = 12.0,
) -> dict[str, Any]:
    """Compute exact TP/EP sharding placement across the Mac Studio cluster."""
    usable = [n for n in nodes if n.mem_gib > kv_budget_gib_per_node]
    if spec.weight_gb_est <= 40.0:
        # Mini (~20 GB) fits comfortably on a single M5 Ultra or TP=2/4 across Studios
        selected = usable[: min(4, len(usable))]
        tp = 1 if len(selected) == 0 else min(len(selected), spec.num_key_value_heads)
        if usable and usable[0].mem_gib >= spec.weight_gb_est + kv_budget_gib_per_node:
            # Prefer single-rank zero-crossing for low-latency Mini, or TP=4 for max BW
            selected = usable[:1]
            tp = 1
    else:
        # Inkling-2-Small (~393 GB): needs 5 nodes (4x M5 Ultra 96G + 1x M5 Max 128G = 512G)
        # or 4x M5 Ultra with EP=4 + streaming cold experts if 5th node absent.
        total_mem = 0.0
        selected = []
        for n in usable:
            selected.append(n)
            total_mem += max(0.0, n.mem_gib - kv_budget_gib_per_node)
            if total_mem >= spec.weight_gb_est:
                break
        tp = len(selected)
        while tp > 1 and (spec.num_key_value_heads % tp != 0 or spec.n_routed_experts % tp != 0):
            # Choose largest divisor of kv_heads (8) & experts (384) -> tp=4 on 4x M5 Ultra,
            # with 5th node hosting overflow expert/KV pages via zero-copy /mesh0 RDMA.
            tp -= 1

    ep = max(1, tp)
    experts_per_rank = spec.n_routed_experts // ep
    q_heads_per_rank = spec.num_attention_heads // ep
    kv_heads_per_rank = spec.num_key_value_heads // ep
    shard_gb = spec.weight_gb_est / max(1, len(selected))
    return {
        "model": spec.model_id,
        "weight_gb": spec.weight_gb_est,
        "tp_size": tp,
        "ep_size": ep,
        "cluster_nodes": [
            {
                "hostname": n.hostname,
                "chip": n.chip,
                "mem_gib": n.mem_gib,
                "weight_shard_gb": round(shard_gb, 2),
                "kv_headroom_gib": round(max(0.0, n.mem_gib - shard_gb), 2),
            }
            for n in selected
        ],
        "q_heads_per_rank": q_heads_per_rank,
        "kv_heads_per_rank": kv_heads_per_rank,
        "experts_per_rank": experts_per_rank,
        "dfly2_extra_kv_B_per_tok": 128 * len(spec.dfly2_visit_ids),
        "native_dspark_extra_kv_B_per_tok": (3 if spec.is_looped else 5)
        * 2
        * spec.num_key_value_heads
        * spec.head_dim
        * 2,
    }


def build_inkling2_callgraph(
    spec: Inkling2MeshSpec,
    topo: TopologyGraph | None = None,
    batch_tokens: int = 8,
    include_dfly2_draft: bool = True,
) -> CallgraphProgram:
    """Parametric SPMD CallgraphProgram for one Inkling-2 decoder visit + DFly2 draft.

    Every parallel operator is declared once as a `ParallelCall(shape, in_shard,
    out_shard, CollectiveEdge)` so `lower_for_rank(r)` compiles the exact local
    Metal + `/mesh0` RDMA collective schedule for any topology (ring, star, or
    heterogeneous Mac Studio cluster) without duplicating rank subgraphs.
    """
    if topo is None:
        topo = default_cluster_topology(n_nodes=1 if spec.weight_gb_est < 40.0 else 4)
    world = max(1, len(topo.nodes))
    d = spec.hidden_size
    q_dim = spec.num_attention_heads * spec.head_dim
    kv_dim = spec.num_key_value_heads * spec.head_dim
    r_dim = spec.num_attention_heads * spec.d_rel
    qkvr_out = 2 * q_dim + r_dim + spec.num_attention_heads + 2 * kv_dim
    sh_bytes = batch_tokens * (d // world) * 4

    calls: list[ParallelCall] = [
        ParallelCall(
            name="inkling2_qkvr_col_tp",
            kernel="gemm_col",
            shape=(batch_tokens, d, qkvr_out),
            in_shard=ShardSpec(None),
            out_shard=ShardSpec(1),
            edge=CollectiveEdge("none", RED_NONE),
        ),
        ParallelCall(
            name="inkling2_rel_gqa_local",
            kernel="rel_gqa_sconv",
            shape=(batch_tokens, spec.num_attention_heads, spec.head_dim),
            in_shard=ShardSpec(1),
            out_shard=ShardSpec(1),
            edge=CollectiveEdge("none", RED_NONE),
        ),
        ParallelCall(
            name="inkling2_attn_wo_row_tp",
            kernel="gemm_c0",
            shape=(batch_tokens, q_dim, d),
            in_shard=ShardSpec(1),
            out_shard=ShardSpec(None),
            edge=CollectiveEdge(
                "allreduce" if world > 1 else "none",
                RED_SUM_F32 if world > 1 else RED_NONE,
                reduce_axis=1,
                bytes_per_shard=sh_bytes,
            ),
        ),
        ParallelCall(
            name="inkling2_moe_ep_routed",
            kernel="moe_nvfp4_ep",
            shape=(batch_tokens * spec.num_experts_per_tok, d, spec.intermediate_size),
            in_shard=ShardSpec(None),
            out_shard=ShardSpec(None),
            edge=CollectiveEdge(
                "allreduce" if world > 1 else "none",
                RED_SUM_F32 if world > 1 else RED_NONE,
                reduce_axis=0,
                bytes_per_shard=sh_bytes,
            ),
        ),
    ]
    if include_dfly2_draft:
        b = spec.dfly2_block_size
        d_sh = b * (d // world) * 4
        calls.extend(
            [
                ParallelCall(
                    name="dfly2_kvshare_q_rel_dsa",
                    kernel="dfly2_kvshare_rel_dsa",
                    shape=(b, spec.num_attention_heads, spec.head_dim),
                    in_shard=ShardSpec(None),
                    out_shard=ShardSpec(1),
                    edge=CollectiveEdge("none", RED_NONE),
                ),
                ParallelCall(
                    name="dfly2_kvshare_wo_row_tp",
                    kernel="gemm_c0",
                    shape=(b, q_dim, d),
                    in_shard=ShardSpec(1),
                    out_shard=ShardSpec(None),
                    edge=CollectiveEdge(
                        "allreduce" if world > 1 else "none",
                        RED_SUM_F32 if world > 1 else RED_NONE,
                        reduce_axis=1,
                        bytes_per_shard=d_sh,
                    ),
                ),
                ParallelCall(
                    name="dfly2_swiglu_col_tp",
                    kernel="gemm_col",
                    shape=(b, d, 2 * spec.dfly2_inter),
                    in_shard=ShardSpec(None),
                    out_shard=ShardSpec(1),
                    edge=CollectiveEdge("none", RED_NONE),
                ),
                ParallelCall(
                    name="dfly2_swiglu_row_tp",
                    kernel="gemm_c0",
                    shape=(b, spec.dfly2_inter, d),
                    in_shard=ShardSpec(1),
                    out_shard=ShardSpec(None),
                    edge=CollectiveEdge(
                        "allreduce" if world > 1 else "none",
                        RED_SUM_F32 if world > 1 else RED_NONE,
                        reduce_axis=1,
                        bytes_per_shard=d_sh,
                    ),
                ),
                ParallelCall(
                    name="dfly2_folded_markov_head",
                    kernel="markov_rnn_folded",
                    shape=(b - 1, spec.dfly2_markov_rank, spec.vocab_size),
                    in_shard=ShardSpec(None),
                    out_shard=ShardSpec(1),
                    edge=CollectiveEdge("none", RED_NONE),
                ),
            ]
        )
    return CallgraphProgram(calls, topo)


def _rms_norm(
    x: torch.Tensor, w: torch.Tensor | None, eps: float = 1e-6, out_dtype: torch.dtype | None = None
) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    if w is not None:
        y = y * w.float()
    return y.to(out_dtype or x.dtype)


def _l2_clip(
    x: torch.Tensor, scale: torch.Tensor, thresh: float, eps: float, tau: torch.Tensor | None = None
) -> torch.Tensor:
    xf = x.float() * scale.float().unsqueeze(0)
    norm = (xf.pow(2).sum(dim=-1, keepdim=True) + eps * eps).sqrt()
    y = xf * torch.clamp(thresh / (norm + eps), max=1.0)
    if tau is not None:
        y = y.to(x.dtype).float() * tau.view(-1, 1, 1)
    return y.to(x.dtype)


def _dequant_nvfp4(
    w_u8: torch.Tensor, s_fp8: torch.Tensor, amax: torch.Tensor, denom: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    lut = torch.tensor(_FP4_E2M1_LUT, device=w_u8.device, dtype=torch.float32)
    lo = lut[(w_u8 & 0x0F).long()]
    hi = lut[((w_u8 >> 4) & 0x0F).long()]
    n_out, k_half = w_u8.shape
    w_fp4 = torch.stack((lo, hi), dim=-1).reshape(n_out, k_half * 2)
    outer = (amax.float() / denom.float().clamp_min(1e-12)).reshape(1, 1)
    w_f32 = w_fp4.view(n_out, -1, 16) * (s_fp8.float() * outer).unsqueeze(-1)
    return w_f32.reshape(n_out, k_half * 2).to(dtype)


def _dequant_mxfp8(w_fp8: torch.Tensor, s_u8: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_out, k_in = w_fp8.shape
    scale = torch.pow(2.0, (s_u8.to(torch.int32) - 127).float())
    return (w_fp8.float().view(n_out, -1, 32) * scale.unsqueeze(-1)).reshape(n_out, k_in).to(dtype)


@dataclass
class _VisitState:
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    k_conv: torch.Tensor | None = None
    v_conv: torch.Tensor | None = None
    attn_conv: torch.Tensor | None = None
    mlp_conv: torch.Tensor | None = None


class MetalInkling2Runtime:
    """Metal/MPS + Mesh runtime executing Inkling-2-Mini / Small + DFly2 / DSpark."""

    def __init__(
        self,
        spec: Inkling2MeshSpec,
        model_dir: str | Path | None = None,
        draft_mode: str = "dfly2_kvshare",
        dfly2_ckpt: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.spec = spec
        self.model_dir = Path(model_dir) if model_dir else None
        self.draft_mode = draft_mode
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.placement = plan_cluster_placement(spec)
        self.topo = default_cluster_topology(n_nodes=max(1, self.placement["tp_size"]))
        self.callgraph = build_inkling2_callgraph(
            spec, self.topo, batch_tokens=spec.dfly2_block_size, include_dfly2_draft=True
        )
        self.compiled_steps, _ = self.callgraph.lower_for_rank(0)
        self.loaded_weights = False
        self.target: Any = None
        self.draft: Any = None
        self.proposer: Any = None
        self.dfly2_ckpt = Path(dfly2_ckpt) if dfly2_ckpt else None
        self.stats: dict[str, Any] = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "verify_steps": 0,
            "accepted_tokens": 0,
            "drafted_tokens": 0,
        }

    def ensure_loaded(self) -> None:
        if self.loaded_weights:
            return
        from vllm.models.inkling2_ref import Inkling2Config, Inkling2RefModel
        from vllm.v1.spec_decode.inkling2_kvshare_draft import (
            Inkling2KVShareConfig,
            Inkling2KVShareDraft,
            Inkling2KVShareProposer,
            Inkling2NativeDSparkDraft,
        )

        if self.model_dir is not None and (self.model_dir / "config.json").exists():
            t_cfg = Inkling2Config.from_checkpoint_dir(self.model_dir)
            self.target = Inkling2RefModel(t_cfg).to(self.device).eval()
            self.target.load_from_safetensors(
                self.model_dir,
                device=self.device,
                keep_experts_on_cpu=(self.spec.weight_gb_est > 80.0),
            )
            if self.draft_mode == "dspark" and (self.model_dir / "mtp.safetensors").exists() and t_cfg.mtp_config:
                dspark = Inkling2NativeDSparkDraft(
                    t_cfg.mtp_config,
                    t_cfg.vocab_size,
                    t_cfg.padded_vocab_size,
                    t_cfg.final_logit_softcapping,
                ).to(self.device).eval()
                dspark.load_mtp_safetensors(self.model_dir / "mtp.safetensors", device=self.device)
                self.draft = dspark
                self.proposer = Inkling2KVShareProposer(self.target, dspark)
            elif self.draft_mode == "dfly2_kvshare":
                preset = "mini_1L_pass3_dsa" if self.spec.is_looped else "small_1L_global_dsa"
                d_cfg = Inkling2KVShareConfig.preset(preset)
                draft = Inkling2KVShareDraft(d_cfg).to(self.device).eval()
                if self.dfly2_ckpt is not None and self.dfly2_ckpt.exists():
                    ck = torch.load(self.dfly2_ckpt, map_location=self.device, weights_only=False)
                    draft.load_state_dict(ck["state_dict"])
                else:
                    phys = d_cfg.layer_specs[0].physical_layer_idx
                    draft.layers[0].self_attn.init_from_target_attention(
                        self.target.layers[phys].attn
                    )
                self.draft = draft
                self.proposer = Inkling2KVShareProposer(self.target, draft)
            self.loaded_weights = True
        else:
            raise FileNotFoundError(
                f"Model weights not found for {self.spec.model_id} at {self.model_dir}"
            )

    @torch.no_grad()
    def generate_tokens(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        eos_token_ids: set[int] | None = None,
    ) -> dict[str, Any]:
        self.ensure_loaded()
        self.stats["requests"] += 1
        self.stats["prompt_tokens"] += len(prompt_ids)
        if self.proposer is not None and self.draft_mode != "none":
            res = self.proposer.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                eos_token_ids=eos_token_ids,
            )
            self.stats["completion_tokens"] += len(res["tokens"])
            self.stats["verify_steps"] += res["steps"]
            self.stats["accepted_tokens"] += int(round(res["mean_accept"] * res["steps"]))
            self.stats["drafted_tokens"] += res["steps"] * (self.spec.dfly2_block_size - 1)
            return res

        from vllm.models.inkling2_ref import Inkling2KVCache

        kv_cache = Inkling2KVCache.create(self.target.config.num_layer_visits)
        eos = eos_token_ids or set()
        out = self.target(
            torch.tensor(prompt_ids, device=self.device, dtype=torch.long),
            kv_cache=kv_cache,
            return_logits=True,
        )
        tok = self._sample(out["logits"][-1], temperature)
        gen = [tok]
        while len(gen) < max_new_tokens and tok not in eos:
            out = self.target(
                torch.tensor([tok], device=self.device, dtype=torch.long),
                kv_cache=kv_cache,
                return_logits=True,
            )
            tok = self._sample(out["logits"][-1], temperature)
            gen.append(tok)
        self.stats["completion_tokens"] += len(gen)
        return {"tokens": gen, "mean_accept": 0.0, "steps": len(gen), "mode": "autoregressive"}

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float) -> int:
        if temperature <= 0.0:
            return int(logits.argmax(dim=-1).item())
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def describe(self) -> dict[str, Any]:
        steps = self.stats["verify_steps"]
        return {
            "model_id": self.spec.model_id,
            "model_type": self.spec.model_type,
            "device": str(self.device),
            "draft_mode": self.draft_mode,
            "weights_present": bool(
                self.model_dir is not None and (self.model_dir / "config.json").exists()
            ),
            "loaded": self.loaded_weights,
            "placement": self.placement,
            "callgraph_ops": [s.call_name for s in self.compiled_steps],
            "stats": {
                **self.stats,
                "mean_accept_tau": round(
                    self.stats["accepted_tokens"] / max(steps, 1), 4
                ),
            },
        }
