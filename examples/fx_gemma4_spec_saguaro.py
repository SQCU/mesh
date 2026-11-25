# End-to-end Lossless Speculative Decoding & Saguaro (SSD, Kumar-Dao-May arXiv:2603.03251)
# through torch.export -> compiler/mesh_compiler/fx_metal.py -> rdma/libmesh-graph.dylib.
#
# Uses REAL google/gemma-4-E2B-it BF16 target weights + REAL trained 77M google/gemma-4-E2B-it-assistant
# MTP draft weights (/tmp/mtp-gemma-4-E2B-it-BF16.gguf), sharing the target's KV caches zero-copy!
#
# Demonstrates three modes on live 1-node (TP=1) and 4-node (TP=4) Apple M5 Ultra Studios:
#   1. Baseline (`spec-off`): 1-token Gemma4DecodeFX (1 CB/tok, 0 Python/tok)
#   2. Single-CB Fused MTP Speculative Decode (`Gemma4SpecRoundFX`):
#      [Parallel K+1 Verify] -> [In-GPU Lossless Accept/Commit] -> [Unrolled K-step Assistant Draft]
#      ALL lowered from ONE torch.export FX graph into ONE pre-encoded native Metal CB per round!
#   3. Saguaro / SSD (`TargetVerifyFX` + batched `AssistantSaguaroFX` with zero-copy shared KV):
#      Theorem-12 geometric fan-out F_k, outcome-keyed speculation cache with sent-token exclusion,
#      Def-14 sigma_{F,C} transform, Leviathan lossless accept/reject + calibration, and async overlap!
from __future__ import annotations

import concurrent.futures
import json
import math
import os
import pathlib
import struct
import sys
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "compiler"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from fx_gemma4_e2b import Gemma4DecodeFX, Gemma4LayerFX, _proportional_freqs, find_gemma4_snapshot, load_text_weights
from mesh_compiler.fx_metal import compile_fx_to_metal


def find_assistant_gguf() -> pathlib.Path:
    for cand in (
        os.environ.get("GEMMA4_ASSISTANT_GGUF", ""),
        "/tmp/mtp-gemma-4-E2B-it-BF16.gguf",
        "/Users/mdot/dox/models/gemma-4-e2b-gguf/mtp-gemma-4-E2B-it-BF16.gguf",
    ):
        if cand and pathlib.Path(cand).exists():
            return pathlib.Path(cand)
    raise FileNotFoundError("Could not locate mtp-gemma-4-E2B-it-BF16.gguf")


def load_gguf_tensors(path: pathlib.Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        _, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))

        def rstr() -> str:
            return f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", errors="replace")

        def skip_val(t: int):
            if t in (0, 1, 7):
                f.read(1)
            elif t in (2, 3):
                f.read(2)
            elif t in (4, 5, 6):
                f.read(4)
            elif t in (10, 11, 12):
                f.read(8)
            elif t == 8:
                rstr()
            elif t == 9:
                et, cnt = struct.unpack("<IQ", f.read(12))
                for _ in range(cnt):
                    skip_val(et)

        for _ in range(n_kv):
            rstr()
            skip_val(struct.unpack("<I", f.read(4))[0])
        infos = []
        for _ in range(n_tensors):
            name = rstr()
            ndim = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{ndim}Q", f.read(8 * ndim))
            ttype, off = struct.unpack("<IQ", f.read(12))
            infos.append((name, dims, ttype, off))
        data_start = (f.tell() + 31) & ~31
        for name, dims, ttype, off in infos:
            f.seek(data_start + off)
            shape = tuple(reversed(dims))
            n = math.prod(shape)
            if ttype == 0:
                t = torch.frombuffer(bytearray(f.read(n * 4)), dtype=torch.float32).reshape(shape)
            elif ttype == 30:
                t = torch.frombuffer(bytearray(f.read(n * 2)), dtype=torch.bfloat16).reshape(shape)
            else:
                raise ValueError(f"Unsupported GGUF tensor type {ttype} for {name}")
            tensors[name] = t.contiguous()
    return tensors


def build_real_prompts(snap: pathlib.Path) -> list[tuple[str, list[int]]]:
    tj = json.loads((snap / "tokenizer.json").read_text())
    vocab = tj["model"]["vocab"]

    def enc(pieces: list[str]) -> list[int]:
        ids = [2]
        for s in pieces:
            if s in vocab:
                ids.append(vocab[s])
        return ids

    return [
        ("python_code", enc(["Write", "▁a", "▁Python", "▁function", "▁to", "▁sort", "▁a", "▁list", "▁of", "▁numbers", ":", "\n"])),
        ("math_explain", enc(["Explain", "▁the", "▁Pythagorean", "▁theorem", "▁and", "▁give", "▁a", "▁step", "-", "by", "-", "step", "▁example", ":", "\n"])),
        ("json_struct", enc(["Generate", "▁a", "▁JSON", "▁object", "▁describing", "▁three", "▁planets", "▁in", "▁the", "▁solar", "▁system", ":", "\n"])),
    ]


class Gemma4AssistantCoreFX(nn.Module):
    """Official 4-layer 77M Gemma-4-E2B Assistant MTP drafter (replicated body, vocab-parallel lm_head).
    Reads the target model's shared KV caches (layer 13 sliding + layer 14 full) with zero KV writes."""

    def __init__(
        self,
        aw: dict[str, torch.Tensor],
        target_embed_w: torch.Tensor,
        tp_rank: int,
        tp_size: int,
        max_seq_sliding: int,
        max_seq_full: int,
    ):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.eps = 1e-6
        self.embed_scale = math.sqrt(1536.0)
        self.max_seq_sliding = max_seq_sliding
        self.max_seq_full = max_seq_full

        self.target_embed_w = nn.Parameter(target_embed_w.contiguous(), requires_grad=False)
        self.pre_proj_w = nn.Parameter(aw["nextn.pre_projection.weight"].to(torch.bfloat16).contiguous(), requires_grad=False)
        self.post_proj_w = nn.Parameter(aw["nextn.post_projection.weight"].to(torch.bfloat16).contiguous(), requires_grad=False)
        self.out_norm_w = nn.Parameter(aw["output_norm.weight"].to(torch.bfloat16).contiguous(), requires_grad=False)

        lm_full = aw["token_embd.weight"].to(torch.bfloat16).contiguous()
        vs = lm_full.shape[0] // tp_size
        self.vocab_offset = tp_rank * vs
        self.lm_head_w = nn.Parameter(lm_full[tp_rank * vs : (tp_rank + 1) * vs].contiguous(), requires_grad=False)

        self.register_buffer("swa_freqs", torch.ones(128, dtype=torch.float32))
        self.register_buffer("full_freqs", aw["rope_freqs.weight"].to(torch.float32).contiguous())

        self.attn_norm_w = nn.ParameterList()
        self.q_w = nn.ParameterList()
        self.q_norm_w = nn.ParameterList()
        self.o_w = nn.ParameterList()
        self.post_attn_ln_w = nn.ParameterList()
        self.ffn_norm_w = nn.ParameterList()
        self.gate_w = nn.ParameterList()
        self.up_w = nn.ParameterList()
        self.down_w = nn.ParameterList()
        self.post_ffw_ln_w = nn.ParameterList()
        self.layer_scalar = nn.ParameterList()
        for l in range(4):
            self.attn_norm_w.append(nn.Parameter(aw[f"blk.{l}.attn_norm.weight"].to(torch.bfloat16).contiguous(), requires_grad=False))
            self.q_w.append(nn.Parameter(aw[f"blk.{l}.attn_q.weight"].to(torch.bfloat16).contiguous(), requires_grad=False))
            self.q_norm_w.append(nn.Parameter(aw[f"blk.{l}.attn_q_norm.weight"].to(torch.bfloat16).contiguous(), requires_grad=False))
            self.o_w.append(nn.Parameter(aw[f"blk.{l}.attn_output.weight"].to(torch.bfloat16).contiguous(), requires_grad=False))
            self.post_attn_ln_w.append(nn.Parameter(aw[f"blk.{l}.post_attention_norm.weight"].to(torch.bfloat16).contiguous(), requires_grad=False))
            self.ffn_norm_w.append(nn.Parameter(aw[f"blk.{l}.ffn_norm.weight"].to(torch.bfloat16).contiguous(), requires_grad=False))
            self.gate_w.append(nn.Parameter(aw[f"blk.{l}.ffn_gate.weight"].to(torch.bfloat16).contiguous(), requires_grad=False))
            self.up_w.append(nn.Parameter(aw[f"blk.{l}.ffn_up.weight"].to(torch.bfloat16).contiguous(), requires_grad=False))
            self.down_w.append(nn.Parameter(aw[f"blk.{l}.ffn_down.weight"].to(torch.bfloat16).contiguous(), requires_grad=False))
            self.post_ffw_ln_w.append(nn.Parameter(aw[f"blk.{l}.post_ffw_norm.weight"].to(torch.bfloat16).contiguous(), requires_grad=False))
            self.layer_scalar.append(nn.Parameter(aw[f"blk.{l}.layer_output_scale.weight"].to(torch.bfloat16).contiguous(), requires_grad=False))

    def step(
        self,
        tok: torch.Tensor,
        hidden_1536: torch.Tensor,
        anchor_pos: torch.Tensor,
        kc_swa: torch.Tensor,
        vc_swa: torch.Tensor,
        kc_full: torch.Tensor,
        vc_full: torch.Tensor,
        shared_kv_b0: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        te = F.embedding(tok, self.target_embed_w) * self.embed_scale
        inp = torch.cat([te, hidden_1536], dim=-1)
        h = F.linear(inp, self.pre_proj_w)
        xn = torch.ops.mesh.rms_norm(h, self.attn_norm_w[0], self.eps)
        for l in range(4):
            is_swa = l < 3
            hd = 256 if is_swa else 512
            window = 512 if is_swa else 0
            max_seq = self.max_seq_sliding if is_swa else self.max_seq_full
            kc = kc_swa if is_swa else kc_full
            vc = vc_swa if is_swa else vc_full
            rope_mode = 1 if is_swa else 2
            rope_base = 10000.0 if is_swa else 1000000.0
            freqs = self.swa_freqs if is_swa else self.full_freqs
            q_raw = F.linear(xn, self.q_w[l])
            q = torch.ops.mesh.head_norm_rope_kv(
                q_raw, self.q_norm_w[l], self.q_norm_w[l], anchor_pos, freqs,
                4, 0, hd, rope_mode, rope_base, max_seq, window, self.eps, kc, vc, 1,
            )
            attn_out = torch.ops.mesh.kv_sdpa_decode(
                q, kc, vc, anchor_pos, 4, 1, hd, max_seq, window, 1.0, 1, shared_kv_b0,
            )
            y = F.linear(attn_out, self.o_w[l])
            h = h + torch.ops.mesh.rms_norm(y, self.post_attn_ln_w[l], self.eps)
            hf = torch.ops.mesh.rms_norm(h, self.ffn_norm_w[l], self.eps)
            act = F.gelu(F.linear(hf, self.gate_w[l]), approximate="tanh") * F.linear(hf, self.up_w[l])
            yd = F.linear(act, self.down_w[l])
            h = (h + torch.ops.mesh.rms_norm(yd, self.post_ffw_ln_w[l], self.eps)) * self.layer_scalar[l]
            nxt_w = self.attn_norm_w[l + 1] if l < 3 else self.out_norm_w
            xn = torch.ops.mesh.rms_norm(h, nxt_w, self.eps)
        proj_h = F.linear(xn, self.post_proj_w)
        logits = F.linear(xn, self.lm_head_w)
        return logits, proj_h


class Gemma4SpecRoundFX(nn.Module):
    """Complete Single-CB Fused MTP Speculative Decode Round exported via torch.export.export()!
    Inside ONE MTLCommandBuffer per round (ZERO Python per op, token, or round):
      1. Target verifies K+1 tokens in parallel (`q_len = K+1`) -> `targ_xn[K+1, 1536]`, `targ_tok[K+1]`
      2. In-GPU `spec_verify_commit` accepts longest matching prefix, updates `pos` & `anchor_pos`,
         logs `[n_acc, t_star, d_0..d_{K-1}]` into `spec_stats[round_idx]`, selects `hidden[n_acc]`
      3. Unrolled K steps of real 77M `Gemma4AssistantCoreFX` draft `d'_0..d'_{K-1}` into `draft_tok[1..K]`!"""

    def __init__(
        self,
        cfg: dict,
        tp_rank: int,
        tp_size: int,
        weights: dict[str, torch.Tensor],
        aw: dict[str, torch.Tensor],
        K: int,
        ctx_len: int,
        max_rounds: int = 128,
    ):
        super().__init__()
        self.K = K
        self.Q = K + 1
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.D = int(cfg["hidden_size"])
        self.L = int(cfg["num_hidden_layers"])
        self.D_pli = int(cfg["hidden_size_per_layer_input"])
        self.eps = float(cfg["rms_norm_eps"])
        self.embed_scale = math.sqrt(float(self.D))
        self.ple_embed_scale = math.sqrt(float(self.D_pli))
        self.ple_proj_scale = 1.0 / math.sqrt(float(self.D))
        self.ple_in_scale = 1.0 / math.sqrt(2.0)

        max_seq_sliding = int(cfg["sliding_window"])
        max_seq_full = max(ctx_len + max_rounds * (K + 1) + 64, 512)
        self.max_seq_sliding = max_seq_sliding
        self.max_seq_full = max_seq_full

        self.embed_w = nn.Parameter(weights["model.language_model.embed_tokens.weight"].contiguous(), requires_grad=False)
        self.embed_pli_w = nn.Parameter(weights["model.language_model.embed_tokens_per_layer.weight"].contiguous(), requires_grad=False)
        self.pli_model_proj_w = nn.Parameter(weights["model.language_model.per_layer_model_projection.weight"].contiguous(), requires_grad=False)
        self.pli_proj_norm_w = nn.Parameter(weights["model.language_model.per_layer_projection_norm.weight"].contiguous(), requires_grad=False)
        self.layer0_ln_w = nn.Parameter(weights["model.language_model.layers.0.input_layernorm.weight"].contiguous(), requires_grad=False)

        V = self.embed_w.shape[0]
        vs = V // tp_size
        self.vocab_offset = tp_rank * vs
        self.unembed_w = nn.Parameter(self.embed_w[tp_rank * vs : (tp_rank + 1) * vs].contiguous(), requires_grad=False)

        n_own = self.L - int(cfg["num_kv_shared_layers"])
        self.n_own = n_own
        last_by_type: dict[str, int] = {}
        kv_source: list[int] = []
        for i in range(self.L):
            lt = cfg["layer_types"][i]
            if i < n_own:
                last_by_type[lt] = i
                kv_source.append(i)
            else:
                kv_source.append(last_by_type[lt])
        self.kv_source = kv_source
        self.swa_kv_src = last_by_type["sliding_attention"]
        self.full_kv_src = last_by_type["full_attention"]

        n_kv = int(cfg["num_key_value_heads"])
        for i in range(n_own):
            is_sl = cfg["layer_types"][i] == "sliding_attention"
            hd = int(cfg["head_dim"]) if is_sl else int(cfg["global_head_dim"])
            ms = max_seq_sliding if is_sl else max_seq_full
            self.register_buffer(f"k_cache_{i}", torch.zeros((1, n_kv, ms, hd), dtype=torch.bfloat16))
            self.register_buffer(f"v_cache_{i}", torch.zeros((1, n_kv, ms, hd), dtype=torch.bfloat16))

        self.layers = nn.ModuleList([
            Gemma4LayerFX(i, cfg, tp_rank, tp_size, weights, kv_source[i], max_seq_sliding, max_seq_full)
            for i in range(self.L)
        ])
        self.next_ln_weights = nn.ParameterList([
            nn.Parameter(
                weights[f"model.language_model.layers.{i + 1}.input_layernorm.weight"].contiguous()
                if i + 1 < self.L
                else weights["model.language_model.norm.weight"].contiguous(),
                requires_grad=False,
            )
            for i in range(self.L)
        ])

        self.assistant = Gemma4AssistantCoreFX(aw, self.embed_w, tp_rank, tp_size, max_seq_sliding, max_seq_full)

        # Persistent round state buffers (all updated on-GPU!)
        self.register_buffer("draft_tok", torch.zeros((self.Q,), dtype=torch.int32))
        self.register_buffer("pos", torch.zeros((1,), dtype=torch.int32))
        self.register_buffer("anchor_pos", torch.zeros((1,), dtype=torch.int32))
        self.register_buffer("spec_stats", torch.zeros((max_rounds, K + 2), dtype=torch.int32))

    def _target_layer_seq(self, layer: Gemma4LayerFX, h, xn, pli, pos, kc, vc, next_ln_w):
        qkv = F.linear(xn, layer.qkv_weight)
        h_kv = layer.n_kv_heads if layer.has_kv else 0
        q = torch.ops.mesh.head_norm_rope_kv(
            qkv, layer.q_norm_w, layer.k_norm_w, pos, layer.rope_freqs,
            layer.n_heads, h_kv, layer.head_dim, layer.rope_mode, layer.rope_base,
            layer.max_seq, layer.window, layer.eps, kc, vc, self.Q,
        )
        attn_out = torch.ops.mesh.kv_sdpa_decode(
            q, kc, vc, pos, layer.n_heads, layer.n_kv_heads,
            layer.head_dim, layer.max_seq, layer.window, 1.0, self.Q, False,
        )
        y = F.linear(attn_out, layer.o_weight)
        if layer.tp_size > 1:
            y = torch.ops.mesh.all_reduce_sum(y)
        h = h + torch.ops.mesh.rms_norm(y, layer.post_attn_ln_w, layer.eps)
        hf = torch.ops.mesh.rms_norm(h, layer.pre_ff_ln_w, layer.eps)
        act = F.gelu(F.linear(hf, layer.gate_weight), approximate="tanh") * F.linear(hf, layer.up_weight)
        yd = F.linear(act, layer.down_weight)
        if layer.tp_size > 1:
            yd = torch.ops.mesh.all_reduce_sum(yd)
        h = h + torch.ops.mesh.rms_norm(yd, layer.post_ff_ln_w, layer.eps)
        pg = torch.ops.mesh.ple_gate(h, layer.ple_gate_w, pli, layer.layer_idx)
        pp = F.linear(pg, layer.ple_proj_w)
        h = (h + torch.ops.mesh.rms_norm(pp, layer.post_ple_ln_w, layer.eps)) * layer.layer_scalar
        xn_next = torch.ops.mesh.rms_norm(h, next_ln_w, layer.eps)
        return h, xn_next

    def forward(self, dummy: torch.Tensor) -> torch.Tensor:
        # 1. Target parallel verification of K+1 tokens (draft_tok[0..K])
        h = F.embedding(self.draft_tok, self.embed_w) * self.embed_scale
        pli_emb = F.embedding(self.draft_tok, self.embed_pli_w) * self.ple_embed_scale
        pli_proj = F.linear(h, self.pli_model_proj_w) * self.ple_proj_scale
        pli_proj = torch.ops.mesh.rms_norm(
            pli_proj.reshape(self.Q * self.L, self.D_pli), self.pli_proj_norm_w, self.eps
        ).reshape(self.Q, self.L, self.D_pli)
        pli = (pli_proj + pli_emb.reshape(self.Q, self.L, self.D_pli)) * self.ple_in_scale
        xn = torch.ops.mesh.rms_norm(h, self.layer0_ln_w, self.eps)
        for i, layer in enumerate(self.layers):
            src = self.kv_source[i]
            kc = getattr(self, f"k_cache_{src}")
            vc = getattr(self, f"v_cache_{src}")
            h, xn = self._target_layer_seq(layer, h, xn, pli, self.pos, kc, vc, self.next_ln_weights[i])

        logits = F.linear(xn, self.unembed_w)
        targ_tok = torch.ops.mesh.vocab_argmax(logits, self.draft_tok, self.pos, self.vocab_offset, False)

        # 2. In-GPU lossless verification + pos/anchor update + hidden selection
        cur_tok, cur_hid = torch.ops.mesh.spec_verify_commit(
            targ_tok, self.draft_tok, xn, self.pos, self.anchor_pos, self.spec_stats, self.K
        )

        # 3. Unrolled K-step real 77M Gemma-4 Assistant MTP draft (reads target's live KV caches!)
        kc_swa = getattr(self, f"k_cache_{self.swa_kv_src}")
        vc_swa = getattr(self, f"v_cache_{self.swa_kv_src}")
        kc_full = getattr(self, f"k_cache_{self.full_kv_src}")
        vc_full = getattr(self, f"v_cache_{self.full_kv_src}")
        for step_k in range(self.K):
            a_lg, cur_hid = self.assistant.step(cur_tok, cur_hid, self.anchor_pos, kc_swa, vc_swa, kc_full, vc_full, False)
            cur_tok = torch.ops.mesh.vocab_argmax(a_lg, cur_tok, self.anchor_pos, self.assistant.vocab_offset, False)
            cur_tok = torch.ops.mesh.write_tok_slot(cur_tok, self.draft_tok, step_k + 1)
        return cur_tok


# ---- Saguaro / SSD (Kumar, Dao, May, arXiv:2603.03251) modules lowered via torch.export ----
class Gemma4TargetVerifyFX(nn.Module):
    """Standalone K+1 parallel verifier lowered via torch.export -> libmesh-graph.dylib (for Saguaro SSD)."""

    def __init__(self, cfg: dict, tp_rank: int, tp_size: int, weights: dict[str, torch.Tensor], K: int, max_seq_full: int, return_logits: bool = False):
        super().__init__()
        self.Q = K + 1
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.return_logits = return_logits
        self.D = int(cfg["hidden_size"])
        self.L = int(cfg["num_hidden_layers"])
        self.D_pli = int(cfg["hidden_size_per_layer_input"])
        self.eps = float(cfg["rms_norm_eps"])
        self.embed_scale = math.sqrt(float(self.D))
        self.ple_embed_scale = math.sqrt(float(self.D_pli))
        self.ple_proj_scale = 1.0 / math.sqrt(float(self.D))
        self.ple_in_scale = 1.0 / math.sqrt(2.0)

        max_seq_sliding = int(cfg["sliding_window"])
        self.max_seq_sliding = max_seq_sliding
        self.max_seq_full = max_seq_full

        self.embed_w = nn.Parameter(weights["model.language_model.embed_tokens.weight"].contiguous(), requires_grad=False)
        self.embed_pli_w = nn.Parameter(weights["model.language_model.embed_tokens_per_layer.weight"].contiguous(), requires_grad=False)
        self.pli_model_proj_w = nn.Parameter(weights["model.language_model.per_layer_model_projection.weight"].contiguous(), requires_grad=False)
        self.pli_proj_norm_w = nn.Parameter(weights["model.language_model.per_layer_projection_norm.weight"].contiguous(), requires_grad=False)
        self.layer0_ln_w = nn.Parameter(weights["model.language_model.layers.0.input_layernorm.weight"].contiguous(), requires_grad=False)

        V = self.embed_w.shape[0]
        vs = V // tp_size
        self.vocab_offset = tp_rank * vs
        self.unembed_w = nn.Parameter(self.embed_w[tp_rank * vs : (tp_rank + 1) * vs].contiguous(), requires_grad=False)

        n_own = self.L - int(cfg["num_kv_shared_layers"])
        self.n_own = n_own
        last_by_type: dict[str, int] = {}
        kv_source: list[int] = []
        for i in range(self.L):
            lt = cfg["layer_types"][i]
            if i < n_own:
                last_by_type[lt] = i
                kv_source.append(i)
            else:
                kv_source.append(last_by_type[lt])
        self.kv_source = kv_source
        self.swa_kv_src = last_by_type["sliding_attention"]
        self.full_kv_src = last_by_type["full_attention"]

        n_kv = int(cfg["num_key_value_heads"])
        for i in range(n_own):
            is_sl = cfg["layer_types"][i] == "sliding_attention"
            hd = int(cfg["head_dim"]) if is_sl else int(cfg["global_head_dim"])
            ms = max_seq_sliding if is_sl else max_seq_full
            self.register_buffer(f"k_cache_{i}", torch.zeros((1, n_kv, ms, hd), dtype=torch.bfloat16))
            self.register_buffer(f"v_cache_{i}", torch.zeros((1, n_kv, ms, hd), dtype=torch.bfloat16))

        self.layers = nn.ModuleList([
            Gemma4LayerFX(i, cfg, tp_rank, tp_size, weights, kv_source[i], max_seq_sliding, max_seq_full)
            for i in range(self.L)
        ])
        self.next_ln_weights = nn.ParameterList([
            nn.Parameter(
                weights[f"model.language_model.layers.{i + 1}.input_layernorm.weight"].contiguous()
                if i + 1 < self.L
                else weights["model.language_model.norm.weight"].contiguous(),
                requires_grad=False,
            )
            for i in range(self.L)
        ])

    def forward(self, tok: torch.Tensor, pos: torch.Tensor):
        h = F.embedding(tok, self.embed_w) * self.embed_scale
        pli_emb = F.embedding(tok, self.embed_pli_w) * self.ple_embed_scale
        pli_proj = F.linear(h, self.pli_model_proj_w) * self.ple_proj_scale
        pli_proj = torch.ops.mesh.rms_norm(
            pli_proj.reshape(self.Q * self.L, self.D_pli), self.pli_proj_norm_w, self.eps
        ).reshape(self.Q, self.L, self.D_pli)
        pli = (pli_proj + pli_emb.reshape(self.Q, self.L, self.D_pli)) * self.ple_in_scale
        xn = torch.ops.mesh.rms_norm(h, self.layer0_ln_w, self.eps)
        for i, layer in enumerate(self.layers):
            src = self.kv_source[i]
            kc = getattr(self, f"k_cache_{src}")
            vc = getattr(self, f"v_cache_{src}")
            qkv = F.linear(xn, layer.qkv_weight)
            h_kv = layer.n_kv_heads if layer.has_kv else 0
            q = torch.ops.mesh.head_norm_rope_kv(
                qkv, layer.q_norm_w, layer.k_norm_w, pos, layer.rope_freqs,
                layer.n_heads, h_kv, layer.head_dim, layer.rope_mode, layer.rope_base,
                layer.max_seq, layer.window, layer.eps, kc, vc, self.Q,
            )
            attn_out = torch.ops.mesh.kv_sdpa_decode(
                q, kc, vc, pos, layer.n_heads, layer.n_kv_heads,
                layer.head_dim, layer.max_seq, layer.window, 1.0, self.Q, False,
            )
            y = F.linear(attn_out, layer.o_weight)
            if layer.tp_size > 1:
                y = torch.ops.mesh.all_reduce_sum(y)
            h = h + torch.ops.mesh.rms_norm(y, layer.post_attn_ln_w, layer.eps)
            hf = torch.ops.mesh.rms_norm(h, layer.pre_ff_ln_w, layer.eps)
            act = F.gelu(F.linear(hf, layer.gate_weight), approximate="tanh") * F.linear(hf, layer.up_weight)
            yd = F.linear(act, layer.down_weight)
            if layer.tp_size > 1:
                yd = torch.ops.mesh.all_reduce_sum(yd)
            h = h + torch.ops.mesh.rms_norm(yd, layer.post_ff_ln_w, layer.eps)
            pg = torch.ops.mesh.ple_gate(h, layer.ple_gate_w, pli, layer.layer_idx)
            pp = F.linear(pg, layer.ple_proj_w)
            h = (h + torch.ops.mesh.rms_norm(pp, layer.post_ple_ln_w, layer.eps)) * layer.layer_scalar
            xn = torch.ops.mesh.rms_norm(h, self.next_ln_weights[i], layer.eps)

        logits = F.linear(xn, self.unembed_w)
        targ_tok = torch.ops.mesh.vocab_argmax(logits, tok, pos, self.vocab_offset, False)
        if self.return_logits:
            return targ_tok, xn, logits
        return targ_tok, xn


class Gemma4AssistantBatchStepFX(nn.Module):
    """Batched N-hypothesis single assistant step lowered via torch.export -> libmesh-graph.dylib.
    Reads target's shared KV caches zero-copy via `shared_buffers` and `shared_kv_b0=True`!"""

    def __init__(self, aw, target_embed_w, max_seq_sliding: int, max_seq_full: int):
        super().__init__()
        self.core = Gemma4AssistantCoreFX(aw, target_embed_w, tp_rank=0, tp_size=1, max_seq_sliding=max_seq_sliding, max_seq_full=max_seq_full)
        self.register_buffer("kc_swa", torch.zeros((1, 1, max_seq_sliding, 256), dtype=torch.bfloat16))
        self.register_buffer("vc_swa", torch.zeros((1, 1, max_seq_sliding, 256), dtype=torch.bfloat16))
        self.register_buffer("kc_full", torch.zeros((1, 1, max_seq_full, 512), dtype=torch.bfloat16))
        self.register_buffer("vc_full", torch.zeros((1, 1, max_seq_full, 512), dtype=torch.bfloat16))

    def forward(self, tok: torch.Tensor, hid: torch.Tensor, anchor_pos: torch.Tensor):
        logits, proj_h = self.core.step(tok, hid, anchor_pos, self.kc_swa, self.vc_swa, self.kc_full, self.vc_full, True)
        return logits, proj_h


class Gemma4AssistantFanoutUnrolledFX(nn.Module):
    """Unrolled K-step batched N-hypothesis Saguaro fan-out in ONE pre-encoded native Metal CB!
    Uses on-GPU vocab_argmax -> returns only N*K int32 tokens (~160 bytes) + K+1 hidden states,
    avoiding any 50 MB CPU logit readback so fan-out finishes inside the target verify window!"""

    def __init__(self, aw, target_embed_w, K: int, max_seq_sliding: int, max_seq_full: int):
        super().__init__()
        self.K = K
        self.core = Gemma4AssistantCoreFX(aw, target_embed_w, tp_rank=0, tp_size=1, max_seq_sliding=max_seq_sliding, max_seq_full=max_seq_full)
        self.register_buffer("kc_swa", torch.zeros((1, 1, max_seq_sliding, 256), dtype=torch.bfloat16))
        self.register_buffer("vc_swa", torch.zeros((1, 1, max_seq_sliding, 256), dtype=torch.bfloat16))
        self.register_buffer("kc_full", torch.zeros((1, 1, max_seq_full, 512), dtype=torch.bfloat16))
        self.register_buffer("vc_full", torch.zeros((1, 1, max_seq_full, 512), dtype=torch.bfloat16))

    def forward(self, tok: torch.Tensor, hid: torch.Tensor, anchor_pos: torch.Tensor):
        outs = []
        cur_t = tok
        cur_h = hid
        for _ in range(self.K):
            lg, cur_h = self.core.step(cur_t, cur_h, anchor_pos, self.kc_swa, self.vc_swa, self.kc_full, self.vc_full, True)
            cur_t = torch.ops.mesh.vocab_argmax(lg, cur_t, anchor_pos, 0, False)
            outs.append(cur_t)
            outs.append(cur_h)
        return tuple(outs)


def saguaro_fanout_allocation(budget: int, K: int, a_p: float = 0.75, r: float = 0.5) -> list[int]:
    """Theorem 12 of Kumar, Dao, May (arXiv:2603.03251): geometric fan-out F_k across k=0..K."""
    a = min(max(a_p, 1e-6), 1.0 - 1e-6)
    rr = max(r, 0.0)
    w = [a ** (k / (1.0 + rr)) for k in range(K)]
    w.append((a ** (K / (1.0 + rr))) * ((1.0 - a) ** (-1.0 / (1.0 + rr))))
    f0 = float(budget) / sum(w)
    return [max(1, int(round(f0 * x))) for x in w]


def apply_sigma_temp(logits_f32: torch.Tensor, F_top: int, C: float, temp: float) -> torch.Tensor:
    """Definition 14 of Kumar, Dao, May (arXiv:2603.03251): Saguaro top-F proposal scaling +log(C)."""
    row = logits_f32.clone()
    if temp > 0 and temp != 1.0:
        row = row / temp
    if F_top > 0 and 0.0 < C < 1.0:
        _, top_idx = torch.topk(row, k=F_top, dim=-1)
        row.scatter_add_(-1, top_idx, torch.full_like(top_idx, math.log(C), dtype=row.dtype))
    return row


def prefill_into_caches(
    cfg: dict, weights: dict[str, torch.Tensor], tp_rank: int, tp_size: int, prompt_ids: list[int], max_seq_full: int
) -> tuple[dict[str, torch.Tensor], int, torch.Tensor]:
    """Run exact prompt prefill via compiled single-token Gemma4DecodeFX (1 CB/tok) and return KV caches + first token + final normed hidden."""
    # Use eager on short prompt (~14 tokens) to get exact initial KV caches & final normed hidden state
    m = Gemma4DecodeFX(cfg, tp_rank, tp_size, weights, batch_size=1, ctx_len=max_seq_full - 64, update_inplace=False)
    m.eval()
    # Note: for TP>1, compute prompt tokens identically on TP=1 so all ranks get exact identical prompt hidden/KV!
    m1 = m if tp_size == 1 else Gemma4DecodeFX(cfg, 0, 1, weights, batch_size=1, ctx_len=max_seq_full - 64, update_inplace=False)
    m1.eval()
    with torch.no_grad():
        for step, tid in enumerate(prompt_ids):
            t_t = torch.tensor([tid], dtype=torch.int32)
            p_t = torch.tensor([step], dtype=torch.int32)
            h = F.embedding(t_t, m1.embed_w) * m1.embed_scale
            pe = F.embedding(t_t, m1.embed_pli_w) * m1.ple_embed_scale
            pp = F.linear(h, m1.pli_model_proj_w) * m1.ple_proj_scale
            pp = torch.ops.mesh.rms_norm(pp.reshape(m1.L, m1.D_pli), m1.pli_proj_norm_w, m1.eps).reshape(1, m1.L, m1.D_pli)
            pli = (pp + pe.reshape(1, m1.L, m1.D_pli)) * m1.ple_in_scale
            xn = torch.ops.mesh.rms_norm(h, m1.layer0_ln_w, m1.eps)
            for i, layer in enumerate(m1.layers):
                src = m1.kv_source[i]
                h, xn = layer(h, xn, pli, p_t, getattr(m1, f"k_cache_{src}"), getattr(m1, f"v_cache_{src}"), m1.next_ln_weights[i])
            first_tok = int(F.linear(xn, m1.unembed_w).float().argmax(-1).item())
            last_xn = xn.clone()
    n_own = cfg["num_hidden_layers"] - cfg["num_kv_shared_layers"]
    kv = {}
    for i in range(n_own):
        kv[f"k_cache_{i}"] = getattr(m1, f"k_cache_{i}").clone()
        kv[f"v_cache_{i}"] = getattr(m1, f"v_cache_{i}").clone()
    return kv, first_tok, last_xn


def run_baseline_ar(
    cfg: dict, weights: dict[str, torch.Tensor], rank: int, world: int, comm_id: str, wire: str,
    kv_init: dict[str, torch.Tensor], first_tok: int, prompt_len: int, n_tokens: int
) -> tuple[list[int], float]:
    """Run baseline 1-token autoregressive decode via torch.export -> libmesh-graph.dylib."""
    mod = Gemma4DecodeFX(cfg, rank, world, weights, batch_size=1, ctx_len=prompt_len + n_tokens + 64, update_inplace=True)
    mod.eval()
    for k, v in kv_init.items():
        buf = getattr(mod, k)
        buf[:, :, : v.shape[2], :].copy_(v[:, :, : buf.shape[2], :])
    ex_tok = torch.tensor([first_tok], dtype=torch.int32)
    ex_pos = torch.tensor([prompt_len], dtype=torch.int32)
    exported = torch.export.export(mod, (ex_tok, ex_pos))
    g = compile_fx_to_metal(exported, world_size=world, rank=rank, comm_id=comm_id, wire_mode=wire, max_decode_steps=n_tokens + 4)
    g.state_view("tok").copy_(ex_tok)
    g.state_view("pos").copy_(ex_pos)
    steady_ms, wall_ms, toks = g.run(n_steps=n_tokens, warm_steps=0, out_tok_stride=1)
    g.close()
    return [first_tok] + toks[: n_tokens - 1], steady_ms


def run_single_cb_mtp_spec(
    cfg: dict, weights: dict[str, torch.Tensor], aw: dict[str, torch.Tensor],
    rank: int, world: int, comm_id: str, wire: str,
    kv_init: dict[str, torch.Tensor], first_tok: int, first_hid: torch.Tensor, prompt_len: int,
    K: int, n_rounds: int,
) -> dict:
    """Mode A: Single-CB Fused MTP Speculative Decode (Verify + Accept + K-step Draft in ONE CB/round!)."""
    max_seq_full = max(prompt_len + n_rounds * (K + 1) + 64, 512)
    spec_mod = Gemma4SpecRoundFX(cfg, rank, world, weights, aw, K=K, ctx_len=prompt_len, max_rounds=n_rounds + 4)
    spec_mod.eval()
    for k, v in kv_init.items():
        buf = getattr(spec_mod, k)
        ml = min(buf.shape[2], v.shape[2])
        buf[:, :, :ml, :].copy_(v[:, :, :ml, :])

    # Compute initial K draft tokens from prompt anchor so round 0 starts full!
    with torch.no_grad():
        cur_t = torch.tensor([first_tok], dtype=torch.int32)
        cur_h = first_hid.clone()
        ap = torch.tensor([prompt_len - 1], dtype=torch.int32)
        kc_swa = getattr(spec_mod, f"k_cache_{spec_mod.swa_kv_src}")
        vc_swa = getattr(spec_mod, f"v_cache_{spec_mod.swa_kv_src}")
        kc_full = getattr(spec_mod, f"k_cache_{spec_mod.full_kv_src}")
        vc_full = getattr(spec_mod, f"v_cache_{spec_mod.full_kv_src}")
        # Use full (TP=1) assistant on CPU for initial 1-round seed drafts (identical on all ranks)
        a1 = Gemma4AssistantCoreFX(aw, spec_mod.embed_w, 0, 1, spec_mod.max_seq_sliding, spec_mod.max_seq_full)
        init_draft = [first_tok]
        for _ in range(K):
            lg, cur_h = a1.step(cur_t, cur_h, ap, kc_swa, vc_swa, kc_full, vc_full, False)
            dt = int(lg.float().argmax(-1).item())
            init_draft.append(dt)
            cur_t = torch.tensor([dt], dtype=torch.int32)

    spec_mod.draft_tok.copy_(torch.tensor(init_draft, dtype=torch.int32))
    spec_mod.pos.fill_(prompt_len)
    spec_mod.anchor_pos.fill_(prompt_len - 1)

    t_exp0 = time.perf_counter()
    exported = torch.export.export(spec_mod, (torch.zeros((1,), dtype=torch.int32),))
    t_exp = (time.perf_counter() - t_exp0) * 1e3

    t_comp0 = time.perf_counter()
    g = compile_fx_to_metal(exported, world_size=world, rank=rank, comm_id=comm_id, wire_mode=wire, max_decode_steps=n_rounds + 4)
    t_comp = (time.perf_counter() - t_comp0) * 1e3

    # Run ALL n_rounds in ONE pure C++/Metal call (ZERO Python per op, per token, or per round!)
    steady_round_ms, wall_ms, _ = g.run(n_steps=n_rounds, warm_steps=0, out_tok_stride=0)
    stats = g.state_view("spec_stats")[:n_rounds].clone()

    emitted = [first_tok]
    acc_counts = []
    for r in range(n_rounds):
        n_acc = int(stats[r, 0].item())
        t_star = int(stats[r, 1].item())
        acc_counts.append(n_acc)
        for j in range(n_acc):
            emitted.append(int(stats[r, 2 + j].item()))
        emitted.append(t_star)

    mean_acc = sum(acc_counts) / len(acc_counts)
    toks_per_round = mean_acc + 1.0
    # Compute steady-state tok/s using steady_round_ms (and also wall_ms excluding round 0)
    steady_tok_s = toks_per_round * 1e3 / steady_round_ms
    wall_tok_s = (len(emitted) - 1) * 1e3 / wall_ms

    rec = {
        "mode": "single-cb-mtp-spec",
        "TP": world,
        "K": K,
        "wire": wire,
        "fx_nodes": g.stats.fx_nodes,
        "fused_ops": g.stats.fused_ops,
        "crossings": g.stats.crossings,
        "arena_mb": round(g.stats.arena_bytes / 1e6, 2),
        "export_ms": round(t_exp, 1),
        "compile_ms": round(t_comp, 1),
        "rounds": n_rounds,
        "tokens_emitted": len(emitted),
        "mean_acc": round(mean_acc, 3),
        "toks_per_round": round(toks_per_round, 3),
        "accept_rate": round(mean_acc / K, 3),
        "steady_round_ms": round(steady_round_ms, 3),
        "wall_ms_round": round(wall_ms / n_rounds, 3),
        "steady_tok_s": round(steady_tok_s, 1),
        "wall_tok_s": round(wall_tok_s, 1),
        "emitted": emitted,
    }
    g.close()
    return rec


def run_saguaro_ssd(
    cfg: dict, weights: dict[str, torch.Tensor], aw: dict[str, torch.Tensor],
    rank: int, world: int, comm_id: str, wire: str,
    kv_init: dict[str, torch.Tensor], first_tok: int, first_hid: torch.Tensor, prompt_len: int,
    K: int, budget: int, n_rounds: int, overlap: bool, temp: float = 0.0, sigma_c: float = 1.0,
) -> dict:
    """Mode B: Saguaro / Speculative Speculative Decoding (Kumar-Dao-May arXiv:2603.03251)
    over lowered torch.export -> libmesh-graph.dylib verifier + batched N-hypothesis assistant (zero-copy KV!)."""
    F_alloc = saguaro_fanout_allocation(budget=budget, K=K, a_p=0.75, r=0.5)
    slot_k = []
    for k, fk in enumerate(F_alloc):
        slot_k.extend([k] * fk)
    N = len(slot_k)

    max_seq_full = max(prompt_len + n_rounds * (K + 1) + 64, 512)
    need_logits = (temp > 0.0 and world == 1)
    v_mod = Gemma4TargetVerifyFX(cfg, rank, world, weights, K=K, max_seq_full=max_seq_full, return_logits=need_logits)
    v_mod.eval()
    for k, v in kv_init.items():
        buf = getattr(v_mod, k)
        ml = min(buf.shape[2], v.shape[2])
        buf[:, :, :ml, :].copy_(v[:, :, :ml, :])

    ex_vtok = torch.zeros((K + 1,), dtype=torch.int32)
    ex_vpos = torch.tensor([prompt_len], dtype=torch.int32)
    v_exp = torch.export.export(v_mod, (ex_vtok, ex_vpos))
    vg = compile_fx_to_metal(v_exp, world_size=world, rank=rank, comm_id=comm_id, wire_mode=wire, max_decode_steps=4)

    # Lower batched N-hypothesis Assistant AND M=1 backup Assistant sharing target's LIVE KV MTLBuffers zero-copy!
    shared_kv = {
        "kc_swa": (vg, f"k_cache_{v_mod.swa_kv_src}"),
        "vc_swa": (vg, f"v_cache_{v_mod.swa_kv_src}"),
        "kc_full": (vg, f"k_cache_{v_mod.full_kv_src}"),
        "vc_full": (vg, f"v_cache_{v_mod.full_kv_src}"),
    }
    a_mod = Gemma4AssistantBatchStepFX(aw, v_mod.embed_w, v_mod.max_seq_sliding, v_mod.max_seq_full)
    a_mod.eval()
    ag_b1 = compile_fx_to_metal(
        torch.export.export(a_mod, (torch.zeros((1,), dtype=torch.int32), torch.zeros((1, 1536), dtype=torch.bfloat16), torch.zeros((1,), dtype=torch.int32))),
        world_size=1, rank=0, wire_mode="bf16", shared_buffers=shared_kv,
    )
    if temp == 0.0:
        # Greedy Saguaro: unroll all K steps for all N hypotheses in ONE native Metal CB (160 B readback!)
        a_fan_mod = Gemma4AssistantFanoutUnrolledFX(aw, v_mod.embed_w, K, v_mod.max_seq_sliding, v_mod.max_seq_full)
        a_fan_mod.eval()
        ag_fan = compile_fx_to_metal(
            torch.export.export(a_fan_mod, (torch.zeros((N,), dtype=torch.int32), torch.zeros((N, 1536), dtype=torch.bfloat16), torch.zeros((1,), dtype=torch.int32))),
            world_size=1, rank=0, wire_mode="bf16", shared_buffers=shared_kv,
        )
    else:
        ag_fan = compile_fx_to_metal(
            torch.export.export(a_mod, (torch.zeros((N,), dtype=torch.int32), torch.zeros((N, 1536), dtype=torch.bfloat16), torch.zeros((1,), dtype=torch.int32))),
            world_size=1, rank=0, wire_mode="bf16", shared_buffers=shared_kv,
        )

    max_fk = max(F_alloc) + 1

    def draft_single(anchor_tok: int, anchor_hid: torch.Tensor, anchor_pos: int):
        """Synchronous backup draft (1 chain, K+1 top-F candidates / logit rows & hidden states)."""
        toks = []
        raw_rows = []
        top_cands = []
        hids = [anchor_hid.clone()]
        cur_t = anchor_tok
        cur_h = anchor_hid
        ag_b1.state_view("anchor_pos").fill_(anchor_pos)
        for step_k in range(K + 1):
            ag_b1.state_view("tok").fill_(cur_t)
            ag_b1.state_view("hid").copy_(cur_h)
            ag_b1.run(1, 0)
            lg, nh = ag_b1.output_views()
            if temp == 0.0:
                # Stay in BF16 and extract only top-(max_fk+1) indices (fast, zero 1 MB float32 copy!)
                _, ti = torch.topk(lg[0], k=max_fk + 1, dim=-1)
                top_list = ti.tolist()
                top_cands.append(top_list)
                raw_rows.append(None)
                if step_k < K:
                    nxt = top_list[0]
                    toks.append(nxt)
                    cur_t = nxt
                    cur_h = nh.clone()
                    hids.append(cur_h)
            else:
                lg_f = lg[0].float().clone()
                raw_rows.append(lg_f)
                _, ti = torch.topk(lg_f, k=max_fk + 1, dim=-1)
                top_cands.append(ti.tolist())
                if step_k < K:
                    q_row = apply_sigma_temp(lg_f, F_alloc[step_k], sigma_c, temp)
                    probs = torch.softmax(q_row, dim=-1)
                    nxt = int(torch.multinomial(probs, 1, generator=rng).item())
                    toks.append(nxt)
                    cur_t = nxt
                    cur_h = nh.clone()
                    hids.append(cur_h)
        return toks, raw_rows, hids, top_cands

    def spec_fanout(draft_toks: list[int], draft_rows, draft_hids: list[torch.Tensor], draft_tops: list[list[int]], base_pos: int):
        """Predict N=(sum F_k) plausible outcomes (k, cand_tok) with sent-token exclusion (§4.1.1) and batch-draft K steps!"""
        t0 = time.perf_counter()
        cands = []
        h_in = torch.empty((N, 1536), dtype=torch.bfloat16)
        idx = 0
        for k, fk in enumerate(F_alloc):
            excl = draft_toks[k] if k < K else -1
            picked = [t for t in draft_tops[k] if t != excl][:fk]
            for ct in picked:
                cands.append((k, ct))
                h_in[idx].copy_(draft_hids[k][0])
                idx += 1

        tok_in = torch.tensor([c[1] for c in cands], dtype=torch.int32)
        ag_fan.state_view("anchor_pos").fill_(base_pos)

        if temp == 0.0:
            # ONE single Metal CB for ALL N hypotheses x K steps!
            ag_fan.state_view("tok").copy_(tok_in)
            ag_fan.state_view("hid").copy_(h_in)
            ag_fan.run(1, 0)
            ov = ag_fan.output_views()
            chains = [[] for _ in range(N)]
            hids_out = [[h_in[i : i + 1].clone()] for i in range(N)]
            for step_k in range(K):
                tk = ov[2 * step_k].tolist()
                hk = ov[2 * step_k + 1]
                for i in range(N):
                    chains[i].append(int(tk[i]))
                    hids_out[i].append(hk[i : i + 1].clone())
            fan_ms = (time.perf_counter() - t0) * 1e3
            cache = {}
            for i, key in enumerate(cands):
                # Store chain + hids; tops for next fan-out are computed lazily if needed on hit
                cache[key] = (chains[i], [None] * (K + 1), hids_out[i], None)
            return cache, fan_ms

        chains = [[] for _ in range(N)]
        rows_out = [[] for _ in range(N)]
        tops_out = [[] for _ in range(N)]
        hids_out = [[h_in[i : i + 1].clone()] for i in range(N)]
        for step_k in range(K + 1):
            ag_fan.state_view("tok").copy_(tok_in)
            ag_fan.state_view("hid").copy_(h_in)
            ag_fan.run(1, 0)
            lg, nh = ag_fan.output_views()
            lg_f = lg.float().clone()
            nh_c = nh.clone()
            _, ti_all = torch.topk(lg_f, k=max_fk + 1, dim=-1)
            for i in range(N):
                rows_out[i].append(lg_f[i])
                tops_out[i].append(ti_all[i].tolist())
            if step_k < K:
                nxts = torch.empty((N,), dtype=torch.int64)
                for i in range(N):
                    q_row = apply_sigma_temp(lg_f[i], F_alloc[step_k], sigma_c, temp)
                    nxts[i] = torch.multinomial(torch.softmax(q_row, dim=-1), 1, generator=rng)[0]
                for i in range(N):
                    chains[i].append(int(nxts[i].item()))
                    hids_out[i].append(nh_c[i : i + 1].clone())
                tok_in = nxts.to(torch.int32)
                h_in = nh_c
        fan_ms = (time.perf_counter() - t0) * 1e3
        cache = {}
        for i, key in enumerate(cands):
            cache[key] = (chains[i], rows_out[i], hids_out[i], tops_out[i])
        return cache, fan_ms

    def compute_tops_for_chain(anchor_tok: int, chain_toks: list[int], chain_hids: list[torch.Tensor], base_pos: int) -> list[list[int]]:
        """Fast 1-row top-F extraction for greedy next-round fan-out candidates (used only when hit lacks tops)."""
        tops = []
        ag_b1.state_view("anchor_pos").fill_(base_pos)
        seq = [anchor_tok] + chain_toks
        for k in range(K + 1):
            ag_b1.state_view("tok").fill_(seq[k])
            ag_b1.state_view("hid").copy_(chain_hids[k])
            ag_b1.run(1, 0)
            lg, _ = ag_b1.output_views()
            _, ti = torch.topk(lg[0], k=max_fk + 1, dim=-1)
            tops.append(ti.tolist())
        return tops

    rng = torch.Generator().manual_seed(7)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    cur_pos = prompt_len
    anchor_tok = first_tok
    draft_toks, draft_rows, draft_hids, draft_tops = draft_single(anchor_tok, first_hid, cur_pos - 1)
    emitted = [first_tok]
    hits = 0
    acc_list = []
    verify_ms_list = []
    fan_ms_list = []
    backup_ms_list = []
    round_ms_list = []
    calib_pred_sum = 0.0
    calib_acc_cnt = 0
    calib_total_cnt = 0

    for r in range(n_rounds):
        tr0 = time.perf_counter()
        vtok = torch.tensor([anchor_tok] + draft_toks, dtype=torch.int32)
        vg.state_view("tok").copy_(vtok)
        vg.state_view("pos").fill_(cur_pos)

        def _run_fan():
            tops_use = draft_tops if draft_tops is not None else compute_tops_for_chain(anchor_tok, draft_toks, draft_hids, cur_pos - 1)
            return spec_fanout(draft_toks, draft_rows, draft_hids, tops_use, cur_pos - 1)

        tv0 = time.perf_counter()
        cb = vg.submit_step(0)
        if overlap:
            fut = pool.submit(_run_fan)
            vg.wait_step(cb)
            v_ms = (time.perf_counter() - tv0) * 1e3
            cache, fan_ms = fut.result()
        else:
            vg.wait_step(cb)
            v_ms = (time.perf_counter() - tv0) * 1e3
            cache, fan_ms = _run_fan()

        outs = vg.output_views()
        targ_tok = outs[0].clone()
        targ_xn = outs[1].clone()

        # Leviathan lossless accept/reject
        n_acc = 0
        if temp == 0.0:
            while n_acc < K and int(targ_tok[n_acc].item()) == draft_toks[n_acc]:
                n_acc += 1
            t_star = int(targ_tok[n_acc].item())
        else:
            targ_lg = outs[2].float().clone()
            t_star = -1
            for k in range(K):
                p_probs = torch.softmax(targ_lg[k] / temp, dim=-1)
                q_row = apply_sigma_temp(draft_rows[k], F_alloc[k], sigma_c, temp)
                q_probs = torch.softmax(q_row, dim=-1)
                d = draft_toks[k]
                calib_pred_sum += float(torch.minimum(p_probs, q_probs).sum().item())
                calib_total_cnt += 1
                ratio = float(p_probs[d].item()) / max(float(q_probs[d].item()), 1e-30)
                u = float(torch.rand((), generator=rng).item())
                if u <= min(1.0, ratio):
                    n_acc += 1
                    calib_acc_cnt += 1
                else:
                    res = torch.clamp(p_probs - q_probs, min=0.0)
                    rs = float(res.sum().item())
                    t_star = int(torch.multinomial(res / rs, 1, generator=rng).item()) if rs > 0 else int(torch.multinomial(p_probs, 1, generator=rng).item())
                    break
            if t_star < 0:
                p_last = torch.softmax(targ_lg[K] / temp, dim=-1)
                t_star = int(torch.multinomial(p_last, 1, generator=rng).item())

        for j in range(n_acc):
            emitted.append(draft_toks[j])
        emitted.append(t_star)
        acc_list.append(n_acc)

        new_anchor_pos = cur_pos + n_acc
        cur_pos = new_anchor_pos + 1
        anchor_tok = t_star

        # Probe Saguaro outcome-keyed speculation cache!
        hit_val = cache.get((n_acc, t_star))
        b_ms = 0.0
        if hit_val is not None:
            hits += 1
            draft_toks, draft_rows, draft_hids, draft_tops = hit_val
        else:
            tb0 = time.perf_counter()
            draft_toks, draft_rows, draft_hids, draft_tops = draft_single(t_star, targ_xn[n_acc : n_acc + 1].clone(), new_anchor_pos)
            b_ms = (time.perf_counter() - tb0) * 1e3

        r_ms = (time.perf_counter() - tr0) * 1e3
        verify_ms_list.append(v_ms)
        fan_ms_list.append(fan_ms)
        backup_ms_list.append(b_ms)
        round_ms_list.append(r_ms)

    pool.shutdown()
    ag_fan.close()
    ag_b1.close()
    vg.close()

    # Use steady interior rounds (skip round 0 warmup)
    sl = slice(1, None) if n_rounds >= 4 else slice(None)
    mean_r_ms = sum(round_ms_list[sl]) / len(round_ms_list[sl])
    mean_v_ms = sum(verify_ms_list[sl]) / len(verify_ms_list[sl])
    mean_f_ms = sum(fan_ms_list[sl]) / len(fan_ms_list[sl])
    mean_b_ms = sum(backup_ms_list[sl]) / len(backup_ms_list[sl])
    mean_acc = sum(acc_list) / len(acc_list)
    toks_per_round = mean_acc + 1.0
    rec = {
        "mode": "saguaro-ssd-overlapped" if overlap else "saguaro-ssd-serialized",
        "TP": world,
        "K": K,
        "budget": budget,
        "F_alloc": F_alloc,
        "N_slots": N,
        "temp": temp,
        "sigma_C": sigma_c,
        "rounds": n_rounds,
        "tokens_emitted": len(emitted),
        "p_hit": round(hits / n_rounds, 3),
        "hits": hits,
        "mean_acc": round(mean_acc, 3),
        "toks_per_round": round(toks_per_round, 3),
        "mean_verify_ms": round(mean_v_ms, 3),
        "mean_fanout_ms": round(mean_f_ms, 3),
        "mean_backup_ms": round(mean_b_ms, 3),
        "mean_round_ms": round(mean_r_ms, 3),
        "steady_tok_s": round(toks_per_round * 1e3 / mean_r_ms, 1),
        "emitted": emitted,
    }
    if calib_total_cnt > 0:
        rec["calib_pred_accept"] = round(calib_pred_sum / calib_total_cnt, 4)
        rec["calib_emp_accept"] = round(calib_acc_cnt / calib_total_cnt, 4)
        rec["calib_n"] = calib_total_cnt
    return rec


def main():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    base_port = int(os.environ.get("BASE_PORT", "29760"))
    master = os.environ.get("MASTER_ADDR", "127.0.0.1")
    wire = os.environ.get("MESH_FX_WIRE", "auto")
    K = int(os.environ.get("SPEC_K", "5"))
    budget = int(os.environ.get("SAGUARO_BUDGET", "8"))
    n_rounds = int(os.environ.get("SPEC_ROUNDS", "24"))
    run_saguaro = os.environ.get("RUN_SAGUARO", "1") == "1"

    snap = find_gemma4_snapshot()
    cfg, weights = load_text_weights(snap)
    aw = load_gguf_tensors(find_assistant_gguf())
    prompts = build_real_prompts(snap)

    if rank == 0:
        print(f"Loaded Gemma-4-E2B ({len(weights)} tensors) + Assistant MTP 77M ({len(aw)} tensors), TP={world}, K={K}, budget={budget}", flush=True)

    port = base_port
    for p_idx, (pname, pids) in enumerate(prompts):
        kv_init, first_tok, first_hid = prefill_into_caches(cfg, weights, rank, world, pids, max_seq_full=512)

        # 1. Single-CB Fused MTP Speculative Decode
        port += 1
        spec_rec = run_single_cb_mtp_spec(
            cfg, weights, aw, rank, world, f"{master}:{port}", wire,
            kv_init, first_tok, first_hid, len(pids), K=K, n_rounds=n_rounds,
        )
        n_check = min(len(spec_rec["emitted"]), 64)

        # 2. Baseline spec-off (exact lossless verification over identical token count!)
        port += 1
        base_toks, base_ms = run_baseline_ar(
            cfg, weights, rank, world, f"{master}:{port}", wire,
            kv_init, first_tok, len(pids), n_tokens=n_check,
        )
        exact_spec = (spec_rec["emitted"][:n_check] == base_toks[:n_check])
        base_tok_s = 1e3 / base_ms

        if rank == 0:
            out_s = {k: v for k, v in spec_rec.items() if k != "emitted"}
            out_s["prompt"] = pname
            out_s["baseline_ms"] = round(base_ms, 3)
            out_s["baseline_tok_s"] = round(base_tok_s, 1)
            out_s["speedup_vs_ar"] = round(spec_rec["steady_tok_s"] / base_tok_s, 2)
            out_s["lossless_exact"] = exact_spec
            out_s["checked_toks"] = n_check
            print(json.dumps(out_s), flush=True)

        # 3. Saguaro / SSD (on first prompt for serialized vs overlapped + stochastic calibration on TP=1)
        if run_saguaro and p_idx == 0:
            for ov in (False, True):
                port += 1
                sag = run_saguaro_ssd(
                    cfg, weights, aw, rank, world, f"{master}:{port}", wire,
                    kv_init, first_tok, first_hid, len(pids), K=K, budget=budget,
                    n_rounds=n_rounds, overlap=ov, temp=0.0, sigma_c=1.0,
                )
                nc = min(len(sag["emitted"]), len(base_toks))
                exact_sag = (sag["emitted"][:nc] == base_toks[:nc])
                if rank == 0:
                    out_g = {k: v for k, v in sag.items() if k != "emitted"}
                    out_g["prompt"] = pname
                    out_g["baseline_tok_s"] = round(base_tok_s, 1)
                    out_g["speedup_vs_ar"] = round(sag["steady_tok_s"] / base_tok_s, 2)
                    out_g["lossless_exact"] = exact_sag
                    out_g["checked_toks"] = nc
                    print(json.dumps(out_g), flush=True)

            if world == 1:
                # Stochastic Leviathan calibration at temp=1.0 + Saguaro sigma_{F,C=0.5}
                for sc in (1.0, 0.5):
                    port += 1
                    cal = run_saguaro_ssd(
                        cfg, weights, aw, rank, world, f"{master}:{port}", wire,
                        kv_init, first_tok, first_hid, len(pids), K=K, budget=budget,
                        n_rounds=32, overlap=True, temp=1.0, sigma_c=sc,
                    )
                    if rank == 0:
                        out_c = {k: v for k, v in cal.items() if k != "emitted"}
                        out_c["prompt"] = pname
                        print(json.dumps(out_c), flush=True)


if __name__ == "__main__":
    main()
