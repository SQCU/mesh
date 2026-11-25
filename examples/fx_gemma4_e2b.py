# End-to-end torch.export FX -> Single Pre-Encoded Native Metal Command Buffer Gemma-4-E2B Runner.
# Loads real google/gemma-4-E2B-it BF16 safetensors, builds a standard torch.nn.Module decode graph,
# captures it with torch.export.export(), lowers the FX graph via compiler/mesh_compiler/fx_metal.py
# into libmesh-graph.dylib (1 MTLCommandBuffer/step, 0 Python dispatch per op or per token!),
# verifies exact numerical/token parity against PyTorch eager, and measures TP=1/2/4 decode tempo.
from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "compiler"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from mesh_compiler.fx_metal import compile_fx_to_metal


def find_gemma4_snapshot() -> pathlib.Path:
    env_dir = os.environ.get("GEMMA4_DIR")
    if env_dir:
        return pathlib.Path(env_dir)
    hub = pathlib.Path.home() / ".cache" / "huggingface" / "hub" / "models--google--gemma-4-E2B-it" / "snapshots"
    for snap in sorted(hub.iterdir()):
        if (snap / "model.safetensors").exists():
            return snap
    raise FileNotFoundError(f"Could not find google/gemma-4-E2B-it model.safetensors under {hub}")


def _proportional_freqs(head_dim: int, base: float, factor: float) -> torch.Tensor:
    rope_angles = int(factor * head_dim / 2)
    inv_freq_rotated = 1.0 / (base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / head_dim))
    pad = head_dim // 2 - rope_angles
    if pad > 0:
        inv_freq = torch.cat([inv_freq_rotated, torch.zeros(pad, dtype=torch.float32)])
    else:
        inv_freq = inv_freq_rotated
    # Store 1 / inv_freq (huge sentinel where inv_freq == 0 so 1/freqs == 0 -> no rotation!)
    freqs = torch.where(inv_freq > 0, 1.0 / inv_freq, torch.full_like(inv_freq, 1e30))
    return freqs.contiguous()


class Gemma4LayerFX(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        cfg: dict,
        tp_rank: int,
        tp_size: int,
        weights: dict[str, torch.Tensor],
        kv_source_idx: int,
        max_seq_sliding: int,
        max_seq_full: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = cfg["layer_types"][layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.has_kv = layer_idx < cfg["num_hidden_layers"] - cfg["num_kv_shared_layers"]
        self.kv_source_idx = kv_source_idx
        self.head_dim = cfg["head_dim"] if self.is_sliding else cfg["global_head_dim"]
        self.n_heads = cfg["num_attention_heads"] // tp_size
        self.n_kv_heads = cfg["num_key_value_heads"]
        self.eps = float(cfg["rms_norm_eps"])
        self.window = int(cfg["sliding_window"]) if self.is_sliding else 0
        self.max_seq = max_seq_sliding if self.is_sliding else max_seq_full
        self.tp_size = tp_size

        rp = cfg["rope_parameters"][self.layer_type]
        self.rope_base = float(rp["rope_theta"])
        if rp.get("rope_type") == "proportional":
            self.rope_mode = 2
            self.register_buffer("rope_freqs", _proportional_freqs(self.head_dim, self.rope_base, float(rp["partial_rotary_factor"])))
        else:
            self.rope_mode = 1
            self.register_buffer("rope_freqs", torch.ones(self.head_dim // 2, dtype=torch.float32))

        p = f"model.language_model.layers.{layer_idx}."
        # Shard Q/O heads and MLP gate/up/down across TP ranks
        wq_full = weights[p + "self_attn.q_proj.weight"]
        hq = wq_full.shape[0] // tp_size
        wq = wq_full[tp_rank * hq : (tp_rank + 1) * hq].contiguous()
        wo_full = weights[p + "self_attn.o_proj.weight"]
        wo = wo_full[:, tp_rank * hq : (tp_rank + 1) * hq].contiguous()

        if self.has_kv:
            wk = weights[p + "self_attn.k_proj.weight"].contiguous()
            wv = weights[p + "self_attn.v_proj.weight"].contiguous()
            self.qkv_weight = nn.Parameter(torch.cat([wq, wk, wv], dim=0).contiguous(), requires_grad=False)
            self.k_norm_w = nn.Parameter(weights[p + "self_attn.k_norm.weight"].contiguous(), requires_grad=False)
        else:
            self.qkv_weight = nn.Parameter(wq, requires_grad=False)
            self.k_norm_w = nn.Parameter(weights[p + "self_attn.q_norm.weight"].contiguous(), requires_grad=False)

        self.q_norm_w = nn.Parameter(weights[p + "self_attn.q_norm.weight"].contiguous(), requires_grad=False)
        self.o_weight = nn.Parameter(wo, requires_grad=False)
        self.post_attn_ln_w = nn.Parameter(weights[p + "post_attention_layernorm.weight"].contiguous(), requires_grad=False)
        self.pre_ff_ln_w = nn.Parameter(weights[p + "pre_feedforward_layernorm.weight"].contiguous(), requires_grad=False)

        wg_full = weights[p + "mlp.gate_proj.weight"]
        wu_full = weights[p + "mlp.up_proj.weight"]
        wd_full = weights[p + "mlp.down_proj.weight"]
        im = wg_full.shape[0] // tp_size
        self.gate_weight = nn.Parameter(wg_full[tp_rank * im : (tp_rank + 1) * im].contiguous(), requires_grad=False)
        self.up_weight = nn.Parameter(wu_full[tp_rank * im : (tp_rank + 1) * im].contiguous(), requires_grad=False)
        self.down_weight = nn.Parameter(wd_full[:, tp_rank * im : (tp_rank + 1) * im].contiguous(), requires_grad=False)
        self.post_ff_ln_w = nn.Parameter(weights[p + "post_feedforward_layernorm.weight"].contiguous(), requires_grad=False)

        self.ple_gate_w = nn.Parameter(weights[p + "per_layer_input_gate.weight"].contiguous(), requires_grad=False)
        self.ple_proj_w = nn.Parameter(weights[p + "per_layer_projection.weight"].contiguous(), requires_grad=False)
        self.post_ple_ln_w = nn.Parameter(weights[p + "post_per_layer_input_norm.weight"].contiguous(), requires_grad=False)
        self.layer_scalar = nn.Parameter(weights[p + "layer_scalar"].contiguous(), requires_grad=False)

    def forward(
        self,
        h: torch.Tensor,
        xn: torch.Tensor,
        pli: torch.Tensor,
        pos: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        next_ln_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qkv = F.linear(xn, self.qkv_weight)
        h_kv = self.n_kv_heads if self.has_kv else 0
        q = torch.ops.mesh.head_norm_rope_kv(
            qkv, self.q_norm_w, self.k_norm_w, pos, self.rope_freqs,
            self.n_heads, h_kv, self.head_dim, self.rope_mode, self.rope_base,
            self.max_seq, self.window, self.eps, k_cache, v_cache,
        )
        attn_out = torch.ops.mesh.kv_sdpa_decode(
            q, k_cache, v_cache, pos, self.n_heads, self.n_kv_heads,
            self.head_dim, self.max_seq, self.window, 1.0,
        )
        y = F.linear(attn_out, self.o_weight)
        if self.tp_size > 1:
            y = torch.ops.mesh.all_reduce_sum(y)
        h = h + torch.ops.mesh.rms_norm(y, self.post_attn_ln_w, self.eps)
        hf = torch.ops.mesh.rms_norm(h, self.pre_ff_ln_w, self.eps)

        # Standard PyTorch GeGLU (automatically fused by fx_metal into MG_OP_GATE_UP_ACT!)
        act = F.gelu(F.linear(hf, self.gate_weight), approximate="tanh") * F.linear(hf, self.up_weight)
        yd = F.linear(act, self.down_weight)
        if self.tp_size > 1:
            yd = torch.ops.mesh.all_reduce_sum(yd)
        h = h + torch.ops.mesh.rms_norm(yd, self.post_ff_ln_w, self.eps)

        # Per-layer input gate + projection + fused AddNorm (with next layer's input_layernorm!)
        pg = torch.ops.mesh.ple_gate(h, self.ple_gate_w, pli, self.layer_idx)
        pp = F.linear(pg, self.ple_proj_w)
        h = (h + torch.ops.mesh.rms_norm(pp, self.post_ple_ln_w, self.eps)) * self.layer_scalar
        xn_next = torch.ops.mesh.rms_norm(h, next_ln_w, self.eps)
        return h, xn_next


class Gemma4DecodeFX(nn.Module):
    """Complete 35-layer Gemma-4-E2B autoregressive decode step exported via torch.export.export()."""

    def __init__(
        self,
        cfg: dict,
        tp_rank: int,
        tp_size: int,
        weights: dict[str, torch.Tensor],
        batch_size: int,
        ctx_len: int,
        update_inplace: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.B = batch_size
        self.D = int(cfg["hidden_size"])
        self.L = int(cfg["num_hidden_layers"])
        self.D_pli = int(cfg["hidden_size_per_layer_input"])
        self.eps = float(cfg["rms_norm_eps"])
        self.embed_scale = math.sqrt(float(self.D))
        self.ple_embed_scale = math.sqrt(float(self.D_pli))
        self.ple_proj_scale = 1.0 / math.sqrt(float(self.D))
        self.ple_in_scale = 1.0 / math.sqrt(2.0)
        self.update_inplace = update_inplace

        max_seq_sliding = int(cfg["sliding_window"])
        max_seq_full = max(ctx_len + 64, 512)
        self.max_seq_sliding = max_seq_sliding
        self.max_seq_full = max_seq_full

        self.embed_w = nn.Parameter(weights["model.language_model.embed_tokens.weight"].contiguous(), requires_grad=False)
        self.embed_pli_w = nn.Parameter(weights["model.language_model.embed_tokens_per_layer.weight"].contiguous(), requires_grad=False)
        self.pli_model_proj_w = nn.Parameter(weights["model.language_model.per_layer_model_projection.weight"].contiguous(), requires_grad=False)
        self.pli_proj_norm_w = nn.Parameter(weights["model.language_model.per_layer_projection_norm.weight"].contiguous(), requires_grad=False)
        self.layer0_ln_w = nn.Parameter(weights["model.language_model.layers.0.input_layernorm.weight"].contiguous(), requires_grad=False)
        self.final_ln_w = nn.Parameter(weights["model.language_model.norm.weight"].contiguous(), requires_grad=False)

        # Vocabulary-parallel unembed shard (tied with embed_tokens)
        V = self.embed_w.shape[0]
        vs = V // tp_size
        self.vocab_offset = tp_rank * vs
        self.unembed_w = nn.Parameter(self.embed_w[tp_rank * vs : (tp_rank + 1) * vs].contiguous(), requires_grad=False)

        # Resolve KV sharing: layers 0..(L - shared - 1) own KV; remaining layers share last matching type
        n_own = self.L - int(cfg["num_kv_shared_layers"])
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

        # Persistent KV cache buffers owned by layers 0..n_own-1
        n_kv = int(cfg["num_key_value_heads"])
        for i in range(n_own):
            is_sl = cfg["layer_types"][i] == "sliding_attention"
            hd = int(cfg["head_dim"]) if is_sl else int(cfg["global_head_dim"])
            ms = max_seq_sliding if is_sl else max_seq_full
            self.register_buffer(f"k_cache_{i}", torch.zeros((batch_size, n_kv, ms, hd), dtype=torch.bfloat16))
            self.register_buffer(f"v_cache_{i}", torch.zeros((batch_size, n_kv, ms, hd), dtype=torch.bfloat16))

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

    def forward(self, tok: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        h = F.embedding(tok, self.embed_w) * self.embed_scale
        pli_emb = F.embedding(tok, self.embed_pli_w) * self.ple_embed_scale
        pli_proj = F.linear(h, self.pli_model_proj_w) * self.ple_proj_scale
        pli_proj = torch.ops.mesh.rms_norm(
            pli_proj.reshape(self.B * self.L, self.D_pli), self.pli_proj_norm_w, self.eps
        ).reshape(self.B, self.L, self.D_pli)
        pli = (pli_proj + pli_emb.reshape(self.B, self.L, self.D_pli)) * self.ple_in_scale

        xn = torch.ops.mesh.rms_norm(h, self.layer0_ln_w, self.eps)
        for i, layer in enumerate(self.layers):
            src = self.kv_source[i]
            kc = getattr(self, f"k_cache_{src}")
            vc = getattr(self, f"v_cache_{src}")
            h, xn = layer(h, xn, pli, pos, kc, vc, self.next_ln_weights[i])

        logits = F.linear(xn, self.unembed_w)
        next_tok = torch.ops.mesh.vocab_argmax(logits, tok, pos, self.vocab_offset, self.update_inplace)
        return next_tok


def load_text_weights(snap: pathlib.Path) -> tuple[dict, dict[str, torch.Tensor]]:
    cfg = json.loads((snap / "config.json").read_text())["text_config"]
    weights: dict[str, torch.Tensor] = {}
    with safe_open(str(snap / "model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("model.language_model."):
                weights[k] = f.get_tensor(k)
    return cfg, weights


def main():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    comm_id = os.environ.get("NCCL_COMM_ID", "127.0.0.1:29700")
    wire = os.environ.get("MESH_FX_WIRE", "auto")
    bs_list = [int(x) for x in os.environ.get("BS", "1,2,4,8,16,32,64,128").split(",")]
    ctx_len = int(os.environ.get("CTX", "128"))
    n_steps = int(os.environ.get("STEPS", "24"))
    warm_steps = int(os.environ.get("WARM", "3"))
    check_eager = os.environ.get("CHECK_EAGER", "0") == "1"

    snap = find_gemma4_snapshot()
    t_load0 = time.perf_counter()
    cfg, weights = load_text_weights(snap)
    t_load = (time.perf_counter() - t_load0) * 1e3
    if rank == 0:
        print(f"Loaded {len(weights)} Gemma-4-E2B BF16 tensors in {t_load:.0f} ms (world={world}, wire={wire}, ctx={ctx_len})", flush=True)

    for B in bs_list:
        t0 = time.perf_counter()
        mod = Gemma4DecodeFX(cfg, rank, world, weights, batch_size=B, ctx_len=ctx_len, update_inplace=True)
        mod.eval()

        # Fill KV caches deterministically (same seed across ranks)
        gen = torch.Generator().manual_seed(1234)
        n_own = cfg["num_hidden_layers"] - cfg["num_kv_shared_layers"]
        for i in range(n_own):
            kc = getattr(mod, f"k_cache_{i}")
            vc = getattr(mod, f"v_cache_{i}")
            fill_n = min(ctx_len, kc.shape[2])
            v = (torch.randn((B, kc.shape[1], fill_n, kc.shape[3]), generator=gen) * 0.5).to(torch.bfloat16)
            kc[:, :, :fill_n, :].copy_(v)
            vc[:, :, :fill_n, :].copy_(v)

        ex_tok = torch.tensor([5 + b for b in range(B)], dtype=torch.int32)
        ex_pos = torch.tensor([ctx_len], dtype=torch.int32)

        # Optional 1-step PyTorch eager parity check (TP=1)
        eager_tok = None
        if check_eager and world == 1:
            t_in = ex_tok.clone()
            p_in = ex_pos.clone()
            with torch.no_grad():
                eager_tok = mod(t_in, p_in).tolist()
            # Restore KV slot overwritten by eager check
            for i in range(n_own):
                kc = getattr(mod, f"k_cache_{i}")
                vc = getattr(mod, f"v_cache_{i}")
                slot = (ctx_len % kc.shape[2]) if cfg["layer_types"][i] == "sliding_attention" else ctx_len
                kc[:, :, slot, :].zero_()
                vc[:, :, slot, :].zero_()

        # 1. Export via torch.export.export()!
        t_exp0 = time.perf_counter()
        exported = torch.export.export(mod, (ex_tok, ex_pos))
        t_exp = (time.perf_counter() - t_exp0) * 1e3

        # 2. Lower FX Graph -> ONE single pre-encoded native Metal CB via libmesh-graph.dylib!
        t_comp0 = time.perf_counter()
        compiled = compile_fx_to_metal(
            exported,
            world_size=world,
            rank=rank,
            comm_id=comm_id,
            wire_mode=wire,
            max_decode_steps=max(n_steps, 4),
        )
        t_comp = (time.perf_counter() - t_comp0) * 1e3

        # Initialize runtime input state views (zero-copy!)
        tok_view = compiled.state_view("tok")
        pos_view = compiled.state_view("pos")
        tok_view.copy_(ex_tok)
        pos_view.copy_(ex_pos)

        # Optional exact 1-step check before warmup advances pos
        first_tok = None
        if check_eager and world == 1:
            _, _, toks1 = compiled.run(n_steps=1, warm_steps=0, out_tok_stride=B)
            first_tok = toks1[:B]
            tok_view.copy_(ex_tok)
            pos_view.copy_(ex_pos)

        # 3. Run multi-token autoregressive decode in pure C++/Metal (ZERO Python dispatch per token!)
        steady_ms, wall_ms, gen_toks = compiled.run(n_steps=n_steps, warm_steps=warm_steps, out_tok_stride=B)
        ms_step = wall_ms / n_steps
        a_tok_s = B * 1e3 / steady_ms
        b_tok_s = 1e3 / steady_ms

        rec = {
            "engine": "torch.export->libmesh-graph",
            "TP": world,
            "B": B,
            "L": ctx_len,
            "wire": wire,
            "fx_nodes": compiled.stats.fx_nodes,
            "fused_ops": compiled.stats.fused_ops,
            "crossings": compiled.stats.crossings,
            "arena_mb": round(compiled.stats.arena_bytes / 1e6, 2),
            "export_ms": round(t_exp, 1),
            "compile_ms": round(t_comp, 1),
            "steady_ms": round(steady_ms, 3),
            "wall_ms_step": round(ms_step, 3),
            "b_tok_s": round(b_tok_s, 1),
            "a_tok_s": round(a_tok_s, 1),
            "toks_head": gen_toks[: min(len(gen_toks), B * 4)],
        }
        if eager_tok is not None:
            rec["eager_tok"] = eager_tok
            rec["metal_tok"] = first_tok
            rec["exact_match"] = (eager_tok == first_tok)
        if rank == 0:
            print(json.dumps(rec), flush=True)

        compiled.close()


if __name__ == "__main__":
    main()
