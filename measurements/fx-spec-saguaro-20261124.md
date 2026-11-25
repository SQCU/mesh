# Single-CB Fused MTP Speculative Decoding & Saguaro (SSD) Through `torch.export → libmesh-graph.dylib`

**Date:** 2026-11-24
**Branch / Commit:** merged onto `origin/main` from `fx-spec-saguaro`
**Hardware:** 1× and 4× Apple M5 Ultra Mac Studios (`m5ult2`, `m5ult1`, `studio3`, `studio2`) + Mac Mini (`mini`) RDMA relay (`mesh-flow` `bc5ffc3f5`).
**Target Model:** Real `google/gemma-4-E2B-it` BF16 (`600` tensors, 35 layers).
**Draft / Speculator Model:** Real trained **77M `google/gemma-4-E2B-it-assistant`** BF16 MTP (`mtp-gemma-4-E2B-it-BF16.gguf`, 4 layers, `d=256`, `ffn=2048`, 4 Q heads, 1 KV head, zero-copy shared target KV caches `k_cache_13`/`v_cache_13` & `k_cache_14`/`v_cache_14`, replicated 18 MB body + vocab-parallel `262,144 × 256` head).

## 1. Architecture Lowered Through `torch.export` → `libmesh-graph.dylib`

Two speculative decoding engines compiled through the exact same `compile_fx_to_metal()` compiler ([fx_metal.py](../compiler/mesh_compiler/fx_metal.py)) and native Metal 4 engine ([mesh-graph.mm](../rdma/mesh-graph.mm)):

1. **Mode A — Single-CB Fused MTP Speculative Decoding (`Gemma4SpecRoundFX`):**
   - Captures the **entire speculative round** in ONE `torch.export.export()` FX graph (`1,806` FX nodes at `TP=4, K=5` → **`635` fused native Metal ops + `76` flat-DMA crossings**, **`1.13 MB` arena**):
     1. Parallel `K+1`-token target verification (`q_len = K+1` causal `mg_head_norm_rope_kv` + `mg_kv_sdpa_decode`, `gemm_rows_wide` reading target weights once for all `K+1` tokens).
     2. In-GPU lossless verification & commit (`MG_OP_SPEC_VERIFY_COMMIT`): computes `n_acc`, bonus token `t*`, logs `[n_acc, t*, d_0..d_{K-1}]`, advances `pos` & `anchor_pos`, and selects target's freshly verified `xn[n_acc]`.
     3. Unrolled `K`-step real 77M `Gemma4AssistantCoreFX` draft directly inside the **same `MTLCommandBuffer`**, writing `d'_0..d'_{K-1}` into `draft_tok[1..K]` (`MG_OP_WRITE_TOK_SLOT`).
   - **Zero Python dispatch per op, per token, or per round**; pipelined 1 round ahead in pure C++.

2. **Mode B — Saguaro / Speculative Speculative Decoding (SSD, Kumar, Dao, May, arXiv:2603.03251):**
   - Separate lowered `TargetVerifyFX` (`K+1` verify) + `Gemma4AssistantFanoutUnrolledFX` (unrolled `K`-step batched `N`-hypothesis fan-out in ONE Metal CB, returning only `N*K = 40` int32 tokens = `160 B` readback!) borrowing target's live KV cache `MTLBuffer`s **zero-copy** via `mesh_graph_import_buffer()` (`shared_kv_b0=True`).
   - Implements Theorem-12 geometric fan-out `F_k` (`budget=8, K=5 → F=[2,1,1,1,1,2], N=8`), outcome-keyed cache `(n_acc, t*)` with sent-token exclusion (§4.1.1), Def-14 $\sigma_{F,C}$ top-$F$ proposal boosting, Leviathan lossless accept/reject + residual sampling, and concurrent async submission (`mesh_graph_submit_step` / `mesh_graph_wait_step`).

## 2. Live 1-Studio (`TP=1`) & 4-Studio (`TP=4`) Single-CB Fused MTP Speculative Decode Results

All runs use real BF16 weights and real prompts (`python_code`, `math_explain`, `json_struct`), checking greedy lossless token identity against baseline 1-token `Gemma4DecodeFX` (`64/64` tokens):

| `TP` | Wire | `K` | Prompt | Baseline AR (`ms` / `tok/s`) | **Spec Round (`ms`)** | `tok/round` (`acc + 1`) | **Steady `tok/s/stream`** | **Speedup vs AR** | Speedup vs MPS Eager (`16.4`) | Lossless Exact (`64/64`) |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 1 | `auto` | 5 | `python_code` | 6.01 ms / 166.5 | **9.04 ms** | **3.792** | **419.6** | **2.52×** | **25.6×** | **True** |
| 1 | `auto` | 5 | `math_explain` | 6.03 ms / 165.9 | **9.02 ms** | 2.792 | **309.6** | **1.87×** | 18.9× | **True** |
| 1 | `auto` | 5 | `json_struct` | 6.01 ms / 166.4 | **9.02 ms** | 2.667 | **295.8** | **1.78×** | 18.0× | **True** |
| 4 | `gather` (exact bf16) | 5 | `python_code` | 4.03 ms / 248.1 | **5.96 ms** | **3.750** | **629.7** | **2.54×** | **38.4×** | **True** |
| 4 | `gather` (exact bf16) | 5 | `math_explain` | 4.01 ms / 249.2 | **5.92 ms** | 2.792 | **471.6** | **1.89×** | 28.8× | **True** |
| 4 | `gather` (exact bf16) | 5 | `json_struct` | 3.96 ms / 252.8 | **5.93 ms** | **3.125** | **527.2** | **2.09×** | 32.1× | **True** |
| 4 | `auto` (`had8g`) | 4 | `python_code` | 3.93 ms / 254.5 | **5.56 ms** | 3.708 (`67.7%`) | **667.0** | **2.62×** | 40.7× | **True** |
| 4 | `auto` (`had8g`) | 4 | `math_explain` | 3.97 ms / 252.2 | **5.52 ms** | 2.875 | **520.6** | **2.06×** | 31.7× | **True** |
| 4 | `auto` (`had8g`) | 5 | `python_code` | 4.03 ms / 248.0 | **5.91 ms** | **4.042** | **684.0** | **2.76×** | **41.7×** | **True** |
| 4 | `auto` (`had8g`) | 6 | `python_code` | 3.97 ms / 251.7 | **6.17 ms** | **4.333** | **702.7** | **2.79×** | **42.8×** | **True** |
| 4 | `auto` (`had8g`) | 6 | `math_explain` | 4.04 ms / 247.3 | **6.05 ms** | 3.042 | **502.4** | **2.03×** | 30.6× | **True** |

*(Note: At `TP=4 wire=gather`, 100% of all prompts are bit-exact `64/64` lossless vs `TP=4` AR; under `wire=auto` (`had8g` 8.5-bpe Hadamard int8 wire), `python_code` and `math_explain` remain `64/64` bit-exact while `json_struct` flips 1 bf16 tie-break.)*

## 3. Saguaro / SSD Validation (`budget=8`, `F=[2,1,1,1,1,2]`, `N=8` Slots, `K=5`, `python_code`)

| Mode | `TP` | `temp` | $\sigma_C$ | Cache Hit Rate $p_{\text{hit}}$ | `mean_acc` | Verify (`ms`) | Fan-out (`ms`) | Lossless / Calibration |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `saguaro-ssd-serialized` | 1 | 0.0 | 1.0 | **75.0%** (18/24) | 1.25 | 7.75 | 2.63 | **55/55 exact (`True`)** |
| `saguaro-ssd-overlapped` | 1 | 0.0 | 1.0 | **75.0%** (18/24) | 1.25 | 7.86 | 4.50 | **55/55 exact (`True`)** |
| `saguaro-ssd-serialized` | 4 (`gather`) | 0.0 | 1.0 | **75.0%** (18/24) | 1.25 | 5.36 | 2.70 | **55/55 exact (`True`)** |
| `saguaro-ssd-overlapped` | 4 (`gather`) | 0.0 | 1.0 | **75.0%** (18/24) | 1.25 | 5.47 | 3.93 | **55/55 exact (`True`)** |
| `saguaro-ssd-overlapped` | 1 | 1.0 | 1.0 | **53.1%** (17/32) | 1.97 | 11.73 | — | Calib: pred `0.6701` vs emp `0.7241` (`n=87`) |
| `saguaro-ssd-overlapped` | 1 | 1.0 | **0.5** | **56.2%** (18/32) | **2.06** | 11.57 | — | Calib: pred `0.7092` vs emp `0.7253` (`n=91`) |

### Key Architectural Finding: Single-CB Fused MTP vs Saguaro SSD on Hidden-State Drafters
- Unrolling the `N=8 × K=5` Saguaro fan-out into ONE native Metal CB cuts fan-out GPU time from `13.9 ms` down to **`2.63–3.93 ms`**, fitting completely inside the `5.47 ms` `TP=4` verify window with a real **`75.0%` cache hit rate** and exact Leviathan calibration (`+3.1%` $p_{\text{hit}}$ and `+0.09` `mean_acc` from Def-14 $\sigma_{F,C=0.5}$).
- However, because `gemma4-assistant` is a **hidden-state-conditioned MTP drafter**, Saguaro's pre-verification fan-out must condition on the assistant's own projected hidden state + pre-verify KV (`pos - 1`) rather than the target's freshly verified `xn[n_acc]` + `pos + n_acc`, exhibiting the paper's Appendix-E self-conditioning drop (`mean_acc`: `3.04 → 1.25`) alongside same-GPU memory-bandwidth contention (`fanout` `2.7 → 3.9 ms`).
- By contrast, **Single-CB Fused MTP Speculative Decoding** keeps target verify + in-GPU accept/commit + unrolled `K`-step assistant drafting inside **ONE single Metal CB** (only `~1.8 ms` total for all `K=5` draft steps on `TP=4`, fresh target hidden states every round, zero Python), delivering **`419.6 tok/s/str` (`TP=1`) and `629.7–702.7 tok/s/str` (`TP=4`, `2.54–2.79×` vs AR, `38–43×` vs PyTorch MPS eager)** in pure native BF16 — decisively superseding the alternate LiteRT/Swift runner.
