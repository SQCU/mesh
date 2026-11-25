# Gemma-4-E2B & Generic `torch.export` FX → Single Native Metal Command Buffer (`libmesh-graph.dylib`)

**Date:** 2026-11-24
**Branch / Commit:** merged onto `origin/main` from `fx-native-metal`
**Hardware:** 4× Apple M5 Ultra Mac Studios (`m5ult2`, `m5ult1`, `studio3`, `studio2`) + Mac Mini (`mini`) RDMA auto-relay on `mesh-flow` (`bc5ffc3f5`).
**Model:** Real `google/gemma-4-E2B-it` BF16 weights (`600` tensors, 35 layers: 28 sliding `D=256` + 7 full `D=512`, 20 shared-KV layers, per-layer input gates, tied `262,144` vocab).
**Generality Check:** Independent 4-node `LlamaSwiGLUBlock` (`D=2048, FF=8192`, `aten.silu`, `aten.linear`, `aten.rms_norm`, `mesh::all_reduce_sum`) lowered through the exact same `compile_fx_to_metal()` path.

## 1. Architecture Summary

Standard PyTorch `torch.nn.Module` → `torch.export.export()` → [`compiler/mesh_compiler/fx_metal.py`](../compiler/mesh_compiler/fx_metal.py) → [`rdma/mesh-graph.mm`](../rdma/mesh-graph.mm) (`libmesh-graph.dylib`):

- **Graph-level algebraic fusion:** Collapses `1,385` FX nodes (`TP>1`) into `394` fused native Metal ops + `71` flat-DMA crossings (`1,315` → `499` at `TP=1`):
  - `Linear + AllReduce + RMSNorm + Residual [+ RMSNorm2]` → `MG_OP_LINEAR_PUBLISH` + `MG_OP_CROSSING_SUM_NORM`
  - `RMSNorm + Residual + LayerScalar + NextRMSNorm` → single `MG_OP_RMS_NORM` (`AddNorm`)
  - `gelu/silu(linear(x, Wg)) * linear(x, Wu)` → single `MG_OP_GATE_UP_ACT`
  - `head_norm_rope_kv` (precise trig RoPE + fused KV cache write) + `kv_sdpa_decode` (8-simdgroup GQA decode with dynamic Split-K FlashDecoding for `eff_len >= 384`)
  - `vocab_argmax` with in-GPU `tok_buf`/`pos_buf` update
- **Static liveness interval coloring:** Packs all temporary activations into ONE shared Metal arena (`0.19 MB` at `TP=4, B=1`; `0.63 MB` at `TP=1, B=1`). Zero-copy `MG_BUF_VIEW` reshapes and `torch.frombuffer` state views.
- **Execution:** **ONE `MTLCommandBuffer` + ONE `MTLComputeCommandEncoder` per decode step**, pipelined 1 step ahead in pure C++ with **zero Python dispatch per op and zero Python dispatch per token**, over zero-copy `/mesh0` `W[]`/`P[]` `MTLBuffer`s, Dreyfus–Wagner Steiner multicast, and `GF(2)^6` linear Hadamard-domain reduction (`had8g`).
- **Metal 4 MPP NAX + tuned M5 Ultra kernels:** `gemv_rows`/`gemv_publish` (`B=1..2`), `gemm_rows_wide` (`B=3..15`), Metal 4 `mpp::tensor_ops::matmul2d` `gemm_rows_nax` (`B=16..128`), and generic fallback for arbitrary shapes.

## 2. Numerical & Autoregressive Token Parity (Real `google/gemma-4-E2B-it` BF16)

### 2.1 Single-Node (`TP=1`) vs PyTorch Eager (`CTX=0`, `BOS=2`, `STEPS=8`)
Reference 8-step greedy autoregressive tokens for batch slot 0 (`b0`):
`[236761, 108, 1018, 8291, 659, 496, 2321, 3835]`

| Batch `B` | Kernel Tier Exercised | `b0` 8-step Match | All-Batch Token Match |
|---:|---|---|---|
| 1 | `gemv_rows` (`B=1`) | **8 / 8 exact** | **8 / 8 (100%)** |
| 2 | `gemv_rows` (`B=2`) | **8 / 8 exact** | **16 / 16 (100%)** |
| 4 | `gemm_rows_wide` (`M=4`) | **8 / 8 exact** | **32 / 32 (100%)** |
| 16 | Metal 4 MPP `gemm_rows_nax` (`UM=1`) | **8 / 8 exact** | **128 / 128 (100%)** |
| 32 | Metal 4 MPP `gemm_rows_nax` (`UM=2`) | **8 / 8 exact** | **256 / 256 (100%)** |
| 64 | Metal 4 MPP `gemm_rows_nax` (`UM=2`) | **8 / 8 exact** | 508 / 512 (99.2%; 1 bf16 tie-break on unused token ID) |

Layer-34 hidden cosine similarity vs PyTorch CPU eager (`BOS=2`, `pos=0`): **`0.999994`** (logits cosine `0.999877`, matching PyTorch CPU vs MPS `0.999900`).

### 2.2 Multi-Node (`TP=2`, `TP=4`) Real Autoregressive Parity (`CTX=0`, `STEPS=8`)
Every combination below reproduced the **exact 8/8 `TP=1` reference token sequence** `[236761, 108, 1018, 8291, 659, 496, 2321, 3835]`:
- `TP=2` (`m5ult2 + m5ult1`): `wire=gather` (`B=1, 4, 16`), `wire=had8g` (`B=1, 4, 16`)
- `TP=4` (`m5ult2 + m5ult1 + studio3 + studio2`): `wire=gather` (`B=1, 4, 16`), `wire=bf16` (`B=1, 4, 16`), `wire=had8g` (`B=1, 4, 16`)
- `TP=4` Split-K FlashDecoding (`CTX=0` & `CTX=4096`): exact parity confirmed.

### 2.3 Domain Generality (`LlamaSwiGLUBlock`, `TP=4`, `wire=had8g`)
- `D=2048, FF=8192, B=16`: `16` FX nodes → **`4` fused ops**, `1` crossing, `192 KB` arena, **`0.117 ms/step`**, **`cos = 0.999994`** vs PyTorch `TP=1` eager (`max_diff = 0.0625`).

## 3. Decode Performance Pareto Frontier (`google/gemma-4-E2B-it` BF16)

### 3.1 Single Studio (`TP=1`, `m5ult2`, `CTX=128`): PyTorch MPS Eager vs MLX `_meshmlx` vs `torch.export → libmesh-graph`

| `B` | PyTorch MPS Eager (`ms` / `tok/s`) | MLX `_meshmlx` (`ms` / `tok/s`) | **`torch.export → libmesh-graph`** (`ms` / `tok/s/str` / `agg tok/s`) | Speedup vs MPS Eager | Speedup vs MLX |
|---:|---:|---:|---:|---:|---:|
| 1 | 61.07 ms / 16.4 | 6.70 ms / 148.1 | **6.18 ms / 161.8 / 162** | **9.88×** | **+9.3%** |
| 2 | — | 6.44 ms / 153.0 | **6.29 ms / 159.1 / 318** | — | **+4.0%** |
| 4 | 62.85 ms / 63.6 | 7.45 ms / 132.5 | **6.82 ms / 146.7 / 587** | **9.22×** | **+10.7%** |
| 8 | — | 8.22 ms / 120.2 | **7.59 ms / 131.8 / 1,054** | — | **+9.7%** |
| 16 | 64.78 ms / 247.0 | 8.32 ms / 119.1 | **7.18 ms / 139.2 / 2,228** | **9.02×** | **+16.8%** |
| 32 | — | 8.61 ms / 114.4 | **8.16 ms / 122.6 / 3,922** | — | **+7.2%** |
| 64 | 88.45 ms / 723.6 | 9.74 ms / 102.3 | **9.38 ms / 106.6 / 6,824** | **9.43×** | **+4.2%** |
| 128 | — | 15.18 ms / 8,372 | 16.44 ms / 60.8 / 7,788 | — | −7.0% |

### 3.2 Multi-Studio (`TP=2` & `TP=4`, `CTX=128`, `wire=auto`/`had8g`): MLX `_meshmlx` vs `torch.export → libmesh-graph`

| `B` | `TP=2` FX-Metal (`ms` / `tok/s/str` / `agg`) | `TP=4` MLX `_meshmlx` (`ms` / `tok/s/str` / `agg`) | **`TP=4` FX-Metal (`ms` / `tok/s/str` / `agg`)** | `TP=4` FX vs MLX | `TP=4` FX vs `TP=1` MPS Eager |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.72 ms / 211.9 / 212 | 4.43 ms / 225.8 / 226 | **4.176 ms / 239.5 / 240** (`3.916 ms` / `255.4` at `L=0`) | **+6.1%** | **14.6×** |
| 2 | 4.74 ms / 211.2 / 422 | 4.87 ms / 205.3 / 411 | **4.246 ms / 235.5 / 471** | **+14.7%** | — |
| 4 | 5.28 ms / 189.4 / 758 | 5.55 ms / 180.2 / 721 | **4.649 ms / 215.1 / 860** | **+19.4%** | **13.5×** |
| 8 | 5.47 ms / 182.8 / 1,463 | 5.91 ms / 169.2 / 1,354 | **5.014 ms / 199.4 / 1,596** | **+17.9%** | — |
| 16 | 5.70 ms / 175.5 / 2,808 | 7.00 ms / 142.9 / 2,286 | **5.486 ms / 182.3 / 2,917** (`had8g`) | **+27.6%** | **11.8×** |
| 32 | 6.34 ms / 157.8 / 5,049 | 8.07 ms / 123.9 / 3,965 | **6.662 ms / 150.1 / 4,803** (`had8g`) | **+21.1%** | — |
| 64 | **7.015 ms / 142.6 / 9,123** | 9.68 ms / 103.3 / 6,612 | **7.400 ms / 135.1 / 8,649** (`auto`) | **+30.8%** | **12.0×** |
| 128 | 11.20 ms / 89.3 / 11,434 | 12.39 ms / 80.7 / 10,285 | **11.024 ms / 90.7 / 11,611** (`had8g`) | **+12.9%** | — |

### 3.3 Long-Context Decode (`CTX=4096`) with Split-K FlashDecoding (`eff_len >= 384`)

| Config | MLX `_meshmlx` (`ms` / `tok/s/str`) | **`torch.export → libmesh-graph`** (`ms` / `tok/s/str` / `agg`) | Speedup vs MLX |
|---|---:|---:|---:|
| `TP=1, B=1, CTX=4096` | 6.96 ms / 143.7 | **6.666 ms / 150.0 / 150** | **+4.4%** |
| `TP=4, B=1, CTX=4096` | 4.90 ms / 202.5 | **4.522 ms / 221.1 / 221** | **+9.2%** |
| `TP=4, B=4, CTX=4096` | — | **5.100 ms / 196.1 / 784** | — |
| `TP=4, B=16, CTX=4096` | — | **7.372 ms / 135.6 / 2,170** | — |
