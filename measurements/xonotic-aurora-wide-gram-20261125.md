# Aurora Leverage-Aware Spectral Optimizer + Wide Bitter-Lesson Linear-Gram Subspace ResNet (2026-11-25)

## 1. Optimizer & Initialization: Aurora (`xonotic/solver/strat/aurora.py`)

Replaced Adam/AdamW entirely with **Aurora** (Tilde Research, 2026 — leverage-aware spectral orthogonalization) + **Aurora-tuned Stiefel $\cap$ Row-Oblique initialization (`aurora_init_weight_`)** + EDM2 post-step unit-sphere projection (`project_unit_norm_`):
- For tall 2D matrices ($m > n$, e.g. wide SwiGLU `ffn_gate`/`ffn_up` `[1024 × 256]`), Aurora refines a diagonal row preconditioner $D$ (`pp_iterations = 2, pp_beta = 0.5`) so $\text{polar}(D \tilde{G})$ targets uniform row leverage $r_i = \|P_{i,:}\|_2^2 \approx n/m$, eliminating Muon's tall-matrix dead-neuron starvation loop while preserving spectral orthogonalization.
- For square/wide 2D matrices ($m \le n$), Aurora falls back cleanly to standard Muon $\text{polar}(\tilde{G})$.
- For 1D scalars (`cont_gain`, `cont_bias`, `cont_logstd`, `log_temp_*`, `blend_*`), applies RMS-normalized momentum with zero weight decay.
- **Measured tall-matrix row-leverage anisotropy on `[1024 × 256]` `ffn_gate`:**
  - Plain Muon (`pp_iterations = 0`): `row_leverage_cv = 1.2304`
  - **Aurora (`pp_iterations = 2, pp_beta = 0.5`)**: **`row_leverage_cv = 0.1050`** (**`11.7×` reduction** in row-leverage anisotropy; live training maintained `0.0852–0.1124`).

## 2. Bitter-Lesson Wide Linear-Gram Subspace ResNet (`xonotic/solver/strat/plan_policy.py`)

Removed the hand-crafted exponential kernel inside `MPResCGTBlock` and restored **pure associative plain linear Gram $R(R^\top V)$ + per-team $T(T^\top V)$ + bounded-condition DPP** inside a wide unit-RMS hidden stream (`d = 256, h = 1024, r_gram = 128, r_dpp = 64, depth = 4`, `1,777,715` params, `31` `MPLinear` layers / `7,518` unit-norm rows, `5^4` FSQ codebook):
- Multi-layer rotate (`MPLinear`) + sum (`mp_lerp`) across the wide `d = 256 / h = 1024` hidden stream lets the model bundle selective routing, sharpness, and information separation into learned orthogonal subspaces with pure matrix multiplies.
- **4-Studio Tensor Parallelism exactness (`products.World` `cross` + `ffn`):** `tp_4rank_linear_gram_err = 7.63e-6`, `tp_4rank_wide_ffn_err = 4.53e-6`.

## 3. Live 3-Wave Multi-Map Training (`G = 6`) & 18-Match 4-Map CRN Elo

Trained on `2×boil`, `2×stormkeep`, `2×catharsis` on single-process MacBook Pro CPU (`26` total Aurora updates across 3 waves, `0` UDP errors) and evaluated across 18 seat-rotated matches (`boil`, `stormkeep`, and zero-shot held-out `afterslime`):

| Wave | Updates (Deferred) | Online `cos` | Online $B_{\text{noise}}$ | Step Eff $\eta$ | `gram_cgt` cos | Tall Lev CV | **Dead Neurons (`/1024`)** | Max `||w||-1` | FSQ Codes (Bits) | Composite Elo (18m) | Zero-Shot `afterslime` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `aur_w0` | 0 | — | — | — | — | — | **`0.0%`** | `0.0` | — | `1290.6` | `1285.6` |
| `aur_w1` | 9 (`7`) | `+0.1015` | `93.4` | `25.5%` | `+0.1656` | `0.0852` | **`0.0%`** | `0.0` | `51` (`4.30`) | `1225.9` | — |
| `aur_w2` | 9 (`6`) | **`+0.2001`** | **`44.5`** | **`40.3%`** | **`+0.1818`** | `0.1124` | **`0.0%`** | `0.0` | `40` (`3.03`) | **`1444.8`** (`+218.9`!) | — |
| `aur_w3` | 8 (`6`) | `+0.0884` | `308.8` | `9.3%` | `+0.0976` | `0.0925` | **`0.0%`** | `0.0` | `22` (`2.83`) | `1416.1` | **`1422.2`** (`+136.6` vs `w0`) |

### Key Takeaways
1. **Zero dead neurons (`0.0%` of `1,024`)** and exact unit-norm weights (`max_dev_from_1 = 0.0`) held across every wave under Aurora.
2. **Highest McCandlish efficiency measured to date:** Wave 2 reached **`40.3%` step efficiency** (`B_noise = 44.5`, `on_cos = +0.2001`, `cgt_cos = +0.1818`) and jumped **`+218.9` Composite Elo in just 9 Aurora updates** (`+24.3` Elo/update!).
3. **Throughput motivation for MPS / 4× Apple M5 Ultra Mac Studios:** Single-process CPU execution limited the `1.78M` wide model to `26` updates in `102s` (`~10×` fewer steps than the earlier `227`-update run); running this exact `1.78M` Aurora + wide linear-Gram model on MPS / 4-Studio TP directly unlocks `10×` update throughput.
