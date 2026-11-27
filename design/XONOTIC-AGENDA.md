# Xonotic Strategy — Grounded Specification & Research/Training Agenda

Every normative constraint here traces directly to verbatim user quotes in [SPECIFICATION.md](SPECIFICATION.md) (§1–§26) and [NAV-SPEC.md](NAV-SPEC.md). Agent-authored historical docs that contradicted or drifted from those quotes have been deleted from `design/` (retained only in git history).

---

## 1. Grounded System Specification

### 1.1 Game & Control Boundary (`SPECIFICATION.md` §1, §2, §12, §16, §19, §26)
- **Two nested subgames:**
  - *Subgame A (laggy discrete cart-checkpoint game):* `K` teams, `J` cart lanes, plural checkpoints per lane; score integrates monotonically `dS_i/dt = α · H_i` to threshold `Q`; instantaneous projected winner `PW(s)` and counterfactual denial succession `SUCC(s)` give a tractable combinatorial-game-theory (nim-like counting game) signal.
  - *Subgame B (continuous Xonotic FPS):* weapon/HP/armor/positioning/combat dynamics inside HavocBot / DarkPlaces.
- **Control surface (to-spec per §26):** Strategic `PLAN` (`PLAN_WIDTH = 16` parametric flow weights/targets over HavocBot), while stock HavocBot navmesh/aiming executes movement/combat (`§12`: *"matmul = WHAT, stock navmesh = HOW"*). Direct per-bot `usercmd` actions are an explicit future extension, not an implicit requirement.
- **No spectator / bot-only acceleration (`§26`):**
  - Paced DarkPlaces oracle: `g_payload_use_view=0` (eliminates the `PRVM_VIEWINTERPRETER` memory bomb), `sv_headless_unthrottled=1` (~`10–11×` realtime when no human client is connected; auto-reverts to 1× when a spectator/player joins), zero-copy POSIX `shm://` `MeshVFSStore` for streaming in-memory IBSP v46 maps.
  - Variable-$\Delta t$ SIMD/GPU engine ([xonotic/sim_dt/](../xonotic/sim_dt/)): exact at integer tick step ($\Delta t = 0.05\text{ s}$), analytic score-out & checkpoint-crossing root integrals for macro $\Delta t$, `O(1)` savestate `snapshot()` / `fork_group()` for counterfactual GRPO rollout groups.

### 1.2 Policy Architecture (`SPECIFICATION.md` §2, §5, §6, §7, §8, §9, §17, §20, §23, §24, §26)
- **Statewise, zero recurrence (`§2`):** *"no part of this is sequential or stateful and that we aren't interested in recurrence (lol)"*. No `Mealy` GRU, no hidden-state carry across ticks.
- **No feature-engineered behavior or truncating bottlenecks (`§7`, `§17`, `§20`):**
  - No hardcoded `cart_dist_bias` nearest-cart prior inside `cart_logits`.
  - No `FSQBottleneck` squeezing IR through 4 scalars.
  - No dead `map_manifold.py` Voronoi encoders (`enc_track`, `enc_cell`, `enc_item`, `cell_q/k`, `fovea_weights`, zero-padded `BOT_GEO_DIM`/`CART_GEO_DIM`).
  - No premature averaging/reduction of raw entity state before the learned trunk.
- **One common wide IR (`§5`, `§8`, `§23`, `§24`):**
  - Shared entity projections (`obs`, `carts`, `teams`) → **factored global + team Gram matrix interaction** + **wide SwiGLU** (no softmax attention in the mixing trunk; optional DPP diversity kernel `§9`).
  - **Both policy and value heads (`W` / `L`) read the exact same final IR** via linear projections—no critic-only privileged side-channel (`value_priv_proj`) and no actor-only masking of HavocBot skill percepts.
- **HavocBot skill percepts (`SPECIFICATION.md` final section):** Bot skill/configuration fields remain **available as percepts** in `obs` and are **masked out of the output action space** (never writable by the policy), serving as robotics-style domain randomization.

### 1.3 Reward & Credit Assignment (`SPECIFICATION.md` §1, §4, §14, §15, §26)
- **Primary objective (CGT sparse role-asymmetric atoms — `§4`, `§14`, `§15`):**
  - `W` (Winnie, projected winner `q_i = 1[i = PW]`): sparse perturbation-resistance reward `r_W = -q_i (1 - q'_i)` (optionally augmented in controlled ablations with the `arXiv:2406.18805` boundary stability margin).
  - `L` (Lou, non-winners `1 - q_i`): sparse upward loser-rank-flip reward `r_L = (1 - q_i) · 1[rank'_i > rank_i]`.
  - `V_W` and `V_L` never cross-bootstrap across role flips (`winner_mask == next_winner_mask`).
  - **No `reward = score` or whole-game RLVR** as the default training signal (kept strictly as explicit ablation arms if ever compared).
- **Mid-match counterfactual rollout groups:** Fork contested mid-match savestates into `G` branches (Common Random Numbers) to estimate within-group counterfactual advantage (GRPO) blended with role-separated GAE.
- **Modular configuration (`§26`):** All loss weights live in [xonotic/solver/configs/losses.json](../xonotic/solver/configs/losses.json) (`losses.py`), all non-loss architecture/optimizer/world/harness/sim_dt parameters live in [xonotic/solver/configs/](../xonotic/solver/configs/) (`train_config.py`).

### 1.4 Hardware Roofline, Compression & Compiler Lowering (`SPECIFICATION.md` §21, §25, §26)
- **Measured cluster hardware (4× Apple M5 Ultra `Mac17,15`, 64-core Metal 4 GPU, 96 GB RAM; 1× Apple M5 Max `Mac17,6`, 40-core Metal 4 GPU, 128 GB RAM):**
  - M5 Ultra sustained unified memory BW: **`961.0 GB/s`**.
  - M5 Ultra `4096³` GEMM: **FP32 = `23.95 TFLOP/s`** (`ridge = 24.9 FLOP/B`) vs **BF16/FP16 = `92.54–92.69 TFLOP/s`** (`ridge = 96.3–96.5 FLOP/B`).
  - Thunderbolt 5 RDMA measured link BW: `8.3–9.3 GB/s` per direction (`ridge ≈ 10,000 FLOP/B` at BF16).
- **No uncompressed FP32 anywhere (`§26`):**
  - Compute weights & activations in **BF16/FP16** on M5 Ultra's Metal 4 Neural Accelerators (`3.87×` compute + `2×` bandwidth over FP32).
  - Compress inter-node collectives, rollout/replay buffers, and activation transport using `main`'s existing Pareto codec library ([compiler/mesh_compiler/codec.py](../compiler/mesh_compiler/codec.py), [rdma/mesh-codec.h](../rdma/mesh-codec.h), [rdma/shaders/had_int8_kernels.metal](../rdma/shaders/had_int8_kernels.metal)): `had_int8` (`8.5 bpe`), `had_int6_pk` (`6.5 bpe`), `had_int4` + error-feedback (`*_ef`) for gradient all-reduce, selected by analytical `(bpe, alpha_us, KL)` lower hull rather than fixed heuristics.
- **Compiled Metal execution (`main`):** Lower static-shape policy graphs through [compiler/mesh_compiler/fx_metal.py](../compiler/mesh_compiler/fx_metal.py) → `rdma/mesh-graph.mm` (`libmesh-graph.dylib`: single `MTLCommandBuffer`, Metal 4 MPP `gemm_rows_nax`, liveness arena coloring, zero-copy `/mesh0` flat-DMA, `had8g`), eliminating eager PyTorch MPS dispatch overhead.
- **Roofline-derived sizing (no arbitrary powers of 2):**
  - Size microbatches by **entity-row budget** `T_micro = E * (N + J + K)` (aligned to SIMD/tile multiple of 32, kept inside cache-resident window above GEMM ridge and below SLC spill), NOT by fixed `minibatch_envs`.
  - Decouple statistical optimizer batch size from `T_micro` via gradient accumulation sized by McCandlish gradient SNR (`B_noise`) and optimizer amortization.

---

## 2. Explicit Domain-Randomization & Parametric World Agenda

Every training and evaluation episode samples from an explicit joint distribution ([WorldDistributionConfig](../xonotic/solver/strat/train_config.py)) streamed zero-copy into memory:

| Axis | Sampling Distribution | Purpose |
|---|---|---|
| Team count `K` | `uniform` / `lognormal` over `[2, 24]` (training range `[4, 12]` in `sim_dt`, up to `22–24` in stress eval) | Prevent 3-team hardcoding; force coalition / leader-denial generalization |
| Cart lanes `J` | `K × mult`, `mult ∈ [2.0, 12.0]`, `J ≤ 256` | Prevent fixed-lane memorization; test multi-lane CGT urgency |
| Player count `N` | `K × per_team`, `per_team ∈ [2, 16]`, `N ∈ [12, 240]` (and `sim_dt` stress up to `10⁴+` once combat switches to spatial hash) | Scale invariance across squad density and match size |
| Map families & tiles `M` | `gas_world_bsp` (procedural multi-tile expander, `M ∈ [3, 16]`, variable joint widths) + `b3313_binary_stitch` (non-Euclidean stitched stock IBSPs) | Zero-copy in-memory `MeshVFSStore` (`shm://`) streaming; geometry transfer |
| Bot mechanical skill | Native calibrated HavocBot roster mixtures (`gaussian_lobby`, `team_skewed`, `within_team_skewed`, `uniform_native`) exposed as read-only percepts | Robotics-style mechanical domain randomization (`SPECIFICATION.md` final section) |
| Physics & tempo | `curriculum.py` perturbations (`baseline`, `fast`, `slow`, `volatile`) + variable `Δt ∈ {0.05, 0.2, 0.4, 1.0}` | Invariance to cart speed, integration step, and horizon |
| League opponents (PFSP + FCP) | `default` + 4 scripted anchors + frozen policy snapshots + mixed-teammate FCP squads, weighted by `(1 - p_win)^p` | Eliminate mutual-passivity fixed points and mirror-only overfitting |

---

## 3. Explicit Research & Experiment Agenda

1. **Architecture & Spec Cleanup (Immediate):**
   - Strip `PlanWally` to a statewise, common-IR, BF16-native Gram+SwiGLU policy with zero dead encoders, zero `Mealy`, zero `FSQ`, zero `cart_dist_bias`, zero actor-input skill masking, and row-budgeted microbatching (`target_microbatch_rows` aligned to 32 + gradient accumulation).
   - Keep CGT sparse `W`/`L` reward (`r_W`, `r_L`) as the default supervision; isolate any dense shaping or `terminal_win` baseline as explicitly named ablation configs in `xonotic/solver/configs/`.
2. **Compressed Execution & Compiler Lowering:**
   - Replace FP32 tensors across `sim_dt` rollout buffers, policy weights/activations, and RDMA all-reduce with BF16 + `M5_CODEC_TABLE` (`had_int8` / `had_int6` / `*_ef`) and measure actual wall time, bandwidth, and KL divergence on 1–4 M5 Ultras.
   - Lower the stripped static-shape policy forward through `fx_metal.py` (`libmesh-graph.dylib`, single-CB Metal 4 MPP) and fuse `DeltaTimePayloadEngine.step()` hot loops into Metal kernels + spatial-hash combat (`O(N)` instead of `O(N²)`).
3. **Controlled Scientific Ablations (Measured on Both `sim_dt` Arena & Seat-Rotated Live DarkPlaces `sv_headless_unthrottled=1`):**
   - **A1 (Reward contract):** Pure sparse `W`/`L` (`r_W`, `r_L`) vs `+ 2406.18805` boundary barrier vs `terminal_win` RLVR — measured on live DarkPlaces Elo, α-Rank, and anti-leader coalition targeting (`outer_validation.py`).
   - **A2 (CGT lane features & DPP):** Raw `(obs, carts, teams)` vs `+ lane thermographs/nimbers` vs `+ DPP Gram` — verify `j-space` linear decodability and Elo gain without any hardcoded `cart_dist_bias`.
   - **A3 (Anticipatory Replicator Dynamics, `2603.03173`):** Standard clipped PPO/GRPO vs anticipatory phase-lead policy updates against shifting multi-team PFSP leagues.
   - **A4 (Sim-to-DarkPlaces transfer & fine-tuning):** Zero-shot transfer from `sim_dt` → unthrottled DarkPlaces vs mixed `sim_dt` pretraining + live DarkPlaces oracle calibration across held-out `gas_world_bsp` and `b3313_binary_stitch` worlds.
   - **A5 (4× M5 Ultra Pareto scaling):** Single-node vs 4-node RDMA (`had8g` / `*_ef`) throughput, useful TFLOP/s, wire bytes, and McCandlish step efficiency across row-budgeted microbatches.
