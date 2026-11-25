# Xonotic 5-Pillar CGT & Control-Theoretic Synthesis — Live Measurement Record (2026-11-24)

## Summary of Integrated Algorithms (`xonotic/solver/strat/`)

All five motivating theoretical strands audited from the Aug 29–Sep 7 transcripts and primary literature were synthesized into the live PyTorch/MPS policy stack without breaking the literal checkpoint-control game (`design/CART-GAME-CONTRACT.md`) or wire schemas (`OBS_WIDTH=84`, `CART_WIDTH=18`, `TEAM_WIDTH=7`):

1. **Subgame-A Singularity-Free Margin, Lane Thermographs & Bounded Nim-Sum (`game_value.py`, `strategy.py`):**
   - Finish rates $\rho_i = \alpha H_i / \max(Q - S_i, \varepsilon)$ and winning-region margin $M_i(\rho) = (\rho_i - \max_{k \neq i}\rho_k)/(\rho_i + \max_{k \neq i}\rho_k + \varepsilon) \in [-1, 1]$ (`500/500` exact equivalence to `evaluate_cartstate` projected winner, including ties and $H_i = 0$).
   - Per-lane Conway thermographic urgency (`T_deny`, `T_push`) and FUN 2022 bounded-step homomorphic Sprague–Grundy XOR (`60/60` exact match vs. recursive `mex`) appended in-graph as 4 cart features (`18 → 22`) and 2 team features (`7 → 9`).
2. **Winning-Region Stability Barrier & Local Controllability (`arXiv:2406.18805`, `game_value.py`, `cast_header.py`, `execution.py`):**
   - Interior-safe quadratic hinge barrier on `W` (`margin=0.25`, `weight=0.5`, identically `0.0` gradient in the safe winning interior $M_i \ge 0.25$, smooth boundary gradient before `PW` flips).
   - Locally action-linear dynamics head $\Delta\hat{\rho} = b_\eta(h) + \langle A_\eta(h), v \rangle$ supervised on observed $\operatorname{asinh}(\rho' - \rho)$ (`LOSS_WEIGHTS["dynamics"] = 0.25`).
3. **Parallel Two-State Leaky Anticipatory Steering (`arXiv:2603.03173` Theorem 6, `state_steering.py`, `strat_responder.py`):**
   - Stateful per-participant fast leaky lead filter $m_{t+1} = e^{-\Delta/\tau_m} m_t + \gamma(1 - e^{-\Delta/\tau_m}) v_t$ ($\tau_m = 0.25\tau$) replacing the unfiltered instantaneous feedthrough spike; closed-form Lyapunov dominance $J(v) = \frac{1}{2\gamma\lambda}\|m(T)\|^2 + \frac{1}{\gamma}\int_0^T\|m\|^2 dt \ge 0$ verified on `200/200` random/adversarial trajectories (`3.81e-06` max rel-err).
4. **Dual Coalition + DPP-Debiased Gram & Woodbury Correlated Action Covariance (`dpp.py`, `strategy.py`, `state_steering.py`):**
   - Exact $O(N r d + r^3)$ DPP-debiased Gram $B(I_r + B^\top B)^{-1}(B^\top V)$ (`8.88e-15` vs. dense $O(N^3)$) wired into `encode_finish` alongside the RDMA-shardable Coalition and Team Grams.
   - Rank-$k$ ($k=4$) Woodbury low-rank + diagonal Gaussian velocity head ($\Sigma = D^2 + FF^\top$, pure-PyTorch MPS-native Cholesky log-det & solve, `1.78e-14` vs. dense MVN, exact `0.0` gradient on read-only skill words).
5. **MPS & Numerical Conditioning Fixes Discovered During Live Runs (`matmul.py`, `strategy.py`, `dpp.py`, `state_steering.py`):**
   - Guarded 0-extent `matrix_multiply*` ops against Apple MPS `Failed to allocate private MTLBuffer for size 0` SIGABRT on tick 1 (`events == 0`).
   - Replaced non-contiguous 3D MPS `bmm` in `logp_of`/`sample` with elementwise rank-$k$ contractions.
   - Normalized `ir` (`norm(ir)`) and averaged critic values over valid pages in `read_heads`, eliminating a live TD(0) runaway explosion (`loss_l: 132.9 → 46.4M / 2.02B` → stabilized to `0.0081 / 0.0021`).
   - Replaced 32-step Python CG loop in `dpp_gram_operator` with exact `32×32` `torch.linalg.solve` (16× MPS emit speedup: `1328.5 ms → 81.7 ms`, 10× learn speedup: `1323.3 ms → 128.3 ms` at `9 bots × 15 pages`).

## Empirical Validation Results

### 1. Deterministic Policy & VJP Parity (`solver.strat.cross`, CPU)
- `passed: true`, `forward_error: 0.0`, `largest_parameter_gradient_error: 0.0` across all **33** parameters (`dpp_metric`, `dpp_quality`, `dpp_out`, `heads`, `dyn_drift`, `dyn_action`, `scale_w1..3`, `rival_metric`, `team_metric`, `neighborhood.*`), `local_s = 13.3 ms`.

### 2. Live 3-Team / 3-Cart Xonotic Server (`boil`, `e35f6628a` `darkplaces-dedicated` + `build set 3493794463` `progs.dat`) on Apple Silicon MPS
- **Configuration:** Team 1 = `matrix_fusion` (online trained), Team 2 = `terminal_win` (online trained counterfactual), Team 3 = `default` (stock Havocbots), `strategy_interval = 0.4 s`, `anticipatory_lead = 1.0`.
- **Mid-match perturbatory probes injected via server stdin:** `t=18s: bot_number 12`, `t=36s: g_payload_speed 90; g_payload_contest_speed 50`, `t=54s: bot_number 15`, `t=72s: bot_number 9; g_payload_reverse_speed 45`.
- **Observed live run (`87.8 s`, 12 full responder frames, 10 online MPS AdamW updates per arm):**
  - Dynamic count & page-width invariance realized live: players `[1, 9, 12]`, `StateWidth: 3,840 → 44,288` (15 → 173 VM pages/bot), **`view_faults = 0`**, **0 transport errors**.
  - Live combat/events realized: `36 damage`, `3 pickups`, `1 kill`.
  - `matrix_fusion` (10 MPS updates, 100% finite): `loss: 0.0165 → 0.0152`, `loss_l: 0.0090 → 0.0081`, `loss_dynamics: 0.0009 → 0.0001`, `loss_moe_balance: 1.1777 → 1.1116`, `gradient_norm: 5.21 → 4.22`.
  - `terminal_win` (10 MPS updates, 100% finite): `loss: 0.0171 → 0.0123`, `loss_l: 0.0102 → 0.0021`, `loss_dynamics: 0.0009 → 0.0006`, `loss_moe_balance: 1.1777 → 1.1052`, `gradient_norm: 5.54 → 0.23`.
  - Trained live checkpoint `j-space` effective rank: **`16.40 / 128`** (non-collapsed; random-init control `15.72 / 128`).

### 3. Pre-Existing Engine / Runtime Findings Surfaced by the Live Run
1. **QuakeC event edict leak (`sv_payload_strategy_io.qc:payload_str_deposit`):** Spawning event edicts faster than they are reaped under 12–15 bots bloated `StateWidth` from `3,840 → 44,288` (`173` VM pages/bot) and hit `Host_Error: PRVM_ED_Alloc: no free edicts` at `t = 87.1 s`.
2. **Background `LiteralJReporter` shutdown stall (`joracle/probe.py`):** Even with `--measure-rows 0`, `LiteralJReporter` still serialized `275 MB` (`2.8M`-entry arrays) and blocked `SIGTERM` exit while waiting on its worker thread.
3. **Headless userspace probe client (`xonotic-osx-sdl-bin`):** Connected cleanly over UDP (`PerturbProbe1 connected` in `server.log`) with `SDL_VIDEODRIVER=dummy +vid_soft 1`, then dropped in `_CSQC_Ent_Update` because stock 2023 `csprogs.dat` lacks our custom `ENT_CLIENT_PAYLOAD_CART` entity while custom `darkplaces-sdl` is built without DPSOFTRAST (`vid_soft`).
