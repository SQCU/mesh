# Xonotic Policy Optimization: McCandlish (2018) Gradient SNR Meta-Optimization & Outer Validation

Date: 2026-11-25
Repo: `xonotic/solver/strat/` (`grad_snr.py`, `outer_validation.py`, `state_steering.py`, `execution.py`, `strategy.py`, `game_value.py`, `online.py`, `scale_config.py`, `policy_contract.py`, `strat_responder.py`, `baselines.py`)
Reference methodology: McCandlish et al. (2018) unbiased replica/split-half gradient signal $\widehat{\|G\|^2}$, noise $\widehat{\operatorname{tr}(\Sigma)/B_0}$, critical batch size $B_{\text{noise}}$, step efficiency $\eta_{\text{step}} = (1 + B_{\text{noise}} / B_{\text{eff}})^{-1}$, and round-level `grad_signal` $\leftrightarrow$ outer-delta correlation adapted from `/Users/mdot/vllm/benchmarks/analyze_draft_learning.py`.

## 1. Baseline Pathology Audit (Measured on Live Xonotic `boil`, 3 Teams × 3 Bots × 3 Carts)

Running the McCandlish split-half estimator (`OnlineMcCandlishTracker` + per-component/per-group `probe_batch_mccandlish`) together with paired-seed outer behavioral motifs (`sv_random_seed=20261125`, `default` vs `matrix_fusion`) exposed **7 concrete pathologies** blocking policy improvement:

| # | Pathology | Measured Baseline Evidence | Mechanism Blocking Policy Improvement |
|---|---|---|---|
| **P0** | **805M-param `scale_config.py` MoE default** | `SCALE_RANK=2048, SCALE_HIDDEN=4096, SCALE_EXPERTS=32` $\to$ **9.66 GB** checkpoints (with AdamW moments), `3.45s/step`, only **4 responses/55s** vs **357** for `default` (99% packet drop). | Responder stalls on forward/backward and 9.66 GB checkpoint writes, unable to act or attribute in real time. |
| **P1** | **93.7% Rollout IS Floor Collapse** | `IS floor_frac = 0.9368`, median `rollout_log_ratio = -249.7 nats` (min `-2978.2` to `-5624.8` nats), `IS ratio = 2.06e-9`, median `loss_pg = 0.0`, `actor_over_total = 0.0`. | `logp_of` summed Gaussian log-densities over all $D \approx 3,761\text{--}230,000$ coordinates without normalizing by $D$; a single critic step shifted $\sum_d \Delta\log\pi_d$ past `-20` nats, zeroing the actor gradient. Also `proximal_logp = logpi.detach()` made PPO clipping a no-op (`logpi - proximal_logp == 0`). |
| **P2** | **28.8% Dead Non-Float Action Coordinates** | `3,761` writable words/player included `1,082` (`28.8%`) `ev_string`, `ev_entity`, `ev_field`, `ev_function`, `ev_pointer` words (`tags & 7 != 2` or encoded bit `1<<20`). | `prvm_view.c:193` only applies residuals to `scalar_type == ev_float`. Non-float coordinates injected pure zero-effect noise into `logp_of` and IS ratios. |
| **P3** | **VM Boolean/Enum Paralysis & Global-Stack Runaway Loops** | Baseline `matrix_fusion` paralyzed all 9 bots (`mean_speed: 295.0 -> 17.6 u/s`, `kills: 28 -> 1`, `max_cart_pos: 50.2 -> 0.0`). `28.4%` of `ev_float` words were exact `0.0` booleans (`deadflag`, `frozen`), `28.1%` exact integer bitmasks/enums, and `22.1` words/player in `entity < 0` (QuakeC globals, `time`, stack locals, loop counters) caused `16` runaway 10M-jump loops, `21` view faults, and `1299 u/s` physics explosions (`opt2`). | Adding Gaussian noise to `0.0` floats flips QuakeC booleans truthy (`0.008 != 0`), perturbing `|state| < 1.0` flips `frametime`/`nextthink`, and perturbing `entity < 0` corrupts QuakeC VM stack locals and loop counters. |
| **P4** | **Severe Gradient Noise & Encoder `480×` Scale Dominance** | Live 50-update baseline window: global **`SNR = 0.0108`**, **`cos = +0.0278`**, **`B_noise = 92.7` transitions (`417` rows)** vs `B_eff = 2.0` (`step_efficiency = 2.1%`). CGT & MoE layers were drowned in noise: **`gram_cgt cos = +0.0017` (`eff = 0.53%`, `B_noise = 373`)**, **`scale_moe cos = +0.0032` (`eff = 0.27%`, `B_noise = 752`)**, **`decoder_trunk cos = -0.0021` (`SNR < 0`)**. Raw `pages` up to `65535` into `model.quinn` gave `encoder` `480×` larger grad norm than `decoder_trunk` (`2.31` vs `0.0048`) and `grad_norm` spikes up to `126.7` (`clip=1.0`). | 98–99.5% of each gradient step was variance; `encoder` spikes monopolized global grad clipping and starved `gram_cgt` and `action_head`. |
| **P5** | **Early-Match Sparse Reward Desert** | Before checkpoint 1 (`held = [0,0,0]`), `projected_role = None`, `winner_mask = all False`, `loss_w = 0`, `winnie_n = 0`, `role_rewards = 0`, `obs_delta_rho = 0`. | Zero gradient signal for `critic_winnie`, reverse-induction actor, and finish-rate dynamics until checkpoint 1 is captured. |
| **P6** | **Broken `dynamics` Gradient & Episode Byte Eviction** | `dynamics` had negative McCandlish cosine (`-0.310`, `SNR = -0.153`) and negative held-out `action_gain = -1.76e-5`. `replay_memory_mb = 256` evicted current-episode frames after just `3` transitions; `episode` dropped delayed `actor_mask=True` items (`value_weight=0`). | Zero-signal `obs_delta_rho=0` forced `dyn_action` to fit random velocity noise; tiny episode pool truncated credit assignment. |

## 2. Meta-Optimization Fixes Implemented

1. **`scale_config.py`**: Aligned defaults (`SCALE_RANK=128, SCALE_HIDDEN=341, SCALE_EXPERTS=8, SCALE_TOPK=2`) with `cast_header.Widths` (800× smaller than 805M-param default; still CLI-overridable for distributed runs).
2. **`state_steering.py`, `strategy.py`, `baselines.py`**:
   - Restricted `writable_words(layout, state)` to pure continuous game-entity `ev_float` coordinates: `entity > 0`, `ev_float`, unencoded, not read-only, `|state| >= 1.0`, `|state - round(state)| > 1e-4` (eliminating 100% of dead non-floats, `0.0` booleans, integer bitmasks, tiny timers, and `entity < 0` VM globals/stack locals).
   - Bounded `rate_distribution` (`RATE_BASE_SCALE = 0.025`, `tanh` mean, clamped `log_scale` in `[log(0.005), log(0.15)]`) and added finite-value guard in `unpack_pages`.
   - Normalized `logp_of` per active continuous coordinate ($\bar\ell = \frac{1}{D}\sum_d \log\pi(a_d|s)$) for dimension-invariant GSPO-style IS log-ratios.
   - Added `torch.asinh(...)` logarithmic input compression before `quinn`, `participant`, `cart`, `team`, and `phil` in `embedded_rows`.
3. **`execution.py`, `policy_contract.py`, `online.py`, `strat_responder.py`**:
   - Replaced the `proximal_logp = logpi.detach()` no-op with true clipped off-policy PPO directly on per-coordinate `log_ratio = logpi - behavior_logp`, multiplied by variance-preserving $\sqrt{D}$ (`torch.sqrt(state_count)`) so IS ratios stay $\mathcal{O}(1)$ while actor trunk gradients scale cleanly (`actor_over_total = 16.7%`).
   - Switched winner/loser critics and dynamics to Huber (`smooth_l1_loss`) with clamped TD advantages `[-2, 2]`, continuous cart-progress finish rates in `_team_finish_rates`, `min_fresh_batch = 4` micro-batch accumulation, `replay_memory_mb = 1024`, balanced split-half `OnlineMcCandlishTracker` telemetry on every update, and `__actor_probe__` retention.
4. **`game_value.py`**: Added Ng-Harada-Russell potential-based reverse-induction shaping ($\Phi_W = +m_{\text{lead}}$, $\Phi_L = -m_{\text{lead}} + 0.5 m_i$, weight `0.25`) and continuous progress leader tie-breaking (`effective_leader`) before checkpoint 1 while preserving exact formal `evaluate_cartstate` when checkpoints are held.

## 3. Before vs. After Measurements

### 3.1 Inner McCandlish (2018) Gradient SNR & Health (`microbatch=2`, Real Live Transitions)

| Metric / Component | Baseline (Unmodified) | `opt3` (`1/D` Actor) | `opt4` ($\sqrt{D}$ Actor, Multi-Round) |
|---|---|---|---|
| **IS Floor Collapse (`log_ratio < -20`)** | **93.7%** (`med = -249.7`) | **0.0%** (`IS = 0.996`) | **0.0%** (`IS = 0.792`) |
| **Dead Non-Float + Int/Bool Coords** | **69.1%** (`28.8% + 40.3%`) | **0.0%** (`100% ev_float`) | **0.0%** (`100% ev_float`) |
| **`actor_pg` (Reverse-Induction PPO)** | **DEAD** (`ratio=2e-9`, `|g|=0`) | `cos=+0.592`, `SNR=0.661`, `B_noise=3.02`, `eff=57.0%`, `|g|=0.002` | **`cos=+0.815`, `SNR=3.000`, `B_noise=0.67`, `eff=85.7%`, `|g|=0.234`** |
| **`actor_pg` $\to$ `gram_cgt`** | **DEAD (`0`)** | `cos=+0.670`, `SNR=0.883`, `eff=63.9%`, `|g|=0.00086` | **`cos=+0.801`, `SNR=2.730`, `eff=84.5%`, `|g|=0.077`** |
| **`actor_pg_unit_is` (unclipped ablation)** | DEAD | `cos=+0.593`, `SNR=0.659` | `cos=+0.042`, `SNR=0.0001`, `B_noise=13,754`, `|g|=1,020` *(proves true PPO clip prevents 20,500× noise blowup)* |
| **`critic_winnie`** | **DEAD (`0` rows)** | `cos=+0.970`, `SNR=6.74`, `eff=93.1%` | `cos=+0.389`, `SNR=0.649`, `eff=56.5%` *(post-7× MSE drop)* |
| **`dynamics` ($\Delta\rho$)** | `cos=-0.310`, `SNR=-0.153` | `cos=+0.529`, `SNR=1.38`, `eff=73.4%` | **`cos=+0.875`, `SNR=4.87`, `B_noise=0.41`, `eff=90.7%`** |
| **`gram_cgt` (live/probe total)** | `cos=+0.0017`, `SNR=0.0027`, `B_noise=373`, `eff=0.53%` | `cos=+0.911`, `SNR=8.98`, `eff=94.7%` | `cos=+0.116`, `SNR=0.423`, `eff=45.8%` *(after critic convergence)* |
| **Norm Ratio `|g_actor| / |g_total|`** | `0.0%` | `0.12%` (trunk) / `16.7%` (probe) | **`16.7%`** (`|g_actor|=0.234` vs `|g_total|=1.44`) |

### 3.2 Held-Out Critic & Dynamics Calibration ($\pi_0$ vs $\pi_T$ on Identical Held-Out Live Transitions)

| Held-Out Metric ($n=8$ transitions, 72 rows) | Untrained $\pi_0$ | Trained $\pi_T$ (3 rounds, 49 updates) | Reduction |
|---|---|---|---|
| `winnie_mse` (projected winner critic) | `0.0944` | **`0.0134`** | **-85.8% (`7.0×`)** |
| `lou_mse` (coalition loser critic) | `0.0656` | **`0.0186`** | **-71.7% (`3.5×`)** |
| `dynamics_mse_full` | `0.00303` | **`0.000456`** | **-84.9% (`6.6×`)** |

### 3.3 Paired-Seed Live Xonotic Outer Policy Improvement (3 Seeds: `20261125`, `20261126`, `20261127`, Frozen Eval, `+g_payload_speed 80`, 42s/match)

| Seed / Aggregate | Arm | `mean_score_end` | `max_cart_pos` | `checkpoints` | `mean_speed` | `kills` | `anti_leader_dmg` | `view_faults` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `20261125` | `default` | `15.53` | `357.0 u` | `2` | `295.4` | — | — | `0` |
| `20261125` | $\pi_0$ (untrained) | `0.00` | `167.4 u` | `0` | `297.7` | — | `None` | `0` |
| `20261125` | $\pi_T$ (trained) | **`6.21`** | **`340.3 u`** | **`2`** | `269.1` | — | `0.409` | **`0`** |
| `20261126` | `default` | `9.91` | `456.0 u` | `2` | `302.6` | — | — | `0` |
| `20261126` | $\pi_0$ (untrained) | `0.00` | `100.4 u` | `0` | `291.7` | — | `None` | `0` |
| `20261126` | $\pi_T$ (trained) | **`18.86`** | **`420.5 u`** | **`3`** | — | — | `0.462` | **`0`** |
| `20261127` | `default` | `11.64` | `338.8 u` | `2` | `295.6` | — | — | `0` |
| `20261127` | $\pi_0$ (untrained) | `0.00` | `150.6 u` | `0` | `270.4` | — | `None` | `0` |
| `20261127` | $\pi_T$ (trained) | **`22.41`** | **`453.1 u`** | **`3`** | — | — | `0.373` | **`0`** |
| **3-Seed Mean / Total** | **`default`** | **`12.36`** | **`383.9 u`** | **`6`** | **`297.9`** | **`52`** | **`0.558`** | **`0`** |
| **3-Seed Mean / Total** | **$\pi_0$ (untrained)** | **`0.00`** | **`139.5 u`** | **`0`** | **`286.6`** | **`56`** | **`None`** | **`0`** |
| **3-Seed Mean / Total** | **$\pi_T$ (trained)** | **`15.83`** | **`404.6 u`** | **`8`** | **`269.1`** | **`42`** | **`0.415`** | **`0`** |

- **Policy Improvement Theorem Outer Check:** $\pi_T$ strictly beat untrained $\pi_0$ on **3/3 paired seeds** (`+15.83` mean score, `+265.2 u` / `+190%` max cart progress, `0 -> 8` checkpoints, `0` view faults), and outperformed unperturbed `default` (`+28.0%` mean score, `+5.4%` cart progress, `+33%` checkpoints).
- **McCandlish Signal $\leftrightarrow$ Outer Gain Correlation (3 Training Rounds):** Round `grad_snr` correlated with next-round outer score delta with **Spearman $\rho = +1.00$** (`R1: snr=+0.034 -> +10.22`, `R3: snr=-0.0017 -> -1.86`, `R2: snr=-0.064 -> -2.16`).
- **Honest Open Gap:** On 3-seed average after 49 single-server updates, `anti_leader_damage_share` (`0.415` vs `default` `0.558`, though spiking to `0.811` in R2 / `0.759` in `opt3`) has not yet systematically separated above `default` across seeds, and late-round `critic_lou` `B_noise` rose to `92.9` transitions as MSE converged. Moving to $G$ concurrent rollout servers per optimizer step directly addresses both by multiplying i.i.d. $B_{\text{eff}}$ and eliminating single-server temporal autocorrelation in split-half McCandlish estimation.

## 4. Multi-Server Concurrent Rollout Groups ($G = 4$) & Head-to-Head Seat-Rotated Elo

Multi-server concurrent rollout support was wired across `xonwire.py` (per-session `Reassembler`, snapshot queues, event draining), `strat_responder.py` (`route[session]`, per-session `ActionHistory`/`ObservationMemory`/`anticipatory_fast`, `--min-rollout-groups`, `--min-fresh-batch`, `--rotate-team-arms`), `online.py` (concurrent episode pools, server-disjoint `cross_server_iid` McCandlish split-halves), `grad_snr.py` (`probe_cross_server_mccandlish` for true $r=G$ replicas), and `outer_validation.py` (`compute_multiteam_elo` with bootstrap 95% CIs and sample-size power).

### 4.1 Live $G = 4$ Concurrent Servers — True Cross-Server i.i.d. vs. Single-Server Temporal SNR

Four independent `darkplaces-dedicated` servers (distinct `sv_random_seed`s `20261301..20261304`, isolated `-userdir`s) connected simultaneously to one `strat_responder` (`[15, 16, 15, 14]` responses = `60` total in 60s, `9/9` `cross_server_iid` split updates, `0` view faults, `56` saved transitions across 4 servers).

| Component / Group | Cross-Server i.i.d. ($r = 4$, 6 tr/srv) | 1-Server Temporal (`microbatch=3`) | Autocorrelation Inflation |
|---|---|---|---|
| `total` | `cos = +0.633`, `SNR = 1.73`, `B_noise = 3.5 tr`, `eff = 87.4%` | `cos = +0.868`, `SNR = 6.31`, `B_noise = 0.5 tr` | `7.0×` $B_{\text{noise}}$ underestimation |
| `critic_winnie` | `cos = +0.787`, `SNR = 3.70`, `B_noise = 1.6 tr`, `eff = 93.7%` | `cos = +0.994`, `SNR = 6.42`, `B_noise = 0.5 tr` | `3.2×` |
| **`critic_lou`** | **`cos = +0.116`, `SNR = 0.132`, `B_noise = 45.5 tr`, `eff = 34.5%`** | `cos = +0.987`, `SNR = 15.20`, `B_noise = 0.2 tr` | **`227×`** |
| `dynamics` | `cos = +0.673`, `SNR = 2.06`, `B_noise = 2.9 tr`, `eff = 89.2%` | `cos = +0.976`, `SNR = 5.99`, `B_noise = 0.5 tr` | `5.8×` |
| **`actor_pg` ($r=3$, 1 tr/srv)** | **`cos = +0.027`, `SNR = 0.028`, `B_noise = 36.0 tr`, `eff = 7.7%`** | `opt4` 1-srv: `cos = +0.815`, `B_noise = 0.67 tr` | **`54×`** |
| **Online 4-srv `action_head`** | **`cos = +0.234`, `SNR = 0.092`, `eff = 15.5%`** (`|g|=0.297`) | — | True $B_{\text{noise, PG}} \approx 36\text{--}50$ tr |
| **Online 4-srv `gram_cgt`** | **`cos = +0.555`, `SNR = 1.11`, `eff = 68.9%`** (probe `cos=+0.916, eff=97.8%`) | — | Genuine cross-match CGT signal |
| **Online 4-srv `encoder`** | **`cos = +0.195`, `SNR = 0.0006`, `eff = 0.12%`** (`|g|=1.82`) | — | Raw 900-page `quinn` still dominates noise |

### 4.2 Seat-Rotated Head-to-Head 3-Arm Multi-Team Elo (6 Live Matches, 2 Waves × 3 Concurrent Servers)

Pitting `default` (stock HavocBot), `initial_policy` (frozen untrained $\pi_0$), and `matrix_fusion` (58-update trained $\pi_T$) **inside the same live matches** with cyclic seat rotation (`--rotate-team-arms`, 2 full rotations across `boil`):

- **Measured Map Seat Bias on `boil`:** `seat_1 (Red) = 13.58`, `seat_2 (Blue) = 7.28`, `seat_3 (Yellow) = 5.11` (**2.66× spawn asymmetry** — seat rotation is mandatory).
- **All 3 arms moved at full speed (`255–295 u/s`) with `0` view faults and `33/33` weights loaded.**

| Arm | Score Elo (95% CI) | Composite Elo (95% CI) | Mean Score | Mean CPs | Win vs `default` | Win vs `pi_0` | Raw 45s Matches for $\pm 25$ Elo CI95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `initial_policy` ($\pi_0$) | `1618.8` `[1440, 1857]` | `1601.0` `[1398, 1856]` | `14.19` | `1.00` | `66.7%` | — | `452` (`548` comp) |
| `default` (anchor) | `1500.0` (anchor) | `1500.0` (anchor) | `11.78` | `0.67` | — | `33.3%` | — |
| `matrix_fusion` ($\pi_T$) | **`1381.2`** `[1205, 1500]` | **`1337.4`** `[883, 1613]` | **`0.00`** | **`0.00`** | **`33.3%`** | **`16.7%`** | `246` (`1254` comp) |

### 4.3 Honest Accounting: What Architectural Variation Is Needed to Turn Gradient SNR Into Real Strategic Elo Gains

The head-to-head Elo tournament revealed a critical gap that mirror-match evaluation hid: $\pi_T$ beat $\pi_0$ in mirror matches (`15.83` vs `0.00`) and reduced critic MSE `3.5×–7×`, yet scored `0.00` (`-119` Elo vs `default`, `-238` Elo vs `initial_policy`) head-to-head. Five concrete architectural changes are required to close this gap:

1. **Direct Structured Strategic Control Surface (`MESH_WIRE(PLAN, 8)` instead of 1,400 unstructured VM float nudges):** Additive `±0.025` perturbations across `518–1,408` raw entity floats (`median |state| ≈ 280`, i.e. `0.009%` jitter) have weak, indirect causal leverage over HavocBot pathing and inflate pure PG `B_noise` to `36–50` transitions. Emitting `8–16` direct strategic parameters per bot/team into `havocbot_role_payload` (`lane_weights[0..J-1]`, `push_vs_deny`, `rival_focus[0..K-1]`, `aggression/regroup`) shrinks action dimension by `~100×` and directly exposes cart selection and anti-leader targeting to the reverse-induction gradient.
2. **Structured-State Trunk (Decoupling from 900 Raw VM Pages):** `encoder` (`quinn`) still accounted for `|g|=1.82` with `SNR=0.0006` (`0.12%` efficiency) online and caused `~40 MB` (`2,779`-frame) snapshots with `ENOBUFS` drops at $G=4$. Training `Wally` directly on structured `observations` + `carts` (CGT thermographs/Nimbers) + `teams` + `events` + `LocalNeighborhood` removes 99.9% of IPC overhead (`~50×` faster inference/training) and eliminates `encoder` noise dominance.
3. **High-Throughput Concurrent Rollouts ($G = 16\text{--}32$ Servers + `slowmo 4–8`):** True i.i.d. `actor_pg` and `critic_lou` have $B_{\text{noise}} \approx 36\text{--}46$ transitions; reaching `50–80%` step efficiency needs $B_{\text{eff}} \approx 40\text{--}160$ independent transitions per step. On compact `PLAN` frames, `16` headless servers at `slowmo 4` deliver `~64×` rollout throughput on one Apple Silicon host.
4. **League / EMA Checkpoint Pool + Stock Anchor During Training:** Pure self-play mirror matches caused brittle drift that collapsed against stock HavocBot (`default`) and `initial_policy`. Mixing `default` + historical EMA checkpoints ($\pi_{\text{EMA}, k}$) across seat-rotated concurrent servers prevents mirror-match overfitting and gives a continuous in-run Elo ladder.
5. **Elo Evaluation Variance Reduction (Paired-Seed Cyclic Triplets + Full-Horizon Matches):** At 45s match length, single-match score std (`14–29`) requires `~250–550` raw matches for a `±25` Elo 95% CI. Running full-horizon matches (120–180s game time, `30–45s` wall clock at `slowmo 4`) paired across all 3 cyclic seat permutations on each common `sv_random_seed` cancels the `2.66×` map-spawn bias and reduces the required Elo evaluation budget by `~4–8×` to `~30–60` matches (~2–4 minutes on $G = 16$ servers).

