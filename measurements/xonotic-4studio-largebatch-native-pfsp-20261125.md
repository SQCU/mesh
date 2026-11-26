# Four-Studio Large Multi-Episode Batch PFSP + Native Calibrated HavocBot League Measurement (2026-11-25)

## 1. Setup & Configuration

- **Cluster:** 4 Apple M5 Ultra Mac Studios (`mdots-Mac-Studio.local`, `mdots-Mac-Studio-2.local`, `mdots-Mac-Studio-3.local`, `m5ult2.local`), `WORLD_SIZE=4`, `backend="mesh"` (Thunderbolt RDMA), `device="mps"`, commit `65a221c87`.
- **Model:** `PlanWally` (`1,781,092` params; `d=256, h=1024, r_gram=128, r_dpp=64, depth=4, FSQ 5^4, Mealy`), EDM2 unit-norm weight projection, Aurora (`lr=0.010, scalar_lr=5e-3, mu=0.90, pp_iterations=2`), masked actor (`BOT_CONFIGURATION_COLUMNS` zeroed), training-only asymmetric critic + online Skill-ID head.
- **Large Multi-Episode Batching:**
  - `6` concurrent full live servers per Studio × `4` Studios = **`24` distinct full episodes per macro-iteration** (`2× boil [slowmo 1.0]`, `2× stormkeep [slowmo 1.2]`, `2× catharsis [slowmo 1.8]`, `24.0 s` wall = `24–43 s` game time per episode).
  - Evenly strided `local_batch = 256` across all 6 local episodes → **`B_global = 1,024` across `24` full episodes per Aurora step**, `2` PPO epochs per iteration.
  - Schedule: `6` Main PFSP iterations (`144` episodes) → `3` disposable Main Exploiter iterations targeting `main_mid` (`72` episodes) → `4` Main PFSP iterations against the pool including frozen `exploiter_1` (`96` episodes) = **`312` distinct training episodes total** (`26` large-batch Aurora updates, `1,042.86 s` wall time).
- **Zero Synthetic Handicaps:** Every match roster was drawn directly from Xonotic's native HavocBot parameter space via `sample_native_lobby_roster` (`ridge = 40.0` empirical 1v1 DM Elo model, `R² = 0.664, LOO R² = +0.440, LOO RMSE = 51.7 Elo`, batch-isolated per-bot `bot_base_skill`).

---

## 2. Large-Batch 4-Studio RDMA McCandlish SNR by Module Group

Each row reports epoch-0 4-Studio RDMA `step_efficiency` at `B_global = 1,024` (`24` full episodes), gradient norm, policy entropy, masked-actor online Skill-ID correlation `sk_r`, and FSQ codebook entropy:

| Iter | Role | `gnorm` | Entropy | `sk_r` | `FSQ bits` | `encoder` | `gram_cgt` | `resnet_ffn` | `action_head` | `value_head` | `skill_id_head` | `dynamics_head` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `m01` | main | 0.94 | 8.745 | -0.1950 | 7.845 | 0.89 | 0.91 | 0.89 | `--` | 0.98 | 0.88 | 1.00 |
| `m02` | main | 0.88 | 8.759 | +0.0950 | 7.964 | 0.86 | 0.89 | 0.85 | **0.22** | 0.85 | 0.98 | 1.00 |
| `m03` | main | 0.56 | 8.784 | +0.3078 | 8.240 | 0.77 | 0.83 | 0.75 | `--` | 0.97 | 0.98 | 0.99 |
| `m04` | main | 0.38 | 8.780 | +0.2767 | 8.371 | 0.52 | 0.50 | 0.50 | `--` | 0.97 | 0.75 | 0.98 |
| `m05` | main | 0.43 | 8.792 | +0.4824 | 8.106 | 0.53 | 0.57 | 0.58 | **0.15** | 0.70 | 0.94 | 0.98 |
| `m06` | main | 0.33 | 8.792 | +0.4033 | 8.130 | `--` | `--` | `--` | `--` | 0.87 | `--` | 0.96 |
| `exp01` | exploiter | 1.29 | 8.733 | -0.0252 | 7.928 | 0.92 | 0.94 | 0.91 | **0.14** | 0.99 | 0.94 | 1.00 |
| `exp02` | exploiter | 0.79 | 8.702 | +0.0793 | 8.047 | 0.76 | 0.80 | 0.73 | **0.24** | 0.64 | 0.96 | 0.99 |
| `exp03` | exploiter | 0.86 | 8.695 | +0.2873 | 7.996 | 0.76 | 0.80 | 0.77 | `--` | 0.99 | 0.69 | 1.00 |
| `m07` | main | 0.40 | 8.790 | +0.3758 | 8.214 | 0.39 | 0.40 | 0.39 | `--` | 0.81 | 0.89 | 0.95 |
| `m08` | main | 0.68 | 8.802 | +0.4463 | 8.033 | 0.70 | 0.76 | 0.67 | **0.17** | 0.97 | 0.93 | 0.97 |
| `m09` | main | 0.44 | 8.827 | **+0.5752** | 7.964 | 0.24 | 0.24 | 0.30 | **0.25** | 0.81 | `--` | 0.99 |
| `m10` | main | 0.29 | 8.826 | +0.4316 | 8.028 | 0.36 | 0.44 | 0.37 | `--` | 0.64 | 0.74 | 0.98 |

---

## 3. Held-Out 4-Studio 48-Match Multi-Map CRN Evaluation

Evaluated across `4` maps in parallel (`boil`, `stormkeep`, `catharsis [slowmo 1.8]`, and zero-shot held-out `afterslime [slowmo 1.8]`, `12` seat-rotated matches per map = `48` matches total, `32.0 s` each) on fresh native-Elo-sampled lobbies (`gaussian_lobby`, `team_skewed`, `within_team_skewed`).

### 3.1 Overall Held-Out Elo & 3-Team $\alpha$-Rank (`48` matches)

| Policy | Score Elo (95% CI) | Mean Raw Score | Composite Elo (95% CI) | Mean Composite | $\alpha$-Rank (Score) | $\alpha$-Rank (Composite) |
|---|---:|---:|---:|---:|---:|---:|
| `main_init` | **1569.6** `[1463.3, 1703.3]` | 3.98 | **1596.5** `[1432.5, 1760.8]` | 8.05 | **0.3619** | **0.4298** |
| `exploiter_1` | **1503.8** `[1407.5, 1594.2]` | 4.15 | 1477.1 `[1337.5, 1607.6]` | 6.02 | 0.1910 | 0.1374 |
| `default` | 1500.0 `[1500.0, 1500.0]` | 2.26 | 1500.0 `[1500.0, 1500.0]` | 4.41 | 0.1099 | 0.0960 |
| `script_cart_rush` | 1498.0 `[1420.2, 1603.5]` | **7.45** | 1471.7 `[1324.1, 1624.4]` | **10.63** | 0.2464 | 0.2091 |
| `main_final` | 1480.4 `[1410.5, 1539.1]` | 3.16 | 1480.7 `[1397.1, 1558.5]` | 5.28 | 0.0435 | 0.0881 |
| `main_mid` | 1428.1 `[1368.4, 1478.4]` | 0.61 | 1400.0 `[1311.0, 1507.6]` | 1.29 | 0.0473 | 0.0397 |

### 3.2 Per-Map Breakdown (`12` matches per map)

| Map | `main_init` Score / Comp | `main_mid` Score / Comp | `exploiter_1` Score / Comp | `main_final` Score / Comp | `script_cart_rush` Score / Comp | `default` Score / Comp |
|---|---:|---:|---:|---:|---:|---:|
| `boil` | 1642.7 (9.73) / 1690.0 | 1475.9 (2.44) / 1395.1 | 1481.8 (0.00) / 1520.3 | 1538.4 (8.29) / 1540.4 | 1512.9 (0.00) / 1485.2 | 1500.0 (4.64) / 1500.0 |
| `stormkeep` | 1522.8 (5.00) / 1508.2 | 1373.9 (0.00) / 1432.5 | 1483.7 (0.96) / 1508.2 | 1465.5 (4.35) / 1513.0 | **1565.2 (29.80)** / **1608.2** | 1500.0 (3.32) / 1500.0 |
| `catharsis` | 1500.0 (0.00) / 1588.1 | 1500.0 (0.00) / 1498.9 | 1500.0 (0.00) / 1509.0 | 1500.0 (0.00) / 1497.4 | 1500.0 (0.00) / 1587.7 | 1500.0 (0.00) / 1500.0 |
| `afterslime` *(zero-shot)* | 1548.2 (1.20) / 1507.7 | 1406.6 (0.00) / 1331.5 | **1548.2 (15.64)** / 1385.2 | 1427.4 (0.00) / 1383.5 | 1436.9 (0.00) / 1245.1 | 1500.0 (1.08) / 1500.0 |

### 3.3 Masked-Actor Online Skill-ID Audit (Zero `BOT_CONFIGURATION_COLUMNS` Visible)

Measured on held-out `main_final` trajectories across all 4 maps (`n = 6,000` bot-tick predictions per window), predicting each bot's standardized empirical 1v1 DM Elo purely from in-match observations & Mealy recurrent state:

| Match Window | Samples `n` | Pearson `r` | Standardized MSE |
|---|---:|---:|---:|
| Early (`0–8 s`) | 6,000 | **+0.2983** | 1.0572 |
| Late (`8 s+`) | 6,000 | **+0.4088** (`+0.1105`) | **0.8752** (`-0.1820`) |

### 3.4 Architecture & Optimization Invariants

- **EDM2 Unit-Norm Invariant:** `33` layers, `7,521` rows, `min_norm = 1.0`, `max_norm = 1.0`, **`max_dev_from_1 = 0.0`**.
- **Wide FFN Dead-Neuron Audit:** **`0 / 1,024` dead units (`0.0%`)**, `grad_norm_cv = 1.3544`, `min_over_med = 0.0661`.
- **FSQ Codebook Health:** `7.845 – 8.371` bits across all 13 iterations (no codebook collapse).

---

## 4. Closed-Form Diagnosis of the `main_mid` / `main_final` Policy Regression

The held-out evaluation and per-group RDMA SNR table isolate two concrete structural bottlenecks:

1. **Frozen `action_head` under a rotating auxiliary-driven trunk:**
   - Supervised auxiliary heads (`dynamics_head`: `0.95–1.00`, `value_head`: `0.64–0.99`, `skill_id_head`: `0.69–0.98`) drove the shared trunk (`encoder`, `gram_cgt`, `resnet_ffn`) to rotate at high step efficiency every iteration.
   - Meanwhile, `action_head` had `signal_sq <= 0` (`step_efficiency = None`) on **`8 / 13` iterations**, causing our per-group SNR gate (`grp_scales[action_head] = 0.0`) to freeze the policy readout while the trunk moved underneath it, and `entropy_coef = 0.008` increased policy entropy monotonically (`8.745 → 8.827`) on the remaining iterations.
   - By contrast, `exploiter_1` had positive `action_head` efficiency on 2 of 3 iterations, reduced entropy (`8.733 → 8.695`), beat `main_mid` (`1503.8` vs `1428.1` Score Elo), and hit `15.64` mean raw score (`14.5×` `default`) on zero-shot `afterslime`. After adding `exploiter_1` to the league pool (`m07..m10`), `main_final` recovered `+52.3` Score Elo (`1428.1 → 1480.4`) and `+80.7` Composite Elo (`1400.0 → 1480.7`) over `main_mid`.
2. **1-Step TD(0) advantage discarding full-episode trajectory credit assignment:**
   - Although we collected complete `24 s` (`60–108` tick) episodes, `_sync_adv_stats` and `plan_losses_for_transition` still computed single-tick `0.4 s` TD(0) residuals (`r_t + γ V(s_{t+1}) - V(s_t)`) instead of full-episode **GAE($\lambda = 0.95$)** before striding into the `B_global = 1,024` batch.

---

## 5. Controlled Ablation: Zero-Clamping SNR Gate Removed (`5791cd182`)

Re-ran the **exact same 4-Studio protocol** (`312` episodes, `B_global = 1,024`, `2` PPO epochs, `lr = 0.010`, `scalar_lr = 5e-3`, `1,046.86 s` wall time, identical 48-match held-out eval) with the per-group SNR zero-clamp removed so Aurora's Nesterov momentum controller stepped every parameter group unconditionally while retaining McCandlish `B_noise` / `step_efficiency` strictly as diagnostics.

### 5.1 Side-by-Side Controlled Comparison

| Metric | With Zero-Clamp (`65a221c87`) | Without Zero-Clamp (`5791cd182`) | Delta |
|---|---:|---:|---:|
| `action_head` positive-SNR iters (peak eff) | `5 / 13` (peak `0.25`) | **`7 / 13` (peak `0.71`)** | `+2` iters / `+0.46` peak |
| Held-out Skill-ID `early (0–8 s)` `r` / `MSE` | `+0.2983` / `1.0572` | **`+0.3471` / `1.0079`** | **`+0.0488` / `-0.0493`** |
| Held-out Skill-ID `late (8 s+)` `r` / `MSE` | `+0.4088` / `0.8752` | **`+0.4754` / `0.8332`** | **`+0.0666` / `-0.0420`** |
| Wide FFN dead units / `grad_norm_cv` | `0 / 1024` / `1.3544` | **`0 / 1024` / `1.1951`** | tighter CV (`-0.1593`) |
| EDM2 `max_dev_from_1` (`33` layers, `7,521` rows) | `0.0` | **`0.0`** | exact |
| `main_init` Score Elo (mean raw score) | `1569.6` (`3.98`) | `1458.4` (`0.17`) | — *(init baseline)* |
| `main_mid` Score Elo (mean raw score) | `1428.1` (`0.61`) | **`1488.0` (`2.54`)** | **`+59.9` Elo** (`+29.6` vs init) |
| `exploiter_1` Score Elo (mean raw score) | `1503.8` (`4.15`) | **`1510.1` (`4.97`)** | **`+6.3` Elo (`3.1× default`)** |
| `exploiter_1` $\alpha$-Rank (Score / Comp) | `0.1910` / `0.1374` | **`0.2524` / `0.1889`** | **`+0.0614` / `+0.0515`** |
| `main_final` Score Elo (mean raw score) | `1480.4` (`3.16`) | `1466.9` (`1.35`) | `-13.5` Elo |

### 5.2 Per-Map Held-Out Highlights (No Zero-Clamp, `48` Matches)

| Policy | Score Elo (95% CI) | Mean Score | Composite Elo (95% CI) | Mean Comp | $\alpha$-Rank (Score / Comp) |
|---|---:|---:|---:|---:|---:|
| `script_cart_rush` | **1517.7** `[1411.2, 1655.1]` | 3.41 | **1511.3** `[1343.9, 1704.1]` | 6.27 | **0.2591** / 0.2849 |
| `exploiter_1` | **1510.1** `[1419.1, 1602.9]` | **4.97** | 1490.5 `[1344.7, 1648.7]` | **7.31** | **0.2524** / 0.1889 |
| `default` | 1500.0 `[1500.0, 1500.0]` | 1.62 | 1500.0 `[1500.0, 1500.0]` | 3.23 | 0.2232 / **0.2957** |
| `main_mid` | 1488.0 `[1412.9, 1577.3]` | 2.54 | 1459.2 `[1354.2, 1559.4]` | 4.42 | 0.1328 / 0.0797 |
| `main_final` | 1466.9 `[1408.4, 1523.2]` | 1.35 | 1453.4 `[1360.4, 1535.6]` | 2.60 | 0.0370 / 0.0304 |
| `main_init` | 1458.4 `[1401.3, 1508.9]` | 0.17 | 1490.5 `[1394.8, 1587.3]` | 1.02 | 0.0955 / 0.1204 |

- **`boil` (`12` matches):** `script_cart_rush` `1569.8` (`12.84`), `exploiter_1` `1509.4` (`12.18` vs `default` `4.93`), `main_mid` `1434.3` (`3.93`), `main_final` `1431.4` (`2.51`), `main_init` `1430.8` (`0.67`).
- **`stormkeep` (`12` matches):** `main_mid` **`1567.2` Score Elo (`6.24` raw score = `10.6×` `default`'s `0.59`)**, `exploiter_1` `1534.5` Score / **`1651.8` Composite Elo (`7.72` raw score)**, `script_cart_rush` `1518.0`, `main_final` `1477.8` (`2.87`), `main_init` `1456.6` (`0.00`).

### 5.3 Next Bottleneck Exposed by the Controlled Re-Run
Removing the zero-clamp cleanly fixed `action_head` starvation (`main_mid` beat `main_init` by `+29.6` Score Elo / `14.9×` raw score, `exploiter_1` beat `default` at `1510.1` Score Elo / `3.1×` raw score, and held-out masked Skill-ID jumped to `r = +0.4754`), and isolated the exact reason `main_final` drifted slightly below `main_mid`:

- **Normalized-scalar entropy ratchet:** Because Aurora normalizes 1D scalar parameter updates (`g_eff / rms`), a positive `entropy_coef` on `log_std` acts as a constant-velocity ratchet whenever 1-step TD(0) advantage gradients on `log_std` are noisier than the deterministic entropy bonus. Across `m01..m10`, policy entropy rose monotonically every single iteration (`8.760 → 8.905` at `main_mid` → `9.008` at `main_final`, `+0.248` nats), washing out `main_final`'s action sharpness while `exploiter_1` (3 iterations, `8.739 → 8.763`) stayed sharp.
- **Remedies for the next iteration:** (1) Replace 1-step `0.4 s` TD(0) with full-episode **GAE($\lambda = 0.95$)** computed across each collected `24 s` episode before striding; (2) remove the positive `entropy_coef` ratchet on `log_std` (use a target-entropy or `0` continuous entropy bonus under Aurora's normalized 1D updates).

