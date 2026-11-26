# Four-Studio 1v1 DM Mechanical Calibration, Masked-Actor Online Skill-ID & PFSP/$\alpha$-Rank League (2026-11-25)

## 1. 4-Studio Parallel 1v1 Deathmatch Mechanical Skill Calibration

Calibrated 9 per-bot HavocBot kinematic profiles (`bots.txt`, single-TAB format via [`xonotic/solver/strat/mech_calibration.py`](../xonotic/solver/strat/mech_calibration.py)) across **44 live 1v1 DM duels in `45.6 s`** across all 4 Apple M5 Ultra Mac Studios (`boil`, `skill 7`, `slowmo 3.0`, anchor `mech_stock = 1000.0`):

| Profile | Mechanical Offsets (`aim/move/dodge/...`) | 1v1 DM Elo `[95% CI]` | Mean Frags / Duel |
|---|---|---:|---:|
| `mech_turret` | `aim +3.5, offset +3.0, weap +2.5, move -2.5` | **`1106.3`** `[931.0, 1276.5]` | `2.0` |
| `mech_veteran` | `+1.2` across axes | **`1106.3`** `[936.4, 1290.4]` | `1.9` |
| `mech_aimbot` | `+3.5 / +4.0` across axes | **`1079.0`** `[901.5, 1248.3]` | `1.8` |
| `mech_mover` | `move/dodge +3.5, aim -2.5` | **`1052.3`** `[879.1, 1237.8]` | `1.8` |
| `mech_casual` | `-1.0` across axes | `1026.1` `[867.1, 1190.1]` | `1.1` |
| `mech_stock` | `0.0` (anchor) | **`1000.0`** | `1.6` |
| `mech_carry` | `+2.5` across axes | `1000.0` `[813.8, 1171.4]` | `1.6` |
| `mech_novice` | `-2.0` across axes | **`799.4`** `[658.9, 942.1]` | `0.2` (`10×` lower) |
| `mech_potato` | `-3.0` across axes | **`799.4`** `[648.5, 947.9]` | `0.3` |

---

## 2. Masked Actor + Asymmetric Critic + Online Skill-ID Head

Updated [`xonotic/solver/strat/plan_policy.py`](../xonotic/solver/strat/plan_policy.py):
- **Actor Observation Mask:** Zeroed all 13 `BOT_CONFIGURATION_COLUMNS` in `prenormalize_obs` so the actor never reads bot skill metadata and operates identically on human players and bots.
- **Asymmetric Critic (`value_priv_proj`):** Training-only privileged residual `[self_skill, own_team_mean_skill, rival_max_team_mean_skill]` into `value_winnie` / `value_lou` (omitted at inference).
- **Auxiliary Online Skill-ID Head (`skill_id_proj`):** Supervises `PlanWally`'s recurrent Mealy state `h_t` + Linear-Gram cross to predict each bot's latent mechanical skill online purely from realized combat/kinematic observables.

---

## 3. 4-Studio PFSP + Main Exploiter + FCP + Skewed-Roster Training (`702.1 s`, `92` Sync Steps, `24` Live Servers)

Trained across 5 Main PFSP waves (`16` sync steps each, `entropy_coef` decayed `0.01 → 0.0025`) + 1 dedicated **Main Exploiter** wave (`exp1`, `12` steps against `pfsp_w3 + default`), with skewed `bots.txt` rosters, physics perturbations (`baseline/fast/slow/volatile`), FCP mixed-teammate servers (`1` `default` bot inside `train`'s squad), and lane-scaled `slowmo` (`1.0` `boil`, `1.2` `stormkeep`, `1.8` `catharsis`/`afterslime`):

| Wave (Role) | Updates | Global $\eta$ | `encoder` $\eta$ | `gram_cgt` $\eta$ | `resnet_ffn` $\eta$ | `value_head` $\eta$ | `skill_id_head` $\eta$ | `dynamics_head` $\eta$ | **Online Skill-ID $r$** | FSQ Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `w1` (`main`) | `16` | **`16.1%`** | **`30.9%`** | **`34.1%`** | **`27.8%`** | `81.6%` | **`89.8%`** | `97.8%` | `+0.104` | `8.09` b |
| `w2` (`main`) | `16` | `None` | `12.7%` | `15.1%` | `12.5%` | `50.0%` | **`79.7%`** | `98.5%` | **`+0.333`** | `8.27` b |
| `w3` (`main`) | `16` | `0.6%` | `3.2%` | `4.1%` | `1.7%` | `58.6%` | `None` | `97.6%` | **`+0.538`** | `8.17` b |
| `exp1` (`exploiter`) | `12` | **`12.8%`** | **`29.4%`** | **`28.3%`** | **`25.7%`** | `84.5%` | **`81.0%`** | `97.5%` | `+0.097` | `8.08` b |
| `w4` (`main`) | `16` | `None` | `None` | `None` | `None` | `48.9%` | `None` | `97.4%` | **`+0.425`** | `8.48` b |
| `w5` (`main`) | `16` | `None` | `None` | `None` | `None` | `None` | `22.7%` | `96.5%` | **`+0.371`** | `8.35` b |

### Key Health & Skill-ID Findings
- **Masked-Actor Online Skill-ID Verified:** Without ever seeing a single `bot_*skill` column, `PlanWally`'s Mealy state `h_t` inferred latent bot mechanical skill up to **`r = +0.538`** during training and **`r = +0.212`** (`n = 4,000` early `0–8s` / `r = +0.197`, `n = 4,000` late `8s+`) zero-shot across all 4 evaluation maps including unseen `afterslime`.
- **Backbone SNR Boost:** Adding `skill_id_head` nearly doubled Wave-1 shared backbone 4-Studio McCandlish efficiency (`encoder`: `16.9% → 30.9%`, `gram_cgt`: `20.6% → 34.1%`, `resnet_ffn`: `16.7% → 27.8%`).
- **Aurora / EDM2 Invariants:** **`0 / 1,024` dead wide-FFN units (`dead_fraction = 0.0`, `min_over_med = 0.077`)**, **`max_dev_from_1 = 0.0`**, **`fsq_ent = 8.08 – 8.48` bits**.

---

## 4. 4-Studio 48-Match Multi-Map Elo, 3-Team $\alpha$-Rank & Underdog Audit

Evaluated simultaneously across all 4 Mac Studios (`boil`, `stormkeep`, `catharsis` [`slowmo=1.8`], and **zero-shot held-out `afterslime`** [`slowmo=1.8`], anchor `default = 1500.0`):

| Arm | Overall Score Elo | Overall Comp Elo | 3-Team $\alpha$-Rank (Score) | 3-Team $\alpha$-Rank (Comp) | `boil` Score (Comp) | `stormkeep` Score (Comp) | **`afterslime` Zero-Shot Score (Raw)** |
|---|---:|---:|---:|---:|---|---|---|
| **`exploiter_1`** (12-step Main Exploiter) | **`1511.4`** | `1457.4` | **`0.2073`** | `0.1593` | **`1602`** (`1519`, `14.5`) | `1451` (`1486`, `14.4`) | `1474` (`0.0`) |
| **`script_cart_rush`** (permanent anchor) | **`1501.8`** | `1441.8` | **`0.2172`** | `0.2432` | `1551` (`1460`, `9.3`) | `1299` (`1325`, `0.3`) | **`1665`** (`23.8`) |
| **`default`** (stock HavocBot) | **`1500.0`** | **`1500.0`** | `0.1795` | **`0.3900`** | `1500` (`1500`, `8.8`) | `1500` (`1500`, `11.5`) | `1500` (`1.1`) |
| `pfsp_w0` (step-0 init) | `1486.1` | `1431.5` | `0.1429` | `0.0828` | `1450` (`1443`, `0.5`) | **`1568`** (`1565`, `37.3`) | `1436` (`0.0`) |
| `pfsp_w2` (`32` steps) | `1473.4` | `1379.1` | `0.1078` | `0.0494` | `1450` (`1281`, `0.0`) | `1451` (`1448`, `7.0`) | `1511` (`0.0`) |
| **`pfsp_w5`** (`80` steps, post-exploiter) | `1472.5` | `1396.8` | `0.0663` | `0.0284` | `1463` (`1412`, `1.6`) | `1415` (`1404`, `5.2`) | **`1521`** (**`5.9`** = `5.4×` stock!) |
| `pfsp_w3` (`48` steps, pre-exploiter) | `1448.7` | `1314.3` | `0.0790` | `0.0469` | `1466` (`1314`, `0.3`) | `1387` (`1354`, `13.4`) | `1485` (`0.0`) |

### Diagnostic Takeaways for Next Iteration
1. **Non-transitivity & Exploiter Value Confirmed:** `script_cart_rush` dominated `afterslime` (`1665`) and `boil` (`1551`) but collapsed on `stormkeep` (`1299`); `exploiter_1` immediately hit `#1` Score Elo (`1511.4`, `1602` on `boil`) in 12 steps and lifted `pfsp_w5` by **`+82.5` Comp Elo (`1314.3 → 1396.8`)** once added to the league; `pfsp_w5` beat `default` by **`+21` Elo (`5.4×` raw score: `5.9` vs `1.1`)** zero-shot on unseen `afterslime`.
2. **Inter-Studio Stratification Fix Needed for `action_head` SNR:** Sampling *different* random roster skews, PFSP opponents, and physics perturbations on each of the 4 Studios added massive inter-Studio variance (driving `action_head` McCandlish $\eta \to \text{None}$ in `w2..w5`). Mirroring the *same* 6-server roster/opponent/physics stratum across all 4 Studios (differing only in match seeds — CRN stratification across ranks) will recover clean cross-Studio `action_head` SNR while preserving full domain randomization.
3. **Underdog Stress Test Calibration:** In the extreme `-307`-Elo (`10×` frag) underdog test (`[novice, novice, potato]` vs `[carry, carry, veteran]`), both `pfsp_w5` (`0.0`) and `default` (`0.033`) were spawn-locked before reaching carts; a moderate `-100..150`-Elo handicap (`[casual, casual, novice]` vs `[veteran, stock, stock]`) or mixed carry+potato roster (`[carry, potato, potato]`) is the informative strategic-compensation regime.
