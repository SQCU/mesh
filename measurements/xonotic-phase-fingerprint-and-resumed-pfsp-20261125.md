# Phase-Space & Markov-Flow Fingerprinting, Live Userspace Client Orbit, and Resumed 4-Studio PFSP Study

**Date:** 2026-11-25  
**Cluster:** 4× Apple M5 Ultra Mac Studio (`100.103.69.53`, `100.65.151.124`, `100.107.135.46`, `100.86.148.87`, Thunderbolt RDMA `backend="mesh"`) + local Apple M5 Max userspace client  
**Key Artifacts:** `xonotic/solver/strat/phase_fingerprint.py`, `xonotic/payload/tools/fingerprint_fuzz.py`, `bin/mesh-xonotic-client`, `xonotic/darkplaces-work/{cl_screen.c,cl_input.c,cl_main.c}`

---

## 1. Rebuilt Xonotic Userspace Client Runtime

Cleanly rebuilt from HEAD (`d21453d72`) and staged across local `/Users/mdot/dox/xonotic/Xonotic/` and all 4 Mac Studios (`~/xonotic-stage/userdir/data/`):

- **Engine client:** `xonotic/darkplaces-work/darkplaces-sdl` (`3.3 MB`, native `arm64` Mach-O, SDL2 + Metal OpenGL 2.1, zero missing dylibs) — wrapped by `bin/mesh-xonotic-client` and copied to `/Users/mdot/dox/xonotic/Xonotic/darkplaces-sdl`.
- **QC VMs (`gmqcc -O3 -Werror -Wall`):** `csprogs.dat` (`4.1 MB`), `menu.dat` (`1.7 MB`), `progs.dat` (`6.8 MB`, includes cart-free `g_mesh_telemetry` mutator for 1v1/2v2/3v3 DM & TDM).
- **Engine-native Viewbuffer + 4×4 Viewmatrix Recorder (`cl_screen.c`, `cl_input.c`):**
  - `cl_viewbuf_dir`, `cl_viewbuf_interval`, `cl_viewbuf_maxframes`: captures live `GL_ReadPixelsBGRA` RGB viewbuffer `.ppm` frames synced with exact `4×4` `r_refdef.view.matrix` JSONL records (`origin`, `viewangles`, `frustum_x/y`, `cl_time`).
  - `cl_orbit_yawrate`, `cl_orbit_speed`: forces `key_game`, enters observer mode, and orbits smoothly in 360° through live arenas.
- **Verified live connections:**
  - **Supermap circular orbit (`b3313_6m_fuzz` on Studio 4, `64` bots fighting across 6 stitched real stock maps):** recorded `40` synchronized `640×360` RGB viewbuffer frames + `40` `4×4` viewmatrices spanning `350.4°` (`yaw ∈ [-177.6°, 172.8°]`, `x ∈ [-203, 112]`, `y ∈ [4794, 5046]`); smoke test of rebuilt `bin/mesh-xonotic-client` confirmed **`Downloaded CSQC over UDP?: False`** (`24` frames, `156.0°` span in 14 s).
  - **Active PFSP policy-optimization rollout connectability:** during active 4-Studio RDMA training (`task-4757`), all **`24`** concurrent rollout servers registered live in the Fleet Server Browser (`http://100.102.120.16:8766/api/servers`), and our local userspace client connected over Tailscale directly into `100.103.69.53:26110` (`Boil`, `progs 50978 crc`) mid-training with zero perturbation to RDMA training or bot observations (`IS_OBSERVER` excluded from `IS_PLAYER`).

---

## 2. Cart-Free 1v1 / 2v2 / 3v3 DM vs. Payload Structural Separation (`boil`)

Using `g_mesh_telemetry 1` (zero cartlanes) vs stock 3-team Payload on identical `boil` geometry:

| Mode on `boil` | Active `256 u` Cells | $H_{\text{occ}}$ (bits) | $v_{\text{rms}}$ (`u/s`) | Straightness | Stuck | Transitions / p·s | Entropy Prod. $\sigma$ | Ally Clump `<160 u` | Engage $r$ | Combat Stack Adv | Pickups / p·min | Mean Stack |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`1v1 DM`** *(cart-free)* | `41` | `4.87` | `357.8` | `0.772` | `0.00%` | `1.49` | `0.436` | `0.00%` | `117 u` | `+9.0` | `20.8` | `112` |
| **`2v2 TDM`** *(cart-free)* | `39` | `4.95` | `361.5` | `0.785` | `0.00%` | `1.59` | `0.459` | `12.10%` | `196 u` | `+19.7` | `14.1` | `93` |
| **`3v3 TDM`** *(cart-free)* | `44` | `5.03` | `345.7` | `0.806` | `0.45%` | `1.52` | `0.553` | `9.08%` | `271 u` | `+25.7` | `12.7` | `98` |
| **`3-Team Payload`** *(stock)* | `26` | `3.33` | `289.1` | `0.727` | `2.78%` | `1.05` | `0.794` | `30.45%` | `239 u` | `+31.2` | `8.6` | `88` |

### Pairwise `PhaseSpaceFingerprint` Divergence Matrix

| Pair | `occ_jsd` | `flux_jsd` | `speed_l1` | `enemy_l1` | `sigma_ratio` | `weirdness` | Anomaly Flags |
|---|---:|---:|---:|---:|---:|---:|---|
| `3v3 TDM` vs `2v2 TDM` | **`0.066`** | **`0.176`** | `0.115` | `0.110` | `1.20` | **`0.113`** | `[]` |
| `3v3 TDM` vs `1v1 DM` | **`0.082`** | **`0.236`** | `0.099` | `0.322` | `1.26` | **`0.147`** | `[]` |
| `2v2 TDM` vs `1v1 DM` | **`0.107`** | **`0.250`** | `0.029` | `0.302` | `1.05` | **`0.144`** | `[]` |
| **`3-Team Payload` vs `1v1 DM`** | **`0.369`** | **`0.581`** | `0.279` | `0.140` | `1.80` | **`0.331`** | `['ALLY_CLUMPING', 'MAP_CAMPING_COLLAPSE', 'HIGH_OCCUPANCY_SHIFT']` |
| **`3-Team Payload` vs `2v2 TDM`** | **`0.529`** | **`0.668`** | `0.272` | `0.205` | `1.71` | **`0.460`** | `['ALLY_CLUMPING', 'MAP_CAMPING_COLLAPSE', 'HIGH_OCCUPANCY_SHIFT']` |
| **`3-Team Payload` vs `3v3 TDM`** | **`0.447`** | **`0.614`** | `0.269` | `0.207` | `1.43` | **`0.406`** | `['HIGH_OCCUPANCY_SHIFT']` |

DM/TDM modes cluster tightly (`occ_jsd = 0.066–0.107`, `0` false flags), with **`5×–8×` separation** from Payload (`occ_jsd = 0.369–0.529`).

---

## 3. Multi-Pad Anti-Telefrag B3313 Supermap Fuzz

Upgrading `b3313_supermap.py` to emit up to `3` verified walkable `info_teleport_destination` pads per portal (using native `TELEPORT_TELEFRAG_AVOID`):

| Supermap | `M / K / J / N` | Brushes / Compile | `n` (`p/Mqu²`, CV) | `v_bulk` (`u/s`) | Crossings | `Q_obs/joint` | `θ_e` | `η_⊥` | `stuck` | Telefrags |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `b3313_4m_w320_v2` | `4 / 6 / 16 / 48` | `7,616` / `2.54 s` | `4.73` (`0.38`) | `315.1` | `182` | `0.611/s` | **`1.124`** | `0.759` | `2.95%` | **`1`** *(down from 8)* |
| `b3313_6m_fuzz` | `6 / 6 / 24 / 64` | `12,537` / `3.72 s` | `3.46` (`0.54`) | `308.6` | `121` | `0.314/s` | `1.286` | `0.674` | `5.10%` | **`0`** |
| `b3313_6m_fuzz` *(orbit smoke, N=48)* | `6 / 6 / 24 / 48` | `12,537` / `3.72 s` | `2.54` (`0.41`) | `318.5` | `—` | `0.247/s` | **`1.008`** | **`0.783`** | `3.03%` | **`0`** |

---

## 4. Resumed 4-Studio RDMA PFSP Training Study with Live `PhaseSpaceFingerprint`

Resumed from Run C (`studio4-largebatch-native-gae/ckpts/main_final_ema.pt`) onto the **fixed `progs.dat` + fixed `.ent` maps** (`192` full episodes, `B_global = 1,024`, full-episode GAE($\lambda=0.95$), Aurora + discrete-only pointer entropy, `633.0 s` wall):

### 4.1 Online Training & Structural Drift Trajectory

| Iter | Role | RDMA `eff` | `sk_r` | `v_rms` | `H_occ` | Straightness | $\sigma$ | Ally Clump | Stack Adv | `occ_jsd` vs `default` | `flux_jsd` | `weirdness` | Live Structural Flags |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `m01` | main | `0.088` | `+0.398` | `312.4` | `5.62` | `0.713` | `0.410` | `16.4%` | `+16.7` | `0.322` | `0.858` | `0.436` | `[]` |
| `m02` | main | `0.620` | `+0.428` | `328.6` | `5.55` | `0.699` | `0.413` | `18.5%` | `+52.0` | `0.252` | `0.850` | `0.380` | `[]` |
| `m03` | main | `0.820` | `+0.216` | `313.0` | `5.28` | `0.623` | `0.352` | `15.3%` | `+14.0` | `0.180` | `0.653` | `0.309` | `[]` |
| `m04` | main | `0.588` | `+0.340` | `315.1` | `5.31` | `0.744` | `0.353` | `18.6%` | `+12.3` | `0.259` | `0.784` | `0.385` | `[]` |
| `m05` | main | `0.726` | `+0.370` | `310.0` | `5.10` | `0.629` | `0.389` | **`28.7%`** | `+22.9` | `0.332` | `0.834` | `0.420` | `['ALLY_CLUMPING']` |
| `m06` | main | `0.641` | `+0.238` | `317.3` | `5.28` | `0.567` | `0.404` | `17.8%` | `+26.2` | `0.236` | `0.799` | `0.388` | `[]` |
| `exp01` | exploiter | `0.414` | `+0.279` | `306.8` | `5.41` | `0.622` | `0.493` | `16.3%` | `+2.7` | **`0.408`** | `0.832` | `0.453` | `['HIGH_OCCUPANCY_SHIFT']` |
| `exp02` | exploiter | `0.548` | `+0.353` | `314.8` | `5.52` | **`0.560`** | **`0.560`** | `15.5%` | `+18.7` | **`0.381`** | `0.789` | **`0.460`** | `['TORTUOUS_ORBITING', 'HIGH_OCCUPANCY_SHIFT']` |

### 4.2 Held-Out 48-Match Multi-Map CRN Audit (on Newly Fixed `progs.dat` + `.ent` Maps)

| Policy | Score Elo (95% CI) | Mean Raw Score | Composite Elo (95% CI) | Mean Composite | $\alpha$-Rank (Score / Comp) |
|---|---:|---:|---:|---:|---|
| `default` *(newly unblacklisted carts!)* | `1500.0` `[1500.0, 1500.0]` | `8.06` *(was 2.26!)* | `1500.0` `[1500.0, 1500.0]` | `12.46` | `0.4401 / 0.5517` |
| `script_cart_rush` | `1473.6` `[1330.2, 1608.3]` | **`12.68`** | `1482.6` `[1279.2, 1688.2]` | **`18.13`** | `0.2479 / 0.2785` |
| **`exploiter_1`** | **`1470.7`** `[1371.9, 1563.4]` | `2.96` (`1554.9` on `afterslime`!) | `1413.7` `[1276.1, 1537.2]` | `4.90` | `0.1667 / 0.0621` |
| **`main_final`** | **`1446.0`** `[1369.2, 1512.6]` | `1.55` | **`1440.9`** `[1319.6, 1536.0]` | `3.11` | `0.0639 / 0.0696` |
| `main_init` *(old Run C ckpt on new maps)* | `1431.7` `[1340.4, 1534.3]` | `2.68` | `1386.7` `[1197.5, 1567.0]` | `3.87` | `0.0636 / 0.0324` |
| `main_mid` | `1389.1` `[1301.2, 1468.5]` | `0.58` | `1358.0` `[1238.7, 1453.9]` | `1.28` | `0.0178 / 0.0057` |

- **Masked Online 1v1 DM Skill-ID:** Early (`0–8 s`) `r = +0.2967` (`MSE = 0.9584`) → Late (`8 s+`) **`r = +0.5235`** (`MSE = 0.7633`, `n = 6,000`).
- **Optimization Invariants:** `0.0%` dead wide-FFN neurons, `max_dev_from_1 = 0.0`, flat `cont_std = 0.378 → 0.377`.
- **Key Scientific Takeaway:** Fixing cart blacklisting & `g_spawn_swizzle` boosted the baseline `default` bot by **`3.6×`** (`2.26 → 8.06` raw score, `16.61` on `stormkeep`), immediately exposing how much the old Run C policy (`main_init`) had overfitted to the broken-cart regime (`1431.7` Score Elo / `1386.7` Composite Elo on the fixed maps). Over 192 resumed episodes, `main_final` recovered **`+54.2` Composite Elo** (`1386.7 → 1440.9`, `2.15×` Composite $\alpha$-Rank), while our live `PhaseSpaceFingerprint` caught `m05`'s transient `ALLY_CLUMPING` (`28.7%`) and exposed `exp02`'s **`TORTUOUS_ORBITING` + `HIGH_OCCUPANCY_SHIFT`** counter-policy (`str = 0.56`, `σ = 0.56`, `1554.9` Score Elo on zero-shot `afterslime`) in real time.
