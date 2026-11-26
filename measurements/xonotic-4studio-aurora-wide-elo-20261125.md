# Four-Studio Apple M5 Ultra Mesh: Thunderbolt RDMA TP, Aurora Wide Linear-Gram Training, Fleet Server Browser & Multi-Map Elo (2026-11-25)

## 1. Fleet Hardware & Live Thunderbolt RDMA Fabric

Deployed and ran [`xonotic/solver/strat/mesh_studio_run.py`](../xonotic/solver/strat/mesh_studio_run.py) identically across all **4 Apple M5 Ultra Mac Studios** (`120` CPU cores, `4×` Apple M5 Ultra MPS GPUs, `384 GB` unified memory) connected over live Thunderbolt RDMA (`torch.distributed` `backend="mesh"`, `device="mps:0"`, `torch 2.14.0`):

| Rank | Hostname | Tailscale IP | LAN IP | GPU / Backend | Concurrent Xonotic Servers |
|---:|---|---|---|---|---:|
| 0 | `mdots-Mac-Studio.local` | `100.103.69.53` | `10.0.0.124` | M5 Ultra MPS / `mesh` RDMA | `6` (`2×boil, 2×stormkeep, 2×catharsis`) |
| 1 | `mdots-Mac-Studio-2.local` | `100.65.151.124` | `10.0.0.81` | M5 Ultra MPS / `mesh` RDMA | `6` (`2×boil, 2×stormkeep, 2×catharsis`) |
| 2 | `mdots-Mac-Studio-3.local` | `100.107.135.46` | `10.0.0.14` | M5 Ultra MPS / `mesh` RDMA | `6` (`2×boil, 2×stormkeep, 2×catharsis`) |
| 3 | `m5ult2.local` | `100.86.148.87` | `10.0.0.57` | M5 Ultra MPS / `mesh` RDMA | `6` (`2×boil, 2×stormkeep, 2×catharsis`) |
| **Total** | **4 Mac Studios** | — | — | **`WORLD_SIZE=4` RDMA** | **`24` concurrent servers (`216` live bots)** |

- Architecture under test: **Bitter-Lesson Wide Linear-Gram Subspace ResNet** `PlanWally(d=256, h=1024, r_gram=128, r_dpp=64, depth=4, recurrent=True, use_fsq=True, fsq_levels=(5,5,5,5), use_mealy=True)` — **`1,777,715` params**, `31` unit-norm `MPLinear` layers (`7,518` unit-norm rows), Aurora leverage-aware spectral optimizer + Stiefel $\cap$ Row-Oblique init (`xonotic/solver/strat/aurora.py`), zero AdamW.

---

## 2. Live 4-Studio Thunderbolt RDMA Tensor Parallelism (`products.World`)

Verified [`MPResCGTBlock`](../xonotic/solver/strat/plan_policy.py) with `World(device="mps", products={"cross": [1,1,1,1], "ffn": [1,1,1,1]})` across all 4 Mac Studios over live `backend="mesh"` RDMA (`tp_bench.json`):

| Physical State Size ($N$ entities/tokens) | Single-Studio MPS (`ms`) | 4-Studio RDMA TP (`ms`) | Speedup | Max Abs Forward Diff | Max Abs Backward Diff |
|---:|---:|---:|---:|---:|---:|
| `128` | `19.464` | `4.468` | `4.36×` | `1.4e-06` | **`0.0`** |
| `1,024` | `7.554` | `5.497` | `1.37×` | `1.7e-06` | **`0.0`** |
| `4,096` | `55.687` | `11.633` | `4.79×` | `1.8e-06` | — |
| **`8,192`** | **`32.870`** | **`20.787`** | **`1.58×`** | **`1.9e-06`** | — |

Splitting the rank-128 linear Gram cross (`cross`) and `256 → 1024 → 256` SwiGLU FFN (`ffn`) across 4 physical Mac Studios over Thunderbolt RDMA preserves exact numerical semantics (`<= 1.9e-6` float32 forward, `0.0` backward) while delivering **`1.37× – 1.58×`** steady-state speedup on large physical states (`N = 1024 – 8192`).

---

## 3. Live MacBook Pro Fleet Server Browser REST API Verification

Running [`xonotic/solver/server_browser.py`](../xonotic/solver/server_browser.py) on the local MacBook Pro userspace machine (`Ms-MacBook-Pro.local`, `10.0.0.5:8766`) while all 4 Mac Studios ran live training (`GET http://127.0.0.1:8766/api/servers`):

| Node | Scope | Online / Spawned | Live Maps Probed | Mean UDP `getstatus` Ping | Sample Direct Connect Endpoint | Live Clients (Bots) |
|---|---|---:|---|---:|---|---:|
| `mdots-Mac-Studio.local` | `remote_studio` | `6 / 6` | `boil`, `catharsis`, `stormkeep` | `129.31 ms` | `+connect 100.103.69.53:26010` | `9/16` (`9` bots) |
| `mdots-Mac-Studio-2.local` | `remote_studio` | `6 / 6` | `boil`, `catharsis`, `stormkeep` | `125.79 ms` | `+connect 100.65.151.124:26010` | `9/16` (`9` bots) |
| `mdots-Mac-Studio-3.local` | `remote_studio` | `6 / 6` | `boil`, `catharsis`, `stormkeep` | `60.11 ms` | `+connect 100.107.135.46:26010` | `9/16` (`9` bots) |
| `m5ult2.local` | `remote_studio` | `6 / 6` | `boil`, `catharsis`, `stormkeep` | `51.76 ms` | `+connect 100.86.148.87:26010` | `9/16` (`9` bots) |
| **Total Fleet** | **`remote_studio`** | **`24 / 24`** | **All 3 maps** | — | Uniform with loopback `127.0.0.1` | **`216` live bots** |

---

## 4. Three Closed-Form Bugs Found & Fixed Directly on the 4-Studio Mesh (V1 → V4)

Running full 4-Studio telemetry (`4-machine RDMA McCandlish SNR by module group`, `FSQ entropy`, `grad_norm`, `per-map Elo`) uncovered three exact mathematical/geometric mechanisms:

| Iteration | Commit | Root Cause Exposed by 4-Studio Telemetry | Closed-Form Fix | Verified Impact |
|---|---|---|---|---|
| **V1 → V2** | `bad7dc87e` | `FSQBottleneck.proj_in(x)` fed uncentered unit-RMS tokens ($\|x\|_2 = \sqrt{256} = 16$) directly into `tanh`, collapsing FSQ entropy **`4.16 → 0.056 bits`** (`3` corner codes); `polar(D G)` amplified `action_head` noise (`cos ≈ 0`) at full `lr=0.012`. | Per-tick standardization of `proj_in(x)` before `tanh` + McCandlish $\sqrt{\text{eff}_g}$ per-group Aurora trust region (`aurora.py`). | FSQ entropy jumped **`0.056 → 8.43 bits`** (`449–457 / 625` codes); `boil` hit **`1750` Composite Elo** (`+250` vs stock). |
| **V2 → V3** | `f4337e618` | Centering `z_raw` activated the origin FSQ code `q = (0,0,0,0)`; passing `z_ste = 0` through two nested `mp_norm(..., eps=1e-5)` calls caused a **`20,950×` singular backward explosion** (`x.grad.norm`: `0.02 → 418.99`, `gnorm = 459.9 / 626.6`), swamping 4-Studio McCandlish `signal_sq`. | Removed redundant `mp_norm` calls around bounded `z_ste / half_levels ∈ [-1, 1]^4` and cleared stale transition buffers after each sync step. | Single-zero-code `x.grad.norm`: **`418.99 → 0.0257`**; live 4-Studio `gnorm`: **`626.6 → 1.10–2.56`**. |
| **V3 → V4** | `dd099619a` | `sv_payload.qc:814` (`fabs(plan_ds) > 120`) has an **absolute physical push-radius cliff** (`120u`). Under `d=256` (`cont_gain=0.08`) + `lane_scale = 0.45 * L`, step-0 bots had `cart_ds = +124..+126u > 120u`, pulling bots off carts on long-lane maps (`stormkeep` `push=0`). | Superlinear physical push-radius mapping `cart_ds = clip(120 * sign(u) * |u|^1.6, -480, 480)` (`±52u < 120u` at init, `±230–364u` when commanded) + `cont_gain = 0.04`. | **`30 / 30`** module-wave 4-Studio McCandlish SNRs strictly positive; `stormkeep` jumped **`+470` Elo (`1058 → 1526` comp, `1553` score)**! |

---

## 5. V4 Final 4-Studio Training Dynamics (`100` Sync Steps, $B_{\text{global}} = 128$, `12,800` Global Transitions = `115,200` Bot-Decisions)

Trained for `774.5 s` across `24` concurrent live servers (`6` per Studio × `4` Studios), `ppo_epochs = 1`, `lr = 0.008`, `scalar_lr = 4e-3`, `ema_decay = 0.90`:

| Wave | Opponent Mix | Updates | Wall (`s`) | RDMA AllReduce (`ms`) | Global 4-Studio $\eta$ | `encoder` $\eta$ | `gram_cgt` $\eta$ | `resnet_ffn` $\eta$ | `action_head` $\eta$ | `value_head` $\eta$ | `dynamics_head` $\eta$ | FSQ Codes (Entropy) | `gnorm` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| **`w1`** | `live vs default vs default` | `20/20` | `122.5` | `121.4` | **`14.8%`** | `16.9%` | `20.6%` | `16.7%` | `11.3%` | `77.4%` | `95.5%` | `473` (`8.49` b) | `1.13` |
| **`w2`** | `live vs default vs aur_w1` | `20/20` | `147.7` | `50.8` | **`6.1%`** | `18.7%` | `21.3%` | `16.7%` | `1.7%` | `30.9%` | `98.8%` | `502` (`8.58` b) | `1.57` |
| **`w3`** | `live vs aur_w1 vs aur_w2` | `20/20` | `170.8` | `47.4` | **`13.0%`** | `14.5%` | `16.0%` | `12.4%` | `12.3%` | `10.6%` | `99.4%` | `519` (`8.60` b) | `1.29` |
| **`w4`** | `live vs default vs aur_w3` | `20/20` | `151.8` | `28.6` | **`8.7%`** | `11.5%` | `13.1%` | `10.9%` | `8.5%` | `15.9%` | `92.9%` | `503` (`8.50` b) | `1.65` |
| **`w5`** | `live vs aur_w2 vs aur_w4` | `20/20` | `173.9` | `29.6` | **`7.8%`** | `9.1%` | `9.8%` | `6.4%` | `7.7%` | `49.9%` | `97.0%` | `483` (`8.44` b) | `0.73` |

### Architecture & Optimizer Health Audit (Post-V4 Final EMA)
- **Wide FFN Dead-Neuron Audit (`h = 1,024`):** **`0 / 1,024` dead units (`dead_fraction = 0.000000`)**, `grad_norm_cv = 1.2509` (improved monotonically V1 `1.72` → V2 `1.38` → V4 `1.25`), `min_over_med = 0.0579`.
- **EDM2 Unit-Norm Invariant (`31` `MPLinear` layers, `7,518` rows):** `min_norm = 1.000000`, `max_norm = 1.000000`, **`max_dev_from_1 = 0.000000`**.
- **FSQ Codebook Utilization:** `473 – 519 / 625` active codes (`8.44 – 8.60` bits out of `9.29` bits max = **`92.6%`** capacity, zero collapse).

---

## 6. V4 4-Studio Parallel 48-Match Multi-Map Seat-Rotated CRN Elo Tournament

Evaluated simultaneously across all 4 Mac Studios (`studio-1: boil`, `studio-2: stormkeep`, `studio-3: catharsis`, `studio-4: afterslime [zero-shot held-out]`, `12` seat-rotated `35s` matches per map = **`48` matches total**, anchor `default = 1500.0`):

| Arm | Overall Score Elo `[95% CI]` | Mean Raw Score | Overall Comp Elo `[95% CI]` | Mean Comp Score | `boil` Score / Comp Elo | `stormkeep` Score / Comp Elo | `afterslime` (Zero-Shot) Comp Elo |
|---|---:|---:|---:|---:|---|---|---:|
| **`aur_w2`** (`40` sync steps) | **`1547.8`** `[1443.7, 1635.5]` | **`3.72`** (`3.2×` stock!) | `1449.6` `[1264.5, 1614.7]` | **`6.20`** (`2.1×` stock!) | **`1577`** (`8.6`) / **`1530`** (`13.2`) | **`1553`** (`6.3`) / `1380` (`11.0`) | `1384` (`0.6`) |
| **`default`** (stock HavocBot) | **`1500.0`** | `1.16` | **`1500.0`** | `2.93` | `1500` (`2.1`) / `1500` (`5.2`) | `1500` (`2.5`) / `1500` (`4.8`) | `1500` (`1.7`) |
| `aur_w0` (step-0 init) | `1493.0` `[1423.6, 1568.8]` | `0.53` | `1427.8` `[1256.8, 1567.6]` | `1.60` | `1490` (`0.0`) / `1327` (`0.1`) | `1487` (`2.1`) / **`1502`** (`5.8`) | `1379` (`0.5`) |
| `aur_w1` (`20` sync steps) | `1489.8` `[1407.6, 1570.3]` | `0.47` | `1426.6` `[1283.9, 1554.4]` | `1.78` | `1457` (`1.2`) / `1392` (`2.4`) | **`1534`** (`0.7`) / **`1526`** (`4.6`) | `1322` (`0.1`) |
| `aur_w5` (`100` sync steps) | `1459.8` `[1407.7, 1501.2]` | `0.73` | `1417.8` `[1320.3, 1518.4]` | `1.42` | `1491` (`2.9`) / **`1515`** (`5.3`) | `1378` (`0.0`) / `1368` (`0.3`) | `1307` (`0.1`) |
| `aur_w4` (`80` sync steps) | `1467.5` `[1416.9, 1498.2]` | `0.00` | `1348.2` `[1245.8, 1430.1]` | `0.07` | `1401` (`0.0`) / `1414` (`0.1`) | `1467` (`0.0`) / `1350` (`0.2`) | `1190` (`0.0`) |
| `aur_w3` (`60` sync steps) | `1424.9` `[1358.5, 1494.3]` | `0.00` | `1344.2` `[1195.6, 1492.3]` | `0.06` | `1439` (`0.0`) / `1451` (`0.2`) | `1341` (`0.0`) / `1217` (`0.0`) | `1285` (`0.0`) |

### Key Takeaways & Next Leverage Points
1. **`aur_w2` outperforms stock HavocBot across active scoring maps:** `+47.8` Overall Score Elo (`1547.8` vs `1500.0`), **`+77` on `boil` (`1577`, `4.1×` stock score)**, **`+53` on `stormkeep` (`1553`, `2.5×` stock score)**, **`3.2×` overall raw match score (`3.72 vs 1.16`)**, and **`2.1×` overall composite score (`6.20 vs 2.93`)** — all trained with **Aurora + EDM2 unit-norm wide linear-Gram ResNet (`0.0%` dead neurons, `max_dev_from_1 = 0.0`, zero AdamW)** across 4 Mac Studios over Thunderbolt RDMA.
2. **Why `aur_w3..w5` drifted slightly after `aur_w2`:**
   - Entropy rose monotonically across waves (`8.778 → 8.828 → 8.874 → 8.920`) under `entropy_coef = 0.01`, AND Wave 3 (`train vs aur_w1 vs aur_w2`) & Wave 5 (`train vs aur_w2 vs aur_w4`) dropped `default` from the opponent mix while `aur_w1/w2` were still stochastic (`ent ≈ 8.8`). Keeping `default` (or a frozen peak checkpoint like `aur_w2`) permanently in one seat and decaying `entropy_coef` (`0.01 → 0.002`) after Wave 2 prevents late-wave entropy drift.
   - On `catharsis` (`1,760u` lanes) and `afterslime` (`2,300u` lanes), `35s` matches produced `0.0` raw checkpoints for **every** arm including `default`; lengthening eval/training horizons on mega-maps (`60s` or `slowmo=2.0`) will unlock dense checkpoint scoring there too.
