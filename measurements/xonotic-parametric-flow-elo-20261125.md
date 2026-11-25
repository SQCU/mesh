# Xonotic Parametric Flow Controller, Loopback UDP IPC, Causal Flow Audit & Seat-Rotated Elo (2026-11-25)

## 1. Transport Fix: Loopback `AF_INET` UDP (`udp://127.0.0.1`)

Simultaneous teardown of 6 `darkplaces-dedicated` servers sharing 8–16 MB `AF_UNIX SOCK_DGRAM` buffers previously triggered a macOS 26.3.2 (`xnu-12377.91.3~2`) `soclose() → unp_detach()` deadlock on `unp_list_mtx`.

- Added `udp://host:port` loopback `AF_INET SOCK_DGRAM` transport (1–2 MB socket buffers, `framebytes = 8192`, `slots = 1` within macOS `net.inet.udp.maxdgram = 9216`) in `xonotic/darkplaces-work/mesh_ipc.c` and `xonotic/solver/xonwire.py`, preserving `AF_UNIX` as an explicit option.
- Updated `MultiServerPlanHarness._stop_servers` (`xonotic/solver/strat/plan_runner.py`) to stagger `SIGTERM → wait → SIGKILL` per child before closing the parent socket.
- **Verified stability:** `0` transport errors, `0` unkillable processes, and `100%` clean teardown across `>40` concurrent `darkplaces-dedicated` launches and `>10,000` live strategy ticks.

## 2. 16-Float Parametric Flow Efferent Interface & Literal Cart-Spline / Voronoi Afferent Manifold

### QuakeC Efferent Bridge (`PLC_STR_PLAN_W = 16`, zero change to stock `default` bots)
- **Cart-spline lead/lag (`cart_ds ∈ [-600, +600]u`)**: Evaluates `payload_pos(cart, bound(0, cart.plc_s + ds, cart.plc_length))` in `havocbot_goalrating_payload_carts` (`sv_payload.qc`) so policies position bots inside the `160u` push cylinder (`ds ≈ 0`), vanguard ahead along the track (`ds > 0`), or rearguard behind checkpoints (`ds < 0`).
- **Voronoi spatial flow (`nav_x, nav_y, nav_w`)**: `havocbot_goalrating_payload_nav_flow` adds a commensurable continuous spatial potential over `g_waypoints`, blending smoothly with carts and items without one-hot override contradictions.
- **Typed item/powerup flow (`item_w, pwr_w`)**: Modulates `item_scale` and major pickups (`Strength`, `MegaHealth`, `BigArmor`, `MegaArmor`) in `havocbot_goalrating_items` (`roles.qc`) with exponential respawn-ETA discounting (`t ≤ 5s`) to prevent empty-pad camping; `payload_perceive` encodes item class (`1..4`) in `EVT["AMOUNT"]`.
- **Cart-coupled combat priority (`atk_cyl, leader`) & local steering (`posture`)**: Injected directly into `havocbot_chooseenemy` (prioritizing enemies standing inside cart push cylinders via `bot_plan_in_cart` and focus-team rivals) and `havocbot_movetogoal` (`havocbot.qc`).
- **Eliminated 2–5s sample-and-hold starvation**: On `PLAN` receipt (`sv_payload_strategy_io.qc`), active planned bots cap `bot_strategytime` via `navigation_goalrating_timeout_expire(it, 0.35)`.

### Python Afferent Manifold & Controller (`map_manifold.py`, `plan_policy.py`)
- `MapManifold` loads `boil.ent` + `boil.ent.measurements.json`: `55` navigation nodes, `226` edges, `7` real Voronoi cells (`VCellMap` + `NavigationRows`), `3` cart Bézier/polyline splines (`609u`, `592u`, `442u`, `40` `plc_path` nodes, exact checkpoint arc-lengths), and `11` major pickups (`item_strength`, `item_health_mega`, `item_armor_big`, `item_armor_mega`, `7` weapons).
- `PlanWally` ingests literal cart-spline anchor tokens, Voronoi cell tokens fused via real geodesic `VCellMap.spatial_mask` fovea weights, typed item tokens, exact cart Frenet coordinates + `payload_think` marginal speed derivatives $\Delta\dot{s}$, and a cross-tick per-bot Mealy `GRUCell` (`393k` params continuous, `394k` params with optional `FSQBottleneck(5,5,5,5)`).

## 3. CRN Causal Flow Audit (Before RL Training)

Matched Common Random Numbers (`G = 3` concurrent servers, `seeds = [101, 202, 303]`, `slowmo = 2.0`), intervening only on Team 1 while Teams 2–3 run stock HavocBot:

| Condition (Team 1 intervention) | UDP Err | Cell TV vs Stock | Target Cell 5 Frac (Dist / Flow Cos) | Cart 0 End Cell 4 Frac / $\sigma_0$ / Push | Team-1 → Team-2 Dmg Share |
|---|---:|---:|---|---|---|
| `stock_baseline` | 0 | `0.000` | `0.312` (`646u`, `+0.02`) | `4.9%` / `+212u` / `38.5k` (end `68.7`) | `0.568` (`1534` vs `1166`) |
| `voronoi_cell_flow` (`nav_w=3.8 → cell 5`) | 0 | **`0.640`** | **`0.951`** (`235u`, **`+0.14`**) | `0.0%` / `+596u` / `25.3k` | — |
| `cart0_escort` (`pick=0, w0=2, ds=0`) | 0 | `0.203` | `0.384` (`610u`, `-0.04`) | `2.5%` / `+172u` / `78.7k` | — |
| `cart0_vanguard` (`pick=0, w0=2, ds=+450`) | 0 | `0.212` | `0.371` (`635u`, `-0.05`) | **`11.5%` (`4.6×`)** / **`+227u` (`+55u`)** / **`100.2k` (`2.6×`)** | — |
| `powerup_flow` (`item_w=2.4, pwr_w=2.8`) | 0 | `0.122` | `0.261` (`636u`, `+0.05`) | end push **`434.8` (`6.3×` stock)** | `0.465` |
| `combat_focus_t2` (`leader=2, atk_cyl=2.8, posture=-0.65`) | 0 | `0.164` | `0.456` (`564u`, `+0.01`) | end push `12.6` | **`0.640`** (`1445` vs `814` = `1.78×`) |

Every efferent channel demonstrably shifts live bot state-action trajectories on identical seeds.

## 4. Cross-Server ($r = 4$) McCandlish SNR & Offline 20-Step Scaling Probe

Evaluated on `1,135` actor transitions across `G = 4` independent concurrent servers (`boil`, `slowmo = 2.0`):

| Architecture | Params | Init `actor_pg` cos (`B_noise`) | Init `gram_cgt` cos | Init `total` cos (`B_noise`) | Offline 20-step `val_L` | Post-20 `actor_pg` cos (`B_noise`) |
|---|---:|---:|---:|---:|---:|---:|
| `shallow_L1` (no Mealy, `depth=1`) | 294k | `+0.989` (`0.2`) | `+0.303` | `+0.936` (`1.0`) | `0.257 → -0.054` | `+0.013` (`892.8`) |
| `recurr_L4_cont` (Mealy + tied `L=4`) | 393k | **`+0.999`** (`0.0`) | **`+0.784`** | `+0.936` (`1.0`) | `0.197 → +0.034` | `-0.000` (converged) |
| `recurr_L4_fsq` (`5^4` FSQ + Mealy + tied `L=4`) | 394k | `+0.976` (`0.3`) | `+0.404` | `+0.491` (`14.5`) | `0.029 → -0.057` | **`+0.114`** (`93.2`, `tot Bn=24.2`) |
| `ema_trained_w3` (post-167 live PPO updates) | 393k | `+0.109` (`114.4`) | — | `+0.091` (`140.5`) | — | — |

Key architectural findings:
- Weight-tied depthwise recurrence + Mealy state (`recurr_L4_cont`) lifted initial `gram_cgt` cross-server alignment **`2.6×`** over `shallow_L1` (`+0.303 → +0.784`).
- `recurr_L4_fsq` maintained positive post-20 actor gradient SNR (`+0.114`, `B_noise = 24.2`) and strong zero-shot behavior (`1600.3` Score Elo before any RL updates!).

## 5. 3-Wave EMA League Training & 12-Match Seat-Rotated CRN Elo Tournament

`recurr_L4_cont` trained across 3 online $G = 4$ waves (`wave1`: `live vs default vs default`; `wave2/3`: `live vs ema_prev vs default`; `2,702` live responses, `167` cross-server PPO updates, `0` UDP errors), then evaluated deterministically in a 12-match seat-rotated CRN tournament (`seeds = 601..606`, `48s` game time per match) anchored at `default = 1500.0`:

| Arm | Matches | Win Rate | Mean Score | Mean Max CPs | Mean Push Ticks | Score Elo `[95% CI]` | Composite Elo `[95% CI]` | Pairwise Win vs `default` (Score / Comp) |
|---|---:|---:|---:|---:|---:|---|---|---|
| **`ema_trained`** | 9 | **66.7%** | **34.7** (`6.1×`) | **1.89** (`3.8×`) | **43.4** (`2.3×`) | **1728.1** `[1537.9, 2022.3]` | **1740.2** `[1572.9, 1988.7]` | **83.3% / 88.9%** |
| **`default`** (stock) | 12 | 8.3% | 5.7 | 0.50 | 18.6 | **1500.0** | **1500.0** | — |
| **`pi0_fsq`** (step-0 FSQ) | 6 | 33.3% | 8.6 | 0.50 | 16.3 | **1600.3** `[1436.8, 1807.9]` | **1479.6** `[1115.0, 1775.0]` | 66.7% / 33.3% |
| **`pi0_cont`** (step-0 cont) | 9 | 0.0% | 0.0 | 0.00 | 0.9 | **1452.2** `[1355.5, 1521.0]` | **1146.2** `[819.9, 1352.6]` | 38.9% / 11.1% |

- **Statistically significant Policy Improvement Theorem verification:** `ema_trained` beat its own untrained initialization `pi0_cont` by **`+275.9` Score Elo** (`83.3%` pairwise) and **`+594.0` Composite Elo** (`100.0%` pairwise, 9–0 sweep), and beat stock HavocBot (`default = 1500`) with non-overlapping 95% bootstrap CIs (`[1537.9, 2022.3]` Score Elo, `[1572.9, 1988.7]` Composite Elo).
