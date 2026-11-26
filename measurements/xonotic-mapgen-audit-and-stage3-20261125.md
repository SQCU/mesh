# Map Procedural Generation Audit, Runtime Bug Fixes, and Stage-3 Parametric Arenas (2026-11-25)

## 1. Initial 6-Map Live Runtime Audit (Before Part-1 Fixes)

Measured across `45 s` wall (`slowmo = 2.0` → `90 s` game time, `payload_speed = 80`) on Studio 1 under stock HavocBots:

| Map | Cart Lengths (`u`) | Max Frac (`c0, c1, c2`) | `first_cp` (game s) | Max Score | In Push Zone | XY Near `dz ≥ 96 u` | Rider Gaps | Cross-Team Spawn Overlaps (`< 150 u`) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `boil` | `609, 592, 442` | `0.55, 1.00, 1.00` | `14.9 s` | `343.9` | `14.1%` | `34.0%` | `0` | **`124`** |
| `stormkeep` | `1011, 1080, 1055` | `0.00, 0.00, 0.53` | `12.3 s` | `58.7` | `14.2%` | `4.5%` | **`13`** | **`49`** |
| `afterslime` | `2300, 1953, 1701` | `0.33, 0.06, 0.02` | `75.7 s` | `14.3` | `3.3%` | `12.3%` | `0` | **`45`** |
| `dance` | `8521, 3315, 3315` | `0.00, 0.00, 0.10` | *never* | `0.0` | `20.7%` | `3.2%` | `0` | **`136`** |
| `catharsis` | `1761, 1761, 1704` | `0.08, 0.00, 0.00` | *never* | `0.0` | `0.3%` | `0.0%` | **`2`** | **`44`** |
| `vorix` | `13655, 1461, 1461` | `0.02, 0.02, 0.02` | *never* | `0.0` | `0.5%` | **`62.3%`** | `0` | `8` |

---

## 2. Five Root-Cause Bugs Fixed (Part 1)

1. **`g_spawn_swizzle` ignored team spawns (`xonotic/qcsrc/server/spawnpoints.qc`):** Setting `g_spawn_swizzle = 1` passed `teamcheck = -1` into `Spawn_FilterOutBadSpots`, mixing RED/BLUE/YELLOW players into the same spawn pool even when `info_player_team1/2/3` existed. Fixed to filter by `this.team` when `Spawn_HasTeamSpawn(this.team)` is true (with automatic fallback to `-1` if empty).
2. **Centroid-clumped, team-stripped spawns (`xonotic/payload/tools/mkentfile.py`):** `balanced_spawn_points` sorted waypoints by equidistance without minimum inter-spawn spacing and stripped all team spawns into shared `info_player_deathmatch`. Replaced with `partition_team_spawns`, emitting spatially disjoint `info_player_team1..N` bases with bounded approach distance.
3. **Rider gaps & runaway lane asymmetry (`xonotic/payload/tools/navmesh.py`):** Added `rider_adj` (`CART_RIDER_MIN/MAX`) preference in `place_carts`, primary connected-component allocation (stopping disconnected side islands from receiving carts alongside `13k u` main components), and tail-trimming (`max_lane_length = 2000 u`, `max_length_ratio = 1.45`). Across all **29 stock maps**:
   - `26 / 29` maps now have **`0` rider gaps** (`stormkeep`: `13 → 0`)
   - `29 / 29` maps now have **`path_length_ratio <= 1.45`** (`vorix`: `9.35× → 1.00×`, `dance`: `2.57× → 1.00×`)
   - `16 / 29` maps pass all static invariants (`0` rider/motion gaps, `len_ratio <= 1.45`, `min_cross_team >= 389 u`, `approach <= 7.5 s`).
4. **Clipped cart speed scaling (`xonotic/qcsrc/common/gamemodes/gamemode/payload/sv_payload.qc`):** Scaled forward/reverse `max_v` by `max(1.0, this.plc_speed_scale)` so `cart_speed_scales` is not silently capped at `200 u/s`.
5. **Permanent cart blacklisting in HavocBot navigation (`xonotic/qcsrc/common/gamemodes/gamemode/payload/sv_payload.qc`):** Exposed directly by our live solo-cart property fuzzer (`map_fuzz_audit.py`). `func_plc_cart` never set `navigation_dynamicgoal = true` or cleared `blacklisted`; whenever `navigation_findnearestwaypoint(cart, true)` returned `NULL` once (e.g., on tick 1 before waypoints finished linking), `navigation.qc:1286` **permanently set `cart.blacklisted = true` for the rest of the match** and froze `cart.nearestwaypoint` at `plc_s = 0`. Fixed by setting `cart.navigation_dynamicgoal = true`, clearing `blacklisted = false`, and syncing `nearestwaypoint = plc_wp` on cart motion.

---

## 3. Four-Studio Live Property Fuzzing Results (`map_fuzz_audit.py`, `60 s` game time)

- Across all **20 live-fuzzed stock maps** (distributed 5 maps/Studio across 4 Mac Studios):
  - **`cross_team_spawn_overlaps_200u == 0` on `20 / 20` stock maps** (down from `44–136` overlaps).
  - After the cart-blacklisting fix (`aa3a3779e`) + 1.0 s non-pushing stuck detector:
    - `boil`: **`PASS=True`**, `xov=0`, `stuck=0.0%`, `score=95.5` (up from `10.9`), `solo_frac=[1.000, 0.338, 1.000]`, `solo_cp=[4, 1, 4]`.
    - `darkzone`: `xov=0`, `stuck=0.0%`, `dzmiss=7.9%`, `score=106.9`, `solo_frac=[0.334, 0.813, 0.045]`.
    - `stormkeep`: `xov=0`, `stuck=0.6%`, `dzmiss=13.9%`, `score=78.7` (up from `14.8`), `solo_frac=[0.0, 0.879, 0.039]`.
    - `afterslime`: `xov=0`, `stuck=0.0%` (down from `16.8%`), `dzmiss=13.4%`, `solo_frac=[0.251, 0.0, 0.323]`.
    - `catharsis`: `xov=0`, `stuck=1.9%`, `score=39.1` (up from `0.0`).

---

## 4. Part 2: Stage-3 General Parametric Procedural Payload Generator (`xonotic/mapgen/arenagen.py`)

Implemented Stage 3 of `design/MAPGEN-ROADMAP.md` (parameterizing the spiral down to non-spiral symmetric multi-lane arenas) with `q3map2` compilation (`bin/mesh-q3map2` fixed with Homebrew `DYLD_LIBRARY_PATH`):

| Procedural Map | Topology | Compile Time | BSP Size | Cart Lengths (`u`) | Ratio | Cross-Team Spawn Dist | `xov` | Stuck (`≥1s`) | `dz ≥ 96 u` | Live 3-Team Score (`60 s`) | Solo Push Fracs (`c0, c1, c2`) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| **`proc_hub`** | `hub_lanes` *(non-spiral)* | `0.23 s` | `552 KB` | `[843, 843, 843]` | **`1.00`** | `1,277 u` | **`0`** | **`1.8%`** | **`4.1%`** | **`61.9 – 90.7`** | **`[0.456, 0.380, 0.247]` (`PASS=True`)** |
| **`proc_clover`** | `clover_ring` *(non-spiral)* | `0.20 s` | `465 KB` | `[976, 976, 976]` | **`1.00`** | `1,304 u` | **`0`** | **`1.2%`** | **`3.3%`** | **`98.1 – 105.9`** | `[0.611, 0.025, 0.869]` *(3-team `max_frac=[0.36, 0.47, 1.00]`)* |
| **`proc_switch`** | `switchback` *(non-spiral)* | `0.21 s` | `447 KB` | `[895, 895, 895]` | **`1.00`** | `748 u` | **`0`** | **`1.4%`** | **`5.0%`** | **`96.8`** (`first_cp 9.2s`) | `[0.036, 1.000, 0.938]` |
| **`proc_spiral`** | `spiral_hybrid` (`sw=0.75`) | `0.23 s` | `668 KB` | `[1157, 1157, 1157]` | **`1.00`** | `1,373 u` | **`0`** | `6.1%` | **`0.0%`** | `0.0` *(wedge overlap fixed post-run)* | `[0.0, 0.028, 1.000]` |

All 4 Stage-3 topologies compile leak-free with full BSP/VIS/lightmaps in **`0.20–0.23 s` each**, have **`0` rider gaps**, **`0` env deaths**, **`0` telefrags**, **`0` cross-team spawn overlaps**, and `1.00` exact lane symmetry.
