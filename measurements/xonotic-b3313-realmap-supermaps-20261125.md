# B3313-Inspired Non-Euclidean Real-Map Variant Supermaps & Portal-Membrane Thermometry

**Date:** 2026-11-25  
**Cluster:** 4× Apple Mac Studio (`100.103.69.53`, `100.65.151.124`, `100.107.135.46`, `100.86.148.87`)  
**Modules:** `xonotic/mapgen/map_variants.py`, `xonotic/mapgen/b3313_supermap.py`, `xonotic/payload/tools/gas_thermometry_audit.py`  
**Sync Commit:** `4e6fa5d6e` (`progs.dat` v3 staged on all 4 nodes)

---

## 1. Architecture (B3313 Structural Correspondence)

To move beyond synthetic box arenas and tile **real existing Xonotic stock maps + mutated architectural variants** into gas-law-regularized supermaps (`K ≤ 24`, `J > K` up to `256`, `N ≤ 255`), we adopted a 4-layer B3313-inspired architecture:

1. **Real Stock-Map Catalog (`map_variants.py`):** 12 verified stock Xonotic maps (`boil`, `runningman`, `warfare`, `stormkeep`, `fuse`, `solarium`, `opium`, `trident`, `leave_em_behind`, `silentsiege`, `dance`, `atelier`) are extracted from `xonotic-20230620-maps.pk3`, decompiled via `bin/mesh-q3map2 -convert -format map` (all 12/12 round-trip with `0` leaks in `0.31–1.14 s`), and parsed into a reusable brush / `patchDef2` / entity / `.waypoints` AST.
2. **Parametric Level Variants & Architectural Geometry Motifs (`SubmapVariantSpec`):**
   - **Exact affine symmetry transforms:** `rot_deg ∈ {0, 90, 180, 270}`, `mirror_axis ∈ {none, x, y}` (reversing brush face plane winding `(p1, p3, p2)` and `patchDef2` control-point row order so normals stay outward), `xy_scale ∈ [0.85, 1.25]`, `z_scale ∈ [0.90, 1.25]`, plus full `target`/`targetname` namespacing (`m{tid}_`) so internal triggers/jumppads never cross-wire.
   - **Material & lighting palettes:** `stock`, `industrial_exx`, `obsidian_ruin`, `crimson_forge`, `cryo_lab` (preserving `common/*`, `skies/*`, `liquids/*`, `clip`, `trigger`).
   - **Injected walkable-aware geometry motifs:**
     - `monolith_baffles`: sightline-breaking cover monoliths on open walkable pads (`d_open ≥ 180 u`).
     - `bunker_archway`: twin fortified side buttresses (`320 u` clear span) + overhead lintel framing thoroughfares.
     - `skybridge_overpass`: solid elevated catwalk deck (`+84 u`) with ground-anchored ramps (`0` hollow crawlspace) + linked bridge waypoint.
     - `jumppad_shaft`: real `trigger_push` + `target_position` vertical kinetic launchers connecting lower and upper walkable floors (`dz ∈ [140, 380] u`).
     - `pickup_remix`: parametric weapon/armor economies (`heavy_ordnance`, `high_mobility`, `fortress_armor`).
3. **Disjoint 3D Spatial Atlas & B3313 Velocity-Preserving Portal-Membrane Joints (`b3313_supermap.py`):**
   - Packs $M$ real stock-map variants (+ optional B3313 Peristyle Nexus foyer chambers) into disjoint 64u-snapped 3D atlas cells (`≤ 16` per Z-layer, keeping coordinates well inside DarkPlaces bounds), each wrapped in a 6-brush hermetic outer containment hull (`96 u` outside tile AABB) guaranteeing **`0` leaks** even under arbitrary floating-point scaling.
   - Connects the $M$ submaps along a non-Euclidean expander graph ($d \ge 3$ per submap for $M \ge 4$) using oriented `trigger_teleport` membranes of exact parametric width $W_e \in [160, 480]\text{ u}$ with **`spawnflags = 2` (`TELEPORT_KEEP_SPEED`)** paired with verified walkable `info_teleport_destination` exits (`115–280 u` away on the same floor facing inward). Xonotic's `teleporters.qc:247` (`waypoint_spawnforteleporter`) natively links all submap waypoint graphs for `havocbot` Dijkstra routing while preserving player kinetic energy $T_e = \frac{1}{2}\langle v^2 \rangle$.
   - Allocates $J > K$ length-trimmed walkable cartpaths across real stock waypoint graphs (`len_ratio = 1.58–1.69`) and $K$ spatially disjoint team spawn bases from verified transformed stock spawn pads (`info_player_team` + `"team"`).

---

## 2. Compilation Benchmarks (`bin/mesh-q3map2 -meta`, 0 Leaks on 4/4)

| Supermap | `K` | `J` | `M` | `W_e` | Real Stock-Map Variants & Motifs | Brushes | Entities | Waypoints | BSP Size | Compile Time | `len_ratio` | Min Enemy Spawn |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `b3313_4m_w160` | 6 | 16 | 4 | `160 u` | `boil[monolith]`, `boil[r90,mx,s1.06,crimson_forge,bunker,pickup]`, `runningman[r90,industrial,skybridge,monolith]`, `warfare[r180,my,obsidian,jumppad,bunker]` | `7,616` | `612` | `232` | `6.6 MB` | `2.60 s` | `1.582` | `855.5 u` |
| `b3313_4m_w320` | 6 | 16 | 4 | `320 u` | *(identical 4-tile real-map variant quartet)* | `7,616` | `612` | `232` | `6.6 MB` | `2.30 s` | `1.582` | `855.5 u` |
| `b3313_4m_w480` | 6 | 16 | 4 | `480 u` | *(identical 4-tile real-map variant quartet)* | `7,616` | `612` | `232` | `6.6 MB` | `2.42 s` | `1.582` | `855.5 u` |
| **`b3313_9m_k8_j64`** | **8** | **64** | **9** | **`320 u`** | `b3313_nexus[peristyle_foyer]` + mutated variants of `runningman`, `warfare`, `stormkeep`, `fuse`, `solarium`, `opium`, `trident`, `boil` | **`17,487`** | **`1,450`** | **`609`** | **`15.0 MB`** | **`6.19 s`** | **`1.691`** | **`4,822.5 u`** |

---

## 3. Live 4-Studio Kinetic-Gas & Portal-Membrane Thermometry Results

All 4 supermaps ran live on `darkplaces-dedicated` across 4 Mac Studios (`0` crashes, `0` `mesh bot view fault`s, `0` cross-team spawn overlaps `< 200 u`, `0` environmental deaths):

| Supermap | `K / J / N` | `M` / Joints | `W_e` | `n` (`p/Mqu²`, CV) | `v_bulk` (`u/s`) | Total Portal Crossings | `Q_obs/joint` | `v_tr` (`u/s`) | `θ = T_e / T_bulk` | `η_⊥` | `κ_Clausing` | `stuck` | `in_push` | Carts `≥5%` / `CP1` | Teams Scoring | Telefrags |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| `b3313_4m_w160` | `6 / 16 / 48` | `4 / 6` | `160 u` | `4.46` (`0.428`) | `311.4` | `140` (`6/6` active) | `0.470/s` | `348.2` | `1.265` | `0.745` | `3.823` | `2.39%` | `21.4%` | `13/16` / `8/16` | `4/6` | `11` |
| `b3313_4m_w320` | `6 / 16 / 48` | `4 / 6` | `320 u` | `4.71` (`0.397`) | `302.3` | `198` (`5/6` active) | `0.664/s` | `323.6` | **`1.168`** | `0.769` | `2.644` | `3.66%` | `19.5%` | `12/16` / `8/16` | `5/6` | `8` |
| `b3313_4m_w480` | `6 / 16 / 48` | `4 / 6` | `480 u` | `4.74` (`0.342`) | `301.2` | `202` (`6/6` active) | `0.678/s` | `326.3` | **`1.193`** | **`0.817`** | `1.692` | `2.74%` | `15.9%` | `11/16` / `9/16` | `5/6` | `4` |
| **`b3313_9m_k8_j64`** | **`8 / 64 / 96`** | **`9 / 15`** | **`320 u`** | **`3.08` (`0.319`)** | **`305.8`** | **`165` (`10/15` active)** | **`0.258/s`** | **`366.5`** | **`1.485`** | **`0.654`** | **`1.598`** | **`8.29%`** | **`17.7%`** | **`25/64` / `20/64`** | **`7/8`** | **`0`** |

### Key Empirical Findings
1. **Real-map gas equilibration under portal-width tuning:** Increasing portal aperture width $W_e$ across the identical 4-submap real-Xonotic quartet monotonically increases cross-submap flux (`140 → 198 → 202` crossings; `Q_obs = 0.470 → 0.664 → 0.678/s/joint`), tightens inter-submap density equilibration (`CV(n) = 0.428 → 0.397 → 0.342`), raises ballistic normal-flow alignment (`η_⊥ = 0.745 → 0.769 → 0.817`), and lowers single-pad portal telefrags (`11 → 8 → 4`), while holding bulk RMS velocity flat (`301.2–311.4 u/s`).
2. **Scale invariance at target mean `(K=8, J=64, N=96)` across 9 real submaps:** `b3313_9m_k8_j64` hits bulk gas density **`3.08 players/Mqu²`** (within `3.1%` of the 29-map stock Xonotic empirical target `3.18 / Mqu²`) with `CV(n) = 0.319`, bulk speed **`305.8 u/s`** (matching the 4-tile quartet within `±1.6%`), `165` live cross-submap portal crossings, `0` telefrags, `0` spawn overlaps, and `0` bot view faults.
3. **Identified next refinement:** On the high-flux 4-tile quartet (`4–11` portal telefrags), emitting 2–3 offset `info_teleport_destination` pads per portal (leveraging Xonotic's native `TELEPORT_TELEFRAG_AVOID` multi-destination selector in `teleporters.qc:192`) will drive portal telefrags to `0` even at peak flux.
