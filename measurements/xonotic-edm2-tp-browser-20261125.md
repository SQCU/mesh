# EDM2 Magnitude-Preserving Controller, Closed-Form Sharp Kernel Gram, 4-Studio TP & Fleet Server Browser (2026-11-25)

## 1. EDM2 Magnitude-Preserving Architecture (Zero AdamW)

Replaced `AdamW` and unconstrained linear/activation paths in `xonotic/solver/strat/plan_policy.py` and `xonotic/solver/strat/map_manifold.py` with Karras et al. (2024) EDM2 magnitude-preserving layers:
- **Forced unit-norm weights (`MPLinear` + `project_unit_norm_`)**: Every output weight row $w_i$ is unit-L2 normalized both in forward and via explicit spherical projection after every plain `Adam(weight_decay=0.0)` step (`31` layers / `3,262` rows: `min_norm = 1.0`, `max_norm = 1.0`, `max_dev_from_1 = 0.000000` across all training waves).
- **Prenormalized map-invariant inputs (`prenormalize_obs/carts/teams`)**: Centers all world coordinates around `map_center` and normalizes by `map_radius`, scales physical channels (`vel`, `health`, `armor`, `ammo`, `path_pos`, `score`) to $O(1)$, zeros non-stationary wall-clock/sequence counters, and projects onto unit-RMS via `mp_norm`.
- **Unit-RMS internal activations (`mp_norm`, `mp_silu`, `mp_lerp`)**: Every residual merge and Mealy state transition preserves unit RMS (`hidden_unit_rms = 0.99999`), eliminating activation outlier growth.
- **Closed-form sharpened positive kernel Gram (`MPResCGTBlock`) + explicit scalar PDF gains**:
  - Linear Gram $R(R^\top V)$ is flat when $\|r_i\|_2 = 1$; replacing it with bounded positive exponential features $\phi_\tau(\hat{u}_i) = \exp(\tau \hat{u}_i)$ ($\tau \in [0.25, 4.0]$) and exact kernel-sum normalization:
    $$\text{ctx}_i = \mathrm{mp\_norm}\!\left(\frac{\phi_\tau(\hat{u}_i)^\top \sum_j \phi_\tau(\hat{u}_j)\hat{v}_j^\top}{\phi_\tau(\hat{u}_i)^\top \sum_j \phi_\tau(\hat{u}_j) + \epsilon}\right)$$
    preserves $O(M r d)$ linear complexity, keeps outputs strictly inside the convex hull of unit-RMS $\hat{v}_j$ (zero outliers), and allows arbitrarily sharp entity selection via the explicit scalar $\tau$.
  - DPP Gram $(I + g_{\text{dpp}} B^\top B)^{-1}(B^\top V)$ has bounded condition number $\le 1 + g_{\text{dpp}} M$ under unit-norm $B$ and expresses repulsive anti-clumping team assignment.
  - Pointer heads use unit-L2 cosine logits times explicit bounded scalar inverse-temperatures (`temp ∈ [0.5, 32.0]`), decoupling PDF sharpness completely from weight/activation norms.

## 2. Exact Semantics of Tensor-Parallel Server Splitting (`products.World` across 4 Mac Studios)

Splitting a server's physical state tensor $X \in \mathbb{R}^{M \times d}$ across ranks $p \in \{0,1,2,3\}$ is **not** a semantic change to the game or policy specification; it computes the identical mathematical operator $Y = f_\theta(X)$ faster when physical states ($M$ entities/cells/tracks/voxels or FFN width $H$) scale past single-node bandwidth:
- **Row-Gram `cross` TP:** Each of the 4 Apple M5 Ultra Mac Studios (`mdots-Mac-Studio`, `mdots-Mac-Studio-2`, `mdots-Mac-Studio-3`, `m5ult2`, 30c / 96 GB each = 120c / 384 GB total) contracts its row span `[lo, hi) = world.span("cross", M)` of $\phi_j \hat{v}_j^\top$, $\phi_j$, $B_j^\top B_j$, and $B_j^\top V_j$, followed by one `world.reduce` all-reduce.
- **Column-sharded `ffn` TP:** Each rank computes its hidden column span `[lo, hi) = world.span("ffn", H)` of the unit-norm SwiGLU FFN and reduces the output sum.
- **Algebraic verification (1-node vs. 4-rank TP):** `tp_4rank_gram_max_abs_err = 1.53e-5`, `tp_4rank_ffn_max_abs_err = 1.67e-6`.

## 3. Fleet Server Browser REST API & Remote UDP Connectivity

- Updated `xonotic/darkplaces-work/mesh_ipc.c` to bind `INADDR_ANY` for non-loopback `udp://host:port` endpoints (preserving `INADDR_LOOPBACK` for `127.0.0.1`) so DarkPlaces servers running on any of the 4 Mac Studios can exchange strategy frames with the MacBook Pro or any Studio rank.
- Built `xonotic/solver/server_browser.py` (REST API + dark-mode Web UI: `GET /api/nodes`, `GET /api/servers`, `POST /api/servers/register`, `DELETE /api/servers/{id}`, `POST /api/servers/launch`, live Quake3/DP UDP `getstatus` probing).
- **Verified live:** `/api/nodes` discovered local `Ms-MacBook-Pro.local` (`127.0.0.1`) + all 4 Apple M5 Ultra Mac Studios (`100.103.69.53`, `100.65.151.124`, `100.107.135.46`, `100.86.148.87`) + `Ms-Mac-mini` (`100.77.188.102`); during Wave 1, `/api/servers` probed all **6** live DarkPlaces servers via UDP `getstatus` (`0.59–0.72 ms` ping, live `map`, `clients=9/16`, `bots=9`, `+connect` commands).

## 4. Wave-2 Stability, McCandlish SNR & 18-Match Multi-Map Elo Verification

Fixed the cumulative-counter shaping bug (`obs[t] - obs[t-1]` per-step deltas + scale-invariant lane progress `Δ(s/L)`), added batch advantage standardization + map-stratified split halves + McCandlish SNR-gated deferring (`signal_sq <= 0` defers and grows the batch rather than stepping on noise), and trained EDM2-MP `recurr_L4_fsq` across 3 multi-map waves (`2×boil`, `2×stormkeep`, `2×catharsis`, `50` total PPO updates, `0` AdamW):

| Checkpoint | Updates (Deferred) | Online `cos` | Online $B_{\text{noise}}$ | Step Eff $\eta$ | Post-Wave `act` / `cgt` cos | FSQ Codes (Bits) | Max `||w||-1` | Composite Elo (18m) | Score Elo (18m) | **Zero-Shot `afterslime` Elo** |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| `edm2_w0` | 0 | — | — | — | — | — | `0.0` | `1236.7` | `1383.3` | `1289.1` |
| `edm2_w1` | 15 (`12`) | `+0.1058` | `75.5` | **`29.2%`** | `+0.096` / `+0.049` | `94` (`5.33`) | `0.0` | `1346.8` (`+110.1`) | `1417.4` | — |
| `edm2_w2` *(near-mirror!)* | 18 (`8`) | **`+0.1807`** | **`47.8`** | **`35.3%`** | `+0.080` / `+0.089` | `93` (`5.14`) | `0.0` | **`1444.4`** (`+97.6`!) | **`1468.8`** (`+51.4`!) | — |
| `edm2_w3` *(50 total upd)* | 17 (`10`) | **`+0.1850`** | **`55.6`** | **`33.1%`** | **`+0.208`** / **`+0.165`** | `73` (`4.53`) | `0.0` | **`1464.4`** (`+227.7` vs `w0`) | `1448.1` | **`1582.9`** (**`+293.8` vs `w0`, `+82.9` vs stock!**) |
| `default` (stock) | — | — | — | — | — | — | — | `1500.0` | `1500.0` | `1500.0` |

- **Wave-2 near-mirror regression eliminated:** Online step efficiency jumped from **`0.0% → 35.3%`** (`B_noise = 47.8`, `8` opposing noise batches deferred), turning `-60.7` Elo into **`+97.6` Composite Elo / `+51.4` Score Elo**!
- **FSQ codebook richness tripled:** `73–94` active codes (`4.53–5.33` bits) vs. `13–26` codes (`2.45` bits) under AdamW.
- **Zero-shot `afterslime` Elo reached `1582.9` (`+82.9` over stock, `+293.8` over `edm2_w0`) in just 50 total updates.**
