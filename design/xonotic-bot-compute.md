<!--
Design study output, not measured fact. Produced by a 13-agent workflow:
4 parallel surveys (DarkPlaces seam, Xonotic scale, Apple GPU saturation, video
legibility) -> 4 independent proposals from different first principles -> 4
adversarial judges scoring each -> synthesis. Run: wf_57cabad1-70d.

Read it as a proposal with its reasoning attached. Numbers citing measurements on
Mac16,11 / Mac17,6 are real; the two load-bearing GPU throughput figures are
ESTIMATES and are flagged as such in section 9. Verify before committing.
-->

# FINAL SPECIFICATION — "SHARED FIELD" BOT PLANNER
### Xonotic/DarkPlaces squad brain, split across mini (M4 Pro) + MacBook (M5 Max) over TB5 RDMA

**Spine chosen: `field` (mean-field congestion equilibrium).** It scored 38 but it is the only one of the four whose *load* survived adversarial review: the softplus + rank-1 coupling genuinely defeats the `ΣγᵏPᵏ` precomputation collapse, and its wire pattern (16,384 B ρ landing on exactly 4 pages) is the natural fit for a 4096-byte forced page size. `physics` scored 40 but its dominant term (68% of demo-point FLOPs) is an all-pairs interaction that a neighbour-grid cull deletes in an afternoon — the judge was right and that is fatal to the load. `policy` puts an ML training program on the critical path. `game` was shown, using its own §5.5, to be one machine at 7.7 Hz playing indistinguishably.

**Grafted in:**
- **`physics`'s placement-not-reduction wire discipline** (its 9/10 section) — replaces `field`'s in-`mesh_f` accumulation entirely, which removes the `local_partial` race the judge found and eliminates the reduce→broadcast round trip.
- **`physics`/`game`'s discrete-capability failure axis** — replaces cadence-slip as the primary failure. This is the single most important change: a 1.5× machine cannot make a *cadence* difference visible, but it can make *29 bots have no planner at all* visible.
- **`physics`'s fire-gate / rocket-lead cluster** as the money shot.
- **`physics`'s strict signed-epoch guard** — fixes the `tag`-mismatch bug the `game` judge found (stale page installs itself as the new epoch and wipes the live partial).
- **`policy`'s length assertion on the `uri_get` reply** and the NUL-byte warning on binary-through-QC-strings.

**Discarded, with reasons:**

| Discarded | Why |
|---|---|
| `field` §5 "Failure B: mean-field collapse of `mbp/*` bots" | **Mathematically false.** Losing the mini's ρ does not remove the mbp squad's *mutual* repulsion, which is computed locally and intact. Worse, `field`'s own carry-forward rule means ρ_mini is frozen, not absent. The predicted conga line will not happen. Do not put it in the video. |
| `field` §3 "ship P over the wire, 16,384 pages" | P(clean transfer) = e^(−16384·4e−4) = **0.14%**. On an unreliable connection with no retransmit this cannot work. Replaced by deterministic local construction + a 1-page hash gate. |
| `game` §5.5 hard-deadline / no-partial-commit | A rig, and the document admits the un-rigged version is invisible. Replaced by per-bot **admission control**, which is the standard way to run a fixed-cadence real-time planner. |
| `physics` all-pairs `Σ_{j≠i}ψ` at 12·N/step | Cullable to ~10 neighbours; 68% of the load evaporates. |
| `policy` per-bot LoRA + learned trunk | 2048 small GEMMs/solve at a 250 µs dispatch floor, and the honest version requires behaviour-cloning a 134 MB trunk. Weeks, on the critical path. |
| fp16 anywhere in the iteration | See §9 risk 5 — but this is a *deferred* decision, not a dismissal. |

---

## 1. THE CHOSEN PROBLEM

### 1.1 Math

Discrete-space mean-field congestion equilibrium over the map, solved by damped entropic-flavoured Jacobi fixed-point iteration.

- **Cells** `N = 4096`. Farthest-point-sampled + Lloyd-relaxed sites over the map's *reachable volume* (not the 66–189 stock waypoints, which are the locomotion graph). ~180 qu mean spacing on a stock arena — finer than a doorway.
- **Operator** `A ∈ ℝ^{4096×4096}`, fp32, 67.1 MB, row-stochastic:

  `A[i][j] = vis(i,j) · exp(−d_geo(i,j)²/2σ²)`, σ = 600 qu, row-normalised.

  "How much does what happens at cell *j* matter at cell *i*, given line of sight and walkable distance." Built once per map, **never on the wire** (§9.6).
- **Columns** `K = 8·B` (B = bot count, 8 channels/bot: approach, retreat, objective, cover, ammo, health, teammate-support, denial).
- **Iterate** `V ∈ ℝ^{4096×K}`, `M = 32` steps, `γ = 0.92`.

```
for m = 0..31:
    Z_m     = softplus( R + V_m − μ · ρ_m ⊗ 1ᵀ )          μ = 0.35
    V_{m+1} = γ · A · Z_m                                  ← the GEMM
    every 4th m:  ρ_local = rowsum(Z_m)                    ← 4096 floats
                  exchange ρ_local with peer               ← the ONLY wire traffic
                  ρ_{m+1} = ρ_local + ρ_peer
```

`R` (N×K) is rebuilt **from live game state every solve**: velocity-extrapolated human position, per-bot health/ammo/weapon range, item respawn timers, recent damage. Nothing about the solve is precomputable across solves.

**Irreducibility (the standard attack, answered):** 32 steps of `V ← γAV + R` would collapse to the single precomputable operator `(Σγᵏ Aᵏ)R`, making the whole 1.1 TFLOP fake. `softplus` is a per-step nonlinearity and `ρ_m` is a per-step data-dependent global coupling. No composition collapses. This is why `field` won the spine.

**Contraction:** `‖A‖_∞ = 1` (row-stochastic), softplus is 1-Lipschitz, so for small μ the map is a γ = 0.92 contraction. `γ³² = 0.069` residual. Fixed M ⇒ fixed, predictable wall-clock, which matters because the demo instruments cadence.

### 1.2 FLOPs per solve

Per iteration per column: GEMM `2N² = 33.554e6` FLOP. Elementwise (softplus ≈5, rank-1 subtract 1, scale 1, rowsum 1) ≈ 8·N = 32,768 FLOP.

```
FLOP/solve = M · K · (2N² + 8N) = 32 · K · 33,587,200
           = 1.0748e9 · K
```

**At B = 128 (K = 1024):**

| term | FLOP/solve |
|---|---|
| GEMM | **1.0996 TFLOP** (99.90%) |
| elementwise + rowsum | 1.074 GFLOP (0.098%) |
| **total** | **1.1007 TFLOP** |
| **at 10 Hz** | **11.01 TFLOP/s sustained** |
| per bot | 85.9 GFLOP/s |

The elementwise term is 0.1% of the FLOPs and one extra ~50 µs dispatch per iteration. It costs nothing and it is what makes the load honest.

**Output:** 128 bots × 32 B = **4096 B = exactly one page.** 1.1007 TFLOP → 4 KB of committed plan = **2.7×10⁸ FLOP per output byte.**

---

## 2. WHY IT EXCEEDS ONE MACHINE — HONESTLY

### 2.1 Compute-bound or memory-bound

For `Y(N×K) = A(N×N)·Z(N×K)` with A (67.1 MB) streamed from unified memory every iteration:

```
AI = 2N²K / (4N² + 8NK) = K / (2 + 4K/N)   flop/byte
```

Machine balance β = sustained fp32 ÷ memory BW:

| node | sustained fp32 | BW | β | its K | AI | margin |
|---|---|---|---|---|---|---|
| MacBook | 12,950 GF/s *(measured, K=1024)* | 362 GB/s | **35.8** | 672 | **252** | **7.0× past ridge** |
| mini | ~4,500 GF/s *(**ESTIMATED, UNMEASURED**)* | 238 GB/s | **18.9** | 352 | **160** | **8.5× past ridge** |

Both **firmly compute-bound**, with a much fatter margin than `physics`'s SDF design (1.4×) had. A-streaming costs 67.1 MB × 32 × 10 Hz = **21.5 GB/s** = 5.9% of the MacBook's bandwidth, 9.0% of the mini's. Not near the wall.

**The counter-example that justifies batching:** at K=1 (one bot, one matvec) the measured rate is 0.1 TF/s = **0.7% of peak**, dispatch-limited — 0.337 ms for 34 MFLOP. 128 independent matvecs = 43 ms for work that one K=1024 GEMM does in 2.65 ms. Batching bots into columns of a *shared* field is the entire trick, and it is also the physically correct model: the bots really do share one map.

### 2.2 The crossover, and how close it is

Per-bot cost **0.08590 TFLOP/s**. Two inputs are unmeasured and both are flagged:

- **MacBook usable-for-solver = 8.5 TF/s.** Assumes ~35% of the 12.95 TF measured ceiling is lost to rendering the human's client + the DarkPlaces server. **UNMEASURED. Risk #1.**
- **mini = 4.5 TF/s.** Extrapolated from 20 cores × 128 ALU × 2 × 1.578 GHz × ~57% MPS efficiency. **UNMEASURED. Risk #2.** (I use 4.5, not `field`'s 4.8, because the `policy` judge's 3.2–3.8 TF estimate is not obviously wrong and 4.5 splits the difference conservatively.)

| B | K | TFLOP/s | MacBook alone (8.5) | mesh (13.0) |
|---|---|---|---|---|
| 32 | 256 | 2.75 | 32% | 21% |
| 64 | 512 | 5.50 | 65% | 42% |
| 96 | 768 | 8.25 | **97% — the edge** | 63% |
| **128** | **1024** | **11.01** | **129% — CANNOT** | **85% ← demo point** |
| 151 | 1208 | 12.97 | 153% | **100% — the mesh's own cliff** |

**Single-machine admission ceiling: 8.5 / 0.0859 = 99 bots. Mesh: 151 bots.**

**Be honest about the size of the win: the mesh buys 1.53×.** That is small, and it is why the failure axis is *not* cadence (§5). A 1.53× cadence change (10 Hz → 6.5 Hz, 100 ms → 154 ms plan age, ~22 qu of extra prediction error = 0.4 player-heights) is **invisible on video.** Any design that leans on it is theatre. This design leans on capability loss instead.

**Be honest about arbitrariness.** `N=4096`, 8 channels/bot, `M=32` are design choices. Xonotic does not need them; the demonstration needs a load. The mitigation is an ablation that must be run and published (§9.4): if a 1024-cell field scores within noise of the 4096-cell field on a behavioural metric, this design is inflated ~4× and should shrink and say so.

**Rule for setting the demo point after measurement:**
```
B_single = floor(TF_macbook_usable / 0.08590)
B_demo   = round_to_multiple_of_8(1.25 × B_single)
B_mesh   = floor((TF_macbook_usable + TF_mini) / 0.08590)
```
Require `B_demo ≤ 0.90 × B_mesh` or the mesh has no jitter margin. With 8.5/4.5 this yields B_single=99, B_demo=128, B_mesh=151 (128/151 = 85% ✓). With a pessimistic 6.0/3.5 it yields B_single=69, B_demo=88, B_mesh=110 (80% ✓) — the design still works, at a smaller headline number.

---

## 3. THE SPLIT AND THE WIRE

**Column split by bot ownership, coupled only through ρ.** Both nodes hold the full A. Neither ever sends a matrix. (Row-split of A costs 8.4 MB/iteration each way = **2.7 GB/s = 31% of link** plus 2048 serialised frames of latency — rejected. Iteration-pipelining adds a full solve of latency, i.e. it manufactures the exact staleness we are measuring — rejected.)

| | node 0 — MacBook | node 1 — mini |
|---|---|---|
| runs | DarkPlaces dedicated server, human's Xonotic client, HTTP sidecar, solver shard | solver shard only |
| bots | `mbp/00 … mbp/83` (**84**, K=672) | `mini/00 … mini/43` (**44**, K=352) |
| GEMM/solve | 0.7213 TFLOP | 0.3778 TFLOP |
| at ceiling | 84.9 ms | 84.0 ms |
| + 16 CB submits × 300 µs | 4.8 ms | 4.8 ms |
| **duty @ 10 Hz** | **89.7%** | **88.8%** |
| resident | A 67.1 MB + V/Z/R 3×11.0 MB ≈ 100 MB | ≈ 92 MB |

Split ratio 65.6:34.4, matched to the assumed 8.5:4.5 capability ratio. **Both nodes at ~89% duty is thin** — this is risk #3 and the contingency is B=112 (79%/78% duty), stated in §9.

### 3.1 What crosses the wire

**ρ exchange — symmetric one-way, no reduce/broadcast.** With exactly two nodes, an all-reduce is just an exchange: each node sends its own partial ρ, each node adds the peer's locally on GPU (4096 adds, free). This is the `physics` graft. It halves latency (one-way ~5 µs, not a 10 µs RTT), removes all arithmetic from `mesh_f`, and removes the read/write race the `policy` judge found in `field`'s reducer.

- `ρ ∈ ℝ^4096` fp32 = **16,384 B = exactly 4 pages of 4096 B.** Zero padding, zero fragmentation. `max_sge=1` never binds because every message is exactly one page.
- 8 exchanges per solve (every 4th of 32 iterations). Each exchange = 4 pages each way = **8 pages, 32,768 B.**

**Plan page — mini → MacBook, once per solve, sent TWICE.**
- 44 bots × 32 B = 1408 B → 1 page. Duplicated for loss immunity (§4.3). **2 pages, 8192 B.**

**Per solve:**

| | pages | bytes |
|---|---|---|
| ρ, MacBook → mini | 32 | 131,072 |
| ρ, mini → MacBook | 32 | 131,072 |
| plan, mini → MacBook | 2 | 8,192 |
| **total** | **66** | **270,336** |

### 3.2 Link utilisation

| quantity | value |
|---|---|
| exchanges/s | 8 × 10 Hz = **80** (plus 10 plan sends) |
| pages/s | **660** |
| bytes/s aggregate | **2,703,360 B/s = 2.70 MB/s** |
| **vs 8.7 GB/s one-way** | **0.031 %** |
| vs 5.72 GB/s each-way-simultaneous | 0.023 % (mini→mbp), 0.023 % (mbp→mini) |
| **FLOP per wire byte** | **4.07 × 10⁶** |
| latency/solve | 8 × ~5 µs one-way + 66 × 4 KB wire time (≈31 µs) ≈ **71 µs of 100,000 µs = 0.07 %** |
| QPs used | 2 of 11 |
| queue depth | 66 pages per 100 ms vs 4095-frame depth |
| registered memory | **528 KB** (§7) of 1.64 GB = 0.03 % |

The GPU command-buffer submit overhead (4.8 ms/solve) is **68× more expensive than the entire RDMA exchange.** The cable is free. That is the design point the brief demanded.

---

## 4. LOSS TOLERANCE

**Yes — this is explicitly an iterative solver that converges anyway, and that is the reason this workload was chosen for an unreliable transport.**

### 4.1 The structural argument

ρ enters the iteration as a **bounded smoothing prior** inside a γ=0.92 contraction, not as a constraint. A perturbed ρ at iteration *m* produces a perturbed iterate `V_{m+1}`, and the remaining `32−m` contraction steps attenuate that perturbation by `γ^(32−m)`. The solve is a fixed-point iteration re-seeded from fresh `R` every 100 ms, so **errors never accumulate across solves.** A damaged iterate is simply a worse starting point for the rest of this solve.

This is the ideal property for a link with no acks: **we never retransmit, and we never need to.**

### 4.2 Quantified

Pages/s = 660. At the measured 0.04% loss: **0.264 lost pages/s ≈ one lost page every 3.8 s.**

Per 4-page ρ slab, `P(incomplete) = 1 − 0.9996⁴ = 0.16%`. With 160 slab-legs/s that is **0.256 incomplete slabs/s**, i.e. ~2.6% of solves see one damaged quarter-slab.

**Policy on an incomplete slab:** the solver polls the arrival bitmask with a **2 ms deadline** (200× the ~10 µs RTT). On expiry it **carries forward that quarter of ρ from the previous exchange** and proceeds. Never blocks, never retransmits.

**Magnitude of the induced error:** near the fixed point ρ changes ~1%/iteration; carrying one quarter forward by 4 iterations ⇒ ≤4% error on 25% of ρ ⇒ ≤1% on ρ ⇒ ×μ=0.35 ⇒ **≤0.35% perturbation of Z**, then attenuated by γ^(32−m). This is three orders of magnitude below the decision margin of the downstream argmax over cells. **Invisible.**

### 4.3 What is NOT loss-tolerant, and how it is handled

The **plan page** is not iterative — it is a committed result. A lost plan page costs 44 bots one extra 100 ms of staleness. At 10 sends/s × 4e-4 that is one loss per 250 s. That is already invisible (100 ms ≈ 40 qu ≈ ⅔ of a player width), but since the demo's whole signal is plan freshness we do not want a background false-positive rate on our own failure signature. **Send the plan page twice.** Cost: 1 extra page per solve (+1.5% of an already-negligible 0.031%). Loss probability drops to 1.6e-7 per solve ≈ **one per 17 hours.**

### 4.4 Corruption vs drop

The measured 0.04% figure is *frame loss*. If any of it is silent *corruption* rather than drop, the arrival bitmask will not catch it. Mitigation: a 4-byte FNV-1a over the payload carried in the last 4 bytes of each page, so ρ is 4095 floats + 1 checksum word per page (4092-dimensional ρ, cells 4092–4095 padded). **Compare as `uint32`, via a bitcast — never as a float** (`game`'s `fnv_sum` returned a float and compared with `!=`, which cannot survive a 24-bit mantissa and is always-true on NaN). A page failing checksum is treated exactly as a lost page: dropped, carry forward.

### 4.5 Epoch safety — the one thing that *is* strict

A *late* page is harmless; a *mis-epoched* page is not. The `stream` field carries `(epoch << 4) | exchange_idx`. `mesh_f` computes a **signed** age against a watermark and drops anything not strictly current-or-newer **before** touching any slot state. This fixes the concrete bug the `game` judge found, where a tag mismatch caused the stale page to install itself as the new epoch, zero the arrival mask, and seed the live partial with garbage.

**Note on wrap:** if `stream` is 16 bits, epoch has 12 bits and wraps every 409.6 s at 10 Hz. The signed-difference test misbehaves at the wrap. Handle it: reset the watermark and invalidate the ring on a detected wrap, costing one solve every 6.8 minutes. Verify the actual width of `stream` in `mesh-flow.c` before building.

---

## 5. CADENCE AND THE VISIBLE FAILURE

### 5.1 Cadence

**Solve rate: 10 Hz (Δt = 100 ms).** Chosen because (a) 89% duty at B=128 is the honest edge, (b) the `uri_get` delivery path costs ~15–30 ms, which is 25% of a 100 ms period but 160% of a 64 Hz tick — a per-tick controller over this path is off the table, and (c) 100 ms makes a 1-second stall unambiguously 10 missed cycles rather than jitter.

**Plan age budget:**

| stage | ms |
|---|---|
| R assembly from server snapshot | 3 |
| 32 iterations (16 CBs × 2 iterations) | 85 |
| plan page mini→MacBook | 0.01 |
| extract + base64 | 2 |
| `uri_get` HTTP + one `Curl_Frame` poll | 15–30 |
| **age at first consumption** | **~105–120 ms** |
| **age at expiry** | **~205–220 ms** |

**Admission control (replaces `game`'s rigged all-or-nothing deadline).** Each node plans as many of *its own* bots as fit inside the 100 ms deadline at fixed M=32, in a deterministic priority order. Bots that do not fit are **not planned** — they hold their last plan indefinitely. This is the standard way to run a fixed-cadence real-time planner (you never let one solve eat the next), and unlike a global commit-nothing rule it is not a rig: it produces graceful behaviour under mild overload and a hard, name-identifiable set of dark bots under real overload.

**How a field becomes motion over Δt.** Per bot at cell *c*:
- **wish direction** = `Σ_{j∈nbr₈(c)} (V[j,b] − V[c,b]) · unit(c→j)`, normalised → drives `havocbot_moveto()`.
- **intercept cell** = `argmax_j V[j,b]` within the 12.5-step horizon → written to the plan record, and spawned as a small world-marker entity.
- **fire gate** (`physics` graft) = a timestamp offset at which the bot's plan says the intercept is reached and a shot should land. `HavocBot_Aim` writes `v_angle` from the plan's aim field.

Between plan ticks the bot holds the last wish vector: 100 ms × 400 qu/s = 40 qu ≈ ⅔ of a player width. Invisible.

**Hard design rule — no fallback, no extrapolation, no blend.** The engineering instinct is to fall back to stock `havocbot` when the plan is stale. That instinct destroys the demonstration, because stock Xonotic AI *looks fine on video*. Stale means stale: hold the last plan, freeze the markers, do not smooth. State this openly as a deliberate choice.

### 5.2 The failure — discrete, not cadence-based

**Trigger: a hand enters frame and pulls the single Thunderbolt cable, mid-fight, at a moment the skeptic chooses.**

Because ownership is static and name-encoded, the failure is **partial and name-correlated**, which is far stronger than a global stall: it rules out "the whole server hitched" from the video alone, with no argument.

**Scale argument.** Xonotic ground speed ~400 qu/s, strafe-jumping 800–1500 qu/s. 1 qu ≈ 1 inch; a player model is 56 qu tall. A 1-second-stale plan points the squad at a location **400–1500 qu = 7 to 27 player-heights = a different room.** Human direction changes at ~2 Hz, so predictive error *saturates at room scale by ~600 ms* — which is exactly the knee where a viewer stops reading "lag" and starts reading "broken."

**Timeline — the 44 `mini/*` bots:**

| t | what the camera sees |
|---|---|
| **0 ms** | Cable out. Nothing. Important — the failure must not read as a physical jolt synchronised with the hand. |
| **0–200 ms** | Nothing. The last plan is still inside its normal 220 ms life. Squad looks perfect. |
| **~200 ms** | **The intercept markers freeze.** Until now they have been gliding *ahead of* the human. Now ~10 of them are nailed to world space while the player's camera pans smoothly past them. **Smooth camera + frozen world markers is the single most legible frame in the project** — it separates "the brain stalled" from "the game stalled" with no text and no overlay. |
| **200–600 ms** | Bots keep moving *perfectly smoothly*. No stutter, no teleport. They are executing a beautifully computed plan for a world that no longer exists. |
| **~400–800 ms** | **The rocket cluster.** Fire gates fire on schedule. Six to ten bots put a tight, obviously-aimed cluster of rockets into the doorway the player left, while the player stands ten metres to the side watching. Spatially coherent, unmistakably deliberate, unmistakably wrong. |
| **600–1000 ms** | The bots physically arrive. 400–1500 qu of error = a full room. They stop dead (goal reached, no new goal) and keep firing at empty geometry. **All wrong in the same way at the same stale point** — coherent, not noisy. A viewer reads coherent wrongness as "the shared thing they were listening to stopped." Random wrongness reads as "bad AI." |
| **whole window** | `showfps` unchanged. `sv_lagreporting` broadcasts **nothing**, because the DarkPlaces tick genuinely stays at 64 Hz — the solver is out of process. **The absence of engine lag text while the bots are visibly broken is a hostile-witness alibi**: DarkPlaces itself testifies the game loop is healthy and something *else* stopped. |
| **audio, free** | The rhythmic 10 Hz burst of footsteps and jump sounds at the plan tick stops. Already in the recording. |
| **reconnect** | Next completed solve: **all 44 bots change direction on the same frame.** A synchronous squad-wide snap is something staggered stock bot AI structurally cannot do. It proves the system is alive and being *fed*, not crashed. Capture at ≥60 fps or this reads as staggered. |

**The 84 `mbp/*` bots — stated honestly, small print.** They keep solving. Their ρ_peer is frozen at its last value (carry-forward), so they gradually stop deconflicting against the mini cohort and begin walking through/near the ghosting mini bots' stale positions. **This is a subtle overlap artifact, not a conga line.** I explicitly do not claim the `field` proposal's "all 80 take the identical route" — their mutual repulsion is computed locally and is fully intact. Do not put a prediction in the video that a viewer with a whiteboard can falsify.

**The single-machine control run (run it FIRST, so the viewer already knows what failure looks like).** Same map, `bot_number 128`, mini absent. The MacBook admits 99 bots at 10 Hz; **29 bots get no planner and ghost permanently**, with exactly the same signature as the cable pull. That the induced failure and the capacity failure are *identical* is the argument.

**What I explicitly do NOT claim:** that 99 planned bots at 10 Hz look different from 128 planned bots at 7.8 Hz. They do not. A 1.53× machine cannot make that visible, and any design that says otherwise is lying. The claim is bounded and defensible: **at the target cadence, one machine can plan 99 bots and the pair can plan 151.**

**Capture spec (from the video survey):** ≥60 fps; FOV ~110; a map with a central arena and long sightlines (corridors show two bots at a time and destroy every legible channel); the human plays erratically and specifically breaks line of sight and reverses; stall duration 1.0–1.5 s (under ~600 ms reads as lag, over ~2 s reads as a crash and loses the recovery payoff); two passes of the same event from the one server demo file (first-person + overhead spectator at the identical timestamp defeats the editing accusation).

**Known limit of the artifact, state it out loud:** `sv_autodemo_perclient` records only what that one client was *sent*, post-PVS and post-`sv_cullentities_trace 1`. The demo will contain 8–15 bots, not 128. **The proof must live in the bots on screen and in the broadcast console text, never in an aggregate global claim.**

---

## 6. THE SCALING PARAMETER

**`bot_number B`, ramped live on camera with `sv_cmd bot_number N`.**

Legible three ways with zero custom UI: the Tab scoreboard (the viewer counts bots), the killfeed, and a server `say` line. Bots are named **`mbp/NN`** and **`mini/NN`**, so node ownership is printed in the game's own UI on every frag message, continuously, for free, and it lands in the demo file.

`K = 8B`, so solver cost is exactly linear in the number the viewer is looking at.

| B | one machine copes? | mesh? | what the viewer sees |
|---|---|---|---|
| 32 | ✅ 32% | ✅ 21% | baseline, both crisp |
| 64 | ✅ 65% | ✅ 42% | both crisp, maneuvers get interesting |
| 96 | ⚠️ 97% | ✅ 63% | single machine at the knife edge; intermittent dark bots |
| **99** | **← single-machine admission ceiling** | ✅ 65% | |
| **128** | ❌ **129% — 29 bots permanently dark** | ✅ **85%** | **record here** |
| **151** | ❌ 153% | **← mesh's own admission ceiling** | |
| 160 | ❌ | ❌ 106% — **the mesh fails too** | show it last, deliberately |

Showing the mesh's own cliff at B≈160 is the right rhetorical move: a demo that shows its own ceiling is much harder to accuse of theatre.

**The confound, stated plainly because every judge flagged it.** Raising `bot_number` also raises pure *engine* load: 64 Hz QC `SV_PlayerPhysics` (~5–15 traceboxes per bot per tick), the O(N²) enemy scan, and uncorrelated `navigation_markroutes` spikes. At B=128 that is real serial CPU load on the single DarkPlaces thread and it is **not offloadable over 4 KB pages.** A viewer cannot separate "the solver needs two machines" from "DarkPlaces cannot host 128 bots."

**The de-confounder, and its own weakness.** Expose `mesh_channels C` (channels per bot, 1–8) so `K = C·B` sweeps solver load **at constant bot count** — identical engine load, 8× GPU load range. Sweeping C from 8→2 at fixed B=128 drops the load from 11.0 to 2.75 TF/s and the dark bots come back to life, with the engine untouched. **This is the clean experiment and it is completely invisible on camera** — the viewer sees only a console line they must take on faith.

That tension is real and unresolved by any of the four proposals: the legible knob is confounded and the clean knob is illegible. **Run both sweeps, show both, and say so in the video.** Do not pretend the bot-count sweep alone isolates the split.

---

## 7. `mesh_f` IMPLEMENTATION SKETCH

No transport change. Every message is exactly one 4096-byte page, so `max_sge=1` never binds. `mesh_f` performs **placement only** — no arithmetic on payloads, which is what makes loss detectable (`arrived` mask incomplete) and recoverable without retransmission, and which removes the reducer/solver race.

### Header field usage

| field | use |
|---|---|
| `stream` | `(epoch << 4) \| exchange_idx`. epoch = wrapping 10 Hz solve counter; exchange_idx ∈ 0..7 |
| `seq` | slab index: `0..3` = ρ quarter (1023 floats + 1 checksum word each); `8` = plan page |
| `flags` | bit0 `RHO`, bit1 `PLAN`, bit2 `CONSUMED` (invariant core retires the page), bit3 `HASH` (map-load operator hash, epoch 0 only) |
| `bytes` | 4096 always |
| `path`/`hops` | compiled 2-bit-per-hop source route. At 2 nodes this is 1 hop and the forward branch never fires; the branch is shown for a >2-node topology. |
| `src`/`dst` | swapped only in the forward branch |

### Static state — allocated at load, never in `mesh_f`

```c
#define ERING 8                              /* 8 epochs = 800 ms of history */

static float    rho_peer[ERING][8][4][1023]; /* epoch × exchange × quarter   */  /* 1.05 MB */
static uint8_t  rho_mask[ERING][8];          /* 4-bit arrival bitmask        */
static uint8_t  plan_in [ERING][4096];       /* mini's plan page, ring       */  /* 32 KB   */
static _Atomic uint8_t plan_rdy[ERING];
static uint32_t watermark;                   /* highest epoch seen           */
static float    A[4096][4096];               /* 67.1 MB BSS, built at map load,
                                                NEVER on the wire, NO MR     */
```

**Registered memory: 1.05 MB + 32 KB ≈ 1.08 MB, plus the 4 KB page pool.** Against the 1.64 GB / 100-MR ceiling this is 0.07%. `A` needs no MR because it never crosses the wire.

### The function

```c
void mesh_f(void *payload, uint32_t bytes, struct wire *h, int node_idx)
{
    /* --- 0. reject malformed. Never trust the wire. --- */
    if (bytes != 4096) return;

    uint32_t ep  = h->stream >> 4;
    uint32_t ex  = h->stream & 0xF;
    uint32_t sq  = h->seq;
    if (ex > 7) return;

    /* --- 1. STRICT signed epoch guard, BEFORE touching any slot state.
             This is the bug that sank an earlier draft: a stale page must
             never install itself as the new epoch and zero the live mask. --- */
    int32_t age = (int32_t)(watermark - ep);        /* >0 = older than current */
    if (age >= ERING) return;                        /* too old / mis-epoched: DROP */
    if (age < 0) {                                   /* strictly newer: advance */
        for (uint32_t e = watermark + 1; e != ep + 1; e++) {
            uint32_t s = e % ERING;
            for (int i = 0; i < 8; i++) rho_mask[s][i] = 0;
            __atomic_store_n(&plan_rdy[s], 0, __ATOMIC_RELAXED);
        }
        watermark = ep;
    }
    uint32_t slot = ep % ERING;

    /* --- 2. checksum: bitcast compare as uint32, never as float --- */
    const uint32_t *w  = (const uint32_t *)payload;
    uint32_t got = w[1023];
    uint32_t want = fnv1a_u32(payload, 1023 * 4);
    if (got != want) return;                         /* corrupt: treat as lost */

    /* --- 3. ROUTING: forward if this page is not addressed to me.
             At 2 nodes this never fires (1-hop compiled route); shown for
             a >2-node topology.  In-place header rewrite, no copy. --- */
    if (h->dst != (uint16_t)node_idx) {
        h->hops++;
        h->path >>= 2;                               /* consume this hop's 2 bits */
        return;                                      /* core re-injects on next hop */
    }

    /* --- 4. PLACEMENT.  No arithmetic on payloads: a lost page must be
             *detectable*, and a sum with a missing addend is not. --- */
    if (h->flags & F_PLAN) {
        __builtin_memcpy(plan_in[slot], payload, 4096);
        __atomic_store_n(&plan_rdy[slot], 1, __ATOMIC_RELEASE);
        h->flags |= F_CONSUMED;
        return;
    }

    if (h->flags & F_RHO) {
        if (sq > 3) return;
        __builtin_memcpy(rho_peer[slot][ex][sq], payload, 1023 * 4);
        /* release-store the mask so the solver sees the data before the bit */
        __atomic_fetch_or(&rho_mask[slot][ex], (uint8_t)(1u << sq),
                          __ATOMIC_RELEASE);
        h->flags |= F_CONSUMED;
        return;
    }

    if (h->flags & F_HASH) { record_peer_operator_hash(payload); return; }
}
```

**Constraints satisfied:** no allocation; no blocking (one release-store, no lock, no wait); the payload pointer is never retained past return; work is bounded and branch-free of loops (one 4 KB memcpy + one FNV pass + ~15 integer ops ≈ **1.8 µs/page**, 660 pages/s ≈ 0.12% of one core); header fields are rewritten in place for routing in the forward branch.

**How partial results accumulate.** The 4 ρ pages are **disjoint 1023-float slices** — there is no cross-page dependency, so arrival order is irrelevant and any subset is usable. `rho_mask` is the 4-bit arrival bitmap. The solver thread polls `rho_mask[slot][ex] == 0xF` with a **2 ms deadline**; on expiry it carries forward the missing quarter from `rho_peer[(slot−1+ERING)%ERING][ex]` and proceeds. The 8-epoch ring gives 800 ms of history — far more than the ~12 ms exchange window, so a page is either inside its window or gone, and aliasing is impossible.

---

## 8. DARKPLACES INTEGRATION

The DarkPlaces server **never links the transport.** The solver + RDMA queue pair live in a ~200-line sidecar process on the MacBook serving HTTP on `127.0.0.1`. `mesh_f` lives entirely in the sidecar.

### 8.1 The seam — four existing mutator hooks, zero engine patch

| Rate | Hook | File:line | Action |
|---|---|---|---|
| 10 Hz | `URI_GetCallback` | `server/main.qc:551` (declared `events.qh:1250`) | decode plan blob → per-bot plan array, stamp `plan_time` |
| 10 Hz | `HavocBot_ChooseRole` | `havocbot/roles.qc:245` (`events.qh:935`) | install our role; role calls `havocbot_moveto(this, plan_pos[b])` |
| 10 Hz | `HavocBot_Aim` | `havocbot/havocbot.qc:1621` (`events.qh:1342`) | write `v_angle` from `plan_aim[b]`, return true |
| 64 Hz | *(untouched)* `havocbot_movetogoal` | `havocbot.qc:446` | stock locomotion steers to the personal waypoint we set |

The mesh sets **where** at 10 Hz; stock Xonotic locomotion handles **how** at 20–34 Hz and `SV_PlayerPhysics` at 64 Hz. Motion stays smooth; only intent is refreshed. That is precisely why the bots keep moving beautifully while their brain is stale.

Retune `bot_ai_strategyinterval 7 → 0.1` so the role hook fires at our cadence.

### 8.2 Transport into QC

**DarkPlaces builtin #513 `uri_get`** (`svvm_cmds.c:3783`, impl `prvm_cmds.c:5430`), non-blocking, reply delivered to `URI_Get_Callback` and forwarded to our mutator hook. Already in production use for the online ban list. `Curl_Frame()` runs on dedicated servers (`libcurl.c:1141`) and its throttle is a no-op at default `curl_maxspeed 0`, so the reply surfaces within one host frame.

Hard limits, all checked:
- **Reply capped at `MAX_INPUTLINE = 16384 B` and SILENTLY TRUNCATED** (`prvm_cmds.c:5380`, `:5409`). Our plan is 128 × 32 B = 4096 B raw → **5464 B base64. 2.9× headroom.** Breaks at ~380 bots.
- **`curl_maxdownloads` defaults to 3** (`libcurl.c:11`). **One batched request per solve for ALL bots.** Per-bot requests queue and stall.
- Each call `Z_Malloc`s a handle. 10/s is fine; kHz is not — another reason not to be tempted upward in cadence.
- Latency ~1 ms HTTP + up to one host frame of poll ≈ **15–30 ms**. Fine for 10 Hz. **Not fine for a 64 Hz controller — do not attempt one over this path.**

**Two mandatory guards on day one:**
1. **Assert the reply length in `URI_GetCallback`.** Silent truncation presents as random bots ghosting and would be misdiagnosed as a mesh failure — poisoning the exact signal the demo depends on.
2. **Base64, never raw binary.** QC strings are NUL-terminated and fp32 plan coefficients are full of `0x00` bytes. (This is the concrete API error the `physics` judge caught; it applies verbatim here.)

### 8.3 What is honestly a fork, and what is not

`progs.dat` is rebuilt with gmqcc — QuakeC mutators are not runtime plugins. **`csprogs.dat` is untouched, so a stock, unmodified Xonotic client connects and plays normally.** That satisfies the "human joins with an ordinary client" requirement. No engine patch, no tick-loop change, `SV_Physics` never blocks on anything.

### 8.4 Where the survey found NO clean seam — stated plainly

**There is no mutator hook to replace `havocbot_movetogoal()`.** We cannot override the per-tick wish vector from QC. Least-bad option, and the one specified here: steer **indirectly** via `havocbot_moveto(entity, vector)` (the existing personal-waypoint API, `havocbot.qc:~1640`) called from inside our role hook. This is sufficient for goal-level control at 10 Hz, which is all four of the legible behaviours (simultaneity, cutoff pathing, formation shape, lead) require. It means the mesh **cannot** express fine strafe/bunnyhop shaping. If the demo later needs visibly superhuman *movement* rather than positioning, that requires patching one line at `havocbot.qc:173` — an additive fork. Prefer not to.

**Escalation path if 10 Hz / 16 KB ever binds:** one additive SSQC builtin in a free slot (`svvm_cmds.c` has `NULL` at #630–638, #640, #641, #643) with non-blocking submit/poll semantics into a QC float array. Additive, no tick-loop change, client still stock. **Not** a tighter `uri_get` loop.

**Free instrumentation, engine-authored, lands in the demo file:** `sv_lagreporting_always 1` + `sv_lagreporting_strict 1` + `sv_maxphysicsframesperserverframe > 1` broadcast `Server lag report: …% CPU, …% lost, offset avg/max/sdev ms` as `svc_print` (`sv_main.c:2520`, `:2575`, `:2680`). Server demo via `sv_autodemo_perclient 1`. Our own sidecar adds one `say` line at 2 Hz: `[MESH] plan age 1240 ms | mbp OK 84/84 | mini STALE 0/44 | K=1024`.

---

## 9. RISKS AND OPEN QUESTIONS — RANKED

**Nothing should be built past the operator-construction tool until items 1–3 have numbers.** Two unmeasured multipliers currently span a ~2× band around a crossover that needs ~1.25× precision.

**1. MacBook GPU headroom with the Xonotic client actually rendering. *(gates everything)***
The 8.5 TF usable figure assumes ~35% of the measured 12.95 TF goes to render + server. `sudo powermetrics --samplers gpu_power -i 1000` with the client running at the real capture resolution and FOV. **20 minutes.** If the client takes only 15%, one machine copes at B=128 and the whole ladder shifts up; if it takes 50%, it shifts down. Re-derive B via the §2.2 rule. Also note: an outside reviewer disputes 12.95 TF fp32 on a 362 GB/s part as Max-class — the number came from a live `MPSMatrixMultiplication` benchmark on this machine (`/Users/mdot/dox/mesh/user/bench/mps.m`), so re-run it in front of the skeptic rather than arguing.

**2. The mini's real MPS fp32 ceiling. *(gates the split ratio and the mesh cliff)***
4.5 TF is extrapolated, not measured. Run `/Users/mdot/dox/mesh/user/bench/mps` on the mini. **10 minutes.** At 3.5 TF the mesh ceiling drops from 151 to 138 bots and the split ratio moves to 71/57.

**3. GPU duty vs render frame pacing. *(the risk that eats the money shot)***
Both nodes run ~89% duty at B=128. Worse, a coarse command buffer is uninterruptible: 8 CBs of 4 iterations ≈ 10.1 ms each against a 16.7 ms render frame would give irregular 30–45 fps footage — in the exact take where "frozen markers, *smooth camera*" is doing half the work. **Spec: 16 CBs of 2 iterations (~5.3 ms each, 4.8 ms total submit overhead).** Measure client frame-time distribution with the solver running. If p99 frame time exceeds 20 ms, fall back to **B=112 (79%/78% duty)** and accept a thinner discrete margin (13 dark bots instead of 29). There is also a nastier second-order failure: if render contention pushes a solve past 100 ms, node 0's bots start ghosting *in the baseline run*, which is indistinguishable on video from the induced failure and destroys the control.

**4. Is N=4096 theatre? *(the honesty risk)***
Stock nav graphs are 66–189 waypoints. A 189-node field is a matvec at 0.7% of peak and needs no mesh. **Required ablation:** a behavioural metric (fraction of committed intercepts the bot reaches before the human passes it, against a fixed scripted human trajectory) at `N_cells ∈ {512, 1024, 4096}`. **If 1024 scores within noise of 4096, this design is inflated ~4×, and the honest response is to shrink N and raise B, and to say so on camera.** Publish the curve or the whole thing is a FLOP generator with a game attached. Same test for the channel count `C` (§6).

**5. fp16. *(the objection a competent skeptic raises in minute one)***
Measured MacBook fp16 is 43.3 TF (3.2× fp32, via the in-core matrix units). At fp16 the B=128 load is 3.4 TF/s and **one machine does it comfortably.** Two honest positions: (a) the `R − μρ` cancellation inside a γ=0.92 contraction is exactly where fp16 loses digits — **demonstrate this by plotting the residual floor in fp16 vs fp32, do not assert it**; (b) the M4 Pro has no GPU matrix units, so fp16 buys the mini only ~1.3×, making an fp16 split *more* lopsided, not less. **If (a) fails to reproduce, say so on camera and state the fp16 crossover instead** (≈B=384, which exceeds the 255-player protocol cap and would force a different scaling parameter). Do not quietly stay in fp32 and hope nobody asks.

**6. Building `A`, and cross-machine bit-identity. *(unpriced in every proposal)***
16.78M entries each nominally needing a visibility query. **Mitigation:** assign each of the 4096 cells to a BSP leaf cluster, use the map's own PVS for cluster-level visibility (a table lookup), and issue an actual `traceline` only for pairs that are cluster-visible **and** within 3σ = 1800 qu — cutting it to an estimated 1–2M traces. Then **quantise `A` to int16 with a fixed scale** and dequantise identically on both nodes, so the two machines are bit-identical by construction rather than by hoping `exp()` matches between M4 and M5 libm. Verify with a single FNV-1a hash page exchanged at epoch 0; **refuse to start on mismatch.** Budget: a day, plus 8–25 s per map load. This replaces the discarded 67 MB wire transfer, which had a 0.14% success probability.

**7. `stream` field width and epoch wrap.** Read `mesh-flow.c` and confirm. If 16 bits, epoch wraps every 409.6 s and the signed-age test misbehaves at the boundary — one bad solve every 6.8 minutes, in the middle of a multi-minute unbroken take. Handle explicitly.

**8. Is the 0.04% loss *drop* or *corruption*?** If it is pure drop, the `arrived` mask covers it and the per-page checksum is 4 bytes spent for nothing (harmless). If any of it is corruption, the checksum is load-bearing. Measure once; keep the checksum either way.

**9. Thermals over a multi-minute unbroken take.** Sustaining ~11 TF/s across two machines while one also renders at 60+ fps will throttle. A late-take ghost caused by throttling rather than by the missing node is an own-goal a skeptic is right to raise. **10-minute soak with `powermetrics` logged alongside the recording**; keep the take under the measured throttle onset, or ship the GPU frequency trace as part of the artifact.

**10. Staleness legibility depends on `R` being dominated by fast terms.** A 1-second-stale field over a *static* map still mostly points the right way — that is "slightly worse," not "obviously wrong," and it would kill the demo. Hard constraint: velocity-extrapolated human position, recent damage, and live threat must carry **≥3× the weight** of item locations and map structure. **Verify by measuring `‖plan(t) − plan(t−1s)‖`; require a median above ~400 qu.** Be transparent that this is a tuning choice made for legibility.

**11. Demo depends on the human's performance.** If the player moves predictably, a 1-second-stale prediction is nearly correct and the ghosting is muted. The shot requires breaking line of sight and reversing. Rehearse it.

---

### One-paragraph summary

128 bots share one 4096-cell influence field, solved as 32 fixed-point iterations of `V ← γ·A·softplus(R + V − μρ1ᵀ)` — **1.1007 TFLOP per solve, 11.01 TFLOP/s at 10 Hz**, compute-bound by 7–8.5× on both machines, irreducible because of the per-step softplus and the per-step global coupling. The two Macs split it by **bot ownership** (84 `mbp/*` + 44 `mini/*`) and are coupled only by a 4096-float occupancy vector that lands on **exactly four 4096-byte pages**: 66 pages and 270,336 bytes per solve, **2.70 MB/s, 0.031% of the link, 4.07 million FLOP per wire byte, 71 µs of latency inside a 100 ms budget.** `mesh_f` does placement only — memcpy, checksum, arrival bit, strict signed-epoch drop — so a lost page is *detectable* and is absorbed by carrying forward one quarter of a smoothing prior into a γ=0.92 contraction, which converges anyway; nothing is ever retransmitted. One machine can plan **99** bots at cadence; the pair can plan **151**. At the demo point of 128, pulling the single Thunderbolt cable leaves 44 named bots with no planner at all: their intercept markers freeze in world space while the player's camera pans smoothly past them, and half a second later they put a coherent cluster of rockets into a doorway the player left a full room ago — while `sv_lagreporting` stays completely silent, because the game loop is fine and only the minds have stopped.
