# Strategy I/O — the server-def boundary between Xonotic and the mesh

This document specifies the **exact** server-side I/O the strategy stack needs: the
hooks it attaches to, the mesh column layout it reads and writes, and the implemented
boundary. It is the contract for the three flows the strategy
operator requires:

1. **cartstate + global feature rows** emitted up to the mesh every strategy step
   (cartstate **guaranteed** present) — the Game-1 state `s`;
2. **perception-gated observation writes** (frustum + LOS + 2-V-cell) feeding the
   observation buffer — **not** omniscient `ITS_AVAILABLE`;
3. the **scatter column schema** carrying per-player instrument weights back into the
   payload rater as `navigation_routerating` biases.

Authorities (where doc and code disagree, the doc is intent): `design/payload-spec.md`
(§2.1-2.2 data flow, §3.2 exec, §4 interface), `design/rl-training-spec.md` (§1 Game-1
computed features, §4 computed/learned/frozen boundary), `design/playerbot-interface.md`
(the levers, file:line), `design/dpp-mixing-and-overlay.md` (§4 cadence = step size).

Nothing here modifies the working cart mechanic. The strategy I/O is additive and
name-spaced; `sv_payload.qc` supplies the strategy-step, rater, spawn, and death hooks.

---

## 0. TL;DR

- **Substrate is real and compiled.** The six mesh builtins
  (`mesh_open`/`gather`/`scatter`/`publish`/`poll`/`stat`, `bridge/qc/mesh_ipc.qh`,
  engine `bridge/engine/mesh_ipc.c`) exist. The response arena is **fixed at 8 floats
  per row** (`MESH_XON_RESPWIDTH`, `mesh_ipc.c:15`) — so the per-player scatter schema
  is **exactly 8 columns**, a hard constraint, not a design choice. Request rows are up
  to 256 floats wide.
- **Hooks are real and compiled.** `.havocbot_role` swap via `HavocBot_ChooseRole`
  (`sv_payload.qc:636`), the payload rater `havocbot_goalrating_payload`
  (`sv_payload.qc:604`), the single lever `navigation_routerating` (`navigation.qc:1220`),
  and the lead-cart strategy gate `if (plc_cart_id == 0)` inside `payload_think`
  (`sv_payload.qc:414`). Cartstate fields (`plc_s`, `plc_length`, `plc_ctrl`,
  `plc_speed_now`, `plc_idle`, `sv_payload.qh`) all exist.
- **The strategy path is implemented.** It fills and gathers all three request streams,
  applies the 8-column scatter, and consumes cart, item, rival, cell, commitment, and
  spawn instruments through real game hooks.
- **Perception is gated.** Item and enemy events enter only after range, PVS, and LOS
  checks. The responder deduplicates the ring, contracts stale observations toward
  priors, and constructs the live V-cell graph from observed motion.

---

## 1. Callsite and cadence

The strategy step reuses the **existing** lead-cart gate — no new think loop is added.

```
payload_think (sv_payload.qc:297)             // per game tick, ~PLC_TICK = 0.1s
  ...
  if (this.plc_cart_id == 0) {                // sv_payload.qc:414  (already the strategy gate)
      // throttle to PLC_STRAT_TICK and call:
      payload_strategy_step(this);            // gather up, scatter down (one mesh exchange)
  }
```

- `PLC_STRAT_TICK = 0.2` (5 Hz) sits inside the spec's 1-10 Hz strategy band
  (`payload-spec §1`). It **is** the forward-Euler step size `Delta`
  (`dpp-mixing-and-overlay §4`): a stability parameter carrying a discrete-time chaos
  threshold, chosen against stability — not a scheduling knob.
- The rater (`havocbot_goalrating_strategy`) runs **later and per-bot**, when each bot
  holds the strategy token in `havocbot_ai` and calls its `.havocbot_role`
  (`havocbot.qc:64`). It reads the freshly-scattered absolute `w_b`. **Both scales
  integrate off-engine** (`payload-spec §3.2`); the engine only reads a buffer.

---

## 2. Three gather streams (observation up)

Distinct `mesh_open` handles keep per-player rows, cartstate rows, and event-ring rows
off a shared column axis. The engine allows up to 4 handles (`MESH_XON_HANDLES`).

### 2.A Per-player observation `x_b` — one row per client edict (width 40)

The bot's **own, known** state only (`payload-spec §2.1`). Enemy and item truth do
**not** live here — they enter solely through the perception ring (2.C).

| col | name | source |
|-----|------|--------|
| 0 | ID | client entnum |
| 1 | TEAM | team index (the `A_team` selector, `rl-training-spec §0`) |
| 2 | HEALTH | `health` |
| 3 | ARMOR | `armorvalue` |
| 4 | AMMO | summed pools / 100 |
| 5-7 | POS_XYZ | `origin` / 1024 |
| 8-10 | VEL_XYZ | `velocity` / 1024 |
| 11 | WEAPONS | `weapons` bitfield (float-encoded) |
| 12 | POWER | powerup timer remaining |
| 13 | TSS | time-since-spawn (spawn state made observable) |
| 14 | CELL | resolved V-cell id (`payload_str_cell_of`, `payload-spec §2.2.2`) |
| 15 | NCART | nearest-cart index |
| 16 | NCART_D | nearest-cart distance / 1024 |
| 17 | ALIVE | 1 alive / 0 dead (drives the spawn scatter) |
| 18 | CONTROL | 1 bot / 0 human; strategy assignments exist for both |
| 19-39 | reserved | zero-filled `x_b` growth |

The responder filters empty client slots but does not filter by `CONTROL`. Bot and
human rows enter the same shared policy. `CONTROL` identifies the realized controller
for off-policy telemetry; the policy does not withhold an assignment from either
class. The havocbot rater enacts bot assignments. A human assignment is presently an
internal advisory signal until a HUD channel or inverse realized-action classifier is
built.

### 2.B Cartstate `s` — one row per cart edict (width 12) — **GUARANTEED**

The guaranteed member of the global feature vector (`rl-training-spec §1`). Filled
**unconditionally every step**; if this emit is ever skipped that is a `[BUILD]` bug,
per spec. `PW(s)` and `SUCC(s)` are **computed on the mesh from these rows** — they are
Game-1, closed-form, ¬learned, and are **not** emitted here.

| col | name | source |
|-----|------|--------|
| 0 | ID | `plc_cart_id` |
| 1 | DEPTH | `plc_s / plc_length` — arclength fraction along the golden path |
| 2 | LENGTH | `plc_length` (raw, for de-norm) |
| 3 | CTRL | controlling team index (0 = uncontrolled) |
| 4 | SPEED | `plc_speed_now` (signed; < 0 = regressing) |
| 5 | IDLE | `plc_idle` (rollback timer) |
| 6 | BANKMASK | banked-team bitmask since last origin touch |
| 7 | PROGRESS | monotone banked score for this cart's lane |
| 8-11 | reserved | — |

### 2.C Perception event ring — edge-triggered rows (width 6, 256 rows)

The **only** producer of enemy/item rows and the **only** path by which enemy positions
enter the system (`payload-spec §2.2.1`). Produced by `payload_perceive` under the
three-gate test (§3). Replaces the omniscient `ITS_AVAILABLE`/`findradius` feeds the
interface doc flagged as the riskiest gap (`playerbot-interface §6, §8`).

| col | name | meaning |
|-----|------|---------|
| 0 | CELL | V-cell id where the event was seen |
| 1 | KIND | 0 item_gone · 1 item_here · 2 enemy_here · 3 rival_here |
| 2 | TEAM | observing team (the per-team buffer key) |
| 3 | SUBJECT | item post `p` / rival entnum `r` (the instrument target id) |
| 4 | VALUE | kind-specific (respawn-phase estimate, threat) |
| 5 | TIME | observation timestamp (feeds temporal contraction, `§2.2.3`) |

---

## 3. The perception gate (`payload_perceive`)

Per bot, per strategy step. An event is deposited into the per-team ring **only** when
all three gates pass (`payload-spec §2.2.1`). All three are necessary; the 2-cell cap
stops one pathological long sightline from rewriting the whole buffer.

1. **Frustum + PVS** — `checkpvs(bot.origin + bot.view_ofs, target)`
   (engine builtin; used across `navigation.qc:1524,1574,1608`).
2. **Line of sight** — `traceline(eye, target.origin, MOVE_NOMONSTERS, bot)` unobstructed
   (the same primitive `roles.qc:171-175` leaves commented out).
3. **Range** — target's V-cell within **2 V-cells** graph distance of `bot.plc_str_cell`.

Deposits are **edge-triggered**: emit only when `(cell, kind, subject)` differs from this
team's last stamp, so the ring carries *events*, not a per-tick dump. Consequences fall
straight out of the gate (`payload-spec §2.2.1`): **stealth is emergent** (an unobserved
body/pickup is simply absent from the enemy buffer), and **enemy positions are
featurized only here**. Dead bots have no viewpoint — a physics fact, expressed by early
return, not a capability refusal (`AGENTS.md`).

---

## 4. Scatter schema (strategy down) — exactly 8 columns

The response arena is fixed at 8 floats/row (`mesh_ipc.c:15`). Columns carry the
**absolute, mesh-integrated** `w_b` (`payload-spec §3.2`: the engine reads the integrated
weight and **never integrates**; any QC-side lerp is cosmetic smoothing, not the flow).
Each column is a `navigation_routerating` bias.

| col | name | field | becomes |
|-----|------|-------|---------|
| 0 | TARGET | `plc_str_target` | packed committed target id (§5) |
| 1 | GAIN | `plc_str_gain` | base rating amplitude `f` for that target |
| 2 | LANE | `plc_str_lane` | arclength-fraction **readout** [0,1]; **not** consumed as navigation — push vs suppress is resolved by the cart's cylinder occupancy law (`payload_occupancy`), not by routing a lane node |
| 3 | HUNT | `plc_str_hunt` | weight rating the observed rival subject |
| 4 | EXPLORE | `plc_str_explore` | weight rating waypoints whose hashed V-cell equals the selected cell |
| 5 | COMMIT | `plc_str_commit` | travel-commitment horizon → `bot_strategytime` (`navigation.qc:51`) |
| 6 | SPAWN | `plc_str_spawn` | spawn-timing delta applied after death by `PlayerPreThink` |
| 7 | LEAD | `plc_str_lead` | swing/leader **readout** flag (intercentrality argmax; `dpp §3` ontology discipline — a derived readout, never a primitive entity) |

Why a packed target + weights rather than a full per-instrument vector: 8 columns cannot
hold a dense velocity over {push·k, suppress·k, contest·p, hunt·r, explore·c} for 4 carts.
The mesh integrates the dense flow and scatters the **compact commitment** — the argmax
target plus the amplitudes the rater needs. This matches the interface doc's existing
`pick + w0..w4 + lead` shape (`playerbot-interface §2.3`), re-pinned to the spec's
instrument set.

---

## 5. Target encoding (`PLC_SC_TARGET`)

One scalar selects the committed instrument's target. Bases are pinned in
`sv_payload_strategy_io.qh` **and** mirrored in `tools/strategy_io_schema.py`
(`encode_target`/`decode_target`, round-trip-tested):

```
cart  k :       0 + k     (PLC_TGT_CART_BASE)
item  p :   65536 + p     (PLC_TGT_ITEM_BASE)
rival r :  131072 + r     (PLC_TGT_RIVAL_BASE)
cell  c :   196608 + c     (PLC_TGT_CELL_BASE)
```

`havocbot_goalrating_strategy` dispatches on the band and issues the
matching `navigation_routerating` call.

### 5.1 Objective vocabulary → target entity (the thin-translator contract)

The adapter is a **pure translator**. It decides nothing and computes no navigation
(no node-finding, no lane geometry, no "which point on the path" — stock navigation
owns HOW). Each objective the matmul can commit to maps to exactly one
`navigation_routerating` call on a **real, stock-navigable entity**, with the weight
the mesh scattered:

| objective (matmul) | target band | routed entity | weight | rangebias |
|--------------------|-------------|---------------|--------|-----------|
| push / suppress cart k | `CART_BASE+k` | `payload_carts[k]` (the cart entity) | `GAIN` | 5000 |
| contest / gather item post p | `ITEM_BASE+p` | the `g_items` entity `etof==p` | `GAIN` | 3000 |
| crush weak / duel strong / hunt rival r | `RIVAL_BASE+r` | the enemy player entity `etof==r` | `HUNT` | 5000 |
| explore / gather cell c | `CELL_BASE+c` | stock `g_waypoints` in cell c | `EXPLORE` | 3000 |

- **push vs suppress is not re-encoded by the adapter.** Both are the same
  routerating on the same cart entity; whether a present body pushes or contests is
  resolved by the cylinder occupancy law (`payload_occupancy`, `sv_payload.qc`;
  `design/cart-force-field.md`), which the adapter must not duplicate. `LANE` is a
  readout column only.
- **crush-weak / duel-strong / hunt-rival** are one primitive: routerating a rival
  player entity. Which rival (weak vs strong) is the mesh's target choice (`r`), not a
  QC decision.
- **explore / gather resources** route stock waypoint / item entities in the target
  cell — never a computed path. The cell→entity step is a filter over the stock
  navmesh, not a project nav graph.
- `COMMIT` extends `bot_strategytime` (the stock travel-commitment lever,
  `navigation.qc:51`) so a committed instrument is not re-diced every tick. `SPAWN`
  biases `respawn_time` in `PlayerPreThink`. `LEAD`/`LANE` are readouts. None of these
  is navigation.

The **cart demonstrator** (`havocbot_goalrating_payload`) likewise rates only the cart
**entities** — it no longer routes bots along `plc_path` nodes. The `plc_path` chain is
the cart's own movement/banking graph, not a bot navigation graph.

---

## 6. Wiring

| piece | status | where |
|-------|--------|-------|
| mesh builtins (open/gather/scatter/publish/poll/stat) | **exposed** | `bridge/qc/mesh_ipc.qh`, `bridge/engine/mesh_ipc.c` |
| 8-col fixed response arena | **exposed** (constraint) | `mesh_ipc.c:15` |
| `.havocbot_role` + `HavocBot_ChooseRole` | **exposed** | `sv_payload.qc:636`, `havocbot.qc:64` |
| `navigation_routerating` (the lever) | **exposed** | `navigation.qc:1220` |
| lead-cart strategy gate | **exposed** | `sv_payload.qc:414` |
| cartstate fields (`plc_s`/`plc_length`/`plc_ctrl`/`plc_speed_now`/`plc_idle`) | **exposed** | `sv_payload.qh` |
| `bot_strategytime` (travel commit) | **exposed** | `navigation.qc:19,51` |
| `respawn_time`/`respawn_flags` (spawn timing) | **exposed** | `client.qc:1341,2154` |
| gather fills + `mesh_gather` for schemas A/B/C | **implemented** | `payload_strategy_gather` |
| perception gate + event ring | **implemented** | `payload_perceive` |
| 8-col `mesh_scatter` apply | **implemented** | `payload_strategy_scatter` |
| strategy-weighted rater | **implemented** | `havocbot_goalrating_strategy` |
| hunt / explore / spawn / commit primitives | **implemented** | strategy rater plus payload death/pre-think hooks |
| V-cell segmentation | **implemented, coarse** | `payload_str_cell_of` |

`_mod.inc` includes the strategy implementation under SVQC. `payload_think` invokes
the throttled exchange at the lead-cart gate, `havocbot_goalrating_payload` adds the
strategy rater, `PlayerSpawn` records time-since-spawn, `PlayerDies` opens the timing
window, and `PlayerPreThink` applies later strategy deltas once rather than extending
the timer every frame. A clean overlay of the stock Xonotic source compiles both SVQC
and CSQC with these hooks.

---

## 7. Determinism and skill-orthogonality (the two guardrails)

- **Determinism boundary** (`rl-training-spec §4`). Everything on the QC side of the wire
  is **stopgrad**: QC emits raw cartstate and raw perception and never sees a gradient.
  `PW`/`SUCC`/the belief/V-cells are state features computed after observation
  (numpy/plain python — `tools/strategy_io_schema.py`); the DPP signal, shared edge
  mixing head, asymmetric `W`/`L` critics, and local dynamics ensemble are learned on the mesh.
  The policy/critic gradient does not flow into QC; transition prediction is supervised
  by later QC observations. This
  file's Python imports **no** mlx and stays test-light by design.
- **Skill-orthogonality** (`playerbot-interface §4`). Every scattered column becomes a
  `navigation_routerating` base-rating bias and **nothing else**. The rater never writes
  `skill` or `bot_*aim*`; aim/dodge live in `aim.qc:164-239`, a separate unit that never
  reads `goalentity` or any `plc_str_*` field. We bias *where* a bot commits; we cannot
  touch *how* it fights. This holds by construction as long as the only outputs are
  routerating and `.havocbot_role`.

---

## 8. Files

| file | role |
|------|------|
| `qcsrc/.../payload/sv_payload_strategy_io.qh` | fields, 4 column schemas, target bases, prototypes |
| `qcsrc/.../payload/sv_payload_strategy_io.qc` | attach/gather/scatter/perception/rater/step implementation |
| `tools/strategy_io_schema.py` | canonical column contract + deterministic PW/SUCC/target-codec (numpy), self-tested |
| `STRATEGY-IO.md` | this document |

Run `python3 tools/strategy_io_schema.py` for the contract self-test (schema
disjointness, target-codec round-trip, and the PW nim-sum invariant "one cart at depth 2
beats two at depth 1").
