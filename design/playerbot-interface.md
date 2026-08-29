# Playerbot Interface: the causal levers from strategy weights to bot behavior

Scope: establish, in *real Xonotic bot AI code*, the concrete mechanisms by which our
per-bot / per-team strategy weights change what bots actually do. Every mechanism claim is
cited `file:line` against `~/dox/xonotic/build-qc/qcsrc`. Items I could not prove from code
are marked **(inference)**. Nothing here modifies engine code.

Paths are relative to `~/dox/xonotic/build-qc/qcsrc/`.

---

## 0. TL;DR

- The **single causal lever that matters** is `navigation_routerating(this, e, f, rangebias)`
  (`server/bot/default/navigation.qc:1220`). It is the *only* way a goal enters a bot's
  route. Our weights become the `f` (base rating) argument for candidate goals. Everything
  else is plumbing around it.
- The **write-path already exists** end-to-end for the payload gametype:
  `mesh_gather`/`mesh_scatter` builtins (`payload/sv_payload_mesh.qc:18-19`) push a per-client
  observation row to the coprocessor and pull back per-bot weights `plc_mesh_w0..w4`, a discrete
  `plc_mesh_pick`, and a `plc_mesh_lead` flag; `havocbot_goalrating_payload`
  (`payload/sv_payload.qc:606`) consumes them as routerating biases. This is real, compiled,
  and wired into the bot role via the `HavocBot_ChooseRole` hook (`payload/sv_payload.qc:664`).
- **Skill-orthogonality is structurally guaranteed**: rating only sets `goalentity`/route
  (`navigation.qc:1355,1423`); aim/dodge/duel live in a physically separate file
  (`aim.qc`) driven exclusively by `skill` and `bot_*aim*` cvars (`aim.qc:164-239`), which
  routerating never touches. Biasing a goal cannot make a bot aim better or worse.
- **Real gaps** (the "must-build" list): `hunt rival r` is not wired into the payload role and
  has no single-target primitive; `explore/grab at map-cell c` has no cell-targeted primitive;
  spawn-timing and travel-commitment levers exist as per-entity fields but are **not** scattered
  from the mesh yet; and the observation row is fed from **omniscient world state**, not bot
  perception — that last one is the riskiest gap.

---

## 1. The one causal chain (the whole game)

A havocbot's per-tick brain is `havocbot_ai` (`havocbot/havocbot.qc:35`). Strategy (goal
selection) only runs when this bot holds the round-robin strategy token
(`havocbot.qc:52`), which calls the bot's role function pointer:

```
this.havocbot_role(this);            // havocbot.qc:64
```

`.havocbot_role` is a **per-bot function-pointer field** (`havocbot/roles.qc:243`,
`havocbot/havocbot.qh`). That is the clean per-bot hook: a gametype swaps it in via the
`HavocBot_ChooseRole` mutator hook (`roles.qc:245` calls the hook; payload registers its own at
`payload/sv_payload.qc:664`). No global patching needed — each bot's strategy is a swappable
function.

Every role body is the same three-part sandwich:

```
navigation_goalrating_start(this);   // navigation.qc:1830  -> bestrating=-1, markroutes
    ... N calls to navigation_routerating(this, candidate, f, rangebias) ...
navigation_goalrating_end(this);     // navigation.qc:1845  -> route to bestgoal
navigation_goalrating_timeout_set(this);
```

`navigation_goalrating_start` resets `navigation_bestrating = -1` and floods the waypoint graph
with travel costs from the bot (`navigation.qc:1836`, `navigation_markroutes`
`navigation.qc:1081`). Then each `navigation_routerating` call:

1. finds the candidate's nearest waypoint and its precomputed travel cost `cost`
   (`navigation.qc:1400-1414`);
2. **discounts the caller-supplied base rating by distance**:
   `f = f * rangebias / (rangebias + cost)` (`navigation.qc:1417`; jetpack path
   `navigation.qc:1349`);
3. keeps the argmax: `if (navigation_bestrating < f) { navigation_bestrating = f;
   navigation_bestgoal = e; }` (`navigation.qc:1419-1423`).

`navigation_goalrating_end` then commits: `navigation_routetogoal(this, navigation_bestgoal, ...)`
pushes the chosen entity onto the goal stack and builds the path (`navigation.qc:1852`,
`navigation_routetogoal` `navigation.qc:1428`). The bot walks it in `havocbot_movetogoal`
(`havocbot.qc:445`, invoked `havocbot.qc:173`).

**Consequence for us:** the base rating `f` we pass for each candidate is the whole lever.
`rangebias` controls how sharply distance discounts it (larger `rangebias` = distance matters
less; see the multiplicative form). A higher `f` for "cart 0's node 3" than for anything else
*is* the bot deciding to go push cart 0 at that spot. This is not an imagined interface —
`havocbot_role_generic` (`roles.qc:213`) and `havocbot_role_payload` (`payload/sv_payload.qc:647`)
are literally built from these calls today.

---

## 2. What already reaches the bots (existing levers)

These are compiled and live. The payload mode already implements a coprocessor bridge that is a
near-exact instance of our pipeline.

### 2.1 The coprocessor FFI

Six engine builtins are bound in QC (`payload/sv_payload_mesh.qc:18-23`):

| builtin        | # | role |
|----------------|---|------|
| `mesh_open`    | 644 | open a `(node, width, maxrows)` shared matrix, returns handle |
| `mesh_gather`  | 648 | copy a QC field column *up* into the matrix (observation) |
| `mesh_scatter` | 649 | copy a matrix column *down* into a QC field (strategy output) |
| `mesh_publish` | 650 | mark a tick's rows ready for the coprocessor |
| `mesh_poll`    | 651 | read back the latest completed plan sequence number |
| `mesh_stat`    | 653 | status/peer query |

Handle opened lazily with retry (`payload_mesh_attach` `sv_payload_mesh.qc:27`), width
`PLC_MESH_WIDTH = 72` columns, `maxclients` rows (`sv_payload_mesh.qh:8`). The whole exchange is
driven once per cart tick from the lead cart's think (`payload/sv_payload.qc:419`, guarded to
`plc_cart_id == 0` at `:414`; cadence `PLC_TICK`).

### 2.2 Gather (observation up) — 72 columns

`payload_mesh_tick` fills `.plc_mrow_*` fields per client then gathers columns 0..71
(`sv_payload_mesh.qc:223-360` fill, `:329-401` the `mesh_gather` calls). Columns carry: bot
id/team/health/armor/ammo, position/velocity (÷1024), nearest-cart distance & progress,
friend/enemy counts in `PLC_MESH_RADIUS=700` (`:225-232`), current team objective, weapon count,
powerup flag, **time-since-spawn** (`plc_mrow_tss` = `time - plc_spawn_stamp`, `:271`), the four
carts' progress/control/regress triples, and eight ranked item-post rows (rank, xyz, availability).

### 2.3 Scatter (strategy down) — the actual per-bot output

After `mesh_poll` reports a newer plan (`sv_payload_mesh.qc:402`), the bridge scatters seven
columns back into per-bot fields (`sv_payload_mesh.qc:419-425`):

```
mesh_scatter(h, 1, plc_mesh_pick, ...);   // discrete instrument index
mesh_scatter(h, 2, plc_mesh_w0, ...);     // continuous weight over 5 objective slots
... w1..w4 ...
mesh_scatter(h, 7, plc_mesh_lead, ...);   // "you are the designated leader" flag
```

- **`plc_mesh_w0..w4`** = a per-bot continuous weight vector over `PLC_MESH_OBJECTIVES = 5`
  slots (`sv_payload_mesh.qh:9,98-102`). Validated as a simplex-ish distribution
  (sum in [0.5,1.5], all ≥ -0.01) in `payload_mesh_weights_valid` (`sv_payload_mesh.qc:148`).
- **`plc_mesh_pick`** = a discrete index selecting either an objective node (`0..nobj-1`) or an
  item post (`PLC_MESH_ITEM_BASE..`) (`sv_payload_mesh.qc:444-449`, consumed `payload/sv_payload.qc:613`).
- **`plc_mesh_lead`** promotes a waypoint sprite for the leader (`sv_payload_mesh.qc:427-433`).

### 2.4 Weights → routerating bias (the payoff)

`havocbot_goalrating_payload` (`payload/sv_payload.qc:606`) is where our numbers become behavior:

- **push/suppress a cart:** every cart is rated by base `ratingscale` (=20000 from the role,
  `sv_payload.qc:610,655`) with a wide `rangebias=5000`.
- **the 5-slot weight vector spreads the bot along the chosen cart's path:** for each objective
  slot `o`, `wgt = payload_mesh_weight(this, o)` and
  `navigation_routerating(this, payload_mesh_node(cbase+o), ratingscale*2*wgt, 3000)`
  (`sv_payload.qc:623-631`). `payload_mesh_node` maps slot index → an actual path node entity by
  fractional position along the cart's node chain (`sv_payload_mesh.qc:159`). So `w0..w4` literally
  answer *where along the route this bot commits*.
- **contest an item post:** if `plc_mesh_pick` selects an item slot, that item entity is rated
  `ratingscale*2` (`sv_payload.qc:615-618`), item entity resolved by `payload_mesh_obj_ent`
  (`sv_payload_mesh.qc:86`).
- **temporal smoothing:** `payload_mesh_weight` lerps the previous plan's value `v` (`plc_mesh_v*`)
  to the new committed value `c` (`plc_mesh_w*`) across `plan_span` (`sv_payload_mesh.qc:131-146`).
  This is where the "integrate over time" story currently lives — **but see §6**.

**This is a working, real causal path.** Change a bot's `plc_mesh_w*` and its rated goal set
changes, its argmax goal changes, and it walks somewhere else — with no code change, purely by
the scattered numbers.

### 2.5 Cadence / commitment already present

`navigation_goalrating_timeout_set` (`navigation.qc:19`) writes `this.bot_strategytime`
per-bot; `navigation_goalrating_timeout` (`navigation.qc:43`) gates re-rating. This is a
**per-entity field**, i.e. already a per-bot travel-commitment knob (see §5).

---

## 3. Instrument-by-instrument causal map

Our instrument set vs. what the engine actually exposes:

| Instrument | Status | Mechanism / where it attaches |
|---|---|---|
| **push cart k** | **EXISTS** | `routerating(payload_carts[k], ...)` `sv_payload.qc:610`; refine spot with `w0..w4` over `payload_mesh_node` `sv_payload.qc:623-631` |
| **suppress cart k** | **PARTIAL** | same *goal entity* as push — cart movement is decided by capture-zone occupancy, so "be at/ahead of the enemy-controlled cart" is expressible via the 5 path-node slots, but there is **no distinct block-behavior** beyond presence. Same routerating primitive; semantics collapse into "go to cart k at node o". Mark **(inference)** that spatial slotting ahead of the cart == suppression. |
| **contest item post p** | **EXISTS** | `plc_mesh_pick` → `payload_mesh_obj_ent(p)` → `routerating(item, ratingscale*2)` `sv_payload.qc:615-618`; item posts ranked/scanned `sv_payload_mesh.qc:45-83` |
| **hunt rival r** | **MUST-BUILD** | the *primitive* exists — `navigation_routerating` accepts `IS_PLAYER(e)` targets and rates them via their nearest waypoint (`navigation.qc:1227-1268`), and `havocbot_goalrating_enemyplayers` (`roles.qc:158`) already does this for the generic role. **But** the payload role never calls it (`sv_payload.qc:652-658` omits it), and the existing function rates *all* visible enemies, not a *specific* rival `r`. Need: a `havocbot_goalrating_hunt(this, target, scale)` that rates one chosen enemy entity, wired into the payload role with a per-bot weight/target scattered from the mesh. |
| **explore / grab pickup at map-cell c** | **PARTIAL / MUST-BUILD** | *grab pickup*: `havocbot_goalrating_items` exists and is called (`sv_payload.qc:656`, def `roles.qc:104`). *explore a specific cell c*: only `havocbot_goalrating_waypoints` exists and it rates waypoints with a **random** `f = 0.5+random()` in an annulus (`roles.qc:16-42`) — there is **no** primitive to bias a *named* map-cell. Need a `havocbot_goalrating_cell(this, cell_org, scale)` that rates waypoints near `cell_org` (routerating on each `g_waypoints` entity within radius — primitive is trivial, wiring is absent). |
| **commit spawn timing** | **LEVER EXISTS, NOT WIRED** | see §5 |
| **commit travel time** | **LEVER EXISTS, NOT WIRED** | see §5 |

---

## 4. Skill-orthogonality: proven by file separation

The owner's constraint is that steering sets *where* a bot commits, never *how well* it fights.
This is not something we must be careful about — it is structural:

- **Where** is decided entirely inside `navigation.qc` goal rating: the only outputs are
  `navigation_bestgoal` / `goalentity` and the route stack (`navigation.qc:1355,1423,1852`).
- **How well** (aim, lead, jitter, reaction, dodge) lives in `aim.qc`, a separate translation
  unit whose behavior is a pure function of `skill`, `this.bot_offsetskill`, `bot_badaimoffset`
  and the `autocvar_bot_ai_aimskill_*` filters (`aim.qc:164-239`; e.g. aim error injected at
  `:195-196`, N-th order aim filters `:229-239`). None of these read `goalentity`,
  `navigation_bestrating`, or any `plc_mesh_*` field.
- Weapon choice (`havocbot_chooseweapon` `havocbot.qc:1492`) and enemy selection
  (`havocbot_chooseenemy` `havocbot.qc:1333`) are likewise skill/threat-driven, invoked
  independently of the role's goal rating (`havocbot.qc:120-133`).

Therefore: our lever is a *base-rating multiplier on a candidate goal*. It cannot reach the aim
pipeline. Skill-orthogonality holds by construction as long as we only ever call
`navigation_routerating` (or set `.havocbot_role`) and never write `skill`/`bot_*aim*` fields.
**This is the guardrail to state in the pipeline contract.**

---

## 5. Spawn / travel commitment

Both are expressible; only the mesh wiring is missing.

**Travel commitment (how long a bot stays married to its goal before re-rating):**
`this.bot_strategytime` is a per-entity field (`navigation.qc:22,24,38-40,51`). Helpers already
exist to push it out (`navigation_goalrating_timeout_extend_if_needed` `navigation.qc:49`) or
pull it in (`navigation_goalrating_timeout_expire` `navigation.qc:35`). **Lever exists per-bot.**
Missing: a scattered mesh column (e.g. `plc_mesh_commit`) that the role applies after
`navigation_goalrating_end` to set this bot's next re-rating horizon. Small add.

**Spawn timing (hold a bot dead / schedule respawn):**
- `this.respawn_time` and `this.respawn_flags` are per-entity (`server/client.qc:1341`
  `calculate_player_respawn_time`, respawn fires when `time > respawn_time`
  `client.qc:2154`, `RESPAWN_DENY` blocks it `client.qc:2162`). Pushing `respawn_time` later
  holds a bot dead; `RESPAWN_DENY`/`RESPAWN_FORCE` gate it. **Lever exists per-bot.**
- `player_blocked` (`server/client.qh:338`) is a per-player field that suppresses firing
  (`server/weapons/weaponsystem.qc:429`) and self-kill (`server/clientkill.qc:210`); payload
  already toggles it at round boundaries (`sv_payload.qc:599,694,706`). Usable to *stage* a
  spawned-but-held bot **(inference — its use here would be novel; verify it does not also
  freeze movement in an undesired way)**.
- The observation side is already present: `plc_mrow_tss` (time-since-spawn,
  `sv_payload_mesh.qc:271`) is gathered, so the coprocessor can see spawn state.

Missing for both: an outbound scatter column and a few lines in the role/spawn hook to apply it.
Neither requires an engine change — both fields are ordinary QC entity fields.

---

## 6. Perception → observation buffer (the honest gap)

The task requires the per-team observation buffer of contextual events ("saw item spawn/despawn")
to come from **bot perception**, not omniscient world state. Today it does **not**:

- Item availability fed to the mesh is read straight off the item entity's global status:
  `payload_mesh_item_avail[p] = (... ie.ItemStatus & ITS_AVAILABLE)` (`sv_payload_mesh.qc:186-190`).
  Every bot's row sees the same world-truth availability regardless of line-of-sight.
- Friend/enemy counts are a `findradius` over the world (`sv_payload_mesh.qc:229-236`) — proximity,
  not vision.
- More broadly, Xonotic bots navigate an omniscient waypoint graph: `havocbot_goalrating_items`
  rates every item in radius (`roles.qc:104`), and the "rate only visible enemies" traceline in
  `havocbot_goalrating_enemyplayers` is **commented out** (`roles.qc:171-175`). There is no
  per-bot perceived-item memory or visibility-gated event log anywhere in the bot code.

So a truthful observation buffer is a **must-build**. The raw materials exist — bots do run
tracelines (`aim.qc`, `navigation.qc` visibility checks) and `havocbot_chooseenemy` maintains a
seen-enemy notion (`havocbot.qc:1333`) — but there is no code that (a) gates item/event visibility
per bot by line-of-sight and (b) writes spawn/despawn *events* (edge-triggered) into a per-team
ring the gather can read. Building this means adding a perception pass that stamps per-bot
"last saw item p available/unavailable at time t" fields, then gathering those instead of the
global `ITS_AVAILABLE`.

**Also note (velocity vs. snapshot):** our stated pipeline output is a *velocity* `dw/dt` that QC
integrates. The current interface scatters **absolute** weights `plc_mesh_w*` and does only an
inter-plan lerp (`sv_payload_mesh.qc:131-146`). If integration is meant to live coprocessor-side
(pipeline integrates internally, scatters the integrated weight), the present code is correct and
the lerp is just smoothing — **(inference)** this is the cleaner reading. If QC-side integration
is required, it is absent and must be added (accumulate `w += dwdt * dt` per bot before rating).
Flag this as a spec decision, not a bug.

---

## 7. Must-build list (strict)

1. **`hunt rival r` primitive + wiring.** Add a single-target enemy rater and call it from the
   payload role with a scattered target id + weight. Primitive supported by
   `navigation_routerating` on `IS_PLAYER` (`navigation.qc:1227`); function and role wiring absent.
2. **`explore/grab at map-cell c` primitive.** Add a cell-targeted waypoint rater (routerating on
   waypoints near a scattered cell origin). Only a *random* waypoint rater exists (`roles.qc:16`).
3. **Spawn-timing scatter.** Add a mesh column that drives `respawn_time`/`respawn_flags`
   (`client.qc:1341,2154,2162`) or `player_blocked` (`client.qh:338`). Levers exist; not wired.
4. **Travel-commitment scatter.** Add a mesh column that sets `bot_strategytime`
   (`navigation.qc:49`) after goal rating. Lever exists; not wired.
5. **Perception-gated observation buffer.** Replace omniscient `ITS_AVAILABLE`/`findradius`
   feeds (`sv_payload_mesh.qc:186-236`) with per-bot line-of-sight-gated item/event perception,
   and add an edge-triggered spawn/despawn event ring. This is genuinely new code.
6. **(spec) velocity integration decision** — confirm coprocessor-side integration (current) vs.
   QC-side (`w += dwdt*dt`), see §6.

Items 1–4 are small (primitives exist; only wiring/one function each). Item 5 is the real build.

---

## 8. The single riskiest gap

**The observation buffer is fed from omniscient world state, not bot perception**
(`sv_payload_mesh.qc:186-236`; visibility traceline for enemies is even commented out at
`roles.qc:171-175`). Every other lever is either present or a few lines of wiring over an existing
primitive. But perception is load-bearing for the whole thesis: if the coprocessor plans on
ground-truth the bots could not legitimately see, the demonstrated "causal impact on bot behavior"
is contaminated by information the bots don't have, and the strategy layer is really steering on
oracle state. Fixing it requires new per-bot perception + an event ring — the one place we must
build rather than wire.

---

## 9. Summary for the owner

- **Existing levers (real, compiled):** `navigation_routerating` base-rating bias
  (`navigation.qc:1220`); per-bot `.havocbot_role` swap via `HavocBot_ChooseRole`
  (`roles.qc:245`, `sv_payload.qc:664`); the full `mesh_open/gather/scatter/publish/poll` bridge
  (`sv_payload_mesh.qc:18-23`) delivering per-bot `plc_mesh_w0..w4` + `plc_mesh_pick` + `plc_mesh_lead`
  and consuming them in `havocbot_goalrating_payload` (`sv_payload.qc:606`) for push-cart,
  path-slot, and item-post instruments; per-bot `bot_strategytime` and `respawn_time` fields.
- **Must-build:** single-target hunt rater; cell-targeted explore rater; spawn-timing and
  travel-commitment scatter columns; and a perception-gated observation buffer with a
  spawn/despawn event ring.
- **Riskiest gap:** the observation buffer is omniscient, not perceived — the only lever that
  needs real new code and the only one that can invalidate the causal-impact claim.
