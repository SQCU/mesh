# Xonotic payload mode (`plc`)

A k-team, k-cart payload gamemode. Each cart is a `MOVETYPE_PUSH` brush driven along
its own waypoint path by contested occupancy. Any number of carts (up to 4) coexist;
every cart is pushed forward by the team that controls it and backward by everyone
else, and teams bank score for every control point a cart crosses while they hold it.

Everything under `qcsrc/` drops into a Xonotic `qcsrc` tree at the same relative path.
`patch/0001-payload-registry-hooks.patch` carries the five one-line-per-file registry
edits to shared files. `cfg/gamemodes-payload.cfg` is the cvar overlay.

## Control and the speed law

Every `PLC_TICK` (0.1 s) each cart recomputes, for each team index `j`:

- `n_j` — live players of team `j` inside `cart.radius` horizontally, inside
  `cart.height` vertically, with line of sight to the cart centre.
- `w_j = Σ_{i=1..min(n_j, push_cap)} push_falloff^(i-1)`.
  `falloff = 1` is Xonotic-native capped-linear; `0.5` is TF2 diminishing returns.

**Control**: the team with the strict maximum `w_j` controls the cart. A tie for the
maximum, or an empty radius, leaves the cart uncontrolled. This is the old occupancy
law with the per-team goal-direction rule deleted: direction is now a property of
control, not of goals.

**Direction**: every cart path has one origin (`s = 0`, the first `plc_path` node) and
one end (`s = L`). The controlling team pushes toward the end; every non-controlling
team's occupancy pushes toward the origin. With `c` the controlling index (or none):

```
P+ = w_c (0 if uncontrolled)     P− = Σ_{j ≠ c} w_j
v  = clamp(cart.speed · (P+ − P−), ±g_payload_max_speed)
s' = clamp(s + v · PLC_TICK, 0, L)
```

Two consequences the mode is built around: a controlled cart still regresses when the
combined non-controllers outweigh the controller, and a cart contested to a tie
regresses under everyone's weight — sustained regression returns it to the path
origin. A stall needs `P+ = P−` exactly with players present.

The per-team `plc_goal` entities no longer carry direction or round targets. They
survive as team declarations only: their team bits drive the team count exactly as
before, and their `target`/`radius` keys are inert.

Rollback: after `g_payload_idle_time` seconds with `Σ w_j == 0`, the cart moves at
`g_payload_rollback_speed` toward the path origin, stopping at the first
`PLC_CHECKPOINT` node it meets.

Cart motion is direct velocity (`velocity = (pos(s') − pos(s)) / PLC_TICK`), not
`SUB_CalcMove`, because `SUB_CalcMove` commits to a destination and a traveltime at
issue time and cannot change speed mid-segment.

## Scoring

### Control-point banking (the accrual rule, exactly as implemented)

Control points are the `plc_path` nodes. Per cart, per node, the mode keeps a bitmask
of teams that have banked that node since the cart last touched its origin.

- When a cart moves forward across a node's arclength (`s_prev < node.s ≤ s_new`)
  during a tick in which team `T` controls it, and `T`'s bit is not set on that node:
  `T` banks the node — `TeamScore_AddToTeam(T, ST_PAYLOAD_POINTS,
  g_payload_point_score)`, the bit is set, and the node's sprite recolours to `T`.
- A node whose bit is already set for `T` re-banks nothing, however many times the
  cart shuttles across it while `T` controls. A *different* team crossing it under
  their own control banks it independently (their bit is separate).
- Crossing a node with no controller banks nothing, and control acquired while
  parked past a node banks nothing — banking happens only at a forward crossing.
- When a cart regresses to `s = 0`, every node's mask on that path is cleared and the
  sprites reset: full accrual potential is restored for whoever takes the cart next.
- The origin node itself (`s = 0`) is never bankable.

So a team's banked total is proportional to the number of control points the cart
crossed from origin while under their control, per origin-to-origin excursion.

Every banking prints to the server log
(`payload: bank cart <id> team <team> point <idx> s <arclength>`) and to the event
log (`:plc:bank:<cartid>:<nodeidx>:<team>`); a wipe prints
`payload: cart <id> regressed to origin, progress cleared` and `:plc:origin:<id>`.
These lines are the demo's evidence stream.

`ST_PAYLOAD_POINTS` is a non-primary team field. An earlier revision fed a team score
field continuously every tick and the match ended after one or two rounds even with
`fraglimit 10` (measured both directions on a live server); discrete bankings are a
different shape, but if early match end reappears, `g_payload_point_score` is the
lever.

### Rounds and the rest

- Team: `ST_PAYLOAD_CAPS` ("caps"), primary, +1 per round won.
- Team: `ST_PAYLOAD_POINTS` ("points"), the bankings above.
- Player: `SP_PAYLOAD_PUSH` — `(s' − s) · g_payload_score_rate` split over the
  controlling team's in-radius players while their cart advances.
- Player: `SP_PAYLOAD_BLOCK` — accrued while a cart is occupied and stalled.

Round end: a cart whose controller has pushed it to within `g_payload_capture_radius`
of its path end (`s ≥ L − r`) is delivered — the controlling team wins the round. An
uncontrolled cart parked at the end delivers the moment someone controls it. On round
timeout the team with strictly the most bankings this round wins; otherwise
`CENTER_ROUND_TIED`. Round start resets every cart to its origin and wipes all masks.

## Map entity format

### `func_plc_cart` (brush, 1 to 4)

Each cart gets an id in spawn order (0-based) and its own path.

| key | meaning |
|---|---|
| `target` | targetname of the first `plc_path` node of this cart's path |
| `speed` | track units/second per unit of net push weight (default `g_payload_speed`) |
| `radius` | horizontal push radius (default `g_payload_push_radius`) |
| `height` | vertical push half-band (default `g_payload_push_height`) |
| `dmg`, `dmgtime` | crush damage via `generic_plat_blocked` |
| `spawnflags` | `1` = `PLC_CART_TURN`, face along the track |

Carts always start at their path origin; the old `plc_start` key is gone, since the
control law needs an unambiguous origin. `view_ofs` is assigned after
`InitMovingBrushTrigger`, not before: mins is empty until the brush model is set, and
an earlier assignment left the tracked point at the entity origin so no pusher was
ever in radius.

### `plc_path` (point, ≥ 2 per cart)

`targetname`, `target` (next node; omit on the last), `curvetarget` (quadratic bezier
control point, same key `path_corner` uses), `spawnflags 1` = `PLC_CHECKPOINT`.
Chains must be disjoint between carts; a `target` pointing at the first or an
already-visited node terminates the chain (bounded at 4096 nodes).

### `plc_goal` (point, one per participating team)

`cnt` = team colour index (`4` red, `13` blue, `12` yellow, `9` pink — same convention
as `dom_team`; the mode adds 1 to get the server team id). **The set of `plc_goal`
teams drives the team count** exactly as before; `target` and `radius` are ignored by
the control law.

### `plc_team` (point, optional)

Mirrors `dom_team`: `netname`, `cnt`. If absent, teams are spawned from
`g_payload_default_teams` / `g_payload_teams_override` unioned with the goal teams.

## HUD and sprites

Cart waypoint sprites are labeled per cart ("Cart 1", "Cart 2"; carts 2+ share the
second label). Path-node sprites recolour to the team that last banked them and reset
on an origin wipe. The `PAYLOAD_GOALS_PACKED` stat is repurposed: 6 bits per cart id,
`1 + floor(progress·62)` (0 = no such cart); the HUD draws cart 0 as the main bar
(marker red-shifted while regressing, per-team push ticks at the marker) and a thin
bar per additional cart.

## Building

```
rsync -a /Applications/Xonotic/source/qcsrc/ build/qcsrc/
rsync -a payload/qcsrc/ build/qcsrc/
patch -p1 -d build/qcsrc < payload/patch/0001-payload-registry-hooks.patch
cd build/qcsrc
make QCC=<abs path to gmqcc> QCCFLAGS_WATERMARK=payload qc
```

`gmqcc` must be built first and passed explicitly: the Makefile default
`QCC ?= ../../../../gmqcc/gmqcc` does not resolve in the shipped layout, and
`QCCFLAGS_WATERMARK ?= $(shell git describe ...)` fails outside a git repo. `gmqcc`'s
own Makefile passes `-Wl,--gc-sections`, which Apple `ld` rejects; relink with
`c++ .build/objs/*.o -o gmqcc`.

`tools/mkpatch.sh` regenerates the registry patch. `check-units.sh` compiles each
payload `.qc` as its own translation unit.

## Running it without a payload map

`tools/mkentfile.py <bsp> <out.ent> [teams] [carts]` reads a stock BSP's entity lump
and appends `carts` (default 2) disjoint 5-node `plc_path` tracks. The team-spawn set
is split into contiguous halves along its wider axis so the two polylines are
geometrically separated, each cart takes the next visible inline brush model, and one
`plc_goal` per team is emitted (teams alternate across tracks). Drop the result in
`<userdir>/data/maps/<name>.ent` with a `<name>.mapinfo` carrying `gametype plc`.
`+sv_autopause 0` is required or an empty dedicated server freezes `sv.time` after
~5 s.

`+bot_join_empty 1` is required or no bot ever joins. `bot_fixcount`
(`server/bot/default/bot.qc:640`) only computes a bot target when
`realplayers || autocvar_bot_join_empty || (currentbots > 0 && time < 5)`; on a
headless dedicated server with no human client all three are false.

`tools/checklaw.py <log> <start_time>` re-derives the control/speed law from a
`plcdbg` log (`debug/plcdbg.patch` adds the instrumentation) and reports mismatches.

## The mesh objective hook

`qcsrc/common/gamemodes/gamemode/payload/sv_payload_mesh.qc` is the SVQC side of the
bridge described in `../bridge/PORT.md`. It declares only the six surviving builtins
(`#644`, `#648`–`#651`, `#653`) and holds all of the fabric state the mode has.

Objectives are (cart, node) addressable as a combined index `cart * 5 + node` in
`[0, k·5)`. Cart 0's think (only) stages the width-26 request of PORT.md §2 —
the sixteen base columns (distance/progress now relative to the bot's nearest cart,
the objective a combined index), the four dominance columns, and six per-cart state
columns (progress, controlling team, regression flag for carts 0 and 1) — then
publishes and polls. Nothing waits; a missed response keeps the previous plan.

The response stays width 8: the pick is a combined index, and the five weight columns
are a distribution over the picked cart's path nodes. Per-team majority over the
picks becomes `payload_mesh_objective[team]`, a combined index that
`havocbot_goalrating_payload` maps through `payload_mesh_node()` to the chosen
stretch of the chosen cart's track, rated above the carts themselves so the solver's
allocation is visible in bot movement, not only in a stat.
