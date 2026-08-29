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

## Diegetic communication layer

How the game shows what the carts, paths and objectives are doing. Four channels,
chosen so one spectator reading no HUD text still gets all four theatres at once, and
so the cost stays client-side or reuses networking that already ships.

### Cart paths — map ribbon (option a/c) + active-front colour (option d)

Each consecutive node pair spawns one `ENT_CLIENT_RADARLINK` entity — the same proven
link primitive Onslaught draws its control-point graph with, so **no new netcode and
no per-frame server cost**. The result is a team-coloured ribbon of every cart's whole
route on the radar/minimap, which is exactly the "watch four theatres at once" view
the owner wants. Colour is a single byte per link (`start_idx | end_idx<<4`,
palette-index `Team-1` like Onslaught):

- **banked segment** → the owning team's colour, so captured track visibly advances
  from the origin as a coloured front;
- **the segment the cart is on** → the controlling team's colour (white if
  uncontested) — the contested front is the brightest thing on the path;
- **track ahead** → white.

Link colours update only when they change (`payload_update_links`, called per tick but
sets `SendFlags` only on a real delta), so the network stays quiet on a still cart.

The same link data is reused a second way, fully client-side: `Payload_Ribbon_Draw` is
a single drawable registered into `g_drawables` that reads `g_radarlinks` and lays a
translucent additive **3-D ground ribbon** along each segment in the world itself
(`R_BeginPolygon(..., false)`), coloured by the same per-segment byte. Zero extra
networking — it renders the bytes the radar already received. Gated by
`hud_panel_modicons_payload_ribbon` (default on, width
`..._ribbon_width`) and self-limited to payload by a staleness stamp the panel
refreshes, so it never draws under Onslaught's links.

Per-node sprites (option b) are kept but decluttered — see below — rather than made the
primary path channel: at 4 carts × ~13 nodes a sprite per node is noise, whereas the
ribbon reads as one continuous coloured line.

### Waypoint overlays — visibility rule

All nodes still exist as sprites (banking and the mesh objective marker both hang off
`node.sprite`), but the **rule is set once at spawn, no per-tick churn**:

- **checkpoint nodes** (`PLC_CHECKPOINT`) → unlimited range: the meaningful control
  points are always visible on the skyline;
- **plain shape/curve nodes** → `maxdistance = g_payload_node_fade_dist` (1100u): they
  fade out at range so a distant path is a clean ribbon, not a picket fence, but resolve
  into individual markers when you are actually working that stretch.

On top of the fixed rule, the existing dynamic colouring stands: a banked node wears
the banking team's colour, the mesh-chosen objective node per team gets
`RADARICON_OBJECTIVE` in that team's colour, everything else stays neutral cyan. The
"which point is contested right now" answer is carried by the ribbon's active-segment
colour rather than by re-colouring a node every tick.

### Cart state — legible per cart

`PAYLOAD_CARTS_STATE` (new packed int stat, 5 bits/cart: 3 control-team index + regress
bit + stall bit) makes every cart's controller and motion readable client-side without
per-cart networking. `PAYLOAD_GOALS_PACKED` still carries 6-bit progress per cart
(`1 + floor(progress·62)`, 0 = no such cart); cart 0 also has the fine
`PAYLOAD_PROGRESS`/`PAYLOAD_SPEED` stats.

The modicons panel is now a **k-cart dashboard**: one row per present cart, sized to the
panel. Each row is a dark track, a team-coloured progress fill, a head marker at the
cart position tinted by motion (team colour advancing, grey stalled, red regressing),
and an advance/regress glyph at the row end. A push-contest band across the top shows
each team's live occupancy weight (cart 0's, the only one `PAYLOAD_PUSH_PACKED`
carries). Team colour = controlling team throughout, so the panel and the map agree.

### Path-link entity budget (why carts are bounded)

The minimap ribbon and the world ribbon both read `ENT_CLIENT_RADARLINK` entities,
which are **always-sent** (`Net_LinkEntity(..., docull=false)`) — every one competes
with player entities for the client's per-snapshot entity budget. `payload_path_build`
runs for *every* `func_plc_cart`, and a fused mega-map carries one cart per fused tile,
so an unbounded "one link per node-pair" would emit hundreds of always-sent entities
and silently starve player CSQCModels out of the snapshot (players simulate and bank
server-side but never draw client-side, with no VM error — the links win the budget,
the players lose it). That was the player-render regression in 2d35b07.

The fix bounds the footprint: each cart's path is **subsampled to at most
`PLC_RIBBON_SEG` (4) straight segments**, and the whole map is capped at
`g_payload_ribbon_max` (32) link entities total; `g_payload_pathlinks 0` disables path
links entirely. Both are live cvars — no rebuild needed to retune on a bigger fused map.
The ribbon is coarser than one-link-per-node (straight hops between sampled nodes rather
than following every curve), which is the deliberate trade for keeping players on screen.

### Showing the minimap as a spectator

The radar/minimap panel is **off by default** in the shipped HUD (`hud_panel_radar 0`),
so an observer sees the world ribbons but not the minimap ribbons — this is panel
visibility, not a data failure (the `RADARLINK` entities do arrive; the world ribbon is
drawn from them). To show it: set `hud_panel_radar 2` (small radar, forced on even outside teamplay; `1`
also suffices since payload is teamplay). The maximized overview map is the `radar` /
`clickradar` client command (`HUD_Radar_Show_Maximized`), bindable to a key.

### What is client-side

Everything the viewer sees per frame — the dashboard and the world ribbon — is CSQC,
zero server cost. The only server work is spawning the link entities once and flipping a
`SendFlags` bit when a segment's colour actually changes.

## What only a live spectator run confirms

Compiled clean (server + client, `-Werror -Wall`); the following are untested because
this was a compile-only pass against a running match:

- **World ribbon rendering** — that `R_BeginPolygon(..., false)` ground quads land at the
  right height, read as a ribbon rather than z-fighting the floor, and that `+6u` is
  enough lift on sloped/curved track. Width and lift may need tuning.
- **Radar-link readability at 4 carts** — whether four overlapping coloured routes on the
  minimap stay legible or need per-cart offset/thinning; and that the palette-index
  colours match the `Team_ColorRGB` used elsewhere for every team (verified only for the
  built 5-team palette).
- **Active-segment colour tracking** — that the contested-front colour visibly moves with
  the cart and doesn't lag or flicker at segment boundaries.
- **Node fade distance** — that 1100u declutters without hiding checkpoints a spectator
  wants; checkpoints are unlimited-range by design but the plain-node number is a guess.
- **Dashboard layout** — row sizing at 1–4 carts inside the real modicons panel aspect,
  and that the motion tint / glyph read at panel scale.
- **Link lifecycle** — links persist across rounds and recolour via `payload_update_links`;
  a live run should confirm nothing double-spawns or leaks on `map_restart`.

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

## Procgen pipeline (tools/)

Two authoritative surfaces, staged from least to most authored:

- `tools/mapfuse.py <seed> [maps... | /path/prefix...]` — roguelike fusion of
  compiled BSPs. Places j sources (stock pk3 maps, or loose `.bsp/.waypoints/
  .waypoints.cache` triples such as mapgen output, addressed by path prefix) on a
  disjoint grid, merges every IBSP v46 lump with index fixups, and joins them with
  synthesized connectors: at most one cart-navigable corridor per map, jump-pad
  shafts and teleporter pairs for everything else. Emits `fused.{bsp,waypoints,
  waypoints.cache,mapinfo,ent,pk3}`. The pk3 is what clients must mount — a client
  without it renders the world as untextured void.
- `tools/mapgen.py <seed> [--rooms=N] [--smoke]` — parametric map SOURCE authoring.
  A small DSL (rooms with doorways, corridors, ledges, jump-pads, teleporters,
  team/dm spawns, lights) emits a `.map` and compiles it with q3map2
  (`-meta`, `-vis`, `-light -fast -samples 2 -bounce 2`) into a textured, lit,
  vis'd BSP; generates grid waypoints+links over the authored floors, runs the
  corridor-gated cart placer for the payload `.ent`, and packages a pk3.
  Compiled arenas can be fed back into mapfuse as tiles by path prefix.

q3map2 comes from netradiant-custom (github.com/Garux/netradiant-custom) built on
macOS arm64 with only its q3map2 target:

    brew install assimp glib libxml2 libpng jpeg-turbo
    make OS=Darwin MACLIBDIR=/opt/homebrew/lib DEPENDENCIES_CHECK=off binaries-q3map2

The threaded vis/light stages SIGBUS on arm64; mapgen pins `-threads 1` for both.
Override paths with `Q3MAP2` and `XON_BASEPATH`. Some stock shader images are
dds-only so q3map2 logs "Couldn't find image" — harmless for compile and runtime.

## Join quality and navigability (mapfuse + joinview)

Every fused join is built to be bot-traversable and classified for prominence:

- **Bot transport**: corridor joins carry a chain of walkable waypoint links in
  the fused `.cache`. Jump-pad and teleporter joins do NOT get synthetic walk
  links — Xonotic autogenerates their bot waypoints from the entities at map init
  (`trigger_push` tracetosses to the real landing then `waypoint_spawnforteleporter`,
  jumppads.qc:551; `trigger_teleport` likewise, teleporters.qc:253), so mapfuse
  emits canonical `trigger_push`+`target_position` / `trigger_teleport`+
  `misc_teleporter_dest` and models the resulting one-way jump in `fused.joins.json`.
  A region flood-fill over (cache walk-links + modeled jumps) asserts all source
  maps land in one bot-reachable component and reports per-join traversability.
- **Prominence rule**: each map-node is classified by edge count. An edge whose
  endpoint is a degree-1 leaf (the sole lifeline to that map/objective) is
  EXCLUSIVE and gets the prominent template — wide mouth, light entities at both
  ends, kept short, corridor (cart-navigable) when geometry and the one-corridor-
  per-map budget allow, else a lit teleporter/pad. Redundant edges (both endpoints
  degree ≥2) may be subtle. The generated classification is printed per topology.
- **Clip carving**: corridor selection now carves stock-map PLAYERCLIP/BOTCLIP/
  MONSTERCLIP brushes (0x430000) that cross the tube, not just solid brushes — a
  clip brush leaves the floor walkable but physically blocks players, which
  silently broke crossings.
- **Sizing**: source maps pack tighter (MARGIN 896, was 2048) and the corridor
  length cap dropped (6000, was 14000); over-long joins become short jump-pads.

`tools/joinview.py <dir>` diagnoses a fused map's joins offline: it writes
`fused.floorplan.svg` (dependency-free top-down plan — map footprints, nav graph,
joins colored by type, prominent=thick/solid vs subtle=dashed, lights) and reports
per edge a **contortion** score (fuzzed walk-distance / straight-line through the
join), a **visual-occlusion** ray count (eye rays from both sides that hit solid
before the opening), and whether the join is **clip-blocked**. Headless engine
screenshots are not available (the dedicated server has no GL), so the egocentric
check is the raycast probe rather than a rendered view.
