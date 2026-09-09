# Xonotic payload mode (`plc`)

A k-team, j-cart checkpoint-control gamemode. Each cart is a `MOVETYPE_PUSH` brush driven along
its own waypoint path by contested occupancy. Up to 256 carts coexist;
every cart is pushed forward by the team that controls it and backward by everyone
else. Team score increases continuously at a rate proportional to held checkpoints.
Teams and lanes are independently configurable; arriving at a path endpoint never
ends the game. The literal algebra and policy contract are in
[CART-GAME-CONTRACT.md](../../design/CART-GAME-CONTRACT.md).

The authoritative game source is `../qcsrc/`. Payload mode and its registry
entries are implemented directly there. `cfg/gamemodes-payload.cfg` is runtime
configuration.

## Control and the speed law

Every `PLC_TICK` (0.1 s) each cart recomputes, for each team index `j`:

- `n_j` — live players of team `j` inside `cart.radius` horizontally and
  `cart.height` vertically.
- `w_j = Σ_{i=1..min(n_j, push_cap)} push_falloff^(i-1)`.
  `falloff = 1` is Xonotic-native capped-linear; `0.5` is TF2 diminishing returns.

**Control**: a neutral cart is claimed by the team with the strict maximum `w_j`.
A claimed cart retains its controller when empty or contested. Opponents must push
it back to the origin to neutralize it before another team can claim it.

**Direction**: every cart path has one origin (`s = 0`, the first `plc_path` node) and
one end (`s = L`). The cart's sticky controller is established at the origin. With
`c` its controller, `w_c` its occupancy weight, and `w_opp = Σ_{j != c} w_j`:

```
w_c > 0:       v = clamp(cart.speed · (w_c − w_opp) / (1 + w_opp²),
                         −g_payload_contest_speed, g_payload_max_speed)
w_c = 0, w_opp > 0:
               v = clamp(−g_payload_reverse_speed · max_j(w_j),
                         −g_payload_max_speed, 0)
empty:         v = 0
```

The local contest regime keeps a defended cart near the fight. Once its controller
leaves, the strongest opposing team walks it backward without the contest damping.
With nobody present, position, controller and captured checkpoints stay fixed.
Held checkpoints continue accumulating score. Stopping partway through a return
also leaves the cart at that exact position until players intervene again.

Per-team `plc_team` entities declare participating teams independently of cart lanes.
There are no delivery-goal entities.

Idle time remains an occupancy measurement. The two former rollback wire columns
are reserved zeros so retained telemetry and clients keep the same row layout;
there is no rollback state or idle-motion configuration.

Cart motion is direct velocity (`velocity = (pos(s') − pos(s)) / PLC_TICK`), not
`SUB_CalcMove`, because `SUB_CalcMove` commits to a destination and a traveltime at
issue time and cannot change speed mid-segment.

## Scoring

### Held-checkpoint score integration

Only `plc_path` nodes flagged `PLC_CHECKPOINT` are scoring checkpoints; additional
vertices shape the path. Advancing across a checkpoint captures it for the sticky
controller. Retreating behind it removes that control, not previously earned score.
The origin is not a scoring checkpoint. The constructor places four checkpoints per
lane by default, preserving the underlying path geometry.

For team `i`, `H_i` is its total held checkpoints across all lanes. The server maintains
`dS_i/dt = g_payload_checkpoint_rate * H_i`. It integrates the old held state before
changing ownership and clips the last integration interval to the earliest threshold
crossing. Checkpoint changes log `payload: checkpoint cart … node … owner …`.

Victory occurs at `g_payload_point_limit`, initially 1,200. At the default rate of
one point per checkpoint per second, holding all eight checkpoints on two lanes
requires 150 seconds of accrual. Acquisition and contention add time. There is no
delivery condition, point subtraction, cap-count terminal condition, or timeout winner.
A simultaneous threshold crossing is a draw. Only a new episode clears earned scores.

- Team: `ST_PAYLOAD_POINTS` ("points"), primary, integer display of the score ledger.
- Player: `SP_PAYLOAD_PUSH` — `(s' − s) · g_payload_score_rate` split over the
  controlling team's in-radius players while their cart advances.
- Player: `SP_PAYLOAD_BLOCK` — accrued while a cart is occupied and stalled.

The solver receives the full float ledger, held-checkpoint count, threshold, rate,
episode identity and terminal flag as a synchronized team table. Scoreboard rounding
does not enter the projected winner or the W/L policy rewards.

## Map entity format

### `func_plc_cart` (brush, 1 to 256)

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
control law needs an unambiguous origin. A path coordinate is the cart's floor-contact
coordinate. The offline constructor derives every path from stock navigation and the
continuous negative-space representation, then subtracts `PLC_CART_RIDE_OFS` when it
emits the entity coordinate. Runtime applies the inverse view offset and does not derive
another path from generic waypoints.

### `plc_path` (point, ≥ 2 per cart)

`targetname`, `target` (next node; omit on the last), `curvetarget` (quadratic bezier
control point, same key `path_corner` uses), `spawnflags 1` = `PLC_CHECKPOINT`.
Chains must be disjoint between carts; a `target` pointing at the first or an
already-visited node terminates the chain.

### `plc_team` (point, optional)

Mirrors `dom_team`: `netname`, `cnt`. If absent, teams are spawned from
`g_payload_default_teams` / `g_payload_teams_override`. `cnt` is the team colour index,
using the `dom_team` convention. Team declarations do not assign carts or destinations.

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

- **controlled checkpoint segment** → the owning team's colour, so captured track visibly advances
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

All nodes still exist as sprites (checkpoint control and the mesh objective marker both hang off
`node.sprite`), but the **rule is set once at spawn, no per-tick churn**:

- **checkpoint nodes** (`PLC_CHECKPOINT`) → unlimited range: the meaningful control
  points are always visible on the skyline;
- **plain shape/curve nodes** → `maxdistance = g_payload_node_fade_dist` (1100u): they
  fade out at range so a distant path is a clean ribbon, not a picket fence, but resolve
  into individual markers when you are actually working that stretch.

On top of the fixed rule, the existing dynamic colouring stands: a held checkpoint wears
its controller's colour, the mesh-chosen objective node per team gets
`RADARICON_OBJECTIVE` in that team's colour, everything else stays neutral cyan. The
"which point is contested right now" answer is carried by the ribbon's active-segment
colour rather than by re-colouring a node every tick.

### The artifact, the cart body, and ink as territory

The mode's fourth channel is the world surface itself. It has three pieces and one
shared definition.

**The artifact.** A knot curve in R^5 — a (2,3) torus knot in x/y/z plus two further
harmonics in w and v — is rotated every frame by a product of five SO(5) Givens
rotations whose rates are mutually incommensurate, then projected to R^3 by two
successive perspective divides (5D→4D on v, 4D→3D on w) and swept as a tube. Because
the rotation folds the two hidden axes into the visible three, the body turns itself
inside out continuously while its 5D shape never changes. Cost is `ART_SEG` curve
samples × 5 sin/cos + 5 Givens (20 multiply-adds) + 2 divides per frame; the polygon
emission dominates the mathematics by two orders of magnitude, so `cl_artifact_segments`
and `cl_artifact_sides` are the knobs, not the geometry.

It drifts on a Lissajous through the level's bounding volume, clipped by one traceline
per frame from the level centre so it slides along the inside of the hull instead of
orbiting through the void, and drops ink globs on a cadence keyed to match time — so
every client places the same glob at the same spot on the same frame with no network
traffic, and a client joining mid-match replays what it missed (`cl_artifact_catchup`).

**The cart is an instance of the same body.** `mkentfile.py` gives a cart whichever
inline brush the source map left visible, so "the cart" is some door from `warfare`
with no silhouette, no size and no team colour. That brush is now hidden
(`g_payload_cart_procedural 1`; it stays solid, it is still what pushes and crushes)
and the cart is drawn as the same knot at a cvar-fixed size — one silhouette and one
set of bounds for every cart on every fused tile — tinted by its control state and
emissive by it: grey uncontrolled, the team colour under a plurality, and streaked
between the two strongest claimants along a helical stripe when its cylinder is
contested. The stripe is spatial rather than a time-alternation so a still frame reads
as contested rather than as a coin flip.

**Ink is cart territory.** Every `PLC_TICK` an advancing cart lays its controller's
colour on the ground it has just covered; a contested cart lays the muddied blend of
the two claimants; a cart being driven back has its own paint overpainted by the team
pushing it home. That writes depth, control, contest and reversal onto the world
surface with no HUD overlay.

**How the drift and the cart ink compose rather than compete.** They write the same
voxels, so they mean different things in them. The volume already separates the two:
coverage (alpha) drives how wet and rubbery a surface is, colour (rgb) drives what it
is tinted. The carts own colour — a narrow, decisive, high-amplitude write along a
track. The drift owns coverage — a wide, low-amplitude wash whose *colour is the mean
colour the world already carries* (`ink_stat(INK_TINT_*)`, the same cached pair the
engine tints the sky with), pulled toward the sour green only as far as the world is
still unpainted. So the drift wets everything and repaints nothing.

**The skybox counterpart.** The fused world is ~152,281 units across; nobody can see
the far theatre. Each cart therefore also hangs supermassive against the sky in the
direction it actually lies, in its own elevation band, wearing its control colour and
its contest streak, with a bright band travelling along the tube at its arclength
fraction. Which cart, where, whose colour and how deep — at sky scale, without a HUD.

Frames and frame times for all of this are captured by `../render/plc-run.sh`;
`../render/PLC-CAPTURE.md` documents the harness, the measured cost and the shipped
`../render/shots/plc-dance-*.png` pair.

### What the client reads, and what it does not re-derive

Everything above is decoded from state the server already networks:

| datum | source |
|---|---|
| control team, regress, stall | `STAT(PAYLOAD_CARTS_STATE)`, 5 bits per cart |
| presence, coarse arclength | `STAT(PAYLOAD_GOALS_PACKED)`, 6 bits per cart |
| cart 0 arclength, full precision | `STAT(PAYLOAD_PROGRESS)` |
| per-cart per-team occupancy | `STAT(PAYLOAD_PUSH_PACKED{,1,2,3})` |
| **cart world origin** | the cart's own waypointsprite, which is spawned with `ref` = the cart and therefore re-sends the cart's origin every time it moves |
| **which cart a sprite is** | the sprite's spare `wp_extra` byte |

The last two are the only additions, and neither is a new entity or a new packing: the
sprite already ships. An earlier revision instead walked `g_radarlinks` with the
arclength fraction, which is wrong twice over — that list holds every cart's links
undifferentiated, and those links are subsampled to at most `PLC_RIBBON_SEG` straight
hops, so the reconstructed point left the track on every curve. That re-derivation is
deleted.

`payload_cart_read()` in `payload.qc` is the single definition of "what is cart c
doing". The modicons row, the world body, the sky body and the ink all read it and
nowhere else; `payload_link_color()` is likewise the single decode of a path-link
colour byte. A second unpacking of the cart state, a second team-colour lookup or a
second control-state palette anywhere in the client is a defect on sight
(`design/CAST.md`).

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
./payload/build.sh
```

`gmqcc` must be built first and passed explicitly: the Makefile default
`QCC ?= ../../../../gmqcc/gmqcc` does not resolve in the shipped layout, and
`QCCFLAGS_WATERMARK ?= $(shell git describe ...)` fails outside a git repo. `gmqcc`'s
own Makefile passes `-Wl,--gc-sections`, which Apple `ld` rejects; relink with
`c++ .build/objs/*.o -o gmqcc`.

## Running it without a payload map

Map parsing, the recovered network/span algorithm, its history, data structures and
validation are documented in [CARTPATHS.md](CARTPATHS.md). `tools/navmesh.py` owns
navigation and lane placement; `tools/mkentfile.py` emits the checkpoint entities.

`tools/mkentfile.py <bsp> <out.ent> [teams] [carts]` reads a stock BSP's entity lump
and appends the requested number of negative-space-constrained `plc_path` tracks. It
replaces team-labeled spawns with a shared spawn set, configures every cart through the
same procedural pusher body, and emits one team declaration per team. The adjacent
measurement artifact reports nondegenerate-path, stock-navigation spawn reachability,
cart-motion clearance and nearby-ground coverage, their cart-advanceability conjunction,
separate rider-volume clearance, and spawn/cart
clearance and origin-occupancy masses. Drop the result in
`<userdir>/data/maps/<name>.ent` with a `<name>.mapinfo` carrying `gametype plc`.
`+sv_autopause 0` is required or an empty dedicated server freezes `sv.time` after
~5 s.

`+bot_join_empty 1` is required or no bot ever joins. `bot_fixcount`
(`server/bot/default/bot.qc:640`) only computes a bot target when
`realplayers || autocvar_bot_join_empty || (currentbots > 0 && time < 5)`; on a
headless dedicated server with no human client all three are false.
