# Xonotic payload mode (`plc`)

A k-team payload/cart gamemode. The cart is a `MOVETYPE_PUSH` brush driven along a
waypoint path by contested occupancy, generalised past the two-team attack/defend shape.

Everything under `qcsrc/` drops into a Xonotic `qcsrc` tree at the same relative path.
`patch/0001-payload-registry-hooks.patch` carries the five one-line-per-file registry
edits to shared files. `cfg/gamemodes-payload.cfg` is the cvar overlay.

## The speed law

Every `PLC_TICK` (0.1 s) the cart recomputes, for each team index `j`:

- `n_j` — live players of team `j` inside `cart.radius` horizontally, inside
  `cart.height` vertically, with line of sight to the cart centre.
- `w_j = Σ_{i=1..min(n_j, push_cap)} push_falloff^(i-1)`.
  `falloff = 1` is Xonotic-native capped-linear; `0.5` is TF2 diminishing returns.
- `d_j = sign(s*_j − s)` where `s*_j` is the arclength of team `j`'s `plc_goal`.

Then

```
P+ = Σ_{d_j > 0} w_j      P− = Σ_{d_j < 0} w_j
v  = clamp(cart.speed · (P+ − P−), ±g_payload_max_speed)
s' = clamp(s + v · PLC_TICK, 0, L)
```

`k` enters only through the two sums. Two teams reproduce Onslaught's
`friendly_count − enemy_count` exactly. Coalitions are emergent, never declared: teams
whose goals lie on the same side of the cart add into the same sum, and stop doing so the
instant the cart passes one of their goals. A tie gives `v = 0` with players present —
the stall the mesh demo reads off.

Rollback: after `g_payload_idle_time` seconds with `Σ w_j == 0`, the cart moves at
`g_payload_rollback_speed` toward the cart's start arclength, stopping at the first
`PLC_CHECKPOINT` node it meets.

Cart motion is direct velocity (`velocity = (pos(s') − pos(s)) / PLC_TICK`), not
`SUB_CalcMove`, because `SUB_CalcMove` commits to a destination and a traveltime at issue
time and cannot change speed mid-segment. This is the same branch `SUB_CalcMove` itself
takes when `traveltime < 0.15`.

## Scoring and round end

- Team: `ST_PAYLOAD_CAPS` ("caps"), primary, +1 per round won.
- Player: `SP_PAYLOAD_PUSH` — partial credit `(|s_prev − s*_j| − |s_now − s*_j|) ·
  g_payload_score_rate`, split over that team's in-radius players via
  `GameRules_scoring_add_float2int`, so coalition pushing pays.
- Player: `SP_PAYLOAD_BLOCK` — accrued while the cart is occupied and stalled.

Progress deliberately does **not** feed a *team* score field. An earlier revision had
`field_team(ST_PAYLOAD_PUSH, "push", 0)` fed every tick; the match then ended after one or
two rounds even with `fraglimit 10`, and setting `g_payload_score_rate 0` made it stop.
Measured, both directions, on a live dedicated server. Team progress score is player-only
for that reason.

Round end: `|s − s*_j| ≤ plc_goal.radius` gives team `j` the round. On round timeout the
team whose goal is nearest wins; an exact tie gives `CENTER_ROUND_TIED`. The mode reuses
the existing `ROUND_TEAM_WIN` / `ROUND_TIED` notification families, so
`common/notifications/` needs no edit (which also avoids the `NOTIF_CHOICE_MAX` trap).

## Map entity format

### `func_plc_cart` (brush, exactly one)

| key | meaning |
|---|---|
| `target` | targetname of the first `plc_path` node |
| `plc_start` | targetname of the node the cart starts at; default is the node nearest the brush |
| `speed` | track units/second per unit of net push weight (default `g_payload_speed`) |
| `radius` | horizontal push radius (default `g_payload_push_radius`) |
| `height` | vertical push half-band (default `g_payload_push_height`) |
| `dmg`, `dmgtime` | crush damage via `generic_plat_blocked` |
| `spawnflags` | `1` = `PLC_CART_TURN`, face along the track |

### `plc_path` (point, ≥ 2)

`targetname`, `target` (next node; omit on the last), `curvetarget` (quadratic bezier
control point, same key `path_corner` uses), `spawnflags 1` = `PLC_CHECKPOINT`.

A `target` pointing back at the first node, or at an already-visited node, terminates the
chain rather than looping. Cyclic tracks are not supported; the chain walk is bounded at
4096 nodes.

### `plc_goal` (point, one per participating team)

`cnt` = team colour index (`4` red, `13` blue, `12` yellow, `9` pink — same convention as
`dom_team`; the mode adds 1 to get the server team id), `target` = a `plc_path`
targetname, `radius` = capture tolerance in arclength units (default 64).

**The set of `plc_goal` teams drives the team count.** `payload_DelayedInit` ORs
`Team_TeamToBit(goal.team)` into `payload_teams` before consulting `plc_team` entities or
`g_payload_default_teams`, so a third goal alone is enough to make it a three-team map.

### `plc_team` (point, optional)

Mirrors `dom_team`: `netname`, `cnt`. If absent, teams are spawned from
`g_payload_default_teams` / `g_payload_teams_override` unioned with the goal teams.

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
`QCCFLAGS_WATERMARK ?= $(shell git describe ...)` fails outside a git repo. `gmqcc`'s own
Makefile passes `-Wl,--gc-sections`, which Apple `ld` rejects; relink with
`c++ .build/objs/*.o -o gmqcc`.

`tools/mkpatch.sh` regenerates the registry patch. `check-units.sh` compiles each payload
`.qc` as its own translation unit (upstream's `tools/compilationunits.sh` only walks
`client/`, `server/`, `menu/` and never touches `common/`).

## Running it without a payload map

`tools/mkentfile.py <bsp> <out.ent> [3]` reads a stock BSP's entity lump, appends a
5-node `plc_path` track through its team spawns, a `func_plc_cart` on an inline brush
model, and two (or three) `plc_goal`s. Drop the result in
`<userdir>/data/maps/<name>.ent` with a `<name>.mapinfo` carrying `gametype plc`.
`+sv_autopause 0` is required or an empty dedicated server freezes `sv.time` after ~5 s.

`tools/checklaw.py <log> <start_time>` re-derives the speed law from a `plcdbg` log
(`debug/plcdbg.patch` adds the instrumentation) and reports mismatches.
