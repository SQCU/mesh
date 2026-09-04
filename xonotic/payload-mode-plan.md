# Xonotic payload mode: implementation plan

Scope of this document: what to write, where, which existing functions to call, and what
a payload map must contain. No code is committed by this plan. Nothing here opens an
RDMA device; the mesh seam is named at the end and is a later change.

Source read: `/Applications/Xonotic/source/qcsrc` (xonotic-20230620 tree, not a git repo).

## 1. What the engine already gives us, and what it does not

### func_train follows a path, but cannot be driven

`common/mapobjects/func/train.qc` walks `path_corner` entities. `func_train_find`
(INITPRIO_FINDTARGET) seeds `.target` / `.future_target` and places the mover; each
arrival calls `train_wait` -> `train_next`, which issues
`SUB_CalcMove(this, targ.origin - this.view_ofs, TSPEED_LINEAR, speed, train_wait)`.

Three properties make it unusable as a payload cart as written:

- `SUB_CalcMove` (`common/mapobjects/subs.qc:265`) commits to a destination and a
  traveltime at issue time. Speed cannot change mid-segment.
- Re-issuing it every tick to change speed is not free: `SUB_CalcMove_Bezier` does
  `delete(this.move_controller)` and `new_pure(SUB_CalcMove_controller)` on every call.
- The path is singly linked (`.target` only). There is no way back; a payload rolls back.

What is reusable, and should be reused verbatim:

- `InitMovingBrushTrigger(this)` (`subs.qc`) - sets `SOLID_BSP`, `SetBrushEntityModel`,
  `set_movetype(this, MOVETYPE_PUSH)`. This is what makes players ride and get pushed.
- `this.view_ofs = this.mins` and `origin = corner.origin - view_ofs` (train's
  brush-origin convention).
- `setblocked(this, generic_plat_blocked)` (`common/mapobjects/platforms.qc:3`) plus
  `.dmg` / `.dmgtime` / `.message` / `.message2`, for crushing.
- `InitializeEntity(this, fn, INITPRIO_FINDTARGET)` to resolve the path after all
  entities exist.
- `spawnfunc(path_corner)` keys, including `curvetarget` for bezier control points.

The cart is therefore a MOVETYPE_PUSH brush driven by direct velocity: every tick set
`this.velocity = (target_point - current_point) * (1 / PL_TICK)` and
`this.nextthink = this.ltime + PL_TICK`. This is exactly the short-move branch
`SUB_CalcMove` itself takes when `traveltime < 0.15` (`subs.qc:296`), so it is a
supported way to drive a pusher, not a novelty.

The path is walked **once** at INITPRIO_FINDTARGET into a doubly-linked node list with
cumulative arclength. Cart state then collapses to one scalar `s` in `[0, L]`. Speed is
`ds/dt` and may be any sign. A path that loops back to its first node is accepted as
cyclic (`L` = loop length); no objerror, no refusal branch.

### How domination registers a mode

`domination/` is five files plus two generated ones, and the registration is spread over
four registries:

| what | where |
|---|---|
| gametype object, menu entry, mapinfo detection | `domination.qh`: `CLASS(Domination, Gametype)` ... `REGISTER_GAMETYPE(DOMINATION, NEW(Domination))` |
| server activation, cvar limits | `sv_domination.qh`: `REGISTER_MUTATOR(dom, false)` with `MUTATOR_STATIC()` and `MUTATOR_ONADD` calling `GameRules_teams(true)`, `GameRules_limit_score()`, `GameRules_limit_lead()`, `dom_Initialize()` |
| build inclusion | `_mod.inc` / `_mod.qh` in the mode dir, and the parent `gamemode/_mod.inc` / `_mod.qh` |
| scoreboard fields | `common/scores.qh` `REGISTER_SP(...)`, plus `ScoreRules_dom()` calling `GameRules_scoring(teams, spprio, stprio, {...})` |
| HUD icon | `ATTRIB(Domination, m_modicons, ..., HUD_Mod_Dom)` - the modicons panel dispatches through the gametype, so `client/hud/panel/modicons.qc` needs no edit |
| cvars | `gamemodes-server.cfg` inside `xonotic-*-data.pk3` (lines 382-397 for dom) |

`gametype_init` (`common/mapinfo.qh:104`) assigns the mode's mapinfo bit as
`MAPINFO_TYPE_ALL |= this.m_flags = (MAPINFO_TYPE_ALL + 1)` - one bit per registered
gametype, in registration order. 21 modes exist today, so payload takes bit 21; the
ceiling is the 24-bit QC float mantissa.

Map auto-detection: `_MapInfo_Generate` (`common/mapinfo.qc:265`) scans the `.ent`/`.bsp`
for `classname` keys and calls `it.m_generate_mapinfo(it, v)` for every gametype
(`mapinfo.qc:401`). Domination matches `"dom_controlpoint"`. Payload matches `"plc_cart"`.

### How contested occupancy is computed elsewhere

Two precedents, and they differ in exactly the way that matters:

- **Domination is not occupancy at all.** `dompointtouch` (`sv_domination.qc`) flips the
  point on *touch*, guarded by `time < this.captime + 0.3`. Last toucher wins. There is
  no notion of two teams present at once.
- **Onslaught proximity decap is occupancy**, and is the model to copy
  (`onslaught/sv_onslaught.qc:474-495`):

```
FOREACH_CLIENT(IS_PLAYER(it) && !IS_DEAD(it), {
    if(vdist(it.origin - this.origin, <, autocvar_g_onslaught_cp_proxydecap_distance))
    {
        if(SAME_TEAM(it, this)) ++_friendly_count; else ++_enemy_count;
    }
});
... GiveResourceWithLimit(this, RES_HEALTH, (_friendly_count - _enemy_count), this.max_health);
```

A per-tick `FOREACH_CLIENT` + `vdist` count, and a **difference** of counts driving a
rate. That difference is the two-team special case of the k-team law below. Note the
binary `SAME_TEAM` bucket: generalising means bucketing by `Team_TeamToIndex(it.team)`
instead.

### How scoring and round end work

- Continuous: `TeamScore_AddToTeam(int t, float scorefield, float score)`
  (`server/scores.qh:64`) plus `GameRules_scoring_add(client, FIELD, value)`. For
  fractional per-tick credit use `GameRules_scoring_add_float2int(client, FIELD, value,
  float_field, factor)`, which exists for exactly this.
- Round based: `round_handler_Spawn(canRoundStart, canRoundEnd, roundStart)` and
  `round_handler_Init(delay, warmup, timelimit)` (`server/round_handler.qh`). The
  `canRoundEnd` callback (`Domination_CheckWinner`) is where a mode: checks
  `round_handler_GetEndTime() - time <= 0` for timeout, decides a winner, sends
  `CENTER_ROUND_TEAM_WIN` / `INFO_ROUND_TEAM_WIN` via `APP_TEAM_NUM(team, ...)`, sets
  `game_stopped = true`, and re-inits the round handler.
- Winner-by-owned-items (`Team_GetWinnerTeam_WithOwnedItems`, with
  `Team_SetNumberOfOwnedItems` / `Team_GetNumberOfOwnedItems`) is a control-point
  abstraction. Payload has one contested object, so it does not use it; it decides the
  winner from the cart's arclength directly.

## 2. The speed law, generalised to k teams

Recomputed every `PL_TICK` (0.1 s, matching `ONS_CP_THINKRATE` practice).

**Occupancy.** For each team index `j` in `1..NUM_TEAMS`:

```
n_j = |{ p : IS_PLAYER(p) && !IS_DEAD(p) && Team_TeamToIndex(p.team) == j
             && vdist(p.origin - cart_center, <, cart.radius)
             && fabs(p.origin.z - cart_center.z) < cart.height
             && traceline LOS p -> cart_center }|
```

**Weight.** `w_j = sum_{i=1..min(n_j, g_payload_push_cap)} g_payload_push_falloff^(i-1)`.
Falloff 1.0 gives linear-capped (Xonotic-native); 0.5 gives TF2's diminishing returns.
One expression, no branch.

**Direction.** The track is one-dimensional. A neutral cart is claimed at the origin by
the unique plurality. That controller remains the cart color until regression reaches
the origin.

**Net push and velocity.**

```
w_c > 0:
    v = clamp(g_payload_speed * (w_c - sum_{j != c} w_j)
              / (1 + (sum_{j != c} w_j)^2),
              -g_payload_contest_speed, g_payload_max_speed)
w_c == 0 and max_{j != c}(w_j) > 0:
    v = clamp(-g_payload_reverse_speed * max_{j != c}(w_j),
              -g_payload_max_speed, 0)
```

Why this is the right generalisation, and not k copies of a two-team rule:

- k enters only through invariant reductions over the live team rows.
- Every team can claim every neutral cart and deliver it at the common path end.
- Ties stall visibly while the controller remains present.
- When the controller leaves, the strongest present opponent reverses the cart.

**Rollback.** If all occupancy weights remain zero for `g_payload_idle_time`, the cart
moves at `g_payload_rollback_speed` toward the nearest preceding `PLC_CHECKPOINT` and
stops there. `ROLLBACK_ACTIVE` and `ROLLBACK_TARGET` are authoritative cart-row
coordinates.

**Scoring, per tick, per team** - partial credit, so coalition pushing is rewarded:

```
delta_j = (|s_prev - s*_j| - |s_now - s*_j|) * g_payload_score_rate
if (delta_j > 0) TeamScore_AddToTeam(Team_IndexToTeam(j), ST_PAYLOAD_PUSH, delta_j)
```

and the same amount split over that team's players inside the radius through
`GameRules_scoring_add_float2int(p, PAYLOAD_PUSH, delta_j / n_j, ...)`. A team with
`w_j > 0` on the losing side while `|v| < stall_eps` accrues `PAYLOAD_BLOCK` instead.

**Round end.** `|s - s*_j| <= plc_goal.radius` -> team j takes the round:
`TeamScore_AddToTeam(t, ST_PAYLOAD_CAPS, 1)`, `Send_Notification(... APP_TEAM_NUM(t,
CENTER_ROUND_TEAM_WIN))`, `game_stopped = true`, `round_handler_Init(5, warmup,
round_timelimit)`. On timeout the team with the smallest `|s - s*_j|` takes the round;
exact ties give no cap and the round simply restarts.

## 3. File-by-file plan

New directory `qcsrc/common/gamemodes/gamemode/payload/`.

### payload.qh (new, all builds)

`CLASS(Payload, Gametype)`; `INIT` calls
`this.gametype_init(this, _("Payload"), "plc", "g_payload", GAMETYPE_FLAG_TEAMPLAY |
GAMETYPE_FLAG_USEPOINTS, "", "timelimit=20 pointlimit=200 teams=2 leadlimit=0", _("Push
the cart to your goal"))`.
`METHOD(m_parse_mapinfo)` handles `teams` -> `cvar_set("g_payload_default_teams", v)`.
`METHOD(m_generate_mapinfo)` sets `MapInfo_Map_supportedGametypes |= this.m_flags` when
`v == "plc_cart"`.
`METHOD(m_configuremenu)` exposes point limit.
`ATTRIB(Payload, m_modicons, ..., HUD_Mod_Payload)` under CSQC.
`REGISTER_GAMETYPE(PAYLOAD, NEW(Payload));`

### payload.qc (new, all builds)

Progress-string helper shared by HUD and server notifications. Exists also because
`tools/genmod.sh` pairs `cl_`/`sv_` files against a base name.

### sv_payload.qh (new)

Autocvar declarations; `IntrusiveList g_payload_nodes;`; `const float ST_PAYLOAD_PUSH = 1;
const float ST_PAYLOAD_CAPS = 2;`; entity field decls (`.entity plc_next, plc_prev;`
`.float plc_dist;` `.float plc_s;` `.vector plc_home;`);
`REGISTER_MUTATOR(plc, false)` with `MUTATOR_STATIC()` and `MUTATOR_ONADD` calling
`GameRules_teams(true)`, `GameRules_limit_score(point_limit)`,
`GameRules_limit_lead(...)`, `payload_Initialize()`.

### sv_payload.qc (new) - the whole mode

| function | what it calls |
|---|---|
| `payload_Initialize()` | `g_payload_nodes = IL_NEW()`; `InitializeEntity(NULL, payload_DelayedInit, INITPRIO_GAMETYPE)` |
| `payload_DelayedInit(entity)` | spawn default `plc_team` entities if the map has none (mirror of `dom_spawnteams`); `TeamBalance_CheckAllowedTeams` / `TeamBalance_GetAllowedTeams` / `TeamBalance_Destroy`; `ScoreRules_payload(teams)`; `round_handler_Spawn(Payload_CheckPlayers, Payload_CheckWinner, Payload_RoundStart)`; `round_handler_Init(5, warmup, round_timelimit)` |
| `spawnfunc(func_plc_cart)` | `InitMovingBrushTrigger(this)`; `this.effects |= EF_LOWPRECISION`; `this.view_ofs = this.mins`; `setblocked(this, generic_plat_blocked)`; defaults for `radius`/`height`/`speed`/`dmg`/`dmgtime`; `InitializeEntity(this, payload_path_build, INITPRIO_FINDTARGET)`; `this.reset = payload_cart_reset` |
| `payload_path_build(entity cart)` | repeated `find(NULL, targetname, ...)` from `cart.target`, filling `.plc_next` / `.plc_prev` / cumulative `.plc_dist`, `IL_PUSH(g_payload_nodes, node)`; sets `L`; `setorigin(cart, payload_pos(0) - cart.view_ofs)`; `WaypointSprite_SpawnFixed(WP_PayloadCart, cart.origin + '0 0 64', cart, sprite, RADARICON_OBJECTIVE)`; `WaypointSprite_UpdateMaxHealth(cart.sprite, L)`; resolves each `plc_goal` to `s*_j`; `setthink(cart, payload_think)`; `cart.nextthink = time` |
| `payload_pos(float s)` | segment lookup by `.plc_dist`, linear lerp, quadratic bezier when the node has `curvetarget` (same key `func_train` uses) |
| `payload_occupancy(entity cart)` | `FOREACH_CLIENT(IS_PLAYER(it) && !IS_DEAD(it), ...)` with `vdist`, z-band, `traceline`/`trace_fraction`; buckets by `Team_TeamToIndex(it.team)` into `payload_w[]` and `payload_n[]` |
| `payload_think(entity cart)` | `cart.nextthink = cart.ltime + PL_TICK`; early-out only on `game_stopped`/`time < game_starttime`; occupancy -> speed law -> `cart.plc_s` integration -> `cart.velocity = (payload_pos(s') - (cart.origin + cart.view_ofs)) * (1 / PL_TICK)`; scoring; `WaypointSprite_UpdateHealth(cart.sprite, cart.plc_s)`; `SUB_UseTargets(cart, cart, NULL)` when a checkpoint is newly passed; rolling `_sound(cart, CH_TRIGGER_SINGLE, cart.noise, VOL_BASE, ATTEN_NORM)` / `stopsoundto` on stall; `FOREACH_CLIENT(IS_REAL_CLIENT(it), set_payload_state(it))` |
| `set_payload_state(entity e)` | `STAT(PAYLOAD_PROGRESS, e)`, `STAT(PAYLOAD_SPEED, e)`, `STAT(PAYLOAD_PUSH_PACKED, e)` (per-team weights packed one byte each) |
| `payload_cart_reset(entity cart)` | `cart.plc_s = cart.plc_home_s`; `setorigin`; `cart.velocity = '0 0 0'` |
| `Payload_CheckWinner()` | timeout via `round_handler_GetEndTime()`; goal test; `TeamScore_AddToTeam(t, ST_PAYLOAD_CAPS, 1)`; `Send_Notification(NOTIF_ALL, NULL, MSG_CENTER, APP_TEAM_NUM(t, CENTER_ROUND_TEAM_WIN))`; `game_stopped = true`; `round_handler_Init(...)` |
| `Payload_CheckPlayers()` / `Payload_RoundStart()` | mirror of the domination pair; `player_blocked = false`; `payload_cart_reset` |
| `ScoreRules_payload(int teams)` | `GameRules_scoring(teams, sp, st, { field_team(ST_PAYLOAD_CAPS, "caps", SFL_SORT_PRIO_PRIMARY); field_team(ST_PAYLOAD_PUSH, "push", 0); field(SP_PAYLOAD_PUSH, "push", 0); field(SP_PAYLOAD_BLOCK, "block", 0); })` |
| `havocbot_goalrating_payload(...)` | `navigation_routerating(this, cart, ratingscale, 5000)`; rate the point `payload_pos(s +/- lookahead)` on the bot's own goal side |
| `havocbot_role_payload(entity this)` | `navigation_goalrating_timeout`, `navigation_goalrating_start`, the above, `havocbot_goalrating_items`, `havocbot_goalrating_waypoints`, `navigation_goalrating_end`, `navigation_goalrating_timeout_set` |
| `MUTATOR_HOOKFUNCTION(plc, HavocBot_ChooseRole)` | assigns `havocbot_role_payload` |
| `MUTATOR_HOOKFUNCTION(plc, TeamBalance_CheckAllowedTeams)` | ORs `Team_TeamToBit(head.team)` over `plc_team` entities, as dom does |
| `MUTATOR_HOOKFUNCTION(plc, reset_map_players / PlayerSpawn / ClientConnect)` | `PutClientInServer`, `player_blocked`, `set_payload_state` |
| `spawnfunc(plc_path)`, `spawnfunc(plc_goal)`, `spawnfunc(plc_team)` | point entities; `plc_team` mirrors `spawnfunc(dom_team)` including the `this.team = this.cnt + 1` palette conversion |

### cl_payload.qh / cl_payload.qc (new)

`HUD_Mod_Payload(vector pos, vector mySize)` - horizontal track bar, one tick per team
goal in `Team_ColorRGB`, cart marker at `STAT(PAYLOAD_PROGRESS)`, per-team push weights
from the packed stat, bar desaturated while `STAT(PAYLOAD_SPEED) == 0`.
`HUD_Mod_Payload_Export(int fh)` writes `hud_panel_modicons_payload_layout`.

### Edits to existing files

| file | edit |
|---|---|
| `common/gamemodes/gamemode/_mod.inc` and `_mod.qh` | add the payload includes. **Hand-edit**: `tools/genmod.sh` needs `git hash-object` and GNU `stat -c`; `/Applications/Xonotic/source` is not a git repo and macOS `stat` rejects `-c` (both verified). |
| `common/scores.qh` | `REGISTER_SP(PAYLOAD_PUSH); REGISTER_SP(PAYLOAD_BLOCK); REGISTER_SP(PAYLOAD_CAPS);` |
| `common/stats.qh` | `REGISTER_STAT(PAYLOAD_PROGRESS, float)`, `(PAYLOAD_SPEED, float)`, `(PAYLOAD_PUSH_PACKED, int)`. One packed int, not one stat per team - the domination precedent (`DOM_PPS_RED/BLUE/YELLOW/PINK`) is what makes k a compile-time constant, and payload must not repeat it. |
| `common/mutators/mutator/waypoints/all.inc` | `REGISTER_WAYPOINT(PayloadCart, _("Cart"), "", '1 0.5 0', 1);` and `REGISTER_WAYPOINT(PayloadGoal, _("Goal"), "", '1 0.5 0', 1);` |
| `common/notifications/all.inc` | `INFO_PAYLOAD_CAPTURE`, `CENTER_PAYLOAD_STALL`, `INFO_PAYLOAD_CHECKPOINT`. Round win/loss reuse the existing `ROUND_TEAM_WIN` family. |
| cfg overlay (new file, shipped as a mod pk3dir, not an edit to the pk3) | `set g_payload 0`, `g_payload_default_teams 2`, `g_payload_teams_override 0`, `g_payload_speed 30`, `g_payload_max_speed 200`, `g_payload_push_radius 160`, `g_payload_push_height 96`, `g_payload_push_cap 3`, `g_payload_push_falloff 1`, `g_payload_idle_time 8`, `g_payload_rollback_speed 20`, `g_payload_score_rate 0.01`, `g_payload_point_limit -1`, `g_payload_point_leadlimit -1`, `g_payload_round_timelimit 300`, `g_payload_warmup 5` |

No edit to `client/hud/panel/modicons.qc` (dispatch is via the gametype ATTRIB) and no
edit to `server/bot/default/` (role assignment is via the mutator hook).

### Build

`gmqcc` is **source only** in this tree - `source/gmqcc/gmqcc` does not exist and must be
built first. Then, from a copy of `qcsrc`:
`make QCC=<abs path>/gmqcc/gmqcc QCCFLAGS_WATERMARK=payload sv`. The default
`QCC ?= ../../../../gmqcc/gmqcc` does not resolve in this layout, and
`QCCFLAGS_WATERMARK ?= $(shell git describe ...)` fails outside a git repo; both must be
passed explicitly.

## 4. Map entity format

A payload map is detected by containing `plc_cart` (see `m_generate_mapinfo`). All keys
below follow existing Xonotic conventions so a mapper needs no new tooling.

### func_plc_cart (brush entity, required, exactly one)

```
classname   func_plc_cart
model       *NN                brushwork; players ride it because it is MOVETYPE_PUSH
target      <targetname>       first plc_path node
speed       30                 units/sec per unit of net push weight
radius      160                push radius, horizontal
height      96                 push radius, vertical half-band
dmg         0                  crush damage (generic_plat_blocked)
dmgtime     0.25
noise       sound/plc/roll.ogg looping while moving
noise1      sound/plc/stall.ogg
spawnflags  BIT(0) PLC_CART_TURN   face along the path, as TRAIN_TURN does
```

### plc_path (point entity, one per waypoint, >= 2)

```
classname   plc_path
targetname  <name>
target      <next node targetname>    omit on the last node of an open track;
                                      point it at the first node for a loop
curvetarget <name of a control point>  optional bezier control, same as path_corner
spawnflags  BIT(0) PLC_CHECKPOINT      rollback floor
```

### plc_goal (point entity, one per participating team, >= 2)

```
classname   plc_goal
cnt         4 | 13 | 12 | 9 | ...   team colour index, same convention as dom_team
target      <plc_path targetname>   the node this goal sits at
radius      64                      capture tolerance in arclength units
message     " has delivered the payload"
```

Two goals at opposite ends is TF2. Three or more goals distributed along the track is the
k-team game: every team's `s*_j` splits the remaining teams into a for-coalition and an
against-coalition, and the split changes as the cart moves.

### plc_team (point entity, optional, mirrors dom_team)

`netname`, `cnt`, `model`, `skin`, `noise`, `noise1`. If absent, defaults are spawned from
`g_payload_default_teams`, exactly as `dom_spawnteams` does.

### Spawns

`info_player_team1` .. `info_player_team4` (and `..._teamN` once `NUM_TEAMS` grows).
`mapinfo` needs `type plc <pointlimit> <timelimit>` and a `teams=<k>` setting, consumed by
`Payload.m_parse_mapinfo`.

## 5. Going past 4 teams

`NUM_TEAMS` is `4` at `common/teams.qh:3`. Measured extent of the hardcoding: 71
references to `NUM_TEAM_4` across `qcsrc`, 29 to `NUM_TEAMS`, and 0 in `menu/`. The work
is concentrated:

- `common/teams.qh`: 6 `case NUM_TEAM_4` switches (`Team_ColorCode`, `Team_ColorRGB`,
  `Team_ColorName`, `Static_Team_ColorName`, `Team_IndexToTeam`, `Team_TeamToIndex`) plus
  `Team_ColorToTeam`. The disabled `TEAMNUMBERS_THAT_ARENT_STUPID` block gives 1..4 and is
  the natural place to grow, since sequential ids make these switches into arithmetic.
- `common/notifications/all.qh:84`: `APP_TEAM_NUM` is a 4-way nested ternary. Any
  k-generic mode must either extend it or avoid per-team notification variants.
- Per-team **stats** are the real cost, which is why payload packs its per-team weights
  into one stat rather than following `DOM_PPS_RED/BLUE/YELLOW/PINK`.

Payload itself contains no `NUM_TEAMS`-shaped constant beyond loop bounds of the form
`for (j = 1; j <= NUM_TEAMS; ++j)`, so raising the constant is sufficient on the mode side.

## 6. The mesh seam (named, not built)

The cart is the readout, not the integration point. Two hooks carry the demo:

- `payload_think` builds, once per tick, the residual rows the solver consumes: per bot,
  the gap to its team's goal, its distance and line of sight to the cart, its health and
  ammo deficits, and the k-1 per-team pressure terms taken straight from `payload_w[]`.
  The team-pressure terms are why the Gram matrix and the game mode want the same object.
- `havocbot_role_payload` consumes the returned per-bot plan. Per `AGENTS.md`, the local
  `navigation_*` rating always runs; a returned plan overrides it. There is no branch in
  which a bot receives no goal, so a starved solver degrades to visibly worse play rather
  than to stopped bots - and the cart's arclength is the integral of that difference.

## 7. Not done

- Nothing was compiled: `gmqcc` is not built in this tree, and building it was out of
  scope for a scoping pass.
- No RDMA device was opened and no `mesh-flow` / `ibv_*` tool was run, per `RDMA-RULES.md`.
- The cart's bezier evaluation reuses `path_corner.curvetarget` semantics but not
  `SUB_CalcMove_Bezier` itself; the arclength of a bezier segment is approximated by
  chord subdivision, and the subdivision count is unmeasured.
- No payload map exists. The entity format above is a specification, not a shipped map.
