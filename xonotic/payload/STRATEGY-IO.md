# Strategy I/O

The payload server and responder exchange three request streams and one response
stream. The executable schemas are mirrored by
`qcsrc/common/gamemodes/gamemode/payload/sv_payload_strategy_io.qh` and
`tools/strategy_io_schema.py`.

## The gather precondition

`mesh_gather(handle, column, .field, firstedict, count)` copies `count`
**consecutive** edicts:

```c
/* bridge/engine/mesh_ipc.c:305-306 */
for (row = 0; row < n; row++)
    m->req[(size_t)row * m->width + col] =
        (float)prog->edictsfields[(size_t)(first + row) * stride + fld];
```

So a staging pool is a valid gather source only if its edict numbers are
consecutive. `spawn()` does not give that: `PRVM_ED_Alloc`
(darkplaces `prvm_edict.c:267`) scans upward from `reserved_edicts + 1` and
returns the first reusable free slot, so a burst of `spawn()` calls after map
load lands in whatever holes the map's self-removing entities left.

`payload_str_pool_run` therefore builds the run explicitly — keep spawning,
restart the run whenever the new edict is not `prev + 1`, and never free the
rejects so they cannot be recycled underneath the pool — and `attach` re-checks
the property against the live edict numbers and logs it:

```
payload: [PLCPOOL] evt_base 5227 rows 256 contiguous 1
payload: [PLCPOOL] cart_base 5483 rows 4 contiguous 1 obs_base 1 rows 16
```

Client edicts need no such treatment: they are `1 .. maxclients` by construction,
which is why the per-player gather passes `firstedict = 1`.

## Requests

Participant rows have width 40. Columns 0–18 are ID, team, health, armor, ammo,
position XYZ, velocity XYZ, weapon bitset, powerup time, time since spawn, V-cell,
nearest cart, nearest-cart distance, alive, and bot/human control. Columns 19–24 are
the applied packed target, whether it resolved, the stock goal's packed target, live
distance to that goal, whether it matches the applied target, and whether the bot has
reached it. The remaining columns are zero. Positions, velocities, and goal distance
are engine units / 1024.

Cart rows have width 12:

| col | name | meaning |
|-----|------|---------|
| 0 | `ID` | `plc_cart_id` |
| 1 | `DEPTH` | `plc_s / plc_length`, arclength fraction |
| 2 | `LENGTH` | `plc_length`, raw |
| 3 | `CTRL` | controlling team index, 0 = uncontrolled |
| 4 | `SPEED` | `plc_speed_now`, signed |
| 5 | `IDLE` | rollback timer |
| 6 | `BANKMASK` | banked-team bitmask |
| 7 | `PROGRESS` | monotone banked score |
| 8 | `POS_X` | cart world position / 1024 |
| 9 | `POS_Y` | |
| 10 | `POS_Z` | |
| 11 | — | reserved |

A cart is a brush model: `sv_payload.qc:743` sets `view_ofs = mins` and
`sv_payload.qc:437` parks the cart at `payload_pos(...) - view_ofs`, so its world
position on the golden path is `origin + view_ofs`. Columns 8–10 and the
nearest-cart columns of the participant row both use that sum; comparing a
player's origin against a cart's unoffset `origin` is wrong by the cart brush's
`mins`.

Perception-event rows have width 6: cell, kind, observing team, subject, value, and
timestamp. Item and rival instruments enter through this gated event path.

## Responses

Every participant receives exactly four floats: packed target, gain, commitment, and
spawn timing. Packed target ranges are cart
`0+id`, item `65536+id`, rival `131072+id`, and cell `196608+id`. The QuakeC adapter
turns these into additive stock `navigation_routerating` calls; it does not compute
paths or replace havocbot navigation.

`COMMIT` is seconds, applied after the stock navigation timeout through
`navigation_goalrating_timeout_extend_if_needed`.
It is a property of an assignment, not of one instrument: committing to a target
is committing to the trip to it, so `instruments.travel_horizon` gives every
objective the horizon its own travel implies (`IDLE` and `SPAWN_TIMING` excepted —
neither is travelling).

Bot rows enact the response. Human rows retain the same policy assignment as an
advisory value and are identified in telemetry.

## Row logging

`g_payload_strategy_log 1` makes `payload_strategy_log` emit the exact rows the
mesh is about to carry, read back off the staged fields rather than recomputed:

```
[PLCPUB]  <seq> <time> carts <j> clients <n> evtrows <r>
[PLCCART] <seq> <c>  ID DEPTH LENGTH CTRL SPEED IDLE BANKMASK PROGRESS POS_X POS_Y POS_Z
[PLCOBS]  <seq> <edict> ID TEAM HEALTH ARMOR AMMO POS_X POS_Y POS_Z VEL_X VEL_Y VEL_Z
                       WEAPONS POWER TSS CELL NCART NCART_D ALIVE CONTROL
[PLCEVT]  <seq> <row> CELL KIND TEAM SUBJECT VALUE TIME
```

At most seven values per `sprintf`: the QuakeC calling convention carries
`MAX_PARMS = 8` (darkplaces `pr_comp.h:143`) and the format string is one of them,
so an eighth value is dropped without a diagnostic. `strcat` has the same limit.

`solver/strat/measure.py rows <server.log> --out <rows.jsonl>` parses these back
into one record per strategy tick carrying the per-player observation rows, the
cart rows, the event rows, and the instrument descriptors `z` and relation rows
reconstructed with the same `build_instruments` the live operator calls.

## Verification

From `xonotic/`:

```
PYTHONPATH=payload/tools python3 payload/tools/strategy_io_schema.py
payload/build.sh
```

The first command checks widths and target round trips. Projected winner and succession
have one implementation in `solver/strat/game.py`, shared by training and telemetry.
The second command overlays the payload source on a clean QC tree and builds client,
menu, and server gamecode. Curriculum matches copy those exact binaries into their
userdirs.
