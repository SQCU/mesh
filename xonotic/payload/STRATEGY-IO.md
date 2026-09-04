# Strategy I/O

The payload server and responder exchange three request streams and one response
stream. The executable schemas are mirrored by
`qcsrc/common/gamemodes/gamemode/payload/sv_payload_strategy_io.qh` and
`tools/strategy_io_schema.py`.

## Interface law

The game producer emits four row types: participant observations, cart observations,
events, and the response destination rows that preserve participant order. Each field
is one named engine value in engine units. Type discriminants and identities occupy
separate fields. The producer may copy, stage, and frame these values; it may not pack,
normalize, aggregate, clip, rank, infer, project, or replace them.

The responder consumes those literal rows. It may derive instruments, the team
observation-support measure, closed-form game semantics, learned projections, policy
samples, and diagnostics only after receipt. Its output producer emits an
instrument-kind discriminant, a target-kind discriminant, the target's literal engine
identity, the two literal target-cell coordinates, and three actuator values. The game
consumes those eight values and alone interprets them as navigation and respawn
behavior.

The distributed expert interface is narrower. Its producer emits the contiguous
float32 IR residual tensor with shape `(rows, d_ir)`. Its consumer applies the learned
scale projection, routing, MoE, residual-feature Gram-matrix context, and learned output projection and returns
one float32 tensor of exactly the same shape. Neither side pads semantic rows, reduces
rows, or changes row order. Execution diagnostics use a separate typed metadata
message and never enter the returned tensor.

## Endpoint routing

`g_payload_mesh_node` is the server's destination node for observations. The
responder's `--peer-node` is independently the destination node for responses.
For the two-host topology, the MBP server uses `g_payload_mesh_node 1` and the
Mini responder uses `--peer-node 0`. The curriculum exposes these as
`--strategy-node` and `--peer-node`; they must not be collapsed into one value,
because each names the other endpoint from a different host. Node 0 remains a
valid server destination for local diagnostics, but production evaluation uses
opposite endpoints so the engine and responder cannot race as consumers of one
local completion queue.

## Gather realizations

`mesh_gather(handle, column, .field, firstedict, count)` copies `count`
**consecutive** edicts:

```c
/* xonotic/darkplaces-work/mesh_ipc.c */
for (row = 0; row < n; row++)
    m->req[(size_t)row * m->width + col] =
        (float)prog->edictsfields[(size_t)(first + row) * stride + fld];
```

Client edicts are `1 .. maxclients` by construction, so participant rows use one
consecutive row gather. Cart state uses a checked consecutive staging run because its
source entities are an intrusive list rather than a consecutive edict interval. Event
state does not use such a run: each event is appended to a linked extent and
`mesh_gather_list` copies that exact list into a dynamically enlarged handle. The cart
pool, participant interval, and event extent are therefore three distinct realizations.

## Requests

The producer is `payload_strategy_gather`. Its contract is to stage engine values,
in engine units, without feature engineering. The gather layer copies those staged
float fields into xonwire pages. It does not normalize, aggregate, rank, clip, infer,
or reserve feature columns.

Participant rows have width 83:

| cols | fields | source |
|---|---|---|
| 0–3 | `ID TEAM HEALTH ARMOR` | edict, team index, health and armor resources |
| 4–9 | `AMMO_SHELLS AMMO_BULLETS AMMO_ROCKETS AMMO_CELLS AMMO_PLASMA AMMO_FUEL` | six distinct resources |
| 10–15 | `POS_X POS_Y POS_Z VEL_X VEL_Y VEL_Z` | raw world position and velocity |
| 16–18 | `WEAPONS_X WEAPONS_Y WEAPONS_Z` | all three stock `WepSet` words |
| 19–21 | `STRENGTH_FINISHED SPAWN_TIME ENGINE_TIME` | absolute engine timestamps |
| 22–25 | `CELL_X CELL_Y ALIVE CONTROL` | literal V-cell lattice coordinates, alive flag, bot/human flag |
| 26–35 | `APPLIED_TARGET_KIND APPLIED_TARGET_ID APPLIED_TARGET_CELL_X APPLIED_TARGET_CELL_Y TARGET_RESOLVED GOAL_TARGET_KIND GOAL_TARGET_ID GOAL_TARGET_CELL_X GOAL_TARGET_CELL_Y GOAL_PRESENT` | applied and stock-navigation target state |
| 36–38 | `GOAL_POS_X GOAL_POS_Y GOAL_POS_Z` | literal stock-navigation goal position |
| 39–46 | `RESPONSE_SEQ RESPONSE_TIME ROUTE_SEQ ROUTE_LATENCY GOAL_SEQ GOAL_LATENCY TOUCH_SEQ TOUCH_LATENCY` | causal instrumentation |
| 47–53 | `ENEMY_DAMAGE_DEALT ENEMY_DAMAGE_TAKEN ENEMY_KILLS DEATHS PICKUPS CART_PUSH CART_CONTEST` | monotone realized counters |
| 54–61 | `OUTCOME_A_SEQ OUTCOME_A_ENEMY_DAMAGE_DEALT OUTCOME_A_ENEMY_DAMAGE_TAKEN OUTCOME_A_ENEMY_KILLS OUTCOME_A_DEATHS OUTCOME_A_PICKUPS OUTCOME_A_CART_PUSH OUTCOME_A_CART_CONTEST` | exact interval outcome measure for one active route |
| 62–69 | `OUTCOME_B_SEQ OUTCOME_B_ENEMY_DAMAGE_DEALT OUTCOME_B_ENEMY_DAMAGE_TAKEN OUTCOME_B_ENEMY_KILLS OUTCOME_B_DEATHS OUTCOME_B_PICKUPS OUTCOME_B_CART_PUSH OUTCOME_B_CART_CONTEST` | exact interval outcome measure for the other route that can occur between gathers |
| 70–82 | `SWIZZLE_ACTIVE SWIZZLE_EPOCH SWIZZLE_PLAYER_COUNT SWIZZLE_SPOT_COUNT SWIZZLE_SLOT_COUNT SWIZZLE_TICKET SWIZZLE_COHORT SWIZZLE_COHORT_COUNT SWIZZLE_GENERATION SWIZZLE_SCHEDULED_TIME SWIZZLE_ACTUAL_TIME SWIZZLE_LANE SWIZZLE_SPOT` | literal respawn-scheduler state and selected spawn entity |

The server gathers before it scatters and can install at most one new response before
the next gather. Therefore an interval contains at most the route already active at
its boundary and the route installed by that one scatter. The two banks retain both
sequence identities and their separate seven-coordinate measures, including zero-valued
coordinates. They are a consequence of the state-machine schedule, not a workload or
history cap.

The swizzle ticket is the dense rank of the live `(within-team ordinal, team)` atom.
The server records player, spot, and slot masses with that ticket when death schedules
the respawn, then uses the recorded ticket and generation when it selects the spawn.
Runtime joins therefore affect later schedules without changing an already scheduled
player. A later schedule that reaches an occupied `(scheduled time, lane)` atom advances
to the next cohort time, so a join cannot alias a pending reservation. `SWIZZLE_EPOCH`
and participant `ID` identify the scheduling event;
`SCHEDULED_TIME`, `ACTUAL_TIME`, `LANE`, and `SPOT` expose its realized state transition.

There is no aggregate ammo column, weapon population count, powerup-remaining
column, time-since-spawn column, nearest-cart pair, goal distance, response age,
goal-match flag, or target-touch flag. Consumers can compute a diagnostic from the
literal source fields; such a diagnostic is not another wire schema and is not fed
back into XAN.

Cart rows have width 16:

| col | name | meaning |
|-----|------|---------|
| 0 | `ID` | `plc_cart_id` |
| 1 | `PATH_POSITION` | raw `plc_s` |
| 2 | `PATH_LENGTH` | raw `plc_length` |
| 3 | `CONTROL_TEAM` | raw `plc_ctrl`, 0 = uncontrolled |
| 4 | `SPEED` | `plc_speed_now`, signed |
| 5 | `IDLE_TIME` | raw `plc_idle` |
| 6 | `LEAD_TEAM` | raw `plc_lead` |
| 7 | `SECOND_TEAM` | raw `plc_second` |
| 8 | `HOME_TEAM` | raw `plc_home` |
| 9–11 | `POS_X POS_Y POS_Z` | raw grounded cart world position |
| 12 | `SUPPORTS_PLAYER` | live trace of a standing player hull supported by the cart |
| 13 | `TEAM_COUNT` | authoritative live `payload_teams` count |
| 14 | `ROLLBACK_ACTIVE` | raw `plc_rollback_active` |
| 15 | `ROLLBACK_TARGET` | raw checkpoint arclength `plc_rollback_target` |

A cart has one server-side physical definition: a solid pushable bounding box with
`view_ofs = '0 0 -24'`. Path placement uses `payload_pos(...) - view_ofs`, so the
authoritative point on the grounded path is always `origin + view_ofs`. The producer
passes that point literally. Normalized path depth is the closed-form consumer
operation `PATH_POSITION / PATH_LENGTH`; it is not a second cart field. The deleted
`BANKMASK` and `PROGRESS` fields read checkpoint-node members from a cart entity and
therefore normally published zeros while claiming to mean values they did not carry.

Event rows have width 17: `KIND TIME OBSERVER TEAM SUBJECT CELL_X CELL_Y
TARGET_CELL_X TARGET_CELL_Y POS_X POS_Y POS_Z RESPAWN_TIME HEALTH LINK_LENGTH AMOUNT
RESPONSE_SEQ`. `OBSERVER` is the observing or acting edict,
`TEAM` its team, `SUBJECT` the observed or affected identity, and position is the exact
observed world position. The last five columns are distinct literal types rather than
a kind-dependent value union. Events are appended on every observation to a live
linked extent and each entity row is gathered once before that extent is released.
The published row count is the exact event mass accumulated since the preceding
strategy step. It is not a ring capacity, history window, configured ceiling, or
cross-device segmentation limit. A zero-event step is a typed zero-row event stream
carrying the same tick identity as its observation and cart streams. The producer does
not suppress a repeated observation or turn an absolute timestamp into a phase or age.

## Consumer interface

`strategy_io_schema.py` is the canonical responder field-index definition, mirrored
by the producer constants in `sv_payload_strategy_io.qh`. `LiveBelief.ingest`
decodes the typed event envelope. `build_runtime_frame` decodes participant and cart
identities and performs the licensed closed-form cart projection. `player_features`
constructs XAN by copying the named scalar columns and losslessly expanding each of
the three 24-bit `WepSet` words into named binary coordinates. It does not scale, clip,
sum, count, or apply a nonlinear transform. RMS normalization belongs after the
learned Quinn projection.

The J measurement uses the same schema-owned types. Real-valued state coordinates
produce ordinary source-to-target differences. Weapon words produce 24 binary
differences. Edict, team, cell-coordinate, target-kind, target-identity, cart-identity, and
response-sequence coordinates produce exact categorical source-to-target atoms and
observed one-hot atoms for the J lens; they are never subtracted, averaged as IDs, or
treated as ordered magnitudes.

ZED contains the instrument kind tag followed by literal target availability,
position, path position, path length, speed, respawn timestamp, health, and observed
timestamp. Variant fields absent for an instrument kind are zero and the kind tag
defines their type. IDs, team IDs, and cell-coordinate pairs route rows and define
observation support; they are never multiplied as ordinal learned features. Team IDs
remain a length-`l` vector and are expanded to equality only while constructing the
participant Gram matrix, rather than being stored as an `l × l` replay feature.

The belief slot is likewise lossless: four event-kind coordinates followed by raw
position XYZ, respawn time, health, link length, and amount. The canonical buffer keeps
the newest current row for each observed subject and subject class, so distinct entities
in one cell remain distinct. RHO contracts every entity row toward zero by elapsed time,
and GIGI repeats the cell's bounded graph-distance weight for each entity row. PHIL
projects each row once; the single weighted integration is RMS-normalized before QUINN.
There is no respawn-phase saturation, health `tanh`, sighting-count transform, fixed
projection, cell-wide latest-row overwrite, or second belief integrator.

## Transport extent

The row counts above describe game tensors; they do not size the RDMA transport. Xonwire
version 3 frames a tensor as a typed stream kind, 64-bit float-value offset, and 64-bit
total value extent. Observation, cart, event, strategy response, expert request, expert
response, and expert telemetry are kinds 1 through 7 respectively; a receiver never
infers a stream type from row width.
Page boundaries may split a row, and the number of pages is derived from the tensor and the
live mesh payload size.

`rdma/mesh-client.c` is the only arena scheduler. Callers either submit under current credit or
give it an arbitrary pending extent; it selects free pages, records ownership at submission,
reclaims pages from bridge completions, and continues through as many arena revolutions as the
workload requires. DarkPlaces, the Python responder, and the expert worker do not own arena
cursors or equal partitions. Reassembly allocates from the received total extent and tracks
exactly the number of page offsets that extent implies. The engine-to-expert relay preserves
each mesh page byte image inside a scatter/gather datagram batch. `xonwire.py` owns both RDMA page waves and local
datagram batching; the expert worker does not inline a second buffer scheduler.

DarkPlaces `meshxhandle.mask` is a local receipt bitmap indexed by derived page offset.
It never crosses the wire or enters game responses. Page mass is derived from
`values_total`. Socket batch size is derived from the live socket buffer and adjusted
to the socket's reported datagram limit.

Expert residual data and expert execution metadata are different message kinds.
Kind 5 carries the residual request, kind 6 carries exactly the residual response, and
kind 7 carries one typed metadata row: residual-feature Gram-matrix minimum, maximum,
finite-coordinate mass, processed row count, elapsed seconds, then one load value per
expert. Process identity and saved policy state belong to process telemetry. The worker
processes every completed request in order.

`maxclients`, cart count, the live event mass, and model tensor shape are semantic
dimensions because they say what the game or operator realized. The 4 KiB page and the granted QP depth are hardware frame
budgets. Deeper page-table rings absorb application polling bursts without pretending that
the QP depth is a tensor extent. These capacities control simultaneous occupancy, never how
much work may pass through.

The consumer joins observation, cart, and event streams by their literal engine tick before
constructing a runtime frame. Arrival order cannot combine rows from different server states,
and an empty event extent cannot cause an older event extent to be replayed.

## Responses

Every participant receives exactly eight floats: instrument kind, target kind, target
identity, target cell X, target cell Y, gain, commitment, and spawn timing. Instrument
kinds are `0 none`, `1 push cart`, `2 suppress cart`, `3 contest post`, `4 hunt rival`,
`5 explore cell`, `6 spawn timing`, and `7 idle`. Target kinds are `0 none`, `1 cart`,
`2 item`, `3 rival`, and `4 cell`; target identity is the literal cart ID or
edict number, while cell targets use the two literal lattice coordinates. The QuakeC
adapter turns the separate kind and target coordinates into additive stock
`navigation_routerating` calls; it does not compute paths or replace havocbot
navigation.

The response producer samples an instrument from the canonical policy logits and selects
`GAIN COMMIT SPAWN` from the learned Gaussian actuator projection on that same
participant/instrument IR row. The projection emits three means and three log-scales;
the sampled policy emits three control values. `response_rows` copies the instrument
kind, target kind, literal identity, and those values. It has no packed target, modulo action repair,
default value, clipping, distance formula, lease, semantic prior, multiplicity
correction, or queue supersession.

The response consumer is the QuakeC behavior adapter. `TARGET` selects the stock
cart, item, rival, or waypoint goal-rating path; `GAIN` is its additive rating weight;
positive `COMMIT` extends the stock navigation timeout; and a changed `SPAWN` value
adjusts the dead participant's respawn time. Those are game semantics at the consumer,
not transport transformations.

Bot rows enact the response. Human rows retain the same policy assignment as an
advisory value and are identified in telemetry.

## Policy release artifact

The online checkpoint is resumable training state: policy parameters, Adam moments,
replay, RNG state, and provenance. The held-out runtime needs only policy parameters
and the update, architecture, policy-version, and reward-contract provenance used by
the loader and study integrity checks. Export that artifact without decoding or
rewriting the parameter arrays:

```
PYTHONPATH=xonotic mesh-python -m solver.strat.checkpoint_release training.npz policy.npz
```

The exporter copies the original NumPy members byte-for-byte, discards optimizer,
replay, and RNG members, and writes the destination atomically. Training continues
from the training checkpoint; inference and held-out studies load the policy artifact.

## Row logging

`g_payload_strategy_log 1` makes `payload_strategy_log` emit the exact rows the
mesh is about to carry, read back off the staged fields rather than recomputed:

```
[PLCPUB]  <seq> <time> carts <j> clients <n> evtrows <r>
[PLCCART] <seq> <c>  ID PATH_POSITION PATH_LENGTH CONTROL_TEAM SPEED IDLE_TIME
                     LEAD_TEAM SECOND_TEAM HOME_TEAM POS_X POS_Y POS_Z SUPPORTS_PLAYER TEAM_COUNT
                     ROLLBACK_ACTIVE ROLLBACK_TARGET
[PLCOBS]  <seq> <edict> <the 83 participant fields above, in schema order>
[PLCEVT]  <seq> <row> KIND TIME OBSERVER TEAM SUBJECT CELL_X CELL_Y
                     TARGET_CELL_X TARGET_CELL_Y POS_X POS_Y POS_Z RESPAWN_TIME
                     HEALTH LINK_LENGTH AMOUNT RESPONSE_SEQ
```

At most seven values per `sprintf`: the QuakeC calling convention carries
`MAX_PARMS = 8` (darkplaces `pr_comp.h:143`) and the format string is one of them,
so an eighth value is dropped without a diagnostic. `strcat` has the same limit.

`solver/strat/measure.py rows <server.log> --out <rows.jsonl>` parses these back
into one record per strategy tick carrying the per-player observation rows, the cart
rows, the event rows, the instrument descriptors `z`, and the team observation-support
measure reconstructed with the same `build_instruments` the live operator calls.

## Verification

From the repository root:

```
xonotic/payload/build.sh
```

The build compiles client, menu, and server gamecode as one release set. Projected
winner and succession have one implementation in `solver/strat/game_value.py`, shared by
training and telemetry. Curriculum matches copy the same release set into their
userdirs.
