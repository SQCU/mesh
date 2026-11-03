# Xonotic on mesh

The game's numerical caller is `planner/plan.py`, the routed-expert bot planner. Each
rank holds a block of every expert's columns (Megatron-LM tensor parallelism), and one
all-reduce per tick from `rdma/mesh-collective.h` completes the expert output over an
explicit link map (`examples/links-pair.conf` for the pair). `rdma/mesh.py` binds it
(`AllReduce`). With the bridges up on the region, run the same command on every node
from the repository root:

    MESH_REGION=<region> PYTHONPATH=xonotic:rdma bin/mesh-python -B -m planner.plan [ticks] [bots]

Pair run, 2026-09-23 (mesh aa36138; bridges on /xonrdma; M5 rank 0 with 1408 of each
expert's 2048 columns, M4 rank 1 with 640; 2000 ticks of 480 bots, width 256, 8 experts):
both ranks completed 960,000 bot plans and ended with the same objective split
`[0, 332, 148, 0, 0]` and 353,792 switches. The M5's share is the critical path: it
publishes every 2.22 ms and spends 0.03 ms of each tick in the collective, while in the
bridge ledger the M4's partial arrives a median 1.25 ms before the M5's own, so the M4
idles for most of each tick. Records: metal-microbench `output_data/restore-20260923-xonotic`.

Measured split, 2026-09-24 (mesh afd40eb): a pair ledger pass at 1408:640 timed each rank's
own columns (`solve_ms`) at 1.393 µs per column on the M5 and 0.946 µs on the M4, so equal
finish falls at 832:1216, now the default in `planner/split.json`. The pair re-run at 832:1216
takes 1.43 ms per tick (M5 publish interval 1.41 ms) against 2.22 ms, with the M4's partial
arriving a median 0.03 ms before the M5's instead of 1.25 ms; both ranks again end with the
objective split `[0, 332, 148, 0, 0]` and 353,792 switches. Records: metal-microbench
`output_data/native-20260923-xonotic-{ledger,split}`.

The old persistent-policy runtime, learner/responder, curriculum, frame runtime and
distributed launcher scripts were removed; they are not alternative entry points.

The game engine, payload, rendering and telemetry/viewer source remain.
Build the payload from this directory with `payload/build.sh`.

## Telemetry and retained viewer

The node telemetry service publishes infrastructure activity on port 8788 and
its whole-mesh observer on port 8787. The retained `solver.strat.joracle.server`
serves recorded J-space and policy observations. From this directory:

```sh
../bin/mesh-python -m solver.strat.joracle.server --run-dir /path/to/recorded/run
```

The default application views are `/j` and `/policy` on port 8795. Existing
viewer processes and game services are independent of the deleted launchers.
The records below describe historical applications; their successful operation
is not a claim that the deleted training entry points remain available.

The following August 30 observations describe an earlier model and transport revision;
they are historical evidence, not current architecture or training claims.

The live run on 2026-08-30 used the already configured `mesh-mini` SSH alias without a
repository checkout change on that node: Xonotic node 0 ran 12 bots across four teams
and three carts; MLX on node 1 consumed 223 observation/cart/event snapshots, emitted
83 responses, and completed 82 online updates in 45 seconds. Cart identities remained
`0,1,2`, all eleven exported tensor families remained finite, the realized IR shape
was `12 x 128`, and both bridge `bad` counters remained zero. Running against the
Mini's system Python 3.9 exposed and removed two accidental Python-3.10 dependencies
(`zip(strict=True)` and `int.bit_count`). The same execution also replaced two tiny
belief-projection matrix multiplies with explicit contraction and weighted reduction,
which removed the Accelerate runtime overflow warnings while preserving the bounded
belief calculation.

The same evaluator found two restart failures. Game requests and solver responses had
shared one sequence counter, so the game repeated its last response ID after a solver
exit; requests now advance independently and each response echoes the request ID it
answers. An immediate replacement could also lose its bridge registration to a stale-PID
census race; the bridge now compare-exchanges the dead PID before cleanup. In the rebuilt
game, a 45-second run delivered 147 complete three-stream snapshots and 92 policy
responses, with the first 39 responses matching all 39 snapshots before online compute
became the limiting rate. A later responder attached to the same still-running server
and began consuming immediately.

- engine bridge: `darkplaces-work/mesh_ipc.c`
- game code: `qcsrc/common/gamemodes/gamemode/payload/sv_payload_strategy_io.{qc,qh}`
- historical solver: removed responder plus retained `solver/xonwire.py`
- build tree (not in this repo): `~/dox/xonotic/build-engine`, `~/dox/xonotic/build-qc`

## Historical transport validation

The following measurements belong to the superseded 16-column A/B worker. They
validate the engine/mesh causal path, not the current learned responder or its training
objective.

Verified 2026-08-28 on the live pair, four 150 s matches plus one kill test, bridges
never restarted (`up_ms` monotonic 2141 s -> 3131 s, PIDs 97484 / 74614 unchanged).

Transport, per match: MBP `sent` 1399 / mini `recvd` 1399, mini `sent` 1398 /
MBP `recvd` 1398, `bad` 0. 1399 publishes in 150 s is the 10 Hz cart think rate,
one slot per tick (16 rows of 16 floats + 28 B header fits one 4090 B slot).

Policy A/B, identical binary, pk3, cvars and map (`runningmanctf`, 5 path nodes,
8 bots, `bot_join_empty 1`); the only difference is the solver's `--policy` flag.
`held` is engine state read back over the fabric: request column 15 is
`payload_mesh_objective[team]` written by `mesh_scatter` on an earlier tick.

| run | picks (share by objective 0..4) | held (share by objective 0..4) | cart progress first -> last | mean bot distance to cart, team 2 |
|---|---|---|---|---|
| nearest 1 | .06 .16 .00 **.78** .00 | .53 .18 .00 .29 .00 | 0.712 -> 0.623 (min 0.610) | 1193 |
| nearest 2 | .07 .14 .01 **.79** .00 | .54 .16 .00 .30 .00 | 0.712 -> 0.620 (min 0.610) | 1160 |
| inverted 1 | .00 .00 **.36** .00 **.64** | .50 .00 .44 .00 .06 | 0.712 -> 0.773 (max 0.774) | 475 |
| inverted 2 | .00 .01 **.31** .00 **.69** | .50 .00 .41 .00 .09 | 0.712 -> 0.742 (max 0.760) | 680 |

The two policies drive disjoint objective sets (`{1,3}` vs `{2,4}`), and the effect
reaches the world: under `nearest` the cart never rises above its 0.712 start and ends
0.62; under `inverted` it never falls below 0.712 and ends 0.74-0.77. Team 2 bots sit
2-2.5x closer to the cart under `inverted`. Both replicates agree within policy and
the between-policy gap is far outside the within-policy spread.

Solver absent: the worker was SIGTERMed 75 s into a match. The server kept publishing
at full rate for the remaining 90 s (`sent` 255301 -> 256214, `recvd` frozen at 8858),
kept simulating, logged no error, and exited 0 on `quit`. `mesh_poll` returns no new
sequence, `payload_mesh_tick` returns early, and the last-known objectives stand.

## Current 256-player / 256-team payload contract

The historical ceiling above no longer describes the payload branch. The engine and
scoreboard admit 256 clients, and shared score, spawn, sound, weapon-effect, hook,
vehicle, damage-text, and team entity streams carry literal short player or team
identities. There is no byte-zero reconstruction of player 256 or team 256. Payload cart state and
ribbon endpoints no longer use the old four-team packed stats or four-bit team aliases:
cart controller/leader/runner-up/home and link endpoints are reliable entity messages
with full team indices. The team dialog exposes a numeric selector alongside the stock
color shortcuts; numeric selection and best-team balance compare the direct team arrays,
so team indices 25–256 do not fall through the legacy 24-bit team mask.

The external strategy tick consumes the native source families and returns one typed
response row per addressed state page. The [policy program](../design/POLICY-PROGRAM.md)
defines the global interaction, common output projection and private-view integration.

The bot-controller frame has a separate native `VCellPlan`. `StartFrame` opens one
controller batch and advances `bot_think` exactly once for every bot before DarkPlaces
begins its per-client physics callbacks. World-aware decisions retain ordinary ordered
QuakeC semantics; due keyboard controllers deposit rows instead of applying their pure
transform inline. The native plan derives a three-dimensional
cell extent and causal-cell radius from live player hulls, speed, and frame time, colors
spatially separated cells, and writes the active wave onto every scheduled bot. The pure
Havocbot keyboard transform consumes each entire wave as one structure-of-arrays vector
loop; there is no fixed 512-row buffer or synthetic 32-lane warp. Entity/global delta
transactions are outside this kernel's contract. Fourteen row inputs scatter into the
kernel and nine controller outputs stable-gather at the wave barrier. Vector results commit
after all vector work completes, then the existing engine callbacks consume
the updated controller state exactly once. `bot_nextthink` prevents those callbacks from
recomputing a controller row in the same frame. `[BOTWARP]`, `[VCELL]`, and `[BOTQC]`
report plan shape, native consumption, producer staging, scalar work, and fallback work;
`g_payload_vcell_log 1` enables these without dumping the full strategy transport. A
missing native field leaves staged rows for the scalar controller, so it cannot suppress
a bot update. Full world physics remains ordered by DarkPlaces' per-client callback;
parallel world-delta execution requires a reentrant trace/collision command-buffer
boundary and is not claimed here.

Static geometry measurements exist for all 30 selectable maps. The fused map's
8-team/4-cart artifact measures path lengths of 1,105 / 1,246 / 1,246 / 752 units,
four origins separated by 4,371 units, zero head-on flow, and 240 spawn-to-track
distances of at least 560 units.

An earlier run on port 26042 recorded gameplay scale: 255 bots plus one observer, 256
configured teams, 32 supported carts, every team index exercised, and four completed
matches won by teams 220, 99, 97, and 145. That process recorded no engine/object/
network-buffer error over nearly three hours. Eight map windows contained at least 9,017
simulated seconds over 10,656 wall seconds (84.6%) at approximately 97% of one host
core. This historical run had no strategy responder attached and no `[PLCBARRIER]`
commit, so it cannot support
the matrix-fusion/Elo or two-host planning claims.

