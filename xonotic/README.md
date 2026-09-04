# xonotic on the sealed mesh

A Xonotic payload-mode dedicated server publishes 83-column participant rows,
16-column cart rows, and 17-column perception events across the RDMA fabric. The mesh
responder returns a 7-column strategy assignment for every client row. Bot rows enact
it as additive havocbot navigation ratings; human rows retain the assignment as an
advisory signal.

## Online policy training

The real training environment is the dedicated server. Run the server with a sampled
map/roster/cart/controller configuration and run the strategy responder with training
enabled on the mesh peer:

```
cd ~/mesh/xonotic
mesh-python -m solver.strat.strat_responder --train --off-policy-players 2
```

The responder is a service and runs until `SIGINT`, `SIGTERM`, or `SIGHUP`, then drains
the pending transition and writes its checkpoint and runstate before exiting. The
curriculum owns match duration and sends `SIGTERM` at the match boundary.

Each server transition updates the asymmetric winner/loser critics, the shared policy,
and the local dynamics ensemble before the next response is emitted. The off-policy
count may be any value from zero through the number of active participants; those rows
receive uniform exploratory assignments and carry their behavior log-probabilities in
telemetry. Bot and human rows are both represented. Bots enact the assignment through
the havocbot rater; a human row is an advisory assignment until a player-facing channel
or realized-action classifier is added. There is no simulator or bootstrap trainer in
the strategy path; evidence comes from live server transitions.

The match curriculum is a distribution over actual server launches: map, team count,
players per team, controller mixture, cart-bearing entity overlay, skill, seed,
perturbation regime, and off-policy participant count. `solver.strat.curriculum`
extracts each BSP, generates that match's exact team/cart entity overlay, launches the
dedicated server and training responder, asks the server to `quit` over stdin, and
continues after a failed match. Every match directory contains the commands, UTC
timestamps, return codes, entity hashes, logs, telemetry summary, and checkpoint
lineage in `match.json`; the run-wide record is `matches.jsonl`.

The curriculum runs this build automatically before its first match; it is also
available directly:

```
payload/build.sh
```

Generate a reproducible mixed-count schedule and execute it:

```
cd ~/dox/mesh/xonotic
mesh-python -m solver.strat.curriculum --generate 96 --seed 20260830 \
  --server-host game-node --remote-engine /opt/xonotic/darkplaces-dedicated \
  --remote-basedir /opt/xonotic/Xonotic \
  --maps runningmanctf,dance --team-counts 2,3,4,5 \
  --players-per-team 2,4,8 --cart-counts 1,2,3,4 \
  --skills 2,5,8 --perturbations baseline,fast,slow,volatile \
  --off-policy-counts 0,1,2,4 --human-counts 0 --heldout-fraction 0.2 \
  --duration 600 --run-dir solver/strat/runs/curriculum-20260830
```

The listed team and cart axes seed the first outer game configurations. Later cycles
sample half/center/double neighborhoods around the observed whole-mesh operating point
from preceding cycles. Within every match the server opens its compiled player capacity
and the profiler grows and brackets bot count in whole-team quanta. The 8787 whole-mesh
stream records memory-bandwidth fraction, FLOP/s bounds, per-node deadline load, and the
disclosed numerical operating loss at every point. The minimum-loss observation is the
center of the next sample neighborhood; no feasibility or player-count filter admits or
rejects a point. The initial `players-per-team` value is a launch seed, not a capacity
claim.

The same command with `--dry-run` resolves and records the complete schedule and
commands without requiring Xonotic, MLX, or RDMA. JSON and JSONL manifests are also
accepted with `--manifest`. A JSON manifest may contain `defaults`, `matches`, and
`heldout`; each match accepts `map`, `bsp` or `entity_file`, `teams`, scalar or list
`players_per_team`, `carts`, `controllers`, `skill`, `duration`, `seed`,
`perturbation`, `server_cvars`, `server_args`, `client_commands`, and
`off_policy_players`. Held-out matches load the realized checkpoints through the same
policy inference path without `--train` and never advance the training lineage. Optional
`client_commands` are launched as argv without a shell for externally controlled participants; the
telemetry record reports the bot/human counts actually observed rather than treating
the requested controller mixture as evidence. Generated schedules can vary human
counts with `--human-counts`; `--human-client-command` launches one command per such
participant and expands `{port}`, `{map}`, `{seed}`, `{match}`, and `{client}` tokens.

Study cycles atomically replace `study.json` with the cumulative release measure. It
contains the mirrored outcome measure, assignment-level and checkpoint-realization Elo
coordinates, every behavior coordinate overall and by perturbation, baseline-centered
perturbation displacements, per-arm spawn-wave timing, the independent participant-fusion
and residual-fusion intervention tensors
intervention, checkpoint/update realization, map advanceability, and whole-fabric
operating coordinates. The fabric section retains the identities of responder and expert
nodes and hosts, rank/hidden/expert/top-k coordinates, remote request and output row mass,
FLOP/s and byte/s intervals, distributed deadline load, and the measured local-only
deadline interval. Missing telemetry remains `null`; it is not converted to zero.

Summarize one or more realized telemetry streams with:

```
mesh-python -m solver.strat.joracle.metrics solver/strat/runs/curriculum-20260830/*/telemetry.jsonl
```

The summary separates winner-retention and loser-acquisition trials and aggregates
importance weights, W/L losses, dynamics error, ensemble disagreement, local-control
singular values, credited horizon, controller class, and behavior-policy class.

## Live mesh measurements

The strategy responder publishes FLOP, byte, row, deadline, J-lens, and J-oracle
coordinates through the generic workload interface. The node telemetry service retains
them in memory on port 8788. The existing whole-mesh observer on port 8787 combines the
laptop and Mini without importing a Xonotic schema. `joracle/demo.sh` checks or
kickstarts that observer; it does not launch another HTTP service or browser pane.

Open `http://127.0.0.1:8787`. The phase-space page displays both nodes and the top-level
scalar coordinates of every producer measure. `http://127.0.0.1:8787/latest.json`
exposes the complete nested `workload.producers[].measures` objects. The J-lens measures
the exact composer features and authoritative server coordinates against the selected
participant-instrument J row. A separately named minimum-norm empirical L2 projection
publishes its per-coordinate residual square measure. The J-oracle joins the exact
source features, authoritative source state,
policy intervention, delivered response, active route, goal, touch, subsequent state,
and realized counters by each server-owned response sequence. There is no simulated input
path in either computation.

Shape, numeric range, finiteness, and W/L values are recorded on every response. The
in-memory J measure consumes the exact selected J rows and all eleven composer arrays
on every response. JSONL records those large arrays every fifty responses by default;
`--model-sample-every` changes only that serialization cadence. Runtime process
multiplicity, host identity, bridge state, telemetry
age, reconnect epochs, and producer mass are published as factual coordinates rather
than collapsed into a service label.

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
- solver: `solver/strat/strat_responder.py`, `solver/xonwire.py`
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

The external strategy tick consumes the full player/instrument state and returns one
typed response row per participant. It is a global matrix-fusion/MoE policy computation, not a
causally independent per-player computation, so it does not pretend that reordering
response rows creates parallel game simulation. The decoder validates the complete row
set and the game commits target kind, target identity, gain, commitment, and spawn delay
together.

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

The live 26042 process now covers gameplay scale: 255 bots plus one observer, 256
configured teams, 32 supported carts, every team index exercised, and four completed
matches won by teams 220, 99, 97, and 145. Its current process has no engine/object/
network-buffer error over nearly three hours. Eight map windows contain at least 9,017
simulated seconds over 10,656 wall seconds (84.6%) at approximately 97% of one host
core. This is a real functional run, not compile evidence. It is not a strategy-policy
run: no responder is attached and no `[PLCBARRIER]` commit appears, so it cannot support
the matrix-fusion/Elo or two-host planning claims.

Post-training release evaluation can run the release-rank matrix-fusion/MoE operator across both hosts without replacing the game or reporter services:

```sh
xonotic/solver/strat/joracle/evaluate-distributed.sh
```

The command resolves the live Mini address, then delegates the entire release to the curriculum instead of running a second responder/expert supervisor. The curriculum incrementally realizes the remote dedicated engine, complete Xonotic data path, matching gamecode userdir, strategy source, RDMA runtime, and managed Python environment. One stable expert PID owns the fixed relay socket across match transitions; every launch starts with a verified TERM transition and every match ends through the same path.

The game host remains the bridge client and relays expert slots to its local MLX worker;
the responder host runs the rest of the policy and records every scale call, round-trip,
worker compute interval, logical and physical row mass, dynamic tensor widths, expert
loads, and both checkpoint tags. The curriculum keeps the expert available across match
transitions and samples a low-cadence local substitution after returning the live
response. The 8787 stream reports both host roles, local/remote output differences,
distributed and local-only deadline loads, achieved FLOP/s bounds, achieved byte/s
bounds, and their sample variances without turning any coordinate into a release gate.
