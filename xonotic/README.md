# xonotic on the sealed mesh

A Xonotic payload-mode dedicated server publishes native observations, cart/team
state, events and addressed Havocbot view pages. The responder returns continuous
rates for those view pages. The [policy program](../design/POLICY-PROGRAM.md) is the
canonical description of source layouts, shared representation, learned outputs,
exact exponential integration, training and execution boundaries.

## Online policy training

The mixed-team training entry point is `solver.strat.curriculum --joint-training`.
It assigns teams randomly between stratCGT-PPO (`matrix_fusion`) and `terminal_win`,
and carries both full policy/optimizer/replay checkpoints across matches. The two use
the same architecture and learner implementation. Current per-policy learning measures
are in each match's `learning.json`; actual round outcomes are in `outcome.json`.
See [the objective, replay and evaluation contract](../design/JOINT-POLICY-LEARNING.md)
for what these measures establish, and what they do not.

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

Training consumes live server transitions through the common policy and W/L value
program. Native application acknowledgments determine actor credit; human observations
can contribute value-learning data. The [policy program](../design/POLICY-PROGRAM.md)
defines the complete loss and continuous view interface, and
[joint policy learning](../design/JOINT-POLICY-LEARNING.md) defines controller ownership
and exploratory rows.

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

Study cycles atomically replace `study.json` with cumulative match outcomes,
checkpoint-realization Elo estimates, observed application and native game measures,
optimization records, and fabric measurements. Missing telemetry remains `null`;
retained historical fields do not establish coverage for the current policy.

Summarize one or more realized telemetry streams with:

```
mesh-python -m solver.strat.joracle.metrics solver/strat/runs/curriculum-20260830/*/telemetry.jsonl
```

The summary records available application acknowledgments, native outcomes, cart and
resource measurements. The policy viewer reports the current optimization measures
described in the [policy program](../design/POLICY-PROGRAM.md).

## Live mesh measurements

The strategy responder publishes FLOP, byte, row, deadline, J-lens, and J-oracle
coordinates through the generic workload interface. The node telemetry service retains
them in memory on port 8788. The existing whole-mesh observer on port 8787 combines the
laptop and Mini without importing a Xonotic schema. `joracle/demo.sh` checks or
kickstarts that observer; it does not launch another HTTP service or browser pane.

Open `http://127.0.0.1:8787` for both nodes' infrastructure activity.
`http://127.0.0.1:8787/latest.json` exposes scalar/array-length projections of nested
`workload.producers[].measures` objects, not the full numerical arrays. Start the two
application views from this directory with:

```
../bin/mesh-python -m solver.strat.joracle.server --run-dir solver/strat/runs/joint-live-20260904
```

Open `http://127.0.0.1:8795/j` for J-space and `http://127.0.0.1:8795/policy` for
policy optimization. Both use one cached reader, following new match directories.
The J views consume the complete source feature vector and actual intermediate/output
arrays from the [policy program](../design/POLICY-PROGRAM.md). Native response identities
connect issued view rates with later application acknowledgments and observed game
outcomes. Array serialization cadence is controlled by `--model-sample-every`.
Runtime process multiplicity, host identity, bridge state, telemetry age and reconnect
epochs remain separately reported coordinates.

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

Post-training release evaluation can run the release-rank matrix-fusion/MoE operator across both hosts without replacing the game or reporter services:

```sh
xonotic/solver/strat/joracle/evaluate-distributed.sh
```

For one real match at the previously exercised 128/341/8/top-2 shape, launch
the existing curriculum from the M5 checkout after synchronizing and building
committed `main` on both participants:

```sh
cd /Users/mdot/dox/mesh/xonotic
../bin/mesh-python -m solver.strat.curriculum \
  --server-host ms-mac-mini.local \
  --ssh-command 'ssh -o BatchMode=yes -o ConnectTimeout=8' \
  --remote-mesh-root /Users/mdot/mesh \
  --remote-python /Users/mdot/mesh/bin/mesh-python \
  --run-dir solver/strat/runs/encoder-migration-20260910 \
  --remote-run-root /tmp/mesh-encoder-migration-20260910 \
  --generate 1 --cycles 1 --study-repetitions 0 --heldout-fraction 0 \
  --maps dance --team-counts 2 --players-per-team 2 --cart-counts 1 \
  --skills 5 --perturbations baseline --off-policy-counts 0 --human-counts 0 \
  --policy-arms matrix_fusion --scale-rank 128 --scale-hidden 341 \
  --scale-experts 8 --scale-topk 2 --replay-batch 1 \
  --score-limit 30 --checkpoint-score-rate 1 \
  --peer-node 1 --strategy-node 0 \
  --distributed-scale --distributed-scale-operation block
```

The responder and its navigation/telemetry paths remain local to the M5;
the dedicated server and matrix worker run on the M4. Application modules load
directly from the respective committed checkouts, with matching revisions
recorded in the run metadata. Explicit `--responder-command` and
`--expert-command` override those local and remote commands respectively;
neither invokes an application snapshot deployment. Transfers performed by
this curriculum contain engine binaries, map assets, generated entity files,
and result artifacts, not application source.

The launcher needs the existing Python environment, the `dance` map assets,
the dedicated engine and payload build tools, and both existing RDMA bridges.
Each realized match's user directory contains its actual map BSP and
`gamemodes-payload.cfg`, including when entity measurements are reused or an
entity overlay was supplied. The existing artifact transfer therefore supplies
the peer with the exact map used for entity generation.
It runs one match, ending on its observed score outcome; `--duration` does not
impose a wall-clock timeout. This exercises actual policy inputs and generated
encoders. `measure.py matrix` measures a different numerical path and is not a
substitute for this application run.

The command resolves the Mini address and uses the curriculum's single lifecycle
owner. The [policy program](../design/POLICY-PROGRAM.md) defines the shared local/remote
mathematics, tensor exchange and parameter ownership. Whole-game deadlines, resume
continuity and behavioral benefit require actual operational measurements.

The [joint-learning runtime](../design/JOINT-POLICY-LEARNING.md) now runs both objective-defined
policies and honors `--human-counts` / `--human-client-command` in joint schedules.
Client templates substitute `{port}`, `{map}`, `{seed}`, `{match}`, `{directory}` and
`{client}`; use the game host's reachable address with `{port}` to follow each match.
Server-observed human rows remain value-learning data but are excluded from direct
PPO actor credit. The policy dashboard at `http://127.0.0.1:8795/policy` exposes
per-policy learning and observed-outcome measures. Full J matrices remain in the node's
latest record and the match's current `j-measures.<telemetry-basename>.npz` artifact; they are not
repeated in the interactive polling payload.

The remote game base defaults to `/Users/mdot/mesh-workloads/cartlane/Xonotic`;
`--remote-basedir` selects another existing installation. The remote engine comes
from `--remote-mesh-root`'s `xonotic/darkplaces-work/darkplaces-dedicated`, or
`--remote-engine`. Each participant builds its own committed checkout. Curriculum
never copies engine binaries, source trees, Python environments or base archives.
Only the selected map, generated entity data and payload configuration enter the
replaceable match userdir; synchronizing it removes obsolete prior match assets.

After recording result hashes and metrics, curriculum removes per-match BSP and
gamecode copies. It retains small logs, metadata, entities and measurements, plus
initial and current generated checkpoints per policy arm and the continuation
bundles they reference. Superseded checkpoint payloads, continuation bundles,
action archives and journals are deleted after their successor is recorded. Externally supplied
checkpoints remain owned by their source. The live userdir is deleted after the
game process exits. Interrupted runs may retain their final artifacts for the
operator to inspect; no cleanup touches the configured base installation.

`bin/mesh-application.py` activates a canonical Git checkout through a lightweight
`current` symlink. Remote deployment transfers committed Git objects, verifies
the revision, and builds both checkouts concurrently. It creates no snapshot
source generations or per-application Python runtime. The canonical checkout uses
the installed shared Python runtime through `bin/mesh-python`.

Curriculum retains the latest completed full J observation archive from its own
matches and removes its superseded full archive after the successor exists and
match results are recorded. Scalar telemetry, logs and match metrics remain.
A match that produces no new observation leaves the last usable archive intact;
archives belonging to another run or an externally configured viewer are untouched.
