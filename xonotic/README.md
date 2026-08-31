# xonotic on the sealed mesh

A Xonotic payload-mode dedicated server publishes 40-column participant rows,
12-column cart rows, and 6-column perception events across the RDMA fabric. The mesh
responder returns an 8-column strategy assignment for every client row. Bot rows enact
it as additive havocbot navigation ratings; human rows retain the assignment as an
advisory signal.

## Online policy training

The real training environment is the dedicated server. Run the server with a sampled
map/roster/cart/controller configuration and run the strategy responder with training
enabled on the mesh peer:

```
cd ~/mesh/xonotic
python3 -m solver.strat.strat_responder --train --off-policy-players 2
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
python3 -m solver.strat.curriculum --generate 96 --seed 20260830 \
  --server-host game-node --remote-engine /opt/xonotic/darkplaces-dedicated \
  --remote-basedir /opt/xonotic/Xonotic \
  --maps runningmanctf,dance --team-counts 2,3,4,5 \
  --players-per-team 2,4,8 --cart-counts 1,2,3,4 \
  --skills 2,5,8 --perturbations baseline,fast,slow,volatile \
  --off-policy-counts 0,1,2,4 --human-counts 0 --heldout-fraction 0.2 \
  --duration 600 --run-dir solver/strat/runs/curriculum-20260830
```

The same command with `--dry-run` resolves and records the complete schedule and
commands without requiring Xonotic, MLX, or RDMA. JSON and JSONL manifests are also
accepted with `--manifest`. A JSON manifest may contain `defaults`, `matches`, and
`heldout`; each match accepts `map`, `bsp` or `entity_file`, `teams`, scalar or list
`players_per_team`, `carts`, `controllers`, `skill`, `duration`, `seed`,
`perturbation`, `server_cvars`, `server_args`, `client_commands`, and
`off_policy_players`. Held-out matches traverse the same `--train` responder path
with learning rate zero and never advance the training checkpoint lineage. Optional
`client_commands` are launched as argv without a shell for externally controlled participants; the
telemetry record reports the bot/human counts actually observed rather than treating
the requested controller mixture as evidence. Generated schedules can vary human
counts with `--human-counts`; `--human-client-command` launches one command per such
participant and expands `{port}`, `{map}`, `{seed}`, `{match}`, and `{client}` tokens.

Summarize one or more realized telemetry streams with:

```
python3 -m solver.strat.metrics solver/strat/runs/curriculum-20260830/*/telemetry.jsonl
```

The summary separates winner-retention and loser-acquisition trials and aggregates
importance weights, W/L losses, dynamics error, ensemble disagreement, local-control
singular values, credited horizon, controller class, and behavior-policy class.

## Live evaluation service

`solver.strat.viewer` is the runtime evaluator for a real game/responder pair. It reads
the responder's JSONL directly, queries both bridge processes, and verifies the local
dedicated-server process and remote responder process every two seconds:

```
cd ~/dox/mesh
python3 xonotic/solver/strat/viewer.py
```

With no arguments it uses the already configured `mesh-mini` SSH alias and selects
the newest realized `live.jsonl` or curriculum `telemetry.jsonl` on that peer. Pass
`--remote-host` or `--telemetry host:/absolute/path.jsonl` to inspect a different
reachable node or a particular run.

Open `http://127.0.0.1:8791`. The page displays cart identity/control/depth, team
resources, every selected instrument, off-policy participants, team focus, W-retention
versus L-promotion phase space, all online losses, and the shape/range/finiteness of
`x`, `beta`, `z`, relation, hierarchy, weights, 128-dimensional IR, Gram matrix,
scores, and both value estimates. It derives violations from those realized values;
there is no simulated input path in the viewer.

Shape, numeric range, finiteness, and W/L values are recorded on every response. The
full operand matrices are recorded every ten responses by default so JSON encoding
does not become part of every control decision; `--model-sample-every` changes that
cadence, including `1` for every response.

The green `LIVE CROSS-RDMA` state requires all of the following at once: exactly one
running Xonotic dedicated server, a running responder on the named peer, telemetry
newer than five seconds, and both bridge nodes reporting `up` with no status-command failure. Multiple
game producers are a violation because the current wire format has no channel namespace. A
completed run remains inspectable but is labeled `RUNTIME DEGRADED`, so an old JSONL
cannot impersonate current computation.

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

- engine bridge: `bridge/PORT.md`, `bridge/engine/mesh_ipc.c`, `bridge/qc/`
- game code: `payload/qcsrc/common/gamemodes/gamemode/payload/sv_payload_strategy_io.{qc,qh}`
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

## Historical run limitations

- **The MoE path never executes.** `mlx` does not import in `~/.venv-mesh` on the mini,
  so `scores_mlx` (8 experts, FF 2048) is dead and every number above comes from
  `scores_np`, a 16 -> 64 -> 5 `tanh` MLP in numpy. This demonstrates the loop, not
  the value of offloading it; the far side is microseconds of CPU.
- **The policy names are not semantics.** `nearest` is `argmax` and `inverted` is
  `argmin` over a random-seeded projection. Neither computes a distance to anything.
  They are a valid A/B lever and nothing more.
- **Loss handling is untested under loss.** `bad` was 0 and every request drew a
  response in all five runs. The §5 posture was exercised only by total solver absence,
  never by an intermittent dropped or reordered slot.
- **One map, no humans.** Only `runningmanctf` with 5 payload nodes and bot-only
  play. Without `+bot_join_empty 1` a headless server spawns zero bots.
- **Sim rate is load-dependent.** One 150 s match produced 920 ticks instead of ~1400;
  the 10 Hz publish rate holds only while the box keeps up.
- **Plan staleness is not measured.** Request and response counts match 1:1, so at
  most one plan is in flight, but the age of the applied plan was never timed.

## Scaling to 48 bots on 5 teams

The QC glue carries no row cap of its own: `sv_payload_mesh.qc` gathers, publishes
and scatters `maxclients` rows every tick, and the engine chunks a block across
slots (63 request rows per slot at width 16, 126 response rows at width 8, up to
64 chunks = 4032 rows). `worker.py` reassembles by chunk mask with `--maxrows 4032`.
The row count is set entirely by `maxplayers` at launch. At `maxplayers 64` a
request is 2 slots and a response 1 slot per 0.1 s tick.

`tools/mkentfile.py <bsp> <out.ent> [teams]` now takes a team count of 2..5. Goals
land on distinct track nodes (red plcn0, blue plcn4, yellow plcn1, pink plcn3,
green plcn2); at 5 teams a start node `plcs` is spliced into the middle of the
track so no goal sits inside the cart's 64-unit capture radius at round start
(measured minimum start-to-goal arclength on runningmanctf: 1977 units). Teams
3..5 have no `info_player_teamN` spawns on runningmanctf and fall back to the 26
`info_player_deathmatch` spawns via `server/spawnpoints.qc`; teams 1..2 keep their
8 team spawns each. Five `plc_goal`s alone drive `payload_teams` to mask 31, no
override cvar needed.

```
mkdir -p /tmp/xonrun48/data/maps
unzip -p ~/dox/xonotic/Xonotic/data/xonotic-20230620-maps.pk3 maps/runningmanctf.bsp \
  > /tmp/xonrun48/runningmanctf.bsp
python3 ~/dox/mesh/xonotic/payload/tools/mkentfile.py \
  /tmp/xonrun48/runningmanctf.bsp /tmp/xonrun48/data/maps/runningmanctf.ent 5
~/dox/xonotic/build-engine/darkplaces-dedicated -xonotic \
  -basedir ~/dox/xonotic/Xonotic -userdir /tmp/xonrun48 \
  +developer 0 +sv_public 0 +port 26012 +sv_autopause 0 \
  +g_payload 1 +g_payload_round_timelimit 600 +timelimit 30 \
  +maxplayers 64 +bot_join_empty 1 +bot_number 48 +skill 5 +g_warmup 0 \
  +map runningmanctf
```

The loose userdir `.ent` overrides the pk3's 2-team copy, so the shared pk3 keeps
the verified A/B demo unchanged. Expected in the log: `payload: teams mask 31` and
48 bots split roughly 9-10 per team by TeamBalance.

Engine headroom: `MAX_SCOREBOARD` 255 and `MAX_EDICTS` 32768 clear 64 players with
room; `maxplayers` is clamped only at 255. The honest ceiling is CPU, not a
constant: sim rate already droops under load at 8 bots (above), and havocbot cost
grows linearly in bots times goal-stack size. 48 at skill 5 is configured and
compile-verified here, not throughput-verified; if the tick rate sags, `skill` and
`bot_number` are the levers, not the bridge.
