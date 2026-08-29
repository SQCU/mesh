# xonotic on the sealed mesh

A Xonotic payload-mode dedicated server on node 0 (MBP) publishes one 16-column row
per client slot every `PLC_TICK` (0.1 s) across the RDMA fabric. Node 1 (mini) scores
the rows and returns one objective index per row. The engine takes a per-team majority
of those indices into `payload_mesh_objective[team]`, and `havocbot_goalrating_payload`
rates the corresponding payload node at `ratingscale * 2` for every bot on that team.

- engine bridge: `bridge/PORT.md`, `bridge/engine/mesh_ipc.c`, `bridge/qc/`
- game code: `payload/qcsrc/common/gamemodes/gamemode/payload/sv_payload_mesh.{qc,qh}`
- solver: `solver/worker.py`, `solver/xonwire.py`
- build tree (not in this repo): `~/dox/xonotic/build-engine`, `~/dox/xonotic/build-qc`

## What runs

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

## What does not run

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
