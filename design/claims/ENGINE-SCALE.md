# Engine-scale implementation claims

Controlling specifications: [`../xonotic-bot-compute.md`](../xonotic-bot-compute.md),
[`../rl-training-spec.md`](../rl-training-spec.md), and
[`../ALGORITHM-CONTRACTS.md`](../ALGORITHM-CONTRACTS.md).

Implementation surfaces:

- `xonotic/darkplaces-work/bot_batch.c` owns player-state scatter, working-set
  execution, and deterministic controller-output gathering.
- `xonotic/darkplaces-work/bot_batch_core.h` owns the vectorizable controller kernel.
- `xonotic/qcsrc/server/main.qc` and `xonotic/qcsrc/server/bot/default/havocbot/havocbot.qc`
  own the world-aware decision stage and frame barrier.
- `xonotic/qcsrc/common/teams.qh` and `xonotic/qcsrc/server/teamplay.qc` own generic
  256-team representation.
- `xonotic/solver/strat/capacity.py` reads the engine, team, and cart architectural
  capacities used by runtime and measurement launchers.

`StartFrame` invokes each world-aware `bot_think` exactly once under ordinary QuakeC
semantics. That stage may advance RNG state, trace geometry, mutate navigation entities,
or allocate and retire personal waypoints; it is deliberately not represented as a
transaction-pure vector kernel. While that stage runs, each due keyboard controller
deposits one complete row instead of applying it. The native plan gathers fourteen input
coordinates per deposited row, partitions the rows into spatial color waves, executes the
pure structure-of-arrays transform over each working set, and stable-gathers nine output
coordinates per row before shared player physics and combat advance.

Color separation is derived from the largest live actor hull and the two-sided distance
actors can move during the current frame. Perception radius is not a controller-output
write horizon, so a map-scale sight radius cannot reduce every working set to one row.
World-aware bot decisions remain ordered; only the deposited player-controller transform
claims synchronous wave semantics.

The claims are substantiated only by per-stage row, input-coordinate, output-coordinate,
buffer-byte, working-set, barrier, team-support, and deadline measures. Payload-only team
arithmetic does not substantiate them. `[VCELLMERGE]` reports the input-coordinate,
output-coordinate, and buffer-byte measures for the last frame; `[BOTWARP]`, `[VCELL]`,
and `[BOTQC]` report row and scalar-fallback measures.

A live 16-bot `runningmanctf` frame population produced 16 occupied cells, 12 working
sets, a three-row peak working set, a 69-unit cell extent, and a two-cell causal radius.
After controller staging became active, 13 rows entered the structure-of-arrays kernel;
the cumulative staged mass reached 392 rows while scalar and fallback mass both remained
zero. The earlier multi-megabyte VM snapshot measurement is superseded because it omitted
engine-owned RNG, trace, allocation, and world-link state and therefore did not describe
a valid transaction. A release measurement of the deposited controller rows supersedes
its byte and coordinate values.

Generic availability is a 256-row collection owned by the balance object. Mutators add
team identities directly, scoring consumes cardinality or that collection, and the
client receives the exact identity relation. Packed bitsets remain only inside named
legacy game modes whose authored rules have fixed small team sets; no generic team API
returns, accepts, or derives availability through such a bitset.
Study launchers set the server connection capacity from the engine declaration rather
than from the initial bot population. Initial players therefore select one observed
operating coordinate without withholding later joins.
