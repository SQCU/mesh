# Xonotic engine-scale computation specification

This specification controls the engine-side causal working-set implementation. The
policy data flow is controlled separately by [`SPECIFICATION.md`](SPECIFICATION.md),
[`rl-training-spec.md`](rl-training-spec.md), and
[`ALGORITHM-CONTRACTS.md`](ALGORITHM-CONTRACTS.md). Implementation claims are in
[`claims/ENGINE-SCALE.md`](claims/ENGINE-SCALE.md).

## Purpose

The reference game must expose useful application parallelism as live participant
population grows. It must not manufacture a fixed matrix size, a fixed player count, or
an artificial accelerator workload to make a hardware crossover coincide with a chosen
demo. Connected players may join or depart at any time. The current roster is therefore
an observed row axis, not a configuration ceiling.

The engine retains ordinary Xonotic world semantics. Tracing, navigation mutation,
entity allocation, random draws, projectile creation, and combat interactions are
ordered world operations. A component may execute as a concurrent working set only when
its complete read and write relation is represented and members of one working set are
causally independent for the frame.

## Frame transaction

`StartFrame` invokes the world-aware decision stage exactly once for every current bot.
When a decision reaches the pure keyboard-controller transform, it deposits one row
containing the participant identity, current controller inputs, destination, current
time, and the already-realized random coordinate. It does not apply the transform
inline.

After all decisions deposit, the native engine performs one scatter/execute/gather
transaction:

1. Gather the deposited rows into a structure-of-arrays buffer.
2. Derive the causal-cell plan from the current actor extents, positions, frame duration,
   and maximum displacement.
3. Stable-sort rows into color waves and execute each independent working set through
   the vectorized controller kernel.
4. Gather the nine output coordinates back by stable participant identity at each
   working-set barrier.
5. Apply the ordinary shared physics and combat stages only after the controller
   transaction completes.

If an engine field or destination cannot be realized, the engine reports the missing
coordinate and the QuakeC implementation applies the same controller transform to the
still-pending row. This is an availability continuation with measured scalar mass, not a
different gameplay policy.

## Causal coloring

Let `h` be the largest live actor hull extent, `v` the configured maximum speed, and
`dt` the current frame duration. The represented two-sided write horizon is

```
d = h + 2 |v dt|
r = ceil(d / h)
```

Actors are assigned to spatial cells of extent `h`. Cell integer coordinates are colored
by their residues modulo `r + 1`. Actors occupying the same cell receive successive
color strata. The resulting waves preserve separation beyond the represented motion
horizon. The plan is rebuilt from the current roster each frame; no map, node, team,
player, or population class selects a smaller implementation.

The current kernel owns fourteen gathered coordinates and nine scattered coordinates
per deposited controller row. Its storage extent is the realized row count multiplied by
the declared structure-of-arrays width. Clang vectorization and interleaving operate over
each working set; buffer depth follows the live row population.

This causal model applies only to the controller transform whose writes are represented
by that row. A perception radius is not substituted for its write horizon. World-aware
decision code is not copied into a byte buffer and called transactional merely because
doing so would create more nominal work.

## Teams and population

Team identity is an integer relation supporting every identity from 1 through 256.
Adding teams, players, carts, or instruments adds rows. Generic team availability,
scoreboard state, policy state, and wire state use row collections or scalar identities;
they do not use a fixed-width team bitset. The DarkPlaces server-information byte encodes
256 clients as zero, and the paired client decodes zero as 256.

Spawn swizzling orders the current respawn population into localized waves over the
available spawn relation. Its epoch, cohort, ticket, lane, scheduled time, actual time,
and realized spot are ordinary server-state coordinates emitted to the strategy and
measurement interfaces. Longer respawn intervals trade simultaneous spawn occupancy for
temporal occupancy without reducing the participant or team relation.

Map fusion and cart-path construction supply large connected environments, localized
spawn pools, and cart-support surfaces. Those geometry interfaces are controlled by
[`FUSION-SPEC.md`](FUSION-SPEC.md) and [`NAV-SPEC.md`](NAV-SPEC.md); the tick loop does
not invent another spatial graph.

## Projectile causality

Projectile entities retain their ordinary finite propagation. Any future cell-local
projectile execution must derive reach from the live projectile swept volume for the
current frame. Hitscan interpolation and finite-charge volumetric attacks require their
own explicit server-rule and network contracts before they can enter this implementation;
they are not implicit properties of the controller batch.

## Measures and operating-point search

The engine publishes, per frame and cumulatively:

- live bot rows and deposited kernel rows;
- occupied cells, color waves, working sets, and peak rows per working set;
- causal-cell radius and cell extent;
- gathered and scattered coordinate mass;
- buffer bytes, barriers, scalar rows, and continuation rows;
- frame and controller elapsed time relative to the game deadline;
- current player, team, and cart populations.

The mesh telemetry layer joins those measures with host CPU/GPU residency, achieved
FLOP/s bounds, memory-byte/s bounds, renderer frame time, audio health, and live fabric
traffic. Operating-point search samples ranges of current player, team, and cart counts.
It selects from observed roofline distance and coexistence measures; it does not encode a
population target in the algorithm.

The desired study region reaches the Mini's measured FLOP roofline while using roughly
half of the MacBook's measured CPU/GPU memory bandwidth without violating the game,
renderer, or audio deadlines. These are measured coordinates with variance and support,
not prewritten workload sizes. Missing coordinates remain missing-measure mass and cause
the sampler to schedule observations rather than push population upward.

## Demonstrated relation to the mesh

Engine working sets and policy rows are distinct axes of the same live workload. The
reference demonstration is substantiated when increasing the runtime roster increases
measured engine working-set mass and strategy row mass, policy forward and optimization
work execute on distinct reachable nodes, and the whole-mesh reporter attributes their
FLOP, byte, deadline, and fabric coordinates to those nodes. Game behavior remains the
literal observable output: aggression, competitive pressure, robustness to controlled
rule perturbations, cart acquisition, cart preservation, and denial are empirical
measures, never a generic improvement label.
