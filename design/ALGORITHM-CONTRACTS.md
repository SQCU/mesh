# Algorithm interface contracts

This document joins requirements already established in `SPECIFICATION.md`,
`NAV-SPEC.md`, `FUSION-SPEC.md`, `rl-training-spec.md`, and
`MATRIX-EXECUTION-SPEC.md`. Those specifications remain authoritative.

## Policy data flow

[POLICY-PROGRAM.md](POLICY-PROGRAM.md) is the single current policy program:
complete native source rows, learned projections and mixing, one common final
representation for policy and value outputs, sampled rates, private-view
integration and the optimization objective. The detailed policy manifests formerly
repeated here are superseded by that page. Primary requirements remain in
[SPECIFICATION.md](SPECIFICATION.md).

The strategy reward stream contains minus one for loss of the previously projected
winner on W rows, and plus one for upward rank flips among nonwinning teams on L rows. Damage,
kills, pickups, contesting, and cart motion are behavioral outcome measures and never
enter reward targets.

The [literal cart game](CART-GAME-CONTRACT.md) integrates held checkpoints into
monotone score. The unique earliest score-threshold crossing under frozen cart control
defines the projected winner. Explicit checkpoint denial recomputes its succession;
first-passage ordering defines loser ranks. Runtime roles and reward events consume
this same object. No portfolio-XOR, terminal delivery or reversible-score substitute
defines another game underneath those labels.

Each response carries source identity, full per-word forcing, residual,
integration duration and relaxation time. The engine applies the response to
that bot's private native state view. Shared game state is not mutated by the
residual adapter. Applied response sequences attribute observed outcomes to the
actual sampled forcing and behavior-policy density.

Participant succession is keyed by stable participant identity. A join introduces a
new row. A departure has no successor and zero bootstrap only for that row. Neither event
terminates another participant's trajectory.

## Distributed scale control flow

Policy tensor placement, full operand derivatives and one owning optimization
state are described in [POLICY-PROGRAM.md](POLICY-PROGRAM.md). The matrix execution
boundary is documented in [MATRIX-EXECUTION-SPEC.md](MATRIX-EXECUTION-SPEC.md).
Placement transports literal tensors and structural framing; it does not create
another learned policy or freeze parameters by host.

## Geometry data flow

Each stock BSP produces the stock playerbot navigation metric graph. Its weighted edges
carry one-dimensional path-length measure, divided exactly between their incident nodes,
and shortest-path distance produces the graph Voronoi decomposition. The same object
feeds cart-path planning, belief integration, and causal working-set construction. The
compiled solid-brush half-spaces define the separate continuous feasibility domain in
which cart curves are realized.

The collision numerics and standalone reconciliation kernel are specified by
`GEOMETRY-RECONCILIATION.md`. The restored cart network/span planner and its simpler
bending-energy curve fit are documented in
[`CARTPATHS.md`](../xonotic/payload/CARTPATHS.md). Geometry identities are never discarded to manufacture
feasibility. Source coordinates are projected, incidence is transduced through the same
map, and displacement moments remain observable.

Cart-path optimization ranges only over curves whose swept cart volume is collision-free
and whose push surface remains within activation distance of continuously walkable floor.
Those constraints define the construction domain. Candidate search, policy exploration,
and measurement may sample candidates from that domain. Sampling is not a substitute for
the domain representation: no finite set of collision probes and no post-hoc
guess-and-check repair may establish whether an emitted curve is feasible. Continuous
half-space interval coverage establishes membership for every curve segment.

Map fusion realizes every requested stock and bridge source. Source failures remain
unfinished work and cannot shrink the realized request or produce a successful empty
bundle. Join prominence, connectivity, and cart-traversable aperture measures derive
from the realized navigation graph.

## Engine scale control flow

Every frame executes each world-aware QuakeC bot decision exactly once, depositing each
due player-controller row rather than applying its pure keyboard transform inline.
Spatial causal horizons derived from live actor hulls and frame motion partition those
deposited rows into independent working sets.
The native structure-of-arrays kernel gathers nineteen input coordinates per row through that bot's state view and
stable-gathers nine output coordinates per row at the working-set barriers before shared
physics and combat advance. RNG, trace, entity allocation, and world-link effects remain
in the ordered QuakeC stage and are never mislabeled as a byte-copy transaction.

Teams are integer row identities from 1 through 256. Generic operations use row sets or
collections, never a fixed-width integer bitset. Participant capacity is independent of
team count.

## Telemetry control flow

Every node publishes an in-memory leased stream of capacity, achieved FLOP/s bounds,
memory-byte/s bounds, fabric counters, workload rows, deadlines, and estimator moments.
The reporter aggregates live utilization over the currently reachable leased node set.
Inventory capacity and stale history remain separate measures.

Each roster node supplies a stable identity and any number of access aliases. Direct HTTP and
SSH-forwarded HTTP consume the identical sequenced ring protocol. Successful samples renew the
node lease independently of discovery address lifetime, so a LAN or fabric partition changes the
transport but does not invent a new node or erase its prior sequence.

Operating-point search for capacity experiments samples player, team, and cart populations. It minimizes distance
over the coordinates present in each observation and records missing-coordinate support.
It separately schedules missing measurements. Missing data never becomes an infinite
distance or a command to increase population. Policy learning instead retains its requested
independent team/cart/player axes and fixed per-match roster; monitoring does not rewrite them.
