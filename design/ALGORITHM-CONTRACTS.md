# Algorithm interface contracts

This document joins requirements already established in `SPECIFICATION.md`,
`NAV-SPEC.md`, `FUSION-SPEC.md`, `rl-training-spec.md`, and
`MATRIX-EXECUTION-SPEC.md`. Those specifications remain authoritative.

## Policy data flow

The engine emits literal participant, cart, event, cell, and outcome rows. The responder
constructs count-independent participant and instrument rows. Learned projections form
one embedding per participant and one embedding per instrument. A single joint
participant embedding produces a positive-semidefinite Gram matrix. That matrix mixes
the participant IR. Instrument allocation, a residual-feature Gram matrix, SwiGLU,
policy controls, W/L value probes, and dynamics probes consume the mixed IR.

Every policy matrix product crosses the owned matrix-execution boundary. The DPP
allocator composes its feature covariance, dimension-counted conjugate-gradient
products, and marginal contraction from that boundary. A mathematical library symbol is
not an execution contract.

The reward stream contains only two positive measures: loss of the previously projected
winner for W rows, and upward rank flips among nonwinning teams for L rows. Damage,
kills, pickups, contesting, and cart motion are behavioral outcome measures and never
enter reward targets.

The cart-game object separates its global formal value from its role projection. A
global nimber exists only when every reachable role has the same complete acyclic
options. Controlled partizan or cyclic positions retain a null global nimber. Each
team's controlled-cart portfolio is independently an impartial heap sum, so its exact
portfolio nimber, the unique projected role, denial succession, and loser ranks remain
defined projection coordinates. Runtime roles and reward events consume this same
object; no second XOR implementation exists beside the formal evaluator.

One selected assignment yields its literal instrument identity, target identity or
cell, gain, spawn control, and a travel commitment derived from actor-to-destination
walking distance on the canonical navigation object. Commitment is stock walking time
plus a learned nonnegative extension and has no map-size ceiling. QC consumes identity
without reconstructing strategy semantics.

Participant succession is keyed by stable participant identity. A join introduces a
new row. A departure has no successor and zero bootstrap only for that row. Neither event
terminates another participant's trajectory.

## Distributed scale control flow

Forward residual-scale work may execute on any mesh node. Training retains the same
placement: backward information and parameter updates cross the same interface or are
owned by the node executing the parameters. The RDMA link transports literal tensor
rows and structural framing only. Workload size is determined by current rows and
available mesh memory, not an application constant.

## Geometry data flow

Each stock BSP produces the stock playerbot navigation metric graph. Its weighted edges
carry one-dimensional path-length measure, divided exactly between their incident nodes,
and shortest-path distance produces the graph Voronoi decomposition. The same object
feeds cart-path planning, belief integration, and causal working-set construction. The
compiled solid-brush half-spaces define the separate continuous feasibility domain in
which cart curves are realized.

The reusable reconciliation kernel and its output measures are specified by
`GEOMETRY-RECONCILIATION.md`. Geometry identities are never discarded to manufacture
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
The native structure-of-arrays kernel gathers fourteen input coordinates per row and
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

Operating-point search samples player, team, and cart populations. It minimizes distance
over the coordinates present in each observation and records missing-coordinate support.
It separately schedules missing measurements. Missing data never becomes an infinite
distance or a command to increase population.
