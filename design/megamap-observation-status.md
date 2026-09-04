# Megamap and observation measures

This document records the current interfaces connecting fused maps, live server
perception, the strategy composer, and the J measures. It contains no release decision.

## Map geometry

`xonotic/payload/tools/mapfuse.py` assembles stock maps into one world-space artifact.
The assembled BSP, entity set, waypoint graph, cart routes, spawn placements, and join
geometry are the runtime inputs. `negspace.py` supplies the shared solid, trace,
clearance, and standability operators over that assembled world. Cart and spawn
measurements retain their source map, route, support surface, and assembled position.

Cart realization is per requested cart. Navigation and negative-space constructions
are atoms in one candidate measure, and the realized set minimizes path-extent,
rider-support, and origin-collision residual mass per cart. Minimum path extent is
derived from the larger of the configured push and capture radii; cart-origin spacing
is derived from the runtime cart hull. A zero-residual navigation construction does
not run the fallback solver.

Neutral spawn placement measures clearance from each cart's initial push volume. It
does not exclude the cart's entire future path: a player may legitimately spawn near
a later segment before the cart reaches it. Path proximity remains a separate
measurement. Every team draws from the same neutral spawn measure, so a finite route
from its support to a cart is also a route available to every team. Map measurement
Schema 11 records the origin-clearance measure and its residual mass, the
negative-space interface schema, and the masses of compiled brush and patch-triangle
collision atoms explicitly.

Map coverage is accumulated from the entity realization used by each actual match.
There is no eager local catalog rebuild before the supervisor starts. Named-map mass,
observed-map mass, and their incidence remain distinct, so a map not yet realized is
missing mass rather than a generated zero or a reason to stop the schedule.

The runtime map graph reaches the responder through `CELL_LINK` event rows. Every link
contains both cell identities and the engine-supplied traversal length. The responder
also retains observed player transitions until waypoint links arrive. V-cell geometry
is the deterministic center of the same 256-unit lattice used by the server, so player
motion cannot move a cell centroid. A monotone topology revision changes whenever a
cell, transition, waypoint link, or shorter link length appears; cached segmentation is
recomputed at that revision.

## Observation interface

The server owns visibility. It emits observations only after its PVS, line-of-sight,
and two-cell causal-reach computation. Python does not reproduce or second-guess that
gate.

The event schema is a typed row:

```
kind, time, observer, team, subject,
cell_x, cell_y, target_cell_x, target_cell_y,
position_x, position_y, position_z,
respawn_time, health, link_length, amount, response_seq
```

The team buffer retains the newest current observation for each distinct subject and
subject class. Distinct items and rivals remain distinct rows even when they occupy the
same V-cell. The eleven slot coordinates are copied literally from the event row: four
kind coordinates, position XYZ, respawn time, health, link length, and amount.

RHO contracts each entity row independently toward the uninformative zero measure.
GIGI reads the entity row's V-cell and repeats that cell's bounded graph-distance weight
for every entity located there. PHIL projects each row once. The composer integrates
the weighted projected rows once and applies parameter-free RMS normalization before
QUINN. Thus entity population changes the direction and composition of belief without
making its magnitude an accidental function of how many entities share a cell.

The live call chain is:

```
server event rows
  -> LiveBelief.ingest
  -> ObservationBuffer
  -> build_observation_slots
  -> temporal_contraction
  -> VCellMap.spatial_mask
  -> strategy: RMSNorm(GIGI @ PHIL(slots))
  -> QUINN
```

`featurize.py` owns V-cell segmentation, slot transduction, temporal contraction, and
the spatial mask. `live_belief.py` owns stream state and calls those operators.
`strategy.py` owns the single learned integration. No second feature implementation is
present on the live path.

## J measures

The source state machine exposes the complete participant and cart rows before
featurization, all eleven composer inputs, selected native-width J, the literal action
response, and later authoritative rows and outcome counters. A source atom is retained
by `(stream epoch, response sequence, participant edict)`.

The J-lens is the empirical joint measure of exact source features and selected J.
The J-oracle is the response-sequence pushforward from authoritative source state, J,
policy intervention, and controls to delivery-, route-, goal-, touch-, behavior-, and
event-owned successor atoms. Continuous and bitset coordinates produce literal
differences; categorical coordinates produce source/target atoms. Missing joins remain
missing mass. Affine projections and covariances are derived measures over those exact
atoms and never replace a server transition.

## Runtime measurement still required

Local compilation establishes the interface and shape closure. Release evidence still
comes from matches over fused maps and reports the observed participant, team, cart,
entity-slot, V-cell, route, spawn-wave, J, outcome, node, host, FLOP, byte, and deadline
measures. Two-host claims require simultaneous source atoms from both hosts. When an
edge is absent, its node/host and distributed-deadline coordinates remain null rather
than becoming zero or a local surrogate.
