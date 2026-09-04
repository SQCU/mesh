# Whole-mesh page-table and policy-compute instrument

This specification controls the macroscopic demonstration of the mesh. The policy
details are controlled by [`rl-training-spec.md`](rl-training-spec.md), matrix
execution by [`MATRIX-EXECUTION-SPEC.md`](MATRIX-EXECUTION-SPEC.md), topology by
[`ACCESS-TOPOLOGY-SPEC.md`](ACCESS-TOPOLOGY-SPEC.md), and the joined flows by
[`ALGORITHM-CONTRACTS.md`](ALGORITHM-CONTRACTS.md).

## Demonstration

The mesh is one reachable compute and memory fabric whose capacity, allocation,
traffic, arithmetic, memory motion, deadline pressure, and application work can be
observed without knowing what the workload means. Xonotic strategy optimization and a
large OCR run use the same telemetry interface. Workload producers name dimensions and
operation envelopes; the mesh reporter stores no game rule, document type, policy arm,
or task-specific limit.

Every reachable leased node contributes its own sample to the whole-fabric view.
Inventory capacity remains visible separately from the live aggregate. A LAN, direct
fabric, or routed path may disappear and later return without changing node identity.
The observer reconnects to the node's in-memory sequenced ring and renews membership
from the new sample.

The page-table phase space is the exact allocation simplex:

- `FREE` is unowned page capacity;
- `RECV` is receive ownership;
- `SEND` is submitted or in-flight ownership;
- `APP` is application ownership.

The three rendered spatial coordinates are a projection of all four barycentric
coordinates. Each node has a distinct diegetic anchor and magnified phase trajectory.
Topology edges join live anchors. Traffic particles appear only from measured bridge
counters, so reachability does not invent transfer.

Capacity halos and the table report characterized FP32 and memory-bandwidth capacity,
GPU and CPU residency, operation-derived achieved-FLOP bounds, hardware and
workload-derived memory-byte bounds, variance, observation mass, and missing-coordinate
mass. The calculations run on telemetry cadence; animation only tweens the last and next
states under a frame budget and stops while hidden.

## Workload envelope

A producer publishes elapsed time, deadline, literal row dimensions, operation labels,
and lower and upper counts for FP32 operations and moved bytes. Exact counts form a
point interval. Partially known algorithms form a numerical bracket. An unspecified
coordinate has missing support, not a zero estimate and not pressure toward a larger
workload.

Machine observation supplies an independent bound. Characterized CPU capacity times
live cluster residency and characterized GPU capacity times GPU residency bound
concurrent arithmetic. IOReport/AMCC/AGX residency histograms bound whole-machine memory
motion. A root sampler adds frequency, power, and thermal coordinates to the same
in-memory stream; it does not own a second cache or service.

The canonical services are one telemetry provider on port 8788 per node and one
whole-fabric observer on port 8787. System and user launch domains realize the same
interfaces. Handoff measures the listening socket and response before the prior
provider exits, and restores the reachable provider when handoff does not complete.

## Hardware-derived operating point

Player, team, and cart counts are coordinates, not constants hidden in the algorithm.
The runtime may add or remove players while a server continues. The operating-profile
search changes the live player population, samples the whole-mesh telemetry ring, and
records every `(players, teams, carts)` point separately.

The target combines:

- a roofline interval containing saturation on every producer node;
- approximately half of the MacBook's measured memory-bandwidth capacity;
- the live strategy deadline;
- a local-only counterfactual whose lower time bound reaches that deadline;
- responder and expert roles observed on distinct live hosts.

Only observed coordinates contribute numerical distance. Missing coordinates retain
their own mass and cause further acquisition at the same population, not an increase
toward the engine ceiling. Search boundaries and capacity exhaustion are measures.
After a center is observed, team and cart schedules take below, center, and above
coordinates from the measured point while preserving the measured total player
population. No relationship between client count and hardware saturation is prescribed.

## Policy compute placed on the fabric

The matrix-fusion policy contains three exact Gram matrices:

1. The participant Gram matrix is the inner-product matrix of the direct-sum rival and
   same-team feature construction.
2. The residual-feature Gram matrix is `residual.T residual / physical_rows`.
3. The DPP feature Gram matrix is the RMS-normalized instrument-row inner-product
   matrix.

The policy also performs participant/instrument products, SwiGLU, action-linear dynamics,
and routed expert products. Repository-owned Metal kernels implement `AB`, `AᵀB`, and
`ABᵀ`, including reverse products. Shapes determine dispatch extent. DPP marginal
inclusion uses feature-side covariance and dimension-counted conjugate gradients; it
does not build an instrument-sized inverse.

The responder owns participant fusion, DPP, policy/value/dynamics heads, action
sampling, and local parameter updates. The expert host owns residual projection,
routing, expert products, residual-feature Gram mixing, reverse input products,
scale-parameter gradient accumulation, and one scale optimizer update per learner
batch. Forward, cotangent, accumulated-gradient, update, checkpoint, FLOP, byte, and
deadline measures identify both hosts for the same batch.

The RDMA and local relay layers transduce literal float tensor rows plus structural
framing. A 64-bit value extent determines page count. Shared page credit and socket
capacity determine how many pages can be submitted in one wave; neither determines the
tensor extent or later workload size.

## Xonotic realization

The reference game supplies a live, controllable state machine with many teams, many
carts, runtime joins and departures, and exact successor state. Stock maps plus a
generated bridge are fused into one connected world. Cart curves are constructed inside
continuous swept-volume and rider-support feasibility, and player belief, commitment
distance, cart planning, and V-cell structure share the stock-navigation/Voronoi
realization.

The strategy actuator preserves push, suppress, contest, rival, cell, spawn, and idle
instrument identities through the server. Policy optimization receives only sparse W/L
transition rewards. Aggression, robustness, competition pressure, survival, objective
conversion, and perturbation response are separate realized behavior measures. A study
compares mirrored interventions with their observation and missing masses; it does not
label a self-play policy “improved.”

The game is one demonstrator, not a special telemetry mode. The macroscopic result is a
live account of where pages, bytes, arithmetic, time, and optimization reside across the
entire reachable mesh, joined to observable consequences in an independently controlled
workload.
