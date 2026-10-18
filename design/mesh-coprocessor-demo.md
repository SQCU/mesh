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

## Policy compute placed on the fabric

The matrix-fusion policy projects complete native observation, cart, team, event,
state/residual and navigation rows before applying learned conditioning or spatial
integration. Each source event is embedded once. Sparse navigable V-cell neighborhoods
mix those projected observations and geometry into the relevant observer/page rows;
event contributions contract with age after embedding.

The interacting observation/page/cart/team rows then receive learned global and
same-team Gram actions. Their exact factorization, such as `R (Rᵀ V)`, retains
destination-specific outputs. A later row-local routed block uses top-k sigmoid
affinities, selected-route normalization and SwiGLU experts with an ordinary
load-balancing auxiliary loss. It has no shared expert. There is no residual
feature-Gram/probe broadcast or DPP instrument allocation on this live path.

Repository-owned paged Metal kernels implement `AB`, `AᵀB`, `ABᵀ` and their reverse
products. Prepared physical capacities determine dispatch extent; active row counts,
addresses, masks and expert assignments are tensor data. New capacities still require
preparation, and persistent allocation-free intermediate storage remains an explicit
execution obligation rather than a consequence of fixed kernel templates.

Each learned policy has one parameter/optimizer/checkpoint lineage. The learner
(`xonotic/solver/strat`, restored on torch; xonotic/README.md "The policy learner") is
one program on every rank of a world on the library: rank 0 serves the game and runs
the responder, every rank runs each model call it announces, and the row-Gram
cross-contraction and the routed scale experts are divided among the ranks by the
placement's parts (parts [0, 1] put the whole cross on the Mini, as on 09-06), their
operand and parameter cotangents derived by autograd through Megatron's mappings. A
rank updates only what it holds (rank 0 the replicated parameters, which every rank
then reads from it; each rank its own expert columns): one optimizer over one
parameter set, not a worker's own. Forward, cotangent, update, checkpoint, FLOP, byte and
deadline measures must identify the actual placement for the same batch. Historical
two-host runs are evidence for their recorded architecture, not automatic validation
of a subsequently changed tensor program.

The RDMA and local relay layers transduce literal float tensor rows plus structural
framing. A 64-bit value extent determines page count. Shared page credit and socket
capacity determine how many pages can be submitted in one wave; neither determines the
tensor extent or later workload size.

## Xonotic realization

The reference game supplies a live, controllable state machine with many teams, many
carts, runtime joins and departures, and exact successor state. Stock maps plus a
generated bridge are fused into one connected world. Cart curves are constructed inside
continuous swept-volume and rider-support feasibility, and player observation
integration, cart planning and V-cell structure share the stock-navigation/Voronoi
realization.

The policy emits a rate mean and Gaussian scale for every witnessed Havocbot state
word. Sampled rates drive exact exponential relaxation of residuals on the bot's
private view of game state; they do not mutate shared engine state or select a
hand-written vocabulary of interventions. Policy optimization receives sparse W/L
transition rewards and the explicitly defined regularizers and balancing loss.
Aggression, robustness, competition pressure, survival, objective
conversion, and perturbation response are separate realized behavior measures. A study
compares mirrored interventions with their observation and missing masses; it does not
label a self-play policy “improved.”

The solver (`xonotic/solver/worker.py`, restored from ef6adb10^; xonotic/README.md "The
solver") is this demonstration's measured operating point: the window of R = 4,096 rows
lifted to 2,048 residual columns, eight experts at a mean of 512 rows each
(T·k/E ≥ 512) and the window's Gram, about 104 GFLOP and 269 MB of resident weights a
solve, its experts and Gram divided among the members by column shares the Balancer
moves on the ticks' own times. Its reproduction target is the Mini alone at 24 ms a
solve (4.29 TF/s, 4905622f) and a 26.6 ms full-scale tick (ac3b543b); on the pair each
member's share and the tick are read against it.

The game is one demonstrator, not a special telemetry mode. The macroscopic result is a
live account of where pages, bytes, arithmetic, time, and optimization reside across the
entire reachable mesh, joined to observable consequences in an independently controlled
workload.
