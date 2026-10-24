# Whole-function routing and allocation

## Fabric

The collective layer owns a physical multigraph. A `mesh_cable` names one
reciprocal cable, its two rank endpoints, and its capacity. A `mesh_fabric`
contains all these cables, including parallel cables. It has no mapping from a
peer to one channel. The bootstrap exchanges every live endpoint, groups streams
by physical cable, and acquires one mutually available stream on each cable.
Another communicator occupying a cable is an explicit initialization failure;
initialization must not silently discard that cable. Every participant builds
the same ordered cable table. Local endpoint bindings index this table.

`mesh_fabric_stripe` lowers a logical crossing into `mesh_transfer` records,
each with an explicit cable index and buffer interval. Parallel capacities add.
The compiler apportions whole transport quanta by weighted largest remainder,
reserving one quantum per cable when the crossing has at least that many quanta.
Smaller crossings cannot use more cables than their number of quanta without
duplicating payload or changing the transport framing. There is no byte-size
threshold selecting a special single-cable implementation.

The native executor binds a transfer by cable index. It never resolves a peer
to a preferred channel. Both ends independently compile identical per-cable
orders. Overlapping receive/write ranges create dependencies; disjoint receive
ranges do not. Cable resources remain separate from node compute resources:
two cables incident on one node are not mutually exclusive.

## Interface

The NCCL calls remain the data-program interface. The collective compiler can
read the immutable physical fabric with `ncclMeshCommFabric`, install spanning
routes and positive per-cable weights with `ncclMeshCommRoutes`, and compile a
complete call sequence with `ncclMeshCommPlan`. Installing routes invalidates
the previous compiled sequence. The caller completes its previous function,
installs the next routes, and compiles the next sequence before starting it.
No route decision belongs inside a tensor operation or recurrent timestep.

`ncclMeshCommSample` exposes byte prefixes supplied, sent, and landed on each
local cable. Supplied minus sent is the ordinary published backlog. A relay may
already be eligible through another cable's landed prefix, before its ordinary
supplied prefix advances, so this sample can undercount relay backlog. It is not
a measurement of compute interference. Sampling a complete execution can measure peak queued
bytes and their time integral; endpoint counters alone cannot recover a queue
that grew and drained during the execution.

The feedback unit is a completed program evaluation, including all layers and
recurrent timesteps. Compute partitions, route weights, and write ordering
belong to one configuration. Finite differences must compare full evaluations
of nearby configurations with the same input workload, retaining their joint
observations. Per-operation timings may explain a result but cannot substitute
for its objective. Own-dependency traffic and relayed dependency traffic share
cable capacities. Compute slowdown from relaying is an empirical question.

## Review dimensions

The fabric, compiler, NCCL implementation, and bindings are collective code.
For C2 and C3, the former peer-to-channel map is below the explicit cable table
and physical transfer records: parallel and unequal-capacity links survive
lowering into execution. For C4, the previous-receive chain is below buffer-range
dependencies: independent ranges on different cables need no ordering edge.
C5 and C7 require a fixed compiled sequence during one evaluation and identical
configuration and evidence at every participant. M is unchanged by this routing
interface: its existing ordered stream entries and supplied/landed words remain
the transport primitive. D retains NCCL and registered buffers as its interface.

Correctness checks do not establish bandwidth or latency improvement. Those
claims require measured complete-function executions, payload comparison, and
per-cable evidence on the actual cluster.

## One-way writes

`mesh_paths` specifies simple paths as physical cable indices. `mesh_write_compile`
splits a typed operand among them. It emits one matching send and copy per path
hop, with no reduction or payload transformation at a
relay. Every rank declares `ncclMeshCallWrite` with the same source, destination,
and element count; each supplies a registered landing buffer of that extent.
Only the source reads its source buffer and only the destination's complete
landing buffer is an output. Other ranks' buffers are relay storage.

`ncclMeshCommWriteRoutes` installs the paths before `ncclMeshCommPlan` compiles
the function's calls. `ncclMeshWrite` executes that declaration. A relay's send
entry reads its receive's registered rows directly and uses the incoming
stream's landed prefix as its supplied word. The bridge can forward a page as
soon as it lands; it need not wait for a host round trip or for the whole range.
There is no chunk-size parameter or extra per-chunk publication in the interface.
The ordinary two-participant NCCL send/receive interface still requires a direct
connection; a multi-hop write declares its participating relays collectively.

## Buffer ownership and GPU payloads

Publication is not send completion. There are no waits on `sent`. The data program
keeps each source and relay lease alive and unmodified through its downstream
uses; independent layers can have independent leases. A complete-function
rendezvous in a test establishes safe reuse between evaluations without adding
per-send synchronization to the executor.

A GPU producer stores its published result through `mesh_step_write`, and a GPU
consumer reads NIC-written bytes through `mesh_step_read`. Ordinary cached stores
can be invisible to the NIC while a command buffer is running; ordinary loads
can retain old contents of a reused receive buffer. The collective's own payload
producers follow the same coherent-store rule. No staging copy is added to repair
an ordinary producer. The DNN example computes its second matrix product directly
into the published lease using the existing Metal tensor-operation pattern.

## Ahead of time

`routing.Fabric` preserves the physical multigraph and enumerates simple physical
paths. For a declared function's pairwise byte demands, `allocate` solves a
multi-commodity linear program: each commodity's path flows sum to its bytes,
and each directed physical cable's aggregate load is bounded by its capacity
times a common duration. A positive per-path quantum floor maintains path
coverage. The second solve minimizes byte-hops at the first solve's duration.
It uses [HiGHS through SciPy](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html).
The result is the fluid model's constrained optimum, not a claim that a real
program with dependency release times attains that duration. Integer transport
quanta, producer times, queues, and dispatch costs remain for measured feedback.

## Finite differences

`whole.FunctionFlow` evolves a continuous vector jointly representing compute
shares and physical path shares. For each allocation, a bounded logistic map
`x_i = low_i + (high_i - low_i) sigmoid(z_i - lambda)` chooses the single
multiplier that conserves its total. Routing weights stay continuous through
installation. Only realization into physical pages or indivisible compute tiles
rounds them; the latent vector retains changes smaller than a tile.

A fixed Hadamard sequence supplies joint directions. Four complete executions
at `z + c*d, z - c*d, z - c*d, z + c*d` measure the directional derivative
`(J_0 - J_1 - J_2 + J_3)/(4*c)`. This cancels additive linear drift in the measured
cost across that stencil. Every stencil unconditionally updates the vector with
bias-corrected first and second gradient moments and a decaying step size.
There is no control/trial comparison, acceptance threshold, rollback, or winner
selection. Best attained measurements are reporting statistics and do not steer
the state. Persisted state contains the vector, moments, and partial stencil.

The current cost is `J = log(T) + w * sum_e(A_e/C_e) / T^2`, where `T` is complete
function latency, `A_e` is integrated queued bytes on directed cable `e`, and
`C_e` is that cable's capacity in bytes/second. The congestion term is continuous
and dimensionless, with default weight `w = 1`. It prices own and relay traffic
sharing links without equating DMA with GPU arithmetic. Every observation names
the exact compiled configuration, workload, and completed function. Incorrect or
partial results are measurement failures, never samples of a slower route.

The finite-difference estimate remains subject to timing noise and tile/page
granularity. This is a measured local descent, not a global-optimality proof or a
promise that each individual evaluation will be faster than its predecessor.

## Measurements

`ncclMeshTraceStart/Stop` observes existing entry lists, eligibility words, and
sent/landed prefixes in a native thread. Relay eligibility comes from each entry's
actual custom word (normally the incoming landed prefix). For every outgoing
entry it records first eligibility through completed sending, bytes, queue area,
and lower/upper timestamps bracketing the observed transitions. Incoming entries
separately record first arrival and completed landing. The observer
neither changes those words nor adds a transport wait. Stop joins the observer
after the complete function's downstream consumers finish. Per-cable byte-prefix
deltas independently check the traffic totals. Poll gaps and incomplete traces
remain explicit; sampled queue integrals are estimates.

## Roofline

Let `W` be the union of each compiled range's first-eligible to last-landed
interval. It includes stalls within an unfinished transaction and excludes
compute-only gaps outside every such interval. Let
`C = sum_e C_e` include **all directed physical cables**, even those idle during
part of `W`. The ideal capacity during those windows is `C*|W|` bytes. With `V`
actual wire bytes (each hop and padded transport page counted once), report
attained wire bandwidth `V/|W|`, utilization `V/(C*|W|)`, unused capacity
`C*|W| - V`, and the all-links ideal duration `V/C`. Also report the tighter bound
`max_e V_e/C_e` for the measured routing. Useful delivered bytes `U` are reported
separately, giving `U/|W|` and hop/padding amplification `V/U`. Extra relay traffic
must not masquerade as extra useful throughput.

`roofline.py` matches outgoing and incoming entries by physical cable, direction,
and cumulative page range. Each pages-to-pages window starts with source
eligibility and ends at the destination's completed landing, rather than
substituting local send completion. It merges overlapping windows rather than
adding their durations and reports ingress and egress for each node separately.
Across nodes, causal TCP request/reply clock observations bound root-minus-local
offsets immediately before each complete evaluation. It expands and contracts
windows by those bounds and the polling brackets, retaining the resulting
latency and bandwidth bounds. Offset constancy is assumed only over that
evaluation. These pages-to-pages windows include clock uncertainty even in their
per-node summaries because their endpoints are on different nodes. A lower
duration bound of zero has no finite bandwidth upper bound; uncertainty is not
silently replaced by a point estimate.

Report best attained whole-function latency and best attained transaction-window
bandwidth with their evaluation IDs and corresponding other measurements. They
may belong to different evaluations. The cluster's advertised 80 Gbit/s per
direction on seven cables gives `C = 140 GB/s`; this is a physical ceiling, not
a demonstrated sustainable payload rate. Positive traffic on all fourteen
directions establishes coverage; the utilization above measures the remaining
gap to that ceiling.
