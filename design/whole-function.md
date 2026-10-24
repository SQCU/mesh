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

`whole.FunctionSearch` treats compute splits and route splits as coordinates of
one bounded integer product of simplices. Every measurement identifies the exact
configuration, workload, and completed function evaluation. The solver brackets
its finite-difference stencil with full baseline evaluations to remove linear
timing drift, forms a joint projected descent proposal, and compares it against
fresh control evaluations in control/trial/trial/control order. Rejected moves
shrink the trust region. Persisted state includes outstanding evaluations;
partial-operation observations and observations of another configuration fail.

Latency of the complete function is primary. Queue area supplies a secondary
direction and tie-break when latency differences are below the declared
resolution. Queue pressure also orders the stencil's routing coordinates. This
does not assume a bandwidth-only or a FLOPS-only machine and does not price DMA
relaying as a tensor computation. It is a local measured optimizer, not a proof
of global optimality on a nonlinear execution substrate.

## Measurements

Sample byte prefixes during the complete execution, retaining timestamps, peak
backlog, and integrated backlog per directed cable alongside end-to-end latency.
Retain before/after counters to distinguish useful traffic on every physical
cable from a nominal route weight. Report transport padding separately from
logical delivered bytes. An advertised capacity or positive weight is not a
measurement of saturation.
