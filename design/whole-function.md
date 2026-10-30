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

`mesh_routed_compile` lowers the complete typed collective graph through
`mesh_write_compile` into `mesh_transfer` records, each with an explicit cable
index and buffer interval. Parallel capacities add.
The compiler apportions payload using the selected path weights and preserves
the total exactly. It does not reserve a mandatory page or quantum for every
cable: a forced floor changes the solver's vector and can introduce unnecessary
relay work. Exact byte spans can be smaller than allocation pages or hardware
credit frames. Packet and datatype constraints are explicit realization inputs;
all-link use is assessed across a complete function's transactions, rather than
duplicating a tiny operand to manufacture cable activity. There is no byte-size
threshold selecting a separate single-cable implementation.

The native executor binds a transfer by cable index. It never resolves a peer
to a preferred channel. Both ends independently compile identical per-cable
orders. Overlapping receive/write ranges create dependencies; disjoint receive
ranges do not. Cable resources remain separate from node compute resources:
two cables incident on one node are not mutually exclusive.

The retained 2026-10-24 morning microprofile configuration had 4/1/2 parallel cables between the
Studio pair/first Studio and Mini/second Studio and Mini, giving ingress
degrees 5/6/3. Its wire identities and advertised capacities are recorded in
`metal-microbench/output_data/review-20261024/microprofile/fixed.json`.
Read-only discovery at 12:57:27 PDT found six paired cables, 4/1/1, with
degrees 5/5/2: second Studio port 2 was `STOPPED` and Mini port 1 `PAIRING`.
The [dated snapshot](../../../metal-microbench/output_data/review-20261024/interface-refactor/topology-current.json)
retains those observations; the [bridge log](../../../metal-microbench/output_data/review-20261024/interface-refactor/bridge-tail-rank2.log)
records the earlier port termination without establishing its cause.
Thus its active advertised aggregate wire ceiling was 120 GB/s, versus the
morning's 140 GB/s. Neither snapshot is a constant in the solver or interface.
Discovery is authoritative for each realization; retain its actual cable
identities and statuses with the execution. An unavailable cable is not a
solver-imposed restriction, and it must not be counted as an active resource.

## Interface

The NCCL calls remain the data-program interface. The collective compiler can
read the immutable physical fabric with `ncclMeshCommFabric`, install complete
physical paths and their weights with `ncclMeshCommWriteRoutes`, and compile a
complete call sequence with `ncclMeshCommPlan`. Installing routes invalidates
the previous compiled sequence. The caller completes its previous function,
installs the next routes, and compiles the next sequence before starting it.
No route decision belongs inside a tensor operation or recurrent timestep.

All collective front ends must lower into the same physical fragment, relay,
byte-range and lifetime representation. A solver with a different objective
retains the same placement and routing capabilities. A peer-sized publication
array, a single-cable fused form, or an operation that excludes declared relays
is an interface defect; preserving it is not a permitted bandwidth tradeoff.
Physical constraints such as registration boundaries belong in realization, with the
attainable bound reported. They do not justify discarding representable routes.
The [client coverage manifest](collective-client-coverage.md) records the NumPy,
Program, C/Objective-C, Swift, LiteRT and example boundaries, their allocation
and membership constraints, and which adapter tests have actually run.

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

Performance is the complete function's elapsed time and useful output rate on
the actual cluster. Extra transfer accounting is not required to report that pair.

## One-way writes

`mesh_paths` specifies simple paths as physical cable indices. `mesh_write_compile`
splits a typed operand among them. It emits one matching send and copy per path
hop, with no reduction or payload transformation at a
relay. Every rank declares `ncclMeshCallWrite` with the same source, destination,
and element count; each supplies a registered landing buffer of that extent.
Only the source reads its source buffer and only the destination's complete
landing buffer is an output. Other ranks' buffers are relay storage.

`ncclMeshCommWriteRoutes` installs paths in the communicator's configuration.
`ncclMeshWritePrepare`, `ncclMeshWriteRowsPrepare` and `ncclMeshWriteSpansPrepare` bind exact source/landing
extents and jointly compile registration/packet cuts before execution;
`ncclMeshWriteExecute` reuses that plan. The interface that combined preparation
and execution has been removed. A relay's send reads its receive's registered
interval directly and uses the incoming stream's landed prefix as its supplied
word. It can forward a completed span without a host round trip or whole-operand
join. Path weights describe total flow; realization conserves bytes without a
mandatory page for every possible path. Physical cable coverage is assessed
across the function's complete set of writes.
Multi-hop execution declares its participating relays in the compiled fabric.
Directed transfers use that declaration. There is no separate paired
Send/Recv executor with endpoint-only participation.

### Fused physical writes

`torch_mesh.fused.writes` declares a group of those writes and returns publication
and landing bindings in physical channel order. Parallel cables remain separate.
The producer releases one cumulative publication per active cable after writing
its data; the consumer awaits the last landing addressed to this rank on each
cable that carries its inputs. Later relay-only arrivals do not delay that
consumer. Relay entries use incoming landed prefixes directly. Neither a relay
GPU wait nor a publication of an unlanded relay page is introduced.

These writes use leases owned by the complete data function. Previous users of
landing leases must have finished **before the group is encoded**: reception is
posted immediately, without waiting for that node's own producer. Source and
relay leases remain immutable through downstream uses. Overlapping landing
ranges, or landing ranges overlapping local sources, fail during realization.
The DNN uses distinct leases per layer. Reuse requires the compiled source-read
and destination-consumer retirement proofs below, not a peer rendezvous.
Its producer publishes from its last threadgroup; its consumer reads the
received values directly. The ordinary write interface remains available for
the control comparison.

The plan compiles all ranks' actual path fragments into a common dependency
graph. `ncclMeshCommWriteRelease` supplies predicted producer release times.
`mesh_write_order` tracks each directed cable's next free time, and each relay's
first-page arrival and last-page dependency. It assigns the earliest-starting
available fragment next, preserving parent-before-child order. Both endpoints
use this common ordinal; queue order cannot introduce a dependency cycle.
An early relay can precede a Studio's local producer, while a relay whose input
is queued upstream can follow an already-ready local transfer. Forwarding is
streamed, not whole-fragment store-and-forward. This is an AOT list schedule,
not a proof of optimal ordering. It adds no runtime arbitration or wait.
Ordinary groups retain their existing ordering.

## Buffer ownership and GPU payloads

Publication is not send completion. There are no waits on `sent`. The data program
keeps each source and relay lease alive and unmodified through its downstream
uses; independent layers can have independent leases. Earlier tests used a
complete-function rendezvous between evaluations. That boundary changes overlap
and cannot serve as the runtime ownership proof or as a valid performance
comparison for continuously submitted calls. Native source retirement follows the bridge's
existing final DMA-read evidence asynchronously; destination consumer lifetime
is separate. The typed rules below specify exactly which proof is needed.

The native scratch pool distinguishes GPU payload storage from CPU-written DMA
descriptor storage. Encoding the last GPU use makes a payload available for a
later use on the same ordered GPU stream; it does not prove that use has already
executed. Descriptor construction runs on the CPU and cannot borrow this stream
ordering. A descriptor allocation therefore never reuses a payload allocation.
Both roles use the same allocator and transport-retirement bookkeeping, with no
new synchronization, and descriptor reuse requires its preceding transport uses
to have completed.

A GPU producer stores its published result through `mesh_step_write`, and a GPU
consumer reads NIC-written bytes through `mesh_step_read`. Ordinary cached stores
can be invisible to the NIC while a command buffer is running; ordinary loads
can retain old contents of a reused receive buffer. The collective's own payload
producers follow the same coherent-store rule. No staging copy is added to repair
an ordinary producer. The DNN example computes its second matrix product directly
into the published lease using the existing Metal tensor-operation pattern.

### Typed one-sided lifetimes

`rdma/lifetimes.py` is an executable compiler for ownership proofs. Its inputs
are invocation-specific page-range `Lease`s; local `Event`s for first write,
publication, final read and landing; physical `Transfer`s; and declared local
`Read`s. Every physical fragment and relay use must appear, including planned
reads not posted yet. A transfer adds only the real publish → first destination
write → landing relation. Local edges are existing arithmetic dependencies.
There is no arbitrary cross-node edge that could smuggle in a missing receipt.

For a lease `p`, let `uses(p)` contain each final arithmetic read and each
outgoing DMA read. A transfer's final DMA read is proved either by its local
bridge-owned final source completion or by the destination landing; neither
requires destination arithmetic to finish. For every use `u`, reuse needs a
known event `e` with `u ≺ e`. A naturally returning result can supply `e` when
the actual remote data graph contains that dependency. An unrelated landing,
an ordinally later collective, or local function completion cannot supply it.
Remote consumers of the independent destination pages have their own uses.

Monotone prefixes imply all earlier contiguous prefixes on the same physical
channel, direction and connection generation. Every outgoing fragment is an
AND obligation, even when several go to one peer. Alternative witnesses of
one final DMA read are OR proofs. `plan.rules(node, bridge=True)` exposes
bridge-owned source completion to the allocator; the default rules exclude it
from data-program observation. `plan.tracker(...).observe(existing_event)`
updates only affected ownership obligations and returns newly retired leases.
It sends no message and inserts no wait, copy, read of private driver state,
or data-program branch. An unproved lifetime remains retained.

The source completion must cover the entire presented range, not only the
first posted requests. Presentation ownership must therefore survive until
every scheduled use has been posted and every actual DMA read has ended.
The final destination consumer's lifetime is not shortened by that event.
Runtime adapters supply actual events; this pure compiler does not observe
hardware or grant permission to fabricate completion.

Static allocation first-fits compatible page intervals only when every old
use already precedes the new **first write**. Publication comes after stores,
so publication-after-read alone cannot justify in-place reuse. Missing
first-write evidence gets separate storage. The allocator introduces no
dependency to obtain a smaller allocation and claims no minimum-memory proof.
Splitting a lease into page/range leases permits retirement at that granularity.

`prove_recurrent_gather(participants, paths)` checks the following two-slot
proof once at realization, retaining parallel physical fragments and relays:

- Each rank's `publish(i+1)` follows its `consume(i)`.
- Each `consume(i)` follows final landing of all ranks' required pieces at i.
- Local `consume(i+1)` therefore proves that every peer consumed its previous
  value i, so the local source slot i can be first-written for i+2.
- Remote `publish(i+2)` follows that peer's `consume(i+1)`, which follows our
  `publish(i+1)`, which follows our `consume(i)`. Thus the next write into our
  destination slot i cannot begin until its old consumer ended. Forwarding
  preserves the same causal chain; relay page reuse is checked as well.

No destination gate or receipt message is needed for this proven recurrence.
One-slot reuse fails the proof. Independent feed-forward publications and
asymmetric repartitioning also fail that recurrence; their source releases
must use actual DMA-read retirement or another natural proof. The tests cover
these counterexamples, multiple physical fragments, independent channel
generations, returning data, local readers, and remote sinks with no return.

`mesh-lease-retire.h` supplies native source ownership bookkeeping. It records
the maximum final **DMA-unit** prefix for each physical stream/holding before
presentation, including unposted SENDs. The bridge's `sent` prefix is contiguous
completed reads across both hardware lanes (`mesh-link.c:link_send/walk_count`).
All dependencies must complete; one peer's progress cannot release another's.
A local holding generation is separate from the wire/QP identity, because the
same wire resets its counts for a later holder. A lost `~0` prefix, changed
wire, changed holding, timeout, or local GPU completion is never retirement.
Such a dependency stays retained until an explicit actual-DMA/quiescent-driver
proof. The helper is nonblocking and adds no transport traffic. Its metadata
destructor is not a proof that the underlying pages may be overwritten.

### Exact sub-page writes and historical masked frames

`migration.compile_rows` represents a partition intersection by an affine byte
map: base, row width, row count and stride at each endpoint. The current
`ncclMeshWriteRowsPrepare` lowering resolves that map directly into immutable
`mesh_dma_span {offset, bytes}` descriptors, with one exact payload SGE per
span. It splits actual registration crossings at each participant's row phase;
allocation pages and 4 KiB hardware credit charges do not pad the payload.
One table covers all rows, with no per-row host execution call. See
[the transport contract](transport.md#1-the-objects) and
[live byte-span validation](transport.md#exact-byte-validation-on-2026-10-24).

`ncclMeshWriteSpansPrepare` extends that same plan with an ordered interval
list inside one lease. Local lists can have different counts, gaps, and physical
ordering while naming the same logical byte sequence. Common cuts are the
union of all participants' view boundaries, packet cuts and actual registration
crossings. Repeated source bytes are permitted; overlapping landing intervals
are rejected. The metadata-only `mesh_view_layout` compiler (rdma/mesh-write-compiler.h) maps ndarray/tensor
shapes and byte strides to this contract, including negative strides. Distinct
leases use ordered shared WRITE declarations. Neither form adds a numerical
packing pass or runtime metadata exchange. Indexed preparation has local
sanitizer tests; the live results above cover the contiguous/affine forms.

The earlier masked-frame implementation carried logical payload intervals in
separate padded transport storage. GPU pack/unpack masks preserved neighboring
values, and its measured cost included those extra memory passes and padding
on every hop. Those tests remain evidence for that historical realization;
they do not require the exact-span path to introduce padding or copies.

Ownership needs no new event kind. Original sources and relay intervals retain
all future and posted DMA reads; destination intervals retain their arithmetic
consumers independently. Immutable descriptors are ordinary closure inputs
retained through every bridge CPU read and SEND/LAND obligation. The final
output's publication follows all stores required by its declared extent. In an
isolated one-way stream, serial receiver consumers do not order already
preposted remote writes, and source DMA retirement does not retire a destination
consumer. Distinct landing generations or an existing causal descendant proof
are required; the recurrent gather's two-slot theorem cannot be assumed.

## Ahead of time

`routing.Fabric` preserves the physical multigraph and enumerates simple physical
paths. For a declared function's pairwise byte demands, `allocate` solves a
multi-commodity linear program: each commodity's path flows sum to its bytes,
and each directed physical cable's aggregate load is bounded by its capacity
times a common duration. Zero is an exact allocation value; all paths remain
coordinates of the empirical solver even when their initial allocation is zero.
It uses [HiGHS through SciPy](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html).
The result is the fluid model's constrained optimum, not a claim that a real
program with dependency release times attains that duration. Integer transport
quanta, producer times, queues, and dispatch costs remain for measured feedback.

On that optimum-duration face, a convex AOT solve minimizes
`sum_e(load_e/capacity_e - T)^2`. This equalizes service completion times,
including unequal cables. There is no intervening minimum-hop constraint:
Mini traffic can traverse the spare Studio links without adding any traffic to
the Mini cut. In the 4/1/2-cable triangle with equal all-pairs demands, the
minimum-hop solution leaves Studio cables idle; equal-finish routing can fill
all fourteen directions at the same fluid duration. The extra hops are reported
as wire amplification, never extra useful bandwidth. General graphs and demands
need not admit equal completion or full utilization on every edge.
Before realization, each node-path's flow is averaged across its parallel
physical paths in capacity proportion. This preserves its total flow and cannot
increase the bundle's maximum service time. It also avoids an arbitrary solver
nullspace assigning one producer to one cable and its relayed complement to
another, despite those cables having the same release dependencies.

## Numerical operands

`assignment.assign(decisions, observations, step=proposal)` uses bounded linear shares. The compiler rounds compute shares to
whole tiles and path shares to Q40 units. Projection preserves each declared total. The numerical fits have positive
regularization; they use neither logarithms of shares nor division by a share that can vanish.

A compute share starts from declared capacity, with equal capacity where none is known. Torch configuration reads
each node's GPU core count as a prior, not as a performance measurement. Every solve fits fixed and work-dependent
costs from complete-call observations. A fixed cost per code version separates changes in the implementation from
changes in its operands. The duration of each complete call fits the stage envelope directly; the model does not
require an optional list of per-operation costs. The affine envelope estimates whole-call cost by work share. Its
coefficient variances retain uncertainty where the completed calls cannot distinguish two resource costs.

The historical equal-finish solver rounds the fitted allocation. Whole-call finite differences move all compute
and route coordinates together. Compiler parameters use the same vector. An explicit proposal ordinal advances
the paired perturbations; retaining only a bounded history cannot freeze their direction or radius. Each compute
or route axis carries its realization grain, so a perturbation can still cross a compiler rounding boundary. No result is accepted or
rejected as a candidate. There is no saved controller or hidden solver lifecycle to restore. The saved data is what
ran, its actual operands, and its measured completion interval.

This is an empirical local solver. A small residual, a compiled plan, or a capacity prior does not establish a global
optimum. The required evidence remains lower whole-function elapsed time and higher useful output per second.

The compiler retains the newest configured number of observations by proposal ordinal. Arrival order cannot
rename an old proposal or turn it into the next finite difference. Regression weights combine the observation's
explicit importance, its distance from the current operand vector, and the width of its completion interval.
The coefficient fit and residual fit are regularized. Analytic derivatives avoid a new objective evaluation for
every routing coordinate. The only outcome sent to preparation is the next bounded vector and its cost estimates.

`etc/compute/parameters.json` declares the FX linear/matrix crossover domain. The chain, LSTM and DNN
configuration operands declare replay grouping, input publication grouping, resident tile threshold and DNN
row grouping where those apply. The demos consume the returned values before building their commands. The
hourly supplies those same configuration documents and calls the demos.

The fitted per-source envelope also informs producer scheduling. `producer_seconds` maps declared holder groups
to physical ranks. Preparation divides that envelope across the actual repeated producer groups, and the shared
`ncclMeshProductionTimes` function maps each group's existing traversal onto that interval. Both Torch and native
call it before `ncclMeshCommPlan`. These are inferred release times used to order descriptors, not measured tile
completion times. Actual publications remain the data dependencies. No timer, wait or polling operation is added.
The initial graph has its declared traversal and no measured time distinction.

## Compilation

`configuration.placement` reads observations before tensors are built. `configuration.bind` declares the complete
function and installs the resulting page placements and routes. The numerical function is absent from execution.
More routes change the compiled tables; they do not add a controller around each transfer.

`rates.step`, the engine example, and the Xonotic planner and solver consume the same assignment operands. They keep
observations as data. They do not reconstruct a controller from a JSON object. A failed invocation is a record, not
a fabricated duration or an instruction to roll back its allocation.

## Observations

Programs execute their own function calls. There is no measurement runner taking a callable. Whole-function
observations use a host clock and ordinary command-buffer completion state. No GPU timing event, timestamp shader,
completion callback, or timing-only host drain is added. The stream's existing publication operation submits output
writes without waiting for their completion.

A separate statistics thread polls that state. It records the interval between its last pending observation and its
first completed observation. Reported elapsed time uses the upper end, so useful bandwidth uses the conservative
lower end. Completion at a shared command-buffer boundary is coarse. Pending records and several late peer records
can coexist; observing them does not halt a subsequent function. Shutdown drains the records after submission ends.

Records contain whole-call time and useful bytes together. Relays do not increase the useful byte count. The hourly
runner uses committed main and the same library functions as the examples. Its scripts supply configuration operands.

Configuration and observation state stays in memory while functions run. A statistics worker never opens a log or
rewrites a configuration file. Rank reports travel over pipes into memory. The hourly writes its complete metric
record only after all workloads end. Seeded inputs, weights and full outputs stay in memory; they have no file format
or disk path. Code installation and compilation happen before execution. Unchanged committed code is reused.

Blocking filesystem reads also stay outside execution. Load source and configuration during preparation, then pass
memory operands to the running function. A page transfer may beat an SSD's first-byte delay (data-model.md section 5);
disk staging and per-call file updates cannot stand in for page-to-page DMA.

## Measurements

A completed function supplies its elapsed time and useful output bytes together. The hourly record stores
microseconds and useful GB/s; row workloads also store rows/s. The committed workload configuration fixes the
shape order and useful byte count. Three significant figures fit the coarse host observation.

There is no per-transfer byte ledger, trace observer, timing event, or timing-only GPU gate. The old
`ncclMeshTrace*`, `ncclMeshCommSample`, and LSTM split-timing interfaces were removed. Their bookkeeping ran on
transfers even when no trace was requested. Historical trace analysis does not belong in the execution path.

The record contains commit IDs, installed node IDs, and these completion measurements. Optimizer observations
stay in the application's memory. They are not restored from the hourly ledger. A fresh process starts from the
compiler's prior; repeated calls in one application can supply their completed observations to the solver.

## Roofline

Use the actual connected fabric's capacities and the function's required work to bound completion time. Relay
bytes are work needed to deliver the result, not extra useful output. Compare measured whole-function latency
and useful throughput together against that bound. Extra accounting or clock-probe traffic must not alter the
function being measured. A compiled schedule alone does not establish that it reaches the bound.
