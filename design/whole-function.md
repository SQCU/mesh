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
index and buffer interval. Parallel capacities add. `mesh_fabric_stripe` is a
direct-edge adapter to that same write allocator, not a second collective
transport or an application routing interface.
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
retains those observations; [data-model.md](data-model.md#8-writes-over-many-hops-and-many-paths)
links the earlier bridge-log event without inferring its cause.
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

`ncclMeshCommWriteRoutes` installs paths in the communicator's configuration.
`ncclMeshWritePrepare`, `ncclMeshWriteRowsPrepare` and `ncclMeshWriteSpansPrepare` bind exact source/landing
extents and jointly compile registration/packet cuts before execution;
`ncclMeshWriteExecute` reuses that plan. The eager `ncclMeshWrite` prepares and
executes the same representation. A relay's send reads its receive's registered
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
are rejected. The metadata-only `views.byte_spans` helper maps ndarray/tensor
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

## Finite differences

`whole.FunctionFlow` uses normalized linear shares stored as nonnegative integer
counts summing to `2^40` for each decision. The declared numerical precision is
one part in `2^40`; this is independent of the compiler's page/tile granularity.
Zero and one are exact. There are no logits, epsilon path shares, exponential
tails, or gradient moments. Euclidean simplex projection enforces bounds and
conservation. Fixed coordinates need no division by their width. Natural units
must give a total within `[2^-40, 2^52]`; invalid dimensions or nonfinite inputs
fail during configuration. Recorded state names this coordinate version and
cannot silently reinterpret old logarithmic states.

Physical route operands are unsigned integer weights, including exact zeros.
The common C allocator computes each `units * weight / sum(weights)` with
128-bit integer products, then assigns residual units by exact integer
remainders in path order. A zero-weight path gets zero units; it remains
available to the next solver proposal. Overflowing sums and all-zero vectors
fail explicitly. Python configuration uses canonical weights summing to `2^40`.
Thus normalization and transport apportionment cannot create subnormals,
underflow-dependent ties, or `log(0)` singularities.

A fixed Hadamard sequence supplies joint directions. Four complete executions
form a `+--+` stencil, canceling additive linear drift in measured cost. Each
allocation's perturbation is at least one unit reported by
`ncclMeshCommPlanRouteUnits`: the greatest independently allocatable unit count
of an individual logical transaction, before repeats and relay amplification.
A million repeated one-page writes therefore do not masquerade as one
million-way divisible write. Projected probes can enter a zero-share path.

Every stencil updates the centre with a bounded gradient step and a decaying
step size. The divisor `1 + max(abs(gradient))` is at least one. Flat cost gives
an exact identity update, without residual momentum. There is no candidate
acceptance, rollback, or winner selection. Latencies are represented in seconds
within `[2^-40, 2^40]`, and useful work in `[2^-40, 2^64]` declared units, keeping
log costs and reported throughput finite. These are numerical domain checks,
not bandwidth restrictions; an input outside them needs explicit rescaling.

A fluid LP bound certifies its declared model. Empirical finite differences do
not certify the unknown hardware's global optimum in a finite number of noisy
observations. Report attained whole-function latency/useful throughput against
the bound for each shape/depth, plus every executed proposal. A static seed is
not evidence of empirical convergence. Logical tree decomposition and its
shares also affect the feasible program; a route-only feedback run cannot be
reported as optimization over all collective decompositions.

The current default cost is `J = log(T/U)`, where `T` is complete function
latency and `U` is fixed useful numerical work. The DNN uses one logical final
output (`rows * width * 4` bytes), independent of replicated partials, relays,
padding or collective decomposition. With fixed `U`, lower latency and higher
whole-function useful throughput are the same objective. Empty queues or
smaller finish variance cannot compensate for a slower function. Queue area,
completion residuals and clock-bracketed finish spread remain model inputs and
reported evidence, not independent rewards. Every observation names the exact
compiled configuration, workload and completed function.

An explicit `transaction_weight = w > 0` selects
`J = log(T/U) + w*log(W/U_transaction)`, where `W` is the conservative upper
bound on the transaction-window union and `U_transaction` is fixed under its
declared contract. This is an explicit throughput tradeoff, not an inferred
congestion preference or strict lexicographic guarantee. The default is zero.
See [objective-20261024.md](objective-20261024.md) for the derivation, unchanged
continuous updates, accounting invariants, and the distinction between final
numerical work and implementation-specific transaction bytes.

The DNN also enables the historical affine `allocate.equal_finish` calculation
(restored from `fa36bea6^`) and its existing `min_max` tile realization. GPU
events bracket each local producer, and their durations are read after the
existing complete-function synchronization. Completed calls fit
`compute_i(s) = a_i + b_i*s` by exponentially weighted nonnegative least squares,
separating fixed and work-dependent cost. A weak prior resolves the initially
underdetermined fit; the record exposes both coefficients rather than claiming
that one share identifies a slope. The previous stencil also supplies input
release offsets: subtract each rank's measured producer duration from its own
publication time, then express those inferred start times relative to the
earliest rank. Equal-finish uses `start_delay_i + a_i + b_i*s`. Thus a rank whose
inputs arrive later receives a shorter compute stretch even if its arithmetic
rate is otherwise equal. The repeated-stage observer averages these offsets
over stages; it is not a per-layer timing model for arbitrary programs.
Communication also remains in the joint whole-function objective. Recorded own
first-hop delivery windows are diagnostics, not added compute intercepts:
those windows also contain waits for upstream producers, and including them
can create circular feedback. Every four-call stencil moves the continuous
compute centre halfway toward the newly fitted equal-finish solution. Its
influence does not decay away while measurements continue. Tile rounding uses
`min_max` and measured slopes, retaining
fractional latent shares.
There is no hard-coded Mini fraction and no measured-candidate rejection.

This is reuse of the historical allocation mathematics, not restoration of
the old `Balancer`/`Coupled` acceptance and trust-region state machines. The
historical documentation also described RLS which was subsequently documented
as absent; the present weighted affine fit is a new implementation, not evidence
that those historical RLS claims were implemented.

The DNN supplies `Fabric.completion_model`, the physical path-incidence matrix
scaled by each commodity's bytes and each directed cable's capacity. After a
four-call stencil, `FunctionFlow` fits the observed completion residual beyond
the physical path-load model. The last 64 complete calls fit a weighted ridge
Jacobian of those residuals against joint route and compute fractions. The
physical capacity Jacobian is the prior in directions not yet identified by
the observations; changed upstream queuing can introduce cross-flow derivatives.
The local model is anchored at the preceding stencil's measured mean finish,
then adjusted for the next compute centre. Its route coordinates satisfy
`finish(x) = b + J*x`, with each commodity's path fractions summing to one.
A convex proximal step minimizes a monotone smooth maximum finish plus
`||x-x0||^2/(2*rate)`, with the same decaying learning rate as the joint
finite differences. This is an incremental step in a local model, not global
minimization of its extrapolation. Finish times are normalized by the preceding
stencil's longest finish; smooth-maximum sharpness is 64. The result becomes
the next vector unconditionally. The next complete calls refresh `b` and `J`
from actual landings; neither delay nor its sensitivity is assumed constant
across executions. Residual fit error, Jacobian, and predicted finishes are
recorded. This supplies a physical
preconditioner for fourteen simultaneous finish residuals while finite
differences retain the full function's measured useful-throughput objective.
The finish-variance term was removed: delaying an early completion must not
improve the predicted performance score. Model corrections and quantized
finite differences do not establish descent on every actual evaluation.
It adds neither runtime routing choices nor measured-candidate rejection.
The harness realizes one canonical configuration on rank zero and distributes
its serialized values before compilation. Constrained numerical solvers can
differ by a few floating-point ulps across machines; independently realizing
those variants is inappropriate for matching page schedules. This planning
exchange is outside the complete-function execution. Source release predictions
use mean own-publication times from the preceding four-call stencil, adjusted
by the fitted compute slope for the next tile allocation. The predictor stays
fixed within a stencil. Predictions affect only the next
compiled entry order, never transport's current readiness words.

`ask(explore=False)` realizes the current centre without perturbing or updating
the optimizer. The harness can measure a fixed number of these complete calls
after training (`--centre-evaluations`), separately from finite-difference
samples. It never picks a historical fastest configuration for this validation.

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
actual payload bytes carried by all hops (including any explicit application padding), report
attained wire bandwidth `V/|W|`, utilization `V/(C*|W|)`, unused capacity
`C*|W| - V`, and the all-links ideal duration `V/C`. Also report the tighter bound
`max_e V_e/C_e` for the measured routing. Useful delivered bytes `U` are reported
separately, giving `U/|W|` and hop/padding amplification `V/U`. Extra relay traffic
must not masquerade as extra useful throughput.

`roofline.py` matches outgoing and incoming entries by physical cable, direction,
and cumulative DMA-unit coverage, validating each unit's exact byte length.
SEND and receive entry boundaries can differ; matching uses interval coverage,
not entry ordinals or peer counts. Each transaction window starts with source
eligibility and ends at the destination's completed landing, rather than
substituting local send completion. It merges overlapping windows rather than
adding their durations and reports ingress and egress for each node separately.
Across nodes, causal TCP request/reply clock observations bound root-minus-local
offsets immediately before each complete evaluation. It expands and contracts
windows by those bounds and the polling brackets, retaining the resulting
latency and bandwidth bounds. Offset constancy is assumed only over that
evaluation. These transaction windows include clock uncertainty even in their
per-node summaries because their endpoints are on different nodes. A lower
duration bound of zero has no finite bandwidth upper bound; uncertainty is not
silently replaced by a point estimate.

Clock probes now reuse the existing TCPStore connection. Recorded span causality
also tightens their offset intervals: source eligibility precedes first arrival.
An all-pairs shortest-path closure propagates these difference constraints.
No symmetric-delay assumption or assumed attained bandwidth enters the bounds.
Inconsistent constraints fail the measurement. Reports retain both raw probes
and refined intervals; historical comparisons apply the same refinement to old
raw traces, so narrower clock uncertainty is not credited as a transport speedup.

Report best attained whole-function latency and best attained transaction-window
bandwidth with their evaluation IDs and corresponding other measurements. They
may belong to different evaluations. The morning microprofile's advertised
80 Gbit/s per direction on seven cables gave `C = 140 GB/s`; the six paired
cables observed at 12:57:27 PDT gave `C = 120 GB/s`. Use the actual discovered
fabric of each execution. These are advertised physical ceilings, not
demonstrated sustainable payload rates. Positive traffic in every discovered
direction establishes coverage; utilization measures the gap to that specific
fabric's capacity. Historical measurements retain their original denominator.
