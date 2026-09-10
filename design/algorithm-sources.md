# Algorithm sources and implementation obligations

## Operator clarification: asynchronous error metadata

Operator instruction, September 9, 2026:

> error status returns are allowed by the specifications we use... but it can never be synchronous or blocking or consumed by a callgraph itself... it can only be passed monadically through a metadata channel... s.t. calling contexts can call `out, meta = meshfunction(x)`, with literal values corresponding to error codes and where/when they happened in a callgraph reported in the meta channel, not the output channel, for calling consumers to interpret and handle...

This supersedes blanket claims that error returns themselves are forbidden and
older requirements to terminate numerical execution on a negative status.
The [return-channel contract](pages-and-functions.md#asynchronous-error-metadata)
requires asynchronous propagation without error-dependent callgraph control.
Configured metadata pages carry the code and callgraph provenance; numerical
outputs remain numerical. Only the calling context interprets metadata. An
error report is not an execution-completion certificate. No new runtime
implementation is claimed by this clarification.

Current source disposition:

- `mesh_rows_receive` returns negative codes synchronously and exits its receive
  pass on those paths. It does not yet publish the required metadata values.
- `mesh_rows_progress` consumes the negative receive return and exits before
  transmission and retirement. This is error-dependent transport control, not
  monadic propagation, even when called by an asynchronous owner.
- `runReduceScatter` reads `mesh_pages_status` in numerical launch/completion
  paths and uses it to suppress work or cancel outputs. Those uses remain
  excluded. Reporting status in the final caller-facing metrics does not cure
  the internal control dependencies or provide per-occurrence provenance.
- Removing all status returns or converting functions to `void` does not by
  itself implement the required metadata channel. Add the configured return
  binding, migrate the single caller, then delete excluded implementations.

### Configuration owns receive input lifetimes

`mesh_rows_realize` now accounts for every logical row, including holes that
previously escaped validation because only produced outputs were examined.
Every row read by a configured function, a remote dependent-read binding or the
calling context's return maps must have a local producer or receive binding.
Its configured use count must equal those reads and its transmit uses. Unused
holes remain legal. This is configuration work before launch and allocates no
auxiliary ownership structure. It uses the existing maps and row-use algebra.

The receive path no longer rereads dependent input stamps or use counts to
validate a remote completion. A configured received output is the dependent-read
proof; the path performs the configured releases directly. Papadopoulos–Culler
(ISCA 1990) provide assigned operand storage and data-dependent firing, as cited
below; the repository's fixed maps supply the specific lifetime proof. Counting
uses alone does not establish acyclicity or protect against an invalidated
mapping receiving a late device write. Those remain distinct integration
obligations, not permission to add per-input runtime guards.

The other receive descriptor/address checks and negative return paths remain
unconverted. This change does not establish asynchronous error propagation or
complete caller migration.

### Asynchronous metadata publication

`mesh_rows_report` implements publication into a configured metadata output
page. Papadopoulos and Culler, *Monsoon: An Explicit Token-Store Architecture*
(ISCA 1990), supply the assigned-storage and presence-publication precedent;
Saltzer, Reed and Clark, *End-to-End Arguments in System Design* (TOCS 1984),
supply the endpoint responsibility for interpretation. Both publications are
listed below. Neither specifies this record layout or this API; the operator's
return-channel contract determines those choices.

The record contains the invocation stamp, occurrence time, configured function
identity, numerical index, peer identity and literal signed error code. The
producer supplies these values; publication neither samples a clock nor branches
on the code. `when` is a producer-local occurrence time in the clock domain and
units established by configuration. It does not imply synchronized peer clocks.
Zero and nonzero codes take the same publication path.

Before launch, realize an ordinary output map with one page per occurrence,
sufficient payload for `sizeof(struct mesh_row_metadata)`, one writer per row,
and return/transmit use counts. Include those outputs in the complete graph's
allocation and lifetime accounting. The `occurrence` argument selects that
configured output row; the record's `index` identifies the numerical operation
being reported. Different reporting sites and overlapping invocations require
distinct live destinations. There is no append counter, metadata queue, overwrite
of an unread event, or allocation during publication. Reuse obeys the same page
lifetime contract as every other output.

An asynchronous producer writes the record directly into the actual sendable
page payload, installs its configured page/use count and publishes the row stamp.
The function returns without waiting for execution, transport or a consumer.
The metadata output has no numerical dependents and publication releases no
numerical inputs. Only the calling context interprets it. Ordinary page transport
can carry it without inspecting its error code. This primitive is not yet bound
to the receive path or the single numerical caller, and does not establish full
error propagation, invalidation or a measured latency bound.

## Operator clarification: caller-owned repetition

### Complete allocation and unconditional invalidation

The operator further requires every page-table allocation for the complete
compute graph to be known before launch. The API must allow the caller to
invalidate its page table at any time, including for pages used by a hung
kernel. Consumption, freeing and explicit invalidation are permitted operations;
data-dependent branching, error recovery and related control flow are not
introduced into numerical functions, transport or reduction.

Every deployed node has NVMe storage or transitive RDMA access to a node with
NVMe. Missing or destroyed parameter pages are reproducible from that storage
and can be retransmitted. This invariant is supplied by the operator; the
numerical layer does not probe, infer or negotiate it.

Papadopoulos–Culler provide the configured operand-storage precedent, and the
caller-owned decision to reject and repeat remains an endpoint responsibility
under Saltzer–Reed–Clark. Neither citation authorizes the excluded control flow.
The implementation still owes a complete invalidation binding: clearing an
entry alone does not revoke an already-issued GPU or NIC memory access, and
late completion must not republish invalidated rows. A helper that only clears
entries would not establish the requested hung-kernel behavior.

Operator instruction, September 9, 2026:

> recovery is something which is only sanctioned through the mechanism of a totally feed forward nfe computation callgraph being re-run by the calling process because they don't like the results they got. there are to be no parity mechanisms or syncs, guards, waits, checks, or inferences in the mesh computation transport and reduction layer, and functions which use this layer are not to introduce data dependent control flow attempting to add overhead of this sort or any sort.

This governs the implementation and supersedes recovery obligations inferred
from older text below. The calling process evaluates the returned result and
may rerun the complete feed-forward NFE. Mesh computation, transport and reduction
do not recover or replay an invocation. Functions using them must not reintroduce
the excluded control flow. Feed-forward numerical dependencies and configured
page addressing do not authorize an additional runtime recovery protocol.

The proposed cancellation/drain/recovery functions were not implemented and are
withdrawn. `mesh_pages_recover` and the old caller's digest admission gate remain
implementation divergences, not required mechanisms. Existing defensive branches
in `mesh_rows_receive`, publication and release likewise do not acquire approval
merely because earlier additions compiled. The replacement must be reviewed
against this instruction before the single caller is migrated; excluded code is
deleted after that migration, in the operator's required order.

The operator requires a citation to one of the authors/publications below for
every new function. Source citations may point to this document; prose stays in
documentation. A citation identifies the mechanism being implemented, not a
claim that the publication implements this repository verbatim.

## Operand matching and storage

Gregory M. Papadopoulos and David E. Culler, *Monsoon: An Explicit Token-Store
Architecture*, ISCA 1990, sections 2–3.
https://www.cs.cmu.edu/~18742/papers/Papadopoulos1990.pdf

Compiler-assigned operand locations and storage-associated presence transitions
realize dataflow firing. Here the canonical RDMA page table is the sole authority
for operand presence, destination ownership and use counts. Configure numerical
functions and their input/output maps before invocation. Selection claims actual
destinations; completion publishes them after device visibility. No Job, stage,
parity cursor, completion token or caller-owned consumed mask may authorize work.
Monsoon's activation frames and token queues do not authorize equivalents here.

`mesh_pages_scan` checks destination stamps before matching input presence and
stamps, then claims the selected destinations. The ordering avoids repeatedly
reading operands for outputs already issued; it does not add state or change
the conjunction defining eligibility. The caller binds normalization's two
ownership spans to this same mechanism before invocation. Matching and claim
still rely on one scanner for overlapping destinations.

Each matching conjunction stops at its first failed condition. Reading the
remaining pages after a missing operand or already-issued destination cannot
make that candidate eligible. The prior eager loops nevertheless read every
page and dominated the sampled numerical thread during full-graph execution.
Short-circuit evaluation changes no persistent state and requires no readiness
cache or scheduler; later scans observe newly published values normally.

The first destination stamp provides a cheap rejection for already-issued
rows. Candidates surviving that check match their inputs before scanning the
complete output spans and physical ownership. The first stamp never authorizes
a claim: every destination still passes the full validation. This avoids
reading thousands of free destination pages for a future function whose first
operand is absent, while preserving the same matching predicate.

Every numerical intermediate, accumulator and index value occupies actual
sendable page payloads. Views gather/scatter through those pages. Lifetimes must
include residual reads, asynchronous device reads and hashing. Reuse follows
completed dependent reads; allocating every intermediate forever or renaming a
stage counter as a page index does not implement that lifetime rule.

### Reusing configured physical spans

`mesh_pages_slot.storage` names a one-based root slot for a shared physical
span; zero retains a separately allocated span. Roots name themselves and
cover the widest participating slot. Logical rows remain distinct. The
compiler allocates the physical span once and initializes its payloads.
This is configuration of ordinary sendable pages, not a storage privacy class.

`storage_ready` reads the existing physical-owner table; `claim_rows` writes
the destination stamp, restores its literal entry and records its physical
owner. As with the existing non-atomic multi-destination claim, one scanner
must own overlapping destinations, including physically overlapping spans.
The current caller assigns only normalization destinations to these spans;
all their claims occur on its single numerical scan thread.

`retire_storage` requires full publication and completion of every configured
dependent output at the corresponding generation. Whole-span completion is a
conservative read-lifetime proof; first-page publication is insufficient.
The existing use counts reflect outstanding dependents. Transported values
also retain storage through NIC completion and local hashing, without waiting
for a peer digest comparison. Retirement clears actual entries and submits
zeroing to the existing progress-to-helper queue. The helper makes the existing
physical-owner entry free only after zeroing the payload. It does not return
local arena pages to the receive bridge's free list.

This implements the storage lifetime obligation scoped by Papadopoulos and
Culler above. It does not infer lifetimes from numerical graph names, call
application kernels, or add caller phases. Correct dependency specifications
remain required. Broader storage reuse and link-recovery acceptance are still
unfinished; configuration-time zeroing during recovery does not establish
that all recovery paths are correct.

## Collective arithmetic and asynchronous reduction

Rolf Rabenseifner, *Optimization of Collective Reduction Operations*, ICCS 2004.
https://fs.hlrs.de/projects/rabenseifner/publ/myreduce_iccs2004_2.pdf

Pitch Patarasuk and Xin Yuan, *Bandwidth Optimal All-reduce Algorithms for
Clusters of Workstations*, JPDC 69(2), 117–124, 2009.
https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf

Use reduce-scatter followed by all-gather. For balanced ownership the sent
payload per participant is 2(n-1)|S|/n. Match input pages, accumulate in FP32,
and emit each result when its numerical inputs are complete. The collective
names do not impose whole-tensor execution barriers. A normalized row requires
all its feature contributions; independent rows need not wait for one another.

`rdma/mesh-pages.c::reduce_step` implements indexed partial addition using
configured maps and destination stamps. `reduce_run` scans that static list
on one runtime worker. `mesh_pages_reduces` realizes the maps; `mesh_pages_start`
and `mesh_pages_stop` manage worker lifetime, not numerical dependencies. This
single worker is an implementation choice supporting operand matching, not a
published guarantee of speedup. Transport progress remains independent of
arithmetic. The existing FP16 materialization and later normalization still
need replacement by the prescribed accumulator/index-page composition.

## Endpoint checking

Jerome H. Saltzer, David P. Reed and David D. Clark, *End-to-End Arguments in
System Design*, ACM TOCS 2(4), 277–288, 1984.
https://web.mit.edu/saltzer/www/publications/endtoend/endtoend.pdf

Endpoint checks establish integrity. The repository additionally requires
post-consumption checking: digests are ordinary pages, mismatch is an evaluation
failure value, and link failure is link status. Digests cannot gate numerical
execution or supply a missing numerical lifetime dependency.

## Overlap and performance evidence

The PyTorch authors, *Introducing Async Tensor Parallelism in PyTorch*, 2024.
https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487

Decompose actual communication/computation dependencies to overlap their work;
preserve efficient local numerical functions. Small operations can lose to launch
cost, poorer matrix utilization and resource contention. One scan issues all
currently ready work for a configured GPU function together. Measure matched
local and RDMA executions, including numerical error, latency, throughput,
submission count and actual communication cost. Published bandwidth bounds do
not attribute seconds of end-to-end time to RDMA.

## Remaining replacement

### Literal row functions

`rdma/mesh-dataflow.c` adds the required operations independently of the old
runtime. Papadopoulos and Culler, *Monsoon* (1990), sections 2–3, are the cited
source for storage-associated matching; the repository supplies the stricter
one-page-table representation. `mesh_row` contains exactly physical page, use
count and stamp. `mesh_rows` and `mesh_row_function` are immutable configuration
views, not per-evaluation state. They do not allocate an owner array, publication
bitmap, progress counter, task, token or queue.

`mesh_rows_validate` checks configuration, `mesh_rows_present` reads actual
rows, `mesh_rows_select` claims destinations in those same rows, and
`mesh_rows_publish` stamps completed visible output writes directly.
`mesh_row_data` resolves a row to its literal page payload. Selection requires
one scan owner for overlapping destinations; publication has one completion
owner per issued function. Selection output is transient numerical indices,
not a persistent consumed mask. `row_maps_overlap` checks the configured logical
and physical spans. `mesh_rows_realize` validates the complete configured
function list and bindings before initializing the canonical rows and zeroing
their assigned local pages. It excludes overlapping writers and overlapping
receive/output destinations. Its current realization assigns distinct physical
output spans to distinct functions; shared physical spans across functions are
not yet realized.

Output maps contain their fixed physical span and configured use count, in
addition to the logical row geometry. These are immutable allocation/lifetime
configuration, not a mutable ownership mirror. Selection accepts a newer NFE
only when the previous value's actual row has no page and zero uses. It retains
the previous completed stamp until claiming the newer value, then reinstalls
the configured physical page and use count. Clearing the stamp to zero on reuse
would incorrectly enable the old generation again, so that is not done.
The initial row state is likewise absent, zero uses and stamp zero.

`mesh_rows_uses` derives each row's use count during configuration from all
function-input occurrences, remote-read input maps, one NIC use per transmit
binding, and the graph's returned-page maps. Returned maps describe one caller
read of each row in their range; they are function-return storage, not a runtime
completion mechanism. Hashing must appear as a function input like every other
read. Neither the query nor use-count validation runs during an invocation.

The configuration helper `row_map_uses` counts the indices `i` satisfying
`0 <= i < rows` and `first + i*stride <= row < first + i*stride + count`.
For nonzero stride these form an integer interval, intersected with the configured
index range; zero stride contributes either all indices or none. Thus overlapping
input ranges and broadcast operands count every consuming function occurrence.
This realizes the compiler-known operand/lifetime relationships attributed to
Papadopoulos–Culler above without a mutable ownership mirror.

`mesh_rows_realize` compares every configured local-output and receive-row count
with that derived count before initializing the table. A map with nonuniform
uses must be split into the corresponding configured output maps. Counts that
exceed the row's 32-bit representation also fail this configuration comparison.
The only additional arguments are the immutable graph-return maps. Invocation
continues to decrement the literal use count directly; it neither recomputes
the graph nor validates a decrement. The caller still has to supply the complete
graph, including arithmetic, hashing, remote reads and returned values.

`mesh_metal_row_table` aliases this same array of physical-page/use-count/stamp
rows for the GPU using the existing no-copy mapping function. The configuration
owner supplies a page-aligned, page-rounded mapping whose lifetime covers the
alias. There is no GPU copy of the row table and no separate indices array.

`mesh_rows_publish` writes the configured output stamps and releases the
configured input uses. It performs no runtime operand validation, issued-stamp
test, or error return. Its loops traverse fixed configured maps. Configuration
and the numerical call graph must establish exactly one publication after each
writer completes; repeated publication is not intercepted by this primitive.
`mesh_row_release` directly decrements the actual row use count. It has no stamp
argument, validation branch, source-level compare/exchange retry loop or result
to interpret as permission. Each call consumes one use already accounted for
before launch. Atomic access applies to the literal row contents, not a separate
synchronization object. Papadopoulos–Culler remain the operand-storage citation.
The caller's complete use accounting and invalidation binding are still required;
removing guards alone does not establish those contracts.
Arrival integration must likewise provide exactly one release per configured
remote read proof. `mesh_row_zero`, called asynchronously after all
uses end, excludes writers through the row stamp, zeros the payload and removes
the physical page while retaining the completed stamp. The release owner retains
the physical page number through this operation and returns it to the appropriate
free list/bridge. Reinstallation occurs in selection while the row carries its
issued stamp. Hardware reads and hashing count toward the lifetime. A row whose
page is already absent is not zeroed or released a second time.

Receive bindings now carry the immutable input-row maps of the remote numerical
function whose output arrives in those rows. For received output index `j`, the
input map names `first + j * stride ..< first + j * stride + count`. Arrival of
that output with stamp `k` proves those configured remote reads have finished.
`mesh_rows_receive` validates the still-live input rows, publishes the arriving
output and releases one use for each mapped input occurrence. The destination's
existing stamp prevents a repeated arrival from releasing those uses again.
There is no acknowledgement page, consumed bitmap or per-function completion
counter. This is the dependent-output lifetime rule of the plain specification,
with Papadopoulos–Culler supplying the storage-associated presence principle.

For two-peer reduce-scatter, an arriving normalized output page can release the
corresponding sent partial page's remote-read use. The source also retains its
independent NIC-completion and hashing uses. Configuration must assign exactly
one output occurrence to each proven read, including when one numerical row has
several output pages. A proof for all inputs can attach to one returned output
only when that output's stamp proves all those reads finished. A digest verdict
is not substituted for the numerical output. The caller still has to realize
these input maps and counts; their existence is not evidence that the old
caller's lifetime logic has been replaced.

### Literal weight pages

`ModelFile.loadRows` in `metal-microbench/model_file.swift` realizes the
compiler-assigned operand placement described by Papadopoulos and Culler,
*Monsoon* (1990), sections 2–3, under the operator's stricter requirement that
every shared-memory operand occupy actual RDMA-sendable pages. Configuration
supplies a claimed output map, selected tensor coordinates and destination
precision. The loader decodes BF16, FP16 or FP32 directly from the model's file
mapping into those pages. It creates no intermediate dense operand buffer.
The source format and destination precision are configuration choices.

One numerical weight row occupies `ceil(columns / payloadElements)` consecutive
logical rows. The output-map stride selects the next numerical row. Each write
stops at the payload boundary; trailing payload elements remain as zeroed by
realization. Vector weights and scalar weights use the same mapping. The caller
owns the claimed pages throughout loading and publishes after loading finishes.
This is configuration-time initialization, not a function invoked during NFE
execution, and the cited paper does not specify model-file conversion.

The bridge currently uses 4096-byte physical pages. A 3840-element FP16 matrix
row exceeds one payload, so merely rebinding its previous dense buffer cannot
realize these maps. Matrix contraction bindings must consume this layout while
preserving the validated backend; the loader alone does not satisfy that work.
The existing caller still uses its prior loaders until the required bindings and
caller migration are complete.

`MatrixView` now describes page breaks along either matrix axis, with transpose
exchanging those axes and slices preserving aligned page origins. This is an
immutable numerical address map, using the same compiler-assigned operand
placement principle. `MatrixOperations` specializes the existing vector and
tensor kernels for weight-page width and physical stride during binding.
Vector loads gather each weight vector from its payload. The tensor kernel
rebases its operand tensor views at each weight-page boundary, with equal
contraction extents on both operands, and retains its FP32 cooperative
accumulator across those reads within one dispatch. Fixed reduction tiles must
divide the payload width. Final conversion remains after the full contraction.
No partial-sum command buffers, dense weight copies, or readiness state are
introduced. These are repository lowerings of indexed gathers and arithmetic;
Monsoon does not describe these Metal APIs. MPS page-break support and measured
numerical/performance validation remain outstanding.

`MatrixOperations.contract` binds one explicit contraction-index interval to
the existing multiplication implementation, writing an FP32 result into supplied
storage. A `MatrixView` slice entirely within one payload becomes an ordinary
strided view of the same bytes; a slice spanning payloads retains its page map.
This lets the configured MPS function address an individual weight payload
without a copy or a backend substitution. The indexed algebra is
`P_j = A[:, K_j] B[K_j, :]`, `C = sum_j P_j`, the local contraction counterpart
of the partial-sum reduction described by Rabenseifner and Patarasuk–Yuan.
Each `P_j` must have its own configured actual FP32 pages and input/output maps.
The binding encodes no sequence of partial contractions and inserts no waits;
each numerical function is independently issuable from its input stamps.
Canonical FP32 page reduction and final conversion remain caller integration
obligations. Splitting a contraction may change rounding and dispatch cost;
neither agreement nor speed follows from compilation.

Apple's [MPSMatrix documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrix)
specifies row-major storage; its
[multiplication documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)
specifies the optional transposes and scalar accumulation algebra. These API
references support the view mapping, not a claim that every input/result type
combination or this decomposition has been operationally validated here.

### Literal page reduction

Rabenseifner (2004) and Patarasuk–Yuan (2009), cited above, supply the
reduce-scatter/all-gather algebra. `mesh_rows_add_f16` adds one matching set of
actual partial pages into claimed FP32 accumulator pages and publishes that
addition's stamp in an actual index page. `mesh_rows_indexed` reads those index
page contents. One arithmetic owner handles each overlapping accumulator/index
span. The accumulator output rows must already have been selected/claimed, and
the index page must have a live configured lifetime. No operation crosses a
payload boundary into an RDMA header. Addition releases each input use after
publishing its accumulator/index values. It does not inspect the index entry to
decide whether to execute again; the configured graph owns exactly one execution
of that addition. Configuration counts hashing and transport uses too.

The addition entrypoints and `mesh_rows_normalize_f32` return no runtime status.
Their arithmetic contains no readiness scans, issued-stamp validation, duplicate
execution suppression or error-return branches. Counts, payload extents, index
locations, positive epsilon, claimed outputs and live inputs are configuration
and call-graph contracts. Arithmetic loops follow those fixed dimensions and
the fixed input order. This does not make the remaining selection, transport,
invalidation or caller implementations compliant; those still require replacement
and integration under the operator's no-added-control-flow instruction.

`mesh_rows_add_f32` applies the same operation to FP32 partial pages, including
the independent contraction partials above. Both entrypoints specialize the
inlined `row_add` at compilation for their source element width; invocation
does not choose an input format from a buffer. The FP32 variant adds directly
into FP32 accumulator pages and publishes the same literal index value, without
an intervening FP16 materialization. Both use the configured input order and
require the same claimed destinations, index-page lifetime and single writer.
The shared implementation avoids duplicating the arithmetic/publication algorithm.
Rabenseifner and Patarasuk–Yuan are the cited partial-sum algebra; they do not
establish numerical equivalence to an unsplit floating-point matrix product.

`mesh_rows_normalize_f32` performs the plain specification's normalization and
residual addition from accumulator, gamma and residual pages into output pages.
Its caller supplies the numerical parameters, proves the full-row inputs from
the index/table, claims outputs and publishes only after the arithmetic finishes.
This is a numerical function, not a model-graph interpreter. The initial C
arithmetic is not a claim that CPU normalization is the fastest backend; the
configured GPU implementation must preserve the same page algebra and be
measured against the validated local function. The published collective papers
do not specify this model's normalization formula.

`metal-microbench/reduce_scatter.swift` supplies `bindRowNormalization` and
`rms_norm_accumulator_rows` for the GPU part of that same page algebra. The
configured maps gather FP32 accumulator elements, FP16 gamma/residual elements
and an FP32 scale from actual local sendable pages through `mesh_row.page`.
The kernel reduces squared FP32 values over the numerical row and scatters the
normalized residual result to FP16 output pages. No materialized FP16 reduced
input lies between addition and normalization. The within-threadgroup barrier
combines the eight SIMD partial sums; it is an arithmetic dependency, not a
transport wait. Papadopoulos–Culler supply the presence/matching principle;
Rabenseifner and Patarasuk–Yuan supply the surrounding collective algebra.
The model supplies RMS normalization, epsilon, gamma and scale.

The binding compiles its pipeline and fixes all operand maps before invocation.
Invocation encodes a supplied ready row range into the caller's command buffer;
it allocates no operand storage and performs no readiness polling or publication.
All four input maps and the output must name local arena pages covered by the
transmit alias. The caller proves full-row accumulator presence and all operand
lifetimes, then publishes on completion. Immutable geometry is encoder argument
data, not a separately allocated shared-memory operand buffer. This binding is
added before caller migration; compilation alone does not establish numerical
agreement or performance, and it has not yet been used in an RDMA evaluation.

These additions precede the caller migration. They do not constitute completed
transport integration, read-proof integration, recovery or RDMA validation.
The operator requires the existing caller to migrate before excluded
implementations and their interfaces are deleted.

### Literal page transport

Papadopoulos–Culler provide the storage-presence mechanism, and
Rabenseifner/Patarasuk–Yuan the page-exchange algebra cited above.
`mesh_rows_send`, `mesh_rows_receive`, `mesh_rows_acknowledge` and
`mesh_rows_return` bind actual pages to the existing SUB/CMP/ACK/REL rings.
`mesh_row_binding` is immutable correspondence between local and peer rows.
Realization requires bindings in increasing local-row order with disjoint
ranges. Receive lookup searches this same configured array by destination row;
it does not allocate a second lookup table. The selected binding still has to
match direction, peer, range and source-row correspondence.
`mesh_row_address` is the page's literal address/epoch/stamp header, not an
application frame-kind or handshake protocol. The transport worker owns those
rings. Each transmitted row has one configured transport use, released by its
actual NIC completion; local computation/checking uses are separate counts in
the same row. No flight array or additional descriptor queue is allocated.

Submission checks the existing page header to avoid sending the same row value
twice. On a full SUB ring it restores the prior header and retries on a later
scan; it does not publish an extra pending flag. The header and payload remain
unchanged through NIC completion. Each transmitted row has one bound adjacent
destination; configurations needing multiple copies must realize their
collective ownership/edges explicitly rather than overwrite an in-flight header.
Generation stamps must not be reused for another value under the same epoch.

Delivery installs the received physical page and configured binding's use count,
then stamps the actual destination. It never overwrites a live destination.
Each arrival restores the configured receive-use count; incoming data cannot
allocate or infer a numerical graph.
`mesh_rows_retire` implements the storage retirement part of Papadopoulos and
Culler’s operand-storage discipline through this specification’s literal use
count. An asynchronous owner scans the existing rows, skips unpublished values
and remaining uses, and zeroes completed values. It owns REL exclusively.
`mesh_rows_return` checks capacity in that existing ring before clearing a
received row. When capacity is absent, the actual page remains in the row;
the scan continues over other rows. When capacity exists, zeroing precedes
publishing the descriptor to REL. The bridge only advances the tail, so capacity
cannot disappear between the check and publication by the sole producer.
No blocking retry, pending list, or separate completion state is involved.
Local pages become absent only after zeroing; their fixed physical addresses
remain in the configured output maps for reuse. This does not yet provide
physical sharing between distinct configured outputs, remote dependent-read
proofs, or the worker integration needed by the caller.

Delivery scans one snapshot of the existing CMP ring and publishes every page
whose destination is available, returning the number published. It removes each
descriptor using the existing ring operation. A busy destination
does not prevent delivery to other available rows. No page or descriptor is moved
to a second pending queue. The single receive owner is required by that ring
operation. Error propagation, remote read proofs and recovery remain integration
obligations. These primitives are not yet used by the NFE.

`mesh_rows_progress` combines NIC-completion release, the receive snapshot,
publication-driven transmission and use-count retirement for the asynchronous
transport owner. Its arguments are the existing page table, fixed epoch and
immutable bindings. It owns no additional persistent state and invokes no model
kernels. Its return value reports activity or the negative receive error; it is
not an operand-readiness signal. This synchronous error path remains noncompliant with the asynchronous metadata
contract above; it must not conclude NFEs inside the callgraph. Papadopoulos–Culler supply the operand
presence/lifetime principle; Rabenseifner/Patarasuk–Yuan supply the page-exchange
algebra. These references do not prove a latency bound for this implementation.

Snapshot erasure is safe with the existing single-consumer ring operation:
removing position `at` moves the old tail descriptor to `at` and advances the
tail. That moved descriptor was already examined in this pass. Every position
after `at` through the captured head remains unexamined and unchanged by erasure.
The producer may reuse released positions, but those lie outside the remaining
snapshot. Busy descriptors remain in the same ring for the next pass. No
persistent cursor, pending list, or caller-owned arrival bitmap is required.

### Literal page checking

Saltzer, Reed and Clark (1984), cited above, motivate endpoint checking.
`mesh_rows_digest` writes the digest of a live page directly into a configured
digest page; it retains no heap hash ring, count or generation cache. The two
hardware CRC polynomials are an implementation choice for error detection, not
a cryptographic claim or an algorithm attributed to those authors. Corresponding
peers must use the same configured seed and numerical payload extent.
`mesh_rows_equal` compares two present, equally stamped pages as a value. The
numerical graph must not depend on its result. A configured post-consumption
function owns the hash read, its output publication and input-use release;
comparison is an endpoint function whose disagreement concludes the NFE with
failure. These primitives alone do not implement replay or link recovery.

The caller's Job/stage control, consumed masks and embedding/head completion
words have been removed in committed revisions of metal-microbench. Distinct
configured functions now have distinct value rows and share the NFE stamp.
Native FFN input, MPS FFN intermediates, MPS attention input and vocabulary
input/output now address sendable payloads. This is not all operand storage:
other backend inputs and weights still require work. MPS/tensor attention Q/K/V
and attended intermediates now also occupy ordinary sendable page payloads;
the corrected indexed output stores were checked against the preceding logits.

Broader local-page recycling, complete read-lifetime proofs, accumulator/index
page reduction, independent admission without digest gates, and link-error
recovery/repetition remain unfinished. Normalized-input slots now share and
recycle configured physical spans after completed dependent reads and
asynchronous zeroing. Other local slots retain separate spans. These citations
provide no exemption.
Do not mark transport and asynchronous
map/reduce complete until both flows use that representation and actual RDMA
measurements establish correctness and performance on the supported workloads.

### FP32 contraction output specialization

`MatrixOperations.contract` requires FP32 destinations, but the initial vector
and tensor bindings rejected that type and their shaders always stored FP16.
That implementation gap is corrected by specializing the output element type
when the numerical binding is realized. The pipeline cache key includes the
output-type specialization. The invocation does not inspect output types or
choose a backend.

The existing vector and tensor kernels now write their FP32 contraction
accumulators directly into FP32 destinations. Ungated contractions do not round
the partial result through FP16. The existing fused gate/up expression retains
its specified FP16 rounding of the two projections before activation; changing
the destination type does not change that nonlinear expression. Ordinary FP16
outputs retain their previous conversion. This adds no scratch allocation,
host accumulation chain, or alternate kernel implementation.

Papadopoulos–Culler (ISCA 1990), cited above, provide configured functions over
assigned output storage. Rabenseifner (ICCS 2004) and Patarasuk–Yuan (JPDC 2009),
also cited above, provide the collective composition into which independent
contraction partials feed. The output precision and Metal specialization are
implementation choices here, not claims made by those publications. This change
does not prove that partitioning every contraction is faster than a full local
contraction, nor validate mixed-precision MPS multiplication or caller migration.
