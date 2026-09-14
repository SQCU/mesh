# Algorithm sources

## Independent verbs progress

Apple's [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
defines nonblocking `ibv_post_send`, `ibv_post_recv`, and polling of their
completion queue. The controller supplies receive credits; application receipts
are not required to submit another independent operation. Work-completion status
reports transport errors to the calling context's metadata.

`mesh-flow.c` gives each direction its own queue-capacity accounting. Each
submission is one literal WR (collective-dependency-ledger.md D1, D3). A failed
post releases the block's occupancy and records the error (D12). Neither
direction retains a pending batch that gates the other.
Every progress pass services both directions and completions with bounded work.
Normal termination enters verbs teardown without a software loop waiting for a
peer to consume committed sends. Driver destruction remains the operation that
releases the device resources.

The removed implementation confused a prepared host descriptor with an occupied
hardware queue and shared that condition across both directions. Its retry flag
could consequently stop receives behind a failed send, or stop sends behind a
failed receive. The descriptor is needed only during the literal post; accepted
work is represented by the device queue and its page-table indices.

## Operand matching and storage

Gregory Papadopoulos and David Culler, *Monsoon: an Explicit Token-Store
Architecture*, ISCA 1990, describe operand matching using indexed storage and
presence state. This is prior art for the function scan, not evidence of this
implementation's measured utilization. Logical row occupancy and physical page
occupancy describe different indexed resources. Reattachment cannot erase
physical ownership held by an outstanding device access.

Rolf Rabenseifner, *Optimization of Collective Reduction Operations*, ICCS 2004,
and Pitch Patarasuk and Xin Yuan, *Bandwidth Optimal All-reduce Algorithms for
Clusters of Workstations*, JPDC 2009, supply the reduction decomposition. They
do not require independently completed partials to wait for a host-side batch
receipt before becoming inputs to reduction.

## Receive storage

Superseded 2026-09-12. A receive is posted on the consumer's own pages (ledger
D4), paired by per-queue order (D5), and its completion is presence (D8). The
receive pool, landed bits, inverse landing index and receive invalidation were
removed with that change; see `collective-dependency-ledger.md`.

## Performance evidence

The acceptance target is less than five percent pipeline stall during the
linear-algebra-intensive interval on real inputs. Timing belongs to the calling
measurement context. A host interval with an outstanding command does not prove
that an accelerator was executing arithmetic. Metal execution timestamps and
host pending intervals must be reported separately; Core ML's opaque execution
must not be counted as measured hardware occupancy. Neither a timing simulation
nor command occupancy alone demonstrates mathematical FLOP utilization.

## Nonblocking table ownership

The indexed operand storage of Papadopoulos and Culler and the literal access
completion defined by TN3205 have different indices: logical rows belong to
functions, whereas backing pages belong to memory accesses. Configuration allocates
from the complement of assigned and outstanding indices. Submission occupies
both the physical source and its logical completion target; completion releases
those indices. Attach does not drain another caller's sends or erase its targets.

Issuing a function marks its output rows as being produced. Presence becomes true
at completion, and reader bits become consumed then. Clearing presence alone was
insufficient: the next scan could otherwise issue the unfinished operation again.

Numerical completion sets reader bits; the bridge reposts a receive on a block
only after its configured reader mask is satisfied (ledger D9). An outstanding
work request keeps its rows and pages occupied until its completion, or until the
bridge destroys the queue pair that held it (D14). No callback waits,
reference-count drain, acknowledgement, or generation check is needed.

## Registered-span defect, closed

The provider registered aligned one-GiB address extents while a multi-page
block selected its key from its first byte, so a block could cross a
registration boundary; and a registration must not cross a 4 GiB
virtual-address boundary, the bank alias recorded in
[RDMA-KERNEL-RECOVERY.md](RDMA-KERNEL-RECOVERY.md). Both hold now by
construction: the bridge maps the region at a 4 GiB-aligned base and the data
origin is a multiple of the block; a power-of-two block gets 1 GiB regions from
the base, which divide the bank and which no block straddles; any other block
gets block-aligned regions from the data origin and the bridge refuses, saying
why, a mapping that reaches a bank boundary. The resolution is measured in the
next section.

## Regions follow blocks

A block is posted as one scatter-gather element whose key comes from its first
byte. Memory regions are therefore cut at block-aligned offsets from the data
origin — region 0 is the header, every data region spans a multiple of the
block — so a block is inside one region by construction. Regions cut at
absolute address boundaries put the tail of a straddling block outside its
key's region; the tail is where the tag lives, so the landing arrived with a
clean completion and no tag, and the bridge rejected it. Measured 2026-09-10
on the M5 (two regions: every landing from the M4 rejected) against the M4 (one
region: every landing accepted); with block-aligned regions both directions
carry every block (160 sent, 160 received, 0 bad on each side). A rejected
landing prints page, status, bytes, tag and base to the bridge log.

## Attach takes over a dead holder

The client word names the process that owns the region. A process that died
without detaching left its pid there and every later attach was refused: a
node that looked provisioned and was not. Attach now takes the word from a
holder that `kill(pid, 0)` reports gone and performs the release the dead
client never did — bases absent, row and page ownership cleared. A live holder
is still refused; the check prevents a demotion and never is one.


## Registered memory views

Apple's mmap MAP_SHARED mapping mechanism permits multiple virtual views of the
same shared-memory file pages. mesh_view_create reserves virtual address space
and maps configured physical-page runs from the existing mesh shared-memory fd
into it; mesh_view_destroy releases that view. No payload is copied and no new
registered memory is allocated. This is configuration-only address realization,
not a receiving-side remapping operation or an invocation-time allocator.

The two installed Thunderbolt providers report max_sge=1 through ibv_devinfo.
The view therefore joins numerical payload pages in virtual memory while the
bridge continues to submit their original registered addresses. Logical indices,
dense tensor offsets, and registered addresses remain separate representations
of the same underlying bytes.


## Independent configured programs

Papadopoulos and Culler, *Monsoon: an Explicit Token-Store Architecture*, ISCA
1990, separate indexed operand ownership from the functions that use the
operands. Each configured program here owns disjoint logical rows, backing
allocations and its own queue pair order (ledger D5). Realizing a
second program begins with the table's existing reader masks, preserving the
readers already assigned to the first program. Newly allocated rows have zero
reader masks through the existing allocator.

`mesh_rows_release` clears only its logical ownership range.
`mesh_arena_release` clears only its physical ownership range; transport HOT
indices remain occupied until intrinsic completion. Neither detaches another
program, waits for a peer, or clears another program's reader registration.

The Swift caller shares one region storage object per configured region name.
Its weak registry does not retain unused regions. Configured values and graph
completion closures retain that object through their existing lexical owners;
only its final destruction detaches and unmaps the client region. Closing a
calling context drops that context's cached values, without detaching surviving
programs. Numerical callbacks retain their graph, so graph-owned values cannot
be retired before their final callback completes. This is storage ownership,
not a new execution readiness condition or a polling mechanism.

## Literal weight pages

Papadopoulos and Culler, *Monsoon: an Explicit Token-Store Architecture* (1990),
provide the indexed operand-store model used here. Configuration resolves the
parameter values and their ownership before numerical invocation. The port's
strided f16/f32 parameter reader materializes each declared shard into its actual
registered operand pages during configuration. Invocation neither consults the
original descriptor nor substitutes a file-backed parameter for a supplied one.
Apple's shared mapping mechanism described under registered memory views supplies
contiguous virtual tensor views over those same backing pages; it does not create
a second payload store. This citation identifies the storage and execution
separation, not a claim that Monsoon specifies today's tensor ABI or weight format.

## Configured binding identities

Superseded 2026-09-12. Binding identities now only order blocks within a
program's queue pair (ledger D5); the receive table, transfer tag, slot reuse
and identity reservation were removed with the receive pool (D4, D14).

## Column and row tensor composition

Shoeybi, Patwary, Puri, LeGresley, Casper and Catanzaro,
[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
(2019), describe intra-layer tensor parallelism. The port uses its complementary
matrix partitions: participant p computes `Z_p = X A_p` from its output-column
slice of A, then `Y_p = Z_p B_p` from the matching input-row slice of B.
Only `sum_p Y_p` requires distributed reduction. A column result is a local
numerical value; no all-gather is inserted between these matching contractions.
Papadopoulos and Culler's indexed operand matching supplies the execution rule
for these configured matrix functions. The existing local MatrixOperations
implementation and specialization remain the arithmetic owner.

FP32 interface values are explicitly converted at numerical boundaries when the
configured model uses FP16 operands. Conversion is an indexed elementwise function
over actual mesh pages, with one dependency extent per configured output extent.
This supports FP32 input/output representation; it does not claim an FP32 model
or replace the configured backend precision. No shadow copy or runtime backend
selection realizes the conversion.

## Online percentage moments

B. P. Welford, [“Note on a Method for Calculating Corrected Sums of Squares and
Products”](https://www.tandfonline.com/doi/abs/10.1080/00401706.1962.10490022),
*Technometrics* 4(3), 419–420 (1962), supplies the constant-storage update:
`n += 1; delta = x - mean; mean += delta/n; M2 += delta*(x-mean)`.
Calling-context percentage reports carry count, mean and sample variance
`M2/(n-1)`. Mean is null for an empty stream; variance is null until two samples
exist. Percentage inputs use the 0–100 scale, so variance has squared
percentage-point units. Warmup observations remain excluded.

These accumulators belong to reporting, outside numerical functions and mesh
transport. Updating moments neither changes a percentage's numerator/denominator
nor adds any readiness condition, traffic or device synchronization.

## Streaming algebra

Gregory Papadopoulos and David Culler, *Monsoon: an Explicit Token-Store
Architecture* (ISCA 1990), provide the indexed storage and presence mechanism
used by the algebra header's extent views, static functions and completion
publication. Rolf Rabenseifner (ICCS 2004) and Patarasuk–Yuan (JPDC 2009), cited
above, provide the reduction/all-gather composition. Jack Dongarra, Jeremy Du
Croz, Sven Hammarling and Iain Duff, *A Set of Level 3 Basic Linear Algebra
Subprograms* (ACM TOMS 1990), supply the matrix contraction interface precedent.
MPS performs the contraction; mesh does not implement a replacement GEMM.

Size Zheng et al., [TileLink: Generating Efficient Compute-Communication
Overlapping Kernels using Tile-Centric Primitives](https://arxiv.org/abs/2503.20313)
(2025), and [FLUX](https://arxiv.org/abs/2406.06858) (2024), motivate exposing
independently usable numerical tiles while retaining efficient compute loops.
The JAX authors' [Pallas collective matmul](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)
illustrates pipeline callbacks and source-lifetime extension; their
[nested pipelines](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html#nested-remote-and-local-dma-pipelines)
separate transfer blocks from local compute tiles. These CUDA/TPU mechanisms
are prior art, not proof of equivalent Metal intrakernel visibility.
Apple's [Metal synchronization events](https://developer.apple.com/documentation/metal/about-synchronization-events)
order completion after preceding GPU commands. The first algebra implementation
publishes at command completion and uses the existing canonical present/read
bits. It makes no claim of intra-dispatch publication.

Saltzer, Reed and Clark, *End-to-End Arguments in System Design* (ACM TOCS 1984),
provide the endpoint acceptance precedent: the example checks the complete
numerical composition and observes early output with an unrelated input absent.
The explicit formulas, shape checks, allocation/view lifetime, deterministic
fixtures and numerical comparison implement that contract as described in
[streaming algebra](streaming-algebra.md). They are not algorithms attributed to
those papers. Welford's update above supplies the example's timing statistics.

The bridge's link_receive factors the existing D2/D4 receive posting into one
function used both in setup and ongoing progress. Apple's TN3205 supplies the
RTR/RTS and posted-receive mechanisms; the operator explicitly authorized the
fixed setup boundary in ledger D16. Initial receive windows are posted after
configuration and synchronized before enabling sends. This is a connection
initialization requirement; no extra tensor-firing predicate follows from it.

## Pallas indexed destinations

The JAX authors' [Pallas collective matmul](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)
allocates a distinct scratch slice for each incoming shard and forwards to an
explicitly indexed remote reference. Its storage choice avoids the backpressure
that would follow from reusing fewer scratch slices. Their
[TPU buffering discussion](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html#double-buffering)
explains how additional slots accommodate additional run-ahead.

mesh_algebra_copy implements a common indexed source/destination declaration.
Every participant enumerates every occurrence; the local participant projects
its send, receive or local copy from that same occurrence. The example's configure
function expands invocation slots into distinct registered tensors and numerical
functions, with shared immutable weights. Its main function produces ahead into
those slots, observes independent outputs before supplying a withheld input, and
reuses each slot after consuming its own outputs. No extra per-message protocol
or numerical scheduler implements this buffering. The algebra and FIFO lowering
proof are in [streaming algebra](streaming-algebra.md).

## Mandatory partial publication

The JAX authors' [Pallas collective matmul](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)
places communication in the numerical pipeline callback. Zheng et al.'s TileLink
and FLUX, cited above, describe tile-level producer/consumer composition.
Dongarra et al.'s Level 3 BLAS supplies the contraction decomposition:
C[I,J] = alpha A[I,:] B[:,J]. Monsoon supplies indexed input presence and
completion publication. These are the sources for dependencies, compare_maps,
bind_part, emit_part, automatic mesh_algebra_bind decomposition, and
mesh_algebra_return_part. The sort merges intersecting page dependencies; it
creates no runtime ordering or scheduler.

Binding now decomposes a logical output into independently published parts.
Local output parts occupy one page; transferable parts occupy one configured
SEND block. Rectangular subviews cover each part, including row-boundary tails.
Only input pages addressed by those subviews are dependencies. MPS descriptors,
Metal specializations and encoding closures are realized before invocation.
Each issued part unconditionally completes through mesh_complete, which publishes
its output and marks configured sends hot. There is no caller-supplied optional
send hook. A completed part can be transferred and consumed while another part
of the same logical operation still lacks input. Metal completion is the actual
visibility boundary; this does not inspect the internal tiles of an MPS dispatch.

Paired copy declarations enumerate block index before indexed extent occurrence.
Both participants derive identical SEND/RECV sequences. This prevents a missing
tail of one indexed operand from preceding every block of the next operand in
that declaration. Independent algebra edges may use independent configured
queues; no claim is made that a FIFO can bypass an earlier missing message.
The acceptance uses a third queue for contraction output, since its producer
queue can legitimately still lack the input tail. A returned first output part
must arrive and match the numerical formula before the missing input tail is
written. This check detects whole-operand publication stalls that completed
whole-program numerical comparisons alone would miss.

## Backend-independent producer and consumer streaming

[Three worked backend examples](backend-streaming.md) apply the indexed dataflow
and BLAS decomposition sources above to MPS, an ANE adapter design, and a symbolic
NumPy-style Metal frontend design. The symbolic frontend uses NumPy's documented
einsum index notation; it is not a claim that eager NumPy streams device values.
The distinction between final indexed regions and reduction contributions makes
both early consumption and valid publication explicit. The examples are mesh
lowering designs derived from that algebra, not claims that Apple or NumPy ships
these mesh integrations. Their implementation and validation status is stated
individually in the document.

`mesh_algebra_contract` lowers corresponding indexed operand partitions into
FP32 contributions and a fixed adjacent-pair reduction tree, using the BLAS
contraction identity above. Odd tree levels carry their unpaired value forward.
The tree and all contribution storage are realized before invocation. The returned
contribution tensor permits ordinary mesh transfers and consumers to name those
values; the caller no longer constructs intermediate contractions and additions.
The existing acceptance orders the available K-panel's transfers before the
intentionally absent panel and observes its complete contribution at the peer.

`mesh_numpy.Array`, its NumPy dispatch hooks, the elementwise constructors,
`einsum`, and `emit_c` implement setup-time symbolic expression capture and
postorder lowering to existing mesh algebra calls. Scalar scale/shift composition
becomes one affine operation. Named intermediate outputs preserve their mesh send
bindings. Contraction operands are corresponding indexed K partitions, and the
shared contraction lowering owns contribution allocation and reduction. The
current frontend supports pointwise operations feeding a matrix contraction;
it does not interpret a graph during numerical invocation.

## CoreML partial execution

Dongarra et al.'s indexed BLAS decomposition and the partial-publication contract
above define the numerical regions used by `native_array`, `native_part`,
`mesh_algebra_coreml`, and `mesh_coreml.compile_part`. Each compiled Core ML
program performs only the rectangular contractions belonging to one publication
part, flattening and concatenating their outputs directly into that part's supplied
MLMultiArray backing. Inputs, including weights, are named views over registered
operand storage. The generator compiles shape specializations during setup;
no compilation, feature binding or operand allocation is performed by mesh during
invocation. Core ML is configured for CPU and Neural Engine execution.

Apple's [prediction and compiled-model documentation](https://apple.github.io/coremltools/docs-guides/source/model-prediction.html)
and [outputBackings contract](https://developer.apple.com/documentation/coreml/mlpredictionoptions/outputbackings)
provide the API mechanism. Core ML may decline a proposed output backing; an
identity mismatch is reported as a numerical execution error and does not publish
success for unwritten canonical pages. No copied-output fallback is introduced.
The common execution completion now owns successful publication for both Metal
command buffers and asynchronous Core ML predictions. Setup-time compute-plan
inspection reports preferred Neural Engine operations; it is not an execution
trace. Output object identity establishes the public backing endpoint, not the
absence of internal Core ML copies or the device placement of every operation.

The `astype` symbolic operation lowers to the existing affine kernel with an
explicit destination scalar type. The acceptance's typed tensor construction and
activation reference declare FP16 rounding before contraction for the native
half-input case. FP32 retains its original 2e-4 contraction error threshold;
the separate FP16 case uses a 1e-3 threshold and reports its scalar mode and
observed maximum error. This does not authorize silently narrowing FP32 programs.


For explicitly FP16 contraction operands, the Core ML numerical specialization
uses at most 32 terms per native dot product, casts each result to FP32, and adds
those results in a fixed tree. This bounds the length of half-precision accumulation
before promotion and prevents the native compiler from narrowing the FP32
combination. These are local numerical subexpressions of an already-ready mesh
part; the user-visible K contributions and their registered storage remain owned
by `mesh_algebra_contract`. The 32-term choice is an accuracy specialization,
not a measured throughput optimum. The initial unsplit FP16 native dot product
failed the existing FP16 acceptance threshold; the threshold is unchanged.


The explicit FP16 case compares each participant's result independently to the
quantized-input numerical reference: heterogeneous native kernels may round
differently. It does not require their locally computed answers to be bitwise
identical. Its additional peer-contribution comparison permits the sum of the
two per-result error bounds. FP32 retains the existing exact peer-replica check.

## CPU indexed execution

Papadopoulos and Culler's Monsoon (1990), cited above, supplies the indexed
presence/completion mechanism. Dongarra, Du Croz, Hammarling and Duff's Level 3
BLAS (1990) supplies the contraction equation. `create_algebra`,
`mesh_algebra_create_cpu`, `cpu_operand`, `cpu_get`, the typed CPU loads/stores,
and `cpu_part` lower those indexed functions to direct scalar-addressed CPU
arithmetic. Scalar type and numerical operation are resolved during configuration.
All writes complete through the same mesh publication owner as the other backends.
The [CPU streaming and copy audit](cpu-streaming-and-copy-audit.md) gives the
address equation, lifetime argument and exact scope of the no-staging proof.
The acceptance's CPU option uses the existing endpoint numerical comparison
(Saltzer, Reed and Clark, 1984), not a separate evaluator or copy-detection experiment.

## Indexed library functions

Papadopoulos and Culler's Monsoon (ISCA 1990), cited above, supplies the
presence-bit firing and last-reader release mechanism. The JAX authors' Pallas
indexed block references supply the configuration-time mapping from grid indices
to numerical regions. Dongarra, Du Croz, Hammarling and Duff's Level 3 BLAS (1990)
supplies the partitioned contraction equation; adjacent-pair addition implements
its explicit reduction. These are mechanism citations, not claims of API identity
or measured performance equivalence.

`mesh_algebra_function`, `bind_dependencies`, `output_used`,
`complete_function`, `complete_part`, and `mesh_algebra_export` bind application
functions to canonical input/output rows and complete them through the existing
publication owner. A submission's completion context is its configured function,
not a newly allocated job or a separately scheduled readiness object. Physical
completion publishes that function's output regions and retires its input reads.

The `mesh` Python package's `Program`, `Tensor`, `Ref`, `BlockSpec`, and `Result`
methods configure this same mechanism. Tensor arithmetic enumerates independent
indexed block functions during setup. Contraction enumerates `(i, j, q)`
contributions before reducing `q`; reductions similarly expose their contributions.
Transpose and slicing construct strided views of the same mapped pages. Grid
calls bind user numerical functions with exactly their declared input regions.
Native grid preparation realizes submission bindings before invocation, while
synchronous NumPy calls bind their borrowed arrays before invocation. The ctypes
ABI declarations, error translation, and setuptools build methods are the host
language and packaging adapters for that mechanism, with no independent scheduler.

The [library contract and source trace](indexed-library.md) specifies ownership,
publication granularity, asynchronous completion, and the limits of overlap claims.

`bind_copy` implements an explicitly requested same-participant copy by disjoint
output parts, reading source pages and writing destination pages directly. It
preserves bytes, including integer indices and floating-point bit patterns, and
publishes each destination part through `complete_part`. This is the copy
operation's payload write, not a hidden transport staging copy.

## Region streaming review

The JAX authors' Pallas software pipelining and indexed reference mechanisms,
and Papadopoulos and Culler's Monsoon presence/read ownership, cited above,
supply the mechanism for `output_region`, `overlaps`, and `BlockSpec.region_map`.
Configuration maps disjoint writable numerical regions onto existing publication
quanta; physical completion publishes only the selected rows. Input/output
intersections are checked during setup to prevent publication of unwritten or
concurrently overwritten data. No new execution state is introduced. See the
[source and literature review](region-streaming-review.md) for the exact scope
and the remaining reduction and contraction dependencies.

The same region mechanism applies to built-in `mesh_algebra_bind` and its backend
lowerings: numerical indices remain relative to the selected view, while output
addresses and publication rows include the view offset. Disjoint destination
sections have independent producers and completion. The region streaming review's
follow-up records the operator's explicit partial-kernel composition and its
section-local reduction dependencies.

## Streaming overlap measurement

The JAX authors' [Pallas pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html)
motivates `examples/streaming-overlap.py`; Amdahl's AFIPS 1967 analysis of
serial fractions motivates end-to-end reporting. `configure` uses the existing
`Program.kernel_call(kernels.matmul)` with matching `BlockSpec` partitions for
`(X W) V`. Both participants allocate the same tensors and peer placement binds
only their own numerical functions. The first contraction runs on participant
zero, the second on participant one, and results return through canonical mesh.
Both directions use queue zero. CPU and Metal use their existing native matmul
bindings; no Python numerical callback or observer runs inside a contraction.

`run_batch` is the external input supplier/result observer. A fixed setup-allocated
slot count bounds concurrent invocations and operand storage. It waits for each
slot's complete returned result before supplying another invocation there, while
other slots continue independently. Five window traversals warm up and drain
before the measured batch. Inputs vary by invocation. Terminal results are
compared against a precomputed reference after the timed interval; intermediate
invocations are timed but not numerically checked. Welford's online `moments`
reports count, mean and sample variance. A timeout reports a stalled observation
without introducing a dependency into numerical execution.

Whole mode is an explicit whole-region comparison experiment, not a second
backend interface. Streamed mode publishes each row region independently, including
a clipped final region. Host-observed first-region and complete-result times are
upper bounds that include input supply and observer delay. `Program.trace` and
`Program.transfer_trace` retain native computation and transport timestamps for
overlap analysis outside the numerical body. Their latest-occurrence retention
must be respected for repeated slots; host intervals do not prove hardware
occupancy or globally earliest publication.

See [the source migration](streaming-overlap-native-2026-09-13.md) for exact
validation limits and the difference from the former NumPy callback comparison.

## Pallas call ergonomics

The JAX authors' [Grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
and [pallas_call reference](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.pallas_call.html)
supply the user-facing decomposition into kernel, grid, operand-independent block
specifications, and output shape/dtype declarations. `ShapeDtypeStruct`,
`BlockSpec.bind/resolve`, `Tensor.region`, and `Program.kernel_call` realize this
configuration through the existing region submission path. An index map produces
block indices, multiplied by block shape to obtain element offsets. Binding and
output allocation happen before execution. Kernel return uses the existing
completion owner to publish the region. The overlap example's ordinary kernel
and index maps now use this interface instead of manual ctypes submissions.

This is a two-dimensional host NumPy kernel interface with mesh publication,
not a JAX tracing backend or full Pallas compatibility. Boundary regions are
clipped instead of padded with masked lanes. A region must fit one configured
backing block, and writes still own complete publication quanta. No hidden
operand gather/copy is introduced when a requested region crosses allocations.

## Single kernel interface

The JAX authors' Pallas call/BlockSpec decomposition, cited above, is the single
public numerical submission interface. `mesh.kernels` supplies backend-realized
numerical kernel descriptors for that call, preserving the existing CPU, Metal,
MPS, and configured Core ML implementations. `affine` supplies scalar constants
at configuration. These descriptors do not introduce another graph or invocation
interface. Tensor arithmetic overloads and public per-operation bind/contract
methods are removed; ctypes submission remains private runtime machinery.

## Streaming FFN

Dongarra et al.'s Level 3 BLAS partitioned product, Blelloch's associative
reduction, and the JAX authors' Pallas region pipelining are cited above.
Ramachandran, Zoph and Le, *Searching for Activation Functions* (2017), define
swish `x * sigmoid(x)` (https://arxiv.org/abs/1710.05941).
`mesh.nn.ffn`, `_linear`, `_sum`, `_pointwise`, and their index maps compose
only `kernel_call` operations. They form first-linear contributions along K,
combine them per hidden section, apply swish to that completed section, form
second-linear contributions, then combine each output section. Optional exchange
is a setup-time binding function over the activated tensor. Swish is not
distributed across addition. Independent sections retain independent completion;
no numerical function polls readiness or waits for another launch.

## Memory warning

Apple's XNU `rusage_info_v2.ri_phys_footprint`, libproc process enumeration,
and Mach `host_statistics64` provide the accounting used by `mesh_memory_warning`
and its descending footprint comparator `mesh_memory_order`.
Source: https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/resource.h
and https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/mach/vm_statistics.h.

Before allocating the shared arena, mesh prints and flushes stdout when its full
layout exceeds 80% of physical RAM or would leave less than 20% of RAM in the
free-plus-inactive memory estimate.
Attaching clients repeat the above-80%-RAM warning without counting the existing
arena as new physical storage. Views alias that arena and are not new allocations.
The warning precedes truncation, mapping, and registration. It does not deny the
requested allocation. Unknown accounting emits a warning as well.

The process list is sorted by physical footprint, including compressed charges,
and continues through 80% of the sum of readable other-process footprints. It excludes
the caller and reports unreadable processes. Process sums can overlap shared
accounting and omit kernel charges; neither they nor reclaimable inactive pages
guarantee allocation headroom. These are setup diagnostics, never numerical
invocation dependencies. This guard covers canonical mesh arenas, not unrelated
application allocations such as the stopped metal-microbench decode process.

`mesh-flow --memory-check` takes the ordinary arena geometry and reports without
allocating or opening verbs. The bridge launcher runs it before starting launchd,
so warnings are visible at the launching terminal as well as the daemon log.

## Publication layout

The JAX authors' Pallas block ownership and pipelining decomposition (cited above)
requires independent output regions to have independent publication ownership.
`mesh_algebra_publication_bytes` exposes the realized transport block size during
setup. `kernel_call` retains contiguous backing for aligned complete row stripes;
it allocates separate padded canonical extents for column tiles and short or
unaligned row stripes. A small `row_sum` output can therefore publish independently
without sharing a publication quantum with its neighbor. No copy is introduced,
and no layout choice occurs during a numerical invocation.

## Application Metal kernels

The JAX authors' Pallas call/BlockSpec split (cited above) separates numerical
source from storage realization and completion. Apple's Metal argument buffers
and `useResource:usage:` declarations bind GPU addresses to resident buffers
(https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/useresource(_:usage:)).
`kernels.Metal` and `MetalDispatch` describe compiled application kernels through
the same `kernel_call` as built-in operations. Buffer zero contains GPU addresses
of the bound input refs followed by output refs, already adjusted for ref offsets.
Constants occupy buffers one onward; each dispatch can select an offset within
buffer four. The descriptor owns no scheduler.

`mesh_algebra_metal` realizes pipelines, constants, address tables, resources, and
dispatch geometry during setup. Its addresses reference the existing canonical
Metal buffers over registered shared pages. Only descriptors are copied.
`submit_metal` uses the native function's physical completion callback to publish
exactly its configured output regions. Multiple dispatches within one region may
initialize and accumulate a reduction; independently published regions are
separate kernel calls. No command completion wait is introduced.

## Xonotic planner migration

Dongarra et al.'s partitioned matrix products and the JAX authors' Pallas region
pipelines, cited above, implement the original Xonotic routed-expert planner.
`model`, `solve`, and `main` in `xonotic/planner/plan.py` preserve routing by maximum
score, selected-expert ReLU FFNs, objective projection, and the position update.
`kernel_calls` lowers the existing application operator source through `kernels.Metal`
and `kernel_call`. Each bot chunk has independent input and return allocations.
Width is now explicit (`--width`, default 256), rather than inferred from frame
transport capacity. `--tile-rows` controls the independently processed bot chunks.
This is a numerical migration, not a claim of equal or improved throughput.

A four-row, width-32 local evaluation selected experts [4,1,4,6] and had maximum
absolute projection error 1.6695066860222818e-7 against the original algebra.
Review uncovered an integer reduction identity bug: converting floating infinity
to an integer could corrupt expert selection. Integer min/max now initialize with
their representable extrema. The undefined `rows` shader argument was also removed.
The new custom Metal binding independently published an int64 input region plus
seven while the second region remained absent, preserving values above 2^40.

## Xonotic frame migration

The JAX authors' region ownership and the canonical configured SEND/RECV ring
(cited above) underlie the application-local `Frames` adapter and `Batch` views.
It handles game/configuration framing, not numerical scheduling. `reserve` borrows
canonical output regions, `send` ends their producer writes, and `read` yields
canonical result regions and consumes them after the caller returns. Scatter
`recvmsg_into` fills these output arrays directly, including across ring wrap.
`mesh_writable`, `mesh_tensor_writable`, and `Ref.writable` expose the existing
producer ownership predicate for host I/O; they add no numerical readiness state.
Multiple Program owners share the process's attachment until the last closes.

## Presence-driven execution

Dennis's dataflow firing rule and the JAX authors' Pallas region pipelines
(cited above) motivate fixed reader adjacency established during realization.
`mesh_execution_add` installs those relationships; `mesh_notify` marks changed
page indices. `mesh_events` visits affected functions and `mesh_fire` submits
eligible region computations. `submit_ready` executes CPU and Python callbacks
on worker contexts, so a numerical callback does not occupy the presence
handler. CPU float32 contractions use Dongarra et al.'s BLAS SGEMM on existing
canonical buffer addresses; matrix bindings are constructed during setup.

This implementation checkpoint uses a shared dirty bitmap and a nonblocking
Unix datagram wake for the consumer handler. It has not yet established its
operational wake-loss and teardown behavior through distributed execution.
The cancellation completion wait is confined to destruction, after numerical
owners have drained; it is not a tensor dependency.

The obsolete Xonotic row-table binding and its generated native encoder are
removed. Alias/adoption compatibility machinery is also removed. Canonical
extent resources have one owner, including partially constructed allocations.
The later explicit-metadata audit removed stale scan callers and the obsolete
persistent policy binding. The composed gold demonstration is recorded below.

## Async index push contract

The operator's September 13 clarification distinguishes publication ordering
from computational barriers. Producers publish independently usable sections
and submit page-table/target index pairs. A dedicated polling transport thread
pushes those indices while producers and consumers continue numerical work.
Queue entries describe canonical registered pages; they do not copy operands.
Consumers depend on the sections used by their particular operation, not the
arrival of unrelated sections or completion of the whole tensor.

`mesh_transfer` retains local row/page, peer row/page, and the binding's explicit
identity/offset. Setup exchanges receive descriptors, checks binding identity,
and stores the peer's actual row/page alongside the local source. Differing
local allocation addresses are therefore retained rather than reconstructed
from send completion count. Payload arrays remain in their registered pages.

`mesh_progress` visits ready descriptors, skipping unpublished entries rather
than ending the queue walk. It batches these small index tuples into a separate
RDMA SEND/RECV queue. `link_receive` uses received target row/page indices to post
payload receives directly into canonical consumer buffers. Payload SENDs are
posted immediately after their descriptor batch, with no software acknowledgement
or receiver-ready message. The descriptor path transfers indices only, never
operand data. This replaces the former static target inference in ledger D5.

Apple TN3205 states that SEND processing uses receive credits. This permits
posting payload SENDs before the peer has handled their index descriptors.
A dedicated descriptor queue has preposted buffers and continues progressing
independently of payload queues. One additional QP is used (at most nine total,
within TN3205's limit of ten). Descriptor storage is allocated and registered
with the arena during setup and included in its memory warning calculation.
Each index message is one 4096-byte frame and holds up to 127 transfer tuples;
partially occupied batches have this same frame cost. Payloads retain the
configured transport block size. Queue counters track outstanding work requests
and descriptor storage reuse, not the destination of a numerical result.

`link_configure`, `link_index_offset`, `link_indices`, `link_post`, `link_receive`,
`link_queue`, `link_error`, and `mesh_progress` implement this software push
mechanism using the MPI asynchronous communication agent and NCCL proxy principle
cited in ledger D2, and the two-sided SEND mechanics of TN3205. Reuse still
requires a target's existing readers to finish before it can be overwritten;
finite pipelines should allocate distinct buffers for independent value instances.

The FFN → RMSNorm → summed learned embedding → FFN → RMSNorm distributed
operational demonstration completed on both Macs at source `602f3cd`; see
[indexed gold observations](indexed-gold-2026-09-13.md). Three invocations
completed section one before section zero was published on a single data queue.
This does not establish a throughput improvement or a per-kernel timing claim.

The generator publishes a completed compiled-model directory using an atomic
rename from a temporary sibling directory. An already published content-key
artifact is reused, never removed while another Program may be loading it.
Concurrent identical configurations can compile independently during setup and
converge on the same completed artifact without introducing a numerical lock.


## Explicit operand metadata

NumPy's ndarray shape/stride representation and the Pallas `BlockSpec` programming
model retain tensor identity and layout as data. The Xonotic numerical compiler
keeps each kernel's graph operand IDs while generating its source, binds only
those operands and its output, and carries actual canonical-buffer strides in
the dispatch view. Logical flat indexing remains separate from physical strides;
read/write address lowering applies the latter explicitly. Setup creates an
ndarray view and assigns its symbolic shape without copying any operand payload.
A shape that cannot be represented by strides fails at setup rather than silently
copying the tensor. One registered buffer pointer names each bound region, so
addressing never infers another operand pointer by dividing a byte offset.

Sources: [NumPy ndarray strides](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html),
[Pallas BlockSpec](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html).

## Streamed normalization and embedding

Biao Zhang and Rico Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467),
NeurIPS 2019, define normalization by the root mean square of a vector. `nn.rmsnorm`
composes square, row reduction, affine epsilon/width scaling, reciprocal square
root, and two broadcast multiplications. Each configured row region depends on
its own complete feature reduction; it never depends on another row region.
Broadcast operands retain their registered pages through zero-stride views.

`nn.embedding` is an indexed gather expressed with Pallas-style BlockSpecs:
a constant lookup table and one index region produce one independently usable
output region. The CPU gather writes each selected table row directly into its
canonical destination row. `nn.summed_embedding` adds those independent gathered
values to its input using the existing pairwise reduction composition. Shape,
dtype, row partition, and all intermediate storage are realized at setup.

[NumPy basic indexing](https://numpy.org/doc/stable/user/basics.indexing.html)
defines selected-row views; [Pallas grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
define the indexed-region composition used here. There are no additional
numerical schedulers, invocation state objects, or operand payload staging areas.

## Direct indexed gather

Harris et al., [Array programming with NumPy](https://doi.org/10.1038/s41586-020-2649-2),
provide the array programming basis. The embedding gather uses basic integer row
indexing and [numpy.copyto](https://numpy.org/doc/stable/reference/generated/numpy.copyto.html)
to write each selected source row into the configured output row. Both row selections
are views; no advanced-indexed dense gathered array or temporary output tensor is
constructed. The copy is the requested gather operation's output write. Publication
and dependencies belong to the enclosing region kernel call, so each row region
can complete while unrelated input index regions remain absent.

Source review of the CPU contraction confirms its matrix descriptors are built
at configuration time. Each descriptor retains canonical extent pointers plus
view offsets, transpose flags and leading dimensions. A transposed input uses
its retained column stride as the physical leading dimension; a nontransposed
input uses its row stride. The output pointer names the current publication
section directly. Execution calls Accelerate SGEMM on these bindings without
constructing an operand array or staging output. The ILP64 interface retains
64-bit dimensions instead of silently truncating the configured size to int.
This source conclusion covers mesh-owned storage and calls, not private packing
inside the BLAS implementation.

The same explicit-operand rule applies to placement. `kernel_calls` resolves
owner zero from its `root_peer` setup argument and other owners from retained
graph-region peer declarations. A cross-owner value has an explicitly allocated
replica indexed by `(value ID, destination peer)` and a canonical copy edge.
Declared output allocations are identical on each participant; the public
`kernel_call(peer=...)` parameter configures numerical execution only on its
owner. Transposed replica views retain the source orientation. No graph protocol
or second numerical scheduler is involved. Distributed callers pass the same
root-peer identity and graph to all participants during setup.

## Declared partition and ownership retention

The JAX authors' BlockSpec and Ref interface, cited above, separates declared
regions from their numerical implementation. `Tensor.broadcast_to` retains
backing references and represents singleton-axis broadcasts with zero strides.
It allocates only view metadata during setup. `Program.kernel_call` now retains
its declared output block partition even when adjacent blocks would align; it
does not silently coalesce independently exported blocks into one whole ref.
The optional `peer` retains configured numerical ownership while realizing
output storage on participants, allowing compiler-owned graph regions to lower
through the same public call and copy operations.

The existing `examples/streaming-algebra.py` operational program now composes
FFN, RMSNorm, two learned indexed embeddings summed with the normalized value,
a second FFN, and a second RMSNorm. It allocates distinct value instances before
realization, partitions rows explicitly, and sends both directions on data queue
zero. Section zero is withheld until section one finishes the entire chain.
Terminal observation loops do not submit numerical work. Reference evaluation
and result comparisons happen outside the numerical graph. Per-run times are
host-observed latencies, not per-kernel timings or a speedup claim.

## Pallas panel composition

The JAX authors' [Pallas matrix multiplication](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html)
uses separate output-row, output-column and reduction dimensions. Its
[software-pipelining derivation](https://docs.jax.dev/en/latest/pallas/pipelining.html)
identifies false dependencies caused by reusing one buffer and removes them with
multiple buffers. The [collective matmul example](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)
composes communication with locally tiled matrix computation.

`nn.linear` realizes explicit M/N output tiles and K input panels. Each K panel
writes its own canonical output tensor; a pairwise addition tree consumes only
matching M/N partials. No mutable shared accumulator or launch-order counter
stands in for a value. `tile_rows`, `tile_k` and `tile_columns` are setup arguments;
the output Tensor reports its realized `block_shape`. Greatest common divisors
respect repeated existing backing boundaries; a single dense backing does not
force its total dimensions into that calculation. Ragged tails use BlockSpec's
existing bounded region resolution.

The FFN applies swish after the required K contributions for its own hidden tile.
The down projection consumes each completed hidden feature tile as an independent
K contribution; it does not require other hidden tiles to begin computing.
Pointwise functions and their reduction trees retain both row and feature
partitions. All of these buffers and dependency edges are realized before invocation.

RMSNorm fuses each feature panel's square and row sum into one expression kernel.
Its row statistic is a sum of those explicit partials. The final output kernel
fuses mean/epsilon scaling, reciprocal square root, and the two multiplications
for each output feature panel. It depends only on the necessary row statistic,
its input panel and its gain panel. Embedding gathers likewise produce separate
row/feature output regions.

The Xonotic compiler lowers compatible ordinary two-dimensional matmul, arithmetic
expressions and row sum/mean reductions through these same library calls while
preserving declared peers. Its specialized expert and neighborhood source still
has separate mathematical indexing requirements; this change does not claim to
have generalized those operators' region lowering. No hidden payload coalescing
is used to cross that remaining boundary.

## Region expression fusion

Tillet et al., [Triton: an intermediate language and compiler for tiled neural
network computations](https://doi.org/10.1145/3315508.3329973), and the published
[Triton Layer Normalization implementation](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
show the forward implementation as a row reduction followed by normalization and
scaling inside one kernel. Local scalar statistics do not require an intermediate
tensor and another launch for every arithmetic operator.
[JAX Pallas BlockSpecs](https://docs.jax.dev/en/latest/pallas/quickstart.html)
keep the surrounding tile's input and output references explicit. Fusion occurs
inside that configured region; it does not merge independent region dependencies.

`kernels.arguments` constructs scalar input expressions. Addition, subtraction,
multiplication, division, exponential, hyperbolic tangent, reciprocal square root,
and last-axis sum compose through `kernels.expression` passed to the existing
`kernel_call`. For example, a normalization of a complete feature region is
`expression(x * ((x*x).sum()/width + epsilon).rsqrt() * gamma)`.
A feature-partitioned normalization instead publishes each region's
`expression((x*x).sum())`, combines the required row statistics through the
configured reduction, and applies
`expression(x * (statistic/width + epsilon).rsqrt() * gamma)` to each feature
region. Fusion removes arithmetic intermediates without pretending that a row's
normalization factor is known before its required contributions arrive.

Setup specializes the expression to retained view strides, scalar types and region
shapes. CPU lowering emits C, compiles it once per source in the Program, binds
canonical operand addresses, and executes the loaded function without Python
callbacks or operand allocation. Temporary compiler artifacts are removed after
loading. Metal lowering emits one SIMD group per row: lanes accumulate feature
stripes, `simd_sum` forms each required row statistic, and lanes write output
stripes directly. No global barrier or additional readiness protocol is introduced.
Both backends publish through the same configured region completion mechanism.
The native source binder contains no model definition or normalization-specific
operation. All compilation and pointer-vector construction occur during setup.

Compilation evidence: the native library builds, the emitted CPU normalization
source compiles to a dylib, and the emitted Metal source builds a compute pipeline
through Metal's runtime compiler. These are compilation checks, not claims about
speedup or numerical results; the ordinary composed example supplies those.

## Region execution timing

Apple's [MTLCommandBuffer GPUStartTime](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime)
and [GPUEndTime](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpuendtime)
report the command's device interval. Each configured function owns one fixed trace
record for its latest invocation: readiness dispatch, worker start, completion
callback, GPU interval where applicable, and submission count. Host timestamps
use local uptime nanoseconds. Readiness dispatch means the configured dependencies
were observed ready and claimed; it is not a remote clock timestamp. CPU timings
include the actual native numerical call, and completion precedes publishing the
function's output rows. Reading a trace does not execute or synchronize the graph.

The trace index identifies the configured function. `first_output` identifies its
first output map and `output_maps` records how many output maps it owns; neither
field invents a contiguous union for a multi-output function. CPU, Metal and Core
ML are recorded as execution kinds zero, one and two. Reused functions overwrite
their latest timestamps and increment a count; finite planned invocations each
retain their own record. Analysis must not treat this bounded record as an
unbounded event history or subtract clocks across nodes.

## Publication work lists

The Linux kernel authors' [lockless list API](https://raw.githubusercontent.com/torvalds/linux/master/include/linux/llist.h)
and [implementation](https://raw.githubusercontent.com/torvalds/linux/master/lib/llist.c)
provide concurrent insertion with batch detachment (`llist_add` / `llist_del_all`).
NVIDIA's [NCCL proxy](https://raw.githubusercontent.com/NVIDIA/nccl/master/src/proxy.cc)
progresses submitted communication independently of numerical callers. Mesh uses
stable shared-memory indices in this established list pattern, rather than pointers
whose virtual addresses differ between processes.

`mesh_notice_push`, `mesh_notice_take`, `mesh_notice_next`, and `mesh_notices`
implement separate compute and transport notification lists. Each row has a
preallocated node per consumer. Repeated notifications coalesce while that node
is linked. The consumer captures the next index before an acquire-release
exchange clears membership, acquiring publications from coalescing producers
before evaluating dependents. Later producers can immediately relink the node.
No numerical caller waits for a missing ring position or a transport completion.

The compute handler visits only detached row notices and their setup-established
reader lists. A per-handler intrusive list deduplicates affected occurrences
before checking their canonical masks once per batch. Transport `link_publications`
visits the corresponding fixed row-to-transfer adjacency and `link_ready` queues
eligible explicit transfer indices. Ready queues preserve source/target tuples
without searching the configured transfer table. Setup alone constructs adjacency
and admits already-present sources. An unsuccessful index post restores selected
indices; announced payloads retain their exact continuation as before.

These lists describe pending notifications/work, not additional tensor readiness.
Canonical presence and reader masks continue to describe values and ownership.
All nodes, links, and ready-index storage are realized before numerical invocation.

### Reduction precision

The [Triton normalization tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
promotes loaded values to float32 before computing normalization statistics.
Mesh RMSNorm now stores each panel statistic and its reduction tree in float32,
including for float16 inputs. The expression kernel loads half inputs into float
arithmetic; the final normalization writes the original input dtype. The
`_row_reduce(output_dtype=...)` setup parameter names that storage choice explicitly.

## Contraction accumulation

The JAX authors' [Pallas mixed-precision matmul](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html#bfloat16-matrix-multiplication)
and Tillet et al.'s [Triton matmul](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
retain FP32 accumulators and convert after contraction. `nn.linear` now gives all
K-panel products and their pairwise sum tree FP32 output storage. Its default
result dtype remains the input dtype; `output_dtype` permits retaining the FP32
result for a surrounding sum. `_cast` uses the existing affine operation on each
completed output tile, or returns an already matching tensor during setup.

`nn.ffn` retains FP32 across input-partition sums and down-projection sums.
Swish reads the complete FP32 hidden tile and stores the activation dtype in the
same launch. Down-projection contributions remain FP32 until their final sum,
then convert once per output tile. No cast waits for other output tiles. This
changes numerical rounding boundaries, not dependency or transport mechanisms.

CPU native contraction already loads half values into float and accumulates in
float; FP32 output storage removes the former premature half store. Apple's
[MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)
receives the existing FP16 operand descriptors and an FP32 result descriptor.
The operational gold and exact cancellation cases confirm this combination on
M5 Max and are recorded separately for the peer. No alternate Metal contraction,
operand conversion array or hidden host staging is introduced. The public MPS
header does not specify every internal arithmetic instruction; the claim here
is retained FP32 output and reduction storage, supported by numerical evidence.

Apple's [typed execution](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html)
and [conversion source](https://github.com/apple/coremltools/blob/main/coremltools/converters/_converters_entry.py)
explain that FLOAT32 conversion preserves declared FP32 operations, rather than
promoting existing FP16 products. `mesh_coreml.compile_part` now casts operands
before matmul and removes the former internal 32-K FP16-product subdivision.
The compiled function still represents one externally publishable region. Public
inputs and output backings remain canonical mesh pages; Core ML's internal cast
storage and device placement remain compiler-owned and are not established as
copy-free or ANE-resident by this source change.

The existing streaming-algebra example accepts float16 inputs and retains its
float64 reference for float32 inputs. The half reference rounds at activation,
FFN output, normalization and embedding-add boundaries. Its half tolerance is
3e-3 absolute/relative; the existing float32 tolerance is unchanged. An additional
ordinary linear expression gives exact outputs 2 and 0 from rows
`[4096, 1, -4096, 1]` and `[60000, 60000, -60000, -60000]` times ones with K panels
of two. This detects lost residuals and premature partial overflow that the
random gold chain alone would miss. It runs through the same library path.

## Indexed expression lowering

The JAX authors' [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)
uses references, indexed accesses and numerical primitives in compiled kernel
bodies. Mesh extends its existing expression tree with integer-preserving loads,
row/column indices, comparisons, bitwise mask composition and selection.
`indices()` denotes coordinates inside the bound output region. `input.at(row,
column, mask=..., other=...)` loads from the bound input region only when the
mask is true; a false mask evaluates the alternate value. CPU and Metal emit
conditional expressions so an invalid masked address is never dereferenced.
Indices must be valid when the mask is true. Shape/stride translation is retained
from the canonical native view, including transposed views.

Integer inputs remain integer expressions rather than passing through float.
Integer literals retain their exact values, including values above 2**53.
Floating inputs load into float arithmetic, preserving the established half
precision accumulation rule. Output reference dtype controls storage; reductions
use float for real outputs and signed/unsigned 64-bit sums for integer outputs.
This is not yet a full dtype-promotion implementation for arbitrary mixed types.
`equal` constructs a numerical comparison; structural Python equality continues
to identify common expression nodes. Comparisons and bitwise mask operations
construct expression nodes through the same kernel-call path.

`nn.embedding` now compiles its indexed load with these operations instead of a
Python/NumPy callback. It retains per-output row/feature bindings and normalizes
negative table indices. The old callback gather has no callers and is removed.
Xonotic's two-dimensional comparison, bitwise and selection nodes use this same
lowering with explicit output dtype. Other operation coverage remains recorded in
`lowering-coverage.md` and the nine-step plan.

The current binding still requires the entire declared source region to be
present. An indexed load inside a compiled kernel does not establish dynamic
selected-page readiness. That remaining dependency lowering is an explicit next
implementation step; this change makes no claim that unrelated missing table
regions can already be bypassed. The existing gold's constant embedding tables
and streamed output regions exercise the compiled numerical path.

The existing streaming-algebra example adds a masked gather/transform over an
int64 index equal to 2**53+1. It must retain exact integer identity and must not
load its masked-out table address. Its numerical output checks failures that
ordinary small embedding indices and random FFN errors would miss.

## Actual frame capacity

Apple's [TN3205, queue-pair allocation and completion polling](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
specifies that Thunderbolt queue depths count 4 KiB frames, that applications
must query assigned QP capabilities, and that completion releases a posted
request. One 8 KiB message costs two queue spaces, exactly as two 4 KiB
messages do. Matching SEND and RECV messages must have equal frame lengths.

`verbs_up` queries each QP's send and receive frame capacities separately.
Each direction has its own completion queue, large enough for its maximum
number of one-frame messages, with the extra CQ entry used by Apple's
allocation example. Device `max_qp_wr` and `max_cqe` constrain the setup request.
`down_pair` destroys these CQs only after destroying every QP.

`link_post` retains the actual frame cost with the posted row/page/index/byte
identity; `mesh_progress` subtracts that cost on that request's completion.
Announced-but-not-yet-posted transfers reserve their actual frame costs until
`link_send_announced` moves them into the posted queue. `link_receive` admits
the next announced receive only when its exact cost fits; announced ordering
must match the UC SEND order. The ready list has no such hardware ordering
obligation before announcement: entries fitting the remaining capacity can
be selected past a larger entry. This is byte-capacity accounting, not another
numerical readiness or acknowledgement protocol.

The index QP uses one-frame messages and its existing setup-allocated shared
buffer slots. That independent storage limit remains explicit. No index
buffer can be reused while its request is posted. CQ identity supplies the
QP and direction directly; retained work-request identity is checked against
the queue's posted entry, rather than searching configured QPs.

[Source proof, compilation and remaining evidence](actual-frame-capacity-2026-09-13.md).

An expression kernel can return multiple values. Binding slices each output's
expression to its actual input references and remaps retained operand indices
before native compilation. Each output therefore has its own dependency list
and publication. This implements ordinary output dependency analysis as described
by the Pallas reference/body model; it does not force unused operands into a
function's readiness mask. The existing example binds a second output requiring
a deliberately absent input, observes the independent indexed output first, then
supplies the second input. Both outputs use the same public kernel_call.

Xonotic selects shared numerical lowering before requesting custom Metal source
for a remaining operation. Its source emitter receives only those remaining
nodes. Eagerly generating every custom kernel used to reject even expressions
that the shared FP16 backend could execute, because the custom emitter did not
support their dtype. Setup now preserves backend choice and emits only code that
will be used. This does not add unsupported half atomic kernels or claim that
all remaining custom operations support half precision.

## Dynamic indexed expression lowering

The JAX authors' [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)
and [Grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
separate logical references, program indices, index maps and physical blocks.
`BlockSpec(None)` retains a whole logical input Tensor's shape, grid, block
shape and per-block views; explicit output BlockSpecs still determine independently
publishable regions. `program_id(axis)` is resolved from the configured grid
coordinate before code generation. It is not a runtime participant identifier.

`_ExpressionKernel.bind` lowers each `.at` access on such an input to a setup-bound
candidate address range and a numerical U32 selector. The selector computes the
candidate ordinal from the same row/column/mask expressions used by the payload
load; masked accesses write `MESH_ABSENT`. It consumes routing inputs, not the
payload whose readiness it selects. Nested indexed addressing recursively lowers
its own true dependencies. `accesses_for` retains conditional access paths,
prunes paths subsumed by an earlier necessary access, and emits lazy conditional
selector expressions. Reductions already emitted as eager reduction loops retain
their actual unconditional dependencies. Multiple outputs remain separate native
functions with their own selectors, so an unrelated operand of another output
does not become a dependency.

The native indexed attachment receives exact flattened source-input positions,
not a guessed view set. This preserves an ordinary use of a buffer even when the
same buffer is also a candidate for an indexed use. Selection and source-reader
lifetime remain in canonical dataflow. Each logical table's candidates are ordered
by retained block coordinates; this works for transposed logical grids without
assuming dictionary insertion order equals physical index order.

Generated payload code indexes the existing native buffer-address vector directly.
It does not construct a second pointer vector on every invocation or copy table
payloads. Uniform block strides are literal constants; ragged/transposed stride
lookups are setup-constant global tables. The private `block_ordinal` operation
uses nonnegative, valid logical indices and positive block dimensions. It is not
a public floor-division operation. As with Pallas indexed loads, callers must
supply valid coordinates or a mask; an invalid unmasked coordinate does not acquire
valid semantics merely because its quotient names an existing last block.

`embedding` now uses the same indexed expression with a logical whole-table input
and a program-indexed global feature column. Row-partitioned tables no longer
require a contiguous full-table region or whole-table arrival. Existing negative
index normalization is retained. A selector and payload function are two native
launches in this lowering; fusion of index production into a preceding producer
is separate optimization work, not an unmeasured zero-overhead claim.

## Dynamic reader lifetimes

Papadopoulos and Culler's [Monsoon: An Explicit Token-Store Architecture](https://www.cs.cmu.edu/~18742/papers/Papadopoulos1990.pdf)
provides explicit operand identity and presence-bit matching. The JAX authors'
[Pallas bounded dynamic slices](https://github.com/jax-ml/jax/blob/main/docs/pallas/tpu/pipelining.md)
provide configured maximum regions with numerical runtime indices. Mesh combines
these mechanisms using produced U32 selector ordinals, exact candidate input
positions and canonical logical result rows. This is a mesh lifecycle extension,
not a claim that Pallas itself implements host page-table readers.

`mesh_algebra_indexed` retains original binding input roles, exact candidate maps,
selector layouts and per-candidate lifetime results. `mesh_index_prepare` retains
selected-candidate membership once per selector publication. `mesh_index_ready`
checks selected ranges only. Numerical completion publishes a logical completion
result; the existing metadata event owner releases selected sources only after
that result, and unselected sources whenever their own values arrive. A retirement
result is published before its source READ bit, so source reuse cannot erase the
selector's history. Selector lifetime ends after all its candidate occurrences
retire. Selector output preparation resets its owned metadata results.

The [full lifetime trace](dynamic-reader-lifetimes.md#implemented-native-contract)
specifies local selector production, duplicate-index and fanout semantics, bounded
repeated occurrences, metadata costs and ABI implications. No source tensor copies,
additional participant scheduler or per-candidate numerical launches implement
these transitions.

The streaming-algebra operational example also binds a produced two-block table
through BlockSpec(None). It publishes only the selected block, consumes the
result while the unrelated block remains absent, then republishes the selected
block before supplying the late unselected block. The next selector occurrence
must consume the new selected value. This checks both independent readiness and
retirement identity across reuse; constant table examples cannot establish them.

### Explicit constant candidate specialization

Pallas's setup specialization principle also applies to dependency lowering.
`Program.constant` retains each successful declaration as the canonical native
`(tensor, extent)` identity. `_ExpressionKernel.bind` omits numerical selectors
and dynamic lifetime attachments only when every candidate extent of that logical
input has an explicit declaration in that Program. Ordinary index and mask inputs
remain ordinary numerical dependencies, and generated indexed payload loads are
unchanged. Candidate refs still bind directly to their actual canonical pages.

This uses retained configuration facts, not contents, current presence or a guess
that an embedding table is usually constant. A produced or received table still
uses dynamic selection even if all its candidates happen to be present during
setup. Mixed constant/produced tables retain dynamic descriptors. Transposed views
share their canonical tensor/extent identity and therefore retain the declaration.
This removes the extra selector launch for fully declared constant tables without
turning current availability into a permanent specialization assumption.

## CPU register contraction

Goto and van de Geijn's [Anatomy of High-Performance Matrix Multiplication](https://doi.org/10.1145/1356052.1356053)
provides register-blocked outer-product accumulation. Arm's
[Advanced SIMD intrinsic reference](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html)
specifies half-to-float vector conversion and lane-broadcast FP32 FMA.
`cpu_contract_tile` uses a 4 by 4 register block with directly addressed canonical
operands. `cpu_f16x4`, `cpu_f32x4` and their strided variants are setup-selected
loads, preserving half storage while accumulating in FP32. Ragged lanes are
masked by numerical extent and never read outside the operand. Output conversion
follows completion of the bound K region. The implementation does not pack or
allocate dense converted operands and retains the existing all-FP32 SGEMM path.
[Source and compilation record](cpu-register-contraction.md) distinguishes this
mechanism from unmeasured throughput and from BNNS internal-storage guarantees.

## Segmented indexed add

Blelloch's [Prefix Sums and Their Applications, sections 1.3 and 1.5](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
supplies stable radix partitioning and segmented reduction; the JAX authors'
[scatter-add contract](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter_add.html)
defines duplicate contribution semantics. `indexed_add` is a root expression
consumed by `_ExpressionKernel.bind_grid` through the existing `kernel_call`.
It takes symbolic base, destination and update references plus a validity
expression. Its initial logical domain is U×1 row destinations, U×F updates,
and a D×F base/output; valid negative destinations normalize by adding D.
Unmasked destinations must name valid rows. Dtypes of base/update/output match;
real partials accumulate and store FP32, and the final region casts once.

`_lower_indexed_add` builds routing once across a configured output grid.
Routing chunk boundaries respect existing index, mask and update row backing
boundaries. `_group_ordinals` performs eight stable four-bit radix passes,
retaining keys and original ordinals in setup-allocated canonical metadata.
CPU uses scalar prefix operations; Metal uses SIMD prefix sums and sums over
32 lanes. A device-memory threadgroup barrier separates radix passes in one
numerical kernel; it is not a host or tensor-wide readiness dependency. Sorted
segment keys and begin/end pairs are compact arrays bounded by chunk updates.
Each segment selects only its contributing source backing through
`_indexed_range`, and `_candidate_load` addresses that canonical backing with
its retained row/column strides. Unused segment slots have empty ranges and
select no update source.

`_compiled_region` shares the numerical source body between CPU and Metal,
changing only scalar/address-space declarations and prefix-operation syntax.
No Python callback executes a numerical kernel. Transposed/ragged stride
metadata is setup-constant, and buffer addresses come directly from the native
binding vector. Per-segment partials are separately published FP32 values; their
producer does not wait for other routing chunks or the merged directory.

The reverse directory groups segment keys and exact partial ordinals using the
same radix implementation. `locate` produces a compact range for each destination
region by binary search. `finish` selects and sums only those named partials,
starting from the actual base region. Empty destinations read no partial values.
This stage necessarily consumes all routing chunks when any chunk can name any
destination, but it never adds all update payloads as ordinary dependencies.
The structural and performance limits are recorded in
[scatter-lowering.md](scatter-lowering.md#first-executable-vector-lowering).

## Xonotic block indexed lowering

The JAX authors' [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)
separates logical indexing from physical reference binding. Xonotic's matrix_view
retains direct strides for rank-one transpose and single-backing reshapes;
rank-one scatter_add invokes the shared segmented indexed-add representation.
Its scalar/vector pointwise and sum consumers use the same expression compiler.
Integer reduction trees retain the existing owner and use unsigned bit-pattern
addition before storage in the declared dtype.

Remaining custom kernels bind an explicit block-pointer sequence and per-block
stride table. Their page_address maps logical flat indices through retained
matrix/block geometry to that exact pointer and stride entry. It never assembles
a dense input or guesses ragged strides. These remaining whole-region source
algorithms are migration backlog; this metadata change preserves access to the
canonical pages while their numerical lowering is replaced.
[Source mapping and compilation limits](xonotic-indexed-add.md).

## Fused indexed update values

The JAX authors' [Pallas pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html)
retains numerical work inside the kernel that owns its accumulation, and
Blelloch's segmented reduction above determines the contribution domain.
`_scalar_expression` is the shared scalar emitter used by ordinary expressions
and indexed segment updates; it owns literal, arithmetic, comparison, select and
transcendental syntax for both CPU and Metal. The scatter lowering does not add a
second implementation of those operators.

The value argument of `indexed_add(base, destinations, value, mask=...)` can be a
pointwise expression such as `updates * scale + other`. `value_dependencies`
retains the exact input references and verifies their broadcast domain against
U×F. Setup aligns routing chunks with every used input's existing row backing.
Each segment's candidate binding names the actual feature region of each used
operand, including row and column broadcasts. Logical row expressions evaluate
to the retained original update ordinal, and logical columns to the feature
stripe's retained global column plus its local index.

`_bind_segment_expression` uses the shared indexed-load emitter for each exact
canonical buffer and emits the scalar expression inside its bounded ordinal
accumulation loop. Half/float
loads are explicitly converted to float before arithmetic; real segment totals
and partial storage remain FP32, with the existing final output cast. Mixed
half/float operands therefore do not introduce an implicit half intermediate.
No transformed-update operand allocation, publication or kernel launch is added.
This fusion is lawful because the value expression is internal to the segment;
separately produced/exported tensors retain their existing publication boundary.

Each nonconstant operand gets its own selected-reader attachment over the same
compact segment range. Empty segments select no candidate pages. Explicitly
constant extents need no dynamic descriptor; merely present operands still do.
Normal expression inputs retain their ordinary value dependencies even in a
`select`; this is distinct from lazy `.at` access selection. Fused update values
currently cover pointwise input-reference expressions, not nested `.at` loads or
feature reductions. Those remain explicit producers; the compiler reports the
unsupported fused form rather than lowering it eagerly or silently changing its
dependencies. Existing segmented publication and reverse-directory lifetimes
are unchanged. Python compilation is evidence for source syntax only; operational
CPU/Metal validation belongs to the existing gold workflow.

## Shared contraction lowering

The JAX authors' [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)
separates numerical kernel expressions and their backend implementation, while
[Pallas pipelining and accumulation](https://docs.jax.dev/en/latest/pallas/pipelining.html#reductions-and-accumulation)
describes tiled reduction dependencies. `dot(left, right, tile_k=...)` is a root
expression through the existing `Program.kernel_call`, with logical whole-input
BlockSpecs and ordinary output-region maps. Its setup K-tile choice is metadata,
not a runtime branch in the numerical call graph.

`_ExpressionRegions.parts` owns contraction partition and reduction construction. K boundaries
respect the actual left/right backing partitions. Every K panel binds the existing
native `kernels.matmul` implementation and writes an FP32 partial. The existing
native `kernels.add` combines them in the same pairwise tree used by `nn.linear`
previously. The final FP32 result writes the preallocated output directly; a
non-FP32 declared output uses the existing affine conversion exactly once after
that completed region's FP32 sum. A single K panel can write an FP32 output
directly. Its `temporary` helper and `_bind_operation` allocate canonical storage and
bind those existing operations; they do not implement another matmul or select a
backend. CPU, MPS and configured Core ML handling remain in the native operation
owner, including their existing operand and accumulation precision paths.

`nn.linear` now supplies the contraction expression, output shape, M/N region
layout and placement. It does not construct K-panel launches, reduction tensors
or cast stages. Its M/N gcd choices still ensure each input region fits an actual
backing. A direct `dot` caller has that same current alignment requirement;
regions spanning multiple M/N backings require further general lowering, not a
hidden dense copy or a whole-operand wait. Independently published output regions
and the true K dependencies remain unchanged.

Scratch allocation belongs to the participant that binds the numerical kernel;
public output and transport endpoint identities are explicit and do not rely on
identical private configuration on both peers. All allocation/binding remains
before realization. This source change preserves the existing FP32 K-panel and
cast behavior; Python compilation is complete, while the existing gold workflow
supplies operational CPU/Metal/Core ML validation. Scratch lifetime packing
remains distinct lowering work.

### Independently typed contraction operands

Computed FP32 panels can contract with original FP16 weight views without a
separate operand conversion or a backend change. `mesh_algebra_bind` accepts
FP16 or FP32 independently for each contraction input; its dimension, physical
stride, output-layout and alias checks remain. CPU `cpu_part` retains the SGEMM
path for all-FP32 operands/output. Its other contraction specializations select
each input's scalar and vector loader independently and accumulate in FP32
registers before the requested output store, as described in
[CPU register contraction](#cpu-register-contraction).

`matrix` already retains each extent's own scalar type, offset and strides in
its MPS descriptor. The existing MPSMatrixMultiplication encode path receives
these original buffer views unchanged. Apple's
[encode contract](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication/encode(commandbuffer:leftmatrix:rightmatrix:resultmatrix:))
specifies dimension constraints but does not enumerate all supported input dtype
combinations. Consequently source review establishes independent typed bindings,
not a universal heterogeneous-MPS guarantee; operational mixed-input coverage
belongs to the existing numerical workflow. No alternate kernel or silent cast
is selected if a combination fails.

Core ML rectangle metadata now retains both input types. `compile_part` declares
separate typed placeholders matching `native_array`'s original typed pointer and
stride views, then keeps the existing FP32 casts, contraction and final output
cast. Apple's [typed execution](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html)
describes this explicit graph precision. The specialization key includes both
type flags and the generator source, so an old single-type model cannot alias a
mixed specialization. Internal Core ML cast storage remains compiler-owned; this
does not establish copy-free execution inside Core ML. This increment changes
no public C ABI and reports no speedup.

### Mapped and whole-reference dot operands

`_ExpressionRegions.parts` resolves each non-None input BlockSpec at every configured grid
coordinate. Its local M/K/N dimensions, transposed strides and offset are the
actual operand, not a hint discarded in favor of the containing Tensor. K
panels slice that resolved Ref. A None BlockSpec retains the whole logical
operand: its outer M/N region follows the output map, and its K panels respect
its canonical backing boundaries. Both forms can be mixed without copying.

Mapped operand outer dimensions must equal that output region's dimensions,
and the two actual K dimensions must match. The lowering does not infer an
unrequested K offset for a whole operand from another operand's mapped view.
For example, a mapped M×K row tile with a whole K×N weight tensor preserves the
row map and selects the output's N region from the weights. Two mapped refs
can independently permute input row and column tiles while writing the normal
output grid. All forms reuse the same FP32 native panel/reduction bindings.


### Recursive pointwise dot epilogues

`_requires_regions` identifies contractions within existing expression nodes;
`_lower_region_expressions` recursively substitutes each contraction with its
FP32 region partials. The same Pallas numerical-expression and tiled-accumulation
sources above motivate this composition. For example, with
`z = kernels.dot(a, b, tile_k=128)`, both `kernels.expression(z*scale+bias)`
and `kernels.expression(z/(1+(0-z).exp()))` use the ordinary kernel-call interface.
Additional numerical inputs retain their explicit BlockSpecs, including mapped
broadcast regions and indexed loads through the existing expression lowering.

Native matmul panels and intermediate native pairwise sums remain unchanged.
The epilogue substitutes the last one or two FP32 partials into the existing
scalar emitter: its final addition, pointwise expression and declared output
conversion occur in one numerical region function. It allocates no completed
whole-tensor contraction or separate pointwise intermediate. All setup storage
is canonical; each output region depends only on its own K panels and actual
additional expression inputs.

The lowering caches contraction nodes by expression identity, output-region
origin and shape within each grid coordinate. This shares native contractions
across expression outputs without combining their dependencies. Per-output
substitution also reuses exact partial input identities when the same dot appears
more than once, as in swish. A separately exposed bare dot publishes without
waiting for another output's bias or scale. A single native panel may write that
bare FP32 output directly; epilogue readers then hold its canonical lifetime.
With multiple panels, the bare output uses its native final reduction while an
epilogue independently fuses those same last partials.

Indexed addition remains an output-root operation. Computed contraction operands
and row reductions use the region-demand lowering described below. There is no fallback that materializes a
whole operand or waits for unrelated regions. Python compilation checks the
implementation; the existing streamed gold example supplies runtime evidence.


### Region-demand computed contraction operands

The JAX authors' [Pallas BlockSpec documentation](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
defines program-specific operand regions, while the cited Pallas pipelining
material distinguishes tiled dependencies from full-array sequencing.
`_ExpressionRegions` applies these region and dependency principles to the
existing expression representation and native operations. For example:

```python
x, w1, w2 = kernels.arguments(3)
h = kernels.dot(x, w1, tile_k=2)
f = kernels.expression(kernels.dot(h / (1 + (0-h).exp()), w2, tile_k=2))
```

`layout` propagates natural two-dimensional shapes, per-axis local/global domains,
and actual backing boundaries through broadcast pointwise expressions and nested
dots. A dot inherits its row domain from its left operand and column domain from
its right operand. Nonbroadcast mapped participants preserve local pointwise
axis ownership even when another input uses whole-tensor storage. Mapped refs
retain their resolved offsets and transposed strides; whole refs retain their
logical coordinates. K-panel boundaries respect the backing cuts propagated
from both operand expressions, including hidden columns produced by an inner dot.
Ragged logical lengths are not treated as additional periodic backing cuts.

`parts` requests each left M×K panel and right K×N panel separately. `panel`
returns actual input refs directly, or allocates only the requested computed
region in canonical FP32 storage. `emit` recursively substitutes inner dot
partials into the shared scalar emitter, fusing their final sum with pointwise
arithmetic. Thus an outer contraction can consume a completed hidden panel while
other hidden panels have not arrived. No full hidden tensor or whole-intermediate
publication exists. `publish` retains the native final-sum/cast path for exposed
bare contractions. Mixed half/float operand handling stays in the existing native
matmul owner; computed intermediates are FP32, without intermediate half stores.

`key` retains the expression, demanded origin/shape and actual used input
identities. Whole tensor identities and mapped ref offsets/strides distinguish
values; irrelevant grid axes do not. The cache is shared across grid coordinates,
so a left panel can serve multiple output column tiles and a right panel multiple
output row tiles. `program_id` is specialized to its original coordinate before
caching, preventing reuse of numerically different programs. Canonical reader
members retain shared regions until every configured consumer has completed.

Computed operands support pointwise expressions, nested dots and row reductions.
Indexed loads whose coordinate expressions determine their shapes retain the
existing selected-reader path; index intrinsics alone do not declare a new
logical tensor shape. Non-singleton operand dimensions must match in their resolved domains;
a whole four-row value plus a mapped two-row value inside one computed operand
requires explicit matching maps. Direct inputs whose requested M/N region spans
multiple backing extents remain outside the native single-ref binding contract;
there is no dense copy or alternate backend fallback. Python compilation and
source review validate setup construction; operational evidence comes from the
existing streamed nested-contraction gold case.


### Streamed row reductions in the shared region owner

The shared expression now exposes `.T` and `.sum(axis=...)` over its resolved
matrix domain. The default remains axis one. Axis zero is the algebra
`x.T.sum().T`; both axes (`None` or `(0,1)`) are `x.sum().T.sum()`.
Negative axes normalize during setup and an empty axis tuple is the identity.
Singleton reduced dimensions remain explicit in the physical two-dimensional
result. Xonotic supplies logical shape metadata separately and uses these
expressions for all nonempty vector/matrix sum and mean axis sets.

Transposition exchanges layout dimensions, backing cuts and origin coordinates.
Input panels retain original pages through transposed Ref views. Computed panels
use the existing region owner; a transposed output root swaps the destination
view so its producer writes directly into the requested output pages. Double
transposition cancels during expression construction. Transposition distributes
through pointwise expressions and reverses contraction operands, preserving
fusion over their transposed source views. Computed panel storage follows the
expression dtype rather than an unconditional FP32 default. Full sums reduce independent
row partials before reducing their transposed statistics. Those intermediate
statistics use the expression's accumulator dtype, preserving FP32 and exact
integer accumulation across both axes instead of inserting an FP16 or float
conversion between them. No additional runtime graph or scheduler is introduced.

The JAX authors' [Pallas reductions and accumulation discussion](https://docs.jax.dev/en/latest/pallas/pipelining.html#reductions-and-accumulation)
describes tiled accumulation and the lifetime of reduction buffers.
`_ExpressionRegions.reduction` implements row statistics with separately published
feature partials and canonical reader lifetimes. The existing `.sum()` expression
reduces the full resolved child's column domain to one column: whole Tensor inputs
use their full logical width, while mapped BlockSpecs use the actual resolved Ref
width. Its row domain and row backing boundaries come from the child expression.

For each demanded row range, setup propagates the child's feature backing cuts,
emits an independent partial for each feature interval, and combines those
partials in a balanced tree. The child's pointwise expression is fused directly
into the partial sum kernel. For example `(x*x).sum()` allocates scalar partials,
not a squared tensor. Likewise `dot(x,w).sum()` requests corresponding output
feature panels from the contraction and reduces their completed values. The
final row statistic necessarily depends on all features of that row; its partial
producers and unrelated rows remain independently usable.

Xonotic last-axis matrix sums and means use this same owner for integer and
boolean inputs as well as real inputs. Matrix rows retain their original backing
blocks; the row-dispatch selection no longer sends integer matrices to the custom
whole-operand emitter. Integer means retain the existing vector path’s final
division semantics. The operational example publishes one integer row at a time
and observes both the direct expression and Xonotic sum before publishing the
next row, including cancellation beyond 2**53 and modulo-2**64 overflow.

Matrix axis-zero sums and means use the same reduction after swapping the
operand view axes during setup. `Tensor.T` preserves the tensor handle and maps
each original block to `Ref.T`, which swaps shape and strides in the native view.
The completed statistic is exposed through the inverse output view. This is
index-map transformation, with no transpose kernel, operand copy or extra
publication. Each original column owns its contributing row partials; other
columns are independent. Keepdims retains the requested logical singleton axis.
The integer observation also binds the same pages as a transposed matrix and
reduces axis zero, so every direct row sum has a column-reduction counterpart.

Real statistics and their combination use FP32. Signed integer and boolean
outputs select int64 accumulators, unsigned outputs uint64, matching the existing
scalar emitter's accumulator selection. Integer partials and their tree remain
integer expressions, avoiding a float conversion that would lose values above
2**24. Integer-valued signed reductions accumulate unsigned 64-bit bits in both
the local loop/SIMD reduction and the partial merge tree. The final expression
interprets those bits as the signed statistic before subsequent arithmetic;
merge operands explicitly use the existing uint64-mask convention. Thus addition
is modulo 2**64 without signed C overflow, including intermediate cancellation.
The scalar emitter's `integral` setup classification retains the existing signed
accumulator path when a signed-output reduction actually consumes floating terms,
so this integer fix does not introduce negative-float-to-uint conversions.
Conversion to a declared narrower output occurs after the completed statistic. Floating accumulation is reassociated by the feature partial tree;
this is not a promise of bitwise equality with a serial sum.

The cache includes the child expression, actual used input identities, row origin,
row count and accumulator dtype; its feature origin is zero and feature extent
one. Different output feature tiles therefore share the same row statistic.
Broadcast epilogues and nested contractions consume that scalar through ordinary
canonical refs, without another scalar copy. A bare statistic can publish directly
to an output of the accumulator dtype. Expressions such as
`x*((x*x).sum()/width+eps).rsqrt()*gamma` use this same owner and selected feature
regions for their output panels. Gamma does not become a dependency of the sum.
All grid maps, allocation, specialization and bindings are realized before
numerical invocation; the existing gold examples provide runtime validation.


### Metal modular 64-bit SIMD sums

Apple's [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
and installed `metal_simdgroup` header define the supported SIMD reduction types.
The installed compiler rejects `simd_sum(ulong)`; integer row sums therefore use
three supported UInt32 reductions, with no alternate execution path or operand
copy. Floating `simd_sum(float)` remains unchanged.

For the configured 32-lane group, write each lane's 64-bit accumulator as
`a_i = l_i + 2**16*m_i + 2**32*h_i`, where the low and middle limbs are 16 bits
and the high limb 32 bits. Compute `L=sum(l_i)`, `M=sum(m_i)+(L>>16)`, and
`H=(sum(h_i)+(M>>16)) mod 2**32` using UInt32 SIMD sums and local carry addition.
Reconstruct `(H<<32) | ((M&65535)<<16) | (L&65535)` as UInt64. This is exactly
`sum(a_i) mod 2**64`: the two low carries account for every bit crossing the
16-bit and 32-bit boundaries, while overflow from the high limb is discarded.
`L <= 32*65535 < 2**21`, and `M <= 32*65535+31 < 2**21`, so neither low carry
calculation can overflow UInt32. High-limb overflow is the intended modular sum.
Signed results use the same bit representation and signed interpretation after
reconstruction. The shared scalar emitter generates these operations for integer
reductions; CPU UInt64 accumulation and the canonical partial tree retain their
existing modular semantics.


### Explicit scalar conversion boundaries

`_Expression.astype(dtype)` adds an explicit typed conversion to the existing
expression representation. It accepts float16, float32, int32, uint32, int64,
uint64, uint8 and bool. The Pallas numerical-expression separation cited above
and Apple's Metal scalar conversion rules apply; this is an ordinary numerical
operation, not a backend or scheduling choice.

The scalar emitter keeps the conversion at its declared position. A real cast
rounds to the requested type and then promotes back to FP32 for surrounding
pointwise arithmetic: half conversion emits `float(half(value))` on Metal and
its `_Float16` equivalent on CPU. Consequently a half boundary cannot disappear
because a later output is FP32, and subsequent arithmetic does not accidentally
execute in half precision. Integer casts retain their numerical scalar type;
finite float-to-integer values must be representable in the target type. No
universal NumPy equivalence is claimed for NaNs, out-of-range floating conversions
or implementation-specific signed narrowing.

The region layout inherits the child's shape, domains and backing cuts. A cast
used as a contraction operand allocates only the demanded canonical panel in the
cast dtype. Thus `dot(h.astype(np.float16), w)` supplies actual rounded half
storage to the existing native matmul while retaining FP32 contraction partials.
An enclosing pointwise expression still uses its ordinary FP32 computed panel.
A same-type cast of a direct input reuses its actual requested Ref without a copy.
The normalized cast dtype is part of expression identity and therefore of the
shared region cache key.

`_expression_dtype` infers value types at conversion boundaries. It keeps
a nested reduction's pre-cast accumulator type rather than propagating the final
output dtype backward through the conversion. `x.sum().astype(np.int64)` for
floating `x=[0.75,0.75]` therefore produces 1, while
`x.astype(np.int64).sum()` produces 0. The existing no-cast output-directed
accumulator convention remains unchanged. Integer classification for modular
summation uses the cast target type, and cached reduction substitutions retain
the accumulator dtype so differently typed statistics cannot alias.


### Mixed contraction orientation

The JAX authors' [Pallas matrix multiplication tutorial](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html)
distinguishes physical layouts and transposed operands. Mesh applies the ordinary
identity `A B = (B.T A.T).T` at setup to preserve native MPS execution when A is
FP16 and B is FP32. The installed MPS implementation asserts that mixed products
have FP32 left and result matrices and an FP16 right matrix; the captured initial
Metal failure provides evidence of this asymmetric backend constraint.

The region compiler allocates reverse-mixed FP32 partial storage with physical
shape N×M and retains its transposed M×N Ref. Native contraction binding accepts
unit-row-stride outputs and normalizes `(x,y,z)` to `(y.T,x.T,z.T)` before building
geometry, exact dependencies and matrix rectangles. The existing MPS operation
then sees FP32-left, FP16-right and a row-major FP32 result on the original
canonical pages. No input conversion, transpose copy, per-row launch scheme or
scalar fallback is introduced. The same normalization applies to CPU and Core ML
bindings, so their geometry and input dependencies describe the same identity.

For one-row/one-column cases both strides may be one. The explicit source dtypes
retain the required reversal in that otherwise ambiguous layout. General rectangular
cases allocate column-major partials. The existing final reduction or epilogue
consumes their actual strides and publishes the declared logical output orientation;
a single-panel bare output uses the existing affine publication operation if its
public allocation has the other orientation. FP32 K-partial precision remains
unchanged, and typed half operands remain half on their actual canonical storage.

Direct native row-major FP16×FP32 output bindings still cannot satisfy MPS and now
fail setup validation instead of aborting during encoding. Column-major mixed
outputs must likewise normalize to the supported operand order. The shared
expression owner chooses the appropriate orientation automatically; routing every
raw `kernels.matmul` caller through that owner remains separate migration work.
The native dylib builds and Python compiles. Existing multi-row rectangular and
streamed typed-panel examples provide CPU/Metal operational validation.


### Logical indexed views

The JAX authors' [Pallas BlockSpec documentation](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
separates logical indexing from the storage region bound to a program.
`argument.reshape(logical_shape).at(*coordinates, mask=True, other=0)` retains
logical shape metadata on an indexed input without reshaping or copying storage.
The ordinary `.at(row, column)` form remains available. Scalar shape `()` accepts
`.at()`; one `-1` dimension may be inferred at setup. `_resolve_logical` validates
rank and volume against the actual bound Tensor or mapped Ref shape.

For logical shape `(d0,...,dn)`, setup emits the row-major logical ordinal by
Horner recurrence `o = o*di + coordinate_i`, then maps that ordinal to
`(o // source.shape[1], o % source.shape[1])`. It retains the original physical
input object, candidate vector, transpose/offset metadata and strides. Per-axis
bounds join the original load mask before flattening can select a source:
`reshape((2,2)).at(0,3)` returns `other`, rather than aliasing flat element 3.
Negative coordinates are masked unless the caller explicitly normalizes them.
Mask composition is conditional, so a false caller mask does not introduce
read dependencies on otherwise unused coordinate loads.

Logical loads become existing `load` nodes before direct binding and region
shape/dependency analysis. They use the same dynamic selector, native candidate
maps and completion/lifetime machinery; there is no rank-specific execution path,
new allocation or dense matrix view. Root indexed-add expressions are normalized
before their existing classifier as well. General indexed-load update values and
arbitrary indexed scatter masks remain unfinished composition work; this rewrite
does not claim that separate capability is implemented.

`//` and `%` are integer expression operations using the shared dtype inference
and scalar emitter, including fused pointwise scatter values. Operands use their
common integral type. Unsigned division/remainder stays integer and retains values
above 2**53. For signed values, let `q=trunc(a/b)` and `r=a%b` from native integer
arithmetic. If `r!=0` and operand signs differ, emit `q-1` and `r+b`; otherwise
emit `q` and `r`. These are floor quotient and divisor-signed remainder without
converting through floating point. The domain requires a nonzero divisor and a
representable quotient, excluding signed `INT_MIN/-1`. No numerical wait or
runtime division guard is introduced. Mixed signed/unsigned operands follow the
same common-type conversion before division. Literal-only folding follows that
same typed rule and retains unsigned result identity; floating operands fail the
integer operation's setup validation. The minimum signed-64 literal is emitted
without an unsigned intermediate token. Python compilation and source review
cover construction; existing arbitrary-rank and wide-integer gold cases supply
native CPU/Metal evidence.


### Logical-rank pointwise callers

`logical_coordinates` supplies the common physical-to-logical output index map
for indexed and pointwise callers. Xonotic's pointwise arithmetic, comparisons, selection, casts and supported unary
operations use the shared expression compiler across logical ranks. Their
broadcast coordinates follow the JAX authors' index-map design: flatten output
leading axes into physical rows, recover each logical coordinate with integer
quotient/remainder, align operand axes from the right, and substitute zero on
singleton operand axes. Logical `.reshape(...).at(...)` resolves those coordinates
against the actual original operand pages. Scalars use the empty coordinate tuple.

Already-aligned matrix operands retain their mapped Ref bindings and existing
backing cuts. Other logical layouts use whole-table index bindings with tiled
output regions; static address specialization resolves their source dependencies.
They do not call `matrix_view` to assemble operands or use the custom whole-output
Metal emitter. These are configuration choices around the same scalar expression,
not separate CPU and Metal numerical bodies. The operational indexed example now
composes rank-three concatenation, broadcast multiplication and scalar addition,
and observes an early half while unrelated source/index regions remain absent.

### Static indexed access specialization

The JAX authors' [Pallas grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
use setup-known index maps to identify a program's storage region. The shared
expression compiler applies that specialization before creating dynamic indexed
reader descriptors. `_specialize_accesses` collects every occurrence of a load,
its complete vector access mask, and its actual output or enclosing reduction
domain. A reduction uses its child's iteration width, not its one-column result.
The configured function graph is fixed. Gathers consume vectors of indices;
`select` and load masks describe numerical values and address validity. These
operations introduce no data-dependent program branching or mesh scheduler.

`_static_value` evaluates only setup expressions: literals, original program IDs,
row/column indices, scalar casts and integer arithmetic. It never reads input
array contents, including declared constants. Enumeration uses bounded chunks of
4096 output points. Operand dtype promotion, casts and modular unsigned arithmetic
match emitted scalar operations; signed addition/subtraction/multiplication are
checked with bounded exact integer arithmetic and remain dynamic on overflow.
Division by zero, unrepresentable signed quotients, nonrepresentable floating
conversions, floating arithmetic predicates and unknown domains also remain on
the existing path. Boolean literals retain the emitted C integer literal type.
This evaluator runs only during binding; it is not a numerical invocation backend.

A load specializes when its address and validity vectors are setup-known over
all of its uses. Numerically produced index or validity vectors retain the
existing indexed dependency representation. Repeated occurrences union their
proven block sets; an unresolved use prevents node-wide specialization. The compiler obtains only selected entries
from the original Tensor's block dictionary, preserving their exact Ref offsets,
ragged dimensions, transpose strides and dtypes. It does not scan the table's
candidate blocks to discover the selected set.

The rewrite uses the existing block-local `.at` and numeric `select`. One selected
block needs no extra block-membership condition; multiple selected blocks use
ordinary coordinate selection, retaining the original load mask. Each exact static
Ref gets an explicit appended input position. Original whole-table positions
remain available for dynamic loads of the same table; unused positions disappear
in the existing remap pass. Native indexed attachment consequently cannot remove
a static ordinary dependency that aliases one of its dynamic candidates. Access
masks retain source traversal order when redundant predicate sets are removed;
their emitted expression order must not depend on Python hash randomization.

Purely static indexed accesses bind ordinary Ref inputs through `algebra_source`,
with no selector numerical function or dynamic reader descriptor. Mixed and dynamic
accesses use the existing selector machinery and canonical lifetimes. The same
specialization runs for direct expression binding and region-generated numerical
functions. Python compilation and source review validate construction; unchanged
streaming gold examples compare configured functions, indexed descriptors and
repeated CPU/Metal results before and after specialization.

## Canonical reader groups

Papadopoulos and Culler's [Monsoon](https://www.cs.cmu.edu/~18742/papers/Papadopoulos1990.pdf)
provides explicit operand matching and presence-bit storage. Mesh uses those
mechanisms to represent each actual source-row/reader completion as a canonical
logical result when direct READ planes cannot represent the configured fanout.
This grouped ownership is a mesh realization choice, not a Monsoon API claim.

Configuration surveys actual numerical/export readers and transport holds.
`mesh_reader_bind` preserves direct reader planes where they fit; overflowing
or uncolorable domains retain member identities and one shared source READ plane.
`mesh_map_ready` checks a function's own member, and completion publishes that
member. The existing metadata event owner AND-compares group results before
releasing the source. No extra numerical function or per-reader dispatch is used.

A group-completed result is published before the aggregate READ bit. It survives
remote source reuse until the source's next notification resets member results;
that distinguishes the new source value from the preceding completed group.
Local producer publication uses that same event owner for member reset. Export readiness and
consume follow the same member contract. Hardware SEND readers retain their
existing exact direct planes. [Shared routing ownership](shared-routing-ownership.md#implemented-reader-group-specialization)
records source disposition, the fast-path boundary, lifetime ordering and remaining
sparse-routing work.

## Xonotic shared indexing

The JAX authors' [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)
expresses numerical accesses through references and grid-relative index maps.
`tensor_metal.gather_expression` translates Xonotic's retained slice/fixed/advanced
index mapping into the existing `.at` expression representation. Global output
coordinates include program_id offsets; singleton index dimensions broadcast,
negative indices normalize against the source axis, and inserted axes retain
logical identity. Concatenation uses the same indexed loads under retained
source-interval predicates. Existing selected-reader lowering owns dependency
selection and lifetime; callers neither construct it nor emit backend kernels.
[Source mapping, derivative scope and validation limits](xonotic-shared-indexing.md).

## Shared sparse routing lowering

Blelloch, Heroux and Zagha's [segmented sparse matrix operations](https://www.cs.cmu.edu/~scandal/papers/CMU-CS-93-173.html)
provide the grouped sparse representation. The JAX authors' [Pallas reference and
index-map design](https://docs.jax.dev/en/latest/pallas/design/design.html) provides
the separation between numerical indexing and physical binding. Mesh's exact
domain lifetime protocol is specified in [shared routing ownership](shared-routing-ownership.md),
using Papadopoulos and Culler's [Monsoon](https://www.cs.cmu.edu/~18742/papers/Papadopoulos1990.pdf)
as prior art for explicit operand identity and presence matching.

`_routing_directory` produces candidate owners and a reverse ordinal directory
with consumer offsets. Consumer identities follow the actual configured output
regions, including permuted grids and uncovered rows. A partial in a feature
stripe has one destination owner; overlapping output ownership is not inferred
from nominal tile sizes. This specialization does not claim unique ownership for
arbitrary fanout. Routing is numerical work over canonical operands.

`_routing_domain` binds each candidate once and retains its address and actual
strides in a shared table. Each numerical consumer uses its directory interval
and the same table, rather than binding every possible candidate separately.
Backend address representation is realized during setup. The table describes
the original registered partial storage; it is not a copied numerical operand.
Native ownership must retain source occurrences and route lifetimes independently
of arrival order. Sparse bindings alone do not eliminate potential empty segment
launches, page allocation overhead, or the directory-production launches.

Apple's [Metal residency sets](https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets)
provide setup-time membership for indirectly addressed GPU buffers. `route_residency`
adds canonical candidate buffers to one set per algebra, attaches it to that
algebra's command queue, and commits membership before numerical execution.
Metal applies the queue's residency set to its command buffers. Consumers do not
walk the candidate list to call `useResource` during invocation. Source lifetime
still comes from mesh's exact row dependencies; residency does not establish
numerical readiness or authorize overwriting a live source. Program teardown
waits for its existing executions, removes the set, then releases its buffers.

## Active segment domains

The JAX authors' [Megablox grouped multiplication](https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/main/src/maxtext/kernels/megablox/backend.py)
retains actual tile counts and explicit tile identities in a bounded domain,
including empty groups only when output initialization requires them. Mesh's
[active segment contract](active-segments.md) applies that distinction to private
scatter partials while preserving final destination initialization. The number
of distinct validated destinations in U updates is at most min(U, D), permitting
that static partial-capacity bound without storage reuse or added dependencies.
The exact producer disposition and late-source lifetime mechanism is mesh's
realization, using the existing canonical presence-based execution owner.


The existing streaming-algebra workflow's optional `--xonotic` case constructs
an actual Xonotic gather/concatenate graph and observes its canonical result blocks.
Two executions change routing on reused source/index buffers, withhold an unrelated
source/index block, and compare early and complete outputs against precomputed
float64 references. This exercises the existing indexed-access contract and
publication interface; no additional numerical execution owner is introduced.
Derivative validation is not implied by this forward case.

The JAX authors' [gather transpose implementation](https://raw.githubusercontent.com/jax-ml/jax/main/jax/_src/lax/slicing.py)
forms a cotangent-typed zero operand and scatter-adds cotangents at the gathered
indices. Xonotic's rank-one and full-feature row-gather VJPs now express that same
construction using existing indexed_add. Setup supplies canonical zero blocks;
index and cotangent pages are the only numerical operands. Primal source shape
is retained, but its values are neither replicated nor read by this derivative.
Shared segmented reduction and publication remain the sole numerical owner.
The covered mappings and remaining derivative cases are recorded in
[xonotic-shared-indexing.md](xonotic-shared-indexing.md#shared-row-gather-transpose).

## Xonotic output liveness

The LLVM/MLIR authors document elimination of unused side-effect-free operations
in [canonicalization](https://mlir.llvm.org/docs/Canonicalization/#globally-applied-rules).
Xonotic kernel_calls performs a backward setup traversal from explicitly requested
output Tensors and stops at supplied numerical inputs. Only live numerical nodes
are lowered. row_gather_gradient identifies the previously implemented gather
transpose mapping whose primal supplies shape only; its indices and cotangent
remain numerical dependencies. The same predicate controls liveness and actual
binding, so unused forward readers cannot block derivative-only input reuse.
This is compilation of the existing functional graph, not another runtime
scheduler. [Contract and caller migration](xonotic-shared-indexing.md#requested-output-liveness).

## FFN shared expression composition

The JAX authors' [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)
separates composable numerical expressions from their tiled backend lowering.
`nn.ffn` now supplies each hidden section as balanced input-partition dot sums
followed by swish in one ordinary `kernel_call`. The existing shared contraction
lowering owns K panels, FP32 partials, and contraction reduction storage and uses
the existing native matmul backend. The FFN supplies its algebra and public
output layout, without recreating contraction launch scheduling.

The hidden output uses the same row and column cuts as the prior per-projection
linear calls followed by pointwise pairwise sums: compute each former linear
tile, then take the pointwise gcd over actual partition boundaries. A whole-axis
tile contributes no artificial boundary. Input-partition sums retain their
balanced association. All arithmetic through swish remains FP32; the published
hidden output casts once to the input dtype, including intentional FP16 rounding.
`exchange(activated)` still receives that same independently publishable hidden
layout. Down projections retain their existing FP32 partial/output accumulation
and final balanced sum and input-dtype cast. No transport edge is fused away.

Full local nesting of the down projection requires a separately explicit typed
intermediate cast in the expression language. An implicitly FP32 computed
operand cannot replace the deliberate FP16 hidden boundary. This migration does
not make that substitution. Existing `examples/streaming-algebra.py` gold cases
exercise the two-input-partition FFN and its configured hidden exchange; the
source retains that validation path rather than adding a second evaluator.

## RMSNorm shared expression composition

Zhang and Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467),
define normalization using the mean of squared features and learned gain.
The JAX authors' [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)
places numerical composition in the caller and implementation of its block
dataflow in the lowering. `nn.rmsnorm` expresses that complete formula as
`x * ((x*x).sum()/width + epsilon).rsqrt() * gamma` through the existing shared
expression and kernel_call interfaces. It no longer constructs feature partials
or reduction trees in the DNN library.

Whole-input BlockSpecs give the statistic its complete logical feature domain.
The shared expression lowering produces a squared-sum partial independently for
each available input feature region, accumulates real statistics in FP32, and
shares each row-region statistic across output feature panels. A missing feature
prevents final normalization of its affected rows but does not prevent other
feature partials from computing and publishing. Each normalized output panel
uses its own input and gain regions plus that statistic.

The caller retains the former output row/column cuts, including gain backing
boundaries and ragged tails. Broadcasting gain creates only existing zero-stride
view metadata; it does not copy payloads. The declared output dtype remains the
input dtype, preserving the final FP16 cast when applicable. `_row_reduce` still
serves Xonotic reduction callers and is not deleted until those callers migrate
to the same shared reduction lowering. Numerical validation remains the existing
streaming-algebra gold and its delayed-feature observations.

## Typed FFN expression composition

The JAX authors' [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)
and [pipelining and accumulation](https://docs.jax.dev/en/latest/pallas/pipelining.html#reductions-and-accumulation)
separate the composed tensor formula from the implementation of partial-result
dataflow. `nn.ffn` now supplies the same hidden expression and down-projection
expression for both local execution and explicit hidden exchange. Its
`_expression_sum` constructs the existing balanced partition association without
allocating partial tensors or scheduling launches.

Every hidden section is the input-partition dot sum followed by swish and an
explicit `astype(input_dtype)`. Without an exchange callback, that expression is
the down dot's operand. Shared lowering produces demanded typed canonical hidden
panels and their FP32 contraction partials. The final balanced down-projection
sum casts to the input dtype. Intentional FP16 hidden rounding therefore survives
removal of the public intermediate tensor. The existing native contraction owner
still selects its configured accelerated backend; the DNN library introduces no
matmul implementation, dense conversion buffer, or runtime scheduler.

An exchange callback is a setup decision about publication. It submits each
already-built hidden expression with its former public shape, dtype and block
cuts, calls the exchange, then substitutes symbols for the returned tensors in
the same down-projection formula. The returned tensors' actual block layouts
determine consumer cuts. Without exchange, the down K tile also respects the
former hidden feature cuts, so removing a public intermediate does not coarsen
producer requests. Both layouts preserve final output row/column cuts and
balanced partition reductions. Unused operands in a hidden submission do not
create readers: shared expression lowering binds only referenced inputs.

Xonotic's existing cast node likewise supplies explicit `astype(value.dtype)`
in its expression; assignment remains identity. No separate cast emitter is
introduced. The unused `nn._cast` launch helper is removed. Source compilation
passed; existing local/paired gold and the typed nested-expression case supply
the operational numerical evidence.

## Xonotic logical indexing

The JAX authors' [Pallas BlockSpec indexing](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
and NumPy authors' [advanced indexing](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing)
and [take_along_axis](https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html)
define the block-coordinate and logical-index operations composed here. Xonotic
`gather_expression` supplies the existing graph's slice, inserted-axis and
broadcast advanced-index mapping through shared `reshape(logical_shape).at(...)`
expressions. `kernel_calls` uses that same shared view for arbitrary-rank gather,
take-along-axis, concatenate and transpose. Output physical rows encode prefix
logical dimensions with exact quotient/remainder; physical columns encode the
last logical dimension. Input storage and its actual per-block strides remain
unchanged, even when a graph reshape aliases a differently shaped Tensor.

Take-along-axis realizes its non-axis broadcast shape at graph construction.
Forward indexed expressions and the retained take VJP use zero coordinates for
source singleton dimensions; index singleton dimensions likewise broadcast.
Signed user indices normalize once. Concatenation retains source-interval
selected-reader dependencies, and transpose inverts its declared permutation.
The replaced custom forward emitters are removed; general gather/take VJP
scatter implementations remain pending migration, not alternate forward paths.

[Caller contract and validation scope](xonotic-shared-indexing.md#logical-rank-indexing-on-physical-pages)
describes these mappings and the existing optional streaming-algebra workflow.
This caller increment compiles with the shared logical-view implementation. The
graph broadcast-shape correction and rank-3 delayed-output observations are
coordinated integration edits committed next by the parent agent; numerical
coverage is not inferred from source compilation.

## Shared scalar/load emission

The JAX authors' [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)
separates tensor expressions from the implementation of their memory accesses.
`_emit_scalar_expression` now owns recursive emission for both the ordinary
expression kernel and the existing bounded indexed-add segment reducer. It takes
the expression, original input dtype metadata, backend language flag, and a
resolver for input values, row/column coordinates, completed reductions and
indexed loads. Indexed-load operands are recursively emitted before resolution;
completed reduction nodes resolve directly to their existing scalar names.
Arithmetic, casts, typed quotient/remainder and block ordinals retain the same
shared emission as before. The resolver describes the existing iteration context;
it does not construct that context or introduce another numerical execution path.

`_indexed_load_expression` takes the actual Ref or Tensor, its retained pointer
table offset, its existing scalar/stride layout, the emitted row/column/mask/other
values, and the backend flag. It preserves direct Ref strides or Tensor block
lookup and per-block strides, promotes real loads to FP32, and retains the
conditional masked load. It allocates no storage and discovers no dependencies.
The ordinary expression source uses this extracted implementation unchanged.

The indexed-add reducer reuses recursive scalar emission with its existing
ordinal and feature-column substitutions. It still iterates exactly from the
segment's lower bound to its upper bound, reads the same selected input refs, and
publishes the same partial. Routing keys, masks, selector generation, reader
lifetimes and supported update operations are unchanged. Source diff review
confirms retained address strings, scalar casts, loop bounds and numerical
expressions; Python compilation passed. No workload was run for this factoring.

Load-valued scatter updates are not enabled by extracting the emitter. Their next
step needs shared predicate/access binding for the same bounded segment domain,
including nested coordinate loads and exact selector ranges. Load-valued masks
must be evaluated per segment rather than holding unrelated destination routing
keys. Selector capacity and lifetime packing also remain explicit setup work;
neither a full-count scan for every segment nor a dense update staging buffer
is an acceptable substitute for that implementation.

## Bounded indexed segment loads

The JAX authors' [Pallas Ref indexing](https://docs.jax.dev/en/latest/pallas/design/design.html#indexing-refs)
composes integer-indexed accesses as numerical expressions. NVIDIA's
[CCCL segmented reduction](https://nvidia.github.io/cccl/unstable/python/compute_api.html)
uses input iterators and explicit segment beginning/ending offsets. The existing
indexed-add lowering now combines these mechanisms without staging update values.
These references motivate the algorithm; mesh's own reader protocol determines
publication and reuse guarantees.

`_indexed_access_paths` is extracted from the existing expression binder and
shared with the segment binder. It retains branch polarity, load masks, nested
coordinate accesses and masked fallback accesses. `_segment_expression` rewrites
update row coordinates through the existing sorted ordinal view and features
through the retained panel origin. Plain operands keep their already-known
chunk/panel Refs and explicit row/column origins; only actual indexed loads keep
whole source tables. An operand used both ways retains both identities rather
than discarding the direct binding.

`_bind_segment_expression` binds the existing exact `[lo, hi)` numerical loop
and recursively creates selectors for dynamic loads. Selector entries use
ordinal-major flattened positions `k*feature_width+c`. Each selector kernel
executes only `[lo*feature_width, hi*feature_width)`, including selectors needed
to evaluate another load's coordinates or predicates. Each numerical or selector
consumer attaches `algebra_indexed_range` with that exact flattened range, actual
candidate pointer positions, and retained source-table block strides. Entries
outside the range are neither initialized nor consumed. The segment activity
count and slot apply to the numerical partial. Derived selectors always publish,
including empty bounded ranges, under the explicit domain lifetime described
below. No whole-chunk coordinate selector becomes an additional readiness barrier.

A setup cache shares identical selector expressions within one segment/panel's
fixed operand/range context. Existing reader fanout retains the selector until
its consumers finish. Flattened bounds are shared by segment and feature width.
Native range endpoints use U32: setup rejects `count*feature_width > UINT32_MAX`
before creating a selector rather than silently wrapping a valid update-count
domain into an invalid flattened range. Selector allocation capacity remains
`count*feature_width` U32 entries per distinct selector in that context, even
though only the bounded interval executes. This is an explicit setup storage
cost; compact capacity packing remains unfinished work.

Original pointwise update expressions retain their direct Ref bindings, scalar
precision, exact segment loops and output partials. Indexed updates can load
values through nested dynamic indices directly from canonical pages, including
conditional load predicates and masked fallback values. The bounded
indexed-validity lowering below moves runtime outer masks into segment
computation without weakening destination closure; constant masks can still
eliminate entries during routing.

Source compilation and diff review are complete. Operational validation uses the
existing streaming-algebra scatter case extended with a lookup tensor, mixed
routing chunks, a delayed selected source page, an empty occurrence and reuse.
No separate evaluator or workload was run by this implementation agent.

## Derived selector active domains

The JAX authors' [Pallas pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html)
and NVIDIA's [CCCL segmented reduction](https://nvidia.github.io/cccl/unstable/python/compute_api.html)
motivate explicit completion lifetimes and bounded input ranges. The following
mechanism is mesh's own implementation of those lifetimes; those publications do
not establish the correctness of mesh's occurrence protocol.

Previously, realization required every selector and bound for an active function
to be inside the same produced output as its active count. This guaranteed that
omission happened after selector publication, and that selector reader lifetimes
also pinned the active-count occurrence. Derived bounded selectors broke those
assumptions. Omitting their producers left selector pages unpublished, while a
later selector reset could erase an already-recorded omission completion.

Bounded selectors now always execute their exact metadata range and publish.
An empty range executes no update loads and initializes no unused entries. Direct
Ref dependencies in these selector functions use the known single-candidate
directory selector and original segment bounds, so an empty range does not wait
for a missing direct operand. Numerical partials alone retain active omission.

`metadata_local` verifies local metadata ownership against receive bindings.
`metadata_descends` checks every ordinary dependency page of immutable configured
metadata producers. Each page must belong to the known count-producing output,
be declared constant, or have an always-producing local ancestor rooted in that
same output. At least one count-rooted path is required; an unrelated ordinary
input cannot be hidden by another direct count input. A setup proof cache rejects
cycles and avoids repeated traversal of shared producers. Omittable producers
cannot establish this ancestry. `indexed_active_domain` requires each derived
selector/bound producer to have that ancestry before replacing containment. It
retains an explicit pointer to the existing active domain and adds its count maps
to indexed lifetime metadata, rejecting candidate overlap. No scheduler or
execution-time graph interpretation is added.

The appended count maps acquire their own reader memberships and reset/watch
edges through existing indexed realization. `vector_maps` continues to describe
only actual selector-vector maps. Existing active readiness already checks count
maps, and indexed readiness checks the extended metadata; ordinary input copies
need no additional count binding. Indexed completion retains the count occurrence
until all selected and unselected candidate memberships have retired, even when
derived selectors live on separate pages.

Indexed retirement recognizes the current active domain's disposition and
omission flags as completion. A derived-selector reset may clear its local
completion bit, but cannot erase that domain fact. Domain count maps remain held
until retirement, so a new count occurrence cannot replace the omission fact
while old candidate readers are outstanding. Existing selector maps still gate
preparation and define the exact flattened span.

The new domain pointer is in the process-local `mesh_indexed_read` structure; it
is not in the shared mapped header or transport records. Both native libraries
must be rebuilt together, but bridge ABI 25 and running bridges remain unchanged.
Native library builds, Python compilation and source diff review passed. The
existing nested-load scatter workflow supplies subsequent runtime evidence,
including empty routing, delayed candidate pages and reuse.

## Bounded indexed validity

The JAX authors' [Pallas Ref indexing](https://docs.jax.dev/en/latest/pallas/design/design.html#indexing-refs)
composes indexed reads with masks; masked values must not introduce the selected
source access when their predicate is false. Indexed-add setup now separates
runtime numerical validity from destination routing. `runtime_mask` identifies
any indexed load or nonconstant direct input in the mask expression, using the
existing constant-extent identities. Those masks become
`select(mask, update_value, 0)` inside the existing bounded segment expression.
Routing keys depend only on normalized destination indices.

The same shared access-path collector propagates mask polarity into nested value
loads and their coordinate loads. A false mask therefore prevents those dynamic
value candidates from becoming readiness requirements. A mask load selects only
the original pages used by that destination's ordinal range and feature panel.
Other destinations can finish while an unrelated mask page remains unpublished.
The exact segment iteration ranges, FP32 real accumulation, integer arithmetic,
and final output casts remain unchanged.

Direct runtime masks also move out of routing, so using a plain mask input does
not introduce a chunk-level routing barrier absent from equivalent indexed
syntax. Such direct masks retain their known chunk/panel Ref bindings and whole
Ref readiness; this does not promise progress within an unpublished mask block.
Boolean direct inputs are allowed in mask expressions. A boolean input also used
as a numerical update operand remains subject to the existing update dtype
contract. This distinction is realized during setup.

Only masks composed entirely from declared constants and static coordinates stay
in routing, where early elimination avoids issuing numerical partials for
eliminated segments. Worst-case partial capacity is still allocated at setup. An
empty routing domain can finish without runtime masks or values. In contrast,
valid destination indices with unknown runtime masks do not establish an empty
domain: their affected destinations must await the relevant mask regions.

The existing scatter workflow supplies delayed mask pages, a false-masked update
whose value lookup points to a delayed page, empty destination routing and
repeated storage reuse. Python compilation and source diff review passed for
this change; operational results are supplied by the parent integration run.

## Xonotic take transpose

The JAX authors describe [scatter dimensions as the mirror of gather dimensions](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py).
For an indexed read y[i] = x[d[i]], its transpose is dx[j] = sum(dy[i] for
i with d[i] = j). Xonotic
`take_along_axis_vjp` now follows that construction through the existing shared
indexed_add rather than its former custom atomic/clear kernel.

`take_coordinates` supplies the same normalized source-coordinate tuple for
forward take and its transpose. It broadcasts index singleton dimensions, maps
source singleton dimensions to zero, and normalizes signed negative indices
once along the selected axis. The derivative decodes each cotangent ordinal,
uses that tuple to construct a scalar destination, and discards out-of-domain
coordinates before linearizing them into a usable destination. The configured
key function reads indices only. Shared indexed_add reads original cotangent
pages through logical indexed loads within each destination's segment; it does
not stage cotangent values into a second tensor.

The primal contributes only shape metadata. Output-root liveness and peer
replication exclude its numerical input for every take transpose, just as for
the existing row-gather transpose. Arbitrary source rank, selected axis and
non-axis broadcast dimensions use the same coordinate formula. Duplicates reduce
through shared segmented sums with the existing FP32 real accumulation and final
output dtype.

Zero/output storage is a scalar column with chunks aligned to complete rows of
the desired physical `(product(prefix_shape), last_dimension)` matrix.
`matrix_view` reframes those chunks individually using their actual View
identities and NumPy stride metadata; it never requests a whole multi-block
region or copies payloads. Tensor/extent/offset identities and the publication
partition remain intact. This retains a normal matrix view for downstream native
contractions and existing host exports.

The custom take-along-axis VJP emitter is removed. General gather transposes
outside the existing efficient row-gather case, and nonvector scatter_add caller
semantics, remain separate migration work. Scalar destination routing also
retains the shared scatter allocator's current worst-case capacity cost; this
change does not claim compact scratch packing. Python compilation and source
diff review passed. The existing optional Xonotic graph supplies repeated
broadcast take gradients, delayed cotangent rows and an absent numerical primal;
operational evidence is recorded by the parent integration run.
