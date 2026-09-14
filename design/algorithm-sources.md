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

`reduce_segment` uses `_candidate_load` for each exact canonical buffer and emits
that scalar value expression inside its ordinal accumulation loop. Half/float
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

`_lower_dot` owns contraction partition and reduction construction. K boundaries
respect the actual left/right backing partitions. Every K panel binds the existing
native `kernels.matmul` implementation and writes an FP32 partial. The existing
native `kernels.add` combines them in the same pairwise tree used by `nn.linear`
previously. The final FP32 result writes the preallocated output directly; a
non-FP32 declared output uses the existing affine conversion exactly once after
that completed region's FP32 sum. A single K panel can write an FP32 output
directly. Its `temporary` and `bind` setup helpers allocate canonical storage and
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
supplies operational CPU/Metal/Core ML validation. Dot epilogue fusion and scratch
lifetime packing remain distinct lowering work.

### Mapped and whole-reference dot operands

`_lower_dot` resolves each non-None input BlockSpec at every configured grid
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
