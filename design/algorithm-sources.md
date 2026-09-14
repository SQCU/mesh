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

The JAX authors' Pallas pipelining (cited in the region streaming review) motivates
the comparison in `examples/streaming-overlap.py`. Amdahl's AFIPS 1967 analysis
of serial fractions motivates reporting end-to-end improvement, rather than
calling pending work device utilization. The client uses `BlockSpec.region_map`
and `call_native` to run the same tiled NumPy matmuls for `(X W) V`: whole-operand
publication versus independent section publication. The first contraction runs
on participant zero, the second on participant one, with results returned over
canonical mesh. Configuration, oracle computation, and numerical comparison are
outside the timed interval. Five window traversals warm up the pipeline before samples. Welford's online
moments (cited above) summarize timings. `mesh_tensor_present` and `Ref.present` observe canonical physical presence
without consuming a reader or gating numerical functions. They report whether a
consumer launches before full reception. Inputs vary by invocation; terminal
results are checked outside steady-state timing. Multiple configured input slots
keep both variants pipelined.

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
Old application imports and scan callers still require removal or rewriting;
this checkpoint is not the completed caller migration or gold demonstration.

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
Each index message is one 4096-byte frame and holds up to 170 transfer tuples;
partially occupied batches have this same frame cost. Payloads retain the
configured transport block size. Queue counters track outstanding work requests
and descriptor storage reuse, not the destination of a numerical result.

`link_configure`, `link_index_offset`, `link_indices`, `link_post`, `link_receive`,
`link_queue`, `link_error`, and `mesh_progress` implement this software push
mechanism using the MPI asynchronous communication agent and NCCL proxy principle
cited in ledger D2, and the two-sided SEND mechanics of TN3205. Reuse still
requires a target's existing readers to finish before it can be overwritten;
finite pipelines should allocate distinct buffers for independent value instances.

The exact FFN → RMSNorm → summed learned embedding → FFN → RMSNorm distributed
operational demonstration remains outstanding. Compilation alone does not
establish overlap or a throughput improvement.

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
