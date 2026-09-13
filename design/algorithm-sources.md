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
outside the timed interval. Five warmups precede samples. Welford's online
moments (cited above) summarize timings. The whole-input observer is a canonical
reader used only to report whether a consumer launches before full reception;
it does not gate the numerical functions.
