# Algorithm references

The [user's requirements](collective-goals.md) define scope. These sources supply
mechanisms, not additional features, architecture, tests or prerequisites. Section
names retain existing source citation anchors; they do not prescribe public APIs.
The deleted implementation is not an implementation template.

## Program

The JAX authors, [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html):
reference for higher-order numerical calls over indexed tensor operands.
The implementation realizes a finite value-index extent once. A section's
`first + index * stride` selects its logical row; shared constants have zero
stride. Submission publishes root indices into existing numerical-worker queues,
while consumers are indexed by operand publication. No function scan or repeated
realization is required. The [execution description](async-collectives.md#execution-and-ownership)
separates shared function metadata from each value's operands and uses.

## Program.tensor

The JAX authors, [BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html),
and Apple [mmap](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/mmap.2.html):
references for indexed sections and virtual mappings of actual shared backing.
George E. Collins, [A method for overlapping and erasure of lists](https://doi.org/10.1145/367487.367501)
(1960): reference counting. The user explicitly requested automatic ownership
release and background pool return, without caller free/done calls.
Contiguous sections use chunk-indexed backing and relative numerical byte offsets,
with one presence bit and ownership record per numerical value. The
[address and ownership derivation](pages-and-functions.md#block-addressing)
shows the receive permutation and the producer's single initial reference.
These are mesh's representation choices, not new algorithms attributed to Pallas
or Collins. Transport chunk counts do not define tensor partitions.
The Berkeley/Apple [mmap specification](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/man/man2/mmap.2)
provides shared mappings of file-backed pages. Apple's
[pshm_mmap implementation](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/posix_shm.c)
maps the selected object offsets with copy-on-write disabled. `wire_map` uses those mappings to
give the device a payload-plus-tag address sequence and the numerical function
a dense address sequence over the same payload pages. Aliases are fixed at bridge
setup. Runtime receive assignment changes page indices, not virtual mappings.
Metal buffers cover each operand's page-aligned window, using Apple's
[no-copy buffer API](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:)).
The whole-arena Metal buffer and transport-capacity query have been removed.

## Program.kernel_call

Yousef Saad, [Iterative Methods for Sparse Linear Systems, second edition](https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf),
SIAM (2003), section 3.4: compressed row storage uses row offsets to index contiguous
entries. Mesh applies that representation to declared consumer references, not
numerical sparse-matrix computation. `mesh_calls_start` counts references, forms
prefix offsets and scatters call pointers into those ranges before transport
activation. Publication traverses only its row's range. Setup also prepares local
operand addresses and records received-input positions; invocation resolves only
those positions. A receive channel's contiguous backing makes native-view selection
an offset divided by the transport block size, eliminating a second index table.

Apple [Metal command buffers](https://developer.apple.com/documentation/metal/mtlcommandbuffer)
and [Core ML prediction](https://developer.apple.com/documentation/coreml/mlmodel):
existing numerical submission and completion interfaces.
Apple [pointer-backed MLMultiArray](https://developer.apple.com/documentation/coreml/mlmultiarray/init(datapointer:shape:datatype:strides:deallocator:))
and [outputBackings](https://developer.apple.com/documentation/coreml/mlpredictionoptions/outputbackings):
Core ML operand interfaces. Their contracts govern the selected backend's actual
operands; they do not require a mesh-owned executor or function scan.
Apple's `MLPredictionOptions.h`, available with the macOS SDK, describes
`outputBackings` as proposed storage: a model lacking support can return separately
allocated output. The supplied function's actual output-storage contract governs
Core ML use. The current path publishes the declared Mesh output after native
completion; it does not turn separately allocated model output into a registered
operand.
Apple [MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)
accepts independently prepared input and result matrices. The SDK's `MPSKernel.h`
describes reuse after encoding and separate kernel instances for concurrent host
encoders. Each mesh call already has one numerical worker. The existing matrix
implementation now binds input/output arrays separately and encodes their indices
through one operation; Core ML selects its prepared feature provider and output
options independently. `MeshBindings` contains the corresponding operand views
and index function. This removes the Cartesian product of address-bound calls,
without changing the numerical algorithm or requiring additional synchronization.
`TensorFunction` owns a preparation function with one view factory per operand.
Each input and output is prepared independently; no unary function assumption
or product of operand address combinations is introduced. Calls, maps and
reductions use this same function value, including when the reduction allocates
its intermediate outputs. The former native-only `Mesh.map` overload is removed.
Shared constants are inputs with zero index stride. All preparation completes
in `start()`; runtime submission receives the realized operand arrays through
the resolved `MeshInvocation.submit` closure. It does not invoke preparation
factories or choose a backend during numerical execution.
Apple's [makeCommandBuffer](https://developer.apple.com/documentation/metal/mtlcommandqueue/makecommandbuffer())
documents blocking when a queue has no free command buffers. The SDK's
`MTLDevice.h` exposes `newCommandQueueWithMaxCommandBufferCount` and specifies
64 as the ordinary queue's default capacity. `TensorFunction.metal` takes a
device and encoder; `MeshInvocation` owns a private queue for that declared call,
with capacity for its `count` invocation records. Both direct calls and native
view factories use this constructor. Setup creates all `count` command buffers;
submission indexes that array using the existing C call index. At the k-th
construction, k-1 buffers occupy a pool of count slots, with k <= count. No
completion is needed to provide a free slot, and unrelated callers cannot occupy
this pool. Creation does not enqueue a buffer; encoding and commit occur on
submission, so an earlier unused index does not hold a queue position.
The former caller-provided queue and example-specific `3 * count` workaround are
removed. This establishes the bound for the current declared extent, not a
completed reusable-stream lifecycle or a claim about all native launch costs.

## Program.copy

Apple [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt),
MLX authors' [JACCL transport](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/jaccl/lib/jaccl/rdma.h),
and rdma-core's [ibv_post_recv](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_post_recv.3):
registered SEND/RECV and work completion. The [hardware notes](collective-dependency-ledger.md)
distinguish substrate facts from the retained bridge's protocol decisions.
TN3205 explicitly limits this transport to `IBV_WR_SEND`; its SDK enum for
`IBV_WR_SEND_WITH_IMM` does not establish hardware support. Both TN3205 and JACCL
show local `wr_id` values returned with completions. Mesh uses these identifiers
for buffer lifetime. Every chunk carries a source-chunk row tag; setup maps that
row to a destination chunk and, for the final chunk, the numerical publication.
The tag arrives in the payload's own work request. There is no index QP or
cross-QP identity join. On September 15, `ibv_devinfo -v` reported `max_sge: 1`
on the local Thunderbolt devices. The alias representation uses one SGE and
does not infer two-entry support from the general verbs API. The
[record layout and execution path](async-collectives.md#execution-and-ownership)
describe this mesh-specific representation. Receive storage is preallocated
across the finite extent; refill does not depend on consumer completion or page
reclamation.
The dedicated TX worker starts posting as each send edge is enqueued, then polls
completion queues. It does not construct a runtime list of all chunks before
posting the first. Completion processing refills
available send slots directly. RX independently refills receives before publishing
the received section. No completion wait or new scheduler is introduced.

The compressed-row representation cited under [numerical submission](#programkernel_call)
also stores each published row's send edges contiguously. Each queue has a
preallocated FIFO of edge indices; its capacity is the declared send count.
Receive targets are grouped by source-chunk row with one advancing cursor per
group, retaining a target for each declared copy. The contiguous receive page run
is addressed arithmetically. These replace linked send/target records and the
receive-address array without changing SEND/RECV ordering or collective semantics.

The rdma-core authors' [queue-pair creation](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_create_qp.3)
and [memory registration](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_reg_mr.3)
associate QPs and registered memory with a protection domain. Mesh now keeps
those device resources once per named device while realizing a separate connection
and TX/RX progress pair for every configured link. Configured peer identities,
channels and operand row ranges are resolved at setup. Publication scatters its
row index to those links' queues through a precomputed membership bitset, independent
of collective verb and peer count. Multiple links register the same canonical
payload backing through their selected devices; they do not create operand copies.
The [configured-peer description](async-collectives.md#configured-peers) records
setup, per-link preposting, shared registration, retirement and configuration.

## Program.write

The JAX authors' [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html):
operand production. The user's contract requires publication of completed sections.
Apple's [vDSP_vramp](https://developer.apple.com/documentation/accelerate/vdsp_vramp)
is the existing numerical generator called by the configured Core ML chain's
initial producers. It writes directly into each registered operand.

## Program.export

MLX authors' [Metal evaluation](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/eval.cpp):
reference for retaining operands through actual operation completion.

## kernels.expression

The JAX authors' [Pallas indexing](https://docs.jax.dev/en/latest/pallas/design/design.html#indexing-refs):
values, indices and masks. This is not a requirement for a general expression compiler.

## kernels.dot

Dongarra, Du Croz, Hammarling and Duff, *A Set of Level 3 Basic Linear Algebra
Subprograms* (1990), Apple [MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication),
[BNNS matmul](https://developer.apple.com/documentation/accelerate/bnnsfiltercreatelayerbroadcastmatmul(_:_:))
and [Core ML MIL operations](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html):
existing contractions. The [reduction algebra](distributed-reduce.md) describes
contributions without prescribing intermediate-buffer or launch counts.

## kernels.add

The JAX authors' [Pallas accumulation](https://docs.jax.dev/en/latest/pallas/pipelining.html#reductions-and-accumulation):
reference for combining numerical contributions.
The engine's existing `encAdd` encoder and `add_inplace` shader now live in its
shared matrix source files, retaining their existing arithmetic and launch
geometry. Callers prepare their pipeline and dimensions and supply `encAdd` with
the actual operand buffers and offsets. Mesh implements no addition kernel;
the unused `MatrixOperations.add` convenience wrapper has been removed.
Apple's [vDSP_vadd](https://developer.apple.com/documentation/accelerate/vdsp_vadd)
is the existing float32 vector sum supplied to reduce-scatter by the configured
Core ML caller. It receives resolved input/output pointers and introduces no
payload allocation, copy, or replacement sum implementation in Mesh.

## collective.reduce_scatter

Rabenseifner, *Optimization of Collective Reduction Operations* (2004), and the
MPI Forum's [collectives](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node114.htm):
reduction semantics and decompositions. A source does not mandate one decomposition
for every collective or caller placement.

## Collective movement

The MPI Forum's [collectives](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node114.htm)
and MLX's [distributed operations](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/ops.cpp):
[distinct communication relations](collective-verbs.md).

## nn.ffn

Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053), and MLX's
[tensor-parallel layers](https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/distributed.py):
local numerical functions composed with collectives. Hendrycks and Gimpel,
[GELU](https://arxiv.org/abs/1606.08415): the activation already supplied by the engine.
The configured function caller uses `Cij = Fij(Xi)` and `Yj = sum_i Cij`, then
passes each `Yj` to its supplied consumer. For linear T, Fij is its input-i,
output-j block. Reduce-scatter places each sum at its caller-selected owner;
there is no assembled whole-tensor intermediate or global stage barrier.
The [caller description](function-chain.md) distinguishes this source composition
from model artifacts and runtime performance evidence.

## nn.rmsnorm

Zhang and Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
(2019): normalization used by the existing numerical implementation.

## nn.embedding

The JAX authors' Pallas indexing and the llama.cpp authors'
[GGUF format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md):
references for indexed operands and the existing model loader.

## collective.sync_on_remote_fill

The MPI Forum's [communication completion](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node74.htm):
explicit completion operations. The user requested an explicit counterexample;
no default collective may invoke it.
