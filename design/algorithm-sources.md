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

## Program.kernel_call

Apple [Metal command buffers](https://developer.apple.com/documentation/metal/mtlcommandbuffer)
and [Core ML prediction](https://developer.apple.com/documentation/coreml/mlmodel):
existing numerical submission and completion interfaces.
Apple [pointer-backed MLMultiArray](https://developer.apple.com/documentation/coreml/mlmultiarray/init(datapointer:shape:datatype:strides:deallocator:))
and [outputBackings](https://developer.apple.com/documentation/coreml/mlpredictionoptions/outputbackings):
Core ML operand interfaces. Their contracts govern the selected backend's actual
operands; they do not require a mesh-owned executor or function scan.

## Program.copy

Apple [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt),
MLX authors' [JACCL transport](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/jaccl/lib/jaccl/rdma.h),
and rdma-core's [ibv_post_recv](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_post_recv.3):
registered SEND/RECV and work completion. The [hardware notes](collective-dependency-ledger.md)
distinguish substrate facts from the retained bridge's protocol decisions.
TN3205 explicitly limits this transport to `IBV_WR_SEND`; its SDK enum for
`IBV_WR_SEND_WITH_IMM` does not establish hardware support. Both TN3205 and JACCL
show local `wr_id` values returned with completions. Mesh uses these identifiers
for buffer lifetime and removes its duplicate completion FIFO. Its registered
record reserves four bytes for an immutable source-row tag, and setup maps peer
source rows to local receive uses. Per-queue frame counts are realized from the
configured payload sizes. The tag arrives in the payload's own work request, so
the previous index QP, index messages and cross-QP join are deleted. The
[record layout and execution path](async-collectives.md#execution-and-ownership)
describe this mesh-specific representation. Receive storage is preallocated
across the finite extent; refill does not depend on consumer completion or page
reclamation.

## Program.write

The JAX authors' [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html):
operand production. The user's contract requires publication of completed sections.

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
