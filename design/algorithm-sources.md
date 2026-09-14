# Algorithm sources

The [asynchronous collective contract](async-collectives.md) defines the scope.
This is a bibliography indexed by source citations, not an implementation plan.

## Independent verbs progress

- [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)

## Operand matching and storage

Gregory Papadopoulos and David Culler, *Monsoon: an Explicit Token-Store Architecture*, ISCA 1990.

## Receive storage

Apple, [TN3205: Low-latency communication with RDMA over Thunderbolt](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt).

## Nonblocking table ownership

Gregory Papadopoulos and David Culler, *Monsoon: an Explicit Token-Store Architecture*, ISCA 1990.

Apple, [TN3205: Low-latency communication with RDMA over Thunderbolt](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt).

## Registered-span defect, closed

Apple, [TN3205: Low-latency communication with RDMA over Thunderbolt](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt).

## Regions follow blocks

Apple, [TN3205: Low-latency communication with RDMA over Thunderbolt](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt).

## Attach takes over a dead holder

POSIX process-lifetime queries and shared-memory ownership.

## Registered memory views

Apple, XNU virtual-memory implementation: shared mappings of the same backing pages.

## Independent configured programs

Gregory Papadopoulos and David Culler, *Monsoon: an Explicit Token-Store Architecture*, ISCA 1990.

## Literal weight pages

Gregory Papadopoulos and David Culler, *Monsoon: an Explicit Token-Store Architecture*, ISCA 1990.

## Configured binding identities

Apple, [TN3205: Low-latency communication with RDMA over Thunderbolt](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt).

## Column and row tensor composition

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

## Streaming algebra

- [TileLink: Generating Efficient Compute-Communication
Overlapping Kernels using Tile-Centric Primitives](https://arxiv.org/abs/2503.20313)
- [FLUX](https://arxiv.org/abs/2406.06858)
- [Pallas collective matmul](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)
- [nested pipelines](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html#nested-remote-and-local-dma-pipelines)
- [Metal synchronization events](https://developer.apple.com/documentation/metal/about-synchronization-events)

## Pallas indexed destinations

- [Pallas collective matmul](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)
- [TPU buffering discussion](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html#double-buffering)

## Mandatory partial publication

- [Pallas collective matmul](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)

## Backend-independent producer and consumer streaming

The JAX authors, [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html) and [software pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html).

## CoreML partial execution

- [prediction and compiled-model documentation](https://apple.github.io/coremltools/docs-guides/source/model-prediction.html)
- [outputBackings contract](https://developer.apple.com/documentation/coreml/mlpredictionoptions/outputbackings)

## CPU indexed execution

Gregory Papadopoulos and David Culler, *Monsoon: an Explicit Token-Store Architecture*, ISCA 1990.

Jack Dongarra, Jeremy Du Croz, Sven Hammarling and Iain Duff, *A Set of Level 3 Basic Linear Algebra Subprograms*, ACM TOMS 1990.

## CPU library contraction

Apple/MLX authors, [CPU BNNS GEMM implementation](https://github.com/ml-explore/mlx/blob/main/mlx/backend/cpu/gemms/bnns.cpp),
and Apple, [BNNSMatMul](https://developer.apple.com/documentation/accelerate/bnnsmatmul(_:_:_:_:_:_:_:_:)).
Mesh replaces its handwritten NEON half/mixed-precision contraction with the
same Accelerate BNNS numerical implementation used by MLX for half precision.
`bnns_operand` translates an existing registered view to BNNS dimensions,
strides, type and data pointer. Setup queries and allocates workspace once for
the configured calls. Invocation passes that workspace explicitly, avoiding
BNNS's optional internal workspace allocation. FP32 contractions retain BLAS.
No operand is copied into a second tensor store. MLX uses the filter interface;
mesh uses the direct workspace-taking API to realize storage before invocation.
These Accelerate APIs are deprecated in favor of BNNSGraph, but remain provided
by the SDK and used by upstream; this change does not introduce a graph compiler.

## Indexed library functions

The JAX authors, [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html) and [software pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html).

Gregory Papadopoulos and David Culler, *Monsoon: an Explicit Token-Store Architecture*, ISCA 1990.

## Region streaming review

The JAX authors, [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html) and [software pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html).

## Pallas call ergonomics

- [Grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
- [pallas_call reference](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.pallas_call.html)

## Single kernel interface

The JAX authors, [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html) and [software pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html).

## Streaming FFN

Jack Dongarra, Jeremy Du Croz, Sven Hammarling and Iain Duff, *A Set of Level 3 Basic Linear Algebra Subprograms*, ACM TOMS 1990.

Guy Blelloch, *Prefix Sums and Their Applications*, CMU-CS-90-190, 1990.

## Memory warning

Apple, XNU `rusage_info_v2.ri_phys_footprint`, libproc process enumeration and Mach `host_statistics64`.

## Publication layout

The JAX authors, [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html) and [software pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html).

## Application Metal kernels

The JAX authors, [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html) and [software pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html).

Apple, Metal argument buffers and resource usage declarations.

## Xonotic planner migration

Jack Dongarra, Jeremy Du Croz, Sven Hammarling and Iain Duff, *A Set of Level 3 Basic Linear Algebra Subprograms*, ACM TOMS 1990.

The JAX authors, [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html) and [software pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html).

## Xonotic frame migration

The JAX authors, [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html) and [software pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html).

Apple, [TN3205: Low-latency communication with RDMA over Thunderbolt](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt).

## Presence-driven execution

Gregory Papadopoulos and David Culler, *Monsoon: an Explicit Token-Store Architecture*, ISCA 1990.

## Async index push contract

Apple, [TN3205: Low-latency communication with RDMA over Thunderbolt](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt).

The JAX authors, [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html) and [software pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html).

## Explicit operand metadata

- [NumPy ndarray strides](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html)
- [Pallas BlockSpec](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)

## Streamed normalization and embedding

- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [NumPy basic indexing](https://numpy.org/doc/stable/user/basics.indexing.html)
- [Pallas grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)

## Direct indexed gather

- [Array programming with NumPy](https://doi.org/10.1038/s41586-020-2649-2)
- [numpy.copyto](https://numpy.org/doc/stable/reference/generated/numpy.copyto.html)

## Declared partition and ownership retention

The JAX authors, [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html) and [software pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html).

## Pallas panel composition

- [Pallas matrix multiplication](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html)
- [software-pipelining derivation](https://docs.jax.dev/en/latest/pallas/pipelining.html)
- [collective matmul example](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)

The MLX authors' [tensor-parallel layers](https://ml-explore.github.io/mlx/build/html/examples/tensor_parallelism.html)
and [implementation](https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/distributed.py)
supply the column-sharded up-projection followed by a row-sharded down-projection.
`examples/streaming-chain.py` uses that decomposition with caller-supplied hidden
partition J_p: D_p = swish(X W_up[:, J_p]) W_down[J_p, :]. Both peers compute
independently from replicated X; the root adds their matching D regions. Unlike
MLX's sharded-to-all layer, this example places the reduced result only at the
caller-selected root. There is no hidden-activation gather or stage-to-stage
handoff. JAX supplies the block-indexed calling syntax; Accelerate BLAS and MPS
supply the existing local contractions. This is a composition of those mechanisms,
not a new numerical or distributed algorithm. The example feeds each completed
reduced region through swish and a further BLAS/MPS/BNNS contraction, using an
explicit consumer-weight operand.

MLX/JACCL source inspected in the local `~/mlx` checkout at `d142de6`:
[`JACCLGroup`](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/jaccl/jaccl.cpp)
dispatches whole-array collectives through its CPU encoder;
[`MeshImpl`](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/jaccl/lib/jaccl/mesh_impl.h)
copies into registered send buffers and local staging, polls completions, reduces
chunks and drains outstanding sends before returning. Its internal pipelining is
not a region-publication API for surrounding numerical producers and consumers.
Wrapping that call would retain the staging and whole-call completion boundary.
Mesh instead binds existing numerical functions to canonical registered regions
and uses its existing SEND/RECV completion publication.

Apple's [Metal command-buffer submission](https://developer.apple.com/documentation/metal/mtlcommandbuffer/commit())
and asynchronous CoreML prediction already supply asynchronous execution.
Realization now binds the worker dispatch only for synchronous CPU functions.
`submit_ready` invokes that prebound submission function directly. Metal/CoreML
no longer take an extra global-worker dispatch before their own submission.
The dispatch group retains in-flight lifetimes for teardown; it is not a barrier
between numerical regions. This change removes a queue hop, not all device or
runtime costs.

## Region expression fusion

- [Triton: an intermediate language and compiler for tiled neural
network computations](https://doi.org/10.1145/3315508.3329973)
- [Triton Layer Normalization implementation](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
- [JAX Pallas BlockSpecs](https://docs.jax.dev/en/latest/pallas/quickstart.html)

## Publication work lists

Duplicate row notices already queued need no additional socket wake. The queue
insertion reports whether it added work; publication wakes the handler only if
it added a compute notice. Installation sends an initial wake for notices queued
before the socket existed. Clearing `queued` precedes examining a row, so a
publication during examination can enqueue it and send another wake. A failed
nonblocking send caused by a full socket leaves an existing wake to drain; the
handler drains the notice list again for every batch of socket messages.

- [lockless list API](https://raw.githubusercontent.com/torvalds/linux/master/include/linux/llist.h)
- [implementation](https://raw.githubusercontent.com/torvalds/linux/master/lib/llist.c)
- [NCCL proxy](https://raw.githubusercontent.com/NVIDIA/nccl/master/src/proxy.cc)
- [Triton normalization tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)

## Contraction accumulation

- [Pallas mixed-precision matmul](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html#bfloat16-matrix-multiplication)
- [Triton matmul](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)
- [typed execution](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html)
- [conversion source](https://github.com/apple/coremltools/blob/main/coremltools/converters/_converters_entry.py)

## Indexed expression lowering

- [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)

## Actual frame capacity

- [TN3205, queue-pair allocation and completion polling](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)

## Dynamic indexed expression lowering

- [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)
- [Grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)

## Dynamic reader lifetimes

- [Monsoon: An Explicit Token-Store Architecture](https://www.cs.cmu.edu/~18742/papers/Papadopoulos1990.pdf)
- [Pallas bounded dynamic slices](https://github.com/jax-ml/jax/blob/main/docs/pallas/tpu/pipelining.md)

## CPU register contraction

- [Anatomy of High-Performance Matrix Multiplication](https://doi.org/10.1145/1356052.1356053)
- [Advanced SIMD intrinsic reference](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html)

## Segmented indexed add

- [Prefix Sums and Their Applications, sections 1.3 and 1.5](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
- [scatter-add contract](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter_add.html)

## Xonotic block indexed lowering

- [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)

## Fused indexed update values

- [Pallas pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html)

## Shared contraction lowering

- [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)
- [Pallas pipelining and accumulation](https://docs.jax.dev/en/latest/pallas/pipelining.html#reductions-and-accumulation)
- [encode contract](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication/encode(commandbuffer:leftmatrix:rightmatrix:resultmatrix:)
- [typed execution](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html)
- [Pallas BlockSpec documentation](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
- [Pallas reductions and accumulation discussion](https://docs.jax.dev/en/latest/pallas/pipelining.html#reductions-and-accumulation)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Pallas matrix multiplication tutorial](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html)
- [Pallas grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)

## Canonical reader groups

- [Monsoon](https://www.cs.cmu.edu/~18742/papers/Papadopoulos1990.pdf)

Papadopoulos and Culler's operand-associated presence supplies the lifetime
mechanism. `mesh_reads_reset` uses mesh's already-realized reader mask, including
transport readers, to clear only planes assigned to the overwritten region.
Both numerical claims and receive completions use this same operation. No reader
outside that mask is consulted by the realized region's consumers. Adding a
reader is configuration work; its assigned plane is cleared on the next write.
The previous unconditional 64-plane atomic sweep is removed from both paths.

## Xonotic shared indexing

- [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)

## Shared sparse routing lowering

- [segmented sparse matrix operations](https://www.cs.cmu.edu/~scandal/papers/CMU-CS-93-173.html)
- [Pallas reference and
index-map design](https://docs.jax.dev/en/latest/pallas/design/design.html)
- [Monsoon](https://www.cs.cmu.edu/~18742/papers/Papadopoulos1990.pdf)
- [Metal residency sets](https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets)

## Active segment domains

- [Megablox grouped multiplication](https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/main/src/maxtext/kernels/megablox/backend.py)
- [gather transpose implementation](https://raw.githubusercontent.com/jax-ml/jax/main/jax/_src/lax/slicing.py)

## Xonotic output liveness

- [canonicalization](https://mlir.llvm.org/docs/Canonicalization/#globally-applied-rules)

## FFN shared expression composition

- [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)

## RMSNorm shared expression composition

- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)

## Typed FFN expression composition

- [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)
- [pipelining and accumulation](https://docs.jax.dev/en/latest/pallas/pipelining.html#reductions-and-accumulation)

## Xonotic logical indexing

- [Pallas BlockSpec indexing](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
- [advanced indexing](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing)
- [take_along_axis](https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html)

## Shared scalar/load emission

- [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)

## Bounded indexed segment loads

- [Pallas Ref indexing](https://docs.jax.dev/en/latest/pallas/design/design.html#indexing-refs)
- [CCCL segmented reduction](https://nvidia.github.io/cccl/unstable/python/compute_api.html)

## Derived selector active domains

- [Pallas pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html)
- [CCCL segmented reduction](https://nvidia.github.io/cccl/unstable/python/compute_api.html)

## Bounded indexed validity

- [Pallas Ref indexing](https://docs.jax.dev/en/latest/pallas/design/design.html#indexing-refs)

## Xonotic take transpose

- [scatter dimensions as the mirror of gather dimensions](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py)

## Xonotic gather transpose

- [gather/scatter transpose implementation](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py)

## Xonotic row scatter

- [scatter-add and gather/scatter transpose implementation](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py)

## Xonotic partitioned reshape

- [Pallas indexing design](https://docs.jax.dev/en/latest/pallas/design/design.html#indexing-refs)
- [reshape contract](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)

## Segment range identity specialization

- [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)

## Segment selector common subexpressions

- [MLIR common-subexpression elimination pass](https://mlir.llvm.org/docs/Passes/#-cse)

## Xonotic neighborhood algebra

- [segment_sum](https://docs.jax.dev/en/latest/_autosummary/jax.ops.segment_sum.html)
- [Pallas grid and index maps](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
- [common-subexpression elimination](https://mlir.llvm.org/docs/Passes/#-cse)

## Explicit index vector domains

- [Pallas vector indexing design](https://docs.jax.dev/en/latest/pallas/design/design.html#indexing-refs)

## Column index vector domains

- [Pallas Ref indexing design](https://docs.jax.dev/en/latest/pallas/design/design.html#indexing-refs)

## Xonotic expert indexed contractions

- [Pallas indexed Ref design](https://docs.jax.dev/en/latest/pallas/design/design.html#indexing-refs)
- [segment_sum](https://docs.jax.dev/en/latest/_autosummary/jax.ops.segment_sum.html)

## Xonotic ranked indexed reductions

- [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)

## Xonotic batched contractions

- [matmul contract](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.matmul.html)

## Xonotic logical pointwise

- [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)

## Selected native contractions

- [Pallas scalar-prefetch and sparse computation](https://docs.jax.dev/en/latest/pallas/tpu/sparse.html)

## Shared associative reductions

- [Prefix Sums and Their Applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
- [Pallas software
pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html)
- [Metal Shading Language Specification,
June 4, 2026, pages 206–207](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)

## Shared elementary functions

- [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)
- [fdlibm log1p discussion](https://www.netlib.org/fdlibm/s_log1p.c)
- [NIST DLMF exponential series, equation 4.2.19](https://dlmf.nist.gov/4.2.E19)
- [fdlibm asinh discussion](https://www.netlib.org/fdlibm/s_asinh.c)
- [elementary math implementation](https://github.com/numpy/numpy/blob/main/numpy/_core/src/npymath/npy_math_internal.h.src)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [C11 draft N1570, F.10.4.4](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
- [scalar-exponent loop specialization](https://github.com/numpy/numpy/blob/main/numpy/_core/src/umath/loops_umath_fp.dispatch.c.src)

## Indexed range generation

- [arange implementation](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py)
- [prefix sums and their applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
- [mean implementation](https://github.com/ml-explore/mlx/blob/main/mlx/ops.cpp)

## Typed integer contractions

- [Pallas tiled matmul](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html)
- [C11 draft N1570, section 6.2.5 paragraph 9](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
- [GraphBLAS C API specification](https://graphblas.org/docs/GraphBLAS_API_C_v2.1.0.pdf)

## Stable indexed ordering

- [Sorting networks and their applications (1968)](https://www.cs.kent.edu/~batcher/sort.pdf)
- [GPU Merge Path (ICS 2012)](https://davidbader.net/publication/2012-gm-ba/2012-gm-ba.pdf)
- [stable argsort contract](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argsort.html)
- [partition contract](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argpartition.html)

## Canonical view replication

- [Pallas indexing design](https://docs.jax.dev/en/latest/pallas/design/design.html#indexing-refs)
- [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)

## Static indexed access specialization

- [Pallas indexing design](https://docs.jax.dev/en/latest/pallas/design/design.html#indexing-refs)

## Composable indexed contractions

- [Pallas tiled matmul](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html)
- [Linalg dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)

## Grouped segment reductions

- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [Megablox transposed grouped multiplication](https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/main/src/maxtext/kernels/megablox/backend.py)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)

## Indexed contraction plans

- [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)

## In-operation publication

- [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)

## View-scoped consumption

Gregory Papadopoulos and David Culler, *Monsoon: an Explicit Token-Store Architecture*, ISCA 1990.

## Page-table backing assignment

Apple, XNU virtual-memory implementation: shared mappings of the same backing pages.

## Literal contiguous materialization

- [ascontiguousarray documentation](https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html)
- [Pallas collective matmul](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)

## View-scoped host production

Gregory Papadopoulos and David Culler, *Monsoon: an Explicit Token-Store Architecture*, ISCA 1990.

Mesh owns each host writer binding for the lifetime of its algebra context.
The caller retains an opaque handle, without duplicating row maps or function
layouts. Issue claims the configured region only after its previous readers
finish. Successful completion publishes that region. Unsuccessful host production
releases its producing claim while leaving presence absent: no unfinished value
is made available, and a later write can retry. This applies to host writes,
which do not publish sections before their context exits; it is not cancellation
of an asynchronously executing kernel or of an already published region.

## Compiled row access domains

- [Grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)

## Compiled column access domains

- [Grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)

## Page-derived reduction leaves

The JAX authors, [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html) and [software pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html).

Guy Blelloch, *Prefix Sums and Their Applications*, CMU-CS-90-190, 1990.

## Realized numerical invocation

The JAX authors, [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html) and [software pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html).

## Recorded Metal commands

- [CPU encoding of indirect command buffers](https://developer.apple.com/documentation/metal/encoding-indirect-command-buffers-on-the-cpu)

## Page-indexed gather dependencies

- [gather](https://docs.jax.dev/en/latest/_autosummary/jax.lax.gather.html)
- [Pallas pipelining guide](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html)

## Independent kernel submission

Apple, Metal `MTLCommandQueue`, and the MLX authors' stream-based execution
(`mlx/backend/metal/device.cpp`). Commands within one queue execute in submission
order. `mesh_algebra_kernel` starts a native queue while configuring a public
kernel call; its bound functions retain that queue. Invocation uses the retained
queue without selecting or allocating one. Different kernel calls therefore have
no shared command-queue ordering dependency; canonical operand presence controls
when their regions become issuable. This permits GPU overlap, not a promise that
the device concurrently executes every ready command. CPU dispatch is unchanged.

## Collective

Rabenseifner (2004), *Optimization of Collective Reduction Operations*; Patarasuk and Yuan (2009), *Bandwidth optimal all-reduce algorithms for clusters of workstations*: `reduce_scatter` reduces each caller-owned block, `all_gather` distributes those blocks, and `all_reduce` composes both using existing copies and additions; Tensor views retain the same registered Refs without copying their operands.

## Partial

PyTorch DTensor authors, `Partial` placement, and the Legion authors, reduction privileges: setup-only `Partial` marks unresolved K contributions; disjoint contribution sets combine under addition and the marker disappears only when all required terms are included.

## Program.kernel_call

Papadopoulos and Culler, *Monsoon: an Explicit Token-Store Architecture* (ISCA 1990), for operand-driven firing; Apple, *Metal Programming Guide*, command submission and completion handlers, for device-write visibility. `mesh_algebra_encode` binds a caller-supplied encoder to canonical input/output views at setup. Mesh invokes it with a command buffer when those inputs are present, commits the buffer and publishes the output rows on successful completion. The encoder contains numerical work only: it does not commit, wait, publish or manage mesh readers. Each binding describes one independently usable tensor region; callers split larger operations into region bindings. Captured pipelines and operand storage are realized before invocation and retained by the binding. Generated pointwise functions use this same entry point; external numerical functions need no source compiler or operation enum.

`mesh_algebra_buffer` returns the existing Metal buffer of a canonical tensor extent. The engine’s `MatrixView` initializer applies the configured region’s element offset and strides to that same buffer; it creates no backing or copied operand. The caller supplies the scalar type. The program owns the extent and must remain alive through numerical completion.

`Program(functions=...)` realizes caller-supplied bindings for public `kernel_call` operations. A binding receives resolved input/output Refs once per configured grid point and registers its numerical function before invocation. Logical addition keeps its setup-time Partial checks even when its numerical implementation is supplied by the caller. The streaming-chain `--numerics` option supplies `gemma_mesh_add` from the engine library: it binds the engine’s existing FP16/FP32 addition kernels through `mesh_algebra_encode`. The engine addition shaders now take an explicit left operand; existing in-place callers bind their destination as that operand. FP16 uses FP32 addition followed by the existing FP16 store, and FP32 uses FP32 addition. The shared `encAdd` launch replaces the two prior host launch functions. No Python callback participates in numerical execution.

A supplied `kernels.dot` binding receives each configured K-partition contribution from the existing expression decomposition. Mesh assigns its Partial marker and retains the addition dependencies; the supplied function computes only that contribution. `gemma_mesh_mps_dot` uses the engine’s existing `MatrixOperations.multiply` MPS backend (Apple, *MPSMatrixMultiplication*) on canonical `MatrixView`s. It captures the realized matrix operation in the encoder, with no backend choice or matrix construction during numerical invocation. The example’s numerical-library configuration supplies both addition and contraction this way.

The default contraction binding calls `mesh_algebra_contract`, which realizes the existing Accelerate, BNNS, MPS or Core ML matrix operation for each publication region. Its signature contains matrix views and the product scale, with no operation selector or accumulation parameter. Each contribution writes fresh output storage; addition remains a separately bound function.

Transport completion publishes only the canonical atomic presence and notice structures in shared memory. A dedicated `mesh.presence` pthread in the compute process spins on its notice head and delivers available numerical work to that process’s existing execution queue. No socket notification, application callback, or numerical dispatch runs on the RDMA progress thread. This uses the existing Monsoon-style presence mechanism and POSIX threads, replacing the Unix-datagram wakeup path.
