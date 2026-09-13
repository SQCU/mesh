# CPU streaming and the operand-copy source audit

The direct CPU backend executes the same indexed program using canonical tensor
addresses. Producer completion publishes a part; a consumer uses that part while
other inputs remain absent. The copy analysis below is a source argument. Running
the acceptance checks numerical results and early progress; it does not discover
or prove the absence of copies inside another implementation.

## CPU execution

`mesh_algebra_create_cpu` realizes CPU execution before tensor allocation and
function binding. It creates no Metal device, queue, library, or buffer wrapper.
`cpu_part` selects the numerical operation and FP16/FP32 load/store functions at
configuration time. Invocation does indexed scalar loads, arithmetic and stores;
it neither selects a backend nor allocates operand storage.

All existing algebra operations have CPU implementations: affine, weighted add,
multiply, tanh, exp, row sum, reciprocal square root, and contraction. The CPU
contraction uses the same input views and K partitions as the other backends.
FP16 input loads promote to FP32; contraction accumulation and the configured
reduction tree use FP32. This source-visible implementation establishes the
streaming and storage contract; it is not a claim of the fastest CPU GEMM.

The operation's output is divided by the existing binder into publication parts.
The existing static scan issues a CPU part only when its indexed inputs are
present. The CPU computes that part synchronously and invokes the shared completion
path. Completion publishes its output and configured send work before the scan
continues. The independent bridge can transmit those pages while later CPU parts
execute. There is no CPU task queue or separate participant scheduler.

For the existing program, let Q1 be available while Q0 is absent:

1. The CPU affine producer publishes Q1's transformed input and its send.
2. Peer arrival enables CPU sum and tanh for the corresponding coordinates.
3. The CPU contraction consumes Q1 and publishes its complete contribution.
4. That contribution reaches the peer before Q0 is written.
5. Only the first 256 rows of Q0 are then produced. CPU arithmetic consumes them
   and the first contraction output part reaches the peer while row 257 is absent.
6. Producing the last input row completes the remaining contraction and reduction.

Both directions occur in numerical operators, including the contraction itself.
The observer does not insert publication callbacks into those operators.

Build the existing executable and use the same three-queue bridge configuration:

```sh
make -C rdma .build/streaming-algebra
rdma/.build/streaming-algebra 0 19 257 8 0 cpu
rdma/.build/streaming-algebra 1 19 257 8 1 cpu
```

The peers use explicit and symbolic setup respectively; both execute entirely on
CPU. Reports expose `cpu_submissions` alongside total completed operations. The
GPU duration and Core ML submission count must be zero for this configuration.
No throughput comparison is asserted.

## Address and lifetime proof

The inspected owners are `rdma/mesh-algebra.m`, `rdma/mesh-dataflow.c`,
`rdma/mesh-flow.c`, `rdma/mesh-verbs.h` and `rdma/mesh.h`.

| Boundary | Source mechanism | Consequence |
|---|---|---|
| Shared region | `mesh_attach` maps the shared-memory file using `MAP_SHARED` | The client mapping names the bridge's shared backing |
| Tensor ownership | `mesh_tensor_create` allocates canonical rows/pages and calls `mesh_map` | Each logical tensor row names its actual backing page |
| Dense numerical view | `mesh_view_create` reserves an address interval, then replaces it with `MAP_SHARED | MAP_FIXED` mappings of those page offsets | The dense view is an alias, not a copied tensor; the initial anonymous reservation holds no operand data |
| CPU inputs | `cpu_operand` retains that address, offset and strides; `cpu_get` calls a typed scalar load | Each numerical input is loaded from the canonical view |
| CPU outputs | `cpu_part` calls the preselected typed store at the part's physical flat output index | Results are written directly into the canonical output pages |
| Visibility | `emit_part` invokes `mesh_complete` after the stores; publication sets atomic presence/send bits | A finished CPU part becomes available to its readers and the bridge |
| Transmission | `link_post` constructs its SGE from `M + data_off + page * pgsz`; `region_sge` supplies that region's registered key | The work request names those payload pages, with no mesh packing buffer |
| Reception | The receive work request uses the configured destination page address | Incoming values land in their canonical consumer storage |
| Reuse | Numerical and transport readers set their existing read planes | A producer reuses storage after its actual readers finish |

More explicitly, element byte index b in an allocated extent's dense alias names
shared-file offset `data_off + extent.page * pgsz + b`. `mesh_map` installs those
same page indices for its logical rows. A SEND for its block names the bridge's
mapping at that same file offset. The virtual addresses may differ between
processes; their backing file offsets agree.

The entire CPU numerical call graph is visible: `cpu_part`'s realized evaluator,
`cpu_get`, the typed loads/stores, scalar arithmetic, and scalar libm calls.
The libm calls receive floating-point values, not tensor pointers. There is no
call to BLAS, Core ML, a tensor allocator, payload memcpy, packer or tensor-sized
scratch routine in that execution path. Captured blocks contain pointers,
geometry and function references; they do not capture tensor payloads by value.
A scalar accumulator or compiler register spill is not another operand store.

`mesh_algebra_contract` allocates named K contributions and reduction results
before invocation. These are numerical outputs in the canonical arena, not hidden
copies of an operand. The `memcpy` in `mesh_realize` copies reader-mask metadata,
not tensor bytes. Numerical affine identity operations explicitly requested by
the caller still perform their declared copy; the proof excludes *implicit*
operand staging, not the mathematical copy operation.

This proves absence of operand staging in the source-owned CPU numerical path
and mesh's user-space transport binding. Kernel/driver implementation details
behind memory mapping and verbs are outside this source scope; work-request
addresses establish the registered-buffer interface rather than a claim about
unavailable driver internals.

## What the Core ML source actually establishes

The Core ML path is different at a specific call boundary:

- `native_array` wraps the canonical pointer, offset, shape and strides in an
  `MLMultiArray`. Mesh makes no input payload copy there.
- `native_part` supplies an output `MLMultiArray` pointing into the output part
  through `MLPredictionOptions.outputBackings`. Mesh adds no copied-output fallback.
- `mesh_coreml.compile_part` constructs a graph of slices, matmuls, casts,
  reductions, reshape and possibly concatenation. Those nodes describe numerical
  values; they do not specify Core ML's physical memory assignment. A reshape or
  cast node alone is not proof of an allocated copy, since lowering may alias or fuse it.
- Invocation calls Apple's `predictionFromFeatures:options:completionHandler:`.
  The inspected SDK provides its declaration, not the native runtime implementation
  that selects input packing, intermediate allocation and output stores.

The source boundary is therefore identified, not deferred to another experiment.
The available coremltools generator source is not the implementation of this
native prediction method. Numerical runs, compute-device preferences, and output
object identity cannot fill that missing source-level memory assignment.

The SDK's `MLPredictionOptions.h` explicitly describes output backing as proposed
storage that the framework may decline. Returning the supplied object establishes
that public output identity; it does not establish how the framework computed
its contents. The callback's identity check remains an output-placement check,
not a copy-free certificate. No Core ML/ANE copy-free claim follows from the
existing adapter source.

Similarly, `newBufferWithBytesNoCopy` proves the supplied Metal buffer aliases
mesh's storage. The explicit elementwise shader accesses that buffer directly.
MPS contraction's internal implementation is not present here; the wrapper alone
does not prove that MPS never uses private packing or numerical workspace.

For the direct CPU path the source-level copy question is closed within the scope
above. For opaque native prediction and MPS internals, source-backed proof would
require the actual memory-lowering implementation or a binding contract that
specifies those internal accesses. Repeating numerical runs is not the next step
for that proof, and streaming remains mandatory for these adapters regardless.
