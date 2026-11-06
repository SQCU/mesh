# Change request: torch programs run as replays of their captured GPU commands

Code is cited at mesh c5782961 and metal-microbench 81a6b59d. Each item gives:
- **Code**: the code as it is;
- **Source**: the published text, library or algorithm, quoted verbatim with where it was read;
- **Change**: what the code must be, which is the source applied and nothing more.

The terms are those of design/problems-of-14b9a757-change-request.md, "Terms": AOT compiled persistent requests,
memory plan, memory schedule and static link schedule, and the AOT allocated page table.

## The turns

> re: point 2: let's treat this as something analogous to the use of cudagraphs in pytorch where we have a foreign
> function api something like a python host can use, compile, and replay for programs invariant up to certain limits
> without recompilation or python-sided or torch-sided overhead thereafter

(2026-09-12)

> is there any reason why you cannot use the exact same 'engine' kernels as litert for all of these?

(2026-10-08)

> why are we letting torch control any resources (after tracing, compilation, lowering, etc, which should be AOT and
> produce serialized objects we cna resume without needing to repeatedly re-compile anything btw) when *we* control
> *our* own code and calling context?

(2026-10-28)

> broadly, i want the meshlib to not define how to compute any linear algebraic function and exclusively take the
> responsibility of defining how collective operations are load balanced across devices, how transactions are load
> balanced along edges, how memory on different machines is consistently described for aot compilation (paging), and
> leaving definitions of functions that run over pages to exist entirely in the data processing headers, which should
> really only be there as helpers for accessing tensors, zero copy, by view, rather than reimplementing or inlining any
> arithmetic related code.

(2026-11-03)

> doing graph capture has a specific relationship to compiling torhc programs into runtimes which don't GIL lock and
> induce latency on dma transactions. there are some requiements for the library code to wrap, transduce, and perhaps
> operation-fuse programs. but that doesn't mean that inlined mps backends are part of the library either, i don't
> think...

(2026-11-03)

> no inlining new numerical backends instead of writing collective and transfer and so on code which schedule the use
> of existing code, basically.

(2026-11-04)

> so can we simply use the existing mechanisms composed s.t. the most recent work presents completed schedules,
> allocation plans, etc. to these apis, resolving this apparent conflict forever ('we can either use torch generally or
> be fast' turned into 'we cna use torch generally *and also* be fast *and also* use the grounded cited scheduler and
> allocation work'). if we can, this is a nearly complete change request document (another one) all by itself.

(2026-11-06)

## How it is implemented

- Each change is the quoted source's mechanism, composed with mechanisms already in the two repositories. Nothing
  else is written: no new kernel, no new allocator, no new wait.
- Language and build mechanics are done as needed.
- A commit names the source it implements.

## 1. Each execution calls torch's operators from Python

- **Code:** rdma/mesh_compile.py:2828 (`_steps`), 2812 (`_instantiate`), 2774 (`_specialized`) and 2975
  (`_execute`). Every execution of a compiled function calls the torch operators of the node's exported program from
  Python (`_called`, 2668).
- **Source:**
  - torch 2.14.0 torch/export/__init__.py:71–73: "`export` takes any nn.Module along with example inputs, and produces
    a traced graph representing only the Tensor computation of the function in an Ahead-of-Time (AOT) fashion, which
    can subsequently be executed with different inputs or serialized."
  - torch 2.14.0 torch/_inductor/cudagraph_trees.py:1–6: "CUDA graph trees are a safety abstraction over CUDAGraphs,
    similar to make_graph_callables, which share the same memory pool. Sharing a memory pool is an extremely important
    optimization when chaining multiple CUDA graphs together, as it prevents you from needing to copy intermediate
    tensors from one graph to the next".
  - metal-microbench metal_recording.h:54, `- (id)capture:(void (^)(void))encode;`, and :112,
    `- (struct MetalReplay)replay:(id)capture;`. tools/e2b_replay.m already runs LiteRT's program this way.
- **Change:**
  - At make(), each node runs its exported program once inside the recorder's `capture:`.
  - Every execution replays that capture (`replay:`, `MetalReplayEncode`). No torch or Python call runs at execution
    time.
  - Writing the capture to disk and reading it back is a further feature, not part of this item.

## 2. The recorder sees torch's MPS commands

- **Code:**
  - torch @08187d9e (the 2.14.0 wheel's `git_version`), aten/src/ATen/mps/MPSDevice.mm:37:
    `NSArray* devices = [MTLCopyAllDevices() autorelease];`
  - The same commit, aten/src/ATen/mps/MPSStream.mm:59:
    `_commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:_commandQueue].retain;`
  - metal-microbench metal_recording.m:3973–3977 interposes both `MTLCreateSystemDefaultDevice` and
    `MTLCopyAllDevices`, so torch's MPS device is the recorder's device.
  - metal-microbench 51e15127 (2026-09-29) recorded torch steps this way: "tp.py and cp.py record one step
    (torch_mesh.record, the process started with the recorder inserted), check its replay at each parts in turn …
    then replay it K times".
- **Source:** dyld, include/mach-o/dyld-interposing.h @fd8d0c4d
  (https://github.com/apple-oss-distributions/dyld/blob/fd8d0c4d52320ebf64db34f3cb280310d905c5ae/include/mach-o/dyld-interposing.h),
  `#define DYLD_INTERPOSE(_replacement,_replacee)`, whose entry is placed
  `__attribute__ ((section ("__DATA,__interpose,interposing")))`.
- **Change:** none to the recorder's entry points. The process that runs make() loads the recorder library
  (DYLD_INSERT_LIBRARIES), as 51e15127 did.

## 3. Captured buffers are bound to the AOT compiled memory plan

- **Code:**
  - metal-microbench metal_recording.h:81, `- (void)place:(id)capture value:(NSUInteger)index in:(id<MTLBuffer>)buffer
    offset:(NSUInteger)offset` and its `stride:copies:` form, which e2b_replay.m uses for LiteRT's buffers.
  - rdma/mesh_compile.py:2505 (`_bind`) binds torch tensors over placed pages by DLPack.
- **Source:**
  - ExecuTorch "Concepts", Memory planning: "a memory planning pass is run before the graph is saved to flatbuffer.
    This assigns a memory ID to each tensor and an offset in the buffer".
  - torch's CUDA graph trees, as quoted in item 1: one memory pool, no copy of intermediates between graphs.
- **Change:** each buffer torch allocates during the capture is bound at replay to its value's pages in the AOT
  compiled memory plan, with `place:value:`. Execution t binds copy t mod copies.

## 4. A value read as one buffer

- **Code:**
  - Torch's MPS operators read each tensor as one MTLBuffer with an offset and strides (PyTorch v2.11.0
    aten/src/ATen/native/mps/OperationUtils.mm: "Starting with macOS 15.0, MPS supports native strides directly in the
    kernels").
  - The AOT compiled memory plan places a value's pages anywhere in the region.
- **Source:** Metal, macOS 26.4 SDK:
  - MTL4CommandQueue.h:394: "Updates multiple regions within a placement sparse buffer to alias specific tiles from a
    Metal heap."
  - MTL4CommandQueue.h:406: `- (void)updateBufferMappings:(id<MTLBuffer>)buffer heap:(nullable id<MTLHeap>)heap
    operations:(const MTL4UpdateSparseBufferMappingOperation [_Nonnull])operations count:(NSUInteger)count;`
  - MTL4UpdateSparseBufferMappingOperation: "Metal walks the tiles in the range in buffer offset order, assigning the
    next tile from the heap in increasing order, starting at ``heapOffset``."
  - MTLResource.h:196–198: `MTLSparsePageSize16`, `MTLSparsePageSize64`, `MTLSparsePageSize256`.
  - PagedAttention: "block tables—the mapping between logical and physical KV blocks of each request".
- **Change:**
  - Each value is one range of a placement sparse buffer. That range's tiles alias the value's pages, mapped once at
    make() from the AOT allocated page table by `updateBufferMappings:`.
  - A kernel reads the value as one buffer with an offset and strides. No bytes are copied.
- **What the implementer reads from these headers and the RDMA headers before writing:**
  - whether the region the transport registers can be the placement heap whose tiles are aliased;
  - which `placementSparsePageSize` matches the region's page size.

## 5. Starts and waits in the replayed stream

- **Code:**
  - metal-microbench metal_recording.h:89–90, `received:operand:remote:` and `published:operand:supplied:`, which mark
    a captured buffer as landed or sent.
  - tools/e2b_replay.m issues each execution's start from the GPU stream (a listener on the step event calling
    `mesh_host_started`) and waits before a landed value's first use (`waitForEvent:value:`).
- **Source:**
  - MPI 4.1 §3.9, MPI_START: "The request becomes active once the call is made. [...] The communication buffer should
    not be modified after the call, and until the operation completes."
  - NCCL User Guide, CUDA Stream Semantics: "NCCL calls are associated to a stream which is passed as the last
    argument of the collective communication function. The NCCL call returns when the operation has been effectively
    enqueued to the given stream, or returns an error."
  - torch 2.14.0 torch/distributed/_functional_collectives.py, `AsyncCollectiveTensor`: "A Tensor wrapper subclass that
    is used to trigger a call to wait prior to first use of the underlying tensor."
  - M. Lam, PLDI 1988, §1: "iterations of a loop in a source program are continuously initiated at constant intervals
    without having to wait for preceding iterations to complete."
- **Change:** the torch capture gets the hooks the LiteRT capture has:
  - each collective's landed operand is marked `received:` and its sent operand `published:`;
  - each execution's AOT compiled persistent requests are started from the GPU stream, `lead` executions ahead;
  - each landed value is waited for before its first use.

## 6. The engine's existing kernels as torch's MPS implementations

- **Code:**
  - metal-microbench matrix_shaders.swift:112–148 and kernels.swift:3190–3193 run Metal Performance Primitives
    `matmul2d` with cooperative tensors.
  - rdma/mesh_compile.py runs torch's own MPS kernels for every operator.
- **Source:**
  - Metal Performance Primitives, MPPTensorOpsMatMul2d.h (macOS 26.4 SDK): "User can apply post processing in-register
    where GEMM output is computed using cooperative_tensor."
  - torch 2.14.0 torch/library.py, `impl`: "Register an implementation for a device type for this operator. … Some
    valid types are: "cpu", "cuda", "xla", "mps", "ipu", "xpu"."
  - torch 2.14.0 torch/mps/__init__.py:140, `compile_shader`: "Compiles compute shader from source and allows one to
    invoke kernels defined there from the comfort of Python runtime".
- **Change:**
  - For each ATen operator that an existing engine kernel computes, the kernel's source is compiled by
    `torch.mps.compile_shader` and registered for "mps" by `torch.library.impl`.
  - The capture of item 1 then records that kernel's dispatches.
  - No kernel is written. An operator with no engine kernel keeps torch's own.

## What this does not change

- **The transport.** The bridge and its files are untouched.
- **What is computed before execution 0.** The AOT compiled persistent requests, memory plan, memory schedule, static
  link schedule and page table are computed as now. This request only changes what consumes them: the replay,
  instead of torch calls from Python.
