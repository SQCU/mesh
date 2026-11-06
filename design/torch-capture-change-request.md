# Change request: torch programs run as replays of their captured GPU commands

Code is cited at mesh c5782961 and metal-microbench 81a6b59d. Code marked "first integration" is cited at mesh
165f2c98 and metal-microbench 0771d07a. Each item gives:
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

- **Code:**
  - rdma/mesh_compile.py:2828 (`_steps`), 2812 (`_instantiate`), 2774 (`_specialized`) and 2975 (`_execute`). Every
    execution of a compiled function calls the torch operators of the node's exported program from Python (`_called`,
    2668).
  - First integration: the recorder runs a fill once, during the capture, and refuses every other copy
    (metal_recording.m:3032, "blit command"). It makes its own pipelines with `supportIndirectCommandBuffers`
    (metal_recording.m:430–434), and raises "Pipeline does not support indirect commands" for any other (:3713), such
    as the one MPS's softmax uses. rdma/mesh_compile.py:3804 refuses a compiled function called more than once in a
    block.
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
  - Metal, macOS 26.4 SDK:
    - MTLIndirectCommandBuffer.h:17–30, `MTLIndirectCommandType`, "A bitfield of commands that may be performed
      indirectly": draws and dispatches, and no copy.
    - MTLComputePipeline.h:70–73, `supportIndirectCommandBuffers`: "This flag makes this pipeline usable with indirect
      command buffers."
    - MTL4ComputeCommandEncoder.h:323: "Encodes a command that copies data from a buffer instance into another." :456:
      "Encodes a command that fills a buffer with a constant value for each byte."
  - MPI 4.1 §3.9: "Often a communication with the same argument list (with the exception of the buffer contents) is
    repeatedly executed within the inner loop of a parallel computation. In such a situation, it may be possible to
    optimize the communication by binding the list of communication arguments to a persistent communication request
    once and then repeatedly using the request to start and complete operations."
- **Change:**
  - At make(), each node runs its exported program once inside the recorder's `capture:`.
  - Every execution replays that capture (`replay:`, `MetalReplayEncode`). No torch or Python call runs at execution
    time.
  - Every command the program issues is replayed:
    - Copies and fills are encoded in the replay's compute encoder, between the ranges of its indirect command buffers,
      because an indirect command buffer can't hold them.
    - A pipeline made through the recorder's device is made with `supportIndirectCommandBuffers`.
    - A dispatch whose pipeline lacks that flag is encoded directly.
  - A function called k times in a block is started k times each execution, as a persistent request is used
    repeatedly.
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
  - First integration: rdma/mesh_compile.py:3811 gives every value of a capture the capture's whole range of levels
    (`value.begin, value.end = 0, max(levels, 1) - 1`), so no two of its values share pages.
- **Source:**
  - ExecuTorch "Concepts", Memory planning: "a memory planning pass is run before the graph is saved to flatbuffer.
    This assigns a memory ID to each tensor and an offset in the buffer".
  - ExecuTorch exir/memory_planning.py:339–342 (at 472953c1), `update_tensor_lifetime`: "Update the lifetime of the
    tensor to cover node_idx. A tensor's lifetime are represented by the index of the first and last node referring
    that tensor in its inputs/outputs."
  - torch's CUDA graph trees, as quoted in item 1: one memory pool, no copy of intermediates between graphs.
- **Change:**
  - Each buffer torch allocates during the capture is bound at replay to its value's pages in the AOT compiled memory
    plan, with `place:value:`. Execution t binds copy t mod copies.
  - A buffer's lifetime runs from the level of the first recorded command that binds it to the level of the last. The
    AOT compiled memory schedule then gives the same pages to buffers whose lifetimes don't overlap.

## 4. A value read as one buffer

- **Code:**
  - Torch's MPS operators read each tensor as one MTLBuffer with an offset and strides (PyTorch v2.11.0
    aten/src/ATen/native/mps/OperationUtils.mm: "Starting with macOS 15.0, MPS supports native strides directly in the
    kernels").
  - The AOT compiled memory plan places a value's pages anywhere in the region.
  - rdma/mesh-dataflow.c:208 (`mesh_view_compose`) reserves one range of addresses and remaps the AOT allocated page
    table's rows over it, in row order and with no copy (its note, :187–206). mesh-collective.h:719
    (`mesh_view_values`) places each value's copies one after another in that range. rdma/mesh_compile.py:2416 wraps
    the range in one MTLBuffer, and each value is bound at its offset in it.
- **Source:**
  - The macOS SDK's mach/mach_vm.h:270, the declaration of `mach_vm_remap`, which mesh-dataflow.c:187–206 quotes;
    its `boolean_t copy` operand is FALSE there.
  - Metal, macOS 26.4 SDK, MTLDevice.h:691–694, `newBufferWithBytesNoCopy:length:options:deallocator:`: "Create a
    buffer by wrapping an existing part of the address space."
  - PagedAttention: "block tables—the mapping between logical and physical KV blocks of each request".
  - Why a placement sparse buffer can't do this, in the same SDK:
    - Such a buffer aliases "specific tiles from a Metal heap" (MTL4CommandQueue.h:394), and the heap must be "An
      MTLHeap you allocate with type MTLHeapTypePlacement" (:403).
    - The only way to make a heap is `newHeapWithDescriptor:` (MTLDevice.h:673–676). Its descriptor gives the
      "Requested size of the heap's backing memory" (MTLHeap.h:46–49) and has no field for memory that already exists.
    - The page table's pages are in the region before make(), so they are not tiles of any heap.
- **Change:** none.
  - Pages are named by their index in the AOT allocated page table, and the compute side maps them by those indices
    alone. It never reads the transport's registration, its DMA addresses or any firmware pointer.
  - A kernel reads each value as one buffer with an offset and strides. No bytes are copied.

## 5. Starts and waits in the replayed stream

- **Code:**
  - metal-microbench metal_recording.h:89–90, `received:operand:remote:` and `published:operand:supplied:`, which mark
    a captured buffer as landed or sent. Each holds one word for the whole buffer.
  - metal_recording.m:1919 (`MetalMaskReaders`) splits each command that reads a landed buffer into a masked first pass
    and a later pass. The first pass is written from the command's MSL source, and where no source was recorded the
    function raises (:1938).
  - MetalReplayEncode commits the first passes, then `[queue waitForEvent:event value:base + p + 1]`, then pass p's
    later passes (metal_recording.h:157–158).
  - rdma/mesh-metal.m:256 (`mesh_metal_waits_start`) sets that event to t × passes + p + 1 once pass p's count words
    reach their targets. It walks repetitions from 0 up to a count fixed when it starts, and a wait ends with ETIMEDOUT
    after MESH_PLATFORM_NS without progress (:213).
  - tools/e2b_replay.m issues each execution's start from the GPU stream (a listener on the step event calling
    `mesh_host_started`). It marks its landed and sent operands (:1623, :1629) and starts the waits (:2098).
  - torch @08187d9e aten/src/ATen/native/mps/OperationUtils.mm:964 loads torch's kernels with
    `library = [device newLibraryWithData:getSectionData(section_name) error:&error];`, so none of them has a source.
  - First integration:
    - A node where a declared transfer names one of its values keeps its torch steps (mesh fcf5d18a).
    - Each execution's start is passed a `lead` of 0 (rdma/mesh_compile.py:3314).
    - Each collective's range is written by torch's `copy_` (rdma/mesh_compile.py:1341, 1751, 2811–2842), which is a
      copy, and item 1 covers copies.
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
  - Each collective's landed range is marked `received:` and each sent range `published:`, with one word per range,
    because one value can hold several ranges (an AllGather's result) or send several (a ReduceScatter's).
  - Each landed value is waited for before its first use. A reader with no recorded source runs whole in the later
    pass, after MetalReplayEncode's `waitForEvent:value:`, and has no masked first pass.
  - Each execution's AOT compiled persistent requests are started from the GPU stream, the function's `lead`
    executions ahead.
  - Execution t's waits begin once its requests have started ("The request becomes active once the call is made"), so
    the number of executions doesn't have to be fixed before the first one.
  - A node with transfers then replays, just as a node without them does.

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
