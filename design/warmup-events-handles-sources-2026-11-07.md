# Warm-up before capture, events in a capture, handle maps, API coverage: sources, quoted verbatim

Retrieved 2026-11-07. Each excerpt below was copied by script from the cited lines of the file named; nothing in a block is retyped. PyTorch was fetched with `curl` from `raw.githubusercontent.com/pytorch/pytorch/<sha>/<path>` into `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/lit5/torch/`. The CUDA guide, apitrace and GFXReconstruct are the downloads design/capture-replay-sources-2026-11-07.md names (`lit3/cr-cuda/`, `lit3/cr-apitrace/`, `lit3/cr-gfxr/`), at the versions and commits it gives. Dawn is the checkout metal-microbench links e2b with, `.build/Dawn`. The Metal headers are the local SDK's.

## 1. PyTorch: warm-up before capture

- **Source:** PyTorch, https://github.com/pytorch/pytorch, branch `main`, commit `0c7eac65dffe892199f608583aaa0b4c025e9c4c` (2026-11-07). License: BSD-3-Clause (`LICENSE` L1–3 below).

`LICENSE` L1–3:

```text
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
```

`docs/source/notes/cuda.md` L1575–1579, "CUDA Graphs":

```markdown
After capture, the graph can be *launched* to run the GPU work as many times as needed.
Each replay runs the same kernels with the same arguments. For pointer arguments this
means the same memory addresses are used.
By filling input memory with new data (e.g., from a new batch) before each replay,
you can rerun the same work on new data.
```

`docs/source/notes/cuda.md` L1607–1610, "PyTorch API":

```markdown
{class}`torch.cuda.graph` is a simple, versatile context manager that
captures CUDA work in its context.
Before capture, warm up the workload to be captured by running
a few eager iterations. Warmup must occur on a side stream.
```

`docs/source/notes/cuda.md` L1683, L1691–1692, "Constraints":

```markdown
* Dynamic control flow (based on CPU or GPU data) is prohibited, unless it is based on GPU data and implemented via higher order operator
* CPU work is not captured. If the captured ops include CPU work, that work will be elided during replay.
* Every replay reads from and writes to the same (virtual) memory addresses.
```

`docs/source/notes/cuda.md` L2071–2073:

```python
# First call will run eager for warmup, second call will do graph
# capture followed by graph replay, third call and beyond will do
# just graph replay.
```

`torch/cuda/graphs.py` L1716–1718, in `make_graphed_callables`:

```python
    # Warmup
    # Hopefully prevents cudnn benchmarking and other lazy-initialization cuda work
    # from ending up in any captures.
```

`docs/source/user_guide/torch_compiler/torch.compiler_cudagraph_trees.md` L15, L55–59, L76:

```markdown
- All input arguments to kernels are fixed to what they were recorded
# the first run warms up each graph, which does things like CuBlas or Triton benchmarking
foo(torch.arange(0, 10, device="cuda"))
# The second run does a CUDA Graph recording, and replays it
foo(torch.arange(0, 10, device="cuda"))
# Finally we hit the optimized, CUDA Graph replay path
First, we would hit the optimized, CUDAGraph.replay() path that we have already recorded in graph 1. Then we would hit Graph 3. Just as before, we will need to warm up the graph once before recording. On the warmup run, the memory addresses are not fixed, so graph 4 will also fallback to the inductor, non-cudagraph invocation.
```

## 2. NVIDIA CUDA Programming Guide v13.5: events in a capture

- **Source:** as design/capture-replay-sources-2026-11-07.md §4 (v13.5, "Last updated on Oct 31, 2026"), the page read as text (`lit3/cr-cuda/cuda-graphs.txt`). Cited by section and by the text file's lines.

`cuda-graphs.txt` L162–165, §4.2.2.1.2.1:

```text
##### 4.2.2.1.2.1. Cross-stream Dependencies and Events#
Stream capture can handle cross-stream dependencies expressed with cudaEventRecord() and cudaStreamWaitEvent(), provided the event being waited upon was recorded into the same capture graph.
When an event is recorded in a stream that is in capture mode, it results in a captured event. A captured event represents a set of nodes in a capture graph.
When a captured event is waited on by a stream, it places the stream in capture mode if it is not already, and the next item in the stream will have additional dependencies on the nodes in the captured event. The two streams are then being captured to the same capture graph.
```

`cuda-graphs.txt` L199, §4.2.2.1.2.2 (heading at L193):

```text
It is invalid to merge two separate capture graphs by waiting on a captured event from a stream which is being captured and is associated with a different capture graph than the event. It is invalid to wait on a non-captured event from a stream which is being captured without specifying the cudaEventWaitExternal flag.
```

`cuda-graphs.txt` L533, L549, L551, §4.2.3.2, node types:

```text
### 4.2.3.2. Individual Node Update#
Event record node
Event wait node
```

## 3. apitrace: the handle map and address regions

- **Source:** as design/capture-replay-sources-2026-11-07.md §2 (commit `3a0ee82a1bcad6930090af643f73532204017650`, MIT).

`retrace/retrace_swizzle.hpp` L37–41, L110, L131, L135:

```cpp
/**
 * Handle map.
 *
 * It is just like a regular std::map<T, T> container, but lookups of missing
 * keys return the key instead of default constructor.
addRegion(trace::Call &call, unsigned long long address, void *buffer, unsigned long long size);
toPointer(trace::Value &value, bool bind = false);
addObj(trace::Call &call, trace::Value &value, void *obj);
```

## 4. GFXReconstruct: handle IDs; code generated from the API registry

- **Source:** as design/capture-replay-sources-2026-11-07.md §1 (commit `f9b0fb0d7c8b3293c7ec3edec5115da877831d18`, MIT).

`framework/decode/common_handle_mapping_util.h` L38, L54:

```cpp
static typename T::HandleType MapHandle(format::HandleId             id,
            GFXRECON_LOG_WARNING("Failed to map handle for object id %" PRIu64, id);
```

`framework/generated/generate_vulkan.py` L33–34, L41:

```python
KHRONOS_REGISTRY_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, '..', '..', 'external', 'Vulkan-Headers', 'registry')
# File names to provide to the Vulkan XML Registry generator script.
```

## 5. Dawn as e2b links it: the queue's event

- **Source:** metal-microbench `.build/Dawn`, commit `466163dc7bf94146ac9a921b86cc3ed08184ebb8 (2026-07-21)`. License: BSD 3-Clause, as design/capture-replay-sources-2026-11-07.md §5.

`src/dawn/native/metal/QueueMTL.mm` L78–79, L224, L246–248:

```objc
    // Don't free `mMtlSharedEvent` because it can be queried after device destruction for
    // synchronization needs. However, we destroy the `mSharedFence` to release its device ref.
    ExecutionSerial pendingSerial = GetPendingCommandSerial();
    DAWN_ASSERT(mSharedFence);
    [*pendingCommands encodeSignalEvent:mSharedFence->GetMTLSharedEvent()
                                  value:static_cast<uint64_t>(pendingSerial)];
```

`src/dawn/native/metal/CommandRecordingContext.mm` L207–215, in `WaitForSharedEvent`:

```objc
    // Skip the wait if it's for the same shared event as the queue i.e. a self wait. These can
    // happen if the client passes us the same shared event that we gave it back to us. If these
    // events are waited on, they seem to cause waitUntilScheduled to block for previous command
    // buffers to complete due to dependencies between consecutive command buffers.
    if (sharedEvent != mQueue->GetMTLSharedEvent()) {
        // There may be an open blit encoder from a copy command or writeBuffer.
        // Wait events are only allowed if there is no encoder open.
        EndBlit();
        [*mCommands encodeWaitForEvent:sharedEvent value:signaledValue];
```

## 6. Metal, macOS SDK 26.4.1

- **Source:** Apple, Metal.framework headers, `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Metal.framework/Headers/`.

`MTL4ComputeCommandEncoder.h` L31, L743–745:

```objc
@protocol MTL4ComputeCommandEncoder <MTL4CommandEncoder>
- (void)writeTimestampWithGranularity:(MTL4TimestampGranularity)granularity
                             intoHeap:(id<MTL4CounterHeap>)counterHeap
                              atIndex:(NSUInteger)index;
```

`MTL4CommandEncoder.h` L97–99, L113–115:

```objc
/// Encode a barrier that guarantees that any subsequent work you encode in the *current command encoder*,
/// corresponding to `beforeEncoderStages`, doesn't begin until all prior commands in this command encoder,
/// corresponding to `afterEncoderStages`, completes.
- (void)barrierAfterEncoderStages:(MTLStages)afterEncoderStages
              beforeEncoderStages:(MTLStages)beforeEncoderStages
                visibilityOptions:(MTL4VisibilityOptions)visibilityOptions;
```

`MTL4ArgumentTable.h` L59, L69–70, L84–86, L96–97, L107–108, L118–119:

```objc
@protocol MTL4ArgumentTable <NSObject>
- (void)setAddress:(MTLGPUAddress)gpuAddress
           atIndex:(NSUInteger)bindingIndex;
- (void)setAddress:(MTLGPUAddress)gpuAddress
   attributeStride:(NSUInteger)stride
           atIndex:(NSUInteger)bindingIndex;
- (void)setResource:(MTLResourceID)resourceID
      atBufferIndex:(NSUInteger)bindingIndex;
- (void)setTexture:(MTLResourceID)resourceID
           atIndex:(NSUInteger)bindingIndex;
- (void)setSamplerState:(MTLResourceID)resourceID
                atIndex:(NSUInteger)bindingIndex;
```

`MTL4CommandBuffer.h` L136–142, L172–173:

```objc

/// Marks a residency set as part of the command buffer's execution.
///
/// Ensures that Metal makes resident the resources that residency set contains during execution of this command buffer.
///
/// - Parameter residencySet: ``MTLResidencySet`` instance to mark resident.
- (void)useResidencySet:(id<MTLResidencySet>)residencySet;
- (void)writeTimestampIntoHeap:(id<MTL4CounterHeap>)counterHeap
                       atIndex:(NSUInteger) index
```

`MTL4CommandQueue.h` L318:

```objc
- (void)addResidencySet:(id<MTLResidencySet>)residencySet;
```

`MTLResidencySet.h` L41–44:

```objc
 @abstract A residency set is responsible for managing resource and heap residency and is referenced
 by a command buffer or command queue in order to ensure that resources and heaps are resident.
 Resources and heaps are added and removed uncommitted and a subsequent commit call applies all
 of the changes in bulk.
```

`MTLComputeCommandEncoder.h` L263–269, the classic encoder:

```objc
 * @method useResource:usage:
 * @abstract Declare that a resource may be accessed by the command encoder through an argument buffer
 * 
 * @discussion For tracked MTLResources, this method protects against data hazards. This method must be called before encoding any dispatch commands which may access the resource through an argument buffer.
 * @warning Prior to iOS 13, macOS 10.15, this method does not protect against data hazards. If you are deploying to older versions of macOS or iOS, use fences to ensure data hazards are resolved.
 */
- (void)useResource:(id <MTLResource>)resource usage:(MTLResourceUsage)usage API_AVAILABLE(macos(10.13), ios(11.0));
```

Declarations in the Metal 4 headers named for managed-resource synchronization, stage-in regions or texture access counters: `grep -n "^- .*synchronize\|^- .*[sS]tageIn\|^- .*AccessCounters" MTL4*.h` in the headers directory prints nothing (exit 1).
