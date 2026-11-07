Sources for the revisions of design/torch-capture-change-request.md, retrieved 2026-11-06/07 by a retrieval agent and recorded here as it wrote them. The local paths it names are the session's download directories.

# Capture and replay: sources, quoted verbatim

Retrieved 2026-11-06/07. Repositories were fetched at the commits named below: either a depth-1 `git fetch` of the commit, or `raw.githubusercontent.com/<repo>/<sha>/<path>`. Web pages were fetched with `curl`. The downloads are in `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/lit3/cr-*/`.

Each `>` excerpt was checked by script to occur, after whitespace normalization, in the cited lines. Each code block was extracted from the cited lines unchanged, except that bullets in pages converted from HTML show as `* `. The NVIDIA pages were read as HTML, so they are cited by section number. Excerpts are deliberately short: the sentence or lines that state the point.

---

## Difference 1: one-time calls (where a capture starts, the state before it, what is replayed repeatedly)

### 1. GFXReconstruct

- **Source:** LunarG, *GFXReconstruct*, https://github.com/LunarG/gfxreconstruct, branch `dev`, commit `f9b0fb0d7c8b3293c7ec3edec5115da877831d18` (2026-11-06). License: MIT (`LICENSES/LICENSE.txt`, "Copyright (c) 2018-2020 LunarG, Inc.").
- **Where read:** a depth-1 fetch, in `lit3/cr-gfxr/gfxreconstruct/`. There is no separate trimming document at this commit. Trimming is documented in `USAGE_desktop_Vulkan.md` and in `docs/GFXReconstruct Architectural Principles.md`.

Trimmed capture (`USAGE_desktop_Vulkan.md` L498–499, "Trimmed Captures"):

> Trimmed captures are created when GFXR is configured to start capturing at some later time in execution.

`GFXRECON_CAPTURE_FRAMES` and `GFXRECON_CAPTURE_TRIGGER` (`USAGE_desktop_Vulkan.md` L290 and L293, capture-options table):

> Specify one or more comma-separated frame ranges to capture.  Each range will be written to its own file.

> Note that frame numbering is 1-based (i.e. the first frame is frame 1).

> Specify a hotkey (any one of F1-F12, TAB, CONTROL) that will be used to start/stop capture.

State at the trim start, as captured (`USAGE_desktop_Vulkan.md` L1231–1233, "Trimmed File Optimization"; `vulkan_dump_resources.md` L356):

> For trimmed capture files, a snapshot of the Vulkan API state is written at the start of the file. This state snapshot includes the data for all buffers and images that were live at the time that capture started.

> The state setup block exists only for trimmed captures and its purpose is to initialize all Vulkan objects that have been created outside the captured region.

Architectural Principles (`docs/GFXReconstruct Architectural Principles.md` L96, L113, L157):

> Because of the nature of trimming, Graphics API calls issued before the beginning of the trim range are likely to be rearranged and many omitted.

> GFXReconstruct may make API calls to query state, read memory contents, and synchronize with the underlying implementation.

> For trimmed sequences, GPU memory and graphics API state in replay at the beginning of the sequence are, as much as possible, the same as they would be if previous commands had not been trimmed.

The snapshot writer (`framework/encode/vulkan_state_writer.cpp` L169–179, inside `VulkanStateWriter::WriteState`):

```cpp
    // Resource creation.
    WriteBufferState(state_table);
    WriteImageState(state_table);
    StandardCreateWrite<vulkan_wrappers::TensorARMWrapper>(state_table);
    WriteDeviceMemoryState(state_table);
    WriteTensorMemoryState(state_table);
    StandardCreateWrite<vulkan_wrappers::TensorViewARMWrapper>(state_table);

    // Bind memory after buffer/image creation and memory allocation. The buffer/image needs to be created before memory
    // allocation for extensions like dedicated allocation that require a valid buffer/image handle at memory allocation.
    WriteResourceMemoryState(state_table, true);
```

How replay recreates a buffer's contents from the snapshot (`framework/decode/vulkan_replay_consumer_base.cpp` L1171–1179, in `ProcessInitBufferCommand`):

```cpp
            else
            {
                VkBufferCopy copy_region;
                copy_region.srcOffset = 0;
                copy_region.dstOffset = 0;
                copy_region.size      = data_size;

                result = initializer->InitializeBuffer(
                    data_size, data, buffer_info->queue_family_index, buffer, buffer_info->usage, 1, &copy_region);
```

Replay option that repeats a frame (`USAGE_desktop_Vulkan.md` L691–696, `gfxrecon-replay` usage; L1138–1139 and L1141–1143, "Frame Looping"):

```text
  --loop-frame <N>      Enable frame repeat (experimental).
                        N specifies the frame number to repeat, where 1 is
                        the first frame.
  --loop-count <N>      Specify the total number of times to play the loop
                        frame. N must be greater than 1. Default is to
                        repeat the frame until replay is interrupted.
```

> Frame looping replays a single frame repeatedly instead of playing the capture through to the end.  It is intended for profiling a frame.

> Replay proceeds normally up to and including frame `N`, and then replays frame `N` again for each iteration.

How the loop handles state the frame changes (`framework/decode/vulkan_replay_frame_loop_consumer.h` L141–143, L182–183):

```cpp
        /// Semaphores waited on by any submit inside the loop range that are expected
        /// to be signaled by a submission enqueued before start looping.
        /// These semaphores will be synthetically signaled before repeating the loop range.
```

```cpp
    /// Tracks a device-local copy of each buffer's contents at the start of the loop range so that GPU
    /// writes made to the buffer during the loop can be undone before each repetition.
```

At each frame boundary before the final iteration (`framework/decode/vulkan_replay_frame_loop_consumer.cpp` L2215–2223):

```cpp
    if (!frame_loop_info_.IsLooping() || frame_loop_info_.IsFinalIteration())
    {
        return;
    }
    FixupDeviceEvents(device);
    FixupDeviceImages(device, queue);
    FixupDeviceFences(device, queue);
    FixupDeviceBuffers(device);
    GetSemaphoreTracking(device).FixupSemaphores(queue);
```

Creation and destruction calls in the looped frame (`framework/generated/generated_vulkan_replay_frame_loop_consumer_base.cpp` L48–49 and L93–95, the `vkCreateInstance`/`vkDestroyInstance` instance of a pattern generated for each object type):

```cpp
    // Pass the call along if we are not looping or
    // if we are looping and the handle is not in allocatedLoopResources
```

```cpp
        // Looping special case:
        // This resource has been allocated BEFORE the loop range.
        // Since it might still be in use during the loop range, ONLY free it in the last iteration.
```

### 2. apitrace

- **Source:** *apitrace*, https://github.com/apitrace/apitrace, branch `master`, commit `3a0ee82a1bcad6930090af643f73532204017650` (2026-11-06). License: MIT (`LICENSE`, "Copyright 2007-2012 VMware, Inc." and others).
- **Where read:** a depth-1 fetch, in `lit3/cr-apitrace/apitrace/`. Issue #800 was read with `gh` and saved as `lit3/cr-apitrace/issue800.txt`.

The `--loop` option (`retrace/retrace_main.cpp` L820, usage text; L1415, the warning printed when `--loop` is used):

```cpp
        "      --loop[=N]          loop N times (N<0 continuously) replaying final frame.\n"
```

```cpp
        std::cerr << "warning: --loop blindly repeats the last frame calls, therefore frames might not necessarily render correctly (https://github.com/apitrace/apitrace/issues/800)" << std::endl;
```

The loop re-issues the stored calls of the last frame (`lib/trace/trace_parser_loop.cpp` L98–105):

```cpp
    if (loopCount) {
        call = *lastFrameIterator;
        ++lastFrameIterator;
        if (lastFrameIterator == lastFrameCalls.end()) {
            /* repeat the frame */
            lastFrameIterator = lastFrameCalls.begin();
            --loopCount;
        }
```

apitrace issue #800, "repeated replay of nheko trace fails", https://github.com/apitrace/apitrace/issues/800. Comments of 2022-06-14 by gerddie (contributor) and jrfonseca (member):

> When looping the last frame it may happen, that a resource that is used in this last frame is then destroyed, and when the frame is played the second time and this now destroyed resources is supposed to be used, replay may crash.

> Looping is full of such holes. It's really one of those _"YMMV"_ features.

> Really, it's up to the user make sure loop is safe to use, as there's no expectation it's _safe_.

`apitrace trim` (`docs/USAGE.markdown` L386–388; `cli/cli_trim.cpp` L51–52):

```markdown
You can truncate a trace by doing:

    apitrace trim --calls 0-12345 -o trimed.trace application.trace
```

```cpp
        "        --calls=CALLSET      Include specified calls in the trimmed output.\n"
        "        --frames=FRAMESET    Include specified frames in the trimmed output.\n"
```

`apitrace gltrim` (`cli/cli_gltrim.cpp` L65; design note `frametrim/frametrim.markdown` L11–13):

```cpp
static const char *synopsis = "Trim an OpenGL trace to a replayable subset of frames";
```

> At the start of the target frame the current state is recorded to the output, and all the calls recorded in the objects used in the callaset corresponding to the frame.

### 3. RenderDoc

- **Source:** Baldur Karlsson et al., *RenderDoc documentation*: "How RenderDoc works" (`docs/behind_scenes/how_works.rst`), "Capture options" (`docs/window/capture_attach.rst`), "In-application API" (`docs/in_application_api.rst`). Code in `renderdoc/core/resource_manager.h`. Repository https://github.com/baldurk/renderdoc, branch `v1.x`, commit `34fbfb4352d65a1fb609afba356aac03af1786a3` (2026-11-04). License: MIT (`LICENSE.md`, "Copyright (c) 2015-2026 Baldur Karlsson").
- **Where read:** a depth-1 fetch, in `lit3/cr-renderdoc/renderdoc/`. Also https://renderdoc.org/docs/behind_scenes/how_works.html (built from `v1.47`). All five `how_works.rst` excerpts below also occur word for word on that page.

Capture (`how_works.rst` L17, L21, L23):

> In some cases the driver might choose to optimise out some of the data-upload calls and lazy initialise the contents of some resources to save on background overhead.

> When the capture button is hit the driver will enter active capturing upon the beginning of the next frame. In this state every API call is serialised out in order and any initial contents and states are saved.

> by default resources which are not referenced are not included in the capture.

Replay (`how_works.rst` L30, L38):

> When replaying, the initial section of the capture (up to the beginning of the frame) is read and executed verbatim.

> When replaying from the beginning of a frame (and not a partial subset of the frame) the initial states of all resources are applied, and the initial pipeline state is restored. Resources which did not have a serialised initial state (e.g. gbuffer textures) have an initial state saved before the first replay of the frame, and this is restored.

Where a capture starts (`capture_attach.rst` L134 and L159, `in_application_api.rst` L120):

> If the application creates a command list early and replays it indefinitely without recreating it, RenderDoc will essentially have missed its chance to capture it by the time you hit the capture key.

> Frame 0 is defined as starting at initialisation and ending at the first presentation.

> This function will immediately begin a capture for the specified device/window combination.

Code (`renderdoc/core/resource_manager.h` L650–651):

```cpp
  // Apply the initial contents for the resources that need them, used at the start of a frame
  void ApplyInitialContents();
```

### 4. NVIDIA CUDA Programming Guide, CUDA Graphs

- **Source (current):** NVIDIA, *CUDA Programming Guide*, v13.5 ("Last updated on Oct 31, 2026"). §4.2 "CUDA Graphs" is at https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html and §4.9 "Lazy Loading" at https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html. The old URL of the *CUDA C++ Programming Guide* now redirects to this renamed guide.
- **Source (last edition under the requested title):** NVIDIA, *CUDA C++ Programming Guide*, v13.0, §6.2.8.7 "CUDA Graphs", https://docs.nvidia.com/cuda/archive/13.0.2/cuda-c-programming-guide/index.html ("Last updated on Nov 02, 2025").
- **Where read:** curl, then HTML-to-text, in `lit3/cr-cuda/`.

Defined once, launched repeatedly (v13.5 §4.2):

> A graph is defined separately from its execution. This allows a graph to be defined once and then launched repeatedly.

> these overhead costs can be paid once for the entire graph during instantiation, and the graph itself can then be launched repeatedly with very little overhead.

The three stages (v13.5 §4.2.2 "Building and Running Graphs"):

```text
Work submission using graphs is separated into three distinct stages: definition, instantiation, and execution.
* During the definition or creation phase, a program creates a description of the operations in the graph along with the dependencies between them.
* Instantiation takes a snapshot of the graph template, validates it, and performs much of the setup and initialization of work with the aim of minimizing what needs to be done at launch. The resulting instance is known as an executable graph.
* An executable graph may be launched into a stream, similar to any other CUDA work. It may be launched any number of times without repeating the instantiation.
```

v13.5 §4.2.3 "Updating Instantiated Graphs":

> A graph may be updated and relaunched repeatedly, so multiple updates/launches can be queued on a stream.

Legacy v13.0 §6.2.8.7.5 "Updating Instantiated Graphs":

> In situations where the workflow is not changing, the overhead of definition and instantiation can be amortized over many executions, and graphs provide a clear advantage over streams.

> A graph is a snapshot of a workflow, including kernels, parameters, and dependencies, in order to replay it as rapidly and efficiently as possible.

Where a capture starts (v13.5 §4.2.2.1.2 "Stream Capture"; the closing Note of §4.2.2.1.2.1):

> A call to cudaStreamBeginCapture() places a stream in capture mode. When a stream is being captured, work launched into the stream is not enqueued for execution. It is instead appended to an internal graph that is progressively being built up.

> When a stream is taken out of capture mode, the next non-captured item in the stream (if any) will still have a dependency on the most recent prior non-captured item, despite intermediate items having been removed.

Prohibited or unsupported during capture (v13.5 §4.2.2.1.2.2 "Prohibited and Unhandled Operations"):

> It is invalid to synchronize or query the execution status of a stream which is being captured or a captured event, because they do not represent items scheduled for execution.

> It is therefore also invalid to call synchronous APIs in this case. One example of a synchronous APIs is cudaMemcpy() which enqueues work to the legacy stream and synchronizes on it before returning.

> As a general rule, when a dependency relation would connect something that is captured with something that was not captured and instead enqueued for execution, CUDA prefers to return an error rather than ignore the dependency.

> A small number of APIs that enqueue asynchronous operations into streams are not currently supported in graphs and will return an error if called with a stream which is being captured, such as cudaStreamAttachMemAsync().

v13.5 §4.2.2.1.2.3 "Invalidation":

> When an invalid operation is attempted during stream capture, any associated capture graphs are invalidated.

> This call will take the associated streams out of capture mode, but will also return an error value and a NULL graph.

v13.5 §4.2.8 "CUDA User Objects":

> They also do not work with stream capture if these considerations are hidden from the caller of the library, and because of use of disallowed APIs during capture.

Work to be done before capture. §4.2 has no statement in so many words about work that must be done *before* capture. Searching §4.2 for "before captur", "prior to captur", "warm" and "outside of captur" found nothing. The nearest statement is in v13.5 §4.9.5.3 "Impact on Performance Measurements" (Lazy Loading), and it is about measurement, not capture:

```text
Lazy loading may skew performance measurements by moving CUDA module initialization into the measured execution window. To avoid this:
* do at least one warmup iteration prior to measurement
* preload the benchmarked kernel prior to launching it
```

### 5. Dawn: the lazy clear

- **Source:** The Dawn & Tint Authors, *Dawn*, https://dawn.googlesource.com/dawn, read from the GitHub mirror https://github.com/google/dawn at commit `797946823414be58054116de563d9ec4636dcb2e` (2026-11-07). License: BSD 3-Clause (`LICENSE`, "Copyright 2017-2026 The Dawn & Tint Authors").
- **Where read:** `raw.githubusercontent.com/google/dawn/<sha>/<path>`, in `lit3/cr-dawn/`.
- **Names at this commit:** there is no function `IsDataInitialized`. The flag is `mIsDataInitialized` (`src/dawn/native/Buffer.h` L287). It is tested by `NeedsInitialization()`, set by `SetInitialized()`, and read by `IsResourceInitialized()`.

The toggle (`src/dawn/native/Toggles.cpp` L71–74):

```cpp
    {Toggle::LazyClearResourceOnFirstUse,
     {"lazy_clear_resource_on_first_use",
      "Clears resource to zero on first usage. This initializes the resource so that no dirty bits "
      "from recycled memory is present in the new resource.",
```

The flag (`src/dawn/native/Buffer.cpp` L1171–1173 and L1230–1232):

```cpp
bool BufferBase::NeedsInitialization() const {
    return !mIsDataInitialized && GetDevice()->IsToggleEnabled(Toggle::LazyClearResourceOnFirstUse);
}
```

```cpp
void BufferBase::SetInitialized(bool initialized) {
    mIsDataInitialized = initialized;
}
```

`Buffer::EnsureDataInitialized`, `Buffer::InitializeToZero`, and the fill in `Buffer::ClearBuffer` (`src/dawn/native/metal/BufferMTL.mm` L247–254, L287–294, L304–306):

```objc
bool Buffer::EnsureDataInitialized(CommandRecordingContext* commandContext) {
    if (!NeedsInitialization()) {
        return false;
    }

    InitializeToZero(commandContext);
    return true;
}
```

```objc
void Buffer::InitializeToZero(CommandRecordingContext* commandContext) {
    DAWN_ASSERT(NeedsInitialization());

    ClearBuffer(commandContext, uint8_t{0});

    SetInitialized(true);
    GetDevice()->IncrementLazyClearCountForTesting();
}
```

```objc
    [commandContext->EnsureBlit() fillBuffer:mMtlBuffer.Get()
                                       range:NSMakeRange(offset, size)
                                       value:clearValue];
```

A write that covers the whole buffer marks it initialized with no clear (`BufferMTL.mm` L263–266, in `EnsureDataInitializedAsDestination`):

```objc
    if (IsFullBufferRange(offset, size)) {
        SetInitialized(true);
        return false;
    }
```

Where the check runs: for every buffer a compute dispatch uses (`src/dawn/native/metal/CommandBufferMTL.mm` L562–564, in `LazyClearSyncScope`, called per dispatch scope at L661–662):

```objc
    for (BufferBase* bufferBase : scope.buffers) {
        ToBackend(bufferBase)->EnsureDataInitialized(commandContext);
    }
```

```objc
    for (const SyncScopeResourceUsage& scope : resourceUsage.dispatchUsages) {
        DAWN_TRY(LazyClearSyncScope(scope, commandContext));
```

### 6. JAX: Python side effects under `jit`

- **Source:** The JAX Authors, "🔪 JAX - The Sharp Bits 🔪" (`docs/notebooks/Common_Gotchas_in_JAX.md`, published at https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) and "Just-in-time compilation" (`docs/jit-compilation.md`, published at https://docs.jax.dev/en/latest/jit-compilation.html). Repository https://github.com/jax-ml/jax, commit `6b35b69f004529125fc89acc89aa73b7133161c0` (2026-11-07). License: Apache-2.0.
- **Where read:** `raw.githubusercontent.com/jax-ml/jax/<sha>/<path>`, in `lit3/cr-jax/`.

"The Sharp Bits", "🔪 Pure functions" (`Common_Gotchas_in_JAX.md` L47 and L59–64):

> JAX transformation and compilation are designed to work only on Python functions that are functionally pure: all the input data is passed through the function parameters, all the results are output through the function results.

```python
# The side-effects appear during the first run
print ("First call: ", jit(impure_print_side_effect)(4.))

# Subsequent runs with parameters of same type and shape may not show the side-effect
# This is because JAX now invokes a cached compilation of the function
print ("Second call: ", jit(impure_print_side_effect)(5.))
```

"Just-in-time compilation", "How JAX transformations work" (`jit-compilation.md` L67, L69) and the first call (L135):

> Since the tracers do not record the Python side-effects, they do not appear in the jaxpr. However, the side-effects still happen during the trace itself.

> Therefore, any `print()` calls will only happen during tracing, and will not appear in the jaxpr

> Subsequent calls to `selu_jit` will use the compiled code directly, skipping the python implementation entirely.

---

## Difference 2: calls a capture tool doesn't support

### 7. GFXReconstruct and apitrace

**GFXReconstruct** (same commit and license as source 1).

The principle (`docs/GFXReconstruct Architectural Principles.md` L89):

> If a call truly cannot be represented in its original form wholly or piecewise, that case must be documented as an exception.

At capture, unsupported device extensions are hidden from the application (`framework/encode/vulkan_entry_base.h` L109–111; `layer/layer_vulkan_entry.cpp` L163–165):

```cpp
    /// An alphabetical list of device extensions which we do not report upstream if
    /// other layers or ICDs expose them to us.
    static const std::vector<const char*> kVulkanUnsupportedDeviceExtensions;
```

```cpp
        // If this function was not called with the layer's name, we expect to dispatch down the chain to obtain the
        // ICD provided extensions. In order to screen out unsupported extensions, we always query the chain twice,
        // and remove those that are present from the count.
```

At capture, `vkGetDeviceProcAddr` returns `nullptr` for a name not in the layer's table (`framework/encode/vulkan_entry_base.cpp` L201–209; `result` is initialized to `nullptr` at L174):

```cpp
        // Only intercept the requested function if there is an implementation available
        if (has_implementation)
        {
            const auto entry = vulkan_function_table_.find(pName);
            if (entry != vulkan_function_table_.end())
            {
                result = entry->second;
            }
        }
```

At capture, a known call whose encoding is unimplemented is logged as an error and passed to the driver without being recorded (`framework/encode/custom_vulkan_api_call_encoders.cpp` L468–471):

```cpp
    // TODO
    GFXRECON_LOG_ERROR("BuildAccelerationStructuresKHR encoding is not supported");
    return vulkan_wrappers::GetDeviceTable(device)->BuildAccelerationStructuresKHR(
        device, deferredOperation, infoCount, pInfos, ppRangeInfos);
```

At capture, an application using an API version newer than the layer supports (`framework/encode/vulkan_capture_manager.cpp` L655–658):

```cpp
            GFXRECON_LOG_WARNING(
                "The application has specified that it uses Vulkan API version %u.%u.%u, which is newer than the "
                "version supported by GFXReconstruct.  Use of unsupported Vulkan features may cause capture or replay "
                "to fail.",
```

At replay (`USAGE_desktop_Vulkan.md` L771–772; `framework/graphics/vulkan_feature_util.cpp` L226; `framework/decode/block_parser.cpp` L163):

```text
  --remove-unsupported  Remove unsupported extensions and features from instance
                        and device creation parameters.
```

```cpp
            GFXRECON_LOG_WARNING("Extension %s, which is not supported by the replay device, will not be enabled",
```

```cpp
        message << "Skipping unrecognized " << compressed << " " << sub_type_label << " with type " << sub_type;
```

**apitrace** (same commit and license as source 2).

`docs/USAGE.markdown` L11–12:

> Problems while tracing (e.g, if the application uses calls/parameters unsupported by apitrace) will be reported via stderr output on Unices.

At trace time (OpenGL), the tracer logs a warning for a `GetProcAddress` name it doesn't know and returns the real, untraced pointer (`wrappers/gltrace.py` L345–346, generator of `_wrapProcAddress`):

```python
            print('    os::log("apitrace: warning: unknown function \\"%s\\"\\n", (const char *)procName);')
            print('    return procPtr;')
```

At trace time (COM/D3D), the tracer warns about an interface it doesn't wrap and leaves the object unwrapped (`wrappers/trace.py` L964):

```python
        print(r'    warnIID(entryName, riid, *ppvObj, "unsupported");')
```

At replay, a call with no retrace callback goes to `unsupported`, which warns and does nothing else (`retrace/retrace.cpp` L107–109 and L139–142):

```cpp
void unsupported(trace::Call &call) {
    warning(call) << "unsupported " << call.name() << " call\n";
}
```

```cpp
        Map::const_iterator it = map.find(call.name());
        if (it == map.end()) {
            callback = &unsupported;
        } else {
```

---

## Difference 3: creation parameters changed at capture

### 8. GFXReconstruct: the capture layer modifies the application's create infos (same commit and license as source 1)

The principle (`docs/GFXReconstruct Architectural Principles.md` L102, L106, L110–111):

> Capture should avoid changing the content of the application’s graphics commands.  Some exceptions are necessary in order to save information required for replay or modify data so that replay is possible.

> a corollary is that API calls modified in capture should be stored in their original form in the capture file; a MetaData Command may encode additional information to signal to the replay application to also modify the calls on replay.

```markdown
  * Capture enables the Vulkan device features bufferDeviceAddressCaptureReplay, rayTracingPipelineShaderGroupHandleCaptureReplay, and accelerationStructureCaptureReplay during capture of Vulkan applications (and in replay) to allow replay on the same device and driver of ray-tracing workloads.  
    * Capture may alter function calls (e.g. add USAGE flags to memory creation) that enable the use of those device features  
```

Device creation enables the feature when the device supports it (`framework/graphics/vulkan_device_util.cpp` L131–135):

```cpp
        // Enable bufferDeviceAddressCaptureReplay if it is supported
        if (supported_features.bufferDeviceAddressCaptureReplay)
        {
            feature_struct->bufferDeviceAddressCaptureReplay = VK_TRUE;
        }
```

Buffer creation (`framework/encode/vulkan_capture_manager.cpp` L877–879 and L889–896, in `VulkanCaptureManager::OverrideCreateBuffer`):

```cpp
    if (IsTrimEnabled())
    {
        modified_create_info->usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
```

```cpp
        // If the buffer has shader device address usage, but the device address capture replay flag was not set, it
        // needs to be set here.  We modify only a copy to prevent the modified pCreateInfo from being
        // written to the capture file.
        if ((pCreateInfo->usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) ==
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
        {
            uses_address = true;
            modified_create_info->flags |= VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT;
```

The address the replay will request is written as a meta-command ahead of the create call (`vulkan_capture_manager.cpp` L936–938):

```cpp
            // If the buffer has a device address, write the 'set buffer address' command before writing the API call to
            // create the buffer.  The address will need to be passed to vkCreateBuffer through the pCreateInfo pNext
            // list.
```

Image creation when trimming (`vulkan_capture_manager.cpp` L1040–1043):

```cpp
    if (IsTrimEnabled())
    {
        modified_create_info.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }
```

Pipeline creation. A creation flag is added to each ray-tracing pipeline's create info (`vulkan_capture_manager.cpp` L1677–1684):

```cpp
    if (device_wrapper->property_feature_info.feature_rayTracingPipelineShaderGroupHandleCaptureReplay)
    {
        auto modified_create_infos = std::make_unique<VkRayTracingPipelineCreateInfoKHR[]>(createInfoCount);
        for (uint32_t i = 0; i < createInfoCount; ++i)
        {
            modified_create_infos[i] = pCreateInfos_unwrapped[i];
            modified_create_infos[i].flags |= VK_PIPELINE_CREATE_RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_BIT_KHR;
        }
```

When the capture device lacks the feature (`vulkan_capture_manager.cpp` L3385–3388):

```cpp
        GFXRECON_LOG_ERROR_ONCE(
            "The application is using vkGetBufferDeviceAddress, which requires the bufferDeviceAddressCaptureReplay "
            "feature for accurate capture and replay. The capture device does not support this feature, so replay of "
            "the captured file may fail.");
```

### 9. The Vulkan specification

- **Source:** The Khronos Group, *Vulkan® 1.4 Specification*. Source at https://github.com/KhronosGroup/Vulkan-Docs, commit `fa30d4e312e843193601be7cb926c7e25d9ec885` ("Change log for November 6, 2026 Vulkan 1.4.368 spec update"). License of the specification source: CC-BY-4.0 (`COPYING.adoc`).
- **Where read:** `raw.githubusercontent.com/KhronosGroup/Vulkan-Docs/<sha>/chapters/*.adoc`, in `lit3/cr-vulkan/`. The quotes are AsciiDoc source, with the spec's markup macros left in.

`VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT` (`chapters/resources.adoc` L977–980, VkBufferCreateFlagBits):

```asciidoc
  * ename:VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT specifies that
    the buffer's address can: be saved and reused on a subsequent run (e.g.
    for trace capture and replay), see
    slink:VkBufferOpaqueCaptureAddressCreateInfo for more detail.
```

`bufferDeviceAddressCaptureReplay` (`chapters/features.adoc` L2888–2891):

```asciidoc
  * [[{anchor-prefix}features-bufferDeviceAddressCaptureReplay]]
    pname:bufferDeviceAddressCaptureReplay indicates that the implementation
    supports saving and reusing buffer and device addresses, e.g. for trace
    capture and replay.
```

The NOTE on VkBufferOpaqueCaptureAddressCreateInfo (`chapters/resources.adoc` L1098–1103):

```asciidoc
The expected usage for this is that a trace capture/replay tool will add the
ename:VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT flag to all buffers
that use ename:VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, and during capture
will save the queried opaque device addresses in the trace.
During replay, the buffers will be created specifying the original address
so any address values stored in the trace data will remain valid.
```

A tool adding flags the application did not request (`chapters/resources.adoc` L8465–8468, acceleration-structure NOTE) and adjusting allocation parameters (`chapters/memory.adoc` L4959–4960):

```asciidoc
This also means that the tool will need to add
ename:VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT to memory allocations to allow
the ename:VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT flag to be set where the
application may not have otherwise required it.
```

```asciidoc
This is, however, not a strict requirement because trace capture/replay
tools may need to adjust memory allocation parameters for imported memory.
```

The pipeline creation flag with the same purpose (`chapters/pipelines.adoc` L371–373):

```asciidoc
  * ename:VK_PIPELINE_CREATE_2_RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_BIT_KHR
    specifies that the shader group handles can: be saved and reused on a
    subsequent run (e.g. for trace capture and replay).
```

---

## Metal (local SDK)

- **Source:** Apple, Metal.framework headers, macOS SDK 26.4.1 (build 25E251). `xcrun --sdk macosx --show-sdk-path` gives `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`, which links to `MacOSX26.4.sdk`. The headers are in `System/Library/Frameworks/Metal.framework/Headers/`. In the tables below, the doc text is the first sentence of the method's doc comment, with the comment markers removed and its wrapped lines joined.

### 10. `MTL4ComputeCommandEncoder.h`

The protocol (L27–29) and what counts as a blit command (L39–40, in the doc of `stages`):

> Use instances of this abstraction to encode a compute pass into ``MTL4CommandBuffer`` instances, as well as commands that copy and modify the underlying memory of various Metal resources, and commands that build or refit acceleration structures.

> Similarly, it sets the bit ``MTLStages/MTLStageBlit`` if this encoder contains any commands to copy or modify buffers, textures, or indirect command buffers.

Every copy, fill and other memory-modifying (blit-stage) method:

| Decl. L | Selector | Doc L | One-line description |
|---|---|---|---|
| 148–149 | `copyFromTexture:toTexture:` | 142–142 | Encodes a command that copies data from a texture to another. |
| 178–185 | `copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:destinationLevel:sliceCount:levelCount:` | 151–151 | Encodes a command that copies slices of a texture to slices of another texture. |
| 218–226 | `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:` | 187–187 | Encodes a command that copies image data from a slice of a texture into a slice of another texture. |
| 261–269 | `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:` | 228–228 | Encodes a command that copies image data from a slice of an ``MTLTexture`` instance to an ``MTLBuffer`` instance. |
| 312–321 | `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:` | 271–272 | Encodes a command that copies image data from a slice of a texture instance to a buffer, with options for special texture formats. |
| 331–335 | `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:` | 323–323 | Encodes a command that copies data from a buffer instance into another. |
| 369–377 | `copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:` | 337–337 | Encodes a command to copy image data from a buffer instance into a texture. |
| 418–427 | `copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:` | 379–379 | Encodes a command to copy image data from a buffer into a texture with options for special texture formats. |
| 439–444 | `copyFromTensor:sourceOrigin:sourceDimensions:toTensor:destinationOrigin:destinationDimensions:` | 429–429 | Encodes a command to copy data from a tensor instance into another. |
| 454–454 | `generateMipmapsForTexture:` | 446–447 | Encodes a command that generates mipmaps for a texture instance from the base mipmap level up to the highest mipmap level. |
| 463–465 | `fillBuffer:range:value:` | 456–456 | Encodes a command that fills a buffer with a constant value for each byte. |
| 479–479 | `optimizeContentsForGPUAccess:` | 467–468 | Encodes a command that modifies the contents of a texture to improve the performance of GPU accesses to its contents. |
| 496–498 | `optimizeContentsForGPUAccess:slice:level:` | 481–482 | Encodes a command that modifies the contents of a texture instance to improve the performance of GPU accesses to its contents in a specific region. |
| 512–512 | `optimizeContentsForCPUAccess:` | 500–501 | Encodes a command that modifies the contents of a texture to improve the performance of CPU accesses to its contents. |
| 529–531 | `optimizeContentsForCPUAccess:slice:level:` | 514–515 | Encodes a command that modifies the contents of a texture to improve the performance of CPU accesses to its contents in a specific region. |
| 538–539 | `resetCommandsInBuffer:withRange:` | 533–533 | Encodes a command that resets a range of commands in an indirect command buffer. |
| 551–554 | `copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:` | 541–541 | Encodes a command that copies commands from an indirect command buffer into another. |
| 561–562 | `optimizeIndirectCommandBuffer:withRange:` | 556–556 | Encode a command to attempt to improve the performance of a range of commands within an indirect command buffer. |

The protocol also declares other methods, not tabulated here: dispatch, indirect-execution and state methods (L46–140, L572); acceleration-structure build/refit/copy/compact commands (L597–726); and `writeTimestampWithGranularity:intoHeap:atIndex:` (L743–745).

### 11. `MTL4CommandQueue.h` events; `MTL4CommandBuffer.h`

`MTL4CommandQueue.h`:

| Decl. L | Selector | Doc L | One-line description |
|---|---|---|---|
| 266–267 | `signalEvent:value:` | 261–261 | Schedules an operation to signal a GPU event with a specific value after all GPU work prior to this point is complete. |
| 274–275 | `waitForEvent:value:` | 269–269 | Schedules an operation to wait for a GPU event of a specific value before continuing to execute any future GPU work. |
| 294–294 | `signalDrawable:` | 277–277 | Schedules a signal operation on the command queue to indicate when rendering to a Metal drawable is complete. |
| 308–308 | `waitForDrawable:` | 296–296 | Schedules a wait operation on the command queue to ensure the display is no longer using a specific Metal drawable. |

`MTL4CommandBuffer.h` declares no method that signals or waits on an event. A case-insensitive search for "event" finds one passage, L195–196, in the doc of `resolveCounterHeap:withRange:intoBuffer:waitFence:updateFence:` (L206–211), which takes fences:

> If your app needs to access `bufferRange` from the CPU, signal an ``MTLSharedEvent`` to notify the CPU when it's ready.

### 12. `supportIndirectCommandBuffers`; compute-pipeline creation methods

`MTLComputePipeline.h` L70–73 (`MTLComputePipelineDescriptor`) and L254–257 (`MTLComputePipelineState`):

```objc
 @property supportIndirectCommandBuffers
 @abstract This flag makes this pipeline usable with indirect command buffers.
 */
@property (readwrite, nonatomic) BOOL supportIndirectCommandBuffers API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(13.0));
```

```objc
 @property supportIndirectCommandBuffers
 @abstract Tells whether this pipeline state is usable through an Indirect Command Buffer.
 */
@property (readonly) BOOL supportIndirectCommandBuffers API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(13.0));
```

The Metal 4 descriptor, which `MTL4Compiler` takes (`MTL4ComputePipeline.h` L60–61; `MTL4PipelineState.h` L72–76):

```objc
/// A value indicating whether the pipeline supports Metal indirect command buffers.
@property (readwrite, nonatomic) MTL4IndirectCommandBufferSupportState supportIndirectCommandBuffers;
```

```objc
    /// Disables support for indirect command buffers.
    MTL4IndirectCommandBufferSupportStateDisabled = 0,
    
    /// Enables support for indirect command buffers.
    MTL4IndirectCommandBufferSupportStateEnabled  = 1,
```

`MTLDevice.h`: every compute-pipeline creation method. There are six, and they are the only `newComputePipelineState…` declarations in the file. The doc column is the header's `/*! @method … @abstract … */` text. Its `@method` tags for the first four name `…WithDescriptor:…` selectors, while the declarations are `…WithFunction:…`.

| Decl. L | Declaration | Doc L | Header doc |
|---|---|---|---|
| 834–834 | `- (nullable id <MTLComputePipelineState>)newComputePipelineStateWithFunction:(id <MTLFunction>)computeFunction error:(__autoreleasing NSError **)error;` | 831–832 | @method newComputePipelineStateWithDescriptor:error: @abstract Create and compile a new MTLComputePipelineState object synchronously. |
| 840–840 | `- (nullable id <MTLComputePipelineState>)newComputePipelineStateWithFunction:(id <MTLFunction>)computeFunction options:(MTLPipelineOption)options reflection:(MTLAutoreleasedComputePipelineReflection * __nullable)reflection error:(__autoreleasing NSError **)error;` | 837–838 | @method newComputePipelineStateWithDescriptor:options:reflection:error: @abstract Create and compile a new MTLComputePipelineState object synchronously. |
| 846–846 | `- (void)newComputePipelineStateWithFunction:(id <MTLFunction>)computeFunction completionHandler:(MTLNewComputePipelineStateCompletionHandler)completionHandler;` | 843–844 | @method newComputePipelineStateWithDescriptor:completionHandler: @abstract Create and compile a new MTLComputePipelineState object asynchronously. |
| 852–852 | `- (void)newComputePipelineStateWithFunction:(id <MTLFunction>)computeFunction options:(MTLPipelineOption)options completionHandler:(MTLNewComputePipelineStateWithReflectionCompletionHandler)completionHandler;` | 849–850 | @method newComputePipelineStateWithDescriptor:options:completionHandler: @abstract Create and compile a new MTLComputePipelineState object asynchronously. |
| 858–858 | `- (nullable id <MTLComputePipelineState>)newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor options:(MTLPipelineOption)options reflection:(MTLAutoreleasedComputePipelineReflection * __nullable)reflection error:(__autoreleasing NSError **)error API_AVAILABLE(macos(10.11), ios(9.0));` | 855–856 | @method newComputePipelineStateWithDescriptor:options:reflection:error: @abstract Create and compile a new MTLComputePipelineState object synchronously. |
| 864–864 | `- (void)newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor options:(MTLPipelineOption)options completionHandler:(MTLNewComputePipelineStateWithReflectionCompletionHandler)completionHandler API_AVAILABLE(macos(10.11), ios(9.0));` | 861–862 | @method newComputePipelineStateWithDescriptor:options:completionHandler: @abstract Create and compile a new MTLComputePipelineState object asynchronously. |

`MTL4Compiler.h`: compute-pipeline creation methods:

| Decl. L | Declaration | Doc L | One-line description |
|---|---|---|---|
| 137–139 | `- (nullable id<MTLComputePipelineState>)newComputePipelineStateWithDescriptor:(MTL4ComputePipelineDescriptor*)descriptor compilerTaskOptions:(nullable MTL4CompilerTaskOptions*)compilerTaskOptions error:(NSError**)error;` | 128–128 | Creates a new compute pipeline state object synchronously. |
| 151–154 | `- (nullable id<MTLComputePipelineState>)newComputePipelineStateWithDescriptor:(MTL4ComputePipelineDescriptor *)descriptor dynamicLinkingDescriptor:(nullable MTL4PipelineStageDynamicLinkingDescriptor*)dynamicLinkingDescriptor compilerTaskOptions:(nullable MTL4CompilerTaskOptions *)compilerTaskOptions error:(NSError **)error;` | 141–141 | Creates a new compute pipeline state synchronously. |
| 270–272 | `- (id<MTL4CompilerTask>)newComputePipelineStateWithDescriptor:(MTL4ComputePipelineDescriptor*)descriptor compilerTaskOptions:(nullable MTL4CompilerTaskOptions*)compilerTaskOptions completionHandler:(MTLNewComputePipelineStateCompletionHandler)completionHandler;` | 261–261 | Creates a new compute pipeline state asynchronously. |
| 284–287 | `- (id<MTL4CompilerTask>)newComputePipelineStateWithDescriptor:(MTL4ComputePipelineDescriptor *)descriptor dynamicLinkingDescriptor:(nullable MTL4PipelineStageDynamicLinkingDescriptor*)dynamicLinkingDescriptor compilerTaskOptions:(nullable MTL4CompilerTaskOptions *)compilerTaskOptions completionHandler:(MTLNewComputePipelineStateCompletionHandler)completionHandler;` | 274–274 | Creates a new compute pipeline state asynchronously. |

Beyond the request: the other `newComputePipelineState…` declarations that a search across all the Metal headers finds.

| File | Decl. L | Selector | Doc L | One-line description |
|---|---|---|---|---|
| `MTL4Archive.h` | 62–63 | `newComputePipelineStateWithDescriptor:error:` | 55–55 | Creates a compute pipeline state from the archive with a descriptor. |
| `MTL4Archive.h` | 74–76 | `newComputePipelineStateWithDescriptor:dynamicLinkingDescriptor:error:` | 65–65 | Creates a compute pipeline state from the archive with a compute descriptor and a dynamic linking descriptor. |
| `MTLComputePipeline.h` | 217–218 | `newComputePipelineStateWithBinaryFunctions:error:` | 210–210 | Allocates a new compute pipeline state by adding binary functions to this pipeline state. |
| `MTLComputePipeline.h` | 277–277 | `newComputePipelineStateWithAdditionalBinaryFunctions:error:` | 275–275 | @brief Allocate a new compute pipeline state by adding binary functions to this pipeline state. |

The topic list of `MTL4Archive.h` (L35–36) also names `newComputePipelineStateWithName:dynamicLinkingDescriptor:error:` and `newComputePipelineStateWithName:error:`. The header declares neither.

---

## Read / not read

All twelve items were read. Nothing was marked "not read".

- Item 1: there is no separate GFXReconstruct trimming document at the commit.
- Item 4: the URL of the *CUDA C++ Programming Guide* now serves the renamed *CUDA Programming Guide* v13.5. Both that and the last edition under the old title, v13.0, were read.
- Item 5: Dawn was read from its GitHub mirror. `IsDataInitialized` does not exist at that commit.
