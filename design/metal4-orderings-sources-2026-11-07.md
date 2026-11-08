# Metal 4 orderings, events, indirect execution, and matmul2d over k: sources, quoted verbatim

Retrieved 2026-11-07. Each excerpt below was copied by script from the cited lines of the file named; nothing in a
block is retyped.

- **Source:** Apple, Metal.framework headers, macOS SDK 26.4 (`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`,
  a link to `MacOSX26.4.sdk`), `System/Library/Frameworks/Metal.framework/Headers/` and
  `System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers/`.

## 1. Barriers and fences in a command encoder

`MTL4CommandEncoder.h` L48–72:

```objc
/// Encodes a consumer barrier on work you commit to the same command queue.
///
/// Encode a barrier that guarantees that any subsequent work you encode in the current command encoder that corresponds
/// to the `beforeStages` stages doesn't proceed until Metal completes all work prior to the current command encoder
/// corresponding to the `afterQueueStages` stages, completes.
///
/// Metal can reorder the exact point where it applies the barrier, so encode the barrier as close to the command that
/// consumes the resource as possible. Don't use this method for synchronizing resource access within the same pass.
///
/// If you need to synchronize work within a pass that you encode with an instance of a subclass of ``MTLCommandEncoder``,
/// use memory barriers instead. For subclasses of ``MTL4CommandEncoder``, use encoder barriers.
///
/// You can specify `afterQueueStages` and `beforeStages` that contain ``MTLStages`` unrelated to the current command
/// encoder.
///
/// - Parameters:
///   - afterQueueStages: ``MTLStages`` mask that represents the stages of work to wait for.
///                        This argument applies to work corresponding to these stages you
///                        encode in prior command encoders, and not for the current encoder.
///   - beforeStages:     ``MTLStages`` mask that represents the stages of work that wait.
///                        This argument applies to work you encode in the current command encoder.
///   - visibilityOptions: ``MTL4VisibilityOptions`` of the barrier.
- (void)barrierAfterQueueStages:(MTLStages)afterQueueStages
                   beforeStages:(MTLStages)beforeStages
              visibilityOptions:(MTL4VisibilityOptions)visibilityOptions;
```

`MTL4CommandEncoder.h` L74–93:

```objc
/// Encodes a producer barrier on work committed to the same command queue.
///
/// This method encodes a barrier that guarantees that any work you encode using *subsequent command encoders*,
/// corresponding to `beforeQueueStages`, don't begin until all commands you previously encode in the current
/// encoder (and prior encoders), corresponding to `afterStages`, complete.
///
/// When calling this method, you can pass any ``MTLStages`` to parameters `afterStages` and `beforeQueueStages`,
/// even stages that don't relate to the current or prior command encoders.
///
/// - Parameters:
///   - afterStages:       ``MTLStages`` mask that represents the stages of work to wait for.
///                         This argument applies to work corresponding to these stages you encode in
///                         the current command encoder prior to this barrier command.
///   - beforeQueueStages: ``MTLStages`` mask that represents the stages of work that need to wait.
///                         This argument applies to subsequent encoders and not to work in the current
///                         command encoder.
///   - visibilityOptions: ``MTL4VisibilityOptions`` of the barrier, controlling cache flush behavior.
- (void)barrierAfterStages:(MTLStages)afterStages
         beforeQueueStages:(MTLStages)beforeQueueStages
         visibilityOptions:(MTL4VisibilityOptions)visibilityOptions;
```

`MTL4CommandEncoder.h` L95–115:

```objc
/// Encodes an intra-pass barrier.
///
/// Encode a barrier that guarantees that any subsequent work you encode in the *current command encoder*,
/// corresponding to `beforeEncoderStages`, doesn't begin until all prior commands in this command encoder,
/// corresponding to `afterEncoderStages`, completes.
///
/// When calling this method, it's your responsibility to ensure parameters `afterEncoderStages` and `beforeEncoderStages`
/// contain a combination of ``MTLStages`` for which this encoder can encode commands. For example, for a
/// ``MTL4ComputeCommandEncoder`` instance, you can provide any combination of ``MTLStages/MTLStageDispatch``,
/// ``MTLStages/MTLStageBlit`` and ``MTLStages/MTLStageAccelerationStructure``.
///
/// - Parameters:
///   - afterEncoderStages:  ``MTLStages`` mask that represents the stages of work to wait for.
///                          This argument only applies to subsequent work you encode in the current command encoder.
///   - beforeEncoderStages: ``MTLStages`` mask that represents the stages of work that wait.
///                          This argument only applies to work you encode in the current command encoder prior to
///                          this barrier.
///   - visibilityOptions: ``MTL4VisibilityOptions`` of the barrier, controlling cache flush behavior.
- (void)barrierAfterEncoderStages:(MTLStages)afterEncoderStages
              beforeEncoderStages:(MTLStages)beforeEncoderStages
                visibilityOptions:(MTL4VisibilityOptions)visibilityOptions;
```

`MTL4CommandEncoder.h` L117–131:

```objc
/// Encodes a command to update a GPU fence.
///
/// This method encodes a command that updates a ``MTLFence`` instance after all previously-encoded commands in the
/// current command encoder, corresponding to `afterEncoderStages`, complete.
///
/// Use parameter `afterEncoderStages` to pass in a combination of ``MTLStages`` for which this encoder can encode work.
/// For example, for a ``MTL4ComputeCommandEncoder`` you can provide any combination of ``MTLStages/MTLStageDispatch``,
/// ``MTLStages/MTLStageBlit`` and ``MTLStages/MTLStageAccelerationStructure``.
///
/// - Parameters:
///   - fence:              ``MTLFence`` instance to update.
///   - afterEncoderStages: ``MTLStages`` value that represents the stages of work to wait for.
///                          This argument only applies to work encoded in the current command encoder.
- (void)updateFence:(id<MTLFence>)fence
 afterEncoderStages:(MTLStages)afterEncoderStages;
```

`MTL4CommandEncoder.h` L133–149:

```objc
/// Encodes a command to wait on a GPU fence.
///
/// Encode a command that guarantees that any subsequent work you encode via this current command encoder,
/// corresponding to `beforeEncoderStages`, doesn't begin until all prior updates to the fence is complete.
///
/// To successfully wait for a fence update, schedule update and wait operations on the same command queue.
///
/// Use parameter `beforeEncoderStages` to pass in a combination of ``MTLStages`` for which this encoder can encode
/// work. For example, for a ``MTL4ComputeCommandEncoder`` you can provide any combination of
/// ``MTLStages/MTLStageDispatch``, ``MTLStages/MTLStageBlit`` and ``MTLStages/MTLStageAccelerationStructure``.
///
/// - Parameters:
///   - fence:              ``MTLFence`` instance to wait for.
///   - beforeEncoderStages:``MTLStages`` value that represents the stages of work that wait.
///                         This argument only applies to work you encode in the current command encoder.
- (void)waitForFence:(id<MTLFence>)fence
 beforeEncoderStages:(MTLStages)beforeEncoderStages;
```

## 2. Commit and events on a command queue

`MTL4CommandQueue.h` L220–235:

```objc
/// Enqueues an array of command buffers for execution.
///
/// The order in which you sort the command buffers in the array is meaningful, especially when it contains suspending/resuming
/// render passes. A suspending/resuming render pass is a render pass you create by calling
/// ``MTL4CommandBuffer/renderCommandEncoderWithDescriptor:options:``,
/// and provide `MTL4RenderEncoderOptionSuspending` or `MTL4RenderEncoderOptionResuming` for the `options` parameter.
///
/// If your command buffers contain suspend/resume render passes, ensure that the first command buffer only suspends,
/// and the last one only resumes. Additionally, make sure that all intermediate command buffers are both suspending
/// and resuming.
///
/// - Parameters:
///   - commandBuffers: an array of ``MTL4CommandBuffer``.
///   - count: the number of ``MTL4CommandBuffer`` instances in the `commandBuffers` array.
- (void)commit:(const id<MTL4CommandBuffer> _Nonnull[_Nonnull])commandBuffers
         count:(NSUInteger)count;
```

`MTL4CommandQueue.h` L261–275:

```objc
/// Schedules an operation to signal a GPU event with a specific value after all GPU work prior to this point is complete.
///
/// - Parameters:
///   - event: ``MTLEvent`` to signal.
///   - value: the value to signal the ``MTLEvent`` with.
- (void)signalEvent:(id<MTLEvent>)event
              value:(uint64_t)value;

/// Schedules an operation to wait for a GPU event of a specific value before continuing to execute any future GPU work.
///
/// - Parameters:
///   - event: ``MTLEvent`` to wait on.
///   - value: the specific value to wait for.
- (void)waitForEvent:(id<MTLEvent>)event
               value:(uint64_t)value;
```

`MTLEvent.h` L17–21:

```objc
/*!
 @property device
 @abstract The device this event can be used with. Will be nil when the event is shared across devices (i.e. MTLSharedEvent).
 */
@property (nullable, readonly) id <MTLDevice> device;
```

`MTLEvent.h` L57–64:

```objc
// Convenience method for creating a shared event handle that may be passed to other processes via XPC.
- (MTLSharedEventHandle *)newSharedEventHandle;

// Synchronously wait for the signaledValue to be greater than or equal to 'value', with a timeout
// specified in milliseconds.   Returns YES if the value was signaled before the timeout, otherwise NO.
- (BOOL)waitUntilSignaledValue:(uint64_t)value timeoutMS:(uint64_t)milliseconds API_AVAILABLE(macos(12.0), ios(15.0)) NS_SWIFT_UNAVAILABLE_FROM_ASYNC("Use 'await valueSignaled(...)' instead.");

@property (readwrite) uint64_t signaledValue; // Read or set signaled value
```

## 3. Indirect dispatch and indirect command buffers

`MTL4ComputeCommandEncoder.h` L90–140:

```objc
/// Encodes a compute dispatch command with a grid that aligns to threadgroup boundaries, using an indirect buffer
/// for arguments.
///
/// This method allows you to supply the threadgroups-per-grid counts indirectly via an ``MTLBuffer`` index. This
/// enables you to calculate this value in the GPU timeline from a shader function, enabling GPU-driven workflows.
///
/// Metal assumes that the buffer contents correspond to the layout of struct ``MTLDispatchThreadgroupsIndirectArguments``.
/// You are responsible for ensuring this address aligns to 4-bytes.
///
/// Use an instance of ``MTLResidencySet`` to mark residency of the indirect buffer that the `indirectBuffer`
/// parameter references.
///
/// - Parameters:
///   - indirectBuffer:        GPUAddress of a ``MTLBuffer`` instance providing compute parameters.
///                            Lay out the data in this buffer as described in the
///                            ``MTLDispatchThreadgroupsIndirectArguments`` structure. This address
///                            requires 4-byte alignment.
///   - threadsPerThreadgroup: A ``MTLSize`` instance that represents the number of threads in one
///                            threadgroup, in each dimension.
- (void)dispatchThreadgroupsWithIndirectBuffer:(MTLGPUAddress)indirectBuffer
                         threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup;

/// Encodes a compute dispatch command with an arbitrarily sized grid, using an indirect buffer for arguments.
///
/// - Parameters:
///   - indirectBuffer: GPUAddress of a ``MTLBuffer`` instance providing arguments. Lay out the data
///                     in this buffer as described in the ``MTLDispatchThreadsIndirectArguments``
///                     structure. This address requires 4-byte alignment.
- (void)dispatchThreadsWithIndirectBuffer:(MTLGPUAddress)indirectBuffer;

/// Encodes a command to execute a series of commands from an indirect command buffer.
///
/// - Parameters:
///   - indirectCommandBuffer: ``MTLIndirectCommandBuffer`` instance containing the commands to execute.
///   - executionRange:        The range of commands to execute.
- (void)executeCommandsInBuffer:(id<MTLIndirectCommandBuffer>)indirectCommandBuffer
                      withRange:(NSRange)executionRange;

/// Encodes an instruction to execute commands from an indirect command buffer, using an indirect buffer for
/// arguments.
///
/// Use an instance of ``MTLResidencySet`` to mark residency of the indirect buffer that the `indirectRangeBuffer`
/// parameter references.
///
/// - Parameters:
///   - indirectCommandbuffer: ``MTLIndirectCommandBuffer`` instance containing the commands to execute.
///   - indirectRangeBuffer:   GPUAddress of a ``MTLBuffer`` containing the execution range. Lay out the data
///                            in this buffer as described in the ``MTLIndirectCommandBufferExecutionRange``
///                            structure. This address requires 4-byte alignment.
- (void)executeCommandsInBuffer:(id<MTLIndirectCommandBuffer>)indirectCommandbuffer
                 indirectBuffer:(MTLGPUAddress)indirectRangeBuffer;
```

`MTLIndirectCommandBuffer.h` L19–30:

```objc
typedef NS_OPTIONS(NSUInteger, MTLIndirectCommandType) {
    MTLIndirectCommandTypeDraw                = (1 << 0),
    MTLIndirectCommandTypeDrawIndexed         = (1 << 1),
    MTLIndirectCommandTypeDrawPatches         API_AVAILABLE(tvos(14.5)) = (1 << 2),
    MTLIndirectCommandTypeDrawIndexedPatches  API_AVAILABLE(tvos(14.5)) = (1 << 3) ,

    MTLIndirectCommandTypeConcurrentDispatch  API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(13.0)) = (1 << 5), /* Dispatch threadgroups with concurrent execution */

    MTLIndirectCommandTypeConcurrentDispatchThreads  API_AVAILABLE(macos(11.0), macCatalyst(14.0), ios(13.0)) = (1 << 6), /* Dispatch threads with concurrent execution */
    MTLIndirectCommandTypeDrawMeshThreadgroups API_AVAILABLE(macos(14.0), ios(17.0), tvos(18.1), visionos(2.1)) = (1 << 7),
    MTLIndirectCommandTypeDrawMeshThreads API_AVAILABLE(macos(14.0), ios(17.0), tvos(18.1), visionos(2.1)) = (1 << 8),
} API_AVAILABLE(macos(10.14), ios(12.0));
```

## 4. Metal Performance Primitives: matmul2d over chunks of k

`MPPTensorOpsMatMul2d.h` L158–200:

```cpp
//
// User can also take ownership of looping over reduction or k-dimension by
// choosing appropriate chunk size in k (called k-tile or tilek) For following
// example, we choose 16.
// kernel void matMulKLoop(tensor<device half, dextents<int32_t, 2>> A,
//                         tensor<device half, dextents<int32_t, 2>> B,
//                         tensor<device float, dextents<int32_t, 2>> C,
//                         constant uint& M, constant uint& N, constant uint& K,
//                         uint2 tgid [[threadgroup_position_in_grid]])
// {
//     // descriptor to create matmul operation that does 64x32 times 32x32
//     // producing 64x32
//     constexpr auto matmulDescriptor = matmul2d_descriptor(64,
//                                                           32,
//                                                           16); // tilek = 16, we loop over K in chunks of 16 rather than
//                                                                // letting matmul op run method looping over K
//                                                                // internally choose tileK
//
//     matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;
//
//     constexpr int tilek = 16;
//
//    // Inside thredgroup in both outer dimensions M and N.
//    if ( tgid.x*64 + 63 < M && tgid.y*32 + 31 < N)
//    {
//      auto tC = C.static_slice<32, 64>(tgid.x*32, tgid.y*64);
//      int k = 0;
//      for (; k + tilek - 1 < K; k += tilek)
//      {
//        auto tA = A.static_slice<tilek, 64>(k,tgid.y*64);
//        auto tB = B.static_slice<32, tilek>(tgid.x*32, k);
//
//        op.run(tA, tB, tC);
//      }
//
//      auto tA = A.static_slice<dynamic_extent, 64>(k, tgid.y*64);
//      auto tB = B.static_slice<32, dynamic_extent>(tgid.x*32, k);
//      op.run(tA, tB, tC);
//    }
//    else
//    {
//      auto tA = A.slice(0, tgid.y*64);
//      auto tB = B.slice(tgid.x*32, 0);
```

`MPPTensorOpsMatMul2d.h` L346–358:

```cpp
  min,
};

struct matmul2d_descriptor
{
  enum class mode
  {
    multiply,
    multiply_accumulate,
  };

  int m, n, k;
  bool transpose_left, transpose_right;
```
