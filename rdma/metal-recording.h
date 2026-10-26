#pragma once
#import <Metal/Metal.h>
#include "metal-capabilities.h"

#ifdef __cplusplus
extern "C" {
#endif

struct MetalReplay {
    _Alignas(MESH_REPLAY_ALIGNMENT) id<MTLIndirectCommandBuffer> commands;
    id<MTLResidencySet> residency;
    id owner;
    uint32_t count;
    uint16_t buffers, memory;
};
_Static_assert(sizeof(struct MetalReplay) == MESH_REPLAY_BYTES, "prepared native replay");
_Static_assert(_Alignof(struct MetalReplay) == MESH_REPLAY_ALIGNMENT, "prepared native replay alignment");

// docs/kernels.md E2, Any program
enum MetalForm { MetalFormIndirect, MetalFormEncoded, MetalFormSerial };
// Build operands from the selected Metal configuration, shared by frontends.
MTLCompileOptions *MetalCompilerOptions(id<MTLDevice> device);

// docs/kernels.md Any program.  A host program's operand whose storage holds `instances` slots of
// `stride` bytes: invocation t binds slot t % instances (0 instances: slot t), its writes landing
// `outputOffset` bytes past it (a stride: the next invocation's input).  Any buffer the block binds
// that is not named is bound as recorded, at every invocation.
struct MetalOperand { const char *name; __unsafe_unretained id<MTLBuffer> buffer; NSUInteger stride, instances, outputOffset; };
static inline NSUInteger MetalInvocationOffset(NSUInteger offset, NSUInteger stride, NSUInteger instances, NSUInteger invocation) {
    return offset + (instances ? invocation % instances : invocation) * stride;
}
// A recording device over `device`, which the caller makes and passes in: the command queues and
// compute pipelines made from it record the dispatches their compute encoders issue inside
// MetalRecord's `encode` (each pipeline's reflection kept for the dataflow, with the reaches each
// dispatch declares: MetalDispatchReaching), and run them otherwise.  Every other request is
// `device`'s.  A pipeline it did not make (a library's, such as libnccl-mesh's collectives) is
// refused inside `encode`: what it issues is not the program's.
id<MTLDevice> MetalRecordingDevice(id<MTLDevice> device, BOOL indirectPipelines);
// tools/mesh/schedule.py
void MetalRecordingList(int on);
// A prepared foreign stage supplies the byte views of its external tensors. Its pipelines
// still supply read/write modes; these views narrow their bindings during command capture.
struct MetalBindingView { __unsafe_unretained id<MTLBuffer> buffer; NSUInteger offset, length; };
void MetalRecordViews(const struct MetalBindingView *, size_t count, void (^encode)(void));
// docs/kernels.md Any program
enum { MetalReachRead = 1, MetalReachWritten = 2, MetalReachBinding = 4, MetalReachResident = 8 };
struct MetalReach { __unsafe_unretained id<MTLBuffer> buffer; NSUInteger offset, length; uint32_t access; };
// Preparation only: carry declared page accesses across encoders on the supplied
// queue. Prepare and enqueue its functions in the same order. Disjoint ranges
// add no dependency. Transport uses this declaration for its shared counters.
void MetalEncoderReaches(id<MTL4ComputeCommandEncoder>, const struct MetalReach *, size_t count);
// docs/kernels.md Any program
void MetalDispatchReaching(id<MTLComputeCommandEncoder> encoder, MTLSize groups, MTLSize threads,
    const struct MetalReach *reaches, size_t count);
// Records the compute commands `encode` issues through the queues of `device` (a recording device)
// as one invocation, and prepares `invocations` of them in `form`.  The replay, or NULL with the
// reason on stderr.
struct MetalReplay *MetalRecord(id<MTLDevice> device, void (^encode)(void), const struct MetalOperand *operands,
    size_t count, NSUInteger invocations, enum MetalForm form);
// docs/kernels.md#collectives
NSUInteger MetalRecordMark(void);
// A configured host share starts after its input commands. Only its declared
// consumers depend on completion. These declarations are captured with the
// numerical graph and use the same prepared schedule at execution.
void MetalRecordHost(id identity, BOOL consume, NSError *(^operation)(void));
// docs/kernels.md#collectives
typedef int (^MetalCrossing)(id<MTL4ComputeCommandEncoder> encoder, id<MTL4ArgumentTable> table, id<MTLResidencySet> residency,
    NSUInteger invocation, NSUInteger mark);
// docs/kernels.md#collectives
struct MetalSchedule;
// The execution context supplies its queue. Preparing invocation slots cannot
// create more queues; the number of slots is not a hardware resource count.
struct MetalSchedule *MetalScheduleCreateOn(id queue, enum MetalForm);
// An already encoded command from an external frontend enters the same
// submission and completion path. The command has not been committed.
struct MetalSchedule *MetalScheduleCommand(id<MTLCommandBuffer>);
void MetalScheduleAfter(struct MetalSchedule *, id<MTLEvent>, uint64_t value);
// Bind a host producer now and its data consumers later. The returned event is
// owned by the caller. Unrelated commands between these declarations can run
// while the CPU/ANE produces the result.
id<MTLSharedEvent> MetalScheduleHostProducer(struct MetalSchedule *, NSError *(^operation)(void));
void MetalScheduleHostConsumer(struct MetalSchedule *, id<MTLSharedEvent>);
id<MTLResidencySet> MetalScheduleResources(struct MetalSchedule *);
// Explicit bindings from a compiler which already has its pipelines. These
// enter the same access ordering and encoding used by captured commands.
struct MetalCommandBinding {
    __unsafe_unretained id<MTLBuffer> buffer;
    NSUInteger offset, length;
    uint32_t index, access;
};
// Native frontends can supply the same records directly instead of capturing
// encoder calls. Both routes use the common access-order and replay compiler.
struct MetalCommandDescription {
    __unsafe_unretained id<MTLComputePipelineState> pipeline;
    const struct MetalCommandBinding *bindings;
    size_t bindingCount;
    const NSUInteger *memory;
    size_t memoryCount;
    const struct MetalReach *reaches;
    size_t reachCount;
    MTLSize groups, threads;
    BOOL threadGrid;
};
struct MetalReplay *MetalReplayCommands(id<MTLDevice>, const struct MetalCommandDescription *, size_t count, enum MetalForm);
void MetalReplayEncode(struct MetalReplay *, id<MTLCommandBuffer>, NSRange commands);
struct MetalCommandBinding MetalScheduleData(struct MetalSchedule *, const void *, size_t bytes, uint32_t index);
void MetalScheduleCompute(struct MetalSchedule *, id<MTLComputePipelineState>,
    const struct MetalCommandBinding *, size_t count, MTLSize groups, MTLSize threads);
int MetalScheduleCross(struct MetalSchedule *, MetalCrossing);
NSError *MetalScheduleAppend(struct MetalSchedule *, struct MetalReplay *, NSRange invocations,
    NSArray<NSNumber *> *marks, MetalCrossing);
void MetalScheduleFinish(struct MetalSchedule *, void (^completed)(NSError *error));
void MetalScheduleEnqueue(struct MetalSchedule *);
// Ordinary completion observation: 0 pending, 1 complete, -1 failed.
int MetalScheduleStatus(const struct MetalSchedule *);
void MetalScheduleFree(struct MetalSchedule *);
// A replay MetalRecord returned, destroyed and freed.
void MetalReplayFree(struct MetalReplay *);

#ifdef __cplusplus
}
#endif
