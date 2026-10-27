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
MTLCompileOptions *MetalDeviceCompilerOptions(id<MTLDevice> device);

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
// still supply read/write modes; these views define their bindings during command capture.
// Every accessed binding needs a view or dispatch reach (setBytes declares its
// own length). A missing view fails preparation; allocation/slot sizes are not
// substitutes. Providers must expose internal tensor views as well as I/O.
struct MetalBindingView { __unsafe_unretained id<MTLBuffer> buffer; NSUInteger offset, length; };
void MetalRecordViews(const struct MetalBindingView *, size_t count, void (^encode)(void));
// docs/kernels.md Any program
enum { MetalReachRead = 1, MetalReachWritten = 2, MetalReachBinding = 4, MetalReachResident = 8 };
// Binding reaches name their actual argument index. Several disjoint ranges
// can describe one view; all follow that argument's prepared invocation slot.
// A non-binding reach uses binding = -1 (memory addressed through an operand).
struct MetalReach {
    __unsafe_unretained id<MTLBuffer> buffer;
    NSUInteger offset, length;
    uint32_t access;
    int32_t binding;
};
// Preparation only: the union of a strided view's byte accesses, returned as
// packed MetalReach records. Strides are bytes; traversal order is irrelevant
// to dependencies. Empty views retain a zero-length binding declaration.
NSData *MetalViewReaches(id<MTLBuffer>, NSUInteger offset, const NSUInteger *shape,
    const NSInteger *strides, size_t rank, NSUInteger elementBytes, uint32_t access, int32_t binding);
// Preparation-only address lowering. The result maps a logical linear index
// to an element offset; shape/stride descriptors do not enter the kernel.
NSString *MetalViewIndexExpression(const NSUInteger *shape, const NSInteger *strides,
    size_t rank, NSString *index);
NSString *MetalViewIndexPrelude(void);
// AOT copy tiles, packed as (origin[rank], shape[rank]) words per tile.
NSData *MetalViewTiles(const NSUInteger *shape, size_t rank, NSUInteger elements);
// A logical slice of an encoded axis. `order` maps stored entries to logical
// indices (-1 for unused entries). Bounds/step are the normalized slice;
// block is the format's indivisible stored group. Returns int32 block indices
// and a compacted int32 order, with -1 for unused lanes of retained blocks.
// Only immutable index metadata is read. Encoded numerical values stay in place.
NSDictionary<NSString *, NSData *> *MetalViewBlockSlice(const int32_t *order, size_t count,
    NSInteger first, NSInteger end, NSInteger step, NSUInteger block);
// One dependency compiler for recorded commands and fused producer/consumer
// tiles. Each entry contains MetalReach records in execution order. The result
// names prior writers/readers of overlapping ranges; shared aliases share keys.
NSArray<NSIndexSet *> *MetalViewDependencies(NSArray<NSData *> *accesses);
// A fused bipartite relation: first `producers` entries write, the remaining
// entries read. Emit fixed adjacency/arrival counts from those dependencies.
NSString *MetalViewDependencyDeclaration(NSString *name, NSArray<NSIndexSet *> *dependencies, NSUInteger producers);
NSString *MetalViewIndexDeclaration(NSUInteger binding, const NSUInteger *shape, const NSInteger *strides,
    size_t rank, NSUInteger elementOffset);
// Axis maps contain signed offsets, in the same units as strides. A mapped
// axis replaces that axis's multiplication by stride. Maps are consumed at
// preparation: generated shaders contain arithmetic or one immutable lookup.
// Normalize affine maps into strides and a base offset before API binding.
NSDictionary<NSString *, id> *MetalViewNormalizeAxes(const NSInteger *strides, size_t rank,
    NSDictionary<NSNumber *, NSData *> *maps);
NSString *MetalViewMappedIndexDeclaration(NSUInteger binding, const NSUInteger *shape, const NSInteger *strides,
    size_t rank, NSUInteger elementOffset, NSDictionary<NSNumber *, NSData *> *maps);
NSData *MetalViewMappedReaches(id<MTLBuffer> buffer, NSUInteger offset, const NSUInteger *shape,
    const NSInteger *strides, size_t rank, NSUInteger elementBytes, uint32_t access, int32_t binding,
    NSDictionary<NSNumber *, NSData *> *maps);
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
// Host operations declare the same read/write views as GPU commands. The
// compiler orders their producers and consumers; there are no consume markers.
// These are prepared native operations, not callbacks into a frontend. CPU-only
// views use a nil buffer and their address as offset; Metal views use buffer-relative
// offsets. The bound operation retains any CPU storage it uses. Capture holds Metal
// buffers. Host views/operations stay bound to these operands across replay slots.
void MetalRecordHost(const struct MetalReach *, size_t count, NSError *(^operation)(void));
// docs/kernels.md#collectives
typedef int (^MetalCrossing)(id<MTL4ComputeCommandEncoder> encoder, id<MTL4ArgumentTable> table, id<MTLResidencySet> residency,
    NSUInteger invocation, NSUInteger mark);
// docs/kernels.md#collectives
struct MetalSchedule;
struct ncclComm;
// The execution context supplies its queue. Preparing invocation slots cannot
// create more queues; the number of slots is not a hardware resource count.
struct MetalSchedule *MetalScheduleCreateOn(id queue, enum MetalForm);
// An already encoded command from an external frontend enters the same
// submission and completion path. The command has not been committed.
struct MetalSchedule *MetalScheduleCommand(id<MTLCommandBuffer>);
void MetalScheduleAfter(struct MetalSchedule *, id<MTLEvent>, uint64_t value);
// Preparation records the function's collective result sources. Observers read
// their existing failure state; this adds no command or transfer-time check.
void MetalScheduleCommunicator(struct MetalSchedule *, struct ncclComm *);
void MetalScheduleHost(struct MetalSchedule *, const struct MetalReach *, size_t count, NSError *(^operation)(void));
id<MTLResidencySet> MetalScheduleResources(struct MetalSchedule *);
// Explicit bindings from a compiler which already has its pipelines. These
// enter the same access ordering and encoding used by captured commands.
struct MetalCommandBinding {
    __unsafe_unretained id<MTLBuffer> buffer;
    NSUInteger offset, length;
    uint32_t index, access;
};
// length is the exact dense extent, including zero for an empty view. Binding
// reaches replace it with a union of strided/masked regions. Neither form is an
// instruction to access the remainder of a buffer.
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
// Finish all local parts and exchange their compiled list ends with peers.
// Returns the collective preparation error, or zero. Enqueue does no exchange.
int MetalScheduleFinish(struct MetalSchedule *, void (^completed)(NSError *error));
void MetalScheduleEnqueue(struct MetalSchedule *);
// Ordinary completion observation: 0 pending, 1 complete, -1 failed.
int MetalScheduleStatus(const struct MetalSchedule *, NSError **error);
void MetalScheduleFree(struct MetalSchedule *);
// A replay MetalRecord returned, destroyed and freed.
void MetalReplayFree(struct MetalReplay *);

#ifdef __cplusplus
}
#endif
