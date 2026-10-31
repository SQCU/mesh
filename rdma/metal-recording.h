#pragma once
#import <Metal/Metal.h>
#include "metal-capabilities.h"
#include "mesh-host.h"
#include "nccl.h"
#import "mesh-tensor-layout.h"

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
enum MetalForm { MetalFormIndirect, MetalFormEncoded };
// Build operands from the selected Metal configuration, shared by frontends.
MTLCompileOptions *MetalDeviceCompilerOptions(id<MTLDevice> device);

// docs/kernels.md Any program.  A host program's operand whose storage holds `instances` slots of
// `stride` bytes: invocation t binds slot t % instances (0 instances: slot t).
// Each command already supplies its selected input/output binding offsets;
// storage naming never relocates writes to other values sharing that buffer.
struct MetalOperand { const char *name; __unsafe_unretained id<MTLBuffer> buffer; NSUInteger stride, instances; };
// Preparation storage from the shared bridge region. The caller owns the buffer.
id<MTLBuffer> MetalLeaseBuffer(id<MTLDevice>, NSUInteger bytes) NS_RETURNS_RETAINED;
static inline NSUInteger MetalInvocationOffset(NSUInteger offset, NSUInteger stride, NSUInteger instances, NSUInteger invocation, NSInteger displacement) {
    return offset + ncclMeshInvocationIndex(invocation, displacement, instances) * stride;
}
// tools/mesh/schedule.py: whether MetalReplayDescribed lists its commands
void MetalRecordingList(int on);
// docs/kernels.md Any program
// Observe qualifies only monotone progress reads. It retains the producer edge
// without an anti-dependency on later publications of the same progress word.
enum { MetalReachRead = 1, MetalReachWritten = 2, MetalReachBinding = 4, MetalReachResident = 8, MetalReachAddress = 16,
    MetalReachObserve = 32 };
// Binding reaches name their actual argument index. Several disjoint ranges
// can describe one view; all follow that argument's prepared invocation slot.
// Address reaches use binding = the command address-record ordinal.
// A non-binding reach uses binding = -1 (memory addressed through an operand).
// Resident-only reaches retain storage without declaring a data dependency.
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
// design/compile-once-2026-10-29.md: the index prelude whose layouts are function constants, one text for every command,
// and an affine view's values for it ([index, type (0 bool, 1 uint, 2 long, 3 ulong), value] each; nil: use the source form)
NSString *MetalViewIndexConstantPrelude(void);
NSArray<NSArray<NSNumber *> *> *MetalViewIndexConstants(NSUInteger binding, const NSUInteger *shape, const NSInteger *strides,
    size_t rank, NSUInteger elementOffset, NSInteger relativeOffset);
// Required API strides are equivalent when they address the same elements.
// Empty and singleton axes do not impose a different physical layout.
BOOL MetalViewLayoutMatches(const NSUInteger *shape, const NSInteger *strides,
    const NSInteger *required, size_t rank);
// AOT copy tiles, packed as (origin[rank], shape[rank]) words per tile.
NSData *MetalViewTiles(const NSUInteger *shape, size_t rank, NSUInteger elements);
// Factor a power-of-two butterfly into local pieces. A stage transforms
// `width` values separated by `step` in the logical row. The operation's
// width is independent of the selected device's local storage capacity.
struct MetalButterflyStage { NSUInteger width, step; };
NSData *MetalButterflyPlan(NSUInteger width, NSUInteger localElements);
// Eligibility of the configured fused schedule, never a limit on the graph.
BOOL MetalHadamardLocal(NSUInteger columns, NSUInteger mix, NSUInteger threads, NSUInteger memory);
// A logical slice of an encoded axis of `stored` entries. The axis's domain is
// its `count` present entries, `present` (ascending stored positions), and
// `order` maps each to its logical index; no entry outside the domain is named.
// Bounds/step are the normalized slice; block is the format's indivisible
// stored group. Returns int32 retained block indices ("blocks"), and the
// result's own domain: its present positions in the retained blocks
// ("present", ascending) and each one's logical index ("order"). A retained
// block's other lanes are outside that domain.
// Only immutable index metadata is read. Encoded numerical values stay in place.
NSDictionary<NSString *, NSData *> *MetalViewBlockSlice(const int32_t *present, const int32_t *order, size_t count,
    size_t stored, NSInteger first, NSInteger end, NSInteger step, NSUInteger block);
// Explicit logical selections, including reordered or repeated indices. A
// stored block can be named more than once when it supplies repeated outputs.
NSDictionary<NSString *, NSData *> *MetalViewBlockGather(const int32_t *present, const int32_t *order, size_t count,
    size_t stored, const int32_t *selected, size_t selectedCount, NSUInteger block);
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
// Composed factor coordinates are emitted source expressions with a proved
// vector span; nil coordinates use the ordinary mixed-radix digits of shape.
NSString *MetalViewMappedIndexDeclaration(NSUInteger binding, const NSUInteger *shape, const NSInteger *strides,
    size_t rank, NSUInteger elementOffset, NSDictionary<NSNumber *, NSData *> *maps,
    NSArray<NSString *> *coordinates, NSUInteger vectorSpan, NSInteger relativeOffset);
// Declared index relations, such as an encoding's row positions. These are
// compiler metadata, never a substitute for a computed index tensor. The
// relation is defined on its `count` values alone: mesh_position<B>(i) for
// i < mesh_position_count<B>(), a compiled constant.
NSString *MetalViewIndexValuesDeclaration(NSUInteger binding, const NSInteger *values, NSUInteger count);
// Adapt an already emitted mesh_index<0> declaration to the position ABI.
// Layout relations and explicit value arrays share this source emitter.
NSString *MetalViewPositionDeclaration(NSUInteger binding, NSString *indices, NSUInteger count, NSUInteger span);
NSData *MetalViewMappedReaches(id<MTLBuffer> buffer, NSUInteger offset, const NSUInteger *shape,
    const NSInteger *strides, size_t rank, NSUInteger elementBytes, uint32_t access, int32_t binding,
    NSDictionary<NSNumber *, NSData *> *maps);
// Union prepared element spans with the same byte-access representation.
NSData *MetalTensorSpanReaches(id<MTLBuffer> buffer, NSUInteger base, NSData *spans,
    NSUInteger elementBytes, uint32_t access, int32_t binding);
// Preparation only: carry declared page accesses across encoders on the supplied
// queue. Prepare and enqueue its functions in the same order. Disjoint ranges
// add no dependency. Transport uses this declaration for its shared counters.
void MetalEncoderReaches(id<MTL4ComputeCommandEncoder>, const struct MetalReach *, size_t count);
// One dispatch's accesses enter the same prepared phase frontier as replayed
// commands. The result requests a barrier only for a real prior hazard.
BOOL MetalEncoderOrder(id<MTL4ComputeCommandEncoder>, const struct MetalReach *, size_t count);
// docs/kernels.md#collectives
typedef int (^MetalCrossing)(id<MTL4ComputeCommandEncoder> encoder, id<MTL4ArgumentTable> table, id<MTLResidencySet> residency,
    NSUInteger invocation, NSUInteger mark);
// docs/kernels.md#collectives
struct MetalSchedule;
struct ncclComm;
struct ncclMeshGroup;
// design/replay-templates-2026-10-29.md: what a frontend makes a host operation or a crossing from again, for a
// template: its kind, its bytes, and the places in them that name a region lease's memory or a communicator. An
// address place holds a host pointer; a buffer place holds an id<MTLBuffer> and, after it, its byte offset (an
// ncclMeshBuffer); a communicator place holds a struct ncclComm *. Instantiation rewrites each place to the instance's
// operand or communicator before the frontend reads the bytes.
enum MetalRelocation { MetalRelocationAddress, MetalRelocationBuffer, MetalRelocationCommunicator };
struct MetalPlace { size_t at; enum MetalRelocation relocation; };
struct MetalDeclaration { const char *kind; const void *bytes; size_t length; const struct MetalPlace *places; size_t count; };
// The execution context supplies its queue. Preparing invocation slots cannot
// create more queues; the number of slots is not a hardware resource count.
struct MetalSchedule *MetalScheduleCreateOn(id<MTL4CommandQueue> queue, enum MetalForm);
void MetalScheduleAfter(struct MetalSchedule *, id<MTLEvent>, uint64_t value);
// Preparation records the function's collective result sources. Observers read
// their existing failure state; this adds no command or transfer-time check.
void MetalScheduleCommunicator(struct MetalSchedule *, struct ncclComm *);
// Preparation binds each invocation's completion and retained task. Submission
// never allocates a task and never stands for output completion.
typedef const char *(^MetalHostOperation)(MeshHostCompletion, void *, MeshHostTask *);
// A host operation over its declared views; its declaration (NULL: none, and its function makes no template).
void MetalScheduleHost(struct MetalSchedule *, const struct MetalReach *, size_t count, const struct MetalDeclaration *,
    MetalHostOperation);
id<MTLResidencySet> MetalScheduleResources(struct MetalSchedule *);
// Explicit bindings from a compiler which already has its pipelines. These
// enter the same access ordering and encoding as described commands.
struct MetalCommandBinding {
    __unsafe_unretained id<MTLBuffer> buffer;
    NSUInteger offset, length;
    uint32_t index, access;
    NSInteger invocation; // Signed phase of this value: slot (t + invocation) modulo instances.
};
// length is the exact dense extent, including zero for an empty view. Binding
// reaches replace it with a union of strided/masked regions. Neither form is an
// instruction to access the remainder of a buffer.
// A pointer word inside one uniform argument. Preparation writes its target
// address for each invocation; the numerical command reads only prepared pages.
// Other uniform bytes are preparation constants. Runtime-produced values and
// page indices remain ordinary page operands, not bytes captured in this table.
struct MetalCommandAddress {
    uint32_t binding;
    NSUInteger offset;
    __unsafe_unretained id<MTLBuffer> buffer;
    NSUInteger origin;
    NSInteger invocation;
};
// Every frontend supplies its commands as these records, to the one access-order
// and replay compiler (MetalReplayCommands, MetalReplayDescribed).
struct MetalCommandDescription {
    __unsafe_unretained id<MTLComputePipelineState> pipeline;
    const struct MetalCommandBinding *bindings;
    size_t bindingCount;
    const NSUInteger *memory;
    size_t memoryCount;
    const struct MetalReach *reaches;
    size_t reachCount;
    const struct MetalCommandAddress *addresses;
    size_t addressCount;
    MTLSize groups, threads;
    BOOL threadGrid;
};
struct MetalReplay *MetalReplayCommands(id<MTLDevice>, const struct MetalCommandDescription *, size_t count, enum MetalForm);
// docs/kernels.md#collectives: a host operation of a described program, run in each invocation before its command `at`
// (after those before it). Host operations declare the same read/write views as GPU commands. The
// compiler orders their producers and consumers; there are no consume markers.
// These are prepared native operations, not callbacks into a frontend. CPU-only
// views use a nil buffer and their address as offset; Metal views use buffer-relative
// offsets. The bound operation retains any CPU storage it uses. The replay holds Metal
// buffers. Host views/operations stay bound to these operands across replay slots.
struct MetalHostDescription {
    NSUInteger at;
    const struct MetalReach *reaches;
    size_t count;
    __unsafe_unretained MetalHostOperation operation;
};
// docs/kernels.md Any program: a program's `count` commands as one invocation, each named operand bound by its slots,
// and its host operations in command order, prepared as `invocations` of them in `form` (the indirect form's command
// buffers optimized), with the program's census on stdout.  The replay, or NULL with the reason on stderr.
// MetalReplayCommands is its single invocation with no operands or host operations, and no census.
struct MetalReplay *MetalReplayDescribed(id<MTLDevice>, const struct MetalCommandDescription *, size_t count,
    const struct MetalHostDescription *hosts, size_t hostCount, const struct MetalOperand *operands, size_t operandCount,
    NSUInteger invocations, enum MetalForm);
/* The bytes of the type a pipeline's kernel declares at buffer `index` (its reflection's bufferDataSize; 0 where its
   pipeline was not made by MetalPipelineMake, or the kernel uses no buffer there), and its label */
NSUInteger MetalPipelineDeclaredBytes(id<MTLComputePipelineState>, NSUInteger index);
NSString *MetalPipelineName(id<MTLComputePipelineState>);
struct MetalCommandBinding MetalScheduleData(struct MetalSchedule *, const void *, size_t bytes, uint32_t index);
void MetalScheduleCompute(struct MetalSchedule *, id<MTLComputePipelineState>,
    const struct MetalCommandBinding *, size_t count, MTLSize groups, MTLSize threads);
// A crossing's actual local accesses, compiled with the same dependency frontier
// as numerical work. Encoding still completes during preparation: callers may
// consume the publication records it produces. A fused descriptor publication
// names residency here; its producer/consumer commands declare payload accesses.
// The group supplies active transport progress words before dependency ordering.
// Templates rebuild that operand after relocation; they never store its raw words.
int MetalScheduleCross(struct MetalSchedule *, const struct MetalReach *, size_t count,
    const struct ncclMeshGroup *, const struct MetalDeclaration *, MetalCrossing);
// A range of a prepared replay's commands (MetalReplayCommands, MetalReplayDescribed), its first invocation's, as the schedule's
// next work, in the replay's form and ordered by its recorded reaches.
void MetalScheduleReplay(struct MetalSchedule *, struct MetalReplay *, NSRange commands);
// `bytes` from `source` at `from` into `destination` at `to`, the Metal 4 encoder's copy, ordered after and before the
// schedule's other work.
void MetalScheduleCopy(struct MetalSchedule *, id<MTLBuffer> source, NSUInteger from, id<MTLBuffer> destination, NSUInteger to,
    NSUInteger bytes);
typedef NSData *(^MetalCrossingViews)(NSUInteger invocation, NSUInteger mark, struct ncclMeshGroup *group);
NSError *MetalScheduleAppend(struct MetalSchedule *, struct MetalReplay *, NSRange invocations,
    NSArray<NSNumber *> *marks, MetalCrossingViews, MetalCrossing);
// Finish all local parts and exchange their compiled list ends with peers.
// Returns the collective preparation error, or zero. Enqueue does no exchange.
int MetalScheduleFinish(struct MetalSchedule *, void (^completed)(NSError *error));
void MetalScheduleEnqueue(struct MetalSchedule *);
// Ordinary completion observation: 0 pending, 1 complete, -1 failed.
int MetalScheduleStatus(const struct MetalSchedule *, NSError **error);
void MetalScheduleFree(struct MetalSchedule *);
// A replay MetalReplayCommands or MetalReplayDescribed returned, destroyed and freed.
void MetalReplayFree(struct MetalReplay *);

// design/replay-templates-2026-10-29.md. A pipeline's declaration: the library it is specialized from, its function,
// its function-constant values ([index, MTLDataType, bytes] each) and its descriptor's fields (label; threads, the most
// threads a threadgroup holds; multiple, a threadgroup size a multiple of the execution width; indirect, support for
// indirect commands). MetalPipelineMake makes the pipeline a declaration names, from `archives` where they hold it,
// and declares it; the caller owns the result. A frontend that makes its pipelines otherwise declares each.
id<MTLComputePipelineState> MetalPipelineMake(id<MTLLibrary> library, NSString *function, NSArray<NSArray *> *constants,
    NSDictionary<NSString *, id> *descriptor, NSArray<id<MTLBinaryArchive>> *archives, NSError **error);
void MetalPipelineDeclare(id<MTLComputePipelineState> pipeline, id<MTLLibrary> library, NSString *function,
    NSArray<NSArray *> *constants, NSDictionary<NSString *, id> *descriptor);
// A program's template: the calls that built each of its `count` functions (schedules), `bound` their first operands
// in order, every other buffer they name kept by value, shared by them all; a plan the functions' crossings assume is
// declared once where it is the same plan. A single function is a program of one. Pipelines are their declarations.
// A bound operand is a view: its buffer and the byte offset of the
// view in it (`offsets`, nil: 0), so an instance's view elsewhere moves every use of that operand's lease with it.
// Its device's compile facts and each pipeline's thread limits are kept, since its libraries are compiled for them.
// Nil, with the reason, where a pipeline, host operation or crossing has no declaration.
NSDictionary *MetalScheduleTemplate(struct MetalSchedule *const *schedules, size_t count, NSArray<id<MTLBuffer>> *bound,
    NSArray<NSNumber *> *offsets, NSError **error);
// A template in `directory`: template.plist, each library as library.<i>.metallib and the pipelines of the device that
// wrote it as pipelines.<architecture>.metallib.
NSError *MetalTemplateWrite(NSDictionary *, NSString *directory);
// A template read from `directory` for `device`: its libraries loaded, and its pipelines for this device's
// architecture where they were written.
NSDictionary *MetalTemplateRead(id<MTLDevice>, NSString *directory, NSError **error);
// This device's pipelines added to the template in `directory`, where it was read without them.
NSError *MetalTemplateArchive(NSDictionary *, NSString *directory);
// A template's calls made on `schedule`, `bound` its first operands; its kept operands are leased anew, and its
// communicators are `communicators` (each a struct ncclComm * as an NSValue), in the order its crossings first named
// theirs. A host operation's or crossing's declaration, its places rewritten to the instance's, is made again by the
// frontend's `host` or `cross` (nil, with the reason, where it cannot be), which returns an autoreleased block that
// lives as long as `schedule`.
typedef MetalHostOperation (^MetalTemplateHost)(NSString *kind, NSData *bytes, NSError **error);
typedef MetalCrossing (^MetalTemplateCross)(NSString *kind, NSData *bytes, struct MetalSchedule *schedule, struct ncclMeshGroup *group, NSError **error);
// A template made where the device's compile facts differ, or where a dispatch's threadgroup holds more threads than
// this device's pipeline takes, fails as a value naming them.
NSError *MetalScheduleInstantiate(struct MetalSchedule *const *schedules, size_t count, NSDictionary *,
    NSArray<id<MTLBuffer>> *bound, NSArray<NSNumber *> *offsets, NSArray<NSValue *> *communicators, MetalTemplateHost host,
    MetalTemplateCross cross);

#ifdef __cplusplus
}
#endif
