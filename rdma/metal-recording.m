#import "metal-recording.h"
#include "nccl.h"
#import <objc/runtime.h>
#include <stdatomic.h>

struct MetalRecordedDispatch {
    id<MTLComputePipelineState> pipeline;
    id<MTLBuffer> buffers[MESH_BUFFER_BINDINGS];
    NSUInteger offsets[MESH_BUFFER_BINDINGS], memory[MESH_MEMORY_BINDINGS],
        strides[MESH_BUFFER_BINDINGS], instances[MESH_BUFFER_BINDINGS], lengths[MESH_BUFFER_BINDINGS];
    MTLSize grid, group;
    BOOL threads, barrier;
    uint64_t access;
    // docs/kernels.md Any program: the reaches the dispatch declared (MetalDispatchReaching), `reaches` of the
    // capture's from `reach`, and the bindings whose extent one of them declares (bit j: binding j)
    uint32_t reach, reaches, extents;
};

@interface MetalCapture : NSObject {
@public
    NSMutableData *dispatches, *reaches;
    NSMutableSet *owners, *resources;
    NSMutableArray *preparation, *hosts;
    NSUInteger buffers, memory;
}
@end

static _Thread_local MetalCapture *activeCapture;
static _Thread_local const struct MetalBindingView *bindingViews;
static _Thread_local size_t bindingViewCount;
static void MetalDispatchAccesses(id<MTLComputeCommandEncoder>,MTLSize,MTLSize,BOOL,const struct MetalReach *,size_t);

void MetalRecordViews(const struct MetalBindingView *views,size_t count,void (^encode)(void)) {
    const struct MetalBindingView *previous=bindingViews;size_t before=bindingViewCount;
    bindingViews=views;bindingViewCount=count;
    @try { encode(); } @finally { bindingViews=previous;bindingViewCount=before; }
}

static char replaySegmentsKey,replayRecordsKey,replayResourcesKey,replayHostsKey,pipelineAccessKey,pipelineDescriptorKey,pipelineBindingsKey,
    pipelineAddressesKey,operandStrideKey,operandInstancesKey,operandOutputOffsetKey;

static void MetalReplayDestroy(struct MetalReplay *replay) {
    [replay->commands release]; [replay->residency release]; [replay->owner release];
    *replay = (struct MetalReplay){0};
}

// docs/kernels.md Any program: whether a reflected structure holds an address (a pointer member,
// or one in a nested structure or array), through which a kernel reaches memory no binding names.
static BOOL MetalStructAddresses(MTLStructType *structure) {
    for (MTLStructMember *member in structure.members) {
        MTLDataType type = member.dataType;
        MTLStructType *inner = member.structType;
        for (MTLArrayType *array = member.arrayType; array; array = array.elementArrayType) {
            type = array.elementType; inner = array.elementStructType;
        }
        if (type == MTLDataTypePointer || (type == MTLDataTypeStruct && MetalStructAddresses(inner))) return YES;
    }
    return NO;
}

// ../dox/mesh/design/prepared-machine.md#M06
// A pipeline of a recording device, with its reflection kept: the buffers it reads and writes, the
// bindings and threadgroup memory it declares, and whether an argument holds an address.
static id<MTLComputePipelineState> MetalCreatePipeline(id<MTLDevice> device,
    MTLComputePipelineDescriptor *descriptor,BOOL indirect,MTLPipelineOption options,
    MTLComputePipelineReflection **reflection,NSError **error) {
    descriptor = [descriptor copy];
    if(indirect)descriptor.supportIndirectCommandBuffers = YES;
    MTLComputePipelineReflection *bindings = nil;
    id result = [device newComputePipelineStateWithDescriptor:descriptor options:options|MTLPipelineOptionBindingInfo|MTLPipelineOptionBufferTypeInfo reflection:&bindings error:error];
    if (reflection) *reflection = bindings;
    if (!result) { [descriptor release]; return nil; }
    uint64_t access = 0, declared = 0;
    BOOL addresses = NO;
    for (id<MTLBinding> binding in bindings.bindings) {
        if(binding.type==MTLBindingTypeThreadgroupMemory && binding.isUsed)
            declared |= UINT64_C(1) << (MESH_ACCESS_WRITE_SHIFT+binding.index);
        if(binding.type!=MTLBindingTypeBuffer)continue;
        declared |= UINT64_C(1) << binding.index;
        if (!binding.isUsed) continue;
        id<MTLBufferBinding> buffer = (id)binding;
        addresses |= buffer.bufferPointerType.elementIsArgumentBuffer || buffer.bufferDataType == MTLDataTypePointer ||
            (buffer.bufferDataType == MTLDataTypeStruct && MetalStructAddresses(buffer.bufferStructType));
        if (binding.access != MTLBindingAccessWriteOnly) access |= UINT64_C(1) << binding.index;
        if (binding.access != MTLBindingAccessReadOnly) access |= UINT64_C(1) << (MESH_ACCESS_WRITE_SHIFT + binding.index);
    }
    objc_setAssociatedObject(result, &pipelineAccessKey, @(access), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    if (addresses) objc_setAssociatedObject(result, &pipelineAddressesKey, @YES, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    objc_setAssociatedObject(result, &pipelineBindingsKey, @(declared), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    objc_setAssociatedObject(result, &pipelineDescriptorKey, descriptor, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    [descriptor release]; return result;
}

// docs/kernels.md Any program: the dataflow's identity of a buffer, its allocation.  A heap's
// buffers may alias one another (makeAliasable), so each is its heap.
static NSValue *MetalAllocationKey(id buffer) {
    id heap=[buffer respondsToSelector:@selector(heap)]?[buffer heap]:nil;
    return [NSValue valueWithPointer:heap?:buffer];
}

// docs/kernels.md Any program: where a buffer starts in its allocation
static NSUInteger MetalAllocationBase(id<MTLBuffer> buffer) {
    return [buffer respondsToSelector:@selector(heap)]&&[buffer heap]?buffer.heapOffset:0;
}

typedef void (^MetalAccess)(id<MTLBuffer> buffer, NSUInteger offset, NSUInteger bytes, BOOL load, BOOL store);
// docs/kernels.md Any program: a command's accesses, each a buffer's bytes [offset, offset + bytes) read or written:
// each binding its pipeline reads or writes, from its offset to its buffer's end (a slot's stride where it rings), but
// where a reach declares its extent (MetalReachBinding); then each reach the dispatch declared.
static void MetalAccesses(const struct MetalRecordedDispatch *command, const struct MetalReach *reaches, NSUInteger invocation, MetalAccess each) {
    for(NSUInteger j=0;j<MESH_BUFFER_BINDINGS;j++) {
        id<MTLBuffer> buffer=command->buffers[j];
        const BOOL load=(command->access>>j)&1,store=(command->access>>(MESH_ACCESS_WRITE_SHIFT+j))&1;
        if(!buffer||(!load&&!store)||((command->extents>>j)&1))continue;
        const NSUInteger offset=MetalInvocationOffset(command->offsets[j],command->strides[j],command->instances[j],invocation);
        each(buffer,offset,command->lengths[j]?command->lengths[j]:command->strides[j]?command->strides[j]:buffer.length>offset?buffer.length-offset:0,load,store);
    }
    for(uint32_t k=0;k<command->reaches;k++) {
        const struct MetalReach *reach=reaches+command->reach+k;
        if(!reach->length)continue;
        NSUInteger offset=reach->offset;
        if(reach->access&MetalReachBinding)
            offset=MetalInvocationOffset(offset,command->strides[reach->binding],command->instances[reach->binding],invocation);
        each(reach->buffer,offset,reach->length,(reach->access&MetalReachRead)!=0,(reach->access&MetalReachWritten)!=0);
    }
}

// docs/kernels.md Any program: bytes [lo, hi) of an allocation, accessed by commands of levels below `done`
struct MetalSpan { NSUInteger lo, hi, done; };

// Accesses still relevant to later encoders. Weak allocation keys avoid keeping
// a finished function's pages alive through its queue. The function itself owns
// every buffer until completion. Overlapping ranges are merged at preparation.
@interface MetalQueueAccess : NSObject {
@public NSMapTable *read, *written;
}
@end
@implementation MetalQueueAccess
- (instancetype)init {
    self=[super init];
    read=[[NSMapTable weakToStrongObjectsMapTable] retain];
    written=[[NSMapTable weakToStrongObjectsMapTable] retain];return self;
}
- (void)dealloc { [read release];[written release];[super dealloc]; }
@end
static char queueAccessKey,encoderAccessKey;

static BOOL MetalRangeOverlaps(NSMapTable *map,id allocation,NSUInteger lo,NSUInteger hi) {
    NSData *data=[map objectForKey:allocation];const struct MetalSpan *spans=data.bytes;
    for(NSUInteger i=0;i<data.length/sizeof(*spans);i++)
        if(spans[i].lo<hi&&lo<spans[i].hi)return YES;
    return NO;
}

static void MetalRangeAdd(NSMapTable *map,id allocation,NSUInteger lo,NSUInteger hi) {
    if(lo==hi)return;
    NSMutableData *data=[map objectForKey:allocation];
    if(!data){data=[NSMutableData data];[map setObject:data forKey:allocation];}
    struct MetalSpan *spans=data.mutableBytes;
    for(NSUInteger i=0;i<data.length/sizeof(*spans);) {
        if(spans[i].lo<=hi&&lo<=spans[i].hi) {
            lo=MIN(lo,spans[i].lo);hi=MAX(hi,spans[i].hi);
            spans[i]=spans[data.length/sizeof(*spans)-1];data.length-=sizeof(*spans);
            spans=data.mutableBytes;i=0;
        } else i++;
    }
    const struct MetalSpan span={lo,hi,0};[data appendBytes:&span length:sizeof span];
}

static void MetalRangesMerge(NSMapTable *into,NSMapTable *from) {
    for(id allocation in from.keyEnumerator) {
        NSData *data=[from objectForKey:allocation];const struct MetalSpan *spans=data.bytes;
        for(NSUInteger i=0;i<data.length/sizeof(*spans);i++)MetalRangeAdd(into,allocation,spans[i].lo,spans[i].hi);
    }
}

void MetalEncoderReaches(id<MTL4ComputeCommandEncoder> encoder,const struct MetalReach *reaches,size_t count) {
    NSArray *scope=objc_getAssociatedObject(encoder,&encoderAccessKey);
    if(!scope)[NSException raise:NSInvalidArgumentException format:@"Metal accesses require a prepared schedule encoder"];
    MetalQueueAccess *previous=scope[0],*current=scope[1];
    BOOL dependency=NO;
    for(size_t i=0;i<count;i++) {
        const struct MetalReach r=reaches[i];if(!r.length||!(r.access&3))continue;
        id allocation=(id)r.buffer.heap?:r.buffer;NSUInteger lo=MetalAllocationBase(r.buffer)+r.offset;
        dependency|=MetalRangeOverlaps(previous->written,allocation,lo,lo+r.length)||
            ((r.access&MetalReachWritten)&&MetalRangeOverlaps(previous->read,allocation,lo,lo+r.length));
    }
    if(dependency) {
        [encoder barrierAfterQueueStages:MTLStageDispatch beforeStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        [previous->read removeAllObjects];[previous->written removeAllObjects];
    }
    for(size_t i=0;i<count;i++) {
        const struct MetalReach r=reaches[i];if(!r.length||!(r.access&3))continue;
        id allocation=(id)r.buffer.heap?:r.buffer;NSUInteger lo=MetalAllocationBase(r.buffer)+r.offset;
        if(r.access&MetalReachRead)MetalRangeAdd(current->read,allocation,lo,lo+r.length);
        if(r.access&MetalReachWritten)MetalRangeAdd(current->written,allocation,lo,lo+r.length);
    }
}

static void MetalEncoderAccesses(id<MTL4ComputeCommandEncoder> encoder,const struct MetalRecordedDispatch *command,
    const struct MetalReach *reaches,NSUInteger invocation) {
    MetalAccesses(command,reaches,invocation,^(id<MTLBuffer> buffer,NSUInteger offset,NSUInteger bytes,BOOL load,BOOL store){
        const struct MetalReach r={buffer,offset,bytes,(load?MetalReachRead:0)|(store?MetalReachWritten:0),-1};
        MetalEncoderReaches(encoder,&r,1);
    });
}

// docs/kernels.md Any program: the greatest `done` of the spans overlapping [lo, hi), each span no later command can
// wait on (its `done` at most `phase`, every later level's floor) dropped
static NSUInteger MetalSpansAfter(NSMutableData *spans, NSUInteger lo, NSUInteger hi, NSUInteger phase) {
    struct MetalSpan *at=spans.mutableBytes;
    NSUInteger count=spans.length/sizeof(*at),kept=0,after=0;
    for(NSUInteger s=0;s<count;s++) {
        if(at[s].done<=phase)continue;
        if(at[s].lo<hi&&lo<at[s].hi)after=MAX(after,at[s].done);
        at[kept++]=at[s];
    }
    spans.length=kept*sizeof(*at);
    return after;
}

static void MetalSpanAdd(NSMutableDictionary<NSValue *,NSMutableData *> *spans, NSValue *key, NSUInteger lo, NSUInteger hi, NSUInteger done) {
    NSMutableData *held=spans[key]?:(spans[key]=[NSMutableData data]);
    const struct MetalSpan span={lo,hi,done};
    [held appendBytes:&span length:sizeof span];
}

// One preparation-time access-order calculation for records, repeated slots and whole functions.
static BOOL MetalOrder(NSMutableDictionary *read,NSMutableDictionary *written,NSUInteger *phase,
    const struct MetalRecordedDispatch *command,const struct MetalReach *reaches,NSUInteger invocation) {
    __block NSUInteger level=*phase;const NSUInteger before=*phase;
    MetalAccesses(command,reaches,invocation,^(id<MTLBuffer> buffer,NSUInteger offset,NSUInteger bytes,BOOL load,BOOL store){
        (void)load;NSValue *key=MetalAllocationKey(buffer);NSUInteger lo=MetalAllocationBase(buffer)+offset;
        level=MAX(level,MetalSpansAfter(written[key],lo,lo+bytes,before));
        if(store)level=MAX(level,MetalSpansAfter(read[key],lo,lo+bytes,before));
    });
    MetalAccesses(command,reaches,invocation,^(id<MTLBuffer> buffer,NSUInteger offset,NSUInteger bytes,BOOL load,BOOL store){
        NSValue *key=MetalAllocationKey(buffer);NSUInteger lo=MetalAllocationBase(buffer)+offset;
        if(load)MetalSpanAdd(read,key,lo,lo+bytes,level+1);
        if(store)MetalSpanAdd(written,key,lo,lo+bytes,level+1);
    });
    BOOL barrier=level!=*phase;*phase=level;return barrier;
}

static void MetalDataflow(MetalCapture *capture) {
    struct MetalRecordedDispatch *commands=capture->dispatches.mutableBytes;
    NSUInteger phase=0,count=capture->dispatches.length/sizeof(*commands);
    NSMutableDictionary *written=[NSMutableDictionary dictionary],*read=[NSMutableDictionary dictionary];
    for(NSUInteger i=0;i<count;i++)commands[i].barrier=MetalOrder(read,written,&phase,commands+i,capture->reaches.bytes,0);
}

@interface MetalForwarder : NSProxy {
@public id target;
}
- (instancetype)initWithTarget:(id)object;
@end

@interface MetalDeviceRecorder : MetalForwarder { @public BOOL indirect; }
@end
@interface MetalComputeRecorder : MetalForwarder {
@public MetalCapture *capture;
    struct MetalRecordedDispatch state;
}
- (void)dispatchGrid:(MTLSize)grid threadsPerThreadgroup:(MTLSize)group threadGrid:(BOOL)threads reaches:(const struct MetalReach *)reaches count:(size_t)count;
@end
@interface MetalCommandRecorder : MetalForwarder
@end
@interface MetalBlitPreparation : MetalForwarder
@end
@interface MetalQueueRecorder : MetalForwarder
@end

@implementation MetalCapture
- (instancetype)init {
    self = [super init];
    dispatches = [NSMutableData new];
    reaches = [NSMutableData new];
    owners = [NSMutableSet new];
    resources = [NSMutableSet new];
    preparation = [NSMutableArray new];
    hosts = [NSMutableArray new];
    return self;
}
- (void)dealloc {
    [dispatches release]; [reaches release]; [owners release]; [resources release]; [preparation release]; [hosts release]; [super dealloc];
}
@end

@implementation MetalForwarder
- (instancetype)initWithTarget:(id)object { target = [object retain]; return self; }
- (void)dealloc { [target release]; [super dealloc]; }
- (NSMethodSignature *)methodSignatureForSelector:(SEL)selector {
    return [target methodSignatureForSelector:selector];
}
- (void)forwardInvocation:(NSInvocation *)invocation { [invocation invokeWithTarget:target]; }
- (BOOL)respondsToSelector:(SEL)selector { return [target respondsToSelector:selector]; }
@end

@implementation MetalDeviceRecorder
- (id<MTLCommandQueue>)newCommandQueue {
    id queue = [target newCommandQueue];
    id recorder = [[MetalQueueRecorder alloc] initWithTarget:queue];
    [queue release]; return recorder;
}
- (id<MTLCommandQueue>)newCommandQueueWithDescriptor:(MTLCommandQueueDescriptor *)descriptor {
    id queue = [target newCommandQueueWithDescriptor:descriptor];
    id recorder = [[MetalQueueRecorder alloc] initWithTarget:queue];
    [queue release]; return recorder;
}
- (id<MTLCommandQueue>)newCommandQueueWithMaxCommandBufferCount:(NSUInteger)count {
    id queue = [target newCommandQueueWithMaxCommandBufferCount:count];
    id recorder = [[MetalQueueRecorder alloc] initWithTarget:queue];
    [queue release]; return recorder;
}
- (id<MTLComputePipelineState>)newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
    options:(MTLPipelineOption)options reflection:(MTLComputePipelineReflection **)reflection error:(NSError **)error {
    return MetalCreatePipeline(target,descriptor,indirect,options,reflection,error);
}
- (id<MTLComputePipelineState>)newComputePipelineStateWithFunction:(id<MTLFunction>)function
    options:(MTLPipelineOption)options reflection:(MTLComputePipelineReflection **)reflection error:(NSError **)error {
    MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
    descriptor.computeFunction = function;
    id result = [self newComputePipelineStateWithDescriptor:descriptor options:options reflection:reflection error:error];
    [descriptor release]; return result;
}
- (id<MTLComputePipelineState>)newComputePipelineStateWithFunction:(id<MTLFunction>)function error:(NSError **)error {
    return [self newComputePipelineStateWithFunction:function options:MTLPipelineOptionNone reflection:NULL error:error];
}
// docs/placement.md#memory
- (id<MTLBuffer>)newBufferWithBytesNoCopy:(void *)pointer length:(NSUInteger)length options:(MTLResourceOptions)options
    deallocator:(void (^)(void *pointer, NSUInteger length))deallocator {
    return [target newBufferWithBytesNoCopy:pointer length:length options:options deallocator:deallocator];
}
@end

@implementation MetalComputeRecorder
- (void)setComputePipelineState:(id<MTLComputePipelineState>)pipeline {
    state.pipeline = pipeline;
    state.access = [objc_getAssociatedObject(pipeline, &pipelineAccessKey) unsignedLongLongValue];
}
- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
    // ../dox/mesh/design/prepared-machine.md#M06: a binding past the argument table fails with its index
    if (index >= MESH_BUFFER_BINDINGS)
        [NSException raise:NSInvalidArgumentException
            format:@"Buffer binding %lu exceeds Metal capacity %d", index, MESH_BUFFER_BINDINGS];
    state.buffers[index] = buffer; state.offsets[index] = offset;
    state.strides[index] = [objc_getAssociatedObject(buffer,&operandStrideKey) unsignedIntegerValue];
    capture->buffers = MAX(capture->buffers, index + 1);
    if (buffer) { [capture->resources addObject:buffer]; [capture->owners addObject:buffer]; }
}
- (void)setBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index {
    if (index >= MESH_BUFFER_BINDINGS)
        [NSException raise:NSInvalidArgumentException
            format:@"Buffer binding %lu exceeds Metal capacity %d", index, MESH_BUFFER_BINDINGS];
    state.offsets[index] = offset;
}
- (void)setBuffers:(const id<MTLBuffer> *)buffers offsets:(const NSUInteger *)offsets withRange:(NSRange)range {
    for (NSUInteger i = 0; i < range.length; ++i) [self setBuffer:buffers[i] offset:offsets[i] atIndex:range.location + i];
}
- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    id<MTLBuffer> buffer = [[target device] newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
    [self setBuffer:buffer offset:0 atIndex:index]; [buffer release];
}
- (void)setThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index {
    if (index >= MESH_MEMORY_BINDINGS)
        [NSException raise:NSInvalidArgumentException
            format:@"Threadgroup memory binding %lu exceeds Metal capacity %d", index, MESH_MEMORY_BINDINGS];
    state.memory[index] = length; capture->memory = MAX(capture->memory, index + 1);
}
// docs/kernels.md Any program: what a dispatch reaches beyond its bindings is declared with it (MetalDispatchReaching)
- (void)refuseUse {
    [NSException raise:NSInvalidArgumentException format:@"command %lu: a resource used by its encoder; a reach is declared with "
        "its dispatch (MetalDispatchReaching, docs/kernels.md Any program)", (unsigned long)(capture->dispatches.length / sizeof state)];
}
- (void)useResource:(id<MTLResource>)resource usage:(MTLResourceUsage)usage { (void)resource; (void)usage; [self refuseUse]; }
- (void)useResources:(const id<MTLResource> *)resources count:(NSUInteger)count usage:(MTLResourceUsage)usage {
    (void)resources; (void)count; (void)usage; [self refuseUse];
}
- (void)useHeap:(id<MTLHeap>)heap { (void)heap; [self refuseUse]; }
- (void)useHeaps:(const id<MTLHeap> *)heaps count:(NSUInteger)count { (void)heaps; (void)count; [self refuseUse]; }
- (void)dispatchThreadgroups:(MTLSize)grid threadsPerThreadgroup:(MTLSize)group {
    state.grid = grid; state.group = group; state.threads = NO; state.reach = 0; state.reaches = 0;
    [self recordDispatch];
}
- (void)dispatchThreads:(MTLSize)grid threadsPerThreadgroup:(MTLSize)group {
    state.grid = grid; state.group = group; state.threads = YES; state.reach = 0; state.reaches = 0;
    [self recordDispatch];
}
// docs/kernels.md Any program: the reaches kept with the capture (their buffers held resident by the replay)
- (void)dispatchGrid:(MTLSize)grid threadsPerThreadgroup:(MTLSize)group threadGrid:(BOOL)threads reaches:(const struct MetalReach *)reaches count:(size_t)count {
    state.grid = grid; state.group = group; state.threads = threads;
    state.reach = (uint32_t)(capture->reaches.length / sizeof *reaches); state.reaches = (uint32_t)count;
    if (count) [capture->reaches appendBytes:reaches length:count * sizeof *reaches];
    for (size_t k = 0; k < count; k++) { [capture->resources addObject:reaches[k].buffer]; [capture->owners addObject:reaches[k].buffer]; }
    [self recordDispatch];
}
// ../dox/mesh/design/prepared-machine.md#M06
// docs/kernels.md Any program: a pipeline of the recording device, whose reflection shows what its
// bindings read and write; one whose arguments hold addresses declares what they reach, and is
// refused where it declares nothing. Binding reaches describe the named argument's
// access ranges, including holes and aliases. Threadgroup memory the pipeline does not declare is not bound. A pipeline the
// recording device did not make is refused.
- (void)recordDispatch {
    NSNumber *declared = objc_getAssociatedObject(state.pipeline, &pipelineBindingsKey);
    const unsigned long index = (unsigned long)(capture->dispatches.length / sizeof state);
    NSString *name = state.pipeline.label ?: @"unnamed";
    if (!declared)
        [NSException raise:NSInvalidArgumentException format:@"command %lu (%@): a pipeline the recording device did not make "
            "(a crossing ends a segment: mesh_rank.h MeshRankCross)", index, name];
    if (declared && !state.reaches && objc_getAssociatedObject(state.pipeline, &pipelineAddressesKey))
        [NSException raise:NSInvalidArgumentException format:@"command %lu (%@): its arguments hold addresses and it declares no reach "
            "(MetalDispatchReaching, docs/kernels.md Any program)", index, name];
    state.extents = 0;
    const struct MetalReach *reaches = (const struct MetalReach *)capture->reaches.bytes + state.reach;
    for (uint32_t k = 0; k < state.reaches; k++) {
        const struct MetalReach *reach = reaches + k;
        const NSUInteger slots = [objc_getAssociatedObject(reach->buffer, &operandInstancesKey) unsignedIntegerValue],
            stride = [objc_getAssociatedObject(reach->buffer, &operandStrideKey) unsignedIntegerValue];
        if (!(reach->access & (MetalReachRead | MetalReachWritten)) || reach->offset + reach->length > reach->buffer.length || (stride && slots != 1 && !(reach->access & MetalReachBinding)))
            [NSException raise:NSInvalidArgumentException format:@"command %lu (%@): reach %u (%@, %lu bytes at %lu of %lu, access %u) is "
                "no access, past its buffer, or into an operand's slots (%lu), which an address cannot follow (docs/kernels.md Any program)",
                index, name, k, reach->buffer.label ?: @"unnamed", (unsigned long)reach->length, (unsigned long)reach->offset,
                (unsigned long)reach->buffer.length, reach->access, (unsigned long)slots];
        if (!(reach->access & MetalReachBinding)) continue;
        if (reach->binding < 0 || reach->binding >= MESH_BUFFER_BINDINGS || state.buffers[reach->binding] != reach->buffer)
            [NSException raise:NSInvalidArgumentException format:@"command %lu (%@): reach %u names no matching buffer binding", index, name, k];
        state.extents |= UINT32_C(1) << reach->binding;
    }
    struct MetalRecordedDispatch recorded = state;
    for(NSUInteger j=0;j<MESH_BUFFER_BINDINGS;j++) {
        recorded.lengths[j]=0;
        for(size_t v=0;v<bindingViewCount;v++) {
            const struct MetalBindingView *view=bindingViews+v;
            if(view->buffer!=state.buffers[j]||state.offsets[j]<view->offset||state.offsets[j]>=view->offset+view->length)continue;
            recorded.lengths[j]=MAX(recorded.lengths[j],view->offset+view->length-state.offsets[j]);
        }
    }
    for (NSUInteger i = 0; i < MESH_MEMORY_BINDINGS; i++)
        if (!(declared.unsignedLongLongValue & (UINT64_C(1) << (MESH_ACCESS_WRITE_SHIFT + i)))) recorded.memory[i] = 0;
    [capture->owners addObject:state.pipeline];
    [capture->dispatches appendBytes:&recorded length:sizeof recorded];
}
- (void)forwardInvocation:(NSInvocation *)invocation {
    // a selector without arguments encodes nothing (a query, endEncoding, popDebugGroup); the
    // dataflow orders the recording, so a scope barrier adds no edge
    NSString *name = NSStringFromSelector(invocation.selector);
    if ([name containsString:@":"] && ![name isEqualToString:@"setLabel:"] &&
        ![name isEqualToString:@"pushDebugGroup:"] && ![name isEqualToString:@"memoryBarrierWithScope:"]) {
        [NSException raise:NSInvalidArgumentException format:@"Unrecorded compute command %@", name];
    }
    [invocation invokeWithTarget:target];
}
@end

@implementation MetalBlitPreparation
- (void)fillBuffer:(id<MTLBuffer>)buffer range:(NSRange)range value:(uint8_t)value {
    [target fillBuffer:buffer range:range value:value];
}
- (void)forwardInvocation:(NSInvocation *)invocation {
    NSString *name = NSStringFromSelector(invocation.selector);
    if (![name isEqualToString:@"endEncoding"] && ![name isEqualToString:@"setLabel:"] &&
        ![name isEqualToString:@"pushDebugGroup:"] && ![name isEqualToString:@"popDebugGroup"])
        [NSException raise:NSInvalidArgumentException format:@"Unrecorded blit command %@", name];
    [invocation invokeWithTarget:target];
}
@end

@implementation MetalCommandRecorder
- (void)commit {
    if (activeCapture) [activeCapture->preparation addObject:target];
    [target commit];
}
- (id<MTLBlitCommandEncoder>)blitCommandEncoder {
    id encoder = [target blitCommandEncoder];
    if (!activeCapture) return encoder;
    return (id)[[[MetalBlitPreparation alloc] initWithTarget:encoder] autorelease];
}
- (void)encodeSignalEvent:(id<MTLEvent>)event value:(uint64_t)value {
    [target encodeSignalEvent:event value:value];
}
- (id<MTLComputeCommandEncoder>)computeCommandEncoder {
    if (!activeCapture) return [target computeCommandEncoder];
    MetalComputeRecorder *encoder = [[MetalComputeRecorder alloc] initWithTarget:[target computeCommandEncoder]];
    encoder->capture = activeCapture;
    return (id)[encoder autorelease];
}
- (id<MTLComputeCommandEncoder>)computeCommandEncoderWithDescriptor:(MTLComputePassDescriptor *)descriptor {
    id encoder = [target computeCommandEncoderWithDescriptor:descriptor];
    if (!activeCapture) return encoder;
    MetalComputeRecorder *recorder = [[MetalComputeRecorder alloc] initWithTarget:encoder];
    recorder->capture = activeCapture;
    return (id)[recorder autorelease];
}
// a serial or concurrent encoder records alike: the recording's order is its dataflow
- (id<MTLComputeCommandEncoder>)computeCommandEncoderWithDispatchType:(MTLDispatchType)type {
    id encoder = [target computeCommandEncoderWithDispatchType:type];
    if (!activeCapture) return encoder;
    MetalComputeRecorder *recorder = [[MetalComputeRecorder alloc] initWithTarget:encoder];
    recorder->capture = activeCapture;
    return (id)[recorder autorelease];
}
- (void)forwardInvocation:(NSInvocation *)invocation {
    NSString *name = NSStringFromSelector(invocation.selector);
    if (activeCapture && ([name hasPrefix:@"encode"] || [name containsString:@"CommandEncoder"]))
        [NSException raise:NSInvalidArgumentException format:@"Unrecorded command %@", name];
    [invocation invokeWithTarget:target];
}
@end

@implementation MetalQueueRecorder
- (id<MTLCommandBuffer>)commandBuffer {
    id command = [target commandBuffer];
    MetalCommandRecorder *recorder = [[MetalCommandRecorder alloc] initWithTarget:command];
    return (id)[recorder autorelease];
}
- (id<MTLCommandBuffer>)commandBufferWithUnretainedReferences {
    id command = [target commandBufferWithUnretainedReferences];
    MetalCommandRecorder *recorder = [[MetalCommandRecorder alloc] initWithTarget:command];
    return (id)[recorder autorelease];
}
- (id<MTLCommandBuffer>)commandBufferWithDescriptor:(MTLCommandBufferDescriptor *)descriptor {
    id command = [target commandBufferWithDescriptor:descriptor];
    MetalCommandRecorder *recorder = [[MetalCommandRecorder alloc] initWithTarget:command];
    return (id)[recorder autorelease];
}
@end

// ../dox/mesh/design/prepared-machine.md#M06
// A recording's replay: its indirect command buffers holding every repetition's commands (the
// one modulus), their barriers, and the residency set.
static void MetalReplayBuild(MetalCapture *capture,id<MTLDevice> device,NSUInteger repetitions,enum MetalForm form,
    struct MetalReplay *result) {
    result->count = (uint32_t)(capture->dispatches.length / sizeof(struct MetalRecordedDispatch));
    if (!result->count) [NSException raise:NSInvalidArgumentException format:@"No compute commands recorded"];
    [capture->owners unionSet:capture->resources];
    result->buffers = (uint16_t)capture->buffers; result->memory = (uint16_t)capture->memory;
    const struct MetalRecordedDispatch *commands = capture->dispatches.bytes;
    NSMutableData *barriers=[NSMutableData dataWithLength:form==MetalFormIndirect?(NSUInteger)result->count*repetitions:0];
    uint8_t *ordered=barriers.mutableBytes;
    NSMutableDictionary *written=[NSMutableDictionary dictionary],*read=[NSMutableDictionary dictionary];
    NSUInteger phase=0;
    for(NSUInteger t=0;form==MetalFormIndirect&&t<repetitions;t++) @autoreleasepool {
        for(NSUInteger i=0;i<result->count;i++)ordered[t*result->count+i]=MetalOrder(read,written,&phase,commands+i,capture->reaches.bytes,t);
    }
    result->owner = [capture->owners retain];
    objc_setAssociatedObject(result->owner,&replayHostsKey,[NSArray arrayWithArray:capture->hosts],OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    NSMutableData *resources=[NSMutableData dataWithLength:capture->resources.count*sizeof(id)];
    id __unsafe_unretained *resourceList=resources.mutableBytes;NSUInteger resourceIndex=0;
    for(id resource in capture->resources)resourceList[resourceIndex++]=resource;
    objc_setAssociatedObject(result->owner,&replayResourcesKey,resources,OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    // docs/kernels.md E2: the records a direct replay encodes (MetalReplayDispatch4)
    objc_setAssociatedObject(result->owner, &replayRecordsKey, @[[NSData dataWithBytes:commands length:capture->dispatches.length],barriers,@(form),[NSData dataWithData:capture->reaches],@(repetitions)],
        OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    MTLResidencySetDescriptor *residency = [MTLResidencySetDescriptor new];
    residency.initialCapacity = capture->owners.count + 1;
    NSError *error = nil;
    result->residency = [device newResidencySetWithDescriptor:residency error:&error];
    [residency release];
    if (!result->residency) [NSException raise:NSMallocException format:@"%@", error];
    for (id allocation in capture->owners) [result->residency addAllocation:allocation];
    if (form != MetalFormIndirect) { [result->residency commit]; [result->residency requestResidency]; return; }
    MTLIndirectCommandBufferDescriptor *descriptor = [MTLIndirectCommandBufferDescriptor new];
    descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatch | MTLIndirectCommandTypeConcurrentDispatchThreads;
    descriptor.inheritBuffers = NO; descriptor.inheritPipelineState = NO;
    descriptor.maxKernelBufferBindCount = capture->buffers;
    descriptor.maxKernelThreadgroupMemoryBindCount = capture->memory;
    // One indirect command buffer holds at most MESH_INDIRECT_COMMANDS commands, so the
    // repetitions fill as many buffers as they need, `span` repetitions each, executed in order
    // by MetalReplayExecute.
    NSUInteger ceiling = MESH_INDIRECT_COMMANDS;
    NSUInteger span = MAX((NSUInteger)1, ceiling / result->count);
    if (result->count > ceiling) [NSException raise:NSInvalidArgumentException format:@"%u commands exceed one indirect buffer's %lu", result->count, (unsigned long)ceiling];
    NSMutableArray *segments = [NSMutableArray array];
    for (NSUInteger first = 0; first < repetitions; first += span) {
        id<MTLIndirectCommandBuffer> segment = [device newIndirectCommandBufferWithDescriptor:descriptor
            maxCommandCount:result->count*MIN(span, repetitions-first) options:0];
        if (!segment) [NSException raise:NSMallocException format:@"indirect command buffer of %lu commands",
            (unsigned long)(result->count*MIN(span, repetitions-first))];
        [segments addObject:segment]; [segment release];
    }
    result->commands = [segments[0] retain];
    objc_setAssociatedObject(result->commands, &replaySegmentsKey, @[segments, @(span*result->count)], OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    [descriptor release];
    for (NSUInteger t = 0; t < repetitions; t++) for (uint32_t i = 0; i < result->count; ++i) {
        const struct MetalRecordedDispatch *source = commands + i;
        if (!source->pipeline.supportIndirectCommandBuffers)
            [NSException raise:NSInvalidArgumentException format:@"command %u (%@) does not support indirect commands: the encoded form runs it (docs/kernels.md Any program)",
                i, source->pipeline.label ?: @"unnamed"];
        id<MTLIndirectComputeCommand> command = [segments[t / span] indirectComputeCommandAtIndex:(t % span)*result->count+i];
        [command setComputePipelineState:source->pipeline];
        // ../dox/mesh/design/prepared-machine.md#M13: the one modulus.  Invocation t binds slot
        // (t % instances), and instances == repetitions is a bare t.
        for (NSUInteger j = 0; j < capture->buffers; ++j)
            if (((source->access | (source->access >> MESH_ACCESS_WRITE_SHIFT)) >> j) & 1) {
                [command setKernelBuffer:source->buffers[j]
                    offset:MetalInvocationOffset(source->offsets[j],source->strides[j],source->instances[j],t) atIndex:j];
            }
        for (NSUInteger j = 0; j < capture->memory; ++j)
            if (source->memory[j]) [command setThreadgroupMemoryLength:source->memory[j] atIndex:j];
        // a segment's first command waits on the segment before it
        if (ordered[t*result->count+i]) [command setBarrier];
        if (source->threads) [command concurrentDispatchThreads:source->grid threadsPerThreadgroup:source->group];
        else [command concurrentDispatchThreadgroups:source->grid threadsPerThreadgroup:source->group];
    }
    for (id<MTLIndirectCommandBuffer> segment in segments) [result->residency addAllocation:segment];
    [result->residency commit];
    [result->residency requestResidency];
}

// docs/kernels.md Any program: one recorded command's line: its level (barriers before it), entry,
// grid and group (`t` a thread grid), threadgroup memory, the bytes of its accesses (MetalAccesses:
// its bindings, then its reaches), and the named operands.
static NSString *MetalCommandLine(const struct MetalRecordedDispatch *command,const struct MetalReach *reaches,NSUInteger i,NSUInteger level,
    NSMutableDictionary *allocations) {
    MTLComputePipelineDescriptor *descriptor=objc_getAssociatedObject(command->pipeline,&pipelineDescriptorKey);
    NSUInteger memory=command->pipeline.staticThreadgroupMemoryLength;
    __block NSUInteger read=0,written=0;
    for(NSUInteger j=0;j<MESH_MEMORY_BINDINGS;j++)memory+=command->memory[j];
    NSMutableString *names=[NSMutableString string],*bindings=[NSMutableString string];
    MetalAccesses(command,reaches,0,^(id<MTLBuffer> buffer,NSUInteger offset,NSUInteger bytes,BOOL load,BOOL store) {
        if(load)read+=bytes;
        if(store)written+=bytes;
        if(buffer.label.length)[names appendFormat:@" %@:%s%s",buffer.label,load?"r":"-",store?"w":"-"];
        NSValue *key=MetalAllocationKey(buffer);
        if(!allocations[key])allocations[key]=@(allocations.count);
        [bindings appendFormat:@" @%lu+%lu:%lu:%s%s",[allocations[key] unsignedLongValue],(unsigned long)(MetalAllocationBase(buffer)+offset),
            (unsigned long)bytes,load?"r":"-",store?"w":"-"];
    });
    return [NSString stringWithFormat:@"command %lu level %lu %s grid %lu,%lu,%lu%s group %lu,%lu,%lu memory %lu read %lu written %lu%s%s%s",
        i,level,descriptor.computeFunction.name.UTF8String?:"?",command->grid.width,command->grid.height,command->grid.depth,
        command->threads?"t":"",command->group.width,command->group.height,command->group.depth,memory,read,written,
        "",bindings.UTF8String,names.UTF8String];
}

// tools/mesh/schedule.py: whether the recordings that follow list their commands (a record-only call's), beside RECORD_ONLY
static _Atomic int listing;
void MetalRecordingList(int on) { atomic_store(&listing,on); }


// ../dox/mesh/design/prepared-machine.md#M06
// Records the compute commands `encode` issues through queues of the recording `device` as one
// repetition: the dataflow's barriers, each operand's slots and output offset (the slot count is
// the buffer's only where the binding's stride is the buffer's own), the replay built in `form`;
// the program's census line, and under RECORD_ONLY one line a command.
static void MetalRecordReplay(id<MTLDevice> device,void (^encode)(void),NSUInteger repetitions,enum MetalForm form,struct MetalReplay *result) {
    MetalCapture *capture = [MetalCapture new];
    @try {
        activeCapture = capture;
        @try { encode(); } @finally { activeCapture = nil; }
        for (id<MTLCommandBuffer> command in capture->preparation) {
            [command waitUntilCompleted];
            if (command.error) [NSException raise:NSInternalInconsistencyException format:@"%@", command.error];
        }
        struct MetalRecordedDispatch *prepared=capture->dispatches.mutableBytes;
        const NSUInteger recorded=capture->dispatches.length/sizeof(*prepared);
        for(NSUInteger i=0;i<recorded;i++)for(NSUInteger j=0;j<capture->buffers;j++) {
            if(prepared[i].access&(UINT64_C(1)<<(MESH_ACCESS_WRITE_SHIFT+j))) {
                NSUInteger shift=[objc_getAssociatedObject(prepared[i].buffers[j],&operandOutputOffsetKey) unsignedIntegerValue];
                struct MetalReach *reaches=capture->reaches.mutableBytes;
                for(uint32_t k=0;k<prepared[i].reaches;k++) {
                    struct MetalReach *reach=reaches+prepared[i].reach+k;
                    if((reach->access&MetalReachBinding)&&reach->binding==(int32_t)j)
                        reach->offset+=shift;
                }
                prepared[i].offsets[j]+=shift;
            }
            NSUInteger own=[objc_getAssociatedObject(prepared[i].buffers[j],&operandStrideKey) unsignedIntegerValue];
            prepared[i].instances[j]=prepared[i].strides[j]==own?
                [objc_getAssociatedObject(prepared[i].buffers[j],&operandInstancesKey) unsignedIntegerValue]:0;
        }
        MetalDataflow(capture);
        MetalReplayBuild(capture, ((MetalForwarder *)device)->target, repetitions, form, result);
        NSUInteger levels=1;
        for(NSUInteger i=1;i<recorded;i++)levels+=prepared[i].barrier;
        printf("prepared program commands %lu levels %lu form %s\n",recorded,levels,(const char *[]){"icb","encoded","serial"}[form]);
        NSMutableDictionary *allocations=[NSMutableDictionary dictionary];
        if(getenv("RECORD_ONLY")||atomic_load(&listing))for(NSUInteger i=0,level=0;i<recorded;i++) @autoreleasepool {
            level+=i&&prepared[i].barrier;
            printf("prepared %s\n",MetalCommandLine(prepared+i,capture->reaches.bytes,i,level,allocations).UTF8String);
        }
    } @catch (NSException *error) {
        MetalReplayDestroy(result);
        @throw;
    } @finally {
        [capture release];
    }
}

static void MetalReplaySegments(struct MetalReplay *replay, NSRange range, void (^each)(id<MTLIndirectCommandBuffer>, NSRange)) {
    if (!replay->commands) return;
    NSArray *held = objc_getAssociatedObject(replay->commands, &replaySegmentsKey);
    NSArray *segments = held ? held[0] : @[replay->commands];
    NSUInteger size = held ? [held[1] unsignedIntegerValue] : NSUIntegerMax;
    for (NSUInteger at = range.location, end = NSMaxRange(range); at < end;) {
        NSUInteger index = at / size, local = at % size, length = MIN(end - at, size - local);
        each(segments[index], NSMakeRange(local, length));
        at += length;
    }
}

// The same recorded operands bind a direct Metal 4 dispatch during function preparation.
static void MetalReplayDispatch4(id<MTL4ComputeCommandEncoder> encoder,id<MTL4ArgumentTable> table,
    struct MetalReplay *replay,const struct MetalRecordedDispatch *source,NSUInteger t) {
    [encoder setArgumentTable:table];[encoder setComputePipelineState:source->pipeline];
    for(NSUInteger j=0;j<replay->buffers;j++)if(((source->access|(source->access>> MESH_ACCESS_WRITE_SHIFT))>>j)&1) {
        [table setAddress:source->buffers[j].gpuAddress+MetalInvocationOffset(source->offsets[j],source->strides[j],source->instances[j],t) atIndex:j];
    }
    for(NSUInteger j=0;j<replay->memory;j++)if(source->memory[j])[encoder setThreadgroupMemoryLength:source->memory[j] atIndex:j];
    if(source->threads)[encoder dispatchThreads:source->grid threadsPerThreadgroup:source->group];
    else [encoder dispatchThreadgroups:source->grid threadsPerThreadgroup:source->group];
}

// ../dox/mesh/design/prepared-machine.md#M13
static void MetalReplayDispatch(id<MTLComputeCommandEncoder> encoder, struct MetalReplay *replay,
    const struct MetalRecordedDispatch *source, NSUInteger t, const struct MetalReach *reaches) {
    [encoder setComputePipelineState:source->pipeline];
    for (NSUInteger j = 0; j < replay->buffers; j++)
        if (((source->access | (source->access >> MESH_ACCESS_WRITE_SHIFT)) >> j) & 1) {
            [encoder setBuffer:source->buffers[j] offset:MetalInvocationOffset(source->offsets[j],source->strides[j],source->instances[j],t) atIndex:j];
        }
    for (NSUInteger j = 0; j < replay->memory; j++)
        if (source->memory[j]) [encoder setThreadgroupMemoryLength:source->memory[j] atIndex:j];
    MetalDispatchAccesses(encoder,source->grid,source->group,source->threads,
        reaches?reaches+source->reach:NULL,reaches?source->reaches:0);
}

struct MetalReplay *MetalReplayCommands(id<MTLDevice> device,const struct MetalCommandDescription *commands,size_t count,enum MetalForm form) {
    if(object_getClass(device)==[MetalDeviceRecorder class])device=((MetalForwarder *)device)->target;
    MetalCapture *capture=[MetalCapture new];struct MetalReplay *replay=NULL;
    @try {
        for(size_t i=0;i<count;i++) {
            const struct MetalCommandDescription *description=commands+i;
            struct MetalRecordedDispatch command={.pipeline=description->pipeline,.grid=description->groups,.group=description->threads,.threads=description->threadGrid,
                .reach=(uint32_t)(capture->reaches.length/sizeof(struct MetalReach)),.reaches=(uint32_t)description->reachCount};
            [capture->owners addObject:command.pipeline];
            for(size_t k=0;k<description->bindingCount;k++) {
                const struct MetalCommandBinding binding=description->bindings[k];
                if(binding.index>=MESH_BUFFER_BINDINGS)[NSException raise:NSInvalidArgumentException format:@"binding index exceeds Metal capacity"];
                const uint32_t at=binding.index;
                command.buffers[at]=binding.buffer;command.offsets[at]=binding.offset;command.lengths[at]=binding.length;
                command.access|=(uint64_t)(binding.access&MetalReachRead?1:0)<<at;
                command.access|=(uint64_t)(binding.access&MetalReachWritten?1:0)<<(MESH_ACCESS_WRITE_SHIFT+at);
                capture->buffers=MAX(capture->buffers,at+1);
                [capture->resources addObject:binding.buffer];
            }
            if(description->memoryCount>MESH_MEMORY_BINDINGS)[NSException raise:NSInvalidArgumentException format:@"threadgroup-memory index exceeds Metal capacity"];
            for(size_t k=0;k<description->memoryCount;k++)command.memory[k]=description->memory[k];
            capture->memory=MAX(capture->memory,description->memoryCount);
            [capture->reaches appendBytes:description->reaches length:description->reachCount*sizeof(struct MetalReach)];
            for(size_t k=0;k<description->reachCount;k++) {
                const struct MetalReach reach=description->reaches[k];
                [capture->resources addObject:reach.buffer];
                if(reach.access&MetalReachBinding) {
                    if(reach.binding<0||reach.binding>=MESH_BUFFER_BINDINGS||command.buffers[reach.binding]!=reach.buffer)
                        [NSException raise:NSInvalidArgumentException format:@"command reach names no matching buffer binding"];
                    command.extents|=UINT32_C(1)<<reach.binding;
                }
            }
            [capture->dispatches appendBytes:&command length:sizeof command];
        }
        MetalDataflow(capture);
        if(posix_memalign((void **)&replay,_Alignof(struct MetalReplay),sizeof *replay))
            [NSException raise:NSMallocException format:@"allocating replay"];
        memset(replay,0,sizeof *replay);
        MetalReplayBuild(capture,device,1,form,replay);
    } @catch(NSException *error) {
        if(replay){MetalReplayDestroy(replay);free(replay);replay=NULL;}
        fprintf(stderr,"record: %s\n",error.reason.UTF8String);
    } @finally { [capture release]; }
    return replay;
}

void MetalReplayEncode(struct MetalReplay *replay,id<MTLCommandBuffer> command,NSRange range) {
    NSArray *held=objc_getAssociatedObject(replay->owner,&replayRecordsKey);
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    if(activeCapture || [held[2] unsignedIntegerValue]!=MetalFormIndirect) {
        const struct MetalRecordedDispatch *records=[held[0] bytes];
        const struct MetalReach *reaches=[held[3] bytes];
        for(NSUInteger i=range.location;i<NSMaxRange(range);i++)MetalReplayDispatch(encoder,replay,records+i,0,reaches);
    } else {
        NSData *resources=objc_getAssociatedObject(replay->owner,&replayResourcesKey);
        [encoder useResources:resources.bytes count:resources.length/sizeof(id) usage:MTLResourceUsageRead|MTLResourceUsageWrite];
        MetalReplaySegments(replay,range,^(id<MTLIndirectCommandBuffer> segment,NSRange local){
            [encoder executeCommandsInBuffer:segment withRange:local];
        });
    }
    [encoder endEncoding];
}

// docs/kernels.md#collectives
static NSError *MetalReplayFailure(NSInteger code, NSString *reason) {
    return [NSError errorWithDomain:@"MetalReplay" code:code userInfo:@{NSLocalizedDescriptionKey:reason}];
}

// One prepared function owns its command stream. Append and Finish run during preparation;
// Enqueue only commits the finished stream. The observer never controls subsequent work.
struct MetalSchedule {
    struct MetalSchedule *next, *tail, *root;
    id<MTLDevice> device;
    enum MetalForm form;
    BOOL finished;
    id queue, command, encoder;
    id<MTL4CommandAllocator> allocator;
    id<MTL4ArgumentTable> table;
    id<MTLResidencySet> residency;
    MTL4CommitOptions *options;
    NSMutableSet *owners;
    NSMutableOrderedSet *communicators;
    NSMutableDictionary *read, *written;
    NSUInteger phase;
    id<MTLEvent> after;
    uint64_t value;
    id<MTLEvent> signal;
    uint64_t signalValue;
    _Atomic int status;
    _Atomic unsigned references;
    _Atomic unsigned pending;
    NSError *error;
    void (^completed)(NSError *);
    id<MTLBuffer> data;
    NSUInteger used;
    NSError *(^host)(void);
};

MTLCompileOptions *MetalDeviceCompilerOptions(id<MTLDevice> device) {
    MTLCompileOptions *options=[MTLCompileOptions new];
    options.languageVersion=MESH_METAL_LANGUAGE;
    options.preprocessorMacros=@{
        @"MESH_DEVICE_THREADGROUP_THREADS": @(device.maxThreadsPerThreadgroup.width),
        @"MESH_DEVICE_THREADGROUP_BYTES": @(device.maxThreadgroupMemoryLength),
#define MESH_METAL_VALUE(name, value) @"MESH_" #name: @(value),
#include "../etc/compute/metal.def"
#undef MESH_METAL_VALUE
    };
    return [options autorelease];
}

static struct MetalSchedule *MetalScheduleTail(struct MetalSchedule *s) { return s->tail?:s; }

struct MetalSchedule *MetalScheduleCreateOn(id queue, enum MetalForm form) {
    if(!queue)return NULL;
    id<MTLDevice> device=[queue device];
    struct MetalSchedule *s=calloc(1,sizeof *s);
    if(!s)return NULL;
    atomic_init(&s->references,1);
    atomic_init(&s->pending,1);s->root=s;
    atomic_init(&s->status,0);
    s->device=[device retain];s->form=form;s->queue=[queue retain];
    s->owners=[NSMutableSet new];s->read=[NSMutableDictionary new];s->written=[NSMutableDictionary new];
    MTLResidencySetDescriptor *resident=[MTLResidencySetDescriptor new];
    s->residency=[device newResidencySetWithDescriptor:resident error:nil];[resident release];
    if(form==MetalFormSerial) {
        s->command=[[(id<MTLCommandQueue>)s->queue commandBufferWithUnretainedReferences] retain];
        // The serial encoder is opened at the first append, after the external input event.
    } else {
        s->allocator=[device newCommandAllocator];
        MTL4ArgumentTableDescriptor *descriptor=[MTL4ArgumentTableDescriptor new];descriptor.maxBufferBindCount=MESH_BUFFER_BINDINGS;
        s->table=[device newArgumentTableWithDescriptor:descriptor error:nil];[descriptor release];
        s->command=[device newCommandBuffer];
        if(s->command&&s->allocator) {
            [(id<MTL4CommandBuffer>)s->command beginCommandBufferWithAllocator:s->allocator];
            s->encoder=[[(id<MTL4CommandBuffer>)s->command computeCommandEncoder] retain];
            MetalQueueAccess *previous=objc_getAssociatedObject(queue,&queueAccessKey);
            if(!previous) {
                previous=[MetalQueueAccess new];
                objc_setAssociatedObject(queue,&queueAccessKey,previous,OBJC_ASSOCIATION_RETAIN_NONATOMIC);[previous release];
            }
            MetalQueueAccess *current=[MetalQueueAccess new];
            objc_setAssociatedObject(s->encoder,&encoderAccessKey,@[previous,current],OBJC_ASSOCIATION_RETAIN_NONATOMIC);
            [current release];
        }
    }
    if(!s->queue||!s->command||!s->residency||(form!=MetalFormSerial&&(!s->allocator||!s->table||!s->encoder))) {
        MetalScheduleFree(s);return NULL;
    }
    return s;
}

struct MetalSchedule *MetalScheduleCommand(id<MTLCommandBuffer> command) {
    struct MetalSchedule *s=calloc(1,sizeof *s);
    if(!s)return NULL;
    atomic_init(&s->references,1);atomic_init(&s->status,0);atomic_init(&s->pending,1);s->root=s;
    s->device=[command.device retain];s->queue=[command.commandQueue retain];
    s->command=[command retain];s->form=MetalFormSerial;
    return s;
}

void MetalScheduleAfter(struct MetalSchedule *s,id<MTLEvent> after,uint64_t value) {
    s=MetalScheduleTail(s);
    s->after=[after retain];s->value=value;
    if(s->form==MetalFormSerial&&after)[(id<MTLCommandBuffer>)s->command encodeWaitForEvent:after value:value];
}

void MetalScheduleCommunicator(struct MetalSchedule *s,struct ncclComm *comm) {
    s=s->root;
    if(!s->communicators)s->communicators=[NSMutableOrderedSet new];
    [s->communicators addObject:[NSValue valueWithPointer:comm]];
}

id<MTLResidencySet> MetalScheduleResources(struct MetalSchedule *s) { return MetalScheduleTail(s)->residency; }

struct MetalCommandBinding MetalScheduleData(struct MetalSchedule *s,const void *data,size_t bytes,uint32_t index) {
    s=MetalScheduleTail(s);
    NSUInteger size=(bytes+MESH_COMMAND_DATA_ALIGNMENT-1)/MESH_COMMAND_DATA_ALIGNMENT*MESH_COMMAND_DATA_ALIGNMENT;
    if(!s->data || s->used+size>s->data.length) {
        [s->data release];s->data=[s->device newBufferWithLength:MAX((NSUInteger)MESH_COMMAND_DATA_BYTES,size) options:MTLResourceStorageModeShared];
        s->used=0;[s->owners addObject:s->data];[s->residency addAllocation:s->data];
    }
    struct MetalCommandBinding result={s->data,s->used,bytes,index,MetalReachRead};
    memcpy((char *)s->data.contents+s->used,data,bytes);s->used+=size;
    return result;
}

void MetalScheduleCompute(struct MetalSchedule *s,id<MTLComputePipelineState> pipeline,
    const struct MetalCommandBinding *bindings,size_t count,MTLSize groups,MTLSize threads) {
    s=MetalScheduleTail(s);
    struct MetalRecordedDispatch command={.pipeline=pipeline,.grid=groups,.group=threads};
    struct MetalReplay view={0};
    [s->owners addObject:pipeline];
    for(size_t k=0;k<count;k++) {
        const struct MetalCommandBinding b=bindings[k];
        if(b.index>=MESH_BUFFER_BINDINGS)[NSException raise:NSInvalidArgumentException format:@"binding index exceeds Metal capacity"];
        command.buffers[b.index]=b.buffer;command.offsets[b.index]=b.offset;command.lengths[b.index]=b.length;
        command.access|=(uint64_t)(b.access&MetalReachRead?1:0)<<b.index;
        command.access|=(uint64_t)(b.access&MetalReachWritten?1:0)<<(MESH_ACCESS_WRITE_SHIFT+b.index);
        view.buffers=MAX(view.buffers,b.index+1);
        [s->owners addObject:b.buffer];
        [s->residency addAllocation:b.buffer];
    }
    BOOL barrier=MetalOrder(s->read,s->written,&s->phase,&command,NULL,0);
    if(s->form==MetalFormSerial) {
        if(!s->encoder)s->encoder=[[(id<MTLCommandBuffer>)s->command computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial] retain];
        MetalReplayDispatch(s->encoder,&view,&command,0,NULL);
    } else {
        MetalEncoderAccesses(s->encoder,&command,NULL,0);
        if(barrier)[(id<MTL4ComputeCommandEncoder>)s->encoder barrierAfterEncoderStages:MTLStageDispatch
            beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        MetalReplayDispatch4(s->encoder,s->table,&view,&command,0);
    }
}

int MetalScheduleCross(struct MetalSchedule *s,MetalCrossing cross) {
    s=MetalScheduleTail(s);
    int error=cross(s->encoder,s->table,s->residency,0,0);
    [s->read removeAllObjects];[s->written removeAllObjects];
    return error;
}

static void MetalScheduleFinishPart(struct MetalSchedule *,void (^completed)(NSError *));

static struct MetalSchedule *MetalScheduleAdvance(struct MetalSchedule *root,
    NSError *(^operation)(void),id<MTLSharedEvent> signal) {
    struct MetalSchedule *before=MetalScheduleTail(root),*next=MetalScheduleCreateOn(before->queue,before->form);
    if(!next)[NSException raise:NSMallocException format:@"preparing host-share continuation"];
    next->root=root;atomic_fetch_add_explicit(&root->pending,1,memory_order_relaxed);
    before->signal=[signal retain];before->signalValue=signal?1:0;
    before->host=[operation copy];
    MetalScheduleFinishPart(before,nil);
    before->next=next;root->tail=next;
    return next;
}

id<MTLSharedEvent> MetalScheduleHostProducer(struct MetalSchedule *root,NSError *(^operation)(void)) {
    id<MTLSharedEvent> event=[root->device newSharedEvent];
    if(!event)[NSException raise:NSMallocException format:@"preparing host-share output event"];
    MetalScheduleAdvance(root,operation,event);
    return event;
}

void MetalScheduleHostConsumer(struct MetalSchedule *root,id<MTLSharedEvent> event) {
    MetalScheduleAfter(MetalScheduleAdvance(root,nil,nil),event,1);
}

static void MetalScheduleRange(struct MetalSchedule *root,struct MetalReplay *replay,NSUInteger t,NSUInteger from,NSUInteger to) {
    struct MetalSchedule *s=MetalScheduleTail(root);
    if(!replay||from==to)return;
    NSArray *held=objc_getAssociatedObject(replay->owner,&replayRecordsKey);
    const struct MetalRecordedDispatch *records=[held[0] bytes];
    const struct MetalReach *reaches=[held[3] bytes];
    const uint8_t *barriers=[held[1] bytes];
    const NSUInteger n=replay->count;
    [s->owners addObject:replay->owner];
    for(id<MTLAllocation> allocation in replay->residency.allAllocations)[s->residency addAllocation:allocation];
    if(s->form==MetalFormSerial&&!s->encoder)
        s->encoder=[[(id<MTLCommandBuffer>)s->command computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial] retain];
    id<MTL4ComputeCommandEncoder> encoder=s->encoder;
    NSUInteger pending=t*n+from;
    for(NSUInteger c=pending;c<t*n+to;c++) @autoreleasepool {
        const struct MetalRecordedDispatch *source=records+c%n;
        if(s->form!=MetalFormSerial)MetalEncoderAccesses(encoder,source,reaches,t);
        BOOL barrier=MetalOrder(s->read,s->written,&s->phase,source,reaches,t);
        if(s->form==MetalFormSerial)MetalReplayDispatch(s->encoder,replay,source,t,NULL);
        else if(s->form==MetalFormEncoded) {
            if(barrier)[encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
            MetalReplayDispatch4(encoder,s->table,replay,source,t);
        } else if(barrier&&(!barriers[c]||c==pending)) {
            MetalReplaySegments(replay,NSMakeRange(pending,c-pending),^(id<MTLIndirectCommandBuffer> segment,NSRange local){
                [encoder executeCommandsInBuffer:segment withRange:local];
            });
            [encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
            pending=c;
        }
    }
    if(s->form==MetalFormIndirect&&t*n+to>pending)MetalReplaySegments(replay,NSMakeRange(pending,t*n+to-pending),^(id<MTLIndirectCommandBuffer> segment,NSRange local){
        [encoder executeCommandsInBuffer:segment withRange:local];
    });
}

NSError *MetalScheduleAppend(struct MetalSchedule *root,struct MetalReplay *replay,NSRange invocations,
    NSArray<NSNumber *> *marks,MetalCrossing cross) {
    struct MetalSchedule *s=MetalScheduleTail(root);
    NSArray *held=replay?objc_getAssociatedObject(replay->owner,&replayRecordsKey):nil;
    if(replay&&[held[2] unsignedIntegerValue]!=s->form)return MetalReplayFailure(EINVAL,@"mixed replay forms in one function");
    if(s->form==MetalFormSerial&&marks.count)return MetalReplayFailure(EINVAL,@"crossings require the configured Metal 4 form");
    if(replay&&NSMaxRange(invocations)>[held[4] unsignedIntegerValue])return MetalReplayFailure(EINVAL,@"invocations exceed the recorded slots");
    NSArray *hosts=replay?objc_getAssociatedObject(replay->owner,&replayHostsKey):nil;
    const NSUInteger n=replay?replay->count:0;
    for(NSUInteger t=invocations.location;t<NSMaxRange(invocations);t++) {
        NSMutableDictionary *events=[NSMutableDictionary dictionary];
        NSUInteger from=0,k=0,h=0;
        while(k<marks.count||h<hosts.count) {
            NSUInteger at=k<marks.count?marks[k].unsignedIntegerValue:n;
            if(h<hosts.count)at=MIN(at,[hosts[h][@"at"] unsignedIntegerValue]);
            if(at<from||at>n)return MetalReplayFailure(EINVAL,@"dependency outside its recorded command range");
            MetalScheduleRange(root,replay,t,from,at);
            while(h<hosts.count&&[hosts[h][@"at"] unsignedIntegerValue]==at) {
                NSDictionary *host=hosts[h++];NSValue *key=[NSValue valueWithPointer:host[@"identity"]];
                if([host[@"consume"] boolValue]) {
                    id<MTLSharedEvent> event=events[key];
                    if(!event)return MetalReplayFailure(EINVAL,@"host consumer has no producer in its declared invocation");
                    MetalScheduleHostConsumer(root,event);
                    [events removeObjectForKey:key];
                } else {
                    if(events[key])return MetalReplayFailure(EINVAL,@"host output is overwritten before its declared consumer");
                    id<MTLSharedEvent> event=MetalScheduleHostProducer(root,host[@"operation"]);events[key]=event;
                    [event release];
                }
            }
            while(k<marks.count&&marks[k].unsignedIntegerValue==at) {
                s=MetalScheduleTail(root);
                int failed=cross(s->encoder,s->table,s->residency,t,k++);
                if(failed)return MetalReplayFailure(failed,@"the declared crossing could not be encoded");
                [s->read removeAllObjects];[s->written removeAllObjects];
            }
            from=at;
        }
        if(events.count)return MetalReplayFailure(EINVAL,@"host share has no declared consumer");
        MetalScheduleRange(root,replay,t,from,n);
    }
    return nil;
}

static void MetalScheduleCompleted(struct MetalSchedule *s,NSError *error) {
    struct MetalSchedule *root=s->root;
    if(s->host&&!error)error=s->host();
    s->error=[error retain];
    atomic_store_explicit(&s->status,error?-1:1,memory_order_release);
    if(s->host)[(id<MTLSharedEvent>)s->signal setSignaledValue:s->signalValue];
    // Completion reports the whole function, including independent earlier
    // encoders. This counter observes callbacks; it never gates GPU submission.
    if(atomic_fetch_sub_explicit(&root->pending,1,memory_order_acq_rel)==1) {
        NSError *failure=nil;
        MetalScheduleStatus(root,&failure);
        if(root->completed)root->completed(failure);
        if(root->next)MetalScheduleFree(root);
    }
    MetalScheduleFree(s);
}

static void MetalScheduleFinishPart(struct MetalSchedule *root,void (^completed)(NSError *)) {
    struct MetalSchedule *s=MetalScheduleTail(root);
    if(completed){[root->completed release];root->completed=[completed copy];}
    [s->residency commit];[s->residency requestResidency];
    if(s->residency) {
        if(s->form==MetalFormSerial)[(id<MTLCommandBuffer>)s->command useResidencySet:s->residency];
        else [(id<MTL4CommandBuffer>)s->command useResidencySet:s->residency];
    }
    if(s->form!=MetalFormSerial) {
        NSArray *scope=objc_getAssociatedObject(s->encoder,&encoderAccessKey);
        MetalQueueAccess *previous=scope[0],*current=scope[1];
        MetalRangesMerge(previous->read,current->read);MetalRangesMerge(previous->written,current->written);
        objc_setAssociatedObject(s->encoder,&encoderAccessKey,nil,OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }
    [s->encoder endEncoding];s->finished=YES;
    if(s->form==MetalFormSerial) {
        [(id<MTLCommandBuffer>)s->command addCompletedHandler:^(id<MTLCommandBuffer> done){
            MetalScheduleCompleted(s,done.error);
        }];
    } else {
        [(id<MTL4CommandBuffer>)s->command endCommandBuffer];
        s->options=[MTL4CommitOptions new];
        [s->options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback){
            MetalScheduleCompleted(s,feedback.error);
        }];
    }
    [s->read release];s->read=nil;[s->written release];s->written=nil;
}

int MetalScheduleFinish(struct MetalSchedule *root,void (^completed)(NSError *)) {
    MetalScheduleFinishPart(root,completed);
    for(NSValue *value in root->communicators) {
        const ncclResult_t error=ncclMeshCommRealized(value.pointerValue);
        if(error)return error;
    }
    return 0;
}

int MetalScheduleStatus(const struct MetalSchedule *s,NSError **error) {
    *error=nil;
    const struct MetalSchedule *root=s->root;
    int result=1;
    for(;s;s=s->next){int state=atomic_load_explicit(&s->status,memory_order_acquire);if(state<0){*error=s->error;return -1;}if(!state)result=0;}
    for(NSValue *value in root->communicators) {
        ncclResult_t status=ncclSuccess;
        ncclResult_t call=ncclCommGetAsyncError(value.pointerValue,&status);
        if(call||status!=ncclSuccess) {
            *error=MetalReplayFailure(ENOTCONN,@(ncclGetLastError(value.pointerValue)));
            return -1;
        }
    }
    return result;
}

void MetalScheduleEnqueue(struct MetalSchedule *s) {
    if(s->next)atomic_fetch_add_explicit(&s->references,1,memory_order_relaxed);
    while(s) {
        // Completion may resume the owner and release the whole function
        // before commit returns. Read the compiled successor before commit.
        struct MetalSchedule *next=s->next;
        atomic_fetch_add_explicit(&s->references,1,memory_order_relaxed);
        if(s->form==MetalFormSerial)[(id<MTLCommandBuffer>)s->command commit];
        else {
            if(s->after)[(id<MTL4CommandQueue>)s->queue waitForEvent:s->after value:s->value];
            [(id<MTL4CommandQueue>)s->queue commit:(id<MTL4CommandBuffer> *)&s->command count:1 options:s->options];
        }
        s=next;
    }
}

void MetalScheduleFree(struct MetalSchedule *s) {
    if(!s)return;
    if(atomic_fetch_sub_explicit(&s->references,1,memory_order_acq_rel)!=1)return;
    MetalScheduleFree(s->next);
    if(!s->finished&&s->encoder) {
        [s->encoder endEncoding];
        if(s->form!=MetalFormSerial)[(id<MTL4CommandBuffer>)s->command endCommandBuffer];
    }
    [s->encoder release];[s->command release];[s->queue release];[s->allocator release];[s->table release];[s->data release];[s->signal release];
    [s->residency release];[s->owners release];[s->options release];[s->read release];[s->written release];
    [s->host release];[s->completed release];[s->error release];[s->after release];[s->device release];[s->communicators release];free(s);
}

id<MTLDevice> MetalRecordingDevice(id<MTLDevice> device,BOOL indirectPipelines) {
    MetalDeviceRecorder *recording=[[MetalDeviceRecorder alloc] initWithTarget:device];
    recording->indirect=indirectPipelines;
    return [(id)recording autorelease];
}

// docs/kernels.md Any program: recorded with the dispatch on a recording encoder; on any other, each reach but a
// binding's extent (bound already) a resource the dispatch uses as its access says
static void MetalDispatchAccesses(id<MTLComputeCommandEncoder> encoder, MTLSize groups, MTLSize threads,
    BOOL threadGrid, const struct MetalReach *reaches, size_t count) {
    if (object_getClass(encoder) == [MetalComputeRecorder class]) {
        [(MetalComputeRecorder *)encoder dispatchGrid:groups threadsPerThreadgroup:threads threadGrid:threadGrid reaches:reaches count:count];
        return;
    }
    for (size_t k = 0; k < count; k++)
        if (!(reaches[k].access & MetalReachBinding))
            [encoder useResource:reaches[k].buffer usage:((reaches[k].access & MetalReachRead) ? MTLResourceUsageRead : 0) |
                ((reaches[k].access & MetalReachWritten) ? MTLResourceUsageWrite : 0)];
    if(threadGrid)[encoder dispatchThreads:groups threadsPerThreadgroup:threads];
    else [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

void MetalDispatchReaching(id<MTLComputeCommandEncoder> encoder, MTLSize groups, MTLSize threads,
    const struct MetalReach *reaches, size_t count) {
    MetalDispatchAccesses(encoder,groups,threads,NO,reaches,count);
}

NSUInteger MetalRecordMark(void) {
    return activeCapture ? activeCapture->dispatches.length / sizeof(struct MetalRecordedDispatch) : NSNotFound;
}

void MetalRecordHost(id identity,BOOL consume,NSError *(^operation)(void)) {
    if(!activeCapture)[NSException raise:NSInvalidArgumentException format:@"host shares require whole-function recording"];
    id held=operation?[[operation copy] autorelease]:[NSNull null];
    [activeCapture->hosts addObject:@{@"at":@(MetalRecordMark()),@"identity":identity,@"consume":@(consume),@"operation":held}];
}

// docs/kernels.md Any program
struct MetalReplay *MetalRecord(id<MTLDevice> device, void (^encode)(void), const struct MetalOperand *operands,
    size_t count, NSUInteger invocations, enum MetalForm form) {
    if (object_getClass(device) != [MetalDeviceRecorder class]) {
        fprintf(stderr, "record: the device is not a recording device (MetalRecordingDevice)\n");
        return NULL;
    }
    struct MetalReplay *replay = NULL;
    @autoreleasepool {
        // ../dox/mesh/design/prepared-machine.md#M13: each operand's slots and output offset travel
        // with its buffer, a later declaration of the same buffer replacing an earlier one
        for (size_t k = 0; k < count; k++) {
            objc_setAssociatedObject(operands[k].buffer, &operandStrideKey, @(operands[k].stride), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
            objc_setAssociatedObject(operands[k].buffer, &operandInstancesKey, @(operands[k].instances), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
            objc_setAssociatedObject(operands[k].buffer, &operandOutputOffsetKey, @(operands[k].outputOffset), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        }
        @try {
            struct MetalReplay recorded = {0};
            MetalRecordReplay(device, encode, invocations, form, &recorded);
            NSArray *held = objc_getAssociatedObject(recorded.owner, &replayRecordsKey);
            const struct MetalRecordedDispatch *records = [held[0] bytes];
            for (size_t k = 0; k < count; k++) {
                NSUInteger binding = 0;
                for (uint32_t i = 0; i < recorded.count; i++) for (NSUInteger j = 0; j < MESH_BUFFER_BINDINGS; j++)
                    binding += records[i].buffers[j] == operands[k].buffer && (((records[i].access | (records[i].access >> MESH_ACCESS_WRITE_SHIFT)) >> j) & 1);
                printf("prepared operand %s bindings %lu instances %lu\n", operands[k].name, binding, operands[k].instances);
            }
            if (posix_memalign((void **)&replay, _Alignof(struct MetalReplay), sizeof *replay)) MetalReplayDestroy(&recorded);
            else {
                *replay = recorded;
                if(replay->commands) {
                    id<MTLCommandQueue> queue=[replay->residency.device newCommandQueue];
                    id<MTLCommandBuffer> command=[queue commandBuffer];
                    id<MTLBlitCommandEncoder> encoder=[command blitCommandEncoder];
                    MetalReplaySegments(replay,NSMakeRange(0,(NSUInteger)replay->count*invocations),^(id<MTLIndirectCommandBuffer> segment,NSRange local){
                        [encoder optimizeIndirectCommandBuffer:segment withRange:local];
                    });
                    [encoder endEncoding];[command commit];[command waitUntilCompleted];
                    NSError *error=command.error;[queue release];
                    if(error){MetalReplayFree(replay);replay=NULL;[NSException raise:NSInternalInconsistencyException format:@"%@",error];}
                }


            }
        } @catch (NSException *error) {
            fprintf(stderr, "record: %s\n", error.reason.UTF8String);
        }
    }
    return replay;
}

void MetalReplayFree(struct MetalReplay *replay) {
    if (!replay) return;
    MetalReplayDestroy(replay);
    free(replay);
}
