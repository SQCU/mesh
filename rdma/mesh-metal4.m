/* design/the-compiler.md, order 4 */
/* design/torch-source-lowering-change-request.md, items 1 and 3 */
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>

struct mesh_metal4_dispatch {
    void *pipeline, *table;
    const uint64_t *address;
    uint32_t addresses, barrier, groups, padding;
    uint64_t grid[3], group[3];
};

@interface MeshMetal4Stream : NSObject
@property id<MTL4CommandAllocator> allocator;
@property id<MTL4CommandBuffer> commands;
@property id<MTLResidencySet> held, viewed;
@property id<MTLSharedEvent> event, arrival;
@property uint64_t value, arrived;
@property(atomic) NSError *error;
@end
@implementation MeshMetal4Stream
@end

static id<MTLDevice> device;
static id<MTL4CommandQueue> queue;
static id<MTL4Compiler> compiler;

static void begin(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        device = MTLCreateSystemDefaultDevice();
        queue = [device newMTL4CommandQueue];
        compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:nil];
    });
}

static void report(NSError *error, char *message, size_t room) {
    if (room) strlcpy(message, error ? error.localizedDescription.UTF8String : "", room);
}

void *mesh_metal4_pipeline(const char *source, const char *name, char *message, size_t room) {
    begin();
    NSError *error = nil;
    MTL4LibraryDescriptor *text = [MTL4LibraryDescriptor new];
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    options.mathMode = MTLMathModeSafe;
    text.source = @(source);
    text.options = options;
    id<MTLLibrary> library = [compiler newLibraryWithDescriptor:text error:&error];
    id<MTLComputePipelineState> pipeline = nil;
    if (library) {
        MTL4LibraryFunctionDescriptor *function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @(name);
        MTL4ComputePipelineDescriptor *descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        pipeline = [compiler newComputePipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
    }
    report(pipeline ? nil : error, message, room);
    return (__bridge_retained void *)pipeline;
}

uint64_t mesh_metal4_width(void *pipeline) {
    return ((__bridge id<MTLComputePipelineState>)pipeline).threadExecutionWidth;
}

void *mesh_metal4_table(uint32_t addresses) {
    begin();
    MTL4ArgumentTableDescriptor *descriptor = [MTL4ArgumentTableDescriptor new];
    descriptor.maxBufferBindCount = addresses;
    return (__bridge_retained void *)[device newArgumentTableWithDescriptor:descriptor error:nil];
}

void *mesh_metal4_buffer(uint64_t bytes) {
    begin();
    return (__bridge_retained void *)[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

void *mesh_metal4_view(void *pages, uint64_t bytes) {
    begin();
    return (__bridge_retained void *)[device newBufferWithBytesNoCopy:pages
                                                               length:bytes
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
}

void mesh_metal4_release(void *object) {
    (void)(__bridge_transfer id)object;
}

uint64_t mesh_metal4_address(void *buffer) {
    return ((__bridge id<MTLBuffer>)buffer).gpuAddress;
}

void *mesh_metal4_contents(void *buffer) {
    return ((__bridge id<MTLBuffer>)buffer).contents;
}

static void add(id<MTLResidencySet> set, void *const *buffer, uint32_t buffers) {
    for (uint32_t i = 0; i < buffers; i++) {
        id<MTLBuffer> one = (__bridge id<MTLBuffer>)buffer[i];
        [set addAllocation:one.heap ? (id<MTLAllocation>)one.heap : one];
    }
    [set commit];
}

void *mesh_metal4_stream(void *const *buffer, uint32_t buffers) {
    begin();
    MeshMetal4Stream *stream = [MeshMetal4Stream new];
    stream.allocator = [device newCommandAllocator];
    stream.commands = [device newCommandBuffer];
    stream.held = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:nil];
    stream.viewed = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:nil];
    stream.event = [device newSharedEvent];
    stream.arrival = [device newSharedEvent];
    add(stream.held, buffer, buffers);
    return (__bridge_retained void *)stream;
}

/* torch/csrc/api/include/torch/mps.h: commit, get_command_buffer, get_dispatch_queue */
void mesh_metal4_after(void *of, void (*commit)(void), void *(*commands)(void), void *(*encoding)(void)) {
    MeshMetal4Stream *stream = (__bridge MeshMetal4Stream *)of;
    id<MTLSharedEvent> arrival = stream.arrival;
    uint64_t arrived = stream.arrived + 1;
    stream.arrived = arrived;
    dispatch_sync((__bridge dispatch_queue_t)encoding(), ^{
        commit();
        [(__bridge id<MTLCommandBuffer>)commands() encodeSignalEvent:arrival value:arrived];
        commit();
    });
    [queue waitForEvent:arrival value:arrived];
}

void *mesh_metal4_fence(void) {
    begin();
    return (__bridge_retained void *)[device newFence];
}

void mesh_metal4_commit(void *of, const struct mesh_metal4_dispatch *dispatch, uint32_t dispatches,
                        void *const *buffer, uint32_t buffers, void *const *after, uint32_t afters, void *before) {
    MeshMetal4Stream *stream = (__bridge MeshMetal4Stream *)of;
    add(stream.viewed, buffer, buffers);
    id<MTL4CommandBuffer> commands = stream.commands;
    [commands beginCommandBufferWithAllocator:stream.allocator];
    [commands useResidencySet:stream.held];
    [commands useResidencySet:stream.viewed];
    id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    for (uint32_t i = 0; i < afters; i++)
        [encoder waitForFence:(__bridge id<MTLFence>)after[i] beforeEncoderStages:MTLStageDispatch];
    for (uint32_t i = 0; i < dispatches; i++) {
        const struct mesh_metal4_dispatch *d = dispatch + i;
        id<MTL4ArgumentTable> table = (__bridge id<MTL4ArgumentTable>)d->table;
        if (d->barrier)
            [encoder barrierAfterEncoderStages:MTLStageDispatch
                           beforeEncoderStages:MTLStageDispatch
                             visibilityOptions:MTL4VisibilityOptionDevice];
        for (uint32_t a = 0; a < d->addresses; a++)
            [table setAddress:d->address[a] atIndex:a];
        [encoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)d->pipeline];
        [encoder setArgumentTable:table];
        MTLSize grid = MTLSizeMake(d->grid[0], d->grid[1], d->grid[2]);
        MTLSize group = MTLSizeMake(d->group[0], d->group[1], d->group[2]);
        if (d->groups)
            [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
        else
            [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    }
    if (before) [encoder updateFence:(__bridge id<MTLFence>)before afterEncoderStages:MTLStageDispatch];
    [encoder endEncoding];
    [commands endCommandBuffer];
    MTL4CommitOptions *options = [MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        if (feedback.error) stream.error = feedback.error;
    }];
    [queue commit:&commands count:1 options:options];
    [queue signalEvent:stream.event value:++stream.value];
}

int mesh_metal4_wait(void *of, char *message, size_t room) {
    MeshMetal4Stream *stream = (__bridge MeshMetal4Stream *)of;
    [stream.event waitUntilSignaledValue:stream.value timeoutMS:UINT64_MAX];
    [stream.allocator reset];
    [stream.viewed removeAllAllocations];
    NSError *error = stream.error;
    report(error, message, room);
    return error != nil;
}
