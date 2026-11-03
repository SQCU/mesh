// metal.h over Apple's Metal 4 API, manual retain and release. 1: MTL4Compiler.h, MTL4LibraryDescriptor.h,
// MTL4LibraryFunctionDescriptor.h, MTL4SpecializedFunctionDescriptor.h (constant values optional), MTL4ComputePipeline.h.
// 2: MTL4CommandBuffer.h (an allocator serves one open command buffer, so one each), MTL4ComputeCommandEncoder.h (the
// argument table is snapshotted at each dispatch, so one table serves every dispatch), MTL4CommandEncoder.h (barriers;
// a fence wait covers every prior update of the fence on the queue). 3: MTL4CommandQueue.h (waitForEvent holds the
// queue's later work, signalEvent follows all its earlier work; neither it nor MTL4CommandBuffer.h gives a command
// buffer a second commit). Every decision is mesh_graph_realize's (mesh-graph.h).
#import "metal.h"
#include <stdlib.h>
#include <string.h>
#include "mesh-graph.h"

static char *stated(NSError *failure, const char *otherwise) {
    return strdup(failure.localizedDescription.UTF8String ?: otherwise);
}

// A region span's buffer, hazard-tracked. Changed from untracked (since 2026-09, no reason recorded) by the FFN
// demonstration session: every MPS tensor of a mesh process is a lease's buffer (torch_mesh allocator.mm), and torch's
// stream is classic Metal, whose encoders are ordered against one another only over tracked resources, as torch's own
// allocator's are (its pools' HAZARD usage). Untracked, torch's cast and the copy after it raced in one command buffer:
// v.float().cpu() read y and w2 wrong on every rank while the CPU read their bytes right (metal-microbench
// docs/facts.md, ffn-2026-11-03 runs 11-13). A Metal 4 queue tracks nothing whatever the mode, so a run's encoding
// is unchanged. What must stay: tracked, while torch's stream runs over leases.
// The metal ticket (2026-11-03) took the device per call instead of a dispatch_once block; the buffer is unchanged.
id<MTLBuffer> MetalWrap(void *address, size_t bytes) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:address length:bytes
        options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked deallocator:nil];
    [device release];
    return buffer;
}

void MetalRelease(id object) { [object release]; }

id<MTLComputePipelineState> MetalCompile(const char *source, const char *function, const struct MetalConstant *constant,
                                         size_t constants, char **error) {
    @autoreleasepool {
        NSError *failure = nil;
        id<MTLDevice> device = [MTLCreateSystemDefaultDevice() autorelease];
        id<MTL4Compiler> compiler = [[device newCompilerWithDescriptor:[[MTL4CompilerDescriptor new] autorelease]
                                                                 error:&failure] autorelease];
        MTLCompileOptions *options = [[MTLCompileOptions new] autorelease];
        options.languageVersion = MESH_METAL_LANGUAGE;
        options.mathMode = MTLMathModeSafe;
        MTL4LibraryDescriptor *text = [[MTL4LibraryDescriptor new] autorelease];
        text.source = @(source);
        text.options = options;
        MTL4LibraryFunctionDescriptor *named = [[MTL4LibraryFunctionDescriptor new] autorelease];
        named.name = @(function);
        named.library = [[compiler newLibraryWithDescriptor:text error:&failure] autorelease];
        MTLFunctionConstantValues *values = [[MTLFunctionConstantValues new] autorelease];
        for (size_t k = 0; k < constants; k++)
            [values setConstantValue:constant[k].value type:(MTLDataType)constant[k].type atIndex:constant[k].index];
        MTL4SpecializedFunctionDescriptor *specialized = [[MTL4SpecializedFunctionDescriptor new] autorelease];
        specialized.functionDescriptor = named;
        specialized.constantValues = values;
        MTL4ComputePipelineDescriptor *described = [[MTL4ComputePipelineDescriptor new] autorelease];
        described.computeFunctionDescriptor = specialized;
        id<MTLComputePipelineState> made = named.library ?
            [compiler newComputePipelineStateWithDescriptor:described compilerTaskOptions:nil error:&failure] : nil;
        if (!made) *error = stated(failure, "no pipeline");
        return made;
    }
}

// A recorded run: its `parts`, part p's command buffer buffer[p]; the waits rows on an outside producer, row j's event
// and producer node, linked by the part waiting (head[p], then next[j]: realize's rows as given, indexed by their
// waiter); signal[p], a host step waits on part p (a kind 1 row: realize's, indexed by its producer); and what the
// command buffers use, which Metal 4 does not retain.
@interface MetalRecorded : NSObject {
@public
    id<MTL4CommandQueue> queue;
    NSMutableArray *held;
    uint32_t parts;
    __unsafe_unretained id<MTL4CommandBuffer> *buffer;
    __unsafe_unretained id<MTLSharedEvent> *event;
    uint32_t *producer, *head, *next;
    unsigned char *signal;
}
@end

@implementation MetalRecorded
- (void)dealloc {
    [queue release];
    [held release];
    free(buffer);
    free(event);
    free(producer);
    free(head);
    free(next);
    free(signal);
    [super dealloc];
}
@end

static id refused(MetalRecorded *run, void *encoder, void *published, NSError *failure, const char *otherwise,
                  char **error) {
    *error = stated(failure, otherwise);
    free(encoder);
    free(published);
    [run release];
    return nil;
}

id MetalRecord(id<MTL4CommandQueue> queue, uint32_t n, const struct MetalNode *node, const uint32_t *part,
               uint32_t parts, const unsigned char *barrier, const unsigned char *fence,
               const struct mesh_graph_crossing *wait, size_t waits, char **error) {
    @autoreleasepool {
        id<MTLDevice> device = queue.device;
        MetalRecorded *run = [MetalRecorded new];
        run->queue = [queue retain];
        run->held = [NSMutableArray new];
        run->parts = parts;
        run->signal = calloc(parts + 1, sizeof *run->signal);
        run->buffer = calloc(parts + 1, sizeof *run->buffer);
        run->event = calloc(waits + 1, sizeof *run->event);
        run->producer = calloc(waits + 1, sizeof *run->producer);
        run->next = calloc(waits + 1, sizeof *run->next);
        run->head = malloc((parts + 1) * sizeof *run->head);
        __unsafe_unretained id<MTL4ComputeCommandEncoder> *encoder = calloc(parts + 1, sizeof *encoder);
        __unsafe_unretained id<MTLFence> *published = calloc(parts + 1, sizeof *published);
        NSError *failure = nil;
        MTL4ArgumentTableDescriptor *described = [[MTL4ArgumentTableDescriptor new] autorelease];
        described.maxBufferBindCount = 31; // Metal's buffer bind limit (torch_mesh kernels.py ARGUMENT_BUFFERS)
        id<MTL4ArgumentTable> table = [[device newArgumentTableWithDescriptor:described error:&failure] autorelease];
        if (!table || !run->signal || !run->buffer || !run->event || !run->producer || !run->next || !run->head ||
            !encoder || !published)
            return refused(run, encoder, published, failure, "out of memory", error);
        memset(run->head, 0xff, (parts + 1) * sizeof *run->head);
        [run->held addObject:table];
        for (uint32_t p = 0; p < parts; p++) {
            id<MTL4CommandAllocator> allocator = [[device newCommandAllocator] autorelease];
            id<MTL4CommandBuffer> buffer = [[device newCommandBuffer] autorelease];
            id<MTLFence> fenced = [[device newFence] autorelease];
            if (!allocator || !buffer || !fenced)
                return refused(run, encoder, published, nil, "no command allocator, buffer or fence", error);
            [run->held addObject:allocator];
            [run->held addObject:buffer];
            [run->held addObject:fenced];
            run->buffer[p] = buffer;
            published[p] = fenced;
            [buffer beginCommandBufferWithAllocator:allocator];
            encoder[p] = [buffer computeCommandEncoder];
            [encoder[p] setArgumentTable:table];
        }
        // a part waits on a part through the producer part's fence, before any of its dispatches; on an outside
        // producer, in the queue before its commit (MetalCommit). A host step waiting on a part reads done's signal
        // after it (MetalCommit); a host step's consumer reads its own guard (metal.h MetalNode), so nothing here
        for (size_t j = 0; j < waits; j++) {
            const struct mesh_graph_crossing w = wait[j];
            if (w.kind == 0)
                [encoder[w.waiter] waitForFence:published[w.producer] beforeEncoderStages:MTLStageDispatch];
            if (w.kind == 1) run->signal[w.producer] = 1;
            if (w.kind == 4 && node[w.producer].event) {
                [run->held addObject:node[w.producer].event];
                run->event[j] = node[w.producer].event;
                run->producer[j] = w.producer;
                run->next[j] = run->head[w.waiter];
                run->head[w.waiter] = (uint32_t)j;
            }
        }
        for (uint32_t v = 0; v < n; v++) {
            if (!node[v].pipeline) continue;
            id<MTL4ComputeCommandEncoder> into = encoder[part[v]];
            if (barrier[v])
                [into barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch
                              visibilityOptions:MTL4VisibilityOptionDevice];
            [run->held addObject:node[v].pipeline];
            [into setComputePipelineState:node[v].pipeline];
            for (uint32_t k = 0; k < node[v].arguments; k++)
                [table setAddress:node[v].argument[k].address atIndex:node[v].argument[k].index];
            [into dispatchThreadgroups:MTLSizeMake(node[v].groups[0], node[v].groups[1], node[v].groups[2])
                 threadsPerThreadgroup:MTLSizeMake(node[v].threads[0], node[v].threads[1], node[v].threads[2])];
            if (fence[v]) [into updateFence:published[part[v]] afterEncoderStages:MTLStageDispatch];
        }
        for (uint32_t p = 0; p < parts; p++) {
            [encoder[p] endEncoding];
            [run->buffer[p] endCommandBuffer];
        }
        free(encoder);
        free(published);
        return run;
    }
}

// The parts since `from` go in one commit up to each queue wait (it holds what follows it) and each done signal (it
// follows what precedes it). Changed by the metal ticket's revision (2026-11-03) from a caller's range [first, end)
// with one signal at its end: the caller worked out the ranges from realize's rows, and a host step waiting on a part
// saw it only at its range's end. What must stay: one commit of every part, a signal after each part a host waits on.
void MetalCommit(id recorded, const uint64_t *value, id<MTLSharedEvent> done, uint64_t base) {
    MetalRecorded *run = recorded;
    uint32_t from = 0;
    for (uint32_t p = 0; p < run->parts; p++) {
        if (run->head[p] != UINT32_MAX) {
            if (p > from) [run->queue commit:run->buffer + from count:p - from];
            from = p;
            for (uint32_t j = run->head[p]; j != UINT32_MAX; j = run->next[j])
                [run->queue waitForEvent:run->event[j] value:value[run->producer[j]]];
        }
        if (!run->signal[p]) continue;
        [run->queue commit:run->buffer + from count:p + 1 - from];
        from = p + 1;
        [run->queue signalEvent:done value:base + p + 1];
    }
    if (run->parts > from) [run->queue commit:run->buffer + from count:run->parts - from];
    [run->queue signalEvent:done value:base + run->parts];
}
