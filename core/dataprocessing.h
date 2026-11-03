/* dataprocessing.h: the program side. Zero-copy views of the program's tensors as pages, and the recorder that
   transduces a captured program (its functions and declared collective calls, in program order) into one recorded GPU
   stream, the collective's publications and guards placed where the program puts its calls, replayed with no host,
   interpreter or lock on the path. It defines no arithmetic: every kernel is the program's (SCOPE.md: composition, not
   computation). Any DNN is such a program (an LSTM, an FFN, Gemma): GOAL.md, "the lowest attainable overhead
   collective operations for arbitrary programs on Apple GPUs ... on any number of Macs in any cabling"; acceptance: no
   overhead over a single program moving the same pages in the same order (design/page-maps-2026-11-02.md:849).
   Sources: design/data-model.md sections 5-6 (each step reads its named inputs and writes its named results when they
   are ready); the rubric's interface table (between program and bridge: nothing).

   Header-only: define DATAPROCESSING_IMPLEMENTATION in one Objective-C translation unit (with collective.h's and
   bridge.h's implementations and rdma/mesh-graph.c), e.g.
       cc -O2 -fobjc-arc -x objective-c program.m rdma/mesh-graph.c -framework Metal -framework Foundation -lrdma */
#ifndef CORE_DATAPROCESSING_H
#define CORE_DATAPROCESSING_H
#include "collective.h"

/* An operand: its pages. A contiguous-typed operand (a function that requires contiguous reads or writes) is one run of
   pages from pages[0], bound as one pointer; every other operand is a paged view of arbitrarily placed pages, bound as
   a constant array of its pages' device pointers (resolved from the page table at recording: the kernel reads literal
   pointers, a tile a page, as a paged attention's block table does). */
struct page_operand { const bridge_page *pages; uint32_t count, contiguous; };
/* A function over pages: its kernel, its operands (bound in order as buffers 0..), its grid and threadgroup sizes. */
struct page_function {
  const char *kernel;
  const struct page_operand *operands; uint32_t operand_count;
  uint32_t grid[3], group[3];
};

/* A program: functions and declared collective calls in program order. A call's pages are produced by the functions
   before it and read by the functions after it. */
struct program_item { uint32_t is_call; struct page_function function; struct collective_call call; };
struct program { const struct program_item *items; uint32_t count; const char *library_source; };

/* Compile one run of the program's calls (collective_plan) and record that run's GPU work once, an indirect command
   buffer ending in collective_next_run; each program_run executes it `repeats` times: 0, or an errno. The node's
   bridge region must be attached (bridge_attach). */
int program_prepare(const struct program *program, const struct collective_fabric *fabric, uint32_t repeats);
/* Run the recorded run `repeats` times in one commit: the microseconds a run on the GPU, from the commit's start to
   its end there (MTL4CommitFeedback), and in `host`, if not NULL, the microseconds a run from this rank's commit to
   its seeing completion, the commit's own latency included. Each call runs the next `repeats` runs of the plan. */
double program_run(double *host);
#endif

#ifdef DATAPROCESSING_IMPLEMENTATION
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <time.h>
#include <stdatomic.h>

static id<MTLDevice> program_device_;
static id<MTL4CommandQueue> program_queue_;
static id<MTL4CommandBuffer> program_buffer_;
static id<MTL4CommandAllocator> program_allocator_;
static id<MTLResidencySet> program_residency_;
static id<MTLIndirectCommandBuffer> program_commands_;   /* one run's commands */
static NSUInteger program_count_;        /* commands a run */
static id<MTLSharedEvent> program_done_;
static uint64_t program_value_;
static uint32_t program_repeats_;

int program_prepare(const struct program *program, const struct collective_fabric *fabric, uint32_t repeats) {
  program_device_ = MTLCreateSystemDefaultDevice();
  program_queue_ = [program_device_ newMTL4CommandQueue];
  id<MTL4Compiler> compiler = [program_device_ newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:nil];
  /* the whole region as one buffer: the words (the region's head) and every page */
  struct bridge_region *head = (struct bridge_region *)((char *)bridge_address(0) - ((sizeof(struct bridge_region) + BRIDGE_PAGE - 1) / BRIDGE_PAGE * BRIDGE_PAGE));
  const size_t head_bytes = (char *)bridge_address(0) - (char *)head, bytes = head_bytes + head->pages * BRIDGE_PAGE;
  id<MTLBuffer> region = [program_device_ newBufferWithBytesNoCopy:head length:bytes options:MTLResourceStorageModeShared deallocator:nil];
  if (!region) return ENOMEM;
  const uint64_t words_at = (uint64_t)((char *)&head->words[0] - (char *)head);
  id<MTLResidencySet> rs = [program_device_ newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:nil];
  [rs addAllocation:region];
  /* the library: the collective's device functions and the program's kernels; every pipeline replayable from an
     indirect command buffer and resident (an indirectly referenced pipeline state is an allocation) */
  MTL4LibraryDescriptor *ld = [MTL4LibraryDescriptor new];
  ld.source = [@COLLECTIVE_DEVICE_FUNCTIONS stringByAppendingString:[NSString stringWithUTF8String:program->library_source]];
  MTLCompileOptions *o = [MTLCompileOptions new]; o.languageVersion = MTLLanguageVersion4_0; ld.options = o;
  NSError *e = nil;
  id<MTLLibrary> library = [compiler newLibraryWithDescriptor:ld error:&e];
  if (!library) { fprintf(stderr, "program: %s\n", e.localizedDescription.UTF8String); return EINVAL; }
  NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines = [NSMutableDictionary dictionary];
  id<MTLComputePipelineState> (^pipeline)(const char *) = ^id<MTLComputePipelineState>(const char *name) {
    NSString *key = [NSString stringWithUTF8String:name];
    if (pipelines[key]) return pipelines[key];
    MTL4LibraryFunctionDescriptor *fd = [MTL4LibraryFunctionDescriptor new]; fd.name = key; fd.library = library;
    MTL4ComputePipelineDescriptor *pd = [MTL4ComputePipelineDescriptor new]; pd.computeFunctionDescriptor = fd;
    pd.supportIndirectCommandBuffers = MTL4IndirectCommandBufferSupportStateEnabled;
    NSError *failed = nil;
    id<MTLComputePipelineState> made = [compiler newComputePipelineStateWithDescriptor:pd compilerTaskOptions:nil error:&failed];
    if (!made) { fprintf(stderr, "program: %s: %s\n", name, failed.localizedDescription.UTF8String); exit(1); }
    [rs addAllocation:made];
    return pipelines[key] = made;
  };
  /* the plan: one run of the program's calls */
  struct collective_call calls[64]; uint32_t call_count = 0;
  for (uint32_t i = 0; i < program->count; i++) if (program->items[i].is_call) calls[call_count++] = program->items[i].call;
  struct collective_work *work = NULL; uint64_t run_publications = 0;
  int err = call_count ? collective_plan(fabric, calls, call_count, &work, &run_publications) : 0;
  if (err) return err;
  const uint32_t links = (uint32_t)head->links;
  /* the commands of a run and the constants they read: the run count, then one run's arguments */
  size_t constant_bytes = 16 + 32; NSUInteger count = 1; uint32_t binds = 3;   /* the run count; the run's end */
  for (uint32_t i = 0, c = 0; i < program->count; i++) {
    const struct program_item *it = &program->items[i];
    if (it->is_call) {
      const struct collective_work *w = &work[c++];
      uint32_t awaits = 0;
      for (uint32_t l = 0; l < links; l++) awaits += w->await[l] != 0;
      count += 1 + 3 * (NSUInteger)w->fold_count + awaits;
      constant_bytes += 32 * (1 + 2 * (size_t)w->fold_count + awaits);
      continue;
    }
    count += 1; binds = MAX(binds, it->function.operand_count);
    for (uint32_t k = 0; k < it->function.operand_count; k++)
      if (!it->function.operands[k].contiguous) constant_bytes += (it->function.operands[k].count * 8 + 15) / 16 * 16;
  }
  id<MTLBuffer> constants = [program_device_ newBufferWithLength:constant_bytes options:MTLResourceStorageModeShared];
  [rs addAllocation:constants];
  memset(constants.contents, 0, 16);
  __block uint64_t used = 16;
  uint64_t (^constant)(const void *, size_t) = ^uint64_t(const void *v, size_t n) {
    memcpy((char *)constants.contents + used, v, n); const uint64_t at = used; used += (n + 15) / 16 * 16; return at; };
  /* record: one run's commands, once, into an indirect command buffer (design/prepared-machine.md M06, the form of the
     deleted metal-microbench metal_recording.m, 35a5cb9f): every run replays the same commands, their counts within a
     run and the run count carried in the constants' first word, so nothing is recorded, bound or built between runs
     (a recorded MTL4CommandBuffer commits once: on a second commit IOGPUMetal4CommandQueue asserts its allocator's
     storage, 2026-11-03, M5 Max). A call is its publication, its combines (each after its landing's guard, then a
     publication) and its completion's guards; a run ends by advancing the run count. Each command waits on the one
     before it (setBarrier). */
  MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
  d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch | MTLIndirectCommandTypeConcurrentDispatchThreads;
  d.inheritBuffers = NO; d.inheritPipelineState = NO; d.maxKernelBufferBindCount = binds;
  id<MTLIndirectCommandBuffer> commands = [program_device_ newIndirectCommandBufferWithDescriptor:d maxCommandCount:count options:0];
  if (!commands) return ENOMEM;
  __block NSUInteger at = 0;
  id<MTLIndirectComputeCommand> (^next)(id<MTLComputePipelineState>) = ^id<MTLIndirectComputeCommand>(id<MTLComputePipelineState> state) {
    id<MTLIndirectComputeCommand> command = [commands indirectComputeCommandAtIndex:at++];
    [command setComputePipelineState:state]; [command setBarrier];
    return command;
  };
  void (^one)(const char *, const uint64_t *) = ^(const char *name, const uint64_t *a) {
    id<MTLIndirectComputeCommand> command = next(pipeline(name));
    [command setKernelBuffer:region offset:words_at atIndex:0];
    [command setKernelBuffer:constants offset:constant(a, 4 * sizeof *a) atIndex:1];
    [command setKernelBuffer:constants offset:0 atIndex:2];
    [command concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  };
  uint64_t receives[BRIDGE_LINKS] = { 0 };   /* each link's receives a run: the last call's completion counts */
  if (call_count) memcpy(receives, work[call_count - 1].await, sizeof receives);
  uint64_t published = 0;
  for (uint32_t i = 0, c = 0; i < program->count; i++) {
    const struct program_item *it = &program->items[i];
    if (!it->is_call) {
      const struct page_function *fn = &it->function;
      id<MTLIndirectComputeCommand> command = next(pipeline(fn->kernel));
      for (uint32_t k = 0; k < fn->operand_count; k++) {
        const struct page_operand *op = &fn->operands[k];
        if (op->contiguous) { [command setKernelBuffer:region offset:head_bytes + (uint64_t)op->pages[0] * BRIDGE_PAGE atIndex:k]; continue; }
        uint64_t pointers[op->count];
        for (uint32_t q = 0; q < op->count; q++) pointers[q] = region.gpuAddress + head_bytes + (uint64_t)op->pages[q] * BRIDGE_PAGE;
        [command setKernelBuffer:constants offset:constant(pointers, sizeof pointers) atIndex:k];
      }
      [command concurrentDispatchThreadgroups:MTLSizeMake(fn->grid[0], fn->grid[1], fn->grid[2])
                        threadsPerThreadgroup:MTLSizeMake(fn->group[0], fn->group[1], fn->group[2])];
      continue;
    }
    const struct collective_work *w = &work[c];
    const struct collective_call *call = &calls[c++];
    one("collective_publish", (const uint64_t[4]){ ++published, run_publications, links });
    for (uint32_t k = 0; k < w->fold_count; k++) {
      const struct collective_fold *g = &w->folds[k];
      one("collective_await", (const uint64_t[4]){ g->after_link, g->after, receives[g->after_link] });
      /* the program's combine over the landed partial and its own page */
      id<MTLComputePipelineState> combine = pipeline(call->combine);
      const NSUInteger threads = g->bytes / call->combine_unit;
      id<MTLIndirectComputeCommand> command = next(combine);
      [command setKernelBuffer:region offset:head_bytes + (uint64_t)g->into * BRIDGE_PAGE atIndex:0];
      [command setKernelBuffer:region offset:head_bytes + (uint64_t)g->from * BRIDGE_PAGE atIndex:1];
      [command concurrentDispatchThreads:MTLSizeMake(threads, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(MIN(threads, combine.maxTotalThreadsPerThreadgroup), 1, 1)];
      one("collective_publish", (const uint64_t[4]){ ++published, run_publications, links });
    }
    for (uint32_t l = 0; l < links; l++) if (w->await[l])
      one("collective_await", (const uint64_t[4]){ l, w->await[l], receives[l] });
  }
  one("collective_next_run", (const uint64_t[4]){ 0 });
  [rs addAllocation:commands]; [rs commit]; [rs requestResidency];
  program_residency_ = rs; program_commands_ = commands; program_count_ = at;
  program_allocator_ = [program_device_ newCommandAllocator];
  program_buffer_ = [program_device_ newCommandBuffer];
  program_done_ = [program_device_ newSharedEvent];
  program_repeats_ = repeats;
  /* optimized once (optimizeIndirectCommandBuffer, as the deleted replay did), before any run */
  [program_buffer_ beginCommandBufferWithAllocator:program_allocator_]; [program_buffer_ useResidencySet:rs];
  id<MTL4ComputeCommandEncoder> enc = [program_buffer_ computeCommandEncoder];
  [enc optimizeIndirectCommandBuffer:commands withRange:NSMakeRange(0, at)];
  [enc endEncoding]; [program_buffer_ endCommandBuffer];
  id<MTL4CommandBuffer> buffers[1] = { program_buffer_ };
  [program_queue_ commit:buffers count:1];
  [program_queue_ signalEvent:program_done_ value:++program_value_];
  while (program_done_.signaledValue < program_value_) {}
  return 0;
}

double program_run(double *host) {
  /* one command buffer: the recorded run executed `repeats` times, each after the last */
  [program_allocator_ reset];
  [program_buffer_ beginCommandBufferWithAllocator:program_allocator_]; [program_buffer_ useResidencySet:program_residency_];
  id<MTL4ComputeCommandEncoder> enc = [program_buffer_ computeCommandEncoder];
  for (uint32_t r = 0; r < program_repeats_; r++) {
    if (r) [enc barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
    [enc executeCommandsInBuffer:program_commands_ withRange:NSMakeRange(0, program_count_)];
  }
  [enc endEncoding]; [program_buffer_ endCommandBuffer];
  __block double gpu = 0; __block _Atomic int fed = 0;
  MTL4CommitOptions *options = [MTL4CommitOptions new];
  [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
    gpu = (feedback.GPUEndTime - feedback.GPUStartTime) * 1e6; atomic_store(&fed, 1); }];
  const uint64_t value = ++program_value_;
  struct timespec a, b;
  clock_gettime(CLOCK_UPTIME_RAW, &a);
  id<MTL4CommandBuffer> buffers[1] = { program_buffer_ };
  [program_queue_ commit:buffers count:1 options:options];
  [program_queue_ signalEvent:program_done_ value:value];
  while (program_done_.signaledValue < value) {}
  clock_gettime(CLOCK_UPTIME_RAW, &b);
  while (!atomic_load(&fed)) {}          /* the feedback follows the completion */
  if (host) *host = ((b.tv_sec - a.tv_sec) * 1e6 + (b.tv_nsec - a.tv_nsec) / 1e3) / program_repeats_;
  return gpu / program_repeats_;
}
#endif
