/* dataprocessing.h: functions over pages, the program side. A function reads named pages and writes named pages as GPU
   work; a program is an ordered set of functions and declared collective calls, compiled once by collective_plan and
   then run with no decision at run time. Any DNN is such a program (an LSTM, an FFN, Gemma): GOAL.md, "the lowest
   attainable overhead collective operations for arbitrary programs on Apple GPUs ... on any number of Macs in any
   cabling"; acceptance: no overhead over a single program moving the same pages in the same order (design/
   page-maps-2026-11-02.md:849). Sources: design/data-model.md sections 5-6 (each step reads its named inputs and writes
   its named results when they are ready); the rubric's interface table (between program and bridge: nothing).

   Header-only: define DATAPROCESSING_IMPLEMENTATION in one Objective-C translation unit (with collective.h's and
   bridge.h's implementations and rdma/mesh-graph.c), e.g.
       cc -O2 -fobjc-arc -x objective-c program.m rdma/mesh-graph.c -framework Metal -framework Foundation -lrdma */
#ifndef CORE_DATAPROCESSING_H
#define CORE_DATAPROCESSING_H
#include "collective.h"

/* A function over pages: its kernel, its operands (each a contiguous run of pages, bound in order as buffers 0..), its
   grid and threadgroup sizes. */
struct page_function {
  const char *kernel;
  const bridge_page *operands; uint32_t operand_count;
  uint32_t grid[3], group[3];
};

/* A program: functions and declared collective calls in program order. A call's pages are produced by the functions
   before it and read by the functions after it. */
struct program_item { uint32_t is_call; struct page_function function; struct collective_call call; };
struct program { const struct program_item *items; uint32_t count; const char *library_source; };

/* Compile the program with its calls (collective_plan, for `repeats` runs) and record its GPU work once: 0, or an
   errno. The node's bridge region must be attached (bridge_attach). */
int program_prepare(const struct program *program, const struct collective_fabric *fabric, uint32_t repeats);
/* Run the recorded repeats; the microseconds a repeat, measured on this rank from commit to completion. */
double program_run(void);
#endif

#ifdef DATAPROCESSING_IMPLEMENTATION
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <time.h>

/* the program's own kernels: a publication (a system fence, then the node's count into every link's supplied word), a
   guard (a link's landed word reaching a count: a consumer's landed-prefix threshold), a fold (a landed partial added
   into the node's own page, halves) */
static NSString *const program_kernels =
  @"#include <metal_stdlib>\nusing namespace metal;\n"
  @"struct words { atomic_ulong supplied, sent, landed; };\n"
  @"kernel void program_publish(device words *w [[buffer(0)]], constant uint2 &a [[buffer(1)]]) {\n"
  @"  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);\n"
  @"  for (uint l = 0; l < a.y; l++) atomic_store_explicit(&w[l].supplied, (ulong)a.x, memory_order_relaxed); }\n"
  @"kernel void program_await(device words *w [[buffer(0)]], constant ulong2 &a [[buffer(1)]]) {\n"
  @"  while (atomic_load_explicit(&w[a.x].landed, memory_order_relaxed) < a.y)\n"
  @"    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system); }\n"
  @"kernel void program_fold(device half *into [[buffer(0)]], device const half *from [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n"
  @"  into[i] += from[i]; }\n";

static id<MTLDevice> program_device_;
static id<MTL4CommandQueue> program_queue_;
static id<MTL4CommandBuffer> program_buffer_;
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
  [rs addAllocation:region]; [rs commit];
  /* the library: the program's kernels and ours */
  MTL4LibraryDescriptor *ld = [MTL4LibraryDescriptor new];
  ld.source = [program_kernels stringByAppendingString:[NSString stringWithUTF8String:program->library_source]];
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
    return pipelines[key] = [compiler newComputePipelineStateWithDescriptor:pd compilerTaskOptions:nil error:nil];
  };
  /* the plan: the program's calls, `repeats` runs of them */
  struct collective_call calls[64]; uint32_t call_count = 0;
  for (uint32_t i = 0; i < program->count; i++) if (program->items[i].is_call) calls[call_count++] = program->items[i].call;
  struct collective_work *work = NULL;
  int err = collective_plan(fabric, calls, call_count, repeats, &work);
  if (err) return err;
  /* record: every repeat's items in program order, each dispatch behind a barrier; a call is its publication, its
     folds (each after its landing's guard, then a publication) and its completion's guards */
  MTL4ArgumentTableDescriptor *td = [MTL4ArgumentTableDescriptor new]; td.maxBufferBindCount = 16;
  id<MTL4ArgumentTable> table = [program_device_ newArgumentTableWithDescriptor:td error:nil];
  id<MTL4CommandAllocator> allocator = [program_device_ newCommandAllocator];
  program_buffer_ = [program_device_ newCommandBuffer];
  [program_buffer_ beginCommandBufferWithAllocator:allocator]; [program_buffer_ useResidencySet:rs];
  id<MTL4ComputeCommandEncoder> enc = [program_buffer_ computeCommandEncoder];
  [enc setArgumentTable:table];
  const MTLStages stages = MTLStageDispatch;
  id<MTLBuffer> constants = [program_device_ newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];
  [rs addAllocation:constants]; [rs commit];
  __block uint64_t used = 0; uint64_t published = 0;
  uint64_t (^constant)(const void *, size_t) = ^uint64_t(const void *v, size_t n) {
    memcpy((char *)constants.contents + used, v, n); const uint64_t at = constants.gpuAddress + used; used += (n + 15) / 16 * 16; return at; };
  void (^one)(const char *, uint64_t, uint64_t) = ^(const char *name, uint64_t a0, uint64_t a1) {
    [enc barrierAfterEncoderStages:stages beforeEncoderStages:stages visibilityOptions:MTL4VisibilityOptionDevice];
    [enc setComputePipelineState:pipeline(name)];
    [table setAddress:a0 atIndex:0]; [table setAddress:a1 atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  };
  const uint64_t words = region.gpuAddress + words_at;
  const uint32_t links = (uint32_t)head->links;
  for (uint32_t r = 0, c = 0; r < repeats; r++) for (uint32_t i = 0; i < program->count; i++) {
    const struct program_item *it = &program->items[i];
    if (!it->is_call) {
      const struct page_function *fn = &it->function;
      [enc barrierAfterEncoderStages:stages beforeEncoderStages:stages visibilityOptions:MTL4VisibilityOptionDevice];
      [enc setComputePipelineState:pipeline(fn->kernel)];
      for (uint32_t k = 0; k < fn->operand_count; k++)
        [table setAddress:region.gpuAddress + head_bytes + (uint64_t)fn->operands[k] * BRIDGE_PAGE atIndex:k];
      [enc dispatchThreadgroups:MTLSizeMake(fn->grid[0], fn->grid[1], fn->grid[2]) threadsPerThreadgroup:MTLSizeMake(fn->group[0], fn->group[1], fn->group[2])];
      continue;
    }
    const struct collective_work *w = &work[c++];
    const uint32_t pub[2] = { (uint32_t)++published, links };
    one("program_publish", words, constant(pub, sizeof pub));
    for (uint32_t k = 0; k < w->fold_count; k++) {
      const struct collective_fold *g = &w->folds[k];
      const uint64_t guard[2] = { g->after_link, g->after };
      one("program_await", words, constant(guard, sizeof guard));
      [enc barrierAfterEncoderStages:stages beforeEncoderStages:stages visibilityOptions:MTL4VisibilityOptionDevice];
      [enc setComputePipelineState:pipeline("program_fold")];
      [table setAddress:region.gpuAddress + head_bytes + (uint64_t)g->into * BRIDGE_PAGE atIndex:0];
      [table setAddress:region.gpuAddress + head_bytes + (uint64_t)g->from * BRIDGE_PAGE atIndex:1];
      [enc dispatchThreads:MTLSizeMake(g->bytes / 2, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      const uint32_t again[2] = { (uint32_t)++published, links };
      one("program_publish", words, constant(again, sizeof again));
    }
    for (uint32_t l = 0; l < links; l++) if (w->await[l]) {
      const uint64_t guard[2] = { l, w->await[l] };
      one("program_await", words, constant(guard, sizeof guard));
    }
  }
  [enc endEncoding]; [program_buffer_ endCommandBuffer];
  program_done_ = [program_device_ newSharedEvent];
  program_repeats_ = repeats;
  return 0;
}

double program_run(void) {
  const uint64_t value = ++program_value_;
  struct timespec a, b;
  clock_gettime(CLOCK_UPTIME_RAW, &a);
  id<MTL4CommandBuffer> buffers[1] = { program_buffer_ };
  [program_queue_ commit:buffers count:1];
  [program_queue_ signalEvent:program_done_ value:value];
  while (program_done_.signaledValue < value) {}
  clock_gettime(CLOCK_UPTIME_RAW, &b);
  return ((b.tv_sec - a.tv_sec) * 1e6 + (b.tv_nsec - a.tv_nsec) / 1e3) / program_repeats_;
}
#endif
