/* The library's Metal interface (design/drift-2026-11-03.md: the metal interface cut to three functions), plain C over
   Metal objects; nothing else of the library touches Metal.
   1. MetalCompile: a kernel's source, function and constants as a compute pipeline (the Metal 4 compiler).
   2. MetalRecord: mesh_graph_realize's outputs (mesh-graph.h: parts, barriers, fences, cross-agent waits) over the
      solve's nodes, encoded once, each part one command buffer.
   3. MetalCommit: every part of a recorded run committed once, in order, and an event the caller owns signalled
      after each part a host step waits on and after the last.
   Every decision is the solve's and mesh_graph_realize's; these encode them and decide nothing.  The caller drives: no
   function takes a handler, block or listener, none calls into the caller, and completion is an event value the caller
   reads.  No host stores an event a part waits on, and a part a landing feeds is committed with the rest, ahead of it:
   a consumer's guard (design/drift-2026-11-03.md 5) is its own kernel's, over its arguments (MetalNode).  Besides
   them, MetalWrap and MetalRelease: a lease's host range as a Metal buffer (nccl-mesh.c's), and the release of what any
   of these returns. */
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __OBJC__
#import <Metal/Metal.h>
#define METAL_OBJECT(type) type
#define METAL_RETAINED NS_RETURNS_RETAINED
#else
#define METAL_OBJECT(type) void *
#define METAL_RETAINED
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* `bytes` from page-aligned `address`, shared storage, no copy; the caller owns the buffer (MetalRelease) */
METAL_OBJECT(id<MTLBuffer>) MetalWrap(void *address, size_t bytes) METAL_RETAINED;
/* releases a buffer, pipeline or run this interface returned */
void MetalRelease(METAL_OBJECT(id) object);

#ifdef __OBJC__
#define METAL_FIELD(type) __unsafe_unretained type
#else
#define METAL_FIELD(type) void *
#endif
struct mesh_graph_crossing;
/* a function constant: `value` of MTLDataType `type` at constant index `index` */
struct MetalConstant { uint32_t index, type; const void *value; };
/* 1: `function` of Metal 4 source text `source` (safe math), specialized by constant[0 .. constants), by the process's
   GPU's Metal 4 compiler; the caller owns it (MetalRelease); NULL with *error, a statement the caller frees */
METAL_OBJECT(id<MTLComputePipelineState>) MetalCompile(const char *source, const char *function,
                                                       const struct MetalConstant *constant, size_t constants,
                                                       char **error) METAL_RETAINED;
/* a dispatch's argument: GPU address `address` (the final lowering's translation of a page the solve placed) at
   argument-table index `index` */
struct MetalArgument { uint64_t address; uint32_t index, padding; };
/* node v of a realized run, in the solve's order (mesh-graph.h mesh_graph_realize's n nodes):
   - with a pipeline, a dispatch of `groups` threadgroups of `threads` threads over argument[0 .. arguments).  A
     consumer of landed bytes, of a SEND's completion or of a host step's output holds its own guard, the solve's (a
     consumer's landed-prefix threshold): two of its arguments are the count's word (a channel's `landed` or `sent`,
     or the caller's count of its host steps) and the threshold's, and its kernel reads them.  Its part is committed
     ahead with every other, so the GPU, already in it, sees the count arrive 1-4 us after its store, where an event a
     host stores starts a waiting part 93-178 us later on an M5 Ultra (metal-microbench docs/facts.md, wake.m);
   - with an event, an outside producer (host[v] 2) whose writer is GPU work: torch's own command buffer (a stream's
     `progress`, _stream.mm inputs), or the queue's signal of `done` at an earlier run's end (the order between
     occurrences, the queue's wait on it in the command processor, no CPU between: a gap of 20-55 us, a median 36 on
     an M5 Ultra, against a fence's 1.6, metal-microbench docs/facts.md, gaps.m);
   - with neither, a host step (host[v] 1) the caller runs, in node order, once the values MetalCommit's rows name.
   Changed by the metal ticket's revision (2026-11-03) from an outside producer for a landing and a host step whose
   event a host loop stored (the bridge's landing never did): a CPU-mediated start at every landing.  What must stay:
   no event a part waits on is stored by a host, and a guard is its consumer's. */
struct MetalNode {
  METAL_FIELD(id<MTLComputePipelineState>) pipeline;
  const struct MetalArgument *argument;
  uint32_t arguments, groups[3], threads[3];
  METAL_FIELD(id<MTLSharedEvent>) event;
};
/* 2: the realized run on `queue` (whose residency sets hold every page an argument addresses): node[0 .. n) and
   mesh_graph_realize's outputs over them, part[v], parts, barrier[v], fence[v] and wait[0 .. waits), each part encoded
   once as one command buffer; it decides nothing.  realize's `deferred` it does not read: a host step's consumer is
   guarded, never held back.  The caller owns it (MetalRelease); NULL with *error, a statement the caller frees */
METAL_OBJECT(id) MetalRecord(METAL_OBJECT(id<MTL4CommandQueue>) queue, uint32_t n, const struct MetalNode *node,
                             const uint32_t *part, uint32_t parts, const unsigned char *barrier,
                             const unsigned char *fence, const struct mesh_graph_crossing *wait, size_t waits,
                             char **error) METAL_RETAINED;
/* 3: every part of `run` committed once, in order, each after its queue waits on the outside producers it waits on,
   producer v's event reaching value[v] (indexed by node); `done` signalled to base + p + 1 after each part p a host
   step waits on (realize's kind 1 rows), and to base + parts after the last.  Once: MTL4CommandBuffer.h and
   MTL4CommandQueue.h give a command buffer no second commit, so a later occurrence is a run of its own.  The caller
   reads done's value, an event it owns, and keeps `run` until done reaches base + parts. */
void MetalCommit(METAL_OBJECT(id) run, const uint64_t *value, METAL_OBJECT(id<MTLSharedEvent>) done, uint64_t base);
#ifdef __cplusplus
}
#endif
