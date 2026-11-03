/* design/page-maps-2026-11-02.md, the solve-recorder contract: the run's effector.

   It encodes the run's solve and decides nothing. Nodes arrive in the solve's order, each with its solve event; the
   solve's edges between events are the only order. mesh_graph_realize makes every decision an effector needs (parts,
   barriers, fences, crossings: mesh-graph.h). This file turns them into Metal 4 calls (command buffers, argument
   tables, barriers, fences, queue events and commit feedback) and host joins (libdispatch groups). It holds no byte
   ranges, infers no dependence, keeps no slot or ring, and holds no lock.

   One run is prepared once (MetalRunFinish) and committed once (MetalRunEnqueue). A run of GPU nodes with no host
   node among their predecessors is one part, committed at enqueue; a part with a host producer commits when that
   producer completes, so no queue holds on its behalf. */
#pragma once
#import <Metal/Metal.h>
#include "mesh-host.h"

#ifdef __cplusplus
extern "C" {
#endif

struct MetalRun;
struct ncclComm;

/* a dispatch's buffer argument: `buffer` from byte `offset` at argument index `index` (a nil buffer: the run's
   constants, MetalRunConstant) */
struct MetalRunBinding { __unsafe_unretained id<MTLBuffer> buffer; NSUInteger offset; uint32_t index; };
/* a native host operation: prepared once at MetalRunFinish (its completion bound), submitted when its producers have
   completed */
typedef const char *(^MetalHostOperation)(MeshHostCompletion, void *, MeshHostTask *);

/* a run on the execution context's queue */
struct MetalRun *MetalRunCreate(id<MTL4CommandQueue> queue);
id<MTLDevice> MetalRunDevice(struct MetalRun *);
/* the solve event of the nodes appended next; nodes of one event keep their appended order */
void MetalRunEvent(struct MetalRun *, uint64_t event);
/* the nodes, in the solve's order */
void MetalRunDispatch(struct MetalRun *, id<MTLComputePipelineState>, const struct MetalRunBinding *, size_t count,
                      MTLSize groups, MTLSize threads, BOOL threadGrid);
void MetalRunCopy(struct MetalRun *, id<MTLBuffer> source, NSUInteger from, id<MTLBuffer> destination, NSUInteger to,
                  NSUInteger bytes);
void MetalRunHost(struct MetalRun *, MetalHostOperation);
/* an outside producer: its event reaching `value` publishes this event's pages */
void MetalRunInput(struct MetalRun *, id<MTLSharedEvent>, uint64_t value);
/* `length` bytes of preparation constants at argument `index`, in the run's one constants lease */
struct MetalRunBinding MetalRunConstant(struct MetalRun *, const void *bytes, size_t length, uint32_t index);
/* the solve's edges (from event, to event); each event must name an appended node, and each edge run forward */
void MetalRunEdges(struct MetalRun *, const uint64_t (*edges)[2], size_t count);
/* a returned page with no local reader: the run is complete once the `bytes`-wide word at `offset` reaches
   `threshold` */
void MetalRunResult(struct MetalRun *, id<MTLBuffer>, NSUInteger offset, uint32_t bytes, uint64_t threshold);
/* an object the run's work uses, held for the run's life; a buffer is also made resident */
void MetalRunHold(struct MetalRun *, id object);
/* a communicator whose occurrences the run records: realized at finish, its error read by MetalRunEnd */
void MetalRunCommunicator(struct MetalRun *, struct ncclComm *);
/* realize, encode every part and prepare every host operation: NULL, or the defect's statement */
const char *MetalRunFinish(struct MetalRun *);
void MetalRunEnqueue(struct MetalRun *);
/* nonblocking: 0 pending, 1 complete (*ended the latest GPU end, host end or result observation, in
   mesh_statistics_now's clock), -1 failed (*error) */
int MetalRunEnd(struct MetalRun *, double *ended, NSError **error);
void MetalRunFree(struct MetalRun *);

#ifdef __cplusplus
}
#endif
