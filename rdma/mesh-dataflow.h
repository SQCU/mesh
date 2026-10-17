#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"
#include "mesh-plan.h"
#include <errno.h>
#include <signal.h>
/* design/pages-and-functions.md#what-the-page-table-is */
struct mesh_ctx { struct hdr *M; size_t len; uint64_t client,send_off,send_bytes,inputs_off,inputs_bytes; uint32_t row_cursor,arena_cursor; int fd; };
/* design/bridge-and-ipc.md#clients: a client slot of this node's one region, its execution `name` (the same on every
   member: its programs' queue pairs pair with the members' of the same name), taken vacant by one compare-exchange
   (mesh_slot_take, which the bridge's own leases take too).  A client takes a vacant slot only: it never retires, waits
   for or names another's, and a slot whose process is gone is the bridge's to retire.  0, EBUSY as a value where none
   is vacant, ESRCH where the node's bridge is gone */
int mesh_attach(struct mesh_ctx *,const char *name);
int mesh_slot_take(struct mesh_ctx *,struct hdr *,uint64_t key);
/* design/bridge-and-ipc.md#clients: a program's slot under its execution `name` on a client's attach (`client`, a
   mesh_attach's), sharing its mapping: a slot is one execution's program, so a client running several takes one each
   on its one attach.  0, EBUSY where none is vacant, ESRCH where the client's bridge is gone */
int mesh_program_attach(struct mesh_ctx *program,const struct mesh_ctx *client,const char *name);
/* the slot retired (mesh_retire), and the region unmapped where the context mapped it (mesh_attach's; a program's slot
   leaves its client's mapping alone) */
int mesh_detach(struct mesh_ctx *);
/* design/bridge-and-ipc.md#region: a process gone (its slot's or the bridge's pid; 0: none) */
static inline int mesh_gone(pid_t process){ return !process || (kill(process,0) && errno==ESRCH); }
/* design/prepared-machine.md#M07, #M08: a receive transfer's frames, a record and a completion word a chunk of a slot
   of an invocation of its range; design/prepared-machine.md#M04: a SEND transfer's cells, a column of its invocations
   and a terminal cell a slot of it (each slot its own column, where its slots send the same rows too) */
static inline uint64_t mesh_transfer_frames(struct hdr *m,const struct mesh_transfer *in,uint32_t invocations){
  return (uint64_t)mesh_row_chunks(m,in->local_row,in->bytes)*in->count*mesh_transfer_active(in,invocations);
}
static inline uint64_t mesh_transfer_cells(const struct mesh_transfer *out,uint32_t invocations){
  return (uint64_t)out->count*(mesh_transfer_active(out,invocations)+1);
}
/* design/algorithm-sources.md#programtensor */
void mesh_retire(struct hdr *,uint64_t client);
/* design/bridge-and-ipc.md#ports, #cables: the one join of two nodes' records, the count of cables between them (0:
   none; the ports each one's record holds paired with the other, the fewer of the two ends' counts), *beta 8e9 over
   the least bandwidth any of those ports at either end reports (0 where none reports one); and the route
   between nodes `from` and `to` on the map of region `m`, its routing table's (mesh-plan.h mesh_routes over the
   records' whole map, each node marked by its bridge's number, as mesh_map's; mesh_route_path: the route from `to`
   to `from` reversed), from `from` to `to` in a path the caller frees: its node count, 0 where none */
int mesh_joined(struct mesh_record *,struct mesh_record *,double *beta);
uint32_t mesh_route(struct hdr *m,uint32_t from,uint32_t to,uint32_t **path);

uint32_t mesh_rows_alloc(struct mesh_ctx *,uint32_t count);
/* design/prepared-machine.md#M09 */
uint32_t mesh_arena_alloc(struct mesh_ctx *,uint32_t pages,uint32_t align);
/* design/bridge-dataflow.md#stripes (mesh 8fc27cbe): exactly arena pages [first, first + pages), claimed by
   compare-exchange as mesh_arena_alloc's are; MESH_ABSENT (errno ENOMEM) where a page of them is held, EINVAL past the
   arena */
uint32_t mesh_arena_claim(struct mesh_ctx *,uint32_t first,uint32_t pages);
/* design/prepared-machine.md#M09: the window's arena pages no lease holds */
uint32_t mesh_arena_free(struct hdr *);
/* design/prepared-machine.md#M09: a lease its client returns while it holds it (mesh-call.h mesh_section_create's
   section of one slot, headed at `row`): its rows released and its rows and arena pages reclaimed by the reclaim a
   retired client's rows take once the bridge releases them; 0, EINVAL where the client holds no lease headed there */
int mesh_lease_return(struct mesh_ctx *,uint32_t row);
void mesh_backing_bind(struct mesh_ctx *,uint32_t first,uint32_t pages,uint32_t page,uint32_t index);
void mesh_rows_release(struct mesh_ctx *,uint32_t first,uint32_t count);
/* design/algorithm-sources.md#programkernel_call; design/prepared-machine.md#M10: the records of the cells the context's
   program releases when `row` is published, one a target (mesh_row_targets) where `records` is given; their count */
uint32_t mesh_publication_prepare(struct mesh_ctx *,uint32_t row,struct prepared_publication *);
void mesh_retired_release(struct hdr *);
/* design/bridge-dataflow.md#streams (mesh 554a3793^ mesh-verbs-stream.c, inside the bridge): a request's kind beside
   mesh-plan.h's MESH_REQUEST_OPEN, _SERVE and _LEDGER, its `node` the receiver (a neighbour) and `bytes` the stream's,
   in whole blocks */
#define MESH_REQUEST_STREAM 4
/* a raw one-way stream of `bytes` from this node's bridge to node `node`'s, every word checked there: the bridges' JSON
   line of the sender's and the receiver's numbers (rate between 10 and 90 percent of the pages landed, mismatched words,
   misordered completions) into `result` (`room` bytes); 0, or the -errno the bridge answered (EHOSTUNREACH no cable,
   EBUSY a stream on the port, ENOMEM no window for it at either end, ETIMEDOUT past the pairing window, a verbs or
   completion error) */
int mesh_stream(uint32_t node,uint64_t bytes,char *result,size_t room);
#ifdef MESH_LOOP_FABRIC
/* design/bridge-dataflow.md#the-loopback-fabric: the loop bridge this process is a client of, before its first attach or
   request (the host build of libmesh alone, make host): its region (a loop bridge's configuration's `region`) and its
   requests' loopback port (its `service`); 0, or EINVAL */
int mesh_loop_bind(const char *region,uint16_t service);
#endif
#endif
