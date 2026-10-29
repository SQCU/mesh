#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"
#include "mesh-plan.h"
#include <errno.h>
#include <libproc.h>
#include <signal.h>
/* design/transport.md */
/* `view` is this client's fixed virtual window, indexed by bridge row. Its
   mappings are prepared once when pages are assigned; execution never remaps. */
struct mesh_ctx { struct hdr *M; size_t len; uint64_t client; uint32_t row_cursor,arena_cursor; int fd; void *view; size_t view_bytes; };
/* design/transport.md: a process's client slot of this node's one region, taken vacant by one compare-exchange
   (mesh_slot_take, which the bridge's own leases take too) and stamped with the process's identity (its pid and
   start): the owner of its leases of the region and of the streams it holds, and a holder of the region the heartbeat
   reads (design/transport.md section 2).  A client takes a vacant slot only: it never retires, waits for or names
   another's, and a slot whose process is gone is the bridge's to retire.  0, EBUSY as a value where none is vacant,
   EINVAL where the region expired as the slot was taken (it is let go) */
int mesh_attach(struct mesh_ctx *);
int mesh_slot_take(struct mesh_ctx *,struct hdr *);
/* design/transport.md section 5: the expiry the process's collective library registers for libmesh's own waits
   (libnccl-mesh's mesh_library_expired, which stops the process's GPU work and posted receives first); none
   registered, mesh.h mesh_expired */
void mesh_expiry_set(void (*expired)(const char *wait,uint64_t ns));
/* the slot retired (mesh_retire), and the region unmapped */
int mesh_detach(struct mesh_ctx *);
/* design/transport.md: a process gone (its slot's or the bridge's pid; 0: none) */
static inline int mesh_gone(pid_t process){ return !process || (kill(process,0) && errno==ESRCH); }
/* design/transport.md section 2: a process's start in microseconds of the wall clock (libproc PROC_PIDTBSDINFO), 0
   where it is gone or in its exit (XNU takes a process from every lookup by pid as its exit begins, before the kernel
   releases its device objects: metal-microbench docs/facts.md); with its pid, the identity of a holder of the region,
   which a later process given the same pid does not share */
static inline uint64_t mesh_started(pid_t process){
  struct proc_bsdinfo info;
  return process && proc_pidinfo(process,PROC_PIDTBSDINFO,0,&info,(int)sizeof info)==(int)sizeof info?
         (uint64_t)info.pbi_start_tvsec*UINT64_C(1000000)+info.pbi_start_tvusec:0;
}
/* the process `pid` of start `start` alive (a start of 0, which a region an earlier bridge or library wrote holds:
   the pid alone) */
static inline int mesh_holds(pid_t pid,uint64_t start){ return start?mesh_started(pid)==start:!mesh_gone(pid); }
/* design/algorithm-sources.md#programtensor */
void mesh_retire(struct hdr *,uint64_t client);
/* design/transport.md section 14: the one join of two nodes' records: the cables each one's bridge reaches the other
   over, the fewer of the two ends' counts (0: none); *beta 8e9 over the least bandwidth either reports of them (0 where
   neither reports one) */
int mesh_joined(struct mesh_record *,struct mesh_record *,double *beta);

uint32_t mesh_rows_alloc(struct mesh_ctx *,uint32_t count);
/* design/transport.md */
uint32_t mesh_arena_alloc(struct mesh_ctx *,uint32_t pages,uint32_t align);
/* design/transport.md: the region's arena pages no lease holds */
uint32_t mesh_arena_free(struct hdr *);
/* design/transport.md: a lease its client returns while it holds it (mesh-call.h mesh_section_create's
   section of one slot, headed at `row`): its rows released and its rows and arena pages reclaimed by the reclaim a
   retired client's rows take once the bridge releases them; 0, EINVAL where the client holds no lease headed there */
int mesh_lease_return(struct mesh_ctx *,uint32_t row);
void mesh_backing_bind(struct mesh_ctx *,uint32_t first,uint32_t pages,uint32_t page,uint32_t index);
void mesh_rows_release(struct mesh_ctx *,uint32_t first,uint32_t count);
uint32_t mesh_retired_release(struct hdr *);
#endif
