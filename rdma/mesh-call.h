#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"
/* design/algorithm-sources.md#programtensor: a lease of `count` slots of `bytes`, its rows and arena pages claimed by
   compare-exchange (mesh-dataflow.c mesh_claim); 0, or the refusal as a value (ENOMEM, ENOSPC) */
struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride; };
int mesh_section_create(struct mesh_ctx *,size_t bytes,uint32_t count,struct mesh_section *);
void *mesh_section_address(struct mesh_ctx *,struct mesh_section,uint32_t index);
/* design/algorithm-sources.md#programcopy: `section` sent to or received from (`direction`) the member `node`, never a
   port, on its `queue` (< MESH_QUEUES), for the invocations [begin, end) alone, invocation t at ring slot (t - begin)
   mod `depth` (0: the program's), its cells, words and slots indexed from `begin`; 0, EINVAL or a refused lease */
int mesh_transfer_bind(struct mesh_ctx *,uint32_t node,uint32_t queue,int direction,uint32_t identity,struct mesh_section,
  uint32_t invocation_pages,uint32_t begin,uint32_t end,uint32_t depth);
/* design/prepared-machine.md#M04, #M08: the program's cells, words and targets laid out for `slots` queue pairs a
   queue, `invocations` invocations at ring `depth`; EINVAL where a transfer has more slots than the program */
int mesh_transfers_prepare(struct mesh_ctx *,uint32_t slots,uint32_t invocations,uint32_t depth);
/* design/prepared-machine.md#M30 */
int mesh_transfers_cyclic(struct mesh_ctx *);
/* design/bridge-and-ipc.md#protocol: the digest of what the program is made from (0: none), named after
   mesh_transfers_prepare and before mesh_transfers_start; a peer's program of the same execution that names another
   fails both as values (ESTALE); EBUSY once started */
int mesh_transfers_digest(struct mesh_ctx *,uint64_t digest);
/* design/bridge-and-ipc.md#clients, design/recovery.md#first: 0 once the node's bridge has paired the program toward
   every node it binds, on its current map, directly or along its route through other nodes' bridges; else the failure,
   and that node's status in the client's slot names the node and the code: ESRCH the bridge gone, ETIMEDOUT past
   MESH_PEER_NS+MESH_PAIR_NS+MESH_PAUSE_NS, EHOSTUNREACH no cable or route reaches it, ENETDOWN, ECONNRESET,
   ECANCELED a link or the peer's program lost, or this program's own slot retired while it waits (a revoke), ESTALE a
   member's program of the execution made from another digest (mesh_transfers_digest; every destination fails, naming
   that member), ENOSPC, ENOMEM a grant refused. A later loss fails the program alike; the next start rereads the map */
int mesh_transfers_start(struct mesh_ctx *);
/* design/bridge-and-ipc.md#progress: the progress threads of the ports a started program's flows ride woken (each port's
   bell), a group of it about to publish or to wait on its landings (a thread parked after MESH_IDLE_NS of nothing looks
   again within 1 ms anyway while a flow is its port's) */
void mesh_transfers_wake(struct mesh_ctx *);
#endif
