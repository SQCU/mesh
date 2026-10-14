#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"
/* design/algorithm-sources.md#programtensor: a lease of `count` slots of `bytes`, its rows and arena pages claimed by
   compare-exchange (mesh-dataflow.c mesh_claim); 0, or the refusal as a value (ENOMEM, ENOSPC) */
struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride; };
int mesh_section_create(struct mesh_ctx *,size_t bytes,uint32_t count,struct mesh_section *);
int mesh_section_slice(struct mesh_ctx *,struct mesh_section,size_t offset,size_t bytes,uint32_t invocations,uint32_t invocation_pages,struct mesh_section *);
void *mesh_section_address(struct mesh_ctx *,struct mesh_section,uint32_t index);
/* design/algorithm-sources.md#programcopy: `section` sent to or received from (`direction`) the member `node`, never a
   port, on its `queue` (< MESH_QUEUES), for the invocations [begin, end) alone, invocation t at ring slot (t - begin)
   mod `depth` (0: the program's), its cells, words and slots indexed from `begin`; 0, EINVAL or a refused lease */
int mesh_transfer_bind(struct mesh_ctx *,uint32_t node,uint32_t queue,int direction,uint32_t identity,struct mesh_section,
  uint32_t invocation_pages,uint32_t begin,uint32_t end,uint32_t depth);
int mesh_transfers_prepare(struct mesh_ctx *,uint32_t slots,uint32_t invocations,uint32_t depth);
/* design/prepared-machine.md#M30 (cyclic), #M27 (trace, ledger) */
int mesh_transfers_cyclic(struct mesh_ctx *);
int mesh_transfers_trace(struct mesh_ctx *);
void mesh_transfers_ledger(struct mesh_ctx *);
/* design/bridge-and-ipc.md#clients, design/recovery.md#first: 0 once the node's bridge has paired the program toward
   every node it binds, on its current map, directly or along its route through other nodes' bridges; else the failure,
   and that node's status in the client's slot names the node and the code: ESRCH the bridge gone, ETIMEDOUT past
   MESH_PEER_NS+MESH_PAIR_NS+MESH_PAUSE_NS, EHOSTUNREACH no route, ENETDOWN, ECONNRESET, ECANCELED a link or the peer's
   program lost, ENOSPC, ENOMEM a grant refused. A later loss fails the program alike; the next start rereads the map */
int mesh_transfers_start(struct mesh_ctx *);
#endif
