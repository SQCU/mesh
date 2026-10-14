#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"
#include "mesh-plan.h"
#include <errno.h>
/* design/pages-and-functions.md#what-the-page-table-is */
struct mesh_ctx { struct hdr *M; size_t len; uint64_t client,send_off,send_bytes,inputs_off,inputs_bytes; uint32_t row_cursor,arena_cursor; int fd; };
/* design/algorithm-sources.md#programcopy */
/* design/bridge-and-ipc.md#clients: a peer node's queue on this node's links to it that are paired now, the next link's
   past MESH_QUEUES; MESH_ABSENT where none is */
static inline uint32_t mesh_peer_channel(struct mesh_ctx *c,uint32_t peer,uint32_t queue){
  for(uint32_t i=0;i<c->M->links;i++){
    struct mesh_link_info *link=&mesh_links(c->M)[i];
    if(atomic_load_explicit(&link->peer,memory_order_relaxed)!=peer || atomic_load_explicit(&link->phase,memory_order_acquire)!=MESH_PAIRED)continue;
    if(queue<MESH_QUEUES)return i*MESH_QUEUES+queue;
    queue-=MESH_QUEUES;
  }
  return MESH_ABSENT;
}
/* design/bridge-and-ipc.md#clients: a client slot of this node's one region, its execution `name` (the same on every
   member: its programs' queue pairs pair with the members' of the same name) */
int mesh_attach(struct mesh_ctx *,const char *name);
int mesh_detach(struct mesh_ctx *);
/* design/algorithm-sources.md#programtensor */
void mesh_retire(struct hdr *,uint64_t client);

uint32_t mesh_rows_alloc(struct mesh_ctx *,uint32_t count);
/* design/prepared-machine.md#M09 */
uint32_t mesh_arena_alloc(struct mesh_ctx *,uint32_t pages,uint32_t align);
void mesh_backing_bind(struct mesh_ctx *,uint32_t first,uint32_t pages,uint32_t page,uint32_t index);
/* design/algorithm-sources.md#device-operands */
void mesh_device_bind(struct mesh_ctx *,uint32_t row,uint64_t address);
void mesh_rows_release(struct mesh_ctx *,uint32_t first,uint32_t count);
/* design/algorithm-sources.md#index-hand-off */
struct mesh_target *mesh_publish_bind(struct mesh_ctx *,uint32_t row,uint32_t queue);
/* design/algorithm-sources.md#programkernel_call */
uint32_t mesh_publication_prepare(struct hdr *,uint32_t row,struct prepared_publication *);
void mesh_retired_release(struct hdr *);
#endif
