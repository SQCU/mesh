#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"
#include <errno.h>
/* design/pages-and-functions.md#what-the-page-table-is */
struct mesh_range { uint32_t first,count; };
struct mesh_ctx { struct hdr *M; size_t len; uint64_t client; uint32_t rows,arena,shared_pages; int fd; };
struct mesh_link_view { uint32_t peer,phase; char device[32]; uint64_t bandwidth; };
/* design/algorithm-sources.md#meshobserve */
int mesh_observe(const char *name,struct mesh_link_view *out,uint32_t capacity,uint32_t *node);
static inline uint32_t mesh_block_pages(const struct mesh_ctx *c){ return c->M->block; }
static inline void *mesh_page_address(struct mesh_ctx *c,uint32_t page){ return mesh_at(c->M,page); }
/* design/algorithm-sources.md#programcopy */
static inline struct mesh_range mesh_receive_range(struct mesh_ctx *c,uint32_t channel){
  uint32_t first=mesh_transfers(c->M,c->client,channel,MESH_RECEIVE)[0].pool;
  return (struct mesh_range){first,mesh_pools(c->M)[first/c->M->block].pages};
}
/* design/algorithm-sources.md#programcopy */
static inline uint32_t mesh_peer_channel(struct mesh_ctx *c,uint32_t peer,uint32_t queue){
  for(uint32_t i=0;i<c->M->links;i++)if(mesh_links(c->M)[i].peer==peer){
    if(queue<c->M->qps)return i*c->M->qps+queue;
    queue-=c->M->qps;
  }
  return MESH_ABSENT;
}
int mesh_attach(struct mesh_ctx *,const char *name);
int mesh_detach(struct mesh_ctx *);
/* design/algorithm-sources.md#programtensor */
void mesh_retire(struct hdr *,uint64_t client);

uint32_t mesh_rows_alloc(struct mesh_ctx *,uint32_t count);
uint32_t mesh_arena_alloc(struct mesh_ctx *,uint32_t pages,uint32_t align);
void mesh_backing_bind(struct mesh_ctx *,uint32_t first,uint32_t pages,uint32_t page);
void mesh_rows_release(struct mesh_ctx *,uint32_t first,uint32_t count);
void mesh_buffer_retain(struct hdr *,uint32_t row);
void mesh_retired_release(struct hdr *);
#endif
