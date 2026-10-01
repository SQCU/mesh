#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"
#include <errno.h>
/* design/pages-and-functions.md#what-the-page-table-is */
struct mesh_range { uint32_t first,count; };
/* A process's mapping of its node's region.  Any number of processes map it at once (operator, 2026-10-01: "ANY
   PROGRAM IN THE UNIVERSE"); none owns any of it.  `session` is the prepared session its transfers are bound in
   (MESH_ABSENT until its first binding claims one), paired with the peer's session of the same `key` (MESH_SESSION
   unset: 0, which a prepared session may not use).  `owned` lists what it allocated, freed at its mesh_detach (a
   process that exits without it leaves its pages allocated until the bridge starts again). */
struct mesh_range;
struct mesh_owned { struct mesh_range *rows,*pages; uint32_t nrows,npages,crows,cpages; uint64_t *rings; uint32_t nrings,crings; uint32_t sessions[MESH_SESSIONS]; uint32_t nsessions; };
struct mesh_ctx { struct hdr *M; size_t len; uint64_t key,send_off,send_bytes; uint32_t session,rows,arena,wire,shared_pages,row_cursor,wire_cursor,bulk_cursor; int fd; struct mesh_owned *owned; };
struct mesh_link_view { uint32_t peer,phase; char device[32]; uint64_t bandwidth; };
/* design/algorithm-sources.md#meshobserve */
int mesh_observe(const char *name,struct mesh_link_view *out,uint32_t capacity,uint32_t *node);
/* design/prepared-machine.md#M09 */
/* The two arena ranges.  [0,wire_pages) is the registered window and the only memory an SGE may
   name; [wire_pages,arena) is addressable and never registered.  An undeclared window is the whole
   arena, and then there is one range over it, exactly as before the split. */
static inline struct mesh_range mesh_arena_range(const struct hdr *m,int wire){
  uint32_t arena=mesh_arena_pages(m),window=m->wire_pages>arena?arena:m->wire_pages;
  uint32_t begin=wire||window==arena?m->zone_pages:window,end=wire?window:arena;
  return (struct mesh_range){begin,end-begin};
}
static inline uint32_t mesh_block_pages(const struct mesh_ctx *c){ return c->M->block; }
static inline void *mesh_page_address(struct mesh_ctx *c,uint32_t page){ return mesh_at(c->M,page); }
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
/* A prepared session slot claimed for the context (once): 0, or EBUSY where all MESH_SESSIONS are in use. */
int mesh_session_claim(struct mesh_ctx *);

uint32_t mesh_rows_alloc(struct mesh_ctx *,uint32_t count);
/* pages [first, first + count) of the arena (`wire`: the window's) given back */
void mesh_arena_release(struct mesh_ctx *,uint32_t first,uint32_t count);
uint32_t mesh_arena_claim(struct mesh_ctx *,uint32_t first,uint32_t pages,int wire);
/* design/prepared-machine.md#M09 */
uint32_t mesh_arena_alloc(struct mesh_ctx *,uint32_t pages,uint32_t align,int wire);
void mesh_backing_bind(struct mesh_ctx *,uint32_t first,uint32_t pages,uint32_t page,uint32_t index);
/* design/algorithm-sources.md#device-operands */
void mesh_device_bind(struct mesh_ctx *,uint32_t row,uint64_t address);
void mesh_rows_release(struct mesh_ctx *,uint32_t first,uint32_t count);
/* design/algorithm-sources.md#index-hand-off */
struct mesh_target *mesh_publish_bind(struct mesh_ctx *,uint32_t row,uint32_t queue);
/* design/algorithm-sources.md#programkernel_call */
uint32_t mesh_publication_prepare(struct hdr *,uint32_t row,struct prepared_publication *);
#endif
