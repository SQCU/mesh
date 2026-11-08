#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"
#include "mesh-holders.h"
#include <errno.h>
/* design/pages-and-functions.md#what-the-page-table-is */
struct mesh_range { uint32_t first,count; };
/* design/disjoint-pages-change-request.md, Appendix C, C3.4; design/plan-to-bridge-change-request.md R8;
   design/aot-compilation-change-request.md A7 and appendix item 6;
   L. Lamport, Time, clocks, and the ordering of events in a distributed system, CACM 21(7), 1978, IR1 */
struct mesh_ctx { struct hdr *M; size_t len; uint64_t client; int fd; uint32_t padding; uint64_t clock,start; };
_Static_assert(sizeof(struct mesh_ctx)==48 && offsetof(struct mesh_ctx,fd)==24 && offsetof(struct mesh_ctx,padding)==28 &&
  offsetof(struct mesh_ctx,clock)==32 && offsetof(struct mesh_ctx,start)==40,"mesh_ctx: rdma/mesh.py Context");
/* design/prepared-machine.md#M09 */
static inline struct mesh_range mesh_arena_range(const struct hdr *m,int wire){
  uint32_t arena=mesh_arena_pages(m),window=m->wire_pages>arena?arena:m->wire_pages;
  uint32_t begin=wire||window==arena?0:window,end=wire?window:arena;
  return (struct mesh_range){begin,end-begin};
}
static inline uint32_t mesh_block_pages(const struct mesh_ctx *c){ return c->M->block; }
static inline void *mesh_page_address(struct mesh_ctx *c,uint32_t page){ return mesh_at(c->M,page); }
/* design/plan-to-bridge-change-request.md R8 */
int mesh_attach(struct mesh_ctx *,const char *name);
int mesh_detach(struct mesh_ctx *);
/* design/algorithm-sources.md#programtensor */
/* design/plan-to-bridge-change-request.md R8 */
void mesh_retire(struct hdr *,uint64_t client);
/* design/aot-compilation-change-request.md, appendix item 6;
   NVIDIA, CUDA Programming Guide §4.2.3.1, cudaGraphExecDestroy() */
void mesh_plan_retire(struct mesh_ctx *,uint64_t plan);
void mesh_backing_bind(struct mesh_ctx *,uint32_t first,uint32_t pages,uint32_t page,uint32_t index);
/* design/torch-capture-change-request.md item 6 */
struct mesh_view { void *address; size_t bytes; };
/* design/algorithm-sources.md#device-operands */
void mesh_device_bind(struct mesh_ctx *,uint32_t row,uint64_t address);
/* P. E. McKenney and J. D. Slingwine, Read-copy update, 1998; K. Fraser, Practical lock-freedom, 2004;
   M. M. Michael, IEEE TPDS 15(6), 2004; os/os_sync_wait_on_address.h:370 os_sync_wake_by_address_all */
uint32_t mesh_retired_release(struct hdr *);

/* design/page-queue-change-request.md, "What the collective decides, and what the bridge decides";
   design/aot-compilation-change-request.md A7 */
#define MESH_NODES_MOST 32
/* JGroups ViewId.java */
struct mesh_nodes { uint64_t serial; uint32_t count,self; uint32_t node[MESH_NODES_MOST];
                    uint64_t capacity[MESH_NODES_MOST][MESH_NODES_MOST];
                    uint32_t sequencer,padding; uint64_t boot; };
_Static_assert(offsetof(struct mesh_nodes,capacity)==144 && offsetof(struct mesh_nodes,sequencer)==8336 &&
  offsetof(struct mesh_nodes,boot)==8344 && sizeof(struct mesh_nodes)==8352,"mesh_nodes");
/* design/aot-compilation-change-request.md A7 */
int mesh_nodes_read(struct mesh_ctx *,struct mesh_nodes *);
#endif
