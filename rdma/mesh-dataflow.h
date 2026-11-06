#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"
#include "mesh-holders.h"
#include <errno.h>
struct mesh_range { uint32_t first,count; };
/* `clock`: L. Lamport, Time, clocks, and the ordering of events in a distributed system, CACM 21(7), 1978, IR1: "Each
   process Pi increments Ci between any two successive events" */
struct mesh_ctx { struct hdr *M; size_t len; uint64_t client; int fd; uint32_t padding; uint64_t clock,start; };
_Static_assert(sizeof(struct mesh_ctx)==48 && offsetof(struct mesh_ctx,fd)==24 && offsetof(struct mesh_ctx,padding)==28 &&
  offsetof(struct mesh_ctx,clock)==32 && offsetof(struct mesh_ctx,start)==40,"mesh_ctx: rdma/mesh.py Context");
static inline struct mesh_range mesh_arena_range(const struct hdr *m,int wire){
  uint32_t arena=mesh_arena_pages(m),window=m->wire_pages>arena?arena:m->wire_pages;
  uint32_t begin=wire||window==arena?0:window,end=wire?window:arena;
  return (struct mesh_range){begin,end-begin};
}
int mesh_attach(struct mesh_ctx *,const char *name);
int mesh_detach(struct mesh_ctx *);
/* P. E. McKenney and J. D. Slingwine, Read-copy update, PDCS 1998.
   Documentation/RCU/whatisRCU.rst (https://raw.githubusercontent.com/torvalds/linux/master/Documentation/RCU/whatisRCU.rst):
   "The removal phase removes references to data items within a data structure (possibly by replacing them with
   references to new versions of these data items), and can run concurrently with readers." */
void mesh_retire(struct hdr *,uint64_t client);
/* NVIDIA, CUDA Programming Guide §4.2.3.1, cudaGraphExecDestroy() */
void mesh_plan_retire(struct mesh_ctx *,uint64_t plan);
void mesh_backing_bind(struct mesh_ctx *,uint32_t first,uint32_t pages,uint32_t page);
struct mesh_view { void *address; size_t bytes; };
void mesh_view_release(struct mesh_view);
/* P. E. McKenney and J. D. Slingwine, Read-copy update, PDCS 1998; M. M. Michael, Hazard pointers: safe memory
   reclamation for lock-free objects, IEEE TPDS 15(6), 2004 */
uint32_t mesh_retired_release(struct hdr *);

#define MESH_NODES_MOST 32
/* `serial`, `sequencer`, `boot`: JGroups ViewId
   (https://raw.githubusercontent.com/belaban/JGroups/master/src/org/jgroups/ViewId.java), "Lamport time of the view"
   and "Address of the creator of this view" */
struct mesh_nodes { uint64_t serial; uint32_t count,self; uint32_t node[MESH_NODES_MOST];
                    uint64_t capacity[MESH_NODES_MOST][MESH_NODES_MOST];
                    uint32_t sequencer,padding; uint64_t boot; };
_Static_assert(offsetof(struct mesh_nodes,capacity)==144 && offsetof(struct mesh_nodes,sequencer)==8336 &&
  offsetof(struct mesh_nodes,boot)==8344 && sizeof(struct mesh_nodes)==8352,"mesh_nodes");
/* Chockler, Keidar and Vitenberg, Property 3.3: "no send events occur before the first view_chng event" */
int mesh_nodes_read(struct mesh_ctx *,struct mesh_nodes *);
#endif
