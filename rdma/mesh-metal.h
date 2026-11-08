#ifndef MESH_METAL_H
#define MESH_METAL_H
#include "mesh-call.h"
/* design/disjoint-pages-change-request.md, Appendix C, C3.1, C3.2, C3.4, C3.6 and C3.7 */
/* design/aot-compilation-change-request.md, appendix item 6 */
struct mesh_metal_transport { void *state; struct mesh_state *words; uint64_t offset,bytes; };
int mesh_metal_transport_create(struct mesh_ctx *,uint64_t plan,void *device,struct mesh_metal_transport *);
void mesh_metal_transport_destroy(struct mesh_metal_transport *);
/* L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, Math. Oper. Res. 22(3), 1997, §2 (`time`) */
struct mesh_metal_input { uint64_t phases,group,count; uint32_t pages,copies; double time; };
int mesh_metal_receive_prepare(struct mesh_ctx *,struct mesh_metal_transport *,uint32_t transfer,struct mesh_metal_input *);
struct mesh_metal_publication { uint64_t supplied; };
int mesh_metal_publication_prepare(struct mesh_ctx *,struct mesh_metal_transport *,uint32_t transfer,
                                   struct mesh_metal_publication *);
struct mesh_metal_wait { uint64_t count; uint32_t pages,copies,pass,padding; };
/* design/disjoint-pages-change-request.md, Appendix C, C3.1 and C3.3, and D.3.7;
   os/os_sync_wait_on_address.h:268 os_sync_wait_on_address_with_timeout */
/* MPI 4.1 §3.9 and p. 110; Linux v6.12 kernel/sched/completion.c:52-78, complete_all; MTL4CommandQueue.h:274-275 */
struct mesh_metal_waiter;
int mesh_metal_waits_start(struct mesh_metal_transport *,void *event,const struct mesh_metal_wait *,size_t waits,
                           uint32_t passes,struct mesh_metal_waiter **);
void mesh_metal_waits_stop(struct mesh_metal_waiter *);
int mesh_metal_waits_end(struct mesh_metal_waiter *);
int mesh_metal_status(struct mesh_metal_transport *,int cause,uint64_t *first);
_Static_assert(sizeof(struct mesh_metal_transport)==32,"mesh_metal_transport");
_Static_assert(sizeof(struct mesh_metal_input)==40 && offsetof(struct mesh_metal_input,group)==8 &&
  offsetof(struct mesh_metal_input,count)==16 && offsetof(struct mesh_metal_input,pages)==24 &&
  offsetof(struct mesh_metal_input,copies)==28 && offsetof(struct mesh_metal_input,time)==32,
  "mesh_metal_input: metal_recording.h struct MetalRemote");
_Static_assert(sizeof(struct mesh_metal_publication)==8,"mesh_metal_publication");
_Static_assert(sizeof(struct mesh_metal_wait)==24 && offsetof(struct mesh_metal_wait,pages)==8 &&
  offsetof(struct mesh_metal_wait,copies)==12 && offsetof(struct mesh_metal_wait,pass)==16,
  "mesh_metal_wait: metal_recording.h struct MetalWait");
#endif
