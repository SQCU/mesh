#ifndef MESH_METAL_H
#define MESH_METAL_H
#include "mesh-call.h"
#include "mesh-graph.h"
/* design/disjoint-pages-change-request.md, Appendix C, C3.1, C3.2, C3.4, C3.6 and C3.7 */
/* design/aot-compilation-change-request.md, appendix item 6 */
/* design/compilation-composition-change-request.md item 6 (a delivered range read by another function's capture): the
   compiled program's plan states in one view, every planned->call[j].plan's state section inside [offset, offset +
   bytes) of region M, offset the lowest state's, bytes page-rounded, `state` its MTLBuffer (the one
   metal_recording.h state:buffer:status: buffer of every capture). Each function below takes the plan (state handle,
   mesh-collective.h struct mesh_planned_call `plan`) whose transfer it names, and every word offset it gives (phases,
   count, supplied, status) is relative to the one `offset`, whichever plan's word it is */
struct mesh_planned;
struct mesh_metal_transport { void *state; uint64_t offset,bytes; };
int mesh_metal_transport_create(struct mesh_ctx *,const struct mesh_planned *planned,void *device,
                                struct mesh_metal_transport *);
void mesh_metal_transport_destroy(struct mesh_metal_transport *);
/* L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, Math. Oper. Res. 22(3), 1997, §2 (`time`) */
struct mesh_metal_input { uint64_t phases,group,count; uint32_t pages,copies; double time; };
/* `copies` the given plan's: a delivered range's words of its producer's execution e are copy e mod copies */
int mesh_metal_receive_prepare(struct mesh_ctx *,struct mesh_metal_transport *,uint64_t plan,uint32_t transfer,
                               struct mesh_metal_input *);
/* design/torch-capture-change-request.md item 3 (metal_recording.h received:operand:remote:fields:): the received
   range `input`'s fields, `input->pages` of them: field (word - (offset + phases)) / 8 x 32 + field / 2 of each RECEIVE
   hop (mesh.h struct mesh_hop) whose count word is the range's names the copy-0 row q < section.stride of `section`
   whose block (block x pgsz bytes from mesh-call.h mesh_section_address(section, q)) holds the hop's landed byte
   data_off + page x pgsz + offset; a field no hop names is 0; EINVAL where a hop lands in no copy-0 row */
int mesh_metal_receive_fields(struct mesh_ctx *,struct mesh_metal_transport *,uint64_t plan,
                              const struct mesh_metal_input *,struct mesh_section section,uint32_t *fields);
struct mesh_metal_publication { uint64_t supplied; };
int mesh_metal_publication_prepare(struct mesh_ctx *,struct mesh_metal_transport *,uint64_t plan,uint32_t transfer,
                                   struct mesh_metal_publication *);
/* design/problems-of-14b9a757-change-request.md, third stops 7; rdma/mesh-collective.c mesh_host_sent: a sent range's
   completion counter of copy c at counter + 2 c, {count, error}; execution e's sends ended at count + error >=
   (e / copies + 1) sends */
struct mesh_metal_sent { void *counter; uint64_t sends,copies; };
int mesh_metal_sent_prepare(struct mesh_ctx *,struct mesh_metal_transport *,uint64_t plan,uint32_t transfer,
                            struct mesh_metal_sent *);
/* MPI 4.1 §3.7.3 (a send reads its source from its release until its completion): the bytes transfer `transfer`'s
   sends at this node read of copy 0 of `section`, the same of every copy: one mesh-graph.h struct mesh_graph_access a
   SEND hop (mesh.h struct mesh_hop) whose count word is the transfer's copy-0 completion word, `key` the copy-0 row
   q < section.stride whose block holds it, [start, end) its bytes within that block, `vertex` and `write` 0;
   ERANGE with *count the number needed where it exceeds `room`; EINVAL where a hop reads no copy-0 row */
int mesh_metal_sent_extents(struct mesh_ctx *,struct mesh_metal_transport *,uint64_t plan,uint32_t transfer,
                            struct mesh_section section,struct mesh_graph_access *extents,size_t room,size_t *count);
/* the plan's status word relative to the view: a capture's state:buffer:status: offset (its own call's plan) */
int mesh_metal_status_prepare(struct mesh_ctx *,struct mesh_metal_transport *,uint64_t plan,uint64_t *status);
int mesh_metal_status(struct mesh_ctx *,uint64_t plan,int cause,uint64_t *first);
_Static_assert(sizeof(struct mesh_metal_transport)==24,"mesh_metal_transport: rdma/mesh_compile.py Transport");
_Static_assert(sizeof(struct mesh_metal_input)==40 && offsetof(struct mesh_metal_input,group)==8 &&
  offsetof(struct mesh_metal_input,count)==16 && offsetof(struct mesh_metal_input,pages)==24 &&
  offsetof(struct mesh_metal_input,copies)==28 && offsetof(struct mesh_metal_input,time)==32,
  "mesh_metal_input: metal_recording.h struct MetalRemote");
_Static_assert(sizeof(struct mesh_metal_publication)==8,"mesh_metal_publication");
_Static_assert(sizeof(struct mesh_metal_sent)==24,"mesh_metal_sent: metal_recording.h struct MetalSent");
#endif
