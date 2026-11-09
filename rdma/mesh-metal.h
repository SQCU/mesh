#ifndef MESH_METAL_H
#define MESH_METAL_H
#include "mesh-section.h"
#include "mesh-graph.h"
#ifdef __cplusplus
extern "C" {
#endif
/* rdma/mesh-section.h, TP-1: mesh-call.h's mesh.h does not compile as C++, so the context is declared, not included,
   and C++ compilers (metal_ir.h, metal_lower.mm) include this header */
struct mesh_ctx;
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
   sends at this node read of a copy of `section`, the same of every copy, in the capture's members (metal_recording.h
   sent:operand:counter:extents:): one mesh-graph.h struct mesh_graph_access a SEND hop (mesh.h struct mesh_hop) whose
   count word is the transfer's copy-0 completion word, `vertex` and `write` 0. Row q is the copy-0 row q <
   section.stride whose block (block x pgsz bytes) holds the hop. `members` NULL (`count` 0): one buffer holds the
   copy, row q at q x block: `key` 0, [q x block + start, q x block + end); else member n holds row members[n] alone:
   `key` n, [start, end) within its block. ERANGE with *made the number needed where it exceeds `room`; EINVAL where a
   hop reads no copy-0 row or a row no member holds */
int mesh_metal_sent_extents(struct mesh_ctx *,struct mesh_metal_transport *,uint64_t plan,uint32_t transfer,
                            struct mesh_section section,const uint32_t *members,uint32_t count,
                            struct mesh_graph_access *extents,size_t room,size_t *made);
/* the plan's status word relative to the view: a capture's state:buffer:status: offset (its own call's plan) */
int mesh_metal_status_prepare(struct mesh_ctx *,struct mesh_metal_transport *,uint64_t plan,uint64_t *status);
int mesh_metal_status(struct mesh_ctx *,uint64_t plan,int cause,uint64_t *first);
/* design/torch-capture-change-request.md item 3; MPI 4.1 §4.2.1 MPI_PRECV_INIT, §4.2.2 MPI_PARRIVED: one received range
   the dispatch reads. `input` its transport words (mesh_metal_receive_prepare), `generation` the producer execution e
   its consumer's execution e_b reads, e = (e_b / every) producer + ordinal ({1, 1, 0} the dispatch's own call),
   `page` the operand's bytes a partition, `partitions` their count, field[f] the partition transport field f lands
   (mesh_metal_receive_fields; UINT32_MAX a field of another buffer), `at` partition 0's byte in the storage its
   consumers' footprints are measured from (metal_ir.h: the storage root of the received value), `transfer` the
   declared transfer of planned call `owner`'s plan whose landing it is (metal_recording.h METAL_NATIVE_ARRIVED) */
struct mesh_metal_range { struct mesh_metal_input input; struct mesh_graph_generation generation; uint32_t partitions;
                          uint64_t page; const uint32_t *field; uint64_t at; uint32_t transfer,owner; };
/* Two stages, before and after mesh_plan (design/compilation-composition-change-request.md, "Complete replacement
   workgroup": structure from the declared program, words from the plan). Declared: the range transfer `transfer` of
   call `owner` lands for call `call`'s value of `bytes` bytes, partition 0 at byte `at` of its storage, `calls` the
   program's calls as mesh_plan takes them: its partitions of block x pgsz bytes (mesh_plan's rows), generation {1, 1, 0}
   where owner is call, else {every of call, every of owner, ordinal of owner} (mesh-collective.h struct
   mesh_planned_call, from the same calls), input zero and field NULL. Prepared: the declared range's plan words and
   `field` (room for `room`; ERANGE with range->input set where its pages exceed it) from `planned`, the value placed
   at `section` */
int mesh_metal_range_declare(struct mesh_ctx *,const uint32_t *calls,uint32_t ncalls,uint32_t call,uint32_t owner,
                             uint32_t transfer,uint64_t bytes,uint64_t at,struct mesh_metal_range *range);
int mesh_metal_range_prepare(struct mesh_ctx *,struct mesh_metal_transport *,const struct mesh_planned *planned,
                             struct mesh_section section,uint32_t *field,uint32_t room,struct mesh_metal_range *range);
/* R. Nasre, M. Burtscher and K. Pingali, IPDPS 2013; Arvind, R. S. Nikhil and K. K. Pingali, ACM TOPLAS 11(4), 1989;
   design/disjoint-pages-change-request.md C3.1, "Landings and cancellations both arrive on the count"; rdma/mesh-flow.c
   count_raise. The readiness obligation of a dispatch reading `ranges` received ranges, every quantity static but the
   view binding (mesh_metal_transport `state`) and the dispatch's execution e_b (its step word). Threadgroup g, linear
   index l = g_x + groups_x (g_y + groups_y g_z), reads transport fields [span[2 (class_of[l] ranges + d)],
   span[... + 1]) of range d (equal ends: none). For range d at
   e: r = e / copies, c = e mod copies; field f is the 2-bit field at bit 2 (f mod 16) of the 32-bit word at view byte
   phases + c group + 4 (f / 16); f has landed where it is mesh.h mesh_phase(r + 1) and is ahead where it is neither
   that nor mesh_phase(r); the range is due where its count word, view byte count + 8 c, is at least (r + 1) pages.
   A first pass marks each threadgroup ready (all its fields landed) or deferred and runs the ready ones; a later pass
   over the deferred groups repeats the test until ready, or fails where a field is ahead or the range is due while a
   field has not landed. A failure adds 1 to the status word's `ahead` (a field ahead) or `behind` (otherwise) and
   raises its `first` to the 64-bit maximum with ~e_b (mesh.h struct mesh_status at view byte `status`) and ends the
   group. A ready group's reads follow a system-scope sequentially consistent fence after its last test */
struct mesh_metal_readiness { uint32_t ranges,classes,groups,padding; const struct mesh_metal_range *range;
                              uint32_t *span,*class_of; uint64_t status; };
/* `readiness` names its ranges, groups (the grid's threadgroups, groups[0] groups[1] groups[2]), status and room for
   span (2 groups ranges entries) and class_of (groups); the classes, class_of and spans from the dispatch's footprints
   (mesh-graph.h struct mesh_graph_footprint, their terms in `term`, evaluated at each threadgroup), footprint k's
   binding a view of range[k] (UINT32_MAX: of no received range): each class one distinct row of the partitions a
   threadgroup reaches (all of a range no footprint names, since the dispatch reads every range it lists), its span of
   range d the least and one past the greatest field whose partition is within that row's bounds of d */
int mesh_metal_mask(const uint32_t groups[3],size_t footprints,const struct mesh_graph_footprint *footprint,
                    const struct mesh_graph_term *term,const uint32_t *range,struct mesh_metal_readiness *readiness);
/* design/disjoint-pages-change-request.md C3.1, C3.2 and C3.7; PTX ISA §8.8; S. Flur et al., POPL 2017: the release
   of a published range's execution e, after all of e's stores to it: a system-scope sequentially consistent fence;
   where the status word's behind and ahead are both 0, the supplied word (view byte `supplied`, mesh_metal_publication)
   raised to the 64-bit maximum with e + 1; the same fence. The range's GPU stores are system-coherent; each writing
   threadgroup of every dispatch writing the range adds 1 to one arrival counter after its stores, the one whose
   addition completes `total` (the lowering's count of those threadgroups, a reader's first and later pass counted
   once) resets the counter to 0 and releases. `transfer` the declared transfer of the writer's own call
   (metal_recording.h METAL_NATIVE_PUBLISH for a CPU writer) */
struct mesh_metal_release { uint64_t supplied,status,total; uint32_t transfer,padding; };
/* declared before mesh_plan as {.transfer} (planned call `call`'s transfer, `total` 0 until the lowering counts it);
   prepared after it: the supplied and status words of that call's plan */
int mesh_metal_release_prepare(struct mesh_ctx *,struct mesh_metal_transport *,const struct mesh_planned *planned,
                               uint32_t call,struct mesh_metal_release *release);
/* design/problems-of-14b9a757-change-request.md, third stops 7; MPI 4.1 §3.7.3: the lifetime obligation of a range
   planned call `call`'s transfer `transfer` sends from this node (its sent ranges and the received ranges it forwards):
   no writer of its copy for execution e runs before execution e - copies's sends ended (`sent`); `extent` the bytes
   those sends read (mesh_metal_sent_extents of one buffer, from the value's storage start); sent.sends 0: none */
struct mesh_metal_sending { struct mesh_metal_sent sent; const struct mesh_graph_access *extent; size_t extents;
                            uint32_t transfer,padding; };
/* `extent` room for `room` (ERANGE with sending->extents the number needed); the value placed at `section` */
int mesh_metal_sending_prepare(struct mesh_ctx *,struct mesh_metal_transport *,const struct mesh_planned *planned,
                               uint32_t call,uint32_t transfer,struct mesh_section section,struct mesh_graph_access *extent,
                               size_t room,struct mesh_metal_sending *sending);
_Static_assert(sizeof(struct mesh_metal_range)==88 &&
  offsetof(struct mesh_metal_range,generation)==40 && offsetof(struct mesh_metal_range,page)==56 &&
  offsetof(struct mesh_metal_range,at)==72 && offsetof(struct mesh_metal_range,transfer)==80 &&
  sizeof(struct mesh_metal_readiness)==48 && sizeof(struct mesh_metal_release)==32 &&
  sizeof(struct mesh_metal_sending)==48 && offsetof(struct mesh_metal_sending,extent)==24,"mesh-metal.h records");
_Static_assert(sizeof(struct mesh_metal_transport)==24,"mesh_metal_transport: rdma/mesh_compile.py Transport");
_Static_assert(sizeof(struct mesh_metal_input)==40 && offsetof(struct mesh_metal_input,group)==8 &&
  offsetof(struct mesh_metal_input,count)==16 && offsetof(struct mesh_metal_input,pages)==24 &&
  offsetof(struct mesh_metal_input,copies)==28 && offsetof(struct mesh_metal_input,time)==32,
  "mesh_metal_input: rdma/mesh_compile.py Remote");
_Static_assert(sizeof(struct mesh_metal_publication)==8,"mesh_metal_publication");
_Static_assert(sizeof(struct mesh_metal_sent)==24,"mesh_metal_sent: rdma/mesh_compile.py MetalSent");
#ifdef __cplusplus
}
#endif
#endif
