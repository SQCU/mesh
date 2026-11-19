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
struct hdr;
struct mesh_state;
/* design/disjoint-pages-change-request.md, Appendix C, C3.1, C3.2, C3.4, C3.6 and C3.7 */
/* design/aot-compilation-change-request.md, appendix item 6 */
/* design/compilation-composition-change-request.md item 6 (a delivered range read by another function's capture): the
   compiled program's plan states in one view, every binding->plan[f]'s state section inside [offset, offset +
   bytes) of region M, offset the lowest state's, bytes page-rounded, `state` its MTLBuffer (the one
   metal_recording.h state:buffer:status: buffer of every capture). Each function below takes the plan (state handle,
   mesh-collective.h struct mesh_planned_call `plan`) whose transfer it names, and every word offset it gives (phases,
   count, supplied, status) is relative to the one `offset`, whichever plan's word it is */
struct mesh_planned;
struct mesh_plan_binding;
struct mesh_metal_transport { void *state; uint64_t offset,bytes; };
int mesh_metal_transport_create(struct mesh_ctx *,const struct mesh_plan_binding *binding,void *device,
                                struct mesh_metal_transport *);
void mesh_metal_transport_destroy(struct mesh_metal_transport *);
/* L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, Math. Oper. Res. 22(3), 1997, §2 (`time`): a received range's
   words, copy 0's phase fields at `phases` (each copy `group` bytes on), copy 0's count word at `count`, its `pages`
   landings an execution, the plan's `copies` (a delivered range's words of its producer's execution e are copy
   e mod copies) */
struct mesh_metal_input { uint64_t phases,group,count; uint32_t pages,copies; double time; };
/* design/problems-of-14b9a757-change-request.md, third stops 7; rdma/mesh-collective.c mesh_host_sent: a sent range's
   completion counter of copy c at counter + 2 c, {count, error}; execution e's sends ended at count + error >=
   (e / copies + 1) sends */
struct mesh_metal_sent { void *counter; uint64_t sends,copies; };
/* S. Williams, A. Waterman and D. Patterson, Roofline, CACM 52(4), 2009, §3 ("created once per multicore computer");
   design/torch-capture-change-request.md, 2026-11-09 appendix: this node's GPU and CPU operands (mesh-graph.h struct
   mesh_graph_machine) from the configuration at `path` (etc/mesh-capacity.json; NULL: the etc/mesh-capacity.json of
   the mesh checkout whose rdma/libmesh.dylib this process loaded, dlfcn.h dladdr): models[sysctl hw.model].machines.gpu
   and .cpu, each with every field of the struct by its name, null for an operand not measured (0: absent,
   mesh_graph_actor_time's ENODATA), a record without `cores` (the CPU's) taking this node's sysctl hw.physicalcpu; ENOENT where the model or its machines are not recorded,
   EINVAL where a field is missing or not a number */
int mesh_metal_machines(const char *path,struct mesh_graph_machine *gpu,struct mesh_graph_machine *cpu);
/* the plan's status word relative to the view: a capture's state:buffer:status: offset (its own call's plan) */
int mesh_metal_status_prepare(struct mesh_ctx *,struct mesh_metal_transport *,uint64_t plan,uint64_t *status);
int mesh_metal_status(struct mesh_ctx *,uint64_t plan,int cause,uint64_t *first);
/* design/torch-capture-change-request.md item 3; MPI 4.1 §4.2.1 MPI_PRECV_INIT, §4.2.2 MPI_PARRIVED: one received range
   the dispatch reads. `input` its transport words, `generation` the producer execution e its consumer's execution e_b
   reads, e = (e_b / every) producer + ordinal ({1, 1, 0} the dispatch's own call), `page` the operand's bytes a
   partition, `partitions` their count, field[f] the partition transport field f lands: the transfer's planned RECEIVE
   row (mesh-collective.h struct mesh_planned_transfer) whose phase field is f names the partition its logical start
   is in (start / (block x pgsz)), a field no row names 0 (EINVAL for a row's field past the range's pages); `at`
   partition 0's byte in the storage its
   consumers' footprints are measured from (metal_ir.h: the storage root of the received value), `transfer` the
   declared transfer of planned call `owner`'s plan whose landing it is (metal_ir.h METAL_IR_ARRIVED) */
struct mesh_metal_range { struct mesh_metal_input input; struct mesh_graph_generation generation; uint32_t partitions;
                          uint64_t page; const uint32_t *field; uint64_t at; uint32_t transfer,owner;
                          const struct mesh_graph_access *extent; size_t extents; };
#define MESH_METAL_CURRENT_CALL UINT32_MAX
/* Two stages, before and after mesh_plan (design/compilation-composition-change-request.md, "Complete replacement
   workgroup": structure from the declared program, words from the plan). Declared: the range transfer `transfer` of
   call `owner` lands for call `call`'s value of `bytes` bytes, partition 0 at byte `at` of its storage, `calls` the
   program's calls as mesh_plan takes them: its partitions of block x pgsz bytes (mesh_plan's rows), generation {1, 1, 0}
   where owner is call, else {every of call, every of owner, ordinal of owner} (mesh-collective.h struct
   mesh_planned_call, from the same calls), input zero and field NULL. Prepared: the declared range's plan words and
   `field` (room for `room`; ERANGE with range->input set where its pages exceed it) from `planned`, the value placed
   at `section`; ENODATA where `planned` holds no planned rows (mesh_plan_load of an archive saved without them) */
int mesh_metal_range_declare(struct mesh_ctx *,const uint32_t *calls,uint32_t ncalls,uint32_t call,uint32_t owner,
                             uint32_t transfer,uint64_t bytes,uint64_t at,struct mesh_metal_range *range);
int mesh_metal_range_geometry(uint64_t page,const uint32_t *calls,uint32_t ncalls,uint32_t call,uint32_t owner,
                              uint32_t transfer,uint64_t bytes,uint64_t at,struct mesh_metal_range *range);
int mesh_metal_range_prepare(struct mesh_ctx *,struct mesh_metal_transport *,const struct mesh_plan_binding *binding,
                             struct mesh_section section,uint32_t *field,uint32_t room,struct mesh_metal_range *range);
int mesh_metal_range_fields(const struct mesh_planned *,struct mesh_metal_range *,uint32_t *field,uint32_t room);
/* R. Nasre, M. Burtscher and K. Pingali, IPDPS 2013; Arvind, R. S. Nikhil and K. K. Pingali, ACM TOPLAS 11(4), 1989;
   design/disjoint-pages-change-request.md C3.1, "Landings and cancellations both arrive on the count"; rdma/mesh-flow.c
   count_raise. The readiness obligation of a dispatch reading `ranges` received ranges, every quantity static but the
   view binding (mesh_metal_transport `state`) and the dispatch's execution e_b (its step word). Threadgroup g, linear
   index l = g_x + groups_x (g_y + groups_y g_z), reads exactly the transport fields of range d its class's membership
   names: member[0] the entries' first word E, entries [member[1 + c ranges + d], member[2 + c ranges + d]) of class c =
   class_of[l] and range d, entry k the pair {member[E + 2 k], member[E + 2 k + 1]}: a phase word w (fields 16 w ..
   16 w + 15) and its fields' mask (bit 2 (f mod 16) for each field f read). For range d at
   e: r = e / copies, c = e mod copies; field f is the 2-bit field at bit 2 (f mod 16) of the 32-bit word at view byte
   phases + c group + 4 (f / 16); f has landed where it is mesh.h mesh_phase(r + 1) and is ahead where it is neither
   that nor mesh_phase(r); the range is due where its count word, view byte count + 8 c, is at least (r + 1) pages.
   A first pass marks each threadgroup ready (all its fields landed) or deferred and runs the ready ones; a later pass
   over the deferred groups repeats the test until ready, or fails where a field is ahead or the range is due while a
   field has not landed. A failure adds 1 to the status word's `ahead` (a field ahead) or `behind` (otherwise) and
   raises its `first` to the 64-bit maximum with ~e_b (mesh.h struct mesh_status at view byte `status`) and ends the
   group. A ready group's reads follow a system-scope sequentially consistent fence after its last test */
struct mesh_metal_readiness { uint32_t ranges,classes,groups,members; const struct mesh_metal_range *range;
                              uint32_t *member,*class_of; uint64_t status; };
/* A numerical read uses its storage's byte coordinates. A JOIN uses all of its declared fields. */
struct mesh_metal_access { uint32_t storage; const struct mesh_graph_access_map *map;
                           const struct mesh_graph_footprint *footprint; uint64_t offset; };
int mesh_metal_mask(const uint32_t groups[3],const uint32_t threads[3],size_t accesses,
                    const struct mesh_metal_access *access,const struct mesh_graph_term *terms,
                    const uint32_t *storage,const uint8_t *whole,struct mesh_metal_readiness *readiness);
/* design/disjoint-pages-change-request.md C3.1, C3.2 and C3.7; PTX ISA §8.8; S. Flur et al., POPL 2017: the release
   of a published range's execution e, after all of e's stores to it: a system-scope sequentially consistent fence;
   where the status word's behind and ahead are both 0, the supplied word (view byte `supplied`, the transfer's first
   target word) raised to the 64-bit maximum with e + 1; the same fence. The range's GPU stores are system-coherent; each writing
   threadgroup of every dispatch writing the range adds 1 to one arrival counter after its stores, the one whose
   addition completes `total` (the lowering's count of those threadgroups, a reader's first and later pass counted
   once) resets the counter to 0 and releases. `transfer` the declared transfer of the writer's own call
   (metal_ir.h METAL_IR_PUBLISH for a CPU writer) */
struct mesh_metal_release { uint64_t supplied,status,total; uint32_t transfer,owner; };
struct mesh_metal_origin { uint64_t word; uint32_t copies,cuts; };
int mesh_metal_origin_slots_prepare(struct mesh_ctx *,struct mesh_metal_transport *,const struct mesh_plan_binding *,
                                    uint32_t call,struct mesh_metal_release *,struct mesh_metal_origin *);
/* declared before mesh_plan as {.transfer,.owner} (one release group of the value: its representative transfer and
   its owner call, `total` 0 until the lowering counts it); prepared after it: the supplied and status words of that
   call's plan, in place (transfer and owner kept) */
int mesh_metal_release_prepare(struct mesh_ctx *,struct mesh_metal_transport *,const struct mesh_plan_binding *binding,
                               uint32_t call,struct mesh_metal_release *release);
/* mesh_plan_cut: the first source cut's supplied word, followed by one u64 word per cut. */
int mesh_metal_origin_prepare(struct mesh_ctx *,struct mesh_metal_transport *,const struct mesh_plan_binding *,
                              uint32_t call,struct mesh_metal_release *,uint64_t *origin);
/* design/problems-of-14b9a757-change-request.md, third stops 7; MPI 4.1 §3.7.3: the lifetime obligation of a range
   planned call `call`'s transfer `transfer` sends from this node (its sent ranges and the received ranges it forwards):
   no writer of its copy for execution e runs before execution e - copies's sends ended (`sent`); `extent` the bytes
   those sends read of a copy, the same of every copy (MPI 4.1 §3.7.3: a send reads its source from its release until
   its completion): the transfer's planned SEND rows (mesh-collective.h struct mesh_planned_transfer), each one send's
   logical bytes of the value, `key`, `vertex` and `write` 0, borrowed from `planned` until mesh_planned_free;
   sent.sends 0: none */
struct mesh_metal_sending { struct mesh_metal_sent sent; const struct mesh_graph_access *extent; size_t extents;
                            uint32_t transfer,padding; };
void mesh_metal_input_words(struct hdr *,const struct mesh_state *,uint32_t transfer,int64_t displacement,
                            struct mesh_metal_input *);
void mesh_metal_release_words(struct hdr *,const struct mesh_state *,int64_t displacement,struct mesh_metal_release *,
                              struct mesh_metal_origin *,uint64_t *legacy);
int mesh_metal_sending_rows(struct hdr *,const struct mesh_state *,const struct mesh_planned *,uint32_t call,
                            uint32_t transfer,uintptr_t address,struct mesh_metal_sending *);
/* ENODATA where `planned` holds no planned rows (mesh_plan_load of an archive saved without them) */
int mesh_metal_sending_prepare(struct mesh_ctx *,struct mesh_metal_transport *,const struct mesh_plan_binding *binding,
                               uint32_t call,uint32_t transfer,struct mesh_metal_sending *sending);
_Static_assert(sizeof(struct mesh_metal_range)==104 &&
  offsetof(struct mesh_metal_range,generation)==40 && offsetof(struct mesh_metal_range,page)==56 &&
  offsetof(struct mesh_metal_range,at)==72 && offsetof(struct mesh_metal_range,transfer)==80 &&
  sizeof(struct mesh_metal_readiness)==48 && sizeof(struct mesh_metal_release)==32 &&
  sizeof(struct mesh_metal_sending)==48 && offsetof(struct mesh_metal_sending,extent)==24,"mesh-metal.h records");
_Static_assert(sizeof(struct mesh_metal_transport)==24,"mesh_metal_transport: rdma/mesh_compile.py Transport");
_Static_assert(sizeof(struct mesh_metal_input)==40 && offsetof(struct mesh_metal_input,group)==8 &&
  offsetof(struct mesh_metal_input,count)==16 && offsetof(struct mesh_metal_input,pages)==24 &&
  offsetof(struct mesh_metal_input,copies)==28 && offsetof(struct mesh_metal_input,time)==32,
  "mesh_metal_input: rdma/mesh_compile.py Remote");
_Static_assert(sizeof(struct mesh_metal_sent)==24,"mesh_metal_sent: rdma/mesh_compile.py MetalSent");
#ifdef __cplusplus
}
#endif
#endif
