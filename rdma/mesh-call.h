#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"

struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride; };

/* the client's start, once mesh_plan has written this node's hops (design/page-queue-change-request.md, in place of
   the configured latch and its unbounded spin; defined in mesh-dataflow.c beside the map reader it reads): for every port whose cable joins this node to a node some hop names, at the plan's
   map serial, a wait, bounded by MESH_PLATFORM_NS from its start, until the port's data path has loaded the plan
   (struct mesh_port_info `prepared` == hdr `plan`); 0 (also where no hop is written), that port's `code` where its
   data path refused the plan (phase MESH_STOPPED), ETIMEDOUT past the bound, ESTALE where the map moved since the plan
   was made (plan again), or the map reader's error value (mesh-dataflow.h mesh_nodes_read) */
int mesh_transfers_start(struct mesh_ctx *);

/* the plan's words free of the plan `generation` (design/disjoint-pages-change-request.md, Appendix A, A3; in place of
   the retention of the plan's words section: the supplied words and the landed list are areas of the layout, hdr
   `word_off` and `landed_off`, which the next plan reuses; defined in mesh-dataflow.c beside mesh_transfers_start): 0
   once no port's `prepared` (struct mesh_port_info) holds `generation`, at once where `generation` is 0; ETIMEDOUT
   past MESH_PLATFORM_NS from its start */
int mesh_plan_erased(struct mesh_ctx *,uint64_t generation);

/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M09 */
/* `count` instances of `bytes` named in the program's claim (mesh-dataflow.h struct mesh_ctx), the next `count` x
   `stride` of its rows, `stride` = `pages` / block rows an instance (`pages` = bytes rounded up to whole blocks),
   instance s at rows `first` + s x `stride`, every row named clean (no page, device or publication).  `wire` 1: a value
   a peer reads or writes, no page bound (an entry's `mapping` MESH_ABSENT, `address` 0): the plan's allocator assigns
   its pages inside the claim's `window` (mesh-collective.h mesh_plan); `wire` 0: a value no plan names, laid out in the
   claim's `local` run in creation order, its pages touched here.  0, EINVAL where `bytes` or `count` is 0, ENOMEM past
   the claim's rows or `local` run */
int mesh_section_create(struct mesh_ctx *,size_t bytes,uint32_t count,int wire,struct mesh_section *);
/* 2026-11-03, design/disjoint-pages-change-request.md, "Scope" (e2b_replay's page-sized ranges) and
   design/disjoint-pages-coordination.md, "The program's section" (sub-assignment 7): mesh_section_slice is gone with
   its one caller's page cut; a range is one declared transfer per destination.  What must stay: no section names
   another's pages */
/* design/algorithm-sources.md#programtensor */
/* instance `index`'s first byte; NULL where its row has no page (a `wire` 1 section before its plan) */
void *mesh_section_address(struct mesh_ctx *,struct mesh_section,uint32_t index);
void mesh_section_constant(struct mesh_ctx *,struct mesh_section);

#endif
