#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"

struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride; };

/* the client's start, once mesh_plan has written this node's hops (design/page-queue-change-request.md, in place of
   the configured latch and its unbounded spin; defined in mesh-dataflow.c beside the map reader it reads): for every
   port whose cable joins this node to a node some hop names, at the plan's map serial, a wait, bounded by
   MESH_PLATFORM_NS from its start, until the port's data path has loaded the plan (struct mesh_port_info `prepared` ==
   hdr `plan`); then, every one loaded, the plan's struct mesh_state `executions` raised to `executions` (a
   compare-and-swap loop that never lowers it, release: the bridge serves executions t < it; design/disjoint-pages-
   revision-coordination.md, D2) and hdr `plan` woken, so no port serves an execution before every named port has
   loaded the plan (2026-11-04, the coordinator's rulings on owner 6's finding and review 15's finding 5: a port that
   carries units of the plan refuses it where its `executions` is above 0, a port with none loads it;
   mesh-dataflow.c).  A second start on the same plan waits as the first and raises it.  0 (also where no hop is
   written), EINVAL where no plan is written (hdr `state` 0), that port's `code` where its data path refused the plan
   (phase MESH_STOPPED), ETIMEDOUT past the bound, ESTALE where the map moved since the plan was made (plan again), or
   the map reader's error value (mesh-dataflow.h mesh_nodes_read), each with `executions` not raised */
int mesh_transfers_start(struct mesh_ctx *,uint64_t executions);

/* 2026-11-04, sub-assignment 5: mesh_plan_erased is removed, and with it its seq_cst fence pair with the ports' data
   path (design/disjoint-pages-change-request.md, Appendix C, C4): a new plan's state is in new pages, never the
   previous plan's, so nothing erases a plan or waits for one to be erased (C3.4).  What must stay: a port announces the
   generation it loads and reads `hops` and `plan` again, which the bridge's release reads (mesh-dataflow.h
   mesh_retired_release; the contract's D10) */

/* 2026-11-04, sub-assignment 5: mesh_section_create is removed (it named a section's rows from the program's claim at
   its call, and laid a `wire` 0 section out in the claim's local run, touching its pages): mesh_plan names every
   value's rows and pages, placed by its one allocator (design/disjoint-pages-change-request.md, Appendix C, C3.2 and
   C4; mesh-collective.h struct mesh_value `section`), and no program touches a page to fault it in.  What must stay: a
   section's rows are its plan's, and a section is read through them */
/* 2026-11-03, design/disjoint-pages-change-request.md, "Scope" (e2b_replay's page-sized ranges) and
   design/disjoint-pages-coordination.md, "The program's section" (sub-assignment 7): mesh_section_slice is gone with
   its one caller's page cut; a range is one declared transfer per destination.  What must stay: no section names
   another's pages */
/* design/algorithm-sources.md#programtensor */
/* copy `index`'s first byte (mesh-collective.h struct mesh_value: copy c at mesh_section_address(section, c)); NULL
   where its row has no page */
void *mesh_section_address(struct mesh_ctx *,struct mesh_section,uint32_t index);
void mesh_section_constant(struct mesh_ctx *,struct mesh_section);

#endif
