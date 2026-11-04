#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"

struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride; };

/* the client's start, once mesh_plan has written this node's hops (design/page-queue-change-request.md, in place of
   the configured latch and its unbounded spin; defined in mesh-dataflow.c beside the map reader it reads): for each
   adjacent node some hop names, at the plan's map serial, a wait, bounded by MESH_PLATFORM_NS from its start, until
   the bridge has loaded the plan toward it (struct mesh_port_info `prepared` == hdr `plan`); then, every one loaded,
   the plan's struct mesh_state `executions` raised to `executions` (a compare-and-swap loop that never lowers it,
   release: the bridge serves executions t < it; design/disjoint-pages-revision-coordination.md, D2) and hdr `plan`
   woken, so no execution is served before the bridge holds the plan toward every adjacent node it names (2026-11-04,
   the coordinator's rulings on owner 6's finding and review 15's finding 5: the bridge refuses a plan it loads after
   its `executions` has risen, where it carries pages of it; mesh-dataflow.c).  Then a seq_cst fence and hdr
   `map_serial` read: where it is not the plan's serial, ESTALE, the raise standing (2026-11-04, sub-assignment 1:
   design/disjoint-pages-change-request.md, D.3.6; design/disjoint-pages-rulings-coordination.md, R3).  A plan stands
   only while its map does: where the map moves, the bridge moves the serial, fences and reads `executions`, and cancels
   every receive of the plan not landed below it (each an arrival on its count, no phase written); the start raises
   `executions`, fences and reads the serial; so at least one sees the other (D10's announcement, fence and re-read;
   K. Fraser, Practical lock-freedom, 2004).  Either the cancellation covers the raised executions, whose readers then
   end at once with an error, or the start returns ESTALE and the program runs none of them and plans again.  The raise
   left standing reads no execution as landed that has not: `executions` only rises, the program reads and releases
   nothing of an execution it does not run, and the plan's end (the next plan, or the program's retirement) cancels
   every receive of it not landed.  It does not end without effect: the bridge has already set the raised executions'
   receives, which end with the plan as any cancelled run's do, by the bridge's own means (review 8's finding 1;
   review 10's finding 5).  A second start on the same plan waits as the first and raises it.  0 (also where no hop
   is written), EINVAL where no plan is written (hdr `state` 0), the bridge's code where it refused the plan toward an
   adjacent node (phase MESH_STOPPED), ETIMEDOUT past the bound, ESTALE where the map moved since the plan was made
   (plan again), or the map reader's error value (mesh-dataflow.h mesh_nodes_read): each with `executions` not
   raised, but the ESTALE after the raise, with it raised */
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
