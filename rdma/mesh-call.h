#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"

struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride; };

/* the client's start, by the plan `plan` (its state section's region offset, mesh-collective.h mesh_plan's `*plan`;
   design/compilation-coordination.md, "The region" and "The appendix's revisions", N2; defined in mesh-dataflow.c
   beside the map reader it reads): for each adjacent node
   some hop of the plan names (mesh.h mesh_hops), at the plan's map serial, a wait, bounded by MESH_PLATFORM_NS from its
   start, until the port toward it has its bit of the plan's `held` or a `refused` code (that code returned); then
   the plan's struct mesh_state `executions` raised to `executions` (a compare-and-swap loop that never lowers it,
   release: the bridge serves executions t < it; design/disjoint-pages-revision-coordination.md, D2), which is the
   program's submission of those function executions to its node's bridge (design/aot-compilation-change-request.md
   A7: "A program submits each of its function executions to its node's bridge"; design/compilation-coordination.md,
   Q1); then hdr `plans` raised and woken, so the bridge sees the submission.  No execution is served before the bridge
   holds the plan toward every adjacent node it names (2026-11-04, the coordinator's rulings on owner 6's finding and
   review 15's finding 5: the bridge refuses a plan it loads after its `executions` has risen, where it carries pages
   of it).  Then a seq_cst fence and hdr `map_serial` read: where it is not the plan's `serial`, ESTALE, the raise
   standing (2026-11-04, sub-assignment 1: design/disjoint-pages-change-request.md, D.3.6;
   design/disjoint-pages-rulings-coordination.md, R3).  A plan stands only while its map does: where the map moves, the
   bridge moves the serial, fences and reads `executions`, and cancels every receive of the plan not landed below it
   (each an arrival on its count, no phase written); the start raises `executions`, fences and reads the serial; so at
   least one sees the other (D10's announcement, fence and re-read; K. Fraser, Practical lock-freedom, 2004).  Either
   the cancellation covers the raised executions, whose readers then end at once with an error, or the start returns
   ESTALE and the program runs none of them and plans again.  The raise left standing reads no execution as landed
   that has not: `executions` only rises, the program reads and releases nothing of an execution it does not run, and
   the plan's end (its claim's close) cancels every receive of it not landed.  It does not end without effect: the
   bridge has already set the raised executions' receives, which end with the plan as any cancelled run's do, by the
   bridge's own means (review 8's finding 1; review 10's finding 5).  A second start on the same plan waits as the
   first and raises it.  One execution is one execution of the plan's compiled function: torch's call t raises it to
   t + 1, e2b's run to its steps at once (design/aot-compilation-change-request.md, appendix item 6: "have each call
   launch one execution of its function"; N4).  0 (also where the plan has no hop here), EINVAL where `plan` is 0, not
   page-aligned, before hdr `data_off` or its struct past the region (mesh-metal.h mesh_metal_transport_create's
   tests), names no plan standing at a claim head of this client (mesh-dataflow.h mesh_claim_take's record and
   mesh_head_state: owner the client, `closed` 0, publication `state` `plan`; N2), or its entries lie outside the
   region, the bridge's code where a port refused the plan
   (`refused`), ETIMEDOUT past the bound, ESTALE where the map moved since the plan was made (plan again), or the map
   reader's error value (mesh-dataflow.h mesh_nodes_read): each with `executions` not raised, but the ESTALE after the
   raise, with it raised.
   2026-11-04, owner L (design/compilation-coordination.md, "The region"): the plan the context's, where it was the
   region's one plan (hdr `state`, `hops`, `plan`), and the wait for each port's `held` bit or `refused` code, where it
   was for its `prepared` to read hdr `plan` and its phase and code after; `plans` raised and woken, where hdr `plan`
   was woken.  What must stay: the wait bounded by one deadline; the raise only after every named port holds the plan;
   the fence and the serial's read after the raise.
   2026-11-04, owner L2 (the appendix, item 6; design/compilation-coordination.md, "The appendix's revisions", N2, N5):
   the plan the caller's `plan`, where it was the context's `state`, gone.  Why: a program holds any number of
   compiled functions at once, each launched by its own calls (item 6).  What must stay: as above.
   2026-11-04, owner L2, the coordinator's ruling on review W1's finding 4: `plan` tested as above, where it was its
   bounds alone.  Why: a retired handle's pages may be another claim's once the bridge releases them.  What must
   stay: EINVAL before anything is raised */
int mesh_transfers_start(struct mesh_ctx *,uint64_t plan,uint64_t executions);

/* 2026-11-04, sub-assignment 5: mesh_plan_erased is removed, and with it its seq_cst fence pair with the ports' data
   path (design/disjoint-pages-change-request.md, Appendix C, C4): a new plan's state is in new pages, never the
   previous plan's, so nothing erases a plan or waits for one to be erased (C3.4).  What must stay: a port announces the
   generation it loads and reads its claim's head again, which the bridge's release reads (mesh-dataflow.h
   mesh_retired_release; the contract's D10; 2026-11-04, owner L: the head, where it was hdr `hops` and `plan`) */

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
