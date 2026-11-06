#include "mesh-call.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* design/algorithm-sources.md#programtensor */
/* 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 6, and "The
   revisions of 4e78f6f4"): row `index` of the section's rows, first + index, where it was copy `index`'s first row,
   first + index x stride (mesh-call.h mesh_section_address's note).  Why: point 6, "Represent a value as its ordered
   page-table rows"; design/plan-to-bridge-change-request.md R2, corrected: "No value is placed as one run", so block k
   of copy c is row c x stride + k, each block placed anywhere, and a copy's bytes are one run only in its value's
   mapping (mesh-collective.h struct mesh_value `view`).  What must stay: an address the row's binding gives, NULL
   where none.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G3; "The
   revisions of 55b9cbad"): struct mesh_value `view` is gone; a copy's bytes are one run only in the mapping the
   program's effector composes over its `run` values' rows (mesh-dataflow.h mesh_view_compose).  What must stay: as
   above */
static uint32_t mesh_section_row(struct mesh_section section,uint32_t index){return section.first+index;}

/* mesh_transfers_start: mesh-dataflow.c, beside the map reader whose copy of the map it reads (its configured latch and
   unbounded spin on `prepared` gave way to a bounded wait for the ports the plan's hops reach) */

/* 2026-11-04, sub-assignment 5: mesh_section_create is removed, sub-assignment 7's 2026-11-03 form, which named a
   section's rows from the program's claim at its call (`row_cursor`), cleaned them, and laid a `wire` 0 section out in
   the claim's `local` run in creation order (`local_cursor`), bound and touched there to fault its pages in.  Why:
   mesh_plan's one allocator places every value and names its rows (design/disjoint-pages-change-request.md, Appendix
   C, C3.2; mesh-collective.h struct mesh_value `section`), nothing is laid out when it is created (C4), and no program
   touches a page to fault it in: the region is wired once, with the region (C3.2).  What must stay: a section is read
   through the rows its plan named */

/* design/algorithm-sources.md#programtensor */
void *mesh_section_address(struct mesh_ctx *context,struct mesh_section section,uint32_t index){
  const uintptr_t address=atomic_load_explicit(&mesh_page(context->M)[mesh_section_row(section,index)].address,memory_order_relaxed);
  return address?(char *)context->M+address:NULL;
}
/* design/algorithm-sources.md#programwrite */
void mesh_section_constant(struct mesh_ctx *context,struct mesh_section section){
  mesh_buffers(context->M)[section.first].constant=1;
}
