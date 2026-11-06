#ifndef MESH_SECTION_H
#define MESH_SECTION_H
#include <stddef.h>
#include <stdint.h>

/* A value's ordered page-table rows (design/plan-to-bridge-change-request.md R2, corrected 2026-11-05: "A value's
   ordered list of pages is its page-table rows (Kilburn et al.; Denning)"; mesh-collective.h struct mesh_value).
   `first` its first row, `pages` P x block (P its blocks, one row a block), `bytes` as the program gave them, `count`
   its copies, `stride` P: copy c's rows first + c x P + k, k < P, each bound to a block placed anywhere (mesh_plan).
   2026-11-05 (design/compilation-coordination.md, "The revisions of 4e78f6f4", the coordinator's finding on "The
   revisions of e4a7398e"): moved here from rdma/mesh-call.h, with stddef.h and stdint.h alone, so rdma/mesh-torch.mm
   includes it where it declared a copy of its own with a static_assert (TP-1: mesh.h's stdatomic.h and
   infiniband/verbs.h do not compile as C++).  What must stay: one declaration of the struct, every field in its place */
struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride; };
#ifdef __cplusplus
static_assert(sizeof(mesh_section)==24 && offsetof(mesh_section,bytes)==8 && offsetof(mesh_section,stride)==20,
              "mesh_section");
#else
_Static_assert(sizeof(struct mesh_section)==24 && offsetof(struct mesh_section,bytes)==8 &&
               offsetof(struct mesh_section,stride)==20,"mesh_section");
#endif
#endif
