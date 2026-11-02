/* design/page-maps-2026-11-02.md: the final lowering's one pointer-producing step. Only effector code includes this:
   the recorder writing a kernel's arguments, the bridge-table writer, a CPU store. No chooser file includes it.

   Each realized invocation's schedule exists before it runs, so specialization happens once per invocation. It writes
   pointer literals, or a base and an affine offset, into the recorded commands. Run time reads constants. No kernel
   evaluates a map, and no effector decides anything: it evaluates a map someone else chose. */
#ifndef MESH_PAGE_LOWER_H
#define MESH_PAGE_LOWER_H
#include "mesh-page-map.h"

/* the region as this process maps it: its base address and its page space. The one place an address of the region
   enters the program */
struct mesh_page_region { unsigned char *base; struct mesh_page_space space; };

/* one specialized piece of a value at an invocation: the address of its first element, and its byte strides. Where
   the piece's elements cross no page-run boundary the address is a base plus an affine offset, as hand-written code
   would hold it */
struct mesh_page_address { unsigned char *first; int64_t stride_bytes[MESH_PIECE_RANK]; uint32_t rank; };

/* value v's pieces at invocation i, cut at the page-run boundaries of v's map, as addresses. Returns the count
   written, or the count needed where it exceeds `capacity` */
size_t mesh_page_specialize(const struct mesh_page_region *region, const struct mesh_value *v, uint64_t invocation,
                            struct mesh_page_address *out, size_t capacity);

/* one SGE of a DMA unit: its region byte offset and its bytes, as the bridge's descriptor tables hold them */
struct mesh_page_sge { uint64_t offset, bytes; };
/* value v's bytes at invocation i as SGEs, cut at page-run boundaries and at the declared unit cuts `cuts` (logical
   byte positions, ascending). Returns as mesh_page_specialize */
size_t mesh_page_sges(const struct mesh_page_region *region, const struct mesh_value *v, uint64_t invocation,
                      const uint64_t *cuts, size_t cut_count, struct mesh_page_sge *out, size_t capacity);

#endif
