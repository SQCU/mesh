/* design/page-maps-2026-11-02.md: the page map, a chooser's only operand.

   This header is all of the address model a chooser sees: placement, page effects, the planner, the native bind.
   It holds integers and maps. It declares no pointer to payload, no byte offset of the region, no Metal object. A
   chooser file includes this header and never includes mesh-page-lower.h, so it cannot ask where anything is.

   - A page is an identity: its index in the region's page space. Its size is the space's, never a chooser's figure.
   - A piece is one affine box of a value's layout: logical coordinates in its box map to virtual elements by integer
     strides and an origin. A layout is a list of pieces with disjoint boxes, every closed form of the layout algebra
     read back this way (mesh-tensor-layout.m, 2026-11-02).
   - A page map gives each invocation's virtual pages their physical pages, as explicit runs. The allocator writes it
     from lifetimes; no formula in it is picked by hand. There are no rings, slots or generations.
   - A value is a layout composed with a page map. A tensor, a landing, a SEND source, a progress word and a
     descriptor table are all values.
   - A lifetime is an interval of the function's event order, transport events included.

   Every operation here is pure: it reads its operands and writes only its outputs. The graph and interval algorithms
   it needs live in mesh-graph.h, with these types as their consumers' operands (rubric C9). */
#ifndef MESH_PAGE_MAP_H
#define MESH_PAGE_MAP_H
#include <stddef.h>
#include <stdint.h>

/* the region's pages: their size in bytes (16 KiB on these members) and their count */
struct mesh_page_space { uint64_t page_bytes, pages; };

/* one affine box: logical coordinates c with 0 <= c[d] - first[d] < extent[d] map to the virtual element
   origin + sum(stride[d] * (c[d] - first[d])); rank at most MESH_PIECE_RANK */
#define MESH_PIECE_RANK 8
struct mesh_piece {
  uint32_t rank;
  int64_t first[MESH_PIECE_RANK], extent[MESH_PIECE_RANK], stride[MESH_PIECE_RANK];
  int64_t origin;
};
/* a layout: its pieces, boxes disjoint and covering the value's logical shape, and its element width in bits, which
   enters only at the final lowering */
struct mesh_layout { uint32_t element_bits; size_t count; const struct mesh_piece *pieces; };

/* `count` virtual pages from `first_virtual` on physical pages from `first_physical`, at one invocation */
struct mesh_page_run { uint64_t first_virtual, first_physical, count; };
/* a page map: invocation i's runs are runs[first[i] .. first[i+1]), sorted by first_virtual, disjoint; invocations
   [0, invocations) */
struct mesh_page_map {
  uint64_t invocations;
  const uint64_t *first;
  const struct mesh_page_run *runs;
};

/* a value: its layout's virtual elements on its map's pages */
struct mesh_value { struct mesh_layout layout; struct mesh_page_map map; };

/* an interval of the function's event order [begin, end): command ordinals and transport events (a post, a SEND's
   completion, a landing, a last read, the bridge leaving a table), times invocations */
struct mesh_lifetime { uint64_t begin, end; };

/* the physical page runs and in-page byte runs value v covers at invocation i, in logical order: a physical byte
   b is page * page_bytes + byte within the page, an identity pair, never an address. Returns the count written, or
   the count needed where it exceeds `capacity` */
struct mesh_page_bytes { uint64_t page, first, bytes; };
size_t mesh_page_image(const struct mesh_page_space *space, const struct mesh_value *v, uint64_t invocation,
                       struct mesh_page_bytes *out, size_t capacity);

/* the logical coordinates of value v that lie in physical byte runs `bytes`, as pieces of v's logical box: the
   preimage a consumer tile's threshold and a DMA unit's coverage are read from. Returns as mesh_page_image */
size_t mesh_page_preimage(const struct mesh_page_space *space, const struct mesh_value *v, uint64_t invocation,
                          const struct mesh_page_bytes *bytes, size_t count, struct mesh_piece *out, size_t capacity);

/* whether values a and b share a physical byte at some invocation where both lifetimes hold: aliasing and hazards */
int mesh_page_meet(const struct mesh_page_space *space, const struct mesh_value *a, const struct mesh_lifetime *la,
                   const struct mesh_value *b, const struct mesh_lifetime *lb);

/* a view of v: layout `view` read through v's layout (logical coordinates of the view to v's logical coordinates),
   the result's pieces written to `out`. Returns as mesh_page_image */
size_t mesh_page_compose(const struct mesh_layout *view, const struct mesh_layout *of, struct mesh_piece *out,
                         size_t capacity);

/* Allocation is not declared here: it is mesh-graph.h's mesh_graph_interval_pack over (value, invocation) items, each
   needing its virtual pages over its lifetime, with this header's types as the consumer's operands (rubric C9). Its
   placements become each value's page map, and pages return through the retire ledger (mesh-lease-retire.h) at their
   lifetime's end. */

#endif
