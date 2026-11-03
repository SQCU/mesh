/* design/page-maps-2026-11-02.md: the page map, a chooser's only operand.

   This header is all of the address model a chooser sees: placement, page effects, the planner, the native bind.
   It holds integers and maps. It declares no pointer to payload, no byte offset of the region, no Metal object. A
   chooser file includes this header and never includes mesh-page-lower.h, so it cannot ask where anything is.

   - A page is an identity: its index in the region's page space. Its size is the space's, never a chooser's figure.
   - A piece is one affine box of a value's layout: logical coordinates in its box map to virtual elements by integer
     strides and an origin. A layout is a list of pieces with disjoint boxes, every closed form of the layout algebra
     read back this way (mesh-tensor-layout.m, 2026-11-02).
   - A page map gives each invocation's virtual pages their physical pages, as explicit runs. The run's solve
     (mesh_run_solve) writes it from lifetimes; no formula in it is picked by hand. There are no rings, slots or
     generations.
   - A value is a layout composed with a page map. A tensor, a landing, a SEND source, a progress word and a
     descriptor table are all values.
   - A lifetime is an interval of the function's event order, transport events included.

   Every operation here is pure: it reads its operands and writes only its outputs. The graph and interval algorithms
   it needs live in mesh-graph.h, with these types as their consumers' operands (rubric C9). */
#ifndef MESH_PAGE_MAP_H
#define MESH_PAGE_MAP_H
#include <stddef.h>
#include <stdint.h>
#include "mesh-graph.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Types, informally (each function below states its own in these terms; the LSTM's dependency chain run end to end on
   every member is what holds them to account, not a separate test):
   - Page       a page index in [0, space.pages).
   - Box(r)     r integer intervals [first, first + extent), extents >= 0.
   - Piece(r)   Box(r) -> Elem, affine: e(c) = origin + sum stride[d] (c[d] - first[d]).
   - Layout(r)  pieces of rank r whose boxes are pairwise disjoint; Elem = the virtual elements they reach.
   - Map        Inv -> (VPage -> Page) as runs, each invocation's runs disjoint in VPage, sorted, and every Page of
                one invocation's runs distinct.
   - Value(r)   Layout(r) x Map, every virtual page the layout reaches mapped at every invocation of the map.
   - View(r,R)  Box(r) -> coordinates of an R-axis value, axis-wise affine (axis[d], step[d]) from base. It has no
                division: a merged axis is written as mixed-radix axes.
   - Life       [begin, end) in the function's event order.
   - Bytes      (Page, first, bytes) with first + bytes <= space.page_bytes: physical bytes as identities. */

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
struct mesh_map_run { uint64_t first_virtual, first_physical, count; };
/* a page map: invocation i's runs are runs[first[i] .. first[i+1]), sorted by first_virtual, disjoint; invocations
   [0, invocations) */
struct mesh_page_map {
  uint64_t invocations;
  const uint64_t *first;
  const struct mesh_map_run *runs;
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
/* type: (Space, Value(r), Inv) -> [Bytes] in logical order, joined within a page; SIZE_MAX iff the map leaves a
   reached virtual page unmapped (then not a Value) */
size_t mesh_page_image(const struct mesh_page_space *space, const struct mesh_value *v, uint64_t invocation,
                       struct mesh_page_bytes *out, size_t capacity);

/* the logical coordinates of value v that lie in physical byte runs `bytes`, as pieces of v's logical box: the
   preimage a consumer tile's threshold and a DMA unit's coverage are read from. Returns as mesh_page_image */
/* type: (Space, Value(r), Inv, [Bytes]) -> Layout(r), a sub-layout of the value's: exactly the coordinates whose
   element bytes lie wholly inside the given Bytes */
size_t mesh_page_preimage(const struct mesh_page_space *space, const struct mesh_value *v, uint64_t invocation,
                          const struct mesh_page_bytes *bytes, size_t count, struct mesh_piece *out, size_t capacity);

/* whether value a at invocation ia and value b at invocation ib share a physical byte while both lifetimes hold:
   aliasing and hazards. 1, 0, or -1 where a value's map leaves one of its pages unmapped */
/* type: (Space, Value, Inv, Life, Value, Inv, Life) -> {0, 1, -1}: 1 iff both lives overlap and the two images share
   a byte; -1 iff an operand is not a Value */
int mesh_page_meet(const struct mesh_page_space *space, const struct mesh_value *a, uint64_t ia,
                   const struct mesh_lifetime *la, const struct mesh_value *b, uint64_t ib, const struct mesh_lifetime *lb);

/* a view: its coordinates c, in the box first/extent of rank `rank`, walk the coordinates of the value it views. View
   axis d moves that value's axis axis[d] by step[d] a unit, from base[]. Several view axes may walk one axis (a
   split), a step may be 0 (a broadcast) or negative (a reversal). A reshape that merges axes is written as more view
   axes, in mixed radix, so composition needs no division. */
struct mesh_view {
  uint32_t rank, of_rank;
  int64_t first[MESH_PIECE_RANK], extent[MESH_PIECE_RANK], step[MESH_PIECE_RANK];
  uint32_t axis[MESH_PIECE_RANK];
  int64_t base[MESH_PIECE_RANK];
};
/* the view's layout: `view` read through `of`'s pieces, each of their boxes over `of`'s logical coordinates, cut where
   the view's image crosses a piece's box. Returns the count written, the count needed where it exceeds `capacity`, or
   SIZE_MAX where two view axes walk one axis across a box edge that their mixed radix does not align with */
/* type: (View(r,R), Layout(R)) -> Layout(r), with e_result(c) = e_of(view(c)) wherever view(c) lies in a box of `of`,
   and boxes cut exactly at `of`'s box edges; SIZE_MAX where the cut would fall inside one mixed-radix step */
size_t mesh_page_compose(const struct mesh_view *view, const struct mesh_layout *of, struct mesh_piece *out,
                         size_t capacity);

/* The run's solve: the library's one allocator and scheduler (design/page-maps-2026-11-02.md, "The run's solve, each
   choice and its ground"). It takes a run's events (numbered in the declared order: commands, receive posts, SENDs,
   their completions, landings, the bridge leaving a table), its accesses to items, the transport's own edges and the
   bins of positions the process owns (one a lease the run touches, the arena last), and decides each item's place and
   the run's order. It knows no address; the lowering reindexes a place through its bin's lease. */
/* an item live from before the run starts, or past its end; MESH_RUN_COHERENT an item every access of which is
   system-coherent or the NIC's (a SEND's source, a landing): it shares no position with an item accessed ordinarily,
   since an M4 Pro's GPU keeps an ordinary store's line dirty across a running buffer's barriers and writes it back at the
   buffer's end, over a landing into the same bytes (metal-microbench docs/facts.md, visible.m "writeback": 2048 of 2048
   words), however the edges order the two */
enum { MESH_RUN_BEFORE = 1, MESH_RUN_AFTER = 2, MESH_RUN_COHERENT = 4 };
#define MESH_RUN_FREE UINT64_MAX
/* an item: a value at an invocation, a state's version, a collective's own value, or a coexisting buffer (pinned, with
   `lives`): its pages; its first position where it is pinned (else MESH_RUN_FREE); `contiguous` where a function typed
   contiguous_input or contiguous_output reads or writes it, so it gets one range in one bin. Every other item's pages
   are any positions in any bins: every other function takes arbitrary views of arbitrarily placed pages. */
struct mesh_run_item { uint64_t pages, pin; uint32_t lives, contiguous; };
/* positions [first, first + pages) of one lease; the last bin is the arena, `pages` its most */
struct mesh_run_bin { uint64_t first, pages; };
/* the solve's result, owned by the caller after mesh_run_solve until mesh_run_plan_free */
struct mesh_run_plan {
  size_t edges;
  struct mesh_graph_dependence *edge; /* every ordering the effectors realize: the dependences and LINK edges */
  uint32_t *order;                    /* the events in issue order, each agent's order its subsequence */
  size_t *run_first;                  /* item k's runs are runs[run_first[k] .. run_first[k + 1]), none where it holds no page */
  uint64_t (*runs)[2];                /* (first position, pages), ascending within an item, each inside one bin */
  uint64_t arena;                     /* positions of the last bin in use: the arena's pages, held from load */
};
/* The last k solutions and their measured per-case rates, for the later passes (design/page-maps-2026-11-02.md, "The
   run's solve", 11). Stubbed: its type is not written yet, and the solve reads none; pass NULL. */
struct mesh_run_history;
/* type: (Space, events, [Access] in the declared order, [(from, to)] links, [Item], [Bin]) -> Plan with:
   - edges: mesh_graph_dependences over the accesses, each access's begin -> end where they differ, and the links;
   - order: the least-numbered ready event first (mesh_graph_order), so the declared order where the edges leave it;
   - runs: two items share a position only where every access of one happens before every access of the other over
     the edges (an item live before the run has nothing before it; one live past it has nothing after) and both or
     neither are MESH_RUN_COHERENT; pinned items
     at their pins, contiguous items one range each, the others page by page (mesh_graph_place).
   No edge is added for reuse. 0; EINVAL for an access outside its item or its events, unordered bins, a pin in no bin,
   or a cycle in the edges; EBUSY where two pins meet; ENOMEM where the run's live set exceeds its bins. */
int mesh_run_solve(const struct mesh_page_space *space, uint32_t events, size_t accesses,
                   const struct mesh_graph_access *access, size_t links, const uint32_t (*link)[2], size_t items,
                   const struct mesh_run_item *item, size_t bins, const struct mesh_run_bin *bin,
                   const struct mesh_run_history *history, struct mesh_run_plan *plan);
void mesh_run_plan_free(struct mesh_run_plan *plan);

#ifdef __cplusplus
}
#endif
#endif
