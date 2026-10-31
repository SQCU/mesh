/* design/rubrics.md:422 (C9): the graph algorithms, one copy each, for every consumer of weights or samples on a graph.
   A function here knows vertices, edges and the operands its consumer passes (weights, visits, a next hop); it knows
   no cable, collective or tensor.  Python's are rdma/mesh_graph.py's. */
#ifndef MESH_GRAPH_H
#define MESH_GRAPH_H
#include <stddef.h>
#include <stdint.h>

/* A path's share of a tree's flow: the path from `from` of `hops` hops, hop k leading from `at` to
   `step(context, k, at)` and carrying `split(context, k, at)` of what reaches it, followed up the tree (parent[v]: v's
   next vertex toward the root).  The product of its hops' splits where every hop follows the tree, else 0; *end the
   vertex it reaches (the one it leaves the tree at, where it does). */
double mesh_graph_tree_path(const uint32_t *parent, uint32_t from, uint32_t hops,
                            uint32_t (*step)(const void *context, uint32_t k, uint32_t at),
                            double (*split)(const void *context, uint32_t k, uint32_t at), const void *context,
                            uint32_t *end);
/* The vertices `from` reaches over `edges` undirected edges, edge e joining `end(context, e, 0)` and
   `end(context, e, 1)` (each below `vertices`), breadth first: their count; `order` (room for `vertices`) the vertices
   in the order reached, `reached` (room for `vertices`) 1 for each and 0 for every other. */
uint32_t mesh_graph_reach(uint32_t vertices, uint32_t edges, uint32_t (*end)(const void *context, uint32_t e, uint32_t side),
                          const void *context, uint32_t from, uint32_t *order, unsigned char *reached);
/* Every simple path from `source` to `destination` over the same edges, depth first in edge order and off the call
   stack: `visit(consumer, path, length)` with each one's edges in order, a path ending where it first reaches
   `destination`.  The first nonzero `visit` returns ends the walk and is returned; else 0, or ENOLINK where no path
   joins them.  `work` holds 3 * vertices words, `seen` vertices bytes. */
int mesh_graph_paths(uint32_t vertices, uint32_t edges, uint32_t (*end)(const void *context, uint32_t e, uint32_t side),
                     const void *context, uint32_t source, uint32_t destination, uint32_t *work, unsigned char *seen,
                     int (*visit)(void *consumer, const uint32_t *path, uint32_t length), void *consumer);

/* Directed edges in compressed rows: vertex v's leave as edges [first[v], first[v + 1]), edge k entering
   `head(context, k)`. */
/* Every vertex reachable over such edges from the `pending` vertices on `stack` (room for each vertex once more),
   marked `seen[v] = stamp` as an edge reaches it (the seeds only where one does): a stamp a query, so `seen` needs no
   clearing between queries. */
void mesh_graph_mark(const size_t *first, uint32_t (*head)(const void *context, size_t k), const void *context,
                     uint32_t *seen, uint32_t stamp, uint32_t *stack, size_t pending);
/* A topological order of such edges' `vertices` vertices: ordinal[v] v's place, each place taken by the
   lowest-numbered vertex whose every edge in leaves a placed one.  `indegree` holds `vertices` words.  0, or EDEADLK
   where a cycle leaves vertices unplaced (their ordinal UINT32_MAX). */
int mesh_graph_order(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                     const void *context, uint32_t *indegree, uint32_t *ordinal);
/* Chu-Liu/Edmonds: the minimum spanning arborescence of nodes [0, N) rooted at `root` in which each other node v
   takes one edge e with to[e] == v (its parent from[e]); chosen[v] that edge, the root's -1; 0, or -1 where a node has
   none.  Each node's lightest edge; a cycle among them contracted, the edges into it weighing their excess over the
   edge they would replace, the contracted graph's arborescence expanded (the edge into a cycle replaces the cycle's
   edge into the same node). */
int mesh_graph_arborescence(uint32_t N, uint32_t root, uint32_t E, const uint32_t *from, const uint32_t *to,
                            const double *weight, int32_t *chosen);
/* The walk up a parent table from `from` (parent[v]: v's next vertex up): `visit(consumer, at, step)` for each vertex
   it meets, `from` at step 0, until one that is `root` (UINT32_MAX: none) or whose parent names none
   (parent[at] >= vertices), visited last; a vertex its own parent (outside a tree) climbs in place.  The steps to that vertex; UINT32_MAX where `vertices` steps meet none (a
   cycle), steps 0 to `vertices` visited.  `visit` NULL: none. */
uint32_t mesh_graph_climb(const uint32_t *parent, uint32_t vertices, uint32_t root, uint32_t from,
                          void (*visit)(void *consumer, uint32_t at, uint32_t step), void *consumer);
/* The length of the cycle a climb from `from` returns to `from` on (the first step past 0 it meets `from` again);
   0 where it does not within `vertices` steps. */
uint32_t mesh_graph_cycle_length(const uint32_t *parent, uint32_t vertices, uint32_t from);
/* A tree's levels over its parent table: depth[v] the steps from v up to `root` (`vertices` where its climb meets no
   end), height[v] the longest climb up to v from a vertex below it (0 for a leaf). */
void mesh_graph_tree_levels(const uint32_t *parent, uint32_t vertices, uint32_t root, uint32_t *depth, uint32_t *height);
/* `from` and every vertex its climb meets up to `root`, marked[v] = 1 (the rest of `marked` untouched). */
void mesh_graph_ancestors(const uint32_t *parent, uint32_t vertices, uint32_t root, uint32_t from, unsigned char *marked);
/* A page schedule: file transfers scheduled as an edge colouring with precedence (Coffman, Garey, Johnson and
   LaPaugh 1985), by highest-level-first list scheduling.  Move k is units[k] units on arc[k], each unit_seconds long
   and its last last_seconds, or, arc[k] == arcs, an event that takes no arc and no time; none of its units is ready
   before release.  Follow e: move after's unit u waits for move before's unit offset + u (MESH_GRAPH_STREAMED), its
   last unit (MESH_GRAPH_COMPLETE) or its first unit's start (MESH_GRAPH_STARTED); nothing else waits.  An arc carries
   one unit at a time and a move's units go in order; each time an arc frees, of the units ready on it the one whose
   move has the most arcs below it goes (ties: the move the arc carried last, then the lower `production`, then the
   lower k), else the first to be ready.  order[k]: k's place by its first unit's time, ties by depth then k (a
   topological order).  *finish (NULL: not wanted): the last unit's end.  0, EINVAL, ENOMEM, or EDEADLK (a cycle). */
enum mesh_graph_wait { MESH_GRAPH_STARTED, MESH_GRAPH_STREAMED, MESH_GRAPH_COMPLETE };
struct mesh_graph_follow { uint32_t before, after, offset; enum mesh_graph_wait wait; };
struct mesh_graph_move { uint32_t arc, units; uint64_t production; double unit_seconds, last_seconds, release; };
int mesh_graph_schedule(const struct mesh_graph_move *move, uint32_t moves, uint32_t arcs,
                        const struct mesh_graph_follow *follow, size_t follows, uint32_t *order, double *finish);
/* A walk along a chain whose links are stored as it grows (a node's successor `follow(context, node)`, 0 until stored;
   node 0 the chain's head) to the item holding position q, each node's `items(context, node)` items in order and item
   i `weight(context, node, i)` positions long: the walk's place (`cursor`: its node, its item there, `before` the
   positions of the items passed, `left` the nodes left) moves past every item whose positions all lie before q and
   into each node linked after its own, `leave(consumer, node, left)` told each node it leaves (NULL: none).  1 where
   it reaches q's item (cursor->item of cursor->node, its first position cursor->before), 0 where the chain ends
   before it.  A place only moves forward: q at or past the last q asked. */
struct mesh_graph_chain { uint64_t node, item, before, left; };
/* Vertex v's edges in compressed rows of 32-bit words (v's edges [first[v], first[v + 1]), edge k reaching head[k]),
   in row order: `visit(consumer, head[k])` for each until one returns nonzero, which is returned; else 0 (`visit`
   NULL: none).  *count (NULL: not wanted) the row's edges, and *last, where it has any, its last edge's head. */
int mesh_graph_row(const uint32_t *first, const uint32_t *head, uint32_t v, int (*visit)(void *consumer, uint32_t head),
                   void *consumer, uint32_t *count, uint32_t *last);
/* The edge v's work goes on along (a SEND, the receive whose bytes it forwards: nccl-mesh.c send_dependencies), in the
   same rows: the head of v's row's first edge (`first_edge`) or its last, where `holds(consumer, head)` (NULL: any);
   -1 where v's row is empty or that head does not hold it */
int64_t mesh_graph_forward(const uint32_t *first, const uint32_t *head, uint32_t v, int first_edge,
                           int (*holds)(const void *consumer, uint32_t head), const void *consumer);
/* Two cuts of one byte range into units, `a` units and, from byte `delta` of it, `b` (a landing's and a SEND that
   forwards part of it), unit k of cut `side` (0: a's) weight(context, side, k) bytes: *at the units of a wholly before
   or across byte delta (a's units added while their bytes stay below delta); 1 where b's units are a's from there,
   unit for unit (delta on a unit boundary of a, each of b's units the bytes of a's at its place), else 0 */
int mesh_graph_cut_at(uint64_t delta, uint64_t a, uint64_t b, uint64_t (*weight)(const void *context, int side, uint64_t k),
                      const void *context, uint64_t *at);
/* A worklist's cells in their global order (design/zorvek.md zelloth: a group's sums' blocks): n segments of length[i]
   units, each cut in order into cells of `cell` units (its last one short); cell q's segment at segment[q] and its
   place in that segment, in cells, at place[q] (NULL, both: counted only).  The cells' count */
uint64_t mesh_graph_cells(const uint64_t *length, size_t n, uint64_t cell, uint64_t *segment, uint64_t *place);
int mesh_graph_chain(struct mesh_graph_chain *cursor, uint64_t q, uint64_t (*follow)(const void *context, uint64_t node),
                     uint64_t (*items)(const void *context, uint64_t node),
                     uint64_t (*weight)(const void *context, uint64_t node, uint64_t item), const void *context,
                     void (*leave)(void *consumer, uint64_t node, uint64_t left), void *consumer);
/* Interval meets (a dependency's edges between the spans a set of writers and a set of readers touch): `a` n spans
   [begin, end) sorted on begin, reach[i] the greatest end among a[0..i] (mesh_graph_span_reach); each a[i] span b
   meets (a[i].begin < b.end and b.begin < a[i].end) told to meet(consumer, i), in descending i.  O(log n plus the
   spans scanned back past b's begin, those it meets where a's spans are disjoint) */
struct mesh_graph_span { uint64_t begin, end; };
void mesh_graph_span_reach(const struct mesh_graph_span *a, size_t n, uint64_t *reach);
/* whether n spans sorted on begin are pairwise disjoint: each begins at or past the end of every one before it */
int mesh_graph_disjoint(const struct mesh_graph_span *a, size_t n);
void mesh_graph_meets(const struct mesh_graph_span *a, const uint64_t *reach, size_t n, struct mesh_graph_span b,
                      void (*meet)(void *consumer, size_t i), void *consumer);
#endif
