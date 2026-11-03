/* design/rubrics.md:422 (C9): the graph algorithms, one copy each, for every consumer of weights or samples on a graph.
   A function here knows vertices, edges and the operands its consumer passes (weights, visits, a next hop); it knows
   no cable, collective or tensor.  Python's are rdma/mesh_graph.py's. */
#ifndef MESH_GRAPH_H
#define MESH_GRAPH_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

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
/* Each marked vertex's depth over the edges into it (vertex v's row [first[v], first[v + 1]) of head[]: the vertices
   whose work it follows): 0 where its row reaches no marked vertex, else one past the greatest depth among the marked
   ones it reaches, taken over a topological order (mesh_graph_order).  0; EDEADLK where the rows hold a cycle; ENOMEM */
int mesh_graph_depth(uint32_t vertices, const size_t *first, const uint32_t *head, const unsigned char *marked, uint32_t *depth);
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
/* A vertex's row reduced to its frontier (a SEND's receives, one for each word their consumptions count: nccl-mesh.c
   send_requirements): the heads of v's edges grouped by key(consumer, head), of each group the head of greatest
   rank(consumer, head), at most `most` groups, their heads at kept[] and keys at keys[] in first-seen order.  The
   groups' count; -1 where a key is 0 (none known) or the groups pass `most` */
int64_t mesh_graph_frontier(const uint32_t *first, const uint32_t *head, uint32_t v,
                            uint64_t (*key)(const void *consumer, uint32_t head),
                            uint64_t (*rank)(const void *consumer, uint32_t head), const void *consumer, uint32_t most,
                            uint32_t *kept, uint64_t *keys);
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
/* For each b[j], the count of a's n values at or below it (through[j]), a sorted ascending: a rank query */
void mesh_graph_through(const uint64_t *a, size_t na, const uint64_t *b, size_t nb, uint64_t *through);
/* Two worklists' cells over one line (mesh_graph_cells: segments [first, first + length) cut into cells of `cell`),
   the second's units `delta` past the first's: for each of the second's cells in order, the count of the first's
   cells starting at or before its last unit (through[]: the first's done prefix it needs, zelloth's later sum over an
   earlier one's result, nccl-mesh-metal.m metal_zelloth).  0; -1 where the first's cells do not start in order, every
   through[] then the first's count */
int mesh_graph_cells_through(const uint64_t *a_first, const uint64_t *a_length, size_t na, const uint64_t *b_first,
                             const uint64_t *b_length, size_t nb, int64_t delta, uint64_t cell, uint64_t *through);
/* A prefix scan over completed output cells. This publishes data availability;
   it neither chooses work nor reconstructs dependencies. */
#define MESH_GRAPH_PREFIX_METAL \
  "static inline uint mesh_graph_prefix(device atomic_uint *state, uint head, uint n, device atomic_uint *at) { " \
  "uint p = mesh_graph_count_load(at); " \
  "while (p < n && (mesh_graph_count_load(state + head + p) & 2u)) " \
  "if (mesh_graph_count_exchange(at, &p, p + 1u)) p++; " \
  "return p; }\n" \
  "template <typename Publish> static inline void mesh_graph_settle(device atomic_uint *state, uint head, uint n, " \
  "device atomic_uint *at, thread Publish &publish) { " \
  "const uint p = mesh_graph_prefix(state, head, n, at); if (p) publish.at(p); }\n"
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
void mesh_graph_meets(const struct mesh_graph_span *a, const uint64_t *reach, size_t n, struct mesh_graph_span b,
                      void (*meet)(void *consumer, size_t i), void *consumer);
/* Several partitions' graphs joined into one (a group's calls, each its steps' prior rows: nccl-mesh.c zabbadoo_plan):
   partition p's vertices the joined [base[p], base[p + 1]), its vertex j's row [first[p][j], first[p][j + 1]) of
   head[p][] (first[p] NULL: a partition with no rows); each marked vertex's row kept, mapped to the joined vertices,
   its edges to unmarked vertices counted in *outside and dropped; *rows (base[parts] + 1 of them) and *heads
   malloc'd, the caller's to free.  0; ENOMEM */
int mesh_graph_join(uint32_t parts, const size_t *base, const uint32_t *const *first, const uint32_t *const *head,
                    const unsigned char *marked, size_t **rows, uint32_t **heads, size_t *outside);
/* Classes of vertices ordered by the edges between them (a group's sums, each a class of its steps: nccl-mesh.c
   zabbadoo_plan): vertex v's row [first[v], first[v + 1]) of head[], class_of[v] its class (UINT32_MAX: none); an edge
   from class a to class b != a wherever a vertex of b's row reaches one of a.  span[k] class k's range and part[k] its
   partition: two classes of different partitions whose spans meet, which no edge orders, are refused.  place[k] k's
   place in a topological order of the edges (mesh_graph_order); class k's in-edges, each class once, its row
   [(*in_first)[k], (*in_first)[k + 1]) of *in_from (both malloc'd, the caller's to free).  0; 1 where spans of two
   partitions meet or the edges hold a cycle; ENOMEM */
int mesh_graph_classes(uint32_t vertices, const size_t *first, const uint32_t *head, const uint32_t *class_of,
                       uint32_t classes, const struct mesh_graph_span *span, const uint32_t *part, uint32_t *place,
                       size_t **in_first, uint32_t **in_from);
/* Records grouped by a key into compressed rows (a routing's trees by their demand, its paths by their scope): record
   k's key is key(context, k) < keys; key v's records are order[first[v]] to order[first[v + 1] - 1], in record order.
   first holds keys + 1 entries and order items: one count, one sum and one scatter. */
void mesh_graph_rows(uint32_t keys, size_t items, uint32_t (*key)(const void *context, size_t k), const void *context,
                     size_t *first, size_t *order);
/* Items chained by a key, latest first, as a solver finds them (its distinct trees of one demand or of one root):
   head[key] that key's latest item (UINT32_MAX: none) and next[k] the one before item k. mesh_graph_keyed_find gives
   the key's item for which equal(context, k) holds, or UINT32_MAX, and reads that key's items alone;
   mesh_graph_keyed_add chains item k to its key. */
uint32_t mesh_graph_keyed_find(const uint32_t *head, const uint32_t *next, uint32_t key,
                               int (*equal)(const void *context, uint32_t k), const void *context);
void mesh_graph_keyed_add(uint32_t *head, uint32_t *next, uint32_t key, uint32_t k);
/* exp and log of IEEE-exact operations alone (+, -, x, / correctly rounded; frexp, ldexp, floor exact), so that every
   node computes the same trees and the same shares from them, whatever its libm */
double mesh_graph_exp(double x);
double mesh_graph_log(double x);
/* A concurrent flow of demands over an undirected graph by multiplicative weights (a planner's routing solve:
   mesh-plan.h mesh_solve). Edge e joins from[e] and to[e], capacity[e] each way. Demand d's tree is rooted at root[d]
   and reaches each vertex v for which leaf(context, d, v) holds; up[d] bytes cross it toward the root and down[d] from
   it. Each of `rounds` rounds routes every demand along its least-length tree, a direction's length
   exp(eps (load / capacity - most) / width) / capacity weighed by the bytes that cross it, and adds a round's share of
   those bytes to the loads. Each demand's distinct trees are kept, in the order found: tree k's demand demand[k], its
   parents parent[k * vertices ..] (the root its own, a vertex off the tree UINT32_MAX), its share share[k] (the
   rounds that chose it over all rounds). Their count; UINT32_MAX where eps is not positive, a root is no vertex, a leaf
   has no route, or more than room trees are found. */
uint32_t mesh_graph_flow(uint32_t vertices, uint32_t edges, const uint32_t *from, const uint32_t *to, const double *capacity,
                         uint32_t count, const uint32_t *root, const double *up, const double *down,
                         int (*leaf)(const void *context, uint32_t d, uint32_t v), const void *context, double eps,
                         uint32_t rounds, uint32_t room, uint32_t *demand, uint32_t *parent, double *share);
/* Spanning trees packed by Garg and Koenemann's rounds (a communicator's own forms: mesh-plan.h mesh_trees_pack). Edge e
   makes vertex to[e] a child of from[e], its route crossing directed hops crossing[through[e]] to
   crossing[through[e + 1] - 1], hop k of capacity[k]. Every root r for which rooted(context, r) holds is given a demand
   of one, packed along least-weight arborescences (mesh_graph_arborescence) until the hops' lengths reach one. Each
   root's `most` heaviest distinct trees are kept, root by root: root[k], parents parent[k * vertices ..] and log
   weight log_weight[k], the log of its share of its root's kept flow. Their count; 0 where no arborescence spans. */
uint32_t mesh_graph_pack(uint32_t vertices, uint32_t edges, const uint32_t *from, const uint32_t *to, const uint32_t *through,
                         const uint32_t *crossing, uint32_t hops, const double *capacity,
                         int (*rooted)(const void *context, uint32_t r), const void *context, double eps, uint32_t most,
                         uint32_t *root, uint32_t *parent, double *log_weight);
/* The items left once every one take(consumer, item) accepts is taken, the passes repeating while one is taken (taking
   one can free another: a composed page view's return unpins its owner leases, nccl-mesh.c leases_reap); a taken item
   leaves the array by swap-removal. The count left. */
size_t mesh_graph_settle(void **items, size_t count, int (*take)(void *consumer, void *item), void *consumer);
/* The classes of ordinals under a pairing (placement's view equalities, mesh-tensor-layout.m MetalTensorPairQuotient):
   each position of a removed lattice joined with the kept position of the same flat index, every kept value below every
   removed position. The classes in order of their least members number an ordinal p as p - C(p), C(p) the lattice's
   positions below p, and a removed position as its kept partner. The lattice is origin + sum_k digit_k stride[k], digit_k
   in [0, extent[k]), axes ascending by stride, each stride at least the span of the axes below (stride[k + 1] >=
   stride[k] extent[k]). Written as integer terms over the ordinal p in [first, end) through the consumer's constructors
   (`kept` the kept side's position at the digits of a removed one), so the only divisions are by the lattice's strides:
   p is in the lattice where each digit of y = p - origin at its stride is within its extent, and C(y) counts axis by
   axis from the top, min(q, extent) blocks of the axes below and those axes' count only while q is within its axis. */
enum mesh_graph_relation { MESH_GRAPH_BELOW, MESH_GRAPH_AT_MOST, MESH_GRAPH_AT_LEAST, MESH_GRAPH_EQUAL };
struct mesh_graph_lattice { int64_t origin; size_t axes; const int64_t *stride, *extent; };
struct mesh_graph_terms {
  void *(*constant)(void *consumer, int64_t value);
  void *(*position)(void *consumer);
  void *(*sum)(void *consumer, void *a, void *b);
  void *(*scale)(void *consumer, void *a, int64_t factor);
  void *(*quotient)(void *consumer, void *a, int64_t divisor, int remainder); /* floor(a / divisor), or a mod divisor */
  void *(*minimum)(void *consumer, void *a, int64_t bound);
  void *(*compare)(void *consumer, void *a, enum mesh_graph_relation relation, int64_t bound);
  void *(*all)(void *consumer, void *const *conditions, size_t count);
  void *(*select)(void *consumer, void *condition, void *yes, void *no);
  void *(*kept)(void *consumer, void *const *digits, size_t axes);
};
void *mesh_graph_pair_classes(const struct mesh_graph_lattice *removed, int64_t first, int64_t end,
                              const struct mesh_graph_terms *terms, void *consumer);
/* Tiles placed in order of their least members, each at the next slot of `width` (placement's packing of a field whose
   elements tile the ordinals, mesh-tensor-layout.m TensorPlacedOnTiles): at[i] the ordinal of tile i / width's
   member i % width, a bijection of [0, count); placed[p] the slot position of ordinal p, its tile's rank times width
   plus its place in the tile. -1 where `at` is no such bijection (placed is then unspecified), 0 otherwise. */
int mesh_graph_tile_slots(size_t count, size_t width, const int64_t *at, int64_t *placed);
/* The run's solve's graph algorithms (design/page-maps-2026-11-02.md, "The run's solve, each choice and its ground"),
   with the solve's records as operands. Events are numbered in the declared order; an item is a value at an
   invocation, or a state's version. */
enum { MESH_GRAPH_READ = 1, MESH_GRAPH_WRITE = 2 };
/* item's bytes [first, first + bytes), read and/or written (mode) over events [begin, end] */
struct mesh_graph_access { uint32_t begin, end, mode; uint64_t item, first, bytes; };
enum { MESH_GRAPH_RAW = 1, MESH_GRAPH_WAW = 2, MESH_GRAPH_WAR = 3, MESH_GRAPH_LINK = 4 };
/* event `from` before event `to`, because of item's bytes [first, first + bytes) (kind RAW, WAW, WAR), or because the
   transport orders them (LINK: item and bytes 0) */
struct mesh_graph_dependence { uint32_t from, to, kind; uint64_t item, first, bytes; };
/* type: [Access] in the declared order -> [Dependence]: for each access j and each byte of it, the edge from the end of
   the latest earlier access of that byte that writes it (RAW where j reads, WAW where j writes), and where j writes,
   one from the end of each access that read it since; none from an event to itself. Joined where they differ only in
   adjacent bytes. Returns the count, written up to `capacity`, or SIZE_MAX where memory ran out. */
size_t mesh_graph_dependences(size_t n, const struct mesh_graph_access *access, struct mesh_graph_dependence *out,
                              size_t capacity);
/* Sets of vertices over directed rows (first, head; mesh_graph_mark): set g's sources source[source_first[g] ..
   source_first[g + 1]) and targets target[target_first[g] .. target_first[g + 1]).
   type: -> precedes, a sets x sets bit matrix (bit g * sets + h of byte (g * sets + h) / 8, the caller's, cleared):
   for g != h, set when every target of h is reached from every source of g by one edge or more (vacuously where
   either list is empty). The run's solve reads it as "every access of item g happens before every access of item h"
   (sources the accesses' ends, targets their begins). One mark from each source; 0 or ENOMEM. */
int mesh_graph_precedence(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                          const void *context, size_t sets, const size_t *source_first, const uint32_t *source,
                          const size_t *target_first, const uint32_t *target, unsigned char *precedes);
/* The realization of a run's order on its agents (design/page-maps-2026-11-02.md, integration step 5): the decisions
   an effector encodes, made here and nowhere in it.
   type: n nodes in issue order (a linear extension of the edges), host[v] whether node v runs on the host (else on the
   GPU queue), m edges (from, to) with from before to ->
   - part[v] for each GPU node: its command buffer, consecutive in the order; a new part starts only at a GPU node
     with a host node among its predecessors not already waited on by the part (a queue waits on an event only between
     command buffers); UINT32_MAX for a host node. *parts their count;
   - barrier[v]: a barrier before GPU node v within its part; the fewest such that every edge between two nodes of one
     part crosses one: interval stabbing, greedy by right endpoint (optimal; A. Darte and R. Schreiber, "A linear-time
     algorithm for optimal barrier placement", PPoPP 2005);
   - waits: each part's or host node's predecessors on other parts or other host nodes, as (waiter, kind, producer)
     rows, waiter a part (kind 0) or a host node (kind 1), producer a part (kind bit 2 clear) or a host node (bit 2
     set), each pair once; written up to `capacity`, *count the number needed.
   0, EINVAL where an edge runs backward in the order, ENOMEM. */
struct mesh_graph_crossing { uint32_t waiter, kind, producer; };
int mesh_graph_realize(uint32_t n, const unsigned char *host, size_t m, const uint32_t (*edge)[2], uint32_t *part,
                       uint32_t *parts, unsigned char *barrier, struct mesh_graph_crossing *waits, size_t capacity,
                       size_t *count);
/* Greedy by size (mesh-graph.c's section ported from TensorFlow Lite's GreedyBySizeAssignment, Apache-2.0):
   `count` items (indices into size and offset) on one line of units from 0, the `placed` pinned items already at their
   offsets. The largest first (ties in the given order), each in the smallest gap left of an item it is not apart from
   that holds it, else after the rightmost such item. type: -> offset[item] for each of `items`, *total the line's
   extent; 0, ENOMEM where *total passes capacity. */
int mesh_graph_greedy_by_size(size_t count, const size_t *items, size_t placed, const size_t *pinned,
                              const uint64_t *size, int (*apart)(const void *context, size_t k, size_t l),
                              const void *context, uint64_t capacity, uint64_t *offset, uint64_t *total);
/* type: n x (size, key, contiguous, pin), bins [first, first + pages) ascending and disjoint, apart -> each item's runs
   (first position, pages), item k's at runs[run_first[k] .. run_first[k + 1]) ascending, *runs the callee's (free it):
   - a pinned item (pin[k] != UINT64_MAX) is one run at its pin, inside one bin;
   - a contiguous item (a contiguous_input or contiguous_output function reads or writes it) is one run in the last
     bin, the arena, placed by mesh_graph_greedy_by_size with the pins in the arena already on its line;
   - every other item takes any free positions, one page at a time, in any bins: no function addresses it but
     through its pages;
   and no two items not apart share a position. apart(k, l) is symmetric: one's accesses all happen before the
   other's. Pins first; then the contiguous items, greedy by size in the arena (Pisarchyk and Lee 2020, TensorFlow
   Lite's code; ties by key); then the others in key order (a topological order of first accesses), page by page from
   the lowest free position of any bin. Page by page leaves no external fragmentation, so the
   hardness of dynamic storage allocation touches the contiguous items alone. The exclusion relation is Desnos et
   al.'s (2013). Zero-size items get no run. 0; EINVAL where a pin lies in no bin; EBUSY where two pins meet unapart;
   ENOMEM where the bins cannot hold an item: compile-time failures. Consumer: the run's solve (mesh-page-map.h). */
int mesh_graph_place(size_t n, const uint64_t *size, const uint64_t *key, const unsigned char *contiguous,
                     const uint64_t *pin, size_t bins, const uint64_t (*bin)[2],
                     int (*apart)(const void *context, size_t k, size_t l), const void *context, size_t *run_first,
                     uint64_t (**runs)[2]);

#ifdef __cplusplus
}
#endif
#endif
