/* design/rubrics.md:422 (C9): the graph algorithms, one copy each, for every consumer of weights or samples on a graph.
   A function here knows vertices, edges and the operands its consumer passes (weights, visits, a next hop); it knows
   nothing of what they stand for, no collective, tensor or transfer.  Python's are rdma/mesh_graph.py's. */
/* Source: pre-return-20261103 rdma/mesh-graph.h, the declarations of the functions mesh-graph.c brings, verbatim;
   mesh_graph_list_schedule is new (design/page-queue-change-request.md, "The rule both ends compute");
   mesh_graph_left_edge is new (design/disjoint-pages-change-request.md, Appendix A, A3, "The assignment";
   design/disjoint-pages-coordination.md, "The allocator"); mesh_graph_lags is new (design/disjoint-pages-change-request.md, Appendix C, C3.3;
   design/compilation-coordination.md, U1; design/disjoint-pages-revision-coordination.md, "The allocator");
   mesh_graph_bottom_levels and mesh_graph_contention_schedule are new, and mesh_graph_reach gains `parent`
   (design/plan-to-bridge-change-request.md, R5; design/compilation-coordination.md, "The graph module" and T3 to T5,
   Q3); mesh_graph_heap_order is new (design/plan-to-bridge-change-request.md, R5, "The order of each cable's list, by
   the reference algorithm"; design/compilation-coordination.md, "The documents' revisions of edd4b583", "The graph
   module" and GO3), its heap mesh_graph_lags' binary heap, one copy.
   2026-11-04, owner C2 (design/compilation-coordination.md, U1): mesh_graph_lags' source cited to C3.3 and U1, where it
   cited design/disjoint-pages-revision-coordination.md, D5.  Why: D5 appears in neither Lam, Rau nor C3.3's text
   (design/aot-compilation-change-request.md, appendix item 1); the least lags decide C3.3's tests at every larger
   distance by U1's reason, its consumer's (every walk into a write passes its range's production; the programs'
   dependences order each node's events of t before those of t + 1).  What must stay: the function as it is.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", LP; design/plan-to-bridge-change-
   request.md R5 at mesh 0e56d8da): mesh_graph_bottom_levels and mesh_graph_contention_schedule are removed, and
   mesh_graph_completion_times is new.  Why: R5's priority is "C̄_j, its completion time in the program's optimal
   solution, the least first", and "It replaces MSCCLang's depth and reverse depth, and edge scheduling's start times,
   which are no longer computed for the order"; R5's Code: "rdma/mesh-graph.{h,c}: the program's rows, the
   cutting-plane loop with Queyranne's separation, and the calls to HiGHS; the heap order by C̄ (`mesh_graph_heap_order`,
   as now)".  What must stay: mesh_graph_heap_order as it is, the heap one copy */
#ifndef MESH_GRAPH_H
#define MESH_GRAPH_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* The vertices `from` reaches over `edges` undirected edges, edge e joining `end(context, e, 0)` and
   `end(context, e, 1)` (each below `vertices`), breadth first: their count; `order` (room for `vertices`) the vertices
   in the order reached, `reached` (room for `vertices`) 1 for each and 0 for every other.  `parent` (room for
   `vertices`; NULL: none): each reached vertex's breadth-first parent, the vertex whose edge first reached it (`from`
   its own; a vertex not reached UINT32_MAX).  Callers pass NULL but mesh-records.c's island tree
   (design/compilation-coordination.md, Q3). */
uint32_t mesh_graph_reach(uint32_t vertices, uint32_t edges, uint32_t (*end)(const void *context, uint32_t e, uint32_t side),
                          const void *context, uint32_t from, uint32_t *order, unsigned char *reached, uint32_t *parent);
/* Directed edges in compressed rows: vertex v's leave as edges [first[v], first[v + 1]), edge k entering
   `head(context, k)`. */
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
/* The least-weight spanning arborescence of nodes [0, N) rooted at `root` in which every node lies within `depth`
   edges of the root (a hop-bounded arborescence; Chu-Liu/Edmonds' has no bound): chosen[v] the edge into v, the
   root's -1; 0, or -1 where no such arborescence spans.  Exact, by depth-first search over each node's edges in, a
   branch cut where its weight with each node left's lightest edge in reaches the best found or a node's walk to the
   root passes `depth`: its cost the product of the nodes' in-degrees at worst (the problem is NP-hard for depth 2
   and beyond), a few hundred leaves on a collective of five. */
int mesh_graph_arborescence_within(uint32_t N, uint32_t root, uint32_t E, const uint32_t *from, const uint32_t *to,
                                   const double *weight, uint32_t depth, int32_t *chosen);
/* The walk up a parent table from `from` (parent[v]: v's next vertex up): `visit(consumer, at, step)` for each vertex
   it meets, `from` at step 0, until one that is `root` (UINT32_MAX: none) or whose parent names none
   (parent[at] >= vertices), visited last; a vertex its own parent (outside a tree) climbs in place.  The steps to that vertex; UINT32_MAX where `vertices` steps meet none (a
   cycle), steps 0 to `vertices` visited.  `visit` NULL: none. */
uint32_t mesh_graph_climb(const uint32_t *parent, uint32_t vertices, uint32_t root, uint32_t from,
                          void (*visit)(void *consumer, uint32_t at, uint32_t step), void *consumer);
/* The length of the cycle a climb from `from` returns to `from` on (the first step past 0 it meets `from` again);
   0 where it does not within `vertices` steps. */
uint32_t mesh_graph_cycle_length(const uint32_t *parent, uint32_t vertices, uint32_t from);
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
/* design/joint-compilation.md §3, row 3 */
uint32_t mesh_graph_flow(uint32_t vertices, uint32_t edges, const uint32_t *from, const uint32_t *to, const double *capacity,
                         uint32_t count, const uint32_t *root, const double *up, const double *down,
                         int (*leaf)(const void *context, uint32_t d, uint32_t v), const void *context, double eps,
                         uint32_t rounds, uint32_t room, uint32_t *demand, uint32_t *parent, double *share);
/* Spanning trees packed by Garg and Koenemann's rounds (a communicator's own forms: mesh-plan.h mesh_trees_pack). Edge e
   makes vertex to[e] a child of from[e], its route crossing directed hops crossing[through[e]] to
   crossing[through[e + 1] - 1], hop k of capacity[k]. Every root r for which rooted(context, r) holds is given a demand
   of one, packed along least-weight arborescences (mesh_graph_arborescence) until the hops' lengths reach one. Each
   root's `most` heaviest distinct trees are kept, root by root: root[k], parents parent[k * vertices ..] and log
   weight log_weight[k], the log of its share of its root's kept flow. Their count; 0 where no arborescence spans.
   `shallow`: each root's arborescences are those within its eccentricity's depth alone (the most edges its
   breadth-first layering puts between it and a vertex; mesh_graph_arborescence_within), so each tree is as shallow as
   a tree from that root can be: where a hop costs a fixed time far past its bytes' (a small collective), a tree's
   time is its depth's hops. */
uint32_t mesh_graph_pack(uint32_t vertices, uint32_t edges, const uint32_t *from, const uint32_t *to, const uint32_t *through,
                         const uint32_t *crossing, uint32_t hops, const double *capacity,
                         int (*rooted)(const void *context, uint32_t r), const void *context, double eps, uint32_t most,
                         int shallow, uint32_t *root, uint32_t *parent, double *log_weight);
/* Trees' loads on their directed edges: each tree k (root root[k], parents parent[k * vertices ..]) of weight weight[k]
   adds weight[k] to load[u * vertices + v] for each of its edges u -> v (u v's parent); `load` is the caller's,
   vertices x vertices, added to. */
void mesh_graph_tree_loads(uint32_t vertices, uint32_t trees, const uint32_t *root, const uint32_t *parent,
                           const double *weight, double *load);
/* [first, first + length) cut at the multiples of `cell` (a grid over the whole operand): start[i], count[i] and the
   cell's index index[i] of each run; their number. A run inside another's range is cut at the same edges, so each of
   its pieces lies within one of the other's (a consumer that gives each cell, by its index, to one of several parallel
   machines gives an inner run's piece the machine of the outer piece holding it, and two parties cutting the same
   operands give it alike). */
uint32_t mesh_graph_grid_cut(uint64_t first, uint64_t length, uint64_t cell, uint64_t *start, uint64_t *count, uint64_t *index);
/* A tree's depth (its root root, parents parent[..], a vertex off it UINT32_MAX): the most edges from the root to a
   vertex on it, mesh_graph_depth's over each vertex's parent; UINT32_MAX where its parents hold a cycle */
uint32_t mesh_graph_tree_depth(uint32_t vertices, uint32_t root, const uint32_t *parent);
/* The rate every root gets at once where each root's flow is cut over its trees by their weights (weights a root's
   summing to one): the least, over directed edges u -> v, of capacity[u * vertices + v] over the trees' load there
   (mesh_graph_tree_loads), in capacity's units; 0 where an edge a tree crosses has no capacity. */
double mesh_graph_trees_rate(uint32_t vertices, uint32_t trees, const uint32_t *root, const uint32_t *parent,
                             const double *weight, const double *capacity);
/* design/joint-compilation.md §3, row 4 */
double mesh_graph_fill(uint32_t parts, const double *latency, const double *rate, uint64_t units, uint64_t *count);
/* design/joint-compilation.md §3, row 10 */
double mesh_graph_list_schedule(uint32_t machines, const double *rate, uint64_t units, const uint64_t *bytes,
                                uint32_t *machine, uint64_t *carried);
/* design/joint-compilation.md §3, row 5 */
int mesh_graph_lags(uint32_t vertices,const size_t *first,uint32_t (*head)(const void *context,size_t k),
                    uint32_t (*lag)(const void *context,size_t k),const void *context,uint32_t sources,
                    const uint32_t *source,uint32_t targets,const uint32_t *target,uint32_t *lags);
/* design/joint-compilation.md §3, row 7 */
int mesh_graph_chains(uint32_t items, const uint64_t *count, int (*before)(const void *context, uint32_t a, uint32_t b),
                      const void *context, uint64_t *chain, uint64_t *chains);
/* design/joint-compilation.md §3, row 7 */
int mesh_graph_fat_cover(uint64_t length, uint32_t intervals, const uint64_t *start, const uint64_t *end,
                         const uint64_t *need, uint64_t *color, uint64_t *colors);
/* design/joint-compilation.md §3, row 8 */
int mesh_graph_completion_times(uint32_t vertices, const double *length, const uint32_t *resource, uint32_t resources,
                                size_t edges, const uint32_t *from, const uint32_t *to, double *completion);
/* design/joint-compilation.md §3, row 8 */
int mesh_graph_heap_order(uint32_t vertices, const double *priority, size_t edges, const uint32_t *from,
                          const uint32_t *to, uint32_t *place);
/* 2026-11-05 (design/aot-compilation-change-request.md, appendix item 2, "Its room is fixed at compile time, by the
   self-timed buffer bound": "The strongly connected components, Convert-to-SC-graph, Bellman–Ford's cycle mean and
   MinDelay's binary search are functions of rdma/mesh_graph.py.  They take the caller's vertices, edges, delays and
   times as operands"; design/compilation-coordination.md, "The revisions of 3f106747", BS-1: here, the graph module's C
   half, since their one caller is mesh_plan, which holds the routes and u they are computed from).  S. S.
   Bhattacharyya, S. Sriram and E. A. Lee, Optimizing synchronization in multiprocessor DSP systems, IEEE Trans. Signal
   Processing 45(6), 1997 (here BSL), over a directed graph of `vertices` vertices whose edge e runs from[e] -> to[e]
   with delay[e] (UINT64_MAX: BSL's "∞", the edge absent from every cycle) and whose vertex v takes time[v] >= 0 (BSL
   §4.1, "execution time t(v)").  What must stay: integer and IEEE-exact operations alone (+, -, x, /, <), so every
   party computes the same components, means and delays from the same operands, as the module's other functions do */
/* The strongly connected components (BSL §3: "A strongly connected component (SCC) of (V, E) is a strongly connected
   subset V′ ⊆ V such that no strongly connected subset of V properly contains V′"), by R. E. Tarjan's depth-first
   search (Depth-first search and linear graph algorithms, SIAM J. Comput. 1(2), 1972), every edge counted whatever its
   delay: component[v] v's, *count of them, numbered in the order Tarjan's search completes them.  0; EINVAL where an
   edge names no vertex; ENOMEM */
int mesh_graph_components(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to,
                          uint32_t *component, uint32_t *count);
/* design/joint-compilation.md §3, row 6 */
int mesh_graph_cycle_mean(uint32_t vertices, const double *time, size_t edges, const uint32_t *from, const uint32_t *to,
                          const uint64_t *delay, double *mean);
/* design/joint-compilation.md §3, row 6 */
int mesh_graph_min_delay(uint32_t vertices, const double *time, size_t edges, const uint32_t *from, const uint32_t *to,
                         const uint64_t *delay, size_t e, double lambda, uint64_t bound, uint64_t *least);
/* design/joint-compilation.md §3, row 6 */
int mesh_graph_convert_sc(uint32_t vertices, const double *time, size_t edges, const uint32_t *from, const uint32_t *to,
                          uint32_t *added, uint32_t *added_from, uint32_t *added_to);
/* design/joint-compilation.md §3, row 6 */
int mesh_graph_determine_delays(uint32_t vertices, const double *time, size_t edges, const uint32_t *from,
                                const uint32_t *to, const uint64_t *delay, uint32_t added, const uint32_t *added_from,
                                const uint32_t *added_to, uint64_t *added_delay);
/* MPI 4.1 §6.1.11: "A datatype may specify overlapping entries.  The use of such a datatype in any communication in
   association with a buffer updated by the operation is erroneous."  interval[0 .. count), each [start, end) with
   start < end, sorted in place by key, then start; an interval is compared with those of its key alone.  1 where an
   interval begins before its predecessor of the same key ends, else 0 */
struct mesh_graph_interval { uint64_t key, start, end; };
int mesh_graph_intersect(size_t count, struct mesh_graph_interval *interval);

/* row[0 .. count) visited in order: visit(context, row[j]) */
void mesh_graph_row_visit(const uint32_t *row, uint32_t count, void (*visit)(void *context, uint32_t k), void *context);

#ifdef __cplusplus
}
#endif
#endif
