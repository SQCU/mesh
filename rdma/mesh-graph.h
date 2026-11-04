/* design/rubrics.md:422 (C9): the graph algorithms, one copy each, for every consumer of weights or samples on a graph.
   A function here knows vertices, edges and the operands its consumer passes (weights, visits, a next hop); it knows
   no cable, collective or tensor.  Python's are rdma/mesh_graph.py's. */
/* Source: pre-return-20261103 rdma/mesh-graph.h, the declarations of the functions mesh-graph.c brings, verbatim;
   mesh_graph_list_schedule is new (design/page-queue-change-request.md, "The rule both ends compute");
   mesh_graph_happens_before and mesh_graph_left_edge are new (design/disjoint-pages-change-request.md, Appendix A,
   A3, "Interference, by happens-before" and "The assignment"; design/disjoint-pages-coordination.md, "The
   allocator"); mesh_graph_lags is new (design/disjoint-pages-change-request.md, Appendix C, C3.3;
   design/disjoint-pages-revision-coordination.md, D5 and "The allocator"). */
#ifndef MESH_GRAPH_H
#define MESH_GRAPH_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* The vertices `from` reaches over `edges` undirected edges, edge e joining `end(context, e, 0)` and
   `end(context, e, 1)` (each below `vertices`), breadth first: their count; `order` (room for `vertices`) the vertices
   in the order reached, `reached` (room for `vertices`) 1 for each and 0 for every other. */
uint32_t mesh_graph_reach(uint32_t vertices, uint32_t edges, uint32_t (*end)(const void *context, uint32_t e, uint32_t side),
                          const void *context, uint32_t from, uint32_t *order, unsigned char *reached);
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
   its pieces lies within one of the other's (libnccl-mesh's spread over a pair's parallel cables: a forward's piece
   within one COPY's piece, the cell's index choosing the channel alike at both ends). */
uint32_t mesh_graph_grid_cut(uint64_t first, uint64_t length, uint64_t cell, uint64_t *start, uint64_t *count, uint64_t *index);
/* A tree's depth (its root root, parents parent[..], a vertex off it UINT32_MAX): the most edges from the root to a
   vertex on it, mesh_graph_depth's over each vertex's parent; UINT32_MAX where its parents hold a cycle */
uint32_t mesh_graph_tree_depth(uint32_t vertices, uint32_t root, const uint32_t *parent);
/* The rate every root gets at once where each root's flow is cut over its trees by their weights (weights a root's
   summing to one): the least, over directed edges u -> v, of capacity[u * vertices + v] over the trees' load there
   (mesh_graph_tree_loads), in capacity's units; 0 where an edge a tree crosses has no capacity. */
double mesh_graph_trees_rate(uint32_t vertices, uint32_t trees, const uint32_t *root, const uint32_t *parent,
                             const double *weight, const double *capacity);
/* `units` equal units over `parts` parallel routes by earliest finish: part p takes its first unit latency[p] after
   the start and then one each 1 / rate[p] (rate in units a time); unit k goes where it would finish soonest, so a
   unit takes the quickest route and many spread over every route with rate, a slower route taking fewer of the first
   (makespan of equal jobs on uniform machines with release times, which earliest finish solves exactly). count[p]
   its units; their completion time. A part of no rate takes none. */
double mesh_graph_fill(uint32_t parts, const double *latency, const double *rate, uint64_t units, uint64_t *count);
/* Units in list order over `links` parallel links by earliest finish (Graham's list scheduling: R. L. Graham, "Bounds
   for certain multiprocessing anomalies", Bell System Technical Journal 45 (1966); each link a machine of its own
   rate, so equal rates give Graham's least-loaded machine): unit k, bytes[k] long, goes to the link whose bytes with
   unit k's added, over rate[l] (bytes a time), end soonest, a tie to the lower l, so parallel links of unequal rate end
   together (a node pair's links, the caller's in order of cable id and then queue-pair index, both ends assigning one
   list alike: its arithmetic is +, / and < alone, correctly rounded, so every node computes the same links from the
   same operands). link[k] its link (UINT32_MAX where no link has rate), carried[l] (room for `links`) the bytes each
   takes; their completion time, INFINITY where a unit has no link. A link of no rate takes none. */
double mesh_graph_list_schedule(uint32_t links, const double *rate, uint64_t units, const uint64_t *bytes,
                                uint32_t *link, uint64_t *carried);
/* Happens-before over a dependence graph repeated `iterations` times, its edges carrying iteration distances (M. Lam,
   PLDI 1988; B. R. Rau, MICRO 1994): template vertex v of iteration i is (v, i); template edge k, in the row
   [first[v], first[v + 1]) of the vertex v it leaves, joins (v, i) to (head(context, k), i + lag(context, k)) for every
   i, lag >= 0.  For source[s] (s < sources) and target[x] (x < targets), bit L % 64 of
   reach[(s * targets + x) * words + L / 64] (words = (iterations + 63) / 64) is set where a path of one or more edges
   joins (source[s], i) to (target[x], i + L) for i + L < iterations (the same for every such i).  0; EDEADLK where the
   lag-0 edges hold a cycle; ENOMEM. */
int mesh_graph_happens_before(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                              uint32_t (*lag)(const void *context, size_t k), const void *context, uint32_t iterations,
                              uint32_t sources, const uint32_t *source, uint32_t targets, const uint32_t *target,
                              uint64_t *reach);
/* The least total lag over walks of one or more edges from source[s] to target[x] (s < sources, x < targets), in the
   directed graph whose vertex v's edges are [first[v], first[v + 1]), edge k entering head(context, k) and carrying
   lag(context, k) >= 0 (E. W. Dijkstra, A note on two problems in connexion with graphs, Numerische Mathematik 1,
   1959): lags[s * targets + x], UINT32_MAX where no walk joins them.  A source that is its own target is given its
   least cycle.  0; ENOMEM */
int mesh_graph_lags(uint32_t vertices,const size_t *first,uint32_t (*head)(const void *context,size_t k),
                    uint32_t (*lag)(const void *context,size_t k),const void *context,uint32_t sources,
                    const uint32_t *source,uint32_t targets,const uint32_t *target,uint32_t *lags);
/* The left-edge colouring of items that each need a run of colours (T. Hashimoto and J. Stevens, DAC 1971; M. C.
   Golumbic 1980; with sizes, greedy by size: Y. Pisarchyk and J. Lee, arXiv 2001.03288, 2020): the items in the
   caller's order order[0 .. items), item i given colours [colour[i], colour[i] + size[i]) at the lowest colour where
   its run overlaps no item placed before it that it interferes with (interferes(context, a, b) != 0, symmetric);
   *used the most colour[i] + size[i].  `interferes` NULL: every pair interferes, so each item in that order takes the
   colours just past the runs before it (colour[order[0]] 0, each next the end of the runs before it, the sum of their
   sizes; an empty run colour 0), in O(items).  0; ENOSPC where a run would pass `capacity`; ENOMEM. */
int mesh_graph_left_edge(uint32_t items, const uint64_t *size, const uint32_t *order,
                         int (*interferes)(const void *context, uint32_t a, uint32_t b), const void *context,
                         uint64_t capacity, uint64_t *colour, uint64_t *used);

#ifdef __cplusplus
}
#endif
#endif
