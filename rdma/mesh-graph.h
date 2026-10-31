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
#endif
