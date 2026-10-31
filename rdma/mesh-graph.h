/* design/rubrics.md:422 (C9): the graph algorithms, one copy each, for every consumer of weights or samples on a graph.
   A function here knows vertices, edges and the operands its consumer passes (weights, visits, a next hop); it knows
   no cable, collective or tensor.  Python's are rdma/mesh_graph.py's. */
#ifndef MESH_GRAPH_H
#define MESH_GRAPH_H
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
#endif
