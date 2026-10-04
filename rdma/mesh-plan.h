#ifndef MESH_PLAN_H
#define MESH_PLAN_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* The mesh's topology and collective planning (libmesh: mesh-collective.c, mesh-dataflow.c), one header that every
   consumer uses as written: C and Objective-C include it, Swift imports it (module.modulemap), torch-mesh's
   extension includes it, Python binds it through mesh_c_build.py (cffi, compiled against this file).  Plain C:
   declarations and integer constants only. */

/* A node's links as its bridge reports them (design/algorithm-sources.md#meshobserve): the peer node, the link's
   phase, its device and its port's bandwidth in bits a second (the RDMA port's active speed and width); returns
   the node's link count (at most `capacity` written) and its node, or -errno. */
struct mesh_link_view { uint32_t peer,phase; char device[32]; uint64_t bandwidth; };
int mesh_observe(const char *name,struct mesh_link_view *out,uint32_t capacity,uint32_t *node);

/* A link map, an operand: `nodes` nodes and `links` links of a kind.
     MESH_LINKS_MESH   every pair is linked (`link` lists only the pairs given costs, if any)
     MESH_LINKS_RING   link[i][1] follows link[i][0] around one cycle through every node
     MESH_LINKS_TREE   link[i][0] is link[i][1]'s parent; the root is nobody's child
     MESH_LINKS_GRAPH  the links as given, either way, any connected graph (a ring, a tree or a mesh's links
                       among them, unoriented)
   cost (NULL: the caller's alpha and beta for every link) is link i's alpha (us a message) and beta (ns a byte)
   either way [Hockney 1994]; a pair the map links without a cost takes the caller's.  `link` and `cost` are the
   caller's storage, or mesh_link_map_read's (freed by mesh_link_map_free). */
enum { MESH_LINKS_MESH, MESH_LINKS_RING, MESH_LINKS_TREE, MESH_LINKS_GRAPH };
struct mesh_link_map { uint32_t kind,nodes,links; uint32_t (*link)[2]; double (*cost)[2]; };
/* 0 where a map's algorithm can run on it, else EINVAL: every node < nodes; a ring one cycle through every node
   with one link into each; a tree nodes-1 links, one into each node but the root, every node reaching the root; a
   graph connected. */
int mesh_link_map_check(const struct mesh_link_map *);
/* A map written as text, for programs that keep one in a file (metal-microbench's engine drivers):
     <kind> <nodes>                    kind is mesh, ring, tree or graph
     <a> <b> [<alpha> <beta>]          one link per line, with its cost; '#' starts a comment line */
int mesh_link_map_read(const char *path,struct mesh_link_map *);
void mesh_link_map_free(struct mesh_link_map *);
/* Whether the map links a and b, either way (a mesh every pair). */
int mesh_link_between(const struct mesh_link_map *,uint32_t a,uint32_t b);

/* Spanning trees over a map's nodes: the routes every collective and every message take (Blink's tree packing
   [Wang et al., MLSys 2020]; ForestColl [Zhao et al. 2024]).  Tree t is rooted at root[t]; node v's parent in it is
   parent[t * nodes + v], a node the map links v to (the root's is the root); log_weight[t] is its unnormalised log
   share of what its root gathers and sends.  The trees of one root split that root's elements by exp(log_weight), in
   tree order, each the floor of its share and the first ones one element more for the remainder.  A reduction toward
   r runs up r's trees, a broadcast from r down them, a message to r along the path from its sender in r's heaviest
   tree (the lowest-numbered of equals).  The trees are an operand, made ahead of time (mesh_trees_pack, or the
   caller's own) and changed only by making them again. */
struct mesh_trees { uint32_t nodes,count; const uint32_t *root,*parent; const double *log_weight; };
/* 0 where the trees fit the map: each spans every node over the map's links and reaches its root, and every node
   roots at least one (a collective reduces toward and broadcasts from every node); else EINVAL. */
int mesh_trees_check(const struct mesh_link_map *,const struct mesh_trees *);
/* A message's path from `from` to `to` in to's heaviest tree, the nodes from `from` to `to` inclusive (room for
   `nodes`); their count, 0 where `to` roots no tree. */
uint32_t mesh_trees_path(const struct mesh_trees *,uint32_t from,uint32_t to,uint32_t *path);
/* Each node's spanning in-arborescences packed fractionally into the map's links: Garg and Konemann's maximum
   concurrent flow [FOCS 1998], every root's trees together carrying one unit, each phase sending every root's unit
   along minimum-length arborescences [Chu & Liu 1965; Edmonds 1967], a link's length (one each way) growing by
   (1 + eps x flow/capacity), its capacity 1/beta (1 where the map gives no cost); a root's equal trees merged, its
   `most` heaviest kept and their weights its shares.  Writes root, parent (nodes a tree) and log_weight for at most
   nodes x most trees; returns the count, 0 where the map is not connected.  At two nodes, each node's one tree. */
uint32_t mesh_trees_pack(const struct mesh_link_map *,double eps,uint32_t most,uint32_t *root,uint32_t *parent,double *log_weight);

/* A typed operand: `type` is the caller's own element-type code, carried through untouched. */
struct mesh_operand { uint32_t type,element_bytes; uint64_t elements; };
/* One step of one rank's collective, in that rank's dependency order.
   SEND   publishes operand elements [first, first+piece.elements) to peer.
   REDUCE receives that typed piece from peer into the next caller-supplied received section;
          the caller's supplied reduction adds it into the operand at `first` before any later
          step that reads those elements.  In place is safe: in a tree whatever a REDUCE or COPY
          brings in was caused by the arrival of every earlier SEND of those elements, and two
          trees carry disjoint elements.
   COPY   receives that typed piece from peer directly into the operand at `first`.
   round  is the step's position in the algorithm; sender and receiver agree on it. */
enum { MESH_STEP_SEND, MESH_STEP_REDUCE, MESH_STEP_COPY };
struct mesh_step { uint32_t op,peer,round,padding; uint64_t first; struct mesh_operand piece; };

/* A collective of one typed operand along trees (metal-microbench docs/kernels.md Collectives).  what:
   MESH_REDUCE_SCATTER leaves at node v the combination of its segment v, reduced up v's trees;
   MESH_ALLGATHER gives every node every node's segment v, broadcast down v's trees; MESH_ALLREDUCE is the
   reduce-scatter and then the all-gather, combining only the nodes of the bit set `contributors` (a bit a
   node, 64 a word; NULL: every node); MESH_REDUCE combines the whole operand at `root` up root's trees;
   MESH_BROADCAST gives every node root's operand down root's trees.  Node v's segment is the v-th of the
   operand's elements cut in `nodes` parts, the first elements%nodes of them one element longer.  Going up a
   tree a node sends to its parent at round its height, after its children (each at its own height); coming
   down a node at depth d receives at round base + d - 1 and sends at base + d, base the reduce's last round
   plus one.  A round's SENDs come before its receives, so a SEND waits on lower rounds alone.  A partial
   combination a node sends on (an interior node going up) is typed at `accumulator_bytes` an element (0: the
   operand's): half partials summed in float cross as float.  A step combines any number of receives. */
enum { MESH_ALLREDUCE, MESH_BROADCAST, MESH_REDUCE, MESH_REDUCE_SCATTER, MESH_ALLGATHER };
struct mesh_collective { uint32_t what,root,accumulator_bytes,padding; const uint64_t *contributors; };
/* The most steps a rank's plan has along `trees` (the storage mesh_collective_plan's steps need). */
uint32_t mesh_collective_steps(const struct mesh_trees *);
/* The rank's steps in its dependency order (at most mesh_collective_steps), each SEND's piece typed
   as the receiving step's; 0 where the trees have fewer than two nodes or the rank is not one. */
uint32_t mesh_collective_plan(const struct mesh_trees *,uint32_t rank,struct mesh_collective,struct mesh_operand,struct mesh_step *steps);
/* Its time in microseconds, every node's plan run in the alpha-beta model [Hockney 1994] with a
   node's sends sharing one port and its receives another (alpha in us, beta in ns a byte, a link's
   own cost where the map gives one); negative where a node has no plan, a receive has no SEND of its
   piece, or the schedule stops. */
double mesh_collective_time(const struct mesh_link_map *,const struct mesh_trees *,struct mesh_collective,struct mesh_operand,double alpha,double beta);

#ifdef __cplusplus
}
#endif
#endif
