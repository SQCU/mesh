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
   either way [Hockney 1994]; a pair the map links without a cost takes the caller's.  mesh_collective_choose
   picks among every algorithm whose links the map has, by the operand's size.  `link` and `cost` are the
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

/* A typed operand: `type` is the caller's own element-type code, carried through untouched. */
struct mesh_operand { uint32_t type,element_bytes; uint64_t elements; };
/* One step of one rank's collective, in that rank's dependency order.
   SEND   publishes operand elements [first, first+piece.elements) to peer.
   REDUCE receives that typed piece from peer into the next caller-supplied received section;
          the caller's supplied reduction adds it into the operand at `first` before any later
          step that reads those elements.  In place is safe on a ring or tree: whatever a REDUCE
          or COPY brings in was caused by the arrival of every earlier SEND of those elements.
          On a mesh (direct exchange) it is not, so the sum goes to the caller's result.
   COPY   receives that typed piece from peer directly into the operand at `first`.
   round  is the step's position in the algorithm; sender and receiver agree on it. */
enum { MESH_STEP_SEND, MESH_STEP_REDUCE, MESH_STEP_COPY };
struct mesh_step { uint32_t op,peer,round,padding; uint64_t first; struct mesh_operand piece; };

/* A collective of one typed operand among a map's nodes (metal-microbench docs/kernels.md
   Collectives).  what: MESH_ALLREDUCE combines the contributions of the nodes of the bit set
   `contributors` (a bit a node, 64 a word; NULL: every node) at every node; MESH_BROADCAST gives every node the operand of
   `root`; MESH_REDUCE combines them at `root` alone; MESH_REDUCE_SCATTER leaves at node v the
   combination of its segment v; MESH_ALLGATHER gives every node every node's segment v.  Node v's
   segment is the v-th of the operand's elements cut in `nodes` parts, the first elements%nodes of
   them one element longer.  how: the direct exchange (every pair linked), the ring's reduce-scatter + all-gather
   [Patarasuk & Yuan 2009] (a ring map's cycle, or rank order where the map links it; either half
   alone), a spanning tree's reduce + broadcast from `root` (any connected map: a tree map's own
   tree, a star; the reduce alone, or a gather + broadcast, or a reduce + scatter), or the binomial
   tree's [Thakur, Rabenseifner & Gropp 2005] (the rank bits' pairs, a mesh's; its reduce alone).  A partial
   combination a node sends on to be combined further (a ring's reduce-scatter after its first round,
   a tree's interior node going up) is typed at `accumulator_bytes` an element (0: the operand's):
   half partials summed in float cross as float.  A step combines any number of receives. */
enum { MESH_ALLREDUCE, MESH_BROADCAST, MESH_REDUCE, MESH_REDUCE_SCATTER, MESH_ALLGATHER };
enum { MESH_DIRECT, MESH_RING, MESH_TREE, MESH_BINOMIAL, MESH_UNAVAILABLE };
struct mesh_collective { uint32_t what,how,root,accumulator_bytes; const uint64_t *contributors; };
/* The most steps a rank's plan has among `nodes` (the storage mesh_collective_plan's steps need). */
uint32_t mesh_collective_steps(uint32_t nodes);
/* The rank's steps in its dependency order (at most mesh_collective_steps), each SEND's piece typed
   as the receiving step's; 0 where the map lacks a pair the algorithm uses, the map has fewer than
   two nodes, or a ring's segment would be empty (fewer elements than nodes). */
uint32_t mesh_collective_plan(const struct mesh_link_map *,uint32_t rank,struct mesh_collective,struct mesh_operand,struct mesh_step *steps);
/* Its time in microseconds, every node's plan run in the alpha-beta model [Hockney 1994] with a
   node's sends sharing one port and its receives another (alpha in us, beta in ns a byte);
   negative where a node has no plan, a receive has no SEND of its piece, or the schedule stops. */
double mesh_collective_time(const struct mesh_link_map *,struct mesh_collective,struct mesh_operand,double alpha,double beta);
/* `c` with `how` (and an all-reduce's, reduce-scatter's or all-gather's tree `root`) of least time
   among the algorithms of the bit set c.how (0: every one) that the map carries; how
   MESH_UNAVAILABLE where none. */
struct mesh_collective mesh_collective_choose(const struct mesh_link_map *,struct mesh_collective c,struct mesh_operand,double alpha,double beta);

#ifdef __cplusplus
}
#endif
#endif
