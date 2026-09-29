#ifndef MESH_COLLECTIVE_H
#define MESH_COLLECTIVE_H
#include "mesh-call.h"

/* An explicit link map, one small text file:
     <kind> <nodes>             kind is mesh, ring or tree
     <a> <b> [<alpha> <beta>]   one link per line, a to b and its crossing cost (alpha us a message,
                                beta ns a byte [Hockney 1994]); '#' starts a comment line
     node <id> ...              a node of the map (tools read the rest of the line)
   ring: b follows a around the ring.  tree: a is b's parent; the root is nobody's child.  mesh: every pair is linked.
   mesh_collective_choose picks among every algorithm whose links the map has, by the operand's size.
   graph (a link table's map, mesh_link_table_map): any links, a ring in rank order where they carry one
   and spanning trees by breadth. */
enum { MESH_LINKS_MESH, MESH_LINKS_RING, MESH_LINKS_TREE, MESH_LINKS_GRAPH };
/* Any number of nodes and links: `link` is the links' storage, the caller's, or mesh_link_map_read's
   (freed by mesh_link_map_free).  `cost`, where given, is each directed pair's (alpha, beta), node a to
   node b at [a * nodes + b], which the planner weighs instead of the one alpha and beta it is passed. */
struct mesh_link_map { uint32_t kind,nodes,links; uint32_t (*link)[2]; const float (*cost)[2]; };
int mesh_link_map_read(const char *path,struct mesh_link_map *);
void mesh_link_map_free(struct mesh_link_map *);
/* The link table's (mesh-dataflow.h) stated configuration from a link-map file: its nodes present (the
   kind line's 0..nodes-1 and every node line's), each link line a to b stated with its alpha and beta;
   what the file does not state is not stated.  Every link's up stays as observed (the bridge's own, the
   other nodes' reports): stating a link never makes it up.  EINVAL where a link line states no cost
   (nothing infers one) or a node lies past the table's nodes. */
int mesh_link_table_state(struct mesh_link_table *,const char *path);
/* The planner's map of a snapshot over `nodes` (rank r is node nodes[r]): ranks a and b linked where
   both directions are stated and up, mesh kind where every pair is, else graph; each directed pair's
   stated cost.  pairs: n (n - 1) / 2 entries, cost: n n, the caller's. */
void mesh_link_table_map(const struct mesh_link_contents *,const uint32_t *nodes,uint32_t n,struct mesh_link_map *,
  uint32_t (*pairs)[2],float (*cost)[2]);

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
   segment is `segments[v]` elements after the segments of the nodes before it (`segments`: one count
   a node, summing to the operand's elements: MPI_Reduce_scatter's recvcounts, MPI_Allgatherv's
   recvcounts in rank order), or without them the v-th of the operand's elements cut in `nodes` parts,
   the first elements%nodes of them one element longer.  how: the direct exchange (every pair linked), the ring's reduce-scatter + all-gather
   [Patarasuk & Yuan 2009] (a ring map's cycle, or rank order where the map links it; either half
   alone), a spanning tree's reduce + broadcast from `root` (any connected map: a tree map's own
   tree, a star; the reduce alone, or a gather + broadcast, or a reduce + scatter), or the binomial
   tree's [Thakur, Rabenseifner & Gropp 2005] (the rank bits' pairs, a mesh's; its reduce alone).  A partial
   combination a node sends on to be combined further (a ring's reduce-scatter after its first round,
   a tree's interior node going up) is typed at `accumulator_bytes` an element (0: the operand's):
   half partials summed in float cross as float.  A step combines any number of receives. */
enum { MESH_ALLREDUCE, MESH_BROADCAST, MESH_REDUCE, MESH_REDUCE_SCATTER, MESH_ALLGATHER };
enum { MESH_DIRECT, MESH_RING, MESH_TREE, MESH_BINOMIAL, MESH_UNAVAILABLE };
struct mesh_collective { uint32_t what,how,root,accumulator_bytes; const uint64_t *contributors,*segments; };
#define MESH_COLLECTIVE_STEPS(nodes) (4*(nodes))
/* The rank's steps in its dependency order (at most MESH_COLLECTIVE_STEPS), each SEND's piece typed
   as the receiving step's; 0 where the map lacks a pair the algorithm uses, the map has fewer than
   two nodes, or a segment would be empty (fewer elements than nodes, a count of 0) or the segments
   do not sum to the operand. */
uint32_t mesh_collective_plan(const struct mesh_link_map *,uint32_t rank,struct mesh_collective,struct mesh_operand,struct mesh_step *steps);
/* Its time in microseconds, every node's plan run in the alpha-beta model [Hockney 1994] with a
   node's sends sharing one port and its receives another (alpha in us, beta in ns a byte: each link's
   own where the map has costs);
   negative where a node has no plan, a receive has no SEND of its piece, or the schedule stops. */
double mesh_collective_time(const struct mesh_link_map *,struct mesh_collective,struct mesh_operand,double alpha,double beta);
/* `c` with `how` (and an all-reduce's, reduce-scatter's or all-gather's tree `root`) of least time
   among the algorithms of the bit set c.how (0: every one) that the map carries; how
   MESH_UNAVAILABLE where none. */
struct mesh_collective mesh_collective_choose(const struct mesh_link_map *,struct mesh_collective c,struct mesh_operand,double alpha,double beta);
/* Binds a plan's steps onto the SEND/RECV transport (mesh_transfer_bind).  `operand` is the
   whole operand's section; `received` holds one section per REDUCE step, in step order, each
   sized for that step's piece.  identity+round is the transfer identity on both ends.  `pieces`,
   when given, receives every step's bound section in step order: what a SEND publishes and where a
   REDUCE or COPY arrives. */
int mesh_collective_bind(struct mesh_ctx *,const struct mesh_step *steps,uint32_t count,uint32_t identity,
  struct mesh_section operand,const struct mesh_section *received,uint32_t invocations,uint32_t invocation_pages,
  struct mesh_section *pieces);

/* A host (CPU) program's side of the prepared transfers, as mesh-metal.m is a GPU program's.
   mesh_host_inputs gives every received chunk one completion word per invocation (M07) and the
   link's cancellation range (M12); call it between mesh_transfers_prepare and mesh_transfers_start.
   mesh_host_publish releases invocation t of a bound SEND section.  mesh_host_arrived acquires
   invocation t of a bound receive section: 0 until it lands, then 1, or UINT64_MAX once its link
   was cancelled. */
int mesh_host_inputs(struct mesh_ctx *);
void mesh_host_publish(struct mesh_ctx *,struct mesh_section,uint32_t invocation);
uint64_t mesh_host_arrived(struct mesh_ctx *,struct mesh_section,uint32_t invocation);
#endif
