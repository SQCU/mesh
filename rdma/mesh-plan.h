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
/* One step of one rank's part of a collective, in that rank's order (a compiled program's moves made elements,
   mesh_program_steps).
   SEND   publishes elements [first, first + piece.elements) of `buffer` to peer.
   REDUCE receives that typed piece from peer and combines it into `buffer` at `first`.
   COPY   receives that typed piece from peer into `buffer` at `first`.
   buffer 0 is the operand, the result made in place; b >= 1 the rank's scratch b, a copy of the operand made at the
   call's start.  round is the step's position: sender and receiver agree on it, and a rank's k-th SEND to a peer
   in a round is that peer's k-th receive from it in the round. */
enum { MESH_STEP_SEND, MESH_STEP_REDUCE, MESH_STEP_COPY };
struct mesh_step { uint32_t op,peer,round,buffer; uint64_t first; struct mesh_operand piece; };

/* A compiled collective, the operand a call runs: every rank's moves, as MSCCL-IR is the program the MSCCL runtime
   interprets [Cowan et al., ASPLOS 2023].  `what` it computes (MESH_*), its `root` (a reduce's or a broadcast's) and
   `nodes`; its operand cut into `parts` parts in order, part p within segment segment[p] at log_weight[p], its share
   there (the operand cut in as many segments as the parts name, the first elements % segments one element longer, so
   that a reduce-scatter's and an all-gather's segment s is node s's NCCL segment; within a segment each part the floor
   of its share, the first ones one element more for the remainder).  Rank r's moves are move[first[r]] ..
   move[first[r + 1] - 1], each over parts [part, part + parts) of a buffer (struct mesh_step's); `scratch` is the most
   scratch buffers a rank uses.  Every rank runs the same program; nothing about it is decided where it runs. */
enum { MESH_ALLREDUCE, MESH_BROADCAST, MESH_REDUCE, MESH_REDUCE_SCATTER, MESH_ALLGATHER };
struct mesh_move { uint32_t op,peer,round,buffer,part,parts; };
struct mesh_program { uint32_t what,root,nodes,parts,scratch,padding; const uint32_t *segment; const double *log_weight;
                      const uint32_t *first; const struct mesh_move *move; };

/* Shortest-path trees: each node's in-tree over the map's links on fewest links (a node's parent, of its neighbours
   one link nearer the root, the one over the cheapest link: its alpha, then its beta, then the lowest-numbered), one
   tree a node at log weight 0; writes nodes trees, returns their count (0 where the map is not connected). */
uint32_t mesh_trees_shortest(const struct mesh_link_map *,uint32_t *root,uint32_t *parent,double *log_weight);
/* The room mesh_compile needs: parts and moves. */
void mesh_compile_room(const struct mesh_trees *,uint32_t *parts,uint32_t *moves);
/* A collective compiled along `trees` into `program` (arrays of mesh_compile_room's room: segment and log_weight a
   part, first nodes + 1, move a move):
     MESH_REDUCE           the whole operand up root's trees, a part a tree (its weight);
     MESH_BROADCAST        the whole operand down root's trees;
     MESH_REDUCE_SCATTER   segment s up s's trees;  MESH_ALLGATHER  segment s down s's trees;
     MESH_ALLREDUCE        whole 0: the reduce-scatter and then the all-gather (each link a share once each way; its
                           dependent crossings twice the trees' heights) [Patarasuk & Yuan 2009; Blink];
                           whole 1: the whole operand reduced toward every node at once, up that node's heaviest tree,
                           a node forwarding its partial for another from a scratch buffer of its own and its own
                           operand from a copy (its dependent crossings the trees' height: the graph's eccentricity
                           along shortest-path trees, 1 where every pair is linked, the direct exchange; each node
                           sends the whole operand once for every other) [single-phase all-reduce: Trivance 2026].
   Going up a tree a node sends at round its height, after its children; coming down, a node at depth d receives at
   base + d - 1 and sends at base + d, base the reduce-scatter's last round plus one; a round's SENDs come before its
   receives, so a SEND waits on lower rounds alone.  A message is coalesced with the next between the same two nodes
   in the same round where their parts adjoin and each end's buffer and step are the same.  `contributors` (a bit a
   node, 64 a word; NULL: every node) are the nodes whose operands an all-reduce combines, and the nodes whose
   segments an all-gather gathers: the operand is theirs in node order, each down its node's trees to every node,
   the others holding none.  0, or EINVAL. */
int mesh_compile(const struct mesh_trees *,uint32_t what,uint32_t root,int whole,const uint64_t *contributors,
                 struct mesh_program *program,uint32_t *segment,double *log_weight,uint32_t *first,struct mesh_move *move);
/* A rank's steps of a program for an operand, a step a move (none where its elements are none), at most first[rank +
   1] - first[rank]; their count. */
uint32_t mesh_program_steps(const struct mesh_program *,uint32_t rank,struct mesh_operand,struct mesh_step *steps);
/* A program's time in microseconds, every rank's steps run in the alpha-beta model [Hockney 1994] (each link its own
   ports, one message at a time each way: a node sends and receives on all its links at once; alpha in us, beta in ns
   a byte, a link's own cost where the map gives one);
   negative where a receive has no SEND of its piece or the schedule stops. */
double mesh_program_time(const struct mesh_link_map *,const struct mesh_program *,struct mesh_operand,double alpha,double beta);

#ifdef __cplusplus
}
#endif
#endif
