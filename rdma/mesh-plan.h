#ifndef MESH_PLAN_H
#define MESH_PLAN_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* The mesh's read side and its collective planning (libmesh: mesh-dataflow.c, mesh-collective.c), one header that
   every consumer uses as written: C and Objective-C include it, Swift imports it (module.modulemap), torch-mesh's
   extension includes it, Python binds it through mesh_c_build.py (cffi, compiled against this file).  Plain C:
   declarations and integer constants only.  Every execution reads its node's bridge here when it starts, and no
   consumer gathers, parses or joins bridge records itself. */

/* This node's bridge (design/algorithm-sources.md#meshobserve): its node, and its ports, view i the port it gave index
   i: the peer node while the link is paired (0 else), the link's phase and the port's bandwidth in bits a second (the
   RDMA port's active speed and width); returns the port count (at most `capacity` written; 0: the node alone), or
   -errno where the bridge is unreachable. */
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_STOPPED };
struct mesh_link_view { uint32_t peer,phase; uint64_t bandwidth; };
int mesh_observe(struct mesh_link_view *out,uint32_t capacity,uint32_t *node);

/* A link map, an operand: `nodes` nodes and `links` links of a kind.
     MESH_LINKS_MESH   every pair is linked (`link` lists only the pairs given costs, if any)
     MESH_LINKS_RING   link[i][1] follows link[i][0] around one cycle through every node
     MESH_LINKS_TREE   link[i][0] is link[i][1]'s parent; the root is nobody's child
     MESH_LINKS_GRAPH  the links as given, either way, any connected graph (a ring, a tree or a mesh's links
                       among them, unoriented)
   cost (NULL: the caller's alpha and beta for every link) is link i's alpha (us a message) and beta (ns a byte)
   either way [Hockney 1994]; a pair the map links without a cost takes the caller's.  `link` and `cost` are the
   caller's storage. */
enum { MESH_LINKS_MESH, MESH_LINKS_RING, MESH_LINKS_TREE, MESH_LINKS_GRAPH };
struct mesh_link_map { uint32_t kind,nodes,links; uint32_t (*link)[2]; double (*cost)[2]; };
/* 0 where a map's algorithm can run on it, else EINVAL: every node < nodes; a ring one cycle through every node
   with one link into each; a tree nodes-1 links, one into each node but the root, every node reaching the root; a
   graph connected. */
int mesh_link_map_check(const struct mesh_link_map *);
/* A link's cost where nothing measured it: alpha the TB5 crossing's one-way latency through the bridges (us a message;
   metal-microbench docs/measurement.md, the 3 KB host round trip 10.29 us), beta where its ports' bandwidth is unknown
   (ns a byte). */
#define MESH_ALPHA_US 5.0
#define MESH_BETA_NS 0.1
/* The map of the node's bridge (design/bridge-and-ipc.md#ports) as of one map serial, from its records of every node's
   ports (mesh_observe's views), so the same records give every member the same map: two nodes linked where each one's
   bridge holds a port paired with the other, at MESH_ALPHA_US and 8e9 over the widest such port's bandwidth
   (MESH_BETA_NS where none is known); node[i] (node NULL: none) map node i's bridge.  n 0: the whole mesh, every node
   the records hold, ascending.  n members (distinct bridges): map node i members[i]; while their links leave them in
   parts, of the pairs across two parts a route joins, the one whose route costs least (its links' alphas summed, then
   betas; the lowest-numbered of equals) is linked at that cost [Kruskal 1956], its route the whole mesh's path from
   the lower to the higher on fewest links (mesh_trees_shortest's parent rule), which the bridges forward along.  The
   kind MESH_LINKS_MESH where every pair is linked, else MESH_LINKS_GRAPH.  On entry map->nodes and map->links are the
   room of `node` and of map->link and map->cost (n and n x (n - 1) / 2 suffice for members); returns the map's nodes,
   or -errno (ENOSPC: the room is short, map->nodes and map->links then the room needed).  name[i] (name NULL: none)
   map node i's host name as its bridge's record holds it, NUL-terminated ("" for a node with no record); its room is
   map->nodes, as node's.  MESH_NAME_BYTES: a LocalHostName of up to 63 bytes and its NUL, the record's name field
   (mesh.h struct mesh_record). */
#define MESH_NAME_BYTES 64
int mesh_map(const uint32_t *members,uint32_t n,uint32_t *node,char (*name)[MESH_NAME_BYTES],struct mesh_link_map *map);
/* The bridge's map serial (never 0; it changes with the records): waits until it differs from *serial or bound_ns
   passes and writes it there; 0 where it differs, ETIMEDOUT where the bound passed, else an unreachable bridge's errno.
   Read (*serial 0) before mesh_map, it makes a later change end the next wait at once. */
int mesh_map_wait(uint64_t *serial,uint64_t bound_ns);
/* Whether the map links a and b, either way (a mesh every pair). */
int mesh_link_between(const struct mesh_link_map *,uint32_t a,uint32_t b);
/* The link a-b's own cost where the map gives one (either way), else alpha and beta. */
void mesh_link_cost(const struct mesh_link_map *,uint32_t a,uint32_t b,double alpha,double beta,double *link_alpha,double *link_beta);

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
