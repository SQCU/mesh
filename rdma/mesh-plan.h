#ifndef MESH_PLAN_H
#define MESH_PLAN_H
#include <stdint.h>
#include "../etc/bridge/layout.h"
#ifdef __cplusplus
extern "C" {
#endif

/* The mesh's read side and its collective planning (libmesh: mesh-dataflow.c, mesh-collective.c), one header that
   every consumer uses as written: C and Objective-C include it, Swift imports it (module.modulemap), torch-mesh's
   extension includes it, Python binds it through mesh_c_build.py (cffi, compiled against this file).  Plain C:
   declarations and integer constants only.  Every execution reads its node's bridge here when it starts, and no
   consumer gathers, parses or joins bridge records itself. */

/* This node's bridge (design/transport.md section 14): its node, and the nodes it reaches, each with the bandwidth of
   its stream toward it in bits a second; returns their count (at most `capacity` written; 0: the node alone), or
   -errno where the bridge is unreachable.  A node is all a client names of the mesh: never a link, port or queue pair
   (design/rubrics.md, Three kinds of source). */
struct mesh_reach { uint32_t node,padding; uint64_t bandwidth; };
int mesh_reached(struct mesh_reach *out,uint32_t capacity,uint32_t *node);
/* The node's bridge's region: its arena's bytes (what its configuration registered at device up, the bridge's own pages
   included) and the bytes no lease holds now; 0, or -errno where the bridge is unreachable. */
int mesh_region(uint64_t *bytes,uint64_t *unleased);
/* The streams held at the node's bridge (0 between calls) and the wall clock in ns at which none last came to be held
   (0 while any is; mesh.h struct hdr `busy`, `idle_since`, design/transport.md; a read that meets the bridge as it
   becomes busy is told busy, 1); 0, or -errno where the bridge is unreachable. */
int mesh_busy(uint64_t *busy,uint64_t *idle_since);

/* A link map, an operand: `nodes` nodes and `links` links of a kind.
     MESH_LINKS_MESH   every pair is linked (`link` lists only the pairs given bandwidths, if any)
     MESH_LINKS_RING   link[i][1] follows link[i][0] around one cycle through every node
     MESH_LINKS_TREE   link[i][0] is link[i][1]'s parent; the root is nobody's child
     MESH_LINKS_GRAPH  the links as given, either way, any connected graph (a ring, a tree or a mesh's links
                       among them, unoriented)
   bandwidth (NULL: none given) is link i's bandwidth either way in bits a second, a hardware fact (mesh_map's: the
   least that either end's record reports of the cables joining the pair; 0 where none reports one), which the routes
   and the packed trees read: of routes of equally few links the one over the faster hops (mesh_routes), and a hop's
   capacity (mesh_trees_pack).  `relays` (a graph's, with a table) more nodes, numbered from `nodes`, that a route may
   cross and that hold no rank with work: a collective's working ranks are the map's first `nodes`, and its links may
   join any of its nodes and relays.  A route through a node is carried only by the node's rank of the communicator
   that runs the collective, so a communicator's map, whose nodes are its ranks, lists such a node among them and has
   no relays (nccl-mesh.c map_of); that rank sends on what lands alone (the identity) only where a program's
   mesh_compile left it out of `contributors`, which libnccl-mesh's own table never does.
   `mark` (NULL: each node's number) is each node's and relay's mark, distinct (mesh_map's: its bridge's node number,
   the same wherever it is numbered).
   `next` (NULL: none, and no relays) is the routing table, an operand (mesh_routes makes one): of N = nodes + relays,
   each node b's in-tree, next[v * N + b] v's parent in it, a node a link joins to v (b's own entry b; v itself where
   no route reaches b).  The route between a and b is the path up the in-tree of the one of the higher mark from the
   other, so the route from b to a is the one from a to b reversed (mesh_route_path).  With a table, two of the map's
   nodes are linked where the route between them crosses no other of them (relays alone), and that hop's bytes cross
   every link of it (mesh_trees_pack).  Without one, two nodes are linked where a link joins them.  Several links listed
   between one pair are parallel: the planner adds their individual bandwidths (mesh_routes,
   mesh_trees_pack); mesh_map lists a link a cable.  `link`, `bandwidth`, `next` and `mark` are the caller's
   storage. */
enum { MESH_LINKS_MESH, MESH_LINKS_RING, MESH_LINKS_TREE, MESH_LINKS_GRAPH };
struct mesh_link_map { uint32_t kind,nodes,links; uint32_t (*link)[2]; uint64_t *bandwidth; uint32_t relays; uint32_t *next,*mark; };
/* 0 where a map is one, else EINVAL: every link's nodes among its nodes and relays; a ring one cycle through every node
   with one link into each; a tree nodes-1 links, one into each node but the root, every node reaching the root; with a
   table (a graph's, where it has relays), its every entry a node or relay its row's node is linked to, or that node.  A
   graph need not join its nodes: the bridges' records are a map while a node is reached by no bridge, a node no link
   joins among its nodes (mesh_map's whole map), and a planner or driver reads it as it is. */
int mesh_link_map_check(const struct mesh_link_map *);
/* 0 where a map (one mesh_link_map_check takes) joins every two of its nodes, else ENOTCONN, apart[0] and apart[1]
   (apart NULL: none) the first two it does not join, the first the lower (EINVAL where it is no map): a graph's links
   connect them, and with a table every route between two of them reaches its end; a ring, a tree and a mesh join
   theirs by their form.  The maps a collective's trees and a communicator's collectives run on (mesh_trees_pack;
   nccl-mesh.c comm_routes fails a collective on a map that does not join its members, naming
   two). */
int mesh_link_map_joined(const struct mesh_link_map *,uint32_t *apart);
/* The routing table of a map's links (into next, N x N as struct mesh_link_map's; map->next is not read): each node
   b's in-tree of shortest routes toward it, by fewest links, then the least sum of their hops' inverse bandwidths
   (a pair's parallel links together: 8e15 over their summed bits a second, in whole fs a byte; a hop
   with no bandwidth counts none), each node's parent among its equals the one of the least hash of its mark, the parent's
   and the root's (equal-cost multipath by a hash, as ECMP spreads flows [RFC 2992]): routes of equal length spread
   over the links rather than piling onto the lowest-numbered node, every bridge and member that holds the same
   records makes the same table, and a route between two nodes is a function of the shortest routes between them
   alone, so it moves only where those change (a link or node one of them crosses lost or its bandwidth changed, a
   link joining or a bandwidth rising that makes another as short): a link lost elsewhere, a node joining by its own
   links or a bandwidth falling off them moves no route.  0, or EINVAL where a link names no node or relay, ENOMEM. */
int mesh_routes(const struct mesh_link_map *,uint32_t *next);
/* The route between the map's nodes or relays a and b on its table (mesh_routes' where it has none: the trees of its
   links), a and b included, into path (room nodes + relays): its node count, 0 where none (a == b, or it does not
   reach). */
uint32_t mesh_route_path(const struct mesh_link_map *,uint32_t a,uint32_t b,uint32_t *path);
/* The members' map of a whole map (one without relays: its nodes, marks, links, bandwidths and table, NULL: the table
   mesh_routes makes of its links): map node i the whole map's node members[i] (one it does not hold is linked to
   nothing), the whole map's other nodes its relays (numbered from n, ascending), each with its mark (the whole
   map's, written where map->mark is given), its links every link of the whole map renumbered (a mesh's every pair,
   at its listed bandwidth or none), and its table the whole map's, written where map->next is given: so two members
   are linked where their route crosses no other member, and that hop's bytes cross each link of its route
   (mesh_trees_pack).  n 0: the whole map itself.  whole_node[i] (NULL: none) the whole map's node map node i is
   (whole->nodes for a member it does not hold).  The kind MESH_LINKS_MESH where there are no relays and a link joins
   every pair (several a pair's parallel links), else MESH_LINKS_GRAPH.  On entry map->nodes and map->links are the
   room of whole_node, mark and next's rows (the map's nodes and relays together), and of link and bandwidth; returns the
   map's nodes, map->relays its relays, or -errno (ENOSPC: the room is short, map->nodes and map->links then the room
   needed; EINVAL: a member twice, relays and map->next or map->mark NULL, or a whole map with relays; ENOMEM).  The
   one rule mesh_map reads the bridges' records by. */
int mesh_map_members(const struct mesh_link_map *whole,const uint32_t *members,uint32_t n,uint32_t *whole_node,struct mesh_link_map *map);
/* The map of the node's bridge (design/transport.md) as of one map serial, from its records of the nodes every node's
   bridge reaches (each record's reach, an entry a cable: mesh-link.c record_reach), so the same records give every
   member the same map: the whole map, its nodes the records' ascending by bridge, two joined by a link a cable each
   one's bridge reaches the other over (the fewer of the two ends' counts: parallel links where several cables join
   them), each at the least bandwidth either end reports (0 where either reports none), each node marked by its
   bridge's number, its table mesh_routes'; then over `members` (bridge numbers; n 0: every node), mesh_map_members of
   it; node[i] (node NULL: none) map node i's bridge.  Room and returns as mesh_map_members'. */
int mesh_map(const uint32_t *members,uint32_t n,uint32_t *node,struct mesh_link_map *map);
/* The bridge's map serial (never 0; it changes with the records): waits until it differs from *serial and writes it
   there, within the deadline bound_ns (0 or UINT64_MAX: 5 s), past which the program ends (design/data-model.md
   section 6); 0 where it differs, else an unreachable bridge's errno.
   Read (*serial 0) before mesh_map, it makes a later change end the next wait at once. */
int mesh_map_wait(uint64_t *serial,uint64_t bound_ns);
/* Whether the map links its nodes a and b, either way (a mesh every pair; with a table, their route crossing relays
   alone). */
int mesh_link_between(const struct mesh_link_map *,uint32_t a,uint32_t b);

/* Spanning trees over a map's nodes: the routes every collective and every message take (Blink's tree packing
   [Wang et al., MLSys 2020]; ForestColl [Zhao et al. 2024]).  Tree t is rooted at root[t]; node v's parent in it is
   parent[t * nodes + v], a node the map links v to (the root's is the root); log_weight[t] is its unnormalised log
   share of what its root gathers and sends.  The trees of one root split that root's elements by exp(log_weight), in
   tree order, each the floor of its share and the first ones one element more for the remainder.  A reduction toward
   r runs up r's trees, a broadcast from r down them, a message to r along the path from its sender in r's heaviest
   tree (the lowest-numbered of equals).  The trees are an operand, made ahead of time (mesh_trees_pack, or the
   caller's own) and changed only by making them again; they span a map that joins its nodes (mesh_link_map_joined). */
struct mesh_trees { uint32_t nodes,count; const uint32_t *root,*parent; const double *log_weight; };
/* 0 where the trees fit the map: each spans every node over the map's links and reaches its root, and every node
   roots at least one (a collective reduces toward and broadcasts from every node); else EINVAL. */
int mesh_trees_check(const struct mesh_link_map *,const struct mesh_trees *);
/* Each root's spanning in-arborescences packed fractionally into the map's links, each way, the roots the nodes of
   `roots` (a bit a node, 64 a word; NULL: every node): Garg and Konemann's maximum
   concurrent flow [FOCS 1998], every root's trees together carrying one unit, each phase sending every root's unit
   along minimum-length arborescences [Chu & Liu 1965; Edmonds 1967] of the map's hops, a hop's length the sum of the
   lengths of the links it crosses, each one way: with a table every link a hop's route crosses, so the hops of a tree
   that cross one link charge it each (hops through relays share the links they cross); a link's length one way grows by
   (1 + eps x flow/capacity) and its capacity is the sum of its pair's parallel links' bandwidths
   (where the map gives some pair no bandwidth, its parallel links' count: every link alike).  A root's equal trees
   merged, its `most` heaviest kept and their flows' shares their log weights, so a part cut by them crosses every
   route and the routes end together.  Writes root, parent (nodes a tree) and log_weight
   for at most nodes x most trees, each root's heaviest first; returns the count, 0 where the map does not join its
   nodes (mesh_link_map_joined) or `roots` holds none.  At two nodes, each node's one tree. */
uint32_t mesh_trees_pack(const struct mesh_link_map *,const uint64_t *roots,double eps,uint32_t most,uint32_t *root,uint32_t *parent,
                         double *log_weight);

/* A typed operand: `type` is the caller's own element-type code, carried through untouched. */
struct mesh_operand { uint32_t type,element_bytes; uint64_t elements; };
/* One step of one rank's part of a collective, in that rank's order (a compiled program's moves made elements,
   mesh_program_steps_cut).
   SEND   publishes elements [first, first + piece.elements) of `buffer` to peer.
   REDUCE receives that typed piece from peer and combines it into `buffer` at `first`.
   COPY   receives that typed piece from peer into `buffer` at `first`.
   buffer 0 is the operand, the result made in place; b >= 1 the rank's scratch b, a copy of the operand made at the
   call's start. round is the step's position: both ends agree on logical order.
   Physical refinement may split a SEND at page boundaries without splitting its
   receive; the ordered wire-page sequence, rather than entry count, agrees. */
enum { MESH_STEP_SEND, MESH_STEP_REDUCE, MESH_STEP_COPY };
struct mesh_step { uint32_t op,peer,round,buffer; uint64_t first; struct mesh_operand piece; };

/* design/whole-function.md#fabric */
struct mesh_cable { uint64_t wire,bandwidth; uint32_t node[2]; };
struct mesh_fabric { uint32_t nodes,cables; const struct mesh_cable *cable; };
struct mesh_transfer { struct mesh_step step; uint32_t cable,padding; };
int mesh_fabric_check(const struct mesh_fabric *);
/* Logical-edge adapter to mesh_write_compile with all directly joining cables
   supplied as its path configuration. Copies/reductions retain their typed
   logical operation, buffer, round and offset. No independent allocation
   algorithm or first-cable transport implementation lives in this adapter. */
int mesh_fabric_stripe(const struct mesh_fabric *,uint32_t rank,const struct mesh_step *,uint64_t page,
                       const uint64_t *weights,struct mesh_transfer *,uint32_t room);
/* Refine physical SENDs where their latest preceding receive changes: a COPY's
   interval is an identity source; REDUCE or no receive is not. Only receives in
   the same buffer with the same element size participate. Adjacent pieces with
   the same latest COPY are kept whole, including where older writes overlap.
   Non-SEND transfers are unchanged; each SEND's pieces retain its order, cable,
   round, peer, type and exact coverage. Every inserted boundary is a multiple
   of page bytes from that SEND's start, so its wire page sequence is unchanged
   and its peer may retain the unsplit receive. Entry-indexed diagnostics must
   match page ranges rather than require one local entry per remote entry.
   Returns the required count (out NULL: count only), or -errno: EINVAL for a
   needed non-page cut or COPY-relative start, invalid operands/page, or aliased input/output; ENOSPC
   for insufficient room; EOVERFLOW for unrepresentable extents/count. out must
   be separate storage; a negative return leaves no usable output. No state,
   receive gate, payload copy or synchronization is introduced. */
int mesh_fabric_refine(const struct mesh_transfer *,uint32_t count,uint64_t page,
                       struct mesh_transfer *out,uint32_t room);
/* Exact linear allocation weights. Zero disables allocation to a path without
   deleting its capability; at least one weight is positive. The checked sum
   fits uint64_t. Allocation uses uint128_t products and integer remainders. */
struct mesh_paths { uint32_t count; const uint32_t *first,*cable; const uint64_t *weight; };
int mesh_paths_check(const struct mesh_fabric *,uint32_t source,uint32_t destination,const struct mesh_paths *);
/* tile is the publication width in elements (0: one publication). Route the
   complete operand once, then cut every path at these logical boundaries.
   Source, relay and destination retain matching independent tile fragments. */
int mesh_write_compile(const struct mesh_fabric *,uint32_t rank,uint32_t source,uint32_t destination,
                        const struct mesh_paths *,struct mesh_operand,uint64_t page,uint64_t tile,
                        struct mesh_transfer *,uint32_t room);

/* Physical lowering of a complete typed dataflow. first/steps contain every
   rank's logical steps in dependency order, including matching endpoint-only
   SEND/COPY declarations. Offsets and buffer numbers may differ at the two
   endpoints. paths is the nodes*nodes table used by mesh_write_compile.
   Logical reductions remain reductions only at their destination; intermediate
   ranks forward immutable fragments in separate, compact, uninitialized leases.
   initialized[r] is the number of the rank's logical buffers. The returned
   transfer lists preserve local dependency order; their round fields give the
   common AOT publication order on every directed cable. consume gives the
   local consumption event order, which may differ from physical arrival order. No runtime
   control exchange, payload copy, or remote acknowledgement is introduced.
   demand[s*nodes+d] counts useful logical bytes before physical relay expansion.
   The result owns its storage; release it with mesh_routed_free. Returns errno.
   This is the common lowering for standard, counted, and declared point calls. */
struct mesh_routed_rank {
  uint32_t count,buffers,initialized,padding;
  struct mesh_transfer *transfer;
  uint32_t *consume; /* per-step AOT consumption event order, distinct from wire order */
  uint32_t *retired_first,*retired; /* prior local SENDs whose source read precedes this COPY arrival */
  uint64_t *extent;
  unsigned char *witnessed; /* a SEND whose arrival precedes a later landing at its rank */
};
struct mesh_routed_program {
  uint32_t nodes,padding;
  struct mesh_routed_rank *rank;
  uint64_t *demand;
  uint64_t *route_units; /* largest logical transaction's independently allocatable units, per pair */
};
int mesh_routed_compile(const struct mesh_fabric *,const struct mesh_paths *,
                         const uint32_t *first,const struct mesh_step *,
                         const uint32_t *initialized,uint64_t page,
                         struct mesh_routed_program *);
void mesh_routed_free(struct mesh_routed_program *);

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
struct mesh_program { uint32_t what,root,nodes,parts,scratch,segments; const uint32_t *segment; const double *log_weight;
                      const uint32_t *first; const struct mesh_move *move; };

/* The room mesh_compile needs: parts and moves. */
void mesh_compile_room(const struct mesh_trees *,uint32_t *parts,uint32_t *moves);
/* Separate operand contributors and result recipients; excluded physical ranks
   remain available as identity relays. Gather/ReduceScatter segments retain
   full rank numbering, including trailing zero-size owners. NULL masks mean
   every rank. A rooted Reduce's root may lie outside its contributor set. */
int mesh_compile_members(const struct mesh_trees *,uint32_t what,uint32_t root,int whole,
                           const uint64_t *contributors,const uint64_t *recipients,
                           struct mesh_program *,uint32_t *segment,double *log_weight,
                           uint32_t *first,struct mesh_move *move);
/* Validate a fixed program's initialized ranges and required outputs against
   the operand roles at configuration. No topology or transfer is replaced. */
int mesh_program_check_members(const struct mesh_program *,struct mesh_operand,const uint64_t *cut,uint64_t page,
                                const uint64_t *contributors,const uint64_t *recipients);
/* A collective compiled along `trees` into `program` (arrays of mesh_compile_room's room: segment and log_weight a
   part, first nodes + 1, move a move):
     MESH_REDUCE           the whole operand up root's trees, a part a tree (its weight);
     MESH_BROADCAST        the whole operand down root's trees;
     MESH_REDUCE_SCATTER   segment s up s's trees;  MESH_ALLGATHER  segment s down s's trees;
     MESH_ALLREDUCE        whole 0: the reduce-scatter and then the all-gather (each link a share once each way; its
                           dependent crossings twice the trees' heights) [Patarasuk & Yuan 2009; Blink];
                           whole 1: the whole operand reduced toward every contributor at once, up its heaviest tree,
                           a node forwarding each partial from distinct scratch. Its original contribution has a
                           separate immutable source binding whenever its result can change concurrently; publishing
                           a SEND does not prove its DMA source was read. Its dependent crossings are the heaviest
                           trees' height, 1 where they are stars; each node
                           sends the whole operand once for every other) [single-phase all-reduce: Trivance 2026].
   A tree's part is one move a hop, uncut: a node that sends on unchanged what it received (a COPY, the identity at a
   rank, design/data-model.md section 9) sends each page as it lands (libnccl-mesh presents the SEND gated page by page
   on that receive's landed count), so such a path pipelines by the page and no count of chunks exists; a node that
   combines sends its sum once its part is combined.  Going up a
   tree a node sends at round its height, after its children; coming down, a node at depth d receives at base + d - 1
   and sends at base + d, base the highest of the trees' heights (the reduce-scatter's last round plus one); a round's
   SENDs come before its receives, so a SEND waits on lower rounds alone.  A
   message is coalesced with the next between the same two nodes in the same round where their parts adjoin and each
   end's buffer and step are the same.  `contributors` (a bit a node, 64 a word; NULL: every node) are the nodes with
   work: those whose operands a reduction combines, whose segments a reduce-scatter, an all-reduce and an all-gather
   cut (in node order: the operand is theirs, the others holding none), and toward which a result goes.  Every other
   node runs the identity: up a tree it takes its first child's partial as it comes and combines the rest into it, and
   sends it on; down a tree it is sent a piece only where a contributor lies below it, and sends it on.  0, or
   EINVAL. */
int mesh_compile(const struct mesh_trees *,uint32_t what,uint32_t root,int whole,const uint64_t *contributors,
                 struct mesh_program *program,uint32_t *segment,double *log_weight,uint32_t *first,struct mesh_move *move);
/* A rank's steps of a program for an operand, a step a move (none where its elements are none), at most first[rank +
   1] - first[rank], over an operand cut into segments of the given sizes (MPI_Allgatherv's and MPI_Reduce_scatter's counts, one
   a segment in segment order: a reduce-scatter's and an all-gather's segment s is node s's, node s's of its
   contributors where it has them): segment s the segments[s] elements after the segments before it, the sizes as many
   as the program's segments (the most segment a part names, plus one) and summing to the operand's elements, a size 0
   a segment whose steps carry nothing (none made); NULL the program's own cut.  Every cut the program makes itself (its
   segments where none are given, and its parts within a segment by their weights) lies on a multiple of `page` bytes
   from the operand's start, the least multiple of both the page and the element (0 or 1: any element), so that a
   step's piece starts on a page wherever its buffer does (design/transport.md section 14: an entry starts on a page);
   a cut that the sizes fix lies where they put it.  Within a segment each part takes the floor of its share of the
   segment's quanta, the first ones one quantum more for the remainder, the last part of the segment ending at its end.
   0 where the sizes do not sum to the operand (mesh 430cf7b2^ mesh_cut, 3ab709aa: the planner's `segments`, cut alike
   by every program). */
uint32_t mesh_program_steps_cut(const struct mesh_program *,uint32_t rank,struct mesh_operand,const uint64_t *segments,uint64_t page,
                                struct mesh_step *steps);
/* A step's entry on its channel (design/transport.md section 14: a holding's send list and landing list, each in its
   own order): its index among the rank's steps, whether it is a SEND (the send list) or a receive (the landing list),
   and its entry's index in that list.  A step is one entry, whatever its bytes: the bridge cuts it into SENDs. */
struct mesh_place { uint32_t step,send; uint64_t entry; };
/* The entries a rank's steps of one collective take on its channel toward `peer`, its SENDs to it on the send list
   (`send` 1) or its receives from it on the landing list (`send` 0), from entry `at`: by round, a round's steps in their
   order, a step of no bytes none; the end, and each step placed in made[*placed ..], *placed counting them.  The one
   order both ends make of their ranks' logical steps; physical SEND refinement
   preserves that order and the peer's exact wire-page sequence (libnccl-mesh schedule_group). */
uint64_t mesh_lay(const struct mesh_step *steps,uint32_t count,uint32_t peer,int send,uint64_t at,struct mesh_place *made,uint32_t *placed);
/* A declared group preserves call order for every operation. Each rank's
   physical steps have the compiler's dependency-safe ordinal. mesh_group_lay
   assigns entries on the directed peer list from `at`, retaining every stripe;
   scratch holds the largest local call and made holds the complete local group.
   Runtime dependencies are overlapping prior receives, not an operation-kind
   ordering rule or a whole-operand join. */
struct mesh_group_call { const uint32_t *count; const struct mesh_step *const *steps; };
struct mesh_group { uint32_t nodes,calls; const struct mesh_group_call *call; uint64_t page; };
struct mesh_group_place { uint32_t call,padding; struct mesh_place place; };
uint64_t mesh_group_lay(const struct mesh_group *,uint32_t rank,uint32_t peer,int send,uint64_t at,struct mesh_place *scratch,
                        struct mesh_group_place *made,uint32_t *placed);
/* A counted all-to-all's steps at `rank` of `nodes`, all of round 0 (libnccl-mesh ncclMeshAlltoAllV, nccl-mesh.c
   counted_steps; both ends hold the counts): segments[q] the entries toward rank q, segments[nodes + q] those from it;
   to each other rank q one SEND of its entries of `element` (buffer 0 at q's segment) and from each one COPY (buffer 1
   at q's).  Their count, at most 2 (nodes - 1). */
uint32_t mesh_counted_steps(uint32_t nodes,uint32_t rank,const uint64_t *segments,struct mesh_operand element,struct mesh_step *steps);

/* design/transport.md: what a node's record carries of its health to every bridge: `booted`, the wall
   clock of the node's boot (ns since the epoch), and its `available` memory as its bridge last sampled it (the bytes
   the kernel can give without pressure, the measure jetsam keeps: kern.memorystatus_level percent of hw.memsize). */
struct mesh_health { uint64_t booted,available; };
/* A node's record as its bridge last sent it: its bridge's node, its bridge's region (bytes its clients may hold), the node's
   memory (hw.memsize) and model (hw.model, NUL-terminated, MESH_MODEL_BYTES with its NUL; mesh.h struct mesh_record),
   and its health. */
struct mesh_node_health { uint32_t node; uint64_t region,memory; char model[MESH_MODEL_BYTES];
                          struct mesh_health health; };
/* Every node's record the bridge holds (mesh_map's nodes, this node's own among them): their count (at most `capacity`
   written), or -errno where the bridge is unreachable. */
int mesh_nodes_health(struct mesh_node_health *out,uint32_t capacity);

#ifdef __cplusplus
}
#endif
#endif
