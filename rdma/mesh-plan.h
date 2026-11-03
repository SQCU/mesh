#ifndef MESH_PLAN_H
#define MESH_PLAN_H
#include <stddef.h>
#include <stdint.h>
#include "../etc/bridge/layout.h"
#include "mesh-fabric.h"
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
   no relays (nccl-mesh.c map_of); that rank's own value counts in every sum (mesh_page_moves: every node of the trees
   contributes).
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
   share of what its root gathers and sends.  A reduction toward r runs up r's trees, a broadcast from r down them
   (mesh_page_moves).  The trees are an operand, made ahead of time (mesh_trees_pack, or the caller's own) and changed
   only by making them again; they span a map that joins its nodes (mesh_link_map_joined).  latency[t] (NULL: none) is
   tree t's time from its root's release to its deepest leaf's first landing (its hops at a hop's cost), and `rate`
   the pages a time every root's segment crosses its trees at together (mesh-graph.h mesh_graph_trees_rate over
   capacities in pages a time; 0: none, and then no latency either).  A root's pages are cut over its trees by
   earliest finish (mesh-graph.h mesh_graph_fill: a page goes where it lands soonest), tree t taking pages one each
   1 / (its share exp(log_weight[t]) of its root's trees' x rate) after latency[t]; without a rate, by the shares
   alone. */
struct mesh_trees { uint32_t nodes,count; const uint32_t *root,*parent; const double *log_weight; const double *latency; double rate; };
/* 0 where the trees fit the map: each spans every node over the map's links and reaches its root, its weight finite,
   and every node of `roots` (a bit a node, 64 a word; NULL: every node) roots at least one: the nodes a collective's
   pages go up toward or down from (mesh_page_moves: a reduce's or a broadcast's root, each node of a reduce-scatter's
   or an all-gather's nonzero segments, an all-reduce's every node).  So a rooted collective takes trees packed for its
   root alone (mesh_trees_pack's `roots`), every link of the map in them carrying its pages, not its root's few of an
   every-root packing; else EINVAL. */
int mesh_trees_check(const struct mesh_link_map *,const struct mesh_trees *,const uint64_t *roots);
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
/* design/measurement-regressions-2026-10-30.md, b7311501: a plan's traffic as demands, each a root, its leaves (NULL
   every node) and the bytes that go up toward the root and down from it; mesh_solve routes them all at once over the
   map's links, Garg and Koenemann's multiplicative weights (FOCS 1998; Fleischer 2000): T = floor(ln 2E / eps^2) + 1
   rounds, in each every demand taking its least-length tree under lengths exp(eps (x_e - x_max) / rho) / c_e (x_e a
   direction's load over its capacity, the parallel cables of a pair one link of their bandwidths summed), the loads
   added after each round. Its output: each distinct tree (`demand` its demand, `parent` n a tree) and its share of its
   demand (`share`, its rounds over T); their count (at most mesh_solve_room), or UINT32_MAX where a demand's leaf has
   no route to its root. mesh_flow_trees gives the trees of demands [first, last) as a struct mesh_trees' arrays. */
struct mesh_demand { uint32_t root,pad; uint64_t up,down; const uint64_t *leaves; };
uint32_t mesh_solve_room(const struct mesh_link_map *,double eps,uint32_t count);
uint32_t mesh_solve(const struct mesh_link_map *,double eps,const struct mesh_demand *,uint32_t count,uint32_t room,
                    uint32_t *demand,uint32_t *parent,double *share);
uint32_t mesh_flow_trees(uint32_t nodes,const uint32_t *demand,const struct mesh_demand *,const uint32_t *parent,const double *share,
                         uint32_t count,uint32_t first,uint32_t last,uint32_t *root,uint32_t *tree,double *log_weight);
uint32_t mesh_trees_pack(const struct mesh_link_map *,const uint64_t *roots,double eps,uint32_t most,int shallow,uint32_t *root,
                         uint32_t *parent,double *log_weight);

/* A typed operand: `type` is the caller's own element-type code, carried through untouched. */
struct mesh_operand { uint32_t type,element_bytes; uint64_t elements; };
/* One step of one rank's part of a routed WRITE program, in that rank's order (mesh_fabric_refine, mesh_routed_compile).
   SEND   publishes elements [first, first + piece.elements) of `buffer` to peer.
   REDUCE receives that typed piece from peer and combines it into `buffer` at `first`.
   COPY   receives that typed piece from peer into `buffer` at `first`.
   buffer 0 is the operand, the result made in place; b >= 1 the rank's scratch b, a copy of the operand made at the
   call's start. round is the step's position: both ends agree on logical order.
   Physical refinement may split a SEND at page boundaries without splitting its
   receive; the ordered wire-page sequence, rather than entry count, agrees. */
enum { MESH_STEP_SEND, MESH_STEP_REDUCE, MESH_STEP_COPY };
struct mesh_step { uint32_t op,peer,round,buffer; uint64_t first; struct mesh_operand piece; };
/* a step's piece's bytes */
static inline uint64_t mesh_step_bytes(const struct mesh_step *step){ return step->piece.elements*step->piece.element_bytes; }
/* The elements a page-aligned cut lies on: the least multiple of `page` bytes and of an element's bytes, in elements
   (1 where a page is 0 or 1 byte, any element; an element of 0 bytes counted as 1) */
static inline uint64_t mesh_page_quantum(uint64_t page,uint64_t element){ if(page<=1)return 1; uint64_t a=page,b=element?element:1; while(b){const uint64_t t=a%b;a=b;b=t;} return page/a; }

/* design/whole-function.md#fabric */
struct mesh_transfer { struct mesh_step step; uint32_t cable,padding; };
int mesh_fabric_check(const struct mesh_fabric *);
/* Refine every physical operation at the boundaries of its local producers
   and consumers. COPY, REDUCE and SEND share exact value cells, including cuts
   within a page. The routed compiler unions both endpoints' cuts and propagates
   them through relays before constructing packets or dependencies; no peer may
   retain a different partition. Pieces preserve order, cable, peer, type and
   exact coverage. Returns required count (out NULL), or -errno for malformed
   operands, insufficient output storage, or unrepresentable extents. Input and
   output storage are distinct. This is preparation algebra: no payload copy,
   receive gate, runtime work selector or serialization is added. */
int mesh_fabric_refine(const struct mesh_transfer *,uint32_t count,uint64_t page,
                       struct mesh_transfer *out,uint32_t room);
/* One preparation scheduler for physical WRITE and collective fragments.
   first has groups*nodes+1 offsets into transfer, grouped by call then rank in
   declared local dependency order. Production operands are per transfer (its
   producer's order and release); NULL means every source ready at zero. page is
   the unit the schedule moves (mesh-graph.h mesh_graph_schedule). publish/consume
   return paired wire and receive/combine event orders; *finish (NULL: not wanted)
   the schedule's last unit's end, in seconds from its start: every page moved as
   soon as it is ready, at its cables' rates, nothing else waited for. */
struct mesh_transfer_production { uint64_t order; double seconds; };
/* Actual placed-page overwrite relation. A COPY writes its landing; a REDUCE
   writes its numerical result after receiving into separate pages. `preserve`
   permits preparation to separate the original source generation if retaining
   its alias would close a cycle. separate[k] reports that allocation decision. */
struct mesh_transfer_alias { uint32_t read,write,preserve; };
int mesh_transfer_order(const struct mesh_fabric *,uint32_t groups,const uint32_t *first,
                        const struct mesh_transfer *,const struct mesh_transfer_production *,
                        const struct mesh_transfer_alias *,size_t aliases,unsigned char *separate,uint64_t page,
                        uint32_t *publish,uint32_t *consume,double *finish);
/* struct mesh_paths (mesh-fabric.h): allocation uses uint128_t products and integer remainders. */
int mesh_paths_check(const struct mesh_fabric *,uint32_t source,uint32_t destination,const struct mesh_paths *);
/* tile is the publication width in elements (0: one publication). Route the
   complete operand once, then cut every path at these logical boundaries.
   Source, relay and destination retain matching independent tile fragments. */
int mesh_write_compile(const struct mesh_fabric *,uint32_t rank,uint32_t source,uint32_t destination,
                        const struct mesh_paths *,struct mesh_operand,uint64_t page,uint64_t tile,
                        struct mesh_transfer *,uint32_t room);
/* The exact allocation used by mesh_write_compile, in operand elements. */
int mesh_write_ranges(const struct mesh_paths *,struct mesh_operand,uint64_t page,uint64_t *first,uint64_t *count);
/* A fabric's write routes (mesh-fabric.c), the communicator's defaults (nccl-mesh.c fabric_discover) and a tool's alike:
   mesh_write_routes_default every simple node path of every ordered pair, parallel cables distinct choices, each of
   weight 1, into *out (NULL on entry), a nodes x nodes table; mesh_write_routes_compose those routes with every pair a
   cable joins realized on its cables (by bandwidth), but where `overrides` marks it; each 0 or an errno, *out unchanged on
   a failure; mesh_write_routes_free a table either made. */
int mesh_write_routes_default(const struct mesh_fabric *,struct mesh_paths **out);
int mesh_write_routes_compose(const struct mesh_fabric *,const struct mesh_paths *routes,const unsigned char *overrides,
                              struct mesh_paths **out);
void mesh_write_routes_free(struct mesh_paths *routes,uint32_t nodes);

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
/* Packet coordinates are explicit compiler operands: a collective's one
   logical operand, or independent transactions with distinct endpoint origins.
   Every physical view uses the same byte-domain representation. */
enum mesh_packet_coordinates { MESH_PACKET_OPERAND, MESH_PACKET_TRANSACTIONS };
/* first names the shared packet-cut domain; offset names bytes within the
   original logical SEND. transaction survives every routed hop and refinement.
   Distinct partial-result generations can share a packet domain. */
struct mesh_packet_view { uint64_t first, offset; uint32_t domain, transaction; };
/* One original logical SEND, before physical paths or backing cuts exist.
   send/receive are the endpoints' local logical step ordinals. Byte origins
   belong to those logical operands, independently of relay allocations. */
struct mesh_transaction {
  uint32_t source,destination,send,receive;
  uint64_t source_first,destination_first;
  struct mesh_operand piece;
};
struct mesh_routed_rank {
  uint32_t count,buffers,initialized,padding;
  struct mesh_transfer *transfer;
  uint32_t *consume; /* per-step AOT consumption event order, distinct from wire order */
  uint64_t *extent;
  struct mesh_packet_view *views;
};
struct mesh_routed_program {
  uint32_t nodes,domains,transactions;
  struct mesh_routed_rank *rank;
  struct mesh_transaction *transaction;
  uint64_t *demand;
  uint64_t *route_units; /* largest logical transaction's independently allocatable units, per pair */
};
int mesh_routed_compile(const struct mesh_fabric *,const struct mesh_paths *,
                         const uint32_t *first,const struct mesh_step *,
                         const uint32_t *initialized,uint64_t page,enum mesh_packet_coordinates,
                         struct mesh_routed_program *);
void mesh_routed_free(struct mesh_routed_program *);

/* What a collective computes (mesh_page_moves) */
enum { MESH_ALLREDUCE, MESH_BROADCAST, MESH_REDUCE, MESH_REDUCE_SCATTER, MESH_ALLGATHER };

/* The collective's joint interface (design/drift-2026-11-03.md, the one solver): the planner states page moves, the
   run's solve orders, places and guards them, the bridge executes channel entries.  Pages only: no byte, offset or
   address here.  A page move, the planner's output per route edge: rank `source` SENDs the collective operand's pages
   [first, first + pages) (page indices of the operand) to rank `destination`, which lands them.  Every landing is a
   version of its own of those operand pages (renaming), placed by the solve: a TRANSFER's or a FORWARD's in the
   destination's result pages where they are not its input's, else in fresh pages, a REDUCE's in fresh pages.  So no
   landing writes a page a SEND reads, and a landing's one edge is post -> landed, never a SEND's completion: the
   all-reduce's write-after-read hold at a lane's head (its down landing into the pages its up SEND read, every later
   landing of the channel behind it: metal-microbench facts 53a0e600, bdceb04f) cannot be stated, as c5e62f6e's
   all-gather in pages of its own removed it.  `kind`:
   - MESH_MOVE_TRANSFER: the source's value of those pages (its own, or the sum its reduce landings made final there);
     `link` UINT32_MAX;
   - MESH_MOVE_FORWARD: the version a move landing on this move's source landed: `link`, that move (its pages hold
     these);
   - MESH_MOVE_REDUCE: folded at the destination into its sum of those operand pages, one fold a grain over all its
     inputs (the destination's own value and every reduce landing of the grain there; never a fold a landing, k passes
     over the sum where one does); `link`, the move that carries the sum on from the destination (its pages hold these;
     UINT32_MAX where the sum is final there).  A reduce move's source sends its sum where a reduce move links to it,
     else its own value (its version before the collective, whatever has landed there since).
   The planner decides nothing more: every move's order, region pages, grains and guards are the solve's, and so is its
   channel, which of the pair's parallel cables carries each page: page by page (struct mesh_channel_entry `channel`),
   each cable a share of the pair's pages by its rate, so a FORWARD's pages stay within the one landing its link names
   however the cables cut them. */
enum { MESH_MOVE_TRANSFER, MESH_MOVE_FORWARD, MESH_MOVE_REDUCE };
struct mesh_page_move { uint32_t source,destination,kind,link; uint64_t first,pages; };
/* The trees' routes of collective `what` (MESH_*) as page moves, every node's value in every sum.  A tree's pages cross
   each of its edges as one move each way they go: up, a REDUCE from a node to its parent, linked to its parent's
   REDUCE (none at the root); down, a TRANSFER from the root to its children and a FORWARD from every other node to its
   children, linked to the move landing on it.
   - A reduce, a broadcast, a reduce-scatter and an all-gather: the operand's pages cut into segments in node order,
     node r's the segment[r] pages after the segments before it (a reduce's or a broadcast's root holds them all, every
     other node none); root r's segment cut over r's trees by earliest finish (struct mesh_trees), each tree's pages up
     it (a reduce, a reduce-scatter) or down it (a broadcast, an all-gather).
   - An all-reduce: its pages, the segments' sum (no node owns them), each page one of two forms over the same trees:
     up and down a tree (summed at its root and sent back down it: the page crosses each link of it once each way, its
     hops twice the tree's height), or up a tree of every root (summed at every node at once: no down half, one hop on
     a star, a REDUCE from every node to every other).  One earliest finish over both, on the trees' capacity, cuts
     them: the up-and-down pages over every tree, a tree's first at twice its latency (mesh-graph.h mesh_graph_fill);
     the others over each root's trees after them, a tree's first once its up-and-down pages have crossed it, plus its
     latency (on the shared up path the pages of the longer tail first: Jackson's rule [Jackson 1955]); their count the
     one whose later end is soonest, the fewer of equals.  So an operand of a hop's pages or fewer is summed at every
     node in one hop, and a long one up and down the trees with a hop's pages of the other at its tail: one cost, no
     case.
   The pages' cut over the trees and the forms is the planner's routing, and nothing else is.  Writes move[0 ..] (move
   NULL: counted only) and returns their count, at most 3 (nodes - 1) x the trees' count for an all-reduce and
   (nodes - 1) x the trees' count for the others (0 at one node or no pages; an all-reduce's up-and-down moves first,
   then those summed at every node); UINT32_MAX where `what` is none or a node the collective's pages go up toward or
   down from roots no tree of finite weight (mesh_trees_check's `roots`: its segment would cross no link).  The moves'
   order in `move` is none: every order, the relays' and the forms' among them, is the solve's (its dependences: a
   FORWARD's and a REDUCE's link, and the operand pages' versions). */
uint32_t mesh_page_moves(const struct mesh_trees *,uint32_t what,const uint64_t *segment,struct mesh_page_move *move);
/* A channel entry, the solve's output and the bridge's table row, one page a unit: on this rank's channel `channel`
   (one of the cables that end at this rank, a stream toward its peer over it, with its `ready`, `sent` and `landed`
   unit counts: mesh.h struct mesh_stream; which of a pair's cables a page takes is the solve's, page by page), `op`
   MESH_CHANNEL_SEND, a SEND from region page `page` (the page table's index, the solve's placement), or
   MESH_CHANNEL_LANDING, a receive posted into it.  A rank's table is its entries in the solve's one order; a
   channel's entries in it are that channel's unit order, so its k-th SEND is unit k of its `ready` and `sent` and its
   k-th landing unit k of its `landed`.  The guard, the solve's (a consumer's landed-prefix threshold; SEND ->
   completion where placement reuses pages an earlier SEND read; produced pages; a landing's none but that reuse, its
   pages a version of its own: struct mesh_page_move): the entry is posted once count `word` (MESH_WORD_*) of channel
   `guard` reaches `threshold` (0: at once).  The bridge turns `page` into an address through the page table and
   decides nothing. */
enum { MESH_CHANNEL_SEND, MESH_CHANNEL_LANDING };
enum { MESH_WORD_READY, MESH_WORD_SENT, MESH_WORD_LANDED };
struct mesh_channel_entry { uint32_t op,channel; uint64_t page; uint32_t guard,word; uint64_t threshold; };

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
