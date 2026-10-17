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
/* The node's bridge's window as its region holds it: its bytes (the arena its configuration registered at device up) and
   the bytes no lease holds now; 0, or -errno where the bridge is unreachable. */
int mesh_window(uint64_t *bytes,uint64_t *unleased);
/* The flows and awaiting programs the node's bridge holds (0 between calls) and the wall clock in ns at which it last
   came to hold none (0 while it holds any; mesh.h struct hdr `busy`, `idle_since`, design/bridge-and-ipc.md#region; a
   read that meets the bridge as it becomes busy is told busy, 1); 0, or -errno where the bridge is unreachable. */
int mesh_busy(uint64_t *busy,uint64_t *idle_since);

/* A link map, an operand: `nodes` nodes and `links` links of a kind.
     MESH_LINKS_MESH   every pair is linked (`link` lists only the pairs given costs, if any)
     MESH_LINKS_RING   link[i][1] follows link[i][0] around one cycle through every node
     MESH_LINKS_TREE   link[i][0] is link[i][1]'s parent; the root is nobody's child
     MESH_LINKS_GRAPH  the links as given, either way, any connected graph (a ring, a tree or a mesh's links
                       among them, unoriented)
   cost (NULL: the caller's alpha and beta for every link) is link i's alpha (us a message) and beta (ns a byte)
   either way [Hockney 1994]; a pair the map links without a cost takes the caller's.
   `relays` (a graph's, with a table) more nodes, numbered from `nodes`, forward and hold no rank: a collective's ranks
   are the map's first `nodes`, and its links may join any of its nodes and relays.  `mark` (NULL: each node's number)
   is each node's and relay's mark, distinct (mesh_map's: its bridge's node number, the same wherever it is numbered).
   `next` (NULL: none, and no relays) is the routing table, an operand (mesh_routes makes one): of N = nodes + relays,
   each node b's in-tree, next[v * N + b] v's parent in it, a node a link joins to v (b's own entry b; v itself where
   no route reaches b).  The route between a and b is the path up the in-tree of the one of the higher mark from the
   other, so the route from b to a is the one from a to b reversed (mesh_route_path), as the bridges forward it.  With
   a table, two of the map's nodes are linked where the route between them crosses no other of them (relays alone):
   that hop costs its links' alphas summed and the greatest of their betas (a relay forwards each slot as it lands,
   mesh-flow.c receive_input), and its bytes cross every link of it (mesh_trees_pack, mesh_program_time).  Without one,
   two nodes are linked where a link joins them.  Several links between one pair are its cables, as the bridges carry
   them (design/bridge-and-ipc.md#cables): a program between the two is striped over them, position t on cable t mod n,
   so the hop runs at n times the slowest one's bandwidth (its beta the greatest of theirs over n, its alpha the
   least), and a routed program is striped end to end (its stripes the least common multiple of its segments' cable
   counts, past a node's ports the fewest; stripe j on cable (h + j) mod c of each segment), each segment carrying it
   over min(stripes, c) of its cables.  `link`, `cost`, `next` and `mark` are the caller's storage. */
enum { MESH_LINKS_MESH, MESH_LINKS_RING, MESH_LINKS_TREE, MESH_LINKS_GRAPH };
struct mesh_link_map { uint32_t kind,nodes,links; uint32_t (*link)[2]; double (*cost)[2]; uint32_t relays; uint32_t *next,*mark; };
/* 0 where a map is one, else EINVAL: every link's nodes among its nodes and relays; a ring one cycle through every node
   with one link into each; a tree nodes-1 links, one into each node but the root, every node reaching the root; with a
   table (a graph's, where it has relays), its every entry a node or relay its row's node is linked to, or that node.  A
   graph need not join its nodes: the bridges' records are a map while a cable is out, a node no link joins among its
   nodes (mesh_map's whole map), and a planner or driver reads it as it is. */
int mesh_link_map_check(const struct mesh_link_map *);
/* 0 where a map (one mesh_link_map_check takes) joins every two of its nodes, else ENOTCONN, apart[0] and apart[1]
   (apart NULL: none) the first two it does not join, the first the lower (EINVAL where it is no map): a graph's links
   connect them, and with a table every route between two of them reaches its end; a ring, a tree and a mesh join
   theirs by their form.  The maps a collective's trees and a communicator's session run on (mesh_trees_pack,
   mesh_trees_shortest; nccl-mesh.c comm_read fails a session on a map that does not join its members, naming two). */
int mesh_link_map_joined(const struct mesh_link_map *,uint32_t *apart);
/* The routing table of a map's links (into next, N x N as struct mesh_link_map's; map->next is not read): each node
   b's in-tree of shortest routes toward it, by fewest links, then the least sum of their betas (each pair's cables
   striped; in whole fs a byte; a link without a cost counts none), each node's parent among its equals the one of
   the least hash of its mark, the parent's and the root's (equal-cost multipath by a hash, as ECMP spreads flows
   [RFC 2992]): routes of equal length spread over the links rather than piling onto the lowest-numbered node, every
   bridge and member that holds the same records makes the same table, and a route between two nodes is a function of
   the shortest routes between them alone, so it moves only where those change (a link or node one of them crosses
   lost or its cost changed, a link joining or a cost falling that makes another as short): a link lost elsewhere, a
   node joining by its own cables or a cost rising off them moves no route.  0, or EINVAL where a link names no node
   or relay, ENOMEM. */
int mesh_routes(const struct mesh_link_map *,uint32_t *next);
/* The route between the map's nodes or relays a and b on its table (mesh_routes' where it has none: the trees of its
   links), a and b included, into path (room nodes + relays): its node count, 0 where none (a == b, or it does not
   reach). */
uint32_t mesh_route_path(const struct mesh_link_map *,uint32_t a,uint32_t b,uint32_t *path);
/* A link's cost where nothing measured it: alpha the TB5 crossing's one-way latency through the bridges (us a message;
   metal-microbench docs/measurement.md, the 3 KB host round trip 10.29 us), beta where its ports' bandwidth is unknown
   (ns a byte). */
#define MESH_ALPHA_US 5.0
#define MESH_BETA_NS 0.1
/* The members' map of a whole map (one without relays: its nodes, marks, links, costs and table, NULL: the table
   mesh_routes makes of its links): map node i the whole map's node members[i] (one it does not hold is linked to
   nothing), the whole map's other nodes its relays (numbered from n, ascending), each with its mark (the whole
   map's, written where map->mark is given), its links every link of the whole map renumbered (a mesh's every pair,
   at its listed cost or none), and its table the whole map's, written where map->next is given: so two members are
   linked where their route crosses no other member, at that route's cost, and its bytes are charged to each link it
   crosses (mesh_link_cost, mesh_trees_pack, mesh_program_time).  n 0: the whole map itself.  whole_node[i] (NULL:
   none) the whole map's node map node i is (whole->nodes for a member it does not hold).  The kind MESH_LINKS_MESH
   where there are no relays and a link joins every pair (several a pair's cables), else MESH_LINKS_GRAPH.  On entry
   map->nodes and map->links are the room of whole_node, mark and next's rows (the map's nodes and relays together),
   and of link and cost; returns the map's nodes, map->relays its relays, or -errno (ENOSPC: the room is short,
   map->nodes and map->links then the room needed; EINVAL: a member twice, relays and map->next or map->mark NULL, or
   a whole map with relays; ENOMEM).  The one rule mesh_map reads the bridges' records by and a planner prices a
   record's members by. */
int mesh_map_members(const struct mesh_link_map *whole,const uint32_t *members,uint32_t n,uint32_t *whole_node,struct mesh_link_map *map);
/* The map of the node's bridge (design/bridge-and-ipc.md#ports) as of one map serial, from its records of every
   node's ports (mesh_observe's views), so the same records give every member the same map: the whole map, its nodes
   the records' ascending by bridge, two joined by a link a cable where each one's bridge holds ports paired with the
   other (the fewer of the two ends' counts), each at MESH_ALPHA_US and 8e9 over the bandwidth of the slowest of
   those ports at either end (MESH_BETA_NS for a port whose bandwidth is unknown; the bridge stripes a program
   between the two over every cable, a routed one end to end, design/bridge-and-ipc.md#cables), each node marked by
   its bridge's number, its table mesh_routes' (the routes the bridges forward along, mesh-dataflow.c mesh_route);
   then over `members` (bridge numbers; n 0: every node), mesh_map_members of it; node[i] (node NULL: none) map node
   i's bridge.  Room and returns as mesh_map_members'.  name[i] (name NULL: none) map node i's host name as its
   bridge's record holds it, NUL-terminated ("" for a node with no record).  MESH_NAME_BYTES: a LocalHostName of up
   to 63 bytes and its NUL, the record's name field (mesh.h struct mesh_record). */
#define MESH_NAME_BYTES 64
int mesh_map(const uint32_t *members,uint32_t n,uint32_t *node,char (*name)[MESH_NAME_BYTES],struct mesh_link_map *map);
/* The bridge's map serial (never 0; it changes with the records): waits until it differs from *serial or bound_ns
   passes and writes it there; 0 where it differs, ETIMEDOUT where the bound passed, else an unreachable bridge's errno.
   Read (*serial 0) before mesh_map, it makes a later change end the next wait at once. */
int mesh_map_wait(uint64_t *serial,uint64_t bound_ns);
/* Whether the map links its nodes a and b, either way (a mesh every pair; with a table, their route crossing relays
   alone). */
int mesh_link_between(const struct mesh_link_map *,uint32_t a,uint32_t b);
/* The hop a-b's cost: its links' (either way; each its own where the map gives one, else alpha and beta) striped, the
   least alpha and the greatest beta over their count (above); with a table and a route through relays, striped end to
   end, each segment of c cables the greatest of their betas over min(stripes, c), the segments' alphas summed and the
   greatest of their betas (a relay forwards each slot as it lands). */
void mesh_link_cost(const struct mesh_link_map *,uint32_t a,uint32_t b,double alpha,double beta,double *link_alpha,double *link_beta);

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
/* A message's path from `from` to `to` in to's heaviest tree, the nodes from `from` to `to` inclusive (room for
   `nodes`); their count, 0 where `to` roots no tree. */
uint32_t mesh_trees_path(const struct mesh_trees *,uint32_t from,uint32_t to,uint32_t *path);
/* Each node's spanning in-arborescences packed fractionally into the map's ports: Garg and Konemann's maximum
   concurrent flow [FOCS 1998], every root's trees together carrying one unit, each phase sending every root's unit
   along minimum-length arborescences [Chu & Liu 1965; Edmonds 1967] of the map's hops, a hop's length the sum of its
   ports' lengths.  A port is a link one way: with a table every link a hop's route crosses, so the hops of a tree that
   cross one link charge it each (relayed hops share the links they cross); its length grows by (1 + eps x flow/capacity)
   and its capacity is 1/beta (1 where the map gives no cost).  A root's equal trees merged, its `most` heaviest kept
   and their weights its shares.  Writes root, parent (nodes a tree) and log_weight for at most nodes x most trees;
   returns the count, 0 where the map does not join its nodes (mesh_link_map_joined).  At two nodes, each node's one
   tree. */
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

/* Shortest-path trees: each node's in-tree over the map's hops (two of its nodes linked where the route between them on
   its table, mesh_routes' where it has none, crosses relays alone, the hop as long as that route), a node's parent one
   hop nearer the root on a shortest path, chosen among equals so the tree set's edges spread over the physical links
   (a local search on the sum of the squares of the links' loads, each the tree edges whose hops cross it [Rosenthal
   1973], then the least hash of the marks), the parents climbing to the root; the set is a communicator's own and
   moves no route; one tree a node at log weight 0; writes nodes trees, returns their count (0 where a hop path does not
   reach). */
uint32_t mesh_trees_shortest(const struct mesh_link_map *,uint32_t *root,uint32_t *parent,double *log_weight);
/* The room mesh_compile needs at `chunks`: parts and moves. */
void mesh_compile_room(const struct mesh_trees *,uint32_t chunks,uint32_t *parts,uint32_t *moves);
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
   Each tree's part is cut into `chunks` parts (at least 1), chunk j of every move at its round + j, so a node forwards
   a part's chunk as it arrives rather than the whole part once all of it has: the trees pipelined [Blink; NCCL's
   chunked trees; MSCCLang's chunks, Cowan et al. 2023]; a segment's parts are chunk-major (chunk 0 of each of its
   trees, then chunk 1, ...), so a message coalesces a chunk across the segment's trees.  Going up a tree a node sends
   chunk j at round its height + j, after its children's chunk j; coming down, a node at depth d receives chunk j at
   base + d - 1 + j and sends it at base + d + j, base the highest of the trees' heights (the reduce-scatter's first
   chunk's last round plus one); a round's SENDs come before its receives, so a SEND waits on lower rounds alone.  A
   message is coalesced with the next between the same two nodes in the same round where their parts adjoin and each
   end's buffer and step are the same.  `contributors` (a bit a node, 64 a word; NULL: every node) are the nodes whose
   operands an all-reduce combines, and the nodes whose segments an all-gather gathers: the operand is theirs in node
   order, each down its node's trees to every node, the others holding none.  0, or EINVAL. */
int mesh_compile(const struct mesh_trees *,uint32_t what,uint32_t root,int whole,uint32_t chunks,const uint64_t *contributors,
                 struct mesh_program *program,uint32_t *segment,double *log_weight,uint32_t *first,struct mesh_move *move);
/* A rank's steps of a program for an operand, a step a move (none where its elements are none), at most first[rank +
   1] - first[rank]; their count. */
uint32_t mesh_program_steps(const struct mesh_program *,uint32_t rank,struct mesh_operand,struct mesh_step *steps);
/* The wire a session's programs run on (libnccl-mesh's rings, nccl-mesh.c session_open): toward each other member two
   channels, a channel a ring of 2 lag slots each way, slot t % (2 lag) carrying position t; a step's piece on the
   `large` channel where it is more than lag small slots (else the small one) in pieces of its slot's bytes (`slot`,
   NCCL_BUFFSIZE over lag rounded up to a block, or `small`, one block: the bridges' block, the page their window's rows
   take, the same on every paired bridge); every position a whole slot on the wire, one where the end has nothing to
   send a filler's; an end publishing position t once it has consumed its peer's t - lag (NCCL's Simple protocol, its
   head credit the reverse direction's positions).  MESH_LAG is NCCL_STEPS and MESH_BUFFSIZE NCCL_BUFFSIZE's default,
   in bytes; mesh_wire_of the rule for a block and an NCCL_BUFFSIZE. */
#define MESH_LAG 8
#define MESH_BUFFSIZE 4194304
struct mesh_wire { uint64_t slot,small; uint32_t lag,padding; };
void mesh_wire_of(uint64_t block,uint64_t buffsize,struct mesh_wire *wire);
/* Whether a piece of `bytes` takes a wire's large channel, and its pieces there */
int mesh_wire_large(const struct mesh_wire *,uint64_t bytes);
uint64_t mesh_wire_pieces(const struct mesh_wire *,uint64_t bytes);
/* A step's positions on one channel: its index among the rank's steps, whether it is the rank's SEND, its first
   position and its pieces. */
struct mesh_place { uint32_t step,send; uint64_t first,pieces; };
/* The positions a rank's steps of one collective take on its channel toward `peer` of class `large` (mesh_wire_large),
   from position `at`: each round's (in the steps' round order, a round's steps in their order) SENDs and receives side
   by side from where the round before ended, a piece's pieces consecutive; the end, and each step it places (one with
   no bytes none) in made[*placed ..], *placed counting them.  The one layout both ends of a channel make of their two
   ranks' steps (libnccl-mesh lay, order_group; mesh_program_time). */
uint64_t mesh_lay(const struct mesh_step *steps,uint32_t count,uint32_t peer,int large,const struct mesh_wire *wire,uint64_t at,
                  struct mesh_place *made,uint32_t *placed);
/* Each step's receive before it (by round, then the steps' order) of any bytes, -1 where none: a step's first piece
   waits until that receive is consumed, and so, by induction, every receive before it.  0, or ENOMEM. */
int mesh_steps_before(const struct mesh_step *steps,uint32_t count,int32_t *before);
/* A program's time in microseconds as libnccl-mesh runs it on `wire` over the map's ports, a link's two ways each
   its own (a node sends and receives on all its links at once, one message at a time each way): every rank's steps
   laid on each channel (mesh_lay), every position of each channel a whole slot of its class (a filler's too)
   published by its sender once the sender has consumed the peer's position lag back and, a piece's first, once the
   receive before its step is consumed (mesh_steps_before), and consumed by its receiver in position order once it
   has landed (a piece's first once the receive before it is consumed); a position crosses each port of its route
   once the port is free, a relay passing it on its alpha after it began to land and finishing no sooner than the
   port before it plus that alpha [Hockney 1994], a direct hop over several cables position t on cable t mod n at
   that cable's cost, a routed position t on stripe t mod stripes, on cable (h + stripe) mod c of each segment (h the
   hash of the route's ends' marks, as the bridges' is of the program's), at that cable's cost.  A rank's own work
   takes no time.  alpha in us and beta in ns a byte where the map gives a link no cost; the last consumption's time,
   negative where a receive has no SEND of its piece, the schedule stops or the map does not join its nodes
   (mesh_link_map_joined: no program crosses a pair no route joins). */
double mesh_program_time(const struct mesh_link_map *,const struct mesh_program *,struct mesh_operand,const struct mesh_wire *wire,
                         double alpha,double beta);
/* A collective's program for one operand chosen by its time: `what` (root's, a rooted one's) compiled along each of
   the `count` tree sets (an all-reduce also whole along each), cut into 1, 2, 4, ... MESH_CHUNKS chunks (each count
   timed: a time need not fall or rise steadily with the chunks, a piece's class and its slot's waste moving with
   them), each timed by mesh_program_time on `wire` at alpha and beta; *choice the fastest, the first of
   equals (in set order, whole after not, fewer chunks first; a later one taken only where it is faster by more than
   the rounding of a time's sums, 2^-30 of it): its set's index, whole, chunks and time.  The
   literature's rule that the best collective depends on the operand, the root and the map [Thakur, Rabenseifner &
   Gropp 2005; NCCL's tuner] as one function, which the default table (libnccl-mesh) and a planner's pricing both call:
   its time is the runtime's, so a chunk that leaves a position's slot part empty is priced at its whole slot.  0,
   ENOMEM, or EINVAL where nothing compiles or times. */
#define MESH_CHUNKS 64u
struct mesh_choice { uint32_t set,whole,chunks,padding; double time; };
/* The size classes a default table chooses at: a call of b bytes runs the choice made at the least 2^k of at least b,
   k from MESH_CLASS_LEAST (1 KiB) to MESH_CLASS_MOST (16 GiB; above it, that one), its operand 2^k / 4 elements of 4
   bytes (libnccl-mesh program_of; a planner prices a call by the same rule). */
#define MESH_CLASS_LEAST 10
#define MESH_CLASS_MOST 34
int mesh_choose(const struct mesh_link_map *,const struct mesh_trees *sets,uint32_t count,uint32_t what,uint32_t root,
                struct mesh_operand,const struct mesh_wire *wire,double alpha,double beta,struct mesh_choice *choice);
/* The planner's rules' version: the trees, routes, compiles, wire and choices above (the library's session handshake
   carries it with its digest of the members' map, tree sets and wire, so members planning by different rules fail the
   session rather than run it). */
#define MESH_PLAN_VERSION 2u

/* design/recovery.md#diagnosis: a node's health as its bridge last sampled it, carried in its record to every bridge.
   `sampled` the wall clock of the sample and `booted` of the node's boot (ns since the epoch); `session` its boot session
   (kern.bootsessionuuid); why it last restarted, `reset` (MESH_RESET_*) read from `shutdown`, the head of
   kern.shutdownreason, its power controller's records newest first ("wdog,..." its watchdog reset it; a "uv,..." record
   among the reset's own, its supply sagged under the controller's watch, an unplug or a rail under lockout; else
   "por,..." its supply was cut); its power source (`power`, MESH_POWER_*, with `charge` percent and `minutes` to empty, -1 where
   none is reported); its `free` and `wired` memory in bytes and the kernel's memory `pressure`
   (kern.memorystatus_vm_pressure_level: 1 normal, 2 warning, 4 critical); its thermal pressure (`thermal`,
   com.apple.system.thermalpressurelevel: 0 nominal, 1 moderate, 2 heavy, 3 trapping, 4 sleeping). */
enum { MESH_RESET_UNKNOWN, MESH_RESET_SUPPLY, MESH_RESET_WATCHDOG, MESH_RESET_OTHER, MESH_RESET_SAG };
enum { MESH_POWER_UNKNOWN, MESH_POWER_AC, MESH_POWER_BATTERY, MESH_POWER_UPS };
struct mesh_health {
  uint64_t sampled,booted,free,wired;
  uint32_t reset,power,pressure,thermal;
  int32_t charge,minutes;
  char session[40],shutdown[80];
};
/* A node's record as its bridge last sent it: its bridge's node and host name, its bridge's window (bytes), the node's
   memory (hw.memsize) and model (hw.model, NUL-terminated, MESH_MODEL_BYTES with its NUL; mesh.h struct mesh_record),
   and its health. */
#define MESH_MODEL_BYTES 32
struct mesh_node_health { uint32_t node; char name[MESH_NAME_BYTES]; uint64_t window,memory; char model[MESH_MODEL_BYTES];
                          struct mesh_health health; };
/* Every node's record the bridge holds (mesh_map's nodes, this node's own among them): their count (at most `capacity`
   written), or -errno where the bridge is unreachable. */
int mesh_nodes_health(struct mesh_node_health *out,uint32_t capacity);
/* design/bridge-and-ipc.md#requests: a channel from a client of this node to a service of a node (RFC 4254 s5, its
   direct-streamlocal channel), carried by the bridges over their links' control sockets, hop by hop along the map's
   route. A client connects to its own bridge at [::1]:MESH_PORT, sends a struct mesh_request (MESH_REQUEST_OPEN, the
   node's bridge number, this node's own too, and the service's name) and reads the bridge's int32 answer: 0 once the
   service's end is open, else an errno (EHOSTUNREACH: the map holds no path to the node; ENOENT: the node serves no
   such service; ECONNREFUSED: the service's socket refused; ETIMEDOUT). The connection then carries the client's
   bytes to the service and the service's back, each direction ended by its writer's half close (shutdown SHUT_WR);
   a channel that fails anywhere resets the connection. A service registers its listening Unix socket
   (MESH_REQUEST_SERVE, its name and its socket's path) on a connection it holds open: the registration lasts while
   that connection does, and a newer one of the same name replaces it. The bridge serves `hold` itself (#holds). */
#define MESH_SERVICE_BYTES 48
#define MESH_SERVICE_PATH 104
enum { MESH_REQUEST_OPEN=1, MESH_REQUEST_SERVE=2 };
struct mesh_request { uint32_t magic,kind,node,reserved; char service[MESH_SERVICE_BYTES]; char path[MESH_SERVICE_PATH]; };
#define MESH_REQUEST_MAGIC 0x4d524551u
/* a channel to `service` at `node`, open: its connection's descriptor, or -errno */
int mesh_request(uint32_t node,const char *service);
/* `service` registered as the Unix socket at `path`: the descriptor of the connection whose life the registration
   lasts (closing it ends the registration; it reads end of file once the bridge stops), or -errno */
int mesh_serve(const char *service,const char *path);

#ifdef __cplusplus
}
#endif
#endif
