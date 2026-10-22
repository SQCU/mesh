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

/* This node's bridge (design/transport.md section 14): its node, and its links up, one entry a link (a cable: two links
   toward one node are two entries), each with the node it reaches, its `streams` (the communicators the link carries at
   once toward that node, as many as its plan has lane pairs for: design/HETEROGENEOUS-LINKS.md; 0 in a record of a
   bridge before stream counts, which states none) and its `bandwidth` in bits a second (its live streams' shares
   together); returns their count (at most `capacity` written; 0: the node alone), or -errno where the bridge is
   unreachable.  A node is all a client names of the mesh: never a port or queue pair (design/rubrics.md, Three kinds of
   source). */
struct mesh_reach { uint32_t node,streams; uint64_t bandwidth; };
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
   lesser of its two ends' as their bridges report them, mesh_reached; 0 where none reports one), which the
   routes and the packed trees read: of routes of equally few links the one over the faster hops (mesh_routes), and a
   hop's capacity (mesh_trees_pack).  `streams` (NULL: none given) is link i's streams, the communicators it carries at
   once (mesh_map's: the lesser of its two ends' as their bridges report them, mesh_reached; 0 where neither states
   one): a communicator holds one stream toward each member it is joined to, so the communicators two nodes take at
   once are their links' streams together (a further one is refused at its init, EBUSY: rdma/NCCL.md).  No planner
   function reads it; mesh_map writes it, and mesh_map_members carries it where both maps give it.
   `relays` (a graph's, with a table) more nodes, numbered from `nodes`, that a route may cross and that hold no rank:
   a collective's ranks are the map's first `nodes`, and its links may join any of its nodes and relays (a
   communicator's map has none: the bridges forward nothing, nccl-mesh.c map_of).  `mark` (NULL: each node's number)
   is each node's and relay's mark, distinct (mesh_map's: its bridge's node number, the same wherever it is numbered).
   `next` (NULL: none, and no relays) is the routing table, an operand (mesh_routes makes one): of N = nodes + relays,
   each node b's in-tree, next[v * N + b] v's parent in it, a node a link joins to v (b's own entry b; v itself where
   no route reaches b).  The route between a and b is the path up the in-tree of the one of the higher mark from the
   other, so the route from b to a is the one from a to b reversed (mesh_route_path).  With a table, two of the map's
   nodes are linked where the route between them crosses no other of them (relays alone), and that hop's bytes cross
   every link of it (mesh_trees_pack).  Without one, two nodes are linked where a link joins them.  Several links listed
   between one pair are parallel: the planner counts them together at n times the slowest one's bandwidth (mesh_routes,
   mesh_trees_pack); mesh_map lists every link, so two nodes joined by two cables are two links.  `link`, `bandwidth`,
   `next`, `mark` and `streams` are the caller's storage. */
enum { MESH_LINKS_MESH, MESH_LINKS_RING, MESH_LINKS_TREE, MESH_LINKS_GRAPH };
struct mesh_link_map { uint32_t kind,nodes,links; uint32_t (*link)[2]; uint64_t *bandwidth; uint32_t relays; uint32_t *next,*mark,*streams; };
/* 0 where a map is one, else EINVAL: every link's nodes among its nodes and relays; a ring one cycle through every node
   with one link into each; a tree nodes-1 links, one into each node but the root, every node reaching the root; with a
   table (a graph's, where it has relays), its every entry a node or relay its row's node is linked to, or that node.  A
   graph need not join its nodes: the bridges' records are a map while a node is reached by no bridge, a node no link
   joins among its nodes (mesh_map's whole map), and a planner or driver reads it as it is. */
int mesh_link_map_check(const struct mesh_link_map *);
/* 0 where a map (one mesh_link_map_check takes) joins every two of its nodes, else ENOTCONN, apart[0] and apart[1]
   (apart NULL: none) the first two it does not join, the first the lower (EINVAL where it is no map): a graph's links
   connect them, and with a table every route between two of them reaches its end; a ring, a tree and a mesh join
   theirs by their form.  The maps a collective's trees and a communicator's collectives run on (mesh_trees_pack,
   mesh_trees_shortest; nccl-mesh.c comm_routes fails a collective on a map that does not join its members, naming
   two). */
int mesh_link_map_joined(const struct mesh_link_map *,uint32_t *apart);
/* The routing table of a map's links (into next, N x N as struct mesh_link_map's; map->next is not read): each node
   b's in-tree of shortest routes toward it, by fewest links, then the least sum of their hops' inverse bandwidths
   (a pair's n parallel links together: 8e15 over n times the slowest one's bits a second, in whole fs a byte; a hop
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
   at its listed bandwidth or none), each with its streams where both maps give them (map->streams and
   whole->streams; 0 for a mesh's pair it lists no link of), and its table the whole map's, written where map->next is
   given: so two members
   are linked where their route crosses no other member, and that hop's bytes cross each link of its route
   (mesh_trees_pack).  n 0: the whole map itself.  whole_node[i] (NULL: none) the whole map's node map node i is
   (whole->nodes for a member it does not hold).  The kind MESH_LINKS_MESH where there are no relays and a link joins
   every pair (several a pair's parallel links), else MESH_LINKS_GRAPH.  On entry map->nodes and map->links are the
   room of whole_node, mark and next's rows (the map's nodes and relays together), and of link, bandwidth and streams;
   returns the
   map's nodes, map->relays its relays, or -errno (ENOSPC: the room is short, map->nodes and map->links then the room
   needed; EINVAL: a member twice, relays and map->next or map->mark NULL, or a whole map with relays; ENOMEM).  The
   one rule mesh_map reads the bridges' records by. */
int mesh_map_members(const struct mesh_link_map *whole,const uint32_t *members,uint32_t n,uint32_t *whole_node,struct mesh_link_map *map);
/* The map of the node's bridge (design/transport.md) as of one map serial, from its records of the links every node's
   bridge has up (mesh_reached's views), so the same records give every member the same map: the whole map, its nodes
   the records' ascending by bridge, two joined where each one's bridge reaches the other, by as many links as the
   fewer of the two records lists toward the other (two cables between them two links: each record's entries toward
   the other ranked by bandwidth, then streams, the faster first, and its k-th taken with the other's k-th), each at the
   lesser of the two ends' bandwidths and of their streams (each 0 where neither end states one), the links of a pair
   in that rank, each node marked by its bridge's number, its table mesh_routes';
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
/* Each node's spanning in-arborescences packed fractionally into the map's links, each way: Garg and Konemann's maximum
   concurrent flow [FOCS 1998], every root's trees together carrying one unit, each phase sending every root's unit
   along minimum-length arborescences [Chu & Liu 1965; Edmonds 1967] of the map's hops, a hop's length the sum of the
   lengths of the links it crosses, each one way: with a table every link a hop's route crosses, so the hops of a tree
   that cross one link charge it each (hops through relays share the links they cross); a link's length one way grows by
   (1 + eps x flow/capacity) and its capacity is its pair's parallel links' count times the slowest one's bandwidth
   (where the map gives some pair no bandwidth, its parallel links' count: every link alike).  A root's equal trees
   merged, its `most` heaviest kept and their flows' shares their log weights (the packing's description: libnccl-mesh cuts a configured form's segments among
   them by the form's shares, nccl.h ncclMeshForm, never by these).  Writes root, parent (nodes a tree) and log_weight
   for at most nodes x most trees, each root's heaviest first; returns the count, 0 where the map does not join its
   nodes (mesh_link_map_joined).  At two nodes, each node's one tree. */
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
                           operand from a copy, or, where it sends that to one node only and forwards nothing (two
                           nodes, the direct exchange), from the operand itself: its one SEND and its one REDUCE of
                           each chunk then lie at the same positions of one channel, and the executors publish a
                           position before they combine into it (libnccl-mesh), so the SEND reads the operand before
                           the sum is made in it (its dependent crossings the trees' height: the graph's eccentricity
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
/* The same over an operand cut into segments of the given sizes (MPI_Allgatherv's and MPI_Reduce_scatter's counts, one
   a segment in segment order: a reduce-scatter's and an all-gather's segment s is node s's, node s's of its
   contributors where it has them): segment s the segments[s] elements after the segments before it, the sizes as many
   as the program's segments (the most segment a part names, plus one) and summing to the operand's elements, a size 0
   a segment whose steps carry nothing (none made); NULL the equal cut, mesh_program_steps'.  Within a segment its parts
   are cut by their weights as there.  0 where the sizes do not sum to the operand (mesh 430cf7b2^ mesh_cut, 3ab709aa:
   the planner's `segments`, cut alike by every program). */
uint32_t mesh_program_steps_cut(const struct mesh_program *,uint32_t rank,struct mesh_operand,const uint64_t *segments,struct mesh_step *steps);
/* The wire a communicator's channels run on (design/transport.md section 14): toward each member it is joined to two
   channels, each a class of the stream it holds toward that member's node (the small class's unit a frame, the large
   class's the bridges' block, `block`, the same at every bridge), position t of a channel a run of the class's units
   both ends lay alike from the pieces before it (libnccl-mesh lay), the piece's bytes in whole units (one where the
   end sends nothing: a filler's), a run that would pass the ring's end starting at its next turn; a step's piece on the large channel where it is more
   than one small position's bytes or its element wider than that (else the small one), in pieces of the whole elements
   a position of its class holds (`small` a small position's bytes, `slot` a large one's: each class's most units a run
   such that the 2 lag positions a receiver holds unconsumed and a run's padding fit its ring, mesh_wire_pieces).  An
   end publishes position t once it has consumed its peer's t - lag (NCCL's Simple protocol, its head credit the
   reverse direction's positions); MESH_LAG is NCCL_STEPS.  mesh_wire_of the rule for a block; mesh_wire_units a
   piece's units in its class. */
#define MESH_LAG 8
struct mesh_wire { uint64_t slot,small,block; uint32_t lag,padding; };
void mesh_wire_of(uint64_t block,struct mesh_wire *wire);
uint64_t mesh_wire_units(const struct mesh_wire *,int large,uint64_t bytes);
/* Whether a piece of `bytes`, in elements of `element` bytes (its operand's, struct mesh_operand element_bytes), takes a
   wire's large channel (more than a small slot's bytes, or an element wider than one), and its pieces there: a
   position carries whole elements, its slot's bytes less their remainder over the element (MPI's rule: no datatype's
   element is split, so a combine of a position's piece combines whole elements; a typed element of 1 to 8 bytes on
   slots of whole blocks or whole frames, as before) */
int mesh_wire_large(const struct mesh_wire *,uint64_t bytes,uint64_t element);
uint64_t mesh_wire_pieces(const struct mesh_wire *,uint64_t bytes,uint64_t element);
/* A step's positions on one channel: its index among the rank's steps, whether it is the rank's SEND, its first
   position and its pieces. */
struct mesh_place { uint32_t step,send; uint64_t first,pieces; };
/* The positions a rank's steps of one collective take on its channel toward `peer` of class `large` (mesh_wire_large),
   from position `at`: each round's (in the steps' round order, a round's steps in their order) SENDs and receives side
   by side from where the round before ended, a piece's pieces consecutive; the end, and each step it places (one with
   no bytes none) in made[*placed ..], *placed counting them.  The one layout both ends of a channel make of their two
   ranks' steps (libnccl-mesh lay, schedule_group). */
uint64_t mesh_lay(const struct mesh_step *steps,uint32_t count,uint32_t peer,int large,const struct mesh_wire *wire,uint64_t at,
                  struct mesh_place *made,uint32_t *placed);
/* Each step's receive before it (by round, then the steps' order) of any bytes, -1 where none: a step's first piece
   waits until that receive is consumed, and so, by induction, every receive before it.  0, or ENOMEM. */
int mesh_steps_before(const struct mesh_step *steps,uint32_t count,int32_t *before);
/* A compiled program's advisories on a wire for an operand (design/transport.md): warnings only,
   each a line "check: ..." handed to warn(arg, line), the check one of projection (each pair's SENDs against its
   receives, and both ends' positions on each channel), priorities (each rank's steps by round, a round's SENDs first;
   a SEND and its receive in one round), depth (an element wider than its slot; a step that waits on its own channel at
   or past its first position), causality (the program's publications and consumptions under the wire's rule and the
   executors' order, a position whose publication reads in place what its consumption writes consumed after it, without
   timing: a cycle among them is a deadlock) and hazard (a receive that may write what a SEND of the same rank still
   reads, piece by piece, or a SEND not ordered after a receive of what it reads).  It refuses nothing and changes nothing; the count
   of warnings, UINT32_MAX where its room could not be had.  It is mesh_group_advise of a group of the one call. */
uint32_t mesh_program_advise(const struct mesh_program *,const struct mesh_wire *,struct mesh_operand,void (*warn)(void *,const char *),void *arg);
/* design/transport.md, a group's calls on one communicator's wire, as libnccl-mesh issues them between
   ncclGroupStart and ncclGroupEnd: call i of kind MESH_GROUP_*, its steps at rank r steps[r][0 .. count[r]) (a row of
   none where count[r] is 0), every rank's calls in one call order (a group's point-to-point calls are one EXCHANGE of
   the whole communicator, every rank's: nccl.h ncclGroupEnd); `polls` its GPU waits' bound (nccl.h ncclMeshStream
   polls, one a group) and `bound` the most loads a wait may make (nccl.h NCCL_MESH_POLLS; 0: not read, a host group).
   mesh_group_lay: the positions rank `rank`'s calls take on its channel toward `peer` of class `large` from `at`, each
   collective's in call order (mesh_lay), then each exchange's; each placement its call and mesh_lay's place in
   made[*placed ..], scratch the room of the most steps a call has at the rank; the end.  EXCHANGE: a pattern's blocks
   forwarded along a routing table (mesh_forward_steps: ncclAlltoAll, ncclGather, ncclScatter, ncclMeshAlltoAllV, a
   counted all-to-all, a group's sends and receives), each final receive into its own place, or, at a member between
   a block's ends, into its relay buffer. */
enum { MESH_GROUP_COLLECTIVE, MESH_GROUP_EXCHANGE };
/* The receive before each step of a call of kind `kind` (mesh_steps_before; the walk's rule, a step's first position
   waiting until that receive is consumed): an EXCHANGE's receives each progress alone, as NCCL's per-peer sends and
   receives do, and a SEND of it from what a receive of an earlier round wrote (the same buffer, first element and
   elements: the part a member between a block's ends forwards, mesh_forward_steps) waits on that receive piece by
   piece: its piece i until the receive's piece i is consumed (the two cut alike, the same bytes in the same elements:
   mesh_wire_pieces), so a forwarded position waits on its own landing alone, never on the whole part's (design/rubrics.md
   C4); every other step on none.  So every wait of an exchange is on a position of a strictly earlier round, and each
   channel lays its rounds in order (mesh_lay), which leaves its waits no cycle at any lag (an event waits only on events
   at earlier or equal positions of its own channel, and on another channel's only at an earlier round).  0, or
   ENOMEM. */
int mesh_group_before(uint32_t kind,const struct mesh_step *steps,uint32_t count,int32_t *before);
struct mesh_group_call { uint32_t kind,padding; const uint32_t *count; const struct mesh_step *const *steps; };
struct mesh_group { uint32_t nodes,calls,polls,bound; const struct mesh_group_call *call; };
struct mesh_group_place { uint32_t call,padding; struct mesh_place place; };
uint64_t mesh_group_lay(const struct mesh_group *,uint32_t rank,uint32_t peer,int large,const struct mesh_wire *,uint64_t at,
                        struct mesh_place *scratch,struct mesh_group_place *made,uint32_t *placed);
/* A routing table, an operand (design/rubrics.md C3: made between calls, passed to each call, never chosen inside
   one): for each ordered pair (a, b) of `nodes` nodes, the routes a block from a to b takes and each one's share of it,
   so a pair's bytes cross several paths at once and every link carries a share (each link far slower than a node's
   memory, through which a member between two others relays).  Pair (a, b)'s routes are [pair[a * nodes + b], pair[a *
   nodes + b + 1]) (pair: nodes x nodes + 1 entries, nondecreasing; a pair a == b none); route k is the nodes
   node[start[k] .. start[k + 1]) from a to b inclusive, every two consecutive ones linked (mesh_link_between), none
   twice, and share[k], above 0, its share: a block of e elements is cut among its pair's routes in their order, each
   the floor of e times its share over the pair's shares together, the first ones one element more for the remainder
   (mesh_compile's cut of a segment among its trees).  `routes` the routes (start: routes + 1 entries).  A pair joined
   by no route moves nothing (a map that does not join its nodes). */
struct mesh_routing { uint32_t nodes,routes; const uint32_t *pair,*start,*node; const double *share; };
/* 0 where a routing table fits a map of as many nodes (struct mesh_routing's rules, every hop one the map links), else
   EINVAL. */
int mesh_routing_check(const struct mesh_link_map *,const struct mesh_routing *);
/* The room a routing table of `nodes` nodes and at most `most` routes a pair takes: *routes its routes (share's
   entries; start's, plus one), *hops its route nodes (node's entries); pair's is nodes x nodes + 1. */
void mesh_routing_room(uint32_t nodes,uint32_t most,uint32_t *routes,uint32_t *hops);
/* A tree set's routing table (into pair, start, node and share, of mesh_routing_room's room at the most trees a root
   has): pair (a, b)'s routes the paths from a up each of b's trees (mesh_trees_path's in each tree, not only the
   heaviest), each share the sum of the weights of the trees that give that path (exp(log_weight) over the heaviest of
   b's), ordered by their nodes (the fewer first, then by node number at the first that differs), so the table is a
   function of the trees' paths and weights alone, not of their order.  Writes *routing over those arrays; its routes
   (UINT32_MAX where its room could not be had, nothing written).  A pair whose destination roots no tree has none. */
uint32_t mesh_routing_trees(const struct mesh_trees *,struct mesh_routing *routing,uint32_t *pair,uint32_t *start,uint32_t *node,double *share);
/* The routing solver (libnccl-mesh's at every communicator's creation, nccl.h ncclMeshCommSetRouting): the map's
   packed trees (mesh_trees_pack at `eps` and `most` trees a root: Garg and Konemann's concurrent flow over the links'
   bandwidths), as mesh_routing_trees makes a table of them, so each pair's bytes split over the paths the packing
   loads, each by its flow; where the map does not join its nodes (no packing), each pair its route on the map's table
   (mesh_route_path), one share, and none for a pair the map does not join.  Room: mesh_routing_room's at `most`.  Its
   routes (UINT32_MAX where its room could not be had). */
uint32_t mesh_routing_pack(const struct mesh_link_map *,double eps,uint32_t most,struct mesh_routing *routing,uint32_t *pair,uint32_t *start,
                           uint32_t *node,double *share);
/* A block of a pattern (who sends how many elements to whom, the same at every rank): `elements` elements of
   `element_bytes` bytes (its type `type`, carried) from rank `from` to rank `to`, read at `from` from its buffer
   `source` from element `first`, landed at `to` in its buffer `target` from element `at`. */
struct mesh_block { uint32_t from,to,source,target,type,element_bytes; uint64_t first,at,elements; };
/* The forwarding core (design/rubrics.md C2: one mechanism on every cabling, a direct exchange a route of one hop): a
   pattern's steps at `rank` along a routing table.  Each block (from != to, of elements) is cut among its pair's
   routes (struct mesh_routing) and each part sent along its route, hop i in round i: the route's first rank SENDs it
   from its source, its last receives it (COPY) into its target, and a rank between them receives it into its buffer
   `relay`, at the next of the bytes it relays (its parts in block, then route order, each at a multiple of its element's
   bytes), and SENDs it on from there in the next round, piece by piece, each piece once that piece has landed
   (mesh_group_before: EXCHANGE).  A round's SENDs, then its receives, each in block, then route order, so a hop's SENDs
   at one end and its receives at the other are the same parts in the same order (mesh_lay).  `blocks` are the same at
   every rank in from, to, elements, element and order (a rank reads `source` and `first` of its own blocks alone, and
   `target` and `at` of the blocks to it alone).  block[k] (NULL: not written) the index of the block step k moves a
   part of; *relayed (NULL: not written) the bytes of `relay` the rank uses.  Their count, at most mesh_forward_room's;
   UINT32_MAX, nothing written, where a block names a rank past the table's nodes or moves elements between two ranks
   the table joins by no route, or its room could not be had. */
uint32_t mesh_forward_steps(const struct mesh_routing *,uint32_t rank,const struct mesh_block *blocks,uint32_t count,uint32_t relay,
                            struct mesh_step *steps,uint32_t *block,uint64_t *relayed);
/* The most steps mesh_forward_steps makes at a rank of a pattern: two a route of each block. */
uint32_t mesh_forward_room(const struct mesh_routing *,const struct mesh_block *blocks,uint32_t count);
/* A group's advisories: mesh_program_advise's checks over its calls laid together (projection of both ends' positions on
   each channel and of each call's steps; causality over every call's steps on the shared positions, an exchange's
   forwarded pieces each after its own landing; each call's priorities, depth and hazards, an exchange's read after
   write piece by piece), and bound (GPU waits of polls 0 or past `bound`: a wait whose landing never comes holding its
   queue past the platform's deadline, design/transport.md).  Returns as mesh_program_advise. */
uint32_t mesh_group_advise(const struct mesh_group *,const struct mesh_wire *,void (*warn)(void *,const char *),void *arg);
/* The planner's rules' version: the trees, routes, compiles and wire above, and libnccl-mesh's table of programs made
   of them (nccl-mesh.c comm_routes: the caller's, or the communicator's configured forms, nccl.h ncclMeshCommForms,
   and then every collective along the caller's paths or the shortest-path trees, uncut).  6: a position is a run of
   its class's units (mesh_wire_units).  7: the single-phase all-reduce's node that sends its operand to one node only
   and forwards nothing sends it from the operand itself (the two-node direct exchange: no scratch). */
#define MESH_PLAN_VERSION 7u

/* design/transport.md: what a node's record carries of its health to every bridge: `booted`, the wall
   clock of the node's boot (ns since the epoch), and its `available` memory as its bridge last sampled it (the bytes
   the kernel can give without pressure, the measure jetsam keeps: kern.memorystatus_level percent of hw.memsize). */
struct mesh_health { uint64_t booted,available; };
/* A node's record as its bridge last sent it: its bridge's node and host name, its bridge's region (bytes its clients may hold), the node's
   memory (hw.memsize) and model (hw.model, NUL-terminated, MESH_MODEL_BYTES with its NUL; mesh.h struct mesh_record),
   and its health. */
#define MESH_MODEL_BYTES 32
struct mesh_node_health { uint32_t node; char name[MESH_NAME_BYTES]; uint64_t region,memory; char model[MESH_MODEL_BYTES];
                          struct mesh_health health; };
/* Every node's record the bridge holds (mesh_map's nodes, this node's own among them): their count (at most `capacity`
   written), or -errno where the bridge is unreachable. */
int mesh_nodes_health(struct mesh_node_health *out,uint32_t capacity);

#ifdef __cplusplus
}
#endif
#endif
