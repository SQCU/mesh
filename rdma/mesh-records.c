/* The bridge's map (design/page-queue-change-request.md, "State of the code": the resident bridge keeps a map): this
   node's record and every other node's newest heard, in the region for the collective and the bridges' rule to read
   (mesh.h struct mesh_record, struct mesh_reach, hdr `map`, `map_serial`), its health samples, and the client's
   watch.  From tag pre-return-20261103 rdma/mesh-records.c, whole; only interfaces are adapted (main's region fields,
   its one client word, the map's area laid out with the region in place of the tag's arena lease), and each line that
   is not the tag's has a note at its spot.  The map holds
   Thunderbolt-5-linked members only: a record no paired path from this node reaches is left out (map_put), and every
   reach entry is a cable a port paired (mesh-ports.c record_reach).
   2026-11-04, owner B2 (design/aot-compilation-change-request.md A7; design/plan-to-bridge-change-request.md R8;
   design/compilation-coordination.md, Q2 to Q6): the client's watch is over the claims' heads, any number of plans
   standing, and this file holds the island's fixed sequencer besides: the island and its tree from the map (island_set),
   the SUBMIT and ORDER messages (order_heard), each plan's submission at its sender (orders_submit) and each ORDER's
   delivery and release here (orders_deliver).  What must stay: every one of them on the bridge parent's one event-loop
   thread, the map's one writer */
#include "mesh-bridge.h"
/* not the tag's: main's mesh-bridge.h does not include the library's header (the tag's included mesh-call.h), and
   mesh_retire and mesh_retired_release are declared there (mesh_arena_alloc, the tag's third, is called here no
   longer: map_grow's note) */
#include "mesh-dataflow.h"
/* MESH_HEALTH_NS is mesh-bridge.h's (review R2, B2) */
#include "mesh-graph.h"

/* tag pre-return-20261103 rdma/mesh-records.c busy_tell (3f72cbde), adapted: a client's hold is main's client word */
/* design/transport.md section 14: the streams held, an owner word naming a client (a stream whose holding ended with
   units outstanding names none: MESH_STREAM_GIVEN alone), told in the region (`busy`), and the wall clock at which
   none last came to be (`idle_since`, 0 while any is) */
/* differs from the tag: main has no streams (mesh-bridge.h: the data path is the page lists'), and what a client holds of
   the bridge is its one client word, hdr `client` (the client's pid in its low 32 bits), so `busy` is 1 while that word
   names a client, 0 where it names none or this bridge (main's mesh_retire writes its retirer's pid there until it
   stores 0), in place of the tag's count of streams whose owner names a client */
/* 2026-11-04, owner B2 (design/plan-to-bridge-change-request.md R8: "there is no client word"; e6bd7db2's section 6,
   "Holders and clients", as design/compilation-coordination.md ruling 1 and Q5 take it: "busy = 1 where any open claim
   names a live client other than the bridge"), reversing the note above's one-client form, in which `busy` read hdr
   `client`, which nothing writes from 110 (L3): `busy` is clients_turn's walk of the claims' heads, 1 where an open
   claim's owner is a process other than this bridge that its watch found live.  What must stay: `idle_since` and `busy`
   stored as the tag stores them */
static void busy_tell(uint64_t busy){
  if(busy)atomic_store_explicit(&M->idle_since,0,memory_order_relaxed);
  else if(!atomic_load_explicit(&M->idle_since,memory_order_relaxed))atomic_store_explicit(&M->idle_since,clock_gettime_nsec_np(CLOCK_REALTIME),memory_order_relaxed);
  atomic_store_explicit(&M->busy,busy,memory_order_release);
}
/* tag pre-return-20261103 rdma/mesh-records.c, the map's lease (3f72cbde), `static uint32_t map_page,map_pages,map_room;`,
   is not here: no lease exists (map_grow's note); the map's room is hdr `map_room` */
/* tag pre-return-20261103 rdma/mesh-records.c available_now (3f72cbde), verbatim */
/* design/transport.md: the bytes the kernel can give without pressure (kern.memorystatus_level percent of
   the node's memory), read now */
static uint64_t available_now(void){
  int level=0;size_t size=sizeof level;
  return sysctlbyname("kern.memorystatus_level",&level,&size,NULL,0) || level<0?0:(uint64_t)level*(memory/100);
}
/* tag pre-return-20261103 rdma/mesh-records.c, the health sampled (3f72cbde), verbatim */
/* design/transport.md: what this node's record carries of its health (mesh-plan.h struct mesh_health), its
   boot read once and its available memory at most MESH_HEALTH_NS apart (health_turn) */
static struct mesh_health health;
static uint64_t health_sent;
/* tag pre-return-20261103 rdma/mesh-records.c sysctl_text (3f72cbde), verbatim */
void sysctl_text(const char *name,char *out,size_t room){
  size_t size=room;
  if(sysctlbyname(name,out,&size,NULL,0))out[0]=0;
  out[room-1]=0;
}
/* tag pre-return-20261103 rdma/mesh-records.c health_init (3f72cbde), verbatim */
void health_init(void){
  struct timeval booted;size_t size=sizeof booted;
  if(!sysctlbyname("kern.boottime",&booted,&size,NULL,0))health.booted=(uint64_t)booted.tv_sec*1000000000u+(uint64_t)booted.tv_usec*1000u;
  health.available=available_now();
}

/* tag pre-return-20261103 rdma/mesh-records.c known_at (3f72cbde), verbatim */
static uint32_t known_count;
static struct known *known_at(uint32_t node){
  for(uint32_t i=0;i<known_count;i++)if(known[i].node==node)return known+i;
  struct known *at=NULL;
  if(known_count<MESH_KNOWN)at=known+known_count++;
  else for(uint32_t i=0;i<known_count;i++)if(!record_held(known[i].node) && (!at || known[i].heard<at->heard))at=known+i;
  if(at)*at=(struct known){.node=node};
  return at;
}
/* tag pre-return-20261103 rdma/mesh-records.c map_grow (3f72cbde), adapted then to main's mesh_arena_alloc, is not
   here.  2026-11-03, design/disjoint-pages-change-request.md, Appendix A, A3 ("Sharing the region between programs":
   "The bridge's own pages (its records) are laid out once with the region, not taken from any program's section") and
   A4 (mesh-records.c's call of mesh_arena_alloc goes), reversing this file's first form, in which map_grow leased the
   records arena pages outside the registered window (mesh_arena_alloc of the bridge's own context `bridge`), leased
   again twice as large when the map outgrew them, and recorded them in struct mesh_bridge_pages `map`, `map_pages` and
   `held_pages`.  Why: the arena is the programs' sections alone, and a page the bridge took there was one call-site
   allocation among those A4 removes.  Now the map's records are its area of the layout: hdr `map_room` records
   (MESH_MAP_ROOM) at hdr `map_off`, both set by mesh_layout (mesh-map.h) before the arena and never moved, so nothing
   grows, nothing is copied and the bridge holds no arena page.  What must stay: the map is written at `map_off` and
   within `map_room` alone; a map past the room leaves its records past it out (map_put), as the tag left out what a
   failed lease could not hold */
/* tag pre-return-20261103 rdma/mesh-records.c map_adopt (3f72cbde), adapted: the records' clearing is a change of the
   map's nodes */
/* design/transport.md section 2: an adopted region's map pages, recorded by the bridge before (mesh-observe.h struct
   mesh_bridge_pages), taken again as this bridge's, its records none until this bridge writes its own */
/* differs from the tag (2026-11-03, design/disjoint-pages-change-request.md, Appendix A, A3, "Sharing the region
   between programs"): the map's area is the layout's (hdr `map_off`, `map_room`), which region_open's region_same
   (mesh-link.c) found the same in the adopted region as in this bridge's layout, so nothing is taken again: the tag's
   test of `map` (MESH_ABSENT where none was leased), its taking of `map`, `map_pages` and the room, the held pages'
   count and the store of `map_off` go, and the area is adopted as it lies */
/* not the tag's (2026-11-04, owner B2): the island computed from the map, after map_put and map_adopt (below) */
static void island_set(void);
/* 2026-11-05 (design/compilation-coordination.md, "The revisions of b5a3c6e6", RV): whether a record map_put wrote has
   the sequencer create a view (below) */
static int view_announced(const struct mesh_record *record);
void map_adopt(void){
  /* differs from the tag: the records cleared between two notifications of `map`, and `map_serial` moved where any
     were held, since every change of the map's nodes moves it (mesh.h `map_serial`): a plan made against the earlier
     bridge's map is refused.  region_adopt (mesh-link.c) has made `map` even before this, as at the tag */
  mesh_notify(&M->map);
  const int moved=atomic_load(&M->map_nodes)!=0;
  if(moved)atomic_fetch_add(&M->map_serial,1);
  atomic_store(&M->map_nodes,0);
  mesh_notify(&M->map);
  /* not the tag's (2026-11-04, sub-assignment 1: design/disjoint-pages-change-request.md, D.3.6; design/disjoint-
     pages-rulings-coordination.md, R2): where the serial moved, the plan made from the earlier bridge's map no longer
     stands, so its receives not landed are cancelled (mesh-flow.c plan_stale), after the write's closing
     notification, on this thread (region_adopt's, main's before the event loop).  What must stay: the call after the
     serial's move and its closing notification */
  if(moved)plan_stale();
  /* 2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", SQ2): hdr `delivered` stored 0 at the
     bridge's start, where this bridge's nextdeliver is 1 (Défago et al., Fig. 5, "Initialization: nextdeliver := 1"),
     so a port reads no count an earlier bridge left.  What must stay: before the event loop */
  atomic_store_explicit(&M->delivered,0,memory_order_release);
  /* not the tag's (2026-11-04, owner B2: design/compilation-coordination.md, Q3 and Q4): where the serial moved, the
     island, its sequencer and its tree are this map's (island_set), the sequencer's number starts again and the
     pending ORDERs of the old serial are dropped (2026-11-04, the coordinator's ruling on review V1's finding 4: kept,
     island_set's note; 2026-11-05, design/compilation-coordination.md, "The revisions of 46ae8fdf", SN: the number is
     never set back, design/aot-compilation-change-request.md, appendix item 2) */
  if(moved)island_set();
}
/* tag pre-return-20261103 rdma/mesh-records.c map_put (3f72cbde, abeffc1b), adapted: node-id order and the map serial */
/* RFC 2328 s14: the map with `record` put in (its node's held record replaced, else added) and every record no paired
   path from this node reaches left out, written only where it differs from the map held (so a record that changes
   nothing, or that the prune drops at once, moves no serial) */
/* differs from the tag: a record added goes to its place in ascending node-id order, not to the end, so the map's
   records stay in that order (the held map is in it, and the prune keeps the order), and a node's index is its place
   (mesh.h `map_serial`); and `map_serial` moves, while `map` is odd, where the write changes the map's nodes (which,
   or how many) or any cable (a record's `reached` or any field of a reach entry: peer node, ports, queue pairs,
   bandwidth), and not where it changes only the rest of a record (its sequence, boot, region, memory, model); and the
   reach from this node is the graph module's (mesh_graph_reach over the joins, C9), not a walk of its own here, its
   reached set the tag's walk's */
/* an edge's end for mesh_graph_reach: join e's record on `side` (map_put) */
static uint32_t map_end(const void *context,uint32_t e,uint32_t side){ return ((const uint32_t (*)[2])context)[e][side]; }
/* not the tag's lines: the joins of records r[0, n) (mesh_joined, symmetric) as edges, (a, b) for a < b in index order,
   into `join` (room n (n - 1) / 2 + 1); their count.  2026-11-04, owner B2: map_put's loop, made one function so that
   island_set reads the map's joins by the same code (design/compilation-coordination.md, Q3).  What must stay: one
   reading of the joins */
static uint32_t map_joins(struct mesh_record *r,uint64_t n,uint32_t (*join)[2]){
  uint32_t joins=0;
  for(uint64_t a=0;a<n;a++)
    for(uint64_t b=a+1;b<n;b++){double beta;if(mesh_joined(r+a,r+b,&beta)){join[joins][0]=(uint32_t)a;join[joins][1]=(uint32_t)b;joins++;}}
  return joins;
}
static void map_put(const struct mesh_record *record){
  const uint64_t n=atomic_load(&M->map_nodes);
  struct mesh_record *r=mesh_records(M);
  uint64_t at=0;
  while(at<n && r[at].node<record->node)at++;
  const uint64_t added=at==n || r[at].node!=record->node;
  const uint64_t count=added?n+1:n;
  struct mesh_record *next=malloc(count*sizeof *next);
  unsigned char *reached=calloc(count,1);
  uint32_t *order=malloc(count*sizeof *order),(*join)[2]=malloc((count*(count-1)/2+1)*sizeof *join);
  if(!next || !reached || !order || !join){free(next);free(reached);free(order);free(join);return;}
  memcpy(next,r,(size_t)at*sizeof *r);
  memcpy(next+at,record,sizeof *record);
  memcpy(next+at+1,r+at+!added,(size_t)(n-at-!added)*sizeof *r);
  /* not the tag's lines: the joins (mesh_joined, symmetric) as edges, and this node's record the reach's start; none
     reached where the map holds no record of this node, as the tag's walk seeded none */
  uint64_t self=count;
  for(uint64_t a=0;a<count;a++)if(next[a].node==M->node)self=a;
  const uint32_t joins=map_joins(next,count,join);
  /* `parent` NULL: no tree is wanted of this reach (mesh-graph.h; design/compilation-coordination.md, "The graph
     module": island_set's is the one tree) */
  if(self<count)mesh_graph_reach((uint32_t)count,joins,map_end,join,(uint32_t)self,order,reached,NULL);
  free(join);
  uint64_t kept=0;
  for(uint64_t k=0;k<count;k++)if(reached[k])memmove(next+kept++,next+k,sizeof *next);
  if(kept!=n || memcmp(next,r,(size_t)n*sizeof *r)){
    /* not the tag's: whether the nodes or a cable change, read before the write (the write replaces the records read) */
    int moved=kept!=n;
    for(uint64_t k=0;!moved && k<kept;k++)
      moved=next[k].node!=r[k].node || next[k].reached!=r[k].reached ||
        memcmp(next[k].reach,r[k].reach,(size_t)MIN(next[k].reached,(uint32_t)MESH_PORTS)*sizeof *next[k].reach);
    mesh_notify(&M->map);
    /* not the tag's (2026-11-03, map_grow's note): the tag's `map_grow(kept);` gone, and the room the layout's hdr
       `map_room`, the tag's leased `map_room`; `moved=1` not the tag's: records past the room, left out as the tag left
       out those a failed lease could not hold, are a change of the map's nodes */
    if(kept>M->map_room){kept=M->map_room;moved=1;}
    memcpy(mesh_records(M),next,(size_t)kept*sizeof *next);
    atomic_store(&M->map_nodes,kept);
    if(moved)atomic_fetch_add(&M->map_serial,1);
    mesh_notify(&M->map);
    /* not the tag's (2026-11-04, sub-assignment 1: design/disjoint-pages-change-request.md, D.3.6; design/disjoint-
       pages-rulings-coordination.md, R2): a plan stands only while its map does, so where this write moved the serial
       the plan made from the map before it has its receives not landed cancelled (mesh-flow.c plan_stale), after the
       write's closing notification, on this thread (the bridge parent's one event-loop thread: every caller of map_put
       is, mesh-flow.c plan_cancel's note).  RFC 2328 section 16: routes come from one version of the database.  What
       must stay: the call after the serial's move and its closing notification */
    if(moved)plan_stale();
    /* not the tag's (2026-11-04, owner B2: design/compilation-coordination.md, Q3 and Q4): the island of the map as
       written (island_set), after the cancellation */
    /* 2026-11-05 (design/aot-compilation-change-request.md A7, "The membership service is the sequencer": "It also
       creates one whenever a member's announcement names a view other than the sequencer's installed one, with a
       counter not less than that view's identifier"; design/compilation-coordination.md, "The revisions of b5a3c6e6",
       RV and RV-1): island_set also where the write moved no serial and the record it put is such an announcement at
       the sequencer (view_announced), so island_set's view_create makes the view, its identifier "computed again from
       the counters it now holds" (A7), the map's.  Why: A7's correction, "a risen counter moved no map, so no view
       followed it".  What must stay: island_set after the cancellation at every move */
    if(moved || view_announced(record))island_set();
  }
  free(next);free(reached);free(order);
}

/* not the tag's (2026-11-04, owner B2): the island's fixed sequencer (design/aot-compilation-change-request.md A7;
   design/plan-to-bridge-change-request.md R8; the operator's ruling 1 of 2026-11-04, design/compilation-coordination.md:
   "it is rqeuired to choose nad use a scheduler").  Function executions of programs that run at once on the island are
   ordered by one sequencer: "To broadcast a message m, a sender sends m to the sequencer.  Upon receiving m, the
   sequencer assigns it a sequence number and relays m with its sequence number to the destinations.  The latter then
   deliver messages according to the sequence numbers" (X. Defago, A. Schiper and P. Urban, Total order broadcast and
   multicast algorithms: taxonomy and survey, ACM Comput. Surv. 36(4) 2004, section 7.1, Fig. 5, the UB variant), first
   in, first out (P. Barham et al., Pathways, MLSys 2022, section 4.4: "Our current implementation simply enqueues work
   in FIFO order").  A message m is a submission of a plan's executions [first, end) (struct mesh_order; Q1, Q2).
   What A7 adds to C3.3's run time, and only that (design/compilation-coordination.md, "The documents' revisions of
   edd4b583", DL6, Q6 as that part leaves it): the sequencer's `seqnum`, this bridge's pending ORDERs (`pending`, the
   head's `first` and `sequence` advanced in place as its executions are published), and each plan's words
   `submitted`, `released`, `taken` and its sequence words (mesh.h struct mesh_state); beside them the island this map
   gives (`island`), which Q3's tree reads (Q6, extended).  None is a page of the region, and none is allocated per
   execution: the pending ORDERs' array grows only where more ORDERs wait at once than ever before.
   2026-11-04, owner B4 (DL4, DL6): the plan this bridge released last (`last`), which Q4's hold read, is gone with
   the hold, and the ports' `served` is their `taken`.  Why: design/aot-compilation-change-request.md, appendix item
   2, "Delivery at a node": the parent publishes each released execution's sequence number in its plan's state, one
   word per copy, and each port merges them; "Nothing holds".  What must stay: nothing of it a page of the region, and
   nothing allocated per execution.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", CH): each plan's `released`, `taken`
   and sequence words are gone from what A7 adds, and the deliveries' channel is added (mesh.h struct mesh_delivery:
   the bridge's own pages, laid out with the region, as the map's are).  Why: design/aot-compilation-change-request.md
   at mesh 0e56d8da, appendix item 2, "through the channel above, in place of the sequence words (`sequences`),
   `taken[p]` and the parent's wait while they are full".  What must stay: nothing allocated per execution in the
   region */
/* the island of this node (design/aot-compilation-change-request.md A7: "The island is the map's Thunderbolt-linked
   component"): `self` this node's map index, `sequencer` the sequencer's (UINT32_MAX both where the map holds no
   record of this node), `parent` the node id of this node's parent in the sequencer's tree (0 at the sequencer and
   where none), `up` the port toward it and `down` the ports toward this node's children in the tree, `downs` of them
   (design/compilation-coordination.md, Q3); MESH_PORTS no port */
static struct island { uint32_t self,sequencer,parent,up,down[MESH_PORTS],downs; } island={UINT32_MAX,UINT32_MAX,0,MESH_PORTS,{0},0};
/* Defago et al., Fig. 5, the sequencer's `seqnum`: the number the next execution submitted takes, from 0 on each map
   serial (Q3: a plan stands only while its map does, D.3.6, so no order outlives its serial; Fig. 5 starts it at 1).
   2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", SN): from 0 at the bridge's start, and
   never set back (island_set's reset gone).  Why: design/aot-compilation-change-request.md, appendix item 2: "remove
   the coordination's Q3′: the sequencer's number never goes back to 0", the operator: "fictional problem + nobody
   asked + no citation + no algorithm + no explanation + idiomatic."  What must stay: the number only rising, Fig. 5's
   "seqnum := seqnum + 1".
   2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", SQ1): from 1 at the bridge's start, one
   number a message, where it was from 0, one an execution (a SUBMIT of [first, end) taking end - first numbers).  Why:
   design/aot-compilation-change-request.md A7: "The algorithm is their Fig. 5, the "UB" variant of §7.1, as written";
   "Sequencer: "Initialization: seqnum := 1"; "when receive (m): sn(m) := seqnum; send (m, sn(m)) to all; seqnum :=
   seqnum + 1""; "m is one submission ... One submission is one message however many executions it carries".  What
   must stay: the number only rising, never set back.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 3f106747", VW), reversing "never set back": 1 again
   at each view this bridge creates (view_create), and only rising within the view.  Why:
   design/aot-compilation-change-request.md A7, "Fig. 5 runs within each view as written.  Its initializations, "seqnum
   := 1" and "nextdeliver := 1; pending := ∅", happen at the view's creation at the sequencer and at its installation at
   each member"; appendix item 2: "the number begins again at 1 only at a view's creation, by A7's views".  What must
   stay: the number only rising within a view */
static uint64_t seqnum=1;
/* The view this bridge has installed (design/aot-compilation-change-request.md A7, "The island is a dynamic group, and
   each of its views is a static system"; G. V. Chockler, I. Keidar and R. Vitenberg, Group communication
   specifications: a comprehensive study, ACM Comput. Surv. 33(4), 2001; design/compilation-coordination.md, "The
   revisions of 3f106747", VW): `view` its identifier, 0 before the first, and so this bridge's counter, Transis's
   "local counter ... increased by a process upon each installation" (§3.1.1), which its record carries (mesh-map.h
   struct mesh_record `view`); `sequencer` and `boot` the process that created it, a bridge's node id and its start
   (hdr `boot`): "A process is one start of a bridge.  A bridge started again is a new process: "In Isis [RB91] a
   process recovering after a crash is assigned a different identifier (using a new incarnation number)"" (A7).  0 at
   this bridge's start, a new process's.  What must stay: `view` only rising (Property 3.2, Local Monotonicity) */
static struct installed { uint64_t view,boot; uint32_t sequencer; } installed;
/* the messages this node has passed through Fig. 5's delivery in every view it installed, each delivered into its
   plan's queue or discarded (§5.2.4), hdr `delivered`'s value and each queue entry's `sequence`: the message's place
   in this node's delivery order, which within a view is its sn(m)'s order and across views Local Monotonicity's.  Why:
   a delivered entry's number is the ports' merge key (mesh-flow.c lane_take: "of the first untaken entries of its
   standing plans, the least sequence number first"), and sn(m) begins again at each view, so the view's own numbers
   would set the key back (design/compilation-coordination.md, "The revisions of 3f106747", VW-4).  What must stay: it
   only rises, 0 at this bridge's start */
static uint64_t passed;
/* Défago et al., Fig. 5, the destinations' "nextdeliver" (2026-11-05, "The revisions of abecd389", SQ2): the number of
   the message this node delivers next, from 1 at the bridge's start ("Initialization: nextdeliver := 1"), one more at
   each message delivered or, for a message whose Dest(m) lacks this node, discarded (§5.2.4: "3. processes not in
   Dest(m) discard m"), never set back; hdr `delivered` is nextdeliver - 1, for the ports (orders_deliver).  Why: A7,
   "Destinations: "Initialization: nextdeliver := 1; pending := ∅"; "when receive (m, seqnum): pending := pending ∪
   {(m, seqnum)}; while ∃(m′, seqnum′) ∈ pending : seqnum′ = nextdeliver do deliver (m′); nextdeliver := nextdeliver +
   1""; the super-group "is the island's nodes.  One sequence numbers every message and every node delivers in its
   order".  What must stay: one more for every message, delivered or discarded.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 3f106747", VW), reversing "never set back": 1
   again, and `pending` emptied, at each view this bridge installs (view_install), after the old view's flush.  Why:
   A7, "Its initializations ... happen at the view's creation at the sequencer and at its installation at each
   member"; hdr `delivered` is `passed`, which only rises, where it was nextdeliver - 1.  What must stay: one more for
   every message of the view, delivered or discarded */
static uint64_t nextdeliver=1;
/* the arrivals this node has heard as the program's sender of a plan (mesh-bridge.h struct mesh_arrival), not yet
   counted into a plan standing here (arrivals_apply), `arrivals_held` of them in room for `arrivals_room`; let go at
   this node's map move (island_set), each node telling its standing plans again on each liveness turn
   (orders_announce).  2026-11-05 ("The revisions of abecd389", PB): new (A7, "A plan's destinations exist before its
   first message") */
static struct mesh_arrival *arrivals;
static size_t arrivals_held,arrivals_room;
/* Defago et al., Fig. 5, the destination's `pending`: the ORDERs this node has received and not delivered, in arrival
   order, which is sequence order (design/compilation-coordination.md, Q3, Q3-tree and DL2), `pendings` of them in room
   for `pending_room`.
   2026-11-04, owner B3 (design/compilation-coordination.md, "The appendix's revisions", O1 and H1;
   design/aot-compilation-change-request.md, appendix items 2 and 3): each entry's arrival time (`arrived`, the loop's
   clock `now`, Q4''s bound: a head held for its plan to stand here held at most MESH_PLATFORM_NS from its arrival) is
   gone.  Why: Defago et al.'s delivery is in sequence-number order (Fig. 5: "while exists (m', seqnum') in pending:
   seqnum' = nextdeliver do deliver (m'); nextdeliver := nextdeliver + 1"), with no timeout and no skip, and the bound
   was a handler for a slow node, which no requirement asks for (item 3).  What must stay: the pending ORDERs in
   arrival order, delivered from the head.
   2026-11-04, owner B4 (design/compilation-coordination.md, "The documents' revisions of edd4b583", DL2 and DL6):
   the head's `first` and `sequence` advance in place as orders_deliver publishes its executions, one at a time
   (`first` + 1, `sequence` + 1 each, so `sequence` stays the number of `first`), where the head was delivered whole
   or not at all.  Why: L. Lamport's Message Buffer sender puts one message at a time ("put message in B[ms mod P]; ms
   := ms + 1"), and may wait between two (Concurrent reading and writing, CACM 20(11), 1977).  What must stay: as
   above */
static struct pending { struct mesh_order order; } *pending;
static size_t pendings,pending_room;
/* 2026-11-04, owner B4 (design/compilation-coordination.md, "The documents' revisions of edd4b583", DL4: "What goes:
   Q4's hold, its test (mesh-records.c orders_served) and `last`"): `last`, the plan this bridge released last (its
   claim head, generation and state section, which orders_served read for Q4's "every port holding the previously
   released plan has served it"), is removed.  Why: no delivery waits on a whole plan's service now (orders_deliver);
   each port merges the published executions of every standing plan (mesh-flow.c).  What must stay: every plan's words
   read on this thread only while the plan stands here (orders_deliver's match) */

/* the port of this node's cable toward adjacent node `node`, MESH_PORTS where none: the first in port order whose link
   is up with a queue pair in its plan, the test by which this node's record lists the cable (mesh-ports.c record_reach,
   its reach entries in port order), so a join of the map between this node and `node` has one.  A tree's hop is one
   control socket, the same for every message of a map serial, so each hop carries its messages in order
   (design/compilation-coordination.md, Q3: "each hop on the existing control socket.  One path from the sequencer to
   each node, each hop in order"; the first port in port order is the recorded decision Q3-tree) */
static uint32_t port_toward(uint32_t node){
  uint32_t p=0;
  while(p<port_count && !(ports[p].link.state==LINK_UP && plan_lanes(ports[p].link.plan) && ports[p].peer==node))p++;
  return p<port_count?p:MESH_PORTS;
}
/* the sequencer's tree as mesh_graph_rows reads it: vertex k keyed by its parent, the root (the sequencer) and a vertex
   not reached (parent UINT32_MAX) by n, past every vertex, as mesh-collective.c plan_tree_parent keys a tree's
   vertices (design/compilation-coordination.md, the C9 rulings) */
struct island_tree { const uint32_t *parent; uint32_t n,root; };
static uint32_t island_parent(const void *context,size_t k){
  const struct island_tree *tree=context;
  return k==tree->root || tree->parent[k]>=tree->n?tree->n:tree->parent[k];
}
/* The island's sequencer (design/aot-compilation-change-request.md A7, "The sequencer is unique and stays where it
   is"): X. Defago, A. Schiper and P. Urban, Total order broadcast and multicast algorithms: taxonomy and survey, ACM
   Comput. Surv. 36(4) 2004, section 7.1: "The sequencer is unique, and the responsibility is not normally transfered
   to another processes"; it is the bridge of the highest-numbered node the map reaches, "the code's choice, which the
   fixed-sequencer class leaves open" (A7): the greatest node id the reach from this node meets over the map's joins
   (mesh_joined, as map_put reads them; mesh-graph.h mesh_graph_reach), the island being "the map's
   Thunderbolt-linked component" (A7), which every bridge holding the same map computes alike; "A plan stands only
   while its map does (design/disjoint-pages-change-request.md D.3.6), so no standing plan's sequencer changes" (A7).
   The tree: the breadth-first tree of the reach from the sequencer (mesh_graph_reach `parent`), ties as the map orders
   nodes (the joins in index order), this node's parent and children in it (Q3).
   2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", SQ and SN), reversing this note's
   first form, which cited H. Garcia-Molina, Elections in a distributed computing system, IEEE Trans. Comput. C-31(1)
   1982 (the bully algorithm's coordinator, "the highest identification number"), and said "On each map serial the
   sequencer's number starts at 0 and the pending ORDERs are dropped (Q3, Q4)".  Why: A7, "(Corrected 2026-11-05: the
   text called this the bully algorithm's coordinator ... No election runs, so that algorithm is not used.)"; and
   appendix item 2, "remove the coordination's Q3′: the sequencer's number never goes back to 0, and no node drops a
   pending order when its map changes" (`seqnum`'s note).  What must stay: the greatest node id reached, picked here;
   no reset of `seqnum` and no drop of a pending ORDER.
   ENOMEM leaves no island, said once a map serial: no function execution is submitted or ordered here on it.
   2026-11-04, owner B2, the coordinator's rulings Q3' and C9 (design/compilation-coordination.md, "Amendments during
   the work"): this node's own map_serial moving is the one reset of `seqnum` and of the pending ORDERs, since D.3.6
   cancels every plan standing here at that move, so no pending order of it is served (no message's `serial` is
   compared: each node's map_serial is its own count); and this node's children are those mesh_graph_rows keys to it
   over the parent array (as mesh-collective.c's plan_tree_parent keys a tree), not a scan of the array.  What must
   stay: the greatest identification number reached, picked here (2026-11-05, SQ: "Garcia-Molina's rule" no longer;
   SN: the reset of `seqnum` gone) */
/* 2026-11-04, owner B2, the coordinator's ruling on review V1's finding 4 (amending Q3'), reversing the note above's
   drop of the pending ORDERs at this node's serial move: they are kept, since delivery is by arrival order, and
   orders_deliver matches a head only to a plan standing here of this node's map serial (a plan of another serial is
   cancelled, D.3.6), the Q4' bound covering a head that never matches.  What must stay: `seqnum` 0 at this node's own
   serial move (2026-11-05, SN: no longer; `seqnum` is never set back).
   2026-11-04, owner B3 (design/compilation-coordination.md, O1 and H1): Q4''s bound is gone, so no bound covers a
   head that never matches: it stays at the head (Defago et al., Fig. 5: delivery in sequence-number order, no timeout
   and no skip; design/aot-compilation-change-request.md, appendix items 2 and 3).  What must stay: as above.
   2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", PB): the arrivals held and not yet
   counted are let go here, at this node's map move, which cancels every plan standing here (D.3.6), and the nodes tell
   their standing plans again on their liveness turns (orders_announce); the pending ORDERs and `nextdeliver` are kept
   (appendix item 2: "no node drops a pending order when its map changes").  What must stay: no reset of `seqnum`, of
   `nextdeliver` or of the pending ORDERs */
static void view_create(const struct mesh_record *r,uint64_t n,const unsigned char *reached);
static void island_set(void){
  arrivals_held=0;
  island=(struct island){.self=UINT32_MAX,.sequencer=UINT32_MAX,.up=MESH_PORTS};
  const uint64_t n=atomic_load(&M->map_nodes);
  struct mesh_record *r=mesh_records(M);
  uint64_t self=n;
  for(uint64_t a=0;a<n;a++)if(r[a].node==M->node)self=a;
  if(self==n)return;
  unsigned char *reached=malloc(n);
  uint32_t *order=malloc(n*sizeof *order),*parent=malloc(n*sizeof *parent),(*join)[2]=malloc((n*(n-1)/2+1)*sizeof *join);
  size_t *first=malloc((n+2)*sizeof *first),*child=malloc(n*sizeof *child);
  if(!reached || !order || !parent || !join || !first || !child){
    free(reached);free(order);free(parent);free(join);free(first);free(child);
    mesh_say("bridge","the island of map serial %llu not computed (ENOMEM): no function execution is ordered on it",
             (unsigned long long)atomic_load(&M->map_serial));
    return;
  }
  const uint32_t joins=map_joins(r,n,join);
  mesh_graph_reach((uint32_t)n,joins,map_end,join,(uint32_t)self,order,reached,NULL);
  uint64_t sequencer=self;
  for(uint64_t k=0;k<n;k++)if(reached[k] && r[k].node>r[sequencer].node)sequencer=k;
  mesh_graph_reach((uint32_t)n,joins,map_end,join,(uint32_t)sequencer,order,reached,parent);
  island.self=(uint32_t)self;island.sequencer=(uint32_t)sequencer;
  if(self!=sequencer){island.parent=r[parent[self]].node;island.up=port_toward(island.parent);}
  const struct island_tree tree={parent,(uint32_t)n,(uint32_t)sequencer};
  mesh_graph_rows((uint32_t)n+1,(size_t)n,island_parent,&tree,first,child);
  for(size_t c=first[self];c<first[self+1] && island.downs<MESH_PORTS;c++){
    const uint32_t p=port_toward(r[child[c]].node);
    if(p<MESH_PORTS)island.down[island.downs++]=p;
  }
  /* 2026-11-05 (design/compilation-coordination.md, "The revisions of 3f106747", VW): at the sequencer, a view of the
     nodes this map reaches, its first message down the tree just made (view_create).  Why:
     design/aot-compilation-change-request.md A7: "The membership service is the sequencer ... A view's members are the
     nodes the sequencer's map reaches.  The sequencer creates a view at its bridge's start and at each move of its
     map": this is the one place both reach (map_put's first write of this bridge's own record moves the map, and every
     later move calls this).  What must stay: the view made after the tree it is sent down.
     2026-11-05 (design/compilation-coordination.md, "The revisions of b5a3c6e6", RV-1): map_put calls this too where a
     member's announcement names another view with a counter not less than the installed view's identifier
     (view_announced), the map unmoved, so the island here is the one computed at its last move and the view's members
     are those nodes.  Why: A7, "It also creates one whenever a member's announcement names a view other than the
     sequencer's installed one"; "A view's members are the nodes the sequencer's map reaches".  What must stay: as
     above */
  if(self==sequencer)view_create(r,n,reached);
  free(reached);free(order);free(parent);free(join);free(first);free(child);
}
/* Defago et al., Fig. 5, the destinations: "when receive (m, seqnum): pending := pending U {(m, seqnum)}".  An ORDER
   whose `nodes` lacks this node's map index is not held, nothing here serving it: the sequencer sends each ORDER to
   every node of the island (design/compilation-coordination.md, Q3 and Q3-tree), and a node publishes only the
   executions of a plan standing here (DL2, DL2′); `nodes` is a set of map indices below 64 (mesh.h struct
   mesh_state).  The array grows by doubling; ENOMEM drops the ORDER, said: its plan's executions are not released
   here, and the head waits at the plan's next ORDER, whose `first` is past its `released` (orders_deliver; DL2′; open,
   "Delivery's liveness").
   2026-11-04, owner B3 (design/compilation-coordination.md, O1 and H1): its arrival time, the loop's clock `now`, is
   no longer kept (struct pending's note).  What must stay: the ORDER appended in arrival order.
   2026-11-05, owner B4 (the coordinator's rulings on reviews X1 #2, #6 and X3 #4: DL2′, "Amendments during the work
   (2026-11-05)"): the note's "delivered at once" (Q4's words) restated with Q3, Q3-tree, DL2 and DL2′, and its drop's
   end true again: under DL2's greater of `first` and `released`, a later ORDER of the plan published past the dropped
   executions' never-written words; under DL2′ the head stays there.  What must stay: as above.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", CH): a dropped ORDER's executions never
   enter the deliveries' channel here, and a lane that meets the plan's next delivery takes nothing past the gap
   (mesh-flow.c lane_take), where the parent's head stayed at it.  What must stay: as above.
   2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", SQ2): every ORDER is held, this node's
   or not, where one whose `nodes` lacked this node's map index was not: delivery compares the sequence numbers
   (orders_deliver), so this node's nextdeliver passes every message's number, discarding those not its own.  Why:
   design/aot-compilation-change-request.md A7, "Multicast to many groups is their §5.2.4, as written: "1. form a
   super-group with the union of all destination groups; 2. whenever a message m is multicast to a group, multicast it
   to the super-group, and 3. processes not in Dest(m) discard m."  The super-group is the island's nodes."  What must
   stay: the ORDER appended in arrival order */
static void order_take(const struct mesh_order *order){
  /* 2026-11-05 (design/compilation-coordination.md, "The revisions of 3f106747", VW): held only where its view is the
     one installed here.  Why: design/aot-compilation-change-request.md A7: "A member holds and delivers only the
     messages of the view it has installed (Sending View Delivery)", "(Sending View Delivery) If a process p receives
     message m in view V, and some process q (possibly p = q) sends m in view V′, then V = V′" (Chockler, Keidar and
     Vitenberg, Property 4.3).  What must stay: the ORDER appended in arrival order */
  if(island.self==UINT32_MAX || order->view!=installed.view)return;
  if(pendings==pending_room){
    const size_t room=2*pending_room+1;
    struct pending *grown=realloc(pending,room*sizeof *grown);
    if(!grown){
      mesh_say("bridge","an ORDER of plan (%llu, %llu) not held (ENOMEM): its executions [%llu, %llu) are not released here",
               (unsigned long long)order->group,(unsigned long long)order->epoch,(unsigned long long)order->first,
               (unsigned long long)order->end);
      return;
    }
    pending=grown;pending_room=room;
  }
  pending[pendings++]=(struct pending){*order};
}
/* One message down the island's tree, to each child's port on its hop's control socket, in arrival order
   (design/compilation-coordination.md, Q3: "an ORDER down to the children, each hop on the existing control socket");
   a child's port not greeted, whose message port_say would drop, said on the bridge's log (the coordinator's ruling on
   review V1's finding 7).  2026-11-05 (design/compilation-coordination.md, "The revisions of 3f106747", VW and BS):
   order_relay's loop made one function, which an ORDER, an ARRIVE, a VIEW and a COUNT going down take, where
   order_relay and arrival_down each held a copy.  Why: the operator, "inlined code is unilaterally bad and to be
   deduplicated and uninlined where possible".  What must stay: every child's port, in order */
static void tree_down(uint32_t kind,const void *payload,uint32_t bytes){
  for(uint32_t k=0;k<island.downs;k++){
    struct port *down=ports+island.down[k];
    if(!down->greeted)
      mesh_say("bridge","link %u (%s): a message of kind %u not relayed: the port toward a child of the island's tree is "
               "not greeted",down->index,down->name,kind);
    else port_say(down,(struct message){.kind=kind},payload,bytes);
  }
}
/* One message toward the node whose id is `to`, over the island's tree, as an ARRIVE went (design/compilation-
   coordination.md, "The revisions of abecd389", PB-1: up the tree as a SUBMIT goes, to `to` where the path meets it,
   else to the sequencer, and from there down the tree as an ORDER goes): 2 where `to` is this node (its caller takes
   it); 1 where it went down from the sequencer (tree_down: `to` among the nodes below takes it) or up on the port toward
   the parent, that port greeted before the queueing and still after it (order_submit's test); else 0, nothing sent.
   2026-11-05 ("The revisions of 3f106747", BS): arrival_up's route made one function, which an ARRIVE and a COUNT
   take.  Why: design/aot-compilation-change-request.md, appendix item 2: "The bridge carries the count from source to
   sink, as it carries an arrival"; one copy of the route.  What must stay: a SUBMIT's route up and an ORDER's down */
static int tree_toward(uint32_t kind,const void *payload,uint32_t bytes,uint32_t to){
  if(to==M->node)return 2;
  if(island.self==UINT32_MAX)return 0;
  if(island.self==island.sequencer){tree_down(kind,payload,bytes);return 1;}
  if(island.up>=MESH_PORTS || !ports[island.up].greeted)return 0;
  port_say(ports+island.up,(struct message){.kind=kind},payload,bytes);
  return ports[island.up].greeted;
}
/* Defago et al., Fig. 5, the sequencer's "send (m, sn(m)) to all", over the island's tree: an ORDER to each child, in
   arrival order, on its hop's control socket, and taken here (design/compilation-coordination.md, Q3: "an ORDER down
   to the children, each hop on the existing control socket") */
/* 2026-11-04, owner B2, the coordinator's ruling on review V1's finding 7: a child's port not greeted, whose ORDER
   port_say would drop, is said on the bridge's log (that child's ORDER is lost; its plan's head there ends at Q4''s
   bound).  What must stay: the ORDER relayed to every child before it is taken here.
   2026-11-04, owner B3 (design/compilation-coordination.md, O1 and H1): Q4''s bound is gone, so the note above's "ends
   at Q4''s bound" no longer holds: the lost ORDER never reaches that child's pending list, its executions are not
   released there, and a wait on them ends at its deadline (C3's bounded waits), as order_take's ENOMEM says.  What
   must stay: as above.
   2026-11-05, owner B4 (DL2′): true again, where DL2's greater of `first` and `released` let a later ORDER of the
   plan publish past the lost one's executions there: that child's head stays at the plan's next ORDER, whose `first`
   is past its `released` (orders_deliver), said on its log (open, "Delivery's liveness").  What must stay: as above.
   2026-11-05 (CH): that child delivers the plan's next ORDER, and its lanes take nothing past the gap (mesh-flow.c
   lane_take).  What must stay: as above.
   2026-11-05 ("The revisions of 3f106747", VW): the relay is tree_down's.  What must stay: as above */
static void order_relay(const struct mesh_order *order){
  tree_down(ORDER,order,sizeof *order);
  order_take(order);
}
/* Defago et al., Fig. 5, the sequencer: "when receive (m): sn(m) := seqnum; send (m, sn(m)) to all; seqnum := seqnum +
   1".  A submission of end - first executions reaches it at once, so they take end - first consecutive numbers, first
   in, first out (Pathways, section 4.4; design/compilation-coordination.md, Q3).
   2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", SQ1): one number a submission, sn(m)
   := seqnum, then seqnum + 1, where a submission took one number an execution.  Why: A7, "m is one submission ... One
   submission is one message however many executions it carries, as Pathways sends "a single message (describing the
   entire subgraph)" (§4.5)"; Fig. 5 "as written".  What must stay: first in, first out.
   2026-11-05 ("The revisions of 3f106747", VW): numbered only where it was sent in the view this bridge created and has
   installed, its ORDER carrying that view; any other discarded, said on the bridge's log.  Why: A7: "When the
   sequencer's map moves, it numbers no further message of the old view"; Sending View Delivery (Property 4.3), the
   message received in the view it was sent in.  What must stay: first in, first out */
static void order_sequence(const struct mesh_order *submission){
  if(!installed.view || submission->view!=installed.view || installed.sequencer!=M->node || installed.boot!=M->boot){
    mesh_say("bridge","a SUBMIT of plan (%llu, %llu), executions [%llu, %llu), discarded: sent in view %llu, this sequencer's "
             "view %llu",(unsigned long long)submission->group,(unsigned long long)submission->epoch,
             (unsigned long long)submission->first,(unsigned long long)submission->end,
             (unsigned long long)submission->view,(unsigned long long)installed.view);
    return;
  }
  struct mesh_order order=*submission;
  order.sequence=seqnum++;
  order_relay(&order);
}
/* The barrier's arrivals (design/aot-compilation-change-request.md A7, "A plan's destinations exist before its first
   message": "The program's sender counts the plan's nodes as each holds the plan, as a centralized barrier counts
   arrivals: "each processor updates a small amount of shared state to indicate its arrival and then polls that state
   to determine when all of the processors have arrived" (J. M. Mellor-Crummey and M. L. Scott, Algorithms for scalable
   synchronization on shared-memory multiprocessors, ACM TOCS 9(1), 1991, §3.1).  It TO-broadcasts no submission of the
   plan before every node of the plan has arrived"; design/compilation-coordination.md, "The revisions of abecd389",
   PB).  The shared state is the sender's plan's `arrived` (mesh.h struct mesh_state), which its bridge parent alone
   writes, so an arrival travels to it as a message: up the island's tree, as a SUBMIT goes (Q3), to the sender where
   the path meets it, else to the sequencer, and from there down the tree to every node, as an ORDER goes, kept at the
   node it names as the sender.  arrival_take holds it there; arrivals_apply counts it into the plan once it stands.
   2026-11-05 ("The revisions of 3f106747", BS): its route is tree_toward's and tree_down's, where arrival_up and
   arrival_down held it.  What must stay: the route as above */
static void arrival_take(const struct mesh_arrival *a){
  if(arrivals_held==arrivals_room){
    const size_t room=2*arrivals_room+1;
    struct mesh_arrival *grown=realloc(arrivals,room*sizeof *grown);
    if(!grown){
      mesh_say("bridge","an arrival at plan (%llu, %llu) of node %u not held (ENOMEM): it is told again on its next liveness "
               "turn",(unsigned long long)a->group,(unsigned long long)a->epoch,a->node);
      return;
    }
    arrivals=grown;arrivals_room=room;
  }
  arrivals[arrivals_held++]=*a;
}
/* Defago et al., Fig. 5, the sender's "send (m) to sequencer", over the island's tree: at the sequencer it is numbered
   here, elsewhere it goes to this node's parent (design/compilation-coordination.md, Q3: "a SUBMIT up to the parent").
   A submission of no executions is dropped (design/compilation-coordination.md, Q3-tree), as is any where the map holds
   no record of this node.
   2026-11-04, owner B2, the coordinator's ruling Q3' (design/compilation-coordination.md, "Amendments during the
   work"), reversing Q3's drop of a submission whose `serial` is not this node's `map_serial`: no receiver compares a
   message's `serial`.  Why: each node's map_serial is its own count of its map's changes, so two nodes on one map hold
   different serials and the drop refused messages between them.  What must stay: the reset at this node's own serial
   move (island_set).
   2026-11-04, owner B2, the coordinator's ruling on review V1's finding 7: 1 where the submission was numbered here or
   queued on the port toward the parent, that port greeted before the queueing and still after it (port_say loses the
   link on a failed send, port_lost), else 0, so the caller knows it was not sent.  What must stay: nothing sent
   twice that was sent once */
static int order_submit(const struct mesh_order *order){
  if(order->end<=order->first || island.self==UINT32_MAX)return 0;
  if(island.self==island.sequencer){order_sequence(order);return 1;}
  if(island.up>=MESH_PORTS || !ports[island.up].greeted)return 0;
  port_say(ports+island.up,(struct message){.kind=SUBMIT},order,sizeof *order);
  return ports[island.up].greeted;
}
/* The island's views (design/aot-compilation-change-request.md A7, "The island is a dynamic group, and each of its
   views is a static system"; G. V. Chockler, I. Keidar and R. Vitenberg, Group communication specifications: a
   comprehensive study, ACM Comput. Surv. 33(4), 2001; X. Défago, A. Schiper and P. Urbán, ACM Comput. Surv. 36(4),
   2004, §6.1 and §8.2; design/compilation-coordination.md, "The revisions of 3f106747", VW).  The membership service
   is the sequencer, which decides each view alone: "Suspected processes are excluded from the group as the result of
   the unilateral decision of a single process" (Défago et al., §9.1.1, Amoeba).  The view properties the code keeps,
   as A7 quotes them:
   - "(Self Inclusion) If process p installs view V, then p is a member of V." (Property 3.1): view_install installs
     only a view whose members name this node;
   - "(Local Monotonicity) If a process p installs view V after installing view V′ then the identifier of V is greater
     than that of V′." (Property 3.2): view_install installs only a view whose identifier passes this bridge's counter,
     and view_create gives a view one more than the greatest counter among its members (Transis's identifier, §3.1.1),
     the counters as the members' records carry them and this bridge's own;
   - "(Initial View Event) Every send, recv and safe_prefix event occurs within some view." (Property 3.3), "so that no
     send events occur before the first view_chng event": no SUBMIT, ARRIVE or COUNT is sent here before this bridge's
     first view (orders_submit, orders_announce, counts_tell);
   - "(Sending View Delivery) If a process p receives message m in view V, and some process q (possibly p = q) sends m
     in view V′, then V = V′." (Property 4.3): every message carries its view's identifier, the sequencer numbers only
     its own view's SUBMITs (order_sequence), a member holds only its installed view's ORDERs (order_take), and the
     sender counts only its installed view's arrivals (arrivals_apply), a sink only its view's counts (count_take). */
/* Defago et al., Fig. 5's destinations' delivery, to message number `last` at most (UINT64_MAX: every one): below */
static void orders_deliver(uint64_t last);
/* A view installed here, `member` its v->members node ids (A7: "A bridge installs a view when the view's first message
   reaches it, if the view names it a member (Self Inclusion) and the view's identifier is greater than its counter
   (Local Monotonicity).  It ignores any other view's messages").  The change blocks and flushes (A7, Chockler, Keidar
   and Vitenberg, §4.2.1: "group communication systems block sending of messages while a view change is taking place
   ... The application responds with a flush message which follows all the messages sent by the application in the old
   view"): "A member that installed the old view from the same process delivers the old view's messages up to that
   number, then installs the new view", the process the view's creator and its start (`sequencer`, `boot`), the number
   the VIEW's `last`; "Where the old view's sequencer is gone (its bridge stopped, or the map moved to another
   sequencer), the old view's undelivered messages are never delivered".  Then Fig. 5's initializations, "nextdeliver :=
   1; pending := ∅" ("at its installation at each member"), the arrivals held let go (of the old view: A7, the sender
   "counts only the arrivals of the view it has installed"), this bridge's counter the view's identifier and its record
   told again (record_changed: "Each bridge's announcement of itself to the sequencer carries its counter"), and every
   plan of the old view ended with ESTALE: hdr `map_serial` moved, while `map` is odd, and plan_stale (A7: "Installing a
   view ends every plan of the old one with `ESTALE`, as D.3.6 ends a plan at its map's move, and the plan's replan
   arrives in the new view").  A plan made in this view has the moved serial, so a plan stands only within the view it
   was made, its nodes' arrivals counted, in.  Decisions (VW-1, VW-2, the record's): the flush delivers the old view's
   messages this node holds, in Fig. 5's order, while each is the next and at most `last`, a message not yet arrived
   never delivered, since nothing bounds when one would; and installation moves the serial D.3.6 reads, so every
   reader of a plan's serial (the ports' loads and cancellations, the start, mesh_plan's second read) ends the old
   view's plans by its one path.  On the bridge parent's one event-loop thread, as every caller of plan_stale is */
static void view_install(const struct mesh_view *v,const uint32_t *member){
  int named=0;
  for(uint32_t i=0;i<v->members;i++)named|=member[i]==M->node;
  if(!named || v->view<=installed.view)return;
  if(installed.view && installed.sequencer==v->sequencer && installed.boot==v->boot)orders_deliver(v->last);
  pendings=0;nextdeliver=1;arrivals_held=0;
  installed=(struct installed){.view=v->view,.boot=v->boot,.sequencer=v->sequencer};
  mesh_notify(&M->map);
  atomic_fetch_add(&M->map_serial,1);
  mesh_notify(&M->map);
  plan_stale();
  record_changed();
}
/* The sequencer's view of the nodes its map reaches (`reached` over the map's records r[0, n), island_set's), at its
   bridge's start and at each move of its map (A7): its identifier one more than the greatest counter among its members
   (A7: "the sequencer gives a view one more than the greatest counter among its members.  So Local Monotonicity holds
   at every member, a bridge started again among them"), this bridge's own counter its installed view's; `last` the
   last sequence number this process gave in the view it created before, seqnum - 1, 0 where the view installed here is
   not this process's ("When the sequencer's map moves, it numbers no further message of the old view.  Its first
   message of the new view carries the last sequence number it gave in the old one"); then Fig. 5's "Initialization:
   seqnum := 1" ("at the view's creation at the sequencer"); its first message, the VIEW, down the tree (tree_down), and
   installed here, the sequencer a member of its own view.  ENOMEM creates no view, said: the old view stands here */
static void view_create(const struct mesh_record *r,uint64_t n,const unsigned char *reached){
  uint64_t greatest=installed.view;
  uint32_t members=0;
  for(uint64_t k=0;k<n;k++)if(reached[k]){members++;if(r[k].node!=M->node && r[k].view>greatest)greatest=r[k].view;}
  const size_t bytes=sizeof(struct mesh_view)+(size_t)members*sizeof(uint32_t);
  unsigned char *message=malloc(bytes);
  if(!message){mesh_say("bridge","no view of map serial %llu created (ENOMEM)",(unsigned long long)atomic_load(&M->map_serial));return;}
  const struct mesh_view v={.view=greatest+1,.last=installed.sequencer==M->node && installed.boot==M->boot?seqnum-1:0,
                            .boot=M->boot,.sequencer=M->node,.members=members};
  memcpy(message,&v,sizeof v);
  uint32_t *member=(uint32_t *)(void *)(message+sizeof v);
  for(uint64_t k=0,i=0;k<n;k++)if(reached[k])member[i++]=r[k].node;
  seqnum=1;
  tree_down(VIEW,message,(uint32_t)bytes);
  view_install(&v,member);
  free(message);
}
/* Whether `record`, which map_put has just written into the map, has this bridge, the island's sequencer, create a
   view (map_put then calls island_set, and island_set view_create): A7, "The membership service is the sequencer":
   "It also creates one whenever a member's announcement names a view other than the sequencer's installed one, with a
   counter not less than that view's identifier.  Such a member installed another sequencer's view after the
   announcement the sequencer counted, and it ignores the sequencer's view by Local Monotonicity ... To tell such a
   member from one that installed the sequencer's own view, each announcement names the process that created its
   bridge's installed view as well as its counter" (design/aot-compilation-change-request.md at mesh b5a3c6e6;
   design/compilation-coordination.md, "The revisions of b5a3c6e6", RV).  The view a record names is its counter and
   its creator (mesh-map.h struct mesh_record `view`, `view_sequencer`, `view_boot`), as struct installed names the
   view installed here; a member that installed the sequencer's own view names it whole, its counter equal to the
   identifier, and creates none, nor does the sequencer's own record (record_changed writes it from `installed`).
   What must stay: 0 at every node but the sequencer */
static int view_announced(const struct mesh_record *record){
  return island.self!=UINT32_MAX && island.self==island.sequencer && record->view>=installed.view &&
    (record->view!=installed.view || record->view_sequencer!=installed.sequencer || record->view_boot!=installed.boot);
}
/* A VIEW heard from this node's parent in the island's tree (order_heard): relayed down, as an ORDER is, then
   installed here where it may be (view_install).  A payload that is not one struct mesh_view and its members is let
   be */
static void view_heard(const struct message *heard,const unsigned char *payload){
  struct mesh_view v;
  if(heard->bytes<sizeof v)return;
  memcpy(&v,payload,sizeof v);
  if(v.members>MESH_MAP_ROOM || heard->bytes!=sizeof v+(uint64_t)v.members*sizeof(uint32_t))return;
  uint32_t member[MESH_MAP_ROOM];
  memcpy(member,payload+sizeof v,(size_t)v.members*sizeof *member);
  tree_down(VIEW,payload,heard->bytes);
  view_install(&v,member);
}
/* a synchronization edge's count taken at its sink's node (below) */
static void count_take(const struct mesh_count *c);
/* the sequencer's messages heard on a port's control socket (mesh-ports.c message_heard; design/compilation-
   coordination.md, Q3): a SUBMIT is passed on toward the sequencer, or numbered there (order_submit); an ORDER from
   this node's parent in the tree is relayed to its children and taken here (order_relay), and one from any other
   node is dropped (design/compilation-coordination.md, Q3-tree).  A payload that is not one struct mesh_order is let
   be.  2026-11-04, owner B2, the coordinator's ruling Q3' (order_submit's note): an ORDER's `serial` is compared no
   more.  2026-11-04, the ruling on review V1's finding 7: a SUBMIT heard and not passed on (order_submit 0) is said on
   the bridge's log.
   2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", PB): an ARRIVE from this node's parent
   goes down (arrival_down), any other up (arrival_up), a payload not one struct mesh_arrival let be.  What must stay:
   as above.
   2026-11-05 ("The revisions of 3f106747", VW and BS): a VIEW from this node's parent is view_heard's, from any other
   node dropped, as an ORDER is; an ARRIVE or a COUNT from the parent goes down (tree_down) and is taken here where it
   names this node, from any other goes toward the node it names (tree_toward), taken here where that is this node
   (arrival_take, count_take).  What must stay: as above */
void order_heard(struct port *from,const struct message *heard,const unsigned char *payload){
  const int down=island.parent && from->peer==island.parent;
  if(heard->kind==VIEW){
    if(down)view_heard(heard,payload);
    return;
  }
  if(heard->kind==ARRIVE || heard->kind==COUNT){
    union { struct mesh_arrival a; struct mesh_count c; } m;
    const uint32_t bytes=heard->kind==ARRIVE?sizeof m.a:sizeof m.c;
    if(heard->bytes!=bytes)return;
    memcpy(&m,payload,bytes);
    const uint32_t to=heard->kind==ARRIVE?m.a.sender:m.c.sink;
    int routed=2*(to==M->node);
    if(down)tree_down(heard->kind,&m,bytes);
    else routed=tree_toward(heard->kind,&m,bytes,to);
    if(routed==2 && heard->kind==ARRIVE)arrival_take(&m.a);
    else if(routed==2)count_take(&m.c);
    else if(!routed && !down)
      mesh_say("bridge","link %u (%s): a message of kind %u toward node %u not passed on",from->index,from->name,
               heard->kind,to);
    return;
  }
  struct mesh_order order;
  if(heard->bytes!=sizeof order)return;
  memcpy(&order,payload,sizeof order);
  if(heard->kind==SUBMIT){
    if(!order_submit(&order))
      mesh_say("bridge","link %u (%s): a SUBMIT of plan (%llu, %llu), executions [%llu, %llu), not passed on toward the "
               "island's sequencer",from->index,from->name,(unsigned long long)order.group,(unsigned long long)order.epoch,
               (unsigned long long)order.first,(unsigned long long)order.end);
  }
  else if(down)order_relay(&order);
}
/* design/compilation-coordination.md, Q2 (Defago et al., section 7.1: the sequencer numbers each message its sender
   sends once, so one bridge submits each plan's executions, the one at the plan's `sender`): for each plan standing
   here (mesh-dataflow.h mesh_head_state) whose `sender` is this node's map index and whose `executions` (acquire, the
   program's start, Q1) exceed its `submitted`, a SUBMIT of [submitted, executions), then `submitted` its executions
   (this bridge its one writer).  2026-11-04, owner B2, the coordinator's ruling Q3': the SUBMIT's `serial` is this
   node's map_serial, for the record (no receiver compares it: order_submit's note), where it was the plan's.
   2026-11-04, the coordinator's ruling on review V1's finding 7, reversing the first form, which raised `submitted`
   whether or not the SUBMIT left: `submitted` is raised only where order_submit numbered it here or queued it toward
   the parent; otherwise it stays, and the next turn sends the same submission (no transfer failed: none was sent),
   said on the bridge's log once until a submission is sent (`unsent`).  What must stay: `submitted` the executions
   sent, never more */
/* 2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", PS and PB), reversing the turn above
   (each plan whose `sender` was this node submitted [submitted, executions) as the heads' walk met it): this node TO-
   broadcasts the submissions of each program it is the sender of (struct mesh_state `sender`, the least of the
   program's nodes) in the order the program made them, one message a submission, each only once its plan's barrier
   holds every node of the plan:
   - the program's submissions are its plans' submissions' queues (mesh.h struct mesh_submission: each start's end and
     the program's Lamport clock), each plan's in its own order, so the program's order is a multiway merge of them on
     the clock (D. E. Knuth, The Art of Computer Programming vol. 3, §5.4.1: of the first untaken entries, the least
     first), as appendix item 2 merges a node's plans' delivered calls on the sequence number;
   - the queues are read in Lamport's opposite order (L. Lamport, Concurrent reading and writing, CACM 20(11), 1977:
     "writing data elements in one order and reading them in the opposite order"): every plan's `submissions` read
     (acquire), then each again with its first untaken entry; the program wrote its entries in clock order, each
     before its `submissions` (release), so an entry acquired in the first reading has every earlier one of the program
     visible to the second, and where the first reading acquired one of the program the least of the second is the
     program's next; where it acquired none, nothing of the program is taken this turn;
   - that entry's plan's barrier (`arrived` holding `nodes`, arrivals_apply) not complete, the program's later
     submissions wait behind it, the head staying for a later turn (A7: "It TO-broadcasts no submission of the plan
     before every node of the plan has arrived"; FIFO Order, Défago, Schiper and Urbán §4.3.1, over the program's one
     sender), nobody blocking;
   - else the message m = (the plan's name, this node's map serial, [`submitted`, the entry's `end`), `nodes`, `sender`)
     is sent toward the sequencer (order_submit: "procedure TO-broadcast(m): send (m) to sequencer"), then `submitted`
     the entry's end and `broadcast` one more (this bridge their one writer); one not sent stays, said once until a
     message is sent (`unsent`).
   Only plans standing here on this node's map serial are read (a cancelled plan's submissions go nowhere: D.3.6).
   Why: design/aot-compilation-change-request.md A7: "sender(m) is the program's one sender ... It TO-broadcasts the
   program's submissions in the order the program makes them"; "With one sender, whose messages reach the sequencer
   first in, first out, Fig. 5's sequencer numbers them in the order they were sent ... So FIFO Order holds, and with
   it A10's order constraints between the program's calls"; "A plan's destinations exist before its first message".
   What must stay: `submitted` the executions sent, never more; nothing sent twice that was sent once */
static int unsent;
/* the plan standing here named (group, epoch) on this node's map serial, NULL where none: the claims' heads walked
   (mesh-dataflow.h mesh_head_after, mesh_head_state), at most one such plan standing (mesh_plan's EEXIST) */
static struct mesh_state *plan_here(uint64_t group,uint64_t epoch){
  const uint64_t serial=atomic_load_explicit(&M->map_serial,memory_order_relaxed);
  for(uint32_t head=mesh_head_after(M,UINT32_MAX);head!=UINT32_MAX;head=mesh_head_after(M,head)){
    struct mesh_state *state=mesh_head_state(M,head);
    if(state && state->group==group && state->epoch==epoch && state->serial==serial)return state;
  }
  return NULL;
}
/* whether a plan's queues lie inside the region, as plan_inside (mesh-flow.c) judges its entries: a store outside it
   would be a fault, not an error value */
/* 2026-11-05 ("The revisions of 3f106747", BS): the submissions' queue's room `submit_room`, where it was
   `outstanding` (mesh.h struct mesh_state's note).  What must stay: as above */
static int queues_inside(const struct mesh_state *s){
  return s->room && s->submit_room && s->deliveries<=M->length &&
    s->room<=(M->length-s->deliveries)/sizeof(struct mesh_delivery) && s->submits<=M->length &&
    s->submit_room<=(M->length-s->submits)/sizeof(struct mesh_submission);
}
/* this node's map index of node id `node` (its place in ascending node-id order among the map's records, mesh.h
   `map_serial`), UINT32_MAX where the map holds none */
static uint32_t map_index_of(uint32_t node){
  const uint64_t n=atomic_load(&M->map_nodes);
  const struct mesh_record *r=mesh_records(M);
  for(uint64_t k=0;k<n;k++)if(r[k].node==node)return (uint32_t)k;
  return UINT32_MAX;
}
/* each arrival held counted into its plan where that plan stands here with this node its sender, its node's map index
   set in `arrived` (A7: "each processor updates a small amount of shared state to indicate its arrival"), and let go;
   one whose plan does not stand here yet is held */
/* 2026-11-05 ("The revisions of 3f106747", VW): an arrival of a view other than the one installed here let go, not
   counted.  Why: A7, "Each arrival carries its view's identifier, and the program's sender counts only the arrivals of
   the view it has installed"; one of a later view is told again on its node's liveness turns.  What must stay: as
   above */
static void arrivals_apply(void){
  for(size_t k=0;k<arrivals_held;){
    struct mesh_state *s=plan_here(arrivals[k].group,arrivals[k].epoch);
    const uint32_t at=map_index_of(arrivals[k].node);
    if(arrivals[k].view!=installed.view){arrivals[k]=arrivals[--arrivals_held];continue;}
    if(!s || s->sender!=island.self){k++;continue;}
    if(at<64)s->arrived|=UINT64_C(1)<<at;
    arrivals[k]=arrivals[--arrivals_held];
  }
}
/* this node's arrival at each plan standing here on its map serial, told to the plan's sender (arrival_up) once the
   plan stands, `announced` 1 once it went, and again on every liveness turn, as record_retell tells this node's record
   again (the tag's: "told again at its current sequence on every greeted port, each liveness turn"), since the sender
   lets go of the arrivals it holds at its map move (island_set) and an arrival lost on the way is told again.
   2026-11-05 ("The revisions of 3f106747", VW and BS): none before this bridge's first view, and each carrying the
   view installed here, its route tree_toward's.  Why: A7, Initial View Event (Property 3.3: "no send events occur
   before the first view_chng event"); "Each arrival carries its view's identifier".  A plan standing on this node's
   map serial was made in that view, since installing one moves the serial (view_install).  What must stay: as above */
static void orders_announce(int liveness){
  if(!installed.view)return;
  const uint64_t n=atomic_load(&M->map_nodes),serial=atomic_load_explicit(&M->map_serial,memory_order_relaxed);
  const struct mesh_record *r=mesh_records(M);
  for(uint32_t head=mesh_head_after(M,UINT32_MAX);head!=UINT32_MAX;head=mesh_head_after(M,head)){
    struct mesh_state *state=mesh_head_state(M,head);
    if(!state || state->serial!=serial || (state->announced && !liveness) || state->sender>=n)continue;
    const struct mesh_arrival a={.group=state->group,.epoch=state->epoch,.node=M->node,.sender=r[state->sender].node,
                                 .view=installed.view};
    const int routed=tree_toward(ARRIVE,&a,sizeof a,a.sender);
    if(routed==2)arrival_take(&a);
    if(routed)state->announced=1;
  }
}
/* A synchronization edge's count, carried from its source's node to its sink's (design/aot-compilation-change-
   request.md, appendix item 2: "An added edge (vj, vi) holds "start(vi, k) ≥ end(vj, k − delay((vj, vi)))" (BSL (7)).
   Its source raises a count that only rises (C3's counted completion: Portals 4 counting events), and its sink reads
   that count before its k-th start.  The bridge carries the count from source to sink, as it carries an arrival";
   design/compilation-coordination.md, "The revisions of 3f106747", BS).  At the sink, the count of the plan standing
   here that names the COUNT's node as its edge's source (mesh.h struct mesh_state `waits`), of the view installed here,
   stored in its `wait_count` where it rises (release), its waiter woken (os_sync_wake_by_address_all, region-shared,
   as mesh-call.h mesh_transfers_start waits on it); any other let be (a plan not standing here yet hears it again on its
   source's liveness turn) */
static void count_take(const struct mesh_count *c){
  struct mesh_state *s=plan_here(c->group,c->epoch);
  const uint32_t from=map_index_of(c->node);
  if(!s || c->view!=installed.view || from==UINT32_MAX || s->waits!=from ||
     c->count<=atomic_load_explicit(&s->wait_count,memory_order_relaxed))return;
  atomic_store_explicit(&s->wait_count,c->count,memory_order_release);
  os_sync_wake_by_address_all(&s->wait_count,sizeof s->wait_count,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
/* At the source: each plan standing here on this node's map serial with an edge out of this node (`tells`), its count
   told toward its sink's node (tree_toward, as orders_announce tells an arrival) where it rose past `told`, and again
   on every liveness turn, as an arrival is ("told again on every liveness turn"), `told` the count once it went (this
   bridge its one writer).  The count is the plan's `submissions` here (acquire), its program's starts: the source's
   ends trail them by A10's bound, the delay on its own edge from a call's completion to the program's later
   submission, which mesh_plan adds to the edge's delay in the queues' rooms (design/compilation-coordination.md, "The
   revisions of 3f106747", BS-4).  None before this bridge's first view (Initial View Event) */
static void counts_tell(int liveness){
  if(!installed.view)return;
  const uint64_t n=atomic_load(&M->map_nodes),serial=atomic_load_explicit(&M->map_serial,memory_order_relaxed);
  const struct mesh_record *r=mesh_records(M);
  for(uint32_t head=mesh_head_after(M,UINT32_MAX);head!=UINT32_MAX;head=mesh_head_after(M,head)){
    struct mesh_state *state=mesh_head_state(M,head);
    if(!state || state->serial!=serial || state->tells>=n)continue;
    const uint64_t count=atomic_load_explicit(&state->submissions,memory_order_acquire);
    if(count<=state->told && !(liveness && count))continue;
    const struct mesh_count c={.group=state->group,.epoch=state->epoch,.view=installed.view,.count=count,
                               .node=M->node,.sink=r[state->tells].node};
    const int routed=tree_toward(COUNT,&c,sizeof c,c.sink);
    if(routed==2)count_take(&c);
    if(routed)state->told=count;
  }
}
/* 2026-11-05 ("The revisions of 3f106747", VW and BS): nothing before this bridge's first view, each message carrying
   the view installed here where it carried this node's map serial, and the queues' room `submit_room`.  Why: A7, "the
   program's sender TO-broadcasts nothing before its first view (Initial View Event)"; "Every message carries its view's
   identifier, which replaces the map serial"; a plan standing on this node's serial was made in that view
   (view_install).  What must stay: as above */
struct program_head { struct mesh_state *state; uint64_t seen; };
static void orders_submit(void){
  if(!installed.view)return;
  const uint64_t serial=atomic_load_explicit(&M->map_serial,memory_order_relaxed);
  for(int sent=1;sent;){
    sent=0;
    struct program_head *heads=NULL;
    size_t count=0,room=0;
    for(uint32_t head=mesh_head_after(M,UINT32_MAX);head!=UINT32_MAX;head=mesh_head_after(M,head)){
      struct mesh_state *state=mesh_head_state(M,head);
      if(!state || state->sender!=island.self || state->serial!=serial || !queues_inside(state))continue;
      if(count==room){
        struct program_head *grown=realloc(heads,(2*room+1)*sizeof *grown);
        if(!grown){free(heads);mesh_say("bridge","the programs' submissions not read (ENOMEM): read on a later turn");return;}
        heads=grown;room=2*room+1;
      }
      heads[count++]=(struct program_head){state,atomic_load_explicit(&state->submissions,memory_order_acquire)};
    }
    for(size_t i=0;i<count;i++){
      if(!heads[i].state)continue;
      const uint64_t group=heads[i].state->group;
      int acquired=0;
      struct mesh_state *least=NULL;
      struct mesh_submission next={0};
      for(size_t j=i;j<count;j++){
        struct mesh_state *s=heads[j].state;
        if(!s || s->group!=group)continue;
        heads[j].state=NULL;
        acquired|=heads[j].seen>s->broadcast;
        if(atomic_load_explicit(&s->submissions,memory_order_acquire)<=s->broadcast)continue;
        const struct mesh_submission e=((const struct mesh_submission *)(const void *)((const char *)M+s->submits))
          [s->broadcast%s->submit_room];
        if(!least || e.clock<next.clock){least=s;next=e;}
      }
      if(!acquired || !least || (least->arrived&least->nodes)!=least->nodes)continue;
      if(next.end>least->submitted){
        const struct mesh_order order={.group=least->group,.epoch=least->epoch,.view=installed.view,.first=least->submitted,
                                       .end=next.end,.nodes=least->nodes,.sender=least->sender};
        if(!order_submit(&order)){
          if(!unsent)
            mesh_say("bridge","a SUBMIT of plan (%llu, %llu), executions [%llu, %llu), not sent: no greeted port toward the "
                     "island's sequencer (sent on a later turn)",(unsigned long long)order.group,
                     (unsigned long long)order.epoch,(unsigned long long)order.first,(unsigned long long)order.end);
          unsent=1;
          continue;
        }
        least->submitted=next.end;unsent=0;
      }
      least->broadcast++;sent=1;
    }
    free(heads);
  }
}
/* 2026-11-04, owner B4 (design/compilation-coordination.md, "The documents' revisions of edd4b583", DL2 and DL4:
   "What goes: Q4's hold, its test (mesh-records.c orders_served) and `last`"): orders_served is removed: the test,
   before every delivery, that the plan this bridge released last no longer stood here, or that every port of its set
   (mesh-map.h mesh_ports_named) had refused it or held it with `served[p]` at least its `released`, and, where that
   set could not be computed, the plan read as served (B2-1), said on the bridge's log.  Why: it was Q4's hold (O2's
   reason: what a delivery left the ports was a count, so the order of deliveries across plans did not reach them);
   the delivery now publishes each execution's sequence number in its copy's word (L. Lamport, Concurrent reading and
   writing, CACM 20(11), 1977, "Message Buffer"), and the ports merge them (mesh-flow.c), so "Nothing holds"
   (design/aot-compilation-change-request.md, appendix item 2, "Delivery at a node"); B2-1's read-as-served would let
   the sender overwrite a word a port still reads (DL2).  What must stay: the start's port set, one copy (L's), which
   orders_deliver reads for Lamport's sender's wait */
/* Defago et al., Fig. 5, the destinations: each delivers the pending messages in sequence-number order, the one
   numbered `nextdeliver` next.  Here, as recorded (design/compilation-coordination.md, Q3, Q3-tree, DL2 and DL2′),
   it differs from Fig. 5: the sequencer's number starts at 0 (Fig. 5: 1); a SUBMIT of [first, end) takes end - first
   numbers, one an execution; and each node receives the ORDERs it takes in sequence order (one path from the
   sequencer down the island's tree, each hop in order), so delivery is by arrival order, the pending list's head
   standing for `nextdeliver`, and an ORDER's `sequence` is not compared.  The head is delivered once its plan stands here (its
   (group, epoch), of this node's map serial: mesh-dataflow.h mesh_head_state), with no timeout and no skip (O1); not
   matched, it stays.  Its delivery is the publication of each of its executions by the sender of L. Lamport's Message
   Buffer (Concurrent reading and writing, CACM 20(11), 1977: "sender: wait until ms < mr + P; put message in B[ms mod
   P]; ms := ms + 1"; design/aot-compilation-change-request.md, appendix item 2, "Delivery at a node": "The bridge
   parent writes each released execution's sequence number into its plan's state, one word per copy";
   design/compilation-coordination.md, "The documents' revisions of edd4b583", DL1 and DL2): P is the plan's `copies`
   u, B[c] its sequence word c (at its `sequences` + 8 c), ms its `released`, and each receiver's mr a port's
   `taken[p]`.  The head's next execution t is the plan's `released` (DL2′: Lamport's ms, one message at a time).
   Where `end` <= t, the ORDER leaves the head, its executions all published.  Where `first` > t, the head stays, said
   on the bridge's log once a head (`unarrived`): the message numbered `nextdeliver` has not arrived (Fig. 5), an
   earlier ORDER of the plan dropped here (order_take, order_relay) or not yet come (open, "Delivery's liveness").
   Where `first` <= t < `end`, t is numbered `sequence` + t - `first`: where t >= u and some port of the plan's port
   set (mesh-map.h mesh_ports_named, the start's set, one copy, over the map as this thread, its writer, holds it)
   that has not refused the plan
   (`refused[p]` 0, acquire) has `taken[p]` (acquire) below t - u + 1, Lamport's sender waits ("wait until ms < mr +
   P"): the head stays, read again on a later turn of the event loop (a port raises hdr `plans` where its `taken[p]`
   rose, DL9, mesh-flow.c, and mesh-link.c control_watch wakes the loop), nobody blocking, as a head that does not stand
   here stays (O1); else the number stored in word t mod u (an atomic store), then `released` t + 1 (release), "put
   message in B[ms mod P]; ms := ms + 1", and the head's `first` and `sequence` advanced past t in place (DL6); then
   on to t + 1.  hdr `plans` is raised and woken (mesh.h mesh_notify, DL9) after a turn that published, so the ports see
   it, and raised before a plan's first number (DL7).  Where this differs from the paper, recorded (DL1): every port of
   the set is a receiver of every message, so the sender's wait is over every receiver's mr; and a port's mr is the
   least of its lanes' next reads (mesh-flow.c).  A port set that cannot be computed (mesh_ports_named's error), and a
   plan whose `copies` is 0, which has no sequence word, or whose sequence
   words do not lie inside the region after its struct (mesh-flow.c plan_sequences_inside: a store there would be a
   fault, not an error value), each of which every port refuses (mesh-flow.c link_prepare, EINVAL), leave the head
   staying, EINVAL said on the bridge's log once a head, each case by its own line (`unnamed`; DL2: reading it as
   served, B2-1, would let the sender overwrite a word a port still reads; 2026-11-05, owner B4: the words' test, the
   coordinator's ruling DL8, and its own log line, X1 #3 and X3 #6).
   The head's
   wait holds every later ORDER behind it: the deliveries are in sequence order, and no port could serve a later number
   before it where it carries both; it is Lamport's sender's wait for the buffer the item names, not the hold of a
   plan's whole batch (DL2, recorded; the operator's to judge).  Re-read on each turn of the event loop; nobody waits.
   2026-11-04, owner B2, the coordinator's ruling Q4' (design/compilation-coordination.md, "Amendments during the
   work"): a head whose plan does not stand here is held at most MESH_PLATFORM_NS from its arrival (the rules' bounded
   wait; the loop's turn is due then, orders_due), then delivered with nothing released, said on the bridge's log as
   order_take's ENOMEM is: its program's peers end by their own bounds.  What must stay: no wait past that bound.
   2026-11-04, the coordinator's ruling on review V1's finding 4: the head is matched to the standing plan of its
   (group, epoch) whose `serial` is this node's map_serial; a standing plan of another serial is cancelled (D.3.6) and
   is not matched (a replan keeps its epoch, so two plans of one name may stand at once briefly after a move).
   2026-11-04, owner B3 (design/compilation-coordination.md, "The appendix's revisions", O1 and H1;
   design/aot-compilation-change-request.md, appendix items 2 and 3), reversing Q4''s note above: a head whose plan
   does not stand here stays at the head, and the delivery returns until a later turn finds it standing; the delivery
   "with nothing released" after MESH_PLATFORM_NS, its log line and orders_due (the loop's turn due at the bound) are
   gone.  Why: the fixed sequencer's delivery is in sequence-number order (Defago et al., Fig. 5), with no timeout and
   no skip (item 2), and the bound was a handler for a slow node, which nothing observed and no requirement asks for
   (item 3).  What must stay: delivery from the head alone, in arrival order (sequence order); the head matched to a
   standing plan of this node's map serial; the hold of orders_served (O2).
   2026-11-04, owner B3 (O2): the hold's reason in the first paragraph was "the operator's ruling 1: "A cable never
   interleaves two executions; it carries each whole, in sequence order"", Claude's words, which the appendix
   withdraws (item 2); it is O2's evidence now.  No behavior changed.  What must stay: the hold until the operator
   names what carries the delivery order from the bridge parent to its ports (O2, open).
   2026-11-04, owner B3 (review W3's finding 2), where `released` was stored the ORDER's `end` whatever it held: it is
   stored only upward, the greater of its value and the ORDER's `end` (this thread its one writer, so its own read is
   current).  Why: design/disjoint-pages-change-request.md, C3 ("Each state word has one writer, and no state word is
   reset while its plan stands") and C3.8 ("No state word is reset while its plan stands, and each has one writer").
   An ORDER of a cancelled plan can still match its replan, which keeps the plan's name (N1): open for the operator,
   recorded by the coordinator.  What must stay: `released` never lowered
   2026-11-04, owner B4 (design/compilation-coordination.md, "The documents' revisions of edd4b583", DL2, DL4 and
   DL6; design/aot-compilation-change-request.md, appendix item 2, "Delivery at a node"), reversing O2's hold kept
   above (the head delivered whole, `released` its `end`, only once every port holding the plan released last had
   served it, orders_served, so this node's cables carried one plan's batch at a time, each execution whole): the
   head's executions are published one at a time, each number in its copy's sequence word and `released` raised past
   it (release), Lamport's sender waiting only where the buffer of u words is full; orders_served, `last` and B2-1's
   read-as-served go.  Why: the item: "Each port serves the released executions of every standing plan in
   sequence-number order: a k-way merge"; "Both ends of a cable read the same sequence numbers, so they merge alike.
   Nothing holds."  The words carry the delivery order to the ports, which O2 found nothing to carry.  What must
   stay: delivery from the head alone, in arrival order (sequence order), no timeout and no skip; the head matched to
   a standing plan of this node's map serial; `released` only rising (O4), this thread its one writer; a word written
   only once no port of the set may still read the number it holds (Lamport's wait).
   2026-11-05, owner B4, the coordinator's ruling DL7 (design/compilation-coordination.md, "The documents' revisions
   of edd4b583", "Amendments during the work (2026-11-05)"): before a plan's first sequence number is stored (its
   `released` reading 0 as its publication begins), hdr `plans` raised by 1 (release) and woken (region-shared):
   mesh.h mesh_notify, whose release orders the raise before the word and `released` stored after it.  Why: DL3's
   reading in Lamport's opposite order (mesh-flow.c lane_take) needs every plan that can hold a number smaller than one
   a port acquired to be visible to that port's second reading, and mesh_plan stands a plan (its head's publication
   `state`, release) before it raises `plans`, so this parent can see a plan standing and publish its first number, and
   a later plan's, before that raise; a port that has not loaded the plan would then pass the `plans` test and take the
   later number first while its peer took the earlier.  With the raise first, a port that acquires any later number
   reads `plans` moved and walks the heads before it takes, and the walk reads the plan standing, since this parent
   read it standing before raising.  What must stay: the raise before the plan's first word and `released` stores.
   2026-11-05, owner B4, the coordinator's ruling DL2′ (reviews X1 #2 and X3 #4), reversing DL2's next execution "the
   greater of its `first` and the plan's `released`": it is `released`, and a head whose `first` is past it stays.
   Why: the greater let `released` pass executions whose words were never written, where an earlier ORDER of the plan
   never reached this node, so a port read them as published with a stale number; and Lamport's ms advances by one
   message at a time ("ms := ms + 1").  What must stay: one execution published at a time, `released` only rising.
   2026-11-05, owner B4, the coordinator's ruling DL9 (reviews X1 #7 and V1 #9), reversing this turn's wake of hdr
   `plans` without a raise after it published (Q4′'s form, "A port wakes `plans` (without raising it)", overturned
   with Q4): `plans` raised and woken (mesh_notify).  Why: a wake with no raise that lands between a waiter's read of
   `plans` and its wait is lost until MESH_PAUSE_NS, and the delivery now hands off between this parent and its ports
   once per execution past the u-th, where Q4's came once an ORDER; os_sync_wait_on_address returns at once where the
   word has moved.  Its cost, recorded: each raise moves `plans`, so each port walks the heads on its next pass
   (mesh-flow.c plans_walk), DL3's `plans` test lets a lane that read the old value take nothing for one pass, and
   this thread's release turn runs again (clients_turn: `plans` moved since `plans_seen`).  What must stay: DL7's raise
   before a plan's first number, as it is.
   2026-11-05 (design/aot-compilation-change-request.md at mesh 0e56d8da, appendix item 2, "Delivery at a node";
   design/compilation-coordination.md, "The revisions of 46ae8fdf", CH, M4 and M5), reversing the notes above from
   "The head is delivered once its plan stands here" on: each node runs Fig. 5's destination code as written ("when
   receive (m, seqnum): pending := pending U {(m, seqnum)}"; "while exists (m', seqnum') in pending: seqnum' =
   nextdeliver do deliver (m'); nextdeliver := nextdeliver + 1"), the pending list's head standing for `nextdeliver`
   as above (arrival order is sequence order), and deliver(m') is the head's one entry appended to the deliveries'
   channel (mesh.h struct mesh_delivery): its plan's name and executions [`first`, `end`), in entry k mod
   `deliveries_room`, k hdr `delivered`, then `delivered` k + 1 (release).  Nothing waits for the plan to stand here,
   for an earlier call to finish or for any port's `taken[p]`: "Delivery does not wait for any earlier call to finish,
   since Fig. 5 has no such condition"; the plan's sequence words, `released`, the port set (mesh_ports_named), the
   match to a standing plan and their log lines are gone ("publish them to each port through the channel above, in
   place of the sequence words (`sequences`), `taken[p]` and the parent's wait while they are full").  Entry k mod
   `deliveries_room` is reused for delivery k only once every port that reads the channel (its link up with a queue
   pair in its plan: port_toward's test, mesh-ports.c record_reach's) has read past delivery k - `deliveries_room`
   (`delivery_read[p]`, acquire, which its child stores after its copies, release): deferred reclamation ("An entry is
   reclaimed once every port has read past it").  Where one has not, the head stays until a later turn of the event
   loop, nobody blocking, said on the bridge's log once (`full`): the one wait left at the writer, which the item's
   "whose writer never waits" does not allow and a channel of fixed room cannot avoid ("The revisions of 46ae8fdf",
   M4).  hdr `plans` raised and woken after a turn that delivered (mesh.h mesh_notify), so the ports read the channel.
   What must stay: delivery from the head alone, in arrival order (sequence order), with no timeout and no skip; the
   entry's stores before `delivered`'s; no entry reused before every reading port has read past it.
   2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", CH2), reversing M4: entry k mod
   `deliveries_room` is written for delivery k with nothing read before it; the test of every reading port's
   `delivery_read[p]`, the head staying where one trailed by the room, and its log line (`full`) are gone.  Why:
   design/aot-compilation-change-request.md, appendix item 2: "The writer never waits and never checks for a full
   channel"; "(M4 and M5 go)"; "(Corrected 2026-11-05: the implementation's M4 kept an ORDER at the channel's head
   while a port trailed by 4096 entries, a wait at the writer.  ...  Neither has a citation, and both go.)"; the
   ports read the entries in place and tell nothing (mesh-flow.c lane_take, deliveries_join).  What the item derives
   for the room, a plan's claim of it ("So a plan claims room for that many entries, as it claims its pages (C3.4).  A
   plan the room cannot hold fails with `ENOMEM` when it is planned"), is not built ("The revisions of f3f7cf46", stop
   1).  What must stay: delivery from the head alone, in arrival order (sequence order), with no timeout and no skip;
   the entry's stores before `delivered`'s */
/* 2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", SQ2 and PQ), reversing CH's delivery
   above (the head of the pending list, standing for nextdeliver, its sequence not compared, appended to the one
   channel for every plan): Fig. 5's destinations' code as written, "while ∃(m′, seqnum′) ∈ pending : seqnum′ =
   nextdeliver do deliver (m′); nextdeliver := nextdeliver + 1", each message's `sequence` compared with `nextdeliver`,
   wherever it sits in `pending`.  deliver(m′) at a node of Dest(m′) (its `nodes` holding this node's map index) is
   its entry appended to its plan's queue here (A7: "deliver(m) at a node of Dest(m) appends m to its plan's queue
   there (appendix item 2)"): entry k mod `room` of the plan standing here of its name on this node's map serial, its
   sequence number and executions [`first`, `end`), then the plan's `delivered` k + 1 (release), nothing read before
   the write and nothing checked of the room (item 2: "Nothing waits to write, and nothing checks for a full queue");
   at a node outside Dest(m′) it is discarded (§5.2.4, step 3).  A message whose plan does not stand here, or whose
   queues lie outside the region, has nowhere to go: discarded, said on the log (its plan's barrier makes it stand
   here before its first message: A7).  After a turn that delivered: hdr `plans` raised and woken (mesh.h mesh_notify),
   then hdr `delivered` stored nextdeliver - 1 (release), then `plans` raised and woken again.  The first raise orders
   every plan standing here before any number a port acquires from `delivered` (a plan stands, then its arrival goes,
   then its messages come: a port that acquires `delivered` reads `plans` moved and walks the heads before it merges:
   mesh-flow.c lane_take); the second wakes a port that read the first and passed before `delivered`'s store.  Why:
   A7, "The algorithm is their Fig. 5 ... as written"; appendix item 2, "Each plan's delivered calls reach the bridge
   at a node through a queue of the plan's own"; "Fig. 5 delivers in sequence-number order (its `nextdeliver`), so every
   entry below the node's delivery count is in its queue already, and no smaller one comes later".  What must stay:
   delivery in sequence-number order, with no timeout and no skip; the entries' stores before `delivered`'s */
/* 2026-11-05 ("The revisions of 3f106747", VW-4): the entry's `sequence` the message's place in this node's delivery
   order (`passed`), where it was its sn(m), which begins again at each view.  What must stay: the entry's stores before
   `delivered`'s */
static void order_deliver(const struct mesh_order *m){
  struct mesh_state *state=plan_here(m->group,m->epoch);
  if(!state || !queues_inside(state)){
    mesh_say("bridge","the message numbered %llu, plan (%llu, %llu) executions [%llu, %llu), discarded: no plan of its name "
             "stands here with its queue in the region",(unsigned long long)m->sequence,(unsigned long long)m->group,
             (unsigned long long)m->epoch,(unsigned long long)m->first,(unsigned long long)m->end);
    return;
  }
  const uint64_t k=atomic_load_explicit(&state->delivered,memory_order_relaxed);
  ((struct mesh_delivery *)(void *)((char *)M+state->deliveries))[k%state->room]=
    (struct mesh_delivery){.sequence=passed,.first=m->first,.end=m->end};
  atomic_store_explicit(&state->delivered,k+1,memory_order_release);
}
/* 2026-11-05 ("The revisions of 3f106747", VW and VW-4): the messages numbered at most `last` alone, for a view's
   flush (view_install; UINT64_MAX on every other turn), and hdr `delivered` `passed`, the messages this node has
   passed in every view, where it was nextdeliver - 1, which begins again at each view.  Why: A7, "A member that
   installed the old view from the same process delivers the old view's messages up to that number, then installs the
   new view"; the ports' merge reads `delivered` as a bound on every entry's number (mesh-flow.c lane_take).  What must
   stay: delivery in sequence-number order, with no timeout and no skip; the entries' stores before `delivered`'s */
static void orders_deliver(uint64_t last){
  int delivered=0;
  for(size_t i=0;i<pendings && nextdeliver<=last;){
    if(pending[i].order.sequence!=nextdeliver){i++;continue;}
    const struct mesh_order m=pending[i].order;
    memmove(pending+i,pending+i+1,(--pendings-i)*sizeof *pending);
    passed++;
    if(island.self<64 && m.nodes>>island.self&1)order_deliver(&m);
    nextdeliver++;delivered=1;i=0;
  }
  if(!delivered)return;
  mesh_notify(&M->plans);
  atomic_store_explicit(&M->delivered,passed,memory_order_release);
  mesh_notify(&M->plans);
}
/* 2026-11-04, owner B3 (design/compilation-coordination.md, O1 and H1): orders_due (the pending head's bound, its
   arrival + MESH_PLATFORM_NS, which mesh-link.c turn_due waited for) is removed with the bound and its declaration
   (mesh-bridge.h).  Why: orders_deliver's note.  What must stay: the loop's turn bounded by its own times (turn_due) */
/* tag pre-return-20261103 rdma/mesh-records.c record_links_same (3f72cbde), adapted: a reach entry compared whole */
/* design/transport.md: a record whose reached nodes (each with its stream's bandwidth), boot and facts are the
   held one's changes only its health and sequence, written under the region's health serial and never the map's: the
   map serial moves only where the mesh changes, and a health sample is no change of the mesh */
/* differs from the tag: the reach entry is mesh.h's per cable, and every field of it (peer node, both ports, queue
   pairs, bandwidth) is the rule's input (struct mesh_hop, "Both ends"), so all are compared, not the tag's node and
   bandwidth alone; its bound MESH_PORTS, the tag's MESH_STREAMS */
/* 2026-11-05 ("The revisions of 3f106747", VW): the record's counter (mesh-map.h struct mesh_record `view`) compared
   too, so a record whose counter rose is written into the map (map_put, which moves no serial for it: neither its
   nodes nor its cables changed) and the sequencer reads the counter there (view_create).  What must stay: a
   health-only change written under the health serial alone */
/* 2026-11-05 (design/compilation-coordination.md, "The revisions of b5a3c6e6", RV): the installed view's creator
   (`view_sequencer`, `view_boot`) compared too, so a record that names another view at an unchanged counter is
   written into the map, where view_announced reads it.  Why: A7, "each announcement names the process that created its
   bridge's installed view as well as its counter".  What must stay: as above */
static int record_links_same(const struct mesh_record *a,const struct mesh_record *b){
  int same=a->node==b->node && a->reached==b->reached && a->boot==b->boot && a->view==b->view &&
    a->view_boot==b->view_boot && a->view_sequencer==b->view_sequencer &&
    a->region==b->region && a->memory==b->memory && !memcmp(a->model,b->model,sizeof a->model);
  for(uint32_t k=0;same && k<a->reached && k<MESH_PORTS;k++)
    same=!memcmp(a->reach+k,b->reach+k,sizeof *a->reach);
  return same;
}
/* tag pre-return-20261103 rdma/mesh-records.c health_put (3f72cbde), verbatim */
static void health_put(struct mesh_record *held,const struct mesh_record *record){
  mesh_notify(&M->health);
  held->sequence=record->sequence;held->health=record->health;
  mesh_notify(&M->health);
}
/* tag pre-return-20261103 rdma/mesh-records.c record_held (3f72cbde), verbatim */
struct mesh_record *record_held(uint32_t node){
  for(uint64_t i=0;i<atomic_load(&M->map_nodes);i++)if(mesh_records(M)[i].node==node)return mesh_records(M)+i;
  return NULL;
}
/* tag pre-return-20261103 rdma/mesh-records.c record_changed (3f72cbde), adapted: the arena's bytes, MESH_PORTS */
/* differs from the tag: `region` is the arena's bytes, as at the tag, whose wire mapping was the arena (tag
   mesh-verbs.h wire_map); main's `wire` is the registered window alone (mesh-verbs.h wire_map).  The reach's room is
   MESH_PORTS cables (mesh.h struct mesh_record), the tag's MESH_STREAMS streams */
/* 2026-11-03, design/disjoint-pages-change-request.md, Appendix A, A3 ("Sharing the region between programs"): the
   tag's `- held_pages*M->pgsz` goes, since the bridge holds no arena page (its map is the layout's area, map_grow's
   note, and it leases no link pages or duds' block, mesh-link.c): every arena byte is the programs'.  What must stay:
   `region` the bytes the programs' sections may take */
/* 2026-11-05 ("The revisions of 3f106747", VW): the record carries this bridge's counter, its installed view's
   identifier (A7: "Each bridge's announcement of itself to the sequencer carries its counter").  What must stay: as
   above */
/* 2026-11-05 (design/compilation-coordination.md, "The revisions of b5a3c6e6", RV): and the process that created that
   view, `installed`'s `sequencer` and `boot` (A7: "each announcement names the process that created its bridge's
   installed view as well as its counter").  What must stay: as above */
void record_changed(void){
  struct mesh_record own={.node=M->node,.boot=M->boot,.sequence=++sequence,.region=(uint64_t)mesh_arena_pages(M)*M->pgsz,.memory=memory,
                          .view=installed.view,.view_boot=installed.boot,.view_sequencer=installed.sequencer};
  memcpy(own.model,model,sizeof own.model);
  own.reached=record_reach(own.reach,MESH_PORTS);
  own.health=health;
  struct mesh_record *held=record_held(M->node);
  if(held && record_links_same(held,&own))health_put(held,&own);
  else map_put(&own);
  for(uint32_t p=0;p<port_count;p++)if(ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&own,sizeof own);
}
/* tag pre-return-20261103 rdma/mesh-records.c record_retell (3f72cbde), verbatim */
/* design/transport.md: this node's record told again at its current sequence on every greeted port, each
   liveness turn (MESH_PAUSE_NS), so each control socket carries a message well within MESH_PLATFORM_NS; a receiver drops
   a record no newer than its newest (record_heard, RFC 2328 s13), its map and
   serials unmoved.  Told from a copy: a port lost meanwhile may write the map */
void record_retell(void){
  const struct mesh_record *held=record_held(M->node);
  if(!held)return;
  struct mesh_record own;memcpy(&own,held,sizeof own);
  for(uint32_t p=0;p<port_count;p++)if(ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&own,sizeof own);
}
/* tag pre-return-20261103 rdma/mesh-records.c health_due (3f72cbde), verbatim */
/* when the next health sample is due (the event loop's turn waits for it: mesh-link.c turn_due) */
uint64_t health_due(void){ return health_sent+MESH_HEALTH_NS; }
/* tag pre-return-20261103 rdma/mesh-records.c health_turn (3f72cbde), verbatim */
void health_turn(void){
  if(now-health_sent<MESH_HEALTH_NS)return;
  health_sent=now;health.available=available_now();
  record_changed();
}
/* tag pre-return-20261103 rdma/mesh-records.c record_heard (3f72cbde, 60b00774, abeffc1b), adapted: MESH_PORTS */
/* a record heard: as much of it as this bridge's record holds, the rest 0 (design/HETEROGENEOUS-LINKS.md section 6,
   item 3: struct mesh_record is only appended to, so a longer one's first bytes are this one; a change of MESH_STREAMS or
   of the model's room is not an append, and moves MESH_VERSION), its reach at most MESH_STREAMS nodes;
   one naming no node, or this node, is not another node's */
/* differs from the tag: the reach's bound is MESH_PORTS cables (mesh.h struct mesh_record), the tag's MESH_STREAMS */
void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes){
  struct mesh_record heard={0};
  memcpy(&heard,payload,MIN((size_t)bytes,sizeof heard));
  if(!heard.node || heard.node==M->node)return;
  heard.reached=MIN(heard.reached,(uint32_t)MESH_PORTS);
  heard.model[sizeof heard.model-1]=0;
  struct mesh_record *held=record_held(heard.node);
  struct known *k=known_at(heard.node);
  /* design/transport.md, RFC 2328 s13: a record no newer than the newest heard of its node, the map holding
     it or not, is dropped and not sent on, so a record the map pruned never circulates (a node's later boot is always
     newer) */
  if(k && k->boot && (heard.boot<k->boot || (heard.boot==k->boot && heard.sequence<=k->sequence))){
    /* the newest known of a node the map dropped while no paired path reached it: put again where a path now does,
       and sent on only then (a pruned record that no path reaches stops here) */
    if(!held && heard.boot==k->boot && heard.sequence==k->sequence){
      map_put(&heard);
      for(uint32_t p=0;record_held(heard.node) && p<port_count;p++)
        if(ports+p!=from && ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&heard,sizeof heard);
    }
    return;
  }
  if(held && (heard.boot<held->boot || (heard.boot==held->boot && heard.sequence<=held->sequence)))return;
  if(k){k->heard=now;k->boot=heard.boot;k->sequence=heard.sequence;}
  if(held && record_links_same(held,&heard))health_put(held,&heard);
  else map_put(&heard);
  for(uint32_t p=0;p<port_count;p++)if(ports+p!=from && ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&heard,sizeof heard);
}
/* tag pre-return-20261103 rdma/mesh-records.c clients_turn (3f72cbde), adapted: main's one client word, not slots */
/* design/transport.md: every client slot read: a slot's process watched from the turn its slot is first
   seen (its exit, the event loop's EVFILT_PROC, retires every slot it holds: a holder's death is that event, not a
   poll), one already gone retired at once; the rows of a retired client released but those a request outstanding on a
   lane names (mesh-dataflow.c mesh_retired_release, mesh_pinned), a stream a gone client held let go (mesh-link.c
   streams_let_go: its port's progress thread, which posts nothing of it once the client retired, ends the holding), and
   the streams held told (busy_tell) */
/* differs from the tag: main has one client word, hdr `client` (its pid in the low 32 bits), not MESH_CLIENTS slots, so
   the tag's loop is its one slot, s = 0: `owners[0]` its last read, its watch's udata 0 (the tag's slot index; the
   event loop's EVFILT_PROC of udata 0 retires hdr `client` where its low 32 bits are the event's ident).  Main's word has
   no MESH_RETIRING bit: a retirement writes its retirer's pid there (mesh-dataflow.c mesh_retire), this bridge's
   skipped as the tag skips its own, another's watched as a client's.  mesh_retired_release keeps the rows a port's plan
   may still name (any port's `prepared` nonzero, mesh-dataflow.c c01fa79d) and returns how many it kept, as the tag's
   did, so `unreleased` stays set until none is kept */
/* 2026-11-04, sub-assignment 7 (design/disjoint-pages-change-request.md, C3.4: "The bridge's own turn releases a
   retired plan's pages once no port holds them", deferred reclamation; design/disjoint-pages-revision-coordination.md,
   "The program's claim and its sections" and D10), the note above made exact for the claims per plan: what
   mesh_retired_release keeps is each closed claim whose generation a port's `prepared` holds (after a seq_cst fence),
   not every closed claim while any port holds any plan, and the count it returns is of the closed claims kept.  And
   the order of this turn's last calls is reversed from the tag's (release, then let go): streams_let_go first, then
   the release, so a cancellation this turn computes from a plan's entries (mesh-flow.c plan_cancel) lands in that
   plan's pages before this turn's release can let them go.  What must stay: the cancellation before the release */
/* 2026-11-04, owner B2 (design/plan-to-bridge-change-request.md R8: "The code must hold nothing for a program but its
   plans' claims ... there is no client word"; e6bd7db2's section 6, "Holders and clients", as design/compilation-
   coordination.md ruling 1 and Q5 take it: "clients_turn walks the heads.  It adds an EVFILT_PROC watch for each open
   claim's pid, idempotent per pid, and retires at once a pid already gone"), reversing the one-client form above, in
   which the turn read hdr `client`, which nothing writes from 110 (L3), and kept its last read in `owners[0]`: the
   turn walks the claims' heads (mesh-dataflow.h mesh_head_after), and for each open claim (its owner not 0, `closed`
   0) of a process other than this bridge adds the watch of its pid's exit, EV_ADD, which kqueue makes once a pid (an
   event added again only updates the one there), udata 0 (the event loop's EVFILT_PROC retires every owner of that
   pid: clients_gone); a pid already gone (ESRCH) has the claim's owner retired at once (mesh_retire: its claims closed,
   `plans` raised).  streams_let_go's call goes: a gone client's claims are closed by its retirement, and the ports end
   its plans (mesh-flow.c).  The turn then submits and delivers the sequencer's work (orders_submit, orders_deliver),
   and the release runs where hdr `plans` moved since the last turn (a claim closed or a plan published: `plans_seen`,
   in `owners[0]`'s place) or on a liveness turn, as before.  What must stay: the release last, after this turn's
   cancellations and deliveries, on this thread */
static uint64_t plans_seen;
void clients_turn(int liveness){
  const pid_t self=getpid();
  uint64_t busy=0;
  for(uint32_t head=mesh_head_after(M,UINT32_MAX);head!=UINT32_MAX;head=mesh_head_after(M,head)){
    struct mesh_buffer *claim=mesh_buffers(M)+head;
    const uint64_t owner=atomic_load_explicit(&claim->owner,memory_order_acquire);
    if(!owner || atomic_load_explicit(&claim->closed,memory_order_acquire) || (pid_t)(uint32_t)owner==self)continue;
    struct kevent64_s watch;EV_SET64(&watch,(uint32_t)owner,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,0,0,0);
    if(kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)<0 && errno==ESRCH)mesh_retire(M,owner);
    else busy=1;
  }
  orders_announce(liveness);
  counts_tell(liveness);
  arrivals_apply();
  orders_submit();
  orders_deliver(UINT64_MAX);
  const uint64_t plans=atomic_load_explicit(&M->plans,memory_order_acquire);
  unreleased|=plans!=plans_seen;plans_seen=plans;
  /* the tag's line again (the coordinator, at integration, once mesh-dataflow.c's mesh_retired_release returns the rows
     it kept, c01fa79d): a retired client's rows a port's plan may still name are kept, and looked at again next turn */
  /* 2026-11-04, sub-assignment 7, for mesh-flow.c plan_cancel (design/disjoint-pages-change-request.md, C3.6;
     design/disjoint-pages-revision-coordination.md, "The bridge": "plan_cancel and the release of retired claims
     (mesh_retired_release) both run on the bridge parent's one event-loop thread"): they do.  In the bridge,
     mesh_retired_release is called here and in mesh-link.c region_adopt, and plan_cancel only from streams_cancel;
     each of those runs on the thread of mesh-link.c main alone: region_adopt before the event loop, its
     streams_cancel(NULL) before its release; streams_cancel from
     mesh-ports.c port_lost and mesh-children.c child_reaped and child_heard, which only the loop's turn and its events
     reach (port_lost's callers: the loop's events, child_reaped, control_heard, hello_heard, network_heard and a
     control socket's connect; its turn's terminations_turn, ports_stalled, link_turn, port_open by child_spawn,
     health_turn and record_retell by port_say; and network_heard from the stop's hold after the loop), and
     streams_cancel(NULL) at the stop, after the loop.  2026-11-04, sub-assignment 1 (design/disjoint-pages-change-
     request.md, D.3.6; design/disjoint-pages-rulings-coordination.md, R2): plan_cancel is called also from mesh-flow.c
     plan_stale, which map_put and map_adopt alone call: map_put from record_changed (mesh-ports.c link_set and
     link_turn, health_turn here, mesh-link.c main before the loop) and record_heard (mesh-ports.c message_heard, from
     control_heard, the loop's events), link_set from port_heard, port_lost, port_opened and ports_found, port_lost as
     above and from port_say and port_flush at the stop; map_adopt from region_adopt, after its release, before the
     loop.  Each on main's thread.
     The parent's other threads call none of them: control_watch only wakes the loop; the finder's queue (mesh-ports.c
     ports_find) only lists the devices; the terminations' queue (mesh-events.c peer_terminated, peer_arrived) only
     stores atomics (port_halt) and wakes the loop.  The port children's threads, which land and send, are other
     processes and call neither.  So a cancellation's stores, computed from a plan's entries read whole, are done before
     any later call of the release can let that plan's pages go.
     2026-11-04, owner B2: streams_let_go, which this turn called, is gone (the note above this function), so this note
     names plan_cancel's callers without it; and the sequencer's functions (island_set from map_put and map_adopt,
     order_heard from mesh-ports.c message_heard, orders_submit and orders_deliver here, clients_gone from the loop's
     EVFILT_PROC) run on this same thread, so orders_served reads a released plan's words only before this thread's
     release can let its claim go.  What must stay: every one of them on main's thread.
     2026-11-04, owner B4 (design/compilation-coordination.md, "The documents' revisions of edd4b583", DL2 and DL4):
     orders_served is gone; orders_deliver reads and writes the words of the plan it matched standing in this call
     (its `copies`, `sequences`, `refused`, `taken`, its sequence words and `released`), before this turn's release
     below, on this thread, so a plan whose claim closes meanwhile keeps its pages until a later release.  What must
     stay: as above */
  if(unreleased || liveness)unreleased=mesh_retired_release(M)!=0;
  busy_tell(busy);
}
/* not the tag's (2026-11-04, owner B2: e6bd7db2's section 6, "Holders and clients", as design/compilation-
   coordination.md ruling 1 and Q5 take it: "The event loop's EVFILT_PROC retires every owner of the event's pid"): each
   open claim whose owner's process is `pid` has that owner retired (mesh-dataflow.h mesh_retire: every claim it owns
   closed, `plans` raised and woken); mesh-link.c's event loop, on main's thread */
void clients_gone(pid_t pid){
  for(uint32_t head=mesh_head_after(M,UINT32_MAX);head!=UINT32_MAX;head=mesh_head_after(M,head)){
    const struct mesh_buffer *claim=mesh_buffers(M)+head;
    const uint64_t owner=atomic_load_explicit(&claim->owner,memory_order_acquire);
    if(owner && !atomic_load_explicit(&claim->closed,memory_order_acquire) && (pid_t)(uint32_t)owner==pid)mesh_retire(M,owner);
  }
}
