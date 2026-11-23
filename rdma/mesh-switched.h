/* The switch (design/PAGE-TABLE-SPEC-20261122.md section 5): what every node's bridge does with a page, as functions of
   its region.  The bridge's port children (mesh-link.c port_pass) and its parent (its event loop as it writes the map,
   the policy's thread while a theta is loaded, a child's end, a region's adoption) are its callers; none of it does
   I/O or posts to hardware, and only the cost-to-go's step (its lines' age: mesh_switch_togo_step) reads a clock: the
   caller passes the time and posts what these functions hand it.

   A page crosses as one SEND from its own memory (mesh.h struct mesh_pd): its first MESH_HEAD bytes the header the
   bridge writes (struct mesh_head: the node that took the write, its destinations as nodes and indices of their page
   tables, the nodes it visited), then its payload, then the lines of lane state it carries in its last frame's slack.
   A SEND fills one receive posted into a page of the lane's class (TN3205: both ends post the same frame count), and
   the lane's child posts that receive again at once (mesh-link.c link_receive: into a free page, else into the page
   that landed, whose landing is then lost), so a lane's receives do not run out while every page of its class is used.
   - Take.  A client's write names its page and its destinations: the writer sets the table entry of each destination
     on this node to the page itself (mesh-call.c mesh_write), and puts the others on its ring (mesh.h struct
     mesh_write); the bridge writes the page's header and forwards the page toward them (mesh_switch_carry).  Each ring
     has one taker, which every port child derives from the port lines (struct mesh_switch `port`): ring s the (s mod
     n)-th, by port index, of the n ports whose child serves.
   - Switching.  At every node a page is at, the destinations it carries on are split over the node's out-neighbours
     (the peers its ports serve toward, as the port lines tell them: mesh_switch_choose): each destination to its
     allowed neighbour of the earliest arrival (that neighbour's cable's wait with the page on it, the link's latency,
     then the cost-to-go from it: the design's rule), or, where the trained policy's split tables have the page's
     context, as the split drawn from them at one uniform u per page (destinations that go to one neighbour share one
     SEND); allowed, the neighbours off the page's path that reach the destination without coming back, over the lanes
     of its payload's class alone (the frames of the least class that holds the payload, mesh.h mesh_class_frames: a
     SEND covers its lane's frames, so on a lane of more frames it would carry the pool's next pages with it).  The
     cable to a neighbour is the one of the least wait (its bytes queued over its rate).  The page is pushed onto the
     inbound ring of each group's cable for the page's class, a use of it a group (struct mesh_switch_inbound: a ring
     for each pushing port, cable and class, its one producer the pushing port's child and its one consumer the
     cable's, which posts each page it finds there on the lane of the page's class with the fewest SENDs outstanding,
     mesh-link.c switch_sends): the page is switched across the node's ports, whichever child has it.
   - Landing.  A page landed sets the table entry of each destination here whose turn it is (mesh_switch_set: the
     index's landed count one more, the page it named before dropped) and is forwarded unchanged for the others it
     carries (mesh_switch_landed).  Nothing of what lands is tested: each field of its header that addresses memory is
     clamped into what it addresses (its bytes to its page's payload, its destinations to MESH_DESTS, its hops to its
     path, a destination's index to the table), so a header that is not one writes only within its page, its path and
     the table.  Nothing waits and nothing is reported: a destination no allowed neighbour reaches, a page a lost link
     took, a landing whose receive went again into its own page are each counted (`unreachable`, `lost`: struct
     mesh_switch_counters), never retried.
   - Uses.  A page's uses (mesh.h) count its table words, its writes pending in a ring, its SENDs pushed (an inbound
     ring's entry) or posted (an in-flight ring's entry: struct mesh_switch_lane) and a landing being carried; its last
     use's drop gives it back to its class's free stack.
   - The routing state.  The port lines (each port's child's state, its link's peer and plan: one word a port, all in
     one cache line, moved by one compare-exchange: mesh_port_move) say which links serve.  The view and the policy's
     split tables are each in two copies, each with one writer (the view the parent's event loop as it writes the map;
     the tables its policy's thread), which writes the copy not published, then publishes it (its version moved, one
     release store); a draw reads the copy published as it began (its version, one acquire load) and retries nothing,
     so draws go on reading the last published copies while one is computed: never a wait.  A draw that outlasts the
     next publish (the other copy's) may read its own copy as its writer writes it again, and at worst chooses a wrong
     route; every index it reads there addresses only within its array (mesh_switch_choose).  Each port child's poller
     makes its own cost-to-go, a class of page each, a step an idle pass, in its own memory (struct mesh_switch_togo).
     The trained recurrent policy (rdma/route) plugs in through struct mesh_switch_policy, as the split tables.
   - The estimators.  Diachronic and incremental: each cable's rate and a page's own time on it, for each class, from
     its SENDs' completions (mesh_switch_cable_sent; each lane's sums and estimate too, mesh_switch_sent); each remote
     link's terms from the lines pages carry (mesh_switch_lines_take), each line's backlog draining on this node's own
     clock from the time it took the line (mesh_switch_line_cost); each node's own links' lines made by the forward that
     sends them, stamped with its boot plus its uptime, which orders that node's own lines alone.  Nothing waits for any
     of it.

   Every hop adds the node to the page's visited set and a lane is taken only toward a neighbour off it that reaches
   the destination avoiding it, so a path is simple and ends; each destination is in exactly one group at every hop.

   The trainer's records.  Every draw of a write's page can be logged, off the hot path, in rdma/route's record format
   (split.h struct route_draw: each (destination, out-lane)'s arrival estimate and validity, the allowed mask, the
   sampled split as its token, and its behaviour log-probability, exact under the coupled draw), and every page set at a
   destination its latency and queueing: a single-producer ring per port child in the region (mesh_switch_log_put drops
   and counts where it is full; it never waits). */
#ifndef MESH_SWITCHED_H
#define MESH_SWITCHED_H
#include "mesh.h"
#include "route/route.h"
#include <errno.h>
#include <math.h>
#include <string.h>

#define MESH_SWITCH_MAGIC 0x53574349u
#define MESH_SWITCH_LANES (MESH_PORTS*2u*MESH_LINK_STREAMS)   /* every port's lanes, its link's queue pairs */
#define MESH_SWITCH_N 8u                                     /* nodes the view has */
#define MESH_SWITCH_LINKS 64u                                /* directed node pairs the view has */
#define MESH_SWITCH_INBOUND 32768u                           /* an inbound ring's entries: every page of the pool once */
#define MESH_SWITCH_FLIGHT 4096u                             /* a lane's in-flight ring: its SENDs outstanding, at most */
#define MESH_SWITCH_LINES 24u                                /* lines a header carries */
#define MESH_SWITCH_LOG 8192u                                /* records a process's log ring */
#define MESH_SWITCH_LOGS (MESH_PORTS+1u)                     /* counters a port child, then the parent's */
#define MESH_SWITCH_PARENT MESH_PORTS                        /* the parent's counters */
#define MESH_SWITCH_CONTEXTS 256u                            /* the policy's contexts at a node, at most */
_Static_assert(MESH_SWITCH_N<=64 && MESH_SWITCH_N<=MESH_NODES,"a node a bit");
_Static_assert(MESH_DESTS<=32,"a destination a bit");
/* A page is on the inbound rings toward a cable at most once: its push holds a use until its SEND completes or is
   given back, and that use blocks the page's next write (mesh-call.c mesh_write's compare-exchange from 1) and its use
   as a receive (a page is posted as one only off its class's free stack), and a forward pushes a page toward each cable
   once (one group a neighbour, a cable serving one peer).  A page goes on the ring of its payload's class, not of its
   own (mesh_switch_forward: a long page whose payload a short page holds, a message's last, goes as a short SEND), so
   a short ring may hold every page of the pool, and a long ring every long page.  So an inbound ring as deep as the
   pool's pages never fills, and its producer tests no room.  An in-flight ring holds a lane's SENDs posted and not
   completed, at most the lane's depth (its queue's share of the frames its queue pair's SENDs take, over the lane's
   frames, and never past MESH_SWITCH_FLIGHT: mesh-link.c lane_reset; a queue pair takes mesh-verbs.h QD 4095 frames at
   most), each at the position its wr_id names (its low 48 bits, of which the ring's size is a divisor) */
_Static_assert(MESH_SWITCH_INBOUND>=MESH_POOL_PAGES && !(MESH_SWITCH_INBOUND&(MESH_SWITCH_INBOUND-1)),
               "an inbound ring has room for every page of the pool once");
_Static_assert(!(MESH_SWITCH_FLIGHT&(MESH_SWITCH_FLIGHT-1)) && MESH_SWITCH_FLIGHT<=(UINT64_C(1)<<48),"an in-flight position a wr_id's");

/* ---- The header (a page's first MESH_HEAD bytes) ------------------------------------------------------------------- */
/* a destination as a header carries it: its node and index of that node's table; the node whose turn it is to carry it
   on (`next`: the node a hop sent it to, which alone switches it; a page goes out unchanged on every group's lane, each
   receiver reading only the destinations it is the next of; UINT32_MAX no one's); and the cost it carried so far (ns:
   queueing at each hop, the wire's time) */
struct mesh_hop { uint32_t node,index,next; float queued,transit; };
/* a line: a directed link as its sender (`from`) measured it at `stamp` (its boot plus its uptime, ns: mesh_switch_stamp,
   which orders that sender's own lines, a restarted bridge's after its last one's, and is never read against another
   node's clock), for each class c of page (0 short, 1 long: mesh_switch_class): `cost[c]`, the time (ns) a page of
   that class pushed then took to land through it, its quickest cable's backlog and the page's own time on that cable
   (INFINITY: the link is down, none of its sender's ports serving toward its peer), and `idle[c]`, the page's own time
   on its quickest cable with nothing queued (0: not measured), from its sender's measures of its own cables (struct
   mesh_switch_cable).  A node that passes on another's line sends its cost as it counts it then
   (mesh_switch_lines_give: its backlog less the time the node has held it), so a line's age is counted on each node's
   own clock in turn */
struct mesh_switch_line { uint32_t from,to; uint64_t stamp; float cost[MESH_CLASSES],idle[MESH_CLASSES]; };
_Static_assert(sizeof(struct mesh_switch_line)==32,"a line's 32 bytes");
/* the header, written by the bridge (never the client): the node that took the write (`source`); the payload's bytes;
   the hops, destinations and lines; the node that sent this SEND (`from`: its receiver appends itself to `path`);
   `unused`, unused; the nodes visited, `hops` + 1 of them; the destinations; and the sequence of the
   headers its first node made (`seq`: the draw's seed, the records' sampling and their join).  Every field names nodes
   by their numbers, never by a view's indices (each node's view is its own).  The lines follow the payload in the last
   frame's slack (mesh_switch_lines_at) */
struct mesh_head {
  uint32_t source,bytes,hops,dests,lines,from,unused;
  uint32_t path[MESH_SWITCH_N];
  struct mesh_hop dest[MESH_DESTS];
  uint64_t seq;
};
_Static_assert(sizeof(struct mesh_head)<=MESH_HEAD && MESH_HEAD%64==0,"a header within its MESH_HEAD bytes");
_Static_assert(sizeof(struct mesh_head)>MESH_HEAD-64,"MESH_HEAD no longer than the header needs");
static inline struct mesh_head *mesh_head_of(struct hdr *m,uint32_t pd){ return (struct mesh_head *)mesh_pd_head(m,pd); }

/* ---- The region's switch (struct hdr switch_off; MESH_SWITCH_BYTES, mesh.h) ------------------------------------- */
/* a link of the view: a directed node pair, its cables, and its nominal rate (all of them, bytes a ns) and latency
   (ns), as the records give them: a link's cost before its sender's lines are heard, and the time scale of the
   policy's features */
struct mesh_switch_link { uint32_t from,to,cables,ends; double rate,latency; };   /* ends: the view indices, from | to << 8 */
/* a copy of the view (struct mesh_switch view[2]): `nodes` at most MESH_SWITCH_N and `links` at most MESH_SWITCH_LINKS
   in every store of them (mesh_switch_view_set), so a loop under either stays within its array */
struct mesh_switch_view {
  uint32_t self,nodes,links,ranks; /* ranks: the view nodes that run collectives' ranks, bits over view indices (0: all) */
  uint32_t node[MESH_SWITCH_N];     /* node numbers, ascending: a node's index is its place here */
  /* each view node's entry of the node table (mesh.h struct mesh_node), the stable index its links' lines are kept
     under (struct mesh_switch `heard`): this node's MESH_NODES, one with no entry UINT32_MAX */
  uint32_t slot[MESH_SWITCH_N];
  /* the policy's features' time scale (mesh_switch_time_scale over the links' nominal terms, made with them): a
     draw's record's unit */
  double scale;
  struct mesh_switch_link link[MESH_SWITCH_LINKS];
  /* reach[avoid][from]: the view nodes `from` reaches (itself among them) entering none of `avoid` (bits over view
     indices), made with the links (mesh_switch_view_set): a draw's allowed neighbours read here, not searched */
  uint8_t reach[1u<<MESH_SWITCH_N][MESH_SWITCH_N];
};
/* the policy's split tables (rdma/route route.h): this node's contexts, each (its pattern's source, the pattern's
   destinations, the destinations still to carry: view indices, the last two as bits) in mesh_switch_contexts'
   sequence, and each one's w [SPLIT_FS], published as the view is (two copies).  `active` set, a page whose
   context is here draws its split from split.h's table under its w (mesh_switch_choose); else the design's rule
   chooses.  `point`: the policy refresh the tables are from */
struct mesh_switch_context { uint32_t source,dest_nodes,onward,pattern; };
struct mesh_switch_splits {
  uint32_t active,contexts,argmax,padding;
  uint64_t point;
  struct mesh_switch_context key[MESH_SWITCH_CONTEXTS];
  float w[MESH_SWITCH_CONTEXTS][SPLIT_FS];
};
/* a cable of this node, port p's (its link's lanes), on a line of its own: the bytes pushed onto its inbound rings or
   posted on its lanes and not yet completed (`queued`), of every class, every pushing child's (mesh_switch_forward) and
   its own child's as its SENDs complete (mesh_switch_sent); and its terms, each its own child's measure of its SENDs' completions
   (mesh_switch_cable_sent), an exponential average, 0 until its first sample: `rate`, the bytes it completes a ns
   while it has SENDs outstanding (the rate its backlog drains at), and `page[c]`, the time (ns) from its post to its
   completion of a SEND of class c (mesh_switch_class) that had the cable to itself (a page's own time on it, nothing
   queued ahead) */
struct mesh_switch_cable { _Alignas(64) _Atomic uint64_t queued; _Atomic float rate,page[MESH_CLASSES]; };
/* a remote link's line as this node took it from the headers that carried it (mesh_switch_lines_take), kept under its
   ends' node table entries (struct mesh_switch_view `slot`), so it outlives every republish of the view that keeps
   the link: its sender's stamp (0: no line heard), the time this node took it (`taken`, its uptime clock,
   CLOCK_UPTIME_RAW ns, from which its backlog drains: mesh_switch_line_cost) and its terms (struct mesh_switch_line) */
struct mesh_switch_heard { _Atomic uint64_t stamp,taken; _Atomic float cost[MESH_CLASSES],idle[MESH_CLASSES]; };
/* an entry of an inbound ring or an in-flight ring: page `pd`, and `at`, its push's clock on an inbound ring (the
   pusher's as its pass began) and its SEND's post on an in-flight ring (the lane child's clock as the post returned) */
struct mesh_switch_entry { uint64_t at; uint32_t pd,unused; };
_Static_assert(sizeof(struct mesh_switch_entry)==16,"mesh_switch_entry");
/* the inbound ring of a pushing port, a cable and a class (struct mesh_switch `inbound`): one producer, the pushing
   port's child, which writes the entry at `tail` and then moves `tail` (release); one consumer, the cable's child, which
   posts the entries from `head` and moves `head` past them, or, its child exited, the parent, which gives them back
   (mesh_switch_inbound_drop) */
struct mesh_switch_inbound { _Alignas(64) _Atomic uint64_t tail; _Alignas(64) _Atomic uint64_t head; struct mesh_switch_entry entry[MESH_SWITCH_INBOUND]; };
/* a lane of this node (port p's lane j, its link's queue pair j: p 2 MESH_LINK_STREAMS + j; its peer its port's, its
   frames mesh_switch_lane_frames of j, served while its port's line is CHILD_SERVING with j in its plan), written by its
   child alone while the child lives, what is left given back by its parent once it has exited
   (mesh_switch_flight_drop): its in-flight ring (`flight`: its SENDs posted and not completed, each at the position its
   wr_id names, from `completed` to `posted`, both 0 as its child opens its lanes),
   and its feedback: SENDs completed, their bytes, their service and wait (ns, summed), pages landed, the estimate its
   child keeps (rate in bytes a ns and wait, two floats in one word) */
struct mesh_switch_lane {
  _Alignas(64) _Atomic uint64_t posted,completed;
  _Atomic uint64_t sends,bytes,service,wait,landed,estimate;
  struct mesh_switch_entry flight[MESH_SWITCH_FLIGHT];
};
/* the trainer's records (rdma/route): a draw (split.h struct route_draw and its behaviour log-probability) or a page
   set at a destination (its latency and queueing), each with the node that made it, the page's `seq` and source, its
   first destination's node and index (a draw's) or the destination's, its bytes, the draw's context at its node
   (UINT32_MAX: none, the design's rule chose) and the policy refresh its split table is from, at the node's clock
   (ns) */
enum { MESH_LOG_DRAW=1, MESH_LOG_SET };
struct mesh_switch_record {
  uint32_t kind,node;
  uint64_t seq,at;
  struct route_draw draw;           /* MESH_LOG_DRAW */
  float logp,latency,queueing;
  uint32_t source,dest,index,bytes,context,point;
};
struct mesh_switch_log { _Alignas(64) _Atomic uint64_t head; _Alignas(64) _Atomic uint64_t tail,dropped; struct mesh_switch_record record[MESH_SWITCH_LOG]; };
/* where a pass's time goes, counted in memory (never storage) by each process of the switch, a set a port child (at
   its port's index) and the parent's (at MESH_SWITCH_PARENT), read by mesh-switch-stat:
   - passes, and of one pass in 1024 its time (`pass_ns`: summed, and the passes timed) and each part's (`part_ns`:
     receive completions, receives posted, writes taken, SENDs posted, SEND completions with an idle pass's step of
     the cost-to-go);
   - writes taken, pages landed, table entries set here, forwards and the groups they made;
   - SENDs posted and their wait from the push onto the inbound ring to the post, the posts' own time (the provider's
     post_send), SENDs completed and their time from post to completion;
   - receives a lane's fill asked for that its class's free stack had no page for (`starved`), the policy's refreshes;
   - destinations no allowed neighbour reaches (`unreachable`), pages an exited child's rings held or its next child
     found on its inbound rings, SENDs the provider did not take, and landings whose receive was posted again into their
     own page, their class's free stack empty (`lost`: once a page, not a destination) */
struct mesh_switch_counters {
  _Alignas(64) _Atomic uint64_t passes,pass_ns[2],part_ns[5];
  _Atomic uint64_t taken,landed,set,forwards,groups;
  _Atomic uint64_t posts,post_wait_ns,post_call_ns,sent,send_ns;
  _Atomic uint64_t starved,refreshes;
  _Atomic uint64_t unreachable,lost;
};
/* a port child's state (design/transport.md section 14, "A process a port"): started, its device not yet open; its
   device open; its link's lanes made; serving them, the one state in which a draw pushes onto its lanes or it takes
   from the clients' rings; leaving, its link lost (nothing more goes to its lanes, and the child ends).  A port with
   no child is STARTED, the state its next child begins in */
enum child_state { CHILD_STARTED, CHILD_DEVICE_OPEN, CHILD_LANES_MADE, CHILD_SERVING, CHILD_LEAVING };
/* `writes`: the headers this node made (the writes it took), each one's `seq`.  `port`: each port's line, its
   child's state (enum child_state) in bits 0-7, its link's plan (the lanes its pairing made, a bit a lane:
   mesh-bridge.h link_plan) in bits 8-15 and its peer node in bits 32-63, every port's in one cache line, which every
   draw and every take pass reads and only a change of a child's state writes (mesh_port_move).  The view and the
   split tables, two copies each: the published one is [version & 1], each version (`view_version`, `splits_version`)
   the publishes of its copies so far, moved by a release store once the copy not published is written whole.
   `cable`: each port's bytes queued and terms.  `heard`: each remote link's line, [its sender's node table
   entry][its receiver's, this node's MESH_NODES].  `inbound`: the inbound rings, [the pushing port][the cable's
   port][the class].  `stamp_offset`: the stamps' clock less the uptime clock
   (mesh_switch_stamp), the parent's to set, once */
_Static_assert(2*MESH_LINK_STREAMS<=8,"a port's plan a byte of its line");
struct mesh_switch {
  uint32_t magic,padding;
  _Atomic int64_t stamp_offset;
  _Atomic uint32_t logging;
  _Alignas(64) _Atomic uint64_t writes;
  _Alignas(64) _Atomic uint64_t port[MESH_PORTS];
  _Alignas(64) _Atomic uint64_t view_version,splits_version;
  struct mesh_switch_view view[2];
  struct mesh_switch_splits splits[2];
  struct mesh_switch_cable cable[MESH_PORTS];
  struct mesh_switch_heard heard[MESH_NODES][MESH_NODES+1];
  struct mesh_switch_lane lane[MESH_SWITCH_LANES];
  struct mesh_switch_inbound inbound[MESH_PORTS][MESH_PORTS][MESH_CLASSES];
  struct mesh_switch_log log[MESH_PORTS];
  struct mesh_switch_counters counters[MESH_SWITCH_LOGS];
};
_Static_assert(sizeof(struct mesh_switch)<=MESH_SWITCH_BYTES,"the switch within its region bytes");
_Static_assert(sizeof(((struct mesh_switch *)0)->port)==64,"the port lines one cache line");
/* a port line's fields (struct mesh_switch `port`) */
static inline uint32_t mesh_port_state(uint64_t line){ return (uint32_t)(line&0xff); }
static inline uint32_t mesh_port_plan(uint64_t line){ return (uint32_t)(line>>8&0xff); }
static inline uint32_t mesh_port_peer(uint64_t line){ return (uint32_t)(line>>32); }
/* port p's line moved to `state` from any of the states `from` (bits over enum child_state), by one compare-exchange
   (sequentially consistent), its peer and plan those given where `state` is CHILD_SERVING, else as they were: 1 where
   it moved, 0 where its state was none of `from`.  Its callers: a port's child's main thread as it opens its device,
   makes and connects its lanes (mesh-link.c device_open, lanes_make, lanes_open), and any of the child's contexts as it
   halts (mesh-link.c child_halt: its poller at a lane's failure, its main thread at a termination heard, a command
   that failed or its end, the fault handler); the parent as it loses the link from the make on, or stops (mesh-link.c
   port_leaving).  The parent alone makes the line new (STARTED, peer and plan 0) once the child has exited
   (mesh-link.c switch_child_free) */
static inline int mesh_port_move(struct mesh_switch *sw,uint32_t p,uint32_t from,uint32_t state,uint32_t peer,uint32_t plan){
  _Atomic uint64_t *word=&sw->port[p%MESH_PORTS];
  uint64_t was=atomic_load(word);
  for(;;){
    if(!(from>>mesh_port_state(was)&1))return 0;
    const uint64_t next=state==CHILD_SERVING?(uint64_t)state|(uint64_t)(plan&0xff)<<8|(uint64_t)peer<<32:(was&~UINT64_C(0xff))|state;
    if(atomic_compare_exchange_weak(word,&was,next))return 1;
  }
}
/* the port of the switch's lane l, and its cable */
static inline uint32_t mesh_switch_port_of(uint32_t l){ return l/(2*MESH_LINK_STREAMS)%MESH_PORTS; }
static inline struct mesh_switch_cable *mesh_switch_cable_of(struct mesh_switch *sw,uint32_t l){ return sw->cable+mesh_switch_port_of(l); }
/* a page's class of `frames` frames, an index of the terms (0 short, 1 long), and a class's frames */
_Static_assert(MESH_CLASSES==2,"two classes: short and long");
static inline uint32_t mesh_switch_class(uint32_t frames){ return frames!=MESH_CLASS_SHORT; }
static inline uint32_t mesh_switch_class_frames(uint32_t c){ return c?MESH_CLASS_LONG:MESH_CLASS_SHORT; }
/* the stamp of uptime `now` (CLOCK_UPTIME_RAW, ns, the clock every process of the node reads): this node's boot plus its
   uptime since, their offset from the uptime clock the parent's, set once as it makes the switch (mesh-link.c
   switch_make): an order of this node's own lines alone, never read against another node's clock */
static inline uint64_t mesh_switch_stamp(const struct mesh_switch *sw,uint64_t now){
  return now+(uint64_t)atomic_load_explicit(&sw->stamp_offset,memory_order_relaxed);
}
/* one sample taken into a cable's term (an exponential average, 1/8 a sample, its first taken whole), by its one
   writer, the cable's child: a load and a store */
static inline void mesh_switch_average(_Atomic float *term,double sample){
  const float was=atomic_load_explicit(term,memory_order_relaxed);
  atomic_store_explicit(term,(float)(was>0?was+(sample-was)/8:sample),memory_order_relaxed);
}
/* counter `field` of set `ring` moved by v by its one writer, a load and a store (no read-modify-write): a port child's
   set its child's, the parent's `refreshes` its policy's thread's and its `lost` its event loop's */
#define MESH_SWITCH_COUNT(sw,ring,field,v) do{ const uint32_t r_=(ring); if(r_<MESH_SWITCH_LOGS){ _Atomic uint64_t *c_=&(sw)->counters[r_].field; \
  atomic_store_explicit(c_,atomic_load_explicit(c_,memory_order_relaxed)+(uint64_t)(v),memory_order_relaxed);} }while(0)
static inline struct mesh_switch *mesh_switch_of(struct hdr *m){ return (struct mesh_switch *)((char *)m+m->switch_off); }
static inline uint64_t mesh_switch_pack(float a,float b){ uint32_t x,y;memcpy(&x,&a,4);memcpy(&y,&b,4);return (uint64_t)x|(uint64_t)y<<32; }
static inline void mesh_switch_unpack(uint64_t v,float *a,float *b){ const uint32_t x=(uint32_t)v,y=(uint32_t)(v>>32);memcpy(a,&x,4);memcpy(b,&y,4); }
/* the frames of a port's lane j, every SEND's and receive's on it (both ends' alike, TN3205): its lane 0 short
   (MESH_CLASS_SHORT), every other long (MESH_CLASS_LONG) */
static inline uint32_t mesh_switch_lane_frames(uint32_t j){ return j?MESH_CLASS_LONG:MESH_CLASS_SHORT; }
/* the bytes every SEND and receive on the switch's lane l (port p's lane j: p 2 MESH_LINK_STREAMS + j) covers: its
   frames', a constant of its index */
static inline uint64_t mesh_switch_lane_bytes(uint32_t l){ return (uint64_t)mesh_switch_lane_frames(l%(2*MESH_LINK_STREAMS))*MESH_FRAME; }
/* a port's lanes of `frames` frames, bits over its lanes as a plan names them, so a draw finds a class's lanes of a
   port from its line alone, no lane read */
static inline uint32_t mesh_switch_class_plan(uint32_t frames){
  uint32_t plan=0;
  for(uint32_t j=0;j<2*MESH_LINK_STREAMS;j++)if(mesh_switch_lane_frames(j)==frames)plan|=UINT32_C(1)<<j;
  return plan;
}
/* the published copy of the view or the split tables by its version (one acquire load: the copy's writes before its
   publish seen): 0 or 1 */
static inline uint32_t mesh_switch_published(const _Atomic uint64_t *version){
  return (uint32_t)(atomic_load_explicit(version,memory_order_acquire)&1);
}

/* the switch made new at the bridge's start, every ring empty and every port STARTED (no ring taken until a port's
   child serves): nothing of a bridge before survives its stop */
static inline void mesh_switch_init(struct mesh_switch *sw){
  memset(sw,0,sizeof *sw);
  sw->magic=MESH_SWITCH_MAGIC;
  atomic_thread_fence(memory_order_release);
}

/* ---- The inbound rings and the in-flight rings ------------------------------------------------------------------- */
/* port `target`'s inbound ring of class c from port `pusher` (each index within its array) */
static inline struct mesh_switch_inbound *mesh_switch_inbound_of(struct mesh_switch *sw,uint32_t pusher,uint32_t target,uint32_t c){
  return &sw->inbound[pusher%MESH_PORTS][target%MESH_PORTS][c%MESH_CLASSES];
}
/* page pd pushed by port `pusher`'s child onto port `target`'s inbound ring of class c, at the pusher's clock `pushed`:
   its entry written, then the tail moved (release), so the consumer that reads the tail reads the entry whole.  The
   ring never fills (MESH_SWITCH_INBOUND's assertion), so nothing is tested */
static inline void mesh_switch_push(struct mesh_switch *sw,uint32_t pusher,uint32_t target,uint32_t c,uint32_t pd,uint64_t pushed){
  struct mesh_switch_inbound *r=mesh_switch_inbound_of(sw,pusher,target,c);
  const uint64_t tail=atomic_load_explicit(&r->tail,memory_order_relaxed);
  r->entry[tail%MESH_SWITCH_INBOUND]=(struct mesh_switch_entry){pushed,pd,0};
  atomic_store_explicit(&r->tail,tail+1,memory_order_release);
}
/* the pages on port `target`'s inbound rings (every pusher's, both classes) given back: each one's frames off the cable's
   queue, its use dropped, counted `lost` in set `ring` (the caller's counters), and each ring's head moved to its tail
   as read.  By the parent once the port's child has exited (switch_child_free, region_adopt) and by the port's next
   child before it serves (mesh-link.c lanes_open: what was pushed after the last one's end); a push that comes after
   the tail's read stays for the ring's next consumer */
static inline void mesh_switch_inbound_drop(struct hdr *m,struct mesh_switch *sw,uint32_t target,uint32_t ring){
  struct mesh_switch_cable *x=sw->cable+target%MESH_PORTS;
  for(uint32_t q=0;q<MESH_PORTS;q++)for(uint32_t c=0;c<MESH_CLASSES;c++){
    struct mesh_switch_inbound *r=mesh_switch_inbound_of(sw,q,target,c);
    const uint64_t tail=atomic_load_explicit(&r->tail,memory_order_acquire),bytes=(uint64_t)mesh_switch_class_frames(c)*MESH_FRAME;
    for(uint64_t at=atomic_load_explicit(&r->head,memory_order_relaxed);at<tail;at++){
      atomic_fetch_sub_explicit(&x->queued,bytes,memory_order_relaxed);
      mesh_pd_drop(m,r->entry[at%MESH_SWITCH_INBOUND].pd);
      MESH_SWITCH_COUNT(sw,ring,lost,1);
    }
    atomic_store_explicit(&r->head,tail,memory_order_relaxed);
  }
}
/* lane l's SENDs posted and not completed given back, its child exited (the parent's: switch_child_free,
   region_adopt): each one's frames off its cable's queue, its page's use dropped, counted `lost` in set `ring`; its
   completed count its posted.  Its device's objects went with its child, so no SEND of them reads a page again */
static inline void mesh_switch_flight_drop(struct hdr *m,struct mesh_switch *sw,uint32_t l,uint32_t ring){
  struct mesh_switch_lane *x=sw->lane+l%MESH_SWITCH_LANES;
  const uint64_t posted=atomic_load_explicit(&x->posted,memory_order_acquire),wire=mesh_switch_lane_bytes(l);
  for(uint64_t at=atomic_load_explicit(&x->completed,memory_order_relaxed);at<posted;at++){
    atomic_fetch_sub_explicit(&mesh_switch_cable_of(sw,l)->queued,wire,memory_order_relaxed);
    mesh_pd_drop(m,x->flight[at%MESH_SWITCH_FLIGHT].pd);
    MESH_SWITCH_COUNT(sw,ring,lost,1);
  }
  atomic_store_explicit(&x->completed,posted,memory_order_relaxed);
}

/* ---- The view and the split tables: two copies each, read by all ------------------------------------------------
   Each is written by one thread of the parent alone: the view by its event loop as it writes the map (mesh-link.c
   switch_view: mesh_switch_view_set), the split tables by its policy's thread while a theta is loaded (switch_watch:
   mesh_switch_policy_refresh), so the copy each publishes stands until its next write of that copy.  The lines are
   apart from them: a remote link's as this node took it, under its ends' node table entries (struct mesh_switch
   `heard`), written in place by each port child's landings (mesh_switch_lines_take); this node's own links' are kept
   nowhere, each forward making them from the cables its draw read (mesh_switch_lines_give) */
/* node's view index: under v->nodes, so under MESH_SWITCH_N; UINT32_MAX where the view has no such node */
static inline uint32_t mesh_switch_index(const struct mesh_switch_view *v,uint32_t node){
  for(uint32_t i=0;i<v->nodes;i++)if(v->node[i]==node)return i;
  return UINT32_MAX;
}
static inline struct mesh_switch_link *mesh_switch_link_of(struct mesh_switch_view *v,uint32_t from,uint32_t to){
  for(uint32_t l=0;l<v->links;l++)if(v->link[l].from==from && v->link[l].to==to)return v->link+l;
  return NULL;
}
/* the line this node keeps of the remote link from view node a to view node b (a not this node), under their node
   table entries; NULL where either is no view node or has no entry */
static inline struct mesh_switch_heard *mesh_switch_heard_at(struct mesh_switch *sw,const struct mesh_switch_view *v,uint32_t a,uint32_t b){
  if(a>=MESH_SWITCH_N || b>=MESH_SWITCH_N)return NULL;
  const uint32_t sa=v->slot[a],sb=v->slot[b];
  return sa<MESH_NODES && sb<=MESH_NODES?&sw->heard[sa][sb]:NULL;
}
/* the same by the link's ends' node numbers, `from` -> `to` (a header's line names them so) */
static inline struct mesh_switch_heard *mesh_switch_heard_of(struct mesh_switch *sw,const struct mesh_switch_view *v,uint32_t from,uint32_t to){
  return mesh_switch_heard_at(sw,v,mesh_switch_index(v,from),mesh_switch_index(v,to));
}
/* view link x's line as this node keeps it, by the link's ends' view indices (no node number looked up) */
static inline struct mesh_switch_heard *mesh_switch_heard_link(struct mesh_switch *sw,const struct mesh_switch_view *v,const struct mesh_switch_link *x){
  return mesh_switch_heard_at(sw,v,x->ends&255,x->ends>>8&255);
}
/* the view made from the fabric as the records give it, in the copy not published, then published: `nodes` node
   numbers (unsorted, at most MESH_SWITCH_N of them taken), each with its node table entry (`slots`: struct
   mesh_switch_view `slot`), and `links` directed pairs (node numbers, at most MESH_SWITCH_LINKS), each its cables,
   rate (bytes a ns, every cable) and latency (ns); the policy's features' time scale made from them.  A remote link
   the published view lacks starts with no line kept (its stamp 0): what this node kept of it is from before it left
   the view, its sender's word on a link since lost (INFINITY, the link down, among them), which no later line of the
   sender's may have replaced */
struct mesh_switch_fabric_link { uint32_t from,to,cables; double rate,latency; };
static inline double mesh_switch_time_scale(const struct mesh_switch_view *v,double page);
static inline void mesh_switch_view_set(struct mesh_switch *sw,uint32_t self,const uint32_t *nodes,const uint32_t *slots,uint32_t n,
                                        const struct mesh_switch_fabric_link *links,uint32_t count){
  const uint64_t version=atomic_load_explicit(&sw->view_version,memory_order_relaxed);
  const struct mesh_switch_view *was=sw->view+(version&1);
  struct mesh_switch_view *v=sw->view+((version+1)&1);
  uint32_t sorted[MESH_SWITCH_N],slot[MESH_SWITCH_N],k=0;
  for(uint32_t i=0;i<n && k<MESH_SWITCH_N;i++){
    uint32_t j=k;while(j && sorted[j-1]>nodes[i]){sorted[j]=sorted[j-1];slot[j]=slot[j-1];j--;}
    sorted[j]=nodes[i];slot[j]=slots[i];k++;
  }
  v->self=self;v->nodes=k;v->ranks=was->ranks;memcpy(v->node,sorted,k*sizeof *sorted);memcpy(v->slot,slot,k*sizeof *slot);
  uint32_t made=0;
  for(uint32_t l=0;l<count && made<MESH_SWITCH_LINKS;l++){
    struct mesh_switch_link *x=v->link+made++;
    x->from=links[l].from;x->to=links[l].to;x->cables=links[l].cables?links[l].cables:1;x->rate=links[l].rate;x->latency=links[l].latency;
    x->ends=(mesh_switch_index(v,x->from)&255)|(mesh_switch_index(v,x->to)&255)<<8;
  }
  v->links=made;
  for(uint32_t l=0;l<made;l++){
    const struct mesh_switch_link *x=v->link+l;
    struct mesh_switch_heard *h=mesh_switch_heard_link(sw,v,x);
    if(h && !mesh_switch_link_of((struct mesh_switch_view *)was,x->from,x->to))atomic_store_explicit(&h->stamp,0,memory_order_relaxed);
  }
  /* each node's reach avoiding each set: a breadth-first search over the links' ends */
  uint32_t next[MESH_SWITCH_N]={0};
  for(uint32_t l=0;l<made;l++){const uint32_t a=v->link[l].ends&255,b=v->link[l].ends>>8&255;if(a<k && b<k)next[a]|=1u<<b;}
  memset(v->reach,0,sizeof v->reach);
  for(uint32_t avoid=0;avoid<(1u<<k);avoid++)
    for(uint32_t from=0;from<k;from++){
      uint32_t seen=1u<<from,frontier=seen;
      while(frontier){
        uint32_t reached=0;
        for(uint32_t a=0;a<k;a++)if(frontier>>a&1)reached|=next[a];
        reached&=~avoid&~seen;seen|=reached;frontier=reached;
      }
      v->reach[avoid][from]=(uint8_t)seen;
    }
  v->scale=mesh_switch_time_scale(v,ROUTE_PAGE);
  atomic_store_explicit(&sw->view_version,version+1,memory_order_release);
}
/* remote view link x's time for a page of class c at this node's uptime `now` (CLOCK_UPTIME_RAW, ns), from its line as
   this node keeps it (`h`, NULL: none), and *idle its idle term (the page's own time on its sender's quickest cable
   with nothing queued; the view's nominal, the page's frames over the link's rate a cable plus its latency, where the
   line measured none).  A line's cost is its idle term and the backlog ahead of the page, and a backlog drains a
   nanosecond each nanosecond, so the link counts at its idle term plus what the line's cost had beyond it less the
   time since this node took the line (`taken`), and never below its idle term: a link whose traffic ended counts at
   its idle term once its last line's backlog has drained.  Each time is on this node's own clock; no node's stamp is
   read against another's.  INFINITY where the line says the link is down (until a later line of its sender's, or its
   return to the view: mesh_switch_view_set).  The nominal stands for both where there is no line */
static inline double mesh_switch_line_cost(const struct mesh_switch_heard *h,const struct mesh_switch_link *x,uint32_t c,uint64_t now,double *idle){
  const uint64_t stamp=h?atomic_load_explicit(&h->stamp,memory_order_relaxed):0;
  const double nominal=(double)mesh_switch_class_frames(c)*MESH_FRAME/(x->rate/x->cables)+x->latency,
               measured=stamp?atomic_load_explicit(&h->idle[c],memory_order_relaxed):0;
  *idle=measured>0?measured:nominal;
  if(!stamp)return *idle;
  const uint64_t taken=atomic_load_explicit(&h->taken,memory_order_relaxed);
  const double left=(double)atomic_load_explicit(&h->cost[c],memory_order_relaxed)-*idle-(now>taken?(double)(now-taken):0);
  return left>0?*idle+left:*idle;
}

/* ---- The estimators ---------------------------------------------------------------------------------------------
   A lane's SEND completed: `bytes`, posted at `posted`, its service beginning at `start` (its post, or the lane's
   previous completion if later) and done at `done` (ns, the child's clock): its cable's queue less its bytes, its sums,
   and its estimate moved (an exponential average of the service rate, 1/8 a SEND, and of the wait) */
static inline void mesh_switch_sent(struct mesh_switch *sw,uint32_t lane,uint64_t bytes,uint64_t posted,uint64_t start,uint64_t done){
  struct mesh_switch_lane *x=sw->lane+lane;
  atomic_fetch_sub_explicit(&mesh_switch_cable_of(sw,lane)->queued,bytes,memory_order_relaxed);
  atomic_fetch_add_explicit(&x->sends,1,memory_order_relaxed);atomic_fetch_add_explicit(&x->bytes,bytes,memory_order_relaxed);
  const uint64_t service=done>start?done-start:1,wait=start>posted?start-posted:0;
  atomic_fetch_add_explicit(&x->service,service,memory_order_relaxed);atomic_fetch_add_explicit(&x->wait,wait,memory_order_relaxed);
  float rate,waited;mesh_switch_unpack(atomic_load_explicit(&x->estimate,memory_order_relaxed),&rate,&waited);
  const float r=(float)bytes/(float)service;
  rate=rate>0?rate+(r-rate)/8:r;waited+=((float)wait-waited)/8;
  atomic_store_explicit(&x->estimate,mesh_switch_pack(rate,waited),memory_order_relaxed);
}
/* Port p's cable's terms (struct mesh_switch_cable `rate`, `page`) from one poll of its SEND completions, by its child:
   the poll read at `done` (its clock, read once a poll), completing `bytes` in all, the earliest of them posted at
   `first`, the cable's previous poll that completed any at `last` (0: none yet); and, where the poll completed a SEND
   that had the cable to itself (posted no sooner than `last`, the only one outstanding as the poll began), its class
   `alone` (else UINT32_MAX) and its post `posted`.
   - The rate: the poll's bytes over the time since `last`, where the earliest was posted before `last` (the cable was
     busy the whole while): the rate a backlog drains at, all its lanes at once.
   - The page's time: that SEND's post to its completion: a page's own time on the cable when nothing is ahead of it
     (on node B, 2026-11-23: 2.7-3.1 us a short SEND and 6.0-6.4 us a long one on the Studios' links, 4.4-5.7 us and
     6.1-8.0 us on the mini's), an average of such SENDs from the time the cable's rate gives the page.  It was the
     first sample as it came, and a page alone after a quiet stretch pays a quiet link's first SEND (design/quiet-link.md:
     1.8-4.9 us more at the median): B's port 4 toward D held 27750 ns for a short page from one sample, B->D was priced
     27.75 us, and no short page went from B to D direct (every node's 4-rank 6 KB all-reduce 17.0 to 19.1 us).  From
     the rate's 0.88 us that sample moves it to 4.2 us, so the link is still drawn and its later samples bring it down.
     The least time of any SEND is no bound on it: completions are polled 2-8 clock ticks (83-333 ns) after a 36 KiB
     SEND's post, which priced every link near nothing and sent 72,854 of a 2-rank 393 KB all-gather's 88,000 pages
     through relays (26.2 to 41.8 us).
   The SENDs a poll completes share its one clock read, so neither is read from a single SEND's service in a busy poll
   (a lane's second completion in one poll has none: mesh_switch_sent's `service` counts it 1 ns), and one SEND's
   service on a lane is its share of the cable while the lanes beside it are busy, not the cable's rate */
static inline void mesh_switch_cable_sent(struct mesh_switch *sw,uint32_t p,uint64_t bytes,uint64_t first,uint64_t last,uint64_t done,
                                          uint32_t alone,uint64_t posted){
  struct mesh_switch_cable *x=sw->cable+p%MESH_PORTS;
  if(bytes && last && first<last && done>last)mesh_switch_average(&x->rate,(double)bytes/(double)(done-last));
  /* the page's time an average that starts from the time the measured rate gives the page (its bytes over the rate,
     the stand-in a draw reads before any sample: mesh_switch_terms_read) and moves an eighth of the way a sample, so
     one sample moves it an eighth: a sample is taken once the rate is measured */
  const float rate=atomic_load_explicit(&x->rate,memory_order_relaxed);
  if(alone<MESH_CLASSES && done>posted && rate>0){
    const float was=atomic_load_explicit(&x->page[alone],memory_order_relaxed);
    const double from=was>0?was:(double)mesh_switch_class_frames(alone)*MESH_FRAME/rate;
    atomic_store_explicit(&x->page[alone],(float)(from+((double)(done-posted)-from)/8),memory_order_relaxed);
  }
}
/* where a header's lines lie: after its payload of `bytes` (8-aligned), within the SEND's whole frames, so they cost
   no wire; and how many fit in `wire` bytes.  `bytes` is the payload's as the caller read and bounded it once, never
   the header's word read again: every client maps the region writable */
static inline struct mesh_switch_line *mesh_switch_lines_at(struct mesh_head *head,uint32_t bytes){
  return (struct mesh_switch_line *)((char *)head+MESH_HEAD+((uint64_t)bytes+7)/8*8);
}
static inline uint32_t mesh_switch_lines_fit(uint32_t bytes,uint64_t wire){
  const uint64_t at=MESH_HEAD+((uint64_t)bytes+7)/8*8,room=wire>at?(wire-at)/sizeof(struct mesh_switch_line):0;
  return room<MESH_SWITCH_LINES?(uint32_t)room:MESH_SWITCH_LINES;
}
/* the lines a header carries (its `lines`, read once, as many as fit after `bytes` in `wire`) taken where newer than
   the one this node keeps of the link (struct mesh_switch `heard`), each with this node's uptime `now` as it took it:
   a link's line is only ever made by its sender, so a stamp is only ever compared with its sender's own, and a sender
   restarted, its boot later, is taken at once.  A node's lines about its own links are its own measures, never
   taken */
static inline void mesh_switch_lines_take(struct mesh_switch *sw,struct mesh_head *head,uint32_t bytes,uint64_t wire,uint64_t now){
  const struct mesh_switch_view *v=sw->view+mesh_switch_published(&sw->view_version);
  const uint32_t carried=__atomic_load_n(&head->lines,__ATOMIC_RELAXED),fit=mesh_switch_lines_fit(bytes,wire),lines=carried<fit?carried:fit;
  const struct mesh_switch_line *line=mesh_switch_lines_at(head,bytes);
  for(uint32_t i=0;i<lines;i++){
    const struct mesh_switch_line *a=line+i;
    if(a->from==v->self)continue;
    struct mesh_switch_heard *h=mesh_switch_heard_of(sw,v,a->from,a->to);
    if(!h)continue;
    uint64_t was=atomic_load_explicit(&h->stamp,memory_order_relaxed);
    while(a->stamp>was)
      if(atomic_compare_exchange_weak_explicit(&h->stamp,&was,a->stamp,memory_order_relaxed,memory_order_relaxed)){
        for(uint32_t c=0;c<MESH_CLASSES;c++){
          atomic_store_explicit(&h->cost[c],a->cost[c],memory_order_relaxed);atomic_store_explicit(&h->idle[c],a->idle[c],memory_order_relaxed);
        }
        atomic_store_explicit(&h->taken,now,memory_order_relaxed);
        break;
      }
  }
}
/* the calling thread's own port (a port child's progress thread, mesh-link.c port_progress; UINT32_MAX elsewhere): of
   cables whose times are equal, its own is posted by this thread in this pass, any other's waits for its own child's
   pass (the handoff through the inbound ring) */
static __thread uint32_t mesh_switch_home=UINT32_MAX;
/* every port's line read once (acquire: what its child did before it served, its inbound rings emptied, is seen) */
static inline void mesh_switch_ports_read(struct mesh_switch *sw,uint64_t *line){
  for(uint32_t p=0;p<MESH_PORTS;p++)line[p]=atomic_load_explicit(&sw->port[p],memory_order_acquire);
}
/* a serving cable's terms as a draw reads them (mesh_switch_terms_read): the time its bytes queued take to drain
   (`backlog`, ns: over its own measured rate) and, for each class, a page's own time on its link (`own`, ns): the
   least measured time alone over the serving cables toward the same peer, every cable of a link the same one.  A
   cable's time alone is measured only from a SEND it carried alone (mesh_switch_cable_sent), so a cable kept by its
   own time is drawn by none and its measure never made again: B's port 3 toward A held 8958 ns for a short page against
   2696-2970 ns on its two siblings, and B's ports 0 and 1 toward A, the same cable, differed by 10% (2970 against
   2696), so every page went to the lesser one whichever child took it, each one through an inbound ring to another
   child's pass (2-rank 6 KB calls 0.6-0.9 us slower, the post wait 0.47 to 0.91 us).  Of one link's cables the times
   now differ by their backlogs alone, so at none the taking child's own port is drawn (mesh_switch_cables_toward) and
   the page is posted in the pass that took it.  Where a measure is not made yet, its link's nominal ones in the view (a
   cable's share of the link's rate, and its latency) stand in: the nominal rate for the rate; for a page's time, its
   frames over the measured rate where there is one (a long page alone takes about that: 3.92-3.96 us against 36 KiB at
   9.25 GB/s), else over the nominal rate plus the nominal latency */
struct mesh_switch_terms { double backlog,own[MESH_CLASSES]; };
static inline void mesh_switch_terms_read(struct mesh_switch *sw,const struct mesh_switch_view *v,const uint64_t *line,struct mesh_switch_terms *t){
  double alone[MESH_PORTS][MESH_CLASSES];
  for(uint32_t p=0;p<MESH_PORTS;p++){
    if(mesh_port_state(line[p])!=CHILD_SERVING)continue;
    const struct mesh_switch_cable *x=sw->cable+p;
    const struct mesh_switch_link *k=mesh_switch_link_of((struct mesh_switch_view *)v,v->self,mesh_port_peer(line[p]));
    const double nominal=k && k->rate>0?k->rate/k->cables:0,latency=k?k->latency:0,measured=atomic_load_explicit(&x->rate,memory_order_relaxed);
    const double rate=measured>0?measured:nominal>0?nominal:1;
    t[p].backlog=(double)atomic_load_explicit(&x->queued,memory_order_relaxed)/rate;
    for(uint32_t c=0;c<MESH_CLASSES;c++){
      const double page=atomic_load_explicit(&x->page[c],memory_order_relaxed),bytes=(double)mesh_switch_class_frames(c)*MESH_FRAME;
      alone[p][c]=page>0?page:INFINITY;
      t[p].own[c]=measured>0?bytes/measured:bytes/rate+latency;
    }
  }
  /* each cable's own time its link's: the least measured over the serving cables toward its peer, else its stand-in */
  for(uint32_t p=0;p<MESH_PORTS;p++){
    if(mesh_port_state(line[p])!=CHILD_SERVING)continue;
    for(uint32_t c=0;c<MESH_CLASSES;c++){
      double least=INFINITY;
      for(uint32_t q=0;q<MESH_PORTS;q++)
        if(mesh_port_state(line[q])==CHILD_SERVING && mesh_port_peer(line[q])==mesh_port_peer(line[p]) && alone[q][c]<least)least=alone[q][c];
      if(least<INFINITY)t[p].own[c]=least;
    }
  }
}
/* Toward `peer`, of the cables whose port serves toward it (`line`: every port's line as the caller read it, once; `t`
   their terms, mesh_switch_terms_read) with lanes of `frames` frames (a page goes on the lanes of its payload's class
   alone), the one a page of that class lands through soonest: its backlog, then the page's own time on it (the design's
   (backlog + bytes) / rate, each term its cable's measure); of equal times this thread's own port's, then the one of
   the lesser backlog.  That time (ns) and *cable its port; INFINITY and UINT32_MAX where no cable serves toward the
   peer.  The lane on it is its child's to choose as it posts (mesh-link.c switch_sends) */
static inline double mesh_switch_cables_toward(const uint64_t *line,const struct mesh_switch_terms *t,uint32_t peer,uint32_t frames,uint32_t *cable){
  double least=INFINITY;uint32_t q=UINT32_MAX;
  const uint32_t of_class=mesh_switch_class_plan(frames),c=mesh_switch_class(frames);
  for(uint32_t p=0;p<MESH_PORTS;p++){
    if(mesh_port_state(line[p])!=CHILD_SERVING || mesh_port_peer(line[p])!=peer || !(mesh_port_plan(line[p])&of_class))continue;
    const double time=t[p].backlog+t[p].own[c];
    const int home=p==mesh_switch_home,best_home=q==mesh_switch_home;
    if(time<least || (time==least && q!=UINT32_MAX && (home>best_home || (home==best_home && t[p].backlog<t[q].backlog)))){least=time;q=p;}
  }
  *cable=q;
  return q==UINT32_MAX?INFINITY:least;
}
/* this node's own link `self` -> `peer` as a line, stamped `stamp`, from its cables (`line`, `t` as the caller read
   them): for each class, the least time over the cables serving toward the peer with that class's lanes
   (mesh_switch_cables_toward's) and the least own time; INFINITY and 0 where none serves (the link down) */
static inline struct mesh_switch_line mesh_switch_own_line(const uint64_t *line,const struct mesh_switch_terms *t,uint32_t self,uint32_t peer,uint64_t stamp){
  struct mesh_switch_line a={self,peer,stamp,{INFINITY,INFINITY},{0,0}};
  for(uint32_t p=0;p<MESH_PORTS;p++){
    if(mesh_port_state(line[p])!=CHILD_SERVING || mesh_port_peer(line[p])!=peer)continue;
    for(uint32_t c=0;c<MESH_CLASSES;c++){
      if(!(mesh_port_plan(line[p])&mesh_switch_class_plan(mesh_switch_class_frames(c))))continue;
      const float time=(float)(t[p].backlog+t[p].own[c]),own=(float)t[p].own[c];
      if(time<a.cost[c])a.cost[c]=time;
      if(!(a.idle[c]>0) || own<a.idle[c])a.idle[c]=own;
    }
  }
  return a;
}
/* the lines a header sends, at most `fit` (as many as its frames' slack holds: mesh_switch_lines_fit), so a full page
   makes none: this node's own links first, each view link from this node as its draw read its cables (`line`, `t`:
   mesh_switch_own_line) stamped with this node's stamp of its uptime `now`, then the remote links' lines this node
   keeps (struct mesh_switch `heard`), each its sender's stamp and its cost as this node counts it at `now`
   (mesh_switch_line_cost: its backlog less the time this node has held it), in the view's order */
static inline uint32_t mesh_switch_lines_give(struct mesh_switch *sw,const struct mesh_switch_view *v,const uint64_t *line,const struct mesh_switch_terms *t,
                                              uint64_t now,uint32_t fit,struct mesh_switch_line *out){
  const uint64_t stamp=mesh_switch_stamp(sw,now);
  uint32_t n=0;
  for(uint32_t l=0;l<v->links && n<fit;l++)
    if(v->link[l].from==v->self)out[n++]=mesh_switch_own_line(line,t,v->self,v->link[l].to,stamp);
  for(uint32_t l=0;l<v->links && n<fit;l++){
    const struct mesh_switch_link *x=v->link+l;
    const struct mesh_switch_heard *h=x->from==v->self?NULL:mesh_switch_heard_link(sw,v,x);
    const uint64_t s=h?atomic_load_explicit(&h->stamp,memory_order_relaxed):0;
    if(!s)continue;
    struct mesh_switch_line *a=out+n++;
    a->from=x->from;a->to=x->to;a->stamp=s;
    for(uint32_t c=0;c<MESH_CLASSES;c++){
      double idle;
      a->cost[c]=(float)mesh_switch_line_cost(h,x,c,now,&idle);a->idle[c]=atomic_load_explicit(&h->idle[c],memory_order_relaxed);
    }
  }
  return n;
}
/* view link l's time for a page of class c at this node's uptime `now` (*idle its time with nothing queued): this
   node's own link's from its cables (`line`, `t` as the caller read them: mesh_switch_own_line, INFINITY where none
   serves), another's from its line as this node keeps it (mesh_switch_line_cost) */
static inline double mesh_switch_link_cost(struct mesh_switch *sw,const struct mesh_switch_view *v,const uint64_t *line,const struct mesh_switch_terms *t,
                                           uint32_t l,uint32_t c,uint64_t now,double *idle){
  const struct mesh_switch_link *x=v->link+l;
  if(x->from!=v->self)return mesh_switch_line_cost(mesh_switch_heard_link(sw,v,x),x,c,now,idle);
  const struct mesh_switch_line a=mesh_switch_own_line(line,t,x->from,x->to,0);
  *idle=a.idle[c];
  return a.cost[c];
}

/* ---- The cost-to-go: each poller's own, a step an idle pass -----------------------------------------------------
   The cost-to-go between view nodes for each class of page (`to`, ns; 1e12 where no path is), which a port child's
   draws read (mesh_switch_choose), in the child's own memory: Floyd-Warshall over the view's links between nodes other
   than this one (a page that comes back here is on no path: a draw's paths avoid the nodes a page visited, this node
   among them), each link's cost the time a page of the class takes through it as the class's step began
   (mesh_switch_line_cost: its line's idle term and what is left of its backlog, else the view's nominal terms).  Made
   in `made` a step at a time (mesh_switch_togo_step: for each class, the links' costs, then the paths through each node
   in turn), copied into `to` as a cycle ends, then made again; made anew from its first step where the view's version
   moved (`view`: the version the cycle is over, plus 1) */
struct mesh_switch_togo {
  float to[MESH_CLASSES][MESH_SWITCH_N][MESH_SWITCH_N],made[MESH_CLASSES][MESH_SWITCH_N][MESH_SWITCH_N];
  uint64_t view;
  uint32_t step;
};
/* one step of the cost-to-go `g` over the published view: a cycle is 2 (n + 1) steps for a view of n nodes, each
   class's links' costs (n^2 stores, a kept line read a link and its age on this node's clock, one clock read)
   and then its paths through each of the n nodes (n^2 additions each), tens of nanoseconds a step (an estimate), so a
   port child's idle pass takes one (mesh-link.c port_pass) and no pass grows past its usual spread.  1 where the step
   ended a cycle (`to` made new) */
static inline int mesh_switch_togo_step(struct mesh_switch *sw,struct mesh_switch_togo *g){
  const uint64_t version=atomic_load_explicit(&sw->view_version,memory_order_acquire);
  const struct mesh_switch_view *v=sw->view+(version&1);
  const uint32_t nodes_read=__atomic_load_n(&v->nodes,__ATOMIC_RELAXED),n=nodes_read<MESH_SWITCH_N?nodes_read:MESH_SWITCH_N;
  const uint32_t self=mesh_switch_index(v,v->self);
  if(g->view!=version+1){g->view=version+1;g->step=0;}
  const uint32_t c=g->step/(n+1)%MESH_CLASSES,k=g->step%(n+1);
  float (*d)[MESH_SWITCH_N]=g->made[c];
  if(!k){
    for(uint32_t i=0;i<MESH_SWITCH_N;i++)for(uint32_t j=0;j<MESH_SWITCH_N;j++)d[i][j]=i==j?0:INFINITY;
    const uint64_t now=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    for(uint32_t l=0;l<v->links && l<MESH_SWITCH_LINKS;l++){
      const struct mesh_switch_link *x=v->link+l;
      const uint32_t a=x->ends&255,b=x->ends>>8&255;
      if(a>=n || b>=n || a==self || b==self)continue;
      double idle;const double cost=mesh_switch_line_cost(mesh_switch_heard_link(sw,v,x),x,c,now,&idle);
      if(cost<d[a][b])d[a][b]=(float)cost;
    }
  }else if(k-1!=self){
    const uint32_t via=k-1;
    for(uint32_t i=0;i<n;i++){
      const float first=d[i][via];
      if(!(first<INFINITY))continue;
      for(uint32_t j=0;j<n;j++){const float through=first+d[via][j];if(through<d[i][j])d[i][j]=through;}
    }
  }
  if(++g->step<MESH_CLASSES*(n+1))return 0;
  for(uint32_t cc=0;cc<MESH_CLASSES;cc++)for(uint32_t i=0;i<MESH_SWITCH_N;i++)for(uint32_t j=0;j<MESH_SWITCH_N;j++)
    g->to[cc][i][j]=g->made[cc][i][j]<INFINITY?g->made[cc][i][j]:1e12f;
  g->step=0;
  return 1;
}
/* the cost-to-go made whole at once, a cycle of its steps: the policy's thread's, which has no idle passes */
static inline void mesh_switch_togo_whole(struct mesh_switch *sw,struct mesh_switch_togo *g){
  g->step=0;
  while(!mesh_switch_togo_step(sw,g)){}
}

/* ---- The draw ---------------------------------------------------------------------------------------------------- */
static inline uint64_t mesh_switch_mix(uint64_t x){
  x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);
}
static inline double mesh_switch_uniform(uint64_t key){ return ((mesh_switch_mix(key)>>11)+0.5)/9007199254740992.0; }
/* whether view node `from` reaches `want` without entering `avoid` (bits over view indices), over the view's links */
static inline int mesh_switch_reaches(const struct mesh_switch_view *v,uint32_t from,uint64_t avoid,uint32_t want){
  if(from==want)return 1;
  if(from>=MESH_SWITCH_N || want>=MESH_SWITCH_N)return 0;
  return v->reach[avoid&((1u<<MESH_SWITCH_N)-1)][from]>>want&1;
}
/* The policy's contexts at this node (the order of a fabric's contexts the policy was trained in, restricted to this
   node): for each rank a in view order
   its patterns, a -> {b} for each other rank b, then a -> every other rank; for each, every set S of the pattern's
   destinations without this node, ascending as bits.  Returns their count (at most MESH_SWITCH_CONTEXTS) */
static inline uint32_t mesh_switch_ranks(const struct mesh_switch_view *v){ return v->ranks?v->ranks:(uint32_t)((UINT64_C(1)<<v->nodes)-1); }
static inline uint32_t mesh_switch_contexts(const struct mesh_switch_view *v,struct mesh_switch_context *out){
  const uint32_t self=mesh_switch_index(v,v->self),ranks=mesh_switch_ranks(v);
  uint32_t n=0,j=0,deg=0;
  for(uint32_t l=0;l<v->links;l++)deg+=v->link[l].from==v->self;
  for(uint32_t a=0;a<v->nodes;a++){
    if(!(ranks>>a&1))continue;
    for(uint32_t pass=0;pass<=v->nodes;pass++){
      uint32_t want=0;
      if(pass<v->nodes){if(pass==a || !(ranks>>pass&1))continue;want=1u<<pass;}
      else want=ranks&~(1u<<a);
      for(uint32_t S=1;S<(1u<<v->nodes) && deg && self<32;S++){
        if((S&~want) || (S>>self&1))continue;
        if(n<MESH_SWITCH_CONTEXTS)out[n++]=(struct mesh_switch_context){a,want,S,j};
      }
      j++;
    }
  }
  return n;
}
/* a header's destinations' nodes as bits over view v */
static inline uint32_t mesh_switch_dest_nodes(const struct mesh_switch_view *v,const struct mesh_head *head){
  uint32_t bits=0;
  for(uint32_t i=0;i<head->dests && i<MESH_DESTS;i++){const uint32_t d=mesh_switch_index(v,head->dest[i].node);if(d<32)bits|=1u<<d;}
  return bits;
}
/* the w of a page's context from the published tables (their copy as its version names it, read once): its source,
   its destinations' nodes (mesh_switch_dest_nodes) and the ones it carries on from here (`onward`), each over view v,
   this node's, whose indices the tables' keys are in (mesh_switch_contexts); 1 found (w filled, *context its index,
   *point the tables' refresh), 0 none or the tables not active; *scale the features' time scale (the view's).  A
   context's index stays under MESH_SWITCH_CONTEXTS whatever `contexts` the copy says */
static inline int mesh_switch_split_w(const struct mesh_switch *sw,const struct mesh_switch_view *v,const struct mesh_head *head,uint32_t onward,float *w,
                                      uint32_t *context,uint64_t *point,uint32_t *argmax,float *scale){
  const struct mesh_switch_splits *x=sw->splits+mesh_switch_published(&sw->splits_version);
  int found=0;
  if(x->active){
    const uint32_t source=mesh_switch_index(v,head->source),dest_nodes=mesh_switch_dest_nodes(v,head);
    for(uint32_t c=0;c<x->contexts && c<MESH_SWITCH_CONTEXTS;c++){
      const struct mesh_switch_context *k=x->key+c;
      if(k->source==source && k->dest_nodes==dest_nodes && k->onward==onward){memcpy(w,x->w[c],sizeof x->w[c]);*context=c;found=1;break;}
    }
  }
  *point=x->point;*argmax=x->argmax;*scale=(float)v->scale;
  return found;
}
/* split.h's table of a draw at u (inverse CDF over its valid, allowed splits; argmax: the most probable): its token,
   and the split's log-probability; UINT32_MAX where the table is empty */
static inline uint32_t mesh_switch_split_draw(const struct route_draw *d,const float *w,double u,int argmax,float *logp){
  const struct split_set a=split_set_of(d);
  if(!a.n)return UINT32_MAX;
  float most=-INFINITY;uint32_t best=0;double total=0;
  struct split_walk k=split_walk_at(&a,0,a.n);
  for(uint32_t i=0;k.left;i++,split_walk_next(&k)){const struct split sp=split_of(d,k.token);const float lg=split_logit(split_tanh(split_dot(&sp,w)));if(lg>most){most=lg;best=i;}}
  k=split_walk_at(&a,0,a.n);
  for(;k.left;split_walk_next(&k)){const struct split sp=split_of(d,k.token);total+=exp((double)split_logit(split_tanh(split_dot(&sp,w)))-most);}
  uint32_t pick=best;
  if(!argmax){
    const double at=u*total;double acc=0;uint32_t i=0;
    k=split_walk_at(&a,0,a.n);
    for(;k.left;i++,split_walk_next(&k)){const struct split sp=split_of(d,k.token);acc+=exp((double)split_logit(split_tanh(split_dot(&sp,w)))-most);if(acc>at){pick=i;break;}}
  }
  const uint32_t tok=split_set_token(&a,pick);
  const struct split sp=split_of(d,tok);
  *logp=(float)((double)split_logit(split_tanh(split_dot(&sp,w)))-most-log(total));
  return tok;
}
/* a group of a page's destinations as a draw makes it: its peer, its cable's port, and its destinations (bits over the
   draw's `away`) */
struct mesh_switch_group { uint32_t peer,port; uint64_t dests; };
/* A page at this node (its header: its destinations and path) split over the node's out-neighbours (the view nodes
   its ports serve toward, as the port lines `line` the caller read tell them, each once, in port order) for the
   destinations it carries on (`away`, `naway` of them: places in the header's dest[]): each group (struct
   mesh_switch_group) the neighbour's cable a page of `frames` frames (the class of the page's payload) lands through
   soonest, by the cables' terms `t` the caller read, among those with lanes of those frames (mesh_switch_cables_toward),
   one group a neighbour.
   The draw's record (split.h struct route_draw, as rdma/route reads it): for each carried destination d and neighbour
   k, the arrival through k (its cable's backlog and the page's own time on it, then the cost-to-go from k to d for a
   page of its class, `togo`, the caller's: struct mesh_switch_togo), whether k reaches d without coming back here
   (valid), and whether it is allowed from the page's path (off it, reaching d avoiding it: mask).
   The draw, at u = the page's seed (its source, its `seq`, its first carried destination's index):
   - where the policy's split tables are active and have the page's context (its source, its destinations' nodes, the
     ones carried here, each over this node's view: mesh_switch_split_w), the split is split.h's table under the
     context's w at u: exactly the trainer's action, *logp its log-probability;
   - else the design's rule: each destination takes its allowed neighbour of the earliest arrival (its cable's
     backlog and the page's own time on it, then the cost-to-go from it); *logp 0 (the rule draws nothing).
   A destination no allowed neighbour reaches (or not in the view) is in no group: its bit set in *unreachable.
   `record` (NULL: none) receives the record, *context the context drawn from (UINT32_MAX: none, the rule chose),
   *point the tables' refresh.  Returns the groups.
   The view's nodes and the port lines are read once, so a bound is the same wherever it is used, and every index
   stays within its array whatever a copy rewritten under the draw has in it: a view index under `nodes` (at most
   MESH_SWITCH_N); a neighbour k under `deg` (at most MESH_SWITCH_N), each a view index under `nodes` (togo[nb] within
   togo); a cable under MESH_PORTS (mesh_switch_cables_toward's) */
static inline uint32_t mesh_switch_choose(const struct mesh_switch *sw,const struct mesh_switch_view *v,const struct mesh_switch_togo *togo,
                                          const uint64_t *line,const struct mesh_switch_terms *t,
                                          const struct mesh_head *head,const uint32_t *away,uint32_t naway,uint32_t frames,
                                          struct mesh_switch_group *group,uint32_t *unreachable,
                                          struct route_draw *record,float *logp,uint32_t *context,uint64_t *point){
  const uint32_t nodes_read=__atomic_load_n(&v->nodes,__ATOMIC_RELAXED),nodes=nodes_read<MESH_SWITCH_N?nodes_read:MESH_SWITCH_N;
  const uint32_t self=mesh_switch_index(v,v->self);
  const uint64_t here=self<64?UINT64_C(1)<<self:0;
  *unreachable=0;
  uint64_t visited=here;
  for(uint32_t h=0;h<=head->hops && h<MESH_SWITCH_N;h++){const uint32_t i=mesh_switch_index(v,head->path[h]);if(i<64)visited|=UINT64_C(1)<<i;}
  const double u=mesh_switch_uniform(((uint64_t)head->source<<32)^head->seq*0x9e3779b97f4a7c15ULL^(naway?head->dest[away[0]].index:0));
  uint32_t choice[MESH_DESTS],node[MESH_DESTS],neighbour[MESH_SWITCH_N],deg=0;
  for(uint32_t p=0;p<MESH_PORTS;p++){
    if(mesh_port_state(line[p])!=CHILD_SERVING)continue;
    const uint32_t nb=mesh_switch_index(v,mesh_port_peer(line[p]));
    uint32_t k=0;
    while(k<deg && neighbour[k]!=nb)k++;
    if(nb<nodes && k==deg && deg<MESH_SWITCH_N)neighbour[deg++]=nb;
  }
  /* each neighbour's first hop: the time a page of this class takes to land through its quickest cable; and the
     cost-to-go of this class */
  uint32_t cable[MESH_SWITCH_N];double wait[MESH_SWITCH_N];
  const float (*to)[MESH_SWITCH_N]=togo->to[mesh_switch_class(frames)];
  for(uint32_t k=0;k<deg;k++)wait[k]=mesh_switch_cables_toward(line,t,v->node[neighbour[k]],frames,cable+k);
  /* the policy's context: the page's source and destinations' nodes, the carried ones (this node's view indices) */
  uint32_t onward=0;
  for(uint32_t j=0;j<naway;j++){node[j]=mesh_switch_index(v,head->dest[away[j]].node);if(node[j]<32)onward|=1u<<node[j];}
  float cw[SPLIT_FS],scale=1;uint32_t ctx=UINT32_MAX,argmax=0;uint64_t at_point=0;
  const int split=nodes<=SPLIT_N && deg<=SPLIT_DEG && mesh_switch_split_w(sw,v,head,onward,cw,&ctx,&at_point,&argmax,&scale);
  if(!split)ctx=UINT32_MAX;
  if(context)*context=ctx;
  if(point)*point=at_point;
  struct route_draw local;
  if(!record && split)record=&local;
  const int logged=record && nodes<=SPLIT_N && deg<=SPLIT_DEG;
  if(logged){memset(record,0,sizeof *record);record->deg=deg;record->unit=scale>0?scale:1;}
  uint32_t ntok=1;
  for(uint32_t j=0;j<naway;j++){
    const uint32_t d=node[j];
    choice[j]=UINT32_MAX;
    if(d>=nodes){*unreachable|=1u<<j;continue;}
    double cost=INFINITY;uint32_t cheapest=UINT32_MAX,allowed=0;
    uint32_t ok[MESH_SWITCH_N];
    for(uint32_t k=0;k<deg;k++){
      const uint32_t nb=neighbour[k];
      ok[k]=cable[k]!=UINT32_MAX && !(visited>>nb&1) && (nb==d || mesh_switch_reaches(v,nb,visited,d));
      if(logged){
        const int valid=nb==d || mesh_switch_reaches(v,nb,here,d);
        const uint32_t b=d*SPLIT_DEG+k;
        record->arrive[b]=(float)(wait[k]+(nb==d?0:to[nb][d]));
        if(valid)record->valid|=1u<<b;
        if(ok[k])record->mask|=1u<<b;
      }
      if(!ok[k])continue;
      allowed++;
      const double c=wait[k]+(nb==d?0:to[nb][d]);
      if(c<cost){cost=c;cheapest=k;}
    }
    if(!allowed){*unreachable|=1u<<j;continue;}
    ntok*=allowed;
    if(logged)record->S|=1u<<d;
    /* the design's rule: the allowed neighbour of the earliest arrival at d, its cable's backlog and this page's own
       time on it, then the cost-to-go from it */
    if(!split)choice[j]=cheapest;
  }
  if(split && record->S){
    /* split.h's table under the context's w (every destination of S has an allowed neighbour, so the table has a
       split): the token's digit for each carried destination, in S's order */
    float lp=0;
    const uint32_t tok=mesh_switch_split_draw(record,cw,u,(int)argmax,&lp);
    uint32_t y=tok;
    for(uint32_t dd=0;dd<SPLIT_N && tok!=UINT32_MAX;dd++){
      if(!(record->S>>dd&1))continue;
      for(uint32_t j=0;j<naway;j++)if(node[j]==dd && !(*unreachable>>j&1))choice[j]=y%deg;
      y/=deg;
    }
    record->token=tok;record->ntok=ntok;
    if(logp)*logp=lp;
  }else if(!split){
    if(logged){
      uint32_t tok=0,place=1;
      for(uint32_t dd=0;dd<SPLIT_N;dd++){
        if(!(record->S>>dd&1))continue;
        for(uint32_t j=0;j<naway;j++)if(node[j]==dd && choice[j]!=UINT32_MAX){tok+=choice[j]*place;break;}
        place*=deg;
      }
      record->token=tok;record->ntok=ntok;
    }
    if(logp)*logp=0;
  }
  uint32_t used=0;
  for(uint32_t j=0;j<naway;j++){
    if(choice[j]==UINT32_MAX){*unreachable|=1u<<j;continue;}
    const uint32_t k=choice[j];
    uint32_t g=0;while(g<used && group[g].port!=cable[k])g++;
    if(g==used)group[used++]=(struct mesh_switch_group){v->node[neighbour[k]],cable[k],0};
    group[g].dests|=UINT64_C(1)<<j;
  }
  return used;
}
/* ---- The policy's interface: its tokens, its splits ------------------------------------------------------------- */
/* the policy's input at one of its refreshes, from this node's state (mesh_switch_tokens_make: route.h's features, made
   here): a token for each of this node's own links not down (lane[i], of view link link_of[i]) and for each of its
   contexts (context[c], key[c] as the split tables have it) */
struct mesh_switch_tokens {
  uint32_t lanes,contexts;
  uint64_t point,at;
  float scale,padding;
  uint32_t link_of[MESH_SWITCH_LINKS];
  float lane[MESH_SWITCH_LINKS][ROUTE_IN];
  struct mesh_switch_context key[MESH_SWITCH_CONTEXTS];
  float context[MESH_SWITCH_CONTEXTS][ROUTE_IN];
};
struct mesh_switch_policy {
  void *context;
  /* the trained policy (rdma/route): each of this node's contexts' w from its tokens (deployed: forward.c's
     route_forward_lanes over the lanes, then route_forward_context for each context, the state carried in `context`);
     0, or -1: the tables stay as they are.  Called by mesh_switch_policy_refresh */
  int (*splits)(void *context,const struct mesh_switch_tokens *tokens,float w[MESH_SWITCH_CONTEXTS][SPLIT_FS]);
};
/* earliest arrival at every view node from `from` (cost `at0`), hop costs `hop`, avoiding `avoid` */
static inline void mesh_switch_arrivals(const struct mesh_switch_view *v,uint32_t from,double at0,uint64_t avoid,const double *hop,double *at){
  uint64_t done=avoid&~(UINT64_C(1)<<from);
  for(uint32_t i=0;i<v->nodes;i++)at[i]=INFINITY;
  at[from]=at0;
  for(;;){
    uint32_t u=UINT32_MAX;
    for(uint32_t i=0;i<v->nodes;i++)if(!(done>>i&1) && isfinite(at[i]) && (u==UINT32_MAX || at[i]<at[u]))u=i;
    if(u==UINT32_MAX)return;
    done|=UINT64_C(1)<<u;
    for(uint32_t l=0;l<v->links;l++){
      if((v->link[l].ends&255)!=u)continue;
      const uint32_t b=v->link[l].ends>>8&255;
      if(b<v->nodes && !(done>>b&1) && at[u]+hop[l]<at[b])at[b]=at[u]+hop[l];
    }
  }
}
/* ---- The trainer's records: one producer a ring, never a wait ---------------------------------------------------- */
static inline void mesh_switch_log_put(struct mesh_switch *sw,uint32_t ring,const struct mesh_switch_record *r){
  if(ring>=MESH_PORTS || !atomic_load_explicit(&sw->logging,memory_order_relaxed))return;
  struct mesh_switch_log *g=sw->log+ring;
  const uint64_t tail=atomic_load_explicit(&g->tail,memory_order_relaxed);
  if(tail-atomic_load_explicit(&g->head,memory_order_acquire)>=MESH_SWITCH_LOG){atomic_fetch_add_explicit(&g->dropped,1,memory_order_relaxed);return;}
  g->record[tail%MESH_SWITCH_LOG]=*r;
  atomic_store_explicit(&g->tail,tail+1,memory_order_release);
}
/* whether a page's records are made: `logging` 0 none, k one page in k, chosen by the page's identity (its source and
   `seq`), so every node records the same pages and a recorded page's draws at every hop join its landings */
static inline int mesh_switch_sampled(const struct mesh_switch *sw,const struct mesh_head *head){
  const uint32_t k=atomic_load_explicit(&sw->logging,memory_order_relaxed);
  return k && (k==1 || mesh_switch_mix(head->seq^(uint64_t)head->source<<40)%k==0);
}
/* the next record of ring `ring`, 0 none (one consumer a ring) */
static inline int mesh_switch_log_take(struct mesh_switch *sw,uint32_t ring,struct mesh_switch_record *r){
  struct mesh_switch_log *g=sw->log+ring;
  const uint64_t head=atomic_load_explicit(&g->head,memory_order_relaxed);
  if(head==atomic_load_explicit(&g->tail,memory_order_acquire))return 0;
  *r=g->record[head%MESH_SWITCH_LOG];
  atomic_store_explicit(&g->head,head+1,memory_order_release);
  return 1;
}

/* ---- The data path: take, forward, landing --------------------------------------------------------------------- */
/* destination i of page pd's header, on this node: its table entry set to the page (mesh.h mesh_entry_put: its landed
   one more, the page it named before dropped), the entry's use of the page counted first.  The index is clamped into
   the table (its low bits, MESH_TABLE a power of two: a word outside the table is never written), never tested.  An
   index no client has now is set as any other: the page goes back as the index is next allocated (mesh-call.c mesh_pages makes its
   word new and drops the page it named) or as a later landing there replaces it, and the lane it landed on keeps its
   receives meanwhile (mesh-link.c link_receive: a landing whose class has no free page posts the landed page again); a
   lane whose fill comes after its class's stack is empty starts with fewer receives than its share (mesh-link.c
   switch_receives).  The index is read once: the one clamped is the one set */
_Static_assert(!(MESH_TABLE&(MESH_TABLE-1)),"a table index clamped by its low bits");
static inline void mesh_switch_set(struct hdr *m,struct mesh_switch *sw,uint32_t pd,const struct mesh_head *head,uint32_t i,uint32_t ring){
  const uint32_t index=__atomic_load_n(&head->dest[i].index,__ATOMIC_RELAXED)&(MESH_TABLE-1);
  mesh_pd_use(m,pd);
  mesh_pd_drop(m,mesh_entry_put(m,index,pd,1));
  MESH_SWITCH_COUNT(sw,ring,set,1);
}
/* The forward: page pd's destinations that this node carries on (`away`: places in its header's dest[]) split
   (mesh_switch_choose) into groups, a cable each, one group a neighbour (so a peer), and the page itself pushed onto
   the inbound ring of every group's cable for its class: one page and no copy, each group's destinations naming that
   cable's peer their `next` (a receiver reads only the destinations it is the next of).  Before any push the header's
   shared parts are written: each destination no group carries named no one's next (another hop's SEND carries it, it
   is set here, or nothing reaches it), the sender, and the lines (this node's own links' made from the cables the draw
   read, then the others' it keeps: as many as the frames of the payload's class fit).  Each group adds a use of the
   page and its frames to its cable's queue.  A destination no allowed neighbour reaches is not carried, counted
   (`unreachable`).  `bytes` the payload's bytes as the caller bounded them by the page's class (the write's, or the
   landed header's read once): the class's frames, so the inbound rings and lanes the page goes on, each SEND's frames
   and the lines' place, come from them and never from the header's word read again (every client maps the region
   writable), so no SEND covers more than the page's own frames.  `ring` the caller's port (its log ring and counters,
   and the pushing port of the inbound rings it pushes onto: a port child's poller, their one producer), `now` its
   clock, `togo` its cost-to-go (its poller's) */
static inline void mesh_switch_forward(struct hdr *m,struct mesh_switch *sw,uint32_t pd,struct mesh_head *head,uint32_t bytes,const uint32_t *away,
                                       uint32_t naway,uint64_t now,uint32_t ring,const struct mesh_switch_togo *togo){
  const uint32_t self=m->node,frames=mesh_class_frames(bytes),c=mesh_switch_class(frames);
  const int logging=mesh_switch_sampled(sw,head);
  uint32_t groups=0,unreachable=0;struct mesh_switch_group group[MESH_DESTS];
  struct route_draw draw;float logp=0;uint32_t context=UINT32_MAX;uint64_t point=0;
  if(logging)memset(&draw,0,sizeof draw);
  /* the draw over the published view where it lies, its copy as its version names it, read once: no retry and no copy
     of it a page (a copy written again under the draw at worst routes the page wrong); the port lines and the cables'
     terms read once, for the draw and this node's own lines alike */
  const struct mesh_switch_view *v=sw->view+mesh_switch_published(&sw->view_version);
  uint64_t line[MESH_PORTS];struct mesh_switch_terms terms[MESH_PORTS];
  mesh_switch_ports_read(sw,line);mesh_switch_terms_read(sw,v,line,terms);
  groups=mesh_switch_choose(sw,v,togo,line,terms,head,away,naway,frames,group,&unreachable,logging?&draw:NULL,&logp,&context,&point);
  if(logging && groups){
    struct mesh_switch_record r;memset(&r,0,sizeof r);
    r.kind=MESH_LOG_DRAW;r.node=self;r.seq=head->seq;r.at=now;r.draw=draw;r.logp=logp;r.source=head->source;
    r.dest=head->dest[away[0]].node;r.index=head->dest[away[0]].index;r.bytes=bytes;r.context=context;r.point=(uint32_t)point;
    mesh_switch_log_put(sw,ring,&r);
  }
  if(unreachable)MESH_SWITCH_COUNT(sw,ring,unreachable,__builtin_popcount(unreachable));
  MESH_SWITCH_COUNT(sw,ring,forwards,1);MESH_SWITCH_COUNT(sw,ring,groups,groups);
  if(!groups)return;
  uint32_t carried=0;
  for(uint32_t g=0;g<groups;g++)for(uint32_t j=0;j<naway;j++)if(group[g].dests>>j&1)carried|=1u<<away[j];
  for(uint32_t i=0;i<head->dests && i<MESH_DESTS;i++)if(!(carried>>i&1))head->dest[i].next=UINT32_MAX;
  head->from=self;
  /* the lines: this node's own links' from the cables the draw read, stamped now, then those it keeps of others' as it
     counts them now */
  struct mesh_switch_line lines[MESH_SWITCH_LINES];
  const uint32_t fit=mesh_switch_lines_fit(bytes,(uint64_t)frames*MESH_FRAME),
                 sent=mesh_switch_lines_give(sw,v,line,terms,now,fit,lines);
  head->lines=sent;memcpy(mesh_switch_lines_at(head,bytes),lines,sent*sizeof *lines);
  /* each group's destinations its peer's, with the cost they carry on (its cable's backlog, the page's own time on
     it), all before the first push makes the page another process's to post */
  for(uint32_t g=0;g<groups;g++){
    const struct mesh_switch_terms *x=terms+group[g].port%MESH_PORTS;
    for(uint32_t j=0;j<naway;j++)if(group[g].dests>>j&1){
      struct mesh_hop *d=head->dest+away[j];d->next=group[g].peer;d->queued+=(float)x->backlog;d->transit+=(float)x->own[c];
    }
  }
  for(uint32_t g=0;g<groups;g++){
    const uint32_t p=group[g].port%MESH_PORTS;
    mesh_pd_use(m,pd);
    atomic_fetch_add_explicit(&sw->cable[p].queued,(uint64_t)frames*MESH_FRAME,memory_order_relaxed);
    mesh_switch_push(sw,ring,p,c,pd,now);
  }
}
/* A client's write carried (the take): its page (`pd`, the page the write's ring use is on) given the write's bytes and
   its header (this node its source, its sender and its path's first node, the node's header counter its `seq`, every
   destination's next this node), and forwarded toward every destination (mesh_switch_forward), each on another node
   (its writer set those on this one: mesh-call.c mesh_write; one here on the ring is reached by no neighbour, counted
   `unreachable`).  Nothing of the cell is tested (its writer, mesh_write, made it of a page, destinations and bytes
   it had bounded): each field that addresses memory is clamped into what it addresses, the page into the pool's
   pages, the destinations to MESH_DESTS and the bytes to the page's payload, so a cell that is not one writes only
   within a page of the pool.  The pool has a page wherever a cell is (mesh_write writes none of a pool of none) */
static inline void mesh_switch_carry(struct hdr *m,struct mesh_switch *sw,const struct mesh_write *w,uint64_t now,uint32_t ring,
                                     const struct mesh_switch_togo *togo){
  const uint32_t self=m->node,pages=mesh_pool(m)->pages,pd=w->pd<pages?w->pd:pages-1,dests=w->dests<MESH_DESTS?w->dests:MESH_DESTS,
                 payload=MESH_PAYLOAD(mesh_pd_at(m,pd)->cls),bytes=w->bytes<payload?w->bytes:payload;
  MESH_SWITCH_COUNT(sw,ring,taken,1);
  atomic_store_explicit(&mesh_pd_at(m,pd)->bytes,bytes,memory_order_relaxed);
  struct mesh_head *head=mesh_head_of(m,pd);
  head->source=self;head->bytes=bytes;head->hops=0;head->dests=dests;head->lines=0;head->from=self;
  memset(head->path,0,sizeof head->path);head->path[0]=self;
  head->seq=atomic_fetch_add_explicit(&sw->writes,1,memory_order_relaxed);
  uint32_t away[MESH_DESTS];
  for(uint32_t i=0;i<dests;i++){head->dest[i]=(struct mesh_hop){w->dest[i].node,w->dest[i].index,self,0,0};away[i]=i;}
  mesh_switch_forward(m,sw,pd,head,bytes,away,dests,now,ring,togo);
}
/* Writes taken from client slot `slot`'s ring and carried (mesh_switch_carry), at most `most`, by a port child: each
   cell holding the write of its position (its sequence the position + 1) copied, then the ring's take moved past it by
   one compare-exchange, and the copy carried where that won (a copy whose compare-exchange lost is let be: another
   child took the position, and the producer may be writing the cell's next lap), then the write's ring use dropped.
   The cell is the producer's again from the compare-exchange on, so a child that exits mid-carry leaves the ring as
   it is and loses only that write (its ring use with it).  A cell that does not hold the write of the position read
   ends the take (the ring is empty there, or another child took on: the next pass reads the take again).  The ring's
   one taker is the port child whose share it is (mesh-link.c switch_takes, from the port lines); as a port starts or
   stops serving, the child whose share it was may still be taking, and each position is one compare-exchange's, so
   each write is one child's.  Returns the writes taken */
static inline uint32_t mesh_switch_ring_dequeue(struct hdr *m,struct mesh_switch *sw,uint32_t slot,uint32_t most,uint64_t now,uint32_t ring,
                                               const struct mesh_switch_togo *togo){
  struct mesh_ring *r=mesh_ring_at(m,slot);
  uint32_t taken=0;
  uint64_t at=atomic_load_explicit(&r->take,memory_order_relaxed);
  while(taken<most){
    struct mesh_cell *cell=&r->cell[at%MESH_RING];
    if(atomic_load_explicit(&cell->sequence,memory_order_acquire)!=at+1)break;
    const struct mesh_write w=cell->w;
    if(!atomic_compare_exchange_weak_explicit(&r->take,&at,at+1,memory_order_acq_rel,memory_order_relaxed))continue;
    mesh_switch_carry(m,sw,&w,now,ring,togo);
    mesh_pd_drop(m,w.pd);
    taken++;at++;
  }
  return taken;
}
/* A receive landed in page pd on lane l (the page of the lane's class): the page live, its one use the landing's, the
   page given the header's bytes, this node appended to its path, its lines taken (as many as the lane's frames fit),
   each destination here whose next this node is set in the table (mesh_switch_set) and its record made where the page
   is sampled, the destinations elsewhere whose next this node is forwarded (mesh_switch_forward), and the landing's
   use dropped.  Nothing of the header is tested: its bytes, destinations and hops are each read once and clamped (the
   bytes to the page's payload, the destinations to MESH_DESTS, the hops to MESH_SWITCH_N - 2, so this node's place in
   the path, hops + 1, is within it), and the values clamped are the ones used (every client maps the region writable,
   and a local reader has the page once a destination here is set).  A header that is not one is carried as one, and
   writes only within its page, its path and the table.  `now` the child's clock at the landing */
static inline void mesh_switch_landed(struct hdr *m,struct mesh_switch *sw,uint32_t l,uint32_t pd,uint64_t now,uint32_t ring,
                                      const struct mesh_switch_togo *togo){
  struct mesh_pd *p=mesh_pd_at(m,pd);
  atomic_store_explicit(&p->uses,1,memory_order_relaxed);
  atomic_store_explicit(&p->state,MESH_PD_LIVE,memory_order_relaxed);
  atomic_fetch_add_explicit(&sw->lane[l].landed,1,memory_order_relaxed);
  MESH_SWITCH_COUNT(sw,ring,landed,1);
  struct mesh_head *head=mesh_head_of(m,pd);
  const uint32_t self=m->node,payload=MESH_PAYLOAD(p->cls),bytes_read=__atomic_load_n(&head->bytes,__ATOMIC_RELAXED),
                 dests_read=__atomic_load_n(&head->dests,__ATOMIC_RELAXED),hops_read=__atomic_load_n(&head->hops,__ATOMIC_RELAXED);
  const uint32_t bytes=bytes_read<payload?bytes_read:payload,dests=dests_read<MESH_DESTS?dests_read:MESH_DESTS,
                 hops=hops_read<MESH_SWITCH_N-2?hops_read:MESH_SWITCH_N-2;
  atomic_store_explicit(&p->bytes,bytes,memory_order_relaxed);
  head->hops=hops+1;head->path[hops+1]=self;
  mesh_switch_lines_take(sw,head,bytes,mesh_switch_lane_bytes(l),now);
  const int logging=mesh_switch_sampled(sw,head);
  uint32_t away[MESH_DESTS],naway=0;
  for(uint32_t i=0;i<dests;i++){
    const struct mesh_hop *d=head->dest+i;
    if(d->next!=self)continue;
    if(d->node!=self){away[naway++]=i;continue;}
    mesh_switch_set(m,sw,pd,head,i,ring);
    if(!logging)continue;
    /* a page's presence at each destination here, recorded as it is set: the cost it carried in-band (each hop's
       queueing on that hop's own clock, the wire's time) */
    struct mesh_switch_record r;memset(&r,0,sizeof r);
    r.kind=MESH_LOG_SET;r.node=self;r.seq=head->seq;r.at=now;r.latency=d->queued+d->transit;r.queueing=d->queued;
    r.source=head->source;r.dest=self;r.index=d->index;r.bytes=bytes;r.context=UINT32_MAX;
    mesh_switch_log_put(sw,ring,&r);
  }
  if(naway)mesh_switch_forward(m,sw,pd,head,bytes,away,naway,now,ring,togo);
  mesh_pd_drop(m,pd);
}

/* ---- A port child exited, a region adopted: what it had, given back (the parent's: mesh-link.c switch_child_free,
   region_adopt) ------------------------------------------------------------------------------------------------- */
/* the pages posted as receives on the switch's lanes [first, first + count) given back (their child exited: nothing
   lands in them again); how many.  A page's state is read (acquire) before its lane and again after, so a lane read
   from a page's earlier receive is never taken with a later one's state: only a page still a receive on a lane of the
   range is given back */
static inline uint32_t mesh_switch_receives_drop(struct hdr *m,uint32_t first,uint32_t count){
  const uint32_t pages=mesh_pool(m)->pages;
  uint32_t n=0;
  for(uint32_t pd=0;pd<pages;pd++){
    struct mesh_pd *p=mesh_pd_at(m,pd);
    if(atomic_load_explicit(&p->state,memory_order_acquire)!=MESH_PD_RECEIVE)continue;
    const uint32_t lane=atomic_load_explicit(&p->lane,memory_order_relaxed);
    if(lane<first || lane-first>=count || atomic_load_explicit(&p->state,memory_order_acquire)!=MESH_PD_RECEIVE)continue;
    atomic_store_explicit(&p->uses,0,memory_order_relaxed);
    mesh_pool_push(m,pd);n++;
  }
  return n;
}

/* ---- The policy's refresh: its features from this node's state, its tables published ---------------------------
   The features' time scale: the slowest pair of ranks' shortest unloaded delay for one page (route.h's stream
   `unit`) */
static inline double mesh_switch_time_scale(const struct mesh_switch_view *v,double page){
  double hop[MESH_SWITCH_LINKS],at[MESH_SWITCH_N],scale=0;
  for(uint32_t l=0;l<v->links;l++)hop[l]=v->link[l].latency+page/(v->link[l].rate/v->link[l].cables);
  const uint32_t ranks=mesh_switch_ranks(v);
  for(uint32_t a=0;a<v->nodes;a++){
    if(!(ranks>>a&1))continue;
    mesh_switch_arrivals(v,a,0,0,hop,at);
    for(uint32_t b=0;b<v->nodes;b++)if(b!=a && ranks>>b&1 && isfinite(at[b]) && at[b]>scale)scale=at[b];
  }
  return scale>0?scale:1;
}
/* route.h's tokens from what this node knows, at its clock `now`, for policy refresh `point`:
   - the fabric is the view (its links route.h's lanes, view indices its nodes, the ranks its compute nodes), its page
     route.h's (ROUTE_PAGE);
   - each lane's wait the time a long page takes through it beyond its idle time (mesh_switch_link_cost): a remote
     link's from its line as the pages carried it, less the time since this node took it (0 once it has drained), each
     of this node's own links' its cables' backlog now; duplex share 1 (a bridge sees no other node's reverse
     traffic); an own link that is down (its wait not finite: no port serves it) no lane of the node now, so no token;
   - the cost-to-go a long page's (`togo`, the caller's, made over the same view: mesh_switch_togo_whole), the bytes in
     flight and posted this node's own cables' queues;
   - the per-pattern features of what is left to send, waiting and received, and the live collectives, 0: the bridge
     carries writes, not collectives.
   Every token is route.h's route_lane_token / route_context_token over that state.  The policy's thread's: the view's
   published copy read in place */
static inline void mesh_switch_tokens_make(struct mesh_switch *sw,const struct mesh_switch_togo *togo,uint64_t now,uint64_t point,struct mesh_switch_tokens *out){
  const struct mesh_switch_view *v=sw->view+mesh_switch_published(&sw->view_version);
  uint64_t line[MESH_PORTS];struct mesh_switch_terms terms[MESH_PORTS];
  mesh_switch_ports_read(sw,line);mesh_switch_terms_read(sw,v,line,terms);
  const float (*to)[MESH_SWITCH_N]=togo->to[mesh_switch_class(MESH_CLASS_LONG)];
  const uint32_t self=mesh_switch_index(v,v->self),ranks=mesh_switch_ranks(v);
  static __thread struct route_stream s;static __thread struct route_state st;
  memset(&s,0,sizeof s);memset(&st,0,sizeof st);
  s.f.nodes=(int)(v->nodes<ROUTE_N?v->nodes:ROUTE_N);s.f.lanes=(int)(v->links<ROUTE_L?v->links:ROUTE_L);s.f.duplex=1;
  for(int l=0;l<s.f.lanes;l++){
    const struct mesh_switch_link *x=v->link+l;
    s.f.from[l]=(int)mesh_switch_index(v,x->from);s.f.to[l]=(int)mesh_switch_index(v,x->to);s.f.cables[l]=(int)x->cables;s.f.rate[l]=x->rate;s.f.latency[l]=x->latency;
  }
  for(uint32_t a=0;a<v->nodes && a<ROUTE_N;a++)if(ranks>>a&1)s.f.compute[s.f.ncompute++]=(int)a;
  s.unit=v->scale;
  struct route_topology t;memset(&t,0,sizeof t);t.srcmask=ranks;
  /* the patterns, in mesh_switch_contexts' order */
  for(uint32_t a=0;a<v->nodes;a++){
    if(!(ranks>>a&1))continue;
    for(uint32_t pass=0;pass<=v->nodes && s.patterns<ROUTE_J;pass++){
      if(pass<v->nodes && (pass==a || !(ranks>>pass&1)))continue;
      s.source[s.patterns]=(int)a;s.dest[s.patterns]=pass<v->nodes?1u<<pass:ranks&~(1u<<a);s.patterns++;
    }
  }
  st.tau=(double)now;
  for(int l=0;l<s.f.lanes;l++){
    double idle;const double cost=mesh_switch_link_cost(sw,v,line,terms,(uint32_t)l,mesh_switch_class(MESH_CLASS_LONG),now,&idle);
    st.wait[l]=!isfinite(cost)?INFINITY:cost>idle?cost-idle:0;st.dup[l]=1;
  }
  for(uint32_t a=0;a<ROUTE_N;a++)for(uint32_t b=0;b<ROUTE_N;b++)st.togo[a][b]=a<v->nodes && b<v->nodes?(a==b?0:to[a][b]):1e12;
  double queued=0;
  for(uint32_t p=0;p<MESH_PORTS;p++)if(mesh_port_state(line[p])==CHILD_SERVING)queued+=(double)atomic_load_explicit(&sw->cable[p].queued,memory_order_relaxed);
  if(self<ROUTE_N)st.inflight[self]=queued;
  st.posted=queued;
  out->point=point;out->at=now;out->scale=(float)s.unit;out->lanes=0;
  for(int l=0;l<s.f.lanes;l++){
    if(v->link[l].from!=v->self || !isfinite(st.wait[l]))continue;
    out->link_of[out->lanes]=(uint32_t)l;route_lane_token(&s,&st,&t,l,out->lane[out->lanes]);out->lanes++;
  }
  out->contexts=mesh_switch_contexts(v,out->key);
  for(uint32_t c=0;c<out->contexts;c++){
    const struct mesh_switch_context *k=out->key+c;
    route_context_token(&s,&st,&t,(int)k->pattern,(int)self,k->onward,(int)point,out->context[c]);
  }
}
/* the split tables published: `n` contexts (keys, each its w), from policy refresh `point`, written in the copy not
   published, then published; active from now on */
static inline void mesh_switch_splits_publish(struct mesh_switch *sw,const struct mesh_switch_context *key,uint32_t n,const float (*w)[SPLIT_FS],
                                              uint64_t point,int argmax){
  const uint64_t version=atomic_load_explicit(&sw->splits_version,memory_order_relaxed);
  struct mesh_switch_splits *x=sw->splits+((version+1)&1);
  if(n>MESH_SWITCH_CONTEXTS)n=MESH_SWITCH_CONTEXTS;
  x->contexts=n;memcpy(x->key,key,n*sizeof *key);memcpy(x->w,w,n*sizeof *w);x->point=point;x->argmax=argmax?1:0;x->active=1;
  atomic_store_explicit(&sw->splits_version,version+1,memory_order_release);
}
/* one refresh of the trained policy: its tokens from this node's state (`togo` its cost-to-go, the caller's), each
   context's w from the policy, published */
static inline int mesh_switch_policy_refresh(struct mesh_switch *sw,const struct mesh_switch_togo *togo,const struct mesh_switch_policy *policy,uint64_t now,
                                             uint64_t point,int argmax,struct mesh_switch_tokens *scratch){
  if(!policy || !policy->splits)return -1;
  mesh_switch_tokens_make(sw,togo,now,point,scratch);
  static __thread float w[MESH_SWITCH_CONTEXTS][SPLIT_FS];
  if(policy->splits(policy->context,scratch,w))return -1;
  mesh_switch_splits_publish(sw,scratch->key,scratch->contexts,(const float (*)[SPLIT_FS])w,point,argmax);
  return 0;
}

#endif
