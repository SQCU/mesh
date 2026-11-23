/* The switch (design/PAGE-TABLE-SPEC-20261122.md section 5): what every node's bridge does with a page, as functions of
   its region.  The bridge's port children (mesh-link.c port_pass) and its parent (the refresh thread, a child's end,
   a region's adoption) are its callers; none of it does I/O or posts to hardware, and only mesh_switch_push reads a
   clock (its cell's `carried`, for observation): the caller passes the time and posts what these functions hand it.

   A page crosses as one SEND from its own memory (mesh.h struct mesh_pd): its first MESH_HEAD bytes the header the
   bridge writes (struct mesh_head: the node that took the write, its destinations as nodes and indices of their page
   tables, the nodes it visited), then its payload, then the lines of lane state it carries in its last frame's slack.
   A SEND fills one receive posted into a page of the lane's class (TN3205: both ends post the same frame count), and
   the lane's child posts that receive again at once (mesh-link.c link_receive: into a free page, else into the page
   that landed, whose landing is then lost), so a lane's receives do not run out while every page of its class is used.
   - Take.  A client's write (mesh.h struct mesh_write, from its ring) names its page and its destinations: the bridge
     writes the page's header, sets the table entry of each destination on this node to the page, and forwards the page
     toward the others (mesh_switch_carry).  Each ring has one taker, the port child the parent names for it (struct
     mesh_switch `taker`: ring s the (s mod n)-th, by port index, of the n ports whose link serves a lane).
   - Switching.  At every node a page is at, the destinations it carries on are split over the node's out-neighbours by
     a draw from the routing weights (mesh_switch_choose): one uniform u per page, each destination's lane the inverse of
     its weights' distribution at u (so destinations whose distributions agree share one SEND), masked to the
     neighbours off the page's path that reach the destination without coming back, over the lanes of its payload's
     class alone (the frames of the least class that holds the payload, mesh.h mesh_class_frames: a SEND covers its
     lane's frames, so on a lane of more frames it would carry the pool's next pages with it).  The cable to a
     neighbour is the one with the least bytes queued.  The page is pushed onto each group's lane's outbox (a ring in
     the region any port child pushes and the lane's port child posts), a use of it a group: the page is switched
     across the node's ports, whichever child has it.
   - Landing.  A page landed sets the table entry of each destination here whose turn it is (mesh_switch_set: the
     index's landed count one more, the page it named before dropped) and is forwarded unchanged for the others it
     carries (mesh_switch_landed).  Nothing waits and nothing is reported: a destination no allowed neighbour reaches,
     a header that is not one, a page a lost link took, a landing whose receive went again into its own page are each
     counted (`unreachable`, `malformed`, `lost`: struct mesh_switch_counters), never retried.
   - Uses.  A page's uses (mesh.h) count its table words, its writes pending in a ring, its SENDs pushed or posted and a
     landing being carried; its last use's drop gives it back to its class's free stack.
   - The weights.  A distribution, for each destination, over the node's out-neighbours, read by every draw.  The
     view, the weights and the policy's split tables are each in two copies: the parent's refresh thread writes the
     copy not published, then publishes it (its version moved, one release store); a draw reads the copy published as
     it began (its version, one acquire load) and retries nothing.  A refresh (mesh_switch_refresh) computes a target
     from the estimators and emits the residual between it and the current weights, applied at the rate eta; draws go
     on reading the last published weights while one is computed: never a wait.  A draw that outlasts the next publish
     (the other copy's) may read its own copy as the next refresh writes it again, and at worst chooses a wrong route;
     every index it reads there addresses only within its array (mesh_switch_choose).  The target is the policy's: by
     default Thompson sampling over paths (each hop's wait drawn from its line), or any struct mesh_switch_policy (the
     trained recurrent policy, rdma/route, plugs in there).
   - The estimators.  Diachronic and incremental: each lane's rate and wait from its SENDs' completions (mesh_switch_sent),
     each remote link's wait from the lines pages carry (mesh_switch_lines_take), each node's own lines stamped by its
     refreshes.  Nothing waits for any of it.

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
#define MESH_HEAD_MAGIC 0x50414745u                          /* a page's header */
#define MESH_SWITCH_LANES (MESH_PORTS*2u*MESH_LINK_STREAMS)   /* every port's lanes, its link's queue pairs */
#define MESH_SWITCH_N 8u                                     /* nodes the view has */
#define MESH_SWITCH_LINKS 64u                                /* directed node pairs the view has */
#define MESH_SWITCH_RING 32768u                              /* a lane's outbox: every page of the pool once */
#define MESH_SWITCH_LINES 24u                                /* lines a header carries */
#define MESH_SWITCH_LOG 8192u                                /* records a process's log ring */
#define MESH_SWITCH_LOGS (MESH_PORTS+1u)                     /* counters a port child, then the parent's */
#define MESH_SWITCH_PARENT MESH_PORTS                        /* the parent's counters */
#define MESH_SWITCH_CONTEXTS 256u                            /* the policy's contexts at a node, at most */
#define MESH_SWITCH_REFRESH_NS 50000.0                       /* the refresh's period (the bridge's thread) */
#define MESH_SWITCH_DRAWS 32u                                /* Thompson sampling's draws a refresh */
#define MESH_SWITCH_ETA 0.5f                                 /* the residual's rate a refresh */
#define MESH_SWITCH_PAGE MESH_CLASS_LONG                      /* the refresh's page, its frames: a long page's */
_Static_assert(MESH_SWITCH_N<=64 && MESH_SWITCH_N<=MESH_NODES,"a node a bit");
_Static_assert(MESH_DESTS<=32,"a destination a bit");
/* a page is in a lane's outbox at most once (its SEND's completion drops the use that blocks its next write or landing),
   so an outbox of every pool page never fills */
_Static_assert(MESH_SWITCH_RING>=MESH_POOL_PAGES && !(MESH_SWITCH_RING&(MESH_SWITCH_RING-1)),"an outbox has room for every page once");

/* ---- The header (a page's first MESH_HEAD bytes) ------------------------------------------------------------------- */
/* a destination as a header carries it: its node and index of that node's table; the node whose turn it is to carry it
   on (`next`: the node a hop sent it to, which alone switches it; a page goes out unchanged on every group's lane, each
   receiver reading only the destinations it is the next of; UINT32_MAX no one's); and the cost it carried so far (ns:
   queueing at each hop, the wire's time) */
struct mesh_hop { uint32_t node,index,next; float queued,transit; };
/* a line: a directed link's predicted wait (ns; INFINITY: the link is down, none of its sender's lanes toward its peer
   live) and its spread, its rate (bytes a ns, every cable) and latency, as its sender (`from`) measured it at its
   refresh `stamp` */
struct mesh_switch_line { uint32_t from,to; uint64_t stamp; float wait,spread,rate,latency; };
/* the header, written by the bridge (never the client): MESH_HEAD_MAGIC; the node that took the write (`source`); the
   payload's bytes; the hops, destinations and lines; the node that sent this SEND (`from`: its receiver appends itself
   to `path`); `unused`, unused; the nodes visited, `hops` + 1 of them; the destinations; and the sequence of the
   headers its first node made (`seq`: the draw's seed, the records' sampling and their join).  Every field names nodes
   by their numbers, never by a view's indices (each node's view is its own).  The lines follow the payload in the last
   frame's slack (mesh_switch_lines_at) */
struct mesh_head {
  uint32_t magic,source,bytes,hops,dests,lines,from,unused;
  uint32_t path[MESH_SWITCH_N];
  struct mesh_hop dest[MESH_DESTS];
  uint64_t seq;
};
_Static_assert(sizeof(struct mesh_head)<=MESH_HEAD && MESH_HEAD%64==0,"a header within its MESH_HEAD bytes");
_Static_assert(sizeof(struct mesh_head)>MESH_HEAD-64,"MESH_HEAD no longer than the header needs");
static inline struct mesh_head *mesh_head_of(struct hdr *m,uint32_t pd){ return (struct mesh_head *)mesh_pd_head(m,pd); }

/* ---- The region's switch (struct hdr switch_off; MESH_SWITCH_BYTES, mesh.h) ------------------------------------- */
/* a link of the view: a directed node pair, its cables, rate (all of them, bytes a ns), latency (ns), and its line's
   wait and spread (two floats in one word) with the line's stamp */
struct mesh_switch_link { uint32_t from,to,cables,ends; double rate,latency; _Atomic uint64_t line,stamp; };   /* ends: the view indices, from | to << 8 */
/* a copy of the view (struct mesh_switch view[2]): `nodes` at most MESH_SWITCH_N and `links` at most MESH_SWITCH_LINKS
   in every store of them (mesh_switch_view_set), so a loop under either stays within its array */
struct mesh_switch_view {
  uint32_t self,nodes,links,ranks; /* ranks: the view nodes that run collectives' ranks, bits over view indices (0: all) */
  uint32_t node[MESH_SWITCH_N];     /* node numbers, ascending: a node's index is its place here */
  struct mesh_switch_link link[MESH_SWITCH_LINKS];
  /* reach[avoid][from]: the view nodes `from` reaches (itself among them) entering none of `avoid` (bits over view
     indices), made with the links (mesh_switch_view_set): a draw's allowed neighbours read here, not searched */
  uint8_t reach[1u<<MESH_SWITCH_N][MESH_SWITCH_N];
};
/* the weights: for the destination of view index d, a probability over the node's out-neighbours (neighbour[k], view
   indices), and the refresh's cost-to-go between view nodes (ns), published as the view is (two copies).  `deg` is at
   most MESH_SWITCH_N and each neighbour a view index under MESH_SWITCH_N in every store of them (mesh_switch_refresh,
   from mesh_switch_state_make) */
struct mesh_switch_weights {
  uint32_t nodes,deg;
  uint32_t neighbour[MESH_SWITCH_N];
  float w[MESH_SWITCH_N][MESH_SWITCH_N];
  float togo[MESH_SWITCH_N][MESH_SWITCH_N];
  uint64_t refreshes;
  uint64_t toward[MESH_SWITCH_N];   /* neighbour k's lanes live at the refresh, bits over the switch's lanes (a draw reads
                                       these lanes alone, each its `live` again) */
};
/* the policy's split tables (rdma/route route.h): this node's contexts, each (its pattern's source, the pattern's
   destinations, the destinations still to carry: view indices, the last two as bits) in mesh_switch_contexts'
   sequence, and each one's w [SPLIT_FS], published as the weights are (two copies).  `active` set, a page
   whose context is here draws its split from split.h's table under its w (mesh_switch_choose); else the weights draw.
   `scale`: the features' time scale (mesh_switch_time_scale), `point`: the policy refresh the tables are from */
struct mesh_switch_context { uint32_t source,dest_nodes,onward,pattern; };
struct mesh_switch_splits {
  uint32_t active,contexts,argmax,padding;
  uint64_t point,scale_view;          /* scale_view: the view's version `scale` was made from, plus 1 */
  float scale,spare;
  struct mesh_switch_context key[MESH_SWITCH_CONTEXTS];
  float w[MESH_SWITCH_CONTEXTS][SPLIT_FS];
};
/* a lane of this node (port p's lane j, its link's queue pair j: p 2 MESH_LINK_STREAMS + j): live, its peer node, the
   bytes queued on it (pushed and not yet completed), its outbox's positions (`tail` the next position pushed, by any port child; `head`
   the next posted and `done` the next completed, by the lane's child), and its feedback: SENDs completed, their bytes,
   their service and wait (ns, summed), pages landed, the estimate its child keeps (rate in bytes a ns and wait, two
   floats in one word); and its frames (mesh_switch_lane_frames of its index, told for observers; 0 while it never
   opened) */
struct mesh_switch_lane {
  _Alignas(64) _Atomic uint32_t live,peer;
  _Atomic uint64_t queued,tail;     /* every pushing child's (mesh_switch_push) */
  /* the lane's child's alone, on a line of its own: its outbox's head and done read every pass, never bounced by a push */
  _Alignas(64) _Atomic uint64_t head,done;
  _Atomic uint64_t sends,bytes,service,wait,landed,estimate;
  _Atomic uint64_t frames;
};
/* an outbox cell, of position `at`: free for it while its sequence is `at`; with page `pd`, pushed at `pushed` (the
   pusher's clock as its take pass began: the page's time on the outbox, for the lane's estimators, counts from there)
   and posted at `posted` (its SEND's post, the lane child's clock), while it is at + 1, until the page's SEND completes
   or its lane is emptied (mesh_switch_lane_drop), so the outbox is also the lane's record of its SENDs posted and not
   completed; then free for at + MESH_SWITCH_RING.  `carried`: the nanoseconds from `pushed` to the push itself (the take,
   the header, the route choice), for observation only; the cell's size and offsets are as before it was recorded.
   `pd` MESH_ABSENT: a hole, no SEND, where a pusher died with the cell won (mesh_switch_left_settle) */
struct mesh_switch_cell { _Atomic uint64_t sequence; uint64_t pushed,posted; uint32_t pd,carried; };
_Static_assert(sizeof(struct mesh_switch_cell)==32,"mesh_switch_cell");
/* the trainer's records (rdma/route): a draw (split.h struct route_draw and its behaviour log-probability) or a page
   set at a destination (its latency and queueing), each with the node that made it, the page's `seq` and source, its
   first destination's node and index (a draw's) or the destination's, its bytes, the draw's context at its node
   (UINT32_MAX: none, the weights drew) and the policy refresh its split table is from, at the node's clock (ns) */
enum { MESH_LOG_DRAW=1, MESH_LOG_SET };
struct mesh_switch_record {
  uint32_t kind,node;
  uint64_t seq,at;
  struct route_draw draw;           /* MESH_LOG_DRAW */
  float logp,latency,queueing;
  uint32_t source,dest,index,bytes,context,point;
};
struct mesh_switch_log { _Alignas(64) _Atomic uint64_t head; _Alignas(64) _Atomic uint64_t tail,dropped; struct mesh_switch_record record[MESH_SWITCH_LOG]; };
/* a port child's hand, which its parent reads once the child has exited and the next bridge reads as it adopts the
   region (mesh_switch_left_settle): the ring position of the write it takes (`take`: (slot + 1) << 56 | the position,
   MESH_HAND_WON once its compare-exchange won, 0 none), and the outbox position it pushes page `page` to (`push`:
   (lane + 1) << 56 | the position, MESH_HAND_WON once won, 0 none) */
#define MESH_HAND_WON (UINT64_C(1)<<55)
#define MESH_HAND_AT (MESH_HAND_WON-1)
struct mesh_switch_hand { _Alignas(64) _Atomic uint64_t take,push; _Atomic uint32_t page; };
/* a hand as its child left it, the parent's copy */
struct mesh_switch_left { uint64_t take,push; uint32_t page; };
/* where a pass's time goes, counted in memory (never storage) by each process of the switch, a set a port child (at
   its port's index) and the parent's (at MESH_SWITCH_PARENT), read by mesh-switch-stat:
   - passes, and of one pass in 1024 its time (`pass_ns`: summed, and the passes timed) and each part's (`part_ns`:
     receive completions, receives posted, writes taken, SENDs posted, SEND completions);
   - writes taken, pages landed, table entries set here, forwards and the groups they made;
   - SENDs posted and their wait from the push onto the outbox to the post, the posts' own time (the provider's
     post_send), SENDs completed and their time from post to completion;
   - receives a lane's fill asked for that its class's free stack had no page for (`starved`), refreshes;
   - destinations no allowed neighbour reaches (`unreachable`), pages a lost lane, an exited child or a full outbox took
     and landings whose receive was posted again into their own page, their class's free stack empty (`lost`: once a
     page, not a destination), and headers and ring cells that are not one and destination indices past
     the table (`malformed`) */
struct mesh_switch_counters {
  _Alignas(64) _Atomic uint64_t passes,pass_ns[2],part_ns[5];
  _Atomic uint64_t taken,landed,set,forwards,groups;
  _Atomic uint64_t posts,post_wait_ns,post_call_ns,sent,send_ns;
  _Atomic uint64_t starved,refreshes;
  _Atomic uint64_t unreachable,lost,malformed;
};
/* `writes`: the headers this node made (the writes it took), each one's `seq`; `weighted`: a draw from the weights
   (else the earliest arrival, the design's rule).  `taker`: each client ring's one taker, the port whose child takes
   its writes (MESH_SWITCH_NO_PORT: none), the parent's to write as its links come up and go (mesh-link.c switch_takers)
   and each port child's to read every pass, on a line of its own.  The view, the weights and the split tables, two
   copies each: the published one is [version & 1], each version (`view_version`, `weights_version`, `splits_version`)
   the publishes of its copies so far, moved by a release store once the copy not published is written whole */
#define MESH_SWITCH_NO_PORT UINT8_MAX
_Static_assert(MESH_PORTS<MESH_SWITCH_NO_PORT,"a port a taker byte");
struct mesh_switch {
  uint32_t magic,padding;
  _Atomic uint64_t stamp;
  _Atomic uint32_t logging,weighted;
  _Alignas(64) _Atomic uint64_t writes;
  _Alignas(64) _Atomic uint8_t taker[MESH_CLIENTS];
  _Alignas(64) _Atomic uint64_t view_version,weights_version,splits_version;
  struct mesh_switch_view view[2];
  struct mesh_switch_weights weights[2];
  struct mesh_switch_splits splits[2];
  struct mesh_switch_lane lane[MESH_SWITCH_LANES];
  struct mesh_switch_hand hand[MESH_PORTS];
  struct mesh_switch_cell ring[MESH_SWITCH_LANES][MESH_SWITCH_RING];
  struct mesh_switch_log log[MESH_PORTS];
  struct mesh_switch_counters counters[MESH_SWITCH_LOGS];
};
_Static_assert(sizeof(struct mesh_switch)<=MESH_SWITCH_BYTES,"the switch within its region bytes");
/* counter `field` of set `ring` moved by v by its one writer, a load and a store (no read-modify-write): a port child's
   set its child's, the parent's `refreshes` its refresh thread's and its `lost` its event loop's */
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
/* the switch's lanes of `frames` frames, bits over its lanes: a port's lanes of those frames, at every port (the port's
   bits times a word with a 1 at each port's first lane), so a draw finds a class's lanes without reading a lane */
_Static_assert(64%(2*MESH_LINK_STREAMS)==0,"a port's lanes a whole number of a mask's bits");
static inline uint64_t mesh_switch_class_lanes(uint32_t frames){
  uint64_t port=0;
  for(uint32_t j=0;j<2*MESH_LINK_STREAMS;j++)if(mesh_switch_lane_frames(j)==frames)port|=UINT64_C(1)<<j;
  return port*(~UINT64_C(0)/((UINT64_C(1)<<(2*MESH_LINK_STREAMS))-1));
}
/* the published copy of the view, the weights or the split tables by its version (one acquire load: the copy's writes
   before its publish seen): 0 or 1 */
static inline uint32_t mesh_switch_published(const _Atomic uint64_t *version){
  return (uint32_t)(atomic_load_explicit(version,memory_order_acquire)&1);
}

/* the switch made new at the bridge's start, every outbox empty and no ring taken (the parent names their takers as
   its links come up): nothing of a bridge before survives its stop */
static inline void mesh_switch_init(struct mesh_switch *sw){
  memset(sw,0,sizeof *sw);
  for(uint32_t s=0;s<MESH_CLIENTS;s++)atomic_store_explicit(&sw->taker[s],MESH_SWITCH_NO_PORT,memory_order_relaxed);
  for(uint32_t l=0;l<MESH_SWITCH_LANES;l++)for(uint32_t i=0;i<MESH_SWITCH_RING;i++)atomic_store_explicit(&sw->ring[l][i].sequence,i,memory_order_relaxed);
  sw->magic=MESH_SWITCH_MAGIC;
  atomic_thread_fence(memory_order_release);
}

/* ---- A lane's outbox: bounded, any port child pushes, the lane's child posts and completes ----------------------- */
/* page pd pushed onto lane l's outbox at the pusher's clock `now`, by any port child, naming its hand (each position in
   it before its compare-exchange, marked won once won, cleared once the cell is filled, so its parent settles a push it
   died in, mesh_switch_left_settle): 1, or 0 where the outbox is full (it has room for every pool page once, so a push
   of a page not already on it never finds it full), the hand cleared */
static inline int mesh_switch_push(struct mesh_switch *sw,uint32_t l,uint32_t pd,uint64_t now,struct mesh_switch_hand *hand){
  struct mesh_switch_lane *x=sw->lane+l;
  uint64_t at=atomic_load_explicit(&x->tail,memory_order_relaxed);
  atomic_store_explicit(&hand->page,pd,memory_order_relaxed);
  for(;;){
    struct mesh_switch_cell *c=sw->ring[l]+at%MESH_SWITCH_RING;
    const uint64_t seq=atomic_load_explicit(&c->sequence,memory_order_acquire);
    if(seq<at){atomic_store_explicit(&hand->push,0,memory_order_release);return 0;}
    if(seq>at){at=atomic_load_explicit(&x->tail,memory_order_relaxed);continue;}
    const uint64_t named=((uint64_t)(l+1)<<56)|at;
    atomic_store_explicit(&hand->push,named,memory_order_relaxed);
    if(!atomic_compare_exchange_weak_explicit(&x->tail,&at,at+1,memory_order_acq_rel,memory_order_relaxed))continue;
    atomic_store_explicit(&hand->push,named|MESH_HAND_WON,memory_order_relaxed);
    const uint64_t pushing=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    c->pd=pd;c->pushed=now;c->carried=pushing>now?(uint32_t)(pushing-now):0;
    atomic_store_explicit(&c->sequence,at+1,memory_order_release);
    atomic_store_explicit(&hand->push,0,memory_order_release);
    return 1;
  }
}
/* lane l's outbox cell of position `at` read: 1 where a page is pushed there (*pd the page, MESH_ABSENT a hole, *pushed
   its push's clock where asked), else 0 */
static inline int mesh_switch_cell_at(const struct mesh_switch *sw,uint32_t l,uint64_t at,uint32_t *pd,uint64_t *pushed){
  const struct mesh_switch_cell *c=sw->ring[l]+at%MESH_SWITCH_RING;
  if(atomic_load_explicit(&c->sequence,memory_order_acquire)!=at+1)return 0;
  *pd=c->pd;
  if(pushed)*pushed=c->pushed;
  return 1;
}
/* lane l's outbox cell of position `at` given back, free for its next lap */
static inline void mesh_switch_cell_free(struct mesh_switch *sw,uint32_t l,uint64_t at){
  atomic_store_explicit(&sw->ring[l][at%MESH_SWITCH_RING].sequence,at+MESH_SWITCH_RING,memory_order_release);
}
/* lane l's outbox cell of position `at`, with a page, posted at `posted` (its lane child's clock) */
static inline void mesh_switch_cell_posted(struct mesh_switch *sw,uint32_t l,uint64_t at,uint64_t posted){
  sw->ring[l][at%MESH_SWITCH_RING].posted=posted;
}
/* lane l's next SEND completed (its child's): the holes before it passed, its cell given back; its page, whose use the
   caller drops (MESH_ABSENT: none posted), and *posted its post's clock (unchanged where none) */
static inline uint32_t mesh_switch_completed(struct mesh_switch *sw,uint32_t l,uint64_t *posted){
  struct mesh_switch_lane *x=sw->lane+l;
  const uint64_t head=atomic_load_explicit(&x->head,memory_order_relaxed);
  uint64_t done=atomic_load_explicit(&x->done,memory_order_relaxed);
  uint32_t pd=MESH_ABSENT;
  while(done<head && pd==MESH_ABSENT){
    if(!mesh_switch_cell_at(sw,l,done,&pd,NULL))break;
    if(pd!=MESH_ABSENT)*posted=sw->ring[l][done%MESH_SWITCH_RING].posted;
    mesh_switch_cell_free(sw,l,done);done++;
  }
  atomic_store_explicit(&x->done,done,memory_order_relaxed);
  return pd;
}
/* lane l's outbox emptied, its child exited or its lane opened anew: each SEND posted and not completed and each page
   pushed and not posted given back, its bytes off the lane's queue, its use dropped and counted `lost` in set `ring`
   (the caller's counters); the positions past them.  The cells from the first a pusher has won and not filled stay
   (its push ends, or its death is settled: mesh_switch_left_settle) */
static inline void mesh_switch_lane_drop(struct hdr *m,struct mesh_switch *sw,uint32_t l,uint32_t ring){
  struct mesh_switch_lane *x=sw->lane+l;
  const uint64_t wire=mesh_switch_lane_bytes(l);
  uint64_t at=atomic_load_explicit(&x->done,memory_order_relaxed);
  uint32_t pd;
  while(mesh_switch_cell_at(sw,l,at,&pd,NULL)){
    if(mesh_pd_valid(m,pd)){
      atomic_fetch_sub_explicit(&x->queued,wire,memory_order_relaxed);
      mesh_pd_drop(m,pd);
      MESH_SWITCH_COUNT(sw,ring,lost,1);
    }
    mesh_switch_cell_free(sw,l,at);at++;
  }
  atomic_store_explicit(&x->head,at,memory_order_relaxed);
  atomic_store_explicit(&x->done,at,memory_order_release);
}

/* ---- The view and the weights: two copies each, read by all -------------------------------------------------------
   Every copy is written by the parent's refresh thread alone (mesh-link.c switch_watch: the view by
   mesh_switch_view_set, the weights and the split tables by the refresh), so the copy it publishes stands until its
   next write of that copy.  The view's lines are written in place in the published copy, by more than one process: each
   of this node's own links' by the refresh, every other link's by each port child's landings (mesh_switch_lines_take) */
/* node's view index: under v->nodes, so under MESH_SWITCH_N; UINT32_MAX where the view has no such node */
static inline uint32_t mesh_switch_index(const struct mesh_switch_view *v,uint32_t node){
  for(uint32_t i=0;i<v->nodes;i++)if(v->node[i]==node)return i;
  return UINT32_MAX;
}
/* the view made from the fabric as the records give it, in the copy not published, then published: `nodes` node
   numbers (unsorted, at most MESH_SWITCH_N of them taken) and `links` directed pairs (node numbers, at most
   MESH_SWITCH_LINKS), each its cables, rate (bytes a ns, every cable) and latency (ns); a link's line carried over from
   the published copy where that copy has the link */
struct mesh_switch_fabric_link { uint32_t from,to,cables; double rate,latency; };
static inline void mesh_switch_view_set(struct mesh_switch *sw,uint32_t self,const uint32_t *nodes,uint32_t n,
                                        const struct mesh_switch_fabric_link *links,uint32_t count){
  const uint64_t version=atomic_load_explicit(&sw->view_version,memory_order_relaxed);
  const struct mesh_switch_view *was=sw->view+(version&1);
  struct mesh_switch_view *v=sw->view+((version+1)&1);
  uint32_t sorted[MESH_SWITCH_N],k=0;
  for(uint32_t i=0;i<n && k<MESH_SWITCH_N;i++){
    uint32_t j=k;while(j && sorted[j-1]>nodes[i]){sorted[j]=sorted[j-1];j--;}
    sorted[j]=nodes[i];k++;
  }
  v->self=self;v->nodes=k;v->ranks=was->ranks;memcpy(v->node,sorted,k*sizeof *sorted);
  uint32_t made=0;
  for(uint32_t l=0;l<count && made<MESH_SWITCH_LINKS;l++){
    struct mesh_switch_link *x=v->link+made++;
    x->from=links[l].from;x->to=links[l].to;x->cables=links[l].cables?links[l].cables:1;x->rate=links[l].rate;x->latency=links[l].latency;
    x->ends=(mesh_switch_index(v,x->from)&255)|(mesh_switch_index(v,x->to)&255)<<8;
    uint64_t line=0,stamp=0;
    for(uint32_t h=0;h<was->links;h++)if(was->link[h].from==x->from && was->link[h].to==x->to){
      line=atomic_load_explicit(&was->link[h].line,memory_order_relaxed);stamp=atomic_load_explicit(&was->link[h].stamp,memory_order_relaxed);
    }
    atomic_store_explicit(&x->line,line,memory_order_relaxed);atomic_store_explicit(&x->stamp,stamp,memory_order_relaxed);
  }
  v->links=made;
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
  atomic_store_explicit(&sw->view_version,version+1,memory_order_release);
}
static inline struct mesh_switch_link *mesh_switch_link_of(struct mesh_switch_view *v,uint32_t from,uint32_t to){
  for(uint32_t l=0;l<v->links;l++)if(v->link[l].from==from && v->link[l].to==to)return v->link+l;
  return NULL;
}

/* ---- The estimators ---------------------------------------------------------------------------------------------
   A lane's SEND completed: `bytes`, posted at `posted`, its service beginning at `start` (its post, or the lane's
   previous completion if later) and done at `done` (ns, the child's clock): the lane's queue less its bytes, its sums,
   and its estimate moved (an exponential average of the service rate, 1/8 a SEND, and of the wait) */
static inline void mesh_switch_sent(struct mesh_switch *sw,uint32_t lane,uint64_t bytes,uint64_t posted,uint64_t start,uint64_t done){
  struct mesh_switch_lane *x=sw->lane+lane;
  atomic_fetch_sub_explicit(&x->queued,bytes,memory_order_relaxed);
  atomic_fetch_add_explicit(&x->sends,1,memory_order_relaxed);atomic_fetch_add_explicit(&x->bytes,bytes,memory_order_relaxed);
  const uint64_t service=done>start?done-start:1,wait=start>posted?start-posted:0;
  atomic_fetch_add_explicit(&x->service,service,memory_order_relaxed);atomic_fetch_add_explicit(&x->wait,wait,memory_order_relaxed);
  float rate,waited;mesh_switch_unpack(atomic_load_explicit(&x->estimate,memory_order_relaxed),&rate,&waited);
  const float r=(float)bytes/(float)service;
  rate=rate>0?rate+(r-rate)/8:r;waited+=((float)wait-waited)/8;
  atomic_store_explicit(&x->estimate,mesh_switch_pack(rate,waited),memory_order_relaxed);
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
/* the lines a header carries (its `lines`, read once, as many as fit after `bytes` in `wire`) taken into the published
   view where newer than its own (a node's lines about its own links are the ones it measured: a link's line is only
   ever made by its sender) */
static inline void mesh_switch_lines_take(struct mesh_switch *sw,struct mesh_head *head,uint32_t bytes,uint64_t wire){
  struct mesh_switch_view *v=sw->view+mesh_switch_published(&sw->view_version);
  const uint32_t carried=__atomic_load_n(&head->lines,__ATOMIC_RELAXED),fit=mesh_switch_lines_fit(bytes,wire),lines=carried<fit?carried:fit;
  const struct mesh_switch_line *line=mesh_switch_lines_at(head,bytes);
  for(uint32_t i=0;i<lines;i++){
    const struct mesh_switch_line *a=line+i;
    if(a->from==v->self)continue;
    struct mesh_switch_link *x=mesh_switch_link_of(v,a->from,a->to);
    if(!x)continue;
    uint64_t was=atomic_load_explicit(&x->stamp,memory_order_relaxed);
    while(a->stamp>was)
      if(atomic_compare_exchange_weak_explicit(&x->stamp,&was,a->stamp,memory_order_relaxed,memory_order_relaxed)){
        atomic_store_explicit(&x->line,mesh_switch_pack(a->wait,a->spread),memory_order_relaxed);break;
      }
  }
}
/* the lines a header sends, from view v: this node's own links first (their wait the least bytes queued over the rate
   of the neighbour's lanes, INFINITY where the link is down: mesh_switch_refresh), then the view's others, newest
   first, as many as fit */
static inline uint32_t mesh_switch_lines_give(struct mesh_switch *sw,const struct mesh_switch_view *v,struct mesh_switch_line *line){
  uint32_t n=0;
  const uint64_t stamp=atomic_load_explicit(&sw->stamp,memory_order_relaxed);
  for(uint32_t l=0;l<v->links && n<MESH_SWITCH_LINES;l++){
    const struct mesh_switch_link *x=v->link+l;
    float wait,spread;mesh_switch_unpack(atomic_load_explicit(&x->line,memory_order_relaxed),&wait,&spread);
    if(x->from==v->self)line[n++]=(struct mesh_switch_line){x->from,x->to,stamp,wait,spread,(float)x->rate,(float)x->latency};
  }
  for(uint32_t l=0;l<v->links && n<MESH_SWITCH_LINES;l++){
    const struct mesh_switch_link *x=v->link+l;
    const uint64_t s=atomic_load_explicit(&x->stamp,memory_order_relaxed);
    if(x->from==v->self || !s)continue;
    float wait,spread;mesh_switch_unpack(atomic_load_explicit(&x->line,memory_order_relaxed),&wait,&spread);
    line[n++]=(struct mesh_switch_line){x->from,x->to,s,wait,spread,(float)x->rate,(float)x->latency};
  }
  return n;
}
_Static_assert(MESH_SWITCH_LANES==64,"a lane a bit, and every bit of a mask a lane");
/* the lanes live toward `peer`, bits over the switch's lanes, of every class (a cable's queue counts all its lanes:
   mesh_switch_lanes_among takes from the mask the lanes of its page's class alone) */
static inline uint64_t mesh_switch_lanes_mask(struct mesh_switch *sw,uint32_t peer){
  uint64_t mask=0;
  for(uint32_t l=0;l<MESH_SWITCH_LANES;l++)
    if(atomic_load_explicit(&sw->lane[l].live,memory_order_acquire) && atomic_load_explicit(&sw->lane[l].peer,memory_order_relaxed)==peer)mask|=UINT64_C(1)<<l;
  return mask;
}
/* the calling thread's own port (a port child's progress thread, mesh-link.c port_progress; UINT32_MAX elsewhere): of
   lanes whose waits are equal, one of its own port's is posted by this thread in this pass, any other's waits for its
   own child's pass (the handoff through the outbox) */
static __thread uint32_t mesh_switch_home=UINT32_MAX;
/* Of the lanes of `mask` (0: every lane; the refresh's: struct mesh_switch_weights `toward`, each read live again)
   live toward `peer` and of `frames` frames (a page goes on the lanes of its payload's class alone: the lanes of
   mesh_switch_class_lanes, no lane read for it), the one of the least wait with `extra` bytes more: its wait (ns),
   *best the lane; INFINITY and UINT32_MAX where none is.  A cable's
   lanes share it: a lane's wait is its cable's (port's) bytes queued over all its live lanes toward the peer, of every
   class, plus `extra`, over the cable's rate (`fallback_rate`, the view's per cable, else its lanes' estimates summed).
   Where none of those frames is found, every lane is read once more where the mask was not every lane (one come up
   since the refresh) or a lane of another class toward the peer was live (a link's lanes come up one after another,
   its short lane first, mesh-link.c lanes_open: a lane read live, by its acquire, shows every lane made live before
   it) */
static inline double mesh_switch_lanes_among(struct mesh_switch *sw,uint64_t mask,uint32_t peer,uint32_t frames,double fallback_rate,double extra,uint32_t *best){
  double least=INFINITY;*best=UINT32_MAX;
  const uint64_t every=~UINT64_C(0)>>(64-MESH_SWITCH_LANES),of_class=mesh_switch_class_lanes(frames);
  if(!mask)mask=every;
  enum { CABLES=MESH_SWITCH_LANES/(2*MESH_LINK_STREAMS) };
  for(uint32_t pass=0;pass<2;pass++,mask=every){
    double queued[CABLES]={0},estimate[CABLES]={0};uint64_t live=0;
    for(uint64_t k=mask;k;k&=k-1){
      const uint32_t l=(uint32_t)__builtin_ctzll(k);
      struct mesh_switch_lane *x=sw->lane+l;
      if(!atomic_load_explicit(&x->live,memory_order_acquire) || atomic_load_explicit(&x->peer,memory_order_relaxed)!=peer)continue;
      live|=UINT64_C(1)<<l;
      const uint32_t c=l/(2*MESH_LINK_STREAMS);
      queued[c]+=(double)atomic_load_explicit(&x->queued,memory_order_relaxed);
      float r,waited;mesh_switch_unpack(atomic_load_explicit(&x->estimate,memory_order_relaxed),&r,&waited);
      estimate[c]+=r>0?r:0;
    }
    for(uint64_t k=live&of_class;k;k&=k-1){
      const uint32_t l=(uint32_t)__builtin_ctzll(k),c=l/(2*MESH_LINK_STREAMS);
      const double each=fallback_rate>0?fallback_rate:estimate[c]>0?estimate[c]:1;
      const double wait=(queued[c]+extra)/each;
      const int home=c==mesh_switch_home,best_home=*best!=UINT32_MAX && *best/(2*MESH_LINK_STREAMS)==mesh_switch_home;
      if(wait<least || (wait==least && *best!=UINT32_MAX && (home>best_home || (home==best_home &&
                        atomic_load_explicit(&sw->lane[l].queued,memory_order_relaxed)<atomic_load_explicit(&sw->lane[*best].queued,memory_order_relaxed))))){least=wait;*best=l;}
    }
    if(*best!=UINT32_MAX || (mask==every && !live))break;
  }
  return least;
}
/* the least wait now toward `peer` on its lanes of `frames` frames, nothing added (mesh_switch_lanes_among over every
   lane): INFINITY where none of them is live */
static inline double mesh_switch_lanes_toward(struct mesh_switch *sw,uint32_t peer,uint32_t frames,double fallback_rate,uint32_t *best){
  return mesh_switch_lanes_among(sw,0,peer,frames,fallback_rate,0,best);
}

/* ---- The draw ---------------------------------------------------------------------------------------------------- */
static inline uint64_t mesh_switch_mix(uint64_t x){
  x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);
}
static inline double mesh_switch_uniform(uint64_t key){ return ((mesh_switch_mix(key)>>11)+0.5)/9007199254740992.0; }
static inline double mesh_switch_normal(uint64_t key){
  const double u=mesh_switch_uniform(key),v=mesh_switch_uniform(key^0x5bd1e995u);
  return sqrt(-2*log(u))*cos(6.283185307179586*v);
}
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
   *point the tables' refresh), 0 none or the tables not active.  A context's index stays under MESH_SWITCH_CONTEXTS
   whatever `contexts` the copy says */
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
  *point=x->point;*argmax=x->argmax;*scale=x->scale;
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
/* A page at this node (its header: its destinations and path) split over the node's out-neighbours for the
   destinations it carries on (`away`, `naway` of them: places in the header's dest[]): each group's lane (the
   neighbour's cable with the least bytes queued, among its lanes of `frames` frames, the class of the page's payload,
   never a lane of more) and its destinations (bits over `away`).
   The draw's record (split.h struct route_draw, as rdma/route reads it): for each carried destination d and neighbour
   k, the arrival through k (its lanes' wait now, the page's time on them, the link's latency, then the refresh's
   cost-to-go from k to d), whether k reaches d without coming back here (valid), and whether it is allowed from the
   page's path (off it, reaching d avoiding it: mask).
   The draw, at u = the page's seed (its source, its `seq`, its first carried destination's index):
   - where the policy's split tables are active and have the page's context (its source, its destinations' nodes, the
     ones carried here, each over this node's view: mesh_switch_split_w), the split is split.h's table under the
     context's w at u: exactly the trainer's action, *logp its log-probability;
   - else each destination i takes the neighbour k at which its masked, renormalised weights' distribution first passes
     u; where its weights give its allowed neighbours nothing, the allowed one of the least cost-to-go; *logp the
     length of u's interval that makes the split.
   A destination no allowed neighbour reaches (or not in the view) is in no group: its bit set in *unreachable.
   `record` (NULL: none) receives the record, *context the context drawn from (UINT32_MAX: the weights), *point the
   tables' refresh.  Returns the groups.
   The view's nodes, the weights' degree and each neighbour are read once, so a bound is the same wherever it is used,
   and every index stays within its array whatever a copy rewritten under the draw has in it: a view index under
   `nodes` (at most MESH_SWITCH_N); a neighbour k under `deg` (at most MESH_SWITCH_N); a neighbour's view index under
   MESH_SWITCH_N, as the refresh stores it (togo[nb] within togo), and under `nodes` before v->node[nb] is read; a
   lane under MESH_SWITCH_LANES (mesh_switch_lanes_among's, or none where the neighbour is no view node) */
static inline uint32_t mesh_switch_choose(struct mesh_switch *sw,const struct mesh_switch_view *v,const struct mesh_switch_weights *w,
                                          const struct mesh_head *head,const uint32_t *away,uint32_t naway,uint32_t frames,
                                          uint32_t *lane,uint64_t *group,uint32_t *unreachable,
                                          struct route_draw *record,float *logp,uint32_t *context,uint64_t *point){
  const uint32_t nodes_read=__atomic_load_n(&v->nodes,__ATOMIC_RELAXED),nodes=nodes_read<MESH_SWITCH_N?nodes_read:MESH_SWITCH_N;
  const uint32_t deg_read=__atomic_load_n(&w->deg,__ATOMIC_RELAXED),deg=deg_read<MESH_SWITCH_N?deg_read:MESH_SWITCH_N;
  const uint32_t self=mesh_switch_index(v,v->self);
  const uint64_t here=self<64?UINT64_C(1)<<self:0;
  *unreachable=0;
  uint64_t visited=here;
  for(uint32_t h=0;h<=head->hops && h<MESH_SWITCH_N;h++){const uint32_t i=mesh_switch_index(v,head->path[h]);if(i<64)visited|=UINT64_C(1)<<i;}
  const double u=mesh_switch_uniform(((uint64_t)head->source<<32)^head->seq*0x9e3779b97f4a7c15ULL^(naway?head->dest[away[0]].index:0));
  uint32_t choice[MESH_DESTS],node[MESH_DESTS],neighbour[MESH_SWITCH_N];
  uint32_t best_lane[MESH_SWITCH_N];double wait[MESH_SWITCH_N],latency[MESH_SWITCH_N];
  for(uint32_t k=0;k<deg;k++){
    const uint32_t nb=neighbour[k]=__atomic_load_n(&w->neighbour[k],__ATOMIC_RELAXED);
    const struct mesh_switch_link *x=nb<nodes?mesh_switch_link_of((struct mesh_switch_view *)v,v->self,v->node[nb]):NULL;
    /* the wait on the lane of the page's class with the page's own frames on it: its time to clear the lane (design:
       (backlog + bytes) / rate) */
    best_lane[k]=UINT32_MAX;
    wait[k]=nb<nodes?mesh_switch_lanes_among(sw,w->toward[k],v->node[nb],frames,x?x->rate/x->cables:0,(double)frames*MESH_FRAME,best_lane+k):INFINITY;
    latency[k]=x?x->latency:0;
  }
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
  double lo=0,hi=1;
  const int logged=record && nodes<=SPLIT_N && deg<=SPLIT_DEG;
  if(logged){memset(record,0,sizeof *record);record->deg=deg;record->unit=scale>0?scale:1;}
  uint32_t ntok=1;
  for(uint32_t j=0;j<naway;j++){
    const uint32_t d=node[j];
    choice[j]=UINT32_MAX;
    if(d>=nodes){*unreachable|=1u<<j;continue;}
    double total=0,cost=INFINITY;uint32_t cheapest=UINT32_MAX,allowed=0;
    uint32_t ok[MESH_SWITCH_N];
    for(uint32_t k=0;k<deg;k++){
      const uint32_t nb=neighbour[k];
      ok[k]=best_lane[k]!=UINT32_MAX && !(visited>>nb&1) && (nb==d || mesh_switch_reaches(v,nb,visited,d));
      if(logged){
        const int valid=nb==d || mesh_switch_reaches(v,nb,here,d);
        const uint32_t b=d*SPLIT_DEG+k;
        const double hop=wait[k]+latency[k];
        record->arrive[b]=(float)(hop+(nb==d?0:w->togo[nb][d]));
        if(valid)record->valid|=1u<<b;
        if(ok[k])record->mask|=1u<<b;
      }
      if(!ok[k])continue;
      allowed++;total+=w->w[d][k];
      const double c=wait[k]+latency[k]+(nb==d?0:w->togo[nb][d]);
      if(c<cost){cost=c;cheapest=k;}
    }
    if(!allowed){*unreachable|=1u<<j;continue;}
    ntok*=allowed;
    if(logged)record->S|=1u<<d;
    if(split)continue;
    /* the design's rule: the allowed neighbour of the earliest arrival at d, its lane's backlog and this page's
       frames on it, the link's latency, then the cost-to-go from it (the weights' draw where `weighted` is set) */
    if(!atomic_load_explicit(&sw->weighted,memory_order_relaxed)){choice[j]=cheapest;continue;}
    if(total>0){
      double at=0;
      for(uint32_t k=0;k<deg && choice[j]==UINT32_MAX;k++){
        if(!ok[k])continue;
        const double p=w->w[d][k]/total;
        if(u<at+p || at+p>=1-1e-12){choice[j]=k;if(at>lo)lo=at;if(at+p<hi)hi=at+p;}
        at+=p;
      }
      if(choice[j]==UINT32_MAX){choice[j]=cheapest;}
    }else choice[j]=cheapest;
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
    if(logp)*logp=hi>lo?(float)log(hi-lo):0;
  }
  uint32_t used=0;
  for(uint32_t j=0;j<naway;j++){
    if(choice[j]==UINT32_MAX){*unreachable|=1u<<j;continue;}
    const uint32_t l=best_lane[choice[j]];
    uint32_t g=0;while(g<used && lane[g]!=l)g++;
    if(g==used){lane[used]=l;group[used]=0;used++;}
    group[g]|=UINT64_C(1)<<j;
  }
  return used;
}
/* ---- The refresh -------------------------------------------------------------------------------------------------
   The weights' target from what the node knows now, for the refresh's page (`frames` frames, a class's): each hop's
   cost its wait (this node's own lanes of that class, their least queue over their rate; a remote link's line) plus
   the page's time on one cable plus its latency, and INFINITY where the link is down (none of those lanes live, or a
   line whose wait is not finite: no path crosses it).  A policy (`policy` NULL: the default) writes target[d][k], a
   distribution over the neighbours for each destination, from that state (the view, the cost-to-go, each neighbour's
   wait now); the default is Thompson sampling over paths: `draws` draws of every hop's wait (its line's mean plus its
   spread times a normal draw), each destination's share of a neighbour the draws in which the earliest path to it
   leaves through that neighbour. */
struct mesh_switch_state {
  const struct mesh_switch_view *view;
  uint32_t self,deg,neighbour[MESH_SWITCH_N];
  double hop[MESH_SWITCH_LINKS],spread[MESH_SWITCH_LINKS];      /* each view link's cost (ns) and its spread */
  /* each of this node's own links' wait now (ns: its lanes' of the refresh's page's class, their least queue over their
     rate; INFINITY where none of them toward its peer is live, the link down), made once a refresh for the state, the
     lines and the tokens alike; 0 for another node's link */
  double wait[MESH_SWITCH_LINKS];
  double first[MESH_SWITCH_N];                                   /* each neighbour's hop from here, now */
  float togo[MESH_SWITCH_N][MESH_SWITCH_N];
  uint64_t stamp;
  double page;
};
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
  /* target[d][k]; 0, or -1: the default's target is taken instead */
  int (*target)(void *context,const struct mesh_switch_state *state,float target[MESH_SWITCH_N][MESH_SWITCH_N]);
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
/* the refresh's state (its view v, the published copy) for a page of `frames` frames: each own link's wait from its
   lanes of those frames, once; each link down (a wait not finite: an own link none of whose lanes is live, a remote one
   whose line says so) a hop of INFINITY and no spread; a neighbour each own link not down whose peer is a view node
   (its index under v->nodes, so under MESH_SWITCH_N) */
static inline void mesh_switch_state_make(struct mesh_switch *sw,const struct mesh_switch_view *v,uint32_t frames,struct mesh_switch_state *s){
  memset(s,0,sizeof *s);
  s->view=v;s->page=(double)frames*MESH_FRAME;s->self=mesh_switch_index(v,v->self);s->stamp=atomic_load_explicit(&sw->stamp,memory_order_relaxed);
  for(uint32_t l=0;l<v->links;l++){
    const struct mesh_switch_link *x=v->link+l;
    float wait,spread;mesh_switch_unpack(atomic_load_explicit(&x->line,memory_order_relaxed),&wait,&spread);
    if(x->from==v->self){uint32_t best;const double now=mesh_switch_lanes_toward(sw,x->to,frames,x->rate/x->cables,&best);
                         s->wait[l]=best==UINT32_MAX?INFINITY:now;wait=(float)s->wait[l];spread=0;}
    const int down=!isfinite(wait);
    s->hop[l]=down?INFINITY:wait+s->page/(x->rate/x->cables)+x->latency;s->spread[l]=down?0:spread;
    const uint32_t b=mesh_switch_index(v,x->to);
    if(x->from==v->self && !down && b<v->nodes && s->deg<MESH_SWITCH_N){s->neighbour[s->deg]=b;s->first[s->deg]=s->hop[l];s->deg++;}
  }
  double at[MESH_SWITCH_N];
  for(uint32_t a=0;a<v->nodes;a++){mesh_switch_arrivals(v,a,0,0,s->hop,at);for(uint32_t b=0;b<v->nodes;b++)s->togo[a][b]=isfinite(at[b])?(float)at[b]:1e12f;}
}
static inline void mesh_switch_thompson(const struct mesh_switch_state *s,uint32_t draws,uint64_t seed,float target[MESH_SWITCH_N][MESH_SWITCH_N]){
  const struct mesh_switch_view *v=s->view;
  uint32_t wins[MESH_SWITCH_N][MESH_SWITCH_N];memset(wins,0,sizeof wins);
  double hop[MESH_SWITCH_LINKS],at[MESH_SWITCH_N],best[MESH_SWITCH_N];uint32_t by[MESH_SWITCH_N],firstlink[MESH_SWITCH_N];
  /* each neighbour's link from here; and whether any hop is uncertain (none: every draw is the same, one stands for all) */
  int spread=0;
  for(uint32_t l=0;l<v->links;l++)spread|=s->spread[l]!=0;
  for(uint32_t k=0;k<s->deg;k++){firstlink[k]=UINT32_MAX;for(uint32_t l=0;l<v->links;l++)if(v->link[l].from==v->self && (v->link[l].ends>>8&255)==s->neighbour[k])firstlink[k]=l;}
  const uint32_t made=spread?draws:(draws?1:0);
  for(uint32_t r=0;r<made;r++){
    for(uint32_t l=0;l<v->links;l++){const double h=s->spread[l]!=0?s->hop[l]+s->spread[l]*mesh_switch_normal(seed^((uint64_t)r<<32)^l):s->hop[l];hop[l]=h>0?h:0;}
    for(uint32_t d=0;d<v->nodes;d++){best[d]=INFINITY;by[d]=UINT32_MAX;}
    for(uint32_t k=0;k<s->deg;k++){
      const double first=firstlink[k]!=UINT32_MAX?hop[firstlink[k]]:s->first[k];
      mesh_switch_arrivals(v,s->neighbour[k],first,UINT64_C(1)<<s->self,hop,at);
      for(uint32_t d=0;d<v->nodes;d++)if(at[d]<best[d]){best[d]=at[d];by[d]=k;}
    }
    for(uint32_t d=0;d<v->nodes;d++)if(by[d]!=UINT32_MAX)wins[d][by[d]]++;
  }
  for(uint32_t d=0;d<v->nodes;d++)for(uint32_t k=0;k<MESH_SWITCH_N;k++)target[d][k]=k<s->deg && made?(float)wins[d][k]/(float)made:0;
}
static inline double mesh_switch_time_scale(const struct mesh_switch_view *v,double page);
/* one refresh, on the refresh thread (the writer of the view, the weights and the split tables, so their published
   copies stand under it but for the view's lines): the state over the published view, the target (the policy's, else
   Thompson's), the residual (target less the published weights) applied at the rate eta (1: the target itself), the
   weights written whole in the copy not published and published, the split tables' time scale made again where the view
   changed, this node's own links' lines from the state's waits, this node's stamp moved.  `frames`: the refresh's
   page's (MESH_SWITCH_PAGE) */
static inline void mesh_switch_refresh(struct mesh_switch *sw,const struct mesh_switch_policy *policy,uint32_t draws,float eta,uint64_t seed,
                                       uint32_t frames){
  const uint64_t view_ver=atomic_load_explicit(&sw->view_version,memory_order_relaxed),weights_ver=atomic_load_explicit(&sw->weights_version,memory_order_relaxed);
  struct mesh_switch_view *v=sw->view+(view_ver&1);
  const struct mesh_switch_weights *was=sw->weights+(weights_ver&1);
  struct mesh_switch_weights *w=sw->weights+((weights_ver+1)&1);
  struct mesh_switch_state s;mesh_switch_state_make(sw,v,frames,&s);
  float target[MESH_SWITCH_N][MESH_SWITCH_N];memset(target,0,sizeof target);
  if(!atomic_load_explicit(&sw->weighted,memory_order_relaxed) && !(policy && policy->target)){
    /* no draw reads the weights (the earliest arrival routes): they stand, and the refresh is the cost-to-go and the
       lines alone (Thompson's draws cost 10 us a refresh of the refresh thread's) */
    for(uint32_t d=0;d<MESH_SWITCH_N;d++)for(uint32_t k=0;k<MESH_SWITCH_N;k++)target[d][k]=was->w[d][k];
    eta=1;
  }else if(!policy || !policy->target || policy->target(policy->context,&s,target))mesh_switch_thompson(&s,draws,seed,target);
  const int moved=was->deg!=s.deg || memcmp(was->neighbour,s.neighbour,sizeof s.neighbour);
  w->nodes=v->nodes;w->deg=s.deg;memcpy(w->neighbour,s.neighbour,sizeof s.neighbour);
  for(uint32_t k=0;k<MESH_SWITCH_N;k++)w->toward[k]=k<s.deg?mesh_switch_lanes_mask(sw,v->node[s.neighbour[k]]):0;
  memset(w->w,0,sizeof w->w);memset(w->togo,0,sizeof w->togo);
  for(uint32_t d=0;d<v->nodes;d++){
    double sum=0;
    for(uint32_t k=0;k<s.deg;k++){
      const float now=moved?0:was->w[d][k],residual=target[d][k]-now;
      w->w[d][k]=moved?target[d][k]:now+eta*residual;sum+=w->w[d][k];
    }
    for(uint32_t k=0;k<s.deg && sum>0;k++)w->w[d][k]=(float)(w->w[d][k]/sum);
    for(uint32_t b=0;b<v->nodes;b++)w->togo[d][b]=s.togo[d][b];
  }
  w->refreshes=was->refreshes+1;
  atomic_store_explicit(&sw->weights_version,weights_ver+1,memory_order_release);
  const uint64_t splits_ver=atomic_load_explicit(&sw->splits_version,memory_order_relaxed);
  const struct mesh_switch_splits *tables=sw->splits+(splits_ver&1);
  if(tables->scale_view!=view_ver+1){   /* the features' time scale made again as the view changes, the tables as they stand */
    struct mesh_switch_splits *x=sw->splits+((splits_ver+1)&1);
    memcpy(x,tables,sizeof *x);
    x->scale=(float)mesh_switch_time_scale(v,ROUTE_PAGE);x->scale_view=view_ver+1;
    atomic_store_explicit(&sw->splits_version,splits_ver+1,memory_order_release);
  }
  /* this node's own links' lines, for the headers it sends: each one's wait now (the state's: INFINITY where the link
     is down, so no node routes through it), its spread the wait's change (0 where the link is down now or was at the
     line before: a down link's wait has no spread) */
  for(uint32_t l=0;l<v->links;l++){
    struct mesh_switch_link *x=v->link+l;
    if(x->from!=v->self)continue;
    const double now=s.wait[l];
    float before,spread;mesh_switch_unpack(atomic_load_explicit(&x->line,memory_order_relaxed),&before,&spread);
    spread=isfinite(now) && isfinite(before)?spread+((float)fabs(now-before)-spread)/4:0;
    atomic_store_explicit(&x->line,mesh_switch_pack((float)now,spread),memory_order_relaxed);
  }
  atomic_fetch_add_explicit(&sw->stamp,1,memory_order_relaxed);
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
   one more, the page it named before dropped), the entry's use of the page counted first; 1 where it was set.  An index
   past the table is set nothing and counted (`malformed`: its word would lie outside the table).  An index no client
   has now is set as any other: the page goes back as the index is next allocated (mesh-call.c mesh_pages makes its
   word new and drops the page it named) or as a later landing there replaces it, and the lane it landed on keeps its
   receives meanwhile (mesh-link.c link_receive: a landing whose class has no free page posts the landed page again); a
   lane whose fill comes after its class's stack is empty starts with fewer receives than its share (mesh-link.c
   switch_receives).  The index is read once: the one bounded is the one set */
static inline int mesh_switch_set(struct hdr *m,struct mesh_switch *sw,uint32_t pd,const struct mesh_head *head,uint32_t i,uint32_t ring){
  const uint32_t index=__atomic_load_n(&head->dest[i].index,__ATOMIC_RELAXED);
  if(index>=m->entries){MESH_SWITCH_COUNT(sw,ring,malformed,1);return 0;}
  mesh_pd_use(m,pd);
  mesh_pd_drop(m,mesh_entry_put(m,index,pd,1));
  MESH_SWITCH_COUNT(sw,ring,set,1);
  return 1;
}
/* The forward: page pd's destinations that this node carries on (`away`: places in its header's dest[]) split
   (mesh_switch_choose) into groups, a lane each, one group a peer, and the page itself pushed onto every group's lane:
   one page and no copy, each group's destinations naming that lane's peer their `next` (a receiver reads only the
   destinations it is the next of).  Before any push the header's shared parts are written: each destination no group
   carries named no one's next (another hop's SEND carries it, it is set here, or nothing reaches it), the sender, and
   the lines (as many as the frames of the payload's class fit).  Each group adds a use of the page and its frames to
   its lane's queue; a push the outbox has no room for gives that use back, counted (`lost`).  A destination no allowed
   neighbour reaches is not carried, counted (`unreachable`).  `bytes` the payload's bytes as the caller bounded them by
   the page's class (the write's, or the landed header's read once): the class's frames, so the lanes the page goes on,
   each SEND's frames and the lines' place, come from them and never from the header's word read again (every client
   maps the region writable), so no SEND covers more than the page's own frames.  `ring` the caller's log ring and
   counters (its port's), `now` its clock, `hand` its hand (a port child's) */
static inline void mesh_switch_forward(struct hdr *m,struct mesh_switch *sw,uint32_t pd,struct mesh_head *head,uint32_t bytes,const uint32_t *away,
                                       uint32_t naway,uint64_t now,uint32_t ring,struct mesh_switch_hand *hand){
  const uint32_t self=m->node,frames=mesh_class_frames(bytes);
  const int logging=mesh_switch_sampled(sw,head);
  uint32_t groups=0,lane[MESH_DESTS],unreachable=0;uint64_t group[MESH_DESTS];
  struct route_draw draw;float logp=0;uint32_t context=UINT32_MAX;uint64_t point=0;
  if(logging)memset(&draw,0,sizeof draw);
  /* the draw over the published view and weights where they lie, each its copy as its version names it, read once:
     no retry and no copy of either a page (a copy written again under the draw at worst routes the page wrong) */
  const struct mesh_switch_view *v=sw->view+mesh_switch_published(&sw->view_version);
  const struct mesh_switch_weights *weights=sw->weights+mesh_switch_published(&sw->weights_version);
  groups=mesh_switch_choose(sw,v,weights,head,away,naway,frames,lane,group,&unreachable,logging?&draw:NULL,&logp,&context,&point);
  if(logging && groups){
    struct mesh_switch_record r;memset(&r,0,sizeof r);
    r.kind=MESH_LOG_DRAW;r.node=self;r.seq=head->seq;r.at=now;r.draw=draw;r.logp=logp;r.source=head->source;
    r.dest=head->dest[away[0]].node;r.index=head->dest[away[0]].index;r.bytes=bytes;r.context=context;r.point=(uint32_t)point;
    mesh_switch_log_put(sw,ring,&r);
  }
  if(unreachable)MESH_SWITCH_COUNT(sw,ring,unreachable,__builtin_popcount(unreachable));
  /* one group a peer: a receiver reads every destination naming it, so two lanes toward one peer would each carry all of them */
  for(uint32_t g=0;g<groups;g++){
    const uint32_t peer=atomic_load_explicit(&sw->lane[lane[g]].peer,memory_order_relaxed);
    for(uint32_t h=g+1;h<groups;)
      if(atomic_load_explicit(&sw->lane[lane[h]].peer,memory_order_relaxed)==peer){group[g]|=group[h];groups--;lane[h]=lane[groups];group[h]=group[groups];}
      else h++;
  }
  MESH_SWITCH_COUNT(sw,ring,forwards,1);MESH_SWITCH_COUNT(sw,ring,groups,groups);
  if(!groups)return;
  uint32_t carried=0;
  for(uint32_t g=0;g<groups;g++)for(uint32_t j=0;j<naway;j++)if(group[g]>>j&1)carried|=1u<<away[j];
  for(uint32_t i=0;i<head->dests && i<MESH_DESTS;i++)if(!(carried>>i&1))head->dest[i].next=UINT32_MAX;
  head->from=self;
  struct mesh_switch_line lines[MESH_SWITCH_LINES];
  const uint32_t nlines=mesh_switch_lines_give(sw,v,lines),fit=mesh_switch_lines_fit(bytes,(uint64_t)frames*MESH_FRAME),
                 sent=nlines<fit?nlines:fit;
  head->lines=sent;memcpy(mesh_switch_lines_at(head,bytes),lines,sent*sizeof *lines);
  /* each group's destinations its peer's, with the cost they carry on (the lane's queue over its rate, its frames' time
     on the wire), all before the first push makes the page another process's to post */
  for(uint32_t g=0;g<groups;g++){
    const uint32_t l=lane[g],peer=atomic_load_explicit(&sw->lane[l].peer,memory_order_relaxed);
    float rate,waited;mesh_switch_unpack(atomic_load_explicit(&sw->lane[l].estimate,memory_order_relaxed),&rate,&waited);
    const double each=rate>0?rate:1;
    const float queued=(float)((double)atomic_load_explicit(&sw->lane[l].queued,memory_order_relaxed)/each),
                transit=(float)((double)mesh_switch_lane_bytes(l)/each);
    for(uint32_t j=0;j<naway;j++)if(group[g]>>j&1){struct mesh_hop *d=head->dest+away[j];d->next=peer;d->queued+=queued;d->transit+=transit;}
  }
  for(uint32_t g=0;g<groups;g++){
    const uint32_t l=lane[g];
    const uint64_t wire=mesh_switch_lane_bytes(l);
    mesh_pd_use(m,pd);
    atomic_fetch_add_explicit(&sw->lane[l].queued,wire,memory_order_relaxed);
    if(mesh_switch_push(sw,l,pd,now,hand))continue;
    atomic_fetch_sub_explicit(&sw->lane[l].queued,wire,memory_order_relaxed);
    mesh_pd_drop(m,pd);
    MESH_SWITCH_COUNT(sw,ring,lost,1);
  }
}
/* A client's write carried (the take): its page (`pd`, the page the write's ring use is on) given the write's bytes and
   its header (MESH_HEAD_MAGIC, this node its source, its sender and its path's first node, the node's header counter
   its `seq`, every destination's next this node), each destination on this node set in the table (mesh_switch_set),
   the others forwarded (mesh_switch_forward).  A cell naming no page of the pool, no destination or more than
   MESH_DESTS, or more bytes than its page's payload is carried nowhere, counted (`malformed`) */
static inline void mesh_switch_carry(struct hdr *m,struct mesh_switch *sw,const struct mesh_write *w,uint64_t now,uint32_t ring,struct mesh_switch_hand *hand){
  const uint32_t self=m->node,pd=w->pd,dests=w->dests;
  if(!mesh_pd_valid(m,pd) || !dests || dests>MESH_DESTS || w->bytes>MESH_PAYLOAD(mesh_pd_at(m,pd)->cls)){
    MESH_SWITCH_COUNT(sw,ring,malformed,1);
    return;
  }
  MESH_SWITCH_COUNT(sw,ring,taken,1);
  atomic_store_explicit(&mesh_pd_at(m,pd)->bytes,w->bytes,memory_order_relaxed);
  struct mesh_head *head=mesh_head_of(m,pd);
  head->magic=MESH_HEAD_MAGIC;head->source=self;head->bytes=w->bytes;head->hops=0;head->dests=dests;head->lines=0;head->from=self;
  memset(head->path,0,sizeof head->path);head->path[0]=self;
  head->seq=atomic_fetch_add_explicit(&sw->writes,1,memory_order_relaxed);
  for(uint32_t i=0;i<dests;i++)head->dest[i]=(struct mesh_hop){w->dest[i].node,w->dest[i].index,self,0,0};
  uint32_t away[MESH_DESTS],naway=0;
  for(uint32_t i=0;i<dests;i++){
    if(head->dest[i].node!=self)away[naway++]=i;
    else mesh_switch_set(m,sw,pd,head,i,ring);
  }
  if(naway)mesh_switch_forward(m,sw,pd,head,w->bytes,away,naway,now,ring,hand);
}
/* Writes taken from client slot `slot`'s ring and carried (mesh_switch_carry), at most `most`, by the port child whose
   hand is `hand`: each ring position named in the hand before its compare-exchange and marked won once won, the write
   carried from its cell, which stays full meanwhile (so a child that exits mid-write leaves the cell for its parent,
   or the next bridge, to settle: mesh_switch_left_settle), then the cell given back and the hand cleared, then the
   write's ring use dropped.  The ring's one taker is the port child its
   parent names for it (struct mesh_switch `taker`); as the parent names another, the child it was named for may still
   be taking, and each position is one compare-exchange's, so each write is one child's.  Returns the writes taken */
static inline uint32_t mesh_switch_ring_dequeue(struct hdr *m,struct mesh_switch *sw,uint32_t slot,uint32_t most,uint64_t now,uint32_t ring,struct mesh_switch_hand *hand){
  struct mesh_ring *r=mesh_ring_at(m,slot);
  uint32_t taken=0;
  uint64_t at=atomic_load_explicit(&r->take,memory_order_relaxed);
  while(taken<most){
    struct mesh_cell *cell=&r->cell[at%MESH_RING];
    const uint64_t full=2*(at/MESH_RING)+1,seq=atomic_load_explicit(&cell->sequence,memory_order_acquire);
    if(seq<full)break;
    if(seq>full){at=atomic_load_explicit(&r->take,memory_order_relaxed);continue;}
    const uint64_t named=((uint64_t)(slot+1)<<56)|at;
    atomic_store_explicit(&hand->take,named,memory_order_relaxed);
    if(!atomic_compare_exchange_weak_explicit(&r->take,&at,at+1,memory_order_acq_rel,memory_order_relaxed))continue;
    atomic_store_explicit(&hand->take,named|MESH_HAND_WON,memory_order_relaxed);
    const struct mesh_write w=cell->w;
    mesh_switch_carry(m,sw,&w,now,ring,hand);
    atomic_store_explicit(&cell->sequence,full+1,memory_order_release);
    atomic_store_explicit(&hand->take,0,memory_order_release);
    mesh_pd_drop(m,w.pd);
    taken++;
    at=atomic_load_explicit(&r->take,memory_order_relaxed);
  }
  atomic_store_explicit(&hand->take,0,memory_order_release);
  return taken;
}
/* A receive landed in page pd on lane l (the page of the lane's class): the page live, its one use the landing's.  A
   header that is not one (its magic, its bytes past the page's payload, no destination or more than MESH_DESTS, a
   path past MESH_SWITCH_N) is counted (`malformed`) and the page given back.  Else the page given the header's bytes,
   this node appended to its path, its lines taken (as many as the lane's frames fit), each destination here whose
   next this node is set in the table (mesh_switch_set) and its record made where the page is sampled, the
   destinations elsewhere whose next this node is forwarded (mesh_switch_forward), and the landing's use dropped.  The
   header's bytes, destinations and hops are each read once, and the values bounded are the ones used (every client
   maps the region writable, and a local reader has the page once a destination here is set).  `now` the child's clock
   at the landing */
static inline void mesh_switch_landed(struct hdr *m,struct mesh_switch *sw,uint32_t l,uint32_t pd,uint64_t now,uint32_t ring,struct mesh_switch_hand *hand){
  struct mesh_pd *p=mesh_pd_at(m,pd);
  atomic_store_explicit(&p->uses,1,memory_order_relaxed);
  atomic_store_explicit(&p->state,MESH_PD_LIVE,memory_order_relaxed);
  atomic_fetch_add_explicit(&sw->lane[l].landed,1,memory_order_relaxed);
  MESH_SWITCH_COUNT(sw,ring,landed,1);
  struct mesh_head *head=mesh_head_of(m,pd);
  const uint32_t self=m->node,bytes=__atomic_load_n(&head->bytes,__ATOMIC_RELAXED),dests=__atomic_load_n(&head->dests,__ATOMIC_RELAXED),
                 hops=__atomic_load_n(&head->hops,__ATOMIC_RELAXED);
  if(head->magic!=MESH_HEAD_MAGIC || bytes>MESH_PAYLOAD(p->cls) || !dests || dests>MESH_DESTS || hops>=MESH_SWITCH_N-1){
    MESH_SWITCH_COUNT(sw,ring,malformed,1);
    mesh_pd_drop(m,pd);
    return;
  }
  atomic_store_explicit(&p->bytes,bytes,memory_order_relaxed);
  head->hops=hops+1;head->path[hops+1]=self;
  mesh_switch_lines_take(sw,head,bytes,mesh_switch_lane_bytes(l));
  const int logging=mesh_switch_sampled(sw,head);
  uint32_t away[MESH_DESTS],naway=0;
  for(uint32_t i=0;i<dests;i++){
    const struct mesh_hop *d=head->dest+i;
    if(d->next!=self)continue;
    if(d->node!=self){away[naway++]=i;continue;}
    if(!mesh_switch_set(m,sw,pd,head,i,ring) || !logging)continue;
    /* a page's presence at each destination here, recorded as it is set: the cost it carried in-band (each hop's
       queueing on that hop's own clock, the wire's time) */
    struct mesh_switch_record r;memset(&r,0,sizeof r);
    r.kind=MESH_LOG_SET;r.node=self;r.seq=head->seq;r.at=now;r.latency=d->queued+d->transit;r.queueing=d->queued;
    r.source=head->source;r.dest=self;r.index=d->index;r.bytes=bytes;r.context=UINT32_MAX;
    mesh_switch_log_put(sw,ring,&r);
  }
  if(naway)mesh_switch_forward(m,sw,pd,head,bytes,away,naway,now,ring,hand);
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
/* whether another child's hand names the position `named` (its MESH_HAND_WON aside) in its take (`push` 0) or its
   push (`push` 1): its child may have won it */
static inline int mesh_switch_hand_names(const struct mesh_switch *sw,uint64_t named,int push){
  int any=0;
  for(uint32_t p=0;p<MESH_PORTS;p++)
    any|=(atomic_load_explicit(push?&sw->hand[p].push:&sw->hand[p].take,memory_order_acquire)&~MESH_HAND_WON)==(named&~MESH_HAND_WON);
  return any;
}
/* A port child's hand as the child left it at its end (`h`), settled as far as it can be now, by the parent's event
   loop (`ring` its counters); what is settled is cleared, and 1 returned once nothing of it is left.  `won` 1: only
   what the child had won (MESH_HAND_WON), so that a cell two dead children's hands name is settled first by its
   winner's.
   - Its take: where the cell is still full past the ring's take and the child had won it, or (its win not marked yet)
     no other hand names that position, the write was carried no further than it got: its ring use dropped, counted
     (`lost`), and its cell given back.  A cell given back, or one past the take that no one won: nothing.
   - Its push of `page`: where the child has its cell unfilled (won, or no other hand names it), the cell made a
     hole; where it won no cell, nothing was pushed.  Either way the page's use for the push dropped, counted (`lost`),
     and its frames off the lane's queue.  A cell it filled: the outbox's.
   Where another hand names the position, it is looked at again later (the parent's next turn): the other child ends
   its own take or push, or its own end is settled */
static inline int mesh_switch_left_settle(struct hdr *m,struct mesh_switch *sw,struct mesh_switch_left *h,int won,uint32_t ring){
  if(h->take && (!won || h->take&MESH_HAND_WON)){
    const uint32_t slot=(uint32_t)(h->take>>56)-1u;
    const uint64_t at=h->take&MESH_HAND_AT,full=2*(at/MESH_RING)+1;
    int later=0;
    if(slot<MESH_CLIENTS){
      struct mesh_ring *r=mesh_ring_at(m,slot);
      struct mesh_cell *cell=&r->cell[at%MESH_RING];
      /* read in this order: the ring's take, the other hands, the cell (a child names its position in its hand from
         before its win until after its cell is given back, so a cell read full where no other hand names it is no
         other taker's) */
      if(atomic_load_explicit(&r->take,memory_order_acquire)>at){
        const int other=mesh_switch_hand_names(sw,h->take,0);
        if(atomic_load_explicit(&cell->sequence,memory_order_acquire)==full){
          if(!(h->take&MESH_HAND_WON) && other)later=1;
          else{
            mesh_pd_drop(m,cell->w.pd);
            MESH_SWITCH_COUNT(sw,ring,lost,1);
            atomic_store_explicit(&cell->sequence,full+1,memory_order_release);
          }
        }
      }
    }
    if(!later)h->take=0;
  }
  if(h->push && (!won || h->push&MESH_HAND_WON)){
    const uint32_t l=(uint32_t)(h->push>>56)-1u;
    const uint64_t at=h->push&MESH_HAND_AT;
    int later=0;
    if(l<MESH_SWITCH_LANES){
      struct mesh_switch_lane *x=sw->lane+l;
      struct mesh_switch_cell *c=sw->ring[l]+at%MESH_SWITCH_RING;
      const int pushed_past=atomic_load_explicit(&x->tail,memory_order_acquire)>at;
      const int other=mesh_switch_hand_names(sw,h->push,1);
      const int unfilled=pushed_past && atomic_load_explicit(&c->sequence,memory_order_acquire)==at;
      int hole=0;
      if(h->push&MESH_HAND_WON)hole=unfilled;
      else if(unfilled && other)later=1;
      else hole=unfilled;
      if(hole){c->pd=MESH_ABSENT;atomic_store_explicit(&c->sequence,at+1,memory_order_release);}
      if(!later && (hole || !(h->push&MESH_HAND_WON)) && mesh_pd_valid(m,h->page)){
        atomic_fetch_sub_explicit(&x->queued,mesh_switch_lane_bytes(l),memory_order_relaxed);
        mesh_pd_drop(m,h->page);
        MESH_SWITCH_COUNT(sw,ring,lost,1);
      }
    }
    if(!later)h->push=0;
  }
  return !h->take && !h->push;
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
   - each lane's wait its line's: a remote link's as the pages carried it, each of this node's own links' the wait the
     refresh just made of its lanes (mesh_switch_refresh's lines, from its state: mesh-link.c switch_watch calls the
     refresh, then this, on the refresh thread); duplex share 1 (a bridge sees no other node's reverse traffic); an own
     link that is down (its wait not finite) no lane of the node now, so no token, as it is no neighbour of the
     weights' (mesh_switch_state_make);
   - the cost-to-go the last refresh's (the weights'), the bytes in flight and posted this node's own lanes' queues;
   - the per-pattern features of what is left to send, waiting and received, and the live collectives, 0: the bridge
     carries writes, not collectives.
   Every token is route.h's route_lane_token / route_context_token over that state.  The refresh thread's (the writer of
   the view and the weights): their published copies read in place */
static inline void mesh_switch_tokens_make(struct mesh_switch *sw,uint64_t now,uint64_t point,struct mesh_switch_tokens *out){
  const struct mesh_switch_view *v=sw->view+mesh_switch_published(&sw->view_version);
  const struct mesh_switch_weights *w=sw->weights+mesh_switch_published(&sw->weights_version);
  const uint32_t self=mesh_switch_index(v,v->self),ranks=mesh_switch_ranks(v);
  static __thread struct route_stream s;static __thread struct route_state st;
  memset(&s,0,sizeof s);memset(&st,0,sizeof st);
  s.f.nodes=(int)(v->nodes<ROUTE_N?v->nodes:ROUTE_N);s.f.lanes=(int)(v->links<ROUTE_L?v->links:ROUTE_L);s.f.duplex=1;
  for(int l=0;l<s.f.lanes;l++){
    const struct mesh_switch_link *x=v->link+l;
    s.f.from[l]=(int)mesh_switch_index(v,x->from);s.f.to[l]=(int)mesh_switch_index(v,x->to);s.f.cables[l]=(int)x->cables;s.f.rate[l]=x->rate;s.f.latency[l]=x->latency;
  }
  for(uint32_t a=0;a<v->nodes && a<ROUTE_N;a++)if(ranks>>a&1)s.f.compute[s.f.ncompute++]=(int)a;
  s.unit=mesh_switch_time_scale(v,ROUTE_PAGE);
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
    float wait,spread;mesh_switch_unpack(atomic_load_explicit(&v->link[l].line,memory_order_relaxed),&wait,&spread);
    st.wait[l]=wait;st.dup[l]=1;
  }
  for(uint32_t a=0;a<ROUTE_N;a++)for(uint32_t b=0;b<ROUTE_N;b++)st.togo[a][b]=a<v->nodes && b<v->nodes?(a==b?0:w->togo[a][b]):1e12;
  double queued=0;
  for(uint32_t l=0;l<MESH_SWITCH_LANES;l++)if(atomic_load_explicit(&sw->lane[l].live,memory_order_relaxed))queued+=(double)atomic_load_explicit(&sw->lane[l].queued,memory_order_relaxed);
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
   published (their time scale's view the published copy's), then published; active from now on */
static inline void mesh_switch_splits_publish(struct mesh_switch *sw,const struct mesh_switch_context *key,uint32_t n,const float (*w)[SPLIT_FS],
                                              uint64_t point,float scale,int argmax){
  const uint64_t version=atomic_load_explicit(&sw->splits_version,memory_order_relaxed);
  struct mesh_switch_splits *x=sw->splits+((version+1)&1);
  if(n>MESH_SWITCH_CONTEXTS)n=MESH_SWITCH_CONTEXTS;
  x->contexts=n;memcpy(x->key,key,n*sizeof *key);memcpy(x->w,w,n*sizeof *w);x->point=point;x->scale=scale;x->argmax=argmax?1:0;x->active=1;
  x->scale_view=sw->splits[version&1].scale_view;
  atomic_store_explicit(&sw->splits_version,version+1,memory_order_release);
}
/* one refresh of the trained policy: its tokens from this node's state, each context's w from the policy, published */
static inline int mesh_switch_policy_refresh(struct mesh_switch *sw,const struct mesh_switch_policy *policy,uint64_t now,uint64_t point,int argmax,
                                             struct mesh_switch_tokens *scratch){
  if(!policy || !policy->splits)return -1;
  mesh_switch_tokens_make(sw,now,point,scratch);
  static __thread float w[MESH_SWITCH_CONTEXTS][SPLIT_FS];
  if(policy->splits(policy->context,scratch,w))return -1;
  mesh_switch_splits_publish(sw,scratch->key,scratch->contexts,(const float (*)[SPLIT_FS])w,point,scratch->scale,argmax);
  return 0;
}

#endif
