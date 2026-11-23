/* The switch (design/PAGE-TABLE-SPEC-20261122.md section 5): what every node's bridge does with a page, as functions of
   its region.  The bridge's port children (mesh-link.c port_pass) and its parent (the refresh thread, a child's
   reclaim, a region's adoption) are its callers; none of it reads a clock, does I/O or posts to hardware: the caller
   passes the time and posts what these functions hand it.

   A page crosses as one SEND from its own memory (mesh.h struct mesh_pd): its first MESH_HEAD bytes the header the
   bridge writes (struct mesh_head: the node that took the write, its destinations as nodes and indices of their page
   tables, the nodes it visited), then its payload, then the lines of lane state it carries in its last frame's slack.
   A SEND fills one receive posted into a page of the lane's class (TN3205: both ends post the same frame count), and
   the lane's child posts that receive again at once (mesh-link.c link_receive: into a free page, else into the page
   that landed, whose landing is then lost), so a lane's receives do not run out while every page of its class is held.
   - Take.  A client's write (mesh.h struct mesh_write, from its ring) names its page and its destinations: the bridge
     writes the page's header, sets the table entry of each destination on this node to the page, and forwards the page
     toward the others (mesh_switch_carry).
   - Switching.  At every node a page is at, the destinations it carries on are split over the node's out-neighbours by
     a draw from the routing weights (mesh_switch_choose): one uniform u per page, each destination's lane the inverse of
     its weights' distribution at u (so destinations whose distributions agree share one SEND), masked to the
     neighbours off the page's path that reach the destination without coming back, over lanes whose frames hold the
     page's payload.  The cable to a neighbour is the one with the least bytes queued.  The page is pushed onto each
     group's lane's outbox (a ring in the region any port child pushes and the lane's port child posts), a use of it a
     group: the page is switched across the node's ports, whichever child holds it.
   - Landing.  A page landed sets the table entry of each destination here whose turn it is (mesh_switch_set: the
     index's landed count one more, the page it named before dropped; an index no client owns is set nothing) and is
     forwarded unchanged for the others it carries (mesh_switch_landed).  Nothing waits and nothing is reported: a
     destination no allowed neighbour reaches, a header that is not one, a page a lost link took, a landing whose
     receive went again into its own page are each counted (`unreachable`, `malformed`, `lost`: struct
     mesh_switch_counters), never retried.
   - Uses.  A page's uses (mesh.h) count its table words, its writes pending in a ring, its SENDs pushed or posted and a
     landing being carried; its last use's drop gives it back to its class's free stack.
   - The weights.  A distribution, for each destination, over the node's out-neighbours, published double-buffered
     (seqlock) and read by every draw.  A refresh (mesh_switch_refresh) computes a target from the estimators and emits
     the residual between it and the current weights, applied at the rate eta; draws go on reading the last published
     weights while one is computed: a mild margin of staleness, never a wait.  The target is the policy's: by default
     Thompson sampling over paths (each hop's wait drawn from its line), or any struct mesh_switch_policy (the trained
     recurrent policy, rdma/route, plugs in there).
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
#define MESH_SWITCH_LANES (MESH_PORTS*2u*MESH_LINK_STREAMS)   /* a port's data lanes, MESH_LANE_FIRST on */
#define MESH_SWITCH_N 8u                                     /* nodes the view holds */
#define MESH_SWITCH_LINKS 64u                                /* directed node pairs the view holds */
#define MESH_SWITCH_RING 32768u                              /* a lane's outbox: every page of the pool once */
#define MESH_SWITCH_LINES 24u                                /* lines a header carries */
#define MESH_SWITCH_LOG 8192u                                /* records a process's log ring */
#define MESH_SWITCH_LOGS (MESH_PORTS+1u)                     /* counters a port child, then the parent's */
#define MESH_SWITCH_PARENT MESH_PORTS                        /* the parent's counters */
#define MESH_SWITCH_CONTEXTS 256u                            /* the policy's contexts at a node, at most */
#define MESH_SWITCH_REFRESH_NS 50000.0                       /* the refresh's period (the bridge's thread) */
#define MESH_SWITCH_DRAWS 32u                                /* Thompson sampling's draws a refresh */
#define MESH_SWITCH_ETA 0.5f                                 /* the residual's rate a refresh */
#define MESH_SWITCH_PAGE ((double)(MESH_CLASS_LONG*MESH_FRAME))   /* the refresh's page: a long page's frames */
_Static_assert(MESH_SWITCH_N<=64 && MESH_SWITCH_N<=MESH_NODES,"a node a bit");
_Static_assert(MESH_DESTS<=32,"a destination a bit");
/* a page is in a lane's outbox at most once (its SEND's completion drops the use that blocks its next write or landing),
   so an outbox of every pool page never fills */
_Static_assert(MESH_SWITCH_RING>=MESH_POOL_PAGES && !(MESH_SWITCH_RING&(MESH_SWITCH_RING-1)),"an outbox holds every page once");

/* ---- The header (a page's first MESH_HEAD bytes) ------------------------------------------------------------------- */
/* a destination as a header carries it: its node and index of that node's table; the node whose turn it is to carry it
   on (`next`: the node a hop sent it to, which alone switches it; a page goes out unchanged on every group's lane, each
   receiver reading only the destinations it is the next of; UINT32_MAX no one's); and the cost it carried so far (ns:
   queueing at each hop, the wire's time) */
struct mesh_hop { uint32_t node,index,next; float queued,transit; };
/* a line: a directed link's predicted wait (ns) and its spread, its rate (bytes a ns, every cable) and latency, as its
   sender (`from`) measured it at its refresh `stamp` */
struct mesh_switch_line { uint32_t from,to; uint64_t stamp; float wait,spread,rate,latency; };
/* the header, written by the bridge (never the client): MESH_HEAD_MAGIC; the node that took the write (`source`); the
   payload's bytes; the hops, destinations and lines; the node that sent this SEND (`from`: its receiver appends itself
   to `path`); `unused`, unused; the nodes visited, `hops` + 1 of them; the destinations; the sequence of the headers
   its first node made (`piece`: the draw's seed, the records' sampling and their join); and the write's destination
   nodes as bits over the source's view (`wanted`: the policy's context).  The lines follow the payload in the last
   frame's slack (mesh_switch_lines_at) */
struct mesh_head {
  uint32_t magic,source,bytes,hops,dests,lines,from,unused;
  uint32_t path[MESH_SWITCH_N];
  struct mesh_hop dest[MESH_DESTS];
  uint64_t piece;
  uint32_t wanted;
};
_Static_assert(sizeof(struct mesh_head)<=MESH_HEAD && MESH_HEAD%64==0,"a header within its MESH_HEAD bytes");
_Static_assert(sizeof(struct mesh_head)>MESH_HEAD-64,"MESH_HEAD no longer than the header needs");
static inline struct mesh_head *mesh_head_of(struct hdr *m,uint32_t pd){ return (struct mesh_head *)mesh_pd_head(m,pd); }

/* ---- The region's switch (struct hdr switch_off; MESH_SWITCH_BYTES, mesh.h) ------------------------------------- */
/* a link of the view: a directed node pair, its cables, rate (all of them, bytes a ns), latency (ns), and its line's
   wait and spread (two floats in one word) with the line's stamp */
struct mesh_switch_link { uint32_t from,to,cables,ends; double rate,latency; _Atomic uint64_t line,stamp; };   /* ends: the view indices, from | to << 8 */
struct mesh_switch_view {
  _Atomic uint64_t serial;          /* odd while the parent writes it */
  uint32_t self,nodes,links,ranks; /* ranks: the view nodes that run collectives' ranks, bits over view indices (0: all) */
  uint32_t node[MESH_SWITCH_N];     /* node numbers, ascending: a node's index is its place here */
  struct mesh_switch_link link[MESH_SWITCH_LINKS];
  /* reach[avoid][from]: the view nodes `from` reaches (itself among them) entering none of `avoid` (bits over view
     indices), made with the links (mesh_switch_view_set): a draw's allowed neighbours read here, not searched */
  uint8_t reach[1u<<MESH_SWITCH_N][MESH_SWITCH_N];
};
/* the weights: for the destination of view index d, a probability over the node's out-neighbours (neighbour[k], view
   indices), and the refresh's cost-to-go between view nodes (ns), published as the view is (seqlock) */
struct mesh_switch_weights {
  _Atomic uint64_t serial;
  uint32_t nodes,deg;
  uint32_t neighbour[MESH_SWITCH_N];
  float w[MESH_SWITCH_N][MESH_SWITCH_N];
  float togo[MESH_SWITCH_N][MESH_SWITCH_N];
  uint64_t refreshes;
  uint64_t toward[MESH_SWITCH_N];   /* neighbour k's lanes live at the refresh, bits over the switch's lanes (a draw reads
                                       these lanes alone, each its `live` again) */
};
/* the policy's split tables (rdma/route route.h): this node's contexts, each (its pattern's source, the
   pattern's destinations, the destinations still owed: view indices, the last two as bits) in mesh_switch_contexts'
   order, and each one's w [SPLIT_FS], published as the weights are (seqlock).  `active` set, a page
   whose context is here draws its split from split.h's table under its w (mesh_switch_choose); else the weights draw.
   `unit`: the features' time unit (mesh_switch_state_make), `point`: the policy refresh the tables are from */
struct mesh_switch_context { uint32_t source,wanted,owed,pattern; };
struct mesh_switch_splits {
  _Atomic uint64_t serial;
  uint32_t active,contexts,argmax,padding;
  uint64_t point,unit_view;          /* unit_view: the view serial `unit` was made from */
  float unit,spare;
  struct mesh_switch_context key[MESH_SWITCH_CONTEXTS];
  float w[MESH_SWITCH_CONTEXTS][SPLIT_FS];
};
/* a lane of this node (port p's data lane j: p 2 MESH_LINK_STREAMS + j): live, its peer node, the bytes queued on it
   (pushed and not yet completed), its outbox's cursors (`tail` the next position pushed, by any port child; `head` the next
   posted and `done` the next completed, by the lane's child), and its feedback: SENDs completed, their bytes, their
   service and wait (ns, summed), pages landed, the estimate its child keeps (rate in bytes a ns and wait, two floats in
   one word); and its frames (every SEND's and receive's on the lane, TN3205: the two ends alike; 0 while it never
   opened) */
struct mesh_switch_lane {
  _Alignas(64) _Atomic uint32_t live,peer;
  _Atomic uint64_t queued,tail;     /* every pushing child's (mesh_switch_push) */
  /* the lane's child's alone, on a line of its own: its outbox's head and done read every pass, never bounced by a push */
  _Alignas(64) _Atomic uint64_t head,done;
  _Atomic uint64_t sends,bytes,service,wait,landed,estimate;
  _Atomic uint64_t frames;
};
/* an outbox cell, of position `at`: free for it while its sequence is `at`; holding page `pd`, pushed at `pushed` (the
   pusher's clock as its take pass began: the page's time on the outbox, for the lane's estimators, counts from there)
   and posted at `posted` (its SEND's post, the lane child's clock), while it is at + 1, until the page's SEND completes
   or its lane is emptied (mesh_switch_lane_drop), so the outbox is also the lane's record of its SENDs posted and not
   completed; then free for at + MESH_SWITCH_RING.  `carried`: the nanoseconds from `pushed` to the push itself (the take,
   the header, the route choice), for observation only; the cell's size and offsets are as before it was recorded.
   `pd` MESH_ABSENT: a hole, no SEND, where a pusher died holding the cell (mesh_switch_left_settle) */
struct mesh_switch_cell { _Atomic uint64_t sequence; uint64_t pushed,posted; uint32_t pd,carried; };
_Static_assert(sizeof(struct mesh_switch_cell)==32,"mesh_switch_cell");
/* the trainer's records (rdma/route): a draw (split.h struct route_draw and its behaviour log-probability) or a page
   set at a destination (its latency and queueing), each with the node that made it, the page's piece and source, its
   first destination's node and index (a draw's) or the destination's, its bytes, the draw's context at its node
   (UINT32_MAX: none, the weights drew) and the policy refresh its split table is from, at the node's clock (ns) */
enum { MESH_LOG_DRAW=1, MESH_LOG_PIECE };
struct mesh_switch_record {
  uint32_t kind,node;
  uint64_t piece,at;
  struct route_draw draw;           /* MESH_LOG_DRAW */
  float logp,latency,queueing;
  uint32_t source,dest,index,bytes,context,point;
};
struct mesh_switch_log { _Alignas(64) _Atomic uint64_t head; _Alignas(64) _Atomic uint64_t tail,dropped; struct mesh_switch_record record[MESH_SWITCH_LOG]; };
/* a port child's hand, which its parent reads once the child is gone and the next bridge reads as it adopts the region
   (mesh_switch_left_settle): the ring position of the write it takes (`take`: (slot + 1) << 56 | the position,
   MESH_HAND_HELD once its compare-exchange won, 0 none), and the outbox position it pushes page `page` to (`push`:
   (lane + 1) << 56 | the position, MESH_HAND_HELD once won, 0 none) */
#define MESH_HAND_HELD (UINT64_C(1)<<55)
#define MESH_HAND_AT (MESH_HAND_HELD-1)
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
   - receives a lane's fill wanted that its class's free stack had no page for (`starved`), refreshes;
   - destinations no allowed neighbour reaches (`unreachable`), pages a lost lane, a gone child or a full outbox took
     and landings whose receive was posted again into their own page, their class's free stack empty (`lost`: once a
     page, not a destination), headers and ring cells that are not one and destination indices past
     the table (`malformed`), and destinations here whose index no client owned, set nothing (`unowned`) */
struct mesh_switch_counters {
  _Alignas(64) _Atomic uint64_t passes,pass_ns[2],part_ns[5];
  _Atomic uint64_t taken,landed,set,forwards,groups;
  _Atomic uint64_t posts,post_wait_ns,post_call_ns,sent,send_ns;
  _Atomic uint64_t starved,refreshes;
  _Atomic uint64_t unreachable,lost,malformed,unowned;
};
/* `writes`: the headers this node made (the writes it took), each one's piece; `weighted`: a draw from the weights
   (else the earliest arrival, the design's rule) */
struct mesh_switch {
  uint32_t magic,padding;
  _Atomic uint64_t stamp;
  _Atomic uint32_t logging,weighted;
  _Alignas(64) _Atomic uint64_t writes;
  struct mesh_switch_view view;
  struct mesh_switch_weights weights;
  struct mesh_switch_splits splits;
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
/* the bytes every SEND and receive on lane l covers: its frames' */
static inline uint64_t mesh_switch_lane_bytes(const struct mesh_switch *sw,uint32_t l){
  return atomic_load_explicit(&sw->lane[l].frames,memory_order_relaxed)*MESH_FRAME;
}

/* the switch made new at the bridge's start, every outbox empty: nothing of a bridge before survives its stop */
static inline void mesh_switch_init(struct mesh_switch *sw){
  memset(sw,0,sizeof *sw);
  for(uint32_t l=0;l<MESH_SWITCH_LANES;l++)for(uint32_t i=0;i<MESH_SWITCH_RING;i++)atomic_store_explicit(&sw->ring[l][i].sequence,i,memory_order_relaxed);
  sw->magic=MESH_SWITCH_MAGIC;
  atomic_thread_fence(memory_order_release);
}

/* ---- A lane's outbox: bounded, any port child pushes, the lane's child posts and completes ----------------------- */
/* page pd pushed onto lane l's outbox at the pusher's clock `now`, by any port child, naming its hand (each position in
   it before its compare-exchange, held once won, cleared once the cell is filled, so its parent settles a push it died
   in, mesh_switch_left_settle): 1, or 0 where the outbox is full (it holds every pool page once, so a push of a page
   not already on it never finds it full), the hand cleared */
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
    atomic_store_explicit(&hand->push,named|MESH_HAND_HELD,memory_order_relaxed);
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
/* lane l's outbox cell of position `at`, holding a page, posted at `posted` (its lane child's clock) */
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
/* lane l's outbox emptied, its child gone or its lane opened anew: each SEND posted and not completed and each page
   pushed and not posted given back, its bytes off the lane's queue, its use dropped and counted `lost` in set `ring`
   (the caller's counters); the cursors past them.  The cells from the first a pusher holds unfilled stay (its push
   ends, or its death is settled: mesh_switch_left_settle) */
static inline void mesh_switch_lane_drop(struct hdr *m,struct mesh_switch *sw,uint32_t l,uint32_t ring){
  struct mesh_switch_lane *x=sw->lane+l;
  const uint64_t wire=mesh_switch_lane_bytes(sw,l);
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

/* ---- The view and the weights: written by one process (the parent's refresh), read by all ----------------------- */
static inline uint32_t mesh_switch_index(const struct mesh_switch_view *v,uint32_t node){
  for(uint32_t i=0;i<v->nodes;i++)if(v->node[i]==node)return i;
  return UINT32_MAX;
}
/* the view made from the fabric as the records give it: `nodes` node numbers (any order) and `links` directed pairs
   (node numbers), each its cables, rate (bytes a ns, every cable) and latency (ns); a link's line kept where it is
   already held */
struct mesh_switch_fabric_link { uint32_t from,to,cables; double rate,latency; };
static inline void mesh_switch_view_set(struct mesh_switch *sw,uint32_t self,const uint32_t *nodes,uint32_t n,
                                        const struct mesh_switch_fabric_link *links,uint32_t count){
  struct mesh_switch_view *v=&sw->view;
  uint32_t sorted[MESH_SWITCH_N],k=0;
  for(uint32_t i=0;i<n && k<MESH_SWITCH_N;i++){
    uint32_t j=k;while(j && sorted[j-1]>nodes[i]){sorted[j]=sorted[j-1];j--;}
    sorted[j]=nodes[i];k++;
  }
  uint64_t lines[MESH_SWITCH_LINKS],stamps[MESH_SWITCH_LINKS];uint32_t from[MESH_SWITCH_LINKS],to[MESH_SWITCH_LINKS],held=v->links;
  for(uint32_t l=0;l<held;l++){lines[l]=atomic_load(&v->link[l].line);stamps[l]=atomic_load(&v->link[l].stamp);from[l]=v->link[l].from;to[l]=v->link[l].to;}
  atomic_fetch_add_explicit(&v->serial,1,memory_order_acq_rel);
  v->self=self;v->nodes=k;memcpy(v->node,sorted,k*sizeof *sorted);
  v->links=0;
  for(uint32_t l=0;l<count && v->links<MESH_SWITCH_LINKS;l++){
    struct mesh_switch_link *x=v->link+v->links++;
    x->from=links[l].from;x->to=links[l].to;x->cables=links[l].cables?links[l].cables:1;x->rate=links[l].rate;x->latency=links[l].latency;
    x->ends=(mesh_switch_index(v,x->from)&255)|(mesh_switch_index(v,x->to)&255)<<8;
    uint64_t line=0,stamp=0;
    for(uint32_t h=0;h<held;h++)if(from[h]==x->from && to[h]==x->to){line=lines[h];stamp=stamps[h];}
    atomic_store_explicit(&x->line,line,memory_order_relaxed);atomic_store_explicit(&x->stamp,stamp,memory_order_relaxed);
  }
  /* each node's reach avoiding each set: a breadth-first search over the links' ends */
  uint32_t next[MESH_SWITCH_N]={0};
  for(uint32_t l=0;l<v->links;l++){const uint32_t a=v->link[l].ends&255,b=v->link[l].ends>>8&255;if(a<k && b<k)next[a]|=1u<<b;}
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
  atomic_fetch_add_explicit(&v->serial,1,memory_order_release);
}
static inline struct mesh_switch_link *mesh_switch_link_of(struct mesh_switch_view *v,uint32_t from,uint32_t to){
  for(uint32_t l=0;l<v->links;l++)if(v->link[l].from==from && v->link[l].to==to)return v->link+l;
  return NULL;
}
/* a consistent copy of the view or the weights (seqlock: retried while one is written) */
static inline void mesh_switch_view_read(const struct mesh_switch *sw,struct mesh_switch_view *out){
  for(;;){
    const uint64_t a=atomic_load_explicit(&sw->view.serial,memory_order_acquire);
    if(a&1)continue;
    memcpy((char *)out+sizeof out->serial,(const char *)&sw->view+sizeof out->serial,sizeof *out-sizeof out->serial);
    atomic_thread_fence(memory_order_acquire);
    if(atomic_load_explicit(&sw->view.serial,memory_order_relaxed)==a){out->serial=a;return;}
  }
}
static inline void mesh_switch_weights_read(const struct mesh_switch *sw,struct mesh_switch_weights *out){
  for(;;){
    const uint64_t a=atomic_load_explicit(&sw->weights.serial,memory_order_acquire);
    if(a&1)continue;
    memcpy((char *)out+sizeof out->serial,(const char *)&sw->weights+sizeof out->serial,sizeof *out-sizeof out->serial);
    atomic_thread_fence(memory_order_acquire);
    if(atomic_load_explicit(&sw->weights.serial,memory_order_relaxed)==a){out->serial=a;return;}
  }
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
/* where a header's lines lie: after its payload (8-aligned), within the SEND's whole frames, so they cost no wire; and
   how many fit in `wire` bytes */
static inline struct mesh_switch_line *mesh_switch_lines_at(struct mesh_head *head){
  return (struct mesh_switch_line *)((char *)head+MESH_HEAD+((uint64_t)head->bytes+7)/8*8);
}
static inline uint32_t mesh_switch_lines_fit(const struct mesh_head *head,uint64_t wire){
  const uint64_t at=MESH_HEAD+((uint64_t)head->bytes+7)/8*8,room=wire>at?(wire-at)/sizeof(struct mesh_switch_line):0;
  return room<MESH_SWITCH_LINES?(uint32_t)room:MESH_SWITCH_LINES;
}
/* the lines a header carries taken into the view where newer than its own (a node's lines about its own links are the
   ones it measured: a link's line is only ever made by its sender) */
static inline void mesh_switch_lines_take(struct mesh_switch *sw,struct mesh_head *head,uint64_t wire){
  struct mesh_switch_view *v=&sw->view;
  const uint32_t lines=head->lines<mesh_switch_lines_fit(head,wire)?head->lines:mesh_switch_lines_fit(head,wire);
  const struct mesh_switch_line *line=mesh_switch_lines_at(head);
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
/* the lines a header sends: this node's own links first (their wait the least bytes queued over the rate of the
   neighbour's lanes), then the view's others, newest first, as many as fit */
static inline uint32_t mesh_switch_lines_give(struct mesh_switch *sw,struct mesh_switch_line *line){
  struct mesh_switch_view *v=&sw->view;
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
/* the wait now on this node's lanes toward `peer` (the least queued bytes over the lane's rate), its lane, and the
   neighbour's lanes' rate summed (bytes a ns, 0: no live lane) */
_Static_assert(MESH_SWITCH_LANES<=64,"a lane a bit");
/* the lanes live toward `peer`, bits over the switch's lanes */
static inline uint64_t mesh_switch_lanes_mask(struct mesh_switch *sw,uint32_t peer){
  uint64_t mask=0;
  for(uint32_t l=0;l<MESH_SWITCH_LANES;l++)
    if(atomic_load_explicit(&sw->lane[l].live,memory_order_acquire) && atomic_load_explicit(&sw->lane[l].peer,memory_order_relaxed)==peer)mask|=UINT64_C(1)<<l;
  return mask;
}
/* mesh_switch_lanes_toward over the lanes of `mask` alone (the refresh's: struct mesh_switch_weights `toward`), each
   read live again; every lane where `mask` is 0 */
static inline double mesh_switch_lanes_among_for(struct mesh_switch *sw,uint64_t mask,uint32_t peer,double fallback_rate,uint32_t *best,double *rate,double extra,uint64_t need);
static inline double mesh_switch_lanes_among(struct mesh_switch *sw,uint64_t mask,uint32_t peer,double fallback_rate,uint32_t *best,double *rate,double extra){
  return mesh_switch_lanes_among_for(sw,mask,peer,fallback_rate,best,rate,extra,0);
}
/* among the lanes whose SENDs carry `need` bytes (0: any), the one of the least wait with `extra` bytes more, or with
   its own SEND's bytes where `extra` is negative */
/* the calling thread's own port (a port child's progress thread, mesh-link.c port_progress; UINT32_MAX elsewhere): of
   lanes whose waits are equal, one of its own port's is posted by this thread in this pass, any other's waits for its
   own child's pass (the handoff through the outbox) */
static __thread uint32_t mesh_switch_home=UINT32_MAX;
static inline double mesh_switch_lanes_among_for(struct mesh_switch *sw,uint64_t mask,uint32_t peer,double fallback_rate,uint32_t *best,double *rate,double extra,uint64_t need){
  double least=INFINITY;*best=UINT32_MAX;*rate=0;
  const uint64_t every=~UINT64_C(0)>>(64-MESH_SWITCH_LANES);
  if(!mask)mask=every;
  const uint64_t given=mask;
  /* a cable's lanes share it: each cable's (port's) bytes queued over all its live lanes toward the peer, and its rate
     (the view's per cable, else its lanes' estimates summed) */
  enum { CABLES=MESH_SWITCH_LANES/(2*MESH_LINK_STREAMS) };
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
  uint32_t counted=0;
  for(uint64_t k=live;k;k&=k-1){
    const uint32_t l=(uint32_t)__builtin_ctzll(k),c=l/(2*MESH_LINK_STREAMS);
    const double each=fallback_rate>0?fallback_rate:estimate[c]>0?estimate[c]:1;
    if(!(counted>>c&1)){*rate+=each;counted|=1u<<c;}
    const uint64_t carries=mesh_switch_lane_bytes(sw,l);
    if(need>carries)continue;
    const double wait=(queued[c]+(extra<0?(double)carries:extra))/each;
    const int home=c==mesh_switch_home,best_home=*best!=UINT32_MAX && *best/(2*MESH_LINK_STREAMS)==mesh_switch_home;
    if(wait<least || (wait==least && *best!=UINT32_MAX && (home>best_home || (home==best_home &&
                      atomic_load_explicit(&sw->lane[l].queued,memory_order_relaxed)<atomic_load_explicit(&sw->lane[*best].queued,memory_order_relaxed))))){least=wait;*best=l;}
  }
  /* none of the mask's lanes live: every lane looked at (one come up since the refresh) */
  if(*best==UINT32_MAX && given!=every)return mesh_switch_lanes_among_for(sw,every,peer,fallback_rate,best,rate,extra,need);
  return least;
}
static inline double mesh_switch_lanes_toward(struct mesh_switch *sw,uint32_t peer,double fallback_rate,uint32_t *best,double *rate){
  return mesh_switch_lanes_among(sw,~UINT64_C(0)>>(64-MESH_SWITCH_LANES),peer,fallback_rate,best,rate,0);
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
/* the w of context (source, wanted, owed) from the published tables, read consistently: 1 found (w filled, *context its
   index, *point the tables' refresh), 0 none or the tables not active */
static inline int mesh_switch_split_w(const struct mesh_switch *sw,uint32_t source,uint32_t wanted,uint32_t owed,float *w,uint32_t *context,uint64_t *point,
                                      uint32_t *argmax,float *unit){
  for(;;){
    const uint64_t a=atomic_load_explicit(&sw->splits.serial,memory_order_acquire);
    if(a&1)continue;
    int found=0;
    if(sw->splits.active)
      for(uint32_t c=0;c<sw->splits.contexts && c<MESH_SWITCH_CONTEXTS;c++){
        const struct mesh_switch_context *k=sw->splits.key+c;
        if(k->source==source && k->wanted==wanted && k->owed==owed){memcpy(w,sw->splits.w[c],sizeof sw->splits.w[c]);*context=c;found=1;break;}
      }
    *point=sw->splits.point;*argmax=sw->splits.argmax;*unit=sw->splits.unit;
    atomic_thread_fence(memory_order_acquire);
    if(atomic_load_explicit(&sw->splits.serial,memory_order_relaxed)==a)return found;
  }
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
   neighbour's cable with the least bytes queued, among lanes whose frames are at least `frames`, the class of the
   page's payload) and its destinations (bits over `away`).
   The draw's record (split.h struct route_draw, as rdma/route reads it): for each carried destination d and neighbour
   k, the arrival through k (its lanes' wait now, the page's time on them, the link's latency, then the refresh's
   cost-to-go from k to d), whether k reaches d without coming back here (valid), and whether it is allowed from the
   page's path (off it, reaching d avoiding it: mask).
   The draw, at u = the page's seed (its source, its piece, its first carried destination's index):
   - where the policy's split tables are active and hold the page's context (its source, its wanted destinations, the
     ones carried here: mesh_switch_split_w), the split is split.h's table under the context's w at u: exactly the
     trainer's action, *logp its log-probability;
   - else each destination i takes the neighbour k at which its masked, renormalised weights' distribution first passes
     u; where its weights give its allowed neighbours nothing, the allowed one of the least cost-to-go; *logp the
     length of u's interval that makes the split.
   A destination no allowed neighbour reaches (or not in the view) is in no group: its bit set in *unreachable.
   `record` (NULL: none) receives the record, *context the context drawn from (UINT32_MAX: the weights), *point the
   tables' refresh.  Returns the groups */
static inline uint32_t mesh_switch_choose(struct mesh_switch *sw,const struct mesh_switch_view *v,const struct mesh_switch_weights *w,
                                          const struct mesh_head *head,const uint32_t *away,uint32_t naway,uint32_t frames,
                                          uint32_t *lane,uint64_t *group,uint32_t *unreachable,
                                          struct route_draw *record,float *logp,uint32_t *context,uint64_t *point){
  const uint32_t self=mesh_switch_index(v,v->self);
  *unreachable=0;
  uint64_t visited=0;
  for(uint32_t h=0;h<=head->hops && h<MESH_SWITCH_N;h++){const uint32_t i=mesh_switch_index(v,head->path[h]);if(i<64)visited|=UINT64_C(1)<<i;}
  if(self<64)visited|=UINT64_C(1)<<self;
  const double u=mesh_switch_uniform(((uint64_t)head->source<<32)^head->piece*0x9e3779b97f4a7c15ULL^(naway?head->dest[away[0]].index:0));
  uint32_t deg=w->deg<MESH_SWITCH_N?w->deg:MESH_SWITCH_N,choice[MESH_DESTS],node[MESH_DESTS];
  uint32_t best_lane[MESH_SWITCH_N];double rate[MESH_SWITCH_N],wait[MESH_SWITCH_N],latency[MESH_SWITCH_N];
  for(uint32_t k=0;k<deg;k++){
    const uint32_t nb=w->neighbour[k];
    const struct mesh_switch_link *x=nb<v->nodes?mesh_switch_link_of((struct mesh_switch_view *)v,v->self,v->node[nb]):NULL;
    /* the lane's wait with this page's own frames on it: its time to clear the lane (design: (backlog + bytes) / rate) */
    wait[k]=nb<v->nodes?mesh_switch_lanes_among_for(sw,w->toward[k],v->node[nb],x?x->rate/x->cables:0,best_lane+k,rate+k,-1,(uint64_t)frames*MESH_FRAME):INFINITY;
    latency[k]=x?x->latency:0;
  }
  /* the policy's context: the page's source and wanted destinations, the carried ones (view indices) */
  uint32_t owed=0;
  for(uint32_t j=0;j<naway;j++){node[j]=mesh_switch_index(v,head->dest[away[j]].node);if(node[j]<32)owed|=1u<<node[j];}
  float cw[SPLIT_FS],unit=1;uint32_t ctx=UINT32_MAX,argmax=0;uint64_t at_point=0;
  const int split=v->nodes<=SPLIT_N && deg<=SPLIT_DEG &&
                  mesh_switch_split_w(sw,mesh_switch_index(v,head->source),head->wanted,owed,cw,&ctx,&at_point,&argmax,&unit);
  if(!split)ctx=UINT32_MAX;
  if(context)*context=ctx;
  if(point)*point=at_point;
  struct route_draw local;
  if(!record && split)record=&local;
  double lo=0,hi=1;
  const int logged=record && v->nodes<=SPLIT_N && deg<=SPLIT_DEG;
  if(logged){memset(record,0,sizeof *record);record->deg=deg;record->unit=unit>0?unit:1;}
  uint32_t ntok=1;
  for(uint32_t j=0;j<naway;j++){
    const uint32_t d=node[j];
    choice[j]=UINT32_MAX;
    if(d>=v->nodes){*unreachable|=1u<<j;continue;}
    double total=0,cost=INFINITY;uint32_t cheapest=UINT32_MAX,allowed=0;
    uint32_t ok[MESH_SWITCH_N];
    for(uint32_t k=0;k<deg;k++){
      const uint32_t nb=w->neighbour[k];
      ok[k]=best_lane[k]!=UINT32_MAX && !(visited>>nb&1) && (nb==d || mesh_switch_reaches(v,nb,visited,d));
      if(logged){
        const int valid=nb==d || mesh_switch_reaches(v,nb,UINT64_C(1)<<self,d);
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
    /* split.h's table under the context's w (every destination of S has an allowed neighbour, so the table holds a
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
   The weights' target from what the node knows now: each hop's cost its wait (this node's own lanes' least queue over
   their rate, a remote link's line) plus a page's time on one cable plus its latency.  A policy (`policy` NULL: the
   default) writes target[d][k], a distribution over the neighbours for each destination, from that state (the view,
   the cost-to-go, each neighbour's wait now); the default is Thompson sampling over paths: `draws` draws of every
   hop's wait (its line's mean plus its spread times a normal draw), each destination's share of a neighbour the draws
   in which the earliest path to it leaves through that neighbour. */
struct mesh_switch_state {
  const struct mesh_switch_view *view;
  uint32_t self,deg,neighbour[MESH_SWITCH_N];
  double hop[MESH_SWITCH_LINKS],spread[MESH_SWITCH_LINKS];      /* each view link's cost (ns) and its spread */
  double first[MESH_SWITCH_N];                                   /* each neighbour's hop from here, now */
  float togo[MESH_SWITCH_N][MESH_SWITCH_N];
  uint64_t stamp;
  double page,unit;                                              /* unit: the features' time (mesh_switch_unit) */
};
/* the policy's input at one of its refreshes, from this node's state (mesh_switch_tokens_make: route.h's features, made
   here): a token for each of this node's own links (lane[i], of view link link_of[i]) and for each of its contexts
   (context[c], key[c] as the split tables hold it) */
struct mesh_switch_tokens {
  uint32_t lanes,contexts;
  uint64_t point,at;
  float unit,padding;
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
static inline void mesh_switch_state_make(struct mesh_switch *sw,const struct mesh_switch_view *v,double page,struct mesh_switch_state *s){
  memset(s,0,sizeof *s);
  s->view=v;s->page=page;s->self=mesh_switch_index(v,v->self);s->stamp=atomic_load_explicit(&sw->stamp,memory_order_relaxed);
  for(uint32_t l=0;l<v->links;l++){
    const struct mesh_switch_link *x=v->link+l;
    float wait,spread;mesh_switch_unpack(atomic_load_explicit(&x->line,memory_order_relaxed),&wait,&spread);
    if(x->from==v->self){uint32_t best;double rate;const double now=mesh_switch_lanes_toward(sw,x->to,x->rate/x->cables,&best,&rate);
                         wait=best==UINT32_MAX?INFINITY:(float)now;spread=0;}
    s->hop[l]=wait+page/(x->rate/x->cables)+x->latency;s->spread[l]=spread;
    if(x->from==v->self && isfinite(wait) && s->deg<MESH_SWITCH_N){const uint32_t b=mesh_switch_index(v,x->to);s->neighbour[s->deg]=b;s->first[s->deg]=s->hop[l];s->deg++;}
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
static inline double mesh_switch_unit(const struct mesh_switch_view *v,double page);
/* one refresh: the state, the target (the policy's, else Thompson's), the residual (target less the weights now) applied
   at the rate eta (1: the target itself), the weights published, this node's stamp moved */
static inline void mesh_switch_refresh(struct mesh_switch *sw,const struct mesh_switch_policy *policy,uint32_t draws,float eta,uint64_t seed,
                                       double page){
  struct mesh_switch_view v;mesh_switch_view_read(sw,&v);
  struct mesh_switch_state s;mesh_switch_state_make(sw,&v,page,&s);
  float target[MESH_SWITCH_N][MESH_SWITCH_N];memset(target,0,sizeof target);
  if(!atomic_load_explicit(&sw->weighted,memory_order_relaxed) && !(policy && policy->target)){
    /* no draw reads the weights (the earliest arrival routes): they stand, and the refresh is the cost-to-go and the
       lines alone (Thompson's draws cost 10 us a refresh of the refresh thread's) */
    for(uint32_t d=0;d<MESH_SWITCH_N;d++)for(uint32_t k=0;k<MESH_SWITCH_N;k++)target[d][k]=sw->weights.w[d][k];
    eta=1;
  }else if(!policy || !policy->target || policy->target(policy->context,&s,target))mesh_switch_thompson(&s,draws,seed,target);
  struct mesh_switch_weights *w=&sw->weights;
  const int moved=w->deg!=s.deg || memcmp(w->neighbour,s.neighbour,sizeof s.neighbour);
  atomic_fetch_add_explicit(&w->serial,1,memory_order_acq_rel);
  w->nodes=v.nodes;w->deg=s.deg;memcpy(w->neighbour,s.neighbour,sizeof s.neighbour);
  for(uint32_t k=0;k<MESH_SWITCH_N;k++)w->toward[k]=k<s.deg && s.neighbour[k]<v.nodes?mesh_switch_lanes_mask(sw,v.node[s.neighbour[k]]):0;
  for(uint32_t d=0;d<v.nodes;d++){
    double sum=0;
    for(uint32_t k=0;k<s.deg;k++){
      const float now=moved?0:w->w[d][k],residual=target[d][k]-now;
      w->w[d][k]=moved?target[d][k]:now+eta*residual;sum+=w->w[d][k];
    }
    for(uint32_t k=0;k<s.deg && sum>0;k++)w->w[d][k]=(float)(w->w[d][k]/sum);
    for(uint32_t b=0;b<v.nodes;b++)w->togo[d][b]=s.togo[d][b];
  }
  w->refreshes++;
  atomic_fetch_add_explicit(&w->serial,1,memory_order_release);
  if(sw->splits.unit_view!=v.serial+1){   /* the features' unit, as the view changes */
    const float unit=(float)mesh_switch_unit(&v,ROUTE_PAGE);
    atomic_fetch_add_explicit(&sw->splits.serial,1,memory_order_acq_rel);
    sw->splits.unit=unit;sw->splits.unit_view=v.serial+1;
    atomic_fetch_add_explicit(&sw->splits.serial,1,memory_order_release);
  }
  /* this node's own links' lines, for the headers it sends: each one's wait now, its spread the wait's change */
  for(uint32_t l=0;l<v.links;l++){
    if(v.link[l].from!=v.self)continue;
    struct mesh_switch_link *x=mesh_switch_link_of(&sw->view,v.link[l].from,v.link[l].to);
    if(!x)continue;
    uint32_t best;double rate;double now=mesh_switch_lanes_toward(sw,x->to,x->rate/x->cables,&best,&rate);
    if(!isfinite(now))now=0;
    float was,spread;mesh_switch_unpack(atomic_load_explicit(&x->line,memory_order_relaxed),&was,&spread);
    spread=spread+((float)fabs(now-was)-spread)/4;
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
   piece), so every node records the same pages and a recorded page's draws at every hop join its landings */
static inline int mesh_switch_sampled(const struct mesh_switch *sw,const struct mesh_head *head){
  const uint32_t k=atomic_load_explicit(&sw->logging,memory_order_relaxed);
  return k && (k==1 || mesh_switch_mix(head->piece^(uint64_t)head->source<<40)%k==0);
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
   past the table is set nothing and counted (`malformed`); one no client owns (its owner byte 0: its consumer left and
   freed it) is set nothing and counted (`unowned`), so no page is left at an index nobody frees */
static inline int mesh_switch_set(struct hdr *m,struct mesh_switch *sw,uint32_t pd,const struct mesh_head *head,uint32_t i,uint32_t ring){
  const struct mesh_hop *d=head->dest+i;
  if(d->index>=m->entries){MESH_SWITCH_COUNT(sw,ring,malformed,1);return 0;}
  if(!atomic_load_explicit(&mesh_owners(m)[d->index],memory_order_acquire)){MESH_SWITCH_COUNT(sw,ring,unowned,1);return 0;}
  mesh_pd_use(m,pd);
  mesh_pd_drop(m,mesh_entry_put(m,d->index,pd,1));
  MESH_SWITCH_COUNT(sw,ring,set,1);
  return 1;
}
/* a header's destinations' nodes as bits over this node's view, read consistently (seqlock) */
static inline uint32_t mesh_switch_wanted(const struct mesh_switch *sw,const struct mesh_head *head){
  for(;;){
    const uint64_t a=atomic_load_explicit(&sw->view.serial,memory_order_acquire);
    if(a&1)continue;
    uint32_t bits=0;
    for(uint32_t i=0;i<head->dests && i<MESH_DESTS;i++){const uint32_t d=mesh_switch_index(&sw->view,head->dest[i].node);if(d<32)bits|=1u<<d;}
    atomic_thread_fence(memory_order_acquire);
    if(atomic_load_explicit(&sw->view.serial,memory_order_relaxed)==a)return bits;
  }
}
/* The forward: page pd's destinations that this node carries on (`away`: places in its header's dest[]) split
   (mesh_switch_choose) into groups, a lane each, one group a peer, and the page itself pushed onto every group's lane:
   one page and no copy, each group's destinations naming that lane's peer their `next` (a receiver reads only the
   destinations it is the next of).  Before any push the header's shared parts are written: each destination no group
   carries named no one's next (another hop's SEND carries it, it is set here, or nothing reaches it), the sender, and
   the lines (as many as the frames of the payload's class hold).  Each group adds a use of the page and its frames to
   its lane's queue; a push the outbox has no room for gives that use back, counted (`lost`).  A destination no allowed
   neighbour reaches is not carried, counted (`unreachable`).  `ring` the caller's log ring and counters (its port's),
   `now` its clock, `hand` its hand (a port child's) */
static inline void mesh_switch_forward(struct hdr *m,struct mesh_switch *sw,uint32_t pd,struct mesh_head *head,const uint32_t *away,uint32_t naway,
                                       uint64_t now,uint32_t ring,struct mesh_switch_hand *hand){
  const uint32_t self=m->node,frames=mesh_class_frames(head->bytes);
  const int logging=mesh_switch_sampled(sw,head);
  uint32_t groups=0,lane[MESH_DESTS],unreachable=0;uint64_t group[MESH_DESTS];
  struct route_draw draw;float logp=0;uint32_t context=UINT32_MAX;uint64_t point=0;
  if(logging)memset(&draw,0,sizeof draw);
  /* the draw over the view and the weights where they lie, again where either was written meanwhile (seqlock): no copy
     of either a page */
  for(;;){
    const uint64_t a=atomic_load_explicit(&sw->view.serial,memory_order_acquire),b=atomic_load_explicit(&sw->weights.serial,memory_order_acquire);
    if((a|b)&1)continue;
    groups=mesh_switch_choose(sw,&sw->view,&sw->weights,head,away,naway,frames,lane,group,&unreachable,logging?&draw:NULL,&logp,&context,&point);
    atomic_thread_fence(memory_order_acquire);
    if(atomic_load_explicit(&sw->view.serial,memory_order_relaxed)==a && atomic_load_explicit(&sw->weights.serial,memory_order_relaxed)==b)break;
  }
  if(logging && groups){
    struct mesh_switch_record r;memset(&r,0,sizeof r);
    r.kind=MESH_LOG_DRAW;r.node=self;r.piece=head->piece;r.at=now;r.draw=draw;r.logp=logp;r.source=head->source;
    r.dest=head->dest[away[0]].node;r.index=head->dest[away[0]].index;r.bytes=head->bytes;r.context=context;r.point=(uint32_t)point;
    mesh_switch_log_put(sw,ring,&r);
  }
  if(unreachable)MESH_SWITCH_COUNT(sw,ring,unreachable,__builtin_popcount(unreachable));
  /* one group a peer: a receiver owes every destination naming it, so two lanes toward one peer would each carry all of them */
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
  const uint32_t nlines=mesh_switch_lines_give(sw,lines),fit=mesh_switch_lines_fit(head,(uint64_t)frames*MESH_FRAME);
  head->lines=nlines<fit?nlines:fit;memcpy(mesh_switch_lines_at(head),lines,head->lines*sizeof *lines);
  /* each group's destinations its peer's, with the cost they carry on (the lane's queue over its rate, its frames' time
     on the wire), all before the first push makes the page another process's to post */
  for(uint32_t g=0;g<groups;g++){
    const uint32_t l=lane[g],peer=atomic_load_explicit(&sw->lane[l].peer,memory_order_relaxed);
    float rate,waited;mesh_switch_unpack(atomic_load_explicit(&sw->lane[l].estimate,memory_order_relaxed),&rate,&waited);
    const double each=rate>0?rate:1;
    const float queued=(float)((double)atomic_load_explicit(&sw->lane[l].queued,memory_order_relaxed)/each),
                transit=(float)((double)mesh_switch_lane_bytes(sw,l)/each);
    for(uint32_t j=0;j<naway;j++)if(group[g]>>j&1){struct mesh_hop *d=head->dest+away[j];d->next=peer;d->queued+=queued;d->transit+=transit;}
  }
  for(uint32_t g=0;g<groups;g++){
    const uint32_t l=lane[g];
    const uint64_t wire=mesh_switch_lane_bytes(sw,l);
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
   its piece, every destination's next this node, the destinations' nodes as view bits its `wanted`), each destination
   on this node set in the table (mesh_switch_set), the others forwarded (mesh_switch_forward).  A cell naming no page
   of the pool, no destination or more than MESH_DESTS, or more bytes than its page's payload is carried nowhere,
   counted (`malformed`) */
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
  head->piece=atomic_fetch_add_explicit(&sw->writes,1,memory_order_relaxed);
  for(uint32_t i=0;i<dests;i++)head->dest[i]=(struct mesh_hop){w->dest[i].node,w->dest[i].index,self,0,0};
  head->wanted=mesh_switch_wanted(sw,head);
  uint32_t away[MESH_DESTS],naway=0;
  for(uint32_t i=0;i<dests;i++){
    if(head->dest[i].node!=self)away[naway++]=i;
    else mesh_switch_set(m,sw,pd,head,i,ring);
  }
  if(naway)mesh_switch_forward(m,sw,pd,head,away,naway,now,ring,hand);
}
/* Writes taken from client slot `slot`'s ring and carried (mesh_switch_carry), at most `most`, by the port child that
   holds hand `hand`: each ring position named in the hand before its compare-exchange and held once won, the write
   carried from its cell, which stays full meanwhile (so a child gone mid-write leaves the cell for its parent, or the
   next bridge, to settle: mesh_switch_left_settle), then the cell
   given back and the hand cleared, then the write's ring use dropped.  Taken alongside the other children's hands
   (a position one compare-exchange's; the port children are a ring's only takers).  Returns the writes taken */
static inline uint32_t mesh_switch_take_ring(struct hdr *m,struct mesh_switch *sw,uint32_t slot,uint32_t most,uint64_t now,uint32_t ring,struct mesh_switch_hand *hand){
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
    atomic_store_explicit(&hand->take,named|MESH_HAND_HELD,memory_order_relaxed);
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
   this node appended to its path, its lines taken (as many as the lane's frames hold), each destination here whose
   next this node is set in the table (mesh_switch_set) and its record made where the page is sampled, the
   destinations elsewhere whose next this node is forwarded (mesh_switch_forward), and the landing's use dropped.
   `now` the child's clock at the landing */
static inline void mesh_switch_landed(struct hdr *m,struct mesh_switch *sw,uint32_t l,uint32_t pd,uint64_t now,uint32_t ring,struct mesh_switch_hand *hand){
  struct mesh_pd *p=mesh_pd_at(m,pd);
  atomic_store_explicit(&p->uses,1,memory_order_relaxed);
  atomic_store_explicit(&p->state,MESH_PD_LIVE,memory_order_relaxed);
  atomic_fetch_add_explicit(&sw->lane[l].landed,1,memory_order_relaxed);
  MESH_SWITCH_COUNT(sw,ring,landed,1);
  struct mesh_head *head=mesh_head_of(m,pd);
  const uint32_t self=m->node;
  if(head->magic!=MESH_HEAD_MAGIC || head->bytes>MESH_PAYLOAD(p->cls) || !head->dests || head->dests>MESH_DESTS || head->hops>=MESH_SWITCH_N-1){
    MESH_SWITCH_COUNT(sw,ring,malformed,1);
    mesh_pd_drop(m,pd);
    return;
  }
  atomic_store_explicit(&p->bytes,head->bytes,memory_order_relaxed);
  head->path[++head->hops]=self;
  mesh_switch_lines_take(sw,head,mesh_switch_lane_bytes(sw,l));
  const int logging=mesh_switch_sampled(sw,head);
  uint32_t away[MESH_DESTS],naway=0;
  for(uint32_t i=0;i<head->dests;i++){
    const struct mesh_hop *d=head->dest+i;
    if(d->next!=self)continue;
    if(d->node!=self){away[naway++]=i;continue;}
    if(!mesh_switch_set(m,sw,pd,head,i,ring) || !logging)continue;
    /* a page's presence at each destination here, recorded as it is set: the cost it carried in-band (each hop's
       queueing on that hop's own clock, the wire's time) */
    struct mesh_switch_record r;memset(&r,0,sizeof r);
    r.kind=MESH_LOG_PIECE;r.node=self;r.piece=head->piece;r.at=now;r.latency=d->queued+d->transit;r.queueing=d->queued;
    r.source=head->source;r.dest=self;r.index=d->index;r.bytes=head->bytes;r.context=UINT32_MAX;
    mesh_switch_log_put(sw,ring,&r);
  }
  if(naway)mesh_switch_forward(m,sw,pd,head,away,naway,now,ring,hand);
  mesh_pd_drop(m,pd);
}

/* ---- A port child gone, a region adopted: what was held, given back (the parent's: mesh-link.c switch_reclaim,
   region_adopt) ------------------------------------------------------------------------------------------------- */
/* the pages posted as receives on the switch's lanes [first, first + count) given back (their child gone: nothing lands
   in them again); how many.  A page's state is read (acquire) before its lane and again after, so a lane read from a
   page's earlier receive is never taken with a later one's state: only a page still a receive on a lane of the range is
   given back */
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
/* whether another child's hand names the position `named` (its MESH_HAND_HELD aside) in its take (`push` 0) or its
   push (`push` 1): its holder may hold it */
static inline int mesh_switch_hand_names(const struct mesh_switch *sw,uint64_t named,int push){
  int any=0;
  for(uint32_t p=0;p<MESH_PORTS;p++)
    any|=(atomic_load_explicit(push?&sw->hand[p].push:&sw->hand[p].take,memory_order_acquire)&~MESH_HAND_HELD)==(named&~MESH_HAND_HELD);
  return any;
}
/* A port child's hand as the child left it at its end (`h`), settled as far as it can be now, by the parent's event
   loop (`ring` its counters); what is settled is cleared, and 1 returned once nothing of it is left.  `held` 1: only
   what the child had won (MESH_HAND_HELD), so that a cell two dead children's hands name is settled first by its
   winner's.
   - Its take: where the cell is still full past the ring's take and the child had won it, or (its win not marked yet)
     no other hand names that position, the write was carried no further than it got: its ring use dropped, counted
     (`lost`), and its cell given back.  A cell given back, or one past the take that no one won: nothing.
   - Its push of `page`: where the child holds its cell unfilled (won, or no other hand names it), the cell made a
     hole; where it won no cell, nothing was pushed.  Either way the page's use for the push dropped, counted (`lost`),
     and its frames off the lane's queue.  A cell it filled: the outbox's.
   Where another hand names the position, it is looked at again later (the parent's next turn): the other child ends
   its own step, or its own end is settled */
static inline int mesh_switch_left_settle(struct hdr *m,struct mesh_switch *sw,struct mesh_switch_left *h,int held,uint32_t ring){
  if(h->take && (!held || h->take&MESH_HAND_HELD)){
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
          if(!(h->take&MESH_HAND_HELD) && other)later=1;
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
  if(h->push && (!held || h->push&MESH_HAND_HELD)){
    const uint32_t l=(uint32_t)(h->push>>56)-1u;
    const uint64_t at=h->push&MESH_HAND_AT;
    int later=0;
    if(l<MESH_SWITCH_LANES){
      struct mesh_switch_lane *x=sw->lane+l;
      struct mesh_switch_cell *c=sw->ring[l]+at%MESH_SWITCH_RING;
      const int claimed=atomic_load_explicit(&x->tail,memory_order_acquire)>at;
      const int other=mesh_switch_hand_names(sw,h->push,1);
      const int unfilled=claimed && atomic_load_explicit(&c->sequence,memory_order_acquire)==at;
      int hole=0;
      if(h->push&MESH_HAND_HELD)hole=unfilled;
      else if(unfilled && other)later=1;
      else hole=unfilled;
      if(hole){c->pd=MESH_ABSENT;atomic_store_explicit(&c->sequence,at+1,memory_order_release);}
      if(!later && (hole || !(h->push&MESH_HAND_HELD)) && mesh_pd_valid(m,h->page)){
        atomic_fetch_sub_explicit(&x->queued,mesh_switch_lane_bytes(sw,l),memory_order_relaxed);
        mesh_pd_drop(m,h->page);
        MESH_SWITCH_COUNT(sw,ring,lost,1);
      }
    }
    if(!later)h->push=0;
  }
  return !h->take && !h->push;
}

/* ---- The policy's refresh: its features from this node's state, its tables published ---------------------------
   The features' time unit: the slowest pair of ranks' shortest unloaded delay for one page (route.h's stream unit) */
static inline double mesh_switch_unit(const struct mesh_switch_view *v,double page){
  double hop[MESH_SWITCH_LINKS],at[MESH_SWITCH_N],unit=0;
  for(uint32_t l=0;l<v->links;l++)hop[l]=v->link[l].latency+page/(v->link[l].rate/v->link[l].cables);
  const uint32_t ranks=mesh_switch_ranks(v);
  for(uint32_t a=0;a<v->nodes;a++){
    if(!(ranks>>a&1))continue;
    mesh_switch_arrivals(v,a,0,0,hop,at);
    for(uint32_t b=0;b<v->nodes;b++)if(b!=a && ranks>>b&1 && isfinite(at[b]) && at[b]>unit)unit=at[b];
  }
  return unit>0?unit:1;
}
/* route.h's tokens from what this node knows, at its clock `now`, for policy refresh `point`:
   - the fabric is the view (its links route.h's lanes, view indices its nodes, the ranks its compute nodes), its page
     route.h's (ROUTE_PAGE);
   - each lane's wait: this node's own lanes' least queue over their rate, a remote link's its line's; duplex share 1
     (a bridge sees no other node's reverse traffic);
   - the cost-to-go the last refresh's (the weights'), the bytes in flight and posted this node's own lanes' queues;
   - the per-pattern features of what is left to send, waiting and received, and the live collectives, 0: the bridge
     holds writes, not collectives.
   Every token is route.h's route_lane_token / route_context_token over that state. */
static inline void mesh_switch_tokens_make(struct mesh_switch *sw,uint64_t now,uint64_t point,struct mesh_switch_tokens *out){
  struct mesh_switch_view v;mesh_switch_view_read(sw,&v);
  struct mesh_switch_weights w;mesh_switch_weights_read(sw,&w);
  const uint32_t self=mesh_switch_index(&v,v.self),ranks=mesh_switch_ranks(&v);
  static __thread struct route_stream s;static __thread struct route_state st;
  memset(&s,0,sizeof s);memset(&st,0,sizeof st);
  s.f.nodes=(int)(v.nodes<ROUTE_N?v.nodes:ROUTE_N);s.f.lanes=(int)(v.links<ROUTE_L?v.links:ROUTE_L);s.f.duplex=1;
  for(int l=0;l<s.f.lanes;l++){
    const struct mesh_switch_link *x=v.link+l;
    s.f.from[l]=(int)mesh_switch_index(&v,x->from);s.f.to[l]=(int)mesh_switch_index(&v,x->to);s.f.cables[l]=(int)x->cables;s.f.rate[l]=x->rate;s.f.latency[l]=x->latency;
  }
  for(uint32_t a=0;a<v.nodes && a<ROUTE_N;a++)if(ranks>>a&1)s.f.compute[s.f.ncompute++]=(int)a;
  s.unit=mesh_switch_unit(&v,ROUTE_PAGE);
  struct route_topology t;memset(&t,0,sizeof t);t.srcmask=ranks;
  /* the patterns, in mesh_switch_contexts' order */
  for(uint32_t a=0;a<v.nodes;a++){
    if(!(ranks>>a&1))continue;
    for(uint32_t pass=0;pass<=v.nodes && s.patterns<ROUTE_J;pass++){
      if(pass<v.nodes && (pass==a || !(ranks>>pass&1)))continue;
      s.source[s.patterns]=(int)a;s.dest[s.patterns]=pass<v.nodes?1u<<pass:ranks&~(1u<<a);s.patterns++;
    }
  }
  st.tau=(double)now;
  for(int l=0;l<s.f.lanes;l++){
    const struct mesh_switch_link *x=v.link+l;
    float wait,spread;mesh_switch_unpack(atomic_load_explicit(&sw->view.link[l].line,memory_order_relaxed),&wait,&spread);
    if(x->from==v.self){uint32_t best;double rate;const double now_=mesh_switch_lanes_toward(sw,x->to,x->rate/x->cables,&best,&rate);wait=best==UINT32_MAX?0:(float)now_;}
    st.wait[l]=wait;st.dup[l]=1;
  }
  for(uint32_t a=0;a<ROUTE_N;a++)for(uint32_t b=0;b<ROUTE_N;b++)st.togo[a][b]=a<v.nodes && b<v.nodes?(a==b?0:w.togo[a][b]):1e12;
  double queued=0;
  for(uint32_t l=0;l<MESH_SWITCH_LANES;l++)if(atomic_load_explicit(&sw->lane[l].live,memory_order_relaxed))queued+=(double)atomic_load_explicit(&sw->lane[l].queued,memory_order_relaxed);
  if(self<ROUTE_N)st.inflight[self]=queued;
  st.posted=queued;
  out->point=point;out->at=now;out->unit=(float)s.unit;out->lanes=0;
  for(int l=0;l<s.f.lanes;l++){
    if(v.link[l].from!=v.self)continue;
    out->link_of[out->lanes]=(uint32_t)l;route_lane_token(&s,&st,&t,l,out->lane[out->lanes]);out->lanes++;
  }
  out->contexts=mesh_switch_contexts(&v,out->key);
  for(uint32_t c=0;c<out->contexts;c++){
    const struct mesh_switch_context *k=out->key+c;
    route_context_token(&s,&st,&t,(int)k->pattern,(int)self,k->owed,(int)point,out->context[c]);
  }
}
/* the split tables published: `n` contexts (keys, each its w), from policy refresh `point`; active from now on */
static inline void mesh_switch_splits_publish(struct mesh_switch *sw,const struct mesh_switch_context *key,uint32_t n,const float (*w)[SPLIT_FS],
                                              uint64_t point,float unit,int argmax){
  struct mesh_switch_splits *x=&sw->splits;
  if(n>MESH_SWITCH_CONTEXTS)n=MESH_SWITCH_CONTEXTS;
  atomic_fetch_add_explicit(&x->serial,1,memory_order_acq_rel);
  x->contexts=n;memcpy(x->key,key,n*sizeof *key);memcpy(x->w,w,n*sizeof *w);x->point=point;x->unit=unit;x->argmax=argmax?1:0;x->active=1;
  atomic_fetch_add_explicit(&x->serial,1,memory_order_release);
}
/* one refresh of the trained policy: its tokens from this node's state, each context's w from the policy, published */
static inline int mesh_switch_policy_refresh(struct mesh_switch *sw,const struct mesh_switch_policy *policy,uint64_t now,uint64_t point,int argmax,
                                             struct mesh_switch_tokens *scratch){
  if(!policy || !policy->splits)return -1;
  mesh_switch_tokens_make(sw,now,point,scratch);
  static __thread float w[MESH_SWITCH_CONTEXTS][SPLIT_FS];
  if(policy->splits(policy->context,scratch,w))return -1;
  mesh_switch_splits_publish(sw,scratch->key,scratch->contexts,(const float (*)[SPLIT_FS])w,point,scratch->unit,argmax);
  return 0;
}

#endif
