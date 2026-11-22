/* Switched multipath DMA at run time (design/switched-multipath-dma.md, work items 2-4): what every node's bridge does
   with a piece, as pure functions of its region.  The bridge's port children (mesh-link.c switch_pass), the bridge's
   parent (its refresh thread) and the N-node simulator (tests/sim-switched.c) are its callers; none of it reads a clock,
   does I/O or posts to hardware: the caller passes the time and posts what these functions hand it.

   What the program declares and what is decided here.  A client declares a program toward a node (mesh.h struct
   mesh_declaration, unchanged): tables of transfers, spans, gates and done words.  That is WHAT moves.  HOW each piece
   moves is decided here, per piece, while it moves:
   - Staging.  A send piece whose step is issued and whose gate holds is claimed by one compare-exchange on its
     program's cursor (mesh_switch_stage) and copied with its name into a landing slot of the node's pool: the slot is
     what crosses.  Its send-side done word rises as it is staged (the client's bytes are free once copied).
   - The name.  A slot is one SEND: its first MESH_SWITCH_HEAD bytes the piece's name (its source node, its program's
     key, execution, and for every destination still to be served through the node it lands on that destination's
     transfer and span, the nodes it has visited, the lines of lane state it carries), then the page frame.  One SEND
     into one preposted receive, of one fixed frame count (TN3205: sender and receiver post the same frame count), with
     no second crossing and no reply.
   - Switching.  At every node it is at, the piece's destinations are split over the node's out-neighbours by a draw
     from the routing weights (mesh_switch_choose): one uniform u per piece, each destination's lane the inverse of its
     weights' distribution at u (so destinations whose distributions agree share one SEND), masked to the neighbours
     off the piece's path that reach the destination without coming back.  The cable to a neighbour is the one with the
     least bytes queued.  Each group's slot is pushed onto its lane's outbox (a ring in the region any process pushes
     and the lane's port child pops): the piece is switched across the node's ports, whichever process holds it.
   - Relays forward by name.  A node a piece lands on delivers it where it is a destination and sends it on for the
     rest, the first group in place, every further group in a copy of the slot.  No client takes part.
   - Delivery.  No receive is tied to a page: a piece lands in the next free slot posted on its queue pair, and its
     name resolves it to the receiving program's transfer and span (the receiving client declared the same program, its
     directions swapped).  It is copied there once the receiving client issued that step and its gate holds
     (mesh_switch_deliver); until then it waits in its slot.  The receive span is the program's contiguity claim, the
     copy its defragmenter.  Every (execution, transfer) keeps a bitmap of its units, so each unit counts once whatever
     path brought it; done words rise in each word's own order (a word several transfers count on rises as their
     prefix), executions completing in any order within a window of MESH_SWITCH_WINDOW.
   - The weights.  A distribution, for each destination, over the node's out-neighbours, published double-buffered
     (seqlock) and read by every draw.  A refresh (mesh_switch_refresh) computes a target from the estimators and emits
     the residual between it and the current weights, applied at the rate eta; draws go on reading the last published
     weights while one is computed: a mild margin of staleness, never a wait.  The target is the policy's: by default
     Thompson sampling over paths (each hop's wait drawn from its line), or any struct mesh_switch_policy (the trained
     recurrent policy, rdma/tests/route, plugs in there).
   - The estimators.  Diachronic and incremental: each lane's rate and wait from its SENDs' completions (mesh_switch_sent),
     each remote link's wait from the lines names carry (mesh_switch_lines_take), each node's own lines stamped by its
     refreshes.  Nothing waits for any of it.

   Every hop adds the node to the piece's visited set and a lane is taken only toward a neighbour off it that reaches
   the destination avoiding it, so a path is simple and ends; each destination is in exactly one group at every hop.

   The trainer's records.  Every draw can be logged, off the hot path, in rdma/tests/route's record format (split.h
   struct route_draw: each (destination, out-lane)'s arrival estimate and validity, the allowed mask, the sampled split
   as its token, and its behaviour log-probability, exact under the coupled draw), every delivered piece its latency
   and queueing, every execution its local completion: a single-producer ring per process in the region
   (mesh_switch_log_put drops and counts where it is full; it never waits), drained by mesh-switch-log. */
#ifndef MESH_SWITCHED_H
#define MESH_SWITCHED_H
#include "mesh.h"
#include "tests/route/route.h"
#include <errno.h>
#include <math.h>
#include <string.h>

#define MESH_SWITCH_MAGIC 0x53574348u
#define MESH_SWITCH_LANES (MESH_PORTS*2u*MESH_LINK_STREAMS)   /* a port's data lanes, MESH_LANE_FIRST on */
#define MESH_SWITCH_N 8u                                     /* nodes the view holds */
#define MESH_SWITCH_LINKS 64u                                /* directed node pairs the view holds */
#define MESH_SWITCH_HEAD 1088u                               /* a slot's name, the piece's bytes after it */
#define MESH_SWITCH_UNITS 2u                                 /* a transfer's consecutive units a slot carries, at most */
#define MESH_SWITCH_SHORT 2u                                 /* a short lane's frames (a link's first data lane, where it has two) */
#define MESH_SWITCH_WINDOW 8u                                /* executions of a transfer in progress at once */
#define MESH_SWITCH_RING 4096u                               /* a lane's outbox */
#define MESH_SWITCH_SLOTS 4096u                              /* the pool's slots, at most */
#define MESH_SWITCH_LINES 24u                                /* lines a name carries */
#define MESH_SWITCH_LOG 2048u                                /* records a process's log ring */
#define MESH_SWITCH_LOGS (MESH_PORTS+2u)                     /* a ring a port child, the parent's, the simulator's */
#define MESH_SWITCH_CONTEXTS 256u
#define MESH_SWITCH_KIN 64u                                  /* programs whose kin (same key, other nodes) a thread keeps */
#define MESH_SWITCH_SIBLING_USES 256u                        /* a program's kin looked for again after this many stagings */                            /* the policy's contexts at a node, at most */
#define MESH_SWITCH_REFRESH_NS 50000.0                       /* the refresh's period (the bridge's thread, the simulator) */
#define MESH_SWITCH_DRAWS 32u                                /* Thompson sampling's draws a refresh */
#define MESH_SWITCH_ETA 0.5f                                 /* the residual's rate a refresh */
_Static_assert(MESH_SWITCH_N<=64 && MESH_SWITCH_N<=MESH_NODES,"a node a bit");

/* ---- The name ------------------------------------------------------------------------------------------------- */
struct mesh_switch_dest { uint32_t node,transfer,span,units; };   /* units: the spans from `span` the piece carries (0: one) */
/* a line: a directed link's predicted wait (ns) and its spread, its rate (bytes a ns, every cable) and latency, as its
   sender (`from`) measured it at its refresh `stamp` */
struct mesh_switch_line { uint32_t from,to; uint64_t stamp; float wait,spread,rate,latency; };
struct mesh_switch_name {
  uint32_t magic,source;            /* the producing node */
  uint64_t key,execution;           /* the program's key (alike at both ends) and its execution */
  uint32_t bytes,hops,dests,lines;
  uint32_t from,kind;               /* the node that sent this SEND; MESH_PIECE_DATA, or MESH_PIECE_CREDIT (mesh_switch_credit) */
  uint32_t path[MESH_SWITCH_N];     /* the nodes visited, in order: `hops` + 1 of them */
  struct mesh_switch_dest dest[MESH_SWITCH_N];
  float queued,transit,floor,cost;  /* ns: queueing so far, wire time so far, the unloaded delay, unused */
  uint64_t born;                    /* the source's clock when it was staged (ns) */
  uint32_t wanted[MESH_SWITCH_N],nwanted;   /* the piece's destinations as staged: its pattern's (with `source`) */
  struct mesh_switch_line line[MESH_SWITCH_LINES];
};
_Static_assert(sizeof(struct mesh_switch_name)<=MESH_SWITCH_HEAD && MESH_SWITCH_HEAD%64==0,"a name within its head");

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
/* the policy's split tables (rdma/tests/route route.h): this node's contexts, each (its pattern's source, the
   pattern's destinations, the destinations still owed: view indices, the last two as bits) in route_contexts' order
   (mesh_switch_contexts), and each one's w [SPLIT_FS], published as the weights are (seqlock).  `active` set, a piece
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
   (pushed and not yet completed), its outbox's cursors, and its feedback: SENDs completed, their bytes, their service
   and wait (ns, summed), and the estimate its child keeps (rate in bytes a ns and wait, two floats in one word) */
struct mesh_switch_lane {
  _Alignas(64) _Atomic uint32_t live,peer;
  _Atomic uint64_t queued,head,tail;
  _Atomic uint64_t sends,bytes,service,wait,landed,estimate;
  _Atomic uint64_t frames;          /* every SEND's and receive's frames on the lane (TN3205: the two ends alike); 0 a whole slot */
};
struct mesh_switch_cell { _Atomic uint64_t sequence,slot; };
/* a slot of the pool: its free-stack link, its references (SENDs posted of it and a delivery pending), and its
   state, with its last post and landing (ns, the caller's clock) */
enum { SLOT_FREE, SLOT_RECEIVE, SLOT_HELD, SLOT_SENDING };
struct mesh_switch_slot {
  _Alignas(64) _Atomic uint64_t next;
  _Atomic uint32_t refs,state;
  uint32_t lane,owner;
  uint64_t posted,landed,staged;
  /* a held slot whose one destination here waits for its client: the word it waits on (region offset, 0 none) and the
     value that word must reach; retried in full only once it has (mesh_switch_forward) */
  uint64_t wait_word,wait_value;
  /* the program that word is its (region offset of its mesh_declared, 0 none: the region's `control`, a declaration's):
     a wait ends where that program is no longer the one of the piece's key (closed and vacated: the piece is dropped) */
  uint64_t wait_program;
};
/* the trainer's records (rdma/tests/route): a draw (split.h struct route_draw and its behaviour log-probability), a
   piece delivered (its latency and queueing over its floor) or an execution completed at a node (its span of local
   time over its floor) */
enum { MESH_LOG_DRAW=1, MESH_LOG_PIECE, MESH_LOG_EXECUTION };
struct mesh_switch_record {
  uint32_t kind,node;               /* the node that made it */
  uint64_t key,execution,at;        /* the program, its execution, the node's clock (ns) */
  struct route_draw draw;           /* MESH_LOG_DRAW */
  float logp,latency,queueing,floor;
  uint32_t source,transfer,span,bytes;
  uint32_t context,point,dest,padding;   /* the draw's context at its node (UINT32_MAX: none, the weights drew), the
                                            policy refresh its split table is from, its first destination */
};
struct mesh_switch_log { _Alignas(64) _Atomic uint64_t head; _Alignas(64) _Atomic uint64_t tail,dropped; struct mesh_switch_record record[MESH_SWITCH_LOG]; };
/* where a crossing's time goes, counted in memory (never storage) by each process of the switch, one writer a set (a
   port child's at its port's index, the parent's at MESH_PORTS, the simulator's after), read by mesh-switch-stat:
   - a progress pass: its count, its time, and each part's (receive completions, receives posted, staging and the held
     retried, SENDs posted, SEND completions);
   - staging: pieces staged and their destinations, the time staging and switching them took, and the stagings put off
     by the credit window, by the transfer's window, or for want of a slot;
   - switching (mesh_switch_forward): calls, their time, groups made, slot copies, calls that left a piece held;
   - the wire: SENDs posted and their wait from the push onto the outbox to the post, SENDs completed and their time
     from post to completion;
   - receiving: pieces landed, destinations delivered here, their wait from landing to delivery, by hops (1, 2, 3+);
   - credits made and taken, and the refreshes and their time */
struct mesh_switch_counters {
  _Alignas(64) _Atomic uint64_t passes,pass_ns,part_ns[5];
  _Atomic uint64_t staged,staged_dests,stage_ns,credit_waits,window_waits,slot_waits;
  _Atomic uint64_t forwards,forward_ns,groups,copies,kept;
  _Atomic uint64_t posts,post_wait_ns,sent,send_ns;
  _Atomic uint64_t landed,delivered,deliver_wait_ns,hops[4];
  _Atomic uint64_t credits,credits_taken,refreshes,refresh_ns;
  _Atomic uint64_t unplaced[8];     /* deliveries put off, by why (mesh_switch_unplaced) */
  _Atomic uint64_t post_call_ns;    /* the posts' own time (the provider's post_send) */
  _Atomic uint64_t orphans;         /* destinations here dropped: their program gone (vacated, or replaced in its slot), or closed before
                                       the piece's execution (mesh_switch_gone) */
};
struct mesh_switch {
  uint32_t magic,slots;
  uint64_t slot_bytes,slot_base;    /* a slot's bytes (MESH_SWITCH_HEAD and a piece, whole frames) and the pool's region offset */
  _Alignas(64) _Atomic uint64_t free_head;
  _Atomic uint64_t free_count,stamp,refreshing;
  _Atomic uint32_t logging,weighted;   /* weighted: a draw from the weights (else the earliest arrival, the design's rule) */
  struct mesh_switch_view view;
  struct mesh_switch_weights weights;
  struct mesh_switch_splits splits;
  struct mesh_switch_lane lane[MESH_SWITCH_LANES];
  struct mesh_switch_slot slot[MESH_SWITCH_SLOTS];
  struct mesh_switch_cell ring[MESH_SWITCH_LANES][MESH_SWITCH_RING];
  struct mesh_switch_log log[MESH_SWITCH_LOGS];
  struct mesh_switch_counters counters[MESH_SWITCH_LOGS];
};
_Static_assert(sizeof(struct mesh_switch)<=MESH_SWITCH_BYTES,"the switch within its region bytes");
/* a counter of ring `ring`'s writer moved by v (one writer a ring: a load and a store, no read-modify-write) */
#define MESH_SWITCH_COUNT(sw,ring,field,v) do{ if((ring)<MESH_SWITCH_LOGS){ _Atomic uint64_t *c_=&(sw)->counters[ring].field; \
  atomic_store_explicit(c_,atomic_load_explicit(c_,memory_order_relaxed)+(uint64_t)(v),memory_order_relaxed);} }while(0)
static inline struct mesh_switch *mesh_switch_of(struct hdr *m){ return (struct mesh_switch *)((char *)m+m->switch_off); }
static inline unsigned char *mesh_switch_slot_at(struct hdr *m,const struct mesh_switch *sw,uint32_t s){
  return (unsigned char *)m+sw->slot_base+(uint64_t)s*sw->slot_bytes;
}
static inline struct mesh_dma_span mesh_switch_slot_span(const struct mesh_switch *sw,uint32_t s){
  return (struct mesh_dma_span){sw->slot_base+(uint64_t)s*sw->slot_bytes,sw->slot_bytes};
}
/* the word at region offset `at` raised to `value` where it is lower (release): a done word only rises */
static inline void mesh_switch_raise(void *m,uint64_t at,uint64_t value){
  if(!at)return;
  _Atomic uint64_t *word=(_Atomic uint64_t *)((char *)m+at);
  uint64_t old=atomic_load_explicit(word,memory_order_relaxed);
  while(old<value && !atomic_compare_exchange_weak_explicit(word,&old,value,memory_order_release,memory_order_relaxed)){}
}
static inline uint64_t mesh_switch_pack(float a,float b){ uint32_t x,y;memcpy(&x,&a,4);memcpy(&y,&b,4);return (uint64_t)x|(uint64_t)y<<32; }
static inline void mesh_switch_unpack(uint64_t v,float *a,float *b){ const uint32_t x=(uint32_t)v,y=(uint32_t)(v>>32);memcpy(a,&x,4);memcpy(b,&y,4); }
/* a piece's bytes at most (mesh-order.h's order_piece_bytes: one allocation block) and the slot that carries one */
static inline uint64_t mesh_switch_piece(const struct hdr *m){ return (uint64_t)m->block*m->pgsz; }
static inline uint64_t mesh_switch_slot_bytes(const struct hdr *m){ return (MESH_SWITCH_HEAD+MESH_SWITCH_UNITS*mesh_switch_piece(m)+4095)/4096*4096; }   /* whole frames of 4096 (mesh-bridge.h MESH_FRAME) */
/* the bytes every SEND on lane l carries (its frames: a short lane's MESH_SWITCH_SHORT, another's a whole slot) */
static inline uint64_t mesh_switch_lane_bytes(const struct mesh_switch *sw,uint32_t l){
  const uint64_t f=atomic_load_explicit(&sw->lane[l].frames,memory_order_relaxed);
  return f && f*4096<sw->slot_bytes?f*4096:sw->slot_bytes;
}

/* the switch made new over `slots` slots of the pool at region offset `base` (the bridge's own arena pages) */
static inline void mesh_switch_init(struct mesh_switch *sw,uint32_t slots,uint64_t slot_bytes,uint64_t base){
  memset(sw,0,sizeof *sw);
  sw->magic=MESH_SWITCH_MAGIC;sw->slots=slots<MESH_SWITCH_SLOTS?slots:MESH_SWITCH_SLOTS;sw->slot_bytes=slot_bytes;sw->slot_base=base;
  for(uint32_t l=0;l<MESH_SWITCH_LANES;l++)for(uint32_t i=0;i<MESH_SWITCH_RING;i++)atomic_store_explicit(&sw->ring[l][i].sequence,i,memory_order_relaxed);
  for(uint32_t s=sw->slots;s-->0;){
    sw->slot[s].next=atomic_load_explicit(&sw->free_head,memory_order_relaxed);
    atomic_store_explicit(&sw->free_head,(uint64_t)(s+1),memory_order_relaxed);
  }
  atomic_store_explicit(&sw->free_count,sw->slots,memory_order_relaxed);
  atomic_thread_fence(memory_order_release);
}

/* ---- The pool: a free stack any process takes from and gives to (a tag against ABA in the head's high half) ----- */
static inline uint32_t mesh_switch_take(struct mesh_switch *sw){
  uint64_t head=atomic_load_explicit(&sw->free_head,memory_order_acquire);
  for(;;){
    const uint32_t s=(uint32_t)head;
    if(!s)return UINT32_MAX;
    const uint64_t next=(atomic_load_explicit(&sw->slot[s-1].next,memory_order_acquire)&UINT32_MAX)|((head>>32)+1)<<32;
    if(atomic_compare_exchange_weak_explicit(&sw->free_head,&head,next,memory_order_acq_rel,memory_order_acquire)){
      atomic_fetch_sub_explicit(&sw->free_count,1,memory_order_relaxed);
      atomic_store_explicit(&sw->slot[s-1].refs,0,memory_order_relaxed);sw->slot[s-1].owner=0;
      return s-1;
    }
  }
}
static inline void mesh_switch_give(struct mesh_switch *sw,uint32_t s){
  atomic_store_explicit(&sw->slot[s].state,SLOT_FREE,memory_order_relaxed);
  uint64_t head=atomic_load_explicit(&sw->free_head,memory_order_acquire);
  for(;;){
    atomic_store_explicit(&sw->slot[s].next,head&UINT32_MAX,memory_order_relaxed);
    const uint64_t next=(uint64_t)(s+1)|((head>>32)+1)<<32;
    if(atomic_compare_exchange_weak_explicit(&sw->free_head,&head,next,memory_order_acq_rel,memory_order_acquire))break;
  }
  atomic_fetch_add_explicit(&sw->free_count,1,memory_order_relaxed);
}
/* one reference of slot s let go; the slot given back with its last */
static inline void mesh_switch_release(struct mesh_switch *sw,uint32_t s){
  if(atomic_fetch_sub_explicit(&sw->slot[s].refs,1,memory_order_acq_rel)==1)mesh_switch_give(sw,s);
}

/* ---- A lane's outbox: bounded, any process pushes, the lane's child pops ----------------------------------------- */
static inline int mesh_switch_push(struct mesh_switch *sw,uint32_t lane,uint32_t slot){
  struct mesh_switch_lane *x=sw->lane+lane;
  uint64_t at=atomic_load_explicit(&x->tail,memory_order_relaxed);
  for(;;){
    struct mesh_switch_cell *c=sw->ring[lane]+at%MESH_SWITCH_RING;
    const uint64_t seq=atomic_load_explicit(&c->sequence,memory_order_acquire);
    if(seq==at){
      if(atomic_compare_exchange_weak_explicit(&x->tail,&at,at+1,memory_order_relaxed,memory_order_relaxed)){
        atomic_store_explicit(&c->slot,slot,memory_order_relaxed);
        atomic_store_explicit(&c->sequence,at+1,memory_order_release);
        return 1;
      }
    }else if(seq<at)return 0;   /* full */
    else at=atomic_load_explicit(&x->tail,memory_order_relaxed);
  }
}
/* the next slot on the lane's outbox, UINT32_MAX none (one consumer: the lane's child) */
static inline uint32_t mesh_switch_peek(struct mesh_switch *sw,uint32_t lane){
  const uint64_t at=atomic_load_explicit(&sw->lane[lane].head,memory_order_relaxed);
  struct mesh_switch_cell *c=sw->ring[lane]+at%MESH_SWITCH_RING;
  return atomic_load_explicit(&c->sequence,memory_order_acquire)==at+1?(uint32_t)atomic_load_explicit(&c->slot,memory_order_relaxed):UINT32_MAX;
}
/* the slot k places behind the next on the lane's outbox, UINT32_MAX none (its consumer's) */
static inline uint32_t mesh_switch_peek_at(struct mesh_switch *sw,uint32_t lane,uint32_t k){
  const uint64_t at=atomic_load_explicit(&sw->lane[lane].head,memory_order_relaxed)+k;
  struct mesh_switch_cell *c=sw->ring[lane]+at%MESH_SWITCH_RING;
  return atomic_load_explicit(&c->sequence,memory_order_acquire)==at+1?(uint32_t)atomic_load_explicit(&c->slot,memory_order_relaxed):UINT32_MAX;
}
static inline void mesh_switch_pop(struct mesh_switch *sw,uint32_t lane){
  const uint64_t at=atomic_load_explicit(&sw->lane[lane].head,memory_order_relaxed);
  atomic_store_explicit(&sw->ring[lane][at%MESH_SWITCH_RING].sequence,at+MESH_SWITCH_RING,memory_order_release);
  atomic_store_explicit(&sw->lane[lane].head,at+1,memory_order_relaxed);
}

/* ---- The view and the weights: written by one process (the parent's refresh, the simulator), read by all -------- */
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
static inline void mesh_switch_sent(struct mesh_switch *sw,uint32_t lane,uint32_t slot,uint64_t bytes,uint64_t posted,uint64_t start,uint64_t done){
  struct mesh_switch_lane *x=sw->lane+lane;
  atomic_fetch_sub_explicit(&x->queued,bytes,memory_order_relaxed);
  atomic_fetch_add_explicit(&x->sends,1,memory_order_relaxed);atomic_fetch_add_explicit(&x->bytes,bytes,memory_order_relaxed);
  const uint64_t service=done>start?done-start:1,wait=start>posted?start-posted:0;
  atomic_fetch_add_explicit(&x->service,service,memory_order_relaxed);atomic_fetch_add_explicit(&x->wait,wait,memory_order_relaxed);
  float rate,waited;mesh_switch_unpack(atomic_load_explicit(&x->estimate,memory_order_relaxed),&rate,&waited);
  const float r=(float)bytes/(float)service;
  rate=rate>0?rate+(r-rate)/8:r;waited+=((float)wait-waited)/8;
  atomic_store_explicit(&x->estimate,mesh_switch_pack(rate,waited),memory_order_relaxed);
  if(slot!=UINT32_MAX)mesh_switch_release(sw,slot);
}
/* the lines a name carries taken into the view where newer than its own (a node's lines about its own links are the
   ones it measured: a link's line is only ever made by its sender) */
static inline void mesh_switch_lines_take(struct mesh_switch *sw,const struct mesh_switch_name *name){
  struct mesh_switch_view *v=&sw->view;
  for(uint32_t i=0;i<name->lines && i<MESH_SWITCH_LINES;i++){
    const struct mesh_switch_line *a=name->line+i;
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
/* the lines a name sends: this node's own links first (their wait the least bytes queued over the rate of the
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
    if(wait<least || (wait==least && *best!=UINT32_MAX &&
                      atomic_load_explicit(&sw->lane[l].queued,memory_order_relaxed)<atomic_load_explicit(&sw->lane[*best].queued,memory_order_relaxed))){least=wait;*best=l;}
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
/* The policy's contexts at this node (route.h route_contexts' order, restricted to it): for each rank a in view order
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
/* A piece at this node (its name: its destinations and path) split over the node's out-neighbours: each group's lane
   (the neighbour's cable with the least bytes queued) and its destinations (bits over the name's dest entries).
   The draw's record (split.h struct route_draw, as rdma/tests/route reads it): for each owed destination d and
   neighbour k, the arrival through k (its lanes' wait now, the piece's time on them, the link's latency, then the
   refresh's cost-to-go from k to d), whether k reaches d without coming back here (valid), and whether it is allowed
   from the piece's path (off it, reaching d avoiding it: mask).
   The draw, at u = the piece's swizzle at this node:
   - where the policy's split tables are active and hold the piece's context (its source, its destinations as staged,
     the ones owed: mesh_switch_split_w), the split is split.h's table under the context's w at u: exactly the
     trainer's action, *logp its log-probability;
   - else each destination i takes the neighbour k at which its masked, renormalised weights' distribution first passes
     u; where its weights give its allowed neighbours nothing, the allowed one of the least cost-to-go; *logp the
     length of u's interval that makes the split.
   `record` (NULL: none) receives the record, *context the context drawn from (UINT32_MAX: the weights), *point the
   tables' refresh.  Returns the groups; UINT32_MAX where a destination is reached by no allowed neighbour (the piece
   waits for the view to change). */
static inline uint32_t mesh_switch_choose(struct mesh_switch *sw,const struct mesh_switch_view *v,const struct mesh_switch_weights *w,
                                          const struct mesh_switch_name *name,uint64_t salt,uint32_t *lane,uint64_t *group,
                                          struct route_draw *record,float *logp,uint32_t *context,uint64_t *point){
  const uint32_t self=mesh_switch_index(v,v->self);
  uint64_t visited=0;
  for(uint32_t h=0;h<=name->hops && h<MESH_SWITCH_N;h++){const uint32_t i=mesh_switch_index(v,name->path[h]);if(i<64)visited|=UINT64_C(1)<<i;}
  if(self<64)visited|=UINT64_C(1)<<self;
  const double u=mesh_switch_uniform(name->key^name->execution*0x9e3779b97f4a7c15ULL^((uint64_t)name->dest[0].transfer<<32)^name->dest[0].span^
                                     ((uint64_t)v->self<<48)^salt);
  uint32_t deg=w->deg<MESH_SWITCH_N?w->deg:MESH_SWITCH_N,choice[MESH_SWITCH_N];
  uint32_t best_lane[MESH_SWITCH_N];double rate[MESH_SWITCH_N],wait[MESH_SWITCH_N],latency[MESH_SWITCH_N];
  for(uint32_t k=0;k<deg;k++){
    const uint32_t nb=w->neighbour[k];
    const struct mesh_switch_link *x=nb<v->nodes?mesh_switch_link_of((struct mesh_switch_view *)v,v->self,v->node[nb]):NULL;
    /* the lane's wait with this piece's own bytes on it: its time to clear the lane (design: (backlog + bytes) / rate) */
    wait[k]=nb<v->nodes?mesh_switch_lanes_among_for(sw,w->toward[k],v->node[nb],x?x->rate/x->cables:0,best_lane+k,rate+k,-1,MESH_SWITCH_HEAD+(uint64_t)name->bytes):INFINITY;
    latency[k]=x?x->latency:0;
  }
  /* the policy's context: the piece's source and destinations as staged, the owed ones (view indices) */
  uint32_t owed=0,wanted=0;
  for(uint32_t i=0;i<name->dests;i++){const uint32_t d=mesh_switch_index(v,name->dest[i].node);if(d<32)owed|=1u<<d;}
  for(uint32_t i=0;i<name->nwanted && i<MESH_SWITCH_N;i++){const uint32_t d=mesh_switch_index(v,name->wanted[i]);if(d<32)wanted|=1u<<d;}
  float cw[SPLIT_FS],unit=1;uint32_t ctx=UINT32_MAX,argmax=0;uint64_t at_point=0;
  const int split=v->nodes<=SPLIT_N && deg<=SPLIT_DEG &&
                  mesh_switch_split_w(sw,mesh_switch_index(v,name->source),wanted,owed,cw,&ctx,&at_point,&argmax,&unit);
  if(!split)ctx=UINT32_MAX;
  if(context)*context=ctx;
  if(point)*point=at_point;
  struct route_draw local;
  if(!record && split)record=&local;
  double lo=0,hi=1;
  const int logged=record && v->nodes<=SPLIT_N && deg<=SPLIT_DEG;
  if(logged){memset(record,0,sizeof *record);record->deg=deg;record->unit=unit>0?unit:1;}
  uint32_t ntok=1;
  for(uint32_t i=0;i<name->dests;i++){
    const uint32_t d=mesh_switch_index(v,name->dest[i].node);
    choice[i]=UINT32_MAX;
    if(d>=v->nodes)return UINT32_MAX;
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
    if(!allowed)return UINT32_MAX;
    ntok*=allowed;
    if(logged)record->S|=1u<<d;
    if(split)continue;
    /* the design's rule: the allowed neighbour of the earliest arrival at d, its lane's backlog and this piece's
       bytes on it, the link's latency, then the cost-to-go from it (the weights' draw where `weighted` is set) */
    if(!atomic_load_explicit(&sw->weighted,memory_order_relaxed)){choice[i]=cheapest;continue;}
    if(total>0){
      double at=0;
      for(uint32_t k=0;k<deg && choice[i]==UINT32_MAX;k++){
        if(!ok[k])continue;
        const double p=w->w[d][k]/total;
        if(u<at+p || at+p>=1-1e-12){choice[i]=k;if(at>lo)lo=at;if(at+p<hi)hi=at+p;}
        at+=p;
      }
      if(choice[i]==UINT32_MAX){choice[i]=cheapest;}
    }else choice[i]=cheapest;
  }
  if(split){
    /* split.h's table under the context's w: the token's digit for each owed destination, in S's order */
    float lp=0;
    const uint32_t tok=mesh_switch_split_draw(record,cw,u,(int)argmax,&lp);
    if(tok==UINT32_MAX)return UINT32_MAX;
    uint32_t y=tok;
    for(uint32_t dd=0;dd<SPLIT_N;dd++){
      if(!(record->S>>dd&1))continue;
      for(uint32_t i=0;i<name->dests;i++)if(mesh_switch_index(v,name->dest[i].node)==dd)choice[i]=y%deg;
      y/=deg;
    }
    record->token=tok;record->ntok=ntok;
    if(logp)*logp=lp;
  }else{
    if(logged){
      uint32_t tok=0,place=1;
      for(uint32_t dd=0;dd<SPLIT_N;dd++){
        if(!(record->S>>dd&1))continue;
        for(uint32_t i=0;i<name->dests;i++)if(mesh_switch_index(v,name->dest[i].node)==dd){tok+=choice[i]*place;break;}
        place*=deg;
      }
      record->token=tok;record->ntok=ntok;
    }
    if(logp)*logp=hi>lo?(float)log(hi-lo):0;
  }
  uint32_t used=0;
  for(uint32_t i=0;i<name->dests;i++){
    const uint32_t k=choice[i],l=best_lane[k];
    uint32_t g=0;while(g<used && lane[g]!=l)g++;
    if(g==used){lane[used]=l;group[used]=0;used++;}
    group[g]|=UINT64_C(1)<<i;
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
  /* the trained policy (rdma/tests/route): each of this node's contexts' w from its tokens (deployed: forward.c's
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
   at the rate eta (1: the target itself), the weights published, this node's stamp moved.  `fixed` set: the target
   is each destination's one cheapest neighbour on the unloaded fabric (every wait 0), the compiled routing's analog,
   for comparison; the weights then never move after their first refresh */
static inline void mesh_switch_refresh(struct mesh_switch *sw,const struct mesh_switch_policy *policy,uint32_t draws,float eta,uint64_t seed,
                                       double page,int fixed){
  struct mesh_switch_view v;mesh_switch_view_read(sw,&v);
  struct mesh_switch_state s;mesh_switch_state_make(sw,&v,page,&s);
  float target[MESH_SWITCH_N][MESH_SWITCH_N];memset(target,0,sizeof target);
  if(fixed){
    for(uint32_t l=0;l<v.links;l++)s.hop[l]=page/(v.link[l].rate/v.link[l].cables)+v.link[l].latency;
    for(uint32_t k=0;k<s.deg;k++)for(uint32_t l=0;l<v.links;l++)if(v.link[l].from==v.self && mesh_switch_index(&v,v.link[l].to)==s.neighbour[k])s.first[k]=s.hop[l];
    for(uint32_t l=0;l<v.links;l++)s.spread[l]=0;
    mesh_switch_thompson(&s,1,seed,target);
    eta=1;
  }else if(!atomic_load_explicit(&sw->weighted,memory_order_relaxed) && !(policy && policy->target)){
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
    const float unit=(float)mesh_switch_unit(&v,page);
    atomic_fetch_add_explicit(&sw->splits.serial,1,memory_order_acq_rel);
    sw->splits.unit=unit;sw->splits.unit_view=v.serial+1;
    atomic_fetch_add_explicit(&sw->splits.serial,1,memory_order_release);
  }
  /* this node's own links' lines, for the names it sends: each one's wait now, its spread the wait's change */
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
  if(ring>=MESH_SWITCH_LOGS || !atomic_load_explicit(&sw->logging,memory_order_relaxed))return;
  struct mesh_switch_log *g=sw->log+ring;
  const uint64_t tail=atomic_load_explicit(&g->tail,memory_order_relaxed);
  if(tail-atomic_load_explicit(&g->head,memory_order_acquire)>=MESH_SWITCH_LOG){atomic_fetch_add_explicit(&g->dropped,1,memory_order_relaxed);return;}
  g->record[tail%MESH_SWITCH_LOG]=*r;
  atomic_store_explicit(&g->tail,tail+1,memory_order_release);
}
/* the next record of ring `ring`, 0 none (one consumer: mesh-switch-log, the simulator) */
static inline int mesh_switch_log_take(struct mesh_switch *sw,uint32_t ring,struct mesh_switch_record *r){
  struct mesh_switch_log *g=sw->log+ring;
  const uint64_t head=atomic_load_explicit(&g->head,memory_order_relaxed);
  if(head==atomic_load_explicit(&g->tail,memory_order_acquire))return 0;
  *r=g->record[head%MESH_SWITCH_LOG];
  atomic_store_explicit(&g->head,head+1,memory_order_release);
  return 1;
}

/* ---- Programs: what the clients declared (mesh.h struct mesh_declared), their progress -------------------------
   A program's progress, each direction (bridge-held, at `progress[d]`, made as it is declared: mesh_switch_progress_init):
   a transfer's executions done in order (`complete`), its done word's next transfer and first (a word's transfers in
   their order), and MESH_SWITCH_WINDOW cells, each an execution's units, its tag that execution + 1, a bit a unit */
struct mesh_switch_transfer { _Atomic uint64_t complete; uint32_t first_same,next_same,words,padding; uint64_t cells; };
struct mesh_switch_cell_head { _Atomic uint64_t tag,count; };
static inline uint64_t mesh_switch_cell_bytes(uint64_t units){ return sizeof(struct mesh_switch_cell_head)+(units+63)/64*8; }
static inline uint64_t mesh_switch_progress_bytes(const struct mesh_crossing *t,uint64_t n){
  uint64_t b=n*sizeof(struct mesh_switch_transfer);
  for(uint64_t i=0;i<n;i++)b+=MESH_SWITCH_WINDOW*mesh_switch_cell_bytes(t[i].units);
  return b;
}
static inline struct mesh_switch_transfer *mesh_switch_progress(struct hdr *m,const struct mesh_declared *p,int d){
  return p->progress[d]?(struct mesh_switch_transfer *)((char *)m+p->progress[d]):NULL;
}
static inline struct mesh_switch_cell_head *mesh_switch_cell(struct hdr *m,const struct mesh_declared *p,int d,uint32_t t,uint64_t e){
  const struct mesh_switch_transfer *x=mesh_switch_progress(m,p,d)+t;
  return (struct mesh_switch_cell_head *)((char *)m+p->progress[d]+x->cells+(e%MESH_SWITCH_WINDOW)*(sizeof(struct mesh_switch_cell_head)+x->words*8));
}
static inline void mesh_switch_progress_init(struct hdr *m,struct mesh_declared *p,int d,const struct mesh_crossing *t,uint64_t n){
  struct mesh_switch_transfer *x=mesh_switch_progress(m,p,d);
  if(!x)return;
  uint64_t at=n*sizeof *x;
  for(uint64_t i=0;i<n;i++){
    x[i]=(struct mesh_switch_transfer){0,(uint32_t)i,UINT32_MAX,(uint32_t)((t[i].units+63)/64),0,at};
    at+=MESH_SWITCH_WINDOW*mesh_switch_cell_bytes(t[i].units);
  }
  for(uint64_t i=0;i<n;i++){
    if(!t[i].done)continue;
    for(uint64_t j=i+1;j<n;j++)if(t[j].done==t[i].done){x[i].next_same=(uint32_t)j;break;}
    for(uint64_t j=0;j<i;j++)if(t[j].done==t[i].done){x[i].first_same=(uint32_t)j;break;}
  }
  for(uint64_t i=0;i<n;i++)for(uint64_t e=0;e<MESH_SWITCH_WINDOW;e++){
    struct mesh_switch_cell_head *c=mesh_switch_cell(m,p,d,(uint32_t)i,e);
    memset(c,0,sizeof *c+x[i].words*8);
    atomic_store_explicit(&c->tag,e+1,memory_order_relaxed);
  }
  atomic_thread_fence(memory_order_release);
}
static inline const struct mesh_crossing *mesh_switch_table(const struct hdr *m,const struct mesh_declared *p,int d){
  const uint64_t at=p->program.table[d],n=p->program.transfers[d];
  if(!at || !n || at%8 || at<m->data_off || at>m->length || n>(m->length-at)/sizeof(struct mesh_crossing))return NULL;
  return (const struct mesh_crossing *)((const char *)m+at);
}
/* the done word of transfer t's word raised as far as its transfers' prefix is done: the executions done of each of
   them, in the word's order, (execution, transfer) the order of the node's pieces */
static inline void mesh_switch_word(struct hdr *m,const struct mesh_declared *p,int d,const struct mesh_crossing *table,uint32_t t){
  const struct mesh_switch_transfer *x=mesh_switch_progress(m,p,d);
  if(!table[t].done || table[t].done%8 || table[t].done+8>m->length)return;
  const uint32_t first=x[t].first_same;
  uint64_t least=UINT64_MAX;
  for(uint32_t k=first;k!=UINT32_MAX;k=x[k].next_same){const uint64_t c=atomic_load_explicit(&x[k].complete,memory_order_acquire);if(c<least)least=c;}
  uint32_t last=UINT32_MAX;uint64_t at=least;
  for(uint32_t k=first;k!=UINT32_MAX;k=x[k].next_same){if(atomic_load_explicit(&x[k].complete,memory_order_acquire)==least)break;last=k;}
  if(last==UINT32_MAX){
    if(!least)return;
    at=least-1;
    for(uint32_t k=first;k!=UINT32_MAX;k=x[k].next_same)last=k;
  }
  mesh_switch_raise(m,table[last].done,table[last].done_value+at*table[last].done_step);
}
/* unit `span` of (execution e, transfer t) of direction d counted (once: its bit); 1 newly, 0 already, -1 not yet
   placeable (e past the window, or its cell not yet made free).  An execution all of whose units are counted moves its
   transfer's `complete` on (with every later one already whole), each move clearing the cell for its execution + W,
   and raises the word */
static inline int mesh_switch_count(struct hdr *m,const struct mesh_declared *p,int d,const struct mesh_crossing *table,uint32_t t,uint64_t e,uint64_t span){
  struct mesh_switch_transfer *x=mesh_switch_progress(m,p,d)+t;
  if(e>=atomic_load_explicit(&x->complete,memory_order_acquire)+MESH_SWITCH_WINDOW)return -1;
  if(e<atomic_load_explicit(&x->complete,memory_order_acquire))return 0;
  struct mesh_switch_cell_head *c=mesh_switch_cell(m,p,d,t,e);
  if(atomic_load_explicit(&c->tag,memory_order_acquire)!=e+1)return -1;
  _Atomic uint64_t *bits=(_Atomic uint64_t *)(c+1);
  const uint64_t bit=UINT64_C(1)<<(span%64);
  if(atomic_fetch_or_explicit(bits+span/64,bit,memory_order_acq_rel)&bit)return 0;
  if(atomic_fetch_add_explicit(&c->count,1,memory_order_acq_rel)+1<table[t].units)return 1;
  /* whole: `complete` moved past every whole execution from it, by whoever wins each move */
  for(;;){
    uint64_t done=atomic_load_explicit(&x->complete,memory_order_acquire);
    struct mesh_switch_cell_head *h=mesh_switch_cell(m,p,d,t,done);
    if(atomic_load_explicit(&h->tag,memory_order_acquire)!=done+1 || atomic_load_explicit(&h->count,memory_order_acquire)<table[t].units)break;
    if(!atomic_compare_exchange_strong_explicit(&x->complete,&done,done+1,memory_order_acq_rel,memory_order_acquire))continue;
    memset((void *)(h+1),0,x->words*8);
    atomic_store_explicit(&h->count,0,memory_order_relaxed);
    atomic_store_explicit(&h->tag,done+1+MESH_SWITCH_WINDOW,memory_order_release);
  }
  mesh_switch_word(m,p,d,table,t);
  return 1;
}
/* where piece n of direction d of program p lies: its execution, transfer and span; 0 where its table holds none */
static inline int mesh_switch_locate(const struct mesh_declared *p,const struct mesh_crossing *t,int d,uint64_t n,uint64_t *e,uint32_t *transfer,uint64_t *span){
  const uint64_t pieces=p->program.pieces[d];
  if(!pieces || !t)return 0;
  *e=n/pieces;const uint64_t within=n%pieces;
  uint32_t lo=0,hi=(uint32_t)p->program.transfers[d];
  while(hi-lo>1){const uint32_t mid=lo+(hi-lo)/2;if(t[mid].first<=within)lo=mid;else hi=mid;}
  if(within<t[lo].first || within-t[lo].first>=t[lo].units)return 0;
  *transfer=lo;*span=within-t[lo].first;
  return 1;
}
/* whether (execution e, transfer t, span s) of direction d of program p may move: its client issued its step, the
   program has not closed before e, and its gate holds (mesh-order.h order_gate's words and thresholds) */
static inline int mesh_switch_ready(const struct hdr *m,const struct mesh_declared *p,const struct mesh_crossing *t,int d,uint64_t e,uint64_t s){
  if(e>=atomic_load_explicit(&p->closed,memory_order_acquire))return 0;
  if(atomic_load_explicit(&p->issued,memory_order_acquire)<=e*p->program.period+t->step)return 0;
  if(!t->gate)return 1;
  const uint64_t add=e*t->gate_step+(d || t->flags&MESH_GATE_WHOLE?0:s+1);
  if(!t->gates){
    const uint64_t value=atomic_load_explicit((const _Atomic uint64_t *)((const char *)m+t->gate),memory_order_acquire);
    return value!=UINT64_MAX && value>=t->gate_base+add;
  }
  const struct mesh_gate_term *g=(const void *)((const char *)m+t->gate);
  for(uint32_t k=0;k<t->gates;k++){
    const uint64_t value=atomic_load_explicit((const _Atomic uint64_t *)((const char *)m+g[k].word),memory_order_acquire);
    if(value==UINT64_MAX || value<g[k].threshold+add)return 0;
  }
  return 1;
}
/* a program slot's owner as the bridge reads it: 0 vacant, UINT64_MAX while a declaration makes it (mesh-call.c) */
static inline int mesh_switch_owned(const struct mesh_declared *p){
  const uint64_t o=atomic_load_explicit(&p->owner,memory_order_acquire);
  return o && o!=UINT64_MAX;
}
/* whether a closed program is wholly done: every transfer of both directions complete through its closing execution */
static inline int mesh_switch_done(const struct hdr *m,const struct mesh_declared *p){
  const uint64_t closed=atomic_load_explicit(&p->closed,memory_order_acquire);
  if(closed==UINT64_MAX)return 0;
  for(int d=0;d<2;d++){
    if(!p->progress[d])continue;
    const struct mesh_switch_transfer *x=(const struct mesh_switch_transfer *)((const char *)m+p->progress[d]);
    for(uint64_t t=0;t<p->program.transfers[d];t++)if(atomic_load_explicit(&x[t].complete,memory_order_acquire)<closed)return 0;
  }
  return 1;
}
static inline struct mesh_node *mesh_switch_order(struct hdr *m,uint32_t node){
  for(uint32_t n=0;n<MESH_NODES;n++){struct mesh_node *x=mesh_node_at(m,n);if(atomic_load_explicit(&x->node,memory_order_acquire)==node)return x;}
  return NULL;
}

/* ---- The receive window (as multipath TCP's, at the data-sequence level) ---------------------------------------
   A slot pool holds what lands until it is placed, so a sender must not run further ahead of its receiver than the
   receiver's pool holds: a piece its receiver cannot place yet (its client has not issued that execution, or one of its
   pieces is still on a slower path) waits in a slot there.  So each receiving end tells the sending end, in a credit (a
   piece of kind MESH_PIECE_CREDIT, switched like any other, naming the program's key), the executions it has wholly
   received; a sender stages an execution only within mesh_switch_ahead executions of the last credit it heard (at
   most MESH_SWITCH_WINDOW, and no more than a quarter of the pool's slots of the program's pieces).  A receiver sends a
   credit each time its wholly received executions pass the last it told by half that, so a sender at the window's edge
   always hears one; the first MESH_SWITCH_WINDOW executions need none.  No queue pair waits on it */
enum { MESH_PIECE_DATA, MESH_PIECE_CREDIT };
static inline uint64_t mesh_switch_ahead(const struct mesh_switch *sw,const struct mesh_declared *p){
  const uint64_t pieces=p->program.pieces[0]>p->program.pieces[1]?p->program.pieces[0]:p->program.pieces[1];
  const uint64_t room=pieces?sw->slots/4/pieces:MESH_SWITCH_WINDOW;
  return room<1?1:room<MESH_SWITCH_WINDOW?room:MESH_SWITCH_WINDOW;
}
/* the executions wholly received of program p (every receive transfer's, the least); UINT64_MAX where it receives nothing */
static inline uint64_t mesh_switch_received(struct hdr *m,const struct mesh_declared *p){
  const struct mesh_switch_transfer *x=mesh_switch_progress(m,p,1);
  if(!x)return UINT64_MAX;
  uint64_t least=UINT64_MAX;
  for(uint64_t t=0;t<p->program.transfers[1];t++){const uint64_t c=atomic_load_explicit(&x[t].complete,memory_order_acquire);if(c<least)least=c;}
  return least;
}
/* program p toward node x: a credit due (its executions wholly received past the last told by half the window) made in
   a free slot, one reference held, in *slot: 1, else 0 */
static inline int mesh_switch_credit(struct hdr *m,struct mesh_switch *sw,struct mesh_node *x,struct mesh_declared *p,uint64_t now,uint32_t *slot){
  const uint64_t received=mesh_switch_received(m,p);
  uint64_t told=atomic_load_explicit(&p->credited,memory_order_acquire);
  const uint64_t step=(mesh_switch_ahead(sw,p)+1)/2;
  if(received==UINT64_MAX || received<told+step)return 0;
  const uint32_t s=mesh_switch_take(sw);
  if(s==UINT32_MAX)return 0;
  if(!atomic_compare_exchange_strong_explicit(&p->credited,&told,received,memory_order_acq_rel,memory_order_acquire)){mesh_switch_give(sw,s);return 0;}
  struct mesh_switch_name *name=(struct mesh_switch_name *)mesh_switch_slot_at(m,sw,s);
  memset(name,0,offsetof(struct mesh_switch_name,line));
  name->magic=MESH_SWITCH_MAGIC;name->source=m->node;name->key=p->program.key;name->execution=received;name->kind=MESH_PIECE_CREDIT;
  name->dests=1;name->from=m->node;name->path[0]=m->node;
  name->dest[0]=(struct mesh_switch_dest){atomic_load_explicit(&x->node,memory_order_relaxed),UINT32_MAX,0,0};
  atomic_store_explicit(&sw->slot[s].refs,1,memory_order_relaxed);
  atomic_store_explicit(&sw->slot[s].state,SLOT_HELD,memory_order_relaxed);
  sw->slot[s].staged=now;sw->slot[s].landed=now;sw->slot[s].wait_word=0;
  *slot=s;
  return 1;
}
/* a credit landed here: the sending program toward its source with its key told the executions received there */
static inline void mesh_switch_credit_take(struct hdr *m,const struct mesh_switch_name *name){
  struct mesh_node *x=mesh_switch_order(m,name->source);
  for(uint32_t k=0;x && k<MESH_PROGRAMS;k++){
    struct mesh_declared *p=x->program+k;
    if(!mesh_switch_owned(p) || p->program.key!=name->key)continue;
    uint64_t was=atomic_load_explicit(&p->credit,memory_order_relaxed);
    while(was<name->execution && !atomic_compare_exchange_weak_explicit(&p->credit,&was,name->execution,memory_order_acq_rel,memory_order_relaxed)){}
  }
}

/* ---- Staging: a send piece claimed and copied into a slot with its name ----------------------------------------
   Program p toward node x's next send piece, where it may move and its transfer's window holds it: claimed (its
   cursor moved by one compare-exchange, so exactly one process stages it), copied into a free slot after its name,
   its send-side unit counted (its done word rising).  The slot, one reference held, in *slot.  1 staged; 0 none; -1 no
   free slot; -2 ready but past the credit window; -3 ready but past its transfer's window */
static inline int mesh_switch_stage(struct hdr *m,struct mesh_switch *sw,struct mesh_node *x,struct mesh_declared *p,uint64_t now,uint32_t *slot){
  const struct mesh_crossing *table=mesh_switch_table(m,p,0);
  if(!table || !p->progress[0])return 0;
  const uint64_t n=atomic_load_explicit(&p->claimed,memory_order_acquire);
  uint64_t e,span;uint32_t t;
  if(!mesh_switch_locate(p,table,0,n,&e,&t,&span) || !mesh_switch_ready(m,p,table+t,0,e,span))return 0;
  if(e>=atomic_load_explicit(&mesh_switch_progress(m,p,0)[t].complete,memory_order_acquire)+MESH_SWITCH_WINDOW)return -3;
  if(e>=atomic_load_explicit(&p->credit,memory_order_acquire)+mesh_switch_ahead(sw,p))return -2;
  const struct mesh_dma_span bytes=((const struct mesh_dma_span *)((const char *)m+table[t].spans))[span];
  const uint32_t s=mesh_switch_take(sw);
  if(s==UINT32_MAX)return -1;
  uint64_t expect=n;
  if(!atomic_compare_exchange_strong_explicit(&p->claimed,&expect,n+1,memory_order_acq_rel,memory_order_acquire)){mesh_switch_give(sw,s);return 0;}
  unsigned char *at=mesh_switch_slot_at(m,sw,s);
  struct mesh_switch_name *name=(struct mesh_switch_name *)at;
  const uint64_t piece=mesh_switch_piece(m);
  const int valid=mesh_dma_span_valid(m,bytes) && bytes.bytes<=piece;
  memset(name,0,offsetof(struct mesh_switch_name,line));
  *name=(struct mesh_switch_name){.magic=MESH_SWITCH_MAGIC,.source=m->node,.key=p->program.key,.execution=e,
    .bytes=valid?(uint32_t)bytes.bytes:0,.hops=0,.dests=1,.from=m->node};
  name->path[0]=m->node;name->born=now;
  name->dest[0]=(struct mesh_switch_dest){atomic_load_explicit(&x->node,memory_order_relaxed),t,(uint32_t)span,1};
  if(valid)memcpy(at+MESH_SWITCH_HEAD,(const char *)m+bytes.offset,bytes.bytes);
  atomic_store_explicit(&sw->slot[s].refs,1,memory_order_relaxed);
  atomic_store_explicit(&sw->slot[s].state,SLOT_HELD,memory_order_relaxed);
  sw->slot[s].staged=now;sw->slot[s].landed=now;sw->slot[s].wait_word=0;
  mesh_switch_count(m,p,0,table,t,e,span);
  /* the transfer's next units, where ready and the slot holds them: claimed with it, one SEND for them all (TP-4
     B=32 8.24 ms/step packed against 8.57 not, B=128 14.9 against 15.9, B=8 within its noise either way) */
  struct mesh_dma_span unit[MESH_SWITCH_UNITS];unit[0]=bytes;
  uint32_t units=1;
  while(valid && units<MESH_SWITCH_UNITS){
    uint64_t e2,span2;uint32_t t2;
    const struct mesh_dma_span next=span+units<table[t].units?((const struct mesh_dma_span *)((const char *)m+table[t].spans))[span+units]:(struct mesh_dma_span){0,0};
    if(span+units>=table[t].units || !mesh_switch_locate(p,table,0,n+units,&e2,&t2,&span2) || e2!=e || t2!=t || span2!=span+units ||
       !mesh_dma_span_valid(m,next) || next.bytes>piece || MESH_SWITCH_HEAD+name->bytes+next.bytes>sw->slot_bytes || !mesh_switch_ready(m,p,table+t,0,e,span2))break;
    uint64_t want=n+units;
    if(!atomic_compare_exchange_strong_explicit(&p->claimed,&want,n+units+1,memory_order_acq_rel,memory_order_acquire))break;
    memcpy(at+MESH_SWITCH_HEAD+name->bytes,(const char *)m+next.offset,next.bytes);
    name->bytes+=(uint32_t)next.bytes;unit[units++]=next;
    mesh_switch_count(m,p,0,table,t,e,span2);
  }
  name->dest[0].units=units;
  /* the same bytes released toward other nodes by programs of the same key (one declaration's: a collective's
     all-gather or broadcast sends one span to several), in the same execution, as their next pieces: claimed with it,
     each a destination of the one slot (multicast: the switch then shares SENDs while their paths agree) */
  /* the programs of p's key toward other nodes, as this thread last found them (looked for again every
     MESH_SWITCH_SIBLING_USES uses, or where p's key moved): each read again below, so a stale one is let be */
  static __thread struct { const struct mesh_declared *p; uint64_t key; uint32_t uses,n; struct mesh_declared *q[MESH_SWITCH_N];struct mesh_node *y[MESH_SWITCH_N]; } kin[MESH_SWITCH_KIN];
  __typeof__(kin[0]) *c=kin+((uintptr_t)p/sizeof *p)%MESH_SWITCH_KIN;
  if(valid && (c->p!=p || c->key!=p->program.key || ++c->uses>=MESH_SWITCH_SIBLING_USES)){
    c->p=p;c->key=p->program.key;c->uses=0;c->n=0;
    for(uint32_t k=0;k<MESH_NODES;k++){
      struct mesh_node *y=mesh_node_at(m,k);
      if(y==x || !atomic_load_explicit(&y->node,memory_order_acquire))continue;
      for(uint32_t j=0;j<MESH_PROGRAMS && c->n<MESH_SWITCH_N;j++){
        struct mesh_declared *q=y->program+j;
        if(mesh_switch_owned(q) && q->program.key==p->program.key){c->q[c->n]=q;c->y[c->n]=y;c->n++;}
      }
    }
  }
  for(uint32_t i=0;i<c->n && valid && name->dests<MESH_SWITCH_N;i++){
    struct mesh_node *y=c->y[i];
    {
      struct mesh_declared *q=c->q[i];
      if(!mesh_switch_owned(q) || q->program.key!=p->program.key || !q->progress[0])continue;
      const struct mesh_crossing *qt=mesh_switch_table(m,q,0);
      uint64_t qn=atomic_load_explicit(&q->claimed,memory_order_acquire),qe,qs;uint32_t qtr;
      if(!qt || !mesh_switch_locate(q,qt,0,qn,&qe,&qtr,&qs) || qe!=e)continue;
      int same=qs+units<=qt[qtr].units;
      for(uint32_t u=0;u<units && same;u++){
        const struct mesh_dma_span other=((const struct mesh_dma_span *)((const char *)m+qt[qtr].spans))[qs+u];
        uint64_t ue,us;uint32_t ut;
        same=other.offset==unit[u].offset && other.bytes==unit[u].bytes && mesh_switch_locate(q,qt,0,qn+u,&ue,&ut,&us) && ue==qe && ut==qtr && us==qs+u &&
             mesh_switch_ready(m,q,qt+qtr,0,qe,qs+u);
      }
      if(!same ||
         qe>=atomic_load_explicit(&mesh_switch_progress(m,q,0)[qtr].complete,memory_order_acquire)+MESH_SWITCH_WINDOW ||
         qe>=atomic_load_explicit(&q->credit,memory_order_acquire)+mesh_switch_ahead(sw,q))continue;
      if(!atomic_compare_exchange_strong_explicit(&q->claimed,&qn,qn+units,memory_order_acq_rel,memory_order_acquire))continue;
      name->dest[name->dests++]=(struct mesh_switch_dest){atomic_load_explicit(&y->node,memory_order_relaxed),qtr,(uint32_t)qs,units};
      for(uint32_t u=0;u<units;u++)mesh_switch_count(m,q,0,qt,qtr,qe,qs+u);
    }
  }
  name->nwanted=name->dests;
  for(uint32_t i=0;i<name->dests;i++)name->wanted[i]=name->dest[i].node;
  *slot=s;
  return 1;
}

/* ---- Delivery: a landed piece copied to the receiving program's span ---------------------------------------------
   dest (one of the name's entries, this node's): the program toward the source with the piece's key, its transfer and
   span; copied there once that program's client issued the step and its gate holds.  1 delivered (or already); 0 not
   yet (no such program yet, or not ready, or past its window: it waits in its slot) */
static inline int mesh_switch_deliver(struct hdr *m,const struct mesh_switch_name *name,const struct mesh_switch_dest *dest,const unsigned char *data,
                                      struct mesh_declared **found){
  struct mesh_node *x=mesh_switch_order(m,name->source);
  if(!x)return 0;
  struct mesh_declared *p=NULL;
  for(uint32_t k=0;k<MESH_PROGRAMS && !p;k++)
    if(mesh_switch_owned(x->program+k) && x->program[k].program.key==name->key && x->program[k].progress[1])p=x->program+k;
  if(!p)return 0;
  if(found)*found=p;
  const struct mesh_crossing *table=mesh_switch_table(m,p,1);
  const uint32_t units=dest->units?dest->units:1;
  if(!table || dest->transfer>=p->program.transfers[1] || (uint64_t)dest->span+units>table[dest->transfer].units)return 0;
  const struct mesh_crossing *t=table+dest->transfer;
  if(!mesh_switch_ready(m,p,t,1,name->execution,dest->span))return 0;
  struct mesh_switch_transfer *progress=mesh_switch_progress(m,p,1)+dest->transfer;
  const uint64_t complete=atomic_load_explicit(&progress->complete,memory_order_acquire);
  if(name->execution<complete)return 1;
  if(name->execution>=complete+MESH_SWITCH_WINDOW)return 0;
  struct mesh_switch_cell_head *c=mesh_switch_cell(m,p,1,dest->transfer,name->execution);
  if(atomic_load_explicit(&c->tag,memory_order_acquire)!=name->execution+1)return 0;
  const _Atomic uint64_t *bits=(const _Atomic uint64_t *)(c+1);
  /* each unit the piece carries at its place in the slot (the units' bytes in order), placed and counted once */
  uint64_t off=0;int placed=1;
  for(uint32_t u=0;u<units;u++){
    const uint64_t sp=(uint64_t)dest->span+u;
    const struct mesh_dma_span bytes=((const struct mesh_dma_span *)((const char *)m+t->spans))[sp];
    const uint64_t here=off<name->bytes?name->bytes-off:0,len=bytes.bytes<here?bytes.bytes:here;
    off+=bytes.bytes;
    if(atomic_load_explicit(bits+sp/64,memory_order_acquire)>>(sp%64)&1)continue;
    if(mesh_dma_span_valid(m,bytes) && bytes.bytes<=mesh_switch_piece(m))memcpy((char *)m+bytes.offset,data+off-bytes.bytes,len);
    atomic_thread_fence(memory_order_release);
    placed&=mesh_switch_count(m,p,1,table,dest->transfer,name->execution,sp)>=0;
  }
  return placed;
}

/* the word a destination here not yet deliverable waits on, and the value it must reach (mesh_switch_ready's first
   unmet condition: the step's issue, else its gate's first word short); 0 where it waits on nothing such (no
   program yet, or the window) */
static inline uint64_t mesh_switch_waits_on(const struct hdr *m,const struct mesh_declared *p,const struct mesh_switch_name *name,
                                           const struct mesh_switch_dest *dest,uint64_t *value){
  if(!p)return 0;   /* no program yet: the caller waits on `control` as it read it before looking (mesh_switch_forward) */
  const struct mesh_crossing *table=mesh_switch_table(m,p,1);
  if(!table || dest->transfer>=p->program.transfers[1])return 0;
  const struct mesh_crossing *t=table+dest->transfer;
  const uint64_t e=name->execution,step=e*p->program.period+t->step;
  if(atomic_load_explicit(&p->issued,memory_order_acquire)<=step){*value=step+1;return (uint64_t)((const char *)&p->issued-(const char *)m);}
  if(t->gate){
    const uint64_t add=e*t->gate_step;   /* a receive's gate: mesh_switch_ready at d 1 */
    if(!t->gates){
      const uint64_t v=atomic_load_explicit((const _Atomic uint64_t *)((const char *)m+t->gate),memory_order_acquire);
      if(v!=UINT64_MAX && v<t->gate_base+add){*value=t->gate_base+add;return t->gate;}
    }else{
      const struct mesh_gate_term *g=(const void *)((const char *)m+t->gate);
      for(uint32_t k=0;k<t->gates;k++){
        const uint64_t v=atomic_load_explicit((const _Atomic uint64_t *)((const char *)m+g[k].word),memory_order_acquire);
        if(v!=UINT64_MAX && v<g[k].threshold+add){*value=g[k].threshold+add;return g[k].word;}
      }
    }
  }
  /* past the transfer's window: its executions done (`complete`) must reach e - W + 1 */
  const struct mesh_switch_transfer *progress=mesh_switch_progress((struct hdr *)m,p,1)+dest->transfer;
  if(e>=atomic_load_explicit(&progress->complete,memory_order_acquire)+MESH_SWITCH_WINDOW){
    *value=e-MESH_SWITCH_WINDOW+1;return (uint64_t)((const char *)&progress->complete-(const char *)m);
  }
  return 0;
}

/* why a destination here was not placed: 0 no program of the key toward its source, 1 its table or bounds, 2 the
   program closed before it, 3 the step not issued, 4 its gate short, 5 past the transfer's window, 6 its cell not
   yet made free, 7 none of these (placed meanwhile) */
static inline uint32_t mesh_switch_unplaced(const struct hdr *m,const struct mesh_switch_name *name,const struct mesh_switch_dest *dest){
  struct mesh_node *x=mesh_switch_order((struct hdr *)m,name->source);
  const struct mesh_declared *p=NULL;
  for(uint32_t k=0;x && k<MESH_PROGRAMS && !p;k++)
    if(mesh_switch_owned(x->program+k) && x->program[k].program.key==name->key && x->program[k].progress[1])p=x->program+k;
  if(!p)return 0;
  const struct mesh_crossing *table=mesh_switch_table(m,p,1);
  if(!table || dest->transfer>=p->program.transfers[1])return 1;
  const struct mesh_crossing *t=table+dest->transfer;
  const uint64_t e=name->execution;
  if(e>=atomic_load_explicit(&p->closed,memory_order_acquire))return 2;
  if(atomic_load_explicit(&p->issued,memory_order_acquire)<=e*p->program.period+t->step)return 3;
  if(!mesh_switch_ready(m,p,t,1,e,dest->span))return 4;
  const struct mesh_switch_transfer *progress=mesh_switch_progress((struct hdr *)m,p,1)+dest->transfer;
  if(e>=atomic_load_explicit(&progress->complete,memory_order_acquire)+MESH_SWITCH_WINDOW)return 5;
  const struct mesh_switch_cell_head *c=mesh_switch_cell((struct hdr *)m,p,1,dest->transfer,e);
  if(atomic_load_explicit(&c->tag,memory_order_acquire)!=e+1)return 6;
  return 7;
}

/* whether a destination here can never be placed: its program, at the piece's source's node slot, was declared and has
   gone (its slot vacated with the key still in it, or given to another program since: `prior`), or it closed before the
   piece's execution.  A program not declared yet is not gone (a piece may land before its receiver declares): keys are
   never reused (nccl-mesh.c: the communicator's name and the process's declaration count), so a key seen gone stays gone */
static inline int mesh_switch_gone(const struct hdr *m,const struct mesh_switch_name *name){
  const struct mesh_node *x=mesh_switch_order((struct hdr *)m,name->source);
  if(!x)return 0;
  for(uint32_t k=0;k<MESH_PROGRAMS;k++){
    const struct mesh_declared *p=x->program+k;
    const uint64_t owner=atomic_load_explicit(&p->owner,memory_order_acquire);
    if(owner==UINT64_MAX)continue;
    if(p->prior==name->key)return 1;
    if(p->program.key!=name->key)continue;
    if(!owner || !p->progress[1])return 1;
    const uint64_t closed=atomic_load_explicit(&p->closed,memory_order_acquire);
    if(closed!=UINT64_MAX && name->execution>=closed)return 1;
  }
  return 0;
}

/* ---- The forward: a held slot's destinations here delivered, the rest switched onto lanes --------------------------
   Slot s holds a piece (staged here, or landed).  Its destinations at this node are delivered, the others split
   (mesh_switch_choose), every group in a copy of the slot but the last, which goes in the slot itself where nothing
   stays owed here; each pushed onto its lane's outbox, its name's path, lines and costs moved.  A destination that
   cannot go now (its delivery not ready; no free slot for its copy; its outbox full; no allowed neighbour) stays owed
   in the slot's name, which the caller retries.  Returns 1 where the slot is done with here (its hold released, or
   handed to a lane), 0 where it is still held with destinations owed.  `ring` the caller's log ring, `now` its clock */
static inline int mesh_switch_forward(struct hdr *m,struct mesh_switch *sw,uint32_t s,uint64_t now,uint32_t ring){
  unsigned char *at=mesh_switch_slot_at(m,sw,s);
  struct mesh_switch_name *name=(struct mesh_switch_name *)at;
  const uint32_t self=m->node;
  /* a slot held for its client's word: one load, until the word reaches its value */
  if(sw->slot[s].wait_word){
    const struct mesh_declared *wp=sw->slot[s].wait_program?(const struct mesh_declared *)((const char *)m+sw->slot[s].wait_program):NULL;
    if(atomic_load_explicit((const _Atomic uint64_t *)((const char *)m+sw->slot[s].wait_word),memory_order_acquire)<sw->slot[s].wait_value &&
       (!wp || (mesh_switch_owned(wp) && wp->program.key==name->key)))return 0;
    sw->slot[s].wait_word=0;sw->slot[s].wait_program=0;
  }
  const int logging=atomic_load_explicit(&sw->logging,memory_order_relaxed);
  struct mesh_switch_dest kept[MESH_SWITCH_N],away[MESH_SWITCH_N];uint32_t nkept=0,owed=0;
  uint64_t waits=0,wait_value=0,wait_program=0;
  /* `control` read before any program is looked for, so a declaration after the look moves it past this value */
  const uint64_t control=atomic_load_explicit(&m->control,memory_order_acquire);
  for(uint32_t i=0;i<name->dests && i<MESH_SWITCH_N;i++){
    if(name->dest[i].node!=self){away[owed++]=name->dest[i];continue;}
    if(name->kind==MESH_PIECE_CREDIT){mesh_switch_credit_take(m,name);MESH_SWITCH_COUNT(sw,ring,credits_taken,1);continue;}
    struct mesh_declared *found=NULL;
    if(mesh_switch_deliver(m,name,name->dest+i,at+MESH_SWITCH_HEAD,&found)){
      MESH_SWITCH_COUNT(sw,ring,delivered,1);
      MESH_SWITCH_COUNT(sw,ring,deliver_wait_ns,now>sw->slot[s].landed?now-sw->slot[s].landed:0);
      MESH_SWITCH_COUNT(sw,ring,hops[name->hops<3?name->hops:3],1);
      if(logging){
        struct mesh_switch_record r;memset(&r,0,sizeof r);
        r.kind=MESH_LOG_PIECE;r.node=self;r.key=name->key;r.execution=name->execution;r.at=now;
        r.latency=name->born && now>name->born?(float)(now-name->born):name->queued+name->transit;r.dest=self;r.context=UINT32_MAX;
        r.queueing=name->queued;r.floor=name->floor;r.source=name->source;r.transfer=name->dest[i].transfer;r.bytes=name->bytes;
        /* a record a unit the piece carried */
        for(uint32_t u=0;u<(name->dest[i].units?name->dest[i].units:1u);u++){r.span=name->dest[i].span+u;mesh_switch_log_put(sw,ring,&r);}
      }
    }else if(mesh_switch_gone(m,name)){
      MESH_SWITCH_COUNT(sw,ring,orphans,1);
    }else{
      kept[nkept++]=name->dest[i];
      if(found)waits=mesh_switch_waits_on(m,found,name,name->dest+i,&wait_value);
      else{waits=(uint64_t)((const char *)&m->control-(const char *)m);wait_value=control+1;}
      wait_program=found?(uint64_t)((const char *)found-(const char *)m):0;
      if(!waits)MESH_SWITCH_COUNT(sw,ring,unplaced[mesh_switch_unplaced(m,name,name->dest+i)],1);
    }
  }
  const struct mesh_switch_name sending=*name;
  uint32_t groups=0,lane[MESH_SWITCH_N];uint64_t group[MESH_SWITCH_N];
  if(owed){
    struct mesh_switch_name split=sending;split.dests=owed;memcpy(split.dest,away,owed*sizeof *away);
    struct route_draw draw;float logp=0;uint32_t context=UINT32_MAX;uint64_t point=0;
    /* the draw over the view and the weights where they lie, again where either was written meanwhile (seqlock): no
       copy of either a piece */
    for(;;){
      const uint64_t a=atomic_load_explicit(&sw->view.serial,memory_order_acquire),b=atomic_load_explicit(&sw->weights.serial,memory_order_acquire);
      if((a|b)&1)continue;
      groups=mesh_switch_choose(sw,&sw->view,&sw->weights,&split,sw->slot[s].landed,lane,group,logging?&draw:NULL,&logp,&context,&point);
      atomic_thread_fence(memory_order_acquire);
      if(atomic_load_explicit(&sw->view.serial,memory_order_relaxed)==a && atomic_load_explicit(&sw->weights.serial,memory_order_relaxed)==b)break;
    }
    if(groups==UINT32_MAX){groups=0;for(uint32_t i=0;i<owed;i++)kept[nkept++]=away[i];}
    else if(logging){
      struct mesh_switch_record r;memset(&r,0,sizeof r);
      r.kind=MESH_LOG_DRAW;r.node=self;r.key=name->key;r.execution=name->execution;r.at=now;r.draw=draw;r.logp=logp;
      r.source=name->source;r.transfer=away[0].transfer;r.span=away[0].span;r.bytes=name->bytes;
      r.context=context;r.point=(uint32_t)point;r.dest=away[0].node;
      mesh_switch_log_put(sw,ring,&r);
    }
  }
  struct mesh_switch_line lines[MESH_SWITCH_LINES];uint32_t nlines=0;
  if(groups)nlines=mesh_switch_lines_give(sw,lines);
  int handed=0;
  MESH_SWITCH_COUNT(sw,ring,forwards,1);MESH_SWITCH_COUNT(sw,ring,groups,groups);
  for(uint32_t g=groups;g-->0;){
    /* the last group (g 0) goes in the slot itself where no destination stays owed here */
    const int own=!g && !nkept;
    const uint32_t target=own?s:mesh_switch_take(sw);
    if(target==UINT32_MAX){for(uint32_t i=0;i<owed;i++)if(group[g]>>i&1)kept[nkept++]=away[i];continue;}
    unsigned char *to=mesh_switch_slot_at(m,sw,target);
    /* the piece's bytes alone (the name says how many; the frame's rest is never read) */
    if(!own){memcpy(to+MESH_SWITCH_HEAD,at+MESH_SWITCH_HEAD,sending.bytes<sw->slot_bytes-MESH_SWITCH_HEAD?sending.bytes:sw->slot_bytes-MESH_SWITCH_HEAD);
             MESH_SWITCH_COUNT(sw,ring,copies,1);}
    struct mesh_switch_name *n=(struct mesh_switch_name *)to;
    *n=sending;n->dests=0;
    for(uint32_t i=0;i<owed;i++)if(group[g]>>i&1)n->dest[n->dests++]=away[i];
    n->from=self;
    const uint32_t peer=atomic_load_explicit(&sw->lane[lane[g]].peer,memory_order_relaxed);
    if(n->hops+1<MESH_SWITCH_N)n->path[n->hops+1]=peer;
    n->hops++;
    float rate,waited;mesh_switch_unpack(atomic_load_explicit(&sw->lane[lane[g]].estimate,memory_order_relaxed),&rate,&waited);
    const double each=rate>0?rate:1;
    n->queued+=(float)((double)atomic_load_explicit(&sw->lane[lane[g]].queued,memory_order_relaxed)/each)+(float)(now>sw->slot[s].landed?now-sw->slot[s].landed:0);
    const uint64_t wire=mesh_switch_lane_bytes(sw,lane[g]);
    n->transit+=(float)((double)wire/each);
    n->lines=nlines;memcpy(n->line,lines,nlines*sizeof *lines);
    atomic_store_explicit(&sw->slot[target].refs,1,memory_order_relaxed);
    atomic_store_explicit(&sw->slot[target].state,SLOT_SENDING,memory_order_relaxed);
    sw->slot[target].posted=now;sw->slot[target].lane=lane[g];
    if(!own)sw->slot[target].landed=sw->slot[s].landed;
    atomic_fetch_add_explicit(&sw->lane[lane[g]].queued,wire,memory_order_relaxed);
    if(mesh_switch_push(sw,lane[g],target)){handed|=own;continue;}
    atomic_fetch_sub_explicit(&sw->lane[lane[g]].queued,wire,memory_order_relaxed);
    for(uint32_t i=0;i<owed;i++)if(group[g]>>i&1)kept[nkept++]=away[i];
    if(!own)mesh_switch_give(sw,target);else atomic_store_explicit(&sw->slot[s].state,SLOT_HELD,memory_order_relaxed);
  }
  if(handed)return 1;
  if(!nkept){mesh_switch_release(sw,s);return 1;}
  /* still held: the name as it came, its owed destinations alone; where its one destination left is here, waiting on
     its client's word, the word kept for the retries */
  MESH_SWITCH_COUNT(sw,ring,kept,1);
  if(nkept==1 && kept[0].node==self && waits){sw->slot[s].wait_word=waits;sw->slot[s].wait_value=wait_value;sw->slot[s].wait_program=wait_program;}
  *name=sending;name->dests=nkept;memcpy(name->dest,kept,nkept*sizeof *kept);
  return 0;
}
/* a receive landed in slot s on lane `lane`: its name's lines taken into the view, then forwarded (mesh_switch_forward) */
static inline int mesh_switch_landed(struct hdr *m,struct mesh_switch *sw,uint32_t lane,uint32_t s,uint64_t now,uint32_t ring){
  struct mesh_switch_name *name=(struct mesh_switch_name *)mesh_switch_slot_at(m,sw,s);
  atomic_fetch_add_explicit(&sw->lane[lane].landed,1,memory_order_relaxed);
  MESH_SWITCH_COUNT(sw,ring,landed,1);
  sw->slot[s].landed=now;sw->slot[s].wait_word=0;sw->slot[s].wait_program=0;
  atomic_store_explicit(&sw->slot[s].refs,1,memory_order_relaxed);
  atomic_store_explicit(&sw->slot[s].state,SLOT_HELD,memory_order_relaxed);
  if(name->magic!=MESH_SWITCH_MAGIC){mesh_switch_release(sw,s);return 1;}
  mesh_switch_lines_take(sw,name);
  return mesh_switch_forward(m,sw,s,now,ring);
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
   - the fabric is the view (its links route.h's lanes, view indices its nodes, the ranks its compute nodes);
   - each lane's wait: this node's own lanes' least queue over their rate, a remote link's its line's; duplex share 1
     (a bridge sees no other node's reverse traffic);
   - the cost-to-go the last refresh's (the weights'), the bytes in flight and posted this node's own lanes' queues;
   - per pattern, from this node's programs: what is left to stage and its bytes (as a source), what each destination
     has received (as one), the patterns' live bytes, and the programs live (distinct keys).
   Every token is route.h's route_lane_token / route_context_token over that state. */
static inline void mesh_switch_tokens_make(struct hdr *m,struct mesh_switch *sw,uint64_t now,uint64_t point,struct mesh_switch_tokens *out){
  struct mesh_switch_view v;mesh_switch_view_read(sw,&v);
  struct mesh_switch_weights w;mesh_switch_weights_read(sw,&w);
  const uint32_t self=mesh_switch_index(&v,v.self),ranks=mesh_switch_ranks(&v);
  const double page=(double)mesh_switch_piece(m);
  static __thread struct route_stream s;static __thread struct route_state st;
  memset(&s,0,sizeof s);memset(&st,0,sizeof st);
  s.f.nodes=(int)(v.nodes<ROUTE_N?v.nodes:ROUTE_N);s.f.lanes=(int)(v.links<ROUTE_L?v.links:ROUTE_L);s.f.duplex=1;
  for(int l=0;l<s.f.lanes;l++){
    const struct mesh_switch_link *x=v.link+l;
    s.f.from[l]=(int)mesh_switch_index(&v,x->from);s.f.to[l]=(int)mesh_switch_index(&v,x->to);s.f.cables[l]=(int)x->cables;s.f.rate[l]=x->rate;s.f.latency[l]=x->latency;
  }
  for(uint32_t a=0;a<v.nodes && a<ROUTE_N;a++)if(ranks>>a&1)s.f.compute[s.f.ncompute++]=(int)a;
  s.unit=mesh_switch_unit(&v,page);
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
  /* this node's programs: as a source (left to stage, waiting, live bytes), as a destination (received) */
  double total[ROUTE_J]={0},unstaged[ROUTE_J]={0},rtotal[ROUTE_J]={0},rgot[ROUTE_J]={0};
  uint64_t keys[MESH_NODES*MESH_PROGRAMS];uint32_t nkeys=0;
  for(uint32_t k=0;k<MESH_NODES;k++){
    struct mesh_node *x=mesh_node_at(m,k);
    const uint32_t peer=mesh_switch_index(&v,atomic_load_explicit(&x->node,memory_order_acquire));
    if(peer>=v.nodes || peer>=ROUTE_N)continue;
    for(uint32_t q=0;q<MESH_PROGRAMS;q++){
      struct mesh_declared *p=x->program+q;
      if(!mesh_switch_owned(p) || mesh_switch_done(m,p))continue;
      uint32_t seen=0;for(uint32_t i=0;i<nkeys;i++)seen|=keys[i]==p->program.key;
      if(!seen && nkeys<MESH_NODES*MESH_PROGRAMS)keys[nkeys++]=p->program.key;
      const uint64_t issued=atomic_load_explicit(&p->issued,memory_order_acquire),closed=atomic_load_explicit(&p->closed,memory_order_acquire);
      const uint64_t runs=issued<closed?issued:closed,period=p->program.period?p->program.period:1,execs=(runs+period-1)/period;
      if(p->program.pieces[0] && self<ROUTE_N){
        /* the unicast pattern self -> peer, and the multicast self -> every other rank where the key goes to several */
        uint32_t peers=0;
        for(uint32_t k2=0;k2<MESH_NODES;k2++){struct mesh_node *y=mesh_node_at(m,k2);for(uint32_t q2=0;q2<MESH_PROGRAMS;q2++){const struct mesh_declared *r=y->program+q2;
          if(mesh_switch_owned(r) && r->program.key==p->program.key && r->program.pieces[0])peers++;}}
        const double all=(double)p->program.pieces[0]*(double)(execs?execs:1),claimed=(double)atomic_load_explicit(&p->claimed,memory_order_relaxed);
        for(int j=0;j<s.patterns;j++){
          if(s.source[j]!=(int)self)continue;
          if(!(s.dest[j]==1u<<peer || (peers>1 && __builtin_popcount(s.dest[j])>1)))continue;
          total[j]+=all;unstaged[j]+=all>claimed?all-claimed:0;
        }
      }
      if(p->program.pieces[1] && p->progress[1]){
        const struct mesh_switch_transfer *x1=mesh_switch_progress(m,p,1);
        double got=0;const double want=(double)p->program.transfers[1]*(double)(closed!=UINT64_MAX?closed:(execs?execs:1));
        for(uint64_t tr=0;tr<p->program.transfers[1];tr++)got+=(double)atomic_load_explicit(&x1[tr].complete,memory_order_relaxed);
        for(int j=0;j<s.patterns;j++)if(s.source[j]==(int)peer && self<ROUTE_N && s.dest[j]>>self&1){rtotal[j]+=want;rgot[j]+=got;}
      }
    }
  }
  st.ops=nkeys;
  for(int j=0;j<s.patterns;j++){
    st.left[j]=total[j]>0?unstaged[j]/total[j]:0;st.pending[j]=unstaged[j]*page;st.demand[j]=total[j]*page;
    if(self<ROUTE_N)st.recv[j][self]=rtotal[j]>0?rgot[j]/rtotal[j]:0;
  }
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
static inline int mesh_switch_policy_refresh(struct hdr *m,struct mesh_switch *sw,const struct mesh_switch_policy *policy,uint64_t now,uint64_t point,int argmax,
                                             struct mesh_switch_tokens *scratch){
  if(!policy || !policy->splits)return -1;
  mesh_switch_tokens_make(m,sw,now,point,scratch);
  static __thread float w[MESH_SWITCH_CONTEXTS][SPLIT_FS];
  if(policy->splits(policy->context,scratch,w))return -1;
  mesh_switch_splits_publish(sw,scratch->key,scratch->contexts,(const float (*)[SPLIT_FS])w,point,scratch->unit,argmax);
  return 0;
}

#endif
