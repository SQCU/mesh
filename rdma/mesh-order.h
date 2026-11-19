/* design/node-addressed-transfers.md: the shared order of a node pair, the bridge's alone.  Every function here is a
   pure function of the region's rings and epochs, or a walk of them, so the two ends of a cable compute the same SEND
   and receive sequence on each of its queue pairs without telling each other anything.  The bridge's port children
   (mesh-link.c) and the two-node simulator (tests/sim-wire.c) are its callers; no client includes it */
#ifndef MESH_ORDER_H
#define MESH_ORDER_H
#include "mesh.h"
#include <errno.h>
#include <string.h>

/* A piece: the bridge cuts each span of a transfer into pieces of at most one allocation block (the region's `block`
   pages of `pgsz`, equal at both ends of every cable: port_heard), the last one the span's rest.  Both ends cut spans of
   equal lengths alike, so piece n of a direction's order has the same length at both ends */
static inline uint64_t order_piece_bytes(const struct hdr *m){ return (uint64_t)m->block*m->pgsz; }

/* the epoch of a direction's order holding piece n: the latest of the generation whose first piece is at most n; NULL
   where none is (no epoch yet) */
static inline const struct mesh_epoch *order_epoch(const struct mesh_node *x,uint32_t epochs,int way,uint64_t n){
  for(uint32_t e=epochs;e-- && epochs-e<=MESH_EPOCHS;){
    const struct mesh_epoch *at=x->epoch+e%MESH_EPOCHS;
    if(at->first[way]<=n)return at;
  }
  return NULL;
}
/* The placement (the shared order's one rule): piece n of the k-th transfer of a direction, in epoch `e`, goes on
   cable (n - first) mod cables, on that cable's queue pair k mod lanes, in the order of the pieces placed there.
   Consecutive pieces of a transfer cross every cable (its bytes take every cable's bandwidth), consecutive transfers
   take a cable's queue pairs in turn (a transfer held by its gate holds only the queue pair its pieces are on, not
   the next transfer's) */
struct order_place { uint32_t cable,lane; };
static inline struct order_place order_place(const struct mesh_epoch *e,int way,uint64_t n,uint64_t k){
  const uint32_t c=(uint32_t)((n-e->first[way])%e->cables);
  return (struct order_place){c,(uint32_t)(k%e->cable[c].lanes)};
}
/* a direction's first piece of the next epoch: past every piece either end has posted (its frontier each way, as each
   end froze its posting), the pieces between still placed by the epoch they began in */
static inline uint64_t order_boundary(uint64_t sender_frontier,uint64_t receiver_frontier){
  return sender_frontier>receiver_frontier?sender_frontier:receiver_frontier;
}

/* the word at region offset `at` raised to `value` where it is lower (release) */
static inline void order_raise(void *m,uint64_t at,uint64_t value){
  if(!at)return;
  _Atomic uint64_t *word=(_Atomic uint64_t *)((char *)m+at);
  uint64_t old=atomic_load_explicit(word,memory_order_relaxed);
  while(old<value && !atomic_compare_exchange_weak_explicit(word,&old,value,memory_order_release,memory_order_relaxed)){}
}
/* a transfer's pieces (each span's, mesh-order.h order_piece_bytes); 0 where its table is not the region's */
static inline uint64_t order_count(const struct hdr *m,const struct mesh_crossing *t){
  if(!mesh_crossing_valid(m,t))return 0;
  const struct mesh_dma_span *spans=(const struct mesh_dma_span *)((const char *)m+t->spans);
  const uint64_t piece=order_piece_bytes(m);
  uint64_t n=0;
  for(uint64_t i=0;i<t->units;i++)n+=(spans[i].bytes+piece-1)/piece;
  return n;
}
/* slot s's piece count, counted once (its `pieces`, the count + 1) */
static inline uint64_t order_pieces(struct hdr *m,struct mesh_slot *s,const struct mesh_crossing *t){
  uint64_t p=atomic_load_explicit(&s->pieces,memory_order_acquire);
  if(p)return p-1;
  const uint64_t n=order_count(m,t);
  uint64_t zero=0;
  atomic_compare_exchange_strong_explicit(&s->pieces,&zero,n+1,memory_order_acq_rel,memory_order_acquire);
  return atomic_load_explicit(&s->pieces,memory_order_acquire)-1;
}
/* whether the client `owner` still holds its slot (a transfer of a client gone is carried without its gates: its
   rows stay held until its transfers retire, mesh-dataflow.c mesh_pinned) */
static inline int order_owned(struct hdr *m,uint64_t owner){
  return owner && atomic_load_explicit(&mesh_client_at(m,owner)->owner,memory_order_acquire)==owner;
}
/* whether a transfer's gate holds for its span s (`way` 0: a SEND of span s; 1: its receives) */
static inline int order_gate(struct hdr *m,const struct mesh_crossing *t,int way,uint64_t s){
  if(!t->gate)return 1;
  const char *base=(const char *)m;
  if(t->gates){
    const struct mesh_gate *g=(const struct mesh_gate *)(base+t->gate);
    for(uint32_t i=0;i<t->gates;i++){
      if(!mesh_region_holds(m,g[i].word,8,8))continue;
      if(atomic_load_explicit((const _Atomic uint64_t *)(base+g[i].word),memory_order_acquire)<g[i].threshold)return 0;
    }
    return 1;
  }
  const uint64_t value=atomic_load_explicit((const _Atomic uint64_t *)(base+t->gate),memory_order_acquire);
  return way?value>=t->base:value>=t->base+s+1;
}

/* The publication of done words, by whichever process completed a transfer's last piece: from the oldest transfer
   not retired, each complete one whose done word no earlier incomplete one names raised to its value (exactly once:
   its `state`), so a word several transfers count on rises as a prefix.  A transfer of another generation is lost */
#define ORDER_WINDOW 256u
static inline void order_publish(struct hdr *m,struct mesh_node *x,int way){
  const uint64_t generation=atomic_load_explicit(&x->generation,memory_order_acquire);
  const uint64_t first=atomic_load_explicit(&x->retired[way],memory_order_acquire);
  uint64_t blocked[32];uint32_t held=0;
  for(uint64_t i=first;i<first+ORDER_WINDOW;i++){
    struct mesh_slot *s=mesh_ring(m,x,way,i);
    if(atomic_load_explicit(&s->sequence,memory_order_acquire)!=2*i+1)break;
    if(atomic_load_explicit(&s->state,memory_order_acquire))continue;
    const struct mesh_crossing t=s->t;
    const uint64_t pieces=atomic_load_explicit(&s->pieces,memory_order_acquire);
    const int lost=t.generation!=generation;
    const int complete=lost || (pieces && atomic_load_explicit(&s->completed,memory_order_acquire)+1>=pieces);
    int behind=0;
    for(uint32_t b=0;b<held && t.done;b++)behind|=blocked[b]==t.done;
    if(complete && (!behind || lost)){
      uint64_t open=0;
      if(atomic_compare_exchange_strong_explicit(&s->state,&open,1,memory_order_acq_rel,memory_order_acquire))
        order_raise(m,t.done,lost || pieces<2?UINT64_MAX:t.value);
    }
    else if(t.done && !behind){blocked[held++]=t.done;if(held==sizeof blocked/sizeof *blocked)break;}
  }
}
/* a piece of slot index i completed: its count, and the publication where it was the last */
static inline void order_completed(struct hdr *m,struct mesh_node *x,int way,uint64_t i){
  struct mesh_slot *s=mesh_ring(m,x,way,i);
  const uint64_t done=atomic_fetch_add_explicit(&s->completed,1,memory_order_acq_rel)+1;
  if(done+1>=atomic_load_explicit(&s->pieces,memory_order_acquire))order_publish(m,x,way);
}

/* A walker: one queue pair's pieces of one direction, in the order's order.  It reads every transfer from the anchor
   on (a walker that finds its next transfer already retired, one none of whose pieces were its own, starts again from
   the anchor), counting transfers (k) and pieces (n), and posts the pieces the placement puts on its own cable and queue pair:
   `port` (the bridge's port of the cable) and `lane` (its queue pair's index among the cable's).  `posted` is one past
   the last piece it posted (its frontier) */
struct order_walker { uint64_t generation,at,k,n,pieces,piece,span,offset,posted; int started; };
/* a post of `span` (invalid: the bridge's own block, `dud`) on the walker's queue pair, for slot index `slot`: 1 posted,
   0 no room (the walk stops, to resume at this piece) */
typedef int (*order_post)(void *context,int way,uint32_t lane,struct mesh_dma_span span,int dud,uint64_t slot);
/* the walker started from the order's anchor (its transfers and pieces before `retired`, read under its seqlock) */
static inline int order_anchor(struct mesh_node *x,int way,struct order_walker *w){
  for(int tries=0;tries<1000;tries++){
    const uint64_t before=atomic_load_explicit(&x->anchored,memory_order_acquire);
    if(before&1)continue;
    const uint64_t generation=atomic_load_explicit(&x->generation,memory_order_acquire);
    const uint64_t at=atomic_load_explicit(&x->retired[way],memory_order_acquire);
    const uint64_t k=atomic_load_explicit(&x->anchor[way][0],memory_order_acquire),n=atomic_load_explicit(&x->anchor[way][1],memory_order_acquire);
    if(atomic_load_explicit(&x->anchored,memory_order_acquire)!=before)continue;
    *w=(struct order_walker){.generation=generation,.at=at,.k=k,.n=n,.started=1};
    return 1;
  }
  return 0;
}
/* the walker's pieces posted as far as their gates, the queue pair's room and the order's appended transfers go;
   none while the order is frozen (an odd `freeze`: the bridges agree the next epoch) or has no epoch */
static inline void order_walk(struct hdr *m,struct mesh_node *x,int way,uint32_t port,uint32_t lane,struct order_walker *w,
                              order_post post,void *context){
  const uint64_t generation=atomic_load_explicit(&x->generation,memory_order_acquire);
  if(!w->started || w->generation!=generation){if(!order_anchor(x,way,w))return;}
  const uint64_t piece_bytes=order_piece_bytes(m);
  for(;;){
    if(atomic_load_explicit(&x->freeze,memory_order_acquire)&1)return;
    const uint32_t epochs=atomic_load_explicit(&x->epochs,memory_order_acquire);
    struct mesh_slot *s=mesh_ring(m,x,way,w->at);
    const uint64_t sequence=atomic_load_explicit(&s->sequence,memory_order_acquire);
    /* a transfer retired before this walker reached it held none of its pieces (each of them posted here before it
       was done), so the walker goes on from the anchor, past every retired one */
    if(sequence>2*w->at+1){
      const uint64_t posted=w->posted;
      if(!order_anchor(x,way,w))return;
      w->posted=posted;continue;
    }
    if(sequence!=2*w->at+1)return;
    const struct mesh_crossing t=s->t;
    if(t.generation!=generation){
      uint64_t open=0;
      if(atomic_compare_exchange_strong_explicit(&s->state,&open,1,memory_order_acq_rel,memory_order_acquire))order_raise(m,t.done,UINT64_MAX);
      w->at++;continue;
    }
    if(!w->pieces)w->pieces=order_pieces(m,s,&t)+1;
    if(w->piece+1>=w->pieces){
      if(w->pieces==1)order_publish(m,x,way);
      w->at++;w->k++;w->n+=w->pieces-1;w->pieces=w->piece=w->span=w->offset=0;continue;
    }
    const uint64_t n=w->n+w->piece;
    const struct mesh_epoch *e=order_epoch(x,epochs,way,n);
    if(!e || !e->cables)return;
    const struct mesh_dma_span span=((const struct mesh_dma_span *)((const char *)m+t.spans))[w->span];
    const uint64_t length=span.bytes-w->offset<piece_bytes?span.bytes-w->offset:piece_bytes;
    const struct order_place place=order_place(e,way,n,w->k);
    if(e->cable[place.cable].port==port && place.lane==lane){
      if(order_owned(m,atomic_load_explicit(&s->owner,memory_order_acquire)) && !order_gate(m,&t,way,w->span))return;
      const struct mesh_dma_span bytes={span.offset+w->offset,length};
      const int dud=!mesh_dma_span_valid(m,bytes);
      if(!post(context,way,lane,bytes,dud,w->at))return;
      if(dud)order_raise(m,t.done,UINT64_MAX);
      w->posted=n+1;
    }
    w->offset+=length;w->piece++;
    if(w->offset==span.bytes){w->span++;w->offset=0;}
  }
}

/* The parent's retirement of a direction's published transfers, in order: each slot freed for its next lap and the
   anchor moved past it (its transfers and pieces, a transfer of another generation counting neither); a transfer of
   another generation that no walker reached is lost here */
static inline void order_retire_held(struct hdr *m,struct mesh_node *x,int way);
/* the retirement by any of the bridge's processes (the parent at its turn, each port child each pass): one at a time,
   by `retiring` (one that finds it held leaves the retirement to its holder's next pass) */
static inline void order_retire(struct hdr *m,struct mesh_node *x,int way){
  uint32_t open=0;
  if(!atomic_compare_exchange_strong_explicit(&x->retiring,&open,1,memory_order_acquire,memory_order_relaxed))return;
  order_retire_held(m,x,way);
  atomic_store_explicit(&x->retiring,0,memory_order_release);
}
static inline void order_retire_held(struct hdr *m,struct mesh_node *x,int way){
  const uint64_t generation=atomic_load_explicit(&x->generation,memory_order_acquire);
  uint64_t r=atomic_load_explicit(&x->retired[way],memory_order_acquire),k=atomic_load_explicit(&x->anchor[way][0],memory_order_relaxed),
           n=atomic_load_explicit(&x->anchor[way][1],memory_order_relaxed);
  const uint64_t from=r;
  for(;;){
    struct mesh_slot *s=mesh_ring(m,x,way,r);
    if(atomic_load_explicit(&s->sequence,memory_order_acquire)!=2*r+1)break;
    const struct mesh_crossing t=s->t;
    if(t.generation!=generation){
      uint64_t open=0;
      if(atomic_compare_exchange_strong_explicit(&s->state,&open,1,memory_order_acq_rel,memory_order_acquire))order_raise(m,t.done,UINT64_MAX);
    }
    else if(!atomic_load_explicit(&s->state,memory_order_acquire))break;
    else{k++;n+=atomic_load_explicit(&s->pieces,memory_order_acquire)-1;}
    r++;
  }
  if(r==from)return;
  atomic_fetch_add_explicit(&x->anchored,1,memory_order_acq_rel);
  atomic_store_explicit(&x->anchor[way][0],k,memory_order_release);atomic_store_explicit(&x->anchor[way][1],n,memory_order_release);
  atomic_store_explicit(&x->retired[way],r,memory_order_release);
  atomic_fetch_add_explicit(&x->anchored,1,memory_order_acq_rel);
  for(uint64_t i=from;i<r;i++){
    struct mesh_slot *s=mesh_ring(m,x,way,i);
    atomic_store_explicit(&s->owner,0,memory_order_relaxed);
    atomic_store_explicit(&s->state,0,memory_order_relaxed);atomic_store_explicit(&s->pieces,0,memory_order_relaxed);
    atomic_store_explicit(&s->completed,0,memory_order_relaxed);
    atomic_store_explicit(&s->sequence,2*(i+MESH_RING),memory_order_release);
  }
}
/* a ring made new: every slot free for its first lap, nothing taken or retired */
static inline void order_rings_new(struct hdr *m,struct mesh_node *x){
  for(int way=0;way<2;way++){
    for(uint64_t i=0;i<MESH_RING;i++){
      struct mesh_slot *s=mesh_ring(m,x,way,i);
      memset(s,0,sizeof *s);
      atomic_store_explicit(&s->sequence,2*i,memory_order_relaxed);
    }
    atomic_store(&x->tail[way],0);atomic_store(&x->retired[way],0);
    atomic_store(&x->anchor[way][0],0);atomic_store(&x->anchor[way][1],0);
  }
  atomic_thread_fence(memory_order_release);
}
/* The order restarted (a cable toward the node lost, the order's epochs exhausted, the region adopted): its
   generation moved, so every transfer not yet done is lost (its done word ~0) and is in no order again; the anchor at
   the first not retired, counting from 0; no epoch until the two bridges agree one (mesh-link.c order_turn) */
static inline void order_restart(struct hdr *m,struct mesh_node *x){
  atomic_store_explicit(&x->live,0,memory_order_seq_cst);
  atomic_fetch_add_explicit(&x->generation,1,memory_order_seq_cst);
  atomic_store_explicit(&x->epochs,0,memory_order_seq_cst);
  atomic_store_explicit(&x->freeze,0,memory_order_seq_cst);
  atomic_store_explicit(&x->retiring,0,memory_order_seq_cst);  /* a holder that died with it held: its link is lost */
  const uint64_t anchored=atomic_load(&x->anchored)|1;  /* odd while written, whatever a holder that died left */
  atomic_store_explicit(&x->anchored,anchored,memory_order_seq_cst);
  for(int way=0;way<2;way++){atomic_store(&x->anchor[way][0],0);atomic_store(&x->anchor[way][1],0);}
  atomic_store_explicit(&x->anchored,anchored+1,memory_order_seq_cst);
  for(int way=0;way<2;way++){
    const uint64_t tail=atomic_load(&x->tail[way]);
    for(uint64_t i=atomic_load(&x->retired[way]);i<tail;i++){
      struct mesh_slot *s=mesh_ring(m,x,way,i);
      if(atomic_load(&s->sequence)!=2*i+1)continue;
      uint64_t open=0;
      if(atomic_compare_exchange_strong(&s->state,&open,1))order_raise(m,s->t.done,UINT64_MAX);
    }
  }
}
/* the next epoch appended: from `first` each way, over `cables` (the cables of both ends' agreed set, sorted by
   identity); 0, or ENOSPC where the generation holds MESH_EPOCHS already (the caller restarts the order) */
static inline int order_extend(struct mesh_node *x,const uint64_t first[2],uint32_t cables,const uint32_t *port,
                               const uint32_t *lanes,const uint64_t *wire){
  const uint32_t e=atomic_load_explicit(&x->epochs,memory_order_acquire);
  if(e>=MESH_EPOCHS)return ENOSPC;
  struct mesh_epoch *at=x->epoch+e%MESH_EPOCHS;
  *at=(struct mesh_epoch){.first={first[0],first[1]},.cables=cables};
  for(uint32_t c=0;c<cables && c<MESH_PORTS;c++){at->cable[c].port=port[c];at->cable[c].lanes=lanes[c];at->cable[c].wire=wire[c];}
  atomic_store_explicit(&x->epochs,e+1,memory_order_release);
  atomic_store_explicit(&x->live,1,memory_order_release);
  return 0;
}
#endif
