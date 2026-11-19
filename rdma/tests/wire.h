/* design/declared-programs.md "Testing": two nodes' bridges over a simulated wire, in one process, running the bridge's
   own two steps (mesh-order.h order_next, order_raise_pass) on each node's region.  The wire enforces what the hardware
   does: a receive names its bytes before the SEND that fills it; each queue pair's SENDs fill its posted receives
   strictly in post order; a SEND waits (the transport's credit) until a receive is posted; and a SEND and its receive are
   of equal length.  It checks more than lengths: each piece carries what the poster took it to be (its program's key,
   execution, transfer and span), and the two ends of every queue pair must agree on each one (any mismatch ends the
   process: the two ends' placements differed).  tests/order.c and tests/sim-wire.c include it */
#ifndef WIRE_H
#define WIRE_H
#include "../mesh-order.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { WIRE_CABLES=8, WIRE_LANES=8 };
struct wire_piece { struct mesh_dma_span span; uint64_t key,execution,transfer,span_index,piece; };
struct wire_fifo { struct wire_piece *sends,*receives; uint64_t sent_head,sent_tail,received_head,received_tail; };
struct wire_node {
  struct hdr *m; struct mesh_node *x;
  uint64_t posted[2][WIRE_CABLES][WIRE_LANES],done[2][WIRE_CABLES][WIRE_LANES];
  uint64_t hint[2][WIRE_CABLES][WIRE_LANES];
  struct order_raiser raiser[WIRE_CABLES][2];
};
struct wire {
  struct wire_node node[2];
  /* fifo[d][c][j]: node d's SENDs on cable c queue pair j, and node 1-d's receives there */
  struct wire_fifo fifo[2][WIRE_CABLES][WIRE_LANES];
  uint32_t cables,lanes,room;
  uint64_t matched,bytes,random,burst;   /* burst: the most matches a queue pair makes a turn (0: 1 to 8 at random) */
};
static inline void wire_make(struct wire *w,uint32_t cables,uint32_t lanes,uint32_t room){
  w->cables=cables;w->lanes=lanes;w->room=room;w->random=0x9e3779b97f4a7c15u;
  for(int d=0;d<2;d++)for(uint32_t c=0;c<cables;c++)for(uint32_t j=0;j<lanes;j++){
    w->fifo[d][c][j].sends=calloc(room,sizeof(struct wire_piece));w->fifo[d][c][j].receives=calloc(room,sizeof(struct wire_piece));
  }
}
static inline uint64_t wire_next_random(struct wire *w){ w->random^=w->random<<13;w->random^=w->random>>7;w->random^=w->random<<17;return w->random; }
/* node n's posts on cable c (port c) queue pair j each way, as far as the bridge's step lets them and the queue holds */
static inline void wire_post(struct wire *w,uint32_t n,uint32_t c,uint32_t j){
  struct wire_node *v=w->node+n;
  const uint32_t epochs=atomic_load_explicit(&v->x->epochs,memory_order_acquire);
  const uint64_t segments=atomic_load_explicit(&v->x->segments,memory_order_acquire);
  for(int way=0;way<2;way++){
    struct wire_fifo *f=&w->fifo[way?1-n:n][c][j];
    for(;;){
      uint64_t *tail=way?&f->received_tail:&f->sent_tail;
      const uint64_t head=way?f->received_head:f->sent_head;
      if(*tail-head==w->room)break;
      uint64_t piece;struct order_at at;
      if(!order_next(v->m,v->x,epochs,segments,way,c,j,v->posted[way][c][j],&v->hint[way][c][j],&piece,&at))break;
      const struct mesh_dma_span span=((const struct mesh_dma_span *)((const char *)v->m+at.t->spans))[at.span];
      if(!mesh_dma_span_valid(v->m,span) || span.bytes>order_piece_bytes(v->m)){
        fprintf(stderr,"wire: node %u posted an invalid span (%#llx, %llu)\n",v->m->node,(unsigned long long)span.offset,(unsigned long long)span.bytes);
        exit(3);
      }
      (way?f->receives:f->sends)[*tail%w->room]=(struct wire_piece){span,v->x->program[at.slot].program.key,at.execution,at.transfer,at.span,piece};
      (*tail)++;v->posted[way][c][j]++;
    }
  }
}
/* the wire's matches on node d's cable c queue pair j toward node 1-d: each SEND at its head fills the receive at its
   head, the two ends' pieces the same piece of the same transfer and of equal length */
static inline void wire_match(struct wire *w,uint32_t d,uint32_t c,uint32_t j,uint64_t most){
  struct wire_fifo *f=&w->fifo[d][c][j];
  for(;most && f->sent_head!=f->sent_tail && f->received_head!=f->received_tail;most--){
    const struct wire_piece s=f->sends[f->sent_head%w->room],r=f->receives[f->received_head%w->room];
    if(s.span.bytes!=r.span.bytes || s.key!=r.key || s.execution!=r.execution || s.transfer!=r.transfer || s.span_index!=r.span_index ||
       s.piece!=r.piece){
      fprintf(stderr,"wire: node %u -> %u cable %u queue pair %u: a SEND (program %llu execution %llu transfer %llu span %llu piece %llu, %llu "
              "bytes) met a receive (program %llu execution %llu transfer %llu span %llu piece %llu, %llu bytes): the placements differ\n",
              w->node[d].m->node,w->node[1-d].m->node,c,j,(unsigned long long)s.key,(unsigned long long)s.execution,(unsigned long long)s.transfer,
              (unsigned long long)s.span_index,(unsigned long long)s.piece,(unsigned long long)s.span.bytes,(unsigned long long)r.key,
              (unsigned long long)r.execution,(unsigned long long)r.transfer,(unsigned long long)r.span_index,(unsigned long long)r.piece,
              (unsigned long long)r.span.bytes);
      exit(3);
    }
    memcpy((char *)w->node[1-d].m+r.span.offset,(const char *)w->node[d].m+s.span.offset,s.span.bytes);
    atomic_thread_fence(memory_order_release);
    f->sent_head++;f->received_head++;w->matched++;w->bytes+=s.span.bytes;
    w->node[d].done[0][c][j]++;w->node[1-d].done[1][c][j]++;
  }
}
/* node n's done words raised through what its queue pairs have done, by each of its cables' raisers in turn, each with
   its own state (the bridge's port children: mesh-link.c order_pass) */
static inline void wire_raise(struct wire *w,uint32_t n){
  struct wire_node *v=w->node+n;
  const uint32_t epochs=atomic_load_explicit(&v->x->epochs,memory_order_acquire);
  if(!epochs)return;
  const uint64_t segments=atomic_load_explicit(&v->x->segments,memory_order_acquire);
  const struct mesh_epoch *last=v->x->epoch+epochs-1;
  for(int way=0;way<2;way++){
    uint64_t done[WIRE_CABLES*WIRE_LANES];uint32_t k=0;
    for(uint32_t c=0;c<last->cables;c++)for(uint32_t j=0;j<last->cable[c].lanes;j++)done[k++]=v->done[way][last->cable[c].port][j];
    for(uint32_t c=0;c<last->cables;c++)order_raise_pass(v->m,v->x,epochs,segments,way,done,&v->raiser[c][way]);
  }
}
/* both nodes' epoch over cables [0, cables), from the later of the two ends' bounds each way (mesh-link.c order_turn:
   each end bounds its posting past its frontier, the two bounds exchanged, the epoch begins at the later), the bound
   lifted.  `slack` the pieces past each frontier (0 for the first epoch) */
static inline void wire_epoch(struct wire *w,uint32_t cables,uint64_t slack){
  uint64_t bound[2][2];
  for(int n=0;n<2;n++){
    struct wire_node *v=w->node+n;
    const uint32_t epochs=atomic_load(&v->x->epochs);
    for(int way=0;way<2;way++){
      uint64_t posted[WIRE_CABLES*WIRE_LANES];uint32_t k=0;
      const struct mesh_epoch *last=epochs?v->x->epoch+epochs-1:NULL;
      for(uint32_t c=0;last && c<last->cables;c++)for(uint32_t j=0;j<last->cable[c].lanes;j++)posted[k++]=v->posted[way][last->cable[c].port][j];
      bound[n][way]=epochs?order_posted_through(v->x,epochs,way,posted)+slack:0;
      atomic_store(&v->x->limit[way],bound[n][way]);
    }
  }
  uint32_t port[WIRE_CABLES],lanes[WIRE_CABLES];uint64_t identity[WIRE_CABLES];
  for(uint32_t c=0;c<cables;c++){port[c]=c;lanes[c]=w->lanes;identity[c]=c+1;}
  for(int n=0;n<2;n++){
    const uint64_t first[2]={bound[n][0]>bound[1-n][1]?bound[n][0]:bound[1-n][1],bound[n][1]>bound[1-n][0]?bound[n][1]:bound[1-n][0]};
    if(order_extend(w->node[n].x,first,cables,port,lanes,identity)){fprintf(stderr,"wire: epochs exhausted\n");exit(3);}
    for(int way=0;way<2;way++)atomic_store(&w->node[n].x->limit[way],UINT64_MAX);
  }
}
/* one turn of the wire: every node's posts, then every queue pair's matches (up to `burst`, else 1 to 8 at random) in a
   rotating order with a quarter of them skipped at random, `lag` (a queue pair index, UINT32_MAX none) served once in `rarely` turns at most, then the raises */
static inline void wire_turn(struct wire *w,uint64_t turn,uint32_t lag,uint32_t rarely){
  const uint32_t cables=w->cables,lanes=w->lanes,q=2*cables*lanes;
  for(uint32_t n=0;n<2;n++)for(uint32_t c=0;c<cables;c++)for(uint32_t j=0;j<lanes;j++)wire_post(w,n,c,j);
  for(uint32_t k=0;k<q;k++){
    const uint32_t at=(uint32_t)((k+turn)%q);
    const uint64_t r=wire_next_random(w);
    if(r%4==0)continue;
    if(at==lag && (r>>8)%rarely)continue;
    wire_match(w,at/(cables*lanes),at/lanes%cables,at%lanes,w->burst?w->burst:1+(r>>20)%8);
  }
  for(uint32_t n=0;n<2;n++)wire_raise(w,n);
}
#endif
