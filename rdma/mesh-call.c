#include "mesh-call.h"
#include "mesh-order.h"
#include "mesh-observe.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* design/algorithm-sources.md#programtensor */
static uint32_t mesh_section_row(struct mesh_section section,uint32_t index){return section.first+index*section.stride;}

/* ---- The orders toward nodes (mesh-call.h; design/declared-programs.md) ----------------------------- */
uint32_t mesh_nodes_reached(struct hdr *m,struct mesh_reach *reach,uint32_t room){
  uint32_t count=0;
  for(uint32_t n=0;m && n<MESH_NODES && count<room;n++){
    const struct mesh_node *x=mesh_node_at(m,n);
    const uint32_t node=atomic_load_explicit(&x->node,memory_order_relaxed);
    if(!node || !atomic_load_explicit(&x->live,memory_order_acquire))continue;
    reach[count++]=(struct mesh_reach){.node=node,.bandwidth=atomic_load_explicit(&x->bandwidth,memory_order_relaxed)};
  }
  return count;
}
int mesh_order_open(struct mesh_ctx *context,uint32_t node,struct mesh_order *order){
  if(!context || !context->M || !order || !node)return EINVAL;
  for(uint32_t n=0;n<MESH_NODES;n++){
    struct mesh_node *x=mesh_node_at(context->M,n);
    if(atomic_load_explicit(&x->node,memory_order_acquire)!=node || !atomic_load_explicit(&x->live,memory_order_acquire))continue;
    *order=(struct mesh_order){.slot=n,.node=node,.generation=(uint32_t)atomic_load_explicit(&x->generation,memory_order_acquire)};
    return mesh_order_live(context,order)?0:ENOLINK;
  }
  return ENOLINK;
}
int mesh_order_live(struct mesh_ctx *context,const struct mesh_order *order){
  if(!context || !context->M || !order || order->slot>=MESH_NODES)return 0;
  const struct mesh_node *x=mesh_node_at(context->M,order->slot);
  return atomic_load_explicit(&x->node,memory_order_acquire)==order->node && atomic_load_explicit(&x->live,memory_order_acquire) &&
         (uint32_t)atomic_load_explicit(&x->generation,memory_order_acquire)==order->generation;
}
/* whether client `owner` still holds its slot */
static int client_live(struct hdr *m,uint64_t owner){
  return owner && atomic_load_explicit(&mesh_client_at(m,owner)->owner,memory_order_acquire)==owner;
}
/* whether the region holds `bytes` at `at`, 8-byte aligned */
static int words_held(const struct hdr *m,uint64_t at,uint64_t bytes){ return mesh_region_holds(m,at,bytes,8); }
/* a program's tables as the region holds them: each direction's transfers in the arena, each one's `first` the pieces
   before it, its spans valid and each at most one piece, its step within the period, its gate and done words words of
   the region; the pieces an execution its units' sum */
static int program_valid(const struct hdr *m,const struct mesh_declaration *p){
  if(!p->period || !p->count)return 0;
  const uint64_t piece=order_piece_bytes(m);
  for(int d=0;d<2;d++){
    const uint64_t n=p->transfers[d];
    if(!n){if(p->pieces[d] || p->table[d])return 0;continue;}
    if(n>UINT32_MAX || p->table[d]<m->data_off || !words_held(m,p->table[d],n*sizeof(struct mesh_crossing)))return 0;
    const struct mesh_crossing *t=(const struct mesh_crossing *)((const char *)m+p->table[d]);
    uint64_t pieces=0;
    for(uint64_t i=0;i<n;i++){
      if(t[i].first!=pieces || !t[i].units || t[i].step>=p->period || t[i].spans<m->data_off ||
         t[i].units>(m->length-m->data_off)/sizeof(struct mesh_dma_span) ||
         !words_held(m,t[i].spans,t[i].units*sizeof(struct mesh_dma_span)) ||
         (t[i].gate && !words_held(m,t[i].gate,8)) || (t[i].done && !words_held(m,t[i].done,8)))return 0;
      const struct mesh_dma_span *spans=(const struct mesh_dma_span *)((const char *)m+t[i].spans);
      for(uint64_t k=0;k<t[i].units;k++)if(!mesh_dma_span_valid(m,spans[k]) || spans[k].bytes>piece)return 0;
      pieces+=t[i].units;
    }
    if(pieces!=p->pieces[d])return 0;
    /* at most ORDER_PROGRAM_WORDS distinct done words a direction (mesh-order.h struct order_raiser: a cursor a word) */
    uint64_t words[ORDER_PROGRAM_WORDS];uint32_t distinct=0;
    for(uint64_t i=0;i<n;i++){
      if(!t[i].done)continue;
      uint32_t w=0;
      while(w<distinct && words[w]!=t[i].done)w++;
      if(w<distinct)continue;
      if(distinct==ORDER_PROGRAM_WORDS)return 0;
      words[distinct++]=t[i].done;
    }
  }
  return 1;
}
/* the node's declaring word taken for `client` (from a client gone, where one holds it); 0, or EBUSY */
static int declaring_take(struct hdr *m,struct mesh_node *x,uint64_t client){
  uint64_t held=0;
  if(atomic_compare_exchange_strong_explicit(&x->declaring,&held,client,memory_order_acq_rel,memory_order_acquire))return 0;
  if(!client_live(m,held) && atomic_compare_exchange_strong_explicit(&x->declaring,&held,client,memory_order_acq_rel,memory_order_acquire))return 0;
  return EBUSY;
}
/* The one declaration step (mesh_declare, mesh_close), under the node's declaring word: the programs of the node's
   last segment that close (this client's own, `only` that slot of them where it is one, and every one of a client gone), a restart asked where one of a client
   gone holds pieces outstanding or ended mid-execution (its peer's count may differ: only a restart realigns the two
   ends), program slots of programs closed and wholly done made vacant, the declared one (`program`, NULL none) given a
   slot, and the next segment published (mesh-order.h order_compose, order_publish_segment) */
static int declare_held(struct mesh_ctx *context,const struct mesh_order *order,struct mesh_node *x,const struct mesh_declaration *program,
                        uint32_t only,struct mesh_run *run){
  struct hdr *m=context->M;
  if(!mesh_order_live(context,order))return ENOLINK;
  const uint64_t segments=atomic_load_explicit(&x->segments,memory_order_acquire);
  uint32_t closing=0,gone=0;
  if(segments){
    const struct mesh_segment *g=x->segment+(segments-1)%MESH_SEGMENTS;
    for(uint32_t k=0;k<g->programs && k<MESH_SEGMENT_PROGRAMS;k++){
      const uint64_t owner=atomic_load_explicit(&x->program[g->slot[k]].owner,memory_order_acquire);
      if(owner==context->client){if(only==MESH_PROGRAMS || only==g->slot[k])closing|=UINT32_C(1)<<g->slot[k];}
      else if(!client_live(m,owner)){closing|=UINT32_C(1)<<g->slot[k];gone|=UINT32_C(1)<<g->slot[k];}
    }
    for(uint32_t k=0;gone && k<g->programs && k<MESH_SEGMENT_PROGRAMS;k++){
      uint64_t rounds;
      if(!(gone>>g->slot[k]&1))continue;
      /* mid-execution (its peer's count may differ), or a piece of it that can never be posted (its gate shut, its
         client gone: order_gate_shut); its pieces posted complete or fail through their own completions */
      int outstanding=order_round_of(x,g,k,&rounds)!=0;
      for(int d=0;!outstanding && d<2;d++)
        outstanding=order_gate_shut(m,x,segments,g->slot[k],d,atomic_load_explicit(&x->raised[d],memory_order_acquire),
                                    g->start[d]+rounds*g->round[d]);
      if(outstanding){atomic_store_explicit(&x->restart,1,memory_order_release);mesh_notify(&m->control);return ENOLINK;}
    }
  }
  /* a closed program of a client gone with a piece that can never be posted (its gate shut: a SEND its client never
     published, e.g. a fused crossing's frame whose producer was never run) holds the queue pair that piece is on, so
     every program after it would wait behind it for ever: the order restarts (a restart asked, as for a gone program
     mid-execution) and this declaration fails as a value.  Its pieces posted or postable complete or fail through their
     own completions, and the programs after them follow them on their queue pairs: nothing is waited for here */
  for(uint32_t p=0;p<MESH_PROGRAMS;p++){
    const struct mesh_declared *d=x->program+p;
    const uint64_t owner=atomic_load_explicit(&d->owner,memory_order_acquire);
    if(!owner || client_live(m,owner) || atomic_load_explicit(&d->closed,memory_order_acquire)==UINT64_MAX)continue;
    for(int k=0;k<2;k++)
      if(order_gate_shut(m,x,segments,p,k,atomic_load_explicit(&x->raised[k],memory_order_acquire),atomic_load_explicit(&d->end[k],memory_order_acquire))){
        atomic_store_explicit(&x->restart,1,memory_order_release);mesh_notify(&m->control);return ENOLINK;
      }
  }
  for(uint32_t p=0;p<MESH_PROGRAMS;p++){
    struct mesh_declared *d=x->program+p;
    uint64_t owner=atomic_load_explicit(&d->owner,memory_order_acquire);
    if(owner && atomic_load_explicit(&d->closed,memory_order_acquire)!=UINT64_MAX &&
       atomic_load_explicit(&x->raised[0],memory_order_acquire)>=atomic_load_explicit(&d->end[0],memory_order_acquire) &&
       atomic_load_explicit(&x->raised[1],memory_order_acquire)>=atomic_load_explicit(&d->end[1],memory_order_acquire))
      atomic_compare_exchange_strong_explicit(&d->owner,&owner,0,memory_order_acq_rel,memory_order_relaxed);
  }
  if(!program && !closing)return 0;
  uint32_t joining=MESH_PROGRAMS;
  if(program){
    for(uint32_t p=0;p<MESH_PROGRAMS && joining==MESH_PROGRAMS;p++){
      uint64_t vacant=0;
      if(atomic_compare_exchange_strong_explicit(&x->program[p].owner,&vacant,context->client,memory_order_acq_rel,memory_order_relaxed))joining=p;
    }
    if(joining==MESH_PROGRAMS)return ENOSPC;
    struct mesh_declared *d=x->program+joining;
    d->program=*program;
    atomic_store_explicit(&d->issued,0,memory_order_relaxed);atomic_store_explicit(&d->closed,UINT64_MAX,memory_order_relaxed);
    atomic_store_explicit(&d->end[0],UINT64_MAX,memory_order_relaxed);atomic_store_explicit(&d->end[1],UINT64_MAX,memory_order_relaxed);
  }
  struct order_next next;
  int error=order_compose(x,closing,joining,&next);
  if(!error && segments>=MESH_SEGMENTS && !order_segment_done(x,segments-MESH_SEGMENTS))error=ENOSPC;
  if(error){
    if(joining<MESH_PROGRAMS)atomic_store_explicit(&x->program[joining].owner,0,memory_order_release);
    return error;
  }
  if(segments){
    const struct mesh_segment *g=x->segment+(segments-1)%MESH_SEGMENTS;
    for(uint32_t k=0;k<g->programs && k<MESH_SEGMENT_PROGRAMS;k++){
      if(!(closing>>g->slot[k]&1))continue;
      struct mesh_declared *d=x->program+g->slot[k];
      atomic_store_explicit(&d->end[0],next.start[0],memory_order_relaxed);atomic_store_explicit(&d->end[1],next.start[1],memory_order_relaxed);
      atomic_store_explicit(&d->closed,g->base[k]+next.rounds*d->program.count,memory_order_release);
    }
  }
  order_publish_segment(x,segments,&next);
  /* the bridge woken: it tells the peer the segment's digest (mesh-order.h order_segments_turn) */
  mesh_notify(&m->control);
  if(run)*run=(struct mesh_run){.order=*order,.program=joining,.issued=joining<MESH_PROGRAMS?&x->program[joining].issued:NULL};
  return 0;
}
static int declare(struct mesh_ctx *context,const struct mesh_order *order,const struct mesh_declaration *program,uint32_t only,
                   struct mesh_run *run){
  if(!context || !context->M || !context->client || !order || order->slot>=MESH_NODES)return EINVAL;
  struct hdr *m=context->M;
  if(program && !program_valid(m,program))return EINVAL;
  struct mesh_node *x=mesh_node_at(m,order->slot);
  /* another client's declaration toward the node under way is awaited (declarations are made one at a time), at most
     MESH_PLATFORM_NS */
  int error;
  for(const uint64_t until=mesh_clock_ns()+MESH_PLATFORM_NS;(error=declaring_take(m,x,context->client)) && mesh_clock_ns()<until;)
    __asm__ volatile("yield");
  if(error)return error;
  error=declare_held(context,order,x,program,only,run);
  uint64_t held=context->client;
  atomic_compare_exchange_strong_explicit(&x->declaring,&held,0,memory_order_acq_rel,memory_order_relaxed);
  return error;
}
int mesh_declare(struct mesh_ctx *context,const struct mesh_order *order,const struct mesh_declaration *program,struct mesh_run *run){
  if(!program || !run)return EINVAL;
  return declare(context,order,program,MESH_PROGRAMS,run);
}
int mesh_close(struct mesh_ctx *context,struct mesh_run *run){
  if(!run || run->program>=MESH_PROGRAMS)return EINVAL;
  return declare(context,&run->order,NULL,run->program,NULL);
}
int mesh_run_done(struct mesh_ctx *context,const struct mesh_run *run){
  if(!context || !context->M || !run || run->program>=MESH_PROGRAMS || run->order.slot>=MESH_NODES)return 1;
  if(!mesh_order_live(context,&run->order))return 1;
  const struct mesh_node *x=mesh_node_at(context->M,run->order.slot);
  const struct mesh_declared *d=x->program+run->program;
  if(atomic_load_explicit(&d->owner,memory_order_acquire)!=context->client)return 1;
  if(atomic_load_explicit(&d->closed,memory_order_acquire)==UINT64_MAX)return 0;
  return atomic_load_explicit(&x->raised[0],memory_order_acquire)>=atomic_load_explicit(&d->end[0],memory_order_acquire) &&
         atomic_load_explicit(&x->raised[1],memory_order_acquire)>=atomic_load_explicit(&d->end[1],memory_order_acquire);
}

/* design/algorithm-sources.md#programtensor: a lease's rows and arena pages, the pages anywhere the region has them */
int mesh_section_create(struct mesh_ctx *context,size_t bytes,uint32_t count,struct mesh_section *section){
  struct hdr *m=context->M;
  if(!bytes || !count)return EINVAL;
  size_t quantum=(size_t)m->block*m->pgsz,capacity=mesh_arena_pages(m);
  if(bytes>capacity*(size_t)m->pgsz)return ENOMEM;
  size_t span=(bytes+quantum-1)/quantum*m->block;
  if(span>capacity/count)return ENOMEM;
  uint32_t stride=(uint32_t)span/m->block,rows=count*stride,first=mesh_rows_alloc(context,rows);
  if(first==MESH_ABSENT)return errno;
  for(uint32_t row=first;row<first+rows;row+=stride){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    buffer->pages=(uint32_t)span;
    atomic_store_explicit(&buffer->owner,context->client,memory_order_release);
  }
  mesh_bits_set(mesh_plane(m,MESH_ROW_HOT),first,rows);
  /* design/transport.md */
  uint32_t page=mesh_arena_alloc(context,(uint32_t)span*count,m->block);
  if(page==MESH_ABSENT){
    int error=errno;
    mesh_bits_clear(mesh_plane(m,MESH_ROW_HOT),first,rows);
    for(uint32_t r=first;r<first+rows;r++){mesh_buffers(m)[r].pages=0;atomic_store_explicit(&mesh_buffers(m)[r].owner,0,memory_order_release);}
    mesh_rows_release(context,first,rows);return error;
  }
  for(uint32_t slot=0;slot<count;slot++)
    mesh_backing_bind(context,first+slot*stride,(uint32_t)span,page+slot*(uint32_t)span,slot);
  *section=(struct mesh_section){first,(uint32_t)span,bytes,count,stride};
  return 0;
}

/* design/algorithm-sources.md#programtensor */
void *mesh_section_address(struct mesh_ctx *context,struct mesh_section section,uint32_t index){return (char *)context->M+atomic_load_explicit(&mesh_page(context->M)[mesh_section_row(section,index)].address,memory_order_relaxed);}

