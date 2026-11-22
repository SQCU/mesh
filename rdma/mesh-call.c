#include "mesh-call.h"
#include "mesh-switched.h"
#include "mesh-observe.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* design/algorithm-sources.md#programtensor */
static uint32_t mesh_section_row(struct mesh_section section,uint32_t index){return section.first+index*section.stride;}

/* ---- The programs toward nodes (mesh-call.h; design/switched-multipath-dma.md) ---------------------------------- */
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
/* the node's slot still names it: a program toward a node lives as long as its client (the map reaching the node
   again or not changes nothing the program declared) */
int mesh_order_live(struct mesh_ctx *context,const struct mesh_order *order){
  if(!context || !context->M || !order || order->slot>=MESH_NODES)return 0;
  const struct mesh_node *x=mesh_node_at(context->M,order->slot);
  return atomic_load_explicit(&x->node,memory_order_acquire)==order->node &&
         (uint32_t)atomic_load_explicit(&x->generation,memory_order_acquire)==order->generation;
}
/* whether client `owner` still holds its slot */
static int client_live(struct hdr *m,uint64_t owner){
  return owner && owner!=UINT64_MAX && atomic_load_explicit(&mesh_client_at(m,owner)->owner,memory_order_acquire)==owner;
}
/* whether the region holds `bytes` at `at`, 8-byte aligned */
static int words_held(const struct hdr *m,uint64_t at,uint64_t bytes){ return mesh_region_holds(m,at,bytes,8); }
/* a program's tables as the region holds them: each direction's transfers in the arena, each one's `first` the pieces
   before it, its spans valid and each at most one piece, its step within the period, its gate and done words words of
   the region; the pieces an execution its units' sum */
static int program_valid(const struct hdr *m,const struct mesh_declaration *p){
  if(!p->period || !p->count)return 0;
  const uint64_t piece=mesh_switch_piece(m);
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
         ((t[i].gates || t[i].gate) && !words_held(m,t[i].gate,t[i].gates?t[i].gates*sizeof(struct mesh_gate_term):8)) ||
         (t[i].done && !words_held(m,t[i].done,8)))return 0;
      const struct mesh_gate_term *gate=(const void *)((const char *)m+t[i].gate);
      for(uint32_t k=0;k<t[i].gates;k++)if(!words_held(m,gate[k].word,8) || gate[k].threshold==UINT64_MAX)return 0;
      const struct mesh_dma_span *spans=(const struct mesh_dma_span *)((const char *)m+t[i].spans);
      for(uint64_t k=0;k<t[i].units;k++)if(!mesh_dma_span_valid(m,spans[k]) || spans[k].bytes>piece)return 0;
      pieces+=t[i].units;
    }
    if(pieces!=p->pieces[d])return 0;
  }
  return 1;
}
/* a program slot made vacant once its program is closed and wholly done, or its client is gone: its progress pages
   given back to the arena */
static void slot_vacate(struct hdr *m,struct mesh_declared *d){
  uint64_t owner=atomic_load_explicit(&d->owner,memory_order_acquire);
  if(!owner || owner==UINT64_MAX)return;
  if(client_live(m,owner) && !mesh_switch_done(m,d))return;
  if(!atomic_compare_exchange_strong_explicit(&d->owner,&owner,UINT64_MAX,memory_order_acq_rel,memory_order_relaxed))return;
  if(d->pages)mesh_bits_clear(mesh_arena_bits(m),d->page,d->pages);
  d->progress[0]=d->progress[1]=0;d->pages=0;
  atomic_store_explicit(&d->owner,0,memory_order_release);
  /* the pieces of its key still held at this node are gone now: their waits on `control` end (mesh-switched.h) */
  mesh_notify(&m->control);
}
/* Declared toward the order's node: a vacant program slot taken (UINT64_MAX while it is made, so no bridge reads it
   half made), the program's progress leased from the arena and made, the declaration copied, the slot given to the
   client.  This client's program there before it, if any, ends at the executions it issued (mesh_close).  Nothing is
   agreed with the peer and nothing is sent: the receiving end finds this program by its key in the name of every
   piece (mesh-switched.h mesh_switch_deliver) */
int mesh_declare(struct mesh_ctx *context,const struct mesh_order *order,const struct mesh_declaration *program,struct mesh_run *run){
  if(!context || !context->M || !context->client || !order || order->slot>=MESH_NODES || !program || !run)return EINVAL;
  struct hdr *m=context->M;
  if(!program_valid(m,program))return EINVAL;
  if(!mesh_order_live(context,order))return ENOLINK;
  struct mesh_node *x=mesh_node_at(m,order->slot);
  for(uint32_t p=0;p<MESH_PROGRAMS;p++){
    struct mesh_declared *d=x->program+p;
    if(atomic_load_explicit(&d->owner,memory_order_acquire)==context->client && atomic_load_explicit(&d->closed,memory_order_acquire)==UINT64_MAX){
      const uint64_t issued=atomic_load_explicit(&d->issued,memory_order_acquire);
      atomic_store_explicit(&d->closed,d->program.period?issued/d->program.period:0,memory_order_release);
    }
    slot_vacate(m,d);
  }
  uint32_t joining=MESH_PROGRAMS;
  for(uint32_t p=0;p<MESH_PROGRAMS && joining==MESH_PROGRAMS;p++){
    uint64_t vacant=0;
    if(atomic_compare_exchange_strong_explicit(&x->program[p].owner,&vacant,UINT64_MAX,memory_order_acq_rel,memory_order_relaxed))joining=p;
  }
  if(joining==MESH_PROGRAMS)return ENOSPC;
  struct mesh_declared *d=x->program+joining;
  const struct mesh_crossing *t[2]={NULL,NULL};uint64_t bytes[2]={0,0};
  for(int k=0;k<2;k++)if(program->transfers[k]){
    t[k]=(const struct mesh_crossing *)((const char *)m+program->table[k]);
    bytes[k]=(mesh_switch_progress_bytes(t[k],program->transfers[k])+63)/64*64;
  }
  const uint64_t page=(uint64_t)m->pgsz,pages=(bytes[0]+bytes[1]+page-1)/page;
  const uint32_t first=pages?mesh_arena_alloc(context,(uint32_t)pages,1):0;
  if(pages && first==MESH_ABSENT){atomic_store_explicit(&d->owner,0,memory_order_release);return ENOMEM;}
  d->prior=d->program.key;
  d->program=*program;
  d->page=first;d->pages=(uint32_t)pages;
  const uint64_t base=m->data_off+(uint64_t)first*page;
  d->progress[0]=bytes[0]?base:0;d->progress[1]=bytes[1]?base+bytes[0]:0;
  for(int k=0;k<2;k++)if(t[k])mesh_switch_progress_init(m,d,k,t[k],program->transfers[k]);
  atomic_store_explicit(&d->issued,0,memory_order_relaxed);atomic_store_explicit(&d->closed,UINT64_MAX,memory_order_relaxed);
  atomic_store_explicit(&d->claimed,0,memory_order_relaxed);
  /* the receive window anew (mesh-switched.h mesh_switch_credit): a slot's last program's credit, kept, let a sender
     run unbounded ahead of its receiver until the receiver's pool held nothing but later executions (every pool wedged) */
  atomic_store_explicit(&d->credit,0,memory_order_relaxed);atomic_store_explicit(&d->credited,0,memory_order_relaxed);
  atomic_store_explicit(&d->owner,context->client,memory_order_release);
  mesh_notify(&m->control);
  *run=(struct mesh_run){.order=*order,.program=joining,.issued=&d->issued};
  return 0;
}
/* the client's program ended at the executions it issued */
int mesh_close(struct mesh_ctx *context,struct mesh_run *run){
  if(!context || !context->M || !run || run->program>=MESH_PROGRAMS || run->order.slot>=MESH_NODES)return EINVAL;
  struct mesh_declared *d=mesh_node_at(context->M,run->order.slot)->program+run->program;
  if(atomic_load_explicit(&d->owner,memory_order_acquire)!=context->client)return 0;
  const uint64_t issued=atomic_load_explicit(&d->issued,memory_order_acquire);
  uint64_t open=UINT64_MAX;
  atomic_compare_exchange_strong_explicit(&d->closed,&open,d->program.period?issued/d->program.period:0,memory_order_acq_rel,memory_order_relaxed);
  slot_vacate(context->M,d);
  return 0;
}
int mesh_run_done(struct mesh_ctx *context,const struct mesh_run *run){
  if(!context || !context->M || !run || run->program>=MESH_PROGRAMS || run->order.slot>=MESH_NODES)return 1;
  const struct mesh_declared *d=mesh_node_at(context->M,run->order.slot)->program+run->program;
  if(atomic_load_explicit(&d->owner,memory_order_acquire)!=context->client)return 1;
  return mesh_switch_done(context->M,d);
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

