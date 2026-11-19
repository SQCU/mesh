#include "mesh-call.h"
#include "mesh-observe.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* design/algorithm-sources.md#programtensor */
static uint32_t mesh_section_row(struct mesh_section section,uint32_t index){return section.first+index*section.stride;}

/* ---- The orders toward nodes (mesh-call.h; design/node-addressed-transfers.md) ----------------------------- */
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
/* the ring's next slot taken by compare-and-swap on its owner word while it is free for that index (struct mesh_slot),
   the tail moved past it by whoever sees it taken; the transfer written, then the slot's sequence its index's */
int mesh_order_append(struct mesh_ctx *context,const struct mesh_order *order,int way,const struct mesh_crossing *transfer){
  if(!context || !context->client || !transfer || way<0 || way>1)return EINVAL;
  struct hdr *m=context->M;
  if(!mesh_order_live(context,order))return ENOLINK;
  if(!mesh_crossing_valid(m,transfer))return EINVAL;
  struct mesh_node *x=mesh_node_at(m,order->slot);
  for(;;){
    const uint64_t i=atomic_load_explicit(&x->tail[way],memory_order_acquire);
    struct mesh_slot *s=mesh_ring(m,x,way,i);
    const uint64_t sequence=atomic_load_explicit(&s->sequence,memory_order_acquire);
    if(sequence<2*i){mesh_notify(&m->control);return EAGAIN;}
    uint64_t vacant=0;
    if(sequence==2*i && atomic_compare_exchange_strong_explicit(&s->owner,&vacant,context->client,memory_order_acq_rel,memory_order_acquire)){
      uint64_t at=i;
      atomic_compare_exchange_strong_explicit(&x->tail[way],&at,i+1,memory_order_acq_rel,memory_order_relaxed);
      s->t=*transfer;s->t.generation=order->generation;
      atomic_store_explicit(&s->sequence,2*i+1,memory_order_release);
      return 0;
    }
    uint64_t at=i;
    atomic_compare_exchange_strong_explicit(&x->tail[way],&at,i+1,memory_order_acq_rel,memory_order_relaxed);
  }
}
uint64_t mesh_order_outstanding(struct mesh_ctx *context,const struct mesh_order *order,int way){
  if(!context || !context->M || !order || order->slot>=MESH_NODES || way<0 || way>1)return 0;
  const struct mesh_node *x=mesh_node_at(context->M,order->slot);
  return atomic_load_explicit(&x->tail[way],memory_order_acquire)-atomic_load_explicit(&x->retired[way],memory_order_acquire);
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

