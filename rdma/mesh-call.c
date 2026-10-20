#include "mesh-call.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

/* design/algorithm-sources.md#programtensor */
static uint32_t mesh_section_row(struct mesh_section section,uint32_t index){return section.first+index*section.stride;}

/* ---- The streams toward nodes (mesh-call.h; design/transport.md section 14) ------------------------- */
/* the live stream toward `node`, NULL where none goes there */
static struct mesh_stream *stream_toward(struct hdr *m,uint32_t node){
  for(uint32_t p=0;node && p<m->links && p<MESH_PORTS;p++){
    struct mesh_stream *x=mesh_stream_at(m,p);
    if(atomic_load_explicit(&x->live,memory_order_acquire) && atomic_load_explicit(&x->peer,memory_order_relaxed)==node)return x;
  }
  return NULL;
}
uint32_t mesh_nodes_reached(struct hdr *m,struct mesh_reach *reach,uint32_t room){
  uint32_t count=0;
  for(uint32_t p=0;m && p<m->links && p<MESH_PORTS && count<room;p++){
    const struct mesh_stream *x=mesh_stream_at(m,p);
    const uint32_t node=atomic_load_explicit(&x->peer,memory_order_relaxed);
    if(!node || !atomic_load_explicit(&x->live,memory_order_acquire))continue;
    int seen=0;
    for(uint32_t k=0;k<count;k++)seen|=reach[k].node==node;
    if(!seen)reach[count++]=(struct mesh_reach){.node=node,.bandwidth=atomic_load_explicit(&x->bandwidth,memory_order_relaxed)};
  }
  return count;
}
int mesh_stream_take(struct mesh_ctx *context,uint32_t node,uint64_t next[2]){
  if(!context || !context->client || !next)return EINVAL;
  struct mesh_stream *x=stream_toward(context->M,node);
  if(!x)return ENOLINK;
  uint64_t vacant=0;
  if(!atomic_load_explicit(&x->free,memory_order_acquire) ||
     !atomic_compare_exchange_strong_explicit(&x->owner,&vacant,context->client,memory_order_acq_rel,memory_order_acquire))return EBUSY;
  next[0]=atomic_load_explicit(&x->next[0],memory_order_relaxed);next[1]=atomic_load_explicit(&x->next[1],memory_order_relaxed);
  atomic_store_explicit(&x->declared[0],next[0],memory_order_relaxed);atomic_store_explicit(&x->declared[1],next[1],memory_order_release);
  return 0;
}
void mesh_stream_give(struct mesh_ctx *context,uint32_t node){
  if(!context || !context->client)return;
  for(uint32_t p=0;p<context->M->links && p<MESH_PORTS;p++){
    struct mesh_stream *x=mesh_stream_at(context->M,p);
    uint64_t mine=context->client;
    if(atomic_load_explicit(&x->peer,memory_order_relaxed)==node)atomic_compare_exchange_strong(&x->owner,&mine,0);
  }
}
int mesh_stream_view(struct mesh_ctx *context,uint32_t node,int large,struct mesh_stream_view *view){
  if(!context || !context->client || !view)return EINVAL;
  struct hdr *m=context->M;
  const struct mesh_stream *x=stream_toward(m,node);
  if(!x)return ENOLINK;
  if(atomic_load_explicit(&x->owner,memory_order_acquire)!=context->client)return EPERM;
  const uint32_t c=!large;
  *view=(struct mesh_stream_view){.in=(unsigned char *)m+x->in[c],.out=(unsigned char *)m+x->out[c],.ready=(_Atomic uint64_t *)((char *)m+x->ready[c]),
    .landed=(_Atomic uint64_t *)((char *)m+x->landed[c]),.unit=x->unit[c],.units=x->units[c]};
  return 0;
}
int mesh_stream_declare(struct mesh_ctx *context,uint32_t node,int large,uint64_t declared,uint64_t deadline){
  if(!context || !context->client)return EINVAL;
  struct mesh_stream *x=stream_toward(context->M,node);
  if(!x)return ENOLINK;
  atomic_store_explicit(&x->deadline,deadline,memory_order_relaxed);
  atomic_store_explicit(&x->declared[!large],declared,memory_order_release);
  return 0;
}

/* design/algorithm-sources.md#programtensor: a lease's rows and arena pages, the pages anywhere the window has them */
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

