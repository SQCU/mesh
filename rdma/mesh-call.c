#include "mesh-call.h"
#include "mesh-observe.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>

/* design/algorithm-sources.md#programtensor */
static uint32_t mesh_section_row(struct mesh_section section,uint32_t index){return section.first+index*section.stride;}

/* ---- The streams toward nodes (mesh-call.h; design/transport.md section 14) ------------------------- */
/* stream `stream` of the region, NULL where it has none such */
static struct mesh_stream *stream_held(struct hdr *m,uint32_t stream){
  return stream<m->streams && stream<MESH_STREAMS?mesh_stream_at(m,stream):NULL;
}
uint32_t mesh_nodes_reached(struct hdr *m,struct mesh_reach *reach,uint32_t room){
  uint32_t count=0;
  for(uint32_t p=0;m && p<m->streams && p<MESH_STREAMS && count<room;p++){
    const struct mesh_stream *x=mesh_stream_at(m,p);
    const uint32_t node=atomic_load_explicit(&x->peer,memory_order_relaxed);
    if(!node || !atomic_load_explicit(&x->live,memory_order_acquire))continue;
    int seen=0;
    for(uint32_t k=0;k<count;k++)seen|=reach[k].node==node;
    if(!seen)reach[count++]=(struct mesh_reach){.node=node,.bandwidth=atomic_load_explicit(&x->bandwidth,memory_order_relaxed)};
  }
  return count;
}
uint32_t mesh_streams_free(struct hdr *m,uint32_t *node,uint64_t *wire,uint32_t room){
  uint32_t count=0;
  for(uint32_t p=0;m && p<m->streams && p<MESH_STREAMS && count<room;p++){
    const struct mesh_stream *x=mesh_stream_at(m,p);
    if(!atomic_load_explicit(&x->live,memory_order_acquire) || atomic_load_explicit(&x->owner,memory_order_relaxed))continue;
    node[count]=atomic_load_explicit(&x->peer,memory_order_relaxed);wire[count++]=atomic_load_explicit(&x->wire,memory_order_relaxed);
  }
  return count;
}
/* design/whole-function.md#fabric */
uint32_t mesh_endpoints(struct hdr *m,struct mesh_endpoint *out,uint32_t room){
  uint32_t count=0;
  for(uint32_t p=0;m && p<m->streams && p<MESH_STREAMS && count<room;p++){
    const struct mesh_stream *x=mesh_stream_at(m,p);
    if(!atomic_load_explicit(&x->live,memory_order_acquire))continue;
    out[count++]=(struct mesh_endpoint){
      .wire=atomic_load_explicit(&x->wire,memory_order_relaxed),
      .bandwidth=atomic_load_explicit(&x->bandwidth,memory_order_relaxed)*MESH_LINK_STREAMS,
      .node=atomic_load_explicit(&x->peer,memory_order_relaxed),.bundle=p/MESH_LINK_STREAMS,
      .available=!atomic_load_explicit(&x->owner,memory_order_relaxed)};
  }
  return count;
}
int mesh_stream_take(struct mesh_ctx *context,uint64_t wire,uint32_t *stream){
  if(!context || !context->client || !stream || !wire)return EINVAL;
  struct hdr *m=context->M;
  for(uint32_t p=0;p<m->streams && p<MESH_STREAMS;p++){
    struct mesh_stream *x=mesh_stream_at(m,p);
    if(!atomic_load_explicit(&x->live,memory_order_acquire) || atomic_load_explicit(&x->wire,memory_order_relaxed)!=wire)continue;
    uint64_t vacant=0;
    if(!atomic_compare_exchange_strong_explicit(&x->owner,&vacant,context->client,memory_order_acq_rel,memory_order_acquire))return EBUSY;
    *stream=p;
    return 0;
  }
  return ENOLINK;
}
void mesh_stream_give(struct mesh_ctx *context,uint32_t stream){
  if(!context || !context->client)return;
  struct mesh_stream *x=stream_held(context->M,stream);
  uint64_t mine=context->client;
  if(x)atomic_compare_exchange_strong(&x->owner,&mine,mine|MESH_STREAM_GIVEN);
}
int mesh_stream_held(struct mesh_ctx *context,uint32_t stream,uint64_t wire){
  const struct mesh_stream *x=context?stream_held(context->M,stream):NULL;
  return x && context->client && atomic_load_explicit(&x->live,memory_order_acquire) &&
         atomic_load_explicit(&x->owner,memory_order_acquire)==context->client && atomic_load_explicit(&x->wire,memory_order_relaxed)==wire;
}
/* the stream `stream` this client holds, live; NULL with *error set where not */
static struct mesh_stream *stream_owned(struct mesh_ctx *context,uint32_t stream,int *error){
  struct mesh_stream *x=context && context->client?stream_held(context->M,stream):NULL;
  *error=!context || !context->client?EINVAL:!x || !atomic_load_explicit(&x->live,memory_order_acquire)?ENOLINK:
         atomic_load_explicit(&x->owner,memory_order_acquire)!=context->client?EPERM:0;
  return *error?NULL:x;
}
int mesh_stream_view(struct mesh_ctx *context,uint32_t stream,struct mesh_stream_view *view){
  int error;
  struct mesh_stream *x=view?stream_owned(context,stream,&error):NULL;
  if(!view)return EINVAL;
  if(!x)return error;
  char *m=(char *)context->M;
  struct mesh_entry *lists=(struct mesh_entry *)(m+x->lists);
  *view=(struct mesh_stream_view){.list={lists,lists+MESH_STREAM_ENTRIES},.presented={&x->sends,&x->lands},.taken={&x->taken[0],&x->taken[1]},
    .ready=&x->ready,.sent=&x->sent,.landed=&x->landed,.zwoggle=&x->zwoggle};
  return 0;
}
int mesh_stream_present(struct mesh_ctx *context,uint32_t stream,int land,struct mesh_entry entry){
  int error;
  struct mesh_stream *x=stream_owned(context,stream,&error);
  if(!x)return error;
  const struct hdr *m=context->M;
  if(land<0 || land>1 || !mesh_entry_valid(m,entry))return EINVAL;
  _Atomic uint64_t *presented=land?&x->lands:&x->sends;
  const uint64_t e=atomic_load_explicit(presented,memory_order_relaxed);
  if(e-atomic_load_explicit(&x->taken[land],memory_order_acquire)>=MESH_STREAM_ENTRIES)return EAGAIN;
  ((struct mesh_entry *)((char *)m+x->lists))[(uint64_t)land*MESH_STREAM_ENTRIES+e%MESH_STREAM_ENTRIES]=entry;
  atomic_store_explicit(presented,e+1,memory_order_release);
  return 0;
}

/* design/algorithm-sources.md#programtensor: a lease's rows and arena pages, the pages anywhere the region has them */
int mesh_section_create(struct mesh_ctx *context,size_t bytes,uint32_t count,enum mesh_backing backing,struct mesh_section *section){
  struct hdr *m=context->M;
  if(!bytes || !count)return EINVAL;
  size_t quantum=(size_t)m->block*m->pgsz,capacity=mesh_arena_pages(m);
  if(bytes>capacity*(size_t)m->pgsz)return ENOMEM;
  size_t span=(bytes+quantum-1)/quantum*m->block;
  if(span>capacity/count)return ENOMEM;
  if(!context->view){
    const size_t length=(size_t)mesh_rows(m)*quantum;
    void *view=mmap(NULL,length,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0);
    if(view==MAP_FAILED)return errno;
    context->view=view;context->view_bytes=length;
  }
  uint32_t stride=(uint32_t)span/m->block,rows=count*stride,first=mesh_rows_alloc(context,rows);
  if(first==MESH_ABSENT)return errno;
  for(uint32_t row=first;row<first+rows;row+=stride){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    buffer->pages=(uint32_t)span;
    atomic_store_explicit(&buffer->owner,context->client,memory_order_release);
  }
  mesh_bits_set(mesh_plane(m,MESH_ROW_HOT),first,rows);
  int error=0;
  const uint32_t take=backing==MESH_BACKING_LINEAR?rows:1;
  for(uint32_t r=0;r<rows;r+=take){
    const uint32_t page=mesh_arena_alloc(context,take*m->block,m->block);
    if(page==MESH_ABSENT){error=errno;break;}
    for(uint32_t k=0;k<take;k++)
      mesh_backing_bind(context,first+r+k,m->block,page+k*m->block,(r+k)/stride);
  }
  /* Coalesce adjacent backing rows into one OS mapping. The virtual window's
     order is logical page order, independent of arena fragmentation. */
  for(uint32_t r=0;!error && r<rows;){
    const uint64_t offset=atomic_load_explicit(&mesh_page(m)[first+r].address,memory_order_relaxed);
    uint32_t end=r+1;
    while(end<rows && atomic_load_explicit(&mesh_page(m)[first+end].address,memory_order_relaxed)==offset+(end-r)*quantum)end++;
    mach_vm_address_t at=(uintptr_t)context->view+(first+r)*quantum;
    vm_prot_t current,maximum;
    const kern_return_t mapped=mach_vm_remap(mach_task_self(),&at,(end-r)*quantum,0,VM_FLAGS_FIXED|VM_FLAGS_OVERWRITE,
      mach_task_self(),(uintptr_t)m+offset,FALSE,&current,&maximum,VM_INHERIT_NONE);
    if(mapped!=KERN_SUCCESS)error=mapped==KERN_NO_SPACE || mapped==KERN_RESOURCE_SHORTAGE?ENOMEM:EINVAL;
    r=end;
  }
  if(error){
    for(uint32_t r=0;r<rows;r++){
      const uint32_t page=atomic_load_explicit(&mesh_page(m)[first+r].mapping,memory_order_relaxed);
      if(page!=MESH_ABSENT)mesh_bits_clear(mesh_arena_bits(m),page,m->block);
      mesh_buffers(m)[first+r].pages=0;
      atomic_store_explicit(&mesh_buffers(m)[first+r].owner,0,memory_order_release);
    }
    mesh_bits_clear(mesh_plane(m,MESH_ROW_HOT),first,rows);
    mesh_rows_release(context,first,rows);return error;
  }
  *section=(struct mesh_section){first,(uint32_t)span,bytes,count,stride};
  return 0;
}

/* design/algorithm-sources.md#programtensor */
void *mesh_section_address(struct mesh_ctx *context,struct mesh_section section,uint32_t index){return (char *)context->view+(size_t)mesh_section_row(section,index)*context->M->block*context->M->pgsz;}

