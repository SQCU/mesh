#include "mesh-call.h"
#include <stdlib.h>
#include <string.h>

/* design/algorithm-sources.md#programtensor */
static uint32_t mesh_section_row(struct mesh_section section,uint32_t index){return section.first+index*section.stride;}

/* design/algorithm-sources.md#programcopy */
int mesh_transfer_bind(struct mesh_ctx *context,uint32_t queue,int receive,uint32_t identity,struct mesh_section section){
  struct hdr *m=context->M;
  if(queue>=m->links*m->qps)return EINVAL;
  _Atomic uint32_t *length=mesh_order_length(m,context->client,queue,receive);
  uint32_t index=atomic_load_explicit(length,memory_order_relaxed);
  if(index==mesh_blocks(m))return ENOSPC;
  if(!receive)for(uint32_t value=0;value<section.count;value++){
    uint32_t row=mesh_section_row(section,value);
    uint32_t link=queue/m->qps;
    mesh_publish_bind(context,row,link)->count++;
  }
  mesh_transfers(m,context->client,queue,receive)[index]=(struct mesh_transfer){section.first,identity,section.count,section.stride,MESH_ABSENT,0,section.bytes};
  atomic_store_explicit(length,index+1,memory_order_release);
  return 0;
}

/* design/prepared-machine.md#M04 */
/* design/algorithm-sources.md#programcopy */
static int mesh_transfer_compare(const void *a,const void *b){
  const struct mesh_transfer *left=*(const struct mesh_transfer *const *)a,*right=*(const struct mesh_transfer *const *)b;
  return (left->local_row>right->local_row)-(left->local_row<right->local_row);
}

/* design/algorithm-sources.md#programcopy */
int mesh_transfers_prepare(struct mesh_ctx *context,uint32_t slots){
  struct hdr *m=context->M;
  if(!slots || slots>mesh_rows(m))return EINVAL;
  /* design/prepared-machine.md#M05 */
  m->instance_count[context->client>>63]=slots;
  uint32_t recurring=0,shared=0;
  for(uint32_t q=0;q<m->links*m->qps;q++)for(int direction=0;direction<2;direction++){
    uint32_t count=atomic_load_explicit(mesh_order_length(m,context->client,q,direction),memory_order_relaxed);
    struct mesh_transfer *transfers=mesh_transfers(m,context->client,q,direction);
    for(uint32_t i=0;i<count;i++){if(transfers[i].stride)recurring++;else shared++;}
  }
  struct mesh_instance *instances=mesh_instances(m,context->client);
  struct mesh_status initial=recurring+shared?(struct mesh_status){MESH_RESULT(MESH_RESULT_BUSY,0,0),0}:(struct mesh_status){0,UINT64_MAX};
  for(uint32_t slot=0;slot<slots;slot++){
    instances[slot].count=recurring;instances[slot].stride=slots;
    atomic_store_explicit(&instances[slot].invocation,slot,memory_order_relaxed);
    atomic_store_explicit(&instances[slot].remaining,recurring+shared,memory_order_relaxed);
    atomic_store_explicit(&instances[slot].status,initial,memory_order_relaxed);
  }
  atomic_store_explicit(&m->result[context->client>>63],initial,memory_order_relaxed);
  /* design/prepared-machine.md#M04 */
  for(uint32_t p=0;p<m->links;p++){
    uint64_t base=m->notice_off+(uint64_t)mesh_notice_queue(m,context->client,p)*m->notice_bytes;
    struct mesh_tx *tx=(void *)((char *)m+base);
    uint32_t count=0,once=0,length=0;
    for(uint32_t q=0;q<m->qps;q++)length+=atomic_load_explicit(mesh_order_length(m,context->client,p*m->qps+q,MESH_SEND),memory_order_relaxed);
    struct mesh_transfer **ordered=malloc((length?length:1)*sizeof *ordered);
    if(!ordered)return ENOMEM;
    uint32_t position=0;
    for(uint32_t q=0;q<m->qps;q++){
      struct mesh_transfer *out=mesh_transfers(m,context->client,p*m->qps+q,MESH_SEND);
      uint32_t length=atomic_load_explicit(mesh_order_length(m,context->client,p*m->qps+q,MESH_SEND),memory_order_relaxed);
      for(uint32_t i=0;i<length;i++){
        ordered[position++]=out+i;
        if(out[i].stride)count++;else once++;
      }
    }
    *tx=(struct mesh_tx){.count=count,.slots=slots,.once=once};
    memset(tx->cells,0,((size_t)count*slots+once)*sizeof *tx->cells);
    qsort(ordered,length,sizeof *ordered,mesh_transfer_compare);
    uint32_t next[2]={0,once};
    for(uint32_t i=0;i<length;i++){
      struct mesh_transfer *out=ordered[i];
      uint32_t varying=out->stride!=0;
      out->first=next[varying];
      for(uint32_t slot=0;slot<out->count;slot++){
        struct mesh_publication *publication=mesh_publication_at(m,out->local_row+slot*out->stride);
        for(uint32_t j=0;j<publication->sends;j++)if(publication->targets[j].stream==base){
          publication->targets[j].stream=base+offsetof(struct mesh_tx,cells)+sizeof(struct mesh_send)*(slot*count+next[varying]);
        }
      }
      next[varying]++;
    }
    free(ordered);
  }

  /* design/prepared-machine.md#M02 */
  for(uint32_t q=0;q<m->links*m->qps;q++){
    uint32_t count=atomic_load(mesh_order_length(m,context->client,q,MESH_RECEIVE)),pages=0;
    struct mesh_transfer *transfers=mesh_transfers(m,context->client,q,MESH_RECEIVE);
    for(uint32_t i=0;i<count;i++)pages+=transfers[i].count*mesh_buffers(m)[transfers[i].local_row].pages;
    if(!pages)continue;
    uint32_t page=mesh_arena_alloc(context,pages,m->block);
    if(page==MESH_ABSENT)return errno;
    struct mesh_pool *pool=&mesh_pools(m)[page/m->block];pool->pages=pages;
    atomic_store_explicit(&pool->owner,context->client,memory_order_release);
    for(uint32_t i=0;i<count;i++){
      transfers[i].pool=page;
      uint32_t span=mesh_buffers(m)[transfers[i].local_row].pages;
      for(uint32_t slot=0;slot<transfers[i].count;slot++){
        mesh_backing_bind(context,transfers[i].local_row+slot*transfers[i].stride,span,page,slot);
        page+=span;
      }
    }
  }
  return 0;
}

/* design/prepared-machine.md#M06 */
/* design/algorithm-sources.md#programcopy */
int mesh_transfers_start(struct mesh_ctx *context){
  struct hdr *m=context->M;
  atomic_store_explicit(&m->configured,context->client,memory_order_release);
  for(uint32_t p=0;p<m->links;p++){
    uint32_t transfers=0;
    for(uint32_t q=0;q<m->qps;q++)for(int d=0;d<2;d++)transfers+=atomic_load(mesh_order_length(m,context->client,p*m->qps+q,d));
    if(!transfers)continue;
    struct mesh_port_info *port=&mesh_links(m)[p].port;
    while(atomic_load_explicit(&port->prepared,memory_order_acquire)!=context->client){}
    if(atomic_load_explicit(&port->phase,memory_order_relaxed)==MESH_STOPPED)
      return (int)atomic_load_explicit(&port->code,memory_order_relaxed);
  }
  return 0;
}

/* design/algorithm-sources.md#programtensor */
int mesh_section_create(struct mesh_ctx *context,size_t bytes,uint32_t count,uint32_t channel,struct mesh_section *section){
  struct hdr *m=context->M;
  if(!bytes || !count)return EINVAL;
  size_t quantum=(size_t)m->block*m->pgsz;
  if(bytes>(size_t)mesh_rows(m)*m->pgsz)return ENOMEM;
  size_t span=(bytes+quantum-1)/quantum*m->block;
  if(span>mesh_rows(m)/count)return ENOMEM;
  uint32_t stride=(uint32_t)span/m->block,rows=count*stride,first=mesh_rows_alloc(context,rows);
  if(first==MESH_ABSENT)return errno;
  for(uint32_t row=first;row<first+rows;row+=stride){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    buffer->pages=(uint32_t)span;buffer->channel=channel;
    atomic_store_explicit(&buffer->owner,context->client,memory_order_release);
  }
  mesh_bits_set(m,MESH_ROW_HOT,first,rows);
  /* design/prepared-machine.md#M01 */
  /* design/prepared-machine.md#M03 */
  if(channel==MESH_ABSENT){
    uint32_t page=mesh_arena_alloc(context,(uint32_t)span*count,m->block);
    if(page==MESH_ABSENT){
      int error=errno;
      mesh_bits_clear(m,MESH_ROW_HOT,first,rows);
      for(uint32_t r=first;r<first+rows;r+=stride)mesh_buffers(m)[r].pages=0;
      mesh_rows_release(context,first,rows);return error;
    }
    for(uint32_t slot=0;slot<count;slot++)
      mesh_backing_bind(context,first+slot*stride,(uint32_t)span,page+slot*(uint32_t)span,slot);
  }
  *section=(struct mesh_section){first,(uint32_t)span,bytes,count,stride,channel};
  return 0;
}
/* design/algorithm-sources.md#programtensor */
uint32_t mesh_row_page(struct mesh_ctx *context,uint32_t row,uint32_t chunk){return atomic_load_explicit(&mesh_page(context->M)[row+chunk].mapping,memory_order_acquire);}
/* design/algorithm-sources.md#programtensor */
void *mesh_section_address(struct mesh_ctx *context,struct mesh_section section,uint32_t index){return mesh_at(context->M,mesh_row_page(context,mesh_section_row(section,index),0));}
/* design/algorithm-sources.md#programwrite */
void mesh_section_constant(struct mesh_ctx *context,struct mesh_section section){
  mesh_buffers(context->M)[section.first].constant=1;
}
