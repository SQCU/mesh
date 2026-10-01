#include "mesh-call.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* design/algorithm-sources.md#programtensor */
static uint32_t mesh_section_row(struct mesh_section section,uint32_t index){return section.first+index*section.stride;}

/* design/prepared-machine.md#M09 */
/* A bound row is an SGE target, so it must lie in the registered window.  Refusing it here is the
   whole enforcement of the split: an operand allocated out of the unregistered arena fails at bind
   with EINVAL instead of reaching the wire with a lkey that does not cover it. */
static uint64_t mesh_section_rows(struct mesh_section section){
  return section.stride?(uint64_t)section.stride*section.count:1;
}
static int mesh_section_wired(struct hdr *m,struct mesh_section section){
  uint64_t block=(uint64_t)m->block*m->pgsz,rows=mesh_section_rows(section);
  for(uint64_t i=0;i<rows;i++){
    struct mesh_page_entry *entry=&mesh_page(m)[section.first+i];
    if(atomic_load_explicit(&entry->mapping,memory_order_relaxed)==MESH_ABSENT)continue;
    uint64_t offset=atomic_load_explicit(&entry->address,memory_order_relaxed)-m->data_off;
    if(!mesh_wired(m,offset,block-offset%block))return 0;
  }
  return 1;
}

/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#programcopy */
int mesh_transfer_bind(struct mesh_ctx *context,uint32_t queue,int receive,uint32_t identity,struct mesh_section section,uint32_t invocation_pages){
  struct hdr *m=context->M;
  if(queue>=m->links*m->qps)return EINVAL;
  if(!section.count || (uint64_t)section.first+mesh_section_rows(section)>mesh_rows(m))return EINVAL;
  /* design/prepared-machine.md#M09 */
  if(!mesh_section_wired(m,section)){
    fprintf(stderr,"transfer binding %u queue %u %s outside the registered window: rows %u+%u*%u, window %llu bytes\n",
      identity,queue,receive?"receive":"send",section.first,section.stride,section.count,
      (unsigned long long)mesh_wire_bytes(m));
    return EINVAL;
  }
  _Atomic uint32_t *length=mesh_order_length(m,context->client,queue,receive);
  uint32_t index=atomic_load_explicit(length,memory_order_relaxed);
  if(index==m->orders)return ENOSPC;
  if(!receive)for(uint32_t value=0;value<section.count;value++){
    uint32_t row=mesh_section_row(section,value);
    uint32_t link=queue/m->qps;
    mesh_publish_bind(context,row,link)->count++;
  }
  mesh_transfers(m,context->client,queue,receive)[index]=(struct mesh_transfer){section.first,identity,section.count,section.stride,invocation_pages,0,section.bytes};
  atomic_store_explicit(length,index+1,memory_order_release);
  return 0;
}

/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#programcopy */
static int mesh_binding_order(const void *a,const void *b){
  const struct mesh_transfer *left=a,*right=b;
  int varying=(left->stride!=0)-(right->stride!=0);
  return varying?varying:(left->binding>right->binding)-(left->binding<right->binding);
}

/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M07 */
/* design/algorithm-sources.md#programcopy */
static int mesh_transfer_compare(const void *a,const void *b){
  return mesh_binding_order(*(const struct mesh_transfer *const *)a,*(const struct mesh_transfer *const *)b);
}

/* design/algorithm-sources.md#programcopy */
int mesh_transfers_prepare(struct mesh_ctx *context,uint32_t slots,uint32_t invocations,uint32_t depth){
  struct hdr *m=context->M;
  if(!slots || slots>mesh_rows(m) || !invocations)return EINVAL;
  /* design/prepared-machine.md#M01 */
  /* The ring depth is one run-wide policy value and there is no room for it in the 32-byte
     mesh_transfer or mesh_tx, so it is published in the header, before mesh_transfers_start hands
     the client to the bridge.  depth==invocations is the unrung machine. */
  if(!depth || depth>invocations)depth=invocations;
  atomic_store_explicit(&m->depth,depth,memory_order_relaxed);
  uint64_t cells=0;
  for(uint32_t q=0;q<m->links*m->qps;q++){
    /* design/prepared-machine.md#M08 */
    for(int d=0;d<2;d++)qsort(mesh_transfers(m,context->client,q,d),
      atomic_load(mesh_order_length(m,context->client,q,d)),sizeof(struct mesh_transfer),mesh_binding_order);
    struct mesh_transfer *out=mesh_transfers(m,context->client,q,MESH_SEND);
    for(uint32_t i=0;i<atomic_load(mesh_order_length(m,context->client,q,MESH_SEND));i++)
      cells+=out[i].stride?slots:1;
  }
  struct mesh_section storage;
  uint64_t column=(uint64_t)invocations+1;
  /* design/prepared-machine.md#M04 */
  /* The SEND cell array is read by the provider and written by the GPU; it never appears in an
     SGE, so it is allocated outside the registered window. */
  int status=mesh_section_create(context,(cells?cells:1)*column*sizeof(struct mesh_send),1,0,&storage);
  if(status)return status;
  void *address=mesh_section_address(context,storage,0);
  memset(address,0,storage.bytes);
  context->send_off=(uintptr_t)address-(uintptr_t)m;
  context->send_bytes=(uint64_t)storage.pages*m->pgsz;
  uint64_t first=context->send_off;
  /* design/prepared-machine.md#M04 */
  for(uint32_t p=0;p<m->links;p++){
    uint64_t base=m->notice_off+(uint64_t)mesh_notice_queue(m,context->client,p)*m->notice_bytes;
    struct mesh_tx *tx=(void *)((char *)m+base);
    uint32_t count=0,once=0,length=0,incoming=0;
    for(uint32_t q=0;q<m->qps;q++){
      length+=atomic_load_explicit(mesh_order_length(m,context->client,p*m->qps+q,MESH_SEND),memory_order_relaxed);
      incoming+=atomic_load_explicit(mesh_order_length(m,context->client,p*m->qps+q,MESH_RECEIVE),memory_order_relaxed);
    }
    uint32_t capacity=length>incoming?length:incoming;
    struct mesh_transfer **ordered=malloc((capacity?capacity:1)*sizeof *ordered);
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
    *tx=(struct mesh_tx){.count=count,.slots=slots,.once=once,.invocations=invocations,.cells=first};
    first+=((uint64_t)count*slots+once)*column*sizeof(struct mesh_send);
    qsort(ordered,length,sizeof *ordered,mesh_transfer_compare);
    uint32_t next[2]={0,once};
    for(uint32_t i=0;i<length;i++){
      struct mesh_transfer *out=ordered[i];
      uint32_t varying=out->stride!=0;
      out->first=next[varying];
      for(uint32_t slot=0;slot<out->count;slot++){
        struct mesh_publication *publication=mesh_publication_at(m,out->local_row+slot*out->stride);
        for(uint32_t j=0;j<publication->sends;j++)if(publication->targets[j].stream==base){
          publication->targets[j].stream=tx->cells+sizeof(struct mesh_send)*column*(slot*count+next[varying]);
          publication->targets[j].stride=(uint32_t)column;
        }
      }
      next[varying]++;
    }
    /* design/prepared-machine.md#M04 */
    struct mesh_send *last[m->qps*slots],*repeat[m->qps*slots];
    memset(last,0,sizeof last);memset(repeat,0,sizeof repeat);
    for(uint32_t varying=0;varying<2;varying++)for(uint32_t i=0;i<length;i++){
      struct mesh_transfer *out=ordered[i];
      if((out->stride!=0)!=varying)continue;
      uint32_t q=(uint32_t)(out-mesh_transfers(m,context->client,p*m->qps,MESH_SEND))/(2*m->orders);
      for(uint32_t slot=0;slot<out->count;slot++){
        uint32_t stream=q*slots+slot;
        struct mesh_send *cell=(void *)((char *)m+tx->cells+sizeof(struct mesh_send)*column*(slot*count+out->first));
        if(last[stream])last[stream]->request.wr_id=(uintptr_t)cell-(uintptr_t)last[stream];
        last[stream]=cell;
        if(varying&&!repeat[stream])repeat[stream]=cell;
      }
    }
    for(uint32_t stream=0;stream<m->qps*slots;stream++)if(last[stream]){
      last[stream]->request.wr_id=(uintptr_t)(repeat[stream]?repeat[stream]+1:last[stream]+invocations)-(uintptr_t)last[stream];
      last[stream]->request.send_flags=IBV_SEND_SIGNALED;
    }
    /* design/prepared-machine.md#M07 */
    position=0;
    for(uint32_t q=0;q<m->qps;q++){
      struct mesh_transfer *in=mesh_transfers(m,context->client,p*m->qps+q,MESH_RECEIVE);
      uint32_t count=atomic_load_explicit(mesh_order_length(m,context->client,p*m->qps+q,MESH_RECEIVE),memory_order_relaxed);
      for(uint32_t i=0;i<count;i++)ordered[position++]=in+i;
    }
    qsort(ordered,position,sizeof *ordered,mesh_transfer_compare);
    uint32_t frame=0;
    for(uint32_t i=0;i<position;i++){
      ordered[i]->first=frame;
      frame+=mesh_row_chunks(m,ordered[i]->local_row,ordered[i]->bytes)*ordered[i]->count;
    }
    free(ordered);
  }

  return 0;
}

/* design/prepared-machine.md#M06 */
/* design/algorithm-sources.md#programcopy */
int mesh_transfers_start(struct mesh_ctx *context){
  struct hdr *m=context->M;
  /* design/prepared-machine.md#M12 */
  uint64_t idle=0;
  if(!atomic_compare_exchange_strong_explicit(&m->configured,&idle,context->client,memory_order_release,memory_order_relaxed))return ECANCELED;
  /* design/prepared-machine.md#M26 */
  mesh_control_notify(m);
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
int mesh_section_create(struct mesh_ctx *context,size_t bytes,uint32_t count,int wire,struct mesh_section *section){
  struct hdr *m=context->M;
  if(!bytes || !count)return EINVAL;
  size_t quantum=(size_t)m->block*m->pgsz;
  /* design/prepared-machine.md#M09 */
  size_t capacity=mesh_arena_range(m,wire).count;
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
  /* design/prepared-machine.md#M01 */
  /* design/prepared-machine.md#M03 */
  uint32_t page=mesh_arena_alloc(context,(uint32_t)span*count,m->block,wire);
  /* design/prepared-machine.md#M01 */
  /* Registration wires its pages eagerly, so until now every operand was resident by the time the
     bridge came up.  Unregistered pages are not, and their first fault would land in the measured
     region; touch the span here, at preparation, where it costs nothing that is being timed. */
  if(!wire && page!=MESH_ABSENT)
    for(uint32_t p=0;p<(uint32_t)span*count;p++)*(volatile unsigned char *)mesh_at(m,page+p)=0;
  if(page==MESH_ABSENT){
    int error=errno;
    mesh_bits_clear(mesh_plane(m,MESH_ROW_HOT),first,rows);
    for(uint32_t r=first;r<first+rows;r+=stride)mesh_buffers(m)[r].pages=0;
    mesh_rows_release(context,first,rows);return error;
  }
  for(uint32_t slot=0;slot<count;slot++)
    mesh_backing_bind(context,first+slot*stride,(uint32_t)span,page+slot*(uint32_t)span,slot);
  *section=(struct mesh_section){first,(uint32_t)span,bytes,count,stride};
  return 0;
}
/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M02 */
int mesh_section_slice(struct mesh_ctx *context,struct mesh_section source,size_t offset,size_t bytes,
  uint32_t invocations,uint32_t invocation_pages,struct mesh_section *section){
  struct hdr *m=context->M;
  uint64_t block=(uint64_t)m->block*m->pgsz,span=(uint64_t)invocation_pages*m->pgsz;
  if(!bytes || !invocations || !invocation_pages || invocation_pages%m->block ||
     offset>span || bytes>span-offset || (uint64_t)invocations*invocation_pages>source.pages)return EINVAL;
  uint32_t chunks=(uint32_t)((offset%block+bytes+block-1)/block);
  uint64_t stride=(uint64_t)chunks*invocations;
  if(stride*source.count>mesh_rows(m))return ENOMEM;
  uint32_t first=mesh_rows_alloc(context,(uint32_t)stride*source.count);
  if(first==MESH_ABSENT)return errno;
  for(uint32_t s=0;s<source.count;s++)for(uint32_t t=0;t<invocations;t++)for(uint32_t k=0;k<chunks;k++){
    struct mesh_page_entry *from=&mesh_page(m)[source.first+s*source.stride+(uint64_t)t*invocation_pages/m->block+offset/block+k];
    struct mesh_page_entry *to=&mesh_page(m)[first+s*stride+(uint64_t)t*chunks+k];
    atomic_store_explicit(&to->mapping,atomic_load_explicit(&from->mapping,memory_order_relaxed),memory_order_relaxed);
    atomic_store_explicit(&to->address,atomic_load_explicit(&from->address,memory_order_relaxed)+(k?0:offset%block),memory_order_relaxed);
  }
  *section=(struct mesh_section){first,chunks*m->block,bytes,source.count,(uint32_t)stride};
  return 0;
}

/* design/algorithm-sources.md#programtensor */
void *mesh_section_address(struct mesh_ctx *context,struct mesh_section section,uint32_t index){return (char *)context->M+atomic_load_explicit(&mesh_page(context->M)[mesh_section_row(section,index)].address,memory_order_relaxed);}
/* design/algorithm-sources.md#programwrite */
void mesh_section_constant(struct mesh_ctx *context,struct mesh_section section){
  mesh_buffers(context->M)[section.first].constant=1;
}
