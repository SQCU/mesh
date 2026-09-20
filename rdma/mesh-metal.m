#import <Metal/Metal.h>
#include "mesh-metal.h"

/* design/algorithm-sources.md#resident-metal */
/* design/prepared-machine.md#M07 */
int mesh_metal_transport_create(struct mesh_ctx *context,void *device,struct mesh_metal_transport *transport){
  *transport=(struct mesh_metal_transport){0};
  /* design/prepared-machine.md#M17 */
  transport->publication=[(id<MTLDevice>)device newBufferWithBytesNoCopy:(char *)context->M+context->send_off
    length:context->send_bytes
    options:MTLResourceStorageModeShared deallocator:nil];
  struct hdr *m=context->M;
  struct mesh_section stop,words;
  uint64_t inputs=0,block=(uint64_t)m->pgsz*m->block;
  for(uint32_t p=0;p<m->links;p++){
    struct mesh_tx *tx=(void *)mesh_events(m,mesh_notice_queue(m,context->client,p));
    for(uint32_t q=p*m->qps;q<(p+1)*m->qps;q++){
      struct mesh_transfer *in=mesh_transfers(m,context->client,q,MESH_RECEIVE);
      for(uint32_t i=0;i<atomic_load(mesh_order_length(m,context->client,q,MESH_RECEIVE));i++)
        inputs+=(in[i].bytes+block-1)/block*in[i].count*tx->invocations;
    }
  }
  int status=mesh_section_create(context,sizeof(struct mesh_input_status)*(inputs?inputs:1),1,MESH_ABSENT,&words);
  if(status){mesh_metal_transport_destroy(transport);return status;}
  void *input=mesh_section_address(context,words,0);memset(input,0,words.bytes);
  transport->inputs=[(id<MTLDevice>)device newBufferWithBytesNoCopy:input length:words.pages*m->pgsz
    options:MTLResourceStorageModeShared deallocator:nil];
  status=mesh_section_create(context,sizeof(struct mesh_cancellation)+sizeof(struct mesh_cancel_range),1,MESH_ABSENT,&stop);
  if(status){mesh_metal_transport_destroy(transport);return status;}
  struct mesh_cancellation *address=mesh_section_address(context,stop,0);memset(address,0,stop.bytes);
  /* design/prepared-machine.md#M12 */
  address->ranges[0]=(struct mesh_cancel_range){.offset=(uintptr_t)input-(uintptr_t)m,.count=inputs};
  atomic_store_explicit(&address->count,1,memory_order_release);
  uint64_t first=0;
  for(uint32_t p=0;p<m->links;p++){
    struct mesh_tx *tx=(void *)mesh_events(m,mesh_notice_queue(m,context->client,p));
    tx->cancel=(uintptr_t)address-(uintptr_t)m;
    uint64_t frames=0;
    for(uint32_t q=p*m->qps;q<(p+1)*m->qps;q++){
      struct mesh_transfer *in=mesh_transfers(m,context->client,q,MESH_RECEIVE);
      for(uint32_t i=0;i<atomic_load(mesh_order_length(m,context->client,q,MESH_RECEIVE));i++)
        frames+=(in[i].bytes+block-1)/block*in[i].count;
    }
    uint64_t frame=0;
    /* design/prepared-machine.md#M07 */
    for(uint32_t q=p*m->qps;q<(p+1)*m->qps;q++){
      struct mesh_transfer *in=mesh_transfers(m,context->client,q,MESH_RECEIVE);
      for(uint32_t i=0;i<atomic_load(mesh_order_length(m,context->client,q,MESH_RECEIVE));i++)
        for(uint32_t s=0;s<in[i].count;s++)for(uint32_t k=0;k<(in[i].bytes+block-1)/block;k++){
          struct mesh_publication *delivery=mesh_publication_at(m,in[i].local_row+s*in[i].stride+k);
          delivery->device_input=(uintptr_t)input-(uintptr_t)m+sizeof(struct mesh_input_status)*(first+frame++);
          delivery->device_stride=sizeof(struct mesh_input_status)*frames;
        }
    }
    first+=frames*tx->invocations;
  }
  transport->stop=[(id<MTLDevice>)device newBufferWithBytesNoCopy:address length:stop.pages*context->M->pgsz
    options:MTLResourceStorageModeShared deallocator:nil];
  if(!transport->publication||!transport->stop||!transport->inputs){mesh_metal_transport_destroy(transport);return ENOMEM;}
  return 0;
}

/* design/algorithm-sources.md#resident-metal */
/* design/prepared-machine.md#M07 */
int mesh_metal_receive_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,
  struct mesh_section operand,struct mesh_metal_input *input){
  uint64_t block=(uint64_t)context->M->pgsz*context->M->block;
  struct mesh_publication *delivery=mesh_publication_at(context->M,operand.first+(operand.bytes+block-1)/block-1);
  id<MTLBuffer> memory=transport->inputs;
  *input=(struct mesh_metal_input){.completion=[memory retain],.stop=transport->stop,
    .offset=(uintptr_t)context->M+delivery->device_input-(uintptr_t)memory.contents,.stride=delivery->device_stride};
  return 0;
}

/* design/algorithm-sources.md#resident-metal */
/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M10 */
int mesh_metal_publication_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,
  struct mesh_section section,uint32_t invocations,struct mesh_metal_publication *publication){
  uint32_t count=mesh_publication_prepare(context->M,section.first,NULL);
  *publication=(struct mesh_metal_publication){.count=count,.stride=sizeof(struct mesh_send)};
  if(!count)return 0;
  struct mesh_section records;
  int status=mesh_section_create(context,count*sizeof(struct prepared_publication),1,MESH_ABSENT,&records);
  if(status)return status;
  struct prepared_publication *prepared=mesh_section_address(context,records,0);
  mesh_publication_prepare(context->M,section.first,prepared);
  /* design/prepared-machine.md#M17 */
  id<MTLBuffer> memory=transport->publication;
  for(uint32_t i=0;i<count;i++)prepared[i].destination-=((uintptr_t)context->M+context->send_off);
  id<MTLDevice> device=memory.device;
  publication->records=[device newBufferWithBytesNoCopy:prepared length:records.pages*context->M->pgsz
    options:MTLResourceStorageModeShared deallocator:nil];
  /* design/prepared-machine.md#M28 */
  if(invocations && section.bytes<=UINT32_MAX){
    struct mesh_section words;
    status=mesh_section_create(context,sizeof(struct mesh_output_status)*(uint64_t)invocations,1,MESH_ABSENT,&words);
    if(status){[(id)publication->records release];return status;}
    struct mesh_output_status *address=mesh_section_address(context,words,0);
    for(uint32_t t=0;t<invocations;t++)address[t]=(struct mesh_output_status){.remaining=(uint32_t)section.bytes};
    publication->completion=[device newBufferWithBytesNoCopy:address length:words.pages*context->M->pgsz
      options:MTLResourceStorageModeShared deallocator:nil];
    if(!publication->completion){[(id)publication->records release];return ENOMEM;}
  }
  return publication->records?0:ENOMEM;
}

/* design/algorithm-sources.md#resident-metal */
void mesh_metal_transport_destroy(struct mesh_metal_transport *transport){
  [(id)transport->publication release];[(id)transport->stop release];[(id)transport->inputs release];
  *transport=(struct mesh_metal_transport){0};
}
