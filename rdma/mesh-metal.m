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
  struct mesh_section stop;
  uint64_t inputs=0;
  for(uint32_t q=0;q<context->M->links*context->M->qps;q++)
    inputs+=atomic_load(mesh_order_length(context->M,context->client,q,MESH_RECEIVE));
  int status=mesh_section_create(context,sizeof(struct mesh_cancellation)+inputs*sizeof(struct mesh_cancel_range),1,MESH_ABSENT,&stop);
  if(status){mesh_metal_transport_destroy(transport);return status;}
  struct mesh_cancellation *address=mesh_section_address(context,stop,0);memset(address,0,stop.bytes);
  /* design/prepared-machine.md#M12 */
  for(uint32_t p=0;p<context->M->links;p++){
    struct mesh_tx *tx=(void *)mesh_events(context->M,mesh_notice_queue(context->M,context->client,p));
    tx->cancel=(uintptr_t)address-(uintptr_t)context->M;
  }
  transport->stop=[(id<MTLDevice>)device newBufferWithBytesNoCopy:address length:stop.pages*context->M->pgsz
    options:MTLResourceStorageModeShared deallocator:nil];
  if(!transport->publication||!transport->stop){mesh_metal_transport_destroy(transport);return ENOMEM;}
  return 0;
}

/* design/algorithm-sources.md#resident-metal */
/* design/prepared-machine.md#M07 */
int mesh_metal_receive_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,
  struct mesh_section operand,uint32_t invocations,struct mesh_metal_input *input){
  struct mesh_section words;
  int status=mesh_section_create(context,sizeof(struct mesh_input_status)*(uint64_t)invocations*operand.count,1,MESH_ABSENT,&words);
  if(status)return status;
  void *address=mesh_section_address(context,words,0);
  memset(address,0,words.bytes);
  /* design/prepared-machine.md#M12 */
  struct mesh_cancellation *cancel=[(id<MTLBuffer>)transport->stop contents];
  uint32_t count=atomic_load_explicit(&cancel->count,memory_order_relaxed);
  cancel->ranges[count]=(struct mesh_cancel_range){.offset=(uintptr_t)address-(uintptr_t)context->M,.count=(uint64_t)invocations*operand.count};
  atomic_store_explicit(&cancel->count,count+1,memory_order_release);
  if(atomic_load_explicit(&cancel->requested,memory_order_relaxed))mesh_cancel(context->M,cancel);
  uint64_t block=(uint64_t)context->M->pgsz*context->M->block;
  for(uint32_t s=0;s<operand.count;s++)for(uint32_t k=0;k<(operand.bytes+block-1)/block;k++)
    mesh_publication_at(context->M,operand.first+s*operand.stride+k)->device_input=
      (uintptr_t)address-(uintptr_t)context->M+sizeof(struct mesh_input_status)*(uint64_t)s*invocations;
  id<MTLDevice> device=[(id<MTLBuffer>)transport->publication device];
  *input=(struct mesh_metal_input){.completion=[device newBufferWithBytesNoCopy:address
    length:words.pages*context->M->pgsz options:MTLResourceStorageModeShared deallocator:nil],
    .stop=transport->stop,.stride=sizeof(struct mesh_input_status)};
  return input->completion?0:ENOMEM;
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
  [(id)transport->publication release];[(id)transport->stop release];
  *transport=(struct mesh_metal_transport){0};
}
