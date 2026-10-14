#import <Metal/Metal.h>
#include "mesh-metal.h"
#include "mesh-collective.h"

/* design/algorithm-sources.md#resident-metal */
/* design/prepared-machine.md#M07 */
int mesh_metal_transport_create(struct mesh_ctx *context,void *device,struct mesh_metal_transport *transport){
  *transport=(struct mesh_metal_transport){0};
  /* design/prepared-machine.md#M17 */
  transport->publication=[(id<MTLDevice>)device newBufferWithBytesNoCopy:(char *)context->M+context->send_off
    length:context->send_bytes
    options:MTLResourceStorageModeShared deallocator:nil];
  /* completion words are written by the host after a completion and read by the GPU; never an SGE target */
  int status=mesh_host_inputs(context);
  if(status){mesh_metal_transport_destroy(transport);return status;}
  transport->inputs=[(id<MTLDevice>)device newBufferWithBytesNoCopy:(char *)context->M+context->inputs_off length:context->inputs_bytes
    options:MTLResourceStorageModeShared deallocator:nil];
  transport->cancel=(void *)((char *)context->M+mesh_client_at(context->M,context->client)->tx[0].cancel);
  if(!transport->publication||!transport->inputs){mesh_metal_transport_destroy(transport);return ENOMEM;}
  return 0;
}

/* design/algorithm-sources.md#resident-metal */
/* design/prepared-machine.md#M07 */
int mesh_metal_receive_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,
  struct mesh_section operand,struct mesh_metal_input *input){
  struct mesh_publication *delivery=mesh_publication_at(context->M,operand.first+mesh_row_chunks(context->M,operand.first,operand.bytes)-1);
  id<MTLBuffer> memory=transport->inputs;
  *input=(struct mesh_metal_input){.completion=[memory retain],
    .offset=(uintptr_t)context->M+delivery->device_input-(uintptr_t)memory.contents,.stride=delivery->device_stride};
  return 0;
}

/* design/algorithm-sources.md#resident-metal */
/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M10 */
int mesh_metal_publication_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,
  const struct mesh_section *sections,uint32_t ranges,uint32_t invocations,struct mesh_metal_publication *publication){
  uint32_t count=mesh_publication_prepare(context->M,sections[0].first,NULL);
  *publication=(struct mesh_metal_publication){.count=count*ranges,.stride=sizeof(struct mesh_send)};
  if(!count)return 0;
  struct mesh_section records;
  /* design/prepared-machine.md#M10 */
  int status=mesh_section_create(context,count*ranges*sizeof(struct prepared_publication),1,0,&records);
  if(status)return status;
  struct prepared_publication *prepared=mesh_section_address(context,records,0);
  for(uint32_t j=0;j<ranges;j++)mesh_publication_prepare(context->M,sections[j].first,prepared+j*count);
  /* design/prepared-machine.md#M17 */
  id<MTLBuffer> memory=transport->publication;
  for(uint32_t i=0;i<count*ranges;i++)prepared[i].destination-=((uintptr_t)context->M+context->send_off);
  id<MTLDevice> device=memory.device;
  publication->records=[device newBufferWithBytesNoCopy:prepared length:records.pages*context->M->pgsz
    options:MTLResourceStorageModeShared deallocator:nil];
  /* design/prepared-machine.md#M28 */
  if(invocations && sections[0].bytes<=UINT32_MAX){
    struct mesh_section words;
    /* design/prepared-machine.md#M28 */
    status=mesh_section_create(context,sizeof(struct mesh_output_status)*(uint64_t)(invocations+1)*ranges,1,0,&words);
    if(status){[(id)publication->records release];return status;}
    struct mesh_output_status *address=mesh_section_address(context,words,0);
    for(uint32_t j=0;j<ranges;j++)for(uint32_t t=0;t<=invocations;t++)
      address[(uint64_t)j*(invocations+1)+t]=(struct mesh_output_status){.remaining=(uint32_t)sections[j].bytes};
    publication->completion=[device newBufferWithBytesNoCopy:address length:words.pages*context->M->pgsz
      options:MTLResourceStorageModeShared deallocator:nil];
    if(!publication->completion){[(id)publication->records release];return ENOMEM;}
  }
  return publication->records?0:ENOMEM;
}

/* design/algorithm-sources.md#resident-metal */
void mesh_metal_transport_destroy(struct mesh_metal_transport *transport){
  [(id)transport->publication release];[(id)transport->inputs release];
  *transport=(struct mesh_metal_transport){0};
}
