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
struct mesh_metal_input mesh_metal_receive_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,struct mesh_section operand){
  struct mesh_delivery *delivery=mesh_delivery_at(context->M,operand.first+mesh_row_chunks(context->M,operand.first,operand.bytes)-1);
  return (struct mesh_metal_input){(uintptr_t)context->M+delivery->device_input-(uintptr_t)((id<MTLBuffer>)transport->inputs).contents,
    delivery->device_stride};
}

/* design/algorithm-sources.md#resident-metal */
void mesh_metal_transport_destroy(struct mesh_metal_transport *transport){
  [(id)transport->publication release];[(id)transport->inputs release];
  *transport=(struct mesh_metal_transport){0};
}
