#import <Metal/Metal.h>
#include "mesh-metal.h"

/* design/algorithm-sources.md#resident-metal */
/* design/prepared-machine.md#M07 */
int mesh_metal_transport_create(struct mesh_ctx *context,void *device,uint32_t invocations,struct mesh_metal_transport *transport){
  NSString *source=@"#include <metal_stdlib>\n"
    "#pragma METAL internals : enable\n"
    "using namespace metal;\n"
    "struct Publication { ulong destination,argument,padding[2]; };\n"
    "kernel void mesh_publish(volatile coherent(system) device uint4 *payload [[buffer(0)]],"
    "constant uint2 &extent [[buffer(1)]],constant Publication *records [[buffer(2)]],"
    "uint lane [[thread_index_in_threadgroup]]) {"
    "for(uint i=lane;i<extent.x;i+=256)payload[i]=payload[i];"
    "atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,static_cast<thread_scope>(3));"
    "threadgroup_barrier(mem_flags::mem_device);"
    "for(uint i=lane;i<extent.y;i+=256) {"
    "auto destination=(volatile coherent(system) device ulong *)records[i].destination;"
    "*destination=records[i].argument;"
    "}"
    "atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,static_cast<thread_scope>(3));"
    "}\n";
  *transport=(struct mesh_metal_transport){0};
  NSError *error=nil;
  id<MTLLibrary> library=[(id<MTLDevice>)device newLibraryWithSource:source options:nil error:&error];
  if(!library){fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);return EINVAL;}
  id<MTLFunction> function=[library newFunctionWithName:@"mesh_publish"];
  MTLComputePipelineDescriptor *descriptor=[MTLComputePipelineDescriptor new];
  descriptor.computeFunction=function;descriptor.supportIndirectCommandBuffers=YES;
  transport->publish=[(id<MTLDevice>)device newComputePipelineStateWithDescriptor:descriptor options:0 reflection:nil error:&error];
  [descriptor release];[function release];[library release];
  if(!transport->publish){fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);return EINVAL;}
  /* design/prepared-machine.md#M17 */
  transport->publication=[(id<MTLDevice>)device newBufferWithBytesNoCopy:(char *)context->M+context->M->notice_off
    length:context->M->data_off-context->M->notice_off
    options:MTLResourceStorageModeShared deallocator:nil];
  struct mesh_section stop;
  int status=mesh_section_create(context,2*sizeof(uint32_t),1,MESH_ABSENT,&stop);
  if(status){mesh_metal_transport_destroy(transport);return status;}
  uint32_t *address=mesh_section_address(context,stop,0);address[0]=address[1]=0;
  /* design/prepared-machine.md#M12 */
  for(uint32_t p=0;p<context->M->links;p++){
    struct mesh_tx *tx=(void *)mesh_events(context->M,mesh_notice_queue(context->M,context->client,p));
    tx->cancel=(uintptr_t)address-(uintptr_t)context->M;tx->invocations=invocations;
  }
  transport->stop=[(id<MTLDevice>)device newBufferWithBytesNoCopy:address length:context->M->pgsz
    options:MTLResourceStorageModeShared deallocator:nil];
  if(!transport->publication||!transport->stop){mesh_metal_transport_destroy(transport);return ENOMEM;}
  return 0;
}

/* design/algorithm-sources.md#resident-metal */
/* design/prepared-machine.md#M07 */
int mesh_metal_receive_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,
  struct mesh_section operand,uint32_t invocations,struct mesh_metal_input *input){
  struct mesh_section words;
  int status=mesh_section_create(context,8*(uint64_t)invocations*operand.count,1,MESH_ABSENT,&words);
  if(status)return status;
  void *address=mesh_section_address(context,words,0);
  memset(address,0,words.bytes);
  for(uint32_t s=0;s<operand.count;s++)for(uint32_t k=0;k<operand.stride;k++)
    mesh_publication_at(context->M,operand.first+s*operand.stride+k)->device_input=
      (uintptr_t)address-(uintptr_t)context->M+8*(uint64_t)s*invocations;
  id<MTLDevice> device=[(id<MTLBuffer>)transport->publication device];
  *input=(struct mesh_metal_input){.completion=[device newBufferWithBytesNoCopy:address
    length:words.pages*context->M->pgsz options:MTLResourceStorageModeShared deallocator:nil],
    .stop=transport->stop,.stride=8};
  return input->completion?0:ENOMEM;
}

/* design/algorithm-sources.md#resident-metal */
/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M10 */
void mesh_metal_publish_encode(struct mesh_ctx *context,struct mesh_metal_transport *transport,
  void *command,void *operand,struct mesh_section section){
  id<MTLComputeCommandEncoder> encoder=command;
  id<MTLBuffer> payload=operand;
  uint32_t extent[]={(uint32_t)((section.bytes+15)/16),mesh_publication_prepare(context->M,section.first,NULL)};
  if(!extent[1])return;
  struct mesh_section records;
  int status=mesh_section_create(context,extent[1]*sizeof(struct prepared_publication),1,MESH_ABSENT,&records);
  if(status)[NSException raise:NSMallocException format:@"publication allocation: %d",status];
  struct prepared_publication *prepared=mesh_section_address(context,records,0);
  mesh_publication_prepare(context->M,section.first,prepared);
  /* design/prepared-machine.md#M17 */
  id<MTLBuffer> memory=transport->publication;
  for(uint32_t i=0;i<extent[1];i++)prepared[i].destination=memory.gpuAddress+prepared[i].destination-(uintptr_t)context->M-context->M->notice_off;
  id<MTLDevice> device=memory.device;
  id<MTLBuffer> bindings=[device newBufferWithBytesNoCopy:prepared length:records.pages*context->M->pgsz
    options:MTLResourceStorageModeShared deallocator:nil];
  [encoder setComputePipelineState:transport->publish];
  [encoder setBuffer:payload offset:0 atIndex:0];[encoder setBytes:extent length:sizeof extent atIndex:1];
  [encoder setBuffer:bindings offset:0 atIndex:2];
  [encoder useResource:memory usage:MTLResourceUsageWrite];
  [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [bindings release];
}

/* design/algorithm-sources.md#resident-metal */
void mesh_metal_transport_destroy(struct mesh_metal_transport *transport){
  [(id)transport->publication release];[(id)transport->publish release];[(id)transport->stop release];
  *transport=(struct mesh_metal_transport){0};
}
