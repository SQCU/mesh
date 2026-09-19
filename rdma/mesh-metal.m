#import <Metal/Metal.h>
#include "mesh-metal.h"

/* design/algorithm-sources.md#resident-metal */
/* design/prepared-machine.md#M07 */
int mesh_metal_transport_create(struct mesh_ctx *context,void *device,struct mesh_metal_transport *transport){
  NSString *source=@"#include <metal_stdlib>\n"
    "#pragma METAL internals : enable\n"
    "using namespace metal;\n"
    "struct Publication { ulong destination,argument,padding[2]; };\n"
    "kernel void mesh_publish(volatile coherent(system) device uint *payload [[buffer(0)]],"
    "constant uint2 &extent [[buffer(1)]],constant Publication *records [[buffer(2)]],"
    "uint lane [[thread_index_in_threadgroup]]) {"
    "for(uint i=lane;i<extent.x;i+=32)payload[i]=payload[i];"
    "atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,static_cast<thread_scope>(3));"
    "threadgroup_barrier(mem_flags::mem_device);"
    "for(uint i=lane;i<extent.y;i+=32) {"
    "auto destination=(volatile coherent(system) device ulong *)records[i].destination;"
    "*destination=records[i].argument;"
    "}"
    "atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,static_cast<thread_scope>(3));"
    "}\n"
    "kernel void mesh_consume(volatile coherent(system) device ulong *completion [[buffer(0)]],"
    "volatile coherent(system) device uint *stop [[buffer(1)]]) {"
    "for(;;) {"
    "atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,static_cast<thread_scope>(3));"
    "if(*completion)break; if(*stop){stop[1]=1;return;}"
    "}"
    "*completion=0;"
    "atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,static_cast<thread_scope>(3));"
    "}\n";
  *transport=(struct mesh_metal_transport){0};
  NSError *error=nil;
  id<MTLLibrary> library=[(id<MTLDevice>)device newLibraryWithSource:source options:nil error:&error];
  if(!library){fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);return EINVAL;}
  void **pipelines[]={&transport->publish,&transport->consume};
  NSArray *names=@[@"mesh_publish",@"mesh_consume"];
  for(unsigned i=0;i<2;i++){
    id<MTLFunction> function=[library newFunctionWithName:names[i]];
    MTLComputePipelineDescriptor *descriptor=[MTLComputePipelineDescriptor new];
    descriptor.computeFunction=function;descriptor.supportIndirectCommandBuffers=YES;
    *pipelines[i]=[(id<MTLDevice>)device newComputePipelineStateWithDescriptor:descriptor options:0 reflection:nil error:&error];
    [descriptor release];[function release];
    if(!*pipelines[i]){fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);[library release];mesh_metal_transport_destroy(transport);return EINVAL;}
  }
  [library release];
  transport->completion=[(id<MTLDevice>)device newBufferWithBytesNoCopy:context->M length:context->M->notice_off
    options:MTLResourceStorageModeShared deallocator:nil];
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
    tx->cancel=(uintptr_t)address-(uintptr_t)context->M;
  }
  transport->stop=[(id<MTLDevice>)device newBufferWithBytesNoCopy:address length:context->M->pgsz
    options:MTLResourceStorageModeShared deallocator:nil];
  if(!transport->completion||!transport->publication||!transport->stop){mesh_metal_transport_destroy(transport);return ENOMEM;}
  return 0;
}

/* design/algorithm-sources.md#resident-metal */
/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M07 */
void mesh_metal_transfer_encode(struct mesh_ctx *context,struct mesh_metal_transport *transport,
  void *command,void *operand,struct mesh_section section,int receive){
  id<MTLComputeCommandEncoder> encoder=command;
  id<MTLBuffer> payload=operand;
  struct mesh_publication *publication=mesh_publication_at(context->M,section.first);
  if(receive){
    publication->device_input=1;
    [encoder setComputePipelineState:transport->consume];
    [encoder setBuffer:transport->completion offset:(uintptr_t)&publication->argument-(uintptr_t)context->M atIndex:0];
    [encoder setBuffer:transport->stop offset:0 atIndex:1];
    [encoder useResource:payload usage:MTLResourceUsageWrite];
    [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
  }else{
    uint32_t extent[]={(uint32_t)((section.bytes+3)/4),mesh_publication_prepare(context->M,section.first,NULL)};
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
    [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [bindings release];
  }
}

/* design/algorithm-sources.md#resident-metal */
void mesh_metal_transport_destroy(struct mesh_metal_transport *transport){
  [(id)transport->completion release];[(id)transport->publication release];[(id)transport->publish release];
  [(id)transport->consume release];[(id)transport->stop release];
  *transport=(struct mesh_metal_transport){0};
}
