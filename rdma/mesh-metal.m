#import <Metal/Metal.h>
#include "mesh-metal.h"

/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M02 */
LiteRtStatus mesh_metal_bind(struct mesh_ctx *context,struct mesh_section section,uint32_t slot,
  LiteRtEnvironment environment,void *device,const LiteRtRankedTensorType *type,
  LiteRtTensorBufferType storage,struct mesh_metal_operand *operand){
  *operand=(struct mesh_metal_operand){0};
  if(storage==kLiteRtTensorBufferTypeHostMemory)
    return LiteRtCreateTensorBufferFromHostMemory(type,mesh_section_address(context,section,slot),
      section.bytes,NULL,&operand->tensor);
  size_t page=context->M->pgsz;
  id<MTLBuffer> buffer=[(id<MTLDevice>)device
    newBufferWithBytesNoCopy:mesh_section_address(context,section,slot)
    length:(section.bytes+page-1)/page*page options:MTLResourceStorageModeShared
    deallocator:nil];
  if(!buffer)return kLiteRtStatusErrorMemoryAllocationFailure;
  LiteRtStatus status=LiteRtCreateTensorBufferFromMetalMemory(environment,type,storage,
    buffer,section.bytes,NULL,&operand->tensor);
  if(status){[buffer release];return status;}
  operand->buffer=buffer;
  return kLiteRtStatusOk;
}

/* design/algorithm-sources.md#programtensor */
void mesh_metal_unbind(struct mesh_metal_operand *operand){
  if(operand->tensor)LiteRtDestroyTensorBuffer(operand->tensor);
  [(id<MTLBuffer>)operand->buffer release];
  *operand=(struct mesh_metal_operand){0};
}

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
    "for(uint i=lane;i<extent.x;i+=256)payload[i]=payload[i];"
    "atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,static_cast<thread_scope>(3));"
    "threadgroup_barrier(mem_flags::mem_device);"
    "for(uint i=lane;i<extent.y;i+=256) {"
    "auto destination=(volatile coherent(system) device ulong *)records[i].destination;"
    "*destination=records[i].argument;"
    "}"
    "atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,static_cast<thread_scope>(3));"
    "}\n"
    "kernel void mesh_consume(volatile coherent(system) device ulong *completion [[buffer(0)]],"
    "volatile coherent(system) device uint *stop [[buffer(1)]]) {"
    "for(;;) {"
    "atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,static_cast<thread_scope>(3));"
    "if(*completion)break; if(*stop)return;"
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
  transport->memory=[(id<MTLDevice>)device newBufferWithBytesNoCopy:context->M length:context->M->data_off
    options:MTLResourceStorageModeShared deallocator:nil];
  struct mesh_section stop;
  int status=mesh_section_create(context,4,1,MESH_ABSENT,&stop);
  if(status){mesh_metal_transport_destroy(transport);return status;}
  uint32_t *address=mesh_section_address(context,stop,0);*address=0;
  transport->stop=[(id<MTLDevice>)device newBufferWithBytesNoCopy:address length:context->M->pgsz
    options:MTLResourceStorageModeShared deallocator:nil];
  if(!transport->memory||!transport->stop){mesh_metal_transport_destroy(transport);return ENOMEM;}
  return 0;
}

/* design/algorithm-sources.md#resident-metal */
/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M07 */
void mesh_metal_transfer_encode(struct mesh_ctx *context,struct mesh_metal_transport *transport,
  void *command,struct mesh_section section,int receive){
  id<MTLComputeCommandEncoder> encoder=command;
  struct mesh_publication *publication=mesh_publication_at(context->M,section.first);
  if(receive){
    publication->device_input=1;
    [encoder setComputePipelineState:transport->consume];
    [encoder setBuffer:transport->memory offset:(uintptr_t)&publication->argument-(uintptr_t)context->M atIndex:0];
    [encoder setBuffer:transport->stop offset:0 atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
  }else{
    uint32_t extent[]={(uint32_t)((section.bytes+3)/4),publication->sends};
    struct mesh_section records;
    int status=mesh_section_create(context,extent[1]*sizeof(struct prepared_publication),1,MESH_ABSENT,&records);
    if(status)[NSException raise:NSMallocException format:@"publication allocation: %d",status];
    struct prepared_publication *prepared=mesh_section_address(context,records,0);
    mesh_publication_prepare(context->M,section.first,prepared);
    id<MTLBuffer> memory=transport->memory;
    for(uint32_t i=0;i<extent[1];i++)prepared[i].destination=memory.gpuAddress+prepared[i].destination-(uintptr_t)context->M;
    id<MTLDevice> device=memory.device;
    id<MTLBuffer> payload=[device newBufferWithBytesNoCopy:mesh_section_address(context,section,0)
      length:(section.bytes+context->M->pgsz-1)/context->M->pgsz*context->M->pgsz options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> bindings=[device newBufferWithBytesNoCopy:prepared length:records.pages*context->M->pgsz
      options:MTLResourceStorageModeShared deallocator:nil];
    [encoder setComputePipelineState:transport->publish];
    [encoder setBuffer:payload offset:0 atIndex:0];[encoder setBytes:extent length:sizeof extent atIndex:1];
    [encoder setBuffer:bindings offset:0 atIndex:2];
    [encoder useResource:memory usage:MTLResourceUsageWrite];
    [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [payload release];[bindings release];
  }
}

/* design/algorithm-sources.md#resident-metal */
void mesh_metal_transport_destroy(struct mesh_metal_transport *transport){
  [(id)transport->memory release];[(id)transport->publish release];
  [(id)transport->consume release];[(id)transport->stop release];
  *transport=(struct mesh_metal_transport){0};
}
