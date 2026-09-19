#import <Metal/Metal.h>
#include "mesh-metal.h"

/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M02 */
LiteRtStatus mesh_metal_bind(struct mesh_ctx *context,struct mesh_section section,uint32_t slot,
  LiteRtEnvironment environment,void *device,const LiteRtRankedTensorType *type,
  LiteRtTensorBufferType storage,struct mesh_metal_operand *operand){
  *operand=(struct mesh_metal_operand){0};
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
