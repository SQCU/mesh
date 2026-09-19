#include "mesh-webgpu.h"

/* design/algorithm-sources.md#programtensor */
static void mesh_webgpu_borrowed(void *context){(void)context;}

/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M02 */
LiteRtStatus mesh_webgpu_bind(struct mesh_ctx *context,struct mesh_section section,uint32_t slot,
  LiteRtEnvironment environment,WGPUDevice device,const LiteRtRankedTensorType *type,
  LiteRtTensorBufferType storage,struct mesh_webgpu_operand *operand){
  *operand=(struct mesh_webgpu_operand){0};
  size_t page=context->M->pgsz;
  WGPUBufferHostMappedPointer memory=WGPU_BUFFER_HOST_MAPPED_POINTER_INIT;
  memory.pointer=mesh_section_address(context,section,slot);
  memory.disposeCallback=mesh_webgpu_borrowed;
  WGPUBufferDescriptor descriptor=WGPU_BUFFER_DESCRIPTOR_INIT;
  descriptor.nextInChain=&memory.chain;
  descriptor.size=(section.bytes+page-1)/page*page;
  descriptor.usage=WGPUBufferUsage_Storage|WGPUBufferUsage_CopySrc|WGPUBufferUsage_CopyDst;
  operand->buffer=wgpuDeviceCreateBuffer(device,&descriptor);
  if(!operand->buffer)return kLiteRtStatusErrorRuntimeFailure;
  LiteRtStatus status=LiteRtCreateTensorBufferFromWebGpuBuffer(environment,type,storage,
    (LiteRtWGPUBuffer)operand->buffer,section.bytes,NULL,&operand->tensor);
  if(status){wgpuBufferRelease(operand->buffer);operand->buffer=NULL;}
  return status;
}

/* design/algorithm-sources.md#programtensor */
void mesh_webgpu_unbind(struct mesh_webgpu_operand *operand){
  if(operand->tensor)LiteRtDestroyTensorBuffer(operand->tensor);
  if(operand->buffer)wgpuBufferRelease(operand->buffer);
  *operand=(struct mesh_webgpu_operand){0};
}
