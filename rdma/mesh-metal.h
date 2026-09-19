#ifndef MESH_METAL_H
#define MESH_METAL_H
#include "mesh-call.h"
#include "dawn/webgpu.h"
#include "litert/c/litert_tensor_buffer.h"

struct mesh_webgpu_operand {
  WGPUBuffer buffer;
  LiteRtTensorBuffer tensor;
};
_Static_assert(sizeof(struct mesh_webgpu_operand)==16,"native operand handles");

/* design/algorithm-sources.md#programtensor */
LiteRtStatus mesh_webgpu_bind(struct mesh_ctx *,struct mesh_section,uint32_t,
  LiteRtEnvironment,WGPUDevice,const LiteRtRankedTensorType *,LiteRtTensorBufferType,
  struct mesh_webgpu_operand *);
/* design/algorithm-sources.md#programtensor */
void mesh_webgpu_unbind(struct mesh_webgpu_operand *);

/* design/prepared-machine.md#M07 */
struct mesh_metal_transport { void *memory,*publish,*consume,*stop; };
_Static_assert(sizeof(struct mesh_metal_transport)==32,"prepared Metal transport");
/* design/algorithm-sources.md#resident-metal */
int mesh_metal_transport_create(struct mesh_ctx *,void *,struct mesh_metal_transport *);
/* design/algorithm-sources.md#resident-metal */
void mesh_metal_transfer_encode(struct mesh_ctx *,struct mesh_metal_transport *,void *,struct mesh_section,int);
/* design/algorithm-sources.md#resident-metal */
void mesh_metal_transport_destroy(struct mesh_metal_transport *);
#endif
