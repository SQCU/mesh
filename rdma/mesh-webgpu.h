#ifndef MESH_WEBGPU_H
#define MESH_WEBGPU_H
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
#endif
