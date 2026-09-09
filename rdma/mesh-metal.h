#ifndef MESH_METAL_H
#define MESH_METAL_H
#import <Metal/Metal.h>
#include "mesh.h"
struct mesh_metal_layout { uint64_t origin; uint32_t stride, payload, capacity; };
struct mesh_metal_rows { uint64_t offset; uint32_t stride, payload, padding, rows_per_page; };
int mesh_metal_row_layout(struct mesh_ctx *context, struct mesh_scope scope, size_t rows,
  size_t row_bytes, size_t alignment, struct mesh_metal_rows *result);
id<MTLBuffer> mesh_metal_receive_pool(id<MTLDevice> device, struct mesh_ctx *context, struct mesh_metal_layout *layout);
id<MTLBuffer> mesh_metal_transmit_pool(id<MTLDevice> device, struct mesh_ctx *context, struct mesh_metal_layout *layout);
#endif
