#ifndef MESH_METAL_H
#define MESH_METAL_H
#import <Metal/Metal.h>
#include "mesh.h"
#include "mesh-dataflow.h"
struct mesh_metal_layout { uint64_t origin; uint32_t stride, payload, capacity; };
id<MTLBuffer> mesh_metal_receive_pool(id<MTLDevice> device, struct mesh_ctx *context, struct mesh_metal_layout *layout);
id<MTLBuffer> mesh_metal_transmit_pool(id<MTLDevice> device, struct mesh_ctx *context, struct mesh_metal_layout *layout);
id<MTLBuffer> mesh_metal_row_table(id<MTLDevice> device, const struct mesh_rows *pages);
id<MTLBuffer> mesh_metal_page_span(id<MTLDevice> device, struct mesh_ctx *context, uint32_t first, uint32_t count);
id<MTLBuffer> mesh_metal_regions(id<MTLDevice> device, struct mesh_rows *pages, NSArray<id<MTLBuffer>> **resources);
NSString *mesh_metal_rows_source(void);
#endif
