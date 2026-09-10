#ifndef MESH_METAL_H
#define MESH_METAL_H
#import <Metal/Metal.h>
#include "mesh.h"
#include "mesh-dataflow.h"
id<MTLBuffer> mesh_metal_row_table(id<MTLDevice> device, const struct mesh_rows *pages);
id<MTLBuffer> mesh_metal_page_span(id<MTLDevice> device, struct mesh_ctx *context, uint32_t first, uint32_t count);
id<MTLBuffer> mesh_metal_regions(id<MTLDevice> device, struct mesh_rows *pages, NSArray<id<MTLBuffer>> **resources);
NSString *mesh_metal_address_source(void);
NSString *mesh_metal_rows_source(void);
#endif
