#ifndef MESH_METAL_H
#define MESH_METAL_H
#import <Metal/Metal.h>
#include "mesh.h"
NSArray<id<MTLBuffer>> *mesh_metal_memory(id<MTLDevice> device,struct mesh_ctx *context);
#endif
