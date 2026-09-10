#import "mesh-metal.h"
#include <errno.h>

// ../design/algorithm-sources.md#contiguous-backing-page-views
NSArray<id<MTLBuffer>> *mesh_metal_memory(id<MTLDevice> device,struct mesh_ctx *context){
  size_t bytes=((size_t)context->M->pool+context->M->arena)*context->M->pgsz;
  size_t span=device.maxBufferLength/context->M->pgsz*context->M->pgsz;
  context->extent_bytes=span;
  NSMutableArray *buffers=[NSMutableArray new];
  for(size_t offset=0;offset<bytes;offset+=span){
    size_t length=bytes-offset; if(length>span) length=span;
    id<MTLBuffer> buffer=[device newBufferWithBytesNoCopy:mesh_at(context->M,0)+offset
      length:length options:MTLResourceStorageModeShared|MTLResourceHazardTrackingModeUntracked
      deallocator:nil];
    if(!buffer){ errno=ENOMEM; return nil; }
    [buffers addObject:buffer];
  }
  return buffers;
}
