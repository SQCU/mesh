#import "mesh-metal.h"
#include <errno.h>
#include <unistd.h>

// ../design/algorithm-sources.md#contiguous-backing-page-views
static id<MTLBuffer> mesh_metal_memory(id<MTLDevice> device,const void *source,size_t bytes,MTLResourceOptions options){
  const struct mesh_memory_span span={source,bytes};
  void *address=NULL;
  size_t length=0;
  if(mesh_memory_view(&span,1,&address,&length)){ errno=ENOMEM; return nil; }
  id<MTLBuffer> buffer=[device newBufferWithBytesNoCopy:address length:length options:options
    deallocator:^(void *pointer,NSUInteger count){ mesh_memory_release(pointer,count); }];
  if(!buffer){ mesh_memory_release(address,length); errno=ENOMEM; }
  return buffer;
}
// ../design/algorithm-sources.md#literal-row-functions
id<MTLBuffer> mesh_metal_row_table(id<MTLDevice> device, const struct mesh_rows *pages){
  size_t alignment=(size_t)getpagesize();
  if(!pages || !pages->count || pages->count>(SIZE_MAX-alignment+1)/sizeof(struct mesh_row)){ errno=EINVAL; return nil; }
  size_t bytes=(pages->count*sizeof(struct mesh_row)+alignment-1)/alignment*alignment;
  if((uintptr_t)pages->table%alignment || bytes>device.maxBufferLength){ errno=EINVAL; return nil; }
  return mesh_metal_memory(device,pages->table,bytes,MTLResourceStorageModeShared);
}
// ../design/algorithm-sources.md#complete-page-ownership
id<MTLBuffer> mesh_metal_page_span(id<MTLDevice> device,struct mesh_ctx *context,uint32_t first,uint32_t count){
  size_t bytes=(size_t)count*context->M->pgsz;
  if(!count || (uint64_t)first+count>(uint64_t)context->M->pool+context->M->arena || bytes>device.maxBufferLength){ errno=EINVAL; return nil; }
  context->mapping_pinned=1;
  return mesh_metal_memory(device,mesh_at(context->M,first),bytes,MTLResourceStorageModeShared);
}
// ../design/algorithm-sources.md#complete-page-ownership
id<MTLBuffer> mesh_metal_regions(id<MTLDevice> device,struct mesh_rows *pages,NSArray<id<MTLBuffer>> **resources){
  const size_t span=UINT64_C(1)<<30;
  size_t bytes=((size_t)pages->memory->pool+pages->memory->arena)*pages->bytes;
  size_t count=(bytes+span-1)/span;
  size_t allocation=mesh_region_table_pages((size_t)pages->memory->pool+pages->memory->arena,pages->bytes);
  if(allocation==SIZE_MAX) return nil;
  uint32_t physical=mesh_rows_allocate(pages,allocation,pages->bytes);
  if(physical==MESH_ROW_ABSENT) return nil;
  uint64_t *addresses=(void*)mesh_at(pages->memory,physical);
  NSMutableArray *buffers=[NSMutableArray new];
  for(size_t i=0;i<count;i++){
    size_t length=bytes-i*span; if(length>span) length=span;
    id<MTLBuffer> buffer=mesh_metal_memory(device,mesh_at(pages->memory,0)+i*span,length,
      MTLResourceStorageModeShared|MTLResourceHazardTrackingModeUntracked);
    if(!buffer) return nil;
    addresses[i]=buffer.gpuAddress;
    [buffers addObject:buffer];
  }
  *resources=buffers;
  return mesh_metal_page_span(device,pages->context,physical,(uint32_t)allocation);
}

// ../design/algorithm-sources.md#complete-page-ownership
NSString *mesh_metal_address_source(void){
  return @"#include <metal_stdlib>\nusing namespace metal;\nstruct MeshRegion{device uchar *bytes;};\n"
    "// ../design/algorithm-sources.md#complete-page-ownership\n"
    "static inline device uchar *mesh_page_address(device const MeshRegion *regions,ulong page,ulong pagebytes,ulong within){\n"
    "  ulong byte=page*pagebytes+within; return regions[byte>>30].bytes+(byte&((1ul<<30)-1));\n}\n";
}

// ../design/algorithm-sources.md#literal-page-reduction
NSString *mesh_metal_rows_source(void){
  return [mesh_metal_address_source() stringByAppendingString:@"constant bool mesh_input_float32 [[function_constant(34)]];\n"
    "// ../design/algorithm-sources.md#literal-page-reduction\n"
    "static inline device uchar *payload(uint logical,device const uint4 *table,device const MeshRegion *regions,constant ulong *g){\n"
    "  ulong page=table[logical].x;\n"
    "  return mesh_page_address(regions,page,g[0],0);\n}\n"
    "// ../design/algorithm-sources.md#literal-page-reduction\n"
    "static inline float number(uint logical,ulong element,bool fp32,device const uint4 *table,device const MeshRegion *regions,constant ulong *g){\n"
    "  ulong byte=element*(fp32?4:2);\n"
    "  device uchar *p=payload(logical+byte/g[0],table,regions,g)+byte%g[0];\n"
    "  return fp32?*(device float*)p:float(*(device half*)p);\n}\n"
    "// ../design/algorithm-sources.md#literal-page-reduction\n"
    "kernel void mesh_add(device const uint4 *table [[buffer(0)]],device const MeshRegion *regions [[buffer(1)]],\n"
    "  constant ulong *g [[buffer(3)]],device const uint *selected [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]){\n"
    "  uint index=selected[group.x+1],row=index*g[12]+group.y;\n"
    "  for(uint c=t;c<g[4];c+=256){\n"
    "    ulong local=(row/g[13])*(g[4]/g[15])*g[18]+(c/g[15])*g[18]+(c%g[15])*g[16]+row%g[13];\n"
    "    ulong remote=(row/g[14])*(g[4]/g[15])*g[19]+(c/g[15])*g[19]+(c%g[15])*g[17]+row%g[14];\n"
    "    float a=number(g[2],local,mesh_input_float32,table,regions,g),b=number(g[6],remote,mesh_input_float32,table,regions,g);\n"
    "    ulong byte=((row/g[12])*g[11]*(g[0]/4)+(row%g[12])*g[4]+c)*4;\n"
    "    *(device float*)(payload(g[7]+byte/g[0],table,regions,g)+byte%g[0])=a+b;\n"
    "  }\n}\n"
    ];
}
