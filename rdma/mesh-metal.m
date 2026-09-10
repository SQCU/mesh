#import "mesh-metal.h"
#include <errno.h>
#include <time.h>
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
static id<MTLBuffer> mesh_metal_pool(id<MTLDevice> device, struct mesh_ctx *context, uint32_t first, uint32_t count, struct mesh_metal_layout *layout){
  size_t alignment=(size_t)getpagesize(), stride=context->M->pgsz;
  unsigned char *base=mesh_at(context->M,0);
  uintptr_t begin=((uintptr_t)base+(size_t)first*stride)/alignment*alignment;
  uintptr_t end=((uintptr_t)base+((size_t)first+count)*stride+alignment-1)/alignment*alignment;
  if(end-begin>device.maxBufferLength){ errno=EOVERFLOW; return nil; }
  context->mapping_pinned=1;
  *layout=(struct mesh_metal_layout){begin-(uintptr_t)base,(uint32_t)stride,(uint32_t)stride,(uint32_t)stride};
  return mesh_metal_memory(device,(void*)begin,end-begin,MTLResourceStorageModeShared);
}
// ../design/algorithm-sources.md#literal-row-functions
id<MTLBuffer> mesh_metal_row_table(id<MTLDevice> device, const struct mesh_rows *pages){
  size_t alignment=(size_t)getpagesize();
  if(!pages || !pages->count || pages->count>(SIZE_MAX-alignment+1)/sizeof(struct mesh_row)){ errno=EINVAL; return nil; }
  size_t bytes=(pages->count*sizeof(struct mesh_row)+alignment-1)/alignment*alignment;
  if((uintptr_t)pages->table%alignment || bytes>device.maxBufferLength){ errno=EINVAL; return nil; }
  return mesh_metal_memory(device,pages->table,bytes,MTLResourceStorageModeShared);
}
id<MTLBuffer> mesh_metal_receive_pool(id<MTLDevice> device, struct mesh_ctx *context, struct mesh_metal_layout *layout){
  return mesh_metal_pool(device,context,0,context->M->pool,layout);
}
id<MTLBuffer> mesh_metal_transmit_pool(id<MTLDevice> device, struct mesh_ctx *context, struct mesh_metal_layout *layout){
  return mesh_metal_pool(device,context,context->M->pool,context->M->arena,layout);
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
  uint32_t physical=mesh_rows_allocate(pages,(count*sizeof(uint64_t)+pages->bytes-1)/pages->bytes,pages->bytes);
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
  return mesh_metal_page_span(device,pages->context,physical,(uint32_t)((count*sizeof(uint64_t)+pages->bytes-1)/pages->bytes));
}

// ../design/algorithm-sources.md#literal-page-reduction
NSString *mesh_metal_rows_source(void){
  NSMutableString *source=[NSMutableString stringWithString:@"#include <metal_stdlib>\nusing namespace metal;\n"];
  uint32_t polynomials[2]={0x82f63b78u,0xedb88320u};
  for(uint32_t p=0;p<2;p++){
    uint32_t lookup[256];
    [source appendFormat:@"constant uint crc%u[256]={",p];
    for(uint32_t i=0;i<256;i++){
      uint32_t value=i;
      for(uint32_t bit=0;bit<8;bit++) value=(value>>1)^(polynomials[p]*(value&1));
      lookup[i]=value; [source appendFormat:@"%uu,",value];
    }
    [source appendFormat:@"};\nconstant uint shift%u[1024]={",p];
    for(uint32_t lane=0;lane<32;lane++) for(uint32_t bit=0;bit<32;bit++){
      uint32_t value=1u<<bit;
      for(size_t byte=0;byte<(31-lane)*(size_t)getpagesize()/32;byte++) value=(value>>8)^lookup[value&255];
      [source appendFormat:@"%uu,",value];
    }
    [source appendString:@"};\n"];
  }
  [source appendString:@"#include <metal_stdlib>\nusing namespace metal;\nstruct MeshRegion{device uchar *bytes;};\nconstant bool mesh_input_float32 [[function_constant(34)]];\n"
    "// ../design/algorithm-sources.md#literal-page-reduction\n"
    "static inline device uchar *payload(uint logical,device const uint4 *table,device const MeshRegion *regions,constant ulong *g){\n"
    "  ulong page=table[logical].x;\n"
    "  ulong byte=page*g[0]; return regions[byte>>30].bytes+(byte&((1ul<<30)-1));\n}\n"
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
    "// ../design/algorithm-sources.md#endpoint-page-digests\n"
    "kernel void mesh_digest(device const uint4 *table [[buffer(0)]],device const MeshRegion *regions [[buffer(1)]],constant ulong *g [[buffer(3)]],device const uint *selected [[buffer(4)]],uint group [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],uint threads [[threads_per_threadgroup]]){\n"
    "  uint index=selected[group+1],words=g[0]/8,lane=t%32,chunk=g[0]/32;\n"
    "  device ulong *out=(device ulong*)payload(g[7]+index,table,regions,g);\n"
    "  for(uint word=t/32;word<words;word+=threads/32){\n"
    "    uint input=index*words+word;\n"
    "    uint2 crc=uint2(0);\n"
    "    if(input<g[3]){\n"
    "      device uchar *value=payload(g[2]+input,table,regions,g)+lane*chunk;\n"
    "      uint first=lane?0:uint(g[5]),second=lane?0:uint(g[5]>>32)^0xffffffffu;\n"
    "      for(uint byte=0;byte<chunk;byte++){\n"
    "        first=(first>>8)^crc0[(first^value[byte])&255];\n"
    "        second=(second>>8)^crc1[(second^value[byte])&255];\n"
    "      }\n"
    "      for(uint bit=0;bit<32;bit++){\n"
    "        crc.x^=shift0[lane*32+bit]*((first>>bit)&1);\n"
    "        crc.y^=shift1[lane*32+bit]*((second>>bit)&1);\n"
    "      }\n"
    "    }\n"
    "    for(uint mask=16;mask;mask>>=1) crc^=simd_shuffle_xor(crc,mask);\n"
    "    if(!lane) out[word]=(ulong(crc.x)<<32)|crc.y;\n"
    "  }\n"
    "}\n"
    "// ../design/algorithm-sources.md#endpoint-page-digests\n"
    "kernel void mesh_compare(device const uint4 *table [[buffer(0)]],device const MeshRegion *regions [[buffer(1)]],constant ulong *g [[buffer(3)]],device const uint *selected [[buffer(4)]],uint group [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]){\n"
    "  uint index=selected[group+1],logical=g[2]+index*g[3];\n"
    "  device const ulong *left=(device const ulong*)payload(logical,table,regions,g);\n"
    "  device const ulong *right=(device const ulong*)payload(g[6]+index*g[3],table,regions,g);\n"
    "  ulong difference=0;\n"
    "  for(uint i=t;i<g[0]/8;i+=256) difference|=left[i]^right[i];\n"
    "  for(uint mask=16;mask;mask>>=1) difference|=simd_shuffle_xor(difference,mask);\n"
    "  threadgroup ulong partial[8];\n"
    "  if(!(t%32)) partial[t/32]=difference;\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  if(t) return;\n"
    "  for(uint i=1;i<8;i++) difference|=partial[i];\n"
    "  device ulong *out=(device ulong*)payload(g[7]+index*g[11],table,regions,g);\n"
    "  ulong stamp=ulong(table[logical].z)|(ulong(table[logical].w)<<32);\n"
    "  out[0]=stamp;out[1]=stamp;\n"
    "  device uint *fields=(device uint*)out;\n"
    "  fields[4]=g[10];fields[5]=index;fields[6]=g[12];fields[7]=0;\n"
    "  out[4]=difference?g[8]:0;fields[10]=g[9];fields[11]=0;\n"
    "}\n"
    ];
  return source;
}
