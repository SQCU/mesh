#import "mesh-metal.h"
#include "mesh-wire.h"
#include <errno.h>
#include <time.h>
#include <unistd.h>

// ../design/algorithm-sources.md#contiguous-backing-page-views
static id<MTLBuffer> mesh_metal_memory(id<MTLDevice> device,const void *source,size_t bytes){
  const struct mesh_memory_span span={source,bytes};
  void *address=NULL;
  size_t length=0;
  if(mesh_memory_view(&span,1,&address,&length)){ errno=ENOMEM; return nil; }
  id<MTLBuffer> buffer=[device newBufferWithBytesNoCopy:address length:length options:MTLResourceStorageModeShared
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
  if(context->mapping_pinned>=0) context->mapping_pinned=1;
  *layout=(struct mesh_metal_layout){begin-(uintptr_t)base,(uint32_t)stride,0,(uint32_t)stride};
  return mesh_metal_memory(device,(void*)begin,end-begin);
}
id<MTLBuffer> mesh_metal_page_table(id<MTLDevice> device, const mesh_pages *pages){
  size_t bytes;
  const uint32_t *table=mesh_pages_table(pages,&bytes);
  if(bytes>device.maxBufferLength){ errno=EOVERFLOW; return nil; }
  return mesh_metal_memory(device,table,bytes);
}
// ../design/algorithm-sources.md#literal-row-functions
id<MTLBuffer> mesh_metal_row_table(id<MTLDevice> device, const struct mesh_rows *pages){
  size_t alignment=(size_t)getpagesize();
  if(!pages || !pages->count || pages->count>(SIZE_MAX-alignment+1)/sizeof(struct mesh_row)){ errno=EINVAL; return nil; }
  size_t bytes=(pages->count*sizeof(struct mesh_row)+alignment-1)/alignment*alignment;
  if((uintptr_t)pages->table%alignment || bytes>device.maxBufferLength){ errno=EINVAL; return nil; }
  return mesh_metal_memory(device,pages->table,bytes);
}
id<MTLBuffer> mesh_metal_receive_pool(id<MTLDevice> device, struct mesh_ctx *context, struct mesh_metal_layout *layout){
  return mesh_metal_pool(device,context,0,context->M->pool,layout);
}
id<MTLBuffer> mesh_metal_transmit_pool(id<MTLDevice> device, struct mesh_ctx *context, struct mesh_metal_layout *layout){
  return mesh_metal_pool(device,context,context->M->pool,context->M->arena,layout);
}
int mesh_metal_row_layout(struct mesh_ctx *context,struct mesh_scope scope,size_t rows,
  size_t row_bytes,size_t alignment,struct mesh_metal_rows *result){
  size_t header=sizeof(struct wire)+(mesh_epoch_set(scope.epoch)?sizeof(struct mesh_frame):MESH_OFF), stride=context->M->pgsz;
  if(!rows || !alignment || stride%alignment || !row_bytes) return EINVAL;
  size_t padding=(alignment-header%alignment)%alignment;
  if(header+padding>=stride || row_bytes>stride-header-padding) return EOVERFLOW;
  size_t packed=1, first=header+padding;
  for(;stride%2==0 && (stride/2)%alignment==0 && rows%(packed*2)==0 && first+row_bytes<=stride/2;packed*=2) stride/=2;
  *result=(struct mesh_metal_rows){first,(uint32_t)stride,(uint32_t)(padding+(packed-1)*stride+row_bytes),(uint32_t)padding,(uint32_t)packed};
  return 0;
}
