#import "mesh-metal.h"
#include "mesh-wire.h"
#include <errno.h>
#include <mach/mach_vm.h>
#include <time.h>
#include <unistd.h>

static id<MTLBuffer> mesh_metal_memory(id<MTLDevice> device,const void *source,size_t bytes){
  mach_vm_address_t address=0;
  vm_prot_t current,maximum;
  kern_return_t status=mach_vm_remap(mach_task_self(),&address,bytes,0,VM_FLAGS_ANYWHERE,
    mach_task_self(),(mach_vm_address_t)source,FALSE,&current,&maximum,VM_INHERIT_NONE);
  if(status!=KERN_SUCCESS){ errno=ENOMEM; return nil; }
  id<MTLBuffer> buffer=[device newBufferWithBytesNoCopy:(void*)address length:bytes options:MTLResourceStorageModeShared
    deallocator:^(void *pointer,NSUInteger length){ mach_vm_deallocate(mach_task_self(),(mach_vm_address_t)pointer,length); }];
  if(!buffer){ mach_vm_deallocate(mach_task_self(),address,bytes); errno=ENOMEM; }
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
