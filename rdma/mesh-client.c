
#include "mesh-dataflow.h"
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_vm.h>

// ../design/algorithm-sources.md#contiguous-backing-page-views
int mesh_memory_view(const struct mesh_memory_span *spans,size_t count,
                     void **address,size_t *bytes){
  if(!address || !bytes) return KERN_INVALID_ARGUMENT;
  *address=NULL; *bytes=0;
  if(!spans || !count) return KERN_INVALID_ARGUMENT;
  size_t alignment=(size_t)getpagesize(), length=0;
  for(size_t i=0;i<count;i++){
    uintptr_t source=(uintptr_t)spans[i].address;
    size_t n=spans[i].bytes;
    if(!n || source%alignment || n%alignment || n>SIZE_MAX-length ||
       source>UINTPTR_MAX-n) return KERN_INVALID_ARGUMENT;
    length+=n;
  }
  mach_vm_address_t base=0;
  kern_return_t status=mach_vm_allocate(mach_task_self(),&base,length,VM_FLAGS_ANYWHERE);
  if(status!=KERN_SUCCESS) return status;
  size_t offset=0;
  for(size_t i=0;i<count;i++){
    mach_vm_address_t target=base+offset;
    vm_prot_t current,maximum;
    status=mach_vm_remap(mach_task_self(),&target,spans[i].bytes,0,
      VM_FLAGS_FIXED|VM_FLAGS_OVERWRITE,mach_task_self(),
      (mach_vm_address_t)spans[i].address,FALSE,&current,&maximum,VM_INHERIT_NONE);
    if(status!=KERN_SUCCESS){
      mach_vm_deallocate(mach_task_self(),base,length);
      return status;
    }
    offset+=spans[i].bytes;
  }
  *address=(void*)base; *bytes=length;
  return KERN_SUCCESS;
}

// ../design/algorithm-sources.md#contiguous-backing-page-views
int mesh_memory_release(void *address,size_t bytes){
  return mach_vm_deallocate(mach_task_self(),(mach_vm_address_t)address,bytes);
}
#endif

static struct mesh_ctx CTX0;

// ../design/algorithm-sources.md#complete-page-ownership
struct mesh_ctx *mesh_context(void){ return &CTX0; }

// ../design/algorithm-sources.md#complete-page-ownership
struct hdr *mesh_region(struct mesh_ctx *context){ return context->M; }

// ../design/algorithm-sources.md#complete-page-ownership
size_t mesh_peers(struct mesh_ctx *context,uint16_t *peers,size_t capacity){
  size_t count=atomic_load_explicit(&context->M->port_count,memory_order_acquire);
  for(size_t i=0;i<count && i<capacity;i++) peers[i]=mesh_ports(context->M)[i].peer;
  return count;
}

// ../design/algorithm-sources.md#link-port-metadata
struct mesh_row_metadata mesh_link_metadata(struct mesh_ctx *context,size_t index){
  const struct mesh_port_info *port=&mesh_ports(context->M)[index];
  return (struct mesh_row_metadata){.stamp=0,.when=port->when,.function=UINT32_MAX,
    .index=(uint32_t)index,.peer=port->peer,.code=port->code,.domain=port->domain};
}

// ../design/algorithm-sources.md#complete-page-ownership
int mesh_attach(struct mesh_ctx *c,const char *name){
  if(c->M) return 0;
  if(!name) name=getenv("MESH_REGION");
  if(!name) name=MESH_NAME;
  int file=shm_open(name,O_RDWR,MESH_MODE);
  if(file<0) return errno;
  struct stat info;
  if(fstat(file,&info)){ int error=errno; close(file); return error; }
  struct hdr *memory=mmap(NULL,(size_t)info.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,file,0);
  int error=errno;
  close(file);
  if(memory==MAP_FAILED) return error;
  if((size_t)info.st_size<sizeof *memory || memory->magic!=MESH_MAGIC || memory->version!=MESH_VERSION){
    munmap(memory,(size_t)info.st_size); return EINVAL;
  }
  uint64_t vacant=0;
  if(!atomic_compare_exchange_strong_explicit(&memory->client,&vacant,(uint64_t)getpid(),memory_order_acq_rel,memory_order_acquire)){
    munmap(memory,(size_t)info.st_size); return EBUSY;
  }
  c->M=memory; c->len=(size_t)info.st_size;
  c->arena=mesh_at(memory,memory->pool);
  for(uint32_t i=0;i<memory->pool;i++)
    *mesh_context_row(memory,i)=(struct mesh_row){.page=MESH_ROW_ABSENT};
  return 0;
}

// ../design/algorithm-sources.md#complete-page-ownership
int mesh_detach(struct mesh_ctx *c){
  if(!c->M) return 0;
  int pending=mesh_rows_close(c);
  if(pending) return pending;
  atomic_store_explicit(&c->M->client,0,memory_order_release);
  int status=munmap(c->M,c->len);
  if(status) return errno;
  *c=(struct mesh_ctx){0};
  return 0;
}

// ../design/algorithm-sources.md#complete-page-ownership
int mesh_link_reset(struct mesh_ctx *c,size_t port){
  if(!c->M || port>=atomic_load_explicit(&c->M->port_count,memory_order_acquire)) return EINVAL;
  atomic_store_explicit(&mesh_ports(c->M)[port].reset_request,1,memory_order_release);
  return 0;
}
