#import "mesh-metal-executor.h"
#include "mesh-wire.h"
#include <errno.h>
#include <unistd.h>

id<MTLBuffer> mesh_metal_indices(id<MTLDevice> device, const mesh_call *call){
  size_t bytes; const uint32_t *indices=mesh_call_indices(call,&bytes);
  if(!bytes || bytes>device.maxBufferLength){ errno=EINVAL; return nil; }
  id<MTLBuffer> buffer=[device newBufferWithBytesNoCopy:(void*)indices length:bytes options:MTLResourceStorageModeShared deallocator:nil];
  if(!buffer) errno=ENOMEM;
  return buffer;
}
void mesh_metal_publish_cpu(uint32_t *generation,uint32_t value){ __atomic_store_n(generation,value,__ATOMIC_RELEASE); }
size_t mesh_metal_row_chunk(struct mesh_ctx *context,struct mesh_scope scope,size_t row_bytes,size_t alignment){
  struct mstream stream={.scope=scope};
  size_t header=sizeof(struct wire)+mesh_stream_header(&stream), stride=context->M->pgsz;
  if(!alignment || stride%alignment || !row_bytes){ errno=EINVAL; return 0; }
  size_t padding=(alignment-header%alignment)%alignment;
  if(header+padding>=stride || row_bytes>stride-header-padding){ errno=EOVERFLOW; return 0; }
  return row_bytes+padding;
}
int mesh_metal_bind_rows(const struct mesh_view *view,struct mesh_metal_layout layout,
  size_t rows,size_t row_bytes,size_t alignment,struct mesh_metal_rows *result){
  if(!rows || !alignment || view->stride%alignment || view->payload>=view->stride || !row_bytes) return EINVAL;
  size_t payload=view->capacity?view->capacity:view->stride-view->payload;
  size_t padding=(alignment-view->payload%alignment)%alignment;
  if(padding>=payload || row_bytes>payload-padding || rows>view->bytes/payload) return EINVAL;
  for(size_t i=0;i<rows;i++) if(view->pages[i]==UINT32_MAX || view->pages[i]!=view->pages[0]+i) return ENOTSUP;
  *result=(struct mesh_metal_rows){(uint64_t)view->pages[0]*view->stride+view->payload+padding-layout.origin,
    view->stride,(uint32_t)payload,(uint32_t)padding,1};
  return 0;
}
id<MTLBuffer> mesh_metal_alias(id<MTLDevice> device, const struct mesh_view *view, struct mesh_metal_layout *layout){
  size_t payload=view->capacity?view->capacity:view->stride-view->payload, count=view->bytes/payload+(view->bytes%payload!=0);
  if(!count){ errno=EINVAL; return nil; }
  uint32_t low=UINT32_MAX,high=0;
  for(size_t i=0;i<count;i++){ if(view->pages[i]<low) low=view->pages[i]; if(view->pages[i]>high) high=view->pages[i]; }
  size_t alignment=(size_t)getpagesize();
  uintptr_t begin=((uintptr_t)view->base+(size_t)low*view->stride)/alignment*alignment;
  uintptr_t end=((uintptr_t)view->base+((size_t)high+1)*view->stride+alignment-1)/alignment*alignment;
  if(end-begin>device.maxBufferLength){ errno=EOVERFLOW; return nil; }
  *layout=(struct mesh_metal_layout){begin-(uintptr_t)view->base,view->stride,view->payload,(uint32_t)payload};
  id<MTLBuffer> memory=[device newBufferWithBytesNoCopy:(void*)begin length:end-begin options:MTLResourceStorageModeShared deallocator:nil];
  if(!memory) errno=ENOMEM;
  return memory;
}
void mesh_metal_completion(id<MTLCommandBuffer> command, void (^complete)(int)){
  [command addCompletedHandler:^(id<MTLCommandBuffer> finished){ complete(finished.status==MTLCommandBufferStatusCompleted?0:EIO); }];
}
ptrdiff_t mesh_metal_advance(id<MTLSharedEvent> event, mesh_call *call, uint64_t base,
  const struct mesh_metal_dependency *dependencies, size_t count){
  uint64_t value=event.signaledValue;
  size_t i=0;
  for(;i<count && base+dependencies[i].value<=value;i++){
    int status=dependencies[i].publish?mesh_publish(call,dependencies[i].argument):mesh_complete(call,dependencies[i].argument,0);
    if(status) return status;
  }
  return (ptrdiff_t)i;
}
int mesh_metal_acquire(id<MTLSharedEvent> event, mesh_call *call, size_t argument, uint64_t value){
  struct mesh_view view;
  int status=mesh_acquire(call,argument,&view);
  if(status==1) event.signaledValue=value;
  return status;
}

NSString *mesh_metal_source(void){
  return @"#include <metal_stdlib>\nusing namespace metal;\n"
    "struct MeshMetalLayout { ulong origin; uint stride,payload,capacity; };\n"
    "constant uint2 MeshMetalShape [[function_constant(23)]];\n"
    "constant uint MeshMetalCapacity [[function_constant(29)]];\n"
    "template<typename T> device const T* mesh_metal_page(device const uchar* memory,device const uint* pages,constant MeshMetalLayout& l,uint page){ uint stride=is_function_constant_defined(MeshMetalShape)?MeshMetalShape.x:l.stride,payload=is_function_constant_defined(MeshMetalShape)?MeshMetalShape.y:l.payload; return (device const T*)(memory+ulong(pages[page])*stride+payload-l.origin); }\n"
    "ulong mesh_metal_address(constant MeshMetalLayout& l,device const uint* pages,ulong i){ uint stride=is_function_constant_defined(MeshMetalShape)?MeshMetalShape.x:l.stride,payload=is_function_constant_defined(MeshMetalShape)?MeshMetalShape.y:l.payload; ulong n=is_function_constant_defined(MeshMetalCapacity)?MeshMetalCapacity:(l.capacity?l.capacity:stride-payload); return ulong(pages[i/n])*stride+payload+i%n-l.origin; }\n"
    "struct MeshMetalSpan { device uchar* memory; ulong first; uint stride,capacity;\n"
    "MeshMetalSpan(device uchar* m,device const uint* pages,constant MeshMetalLayout& l):memory(m){ stride=is_function_constant_defined(MeshMetalShape)?MeshMetalShape.x:l.stride; uint payload=is_function_constant_defined(MeshMetalShape)?MeshMetalShape.y:l.payload; capacity=(is_function_constant_defined(MeshMetalCapacity)?MeshMetalCapacity:(l.capacity?l.capacity:stride-payload))/2; first=ulong(pages[0])*stride+payload-l.origin; }\n"
    "void store_half(ulong i,half value) const thread { *(device half*)(memory+first+(i/capacity)*stride+(i%capacity)*2)=value; }\n"
    "template<typename T> void store(ulong i,T value) const thread { ulong n=(ulong(capacity)*2)/sizeof(T); *(device T*)(memory+first+(i/n)*stride+(i%n)*sizeof(T))=value; } };\n"
    "struct MeshMetalGather { device const uchar* memory; ulong first,origin; uint page,stride,payload,capacity,begin;\n"
    "MeshMetalGather(device const uchar* m,device const uint* pages,constant MeshMetalLayout& l,ulong start,uint count,uint lane):memory(m),first(start),origin(l.origin){ stride=is_function_constant_defined(MeshMetalShape)?MeshMetalShape.x:l.stride; payload=is_function_constant_defined(MeshMetalShape)?MeshMetalShape.y:l.payload; capacity=(is_function_constant_defined(MeshMetalCapacity)?MeshMetalCapacity:(l.capacity?l.capacity:stride-payload))/2; begin=uint(start/capacity); uint end=uint((start+count+capacity-1)/capacity); page=begin+lane<end?pages[begin+lane]:0; }\n"
    "half load_half(uint i) const thread { ulong at=first+i; ulong base=ulong(simd_shuffle(page,ushort(uint(at/capacity)-begin)))*stride+payload-origin; return *(device const half*)(memory+base+(at%capacity)*2); } };\n"
    "template<typename T> T mesh_metal_load(device const uchar* memory,device const uint* pages,constant MeshMetalLayout& layout,ulong i){ if constexpr(sizeof(T)==2) return *(device const T*)(memory+mesh_metal_address(layout,pages,i*2)); T value; thread uchar* out=(thread uchar*)&value; for(uint b=0;b<sizeof(T);b++) out[b]=memory[mesh_metal_address(layout,pages,i*sizeof(T)+b)]; return value; }\n"
    "template<typename T> void mesh_metal_store(device uchar* memory,device const uint* pages,constant MeshMetalLayout& layout,ulong i,T value){ if constexpr(sizeof(T)==2){ *(device T*)(memory+mesh_metal_address(layout,pages,i*2))=value; return; } thread uchar* in=(thread uchar*)&value; for(uint b=0;b<sizeof(T);b++) memory[mesh_metal_address(layout,pages,i*sizeof(T)+b)]=in[b]; }\n";
}

#define MESH_ADD_PAGES(T,A) \
  for(size_t k=0;k<count;k++){ \
    size_t page=indices[k]; \
    T *out=(T*)(destination->base+(size_t)destination->pages[page]*destination->stride+destination->payload+padding); \
    const T *in=(const T*)(source->base+(size_t)source->pages[page]*source->stride+source->payload+padding); \
    for(size_t i=0;i<elements;i++) out[i]=(T)((A)out[i]+(A)in[i]); \
  }
static void mesh_add_pages(struct mesh_metal_reduce_policy policy,const struct mesh_view *destination,
  const struct mesh_view *source,size_t elements,size_t padding,const uint32_t *indices,size_t count){
  if(policy.element_bytes==4){ MESH_ADD_PAGES(float,float) }
  else if(policy.accumulator_bytes==2){ MESH_ADD_PAGES(_Float16,_Float16) }
  else { MESH_ADD_PAGES(_Float16,float) }
}
#undef MESH_ADD_PAGES
static int mesh_metal_fail(const struct mesh_metal_reduction *edges,size_t count,struct mesh_metal_reduce_policy policy,int error){
  if(policy.cancel_on_error) for(size_t i=0;i<count;i++) mesh_call_cancel(edges[i].call,error);
  return -error;
}
ptrdiff_t mesh_metal_select(struct mesh_metal_reduction *edges,size_t count,struct mesh_metal_reduce_policy policy,
  struct mesh_metal_span *work,size_t *work_count){
  size_t done=0,n=0; *work_count=0;
  for(size_t i=0;i<count;i++){
    struct mesh_metal_reduction *edge=&edges[i];
    int status=mesh_call_status(edge->call); if(status<0) return mesh_metal_fail(edges,count,policy,-status);
    const struct mesh_view *local=mesh_call_view(edge->call,edge->argument);
    const struct mesh_view *remote=mesh_call_view(edge->call,edge->source);
    uint64_t generation=local->epoch->low;
    if(edge->value!=generation){ edge->value=generation; edge->selection=(struct mesh_selection){0}; }
    size_t pages=local->bytes/local->capacity+(local->bytes%local->capacity!=0);
    size_t publish=pages*(edge->event && edge->selection.published<local->bytes &&
      mesh_ready_pages(edge->call,edge->argument)<(ptrdiff_t)pages && edge->event.signaledValue>=edge->value);
    if(publish){
      status=mesh_publish(edge->call,edge->argument);
      if(status) return mesh_metal_fail(edges,count,policy,-status);
      ptrdiff_t flushed=mesh_flush(edge->call,edge->argument);
      if(flushed<0) return mesh_metal_fail(edges,count,policy,(int)-flushed);
    }
    const uint32_t *indices=NULL;
    ptrdiff_t selected=mesh_select_arrivals(edge->call,edge->source,edge->argument,&edge->selection,&indices);
    if(selected<0) return mesh_metal_fail(edges,count,policy,(int)-selected);
    edge->selection.consumed+=(size_t)selected;
    work[n]=(struct mesh_metal_span){i,indices,(size_t)selected}; n+=selected>0;
    done+=(edge->selection.consumed==pages) && mesh_epoch_equal(*local->epoch,*remote->epoch);
  }
  *work_count=n; return (ptrdiff_t)done;
}
ptrdiff_t mesh_metal_reduce(struct mesh_metal_reduction *edges,size_t count,struct mesh_metal_reduce_policy policy,
  struct mesh_metal_span *work,size_t *work_count){
  ptrdiff_t done=mesh_metal_select(edges,count,policy,work,work_count);
  if(done<0) return done;
  for(size_t i=0;i<*work_count;i++){
    struct mesh_metal_reduction *edge=&edges[work[i].edge];
    mesh_add_pages(policy,mesh_call_view(edge->call,edge->source),mesh_call_view(edge->call,edge->argument),
      edge->elements,edge->padding,work[i].indices,work[i].count);
  }
  return done;
}
int mesh_metal_gather(mesh_executor *executor,struct mesh_metal_reduction *edges,size_t count,struct mesh_metal_reduce_policy policy,struct mesh_metal_span *work){
  if((policy.element_bytes!=2 && policy.element_bytes!=4) || (policy.accumulator_bytes!=2 && policy.accumulator_bytes!=4) ||
     policy.accumulator_bytes<policy.element_bytes) return -EINVAL;
  ptrdiff_t done;
  do {
    mesh_progress(executor);
    size_t selected;
    done=mesh_metal_reduce(edges,count,policy,work,&selected);
  } while(done>=0 && (size_t)done<count);
  return done<0?(int)done:0;
}
