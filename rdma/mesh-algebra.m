#import <Foundation/Foundation.h>
#define ACCELERATE_NEW_LAPACK
#define ACCELERATE_LAPACK_ILP64
#import <Accelerate/Accelerate.h>
#import <CommonCrypto/CommonDigest.h>
#import <Metal/Metal.h>
#import <CoreML/CoreML.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "mesh-algebra.h"
#include "mesh-dataflow.h"
#include "mesh-kernel.h"
#include <limits.h>
#include <math.h>
#include <dlfcn.h>

/* design/algorithm-sources.md#programkernel_call */
static size_t scalar_bytes(enum mesh_scalar scalar) {
  static const size_t bytes[]={2,4,4,4,8,8,1,1};
  return (unsigned)scalar<sizeof bytes/sizeof *bytes?bytes[scalar]:0;
}

struct mesh_extent {
  uint32_t first,pages,quantum;
  size_t bytes;
  void *address;
  _Atomic uint32_t *host_pages;
  CFTypeRef buffer;
  struct mesh_shape shape;
};
struct mesh_tensor { struct mesh_ctx *context; size_t count; struct mesh_extent *extents; };
struct mesh_writer { struct mesh_ctx *context; struct mesh_row_map output; struct mesh_row_function function; struct mesh_writer *next; };

typedef void (*mesh_cpu_kernel)(const uintptr_t *,const struct mesh_kernel_publication *);
@interface MeshCPUCode : NSObject
@property void *handle;
@property mesh_cpu_kernel kernel;
@end
@implementation MeshCPUCode
/* design/algorithm-sources.md#kernelsexpression */
- (void)dealloc {if(self.handle)dlclose(self.handle);}
@end

@class MeshAlgebra;
@interface MeshFunction : NSObject {
@public
  struct mesh_row_map output;
  struct mesh_row_function function;
  struct mesh_kernel_publication publication;
}
@property MeshCPUCode *cpuCode;
@property id<MTLComputePipelineState> metalPipeline;
@property id<MTLCommandQueue> queue;
@property NSData *cpuArguments;
@property NSMutableData *bnnsFilters;
@property NSMutableData *publicationSections;
@property NSMutableData *dependencies,*results;
@property(nonatomic,assign) MeshAlgebra *owner;
@property(copy) void (^encode)(id<MTLCommandBuffer>);
@property(copy) void (^execute)(MeshFunction *);
@end
@implementation MeshFunction
/* design/algorithm-sources.md#kernelsdot */
- (void)dealloc {
  const BNNSFilter *filters=self.bnnsFilters.bytes;
  for(size_t i=0;i<self.bnnsFilters.length/sizeof *filters;i++)BNNSFilterDestroy(filters[i]);
}
@end

@interface MeshAlgebra : NSObject {
@public
  struct mesh_ctx *context;
  struct mesh_writer *writers;
  uint32_t copies;
  _Atomic int64_t code;
}
@property BOOL realized,cpu;
@property dispatch_group_t executions;
@property id<MTLDevice> device;
@property id<MTLCommandQueue> queue;
@property NSMutableDictionary<NSString *,id<MTLLibrary>> *libraries;
@property NSMutableDictionary<NSString *,MeshCPUCode *> *cpuCode;
@property NSMutableDictionary<NSArray *,NSArray<MPSMatrix *> *> *matrixViews;
@property NSMutableDictionary<NSNumber *,NSArray<id<MTLBuffer>> *> *matrixBuffers;
@property NSMutableArray<MeshFunction *> *functions;
@property NSArray<id<MTLBuffer>> *banks;
@property id<MTLBuffer> pageTable,bankAddresses;
@property size_t bankBytes;
@property NSMutableData *tensors;
@property NSMutableData *bindings;
@property NSMutableData *returns;
@property NSString *coremlPython,*coremlGenerator,*coremlCache;
@property NSMutableDictionary<NSString *,MLModel *> *models;
@end
@implementation MeshAlgebra
/* design/algorithm-sources.md#programkernel_call */
- (void)dealloc {
  for(MeshFunction *f in self.functions){
    for(uint32_t i=0;i<f->function.inputs;i++)mesh_reader_unbind(context,&f->function.input[i]);
  }
  while(writers){struct mesh_writer *next=writers->next;free(writers);writers=next;}
  struct mesh_row_map *returns=self.returns.mutableBytes;
  for(size_t i=0;i<self.returns.length/sizeof *returns;i++)mesh_reader_unbind(context,&returns[i]);
  self.functions=nil;
  struct mesh_tensor **tensors=self.tensors.mutableBytes;
  for(size_t i=0;i<self.tensors.length/sizeof *tensors;i++) {
    struct mesh_tensor *t=tensors[i];
    for(size_t j=0;j<t->count;j++) {
      struct mesh_extent *e=&t->extents[j];
      if(e->buffer)CFRelease(e->buffer);
      if(e->address)mesh_view_destroy(e->address,e->bytes);
      free(e->host_pages);
      if(e->first!=MESH_ABSENT){mesh_backing_release(context,e->first,e->pages);mesh_rows_release(context,e->first,e->pages);}
    }
    free(t->extents); free(t);
  }
}
@end

/* design/algorithm-sources.md#program */
static void complete_part(MeshFunction *f,int64_t error) {
  MeshAlgebra *a=f.owner;
  if(error)atomic_store(&a->code,error);
  else mesh_complete(a->context,&f->function);
  dispatch_group_leave(a.executions);
}
/* design/algorithm-sources.md#programkernel_call */
static MeshAlgebra *owner(struct mesh_algebra *a) { return (__bridge MeshAlgebra *)a; }
/* design/algorithm-sources.md#programkernel_call */
static struct mesh_algebra *create_algebra(struct mesh_ctx *context,BOOL cpu) {
  if(!context || !context->M){errno=EINVAL;return NULL;}
  MeshAlgebra *a=[MeshAlgebra new]; a->context=context;a.cpu=cpu;
  if(!cpu) {
    a.device=MTLCreateSystemDefaultDevice(); a.queue=[a.device newCommandQueue];
    if(!a.queue){errno=ENODEV;return NULL;}
    struct hdr *m=context->M;
    size_t block_bytes=(size_t)m->block*m->pgsz;
    a.bankBytes=a.device.maxBufferLength/block_bytes*block_bytes;
    NSMutableArray<id<MTLBuffer>> *banks=[NSMutableArray new];
    size_t bytes=(size_t)mesh_rows(m)*m->pgsz;
    for(size_t offset=0;offset<bytes;offset+=a.bankBytes){
      size_t length=bytes-offset<a.bankBytes?bytes-offset:a.bankBytes;
      id<MTLBuffer> bank=[a.device newBufferWithBytesNoCopy:mesh_at(m,0)+offset length:length options:MTLResourceStorageModeShared|MTLResourceHazardTrackingModeUntracked deallocator:nil];
      if(!bank){errno=ENOMEM;return NULL;}[banks addObject:bank];
    }
    a.banks=banks;
    a.bankAddresses=[a.device newBufferWithLength:banks.count*sizeof(uint64_t) options:MTLResourceStorageModeShared];
    size_t start=m->page_off/m->pgsz*m->pgsz,end=(m->page_off+mesh_rows(m)*sizeof(uint32_t)+m->pgsz-1)/m->pgsz*m->pgsz;
    a.pageTable=[a.device newBufferWithBytesNoCopy:(char *)m+start length:end-start options:MTLResourceStorageModeShared deallocator:nil];
    if(!a.bankAddresses || !a.pageTable){errno=ENOMEM;return NULL;}
    for(size_t i=0;i<banks.count;i++)((uint64_t *)a.bankAddresses.contents)[i]=banks[i].gpuAddress;

  }
  a.executions=dispatch_group_create();
  a.libraries=[NSMutableDictionary new];a.cpuCode=[NSMutableDictionary new];
  a.matrixViews=[NSMutableDictionary new];a.matrixBuffers=[NSMutableDictionary new];
  a.functions=[NSMutableArray new];
  a.tensors=[NSMutableData new]; a.bindings=[NSMutableData new]; a.returns=[NSMutableData new];
  return (__bridge_retained struct mesh_algebra *)a;
}
/* design/algorithm-sources.md#kernelsexpression */
struct mesh_algebra *mesh_algebra_create(struct mesh_ctx *context) {return create_algebra(context,NO);}
/* design/algorithm-sources.md#kernelsexpression */
struct mesh_algebra *mesh_algebra_create_cpu(struct mesh_ctx *context) {return create_algebra(context,YES);}
/* design/algorithm-sources.md#programkernel_call */
int mesh_algebra_kernel(struct mesh_algebra *handle) {
  MeshAlgebra *a=owner(handle);if(a.realized)return EBUSY;
  if(!a.cpu){a.queue=[a.device newCommandQueue];if(!a.queue)return ENOMEM;}
  return 0;
}
/* design/algorithm-sources.md#programkernel_call */
uint32_t mesh_algebra_node(struct mesh_algebra *handle) {return owner(handle)->context->M->node;}
/* design/algorithm-sources.md#kernelsexpression */
size_t mesh_algebra_page_bytes(struct mesh_algebra *handle) {return owner(handle)->context->M->pgsz;}
/* design/algorithm-sources.md#kernelsdot */
int mesh_algebra_coreml(struct mesh_algebra *handle,const char *python,const char *generator,const char *cache) {
  MeshAlgebra *a=owner(handle);
  if(a.realized || a.functions.count)return EBUSY;
  if(a.cpu || !python || !generator || !cache)return EINVAL;
  a.coremlPython=@(python);a.coremlGenerator=@(generator);a.coremlCache=@(cache);a.models=[NSMutableDictionary new];return 0;
}
/* design/algorithm-sources.md#programkernel_call */
void mesh_algebra_destroy(struct mesh_algebra *handle) {
  if(!handle)return;MeshAlgebra *a=owner(handle);
  mesh_execution_remove(a->context,handle);
  dispatch_group_wait(a.executions,DISPATCH_TIME_FOREVER);
  CFBridgingRelease(handle);
}

/* design/algorithm-sources.md#programkernel_call */
struct mesh_tensor *mesh_tensor_create(struct mesh_algebra *handle,const struct mesh_shape *shapes,size_t count,int transferable,int contiguous) {
  MeshAlgebra *a=owner(handle);
  if(a.realized){errno=EBUSY;return NULL;}
  size_t limit=a.cpu?SIZE_MAX:a.device.maxBufferLength;
  if(!count || !shapes || count>UINT32_MAX){errno=EINVAL;return NULL;}
  for(size_t i=0;i<count;i++) {
    size_t bytes=scalar_bytes(shapes[i].scalar);
    if(!bytes || !shapes[i].rows || !shapes[i].columns || shapes[i].rows>SIZE_MAX/shapes[i].columns/bytes){errno=EINVAL;return NULL;}
    if(shapes[i].rows*shapes[i].columns*bytes>limit){errno=EOVERFLOW;return NULL;}
  }
  struct mesh_tensor *t=calloc(1,sizeof *t);
  if(!t){errno=ENOMEM;return NULL;}
  t->extents=calloc(count,sizeof *t->extents);
  if(!t->extents){free(t);errno=ENOMEM;return NULL;}
  t->context=a->context; t->count=count;
  for(size_t i=0;i<count;i++)t->extents[i].first=MESH_ABSENT;
  [a.tensors appendBytes:&t length:sizeof t];
  size_t pg=a->context->M->pgsz,align=transferable?a->context->M->block:1;
  for(size_t i=0;i<count;i++) {
    struct mesh_extent *e=&t->extents[i]; e->shape=shapes[i];
    size_t bytes=shapes[i].rows*shapes[i].columns*(scalar_bytes(shapes[i].scalar));
    size_t pages=(bytes+pg-1)/pg; pages=(pages+align-1)/align*align;
    if(pages>UINT32_MAX || pages*pg>limit){errno=EOVERFLOW;return NULL;}
    e->pages=(uint32_t)pages; e->bytes=pages*pg; e->quantum=(uint32_t)align;
    e->first=mesh_rows_alloc(a->context,e->pages);
    if(e->first==MESH_ABSENT)return NULL;
    int error=mesh_backing_alloc(a->context,e->first,e->pages,e->quantum,contiguous);
    if(error){errno=error;return NULL;}
    e->host_pages=malloc(pages*sizeof *e->host_pages);
    if(!e->host_pages){errno=ENOMEM;return NULL;}
    e->address=mesh_view_create(a->context,e->first,pages,e->host_pages);
    if(!e->address)return NULL;
    if(!a.cpu) {
      e->buffer=CFBridgingRetain([a.device newBufferWithBytesNoCopy:e->address length:e->bytes options:MTLResourceStorageModeShared|MTLResourceHazardTrackingModeUntracked deallocator:nil]);
      if(!e->buffer){errno=ENOMEM;return NULL;}
    }
  }
  return t;
}

/* design/algorithm-sources.md#programkernel_call */
struct mesh_view mesh_tensor_view(struct mesh_tensor *t,uint32_t i) {
  if(!t || i>=t->count)return (struct mesh_view){0};
  struct mesh_shape s=t->extents[i].shape;
  return (struct mesh_view){t,i,0,s.rows,s.columns,s.columns,1};
}
/* design/algorithm-sources.md#programkernel_call */
struct mesh_view mesh_view_slice(struct mesh_view v,size_t row,size_t column,size_t rows,size_t columns) {
  if(row>v.rows || column>v.columns || rows>v.rows-row || columns>v.columns-column)return (struct mesh_view){0};
  v.offset+=row*v.row_stride+column*v.column_stride;v.rows=rows;v.columns=columns;return v;
}
/* design/algorithm-sources.md#programkernel_call */
struct mesh_view mesh_view_transpose(struct mesh_view v) {
  size_t n=v.rows;v.rows=v.columns;v.columns=n;n=v.row_stride;v.row_stride=v.column_stride;v.column_stride=n;return v;
}
/* design/algorithm-sources.md#programkernel_call */
struct mesh_view mesh_view_broadcast(struct mesh_view v,size_t rows,size_t columns) {
  if((v.rows!=rows && v.rows!=1) || (v.columns!=columns && v.columns!=1))return (struct mesh_view){0};
  if(v.rows!=rows)v.row_stride=0;
  if(v.columns!=columns)v.column_stride=0;
  v.rows=rows;v.columns=columns;return v;
}
/* design/algorithm-sources.md#kernelsexpression */
size_t mesh_tensor_publication_bytes(struct mesh_tensor *t,uint32_t i) {
  return t && i<t->count?(size_t)t->extents[i].quantum*t->context->M->pgsz:0;
}
/* design/algorithm-sources.md#programtensor */
void *mesh_view_data(struct mesh_view v) {
  if(!v.tensor || v.extent>=v.tensor->count){errno=EINVAL;return NULL;}
  struct mesh_extent *e=&v.tensor->extents[v.extent];struct mesh_ctx *c=v.tensor->context;
  size_t scalar=scalar_bytes(e->shape.scalar),page=c->M->pgsz;
  if(v.row_stride<v.column_stride)v=mesh_view_transpose(v);
  if(v.column_stride==1 && v.row_stride==v.columns){v.columns*=v.rows;v.rows=1;}
  for(size_t r=0;r<v.rows;r++) {
    size_t first=(v.offset+r*v.row_stride)*scalar/page;
    size_t end=((v.offset+r*v.row_stride+(v.columns-1)*v.column_stride)*scalar)/page+1;
    int error=mesh_view_bind(c,e->first+(uint32_t)first,end-first,(char *)e->address+first*page,e->host_pages+first);
    if(error){errno=error;return NULL;}
  }
  return (char *)e->address+v.offset*scalar;
}
/* design/algorithm-sources.md#programkernel_call */
int mesh_tensor_constant(struct mesh_tensor *t,uint32_t i) {
  if(!t || i>=t->count)return EINVAL;
  struct mesh_extent *e=&t->extents[i];mesh_constant(t->context,e->first,e->pages);return 0;
}
/* design/algorithm-sources.md#programwrite */
int mesh_writer_issue(struct mesh_writer *w) {
  return mesh_issue(w->context,&w->function);
}
/* design/algorithm-sources.md#programwrite */
void mesh_writer_complete(struct mesh_writer *w,int publish) {
  if(publish)mesh_complete(w->context,&w->function);
  else mesh_bits_clear(w->context->M,MESH_PRODUCING,w->output.first,w->output.count);
}
/* design/algorithm-sources.md#programkernel_call */
static int valid_view(MeshAlgebra *a,struct mesh_view v) {
  if(!v.tensor || v.extent>=v.tensor->count || !v.rows || !v.columns || v.tensor->context!=a->context)return 0;
  size_t elements=v.tensor->extents[v.extent].shape.rows*v.tensor->extents[v.extent].shape.columns;
  if(v.offset>=elements || (v.row_stride && v.rows-1>(elements-1-v.offset)/v.row_stride))return 0;
  size_t last=v.offset+(v.rows-1)*v.row_stride;
  return !v.column_stride || v.columns-1<=(elements-1-last)/v.column_stride;
}

/* design/algorithm-sources.md#programkernel_call */
id<MTLBuffer> mesh_algebra_buffer(struct mesh_algebra *handle,struct mesh_view view) {
  MeshAlgebra *a=owner(handle);
  if(!valid_view(a,view)){errno=EINVAL;return nil;}
  return (__bridge id<MTLBuffer>)view.tensor->extents[view.extent].buffer;
}

/* design/algorithm-sources.md#programkernel_call */
struct mesh_matrix_address {
  _Atomic uint32_t *page;
  size_t offset,page_bytes;
  void *base;
};
/* design/algorithm-sources.md#kernelsdot */
static struct mesh_matrix_address matrix_address(struct mesh_view v) {
  static _Atomic uint32_t pinned_page;
  struct mesh_extent *e=&v.tensor->extents[v.extent];
  struct hdr *m=v.tensor->context->M;
  size_t offset=v.offset*scalar_bytes(e->shape.scalar);
  if(mesh_bits_all(m,MESH_CONSTANT,e->first,e->pages))
    return (struct mesh_matrix_address){&pinned_page,offset,m->pgsz,e->address};
  return (struct mesh_matrix_address){mesh_page(m)+e->first+offset/m->pgsz,offset%m->pgsz,m->pgsz,mesh_at(m,0)};
}
/* design/algorithm-sources.md#kernelsdot */
static size_t matrix_offset(struct mesh_matrix_address address) {
  return (size_t)atomic_load_explicit(address.page,memory_order_acquire)*address.page_bytes+address.offset;
}
/* design/algorithm-sources.md#kernelsdot */
struct mesh_mps_address {struct mesh_matrix_address storage;size_t row_bytes,residue_step,variants,bank_bytes;};
/* design/algorithm-sources.md#kernelsdot */
static NSArray<MPSMatrix *> *matrix(MeshAlgebra *a,struct mesh_view v,BOOL transpose,BOOL result,struct mesh_mps_address *address) {
  struct mesh_extent *e=&v.tensor->extents[v.extent];
  size_t bytes=scalar_bytes(e->shape.scalar),rows=transpose?v.columns:v.rows,columns=transpose?v.rows:v.columns;
  size_t stride=(transpose?v.column_stride:v.row_stride)*bytes;
  MPSDataType type=bytes==2?MPSDataTypeFloat16:MPSDataTypeFloat32;
  address->storage=matrix_address(v);
  if(address->storage.base==e->address) {
    MPSMatrixDescriptor *d=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:columns rowBytes:stride dataType:type];
    MPSMatrix *view=[[MPSMatrix alloc]initWithBuffer:(__bridge id<MTLBuffer>)e->buffer offset:v.offset*bytes descriptor:d];
    address->storage.offset=0;address->row_bytes=MAX(stride,bytes);address->residue_step=1;address->variants=1;address->bank_bytes=SIZE_MAX;
    return @[view];
  }
  size_t pg=a->context->M->pgsz;
  size_t quantum=result?e->quantum:0,bank_bytes=quantum?quantum*pg:a.bankBytes;
  NSArray<id<MTLBuffer>> *buffers=a.banks;
  if(quantum) {
    buffers=a.matrixBuffers[@(quantum)];
    if(!buffers) {
      NSMutableArray<id<MTLBuffer>> *blocks=[NSMutableArray new];
      size_t arena_bytes=(size_t)mesh_rows(a->context->M)*pg;
      for(size_t offset=0;offset<arena_bytes;offset+=bank_bytes)
        [blocks addObject:[a.device newBufferWithBytesNoCopy:mesh_at(a->context->M,0)+offset length:MIN(bank_bytes,arena_bytes-offset) options:MTLResourceStorageModeShared deallocator:nil]];
      a.matrixBuffers[@(quantum)]=blocks;buffers=blocks;
    }
  }
  if(rows==1)stride=(columns*bytes+pg-1)/pg*pg;
  size_t divisor=stride,remainder=pg;
  while(remainder){size_t next=divisor%remainder;divisor=remainder;remainder=next;}
  size_t residue=address->storage.offset%divisor;
  *address=(struct mesh_mps_address){address->storage,stride,divisor,stride/divisor,bank_bytes};
  NSArray *key=@[@(stride),@(columns),@(type),@(residue),@(quantum)];
  NSArray<MPSMatrix *> *cached=a.matrixViews[key];if(cached)return cached;
  NSMutableArray<MPSMatrix *> *views=[NSMutableArray new];
  for(id<MTLBuffer> bank in buffers)for(size_t offset=residue;offset<stride;offset+=divisor) {
    if(offset+columns*bytes>bank.length){[views addObject:(id)NSNull.null];continue;}
    size_t count=(bank.length-offset-columns*bytes)/stride+1;
    MPSMatrixDescriptor *d=[MPSMatrixDescriptor matrixDescriptorWithRows:count columns:columns rowBytes:stride dataType:type];
    [views addObject:[[MPSMatrix alloc]initWithBuffer:bank offset:offset descriptor:d]];
  }
  a.matrixViews[key]=views;return views;
}
/* design/algorithm-sources.md#kernelsdot */
static MPSMatrix *matrix_resolve(NSArray<MPSMatrix *> *views,struct mesh_mps_address address,MTLOrigin *origin) {
  size_t byte=matrix_offset(address.storage),offset=byte%address.bank_bytes;
  *origin=MTLOriginMake(offset/address.row_bytes,0,0);
  return views[byte/address.bank_bytes*address.variants+(offset%address.row_bytes)/address.residue_step];
}

/* design/algorithm-sources.md#programkernel_call */
static void dependencies(NSMutableData *maps,struct mesh_view v) {
  struct mesh_extent *e=&v.tensor->extents[v.extent];
  size_t unit=v.tensor->context->M->pgsz/(scalar_bytes(e->shape.scalar));
  if(v.row_stride<v.column_stride)v=mesh_view_transpose(v);
  for(size_t r=0;r<v.rows;r++) {
    size_t first=v.offset+r*v.row_stride;
    for(size_t c=0;c<v.columns;) {
      size_t last=v.column_stride<=unit?v.columns-1:c;
      size_t lo=(first+c*v.column_stride)/unit,hi=(first+last*v.column_stride)/unit;
      struct mesh_row_map m={.first=e->first+(uint32_t)lo,.count=(uint32_t)(hi-lo+1)};
      [maps appendBytes:&m length:sizeof m];
      c=last+1;
    }
  }
}
/* design/algorithm-sources.md#programkernel_call */
static int compare_maps(const void *a,const void *b) {
  uint32_t x=((const struct mesh_row_map *)a)->first,y=((const struct mesh_row_map *)b)->first;
  return (x>y)-(x<y);
}
/* design/algorithm-sources.md#program */
static size_t merge_maps(NSMutableData *storage,size_t first) {
  struct mesh_row_map *maps=(struct mesh_row_map *)storage.mutableBytes+first;size_t length=storage.length/sizeof *maps-first,used=0;
  qsort(maps,length,sizeof *maps,compare_maps);
  for(size_t i=0;i<length;i++) {
    if(used && maps[i].first<=maps[used-1].first+maps[used-1].count)maps[used-1].count=MAX(maps[used-1].first+maps[used-1].count,maps[i].first+maps[i].count)-maps[used-1].first;
    else maps[used++]=maps[i];
  }
  storage.length=(first+used)*sizeof *maps;return used;
}
/* design/algorithm-sources.md#program */
static int output_used(MeshAlgebra *a,struct mesh_row_map m) {
  for(MeshFunction *f in a.functions)for(uint32_t i=0;i<f->function.outputs;i++) {
    struct mesh_row_map out=f->function.output[i];
    if(m.first<out.first+out.count && out.first<m.first+m.count)return 1;
  }
  return 0;
}
/* design/algorithm-sources.md#programkernel_call */
static int output_region(MeshAlgebra *a,struct mesh_view v,struct mesh_row_map *m) {
  if(!valid_view(a,v))return EINVAL;
  struct mesh_extent *e=&v.tensor->extents[v.extent];
  size_t scalar=scalar_bytes(e->shape.scalar),unit=a->context->M->pgsz/scalar;
  size_t quantum=(size_t)e->quantum*unit,elements=e->shape.rows*e->shape.columns;
  if(!((v.column_stride==1 && (v.rows==1 || v.row_stride==v.columns)) || (v.row_stride==1 && (v.columns==1 || v.column_stride==v.rows))))return EINVAL;
  if(v.rows>SIZE_MAX/v.columns)return EOVERFLOW;
  size_t count=v.rows*v.columns,end=v.offset+count;
  if(v.offset%quantum || (end!=elements && end%quantum))return EINVAL;
  *m=(struct mesh_row_map){.first=e->first+(uint32_t)(v.offset/unit),.count=(uint32_t)(((count+quantum-1)/quantum)*e->quantum)};
  return 0;
}
/* design/algorithm-sources.md#programwrite */
int mesh_algebra_writer(struct mesh_algebra *handle,struct mesh_view view,struct mesh_writer **result) {
  if(!result)return EINVAL;
  *result=NULL;
  MeshAlgebra *a=owner(handle);
  struct mesh_row_map output;
  int error=output_region(a,view,&output);if(error)return error;
  struct mesh_writer *w=calloc(1,sizeof *w);if(!w)return ENOMEM;
  w->context=a->context;w->output=output;
  w->function=(struct mesh_row_function){.output=&w->output,.outputs=1};
  w->next=a->writers;a->writers=w;*result=w;
  return 0;
}
/* design/algorithm-sources.md#programkernel_call */
static int overlaps(struct mesh_row_map a,struct mesh_row_map b) {
  return a.first<b.first+b.count && b.first<a.first+a.count;
}
/* design/algorithm-sources.md#program */
static int bind_function(struct mesh_algebra *handle,const struct mesh_view *inputs,size_t input_count,const struct mesh_view *outputs,size_t output_count,void (*submit)(void *)) {
  MeshAlgebra *a=owner(handle);
  if(a.realized)return EBUSY;
  if(!submit || !output_count || output_count>UINT32_MAX || input_count>UINT32_MAX || !outputs || (input_count && !inputs))return EINVAL;
  MeshFunction *f=[MeshFunction new];f.owner=a;f.queue=a.queue;f.dependencies=[NSMutableData new];f.results=[NSMutableData new];
  for(size_t i=0;i<input_count;i++){if(!valid_view(a,inputs[i]))return EINVAL;dependencies(f.dependencies,inputs[i]);}
  for(size_t i=0;i<output_count;i++) {
    struct mesh_row_map m;int error=output_region(a,outputs[i],&m);if(error)return error;
    if(output_used(a,m))return EINVAL;
    struct mesh_row_map *results=f.results.mutableBytes,*reads=f.dependencies.mutableBytes;
    for(size_t j=0;j<i;j++)if(overlaps(results[j],m))return EINVAL;
    for(size_t j=0;j<f.dependencies.length/sizeof *reads;j++)if(overlaps(reads[j],m))return EINVAL;
    [f.results appendBytes:&m length:sizeof m];
  }
  f->function=(struct mesh_row_function){.output=f.results.mutableBytes,.outputs=(uint32_t)output_count};f->function.inputs=(uint32_t)merge_maps(f.dependencies,0);f->function.input=f.dependencies.mutableBytes;
  f->function.submit=submit;
  [a.functions addObject:f];return 0;
}

/* design/algorithm-sources.md#kernelsdot */
int mesh_algebra_view_pages(struct mesh_algebra *handle,struct mesh_view view,struct mesh_view *pages,size_t capacity,size_t *count) {
  MeshAlgebra *a=owner(handle);
  if(!count || !valid_view(a,view) || (!pages && capacity))return EINVAL;
  NSMutableData *coverage=[NSMutableData new];dependencies(coverage,view);merge_maps(coverage,0);
  const struct mesh_row_map *maps=coverage.bytes;size_t ranges=coverage.length/sizeof *maps;
  size_t needed=0;
  for(uint32_t i=0;i<ranges;i++)needed+=maps[i].count;
  *count=needed;
  if(!pages)return 0;
  if(capacity<needed)return ENOSPC;
  struct mesh_extent *extent=&view.tensor->extents[view.extent];
  size_t unit=a->context->M->pgsz/scalar_bytes(extent->shape.scalar),elements=extent->shape.rows*extent->shape.columns,position=0;
  for(uint32_t i=0;i<ranges;i++)for(uint32_t j=0;j<maps[i].count;j++){
    size_t offset=((size_t)maps[i].first+j-extent->first)*unit,columns=MIN(unit,elements-offset);
    pages[position++]=(struct mesh_view){.tensor=view.tensor,.extent=view.extent,.offset=offset,.rows=1,.columns=columns,.row_stride=columns,.column_stride=1};
  }
  return 0;
}
/* design/algorithm-sources.md#kernelsexpression */
static MTLCompileOptions *source_options(void) {
  MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;return options;
}
/* design/algorithm-sources.md#programkernel_call */
static void submit_metal(void *argument) {
  MeshFunction *f=(__bridge MeshFunction *)argument;
  dispatch_group_enter(f.owner.executions);
  @autoreleasepool{
    id<MTLCommandBuffer> command=[f.queue commandBuffer];f.encode(command);
    [command addCompletedHandler:^(id<MTLCommandBuffer> done){complete_part(f,done.error.code);}];
    [command commit];
  }
}
/* design/algorithm-sources.md#programkernel_call */
static void submit_cpu(void *argument) {
  MeshFunction *f=(__bridge MeshFunction *)argument;
  dispatch_group_enter(f.owner.executions);
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^{@autoreleasepool{f.execute(f);}});
}
/* design/algorithm-sources.md#programkernel_call */
static void submit_coreml(void *argument) {
  MeshFunction *f=(__bridge MeshFunction *)argument;
  dispatch_group_enter(f.owner.executions);
  @autoreleasepool{f.execute(f);}
}
/* design/algorithm-sources.md#programkernel_call */
int mesh_algebra_encode(struct mesh_algebra *handle,const struct mesh_view *inputs,size_t input_count,const struct mesh_view *outputs,size_t output_count,void (^encode)(id<MTLCommandBuffer>)) {
  MeshAlgebra *a=owner(handle);
  if(a.cpu || !encode)return EINVAL;
  int error=bind_function(handle,inputs,input_count,outputs,output_count,submit_metal);
  if(error)return error;
  MeshFunction *f=a.functions.lastObject;f.encode=encode;
  return 0;
}
/* design/algorithm-sources.md#programkernel_call */
static int bind_metal(struct mesh_algebra *handle,const char *text,size_t rows,const uint64_t domain[3],const struct mesh_view *inputs,size_t input_count,struct mesh_view output,const struct mesh_view *reads,struct mesh_view write) {
  MeshAlgebra *a=owner(handle);
  NSError *error=nil;MTLCompileOptions *options=source_options();
  NSString *source=@(text);id<MTLLibrary> library=a.libraries[source];
  if(!library){
    library=[a.device newLibraryWithSource:source options:options error:&error];
    if(library)a.libraries[source]=library;
  }
  if(!library){fprintf(stderr,"mesh Metal kernel: %s\n",error.description.UTF8String);return EINVAL;}
  id<MTLFunction> function=[library newFunctionWithName:@"mesh_expression"];
  if(!function)return ENOENT;
  MTLComputePipelineDescriptor *pipelineDescription=[MTLComputePipelineDescriptor new];
  pipelineDescription.computeFunction=function;pipelineDescription.supportIndirectCommandBuffers=YES;
  id<MTLComputePipelineState> pipeline=[a.device newComputePipelineStateWithDescriptor:pipelineDescription options:MTLPipelineOptionNone reflection:nil error:&error];
  if(!pipeline){fprintf(stderr,"mesh Metal pipeline: %s\n",error.description.UTF8String);return EINVAL;}
  if(pipeline.maxTotalThreadsPerThreadgroup<32)return EINVAL;
  NSMutableArray<id<MTLBuffer>> *resources=[a.banks mutableCopy];
  [resources addObject:a.pageTable];[resources addObject:a.bankAddresses];
  size_t count=input_count+1;
  id<MTLBuffer> addresses=[a.device newBufferWithLength:count*6*sizeof(uint64_t) options:MTLResourceStorageModeShared];
  if(!addresses)return ENOMEM;
  uint64_t *pointers=addresses.contents,*descriptors=pointers+count;
  struct hdr *m=a->context->M;
  for(size_t i=0;i<count;i++) {
    struct mesh_view v=i<input_count?inputs[i]:output;
    struct mesh_extent *extent=&v.tensor->extents[v.extent];
    pointers[i]=addresses.gpuAddress+(count+5*i)*sizeof(uint64_t);
    uint64_t *d=descriptors+5*i;
    d[0]=a.bankAddresses.gpuAddress;d[1]=a.pageTable.gpuAddress+m->page_off%m->pgsz;
    d[2]=(uint64_t)extent->first*m->pgsz+v.offset*scalar_bytes(extent->shape.scalar);
    d[3]=m->pgsz;d[4]=a.bankBytes;
  }
  id<MTLBuffer> bounds=[a.device newBufferWithBytes:domain length:3*sizeof(uint64_t) options:MTLResourceStorageModeShared];
  if(!bounds)return ENOMEM;
  MTLIndirectCommandBufferDescriptor *description=[MTLIndirectCommandBufferDescriptor new];
  description.commandTypes=MTLIndirectCommandTypeConcurrentDispatch;
  description.inheritBuffers=NO;description.inheritPipelineState=NO;
  description.maxKernelBufferBindCount=2;
  id<MTLIndirectCommandBuffer> commands=[a.device newIndirectCommandBufferWithDescriptor:description maxCommandCount:1 options:MTLResourceStorageModeShared];
  if(!commands)return ENOMEM;
  id<MTLIndirectComputeCommand> recorded=[commands indirectComputeCommandAtIndex:0];
  [recorded setComputePipelineState:pipeline];
  [recorded setKernelBuffer:addresses offset:0 atIndex:0];
  [recorded setKernelBuffer:bounds offset:0 atIndex:1];
  [recorded concurrentDispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
  [resources addObject:addresses];[resources addObject:bounds];
  int status=mesh_algebra_encode(handle,reads,input_count,&write,1,^(id<MTLCommandBuffer> command){
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    for(id<MTLBuffer> buffer in resources)[encoder useResource:buffer usage:MTLResourceUsageRead|MTLResourceUsageWrite];
    [encoder executeCommandsInBuffer:commands withRange:NSMakeRange(0,1)];
    [encoder endEncoding];
  });
  if(!status)a.functions.lastObject.metalPipeline=pipeline;
  return status;
}

/* design/algorithm-sources.md#programkernel_call */
static void publish_cpu(void *context,uint32_t first,uint32_t count) {
  mesh_publish_partial(context,first,count);
}
/* design/algorithm-sources.md#programkernel_call */
static void bind_publication(MeshFunction *f,struct mesh_view output) {
  struct mesh_extent *extent=&output.tensor->extents[output.extent];
  struct mesh_row_map map=f->function.output[0];
  size_t unit=f.owner->context->M->pgsz/scalar_bytes(extent->shape.scalar);
  size_t quantum=(size_t)extent->quantum*unit,elements=output.rows*output.columns;
  NSMutableData *sections=[NSMutableData new];uint64_t previous=0;
  for(uint32_t page=0;page<map.count;page+=extent->quantum){
    size_t begin=(size_t)page*unit,end=MIN(begin+quantum,elements);
    uint64_t row_end=output.column_stride==1?(end+output.columns-1)/output.columns:
      (begin/output.rows==(end-1)/output.rows?(end-1)%output.rows+1:output.rows);
    row_end=MAX(previous,row_end);
    size_t count=sections.length/sizeof(struct mesh_kernel_section);
    if(count && row_end==previous){
      struct mesh_kernel_section *parts=sections.mutableBytes;parts[count-1].count+=extent->quantum;
    }else{
      struct mesh_kernel_section part={.row_begin=previous,.row_end=row_end,.first=map.first+page,.count=extent->quantum};
      [sections appendBytes:&part length:sizeof part];
    }
    previous=row_end;
  }
  f.publicationSections=sections;
  f->publication=(struct mesh_kernel_publication){.sections=sections.bytes,
    .count=sections.length/sizeof(struct mesh_kernel_section),.context=f.owner->context,.publish=publish_cpu};
}
/* design/algorithm-sources.md#kernelsexpression */
int mesh_algebra_source(struct mesh_algebra *handle,const char *cpu_source,const char *metal_source,const struct mesh_view *inputs,size_t input_count,struct mesh_view output,const uint8_t *access_axes,size_t row_begin,size_t row_count,size_t column_begin,size_t column_count) {
  MeshAlgebra *a=owner(handle);
  if(a.realized || !cpu_source || !metal_source || !valid_view(a,output))return EINVAL;
  if(!row_count || row_begin>output.rows || row_count>output.rows-row_begin || !column_count || column_begin>output.columns || column_count>output.columns-column_begin || (input_count && (!inputs || !access_axes)))return EINVAL;
  struct mesh_view region=mesh_view_slice(output,row_begin,column_begin,row_count,column_count);
  NSMutableData *domains=[NSMutableData dataWithLength:input_count*sizeof(struct mesh_view)];
  struct mesh_view *reads=domains.mutableBytes;
  for(size_t i=0;i<input_count;i++) {
    if(!valid_view(a,inputs[i]) || ((access_axes[i]&1) && inputs[i].rows!=1 && inputs[i].rows!=output.rows) || ((access_axes[i]&2) && inputs[i].columns!=1 && inputs[i].columns!=output.columns))return EINVAL;
    int row=(access_axes[i]&1) && inputs[i].rows!=1,column=(access_axes[i]&2) && inputs[i].columns!=1;
    reads[i]=mesh_view_slice(inputs[i],row?row_begin:0,column?column_begin:0,row?row_count:inputs[i].rows,column?column_count:inputs[i].columns);
  }
  uint64_t domain[]={row_begin,column_begin,column_begin+column_count};
  if(!a.cpu)return bind_metal(handle,metal_source,row_count,domain,inputs,input_count,output,reads,region);
  NSString *source=[@MESH_KERNEL_SOURCE stringByAppendingString:@(cpu_source)];
  MeshCPUCode *library=a.cpuCode[source];
  if(!library) {
    library=[MeshCPUCode new];
    NSError *error=nil;NSFileManager *files=NSFileManager.defaultManager;
    NSString *directory=[NSTemporaryDirectory() stringByAppendingPathComponent:[@"mesh-cpu-" stringByAppendingString:NSUUID.UUID.UUIDString]];
    if(![files createDirectoryAtPath:directory withIntermediateDirectories:YES attributes:nil error:&error])return (int)error.code;
    NSString *input=[directory stringByAppendingPathComponent:@"kernel.c"],*output=[directory stringByAppendingPathComponent:@"kernel.dylib"];
    if(![source writeToFile:input atomically:YES encoding:NSUTF8StringEncoding error:&error]){[files removeItemAtPath:directory error:nil];return (int)error.code;}
    NSTask *task=[NSTask new];task.executableURL=[NSURL fileURLWithPath:@"/usr/bin/clang"];
    task.arguments=@[@"-O3",@"-dynamiclib",input,@"-o",output];
    if(![task launchAndReturnError:&error]){[files removeItemAtPath:directory error:nil];return (int)error.code;}
    [task waitUntilExit];
    if(task.terminationStatus){[files removeItemAtPath:directory error:nil];return EIO;}
    library.handle=dlopen(output.fileSystemRepresentation,RTLD_NOW|RTLD_LOCAL);
    [files removeItemAtPath:directory error:nil];
    if(!library.handle){fprintf(stderr,"mesh CPU kernel: %s\n",dlerror());return EIO;}
    library.kernel=(mesh_cpu_kernel)dlsym(library.handle,"mesh_expression");
    if(!library.kernel){fprintf(stderr,"mesh CPU symbol: %s\n",dlerror());dlclose(library.handle);library.handle=NULL;return EIO;}
    a.cpuCode[source]=library;
  }
  size_t count=input_count+1;
  NSMutableData *addresses=[NSMutableData dataWithLength:(count*6+1)*sizeof(uintptr_t)];
  uintptr_t *pointers=addresses.mutableBytes,*descriptors=pointers+count,*banks=pointers+count*6;
  struct hdr *m=a->context->M;*banks=(uintptr_t)mesh_at(m,0);
  for(size_t i=0;i<count;i++) {
    struct mesh_view v=i<input_count?inputs[i]:output;
    if(!valid_view(a,v))return EINVAL;
    struct mesh_extent *extent=&v.tensor->extents[v.extent];
    uintptr_t *d=descriptors+5*i;pointers[i]=(uintptr_t)d;
    d[0]=(uintptr_t)banks;d[1]=(uintptr_t)mesh_page(m);
    d[2]=(uintptr_t)extent->first*m->pgsz+v.offset*scalar_bytes(extent->shape.scalar);
    d[3]=m->pgsz;d[4]=(size_t)mesh_rows(m)*m->pgsz;
  }
  int status=bind_function(handle,reads,input_count,&region,1,submit_cpu);
  if(status)return status;
  MeshFunction *f=a.functions.lastObject;
  bind_publication(f,region);
  struct mesh_kernel_section *sections=f.publicationSections.mutableBytes;
  for(size_t i=0;i<f->publication.count;i++){sections[i].row_begin+=row_begin;sections[i].row_end+=row_begin;sections[i].column_begin=column_begin;sections[i].column_end=column_begin+column_count;}
  f.cpuCode=library;f.cpuArguments=addresses;
  f.execute=^(MeshFunction *function){
    function.cpuCode.kernel(function.cpuArguments.bytes,&function->publication);complete_part(function,0);
  };
  return 0;
}

/* design/algorithm-sources.md#kernelsdot */
static BNNSNDArrayDescriptor bnns_operand(struct mesh_view v) {
  struct mesh_extent *e=&v.tensor->extents[v.extent];
  BOOL transpose=v.column_stride!=1;
  return (BNNSNDArrayDescriptor){.layout=BNNSDataLayoutRowMajorMatrix,
    .size={transpose?v.rows:v.columns,transpose?v.columns:v.rows},
    .stride={1,transpose?v.column_stride:v.row_stride},
    .data_type=e->shape.scalar==MESH_F16?BNNSDataTypeFloat16:BNNSDataTypeFloat32,.data_scale=1};
}
/* design/algorithm-sources.md#kernelsdot */
struct mesh_matrix_part { struct mesh_view x,y,z;float beta; };
/* design/algorithm-sources.md#kernelsdot */
static void matrix_parts(NSMutableData *parts,struct mesh_view x,struct mesh_view y,struct mesh_view z,size_t k) {
  struct mesh_view operands[]={x,y,z};
  for(size_t i=0;i<3;i++) {
    struct mesh_view v=operands[i];struct mesh_extent *e=&v.tensor->extents[v.extent];struct hdr *m=v.tensor->context->M;
    if(mesh_bits_all(m,MESH_CONSTANT,e->first,e->pages))continue;
    size_t unit=(size_t)e->quantum*m->pgsz/scalar_bytes(e->shape.scalar);
    size_t sizes[]={v.rows,v.columns},strides[]={v.row_stride,v.column_stride};
    size_t end=v.offset+(v.rows-1)*v.row_stride+(v.columns-1)*v.column_stride;
    if(v.offset/unit==end/unit)continue;
    size_t fast=v.row_stride<v.column_stride?0:1,slow=1-fast;
    size_t available=unit-v.offset%unit,span=(sizes[fast]-1)*strides[fast];
    size_t axis=span>=available?fast:slow;
    size_t cut=(available-1-(axis==slow?span:0))/strides[axis]+1;
    size_t dimension=i==0?(axis==0?0:2):i==1?(axis==0?2:1):(axis==0?0:1);
    if(dimension==0) {
      matrix_parts(parts,mesh_view_slice(x,0,0,cut,x.columns),y,mesh_view_slice(z,0,0,cut,z.columns),k);
      matrix_parts(parts,mesh_view_slice(x,cut,0,x.rows-cut,x.columns),y,mesh_view_slice(z,cut,0,z.rows-cut,z.columns),k);
    } else if(dimension==1) {
      matrix_parts(parts,x,mesh_view_slice(y,0,0,y.rows,cut),mesh_view_slice(z,0,0,z.rows,cut),k);
      matrix_parts(parts,x,mesh_view_slice(y,0,cut,y.rows,y.columns-cut),mesh_view_slice(z,0,cut,z.rows,z.columns-cut),k);
    } else {
      matrix_parts(parts,mesh_view_slice(x,0,0,x.rows,cut),mesh_view_slice(y,0,0,cut,y.columns),z,k);
      matrix_parts(parts,mesh_view_slice(x,0,cut,x.rows,x.columns-cut),mesh_view_slice(y,cut,0,y.rows-cut,y.columns),z,k+cut);
    }
    return;
  }
  struct mesh_matrix_part part={x,y,z,k?1:0};[parts appendBytes:&part length:sizeof part];
}
/* design/algorithm-sources.md#kernelsdot */
static int cpu_part(MeshFunction *f,const struct mesh_matrix_part *parts,size_t count,float alpha) {
  struct mesh_view x=parts[0].x,y=parts[0].y,z=parts[0].z;

  if(x.tensor->extents[x.extent].shape.scalar==MESH_F32 && y.tensor->extents[y.extent].shape.scalar==MESH_F32 && z.tensor->extents[z.extent].shape.scalar==MESH_F32){

    struct gemm {struct mesh_matrix_address a,b,c;__LAPACK_int m,n,k,lda,ldb,ldc;enum CBLAS_TRANSPOSE tx,ty;float beta;};
    NSMutableData *calls=[NSMutableData new];
    for(size_t i=0;i<count;i++){
      struct mesh_view x=parts[i].x,y=parts[i].y,z=parts[i].z;
      BOOL tx=x.column_stride!=1,ty=y.column_stride!=1;
      struct gemm g={.a=matrix_address(x),.b=matrix_address(y),.c=matrix_address(z),
        .m=(__LAPACK_int)z.rows,.n=(__LAPACK_int)z.columns,.k=(__LAPACK_int)x.columns,.lda=(__LAPACK_int)(tx?x.column_stride:x.row_stride),
        .ldb=(__LAPACK_int)(ty?y.column_stride:y.row_stride),.ldc=(__LAPACK_int)z.row_stride,.tx=tx?CblasTrans:CblasNoTrans,.ty=ty?CblasTrans:CblasNoTrans,.beta=parts[i].beta};
      [calls appendBytes:&g length:sizeof g];
    }
    f.execute=^(MeshFunction *function){
      const struct gemm *g=calls.bytes;
      for(size_t i=0;i<calls.length/sizeof *g;i++)
        cblas_sgemm(CblasRowMajor,g[i].tx,g[i].ty,g[i].m,g[i].n,g[i].k,alpha,
          g[i].a.base+matrix_offset(g[i].a),g[i].lda,g[i].b.base+matrix_offset(g[i].b),g[i].ldb,
          g[i].beta,g[i].c.base+matrix_offset(g[i].c),g[i].ldc);
      complete_part(function,0);
    };
    return 0;
  }

  struct gemm {BNNSFilter filter;struct mesh_matrix_address x,y,z;};
  NSMutableData *calls=[NSMutableData new];f.bnnsFilters=[NSMutableData new];
  BOOL tx=x.column_stride!=1,ty=y.column_stride!=1;
  for(size_t i=0;i<count;i++){
    BNNSLayerParametersBroadcastMatMul parameters={.alpha=alpha,.beta=parts[i].beta,.transA=tx,.transB=ty,
      .iA_desc=bnns_operand(parts[i].x),.iB_desc=bnns_operand(parts[i].y),.o_desc=bnns_operand(parts[i].z)};
    BNNSFilter filter=BNNSFilterCreateLayerBroadcastMatMul(&parameters,NULL);if(!filter)return EINVAL;
    [f.bnnsFilters appendBytes:&filter length:sizeof filter];
    struct gemm g={filter,matrix_address(parts[i].x),matrix_address(parts[i].y),matrix_address(parts[i].z)};
    [calls appendBytes:&g length:sizeof g];
  }
  f.execute=^(MeshFunction *function){
    const struct gemm *g=calls.bytes;int error=0;
    for(size_t i=0;i<calls.length/sizeof *g && !error;i++)
      error=BNNSFilterApplyTwoInput(g[i].filter,g[i].x.base+matrix_offset(g[i].x),
        g[i].y.base+matrix_offset(g[i].y),g[i].z.base+matrix_offset(g[i].z));
    complete_part(function,error);
  };
  return 0;
}
/* design/algorithm-sources.md#kernelsdot */
static MLMultiArray *native_array(struct mesh_view v,NSError **error) {
  struct mesh_extent *e=&v.tensor->extents[v.extent];size_t bytes=scalar_bytes(e->shape.scalar);
  return [[MLMultiArray alloc]initWithDataPointer:(char *)e->address+v.offset*bytes shape:@[@(v.rows),@(v.columns)] dataType:bytes==2?MLMultiArrayDataTypeFloat16:MLMultiArrayDataTypeFloat32 strides:@[@(v.row_stride),@(v.column_stride)] deallocator:nil error:error];
}
/* design/algorithm-sources.md#kernelsdot */
static int native_part(MeshAlgebra *a,MeshFunction *f,NSArray *rectangles,NSDictionary *features,struct mesh_view z,size_t first,size_t count,float alpha) {
  NSError *error=nil;
  NSDictionary *spec=@{@"rectangles":rectangles,@"alpha":@(alpha),@"output_half":@(z.tensor->extents[z.extent].shape.scalar==MESH_F16)};
  NSData *json=[NSJSONSerialization dataWithJSONObject:spec options:NSJSONWritingSortedKeys error:&error];
  NSString *key=[[NSString alloc]initWithData:json encoding:NSUTF8StringEncoding];
  MLModel *model=a.models[key];
  if(!model) {
    NSMutableData *specialization=[json mutableCopy];
    NSData *generator=[NSData dataWithContentsOfFile:a.coremlGenerator options:0 error:&error];
    if(!generator)return error?(int)error.code:EIO;
    [specialization appendData:generator];
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256(specialization.bytes,(CC_LONG)specialization.length,digest);
    NSMutableString *name=[NSMutableString stringWithString:@"part-"];
    for(size_t i=0;i<sizeof digest;i++)[name appendFormat:@"%02x",digest[i]];
    NSString *stem=[a.coremlCache stringByAppendingPathComponent:name];
    [[NSFileManager defaultManager]createDirectoryAtPath:a.coremlCache withIntermediateDirectories:YES attributes:nil error:&error];
    NSString *request=[stem stringByAppendingPathExtension:@"json"];
    if(![json writeToFile:request options:NSDataWritingAtomic error:&error])return (int)error.code;
    NSURL *compiled=[NSURL fileURLWithPath:[stem stringByAppendingPathExtension:@"mlmodelc"]];
    if(![[NSFileManager defaultManager]fileExistsAtPath:compiled.path]) {
      NSTask *task=[NSTask new];task.executableURL=[NSURL fileURLWithPath:a.coremlPython];
      task.arguments=@[a.coremlGenerator,request,stem];
      if(![task launchAndReturnError:&error])return (int)error.code;
      [task waitUntilExit];if(task.terminationStatus)return EIO;
    }
    MLModelConfiguration *configuration=[MLModelConfiguration new];configuration.computeUnits=MLComputeUnitsCPUAndNeuralEngine;
    model=[MLModel modelWithContentsOfURL:compiled configuration:configuration error:&error];
    if(!model){fprintf(stderr,"mesh CoreML binding: %s\n",error.description.UTF8String);return (int)error.code;}
    a.models[key]=model;
  }
  struct mesh_extent *out=&z.tensor->extents[z.extent];size_t bytes=scalar_bytes(out->shape.scalar);
  MLMultiArray *output=[[MLMultiArray alloc]initWithDataPointer:(char *)out->address+(z.offset+first)*bytes shape:@[@(count)] dataType:bytes==2?MLMultiArrayDataTypeFloat16:MLMultiArrayDataTypeFloat32 strides:@[@1] deallocator:nil error:&error];
  MLDictionaryFeatureProvider *inputs=[[MLDictionaryFeatureProvider alloc]initWithDictionary:features error:&error];
  if(!output || !inputs)return (int)error.code;
  MLPredictionOptions *options=[MLPredictionOptions new];options.outputBackings=@{@"z":output};
  f.execute=^(MeshFunction *function) {
    [model predictionFromFeatures:inputs options:options completionHandler:^(id<MLFeatureProvider> prediction,NSError *failure){
      MLMultiArray *actual=[prediction featureValueForName:@"z"].multiArrayValue;
      BOOL backing=actual && actual.dataPointer==output.dataPointer && actual.dataType==output.dataType && [actual.shape isEqualToArray:output.shape] && [actual.strides isEqualToArray:output.strides];
      complete_part(function,failure?failure.code:backing?0:EPROTO);
    }];
  };
  return 0;
}
/* design/algorithm-sources.md#kernelsdot */
static int bind_part(MeshAlgebra *a,struct mesh_view x,struct mesh_view y,struct mesh_view z,float alpha,size_t first,size_t count) {
  MeshFunction *f=[MeshFunction new];f.owner=a;f.queue=a.queue;f.dependencies=[NSMutableData new];
  struct mesh_extent *out=&z.tensor->extents[z.extent];
  size_t scalar=scalar_bytes(out->shape.scalar);
  f->output=(struct mesh_row_map){.first=out->first+(uint32_t)((z.offset+first)*scalar/a->context->M->pgsz),.count=out->quantum};
  BOOL dense=z.column_stride==1;
  size_t columns=dense?z.columns:z.rows;
  NSMutableData *parts=[NSMutableData new];
  NSMutableArray *rectangles=[NSMutableArray new];NSMutableDictionary *features=[NSMutableDictionary new];
  for(size_t at=first,left=count;left;) {
    size_t row=at/columns,column=at%columns,nr=1,nc=MIN(left,columns-column);
    if(!column && left>=columns){nr=left/columns;nc=columns;}
    size_t zr=dense?row:column,zc=dense?column:row,rr=dense?nr:nc,cc=dense?nc:nr;
    struct mesh_view xv=mesh_view_slice(x,zr,0,rr,x.columns);
    struct mesh_view yv=mesh_view_slice(y,0,zc,y.rows,cc);
    struct mesh_view zv=mesh_view_slice(z,zr,zc,rr,cc);
    dependencies(f.dependencies,xv);dependencies(f.dependencies,yv);
    if(!a.cpu && a.coremlPython) {
      NSError *error=nil;MLMultiArray *left=native_array(xv,&error),*right=native_array(yv,&error);
      if(!left || !right)return (int)error.code;
      NSUInteger i=rectangles.count;
      features[[NSString stringWithFormat:@"x%lu",(unsigned long)i]]=left;
      features[[NSString stringWithFormat:@"w%lu",(unsigned long)i]]=right;
      [rectangles addObject:@[@(rr),@(cc),@(x.columns),@(xv.tensor->extents[xv.extent].shape.scalar==MESH_F16),@(yv.tensor->extents[yv.extent].shape.scalar==MESH_F16)]];
    } else matrix_parts(parts,xv,yv,zv,0);
    at+=nr*nc;left-=nr*nc;
  }
  f->function=(struct mesh_row_function){.output=&f->output,.outputs=1};
  f->function.inputs=(uint32_t)merge_maps(f.dependencies,0);f->function.input=f.dependencies.mutableBytes;
  if(a.cpu) {
    f->function.submit=submit_cpu;
    int error=cpu_part(f,parts.bytes,parts.length/sizeof(struct mesh_matrix_part),alpha);if(error)return error;
  } else if(a.coremlPython) {
    f->function.submit=submit_coreml;
    int error=native_part(a,f,rectangles,features,z,first,count,alpha);if(error)return error;
  } else {
    f->function.submit=submit_metal;
    NSMutableArray<MPSMatrixMultiplication *> *products=[NSMutableArray new];
    NSMutableArray<NSArray<MPSMatrix *> *> *matrices=[NSMutableArray new];
    NSMutableData *addresses=[NSMutableData new];
    const struct mesh_matrix_part *values=parts.bytes;
    for(size_t i=0;i<parts.length/sizeof *values;i++) {
      struct mesh_matrix_part p=values[i];BOOL tx=p.x.column_stride!=1,ty=p.y.column_stride!=1;
      struct mesh_mps_address bindings[3];
      [matrices addObject:matrix(a,p.x,tx,NO,&bindings[0])];
      [matrices addObject:matrix(a,p.y,ty,NO,&bindings[1])];
      [matrices addObject:matrix(a,p.z,NO,YES,&bindings[2])];
      [addresses appendBytes:bindings length:sizeof bindings];
      [products addObject:[[MPSMatrixMultiplication alloc]initWithDevice:a.device transposeLeft:tx transposeRight:ty resultRows:p.z.rows resultColumns:p.z.columns interiorColumns:p.x.columns alpha:alpha beta:p.beta]];
    }
    f.encode=^(id<MTLCommandBuffer> command){
      const struct mesh_mps_address *bindings=addresses.bytes;
      for(NSUInteger i=0;i<products.count;i++) {
        MTLOrigin origins[3];MPSMatrix *operands[3];
        for(size_t j=0;j<3;j++)operands[j]=matrix_resolve(matrices[3*i+j],bindings[3*i+j],&origins[j]);
        products[i].leftMatrixOrigin=origins[0];products[i].rightMatrixOrigin=origins[1];products[i].resultMatrixOrigin=origins[2];
        [products[i] encodeToCommandBuffer:command leftMatrix:operands[0] rightMatrix:operands[1] resultMatrix:operands[2]];
      }
    };
  }
  [a.functions addObject:f];return 0;
}
/* design/algorithm-sources.md#kernelsdot */
static int contraction_views(MeshAlgebra *a,struct mesh_view *left,struct mesh_view *right,struct mesh_view *output) {
  struct mesh_view x=*left,y=*right,z=*output;
  if(!valid_view(a,x) || !valid_view(a,y) || !valid_view(a,z))return EINVAL;
  enum mesh_scalar xs=x.tensor->extents[x.extent].shape.scalar,ys=y.tensor->extents[y.extent].shape.scalar,zs=z.tensor->extents[z.extent].shape.scalar;
  if(xs>MESH_F32 || ys>MESH_F32 || zs>MESH_F32)return EINVAL;
  if(x.columns!=y.rows || z.rows!=x.rows || z.columns!=y.columns || (z.column_stride!=1 && z.row_stride!=1) || !x.row_stride || !x.column_stride || !y.row_stride || !y.column_stride || (x.column_stride!=1 && x.row_stride!=1) || (y.column_stride!=1 && y.row_stride!=1))return EINVAL;
  BOOL reverse=xs==MESH_F16 && ys==MESH_F32;
  if(z.row_stride==1 && (z.column_stride!=1 || reverse)){
    struct mesh_view swap=mesh_view_transpose(y);y=mesh_view_transpose(x);x=swap;z=mesh_view_transpose(z);
  }
  if(!a.cpu && !a.coremlPython && x.tensor->extents[x.extent].shape.scalar!=y.tensor->extents[y.extent].shape.scalar && (x.tensor->extents[x.extent].shape.scalar!=MESH_F32 || zs!=MESH_F32))return EINVAL;
  *left=x;*right=y;*output=z;return 0;
}
/* design/algorithm-sources.md#programkernel_call */
int mesh_algebra_contract(struct mesh_algebra *handle,struct mesh_view x,struct mesh_view y,struct mesh_view z,float alpha) {
  MeshAlgebra *a=owner(handle);if(a.realized)return EBUSY;
  if(!valid_view(a,x) || !valid_view(a,z) || !valid_view(a,y))return EINVAL;
  struct mesh_extent *out=&z.tensor->extents[z.extent];size_t elements=z.rows*z.columns;
  struct mesh_row_map output;int region_error=output_region(a,z,&output);if(region_error)return region_error;
  if(output_used(a,output))return EINVAL;
  if(x.tensor->extents[x.extent].shape.scalar>MESH_F32 || out->shape.scalar>MESH_F32 || y.tensor->extents[y.extent].shape.scalar>MESH_F32)return EINVAL;
  int error=contraction_views(a,&x,&y,&z);if(error)return error;
  NSMutableData *reads=[NSMutableData new];dependencies(reads,x);dependencies(reads,y);
  struct mesh_row_map *maps=reads.mutableBytes;
  for(size_t i=0;i<reads.length/sizeof *maps;i++)if(overlaps(maps[i],output))return EINVAL;
  size_t step=(size_t)out->quantum*a->context->M->pgsz/(scalar_bytes(out->shape.scalar));
  for(size_t first=0;first<elements;first+=step){int error=bind_part(a,x,y,z,alpha,first,MIN(step,elements-first));if(error)return error;}
  return 0;
}

/* design/algorithm-sources.md#programcopy */
struct mesh_copy_segment { size_t source,destination,elements,stride,bytes; };

/* design/algorithm-sources.md#programcopy */
int mesh_algebra_materialize(struct mesh_algebra *handle,const struct mesh_copy_region *regions,size_t count,struct mesh_view destination) {
  MeshAlgebra *a=owner(handle);
  if(a.realized)return EBUSY;
  if(!regions || !count || count>SIZE_MAX/sizeof *regions || !valid_view(a,destination))return EINVAL;
  __attribute__((objc_precise_lifetime)) NSMutableData *transposed=nil;
  if(destination.column_stride!=1) {
    destination=mesh_view_transpose(destination);
    transposed=[NSMutableData dataWithLength:count*sizeof *regions];
    struct mesh_copy_region *mapped=transposed.mutableBytes;
    for(size_t i=0;i<count;i++)mapped[i]=(struct mesh_copy_region){.source=mesh_view_transpose(regions[i].source),.row=regions[i].column,.column=regions[i].row};
    regions=transposed.bytes;
  }
  struct mesh_extent *d=&destination.tensor->extents[destination.extent];
  size_t elements=destination.rows*destination.columns,scalar=scalar_bytes(d->shape.scalar),covered=0;
  struct mesh_ctx *context=a->context;size_t page=context->M->pgsz;
  if((destination.rows!=1 && destination.row_stride!=destination.columns) || destination.column_stride!=1)return EINVAL;
  struct mesh_row_map output;int error=output_region(a,destination,&output);if(error)return error;
  if(output_used(a,output))return EINVAL;
  for(size_t i=0;i<count;i++) {
    struct mesh_copy_region r=regions[i];
    if(!valid_view(a,r.source) || r.source.tensor->extents[r.source.extent].shape.scalar!=d->shape.scalar)return EINVAL;
    NSMutableData *reads=[NSMutableData new];dependencies(reads,r.source);
    const struct mesh_row_map *maps=reads.bytes;
    for(size_t j=0;j<reads.length/sizeof *maps;j++)if(overlaps(maps[j],output))return EINVAL;
    if(r.row>destination.rows || r.column>destination.columns || r.source.rows>destination.rows-r.row || r.source.columns>destination.columns-r.column)return EINVAL;
    size_t area=r.source.rows*r.source.columns;
    if(area>elements-covered)return EINVAL;
    covered+=area;
    for(size_t j=0;j<i;j++) {
      struct mesh_copy_region q=regions[j];
      if(r.row<q.row+q.source.rows && q.row<r.row+r.source.rows && r.column<q.column+q.source.columns && q.column<r.column+r.source.columns)return EINVAL;
    }
  }
  if(covered!=elements)return EINVAL;
  size_t step=(size_t)d->quantum*a->context->M->pgsz/scalar;
  for(size_t first=0;first<elements;first+=step) {
    size_t end=MIN(elements,first+step);
    MeshFunction *f=[MeshFunction new];f.owner=a;f.queue=a.queue;f.dependencies=[NSMutableData new];
    NSMutableData *segments=[NSMutableData new];
    for(size_t i=0;i<count;i++) {
      struct mesh_copy_region r=regions[i];
      size_t row_end=MIN(r.row+r.source.rows,(end-1)/destination.columns+1);
      for(size_t row=MAX(r.row,first/destination.columns);row<row_end;row++) {
        size_t base=row*destination.columns,lo=MAX(first,base+r.column),hi=MIN(end,base+r.column+r.source.columns);
        if(lo>=hi)continue;
        struct mesh_view source=mesh_view_slice(r.source,row-r.row,lo-base-r.column,1,hi-lo);
        dependencies(f.dependencies,source);
        size_t sourceIndex=(size_t)source.tensor->extents[source.extent].first*page+source.offset*scalar;
        size_t destinationIndex=(size_t)d->first*page+(destination.offset+lo)*scalar;
        for(size_t copied=0;copied<hi-lo;){
          struct mesh_copy_segment segment={.source=sourceIndex+copied*source.column_stride*scalar,
            .destination=destinationIndex+copied*scalar,.stride=source.column_stride*scalar,.bytes=scalar};
          size_t available=segment.stride?1+(page-segment.source%page-scalar)/segment.stride:hi-lo-copied;
          segment.elements=MIN(hi-lo-copied,MIN(available,(page-segment.destination%page)/scalar));
          copied+=segment.elements;
          if(segment.stride==scalar){segment.bytes*=segment.elements;segment.elements=1;}
          struct mesh_copy_segment *previous=segments.length?(struct mesh_copy_segment *)segments.mutableBytes+segments.length/sizeof segment-1:NULL;
          if(previous && previous->elements==1 && segment.elements==1 &&
             previous->source/page==segment.source/page && previous->destination/page==segment.destination/page &&
             previous->source+previous->bytes==segment.source && previous->destination+previous->bytes==segment.destination)previous->bytes+=segment.bytes;
          else [segments appendBytes:&segment length:sizeof segment];
        }
      }
    }
    f->output=(struct mesh_row_map){.first=output.first+(uint32_t)(first*scalar/a->context->M->pgsz),.count=d->quantum};
    f->function=(struct mesh_row_function){.output=&f->output,.outputs=1};f->function.inputs=(uint32_t)merge_maps(f.dependencies,0);f->function.input=f.dependencies.mutableBytes;

    f->function.submit=submit_cpu;
    f.execute=^(MeshFunction *function){
      const struct mesh_copy_segment *parts=segments.bytes;
      for(size_t i=0;i<segments.length/sizeof *parts;i++) {
        struct mesh_copy_segment p=parts[i];
        const char *source=mesh_row_data(context,(uint32_t)(p.source/page));
        char *destination=mesh_row_data(context,(uint32_t)(p.destination/page));
        source+=p.source%page;destination+=p.destination%page;
        for(size_t j=0;j<p.elements;j++)memcpy(destination+j*p.bytes,source+j*p.stride,p.bytes);
      }
      complete_part(function,0);
    };
    [a.functions addObject:f];
  }
  return 0;
}
/* design/algorithm-sources.md#programcopy */
int mesh_algebra_copy(struct mesh_algebra *handle,struct mesh_view source,uint32_t sender,struct mesh_view destination,uint32_t receiver,uint16_t queue) {
  MeshAlgebra *a=owner(handle);
  if(a.realized)return EBUSY;
  if(!valid_view(a,source) || !valid_view(a,destination))return EINVAL;
  struct mesh_extent *s=&source.tensor->extents[source.extent],*d=&destination.tensor->extents[destination.extent];
  if(source.rows!=destination.rows || source.columns!=destination.columns || s->shape.scalar!=d->shape.scalar)return EINVAL;
  if(sender==receiver) {
    if(sender!=a->context->M->node)return 0;
    struct mesh_copy_region region={.source=source};
    return mesh_algebra_materialize(handle,&region,1,destination);
  }
  if(source.rows>1 && source.column_stride==1 && destination.column_stride==1 &&
     (source.row_stride!=source.columns || destination.row_stride!=destination.columns)) {
    if(destination.row_stride<destination.columns)return EINVAL;
    for(size_t row=0;row<source.rows;row++) {
      int error=mesh_algebra_copy(handle,mesh_view_slice(source,row,0,1,source.columns),sender,
        mesh_view_slice(destination,row,0,1,destination.columns),receiver,queue);if(error)return error;
    }
    return 0;
  }
  if(source.columns>1 && source.row_stride==1 && destination.row_stride==1 &&
     (source.column_stride!=source.rows || destination.column_stride!=destination.rows)) {
    if(destination.column_stride<destination.rows)return EINVAL;
    for(size_t column=0;column<source.columns;column++) {
      int error=mesh_algebra_copy(handle,mesh_view_slice(source,0,column,source.rows,1),sender,
        mesh_view_slice(destination,0,column,destination.rows,1),receiver,queue);if(error)return error;
    }
    return 0;
  }
  struct mesh_row_map src,dst;
  int error=output_region(a,source,&src);if(error)return error;
  error=output_region(a,destination,&dst);if(error)return error;
  if((source.rows>1 && source.row_stride!=destination.row_stride) || (source.columns>1 && source.column_stride!=destination.column_stride))return EINVAL;
  size_t bytes=source.rows*source.columns*scalar_bytes(s->shape.scalar);
  uint32_t block=a->context->M->block;
  if(src.count!=dst.count || src.count%block)return EINVAL;
  if(src.count/block>UINT32_MAX-a->copies)return EOVERFLOW;
  for(uint32_t offset=0;offset<src.count;offset+=block) {
    uint32_t identity=a->copies++;
    if(sender==a->context->M->node || receiver==a->context->M->node) {
      int receive=receiver==a->context->M->node;
      struct mesh_row_binding binding={.first=(receive?dst:src).first+offset,.count=block,
        .bytes=(uint32_t)MIN((size_t)block*a->context->M->pgsz,bytes-(size_t)offset*a->context->M->pgsz),
        .binding=identity,.queue=queue,.receive=receive};
      [a.bindings appendBytes:&binding length:sizeof binding];
    }
  }
  return 0;
}

/* design/algorithm-sources.md#programexport */
int mesh_algebra_export(struct mesh_algebra *handle,struct mesh_view view,size_t *first,size_t *count) {
  MeshAlgebra *a=owner(handle);
  if(a.realized)return EBUSY;
  if(!first || !count || !valid_view(a,view))return EINVAL;
  *first=a.returns.length/sizeof(struct mesh_row_map);
  dependencies(a.returns,view);
  *count=merge_maps(a.returns,*first);return 0;
}
/* design/algorithm-sources.md#programkernel_call */
int mesh_algebra_realize(struct mesh_algebra *handle) {
  MeshAlgebra *a=owner(handle);if(a.realized)return 0;size_t count=a.functions.count;
  struct mesh_row_function **functions=calloc(count?count:1,sizeof *functions);
  if(!functions)return ENOMEM;
  for(size_t i=0;i<count;i++)functions[i]=&a.functions[i]->function;
  int error=mesh_realize(a->context,functions,count,a.bindings.mutableBytes,a.bindings.length/sizeof(struct mesh_row_binding),a.returns.mutableBytes,a.returns.length/sizeof(struct mesh_row_map));
  free(functions);
  if(!error){
    size_t arenaBytes=0;
    struct mesh_tensor **tensors=a.tensors.mutableBytes;
    for(size_t i=0;i<a.tensors.length/sizeof *tensors;i++)
      for(size_t j=0;j<tensors[i]->count;j++)arenaBytes+=tensors[i]->extents[j].bytes;
    fprintf(stderr,"mesh realize: participant=%u planned_arena_bytes=%zu\n",a->context->M->node,arenaBytes);
    a.realized=YES;
    for(MeshFunction *f in a.functions){
      error=mesh_execution_add(a->context,&f->function,handle,(__bridge void *)f);
      if(error)break;
    }
  }
  return error;
}

/* design/algorithm-sources.md#programkernel_call */
int mesh_algebra_available(struct mesh_algebra *handle,size_t index) {
  MeshAlgebra *a=owner(handle);if(index>=a.returns.length/sizeof(struct mesh_row_map))return 0;
  return mesh_available(a->context,((struct mesh_row_map *)a.returns.bytes)[index]);
}
/* design/algorithm-sources.md#programkernel_call */
void mesh_algebra_consume(struct mesh_algebra *handle,size_t index) {
  MeshAlgebra *a=owner(handle);if(index<a.returns.length/sizeof(struct mesh_row_map))mesh_consume(a->context,((struct mesh_row_map *)a.returns.bytes)[index]);
}
/* design/algorithm-sources.md#programkernel_call */
struct mesh_algebra_report mesh_algebra_report(struct mesh_algebra *handle) {
  MeshAlgebra *a=owner(handle);
  int64_t code=atomic_load(&a->code);
  if(!code && a.bindings.length)code=atomic_load(&a->context->M->port.code);
  return (struct mesh_algebra_report){.code=code};
}
