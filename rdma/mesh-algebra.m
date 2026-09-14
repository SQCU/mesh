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
#include <time.h>

/* design/algorithm-sources.md#application-metal-kernels */
static size_t scalar_bytes(enum mesh_scalar scalar) {
  static const size_t bytes[]={2,4,4,4,8,8,1,1};
  return (unsigned)scalar<sizeof bytes/sizeof *bytes?bytes[scalar]:0;
}

struct mesh_extent {
  uint32_t first,pages,quantum;
  size_t bytes;
  void *address;
  struct mesh_shape shape;
};
struct mesh_tensor { struct mesh_ctx *context; size_t count; struct mesh_extent *extents; };
struct mesh_writer { struct mesh_ctx *context; struct mesh_row_map output; struct mesh_row_function function; struct mesh_writer *next; };

@interface MeshExtent : NSObject
@property struct mesh_extent extent;
@property id<MTLBuffer> buffer;
@end
@implementation MeshExtent
@end

enum mesh_execution_kind { MESH_EXECUTION_CPU, MESH_EXECUTION_METAL, MESH_EXECUTION_COREML };

typedef void (*mesh_cpu_kernel)(const uintptr_t *,const struct mesh_kernel_publication *);
@interface MeshCode : NSObject
@property NSString *source;
@end
@implementation MeshCode
@end
@interface MeshMetalCode : MeshCode
@property id<MTLLibrary> library;
@end
@implementation MeshMetalCode
@end
@interface MeshCPUCode : MeshCode
@property void *handle;
@property mesh_cpu_kernel kernel;
@end
@implementation MeshCPUCode
/* design/algorithm-sources.md#region-expression-fusion */
- (void)dealloc {if(self.handle)dlclose(self.handle);}
@end

@class MeshAlgebra;
@interface MeshFunction : NSObject {
@public
  struct mesh_row_map output;
  struct mesh_row_function function;
  struct mesh_kernel_publication publication;
  uint32_t occurrence;
  enum mesh_execution_kind executionKind;
}
@property NSArray<MeshExtent *> *operands;
@property MeshCPUCode *cpuCode;
@property id<MTLComputePipelineState> metalPipeline;
@property id<MTLCommandQueue> queue;
@property NSData *cpuArguments;
@property NSMutableData *publicationSections;
@property NSMutableData *dependencies,*results;
@property NSData *inputViews;
@property NSMutableIndexSet *indexedInputs;
@property(nonatomic,assign) MeshAlgebra *owner;
@property(copy) void (^encode)(id<MTLCommandBuffer>);
@property(copy) void (^execute)(MeshFunction *);
@end
@implementation MeshFunction
/* design/algorithm-sources.md#dynamic-reader-lifetimes */
- (void)dealloc {
  struct mesh_indexed_read *d=function.indexed;
  while(d){
    struct mesh_indexed_read *next=d->next;
    for(uint32_t i=0;i<d->candidates;i++)free(d->candidate[i].maps);
    free(d->candidate);free(d->selector);free(d);d=next;
  }
}
@end

@interface MeshAlgebra : NSObject {
@public
  struct mesh_ctx *context;
  struct mesh_writer *writers;
  _Atomic uint64_t submitted;
  uint32_t copies;
  _Atomic uint64_t completed;
  _Atomic int64_t code;
}
@property BOOL realized,cpu;
@property dispatch_group_t executions;
@property id<MTLDevice> device;
@property id<MTLCommandQueue> queue;
@property NSMutableDictionary<NSString *,MeshMetalCode *> *libraries;
@property NSMutableDictionary<NSString *,MeshCPUCode *> *cpuCode;
@property NSMutableArray<MeshFunction *> *functions;
@property NSMutableArray<MeshExtent *> *extents;
@property NSMutableDictionary<NSValue *,MeshExtent *> *lookup;
@property NSMutableData *tensors;
@property NSMutableData *bindings;
@property NSMutableData *returns;
@property NSString *coremlPython,*coremlGenerator,*coremlCache;
@property NSMutableDictionary<NSString *,MLModel *> *models;
@end
@implementation MeshAlgebra
/* design/algorithm-sources.md#streaming-algebra */
- (void)dealloc {
  for(MeshFunction *f in self.functions){
    for(uint32_t i=0;i<f->function.inputs;i++)mesh_reader_unbind(context,&f->function.input[i]);
    for(struct mesh_indexed_read *d=f->function.indexed;d;d=d->next){
      for(uint32_t i=0;i<d->selectors;i++)mesh_reader_unbind(context,&d->selector[i]);
      for(uint32_t i=0;i<d->candidates;i++)for(uint32_t j=0;j<d->candidate[i].count;j++)mesh_reader_unbind(context,&d->candidate[i].maps[j]);
      mesh_rows_release(context,d->retired,2*d->candidates+2);
    }
  }
  while(writers){struct mesh_writer *next=writers->next;free(writers);writers=next;}
  struct mesh_row_map *returns=self.returns.mutableBytes;
  for(size_t i=0;i<self.returns.length/sizeof *returns;i++)mesh_reader_unbind(context,&returns[i]);
  self.functions=nil;
  self.lookup=nil;
  self.extents=nil;
  struct mesh_tensor **tensors=self.tensors.mutableBytes;
  for(size_t i=0;i<self.tensors.length/sizeof *tensors;i++) {
    struct mesh_tensor *t=tensors[i];
    for(size_t j=0;j<t->count;j++) {
      struct mesh_extent *e=&t->extents[j];
      if(e->address)mesh_view_destroy(e->address,e->bytes);
      if(e->first!=MESH_ABSENT){mesh_backing_release(context,e->first,e->pages);mesh_rows_release(context,e->first,e->pages);}
    }
    free(t->extents); free(t);
  }
}
@end

/* design/algorithm-sources.md#indexed-library-functions */
static void complete_part(MeshFunction *f,int64_t error) {
  MeshAlgebra *a=f.owner;
  if(error)atomic_store(&a->code,error);
  else mesh_complete(a->context,&f->function,&f->occurrence,1);
  atomic_fetch_add(&a->completed,1);dispatch_group_leave(a.executions);
}
/* design/algorithm-sources.md#streaming-algebra */
static MeshAlgebra *owner(struct mesh_algebra *a) { return (__bridge MeshAlgebra *)a; }
/* design/algorithm-sources.md#streaming-algebra */
static struct mesh_algebra *create_algebra(struct mesh_ctx *context,BOOL cpu) {
  if(!context || !context->M){errno=EINVAL;return NULL;}
  MeshAlgebra *a=[MeshAlgebra new]; a->context=context;a.cpu=cpu;
  if(!cpu) {
    a.device=MTLCreateSystemDefaultDevice(); a.queue=[a.device newCommandQueue];
    if(!a.queue){errno=ENODEV;return NULL;}
  }
  a.executions=dispatch_group_create();
  a.libraries=[NSMutableDictionary new];a.cpuCode=[NSMutableDictionary new];
  a.functions=[NSMutableArray new]; a.extents=[NSMutableArray new]; a.lookup=[NSMutableDictionary new];
  a.tensors=[NSMutableData new]; a.bindings=[NSMutableData new]; a.returns=[NSMutableData new];
  return (__bridge_retained struct mesh_algebra *)a;
}
/* design/algorithm-sources.md#cpu-indexed-execution */
struct mesh_algebra *mesh_algebra_create(struct mesh_ctx *context) {return create_algebra(context,NO);}
/* design/algorithm-sources.md#cpu-indexed-execution */
struct mesh_algebra *mesh_algebra_create_cpu(struct mesh_ctx *context) {return create_algebra(context,YES);}
/* design/algorithm-sources.md#independent-kernel-submission */
int mesh_algebra_kernel(struct mesh_algebra *handle) {
  MeshAlgebra *a=owner(handle);if(a.realized)return EBUSY;
  if(!a.cpu){a.queue=[a.device newCommandQueue];if(!a.queue)return ENOMEM;}
  return 0;
}
/* design/algorithm-sources.md#application-metal-kernels */
uint32_t mesh_algebra_node(struct mesh_algebra *handle) {return owner(handle)->context->M->node;}
/* design/algorithm-sources.md#page-derived-reduction-leaves */
size_t mesh_algebra_page_bytes(struct mesh_algebra *handle) {return owner(handle)->context->M->pgsz;}
/* design/algorithm-sources.md#publication-layout */
size_t mesh_algebra_publication_bytes(struct mesh_algebra *handle) {
  struct hdr *m=owner(handle)->context->M; return (size_t)m->block*m->pgsz;
}
/* design/algorithm-sources.md#coreml-partial-execution */
int mesh_algebra_coreml(struct mesh_algebra *handle,const char *python,const char *generator,const char *cache) {
  MeshAlgebra *a=owner(handle);
  if(a.realized || a.functions.count)return EBUSY;
  if(a.cpu || !python || !generator || !cache)return EINVAL;
  a.coremlPython=@(python);a.coremlGenerator=@(generator);a.coremlCache=@(cache);a.models=[NSMutableDictionary new];return 0;
}
/* design/algorithm-sources.md#streaming-algebra */
void mesh_algebra_destroy(struct mesh_algebra *handle) {
  if(!handle)return;MeshAlgebra *a=owner(handle);
  mesh_execution_remove(a->context,handle);
  dispatch_group_wait(a.executions,DISPATCH_TIME_FOREVER);
  CFBridgingRelease(handle);
}

/* design/algorithm-sources.md#streaming-algebra */
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
    e->address=mesh_view_create(a->context,e->first,pages);
    if(!e->address)return NULL;
    MeshExtent *storage=[MeshExtent new]; storage.extent=*e;
    if(!a.cpu) {
      storage.buffer=[a.device newBufferWithBytesNoCopy:e->address length:e->bytes options:MTLResourceStorageModeShared|MTLResourceHazardTrackingModeUntracked deallocator:nil];
      if(!storage.buffer){errno=ENOMEM;return NULL;}
    }
    [a.extents addObject:storage]; a.lookup[[NSValue valueWithPointer:e]]=storage;
  }
  return t;
}

/* design/algorithm-sources.md#streaming-algebra */
struct mesh_view mesh_tensor_view(struct mesh_tensor *t,uint32_t i) {
  if(!t || i>=t->count)return (struct mesh_view){0};
  struct mesh_shape s=t->extents[i].shape;
  return (struct mesh_view){t,i,0,s.rows,s.columns,s.columns,1};
}
/* design/algorithm-sources.md#streaming-algebra */
struct mesh_view mesh_view_slice(struct mesh_view v,size_t row,size_t column,size_t rows,size_t columns) {
  if(row>v.rows || column>v.columns || rows>v.rows-row || columns>v.columns-column)return (struct mesh_view){0};
  v.offset+=row*v.row_stride+column*v.column_stride;v.rows=rows;v.columns=columns;return v;
}
/* design/algorithm-sources.md#streaming-algebra */
struct mesh_view mesh_view_transpose(struct mesh_view v) {
  size_t n=v.rows;v.rows=v.columns;v.columns=n;n=v.row_stride;v.row_stride=v.column_stride;v.column_stride=n;return v;
}
/* design/algorithm-sources.md#streaming-algebra */
struct mesh_view mesh_view_broadcast(struct mesh_view v,size_t rows,size_t columns) {
  if((v.rows!=rows && v.rows!=1) || (v.columns!=columns && v.columns!=1))return (struct mesh_view){0};
  if(v.rows!=rows)v.row_stride=0;
  if(v.columns!=columns)v.column_stride=0;
  v.rows=rows;v.columns=columns;return v;
}
/* design/algorithm-sources.md#compiled-row-access-domains */
size_t mesh_tensor_publication_bytes(struct mesh_tensor *t,uint32_t i) {
  return t && i<t->count?(size_t)t->extents[i].quantum*t->context->M->pgsz:0;
}
/* design/algorithm-sources.md#streaming-algebra */
void *mesh_tensor_data(struct mesh_tensor *t,uint32_t i) { return t && i<t->count?t->extents[i].address:NULL; }
/* design/algorithm-sources.md#streaming-algebra */
static struct mesh_row_range mesh_tensor_rows(struct mesh_tensor *t,uint32_t i) {
  if(!t || i>=t->count)return (struct mesh_row_range){0};
  return (struct mesh_row_range){.first=t->extents[i].first,.count=t->extents[i].pages};
}
/* design/algorithm-sources.md#streaming-algebra */
int mesh_tensor_constant(struct mesh_tensor *t,uint32_t i) {
  if(!t || i>=t->count)return EINVAL;
  struct mesh_row_range m=mesh_tensor_rows(t,i);mesh_constant(t->context,m.first,m.count);return 0;
}
/* design/algorithm-sources.md#view-scoped-host-production */
int mesh_writer_writable(struct mesh_writer *w) {
  return mesh_writable(w->context,w->output.first,w->output.count);
}
/* design/algorithm-sources.md#view-scoped-host-production */
int mesh_writer_issue(struct mesh_writer *w) {
  uint32_t index=0;return mesh_issue(w->context,&w->function,&index,1)!=0;
}
/* design/algorithm-sources.md#view-scoped-host-production */
void mesh_writer_complete(struct mesh_writer *w,int publish) {
  if(publish){uint32_t index=0;mesh_complete(w->context,&w->function,&index,1);}
  else mesh_bits_clear(w->context->M,MESH_PRODUCING,w->output.first,w->output.count);
}
/* design/algorithm-sources.md#streaming-algebra */
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
  return a.lookup[[NSValue valueWithPointer:&view.tensor->extents[view.extent]]].buffer;
}

/* design/algorithm-sources.md#streaming-algebra */
static MPSMatrix *matrix(MeshExtent *e,struct mesh_view v,BOOL transpose) {
  size_t bytes=scalar_bytes(e.extent.shape.scalar);
  MPSMatrixDescriptor *d=[MPSMatrixDescriptor matrixDescriptorWithRows:transpose?v.columns:v.rows columns:transpose?v.rows:v.columns rowBytes:(transpose?v.column_stride:v.row_stride)*bytes dataType:bytes==2?MPSDataTypeFloat16:MPSDataTypeFloat32];
  return [[MPSMatrix alloc]initWithBuffer:e.buffer offset:v.offset*bytes descriptor:d];
}

/* design/algorithm-sources.md#mandatory-partial-publication */
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
/* design/algorithm-sources.md#mandatory-partial-publication */
static int compare_maps(const void *a,const void *b) {
  uint32_t x=((const struct mesh_row_map *)a)->first,y=((const struct mesh_row_map *)b)->first;
  return (x>y)-(x<y);
}
/* design/algorithm-sources.md#indexed-library-functions */
static void bind_dependencies(MeshFunction *f) {
  struct mesh_row_map *maps=f.dependencies.mutableBytes;size_t length=f.dependencies.length/sizeof *maps,used=0;
  qsort(maps,length,sizeof *maps,compare_maps);
  for(size_t i=0;i<length;i++) {
    if(used && maps[i].first<=maps[used-1].first+maps[used-1].count)maps[used-1].count=MAX(maps[used-1].first+maps[used-1].count,maps[i].first+maps[i].count)-maps[used-1].first;
    else maps[used++]=maps[i];
  }
  f->function.input=maps;f->function.inputs=(uint32_t)used;
}
/* design/algorithm-sources.md#indexed-library-functions */
static int output_used(MeshAlgebra *a,struct mesh_row_map m) {
  for(MeshFunction *f in a.functions)for(uint32_t i=0;i<f->function.outputs;i++) {
    struct mesh_row_map out=f->function.output[i];
    if(m.first<out.first+out.count && out.first<m.first+m.count)return 1;
  }
  return 0;
}
/* design/algorithm-sources.md#region-streaming-review */
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
/* design/algorithm-sources.md#view-scoped-host-production */
int mesh_algebra_writer(struct mesh_algebra *handle,struct mesh_view view,struct mesh_writer **result) {
  if(!result)return EINVAL;
  *result=NULL;
  MeshAlgebra *a=owner(handle);
  struct mesh_row_map output;
  int error=output_region(a,view,&output);if(error)return error;
  struct mesh_writer *w=calloc(1,sizeof *w);if(!w)return ENOMEM;
  w->context=a->context;w->output=output;
  w->function=(struct mesh_row_function){.output=&w->output,.outputs=1,.rows=1};
  w->next=a->writers;a->writers=w;*result=w;
  return 0;
}
/* design/algorithm-sources.md#region-streaming-review */
static int overlaps(struct mesh_row_map a,struct mesh_row_map b) {
  return a.first<b.first+b.count && b.first<a.first+a.count;
}
/* design/algorithm-sources.md#indexed-library-functions */
static int bind_function(struct mesh_algebra *handle,const struct mesh_view *inputs,size_t input_count,const struct mesh_view *outputs,size_t output_count,void (*submit)(MeshFunction *)) {
  MeshAlgebra *a=owner(handle);
  if(a.realized)return EBUSY;
  if(!submit || !output_count || output_count>UINT32_MAX || input_count>UINT32_MAX || !outputs || (input_count && !inputs))return EINVAL;
  MeshFunction *f=[MeshFunction new];f.owner=a;f.queue=a.queue;f.dependencies=[NSMutableData new];f.results=[NSMutableData new];
  f.inputViews=[NSData dataWithBytes:inputs length:input_count*sizeof *inputs];f.indexedInputs=[NSMutableIndexSet new];
  for(size_t i=0;i<input_count;i++){if(!valid_view(a,inputs[i]))return EINVAL;dependencies(f.dependencies,inputs[i]);}
  for(size_t i=0;i<output_count;i++) {
    struct mesh_row_map m;int error=output_region(a,outputs[i],&m);if(error)return error;
    if(output_used(a,m))return EINVAL;
    struct mesh_row_map *results=f.results.mutableBytes,*reads=f.dependencies.mutableBytes;
    for(size_t j=0;j<i;j++)if(overlaps(results[j],m))return EINVAL;
    for(size_t j=0;j<f.dependencies.length/sizeof *reads;j++)if(overlaps(reads[j],m))return EINVAL;
    [f.results appendBytes:&m length:sizeof m];
  }
  f->function=(struct mesh_row_function){.output=f.results.mutableBytes,.outputs=(uint32_t)output_count,.rows=1};bind_dependencies(f);
  f.execute=^(MeshFunction *function){submit(function);};
  [a.functions addObject:f];return 0;
}

/* design/algorithm-sources.md#dynamic-reader-lifetimes */
static struct mesh_index_candidate indexed_maps(struct mesh_view view){
  MeshFunction *f=[MeshFunction new];f.dependencies=[NSMutableData new];dependencies(f.dependencies,view);bind_dependencies(f);
  size_t bytes=f->function.inputs*sizeof(struct mesh_row_map);
  struct mesh_row_map *maps=malloc(bytes);if(maps)memcpy(maps,f->function.input,bytes);
  return (struct mesh_index_candidate){.maps=maps,.count=f->function.inputs};
}
/* design/algorithm-sources.md#selected-native-contractions */
int mesh_algebra_view_pages(struct mesh_algebra *handle,struct mesh_view view,struct mesh_view *pages,size_t capacity,size_t *count) {
  MeshAlgebra *a=owner(handle);
  if(!count || !valid_view(a,view) || (!pages && capacity))return EINVAL;
  struct mesh_index_candidate coverage=indexed_maps(view);
  if(!coverage.maps)return ENOMEM;
  size_t needed=0;
  for(uint32_t i=0;i<coverage.count;i++)needed+=coverage.maps[i].count;
  *count=needed;
  if(!pages){free(coverage.maps);return 0;}
  if(capacity<needed){free(coverage.maps);return ENOSPC;}
  struct mesh_extent *extent=&view.tensor->extents[view.extent];
  size_t unit=a->context->M->pgsz/scalar_bytes(extent->shape.scalar),elements=extent->shape.rows*extent->shape.columns,position=0;
  for(uint32_t i=0;i<coverage.count;i++)for(uint32_t j=0;j<coverage.maps[i].count;j++){
    size_t offset=((size_t)coverage.maps[i].first+j-extent->first)*unit,columns=MIN(unit,elements-offset);
    pages[position++]=(struct mesh_view){.tensor=view.tensor,.extent=view.extent,.offset=offset,.rows=1,.columns=columns,.row_stride=columns,.column_stride=1};
  }
  free(coverage.maps);return 0;
}
/* design/algorithm-sources.md#dynamic-reader-lifetimes */
int mesh_algebra_indexed(struct mesh_algebra *handle,size_t index,struct mesh_view selector,const size_t *candidate_inputs,size_t input_count,const struct mesh_view *candidates,size_t count){
  MeshAlgebra *a=owner(handle);
  if(a.realized || index>=a.functions.count || !count || count>(UINT32_MAX-2)/2 || (input_count && !candidate_inputs) || !candidates || !valid_view(a,selector))return EINVAL;
  if(selector.tensor->extents[selector.extent].shape.scalar!=MESH_U32)return EINVAL;
  MeshFunction *f=a.functions[index];const struct mesh_view *inputs=f.inputViews.bytes;size_t available_inputs=f.inputViews.length/sizeof *inputs;
  for(size_t i=0;i<input_count;i++)if(candidate_inputs[i]>=available_inputs || !valid_view(a,inputs[candidate_inputs[i]]))return EINVAL;
  for(size_t i=0;i<count;i++)if(!valid_view(a,candidates[i]))return EINVAL;
  struct mesh_indexed_read *d=calloc(1,sizeof *d);if(!d)return ENOMEM;
  d->candidate=calloc(count,sizeof *d->candidate);if(!d->candidate){free(d);return ENOMEM;}
  d->candidates=(uint32_t)count;
  struct mesh_index_candidate selection=indexed_maps(selector);d->selector=selection.maps;d->selectors=selection.count;
  int error=d->selector?0:ENOMEM;
  for(size_t i=0;i<count && !error;i++){
    struct mesh_view candidate=candidates[i];
    d->candidate[i]=indexed_maps(candidate);
    if(!d->candidate[i].maps){error=ENOMEM;break;}
  }
  for(size_t i=0;i<count && !error;i++)for(uint32_t j=0;j<d->candidate[i].count;j++){
    struct mesh_row_map map=d->candidate[i].maps[j];
    for(uint32_t k=0;k<d->selectors;k++)if(overlaps(map,d->selector[k]))error=EINVAL;
    for(size_t k=0;k<i;k++)for(uint32_t q=0;q<d->candidate[k].count;q++)if(overlaps(map,d->candidate[k].maps[q]))error=EINVAL;
  }
  if(error){for(size_t i=0;i<count;i++)free(d->candidate[i].maps);free(d->candidate);free(d->selector);free(d);return error;}
  d->retired=mesh_rows_alloc(a->context,2*(uint32_t)count+2);
  if(d->retired==MESH_ABSENT){for(size_t i=0;i<count;i++)free(d->candidate[i].maps);free(d->candidate);free(d->selector);free(d);return errno;}
  d->indices=(const uint32_t *)selector.tensor->extents[selector.extent].address+selector.offset;
  d->rows=selector.rows;d->columns=selector.columns;d->row_stride=selector.row_stride;d->column_stride=selector.column_stride;
  d->selected=d->retired+d->candidates;d->completed=d->selected+d->candidates;d->mapped=d->completed+1;
  for(size_t i=0;i<input_count;i++)[f.indexedInputs addIndex:candidate_inputs[i]];
  d->next=f->function.indexed;f->function.indexed=d;f.dependencies=[NSMutableData new];
  for(size_t i=0;i<available_inputs;i++)if(![f.indexedInputs containsIndex:i])dependencies(f.dependencies,inputs[i]);
  for(struct mesh_indexed_read *part=f->function.indexed;part;part=part->next)
    [f.dependencies appendBytes:part->selector length:part->selectors*sizeof *part->selector];
  bind_dependencies(f);
  return 0;
}

/* design/algorithm-sources.md#region-expression-fusion */
static MeshCode *source_code(MeshAlgebra *a,const char *text,BOOL cpu) {
  NSMutableDictionary *cache=cpu?(NSMutableDictionary *)a.cpuCode:(NSMutableDictionary *)a.libraries;NSString *source=@(text);MeshCode *code=cache[source];
  if(!code){
    code=cpu?[MeshCPUCode new]:[MeshMetalCode new];code.source=source;
    cache[source]=code;
  }
  return code;
}
/* design/algorithm-sources.md#region-expression-fusion */
static MTLCompileOptions *source_options(void) {
  MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;return options;
}
/* design/algorithm-sources.md#realized-numerical-invocation */
static void submit_encoded_metal(MeshFunction *f,void (^encode)(id<MTLCommandBuffer>)) {
  id<MTLCommandBuffer> command=[f.queue commandBuffer];encode(command);
  [command addCompletedHandler:^(id<MTLCommandBuffer> done){
    complete_part(f,done.error.code);
  }];
  [command commit];
}
/* design/algorithm-sources.md#realized-numerical-invocation */
static void submit_metal(MeshFunction *f) {submit_encoded_metal(f,f.encode);}
/* design/algorithm-sources.md#programkernel_call */
int mesh_algebra_encode(struct mesh_algebra *handle,const struct mesh_view *inputs,size_t input_count,const struct mesh_view *outputs,size_t output_count,void (^encode)(id<MTLCommandBuffer>)) {
  MeshAlgebra *a=owner(handle);
  if(a.cpu || !encode)return EINVAL;
  int error=bind_function(handle,inputs,input_count,outputs,output_count,submit_metal);
  if(error)return error;
  MeshFunction *f=a.functions.lastObject;f.encode=encode;f->executionKind=MESH_EXECUTION_METAL;
  return 0;
}
/* design/algorithm-sources.md#recorded-metal-commands */
static int bind_metal(struct mesh_algebra *handle,const char *text,size_t rows,const uint64_t domain[3],const struct mesh_view *inputs,size_t input_count,struct mesh_view output,const struct mesh_view *reads,struct mesh_view write) {
  MeshAlgebra *a=owner(handle);
  NSError *error=nil;MTLCompileOptions *options=source_options();
  MeshMetalCode *code=(MeshMetalCode *)source_code(a,text,NO);id<MTLLibrary> library=code.library;
  if(!library){library=[a.device newLibraryWithSource:code.source options:options error:&error];code.library=library;}
  if(!library){fprintf(stderr,"mesh Metal kernel: %s\n",error.description.UTF8String);return EINVAL;}
  id<MTLFunction> function=[library newFunctionWithName:@"mesh_expression"];
  if(!function)return ENOENT;
  MTLComputePipelineDescriptor *pipelineDescription=[MTLComputePipelineDescriptor new];
  pipelineDescription.computeFunction=function;pipelineDescription.supportIndirectCommandBuffers=YES;
  id<MTLComputePipelineState> pipeline=[a.device newComputePipelineStateWithDescriptor:pipelineDescription options:MTLPipelineOptionNone reflection:nil error:&error];
  if(!pipeline){fprintf(stderr,"mesh Metal pipeline: %s\n",error.description.UTF8String);return EINVAL;}
  if(pipeline.maxTotalThreadsPerThreadgroup<32)return EINVAL;
  NSMutableArray<id<MTLBuffer>> *resources=[NSMutableArray new];
  id<MTLBuffer> addresses=[a.device newBufferWithLength:(input_count+1)*sizeof(uint64_t) options:MTLResourceStorageModeShared];
  if(!addresses)return ENOMEM;
  for(size_t i=0;i<input_count+1;i++) {
    struct mesh_view v=i<input_count?inputs[i]:output;
    struct mesh_extent *extent=&v.tensor->extents[v.extent];
    id<MTLBuffer> buffer=a.lookup[[NSValue valueWithPointer:extent]].buffer;
    ((uint64_t *)addresses.contents)[i]=buffer.gpuAddress+v.offset*scalar_bytes(extent->shape.scalar);
    [resources addObject:buffer];
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

/* design/algorithm-sources.md#region-expression-fusion */
static void submit_cpu(MeshFunction *f) {
  f.cpuCode.kernel(f.cpuArguments.bytes,&f->publication);complete_part(f,0);
}
/* design/algorithm-sources.md#in-operation-publication */
static void publish_cpu(void *context,uint32_t first,uint32_t count) {
  mesh_publish_partial(context,first,count);
}
/* design/algorithm-sources.md#in-operation-publication */
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
/* design/algorithm-sources.md#region-expression-fusion */
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
  NSString *cpu_text=[@MESH_KERNEL_SOURCE stringByAppendingString:@(cpu_source)];
  MeshCPUCode *library=(MeshCPUCode *)source_code(a,cpu_text.UTF8String,YES);
  NSString *source=library.source;
  if(!library.handle) {
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
  }
  NSMutableData *addresses=[NSMutableData dataWithLength:(input_count+1)*sizeof(uintptr_t)];
  uintptr_t *pointers=addresses.mutableBytes;
  for(size_t i=0;i<input_count+1;i++) {
    struct mesh_view v=i<input_count?inputs[i]:output;
    if(!valid_view(a,v))return EINVAL;
    struct mesh_extent *extent=&v.tensor->extents[v.extent];
    pointers[i]=(uintptr_t)extent->address+v.offset*scalar_bytes(extent->shape.scalar);
  }
  int status=bind_function(handle,reads,input_count,&region,1,submit_cpu);
  if(status)return status;
  MeshFunction *f=a.functions.lastObject;f->executionKind=MESH_EXECUTION_CPU;
  bind_publication(f,region);
  struct mesh_kernel_section *sections=f.publicationSections.mutableBytes;
  for(size_t i=0;i<f->publication.count;i++){sections[i].row_begin+=row_begin;sections[i].row_end+=row_begin;sections[i].column_begin=column_begin;sections[i].column_end=column_begin+column_count;}
  f.cpuCode=library;f.cpuArguments=addresses;
  return 0;
}

/* design/algorithm-sources.md#cpu-library-contraction */
static BNNSNDArrayDescriptor bnns_operand(struct mesh_view v) {
  struct mesh_extent *e=&v.tensor->extents[v.extent];
  BOOL transpose=v.column_stride!=1;
  return (BNNSNDArrayDescriptor){.layout=BNNSDataLayoutRowMajorMatrix,
    .size={transpose?v.rows:v.columns,transpose?v.columns:v.rows},
    .stride={1,transpose?v.column_stride:v.row_stride},
    .data=(char *)e->address+v.offset*scalar_bytes(e->shape.scalar),
    .data_type=e->shape.scalar==MESH_F16?BNNSDataTypeFloat16:BNNSDataTypeFloat32};
}
/* design/algorithm-sources.md#cpu-indexed-execution */
static int cpu_part(MeshFunction *f,struct mesh_view x,struct mesh_view y,struct mesh_view z,float alpha,size_t first,size_t count) {

  if(x.tensor->extents[x.extent].shape.scalar==MESH_F32 && y.tensor->extents[y.extent].shape.scalar==MESH_F32 && z.tensor->extents[z.extent].shape.scalar==MESH_F32){

    struct gemm {const float *a,*b;float *c;__LAPACK_int m,n,k,lda,ldb,ldc;enum CBLAS_TRANSPOSE tx,ty;};
    NSMutableData *calls=[NSMutableData new];
    for(size_t at=first,left=count;left;){
      size_t row=at/z.columns,column=at%z.columns,nr=1,nc=MIN(left,z.columns-column);
      if(!column && left>=z.columns){nr=left/z.columns;nc=z.columns;}
      BOOL tx=x.column_stride!=1,ty=y.column_stride!=1;
      struct gemm g={.a=(float *)x.tensor->extents[x.extent].address+x.offset+row*x.row_stride,
        .b=(float *)y.tensor->extents[y.extent].address+y.offset+column*y.column_stride,
        .c=(float *)z.tensor->extents[z.extent].address+z.offset+at,
        .m=(__LAPACK_int)nr,.n=(__LAPACK_int)nc,.k=(__LAPACK_int)x.columns,.lda=(__LAPACK_int)(tx?x.column_stride:x.row_stride),
        .ldb=(__LAPACK_int)(ty?y.column_stride:y.row_stride),.ldc=(__LAPACK_int)z.row_stride,.tx=tx?CblasTrans:CblasNoTrans,.ty=ty?CblasTrans:CblasNoTrans};
      [calls appendBytes:&g length:sizeof g];at+=nr*nc;left-=nr*nc;
    }
    f.execute=^(MeshFunction *function){
      const struct gemm *g=calls.bytes;
      for(size_t i=0;i<calls.length/sizeof *g;i++)cblas_sgemm(CblasRowMajor,g[i].tx,g[i].ty,g[i].m,g[i].n,g[i].k,alpha,g[i].a,g[i].lda,g[i].b,g[i].ldb,0,g[i].c,g[i].ldc);
      complete_part(function,0);
    };
    return 0;
  }

  struct gemm {BNNSNDArrayDescriptor a,b,c;};
  NSMutableData *calls=[NSMutableData new];size_t workspaceSize=1;
  BOOL tx=x.column_stride!=1,ty=y.column_stride!=1;
  for(size_t at=first,left=count;left;){
    size_t row=at/z.columns,column=at%z.columns,nr=1,nc=MIN(left,z.columns-column);
    if(!column && left>=z.columns){nr=left/z.columns;nc=z.columns;}
    struct gemm g={bnns_operand(mesh_view_slice(x,row,0,nr,x.columns)),
      bnns_operand(mesh_view_slice(y,0,column,y.rows,nc)),
      bnns_operand(mesh_view_slice(z,row,column,nr,nc))};
    ssize_t bytes=BNNSMatMulWorkspaceSize(tx,ty,alpha,&g.a,&g.b,&g.c,NULL);
    if(bytes<0)return EINVAL;
    workspaceSize=MAX(workspaceSize,(size_t)bytes);
    [calls appendBytes:&g length:sizeof g];at+=nr*nc;left-=nr*nc;
  }
  NSMutableData *workspace=[NSMutableData dataWithLength:workspaceSize];
  if(!workspace)return ENOMEM;
  f.cpuArguments=workspace;
  void *scratch=workspace.mutableBytes;
  f.execute=^(MeshFunction *function){
    const struct gemm *g=calls.bytes;int error=0;
    for(size_t i=0;i<calls.length/sizeof *g && !error;i++)
      error=BNNSMatMul(tx,ty,alpha,&g[i].a,&g[i].b,&g[i].c,scratch,NULL);
    complete_part(function,error);
  };
  return 0;
}
/* design/algorithm-sources.md#coreml-partial-execution */
static MLMultiArray *native_array(struct mesh_view v,NSError **error) {
  struct mesh_extent *e=&v.tensor->extents[v.extent];size_t bytes=scalar_bytes(e->shape.scalar);
  return [[MLMultiArray alloc]initWithDataPointer:(char *)e->address+v.offset*bytes shape:@[@(v.rows),@(v.columns)] dataType:bytes==2?MLMultiArrayDataTypeFloat16:MLMultiArrayDataTypeFloat32 strides:@[@(v.row_stride),@(v.column_stride)] deallocator:nil error:error];
}
/* design/algorithm-sources.md#coreml-partial-execution */
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
/* design/algorithm-sources.md#selected-native-contractions */
static int prepare_part(MeshAlgebra *a,MeshFunction *f,struct mesh_view x,struct mesh_view y,struct mesh_view z,float alpha,size_t first,size_t count) {
  f.owner=a;f.queue=a.queue;f.dependencies=[NSMutableData new];
  struct mesh_extent *out=&z.tensor->extents[z.extent];
  size_t scalar=scalar_bytes(out->shape.scalar);
  f->output=(struct mesh_row_map){.first=out->first+(uint32_t)((z.offset+first)*scalar/a->context->M->pgsz),.count=out->quantum};
  f.operands=@[a.lookup[[NSValue valueWithPointer:&x.tensor->extents[x.extent]]],a.lookup[[NSValue valueWithPointer:&y.tensor->extents[y.extent]]],a.lookup[[NSValue valueWithPointer:out]]];
  BOOL dense=z.column_stride==1;
  size_t columns=dense?z.columns:z.rows;
  NSMutableArray<MPSMatrixMultiplication *> *products=[NSMutableArray new];
  NSMutableArray<NSArray<MPSMatrix *> *> *matrices=[NSMutableArray new];
  NSMutableArray *rectangles=[NSMutableArray new];NSMutableDictionary *features=[NSMutableDictionary new];
  for(size_t at=first,left=count;left;) {
    size_t row=at/columns,column=at%columns,nr=1,nc=MIN(left,columns-column);
    if(!column && left>=columns){nr=left/columns;nc=columns;}
    size_t zr=dense?row:column,zc=dense?column:row,rr=dense?nr:nc,cc=dense?nc:nr;
    struct mesh_view xv=mesh_view_slice(x,zr,0,rr,x.columns);
    struct mesh_view yv=mesh_view_slice(y,0,zc,y.rows,cc);
    dependencies(f.dependencies,xv);dependencies(f.dependencies,yv);
    if(!a.cpu && a.coremlPython) {
      NSError *error=nil;MLMultiArray *left=native_array(xv,&error),*right=native_array(yv,&error);
      if(!left || !right)return (int)error.code;
      NSUInteger i=rectangles.count;
      features[[NSString stringWithFormat:@"x%lu",(unsigned long)i]]=left;
      features[[NSString stringWithFormat:@"w%lu",(unsigned long)i]]=right;
      [rectangles addObject:@[@(rr),@(cc),@(x.columns),@(xv.tensor->extents[xv.extent].shape.scalar==MESH_F16),@(yv.tensor->extents[yv.extent].shape.scalar==MESH_F16)]];
    } else if(!a.cpu) {
      BOOL tx=xv.column_stride!=1,ty=yv.column_stride!=1;
      struct mesh_view zv=mesh_view_slice(z,zr,zc,rr,cc);
      [matrices addObject:@[matrix(f.operands[0],xv,tx),matrix(f.operands[1],yv,ty),matrix(f.operands[2],zv,NO)]];
      [products addObject:[[MPSMatrixMultiplication alloc]initWithDevice:a.device transposeLeft:tx transposeRight:ty resultRows:rr resultColumns:cc interiorColumns:x.columns alpha:alpha beta:0]];
    }
    at+=nr*nc;left-=nr*nc;
  }
  f->function=(struct mesh_row_function){.output=&f->output,.outputs=1,.rows=1};
  bind_dependencies(f);
  if(a.cpu) {
    f->executionKind=MESH_EXECUTION_CPU;
    int error=cpu_part(f,x,y,z,alpha,first,count);if(error)return error;
  } else if(a.coremlPython) {
    f->executionKind=MESH_EXECUTION_COREML;
    int error=native_part(a,f,rectangles,features,z,first,count,alpha);if(error)return error;
  } else {

    f.encode=^(id<MTLCommandBuffer> command){for(NSUInteger i=0;i<products.count;i++)[products[i] encodeToCommandBuffer:command leftMatrix:matrices[i][0] rightMatrix:matrices[i][1] resultMatrix:matrices[i][2]];};
  }
  if(!f.execute) {
    f->executionKind=MESH_EXECUTION_METAL;
    void (^encode)(id<MTLCommandBuffer>)=f.encode;
    f.execute=^(MeshFunction *function){submit_encoded_metal(function,encode);};
  }
  return 0;
}
/* design/algorithm-sources.md#selected-native-contractions */
static int bind_part(MeshAlgebra *a,struct mesh_view x,struct mesh_view y,struct mesh_view z,float alpha,size_t first,size_t count) {
  MeshFunction *f=[MeshFunction new];
  int error=prepare_part(a,f,x,y,z,alpha,first,count);if(error)return error;
  [a.functions addObject:f];return 0;
}
/* design/algorithm-sources.md#selected-native-contractions */
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

/* design/algorithm-sources.md#literal-contiguous-materialization */
struct mesh_copy_segment { const char *source; char *destination; size_t elements,stride,bytes; };

/* design/algorithm-sources.md#literal-contiguous-materialization */
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
        struct mesh_copy_segment segment={
          .source=(const char *)source.tensor->extents[source.extent].address+source.offset*scalar,
          .destination=(char *)d->address+(destination.offset+lo)*scalar,.elements=hi-lo,.stride=source.column_stride*scalar,.bytes=scalar};
        if(segment.stride==scalar){segment.bytes*=segment.elements;segment.elements=1;}
        struct mesh_copy_segment *previous=segments.length?(struct mesh_copy_segment *)segments.mutableBytes+segments.length/sizeof segment-1:NULL;
        if(previous && previous->elements==1 && segment.elements==1 && previous->source+previous->bytes==segment.source && previous->destination+previous->bytes==segment.destination)previous->bytes+=segment.bytes;
        else [segments appendBytes:&segment length:sizeof segment];
      }
    }
    f->output=(struct mesh_row_map){.first=output.first+(uint32_t)(first*scalar/a->context->M->pgsz),.count=d->quantum};
    f->function=(struct mesh_row_function){.output=&f->output,.outputs=1,.rows=1};bind_dependencies(f);

    f.execute=^(MeshFunction *function){
      const struct mesh_copy_segment *parts=segments.bytes;
      for(size_t i=0;i<segments.length/sizeof *parts;i++) {
        struct mesh_copy_segment p=parts[i];
        for(size_t j=0;j<p.elements;j++)memcpy(p.destination+j*p.bytes,p.source+j*p.stride,p.bytes);
      }
      complete_part(function,0);
    };
    [a.functions addObject:f];
  }
  return 0;
}
/* design/algorithm-sources.md#pallas-indexed-destinations */
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

/* design/algorithm-sources.md#view-scoped-consumption */
int mesh_algebra_export(struct mesh_algebra *handle,struct mesh_view view,size_t *first,size_t *count) {
  MeshAlgebra *a=owner(handle);
  if(a.realized)return EBUSY;
  if(!first || !count || !valid_view(a,view))return EINVAL;
  struct mesh_index_candidate coverage=indexed_maps(view);if(!coverage.maps)return ENOMEM;
  *first=a.returns.length/sizeof(struct mesh_row_map);*count=coverage.count;
  [a.returns appendBytes:coverage.maps length:coverage.count*sizeof *coverage.maps];free(coverage.maps);return 0;
}
/* design/algorithm-sources.md#presence-driven-execution */
static void submit_ready(void *argument,uint32_t occurrence) {
  MeshFunction *f=(__bridge MeshFunction *)argument;MeshAlgebra *a=f.owner;
  f->occurrence=occurrence;
  atomic_fetch_add(&a->submitted,1);
  dispatch_group_enter(a.executions);
  @autoreleasepool{f.execute(f);}
}
/* design/algorithm-sources.md#streaming-algebra */
int mesh_algebra_realize(struct mesh_algebra *handle) {
  MeshAlgebra *a=owner(handle);if(a.realized)return 0;size_t count=a.functions.count;
  for(MeshFunction *f in a.functions)for(struct mesh_indexed_read *d=f->function.indexed;d;d=d->next)
    for(uint32_t i=0;i<d->selectors;i++){
      struct mesh_row_map map=d->selector[i];
      for(uint32_t r=map.first;r<map.first+map.count;r++)if(!output_used(a,(struct mesh_row_map){.first=r,.count=1}))return EINVAL;
      const struct mesh_row_binding *bindings=a.bindings.bytes;
      for(size_t j=0;j<a.bindings.length/sizeof *bindings;j++)
        if(bindings[j].receive && overlaps(map,(struct mesh_row_map){.first=bindings[j].first,.count=bindings[j].count}))return EINVAL;
    }
  struct mesh_row_function *functions=calloc(count?count:1,sizeof *functions);
  if(!functions)return ENOMEM;
  for(size_t i=0;i<count;i++)functions[i]=a.functions[i]->function;
  int error=mesh_realize(a->context,functions,count,a.bindings.mutableBytes,a.bindings.length/sizeof(struct mesh_row_binding),a.returns.mutableBytes,a.returns.length/sizeof(struct mesh_row_map));
  free(functions);
  if(!error){
    size_t arenaBytes=0;
    for(MeshExtent *extent in a.extents)arenaBytes+=extent.extent.bytes;
    fprintf(stderr,"mesh realize: participant=%u planned_arena_bytes=%zu\n",a->context->M->node,arenaBytes);
    a.realized=YES;
    for(MeshFunction *f in a.functions)for(struct mesh_indexed_read *d=f->function.indexed;d && !error;d=d->next)
      error=mesh_execution_indexed(a->context,d,handle);
    if(error)return error;
    for(MeshFunction *f in a.functions){
      if(f->executionKind==MESH_EXECUTION_CPU){
        void (^execute)(MeshFunction *)=f.execute;
        f.execute=^(MeshFunction *function){
          dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^{@autoreleasepool{execute(function);}});
        };
      }
      error=mesh_execution_add(a->context,&f->function,handle,submit_ready,(__bridge void *)f);
      if(error)break;
    }
  }
  return error;
}

/* design/algorithm-sources.md#streaming-algebra */
int mesh_algebra_available(struct mesh_algebra *handle,size_t index) {
  MeshAlgebra *a=owner(handle);if(index>=a.returns.length/sizeof(struct mesh_row_map))return 0;
  return mesh_available(a->context,((struct mesh_row_map *)a.returns.bytes)[index],0);
}
/* design/algorithm-sources.md#streaming-algebra */
void mesh_algebra_consume(struct mesh_algebra *handle,size_t index) {
  MeshAlgebra *a=owner(handle);if(index<a.returns.length/sizeof(struct mesh_row_map))mesh_consume(a->context,((struct mesh_row_map *)a.returns.bytes)[index],0);
}
/* design/algorithm-sources.md#streaming-algebra */
struct mesh_algebra_report mesh_algebra_report(struct mesh_algebra *handle) {
  MeshAlgebra *a=owner(handle);return (struct mesh_algebra_report){.submitted=atomic_load(&a->submitted),.completed=atomic_load(&a->completed),.code=atomic_load(&a->code)};
}

/* design/algorithm-sources.md#indexed-library-functions */
size_t mesh_algebra_function_count(struct mesh_algebra *handle) {return owner(handle).functions.count;}
