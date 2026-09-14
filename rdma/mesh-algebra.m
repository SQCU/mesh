#import <Foundation/Foundation.h>
#define ACCELERATE_NEW_LAPACK
#define ACCELERATE_LAPACK_ILP64
#import <Accelerate/Accelerate.h>
#import <CommonCrypto/CommonDigest.h>
#import <Metal/Metal.h>
#import <CoreML/CoreML.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "mesh-algebra.h"
#include "mesh-kernel.h"
#include <limits.h>
#include <math.h>
#include <dlfcn.h>
#include <time.h>
#include <arm_neon.h>

/* design/algorithm-sources.md#streaming-algebra */
static NSString *const source = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct V { ulong offset,rows,columns,row_stride,column_stride; };\n"
"struct G { V a,b,o; float alpha,beta; ulong first,count; };\n"
"constant uint op [[function_constant(0)]];\n"
"constant bool a16 [[function_constant(1)]],b16 [[function_constant(2)]],o16 [[function_constant(3)]];\n"
"float get(device const uchar *p, V v, ulong r, ulong c, bool h) { ulong i=v.offset+r*v.row_stride+c*v.column_stride; return h?float(((device const half*)p)[i]):((device const float*)p)[i]; }\n"
"kernel void elementwise(device const uchar *a [[buffer(0)]],device const uchar *b [[buffer(1)]],device uchar *o [[buffer(2)]],constant G &g [[buffer(3)]],uint i [[thread_position_in_grid]]) {\n"
" if(i>=g.count)return; ulong flat=g.first+i; bool dense=g.o.column_stride==1; ulong r=dense?flat/g.o.columns:flat%g.o.rows,c=dense?flat%g.o.columns:flat/g.o.rows; float x=get(a,g.a,r,c,a16),y=0;\n"
" if(op==0)y=g.alpha*x+g.beta;\n"
" if(op==1)y=g.alpha*x+g.beta*get(b,g.b,r,c,b16);\n"
" if(op==2)y=x*get(b,g.b,r,c,b16);\n"
" if(op==3)y=tanh(x); if(op==4)y=exp(x); if(op==7)y=rsqrt(x); if(op==8)y=x/(1.0f+exp(-x));\n"
" if(op==5){y=0; for(ulong k=0;k<g.a.columns;k++)y+=get(a,g.a,r,k,a16);}\n"
" ulong j=g.o.offset+r*g.o.row_stride+c*g.o.column_stride; if(o16)((device half*)o)[j]=half(y);else ((device float*)o)[j]=y;\n"
"}\n";

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
struct geometry_view { uint64_t offset,rows,columns,row_stride,column_stride; };
struct geometry { struct geometry_view a,b,o; float alpha,beta; uint64_t first,count; };

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
  struct geometry geometry;
  struct mesh_kernel_publication publication;
  uint32_t occurrence;
  enum mesh_execution_kind executionKind;
  _Atomic uint64_t readyNs,startNs,completeNs,gpuStartNs,gpuEndNs,invocations;
}
@property NSArray<MeshFunction *> *plans;
@property NSArray<MeshExtent *> *operands;
@property MeshCPUCode *cpuCode;
@property id<MTLComputePipelineState> metalPipeline;
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
  if(function.active){free(function.active->count_maps);free(function.active);}
  struct mesh_route_use *u=function.routes;while(u){struct mesh_route_use *next=u->next;free(u);u=next;}
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
  struct mesh_route *routes;
  struct mesh_writer *writers;
  _Atomic uint64_t submitted;
  _Atomic uint64_t nativeSubmitted,cpuSubmitted;
  uint32_t copies;
  _Atomic uint64_t completed,gpuNanoseconds,nativeBackings;
  _Atomic int64_t code;
}
@property BOOL realized,cpu;
@property dispatch_group_t executions;
@property id<MTLDevice> device;
@property id<MTLCommandQueue> queue;
@property id<MTLResidencySet> routeResidency;
@property id<MTLLibrary> library;
@property NSMutableDictionary<NSString *,MeshMetalCode *> *libraries;
@property NSMutableDictionary<NSString *,MeshCPUCode *> *cpuCode;
@property NSMutableArray<MeshFunction *> *functions;
@property NSMutableArray<MeshExtent *> *extents;
@property NSMutableDictionary<NSValue *,MeshExtent *> *lookup;
@property NSMutableData *tensors;
@property NSMutableData *bindings;
@property NSMutableData *returns,*observations;
@property NSString *coremlPython,*coremlGenerator,*coremlCache;
@property NSMutableDictionary<NSString *,MLModel *> *models;
@end
@implementation MeshAlgebra
/* design/algorithm-sources.md#streaming-algebra */
- (void)dealloc {
  for(MeshFunction *f in self.functions){
    for(uint32_t i=0;i<f->function.inputs;i++)mesh_reader_unbind(context,&f->function.input[i]);
    if(f->function.active){struct mesh_active *active=f->function.active;for(uint32_t i=0;i<active->maps;i++)mesh_reader_unbind(context,&active->count_maps[i]);mesh_rows_release(context,active->disposition,2);if(active->retired!=MESH_ABSENT)mesh_rows_release(context,active->retired,active->inputs);}
    for(struct mesh_indexed_read *d=f->function.indexed;d;d=d->next){
      for(uint32_t i=0;i<d->selectors;i++)mesh_reader_unbind(context,&d->selector[i]);
      for(uint32_t i=0;i<d->candidates;i++)for(uint32_t j=0;j<d->candidate[i].count;j++)mesh_reader_unbind(context,&d->candidate[i].maps[j]);
      mesh_rows_release(context,d->retired,2*d->candidates+2);
    }
  }
  while(writers){struct mesh_writer *next=writers->next;free(writers);writers=next;}
  struct mesh_route *route=routes;
  while(route){
    struct mesh_route *next=route->next;
    for(uint32_t i=0;i<route->metadata_count;i++)mesh_reader_unbind(context,&route->metadata[i]);
    for(uint32_t i=0;i<route->candidates;i++){
      for(uint32_t j=0;j<route->candidate[i].count;j++)mesh_reader_unbind(context,&route->candidate[i].maps[j]);
      if(route->candidate[i].producer)mesh_reader_unbind(context,&route->candidate[i].disposition);
      free(route->candidate[i].maps);
    }
    mesh_rows_release(context,route->retired,route->candidates+route->consumers+1);
    free(route->metadata);free(route->candidate);free(route->watches);free(route->functions);free(route);route=next;
  }
  struct mesh_row_map *returns=self.returns.mutableBytes;
  for(size_t i=0;i<self.returns.length/sizeof *returns;i++)mesh_reader_unbind(context,&returns[i]);
  self.functions=nil;
  if(self.routeResidency)[self.queue removeResidencySet:self.routeResidency];
  self.routeResidency=nil;
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
static void complete_part(MeshFunction *f,int64_t error,uint64_t nanoseconds) {
  MeshAlgebra *a=f.owner;uint64_t complete=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  atomic_store(&f->completeNs,complete);
  if(error)atomic_store(&a->code,error);
  else mesh_complete(a->context,&f->function,&f->occurrence,1);
  atomic_fetch_add(&a->gpuNanoseconds,nanoseconds);atomic_fetch_add(&a->completed,1);dispatch_group_leave(a.executions);
}
/* design/algorithm-sources.md#streaming-algebra */
static MeshAlgebra *owner(struct mesh_algebra *a) { return (__bridge MeshAlgebra *)a; }
/* design/algorithm-sources.md#streaming-algebra */
static struct mesh_algebra *create_algebra(struct mesh_ctx *context,BOOL cpu) {
  if(!context || !context->M){errno=EINVAL;return NULL;}
  MeshAlgebra *a=[MeshAlgebra new]; a->context=context;a.cpu=cpu;
  if(!cpu) {
    a.device=MTLCreateSystemDefaultDevice(); a.queue=[a.device newCommandQueue];
    NSError *error=nil; a.library=[a.device newLibraryWithSource:source options:nil error:&error];
    if(!a.queue || !a.library){fprintf(stderr,"mesh algebra: %s\n",error.description.UTF8String);errno=ENODEV;return NULL;}
  }
  a.executions=dispatch_group_create();
  a.libraries=[NSMutableDictionary new];a.cpuCode=[NSMutableDictionary new];
  a.functions=[NSMutableArray new]; a.extents=[NSMutableArray new]; a.lookup=[NSMutableDictionary new];
  a.tensors=[NSMutableData new]; a.bindings=[NSMutableData new]; a.returns=[NSMutableData new]; a.observations=[NSMutableData new];
  return (__bridge_retained struct mesh_algebra *)a;
}
/* design/algorithm-sources.md#cpu-indexed-execution */
struct mesh_algebra *mesh_algebra_create(struct mesh_ctx *context) {return create_algebra(context,NO);}
/* design/algorithm-sources.md#cpu-indexed-execution */
struct mesh_algebra *mesh_algebra_create_cpu(struct mesh_ctx *context) {return create_algebra(context,YES);}
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
struct mesh_row_range mesh_tensor_rows(struct mesh_tensor *t,uint32_t i) {
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
/* design/algorithm-sources.md#streaming-algebra */
static struct geometry_view geometry(struct mesh_view v) {
  return (struct geometry_view){v.offset,v.rows,v.columns,v.row_stride,v.column_stride};
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
  MeshFunction *f=[MeshFunction new];f.owner=a;f.dependencies=[NSMutableData new];f.results=[NSMutableData new];
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

/* design/algorithm-sources.md#shared-sparse-routing-lowering */
static int route_residency(MeshAlgebra *a,const struct mesh_view *candidates,size_t count){
  if(a.cpu)return 0;
  if(!a.routeResidency){
    MTLResidencySetDescriptor *descriptor=[MTLResidencySetDescriptor new];descriptor.initialCapacity=count;
    NSError *error=nil;a.routeResidency=[a.device newResidencySetWithDescriptor:descriptor error:&error];
    if(!a.routeResidency){fprintf(stderr,"mesh route residency: %s\n",error.description.UTF8String);return ENOMEM;}
    [a.queue addResidencySet:a.routeResidency];
  }
  for(size_t i=0;i<count;i++){
    struct mesh_extent *extent=&candidates[i].tensor->extents[candidates[i].extent];
    [a.routeResidency addAllocation:a.lookup[[NSValue valueWithPointer:extent]].buffer];
  }
  return 0;
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
/* design/algorithm-sources.md#shared-sparse-routing-lowering */
static struct mesh_route_vector route_vector(struct mesh_view view){
  return (struct mesh_route_vector){.values=(const uint32_t *)view.tensor->extents[view.extent].address+view.offset,.columns=view.columns,.row_stride=view.row_stride,.column_stride=view.column_stride,.length=view.rows*view.columns};
}
/* design/algorithm-sources.md#shared-sparse-routing-lowering */
struct mesh_route *mesh_algebra_route_create(struct mesh_algebra *handle,struct mesh_view owners,struct mesh_view ordinals,struct mesh_view offsets,const struct mesh_view *candidates,size_t count,size_t consumers){
  MeshAlgebra *a=owner(handle);int error=0;
  if(a.realized || !candidates || !count || !consumers || count>=UINT32_MAX || consumers>=UINT32_MAX-count){errno=EINVAL;return NULL;}
  struct mesh_view metadata[]={owners,ordinals,offsets};
  for(size_t i=0;i<3;i++)if(!valid_view(a,metadata[i]) || metadata[i].tensor->extents[metadata[i].extent].shape.scalar!=MESH_U32){errno=EINVAL;return NULL;}
  if(owners.rows*owners.columns!=count || ordinals.rows*ordinals.columns!=count || offsets.rows*offsets.columns!=consumers+1){errno=EINVAL;return NULL;}
  for(size_t i=0;i<count;i++)if(!valid_view(a,candidates[i])){errno=EINVAL;return NULL;}
  struct mesh_route *d=calloc(1,sizeof *d);if(!d)return NULL;
  d->retired=MESH_ABSENT;d->candidates=(uint32_t)count;d->consumers=(uint32_t)consumers;d->authority=handle;
  d->candidate=calloc(count,sizeof *d->candidate);d->watches=calloc(consumers,sizeof *d->watches);d->functions=malloc(consumers*sizeof *d->functions);
  if(!d->candidate || !d->watches || !d->functions)error=ENOMEM;
  if(!error)for(size_t i=0;i<consumers;i++)d->functions[i]=SIZE_MAX;
  MeshFunction *holder=[MeshFunction new];holder.dependencies=[NSMutableData new];
  for(size_t i=0;i<3;i++)dependencies(holder.dependencies,metadata[i]);bind_dependencies(holder);
  d->metadata_count=holder->function.inputs;d->metadata=malloc(d->metadata_count*sizeof *d->metadata);
  if(!d->metadata)error=ENOMEM;else memcpy(d->metadata,holder->function.input,d->metadata_count*sizeof *d->metadata);
  for(size_t i=0;i<count && !error;i++){d->candidate[i]=indexed_maps(candidates[i]);d->candidate[i].input=i;if(!d->candidate[i].maps)error=ENOMEM;}
  for(size_t i=0;i<count && !error;i++)for(uint32_t j=0;j<d->candidate[i].count;j++){
    struct mesh_row_map map=d->candidate[i].maps[j];
    for(uint32_t k=0;k<d->metadata_count;k++)if(overlaps(map,d->metadata[k]))error=EINVAL;
    for(size_t k=0;k<i;k++)for(uint32_t q=0;q<d->candidate[k].count;q++)if(overlaps(map,d->candidate[k].maps[q]))error=EINVAL;
  }
  if(!error){d->retired=mesh_rows_alloc(a->context,(uint32_t)(count+consumers+1));if(d->retired==MESH_ABSENT)error=errno;}
  if(!error)error=route_residency(a,candidates,count);
  if(!error){
    struct mesh_shape shape={.rows=count,.columns=3,.scalar=MESH_U64};
    d->table=mesh_tensor_create(handle,&shape,1,0,1);if(!d->table)error=errno;
  }
  if(error){
    if(d->retired!=MESH_ABSENT)mesh_rows_release(a->context,d->retired,(uint32_t)(count+consumers+1));
    if(d->candidate)for(size_t i=0;i<count;i++)free(d->candidate[i].maps);
    free(d->metadata);free(d->candidate);free(d->watches);free(d->functions);free(d);errno=error;return NULL;
  }
  uint64_t *table=mesh_tensor_data(d->table,0);
  for(size_t i=0;i<count;i++){
    struct mesh_view v=candidates[i];struct mesh_extent *extent=&v.tensor->extents[v.extent];
    MeshExtent *storage=a.lookup[[NSValue valueWithPointer:extent]];
    table[3*i]=(a.cpu?(uint64_t)(uintptr_t)extent->address:storage.buffer.gpuAddress)+v.offset*scalar_bytes(extent->shape.scalar);
    table[3*i+1]=v.row_stride;table[3*i+2]=v.column_stride;
  }
  mesh_tensor_constant(d->table,0);
  d->owners=route_vector(owners);d->ordinals=route_vector(ordinals);d->offsets=route_vector(offsets);
  d->completed=d->retired+d->candidates;d->prepared=d->completed+d->consumers;
  struct mesh_route **tail=&a->routes;while(*tail)tail=&(*tail)->next;*tail=d;
  return d;
}
/* design/algorithm-sources.md#shared-sparse-routing-lowering */
struct mesh_view mesh_algebra_route_table(struct mesh_algebra *handle,struct mesh_route *d){
  if(!d || d->authority!=handle){errno=EINVAL;return (struct mesh_view){0};}
  return mesh_tensor_view(d->table,0);
}
/* design/algorithm-sources.md#active-segment-domains */
int mesh_algebra_active(struct mesh_algebra *handle,size_t index,struct mesh_view count,size_t slot){
  MeshAlgebra *a=owner(handle);
  if(a.realized || index>=a.functions.count || slot>UINT32_MAX || !valid_view(a,count) || count.rows*count.columns!=1 || count.tensor->extents[count.extent].shape.scalar!=MESH_U32)return EINVAL;
  MeshFunction *f=a.functions[index];if(f->function.active || f->function.rows!=1 || f->function.routes)return EINVAL;
  struct mesh_index_candidate maps=indexed_maps(count);if(!maps.maps)return ENOMEM;
  struct mesh_active *active=calloc(1,sizeof *active);if(!active){free(maps.maps);return ENOMEM;}
  active->disposition=mesh_rows_alloc(a->context,2);
  if(active->disposition==MESH_ABSENT){free(maps.maps);free(active);return errno;}
  active->count_maps=maps.maps;active->maps=maps.count;active->slot=(uint32_t)slot;active->omitted=active->disposition+1;active->retired=MESH_ABSENT;
  active->count=(const uint32_t *)count.tensor->extents[count.extent].address+count.offset;active->function=&f->function;
  f->function.active=active;return 0;
}
/* design/algorithm-sources.md#active-segment-domains */
int mesh_algebra_route_producers(struct mesh_algebra *handle,struct mesh_route *d,const size_t *indices,size_t count){
  MeshAlgebra *a=owner(handle);
  if(a.realized || !d || d->authority!=handle || !indices || count!=d->candidates)return EINVAL;
  for(uint32_t i=0;i<d->consumers;i++)if(d->functions[i]!=SIZE_MAX)return EBUSY;
  for(size_t i=0;i<count;i++){
    if(indices[i]>=a.functions.count)return EINVAL;MeshFunction *f=a.functions[indices[i]];
    if(!f->function.active)return EINVAL;
    for(uint32_t j=0;j<d->candidate[i].count;j++){
      struct mesh_row_map source=d->candidate[i].maps[j];int covered=0;
      for(uint32_t k=0;k<f->function.outputs;k++){struct mesh_row_map out=f->function.output[k];covered|=out.first<=source.first && out.first+out.count>=source.first+source.count;}
      if(!covered)return EINVAL;
    }
  }
  for(size_t i=0;i<count;i++){
    struct mesh_index_candidate *v=&d->candidate[i];v->producer=a.functions[indices[i]]->function.active;v->function=indices[i];
    v->disposition=(struct mesh_row_map){.first=v->producer->disposition,.count=1};
  }
  return 0;
}

/* design/algorithm-sources.md#shared-sparse-routing-lowering */
int mesh_algebra_route_hold(struct mesh_algebra *handle,struct mesh_route *d,const struct mesh_view *views,size_t count){
  MeshAlgebra *a=owner(handle);
  if(a.realized || !d || d->authority!=handle || (count && !views))return EINVAL;
  for(uint32_t i=0;i<d->consumers;i++)if(d->functions[i]!=SIZE_MAX)return EBUSY;
  MeshFunction *holder=[MeshFunction new];holder.dependencies=[NSMutableData dataWithBytes:d->metadata length:d->metadata_count*sizeof *d->metadata];
  for(size_t i=0;i<count;i++){if(!valid_view(a,views[i]))return EINVAL;dependencies(holder.dependencies,views[i]);}bind_dependencies(holder);
  for(uint32_t i=0;i<holder->function.inputs;i++)for(uint32_t j=0;j<d->candidates;j++)for(uint32_t k=0;k<d->candidate[j].count;k++)
    if(overlaps(holder->function.input[i],d->candidate[j].maps[k]))return EINVAL;
  size_t bytes=holder->function.inputs*sizeof *d->metadata;struct mesh_row_map *maps=malloc(bytes);if(!maps)return ENOMEM;
  memcpy(maps,holder->function.input,bytes);free(d->metadata);d->metadata=maps;d->metadata_count=holder->function.inputs;return 0;
}
/* design/algorithm-sources.md#shared-sparse-routing-lowering */
static void route_dependencies(MeshFunction *f){
  NSData *source=[f.dependencies copy];const struct mesh_row_map *maps=source.bytes;
  f.dependencies=[NSMutableData new];
  for(size_t i=0;i<source.length/sizeof *maps;i++){
    uint32_t first=maps[i].first,end=first+maps[i].count;
    while(first<end){
      uint32_t lo=end,hi=end;
      for(struct mesh_route_use *u=f->function.routes;u;u=u->next)for(uint32_t j=0;j<=u->domain->metadata_count;j++){
        struct mesh_row_range held=j<u->domain->metadata_count?mesh_range(u->domain->metadata[j],0):mesh_tensor_rows(u->domain->table,0);
        if(held.first<end && held.first+held.count>first && held.first<lo){lo=held.first;hi=held.first+held.count;}
      }
      if(lo>first){struct mesh_row_map kept={.first=first,.count=lo-first};[f.dependencies appendBytes:&kept length:sizeof kept];}
      first=hi;
    }
  }
  bind_dependencies(f);
}
/* design/algorithm-sources.md#shared-sparse-routing-lowering */
int mesh_algebra_route_attach(struct mesh_algebra *handle,size_t index,struct mesh_route *d,size_t consumer){
  MeshAlgebra *a=owner(handle);
  if(a.realized || !d || d->authority!=handle || index>=a.functions.count || consumer>=d->consumers || d->functions[consumer]!=SIZE_MAX)return EINVAL;
  MeshFunction *f=a.functions[index];if(f->function.rows!=1 || f->function.active)return EINVAL;
  for(uint32_t i=0;i<f->function.outputs;i++){
    for(uint32_t j=0;j<d->metadata_count;j++)if(overlaps(f->function.output[i],d->metadata[j]))return EINVAL;
    for(uint32_t j=0;j<d->candidates;j++)for(uint32_t k=0;k<d->candidate[j].count;k++)if(overlaps(f->function.output[i],d->candidate[j].maps[k]))return EINVAL;
  }
  struct mesh_route_use *u=calloc(1,sizeof *u);if(!u)return ENOMEM;
  *u=(struct mesh_route_use){.domain=d,.consumer=(uint32_t)consumer,.next=f->function.routes};f->function.routes=u;d->functions[consumer]=index;
  route_dependencies(f);
  return 0;
}

/* design/algorithm-sources.md#dynamic-reader-lifetimes */
int mesh_algebra_indexed_range(struct mesh_algebra *handle,size_t index,struct mesh_view selector,struct mesh_view range,const size_t *candidate_inputs,size_t input_count,const struct mesh_view *candidates,size_t count){
  MeshAlgebra *a=owner(handle);
  if(a.realized || index>=a.functions.count || !count || count>(UINT32_MAX-2)/2 || (input_count && !candidate_inputs) || !candidates || !valid_view(a,selector))return EINVAL;
  if(selector.tensor->extents[selector.extent].shape.scalar!=MESH_U32)return EINVAL;
  if(range.tensor && (!valid_view(a,range) || range.rows*range.columns!=2 || range.tensor->extents[range.extent].shape.scalar!=MESH_U32))return EINVAL;
  MeshFunction *f=a.functions[index];const struct mesh_view *inputs=f.inputViews.bytes;size_t available_inputs=f.inputViews.length/sizeof *inputs;
  for(size_t i=0;i<input_count;i++)if(candidate_inputs[i]>=available_inputs || !valid_view(a,inputs[candidate_inputs[i]]))return EINVAL;
  for(size_t i=0;i<count;i++)if(!valid_view(a,candidates[i]))return EINVAL;
  struct mesh_indexed_read *d=calloc(1,sizeof *d);if(!d)return ENOMEM;
  d->candidate=calloc(count,sizeof *d->candidate);if(!d->candidate){free(d);return ENOMEM;}
  d->candidates=(uint32_t)count;
  struct mesh_index_candidate selection=indexed_maps(selector);d->selector=selection.maps;d->selectors=d->vector_maps=selection.count;
  int error=d->selector?0:ENOMEM;
  if(range.tensor && !error){
    struct mesh_index_candidate bounds=indexed_maps(range);
    struct mesh_row_map *maps=bounds.maps?realloc(d->selector,(d->selectors+bounds.count)*sizeof *maps):NULL;
    if(!maps)error=ENOMEM;
    else{d->selector=maps;memcpy(maps+d->selectors,bounds.maps,bounds.count*sizeof *maps);d->selectors+=bounds.count;}
    free(bounds.maps);
  }
  for(size_t i=0;i<count && !error;i++){
    struct mesh_view candidate=candidates[i];
    d->candidate[i]=indexed_maps(candidate);d->candidate[i].input=SIZE_MAX;
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
  if(range.tensor){d->bounds=(const uint32_t *)range.tensor->extents[range.extent].address+range.offset;d->bounds_stride=range.columns==2?range.column_stride:range.row_stride;}
  d->indices=(const uint32_t *)selector.tensor->extents[selector.extent].address+selector.offset;
  d->rows=selector.rows;d->columns=selector.columns;d->row_stride=selector.row_stride;d->column_stride=selector.column_stride;
  d->selected=d->retired+d->candidates;d->completed=d->selected+d->candidates;d->mapped=d->completed+1;
  for(size_t i=0;i<input_count;i++)[f.indexedInputs addIndex:candidate_inputs[i]];
  d->next=f->function.indexed;f->function.indexed=d;f.dependencies=[NSMutableData new];
  for(size_t i=0;i<available_inputs;i++)if(![f.indexedInputs containsIndex:i])dependencies(f.dependencies,inputs[i]);
  for(struct mesh_indexed_read *part=f->function.indexed;part;part=part->next)
    [f.dependencies appendBytes:part->selector length:part->selectors*sizeof *part->selector];
  if(f->function.routes)route_dependencies(f);else bind_dependencies(f);
  return 0;
}

/* design/algorithm-sources.md#dynamic-reader-lifetimes */
int mesh_algebra_indexed(struct mesh_algebra *handle,size_t index,struct mesh_view selector,const size_t *candidate_inputs,size_t input_count,const struct mesh_view *candidates,size_t count){
  return mesh_algebra_indexed_range(handle,index,selector,(struct mesh_view){0},candidate_inputs,input_count,candidates,count);
}

struct cpu_operand {const void *address;struct geometry_view view;float (*load)(const void *,size_t);};
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
  id<MTLCommandBuffer> command=[f.owner.queue commandBuffer];encode(command);
  [command addCompletedHandler:^(id<MTLCommandBuffer> done){
    atomic_store(&f->gpuStartNs,(uint64_t)(done.GPUStartTime*1e9));atomic_store(&f->gpuEndNs,(uint64_t)(done.GPUEndTime*1e9));
    atomic_fetch_add(&f.owner->gpuNanoseconds,(uint64_t)((done.GPUEndTime-done.GPUStartTime)*1e9));
    complete_part(f,done.error.code,0);
  }];
  [command commit];
}
/* design/algorithm-sources.md#realized-numerical-invocation */
static void submit_metal(MeshFunction *f) {submit_encoded_metal(f,f.encode);}
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
  int status=bind_function(handle,reads,input_count,&write,1,submit_metal);
  if(status)return status;
  MeshFunction *f=a.functions.lastObject;f.metalPipeline=pipeline;f->executionKind=MESH_EXECUTION_METAL;
  f.encode=^(id<MTLCommandBuffer> command){
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    for(id<MTLBuffer> buffer in resources)[encoder useResource:buffer usage:MTLResourceUsageRead|MTLResourceUsageWrite];
    [encoder executeCommandsInBuffer:commands withRange:NSMakeRange(0,1)];
    [encoder endEncoding];
  };
  return 0;
}

/* design/algorithm-sources.md#region-expression-fusion */
static void submit_cpu(MeshFunction *f) {
  f.cpuCode.kernel(f.cpuArguments.bytes,&f->publication);complete_part(f,0,0);
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

/* design/algorithm-sources.md#cpu-indexed-execution */
static float cpu_f32(const void *p,size_t i) {return ((const float *)p)[i];}
/* design/algorithm-sources.md#cpu-indexed-execution */
static float cpu_f16(const void *p,size_t i) {return ((const _Float16 *)p)[i];}
/* design/algorithm-sources.md#cpu-indexed-execution */
static void cpu_store_f32(void *p,size_t i,float x) {((float *)p)[i]=x;}
/* design/algorithm-sources.md#cpu-indexed-execution */
static void cpu_store_f16(void *p,size_t i,float x) {((_Float16 *)p)[i]=(_Float16)x;}
/* design/algorithm-sources.md#cpu-indexed-execution */
static struct cpu_operand cpu_operand(struct mesh_view v) {
  struct mesh_extent *e=&v.tensor->extents[v.extent];
  return (struct cpu_operand){e->address,geometry(v),e->shape.scalar==MESH_F16?cpu_f16:cpu_f32};
}
/* design/algorithm-sources.md#cpu-indexed-execution */
static float cpu_get(struct cpu_operand p,size_t r,size_t c) {return p.load(p.address,p.view.offset+r*p.view.row_stride+c*p.view.column_stride);}
/* design/algorithm-sources.md#cpu-register-contraction */
static float32x4_t cpu_f16x4(const void *p,size_t i,size_t stride) {
  (void)stride;return vcvt_f32_f16(vld1_f16((const float16_t *)p+i));
}
/* design/algorithm-sources.md#cpu-register-contraction */
static float32x4_t cpu_f32x4(const void *p,size_t i,size_t stride) {
  (void)stride;return vld1q_f32((const float *)p+i);
}
/* design/algorithm-sources.md#cpu-register-contraction */
static float32x4_t cpu_f16x4_strided(const void *p,size_t i,size_t stride) {
  const _Float16 *v=(const _Float16 *)p+i;
  return (float32x4_t){v[0],v[stride],v[2*stride],v[3*stride]};
}
/* design/algorithm-sources.md#cpu-register-contraction */
static float32x4_t cpu_f32x4_strided(const void *p,size_t i,size_t stride) {
  const float *v=(const float *)p+i;
  return (float32x4_t){v[0],v[stride],v[2*stride],v[3*stride]};
}
struct cpu_contract {
  struct cpu_operand a,b;
  void *output;
  struct geometry_view z;
  float32x4_t (*left)(const void *,size_t,size_t),(*right)(const void *,size_t,size_t);
  void (*write)(void *,size_t,float);
  size_t row,column,rows,columns;
  float alpha;
};
/* design/algorithm-sources.md#cpu-register-contraction */
static void cpu_contract_tile(const struct cpu_contract *g) {
  for(size_t r=g->row;r<g->row+g->rows;r+=4)for(size_t c=g->column;c<g->column+g->columns;c+=4){
    size_t nr=MIN(4,g->row+g->rows-r),nc=MIN(4,g->column+g->columns-c);
    float32x4_t s0=vdupq_n_f32(0),s1=s0,s2=s0,s3=s0;
    for(size_t k=0;k<g->a.view.columns;k++){
      size_t ai=g->a.view.offset+r*g->a.view.row_stride+k*g->a.view.column_stride;
      size_t bi=g->b.view.offset+k*g->b.view.row_stride+c*g->b.view.column_stride;
      float32x4_t av,bv;
      if(nr==4)av=g->left(g->a.address,ai,g->a.view.row_stride);
      else {float v[4]={0};for(size_t j=0;j<nr;j++)v[j]=g->a.load(g->a.address,ai+j*g->a.view.row_stride);av=vld1q_f32(v);}
      if(nc==4)bv=g->right(g->b.address,bi,g->b.view.column_stride);
      else {float v[4]={0};for(size_t j=0;j<nc;j++)v[j]=g->b.load(g->b.address,bi+j*g->b.view.column_stride);bv=vld1q_f32(v);}
      s0=vfmaq_laneq_f32(s0,bv,av,0);s1=vfmaq_laneq_f32(s1,bv,av,1);
      s2=vfmaq_laneq_f32(s2,bv,av,2);s3=vfmaq_laneq_f32(s3,bv,av,3);
    }
    float32x4_t sums[4]={s0,s1,s2,s3};
    for(size_t i=0;i<nr;i++){
      float values[4];vst1q_f32(values,vmulq_n_f32(sums[i],g->alpha));
      for(size_t j=0;j<nc;j++)g->write(g->output,g->z.offset+(r+i)*g->z.row_stride+(c+j)*g->z.column_stride,values[j]);
    }
  }
}
/* design/algorithm-sources.md#cpu-indexed-execution */
static void cpu_part(MeshFunction *f,enum mesh_algebra_op op,struct mesh_view x,struct mesh_view y,struct mesh_view z,float alpha,float beta,size_t first,size_t count) {

  if(op==MESH_CONTRACT && x.tensor->extents[x.extent].shape.scalar==MESH_F32 && y.tensor->extents[y.extent].shape.scalar==MESH_F32 && z.tensor->extents[z.extent].shape.scalar==MESH_F32){

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
      complete_part(function,0,0);
    };
    return;
  }

  struct cpu_operand a=cpu_operand(x),b=cpu_operand(y);
  if(op==MESH_CONTRACT){

    struct mesh_extent *out=&z.tensor->extents[z.extent];
    struct cpu_contract g={.a=a,.b=b,.output=out->address,.z=geometry(z),.alpha=alpha,
      .left=a.load==cpu_f16?(x.row_stride==1?cpu_f16x4:cpu_f16x4_strided):(x.row_stride==1?cpu_f32x4:cpu_f32x4_strided),
      .right=b.load==cpu_f16?(y.column_stride==1?cpu_f16x4:cpu_f16x4_strided):(y.column_stride==1?cpu_f32x4:cpu_f32x4_strided),
      .write=out->shape.scalar==MESH_F16?cpu_store_f16:cpu_store_f32};
    NSMutableData *calls=[NSMutableData new];BOOL dense=z.column_stride==1;
    size_t columns=dense?z.columns:z.rows;
    for(size_t at=first,left=count;left;){
      size_t row=at/columns,column=at%columns,nr=1,nc=MIN(left,columns-column);
      if(!column && left>=columns){nr=left/columns;nc=columns;}
      g.row=dense?row:column;g.column=dense?column:row;g.rows=dense?nr:nc;g.columns=dense?nc:nr;
      [calls appendBytes:&g length:sizeof g];at+=nr*nc;left-=nr*nc;
    }
    f.execute=^(MeshFunction *function){
      const struct cpu_contract *g=calls.bytes;
      for(size_t i=0;i<calls.length/sizeof *g;i++)cpu_contract_tile(&g[i]);
      complete_part(function,0,0);
    };
    return;
  }
  float (^value)(size_t,size_t)=nil;
  switch(op) {
    case MESH_AFFINE:value=^float(size_t r,size_t c){return alpha*cpu_get(a,r,c)+beta;};break;
    case MESH_ADD:value=^float(size_t r,size_t c){return alpha*cpu_get(a,r,c)+beta*cpu_get(b,r,c);};break;
    case MESH_MULTIPLY:value=^float(size_t r,size_t c){return cpu_get(a,r,c)*cpu_get(b,r,c);};break;
    case MESH_TANH:value=^float(size_t r,size_t c){return tanhf(cpu_get(a,r,c));};break;
    case MESH_EXP:value=^float(size_t r,size_t c){return expf(cpu_get(a,r,c));};break;
    case MESH_RSQRT:value=^float(size_t r,size_t c){return 1.0f/sqrtf(cpu_get(a,r,c));};break;
    case MESH_SWISH:value=^float(size_t r,size_t c){float x=cpu_get(a,r,c);return x/(1.0f+expf(-x));};break;
    case MESH_SUM:value=^float(size_t r,size_t c){(void)c;float sum=0;for(size_t k=0;k<a.view.columns;k++)sum+=cpu_get(a,r,k);return sum;};break;
    case MESH_CONTRACT:__builtin_unreachable();
  }
  struct mesh_extent *out=&z.tensor->extents[z.extent];void *address=out->address;
  void (*write)(void *,size_t,float)=out->shape.scalar==MESH_F16?cpu_store_f16:cpu_store_f32;
  size_t rd=z.column_stride==1?z.columns:1,cd=z.column_stride==1?1:z.rows,rows=z.rows,columns=z.columns,offset=z.offset;
  f.execute=^(MeshFunction *function) {
    for(size_t i=first;i<first+count;i++)write(address,offset+i,value((i/rd)%rows,(i/cd)%columns));
    complete_part(function,0,0);
  };
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
  __unsafe_unretained MeshAlgebra *context=a;
  f.execute=^(MeshFunction *function) {
    [model predictionFromFeatures:inputs options:options completionHandler:^(id<MLFeatureProvider> prediction,NSError *failure){
      MLMultiArray *actual=[prediction featureValueForName:@"z"].multiArrayValue;
      BOOL backing=actual && actual.dataPointer==output.dataPointer && actual.dataType==output.dataType && [actual.shape isEqualToArray:output.shape] && [actual.strides isEqualToArray:output.strides];
      if(backing)atomic_fetch_add(&context->nativeBackings,1);
      complete_part(function,failure?failure.code:backing?0:EPROTO,0);
    }];
  };
  return 0;
}
/* design/algorithm-sources.md#selected-native-contractions */
static int prepare_part(MeshAlgebra *a,MeshFunction *f,enum mesh_algebra_op op,struct mesh_view x,struct mesh_view y,struct mesh_view z,float alpha,float beta,size_t first,size_t count) {
  f.owner=a;f.dependencies=[NSMutableData new];
  struct mesh_extent *out=&z.tensor->extents[z.extent];
  size_t scalar=scalar_bytes(out->shape.scalar);
  f->output=(struct mesh_row_map){.first=out->first+(uint32_t)((z.offset+first)*scalar/a->context->M->pgsz),.count=out->quantum};
  f->geometry=(struct geometry){geometry(x),geometry(y),geometry(z),alpha,beta,first,count};
  f.operands=@[a.lookup[[NSValue valueWithPointer:&x.tensor->extents[x.extent]]],a.lookup[[NSValue valueWithPointer:&y.tensor->extents[y.extent]]],a.lookup[[NSValue valueWithPointer:out]]];
  BOOL dense=z.column_stride==1,binary=op==MESH_ADD || op==MESH_MULTIPLY || op==MESH_CONTRACT;
  size_t columns=dense?z.columns:z.rows;
  NSMutableArray<MPSMatrixMultiplication *> *products=[NSMutableArray new];
  NSMutableArray<NSArray<MPSMatrix *> *> *matrices=[NSMutableArray new];
  NSMutableArray *rectangles=[NSMutableArray new];NSMutableDictionary *features=[NSMutableDictionary new];
  for(size_t at=first,left=count;left;) {
    size_t row=at/columns,column=at%columns,nr=1,nc=MIN(left,columns-column);
    if(!column && left>=columns){nr=left/columns;nc=columns;}
    size_t zr=dense?row:column,zc=dense?column:row,rr=dense?nr:nc,cc=dense?nc:nr;
    struct mesh_view xv=mesh_view_slice(x,zr,op==MESH_CONTRACT || op==MESH_SUM?0:zc,rr,op==MESH_CONTRACT || op==MESH_SUM?x.columns:cc);
    struct mesh_view yv=op==MESH_CONTRACT?mesh_view_slice(y,0,zc,y.rows,cc):mesh_view_slice(y,zr,zc,rr,cc);
    dependencies(f.dependencies,xv);if(binary)dependencies(f.dependencies,yv);
    if(op==MESH_CONTRACT && !a.cpu && a.coremlPython) {
      NSError *error=nil;MLMultiArray *left=native_array(xv,&error),*right=native_array(yv,&error);
      if(!left || !right)return (int)error.code;
      NSUInteger i=rectangles.count;
      features[[NSString stringWithFormat:@"x%lu",(unsigned long)i]]=left;
      features[[NSString stringWithFormat:@"w%lu",(unsigned long)i]]=right;
      [rectangles addObject:@[@(rr),@(cc),@(x.columns),@(xv.tensor->extents[xv.extent].shape.scalar==MESH_F16),@(yv.tensor->extents[yv.extent].shape.scalar==MESH_F16)]];
    } else if(op==MESH_CONTRACT && !a.cpu) {
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
    cpu_part(f,op,x,y,z,alpha,beta,first,count);
  } else if(op==MESH_CONTRACT && a.coremlPython) {
    f->executionKind=MESH_EXECUTION_COREML;
    int error=native_part(a,f,rectangles,features,z,first,count,alpha);if(error)return error;
  } else if(op==MESH_CONTRACT) {

    f.encode=^(id<MTLCommandBuffer> command){for(NSUInteger i=0;i<products.count;i++)[products[i] encodeToCommandBuffer:command leftMatrix:matrices[i][0] rightMatrix:matrices[i][1] resultMatrix:matrices[i][2]];};
  } else {

    MTLFunctionConstantValues *values=[MTLFunctionConstantValues new];uint32_t operation=(uint32_t)op;
    [values setConstantValue:&operation type:MTLDataTypeUInt atIndex:0];
    for(NSUInteger i=0;i<3;i++){BOOL half=f.operands[i].extent.shape.scalar==MESH_F16;[values setConstantValue:&half type:MTLDataTypeBool atIndex:i+1];}
    NSError *error=nil;id<MTLFunction> kernel=[a.library newFunctionWithName:@"elementwise" constantValues:values error:&error];
    id<MTLComputePipelineState> pipeline=[a.device newComputePipelineStateWithFunction:kernel error:&error];
    if(!pipeline){fprintf(stderr,"mesh algebra binding: %s\n",error.description.UTF8String);return EINVAL;}
    NSArray<MeshExtent *> *operands=f.operands;struct geometry g=f->geometry;
    MTLSize threads=MTLSizeMake(g.count,1,1),group=MTLSizeMake(MIN(256,pipeline.maxTotalThreadsPerThreadgroup),1,1);
    f.encode=^(id<MTLCommandBuffer> command){
      id<MTLComputeCommandEncoder> e=[command computeCommandEncoder];[e setComputePipelineState:pipeline];
      for(NSUInteger i=0;i<3;i++)[e setBuffer:operands[i].buffer offset:0 atIndex:i];
      [e setBytes:&g length:sizeof g atIndex:3];
      [e dispatchThreads:threads threadsPerThreadgroup:group];
      [e endEncoding];
    };
  }
  if(!f.execute) {
    f->executionKind=MESH_EXECUTION_METAL;
    void (^encode)(id<MTLCommandBuffer>)=f.encode;
    f.execute=^(MeshFunction *function){submit_encoded_metal(function,encode);};
  }
  return 0;
}
/* design/algorithm-sources.md#selected-native-contractions */
static int bind_part(MeshAlgebra *a,enum mesh_algebra_op op,struct mesh_view x,struct mesh_view y,struct mesh_view z,float alpha,float beta,size_t first,size_t count) {
  MeshFunction *f=[MeshFunction new];
  int error=prepare_part(a,f,op,x,y,z,alpha,beta,first,count);if(error)return error;
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
/* design/algorithm-sources.md#selected-native-contractions */
static int maps_cover(NSMutableData *available,NSMutableData *required) {
  const struct mesh_row_map *supplied=available.bytes,*needed=required.bytes;
  for(size_t i=0;i<required.length/sizeof *needed;i++){
    uint64_t first=needed[i].first,end=first+needed[i].count;
    while(first<end){
      uint64_t next=first;
      for(size_t j=0;j<available.length/sizeof *supplied;j++)
        if(supplied[j].first<=first && (uint64_t)supplied[j].first+supplied[j].count>next)next=(uint64_t)supplied[j].first+supplied[j].count;
      if(next==first)return 0;
      first=next;
    }
  }
  return 1;
}
/* design/algorithm-sources.md#selected-native-contractions */
int mesh_algebra_contract_select(struct mesh_algebra *handle,struct mesh_view selector,const struct mesh_view *left,const struct mesh_view *right,size_t plan_count,const struct mesh_view *inputs,size_t input_count,struct mesh_view output,float alpha,size_t *function_index) {
  MeshAlgebra *a=owner(handle);
  if(a.realized)return EBUSY;
  if(!left || !right || !plan_count || plan_count>=UINT32_MAX || !inputs || !input_count || input_count>SIZE_MAX/sizeof *inputs-1 || !function_index || !valid_view(a,selector) || selector.rows!=1 || selector.columns!=1 || selector.tensor->extents[selector.extent].shape.scalar!=MESH_U32)return EINVAL;
  struct mesh_row_map out;int error=output_region(a,output,&out);if(error)return error;
  if(out.count!=output.tensor->extents[output.extent].quantum || output_used(a,out))return EINVAL;
  NSMutableData *reads=[NSMutableData new],*views=[NSMutableData dataWithBytes:&selector length:sizeof selector];
  for(size_t i=0;i<input_count;i++){
    if(!valid_view(a,inputs[i]))return EINVAL;
    dependencies(reads,inputs[i]);
  }
  [views appendBytes:inputs length:input_count*sizeof *inputs];
  NSMutableArray<MeshFunction *> *plans=[NSMutableArray new];
  for(size_t i=0;i<plan_count;i++){
    struct mesh_view x=left[i],y=right[i],z=output;
    error=contraction_views(a,&x,&y,&z);if(error)return error;
    NSMutableData *needed=[NSMutableData new];dependencies(needed,x);dependencies(needed,y);
    if(!maps_cover(reads,needed))return EINVAL;
    MeshFunction *plan=[MeshFunction new];
    error=prepare_part(a,plan,MESH_CONTRACT,x,y,z,alpha,0,0,z.rows*z.columns);if(error)return error;
    [plans addObject:plan];
  }
  dependencies(reads,selector);
  const struct mesh_row_map *maps=reads.bytes;
  for(size_t i=0;i<reads.length/sizeof *maps;i++)if(overlaps(maps[i],out))return EINVAL;
  MeshFunction *f=[MeshFunction new];f.owner=a;f.dependencies=reads;f.inputViews=views;f.indexedInputs=[NSMutableIndexSet new];
  f->output=out;f->function=(struct mesh_row_function){.output=&f->output,.outputs=1,.rows=1};bind_dependencies(f);
  f.plans=plans;f->executionKind=plans[0]->executionKind;
  const uint32_t *selection=(const uint32_t *)selector.tensor->extents[selector.extent].address+selector.offset;
  f.execute=^(MeshFunction *function){
    uint32_t selected=*selection;
    if(selected>=plans.count){complete_part(function,EINVAL,0);return;}
    plans[selected].execute(function);
  };
  *function_index=a.functions.count;[a.functions addObject:f];return 0;
}
/* design/algorithm-sources.md#mandatory-partial-publication */
int mesh_algebra_bind(struct mesh_algebra *handle,enum mesh_algebra_op op,struct mesh_view x,struct mesh_view y,struct mesh_view z,float alpha,float beta) {
  MeshAlgebra *a=owner(handle);if(a.realized)return EBUSY;
  BOOL binary=op==MESH_ADD || op==MESH_MULTIPLY || op==MESH_CONTRACT;
  if(op>MESH_SWISH || !valid_view(a,x) || !valid_view(a,z) || (binary && !valid_view(a,y)))return EINVAL;
  struct mesh_extent *out=&z.tensor->extents[z.extent];size_t elements=z.rows*z.columns;
  struct mesh_row_map output;int region_error=output_region(a,z,&output);if(region_error)return region_error;
  if(output_used(a,output))return EINVAL;
  if(x.tensor->extents[x.extent].shape.scalar>MESH_F32 || out->shape.scalar>MESH_F32 || (binary && y.tensor->extents[y.extent].shape.scalar>MESH_F32))return EINVAL;
  if(!binary)y=x;
  if(op==MESH_CONTRACT) {
    int error=contraction_views(a,&x,&y,&z);if(error)return error;
  } else if(z.rows!=x.rows || z.columns!=(op==MESH_SUM?1:x.columns) || (binary && (x.rows!=y.rows || x.columns!=y.columns)))return EINVAL;
  NSMutableData *reads=[NSMutableData new];dependencies(reads,x);if(binary)dependencies(reads,y);
  struct mesh_row_map *maps=reads.mutableBytes;
  for(size_t i=0;i<reads.length/sizeof *maps;i++)if(overlaps(maps[i],output))return EINVAL;
  size_t step=(size_t)out->quantum*a->context->M->pgsz/(scalar_bytes(out->shape.scalar));
  for(size_t first=0;first<elements;first+=step){int error=bind_part(a,op,x,y,z,alpha,beta,first,MIN(step,elements-first));if(error)return error;}
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
    MeshFunction *f=[MeshFunction new];f.owner=a;f.dependencies=[NSMutableData new];
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
      complete_part(function,0,0);
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
int mesh_algebra_observe(struct mesh_algebra *handle,struct mesh_view view,size_t *first,size_t *count) {
  MeshAlgebra *a=owner(handle);
  if(a.realized)return EBUSY;
  if(!first || !count || !valid_view(a,view))return EINVAL;
  struct mesh_index_candidate coverage=indexed_maps(view);if(!coverage.maps)return ENOMEM;
  *first=a.observations.length/sizeof(struct mesh_row_map);*count=coverage.count;
  [a.observations appendBytes:coverage.maps length:coverage.count*sizeof *coverage.maps];free(coverage.maps);return 0;
}
/* design/algorithm-sources.md#view-scoped-consumption */
int mesh_algebra_present(struct mesh_algebra *handle,size_t first,size_t count) {
  MeshAlgebra *a=owner(handle);const struct mesh_row_map *maps=a.observations.bytes;
  size_t length=a.observations.length/sizeof *maps;if(first>length || count>length-first)return 0;
  for(size_t i=0;i<count;i++)if(!mesh_present(a->context,maps[first+i],0))return 0;
  return 1;
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
  atomic_store(&f->readyNs,clock_gettime_nsec_np(CLOCK_UPTIME_RAW));atomic_store(&f->startNs,0);atomic_store(&f->completeNs,0);atomic_store(&f->gpuStartNs,0);atomic_store(&f->gpuEndNs,0);atomic_fetch_add(&f->invocations,1);
  atomic_fetch_add(&a->submitted,1);
  if(f->executionKind==MESH_EXECUTION_CPU)atomic_fetch_add(&a->cpuSubmitted,1);
  if(f->executionKind==MESH_EXECUTION_COREML)atomic_fetch_add(&a->nativeSubmitted,1);
  dispatch_group_enter(a.executions);
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^{@autoreleasepool{atomic_store(&f->startNs,clock_gettime_nsec_np(CLOCK_UPTIME_RAW));f.execute(f);}});
}
/* design/algorithm-sources.md#derived-selector-active-domains */
static int metadata_local(MeshAlgebra *a,struct mesh_row_map map) {
  const struct mesh_row_binding *bindings=a.bindings.bytes;
  for(size_t i=0;i<a.bindings.length/sizeof *bindings;i++)
    if(bindings[i].receive && overlaps(map,(struct mesh_row_map){.first=bindings[i].first,.count=bindings[i].count}))return 0;
  return 1;
}
/* design/algorithm-sources.md#derived-selector-active-domains */
static int metadata_descends(MeshAlgebra *a,MeshFunction *source,struct mesh_row_map root,NSMutableDictionary<NSValue *,NSNumber *> *proof) {
  NSValue *identity=[NSValue valueWithPointer:(__bridge const void *)source];
  NSNumber *known=proof[identity];if(known)return known.intValue;
  proof[identity]=@0;
  if(source->function.active)return 0;
  for(uint32_t i=0;i<source->function.outputs;i++)if(!metadata_local(a,source->function.output[i]))return 0;
  int rooted=0;
  for(uint32_t i=0;i<source->function.inputs;i++){
    struct mesh_row_map input=source->function.input[i];
    for(uint32_t row=input.first;row<input.first+input.count;row++){
      if(root.first<=row && row<root.first+root.count){rooted=1;continue;}
      if(mesh_bits_all(a->context->M,MESH_CONSTANT,row,1))continue;
      int covered=0;
      for(MeshFunction *producer in a.functions){
        int contributes=0;for(uint32_t j=0;j<producer->function.outputs;j++){
          struct mesh_row_map out=producer->function.output[j];contributes|=out.first<=row && row<out.first+out.count;
        }
        if(contributes && metadata_descends(a,producer,root,proof)){covered=1;break;}
      }
      if(!covered)return 0;
      rooted=1;
    }
  }
  proof[identity]=@(rooted);return rooted;
}
/* design/algorithm-sources.md#derived-selector-active-domains */
static int indexed_active_domain(MeshAlgebra *a,struct mesh_indexed_read *d,struct mesh_active *active,struct mesh_row_map root) {
  for(uint32_t i=0;i<d->selectors;i++){
    struct mesh_row_map map=d->selector[i];
    if(root.first<=map.first && root.first+root.count>=map.first+map.count)continue;
    int rooted=0;
    for(MeshFunction *source in a.functions){
      int covers=0;for(uint32_t j=0;j<source->function.outputs;j++){
        struct mesh_row_map out=source->function.output[j];covers|=out.first<=map.first && out.first+out.count>=map.first+map.count;
      }
      if(covers && metadata_descends(a,source,root,[NSMutableDictionary new])){rooted=1;break;}
    }
    if(!rooted)return EINVAL;
  }
  if(d->domain)return d->domain==active?0:EINVAL;
  for(uint32_t i=0;i<active->maps;i++)for(uint32_t j=0;j<d->candidates;j++)for(uint32_t k=0;k<d->candidate[j].count;k++)
    if(overlaps(active->count_maps[i],d->candidate[j].maps[k]))return EINVAL;
  struct mesh_row_map *maps=realloc(d->selector,(d->selectors+active->maps)*sizeof *maps);if(!maps)return ENOMEM;
  d->selector=maps;
  for(uint32_t i=0;i<active->maps;i++){
    struct mesh_row_map map=active->count_maps[i];int covered=0;
    for(uint32_t j=0;j<d->selectors;j++)covered|=d->selector[j].first<=map.first && d->selector[j].first+d->selector[j].count>=map.first+map.count;
    if(!covered)d->selector[d->selectors++]=map;
  }
  d->domain=active;return 0;
}
/* design/algorithm-sources.md#streaming-algebra */
int mesh_algebra_realize(struct mesh_algebra *handle) {
  MeshAlgebra *a=owner(handle);if(a.realized)return 0;size_t count=a.functions.count;
  [a.routeResidency commit];
  for(MeshFunction *f in a.functions)for(struct mesh_indexed_read *d=f->function.indexed;d;d=d->next)
    for(uint32_t i=0;i<d->selectors;i++){
      struct mesh_row_map map=d->selector[i];
      for(uint32_t r=map.first;r<map.first+map.count;r++)if(!output_used(a,(struct mesh_row_map){.first=r,.count=1}))return EINVAL;
      const struct mesh_row_binding *bindings=a.bindings.bytes;
      for(size_t j=0;j<a.bindings.length/sizeof *bindings;j++)
        if(bindings[j].receive && overlaps(map,(struct mesh_row_map){.first=bindings[j].first,.count=bindings[j].count}))return EINVAL;
    }
  for(struct mesh_route *d=a->routes;d;d=d->next){
    for(uint32_t i=0;i<d->consumers;i++)if(d->functions[i]==SIZE_MAX)return EINVAL;
    for(uint32_t i=0;i<d->metadata_count;i++){
      struct mesh_row_map map=d->metadata[i];
      for(uint32_t r=map.first;r<map.first+map.count;r++)if(!output_used(a,(struct mesh_row_map){.first=r,.count=1}))return EINVAL;
      const struct mesh_row_binding *bindings=a.bindings.bytes;
      for(size_t j=0;j<a.bindings.length/sizeof *bindings;j++)if(bindings[j].receive && overlaps(map,(struct mesh_row_map){.first=bindings[j].first,.count=bindings[j].count}))return EINVAL;
    }
  }
  for(MeshFunction *f in a.functions)if(f->function.active){
    struct mesh_active *active=f->function.active;struct mesh_row_map produced={0};
    for(MeshFunction *source in a.functions)for(uint32_t i=0;i<source->function.outputs;i++){
      struct mesh_row_map out=source->function.output[i];int covers=1;
      for(uint32_t j=0;j<active->maps;j++){struct mesh_row_map map=active->count_maps[j];covers&=out.first<=map.first && out.first+out.count>=map.first+map.count;}
      if(covers)produced=out;
    }
    if(!produced.count || !metadata_local(a,produced))return EINVAL;
    for(struct mesh_indexed_read *d=f->function.indexed;d;d=d->next){int error=indexed_active_domain(a,d,active,produced);if(error)return error;}
    if(active->retired!=MESH_ABSENT && active->inputs!=f->function.inputs){mesh_rows_release(a->context,active->retired,active->inputs);active->retired=MESH_ABSENT;}
    active->inputs=f->function.inputs;
    if(active->inputs && active->retired==MESH_ABSENT){active->retired=mesh_rows_alloc(a->context,active->inputs);if(active->retired==MESH_ABSENT)return errno;}
  }
  struct mesh_row_function *functions=calloc(count?count:1,sizeof *functions);
  if(!functions)return ENOMEM;
  for(size_t i=0;i<count;i++)functions[i]=a.functions[i]->function;
  int error=mesh_realize(a->context,functions,count,a.bindings.mutableBytes,a.bindings.length/sizeof(struct mesh_row_binding),a.returns.mutableBytes,a.returns.length/sizeof(struct mesh_row_map));
  free(functions);
  if(!error){
    a.realized=YES;
    for(MeshFunction *f in a.functions)for(struct mesh_indexed_read *d=f->function.indexed;d && !error;d=d->next)
      error=mesh_execution_indexed(a->context,d,handle);
    for(MeshFunction *f in a.functions)if(f->function.active && !error)error=mesh_execution_active(a->context,f->function.active,handle);
    for(struct mesh_route *d=a->routes;d && !error;d=d->next)error=mesh_execution_route(a->context,d,handle);
    if(error)return error;
    for(MeshFunction *f in a.functions){
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
  MeshAlgebra *a=owner(handle);return (struct mesh_algebra_report){.submitted=a->submitted,.completed=atomic_load(&a->completed),.native_submitted=atomic_load(&a->nativeSubmitted),.native_backings=atomic_load(&a->nativeBackings),.code=atomic_load(&a->code),.gpu_seconds=atomic_load(&a->gpuNanoseconds)/1e9,.cpu_submitted=atomic_load(&a->cpuSubmitted)};
}

/* design/algorithm-sources.md#region-execution-timing */
size_t mesh_algebra_trace_count(struct mesh_algebra *handle) {return owner(handle).functions.count;}
/* design/algorithm-sources.md#region-execution-timing */
struct mesh_algebra_event mesh_algebra_trace(struct mesh_algebra *handle,size_t index) {
  MeshAlgebra *a=owner(handle);if(index>=a.functions.count)return (struct mesh_algebra_event){0};
  MeshFunction *f=a.functions[index];
  return (struct mesh_algebra_event){.ready_ns=atomic_load(&f->readyNs),.start_ns=atomic_load(&f->startNs),.complete_ns=atomic_load(&f->completeNs),.gpu_start_ns=atomic_load(&f->gpuStartNs),.gpu_end_ns=atomic_load(&f->gpuEndNs),.submissions=atomic_load(&f->invocations),.first_output=f->function.output[0].first,.output_maps=f->function.outputs,.kind=f->executionKind,.input_maps=f->function.inputs};
}
/* design/algorithm-sources.md#region-execution-timing */
struct mesh_row_range mesh_algebra_trace_input(struct mesh_algebra *handle,size_t index,size_t input) {
  MeshAlgebra *a=owner(handle);if(index>=a.functions.count)return (struct mesh_row_range){0};
  MeshFunction *f=a.functions[index];if(input>=f->function.inputs)return (struct mesh_row_range){0};
  return mesh_range(f->function.input[input],f->occurrence);
}

/* design/algorithm-sources.md#dynamic-reader-lifetimes */
struct mesh_row_range mesh_algebra_trace_output(struct mesh_algebra *handle,size_t index,size_t output){
  MeshAlgebra *a=owner(handle);if(index>=a.functions.count)return (struct mesh_row_range){0};
  MeshFunction *f=a.functions[index];if(output>=f->function.outputs)return (struct mesh_row_range){0};
  return mesh_range(f->function.output[output],f->occurrence);
}
/* design/algorithm-sources.md#dynamic-reader-lifetimes */
size_t mesh_algebra_trace_indexed_count(struct mesh_algebra *handle,size_t index){
  MeshAlgebra *a=owner(handle);if(index>=a.functions.count)return 0;size_t count=0;
  for(struct mesh_indexed_read *d=a.functions[index]->function.indexed;d;d=d->next){
    count+=d->selectors;
    for(uint32_t i=0;i<d->candidates;i++)count+=d->candidate[i].count+(d->candidate[i].producer?1:0);
  }
  return count;
}
/* design/algorithm-sources.md#dynamic-reader-lifetimes */
struct mesh_indexed_event mesh_algebra_trace_indexed(struct mesh_algebra *handle,size_t index,size_t entry){
  MeshAlgebra *a=owner(handle);if(index>=a.functions.count)return (struct mesh_indexed_event){0};uint32_t descriptor=0;
  for(struct mesh_indexed_read *d=a.functions[index]->function.indexed;d;d=d->next,descriptor++){
    uint32_t flags=(mesh_bits_all(a->context->M,MESH_PRESENT,d->completed,1)?8:0)|(mesh_bits_all(a->context->M,MESH_PRESENT,d->mapped,1)?16:0);
    int present=1;for(uint32_t i=0;i<d->selectors;i++)present&=mesh_bits_all(a->context->M,MESH_PRESENT,d->selector[i].first,d->selector[i].count);
    flags|=present?1:0;
    for(uint32_t i=0;i<=d->candidates;i++){
      const struct mesh_row_map *maps=i?d->candidate[i-1].maps:d->selector;
      uint32_t count=i?d->candidate[i-1].count:d->selectors;
      if(entry>=count){entry-=count;continue;}
      struct mesh_row_map map=maps[entry];uint32_t selected=i?d->selected+i-1:d->selected,retired=i?d->retired+i-1:d->retired;
      if(i)flags|=(mesh_bits_all(a->context->M,MESH_PRESENT,selected,1)?2:0)|(mesh_bits_all(a->context->M,MESH_PRESENT,retired,1)?4:0);
      return (struct mesh_indexed_event){.input=i?d->candidate[i-1].input:UINT64_MAX,.descriptor=descriptor,.role=i?1:(entry<d->vector_maps?0:2),.candidate=i?i-1:MESH_ABSENT,.first=map.first,.count=map.count,.plane=map.plane,.retired=retired,.selected=selected,.completed=d->completed,.mapped=d->mapped,.flags=flags};
    }
  }
  return (struct mesh_indexed_event){0};
}

/* design/algorithm-sources.md#canonical-reader-groups */
struct mesh_reader_event mesh_algebra_trace_input_reader(struct mesh_algebra *handle,size_t index,size_t input,uint32_t row){
  MeshAlgebra *a=owner(handle);if(index>=a.functions.count)return (struct mesh_reader_event){0};
  MeshFunction *f=a.functions[index];if(input>=f->function.inputs)return (struct mesh_reader_event){0};
  return mesh_reader_trace(a->context,f->function.input[input],f->occurrence,row);
}
/* design/algorithm-sources.md#canonical-reader-groups */
struct mesh_reader_event mesh_algebra_trace_indexed_reader(struct mesh_algebra *handle,size_t index,size_t entry,uint32_t row){
  MeshAlgebra *a=owner(handle);if(index>=a.functions.count)return (struct mesh_reader_event){0};
  for(struct mesh_indexed_read *d=a.functions[index]->function.indexed;d;d=d->next){
    for(uint32_t i=0;i<=d->candidates;i++){
      const struct mesh_row_map *maps=i?d->candidate[i-1].maps:d->selector;
      uint32_t count=i?d->candidate[i-1].count:d->selectors;
      if(entry>=count){entry-=count;continue;}
      return mesh_reader_trace(a->context,maps[entry],0,row);
    }
  }
  return (struct mesh_reader_event){0};
}

/* design/algorithm-sources.md#shared-sparse-routing-lowering */
static const struct mesh_row_map *route_trace_entry(MeshAlgebra *a,size_t entry,struct mesh_route **domain,uint32_t *identity,uint32_t *role,uint32_t *index){
  uint32_t number=0;
  for(struct mesh_route *d=a->routes;d;d=d->next,number++){
    *domain=d;*identity=number;
    if(entry<d->metadata_count){*role=0;*index=(uint32_t)entry;return &d->metadata[entry];}entry-=d->metadata_count;
    for(uint32_t i=0;i<d->candidates;i++){
      if(entry<d->candidate[i].count){*role=1;*index=i;return &d->candidate[i].maps[entry];}entry-=d->candidate[i].count;
      if(d->candidate[i].producer){if(!entry){*role=4;*index=i;return &d->candidate[i].disposition;}entry--;}
    }
    if(entry<d->consumers){*role=2;*index=(uint32_t)entry;return NULL;}entry-=d->consumers;
    if(!entry){*role=3;*index=0;return NULL;}entry--;
  }
  *domain=NULL;return NULL;
}
/* design/algorithm-sources.md#shared-sparse-routing-lowering */
size_t mesh_algebra_trace_route_count(struct mesh_algebra *handle){
  size_t count=0;
  for(struct mesh_route *d=owner(handle)->routes;d;d=d->next){count+=d->metadata_count+d->consumers+1;for(uint32_t i=0;i<d->candidates;i++)count+=d->candidate[i].count+(d->candidate[i].producer?1:0);}
  return count;
}
/* design/algorithm-sources.md#shared-sparse-routing-lowering */
struct mesh_route_event mesh_algebra_trace_route(struct mesh_algebra *handle,size_t entry){
  MeshAlgebra *a=owner(handle);struct mesh_route *d;uint32_t domain=0,role=0,index=0;
  const struct mesh_row_map *map=route_trace_entry(a,entry,&d,&domain,&role,&index);
  if(!d)return (struct mesh_route_event){0};
  struct mesh_route_event result={.function=UINT64_MAX,.domain=domain,.role=role,.index=index,.first=MESH_ABSENT,.plane=MESH_ABSENT,.retired=d->retired,.completed=d->completed,.prepared=d->prepared,.consumer=MESH_ABSENT};
  if(mesh_bits_all(a->context->M,MESH_PRESENT,d->prepared,1))result.flags|=1;
  if(mesh_bits_all(a->context->M,MESH_PRESENT,d->retired,d->candidates) && mesh_bits_all(a->context->M,MESH_PRESENT,d->completed,d->consumers))result.flags|=8;
  if(map){result.first=map->first;result.count=map->count;result.plane=map->plane;}
  if(role==1 || role==4){
    result.retired+=index;if(mesh_bits_all(a->context->M,MESH_PRESENT,result.retired,1))result.flags|=2;
    if(result.flags&9){struct mesh_route_vector v=d->owners;result.consumer=v.values[(index/v.columns)*v.row_stride+(index%v.columns)*v.column_stride];}
  }else if(role==2){result.consumer=index;result.first=d->completed+index;result.count=1;}
  else if(role==3){struct mesh_row_range table=mesh_tensor_rows(d->table,0);result.first=table.first;result.count=table.count;}
  if(result.consumer<d->consumers){result.function=d->functions[result.consumer];result.completed+=result.consumer;if(mesh_bits_all(a->context->M,MESH_PRESENT,result.completed,1))result.flags|=4;}
  return result;
}
/* design/algorithm-sources.md#shared-sparse-routing-lowering */
struct mesh_reader_event mesh_algebra_trace_route_reader(struct mesh_algebra *handle,size_t entry,uint32_t row){
  MeshAlgebra *a=owner(handle);struct mesh_route *d;uint32_t domain=0,role=0,index=0;
  const struct mesh_row_map *map=route_trace_entry(a,entry,&d,&domain,&role,&index);
  return map?mesh_reader_trace(a->context,*map,0,row):(struct mesh_reader_event){.source=MESH_ABSENT,.member=MESH_ABSENT,.plane=MESH_ABSENT,.completed=MESH_ABSENT};
}


/* design/algorithm-sources.md#active-segment-domains */
struct mesh_active_event mesh_algebra_trace_active(struct mesh_algebra *handle,size_t index){
  MeshAlgebra *a=owner(handle);struct mesh_active_event event={.function=UINT64_MAX,.disposition=MESH_ABSENT,.omitted=MESH_ABSENT,.retired=MESH_ABSENT,.count_first=MESH_ABSENT};
  if(index>=a.functions.count || !a.functions[index]->function.active)return event;
  struct mesh_active *active=a.functions[index]->function.active;
  event=(struct mesh_active_event){.function=index,.omissions=atomic_load_explicit(&active->omissions,memory_order_relaxed),.slot=active->slot,.count_first=active->count_maps[0].first,.count_maps=active->maps,.disposition=active->disposition,.omitted=active->omitted,.retired=active->retired,.inputs=active->inputs};
  if(mesh_bits_all(a->context->M,MESH_PRESENT,active->disposition,1))event.flags|=1;
  if(mesh_bits_all(a->context->M,MESH_PRESENT,active->omitted,1))event.flags|=2;
  if(!active->inputs || mesh_bits_all(a->context->M,MESH_PRESENT,active->retired,active->inputs))event.flags|=4;
  return event;
}
/* design/algorithm-sources.md#active-segment-domains */
struct mesh_row_range mesh_algebra_trace_active_count(struct mesh_algebra *handle,size_t index,size_t map){
  MeshAlgebra *a=owner(handle);if(index>=a.functions.count)return (struct mesh_row_range){0};
  struct mesh_active *active=a.functions[index]->function.active;
  return active && map<active->maps?mesh_range(active->count_maps[map],0):(struct mesh_row_range){0};
}
/* design/algorithm-sources.md#active-segment-domains */
struct mesh_reader_event mesh_algebra_trace_active_reader(struct mesh_algebra *handle,size_t index,size_t map,uint32_t row){
  MeshAlgebra *a=owner(handle);if(index>=a.functions.count)return (struct mesh_reader_event){0};
  struct mesh_active *active=a.functions[index]->function.active;
  return active && map<active->maps?mesh_reader_trace(a->context,active->count_maps[map],0,row):(struct mesh_reader_event){0};
}
/* design/algorithm-sources.md#active-segment-domains */
struct mesh_active_event mesh_algebra_trace_route_producer(struct mesh_algebra *handle,size_t entry){
  MeshAlgebra *a=owner(handle);struct mesh_route *d;uint32_t domain=0,role=0,index=0;
  route_trace_entry(a,entry,&d,&domain,&role,&index);
  return mesh_algebra_trace_active(handle,d && (role==1 || role==4) && d->candidate[index].producer?d->candidate[index].function:SIZE_MAX);
}
