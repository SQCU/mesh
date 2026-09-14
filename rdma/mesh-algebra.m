#import <Foundation/Foundation.h>
#define ACCELERATE_NEW_LAPACK
#define ACCELERATE_LAPACK_ILP64
#import <Accelerate/Accelerate.h>
#import <CommonCrypto/CommonDigest.h>
#import <Metal/Metal.h>
#import <CoreML/CoreML.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "mesh-algebra.h"
#include <limits.h>
#include <math.h>
#include <dlfcn.h>
#include <time.h>

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
  uint32_t first,page,pages,quantum;
  size_t bytes;
  void *address;
  struct mesh_shape shape;
  struct mesh_row_map output;
  struct mesh_row_function producer;
};
struct mesh_tensor { struct mesh_ctx *context; size_t count; struct mesh_extent *extents; };
struct geometry_view { uint64_t offset,rows,columns,row_stride,column_stride; };
struct geometry { struct geometry_view a,b,o; float alpha,beta; uint64_t first,count; };

@interface MeshExtent : NSObject
@property struct mesh_extent extent;
@property id<MTLBuffer> buffer;
@end
@implementation MeshExtent
@end

enum mesh_execution_kind { MESH_EXECUTION_CPU, MESH_EXECUTION_METAL, MESH_EXECUTION_COREML };

typedef void (*mesh_cpu_kernel)(const uintptr_t *);
@interface MeshCPUCode : NSObject
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
  uint32_t occurrence;
  enum mesh_execution_kind executionKind;
  _Atomic uint64_t readyNs,startNs,completeNs,gpuStartNs,gpuEndNs,invocations;
}
@property NSArray<MeshExtent *> *operands;
@property MeshCPUCode *cpuCode;
@property NSData *cpuArguments;
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
  _Atomic uint64_t submitted;
  _Atomic uint64_t nativeSubmitted,cpuSubmitted;
  uint64_t nePlannedOperations;
  uint32_t copies;
  _Atomic uint64_t completed,gpuNanoseconds,nativeBackings;
  _Atomic int64_t code;
}
@property BOOL realized,cpu;
@property dispatch_group_t executions;
@property id<MTLDevice> device;
@property id<MTLCommandQueue> queue;
@property id<MTLLibrary> library;
@property NSMutableDictionary<NSString *,id<MTLLibrary>> *libraries;
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
  for(MeshFunction *f in self.functions)for(struct mesh_indexed_read *d=f->function.indexed;d;d=d->next)
    mesh_rows_release(context,d->retired,2*d->candidates+2);
  self.functions=nil;
  self.lookup=nil;
  self.extents=nil;
  struct mesh_tensor **tensors=self.tensors.mutableBytes;
  for(size_t i=0;i<self.tensors.length/sizeof *tensors;i++) {
    struct mesh_tensor *t=tensors[i];
    for(size_t j=0;j<t->count;j++) {
      struct mesh_extent *e=&t->extents[j];
      if(e->address)mesh_view_destroy(e->address,e->bytes);
      if(e->page!=MESH_ABSENT)mesh_arena_release(context,e->page,e->pages);
      if(e->first!=MESH_ABSENT)mesh_rows_release(context,e->first,e->pages);
    }
    free(t->extents); free(t);
  }
}
@end

/* design/algorithm-sources.md#indexed-library-functions */
static void complete_part(MeshFunction *f,int64_t error,uint64_t nanoseconds) {
  MeshAlgebra *a=f.owner;atomic_store(&f->completeNs,clock_gettime_nsec_np(CLOCK_UPTIME_RAW));
  if(error)atomic_store(&a->code,error);
  else mesh_complete(a->context,&f->function,&f->occurrence,1);
  atomic_fetch_add(&a->gpuNanoseconds,nanoseconds);atomic_fetch_add(&a->completed,1);dispatch_group_leave(a.executions);
}
/* design/algorithm-sources.md#indexed-library-functions */
static void complete_function(void *context,int64_t error) {complete_part((__bridge MeshFunction *)context,error,0);}
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
  a.libraries=[NSMutableDictionary new];
  a.functions=[NSMutableArray new]; a.extents=[NSMutableArray new]; a.lookup=[NSMutableDictionary new];
  a.tensors=[NSMutableData new]; a.bindings=[NSMutableData new]; a.returns=[NSMutableData new];
  return (__bridge_retained struct mesh_algebra *)a;
}
/* design/algorithm-sources.md#cpu-indexed-execution */
struct mesh_algebra *mesh_algebra_create(struct mesh_ctx *context) {return create_algebra(context,NO);}
/* design/algorithm-sources.md#cpu-indexed-execution */
struct mesh_algebra *mesh_algebra_create_cpu(struct mesh_ctx *context) {return create_algebra(context,YES);}
/* design/algorithm-sources.md#application-metal-kernels */
uint32_t mesh_algebra_node(struct mesh_algebra *handle) {return owner(handle)->context->M->node;}
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
struct mesh_tensor *mesh_tensor_create(struct mesh_algebra *handle,const struct mesh_shape *shapes,size_t count,int transferable) {
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
  for(size_t i=0;i<count;i++)t->extents[i].first=t->extents[i].page=MESH_ABSENT;
  [a.tensors appendBytes:&t length:sizeof t];
  size_t pg=a->context->M->pgsz,align=transferable?a->context->M->block:1;
  for(size_t i=0;i<count;i++) {
    struct mesh_extent *e=&t->extents[i]; e->shape=shapes[i];
    size_t bytes=shapes[i].rows*shapes[i].columns*(scalar_bytes(shapes[i].scalar));
    size_t pages=(bytes+pg-1)/pg; pages=(pages+align-1)/align*align;
    if(pages>UINT32_MAX || pages*pg>limit){errno=EOVERFLOW;return NULL;}
    e->pages=(uint32_t)pages; e->bytes=pages*pg; e->quantum=(uint32_t)align;
    e->first=mesh_rows_alloc(a->context,e->pages);
    e->page=mesh_arena_alloc(a->context,e->pages,(uint32_t)align);
    if(e->first==MESH_ABSENT || e->page==MESH_ABSENT)return NULL;
    mesh_map(a->context,e->first,e->pages,e->page);
    e->output=(struct mesh_row_map){.first=e->first,.count=e->pages};
    e->producer=(struct mesh_row_function){.output=&e->output,.outputs=1,.rows=1};
    uint32_t *indices=malloc(pages*sizeof *indices);
    if(!indices){errno=ENOMEM;return NULL;}
    for(size_t j=0;j<pages;j++)indices[j]=e->page+(uint32_t)j;
    e->address=mesh_view_create(a->context,indices,pages); free(indices);
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
/* design/algorithm-sources.md#streaming-algebra */
void *mesh_tensor_data(struct mesh_tensor *t,uint32_t i) { return t && i<t->count?t->extents[i].address:NULL; }
/* design/algorithm-sources.md#streaming-algebra */
struct mesh_row_map mesh_tensor_rows(struct mesh_tensor *t,uint32_t i) {
  if(!t || i>=t->count)return (struct mesh_row_map){0};
  return (struct mesh_row_map){.first=t->extents[i].first,.count=t->extents[i].pages};
}
/* design/algorithm-sources.md#streaming-overlap-measurement */
int mesh_tensor_present(struct mesh_tensor *t,uint32_t extent) {
  return t && extent<t->count && mesh_present(t->context,mesh_tensor_rows(t,extent),0);
}
/* design/algorithm-sources.md#streaming-algebra */
int mesh_tensor_constant(struct mesh_tensor *t,uint32_t i) {
  if(!t || i>=t->count)return EINVAL;
  struct mesh_row_map m=mesh_tensor_rows(t,i);mesh_constant(t->context,m.first,m.count);return 0;
}
/* design/algorithm-sources.md#streaming-algebra */
int mesh_tensor_writable(struct mesh_tensor *t,uint32_t i) {
  if(!t || i>=t->count)return 0;struct mesh_extent *e=&t->extents[i];
  return mesh_writable(t->context,e->first,e->pages);
}
/* design/algorithm-sources.md#streaming-algebra */
int mesh_tensor_issue(struct mesh_tensor *t,uint32_t i) {
  if(!t || i>=t->count)return 0;
  uint32_t index=0;return mesh_issue(t->context,&t->extents[i].producer,&index,1)!=0;
}
/* design/algorithm-sources.md#streaming-algebra */
void mesh_tensor_complete(struct mesh_tensor *t,uint32_t i) {
  uint32_t index=0;mesh_complete(t->context,&t->extents[i].producer,&index,1);
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
      [maps appendBytes:&m length:sizeof m];c=last+1;
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
  if(!((v.column_stride==1 && v.row_stride==v.columns) || (v.row_stride==1 && v.column_stride==v.rows)))return EINVAL;
  if(v.rows>SIZE_MAX/v.columns)return EOVERFLOW;
  size_t count=v.rows*v.columns,end=v.offset+count;
  if(v.offset%quantum || (end!=elements && end%quantum))return EINVAL;
  *m=(struct mesh_row_map){.first=e->first+(uint32_t)(v.offset/unit),.count=(uint32_t)(((count+quantum-1)/quantum)*e->quantum)};
  return 0;
}
/* design/algorithm-sources.md#region-streaming-review */
static int overlaps(struct mesh_row_map a,struct mesh_row_map b) {
  return a.first<b.first+b.count && b.first<a.first+a.count;
}
/* design/algorithm-sources.md#indexed-library-functions */
int mesh_algebra_function(struct mesh_algebra *handle,const struct mesh_view *inputs,size_t input_count,const struct mesh_view *outputs,size_t output_count,mesh_submission submit,void *binding) {
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
  f.execute=^(MeshFunction *function){submit(binding,complete_function,(__bridge void *)function);};
  [a.functions addObject:f];return 0;
}

/* design/algorithm-sources.md#dynamic-reader-lifetimes */
static struct mesh_index_candidate indexed_maps(struct mesh_view view){
  MeshFunction *f=[MeshFunction new];f.dependencies=[NSMutableData new];dependencies(f.dependencies,view);bind_dependencies(f);
  size_t bytes=f->function.inputs*sizeof(struct mesh_row_map);
  struct mesh_row_map *maps=malloc(bytes);if(maps)memcpy(maps,f->function.input,bytes);
  return (struct mesh_index_candidate){.maps=maps,.count=f->function.inputs};
}
/* design/algorithm-sources.md#dynamic-reader-lifetimes */
int mesh_algebra_indexed(struct mesh_algebra *handle,size_t index,struct mesh_view selector,const size_t *candidate_inputs,size_t count){
  MeshAlgebra *a=owner(handle);
  if(a.realized || index>=a.functions.count || !count || count>(UINT32_MAX-2)/2 || !candidate_inputs || !valid_view(a,selector))return EINVAL;
  if(selector.tensor->extents[selector.extent].shape.scalar!=MESH_U32)return EINVAL;
  MeshFunction *f=a.functions[index];const struct mesh_view *inputs=f.inputViews.bytes;size_t input_count=f.inputViews.length/sizeof *inputs;
  for(size_t i=0;i<count;i++)if(candidate_inputs[i]>=input_count || !valid_view(a,inputs[candidate_inputs[i]]))return EINVAL;
  struct mesh_indexed_read *d=calloc(1,sizeof *d);if(!d)return ENOMEM;
  d->candidate=calloc(count,sizeof *d->candidate);if(!d->candidate){free(d);return ENOMEM;}
  d->candidates=(uint32_t)count;
  struct mesh_index_candidate selection=indexed_maps(selector);d->selector=selection.maps;d->selectors=selection.count;
  int error=d->selector?0:ENOMEM;
  for(size_t i=0;i<count && !error;i++){d->candidate[i]=indexed_maps(inputs[candidate_inputs[i]]);if(!d->candidate[i].maps)error=ENOMEM;}
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
  for(size_t i=0;i<count;i++)[f.indexedInputs addIndex:candidate_inputs[i]];
  d->next=f->function.indexed;f->function.indexed=d;f.dependencies=[NSMutableData new];
  for(size_t i=0;i<input_count;i++)if(![f.indexedInputs containsIndex:i])dependencies(f.dependencies,inputs[i]);
  for(struct mesh_indexed_read *part=f->function.indexed;part;part=part->next)
    [f.dependencies appendBytes:part->selector length:part->selectors*sizeof *part->selector];
  bind_dependencies(f);
  return 0;
}

struct cpu_operand {const void *address;struct geometry_view view;float (*load)(const void *,size_t);};
/* design/algorithm-sources.md#application-metal-kernels */
static void submit_metal(void *binding,mesh_completion complete,void *context) {
  MeshFunction *f=(__bridge MeshFunction *)context;
  id<MTLCommandBuffer> command=[f.owner.queue commandBuffer];f.encode(command);
  [command addCompletedHandler:^(id<MTLCommandBuffer> done){
    atomic_store(&f->gpuStartNs,(uint64_t)(done.GPUStartTime*1e9));atomic_store(&f->gpuEndNs,(uint64_t)(done.GPUEndTime*1e9));
    atomic_fetch_add(&f.owner->gpuNanoseconds,(uint64_t)((done.GPUEndTime-done.GPUStartTime)*1e9));
    complete(context,done.error.code);
  }];
  [command commit];
}
/* design/algorithm-sources.md#application-metal-kernels */
int mesh_algebra_metal(struct mesh_algebra *handle,const char *text,const struct mesh_metal_dispatch *dispatches,size_t dispatch_count,const struct mesh_metal_constant *constants,size_t constant_count,const struct mesh_view *inputs,size_t input_count,const struct mesh_view *outputs,size_t output_count) {
  MeshAlgebra *a=owner(handle);
  if(a.realized)return EBUSY;
  if(a.cpu || !text || !dispatch_count || !dispatches || constant_count>30 || (constant_count && !constants))return EINVAL;
  NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
  NSString *key=@(text);id<MTLLibrary> library=a.libraries[key];
  if(!library){library=[a.device newLibraryWithSource:key options:options error:&error];if(library)a.libraries[key]=library;}
  if(!library){fprintf(stderr,"mesh Metal kernel: %s\n",error.description.UTF8String);return EINVAL;}
  NSMutableArray<id<MTLComputePipelineState>> *pipelines=[NSMutableArray new];
  for(size_t i=0;i<dispatch_count;i++) {
    const struct mesh_metal_dispatch *d=&dispatches[i];
    if(!d->name || !d->grid[0] || !d->grid[1] || !d->grid[2] || !d->group[0] || !d->group[1] || !d->group[2])return EINVAL;
    id<MTLFunction> function=[library newFunctionWithName:@(d->name)];
    if(!function)return ENOENT;
    id<MTLComputePipelineState> pipeline=[a.device newComputePipelineStateWithFunction:function error:&error];
    if(!pipeline){fprintf(stderr,"mesh Metal pipeline: %s\n",error.description.UTF8String);return EINVAL;}
    if(d->group[0]>pipeline.maxTotalThreadsPerThreadgroup/d->group[1]/d->group[2])return EINVAL;
    if(d->argument_buffer>constant_count || (!d->argument_buffer && d->argument_offset) || (d->argument_buffer && d->argument_offset>=constants[d->argument_buffer-1].length))return EINVAL;
    [pipelines addObject:pipeline];
  }
  NSMutableArray<id<MTLBuffer>> *buffers=[NSMutableArray new],*resources=[NSMutableArray new];
  id<MTLBuffer> addresses=[a.device newBufferWithLength:(input_count+output_count)*sizeof(uint64_t) options:MTLResourceStorageModeShared];
  if(!addresses)return ENOMEM;
  for(size_t i=0;i<input_count+output_count;i++) {
    struct mesh_view v=i<input_count?inputs[i]:outputs[i-input_count];
    if(!valid_view(a,v))return EINVAL;
    struct mesh_extent *extent=&v.tensor->extents[v.extent];
    id<MTLBuffer> buffer=a.lookup[[NSValue valueWithPointer:extent]].buffer;
    ((uint64_t *)addresses.contents)[i]=buffer.gpuAddress+v.offset*(scalar_bytes(extent->shape.scalar));
    [resources addObject:buffer];

  }
  for(size_t i=0;i<constant_count;i++) {
    if(!constants[i].bytes || !constants[i].length)return EINVAL;
    id<MTLBuffer> buffer=[a.device newBufferWithBytes:constants[i].bytes length:constants[i].length options:MTLResourceStorageModeShared];
    if(!buffer)return ENOMEM;[buffers addObject:buffer];
  }
  NSData *geometry=[NSData dataWithBytes:dispatches length:dispatch_count*sizeof *dispatches];
  int status=mesh_algebra_function(handle,inputs,input_count,outputs,output_count,submit_metal,NULL);
  if(status)return status;
  MeshFunction *f=a.functions.lastObject;f->executionKind=MESH_EXECUTION_METAL;
  f.encode=^(id<MTLCommandBuffer> command){
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    [encoder setBuffer:addresses offset:0 atIndex:0];
    for(NSUInteger i=0;i<buffers.count;i++)[encoder setBuffer:buffers[i] offset:0 atIndex:i+1];
    for(id<MTLBuffer> buffer in resources)[encoder useResource:buffer usage:MTLResourceUsageRead|MTLResourceUsageWrite];
    const struct mesh_metal_dispatch *d=geometry.bytes;
    for(NSUInteger i=0;i<pipelines.count;i++) {
      [encoder setComputePipelineState:pipelines[i]];
      if(d[i].argument_buffer)[encoder setBuffer:buffers[d[i].argument_buffer-1] offset:d[i].argument_offset atIndex:d[i].argument_buffer];
      [encoder dispatchThreadgroups:MTLSizeMake(d[i].grid[0],d[i].grid[1],d[i].grid[2]) threadsPerThreadgroup:MTLSizeMake(d[i].group[0],d[i].group[1],d[i].group[2])];
    }
    [encoder endEncoding];
  };
  return 0;
}

/* design/algorithm-sources.md#region-expression-fusion */
static void submit_cpu(void *binding,mesh_completion complete,void *context) {
  MeshFunction *f=(__bridge MeshFunction *)context;f.cpuCode.kernel(f.cpuArguments.bytes);complete(context,0);
}
/* design/algorithm-sources.md#region-expression-fusion */
int mesh_algebra_source(struct mesh_algebra *handle,const char *cpu_source,const char *metal_source,const struct mesh_view *inputs,size_t input_count,struct mesh_view output) {
  MeshAlgebra *a=owner(handle);
  if(a.realized || !cpu_source || !metal_source || !valid_view(a,output))return EINVAL;
  if(!a.cpu) {
    struct mesh_metal_dispatch dispatch={.name="mesh_expression",.grid={output.rows,1,1},.group={32,1,1}};
    return mesh_algebra_metal(handle,metal_source,&dispatch,1,NULL,0,inputs,input_count,&output,1);
  }
  NSString *source=@(cpu_source);if(!a.cpuCode)a.cpuCode=[NSMutableDictionary new];
  MeshCPUCode *library=a.cpuCode[source];
  if(!library) {
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
    library=[MeshCPUCode new];library.handle=dlopen(output.fileSystemRepresentation,RTLD_NOW|RTLD_LOCAL);
    [files removeItemAtPath:directory error:nil];
    if(!library.handle){fprintf(stderr,"mesh CPU kernel: %s\n",dlerror());return EIO;}
    library.kernel=(mesh_cpu_kernel)dlsym(library.handle,"mesh_expression");
    if(!library.kernel){fprintf(stderr,"mesh CPU symbol: %s\n",dlerror());return EIO;}
    a.cpuCode[source]=library;
  }
  NSMutableData *addresses=[NSMutableData dataWithLength:(input_count+1)*sizeof(uintptr_t)];
  uintptr_t *pointers=addresses.mutableBytes;
  for(size_t i=0;i<input_count+1;i++) {
    struct mesh_view v=i<input_count?inputs[i]:output;
    if(!valid_view(a,v))return EINVAL;
    struct mesh_extent *extent=&v.tensor->extents[v.extent];
    pointers[i]=(uintptr_t)extent->address+v.offset*scalar_bytes(extent->shape.scalar);
  }
  int status=mesh_algebra_function(handle,inputs,input_count,&output,1,submit_cpu,NULL);
  if(status)return status;
  MeshFunction *f=a.functions.lastObject;f->executionKind=MESH_EXECUTION_CPU;
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
    case MESH_CONTRACT:value=^float(size_t r,size_t c){float sum=0;for(size_t k=0;k<a.view.columns;k++)sum+=cpu_get(a,r,k)*cpu_get(b,k,c);return alpha*sum;};break;
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
    dispatch_semaphore_t ready=dispatch_semaphore_create(0);
    [MLComputePlan loadContentsOfURL:compiled configuration:configuration completionHandler:^(MLComputePlan *plan,NSError *failure){
      for(MLModelStructureProgramFunction *function in plan.modelStructure.program.functions.allValues)for(MLModelStructureProgramOperation *operation in function.block.operations) {
        id<MLComputeDeviceProtocol> device=[plan computeDeviceUsageForMLProgramOperation:operation].preferredComputeDevice;
        if([device isKindOfClass:MLNeuralEngineComputeDevice.class])a->nePlannedOperations++;
      }
      if(failure)fprintf(stderr,"mesh CoreML plan: %s\n",failure.description.UTF8String);
      dispatch_semaphore_signal(ready);
    }];
    dispatch_semaphore_wait(ready,DISPATCH_TIME_FOREVER);
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
/* design/algorithm-sources.md#mandatory-partial-publication */
static int bind_part(MeshAlgebra *a,enum mesh_algebra_op op,struct mesh_view x,struct mesh_view y,struct mesh_view z,float alpha,float beta,size_t first,size_t count) {
  MeshFunction *f=[MeshFunction new];f.owner=a;f.dependencies=[NSMutableData new];
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
      [rectangles addObject:@[@(rr),@(cc),@(x.columns),@(xv.tensor->extents[xv.extent].shape.scalar==MESH_F16)]];
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
    f.encode=^(id<MTLCommandBuffer> command){
      id<MTLComputeCommandEncoder> e=[command computeCommandEncoder];[e setComputePipelineState:pipeline];
      for(NSUInteger i=0;i<3;i++)[e setBuffer:operands[i].buffer offset:0 atIndex:i];
      [e setBytes:&g length:sizeof g atIndex:3];
      [e dispatchThreads:MTLSizeMake(g.count,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(256,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
      [e endEncoding];
    };
  }
  if(!f.execute) {
    f->executionKind=MESH_EXECUTION_METAL;
    void (^encode)(id<MTLCommandBuffer>)=f.encode;id<MTLCommandQueue> queue=a.queue;
    f.execute=^(MeshFunction *function) {
      id<MTLCommandBuffer> command=[queue commandBuffer];encode(command);
      [command addCompletedHandler:^(id<MTLCommandBuffer> done){atomic_store(&function->gpuStartNs,(uint64_t)(done.GPUStartTime*1e9));atomic_store(&function->gpuEndNs,(uint64_t)(done.GPUEndTime*1e9));complete_part(function,done.error.code,(uint64_t)((done.GPUEndTime-done.GPUStartTime)*1e9));}];
      [command commit];
    };
  }
  [a.functions addObject:f];return 0;
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
    if(x.columns!=y.rows || z.rows!=x.rows || z.columns!=y.columns || z.column_stride!=1 || !x.row_stride || !x.column_stride || !y.row_stride || !y.column_stride || (x.column_stride!=1 && x.row_stride!=1) || (y.column_stride!=1 && y.row_stride!=1))return EINVAL;
    if(x.tensor->extents[x.extent].shape.scalar!=y.tensor->extents[y.extent].shape.scalar)return EINVAL;
  } else if(z.rows!=x.rows || z.columns!=(op==MESH_SUM?1:x.columns) || (binary && (x.rows!=y.rows || x.columns!=y.columns)))return EINVAL;
  NSMutableData *reads=[NSMutableData new];dependencies(reads,x);if(binary)dependencies(reads,y);
  struct mesh_row_map *maps=reads.mutableBytes;
  for(size_t i=0;i<reads.length/sizeof *maps;i++)if(overlaps(maps[i],output))return EINVAL;
  size_t step=(size_t)out->quantum*a->context->M->pgsz/(scalar_bytes(out->shape.scalar));
  for(size_t first=0;first<elements;first+=step){int error=bind_part(a,op,x,y,z,alpha,beta,first,MIN(step,elements-first));if(error)return error;}
  return 0;
}

/* design/algorithm-sources.md#indexed-library-functions */
static int bind_copy(MeshAlgebra *a,struct mesh_extent *s,struct mesh_extent *d) {
  struct mesh_row_map output={.first=d->first,.count=d->pages};
  if(output_used(a,output) || s==d)return EINVAL;
  size_t bytes=s->shape.rows*s->shape.columns*(scalar_bytes(s->shape.scalar));
  size_t step=(size_t)d->quantum*a->context->M->pgsz;
  for(size_t offset=0;offset<bytes;offset+=step) {
    MeshFunction *f=[MeshFunction new];f.owner=a;f.dependencies=[NSMutableData new];
    size_t count=MIN(step,bytes-offset);
    struct mesh_row_map input={.first=s->first+(uint32_t)(offset/a->context->M->pgsz),.count=(uint32_t)((count+a->context->M->pgsz-1)/a->context->M->pgsz)};
    [f.dependencies appendBytes:&input length:sizeof input];
    f->output=(struct mesh_row_map){.first=d->first+(uint32_t)(offset/a->context->M->pgsz),.count=d->quantum};
    f->function=(struct mesh_row_function){.output=&f->output,.outputs=1,.rows=1};bind_dependencies(f);
    f.execute=^(MeshFunction *function){memcpy((char *)d->address+offset,(char *)s->address+offset,count);complete_part(function,0,0);};
    [a.functions addObject:f];
  }
  return 0;
}
/* design/algorithm-sources.md#pallas-indexed-destinations */
int mesh_algebra_copy(struct mesh_algebra *handle,struct mesh_endpoint source,struct mesh_endpoint destination,size_t count,uint16_t queue) {
  MeshAlgebra *a=owner(handle);
  if(a.realized)return EBUSY;
  if(!source.tensor || !destination.tensor || source.tensor->context!=a->context || destination.tensor->context!=a->context || !count || count>UINT32_MAX-a->copies)return EINVAL;
  for(size_t i=0;i<count;i++) {
    uint64_t si=source.first+(uint64_t)i*source.stride,di=destination.first+(uint64_t)i*destination.stride;
    if(si>=source.tensor->count || di>=destination.tensor->count)return EINVAL;
    struct mesh_extent *s=&source.tensor->extents[si],*d=&destination.tensor->extents[di];
    if(s->shape.rows!=d->shape.rows || s->shape.columns!=d->shape.columns || s->shape.scalar!=d->shape.scalar || (source.peer!=destination.peer && s->pages!=d->pages))return EINVAL;
  }
  uint32_t block=a->context->M->block,maximum=0;uint64_t messages=0;
  for(size_t i=0;i<count;i++) {
    struct mesh_extent *s=&source.tensor->extents[source.first+i*source.stride];
    if(source.peer==destination.peer) {
      if(source.peer==a->context->M->node) {
        int error=bind_copy(a,s,&destination.tensor->extents[destination.first+i*destination.stride]);
        if(error)return error;
      }
    } else {if(s->pages%block)return EINVAL;maximum=MAX(maximum,s->pages);messages+=s->pages/block;}
  }
  if(messages>UINT32_MAX-a->copies)return EOVERFLOW;
  for(uint32_t offset=0;offset<maximum;offset+=block)for(size_t i=0;i<count;i++) {
    struct mesh_extent *s=&source.tensor->extents[source.first+i*source.stride],*d=&destination.tensor->extents[destination.first+i*destination.stride];
    if(offset>=s->pages)continue;
    uint32_t identity=a->copies++;
    if(source.peer==a->context->M->node || destination.peer==a->context->M->node) {
      int receive=destination.peer==a->context->M->node;
      struct mesh_row_binding b={.first=(receive?d:s)->first+offset,.count=block,.bytes=(uint32_t)MIN((size_t)block*a->context->M->pgsz,s->shape.rows*s->shape.columns*scalar_bytes(s->shape.scalar)-(size_t)offset*a->context->M->pgsz),.binding=identity,.queue=queue,.receive=receive};
      [a.bindings appendBytes:&b length:sizeof b];
    }
  }
  return 0;
}

/* design/algorithm-sources.md#streaming-algebra */
int mesh_algebra_return(struct mesh_algebra *handle,struct mesh_tensor *t,uint32_t i) {
  MeshAlgebra *a=owner(handle);
  if(a.realized)return EBUSY;
  if(!t || i>=t->count || t->context!=a->context)return EINVAL;
  struct mesh_row_map m=mesh_tensor_rows(t,i);[a.returns appendBytes:&m length:sizeof m];return 0;
}
/* design/algorithm-sources.md#indexed-library-functions */
int mesh_algebra_export(struct mesh_algebra *handle,struct mesh_tensor *t,uint32_t extent,size_t *index) {
  if(!index)return EINVAL;
  size_t next=owner(handle).returns.length/sizeof(struct mesh_row_map);
  int error=mesh_algebra_return(handle,t,extent);if(!error)*index=next;return error;
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
    a.realized=YES;
    for(MeshFunction *f in a.functions)for(struct mesh_indexed_read *d=f->function.indexed;d && !error;d=d->next)
      error=mesh_execution_indexed(a->context,d,handle);
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
  MeshAlgebra *a=owner(handle);return (struct mesh_algebra_report){.submitted=a->submitted,.completed=atomic_load(&a->completed),.native_submitted=atomic_load(&a->nativeSubmitted),.native_backings=atomic_load(&a->nativeBackings),.ne_planned_operations=a->nePlannedOperations,.code=atomic_load(&a->code),.gpu_seconds=atomic_load(&a->gpuNanoseconds)/1e9,.cpu_submitted=atomic_load(&a->cpuSubmitted)};
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
