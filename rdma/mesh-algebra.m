#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreML/CoreML.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "mesh-algebra.h"
#include <limits.h>

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
" if(op==3)y=tanh(x); if(op==4)y=exp(x); if(op==7)y=rsqrt(x);\n"
" if(op==5){y=0; for(ulong k=0;k<g.a.columns;k++)y+=get(a,g.a,r,k,a16);}\n"
" ulong j=g.o.offset+r*g.o.row_stride+c*g.o.column_stride; if(o16)((device half*)o)[j]=half(y);else ((device float*)o)[j]=y;\n"
"}\n";

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

@interface MeshFunction : NSObject {
@public
  struct mesh_row_map output;
  struct mesh_row_function function;
  struct geometry geometry;
}
@property NSArray<MeshExtent *> *operands;
@property NSMutableData *dependencies;
@property(copy) void (^encode)(id<MTLCommandBuffer>);
@property(copy) void (^execute)(void (^)(int64_t,uint64_t));
@end
@implementation MeshFunction
@end

@interface MeshAlgebra : NSObject {
@public
  struct mesh_ctx *context;
  uint64_t submitted,nativeSubmitted,nePlannedOperations;
  uint32_t copies;
  _Atomic uint64_t completed,gpuNanoseconds,nativeBackings;
  _Atomic int64_t code;
}
@property BOOL realized;
@property id<MTLDevice> device;
@property id<MTLCommandQueue> queue;
@property id<MTLLibrary> library;
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
  self.functions=nil;
  self.lookup=nil;
  self.extents=nil;
  struct mesh_tensor **tensors=self.tensors.mutableBytes;
  for(size_t i=0;i<self.tensors.length/sizeof *tensors;i++) {
    struct mesh_tensor *t=tensors[i];
    for(size_t j=0;j<t->count;j++) {
      struct mesh_extent *e=&t->extents[j];
      if(e->address) mesh_view_destroy(e->address,e->bytes);
      if(e->first!=MESH_ABSENT)mesh_rows_release(context,e->first,e->pages);
      if(e->page!=MESH_ABSENT)mesh_arena_release(context,e->page,e->pages);
    }
    free(t->extents); free(t);
  }
}
@end

/* design/algorithm-sources.md#streaming-algebra */
static MeshAlgebra *owner(struct mesh_algebra *a) { return (__bridge MeshAlgebra *)a; }
/* design/algorithm-sources.md#streaming-algebra */
struct mesh_algebra *mesh_algebra_create(struct mesh_ctx *context) {
  if(!context || !context->M){errno=EINVAL;return NULL;}
  MeshAlgebra *a=[MeshAlgebra new]; a->context=context;
  a.device=MTLCreateSystemDefaultDevice(); a.queue=[a.device newCommandQueue];
  NSError *error=nil; a.library=[a.device newLibraryWithSource:source options:nil error:&error];
  if(!a.queue || !a.library){fprintf(stderr,"mesh algebra: %s\n",error.description.UTF8String);errno=ENODEV;return NULL;}
  a.functions=[NSMutableArray new]; a.extents=[NSMutableArray new]; a.lookup=[NSMutableDictionary new];
  a.tensors=[NSMutableData new]; a.bindings=[NSMutableData new]; a.returns=[NSMutableData new];
  return (__bridge_retained struct mesh_algebra *)a;
}
/* design/algorithm-sources.md#coreml-partial-execution */
int mesh_algebra_coreml(struct mesh_algebra *handle,const char *python,const char *generator,const char *cache) {
  MeshAlgebra *a=owner(handle);
  if(a.realized || a.functions.count)return EBUSY;
  if(!python || !generator || !cache)return EINVAL;
  a.coremlPython=@(python);a.coremlGenerator=@(generator);a.coremlCache=@(cache);a.models=[NSMutableDictionary new];return 0;
}
/* design/algorithm-sources.md#streaming-algebra */
void mesh_algebra_destroy(struct mesh_algebra *a) { if(a)CFBridgingRelease(a); }

/* design/algorithm-sources.md#streaming-algebra */
struct mesh_tensor *mesh_tensor_create(struct mesh_algebra *handle,const struct mesh_shape *shapes,size_t count,int transferable) {
  MeshAlgebra *a=owner(handle);
  if(a.realized){errno=EBUSY;return NULL;}
  if(!count || !shapes || count>UINT32_MAX){errno=EINVAL;return NULL;}
  for(size_t i=0;i<count;i++) {
    size_t bytes=shapes[i].scalar==MESH_F16?2:4;
    if(shapes[i].scalar>MESH_F32 || !shapes[i].rows || !shapes[i].columns || shapes[i].rows>SIZE_MAX/shapes[i].columns/bytes){errno=EINVAL;return NULL;}
    if(shapes[i].rows*shapes[i].columns*bytes>a.device.maxBufferLength){errno=EOVERFLOW;return NULL;}
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
    size_t bytes=shapes[i].rows*shapes[i].columns*(shapes[i].scalar==MESH_F16?2:4);
    size_t pages=(bytes+pg-1)/pg; pages=(pages+align-1)/align*align;
    if(pages>UINT32_MAX || pages*pg>a.device.maxBufferLength){errno=EOVERFLOW;return NULL;}
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
    storage.buffer=[a.device newBufferWithBytesNoCopy:e->address length:e->bytes options:MTLResourceStorageModeShared|MTLResourceHazardTrackingModeUntracked deallocator:nil];
    if(!storage.buffer){errno=ENOMEM;return NULL;}
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
/* design/algorithm-sources.md#streaming-algebra */
int mesh_tensor_constant(struct mesh_tensor *t,uint32_t i) {
  if(!t || i>=t->count)return EINVAL;
  struct mesh_row_map m=mesh_tensor_rows(t,i);mesh_constant(t->context,m.first,m.count);return 0;
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
int mesh_tensor_publish(struct mesh_tensor *t,uint32_t i) {
  if(!mesh_tensor_issue(t,i))return 0;
  mesh_tensor_complete(t,i);return 1;
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
  size_t bytes=e.extent.shape.scalar==MESH_F16?2:4;
  MPSMatrixDescriptor *d=[MPSMatrixDescriptor matrixDescriptorWithRows:transpose?v.columns:v.rows columns:transpose?v.rows:v.columns rowBytes:(transpose?v.column_stride:v.row_stride)*bytes dataType:bytes==2?MPSDataTypeFloat16:MPSDataTypeFloat32];
  return [[MPSMatrix alloc]initWithBuffer:e.buffer offset:v.offset*bytes descriptor:d];
}

/* design/algorithm-sources.md#mandatory-partial-publication */
static void dependencies(NSMutableData *maps,struct mesh_view v) {
  struct mesh_extent *e=&v.tensor->extents[v.extent];
  size_t unit=v.tensor->context->M->pgsz/(e->shape.scalar==MESH_F16?2:4);
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
/* design/algorithm-sources.md#coreml-partial-execution */
static MLMultiArray *native_array(struct mesh_view v,NSError **error) {
  struct mesh_extent *e=&v.tensor->extents[v.extent];size_t bytes=e->shape.scalar==MESH_F16?2:4;
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
    NSString *stem=[a.coremlCache stringByAppendingPathComponent:[NSString stringWithFormat:@"part-%lu",(unsigned long)a.models.count]];
    [[NSFileManager defaultManager]createDirectoryAtPath:a.coremlCache withIntermediateDirectories:YES attributes:nil error:&error];
    NSString *request=[stem stringByAppendingPathExtension:@"json"];
    if(![json writeToFile:request options:NSDataWritingAtomic error:&error])return (int)error.code;
    NSTask *task=[NSTask new];task.executableURL=[NSURL fileURLWithPath:a.coremlPython];
    task.arguments=@[a.coremlGenerator,request,stem];
    if(![task launchAndReturnError:&error])return (int)error.code;
    [task waitUntilExit];if(task.terminationStatus)return EIO;
    NSURL *compiled=[NSURL fileURLWithPath:[stem stringByAppendingPathExtension:@"mlmodelc"]];
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
  struct mesh_extent *out=&z.tensor->extents[z.extent];size_t bytes=out->shape.scalar==MESH_F16?2:4;
  MLMultiArray *output=[[MLMultiArray alloc]initWithDataPointer:(char *)out->address+first*bytes shape:@[@(count)] dataType:bytes==2?MLMultiArrayDataTypeFloat16:MLMultiArrayDataTypeFloat32 strides:@[@1] deallocator:nil error:&error];
  MLDictionaryFeatureProvider *inputs=[[MLDictionaryFeatureProvider alloc]initWithDictionary:features error:&error];
  if(!output || !inputs)return (int)error.code;
  MLPredictionOptions *options=[MLPredictionOptions new];options.outputBackings=@{@"z":output};
  __unsafe_unretained MeshAlgebra *context=a;
  f.execute=^(void (^complete)(int64_t,uint64_t)) {
    context->nativeSubmitted++;
    [model predictionFromFeatures:inputs options:options completionHandler:^(id<MLFeatureProvider> prediction,NSError *failure){
      BOOL backing=[prediction featureValueForName:@"z"].multiArrayValue==output;
      if(backing)atomic_fetch_add(&context->nativeBackings,1);
      complete(failure?failure.code:backing?0:EPROTO,0);
    }];
  };
  return 0;
}
/* design/algorithm-sources.md#mandatory-partial-publication */
static int bind_part(MeshAlgebra *a,enum mesh_algebra_op op,struct mesh_view x,struct mesh_view y,struct mesh_view z,float alpha,float beta,size_t first,size_t count) {
  MeshFunction *f=[MeshFunction new];f.dependencies=[NSMutableData new];
  struct mesh_extent *out=&z.tensor->extents[z.extent];
  size_t scalar=out->shape.scalar==MESH_F16?2:4;
  f->output=(struct mesh_row_map){.first=out->first+(uint32_t)(first*scalar/a->context->M->pgsz),.count=out->quantum};
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
    if(op==MESH_CONTRACT && a.coremlPython) {
      NSError *error=nil;MLMultiArray *left=native_array(xv,&error),*right=native_array(yv,&error);
      if(!left || !right)return (int)error.code;
      NSUInteger i=rectangles.count;
      features[[NSString stringWithFormat:@"x%lu",(unsigned long)i]]=left;
      features[[NSString stringWithFormat:@"w%lu",(unsigned long)i]]=right;
      [rectangles addObject:@[@(rr),@(cc),@(x.columns),@(xv.tensor->extents[xv.extent].shape.scalar==MESH_F16)]];
    } else if(op==MESH_CONTRACT) {
      BOOL tx=xv.column_stride!=1,ty=yv.column_stride!=1;
      struct mesh_view zv=mesh_view_slice(z,zr,zc,rr,cc);
      [matrices addObject:@[matrix(f.operands[0],xv,tx),matrix(f.operands[1],yv,ty),matrix(f.operands[2],zv,NO)]];
      [products addObject:[[MPSMatrixMultiplication alloc]initWithDevice:a.device transposeLeft:tx transposeRight:ty resultRows:rr resultColumns:cc interiorColumns:x.columns alpha:alpha beta:0]];
    }
    at+=nr*nc;left-=nr*nc;
  }
  struct mesh_row_map *maps=f.dependencies.mutableBytes;size_t length=f.dependencies.length/sizeof *maps,used=0;
  qsort(maps,length,sizeof *maps,compare_maps);
  for(size_t i=0;i<length;i++) {
    if(used && maps[i].first<=maps[used-1].first+maps[used-1].count)maps[used-1].count=MAX(maps[used-1].first+maps[used-1].count,maps[i].first+maps[i].count)-maps[used-1].first;
    else maps[used++]=maps[i];
  }
  f->function=(struct mesh_row_function){maps,&f->output,(uint32_t)used,1,1};
  if(op==MESH_CONTRACT && a.coremlPython) {
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
    void (^encode)(id<MTLCommandBuffer>)=f.encode;id<MTLCommandQueue> queue=a.queue;
    f.execute=^(void (^complete)(int64_t,uint64_t)) {
      id<MTLCommandBuffer> command=[queue commandBuffer];encode(command);
      [command addCompletedHandler:^(id<MTLCommandBuffer> done){complete(done.error.code,(uint64_t)((done.GPUEndTime-done.GPUStartTime)*1e9));}];
      [command commit];
    };
  }
  [a.functions addObject:f];return 0;
}
/* design/algorithm-sources.md#mandatory-partial-publication */
int mesh_algebra_bind(struct mesh_algebra *handle,enum mesh_algebra_op op,struct mesh_view x,struct mesh_view y,struct mesh_view z,float alpha,float beta) {
  MeshAlgebra *a=owner(handle);if(a.realized)return EBUSY;
  BOOL binary=op==MESH_ADD || op==MESH_MULTIPLY || op==MESH_CONTRACT;
  if(op>MESH_RSQRT || !valid_view(a,x) || !valid_view(a,z) || (binary && !valid_view(a,y)))return EINVAL;
  struct mesh_extent *out=&z.tensor->extents[z.extent];size_t elements=out->shape.rows*out->shape.columns;
  if(z.offset || z.rows>SIZE_MAX/z.columns || z.rows*z.columns!=elements || !((z.column_stride==1 && z.row_stride==z.columns) || (z.row_stride==1 && z.column_stride==z.rows)))return EINVAL;
  for(MeshFunction *existing in a.functions)if(existing->output.first>=out->first && existing->output.first<out->first+out->pages)return EINVAL;
  if(!binary)y=x;
  if(op==MESH_CONTRACT) {
    if(x.columns!=y.rows || z.rows!=x.rows || z.columns!=y.columns || z.column_stride!=1 || !x.row_stride || !x.column_stride || !y.row_stride || !y.column_stride || (x.column_stride!=1 && x.row_stride!=1) || (y.column_stride!=1 && y.row_stride!=1))return EINVAL;
    if(x.tensor->extents[x.extent].shape.scalar!=y.tensor->extents[y.extent].shape.scalar)return EINVAL;
  } else if(z.rows!=x.rows || z.columns!=(op==MESH_SUM?1:x.columns) || (binary && (x.rows!=y.rows || x.columns!=y.columns)))return EINVAL;
  if((x.tensor==z.tensor && x.extent==z.extent) || (binary && y.tensor==z.tensor && y.extent==z.extent))return EINVAL;
  size_t step=(size_t)out->quantum*a->context->M->pgsz/(out->shape.scalar==MESH_F16?2:4);
  for(size_t first=0;first<elements;first+=step){int error=bind_part(a,op,x,y,z,alpha,beta,first,MIN(step,elements-first));if(error)return error;}
  return 0;
}

/* design/algorithm-sources.md#backend-independent-producer-and-consumer-streaming */
struct mesh_tensor *mesh_algebra_contract(struct mesh_algebra *handle,const struct mesh_view *x,const struct mesh_view *y,size_t count,struct mesh_view z,float alpha) {
  MeshAlgebra *a=owner(handle);
  if(a.realized){errno=EBUSY;return NULL;}
  if(!count || count>UINT32_MAX || !x || !y || !valid_view(a,z)){errno=EINVAL;return NULL;}
  for(size_t i=0;i<count;i++)if(!valid_view(a,x[i]) || !valid_view(a,y[i]) || x[i].columns!=y[i].rows || x[i].rows!=z.rows || y[i].columns!=z.columns){errno=EINVAL;return NULL;}
  struct mesh_shape shape={z.rows,z.columns,MESH_F32};
  struct mesh_shape *shapes=malloc(count*sizeof *shapes);
  struct mesh_view *level=malloc(count*sizeof *level);
  if(!shapes || !level){free(shapes);free(level);errno=ENOMEM;return NULL;}
  for(size_t i=0;i<count;i++)shapes[i]=shape;
  struct mesh_tensor *partial=mesh_tensor_create(handle,shapes,count,1);free(shapes);
  int error=partial?0:errno;
  for(size_t i=0;i<count && !error;i++) {
    level[i]=mesh_tensor_view(partial,(uint32_t)i);
    error=mesh_algebra_bind(handle,MESH_CONTRACT,x[i],y[i],level[i],alpha,0);
  }
  size_t width=count;
  while(width>1 && !error) {
    size_t next=0;
    for(size_t i=0;i<width && !error;i+=2) {
      if(i+1==width){level[next++]=level[i];continue;}
      struct mesh_view out=z;
      if(width>2) {
        struct mesh_tensor *sum=mesh_tensor_create(handle,&shape,1,0);
        if(!sum){error=errno;break;}out=mesh_tensor_view(sum,0);
      }
      error=mesh_algebra_bind(handle,MESH_ADD,level[i],level[i+1],out,1,1);level[next++]=out;
    }
    width=next;
  }
  if(count==1 && !error)error=mesh_algebra_bind(handle,MESH_AFFINE,level[0],(struct mesh_view){0},z,1,0);
  free(level);if(error){errno=error;return NULL;}return partial;
}

/* design/algorithm-sources.md#streaming-algebra */
int mesh_algebra_transfer(struct mesh_algebra *handle,struct mesh_tensor *t,uint32_t i,uint32_t identity,uint16_t queue,int receive) {
  MeshAlgebra *a=owner(handle);
  if(a.realized)return EBUSY;
  if(!t || i>=t->count || t->context!=a->context)return EINVAL;
  struct mesh_row_map m=mesh_tensor_rows(t,i);
  struct mesh_row_binding b={.first=m.first,.count=m.count,.binding=identity,.queue=queue,.receive=!!receive};
  [a.bindings appendBytes:&b length:sizeof b];return 0;
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
        int error=mesh_algebra_bind(handle,MESH_AFFINE,mesh_tensor_view(source.tensor,source.first+(uint32_t)i*source.stride),(struct mesh_view){0},mesh_tensor_view(destination.tensor,destination.first+(uint32_t)i*destination.stride),1,0);
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
      struct mesh_row_binding b={.first=(receive?d:s)->first+offset,.count=block,.binding=identity,.queue=queue,.receive=receive};
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
/* design/algorithm-sources.md#mandatory-partial-publication */
int mesh_algebra_return_part(struct mesh_algebra *handle,struct mesh_tensor *t,uint32_t i,uint32_t part) {
  MeshAlgebra *a=owner(handle);
  if(a.realized)return EBUSY;
  if(!t || i>=t->count || t->context!=a->context)return EINVAL;
  struct mesh_extent *e=&t->extents[i];
  if(part>=e->pages/e->quantum)return EINVAL;
  struct mesh_row_map m={.first=e->first+part*e->quantum,.count=e->quantum};
  [a.returns appendBytes:&m length:sizeof m];return 0;
}
/* design/algorithm-sources.md#streaming-algebra */
int mesh_algebra_realize(struct mesh_algebra *handle) {
  MeshAlgebra *a=owner(handle);if(a.realized)return 0;size_t count=a.functions.count;
  struct mesh_row_function *functions=calloc(count?count:1,sizeof *functions);
  if(!functions)return ENOMEM;
  for(size_t i=0;i<count;i++)functions[i]=a.functions[i]->function;
  int error=mesh_realize(a->context,functions,count,a.bindings.mutableBytes,a.bindings.length/sizeof(struct mesh_row_binding),a.returns.mutableBytes,a.returns.length/sizeof(struct mesh_row_map));
  free(functions);if(!error)a.realized=YES;return error;
}

/* design/algorithm-sources.md#mandatory-partial-publication */
static void emit_part(MeshAlgebra *a,MeshFunction *f) {
  a->submitted++;
  f.execute(^(int64_t error,uint64_t nanoseconds){
    if(error)atomic_store(&a->code,error);
    else {uint32_t zero=0;mesh_complete(a->context,&f->function,&zero,1);}
    atomic_fetch_add(&a->gpuNanoseconds,nanoseconds);atomic_fetch_add(&a->completed,1);
  });
}
/* design/algorithm-sources.md#mandatory-partial-publication */
void mesh_algebra_scan(struct mesh_algebra *handle) { @autoreleasepool {
  MeshAlgebra *a=owner(handle);if(!a.realized)return;
  for(MeshFunction *f in a.functions){uint32_t index=0;if(mesh_issue(a->context,&f->function,&index,1))emit_part(a,f);}
} }
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
  MeshAlgebra *a=owner(handle);return (struct mesh_algebra_report){.submitted=a->submitted,.completed=atomic_load(&a->completed),.native_submitted=a->nativeSubmitted,.native_backings=atomic_load(&a->nativeBackings),.ne_planned_operations=a->nePlannedOperations,.code=atomic_load(&a->code),.gpu_seconds=atomic_load(&a->gpuNanoseconds)/1e9};
}
