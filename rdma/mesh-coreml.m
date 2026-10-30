/* Prepared Core ML bindings shared by the FX and native frontends.
   Compilation, model loading and named view binding are preparation operations.
   Run invokes the already-bound model; no copied-output substitute exists. */
#include "mesh-coreml.h"
#include "../etc/bridge/layout.h"
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

static NSMutableArray *models_;
static NSMutableDictionary *loaded_;
static NSMutableArray *placements_, *placement_text_;
static NSMutableArray *references_, *keys_;
static _Thread_local char error_[2048];

/* Prepared CPU numerical commands, on the same shared views as graph shares.
   Weights and scratch are bound by the compiler; execution allocates nothing. */
void mesh_cpu_product(const MeshCpuCommand *c) {
      cblas_sgemm(CblasRowMajor,c->input_transpose,c->weight_transpose,c->rows,c->columns,c->inner,
                  1,c->input,c->input_stride,c->weight,c->weight_stride,0,c->output,c->output_stride);
}
typedef struct {
  BNNSNDArrayDescriptor input, weight, output;
  void *workspace;
} MeshCpuProduct;

static void mesh_cpu_product_strided(const MeshCpuCommand *c) {
  MeshCpuProduct *p=(MeshCpuProduct *)c->scratch;
  if(BNNSMatMul(false,true,1,&p->input,&p->weight,&p->output,p->workspace,NULL)) {
    fputs("mesh: prepared CPU contraction failed\n",stderr);abort();
  }
}

/* Preparation chooses a whole-matrix instruction and binds its storage.
   Transposing the result swaps the operands: (X W^T)^T = W X^T. */
int64_t mesh_cpu_product_bind(MeshCpuCommand *c) {
  if(c->input_column<=0||c->input_stride<=0||c->weight_column<=0||c->weight_stride<=0||
     c->output_column<=0||c->output_stride<=0)return -1;
  if(((c->input_column==1&&c->input_stride>=c->inner)||(c->input_stride==1&&c->input_column>=c->rows)) &&
     ((c->weight_column==1&&c->weight_stride>=c->inner)||(c->weight_stride==1&&c->weight_column>=c->columns)) &&
     ((c->output_column==1&&c->output_stride>=c->columns)||(c->output_stride==1&&c->output_column>=c->rows))) {
    if(!(c->output_column==1&&c->output_stride>=c->columns)) {
      int n=c->rows;c->rows=c->columns;c->columns=n;
      float *p=c->input;c->input=c->weight;c->weight=p;
      n=c->input_stride;c->input_stride=c->weight_stride;c->weight_stride=n;
      n=c->input_column;c->input_column=c->weight_column;c->weight_column=n;
      c->output_stride=c->output_column;c->output_column=1;
    }
    c->input_transpose=c->input_column==1&&c->input_stride>=c->inner?CblasNoTrans:CblasTrans;
    c->weight_transpose=c->weight_column==1&&c->weight_stride>=c->inner?CblasTrans:CblasNoTrans;
    if(c->input_transpose==CblasTrans)c->input_stride=c->input_column;
    if(c->weight_transpose==CblasNoTrans)c->weight_stride=c->weight_column;
    c->run=mesh_cpu_product;
    return 0;
  }
  MeshCpuProduct *p=calloc(1,sizeof *p);if(!p)return -1;
  p->input=(BNNSNDArrayDescriptor){.layout=BNNSDataLayoutRowMajorMatrix,
    .size={c->inner,c->rows},.stride={c->input_column,c->input_stride},
    .data=c->input,.data_type=BNNSDataTypeFloat32,.data_scale=1};
  p->weight=(BNNSNDArrayDescriptor){.layout=BNNSDataLayoutRowMajorMatrix,
    .size={c->inner,c->columns},.stride={c->weight_column,c->weight_stride},
    .data=c->weight,.data_type=BNNSDataTypeFloat32,.data_scale=1};
  p->output=(BNNSNDArrayDescriptor){.layout=BNNSDataLayoutRowMajorMatrix,
    .size={c->columns,c->rows},.stride={c->output_column,c->output_stride},
    .data=c->output,.data_type=BNNSDataTypeFloat32,.data_scale=1};
  ssize_t bytes=BNNSMatMulWorkspaceSize(false,true,1,&p->input,&p->weight,&p->output,NULL);
  if(bytes<0){free(p);return -1;}
  c->scratch=(float *)p;c->run=mesh_cpu_product_strided;
  /* Even a zero-size requirement needs a non-null address to keep BNNS from
     allocating internally. One addressed byte satisfies that API contract. */
  return bytes ? bytes : sizeof(uint8_t);
}

void mesh_cpu_product_workspace(MeshCpuCommand *c, void *workspace) {
  MeshCpuProduct *p=(MeshCpuProduct *)c->scratch;p->workspace=workspace;
}

void mesh_cpu_command_release(MeshCpuCommand *c) {
  if(c->run==mesh_cpu_product_strided) {
    free(c->scratch);
  }
}
void mesh_cpu_gelu(const MeshCpuCommand *c) {
      const int n=c->rows*c->columns;
      for(int i=0;i<n;i++) {
        const float x=c->input[i];c->scratch[i]=0.7978845608028654f*(x+0.044715f*x*x*x);
      }
      vvtanhf(c->scratch,c->scratch,&n);
      for(int i=0;i<n;i++)c->output[i]=0.5f*c->input[i]*(1+c->scratch[i])*c->weight[i];
}
typedef struct {
  _Atomic unsigned references;
  size_t count;
  MeshCpuCommand commands[];
} MeshCpuProgram;

/* Preparation transfers command-owned workspace into this native lifetime.
   The schedule retains it independently of the frontend that constructed it. */
void *mesh_cpu_program_create(const MeshCpuCommand *commands, size_t count) {
  MeshCpuProgram *p=malloc(sizeof *p+count*sizeof *commands);
  if(!p)return NULL;
  atomic_init(&p->references,1);p->count=count;
  memcpy(p->commands,commands,count*sizeof *commands);
  return p;
}
void mesh_cpu_program_retain(void *program) {
  MeshCpuProgram *p=program;atomic_fetch_add_explicit(&p->references,1,memory_order_relaxed);
}
void mesh_cpu_program_release(void *program) {
  MeshCpuProgram *p=program;
  if(atomic_fetch_sub_explicit(&p->references,1,memory_order_acq_rel)!=1)return;
  for(size_t j=0;j<p->count;j++)mesh_cpu_command_release(p->commands+j);
  free(p);
}
const char *mesh_cpu_program_run(void *program) {
  MeshCpuProgram *p=program;
  for(size_t j=0;j<p->count;j++)p->commands[j].run(p->commands+j);
  return NULL;
}

const char *mesh_coreml_error(void) { return error_; }

// Walk nested blocks as well as the selected function's outer operations.
// This is the provider's anticipated placement; no prediction is submitted.
static void placement_block(MLComputePlan *plan,MLModelStructureProgramBlock *block,NSMutableDictionary *counts,NSMutableDictionary *unresolved)
    API_AVAILABLE(macos(14.4)) {
  for(MLModelStructureProgramOperation *op in block.operations) {
    MLComputePlanDeviceUsage *usage=[plan computeDeviceUsageForMLProgramOperation:op];
    if(usage) {
      id<MLComputeDeviceProtocol> device=usage.preferredComputeDevice;
      NSString *name=[device isKindOfClass:MLNeuralEngineComputeDevice.class]?@"ane":
        [device isKindOfClass:MLGPUComputeDevice.class]?@"gpu":[device isKindOfClass:MLCPUComputeDevice.class]?@"cpu":@"unknown";
      counts[name]=@([counts[name] unsignedLongValue]+1);
    } else unresolved[op.operatorName]=@([unresolved[op.operatorName] unsignedLongValue]+1);
    for(MLModelStructureProgramBlock *child in op.blocks)placement_block(plan,child,counts,unresolved);
  }
}
static NSDictionary *model_placement(NSURL *compiled,MLModelConfiguration *configuration) {
  if(@available(macOS 15.0,*)) {
    dispatch_semaphore_t done=dispatch_semaphore_create(0);
    __block MLComputePlan *plan=nil;__block NSError *failure=nil;
    [MLComputePlan loadContentsOfURL:compiled configuration:configuration completionHandler:^(MLComputePlan *loaded,NSError *error) {
      plan=loaded;failure=error;dispatch_semaphore_signal(done);
    }];
    if(dispatch_semaphore_wait(done,dispatch_time(DISPATCH_TIME_NOW,(int64_t)MESH_PREPARATION_NS))) {
      snprintf(error_,sizeof error_,"compute plan: preparation deadline of %.0f s",(double)MESH_PREPARATION_NS/1e9);return nil;
    }
    if(!plan){snprintf(error_,sizeof error_,"compute plan: %s",failure.localizedDescription.UTF8String);return nil;}
    MLModelStructureProgramFunction *function=plan.modelStructure.program.functions[configuration.functionName];
    if(!function){snprintf(error_,sizeof error_,"compute plan has no function %s",configuration.functionName.UTF8String);return nil;}
    NSMutableDictionary *counts=[NSMutableDictionary dictionary],*unresolved=[NSMutableDictionary dictionary];
    placement_block(plan,function.block,counts,unresolved);
    return @{@"preferred":counts,@"unresolved":unresolved};
  }
  snprintf(error_,sizeof error_,"named Core ML compute plans require macOS 15");return nil;
}
static BOOL placement_allowed(NSDictionary *placement,NSArray *devices) {
  NSDictionary *preferred=placement[@"preferred"];
  NSMutableSet *outside=[NSMutableSet setWithArray:preferred.allKeys];[outside minusSet:[NSSet setWithArray:devices]];
  if(!outside.count)return YES;
  snprintf(error_,sizeof error_,"Core ML preferred devices %s exceed declared devices %s",preferred.description.UTF8String,devices.description.UTF8String);
  return NO;
}
long mesh_coreml_load(const char *package,const char *configuration_json) {
  @autoreleasepool {
   NSDictionary *given=configuration_json?[NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:configuration_json length:strlen(configuration_json)] options:0 error:nil]:nil;
   if(![given isKindOfClass:NSDictionary.class]||![given[@"units"] isKindOfClass:NSNumber.class]||
      ![given[@"function"] isKindOfClass:NSString.class]||![given[@"devices"] isKindOfClass:NSArray.class]||![given[@"devices"] count]) {
     snprintf(error_,sizeof error_,"Core ML requires units, function and allowed devices");return -1;
   }
   NSArray *devices=given[@"devices"];NSString *function=given[@"function"];int units=[given[@"units"] intValue];
   for(id device in devices)if(![@[@"cpu",@"gpu",@"ane"] containsObject:device]) {
     snprintf(error_,sizeof error_,"unknown declared Core ML device %s",[device description].UTF8String);return -1;
   }
   const MLComputeUnits choices[]={MLComputeUnitsCPUAndNeuralEngine,MLComputeUnitsAll,MLComputeUnitsCPUOnly,MLComputeUnitsCPUAndGPU};
   if((unsigned)units>=sizeof choices/sizeof *choices){snprintf(error_,sizeof error_,"invalid compute units");return -1;}
   @synchronized([MLModel class]) {
    if (!models_) { models_=[NSMutableArray new];loaded_=[NSMutableDictionary new];placements_=[NSMutableArray new];placement_text_=[NSMutableArray new];references_=[NSMutableArray new];keys_=[NSMutableArray new]; }
    NSArray *key=@[@(units),function,@(package)];
    NSNumber *held=loaded_[key];
    if(held) {
      long h=held.longValue;if(!placement_allowed(placements_[h],devices))return -1;
      references_[h]=@([references_[h] unsignedLongValue]+1);return h;
    }
    NSError *error=nil;NSURL *compiled=[NSURL fileURLWithPath:@(package)];
    MLModelConfiguration *configuration=[MLModelConfiguration new];configuration.computeUnits=choices[units];
    if(@available(macOS 15.0,*))configuration.functionName=function;
    else {snprintf(error_,sizeof error_,"named Core ML functions require macOS 15");return -1;}
    NSDictionary *placement=model_placement(compiled,configuration);
    if(!placement||!placement_allowed(placement,devices))return -1;
    MLModel *model=[MLModel modelWithContentsOfURL:compiled configuration:configuration error:&error];
    if(!model){snprintf(error_,sizeof error_,"load: %s",error.localizedDescription.UTF8String);return -1;}
    NSString *text=[[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:placement options:NSJSONWritingSortedKeys error:nil] encoding:NSUTF8StringEncoding];
    NSUInteger handle=[models_ indexOfObject:NSNull.null];
    if(handle==NSNotFound){handle=models_.count;[models_ addObject:model];[placements_ addObject:placement];[placement_text_ addObject:text];[references_ addObject:@1];[keys_ addObject:key];}
    else{models_[handle]=model;placements_[handle]=placement;placement_text_[handle]=text;references_[handle]=@1;keys_[handle]=key;}
    loaded_[key]=@(handle);return (long)handle;
   }
  }
}

void mesh_coreml_release(long handle) {
  @synchronized([MLModel class]) {
    NSUInteger left=[references_[handle] unsignedLongValue]-1;references_[handle]=@(left);
    if(!left){[loaded_ removeObjectForKey:keys_[handle]];models_[handle]=[NSNull null];placements_[handle]=[NSNull null];placement_text_[handle]=[NSNull null];keys_[handle]=[NSNull null];}
  }
}

static MLMultiArray *wrap(const MeshModelView *view, NSError **error) {
  const MLMultiArrayDataType types[]={MLMultiArrayDataTypeFloat16,MLMultiArrayDataTypeFloat32,
                                      MLMultiArrayDataTypeInt32,MLMultiArrayDataTypeDouble};
  NSMutableArray *dims = [NSMutableArray array], *strides = [NSMutableArray array];
  for (int i = 0; i < view->rank; i++) { [dims addObject:@(view->shape[i])]; [strides addObject:@(view->strides[i])]; }
  return [[MLMultiArray alloc] initWithDataPointer:view->data shape:dims
                                       dataType:types[view->dtype] strides:strides
                                       deallocator:nil error:error];
}

@interface MeshPrediction : NSObject
@property long handle;
@property(retain) MLModel *model;
@property(retain) MLDictionaryFeatureProvider *features;
@property(retain) MLPredictionOptions *options;
@property(retain) NSDictionary<NSString *, MLMultiArray *> *outputs;
@end
@implementation MeshPrediction
- (void)dealloc { mesh_coreml_release(_handle); }
@end

void *mesh_coreml_bind(long handle, const MeshModelView *inputs, size_t ni, const MeshModelView *outputs, size_t no) {
  @autoreleasepool {
    MLModel *model;
    @synchronized([MLModel class]) {
      if (handle < 0 || handle >= (long)models_.count || models_[handle]==[NSNull null]) {
        snprintf(error_,sizeof error_,"no model %ld",handle);return NULL;
      }
      model=models_[handle];
    }
    NSError *error=nil;
    NSMutableDictionary *features=[NSMutableDictionary new], *backings=[NSMutableDictionary new];
    for (size_t i=0;i<ni+no;i++) {
      const MeshModelView *view=i<ni ? inputs+i : outputs+i-ni;
      if ((unsigned)view->dtype >= 4) { snprintf(error_,sizeof error_,"unsupported view dtype");return NULL; }
      if(!view->name||!view->name[0]){snprintf(error_,sizeof error_,"each model view requires its declared feature name");return NULL;}
      NSString *name=@(view->name);
      NSDictionary *descriptions=i<ni ? model.modelDescription.inputDescriptionsByName : model.modelDescription.outputDescriptionsByName;
      if(!descriptions[name]){snprintf(error_,sizeof error_,"model has no declared %s feature %s",i<ni?"input":"output",view->name);return NULL;}
      MLMultiArray *array=wrap(view,&error);
      if(!array){snprintf(error_,sizeof error_,"arrays: %s",error.localizedDescription.UTF8String);return NULL;}
      if(![descriptions[name] isAllowedValue:[MLFeatureValue featureValueWithMultiArray:array]]){
        snprintf(error_,sizeof error_,"view %s does not match its model feature declaration",view->name);return NULL;
      }
      if(i<ni)features[name]=[MLFeatureValue featureValueWithMultiArray:array];else backings[name]=array;
    }
    MeshPrediction *p=[MeshPrediction new];p.handle=handle;
    @synchronized([MLModel class]) { references_[handle]=@([references_[handle] unsignedLongValue]+1); }
    p.model=model;p.outputs=backings;
    p.features=[[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error];
    p.options=[MLPredictionOptions new];p.options.outputBackings=backings;
    if(!p.features){snprintf(error_,sizeof error_,"features: %s",error.localizedDescription.UTF8String);return NULL;}
    return (__bridge_retained void *)p;
  }
}

const char *mesh_coreml_run(void *binding) {
  @autoreleasepool {
    MeshPrediction *p=(__bridge MeshPrediction *)binding;NSError *error=nil;
    id<MLFeatureProvider> result=[p.model predictionFromFeatures:p.features options:p.options error:&error];
    if(!result){snprintf(error_,sizeof error_,"predict: %s",error.localizedDescription.UTF8String);return error_;}
    for (NSString *name in p.outputs) {
      if([result featureValueForName:name].multiArrayValue.dataPointer!=p.outputs[name].dataPointer){
        snprintf(error_,sizeof error_,"Core ML did not use the configured output backing");return error_;
      }
    }
    return NULL;
  }
}

void mesh_coreml_retain(void *binding) { CFRetain(binding); }
void mesh_coreml_unbind(void *binding) { CFRelease(binding); }

/* Placement is a property of the prepared model. Reading it never reloads a
   compute plan or waits for a provider callback. The handle owns the string. */
const char *mesh_coreml_placement(long handle) {
  @synchronized([MLModel class]) { return [placement_text_[handle] UTF8String]; }
}

/* A model package compiled (MLModel compileModelAtURL) into `destination` (a .mlmodelc, replaced where present), so a
   process loads it without compiling: 0, or -1. */
int mesh_coreml_compile(const char *package, const char *destination) {
  @autoreleasepool {
    NSError *error = nil;
    NSURL *compiled = [MLModel compileModelAtURL:[NSURL fileURLWithPath:@(package)] error:&error];
    if (!compiled) { snprintf(error_, sizeof error_, "compile: %s", error.localizedDescription.UTF8String); return -1; }
    NSURL *target = [NSURL fileURLWithPath:@(destination)];
    [[NSFileManager defaultManager] removeItemAtURL:target error:nil];
    if (![[NSFileManager defaultManager] moveItemAtURL:compiled toURL:target error:&error]) {
      snprintf(error_, sizeof error_, "move: %s", error.localizedDescription.UTF8String); return -1;
    }
    return 0;
  }
}
