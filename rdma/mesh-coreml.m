/* mesh-coreml.m: a Core ML model's predictions on caller memory, for torch-mesh's Neural Engine engine
   (torch-mesh/torch_mesh/engines.py Ane).  A prediction's input is the caller's memory as a float16 MLMultiArray (no
   copy: on unified memory, torch's MPS tensors' shared storage), and its output is written into the caller's memory
   (Core ML's output backing), so the CPU does no copying on the prediction's critical path: through coremltools each
   prediction copied its input and output on the CPU, and once the CPU's performance cores clocked down beside a busy
   GPU (1.3-2.8 GHz from 4.3 on an M4 Pro), a 3 ms prediction took 9. Preparation binds native entry points and
   their operands into the shared schedule; execution does not enter the interpreter.

   mesh_coreml_load(package, input, output, units): compiles a model package (MLModel compileModelAtURL; a compiled
   .mlmodelc loads as it is) and loads it on `units` (0 CPU and Neural Engine, 1 all, 2 CPU only), its input and output
   named (empty: the model's own first); its handle, or -1 (mesh_coreml_error).
   mesh_coreml_bind fixes the input and output views before execution.
   mesh_coreml_run executes that binding; an incompatible output backing fails the call.
   There is no copied-output execution substitute. */
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

static NSMutableArray *models_;
static NSMutableArray *names_;
static NSMutableArray *compiled_;
static NSMutableArray *units_;
static _Thread_local char error_[2048];

/* Prepared CPU numerical commands, on the same shared views as graph shares.
   Weights and scratch are bound by the compiler; execution allocates nothing. */
typedef struct MeshCpuCommand {
  void (*run)(const struct MeshCpuCommand *);
  int rows, columns, inner, input_stride, weight_stride, output_stride;
  int input_column, weight_column, output_column;
  float *input, *weight, *output, *scratch;
  int input_transpose, weight_transpose;
} MeshCpuCommand;

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
int mesh_cpu_product_bind(MeshCpuCommand *c) {
  if(c->input_column<=0||c->input_stride<=0||c->weight_column<=0||c->weight_stride<=0||
     c->output_column<=0||c->output_stride<=0)return -1;
  if((c->input_column==1||c->input_stride==1) &&
     (c->weight_column==1||c->weight_stride==1) &&
     (c->output_column==1||c->output_stride==1)) {
    if(c->output_column!=1) {
      int n=c->rows;c->rows=c->columns;c->columns=n;
      float *p=c->input;c->input=c->weight;c->weight=p;
      n=c->input_stride;c->input_stride=c->weight_stride;c->weight_stride=n;
      n=c->input_column;c->input_column=c->weight_column;c->weight_column=n;
      c->output_stride=c->output_column;c->output_column=1;
    }
    c->input_transpose=c->input_column==1?CblasNoTrans:CblasTrans;
    c->weight_transpose=c->weight_column==1?CblasTrans:CblasNoTrans;
    if(c->input_column!=1)c->input_stride=c->input_column;
    if(c->weight_column!=1)c->weight_stride=c->weight_column;
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
  p->workspace=malloc(bytes?bytes:1);
  if(!p->workspace){free(p);return -1;}
  c->scratch=(float *)p;c->run=mesh_cpu_product_strided;
  return 0;
}

void mesh_cpu_command_release(MeshCpuCommand *c) {
  if(c->run==mesh_cpu_product_strided) {
    MeshCpuProduct *p=(MeshCpuProduct *)c->scratch;free(p->workspace);free(p);
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

long mesh_coreml_load(const char *package, const char *input, const char *output, int units) {
  @autoreleasepool {
    if (!models_) { models_ = [NSMutableArray new]; names_ = [NSMutableArray new]; compiled_ = [NSMutableArray new]; units_ = [NSMutableArray new]; }
    NSError *error = nil;
    NSURL *given = [NSURL fileURLWithPath:@(package)];
    NSURL *compiled = [given.pathExtension isEqualToString:@"mlmodelc"] ? given : [MLModel compileModelAtURL:given error:&error];
    if (!compiled) { snprintf(error_, sizeof error_, "compile: %s", error.localizedDescription.UTF8String); return -1; }
    MLModelConfiguration *configuration = [MLModelConfiguration new];
    configuration.computeUnits = units == 1 ? MLComputeUnitsAll : units == 2 ? MLComputeUnitsCPUOnly : MLComputeUnitsCPUAndNeuralEngine;
    MLModel *model = [MLModel modelWithContentsOfURL:compiled configuration:configuration error:&error];
    if (!model) { snprintf(error_, sizeof error_, "load: %s", error.localizedDescription.UTF8String); return -1; }
    NSString *in = strlen(input) ? @(input) : model.modelDescription.inputDescriptionsByName.allKeys.firstObject;
    NSString *out = strlen(output) ? @(output) : model.modelDescription.outputDescriptionsByName.allKeys.firstObject;
    if (!in || !out) { snprintf(error_, sizeof error_, "load: the model names no input or output"); return -1; }
    [models_ addObject:model];
    [names_ addObject:@[in, out]];
    [compiled_ addObject:compiled];
    [units_ addObject:@(configuration.computeUnits)];
    return (long)models_.count - 1;
  }
}

typedef struct {
  const char *name;
  void *data;
  const int64_t *shape, *strides;
  int rank, dtype;
} MeshModelView;

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
@property(retain) MLModel *model;
@property(retain) MLDictionaryFeatureProvider *features;
@property(retain) MLPredictionOptions *options;
@property(retain) NSDictionary<NSString *, MLMultiArray *> *outputs;
@end
@implementation MeshPrediction
@end

void *mesh_coreml_bind(long handle, const MeshModelView *inputs, size_t ni, const MeshModelView *outputs, size_t no) {
  @autoreleasepool {
    if (handle < 0 || handle >= (long)models_.count) { snprintf(error_, sizeof error_, "no model %ld", handle); return NULL; }
    NSArray *names=names_[handle];NSError *error=nil;
    NSMutableDictionary *features=[NSMutableDictionary new], *backings=[NSMutableDictionary new];
    for (size_t i=0;i<ni+no;i++) {
      const MeshModelView *view=i<ni ? inputs+i : outputs+i-ni;
      if ((unsigned)view->dtype >= 4) { snprintf(error_,sizeof error_,"unsupported view dtype");return NULL; }
      NSString *name=strlen(view->name) ? @(view->name) : names[i<ni ? 0 : 1];
      MLMultiArray *array=wrap(view,&error);
      if(!array){snprintf(error_,sizeof error_,"arrays: %s",error.localizedDescription.UTF8String);return NULL;}
      if(i<ni)features[name]=[MLFeatureValue featureValueWithMultiArray:array];else backings[name]=array;
    }
    MeshPrediction *p=[MeshPrediction new];p.model=models_[handle];p.outputs=backings;
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

/* Where Core ML places each operation of a loaded model (MLComputePlan): "operator=device" a line, the device cpu, gpu,
   ane or unknown, written to `out` (at most `size` bytes); the operations' count, or -1.  Placement is Core ML's own
   choice within the compute units the model was loaded on: an operation on the CPU shares the performance cores with
   anything else running there. */
long mesh_coreml_placement(long handle, char *out, size_t size) {
  @autoreleasepool {
    if (handle < 0 || handle >= (long)models_.count) { snprintf(error_, sizeof error_, "no model %ld", handle); return -1; }
    if (@available(macOS 14.4, *)) {
      MLModelConfiguration *configuration = [MLModelConfiguration new];
      configuration.computeUnits = (MLComputeUnits)[units_[handle] integerValue];
      dispatch_semaphore_t done = dispatch_semaphore_create(0);
      __block MLComputePlan *plan = nil;
      __block NSError *failure = nil;
      [MLComputePlan loadContentsOfURL:compiled_[handle] configuration:configuration
                     completionHandler:^(MLComputePlan *loaded, NSError *error) { plan = loaded; failure = error; dispatch_semaphore_signal(done); }];
      dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
      if (!plan) { snprintf(error_, sizeof error_, "compute plan: %s", failure.localizedDescription.UTF8String); return -1; }
      MLModelStructureProgramFunction *main = plan.modelStructure.program.functions[@"main"];
      NSMutableString *text = [NSMutableString string];
      long count = 0;
      for (MLModelStructureProgramOperation *op in main.block.operations) {
        id<MLComputeDeviceProtocol> device = [plan computeDeviceUsageForMLProgramOperation:op].preferredComputeDevice;
        NSString *where = [device isKindOfClass:[MLNeuralEngineComputeDevice class]] ? @"ane"
                        : [device isKindOfClass:[MLGPUComputeDevice class]] ? @"gpu"
                        : [device isKindOfClass:[MLCPUComputeDevice class]] ? @"cpu" : @"unknown";
        [text appendFormat:@"%@=%@\n", op.operatorName, where];
        count++;
      }
      snprintf(out, size, "%s", text.UTF8String);
      return count;
    }
    snprintf(error_, sizeof error_, "MLComputePlan needs macOS 14.4");
    return -1;
  }
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
