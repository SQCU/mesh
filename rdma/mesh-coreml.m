/* Prepared Core ML bindings shared by the FX and native frontends.
   Compilation, model loading and named view binding are preparation operations.
   Run invokes the already-bound model; no copied-output substitute exists. */
#include "mesh-coreml.h"
#include "mesh-tensor-layout.h"
#include "../etc/bridge/layout.h"
#import <CoreML/CoreML.h>
#import <IOSurface/IOSurface.h>
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <errno.h>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

static NSMutableArray *models_;
static NSMutableDictionary *loaded_;
static NSMutableArray *placements_, *placement_text_;
static NSMutableArray *contracts_, *contract_text_;
static NSMutableArray *references_, *keys_, *model_resources_;
static _Thread_local char error_[2048];
extern char **environ;

/* Both frontends supply the same final source and explicit compiler command.
   This cache stores compilation artifacts, not bound numerical operands. */
@interface MeshCpuCompilation : NSObject
@property NSString *directory;
@property NSData *path;
@end
@implementation MeshCpuCompilation
- (void)dealloc {
  if(_directory)[[NSFileManager defaultManager] removeItemAtPath:_directory error:nil];
}
@end
static NSMutableDictionary<NSArray *,MeshCpuCompilation *> *cpu_artifacts_;
static void cpu_compilation_end(void) { @autoreleasepool { cpu_artifacts_=nil; } }
const char *mesh_cpu_compile(const char *source,const char *configuration_json) {
  @autoreleasepool { @try {
    NSString *text=source?[NSString stringWithUTF8String:source]:nil;
    id configuration=configuration_json?[NSJSONSerialization JSONObjectWithData:
      [NSData dataWithBytes:configuration_json length:strlen(configuration_json)] options:0 error:nil]:nil;
    if(!text||![configuration isKindOfClass:NSDictionary.class]||
       ![configuration[@"argv"] isKindOfClass:NSArray.class]||![configuration[@"argv"] count]||
       ![configuration[@"source_argument"] isKindOfClass:NSString.class]||
       ![configuration[@"output_argument"] isKindOfClass:NSString.class]||
       [configuration[@"source_argument"] isEqual:configuration[@"output_argument"]]) {
      snprintf(error_,sizeof error_,"CPU compilation needs source and explicit argv/source_argument/output_argument");return NULL;
    }
    NSArray *declared=configuration[@"argv"];NSUInteger sources=0,outputs=0;
    for(id value in declared) {
      if(![value isKindOfClass:NSString.class]){snprintf(error_,sizeof error_,"CPU compiler arguments must be strings");return NULL;}
      sources += [value isEqual:configuration[@"source_argument"]];outputs += [value isEqual:configuration[@"output_argument"]];
    }
    if(sources!=1||outputs!=1||[declared[0] isEqual:configuration[@"source_argument"]]||
       [declared[0] isEqual:configuration[@"output_argument"]]) {
      snprintf(error_,sizeof error_,"CPU compiler command needs one source token and one output token");return NULL;
    }
    @synchronized([MeshCpuCompilation class]) {
      if(!cpu_artifacts_){cpu_artifacts_=[NSMutableDictionary dictionary];atexit(cpu_compilation_end);}
      NSArray *key=@[text,configuration];MeshCpuCompilation *held=cpu_artifacts_[key];if(held)return held.path.bytes;
      NSString *pattern=[NSTemporaryDirectory() stringByAppendingPathComponent:@"mesh-cpu-XXXXXX"];
      char *directory=strdup(pattern.fileSystemRepresentation);
      if(!directory||!mkdtemp(directory)){free(directory);snprintf(error_,sizeof error_,"CPU artifact directory: %s",strerror(errno));return NULL;}
      NSString *root=[NSString stringWithUTF8String:directory];free(directory);
      MeshCpuCompilation *artifact=[MeshCpuCompilation new];artifact.directory=root;
      NSString *input=[root stringByAppendingPathComponent:@"kernel.cpp"],*output=[root stringByAppendingPathComponent:@"kernel.so"],
               *diagnostics=[root stringByAppendingPathComponent:@"compiler.log"];
      NSError *error=nil;
      if(![text writeToFile:input atomically:YES encoding:NSUTF8StringEncoding error:&error]) {
        snprintf(error_,sizeof error_,"CPU source preparation: %s",error.localizedDescription.UTF8String);return NULL;
      }
      NSMutableArray<NSString *> *arguments=[NSMutableArray array];
      for(NSString *value in declared)[arguments addObject:[value isEqual:configuration[@"source_argument"]]?input:
        [value isEqual:configuration[@"output_argument"]]?output:value];
      char **argv=calloc(arguments.count+1,sizeof *argv);
      if(!argv){snprintf(error_,sizeof error_,"CPU compiler argument allocation");return NULL;}
      for(NSUInteger i=0;i<arguments.count;i++)argv[i]=(char *)arguments[i].UTF8String;
      int fd=open(diagnostics.fileSystemRepresentation,O_WRONLY|O_CREAT|O_TRUNC,0600);
      if(fd<0){free(argv);snprintf(error_,sizeof error_,"CPU compiler diagnostics: %s",strerror(errno));return NULL;}
      posix_spawn_file_actions_t actions;int status=posix_spawn_file_actions_init(&actions);BOOL initialized=!status;
      if(!status)status=posix_spawn_file_actions_adddup2(&actions,fd,STDOUT_FILENO);
      if(!status)status=posix_spawn_file_actions_adddup2(&actions,fd,STDERR_FILENO);
      if(!status)status=posix_spawn_file_actions_addclose(&actions,fd);
      pid_t child=0;if(!status)status=posix_spawnp(&child,argv[0],&actions,NULL,argv,environ);
      if(initialized)posix_spawn_file_actions_destroy(&actions);close(fd);free(argv);
      if(status){snprintf(error_,sizeof error_,"CPU compiler launch: %s (%s)",strerror(status),diagnostics.fileSystemRepresentation);return NULL;}
      pid_t ended;do { ended=waitpid(child,&status,0); } while(ended<0&&errno==EINTR);
      if(ended<0){snprintf(error_,sizeof error_,"CPU compiler completion: %s",strerror(errno));return NULL;}
      if(!WIFEXITED(status)||WEXITSTATUS(status)||![[NSFileManager defaultManager] fileExistsAtPath:output]) {
        NSString *detail=[NSString stringWithContentsOfFile:diagnostics encoding:NSUTF8StringEncoding error:nil];
        snprintf(error_,sizeof error_,"CPU source preparation (%s): %s",diagnostics.fileSystemRepresentation,detail.UTF8String?:"compiler produced no library");return NULL;
      }
      const char *path=output.fileSystemRepresentation;
      artifact.path=[NSData dataWithBytes:path length:strlen(path)+1];
      unlink(input.fileSystemRepresentation);unlink(diagnostics.fileSystemRepresentation);
      cpu_artifacts_[key]=artifact;return artifact.path.bytes;
    }
  } @catch(NSException *failure) {snprintf(error_,sizeof error_,"CPU source preparation: %s",failure.reason.UTF8String);return NULL;} }
}

/* Explicit resource names are preparation operands. Live prepared programs
   share one identity; the final holder removes it from this registry. */
@interface MeshDeclaredResource : NSObject
@property NSData *key;
@property size_t capacity, holders;
@end
@implementation MeshDeclaredResource
@end
static NSMutableDictionary<NSData *, MeshDeclaredResource *> *host_resources_;
@interface MeshResourceBinding : NSObject { @public MeshHostResource *entries; size_t count; }
@property NSArray<MeshDeclaredResource *> *owners;
@end
@implementation MeshResourceBinding
- (void)dealloc {
  @synchronized([MeshResourceBinding class]) {
    for(MeshDeclaredResource *resource in _owners)
      if(!--resource.holders)[host_resources_ removeObjectForKey:resource.key];
  }
  free(entries);
}
@end

static MeshResourceBinding *host_resources(id declaration) {
  if(![declaration isKindOfClass:NSArray.class]) {
    snprintf(error_,sizeof error_,"implementation resources must be an explicit array");return nil;
  }
  NSMutableDictionary<NSData *, NSNumber *> *capacities=[NSMutableDictionary dictionary];
  NSMutableArray<NSData *> *keys=[NSMutableArray array];
  for(id row in declaration) {
    if(![row isKindOfClass:NSDictionary.class]||!row[@"identity"]||![row[@"capacity"] isKindOfClass:NSNumber.class]) {
      snprintf(error_,sizeof error_,"host resource needs an explicit JSON identity and positive integer capacity");return nil;
    }
    NSNumber *capacity=row[@"capacity"];
    if(CFGetTypeID((__bridge CFTypeRef)capacity)==CFBooleanGetTypeID()||
       [capacity compare:@0]!=NSOrderedDescending||[capacity compare:@(SIZE_MAX)]==NSOrderedDescending||
       [capacity compare:@(capacity.unsignedLongLongValue)]!=NSOrderedSame) {
      snprintf(error_,sizeof error_,"host resource capacity must be a positive integer");return nil;
    }
    NSData *key=[NSJSONSerialization dataWithJSONObject:row[@"identity"]
      options:NSJSONWritingSortedKeys|NSJSONWritingFragmentsAllowed|NSJSONWritingWithoutEscapingSlashes error:nil];
    if(!key){snprintf(error_,sizeof error_,"host resource identity must be JSON");return nil;}
    if(capacities[key]&&![capacities[key] isEqual:capacity]) {
      snprintf(error_,sizeof error_,"one host resource has conflicting declared capacities");return nil;
    }
    if(!capacities[key])[keys addObject:key];capacities[key]=capacity;
  }
  MeshResourceBinding *binding=[MeshResourceBinding new];
  binding->entries=calloc(MAX(keys.count,(NSUInteger)1),sizeof *binding->entries);
  if(!binding->entries){snprintf(error_,sizeof error_,"allocating declared host resources");return nil;}
  @synchronized([MeshResourceBinding class]) {
    if(!host_resources_)host_resources_=[NSMutableDictionary dictionary];
    for(NSData *key in keys)if(host_resources_[key]&&host_resources_[key].capacity!=capacities[key].unsignedLongLongValue) {
      snprintf(error_,sizeof error_,"live declarations give one host resource conflicting capacities");return nil;
    }
    NSMutableArray *owners=[NSMutableArray array];
    for(NSData *key in keys) {
      MeshDeclaredResource *resource=host_resources_[key];
      if(!resource) {
        resource=[MeshDeclaredResource new];resource.key=key;resource.capacity=capacities[key].unsignedLongLongValue;
        host_resources_[key]=resource;
      }
      resource.holders++;[owners addObject:resource];
      binding->entries[binding->count++]=(MeshHostResource){(__bridge const void *)resource,resource.capacity};
    }
    binding.owners=owners;
  }
  return binding;
}

/* Prepared CPU numerical commands, on the same shared views as graph shares.
   Weights and scratch are bound by the compiler; execution allocates nothing. */
void mesh_cpu_product(const void *arguments) {
      const MeshCpuCommand *c=arguments;
      cblas_sgemm(CblasRowMajor,c->input_transpose,c->weight_transpose,c->rows,c->columns,c->inner,
                  1,c->input,c->input_stride,c->weight,c->weight_stride,0,c->output,c->output_stride);
}
typedef struct {
  BNNSNDArrayDescriptor input, weight, output;
  void *workspace;
} MeshCpuProduct;

static void mesh_cpu_product_strided(const void *arguments) {
  const MeshCpuCommand *c=arguments;
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

void mesh_cpu_command_release(void *arguments) {
  MeshCpuCommand *c=arguments;
  if(c->run==mesh_cpu_product_strided) {
    free(c->scratch);
  }
}
void mesh_cpu_gelu(const void *arguments) {
      const MeshCpuCommand *c=arguments;
      const int n=c->rows*c->columns;
      for(int i=0;i<n;i++) {
        const float x=c->input[i];c->scratch[i]=0.7978845608028654f*(x+0.044715f*x*x*x);
      }
      vvtanhf(c->scratch,c->scratch,&n);
      for(int i=0;i<n;i++)c->output[i]=0.5f*c->input[i]*(1+c->scratch[i])*c->weight[i];
}
typedef struct { void (*run)(const void *); void *arguments; } MeshCpuBound;
typedef struct { void (*release_arguments)(void *); void *library; } MeshCpuOwnership;
typedef struct {
  _Atomic unsigned references;
  size_t count;
  void *resources; /* retained MeshResourceBinding */
  MeshCpuOwnership *owners;
  MeshCpuBound instructions[];
} MeshCpuProgram;

/* Preparation transfers command-owned workspace into this native lifetime.
   The schedule retains it independently of the frontend that constructed it. */
void *mesh_cpu_program_create(const MeshCpuInstruction *instructions, size_t count, const char *resources_json) {
  if(count&&!instructions){snprintf(error_,sizeof error_,"CPU program has no declared instructions");return NULL;}
  id declared=resources_json?[NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:resources_json length:strlen(resources_json)] options:0 error:nil]:nil;
  MeshResourceBinding *resources=host_resources(declared);if(!resources)return NULL;
  if(count>(SIZE_MAX-sizeof(MeshCpuProgram))/sizeof(MeshCpuBound)||count>SIZE_MAX/sizeof(MeshCpuOwnership)){
    snprintf(error_,sizeof error_,"CPU instruction extent overflows");return NULL;
  }
  MeshCpuProgram *p=calloc(1,sizeof *p+count*sizeof(MeshCpuBound));
  if(!p){snprintf(error_,sizeof error_,"allocating CPU program");return NULL;}
  p->owners=calloc(MAX(count,(size_t)1),sizeof *p->owners);
  if(!p->owners){free(p);snprintf(error_,sizeof error_,"allocating CPU instruction ownership");return NULL;}
  for(size_t i=0;i<count;i++) {
    const MeshCpuInstruction *source=instructions+i;MeshCpuBound *bound=p->instructions+i;
    if((source->argument_bytes&&!source->arguments)||
       (source->run?(source->library||source->symbol):(!source->library||!source->symbol))) {
      snprintf(error_,sizeof error_,"CPU instruction needs explicit arguments and one static entry or library/symbol");goto failed;
    }
    bound->run=source->run;
    if(source->library) {
      p->owners[i].library=dlopen(source->library,RTLD_NOW|RTLD_LOCAL);
      if(!p->owners[i].library){snprintf(error_,sizeof error_,"CPU instruction library: %s",dlerror());goto failed;}
      bound->run=(void(*)(const void *))dlsym(p->owners[i].library,source->symbol);
      if(!bound->run){snprintf(error_,sizeof error_,"CPU instruction symbol: %s",dlerror());goto failed;}
    }
    bound->arguments=malloc(MAX(source->argument_bytes,(size_t)1));
    if(!bound->arguments){snprintf(error_,sizeof error_,"copying CPU instruction arguments");goto failed;}
    if(source->argument_bytes)memcpy(bound->arguments,source->arguments,source->argument_bytes);
    p->owners[i].release_arguments=source->release_arguments;
  }
  p->resources=(__bridge_retained void *)resources;
  atomic_init(&p->references,1);p->count=count;
  return p;
failed:
  for(size_t i=0;i<count;i++) {
    free(p->instructions[i].arguments);
    if(p->owners[i].library)dlclose(p->owners[i].library);
  }
  free(p->owners);free(p);return NULL;
}
void mesh_cpu_program_retain(void *program) {
  MeshCpuProgram *p=program;atomic_fetch_add_explicit(&p->references,1,memory_order_relaxed);
}
void mesh_cpu_program_release(void *program) {
  MeshCpuProgram *p=program;
  if(atomic_fetch_sub_explicit(&p->references,1,memory_order_acq_rel)!=1)return;
  for(size_t j=0;j<p->count;j++) {
    if(p->owners[j].release_arguments)p->owners[j].release_arguments(p->instructions[j].arguments);
    free(p->instructions[j].arguments);
    if(p->owners[j].library)dlclose(p->owners[j].library);
  }
  CFRelease(p->resources);free(p->owners);free(p);
}
typedef struct { MeshCpuProgram *program; MeshHostCompletion complete; void *context; } MeshCpuTask;
static void cpu_submit(void *bound) {
  MeshCpuTask *task=bound;MeshCpuProgram *p=task->program;
  for(size_t j=0;j<p->count;j++)p->instructions[j].run(p->instructions[j].arguments);
  task->complete(task->context,NULL);
}
static void cpu_release(void *bound) {
  MeshCpuTask *task=bound;mesh_cpu_program_release(task->program);free(task);
}
const char *mesh_cpu_program_prepare(void *program,MeshHostCompletion complete,void *context,MeshHostTask *result) {
  MeshCpuTask *task=malloc(sizeof *task);
  if(!task)return "preparing CPU completion operand";
  mesh_cpu_program_retain(program);*task=(MeshCpuTask){program,complete,context};
  MeshResourceBinding *resources=(__bridge MeshResourceBinding *)task->program->resources;
  *result=(MeshHostTask){task,cpu_submit,cpu_release,resources->entries,resources->count};return NULL;
}

const char *mesh_coreml_error(void) { return error_; }

// Walk nested blocks as well as the selected function's outer operations.
// This is the provider's anticipated placement; no prediction is submitted.
static NSString *device_name(id<MLComputeDeviceProtocol> device) API_AVAILABLE(macos(14.0)) {
  return [device isKindOfClass:MLNeuralEngineComputeDevice.class]?@"ane":
    [device isKindOfClass:MLGPUComputeDevice.class]?@"gpu":[device isKindOfClass:MLCPUComputeDevice.class]?@"cpu":@"unknown";
}
static NSArray *placement_block(MLComputePlan *plan,MLModelStructureProgramBlock *block,
    NSMutableDictionary *preferred,NSMutableDictionary *supported,NSMutableDictionary *unresolved)
    API_AVAILABLE(macos(14.4)) {
  NSMutableArray *operations=[NSMutableArray array];
  for(MLModelStructureProgramOperation *op in block.operations) {
    MLComputePlanDeviceUsage *usage=[plan computeDeviceUsageForMLProgramOperation:op];
    NSMutableArray *eligible=[NSMutableArray array],*outputs=[NSMutableArray array],*blocks=[NSMutableArray array];
    NSString *name=nil;
    if(usage) {
      name=device_name(usage.preferredComputeDevice);
      preferred[name]=@([preferred[name] unsignedLongValue]+1);
      for(id<MLComputeDeviceProtocol> device in usage.supportedComputeDevices) {
        NSString *kind=device_name(device);
        if(![eligible containsObject:kind])[eligible addObject:kind];
      }
      for(NSString *kind in eligible)supported[kind]=@([supported[kind] unsignedLongValue]+1);
    } else unresolved[op.operatorName]=@([unresolved[op.operatorName] unsignedLongValue]+1);
    for(MLModelStructureProgramNamedValueType *output in op.outputs)[outputs addObject:output.name];
    for(MLModelStructureProgramBlock *child in op.blocks)
      [blocks addObject:placement_block(plan,child,preferred,supported,unresolved)];
    [operations addObject:@{@"operator":op.operatorName,@"outputs":outputs,@"preferred":name?:NSNull.null,
      @"supported":usage?eligible:NSNull.null,@"blocks":blocks}];
  }
  return operations;
}
static NSDictionary *model_placement(NSURL *compiled,MLModelConfiguration *configuration,NSArray *execution_devices) {
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
    NSMutableDictionary *preferred=[NSMutableDictionary dictionary],*supported=[NSMutableDictionary dictionary],
      *unresolved=[NSMutableDictionary dictionary];
    NSArray *operations=placement_block(plan,function.block,preferred,supported,unresolved);
    return @{@"allowed_devices":execution_devices,@"preferred":preferred,@"supported":supported,
      @"unresolved":unresolved,@"operations":operations};
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

/* Core ML emits the MIL artifact. An explicitly selected direct ANE model
   consumes that artifact and publishes its compiled storage requirements. */
typedef BOOL (*MeshANEAction)(id,SEL,id,id,unsigned,NSError **);
typedef BOOL (*MeshANEMap)(id,SEL,id,id,BOOL,NSError **);
typedef void (*MeshANEUnmap)(id,SEL,id,id);
typedef BOOL (*MeshANEEvaluate)(id,SEL,id,id,id,unsigned,NSError **);
@interface MeshANEModel : NSObject
@property id model, client, requestResource;
@property NSDictionary *options, *contract, *capabilities;
@property MeshResourceBinding *declaredResources;
@property NSData *resources;
@property unsigned qos;
@property size_t capacity;
@property BOOL loaded, cacheMapping;
@property MeshANEAction unload;
@property MeshANEMap map;
@property MeshANEUnmap unmap;
@property MeshANEEvaluate evaluate;
@end
@implementation MeshANEModel
- (void)dealloc {
  if(_loaded){NSError *error=nil;_unload(_client,@selector(unloadModel:options:qos:error:),_model,_options,_qos,&error);}
}
@end

static void *ane_framework_;
static id ane_symbol(NSString *name) {
  CFStringRef *symbol=dlsym(ane_framework_,name.UTF8String);
  if(!symbol){snprintf(error_,sizeof error_,"direct ANE capability has no symbol %s",name.UTF8String);return nil;}
  return (__bridge id)*symbol;
}
static BOOL ane_methods(id object,NSArray<NSString *> *names,NSDictionary *abi) {
  Class kind=object_isClass(object)?object:[object class];NSDictionary *declared=abi[NSStringFromClass(kind)];
  for(NSString *name in names) {
    Method method=object_isClass(object)?class_getClassMethod(kind,NSSelectorFromString(name)):class_getInstanceMethod(kind,NSSelectorFromString(name));
    NSString *encoding=method?@(method_getTypeEncoding(method)):nil;
    if(!encoding||![encoding isEqual:declared[name]]) {
      snprintf(error_,sizeof error_,"direct ANE %s.%s ABI %s differs from its declared capability %s",NSStringFromClass(kind).UTF8String,name.UTF8String,
        encoding.UTF8String?:"absent",[declared[name] UTF8String]?:"absent");return NO;
    }
  }
  return YES;
}
static const size_t model_element_[]={2,4,4,8,1};
static NSDictionary *ane_features(NSArray *live,NSArray *symbols,NSIndexSet *used,NSArray *alignment,
                                  NSDictionary *names,NSDictionary *capabilities) {
  NSMutableDictionary *result=[NSMutableDictionary dictionary],*bySymbol=[NSMutableDictionary dictionary];
  for(NSDictionary *feature in live)bySymbol[feature[@"Symbol"]]=feature;
  if(names.count!=live.count){snprintf(error_,sizeof error_,"direct ANE needs an explicit name for every compiled feature");return nil;}
  NSMutableSet *assigned=[NSMutableSet set];
  for(NSString *name in names) {
    NSString *symbol=names[name];NSDictionary *feature=bySymbol[symbol];NSUInteger index=[symbols indexOfObject:symbol];
    if(!feature||index==NSNotFound||![used containsIndex:index]||[assigned containsObject:symbol]) {
      snprintf(error_,sizeof error_,"direct ANE feature %s has no distinct compiled symbol %s",name.UTF8String,symbol.UTF8String);return nil;
    }
    [assigned addObject:symbol];
    NSNumber *dtype=capabilities[@"types"][feature[@"Type"]];
    if(!dtype||dtype.unsignedIntegerValue>=MeshModelTypeCount){snprintf(error_,sizeof error_,"direct ANE scalar type %s has no declared type mapping",[feature[@"Type"] UTF8String]);return nil;}
    const size_t element=model_element_[dtype.unsignedIntegerValue];
    for(NSString *key in capabilities[@"layout"])if(![feature[key] isEqual:capabilities[@"layout"][key]]) {
      snprintf(error_,sizeof error_,"compiled ANE %s=%s differs from the declared layout decoder %s",key.UTF8String,[feature[key] description].UTF8String,[capabilities[@"layout"][key] description].UTF8String);return nil;
    }
    NSArray *dimensions=@[@"Batches",@"Channels",@"Depth",@"Height",@"Width"],
      *steps=@[@"BatchStride",@"PlaneStride",@"DepthStride",@"RowStride"];
    NSMutableArray *shape=[NSMutableArray array],*strides=[NSMutableArray array];
    for(NSString *axis in dimensions) {
      NSNumber *extent=feature[axis];if(![extent isKindOfClass:NSNumber.class]||extent.longLongValue<=0){snprintf(error_,sizeof error_,"compiled ANE needs its %s extent",axis.UTF8String);return nil;}
      [shape addObject:extent];
    }
    for(NSString *axis in steps) {
      NSNumber *stride=feature[axis];if(![stride isKindOfClass:NSNumber.class]||stride.longLongValue<=0||stride.unsignedLongLongValue%element){snprintf(error_,sizeof error_,"compiled ANE needs a scalar-aligned %s",axis.UTF8String);return nil;}
      [strides addObject:@(stride.unsignedLongLongValue/element)];
    }
    [strides addObject:@1];
    if(![feature[@"PlaneCount"] isKindOfClass:NSNumber.class]){snprintf(error_,sizeof error_,"compiled ANE needs its stored plane count");return nil;}
    const uint64_t bytes=[feature[@"PlaneStride"] unsignedLongLongValue]*[feature[@"PlaneCount"] unsignedLongLongValue];
    MetalTensorLayout *layout=[MetalTensorLayout shape:shape strides:strides offset:0 maps:@{} indices:nil];
    struct MetalTensorSpan span=layout.storageSpan;
    if(span.offset<0||((uint64_t)span.offset+span.count)*element>bytes){snprintf(error_,sizeof error_,"compiled ANE layout exceeds its declared stored planes");return nil;}
    const uint64_t boundary=[alignment[index] boolValue]?[capabilities[@"alignment_bytes"] unsignedLongLongValue]:element;
    result[name]=@{@"symbol":symbol,@"index":@(index),@"dtype":dtype,@"element":@(element),@"shape":shape,
      @"strides":strides,@"bytes":@(bytes),@"alignment":@(boundary),@"provider":feature};
  }
  return result;
}
static MeshANEModel *ane_model(NSURL *compiled,NSDictionary *given,MeshResourceBinding *declaredResources) {
  NSDictionary *declared=given[@"ane"],*capabilities=declared[@"capabilities"];
  if(![declared[@"model_key"] isKindOfClass:NSString.class]||![declared[@"model_type"] isKindOfClass:NSString.class]||
     ![declared[@"qos"] isKindOfClass:NSNumber.class]||![declared[@"cache_mapping"] isKindOfClass:NSNumber.class]||![declared[@"options"] isKindOfClass:NSDictionary.class]||
     ![declared[@"symbols"] isKindOfClass:NSDictionary.class]||![capabilities[@"types"] isKindOfClass:NSDictionary.class]||
     ![capabilities[@"layout"] isKindOfClass:NSDictionary.class]||![capabilities[@"abi"] isKindOfClass:NSDictionary.class]||
     ![capabilities[@"alignment_bytes"] isKindOfClass:NSNumber.class]||![capabilities[@"alignment_bytes"] unsignedLongLongValue]||
     ![capabilities[@"completion_arguments"] isEqual:@[@"BOOL",@"NSError"]]) {
    snprintf(error_,sizeof error_,"direct ANE requires model key/type, qos, cache mapping, options, symbol names and storage capabilities");return nil;
  }
  if(![given[@"devices"] isEqual:@[@"ane"]]){snprintf(error_,sizeof error_,"direct ANE requires explicit ANE-only placement");return nil;}
  if(!ane_framework_)ane_framework_=dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_LOCAL|RTLD_LAZY);
  Class modelClass=NSClassFromString(@"_ANEModel"),clientClass=NSClassFromString(@"_ANEClient");
  if(!ane_framework_||!modelClass||!clientClass){snprintf(error_,sizeof error_,"direct ANE framework is unavailable");return nil;}
  if(!ane_methods(modelClass,@[@"modelAtURL:key:"],capabilities[@"abi"])||!ane_methods(clientClass,@[@"sharedConnection"],capabilities[@"abi"]))return nil;
  id model=((id(*)(id,SEL,id,id))objc_msgSend)(modelClass,@selector(modelAtURL:key:),compiled,declared[@"model_key"]);
  id client=((id(*)(id,SEL))objc_msgSend)(clientClass,@selector(sharedConnection));
  if(!model||!client){snprintf(error_,sizeof error_,"direct ANE model/client preparation failed");return nil;}
  if(!ane_methods(client,@[@"compileModel:options:qos:error:",@"loadModel:options:qos:error:",@"unloadModel:options:qos:error:",
      @"mapIOSurfacesWithModel:request:cacheInference:error:",@"unmapIOSurfacesWithModel:request:",@"evaluateWithModel:options:request:qos:error:"],capabilities[@"abi"])||
     !ane_methods(model,@[@"modelAttributes",@"inputSymbolIndicesForProcedureIndex:",@"outputSymbolIndicesForProcedureIndex:",@"queueDepth",@"program"],capabilities[@"abi"]))return nil;
  NSMutableDictionary *options=[NSMutableDictionary dictionary];
  for(NSString *key in declared[@"options"]){id symbol=ane_symbol(key);if(!symbol)return nil;options[symbol]=declared[@"options"][key];}
  id type=ane_symbol(declared[@"model_type"]),typeKey=ane_symbol(@"kANEFModelTypeKey");if(!type||!typeKey)return nil;options[typeKey]=type;
  MeshANEModel *prepared=[MeshANEModel new];prepared.client=client;prepared.model=model;prepared.options=options;prepared.capabilities=capabilities;prepared.qos=[declared[@"qos"] unsignedIntValue];prepared.cacheMapping=[declared[@"cache_mapping"] boolValue];
  prepared.unload=(MeshANEAction)[client methodForSelector:@selector(unloadModel:options:qos:error:)];
  prepared.map=(MeshANEMap)[client methodForSelector:@selector(mapIOSurfacesWithModel:request:cacheInference:error:)];
  prepared.unmap=(MeshANEUnmap)[client methodForSelector:@selector(unmapIOSurfacesWithModel:request:)];
  prepared.evaluate=(MeshANEEvaluate)[client methodForSelector:@selector(evaluateWithModel:options:request:qos:error:)];
  NSError *error=nil;
  MeshANEAction compile=(MeshANEAction)[client methodForSelector:@selector(compileModel:options:qos:error:)];
  if(!compile(client,@selector(compileModel:options:qos:error:),model,options,prepared.qos,&error)){
    snprintf(error_,sizeof error_,"direct ANE compile: %s",error.localizedDescription.UTF8String);return nil;
  }
  MeshANEAction load=(MeshANEAction)[client methodForSelector:@selector(loadModel:options:qos:error:)];
  if(!load(client,@selector(loadModel:options:qos:error:),model,options,prepared.qos,&error)){
    snprintf(error_,sizeof error_,"direct ANE load: %s",error.localizedDescription.UTF8String);return nil;
  }
  prepared.loaded=YES;
  prepared.requestResource=((id(*)(id,SEL))objc_msgSend)(model,@selector(program));
  const signed char capacity=((signed char(*)(id,SEL))objc_msgSend)(model,@selector(queueDepth));
  if(!prepared.requestResource||capacity<=0){snprintf(error_,sizeof error_,"compiled ANE model has no request resource/capacity");return nil;}
  prepared.capacity=(size_t)capacity;
  prepared.declaredResources=declaredResources;
  NSMutableData *resources=[NSMutableData dataWithBytes:prepared.declaredResources->entries
      length:prepared.declaredResources->count*sizeof(MeshHostResource)];
  const MeshHostResource request={(__bridge const void *)prepared.requestResource,prepared.capacity};
  [resources appendBytes:&request length:sizeof request];prepared.resources=resources;
  NSDictionary *attributes=((id(*)(id,SEL))objc_msgSend)(model,@selector(modelAttributes));
  id descriptionKey=ane_symbol(@"kANEFModelDescriptionKey"),procedureKey=ane_symbol(@"kANEFModelProcedureNameToIDMapKey");
  if(!descriptionKey||!procedureKey)return nil;
  NSDictionary *description=attributes[descriptionKey];NSNumber *procedure=description[procedureKey][given[@"function"]];
  if(!procedure){snprintf(error_,sizeof error_,"compiled ANE has no declared function %s",[given[@"function"] UTF8String]);return nil;}
  NSDictionary *network=nil;
  for(NSDictionary *candidate in attributes[@"NetworkStatusList"])if([candidate[@"Name"] isEqual:given[@"function"]]){network=candidate;break;}
  if(!network){snprintf(error_,sizeof error_,"compiled ANE has no storage schema for function %s",[given[@"function"] UTF8String]);return nil;}
  if([network[@"LiveStateList"] count]){snprintf(error_,sizeof error_,"ANE state must be functionalized into declared input/output pages before compilation");return nil;}
  NSMutableDictionary *contract=[@{@"backend":@"ane",@"output_storage":@"caller-pages",@"function":given[@"function"],@"procedure":procedure,@"request_capacity":@(prepared.capacity),
    @"resources":given[@"implementation"][@"resources"]} mutableCopy];
  NSArray *sides=@[@[@"inputs",@"LiveInputList",@"kANEFModelInputSymbolsArrayKey",@"kANEFModelInput16KAlignmentArrayKey",@"inputSymbolIndicesForProcedureIndex:"],
                   @[@"outputs",@"LiveOutputList",@"kANEFModelOutputSymbolsArrayKey",@"kANEFModelOutput16KAlignmentArrayKey",@"outputSymbolIndicesForProcedureIndex:"]];
  for(NSArray *side in sides) {
    id namesKey=ane_symbol(side[2]),alignmentKey=ane_symbol(side[3]);if(!namesKey||!alignmentKey)return nil;
    NSIndexSet *indices=((id(*)(id,SEL,unsigned))objc_msgSend)(model,NSSelectorFromString(side[4]),procedure.unsignedIntValue);
    NSDictionary *features=ane_features(network[side[1]],description[namesKey],indices,description[alignmentKey],declared[@"symbols"][side[0]],capabilities);
    if(!features)return nil;contract[side[0]]=features;
  }
  prepared.contract=contract;return prepared;
}
static NSDictionary *coreml_contract(MLModel *model,NSString *function) {
  if(@available(macOS 15.0,*))if(model.modelDescription.stateDescriptionsByName.count){snprintf(error_,sizeof error_,"Core ML state must be functionalized into declared input/output pages before compilation");return nil;}
  NSMutableDictionary *contract=[@{@"backend":@"coreml",@"output_storage":@"advisory",@"function":function} mutableCopy];
  for(NSArray *side in @[@[@"inputs",model.modelDescription.inputDescriptionsByName],@[@"outputs",model.modelDescription.outputDescriptionsByName]]) {
    NSMutableDictionary *features=[NSMutableDictionary dictionary];
    for(NSString *name in side[1]) {
      MLMultiArrayConstraint *type=[side[1][name] multiArrayConstraint];
      id dtype=NSNull.null,element=NSNull.null;
      const MLMultiArrayDataType types[]={MLMultiArrayDataTypeFloat16,MLMultiArrayDataTypeFloat32,MLMultiArrayDataTypeInt32,MLMultiArrayDataTypeDouble};
      for(NSUInteger i=0;i<sizeof types/sizeof *types;i++)if(type&&type.dataType==types[i]){dtype=@(i);element=@(model_element_[i]);}
      if(@available(macOS 26.0,*))if(type&&type.dataType==MLMultiArrayDataTypeInt8){dtype=@(MeshModelInt8);element=@(model_element_[MeshModelInt8]);}
      features[name]=@{@"shape":type.shape?:@[],@"strides":NSNull.null,@"bytes":NSNull.null,@"alignment":@1,@"dtype":dtype,@"element":element};
    }
    contract[side[0]]=features;
  }
  return contract;
}
long mesh_coreml_load(const char *package,const char *configuration_json) {
  @autoreleasepool { @try {
   NSDictionary *given=configuration_json?[NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:configuration_json length:strlen(configuration_json)] options:0 error:nil]:nil;
   NSString *backend=given[@"backend"];
   if(![given isKindOfClass:NSDictionary.class]||![@[@"coreml",@"ane"] containsObject:backend]||
      ![given[@"units"] isKindOfClass:NSNumber.class]||![given[@"function"] isKindOfClass:NSString.class]||
      ![given[@"devices"] isKindOfClass:NSArray.class]||![given[@"devices"] count]||![given[@"state_pairs"] isKindOfClass:NSDictionary.class]) {
     snprintf(error_,sizeof error_,"model requires explicit backend, units, function, devices and state_pairs");return -1;
   }
   NSArray *devices=given[@"devices"];NSString *function=given[@"function"];int units=[given[@"units"] intValue];
   const MLComputeUnits choices[]={MLComputeUnitsCPUAndNeuralEngine,MLComputeUnitsAll,MLComputeUnitsCPUOnly,MLComputeUnitsCPUAndGPU};
   NSArray *execution_devices=@[@[@"cpu",@"ane"],@[@"cpu",@"gpu",@"ane"],@[@"cpu"],@[@"cpu",@"gpu"]];
   if((unsigned)units>=sizeof choices/sizeof *choices){snprintf(error_,sizeof error_,"invalid compute units");return -1;}
   @synchronized([MLModel class]) {
    if(!models_){models_=[NSMutableArray new];loaded_=[NSMutableDictionary new];placements_=[NSMutableArray new];placement_text_=[NSMutableArray new];
      contracts_=[NSMutableArray new];contract_text_=[NSMutableArray new];references_=[NSMutableArray new];keys_=[NSMutableArray new];model_resources_=[NSMutableArray new];}
    NSArray *key=@[@(package),given];NSNumber *held=loaded_[key];
    if(held){long h=held.longValue;references_[h]=@([references_[h] unsignedLongValue]+1);return h;}
    MeshResourceBinding *resources=host_resources(given[@"implementation"][@"resources"]);if(!resources)return -1;
    NSError *error=nil;NSURL *compiled=[NSURL fileURLWithPath:@(package)];id model;NSDictionary *placement,*contract;
    if([backend isEqual:@"ane"]) {
      MeshANEModel *prepared=ane_model(compiled,given,resources);if(!prepared)return -1;model=prepared;contract=prepared.contract;
      placement=@{@"backend":@"ane",@"allowed_devices":@[@"ane"],@"preferred":@{},@"supported":@{},@"unresolved":@{},@"operations":@[]};
    } else {
      for(id device in devices)if(![@[@"cpu",@"gpu",@"ane"] containsObject:device]) {
        snprintf(error_,sizeof error_,"unknown declared Core ML device %s",[device description].UTF8String);return -1;
      }
      MLModelConfiguration *configuration=[MLModelConfiguration new];configuration.computeUnits=choices[units];
      if(@available(macOS 15.0,*))configuration.functionName=function;
      else{snprintf(error_,sizeof error_,"named Core ML functions require macOS 15");return -1;}
      placement=model_placement(compiled,configuration,execution_devices[units]);
      if(!placement||!placement_allowed(placement,devices))return -1;
      model=[MLModel modelWithContentsOfURL:compiled configuration:configuration error:&error];
      if(!model){snprintf(error_,sizeof error_,"load: %s",error.localizedDescription.UTF8String);return -1;}
      contract=coreml_contract(model,function);if(!contract)return -1;
    }
    for(NSString *name in given[@"state_pairs"]) {
      NSDictionary *pair=given[@"state_pairs"][name];
      if(![pair isKindOfClass:NSDictionary.class]||!contract[@"inputs"][pair[@"input"]]||!contract[@"outputs"][pair[@"output"]]) {
        snprintf(error_,sizeof error_,"functional state %s requires declared input/output feature names",name.UTF8String);return -1;
      }
    }
    NSMutableDictionary *stateful=[contract mutableCopy];stateful[@"state_pairs"]=given[@"state_pairs"];
    stateful[@"resources"]=given[@"implementation"][@"resources"];contract=stateful;
    if([model isKindOfClass:MeshANEModel.class])[(MeshANEModel *)model setContract:contract];
    NSString *text=[[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:placement options:NSJSONWritingSortedKeys error:nil] encoding:NSUTF8StringEncoding];
    NSString *schema=[[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:contract options:NSJSONWritingSortedKeys error:nil] encoding:NSUTF8StringEncoding];
    NSUInteger handle=[models_ indexOfObject:NSNull.null];
    if(handle==NSNotFound){handle=models_.count;[models_ addObject:model];[placements_ addObject:placement];[placement_text_ addObject:text];
      [contracts_ addObject:contract];[contract_text_ addObject:schema];[references_ addObject:@1];[keys_ addObject:key];[model_resources_ addObject:resources];}
    else{models_[handle]=model;placements_[handle]=placement;placement_text_[handle]=text;contracts_[handle]=contract;contract_text_[handle]=schema;references_[handle]=@1;keys_[handle]=key;model_resources_[handle]=resources;}
    loaded_[key]=@(handle);return (long)handle;
   }
  } @catch(NSException *failure){snprintf(error_,sizeof error_,"model preparation: %s",failure.reason.UTF8String);return -1;} }
}

void mesh_coreml_release(long handle) {
  @synchronized([MLModel class]) {
    NSUInteger left=[references_[handle] unsignedLongValue]-1;references_[handle]=@(left);
    if(!left){[loaded_ removeObjectForKey:keys_[handle]];models_[handle]=[NSNull null];placements_[handle]=[NSNull null];placement_text_[handle]=[NSNull null];contracts_[handle]=[NSNull null];contract_text_[handle]=[NSNull null];keys_[handle]=[NSNull null];model_resources_[handle]=[NSNull null];}
  }
}

@interface MeshModelInvocation : NSObject { @public const char *(*prepare)(id,MeshHostCompletion,void *,MeshHostTask *); }
@end
@implementation MeshModelInvocation
@end

static void model_task_release(void *context) { CFRelease(context); }

@interface MeshANEInvocation : MeshModelInvocation
@property long handle;
@property MeshANEModel *prepared;
@property NSArray *inputs,*outputs,*inputIndices,*outputIndices;
@property NSDictionary *surfaces;
@end
@implementation MeshANEInvocation
- (void)dealloc { mesh_coreml_release(_handle); }
@end
@interface MeshANETask : NSObject { @public
  id client, model, options, request;
  MeshANEEvaluate evaluate;
  unsigned qos;
  MeshHostCompletion complete;
  void *context;
}
@property MeshANEInvocation *source;
@property BOOL mapped;
@end
@implementation MeshANETask
- (void)dealloc {
  MeshANEModel *prepared=_source.prepared;
  if(_mapped)prepared.unmap(prepared.client,@selector(unmapIOSurfacesWithModel:request:),prepared.model,request);
}
@end
static void ane_submit(void *context) {
  MeshANETask *task=(__bridge MeshANETask *)context;NSError *error=nil;
  if(!task->evaluate(task->client,@selector(evaluateWithModel:options:request:qos:error:),task->model,task->options,task->request,task->qos,&error)) {
    task->complete(task->context,error.localizedDescription.UTF8String?:"direct ANE submission failed");
  }
}
static const char *ane_prepare(id object,MeshHostCompletion complete,void *context,MeshHostTask *result) {
  MeshANEInvocation *source=object;MeshANEModel *model=source.prepared;
  Class requestClass=NSClassFromString(@"_ANERequest");
  MeshANETask *task=[MeshANETask new];task.source=source;task->complete=complete;task->context=context;
  task->client=model.client;task->model=model.model;task->options=model.options;task->evaluate=model.evaluate;task->qos=model.qos;
  task->request=((id(*)(id,SEL,id,id,id,id,id))objc_msgSend)(requestClass,@selector(requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:),
    source.inputs,source.inputIndices,source.outputs,source.outputIndices,model.contract[@"procedure"]);
  if(!task->request)return "direct ANE request construction failed";
  if(!ane_methods(task->request,@[@"setCompletionHandler:"],model.capabilities[@"abi"]))return error_;
  void (^completed)(BOOL,NSError *)=^(BOOL success,NSError *error) {
    complete(context,success?NULL:(error.localizedDescription.UTF8String?:"direct ANE execution failed"));
  };
  ((void(*)(id,SEL,id))objc_msgSend)(task->request,@selector(setCompletionHandler:),completed);
  NSError *error=nil;task.mapped=model.map(model.client,@selector(mapIOSurfacesWithModel:request:cacheInference:error:),model.model,task->request,model.cacheMapping,&error);
  if(!task.mapped){snprintf(error_,sizeof error_,"direct ANE page mapping: %s",error.localizedDescription.UTF8String);return error_;}
  *result=(MeshHostTask){(__bridge_retained void *)task,ane_submit,model_task_release,model.resources.bytes,model.resources.length/sizeof(MeshHostResource)};return NULL;
}
static void *ane_bind(long handle,MeshANEModel *model,const MeshModelView *inputs,size_t ni,const MeshModelView *outputs,size_t no) {
  Class surfaceClass=NSClassFromString(@"_ANEIOSurfaceObject"),requestClass=NSClassFromString(@"_ANERequest");
  if(!ane_methods(surfaceClass,@[@"objectWithIOSurface:startOffset:"],model.capabilities[@"abi"])||
     !ane_methods(requestClass,@[@"requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:"],model.capabilities[@"abi"]))return NULL;
  CFStringRef *clientAddress=dlsym(RTLD_DEFAULT,"kIOSurfaceClientAddress");
  if(!clientAddress){snprintf(error_,sizeof error_,"IOSurface client-page wrapping is unavailable");return NULL;}
  NSMutableDictionary *surfaces=[NSMutableDictionary dictionary];
  NSMutableArray *in=[NSMutableArray array],*out=[NSMutableArray array],*inIndices=[NSMutableArray array],*outIndices=[NSMutableArray array];
  for(NSUInteger side=0;side<2;side++) {
    const MeshModelView *views=side?outputs:inputs;size_t count=side?no:ni;
    NSDictionary *features=model.contract[side?@"outputs":@"inputs"];NSMutableSet *names=[NSMutableSet set];
    if(count!=features.count){snprintf(error_,sizeof error_,"direct ANE requires every declared %s feature",side?"output":"input");return NULL;}
    for(size_t i=0;i<count;i++) {
      const MeshModelView *view=views+i;NSString *name=view->name?@(view->name):nil;NSDictionary *feature=name?features[name]:nil;
      if(!feature||[names containsObject:name]){snprintf(error_,sizeof error_,"direct ANE binding has no distinct declared feature");return NULL;}[names addObject:name];
      if(view->dtype!=[feature[@"dtype"] intValue]){snprintf(error_,sizeof error_,"direct ANE %s has a different scalar type",view->name);return NULL;}
      NSMutableArray *shape=[NSMutableArray array],*steps=[NSMutableArray array];
      for(int j=0;j<view->rank;j++){[shape addObject:@(view->shape[j])];[steps addObject:@(view->strides[j])];}
      MetalTensorLayout *expected=[[MetalTensorLayout shape:feature[@"shape"] strides:feature[@"strides"] offset:0 maps:@{} indices:nil] reshape:shape];
      MetalTensorLayout *actual=[MetalTensorLayout shape:shape strides:steps offset:0 maps:@{} indices:nil];
      if(![actual equivalent:expected]){snprintf(error_,sizeof error_,"direct ANE %s needs its prepared provider layout",view->name);return NULL;}
      const MeshModelPages pages=view->pages;const size_t bytes=[feature[@"bytes"] unsignedLongLongValue],alignment=[feature[@"alignment"] unsignedLongLongValue];
      if(!pages.base||pages.offset>pages.bytes||bytes>pages.bytes-pages.offset||((uintptr_t)pages.base+pages.offset)%alignment){
        snprintf(error_,sizeof error_,"direct ANE %s needs its declared page footprint/alignment",view->name);return NULL;
      }
      NSArray *key=@[@((uintptr_t)pages.base),@(pages.bytes)];id surface=surfaces[key];
      if(!surface) {
        surface=CFBridgingRelease(IOSurfaceCreate((__bridge CFDictionaryRef)@{(__bridge id)*clientAddress:@((uintptr_t)pages.base),(__bridge id)kIOSurfaceAllocSize:@(pages.bytes)}));
        if(!surface||IOSurfaceGetBaseAddress((__bridge IOSurfaceRef)surface)!=pages.base||IOSurfaceGetAllocSize((__bridge IOSurfaceRef)surface)!=pages.bytes){
          snprintf(error_,sizeof error_,"IOSurface did not bind the declared bridge pages");return NULL;
        }
        surfaces[key]=surface;
      }
      id operand=((id(*)(id,SEL,IOSurfaceRef,id))objc_msgSend)(surfaceClass,@selector(objectWithIOSurface:startOffset:),(__bridge IOSurfaceRef)surface,@(pages.offset));
      [(side?out:in) addObject:operand];[(side?outIndices:inIndices) addObject:feature[@"index"]];
    }
  }
  MeshANEInvocation *p=[MeshANEInvocation new];p->prepare=ane_prepare;p.handle=handle;p.prepared=model;
  p.inputs=in;p.outputs=out;p.inputIndices=inIndices;p.outputIndices=outIndices;p.surfaces=surfaces;
  @synchronized([MLModel class]){references_[handle]=@([references_[handle] unsignedLongValue]+1);}
  return (__bridge_retained void *)p;
}
void *mesh_coreml_bind(long handle,const MeshModelView *inputs,size_t ni,const MeshModelView *outputs,size_t no) {
  @autoreleasepool { @try {
    id model;
    @synchronized([MLModel class]) {
      if(handle<0||handle>=(long)models_.count||models_[handle]==NSNull.null){snprintf(error_,sizeof error_,"no model %ld",handle);return NULL;}
      model=models_[handle];
    }
    if(![contracts_[handle][@"output_storage"] isEqual:@"caller-pages"]) {
      snprintf(error_,sizeof error_,"prepared page execution requires guaranteed caller-owned output pages; public Core ML outputBackings are advisory");return NULL;
    }
    return ane_bind(handle,model,inputs,ni,outputs,no);
  } @catch(NSException *failure){snprintf(error_,sizeof error_,"model binding: %s",failure.reason.UTF8String);return NULL;} }
}
const char *mesh_coreml_prepare(void *binding,MeshHostCompletion complete,void *context,MeshHostTask *task) {
  MeshModelInvocation *p=(__bridge MeshModelInvocation *)binding;return p->prepare(p,complete,context,task);
}
const char *mesh_coreml_contract(long handle) {
  @synchronized([MLModel class]){return [contract_text_[handle] UTF8String];}
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
