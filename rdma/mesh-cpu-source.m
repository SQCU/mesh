/* Shared preparation compiler and page-bound CPU programs. */
#include "mesh-cpu-source.h"
#import <Accelerate/Accelerate.h>
#include <dlfcn.h>
#include <stdatomic.h>
#include <errno.h>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
_Thread_local char mesh_host_error_text[2048];
extern char **environ;
const char *mesh_cpu_error(void) { return mesh_host_error_text; }

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
      snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"CPU compilation needs source and explicit argv/source_argument/output_argument");return NULL;
    }
    NSArray *declared=configuration[@"argv"];NSUInteger sources=0,outputs=0;
    for(id value in declared) {
      if(![value isKindOfClass:NSString.class]){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"CPU compiler arguments must be strings");return NULL;}
      sources += [value isEqual:configuration[@"source_argument"]];outputs += [value isEqual:configuration[@"output_argument"]];
    }
    if(sources!=1||outputs!=1||[declared[0] isEqual:configuration[@"source_argument"]]||
       [declared[0] isEqual:configuration[@"output_argument"]]) {
      snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"CPU compiler command needs one source token and one output token");return NULL;
    }
    @synchronized([MeshCpuCompilation class]) {
      if(!cpu_artifacts_){cpu_artifacts_=[NSMutableDictionary dictionary];atexit(cpu_compilation_end);}
      NSArray *key=@[text,configuration];MeshCpuCompilation *held=cpu_artifacts_[key];if(held)return held.path.bytes;
      NSString *pattern=[NSTemporaryDirectory() stringByAppendingPathComponent:@"mesh-cpu-XXXXXX"];
      char *directory=strdup(pattern.fileSystemRepresentation);
      if(!directory||!mkdtemp(directory)){free(directory);snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"CPU artifact directory: %s",strerror(errno));return NULL;}
      NSString *root=[NSString stringWithUTF8String:directory];free(directory);
      MeshCpuCompilation *artifact=[MeshCpuCompilation new];artifact.directory=root;
      NSString *input=[root stringByAppendingPathComponent:@"kernel.cpp"],*output=[root stringByAppendingPathComponent:@"kernel.so"],
               *diagnostics=[root stringByAppendingPathComponent:@"compiler.log"];
      NSError *error=nil;
      if(![text writeToFile:input atomically:YES encoding:NSUTF8StringEncoding error:&error]) {
        snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"CPU source preparation: %s",error.localizedDescription.UTF8String);return NULL;
      }
      NSMutableArray<NSString *> *arguments=[NSMutableArray array];
      for(NSString *value in declared)[arguments addObject:[value isEqual:configuration[@"source_argument"]]?input:
        [value isEqual:configuration[@"output_argument"]]?output:value];
      char **argv=calloc(arguments.count+1,sizeof *argv);
      if(!argv){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"CPU compiler argument allocation");return NULL;}
      for(NSUInteger i=0;i<arguments.count;i++)argv[i]=(char *)arguments[i].UTF8String;
      int fd=open(diagnostics.fileSystemRepresentation,O_WRONLY|O_CREAT|O_TRUNC,0600);
      if(fd<0){free(argv);snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"CPU compiler diagnostics: %s",strerror(errno));return NULL;}
      posix_spawn_file_actions_t actions;int status=posix_spawn_file_actions_init(&actions);BOOL initialized=!status;
      if(!status)status=posix_spawn_file_actions_adddup2(&actions,fd,STDOUT_FILENO);
      if(!status)status=posix_spawn_file_actions_adddup2(&actions,fd,STDERR_FILENO);
      if(!status)status=posix_spawn_file_actions_addclose(&actions,fd);
      pid_t child=0;if(!status)status=posix_spawnp(&child,argv[0],&actions,NULL,argv,environ);
      if(initialized)posix_spawn_file_actions_destroy(&actions);close(fd);free(argv);
      if(status){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"CPU compiler launch: %s (%s)",strerror(status),diagnostics.fileSystemRepresentation);return NULL;}
      pid_t ended;do { ended=waitpid(child,&status,0); } while(ended<0&&errno==EINTR);
      if(ended<0){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"CPU compiler completion: %s",strerror(errno));return NULL;}
      if(!WIFEXITED(status)||WEXITSTATUS(status)||![[NSFileManager defaultManager] fileExistsAtPath:output]) {
        NSString *detail=[NSString stringWithContentsOfFile:diagnostics encoding:NSUTF8StringEncoding error:nil];
        snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"CPU source preparation (%s): %s",diagnostics.fileSystemRepresentation,detail.UTF8String?:"compiler produced no library");return NULL;
      }
      const char *path=output.fileSystemRepresentation;
      artifact.path=[NSData dataWithBytes:path length:strlen(path)+1];
      unlink(input.fileSystemRepresentation);unlink(diagnostics.fileSystemRepresentation);
      cpu_artifacts_[key]=artifact;return artifact.path.bytes;
    }
  } @catch(NSException *failure) {snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"CPU source preparation: %s",failure.reason.UTF8String);return NULL;} }
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
@implementation MeshResourceBinding
- (void)dealloc {
  @synchronized([MeshResourceBinding class]) {
    for(MeshDeclaredResource *resource in _owners)
      if(!--resource.holders)[host_resources_ removeObjectForKey:resource.key];
  }
  free(entries);
}
@end

MeshResourceBinding *mesh_host_resources(id declaration) {
  if(![declaration isKindOfClass:NSArray.class]) {
    snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"implementation resources must be an explicit array");return nil;
  }
  NSMutableDictionary<NSData *, NSNumber *> *capacities=[NSMutableDictionary dictionary];
  NSMutableArray<NSData *> *keys=[NSMutableArray array];
  for(id row in declaration) {
    if(![row isKindOfClass:NSDictionary.class]||!row[@"identity"]||![row[@"capacity"] isKindOfClass:NSNumber.class]) {
      snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"host resource needs an explicit JSON identity and positive integer capacity");return nil;
    }
    NSNumber *capacity=row[@"capacity"];
    if(CFGetTypeID((__bridge CFTypeRef)capacity)==CFBooleanGetTypeID()||
       [capacity compare:@0]!=NSOrderedDescending||[capacity compare:@(SIZE_MAX)]==NSOrderedDescending||
       [capacity compare:@(capacity.unsignedLongLongValue)]!=NSOrderedSame) {
      snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"host resource capacity must be a positive integer");return nil;
    }
    NSData *key=[NSJSONSerialization dataWithJSONObject:row[@"identity"]
      options:NSJSONWritingSortedKeys|NSJSONWritingFragmentsAllowed|NSJSONWritingWithoutEscapingSlashes error:nil];
    if(!key){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"host resource identity must be JSON");return nil;}
    if(capacities[key]&&![capacities[key] isEqual:capacity]) {
      snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"one host resource has conflicting declared capacities");return nil;
    }
    if(!capacities[key])[keys addObject:key];capacities[key]=capacity;
  }
  MeshResourceBinding *binding=[MeshResourceBinding new];
  binding->entries=calloc(MAX(keys.count,(NSUInteger)1),sizeof *binding->entries);
  if(!binding->entries){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"allocating declared host resources");return nil;}
  @synchronized([MeshResourceBinding class]) {
    if(!host_resources_)host_resources_=[NSMutableDictionary dictionary];
    for(NSData *key in keys)if(host_resources_[key]&&host_resources_[key].capacity!=capacities[key].unsignedLongLongValue) {
      snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"live declarations give one host resource conflicting capacities");return nil;
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
#include "mesh-cpu-kernels.h"
void mesh_cpu_product(const void *arguments) { mesh_cpu_product_body(arguments); }
static void mesh_cpu_product_strided(const void *arguments) { mesh_cpu_product_strided_body(arguments); }

/* Preparation chooses a whole-matrix instruction and binds its storage.
   Transposing the result swaps the operands: (X W^T)^T = W X^T. */
#define CPU_ENTRY(name) "#include \"mesh-cpu-kernels.h\"\n" \
  "struct MeshCpuArguments {MeshCpuCommand command;};\n" \
  "extern \"C\" void mesh_cpu_entry(const void *arguments) {\n" \
  "const auto &a=*static_cast<const MeshCpuArguments *>(arguments);\n" #name "(&a.command);\n}\n"
enum { MeshCpuDenseProduct,MeshCpuStridedProduct,MeshCpuGelu };
MeshCpuDeclaration mesh_cpu_product_declare(const MeshCpuCommand *c) {
  BOOL dense=((c->input_column==1&&c->input_stride>=c->inner)||(c->input_stride==1&&c->input_column>=c->rows)) &&
     ((c->weight_column==1&&c->weight_stride>=c->inner)||(c->weight_stride==1&&c->weight_column>=c->columns)) &&
     ((c->output_column==1&&c->output_stride>=c->columns)||(c->output_stride==1&&c->output_column>=c->rows));
  return (MeshCpuDeclaration){dense?MeshCpuDenseProduct:MeshCpuStridedProduct,
    dense?CPU_ENTRY(mesh_cpu_product_body):CPU_ENTRY(mesh_cpu_product_strided_body)};
}
MeshCpuDeclaration mesh_cpu_gelu_declare(void) {return (MeshCpuDeclaration){MeshCpuGelu,CPU_ENTRY(mesh_cpu_gelu_body)};}
int64_t mesh_cpu_command_bind(MeshCpuCommand *c,uint32_t kind) {
  if(kind==MeshCpuGelu){c->run=mesh_cpu_gelu;return 0;}
  if(kind==MeshCpuDenseProduct) {
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
void mesh_cpu_gelu(const void *arguments) { mesh_cpu_gelu_body(arguments); }
#undef CPU_ENTRY
typedef struct MeshCpuProgram MeshCpuProgram;
typedef struct {
  void (*run)(const void *);
  void *arguments;
  MeshHostAccess *accesses;
  MeshCpuProgram *program;
} MeshCpuBound;
typedef struct { void (*release_arguments)(void *); void *library; } MeshCpuOwnership;
struct MeshCpuProgram {
  _Atomic unsigned references;
  size_t count;
  void *resources; /* retained MeshResourceBinding */
  MeshCpuOwnership *owners;
  MeshHostCommand *commands;
  MeshCpuBound instructions[];
};
static const char *cpu_instruction_prepare(void *,MeshHostCompletion,void *,MeshHostTask *);

/* Preparation transfers command-owned workspace into this native lifetime.
   The schedule retains it independently of the frontend that constructed it. */
void *mesh_cpu_program_create(const MeshCpuInstruction *instructions, size_t count, const char *resources_json) {
  if(count&&!instructions){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"CPU program has no declared instructions");return NULL;}
  id declared=resources_json?[NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:resources_json length:strlen(resources_json)] options:0 error:nil]:nil;
  MeshResourceBinding *resources=mesh_host_resources(declared);if(!resources)return NULL;
  if(count>(SIZE_MAX-sizeof(MeshCpuProgram))/sizeof(MeshCpuBound)||count>SIZE_MAX/sizeof(MeshCpuOwnership)){
    snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"CPU instruction extent overflows");return NULL;
  }
  MeshCpuProgram *p=calloc(1,sizeof *p+count*sizeof(MeshCpuBound));
  if(!p){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"allocating CPU program");return NULL;}
  p->owners=calloc(MAX(count,(size_t)1),sizeof *p->owners);
  if(!p->owners){free(p);snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"allocating CPU instruction ownership");return NULL;}
  p->commands=calloc(MAX(count,(size_t)1),sizeof *p->commands);
  if(!p->commands){free(p->owners);free(p);snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"allocating CPU instruction declarations");return NULL;}
  for(size_t i=0;i<count;i++) {
    const MeshCpuInstruction *source=instructions+i;MeshCpuBound *bound=p->instructions+i;
    if(!source->access_count||!source->accesses||source->access_count>SIZE_MAX/sizeof(MeshHostAccess)) {
      snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"CPU instruction requires its declared page accesses");goto failed;
    }
    if((source->argument_bytes&&!source->arguments)||
       (source->run?(source->library||source->symbol):(!source->library||!source->symbol))) {
      snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"CPU instruction needs explicit arguments and one static entry or library/symbol");goto failed;
    }
    bound->run=source->run;
    if(source->library) {
      p->owners[i].library=dlopen(source->library,RTLD_NOW|RTLD_LOCAL);
      if(!p->owners[i].library){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"CPU instruction library: %s",dlerror());goto failed;}
      bound->run=(void(*)(const void *))dlsym(p->owners[i].library,source->symbol);
      if(!bound->run){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"CPU instruction symbol: %s",dlerror());goto failed;}
    }
    bound->arguments=malloc(MAX(source->argument_bytes,(size_t)1));
    if(!bound->arguments){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"copying CPU instruction arguments");goto failed;}
    if(source->argument_bytes)memcpy(bound->arguments,source->arguments,source->argument_bytes);
    p->owners[i].release_arguments=source->release_arguments;
    bound->accesses=malloc(source->access_count*sizeof *bound->accesses);
    if(!bound->accesses){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"copying CPU instruction page accesses");goto failed;}
    memcpy(bound->accesses,source->accesses,source->access_count*sizeof *bound->accesses);
    bound->program=p;
    p->commands[i]=(MeshHostCommand){cpu_instruction_prepare,bound,bound->accesses,source->access_count};
  }
  p->resources=(__bridge_retained void *)resources;
  atomic_init(&p->references,1);p->count=count;
  return p;
failed:
  for(size_t i=0;i<count;i++) {
    free(p->instructions[i].accesses);
    free(p->instructions[i].arguments);
    if(p->owners[i].library)dlclose(p->owners[i].library);
  }
  free(p->commands);free(p->owners);free(p);return NULL;
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
    free(p->instructions[j].accesses);
    if(p->owners[j].library)dlclose(p->owners[j].library);
  }
  CFRelease(p->resources);free(p->commands);free(p->owners);free(p);
}
typedef struct { MeshCpuBound *instruction; MeshHostCompletion complete; void *context; } MeshCpuTask;
static void cpu_submit(void *bound) {
  MeshCpuTask *task=bound;MeshCpuBound *instruction=task->instruction;
  instruction->run(instruction->arguments);
  task->complete(task->context,NULL);
}
static void cpu_release(void *bound) {
  MeshCpuTask *task=bound;mesh_cpu_program_release(task->instruction->program);free(task);
}
static const char *cpu_instruction_prepare(void *declaration,MeshHostCompletion complete,void *context,MeshHostTask *result) {
  MeshCpuBound *instruction=declaration;MeshCpuProgram *program=instruction->program;
  MeshCpuTask *task=malloc(sizeof *task);
  if(!task)return "preparing CPU completion operand";
  mesh_cpu_program_retain(program);*task=(MeshCpuTask){instruction,complete,context};
  MeshResourceBinding *resources=(__bridge MeshResourceBinding *)program->resources;
  *result=(MeshHostTask){.context=task,.submit=cpu_submit,.release=cpu_release,
    .resources=resources->entries,.resource_count=resources->count};return NULL;
}
const char *mesh_cpu_program_prepare(void *program,MeshHostCompletion complete,void *context,MeshHostTask *result) {
  (void)complete;(void)context;
  MeshCpuProgram *p=program;
  mesh_cpu_program_retain(p);
  *result=(MeshHostTask){.context=p,.release=mesh_cpu_program_release,.commands=p->commands,.command_count=p->count};return NULL;
}
