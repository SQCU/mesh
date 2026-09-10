#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "mesh-tensor.h"
#include "mesh-metal.h"
#include <errno.h>

@interface MeshTensorFunction : NSObject
@property struct mesh_row_function function;
@property uint32_t identifier;
@property struct mesh_row_map metadata;
@property NSData *commands;
@property id<MTLBuffer> inputs;
@property id<MTLBuffer> outputs;
@property NSArray<id<MTLBuffer>> *reads;
@property NSArray<id<MTLBuffer>> *writes;
@property NSArray<id<MTLBuffer>> *readWrites;
@end
@implementation MeshTensorFunction
@end

@interface MeshTensorProgram : NSObject
@property id<MTLDevice> device;
@property id<MTLCommandQueue> queue;
@property id<MTLLibrary> library;
@property id<MTLBuffer> views;
@property id<MTLBuffer> dimensions;
@property id<MTLBuffer> arguments;
@property id<MTLBuffer> table;
@property id<MTLBuffer> addresses;
@property NSArray<id<MTLBuffer>> *regions;
@property NSMutableArray<id<MTLComputePipelineState>> *kernels;
@property NSMutableArray<MeshTensorFunction*> *functions;
@property struct mesh_rows *pages;
@property NSString *error;
@end
@implementation MeshTensorProgram
@end

static NSString *mesh_tensor_creation_error;

// ../design/algorithm-sources.md#complete-page-ownership
void *mesh_tensor_create(const char *source,size_t functions){
  MeshTensorProgram *program=[MeshTensorProgram new];
  program.device=MTLCreateSystemDefaultDevice();
  program.queue=functions?[program.device newCommandQueueWithMaxCommandBufferCount:functions]:nil;
  program.kernels=[NSMutableArray new]; program.functions=[NSMutableArray new];
  MTLCompileOptions *options=[MTLCompileOptions new]; options.mathMode=MTLMathModeSafe;
  NSError *error=nil;
  program.library=[program.device newLibraryWithSource:[mesh_metal_address_source() stringByAppendingString:[NSString stringWithUTF8String:source]] options:options error:&error];
  if(!program.library){ mesh_tensor_creation_error=error.localizedDescription; return NULL; }
  return (__bridge_retained void*)program;
}

// ../design/algorithm-sources.md#complete-page-ownership
const char *mesh_tensor_error(void *handle){
  MeshTensorProgram *program=(__bridge MeshTensorProgram*)handle;
  return (program?program.error:mesh_tensor_creation_error).UTF8String;
}

// ../design/algorithm-sources.md#complete-page-ownership
int mesh_tensor_kernel(void *handle,const char *name){
  MeshTensorProgram *program=(__bridge MeshTensorProgram*)handle;
  NSError *error=nil;
  id<MTLFunction> function=[program.library newFunctionWithName:[NSString stringWithUTF8String:name]];
  id<MTLComputePipelineState> kernel=[program.device newComputePipelineStateWithFunction:function error:&error];
  if(!kernel){ program.error=error.localizedDescription; return -1; }
  int index=(int)program.kernels.count; [program.kernels addObject:kernel]; return index;
}

// ../design/algorithm-sources.md#complete-page-ownership
static id<MTLBuffer> mesh_tensor_values(MeshTensorProgram *program,const void *source,size_t bytes){
  struct mesh_rows *p=program.pages;
  size_t count=MAX((bytes+p->bytes-1)/p->bytes,1);
  uint32_t first=mesh_rows_allocate(p,count,p->bytes);
  if(first==MESH_ROW_ABSENT) return nil;
  void *address=mesh_at(p->memory,first);
  if(bytes) memcpy(address,source,bytes);
  return [program.device newBufferWithBytesNoCopy:address length:count*p->bytes options:MTLResourceStorageModeShared deallocator:nil];
}

// ../design/algorithm-sources.md#contiguous-backing-page-views
int mesh_tensor_reserve(void *handle,struct mesh_rows *pages,
  const struct mesh_tensor_view *views,size_t count,const uint64_t *dimensions,size_t rank,
  const uint32_t *arguments,size_t argument_count){
  MeshTensorProgram *program=(__bridge MeshTensorProgram*)handle;
  program.pages=pages;
  program.views=mesh_tensor_values(program,views,count*sizeof(*views));
  program.dimensions=mesh_tensor_values(program,dimensions,rank*sizeof(*dimensions));
  program.arguments=mesh_tensor_values(program,arguments,argument_count*sizeof(*arguments));
  size_t bytes=(pages->count*sizeof(struct mesh_row)+pages->bytes-1)/pages->bytes*pages->bytes;
  program.table=[program.device newBufferWithBytesNoCopy:pages->table length:bytes options:MTLResourceStorageModeShared deallocator:nil];
  NSArray<id<MTLBuffer>> *resources=nil;
  program.addresses=mesh_metal_regions(program.device,pages,&resources);
  program.regions=resources;
  return program.views && program.dimensions && program.arguments && program.table && program.addresses?0:ENOMEM;
}

// ../design/algorithm-sources.md#literal-row-functions
int mesh_tensor_function(void *handle,uint32_t index,uint32_t identifier,const struct mesh_row_function *function,
  struct mesh_row_map metadata,const struct mesh_tensor_command *commands,size_t count,
  const struct mesh_tensor_resource *resources,size_t resource_count){
  MeshTensorProgram *program=(__bridge MeshTensorProgram*)handle;
  if(index!=program.functions.count || !count) return EINVAL;
  MeshTensorFunction *binding=[MeshTensorFunction new];
  binding.inputs=mesh_tensor_values(program,function->input,function->inputs*sizeof(struct mesh_row_map));
  binding.outputs=mesh_tensor_values(program,function->output,function->outputs*sizeof(struct mesh_row_map));
  if(!binding.inputs || !binding.outputs) return ENOMEM;
  struct mesh_row_function realized=*function;
  realized.input=binding.inputs.contents; realized.output=binding.outputs.contents;
  binding.function=realized; binding.metadata=metadata; binding.identifier=identifier;
  binding.commands=[NSData dataWithBytes:commands length:count*sizeof(*commands)];
  NSMutableArray *reads=[NSMutableArray new],*writes=[NSMutableArray new],*readWrites=[NSMutableArray new];
  for(size_t i=0;i<resource_count;i++){
    id<MTLBuffer> region=program.regions[resources[i].region];
    if(resources[i].usage==(MTLResourceUsageRead|MTLResourceUsageWrite)) [readWrites addObject:region];
    else if(resources[i].usage==MTLResourceUsageRead) [reads addObject:region];
    else [writes addObject:region];
  }
  binding.reads=reads; binding.writes=writes; binding.readWrites=readWrites;
  [program.functions addObject:binding]; return 0;
}

// ../design/algorithm-sources.md#asynchronous-metadata-publication
void mesh_tensor_submit(void *handle,uint32_t function,uint64_t stamp,uint32_t indices){
  MeshTensorProgram *program=(__bridge MeshTensorProgram*)handle;
  MeshTensorFunction *binding=program.functions[function];
  id<MTLCommandBuffer> command=[program.queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
  [encoder setBuffer:program.addresses offset:0 atIndex:0];
  [encoder setBuffer:program.views offset:0 atIndex:1];
  [encoder setBuffer:program.dimensions offset:0 atIndex:2];
  [encoder setBuffer:program.table offset:0 atIndex:3];
  for(id<MTLBuffer> region in binding.reads) [encoder useResource:region usage:MTLResourceUsageRead];
  for(id<MTLBuffer> region in binding.writes) [encoder useResource:region usage:MTLResourceUsageWrite];
  for(id<MTLBuffer> region in binding.readWrites) [encoder useResource:region usage:MTLResourceUsageRead|MTLResourceUsageWrite];
  const struct mesh_tensor_command *commands=binding.commands.bytes;
  for(size_t i=0;i<binding.commands.length/sizeof(*commands);i++){
    const struct mesh_tensor_command *c=&commands[i];
    [encoder setComputePipelineState:program.kernels[c->kernel]];
    [encoder setBuffer:program.arguments offset:(size_t)c->argument_offset*sizeof(uint32_t) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(c->grid[0],c->grid[1],c->grid[2])
      threadsPerThreadgroup:MTLSizeMake(c->group[0],c->group[1],c->group[2])];
  }
  [encoder endEncoding];
  [command addCompletedHandler:^(id<MTLCommandBuffer> completed){
    NSError *error=completed.error;
    struct mesh_row_metadata metadata={.stamp=stamp,.when=stamp,
      .function=binding.identifier,.peer=program.pages->memory->node,.code=error.code,.domain=error?4:0};
    mesh_rows_report(program.pages,binding.metadata,0,metadata);
    struct mesh_row_function completed_function=binding.function;
    mesh_rows_complete(program.pages,&completed_function,stamp,indices);
  }];
  [command commit];
}

// ../design/algorithm-sources.md#complete-page-ownership
void mesh_tensor_free(void *handle){ (void)CFBridgingRelease(handle); }
