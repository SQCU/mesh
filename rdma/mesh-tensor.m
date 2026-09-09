#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "mesh-tensor.h"
#include <errno.h>

@interface MeshTensorProgram : NSObject
@property id<MTLDevice> device;
@property id<MTLCommandQueue> queue;
@property id<MTLLibrary> library;
@property id<MTLBuffer> arena;
@property id<MTLBuffer> views;
@property id<MTLBuffer> dimensions;
@property id<MTLBuffer> arguments;
@property id<MTLBuffer> receive;
@property id<MTLBuffer> transmit;
@property id<MTLBuffer> pages;
@property NSMutableArray<id<MTLComputePipelineState>> *kernels;
@property NSMutableArray *phases;
@property NSMutableArray<NSNumber*> *counts;
@property NSMutableArray<NSData*> *specifications;
@property id<MTLCommandBuffer> pending;
@property NSString *error;
@end
@implementation MeshTensorProgram
@end

static NSString *mesh_tensor_creation_error;

void *mesh_tensor_create(const char *source){
  @autoreleasepool {
    MeshTensorProgram *program=[MeshTensorProgram new];
    program.device=MTLCreateSystemDefaultDevice();
    program.queue=[program.device newCommandQueue];
    program.kernels=[NSMutableArray new];
    program.phases=[NSMutableArray new];
    program.counts=[NSMutableArray new];
    program.specifications=[NSMutableArray new];
    MTLCompileOptions *options=[MTLCompileOptions new];
    options.mathMode=MTLMathModeSafe;
    NSError *error=nil;
    program.library=[program.device newLibraryWithSource:[NSString stringWithUTF8String:source] options:options error:&error];
    if(!program.library){ mesh_tensor_creation_error=error.localizedDescription; return NULL; }
    return (__bridge_retained void*)program;
  }
}

const char *mesh_tensor_error(void *handle){
  MeshTensorProgram *program=(__bridge MeshTensorProgram*)handle;
  return (program?program.error:mesh_tensor_creation_error).UTF8String;
}

int mesh_tensor_kernel(void *handle,const char *name){
  MeshTensorProgram *program=(__bridge MeshTensorProgram*)handle;
  MTLComputePipelineDescriptor *descriptor=[MTLComputePipelineDescriptor new];
  descriptor.computeFunction=[program.library newFunctionWithName:[NSString stringWithUTF8String:name]];
  descriptor.supportIndirectCommandBuffers=YES;
  NSError *error=nil;
  id<MTLComputePipelineState> kernel=[program.device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&error];
  if(!kernel){ program.error=error.localizedDescription; return -1; }
  int index=(int)program.kernels.count;
  [program.kernels addObject:kernel];
  return index;
}

static id<MTLBuffer> mesh_tensor_buffer(MeshTensorProgram *program,id<MTLBuffer> previous,size_t bytes){
  if(previous.length>=bytes) return previous;
  id<MTLBuffer> buffer=[program.device newBufferWithLength:MAX(bytes,256) options:MTLResourceStorageModeShared];
  if(buffer && previous) memcpy(buffer.contents,previous.contents,previous.length);
  return buffer;
}

int mesh_tensor_reserve(void *handle,size_t bytes,const struct mesh_tensor_view *views,size_t count,
                        const uint64_t *dimensions,size_t rank,const uint32_t *arguments,size_t argument_count){
  MeshTensorProgram *program=(__bridge MeshTensorProgram*)handle;
  if(program.pending && program.pending.status<MTLCommandBufferStatusCompleted) return EBUSY;
  id<MTLBuffer> arena=mesh_tensor_buffer(program,program.arena,bytes);
  id<MTLBuffer> metadata=mesh_tensor_buffer(program,program.views,count*sizeof(*views));
  id<MTLBuffer> dims=mesh_tensor_buffer(program,program.dimensions,rank*sizeof(*dimensions));
  id<MTLBuffer> args=mesh_tensor_buffer(program,program.arguments,argument_count*sizeof(*arguments));
  if(!arena || !metadata || !dims || !args){ program.error=@"persistent tensor allocation failed"; return ENOMEM; }
  program.arena=arena; program.views=metadata; program.dimensions=dims; program.arguments=args;
  memcpy(program.views.contents,views,count*sizeof(*views));
  memcpy(program.dimensions.contents,dimensions,rank*sizeof(*dimensions));
  memcpy(program.arguments.contents,arguments,argument_count*sizeof(*arguments));
  return 0;
}

void *mesh_tensor_data(void *handle){ return ((__bridge MeshTensorProgram*)handle).arena.contents; }
void *mesh_tensor_memory(void *handle){ return (__bridge_retained void*)((__bridge MeshTensorProgram*)handle).arena; }
void mesh_tensor_memory_free(void *handle){ (void)CFBridgingRelease(handle); }

int mesh_tensor_phase(void *handle,uint32_t phase,const struct mesh_tensor_command *commands,size_t count){
  MeshTensorProgram *program=(__bridge MeshTensorProgram*)handle;
  if(program.pending && program.pending.status<MTLCommandBufferStatusCompleted) return EBUSY;
  MTLIndirectCommandBufferDescriptor *descriptor=[MTLIndirectCommandBufferDescriptor new];
  descriptor.commandTypes=MTLIndirectCommandTypeConcurrentDispatch;
  descriptor.inheritBuffers=NO;
  descriptor.inheritPipelineState=NO;
  descriptor.maxKernelBufferBindCount=7;
  id<MTLIndirectCommandBuffer> buffer=phase<program.phases.count && program.counts[phase].unsignedIntegerValue>=count?program.phases[phase]:nil;
  if(buffer) [buffer resetWithRange:NSMakeRange(0,program.counts[phase].unsignedIntegerValue)];
  else buffer=[program.device newIndirectCommandBufferWithDescriptor:descriptor maxCommandCount:MAX(count,1) options:MTLResourceStorageModePrivate];
  if(!buffer){ program.error=@"persistent tensor command allocation failed"; return ENOMEM; }
  for(size_t i=0;i<count;i++){
    id<MTLIndirectComputeCommand> command=[buffer indirectComputeCommandAtIndex:i];
    [command setComputePipelineState:program.kernels[commands[i].kernel]];
    [command setKernelBuffer:program.arena offset:0 atIndex:0];
    [command setKernelBuffer:program.views offset:0 atIndex:1];
    [command setKernelBuffer:program.dimensions offset:0 atIndex:2];
    [command setKernelBuffer:program.receive?:program.arena offset:0 atIndex:3];
    [command setKernelBuffer:program.transmit?:program.arena offset:0 atIndex:4];
    [command setKernelBuffer:program.pages?:program.arena offset:0 atIndex:5];
    [command setKernelBuffer:program.arguments offset:(size_t)commands[i].argument_offset*sizeof(uint32_t) atIndex:6];
    [command concurrentDispatchThreadgroups:MTLSizeMake(commands[i].grid[0],commands[i].grid[1],commands[i].grid[2])
                     threadsPerThreadgroup:MTLSizeMake(commands[i].group[0],commands[i].group[1],commands[i].group[2])];
    [command setBarrier];
  }
  while(program.phases.count<=phase){ [program.phases addObject:[NSNull null]]; [program.counts addObject:@0]; [program.specifications addObject:[NSData data]]; }
  program.phases[phase]=buffer;
  program.counts[phase]=@(count);
  program.specifications[phase]=[NSData dataWithBytes:commands length:count*sizeof(*commands)];
  return 0;
}

int mesh_tensor_submit(void *handle,uint32_t phase){
  @autoreleasepool {
    MeshTensorProgram *program=(__bridge MeshTensorProgram*)handle;
    if(program.pending && program.pending.status<MTLCommandBufferStatusCompleted) return EBUSY;
    id<MTLCommandBuffer> command=[program.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    [encoder useResource:program.arena usage:MTLResourceUsageRead|MTLResourceUsageWrite];
    [encoder useResource:program.views usage:MTLResourceUsageRead];
    [encoder useResource:program.dimensions usage:MTLResourceUsageRead];
    [encoder useResource:program.arguments usage:MTLResourceUsageRead];
    if(program.receive) [encoder useResource:program.receive usage:MTLResourceUsageRead];
    if(program.transmit) [encoder useResource:program.transmit usage:MTLResourceUsageRead|MTLResourceUsageWrite];
    if(program.pages) [encoder useResource:program.pages usage:MTLResourceUsageRead];
    [encoder executeCommandsInBuffer:program.phases[phase] withRange:NSMakeRange(0,program.counts[phase].unsignedIntegerValue)];
    [encoder endEncoding];
    [command commit];
    program.pending=command;
    return 0;
  }
}

int mesh_tensor_status(void *handle){
  MeshTensorProgram *program=(__bridge MeshTensorProgram*)handle;
  if(!program.pending || program.pending.status==MTLCommandBufferStatusCompleted) return 1;
  if(program.pending.status==MTLCommandBufferStatusError){ program.error=program.pending.error.localizedDescription; return -EIO; }
  return 0;
}

void mesh_tensor_free(void *handle){
  (void)CFBridgingRelease(handle);
}

#import "mesh-metal-executor.h"
#include "mesh-wire.h"

@class MeshTensorChannel;
@interface MeshTensorTransport : NSObject
@property struct mesh_ctx *context;
@property mesh_executor *executor;
@property id<MTLDevice> device;
@property id<MTLBuffer> receive;
@property id<MTLBuffer> transmit;
@property struct mesh_metal_layout receiveLayout;
@property struct mesh_metal_layout transmitLayout;
@property id<MTLComputePipelineState> transferKernel;
@property id<MTLComputePipelineState> commitKernel;
@property NSMutableArray<MeshTensorChannel*> *channels;
@property size_t bytes;
@end
@implementation MeshTensorTransport
@end

struct mesh_tensor_transfer_parameters { uint64_t offset, bytes, extent, origin; uint32_t stride, payload, receive, spans; };
@interface MeshTensorChannel : NSObject
@property MeshTensorTransport *transport;
@property MeshTensorProgram *program;
@property mesh_call *call;
@property id<MTLBuffer> indices;
@property id<MTLBuffer> parameters;
@property id<MTLBuffer> spans;
@property id<MTLBuffer> bindings;
@property id<MTLBuffer> arena;
@property id<MTLIndirectCommandBuffer> command;
@property id<MTLIndirectCommandBuffer> commit;
@property id<MTLCommandBuffer> pending;
@property uint64_t epoch;
@property int state;
@property int error;
@property BOOL receiving;
@property size_t bytes;
@property size_t streamBytes;
@property size_t spanCount;
@property size_t bindingCount;
@end
@implementation MeshTensorChannel
@end

void *mesh_tensor_transport(void *context){
  @autoreleasepool {
    MeshTensorTransport *transport=[MeshTensorTransport new];
    transport.context=context;
    struct mesh_ctx *ctx=context;
    size_t window=MIN(ctx->M->pool,ctx->M->arena)/4;
    transport.executor=mesh_executor_create(ctx,window);
    if(!transport.executor) return NULL;
    transport.bytes=MIN((size_t)1048576,window*(mesh_pay(ctx->M)-sizeof(struct mesh_frame)));
    transport.device=MTLCreateSystemDefaultDevice();
    struct mesh_metal_layout receive,transmit;
    transport.receive=mesh_metal_receive_pool(transport.device,ctx,&receive);
    transport.transmit=mesh_metal_transmit_pool(transport.device,ctx,&transmit);
    transport.receiveLayout=receive; transport.transmitLayout=transmit;
    transport.channels=[NSMutableArray new];
    NSString *source=@"#include <metal_stdlib>\nusing namespace metal;\n"
      "struct Transfer { ulong offset,bytes,extent,origin; uint stride,payload,receive,spans; };\n"
      "struct Span { ulong stream,offset,bytes,target; };\n"
      "kernel void mesh_tensor_copy(device uchar* arena [[buffer(0)]],device uchar* pool [[buffer(1)]],device const uint* pages [[buffer(2)]],constant Transfer& p [[buffer(3)]],device const Span* spans [[buffer(4)]],uint t [[thread_position_in_grid]]){\n"
      "if(t>=p.extent) return; ulong n=p.stride-p.payload; ulong at=ulong(pages[t/n])*p.stride+p.payload+t%n-p.origin;\n"
      "if(t>=p.bytes){ if(!p.receive) pool[at]=0; return; } ulong logical=p.offset+t; uint low=0,high=p.spans;\n"
      "while(low+1<high){ uint middle=(low+high)/2; if(spans[middle].stream<=logical) low=middle; else high=middle; }\n"
      "ulong address=spans[low].offset+logical-spans[low].stream; if(p.receive) arena[address]=pool[at]; else pool[at]=arena[address]; }\n"
      "kernel void mesh_tensor_commit(device uchar* arena [[buffer(0)]],constant Transfer& p [[buffer(1)]],device const Span* spans [[buffer(2)]],uint tid [[thread_position_in_grid]]){\n"
      "for(ulong t=tid;t<p.bytes;t+=65536){ uint low=0,high=p.spans; while(low+1<high){ uint middle=(low+high)/2; if(spans[middle].stream<=t) low=middle; else high=middle; }\n"
      "ulong at=t-spans[low].stream; arena[spans[low].target+at]=arena[spans[low].offset+at]; }}\n";
    NSError *error=nil;
    id<MTLLibrary> library=[transport.device newLibraryWithSource:source options:nil error:&error];
    MTLComputePipelineDescriptor *descriptor=[MTLComputePipelineDescriptor new];
    descriptor.computeFunction=[library newFunctionWithName:@"mesh_tensor_copy"];
    descriptor.supportIndirectCommandBuffers=YES;
    transport.transferKernel=[transport.device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&error];
    descriptor.computeFunction=[library newFunctionWithName:@"mesh_tensor_commit"];
    transport.commitKernel=[transport.device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&error];
    if(!transport.receive || !transport.transmit || !transport.transferKernel || !transport.commitKernel){
      mesh_tensor_creation_error=error.localizedDescription?:@"mesh page alias allocation failed";
      mesh_executor_free(transport.executor); return NULL;
    }
    return (__bridge_retained void*)transport;
  }
}

size_t mesh_tensor_transport_bytes(void *handle){ return ((__bridge MeshTensorTransport*)handle).bytes; }

static int mesh_tensor_extent(void *capture,size_t index,struct mesh_extent *extent){
  *extent=*(struct mesh_extent*)capture; return 0;
}

void *mesh_tensor_channel(void *handle,void *program,int peer,int receive,uint32_t channel,uint64_t epoch,const unsigned char plan[32],size_t bytes){
  MeshTensorTransport *transport=(__bridge MeshTensorTransport*)handle;
  struct mesh_extent extent={.bytes=bytes,.peer=peer,.receive=receive};
  mesh_function *function=mesh_compile(1,mesh_tensor_extent,&extent);
  if(!function) return NULL;
  struct mesh_scope scope={.epoch={epoch,1}};
  memcpy(scope.plan,plan,32);
  mesh_call *call=mesh_bind_scoped(transport.executor,function,channel,scope);
  mesh_function_free(function);
  if(!call) return NULL;
  mesh_call_retain_transmit(call);
  MeshTensorChannel *binding=[MeshTensorChannel new];
  binding.transport=transport; binding.program=(__bridge MeshTensorProgram*)program;
  binding.call=call; binding.receiving=receive;
  binding.epoch=epoch;
  binding.bytes=bytes;
  binding.indices=mesh_metal_indices(transport.device,call);
  binding.parameters=[transport.device newBufferWithLength:256 options:MTLResourceStorageModeShared];
  size_t count=binding.program.views.length/sizeof(struct mesh_tensor_view);
  binding.spans=[transport.device newBufferWithLength:MAX(256,count*sizeof(struct mesh_tensor_span)) options:MTLResourceStorageModeShared];
  binding.bindings=[transport.device newBufferWithLength:MAX(256,count*sizeof(struct mesh_tensor_binding)) options:MTLResourceStorageModeShared];
  if(!binding.indices || !binding.parameters || !binding.spans || !binding.bindings) binding.error=ENOMEM;
  MeshTensorProgram *tensor=binding.program;
  BOOL changed=tensor.receive!=transport.receive || !tensor.pages || tensor.pages.length<binding.indices.length;
  tensor.receive=transport.receive; tensor.transmit=transport.transmit;
  tensor.pages=mesh_tensor_buffer(tensor,tensor.pages,binding.indices.length);
  if(!tensor.pages) binding.error=ENOMEM;
  if(changed) for(NSUInteger i=0;i<tensor.specifications.count;i++){
    NSData *commands=tensor.specifications[i];
    int status=mesh_tensor_phase((__bridge void*)tensor,(uint32_t)i,commands.bytes,commands.length/sizeof(struct mesh_tensor_command));
    if(status) binding.error=status;
  }
  [transport.channels addObject:binding];
  return (__bridge_retained void*)binding;
}

int mesh_tensor_transfer_plan(void *handle,const struct mesh_tensor_span *spans,size_t count,const struct mesh_tensor_binding *bindings,size_t binding_count){
  MeshTensorChannel *channel=(__bridge MeshTensorChannel*)handle;
  if(channel.error) return channel.error;
  if(channel.state && mesh_call_status(channel.call)!=1) return EBUSY;
  if(count*sizeof(*spans)>channel.spans.length || binding_count*sizeof(*bindings)>channel.bindings.length) return EOVERFLOW;
  memcpy(channel.spans.contents,spans,count*sizeof(*spans));
  memcpy(channel.bindings.contents,bindings,binding_count*sizeof(*bindings));
  channel.spanCount=count; channel.bindingCount=binding_count;
  channel.streamBytes=count?spans[count-1].stream+spans[count-1].bytes:0;
  return 0;
}

int mesh_tensor_transfer_release(void *handle){
  MeshTensorChannel *channel=(__bridge MeshTensorChannel*)handle;
  if(channel.state!=4) return 0;
  if(channel.program.pending && channel.program.pending.status<MTLCommandBufferStatusCompleted) return EBUSY;
  struct mesh_tensor_view *views=channel.program.views.contents;
  const struct mesh_tensor_binding *bindings=channel.bindings.contents;
  for(size_t i=0;i<channel.bindingCount;i++){
    views[bindings[i].tensor].pool=0; views[bindings[i].tensor].offset=bindings[i].offset;
  }
  int status=mesh_complete(channel.call,0,channel.error);
  channel.state=3;
  return status;
}

static void mesh_tensor_transfer_record(MeshTensorChannel *channel){
  MTLIndirectCommandBufferDescriptor *descriptor=[MTLIndirectCommandBufferDescriptor new];
  descriptor.commandTypes=MTLIndirectCommandTypeConcurrentDispatch;
  descriptor.inheritBuffers=NO; descriptor.inheritPipelineState=NO;
  descriptor.maxKernelBufferBindCount=5;
  channel.command=[channel.transport.device newIndirectCommandBufferWithDescriptor:descriptor maxCommandCount:1 options:MTLResourceStorageModePrivate];
  channel.arena=channel.program.arena;
  id<MTLIndirectComputeCommand> command=[channel.command indirectComputeCommandAtIndex:0];
  [command setComputePipelineState:channel.transport.transferKernel];
  [command setKernelBuffer:channel.arena offset:0 atIndex:0];
  [command setKernelBuffer:channel.receiving?channel.transport.receive:channel.transport.transmit offset:0 atIndex:1];
  [command setKernelBuffer:channel.indices offset:0 atIndex:2];
  [command setKernelBuffer:channel.parameters offset:0 atIndex:3];
  [command setKernelBuffer:channel.spans offset:0 atIndex:4];
  [command concurrentDispatchThreadgroups:MTLSizeMake((channel.bytes+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  channel.commit=[channel.transport.device newIndirectCommandBufferWithDescriptor:descriptor maxCommandCount:1 options:MTLResourceStorageModePrivate];
  command=[channel.commit indirectComputeCommandAtIndex:0];
  [command setComputePipelineState:channel.transport.commitKernel];
  [command setKernelBuffer:channel.arena offset:0 atIndex:0];
  [command setKernelBuffer:channel.parameters offset:0 atIndex:1];
  [command setKernelBuffer:channel.spans offset:0 atIndex:2];
  [command concurrentDispatchThreadgroups:MTLSizeMake(256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}

int mesh_tensor_transfer_commit(void *handle){
  MeshTensorChannel *channel=(__bridge MeshTensorChannel*)handle;
  if(mesh_call_status(channel.call)!=1 || channel.program.pending.status<MTLCommandBufferStatusCompleted) return EBUSY;
  struct mesh_tensor_transfer_parameters *parameters=channel.parameters.contents;
  parameters->bytes=channel.streamBytes;
  id<MTLCommandBuffer> command=[channel.program.queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
  [encoder useResource:channel.arena usage:MTLResourceUsageRead|MTLResourceUsageWrite];
  [encoder useResource:channel.parameters usage:MTLResourceUsageRead];
  [encoder useResource:channel.spans usage:MTLResourceUsageRead];
  [encoder executeCommandsInBuffer:channel.commit withRange:NSMakeRange(0,1)];
  [encoder endEncoding]; [command commit]; channel.program.pending=command;
  return 0;
}

int mesh_tensor_transfer(void *handle,uint64_t epoch,uint64_t offset,uint64_t bytes){
  MeshTensorChannel *channel=(__bridge MeshTensorChannel*)handle;
  if(channel.state && mesh_call_status(channel.call)!=1) return EBUSY;
  if(bytes>channel.bytes || offset>channel.streamBytes || bytes>channel.streamBytes-offset) return EOVERFLOW;
  if(channel.state){
    int status=mesh_call_rearm(channel.call,(struct mesh_epoch){channel.epoch,epoch});
    if(status) return status;
  }
  else if(epoch!=1) return EINVAL;
  if(channel.arena!=channel.program.arena) mesh_tensor_transfer_record(channel);
  const struct mesh_view *view=mesh_call_view(channel.call,0);
  struct mesh_metal_layout layout=channel.receiving?channel.transport.receiveLayout:channel.transport.transmitLayout;
  struct mesh_tensor_transfer_parameters parameters={offset,bytes,channel.bytes,layout.origin,view->stride,view->payload,channel.receiving,(uint32_t)channel.spanCount};
  memcpy(channel.parameters.contents,&parameters,sizeof parameters);
  channel.state=1; channel.error=0;
  mesh_request(channel.call,0);
  return 0;
}

static void mesh_tensor_channel_progress(MeshTensorChannel *channel){
  if(channel.state==1){
    struct mesh_view view;
    int status=mesh_acquire(channel.call,0,&view);
    if(status==1){
      id<MTLCommandBuffer> command=[channel.program.queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
      [encoder useResource:channel.arena usage:MTLResourceUsageRead|MTLResourceUsageWrite];
      [encoder useResource:channel.receiving?channel.transport.receive:channel.transport.transmit usage:MTLResourceUsageRead|MTLResourceUsageWrite];
      [encoder useResource:channel.indices usage:MTLResourceUsageRead];
      [encoder useResource:channel.parameters usage:MTLResourceUsageRead];
      [encoder useResource:channel.spans usage:MTLResourceUsageRead];
      [encoder executeCommandsInBuffer:channel.command withRange:NSMakeRange(0,1)];
      [encoder endEncoding]; [command commit];
      channel.pending=command; channel.program.pending=command; channel.state=2;
    } else if(status<0) channel.error=-status;
  }
  if(channel.state==2 && channel.pending.status>=MTLCommandBufferStatusCompleted){
    int error=channel.error?channel.error:(channel.pending.status==MTLCommandBufferStatusError?EIO:0);
    channel.error=error;
    if(!error && channel.receiving && channel.bindingCount && channel.streamBytes<=channel.bytes){
      memcpy(channel.program.pages.contents,channel.indices.contents,channel.indices.length);
      const struct mesh_view *view=mesh_call_view(channel.call,0);
      struct mesh_tensor_view *views=channel.program.views.contents;
      const struct mesh_tensor_binding *bindings=channel.bindings.contents;
      for(size_t i=0;i<channel.bindingCount;i++){
        struct mesh_tensor_view *tensor=&views[bindings[i].tensor];
        tensor->offset=bindings[i].stream; tensor->pool=1; tensor->page_start=0;
        tensor->origin=channel.transport.receiveLayout.origin; tensor->page_stride=view->stride; tensor->payload=view->payload;
      }
      channel.state=4;
    } else { mesh_complete(channel.call,0,error); channel.state=3; }
  }
  if(channel.state==4 && channel.error) mesh_tensor_transfer_release((__bridge void*)channel);
}

int mesh_tensor_transport_progress(void *handle){
  @autoreleasepool {
    MeshTensorTransport *transport=(__bridge MeshTensorTransport*)handle;
    int activity=mesh_progress(transport.executor);
    for(MeshTensorChannel *channel in transport.channels) mesh_tensor_channel_progress(channel);
    return activity;
  }
}

int mesh_tensor_transfer_status(void *handle){
  MeshTensorChannel *channel=(__bridge MeshTensorChannel*)handle;
  int status=mesh_call_status(channel.call);
  return channel.error?-channel.error:channel.state==4?2:status;
}

void mesh_tensor_transfer_cancel(void *handle,int error){
  MeshTensorChannel *channel=(__bridge MeshTensorChannel*)handle;
  channel.error=error; mesh_call_cancel(channel.call,error);
}

int mesh_tensor_channel_free(void *handle){
  MeshTensorChannel *channel=(__bridge MeshTensorChannel*)handle;
  if(channel.program.pending && channel.program.pending.status<MTLCommandBufferStatusCompleted) return EBUSY;
  int status=mesh_call_status(channel.call)<0?mesh_call_abandon(channel.call):mesh_call_retire(channel.call);
  if(status) return status;
  [channel.transport.channels removeObjectIdenticalTo:channel];
  BOOL shared=NO;
  for(MeshTensorChannel *other in channel.transport.channels) shared|=other.program==channel.program;
  if(!shared){
    MeshTensorProgram *tensor=channel.program;
    tensor.receive=nil; tensor.transmit=nil;
    for(NSUInteger i=0;i<tensor.specifications.count;i++){
      NSData *commands=tensor.specifications[i];
      mesh_tensor_phase((__bridge void*)tensor,(uint32_t)i,commands.bytes,commands.length/sizeof(struct mesh_tensor_command));
    }
  }
  channel.transport=nil; channel.program=nil;
  (void)CFBridgingRelease(handle); return 0;
}

int mesh_tensor_transport_free(void *handle){
  MeshTensorTransport *transport=(__bridge MeshTensorTransport*)handle;
  if(transport.channels.count) return EBUSY;
  int status=mesh_executor_free(transport.executor);
  if(status) return status;
  (void)CFBridgingRelease(handle); return 0;
}
