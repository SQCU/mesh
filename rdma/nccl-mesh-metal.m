#import <Metal/Metal.h>
#include <stdint.h>

/* The stream's Metal side for nccl-mesh.c (nccl.h's struct ncclMeshStream): a shared event on the
   queue's device, its value read and signalled by the host, and command buffers committed to the
   queue that signal it (the queue's prior work done) or wait for it (the call's completion). */
__attribute__((visibility("hidden"))) void *nccl_mesh_event_create(void *queue){
  @autoreleasepool {
    id<MTLDevice> device=queue?[(id<MTLCommandQueue>)queue device]:MTLCreateSystemDefaultDevice();
    if(!device)return NULL;
    id<MTLSharedEvent> event=[device newSharedEvent];
    if(!queue)[device release];
    event.signaledValue=0;
    return event;
  }
}
__attribute__((visibility("hidden"))) void nccl_mesh_event_release(void *event){[(id)event release];}
__attribute__((visibility("hidden"))) uint64_t nccl_mesh_event_value(void *event){return [(id<MTLSharedEvent>)event signaledValue];}
__attribute__((visibility("hidden"))) void nccl_mesh_event_signal(void *event,uint64_t value){
  id<MTLSharedEvent> shared=event;
  if(shared.signaledValue<value)shared.signaledValue=value;
}
__attribute__((visibility("hidden"))) int nccl_mesh_queue_signal(void *queue,void *event,uint64_t value){
  @autoreleasepool {
    id<MTLCommandBuffer> buffer=[(id<MTLCommandQueue>)queue commandBuffer];
    if(!buffer)return -1;
    [buffer encodeSignalEvent:(id<MTLSharedEvent>)event value:value];
    [buffer commit];
    return 0;
  }
}
__attribute__((visibility("hidden"))) int nccl_mesh_queue_wait(void *queue,void *event,uint64_t value){
  @autoreleasepool {
    id<MTLCommandBuffer> buffer=[(id<MTLCommandQueue>)queue commandBuffer];
    if(!buffer)return -1;
    [buffer encodeWaitForEvent:(id<MTLSharedEvent>)event value:value];
    [buffer commit];
    return 0;
  }
}
