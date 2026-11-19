/* design/declared-programs.md "Testing": one rank of a two-rank libnccl-mesh communicator on the Metal path over the
   two-node simulator: a program of two steps (an fp32 AllReduce (sum) of `bytes`, then a Broadcast from rank 0),
   declared with its buffers and stream kind (ncclMeshDeclareBegin, the calls on an ncclMeshStream, ncclMeshDeclareEnd:
   nothing is encoded), then run ROUNDS times, each step encoded into its own command buffer, committed and awaited,
   every value checked.  Each rank attaches its own node's region (MESH_REGION); the two meet at NCCL_COMM_ID.  The
   library's kernels are its build's (.build/libnccl-mesh.<SHA-256>.metallib, beside the library: tests/sim-two-nodes.sh).

   BYTES may be a list (6144,98304,...): the program declared again at each size, one process redeclaring.  With DIRTY
   set, the process first takes 64 MiB of leases, fills them with ones and gives them back, so its own leases come
   from pages a process wrote before: nothing a program posts may rest on a lease's prior contents.

   usage: sim-metal RANK [ROUNDS [BYTES[,BYTES...]]] */
#import <Metal/Metal.h>
#include "../nccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(ncclResult_t r,const char *what){
  if(r!=ncclSuccess){fprintf(stderr,"sim-metal: %s: %s (%s)\n",what,ncclGetErrorString(r),ncclGetLastError(NULL));return 1;}
  return 0;
}
static float value(int rank,size_t i,int round){ return (float)((rank+1)*100+(int)(i%251)+round); }
int main(int argc,char **argv){
  @autoreleasepool {
    if(argc<2)return 2;
    const int rank=atoi(argv[1]),rounds=argc>2?atoi(argv[2]):8;
    const char *sizes=argc>3?argv[3]:"98304";
    const char *named=getenv("NCCL_COMM_ID");
    if(!named)return 2;
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(!device){fprintf(stderr,"sim-metal: no Metal device\n");return 77;}
    id<MTLCommandQueue> queue=[device newCommandQueue];
    ncclUniqueId unique;memset(&unique,0,sizeof unique);snprintf(unique.internal,sizeof unique.internal,"%s",named);
    if(getenv("DIRTY")){
      void *dirt[64];
      for(int k=0;k<64;k++){if(check(ncclMemAlloc(&dirt[k],1<<20),"ncclMemAlloc"))return 1;memset(dirt[k],0xff,1<<20);}
      for(int k=0;k<64;k++)ncclMemFree(dirt[k]);
    }
    ncclComm_t comm;
    if(check(ncclCommInitRank(&comm,2,unique,rank),"ncclCommInitRank"))return 1;
    int failures=0;
    for(const char *at=sizes;*at && !failures;at=strchr(at,',')?strchr(at,',')+1:at+strlen(at)){
    const size_t bytes=(size_t)atol(at),n=bytes/sizeof(float);
    float *send,*recv,*bcast;
    if(check(ncclMemAlloc((void **)&send,bytes),"ncclMemAlloc") || check(ncclMemAlloc((void **)&recv,bytes),"ncclMemAlloc") ||
       check(ncclMemAlloc((void **)&bcast,bytes),"ncclMemAlloc"))return 1;
    ncclMeshBuffer s,r,b;
    if(check(ncclMeshMemBuffer(send,&s.buffer,&s.offset),"ncclMeshMemBuffer") || check(ncclMeshMemBuffer(recv,&r.buffer,&r.offset),"ncclMeshMemBuffer") ||
       check(ncclMeshMemBuffer(bcast,&b.buffer,&b.offset),"ncclMeshMemBuffer"))return 1;
    /* the program: its two steps on a stream (its command buffer unused while declaring) */
    id<MTLCommandBuffer> unused=[queue commandBuffer];
    ncclMeshStream declaring={.commandBuffer=(__bridge void *)unused};
    if(check(ncclMeshDeclareBegin(),"ncclMeshDeclareBegin") ||
       check(ncclAllReduce(&s,&r,n,ncclFloat,ncclSum,comm,&declaring),"ncclAllReduce (declared)") ||
       check(ncclBroadcast(&b,&b,n,ncclFloat,0,comm,&declaring),"ncclBroadcast (declared)") ||
       check(ncclMeshDeclareEnd(),"ncclMeshDeclareEnd"))return 1;
    for(int round=0;round<rounds && !failures;round++){
      for(size_t i=0;i<n;i++){send[i]=value(rank,i,round);recv[i]=0;bcast[i]=rank?0:value(0,i,round);}
      id<MTLCommandBuffer> reduce=[queue commandBuffer];
      ncclMeshStream stream={.commandBuffer=(__bridge void *)reduce};
      if(check(ncclAllReduce(&s,&r,n,ncclFloat,ncclSum,comm,&stream),"ncclAllReduce"))return 1;
      [reduce commit];[reduce waitUntilCompleted];
      if(reduce.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"sim-metal: the AllReduce's command buffer ended %ld\n",(long)reduce.status);return 1;}
      id<MTLCommandBuffer> broadcast=[queue commandBuffer];
      stream=(ncclMeshStream){.commandBuffer=(__bridge void *)broadcast};
      if(check(ncclBroadcast(&b,&b,n,ncclFloat,0,comm,&stream),"ncclBroadcast"))return 1;
      [broadcast commit];[broadcast waitUntilCompleted];
      for(size_t i=0;i<n;i++)if(recv[i]!=value(0,i,round)+value(1,i,round)){
        fprintf(stderr,"allreduce round %d: [%zu] %g, not %g\n",round,i,recv[i],value(0,i,round)+value(1,i,round));failures++;break;}
      for(size_t i=0;i<n;i++)if(bcast[i]!=value(0,i,round)){fprintf(stderr,"broadcast round %d: [%zu] %g\n",round,i,bcast[i]);failures++;break;}
      for(int ready=0;!ready;)if(check(ncclMeshMemReusable(send,&ready),"ncclMeshMemReusable"))return 1;
      for(int ready=0;!ready;)if(check(ncclMeshMemReusable(bcast,&ready),"ncclMeshMemReusable"))return 1;
    }
    printf("{\"rank\":%d,\"path\":\"metal\",\"rounds\":%d,\"bytes\":%zu,\"checked\":%s}\n",rank,rounds,bytes,failures?"false":"true");
    }
    if(check(ncclCommDestroy(comm),"ncclCommDestroy"))return 1;
    return failures?1:0;
  }
}
