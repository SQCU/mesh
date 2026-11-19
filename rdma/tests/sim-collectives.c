/* design/declared-programs.md "Testing": one rank of a real libnccl-mesh communicator of two ranks over the two-node
   simulator (tests/sim-wire.c): init, then, at each size (fp32 at 6 KB, 98 KB and 1.5 MB), the program of two rounds
   declared (each round on each communicator an AllReduce (sum), an AllGather and a Broadcast from root round mod 2:
   ncclMeshDeclareBegin, the calls, ncclMeshDeclareEnd) and run `rounds` rounds, every value checked; then destroy.
   Each rank attaches its own node's region (MESH_REGION); the two meet at NCCL_COMM_ID.  Communicators made `comms` at
   once are steps of the one program, each running every collective in turn.

   usage: sim-collectives RANK [COMMS [ROUNDS]]   (environment: MESH_REGION, NCCL_COMM_ID) */
#include "../nccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int check(ncclResult_t r,const char *what){
  if(r!=ncclSuccess){fprintf(stderr,"sim-collectives: %s: %s (%s)\n",what,ncclGetErrorString(r),ncclGetLastError(NULL));return 1;}
  return 0;
}
/* a send buffer written again only once every DMA read of it has ended (nccl.h ncclMeshMemReusable: a call returns
   before its SENDs are done) */
static int reusable(const void *buffer){
  for(int ready=0;!ready;)if(check(ncclMeshMemReusable(buffer,&ready),"ncclMeshMemReusable"))return 1;
  return 0;
}
static float value(int rank,size_t i,int round){ return (float)((rank+1)*1000+(int)(i%997)+round); }
int main(int argc,char **argv){
  if(argc<2)return 2;
  const int rank=atoi(argv[1]),comms=argc>2?atoi(argv[2]):1,rounds=argc>3?2*((atoi(argv[3])+1)/2):2;
  const char *id=getenv("NCCL_COMM_ID");
  if(!id || comms<1 || comms>4){fprintf(stderr,"sim-collectives: NCCL_COMM_ID unset\n");return 2;}
  ncclComm_t comm[4];
  for(int c=0;c<comms;c++){
    ncclUniqueId unique;memset(&unique,0,sizeof unique);
    /* each communicator its own bootstrap port: the id's port + c */
    char text[NCCL_UNIQUE_ID_BYTES];snprintf(text,sizeof text,"%s",id);
    char *colon=strrchr(text,':');snprintf(colon+1,sizeof text-(size_t)(colon+1-text),"%d",atoi(colon+1)+c);
    snprintf(unique.internal,sizeof unique.internal,"%s",text);
    if(check(ncclCommInitRank(comm+c,2,unique,rank),"ncclCommInitRank"))return 1;
  }
  const size_t sizes[]={6*1024,98*1024,1536*1024};
  int failures=0;
  for(size_t s=0;s<sizeof sizes/sizeof *sizes;s++){
    const size_t n=sizes[s]/sizeof(float);
    float *send,*recv;
    if(check(ncclMemAlloc((void **)&send,n*sizeof *send),"ncclMemAlloc") || check(ncclMemAlloc((void **)&recv,2*n*sizeof *recv),"ncclMemAlloc"))return 1;
    /* the program: two rounds of every communicator's three collectives, captured (nothing moves) */
    if(check(ncclMeshDeclareBegin(),"ncclMeshDeclareBegin"))return 1;
    for(int round=0;round<2;round++)for(int c=0;c<comms;c++)
      if(check(ncclAllReduce(send,recv,n,ncclFloat,ncclSum,comm[c],NULL),"ncclAllReduce (declared)") ||
         check(ncclAllGather(send,recv,n,ncclFloat,comm[c],NULL),"ncclAllGather (declared)") ||
         check(ncclBroadcast(send,recv,n,ncclFloat,round%2,comm[c],NULL),"ncclBroadcast (declared)"))return 1;
    if(check(ncclMeshDeclareEnd(),"ncclMeshDeclareEnd"))return 1;
    for(int round=0;round<rounds;round++)for(int c=0;c<comms;c++){
      struct timespec t0,t1;
      /* AllReduce */
      if(reusable(send) || reusable(recv))return 1;
      for(size_t i=0;i<n;i++)send[i]=value(rank,i,round+c);
      memset(recv,0,n*sizeof *recv);
      clock_gettime(CLOCK_MONOTONIC,&t0);
      if(check(ncclAllReduce(send,recv,n,ncclFloat,ncclSum,comm[c],NULL),"ncclAllReduce"))return 1;
      clock_gettime(CLOCK_MONOTONIC,&t1);
      for(size_t i=0;i<n;i++)if(recv[i]!=value(0,i,round+c)+value(1,i,round+c)){
        fprintf(stderr,"allreduce %zu bytes: [%zu] %g, not %g\n",sizes[s],i,recv[i],value(0,i,round+c)+value(1,i,round+c));failures++;break;}
      const double reduce=(t1.tv_sec-t0.tv_sec)*1e6+(t1.tv_nsec-t0.tv_nsec)/1e3;
      /* AllGather */
      if(reusable(recv))return 1;
      memset(recv,0,2*n*sizeof *recv);
      if(check(ncclAllGather(send,recv,n,ncclFloat,comm[c],NULL),"ncclAllGather"))return 1;
      for(int r=0;r<2;r++)for(size_t i=0;i<n;i++)if(recv[(size_t)r*n+i]!=value(r,i,round+c)){
        fprintf(stderr,"allgather %zu bytes: rank %d [%zu] %g, not %g\n",sizes[s],r,i,recv[(size_t)r*n+i],value(r,i,round+c));failures++;r=2;break;}
      /* Broadcast from root round % 2 */
      const int root=round%2;
      if(reusable(recv))return 1;
      memset(recv,0,n*sizeof *recv);
      if(check(ncclBroadcast(send,recv,n,ncclFloat,root,comm[c],NULL),"ncclBroadcast"))return 1;
      for(size_t i=0;i<n;i++)if(recv[i]!=value(root,i,round+c)){
        fprintf(stderr,"broadcast %zu bytes: [%zu] %g, not %g\n",sizes[s],i,recv[i],value(root,i,round+c));failures++;break;}
      printf("{\"rank\":%d,\"comm\":%d,\"bytes\":%zu,\"round\":%d,\"allreduce_us\":%.1f,\"checked\":%s}\n",rank,c,sizes[s],round,reduce,
             failures?"false":"true");
    }
    if(reusable(send) || reusable(recv))return 1;
    ncclMemFree(send);ncclMemFree(recv);
  }
  for(int c=0;c<comms;c++)if(check(ncclCommDestroy(comm[c]),"ncclCommDestroy"))return 1;
  return failures?1:0;
}
