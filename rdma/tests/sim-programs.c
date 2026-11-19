/* design/declared-programs.md "Testing": one rank of a two-rank libnccl-mesh communicator over the two-node simulator,
   running the declared programs of the calls that are not plain collectives: a one-way WRITE plan (ncclMeshWritePrepare,
   then ncclMeshWriteExecute, rank 0 to rank 1) and an AlltoAllV of uneven counts, declared once (ncclMeshDeclareBegin,
   the calls, ncclMeshDeclareEnd) and run `rounds` rounds, every value checked; then two communicators' AllReduces
   declared alongside (ncclMeshDeclareAlongside) and run the same way.

   usage: sim-programs RANK [ROUNDS]   (environment: MESH_REGION, NCCL_COMM_ID) */
#include "../nccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(ncclResult_t r,const char *what){
  if(r!=ncclSuccess){fprintf(stderr,"sim-programs: %s: %s (%s)\n",what,ncclGetErrorString(r),ncclGetLastError(NULL));return 1;}
  return 0;
}
static int reusable(const void *buffer){
  for(int ready=0;!ready;)if(check(ncclMeshMemReusable(buffer,&ready),"ncclMeshMemReusable"))return 1;
  return 0;
}
static float value(int rank,size_t i,int round){ return (float)((rank+1)*1000+(int)(i%997)+round); }
static int comm_at(const char *id,int c,int rank,ncclComm_t *comm){
  ncclUniqueId unique;memset(&unique,0,sizeof unique);
  char text[NCCL_UNIQUE_ID_BYTES];snprintf(text,sizeof text,"%s",id);
  char *colon=strrchr(text,':');snprintf(colon+1,sizeof text-(size_t)(colon+1-text),"%d",atoi(colon+1)+c);
  snprintf(unique.internal,sizeof unique.internal,"%s",text);
  return check(ncclCommInitRank(comm,2,unique,rank),"ncclCommInitRank");
}
int main(int argc,char **argv){
  if(argc<2)return 2;
  const int rank=atoi(argv[1]),rounds=argc>2?atoi(argv[2]):6;
  const char *id=getenv("NCCL_COMM_ID");
  if(!id)return 2;
  ncclComm_t comm[2];
  if(comm_at(id,0,rank,comm) || comm_at(id,1,rank,comm+1))return 1;
  size_t page=0;
  if(check(ncclMeshCommPage(comm[0],&page),"ncclMeshCommPage"))return 1;
  const size_t n=3*page/sizeof(float)+77;           /* a WRITE of more than three pieces, its last one short */
  /* AlltoAllV of uneven counts: rank r sends q + 2r + 1 pages' worth to rank q, its segments packed in rank order (each on
     a page, as the call requires) */
  size_t sendcounts[2],recvcounts[2],at=0,in=0;
  for(int q=0;q<2;q++){
    sendcounts[q]=(size_t)(q+2*rank+1)*page/sizeof(float);recvcounts[q]=(size_t)(rank+2*q+1)*page/sizeof(float);
    at+=sendcounts[q];in+=recvcounts[q];
  }
  float *src,*land,*a2s,*a2r,*ar[2],*arr[2];
  if(check(ncclMemAlloc((void **)&src,n*sizeof(float)),"ncclMemAlloc") || check(ncclMemAlloc((void **)&land,n*sizeof(float)),"ncclMemAlloc") ||
     check(ncclMemAlloc((void **)&a2s,at*sizeof(float)),"ncclMemAlloc") || check(ncclMemAlloc((void **)&a2r,in*sizeof(float)),"ncclMemAlloc"))return 1;
  for(int c=0;c<2;c++)
    if(check(ncclMemAlloc((void **)&ar[c],page),"ncclMemAlloc") || check(ncclMemAlloc((void **)&arr[c],page),"ncclMemAlloc"))return 1;
  const size_t packed_s[2]={0,sendcounts[0]},packed_r[2]={0,recvcounts[0]};
  ncclMeshWritePlan_t plan=NULL;
  if(check(ncclMeshWritePrepare(src,land,n,ncclFloat,0,1,comm[0],0,1,0,0,&plan),"ncclMeshWritePrepare"))return 1;
  if(check(ncclMeshDeclareBegin(),"ncclMeshDeclareBegin") ||
     check(ncclMeshWriteExecute(plan,NULL),"ncclMeshWriteExecute (declared)") ||
     check(ncclMeshAlltoAllV(a2s,sendcounts,a2r,recvcounts,ncclFloat,comm[0],NULL),"ncclMeshAlltoAllV (declared)") ||
     check(ncclMeshDeclareEnd(),"ncclMeshDeclareEnd"))return 1;
  int failures=0;
  for(int round=0;round<rounds && !failures;round++){
    if(reusable(src) || reusable(a2s))return 1;
    for(size_t i=0;i<n;i++){src[i]=value(rank,i,round);land[i]=0;}
    for(int q=0;q<2;q++)for(size_t i=0;i<sendcounts[q];i++)a2s[packed_s[q]+i]=value(rank,i+(size_t)q*7,round);
    memset(a2r,0,in*sizeof(float));
    if(check(ncclMeshWriteExecute(plan,NULL),"ncclMeshWriteExecute") ||
       check(ncclMeshAlltoAllV(a2s,sendcounts,a2r,recvcounts,ncclFloat,comm[0],NULL),"ncclMeshAlltoAllV"))return 1;
    if(rank==1)for(size_t i=0;i<n;i++)if(land[i]!=value(0,i,round)){fprintf(stderr,"write round %d: [%zu] %g\n",round,i,land[i]);failures++;break;}
    for(int q=0;q<2;q++)for(size_t i=0;i<recvcounts[q];i++)if(a2r[packed_r[q]+i]!=value(q,i+(size_t)rank*7,round)){
      fprintf(stderr,"alltoallv round %d: from %d [%zu] %g, not %g\n",round,q,i,a2r[packed_r[q]+i],value(q,i+(size_t)rank*7,round));failures++;q=2;break;}
  }
  printf("{\"rank\":%d,\"program\":\"write+alltoallv\",\"rounds\":%d,\"checked\":%s}\n",rank,rounds,failures?"false":"true");
  /* two communicators' AllReduces alongside */
  const size_t m=page/sizeof(float);
  if(check(ncclMeshDeclareBegin(),"ncclMeshDeclareBegin") ||
     check(ncclAllReduce(ar[0],arr[0],m,ncclFloat,ncclSum,comm[0],NULL),"ncclAllReduce (declared)") ||
     check(ncclMeshDeclareAlongside(),"ncclMeshDeclareAlongside") ||
     check(ncclAllReduce(ar[1],arr[1],m,ncclFloat,ncclSum,comm[1],NULL),"ncclAllReduce (declared)") ||
     check(ncclMeshDeclareEnd(),"ncclMeshDeclareEnd"))return 1;
  int bad=0;
  for(int round=0;round<rounds && !bad;round++){
    for(int c=0;c<2;c++){if(reusable(ar[c]))return 1;for(size_t i=0;i<m;i++)ar[c][i]=value(rank,i,round+c);}
    for(int c=0;c<2;c++)if(check(ncclAllReduce(ar[c],arr[c],m,ncclFloat,ncclSum,comm[c],NULL),"ncclAllReduce"))return 1;
    for(int c=0;c<2;c++)for(size_t i=0;i<m;i++)if(arr[c][i]!=value(0,i,round+c)+value(1,i,round+c)){
      fprintf(stderr,"alongside allreduce %d round %d: [%zu] %g\n",c,round,i,arr[c][i]);bad++;c=2;break;}
  }
  printf("{\"rank\":%d,\"program\":\"alongside\",\"rounds\":%d,\"checked\":%s}\n",rank,rounds,bad?"false":"true");
  ncclMeshWritePlanDestroy(plan);
  for(int c=0;c<2;c++)if(check(ncclCommDestroy(comm[c]),"ncclCommDestroy"))return 1;
  return failures||bad?1:0;
}
