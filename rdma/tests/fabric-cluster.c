#include "nccl.h"
#include "mesh-plan.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>

static void checked(ncclResult_t result){
  if(result){fprintf(stderr,"%s\n",ncclGetLastError(NULL));exit(2);}
}

static void paths(const struct mesh_fabric *f,uint32_t at,uint32_t destination,uint32_t seen,uint32_t *walk,
                  uint32_t depth,uint32_t *first,uint32_t *cable,double *weights,uint32_t *count){
  if(at==destination){
    assert(*count<256);
    memcpy(cable+first[*count],walk,depth*sizeof *walk);
    first[*count+1]=first[*count]+depth;
    weights[*count]=log(depth==1?1.0:7.0);
    (*count)++;
    return;
  }
  for(uint32_t e=0;e<f->cables;e++){
    const struct mesh_cable *c=f->cable+e;
    if(c->node[0]!=at && c->node[1]!=at)continue;
    const uint32_t other=c->node[c->node[0]==at];
    if(seen&(1u<<other))continue;
    walk[depth]=e;
    paths(f,other,destination,seen|(1u<<other),walk,depth+1,first,cable,weights,count);
  }
}

static void writes(ncclComm_t comm,const struct mesh_fabric *f,int rank,size_t count,int *input){
  assert(f->nodes<=8);
  const int pairs=(int)(f->nodes*(f->nodes-1));
  ncclMeshCall calls[pairs];
  enum { iterations=12 };
  int *landing[iterations][pairs], *sources[iterations], at=0;
  for(int t=0;t<iterations;t++)checked(ncclMemAlloc((void **)&sources[t],count*sizeof *input));
  for(uint32_t source=0;source<f->nodes;source++)for(uint32_t destination=0;destination<f->nodes;destination++){
    if(source==destination)continue;
    uint32_t first[257]={0},cable[2048],walk[8],n=0;
    double weights[256];
    paths(f,source,destination,1u<<source,walk,0,first,cable,weights,&n);
    const struct mesh_paths routes={n,first,cable,weights};
    checked(ncclMeshCommWriteRoutes(comm,(int)source,(int)destination,&routes));
    calls[at]=(ncclMeshCall){.what=ncclMeshCallWrite,.count=count,.type=ncclInt32,.peer=(int)source,
      .destination=(int)destination,.group=0,.repeat=1};
    for(int t=0;t<iterations;t++)checked(ncclMemAlloc((void **)&landing[t][at],count*sizeof *input));at++;
  }
  checked(ncclMeshCommPlan(comm,calls,pairs,NULL));
  for(int t=0;t<iterations;t++){
    for(size_t i=0;i<count;i++)sources[t][i]=(int)(i%997)+rank*1000+t*10000;
    checked(ncclGroupStart());
    for(int k=0;k<pairs;k++)checked(ncclMeshWrite(sources[t],landing[t][k],count,ncclInt32,calls[k].peer,calls[k].destination,comm,NULL));
    checked(ncclGroupEnd());
    for(int k=0;k<pairs;k++)if(calls[k].destination==rank)for(size_t i=0;i<count;i++){
      const int expected=(int)(i%997)+calls[k].peer*1000+t*10000;
      if(landing[t][k][i]!=expected){
        fprintf(stderr,"write %d->%d, iteration %d, element %zu: %d != %d\n",calls[k].peer,rank,t,i,landing[t][k][i],expected);exit(4);
      }
    }
  }
  /* A full-program rendezvous: every destination validated its output before any
     rank releases a source or relay lease. There is no per-send completion wait. */
  int *done,*joined;
  checked(ncclMemAlloc((void **)&done,sizeof *done));
  checked(ncclMemAlloc((void **)&joined,sizeof *joined));
  *done=1;
  const ncclMeshCall end={.what=MESH_ALLREDUCE,.count=1,.type=ncclInt32,.op=ncclSum,.repeat=1};
  checked(ncclMeshCommPlan(comm,&end,1,NULL));
  checked(ncclAllReduce(done,joined,1,ncclInt32,ncclSum,comm,NULL));
  assert(*joined==(int)f->nodes);
  for(int t=0;t<iterations;t++){
    checked(ncclMemFree(sources[t]));
    for(int k=0;k<pairs;k++)checked(ncclMemFree(landing[t][k]));
  }
  checked(ncclMemFree(done));checked(ncclMemFree(joined));
  printf("{\"rank\":%d,\"one_way_all_pairs\":%d,\"iterations\":12,\"verified\":true}\n",rank,pairs);fflush(stdout);
}

int main(int argc,char **argv){
  if(argc!=5)return 1;
  const int rank=atoi(argv[1]),ranks=atoi(argv[2]);
  const size_t count=strtoull(argv[4],NULL,10);
  ncclUniqueId id={0};
  memcpy(&id,argv[3],strlen(argv[3]));
  ncclComm_t comm;
  checked(ncclCommInitRank(&comm,ranks,id,rank));
  const struct mesh_fabric *fabric;
  checked(ncclMeshCommFabric(comm,&fabric));
  unsigned degree=0;
  for(unsigned e=0;e<fabric->cables;e++)degree+=fabric->cable[e].node[0]==(unsigned)rank || fabric->cable[e].node[1]==(unsigned)rank;
  printf("{\"rank\":%d,\"cables\":%u,\"degree\":%u}\n",rank,fabric->cables,degree);fflush(stdout);
  int *input,*output;
  checked(ncclMemAlloc((void **)&input,count*sizeof *input));
  checked(ncclMemAlloc((void **)&output,count*sizeof *output));
  const ncclMeshCall calls[]={{.what=MESH_ALLREDUCE,.count=count,.type=ncclInt32,.op=ncclSum,.group=0,.repeat=1}};
  checked(ncclMeshCommPlan(comm,calls,1,NULL));
  const int iterations=4;
  struct timespec start,end;
  clock_gettime(CLOCK_MONOTONIC,&start);
  for(int t=0;t<iterations;t++){
    for(size_t i=0;i<count;i++)input[i]=(int)(i%113)+rank+t;
    checked(ncclAllReduce(input,output,count,ncclInt32,ncclSum,comm,NULL));
    for(size_t i=0;i<count;i++){
      const int expected=ranks*((int)(i%113)+t)+ranks*(ranks-1)/2;
      if(output[i]!=expected){fprintf(stderr,"rank %d, iteration %d, element %zu: %d != %d\n",rank,t,i,output[i],expected);return 3;}
    }
  }
  clock_gettime(CLOCK_MONOTONIC,&end);
  struct ncclMeshCableSample samples[64];int n;
  checked(ncclMeshCommSample(comm,samples,64,&n));
  const double seconds=end.tv_sec-start.tv_sec+(end.tv_nsec-start.tv_nsec)*1e-9;
  printf("{\"rank\":%d,\"elements\":%zu,\"iterations\":%d,\"seconds\":%.9f,\"verified\":true,\"edges\":[",rank,count,iterations,seconds);
  for(int e=0;e<n;e++)printf("%s{\"cable\":%u,\"wire\":\"%016llx\",\"sent\":%llu,\"landed\":%llu}",e?",":"",samples[e].cable,
    (unsigned long long)samples[e].wire,(unsigned long long)samples[e].sent,(unsigned long long)samples[e].landed);
  puts("]}");fflush(stdout);
  writes(comm,fabric,rank,count,input);
  checked(ncclCommDestroy(comm));
  checked(ncclMemFree(input));checked(ncclMemFree(output));
}
