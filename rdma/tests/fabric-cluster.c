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

/* every ordered pair's one-way WRITE, each declared from its source to its destination alone: the bridges switch its
   pieces over the cables and relays as they move (no route is installed) */
static void writes(ncclComm_t comm,const struct mesh_fabric *f,int rank,size_t count,int *input){
  assert(f->nodes<=8);
  const int pairs=(int)(f->nodes*(f->nodes-1));
  ncclMeshCall calls[pairs];
  enum { iterations=12 };
  int *landing[iterations][pairs], *sources[iterations], at=0;
  for(int t=0;t<iterations;t++)checked(ncclMemAlloc((void **)&sources[t],count*sizeof *input));
  for(uint32_t source=0;source<f->nodes;source++)for(uint32_t destination=0;destination<f->nodes;destination++){
    if(source==destination)continue;
    calls[at]=(ncclMeshCall){.what=ncclMeshCallWrite,.count=count,.type=ncclInt32,.peer=(int)source,
      .destination=(int)destination,.group=0,.repeat=1};
    for(int t=0;t<iterations;t++)checked(ncclMemAlloc((void **)&landing[t][at],count*sizeof *input));at++;
  }
  checked(ncclMeshCommPlan(comm,calls,pairs,NULL));
  /* each iteration's writes prepared once (their bindings: configuration), then declared as the program's steps, one
     group an iteration on its own buffers (design/declared-programs.md), and run as declared */
  ncclMeshWritePlan_t plan[iterations][pairs];
  for(int t=0;t<iterations;t++)for(int k=0;k<pairs;k++){
    plan[t][k]=NULL;
    checked(ncclMeshWritePrepare(sources[t],landing[t][k],count,ncclInt32,calls[k].peer,calls[k].destination,comm,0,1,0,0,&plan[t][k]));
  }
  checked(ncclMeshDeclareBegin());
  for(int t=0;t<iterations;t++){
    checked(ncclGroupStart());
    for(int k=0;k<pairs;k++)checked(ncclMeshWriteExecute(plan[t][k],NULL));
    checked(ncclGroupEnd());
  }
  checked(ncclMeshDeclareEnd());
  for(int t=0;t<iterations;t++){
    for(size_t i=0;i<count;i++)sources[t][i]=(int)(i%997)+rank*1000+t*10000;
    checked(ncclGroupStart());
    for(int k=0;k<pairs;k++)checked(ncclMeshWriteExecute(plan[t][k],NULL));
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
  checked(ncclMeshDeclareBegin());checked(ncclAllReduce(done,joined,1,ncclInt32,ncclSum,comm,NULL));checked(ncclMeshDeclareEnd());
  checked(ncclAllReduce(done,joined,1,ncclInt32,ncclSum,comm,NULL));
  assert(*joined==(int)f->nodes);
  for(int t=0;t<iterations;t++)for(int k=0;k<pairs;k++)checked(ncclMeshWritePlanDestroy(plan[t][k]));
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
  checked(ncclMeshDeclareBegin());checked(ncclAllReduce(input,output,count,ncclInt32,ncclSum,comm,NULL));checked(ncclMeshDeclareEnd());
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
  struct ncclMeshPeerSample samples[64];int n;
  checked(ncclMeshCommSample(comm,samples,64,&n));
  const double seconds=end.tv_sec-start.tv_sec+(end.tv_nsec-start.tv_nsec)*1e-9;
  printf("{\"rank\":%d,\"elements\":%zu,\"iterations\":%d,\"seconds\":%.9f,\"verified\":true,\"edges\":[",rank,count,iterations,seconds);
  for(int e=0;e<n;e++)printf("%s{\"link\":%u,\"pair\":\"%016llx\",\"sent\":%llu,\"landed\":%llu}",e?",":"",samples[e].channel,
    (unsigned long long)samples[e].pair,(unsigned long long)samples[e].sent,(unsigned long long)samples[e].landed);
  puts("]}");fflush(stdout);
  writes(comm,fabric,rank,count,input);
  checked(ncclCommDestroy(comm));
  checked(ncclMemFree(input));checked(ncclMemFree(output));
}
