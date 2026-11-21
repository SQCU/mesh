/* The all-reduce and the all-gather timed over libnccl-mesh at each size, one JSON line a (collective, size) at every
   rank (design/switched-multipath-dma.md "Deploy by git, then measure"): each collective declared once as its program
   (ncclMeshCommPlan, ncclMeshDeclareBegin/End), run twice untimed, then CALLS times timed back to back, its result
   checked exactly (int32 sums and gathered values); microseconds a call, the algorithm's bandwidth (bytes a rank's
   operand over the time, GB/s) and the bus bandwidth (NCCL's: the all-reduce 2 (n-1)/n of it, the all-gather (n-1)/n
   of the gathered bytes), with the median of LATENCY single calls each between two barriers' worth of waiting (an
   all-reduce of one word).

   RELAY (a rank number, -1 none) is a rank that contributes nothing and receives nothing (nccl.h ncclMeshCall
   contributors and recipients): the compiled routing's way to put a relay node such as the Mac mini on routes (it
   takes part as a rank); the switched bridges relay by name with no client, so the switched run leaves the relay out
   of the communicator.

   usage: collectives-bench RANK RANKS ID RELAY [BYTES ...]   (ID: the bootstrap root's host:port; BYTES: the
          collective's operand per rank, default 6144 24576 98304 393216 1572864 3145728 12582912) */
#include "nccl.h"
#include "mesh-plan.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void checked(ncclResult_t result,const char *what){
  if(result){fprintf(stderr,"collectives-bench: %s: %s\n",what,ncclGetLastError(NULL));exit(2);}
}
static double now_s(void){ struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9; }
static int cmp(const void *a,const void *b){ const double x=*(const double *)a,y=*(const double *)b; return (x>y)-(x<y); }

int main(int argc,char **argv){
  if(argc<5){fprintf(stderr,"usage: collectives-bench RANK RANKS ID RELAY [BYTES ...]\n");return 1;}
  const int rank=atoi(argv[1]),ranks=atoi(argv[2]),relay=atoi(argv[4]);
  ncclUniqueId id={0};memcpy(&id,argv[3],strlen(argv[3])<sizeof id?strlen(argv[3]):sizeof id-1);
  ncclComm_t comm;
  checked(ncclCommInitRank(&comm,ranks,id,rank),"ncclCommInitRank");
  static const size_t defaults[]={6144,24576,98304,393216,1572864,3145728,12582912};
  size_t sizes[32];int n=0;
  for(int i=5;i<argc && n<32;i++)sizes[n++]=strtoull(argv[i],NULL,10);
  if(!n)for(size_t i=0;i<sizeof defaults/sizeof *defaults;i++)sizes[n++]=defaults[i];
  uint64_t members[1]={0};int working=0;
  for(int r=0;r<ranks;r++)if(r!=relay){members[0]|=UINT64_C(1)<<r;working++;}
  const int works=rank!=relay;
  int *word,*joined;
  checked(ncclMemAlloc((void **)&word,sizeof *word),"ncclMemAlloc");checked(ncclMemAlloc((void **)&joined,sizeof *joined),"ncclMemAlloc");
  const ncclMeshCall barrier={.what=MESH_ALLREDUCE,.count=1,.type=ncclInt32,.op=ncclSum,.group=1,.repeat=1};   /* a group of its own */
  for(int k=0;k<n;k++)for(int what=0;what<2;what++){
    const size_t bytes=sizes[k]/4*4,count=bytes/4;
    if(!count)continue;
    int *input,*output;
    checked(ncclMemAlloc((void **)&input,bytes),"ncclMemAlloc");
    checked(ncclMemAlloc((void **)&output,what?bytes*ranks:bytes),"ncclMemAlloc");
    const ncclMeshCall call={.what=what?MESH_ALLGATHER:MESH_ALLREDUCE,.count=count,.type=ncclInt32,.op=ncclSum,.repeat=1,
                             .contributors=relay>=0?members:NULL,.recipients=relay>=0?members:NULL};
    const ncclMeshCall both[2]={call,barrier};
    checked(ncclMeshCommPlan(comm,both,2,NULL),"ncclMeshCommPlan");
    for(size_t i=0;i<count;i++)input[i]=works?(int)(i%1009)+rank*7:0;
    #define RUN() (what?ncclAllGather(input,output,count,ncclInt32,comm,NULL):ncclAllReduce(input,output,count,ncclInt32,ncclSum,comm,NULL))
    checked(ncclMeshDeclareBegin(),"ncclMeshDeclareBegin");
    checked(RUN(),"declaring the collective");checked(ncclAllReduce(word,joined,1,ncclInt32,ncclSum,comm,NULL),"declaring the barrier");
    checked(ncclMeshDeclareEnd(),"ncclMeshDeclareEnd");
    /* the collective then the barrier, alternately, as declared */
    for(int w=0;w<2;w++){checked(RUN(),"warm");checked(ncclAllReduce(word,joined,1,ncclInt32,ncclSum,comm,NULL),"barrier");}
    int calls=(int)(UINT64_C(268435456)/(bytes*(what?ranks:2)));calls=calls<8?8:calls>256?256:calls;
    double total=0,once[64];int latency=0;
    for(int c=0;c<calls;c++){
      const double a=now_s();checked(RUN(),"timed");const double b=now_s();
      checked(ncclAllReduce(word,joined,1,ncclInt32,ncclSum,comm,NULL),"barrier");
      total+=b-a;if(latency<64)once[latency++]=b-a;
    }
    qsort(once,(size_t)latency,sizeof *once,cmp);
    int bad=0;
    if(works){
      if(!what)for(size_t i=0;i<count && !bad;i++){
        long want=0;for(int r=0;r<ranks;r++)if(r!=relay)want+=(long)(i%1009)+r*7;
        bad=output[i]!=(int)want;
      }else for(int r=0;r<ranks && !bad;r++)for(size_t i=0;i<count && r!=relay;i++)if(output[(size_t)r*count+i]!=(int)(i%1009)+r*7){bad=1;break;}
    }
    const double us=total/calls*1e6,alg=bytes/(total/calls)/1e9,bus=what?alg*(working-1):alg*2*(working-1)/working;
    printf("{\"rank\":%d,\"ranks\":%d,\"relay\":%d,\"what\":\"%s\",\"bytes\":%zu,\"calls\":%d,\"us_per_call\":%.2f,\"median_us\":%.2f,"
           "\"algorithm_gbps\":%.3f,\"bus_gbps\":%.3f,\"verified\":%s}\n",rank,ranks,relay,what?"all_gather":"all_reduce",bytes,calls,us,
           once[latency/2]*1e6,alg,bus,bad?"false":"true");
    fflush(stdout);
    if(bad){fprintf(stderr,"collectives-bench: rank %d: %s of %zu bytes: wrong result\n",rank,what?"all-gather":"all-reduce",bytes);return 3;}
    checked(ncclMemFree(input),"ncclMemFree");checked(ncclMemFree(output),"ncclMemFree");
  }
  checked(ncclCommDestroy(comm),"ncclCommDestroy");
  return 0;
}
