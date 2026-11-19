/* design/declared-programs.md "Testing": one rank of a two-rank libnccl-mesh communicator over the two-node simulator,
   running ahead: a program of one step, a Broadcast of `bytes` from rank 0, issued CALLS times back to back.  The root's
   call returns at its issue, so it runs CALLS steps ahead of rank 1, whose every call returns once its bytes landed,
   each checked (rank 1 zeroes its buffer between calls: its next receive is posted only at its next call's issue).
   Run beside other process pairs of its own (PROGRAMS of them, each its own program toward the same node, composed by
   rounds), each pair declares, then waits at a barrier of files in BARRIER (its own agreement with the others, ahead of
   any issue: design/declared-programs.md section 4), runs, and waits there again before it ends its program.

   usage: sim-runahead RANK PROGRAM [CALLS [BYTES]]   (environment: MESH_REGION, NCCL_COMM_ID, BARRIER, PROGRAMS) */
#include "../nccl.h"
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int check(ncclResult_t r,const char *what){
  if(r!=ncclSuccess){fprintf(stderr,"sim-runahead: %s: %s (%s)\n",what,ncclGetErrorString(r),ncclGetLastError(NULL));return 1;}
  return 0;
}
/* the barrier `phase`: this process's file made, then every one of the PROGRAMS processes of its rank awaited */
static void barrier(const char *dir,int rank,int program,int programs,const char *phase){
  if(!dir || programs<2)return;
  char path[512];
  snprintf(path,sizeof path,"%s/%s.%d.%d",dir,phase,rank,program);
  const int fd=open(path,O_CREAT|O_WRONLY,0600);
  if(fd>=0)close(fd);
  char prefix[64];snprintf(prefix,sizeof prefix,"%s.%d.",phase,rank);
  for(int seen=0;seen<programs;){
    seen=0;
    DIR *d=opendir(dir);
    for(struct dirent *e;d && (e=readdir(d));)seen+=!strncmp(e->d_name,prefix,strlen(prefix));
    if(d)closedir(d);
    if(seen<programs)usleep(1000);
  }
}
static unsigned char value(size_t i,int program){ return (unsigned char)(i*31+7+(size_t)program*101); }
int main(int argc,char **argv){
  if(argc<3)return 2;
  const int rank=atoi(argv[1]),program=atoi(argv[2]);
  const long calls=argc>3?atol(argv[3]):12000;
  const size_t bytes=argc>4?(size_t)atol(argv[4]):6144;
  const char *id=getenv("NCCL_COMM_ID"),*dir=getenv("BARRIER");
  const int programs=getenv("PROGRAMS")?atoi(getenv("PROGRAMS")):1;
  if(!id){fprintf(stderr,"sim-runahead: NCCL_COMM_ID unset\n");return 2;}
  ncclUniqueId unique;memset(&unique,0,sizeof unique);snprintf(unique.internal,sizeof unique.internal,"%s",id);
  ncclComm_t comm;
  if(check(ncclCommInitRank(&comm,2,unique,rank),"ncclCommInitRank"))return 1;
  unsigned char *x;
  if(check(ncclMemAlloc((void **)&x,bytes),"ncclMemAlloc"))return 1;
  for(size_t i=0;i<bytes;i++)x[i]=rank?0:value(i,program);
  if(check(ncclMeshDeclareBegin(),"ncclMeshDeclareBegin") || check(ncclBroadcast(x,x,bytes,ncclUint8,0,comm,NULL),"ncclBroadcast (declared)") ||
     check(ncclMeshDeclareEnd(),"ncclMeshDeclareEnd"))return 1;
  barrier(dir,rank,program,programs,"declared");
  struct timespec t0,t1;
  clock_gettime(CLOCK_MONOTONIC,&t0);
  long failures=0;
  for(long k=0;k<calls;k++){
    if(check(ncclBroadcast(x,x,bytes,ncclUint8,0,comm,NULL),"ncclBroadcast"))return 1;
    if(rank){
      for(size_t i=0;i<bytes;i++)if(x[i]!=value(i,program)){if(!failures++)fprintf(stderr,"broadcast %ld: [%zu] %u, not %u\n",k,i,x[i],value(i,program));break;}
      memset(x,0,bytes);
    }
  }
  clock_gettime(CLOCK_MONOTONIC,&t1);
  printf("{\"rank\":%d,\"program\":%d,\"calls\":%ld,\"bytes\":%zu,\"us_a_call\":%.2f,\"checked\":%s}\n",rank,program,calls,bytes,
         ((t1.tv_sec-t0.tv_sec)*1e6+(t1.tv_nsec-t0.tv_nsec)/1e3)/(double)calls,failures?"false":"true");
  barrier(dir,rank,program,programs,"ran");
  for(int ready=0;!ready;)if(check(ncclMeshMemReusable(x,&ready),"ncclMeshMemReusable"))return 1;
  if(check(ncclCommDestroy(comm),"ncclCommDestroy"))return 1;
  ncclMemFree(x);
  return failures?1:0;
}
