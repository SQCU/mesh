// see RDMA-FIRST.md
#include "mesh.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
static double now(void){ struct timeval t; gettimeofday(&t,NULL); return t.tv_sec+t.tv_usec/1e6; }
int main(int argc,char**argv){
  size_t N=(size_t)((argc>3?atof(argv[3]):20.0)*1e9);
  uint64_t *a=malloc(N); if(!a){ fprintf(stderr,"malloc\n"); return 1; }
  if(argc>1 && !strcmp(argv[1],"yell")){
    for(size_t i=0;i<N/8;i++) a[i]=i;
    double t0=now(); size_t s=mesh_yell(a,N,argc>2?atoi(argv[2]):1); double dt=now()-t0;
    printf("yell %.2f GB in %.2f s = %.2f GB/s\n",s/1e9,dt,s/dt/1e9);
  } else {
    memset(a,0,N);
    double t0=now();
    size_t g=mesh_lissen(a,N);
    double dt=now()-t0;
    size_t bad=0;
    for(size_t i=0;i<N/8;i+=131072) if(a[i]!=i) bad++;
    printf("lissen %.2f GB in %.2f s = %.2f GB/s, sampled wrong %zu\n",g/1e9,dt,g/dt/1e9,bad);
  }
  return 0; }
