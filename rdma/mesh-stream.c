// see RDMA-FIRST.md
#include "mesh.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
static double now(void){ struct timeval t; gettimeofday(&t,NULL); return t.tv_sec+t.tv_usec/1e6; }
int main(int argc,char**argv){
  const char *mode=argc>1?argv[1]:"";
  int node=argc>2?atoi(argv[2]):1;
  size_t N=(size_t)((argc>3?atof(argv[3]):20.0)*1e9);
  uint64_t *a=malloc(N); if(!a){ fprintf(stderr,"malloc\n"); return 1; }
  if(!strcmp(mode,"yell")){
    for(size_t i=0;i<N/8;i++) a[i]=i;
    double t0=now(); size_t s=mesh_yell(a,N,node); double dt=now()-t0;
    printf("yell %.2f GB in %.2f s = %.2f GB/s\n",s/1e9,dt,s/dt/1e9);
  } else if(!strcmp(mode,"lissen")){
    double t0=now(); size_t g=mesh_lissen(a,N); double dt=now()-t0;
    size_t bad=0;
    for(size_t i=0;i<N/8;i+=131072) if(a[i]!=i) bad++;
    printf("lissen %.2f GB in %.2f s = %.2f GB/s, sampled wrong %zu\n",g/1e9,dt,g/dt/1e9,bad);
  } else if(!strcmp(mode,"map")){
    struct mstream in, out, *v[2]={&in,&out};
    mesh_lissen_start(&in,a,N,1);
    while(in.st==MS_RUN) mesh_poll(v,1);
    for(size_t i=0;i<N/8;i++) a[i]*=2;
    mesh_yell_start(&out,a,N,in.node,101);
    while(out.st==MS_RUN) mesh_poll(&v[1],1);
    printf("map: in %.2f GB out %s\n",in.done/1e9,out.st==MS_DONE?"confirmed":"FAILED");
  } else if(!strcmp(mode,"mapreduce")){
    uint64_t *b=malloc(N); if(!b){ fprintf(stderr,"malloc\n"); return 1; }
    for(size_t i=0;i<N/8;i++) a[i]=i;
    int nodes[1]={node};
    struct mstream sc[1], ga[1], *v[2];
    double t0=now();
    mesh_gather(ga,b,N,1,101);
    mesh_scatter(sc,a,N,nodes,1,1);
    v[0]=&sc[0]; v[1]=&ga[0];
    while(mesh_poll(v,2)<2){}
    double dt=now()-t0;
    size_t bad=0;
    for(size_t i=0;i<N/8;i+=131072) if(b[i]!=2*a[i]) bad++;
    printf("mapreduce %.2f GB out + %.2f GB back in %.2f s = %.2f GB/s aggregate, sampled wrong %zu\n",
           N/1e9,ga[0].done/1e9,dt,(N+ga[0].done)/dt/1e9,bad);
  } else { fprintf(stderr,"usage: mesh-stream yell|lissen|map|mapreduce <node> [GB]\n"); return 2; }
  return 0; }
