
#include "mesh.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
int main(int argc,char**argv){
  static const char *N[NOWN]={"free","recv","send","app"};
  static const char *Q[NRING]={"sub","cmp","rel","ack"};
  const char *name=argc>1?argv[1]:getenv("MESH_REGION");
  if(!name) name=MESH_NAME;
  int f=shm_open(name,O_RDONLY,MESH_MODE);
  struct hdr *h = f<0 ? MAP_FAILED : mmap(NULL,sizeof *h,PROT_READ,MAP_SHARED,f,0);
  if(h==MAP_FAILED||h->magic!=MESH_MAGIC){ printf("{\"up\":false}\n"); return 0; }
  #define A(x) (unsigned long long)atomic_load_explicit(&h->x,memory_order_relaxed)
  uint64_t client=atomic_load_explicit(&h->client,memory_order_relaxed);
  int alive=client && (!kill((pid_t)client,0)||errno==EPERM);
  printf("{\"up\":true,\"region\":\"%s\",\"node\":%u,\"pgsz\":%u,\"pool\":%u,\"arena\":%u,"
    "\"client\":%llu,\"client_alive\":%s,\"sent\":%llu,\"recvd\":%llu,\"bad\":%llu,\"up_ms\":%llu",
    name,h->node,h->pgsz,h->pool,h->arena,(unsigned long long)client,alive?"true":"false",A(sent),A(recvd),A(bad),A(up_ms));
  for(int i=0;i<NRING;i++){
    unsigned long long head=atomic_load_explicit(&h->r[i].head,memory_order_relaxed);
    unsigned long long tail=atomic_load_explicit(&h->r[i].tail,memory_order_relaxed);
    printf(",\"%s_head\":%llu,\"%s_tail\":%llu,\"%s_depth\":%llu",Q[i],head,Q[i],tail,Q[i],head-tail); }
  for(int i=0;i<NOWN;i++) printf(",\"%s\":%llu,\"sd_%s\":%llu",N[i],A(mean[i]),N[i],A(sd[i]));
  printf("}\n"); return 0; }
