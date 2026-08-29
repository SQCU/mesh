// see RDMA-FIRST.md
#include "mesh.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
int main(int argc,char**argv){
  static const char *N[NOWN]={"free","recv","send","app"};
  int f=shm_open(argc>1?argv[1]:MESH_NAME,O_RDONLY,MESH_MODE);
  struct hdr *h = f<0 ? MAP_FAILED : mmap(NULL,sizeof *h,PROT_READ,MAP_SHARED,f,0);
  if(h==MAP_FAILED||h->magic!=MESH_MAGIC){ printf("{\"up\":false}\n"); return 0; }
  #define A(x) (unsigned long long)atomic_load_explicit(&h->x,memory_order_relaxed)
  printf("{\"up\":true,\"node\":%u,\"pgsz\":%u,\"pool\":%u,\"arena\":%u,"
    "\"client\":%llu,\"sent\":%llu,\"recvd\":%llu,\"up_ms\":%llu",
    h->node,h->pgsz,h->pool,h->arena,A(client),A(sent),A(recvd),A(up_ms));
  for(int i=0;i<NOWN;i++) printf(",\"%s\":%llu,\"sd_%s\":%llu",N[i],A(mean[i]),N[i],A(sd[i]));
  printf("}\n"); return 0; }
