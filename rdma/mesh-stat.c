// see RDMA-FIRST.md
#include "mesh.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
int main(int argc,char**argv){
  int f=shm_open(argc>1?argv[1]:MESH_NAME,O_RDONLY,MESH_MODE);
  if(f<0){ printf("{\"up\":false}\n"); return 0; }
  struct hdr *h=mmap(NULL,sizeof *h,PROT_READ,MAP_SHARED,f,0);
  if(h==MAP_FAILED||h->magic!=MESH_MAGIC){ printf("{\"up\":false}\n"); return 0; }
  #define A(x) (unsigned long long)atomic_load_explicit(&h->x,memory_order_relaxed)
  printf("{\"up\":true,\"node\":%u,\"pgsz\":%u,\"pool\":%u,\"arena\":%u,"
    "\"client\":%llu,\"sent\":%llu,\"recvd\":%llu,\"up_ms\":%llu,"
    "\"free\":%llu,\"recv\":%llu,\"send\":%llu,\"app\":%llu,"
    "\"sd_free\":%llu,\"sd_recv\":%llu,\"sd_send\":%llu,\"sd_app\":%llu}\n",
    h->node,h->pgsz,h->pool,h->arena,A(client),A(sent),A(recvd),A(up_ms),
    A(mean[FREE]),A(mean[RECV]),A(mean[SEND]),A(mean[APP]),
    A(sd[FREE]),A(sd[RECV]),A(sd[SEND]),A(sd[APP]));
  return 0; }
