// Read one node's live region and print it as JSON. Read-only: it attaches to
#include "mesh.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
int main(int argc,char**argv){
  const char *name = argc>1?argv[1]:"/mesh0";
  int fd=shm_open(name,O_RDONLY,0600);
  if(fd<0){ printf("{\"up\":false}\n"); return 0; }
  struct stat st; fstat(fd,&st);
  void *b=mmap(NULL,sizeof(struct mesh_hdr),PROT_READ,MAP_SHARED,fd,0);
  if(b==MAP_FAILED){ printf("{\"up\":false}\n"); close(fd); return 0; }
  struct mesh_hdr *h=(struct mesh_hdr*)b;
  if(h->magic!=MESH_MAGIC){ printf("{\"up\":false}\n"); return 0; }
  #define A(f) (unsigned long long)atomic_load_explicit(&h->f,memory_order_relaxed)
  printf("{\"up\":true,\"node\":%u,\"pgsz\":%u,\"pages\":%u,\"arena_pages\":%u,"
         "\"bytes\":%llu,\"node_ram\":%llu,\"client\":%llu,"
         "\"sent\":%llu,\"recvd\":%llu,\"uptime_ms\":%llu,"
         "\"pool\":%llu,"
         "\"sd_free\":%llu,\"sd_posted\":%llu,\"sd_sending\":%llu,"
         "\"mean_free\":%llu,\"mean_posted\":%llu,\"mean_sending\":%llu,"
         "\"mean_held\":%llu,\"sd_held\":%llu}\n",
    h->node,h->pgsz,h->npages,h->arena_pages,
    (unsigned long long)h->bytes,(unsigned long long)h->node_ram,A(client_pid),
    A(sent),A(recvd),A(uptime_ms),
    A(occ_pool),
    A(sd_free),A(sd_posted),A(sd_sending),
    A(mean_free),A(mean_posted),A(mean_sending),A(mean_held),A(sd_held));
  munmap(b,sizeof *h); close(fd); return 0;
}
