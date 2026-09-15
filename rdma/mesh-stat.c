#include "mesh.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
/* ledger D12: out-of-band reader of the region's port metadata. Pairing follows each client (ledger D14), so
   readiness is a running bridge with a valid region, not a paired connection. */
int main(int argc,char **argv){
  int readiness=argc>1 && !strcmp(argv[1],"--ready");
  const char *name=argc>1+readiness?argv[1+readiness]:getenv("MESH_REGION");
  if(!name) name=MESH_NAME;
  int f=shm_open(name,O_RDONLY,MESH_MODE);
  struct stat info;
  size_t bytes=sizeof(struct hdr);
  struct hdr *h=f<0 || fstat(f,&info) || info.st_size<(off_t)bytes?MAP_FAILED:mmap(NULL,bytes,PROT_READ,MAP_SHARED,f,0);
  if(h==MAP_FAILED || h->magic!=MESH_MAGIC || h->version!=MESH_VERSION){
    printf("{\"up\":false,\"ready\":false}\n");
    if(h!=MAP_FAILED) munmap(h,bytes);
    if(f>=0) close(f);
    return readiness?1:0;
  }
  uint64_t pid=atomic_load(&h->bridge_pid);
  int alive=pid && (!kill((pid_t)pid,0) || errno==EPERM);
  int paired=alive && atomic_load(&h->port.phase)==MESH_PAIRED;
  printf("{\"up\":%s,\"ready\":%s,\"paired\":%s,\"bridge_pid\":%llu,\"client\":%llu,\"version\":%u,\"node\":%u,\"pgsz\":%u,\"block\":%u,\"rows\":%u,\"qps\":%u,\"code\":%lld,\"domain\":%u}\n",
    alive?"true":"false",alive?"true":"false",paired?"true":"false",(unsigned long long)pid,(unsigned long long)(uint32_t)atomic_load(&h->client),
    h->version,h->node,h->pgsz,h->block,h->rows,h->qps,(long long)h->port.code,h->port.domain);
  munmap(h,bytes); close(f); return readiness && !alive;
}
