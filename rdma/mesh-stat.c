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
/* design/algorithm-sources.md#programcopy */
int main(int argc,char **argv){
  int readiness=argc>1 && !strcmp(argv[1],"--ready");
  const char *name=argc>1+readiness?argv[1+readiness]:getenv("MESH_REGION");
  if(!name) name=MESH_NAME;
  int f=shm_open(name,O_RDONLY,MESH_MODE);
  struct stat info;
  size_t bytes=sizeof(struct hdr);
  struct hdr *h=MAP_FAILED;
  if(f>=0 && !fstat(f,&info) && info.st_size>=(off_t)bytes){bytes=(size_t)info.st_size;h=mmap(NULL,bytes,PROT_READ,MAP_SHARED,f,0);}
  if(h==MAP_FAILED || h->magic!=MESH_MAGIC || h->version!=MESH_VERSION){
    printf("{\"up\":false,\"ready\":false}\n");
    if(h!=MAP_FAILED) munmap(h,bytes);
    if(f>=0) close(f);
    return readiness?1:0;
  }
  uint64_t pid=atomic_load(&h->bridge_pid);
  int alive=pid && (!kill((pid_t)pid,0) || errno==EPERM);
  uint32_t paired=0;
  for(uint32_t i=0;i<h->links;i++)paired+=alive && atomic_load(&mesh_links(h)[i].port.phase)==MESH_PAIRED;
  uint32_t sessions=0,rings=0;
  for(uint32_t s=0;s<MESH_SESSIONS;s++)sessions+=atomic_load(&mesh_sessions(h)[s].served)!=0;
  for(uint32_t r=0;r<MESH_RINGS;r++)rings+=atomic_load(&mesh_rings(h)[r])!=0;
  printf("{\"up\":%s,\"ready\":%s,\"paired\":%s,\"bridge_pid\":%llu,\"sessions\":%u,\"rings\":%u,\"version\":%u,\"node\":%u,\"pgsz\":%u,\"block\":%u,\"rows\":%u,\"qps\":%u,\"links\":%u,\"paired_links\":%u,\"arena_pages\":%u,\"window_pages\":%u,\"code\":%lld,\"domain\":%u,\"peers\":[",
    alive?"true":"false",alive?"true":"false",paired?"true":"false",(unsigned long long)pid,sessions,rings,
    h->version,h->node,h->pgsz,h->block,h->rows,h->qps,h->links,paired,mesh_arena_pages(h),h->wire_pages,(long long)h->port.code,h->port.domain);
  for(uint32_t i=0;i<h->links;i++){
    struct mesh_link_info *link=&mesh_links(h)[i];
    printf("%s{\"node\":%u,\"phase\":%u,\"code\":%lld,\"domain\":%u,\"device\":\"%s\",\"bandwidth\":%llu}",i?",":"",link->peer,atomic_load(&link->port.phase),(long long)link->port.code,link->port.domain,link->device,(unsigned long long)__atomic_load_n(&link->bandwidth,__ATOMIC_RELAXED));
  }
  printf("]}\n");
  munmap(h,bytes); close(f); return readiness && !alive;
}
