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
  int f=shm_open(MESH_NAME,O_RDONLY,MESH_MODE);
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
  uint32_t paired=0,clients=0;
  for(uint32_t i=0;i<h->links;i++)paired+=alive && atomic_load(&mesh_links(h)[i].phase)==MESH_PAIRED;
  for(uint32_t s=0;s<h->clients;s++)clients+=atomic_load(&((struct mesh_client *)((char *)h+h->client_off))[s].owner)!=0;
  printf("{\"up\":%s,\"ready\":%s,\"paired\":%s,\"bridge_pid\":%llu,\"clients\":%u,\"version\":%u,\"node\":%u,\"pgsz\":%u,\"block\":%u,\"rows\":%u,\"links\":%u,\"paired_links\":%u,\"peers\":[",
    alive?"true":"false",alive?"true":"false",paired?"true":"false",(unsigned long long)pid,clients,h->version,h->node,h->pgsz,h->block,h->rows,h->links,paired);
  for(uint32_t i=0,listed=0;i<h->links;i++){
    struct mesh_link_info *link=&mesh_links(h)[i];
    if(!link->device[0])continue;
    printf("%s{\"node\":%u,\"phase\":%u,\"code\":%lld,\"device\":\"%s\",\"bandwidth\":%llu,\"connection\":%llu}",listed++?",":"",atomic_load(&link->peer),
      atomic_load(&link->phase),(long long)atomic_load(&link->code),link->device,(unsigned long long)atomic_load(&link->bandwidth),(unsigned long long)atomic_load(&link->connection));
  }
  printf("]}\n");
  munmap(h,bytes); close(f); return readiness && !alive;
}
