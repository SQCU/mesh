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
  printf("{\"up\":%s,\"ready\":%s,\"paired\":%s,\"bridge_pid\":%llu,\"client\":%llu,\"version\":%u,\"node\":%u,\"pgsz\":%u,\"block\":%u,\"rows\":%u,\"qps\":%u,\"links\":%u,\"paired_links\":%u,\"code\":%lld,\"domain\":%u,\"peers\":[",
    alive?"true":"false",alive?"true":"false",paired?"true":"false",(unsigned long long)pid,(unsigned long long)(uint32_t)atomic_load(&h->client),
    h->version,h->node,h->pgsz,h->block,h->rows,h->qps,h->links,paired,(long long)h->port.code,h->port.domain);
  /* each link's live counts (mesh.h mesh_net_link): flow control, both paths' traffic, the communicator
     session and the device's registered regions */
  for(uint32_t i=0;i<h->links;i++){
    struct mesh_link_info *link=&mesh_links(h)[i];
    struct mesh_net_link *n=mesh_net_links(h)+i;
    printf("%s{\"node\":%u,\"phase\":%u,\"code\":%lld,\"domain\":%u,\"device\":\"%s\",\"bandwidth\":%llu,",i?",":"",link->peer,atomic_load(&link->port.phase),(long long)link->port.code,link->port.domain,link->device,(unsigned long long)__atomic_load_n(&link->bandwidth,__ATOMIC_RELAXED));
    printf("\"flow\":{\"send_stalls\":%llu,\"receive_stalls\":%llu,\"credit_waits\":%llu,\"sends\":%llu,\"send_bytes\":%llu,\"receives\":%llu,\"receive_bytes\":%llu,"
      "\"net_sends\":%llu,\"net_send_bytes\":%llu,\"net_receives\":%llu,\"net_receive_bytes\":%llu},",
      (unsigned long long)atomic_load(&n->send_stalls),(unsigned long long)atomic_load(&n->receive_stalls),(unsigned long long)atomic_load(&n->credit_waits),
      (unsigned long long)atomic_load(&n->sends),(unsigned long long)atomic_load(&n->send_bytes),(unsigned long long)atomic_load(&n->receives),(unsigned long long)atomic_load(&n->receive_bytes),
      (unsigned long long)atomic_load(&n->net_sends),(unsigned long long)atomic_load(&n->net_send_bytes),(unsigned long long)atomic_load(&n->net_receives),(unsigned long long)atomic_load(&n->net_receive_bytes));
    printf("\"session\":{\"phase\":%u,\"code\":%lld,\"sessions\":%llu,\"chunk_frames\":%u},\"regions\":{\"wire\":%u}}",
      atomic_load(&n->phase),(long long)atomic_load(&n->code),(unsigned long long)atomic_load(&n->sessions),atomic_load(&n->chunk_frames),
      atomic_load(&n->wire_regions));
  }
  printf("],\"clients\":[");
  uint32_t listed=0;
  for(uint32_t i=0;i<MESH_NET_CLIENTS;i++){
    uint64_t owner=atomic_load(&mesh_net_clients(h)[i].owner);
    if(owner)printf("%s%u",listed++?",":"",(uint32_t)owner);
  }
  uint64_t client_bytes=0;
  for(uint32_t i=0;i<MESH_NET_MEMORY;i++)if(atomic_load(&mesh_net_memory(h)[i].owner))client_bytes+=(uint64_t)mesh_net_memory(h)[i].pages*h->pgsz;
  printf("],\"client_bytes\":%llu",(unsigned long long)client_bytes);
  static const char *const states[]={"free","claimed","listen","connecting","acceptable","send","recv","closing","failed"};
  printf(",\"communicators\":[");
  listed=0;
  for(uint32_t i=0;i<MESH_NET_COMMS;i++){
    struct mesh_net_comm *comm=mesh_net_comms(h)+i;
    uint32_t state=atomic_load(&comm->state);
    if(state==MESH_NET_FREE || state==MESH_NET_CLAIMED || state>MESH_NET_FAILED)continue;
    printf("%s{\"comm\":%u,\"state\":\"%s\",\"link\":%u,\"peer\":%u,\"pid\":%u,\"key\":\"%016llx\",\"posted\":%llu,\"completions\":%llu,\"bytes\":%llu,\"credit_waits\":%llu,\"error\":%d}",
      listed++?",":"",i,states[state],comm->link,comm->link<h->links?mesh_links(h)[comm->link].peer:UINT32_MAX,(uint32_t)comm->owner,(unsigned long long)comm->key,
      (unsigned long long)atomic_load(&comm->posted),(unsigned long long)atomic_load(&comm->completions),(unsigned long long)atomic_load(&comm->bytes),
      (unsigned long long)atomic_load(&comm->credit_waits),atomic_load(&comm->error));
  }
  printf("]}\n");
  munmap(h,bytes); close(f); return readiness && !alive;
}
