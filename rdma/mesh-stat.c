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
#include <time.h>
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
  for(uint32_t i=0;i<h->links;i++)paired+=alive && atomic_load(&mesh_net_links(h)[i].phase)==MESH_PAIRED;
  printf("{\"up\":%s,\"ready\":%s,\"paired\":%s,\"bridge_pid\":%llu,\"version\":%u,\"node\":%u,\"pgsz\":%u,\"block\":%u,\"pages\":%u,\"links\":%u,\"paired_links\":%u,\"peers\":[",
    alive?"true":"false",alive?"true":"false",paired?"true":"false",(unsigned long long)pid,
    h->version,h->node,h->pgsz,h->block,mesh_arena_pages(h),h->links,paired);
  /* each link's counts (mesh.h mesh_stats_link: flow control, the communicators' traffic, the session's
     pairings, heartbeats, silence and resumptions) read lagged, as every reader reads them: the ring's entry of
     the last evaluation the lag lets this read (none before it); live, only the link's state and when its
     bridge last heard its peer (its liveness), and the device's registered regions */
  const uint64_t ended=atomic_load(&h->evaluations);
  struct mesh_stats_entry *lagged=calloc(1,h->stats_stride?h->stats_stride:sizeof *lagged);
  const int seen=lagged && ended>h->stats_lag && mesh_stats_read(h,ended,ended-h->stats_lag,lagged,1)==1;
  struct timespec clock;clock_gettime(CLOCK_MONOTONIC,&clock);
  const uint64_t now=(uint64_t)clock.tv_sec*1000000000+(uint64_t)clock.tv_nsec;
  for(uint32_t i=0;i<h->links;i++){
    struct mesh_link_info *link=&mesh_links(h)[i];
    struct mesh_net_link *n=mesh_net_links(h)+i;
    const struct mesh_stats_link f=seen && i<lagged->links?lagged->link[i]:(struct mesh_stats_link){0};
    const uint64_t heard=atomic_load(&n->heard_ns);
    printf("%s{\"node\":%u,\"phase\":%u,\"device\":\"%s\",\"bandwidth\":%llu,",i?",":"",link->peer,atomic_load(&n->phase),link->device,
      (unsigned long long)__atomic_load_n(&link->bandwidth,__ATOMIC_RELAXED));
    printf("\"flow\":{\"send_stalls\":%llu,\"receive_stalls\":%llu,\"credit_waits\":%llu,"
      "\"net_sends\":%llu,\"net_send_bytes\":%llu,\"net_receives\":%llu,\"net_receive_bytes\":%llu,\"heartbeats_sent\":%llu,\"heartbeats_heard\":%llu,"
      "\"silence_ns\":%llu,\"resumes\":%llu,\"resends\":%llu,\"reposts\":%llu},",
      (unsigned long long)f.send_stalls,(unsigned long long)f.receive_stalls,(unsigned long long)f.credit_waits,(unsigned long long)f.net_sends,
      (unsigned long long)f.net_send_bytes,(unsigned long long)f.net_receives,(unsigned long long)f.net_receive_bytes,(unsigned long long)f.heartbeats_sent,
      (unsigned long long)f.heartbeats_heard,(unsigned long long)f.silence_ns,(unsigned long long)f.resumes,(unsigned long long)f.resends,
      (unsigned long long)f.reposts);
    printf("\"session\":{\"phase\":%u,\"code\":%lld,\"sessions\":%llu,\"chunk_frames\":%u,\"heard_ms_ago\":%.1f},\"regions\":{\"wire\":%u}}",
      atomic_load(&n->phase),(long long)atomic_load(&n->code),(unsigned long long)f.sessions,atomic_load(&n->chunk_frames),
      heard && now>heard?(double)(now-heard)/1e6:-1.0,atomic_load(&n->wire_regions));
  }
  printf("],\"stats\":{\"evaluations\":%llu,\"lag\":%u,\"entry\":%llu},\"clients\":[",(unsigned long long)ended,h->stats_lag,
    (unsigned long long)(seen?atomic_load(&lagged->evaluation):0));
  free(lagged);
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
    printf("%s{\"comm\":%u,\"state\":\"%s\",\"link\":%u,\"peer\":%u,\"pid\":%u,\"key\":\"%016llx\",\"error\":%d}",
      listed++?",":"",i,states[state],comm->link,comm->link<h->links?mesh_links(h)[comm->link].peer:UINT32_MAX,(uint32_t)comm->owner,(unsigned long long)comm->key,
      atomic_load(&comm->error));
  }
  /* whether a client of the region is alive (its bridge, stopped, keeps the region for it): a communicator client's
     process, or one owning a comm */
  int attached=0;
  for(uint32_t i=0;i<MESH_NET_CLIENTS && !attached;i++){const uint64_t o=atomic_load(&mesh_net_clients(h)[i].owner);attached=o && (!kill((pid_t)(uint32_t)o,0) || errno==EPERM);}
  for(uint32_t i=0;i<MESH_NET_COMMS && !attached;i++){
    const uint32_t state=atomic_load(&mesh_net_comms(h)[i].state);const uint64_t o=mesh_net_comms(h)[i].owner;
    attached=state!=MESH_NET_FREE && state!=MESH_NET_CLAIMED && o && (!kill((pid_t)(uint32_t)o,0) || errno==EPERM);
  }
  printf("],\"attached\":%s}\n",attached?"true":"false");
  munmap(h,bytes); close(f); return readiness && !alive;
}
