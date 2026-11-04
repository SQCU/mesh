#include "mesh.h"
#include "mesh-holders.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
/* tag pre-return-20261103 etc/bridge/layout.h MESH_PLATFORM_NS, its value (mesh-bridge.h holds it for the bridge, whose
   header this tool does not include): the bound of the map's read (map_read) */
#ifndef MESH_PLATFORM_NS
#define MESH_PLATFORM_NS UINT64_C(5000000000)
#endif
/* tag pre-return-20261103 rdma/mesh-dataflow.c mesh_records_read (a3c8df9e, c119d3e2), adapted: the region mapped here
   read-only, the bound a deadline of its own, the serial read with the records */
/* the map's records as one serial left them, the seqlock's reader (mesh.h mesh_map_whole): a copy in *out the caller
   frees, its count, and in *serial the map serial they are at (hdr `map_serial`, moved only while `map` is odd); or
   -errno: ESRCH where no bridge runs (mesh-holders.h mesh_gone, as at tag rdma/mesh-stat.c:45, 3f72cbde 5dcb4f69),
   ETIMEDOUT where no whole read came within MESH_PLATFORM_NS of its start (the tag's deadline there; its expiry hook,
   libmesh's, is not this tool's), EINVAL where the map lies outside the region, ENOMEM */
static int map_read(struct hdr *m,size_t length,struct mesh_record **out,uint64_t *serial){
  const uint64_t until=clock_gettime_nsec_np(CLOCK_MONOTONIC)+MESH_PLATFORM_NS;
  for(;;){
    if(mesh_gone((pid_t)atomic_load(&m->bridge_pid)))return -ESRCH;
    uint64_t before=atomic_load_explicit(&m->map,memory_order_acquire);
    const uint64_t t=clock_gettime_nsec_np(CLOCK_MONOTONIC);
    if(t>=until)return -ETIMEDOUT;
    if(before&1){
      os_sync_wait_on_address_with_timeout(&m->map,before,sizeof m->map,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
        OS_CLOCK_MACH_ABSOLUTE_TIME,until-t);
      continue;
    }
    uint64_t count=atomic_load_explicit(&m->map_nodes,memory_order_relaxed),at=atomic_load_explicit(&m->map_off,memory_order_relaxed);
    *serial=atomic_load_explicit(&m->map_serial,memory_order_relaxed);
    int fits=at<=length && count<=(length-at)/sizeof(struct mesh_record);
    struct mesh_record *copy=fits?malloc((count?count:1)*sizeof *copy):NULL;
    if(copy)memcpy(copy,(char *)m+at,count*sizeof *copy);
    if(mesh_map_whole(m,before)){
      if(!copy)return fits?-ENOMEM:-EINVAL;
      *out=copy;return (int)count;
    }
    free(copy);
  }
}
/* the place of node `node` among the map's n records (ascending node id: its index), or -1 */
static int map_index(const struct mesh_record *r,int n,uint32_t node){
  for(int i=0;i<n;i++)if(r[i].node==node)return i;
  return -1;
}
/* mesh-stat --map: the map for the drivers, one JSON line, read from a mapping of the region to read alone, under the
   map's seqlock (map_read): {"serial": the map serial, "self": this node's index, "nodes": [{"index": its index,
   "node": its node id, "model": its hw.model, "reach": [{"node": the peer's index, "bandwidth": bits a second, "port",
   "peer_port", "qps"}, ...]}, ...]}.  Indices are the map's, its records in ascending node id.  "reach" is the
   record's reach as the map holds it, an entry a cable (struct mesh_reach), so a node pair several cables join is
   listed once a cable, each at the bandwidth that record gives it: no cable is matched, summed or taken at the lesser of
   its two ends here (that join is the map reader's and the bridges' rule's, one copy, design/rubrics.md:422 C9).  An
   entry naming a node the map holds no record of (a cable its peer's record does not yet list) has no index and is left
   out.  The drivers read "node" and "bandwidth"; "port", "peer_port" and "qps" are the bridge's (the rule's inputs,
   CONTRACT "mesh-stat (8)": its cables).  On failure {"up":false,"errno":E} and status 1 */
static int map_print(struct hdr *m,size_t length){
  struct mesh_record *r=NULL;
  uint64_t serial=0;
  const int n=map_read(m,length,&r,&serial);
  if(n<0){printf("{\"up\":false,\"errno\":%d}\n",-n);return 1;}
  const int self=map_index(r,n,m->node);
  printf("{\"serial\":%llu,\"self\":",(unsigned long long)serial);
  if(self<0)printf("null");
  else printf("%d",self);
  printf(",\"nodes\":[");
  for(int i=0;i<n;i++){
    printf("%s{\"index\":%d,\"node\":%u,\"model\":\"",i?",":"",i,r[i].node);
    r[i].model[sizeof r[i].model-1]=0;
    for(const char *c=r[i].model;*c;c++)printf(*c=='"' || *c=='\\'?"\\%c":"%c",*c);
    printf("\",\"reach\":[");
    int first=1;
    for(uint32_t k=0;k<r[i].reached && k<MESH_PORTS;k++){
      const struct mesh_reach *e=r[i].reach+k;
      const int j=map_index(r,n,e->node);
      if(j<0)continue;
      printf("%s{\"node\":%d,\"bandwidth\":%llu,\"port\":%u,\"peer_port\":%u,\"qps\":%u}",first?"":",",j,
             (unsigned long long)e->bandwidth,e->port,e->peer_port,e->qps);
      first=0;
    }
    printf("]}");
  }
  printf("]}\n");
  free(r);
  return 0;
}
/* design/algorithm-sources.md#programcopy */
/* `--map` (map_print) beside main's `--ready`, each followed by the region's name where one is given */
int main(int argc,char **argv){
  int readiness=argc>1 && !strcmp(argv[1],"--ready"),map=argc>1 && !strcmp(argv[1],"--map");
  const char *name=argc>1+readiness+map?argv[1+readiness+map]:getenv("MESH_REGION");
  if(!name) name=MESH_NAME;
  int f=shm_open(name,O_RDONLY,MESH_MODE),error=f<0?errno:EINVAL;
  struct stat info;
  size_t bytes=sizeof(struct hdr);
  struct hdr *h=MAP_FAILED;
  if(f>=0 && !fstat(f,&info) && info.st_size>=(off_t)bytes){bytes=(size_t)info.st_size;h=mmap(NULL,bytes,PROT_READ,MAP_SHARED,f,0);if(h==MAP_FAILED)error=errno;}
  if(h==MAP_FAILED || h->magic!=MESH_MAGIC || h->version!=MESH_VERSION){
    if(map)printf("{\"up\":false,\"errno\":%d}\n",error);
    else printf("{\"up\":false,\"ready\":false}\n");
    if(h!=MAP_FAILED) munmap(h,bytes);
    if(f>=0) close(f);
    return readiness||map?1:0;
  }
  if(map){const int result=map_print(h,bytes);munmap(h,bytes);close(f);return result;}
  uint64_t pid=atomic_load(&h->bridge_pid);
  /* the bridge test mesh-holders.h mesh_gone, the tag's (tag pre-return-20261103 rdma/mesh-stat.c:45), in place of
     main's `pid && (!kill(pid,0) || errno==EPERM)`, which it equals: what this prints is unchanged */
  int alive=!mesh_gone((pid_t)pid);
  uint32_t paired=0;
  for(uint32_t i=0;i<h->links;i++)paired+=alive && atomic_load(&mesh_links(h)[i].port.phase)==MESH_PAIRED;
  printf("{\"up\":%s,\"ready\":%s,\"paired\":%s,\"bridge_pid\":%llu,\"client\":%llu,\"version\":%u,\"node\":%u,\"pgsz\":%u,\"block\":%u,\"rows\":%u,\"qps\":%u,\"links\":%u,\"paired_links\":%u,\"code\":%lld,\"domain\":%u,\"peers\":[",
    alive?"true":"false",alive?"true":"false",paired?"true":"false",(unsigned long long)pid,(unsigned long long)(uint32_t)atomic_load(&h->client),
    h->version,h->node,h->pgsz,h->block,h->rows,h->qps,h->links,paired,(long long)h->port.code,h->port.domain);
  for(uint32_t i=0;i<h->links;i++){
    struct mesh_link_info *link=&mesh_links(h)[i];
    printf("%s{\"node\":%u,\"phase\":%u,\"code\":%lld,\"domain\":%u,\"device\":\"%s\",\"bandwidth\":%llu}",i?",":"",link->peer,atomic_load(&link->port.phase),(long long)link->port.code,link->port.domain,link->device,(unsigned long long)__atomic_load_n(&link->bandwidth,__ATOMIC_RELAXED));
  }
  printf("]}\n");
  munmap(h,bytes); close(f); return readiness && !alive;
}
