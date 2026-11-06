#include "mesh.h"
/* the map's records and the link table this tool prints, bridge-side (review R1, C1: moved out of mesh.h) */
#include "mesh-map.h"
#include "mesh-holders.h"
/* 2026-11-04, owner B2: the walk of the claims' heads (mesh_head_after), for "clients" (clients_print) */
/* 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5): mesh-dataflow.h
   no longer included: the walk is gone, and "clients" reads the program's root (mesh.h mesh_program) */
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
/* MESH_PLATFORM_NS, the bound of the map's read (map_read), is mesh.h's, its one copy (review R2, B2: this file's
   removed) */
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
   "node": its node id, "model": its hw.model, "reach": [{"node": the peer's index, "bandwidth": bits a second},
   ...]}, ...]}.  Indices are the map's, its records in ascending node id.  "reach" is the
   record's reach as the map holds it, an entry a cable (struct mesh_reach), so a node pair several cables join is
   listed once a cable, each at the bandwidth that record gives it: no cable is matched, summed or taken at the lesser of
   its two ends here (that join is the map reader's and the bridges' rule's, one copy, design/rubrics.md:422 C9).  An
   entry naming a node the map holds no record of (a cable its peer's record does not yet list) has no index and is left
   out.  An entry is a node pair and its bandwidth alone: the cable's ports and queue pairs (the rule's inputs) stay the
   bridge's and are not printed.  On failure {"up":false,"errno":E} and status 1 */
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
      /* the coordinator, at integration (review R2, final): a reach entry tells the drivers a node pair and its
         bandwidth alone, as CONTRACT's mesh-stat line fixes; no port, peer port or queue-pair count leaves the bridge
         for a driver (design/page-queue-change-request.md: no cable, queue pair or link identity outside it) */
      printf("%s{\"node\":%d,\"bandwidth\":%llu}",first?"":",",j,(unsigned long long)e->bandwidth);
      first=0;
    }
    printf("]}");
  }
  printf("]}\n");
  free(r);
  return 0;
}
/* not the tag's (2026-11-04, owner B2: design/plan-to-bridge-change-request.md R8; e6bd7db2's section 6, "Holders and
   clients", as design/compilation-coordination.md ruling 1 and Q5 take it: "mesh-stat prints "clients":[pids] in place
   of client"): the process of an open claim at head `head` (its owner word's low 32 bits; the owner not 0 and `closed`
   0, each read with acquire), 0 where none is open there.
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
   revisions of 4e78f6f4"): the process of the program's root (mesh.h struct mesh_program, at the layout's one place)
   where it is open (its owner word's low 32 bits; `owner` not 0 and `closed` 0, each read with acquire), 0 where it is
   not, where it was a claim head's.  Why: point 5, "These go: ... the claim's head record"; one program's operand at
   the layout's place (A9, "What the bridge runs: the self-timed schedule": "searches for nothing").  What must stay:
   the owner's process, 0 where none.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G4; "The
   revisions of 55b9cbad"), reversing the root's `owner` and `closed`: the process of the operand's holder (hdr
   `client`'s low 32 bits, read after the publication is acquired: mesh_plan stores it before `published`) where the
   operand (mesh.h struct mesh_algo) is published and not retired (`published` not 0 and not `retired`, each acquire),
   0 where it is not.  Why: G4, "Its fields are what A9 and the failport's lifecycle need: the operand's publication
   (B2's single writer) and the holder the heartbeat already knows"; the record is MSCCL-IR's `<algo>` and is not
   called the root ("Root" names the trees' roots).  What must stay: the holder's process, 0 where none.
   2026-11-06 ("Review of 4e78f6f4 against the requirement lines", G2: "merge duplicated patterns"; the coordinator's
   review): the operand's openness by mesh.h mesh_algo_open (mesh 52a12eb0), hdr `client` read after it.  What must
   stay: as above
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-8 and
   R-10: "the holder's read and gone-holder test" merged; W13: "(mesh-link.c 377–381; mesh-records.c 1425–1430,
   1491–1493; mesh-stat.c 118–120)"; "The revisions of 14b9a757", R-10): the holder by mesh-holders.h mesh_algo_holder,
   its one read ({0} where the operand is not open), and its process where it holds the region by mesh_holds(pid,
   start), else 0.  Why: R-8, "the bridge writes only `released`, and a gone holder is read from the heartbeat's words":
   a gone holder's operand stays open until the bridge releases it, so the gone-holder test is read here as at every
   site, and "client" names a live holder, as it did when the bridge retired a gone one (bin/mesh-bridge.sh's restart
   guard reads it).  What must stay: the holder's process, 0 where none holds */
static pid_t clients_open(struct hdr *h){
  const struct mesh_algo_holder holder=mesh_algo_holder(h);
  return mesh_holds(holder.pid,holder.start)?holder.pid:0;
}
/* "clients": the processes of the region's open claims, each once, in the order of their first claim's head
   (mesh-dataflow.h mesh_head_after, a walk of a moving set: a claim taken or closed meanwhile may be met or missed).
   2026-11-04, owner B2, the coordinator's ruling on review V1's finding 1: "client" printed before it, the process of
   the first open claim's owner, 0 where none, since bin/mesh-bridge.sh (a deployment script this unit may not change)
   holds a restart while mesh-stat prints "client":[1-9] (its client_attached), and without the field the guard never
   held and a restart could end a running program.  What must stay: "client" while that script reads it.
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5): the open
   root's process alone, "client" it and "clients" [it] ([] and 0 where the root is not open), where the claims' heads
   were walked and each process printed once.  Why: point 5 and the root's note (mesh.h struct mesh_program: "the
   region holds one program's operand at a time").  What must stay: both fields, "client" while bin/mesh-bridge.sh
   reads it */
static void clients_print(struct hdr *h){
  const pid_t client=clients_open(h);
  printf("\"client\":%d,\"clients\":[",(int)client);
  if(client)printf("%d",(int)client);
  printf("],");
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
  /* 2026-11-04, owner B2 (clients_print's note): "clients" the open claims' processes in place of "client", hdr
     `client`'s low 32 bits, which nothing writes from 110 (design/compilation-coordination.md, L3).  What must stay:
     every other field as before, in its place */
  printf("{\"up\":%s,\"ready\":%s,\"paired\":%s,\"bridge_pid\":%llu,",alive?"true":"false",alive?"true":"false",
    paired?"true":"false",(unsigned long long)pid);
  clients_print(h);
  printf("\"version\":%u,\"node\":%u,\"pgsz\":%u,\"block\":%u,\"rows\":%u,\"links\":%u,\"paired_links\":%u,\"code\":%lld,\"domain\":%u,\"peers\":[",
    h->version,h->node,h->pgsz,h->block,h->rows,h->links,paired,(long long)h->port.code,h->port.domain);
  for(uint32_t i=0;i<h->links;i++){
    struct mesh_link_info *link=&mesh_links(h)[i];
    printf("%s{\"node\":%u,\"phase\":%u,\"code\":%lld,\"domain\":%u,\"device\":\"%s\",\"bandwidth\":%llu}",i?",":"",link->peer,atomic_load(&link->port.phase),(long long)link->port.code,link->port.domain,link->device,(unsigned long long)__atomic_load_n(&link->bandwidth,__ATOMIC_RELAXED));
  }
  printf("]}\n");
  munmap(h,bytes); close(f); return readiness && !alive;
}
