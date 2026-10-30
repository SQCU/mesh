#include "mesh-observe.h"
#include "mesh-dataflow.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
/* the bridge's ports as it publishes them for its observation tools (mesh-observe.h), mesh-stat being one: each port's
   peer node, phase and bandwidth, a port one the bridge found (its phase not MESH_UNKNOWN), the bridge finding them in
   index order; the port count (at most `capacity` written), or -errno (ESRCH: no bridge runs) */
struct port_view { uint32_t peer,phase; uint64_t bandwidth,landed_ns,landed_bytes; };
/* port i's bytes landed and the bridge's clock as it counted them (mesh-observe.h mesh_link_landed), the pair read
   again where the bridge rewrote its slot meanwhile */
static void landed_read(struct hdr *m,uint32_t i,uint64_t *ns,uint64_t *bytes){
  struct mesh_link_landed *c=mesh_links_landed(m)+i;
  for(;;){
    const uint64_t s=atomic_load_explicit(&c->sequence,memory_order_acquire);
    *ns=atomic_load_explicit(&c->at[s&1][0],memory_order_relaxed);
    *bytes=atomic_load_explicit(&c->at[s&1][1],memory_order_relaxed);
    atomic_thread_fence(memory_order_acquire);
    if(atomic_load_explicit(&c->sequence,memory_order_relaxed)-s<2)return;
  }
}
/* the region, mapped to read (its size in *bytes), or NULL with errno */
static struct hdr *region_read(size_t *bytes){
  const int file=shm_open(MESH_NAME,O_RDONLY,0);
  if(file<0)return NULL;
  struct stat info;
  if(fstat(file,&info) || info.st_size<(off_t)sizeof(struct hdr)){const int error=errno?errno:EINVAL;close(file);errno=error;return NULL;}
  struct hdr *m=mmap(NULL,(size_t)info.st_size,PROT_READ,MAP_SHARED,file,0);
  close(file);
  if(m==MAP_FAILED)return NULL;
  *bytes=(size_t)info.st_size;
  return m;
}
static int ports_read(struct port_view *out,uint32_t capacity,uint32_t *node,uint32_t *layout){
  size_t size=0;
  struct hdr *m=region_read(&size);
  if(!m)return -errno;
  *layout=m->version;
  int count=m->magic!=MESH_MAGIC || m->version!=MESH_LAYOUT?-EINVAL:mesh_gone((pid_t)atomic_load(&m->bridge_pid))?-ESRCH:0;
  if(!count)*node=m->node;
  for(uint32_t i=0;count>=0 && i<MESH_PORTS;i++){
    const struct mesh_link_info *link=mesh_links(m)+i;
    const uint32_t phase=atomic_load_explicit(&link->phase,memory_order_acquire);
    if(phase==MESH_UNKNOWN)break;
    if((uint32_t)count<capacity){
      out[count]=(struct port_view){atomic_load_explicit(&link->peer,memory_order_relaxed),phase,
        atomic_load_explicit(&link->bandwidth,memory_order_relaxed),0,0};
      landed_read(m,i,&out[count].landed_ns,&out[count].landed_bytes);
    }
    count++;
  }
  munmap(m,size);
  return count;
}
/* mesh-stat landed <period_us> <seconds>: every port's bytes landed, a line each time any port's pair moves, looked at
   every period_us for `seconds` from a mapping of the region to read: {"ns": the bridge's clock of the latest pair,
   "ports": [[port, peer node, ns, bytes], ...]}.  It reads the region alone, so the program it watches runs as it
   would without it */
static int landed_watch(uint64_t period_us,double seconds){
  size_t size=0;
  struct hdr *m=region_read(&size);
  if(!m || m->magic!=MESH_MAGIC || m->version!=MESH_LAYOUT){printf("{\"up\":false,\"errno\":%d}\n",m?EINVAL:errno);return 1;}
  uint64_t seen[MESH_PORTS]={0};
  const uint64_t end=clock_gettime_nsec_np(CLOCK_MONOTONIC)+(uint64_t)(seconds*1e9);
  const struct timespec pause={(time_t)(period_us/1000000),(long)(period_us%1000000*1000)};
  while(clock_gettime_nsec_np(CLOCK_MONOTONIC)<end){
    uint64_t ns[MESH_PORTS],bytes[MESH_PORTS],latest=0;
    uint32_t peer[MESH_PORTS],ports=0,moved=0;
    for(uint32_t i=0;i<MESH_PORTS;i++){
      const struct mesh_link_info *link=mesh_links(m)+i;
      if(atomic_load_explicit(&link->phase,memory_order_acquire)==MESH_UNKNOWN)break;
      peer[i]=atomic_load_explicit(&link->peer,memory_order_relaxed);
      landed_read(m,i,ns+i,bytes+i);
      moved|=bytes[i]!=seen[i];seen[i]=bytes[i];
      if(ns[i]>latest)latest=ns[i];
      ports++;
    }
    if(moved){
      printf("{\"ns\":%llu,\"ports\":[",(unsigned long long)latest);
      for(uint32_t i=0;i<ports;i++)printf("%s[%u,%u,%llu,%llu]",i?",":"",i,peer[i],(unsigned long long)ns[i],(unsigned long long)bytes[i]);
      printf("]}\n");
    }
    nanosleep(&pause,NULL);
  }
  fflush(stdout);
  munmap(m,size);
  return 0;
}
/* design/transport.md: the node's bridge as it publishes itself, one JSON line: whether it answers
   (`up`, else the errno), its node, its ports by index (ports_read: each port's peer node, phase, bandwidth and bytes landed with the bridge's clock of them) and the
   mesh's map (mesh_map over every node its records hold: each node's bridge number, and each link by the two nodes'
   numbers with its bandwidth in bits a second, null where none is reported), its region in bytes with the bytes no lease holds (mesh_region), and every node's
   record as the bridge holds it: its node, its bridge's region, its memory and hardware model, its boot and its
   available memory (mesh_nodes_health, design/transport.md) */
int main(int argc,char **argv){
  if(argc==4 && !strcmp(argv[1],"landed"))return landed_watch(strtoull(argv[2],NULL,10),strtod(argv[3],NULL));
  struct port_view port[MESH_PORTS];
  uint32_t node=0,layout=0,*nodes=NULL,(*link)[2]=NULL,room=64,links=64;
  uint64_t *bandwidth=NULL;
  const int ports=ports_read(port,MESH_PORTS,&node,&layout);
  if(ports<0){printf("{\"up\":false,\"errno\":%d,\"region_layout\":%u,\"expected_layout\":%u}\n",-ports,layout,MESH_LAYOUT);return 0;}
  struct mesh_link_map map={0};
  int count=-ENOSPC;
  while(count==-ENOSPC){
    nodes=realloc(nodes,(room+1)*sizeof *nodes);link=realloc(link,(links+1)*sizeof *link);bandwidth=realloc(bandwidth,(links+1)*sizeof *bandwidth);
    if(!nodes || !link || !bandwidth)return 1;
    map=(struct mesh_link_map){.kind=MESH_LINKS_MESH,.nodes=room,.links=links,.link=link,.bandwidth=bandwidth};
    count=mesh_map(NULL,0,nodes,&map);
    room=map.nodes;links=map.links;
  }
  uint64_t region=0,unleased=0,busy=0,idle_since=0;
  if(mesh_region(&region,&unleased))region=unleased=0;
  const int carrying=mesh_busy(&busy,&idle_since);
  printf("{\"up\":true,\"node\":%u,\"gpu_cores\":%u,\"region\":%llu,\"unleased\":%llu,",node,mesh_gpu_cores(),(unsigned long long)region,
         (unsigned long long)unleased);
  if(!carrying)printf("\"idle\":%s,\"idle_since\":%llu,",busy?"false":"true",(unsigned long long)(idle_since/1000000000u));
  printf("\"peers\":[");
  static const char *const phase_name[]={[MESH_UNKNOWN]="UNKNOWN",[MESH_PAIRING]="PAIRING",[MESH_PAIRED]="PAIRED",[MESH_STOPPED]="STOPPED"};
  for(int i=0;i<ports && i<(int)MESH_PORTS;i++)
    printf("%s{\"port\":%d,\"node\":%u,\"phase\":\"%s\",\"bandwidth\":%llu,\"landed\":[%llu,%llu]}",i?",":"",i,port[i].peer,
      port[i].phase<=MESH_STOPPED?phase_name[port[i].phase]:"?",(unsigned long long)port[i].bandwidth,
      (unsigned long long)port[i].landed_ns,(unsigned long long)port[i].landed_bytes);
  if(count<0)printf("],\"map\":{\"errno\":%d}",-count);
  else{
    printf("],\"map\":{\"nodes\":[");
    for(int i=0;i<count;i++)printf("%s%u",i?",":"",nodes[i]);
    printf("],\"links\":[");
    for(uint32_t l=0;l<map.links;l++){
      printf("%s[%u,%u",l?",":"",nodes[map.link[l][0]],nodes[map.link[l][1]]);
      if(map.bandwidth[l])printf(",%llu]",(unsigned long long)map.bandwidth[l]);
      else printf(",null]");
    }
    printf("]}");
  }
  struct mesh_node_health health[64];
  const int known=mesh_nodes_health(health,64);
  printf(",\"nodes\":[");
  for(int i=0;i<known && i<64;i++){
    const struct mesh_health *h=&health[i].health;
    printf("%s{\"node\":%u,\"region\":%llu,\"memory\":%llu,\"model\":\"",i?",":"",health[i].node,(unsigned long long)health[i].region,(unsigned long long)health[i].memory);
    for(const char *c=health[i].model;*c;c++)printf(*c=='"' || *c=='\\'?"\\%c":"%c",*c);
    printf("\",\"booted\":%llu,\"available\":%llu}",(unsigned long long)h->booted,(unsigned long long)h->available);
  }
  printf("]}\n");
  free(nodes);free(link);free(bandwidth);
  return 0;
}
