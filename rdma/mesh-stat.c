#include "mesh-observe.h"
#include "mesh-dataflow.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
/* the bridge's ports as it publishes them for its observation tools (mesh-observe.h), mesh-stat being one: each port's
   peer node, phase and bandwidth, a port one the bridge found (its phase not MESH_UNKNOWN), the bridge finding them in
   index order; the port count (at most `capacity` written), or -errno (ESRCH: no bridge runs) */
struct port_view { uint32_t peer,phase; uint64_t bandwidth; };
static int ports_read(struct port_view *out,uint32_t capacity,uint32_t *node){
  const int file=shm_open(MESH_NAME,O_RDONLY,0);
  if(file<0)return -errno;
  struct stat info;
  if(fstat(file,&info) || info.st_size<(off_t)sizeof(struct hdr)){const int error=errno?errno:EINVAL;close(file);return -error;}
  struct hdr *m=mmap(NULL,(size_t)info.st_size,PROT_READ,MAP_SHARED,file,0);
  close(file);
  if(m==MAP_FAILED)return -errno;
  int count=m->magic!=MESH_MAGIC || m->version!=MESH_LAYOUT?-EINVAL:mesh_gone((pid_t)atomic_load(&m->bridge_pid))?-ESRCH:0;
  if(!count)*node=m->node;
  for(uint32_t i=0;count>=0 && i<MESH_PORTS;i++){
    const struct mesh_link_info *link=mesh_links(m)+i;
    const uint32_t phase=atomic_load_explicit(&link->phase,memory_order_acquire);
    if(phase==MESH_UNKNOWN)break;
    if((uint32_t)count<capacity)out[count]=(struct port_view){atomic_load_explicit(&link->peer,memory_order_relaxed),phase,
      atomic_load_explicit(&link->bandwidth,memory_order_relaxed)};
    count++;
  }
  munmap(m,(size_t)info.st_size);
  return count;
}
/* design/transport.md: the node's bridge as it publishes itself, one JSON line: whether it answers
   (`up`, else the errno), its node, its ports by index (ports_read: each port's peer node, phase and bandwidth) and the
   mesh's map (mesh_map over every node its records hold: each node's bridge number, and each link by the two nodes'
   numbers with its bandwidth in bits a second, null where none is reported), its region in bytes with the bytes no lease holds (mesh_region), and every node's
   record as the bridge holds it: its node, its bridge's region, its memory and hardware model, its boot and its
   available memory (mesh_nodes_health, design/transport.md) */
int main(void){
  struct port_view port[MESH_PORTS];
  uint32_t node=0,*nodes=NULL,(*link)[2]=NULL,room=64,links=64;
  uint64_t *bandwidth=NULL;
  const int ports=ports_read(port,MESH_PORTS,&node);
  if(ports<0){printf("{\"up\":false,\"errno\":%d}\n",-ports);return 0;}
  struct mesh_link_map map={0};
  int count=-ENOSPC;
  while(count==-ENOSPC){
    nodes=realloc(nodes,(room+1)*sizeof *nodes);link=realloc(link,(links+1)*sizeof *link);bandwidth=realloc(bandwidth,(links+1)*sizeof *bandwidth);
    if(!nodes || !link || !bandwidth)return 1;
    map=(struct mesh_link_map){.kind=MESH_LINKS_MESH,.nodes=room,.links=links,.link=link,.bandwidth=bandwidth};
    count=mesh_map(NULL,0,nodes,NULL,&map);
    room=map.nodes;links=map.links;
  }
  uint64_t region=0,unleased=0,busy=0,idle_since=0;
  if(mesh_region(&region,&unleased))region=unleased=0;
  const int carrying=mesh_busy(&busy,&idle_since);
  printf("{\"up\":true,\"node\":%u,\"region\":%llu,\"unleased\":%llu,",node,(unsigned long long)region,(unsigned long long)unleased);
  if(!carrying)printf("\"idle\":%s,\"idle_since\":%llu,",busy?"false":"true",(unsigned long long)(idle_since/1000000000u));
  printf("\"peers\":[");
  static const char *const phase_name[]={[MESH_UNKNOWN]="UNKNOWN",[MESH_PAIRING]="PAIRING",[MESH_PAIRED]="PAIRED",[MESH_STOPPED]="STOPPED"};
  for(int i=0;i<ports && i<(int)MESH_PORTS;i++)
    printf("%s{\"port\":%d,\"node\":%u,\"phase\":\"%s\",\"bandwidth\":%llu}",i?",":"",i,port[i].peer,
      port[i].phase<=MESH_STOPPED?phase_name[port[i].phase]:"?",(unsigned long long)port[i].bandwidth);
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
