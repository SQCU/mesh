#include "mesh-observe.h"
#include "mesh-call.h"
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
/* design/transport.md section 15, "mesh-stat"; section 16, "The statistic" */
struct port_view { uint32_t peer,phase; uint64_t bandwidth; };
static int ports_read(struct port_view *out,uint32_t capacity,uint32_t *node,uint64_t hang[3]){
  const int file=shm_open(MESH_NAME,O_RDONLY,0);
  if(file<0)return -errno;
  struct stat info;
  if(fstat(file,&info) || info.st_size<(off_t)sizeof(struct hdr)){const int error=errno?errno:EINVAL;close(file);return -error;}
  struct hdr *m=mmap(NULL,(size_t)info.st_size,PROT_READ,MAP_SHARED,file,0);
  close(file);
  if(m==MAP_FAILED)return -errno;
  int count=m->magic!=MESH_MAGIC || m->version!=MESH_LAYOUT?-EINVAL:!atomic_load(&m->bridge_pid)?-ESRCH:0;
  const struct mesh_bridge_pages *own=mesh_bridge_pages(m);
  if(count!=-EINVAL){hang[0]=atomic_load(&own->active);hang[1]=atomic_load(&own->hung);hang[2]=atomic_load(&own->hangs);}
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
static void hang_say(const uint64_t hang[3]){
  printf("\"hang\":{\"active\":%llu,\"hung\":%llu,\"hangs\":%llu,\"share\":%.6f}",(unsigned long long)hang[0],(unsigned long long)hang[1],
         (unsigned long long)hang[2],hang[0]?(double)hang[1]/(double)hang[0]:0.0);
}
int main(void){
  struct port_view port[MESH_PORTS];
  uint32_t node=0,*nodes=NULL,room=64,edges=64;
  struct mesh_edge *edge=NULL;
  uint64_t hang[3]={0};
  const int ports=ports_read(port,MESH_PORTS,&node,hang);
  if(ports<0){printf("{\"up\":false,\"errno\":%d,",-ports);hang_say(hang);printf("}\n");return 0;}
  struct mesh_link_map map={0};
  int count=-ENOSPC;
  while(count==-ENOSPC){
    nodes=realloc(nodes,(room+1)*sizeof *nodes);edge=realloc(edge,(edges+1)*sizeof *edge);
    if(!nodes || !edge)return 1;
    map=(struct mesh_link_map){.nodes=room,.edges=edges,.edge=edge};
    count=mesh_map(NULL,0,nodes,&map);
    room=map.nodes;edges=map.edges;
  }
  uint64_t region=0,unleased=0;
  if(mesh_region(&region,&unleased))region=unleased=0;
  printf("{\"up\":true,\"node\":%u,\"region\":%llu,\"unleased\":%llu,",node,(unsigned long long)region,(unsigned long long)unleased);
  hang_say(hang);printf(",");
  printf("\"peers\":[");
  static const char *const phase_name[]={[MESH_UNKNOWN]="UNKNOWN",[MESH_PAIRING]="PAIRING",[MESH_PAIRED]="PAIRED",[MESH_STOPPED]="STOPPED"};
  for(int i=0;i<ports && i<(int)MESH_PORTS;i++)
    printf("%s{\"port\":%d,\"node\":%u,\"phase\":\"%s\",\"bandwidth\":%llu}",i?",":"",i,port[i].peer,
      port[i].phase<=MESH_STOPPED?phase_name[port[i].phase]:"?",(unsigned long long)port[i].bandwidth);
  if(count<0)printf("],\"map\":{\"errno\":%d}",-count);
  else{
    printf("],\"map\":{\"nodes\":[");
    for(int i=0;i<count;i++)printf("%s%u",i?",":"",nodes[i]);
    printf("],\"edges\":[");
    for(uint32_t l=0;l<map.edges;l++)
      printf("%s[%u,%u,%u,%llu]",l?",":"",nodes[map.edge[l].a],nodes[map.edge[l].b],map.edge[l].cable,(unsigned long long)map.edge[l].capacity);
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
  free(nodes);free(edge);
  return 0;
}
