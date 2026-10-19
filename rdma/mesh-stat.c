#include "mesh.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
/* design/bridge-and-ipc.md#ports: the node's bridge as libmesh reads it (mesh-plan.h), one JSON line: whether it answers
   (`up`, else the errno), its node, its ports by index (mesh_observe: each port's peer node, phase and bandwidth) and the
   mesh's map (mesh_map over every node its records hold: each node's bridge number, and each link by the two nodes'
   numbers with its bandwidth in bits a second, null where none is reported), its window in bytes with the bytes no lease holds (mesh_window), and every node's
   record as the bridge holds it: its name, its bridge's window, its memory and hardware model, its boot and its
   available memory (mesh_nodes_health, design/bridge-and-ipc.md#records) */
int main(void){
  struct mesh_link_view port[MESH_PORTS];
  uint32_t node=0,*nodes=NULL,(*link)[2]=NULL,room=64,links=64;
  uint64_t *bandwidth=NULL;
  const int ports=mesh_observe(port,MESH_PORTS,&node);
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
  uint64_t window=0,unleased=0,busy=0,idle_since=0;
  if(mesh_window(&window,&unleased))window=unleased=0;
  const int carrying=mesh_busy(&busy,&idle_since);
  printf("{\"up\":true,\"node\":%u,\"window\":%llu,\"unleased\":%llu,",node,(unsigned long long)window,(unsigned long long)unleased);
  if(!carrying)printf("\"idle\":%s,\"idle_since\":%llu,",busy?"false":"true",(unsigned long long)(idle_since/1000000000u));
  printf("\"peers\":[");
  for(int i=0;i<ports && i<MESH_PORTS;i++)
    printf("%s{\"port\":%d,\"node\":%u,\"phase\":%u,\"bandwidth\":%llu}",i?",":"",i,port[i].peer,port[i].phase,(unsigned long long)port[i].bandwidth);
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
    printf("%s{\"node\":%u,\"name\":\"",i?",":"",health[i].node);
    for(const char *c=health[i].name;*c;c++)printf(*c=='"' || *c=='\\'?"\\%c":"%c",*c);
    printf("\",\"window\":%llu,\"memory\":%llu,\"model\":\"",(unsigned long long)health[i].window,(unsigned long long)health[i].memory);
    for(const char *c=health[i].model;*c;c++)printf(*c=='"' || *c=='\\'?"\\%c":"%c",*c);
    printf("\",\"booted\":%llu,\"available\":%llu}",(unsigned long long)h->booted,(unsigned long long)h->available);
  }
  printf("]}\n");
  free(nodes);free(link);free(bandwidth);
  return 0;
}
