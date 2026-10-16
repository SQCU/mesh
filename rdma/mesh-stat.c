#include "mesh.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
/* design/bridge-and-ipc.md#ports: the node's bridge as libmesh reads it (mesh-plan.h), one JSON line: whether it answers
   (`up`, else the errno), its node, its ports by index (mesh_observe: each port's peer node, phase and bandwidth) and the
   mesh's map (mesh_map over every node its records hold: each node's bridge number, and each link by the two nodes'
   numbers with its alpha and beta), its window in bytes with the bytes no lease holds (mesh_window), and every node's
   health as the records hold it (mesh_nodes_health, design/recovery.md#diagnosis) */
int main(void){
  struct mesh_link_view port[MESH_PORTS];
  uint32_t node=0,*nodes=NULL,(*link)[2]=NULL,room=64,links=64;
  double (*cost)[2]=NULL;
  const int ports=mesh_observe(port,MESH_PORTS,&node);
  if(ports<0){printf("{\"up\":false,\"errno\":%d}\n",-ports);return 0;}
  struct mesh_link_map map={0};
  int count=-ENOSPC;
  while(count==-ENOSPC){
    nodes=realloc(nodes,(room+1)*sizeof *nodes);link=realloc(link,(links+1)*sizeof *link);cost=realloc(cost,(links+1)*sizeof *cost);
    if(!nodes || !link || !cost)return 1;
    map=(struct mesh_link_map){.kind=MESH_LINKS_MESH,.nodes=room,.links=links,.link=link,.cost=cost};
    count=mesh_map(NULL,0,nodes,NULL,&map);
    room=map.nodes;links=map.links;
  }
  uint64_t window=0,unleased=0;
  if(mesh_window(&window,&unleased))window=unleased=0;
  printf("{\"up\":true,\"node\":%u,\"window\":%llu,\"unleased\":%llu,\"peers\":[",node,(unsigned long long)window,(unsigned long long)unleased);
  for(int i=0;i<ports && i<MESH_PORTS;i++)
    printf("%s{\"port\":%d,\"node\":%u,\"phase\":%u,\"bandwidth\":%llu}",i?",":"",i,port[i].peer,port[i].phase,(unsigned long long)port[i].bandwidth);
  if(count<0)printf("],\"map\":{\"errno\":%d}",-count);
  else{
    printf("],\"map\":{\"nodes\":[");
    for(int i=0;i<count;i++)printf("%s%u",i?",":"",nodes[i]);
    printf("],\"links\":[");
    for(uint32_t l=0;l<map.links;l++)
      printf("%s[%u,%u,%g,%g]",l?",":"",nodes[map.link[l][0]],nodes[map.link[l][1]],map.cost[l][0],map.cost[l][1]);
    printf("]}");
  }
  static const char *resets[]={"unknown","supply","watchdog","other"},*powers[]={"unknown","AC","battery","UPS"};
  struct mesh_node_health health[64];
  const int known=mesh_nodes_health(health,64);
  printf(",\"nodes\":[");
  for(int i=0;i<known && i<64;i++){
    const struct mesh_health *h=&health[i].health;
    printf("%s{\"node\":%u,\"name\":\"",i?",":"",health[i].node);
    for(const char *c=health[i].name;*c;c++)printf(*c=='"' || *c=='\\'?"\\%c":"%c",*c);
    printf("\",\"sampled\":%llu,\"booted\":%llu,\"session\":\"%.36s\",\"reset\":\"%s\",\"shutdown\":\"",
           (unsigned long long)h->sampled,(unsigned long long)h->booted,h->session,h->reset<4?resets[h->reset]:"unknown");
    for(const char *c=h->shutdown;*c;c++)printf(*c=='"' || *c=='\\'?"\\%c":"%c",*c);
    printf("\",\"power\":\"%s\",\"charge\":%d,\"minutes\":%d,\"free\":%llu,\"wired\":%llu,\"pressure\":%u,\"thermal\":%u}",
           h->power<4?powers[h->power]:"unknown",h->charge,h->minutes,(unsigned long long)h->free,(unsigned long long)h->wired,
           h->pressure,h->thermal);
  }
  printf("]}\n");
  free(nodes);free(link);free(cost);
  return 0;
}
