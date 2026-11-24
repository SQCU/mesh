#include "mesh-view.h"
#include <errno.h>
#include <stdlib.h>

/* mesh-view.h: a link map's form.  A mesh and a graph take any links among their nodes; a ring's and a tree's links
   are read as parent -> child (link[l][0] -> link[l][1]), one into each node */
int mesh_link_map_check(const struct mesh_link_map *map){
  const uint32_t n=map->nodes;
  uint32_t at,steps;
  if(map->kind>MESH_LINKS_GRAPH || !n || (map->links && !map->link))return EINVAL;
  for(uint32_t l=0;l<map->links;l++)if(map->link[l][0]>=n || map->link[l][1]>=n || map->link[l][0]==map->link[l][1])return EINVAL;
  if(map->kind==MESH_LINKS_MESH || map->kind==MESH_LINKS_GRAPH)return 0;
  uint32_t *up=malloc((size_t)n*sizeof *up);
  if(!up)return ENOMEM;
  int result=0;
  for(uint32_t v=0;v<n;v++)up[v]=n;
  for(uint32_t l=0;l<map->links && !result;l++){
    if(up[map->link[l][1]]<n)result=EINVAL;
    else up[map->link[l][1]]=map->link[l][0];
  }
  if(!result && map->kind==MESH_LINKS_RING){
    at=0;steps=0;
    do{at=up[at];steps++;}while(at<n && at && steps<n);
    result=at==0 && steps==n?0:EINVAL;
  }else if(!result){
    if(map->links!=n-1)result=EINVAL;
    for(uint32_t v=0;v<n && !result;v++){
      for(at=v,steps=0;up[at]<n && steps<n;steps++)at=up[at];
      if(up[at]<n)result=EINVAL;
    }
  }
  free(up);
  return result;
}
