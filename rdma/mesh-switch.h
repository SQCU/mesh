/* Switched multipath DMA: the bridge's per-piece forwarding rule (design/switched-multipath-dma.md).

   A fabric is the live directed links between nodes, each with its measured rate (bytes a ns) and per-hop latency
   (ns). A hop's one-page delay is its latency plus one page time, so every edge is positive. A node holding a piece
   sends it on for each of its destinations. For each one it takes the outgoing lane with the earliest estimated
   arrival there, among the lanes to a node the piece hasn't visited. The estimate is the lane's backlog, the piece's
   send time and latency, then the shortest one-page delay on to the destination avoiding every visited node.
   Destinations choosing the same lane share one SEND, whose name carries their subset and the visited set.

   Every hop adds a node to the visited set, so a piece's path is simple: at most as many hops as nodes, and no cycle.
   A lane is taken only if its far end can still reach the destination avoiding the visited set, so no piece ends at a
   dead end. Every destination is in exactly one subset at every hop, so each is reached exactly once. These all hold
   by construction, and nothing here checks them at run time. Pure: no I/O, no clock, no global. */
#ifndef MESH_SWITCH_H
#define MESH_SWITCH_H
#include <errno.h>
#include <math.h>
#include <stdint.h>

#define MESH_SWITCH_NODES 64

struct mesh_switch_link { uint32_t from,to; double rate,latency; };
struct mesh_switch_fabric {
  uint32_t nodes,links;
  const struct mesh_switch_link *link;
  double page;                                          /* bytes a page: the store-and-forward unit */
  double dist[MESH_SWITCH_NODES][MESH_SWITCH_NODES];   /* shortest one-page delay, ns; INFINITY unreachable */
};
/* A piece's name, in its landing slot directly beside the page frame (one contiguous range a request): its
   transaction, execution and page, its bytes, and the destinations still to be served through the node it lands on, and the nodes it has visited, the one it
   lands on not yet among them. */
struct mesh_switch_name { uint64_t transaction; uint32_t execution,page; uint64_t to,visited; uint32_t bytes,reserved; };

/* the fabric's distances from its links; 0, or EINVAL (too many nodes, a link off the nodes or of no rate) */
static inline int mesh_switch_fabric_make(struct mesh_switch_fabric *f,uint32_t nodes,const struct mesh_switch_link *link,
                                          uint32_t links,double page){
  if(nodes>MESH_SWITCH_NODES || !(page>0))return EINVAL;
  f->nodes=nodes;f->links=links;f->link=link;f->page=page;
  for(uint32_t i=0;i<nodes;i++)for(uint32_t j=0;j<nodes;j++)f->dist[i][j]=i==j?0:INFINITY;
  for(uint32_t l=0;l<links;l++){
    const struct mesh_switch_link *x=link+l;
    if(x->from>=nodes || x->to>=nodes || x->from==x->to || !(x->rate>0) || !(x->latency>=0))return EINVAL;
    const double c=x->latency+page/x->rate;
    if(c<f->dist[x->from][x->to])f->dist[x->from][x->to]=c;
  }
  for(uint32_t k=0;k<nodes;k++)for(uint32_t i=0;i<nodes;i++)for(uint32_t j=0;j<nodes;j++)
    if(f->dist[i][k]+f->dist[k][j]<f->dist[i][j])f->dist[i][j]=f->dist[i][k]+f->dist[k][j];
  return 0;
}

/* each node's shortest one-page delay to `d` over nodes outside `avoid` (d itself outside it): INFINITY if none */
static inline void mesh_switch_rest(const struct mesh_switch_fabric *f,uint32_t d,uint64_t avoid,double *rest){
  uint64_t done=0;
  for(uint32_t v=0;v<f->nodes;v++)rest[v]=v==d?0:INFINITY;
  for(;;){
    uint32_t u=MESH_SWITCH_NODES;
    for(uint32_t v=0;v<f->nodes;v++)if(!(done>>v&1) && isfinite(rest[v]) && (u==MESH_SWITCH_NODES || rest[v]<rest[u]))u=v;
    if(u==MESH_SWITCH_NODES)return;
    done|=UINT64_C(1)<<u;
    for(uint32_t l=0;l<f->links;l++){
      const struct mesh_switch_link *x=f->link+l;
      if(x->to!=u || avoid>>x->from&1)continue;
      const double c=rest[u]+x->latency+f->page/x->rate;
      if(c<rest[x->from])rest[x->from]=c;
    }
  }
}

/* Node `self` sends on a piece of `bytes` for the destinations `to`; `visited` holds the nodes it has been at, self
   included. `ahead[l]` is how long outgoing link l stays busy with what is already posted on it (ns from now; only
   self's links are read). Writes the lanes used and each one's subset, and returns how many. UINT32_MAX if a
   destination cannot be reached avoiding `visited`. Farthest destinations choose first. A lane already chosen for this
   piece carries another destination at no further send time, because it is the same SEND. */
static inline uint32_t mesh_switch_route(const struct mesh_switch_fabric *f,uint32_t self,uint64_t to,uint64_t visited,
                                         double bytes,const double *ahead,uint32_t *lane,uint64_t *subset){
  uint32_t used=0;
  double finish[MESH_SWITCH_NODES*4],rest[MESH_SWITCH_NODES];
  visited|=UINT64_C(1)<<self;
  uint64_t left=to&~visited;
  while(left){
    uint32_t d=MESH_SWITCH_NODES;
    for(uint32_t v=0;v<f->nodes;v++)if(left>>v&1 && (d==MESH_SWITCH_NODES || f->dist[self][v]>f->dist[self][d]))d=v;
    left&=~(UINT64_C(1)<<d);
    mesh_switch_rest(f,d,visited,rest);
    uint32_t best=UINT32_MAX,best_used=UINT32_MAX;double arrive=INFINITY;
    for(uint32_t l=0;l<f->links;l++){
      const struct mesh_switch_link *x=f->link+l;
      if(x->from!=self || visited>>x->to&1 || !isfinite(rest[x->to]))continue;
      uint32_t u=0;while(u<used && lane[u]!=l)u++;
      const double end=u<used?finish[u]:ahead[l]+bytes/x->rate;
      const double t=end+x->latency+rest[x->to];
      if(t<arrive){arrive=t;best=l;best_used=u;}
    }
    if(best==UINT32_MAX)return UINT32_MAX;
    if(best_used==used){
      if(used==sizeof finish/sizeof *finish)return UINT32_MAX;
      lane[used]=best;subset[used]=0;finish[used]=ahead[best]+bytes/f->link[best].rate;used++;
    }
    subset[best_used]|=UINT64_C(1)<<d;
  }
  return used;
}
#endif
