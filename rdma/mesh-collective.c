#include "mesh-call.h"
#include "mesh-plan.h"
#include "mesh-graph.h"
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* design/algorithm-sources.md#topology: every node computes the same trees, routes and programs from the same map, so
   no product is fused into a sum where one node's compiler might fuse it and another's not */
#pragma STDC FP_CONTRACT OFF

/* The map's nodes and relays (relays count with a table alone). */
static uint32_t mesh_all(const struct mesh_link_map *map){ return map->nodes+(map->next?map->relays:0); }

/* Whether a link of the map joins u and x, either way, of its nodes and relays (a mesh's every pair of nodes). */
static int mesh_joins(const struct mesh_link_map *map,uint32_t u,uint32_t x){
  const uint32_t N=mesh_all(map);
  if(u==x || u>=N || x>=N)return 0;
  if(map->kind==MESH_LINKS_MESH && u<map->nodes && x<map->nodes)return 1;
  for(uint32_t l=0;l<map->links;l++)
    if((map->link[l][0]==u && map->link[l][1]==x) || (map->link[l][0]==x && map->link[l][1]==u))return 1;
  return 0;
}

/* design/whole-function.md#fabric: parallel physical capacities add. */
static uint32_t mesh_parallel(const struct mesh_link_map *map,uint32_t u,uint32_t x,uint64_t *capacity){
  uint32_t count=0;
  *capacity=0;
  for(uint32_t l=0;l<map->links;l++)
    if((map->link[l][0]==u && map->link[l][1]==x) || (map->link[l][0]==x && map->link[l][1]==u)){
      *capacity+=map->bandwidth?map->bandwidth[l]:0;
      count++;
    }
  return count;
}
static uint64_t mesh_hop_inverse(const struct mesh_link_map *map,uint32_t u,uint32_t x){
  uint64_t capacity;
  mesh_parallel(map,u,x,&capacity);
  return capacity?(uint64_t)llround(8e15/(double)capacity):0;
}

/* A node's mark (the map's, else its number) */
static uint32_t mesh_mark(const struct mesh_link_map *map,uint32_t v){ return map->mark?map->mark[v]:v; }

/* The route between a and b over the map's nodes and relays, a and b included, into path (room nodes + relays): its
   node count, 0 where there is none (a == b, or it does not reach): with a table, the path up the in-tree of the one of
   the higher mark from the other (reversed where that is a), so a route and its reverse are one path; else the link
   joining them */
static uint32_t mesh_route_nodes(const struct mesh_link_map *map,uint32_t a,uint32_t b,uint32_t *path){
  const uint32_t N=mesh_all(map);
  if(a==b || a>=N || b>=N)return 0;
  if(!map->next){
    if(!mesh_joins(map,a,b))return 0;
    path[0]=a;path[1]=b;
    return 2;
  }
  const int up=mesh_mark(map,a)<mesh_mark(map,b);
  const uint32_t from=up?a:b,to=up?b:a;
  uint32_t count=0;
  for(uint32_t at=from;count<N;){
    path[count++]=at;
    if(at==to)break;
    const uint32_t step=map->next[(size_t)at*N+to];
    if(step==at || !mesh_joins(map,at,step))return 0;
    at=step;
  }
  if(!count || path[count-1]!=to)return 0;
  for(uint32_t i=0;!up && i<count/2;i++){const uint32_t t=path[i];path[i]=path[count-1-i];path[count-1-i]=t;}
  return count;
}

uint32_t mesh_route_path(const struct mesh_link_map *map,uint32_t a,uint32_t b,uint32_t *path){
  const uint32_t N=mesh_all(map);
  if(map->next || a>=N || b>=N || a==b)return mesh_route_nodes(map,a,b,path);
  struct mesh_link_map with=*map;
  uint32_t *made=malloc((size_t)N*N*sizeof *made),count=0;
  if(made && !mesh_routes(map,made)){with.next=made;count=mesh_route_nodes(&with,a,b,path);}
  free(made);
  return count;
}

int mesh_link_map_check(const struct mesh_link_map *map){
  const uint32_t n=map->nodes,N=mesh_all(map);
  if(map->kind>MESH_LINKS_GRAPH || !n || (map->links && !map->link) || (map->relays && (!map->next || map->kind!=MESH_LINKS_GRAPH)))
    return EINVAL;
  for(uint32_t l=0;l<map->links;l++)if(map->link[l][0]>=N || map->link[l][1]>=N || map->link[l][0]==map->link[l][1])return EINVAL;
  if(map->next){
    for(uint32_t a=0;a<N;a++)for(uint32_t b=0;b<N;b++){
      const uint32_t to=map->next[(size_t)a*N+b];
      if(to>=N || (a==b && to!=a) || (to!=a && !mesh_joins(map,a,to)))return EINVAL;
    }
    return 0;
  }
  if(map->kind==MESH_LINKS_MESH || map->kind==MESH_LINKS_GRAPH)return 0;
  uint32_t *up=malloc((size_t)n*sizeof *up);
  if(!up)return ENOMEM;
  int result=0;
  for(uint32_t v=0;v<n;v++)up[v]=n;
  for(uint32_t l=0;l<map->links && !result;l++){
    if(up[map->link[l][1]]<n)result=EINVAL;
    else up[map->link[l][1]]=map->link[l][0];
  }
  /* a ring's links one cycle up from node 0 through every node (its first return n steps on); a tree's every
     chain ending at a root (mesh-graph.h) */
  if(!result && map->kind==MESH_LINKS_RING)result=mesh_graph_cycle_length(up,n,0)==n?0:EINVAL;
  else if(!result){
    if(map->links!=n-1)result=EINVAL;
    for(uint32_t v=0;v<n && !result;v++)if(mesh_graph_climb(up,n,UINT32_MAX,v,NULL,NULL)==UINT32_MAX)result=EINVAL;
  }
  free(up);
  return result;
}

/* link l's end `side` (mesh-graph.h's edge ends) */
static uint32_t mesh_link_end(const void *context,uint32_t l,uint32_t side){ return ((const struct mesh_link_map *)context)->link[l][side]; }
int mesh_link_map_joined(const struct mesh_link_map *map,uint32_t *apart){
  const int form=mesh_link_map_check(map);
  if(form)return form;
  const uint32_t n=map->nodes;
  if(map->next){
    uint32_t path[mesh_all(map)+1];
    for(uint32_t a=0;a<n;a++)for(uint32_t b=a+1;b<n;b++)
      if(!mesh_route_nodes(map,a,b,path)){
        if(apart){apart[0]=a;apart[1]=b;}
        return ENOTCONN;
      }
    return 0;
  }
  if(map->kind!=MESH_LINKS_GRAPH)return 0;
  /* the nodes node 0 reaches over the links either way (mesh-graph.h) */
  uint32_t *order=malloc((size_t)n*sizeof *order);
  unsigned char *seen=malloc(n);
  if(!order || !seen){free(order);free(seen);return ENOMEM;}
  mesh_graph_reach(n,map->links,mesh_link_end,map,0,order,seen);
  uint32_t first=0;
  while(first<n && seen[first])first++;
  free(order);free(seen);
  if(first==n)return 0;
  if(apart){apart[0]=0;apart[1]=first;}
  return ENOTCONN;
}

/* Whether node v is in a bit set of nodes (a bit a node, 64 a word; NULL: every node). */
static int mesh_member(const uint64_t *set,uint32_t v){ return !set || (set[v/64]>>(v%64)&1); }

/* Whether the map links its nodes a and b, either way: a link joins them, or (with a table) their route crosses relays
   alone. */
int mesh_link_between(const struct mesh_link_map *map,uint32_t a,uint32_t b){
  if(a==b || a>=map->nodes || b>=map->nodes)return 0;
  if(!map->next)return mesh_joins(map,a,b);
  uint32_t path[mesh_all(map)+1];
  const uint32_t count=mesh_route_nodes(map,a,b,path);
  for(uint32_t i=1;i+1<count;i++)if(path[i]<map->nodes)return 0;
  return count>=2;
}

/* -- routes -------------------------------------------------------------------------------------------- */

/* 48 bits of splitmix64's finalizer [Steele, Lea & Flood, OOPSLA 2014] over three marks */
static uint64_t mesh_hash(uint32_t x,uint32_t y,uint32_t z){
  uint64_t h=(((uint64_t)x<<32)|y)^((uint64_t)z*UINT64_C(0x9e3779b97f4a7c15));
  h=(h^(h>>30))*UINT64_C(0xbf58476d1ce4e5b9);
  h=(h^(h>>27))*UINT64_C(0x94d049bb133111eb);
  return (h^(h>>31))>>16;
}
/* A route's length toward a root: its links, then the sum of their hops' inverse bandwidths, compared in that order */
struct mesh_length { uint64_t hops,inverse; };
static int mesh_shorter(struct mesh_length a,struct mesh_length b){ return a.hops!=b.hops?a.hops<b.hops:a.inverse<b.inverse; }
static int mesh_equal(struct mesh_length a,struct mesh_length b){ return a.hops==b.hops && a.inverse==b.inverse; }

/* A root's distances by Dijkstra [1959] in its dense form (every length an exact integer, so every node finds the same) */
static void mesh_distances(uint32_t N,const unsigned char *joined,const uint64_t *inverse,uint32_t b,struct mesh_length *length,unsigned char *done){
  for(uint32_t v=0;v<N;v++){length[v]=(struct mesh_length){UINT64_MAX,0};done[v]=0;}
  length[b]=(struct mesh_length){0,0};
  for(;;){
    uint32_t at=N;
    for(uint32_t v=0;v<N;v++)if(!done[v] && length[v].hops!=UINT64_MAX && (at==N || mesh_shorter(length[v],length[at])))at=v;
    if(at==N)break;
    done[at]=1;
    for(uint32_t w=0;w<N;w++){
      if(done[w] || !joined[(size_t)at*N+w])continue;
      const struct mesh_length d={length[at].hops+1,length[at].inverse+inverse[(size_t)at*N+w]};
      if(length[w].hops==UINT64_MAX || mesh_shorter(d,length[w]))length[w]=d;
    }
  }
}
/* Each root's in-tree of shortest routes (fewest links, then the least inverse bandwidths, a pair's parallel links
   together, mesh_hop_inverse): each node's parent among those one equal step nearer the root, the least hash of its mark, the
   parent's and the root's.  A node's parent reads its own candidates alone, so a route is a function of
   the shortest routes between its ends.  Integers throughout. */
int mesh_routes(const struct mesh_link_map *map,uint32_t *next){
  const uint32_t N=map->nodes+map->relays;
  if(!N || map->kind>MESH_LINKS_GRAPH || (map->links && !map->link) || (map->relays && map->kind==MESH_LINKS_MESH))return EINVAL;
  for(uint32_t l=0;l<map->links;l++)if(map->link[l][0]>=N || map->link[l][1]>=N || map->link[l][0]==map->link[l][1])return EINVAL;
  const size_t pairs=(size_t)N*N;
  uint64_t *inverse=calloc(pairs,sizeof *inverse);
  unsigned char *joined=calloc(pairs,1),*done=malloc(N);
  struct mesh_length *length=malloc(N*sizeof *length);
  const int status=inverse && joined && done && length?0:ENOMEM;
  /* a pair's hop: joined where a link joins it (a mesh's every pair), at its parallel links' inverse bandwidth
     together (none given counts none) */
  for(uint32_t u=0;!status && map->kind==MESH_LINKS_MESH && u<N;u++)for(uint32_t x=0;x<N;x++)joined[(size_t)u*N+x]=u!=x;
  for(uint32_t l=0;!status && l<map->links;l++){
    const uint32_t u=map->link[l][0],x=map->link[l][1];
    inverse[(size_t)u*N+x]=inverse[(size_t)x*N+u]=mesh_hop_inverse(map,u,x);
    joined[(size_t)u*N+x]=joined[(size_t)x*N+u]=1;
  }
  for(uint32_t b=0;!status && b<N;b++){
    mesh_distances(N,joined,inverse,b,length,done);
    const uint32_t root=mesh_mark(map,b);
    for(uint32_t v=0;v<N;v++){
      uint32_t up=v;
      for(uint32_t u=0;v!=b && length[v].hops!=UINT64_MAX && u<N;u++)
        if(joined[(size_t)v*N+u] && length[u].hops!=UINT64_MAX &&
           mesh_equal((struct mesh_length){length[u].hops+1,length[u].inverse+inverse[(size_t)v*N+u]},length[v]) &&
           (up==v || mesh_hash(mesh_mark(map,v),mesh_mark(map,u),root)<mesh_hash(mesh_mark(map,v),mesh_mark(map,up),root)))
          up=u;
      next[(size_t)v*N+b]=up;
    }
  }
  free(inverse);free(joined);free(done);free(length);
  return status;
}

int mesh_map_members(const struct mesh_link_map *whole,const uint32_t *members,uint32_t n,uint32_t *whole_node,struct mesh_link_map *map){
  const uint32_t W=whole->nodes,N=n?n:W;
  if(whole->relays || (whole->links && !whole->link) || whole->kind>MESH_LINKS_GRAPH)return -EINVAL;
  /* the whole map's links as listed, or a mesh's every pair: each of its listed links at its bandwidth, one at none
     where it lists none */
  const int every=whole->kind==MESH_LINKS_MESH;
  uint32_t L=every?W*(W?W-1:0)/2:whole->links;
  for(uint32_t a=0;every && a<W;a++)for(uint32_t b=a+1;b<W;b++){
    uint32_t listed=0;
    for(uint32_t k=0;k<whole->links;k++)
      listed+=(whole->link[k][0]==a && whole->link[k][1]==b) || (whole->link[k][0]==b && whole->link[k][1]==a);
    if(listed>1)L+=listed-1;
  }
  uint32_t *place=malloc(((size_t)W+1)*sizeof *place),*at=malloc(((size_t)N+W+1)*sizeof *at),*made=NULL,relays=0;
  int error=place && at?0:ENOMEM;
  for(uint32_t k=0;!error && k<W;k++)place[k]=n?UINT32_MAX:k;
  for(uint32_t i=0;!error && i<N;i++){
    at[i]=n?members[i]:i;
    if(n && members[i]<W){
      if(place[members[i]]!=UINT32_MAX)error=EINVAL;
      else place[members[i]]=i;
    }else if(n)at[i]=W;
  }
  for(uint32_t k=0;!error && k<W;k++)if(place[k]==UINT32_MAX){place[k]=N+relays;at[N+relays]=k;relays++;}
  const uint32_t T=N+relays;
  if(!error && (map->nodes<T || map->links<L)){map->nodes=T;map->links=L;error=ENOSPC;}
  else if(!error && relays && (!map->next || !map->mark))error=EINVAL;
  const uint32_t *table=whole->next;
  if(!error && map->next && !table && W){
    if(!(made=malloc((size_t)W*W*sizeof *made)))error=ENOMEM;
    else error=mesh_routes(whole,made);
    table=made;
  }
  if(!error){
    uint32_t l=0;
    for(uint32_t a=0;every && a<W;a++)for(uint32_t b=a+1;b<W;b++){
      const uint32_t from=l;
      for(uint32_t k=0;k<whole->links;k++)
        if((whole->link[k][0]==a && whole->link[k][1]==b) || (whole->link[k][0]==b && whole->link[k][1]==a)){
          map->link[l][0]=place[a];map->link[l][1]=place[b];
          if(map->bandwidth)map->bandwidth[l]=whole->bandwidth?whole->bandwidth[k]:0;
          l++;
        }
      if(l==from){
        map->link[l][0]=place[a];map->link[l][1]=place[b];
        if(map->bandwidth)map->bandwidth[l]=0;
        l++;
      }
    }
    for(uint32_t k=0;!every && k<L;k++){
      map->link[k][0]=place[whole->link[k][0]];map->link[k][1]=place[whole->link[k][1]];
      if(map->bandwidth)map->bandwidth[k]=whole->bandwidth?whole->bandwidth[k]:0;
    }
    for(uint32_t i=0;map->next && i<T;i++)for(uint32_t j=0;j<T;j++)
      map->next[(size_t)i*T+j]=at[i]<W && at[j]<W?place[table[(size_t)at[i]*W+at[j]]]:i;
    for(uint32_t i=0;whole_node && i<T;i++)whole_node[i]=at[i];
    for(uint32_t i=0;map->mark && i<T;i++)map->mark[i]=at[i]<W?mesh_mark(whole,at[i]):UINT32_MAX-i;
    map->nodes=N;map->links=L;map->relays=relays;map->kind=MESH_LINKS_GRAPH;
    uint64_t linked=0;
    for(uint32_t a=0;!relays && a<N;a++)for(uint32_t b=a+1;b<N;b++)linked+=mesh_joins(map,a,b);
    map->kind=!relays && linked*2==(uint64_t)N*(N?N-1:0)?MESH_LINKS_MESH:MESH_LINKS_GRAPH;
  }
  free(place);free(at);free(made);
  return error?-error:(int)N;
}

/* -- trees --------------------------------------------------------------------------------------------- */

/* a climb's last vertex (mesh-graph.h mesh_graph_climb's visit) */
static void mesh_climb_last(void *consumer,uint32_t at,uint32_t step){ (void)step; *(uint32_t *)consumer=at; }

int mesh_trees_check(const struct mesh_link_map *map,const struct mesh_trees *trees,const uint64_t *roots){
  const uint32_t n=trees->nodes;
  if(!n || n!=map->nodes || (trees->count && (!trees->root || !trees->parent || !trees->log_weight)))return EINVAL;
  unsigned char rooted[n];
  memset(rooted,0,n);
  for(uint32_t t=0;t<trees->count;t++){
    const uint32_t r=trees->root[t],*parent=trees->parent+(size_t)t*n;
    if(r>=n || parent[r]!=r || !isfinite(trees->log_weight[t]))return EINVAL;
    rooted[r]=1;
    /* each node's hop to its parent a link of the map, and its climb (mesh-graph.h) ending at the root */
    for(uint32_t v=0;v<n;v++){
      uint32_t last=v;
      if(v!=r && (!mesh_link_between(map,v,parent[v]) ||
                  mesh_graph_climb(parent,n,r,v,mesh_climb_last,&last)==UINT32_MAX || last!=r))return EINVAL;
    }
  }
  for(uint32_t r=0;n>1 && r<n;r++)if(mesh_member(roots,r) && !rooted[r])return EINVAL;
  return 0;
}

/* -- moves: the trees' routes as page moves (mesh-plan.h mesh_page_moves) ------------------------------------------ */

/* Tree t's pages [first, first + pages) along each of its edges, from move k (move NULL: counted only): node v's edge
   to or from its parent u the (v - (v > root))-th of the tree's up moves (a REDUCE v -> u linked to u's own up move,
   none where u is the root), then of its down moves (a TRANSFER from the root, else a FORWARD from u linked to the
   move landing on u).  The moves made. */
static uint32_t mesh_tree_moves(const struct mesh_trees *trees,uint32_t t,uint64_t first,uint64_t pages,int up,int down,
                                uint32_t k,struct mesh_page_move *move){
  const uint32_t n=trees->nodes,r=trees->root[t],*parent=trees->parent+(size_t)t*n,edges=n-1,lower=k+(up?edges:0);
  for(uint32_t v=0;move && v<n;v++){
    if(v==r)continue;
    const uint32_t u=parent[v],mine=v-(v>r),its=u-(u>r);
    if(up)move[k+mine]=(struct mesh_page_move){v,u,MESH_MOVE_REDUCE,u==r?UINT32_MAX:k+its,first,pages};
    if(down)move[lower+mine]=(struct mesh_page_move){u,v,u==r?MESH_MOVE_TRANSFER:MESH_MOVE_FORWARD,
                                                      u==r?UINT32_MAX:lower+its,first,pages};
  }
  return ((up?1:0)+(down?1:0))*edges;
}

/* Root r's `units` pages over its trees by earliest finish (mesh-graph.h mesh_graph_fill: tree t's first release[t]
   after the start, then one each 1 / rate[t]): count[t] for each of r's trees (the others' untouched, `alone` and
   `got` scratch a tree); the last one's finish. */
static double mesh_root_fill(const struct mesh_trees *trees,uint32_t r,const double *release,const double *rate,uint64_t units,
                             double *alone,uint64_t *got,uint64_t *count){
  const uint32_t c=trees->count;
  for(uint32_t t=0;t<c;t++)alone[t]=trees->root[t]==r?rate[t]:0;
  const double end=mesh_graph_fill(c,release,alone,units,got);
  for(uint32_t t=0;t<c;t++)if(trees->root[t]==r)count[t]=got[t];
  return end;
}

/* An all-reduce of `pages` pages, b of them summed at every node at once (up a tree of every root) and the rest up and
   down a tree: across[t] tree t's up-and-down pages, by earliest finish over every tree, its first at twice its
   latency; toward[t] its root's share of the b, by earliest finish over the root's trees, a tree's first after its
   up-and-down pages have crossed it and its latency (the shared up path's pages of the longer tail first: Jackson's
   rule [J. R. Jackson, "Scheduling a production line to minimize maximum tardiness", UCLA 1955]).  *tail the b
   pages' end; returns the rest's. */
static double mesh_reduce_all_ends(const struct mesh_trees *trees,const double *latency,const double *rate,uint64_t pages,
                                   uint64_t b,double *release,double *alone,uint64_t *got,uint64_t *across,uint64_t *toward,
                                   double *tail){
  const uint32_t n=trees->nodes,c=trees->count;
  for(uint32_t t=0;t<c;t++)release[t]=2*latency[t];
  const double end=mesh_graph_fill(c,release,rate,pages-b,across);
  for(uint32_t t=0;t<c;t++){toward[t]=0;release[t]=latency[t]+(rate[t]>0?(double)across[t]/rate[t]:0);}
  *tail=0;
  for(uint32_t r=0;b && r<n;r++){
    const double e=mesh_root_fill(trees,r,release,rate,b,alone,got,toward);
    if(e>*tail)*tail=e;
  }
  return end;
}

uint32_t mesh_page_moves(const struct mesh_trees *trees,uint32_t what,const uint64_t *segment,struct mesh_page_move *move){
  const uint32_t n=trees->nodes,c=trees->count,C=c?c:1;
  if(what>MESH_ALLGATHER)return UINT32_MAX;
  if(n<2)return 0;
  const int all=what==MESH_ALLREDUCE,up=what!=MESH_BROADCAST && what!=MESH_ALLGATHER,down=what!=MESH_REDUCE && what!=MESH_REDUCE_SCATTER;
  const int timed=trees->latency && trees->rate>0;
  const double joint=trees->rate>0?trees->rate:1;
  double most[n],sum[n],rate[C],latency[C],release[C],alone[C];
  uint64_t across[C],toward[C],got[C],at[n],pages=0;
  /* tree t's rate: its share exp(log_weight[t]) of its root's trees' (none where its weight is none) x the trees' rate */
  for(uint32_t r=0;r<n;r++){most[r]=-INFINITY;sum[r]=0;pages+=segment[r];}
  for(uint32_t t=0;t<c;t++){
    const uint32_t r=trees->root[t];
    if(r<n && trees->log_weight[t]>most[r])most[r]=trees->log_weight[t];
  }
  for(uint32_t t=0;t<c;t++){
    const uint32_t r=trees->root[t];
    rate[t]=r<n && isfinite(most[r])?mesh_graph_exp(trees->log_weight[t]-most[r]):0;
    if(r<n)sum[r]+=rate[t];
    latency[t]=timed?trees->latency[t]:0;
    across[t]=toward[t]=0;
  }
  for(uint32_t t=0;t<c;t++)rate[t]=rate[t]>0?rate[t]*joint/sum[trees->root[t]]:0;
  if(!pages)return 0;
  /* every node the pages go up toward or down from roots a tree: an all-reduce's every node, the others' each node of
     a segment (mesh_trees_check's `roots`); else the segment would cross no link */
  for(uint32_t r=0;r<n;r++)if((all || segment[r]) && !(sum[r]>0))return UINT32_MAX;
  uint32_t k=0;
  if(!all){
    /* root r's segment over its trees, each tree's pages one way along it */
    uint64_t first=0;
    for(uint32_t r=0;r<n;first+=segment[r++]){
      at[r]=first;
      if(segment[r])mesh_root_fill(trees,r,latency,rate,segment[r],alone,got,toward);
    }
    for(uint32_t t=0;t<c;t++)if(toward[t]){
      k+=mesh_tree_moves(trees,t,at[trees->root[t]],toward[t],up,down,k,move);
      at[trees->root[t]]+=toward[t];
    }
    return k;
  }
  /* the all-reduce's cut: b, of the least where the summed-at-every-node pages end no sooner than the rest, and the one
     before it, the one whose later end is sooner (the fewer of equals: a page summed at every node crosses every
     tree of every root) */
  double tail,lower_tail;
  uint64_t lo=0,hi=pages;
  while(hi-lo>1){
    const uint64_t mid=lo+(hi-lo)/2;
    if(mesh_reduce_all_ends(trees,latency,rate,pages,mid,release,alone,got,across,toward,&tail)<=tail)hi=mid;else lo=mid;
  }
  const double lower_end=mesh_reduce_all_ends(trees,latency,rate,pages,lo,release,alone,got,across,toward,&lower_tail);
  const double higher_end=mesh_reduce_all_ends(trees,latency,rate,pages,hi,release,alone,got,across,toward,&tail);
  uint64_t b=hi;
  if(!((higher_end>tail?higher_end:tail)<(lower_end>lower_tail?lower_end:lower_tail))){
    b=lo;
    mesh_reduce_all_ends(trees,latency,rate,pages,lo,release,alone,got,across,toward,&tail);
  }
  /* the up-and-down pages first, each tree's after the trees' before it; then each root's share of the b pages
     summed at every node, from the same first page at every root, up its trees */
  uint64_t first=0;
  for(uint32_t t=0;t<c;first+=across[t++])if(across[t])k+=mesh_tree_moves(trees,t,first,across[t],1,1,k,move);
  for(uint32_t r=0;r<n;r++)at[r]=pages-b;
  for(uint32_t t=0;t<c;t++)if(toward[t]){
    k+=mesh_tree_moves(trees,t,at[trees->root[t]],toward[t],1,0,k,move);
    at[trees->root[t]]+=toward[t];
  }
  return k;
}

/* A packed tree while packing: its root, parents and the flow it carries */

/* the map's links among its nodes, a pair's parallel cables one link of their bandwidths summed (mesh-plan.h
   mesh_solve); their count, each link's ends and capacity where `from` is given */
static uint32_t mesh_solve_links(const struct mesh_link_map *map,uint32_t *from,uint32_t *to,double *cap){
  const uint32_t n=map->nodes;
  uint32_t E=0;
  for(uint32_t l=0;l<map->links;l++){
    const uint32_t a=map->link[l][0],b=map->link[l][1];
    if(a>=n || b>=n || a==b)continue;
    const double c=map->bandwidth && map->bandwidth[l]?(double)map->bandwidth[l]:1;
    uint32_t e=0;
    if(from)for(;e<E && !((from[e]==a && to[e]==b) || (from[e]==b && to[e]==a));e++);
    else{
      /* counting: a link is new where no earlier link joins its pair */
      uint32_t seen=0;
      for(uint32_t k=0;k<l && !seen;k++)seen=(map->link[k][0]==a && map->link[k][1]==b) || (map->link[k][0]==b && map->link[k][1]==a);
      E+=!seen;continue;
    }
    if(e==E){from[E]=a;to[E]=b;cap[E]=0;E++;}
    cap[e]+=c;
  }
  return E;
}

uint32_t mesh_solve_room(const struct mesh_link_map *map,double eps,uint32_t count){
  const uint32_t links=2*mesh_solve_links(map,NULL,NULL,NULL);
  const double e=eps>0?eps:1;
  const uint64_t rounds=(uint64_t)(mesh_graph_log(links>1?(double)links:2.0)/(e*e))+1,room=rounds*count;
  return room>=UINT32_MAX?UINT32_MAX-1:(uint32_t)room;
}

/* the map's links and the demands as mesh-graph.h mesh_graph_flow takes them: the graph module's solve
   (design/rubrics.md C9). Changed by the LSTM demonstration session: the multiplicative-weights rounds and their
   least-length trees were written here (design/primitive-copies-routing-2026-11-01.md groups 6, 17 and 18). What must
   stay: the same trees and shares, from the same rounds. */
static int mesh_demand_leaf(const void *context,uint32_t d,uint32_t v){
  return mesh_member(((const struct mesh_demand *)context)[d].leaves,v);
}

uint32_t mesh_solve(const struct mesh_link_map *map,double eps,const struct mesh_demand *demands,uint32_t count,uint32_t room,
                    uint32_t *demand,uint32_t *parent,double *share){
  const uint32_t n=map->nodes,E=mesh_solve_links(map,NULL,NULL,NULL);
  if(!n || !count)return 0;
  if(!(eps>0))return UINT32_MAX;
  uint32_t *from=calloc((size_t)E+1,sizeof *from),*to=calloc((size_t)E+1,sizeof *to),*root=calloc((size_t)count+1,sizeof *root);
  double *cap=calloc((size_t)E+1,sizeof *cap),*up=calloc((size_t)count+1,sizeof *up),*down=calloc((size_t)count+1,sizeof *down);
  uint32_t made=UINT32_MAX;
  if(from && to && root && cap && up && down){
    mesh_solve_links(map,from,to,cap);
    for(uint32_t d=0;d<count;d++){root[d]=demands[d].root;up[d]=(double)demands[d].up;down[d]=(double)demands[d].down;}
    const uint32_t rounds=(uint32_t)(mesh_graph_log(2*E>1?(double)(2*E):2.0)/(eps*eps))+1;
    made=mesh_graph_flow(n,E,from,to,cap,count,root,up,down,mesh_demand_leaf,demands,eps,rounds,room,demand,parent,share);
  }
  free(from);free(to);free(root);free(cap);free(up);free(down);
  return made;
}

uint32_t mesh_flow_trees(uint32_t nodes,const uint32_t *demand,const struct mesh_demand *demands,const uint32_t *parent,const double *share,
                         uint32_t count,uint32_t first,uint32_t last,uint32_t *root,uint32_t *tree,double *log_weight){
  uint32_t trees=0;
  for(uint32_t k=0;k<count;k++){
    if(demand[k]<first || demand[k]>=last)continue;
    root[trees]=demands[demand[k]].root;
    memcpy(tree+(size_t)trees*nodes,parent+(size_t)k*nodes,nodes*sizeof *tree);
    log_weight[trees++]=mesh_graph_log(share[k]);
  }
  return trees;
}

/* the map's edges, their routes' directed hops and the hops' capacities as mesh-graph.h mesh_graph_pack takes them:
   the graph module's packing (design/rubrics.md C9). Changed by the LSTM demonstration session: its Garg and Koenemann
   rounds were written here (design/primitive-copies-routing-2026-11-01.md groups 6 and 18). What must stay: the same
   trees and weights. */
static int mesh_root_member(const void *context,uint32_t r){ return mesh_member(context,r); }

uint32_t mesh_trees_pack(const struct mesh_link_map *map,const uint64_t *roots,double eps,uint32_t most,int shallow,uint32_t *root,
                         uint32_t *parent,double *log_weight){
  const uint32_t n=map->nodes,N=mesh_all(map);
  uint32_t rooted=0;
  for(uint32_t r=0;r<n;r++)rooted+=mesh_member(roots,r);
  if(n==1){root[0]=0;parent[0]=0;log_weight[0]=0;return 1;}
  if(n<2 || !most || !rooted || mesh_link_map_joined(map,NULL))return 0;
  if(!(eps>0 && eps<1))eps=0.1;
  uint32_t m=0,P=0;
  for(uint32_t a=0;a<n;a++)for(uint32_t b=0;b<n;b++)m+=mesh_link_between(map,a,b);
  /* hop v -> w (v's parent w): edge e, from[e] = w, to[e] = v, the directed hops its route crosses crossing[through[e]
     .. through[e + 1]); directed hop k (one way of a pair, u to x, numbered as first met) its pair's parallel links and
     their least bandwidth */
  uint32_t *from=malloc((m+1)*sizeof *from),*to=malloc((m+1)*sizeof *to),*through=malloc((m+2)*sizeof *through);
  uint32_t *crossing=malloc(((size_t)m*N+1)*sizeof *crossing),*hop=malloc((size_t)N*N*sizeof *hop);
  double *capacity=malloc(((size_t)m*N+1)*sizeof *capacity);
  uint32_t *parallel=malloc(((size_t)m*N+1)*sizeof *parallel);
  uint64_t *aggregate=malloc(((size_t)m*N+1)*sizeof *aggregate);
  uint32_t written=0,path[N+1];
  if(!from || !to || !through || !crossing || !hop || !capacity || !parallel || !aggregate)goto done;
  for(size_t k=0;k<(size_t)N*N;k++)hop[k]=UINT32_MAX;
  m=0;through[0]=0;
  for(uint32_t v=0;v<n;v++)for(uint32_t w=0;w<n;w++)if(mesh_link_between(map,v,w)){
    const uint32_t hops=mesh_route_nodes(map,v,w,path);
    through[m+1]=through[m];
    for(uint32_t i=0;i+1<hops;i++){
      uint32_t *k=hop+(size_t)path[i]*N+path[i+1];
      if(*k==UINT32_MAX){
        const uint32_t c=mesh_parallel(map,path[i],path[i+1],aggregate+P);
        parallel[P]=c?c:1;*k=P++;
      }
      crossing[through[m+1]++]=*k;
    }
    from[m]=w;to[m]=v;m++;
  }
  /* a directed hop's capacity in bytes a ns: its parallel links' summed bandwidth where the map
     gives every hop one, else its parallel links' count (every link alike) */
  int known=P>0;
  for(uint32_t k=0;k<P;k++)known=known && aggregate[k];
  for(uint32_t k=0;k<P;k++){
    double inverse=known?8e9/(double)aggregate[k]:1.0/parallel[k];
    capacity[k]=1/inverse;
  }
  written=mesh_graph_pack(n,m,from,to,through,crossing,P,capacity,mesh_root_member,roots,eps,most,shallow,root,parent,log_weight);
done:
  free(from);free(to);free(through);free(crossing);free(hop);free(capacity);free(parallel);free(aggregate);
  return written;
}
