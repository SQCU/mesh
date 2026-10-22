#include "mesh-collective.h"
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
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

/* The links of the map joining u and x (either way, parallel where the map lists several): their count, and *slowest
   the least bandwidth the map gives any of them (0: none given) */
static uint32_t mesh_parallel(const struct mesh_link_map *map,uint32_t u,uint32_t x,uint64_t *slowest){
  uint32_t count=0;
  *slowest=0;
  for(uint32_t l=0;l<map->links;l++)
    if((map->link[l][0]==u && map->link[l][1]==x) || (map->link[l][0]==x && map->link[l][1]==u)){
      const uint64_t b=map->bandwidth?map->bandwidth[l]:0;
      if(b && (!*slowest || b<*slowest))*slowest=b;
      count++;
    }
  return count;
}
/* A pair's hop over its n parallel links of the map, which together carry n times the slowest one's bandwidth: its
   inverse bandwidth in whole fs a byte, 8e15 over n times its slowest link's bits a second; 0 where the map gives none
   of them a bandwidth */
static uint64_t mesh_hop_inverse(const struct mesh_link_map *map,uint32_t u,uint32_t x){
  uint64_t slowest;
  const uint32_t n=mesh_parallel(map,u,x,&slowest);
  if(!slowest)return 0;
  double inverse=8e9/(double)slowest;
  if(n>1)inverse/=n;
  return (uint64_t)llround(inverse*1e6);
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
  uint32_t at,steps;
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
  /* breadth first from node 0 over the links either way */
  uint32_t *order=malloc((size_t)n*sizeof *order),reached=1;
  unsigned char *seen=calloc(n,1);
  if(!order || !seen){free(order);free(seen);return ENOMEM;}
  order[0]=0;seen[0]=1;
  for(uint32_t head=0;head<reached;head++)for(uint32_t l=0;l<map->links;l++)for(uint32_t e=0;e<2;e++)
    if(map->link[l][e]==order[head] && !seen[map->link[l][1-e]]){seen[map->link[l][1-e]]=1;order[reached++]=map->link[l][1-e];}
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

/* exp and log of IEEE-exact operations alone (+, -, x, / correctly rounded; frexp, ldexp, floor exact), so that every
   node computes the same trees and the same shares from them, whatever its libm */
static double mesh_exp(double x){
  if(x<-745)return 0;
  if(x>709)return INFINITY;
  const double ln2_high=0.693147180369123816490,ln2_low=1.90821492927058770002e-10;
  const double k=floor(x/0.693147180559945309417+0.5),r=(x-k*ln2_high)-k*ln2_low;
  double term=1,sum=1;
  for(int i=1;i<=20;i++){term=term*r/i;sum+=term;}
  return ldexp(sum,(int)k);
}
static double mesh_log(double x){
  if(!(x>0))return -INFINITY;
  int e;
  const double m=frexp(x,&e),s=(m-1)/(m+1),s2=s*s;
  double term=s,sum=0;
  for(int i=1;i<=61;i+=2){sum+=term/i;term*=s2;}
  return 2*sum+e*0.693147180559945309417;
}

/* -- trees --------------------------------------------------------------------------------------------- */

int mesh_trees_check(const struct mesh_link_map *map,const struct mesh_trees *trees){
  const uint32_t n=trees->nodes;
  if(!n || n!=map->nodes || (trees->count && (!trees->root || !trees->parent || !trees->log_weight)))return EINVAL;
  for(uint32_t r=0;n>1 && r<n;r++){
    uint32_t t=0;
    while(t<trees->count && trees->root[t]!=r)t++;
    if(t==trees->count)return EINVAL;
  }
  for(uint32_t t=0;t<trees->count;t++){
    const uint32_t r=trees->root[t],*parent=trees->parent+(size_t)t*n;
    if(r>=n || parent[r]!=r || !isfinite(trees->log_weight[t]))return EINVAL;
    for(uint32_t v=0;v<n;v++){
      if(v!=r && !mesh_link_between(map,v,parent[v]))return EINVAL;
      uint32_t at=v,hops=0;
      while(at!=r && hops<n){at=parent[at];hops++;}
      if(at!=r)return EINVAL;
    }
  }
  return 0;
}

/* Root r's heaviest tree (the lowest-numbered of equals), count where r roots none. */
static uint32_t mesh_trees_heaviest(const struct mesh_trees *trees,uint32_t r){
  uint32_t best=trees->count;
  for(uint32_t t=0;t<trees->count;t++)
    if(trees->root[t]==r && (best==trees->count || trees->log_weight[t]>trees->log_weight[best]))best=t;
  return best;
}

uint32_t mesh_trees_path(const struct mesh_trees *trees,uint32_t from,uint32_t to,uint32_t *path){
  const uint32_t n=trees->nodes,t=to<n?mesh_trees_heaviest(trees,to):trees->count;
  if(from>=n || t==trees->count)return 0;
  uint32_t count=0;
  for(uint32_t at=from;count<n;at=trees->parent[(size_t)t*n+at]){
    path[count++]=at;
    if(at==to)return count;
  }
  return 0;
}

/* The tree set's members' hops: each linked pair's route on the map's table, its physical links (u N + x, the sender's
   way) at hop[(a n + b) N ..] (count[a n + b]), and its length (its links, then its pairs' striped inverse
   bandwidths, as mesh_routes measures) */
struct mesh_hops { uint32_t n,N,*count,*hop; struct mesh_length *length; };
/* a tree edge, member v's hop to its parent u, put on (sign 1) or taken off (-1) every physical link it crosses the way
   from v to u: a reduction's part goes up each edge once and a broadcast's comes down it, so the loads down mirror the
   loads up and balancing one balances the other */
static void mesh_carry(const struct mesh_hops *h,uint64_t *load,uint32_t v,uint32_t u,int sign){
  const size_t pair=(size_t)v*h->n+u;
  for(uint32_t i=0;i<h->count[pair];i++){
    if(sign>0)load[h->hop[pair*h->N+i]]++;
    else load[h->hop[pair*h->N+i]]--;
  }
}
/* the members farthest first from a root (its tree's lengths), then by mark */
struct mesh_far { struct mesh_length length; uint32_t mark,node; };
static int mesh_farther(const void *x,const void *y){
  const struct mesh_far *a=x,*b=y;
  if(!mesh_equal(a->length,b->length))return mesh_shorter(a->length,b->length)?1:-1;
  return (a->mark>b->mark)-(a->mark<b->mark);
}
/* Each member's in-tree of shortest paths over the members' hops (two members linked where the table's route between
   them crosses relays alone, a hop as long as that route), a member's parent among those one equal hop nearer the root:
   first the least hash of its mark, the parent's and the root's; then MESH_TREE_PASSES passes of local search on the
   sum of the squares of the physical links' loads, a load the tree edges (every tree's) whose hops cross the link one
   way from child to parent, as each edge carries its tree's part once (the potential of the congestion game [Rosenthal
   1973], which every move lowers), each root in mark order, each member farthest first: its edge taken off, then put on to the
   parent whose hop adds least to the potential (then the least hash); the set of the pass whose most loaded link
   carries fewest edges kept (then the least potential, then the earliest).  The tree set is the communicator's alone
   (its members agree on it: nccl-mesh.c map_digest), so balancing it moves no route of the bridges'.  Integers
   throughout. */
#define MESH_TREE_PASSES 4
uint32_t mesh_trees_shortest(const struct mesh_link_map *map,uint32_t *root,uint32_t *parent,double *log_weight){
  const uint32_t n=map->nodes,N=mesh_all(map);
  if(!n || (!map->next && map->relays))return 0;
  struct mesh_link_map with=*map;
  uint32_t *made=map->next?NULL:malloc((size_t)N*N*sizeof *made),count=n;
  if(made && !mesh_routes(map,made))with.next=made;
  const size_t pairs=(size_t)n*n;
  struct mesh_hops h={n,N,calloc(pairs+1,sizeof *h.count),malloc((pairs*N+1)*sizeof *h.hop),malloc((pairs+1)*sizeof *h.length)};
  uint32_t *path=malloc(((size_t)N+1)*sizeof *path),*trial=malloc((pairs+1)*sizeof *trial),*reached=calloc((size_t)n+1,sizeof *reached);
  uint64_t *load=calloc((size_t)N*N+1,sizeof *load);
  struct mesh_length *length=malloc((pairs+1)*sizeof *length);
  struct mesh_far *far=malloc((pairs+1)*sizeof *far),*roots=malloc(((size_t)n+1)*sizeof *roots);
  unsigned char *done=malloc((size_t)n+1),*linked=calloc(pairs+1,1);
  uint64_t *inverse=calloc(pairs+1,sizeof *inverse);
  if(!with.next || !h.count || !h.hop || !h.length || !path || !trial || !reached || !load || !length || !far || !roots || !done ||
     !linked || !inverse)count=0;
  /* the hops: a linked pair's route, its physical links and its length */
  for(uint32_t a=0;count && a<n;a++)for(uint32_t b=0;b<n;b++){
    if(a==b || !mesh_link_between(&with,a,b))continue;
    const uint32_t hops=mesh_route_nodes(&with,a,b,path);
    uint64_t sum=0;
    for(uint32_t i=0;i+1<hops;i++){
      sum+=mesh_hop_inverse(map,path[i],path[i+1]);
      h.hop[((size_t)a*n+b)*N+h.count[(size_t)a*n+b]++]=path[i]*N+path[i+1];
    }
    h.length[(size_t)a*n+b]=(struct mesh_length){hops-1,sum};
    linked[(size_t)a*n+b]=1;
    inverse[(size_t)a*n+b]=sum;
  }
  /* each root's lengths over the hops (Dijkstra, mesh_distances' dense form on the members), its members farthest
     first, and each member's parent the least hash among its equals */
  for(uint32_t v=0;count && v<n;v++)roots[v]=(struct mesh_far){{0,0},mesh_mark(map,v),v};
  if(count)qsort(roots,n,sizeof *roots,mesh_farther);
  for(uint32_t k=0;count && k<n;k++){
    const uint32_t r=roots[k].node;
    struct mesh_length *at=length+(size_t)r*n;
    for(uint32_t v=0;v<n;v++){at[v]=(struct mesh_length){UINT64_MAX,0};done[v]=0;}
    at[r]=(struct mesh_length){0,0};
    for(;;){
      uint32_t x=n;
      for(uint32_t v=0;v<n;v++)if(!done[v] && at[v].hops!=UINT64_MAX && (x==n || mesh_shorter(at[v],at[x])))x=v;
      if(x==n)break;
      done[x]=1;
      for(uint32_t w=0;w<n;w++){
        if(done[w] || !linked[(size_t)w*n+x])continue;
        const struct mesh_length d={at[x].hops+h.length[(size_t)w*n+x].hops,at[x].inverse+inverse[(size_t)w*n+x]};
        if(at[w].hops==UINT64_MAX || mesh_shorter(d,at[w]))at[w]=d;
      }
    }
    for(uint32_t v=0;v<n;v++){
      if(at[v].hops==UINT64_MAX){count=0;break;}
      if(v!=r)far[(size_t)r*n+reached[r]++]=(struct mesh_far){at[v],mesh_mark(map,v),v};
    }
    if(!count)break;
    qsort(far+(size_t)r*n,reached[r],sizeof *far,mesh_farther);
    trial[(size_t)r*n+r]=r;
    for(uint32_t i=0;i<reached[r];i++){
      const uint32_t v=far[(size_t)r*n+i].node;
      uint32_t up=v;
      for(uint32_t u=0;u<n;u++)
        if(linked[(size_t)v*n+u] && mesh_equal((struct mesh_length){at[u].hops+h.length[(size_t)v*n+u].hops,at[u].inverse+inverse[(size_t)v*n+u]},at[v]) &&
           (up==v || mesh_hash(mesh_mark(map,v),mesh_mark(map,u),roots[k].mark)<mesh_hash(mesh_mark(map,v),mesh_mark(map,up),roots[k].mark)))
          up=u;
      trial[(size_t)r*n+v]=up;
    }
  }
  for(uint32_t r=0;count && r<n;r++)for(uint32_t v=0;v<n;v++)if(v!=r)mesh_carry(&h,load,v,trial[(size_t)r*n+v],1);
  uint64_t least=UINT64_MAX,least_potential=UINT64_MAX;
  for(uint32_t pass=0;count && pass<=MESH_TREE_PASSES;pass++){
    for(uint32_t k=0;pass && k<n;k++){
      const uint32_t r=roots[k].node;
      uint32_t *tree=trial+(size_t)r*n;
      const struct mesh_length *at=length+(size_t)r*n;
      for(uint32_t i=0;i<reached[r];i++){
        const uint32_t v=far[(size_t)r*n+i].node;
        uint32_t up=tree[v];
        mesh_carry(&h,load,v,up,-1);
        uint64_t best=UINT64_MAX;
        for(uint32_t u=0;u<n;u++){
          if(!linked[(size_t)v*n+u] ||
             !mesh_equal((struct mesh_length){at[u].hops+h.length[(size_t)v*n+u].hops,at[u].inverse+inverse[(size_t)v*n+u]},at[v]))continue;
          uint64_t added=0;
          for(uint32_t i2=0;i2<h.count[(size_t)v*n+u];i2++)added+=2*load[h.hop[((size_t)v*n+u)*N+i2]]+1;
          if(best==UINT64_MAX || added<best ||
             (added==best && mesh_hash(mesh_mark(map,v),mesh_mark(map,u),roots[k].mark)<mesh_hash(mesh_mark(map,v),mesh_mark(map,up),roots[k].mark))){
            best=added;up=u;
          }
        }
        tree[v]=up;
        mesh_carry(&h,load,v,up,1);
      }
    }
    uint64_t most=0,potential=0;
    for(size_t x=0;x<(size_t)N*N;x++){if(load[x]>most)most=load[x];potential+=load[x]*load[x];}
    if(most<least || (most==least && potential<least_potential)){least=most;least_potential=potential;memcpy(parent,trial,pairs*sizeof *parent);}
  }
  for(uint32_t r=0;count && r<n;r++){root[r]=r;log_weight[r]=0;}
  free(made);free(h.count);free(h.hop);free(h.length);free(path);free(trial);free(reached);free(load);free(length);free(far);free(roots);
  free(done);free(linked);free(inverse);
  return count;
}

/* A tree's shape: each node's height (its longest path up from a leaf below it), depth, whether its subtree holds a
   contributor (`live`) and whether a child's does (`inner`) */
struct mesh_shape { uint32_t height,depth; int live,inner; };
static void mesh_tree_shape(const struct mesh_trees *trees,uint32_t t,const uint64_t *contributors,struct mesh_shape *shape){
  const uint32_t n=trees->nodes,r=trees->root[t],*parent=trees->parent+(size_t)t*n;
  for(uint32_t v=0;v<n;v++)shape[v]=(struct mesh_shape){0,0,0,0};
  for(uint32_t v=0;v<n;v++){
    uint32_t at=v,k=0;
    for(;at!=r && k<n;k++){
      const uint32_t up=parent[at];
      if(shape[up].height<k+1)shape[up].height=k+1;
      at=up;
    }
    shape[v].depth=k;
  }
  for(uint32_t v=0;v<n;v++)
    if(mesh_member(contributors,v))for(uint32_t at=v,k=0;k<=n;at=parent[at],k++){shape[at].live=1;if(at==r)break;}
  for(uint32_t v=0;v<n;v++)if(v!=r && shape[v].live)shape[parent[v]].inner=1;
}

/* Whether u is the first of w's live children in tree t (by height, then number): a node outside the contributors
   takes its first child's partial as it comes (COPY) and combines the rest */
static int mesh_first_child(const struct mesh_trees *trees,uint32_t t,const struct mesh_shape *shape,uint32_t w,uint32_t u){
  const uint32_t n=trees->nodes,r=trees->root[t],*parent=trees->parent+(size_t)t*n;
  for(uint32_t c=0;c<n;c++)
    if(c!=r && c!=u && parent[c]==w && shape[c].live &&
       (shape[c].height<shape[u].height || (shape[c].height==shape[u].height && c<u)))return 0;
  return 1;
}

/* A message the compile places: from -> to in a round over parts [part, part + parts), the sender's buffer, the
   receiver's buffer and step, and its order among the pair's messages of the round */
struct mesh_message { uint32_t from,to,round,part,parts,out,in,op,key; };
static int mesh_message_order(const void *a,const void *b){
  const struct mesh_message *x=a,*y=b;
  const uint32_t l[5]={x->from,x->to,x->round,x->key,x->part},r[5]={y->from,y->to,y->round,y->key,y->part};
  for(int i=0;i<5;i++)if(l[i]!=r[i])return l[i]<r[i]?-1:1;
  return 0;
}
/* a rank's move with its order: by round, a round's SENDs first, then by peer and the message's order */
struct mesh_ordered { uint32_t rank,round,receive,peer,key; struct mesh_move move; };
static int mesh_move_order(const void *a,const void *b){
  const struct mesh_ordered *x=a,*y=b;
  const uint32_t l[6]={x->rank,x->round,x->receive,x->peer,x->key,x->move.part},r[6]={y->rank,y->round,y->receive,y->peer,y->key,y->move.part};
  for(int i=0;i<6;i++)if(l[i]!=r[i])return l[i]<r[i]?-1:1;
  return 0;
}

void mesh_compile_room(const struct mesh_trees *trees,uint32_t chunks,uint32_t *parts,uint32_t *moves){
  const uint32_t n=trees->nodes,T=trees->count,c=chunks?chunks:1;
  *parts=(T>1?T:1)*c;
  *moves=(4*T*n+2*n*n+2)*c;
}

int mesh_compile(const struct mesh_trees *trees,uint32_t what,uint32_t root,int whole,uint32_t chunks,const uint64_t *contributors,
                 struct mesh_program *program,uint32_t *segment,double *log_weight,uint32_t *first,struct mesh_move *move){
  const uint32_t n=trees->nodes,T=trees->count,c=chunks;
  if(!n || !c || what>MESH_ALLGATHER || ((what==MESH_REDUCE || what==MESH_BROADCAST) && root>=n))return EINVAL;
  uint32_t room_parts,room;
  mesh_compile_room(trees,c,&room_parts,&room);
  const uint64_t *mine=what==MESH_ALLREDUCE?contributors:NULL;
  struct mesh_shape *shape=malloc(((size_t)T*n+1)*sizeof *shape);
  struct mesh_message *message=malloc((room+1)*sizeof *message);
  struct mesh_ordered *ordered=malloc((2*(size_t)room+1)*sizeof *ordered);
  /* part_of[t * c + j]: tree t's chunk j's part (UINT32_MAX: the program moves none along t) */
  uint32_t *part_of=malloc(((size_t)T*c+1)*sizeof *part_of),*buffer=calloc((size_t)n*n+1,sizeof *buffer);
  uint32_t messages=0,parts=0,scratch=0;
  int status=EINVAL;
  if(!shape || !message || !ordered || !part_of || !buffer)goto done;
  for(uint32_t t=0;t<T;t++)mesh_tree_shape(trees,t,mine,shape+(size_t)t*n);
  for(size_t k=0;k<(size_t)T*c;k++)part_of[k]=UINT32_MAX;
  if(what==MESH_ALLREDUCE && whole){
    /* the whole operand toward every node v up v's heaviest tree: u's partial for v from scratch of its own where it
       has children there, else its own operand, from buffer[u * n + u]: the operand itself (buffer 0) where u sends it
       to one node only and forwards nothing (a connected map's two nodes: the direct exchange, in which u's one SEND and
       its one REDUCE, of its peer's operand, are chunk j's at round j each and lie at the same positions of one
       channel, each position published before it is combined into: libnccl-mesh complete_track, sends_what_lands),
       else its copy (scratch 1), which nothing changes; chunk j its part j */
    parts=c;
    for(uint32_t j=0;j<c;j++){segment[j]=0;log_weight[j]=0;}
    for(uint32_t u=0;u<n;u++){
      uint32_t next=2,own=0;
      for(uint32_t v=0;v<n;v++){
        const uint32_t t=mesh_trees_heaviest(trees,v);
        if(t==T)goto done;
        if(u==v || !shape[(size_t)t*n+u].live)continue;
        if(shape[(size_t)t*n+u].inner)buffer[(size_t)u*n+v]=next++;
        else own++;
      }
      buffer[(size_t)u*n+u]=own==1 && next==2?0:1;
      const uint32_t used=next>2?next-1:own && buffer[(size_t)u*n+u]?1:0;
      if(used>scratch)scratch=used;
    }
    for(uint32_t v=0;v<n;v++){
      const uint32_t t=mesh_trees_heaviest(trees,v),*parent=trees->parent+(size_t)t*n;
      const struct mesh_shape *s=shape+(size_t)t*n;
      for(uint32_t u=0;u<n;u++){
        if(u==v || !s[u].live)continue;
        const uint32_t w=parent[u],op=!mesh_member(mine,w) && mesh_first_child(trees,t,s,w,u)?MESH_STEP_COPY:MESH_STEP_REDUCE;
        for(uint32_t j=0;j<c;j++)
          message[messages++]=(struct mesh_message){u,w,s[u].height+j,j,1,s[u].inner?buffer[(size_t)u*n+v]:buffer[(size_t)u*n+u],w==v?0:buffer[(size_t)w*n+v],op,v};
      }
    }
  }else{
    /* the parts: a rooted collective's root's trees, else each segment's own node's, each a part at its weight, chunk
       by chunk (chunk 0 of each of the segment's trees, then chunk 1, ...); an all-gather's segments its contributors'
       alone, in node order */
    const int rooted=what==MESH_REDUCE || what==MESH_BROADCAST;
    for(uint32_t g=0,index=0;g<(rooted?1:n);g++){
      if(what==MESH_ALLGATHER && !mesh_member(contributors,g))continue;
      for(uint32_t j=0;j<c;j++)for(uint32_t t=0;t<T;t++)
        if(trees->root[t]==(rooted?root:g)){
          part_of[(size_t)t*c+j]=parts;segment[parts]=rooted?0:index;log_weight[parts]=trees->log_weight[t];parts++;
        }
      index++;
    }
    if(!parts)goto done;
    const int up=what==MESH_ALLREDUCE || what==MESH_REDUCE || what==MESH_REDUCE_SCATTER;
    const int down=what==MESH_ALLREDUCE || what==MESH_BROADCAST || what==MESH_ALLGATHER;
    uint32_t base=0;
    for(uint32_t t=0;up && down && t<T;t++)
      if(part_of[(size_t)t*c]!=UINT32_MAX && shape[(size_t)t*n+trees->root[t]].height>base)base=shape[(size_t)t*n+trees->root[t]].height;
    for(uint32_t t=0;t<T;t++){
      if(part_of[(size_t)t*c]==UINT32_MAX)continue;
      const uint32_t r=trees->root[t],*parent=trees->parent+(size_t)t*n;
      const struct mesh_shape *s=shape+(size_t)t*n;
      for(uint32_t j=0;j<c;j++){
        const uint32_t p=part_of[(size_t)t*c+j];
        for(uint32_t u=0;up && u<n;u++){
          if(u==r || !s[u].live)continue;
          const uint32_t w=parent[u],op=!mesh_member(mine,w) && mesh_first_child(trees,t,s,w,u)?MESH_STEP_COPY:MESH_STEP_REDUCE;
          message[messages++]=(struct mesh_message){u,w,s[u].height+j,p,1,0,0,op,p};
        }
        for(uint32_t u=0;down && u<n;u++)
          if(u!=r)message[messages++]=(struct mesh_message){parent[u],u,base+s[u].depth-1+j,p,1,0,0,MESH_STEP_COPY,p};
      }
    }
  }
  /* coalesced: a message joins the one before it between the same two nodes in the same round where their parts adjoin
     and each end's buffer and step are the same */
  qsort(message,messages,sizeof *message,mesh_message_order);
  uint32_t kept=0;
  for(uint32_t i=0;i<messages;i++){
    struct mesh_message *last=kept?message+kept-1:NULL;
    const struct mesh_message *m=message+i;
    if(last && last->from==m->from && last->to==m->to && last->round==m->round && last->part+last->parts==m->part &&
       last->out==m->out && last->in==m->in && last->op==m->op)last->parts+=m->parts;
    else message[kept++]=*m;
  }
  /* each rank's moves: its SENDs and its receives, in order */
  uint32_t moves=0;
  for(uint32_t i=0;i<kept;i++){
    const struct mesh_message *m=message+i;
    ordered[moves++]=(struct mesh_ordered){m->from,m->round,0,m->to,m->key,{MESH_STEP_SEND,m->to,m->round,m->out,m->part,m->parts}};
    ordered[moves++]=(struct mesh_ordered){m->to,m->round,1,m->from,m->key,{m->op,m->from,m->round,m->in,m->part,m->parts}};
  }
  qsort(ordered,moves,sizeof *ordered,mesh_move_order);
  for(uint32_t r=0,k=0;r<=n;r++){
    while(k<moves && ordered[k].rank<r)k++;
    first[r]=k;
  }
  for(uint32_t k=0;k<moves;k++)move[k]=ordered[k].move;
  *program=(struct mesh_program){what,root,n,parts,scratch,0,segment,log_weight,first,move};
  status=0;
done:
  free(shape);free(message);free(ordered);free(part_of);free(buffer);
  return status;
}

/* The program's segments: the most segment a part names, plus one */
static uint32_t mesh_program_segments(const struct mesh_program *program){
  uint32_t segments=1;
  for(uint32_t p=0;p<program->parts;p++)if(program->segment[p]+1>segments)segments=program->segment[p]+1;
  return segments;
}

/* The elements a cut of the program's own lies on: the least multiple of `page` bytes and of an element's bytes, in
   elements (1: any element) */
static uint64_t mesh_quantum(uint64_t page,uint32_t element){
  uint64_t a=page,b=element?element:1;
  if(page<=1)return 1;
  while(b){const uint64_t t=a%b;a=b;b=t;}
  return page/a;
}

/* The operand's segments: `sizes` where given (segment g the sizes[g] elements after the segments before it: mesh
   430cf7b2^ mesh_cut's rule, MPI_Reduce_scatter's and MPI_Allgatherv's counts), else its quanta of `q` elements cut
   units / segments each, the first units % segments one quantum longer, the last segment ending at the operand's end
   (q 1: elements / segments each, the first elements % segments one element longer); 0 where the sizes do not sum to
   the operand's elements */
static int mesh_cut(uint32_t segments,uint64_t elements,const uint64_t *sizes,uint64_t q,uint64_t *base,uint64_t *length){
  const uint64_t units=(elements+q-1)/q;
  uint64_t sum=0;
  for(uint32_t g=0;g<segments;g++){
    if(sizes){base[g]=sum;length[g]=sizes[g];sum+=sizes[g];continue;}
    const uint64_t from=q*(g*(units/segments)+(g<units%segments?g:units%segments));
    const uint64_t to=q*((g+1)*(units/segments)+(g+1<units%segments?g+1:units%segments));
    base[g]=from<elements?from:elements;
    length[g]=(to<elements?to:elements)-base[g];
    sum+=length[g];
  }
  return sum==elements;
}

/* Each part's elements of an operand cut by mesh_cut: [lo[p], lo[p] + count[p]), a segment's parts by the program's
   weights (its trees': the caller's, a configured form's shares, or equal; libnccl-mesh never compiles the packing's
   flows into them), each a whole number of quanta of `q` elements from the segment's start but the segment's last,
   the remainder's quanta one each to the first parts of weight above 0 (a part of weight 0, a tree of share 0, takes
   none); 0 where the cut is none */
static int mesh_program_ranges(const struct mesh_program *program,uint64_t elements,const uint64_t *sizes,uint64_t q,uint64_t *lo,
                               uint64_t *count){
  const uint32_t segments=mesh_program_segments(program);
  uint64_t *bases=malloc(2*(size_t)segments*sizeof *bases);
  if(!bases)return 0;
  if(!mesh_cut(segments,elements,sizes,q,bases,bases+segments)){free(bases);return 0;}
  for(uint32_t g=0;g<segments;g++){
    const uint64_t base=bases[g],length=bases[segments+g],units=(length+q-1)/q;
    double most=-INFINITY,total=0;
    for(uint32_t p=0;p<program->parts;p++)if(program->segment[p]==g && program->log_weight[p]>most)most=program->log_weight[p];
    for(uint32_t p=0;p<program->parts;p++)if(program->segment[p]==g)total+=mesh_exp(program->log_weight[p]-most);
    uint64_t floors=0;
    for(uint32_t p=0;p<program->parts;p++)
      if(program->segment[p]==g)floors+=(uint64_t)floor((double)units*mesh_exp(program->log_weight[p]-most)/total);
    uint64_t remainder=floors<units?units-floors:0,at=base;
    for(uint32_t p=0;p<program->parts;p++){
      if(program->segment[p]!=g)continue;
      const double weight=mesh_exp(program->log_weight[p]-most);
      uint64_t share=(uint64_t)floor((double)units*weight/total);
      if(remainder && weight>0 && at+share*q<base+length){share++;remainder--;}
      share*=q;
      if(at+share>base+length)share=base+length-at;
      lo[p]=at;count[p]=share;at+=share;
    }
  }
  free(bases);
  return 1;
}

uint32_t mesh_program_steps(const struct mesh_program *program,uint32_t rank,struct mesh_operand operand,struct mesh_step *steps){
  return mesh_program_steps_cut(program,rank,operand,NULL,0,steps);
}

uint32_t mesh_program_steps_cut(const struct mesh_program *program,uint32_t rank,struct mesh_operand operand,const uint64_t *segments,
                                uint64_t page,struct mesh_step *steps){
  if(rank>=program->nodes || !program->parts)return 0;
  uint64_t *lo=malloc(program->parts*sizeof *lo),*count=malloc(program->parts*sizeof *count);
  uint32_t k=0;
  if(lo && count && mesh_program_ranges(program,operand.elements,segments,mesh_quantum(page,operand.element_bytes),lo,count)){
    for(uint32_t i=program->first[rank];i<program->first[rank+1];i++){
      const struct mesh_move *m=program->move+i;
      const uint32_t last=m->part+m->parts-1;
      const uint64_t elements=lo[last]+count[last]-lo[m->part];
      if(!elements)continue;
      struct mesh_operand piece=operand;
      piece.elements=elements;
      steps[k++]=(struct mesh_step){m->op,m->peer,m->round,m->buffer,lo[m->part],piece};
    }
  }
  free(lo);free(count);
  return k;
}

/* Chu-Liu/Edmonds: the minimum spanning arborescence of nodes [0, N) rooted at `root` in which each other node v
   takes one edge e with to[e] == v (its parent from[e]); chosen[v] that edge, the root's -1; 0, or -1 where a node has
   none.  Each node's lightest edge; a cycle among them contracted, the edges into it weighing their excess over the
   edge they would replace, the contracted graph's arborescence expanded (the edge into a cycle replaces the cycle's
   edge into the same node). */
static int mesh_arborescence(uint32_t N,uint32_t root,uint32_t E,const uint32_t *from,const uint32_t *to,const double *weight,int32_t *chosen){
  int32_t *best=malloc(N*sizeof *best),*id=malloc(N*sizeof *id),*seen=malloc(N*sizeof *seen);
  uint32_t *nfrom=malloc((E+1)*sizeof *nfrom),*nto=malloc((E+1)*sizeof *nto),*origin=malloc((E+1)*sizeof *origin);
  double *nweight=malloc((E+1)*sizeof *nweight);
  int32_t *nchosen=malloc(N*sizeof *nchosen);
  unsigned char *cyclic=calloc(N,1);
  int status=-1;
  if(!best || !id || !seen || !nfrom || !nto || !origin || !nweight || !nchosen || !cyclic)goto done;
  for(uint32_t v=0;v<N;v++){best[v]=-1;id[v]=-1;seen[v]=-1;}
  for(uint32_t e=0;e<E;e++)
    if(from[e]!=to[e] && to[e]!=root && (best[to[e]]<0 || weight[e]<weight[best[to[e]]]))best[to[e]]=(int32_t)e;
  for(uint32_t v=0;v<N;v++)if(v!=root && best[v]<0)goto done;
  uint32_t components=0;
  for(uint32_t v=0;v<N;v++){
    uint32_t at=v;
    while(at!=root && seen[at]<0 && id[at]<0){seen[at]=(int32_t)v;at=from[best[at]];}
    if(at!=root && id[at]<0 && seen[at]==(int32_t)v){
      uint32_t x=at;
      do{id[x]=(int32_t)components;x=from[best[x]];}while(x!=at);
      cyclic[components++]=1;
    }
  }
  if(!components){
    for(uint32_t v=0;v<N;v++)chosen[v]=v==root?-1:best[v];
    status=0;goto done;
  }
  for(uint32_t v=0;v<N;v++)if(id[v]<0)id[v]=(int32_t)components++;
  uint32_t m=0;
  for(uint32_t e=0;e<E;e++){
    const uint32_t u=(uint32_t)id[from[e]],w=(uint32_t)id[to[e]];
    if(u==w)continue;
    nfrom[m]=u;nto[m]=w;origin[m]=e;
    nweight[m]=cyclic[w]?weight[e]-weight[best[to[e]]]:weight[e];
    m++;
  }
  if(mesh_arborescence(components,(uint32_t)id[root],m,nfrom,nto,nweight,nchosen))goto done;
  for(uint32_t v=0;v<N;v++)chosen[v]=v==root?-1:best[v];
  for(uint32_t x=0;x<components;x++)
    if(x!=(uint32_t)id[root]){const uint32_t e=origin[nchosen[x]];chosen[to[e]]=(int32_t)e;}
  status=0;
done:
  free(best);free(id);free(seen);free(nfrom);free(nto);free(origin);free(nweight);free(nchosen);free(cyclic);
  return status;
}

/* A packed tree while packing: its root, parents and the flow it carries */
struct mesh_packed { uint32_t root; double flow; uint32_t *parent; };
static int mesh_packed_heavier(const void *a,const void *b){
  const struct mesh_packed *x=a,*y=b;
  if(x->root!=y->root)return x->root<y->root?-1:1;
  return x->flow>y->flow?-1:x->flow<y->flow;
}

uint32_t mesh_trees_pack(const struct mesh_link_map *map,double eps,uint32_t most,uint32_t *root,uint32_t *parent,double *log_weight){
  const uint32_t n=map->nodes,N=mesh_all(map);
  if(n==1){root[0]=0;parent[0]=0;log_weight[0]=0;return 1;}
  if(n<2 || !most || mesh_link_map_joined(map,NULL))return 0;
  if(!(eps>0 && eps<1))eps=0.1;
  uint32_t m=0,P=0;
  for(uint32_t a=0;a<n;a++)for(uint32_t b=0;b<n;b++)m+=mesh_link_between(map,a,b);
  /* hop v -> w (v's parent w): edge e, from[e] = w, to[e] = v, the directed hops its route crosses crossing[through[e]
     .. through[e + 1]); directed hop k (one way of a pair, u to x, numbered as first met) its pair's parallel links and
     their least bandwidth */
  uint32_t *from=malloc((m+1)*sizeof *from),*to=malloc((m+1)*sizeof *to),*through=malloc((m+2)*sizeof *through);
  uint32_t *crossing=malloc(((size_t)m*N+1)*sizeof *crossing),*hop=malloc((size_t)N*N*sizeof *hop),*uses=calloc((size_t)m*N+1,sizeof *uses);
  double *capacity=malloc(((size_t)m*N+1)*sizeof *capacity),*length=malloc(((size_t)m*N+1)*sizeof *length),*weight=malloc((m+1)*sizeof *weight);
  uint32_t *parallel=malloc(((size_t)m*N+1)*sizeof *parallel);
  uint64_t *slowest=malloc(((size_t)m*N+1)*sizeof *slowest);
  int32_t *chosen=malloc(n*sizeof *chosen);
  struct mesh_packed *packed=NULL;
  size_t trees=0,room=0;
  uint32_t written=0,path[N+1];
  if(!from || !to || !through || !crossing || !hop || !uses || !capacity || !length || !weight || !chosen || !parallel || !slowest)goto done;
  for(size_t k=0;k<(size_t)N*N;k++)hop[k]=UINT32_MAX;
  m=0;through[0]=0;
  for(uint32_t v=0;v<n;v++)for(uint32_t w=0;w<n;w++)if(mesh_link_between(map,v,w)){
    const uint32_t hops=mesh_route_nodes(map,v,w,path);
    through[m+1]=through[m];
    for(uint32_t i=0;i+1<hops;i++){
      uint32_t *k=hop+(size_t)path[i]*N+path[i+1];
      if(*k==UINT32_MAX){
        const uint32_t c=mesh_parallel(map,path[i],path[i+1],slowest+P);
        parallel[P]=c?c:1;*k=P++;
      }
      crossing[through[m+1]++]=*k;
    }
    from[m]=w;to[m]=v;m++;
  }
  /* a directed hop's capacity in bytes a ns: its parallel links' count times the slowest one's bandwidth where the map
     gives every hop one, else its parallel links' count (every link alike) */
  int known=P>0;
  for(uint32_t k=0;k<P;k++)known=known && slowest[k];
  for(uint32_t k=0;k<P;k++){
    double inverse=known?8e9/(double)slowest[k]:1;
    if(parallel[k]>1)inverse/=parallel[k];
    capacity[k]=1/inverse;
  }
  const double delta=mesh_exp(-mesh_log(P/(1-eps))/eps);
  double D=0;
  for(uint32_t k=0;k<P;k++){length[k]=delta/capacity[k];D+=length[k]*capacity[k];}
  while(D<1)for(uint32_t r=0;r<n && D<1;r++)for(double demand=1;demand>0 && D<1;){
    for(uint32_t e=0;e<m;e++){weight[e]=0;for(uint32_t i=through[e];i<through[e+1];i++)weight[e]+=length[crossing[i]];}
    if(mesh_arborescence(n,r,m,from,to,weight,chosen))goto done;
    /* the tree's flow: its demand, at most each directed hop's capacity over the tree's edges crossing it */
    for(uint32_t v=0;v<n;v++)if(v!=r)for(uint32_t i=through[chosen[v]];i<through[chosen[v]+1];i++)uses[crossing[i]]++;
    double f=demand;
    for(uint32_t v=0;v<n;v++)if(v!=r)for(uint32_t i=through[chosen[v]];i<through[chosen[v]+1];i++)
      if(capacity[crossing[i]]/uses[crossing[i]]<f)f=capacity[crossing[i]]/uses[crossing[i]];
    demand-=f;
    size_t t=0;
    for(;t<trees;t++){
      if(packed[t].root!=r)continue;
      uint32_t v=0;
      while(v<n && (v==r || packed[t].parent[v]==from[chosen[v]]))v++;
      if(v==n)break;
    }
    if(t==trees){
      if(trees==room){
        struct mesh_packed *grown=realloc(packed,(room=room?2*room:64)*sizeof *grown);
        if(!grown)goto done;
        packed=grown;
      }
      if(!(packed[t].parent=malloc(n*sizeof *packed[t].parent)))goto done;
      packed[t].root=r;packed[t].flow=0;
      for(uint32_t v=0;v<n;v++)packed[t].parent[v]=v==r?r:from[chosen[v]];
      trees++;
    }
    packed[t].flow+=f;
    for(uint32_t v=0;v<n;v++)if(v!=r)for(uint32_t i=through[chosen[v]];i<through[chosen[v]+1];i++){
      const uint32_t k=crossing[i];
      if(!uses[k])continue;
      D+=length[k]*eps*f*uses[k];
      length[k]*=1+eps*f*uses[k]/capacity[k];
      uses[k]=0;
    }
  }
  qsort(packed,trees,sizeof *packed,mesh_packed_heavier);
  for(size_t t=0;t<trees;){
    size_t end=t,keep;
    double total=0;
    while(end<trees && packed[end].root==packed[t].root)end++;
    keep=end-t<most?end-t:most;
    for(size_t u=t;u<t+keep;u++)total+=packed[u].flow;
    for(size_t u=t;u<t+keep;u++){
      root[written]=packed[u].root;
      memcpy(parent+(size_t)written*n,packed[u].parent,n*sizeof *parent);
      log_weight[written++]=mesh_log(packed[u].flow/total);
    }
    t=end;
  }
done:
  for(size_t t=0;t<trees;t++)free(packed[t].parent);
  free(packed);free(from);free(to);free(through);free(crossing);free(hop);free(uses);free(capacity);free(length);free(weight);free(chosen);
  free(parallel);free(slowest);
  return written;
}

/* -- the lists: the entries a program's steps take on a holding's send list and landing list --------------------------- */

static uint64_t mesh_step_bytes(const struct mesh_step *step){ return step->piece.elements*step->piece.element_bytes; }

/* whether steps are in round order (mesh_program_steps' order: a compiled program's are) */
static int mesh_rounds_sorted(const struct mesh_step *steps,uint32_t count){
  for(uint32_t k=1;k<count;k++)if(steps[k].round<steps[k-1].round)return 0;
  return 1;
}

uint64_t mesh_lay(const struct mesh_step *steps,uint32_t count,uint32_t peer,int send,uint64_t at,struct mesh_place *made,uint32_t *placed){
  const int sorted=mesh_rounds_sorted(steps,count);
  uint32_t rounds=0;
#define MESH_LAID(k) (steps[k].peer==peer && mesh_step_bytes(steps+(k)) && (steps[k].op==MESH_STEP_SEND)==(send!=0))
  for(uint32_t k=0;sorted && k<count;k++)if(MESH_LAID(k))made[(*placed)++]=(struct mesh_place){k,send!=0,at++};
  for(uint32_t k=0;!sorted && k<count;k++)if(MESH_LAID(k) && steps[k].round+1>rounds)rounds=steps[k].round+1;
  for(uint32_t r=0;!sorted && r<rounds;r++)
    for(uint32_t k=0;k<count;k++)if(MESH_LAID(k) && steps[k].round==r)made[(*placed)++]=(struct mesh_place){k,send!=0,at++};
#undef MESH_LAID
  return at;
}

/* steps by round, then index */
struct mesh_ranked { uint32_t round,index; };
static int mesh_ranked_order(const void *a,const void *b){
  const struct mesh_ranked *x=a,*y=b;
  if(x->round!=y->round)return x->round<y->round?-1:1;
  return (x->index>y->index)-(x->index<y->index);
}
int mesh_steps_before(const struct mesh_step *steps,uint32_t count,int32_t *before){
  struct mesh_ranked *order=mesh_rounds_sorted(steps,count)?NULL:malloc(((size_t)count+1)*sizeof *order);
  if(!order && !mesh_rounds_sorted(steps,count))return ENOMEM;
  for(uint32_t k=0;order && k<count;k++)order[k]=(struct mesh_ranked){steps[k].round,k};
  if(order)qsort(order,count,sizeof *order,mesh_ranked_order);
  int32_t last=-1;
  for(uint32_t i=0;i<count;i++){
    const uint32_t k=order?order[i].index:i;
    before[k]=last;
    if(steps[k].op!=MESH_STEP_SEND && mesh_step_bytes(steps+k))last=(int32_t)k;
  }
  free(order);
  return 0;
}
int mesh_group_before(uint32_t kind,const struct mesh_step *steps,uint32_t count,int32_t *before){
  if(kind!=MESH_GROUP_HOSTED)return mesh_steps_before(steps,count,before);
  for(uint32_t k=0;k<count;k++)before[k]=-1;
  return 0;
}

/* design/transport.md section 14: a group's calls laid together (libnccl-mesh schedule_group), a counted all-to-all's
   steps (libnccl-mesh counted_steps), and the advisories.  The lists' rule, as libnccl-mesh runs it (nccl-mesh.c run,
   complete): each direction between two ranks is one list, the sender's send list and the receiver's landing list,
   entry for entry; the sender publishes its entries in order (its holding's `ready`, one count), the receiver consumes
   them in order, each once it has landed (`landed`, one count); a step's first event, published or consumed, waits until
   the receive before it in its call (mesh_steps_before) is consumed; a combine into bytes a SEND of the same rank on the
   same channel reads waits until that SEND's bytes are sent (`sent`), after its publication.  Nothing else orders them:
   no credit, no fillers (the hardware's credit holds a SEND until its receive is posted, design/transport.md). */
static uint64_t mesh_segment_first(const uint64_t *segment,uint32_t q){
  uint64_t at=0;
  for(uint32_t r=0;r<q;r++)at+=segment[r];
  return at;
}
uint32_t mesh_counted_steps(uint32_t nodes,uint32_t rank,const uint64_t *segments,int hosted,struct mesh_operand element,
                            struct mesh_operand counts,struct mesh_step *steps){
  const uint64_t *sent=segments,*got=segments+nodes;
  uint32_t k=0;
  for(uint32_t q=0;hosted && q<2*nodes;q++)
    if(q%nodes!=rank && segments[q])
      steps[k++]=(struct mesh_step){q<nodes?MESH_STEP_SEND:MESH_STEP_COPY,q%nodes,0,q>=nodes,mesh_segment_first(segments+(q<nodes?0:nodes),q%nodes),
                                    {element.type,element.element_bytes,segments[q]}};
  if(hosted)return k;
  for(uint32_t q=0;q<nodes;q++){
    if(q==rank || !sent[q])continue;
    steps[k++]=(struct mesh_step){MESH_STEP_SEND,q,0,0,mesh_segment_first(sent,q),{counts.type,counts.element_bytes,sent[q]}};
    if(element.elements)steps[k++]=(struct mesh_step){MESH_STEP_SEND,q,0,1,0,element};
  }
  for(uint32_t q=0;q<nodes;q++)
    if(q!=rank && got[q])steps[k++]=(struct mesh_step){MESH_STEP_COPY,q,0,2,mesh_segment_first(got,q),{counts.type,counts.element_bytes,got[q]}};
  for(uint32_t q=0;element.elements && q<nodes;q++)
    if(q!=rank && got[q])steps[k++]=(struct mesh_step){MESH_STEP_COPY,q,0,3,0,element};
  return k;
}
uint64_t mesh_group_lay(const struct mesh_group *g,uint32_t rank,uint32_t peer,int send,uint64_t at,struct mesh_place *scratch,
                        struct mesh_group_place *made,uint32_t *placed){
  *placed=0;
  for(uint32_t pass=0;pass<2;pass++)for(uint32_t i=0;i<g->calls;i++){
    const struct mesh_group_call *c=g->call+i;
    if(c->kind==MESH_GROUP_POINT || (c->kind==MESH_GROUP_COUNTED || c->kind==MESH_GROUP_HOSTED)!=pass)continue;
    uint32_t laid=0;
    at=mesh_lay(c->steps[rank],c->count[rank],peer,send,at,scratch,&laid);
    for(uint32_t k=0;k<laid;k++)made[(*placed)++]=(struct mesh_group_place){i,0,scratch[k]};
  }
  for(uint32_t i=0;i<g->calls;i++){
    const struct mesh_group_call *c=g->call+i;
    if(c->kind!=MESH_GROUP_POINT || !c->count[rank])continue;
    const struct mesh_step *step=c->steps[rank];
    if(step->peer==peer && mesh_step_bytes(step) && (step->op==MESH_STEP_SEND)==(send!=0))
      made[(*placed)++]=(struct mesh_group_place){i,0,{0,send!=0,at++}};
  }
  return at;
}


enum { MESH_ADVISE_PROJECTION, MESH_ADVISE_PRIORITIES, MESH_ADVISE_ALIGNMENT, MESH_ADVISE_CAUSALITY, MESH_ADVISE_HAZARD, MESH_ADVISE_INVOCATION,
       MESH_ADVISE_BOUND, MESH_ADVISE_CHECKS };
static const char *const mesh_advise_names[MESH_ADVISE_CHECKS]={"projection","priorities","alignment","causality","hazard","invocation","bound"};
struct mesh_advice { void (*warn)(void *,const char *); void *arg; uint32_t count; };
static void mesh_advise_say(struct mesh_advice *a,uint32_t check,const char *format,...) __attribute__((format(printf,3,4)));
static void mesh_advise_say(struct mesh_advice *a,uint32_t check,const char *format,...){
  a->count++;
  if(!a->warn)return;
  char line[768];
  const int head=snprintf(line,sizeof line,"%s: ",mesh_advise_names[check]);
  va_list list;
  va_start(list,format);
  vsnprintf(line+head,sizeof line-(size_t)head,format,list);
  va_end(list);
  a->warn(a->arg,line);
}
static const char *mesh_step_name(uint32_t op){ return op==MESH_STEP_SEND?"SEND":op==MESH_STEP_REDUCE?"REDUCE":"COPY"; }
/* a step's entry: its list (UINT32_MAX: none, no bytes) and its index there */
struct mesh_spot { uint32_t lane; uint64_t entry; };
/* a list: rank a's sends to rank b, b's receives from a, its entries, its events from `base` (publications, then
   consumptions) */
struct mesh_list { uint32_t a,b; uint64_t entries,base; };
static uint64_t mesh_event(const struct mesh_list *l,uint32_t consume,uint64_t e){ return l->base+(consume?l->entries:0)+e; }
static void mesh_event_text(const struct mesh_list *lists,uint32_t count,uint64_t v,char *text,size_t room){
  uint32_t h=0;
  while(h+1<count && lists[h+1].base<=v)h++;
  const struct mesh_list *l=lists+h;
  const uint64_t x=v-l->base,e=x%l->entries;
  if(x>=l->entries)snprintf(text,room,"rank %u consumes rank %u's entry %llu",l->b,l->a,(unsigned long long)e);
  else snprintf(text,room,"rank %u publishes entry %llu to rank %u",l->a,(unsigned long long)e,l->b);
}
static int mesh_overlap(const struct mesh_step *x,const struct mesh_step *y){
  return x->buffer==y->buffer && x->first<y->first+y->piece.elements && y->first<x->first+x->piece.elements;
}
/* the events reachable from `from` along `start`/`list` (forward or reverse), marked `generation` */
static void mesh_reach(uint64_t from,const uint64_t *start,const uint64_t *list,uint32_t *mark,uint32_t generation,uint64_t *queue){
  uint64_t head=0,tail=0;
  mark[from]=generation;queue[tail++]=from;
  while(head<tail){
    const uint64_t v=queue[head++];
    for(uint64_t i=start[v];i<start[v+1];i++)if(mark[list[i]]!=generation){mark[list[i]]=generation;queue[tail++]=list[i];}
  }
}
/* step k of a rank's group: its move in its call, and the call where the group has several */
struct mesh_grouped { const struct mesh_group *g; uint32_t **owner,*first; };
static const char *mesh_move(const struct mesh_grouped *x,uint32_t r,uint32_t k,char *text,size_t room){
  const uint32_t i=x->owner[r][k];
  if(x->g->calls>1)snprintf(text,room,"call %u's move %u",i,k-x->first[(size_t)i*x->g->nodes+r]);
  else snprintf(text,room,"move %u",k);
  return text;
}
/* the event at which receive v's bytes are written into its buffer: a combine's at its consumption; a COPY lands in
   place, so at the earliest when its peer publishes it */
static uint64_t mesh_written(const struct mesh_list *lists,const struct mesh_step *v,const struct mesh_spot *y){
  return mesh_event(lists+y->lane,v->op==MESH_STEP_REDUCE,y->entry);
}

uint32_t mesh_group_advise(const struct mesh_group *g,void (*warn)(void *,const char *),void *arg){
  struct mesh_advice advice={warn,arg,0};
  const uint32_t n=g->nodes,calls=g->calls;
  /* bound: a GPU wait's loads past the platform's deadline (design/transport.md) */
  if(g->bound && !g->polls)
    mesh_advise_say(&advice,MESH_ADVISE_BOUND,"the group's GPU waits are unbounded (polls 0): a wait whose landing never comes holds its "
                    "queue past the %u loads that conclude within the platform's deadline",g->bound);
  else if(g->bound && g->polls>g->bound)
    mesh_advise_say(&advice,MESH_ADVISE_BOUND,"the group's GPU waits make up to %u loads, past the %u that conclude within the platform's "
                    "deadline",g->polls,g->bound);
  struct mesh_step **steps=calloc((size_t)n+1,sizeof *steps);
  int32_t **before=calloc((size_t)n+1,sizeof *before);
  struct mesh_spot **spot=calloc((size_t)n+1,sizeof *spot);
  uint32_t **owner=calloc((size_t)n+1,sizeof *owner),*first=calloc((size_t)calls*n+1,sizeof *first);
  uint32_t *count=calloc((size_t)n+1,sizeof *count),most=1,widest=1,result=UINT32_MAX;
  struct mesh_list *lists=malloc(((size_t)n*n+1)*sizeof *lists);
  struct mesh_place *scratch=NULL;
  struct mesh_group_place *ma=NULL,*mb=NULL;
  uint64_t *from=NULL,*to=NULL,*start=NULL,*list=NULL,*back=NULL,*rlist=NULL,*queue=NULL,*remaining=NULL;
  uint32_t *mark=NULL;
  const struct mesh_grouped grouped={g,owner,first};
  char text[2][64];
  if(!steps || !before || !spot || !owner || !first || !count || !lists)goto done;
  int points=0;
  for(uint32_t i=0;i<calls;i++)points|=g->call[i].kind==MESH_GROUP_POINT;
  for(uint32_t r=0;r<n;r++){
    for(uint32_t i=0;i<calls;i++){count[r]+=g->call[i].count[r];if(g->call[i].count[r]+1>widest)widest=g->call[i].count[r]+1;}
    const uint32_t room=count[r]+1;
    steps[r]=malloc(room*sizeof **steps);before[r]=malloc(room*sizeof **before);spot[r]=malloc(room*sizeof **spot);
    owner[r]=malloc(room*sizeof **owner);
    if(!steps[r] || !before[r] || !spot[r] || !owner[r])goto done;
    for(uint32_t i=0,k=0;i<calls;k+=g->call[i].count[r],i++){
      const uint32_t c=g->call[i].count[r];
      first[(size_t)i*n+r]=k;
      if(c)memcpy(steps[r]+k,g->call[i].steps[r],c*sizeof **steps);
      if(mesh_group_before(g->call[i].kind,steps[r]+k,c,before[r]+k))goto done;
      for(uint32_t j=k;j<k+c;j++){owner[r][j]=i;if(before[r][j]>=0)before[r][j]+=(int32_t)k;}
    }
    for(uint32_t k=0;k<count[r];k++)spot[r][k]=(struct mesh_spot){UINT32_MAX,0};
    if(count[r]+1>most)most=count[r]+1;
  }
  /* priorities: each rank's steps of a call by round, a round's SENDs before its receives (Dardha & Gay's priorities) */
  for(uint32_t r=0;r<n;r++)for(uint32_t k=1;k<count[r];k++){
    const struct mesh_step *p=steps[r]+k-1,*s=steps[r]+k;
    if(owner[r][k]==owner[r][k-1] && (s->round<p->round || (s->round==p->round && s->op==MESH_STEP_SEND && p->op!=MESH_STEP_SEND)))
      mesh_advise_say(&advice,MESH_ADVISE_PRIORITIES,"rank %u's %s (%s, round %u) follows its %s (%s, round %u): not in round order, "
                      "a round's SENDs first",r,mesh_move(&grouped,r,k,text[0],sizeof *text),mesh_step_name(s->op),s->round,
                      mesh_move(&grouped,r,k-1,text[1],sizeof *text),mesh_step_name(p->op),p->round);
  }
  /* projection: within each call (the point-to-point calls together, in call order), rank r's i-th SEND to p against
     p's i-th receive from r, piece for piece (a collective's at one place of its operand, an all-to-all's or a
     point-to-point call's each at its own end's); a SEND and its receive in one round */
  for(uint32_t r=0;r<n;r++)for(uint32_t p=0;p<n;p++)for(uint32_t at=0;p!=r && at<=calls;at++){
    if(at<calls?g->call[at].kind==MESH_GROUP_POINT:!points)continue;
    const int placed=at<calls && g->call[at].kind==MESH_GROUP_COLLECTIVE;
#define MESH_OF(x,k) (at<calls?owner[x][k]==at:g->call[owner[x][k]].kind==MESH_GROUP_POINT)
    uint32_t i=0,j=0,sends=0,receives=0;
    for(;;){
      while(i<count[r] && !(steps[r][i].op==MESH_STEP_SEND && steps[r][i].peer==p && MESH_OF(r,i)))i++;
      while(j<count[p] && !(steps[p][j].op!=MESH_STEP_SEND && steps[p][j].peer==r && MESH_OF(p,j)))j++;
      if(i>=count[r] || j>=count[p])break;
      const struct mesh_step *s=steps[r]+i,*v=steps[p]+j;
      if((placed && s->first!=v->first) || s->piece.elements!=v->piece.elements || s->piece.element_bytes!=v->piece.element_bytes || s->piece.type!=v->piece.type)
        mesh_advise_say(&advice,MESH_ADVISE_PROJECTION,"rank %u's SEND %u to rank %u (%s, round %u: elements [%llu, +%llu) of %u bytes) and "
                        "its %s at rank %u (%s, round %u: [%llu, +%llu) of %u bytes) differ",r,sends,p,mesh_move(&grouped,r,i,text[0],sizeof *text),s->round,
                        (unsigned long long)s->first,(unsigned long long)s->piece.elements,s->piece.element_bytes,mesh_step_name(v->op),p,
                        mesh_move(&grouped,p,j,text[1],sizeof *text),v->round,(unsigned long long)v->first,(unsigned long long)v->piece.elements,
                        v->piece.element_bytes);
      else if(s->round!=v->round)
        mesh_advise_say(&advice,MESH_ADVISE_PRIORITIES,"rank %u's SEND %u to rank %u (%s) is in round %u, its %s at rank %u (%s) in round %u",
                        r,sends,p,mesh_move(&grouped,r,i,text[0],sizeof *text),s->round,mesh_step_name(v->op),p,mesh_move(&grouped,p,j,text[1],sizeof *text),v->round);
      sends++;receives++;i++;j++;
    }
    for(;i<count[r];i++)sends+=steps[r][i].op==MESH_STEP_SEND && steps[r][i].peer==p && MESH_OF(r,i);
    for(;j<count[p];j++)receives+=steps[p][j].op!=MESH_STEP_SEND && steps[p][j].peer==r && MESH_OF(p,j);
#undef MESH_OF
    if(sends!=receives){
      if(at<calls)snprintf(text[0],64,"call %u",at);
      else snprintf(text[0],64,"the point-to-point calls");
      mesh_advise_say(&advice,MESH_ADVISE_PROJECTION,"rank %u sends %u messages to rank %u, which receives %u from it (%s)",r,sends,p,receives,
                      text[0]);
    }
  }
  /* alignment: a SEND's piece is read from its buffer and a COPY's lands in it, each an entry that starts on a page
     (design/transport.md section 14); one off a page fails its call as a value (a combine lands in pages of the
     library's own) */
  for(uint32_t r=0;g->page>1 && r<n;r++)for(uint32_t k=0;k<count[r];k++){
    const struct mesh_step *s=steps[r]+k;
    const uint64_t at=s->first*s->piece.element_bytes;
    if(s->op!=MESH_STEP_REDUCE && mesh_step_bytes(s) && at%g->page)
      mesh_advise_say(&advice,MESH_ADVISE_ALIGNMENT,"rank %u's %s (%s, round %u, rank %u) starts at byte %llu of buffer %u, not on a page of %llu "
                      "bytes: the call fails as a value",r,mesh_move(&grouped,r,k,text[0],sizeof *text),mesh_step_name(s->op),s->round,s->peer,
                      (unsigned long long)at,s->buffer,(unsigned long long)g->page);
  }
  /* projection on the lists: rank a's send list toward b against b's landing list from a (mesh_group_lay), entry for
     entry, each of the same bytes */
  scratch=malloc((size_t)widest*sizeof *scratch);ma=malloc((size_t)most*sizeof *ma);mb=malloc((size_t)most*sizeof *mb);
  if(!scratch || !ma || !mb)goto done;
  uint32_t list_count=0;
  uint64_t events=0;
  int balanced=1;
  for(uint32_t a=0;a<n;a++)for(uint32_t b=0;b<n;b++){
    if(a==b)continue;
    uint32_t na=0,nb=0;
    const uint64_t ea=mesh_group_lay(g,a,b,1,0,scratch,ma,&na),eb=mesh_group_lay(g,b,a,0,0,scratch,mb,&nb);
    if(!ea && !eb)continue;
#define MESH_AT(who,m) (first[(size_t)(m).call*n+(who)]+(m).place.step)
    if(ea!=eb){
      balanced=0;
      mesh_advise_say(&advice,MESH_ADVISE_PROJECTION,"rank %u lays %llu entries on its send list to rank %u, which lays %llu on its landing list "
                      "from it",a,(unsigned long long)ea,b,(unsigned long long)eb);
    }
    for(uint32_t k=0;k<na && k<nb;k++){
      const struct mesh_step *s=steps[a]+MESH_AT(a,ma[k]),*v=steps[b]+MESH_AT(b,mb[k]);
      if(mesh_step_bytes(s)!=mesh_step_bytes(v)){
        balanced=0;
        mesh_advise_say(&advice,MESH_ADVISE_PROJECTION,"entry %u from rank %u to rank %u: its SEND (%s) carries %llu bytes, its %s (%s) %llu",k,a,b,
                        mesh_move(&grouped,a,MESH_AT(a,ma[k]),text[0],sizeof *text),(unsigned long long)mesh_step_bytes(s),mesh_step_name(v->op),
                        mesh_move(&grouped,b,MESH_AT(b,mb[k]),text[1],sizeof *text),(unsigned long long)mesh_step_bytes(v));
      }
    }
    const uint32_t h=list_count++;
    lists[h]=(struct mesh_list){a,b,ea>eb?ea:eb,events};
    events+=2*lists[h].entries;
    for(uint32_t k=0;k<na;k++)spot[a][MESH_AT(a,ma[k])]=(struct mesh_spot){h,ma[k].place.entry};
    for(uint32_t k=0;k<nb;k++)spot[b][MESH_AT(b,mb[k])]=(struct mesh_spot){h,mb[k].place.entry};
#undef MESH_AT
  }
  if(!balanced || !events){result=advice.count;goto done;}
  /* causality: the events under the lists' rule, a marked graph, whose every event occurs (in some order, no timing)
     where it has no cycle (Lee & Messerschmitt's admissible schedule; Lustre's instantaneous dependency check) */
  uint64_t edges=1;
  for(uint32_t r=0;r<n;r++)edges+=2*(uint64_t)count[r]*count[r]+count[r];
  edges+=3*events;
  from=malloc(edges*sizeof *from);to=malloc(edges*sizeof *to);
  start=calloc(events+1,sizeof *start);back=calloc(events+1,sizeof *back);remaining=calloc(events+1,sizeof *remaining);
  queue=malloc((events+1)*sizeof *queue);mark=calloc(events+1,sizeof *mark);
  if(!from || !to || !start || !back || !remaining || !queue || !mark)goto done;
  uint64_t E=0;
#define MESH_EDGE(x,y) do { from[E]=(x);to[E]=(y);E++; } while(0)
  for(uint32_t h=0;h<list_count;h++){
    const struct mesh_list *l=lists+h;
    for(uint64_t e=0;e<l->entries;e++){
      if(e){MESH_EDGE(mesh_event(l,0,e-1),mesh_event(l,0,e));MESH_EDGE(mesh_event(l,1,e-1),mesh_event(l,1,e));}
      MESH_EDGE(mesh_event(l,0,e),mesh_event(l,1,e));
    }
  }
  for(uint32_t r=0;r<n;r++)for(uint32_t k=0;k<count[r];k++){
    const struct mesh_spot *s=spot[r]+k;
    if(s->lane==UINT32_MAX || before[r][k]<0)continue;
    const struct mesh_spot *w=spot[r]+before[r][k];
    if(w->lane==UINT32_MAX)continue;
    MESH_EDGE(mesh_event(lists+w->lane,1,w->entry),mesh_event(lists+s->lane,steps[r][k].op!=MESH_STEP_SEND,s->entry));
  }
  /* a combine into bytes that an earlier SEND of the same rank, call and channel reads (mesh_compile's two-node direct
     exchange) waits until those bytes are sent, after the SEND's publication (libnccl-mesh consume) */
  for(uint32_t r=0;r<n;r++)for(uint32_t k=0;k<count[r];k++){
    const struct mesh_step *s=steps[r]+k;
    const struct mesh_spot *x=spot[r]+k;
    if(s->op!=MESH_STEP_SEND || x->lane==UINT32_MAX)continue;
    for(uint32_t w=k+1;w<count[r];w++){
      const struct mesh_step *v=steps[r]+w;
      const struct mesh_spot *y=spot[r]+w;
      if(v->op==MESH_STEP_REDUCE && owner[r][w]==owner[r][k] && y->lane!=UINT32_MAX && v->peer==s->peer && mesh_overlap(s,v))
        MESH_EDGE(mesh_event(lists+x->lane,0,x->entry),mesh_event(lists+y->lane,1,y->entry));
    }
  }
#undef MESH_EDGE
  list=malloc((E+1)*sizeof *list);rlist=malloc((E+1)*sizeof *rlist);
  if(!list || !rlist)goto done;
  for(uint64_t i=0;i<E;i++){start[from[i]+1]++;back[to[i]+1]++;}
  for(uint64_t v=0;v<events;v++){start[v+1]+=start[v];back[v+1]+=back[v];}
  {
    uint64_t *fill=remaining;
    for(uint64_t v=0;v<events;v++)fill[v]=start[v];
    for(uint64_t i=0;i<E;i++)list[fill[from[i]]++]=to[i];
    for(uint64_t v=0;v<events;v++)fill[v]=back[v];
    for(uint64_t i=0;i<E;i++)rlist[fill[to[i]]++]=from[i];
  }
  uint64_t head=0,tail=0;
  for(uint64_t v=0;v<events;v++){remaining[v]=back[v+1]-back[v];if(!remaining[v])queue[tail++]=v;}
  while(head<tail){
    const uint64_t v=queue[head++];
    for(uint64_t i=start[v];i<start[v+1];i++)if(!--remaining[list[i]])queue[tail++]=list[i];
  }
  if(tail<events){
    /* a witness: back along events that never occur until one repeats, a cycle */
    uint64_t v=0;
    while(v<events && !remaining[v])v++;
    for(uint64_t v2=0;v2<events;v2++)mark[v2]=0;
    uint32_t steps_back=0;
    while(!mark[v]){
      mark[v]=++steps_back;
      queue[steps_back-1]=v;
      uint64_t u=v;
      for(uint64_t i=back[v];i<back[v+1];i++)if(remaining[rlist[i]]){u=rlist[i];break;}
      if(u==v)break;
      v=u;
    }
    /* queue[i + 1] is an event queue[i] waits on: the cycle from v's first place, each waiting on the next */
    char cycle[512]="",event[128];
    size_t used=0;
    const uint32_t first_on=mark[v]-1,length=steps_back-first_on;
    for(uint32_t i=0;i<length && i<6 && used<sizeof cycle;i++){
      mesh_event_text(lists,list_count,queue[first_on+i],event,sizeof event);
      used+=(size_t)snprintf(cycle+used,sizeof cycle-used,"%s%s",i?" <- ":"",event);
    }
    mesh_advise_say(&advice,MESH_ADVISE_CAUSALITY,"%llu of the group's %llu events on its lists can never occur: a cycle of %u, %s%s",
                    (unsigned long long)(events-tail),(unsigned long long)events,length,cycle,length>6?" <- ...":"");
    result=advice.count;
    goto done;
  }
  /* hazard: within a call, a SEND's bytes are read by the NIC from its publication until its peer has them (that
     peer's consumption of it), so a receive after it that writes them (a combine at its consumption, a COPY where its
     peer publishes it, landing in place) must follow that consumption, but for a combine on the SEND's own channel,
     which waits for the bytes sent (write after read); and a receive before a SEND of what it writes must be consumed
     before the SEND's publication (read after write) */
  uint32_t generation=0;
  for(uint32_t r=0;r<n;r++)for(uint32_t k=0;k<count[r];k++){
    const struct mesh_step *s=steps[r]+k;
    const struct mesh_spot *x=spot[r]+k;
    if(s->op!=MESH_STEP_SEND || x->lane==UINT32_MAX)continue;
    mesh_reach(mesh_event(lists+x->lane,1,x->entry),start,list,mark,++generation,queue);
    for(uint32_t w=k+1;w<count[r];w++){
      const struct mesh_step *v=steps[r]+w;
      const struct mesh_spot *y=spot[r]+w;
      if(owner[r][w]!=owner[r][k] || v->op==MESH_STEP_SEND || y->lane==UINT32_MAX || !mesh_overlap(s,v) ||
         (v->op==MESH_STEP_REDUCE && v->peer==s->peer) || mark[mesh_written(lists,v,y)]==generation)
        continue;
      mesh_advise_say(&advice,MESH_ADVISE_HAZARD,"rank %u's %s %s (round %u, from rank %u) writes buffer %u's elements [%llu, +%llu) that its "
                      "SEND %s (round %u, to rank %u) reads, with no order from the SEND's landing at rank %u to that write",r,mesh_step_name(v->op),
                      mesh_move(&grouped,r,w,text[0],sizeof *text),v->round,v->peer,v->buffer,(unsigned long long)v->first,
                      (unsigned long long)v->piece.elements,mesh_move(&grouped,r,k,text[1],sizeof *text),s->round,s->peer,s->peer);
    }
    mesh_reach(mesh_event(lists+x->lane,0,x->entry),back,rlist,mark,++generation,queue);
    for(uint32_t w=0;w<k;w++){
      const struct mesh_step *v=steps[r]+w;
      const struct mesh_spot *y=spot[r]+w;
      if(owner[r][w]!=owner[r][k] || v->op==MESH_STEP_SEND || y->lane==UINT32_MAX || !mesh_overlap(s,v))continue;
      if(mark[mesh_event(lists+y->lane,1,y->entry)]!=generation)
        mesh_advise_say(&advice,MESH_ADVISE_HAZARD,"rank %u's SEND %s (round %u, to rank %u) reads buffer %u's elements [%llu, +%llu) that its %s "
                        "%s (round %u, from rank %u) writes, with no order from the receive's consumption to the SEND's publication",r,
                        mesh_move(&grouped,r,k,text[0],sizeof *text),s->round,s->peer,s->buffer,(unsigned long long)s->first,
                        (unsigned long long)s->piece.elements,mesh_step_name(v->op),mesh_move(&grouped,r,w,text[1],sizeof *text),v->round,v->peer);
    }
  }
  result=advice.count;
done:
  for(uint32_t r=0;r<n;r++){
    if(steps)free(steps[r]);
    if(before)free(before[r]);
    if(spot)free(spot[r]);
    if(owner)free(owner[r]);
  }
  free(steps);free(before);free(spot);free(owner);free(first);free(count);free(lists);free(scratch);free(ma);free(mb);free(from);free(to);
  free(start);free(list);free(back);free(rlist);free(queue);free(remaining);free(mark);
  return result;
}

uint32_t mesh_program_advise(const struct mesh_program *program,struct mesh_operand operand,uint64_t page,void (*warn)(void *,const char *),
                             void *arg){
  const uint32_t n=program->nodes,segments=mesh_program_segments(program);
  const int nccl=program->what==MESH_REDUCE_SCATTER || program->what==MESH_ALLGATHER;
  struct mesh_step **steps=calloc((size_t)n+1,sizeof *steps);
  uint32_t *count=calloc((size_t)n+1,sizeof *count),result=UINT32_MAX;
  uint64_t *sizes=malloc(((size_t)segments+1)*sizeof *sizes);
  for(uint32_t s=0;sizes && s<segments;s++)sizes[s]=operand.elements/segments+(s<operand.elements%segments);
  for(uint32_t r=0;steps && count && sizes && r<n;r++){
    if(!(steps[r]=malloc((program->first[r+1]-program->first[r]+1)*sizeof **steps)))goto done;
    count[r]=mesh_program_steps_cut(program,r,operand,nccl?sizes:NULL,page,steps[r]);
  }
  if(steps && count && sizes){
    const struct mesh_group_call call={MESH_GROUP_COLLECTIVE,0,count,(const struct mesh_step *const *)steps};
    const struct mesh_group group={n,1,0,0,&call,page};
    result=mesh_group_advise(&group,warn,arg);
  }
done:
  for(uint32_t r=0;steps && r<n;r++)free(steps[r]);
  free(steps);free(count);free(sizes);
  return result;
}
