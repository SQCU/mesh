#include "mesh-collective.h"
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* design/algorithm-sources.md#topology: every node computes the same trees, routes, programs and choices from the same
   map, so no product is fused into a sum where one node's compiler might fuse it and another's not */
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

/* The cost of the links joining u and x (either way): each its own where the map gives one, else alpha and beta; several
   such links (cables between one pair: design/bridge-and-ipc.md#cables, a program striped over them) one link of the
   least alpha and their bandwidths summed, 1/beta = the sum of their 1/beta.  alpha and beta where none joins them. */
static void mesh_hop_cost(const struct mesh_link_map *map,uint32_t u,uint32_t x,double alpha,double beta,double *link_alpha,double *link_beta){
  double least=INFINITY,rate=0;
  for(uint32_t l=0;l<map->links;l++)
    if((map->link[l][0]==u && map->link[l][1]==x) || (map->link[l][0]==x && map->link[l][1]==u)){
      const int own=map->cost && !isnan(map->cost[l][0]);
      const double a=own?map->cost[l][0]:alpha,b=own?map->cost[l][1]:beta;
      if(a<least)least=a;
      rate+=b>0?1/b:INFINITY;
    }
  *link_alpha=least<INFINITY?least:alpha;
  *link_beta=rate>0?(isinf(rate)?0:1/rate):beta;
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
    uint32_t path[N+1];
    for(uint32_t a=0;a<n;a++)for(uint32_t b=0;b<n;b++)if(a!=b && !mesh_route_nodes(map,a,b,path))return EINVAL;
    return 0;
  }
  if(map->kind==MESH_LINKS_MESH)return 0;
  uint32_t *up=malloc((size_t)n*sizeof *up),*order=malloc((size_t)n*sizeof *order);
  if(!up || !order){free(up);free(order);return ENOMEM;}
  int result=0;
  if(map->kind==MESH_LINKS_GRAPH){
    /* connected: breadth first from node 0 over the links either way */
    uint32_t reached=1;
    for(uint32_t v=0;v<n;v++)up[v]=n;
    up[0]=0;order[0]=0;
    for(uint32_t head=0;head<reached;head++)for(uint32_t l=0;l<map->links;l++)for(uint32_t e=0;e<2;e++)
      if(map->link[l][e]==order[head] && up[map->link[l][1-e]]==n){up[map->link[l][1-e]]=order[head];order[reached++]=map->link[l][1-e];}
    result=reached==n?0:EINVAL;
  }else{
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
  }
  free(up);free(order);
  return result;
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

/* The hop a-b's cost: its link's own where the map gives one, else alpha and beta; with a table, its route's links'
   alphas summed and the greatest of their betas (a relay forwards each slot as it lands). */
void mesh_link_cost(const struct mesh_link_map *map,uint32_t a,uint32_t b,double alpha,double beta,double *link_alpha,double *link_beta){
  uint32_t path[mesh_all(map)+2];
  const uint32_t count=map->next?mesh_route_nodes(map,a,b,path):0;
  if(count<2){mesh_hop_cost(map,a,b,alpha,beta,link_alpha,link_beta);return;}
  *link_alpha=0;*link_beta=0;
  for(uint32_t i=0;i+1<count;i++){
    double x,y;
    mesh_hop_cost(map,path[i],path[i+1],alpha,beta,&x,&y);
    *link_alpha+=x;
    if(y>*link_beta)*link_beta=y;
  }
}

/* -- routes -------------------------------------------------------------------------------------------- */

/* 48 bits of splitmix64's finalizer [Steele, Lea & Flood, OOPSLA 2014] over three marks */
static uint64_t mesh_hash(uint32_t x,uint32_t y,uint32_t z){
  uint64_t h=(((uint64_t)x<<32)|y)^((uint64_t)z*UINT64_C(0x9e3779b97f4a7c15));
  h=(h^(h>>30))*UINT64_C(0xbf58476d1ce4e5b9);
  h=(h^(h>>27))*UINT64_C(0x94d049bb133111eb);
  return (h^(h>>31))>>16;
}
/* A route's length toward a root: its links, then the sum of their betas, compared in that order */
struct mesh_length { uint64_t hops,betas; };
static int mesh_shorter(struct mesh_length a,struct mesh_length b){ return a.hops!=b.hops?a.hops<b.hops:a.betas<b.betas; }
static int mesh_equal(struct mesh_length a,struct mesh_length b){ return a.hops==b.hops && a.betas==b.betas; }
/* the nodes ordered for a root's in-tree: farther first, then by mark */
struct mesh_far { struct mesh_length length; uint32_t mark,node; };
static int mesh_farther(const void *x,const void *y){
  const struct mesh_far *a=x,*b=y;
  if(!mesh_equal(a->length,b->length))return mesh_shorter(a->length,b->length)?1:-1;
  return (a->mark>b->mark)-(a->mark<b->mark);
}

/* A root's distances by Dijkstra [1959] in its dense form (every length an exact integer, so every node finds the same) */
static void mesh_distances(uint32_t N,const unsigned char *joined,const uint64_t *beta,uint32_t b,struct mesh_length *length,unsigned char *done){
  for(uint32_t v=0;v<N;v++){length[v]=(struct mesh_length){UINT64_MAX,0};done[v]=0;}
  length[b]=(struct mesh_length){0,0};
  for(;;){
    uint32_t at=N;
    for(uint32_t v=0;v<N;v++)if(!done[v] && length[v].hops!=UINT64_MAX && (at==N || mesh_shorter(length[v],length[at])))at=v;
    if(at==N)break;
    done[at]=1;
    for(uint32_t w=0;w<N;w++){
      if(done[w] || !joined[(size_t)at*N+w])continue;
      const struct mesh_length d={length[at].hops+1,length[at].betas+beta[(size_t)at*N+w]};
      if(length[w].hops==UINT64_MAX || mesh_shorter(d,length[w]))length[w]=d;
    }
  }
}
/* `flow` routes put on (sign 1) or taken off (-1) the links of v's route up root b's in-tree */
static void mesh_lay(uint32_t N,const uint32_t *next,uint64_t *load,uint32_t v,uint32_t b,uint64_t flow,int sign){
  for(uint32_t at=v,hops=0;at!=b && hops<N && next[(size_t)at*N+b]!=at;hops++){
    const uint32_t up=next[(size_t)at*N+b];
    if(sign>0){load[(size_t)at*N+up]+=flow;load[(size_t)up*N+at]+=flow;}
    else{load[(size_t)at*N+up]-=flow;load[(size_t)up*N+at]-=flow;}
    at=up;
  }
}
/* Each root's in-tree of shortest routes (fewest links, then the least betas): first each node's parent among those one
   equal step nearer the least hash of the three marks; then MESH_ROUTE_PASSES passes of local search on the sum of the
   squares of the links' loads (each the routes between two nodes that cross it: the potential of the congestion game
   [Rosenthal 1973], which every move lowers), each root in mark order, each node farthest first: its flow (the routes
   toward the root it carries: its own, where its mark is the lower, and its subtree's) taken off its route, then put
   on through the parent whose route adds least to the potential (then the least hash); the table of the pass whose most
   loaded link carries fewest routes kept (then the least potential, then the earliest).  Integers throughout. */
#define MESH_ROUTE_PASSES 4
int mesh_routes(const struct mesh_link_map *map,uint32_t *next){
  const uint32_t N=map->nodes+map->relays;
  if(!N || map->kind>MESH_LINKS_GRAPH || (map->links && !map->link) || (map->relays && map->kind==MESH_LINKS_MESH))return EINVAL;
  for(uint32_t l=0;l<map->links;l++)if(map->link[l][0]>=N || map->link[l][1]>=N || map->link[l][0]==map->link[l][1])return EINVAL;
  const size_t pairs=(size_t)N*N;
  uint64_t *beta=calloc(pairs,sizeof *beta),*load=calloc(pairs,sizeof *load),*flow=calloc(N,sizeof *flow);
  unsigned char *joined=calloc(pairs,1),*done=malloc(N);
  uint32_t *trial=malloc(pairs*sizeof *trial);
  struct mesh_length *length=malloc(pairs*sizeof *length);
  struct mesh_far *far=malloc(pairs*sizeof *far),*roots=malloc(N*sizeof *roots);
  int status=beta && load && flow && joined && done && trial && length && far && roots?0:ENOMEM;
  /* a pair's hop: joined where a link joins it (a mesh's every pair), at its links' beta together (mesh_hop_cost; a
     link without a cost counts none) */
  for(uint32_t u=0;!status && map->kind==MESH_LINKS_MESH && u<N;u++)for(uint32_t x=0;x<N;x++)joined[(size_t)u*N+x]=u!=x;
  for(uint32_t l=0;!status && l<map->links;l++){
    const uint32_t u=map->link[l][0],x=map->link[l][1];
    double a,b;
    mesh_hop_cost(map,u,x,0,0,&a,&b);
    const uint64_t q=b>0 && isfinite(b)?(uint64_t)llround(b*1e6):0;
    beta[(size_t)u*N+x]=beta[(size_t)x*N+u]=q;
    joined[(size_t)u*N+x]=joined[(size_t)x*N+u]=1;
  }
  for(uint32_t v=0;!status && v<N;v++)roots[v]=(struct mesh_far){{0,0},mesh_mark(map,v),v};
  if(!status)qsort(roots,N,sizeof *roots,mesh_farther);
  /* root k's distances at length + k N, and its nodes but itself farthest first at far + k N (`reached[k]` of them) */
  uint32_t *reached=status?NULL:calloc(N,sizeof *reached);
  if(!status && !reached)status=ENOMEM;
  for(uint32_t k=0;!status && k<N;k++){
    const uint32_t b=roots[k].node;
    struct mesh_length *at=length+(size_t)k*N;
    mesh_distances(N,joined,beta,b,at,done);
    for(uint32_t v=0;v<N;v++)if(v!=b && at[v].hops!=UINT64_MAX)far[(size_t)k*N+reached[k]++]=(struct mesh_far){at[v],mesh_mark(map,v),v};
    qsort(far+(size_t)k*N,reached[k],sizeof *far,mesh_farther);
    for(uint32_t v=0;v<N;v++)trial[(size_t)v*N+b]=v;
    for(uint32_t i=reached[k];i-->0;){
      const uint32_t v=far[(size_t)k*N+i].node;
      uint32_t up=v;
      for(uint32_t u=0;u<N;u++)
        if(joined[(size_t)v*N+u] && at[u].hops!=UINT64_MAX &&
           mesh_equal((struct mesh_length){at[u].hops+1,at[u].betas+beta[(size_t)v*N+u]},at[v]) &&
           (up==v || mesh_hash(mesh_mark(map,v),mesh_mark(map,u),roots[k].mark)<mesh_hash(mesh_mark(map,v),mesh_mark(map,up),roots[k].mark)))
          up=u;
      trial[(size_t)v*N+b]=up;
    }
  }
  for(uint32_t k=0;!status && k<N;k++)
    for(uint32_t a=0;a<N;a++)if(mesh_mark(map,a)<roots[k].mark)mesh_lay(N,trial,load,a,roots[k].node,1,1);
  uint64_t least=UINT64_MAX,least_potential=UINT64_MAX;
  for(uint32_t pass=0;!status && pass<=MESH_ROUTE_PASSES;pass++){
    for(uint32_t k=0;pass && k<N;k++){
      const uint32_t b=roots[k].node;
      const struct mesh_length *at=length+(size_t)k*N;
      for(uint32_t v=0;v<N;v++)flow[v]=v!=b && mesh_mark(map,v)<roots[k].mark;
      for(uint32_t i=0;i<reached[k];i++){
        const uint32_t v=far[(size_t)k*N+i].node,f=(uint32_t)flow[v];
        uint32_t up=trial[(size_t)v*N+b];
        if(f){
          mesh_lay(N,trial,load,v,b,f,-1);
          uint64_t best=UINT64_MAX;
          for(uint32_t u=0;u<N;u++){
            if(!joined[(size_t)v*N+u] || at[u].hops==UINT64_MAX ||
               !mesh_equal((struct mesh_length){at[u].hops+1,at[u].betas+beta[(size_t)v*N+u]},at[v]))continue;
            uint64_t price=2*load[(size_t)v*N+u]*f+(uint64_t)f*f;
            for(uint32_t x=u,hops=0;x!=b && hops<N;hops++){
              const uint32_t y=trial[(size_t)x*N+b];
              price+=2*load[(size_t)x*N+y]*f+(uint64_t)f*f;
              x=y;
            }
            if(best==UINT64_MAX || price<best ||
               (price==best && mesh_hash(mesh_mark(map,v),mesh_mark(map,u),roots[k].mark)<mesh_hash(mesh_mark(map,v),mesh_mark(map,up),roots[k].mark))){
              best=price;up=u;
            }
          }
          trial[(size_t)v*N+b]=up;
          mesh_lay(N,trial,load,v,b,f,1);
        }
        flow[up]+=flow[v];
      }
    }
    uint64_t most=0,potential=0;
    for(size_t k=0;k<pairs;k++){if(load[k]>most)most=load[k];potential+=load[k]*load[k];}
    if(most<least || (most==least && potential<least_potential)){least=most;least_potential=potential;memcpy(next,trial,pairs*sizeof *next);}
  }
  free(beta);free(load);free(flow);free(joined);free(done);free(trial);free(length);free(far);free(roots);free(reached);
  return status;
}

int mesh_map_members(const struct mesh_link_map *whole,const uint32_t *members,uint32_t n,uint32_t *whole_node,struct mesh_link_map *map){
  const uint32_t W=whole->nodes,N=n?n:W;
  if(whole->relays || (whole->links && !whole->link) || whole->kind>MESH_LINKS_GRAPH)return -EINVAL;
  /* the whole map's links as listed, or a mesh's every pair (its listed cost where it has one) */
  const int every=whole->kind==MESH_LINKS_MESH;
  const uint32_t L=every?W*(W?W-1:0)/2:whole->links;
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
      double alpha=NAN,beta=NAN;
      for(uint32_t k=0;whole->cost && k<whole->links;k++)
        if((whole->link[k][0]==a && whole->link[k][1]==b) || (whole->link[k][0]==b && whole->link[k][1]==a)){alpha=whole->cost[k][0];beta=whole->cost[k][1];}
      map->link[l][0]=place[a];map->link[l][1]=place[b];
      if(map->cost){map->cost[l][0]=alpha;map->cost[l][1]=beta;}
      l++;
    }
    for(uint32_t k=0;!every && k<L;k++){
      map->link[k][0]=place[whole->link[k][0]];map->link[k][1]=place[whole->link[k][1]];
      if(map->cost){map->cost[k][0]=whole->cost?whole->cost[k][0]:NAN;map->cost[k][1]=whole->cost?whole->cost[k][1]:NAN;}
    }
    for(uint32_t i=0;map->next && i<T;i++)for(uint32_t j=0;j<T;j++)
      map->next[(size_t)i*T+j]=at[i]<W && at[j]<W?place[table[(size_t)at[i]*W+at[j]]]:i;
    for(uint32_t i=0;whole_node && i<T;i++)whole_node[i]=at[i];
    for(uint32_t i=0;map->mark && i<T;i++)map->mark[i]=at[i]<W?mesh_mark(whole,at[i]):UINT32_MAX-i;
    map->nodes=N;map->links=L;map->relays=relays;
    map->kind=!relays && (uint64_t)L*2==(uint64_t)N*(N?N-1:0)?MESH_LINKS_MESH:MESH_LINKS_GRAPH;
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

uint32_t mesh_trees_shortest(const struct mesh_link_map *map,uint32_t *root,uint32_t *parent,double *log_weight){
  const uint32_t n=map->nodes,N=mesh_all(map);
  if(!n || (!map->next && map->relays))return 0;
  struct mesh_link_map with=*map;
  uint32_t *made=map->next?NULL:malloc((size_t)N*N*sizeof *made),*path=malloc(((size_t)N+1)*sizeof *path),count=n;
  if(made && !mesh_routes(map,made))with.next=made;
  if(!path || !with.next)count=0;
  for(uint32_t r=0;r<n && count;r++){
    root[r]=r;log_weight[r]=0;
    /* a node's parent: the first of the map's nodes its route to r reaches (r itself across relays alone); each a
       step of a shortest route, so the parents climb to r */
    for(uint32_t v=0;v<n && count;v++){
      uint32_t up=v;
      const uint32_t hops=v==r?0:mesh_route_nodes(&with,v,r,path);
      for(uint32_t i=1;i<hops;i++)if(path[i]<n){up=path[i];break;}
      if(v!=r && up==v)count=0;
      parent[(size_t)r*n+v]=v==r?r:up;
    }
  }
  free(made);free(path);
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
       has children there, else its own operand from its copy (scratch 1), which nothing changes; chunk j its part j */
    parts=c;
    for(uint32_t j=0;j<c;j++){segment[j]=0;log_weight[j]=0;}
    for(uint32_t u=0;u<n;u++){
      uint32_t next=2;
      for(uint32_t v=0;v<n;v++){
        const uint32_t t=mesh_trees_heaviest(trees,v);
        if(t==T)goto done;
        if(u!=v && shape[(size_t)t*n+u].inner)buffer[(size_t)u*n+v]=next++;
      }
      if(next-1>scratch)scratch=next-1;
    }
    for(uint32_t v=0;v<n;v++){
      const uint32_t t=mesh_trees_heaviest(trees,v),*parent=trees->parent+(size_t)t*n;
      const struct mesh_shape *s=shape+(size_t)t*n;
      for(uint32_t u=0;u<n;u++){
        if(u==v || !s[u].live)continue;
        const uint32_t w=parent[u],op=!mesh_member(mine,w) && mesh_first_child(trees,t,s,w,u)?MESH_STEP_COPY:MESH_STEP_REDUCE;
        for(uint32_t j=0;j<c;j++)
          message[messages++]=(struct mesh_message){u,w,s[u].height+j,j,1,s[u].inner?buffer[(size_t)u*n+v]:1,w==v?0:buffer[(size_t)w*n+v],op,v};
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

/* Each part's elements of an operand: [lo[p], lo[p] + count[p]) */
static void mesh_program_ranges(const struct mesh_program *program,uint64_t elements,uint64_t *lo,uint64_t *count){
  uint32_t segments=1;
  for(uint32_t p=0;p<program->parts;p++)if(program->segment[p]+1>segments)segments=program->segment[p]+1;
  const uint64_t size=elements/segments,residual=elements%segments;
  for(uint32_t g=0;g<segments;g++){
    const uint64_t base=g*size+(g<residual?g:residual),length=size+(g<residual);
    double most=-INFINITY,total=0;
    for(uint32_t p=0;p<program->parts;p++)if(program->segment[p]==g && program->log_weight[p]>most)most=program->log_weight[p];
    for(uint32_t p=0;p<program->parts;p++)if(program->segment[p]==g)total+=mesh_exp(program->log_weight[p]-most);
    uint64_t floors=0;
    for(uint32_t p=0;p<program->parts;p++)
      if(program->segment[p]==g)floors+=(uint64_t)floor((double)length*mesh_exp(program->log_weight[p]-most)/total);
    uint64_t remainder=floors<length?length-floors:0,at=base;
    for(uint32_t p=0;p<program->parts;p++){
      if(program->segment[p]!=g)continue;
      uint64_t share=(uint64_t)floor((double)length*mesh_exp(program->log_weight[p]-most)/total);
      if(remainder && at+share<base+length){share++;remainder--;}
      if(at+share>base+length)share=base+length-at;
      lo[p]=at;count[p]=share;at+=share;
    }
  }
}

uint32_t mesh_program_steps(const struct mesh_program *program,uint32_t rank,struct mesh_operand operand,struct mesh_step *steps){
  if(rank>=program->nodes || !program->parts)return 0;
  uint64_t *lo=malloc(program->parts*sizeof *lo),*count=malloc(program->parts*sizeof *count);
  uint32_t k=0;
  if(lo && count){
    mesh_program_ranges(program,operand.elements,lo,count);
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
   none.  Each node's cheapest edge; a cycle among them contracted, the edges into it costing their excess over the
   edge they would replace, the contracted graph's arborescence expanded (the edge into a cycle replaces the cycle's
   edge into the same node). */
static int mesh_arborescence(uint32_t N,uint32_t root,uint32_t E,const uint32_t *from,const uint32_t *to,const double *cost,int32_t *chosen){
  int32_t *best=malloc(N*sizeof *best),*id=malloc(N*sizeof *id),*seen=malloc(N*sizeof *seen);
  uint32_t *nfrom=malloc((E+1)*sizeof *nfrom),*nto=malloc((E+1)*sizeof *nto),*origin=malloc((E+1)*sizeof *origin);
  double *ncost=malloc((E+1)*sizeof *ncost);
  int32_t *nchosen=malloc(N*sizeof *nchosen);
  unsigned char *cyclic=calloc(N,1);
  int status=-1;
  if(!best || !id || !seen || !nfrom || !nto || !origin || !ncost || !nchosen || !cyclic)goto done;
  for(uint32_t v=0;v<N;v++){best[v]=-1;id[v]=-1;seen[v]=-1;}
  for(uint32_t e=0;e<E;e++)
    if(from[e]!=to[e] && to[e]!=root && (best[to[e]]<0 || cost[e]<cost[best[to[e]]]))best[to[e]]=(int32_t)e;
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
    ncost[m]=cyclic[w]?cost[e]-cost[best[to[e]]]:cost[e];
    m++;
  }
  if(mesh_arborescence(components,(uint32_t)id[root],m,nfrom,nto,ncost,nchosen))goto done;
  for(uint32_t v=0;v<N;v++)chosen[v]=v==root?-1:best[v];
  for(uint32_t x=0;x<components;x++)
    if(x!=(uint32_t)id[root]){const uint32_t e=origin[nchosen[x]];chosen[to[e]]=(int32_t)e;}
  status=0;
done:
  free(best);free(id);free(seen);free(nfrom);free(nto);free(origin);free(ncost);free(nchosen);free(cyclic);
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
  if(n<2 || !most || mesh_link_map_check(map))return 0;
  if(!(eps>0 && eps<1))eps=0.1;
  uint32_t m=0,P=0;
  for(uint32_t a=0;a<n;a++)for(uint32_t b=0;b<n;b++)m+=mesh_link_between(map,a,b);
  /* hop v -> w (v's parent w): edge e, from[e] = w, to[e] = v, the ports its route crosses crossing[through[e] ..
     through[e + 1]); port k (one way of a link, u to x, numbered as first met) its capacity 1/beta of that link */
  uint32_t *from=malloc((m+1)*sizeof *from),*to=malloc((m+1)*sizeof *to),*through=malloc((m+2)*sizeof *through);
  uint32_t *crossing=malloc(((size_t)m*N+1)*sizeof *crossing),*port=malloc((size_t)N*N*sizeof *port),*uses=calloc((size_t)m*N+1,sizeof *uses);
  double *capacity=malloc(((size_t)m*N+1)*sizeof *capacity),*length=malloc(((size_t)m*N+1)*sizeof *length),*cost=malloc((m+1)*sizeof *cost);
  int32_t *chosen=malloc(n*sizeof *chosen);
  struct mesh_packed *packed=NULL;
  size_t trees=0,room=0;
  uint32_t written=0,path[N+1];
  if(!from || !to || !through || !crossing || !port || !uses || !capacity || !length || !cost || !chosen)goto done;
  for(size_t k=0;k<(size_t)N*N;k++)port[k]=UINT32_MAX;
  m=0;through[0]=0;
  for(uint32_t v=0;v<n;v++)for(uint32_t w=0;w<n;w++)if(mesh_link_between(map,v,w)){
    const uint32_t hops=mesh_route_nodes(map,v,w,path);
    through[m+1]=through[m];
    for(uint32_t i=0;i+1<hops;i++){
      uint32_t *k=port+(size_t)path[i]*N+path[i+1];
      if(*k==UINT32_MAX){
        double alpha,beta;
        mesh_hop_cost(map,path[i],path[i+1],0,1,&alpha,&beta);
        capacity[P]=beta>0 && isfinite(beta)?1/beta:1;*k=P++;
      }
      crossing[through[m+1]++]=*k;
    }
    from[m]=w;to[m]=v;m++;
  }
  const double delta=mesh_exp(-mesh_log(P/(1-eps))/eps);
  double D=0;
  for(uint32_t k=0;k<P;k++){length[k]=delta/capacity[k];D+=length[k]*capacity[k];}
  while(D<1)for(uint32_t r=0;r<n && D<1;r++)for(double demand=1;demand>0 && D<1;){
    for(uint32_t e=0;e<m;e++){cost[e]=0;for(uint32_t i=through[e];i<through[e+1];i++)cost[e]+=length[crossing[i]];}
    if(mesh_arborescence(n,r,m,from,to,cost,chosen))goto done;
    /* the tree's flow: its demand, at most each port's capacity over the tree's hops crossing it */
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
  free(packed);free(from);free(to);free(through);free(crossing);free(port);free(uses);free(capacity);free(length);free(cost);free(chosen);
  return written;
}

/* A step's place among the messages between two ranks in a round: sender, receiver, round, its ordinal there, and its
   index among every rank's steps */
struct mesh_key { uint32_t from,to,round,ordinal,index; };
static int mesh_key_order(const void *a,const void *b){
  const struct mesh_key *x=a,*y=b;
  const uint32_t l[5]={x->from,x->to,x->round,x->ordinal,x->index},r[5]={y->from,y->to,y->round,y->ordinal,y->index};
  for(int i=0;i<5;i++)if(l[i]!=r[i])return l[i]<r[i]?-1:1;
  return 0;
}

/* Every rank's steps of a program, then its time under the alpha-beta model [Hockney 1994] with a node's sends sharing
   each outgoing port and its receives each incoming one, as Thakur, Rabenseifner and Gropp cost these algorithms: each
   node takes its steps in order; a SEND leaves when the node has reached it, crossing each port of its route once that
   port is free (a relay passes it on its link's alpha after it began to land, finishing no sooner than the port before
   it plus that alpha), and arrives the last link's alpha after its last port finished; a receive waits for its SEND
   (the sender's k-th to it in the round is its k-th from the sender) and its last port.  The time is the last node's;
   negative where a receive has no SEND of its piece, or the schedule stops. */
double mesh_program_time(const struct mesh_link_map *map,const struct mesh_program *program,struct mesh_operand operand,double alpha,double beta){
  const uint32_t n=program->nodes,N=mesh_all(map),total=program->first[n];
  if(n<2)return 0;
  if(map->nodes!=n)return -1;
  struct mesh_step *steps=calloc((size_t)total+1,sizeof *steps);
  struct mesh_key *sends=malloc(((size_t)total+1)*sizeof *sends),*receives=malloc(((size_t)total+1)*sizeof *receives);
  double *arrival=calloc((size_t)total+1,sizeof *arrival),result=-1;
  int64_t *match=malloc(((size_t)total+1)*sizeof *match);
  /* each rank pair's route: its ports (u x N + x, one way of the link u-x) and their links' alpha and beta */
  uint32_t *hops=calloc((size_t)n*n,sizeof *hops),*hop=malloc(((size_t)n*n*N+1)*sizeof *hop),path[N+1];
  double *hop_alpha=malloc(((size_t)n*n*N+1)*sizeof *hop_alpha),*hop_beta=malloc(((size_t)n*n*N+1)*sizeof *hop_beta);
  double *out_free=calloc((size_t)N*N,sizeof *out_free),*in_free=calloc((size_t)N*N,sizeof *in_free);
  uint32_t count[n],cursor[n],S=0,R=0;
  double clock[n];
  memset(cursor,0,sizeof cursor);memset(clock,0,sizeof clock);
  if(!steps || !sends || !receives || !arrival || !match || !hops || !hop || !hop_alpha || !hop_beta || !out_free || !in_free)goto done;
  for(uint32_t a=0;a<n;a++)for(uint32_t b=0;b<n;b++){
    const size_t pair=(size_t)a*n+b;
    uint32_t h=a==b?0:mesh_route_nodes(map,a,b,path);
    if(a!=b && h<2){path[0]=a;path[1]=b;h=2;}
    for(uint32_t i=0;i+1<h;i++){
      hop[pair*N+i]=path[i]*N+path[i+1];
      mesh_hop_cost(map,path[i],path[i+1],alpha,beta,hop_alpha+pair*N+i,hop_beta+pair*N+i);
    }
    hops[pair]=h?h-1:0;
  }
  /* every receive's SEND: a step's ordinal among its rank's steps of its kind toward (from) its peer in its round */
  for(uint32_t r=0;r<n;r++){
    count[r]=mesh_program_steps(program,r,operand,steps+program->first[r]);
    for(uint32_t k=0;k<count[r];k++){
      const struct mesh_step *step=steps+program->first[r]+k;
      const uint32_t index=program->first[r]+k;
      if(step->op==MESH_STEP_SEND)sends[S++]=(struct mesh_key){r,step->peer,step->round,0,index};
      else receives[R++]=(struct mesh_key){step->peer,r,step->round,0,index};
    }
  }
  qsort(sends,S,sizeof *sends,mesh_key_order);qsort(receives,R,sizeof *receives,mesh_key_order);
  for(uint32_t i=1;i<S;i++)if(sends[i].from==sends[i-1].from && sends[i].to==sends[i-1].to && sends[i].round==sends[i-1].round)sends[i].ordinal=sends[i-1].ordinal+1;
  for(uint32_t i=1;i<R;i++)if(receives[i].from==receives[i-1].from && receives[i].to==receives[i-1].to && receives[i].round==receives[i-1].round)receives[i].ordinal=receives[i-1].ordinal+1;
  for(uint32_t i=0;i<total;i++)match[i]=-1;
  for(uint32_t i=0,j=0;i<R;i++){
    while(j<S && mesh_key_order(&(struct mesh_key){sends[j].from,sends[j].to,sends[j].round,sends[j].ordinal,0},
                                &(struct mesh_key){receives[i].from,receives[i].to,receives[i].round,receives[i].ordinal,0})<0)j++;
    if(j<S && sends[j].from==receives[i].from && sends[j].to==receives[i].to && sends[j].round==receives[i].round &&
       sends[j].ordinal==receives[i].ordinal)match[receives[i].index]=sends[j].index;
  }
  for(int progress=1;progress;){
    progress=0;
    for(uint32_t r=0;r<n;r++)while(cursor[r]<count[r]){
      const uint32_t index=program->first[r]+cursor[r];
      const struct mesh_step *step=steps+index;
      const double bytes=(double)step->piece.elements*step->piece.element_bytes;
      const size_t pair=step->op==MESH_STEP_SEND?(size_t)r*n+step->peer:(size_t)step->peer*n+r;
      if(step->op==MESH_STEP_SEND){
        double ready=clock[r],before=0,end=0;
        for(uint32_t i=0;i<hops[pair];i++){
          double *port=out_free+hop[pair*N+i];
          const double start=ready>*port?ready:*port;
          end=start+bytes*hop_beta[pair*N+i]/1e3;
          if(i && end<before+hop_alpha[pair*N+i-1])end=before+hop_alpha[pair*N+i-1];
          *port=end;
          ready=start+hop_alpha[pair*N+i];
          before=end;
        }
        arrival[index]=end+(hops[pair]?hop_alpha[pair*N+hops[pair]-1]:0);
        cursor[r]++;
        progress=1;
        continue;
      }
      const int64_t k=match[index];
      const struct mesh_step *send=k<0?NULL:steps+k;
      if(!send || send->first!=step->first || send->piece.elements!=step->piece.elements ||
         send->piece.element_bytes!=step->piece.element_bytes)goto done;
      if(program->first[step->peer]+cursor[step->peer]<=(uint64_t)k)break;
      const uint32_t last=hops[pair]?hops[pair]-1:0;
      double *port=in_free+(hops[pair]?hop[pair*N+last]:(size_t)step->peer*N+r);
      double complete=*port+bytes*(hops[pair]?hop_beta[pair*N+last]:beta)/1e3;
      if(arrival[k]>complete)complete=arrival[k];
      *port=complete;
      if(complete>clock[r])clock[r]=complete;
      cursor[r]++;
      progress=1;
    }
  }
  result=0;
  for(uint32_t r=0;r<n;r++){
    if(cursor[r]<count[r]){result=-1;break;}
    if(clock[r]>result)result=clock[r];
  }
done:
  free(steps);free(sends);free(receives);free(arrival);free(match);free(hops);free(hop);free(hop_alpha);free(hop_beta);free(out_free);free(in_free);
  return result;
}

/* A candidate's time: `what` along trees at chunks, compiled into rooms of its own and timed; 0, ENOMEM, or EINVAL */
static int mesh_candidate(const struct mesh_link_map *map,const struct mesh_trees *trees,uint32_t what,uint32_t root,int whole,uint32_t chunks,
                          struct mesh_operand operand,double alpha,double beta,double *time){
  uint32_t parts,moves;
  mesh_compile_room(trees,chunks,&parts,&moves);
  uint32_t *segment=malloc((parts+1)*sizeof *segment),*first=malloc((trees->nodes+1)*sizeof *first);
  double *weight=malloc((parts+1)*sizeof *weight);
  struct mesh_move *move=malloc((moves+1)*sizeof *move);
  struct mesh_program program;
  int status=!segment || !first || !weight || !move?ENOMEM:
    mesh_compile(trees,what,root,whole,chunks,NULL,&program,segment,weight,first,move)?EINVAL:0;
  if(!status)*time=mesh_program_time(map,&program,operand,alpha,beta);
  if(!status && *time<0)status=EINVAL;
  free(segment);free(first);free(weight);free(move);
  return status;
}

int mesh_choose(const struct mesh_link_map *map,const struct mesh_trees *sets,uint32_t count,uint32_t what,uint32_t root,
                struct mesh_operand operand,double alpha,double beta,struct mesh_choice *choice){
  int found=0;
  for(uint32_t s=0;s<count;s++)for(int whole=0;whole<=(what==MESH_ALLREDUCE);whole++){
    double last=INFINITY,time;
    for(uint32_t chunks=1;chunks<=MESH_CHUNKS;chunks*=2){
      const int status=mesh_candidate(map,sets+s,what,root,whole,chunks,operand,alpha,beta,&time);
      if(status==ENOMEM)return ENOMEM;
      if(status)break;
      if(!found || time<choice->time){*choice=(struct mesh_choice){s,(uint32_t)whole,chunks,0,time};found=1;}
      if(!(time<last))break;
      last=time;
    }
  }
  return found?0:EINVAL;
}

/* design/prepared-machine.md#M07 */
/* design/prepared-machine.md#M12 */
/* Each destination's completion words and the cancellation ranges over them: destination d's words follow d-1's
   (16-word aligned), one per received chunk an invocation of each receive transfer's range, in the frame order
   mesh_transfers_prepare assigned (a chunk's word for the transfer's invocation u is first + (u*count + slot)*chunks
   + chunk, its stride count*chunks words; a row's delivery names its slot's chunk word of invocation 0, slot 0's
   where every slot lands on the same rows, the bridge adding slot*chunks words, mesh-flow.c receive_input); the words
   in context->inputs_off, inputs_bytes. */
int mesh_host_inputs(struct mesh_ctx *context){
  struct hdr *m=context->M;
  struct mesh_client *client=mesh_client_at(m,context->client);
  uint64_t frames[MESH_DESTINATIONS],words=0;
  const uint32_t destinations=mesh_destination(client,0);
  for(uint32_t d=0;d<destinations;d++){
    words=(words+15)&~UINT64_C(15);
    frames[d]=0;
    for(uint32_t q=d*MESH_QUEUES;q<(d+1)*MESH_QUEUES;q++){
      struct mesh_transfer *in=mesh_transfers(m,context->client,q,MESH_RECEIVE);
      for(uint32_t i=0;i<atomic_load(mesh_order_length(m,context->client,q,MESH_RECEIVE));i++)frames[d]+=mesh_transfer_frames(m,in+i,client->tx[d].invocations);
    }
    words+=frames[d];
  }
  struct mesh_section input,stop;
  int status=mesh_section_create(context,sizeof(uint64_t)*(words?words:1),1,&input);
  if(!status)status=mesh_section_create(context,sizeof(struct mesh_cancellation)+(destinations?destinations:1)*sizeof(struct mesh_cancel_range),1,&stop);
  if(status)return status;
  uint64_t *word=mesh_section_address(context,input,0);
  struct mesh_cancellation *cancel=mesh_section_address(context,stop,0);
  memset(word,0,input.bytes);memset(cancel,0,stop.bytes);
  cancel->cyclic=atomic_load(&client->cyclic);
  context->inputs_off=(uint64_t)((char *)word-(char *)m);context->inputs_bytes=(uint64_t)input.pages*m->pgsz;
  uint64_t first=0;
  for(uint32_t d=0;d<destinations;d++){
    first=(first+15)&~UINT64_C(15);
    client->tx[d].cancel=(uint64_t)((char *)cancel-(char *)m);
    cancel->ranges[d]=(struct mesh_cancel_range){.offset=(uint64_t)((char *)(word+first)-(char *)m),.count=frames[d]};
    for(uint32_t q=d*MESH_QUEUES;q<(d+1)*MESH_QUEUES;q++){
      struct mesh_transfer *in=mesh_transfers(m,context->client,q,MESH_RECEIVE);
      for(uint32_t i=0;i<atomic_load(mesh_order_length(m,context->client,q,MESH_RECEIVE));i++){
        uint32_t chunks=mesh_row_chunks(m,in[i].local_row,in[i].bytes);
        for(uint32_t s=0;s<(in[i].stride?in[i].count:1);s++)for(uint32_t k=0;k<chunks;k++){
          struct mesh_delivery *delivery=mesh_delivery_at(m,in[i].local_row+s*in[i].stride+k);
          delivery->device_input=(uint64_t)((char *)(word+first+in[i].first+(uint64_t)s*chunks+k)-(char *)m);
          delivery->device_stride=sizeof(uint64_t)*(uint64_t)in[i].count*chunks;
        }
      }
    }
    first+=frames[d];
  }
  return 0;
}

/* What the GPU publication kernel stores (M04/M10): each prepared cell's continuation, into the
   cell of invocation t, released after the caller's writes to the section. */
/* design/prepared-machine.md#M30: running invocation T is the program's invocation T mod N in cycle T / N */
static uint32_t program_invocations(struct mesh_ctx *context){
  return mesh_client_at(context->M,context->client)->tx[0].invocations;
}

void mesh_host_publish(struct mesh_ctx *context,struct mesh_section section,uint32_t invocation){
  uint32_t count=mesh_publication_prepare(context,section.first,NULL),n=program_invocations(context);
  struct prepared_publication records[count?count:1];
  mesh_publication_prepare(context,section.first,records);
  const uint32_t t=n?invocation%n:invocation,cycle=n?invocation/n:0;
  for(uint32_t i=0;i<count;i++)
    atomic_store_explicit((_Atomic uint64_t *)(uintptr_t)(records[i].destination+(uint64_t)t*sizeof(struct mesh_send)),
      records[i].argument+cycle,memory_order_release);
}

/* The last chunk's word, as mesh_metal_receive_prepare reads it: chunks of one receive land in order. */
uint64_t mesh_host_arrived(struct mesh_ctx *context,struct mesh_section section,uint32_t invocation){
  struct hdr *m=context->M;
  struct mesh_delivery *delivery=mesh_delivery_at(m,section.first+mesh_row_chunks(m,section.first,section.bytes)-1);
  const uint32_t n=program_invocations(context),t=n?invocation%n:invocation;
  return atomic_load_explicit((_Atomic uint64_t *)((char *)m+delivery->device_input+(uint64_t)t*delivery->device_stride),
    memory_order_acquire);
}
