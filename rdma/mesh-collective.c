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

/* A tree's shape: each node's height (its longest path up from a leaf below it), depth, whether its subtree holds a
   contributor (`live`) and whether a child's does (`inner`) */
struct mesh_shape { uint32_t height,depth; int live,inner; };
static void mesh_tree_shape(const struct mesh_trees *trees,uint32_t t,const uint64_t *contributors,struct mesh_shape *shape){
  const uint32_t n=trees->nodes,r=trees->root[t],*parent=trees->parent+(size_t)t*n;
  /* the tree's levels and its contributors' ancestors (mesh-graph.h) */
  uint32_t depth[n?n:1],height[n?n:1];
  unsigned char live[n?n:1];
  mesh_graph_tree_levels(parent,n,r,depth,height);
  memset(live,0,sizeof live);
  for(uint32_t v=0;v<n;v++)if(mesh_member(contributors,v))mesh_graph_ancestors(parent,n,r,v,live);
  for(uint32_t v=0;v<n;v++)shape[v]=(struct mesh_shape){height[v],depth[v],live[v],0};
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

void mesh_compile_room(const struct mesh_trees *trees,uint32_t *parts,uint32_t *moves){
  const uint32_t n=trees->nodes,T=trees->count;
  *parts=T>1?T:1;
  *moves=4*T*n+2*n*n+2;
}

static int mesh_compile_sets(const struct mesh_trees *trees,uint32_t what,uint32_t root,int whole,
                 const uint64_t *contributors,const uint64_t *recipients,int global_segments,
                 struct mesh_program *program,uint32_t *segment,double *log_weight,uint32_t *first,struct mesh_move *move){
  const uint32_t n=trees->nodes,T=trees->count;
  if(!n || what>MESH_ALLGATHER || ((what==MESH_REDUCE || what==MESH_BROADCAST) && root>=n))return EINVAL;
  uint32_t room_parts,room;
  mesh_compile_room(trees,&room_parts,&room);
  const uint64_t *mine=contributors,*owners=what==MESH_ALLGATHER?contributors:recipients;
  /* The two reachability closures are different: a relay may read incoming
     partials without owning an input, and a recipient may have no input at all. */
  struct mesh_shape *shape=malloc((2*(size_t)T*n+1)*sizeof *shape);
  struct mesh_message *message=malloc((room+1)*sizeof *message);
  struct mesh_ordered *ordered=malloc((2*(size_t)room+1)*sizeof *ordered);
  /* part_of[t]: tree t's part (UINT32_MAX: the program moves none along t) */
  uint32_t *part_of=malloc(((size_t)T+1)*sizeof *part_of),*buffer=calloc((size_t)n*n+1,sizeof *buffer);
  uint32_t messages=0,parts=0,scratch=0;
  int status=EINVAL;
  if(!shape || !message || !ordered || !part_of || !buffer)goto done;
  int inputs=0,outputs=0;
  for(uint32_t u=0;u<n;u++){inputs|=mesh_member(mine,u);outputs|=mesh_member(recipients,u);}
  if(global_segments && ((!outputs && what!=MESH_REDUCE) || (!inputs && what==MESH_ALLGATHER))){
    memset(first,0,((size_t)n+1)*sizeof *first);
    *program=(struct mesh_program){what,root,n,0,0,
      what==MESH_ALLGATHER || what==MESH_REDUCE_SCATTER?n:0,segment,log_weight,first,move};
    status=0;goto done;
  }
  if(global_segments && ((!inputs && what!=MESH_BROADCAST) ||
      (what==MESH_BROADCAST && !mesh_member(mine,root))))goto done;
  for(uint32_t t=0;t<T;t++){
    mesh_tree_shape(trees,t,mine,shape+(size_t)t*n);
    mesh_tree_shape(trees,t,recipients,shape+((size_t)T+t)*n);
  }
  for(size_t k=0;k<T;k++)part_of[k]=UINT32_MAX;
  if(what==MESH_ALLREDUCE && whole){
    /* the whole operand toward every recipient v up v's heaviest tree: u's partial for v from scratch of its own
       where it has children there, else its own operand, from buffer[u * n + u]: the operand itself (buffer 0) where u
       sends it to one node only, forwards nothing and is not a recipient. A
       recipient's original source is a distinct immutable binding (buffer 1),
       while buffer 0 may already receive its reduced result; one part. */
    parts=1;
    segment[0]=0;log_weight[0]=0;
    for(uint32_t u=0;u<n;u++){
      uint32_t next=2,own=0;
      for(uint32_t v=0;v<n;v++){
        if(!mesh_member(recipients,v))continue;
        const uint32_t t=mesh_trees_heaviest(trees,v);
        if(t==T)goto done;
        if(u==v || !shape[(size_t)t*n+u].live)continue;
        if(shape[(size_t)t*n+u].inner)buffer[(size_t)u*n+v]=next++;
        else own++;
      }
      /* A recipient's result is mutable independently of its
         original source DMA reads. Keep that source binding separate even for
         one outgoing leaf send; SEND submission is not read retirement. */
      buffer[(size_t)u*n+u]=own==1 && next==2 && !mesh_member(recipients,u)?0:1;
      const uint32_t used=next>2?next-1:own && buffer[(size_t)u*n+u]?1:0;
      if(used>scratch)scratch=used;
    }
    for(uint32_t v=0;v<n;v++){
      if(!mesh_member(recipients,v))continue;
      const uint32_t t=mesh_trees_heaviest(trees,v),*parent=trees->parent+(size_t)t*n;
      const struct mesh_shape *s=shape+(size_t)t*n;
      for(uint32_t u=0;u<n;u++){
        if(u==v || !s[u].live)continue;
        const uint32_t w=parent[u],op=!mesh_member(mine,w) && mesh_first_child(trees,t,s,w,u)?MESH_STEP_COPY:MESH_STEP_REDUCE;
        message[messages++]=(struct mesh_message){u,w,s[u].height,0,1,s[u].inner?buffer[(size_t)u*n+v]:buffer[(size_t)u*n+u],w==v?0:buffer[(size_t)w*n+v],op,v};
      }
    }
  }else{
    /* Gather segments belong to contributors; scatter and all-reduce segments
       belong to recipients. V forms retain global rank indices, including zero
       gaps. Legacy mesh_compile retains its compact contributor numbering. */
    const int rooted=what==MESH_REDUCE || what==MESH_BROADCAST;
    for(uint32_t g=0,index=0;g<(rooted?1:n);g++){
      if(!rooted && !mesh_member(owners,g))continue;
      const uint32_t before=parts;
      for(uint32_t t=0;t<T;t++)
        if(trees->root[t]==(rooted?root:g)){
          part_of[t]=parts;
          segment[parts]=rooted?0:global_segments && (what==MESH_ALLGATHER || what==MESH_REDUCE_SCATTER)?g:index;
          log_weight[parts]=trees->log_weight[t];parts++;
        }
      if(global_segments && parts==before)goto done;
      index++;
    }
    if(!parts)goto done;
    const int up=what==MESH_ALLREDUCE || what==MESH_REDUCE || what==MESH_REDUCE_SCATTER;
    const int down=what==MESH_ALLREDUCE || what==MESH_BROADCAST || what==MESH_ALLGATHER;
    uint32_t base=0;
    for(uint32_t t=0;up && down && t<T;t++)
      if(part_of[t]!=UINT32_MAX && shape[(size_t)t*n+trees->root[t]].height>base)base=shape[(size_t)t*n+trees->root[t]].height;
    for(uint32_t t=0;t<T;t++){
      if(part_of[t]==UINT32_MAX)continue;
      const uint32_t r=trees->root[t],*parent=trees->parent+(size_t)t*n,p=part_of[t];
      const struct mesh_shape *s=shape+(size_t)t*n,*d=shape+((size_t)T+t)*n;
      for(uint32_t u=0;up && u<n;u++){
        if(u==r || !s[u].live)continue;
        const uint32_t w=parent[u],op=!mesh_member(mine,w) && mesh_first_child(trees,t,s,w,u)?MESH_STEP_COPY:MESH_STEP_REDUCE;
        message[messages++]=(struct mesh_message){u,w,s[u].height,p,1,0,0,op,p};
      }
      for(uint32_t u=0;down && u<n;u++)
        if(u!=r && d[u].live)message[messages++]=(struct mesh_message){parent[u],u,base+d[u].depth-1,p,1,0,0,MESH_STEP_COPY,p};
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
  *program=(struct mesh_program){what,root,n,parts,scratch,
    global_segments && (what==MESH_ALLGATHER || what==MESH_REDUCE_SCATTER)?n:0,segment,log_weight,first,move};
  status=0;
done:
  free(shape);free(message);free(ordered);free(part_of);free(buffer);
  return status;
}

int mesh_compile_members(const struct mesh_trees *trees,uint32_t what,uint32_t root,int whole,
                 const uint64_t *contributors,const uint64_t *recipients,
                 struct mesh_program *program,uint32_t *segment,double *log_weight,uint32_t *first,struct mesh_move *move){
  return mesh_compile_sets(trees,what,root,whole,contributors,recipients,1,program,segment,log_weight,first,move);
}

int mesh_compile(const struct mesh_trees *trees,uint32_t what,uint32_t root,int whole,const uint64_t *contributors,
                 struct mesh_program *program,uint32_t *segment,double *log_weight,uint32_t *first,struct mesh_move *move){
  return mesh_compile_sets(trees,what,root,whole,contributors,contributors,0,program,segment,log_weight,first,move);
}

/* Explicit global rank extent for member-aware V forms, otherwise the legacy
   extent inferred from the highest segment a part names. */
static uint32_t mesh_program_segments(const struct mesh_program *program){
  if(program->segments)return program->segments;
  uint32_t segments=1;
  for(uint32_t p=0;p<program->parts;p++)if(program->segment[p]+1>segments)segments=program->segment[p]+1;
  return segments;
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
    /* A skipped global rank is a zero-length segment, never permission to
       silently discard a caller's nonzero count. */
    if(program->segments && length && !(total>0)){free(bases);return 0;}
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

uint32_t mesh_program_steps_cut(const struct mesh_program *program,uint32_t rank,struct mesh_operand operand,const uint64_t *segments,
                                uint64_t page,struct mesh_step *steps){
  if(rank>=program->nodes || !program->parts)return 0;
  uint64_t *lo=malloc(program->parts*sizeof *lo),*count=malloc(program->parts*sizeof *count);
  uint32_t k=0;
  if(lo && count && mesh_program_ranges(program,operand.elements,segments,mesh_page_quantum(page,operand.element_bytes),lo,count)){
    /* a move's parts in one segment are one step: no step spans two segments, so a rank's own segment of a
       reduce-scatter or an all-gather is a buffer of its own (libnccl-mesh partial_split) */
    for(uint32_t i=program->first[rank];i<program->first[rank+1];i++){
      const struct mesh_move *m=program->move+i;
      for(uint32_t p=m->part,end;p<m->part+m->parts;p=end){
        for(end=p+1;end<m->part+m->parts && program->segment[end]==program->segment[p];end++);
        struct mesh_operand piece=operand;
        piece.elements=lo[end-1]+count[end-1]-lo[p];
        if(piece.elements)steps[k++]=(struct mesh_step){m->op,m->peer,m->round,m->buffer,lo[p],piece};
      }
    }
  }
  free(lo);free(count);
  return k;
}

/* Configuration-only definite initialization. Keep unions of intervals, not a
   bitmap proportional to the tensor. A COPY defines its exact destination;
   publication alone never defines another local buffer or retires a source. */
struct mesh_defined_range { uint64_t first,last; };
struct mesh_defined { struct mesh_defined_range *range; size_t count,room; };

static int mesh_defined_has(const struct mesh_defined *set,uint64_t first,uint64_t last){
  if(first==last)return 1;
  for(size_t i=0;i<set->count && set->range[i].first<=first;i++)
    if(set->range[i].last>=last)return 1;
  return 0;
}

static int mesh_defined_add(struct mesh_defined *set,uint64_t first,uint64_t last){
  if(first==last)return 0;
  size_t begin=0,end;
  while(begin<set->count && set->range[begin].last<first)begin++;
  end=begin;
  while(end<set->count && set->range[end].first<=last){
    if(set->range[end].first<first)first=set->range[end].first;
    if(set->range[end].last>last)last=set->range[end].last;
    end++;
  }
  if(begin==end && set->count==set->room){
    const size_t room=set->room?2*set->room:4;
    if(room<set->room || room>SIZE_MAX/sizeof *set->range)return EOVERFLOW;
    struct mesh_defined_range *grown=realloc(set->range,room*sizeof *grown);
    if(!grown)return ENOMEM;
    set->range=grown;set->room=room;
  }
  memmove(set->range+begin+1,set->range+end,(set->count-end)*sizeof *set->range);
  set->count-=end-begin;
  set->range[begin]=(struct mesh_defined_range){first,last};set->count++;
  return 0;
}

int mesh_program_check_members(const struct mesh_program *program,struct mesh_operand operand,const uint64_t *cut,
                                uint64_t page,const uint64_t *contributors,const uint64_t *recipients){
  if(!program || !program->nodes || program->what>MESH_ALLGATHER || !program->first ||
     program->scratch==UINT32_MAX || !operand.element_bytes ||
     (program->parts && (!program->segment || !program->log_weight)) ||
     (program->first[program->nodes] && !program->move))return EINVAL;
  if(operand.elements>UINT64_MAX/operand.element_bytes)return EOVERFLOW;
  if((program->what==MESH_REDUCE || program->what==MESH_BROADCAST) && program->root>=program->nodes)return EINVAL;
  if(program->what==MESH_BROADCAST && !mesh_member(contributors,program->root))return EINVAL;
  if(program->first[0])return EINVAL;
  for(uint32_t r=0;r<program->nodes;r++)if(program->first[r]>program->first[r+1])return EINVAL;
  for(uint32_t p=0;p<program->parts;p++)
    if(program->segment[p]==UINT32_MAX || (program->segments && program->segment[p]>=program->segments) ||
       isnan(program->log_weight[p]) || program->log_weight[p]==INFINITY)return EINVAL;
  const uint32_t segments=mesh_program_segments(program),buffers=program->scratch+1;
  const int gathered=program->what==MESH_ALLGATHER,scattered=program->what==MESH_REDUCE_SCATTER;
  if((gathered || scattered) && segments!=program->nodes)return EINVAL;
  for(uint32_t g=0;g<segments;g++){
    int found=0,positive=0;
    for(uint32_t p=0;p<program->parts;p++)if(program->segment[p]==g){
      found=1;positive|=isfinite(program->log_weight[p]);
    }
    if(found && !positive)return EINVAL;
  }
  const uint64_t quantum=mesh_page_quantum(page,operand.element_bytes);
  if(operand.elements>UINT64_MAX-(quantum-1))return EOVERFLOW;
  if(cut){
    uint64_t sum=0;
    for(uint32_t g=0;g<segments;g++){
      if(cut[g]>operand.elements-sum)return EINVAL;
      sum+=cut[g];
    }
    if(sum!=operand.elements)return EINVAL;
  }
  if((size_t)segments>SIZE_MAX/(2*sizeof(uint64_t)) ||
     (size_t)program->parts>SIZE_MAX/(2*sizeof(uint64_t)) ||
     (size_t)buffers>SIZE_MAX/sizeof(struct mesh_defined))return EOVERFLOW;
  uint64_t *bounds=calloc(2*(size_t)segments,sizeof *bounds);
  uint64_t *ranges=calloc(2*(size_t)program->parts+1,sizeof *ranges);
  int result=bounds && ranges?0:ENOMEM;
  if(!result && !mesh_cut(segments,operand.elements,cut,quantum,bounds,bounds+segments))result=EINVAL;
  if(!result && program->parts &&
     !mesh_program_ranges(program,operand.elements,cut,quantum,ranges,ranges+program->parts))result=EINVAL;
  for(uint32_t r=0;!result && r<program->nodes;r++){
    struct mesh_defined *defined=calloc(buffers,sizeof *defined);
    if(!defined){result=ENOMEM;break;}
    const int contributes=mesh_member(contributors,r),receives=mesh_member(recipients,r);
    uint64_t from=gathered?bounds[r]:0,length=gathered?bounds[segments+r]:operand.elements;
    if((gathered && !contributes && length) || (scattered && !receives && bounds[segments+r]))result=EINVAL;
    const int initial=contributes && (program->what!=MESH_BROADCAST || r==program->root);
    for(uint32_t b=0;initial && !result && b<buffers;b++)result=mesh_defined_add(defined+b,from,from+length);
    for(uint32_t i=program->first[r];!result && i<program->first[r+1];i++){
      const struct mesh_move *m=program->move+i;
      if(m->op>MESH_STEP_COPY || m->peer>=program->nodes || m->peer==r || m->buffer>=buffers ||
         !m->parts || m->part>=program->parts || m->parts>program->parts-m->part){result=EINVAL;break;}
      const uint32_t last=m->part+m->parts-1;
      const uint64_t first=ranges[m->part],end=ranges[last]+ranges[program->parts+last];
      if(end<first || end>operand.elements){result=EINVAL;break;}
      if(m->op==MESH_STEP_COPY)result=mesh_defined_add(defined+m->buffer,first,end);
      else if(!mesh_defined_has(defined+m->buffer,first,end))result=EINVAL;
    }
    const int required=program->what==MESH_REDUCE?r==program->root:receives;
    from=scattered?bounds[r]:0;length=scattered?bounds[segments+r]:operand.elements;
    if(!result && required && !mesh_defined_has(defined,from,from+length))result=EINVAL;
    for(uint32_t b=0;b<buffers;b++)free(defined[b].range);
    free(defined);
  }
  free(bounds);free(ranges);
  return result;
}

/* A packed tree while packing: its root, parents and the flow it carries */
struct mesh_packed { uint32_t root; double flow; uint32_t *parent; };
static int mesh_packed_heavier(const void *a,const void *b){
  const struct mesh_packed *x=a,*y=b;
  if(x->root!=y->root)return x->root<y->root?-1:1;
  return x->flow>y->flow?-1:x->flow<y->flow;
}

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
  const uint64_t rounds=(uint64_t)(mesh_log(links>1?(double)links:2.0)/(e*e))+1,room=rounds*count;
  return room>=UINT32_MAX?UINT32_MAX-1:(uint32_t)room;
}

uint32_t mesh_solve(const struct mesh_link_map *map,double eps,const struct mesh_demand *demands,uint32_t count,uint32_t room,
                    uint32_t *demand,uint32_t *parent,double *share){
  const uint32_t n=map->nodes,E=mesh_solve_links(map,NULL,NULL,NULL);
  if(!n || !count)return 0;
  uint32_t *from=calloc((size_t)E+1,sizeof *from),*to=calloc((size_t)E+1,sizeof *to);
  double *cap=calloc((size_t)E+1,sizeof *cap),*load=calloc(2*(size_t)E+1,sizeof *load),*add=calloc(2*(size_t)E+1,sizeof *add),
         *len=calloc(2*(size_t)E+1,sizeof *len),*dist=calloc((size_t)n+1,sizeof *dist);
  uint32_t *up=calloc((size_t)n+1,sizeof *up),*upl=calloc((size_t)n+1,sizeof *upl),*tree=calloc((size_t)n+1,sizeof *tree),
           *treel=calloc((size_t)n+1,sizeof *treel),*hits=calloc((size_t)room+1,sizeof *hits),made=0;
  unsigned char *done=calloc((size_t)n+1,1);
  int failed=!from || !to || !cap || !load || !add || !len || !dist || !up || !upl || !tree || !treel || !hits || !done || !(eps>0);
  if(!failed)mesh_solve_links(map,from,to,cap);
  double least=0,total=0;
  for(uint32_t e=0;!failed && e<E;e++)if(!least || cap[e]<least)least=cap[e];
  for(uint32_t d=0;!failed && d<count;d++)total+=(double)demands[d].up+(double)demands[d].down;
  const uint32_t rounds=failed?0:(uint32_t)(mesh_log(2*E>1?(double)(2*E):2.0)/(eps*eps))+1;
  const double width=total>0 && least>0?total/rounds/least:1;
  for(uint32_t r=0;!failed && r<rounds;r++){
    double most=0;
    for(size_t e=0;e<2*(size_t)E;e++)if(load[e]/cap[e/2]>most)most=load[e]/cap[e/2];
    for(size_t e=0;e<2*(size_t)E;e++){len[e]=mesh_exp(eps*(load[e]/cap[e/2]-most)/width)/cap[e/2];add[e]=0;}
    for(uint32_t d=0;!failed && d<count;d++){
      const struct mesh_demand *x=demands+d;
      const double u=(double)x->up,w=(double)x->down;
      for(uint32_t v=0;v<n;v++){dist[v]=INFINITY;up[v]=UINT32_MAX;upl[v]=UINT32_MAX;done[v]=0;tree[v]=UINT32_MAX;treel[v]=UINT32_MAX;}
      if(x->root>=n){failed=1;break;}
      dist[x->root]=0;up[x->root]=x->root;
      /* the least-length tree toward the root: a direction's length weighed by the bytes that cross it, up toward the
         root (from the leaf's side) and down from it */
      for(uint32_t step=0;step<n;step++){
        uint32_t at=n;
        for(uint32_t v=0;v<n;v++)if(!done[v] && dist[v]<INFINITY && (at==n || dist[v]<dist[at]))at=v;
        if(at==n)break;
        done[at]=1;
        for(uint32_t e=0;e<E;e++){
          if(from[e]!=at && to[e]!=at)continue;
          const uint32_t v=from[e]==at?to[e]:from[e],toward=from[e]==v?0:1;
          if(done[v])continue;
          const double through=dist[at]+u*len[2*(size_t)e+toward]+w*len[2*(size_t)e+(toward^1)];
          if(through<dist[v]){dist[v]=through;up[v]=at;upl[v]=e;}
        }
      }
      tree[x->root]=x->root;
      for(uint32_t v=0;v<n && !failed;v++){
        if(v==x->root || !mesh_member(x->leaves,v))continue;
        if(up[v]==UINT32_MAX){failed=1;break;}
        for(uint32_t at=v;tree[at]==UINT32_MAX;at=up[at]){tree[at]=up[at];treel[at]=upl[at];}
      }
      if(failed)break;
      uint32_t k=0;
      while(k<made && !(demand[k]==d && !memcmp(parent+(size_t)k*n,tree,n*sizeof *tree)))k++;
      if(k==made){
        if(made>=room){failed=1;break;}
        demand[made]=d;
        memcpy(parent+(size_t)made*n,tree,n*sizeof *tree);
        made++;
      }
      hits[k]++;
      for(uint32_t v=0;v<n;v++){
        if(v==x->root || tree[v]==UINT32_MAX)continue;
        const uint32_t e=treel[v],toward=from[e]==v?0:1;
        add[2*(size_t)e+toward]+=u/rounds;
        add[2*(size_t)e+(toward^1)]+=w/rounds;
      }
    }
    for(size_t e=0;e<2*(size_t)E;e++)load[e]+=add[e];
  }
  for(uint32_t k=0;!failed && k<made;k++)share[k]=(double)hits[k]/rounds;
  free(from);free(to);free(cap);free(load);free(add);free(len);free(dist);free(up);free(upl);free(tree);free(treel);free(hits);free(done);
  return failed?UINT32_MAX:made;
}

uint32_t mesh_flow_trees(uint32_t nodes,const uint32_t *demand,const struct mesh_demand *demands,const uint32_t *parent,const double *share,
                         uint32_t count,uint32_t first,uint32_t last,uint32_t *root,uint32_t *tree,double *log_weight){
  uint32_t trees=0;
  for(uint32_t k=0;k<count;k++){
    if(demand[k]<first || demand[k]>=last)continue;
    root[trees]=demands[demand[k]].root;
    memcpy(tree+(size_t)trees*nodes,parent+(size_t)k*nodes,nodes*sizeof *tree);
    log_weight[trees++]=mesh_log(share[k]);
  }
  return trees;
}

uint32_t mesh_trees_pack(const struct mesh_link_map *map,const uint64_t *roots,double eps,uint32_t most,uint32_t *root,uint32_t *parent,
                         double *log_weight){
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
  uint32_t *crossing=malloc(((size_t)m*N+1)*sizeof *crossing),*hop=malloc((size_t)N*N*sizeof *hop),*uses=calloc((size_t)m*N+1,sizeof *uses);
  double *capacity=malloc(((size_t)m*N+1)*sizeof *capacity),*length=malloc(((size_t)m*N+1)*sizeof *length),*weight=malloc((m+1)*sizeof *weight);
  uint32_t *parallel=malloc(((size_t)m*N+1)*sizeof *parallel);
  uint64_t *aggregate=malloc(((size_t)m*N+1)*sizeof *aggregate);
  int32_t *chosen=malloc(n*sizeof *chosen);
  struct mesh_packed *packed=NULL;
  size_t trees=0,room=0;
  uint32_t written=0,path[N+1];
  if(!from || !to || !through || !crossing || !hop || !uses || !capacity || !length || !weight || !chosen || !parallel || !aggregate)goto done;
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
  const double delta=mesh_exp(-mesh_log(P/(1-eps))/eps);
  double D=0;
  for(uint32_t k=0;k<P;k++){length[k]=delta/capacity[k];D+=length[k]*capacity[k];}
  while(D<1)for(uint32_t r=0;r<n && D<1;r++)for(double demand=mesh_member(roots,r);demand>0 && D<1;){
    for(uint32_t e=0;e<m;e++){weight[e]=0;for(uint32_t i=through[e];i<through[e+1];i++)weight[e]+=length[crossing[i]];}
    if(mesh_graph_arborescence(n,r,m,from,to,weight,chosen))goto done;
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
  free(parallel);free(aggregate);
  return written;
}

/* -- the lists: the entries a program's steps take on a holding's send list and landing list --------------------------- */


/* whether steps are in round order (mesh_program_steps_cut's order: a compiled program's are) */
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

/* Counted exchanges describe logical transactions. Physical routing and range
   dependencies are supplied by the same compiler as ordinary collectives. */
uint32_t mesh_counted_steps(uint32_t nodes,uint32_t rank,const uint64_t *segments,struct mesh_operand element,struct mesh_step *steps){
  uint32_t k=0;
  for(uint32_t q=0;q<2*nodes;q++)
    if(q%nodes!=rank && segments[q])
      steps[k++]=(struct mesh_step){q<nodes?MESH_STEP_SEND:MESH_STEP_COPY,q%nodes,0,q>=nodes,mesh_segment_first(segments+(q<nodes?0:nodes),q%nodes),
                                    {element.type,element.element_bytes,segments[q]}};
  return k;
}
uint64_t mesh_group_lay(const struct mesh_group *g,uint32_t rank,uint32_t peer,int send,uint64_t at,struct mesh_place *scratch,
                        struct mesh_group_place *made,uint32_t *placed){
  *placed=0;
  for(uint32_t i=0;i<g->calls;i++){
    const struct mesh_group_call *c=g->call+i;
    uint32_t laid=0;
    at=mesh_lay(c->steps[rank],c->count[rank],peer,send,at,scratch,&laid);
    for(uint32_t k=0;k<laid;k++)made[(*placed)++]=(struct mesh_group_place){i,0,scratch[k]};
  }
  return at;
}
