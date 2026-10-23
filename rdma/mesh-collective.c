#include "mesh-call.h"
#include "mesh-plan.h"
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#pragma STDC FP_CONTRACT OFF

static uint32_t mesh_all(const struct mesh_link_map *map){ return map->nodes+(map->next?map->relays:0); }

static int mesh_joins(const struct mesh_link_map *map,uint32_t u,uint32_t x){
  const uint32_t N=mesh_all(map);
  if(u==x || u>=N || x>=N)return 0;
  if(map->kind==MESH_LINKS_MESH && u<map->nodes && x<map->nodes)return 1;
  for(uint32_t l=0;l<map->links;l++)
    if((map->link[l][0]==u && map->link[l][1]==x) || (map->link[l][0]==x && map->link[l][1]==u))return 1;
  return 0;
}

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

static uint64_t mesh_hop_inverse(const struct mesh_link_map *map,uint32_t u,uint32_t x){
  uint64_t slowest;
  const uint32_t n=mesh_parallel(map,u,x,&slowest);
  if(!slowest)return 0;
  double inverse=8e9/(double)slowest;
  if(n>1)inverse/=n;
  return (uint64_t)llround(inverse*1e6);
}

static uint32_t mesh_mark(const struct mesh_link_map *map,uint32_t v){ return map->mark?map->mark[v]:v; }

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

static int mesh_member(const uint64_t *set,uint32_t v){ return !set || (set[v/64]>>(v%64)&1); }

int mesh_link_between(const struct mesh_link_map *map,uint32_t a,uint32_t b){
  if(a==b || a>=map->nodes || b>=map->nodes)return 0;
  if(!map->next)return mesh_joins(map,a,b);
  uint32_t path[mesh_all(map)+1];
  const uint32_t count=mesh_route_nodes(map,a,b,path);
  for(uint32_t i=1;i+1<count;i++)if(path[i]<map->nodes)return 0;
  return count>=2;
}

static uint64_t mesh_hash(uint32_t x,uint32_t y,uint32_t z){
  uint64_t h=(((uint64_t)x<<32)|y)^((uint64_t)z*UINT64_C(0x9e3779b97f4a7c15));
  h=(h^(h>>30))*UINT64_C(0xbf58476d1ce4e5b9);
  h=(h^(h>>27))*UINT64_C(0x94d049bb133111eb);
  return (h^(h>>31))>>16;
}

struct mesh_length { uint64_t hops,inverse; };
static int mesh_shorter(struct mesh_length a,struct mesh_length b){ return a.hops!=b.hops?a.hops<b.hops:a.inverse<b.inverse; }
static int mesh_equal(struct mesh_length a,struct mesh_length b){ return a.hops==b.hops && a.inverse==b.inverse; }

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

int mesh_routes(const struct mesh_link_map *map,uint32_t *next){
  const uint32_t N=map->nodes+map->relays;
  if(!N || map->kind>MESH_LINKS_GRAPH || (map->links && !map->link) || (map->relays && map->kind==MESH_LINKS_MESH))return EINVAL;
  for(uint32_t l=0;l<map->links;l++)if(map->link[l][0]>=N || map->link[l][1]>=N || map->link[l][0]==map->link[l][1])return EINVAL;
  const size_t pairs=(size_t)N*N;
  uint64_t *inverse=calloc(pairs,sizeof *inverse);
  unsigned char *joined=calloc(pairs,1),*done=malloc(N);
  struct mesh_length *length=malloc(N*sizeof *length);
  const int status=inverse && joined && done && length?0:ENOMEM;

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

static uint32_t mesh_trees_heaviest(const struct mesh_trees *trees,uint32_t r){
  uint32_t best=trees->count;
  for(uint32_t t=0;t<trees->count;t++)
    if(trees->root[t]==r && (best==trees->count || trees->log_weight[t]>trees->log_weight[best]))best=t;
  return best;
}

struct mesh_hops { uint32_t n,N,*count,*hop; struct mesh_length *length; };

static void mesh_carry(const struct mesh_hops *h,uint64_t *load,uint32_t v,uint32_t u,int sign){
  const size_t pair=(size_t)v*h->n+u;
  for(uint32_t i=0;i<h->count[pair];i++){
    if(sign>0)load[h->hop[pair*h->N+i]]++;
    else load[h->hop[pair*h->N+i]]--;
  }
}

struct mesh_far { struct mesh_length length; uint32_t mark,node; };
static int mesh_farther(const void *x,const void *y){
  const struct mesh_far *a=x,*b=y;
  if(!mesh_equal(a->length,b->length))return mesh_shorter(a->length,b->length)?1:-1;
  return (a->mark>b->mark)-(a->mark<b->mark);
}

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

static int mesh_first_child(const struct mesh_trees *trees,uint32_t t,const struct mesh_shape *shape,uint32_t w,uint32_t u){
  const uint32_t n=trees->nodes,r=trees->root[t],*parent=trees->parent+(size_t)t*n;
  for(uint32_t c=0;c<n;c++)
    if(c!=r && c!=u && parent[c]==w && shape[c].live &&
       (shape[c].height<shape[u].height || (shape[c].height==shape[u].height && c<u)))return 0;
  return 1;
}

struct mesh_message { uint32_t from,to,round,part,parts,out,in,op,key; };
static int mesh_message_order(const void *a,const void *b){
  const struct mesh_message *x=a,*y=b;
  const uint32_t l[5]={x->from,x->to,x->round,x->key,x->part},r[5]={y->from,y->to,y->round,y->key,y->part};
  for(int i=0;i<5;i++)if(l[i]!=r[i])return l[i]<r[i]?-1:1;
  return 0;
}

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

  uint32_t *part_of=malloc(((size_t)T*c+1)*sizeof *part_of),*buffer=calloc((size_t)n*n+1,sizeof *buffer);
  uint32_t messages=0,parts=0,scratch=0;
  int status=EINVAL;
  if(!shape || !message || !ordered || !part_of || !buffer)goto done;
  for(uint32_t t=0;t<T;t++)mesh_tree_shape(trees,t,mine,shape+(size_t)t*n);
  for(size_t k=0;k<(size_t)T*c;k++)part_of[k]=UINT32_MAX;
  if(what==MESH_ALLREDUCE && whole){

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

  qsort(message,messages,sizeof *message,mesh_message_order);
  uint32_t kept=0;
  for(uint32_t i=0;i<messages;i++){
    struct mesh_message *last=kept?message+kept-1:NULL;
    const struct mesh_message *m=message+i;
    if(last && last->from==m->from && last->to==m->to && last->round==m->round && last->part+last->parts==m->part &&
       last->out==m->out && last->in==m->in && last->op==m->op)last->parts+=m->parts;
    else message[kept++]=*m;
  }

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

static uint32_t mesh_program_segments(const struct mesh_program *program){
  uint32_t segments=1;
  for(uint32_t p=0;p<program->parts;p++)if(program->segment[p]+1>segments)segments=program->segment[p]+1;
  return segments;
}

static uint64_t mesh_quantum(uint64_t page,uint32_t element){
  uint64_t a=page,b=element?element:1;
  if(page<=1)return 1;
  while(b){const uint64_t t=a%b;a=b;b=t;}
  return page/a;
}

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

void mesh_routing_room(uint32_t nodes,uint32_t most,uint32_t *routes,uint32_t *hops){
  const uint64_t r=(uint64_t)nodes*(nodes?nodes-1:0)*(most?most:1);
  *routes=r>UINT32_MAX?UINT32_MAX:(uint32_t)r;
  *hops=r*nodes>UINT32_MAX?UINT32_MAX:(uint32_t)(r*nodes);
}

int mesh_routing_check(const struct mesh_link_map *map,const struct mesh_routing *g){
  const uint32_t n=g->nodes;
  if(!n || n!=map->nodes || !g->pair || (g->routes && (!g->start || !g->node || !g->share)) || g->pair[0] || g->pair[(size_t)n*n]!=g->routes ||
     (g->routes && g->start[0]))
    return EINVAL;
  unsigned char *seen=malloc(n);
  int status=seen?0:ENOMEM;
  for(uint32_t a=0;!status && a<n;a++)for(uint32_t b=0;!status && b<n;b++){
    const size_t p=(size_t)a*n+b;
    if(g->pair[p+1]<g->pair[p] || (a==b && g->pair[p+1]!=g->pair[p]))status=EINVAL;
    for(uint32_t k=g->pair[p];!status && k<g->pair[p+1];k++){
      const uint32_t lo=g->start[k],hi=g->start[k+1];
      if(hi<lo+2 || hi-lo>n || g->node[lo]!=a || g->node[hi-1]!=b || !(g->share[k]>0) || !isfinite(g->share[k])){status=EINVAL;break;}
      memset(seen,0,n);
      for(uint32_t i=lo;!status && i<hi;i++){
        if(g->node[i]>=n || seen[g->node[i]] || (i>lo && !mesh_link_between(map,g->node[i-1],g->node[i])))status=EINVAL;
        else seen[g->node[i]]=1;
      }
    }
  }
  free(seen);
  return status;
}

static int mesh_route_before(const uint32_t *x,uint32_t xn,const uint32_t *y,uint32_t yn){
  if(xn!=yn)return xn<yn;
  for(uint32_t i=0;i<xn;i++)if(x[i]!=y[i])return x[i]<y[i];
  return 0;
}
uint32_t mesh_routing_trees(const struct mesh_trees *trees,struct mesh_routing *routing,uint32_t *pair,uint32_t *start,uint32_t *node,double *share){
  const uint32_t n=trees->nodes,T=trees->count;
  uint32_t routes=0,hops=0,*nodes=malloc(((size_t)T*n+1)*sizeof *nodes),*length=malloc(((size_t)T+1)*sizeof *length),*order=malloc(((size_t)T+1)*sizeof *order);
  double *weight=malloc(((size_t)T+1)*sizeof *weight);
  if(!nodes || !length || !order || !weight){free(nodes);free(length);free(order);free(weight);return UINT32_MAX;}
  for(uint32_t a=0;a<n;a++)for(uint32_t b=0;b<n;b++){
    pair[(size_t)a*n+b]=routes;
    if(a==b)continue;
        double most=-INFINITY;
    for(uint32_t t=0;t<T;t++)if(trees->root[t]==b && trees->log_weight[t]>most)most=trees->log_weight[t];
    uint32_t found=0;
    for(uint32_t t=0;t<T;t++){
      if(trees->root[t]!=b)continue;
      uint32_t *path=nodes+(size_t)found*n,count=0;
      for(uint32_t at=a;count<n;at=trees->parent[(size_t)t*n+at]){
        path[count++]=at;
        if(at==b)break;
      }
      const double w=mesh_exp(trees->log_weight[t]-most);
      if(!count || path[count-1]!=b || !(w>0))continue;
      uint32_t same=0;
      while(same<found && !(length[same]==count && !memcmp(nodes+(size_t)same*n,path,count*sizeof *path)))same++;
      if(same<found){weight[same]+=w;continue;}
      length[found]=count;weight[found]=w;found++;
    }
        for(uint32_t i=0;i<found;i++){
      uint32_t j=i;
      while(j && mesh_route_before(nodes+(size_t)i*n,length[i],nodes+(size_t)order[j-1]*n,length[order[j-1]])){order[j]=order[j-1];j--;}
      order[j]=i;
    }
    for(uint32_t i=0;i<found;i++){
      const uint32_t x=order[i];
      start[routes]=hops;share[routes]=weight[x];routes++;
      memcpy(node+hops,nodes+(size_t)x*n,length[x]*sizeof *node);
      hops+=length[x];
    }
  }
  pair[(size_t)n*n]=routes;start[routes]=hops;
  *routing=(struct mesh_routing){n,routes,pair,start,node,share};
  free(nodes);free(length);free(order);free(weight);
  return routes;
}

uint32_t mesh_routing_pack(const struct mesh_link_map *map,double eps,uint32_t most,struct mesh_routing *routing,uint32_t *pair,uint32_t *start,
                           uint32_t *node,double *share){
  const uint32_t n=map->nodes,m=most?most:1,N=mesh_all(map);
  uint32_t *root=malloc(((size_t)n*m+1)*sizeof *root),*parent=malloc(((size_t)n*n*m+1)*sizeof *parent);
  double *weight=malloc(((size_t)n*m+1)*sizeof *weight);
  if(!root || !parent || !weight){free(root);free(parent);free(weight);return UINT32_MAX;}
  const int apart=n>1?mesh_link_map_joined(map,NULL):ENOTCONN;
  const uint32_t trees=!apart?mesh_trees_pack(map,eps,m,root,parent,weight):0;
  uint32_t routes=0;
  if((apart && apart!=ENOTCONN) || (!apart && !trees))routes=UINT32_MAX;
  else if(trees){
    const struct mesh_trees packed={n,trees,root,parent,weight};
    routes=mesh_routing_trees(&packed,routing,pair,start,node,share);
  }else{
    uint32_t path[N+1],hops=0;
    for(uint32_t a=0;a<n;a++)for(uint32_t b=0;b<n;b++){
      pair[(size_t)a*n+b]=routes;
      const uint32_t count=a==b?0:mesh_route_path(map,a,b,path);
      uint32_t kept=0;
      for(uint32_t i=0;i<count;i++)if(path[i]<n)path[kept++]=path[i];
      int linked=kept>=2;
      for(uint32_t i=1;linked && i<kept;i++)linked=mesh_link_between(map,path[i-1],path[i]);
      if(!linked)continue;
      start[routes]=hops;share[routes]=1;routes++;
      memcpy(node+hops,path,kept*sizeof *node);
      hops+=kept;
    }
    pair[(size_t)n*n]=routes;start[routes]=hops;
    *routing=(struct mesh_routing){n,routes,pair,start,node,share};
  }
  free(root);free(parent);free(weight);
  return routes;
}

static int mesh_block_moves(const struct mesh_routing *g,const struct mesh_block *x){
  return x->from!=x->to && x->from<g->nodes && x->to<g->nodes && x->elements;
}
static int mesh_block_stray(const struct mesh_routing *g,const struct mesh_block *x){
  if(x->from>=g->nodes || x->to>=g->nodes)return 1;
  const size_t p=(size_t)x->from*g->nodes+x->to;
  return mesh_block_moves(g,x) && g->pair[p]==g->pair[p+1];
}

static uint32_t mesh_route_size(const struct mesh_routing *g,uint32_t k){ return g->start[k+1]-g->start[k]; }
uint32_t mesh_block_parts(const struct mesh_routing *g,const struct mesh_block *x,int shortest,uint64_t page,struct mesh_part *parts){
  if(!mesh_block_moves(g,x) || mesh_block_stray(g,x))return 0;
  const size_t p=(size_t)x->from*g->nodes+x->to;
  const uint32_t r0=g->pair[p],r1=g->pair[p+1];
  uint32_t fewest=UINT32_MAX,count=0;
  for(uint32_t k=r0;k<r1;k++)if(mesh_route_size(g,k)<fewest)fewest=mesh_route_size(g,k);
#define MESH_TAKEN(k) (!shortest || mesh_route_size(g,k)==fewest)
  const uint64_t q=mesh_quantum(page,x->element_bytes),quanta=(x->elements+q-1)/q;
  double total=0;
  for(uint32_t k=r0;k<r1;k++)if(MESH_TAKEN(k))total+=g->share[k];
  uint64_t floors=0;
  for(uint32_t k=r0;k<r1;k++)if(MESH_TAKEN(k))floors+=(uint64_t)floor(g->share[k]/total*(double)quanta);
  uint64_t remainder=floors<quanta?quanta-floors:0,at=0;
  for(uint32_t k=r0;k<r1;k++){
    if(!MESH_TAKEN(k))continue;
    uint64_t e=(uint64_t)floor(g->share[k]/total*(double)quanta);
    if(remainder){e++;remainder--;}
    if(at+e>quanta)e=quanta-at;
    if(!e)continue;
    const uint64_t first=at*q,last=(at+e)*q<x->elements?(at+e)*q:x->elements;
    parts[count++]=(struct mesh_part){k,0,first,last-first};
    at+=e;
  }
#undef MESH_TAKEN
  if(count && at<quanta)parts[count-1].elements=x->elements-parts[count-1].first;
  return count;
}

uint32_t mesh_exchange_room(const struct mesh_routing *g,const struct mesh_block *blocks,uint32_t count,uint32_t *parts){
  uint64_t room=0;
  uint32_t most=0;
  for(uint32_t i=0;i<count;i++){
    if(!mesh_block_moves(g,blocks+i) || mesh_block_stray(g,blocks+i))continue;
    const size_t p=(size_t)blocks[i].from*g->nodes+blocks[i].to;
    const uint32_t routes=g->pair[p+1]-g->pair[p];
    room+=2*(uint64_t)routes;
    if(routes>most)most=routes;
  }
  if(parts)*parts=most;
  return room>=UINT32_MAX?UINT32_MAX-1:(uint32_t)room;
}

uint32_t mesh_exchange_steps(const struct mesh_routing *g,uint32_t rank,const struct mesh_block *blocks,uint32_t count,int pairwise,uint32_t relay,
                             uint64_t page,struct mesh_step *steps,uint32_t *block,uint64_t *relayed){
  for(uint32_t i=0;i<count;i++)if(mesh_block_stray(g,blocks+i))return UINT32_MAX;
  uint64_t parts=0;
  for(uint32_t i=0;i<count;i++)parts+=mesh_block_moves(g,blocks+i)?g->pair[(size_t)blocks[i].from*g->nodes+blocks[i].to+1]-
                                                                     g->pair[(size_t)blocks[i].from*g->nodes+blocks[i].to]:0;
  struct mesh_part *part=malloc(((size_t)parts+1)*sizeof *part);
  uint32_t *first=malloc(((size_t)count+1)*sizeof *first),*made=malloc(((size_t)count+1)*sizeof *made);
  uint64_t *place=malloc(((size_t)parts+1)*sizeof *place),bytes=0;
  if(!part || !first || !made || !place){free(part);free(first);free(made);free(place);return UINT32_MAX;}
  uint32_t rounds=0,at=0,k=0;
  for(uint32_t i=0;i<count;i++){
    const struct mesh_block *x=blocks+i;
    first[i]=at;
    made[i]=mesh_block_parts(g,x,pairwise,page,part+at);
    for(uint32_t j=0;j<made[i];j++){
      const uint32_t lo=g->start[part[at+j].route],h=g->start[part[at+j].route+1]-lo;
      if(pairwise && h>2)continue;
      if(h-1>rounds)rounds=h-1;
      uint32_t r=0;
      while(r<h && g->node[lo+r]!=rank)r++;
      place[at+j]=0;
      if(!r || r+1>=h)continue;
      const uint64_t e=x->element_bytes?x->element_bytes:1,quantum=mesh_quantum(page,(uint32_t)e)*e;
      bytes=(bytes+quantum-1)/quantum*quantum;
      place[at+j]=bytes/e;
      bytes+=part[at+j].elements*e;
    }
    at+=made[i];
  }
    for(uint32_t round=0;round<rounds;round++)for(uint32_t receive=0;receive<2;receive++)
    for(uint32_t i=0;i<count;i++){
      const struct mesh_block *x=blocks+i;
      for(uint32_t j=0;j<made[i];j++){
        const struct mesh_part *y=part+first[i]+j;
        const uint32_t lo=g->start[y->route],h=g->start[y->route+1]-lo;
        if(pairwise && h>2)continue;
        uint32_t r=0;
        while(r<h && g->node[lo+r]!=rank)r++;
        if(r>=h)continue;
        const struct mesh_operand piece={x->type,x->element_bytes,y->elements};
        const uint64_t relay_at=place[first[i]+j];
        if(!receive && r==round && r+1<h){
          if(block)block[k]=i;
          steps[k++]=(struct mesh_step){MESH_STEP_SEND,g->node[lo+r+1],round,r?relay:x->source,r?relay_at:x->first+y->first,piece};
        }
        if(receive && r && r-1==round){
          if(block)block[k]=i;
          steps[k++]=(struct mesh_step){MESH_STEP_COPY,g->node[lo+r-1],round,r+1<h?relay:x->target,r+1<h?relay_at:x->at+y->first,piece};
        }
      }
    }
  if(relayed)*relayed=bytes;
  free(part);free(first);free(made);free(place);
  return k;
}

static uint64_t mesh_step_bytes(const struct mesh_step *step){ return step->piece.elements*step->piece.element_bytes; }

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

static uint64_t mesh_segment_first(const uint64_t *segment,uint32_t q){
  uint64_t at=0;
  for(uint32_t r=0;r<q;r++)at+=segment[r];
  return at;
}
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
  for(uint32_t pass=0;pass<2;pass++)for(uint32_t i=0;i<g->calls;i++){
    const struct mesh_group_call *c=g->call+i;
    if(c->kind==MESH_GROUP_POINT || (c->kind==MESH_GROUP_HOSTED)!=pass)continue;
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
