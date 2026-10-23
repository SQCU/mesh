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

int mesh_link_map_joined(const struct mesh_link_map *map,uint32_t *apart){
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

static uint32_t mesh_trees_heaviest(const struct mesh_trees *trees,uint32_t r){
  uint32_t best=trees->count;
  for(uint32_t t=0;t<trees->count;t++)
    if(trees->root[t]==r && (best==trees->count || trees->log_weight[t]>trees->log_weight[best]))best=t;
  return best;
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

uint32_t mesh_solve_room(const struct mesh_link_map *map,double eps,uint32_t count){
  const uint32_t n=map->nodes;
  uint32_t edges=0;
  for(uint32_t a=0;a<n;a++)for(uint32_t b=0;b<n;b++)edges+=a!=b && mesh_joins(map,a,b);
  const double e=eps>0?eps:1;
  const uint64_t rounds=(uint64_t)(mesh_log(edges>1?(double)edges:2.0)/(e*e))+1,room=rounds*count;
  return room>=UINT32_MAX?UINT32_MAX-1:(uint32_t)room;
}

uint32_t mesh_solve(const struct mesh_link_map *map,double eps,double hop,const struct mesh_demand *demands,uint32_t count,uint32_t room,
                    uint32_t *demand,uint32_t *parent,double *share){
  const uint32_t n=map->nodes;
  if(!n || !count)return 0;
  const size_t nn=(size_t)n*n;
  double *cap=calloc(nn+1,sizeof *cap),*load=calloc(nn+1,sizeof *load),*add=calloc(nn+1,sizeof *add),*len=calloc(nn+1,sizeof *len),
         *dist=calloc((size_t)n+1,sizeof *dist);
  uint32_t *up=calloc((size_t)n+1,sizeof *up),*tree=calloc((size_t)n+1,sizeof *tree),*hits=calloc((size_t)room+1,sizeof *hits),made=0;
  unsigned char *done=calloc((size_t)n+1,1);
  uint32_t edges=0;
  double least=0,total=0;
  int failed=!cap || !load || !add || !len || !dist || !up || !tree || !hits || !done || !(eps>0);
  for(uint32_t a=0;!failed && a<n;a++)for(uint32_t b=0;b<n;b++){
    if(!mesh_joins(map,a,b))continue;
    double c=0;
    for(uint32_t l=0;l<map->links;l++)
      if((map->link[l][0]==a && map->link[l][1]==b) || (map->link[l][0]==b && map->link[l][1]==a))
        c+=map->bandwidth && map->bandwidth[l]?(double)map->bandwidth[l]/8:1;
    cap[(size_t)a*n+b]=c>0?c:1;
  }
  for(size_t e=0;!failed && e<nn;e++)if(cap[e]>0){edges++;if(!least || cap[e]<least)least=cap[e];}
  for(uint32_t d=0;!failed && d<count;d++)total+=(double)demands[d].up+(double)demands[d].down;
  const uint32_t rounds=failed?0:(uint32_t)(mesh_log(edges>1?(double)edges:2.0)/(eps*eps))+1;
  const double width=total>0 && least>0?total/rounds/least:1;
  for(uint32_t r=0;!failed && r<rounds;r++){
    double most=0;
    for(size_t e=0;e<nn;e++)if(cap[e]>0 && load[e]/cap[e]>most)most=load[e]/cap[e];
    for(size_t e=0;e<nn;e++){len[e]=cap[e]>0?mesh_exp(eps*(load[e]/cap[e]-most)/width)/cap[e]:0;add[e]=0;}
    for(uint32_t d=0;!failed && d<count;d++){
      const struct mesh_demand *x=demands+d;
      const double u=(double)x->up,w=(double)x->down;
      for(uint32_t v=0;v<n;v++){dist[v]=INFINITY;up[v]=UINT32_MAX;done[v]=0;tree[v]=UINT32_MAX;}
      if(x->root>=n){failed=1;break;}
      dist[x->root]=0;up[x->root]=x->root;
      for(uint32_t step=0;step<n;step++){
        uint32_t at=n;
        for(uint32_t v=0;v<n;v++)if(!done[v] && dist[v]<INFINITY && (at==n || dist[v]<dist[at]))at=v;
        if(at==n)break;
        done[at]=1;
        for(uint32_t v=0;v<n;v++){
          if(done[v] || !(cap[(size_t)v*n+at]>0))continue;
          const double through=dist[at]+u*len[(size_t)v*n+at]+w*len[(size_t)at*n+v]+hop;
          if(through<dist[v]){dist[v]=through;up[v]=at;}
        }
      }
      tree[x->root]=x->root;
      for(uint32_t v=0;v<n && !failed;v++){
        if(v==x->root || !mesh_member(x->leaves,v))continue;
        if(up[v]==UINT32_MAX){failed=1;break;}
        for(uint32_t at=v;tree[at]==UINT32_MAX;at=up[at])tree[at]=up[at];
      }
      if(failed)break;
      uint32_t k=0;
      while(k<made && !(demand[k]==d && !memcmp(parent+(size_t)k*n,tree,n*sizeof *tree)))k++;
      if(k==made && made==room){failed=1;break;}
      if(k==made){demand[made]=d;memcpy(parent+(size_t)made*n,tree,n*sizeof *tree);made++;}
      hits[k]++;
      for(uint32_t v=0;v<n;v++){
        if(v==x->root || tree[v]==UINT32_MAX)continue;
        add[(size_t)v*n+tree[v]]+=u/rounds;
        add[(size_t)tree[v]*n+v]+=w/rounds;
      }
    }
    for(size_t e=0;e<nn;e++)load[e]+=add[e];
  }
  for(uint32_t k=0;!failed && k<made;k++)share[k]=(double)hits[k]/rounds;
  free(cap);free(load);free(add);free(len);free(dist);free(up);free(tree);free(hits);free(done);
  return failed?UINT32_MAX:made;
}

uint32_t mesh_flow_trees(uint32_t nodes,const struct mesh_demand *demands,const uint32_t *demand,const uint32_t *parent,const double *share,
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

uint32_t mesh_flow_routing(uint32_t nodes,const struct mesh_demand *demands,const uint32_t *demand,const uint32_t *parent,const double *share,
                           uint32_t count,uint32_t first,uint32_t last,struct mesh_routing *routing,uint32_t *pair,uint32_t *start,uint32_t *node,
                           double *weight){
  uint32_t routes=0,hops=0;
  for(uint32_t a=0;a<nodes;a++)for(uint32_t b=0;b<nodes;b++){
    pair[(size_t)a*nodes+b]=routes;
    for(uint32_t k=0;a!=b && k<count;k++){
      if(demand[k]<first || demand[k]>=last || demands[demand[k]].root!=b || !mesh_member(demands[demand[k]].leaves,a))continue;
      const uint32_t *up=parent+(size_t)k*nodes;
      start[routes]=hops;weight[routes]=share[k];
      for(uint32_t at=a,steps=0;steps<nodes;at=up[at],steps++){
        node[hops++]=at;
        if(at==b || up[at]==UINT32_MAX)break;
      }
      routes++;
    }
  }
  pair[(size_t)nodes*nodes]=routes;start[routes]=hops;
  *routing=(struct mesh_routing){nodes,routes,pair,start,node,weight};
  return routes;
}

void mesh_cut_shares(uint64_t elements,uint32_t element_bytes,uint64_t page,const double *share,uint32_t count,uint64_t *first,uint64_t *length){
  const uint64_t q=mesh_quantum(page,element_bytes),quanta=(elements+q-1)/q;
  double total=0;
  for(uint32_t k=0;k<count;k++)total+=share[k];
  uint64_t floors=0;
  for(uint32_t k=0;k<count;k++)floors+=total>0?(uint64_t)floor(share[k]/total*(double)quanta):0;
  uint64_t remainder=floors<quanta?quanta-floors:0,at=0;
  for(uint32_t k=0;k<count;k++){
    uint64_t e=total>0?(uint64_t)floor(share[k]/total*(double)quanta):0;
    if(remainder){e++;remainder--;}
    if(at+e>quanta)e=quanta-at;
    first[k]=at*q<elements?at*q:elements;
    length[k]=((at+e)*q<elements?(at+e)*q:elements)-first[k];
    at+=e;
  }
  if(count && at<quanta)length[count-1]=elements-first[count-1];
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
  double *share=malloc(((size_t)(r1-r0)+1)*sizeof *share);
  uint64_t *first=malloc(((size_t)(r1-r0)+1)*sizeof *first),*length=malloc(((size_t)(r1-r0)+1)*sizeof *length);
  uint32_t *route=malloc(((size_t)(r1-r0)+1)*sizeof *route),taken=0;
  for(uint32_t k=r0;share && first && length && route && k<r1;k++)if(MESH_TAKEN(k)){route[taken]=k;share[taken++]=g->share[k];}
#undef MESH_TAKEN
  if(share && first && length && route)mesh_cut_shares(x->elements,x->element_bytes,page,share,taken,first,length);
  for(uint32_t i=0;share && first && length && route && i<taken;i++)if(length[i])parts[count++]=(struct mesh_part){route[i],0,first[i],length[i]};
  free(share);free(first);free(length);free(route);
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
