#include "mesh-call.h"
#include "mesh-plan.h"
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#pragma STDC FP_CONTRACT OFF

static uint32_t mesh_all(const struct mesh_link_map *map){ return map->nodes+map->relays; }

static int mesh_edge_joins(const struct mesh_edge *e,uint32_t u,uint32_t x){ return (e->a==u && e->b==x) || (e->a==x && e->b==u); }

static int mesh_joins(const struct mesh_link_map *map,uint32_t u,uint32_t x){
  const uint32_t N=mesh_all(map);
  if(u==x || u>=N || x>=N)return 0;
  for(uint32_t l=0;l<map->edges;l++)if(mesh_edge_joins(map->edge+l,u,x))return 1;
  return 0;
}

static uint64_t mesh_capacity(const struct mesh_link_map *map,uint32_t u,uint32_t x){
  uint64_t sum=0;
  for(uint32_t l=0;l<map->edges;l++)if(mesh_edge_joins(map->edge+l,u,x))sum+=map->edge[l].capacity;
  return sum;
}

static uint64_t mesh_hop_inverse(const struct mesh_link_map *map,uint32_t u,uint32_t x){
  const uint64_t capacity=mesh_capacity(map,u,x);
  return capacity?(uint64_t)llround(1e9/(double)capacity*1e6):0;
}

static uint32_t mesh_mark(const struct mesh_link_map *map,uint32_t v){ return map->mark[v]; }

uint32_t mesh_route_path(const struct mesh_link_map *map,uint32_t a,uint32_t b,uint32_t *path){
  const uint32_t N=mesh_all(map);
  if(a==b || a>=N || b>=N || !map->next)return 0;
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

static int mesh_member(const uint64_t *set,uint32_t v){ return !set || (set[v/64]>>(v%64)&1); }

int mesh_link_between(const struct mesh_link_map *map,uint32_t a,uint32_t b){
  if(a==b || a>=map->nodes || b>=map->nodes)return 0;
  uint32_t path[mesh_all(map)+1];
  const uint32_t count=mesh_route_path(map,a,b,path);
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
  if(!N || (map->edges && !map->edge) || !map->mark)return EINVAL;
  for(uint32_t l=0;l<map->edges;l++)if(map->edge[l].a>=N || map->edge[l].b>=N || map->edge[l].a==map->edge[l].b)return EINVAL;
  const size_t pairs=(size_t)N*N;
  uint64_t *inverse=calloc(pairs,sizeof *inverse);
  unsigned char *joined=calloc(pairs,1),*done=malloc(N);
  struct mesh_length *length=malloc(N*sizeof *length);
  const int status=inverse && joined && done && length?0:ENOMEM;

  for(uint32_t l=0;!status && l<map->edges;l++){
    const uint32_t u=map->edge[l].a,x=map->edge[l].b;
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
  if(whole->relays || (whole->edges && !whole->edge) || !whole->next || !whole->mark)return -EINVAL;
  const uint32_t L=whole->edges;
  for(uint32_t k=0;k<L;k++)if(whole->edge[k].a>=W || whole->edge[k].b>=W || !whole->edge[k].capacity)return -EINVAL;
  uint32_t *place=malloc(((size_t)W+1)*sizeof *place),*at=malloc(((size_t)N+W+1)*sizeof *at),relays=0;
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
  if(!error && (map->nodes<T || map->edges<L)){map->nodes=T;map->edges=L;error=ENOSPC;}
  else if(!error && relays && (!map->next || !map->mark))error=EINVAL;
  const uint32_t *table=whole->next;
  if(!error){
    for(uint32_t k=0;k<L;k++)
      map->edge[k]=(struct mesh_edge){place[whole->edge[k].a],place[whole->edge[k].b],whole->edge[k].cable,whole->edge[k].capacity};
    for(uint32_t i=0;map->next && i<T;i++)for(uint32_t j=0;j<T;j++)
      map->next[(size_t)i*T+j]=at[i]<W && at[j]<W?place[table[(size_t)at[i]*W+at[j]]]:i;
    for(uint32_t i=0;whole_node && i<T;i++)whole_node[i]=at[i];
    for(uint32_t i=0;map->mark && i<T;i++)map->mark[i]=at[i]<W?mesh_mark(whole,at[i]):UINT32_MAX-i;
    map->nodes=N;map->edges=L;map->relays=relays;
  }
  free(place);free(at);
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

struct mesh_shape { uint32_t height,depth; int inner; };
static void mesh_tree_shape(const struct mesh_trees *trees,uint32_t t,struct mesh_shape *shape){
  const uint32_t n=trees->nodes,r=trees->root[t],*parent=trees->parent+(size_t)t*n;
  for(uint32_t v=0;v<n;v++)shape[v]=(struct mesh_shape){0,0,0};
  for(uint32_t v=0;v<n;v++){
    uint32_t at=v,k=0;
    for(;at!=r && k<n;k++){
      const uint32_t up=parent[at];
      if(shape[up].height<k+1)shape[up].height=k+1;
      at=up;
    }
    shape[v].depth=k;
  }
  for(uint32_t v=0;v<n;v++)if(v!=r)shape[parent[v]].inner=1;
}

struct mesh_message { uint32_t from,to,round,part,parts,out,in,op,key,way; };
static int mesh_message_order(const void *a,const void *b){
  const struct mesh_message *x=a,*y=b;
  const uint32_t l[6]={x->from,x->to,x->way,x->round,x->key,x->part},r[6]={y->from,y->to,y->way,y->round,y->key,y->part};
  for(int i=0;i<6;i++)if(l[i]!=r[i])return l[i]<r[i]?-1:1;
  return 0;
}

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

int mesh_compile(const struct mesh_trees *trees,uint32_t what,uint32_t root,struct mesh_program *program,uint32_t *segment,double *log_weight,
                 uint32_t *first,struct mesh_move *move){
  const uint32_t n=trees->nodes,T=trees->count;
  if(!n || what>MESH_ALLGATHER || ((what==MESH_REDUCE || what==MESH_BROADCAST) && root>=n))return EINVAL;
  uint32_t room_parts,room;
  mesh_compile_room(trees,&room_parts,&room);
  struct mesh_shape *shape=malloc(((size_t)T*n+1)*sizeof *shape);
  struct mesh_message *message=malloc((room+1)*sizeof *message);
  struct mesh_ordered *ordered=malloc((2*(size_t)room+1)*sizeof *ordered);
  uint32_t *part_of=malloc(((size_t)T+1)*sizeof *part_of),*buffer=calloc((size_t)n*n+1,sizeof *buffer);
  uint32_t messages=0,parts=0,scratch=0;
  int status=EINVAL;
  if(!shape || !message || !ordered || !part_of || !buffer)goto done;
  for(uint32_t t=0;t<T;t++)mesh_tree_shape(trees,t,shape+(size_t)t*n);
  for(size_t k=0;k<T;k++)part_of[k]=UINT32_MAX;
  if(what==MESH_ALLREDUCE){
    parts=1;segment[0]=0;log_weight[0]=0;
    for(uint32_t u=0;u<n;u++){
      uint32_t next=2,own=0;
      for(uint32_t v=0;v<n;v++){
        const uint32_t t=mesh_trees_heaviest(trees,v);
        if(t==T)goto done;
        if(u==v)continue;
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
        if(u==v)continue;
        const uint32_t w=parent[u];
        message[messages++]=(struct mesh_message){u,w,s[u].height,0,1,s[u].inner?buffer[(size_t)u*n+v]:buffer[(size_t)u*n+u],w==v?0:buffer[(size_t)w*n+v],
                                                  MESH_STEP_REDUCE,v,trees->way?trees->way[(size_t)t*n+u]:0};
      }
    }
  }else{
    const int rooted=what==MESH_REDUCE || what==MESH_BROADCAST;
    for(uint32_t g=0;g<(rooted?1:n);g++)
      for(uint32_t t=0;t<T;t++)
        if(trees->root[t]==(rooted?root:g)){part_of[t]=parts;segment[parts]=rooted?0:g;log_weight[parts]=trees->log_weight[t];parts++;}
    if(!parts)goto done;
    const int up=what==MESH_REDUCE || what==MESH_REDUCE_SCATTER;
    for(uint32_t t=0;t<T;t++){
      if(part_of[t]==UINT32_MAX)continue;
      const uint32_t r=trees->root[t],*parent=trees->parent+(size_t)t*n,p=part_of[t];
      const struct mesh_shape *s=shape+(size_t)t*n;
      for(uint32_t u=0;u<n;u++){
        if(u==r)continue;
        message[messages++]=up?(struct mesh_message){u,parent[u],s[u].height,p,1,0,0,MESH_STEP_REDUCE,p,trees->way?trees->way[(size_t)t*n+u]:0}
                              :(struct mesh_message){parent[u],u,s[u].depth-1,p,1,0,0,MESH_STEP_COPY,p,trees->way?trees->way[(size_t)t*n+u]:0};
      }
    }
  }

  qsort(message,messages,sizeof *message,mesh_message_order);
  uint32_t moves=0;
  for(uint32_t i=0;i<messages;i++){
    const struct mesh_message *m=message+i;
    ordered[moves++]=(struct mesh_ordered){m->from,m->round,0,m->to,m->key,{MESH_STEP_SEND,m->to,m->round,m->out,m->part,m->parts,m->way,0}};
    ordered[moves++]=(struct mesh_ordered){m->to,m->round,1,m->from,m->key,{m->op,m->from,m->round,m->in,m->part,m->parts,m->way,0}};
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
      steps[k++]=(struct mesh_step){m->op,m->peer,m->round,m->buffer,lo[m->part],piece,m->way,0};
    }
  }
  free(lo);free(count);
  return k;
}

static uint32_t mesh_edges(const struct mesh_link_map *map,uint32_t *from,uint32_t *to,double *cap,uint32_t *way){
  const uint32_t n=map->nodes;
  uint32_t E=0;
  for(uint32_t l=0;l<map->edges;l++){
    const uint32_t a=map->edge[l].a,b=map->edge[l].b;
    if(a>=n || b>=n || a==b)continue;
    if(from){
      uint32_t w=0;
      for(uint32_t e=0;e<E;e++)w+=(from[e]==a && to[e]==b) || (from[e]==b && to[e]==a);
      from[E]=a;to[E]=b;way[E]=w;cap[E]=(double)map->edge[l].capacity;
    }
    E++;
  }
  return E;
}

uint32_t mesh_solve_room(const struct mesh_link_map *map,double eps,uint32_t count){
  const uint32_t edges=2*mesh_edges(map,NULL,NULL,NULL,NULL);
  const double e=eps>0?eps:1;
  const uint64_t rounds=(uint64_t)(mesh_log(edges>1?(double)edges:2.0)/(e*e))+1,room=rounds*count;
  return room>=UINT32_MAX?UINT32_MAX-1:(uint32_t)room;
}

uint32_t mesh_solve(const struct mesh_link_map *map,double eps,double hop,const struct mesh_demand *demands,uint32_t count,uint32_t room,
                    uint32_t *demand,uint32_t *parent,uint32_t *way,double *share){
  const uint32_t n=map->nodes,E=mesh_edges(map,NULL,NULL,NULL,NULL);
  if(!n || !count)return 0;
  uint32_t *from=calloc((size_t)E+1,sizeof *from),*to=calloc((size_t)E+1,sizeof *to),*ways=calloc((size_t)E+1,sizeof *ways);
  double *cap=calloc((size_t)E+1,sizeof *cap),*load=calloc(2*(size_t)E+1,sizeof *load),*add=calloc(2*(size_t)E+1,sizeof *add),
         *len=calloc(2*(size_t)E+1,sizeof *len),*dist=calloc((size_t)n+1,sizeof *dist);
  uint32_t *up=calloc((size_t)n+1,sizeof *up),*upl=calloc((size_t)n+1,sizeof *upl),*tree=calloc((size_t)n+1,sizeof *tree),
           *treel=calloc((size_t)n+1,sizeof *treel),*links=calloc((size_t)room*n+1,sizeof *links),*hits=calloc((size_t)room+1,sizeof *hits),made=0;
  unsigned char *done=calloc((size_t)n+1,1);
  int failed=!from || !to || !ways || !cap || !load || !add || !len || !dist || !up || !upl || !tree || !treel || !links || !hits || !done || !(eps>0);
  if(!failed)mesh_edges(map,from,to,cap,ways);
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
      for(uint32_t step=0;step<n;step++){
        uint32_t at=n;
        for(uint32_t v=0;v<n;v++)if(!done[v] && dist[v]<INFINITY && (at==n || dist[v]<dist[at]))at=v;
        if(at==n)break;
        done[at]=1;
        for(uint32_t e=0;e<E;e++){
          if(from[e]!=at && to[e]!=at)continue;
          const uint32_t v=from[e]==at?to[e]:from[e],toward=from[e]==v?0:1;
          if(done[v])continue;
          const double through=dist[at]+u*len[2*(size_t)e+toward]+w*len[2*(size_t)e+(toward^1)]+hop;
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
      while(k<made && !(demand[k]==d && !memcmp(parent+(size_t)k*n,tree,n*sizeof *tree) && !memcmp(links+(size_t)k*n,treel,n*sizeof *treel)))k++;
      if(k==made){
        demand[made]=d;
        memcpy(parent+(size_t)made*n,tree,n*sizeof *tree);
        memcpy(links+(size_t)made*n,treel,n*sizeof *treel);
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
  for(uint32_t k=0;!failed && k<made;k++){
    share[k]=(double)hits[k]/rounds;
    for(uint32_t v=0;v<n;v++)way[(size_t)k*n+v]=links[(size_t)k*n+v]==UINT32_MAX?0:ways[links[(size_t)k*n+v]];
  }
  free(from);free(to);free(ways);free(cap);free(load);free(add);free(len);free(dist);free(up);free(upl);free(tree);free(treel);free(links);free(hits);
  free(done);
  return failed?UINT32_MAX:made;
}

uint32_t mesh_flow_trees(uint32_t nodes,const struct mesh_demand *demands,const uint32_t *demand,const uint32_t *parent,const uint32_t *way,
                         const double *share,uint32_t count,uint32_t first,uint32_t last,uint32_t *root,uint32_t *tree,uint32_t *tree_way,
                         double *log_weight){
  uint32_t trees=0;
  for(uint32_t k=0;k<count;k++){
    if(demand[k]<first || demand[k]>=last)continue;
    root[trees]=demands[demand[k]].root;
    memcpy(tree+(size_t)trees*nodes,parent+(size_t)k*nodes,nodes*sizeof *tree);
    memcpy(tree_way+(size_t)trees*nodes,way+(size_t)k*nodes,nodes*sizeof *tree_way);
    log_weight[trees++]=mesh_log(share[k]);
  }
  return trees;
}

uint32_t mesh_flow_routing(uint32_t nodes,const struct mesh_demand *demands,const uint32_t *demand,const uint32_t *parent,const uint32_t *way,
                           const double *share,uint32_t count,uint32_t first,uint32_t last,struct mesh_routing *routing,uint32_t *pair,
                           uint32_t *start,uint32_t *node,uint32_t *hop_way,double *weight){
  uint32_t routes=0,hops=0;
  for(uint32_t a=0;a<nodes;a++)for(uint32_t b=0;b<nodes;b++){
    pair[(size_t)a*nodes+b]=routes;
    for(uint32_t k=0;a!=b && k<count;k++){
      if(demand[k]<first || demand[k]>=last || demands[demand[k]].root!=b || !mesh_member(demands[demand[k]].leaves,a))continue;
      const uint32_t *up=parent+(size_t)k*nodes;
      start[routes]=hops;weight[routes]=share[k];
      for(uint32_t at=a,steps=0;steps<nodes;at=up[at],steps++){
        hop_way[hops]=way[(size_t)k*nodes+at];
        node[hops++]=at;
        if(at==b || up[at]==UINT32_MAX)break;
      }
      routes++;
    }
  }
  pair[(size_t)nodes*nodes]=routes;start[routes]=hops;
  *routing=(struct mesh_routing){nodes,routes,pair,start,node,weight,hop_way};
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

static uint32_t mesh_route_size(const struct mesh_routing *g,uint32_t k){ return g->start[k+1]-g->start[k]; }
uint32_t mesh_block_parts(const struct mesh_routing *g,const struct mesh_block *x,int shortest,uint64_t page,struct mesh_part *parts){
  if(!mesh_block_moves(g,x))return 0;
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
    if(!mesh_block_moves(g,blocks+i))continue;
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
          steps[k++]=(struct mesh_step){MESH_STEP_SEND,g->node[lo+r+1],round,r?relay:x->source,r?relay_at:x->first+y->first,piece,
                                        g->way?g->way[lo+r]:0,0};
        }
        if(receive && r && r-1==round){
          if(block)block[k]=i;
          steps[k++]=(struct mesh_step){MESH_STEP_COPY,g->node[lo+r-1],round,r+1<h?relay:x->target,r+1<h?relay_at:x->at+y->first,piece,
                                        g->way?g->way[lo+r-1]:0,0};
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

uint64_t mesh_lay(const struct mesh_step *steps,uint32_t count,uint32_t peer,uint32_t way,int send,uint64_t at,struct mesh_place *made,uint32_t *placed){
  const int sorted=mesh_rounds_sorted(steps,count);
  uint32_t rounds=0;
#define MESH_LAID(k) (steps[k].peer==peer && steps[k].way==way && mesh_step_bytes(steps+(k)) && (steps[k].op==MESH_STEP_SEND)==(send!=0))
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

uint64_t mesh_group_lay(const struct mesh_group *g,uint32_t rank,uint32_t peer,uint32_t way,int send,uint64_t at,struct mesh_place *scratch,
                        struct mesh_group_place *made,uint32_t *placed){
  *placed=0;
  for(uint32_t pass=0;pass<2;pass++)for(uint32_t i=0;i<g->calls;i++){
    const struct mesh_group_call *c=g->call+i;
    if(c->kind==MESH_GROUP_POINT || (c->kind==MESH_GROUP_HOSTED)!=pass)continue;
    uint32_t laid=0;
    at=mesh_lay(c->steps[rank],c->count[rank],peer,way,send,at,scratch,&laid);
    for(uint32_t k=0;k<laid;k++)made[(*placed)++]=(struct mesh_group_place){i,0,scratch[k]};
  }
  for(uint32_t i=0;i<g->calls;i++){
    const struct mesh_group_call *c=g->call+i;
    if(c->kind!=MESH_GROUP_POINT)continue;
    for(uint32_t k=0;k<c->count[rank];k++){
      const struct mesh_step *step=c->steps[rank]+k;
      if(step->peer==peer && step->way==way && mesh_step_bytes(step) && (step->op==MESH_STEP_SEND)==(send!=0))
        made[(*placed)++]=(struct mesh_group_place){i,0,{k,send!=0,at++}};
    }
  }
  return at;
}
