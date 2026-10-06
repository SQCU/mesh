#include "mesh-collective.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int mesh_link_map_check(const struct mesh_link_map *map){
  const uint32_t n=map->nodes;
  uint32_t at,steps;
  if(map->kind>MESH_LINKS_GRAPH || !n || (map->links && !map->link))return EINVAL;
  for(uint32_t l=0;l<map->links;l++)if(map->link[l][0]>=n || map->link[l][1]>=n || map->link[l][0]==map->link[l][1])return EINVAL;
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

/* The text form: the MLX JACCL hostfile idea (github.com/ml-explore/mlx docs/src/usage/distributed.rst) reduced to a
   kind, a node count and links with their costs; refused where mesh_link_map_check refuses it. */
int mesh_link_map_read(const char *path,struct mesh_link_map *map){
  static const char *const kinds[]={"mesh","ring","tree","graph"};
  FILE *file=fopen(path,"r");
  if(!file)return errno;
  char line[256],kind[8]="",rest[2];
  uint32_t a,b,capacity=0,costed=0;
  double alpha,beta;
  *map=(struct mesh_link_map){.kind=UINT32_MAX};
  while(fgets(line,sizeof line,file)){
    const int fields=sscanf(line," %u %u %lf %lf %1s",&a,&b,&alpha,&beta,rest);
    if(fields>=2){
      if(map->links==capacity){
        capacity=capacity?2*capacity:64;
        uint32_t (*grown)[2]=realloc(map->link,(size_t)capacity*sizeof *map->link);
        double (*priced)[2]=grown?realloc(map->cost,(size_t)capacity*sizeof *map->cost):NULL;
        if(grown)map->link=grown;
        if(priced)map->cost=priced;
        if(!grown || !priced){fclose(file);mesh_link_map_free(map);return ENOMEM;}
      }
      map->link[map->links][0]=a;map->link[map->links][1]=b;
      map->cost[map->links][0]=fields==4?alpha:NAN;map->cost[map->links][1]=fields==4?beta:NAN;
      costed+=fields==4;map->links++;
    }else if(map->kind==UINT32_MAX && sscanf(line,"%7s %u",kind,&a)==2)
      for(uint32_t k=0;k<4;k++)if(!strcmp(kind,kinds[k])){map->kind=k;map->nodes=a;}
  }
  fclose(file);
  if(!costed){free(map->cost);map->cost=NULL;}
  int result=map->kind==UINT32_MAX?EINVAL:mesh_link_map_check(map);
  if(result)mesh_link_map_free(map);
  return result;
}

void mesh_link_map_free(struct mesh_link_map *map){
  free(map->link);free(map->cost);map->link=NULL;map->cost=NULL;map->links=0;
}

/* Whether node v is in a bit set of nodes (a bit a node, 64 a word; NULL: every node). */
static int mesh_member(const uint64_t *set,uint32_t v){ return !set || (set[v/64]>>(v%64)&1); }

/* Whether the map links a and b, either way (a mesh links every pair; a ring, a tree or a graph its links). */
static int mesh_linked(const struct mesh_link_map *map,uint32_t a,uint32_t b){
  if(a==b || a>=map->nodes || b>=map->nodes)return 0;
  if(map->kind==MESH_LINKS_MESH)return 1;
  for(uint32_t l=0;l<map->links;l++)
    if((map->link[l][0]==a && map->link[l][1]==b) || (map->link[l][0]==b && map->link[l][1]==a))return 1;
  return 0;
}

int mesh_link_between(const struct mesh_link_map *map,uint32_t a,uint32_t b){ return mesh_linked(map,a,b); }

/* The cost of the link between a and b: its own where the map gives one, else alpha and beta. */
static void mesh_link_cost(const struct mesh_link_map *map,uint32_t a,uint32_t b,double alpha,double beta,double *link_alpha,double *link_beta){
  *link_alpha=alpha;*link_beta=beta;
  for(uint32_t l=0;map->cost && l<map->links;l++)
    if(((map->link[l][0]==a && map->link[l][1]==b) || (map->link[l][0]==b && map->link[l][1]==a)) && !isnan(map->cost[l][0])){
      *link_alpha=map->cost[l][0];*link_beta=map->cost[l][1];return;
    }
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
      if(v!=r && !mesh_linked(map,v,parent[v]))return EINVAL;
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
  const uint32_t n=map->nodes;
  uint32_t *distance=malloc((n+1)*sizeof *distance),*queue=malloc((n+1)*sizeof *queue),made=n;
  if(!distance || !queue){free(distance);free(queue);return 0;}
  for(uint32_t r=0;r<n && made;r++){
    for(uint32_t v=0;v<n;v++)distance[v]=UINT32_MAX;
    uint32_t head=0,tail=0;
    distance[r]=0;queue[tail++]=r;
    while(head<tail){
      const uint32_t at=queue[head++];
      for(uint32_t w=0;w<n;w++)if(distance[w]==UINT32_MAX && mesh_linked(map,at,w)){distance[w]=distance[at]+1;queue[tail++]=w;}
    }
    if(tail<n){made=0;break;}
    root[r]=r;log_weight[r]=0;
    /* a node's parent: of its neighbours one link nearer the root, the one over the cheapest link (its alpha, then
       its beta), the lowest-numbered of equals: the trees follow the map's costs, the ids only break ties */
    for(uint32_t v=0;v<n;v++){
      uint32_t up=v;
      double best_alpha=0,best_beta=0;
      for(uint32_t w=0;v!=r && w<n;w++){
        if(distance[w]+1!=distance[v] || !mesh_linked(map,v,w))continue;
        double a,b;
        mesh_link_cost(map,v,w,0,0,&a,&b);
        if(up==v || a<best_alpha || (a==best_alpha && b<best_beta)){up=w;best_alpha=a;best_beta=b;}
      }
      parent[(size_t)r*n+v]=v==r?r:up;
    }
  }
  free(distance);free(queue);
  return made;
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

void mesh_compile_room(const struct mesh_trees *trees,uint32_t *parts,uint32_t *moves){
  const uint32_t n=trees->nodes,T=trees->count;
  *parts=T>1?T:1;
  *moves=4*T*n+2*n*n+2;
}

int mesh_compile(const struct mesh_trees *trees,uint32_t what,uint32_t root,int whole,const uint64_t *contributors,
                 struct mesh_program *program,uint32_t *segment,double *log_weight,uint32_t *first,struct mesh_move *move){
  const uint32_t n=trees->nodes,T=trees->count;
  if(!n || what>MESH_ALLGATHER || ((what==MESH_REDUCE || what==MESH_BROADCAST) && root>=n))return EINVAL;
  uint32_t room_parts,room;
  mesh_compile_room(trees,&room_parts,&room);
  const uint64_t *mine=what==MESH_ALLREDUCE?contributors:NULL;
  struct mesh_shape *shape=malloc(((size_t)T*n+1)*sizeof *shape);
  struct mesh_message *message=malloc((room+1)*sizeof *message);
  struct mesh_ordered *ordered=malloc((2*(size_t)room+1)*sizeof *ordered);
  uint32_t *part_of=malloc((T+1)*sizeof *part_of),*buffer=calloc((size_t)n*n+1,sizeof *buffer);
  uint32_t messages=0,parts=0,scratch=0;
  int status=EINVAL;
  if(!shape || !message || !ordered || !part_of || !buffer)goto done;
  for(uint32_t t=0;t<T;t++){mesh_tree_shape(trees,t,mine,shape+(size_t)t*n);part_of[t]=UINT32_MAX;}
  if(what==MESH_ALLREDUCE && whole){
    /* the whole operand toward every node v up v's heaviest tree: u's partial for v from scratch of its own where it
       has children there, else its own operand from its copy (scratch 1), which nothing changes */
    parts=1;segment[0]=0;log_weight[0]=0;
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
        message[messages++]=(struct mesh_message){u,w,s[u].height,0,1,s[u].inner?buffer[(size_t)u*n+v]:1,w==v?0:buffer[(size_t)w*n+v],op,v};
      }
    }
  }else{
    /* the parts: a rooted collective's root's trees, else each segment's own node's, each a part at its weight; an
       all-gather's segments its contributors' alone, in node order */
    const int rooted=what==MESH_REDUCE || what==MESH_BROADCAST;
    for(uint32_t g=0,index=0;g<(rooted?1:n);g++){
      if(what==MESH_ALLGATHER && !mesh_member(contributors,g))continue;
      for(uint32_t t=0;t<T;t++)
        if(trees->root[t]==(rooted?root:g)){part_of[t]=parts;segment[parts]=rooted?0:index;log_weight[parts]=trees->log_weight[t];parts++;}
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
      const struct mesh_shape *s=shape+(size_t)t*n;
      for(uint32_t u=0;up && u<n;u++){
        if(u==r || !s[u].live)continue;
        const uint32_t w=parent[u],op=!mesh_member(mine,w) && mesh_first_child(trees,t,s,w,u)?MESH_STEP_COPY:MESH_STEP_REDUCE;
        message[messages++]=(struct mesh_message){u,w,s[u].height,p,1,0,0,op,p};
      }
      for(uint32_t u=0;down && u<n;u++)
        if(u!=r)message[messages++]=(struct mesh_message){parent[u],u,base+s[u].depth-1,p,1,0,0,MESH_STEP_COPY,p};
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
  const uint32_t n=map->nodes;
  if(n==1){root[0]=0;parent[0]=0;log_weight[0]=0;return 1;}
  if(n<2 || !most || mesh_link_map_check(map))return 0;
  if(!(eps>0 && eps<1))eps=0.1;
  uint32_t m=0;
  for(uint32_t a=0;a<n;a++)for(uint32_t b=0;b<n;b++)m+=mesh_linked(map,a,b);
  uint32_t *from=malloc((m+1)*sizeof *from),*to=malloc((m+1)*sizeof *to),*index=malloc((size_t)n*n*sizeof *index);
  double *capacity=malloc((m+1)*sizeof *capacity),*length=malloc((m+1)*sizeof *length);
  int32_t *chosen=malloc(n*sizeof *chosen);
  struct mesh_packed *packed=NULL;
  size_t trees=0,room=0;
  uint32_t written=0;
  if(!from || !to || !index || !capacity || !length || !chosen)goto done;
  /* arc v -> w (v's parent w): edge e, from[e] = w, to[e] = v; its capacity 1/beta of the link's cost */
  m=0;
  for(uint32_t v=0;v<n;v++)for(uint32_t w=0;w<n;w++)if(mesh_linked(map,v,w)){
    double alpha,beta;
    mesh_link_cost(map,v,w,0,1,&alpha,&beta);
    from[m]=w;to[m]=v;capacity[m]=beta>0 && isfinite(beta)?1/beta:1;index[(size_t)v*n+w]=m;m++;
  }
  const double delta=mesh_exp(-mesh_log(m/(1-eps))/eps);
  double D=0;
  for(uint32_t e=0;e<m;e++){length[e]=delta/capacity[e];D+=length[e]*capacity[e];}
  while(D<1)for(uint32_t r=0;r<n && D<1;r++)for(double demand=1;demand>0 && D<1;){
    if(mesh_arborescence(n,r,m,from,to,length,chosen))goto done;
    double f=demand;
    for(uint32_t v=0;v<n;v++)if(v!=r && capacity[chosen[v]]<f)f=capacity[chosen[v]];
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
    for(uint32_t v=0;v<n;v++)if(v!=r){
      const uint32_t e=(uint32_t)chosen[v];
      D+=length[e]*eps*f;
      length[e]*=1+eps*f/capacity[e];
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
  free(packed);free(from);free(to);free(index);free(capacity);free(length);free(chosen);
  return written;
}

/* Every rank's steps of a program, then its time under the alpha-beta model [Hockney 1994] with a node's sends sharing
   its outgoing port and its receives its incoming one, as Thakur, Rabenseifner and Gropp cost these algorithms: each
   node takes its steps in order; a SEND leaves when the node has reached it and its port is free, and arrives its
   link's alpha after its bytes x its link's beta; a receive waits for its SEND (the sender's k-th to it in the round
   is its k-th from the sender) and its port.  The time is the last node's; negative where a receive has no SEND of
   its piece, or the schedule stops. */
double mesh_program_time(const struct mesh_link_map *map,const struct mesh_program *program,struct mesh_operand operand,double alpha,double beta){
  const uint32_t n=program->nodes,capacity=program->first[n]+1;
  if(n<2)return 0;
  struct mesh_step *steps=calloc((size_t)n*capacity,sizeof *steps);
  double *arrival=calloc((size_t)n*capacity,sizeof *arrival),result=-1;
  uint32_t count[n],cursor[n];
  double clock[n];
  /* each link its own ports: a node sends on every cable at once and receives on every cable at once, one message at
     a time each way of each link */
  double *out_free=calloc((size_t)n*n,sizeof *out_free),*in_free=calloc((size_t)n*n,sizeof *in_free);
  memset(cursor,0,sizeof cursor);memset(clock,0,sizeof clock);
  if(!steps||!arrival||!out_free||!in_free)goto done;
  for(uint32_t r=0;r<n;r++)count[r]=mesh_program_steps(program,r,operand,steps+(size_t)r*capacity);
  for(int progress=1;progress;){
    progress=0;
    for(uint32_t r=0;r<n;r++)while(cursor[r]<count[r]){
      const struct mesh_step *step=steps+(size_t)r*capacity+cursor[r];
      const double bytes=(double)step->piece.elements*step->piece.element_bytes;
      double link_alpha,link_beta;
      mesh_link_cost(map,r,step->peer,alpha,beta,&link_alpha,&link_beta);
      if(step->op==MESH_STEP_SEND){
        double *port=out_free+(size_t)r*n+step->peer;
        const double start=clock[r]>*port?clock[r]:*port;
        *port=start+bytes*link_beta/1e3;
        arrival[(size_t)r*capacity+cursor[r]++]=*port+link_alpha;
        progress=1;
        continue;
      }
      const uint32_t p=step->peer;
      uint32_t ordinal=0,k=0,seen=0;
      for(uint32_t i=0;i<cursor[r];i++){
        const struct mesh_step *s=steps+(size_t)r*capacity+i;
        if(s->op!=MESH_STEP_SEND && s->peer==p && s->round==step->round)ordinal++;
      }
      for(;k<count[p];k++){
        const struct mesh_step *s=steps+(size_t)p*capacity+k;
        if(s->op==MESH_STEP_SEND && s->peer==r && s->round==step->round && seen++==ordinal)break;
      }
      const struct mesh_step *send=steps+(size_t)p*capacity+k;
      if(k==count[p] || send->first!=step->first || send->piece.elements!=step->piece.elements ||
         send->piece.element_bytes!=step->piece.element_bytes)goto done;
      if(cursor[p]<=k)break;
      double *port=in_free+(size_t)r*n+p;
      double complete=*port+bytes*link_beta/1e3;
      if(arrival[(size_t)p*capacity+k]>complete)complete=arrival[(size_t)p*capacity+k];
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
  free(steps);free(arrival);free(out_free);free(in_free);
  return result;
}

/* Each step becomes one prepared transfer: a section over the typed piece, the peer's channel on
   queue identity%qps, and the same identity on the sending and the receiving rank. */
int mesh_collective_bind(struct mesh_ctx *context,const struct mesh_step *steps,uint32_t count,uint32_t identity,
  struct mesh_section operand,const struct mesh_section *received,uint32_t invocations,uint32_t invocation_pages,
  struct mesh_section *pieces){
  for(uint32_t i=0;i<count;i++)if(steps[i].buffer)return EINVAL;  /* the host transfers bind one operand: no scratch */
  for(uint32_t i=0,reduced=0;i<count;i++){
    const struct mesh_step *step=steps+i;
    struct mesh_section piece;
    if(step->op==MESH_STEP_REDUCE)piece=received[reduced++];
    else{
      int status=mesh_section_slice(context,operand,step->first*step->piece.element_bytes,
        step->piece.elements*step->piece.element_bytes,invocations,invocation_pages,&piece);
      if(status)return status;
    }
    if(pieces)pieces[i]=piece;
    uint32_t transfer=identity+step->round;
    int status=mesh_transfer_bind(context,mesh_peer_channel(context,step->peer,transfer%context->M->qps),
      step->op==MESH_STEP_SEND?MESH_SEND:MESH_RECEIVE,transfer,piece,piece.pages,0,UINT32_MAX,0);
    if(status)return status;
  }
  return 0;
}

/* The word layout and cancellation ranges of mesh_metal_transport_create, without the Metal
   buffers: link p's words follow link p-1's, invocation-major, one per received chunk in the
   frame order mesh_transfers_prepare assigned. */
int mesh_host_inputs(struct mesh_ctx *context){
  struct hdr *m=context->M;
  uint64_t frames[m->links?m->links:1],words=0;
  for(uint32_t p=0;p<m->links;p++){
    struct mesh_tx *tx=mesh_events(m,mesh_notice_queue(m,context->client,p));
    frames[p]=0;
    for(uint32_t q=p*m->qps;q<(p+1)*m->qps;q++){
      struct mesh_transfer *in=mesh_transfers(m,context->client,q,MESH_RECEIVE);
      for(uint32_t i=0;i<atomic_load(mesh_order_length(m,context->client,q,MESH_RECEIVE));i++)
        frames[p]+=(uint64_t)mesh_row_chunks(m,in[i].local_row,in[i].bytes)*in[i].count*mesh_transfer_active(in+i,tx->invocations);
    }
    words+=frames[p];
  }
  struct mesh_section input,stop;
  int status=mesh_section_create(context,sizeof(uint64_t)*(words?words:1),1,0,&input);
  if(!status)status=mesh_section_create(context,sizeof(struct mesh_cancellation)+m->links*sizeof(struct mesh_cancel_range),1,0,&stop);
  if(status)return status;
  uint64_t *word=mesh_section_address(context,input,0);
  struct mesh_cancellation *cancel=mesh_section_address(context,stop,0);
  memset(word,0,input.bytes);memset(cancel,0,stop.bytes);
  for(uint32_t p=0;p<m->links;p++){
    struct mesh_tx *tx=mesh_events(m,mesh_notice_queue(m,context->client,p));
    tx->cancel=(uintptr_t)cancel-(uintptr_t)m;
    cancel->ranges[p]=(struct mesh_cancel_range){.offset=(uintptr_t)word-(uintptr_t)m,.count=frames[p]};
    for(uint32_t q=p*m->qps;q<(p+1)*m->qps;q++){
      struct mesh_transfer *in=mesh_transfers(m,context->client,q,MESH_RECEIVE);
      for(uint32_t i=0;i<atomic_load(mesh_order_length(m,context->client,q,MESH_RECEIVE));i++){
        uint32_t chunks=mesh_row_chunks(m,in[i].local_row,in[i].bytes);
        for(uint32_t s=0;s<in[i].count;s++)for(uint32_t k=0;k<chunks;k++){
          struct mesh_publication *delivery=mesh_publication_at(m,in[i].local_row+s*in[i].stride+k);
          delivery->device_input=(uintptr_t)(word+in[i].first+(uint64_t)s*chunks+k)-(uintptr_t)m;
          delivery->device_stride=sizeof(uint64_t)*(uint64_t)in[i].count*chunks;
        }
      }
    }
    word+=frames[p];
  }
  return 0;
}

/* What the GPU publication kernel stores (M04/M10): each prepared cell's continuation, into the
   cell of invocation t, released after the caller's writes to the section. */
/* design/prepared-machine.md#M30: running invocation T is the program's invocation T mod N in cycle T / N */
static uint32_t program_invocations(struct mesh_ctx *context){
  return ((struct mesh_tx *)mesh_events(context->M,mesh_notice_queue(context->M,context->client,0)))->invocations;
}

void mesh_host_publish(struct mesh_ctx *context,struct mesh_section section,uint32_t invocation){
  uint32_t count=mesh_publication_prepare(context->M,section.first,NULL),n=program_invocations(context);
  struct prepared_publication records[count?count:1];
  mesh_publication_prepare(context->M,section.first,records);
  const uint32_t t=n?invocation%n:invocation,cycle=n?invocation/n:0;
  for(uint32_t i=0;i<count;i++)
    atomic_store_explicit((_Atomic uint64_t *)(uintptr_t)(records[i].destination+(uint64_t)t*sizeof(struct mesh_send)),
      records[i].argument+cycle,memory_order_release);
}

/* The last chunk's word, as mesh_metal_receive_prepare reads it: chunks of one receive land in order. */
uint64_t mesh_host_arrived(struct mesh_ctx *context,struct mesh_section section,uint32_t invocation){
  struct hdr *m=context->M;
  struct mesh_publication *delivery=mesh_publication_at(m,section.first+mesh_row_chunks(m,section.first,section.bytes)-1);
  const uint32_t n=program_invocations(context),t=n?invocation%n:invocation;
  return atomic_load_explicit((_Atomic uint64_t *)((char *)m+delivery->device_input+(uint64_t)t*delivery->device_stride),
    memory_order_acquire);
}
