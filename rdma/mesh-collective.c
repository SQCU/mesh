#include "mesh-collective.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The link map is configuration, read as written: the MLX JACCL hostfile idea
   (github.com/ml-explore/mlx docs/src/usage/distributed.rst) reduced to a kind, a node count and links. */
int mesh_link_map_read(const char *path,struct mesh_link_map *map){
  static const char *const kinds[]={"mesh","ring","tree"};
  FILE *file=fopen(path,"r");
  if(!file)return errno;
  char line[128],kind[8]="";
  uint32_t a,b,capacity=0;
  *map=(struct mesh_link_map){.kind=UINT32_MAX};
  while(fgets(line,sizeof line,file)){
    if(sscanf(line,"%u %u",&a,&b)==2){
      if(map->links==capacity){
        uint32_t (*grown)[2]=realloc(map->link,(size_t)(capacity=capacity?2*capacity:64)*sizeof *map->link);
        if(!grown){fclose(file);mesh_link_map_free(map);return ENOMEM;}
        map->link=grown;
      }
      map->link[map->links][0]=a;map->link[map->links++][1]=b;
    }else if(map->kind==UINT32_MAX && sscanf(line,"%7s %u",kind,&a)==2)
      for(uint32_t k=0;k<3;k++)if(!strcmp(kind,kinds[k])){map->kind=k;map->nodes=a;}
  }
  fclose(file);
  /* Refuse a map its algorithm cannot run, instead of indexing past the planner's arrays or
     walking a broken ring forever: every node < nodes and at most one link into it; a ring is one
     cycle through every node; a tree has nodes-1 links and every node reaches the root. */
  uint32_t n=map->nodes,at,steps;
  if(map->kind==UINT32_MAX || !n)return EINVAL;
  if(map->kind==MESH_LINKS_MESH)return 0;
  uint32_t *up=malloc((size_t)n*sizeof *up);
  if(!up)return ENOMEM;
  int result=0;
  for(uint32_t v=0;v<n;v++)up[v]=n;
  for(uint32_t l=0;l<map->links && !result;l++){
    if(map->link[l][0]>=n || map->link[l][1]>=n || up[map->link[l][1]]<n)result=EINVAL;
    else up[map->link[l][1]]=map->link[l][0];
  }
  if(!result && map->kind==MESH_LINKS_RING){
    at=0;steps=0;
    do{at=up[at];steps++;}while(at<n && at && steps<n);
    result=at==0 && steps==n?0:EINVAL;
  } else if(!result){
    if(map->links!=n-1)result=EINVAL;
    for(uint32_t v=0;v<n && !result;v++){
      for(at=v,steps=0;up[at]<n && steps<n;steps++)at=up[at];
      if(up[at]<n)result=EINVAL;
    }
  }
  free(up);
  return result;
}

void mesh_link_map_free(struct mesh_link_map *map){
  free(map->link);map->link=NULL;map->links=0;
}

struct mesh_link_stated { struct mesh_link_contents file; uint32_t node; };
/* What is stated replaced; every link's up stays as observed (its node's bridge, or its report). */
static void mesh_link_restate(struct mesh_link_contents *c,const void *argument){
  const struct mesh_link_stated *s=argument;
  for(uint32_t a=0;a<MESH_LINK_NODES;a++)for(uint32_t b=0;b<MESH_LINK_NODES;b++){
    struct mesh_link_state next=s->file.link[a][b];
    next.up=c->link[a][b].up;
    c->link[a][b]=next;
  }
  memcpy(c->present,s->file.present,sizeof c->present);
}
int mesh_link_table_state(struct mesh_link_table *table,const char *path){
  FILE *file=fopen(path,"r");
  if(!file)return errno;
  struct mesh_link_stated s={.node=table->node};
  char line[256],word[8];
  unsigned a,b,n;
  float alpha,beta;
  int result=0,fields,counted=0;
  while(!result && fgets(line,sizeof line,file)){
    if(line[0]=='#')continue;
    if((fields=sscanf(line,"%u %u %f %f",&a,&b,&alpha,&beta))>=2){
      if(fields!=4 || a>=MESH_LINK_NODES || b>=MESH_LINK_NODES || a==b)result=EINVAL;
      else s.file.link[a][b]=(struct mesh_link_state){alpha,beta,1,0};
    }else if(sscanf(line,"node %u",&n)==1){
      if(n>=MESH_LINK_NODES)result=EINVAL;
      else s.file.present[n]=1;
    }else if(!counted && sscanf(line,"%7s %u",word,&n)==2 && (!strcmp(word,"mesh") || !strcmp(word,"ring") || !strcmp(word,"tree"))){
      counted=1;
      if(n>MESH_LINK_NODES)result=EINVAL;
      for(uint32_t v=0;v<n && !result;v++)s.file.present[v]=1;
    }
  }
  fclose(file);
  if(!result)mesh_link_table_write(table,mesh_link_restate,&s);
  return result;
}
void mesh_link_table_map(const struct mesh_link_contents *c,const uint32_t *nodes,uint32_t n,struct mesh_link_map *map,
  uint32_t (*pairs)[2],float (*cost)[2]){
  uint32_t links=0,every=1;
  for(uint32_t i=0;i<n;i++)for(uint32_t j=0;j<n;j++){
    const struct mesh_link_state *l=&c->link[nodes[i]][nodes[j]];
    cost[i*n+j][0]=l->alpha;cost[i*n+j][1]=l->beta;
    if(j<=i)continue;
    const struct mesh_link_state *back=&c->link[nodes[j]][nodes[i]];
    if(l->stated && l->up && back->stated && back->up){pairs[links][0]=i;pairs[links++][1]=j;}
    else every=0;
  }
  *map=(struct mesh_link_map){every?MESH_LINKS_MESH:MESH_LINKS_GRAPH,n,links,pairs,(const float (*)[2])cost};
}

/* Whether node `v` contributes to `c`: a broadcast's root; an all-reduce's contributors, every node
   without a set. */
static int mesh_contributes(struct mesh_collective c,uint32_t v){
  if(c.what==MESH_BROADCAST)return v==c.root;
  return !c.contributors || (c.contributors[v/64]>>(v%64)&1);
}

static struct mesh_step mesh_piece(uint32_t op,uint32_t peer,uint32_t round,uint64_t first,uint64_t elements,struct mesh_operand operand){
  operand.elements=elements;
  return (struct mesh_step){.op=op,.peer=peer,.round=round,.first=first,.piece=operand};
}

/* The nodes' segments of the operand, their sizes in node order: a reduce-scatter's or an
   all-gather's `segments` where it gives them, else elements/nodes elements, one more for each of the
   first elements%nodes nodes (the ring's segment sizes).  0 where a segment is empty or they do not
   sum to the operand's elements. */
static int mesh_cut(struct mesh_collective c,uint32_t nodes,uint64_t elements,uint64_t *sizes){
  const uint64_t *given=c.what==MESH_REDUCE_SCATTER || c.what==MESH_ALLGATHER?c.segments:NULL;
  uint64_t sum=0;
  for(uint32_t v=0;v<nodes;v++){
    sizes[v]=given?given[v]:elements/nodes+(v<elements%nodes);
    if(!sizes[v])return 0;
    sum+=sizes[v];
  }
  return sum==elements;
}
/* Node v's segment of the cut `sizes` as one step's piece: after the segments of the nodes before it. */
static struct mesh_step mesh_segment(uint32_t op,uint32_t peer,uint32_t round,uint32_t v,const uint64_t *sizes,struct mesh_operand operand){
  uint64_t first=0;
  for(uint32_t u=0;u<v;u++)first+=sizes[u];
  return mesh_piece(op,peer,round,first,sizes[v],operand);
}

/* The operand as a partial combination crosses: at the collective's accumulator, where it has one. */
static struct mesh_operand mesh_accumulated(struct mesh_operand operand,struct mesh_collective c){
  if(c.accumulator_bytes)operand.element_bytes=c.accumulator_bytes;
  return operand;
}

/* Whether the map links a and b, either way (a mesh links every pair). */
static int mesh_linked(const struct mesh_link_map *map,uint32_t a,uint32_t b){
  if(a==b || a>=map->nodes || b>=map->nodes)return 0;
  if(map->kind==MESH_LINKS_MESH)return 1;
  for(uint32_t l=0;l<map->links;l++)
    if((map->link[l][0]==a && map->link[l][1]==b) || (map->link[l][0]==b && map->link[l][1]==a))return 1;
  return 0;
}

/* The ring a ring all-reduce runs on: a ring map's own cycle, from its first link's first node;
   else rank order 0, 1, ..., nodes-1 where the map links each node to the next (a mesh does).
   0 where the map has neither. */
static int mesh_ring_order(const struct mesh_link_map *map,uint32_t *next,uint32_t *previous,uint32_t *first){
  if(map->kind==MESH_LINKS_RING){
    for(uint32_t l=0;l<map->links;l++){next[map->link[l][0]]=map->link[l][1];previous[map->link[l][1]]=map->link[l][0];}
    *first=map->link[0][0];
    return 1;
  }
  for(uint32_t v=0;v<map->nodes;v++){
    if(!mesh_linked(map,v,(v+1)%map->nodes))return 0;
    next[v]=(v+1)%map->nodes;previous[(v+1)%map->nodes]=v;
  }
  *first=0;
  return 1;
}

/* Ring all-reduce: Patarasuk and Yuan, "Bandwidth optimal all-reduce algorithms for clusters of
   workstations", JPDC 69(2) 2009, the ring reduce-scatter + all-gather (Rabenseifner, ICCS 2004).
   Body transcribed from baidu-research/baidu-allreduce collectives.cu RingAllreduce (Gibiansky 2017):
   segment sizes, recv_from/send_to, and the two loops, with `rank` read as the ring position and
   each MPI call replaced by the step it performs.  Every rank sends and receives 2(size-1)/size of
   the operand; a reduce-scatter piece after the first round is a partial sum, at the accumulator.
   A reduce-scatter or an all-gather is its loop alone, node v's own segment v (mesh_cut: the
   collective's segments, where it gives them).
   None where the map has no ring or the cut has an empty segment (fewer elements than nodes). */
static uint32_t mesh_ring_collective(const struct mesh_link_map *map,uint32_t node,struct mesh_collective c,struct mesh_operand operand,struct mesh_step *out){
  const uint32_t size=map->nodes;
  uint32_t next[size],previous[size],start,rank=0,count=0;
  // Compute the sizes of the chunks, and where each chunk ends.
  uint64_t segment_sizes[size],segment_ends[size];
  if(!mesh_cut(c,size,operand.elements,segment_sizes) || !mesh_ring_order(map,next,previous,&start))return 0;
  for(uint32_t at=start;at!=node;at=next[at])rank++;
  const struct mesh_operand partial=mesh_accumulated(operand,c);
  segment_ends[0]=segment_sizes[0];
  for(uint32_t i=1;i<size;i++)segment_ends[i]=segment_sizes[i]+segment_ends[i-1];
  /* The segment of chunk c: c in an all-reduce; alone, the segment of the node at ring position c-1,
     where the reduce-scatter leaves chunk c reduced and the all-gather starts from it. */
  uint32_t segment[size];
  for(uint32_t i=0,at=start;i<size;i++,at=next[at])segment[(i+1)%size]=c.what==MESH_ALLREDUCE?(i+1)%size:at;
  // Receive from your left neighbor; send to your right neighbor.
  const uint32_t recv_from=previous[node],send_to=next[node];
  // At the i'th iteration, sends segment (rank - i) and receives segment (rank - i - 1).
  for(uint32_t i=0;c.what!=MESH_ALLGATHER && i<size-1;i++){
    uint32_t recv_chunk=segment[(rank-i-1+size)%size],send_chunk=segment[(rank-i+size)%size];
    out[count++]=mesh_piece(MESH_STEP_SEND,send_to,i,segment_ends[send_chunk]-segment_sizes[send_chunk],segment_sizes[send_chunk],i?partial:operand);
    out[count++]=mesh_piece(MESH_STEP_REDUCE,recv_from,i,segment_ends[recv_chunk]-segment_sizes[recv_chunk],segment_sizes[recv_chunk],i?partial:operand);
  }
  // Pipelined ring allgather: at the i'th iteration, sends segment (rank + 1 - i), receives (rank - i).
  for(uint32_t i=0;c.what!=MESH_REDUCE_SCATTER && i<size-1;i++){
    uint32_t send_chunk=segment[(rank-i+1+size)%size],recv_chunk=segment[(rank-i+size)%size];
    out[count++]=mesh_piece(MESH_STEP_SEND,send_to,size-1+i,segment_ends[send_chunk]-segment_sizes[send_chunk],segment_sizes[send_chunk],operand);
    out[count++]=mesh_piece(MESH_STEP_COPY,recv_from,size-1+i,segment_ends[recv_chunk]-segment_sizes[recv_chunk],segment_sizes[recv_chunk],operand);
  }
  return count;
}

/* A spanning tree of the map from `root`, breadth first, each node's children its neighbours not yet
   reached in link order (node order on a mesh): a tree map from its own root is the configured tree,
   a ring from any node its two arms, a mesh from any node a star.  0 where the links do not reach
   every node. */
static int mesh_spanning_tree(const struct mesh_link_map *map,uint32_t root,uint32_t *parent,uint32_t *order){
  const uint32_t n=map->nodes;
  uint32_t reached=1;
  for(uint32_t v=0;v<n;v++)parent[v]=UINT32_MAX;
  parent[root]=root;order[0]=root;
  for(uint32_t head=0;head<reached;head++){
    const uint32_t u=order[head];
    if(map->kind==MESH_LINKS_MESH){
      for(uint32_t v=0;v<n;v++)if(parent[v]==UINT32_MAX){parent[v]=u;order[reached++]=v;}
      continue;
    }
    for(uint32_t l=0;l<map->links;l++)for(uint32_t e=0;e<2;e++){
      const uint32_t a=map->link[l][e],b=map->link[l][1-e];
      if(a==u && b<n && parent[b]==UINT32_MAX){parent[b]=u;order[reached++]=b;}
    }
  }
  return reached==n;
}

/* Tree all-reduce: MPICH MPIR_Reduce_intra_binomial followed by MPIR_Bcast_intra_binomial
   (src/mpi/coll/reduce/reduce_intra_binomial.c, src/mpi/coll/bcast/bcast_intra_binomial.c), with the
   children and parent that MPICH derives from rank bits read from a spanning tree of the map instead
   (mesh_spanning_tree): on a tree map the configured tree, on a star the star.  Reduce: receive and
   reduce each child's whole operand, then send the result to the parent (an interior node's result
   a partial sum, at the accumulator).  Bcast: receive the whole result from the parent, then send it
   to each child; a broadcast is the bcast alone.  On a star the root sends and receives (nodes-1)
   operands and every leaf one each way.  A reduce is the reduce alone.  An all-gather gathers up
   first (MPICH's gather on this tree: each node receives its children's subtrees' segments and sends
   its own subtree's up, segment w at round w) and then broadcasts; a reduce-scatter reduces and then
   scatters (each node receives its own subtree's segments from its parent and sends each child its
   subtree's, segment w at round 1+w), in node order both. */
static int mesh_below(const uint32_t *parent,uint32_t w,uint32_t v){
  for(uint32_t at=w;;at=parent[at]){if(at==v)return 1;if(parent[at]==at)return 0;}
}
static uint32_t mesh_tree_collective(const struct mesh_link_map *map,uint32_t rank,struct mesh_collective c,struct mesh_operand operand,struct mesh_step *out){
  const uint32_t n=map->nodes;
  uint32_t parent[n],order[n],children[n],interior[n],k=0,count=0;
  uint64_t sizes[n];
  memset(interior,0,sizeof interior);
  if(c.root>=n || !mesh_spanning_tree(map,c.root,parent,order))return 0;
  if((c.what==MESH_REDUCE_SCATTER || c.what==MESH_ALLGATHER) && !mesh_cut(c,n,operand.elements,sizes))return 0;
  for(uint32_t i=1;i<n;i++){interior[parent[order[i]]]=1;if(parent[order[i]]==rank)children[k++]=order[i];}
  const struct mesh_operand partial=mesh_accumulated(operand,c);
  const uint32_t down=c.what==MESH_ALLGATHER?n:c.what==MESH_ALLREDUCE;
  if(c.what==MESH_ALLREDUCE || c.what==MESH_REDUCE || c.what==MESH_REDUCE_SCATTER){
    for(uint32_t j=0;j<k;j++)out[count++]=mesh_piece(MESH_STEP_REDUCE,children[j],0,0,operand.elements,interior[children[j]]?partial:operand);
    if(rank!=c.root)out[count++]=mesh_piece(MESH_STEP_SEND,parent[rank],0,0,operand.elements,interior[rank]?partial:operand);
  }
  if(c.what==MESH_ALLGATHER){
    for(uint32_t j=0;j<k;j++)for(uint32_t w=0;w<n;w++)if(mesh_below(parent,w,children[j]))out[count++]=mesh_segment(MESH_STEP_COPY,children[j],w,w,sizes,operand);
    for(uint32_t w=0;rank!=c.root && w<n;w++)if(mesh_below(parent,w,rank))out[count++]=mesh_segment(MESH_STEP_SEND,parent[rank],w,w,sizes,operand);
  }
  if(c.what==MESH_REDUCE_SCATTER){
    for(uint32_t w=0;rank!=c.root && w<n;w++)if(mesh_below(parent,w,rank))out[count++]=mesh_segment(MESH_STEP_COPY,parent[rank],1+w,w,sizes,operand);
    for(uint32_t j=0;j<k;j++)for(uint32_t w=0;w<n;w++)if(mesh_below(parent,w,children[j]))out[count++]=mesh_segment(MESH_STEP_SEND,children[j],1+w,w,sizes,operand);
  }
  if(c.what==MESH_REDUCE || c.what==MESH_REDUCE_SCATTER)return count;
  if(rank!=c.root)out[count++]=mesh_piece(MESH_STEP_COPY,parent[rank],down,0,operand.elements,operand);
  for(uint32_t j=0;j<k;j++)out[count++]=mesh_piece(MESH_STEP_SEND,children[j],down,0,operand.elements,operand);
  return count;
}

/* The children a binomial tree gives relative rank `relrank` of `size`: relrank | mask for each mask
   below its lowest set bit (every mask for the root) within the group. */
static uint32_t mesh_binomial_children(int relrank,int size){
  uint32_t children=0;
  for(int mask=0x1;mask<size && !(mask&relrank);mask<<=1)children+=(relrank|mask)<size;
  return children;
}

/* Binomial all-reduce and broadcast: Thakur, Rabenseifner and Gropp, "Optimization of collective
   communication operations in MPICH", IJHPCA 19(1) 2005, the binomial reduce followed by the
   binomial broadcast from `root`.  Bodies transcribed from MPICH MPIR_Reduce_intra_binomial
   (commutative case: lroot = root) and MPIR_Bcast_intra_binomial, each MPIC_Recv and MPIC_Send
   replaced by the step it performs; a child that received from children of its own sends a partial
   sum, at the accumulator.  ceil(log2 nodes) rounds up and down; the pairs are the rank bits', so a
   map that lacks one of them has none.  A reduce is the binomial reduce alone. */
static uint32_t mesh_binomial_collective(const struct mesh_link_map *map,uint32_t node,struct mesh_collective c,struct mesh_operand operand,struct mesh_step *out){
  const int comm_size=(int)map->nodes,root=(int)c.root,rank=(int)node;
  const struct mesh_operand partial=mesh_accumulated(operand,c);
  const uint32_t down=c.what==MESH_ALLREDUCE;
  uint32_t count=0;
  int mask,source,src,dst;
  if(root>=comm_size || c.what==MESH_REDUCE_SCATTER || c.what==MESH_ALLGATHER)return 0;
  /* every node's plan or none: each relative rank's pair with its parent, its least significant 1
     bit cleared, is a link */
  for(int relative=1;relative<comm_size;relative++)
    if(!mesh_linked(map,(uint32_t)((relative+root)%comm_size),(uint32_t)(((relative&(relative-1))+root)%comm_size)))return 0;
  if(c.what==MESH_ALLREDUCE || c.what==MESH_REDUCE){
    const int lroot=root,relrank=(rank-lroot+comm_size)%comm_size;
    mask=0x1;
    while(mask<comm_size){
      /* Receive */
      if((mask&relrank)==0){
        source=(relrank|mask);
        if(source<comm_size){
          const struct mesh_operand piece=mesh_binomial_children(source,comm_size)?partial:operand;
          source=(source+lroot)%comm_size;
          out[count++]=mesh_piece(MESH_STEP_REDUCE,(uint32_t)source,0,0,operand.elements,piece);
        }
      }else{
        /* I've received all that I'm going to.  Send my result to my parent */
        source=((relrank&(~mask))+lroot)%comm_size;
        out[count++]=mesh_piece(MESH_STEP_SEND,(uint32_t)source,0,0,operand.elements,
          mesh_binomial_children(relrank,comm_size)?partial:operand);
        break;
      }
      mask<<=1;
    }
    if(c.what==MESH_REDUCE)return count;
  }
  const int relative_rank=(rank>=root)?rank-root:rank-root+comm_size;
  /* 1. Wait for arrival of data: the source is the process whose relative rank has the least
     significant 1 bit cleared. */
  mask=0x1;
  while(mask<comm_size){
    if(relative_rank&mask){
      src=rank-mask;
      if(src<0)src+=comm_size;
      out[count++]=mesh_piece(MESH_STEP_COPY,(uint32_t)src,down,0,operand.elements,operand);
      break;
    }
    mask<<=1;
  }
  /* 2. Forward to my subtree: every process with bits set from the LSB up to (not including) mask. */
  mask>>=1;
  while(mask>0){
    if(relative_rank+mask<comm_size){
      dst=rank+mask;
      if(dst>=comm_size)dst-=comm_size;
      out[count++]=mesh_piece(MESH_STEP_SEND,(uint32_t)dst,down,0,operand.elements,operand);
    }
    mask>>=1;
  }
  return count;
}

/* Direct exchange on a full mesh: each contributor sends its whole partial once to every other rank
   and each rank sums what arrives, in any order (Megatron-LM's all-reduce of row-parallel partials,
   Shoeybi et al. 2019, over MPI_Allreduce's semantics); a broadcast's root sends its operand to
   every other rank.  At two nodes this is the pair's existing exchange and has the ring's per-rank
   bytes.  Its SENDs may still be reading the operand when a REDUCE lands, so the sum goes to the
   caller's result operand, as in the pair's existing programs.  A reduce sends to the root alone; a
   reduce-scatter sends each node its segment and an all-gather each node its own. */
static uint32_t mesh_direct_collective(const struct mesh_link_map *map,uint32_t rank,struct mesh_collective c,struct mesh_operand operand,struct mesh_step *out){
  const uint32_t n=map->nodes,segmented=c.what==MESH_REDUCE_SCATTER || c.what==MESH_ALLGATHER;
  uint32_t count=0,any=0;
  uint64_t sizes[n];
  if((c.what==MESH_BROADCAST || c.what==MESH_REDUCE) && c.root>=n)return 0;
  if(segmented && !mesh_cut(c,n,operand.elements,sizes))return 0;
  for(uint32_t v=0;v<n;v++)any|=(uint32_t)mesh_contributes(c,v);
  if(!any)return 0;
  /* every node's plan or none: each contributor linked to every node it sends to */
  for(uint32_t a=0;a<n;a++)for(uint32_t b=0;b<n;b++)
    if(a!=b && mesh_contributes(c,a) && (c.what!=MESH_REDUCE || b==c.root) && !mesh_linked(map,a,b))return 0;
  if(mesh_contributes(c,rank))for(uint32_t peer=0;peer<n;peer++)if(peer!=rank && (c.what!=MESH_REDUCE || peer==c.root))
    out[count++]=segmented?mesh_segment(MESH_STEP_SEND,peer,0,c.what==MESH_ALLGATHER?rank:peer,sizes,operand):
      mesh_piece(MESH_STEP_SEND,peer,0,0,operand.elements,operand);
  for(uint32_t peer=0;peer<n && (c.what!=MESH_REDUCE || rank==c.root);peer++)if(peer!=rank && mesh_contributes(c,peer)){
    const uint32_t op=c.what==MESH_BROADCAST || c.what==MESH_ALLGATHER?MESH_STEP_COPY:MESH_STEP_REDUCE;
    out[count++]=segmented?mesh_segment(op,peer,0,c.what==MESH_ALLGATHER?peer:rank,sizes,operand):
      mesh_piece(op,peer,0,0,operand.elements,operand);
  }
  return count;
}

uint32_t mesh_collective_plan(const struct mesh_link_map *map,uint32_t rank,struct mesh_collective c,struct mesh_operand operand,struct mesh_step *steps){
  if(map->nodes<2 || rank>=map->nodes)return 0;
  switch(c.how){
    case MESH_DIRECT:return mesh_direct_collective(map,rank,c,operand,steps);
    case MESH_RING:return c.what==MESH_ALLREDUCE || c.what==MESH_REDUCE_SCATTER || c.what==MESH_ALLGATHER?
      mesh_ring_collective(map,rank,c,operand,steps):0;
    case MESH_TREE:return mesh_tree_collective(map,rank,c,operand,steps);
    case MESH_BINOMIAL:return mesh_binomial_collective(map,rank,c,operand,steps);
    default:return 0;
  }
}

/* Every node's plan, then its time under the alpha-beta model [Hockney 1994] with a node's sends
   sharing its outgoing port and its receives its incoming one, as Thakur, Rabenseifner and Gropp
   cost these algorithms: each node takes its steps in order; a SEND leaves when the node has
   reached it and its port is free, and arrives alpha after its bytes x beta; a REDUCE or COPY
   waits for its SEND's arrival and its port.  The time is the last node's; negative where a node
   has no plan, a receive has no matching SEND of the same piece (a mismatched schedule), the
   schedule stops (a deadlock).  A step combines any number of receives: a kernel past its
   buffer slots chains its combine (metal-microbench decode_crossings.swift). */
static double mesh_collective_evaluate(const struct mesh_link_map *map,struct mesh_collective c,struct mesh_operand operand,
  double alpha,double beta){
  const uint32_t n=map->nodes,capacity=MESH_COLLECTIVE_STEPS(n);
  if(n<2)return 0;
  struct mesh_step *steps=calloc((size_t)n*capacity,sizeof *steps);
  double *arrival=calloc((size_t)n*capacity,sizeof *arrival),result=-1;
  uint32_t count[n],cursor[n];
  double clock[n],out_free[n],in_free[n];
  memset(cursor,0,sizeof cursor);memset(clock,0,sizeof clock);memset(out_free,0,sizeof out_free);memset(in_free,0,sizeof in_free);
  if(!steps||!arrival)goto done;
  for(uint32_t r=0;r<n;r++){
    if(!(count[r]=mesh_collective_plan(map,r,c,operand,steps+(size_t)r*capacity)))goto done;
  }
  for(int progress=1;progress;){
    progress=0;
    for(uint32_t r=0;r<n;r++)while(cursor[r]<count[r]){
      const struct mesh_step *step=steps+(size_t)r*capacity+cursor[r];
      const double bytes=(double)step->piece.elements*step->piece.element_bytes;
      if(step->op==MESH_STEP_SEND){
        const double start=clock[r]>out_free[r]?clock[r]:out_free[r];
        out_free[r]=start+bytes*(map->cost?map->cost[(size_t)r*n+step->peer][1]:beta)/1e3;
        arrival[(size_t)r*capacity+cursor[r]++]=out_free[r]+(map->cost?map->cost[(size_t)r*n+step->peer][0]:alpha);
        progress=1;
        continue;
      }
      const uint32_t p=step->peer;
      uint32_t k=0;
      while(k<count[p]){
        const struct mesh_step *s=steps+(size_t)p*capacity+k;
        if(s->op==MESH_STEP_SEND && s->peer==r && s->round==step->round && s->first==step->first)break;
        k++;
      }
      const struct mesh_step *send=steps+(size_t)p*capacity+k;
      if(k==count[p] || send->piece.elements!=step->piece.elements || send->piece.element_bytes!=step->piece.element_bytes)goto done;
      if(cursor[p]<=k)break;
      double complete=in_free[r]+bytes*(map->cost?map->cost[(size_t)p*n+r][1]:beta)/1e3;
      if(arrival[(size_t)p*capacity+k]>complete)complete=arrival[(size_t)p*capacity+k];
      in_free[r]=complete;
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
  free(steps);free(arrival);
  return result;
}

double mesh_collective_time(const struct mesh_link_map *map,struct mesh_collective c,struct mesh_operand operand,double alpha,double beta){
  return mesh_collective_evaluate(map,c,operand,alpha,beta);
}

/* The algorithm of least time for this operand on this map among those of the bit set template.how
   (0: every one): the direct exchange, the ring, the spanning tree and the binomial tree, each where
   the map carries it; an all-reduce's, reduce-scatter's and all-gather's trees from
   every root (every node takes a result).  Ties go to the earlier in that order, then the lower
   root. */
struct mesh_collective mesh_collective_choose(const struct mesh_link_map *map,struct mesh_collective template,struct mesh_operand operand,double alpha,double beta){
  const uint32_t allowed=template.how?template.how:(UINT32_C(1)<<MESH_UNAVAILABLE)-1;
  struct mesh_collective best=template;
  best.how=map->nodes<2?MESH_DIRECT:MESH_UNAVAILABLE;
  double least=INFINITY;
  for(uint32_t how=MESH_DIRECT;map->nodes>=2 && how<MESH_UNAVAILABLE;how++)if(allowed>>how&1){
    const int any=(template.what==MESH_ALLREDUCE || template.what==MESH_REDUCE_SCATTER || template.what==MESH_ALLGATHER) &&
      (how==MESH_TREE || how==MESH_BINOMIAL);
    for(uint32_t root=any?0:template.root;root<(any?map->nodes:template.root+1);root++){
      struct mesh_collective c=template;
      c.how=how;c.root=root;
      const double t=mesh_collective_evaluate(map,c,operand,alpha,beta);
      if(t>=0 && t<least){least=t;best=c;}
    }
  }
  return best;
}

/* Each step becomes one prepared transfer: a section over the typed piece, the peer's channel on
   queue identity%qps, and the same identity on the sending and the receiving rank. */
int mesh_collective_bind(struct mesh_ctx *context,const struct mesh_step *steps,uint32_t count,uint32_t identity,
  struct mesh_section operand,const struct mesh_section *received,uint32_t invocations,uint32_t invocation_pages,
  struct mesh_section *pieces){
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
void mesh_host_publish(struct mesh_ctx *context,struct mesh_section section,uint32_t invocation){
  uint32_t count=mesh_publication_prepare(context->M,section.first,NULL);
  struct prepared_publication records[count?count:1];
  mesh_publication_prepare(context->M,section.first,records);
  for(uint32_t i=0;i<count;i++)
    atomic_store_explicit((_Atomic uint64_t *)(uintptr_t)(records[i].destination+(uint64_t)invocation*sizeof(struct mesh_send)),
      records[i].argument,memory_order_release);
}

/* The last chunk's word, as mesh_metal_receive_prepare reads it: chunks of one receive land in order. */
uint64_t mesh_host_arrived(struct mesh_ctx *context,struct mesh_section section,uint32_t invocation){
  struct hdr *m=context->M;
  struct mesh_publication *delivery=mesh_publication_at(m,section.first+mesh_row_chunks(m,section.first,section.bytes)-1);
  return atomic_load_explicit((_Atomic uint64_t *)((char *)m+delivery->device_input+(uint64_t)invocation*delivery->device_stride),
    memory_order_acquire);
}
