#include "mesh-collective.h"
#include "mesh-graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* tag pre-return-20261103 rdma/mesh-collective.c, verbatim: every node computes the same trees, routes and programs
   from the same map, so no product is fused into a sum where one node's compiler might fuse it and another's not */
#pragma STDC FP_CONTRACT OFF

_Static_assert(MESH_NODES_MOST<=64,"a demand's destinations are one 64-bit set of map indices");

/* The link map is configuration, read as written: the MLX JACCL hostfile idea
   (github.com/ml-explore/mlx docs/src/usage/distributed.rst) reduced to a kind, a node count and links. */
/* replaced by mesh_plan (over the bridge's map, mesh_nodes_read); removed at sub-assignment 16 */
int mesh_link_map_read(const char *path,struct mesh_link_map *map){
  static const char *const kinds[]={"mesh","ring","tree"};
  FILE *file=fopen(path,"r");
  if(!file)return errno;
  char line[128],kind[8]="";
  uint32_t a,b;
  *map=(struct mesh_link_map){.kind=UINT32_MAX};
  while(fgets(line,sizeof line,file)){
    if(sscanf(line,"%u %u",&a,&b)==2 && map->links<MESH_LINK_MAP_NODES*(MESH_LINK_MAP_NODES-1)/2){
      map->link[map->links][0]=a;map->link[map->links++][1]=b;
    }else if(map->kind==UINT32_MAX && sscanf(line,"%7s %u",kind,&a)==2)
      for(uint32_t k=0;k<3;k++)if(!strcmp(kind,kinds[k])){map->kind=k;map->nodes=a;}
  }
  fclose(file);
  /* Refuse a map its algorithm cannot run, instead of indexing past the planner's arrays or
     walking a broken ring forever: every node < nodes and at most one link into it; a ring is one
     cycle through every node; a tree has nodes-1 links and every node reaches the root. */
  uint32_t n=map->nodes,up[MESH_LINK_MAP_NODES],at,steps;
  if(map->kind==UINT32_MAX || !n || n>MESH_LINK_MAP_NODES)return EINVAL;
  if(map->kind==MESH_LINKS_MESH)return 0;
  for(uint32_t v=0;v<n;v++)up[v]=n;
  for(uint32_t l=0;l<map->links;l++){
    if(map->link[l][0]>=n || map->link[l][1]>=n || up[map->link[l][1]]<n)return EINVAL;
    up[map->link[l][1]]=map->link[l][0];
  }
  if(map->kind==MESH_LINKS_RING){
    at=0;steps=0;
    do{at=up[at];steps++;}while(at<n && at && steps<n);
    return at==0 && steps==n?0:EINVAL;
  }
  if(map->links!=n-1)return EINVAL;
  for(uint32_t v=0;v<n;v++){
    for(at=v,steps=0;up[at]<n && steps<n;steps++)at=up[at];
    if(up[at]<n)return EINVAL;
  }
  return 0;
}

static struct mesh_step mesh_piece(uint32_t op,uint32_t peer,uint32_t round,uint64_t first,uint64_t elements,struct mesh_operand operand){
  operand.elements=elements;
  return (struct mesh_step){.op=op,.peer=peer,.round=round,.first=first,.piece=operand};
}

/* Ring all-reduce: Patarasuk and Yuan, "Bandwidth optimal all-reduce algorithms for clusters of
   workstations", JPDC 69(2) 2009, the ring reduce-scatter + all-gather (Rabenseifner, ICCS 2004).
   Body transcribed from baidu-research/baidu-allreduce collectives.cu RingAllreduce (Gibiansky 2017):
   segment sizes, recv_from/send_to, and the two loops, with `rank` read as the ring position and
   each MPI call replaced by the step it performs.  Every rank sends and receives 2(size-1)/size of
   the operand. */
static uint32_t mesh_ring_allreduce(const struct mesh_link_map *map,uint32_t node,struct mesh_operand operand,struct mesh_step *out){
  const uint32_t size=map->nodes;
  uint32_t next[MESH_LINK_MAP_NODES],previous[MESH_LINK_MAP_NODES],rank=0,count=0;
  for(uint32_t l=0;l<map->links;l++){next[map->link[l][0]]=map->link[l][1];previous[map->link[l][1]]=map->link[l][0];}
  for(uint32_t at=map->link[0][0];at!=node;at=next[at])rank++;
  // Compute the sizes of the chunks, and where each chunk ends.
  uint64_t segment_sizes[MESH_LINK_MAP_NODES],segment_ends[MESH_LINK_MAP_NODES];
  const uint64_t segment_size=operand.elements/size,residual=operand.elements%size;
  for(uint32_t i=0;i<size;i++)segment_sizes[i]=segment_size+(i<residual);
  segment_ends[0]=segment_sizes[0];
  for(uint32_t i=1;i<size;i++)segment_ends[i]=segment_sizes[i]+segment_ends[i-1];
  // Receive from your left neighbor; send to your right neighbor.
  const uint32_t recv_from=previous[node],send_to=next[node];
  // At the i'th iteration, sends segment (rank - i) and receives segment (rank - i - 1).
  for(uint32_t i=0;i<size-1;i++){
    uint32_t recv_chunk=(rank-i-1+size)%size,send_chunk=(rank-i+size)%size;
    out[count++]=mesh_piece(MESH_STEP_SEND,send_to,i,segment_ends[send_chunk]-segment_sizes[send_chunk],segment_sizes[send_chunk],operand);
    out[count++]=mesh_piece(MESH_STEP_REDUCE,recv_from,i,segment_ends[recv_chunk]-segment_sizes[recv_chunk],segment_sizes[recv_chunk],operand);
  }
  // Pipelined ring allgather: at the i'th iteration, sends segment (rank + 1 - i), receives (rank - i).
  for(uint32_t i=0;i<size-1;i++){
    uint32_t send_chunk=(rank-i+1+size)%size,recv_chunk=(rank-i+size)%size;
    out[count++]=mesh_piece(MESH_STEP_SEND,send_to,size-1+i,segment_ends[send_chunk]-segment_sizes[send_chunk],segment_sizes[send_chunk],operand);
    out[count++]=mesh_piece(MESH_STEP_COPY,recv_from,size-1+i,segment_ends[recv_chunk]-segment_sizes[recv_chunk],segment_sizes[recv_chunk],operand);
  }
  return count;
}

/* Tree all-reduce: MPICH MPIR_Reduce_intra_binomial followed by MPIR_Bcast_intra_binomial
   (src/mpi/coll/reduce/reduce_intra_binomial.c, src/mpi/coll/bcast/bcast_intra_binomial.c), with the
   children and parent that MPICH derives from rank bits read from the configured tree instead.
   Reduce: receive and reduce each child's whole operand, then send the result to the parent.
   Bcast: receive the whole result from the parent, then send it to each child.  On a star the
   root sends and receives (nodes-1) operands and every leaf one each way. */
static uint32_t mesh_tree_allreduce(const struct mesh_link_map *map,uint32_t rank,struct mesh_operand operand,struct mesh_step *out){
  uint32_t count=0;
  for(uint32_t l=0;l<map->links;l++)if(map->link[l][0]==rank)
    out[count++]=mesh_piece(MESH_STEP_REDUCE,map->link[l][1],0,0,operand.elements,operand);
  for(uint32_t l=0;l<map->links;l++)if(map->link[l][1]==rank){
    out[count++]=mesh_piece(MESH_STEP_SEND,map->link[l][0],0,0,operand.elements,operand);
    out[count++]=mesh_piece(MESH_STEP_COPY,map->link[l][0],1,0,operand.elements,operand);
  }
  for(uint32_t l=0;l<map->links;l++)if(map->link[l][0]==rank)
    out[count++]=mesh_piece(MESH_STEP_SEND,map->link[l][1],1,0,operand.elements,operand);
  return count;
}

/* Direct exchange on a full mesh: each rank sends its whole partial once to every other rank and
   sums what arrives, in any order (Megatron-LM's all-reduce of row-parallel partials, Shoeybi et al.
   2019, over MPI_Allreduce's semantics).  At two nodes this is the pair's existing exchange and has
   the ring's per-rank bytes.  Its SENDs may still be reading the operand when a REDUCE lands, so
   the sum goes to the caller's result operand, as in the pair's existing programs. */
static uint32_t mesh_direct_allreduce(const struct mesh_link_map *map,uint32_t rank,struct mesh_operand operand,struct mesh_step *out){
  uint32_t count=0;
  for(uint32_t peer=0;peer<map->nodes;peer++)if(peer!=rank)out[count++]=mesh_piece(MESH_STEP_SEND,peer,0,0,operand.elements,operand);
  for(uint32_t peer=0;peer<map->nodes;peer++)if(peer!=rank)out[count++]=mesh_piece(MESH_STEP_REDUCE,peer,0,0,operand.elements,operand);
  return count;
}

/* replaced by mesh_plan (the steps' transfers declared to it); removed at sub-assignment 16 */
uint32_t mesh_allreduce_plan(const struct mesh_link_map *map,uint32_t rank,struct mesh_operand operand,struct mesh_step *steps){
  return map->kind==MESH_LINKS_RING?mesh_ring_allreduce(map,rank,operand,steps):
    map->kind==MESH_LINKS_TREE?mesh_tree_allreduce(map,rank,operand,steps):mesh_direct_allreduce(map,rank,operand,steps);
}

/* Each step becomes one prepared transfer, bound exactly as the pair's direct push binds its
   partials: a section over the typed piece, the peer's channel on queue identity%qps, and the
   same identity on the sending and the receiving rank. */
/* replaced by mesh_plan (the steps' transfers declared to it); removed at sub-assignment 16 */
int mesh_allreduce_bind(struct mesh_ctx *context,const struct mesh_step *steps,uint32_t count,uint32_t identity,
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
      step->op==MESH_STEP_SEND?MESH_SEND:MESH_RECEIVE,transfer,piece,piece.pages);
    if(status)return status;
  }
  return 0;
}

/* -- a host program's side of the plan's words ------------------------------------------------------------------- */

/* Changed from main's form (sub-assignment 11, design/page-queue-change-request.md), which gave every received chunk of
   the per-link, per-queue transfer lists a completion word and each link a cancellation range: the words are
   mesh_plan's, allocated, zeroed and recorded in each hop page's row (struct mesh_publication), and a cancellation is
   the bridge's store of UINT64_MAX into this node's landed words (mesh.h struct mesh_hop).  What must stay: a host
   program's call between planning and start (mesh.py) succeeds and changes nothing a plan wrote. */
int mesh_host_inputs(struct mesh_ctx *context){
  (void)context;
  return 0;
}

/* What the GPU's publication stores (mesh-metal.m: prepared_publication `argument`, the word's position): each release
   word the range's first row records, execution t's, released after the caller's writes to the range.  Changed from
   main's form (sub-assignment 11), which took mesh_publication_prepare's records, a SEND cell's continuation each: a
   word of the plan is no SEND cell.  What must stay: the first row's words, as mesh_metal_publication_prepare takes a
   range's, at the word's offset + t x hop_stride. */
void mesh_host_publish(struct mesh_ctx *context,struct mesh_section section,uint32_t invocation){
  struct hdr *m=context->M;
  struct mesh_publication *publication=mesh_publication_at(m,section.first);
  const uint64_t argument=atomic_load_explicit(&publication->argument,memory_order_relaxed);
  for(uint32_t i=0;i<publication->sends;i++)for(uint32_t k=0;k<publication->targets[i].count;k++){
    const uint64_t word=publication->targets[i].stream+m->hop_stride*((uint64_t)publication->targets[i].stride*k+invocation);
    atomic_store_explicit((_Atomic uint64_t *)((char *)m+word),argument,memory_order_release);
  }
}

/* Each page's landed word, execution t's: a range's pages spread over every cable and queue pair and land in any
   order, each telling of itself (mesh.h struct mesh_hop).  Changed from main's form (sub-assignment 11), which read
   the last chunk's word alone, one queue pair's chunks landing in order.  What must stay: 0 until the range has
   landed, then 1; UINT64_MAX once a word of it was cancelled. */
uint64_t mesh_host_arrived(struct mesh_ctx *context,struct mesh_section section,uint32_t invocation){
  struct hdr *m=context->M;
  const uint32_t pages=mesh_row_chunks(m,section.first,section.bytes);
  uint64_t arrived=1;
  for(uint32_t k=0;k<pages;k++){
    struct mesh_publication *delivery=mesh_publication_at(m,section.first+k);
    if(!delivery->device_input)return 0;
    const uint64_t value=atomic_load_explicit((_Atomic uint64_t *)((char *)m+delivery->device_input+
      (uint64_t)invocation*delivery->device_stride),memory_order_acquire);
    if(value==UINT64_MAX)return UINT64_MAX;
    if(!value)arrived=0;
  }
  return arrived;
}

/* -- the plan call (mesh-collective.h mesh_plan) ------------------------------------------------------------------ */

/* tag pre-return-20261103 rdma/nccl-mesh.c MESH_SOLVE_EPS: the concurrent-flow solve's epsilon, its rounds and room
   the tag's mesh_solve and mesh_solve_room (rdma/mesh-collective.c) */
#define MESH_PLAN_EPS 0.1
/* every word's position, a release word's and a landed word's: each execution's word is its own and reads 0 as the
   plan is loaded (mesh.h struct mesh_hop), so one value releases or tells every one */
#define MESH_PLAN_POSITION 1

/* a demand: the declared transfers of one identity from one source, one range to each of their destinations */
struct plan_demand {
  uint32_t source,identity,order,first;   /* first: its first declared transfer, the tie after `order` */
  uint32_t mine;                          /* the declared transfer whose section holds this node's pages of it (this
                                             node its source or a destination), else MESH_ABSENT */
  uint32_t tree,trees;                    /* its trees, plan `tree` [tree, tree + trees) */
  uint32_t release;                       /* this node's release word of it, where this node is its source */
  int relayed;                            /* this node receives a page of it and is neither source nor destination */
  uint64_t leaves,bytes,pages;            /* its destinations (a set of map indices), its bytes and pages */
  struct mesh_section relay;              /* this node's relay pages of it, `depth` slots of `pages` */
};
struct plan {
  struct mesh_ctx *context;
  const struct mesh_declared *declared;
  struct plan_demand *demand;             /* the demands, by order and first transfer */
  uint32_t demands,n,self,depth,executions;
  uint64_t serial;
  size_t *tree;                           /* tree[j] the solve's tree, demand by demand (mesh_graph_rows) */
  uint32_t *parent;                       /* the solve's parents, tree by tree */
  uint64_t *run;                          /* run[j]: tree[j]'s pages of its demand */
  uint64_t hops,releases,landed,words;    /* this node's hops a slot, release and landed words; the words' offset */
};

struct plan_key { uint32_t source,identity,index; };
static int plan_key_order(const void *a,const void *b){
  const struct plan_key *x=a,*y=b;
  if(x->source!=y->source)return x->source<y->source?-1:1;
  if(x->identity!=y->identity)return x->identity<y->identity?-1:1;
  return (x->index>y->index)-(x->index<y->index);
}
/* lower order first (a transfer that blocks a pipeline stage), then the declared order */
static int plan_demand_order(const void *a,const void *b){
  const struct plan_demand *x=a,*y=b;
  if(x->order!=y->order)return x->order<y->order?-1:1;
  return (x->first>y->first)-(x->first<y->first);
}
/* mesh_graph_reach's edges: the node graph's pair e, its ends from[e] and to[e] */
struct plan_edges { const uint32_t *from,*to; };
static uint32_t plan_end(const void *context,uint32_t e,uint32_t side){
  const struct plan_edges *graph=context;
  return side?graph->to[e]:graph->from[e];
}
/* mesh_graph_flow's leaves: demand d's destinations */
static int plan_leaf(const void *context,uint32_t d,uint32_t v){
  return (int)(((const struct plan_demand *)context)[d].leaves>>v&1);
}
/* mesh_graph_rows' keys: a solved tree's demand (its trees grouped by demand) */
static uint32_t plan_tree_demand(const void *context,size_t k){ return ((const uint32_t *)context)[k]; }
/* mesh_graph_rows' keys: a vertex's parent on a tree, so a vertex's row its children; the root and a vertex off the
   tree key `n`, past every vertex */
struct plan_tree { const uint32_t *parent; uint32_t n,root; };
static uint32_t plan_tree_parent(const void *context,size_t v){
  const struct plan_tree *tree=context;
  return v==tree->root || tree->parent[v]>=tree->n?tree->n:tree->parent[v];
}
/* mesh_graph_climb's visit: the vertex one step up, a vertex's parent */
static void plan_tree_up(void *consumer,uint32_t at,uint32_t step){ if(step==1)*(uint32_t *)consumer=at; }

/* slot s's page k of a section, its row (mesh-collective.h `section`) */
static uint32_t plan_row(const struct hdr *m,const struct mesh_section *section,uint32_t slot,uint64_t page){
  return section->first+slot*(section->pages/m->block)+(uint32_t)page;
}
/* a section holding `depth` slots of `pages` pages, each a row that starts its block in the registered window: 0, or
   EINVAL (a hop moves a whole page, and every page a plan names is in the window) */
static int plan_pages(struct hdr *m,const struct mesh_section *section,uint64_t pages,uint32_t depth){
  if(section->first==MESH_ABSENT || !section->pages || section->pages%m->block)return EINVAL;
  const uint64_t slot=section->pages/m->block,rows=section->stride?(uint64_t)section->stride*section->count:slot;
  if(slot<pages || (uint64_t)(depth-1)*slot+pages>rows || (uint64_t)section->first+rows>mesh_rows(m))return EINVAL;
  for(uint32_t s=0;s<depth;s++)for(uint64_t k=0;k<pages;k++){
    struct mesh_page_entry *entry=&mesh_page(m)[plan_row(m,section,s,k)];
    const uint64_t mapping=atomic_load_explicit(&entry->mapping,memory_order_relaxed);
    const uint64_t offset=atomic_load_explicit(&entry->address,memory_order_relaxed)-m->data_off;
    if(mapping==MESH_ABSENT || (uint32_t)mapping%m->block || offset!=(uint64_t)(uint32_t)mapping*m->pgsz ||
       !mesh_wired(m,offset,(uint64_t)m->block*m->pgsz))return EINVAL;
  }
  return 0;
}
/* word w's region offset in execution 0: the words in lines of hop_stride bytes (32 words), each line's executions
   hop_stride apart, so execution t's word is at it + t x hop_stride and no two words of any executions meet */
static uint64_t plan_word(const struct plan *p,uint64_t w){
  const uint64_t line=sizeof(struct mesh_send)/sizeof(uint64_t);
  return p->words+(w/line)*p->executions*sizeof(struct mesh_send)+(w%line)*sizeof(uint64_t);
}

/* this node's hop `hop` (of p->hops a slot) in every slot: demand d's page k sent to or received from `node`, its word
   the plan's word w; the hop page's row records the word: a receive's landed word, a source's release word */
static int plan_hop(struct plan *p,const struct plan_demand *d,uint64_t k,uint32_t direction,uint32_t node,uint64_t w,
                    uint64_t hop){
  struct hdr *m=p->context->M;
  const struct mesh_section *section=d->mine!=MESH_ABSENT?&p->declared[d->mine].section:&d->relay;
  const uint64_t word=plan_word(p,w);
  const uint32_t room=(uint32_t)((m->target_stride-offsetof(struct mesh_publication,targets))/sizeof(struct mesh_target));
  for(uint32_t s=0;s<p->depth;s++){
    const uint32_t row=plan_row(m,section,s,k);
    mesh_hops(m)[(size_t)s*p->hops+hop]=(struct mesh_hop){
      .page=(uint32_t)atomic_load_explicit(&mesh_page(m)[row].mapping,memory_order_relaxed),.node=node,
      .direction=direction,.word=word,.position=MESH_PLAN_POSITION,.serial=p->serial};
    struct mesh_publication *publication=mesh_publication_at(m,row);
    if(direction==MESH_RECEIVE){
      /* a landed word has one writer: another receive into this page would be two posted receives on one page */
      if(publication->device_input && publication->device_input!=word)return EINVAL;
      publication->device_input=word;
      publication->device_stride=sizeof(struct mesh_send);
    }else if(d->source==p->self){
      /* a page sent to several nodes, or the source of several ranges: one word each, released by one store */
      uint32_t i=0;
      while(i<publication->sends && publication->targets[i].stream!=word)i++;
      if(i<publication->sends)continue;
      if(i==room)return ENOSPC;
      publication->targets[i]=(struct mesh_target){.stream=word,.count=1,.stride=1};
      publication->sends=i+1;
      atomic_store_explicit(&publication->argument,MESH_PLAN_POSITION,memory_order_relaxed);
    }
  }
  return 0;
}

/* Every page of the plan in one order, the same at every rank: the demands by order and first transfer, a demand's
   pages tree by tree.  A tree carries a page over a node pair once at most, so visiting this node's hops page by page
   in that order, for a pair a -> b, a's sends to b and b's receives from a are the same hops in the same order (mesh.h
   "Both ends").  A page's hops here: its receive from this node's parent on the tree (mesh-graph.h mesh_graph_climb's
   first step), then its sends to this node's children (its row of the tree's vertices keyed by parent,
   mesh_graph_rows), in their order.  Counting (write 0): this node's hops a slot, its landed words and the demands it
   relays; writing: each hop in every slot.  The tree's edges are the graph module's: no walk over a parent table here
   (the C9 reviews of c5bffd0c and c9ffedcf). */
static int plan_walk(struct plan *p,int write){
  const uint32_t n=p->n,self=p->self;
  size_t first[(size_t)n+2],child[(size_t)n+1];
  uint64_t hop=0,landed=0;
  for(uint32_t i=0;i<p->demands;i++){
    struct plan_demand *d=p->demand+i;
    uint64_t k=0;
    for(uint32_t j=d->tree;j<d->tree+d->trees;j++){
      const uint64_t end=k+p->run[j];
      const struct plan_tree tree={p->parent+p->tree[j]*n,n,d->source};
      uint32_t in=MESH_ABSENT;
      mesh_graph_climb(tree.parent,n,d->source,self,plan_tree_up,&in);
      mesh_graph_rows(n+1,n,plan_tree_parent,&tree,first,child);
      if(in==MESH_ABSENT && first[self+1]==first[self]){k=end;continue;}
      for(;k<end;k++){
        /* this node's word of page k: the source's release word, else its receive's landed word, which releases the
           sends it forwards (a relay's, or a destination's on the way to another) */
        const uint64_t w=self==d->source?d->release:p->releases+landed++;
        if(in!=MESH_ABSENT){
          if(write){
            const int status=plan_hop(p,d,k,MESH_RECEIVE,in,w,hop);
            if(status)return status;
          }else if(d->mine==MESH_ABSENT)d->relayed=1;
          hop++;
        }
        for(size_t c=first[self];c<first[self+1];c++,hop++)if(write){
          const int status=plan_hop(p,d,k,MESH_SEND,(uint32_t)child[c],w,hop);
          if(status)return status;
        }
      }
    }
  }
  if(!write){p->hops=hop;p->landed=landed;}
  return 0;
}

/* mesh-collective.h mesh_plan.  The demands: the declared transfers grouped by source and identity, a range to each of
   its destinations, ordered by `order` and then the declared order.  The routing: every demand at once over the map's
   node graph (a node pair an edge where both its records give it cables, of the lesser of their summed capacities;
   no cable, queue pair or link), a destination its source reaches by no path refused (mesh-graph.h mesh_graph_reach),
   by the concurrent-flow solve (mesh_graph_flow, Garg and Koenemann's
   multiplicative weights, the tag's mesh_solve rounds and room), demand d a tree rooted at its source reaching its
   destinations, its bytes down from the root: each demand's trees and their shares, a tree's vertices past its source
   and destinations the relays, members all.  The pages: each demand's pages cut over its trees by their shares by
   earliest finish (mesh_graph_fill), a tree's pages one run of them.  Every rank computes the same demands, trees and
   pages from the same declared transfers and map, and writes its own hops of them (plan_walk). */
int mesh_plan(struct mesh_ctx *context,const struct mesh_declared *declared,uint32_t count,uint32_t executions,uint32_t depth){
  struct hdr *m=context->M;
  if(!executions || (count && !declared))return EINVAL;
  if(!depth || depth>executions)depth=executions;
  struct mesh_nodes nodes,after;
  int status=mesh_nodes_read(context,&nodes);
  if(status)return status;
  const uint32_t n=nodes.count;
  if(nodes.self>=n)return EHOSTUNREACH;
  const uint64_t block=(uint64_t)m->block*m->pgsz;
  struct plan p={.context=context,.declared=declared,.n=n,.self=nodes.self,.depth=depth,.executions=executions,
                 .serial=nodes.serial};
  struct plan_key *key=malloc(((size_t)count+1)*sizeof *key);
  uint32_t *root=NULL,*demand=NULL;
  size_t *trees=NULL;
  double *up=NULL,*down=NULL,*share=NULL,*rate=NULL,*latency=NULL;
  p.demand=calloc((size_t)count+1,sizeof *p.demand);
  if(!key || !p.demand){status=ENOMEM;goto done;}
  for(uint32_t i=0;i<count;i++){
    const struct mesh_declared *t=declared+i;
    if(t->source>=n || t->destination>=n){status=EHOSTUNREACH;goto done;}
    /* a transfer to its own source crosses no cable, and nothing is copied */
    if(t->source==t->destination || !t->section.bytes){status=EINVAL;goto done;}
    key[i]=(struct plan_key){t->source,t->identity,i};
  }
  qsort(key,count,sizeof *key,plan_key_order);
  for(uint32_t i=0;i<count;i++){
    const struct mesh_declared *t=declared+key[i].index;
    if(!i || key[i].source!=key[i-1].source || key[i].identity!=key[i-1].identity)
      p.demand[p.demands++]=(struct plan_demand){.source=t->source,.identity=t->identity,.order=t->order,
        .first=key[i].index,.mine=MESH_ABSENT,.bytes=t->section.bytes,.pages=(t->section.bytes+block-1)/block,
        .relay={.first=MESH_ABSENT}};
    struct plan_demand *d=p.demand+p.demands-1;
    /* one range: its bytes alike, each destination once, the source's pages one section */
    if(t->section.bytes!=d->bytes || d->leaves>>t->destination&1){status=EINVAL;goto done;}
    d->leaves|=UINT64_C(1)<<t->destination;
    if(t->order<d->order)d->order=t->order;
    if(t->destination==p.self)d->mine=key[i].index;
    if(t->source==p.self){
      const struct mesh_section *a=&declared[d->first].section,*b=&t->section;
      if(a->first!=b->first || a->pages!=b->pages || a->count!=b->count || a->stride!=b->stride){status=EINVAL;goto done;}
      d->mine=d->first;
    }
  }
  qsort(p.demand,p.demands,sizeof *p.demand,plan_demand_order);
  for(uint32_t i=0;i<p.demands;i++)if(p.demand[i].source==p.self)p.demand[i].release=(uint32_t)p.releases++;

  if(p.demands){
    uint32_t from[n*(n-1)/2+1],to[n*(n-1)/2+1],E=0;
    double capacity[n*(n-1)/2+1];
    for(uint32_t a=0;a<n;a++)for(uint32_t b=a+1;b<n;b++){
      const uint64_t c=nodes.capacity[a][b]<nodes.capacity[b][a]?nodes.capacity[a][b]:nodes.capacity[b][a];
      if(c){from[E]=a;to[E]=b;capacity[E]=(double)c;E++;}
    }
    /* a destination its source reaches by no Thunderbolt-5 path within the map: refused (mesh_graph_reach, once from
       each source) */
    const struct plan_edges graph={from,to};
    uint32_t order[n];
    unsigned char reached[n];
    uint64_t reach[n],known=0;
    for(uint32_t i=0;i<p.demands;i++){
      const uint32_t s=p.demand[i].source;
      if(!(known>>s&1)){
        mesh_graph_reach(n,E,plan_end,&graph,s,order,reached);
        reach[s]=0;
        for(uint32_t v=0;v<n;v++)if(reached[v])reach[s]|=UINT64_C(1)<<v;
        known|=UINT64_C(1)<<s;
      }
      if(p.demand[i].leaves&~reach[s]){status=EHOSTUNREACH;goto done;}
    }
    const uint32_t rounds=(uint32_t)(mesh_graph_log(2*E>1?(double)(2*E):2.0)/(MESH_PLAN_EPS*MESH_PLAN_EPS))+1;
    const uint64_t most=(uint64_t)rounds*p.demands;
    const uint32_t room=most>=UINT32_MAX?UINT32_MAX-1:(uint32_t)most;
    root=malloc((size_t)p.demands*sizeof *root);
    up=calloc(p.demands,sizeof *up);
    down=malloc((size_t)p.demands*sizeof *down);
    demand=malloc((size_t)room*sizeof *demand);
    p.parent=malloc((size_t)room*n*sizeof *p.parent);
    share=malloc((size_t)room*sizeof *share);
    if(!root || !up || !down || !demand || !p.parent || !share){status=ENOMEM;goto done;}
    for(uint32_t i=0;i<p.demands;i++){root[i]=p.demand[i].source;down[i]=(double)(p.demand[i].pages*block);}
    /* UINT32_MAX: its own allocations failed (its eps is positive, its roots members, every leaf reached and its room
       the most trees its rounds can find) */
    const uint32_t made=mesh_graph_flow(n,E,from,to,capacity,p.demands,root,up,down,plan_leaf,p.demand,MESH_PLAN_EPS,
                                        rounds,room,demand,p.parent,share);
    if(made==UINT32_MAX){status=ENOMEM;goto done;}
    p.tree=malloc(((size_t)made+1)*sizeof *p.tree);
    p.run=calloc((size_t)made+1,sizeof *p.run);
    rate=malloc(((size_t)made+1)*sizeof *rate);
    latency=calloc((size_t)made+1,sizeof *latency);
    trees=malloc(((size_t)p.demands+1)*sizeof *trees);
    if(!p.tree || !p.run || !rate || !latency || !trees){status=ENOMEM;goto done;}
    /* the solved trees grouped by demand, in the order the solve found them */
    mesh_graph_rows(p.demands,made,plan_tree_demand,demand,trees,p.tree);
    for(uint32_t i=0;i<p.demands;i++){p.demand[i].tree=(uint32_t)trees[i];p.demand[i].trees=(uint32_t)(trees[i+1]-trees[i]);}
    for(uint32_t j=0;j<made;j++)rate[j]=share[p.tree[j]];
    for(uint32_t i=0;i<p.demands;i++){
      struct plan_demand *d=p.demand+i;
      /* a tree's rate its share of the demand's flow; no tree is known to start later than another (latency 0) */
      mesh_graph_fill(d->trees,latency+d->tree,rate+d->tree,d->pages,p.run+d->tree);
      uint64_t cut=0;
      for(uint32_t j=d->tree;j<d->tree+d->trees;j++)cut+=p.run[j];
      if(cut!=d->pages){status=EHOSTUNREACH;goto done;}
    }
  }

  plan_walk(&p,0);
  if((uint64_t)depth*p.hops>m->hop_capacity){status=ENOSPC;goto done;}
  for(uint32_t i=0;i<p.demands;i++){
    struct plan_demand *d=p.demand+i;
    if(d->mine!=MESH_ABSENT && (status=plan_pages(m,&declared[d->mine].section,d->pages,depth)))goto done;
  }
  /* this node's relay pages, in the registered window, and the plan's words, outside it, zeroed */
  for(uint32_t i=0;i<p.demands;i++){
    struct plan_demand *d=p.demand+i;
    if(!d->relayed)continue;
    if((status=mesh_section_create(context,d->pages*block,depth,1,&d->relay)))goto done;
    if((status=plan_pages(m,&d->relay,d->pages,depth)))goto done;
  }
  const uint64_t line=sizeof(struct mesh_send)/sizeof(uint64_t),total=p.releases+p.landed;
  struct mesh_section words;
  if((status=mesh_section_create(context,(total?(total+line-1)/line:1)*executions*sizeof(struct mesh_send),1,0,&words)))goto done;
  void *address=mesh_section_address(context,words,0);
  memset(address,0,words.bytes);
  p.words=(uintptr_t)address-(uintptr_t)m;
  /* mesh-metal.m's publication buffer: the plan's words, the GPU's release stores and its waits on landed words */
  context->send_off=p.words;
  context->send_bytes=(uint64_t)words.pages*m->pgsz;

  /* mesh.h's order of writing a plan: `hops` 0, the entries, executions, depth, hop_stride and plan, then `hops` */
  atomic_store_explicit(&m->hops,0,memory_order_release);
  for(uint32_t i=0;i<p.demands;i++){
    const struct plan_demand *d=p.demand+i;
    if(d->mine==MESH_ABSENT)continue;
    for(uint32_t s=0;s<depth;s++)for(uint64_t k=0;k<d->pages;k++){
      struct mesh_publication *publication=mesh_publication_at(m,plan_row(m,&declared[d->mine].section,s,k));
      publication->sends=0;publication->device_input=publication->device_stride=0;
      atomic_store_explicit(&publication->argument,0,memory_order_relaxed);
    }
  }
  if((status=plan_walk(&p,1)))goto done;
  m->executions=executions;
  atomic_store_explicit(&m->depth,depth,memory_order_relaxed);
  m->hop_stride=sizeof(struct mesh_send);
  /* `plan` the plan's generation, its previous value + 1 and never 0, and the map serial the plan was made against its
     entries' `serial` alone, so a second plan on an unchanged map is told from the first (the coordinator's change to
     CONTRACT.md's interface and mesh.h's comment above struct mesh_hop; 17612857 had `plan` the map serial).  What
     must stay: the bridge refuses a plan whose entries' serial is not `map_serial` as it loads it */
  const uint64_t generation=atomic_load_explicit(&m->plan,memory_order_relaxed)+1;
  atomic_store_explicit(&m->plan,generation?generation:1,memory_order_relaxed);
  if(!(status=mesh_nodes_read(context,&after)) && after.serial!=p.serial)status=ESTALE;
  if(status)goto done;
  atomic_store_explicit(&m->hops,(uint64_t)depth*p.hops,memory_order_release);
  /* the bridge's progress threads watch `plan` with bounded waits (CONTRACT.md "Wave-2 interfaces") */
  os_sync_wake_by_address_all(&m->plan,sizeof m->plan,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
done:
  free(key);free(p.demand);free(root);free(up);free(down);free(demand);free(p.parent);free(share);
  free(p.tree);free(p.run);free(rate);free(latency);free(trees);
  return status;
}

