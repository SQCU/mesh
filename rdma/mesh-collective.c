#include "mesh-collective.h"
#include "mesh-graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* tag pre-return-20261103 rdma/mesh-collective.c, verbatim: every node computes the same trees, routes and programs
   from the same map, so no product is fused into a sum where one node's compiler might fuse it and another's not */
#pragma STDC FP_CONTRACT OFF

_Static_assert(MESH_NODES_MOST<=64,"a demand's destinations are one 64-bit set of map indices");

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
    /* 2026-11-03, sub-assignment 3: MESH_WORD_STRIDE in place of hdr `hop_stride`, vacant at 108 (mesh.h), so the file
       compiles; sub-assignment 5 writes this body */
    const uint64_t word=publication->targets[i].stream+
                        MESH_WORD_STRIDE*((uint64_t)publication->targets[i].stride*k+invocation);
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
  /* its part of the dependence graph's template (plan_template) and its value here: */
  uint64_t on;                            /* the nodes its trees reach past its source (a set of map indices), the
                                             trees the fill gave pages alone: each has its W and L */
  uint32_t forwards;                      /* its distinct tree edges y -> c over those trees, each its F */
  uint32_t vertex;                        /* its first template vertex, P(d); the rest follow (plan_landed) */
  uint32_t value;                         /* its value at this node, p->value[value]; MESH_ABSENT where this node
                                             holds none of its pages */
};
/* A value at this node (design/disjoint-pages-change-request.md, Appendix A, A3, "Its inputs";
   design/disjoint-pages-coordination.md, "The allocator"): a demand whose pages this node holds, its instance (v, t)
   those pages in execution t, t < executions, `blocks` pages each. */
struct plan_value {
  uint32_t demand;                        /* its demand, p->demand[demand] */
  uint32_t declared;                      /* the declared transfer whose section's rows name it, its demand's `mine`
                                             (this node the demand's source or a destination): instance t at its rows
                                             first + t x stride + k, k < blocks; MESH_ABSENT for a relay's own pages,
                                             which no row names */
  uint64_t blocks;                        /* its pages an instance, blocks of block x pgsz bytes (its demand's
                                             `pages`) */
  uint32_t write;                         /* its write's template vertex: P(d) at the source, W(d, self) elsewhere */
  uint32_t read,reads;                    /* its reads' template vertices, p->read[read .. read + reads): F(d, self, c)
                                             for each c this node forwards it to, ascending, then C(d, self) where
                                             this node is a destination; at least one */
};
struct plan {
  struct mesh_ctx *context;
  const struct mesh_declared *declared;
  struct plan_demand *demand;             /* the demands, by order and first transfer */
  uint32_t demands,n,self,executions;
  uint32_t depth;                         /* the old tail's slots, `executions` (sub-assignment 5 replaces that tail) */
  uint64_t serial;
  uint32_t *of;                           /* of[i]: declared[i]'s demand, p->demand[of[i]] */
  size_t *tree;                           /* tree[j] the solve's tree, demand by demand (mesh_graph_rows) */
  uint32_t *parent;                       /* the solve's parents, tree by tree */
  uint64_t *run;                          /* run[j]: tree[j]'s pages of its demand */
  uint64_t hops,releases,landed,words;    /* this node's hops a slot, release and landed words; the words' offset */
  /* The values at this node and the plan's dependence graph (design/disjoint-pages-change-request.md, Appendix A, A3,
     "Its inputs"), the allocator's inputs (plan_allocate).  The graph is periodic: a template, one copy an execution,
     each edge carrying the execution distance `lag` from its tail's execution to its head's (0 within one).  It is the
     whole collective's, every rank's the same from the same declared transfers, dependences and map; template vertex
     (v, t) is v in execution t (mesh-graph.h mesh_graph_happens_before). */
  uint64_t *child;                        /* child[i x n + y]: the nodes y sends demand i's pages to over its trees (a
                                             set of map indices; 0 for a node off them) */
  struct plan_value *value;               /* the values at this node, in the plan's demand order */
  uint32_t values;
  uint32_t *read;                         /* the values' read vertices (struct plan_value `read`, `reads`) */
  uint32_t vertices;                      /* the template's vertices, demand by demand from each one's `vertex` */
  size_t edges;                           /* the template's edges */
  size_t *first;                          /* vertex v's edges, in compressed rows: [first[v], first[v + 1]),
                                             vertices + 1 entries (mesh_graph_rows by tail) */
  uint32_t *head,*lag;                    /* edge k's head vertex, and its execution distance (the head's execution is
                                             the tail's + lag) */
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

/* A declared section this node holds (mesh-collective.h `section`): a mesh_section_create(bytes, executions, 1)
   section, `count` == executions, P pages an instance (`pages` == P x block, `stride` >= P), every instance's row
   unbound (MESH_ABSENT) or bound by an earlier plan to a block inside the program's window (mesh-dataflow.h struct
   mesh_ctx `window`).  0, or EINVAL.  Replaces plan_pages' `depth` slots for the declared sections
   (design/disjoint-pages-change-request.md, Appendix A, A4). */
static int plan_section(struct mesh_ctx *context,const struct mesh_section *section,uint64_t pages,uint32_t executions){
  struct hdr *m=context->M;
  const struct mesh_range window=context->window;
  if(section->first==MESH_ABSENT || section->count!=executions || section->pages!=pages*m->block ||
     section->stride<pages || (uint64_t)section->first+(uint64_t)(executions-1)*section->stride+pages>mesh_rows(m))
    return EINVAL;
  for(uint32_t t=0;t<executions;t++)for(uint64_t k=0;k<pages;k++){
    struct mesh_page_entry *entry=&mesh_page(m)[section->first+(uint64_t)t*section->stride+k];
    const uint32_t page=(uint32_t)atomic_load_explicit(&entry->mapping,memory_order_relaxed);
    if(page==MESH_ABSENT)continue;
    const uint64_t offset=atomic_load_explicit(&entry->address,memory_order_relaxed)-m->data_off;
    if(page%m->block || page<window.first || (uint64_t)page+m->block>(uint64_t)window.first+window.count ||
       offset!=(uint64_t)page*m->pgsz)return EINVAL;
  }
  return 0;
}

/* -- the dependence graph's template (design/disjoint-pages-coordination.md, "The allocator") --------------------- */

/* v's place among a set's members (map indices below 64) */
static uint32_t plan_rank(uint64_t set,uint32_t v){ return (uint32_t)__builtin_popcountll(set&((UINT64_C(1)<<v)-1)); }
/* Demand i's template vertices, from its `vertex`: P(i), its production at its source (an edge into it is before its
   first write, an edge out of it after its last); for each node y of `on` in ascending order, W(i, y) its first
   landing at y and L(i, y) its last; for each tree edge y -> c in (y, c) order, F(i, y, c), y's last forward to c; for
   each destination z in ascending order, C(i, z), z's consumption. */
static uint32_t plan_produced(const struct plan *p,uint32_t i){ return p->demand[i].vertex; }
static uint32_t plan_landed(const struct plan *p,uint32_t i,uint32_t y,int last){
  return p->demand[i].vertex+1+2*plan_rank(p->demand[i].on,y)+(uint32_t)!!last;
}
static uint32_t plan_forwarded(const struct plan *p,uint32_t i,uint32_t y,uint32_t c){
  const uint64_t *child=p->child+(size_t)i*p->n;
  uint32_t at=p->demand[i].vertex+1+2*(uint32_t)__builtin_popcountll(p->demand[i].on);
  for(uint32_t x=0;x<y;x++)at+=(uint32_t)__builtin_popcountll(child[x]);
  return at+plan_rank(child[y],c);
}
static uint32_t plan_consumed(const struct plan *p,uint32_t i,uint32_t z){
  const struct plan_demand *d=p->demand+i;
  return d->vertex+1+2*(uint32_t)__builtin_popcountll(d->on)+d->forwards+plan_rank(d->leaves,z);
}
/* a template edge before its row is known (mesh_graph_rows' records, keyed by tail) */
struct plan_edge { uint32_t tail,head,lag; };
static uint32_t plan_edge_tail(const void *context,size_t k){ return ((const struct plan_edge *)context)[k].tail; }

/* The plan's dependence graph and the values at this node (design/disjoint-pages-change-request.md, Appendix A, A3,
   "Its inputs"; design/disjoint-pages-coordination.md, "The allocator"), after the routing.
   - Each demand's trees, the ones the fill gave pages (a tree of none carries nothing): each vertex's children on
     them, from its row of the tree's vertices keyed by parent (mesh_graph_rows, as plan_walk reads them; no walk over
     a parent table here, C9), make `child` and `on`.
   - The template's edges, all of lag 0 but the program's: P -> W(y) for every y of `on`; P -> F(source, c); W(y) ->
     L(y); W(y) -> F(y, c); F(y, c) -> L(c); L(z) -> C(z); and each dependence e, C(of[consumed], destination of
     declared[consumed]) -> P(of[produced]) with lag e.lag.  Every edge is sound: its tail happens before its head
     whatever the cables do (a range is produced before it is sent; a page lands before it is forwarded, is sent
     before it lands, and lands before it is consumed).  Grouped into rows by tail (mesh_graph_rows), so every rank
     numbers them alike.
   - The values: each demand whose pages this node holds, as its source, a destination or a relay (on its trees, no
     destination), in the plan's demand order; its write P at the source, W(self) elsewhere; its reads its forwards
     from here and, at a destination, its consumption.  A tree's every leaf is a destination (mesh_graph_flow joins
     each leaf to its root), so a relay forwards and every value has a read.
   0, or ENOMEM. */
static int plan_template(struct plan *p,const struct mesh_dependence *dependence,uint32_t dependences){
  const uint32_t n=p->n,self=p->self;
  size_t first[(size_t)n+2],child[(size_t)n+1];
  p->child=calloc((size_t)p->demands*n+1,sizeof *p->child);
  p->value=malloc(((size_t)p->demands+1)*sizeof *p->value);
  p->read=malloc(((size_t)p->demands*(n+1)+1)*sizeof *p->read);
  if(!p->child || !p->value || !p->read)return ENOMEM;
  uint64_t vertices=0,edges=dependences;
  for(uint32_t i=0;i<p->demands;i++){
    struct plan_demand *d=p->demand+i;
    uint64_t *below=p->child+(size_t)i*n;
    for(uint32_t j=d->tree;j<d->tree+d->trees;j++){
      if(!p->run[j])continue;
      const struct plan_tree tree={p->parent+p->tree[j]*n,n,d->source};
      mesh_graph_rows(n+1,n,plan_tree_parent,&tree,first,child);
      for(uint32_t y=0;y<n;y++)for(size_t c=first[y];c<first[y+1];c++)below[y]|=UINT64_C(1)<<child[c];
    }
    d->on=0;d->forwards=0;
    for(uint32_t y=0;y<n;y++){d->on|=below[y];d->forwards+=(uint32_t)__builtin_popcountll(below[y]);}
    const uint64_t on=(uint64_t)__builtin_popcountll(d->on),leaves=(uint64_t)__builtin_popcountll(d->leaves);
    d->vertex=(uint32_t)vertices;
    vertices+=1+2*on+d->forwards+leaves;
    edges+=2*on+2*(uint64_t)d->forwards+leaves;
    if(vertices>=UINT32_MAX)return ENOMEM;
  }
  p->vertices=(uint32_t)vertices;
  p->edges=(size_t)edges;
  struct plan_edge *edge=malloc((p->edges+1)*sizeof *edge);
  size_t *order=malloc((p->edges+1)*sizeof *order);
  p->first=malloc(((size_t)p->vertices+1)*sizeof *p->first);
  p->head=malloc((p->edges+1)*sizeof *p->head);
  p->lag=malloc((p->edges+1)*sizeof *p->lag);
  if(!edge || !order || !p->first || !p->head || !p->lag){free(edge);free(order);return ENOMEM;}
  size_t e=0;
  for(uint32_t i=0;i<p->demands;i++){
    const struct plan_demand *d=p->demand+i;
    const uint64_t *below=p->child+(size_t)i*n;
    const uint32_t produced=plan_produced(p,i);
    for(uint64_t s=d->on;s;s&=s-1){
      const uint32_t y=(uint32_t)__builtin_ctzll(s);
      edge[e++]=(struct plan_edge){produced,plan_landed(p,i,y,0),0};
      edge[e++]=(struct plan_edge){plan_landed(p,i,y,0),plan_landed(p,i,y,1),0};
    }
    for(uint32_t y=0;y<n;y++)for(uint64_t s=below[y];s;s&=s-1){
      const uint32_t c=(uint32_t)__builtin_ctzll(s),forward=plan_forwarded(p,i,y,c);
      edge[e++]=(struct plan_edge){y==d->source?produced:plan_landed(p,i,y,0),forward,0};
      edge[e++]=(struct plan_edge){forward,plan_landed(p,i,c,1),0};
    }
    for(uint64_t s=d->leaves;s;s&=s-1){
      const uint32_t z=(uint32_t)__builtin_ctzll(s);
      edge[e++]=(struct plan_edge){plan_landed(p,i,z,1),plan_consumed(p,i,z),0};
    }
  }
  for(uint32_t k=0;k<dependences;k++){
    const struct mesh_dependence *x=dependence+k;
    edge[e++]=(struct plan_edge){plan_consumed(p,p->of[x->consumed],p->declared[x->consumed].destination),
                                 plan_produced(p,p->of[x->produced]),x->lag};
  }
  mesh_graph_rows(p->vertices,p->edges,plan_edge_tail,edge,p->first,order);
  for(size_t k=0;k<p->edges;k++){p->head[k]=edge[order[k]].head;p->lag[k]=edge[order[k]].lag;}
  free(edge);free(order);
  /* the values at this node */
  uint32_t reads=0;
  p->values=0;
  for(uint32_t i=0;i<p->demands;i++){
    struct plan_demand *d=p->demand+i;
    d->value=MESH_ABSENT;
    if(d->source!=self && !(d->on>>self&1))continue;
    struct plan_value *v=p->value+p->values;
    *v=(struct plan_value){.demand=i,.declared=d->mine,.blocks=d->pages,
      .write=d->source==self?plan_produced(p,i):plan_landed(p,i,self,0),.read=reads};
    for(uint64_t s=p->child[(size_t)i*n+self];s;s&=s-1)
      p->read[reads++]=plan_forwarded(p,i,self,(uint32_t)__builtin_ctzll(s));
    if(d->leaves>>self&1)p->read[reads++]=plan_consumed(p,i,self);
    v->reads=reads-v->read;
    d->value=p->values++;
  }
  return 0;
}

/* The one allocator (design/disjoint-pages-change-request.md, Appendix A, A3 and Appendix B;
   design/disjoint-pages-coordination.md, "The allocator"): every instance (v, t) of p->value, t < executions, its pages
   inside the program's window, by happens-before over p's template.  Called once a plan, after the template and
   before anything is erased or written.  Sub-assignment 4 writes it: until then it assigns nothing. */
static int plan_allocate(struct plan *p){
  (void)p;
  return 0;
}

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
/* A plan's own section (its relay pages, its words) released by the region's closed path, as mesh-dataflow.c
   mesh_retire releases a retired client's rows: each row of it this client owns closed where it heads a buffer, and
   given up (mesh_rows_release).  The bridge's mesh_retired_release frees closed rows only while no port's `prepared`
   holds a plan, since a port's outstanding requests may still name their pages.  The allocator's reclaim
   (mesh_reclaim) then takes their pages back.  So the pages stay held while a port may still post on them, and are
   freed within a liveness turn of the bridge (MESH_PAUSE_NS) once none does.  Added for review R1 round 2 (P1):
   c5bffd0c's plan call released none of its sections. */
static void plan_release(struct mesh_ctx *context,struct mesh_section section){
  struct hdr *m=context->M;
  if(section.first==MESH_ABSENT)return;
  const uint32_t rows=section.stride?section.count*section.stride:1;
  for(uint32_t r=section.first;r<section.first+rows;r++){
    struct mesh_buffer *buffer=&mesh_buffers(m)[r];
    if(atomic_load_explicit(&buffer->owner,memory_order_acquire)==context->client && buffer->pages)
      atomic_store_explicit(&buffer->closed,1,memory_order_release);
  }
  /* 2026-11-03, sub-assignment 3: mesh_rows_release is gone with the call-site allocators (mesh-dataflow.h,
     sub-assignment 7), so the file compiles; sub-assignment 5 removes the relay and words sections this releases */
}
/* The sections of the plan this process last wrote (its relay pages and its words), with the region and client that
   own them, so the plan that replaces it releases them (plan_release).  Kept here because struct mesh_ctx
   (mesh-dataflow.h) has no field for them.  One record: the plan of another region or client is not this one's to
   release, and a retired client's rows were closed by its retirement (mesh_retire). */
static struct plan_held { struct hdr *m; uint64_t client; struct mesh_section relay,words; } plan_held={
  .relay={.first=MESH_ABSENT},.words={.first=MESH_ABSENT}};

/* word w's region offset in execution 0: each word of each execution its own hop_stride-byte cell (mesh.h struct
   mesh_hop: the word at offset 0, a landed word's page's sent count at offset 8), word w's cells its executions'
   in turn, so execution t's word is at it + t x hop_stride and no two words' cells overlap.  Changed from c5bffd0c's
   32 words to a 256-byte line (which put a neighbouring word at a landed word's offset 8), by mesh.h 90962999's word cells; what
   must stay: one cell a word and execution */
static uint64_t plan_word(const struct plan *p,uint64_t w){
  return p->words+w*p->executions*sizeof(struct mesh_send);
}

/* this node's hop `hop` (of p->hops a slot) in every slot: demand d's page k sent to or received from `node`, its word
   the plan's word w, `forwards` as mesh.h gives it (a receive's: this slot's sends naming its landed word; a send's: 1
   where its word is a landed word here, a forwarding SEND); the hop page's row records the word: a receive's landed
   word, a source's release word */
static int plan_hop(struct plan *p,const struct plan_demand *d,uint64_t k,uint32_t direction,uint32_t node,uint64_t w,
                    uint32_t forwards,uint64_t hop){
  struct hdr *m=p->context->M;
  /* 2026-11-03, sub-assignment 3: struct mesh_hop's `forwards` is gone and `position` is its `mask` at 108 (mesh.h), so
     the file compiles; sub-assignment 5 replaces the hop writing */
  (void)forwards;
  const struct mesh_section *section=d->mine!=MESH_ABSENT?&p->declared[d->mine].section:&d->relay;
  const uint64_t word=plan_word(p,w);
  const uint32_t room=(uint32_t)((m->target_stride-offsetof(struct mesh_publication,targets))/sizeof(struct mesh_target));
  for(uint32_t s=0;s<p->depth;s++){
    const uint32_t row=plan_row(m,section,s,k);
    mesh_hops(m)[(size_t)s*p->hops+hop]=(struct mesh_hop){
      .page=(uint32_t)atomic_load_explicit(&mesh_page(m)[row].mapping,memory_order_relaxed),.node=node,
      .direction=direction,.word=word,.mask=MESH_PLAN_POSITION,.serial=p->serial};
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
        /* the page's forwards: its sends here, each naming the receive's landed word (none at the source) */
        const uint32_t forwards=(uint32_t)(first[self+1]-first[self]),forwarding=self!=d->source;
        if(in!=MESH_ABSENT){
          if(write){
            const int status=plan_hop(p,d,k,MESH_RECEIVE,in,w,forwards,hop);
            if(status)return status;
          }else if(d->mine==MESH_ABSENT)d->relayed=1;
          hop++;
        }
        for(size_t c=first[self];c<first[self+1];c++,hop++)if(write){
          const int status=plan_hop(p,d,k,MESH_SEND,(uint32_t)child[c],w,forwarding,hop);
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
   pages from the same declared transfers and map, and writes its own hops of them (plan_walk).
   2026-11-03, sub-assignment 3 (design/disjoint-pages-change-request.md, "Claude's proposal: the canonical path" item
   1, Appendix A, A3, and Appendix B): the executions are unrolled, not ringed, so `depth` left the call.  Each
   dependence is checked as it arrives and each section this node holds before the routing (plan_section, in place of
   plan_pages' slots); after the routing, the values at this node and the plan's dependence graph (plan_template), then
   their pages, once, by the one allocator (plan_allocate), all before anything is erased or written, so a failure
   leaves the previous plan standing.  What must stay: the demands, mesh_graph_reach, mesh_graph_flow and
   mesh_graph_fill as they were, so every rank's trees and template are the same. */
int mesh_plan(struct mesh_ctx *context,const struct mesh_declared *declared,uint32_t count,uint32_t executions,
              const struct mesh_dependence *dependence,uint32_t dependences){
  struct hdr *m=context->M;
  if(!executions || (count && !declared) || (dependences && !dependence))return EINVAL;
  /* a program's data edge joins a range consumed at a node to a range that node produces (mesh-collective.h struct
     mesh_dependence) */
  for(uint32_t e=0;e<dependences;e++)
    if(dependence[e].consumed>=count || dependence[e].produced>=count ||
       declared[dependence[e].consumed].destination!=declared[dependence[e].produced].source)return EINVAL;
  /* the old tail's slots, one an execution: sub-assignment 5 replaces that tail (no depth: design/disjoint-pages-
     change-request.md, Appendix A, A4) */
  const uint32_t depth=executions;
  struct mesh_nodes nodes,after;
  int status=mesh_nodes_read(context,&nodes);
  if(status)return status;
  const uint32_t n=nodes.count;
  if(nodes.self>=n)return EHOSTUNREACH;
  const uint64_t block=(uint64_t)m->block*m->pgsz;
  struct plan p={.context=context,.declared=declared,.n=n,.self=nodes.self,.depth=depth,.executions=executions,
                 .serial=nodes.serial};
  /* this call's sections, released where it fails (plan_release) */
  struct mesh_section relay={.first=MESH_ABSENT},words={.first=MESH_ABSENT};
  struct plan_key *key=malloc(((size_t)count+1)*sizeof *key);
  uint32_t *root=NULL,*demand=NULL,*placed=malloc(((size_t)count+1)*sizeof *placed);
  size_t *trees=NULL;
  double *up=NULL,*down=NULL,*share=NULL,*rate=NULL,*latency=NULL;
  p.demand=calloc((size_t)count+1,sizeof *p.demand);
  p.of=malloc(((size_t)count+1)*sizeof *p.of);
  if(!key || !placed || !p.demand || !p.of){status=ENOMEM;goto done;}
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
    p.of[key[i].index]=p.demands-1;
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
  /* each declared transfer's demand in the plan's order: a demand's place before the sort is its first transfer's */
  for(uint32_t i=0;i<p.demands;i++)placed[p.of[p.demand[i].first]]=i;
  for(uint32_t i=0;i<count;i++)p.of[i]=placed[p.of[i]];
  for(uint32_t i=0;i<p.demands;i++)if(p.demand[i].source==p.self)p.demand[i].release=(uint32_t)p.releases++;
  /* this node's sections of the ranges it is the source or a destination of, each instance's rows (plan_section) */
  for(uint32_t i=0;i<p.demands;i++){
    const struct plan_demand *d=p.demand+i;
    if(d->mine!=MESH_ABSENT && (status=plan_section(context,&declared[d->mine].section,d->pages,executions)))goto done;
  }

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
  /* the values at this node and the dependence graph (plan_template), then their pages (plan_allocate), before
     anything is erased or written */
  if((status=plan_template(&p,dependence,dependences)) || (status=plan_allocate(&p)))goto done;

  plan_walk(&p,0);
  if((uint64_t)depth*p.hops>m->hop_capacity){status=ENOSPC;goto done;}
  for(uint32_t i=0;i<p.demands;i++){
    struct plan_demand *d=p.demand+i;
    if(d->mine!=MESH_ABSENT && (status=plan_pages(m,&declared[d->mine].section,d->pages,depth)))goto done;
  }
  /* this node's relay pages, in the registered window: one section for the plan, its `depth` slots each every relayed
     demand's pages in turn, so a plan holds two sections (these and its words) to release */
  uint64_t relays=0;
  for(uint32_t i=0;i<p.demands;i++)if(p.demand[i].relayed)relays+=p.demand[i].pages;
  if(relays){
    if((status=mesh_section_create(context,relays*block,depth,1,&relay)))goto done;
    if((status=plan_pages(m,&relay,relays,depth)))goto done;
    /* demand d's relay pages: a view of the section from its first page, slot s's rows `relays` apart (plan_row) */
    for(uint32_t i=0,at=0;i<p.demands;i++){
      struct plan_demand *d=p.demand+i;
      if(!d->relayed)continue;
      d->relay=(struct mesh_section){relay.first+at,relay.pages,d->bytes,depth,relay.stride};
      at+=(uint32_t)d->pages;
    }
  }
  /* the plan's words, outside the window, zeroed: a cell a word and execution (plan_word), at least one word's */
  const uint64_t total=p.releases+p.landed;
  if((status=mesh_section_create(context,(total?total:1)*executions*sizeof(struct mesh_send),1,0,&words)))goto done;
  void *address=mesh_section_address(context,words,0);
  memset(address,0,words.bytes);
  p.words=(uintptr_t)address-(uintptr_t)m;

  /* mesh.h's order of writing a plan: `hops` 0, the entries, executions, depth, hop_stride and plan, then `hops`.  The
     release fence keeps the entries' writes after the store of 0 (review R1 round 2, S2: a release store orders what
     precedes it, not what follows), so a bridge that read `hops` and `plan`, then the entries, then them again after
     an acquire fence (mesh-flow.c, b3cc9990) sees a change wherever it read a torn list */
  atomic_store_explicit(&m->hops,0,memory_order_release);
  atomic_thread_fence(memory_order_release);
  /* the previous plan is gone: its sections released (plan_release keeps them held while a port may still post on
     them), and its words no longer named for the GPU */
  if(plan_held.m==m && plan_held.client==context->client){
    plan_release(context,plan_held.relay);
    plan_release(context,plan_held.words);
  }
  plan_held=(struct plan_held){.relay={.first=MESH_ABSENT},.words={.first=MESH_ABSENT}};
  /* 2026-11-03, sub-assignment 3: struct mesh_ctx's send_off and send_bytes are gone (mesh-dataflow.h, sub-assignment
     7), so they are no longer cleared or set here; sub-assignment 5 writes the words' areas */
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
  /* this plan's sections, the next plan's to release; mesh-metal.m's publication buffer its words, the GPU's release
     stores and its waits on landed words */
  plan_held=(struct plan_held){m,context->client,relay,words};
done:
  /* a failed call's own sections released */
  if(status){plan_release(context,relay);plan_release(context,words);}
  free(key);free(placed);free(p.demand);free(p.of);
  free(root);free(up);free(down);free(demand);free(p.parent);free(share);
  free(p.tree);free(p.run);free(rate);free(latency);free(trees);
  free(p.child);free(p.value);free(p.read);free(p.first);free(p.head);free(p.lag);
  return status;
}

