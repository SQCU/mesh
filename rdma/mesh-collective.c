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
   mesh_plan's, zeroed by it where it uses them and recorded in instance 0's rows (struct mesh_publication), and a
   cancellation is UINT64_MAX in an execution's cancel word (mesh.h struct mesh_hop, "Cancellation").  What must stay:
   a host program's call between planning and start (mesh.py) succeeds and changes nothing a plan wrote. */
int mesh_host_inputs(struct mesh_ctx *context){
  (void)context;
  return 0;
}

/* What the GPU's publication stores (mesh-metal.m mesh_metal_publication_prepare: prepared_publication `argument`),
   from the host: MESH_SUPPLIED stored (release) into each supplied word the range's first row names (its `targets`,
   which mesh_plan writes), execution t's at `stream` + t x MESH_WORD_STRIDE (a target's word k at `stream` +
   MESH_WORD_STRIDE x `stride` x k, as mesh-dataflow.c mesh_publication_prepare reads it), after the caller's writes to
   the range: one store releases every send of the range's pages (design/disjoint-pages-change-request.md, "Claude's
   proposal: the canonical path", item 3).  A row no plan made a range's source names no target, and nothing is stored.
   2026-11-03, sub-assignment 5: reversing the word at its offset + t x hop_stride (a 256-byte cell a word and
   execution, sub-assignment 11's form) and the row's `argument` as the value stored.  Why: the cells are gone (mesh.h,
   108); a supplied word's executions are MESH_WORD_STRIDE apart (mesh_supplied), and MESH_SUPPLIED is the one value a
   producer stores.  What must stay: the first row's words, as mesh_metal_publication_prepare takes a range's. */
void mesh_host_publish(struct mesh_ctx *context,struct mesh_section section,uint32_t invocation){
  struct hdr *m=context->M;
  const struct mesh_publication *publication=mesh_publication_at(m,section.first);
  for(uint32_t i=0;i<publication->sends;i++)for(uint32_t k=0;k<publication->targets[i].count;k++){
    const uint64_t word=publication->targets[i].stream+
                        MESH_WORD_STRIDE*((uint64_t)publication->targets[i].stride*k+invocation);
    atomic_store_explicit((_Atomic uint64_t *)((char *)m+word),MESH_SUPPLIED,memory_order_release);
  }
}

/* Execution t (`invocation`) of a declared range at a destination, from execution t's landed record (mesh.h struct
   mesh_hop, "The landed list"; design/disjoint-pages-change-request.md, "Claude's proposal: the canonical path", item
   2): the record the range's first row names (`device_input`, execution 0's, + t x `device_stride`) and the range's
   bit run [first, first + P), `first` that row's `landed` and P the range's pages (mesh_plan writes instance 0's rows
   so, the run's bits consecutive): UINT64_MAX where the record's cancel word is, 1 once every bit of the run is set,
   else 0; 0 where the row has no receive (`device_input` 0).  The cancel word and the run's words are read (acquire),
   a few words of one record, not a word a page.
   2026-11-03, sub-assignment 5: reversing a landed word a page, every page's row read (sub-assignment 11's form).  Why:
   the per-page words are gone; a landing sets one bit of its execution's record (the change request, "The turns").
   What must stay: 0 until the range has landed, then 1; UINT64_MAX once its execution was cancelled. */
uint64_t mesh_host_arrived(struct mesh_ctx *context,struct mesh_section section,uint32_t invocation){
  struct hdr *m=context->M;
  const struct mesh_publication *head=mesh_publication_at(m,section.first);
  if(!head->device_input)return 0;
  const uint64_t quantum=(uint64_t)m->block*m->pgsz;
  const uint32_t first=head->landed,pages=(uint32_t)((section.bytes+quantum-1)/quantum);
  _Atomic uint64_t *record=(_Atomic uint64_t *)((char *)m+head->device_input+(uint64_t)invocation*head->device_stride);
  if(atomic_load_explicit(record,memory_order_acquire)==UINT64_MAX)return UINT64_MAX;
  for(uint32_t w=first/64;pages && w<=(first+pages-1)/64;w++){
    const uint64_t mask=mesh_word_mask(first,pages,w);
    if((atomic_load_explicit(record+1+w,memory_order_acquire)&mask)!=mask)return 0;
  }
  return 1;
}

/* -- the plan call (mesh-collective.h mesh_plan) ------------------------------------------------------------------ */

/* tag pre-return-20261103 rdma/nccl-mesh.c MESH_SOLVE_EPS: the concurrent-flow solve's epsilon, its rounds and room
   the tag's mesh_solve and mesh_solve_room (rdma/mesh-collective.c) */
#define MESH_PLAN_EPS 0.1

/* a demand: the declared transfers of one identity from one source, one range to each of their destinations */
struct plan_demand {
  uint32_t source,identity,order,first;   /* first: its first declared transfer, the tie after `order` */
  uint32_t mine;                          /* the declared transfer whose section holds this node's pages of it (this
                                             node its source or a destination), else MESH_ABSENT */
  uint32_t tree,trees;                    /* its trees, plan `tree` [tree, tree + trees) */
  uint32_t release;                       /* this node's release of it (mesh.h mesh_supplied's `release`), where this
                                             node is its source */
  uint64_t leaves,bytes,pages;            /* its destinations (a set of map indices), its bytes and pages */
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
/* One of this node's hops of an execution (plan_walk; mesh.h struct mesh_hop), every execution's the same: value
   `value`'s page `page` (of its instance in that execution, plan_page), sent to or received from `node`.  Its word in
   execution t: where `bit` is a bit (a receive's, its execution's bit-th receive in list order; a forwarding send's,
   its receive's of the same page), the word of execution t's landed record holding it, its `mask` 1 << (bit % 64);
   where `bit` is MESH_ABSENT (a send from the range's source), release `release`'s supplied word of execution t, its
   `mask` MESH_SUPPLIED (design/disjoint-pages-coordination.md, "Hop entries"). */
struct plan_entry { uint32_t value,node,direction,bit; uint64_t page,release; };
struct plan {
  struct mesh_ctx *context;
  const struct mesh_declared *declared;
  struct plan_demand *demand;             /* the demands, by order and first transfer */
  uint32_t demands,n,self,executions;
  uint64_t serial;
  uint32_t *of;                           /* of[i]: declared[i]'s demand, p->demand[of[i]] */
  size_t *tree;                           /* tree[j] the solve's tree, demand by demand (mesh_graph_rows) */
  uint32_t *parent;                       /* the solve's parents, tree by tree */
  uint64_t *run;                          /* run[j]: tree[j]'s pages of its demand */
  uint64_t releases;                      /* this node's releases, the demands it is the source of */
  /* this node's hops of one execution, every execution's in this order (plan_walk): */
  struct plan_entry *entry;               /* entry[0 .. entries), NULL while plan_walk counts */
  uint64_t entries,receives;              /* their count (n of mesh.h's executions x n) and their receives (R) */
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
  /* the allocator's output (plan_allocate) */
  uint64_t *colour;                       /* instance (v, t)'s first block in the window: colour[v x executions + t],
                                             its page k arena page window.first + (colour + k) x block */
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
   (design/disjoint-pages-change-request.md, Appendix A, A4).
   2026-11-03, sub-assignment 5 (review 12's finding 6): its rows are checked to lie among the claim's section rows,
   [rows.first + 2, rows.first + rows.count) (mesh-dataflow.h struct mesh_ctx `rows`), where they were checked against
   the whole page table (mesh_rows).  Why: the plan binds them (mesh_backing_bind), so a row of another claim, or one of
   this claim's two head rows, would be overwritten.  What must stay: the plan binds no row outside its claim's
   sections. */
static int plan_section(struct mesh_ctx *context,const struct mesh_section *section,uint64_t pages,uint32_t executions){
  struct hdr *m=context->M;
  const struct mesh_range window=context->window,rows=context->rows;
  if(section->first==MESH_ABSENT || section->count!=executions || section->pages!=pages*m->block ||
     section->stride<pages || rows.count<2 || section->first<(uint64_t)rows.first+2 ||
     (uint64_t)section->first+(uint64_t)(executions-1)*section->stride+pages>(uint64_t)rows.first+rows.count)
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
/* The sections this node holds, each one demand's alone.  A value is one demand's pages at this node (struct
   plan_value; design/disjoint-pages-coordination.md, "The allocator"), so a section two demands here named (a range
   received and sent on under another identity, or sent under two) would be two values, each given pages of its own by
   the allocator, and its rows could be bound to one of them only: the other's landings or sends would name pages the
   program never reads or writes.  Refused, EINVAL, where two held sections' rows [first, first + (executions - 1) x
   stride + P) overlap.  Added 2026-11-03, sub-assignment 5: plan_hop refused the receive half of this as it wrote (a
   second receive into a page's landed word) and let one source row carry several ranges' words (a target each), which
   the slots allowed and the allocator's values do not.  0; EINVAL; ENOMEM */
struct plan_span { uint64_t first,end; };
static int plan_span_order(const void *a,const void *b){
  const struct plan_span *x=a,*y=b;
  return (x->first>y->first)-(x->first<y->first);
}
static int plan_disjoint(const struct plan *p){
  struct plan_span *span=malloc(((size_t)p->demands+1)*sizeof *span);
  if(!span)return ENOMEM;
  uint32_t spans=0;
  for(uint32_t i=0;i<p->demands;i++){
    const struct plan_demand *d=p->demand+i;
    if(d->mine==MESH_ABSENT)continue;
    const struct mesh_section *s=&p->declared[d->mine].section;
    span[spans++]=(struct plan_span){s->first,s->first+(uint64_t)(p->executions-1)*s->stride+d->pages};
  }
  qsort(span,spans,sizeof *span,plan_span_order);
  int status=0;
  for(uint32_t j=1;j<spans && !status;j++)if(span[j-1].end>span[j].first)status=EINVAL;
  free(span);
  return status;
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

/* -- the one allocator (design/disjoint-pages-change-request.md, Appendix A, A3, and Appendix B) ------------------- */

/* the items' order (design/disjoint-pages-coordination.md, "The allocator"): a value's blocks, largest first, then its
   place in the plan's demand order; each value's executions follow in turn */
struct plan_sized { uint64_t blocks; uint32_t value; };
static int plan_sized_order(const void *a,const void *b){
  const struct plan_sized *x=a,*y=b;
  if(x->blocks!=y->blocks)return x->blocks>y->blocks?-1:1;
  return (x->value>y->value)-(x->value<y->value);
}
/* mesh_graph_happens_before's edges: the template's, edge k's head and execution distance */
static uint32_t plan_head(const void *context,size_t k){ return ((const struct plan *)context)->head[k]; }
static uint32_t plan_lag(const void *context,size_t k){ return ((const struct plan *)context)->lag[k]; }
/* mesh_graph_happens_before's table at this node: its sources this node's reads p->read[0 .. reads), its targets the
   values' writes (value x's target x), `words` words a pair */
struct plan_reach { const struct plan *p; const uint64_t *reach; size_t words; };
/* every read of value v in execution t happens before value w's write in execution t + distance: a path of one or
   more edges from each read to the write, the table's bit at that distance */
static int plan_before(const struct plan_reach *table,uint32_t v,uint32_t w,uint32_t distance){
  const struct plan_value *x=table->p->value+v;
  for(uint32_t j=x->read;j<x->read+x->reads;j++)
    if(!(table->reach[((size_t)j*table->p->values+w)*table->words+distance/64]>>(distance%64)&1))return 0;
  return 1;
}
/* mesh_graph_left_edge's interference: item v x executions + t is instance (v, t).  A = (v, t) and B = (w, u), t <=
   u, interfere unless every read of A happens before B's write at distance u - t or, in one execution (t == u), every
   read of B before A's write; a path never reaches an earlier execution (no lag is negative), so for t < u B's reads
   are never before A's write (design/disjoint-pages-change-request.md, Appendix A, A3, "Interference, by
   happens-before"; design/disjoint-pages-coordination.md, "The allocator") */
static int plan_interferes(const void *context,uint32_t a,uint32_t b){
  const struct plan_reach *table=context;
  const uint32_t executions=table->p->executions;
  uint32_t v=a/executions,t=a-v*executions,w=b/executions,u=b-w*executions;
  if(t>u){const uint32_t x=v,s=t;v=w;t=u;w=x;u=s;}
  return !plan_before(table,v,w,u-t) && (t!=u || !plan_before(table,w,v,0));
}

/* The one allocator (design/disjoint-pages-change-request.md, Appendix A, A3, "What it is", "Interference, by
   happens-before", "The assignment" and "Its failure", and Appendix B; design/disjoint-pages-coordination.md, "The
   allocator"): every instance (v, t) of p->value, t < executions, its pages inside the program's window, decided
   once a plan, after the template and before anything is erased or written.  2026-11-03, sub-assignment 4: the body
   owner 3 left empty.
   - The items: item v x executions + t is instance (v, t), its value's `blocks` blocks, ordered by blocks (largest
     first), then by value in the plan's demand order (p->value's), then by execution.
   - Happens-before over the template unrolled across the executions (mesh-graph.h mesh_graph_happens_before), its
     sources this node's reads and its targets the values' writes.  Each template vertex is one demand's and each
     demand at most one value here, so p->read names no vertex twice and no two values share a write: the index tables
     are p->read and the values' writes as they are.  A lag-0 cycle in the declared dependences (EDEADLK) is
     mesh_plan's EINVAL.
   - Two instances share a block only where the table orders every read of one before the other's write
     (plan_interferes): the program's own edges then order the last read before the next write, so nothing is
     counted, checked or waited on while the plan runs; no other reuse.
   - The colours are blocks of the program's window (mesh-dataflow.h struct mesh_ctx `window`), window.count / block
     of them: the left-edge colouring, greedy by size (mesh-graph.h mesh_graph_left_edge), instance (v, t) at blocks
     [colour, colour + blocks), its page k arena page window.first + (colour + k) x block.  Where the table holds no
     bit (no read at this node happens before any write at this node, at any distance: no program edge reaches back,
     as where there are no dependences), every pair interferes: the colouring is given no interference test
     (`interferes` NULL), and places each instance at the end of the one before it in that order, without pairwise
     tests.  A value's executions are then one run, instance t at its first + t x blocks.
     2026-11-03, sub-assignment 5: that placement, which sub-assignment 4 computed inline here, is mesh_graph_left_edge's
     own (cc08dc6a), so the colouring's result is the graph module's in both cases (design/rubrics.md C9; the C9 hook's
     remark on 38162b8d).  What must stay: the items' order, and no pairwise test where no bit is set.
   - No room in the window (the colouring's ENOSPC) is ENOMEM, before anything runs: no fallback, nothing allocated
     at run time.
   Cost: mesh_graph_happens_before's O(V^2 + E) order, O(executions x (V + E) x ceil(reads / 64)) word operations and
   a table of reads x values x ceil(executions / 64) words; one pass over the table; then, with no bit, one pass over
   the items (mesh_graph_left_edge's NULL form), else its walk, O(items^2) interference tests at worst, each over A's
   reads, a table word each.  e2b's plan (values ~ 100, executions up to 2048, no dependences, so no bit): 204,800 items placed in
   one pass, a table of ~ 10^6 words and ~ 10^8 word operations, inside a second.  A plan of that size whose program
   edges reach back, so that blocks are reused, bounds the walk at ~ 2 x 10^10 tests, past ~8 s.
   0; EINVAL; ENOMEM. */
static int plan_allocate(struct plan *p){
  const uint32_t values=p->values,executions=p->executions;
  if(!values)return 0;
  const uint64_t capacity=p->context->window.count/p->context->M->block;
  const uint32_t reads=p->value[values-1].read+p->value[values-1].reads;
  const size_t words=((size_t)executions+63)/64;
  size_t pairs,cells;
  if((uint64_t)values*executions>=UINT32_MAX || __builtin_mul_overflow((size_t)reads,(size_t)values,&pairs) ||
     __builtin_mul_overflow(pairs,words,&cells) || cells>=SIZE_MAX/sizeof(uint64_t))return ENOMEM;
  const uint32_t items=values*executions;
  struct plan_sized *sized=malloc((size_t)values*sizeof *sized);
  uint32_t *write=malloc((size_t)values*sizeof *write),*order=NULL;
  uint64_t *reach=calloc(cells,sizeof *reach),*size=NULL,used=0;
  p->colour=malloc((size_t)items*sizeof *p->colour);
  int status=!sized || !write || !reach || !p->colour?ENOMEM:0;
  if(!status){
    for(uint32_t x=0;x<values;x++){write[x]=p->value[x].write;sized[x]=(struct plan_sized){p->value[x].blocks,x};}
    qsort(sized,values,sizeof *sized,plan_sized_order);
    status=mesh_graph_happens_before(p->vertices,p->first,plan_head,plan_lag,p,executions,reads,p->read,values,write,
                                     reach);
    if(status==EDEADLK)status=EINVAL;
  }
  uint64_t any=0;
  for(size_t c=0;!status && c<cells && !any;c++)any=reach[c];
  if(!status){
    order=malloc((size_t)items*sizeof *order);
    size=malloc((size_t)items*sizeof *size);
    if(!order || !size)status=ENOMEM;
    for(uint32_t s=0,at=0;!status && s<values;s++)for(uint32_t t=0;t<executions;t++){
      const uint32_t item=sized[s].value*executions+t;
      order[at++]=item;
      size[item]=sized[s].blocks;
    }
    /* no bit: every pair interferes, the colouring's NULL form */
    const struct plan_reach table={p,reach,words};
    if(!status)status=mesh_graph_left_edge(items,size,order,any?plan_interferes:NULL,&table,capacity,p->colour,&used);
    if(status==ENOSPC)status=ENOMEM;
  }
  free(sized);free(write);free(reach);free(order);free(size);
  return status;
}

/* -- this node's hops (design/disjoint-pages-change-request.md, "Claude's proposal: the canonical path", items 1-3) -- */

/* 2026-11-03, sub-assignment 5 (design/disjoint-pages-change-request.md, "Claude's proposal: the canonical path", item
   1; Appendix A, A4: "`plan_pages`' `depth` slots and `plan_row`'s slot arithmetic"; "Scope": "`mesh_plan`'s slots and
   per-slot relay pages"): removed here, with the hops written slot by slot (plan_hop), the plan's own sections and
   their release (plan_release, plan_held) and the word cells (plan_word, MESH_PLAN_POSITION):
   - plan_row and plan_pages: a section as `depth` slots of its pages, execution t on slot t mod depth's rows, every
     row bound before the plan;
   - the relay section (`depth` slots of every relayed demand's pages) and the words section (a 256-byte cell a word
     and execution), each mesh_section_create'd by every plan, released by the next (plan_release closing the rows
     this client owned) and recorded process-wide (plan_held).
   Why: the ring reused `depth` copies of a page under waits between executions, which nothing asked for (the change
   request, "The turns"); the one allocator gives every instance its own pages before the first execution
   (plan_allocate), a relay's a value like any other, and the plan's words are areas of the layout (mesh.h hdr
   `word_off`, `landed_off`) that the next plan reuses once no port holds this one (mesh-call.h mesh_plan_erased).
   What must stay: mesh_plan creates no section and takes no row; it closes no row (only mesh_retire closes one, the
   claim's head row: mesh-dataflow.c) and keys nothing on a section row's owner, which is 0 (mesh-call.c
   mesh_section_create). */

/* instance (v, t)'s page k: arena page window.first + (colour + k) x block (plan_allocate; mesh-dataflow.h struct
   mesh_ctx `window`), the first page of a block inside the registered window */
static uint32_t plan_page(const struct plan *p,uint32_t v,uint32_t t,uint64_t k){
  return (uint32_t)(p->context->window.first+(p->colour[(size_t)v*p->executions+t]+k)*p->context->M->block);
}

/* Every page of the plan in one order, the same at every rank: the demands by order and first transfer, a demand's
   pages tree by tree.  A tree carries a page over a node pair once at most, so visiting this node's hops page by page
   in that order, for a pair a -> b, a's sends to b and b's receives from a are the same hops in the same order (mesh.h
   "Both ends").  A page's hops here: its receive from this node's parent on the tree (mesh-graph.h mesh_graph_climb's
   first step), then its sends to this node's children (its row of the tree's vertices keyed by parent,
   mesh_graph_rows), in their order.  The tree's edges are the graph module's: no walk over a parent table here (the
   C9 reviews of c5bffd0c and c9ffedcf).  A page's children are its own tree's, which p->child (every tree's union,
   plan_template) does not tell, so the rows are made here again.
   This is one execution's list; every execution's is the same hops in the same order (mesh.h, "Executions"; mesh_plan
   writes execution t's at [t n, t n + n)).  Counting (p->entry NULL) gives p->entries (n) and p->receives (R);
   filling writes p->entry[0 .. n).  Each receive takes the execution's next bit, so a destination's pages of one
   range, consecutive receives here (its demand's pages in turn, no other demand's receive between them), are one run
   of bits; a send from the range's source names its release, any other send its receive's bit (a relay's forward, or
   a destination's on the way on).
   2026-11-03, sub-assignment 5 (design/disjoint-pages-change-request.md, "Claude's proposal: the canonical path",
   items 1-3): reversing the walk's writing of each hop in each of `depth` slots (a slot's row's page, a 256-byte word
   cell a hop, a relay's pages its own section's: plan_hop) and its counts of landed words and relayed demands.  Why:
   the slots and cells are gone; each execution's pages are its instances' (plan_allocate) and a landing is one bit of
   its execution's landed record.  What must stay: the order above, the same at every rank and in every execution, so
   both ends' lists agree; one receive a page at a node, so each landed bit has one writer. */
static void plan_walk(struct plan *p){
  const uint32_t n=p->n,self=p->self;
  size_t first[(size_t)n+2],child[(size_t)n+1];
  uint64_t at=0,receives=0;
  for(uint32_t i=0;i<p->demands;i++){
    const struct plan_demand *d=p->demand+i;
    uint64_t k=0;
    for(uint32_t j=d->tree;j<d->tree+d->trees;j++){
      const uint64_t end=k+p->run[j];
      const struct plan_tree tree={p->parent+p->tree[j]*n,n,d->source};
      uint32_t in=MESH_ABSENT;
      mesh_graph_climb(tree.parent,n,d->source,self,plan_tree_up,&in);
      mesh_graph_rows(n+1,n,plan_tree_parent,&tree,first,child);
      if(in==MESH_ABSENT && first[self+1]==first[self]){k=end;continue;}
      for(;k<end;k++){
        /* the page's bit: its receive's here; none at the source, whose sends its release word releases */
        uint32_t bit=MESH_ABSENT;
        if(in!=MESH_ABSENT){
          bit=(uint32_t)receives++;
          if(p->entry)p->entry[at]=(struct plan_entry){d->value,in,MESH_RECEIVE,bit,k,0};
          at++;
        }
        for(size_t c=first[self];c<first[self+1];c++,at++)
          if(p->entry)p->entry[at]=(struct plan_entry){d->value,(uint32_t)child[c],MESH_SEND,bit,k,d->release};
      }
    }
  }
  p->entries=at;p->receives=receives;
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
   mesh_graph_fill as they were, so every rank's trees and template are the same.
   2026-11-03, sub-assignment 5 (the change request, "Claude's proposal: the canonical path", items 1-3; Appendix A,
   A3, "Its output" and "Its failure", and A4): the tail writes the unrolled list in the order
   design/disjoint-pages-coordination.md, "Writing a plan" gives: this node's hops of one execution walked (plan_walk)
   and every room checked before anything is touched; then `hops` 0, the previous plan erased (mesh-call.h
   mesh_plan_erased), every declared instance's rows bound to its pages and those pages zeroed, the supplied words and
   landed records zeroed, the rows' publications and the entries written, the map read again and `hops` stored.  It
   reverses the slots' tail: plan_pages' check of rows bound before the plan, the relay and words sections, hops
   written slot by slot (the note before plan_page).  What must stay: every failure but the erase's ETIMEDOUT and the
   map's ESTALE found before `hops` is stored 0. */
int mesh_plan(struct mesh_ctx *context,const struct mesh_declared *declared,uint32_t count,uint32_t executions,
              const struct mesh_dependence *dependence,uint32_t dependences){
  struct hdr *m=context->M;
  if(!executions || (count && !declared) || (dependences && !dependence))return EINVAL;
  /* a program's data edge joins a range consumed at a node to a range that node produces (mesh-collective.h struct
     mesh_dependence) */
  for(uint32_t e=0;e<dependences;e++)
    if(dependence[e].consumed>=count || dependence[e].produced>=count ||
       declared[dependence[e].consumed].destination!=declared[dependence[e].produced].source)return EINVAL;
  struct mesh_nodes nodes,after;
  int status=mesh_nodes_read(context,&nodes);
  if(status)return status;
  const uint32_t n=nodes.count;
  if(nodes.self>=n)return EHOSTUNREACH;
  const uint64_t block=(uint64_t)m->block*m->pgsz;
  struct plan p={.context=context,.declared=declared,.n=n,.self=nodes.self,.executions=executions,.serial=nodes.serial};
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
        .first=key[i].index,.mine=MESH_ABSENT,.bytes=t->section.bytes,.pages=(t->section.bytes+block-1)/block};
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
  /* this node's sections of the ranges it is the source or a destination of, each instance's rows (plan_section), no
     row two demands' (plan_disjoint) */
  for(uint32_t i=0;i<p.demands;i++){
    const struct plan_demand *d=p.demand+i;
    if(d->mine!=MESH_ABSENT && (status=plan_section(context,&declared[d->mine].section,d->pages,executions)))goto done;
  }
  if((status=plan_disjoint(&p)))goto done;

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

  /* this node's hops of one execution (plan_walk: counted, then filled) and the rooms they take in the layout, checked
     before anything is erased (design/disjoint-pages-coordination.md, "Writing a plan"): executions x n entries past
     `hop_capacity`, releases x executions supplied words (MESH_WORD_STRIDE bytes each) past `word_capacity`, or
     executions landed records past `landed_capacity`, ENOSPC.  A record is `stride` = 32 x ceil((1 + ceil(R / 64)) /
     4) bytes, its cancel word and R bits in whole 32-byte vectors (mesh.h, "The landed list"; R = 0 gives 32) */
  plan_walk(&p);
  const uint64_t stride=32*((1+(p.receives+63)/64+3)/4);
  uint64_t entries,supplied,records;
  if(p.receives>=UINT32_MAX || __builtin_mul_overflow(p.entries,(uint64_t)executions,&entries) ||
     entries>m->hop_capacity || __builtin_mul_overflow(p.releases,(uint64_t)executions*MESH_WORD_STRIDE,&supplied) ||
     supplied>m->word_capacity || __builtin_mul_overflow(stride,(uint64_t)executions,&records) ||
     records>m->landed_capacity){status=ENOSPC;goto done;}
  if(!(p.entry=malloc((p.entries+1)*sizeof *p.entry))){status=ENOMEM;goto done;}
  plan_walk(&p);

  /* Writing a plan (mesh.h struct mesh_hop, "Writing a plan"; design/disjoint-pages-coordination.md, "Writing a
     plan"), everything above computed first, so a failure there leaves the previous plan standing.  `hops` 0
     (release), and a release fence that keeps every write below after it (review R1 round 2, S2: a release store
     orders what precedes it, not what follows), so a bridge that read `hops` and `plan`, then the entries, then them
     again after an acquire fence (mesh-flow.c link_prepare, plan_cancel) sees a change wherever it read a torn list;
     then `plan` woken, so the ports' progress threads, which wait on it, let the previous plan go now */
  atomic_store_explicit(&m->hops,0,memory_order_release);
  atomic_thread_fence(memory_order_release);
  os_sync_wake_by_address_all(&m->plan,sizeof m->plan,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
  /* no port's data path holds the previous plan, `plan` as it stands (mesh-call.h mesh_plan_erased, bounded at
     MESH_PLATFORM_NS from its start; its ETIMEDOUT returned as is): no receive of it is posted into a page this plan
     binds or zeroes, and no send of it reads a word this plan zeroes */
  const uint64_t previous=atomic_load_explicit(&m->plan,memory_order_relaxed);
  if((status=mesh_plan_erased(context,previous)))goto done;
  /* this plan's `executions` and `landed_stride` first, while `hops` is 0, so that mesh.h's helpers (mesh_supplied,
     mesh_landed_record) name this plan's words wherever they are zeroed and written below (review 12's finding 7).
     The contract lists them with the entries; the bridge reads either only after `hops` (acquire) and drops what it
     read where `hops` or `plan` moved (mesh-flow.c link_prepare, plan_cancel), so the earlier store changes nothing it
     can take as whole */
  m->executions=executions;
  m->landed_stride=stride;
  /* each declared section's instances bound to their pages: instance t's rows first + t x stride .. + P to its pages
     (plan_page), t its index (mesh-dataflow.h mesh_backing_bind), so every instance's pages are in the page table
     before the first execution (Appendix A, A3, "Its output"); after the erase, since rows an earlier plan bound may
     take other pages now.  A relay's value has no rows */
  for(uint32_t v=0;v<p.values;v++){
    if(p.value[v].declared==MESH_ABSENT)continue;
    const struct mesh_section *s=&declared[p.value[v].declared].section;
    for(uint32_t t=0;t<executions;t++)mesh_backing_bind(context,s->first+t*s->stride,s->pages,plan_page(&p,v,t,0),t);
  }
  /* Every page bound to a declared section's instance zeroed here, before `hops` is stored (the coordinator's ruling on
     sub-assignment 11's finding, design/disjoint-pages-coordination.md, the record): a peer's SEND that the hardware
     holds in credit lands as soon as this node's bridge loads the plan and posts its receive, so the program can no
     longer zero a routed operand after mesh_plan (tools/e2b_replay.m stopped doing so, 01499f5f).  A page two
     instances share is zeroed for each, never more bytes than a plan with no reuse zeroes once.  A relay's pages are
     read only by their forwards, each after its landing, and are not zeroed.  What must stay: the zeroing before
     `hops` is stored. */
  for(uint32_t v=0;v<p.values;v++){
    if(p.value[v].declared==MESH_ABSENT)continue;
    for(uint32_t t=0;t<executions;t++)memset(mesh_at(m,plan_page(&p,v,t,0)),0,(size_t)(p.value[v].blocks*block));
  }
  /* the supplied words and the landed records this plan uses, zeroed (mesh.h, "Supplied words" and "The landed
     list"): releases x executions words from mesh_supplied(m, 0, 0), release r's executions in turn, and records 0 ..
     executions - 1 from mesh_landed_record(m, 0), contiguous; word by word, each an _Atomic uint64_t others load */
  for(uint64_t w=0;w<supplied/sizeof(uint64_t);w++)atomic_store_explicit(mesh_supplied(m,0,0)+w,0,memory_order_relaxed);
  for(uint64_t w=0;w<records/sizeof(uint64_t);w++)atomic_store_explicit(mesh_landed_record(m,0)+w,0,memory_order_relaxed);
  /* Instance 0's rows of each declared section record the plan's words for the program (struct mesh_publication;
     design/disjoint-pages-coordination.md, "Rows' publications"), cleared first.  A destination's page: its bit
     (`landed`), `device_input` execution 0's landed record and `device_stride` the records' stride, which
     mesh-metal.m mesh_metal_receive_prepare and mesh_host_arrived read (a range's bits one run: plan_walk); a row with
     no receive keeps `device_input` 0.  A source range's first row: its supplied word of execution 0 (`targets[0]`,
     count 1, stride 1), `sends` 1 and `argument` MESH_SUPPLIED, which mesh-dataflow.c mesh_publication_prepare and
     mesh_host_publish read, one store releasing the range (proposal item 3).  A section is one demand's alone
     (plan_disjoint), so a source row names one release; every row has room for two targets (mesh-map.h mesh_layout's
     `target_stride`, at least 64 bytes), so no target is refused */
  for(uint32_t v=0;v<p.values;v++){
    if(p.value[v].declared==MESH_ABSENT)continue;
    const struct mesh_section *s=&declared[p.value[v].declared].section;
    for(uint64_t k=0;k<p.value[v].blocks;k++){
      struct mesh_publication *publication=mesh_publication_at(m,s->first+(uint32_t)k);
      publication->sends=publication->landed=0;
      publication->device_input=publication->device_stride=0;
      atomic_store_explicit(&publication->argument,0,memory_order_relaxed);
    }
  }
  for(uint64_t j=0;j<p.entries;j++){
    const struct plan_entry *e=p.entry+j;
    const uint32_t held=p.value[e->value].declared;
    if(e->direction!=MESH_RECEIVE || held==MESH_ABSENT)continue;
    struct mesh_publication *publication=mesh_publication_at(m,declared[held].section.first+(uint32_t)e->page);
    publication->landed=e->bit;
    publication->device_input=(uint64_t)((char *)mesh_landed_record(m,0)-(char *)m);
    publication->device_stride=stride;
  }
  for(uint32_t i=0;i<p.demands;i++){
    const struct plan_demand *d=p.demand+i;
    if(d->source!=p.self)continue;
    struct mesh_publication *publication=mesh_publication_at(m,declared[d->mine].section.first);
    publication->targets[0]=(struct mesh_target){.stream=(uint64_t)((char *)mesh_supplied(m,d->release,0)-(char *)m),
                                                 .count=1,.stride=1};
    publication->sends=1;
    atomic_store_explicit(&publication->argument,MESH_SUPPLIED,memory_order_relaxed);
  }
  /* the entries, execution t's n at [t n, t n + n), each naming its instance's page and its execution's word
     (design/disjoint-pages-coordination.md, "Hop entries"): a receive's, and a forwarding send's, the word of execution
     t's landed record holding its bit, `mask` 1 << (bit % 64); a source's send its release's supplied word of execution
     t, `mask` MESH_SUPPLIED */
  struct mesh_hop *hop=mesh_hops(m);
  for(uint32_t t=0;t<executions;t++)for(uint64_t j=0;j<p.entries;j++){
    const struct plan_entry *e=p.entry+j;
    const _Atomic uint64_t *word=e->bit==MESH_ABSENT?mesh_supplied(m,e->release,t):mesh_landed_record(m,t)+1+e->bit/64;
    *hop++=(struct mesh_hop){.page=plan_page(&p,e->value,t,e->page),.node=e->node,.direction=e->direction,
      .word=(uint64_t)((const char *)word-(const char *)m),
      .mask=e->bit==MESH_ABSENT?MESH_SUPPLIED:UINT64_C(1)<<(e->bit%64),.serial=p.serial};
  }
  /* `plan` the plan's generation, its previous value + 1 and never 0, and the map serial the plan was made against its
     entries' `serial` alone, so a second plan on an unchanged map is told from the first (the coordinator's change to
     CONTRACT.md's interface and mesh.h's comment above struct mesh_hop; 17612857 had `plan` the map serial).  What
     must stay: the bridge refuses a plan whose entries' serial is not `map_serial` as it loads it */
  const uint64_t generation=previous+1;
  atomic_store_explicit(&m->plan,generation?generation:1,memory_order_relaxed);
  if(!(status=mesh_nodes_read(context,&after)) && after.serial!=p.serial)status=ESTALE;
  if(status)goto done;
  atomic_store_explicit(&m->hops,entries,memory_order_release);
  /* the bridge's progress threads watch `plan` with bounded waits (CONTRACT.md "Wave-2 interfaces") */
  os_sync_wake_by_address_all(&m->plan,sizeof m->plan,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
done:
  free(key);free(placed);free(p.demand);free(p.of);
  free(root);free(up);free(down);free(demand);free(p.parent);free(share);
  free(p.tree);free(p.run);free(rate);free(latency);free(trees);
  free(p.child);free(p.value);free(p.read);free(p.first);free(p.head);free(p.lag);free(p.colour);free(p.entry);
  return status;
}

