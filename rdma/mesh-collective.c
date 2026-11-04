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

/* mesh-collective.h mesh_host_publish (design/disjoint-pages-change-request.md, C3.7; design/disjoint-pages-revision-
   coordination.md, "The region", `supplied`, and D9): execution + 1 stored (release) into the supplied word a source
   range's first row names (`targets[0].stream`, which mesh_plan writes), after the caller's writes to the range in that
   execution.  One store releases every send of the range's pages, and the word only rises while the caller's
   executions run in order.  A row that is no source range's first names no target (`sends` 0), and nothing is stored.
   2026-11-04, sub-assignment 3: reversing 2026-11-03's form (sub-assignment 5), MESH_SUPPLIED stored into execution t's
   own word at `stream` + t x MESH_WORD_STRIDE.  Why: one word serves every execution, its value saying which, and no
   word is reset while its plan stands (C3.7).  What must stay: one store, release, after the range's last write. */
void mesh_host_publish(struct mesh_ctx *context,struct mesh_section section,uint64_t execution){
  struct hdr *m=context->M;
  const struct mesh_publication *publication=mesh_publication_at(m,section.first);
  if(!publication->sends)return;
  atomic_store_explicit((_Atomic uint64_t *)((char *)m+publication->targets[0].stream),execution+1,memory_order_release);
}

/* mesh-collective.h mesh_host_arrived (C3.1, C3.7; the contract's "The region", the phases and `counts`, and D1):
   execution t of a received range, from its first row's publication (`pages` P, `phases` its copy-0 group, `count` its
   copy-0 count word, `state` its plan's struct mesh_state, whose `copies` is u).  Copy c = t mod u: its count word
   (count + 8 c) is read first (acquire), then its group's words (phases + c x mesh_group_bytes(P), 32 fields a word,
   acquire).  A landing XORs its field before it adds to the count (both release, and every write of a count is a
   read-modify-write), so a count read at (t / u + 1) x P makes every landing it counts visible to the reads after it:
   a field still one behind then is a cancelled receive, never a landing in flight.  1 where every field reads
   mesh_phase(t / u + 1); UINT64_MAX where one reads neither that nor mesh_phase(t / u) (a write ran ahead of the
   allocator's order), or where the count has reached (t / u + 1) x P with a field behind (cancelled); else 0.  0 where
   the row is no received range's first (`pages` 0).
   2026-11-04, sub-assignment 3: reversing 2026-11-03's form (sub-assignment 5), the range's bit run in execution t's
   landed record and its cancel word.  Why: the records and the cancel word are gone (C4); a landing is a phase, a
   cancellation an arrival on the count (C3.1, C3.7).  What must stay: 0 until the range has landed, then 1; UINT64_MAX
   once a receive of it was cancelled; a range read by its words, never a word a page. */
uint64_t mesh_host_arrived(struct mesh_ctx *context,struct mesh_section section,uint64_t execution){
  struct hdr *m=context->M;
  const struct mesh_publication *head=mesh_publication_at(m,section.first);
  if(!head->pages || !head->state)return 0;
  const struct mesh_state *state=(const struct mesh_state *)((char *)m+head->state);
  const uint64_t copies=state->copies,copy=execution%copies,round=execution/copies,pages=head->pages;
  const uint32_t expected=mesh_phase(round+1),behind=mesh_phase(round);
  const uint64_t count=atomic_load_explicit((_Atomic uint64_t *)((char *)m+head->count+8*copy),memory_order_acquire);
  _Atomic uint64_t *group=(_Atomic uint64_t *)((char *)m+head->phases+copy*mesh_group_bytes(head->pages));
  int landed=1;
  for(uint64_t w=0;w<(pages+31)/32;w++){
    const uint64_t word=atomic_load_explicit(group+w,memory_order_acquire);
    for(uint64_t k=w*32;k<pages && k<w*32+32;k++){
      const uint32_t field=(uint32_t)(word>>(2*(k%32))&3);
      if(field==expected)continue;
      if(field!=behind)return UINT64_MAX;
      landed=0;
    }
  }
  if(landed)return 1;
  return count>=(round+1)*pages?UINT64_MAX:0;
}

/* -- the plan call (mesh-collective.h mesh_plan) ------------------------------------------------------------------ */

/* tag pre-return-20261103 rdma/nccl-mesh.c MESH_SOLVE_EPS: the concurrent-flow solve's epsilon, its rounds and room
   the tag's mesh_solve and mesh_solve_room (rdma/mesh-collective.c) */
#define MESH_PLAN_EPS 0.1

/* a demand: the declared transfers of one identity from one source, one range to each of their destinations */
struct plan_demand {
  uint32_t source,identity,order,first;   /* first: its first declared transfer, the tie after `order` */
  uint32_t mine;                          /* the index in `values` of this node's value of it (this node its source or
                                             a destination), else MESH_ABSENT */
  uint32_t tree,trees;                    /* its trees, plan `tree` [tree, tree + trees) */
  uint32_t release;                       /* this node's release of it, its supplied word's index (struct mesh_state
                                             `supplied`), where this node is its source */
  uint64_t leaves,bytes,pages;            /* its destinations (a set of map indices), its bytes and pages */
  /* its part of the dependence template (plan_template) and its value here: */
  uint64_t on;                            /* the nodes its trees reach past its source (a set of map indices), the
                                             trees the fill gave pages alone: each has its W and L */
  uint32_t forwards;                      /* its distinct tree edges y -> c over those trees, each its F */
  uint32_t own;                           /* 1 where a dependence names its source's own read: it has S(d) */
  uint32_t vertex;                        /* its first template vertex, P(d); the rest follow (plan_landed) */
  uint32_t value;                         /* its routed value at this node, p->here[value]; MESH_ABSENT where this node
                                             holds none of its pages */
  /* this node's received range of it (plan_walk's count): g, in the plan's demand order (MESH_ABSENT where none of its
     pages lands here), and its pages that land here in one copy, P_g (its pages P at a destination; a relay's those of
     the trees it is on) */
  uint32_t range,received;
  /* its consumed range here: h, among the received ranges this node is a destination of, in the plan's demand order,
     its count words' row (MESH_ABSENT elsewhere: a relay's receives count nothing, review 15's finding 1) */
  uint32_t consumed;
  uint64_t group;                         /* range g's copy-0 phase group, its offset in the phase list (plan_layout) */
};
/* A routed value (design/disjoint-pages-change-request.md, C3.3; design/disjoint-pages-revision-coordination.md, "The
   allocator"): demand `demand`'s pages at node `node`, its source or a node its trees reach (a destination or a relay),
   at every node, since u is the collective's.  Its write is P at the source, W(node) elsewhere; its reads
   p->read[read .. read + reads), F(node, c) for each c it forwards to, ascending, then C(node) at a destination, then
   S at the source where its demand has one (the source's own read): at least one (every leaf of a tree is a
   destination, so a relay forwards).  `source`: its first row among the least lags' sources (plan_unroll), its write,
   then its reads in turn. */
struct plan_value { uint32_t demand,node,write,read,reads,source; };
/* One of this node's hops of one copy (plan_walk; mesh.h struct mesh_hop), every copy's the same: routed value
   p->here[value]'s page `page`, sent to or received from `node`, of demand `demand`.  `field`: a receive's field in its
   range's group (its demand's receives here in walk order, 0 .. P_g - 1); a forwarding send's, its receive's of the
   same page; MESH_ABSENT for a send from the range's source, which its release's supplied word releases. */
struct plan_entry { uint32_t value,node,direction,demand,field; uint64_t page; };
struct plan {
  struct mesh_ctx *context;
  const struct mesh_declared *declared;
  struct mesh_value *given;               /* the caller's values, `givens` of them */
  uint32_t givens;
  uint32_t *named;                        /* named[j]: the first declared transfer of the demand whose range values[j]
                                             is here, MESH_ABSENT where none (a local value) */
  struct plan_demand *demand;             /* the demands, by order and first transfer */
  uint32_t demands,n,self,copies;         /* copies: u (plan_unroll) */
  uint64_t serial;
  uint32_t *of;                           /* of[i]: declared[i]'s demand, p->demand[of[i]] */
  size_t *tree;                           /* tree[j] the solve's tree, demand by demand (mesh_graph_rows) */
  uint32_t *parent;                       /* the solve's parents, tree by tree */
  uint64_t *run;                          /* run[j]: tree[j]'s pages of its demand */
  uint64_t releases;                      /* this node's releases, the demands it is the source of */
  /* this node's hops of one copy, every copy's in this order (plan_walk): */
  struct plan_entry *entry;               /* entry[0 .. entries), NULL while plan_walk counts */
  uint64_t entries,receives;              /* their count (n of mesh.h's u x n) and their receives */
  uint32_t ranges,consumed;               /* this node's received ranges, and those it consumes (a destination's) */
  /* The dependence template (plan_template), one execution's, every rank's the same from the same declared transfers,
     dependences and map; each edge carries the execution distance `lag` from its tail's execution to its head's. */
  uint64_t *child;                        /* child[i x n + y]: the nodes y sends demand i's pages to over its trees (a
                                             set of map indices; 0 for a node off them) */
  struct plan_value *value;               /* every node's routed values, demand by demand, a demand's by node */
  uint32_t values;
  uint32_t *read;                         /* their read vertices (struct plan_value `read`, `reads`) */
  uint32_t *here,local;                   /* this node's routed values, here[0 .. local), p->value's, in demand order */
  uint32_t vertices;                      /* the template's vertices, demand by demand from each one's `vertex` */
  size_t edges;                           /* the template's edges */
  size_t *first;                          /* vertex v's edges, in compressed rows: [first[v], first[v + 1]),
                                             vertices + 1 entries (mesh_graph_rows by tail) */
  uint32_t *head,*lag;                    /* edge k's head vertex, and its execution distance */
  uint32_t *between;                      /* between[a x local + b]: D_ab, the largest least lag from a read of this
                                             node's routed value a to b's write (plan_unroll) */
  /* the state section (plan_layout), offsets from its start */
  uint64_t phases,counts,supplied,status,words;
  /* the items (plan_place): routed copies (a, c) at item a x u + c, then this node's other values (others[k] the index
     in `values` of item local x u + k), then the state section, item `state` */
  uint32_t *others,other,state;
  uint64_t *colour;                       /* item i's first block in its space */
  uint64_t window,outside;                /* the blocks each colouring uses (mesh_graph_left_edge's `used`) */
  struct mesh_claim claim;
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
/* a value's blocks: `bytes` in whole blocks of block x pgsz bytes */
static uint64_t plan_blocks(const struct hdr *m,uint64_t bytes){
  const uint64_t quantum=(uint64_t)m->block*m->pgsz;
  return (bytes+quantum-1)/quantum;
}

/* -- the dependence template (design/disjoint-pages-revision-coordination.md, "The allocator") -------------------- */

/* v's place among a set's members (map indices below 64) */
static uint32_t plan_rank(uint64_t set,uint32_t v){ return (uint32_t)__builtin_popcountll(set&((UINT64_C(1)<<v)-1)); }
/* Demand i's template vertices, from its `vertex`: P(i), its production at its source (an edge into it is before its
   first write, an edge out of it after its last); for each node y of `on` in ascending order, W(i, y) its first
   landing at y and L(i, y) its last; for each tree edge y -> c in (y, c) order, F(i, y, c), y's last forward to c; for
   each destination z in ascending order, C(i, z), z's consumption (an edge into it is before its first read, an edge
   out of it after its last); last, where its `own` is 1, S(i), its source's own read of it (the same).
   2026-11-04, sub-assignment 3 (the coordinator's ruling on 13's and 14's finding; design/disjoint-pages-revision-
   coordination.md, "The allocator"): S(i) added.  Why: a source's local read of the range it sends was in no event,
   its value's reads its forwards alone, so the allocator could let a later landing reuse its pages while the source
   still read them (in e2b, a peer's exchange k + 1 landing in the pages of this node's exchange-k partial as its own
   combine reads it).  S(i) exists only where a dependence names it: a read with no edge out has no unroll bound.
   What must stay: every other vertex's place, so the offsets below hold. */
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
static uint32_t plan_own(const struct plan *p,uint32_t i){
  const struct plan_demand *d=p->demand+i;
  return d->vertex+1+2*(uint32_t)__builtin_popcountll(d->on)+d->forwards+(uint32_t)__builtin_popcountll(d->leaves);
}
/* a program event (mesh-collective.h struct mesh_dependence), from its end's bits (`produced` MESH_FROM_PRODUCED or
   MESH_TO_PRODUCED, `own` MESH_FROM_SOURCE_READ or MESH_TO_SOURCE_READ): declared[i]'s production at its source, P of
   its demand; its source's own read, S of its demand; or its consumption at its destination, C of its demand there */
static uint32_t plan_event(const struct plan *p,uint32_t i,int produced,int own){
  if(produced)return plan_produced(p,p->of[i]);
  if(own)return plan_own(p,p->of[i]);
  return plan_consumed(p,p->of[i],p->declared[i].destination);
}
/* the node an event is at: the source for a production or the source's own read, else the destination */
static uint32_t plan_event_node(const struct mesh_declared *declared,uint32_t i,int produced,int own){
  return produced || own?declared[i].source:declared[i].destination;
}
/* a template edge before its row is known (mesh_graph_rows' records, keyed by tail) */
struct plan_edge { uint32_t tail,head,lag; };
static uint32_t plan_edge_tail(const void *context,size_t k){ return ((const struct plan_edge *)context)[k].tail; }

/* The dependence template and the routed values (design/disjoint-pages-change-request.md, C3.3;
   design/disjoint-pages-revision-coordination.md, "The allocator"), after the routing.
   - Each demand's trees, the ones the fill gave pages (a tree of none carries nothing): each vertex's children on
     them, from its row of the tree's vertices keyed by parent (mesh_graph_rows, as plan_walk reads them; no walk over
     a parent table here, C9), make `child` and `on`.
   - The template's edges, all of lag 0 but the program's: P -> W(y) for every y of `on`; P -> F(source, c); W(y) ->
     L(y); W(y) -> F(y, c); F(y, c) -> L(c); L(z) -> C(z); P -> S where the demand has S; and each dependence e one
     edge from its `from` event's vertex to its `to` event's (plan_event: P of the demand where its produced bit is
     set, S where its source-read bit is, else C at the declared transfer's destination), of lag e.lag.  A demand has S
     where a dependence's end names its source's own read.  Every edge is sound: its tail happens before its head
     whatever the cables do (a range is produced before it is sent and before its source reads it; a page lands before
     it is forwarded, is sent before it lands, and lands before it is consumed; the program's edges are its own
     claims).  Grouped into rows by tail (mesh_graph_rows), so every rank
     numbers them alike.
   - The routed values, at every node, demand by demand and a demand's by ascending node: its source's, and each node
     of `on`'s (a destination's or a relay's), each with its write and reads (struct plan_value); this node's among them
     (`here`, and each demand's `value`).  Every node's, since u is the largest L over the collective's values (the
     contract's D5), which every rank computes alike.
   2026-11-04, sub-assignment 3: reversing 2026-11-03's (sub-assignment 3's) program edges, one kind, C(consumed) ->
   P(produced), and its values at this node alone.  Why: the contract's struct mesh_dependence names either event of
   either declared transfer (four kinds), and u needs every node's values.  What must stay: the transport's edges as
   they were, every edge sound, every rank's template the same.
   0, or ENOMEM. */
static int plan_template(struct plan *p,const struct mesh_dependence *dependence,uint32_t dependences){
  const uint32_t n=p->n,self=p->self;
  size_t first[(size_t)n+2],child[(size_t)n+1];
  p->child=calloc((size_t)p->demands*n+1,sizeof *p->child);
  if(!p->child)return ENOMEM;
  uint64_t vertices=0,edges=dependences,values=0,reads=0;
  for(uint32_t k=0;k<dependences;k++){
    const struct mesh_dependence *x=dependence+k;
    if(x->kinds&MESH_FROM_SOURCE_READ)p->demand[p->of[x->from]].own=1;
    if(x->kinds&MESH_TO_SOURCE_READ)p->demand[p->of[x->to]].own=1;
  }
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
    vertices+=1+2*on+d->forwards+leaves+d->own;
    edges+=2*on+2*(uint64_t)d->forwards+leaves+d->own;
    values+=1+on;
    reads+=d->forwards+leaves+d->own;
    if(vertices>=UINT32_MAX || reads+values>=UINT32_MAX)return ENOMEM;
  }
  p->vertices=(uint32_t)vertices;
  p->edges=(size_t)edges;
  p->value=malloc(((size_t)values+1)*sizeof *p->value);
  p->read=malloc(((size_t)reads+1)*sizeof *p->read);
  p->here=malloc(((size_t)p->demands+1)*sizeof *p->here);
  struct plan_edge *edge=malloc((p->edges+1)*sizeof *edge);
  size_t *order=malloc((p->edges+1)*sizeof *order);
  p->first=malloc(((size_t)p->vertices+1)*sizeof *p->first);
  p->head=malloc((p->edges+1)*sizeof *p->head);
  p->lag=malloc((p->edges+1)*sizeof *p->lag);
  if(!p->value || !p->read || !p->here || !edge || !order || !p->first || !p->head || !p->lag){
    free(edge);free(order);return ENOMEM;
  }
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
    if(d->own)edge[e++]=(struct plan_edge){produced,plan_own(p,i),0};
  }
  for(uint32_t k=0;k<dependences;k++){
    const struct mesh_dependence *x=dependence+k;
    edge[e++]=(struct plan_edge){
      plan_event(p,x->from,!!(x->kinds&MESH_FROM_PRODUCED),!!(x->kinds&MESH_FROM_SOURCE_READ)),
      plan_event(p,x->to,!!(x->kinds&MESH_TO_PRODUCED),!!(x->kinds&MESH_TO_SOURCE_READ)),x->lag};
  }
  mesh_graph_rows(p->vertices,p->edges,plan_edge_tail,edge,p->first,order);
  for(size_t k=0;k<p->edges;k++){p->head[k]=edge[order[k]].head;p->lag[k]=edge[order[k]].lag;}
  free(edge);free(order);
  /* the routed values, every node's */
  uint32_t at=0,source=0;
  p->values=0;p->local=0;
  for(uint32_t i=0;i<p->demands;i++){
    struct plan_demand *d=p->demand+i;
    d->value=MESH_ABSENT;
    for(uint64_t s=d->on|UINT64_C(1)<<d->source;s;s&=s-1){
      const uint32_t x=(uint32_t)__builtin_ctzll(s);
      struct plan_value *v=p->value+p->values;
      *v=(struct plan_value){.demand=i,.node=x,.write=x==d->source?plan_produced(p,i):plan_landed(p,i,x,0),.read=at,
                             .source=source};
      for(uint64_t c=p->child[(size_t)i*n+x];c;c&=c-1)p->read[at++]=plan_forwarded(p,i,x,(uint32_t)__builtin_ctzll(c));
      if(d->leaves>>x&1)p->read[at++]=plan_consumed(p,i,x);
      if(x==d->source && d->own)p->read[at++]=plan_own(p,i);
      v->reads=at-v->read;
      source+=1+v->reads;
      if(x==self){d->value=p->local;p->here[p->local++]=p->values;}
      p->values++;
    }
  }
  return 0;
}

/* -- the one allocator (design/disjoint-pages-change-request.md, C3.2-C3.4; the contract's "The allocator") -------- */

/* mesh_graph_lags' and mesh_graph_order's edges: the template's, edge k's head and execution distance; and the lag-0
   edges' heads alone (plan_acyclic's rows) */
static uint32_t plan_head(const void *context,size_t k){ return ((const struct plan *)context)->head[k]; }
static uint32_t plan_lag(const void *context,size_t k){ return ((const struct plan *)context)->lag[k]; }
static uint32_t plan_head_of(const void *context,size_t k){ return ((const uint32_t *)context)[k]; }

/* No lag-0 cycle (the contract's "The plan call": EINVAL for a lag-0 cycle): the template's lag-0 edges, in rows by
   tail as the template's, have a topological order (mesh-graph.h mesh_graph_order; its EDEADLK is EINVAL), as
   2026-11-03's mesh_graph_happens_before refused one.  The template's own edges are acyclic (each demand's run down
   from P), so a cycle is the program's: one execution's events each before the next and the last before the first.
   The contract asks for D(x -> x) == 0 on any routed vertex; this tests every vertex at once, since a least lag shows
   a cycle only at a vertex both a source and a target of mesh_graph_lags (whose targets are the writes), and a cycle
   of consumptions alone passes no write (owner 2's remark, routed by the coordinator).  With it, every D(w -> w) and
   every L(v) below is at least 1.  0; EINVAL; ENOMEM. */
static int plan_acyclic(const struct plan *p){
  size_t *first=malloc(((size_t)p->vertices+1)*sizeof *first);
  uint32_t *head=malloc((p->edges+1)*sizeof *head),*indegree=malloc(((size_t)p->vertices+1)*sizeof *indegree),
           *ordinal=malloc(((size_t)p->vertices+1)*sizeof *ordinal);
  int status=!first || !head || !indegree || !ordinal?ENOMEM:0;
  if(!status){
    size_t e=0;
    for(uint32_t v=0;v<p->vertices;v++){
      first[v]=e;
      for(size_t k=p->first[v];k<p->first[v+1];k++)if(!p->lag[k])head[e++]=p->head[k];
    }
    first[p->vertices]=e;
    status=mesh_graph_order(p->vertices,first,plan_head_of,head,indegree,ordinal)?EINVAL:0;
  }
  free(first);free(head);free(indegree);free(ordinal);
  return status;
}

/* a routed value named on stderr (the contract's "The plan call", ENOMEM): its range's identity and source, its node,
   and here its index in `values` (or that it is this node's relay); the least lag found */
static void plan_name(const struct plan *p,uint32_t v,const char *what,uint32_t lag){
  const struct plan_value *x=p->value+v;
  const struct plan_demand *d=p->demand+x->demand;
  char here[48]="";
  if(x->node==p->self){
    if(d->mine!=MESH_ABSENT)snprintf(here,sizeof here," (values[%u] here)",d->mine);
    else snprintf(here,sizeof here," (this node's relay)");
  }
  if(lag==UINT32_MAX)
    fprintf(stderr,"mesh_plan: the value of range %u from rank %u at rank %u%s: %s (no walk)\n",d->identity,d->source,
            x->node,here,what);
  else fprintf(stderr,"mesh_plan: the value of range %u from rank %u at rank %u%s: %s (least lag %u)\n",d->identity,
               d->source,x->node,here,what,lag);
}

/* Unrolling by modulo variable expansion (design/disjoint-pages-change-request.md, C3.3: M. Lam, Software pipelining,
   PLDI 1988; B. R. Rau, Iterative modulo scheduling, MICRO 1994; the contract's D5 and "The allocator").  The least
   lags D(x -> y) over the template (mesh-graph.h mesh_graph_lags, Dijkstra with the lags as weights), from every routed
   value's write and reads, at every node, to every routed value's write:
   - every write w has D(w -> w) == 1, its next instance reached at lag 1, else ENOMEM naming its value (D5: with it, an
     order at a lag holds at every later lag, so the least lags decide every test below exactly); D(w -> w) == 0, a
     lag-0 cycle, plan_acyclic has refused;
   - L(v) = the largest D(r -> write v) over v's reads r, the least lag at which every read of v in execution t happens
     before its write in t + L; UINT32_MAX (a read nothing orders before a later write) is ENOMEM naming the value;
   - u = the largest L(v) over every routed value at every node, 1 where there is none.  Every rank computes the same
     template, lags and u, so both ends of every cable hold the same copies;
   - between[a x local + b] = D_ab, the largest D(r -> write b) over this node's routed value a's reads r, for every
     pair of this node's routed values, so the colouring's tests (plan_interferes) are table reads.
   0; ENOMEM. */
static int plan_unroll(struct plan *p){
  const uint32_t values=p->values;
  p->copies=1;
  if(!values)return 0;
  const struct plan_value *last=p->value+values-1;
  const uint32_t sources=last->source+1+last->reads;
  size_t cells,pairs;
  if(__builtin_mul_overflow((size_t)sources,(size_t)values,&cells) || cells>=SIZE_MAX/sizeof(uint32_t) ||
     __builtin_mul_overflow((size_t)p->local,(size_t)p->local,&pairs) || pairs>=SIZE_MAX/sizeof(uint32_t))return ENOMEM;
  uint32_t *source=malloc((size_t)sources*sizeof *source),*target=malloc((size_t)values*sizeof *target),
           *lags=malloc((cells+1)*sizeof *lags);
  p->between=malloc((pairs+1)*sizeof *p->between);
  int status=!source || !target || !lags || !p->between?ENOMEM:0;
  for(uint32_t v=0;!status && v<values;v++){
    const struct plan_value *x=p->value+v;
    source[x->source]=x->write;
    for(uint32_t j=0;j<x->reads;j++)source[x->source+1+j]=p->read[x->read+j];
    target[v]=x->write;
  }
  if(!status)status=mesh_graph_lags(p->vertices,p->first,plan_head,plan_lag,p,sources,source,values,target,lags);
  for(uint32_t v=0;!status && v<values;v++){
    const struct plan_value *x=p->value+v;
    const uint32_t cycle=lags[(size_t)x->source*values+v];
    if(cycle!=1){plan_name(p,v,"its write does not reach its next instance at lag 1",cycle);status=ENOMEM;break;}
    uint32_t most=0;
    for(uint32_t j=0;j<x->reads;j++){
      const uint32_t lag=lags[((size_t)x->source+1+j)*values+v];
      if(lag>most)most=lag;
    }
    if(most==UINT32_MAX){plan_name(p,v,"a read of it happens before no later write of it",most);status=ENOMEM;break;}
    if(most>p->copies)p->copies=most;
  }
  for(uint32_t a=0;!status && a<p->local;a++)for(uint32_t b=0;b<p->local;b++){
    const struct plan_value *x=p->value+p->here[a];
    uint32_t most=0;
    for(uint32_t j=0;j<x->reads;j++){
      const uint32_t lag=lags[((size_t)x->source+1+j)*values+p->here[b]];
      if(lag>most)most=lag;
    }
    p->between[(size_t)a*p->local+b]=most;
  }
  free(source);free(target);free(lags);
  return status;
}

/* The state section's layout (design/disjoint-pages-change-request.md, C3.2 and C3.7; the contract's "The region"),
   offsets from its start, each part 64-byte aligned, its total whole blocks (plan_place sizes it):
   - 0: struct mesh_state (64 bytes);
   - `phases` (64): the phase list, received range g's copy c group at phases + group_g + c x mesh_group_bytes(P_g),
     group_g the sum of u x mesh_group_bytes(P_g') over g' < g (groups in (g, c) order, g major; each whole lanes, so a
     lane's MESH_LANE_BYTES never spans two groups); page k of the range (k < P_g) at bits [2 (k % 32), 2 (k % 32) + 2)
     of uint64 word k / 32 of its group;
   - `counts`: consumed x u uint64 words, consumed range h's copy c at index h x u + c: a count word for each range
     this node is a destination of, none for a relay's receives, which no consumer here waits on and whose forwards
     their phases release (review 15's finding 1, the coordinator's ruling); struct mesh_state `ranges` is `consumed`,
     the counts' rows, while the phase list holds a group for every received range, a relay's among them;
   - `supplied`: releases uint64 words, release r's at index r;
   - `status`: one struct mesh_status;
   - `words`: the end of the status, the bytes mesh_plan starts (D3).
   Every offset in struct mesh_state is a region offset: the section's own plus these. */
static uint64_t plan_align(uint64_t at){ return (at+63)/64*64; }
static void plan_layout(struct plan *p){
  uint64_t at=sizeof(struct mesh_state);
  p->phases=at;
  at=0;
  for(uint32_t i=0;i<p->demands;i++){
    struct plan_demand *d=p->demand+i;
    if(d->range==MESH_ABSENT)continue;
    d->group=at;
    at+=(uint64_t)p->copies*mesh_group_bytes(d->received);
  }
  p->counts=plan_align(p->phases+at);
  p->supplied=plan_align(p->counts+(uint64_t)p->consumed*p->copies*sizeof(uint64_t));
  p->status=plan_align(p->supplied+p->releases*sizeof(uint64_t));
  p->words=p->status+sizeof(struct mesh_status);
}

/* the colouring's order (the contract's "The allocator"): greedy by size, an item's blocks largest first, then its
   index, which is the value order (this node's routed values in demand order, then its other values in `values`'s
   order, then the state section) and, within a routed value, the copy */
struct plan_sized { uint64_t blocks; uint32_t item; };
static int plan_sized_order(const void *a,const void *b){
  const struct plan_sized *x=a,*y=b;
  if(x->blocks!=y->blocks)return x->blocks>y->blocks?-1:1;
  return (x->item>y->item)-(x->item<y->item);
}
/* mesh_graph_left_edge's interference, the contract's rule (two items interfere unless it lets them share):
   - an invariant (the state section among them) interferes with every other item;
   - two local execution values share only where their [begin, end] on this node's program order do not overlap (this
     node's work runs in that order, execution after execution); a local execution value interferes with every routed
     copy;
   - routed copies (v, c) and (w, c'), D_vw = between[v x local + w], d = (c' - c) mod u: where d > 0 they share only
     if d >= D_vw and u - d >= D_wv; where d == 0 (v != w) only if D_vw <= u, D_wv <= u and D_vw == 0 or D_wv == 0.
     These are the distances at which their executions meet (t and t' with t' - t = d mod u: d, d - u, d + u, ...);
     by D5 an order at the least such distance holds at every larger one of its class, and a value's every read
     before the other's write orders its writes too.  Two copies of one value follow the d > 0 rule.
   Symmetric: swapping the items swaps D_vw and D_wv and takes d to u - d. */
static int plan_interferes(const void *context,uint32_t a,uint32_t b){
  const struct plan *p=context;
  const uint32_t u=p->copies,routed=p->local*u;
  if(a>=routed || b>=routed){
    if(a<routed || b<routed || a==p->state || b==p->state)return 1;
    const struct mesh_value *x=p->given+p->others[a-routed],*y=p->given+p->others[b-routed];
    if(x->kind==MESH_INVARIANT || y->kind==MESH_INVARIANT)return 1;
    return x->begin<=y->end && y->begin<=x->end;
  }
  const uint32_t v=a/u,c=a%u,w=b/u,c2=b%u,d=(c2+u-c)%u;
  const uint64_t vw=p->between[(size_t)v*p->local+w],wv=p->between[(size_t)w*p->local+v];
  if(d)return !(d>=vw && u-d>=wv);
  return !(vw<=u && wv<=u && (!vw || !wv));
}

/* The one allocator (design/disjoint-pages-change-request.md, C3.2-C3.4: every value placed once, at compile time, by
   one allocator, a program holding exactly the pages its plan uses; the contract's "The allocator"): every item at
   this node, decided once a plan, after the least lags and before anything is claimed or written.
   - The items: each of this node's routed values' u copies (a, c), its demand's P blocks each, a relay's value among
     them (this node on a demand's trees but neither its source nor a destination: no `values` entry names it and no
     row holds it); each other value in `values` (a local invariant or a local execution value), its bytes in blocks,
     one each; the state section (plan_layout's `words` in blocks), one, invariant, no rows.
   - Two colourings by mesh_graph_left_edge, greedy by size (Y. Pisarchyk and J. Lee, arXiv 2001.03288, 2020; the
     order plan_sized_order gives): the wire items (every routed copy, and the other values whose `wire` is 1) in the
     window space, every other item (the state section among them) in the local space, each in blocks, its capacity
     the arena's range of that kind (mesh-dataflow.h mesh_arena_range).  Interference is plan_interferes, its tests
     table reads; under happens-before the lifetimes are not intervals on one order (a landing's write is set by another
     node's send), so left-edge's optimality does not apply, and greedy by size is the heuristic (C3.3).
   - `used` of each colouring (`window`, `outside`) is the claim's size in that space: the program holds exactly that.
   - No room in a space (the colouring's ENOSPC) is ENOMEM, before anything is claimed: no fallback, nothing
     allocated at run time.
   Cost, once a plan and never in the run's length (C3.3): plan_acyclic's mesh_graph_order, O(V^2 + E); the least lags,
   Dijkstra from each source, O(sources x (E + V) log V), sources the routed values' reads and writes at every node,
   and their table, sources x values words; the colouring's walk, quadratic at worst in this node's items (its values
   x u), each test one table read.
   2026-11-04, sub-assignment 3: reversing 2026-11-03's (sub-assignments 4 and 5's) allocator, items a value's every
   execution (values x executions), interference by a happens-before table over the executions (plan_before,
   plan_interferes over mesh_graph_happens_before's bits) and the window alone.  Why: the unroll factor is the longest
   lifetime, not the trip count (C1, C3.3), and the allocator places every value, the state section's words among them
   (C3.2).  What must stay: the items' order, greedy by size; ENOMEM before anything is written.
   0; ENOMEM. */
static int plan_place(struct plan *p){
  struct hdr *m=p->context->M;
  const uint32_t u=p->copies;
  uint64_t routed;
  if(__builtin_mul_overflow((uint64_t)p->local,(uint64_t)u,&routed) || routed+p->givens+1>=UINT32_MAX)return ENOMEM;
  p->others=malloc(((size_t)p->givens+1)*sizeof *p->others);
  if(!p->others)return ENOMEM;
  /* this node's values no declared transfer names, in `values`'s order */
  p->other=0;
  for(uint32_t j=0;j<p->givens;j++)if(p->named[j]==MESH_ABSENT)p->others[p->other++]=j;
  p->state=(uint32_t)routed+p->other;
  const uint32_t items=p->state+1;
  struct plan_sized *sized=malloc((size_t)items*sizeof *sized);
  uint32_t *wired=malloc((size_t)items*sizeof *wired),*outside=malloc((size_t)items*sizeof *outside);
  uint64_t *size=malloc((size_t)items*sizeof *size);
  p->colour=calloc(items,sizeof *p->colour);
  int status=!sized || !wired || !outside || !size || !p->colour?ENOMEM:0;
  if(!status){
    for(uint32_t a=0;a<p->local;a++)for(uint32_t c=0;c<u;c++)
      size[a*u+c]=p->demand[p->value[p->here[a]].demand].pages;
    for(uint32_t k=0;k<p->other;k++)size[routed+k]=plan_blocks(m,p->given[p->others[k]].section.bytes);
    size[p->state]=plan_blocks(m,p->words);
    for(uint32_t i=0;i<items;i++)sized[i]=(struct plan_sized){size[i],i};
    qsort(sized,items,sizeof *sized,plan_sized_order);
    uint32_t w=0,o=0;
    for(uint32_t k=0;k<items;k++){
      const uint32_t i=sized[k].item;
      if(i<routed || (i!=p->state && p->given[p->others[i-routed]].wire))wired[w++]=i;
      else outside[o++]=i;
    }
    const uint64_t window=mesh_arena_range(m,1).count/m->block,local=mesh_arena_range(m,0).count/m->block;
    status=mesh_graph_left_edge(w,size,wired,plan_interferes,p,window,p->colour,&p->window);
    if(!status)status=mesh_graph_left_edge(o,size,outside,plan_interferes,p,local,p->colour,&p->outside);
    if(status==ENOSPC)status=ENOMEM;
  }
  free(sized);free(wired);free(outside);free(size);
  return status;
}

/* item i's first arena page (plan_place's colour in its space, a block of the claim's run there) */
static uint32_t plan_item_page(const struct plan *p,uint32_t i,int wire){
  const struct mesh_range run=wire?p->claim.window:p->claim.local;
  return (uint32_t)(run.first+p->colour[i]*p->context->M->block);
}
/* this node's routed value a's copy c, its page k: arena page window.first + (colour + k) x block, the first page of a
   block inside the registered window */
static uint32_t plan_page(const struct plan *p,uint32_t a,uint32_t c,uint64_t k){
  return (uint32_t)(plan_item_page(p,a*p->copies+c,1)+k*p->context->M->block);
}

/* -- this node's hops (design/disjoint-pages-change-request.md, C3.3 and C3.7) ------------------------------------ */

/* Every page of the plan in one order, the same at every rank: the demands by order and first transfer, a demand's
   pages tree by tree.  A tree carries a page over a node pair once at most, so visiting this node's hops page by page
   in that order, for a pair a -> b, a's sends to b and b's receives from a are the same hops in the same order (mesh.h
   "Both ends").  A page's hops here: its receive from this node's parent on the tree (mesh-graph.h mesh_graph_climb's
   first step), then its sends to this node's children (its row of the tree's vertices keyed by parent,
   mesh_graph_rows), in their order.  The tree's edges are the graph module's: no walk over a parent table here (the
   C9 reviews of c5bffd0c and c9ffedcf).  A page's children are its own tree's, which p->child (every tree's union,
   plan_template) does not tell, so the rows are made here again.
   This is one copy's list; every copy's is the same hops in the same order (mesh.h, "Copies"; mesh_plan writes copy
   c's at [c n, c n + n)).  Counting (p->entry NULL) gives p->entries (n), p->receives, and each demand's received
   range (`range`, in the plan's demand order; p->ranges of them) and its pages landing here (`received`, P_g);
   filling writes p->entry[0 .. n).  Each receive takes the next field of its range's group, so a range's receives
   here are fields 0 .. P_g - 1 of its group; a send from the range's source names its release, any other send its
   receive's field (a relay's forward, or a destination's on the way on).
   2026-11-04, sub-assignment 3: reversing 2026-11-03's (sub-assignment 5's) bit a receive in one landed record an
   execution, counted over all of the execution's receives.  Why: a landing is a phase field of its range's group and
   an arrival on its range's count (C3.7; the contract's D1, counts per received range and copy).  What must stay: the
   order above, the same at every rank and in every copy, so both ends' lists agree; one receive a page at a node, so
   each field has one writer. */
static void plan_walk(struct plan *p){
  const uint32_t n=p->n,self=p->self;
  size_t first[(size_t)n+2],child[(size_t)n+1];
  uint64_t at=0,receives=0;
  if(!p->entry)p->ranges=p->consumed=0;
  for(uint32_t i=0;i<p->demands;i++){
    struct plan_demand *d=p->demand+i;
    uint32_t field=0;
    uint64_t k=0;
    for(uint32_t j=d->tree;j<d->tree+d->trees;j++){
      const uint64_t end=k+p->run[j];
      const struct plan_tree tree={p->parent+p->tree[j]*n,n,d->source};
      uint32_t in=MESH_ABSENT;
      mesh_graph_climb(tree.parent,n,d->source,self,plan_tree_up,&in);
      mesh_graph_rows(n+1,n,plan_tree_parent,&tree,first,child);
      if(in==MESH_ABSENT && first[self+1]==first[self]){k=end;continue;}
      for(;k<end;k++){
        /* the page's field: its receive's here; none at the source, whose sends its release's supplied word releases */
        uint32_t mine=MESH_ABSENT;
        if(in!=MESH_ABSENT){
          mine=field++;
          if(p->entry)p->entry[at]=(struct plan_entry){d->value,in,MESH_RECEIVE,i,mine,k};
          at++;
        }
        for(size_t c=first[self];c<first[self+1];c++,at++)
          if(p->entry)p->entry[at]=(struct plan_entry){d->value,(uint32_t)child[c],MESH_SEND,i,mine,k};
      }
    }
    if(!p->entry){
      d->received=field;
      d->range=field?p->ranges++:MESH_ABSENT;
      d->consumed=field && d->leaves>>self&1?p->consumed++:MESH_ABSENT;
      receives+=field;
    }
  }
  p->entries=at;
  if(!p->entry)p->receives=receives;
}

/* mesh-collective.h mesh_plan.  The demands: the declared transfers grouped by source and identity, a range to each of
   its destinations, ordered by `order` and then the declared order.  The routing: every demand at once over the map's
   node graph (a node pair an edge where both its records give it cables, of the lesser of their summed capacities;
   no cable, queue pair or link), a destination its source reaches by no path refused (mesh-graph.h mesh_graph_reach),
   by the concurrent-flow solve (mesh_graph_flow, Garg and Koenemann's multiplicative weights, the tag's mesh_solve
   rounds and room), demand d a tree rooted at its source reaching its destinations, its bytes down from the root: each
   demand's trees and their shares, a tree's vertices past its source and destinations the relays, members all.  The
   pages: each demand's pages cut over its trees by their shares by earliest finish (mesh_graph_fill), a tree's pages
   one run of them.  Every rank computes the same demands, trees, pages, template and u from the same declared
   transfers, dependences and map, and writes its own hops of them (plan_walk).
   Then, in the contract's order ("Writing a plan"):
   1. everything computed first: the template (plan_template), no lag-0 cycle (plan_acyclic), the least lags and u
      (plan_unroll), this node's entries of one copy counted and every room (ENOSPC where u x n passes hdr
      `hop_capacity`), the state section's layout (plan_layout), the items and both colourings (plan_place);
   2. the generation g = hdr `plan` + 1 (never 0), and the claim (mesh-dataflow.h mesh_claim_take): rows, the head and
      every value's count x P after it; window pages, the window colouring's blocks x block; local pages, the local
      one's.  ENOMEM leaves the previous plan standing;
   3. the state section started once (D3: its words 0, then struct mesh_state's `copies`, `receives`, `ranges`,
      `releases` and region offsets), every `state` 1 value's bytes 0 once (D3), every value's rows named and bound
      copy by copy (mesh_backing_bind), every value row's publication written; no data page is written;
   4. `hops` 0 (release), a release fence, `plan` woken;
   5. the entries, u x n; hdr `state`; `plan` = g;
   6. the map read again: ESTALE where its serial moved (or the reader's error): g's claim closed and every earlier one
      (the previous plan no longer stands), hdr `state` 0, `hops` 0;
   7. `hops` (release), `plan` woken;
   8. every earlier claim of this client closed (mesh_claim_close(context, g, 1)), which the bridge releases once no
      port holds its generation (D10).  Nothing waits for it.
   2026-11-04, sub-assignment 3: reversing 2026-11-03's (sub-assignments 3 and 5's) tail: `executions` as an input and
   its per-execution items; plan_section's check of sections mesh_section_create made (values now come in through
   `values`); the memset of every declared instance's pages before `hops`; the word-by-word zeroing of the supplied
   words and landed records; the call to mesh_plan_erased and its ETIMEDOUT; the layout's fixed areas of words.  Why:
   C4; a data page holds payload only and nothing is zeroed or reset while a plan stands (C3.7); a new plan's state is
   in new pages, so nothing is erased or waited for (C3.4).  What must stay: the demands, mesh_graph_reach,
   mesh_graph_flow and mesh_graph_fill as they were, so every rank's trees and template are the same; every failure but
   ESTALE and the second map read's found before `hops` is stored 0. */
int mesh_plan(struct mesh_ctx *context,const struct mesh_declared *declared,uint32_t count,struct mesh_value *values,
              uint32_t given,const struct mesh_dependence *dependence,uint32_t dependences){
  struct hdr *m=context->M;
  if((count && !declared) || (given && !values) || (dependences && !dependence))return EINVAL;
  /* a value: its bytes, its kind, and its wire and state flags */
  for(uint32_t j=0;j<given;j++){
    const struct mesh_value *v=values+j;
    if(!v->section.bytes || v->kind>MESH_EXECUTION || v->wire>1 || v->state>1 ||
       (v->state && (v->wire || v->kind!=MESH_INVARIANT)))return EINVAL;
    if(plan_blocks(m,v->section.bytes)>=UINT32_MAX/m->block)return ENOMEM;
  }
  /* a program's data edge: its two declared transfers, its kinds (no end both produced and its source's own read),
     both events at one node, a lag a least lag can hold (mesh_graph_lags: UINT32_MAX is no walk) */
  const uint32_t kinds=MESH_FROM_PRODUCED|MESH_TO_PRODUCED|MESH_FROM_SOURCE_READ|MESH_TO_SOURCE_READ;
  for(uint32_t e=0;e<dependences;e++){
    const struct mesh_dependence *x=dependence+e;
    const int from_produced=!!(x->kinds&MESH_FROM_PRODUCED),from_own=!!(x->kinds&MESH_FROM_SOURCE_READ),
              to_produced=!!(x->kinds&MESH_TO_PRODUCED),to_own=!!(x->kinds&MESH_TO_SOURCE_READ);
    if(x->from>=count || x->to>=count || x->kinds&~kinds || (from_produced && from_own) || (to_produced && to_own) ||
       x->lag==UINT32_MAX || plan_event_node(declared,x->from,from_produced,from_own)!=
       plan_event_node(declared,x->to,to_produced,to_own))return EINVAL;
  }
  struct mesh_nodes nodes,after;
  int status=mesh_nodes_read(context,&nodes);
  if(status)return status;
  const uint32_t n=nodes.count;
  if(nodes.self>=n)return EHOSTUNREACH;
  const uint64_t block=(uint64_t)m->block*m->pgsz;
  struct plan p={.context=context,.declared=declared,.given=values,.givens=given,.n=n,.self=nodes.self,
                 .serial=nodes.serial};
  struct plan_key *key=malloc(((size_t)count+1)*sizeof *key);
  uint32_t *root=NULL,*demand=NULL,*placed=malloc(((size_t)count+1)*sizeof *placed),
           *named=malloc(((size_t)given+1)*sizeof *named);
  size_t *trees=NULL;
  double *up=NULL,*down=NULL,*share=NULL,*rate=NULL,*latency=NULL;
  p.demand=calloc((size_t)count+1,sizeof *p.demand);
  p.of=malloc(((size_t)count+1)*sizeof *p.of);
  if(!key || !placed || !named || !p.demand || !p.of){status=ENOMEM;goto done;}
  p.named=named;
  for(uint32_t j=0;j<given;j++)named[j]=MESH_ABSENT;
  for(uint32_t i=0;i<count;i++){
    const struct mesh_declared *t=declared+i;
    if(t->source>=n || t->destination>=n){status=EHOSTUNREACH;goto done;}
    /* a transfer to its own source crosses no cable, and nothing is copied */
    if(t->source==t->destination || !t->bytes){status=EINVAL;goto done;}
    if(plan_blocks(m,t->bytes)>=UINT32_MAX/m->block){status=ENOMEM;goto done;}
    key[i]=(struct plan_key){t->source,t->identity,i};
  }
  qsort(key,count,sizeof *key,plan_key_order);
  for(uint32_t i=0;i<count;i++){
    const struct mesh_declared *t=declared+key[i].index;
    if(!i || key[i].source!=key[i-1].source || key[i].identity!=key[i-1].identity)
      p.demand[p.demands++]=(struct plan_demand){.source=t->source,.identity=t->identity,.order=t->order,
        .first=key[i].index,.mine=MESH_ABSENT,.bytes=t->bytes,.pages=plan_blocks(m,t->bytes),.range=MESH_ABSENT};
    struct plan_demand *d=p.demand+p.demands-1;
    p.of[key[i].index]=p.demands-1;
    /* one range: its bytes alike, each destination once */
    if(t->bytes!=d->bytes || d->leaves>>t->destination&1){status=EINVAL;goto done;}
    d->leaves|=UINT64_C(1)<<t->destination;
    if(t->order<d->order)d->order=t->order;
    /* `value`: MESH_ABSENT where this node is neither the transfer's source nor its destination; else an execution
       value in the window of the range's bytes, the source's one value of every transfer of the range, one range's
       alone */
    if(t->source!=p.self && t->destination!=p.self){
      if(t->value!=MESH_ABSENT){status=EINVAL;goto done;}
      continue;
    }
    if(t->value>=given){status=EINVAL;goto done;}
    const struct mesh_value *v=values+t->value;
    if(v->kind!=MESH_EXECUTION || !v->wire || v->section.bytes!=t->bytes ||
       (t->source==p.self && d->mine!=MESH_ABSENT && d->mine!=t->value) ||
       (named[t->value]!=MESH_ABSENT && named[t->value]!=d->first)){status=EINVAL;goto done;}
    named[t->value]=d->first;
    d->mine=t->value;
  }
  /* a local execution value's positions */
  for(uint32_t j=0;j<given;j++)
    if(named[j]==MESH_ABSENT && values[j].kind==MESH_EXECUTION && values[j].begin>values[j].end){status=EINVAL;goto done;}
  qsort(p.demand,p.demands,sizeof *p.demand,plan_demand_order);
  /* each declared transfer's demand in the plan's order: a demand's place before the sort is its first transfer's */
  for(uint32_t i=0;i<p.demands;i++)placed[p.of[p.demand[i].first]]=i;
  for(uint32_t i=0;i<count;i++)p.of[i]=placed[p.of[i]];
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
  /* 1. the template, no lag-0 cycle, the least lags and u; this node's entries of one copy counted and its rooms; the
     state section's layout; the items and both colourings: all before anything is claimed or written */
  if((status=plan_template(&p,dependence,dependences)) || (status=plan_acyclic(&p)) || (status=plan_unroll(&p)))
    goto done;
  plan_walk(&p);
  const uint32_t u=p.copies;
  uint64_t entries,rows=1;
  if(p.receives>=UINT32_MAX || __builtin_mul_overflow(p.entries,(uint64_t)u,&entries) || entries>m->hop_capacity){
    status=ENOSPC;goto done;
  }
  plan_layout(&p);
  if((status=plan_place(&p)))goto done;
  /* every value's rows: count x P, count u for a routed value and 1 for every other, after the claim's head */
  for(uint32_t j=0;j<given;j++)
    rows+=(named[j]!=MESH_ABSENT?(uint64_t)u:1)*plan_blocks(m,values[j].section.bytes);
  if(rows>=UINT32_MAX || p.window*m->block>=UINT32_MAX || p.outside*m->block>=UINT32_MAX){status=ENOMEM;goto done;}
  if(!(p.entry=malloc((p.entries+1)*sizeof *p.entry))){status=ENOMEM;goto done;}
  plan_walk(&p);

  /* 2. the generation and the claim: exactly the rows and pages this plan places, whole or not at all */
  const uint64_t previous=atomic_load_explicit(&m->plan,memory_order_relaxed);
  const uint64_t generation=previous+1?previous+1:1;
  if((status=mesh_claim_take(context,generation,(uint32_t)rows,(uint32_t)(p.window*m->block),
                             (uint32_t)(p.outside*m->block),&p.claim)))goto done;

  /* 3. the state section started once, where it is placed (D3): its words to `words` 0, then struct mesh_state's
     fields; nothing else of it is written by the plan, and no word of it is reset while the plan stands */
  const uint32_t at=plan_item_page(&p,p.state,0);
  const uint64_t base=m->data_off+(uint64_t)at*m->pgsz;
  memset(mesh_at(m,at),0,(size_t)p.words);
  struct mesh_state *state=(struct mesh_state *)mesh_at(m,at);
  state->copies=u;
  state->receives=(uint32_t)p.receives;
  state->ranges=p.consumed;
  state->releases=(uint32_t)p.releases;
  state->phases=base+p.phases;
  state->counts=base+p.counts;
  state->supplied=base+p.supplied;
  state->status=base+p.status;
  /* every value's rows, from the head's next, copy c's rows first + c x P + k bound to its copy's pages (an item's
     first page, plan_item_page), named clean as mesh_section_create named them (no device, no owner, no publication:
     every field of a row's publication 0 but its range's words below); a `state` 1 value's bytes 0 once, its one copy
     (D3: a local invariant outside the window alone); a routed value's copies its demand's routed value's here
     (plan_page), a relay's value no row's */
  uint32_t row=p.claim.rows.first+1;
  for(uint32_t j=0,other=0;j<given;j++){
    struct mesh_value *v=values+j;
    const struct plan_demand *d=named[j]!=MESH_ABSENT?p.demand+p.of[named[j]]:NULL;
    const uint32_t P=(uint32_t)plan_blocks(m,v->section.bytes),copies=d?u:1;
    v->section=(struct mesh_section){row,P*m->block,v->section.bytes,copies,P};
    for(uint32_t r=row;r<row+copies*P;r++){
      atomic_store_explicit(&mesh_page(m)[r].device,0,memory_order_relaxed);
      struct mesh_buffer *buffer=&mesh_buffers(m)[r];
      atomic_store_explicit(&buffer->owner,0,memory_order_relaxed);
      atomic_store_explicit(&buffer->closed,0,memory_order_relaxed);
      buffer->pages=buffer->constant=0;
      memset(mesh_publication_at(m,r),0,(size_t)m->target_stride);
    }
    for(uint32_t c=0;c<copies;c++){
      const uint32_t page=d?plan_page(&p,d->value,c,0):plan_item_page(&p,u*p.local+other,v->wire);
      mesh_backing_bind(context,row+c*P,P*m->block,page,c);
      if(v->state)memset(mesh_at(m,page),0,v->section.bytes);
    }
    if(!d)other++;
    /* a received range's first row: its pages, its copy-0 group and count word, the state section; a source range's:
       its supplied word, the one target a host or GPU producer stores into */
    struct mesh_publication *publication=mesh_publication_at(m,row);
    if(d && d->source==p.self){
      publication->sends=1;
      publication->targets[0]=(struct mesh_target){.stream=base+p.supplied+(uint64_t)d->release*sizeof(uint64_t),
                                                   .count=1,.stride=0};
      publication->state=base;
    }else if(d){
      publication->pages=d->received;
      publication->phases=base+p.phases+d->group;
      publication->count=base+p.counts+(uint64_t)d->consumed*u*sizeof(uint64_t);
      publication->state=base;
    }
    row+=copies*P;
  }

  /* 4. `hops` 0 (release), and a release fence that keeps every write below after it (review R1 round 2, S2: a release
     store orders what precedes it, not what follows), so a bridge that read `hops` and `plan`, then the entries, then
     them again after an acquire fence (mesh-flow.c link_prepare, plan_cancel) sees a change wherever it read a torn
     list; then `plan` woken, so the ports' progress threads, which wait on it, let the previous plan go now */
  atomic_store_explicit(&m->hops,0,memory_order_release);
  atomic_thread_fence(memory_order_release);
  os_sync_wake_by_address_all(&m->plan,sizeof m->plan,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
  /* 5. the entries, copy c's n at [c n, c n + n), each naming copy c's page and words (mesh.h struct mesh_hop): a
     receive its field's word and low bit in its range's copy-c group, its range's copy-c count word where this node
     consumes the range (0 at a relay, whose receive the bridge counts nothing for: review 15's finding 1) and P_g; a
     forwarding send its receive's word and field; a send from the range's source its release's supplied word */
  struct mesh_hop *hop=mesh_hops(m);
  for(uint32_t c=0;c<u;c++)for(uint64_t j=0;j<p.entries;j++){
    const struct plan_entry *e=p.entry+j;
    const struct plan_demand *d=p.demand+e->demand;
    struct mesh_hop h={.page=plan_page(&p,e->value,c,e->page),.node=e->node,.direction=e->direction,
                       .field=MESH_ABSENT,.serial=p.serial};
    if(e->field==MESH_ABSENT)h.word=base+p.supplied+(uint64_t)d->release*sizeof(uint64_t);
    else{
      h.word=base+p.phases+d->group+(uint64_t)c*mesh_group_bytes(d->received)+(uint64_t)e->field/32*sizeof(uint64_t);
      h.field=2*(e->field%32);
      if(e->direction==MESH_RECEIVE){
        if(d->consumed!=MESH_ABSENT)h.count=base+p.counts+((uint64_t)d->consumed*u+c)*sizeof(uint64_t);
        h.pages=d->received;
      }
    }
    *hop++=h;
  }
  /* hdr `state` this plan's section; `plan` the plan's generation, the map serial the plan was made against its
     entries' `serial` alone, so a second plan on an unchanged map is told from the first.  What must stay: the bridge
     refuses a plan whose entries' serial is not `map_serial` as it loads it */
  atomic_store_explicit(&m->state,base,memory_order_release);
  atomic_store_explicit(&m->plan,generation,memory_order_relaxed);
  /* 6. the map read again: where it moved, or the reader failed, no plan stands (`hops` 0, hdr `state` 0) and this
     claim and every earlier one of the client are closed */
  if(!(status=mesh_nodes_read(context,&after)) && after.serial!=p.serial)status=ESTALE;
  if(status){
    atomic_store_explicit(&m->state,0,memory_order_release);
    mesh_claim_close(context,generation,0);
    mesh_claim_close(context,generation,1);
    goto done;
  }
  /* 7. `hops`; the bridge's progress threads watch `plan` with bounded waits (CONTRACT.md "Wave-2 interfaces") */
  atomic_store_explicit(&m->hops,entries,memory_order_release);
  os_sync_wake_by_address_all(&m->plan,sizeof m->plan,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
  /* 8. every earlier claim of this client closed, released by the bridge once no port holds its generation (D10) */
  mesh_claim_close(context,generation,1);
done:
  free(key);free(placed);free(named);free(p.demand);free(p.of);
  free(root);free(up);free(down);free(demand);free(p.parent);free(share);
  free(p.tree);free(p.run);free(rate);free(latency);free(trees);
  free(p.child);free(p.value);free(p.read);free(p.here);free(p.first);free(p.head);free(p.lag);free(p.between);
  free(p.others);free(p.colour);free(p.entry);
  return status;
}
