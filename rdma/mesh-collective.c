#include "mesh-collective.h"
#include "mesh-graph.h"
#include <float.h>
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
   range's first row names (its publication's first target, which mesh_plan writes), after the caller's writes to the
   range in that execution.  One store releases every send of the range's pages, and the word only rises while the
   caller's executions run in order.  A row that is no source range's first names no target (`sends` 0), and nothing is
   stored.
   2026-11-04, sub-assignment 3: reversing 2026-11-03's form (sub-assignment 5), MESH_SUPPLIED stored into execution t's
   own word at the target's word + t x MESH_WORD_STRIDE.  Why: one word serves every execution, its value saying which,
   and no word is reset while its plan stands (C3.7).  What must stay: one store, release, after the range's last write.
   2026-11-05 (design/aot-compilation-change-request.md, appendix item 7, "A publish waits for nothing": "the publish
   is done in that event's notification"; design/compilation-coordination.md, "The revisions of e4a7398e", TP-3): the
   word raised to execution + 1 where it is below it (a compare-and-swap loop, release, that never lowers it), where it
   was stored.  Why: a torch program's publishes run in torch MPSEvents' notifications (rdma/mesh-torch.mm), and
   "Notifications may come in any order, so the count only rises" (mesh fba41159), so execution t's may run after t + 1's;
   C3.7: "supplied words ... only rise".  The loop retries only where another publish of the range raised the word
   meanwhile, and ends once the word is at least execution + 1.  What must stay: one release store raising the word,
   after the range's last write. */
void mesh_host_publish(struct mesh_ctx *context,struct mesh_section section,uint64_t execution){
  struct hdr *m=context->M;
  const struct mesh_publication *publication=mesh_publication_at(m,section.first);
  if(!publication->sends)return;
  _Atomic uint64_t *word=(_Atomic uint64_t *)((char *)m+publication->targets[0].word);
  uint64_t seen=atomic_load_explicit(word,memory_order_relaxed);
  while(seen<execution+1 &&
        !atomic_compare_exchange_weak_explicit(word,&seen,execution+1,memory_order_release,memory_order_relaxed));
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
  uint32_t source,identity,first;         /* first: its first declared transfer, which numbers it (GO5) */
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
  uint32_t set;                           /* 1 where its range is a set: every declared transfer of it MESH_PAGES
                                             (design/compilation-coordination.md, S1), each tree's pages one set */
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
  /* its count words' row here, among every received range's (plan_layout): `consumed` where this node consumes it,
     else the consumed ranges' number plus its place among the relayed ones, in the plan's demand order (MESH_ABSENT
     where none of its pages lands here; design/compilation-coordination.md, "The revisions of abecd389", RC) */
  uint32_t counted;
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
   same page; MESH_ABSENT for a send from the range's source, which its release's supplied word releases.
   `communication`: the page's crossing of the tree edge the hop is (plan_communication), one instruction of the
   plan's instruction DAG (plan_schedule; design/compilation-coordination.md, "The documents' revisions of edd4b583",
   GO1); `place`: that instruction's place in the global topological order (GO3), the entry's place in its list.
   `set` (S1, S2): j + 1 where its demand's range is a set and its page is on tree j (the solve's trees grouped by
   demand, p->tree's index), else 0.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 67ca86f8", SP): `at` (its group's least `place`,
   the place of the set's first page, GO4) is gone.  Why: design/plan-to-bridge-change-request.md R5, "A set's pages
   ... keep their places in that order.  The set's random permutation is drawn over those places.  (Corrected
   2026-11-05: the first text moved a set's pages to consecutive places, which nothing requires.)"  What must stay:
   every other field as plan_walk assigns it.
   2026-11-04, owner C3 (GO4): `walk` (the hop's place in plan_walk's order, T6's tie), `start` (its link's scheduled
   start of its page, T6), `earliest` (its group's least start) and `least` (its group's least walk, S3) are gone,
   `place` and `at` in their place.  Why: each cable's list is a subsequence of one global topological order built as
   MSCCLang §5.2 states in its step 3, and "Scheduled start times do not order the lists"
   (design/plan-to-bridge-change-request.md, R5), which overturns T6 and S3.  What must stay: every other field as
   plan_walk assigns it.
   2026-11-05, owner C3 (design/compilation-coordination.md, "The documents' revisions of edd4b583", "Amendments
   during the work (2026-11-05)", GO4′: "`walk` goes with T6's ties"): the removal of `walk` cited to the ruling. */
struct plan_entry { uint32_t value,node,direction,demand,field,communication,set,place; uint64_t page; };
struct plan {
  struct mesh_ctx *context;
  const struct mesh_declared *declared;
  uint32_t transfers;                     /* the declared transfers, declared[0 .. transfers) */
  struct mesh_value *given;               /* the caller's values, `givens` of them */
  uint32_t givens;
  uint32_t *named;                        /* named[j]: the first declared transfer of the demand whose range values[j]
                                             is here, MESH_ABSENT where none (a local value) */
  struct plan_demand *demand;             /* the demands, by their first declared transfer (GO5) */
  uint32_t demands,n,self,copies;         /* copies: u (plan_unroll) */
  uint64_t serial;
  uint32_t *of;                           /* of[i]: declared[i]'s demand, p->demand[of[i]] */
  /* the node graph (design/compilation-coordination.md, T2): pair e joins from[e] < to[e], of capacity[e], the lesser
     of the two directions' map capacities; `pairs` of them */
  uint32_t pairs,from[MESH_NODES_MOST*(MESH_NODES_MOST-1)/2+1],to[MESH_NODES_MOST*(MESH_NODES_MOST-1)/2+1];
  double capacity[MESH_NODES_MOST*(MESH_NODES_MOST-1)/2+1];
  size_t *tree;                           /* tree[j] the solve's tree, demand by demand (mesh_graph_rows) */
  uint32_t *parent;                       /* the solve's parents, tree by tree */
  uint64_t *run;                          /* run[j]: tree[j]'s pages of its demand */
  uint32_t made;                          /* the solve's trees, j < made */
  /* each tree's communications (plan_trees): members[j] its vertices but its root (a set of map indices), lead[j] its
     demand's first page on it, comm[j] its first communication; `comms` of them in all */
  uint64_t *members,*lead,*comm,comms;
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
  /* the state section (plan_layout), offsets from its start: `hops` its entries', `words` its end (2026-11-05, "The
     revisions of 46ae8fdf", LP9: `sequences`, the sequence words', gone) */
  uint64_t phases,counts,supplied,status,hops,words;
  /* 2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", PS and PQ): the plan's two queues in
     the section, `deliveries` (room `room` struct mesh_delivery) and `submits` (room `outstanding` struct
     mesh_submission), between the status and the entries.
     2026-11-05 ("The revisions of 3f106747", BS): the submissions' queue's room `submit_room`, and `room` and it the
     queues' self-timed buffer bounds (plan_rooms), `outstanding` A10's bound alone; the synchronization edge into this
     node (`waits` its source's map index, `wait_delay` its delay) and out of it (`tells` its sink's), MESH_ABSENT none */
  uint64_t deliveries,submits,room,outstanding,submit_room,wait_delay;
  uint32_t waits,tells;
  /* the items (plan_place): routed copies (a, c) at item a x u + c, then this node's other values (others[k] the index
     in `values` of item local x u + k; a value of lag k >= 1 k consecutive items, its copies in order), then the state
     section, item `state` */
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
/* 2026-11-04, owner C (design/compilation-coordination.md, T6): plan_demand_order (lower `order` first, a transfer
   that blocks a pipeline stage, then the declared order) is removed with struct mesh_declared's `order`.  Why: the
   order of every node pair's sends is the static schedule's (design/plan-to-bridge-change-request.md, R5;
   plan_schedule), not a hand priority.  What must stay: the demands numbered alike at every rank, now by their first
   declared transfer (mesh_plan). */
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
     however the pages travel (a range is produced before it is sent and before its source reads it; a page lands before
     it is forwarded, is sent before it lands, and lands before it is consumed; the program's edges are its own
     claims).  Grouped into rows by tail (mesh_graph_rows), so every rank
     numbers them alike.
   - The routed values, at every node, demand by demand and a demand's by ascending node: its source's, and each node
     of `on`'s (a destination's or a relay's), each with its write and reads (struct plan_value); this node's among them
     (`here`, and each demand's `value`).  Every node's, since u is the largest L over the collective's values
     (design/disjoint-pages-change-request.md, C3.3: "The unroll factor u is the largest L"), which every rank computes
     alike.
   - Every walk of this template into a write passes the write's range's production P: a landing's write W(y) has one
     edge in, P -> W(y), and the edges into P are the program's dependences alone.  That is the half of
     design/compilation-coordination.md, U1, which this template holds (plan_unroll).
   2026-11-04, owner C2 (design/compilation-coordination.md, U1): u's citation, the contract's D5, is C3.3's.  Why: D5
     appears in neither Lam, Rau nor C3.3's text (design/aot-compilation-change-request.md, appendix item 1).  What
     must stay: u every node's routed values', so every rank computes the same u.
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
   PLDI 1988; B. R. Rau, Iterative modulo scheduling, MICRO 1994; design/compilation-coordination.md, U1, and
   design/disjoint-pages-revision-coordination.md, "The allocator").  The least lags D(x -> y) over the template
   (mesh-graph.h mesh_graph_lags, Dijkstra with the lags as weights), from every routed value's write and reads, at
   every node, to every routed value's write:
   - L(v) = the largest D(r -> write v) over v's reads r, the least lag at which every read of v in execution t happens
     before its write in t + L (C3.3: "For each value, L is the smallest lag at which every read of it in execution t
     happens before its write in execution t + L"); UINT32_MAX (a read that reaches no later write of it) is ENOMEM
     naming the value (C3.3: "A value with no such L fails the plan with ENOMEM, naming the value"), and nothing else
     is;
   - u = the largest L(v) over every routed value at every node, 1 where there is none (C3.3: "The unroll factor u is
     the largest L").  Every rank computes the same template, lags and u, so both nodes of every adjacent pair hold the
     same copies;
   - between[a x local + b] = D_ab, the largest D(r -> write b) over this node's routed value a's reads r, for every
     pair of this node's routed values, so the colouring's tests (plan_interferes) are table reads.
   The least lags are C3.3's happens-before at every larger distance too (U1): every walk of the template into a write
   passes its range's production P (plan_template, the plan's own); and, the premise the least lags rest on, a property
   of the program's dependences and not of the plan (mesh-collective.h struct mesh_dependence), every event of a node
   in execution t precedes each of its events in t + 1 by the program's own edges, as both programs give them
   (tools/e2b_replay.m e2b_ir's rank chains, "r's last to its first at lag 1"; rdma/mesh_compile.py _dependences,
   "every event no lag-0 edge leaves to every event none enters (lag 1)"), so a production reaches its next instance
   at lag 1 and a walk at lag L through P continues through P's next instances at every lag past L.  This function
   does not test the premise (design/aot-compilation-change-request.md, appendix item 1: ENOMEM refuses only a value
   with no such L).  D(w -> w) == 0, a lag-0 cycle, plan_acyclic has refused.
   2026-11-04, owner C2 (design/compilation-coordination.md, U1; design/aot-compilation-change-request.md, appendix
   item 1): the refusal of a write whose least cycle D(w -> w) is not 1 (design/disjoint-pages-revision-coordination.md,
   D5), ENOMEM naming its value, is gone.  Why: C3.3 fails only a value with no such L, and D5 appears in neither Lam,
   Rau nor C3.3's text; it refused a lone AllGather or AllToAll, whose landing reaches its next instance at lag 2 (u 2
   now).  What must stay: L(v), u, between and plan_interferes as they were; ENOMEM, named, for a read that reaches no
   later write.
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
     2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", RC): ranges x u words, every
     received range's row its `counted` (the consumed ranges' first, at their `consumed`, then the relayed ones), where
     a relay's receives had none.  Why: design/aot-compilation-change-request.md, appendix item 5: "The code must name
     a page for round j by that count, in place of PM2's test of the phase field", the count "each range's landed
     count of each copy, which every landing raises"; a relay's receive is named for its rounds as a destination's is
     (mesh-flow.c link_receive_drain).  What must stay: a consumed range's row its `consumed`, `ranges` its number;
   - `supplied`: releases uint64 words, release r's at index r;
   - `status`: one struct mesh_status;
   - `deliveries`: the plan's delivered calls' queue here, `room` struct mesh_delivery, and `submits`, the program's
     submissions' queue, `outstanding` struct mesh_submission (mesh.h struct mesh_state's notes; 2026-11-05, "The
     revisions of abecd389", PQ and PS: design/aot-compilation-change-request.md, appendix item 2, "It is laid out with
     the plan's pages when the plan is placed, and claimed and released with them (C3.2, C3.4)");
   - `hops`: this node's entries, u x n struct mesh_hop, copy c's n at [c n, c n + n) (design/compilation-
     coordination.md, "The region", struct mesh_state: "the entries; each 64-aligned"); [0, hops) the bytes mesh_plan
     starts (D3), the entries written whole after them;
   - `words`: the end of the entries, the section's bytes.
   Every offset in struct mesh_state is a region offset: the section's own plus these.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", LP9): reversing DL1's sequence words
   (u uint64 words after the status, L. Lamport's "Message Buffer" B[0 .. u - 1]): the entries follow the status
   again, and struct mesh_state `sequences` is 0.  Why: design/aot-compilation-change-request.md, appendix item 2 at
   mesh 0e56d8da: "publish them to each port through the channel above, in place of the sequence words (`sequences`),
   `taken[p]` and the parent's wait while they are full (Lamport's buffer; the coordination's DL1)"; "How the bridge
   holds the channel is the bridge's own".  What must stay: every offset before the status's end as it was; every part
   64-aligned.
   2026-11-04, owner C3 (DL1): the sequence words placed after the status and before the entries, where the entries
   followed the status.  Why: the parent's delivery publishes each released execution's sequence number in its plan's
   state, one word per copy, each word one writer (appendix item 2, "Delivery at a node").  What must stay: every
   offset before the sequence words as it was; every part 64-aligned.
   2026-11-04, owner C (design/compilation-coordination.md, "The region" and Q5): the entries in the section, where they
   were hdr's hop area (`hop_off`, room `hop_capacity`).  Why: any number of plans stand at once, each in its own claim
   (design/plan-to-bridge-change-request.md, R8).  What must stay: the words' offsets before the entries as they
   were. */
static uint64_t plan_align(uint64_t at){ return (at+63)/64*64; }
static void plan_layout(struct plan *p){
  uint64_t at=sizeof(struct mesh_state);
  p->phases=at;
  at=0;
  uint32_t relayed=p->consumed;
  for(uint32_t i=0;i<p->demands;i++){
    struct plan_demand *d=p->demand+i;
    d->counted=d->range==MESH_ABSENT?MESH_ABSENT:d->consumed!=MESH_ABSENT?d->consumed:relayed++;
    if(d->range==MESH_ABSENT)continue;
    d->group=at;
    at+=(uint64_t)p->copies*mesh_group_bytes(d->received);
  }
  p->counts=plan_align(p->phases+at);
  p->supplied=plan_align(p->counts+(uint64_t)p->ranges*p->copies*sizeof(uint64_t));
  p->status=plan_align(p->supplied+p->releases*sizeof(uint64_t));
  p->deliveries=plan_align(p->status+sizeof(struct mesh_status));
  p->submits=plan_align(p->deliveries+p->room*sizeof(struct mesh_delivery));
  p->hops=plan_align(p->submits+p->submit_room*sizeof(struct mesh_submission));
  p->words=p->hops+p->entries*p->copies*sizeof(struct mesh_hop);
}

/* the colouring's order (the contract's "The allocator"): greedy by size, an item's blocks largest first, then its
   index, which is the value order (this node's routed values in demand order, then its other values in `values`'s
   order, then the state section) and, within a routed value or a value of lag k >= 1, the copy */
struct plan_sized { uint64_t blocks; uint32_t item; };
static int plan_sized_order(const void *a,const void *b){
  const struct plan_sized *x=a,*y=b;
  if(x->blocks!=y->blocks)return x->blocks>y->blocks?-1:1;
  return (x->item>y->item)-(x->item<y->item);
}
/* mesh_graph_left_edge's interference, the contract's rule (two items interfere unless it lets them share):
   - an invariant (the state section among them) interferes with every other item;
   - a copy of a value of lag k >= 1 (mesh-collective.h struct mesh_value `lag`) interferes with every other item, its
     own value's other copies among them: its write is ordered by an event of the program's, a host producer's, not by
     an edge of the template, so no order the plan knows lets it share a page (design/disjoint-pages-change-request.md,
     D.3.3; design/disjoint-pages-rulings-coordination.md, R4);
   - two local execution values of lag 0 share only where their [begin, end] on this node's program order do not
     overlap (this node's work runs in that order, execution after execution); a local execution value interferes with
     every routed copy;
   - routed copies (v, c) and (w, c'), D_vw = between[v x local + w], d = (c' - c) mod u: where d > 0 they share only
     if d >= D_vw and u - d >= D_wv; where d == 0 (v != w) only if D_vw <= u, D_wv <= u and D_vw == 0 or D_wv == 0.
     These are the distances at which their executions meet (t and t' with t' - t = d mod u: d, d - u, d + u, ...);
     an order at the least such distance holds at every larger one of its class (design/compilation-coordination.md,
     U1: every walk into a write passes its range's production P; and, the premise the least lags rest on, the
     program's dependences' and untested (mesh-collective.h struct mesh_dependence), each node's events of execution t
     precede those of t + 1, so a walk at lag L through P continues at every lag past L; plan_unroll), and a value's
     every read before the other's write orders its writes too.  Two copies of one value
     follow the d > 0 rule.
   Symmetric: swapping the items swaps D_vw and D_wv and takes d to u - d.
   2026-11-04, owner C2 (U1): this order at every larger distance cited to U1, where it cited the contract's D5.  Why:
   D5's refusal is gone (design/aot-compilation-change-request.md, appendix item 1).  What must stay: the tests as they
   are, the least lags' at the distances the copies' executions meet (C3.3). */
static int plan_interferes(const void *context,uint32_t a,uint32_t b){
  const struct plan *p=context;
  const uint32_t u=p->copies,routed=p->local*u;
  if(a>=routed || b>=routed){
    if(a<routed || b<routed || a==p->state || b==p->state)return 1;
    const struct mesh_value *x=p->given+p->others[a-routed],*y=p->given+p->others[b-routed];
    if(x->kind==MESH_INVARIANT || y->kind==MESH_INVARIANT || x->lag || y->lag)return 1;
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
     one each but a value of lag k >= 1, which has k, its copies, P blocks each (Lam's modulo variable expansion: a
     value live across k executions gets k copies; R4); the state section (plan_layout's `words` in blocks), one,
     invariant, no rows.
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
   2026-11-04, sub-assignment 2 (design/disjoint-pages-rulings-coordination.md, R4): reversing one copy for every value
   but a routed one.  A value of lag k >= 1 (a host producer's input, ordered by the program's events) is k items, its
   copies, each interfering with every other item, all outside the window.  Why: the per-step host inputs had an
   instance per step, sized by the run (design/disjoint-pages-revision-coordination.md, D12); D.3.3 gives them copies
   by the lag the program declares, as modulo variable expansion does for any value, so nothing bound grows with the
   steps.  What must stay: a value of lag 0 one item as before; the items' order; ENOMEM before anything is written.
   0; ENOMEM. */
static int plan_place(struct plan *p){
  struct hdr *m=p->context->M;
  const uint32_t u=p->copies;
  uint64_t routed,others=0;
  if(__builtin_mul_overflow((uint64_t)p->local,(uint64_t)u,&routed) || routed>=UINT32_MAX)return ENOMEM;
  /* this node's values no declared transfer names, in `values`'s order, a value of lag k its k copies in copy order */
  for(uint32_t j=0;j<p->givens;j++)
    if(p->named[j]==MESH_ABSENT && (others+=p->given[j].lag?p->given[j].lag:1)>=UINT32_MAX)return ENOMEM;
  if(routed+others+1>=UINT32_MAX || !(p->others=malloc(((size_t)others+1)*sizeof *p->others)))return ENOMEM;
  p->other=0;
  for(uint32_t j=0;j<p->givens;j++)if(p->named[j]==MESH_ABSENT)
    for(uint32_t c=0;c<(p->given[j].lag?p->given[j].lag:1);c++)p->others[p->other++]=j;
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

/* Each tree's communications (design/compilation-coordination.md, T4: "a (page, link) is one communication"; the
   instructions after the tasks, "The documents' revisions of edd4b583", GO1 and GO3, numbered here for the order's
   ties).  A tree carries a page over each of its edges once (mesh.h, "A tree carries a page over a node pair once at
   most"), so one communication is a (page, tree edge): the tree's page `page` (its place among the demand's pages on
   that tree, from 0) over the edge into vertex `head`.  They are numbered tree by tree in the solve's order grouped by
   demand, then page by page, then by the head's place among the tree's vertices but its root (`members`, from its row
   of the tree's vertices keyed by parent, mesh_graph_rows, as plan_walk reads a tree; no walk over a parent table
   here, C9), the same at every rank.  Every one lies on some destination's route (plan_schedule): mesh_graph_flow
   makes a tree of its leaves' paths to its root, and its leaves are the demand's destinations.  0; ENOMEM where they
   pass 32 bits (each is instruction tasks + c, below 32 bits: plan_schedule; 2026-11-05, "The revisions of 46ae8fdf",
   LP: where it cited mesh_graph_contention_schedule's `communication`, which is removed) or the call's own memory. */
static int plan_trees(struct plan *p){
  const uint32_t n=p->n;
  size_t first[(size_t)n+2],child[(size_t)n+1];
  p->members=malloc(((size_t)p->made+1)*sizeof *p->members);
  p->lead=malloc(((size_t)p->made+1)*sizeof *p->lead);
  p->comm=malloc(((size_t)p->made+1)*sizeof *p->comm);
  if(!p->members || !p->lead || !p->comm)return ENOMEM;
  p->comms=0;
  for(uint32_t i=0;i<p->demands;i++){
    const struct plan_demand *d=p->demand+i;
    uint64_t k=0;
    for(uint32_t j=d->tree;j<d->tree+d->trees;j++){
      const struct plan_tree tree={p->parent+p->tree[j]*n,n,d->source};
      mesh_graph_rows(n+1,n,plan_tree_parent,&tree,first,child);
      p->members[j]=0;
      for(size_t c=first[0];c<first[n];c++)p->members[j]|=UINT64_C(1)<<child[c];
      p->lead[j]=k;
      p->comm[j]=p->comms;
      k+=p->run[j];
      p->comms+=p->run[j]*(uint64_t)__builtin_popcountll(p->members[j]);
      if(p->comms>=UINT32_MAX)return ENOMEM;
    }
  }
  return 0;
}
/* tree j's page `page` (from 0 on that tree) over its edge into `head`, a member of the tree (plan_trees) */
static uint32_t plan_communication(const struct plan *p,uint32_t j,uint64_t page,uint32_t head){
  return (uint32_t)(p->comm[j]+page*(uint64_t)__builtin_popcountll(p->members[j])+plan_rank(p->members[j],head));
}

/* Every page of the plan in one order, the same at every rank: the demands by their first declared transfer, a
   demand's pages tree by tree.  A tree carries a page over a node pair once at most, so visiting this node's hops page
   by page in that order, for a pair a -> b, a's sends to b and b's receives from a are the same hops in the same order
   (mesh.h "Both ends").  A page's hops here: its receive from this node's parent on the tree (mesh-graph.h
   mesh_graph_climb's first step), then its sends to this node's children (its row of the tree's vertices keyed by
   parent, mesh_graph_rows), in their order.  The tree's edges are the graph module's: no walk over a parent table
   here (the C9 reviews of c5bffd0c and c9ffedcf).  A page's children are its own tree's, which p->child (every tree's
   union, plan_template) does not tell, so the rows are made here again.
   This is one copy's list; every copy's is the same hops in the same order (mesh.h, "Copies"; mesh_plan writes copy
   c's at [c n, c n + n)).  Counting (p->entry NULL) gives p->entries (n), p->receives, and each demand's received
   range (`range`, in the plan's demand order; p->ranges of them) and its pages landing here (`received`, P_g);
   filling writes p->entry[0 .. n).  Each receive takes the next field of its range's group, so a range's receives
   here are fields 0 .. P_g - 1 of its group; a send from the range's source names its release, any other send its
   receive's field (a relay's forward, or a destination's on the way on).  Filling also gives each entry its
   communication (plan_communication: a receive's the tree edge into this node, a send's the edge into its child), and
   its `set`: j + 1 for a page on tree j of a demand whose range is a set (its every declared transfer MESH_PAGES), j
   the index this walk iterates, d->tree .. d->tree + d->trees into p->tree and p->run, the same at every rank; 0 for
   every other entry (design/compilation-coordination.md, S1 and S2: the set's unit is one tree's pages, every edge of
   tree j carrying the same run of them).  plan_schedule then orders the entries by their places in the global
   topological order of the instructions (design/plan-to-bridge-change-request.md R5; design/compilation-
   coordination.md, "The documents' revisions of edd4b583", GO3), and every field stays as assigned here.
   2026-11-04, owner C3 (GO4): no entry records its place in this order (`walk`, T6's tie and S3's least walk, gone
   with them).  What must stay: the order above, in which tree j's entries of one node and direction are one run
   (plan_schedule's groups).
   2026-11-05 (design/compilation-coordination.md, "The revisions of 67ca86f8", SP): plan_schedule forms no groups now
   (R5: a set's pages "keep their places in that order"), so that run is no group's.  What must stay: every field as
   assigned here; the order above, the same at every rank and in every copy.
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
      const uint32_t set=d->set?j+1:0;
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
          if(p->entry)p->entry[at]=(struct plan_entry){.value=d->value,.node=in,.direction=MESH_RECEIVE,.demand=i,
                                                       .field=mine,
                                                       .communication=plan_communication(p,j,k-p->lead[j],self),
                                                       .set=set,.page=k};
          at++;
        }
        for(size_t c=first[self];c<first[self+1];c++,at++)
          if(p->entry)p->entry[at]=(struct plan_entry){.value=d->value,.node=(uint32_t)child[c],.direction=MESH_SEND,
                                                       .demand=i,.field=mine,
                                                       .communication=plan_communication(p,j,k-p->lead[j],
                                                                                         (uint32_t)child[c]),
                                                       .set=set,.page=k};
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

/* -- the task graph and its schedule (design/plan-to-bridge-change-request.md, R3 to R5) --------------------------- */

/* mesh_graph_climb's visit, the route of a page down its tree from the root to `below` (plan_schedule): the vertex
   met at step s >= 1 is the parent of the one met at step s - 1, and the tree edge between them is the route's link
   depth - s, in route order from the root; `head` that link's lower vertex.  A visit of mesh_graph_climb recording the
   edges it visits, as plan_tree_up does, so it stays here (design/compilation-coordination.md, C9 rulings); the route
   is the tree path the fill gave the page (T1) */
struct plan_path { const uint32_t *link_of; uint32_t n,depth,below,*link,*head; };
static void plan_path_step(void *consumer,uint32_t at,uint32_t step){
  struct plan_path *path=consumer;
  if(step){
    path->link[path->depth-step]=path->link_of[(size_t)at*path->n+path->below];
    path->head[path->depth-step]=path->below;
  }
  path->below=at;
}
/* an event's tasks (plan_schedule; design/compilation-coordination.md, T1: the tasks are the template's events, a
   consumption one task for MESH_JOIN and one a page for MESH_PAGES, design/plan-to-bridge-change-request.md, R4), its
   first and how many: P(d) and S(d) one each, a consumption one or its range's pages */
struct plan_span { uint32_t first,count; };
static struct plan_span plan_tasks(const struct plan *p,const struct plan_span *consumer,const uint32_t *reader,
                                   uint32_t i,int produced,int own){
  if(produced)return (struct plan_span){p->of[i],1};
  if(own)return (struct plan_span){reader[p->of[i]],1};
  return consumer[i];
}
/* design/plan-to-bridge-change-request.md R5, "The order of each cable's list, by the reference algorithm": the
   entries by their own place (`place`), each list a subsequence of the one global topological order, a set's pages
   keeping their places in it.  A total order, so every rank's sort ends alike: each communication has one place, and
   no two of this node's entries are one communication (a receive's is its tree edge into this node, each send's the
   edge into its own child).
   2026-11-05 (design/compilation-coordination.md, "The revisions of 67ca86f8", SP): reversing GO4's first key, the
   entry's group's least place (`at`), by which a set's entries on one node pair and direction took consecutive places
   at the place of its first page.  Why: R5, "A set's pages (design/aot-compilation-change-request.md, appendix item
   5) keep their places in that order.  The set's random permutation is drawn over those places.  (Corrected
   2026-11-05: the first text moved a set's pages to consecutive places, which nothing requires.)"  What must stay: a
   total order of the entries both ends of a pair agree on.
   2026-11-04, owner C3 (GO4): reversing T6 as S3 amended it (the group's least scheduled start, then its least walk,
   then the entry's walk).  Why: each cable's list is a subsequence of MSCCLang §5.2 step 3's global topological order,
   "Scheduled start times do not order the lists", and a set's pages "take consecutive places at the place of the
   set's first page in that order" (design/plan-to-bridge-change-request.md, R5).  What must stay: a total order of the
   entries both ends of a pair agree on; a set's entries contiguous on their node pair and direction within each copy
   (the bridge's EINVAL where they are not: design/compilation-coordination.md, S7). */
static int plan_entry_order(const void *a,const void *b){
  const struct plan_entry *x=a,*y=b;
  return (x->place>y->place)-(x->place<y->place);
}
/* 2026-11-05 (design/compilation-coordination.md, "The revisions of 67ca86f8", SP): plan_entry_side (mesh_graph_rows'
   keys, an entry's node pair and direction, by which plan_schedule found each set's group and its least place) is gone
   with the groups.  Why: R5's corrected text, at plan_entry_order.  What must stay: nothing of it */

/* The task graph, its schedule and the order of the lists (design/plan-to-bridge-change-request.md, R3 to R5;
   design/aot-compilation-change-request.md, A6; design/compilation-coordination.md, T1, T2 and T4, and "The
   documents' revisions of edd4b583", GO1 to GO5; "The revisions of 46ae8fdf", LP), one execution's, every rank's the
   same from the same declared transfers, weights, dependences and map.
   - The task graph G = (V, E, w, c) (O. Sinnen, Task Scheduling for Parallel Systems, Wiley, 2007; O. Sinnen and L. A.
     Sousa, IEEE TPDS 16(6) 2005, Definition 1), T1: its tasks the template's events (plan_template), each on the node
     its event is at (the placement operand: the declared transfers' sources and destinations).  Task d is demand d's
     production at its source, P(d), weighing the weighting operand's `produced` (design/aot-compilation-change-
     request.md, A3); then each demand's source read, S(d), where a dependence names it, weighing `read`; then each
     declared transfer's consumption at its destination, in declared order, one task weighing `consumed` for MESH_JOIN
     (function A, design/plan-to-bridge-change-request.md, R4) and one a page, each weighing `consumed`, for MESH_PAGES
     (function B).  Its edges, in this order: for each declared transfer in declared order, page by page, page k's
     edge from P to the consumption (to page k's task for MESH_PAGES), of cost `page`, its route the tree path the
     fill gave page k, from the source down to the destination (mesh_graph_climb, plan_path_step); each P(d) -> S(d);
     each dependence of lag 0 in the dependences' order, from each task of its `from` event to each of its `to`
     event's (T1: an edge out of a MESH_PAGES consumption leaves from each of its page tasks; design/compilation-
     coordination.md, T1': an edge into one enters each of them, each page's task the function's work on that page
     after its local predecessors).  A local edge (P -> S, a lag-0 dependence) has no route and costs 0 (T1': Sinnen
     and Sousa's classic model, local communication free, Definition 2).
   - The topology graph (Sinnen and Sousa, Definition 4), T2: the node graph's pairs, pair e two directed links, 2e
     from[e] -> to[e] and 2e + 1 back, each of relative data rate b(L) the pair's capacity over the greatest pair's.
   - T4: every route entry of one page over one tree edge is one communication (plan_communication), one instruction
     on its link.  Every tree edge lies on some destination's route (plan_trees).
   - The instructions (GO1): the task graph's tasks as numbered above, 0 .. tasks - 1, then the plan's communications
     (T4: a page over a tree edge; plan_trees and plan_communication number them), communication c at tasks + c.  A
     communication is one instruction: its send at the edge's upper node and its receive at its lower node take its
     one place, so a node pair's sends at one end and its receives at the other are the same hops in the same order
     (mesh.h, "Both ends"; R1: "the bridge's records name the same pages").  MSCCLang (M. Cowan et al., ASPLOS 2023)
     §4.2 expands a remote copy "into a send and a receive instruction" joined by "a communication edge"; joining them
     in one instruction is where this differs from it, so that both ends' lists agree (GO1).  The dependencies: each
     local edge of the task graph as it is (P(d) -> S(d), a lag-0 dependence); each remote edge a chain, its tail task
     -> its route's first communication -> each next -> its last -> its head task: MSCCLang §4.2's communication and
     processing edges, a page sent before it lands and landed before it is forwarded or consumed (plan_template's P ->
     W(y), W(y) -> F(y, c), L(z) -> C(z)).  Routes sharing a prefix (T4) give a dependency more than once, which
     adds no row (mesh-graph.h mesh_graph_completion_times: each edge once, LP4) and changes no order
     (mesh_graph_heap_order: an edge given twice counts twice and is removed twice).  Every rank builds the same
     instructions and dependencies, in the same order.
   - The priority (R5 at mesh 0e56d8da, "How: the program"): "one variable C_j, a completion time, for each instruction
     j of one execution's task graph (R3): each task at its node, and each hop of each page's route, on its cable
     (design/compilation-coordination.md GO2)"; "Instruction j's length d_j is its weight from the weighting operand
     (aot-compilation A3): w for a task, and for a hop c(e) over its cable's bandwidth".  A task's d_j is its weight
     (A3); communication c's on link L is c(e) / b(L) (b(L) T2's rate), the greatest over the route entries of the
     task graph's edges e that carry it (T4: one instruction for every edge that shares it), one correctly rounded /
     each and the max by comparisons, as GO2 took it.  Its resource (R5: "A resource is a cable in one direction, or
     a node's device"): a task's its node's device, numbered by the node's map index; a communication's its link L in
     one direction (T2), numbered n + L.  The rows are R5's, over these instructions and the dependencies above
     (mesh-graph.h mesh_graph_completion_times: Chen's (14), (14a), (1b), (1c), Hall et al.'s (5) on each resource by
     cutting planes, each linear program by HiGHS); the priority is each instruction's C̄_j in its solution.
   - The order (R5, "The order of each cable's list, by the reference algorithm"): MSCCLang §5.2, step 3: "Sort
     instructions into a global topological order with respect to their dependencies with a heap using the priority to
     order instructions" (mesh-graph.h mesh_graph_heap_order), "the least C̄_j first.  Where two are equal, the heap
     takes the lower instruction" (R5; given as -C̄, LP7), the same at every rank (F. B. Schneider, Implementing
     fault-tolerant services using the state machine approach, ACM Comput. Surv., 1990).  That order "is consistent
     with the precedence relation, as Hall et al.'s ties require" (R5): the heap places a vertex only after every
     vertex an edge into it leaves.
   - The lists (R5, "The order of each cable's list, by the reference algorithm"): each of this node's entries
     (plan_walk; every field as it assigns them) takes its communication's place (`place`), and the entries are sorted
     by it (plan_entry_order), a total order.  A set's entries (S1, S2: its nonzero `set`, tree j's pages over a node
     pair) keep their places like every other entry (R5: "A set's pages ... keep their places in that order.  The
     set's random permutation is drawn over those places"); the bridge draws it (mesh-flow.c link_prepare and
     position_record).  So each node pair's list at each end is a subsequence of the one global order, both ends'
     alike (a send at x to y and its receive at y are one communication, at one place), and every copy's the same
     (mesh_plan writes this one order for each copy).
   - Where it is a topological order.  The global order is a topological order of the instructions (GO3: mesh-graph.h
     mesh_graph_heap_order), each list a subsequence of it, and every lag points forward, so each cable's sequence of
     executions in that order is one (R5: "That order is a topological order of the unrolled executions, since every
     lag points forward").  Its dependencies are the bridge's store-and-forward waits (a forward released by its
     receive's landing, a source's send by its production: GO1), so in that order the cables' sequences cannot wait on
     each other in a cycle (MSCCLang §5.2: "By assigning instructions to thread blocks in a topological order that
     respects communication and processing edges, all implicit dependencies introduced by thread block sequential
     execution cannot produce cycles").  Each execution's one permutation of a set's every position on a pair, each
     cable carrying its own members in its order (design/aot-compilation-change-request.md, appendix item 5: "Every
     edge and every cable of the set's tree carries its share of the set's pages in that one permutation's order";
     mesh-flow.c), changes which page each of those places carries.  The item's argument for the permuted sequences,
     "each cable's sequence is the same renaming of them", holds where one cable carries every position of the set on
     each pair of its tree, and is not shown where the deal splits a set's positions over a pair's cables
     (design/compilation-coordination.md, "The revisions of 46ae8fdf", stop 3; 2026-11-05: this said the documents
     state no argument, "The revisions of 67ca86f8", stop 3).
   2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", LP; design/plan-to-bridge-change-
   request.md R5 at mesh 0e56d8da): reversing GO2's and GO3's priority (each instruction's bottom level by Sinnen and
   Sousa's eq. (7), the greatest first) and GO5's call of the contention schedule (Sinnen and Sousa's Algorithm 1 with
   Algorithm 2, whose starts ordered nothing), with their `start`, `link_start`, `level` and `delay`.  Why: R5, "Each
   instruction's priority is C̄_j, its completion time in the program's optimal solution, the least first"; "It
   replaces MSCCLang's depth and reverse depth, and edge scheduling's start times, which are no longer computed for the
   order.  Edge scheduling's part, one communication at a time on each link, is now the resources' rows."  What must
   stay: GO1's instructions and dependencies; GO2's lengths; every entry's fields as plan_walk assigns them; both
   ends' lists of a node pair the same hops in the same order; every copy's the same order.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 67ca86f8", SP): reversing GO4's groups, a set's
   entries on one node pair and direction at consecutive places at the place of its first page (its least place,
   found in one pass over the entries' rows by node and direction, mesh_graph_rows), the entries sorted by their
   group's least place, then their own; and with them GO4's and GO4′'s argument that the groups in order of their
   least places were a topological order with each group whole, which made any permutation within a group one too.
   Why: R5, "keep their places in that order.  The set's random permutation is drawn over those places.  (Corrected
   2026-11-05: the first text moved a set's pages to consecutive places, which nothing requires.)"  What must stay:
   both ends' lists of a node pair the same hops in the same order; every copy's the same order.
   2026-11-04, owner C3 (GO1 to GO5; design/plan-to-bridge-change-request.md, R5, "The order of each cable's list, by
   the reference algorithm"): reversing T6 as S3 amended it, each entry at its link's scheduled start of its page
   (`start`, through comm_start, each communication's one start on its link), each group's least start (`earliest`)
   and least walk (`least`), the entries sorted by them and then by their walk.  Why: R5 (the operator: "what
   algorithm is cited here, if any? no citations? use the reference algorithm instead") orders each cable's list as a
   subsequence of MSCCLang §5.2 step 3's global topological order by bottom level, and "Scheduled start times do not
   order the lists"; it overturns T6 and S3 (design/compilation-coordination.md, "The documents' revisions of
   edd4b583").  What must stay: the schedule's call (GO5); every entry's fields as plan_walk assigns them; both ends'
   lists of a node pair the same hops in the same order; every copy's the same order; a set's entries contiguous on
   their node pair and direction within each copy (S7).
   2026-11-04, owner C3 (GO4; review W3 #6): the claim owner C2 withdrew here (the start order not shown to be a
   topological order of the bridge's store-and-forward waits, the contention schedule's (14) being nonaligned,
   cut-through) is answered above: the lists' order is now the instruction DAG's, whose dependencies are those waits,
   not the schedule's starts (GO4: "This answers the open item 'No cycle on a cable' (W3 #6)").
   0; EINVAL, EDOM or ENOMEM (mesh_graph_completion_times': a program with no solution, HiGHS's failure; and
   mesh_graph_heap_order's EINVAL, a cycle); ENOMEM. */
static int plan_schedule(struct plan *p,const struct mesh_weight *weight,const struct mesh_dependence *dependence,
                         uint32_t dependences){
  const uint32_t n=p->n,demands=p->demands,count=p->transfers;
  const struct mesh_declared *declared=p->declared;
  uint32_t link_of[(size_t)n*n+1],path_link[(size_t)n+1],path_head[(size_t)n+1];
  uint32_t *reader=malloc(((size_t)demands+1)*sizeof *reader);
  struct plan_span *consumer=malloc(((size_t)count+1)*sizeof *consumer);
  if(!reader || !consumer){free(reader);free(consumer);return ENOMEM;}
  /* the tasks, the edges and their route entries counted (each below 32 bits, as the template's vertices) */
  uint64_t tasks=demands,edges=0,entries=0;
  int status=0;
  for(uint32_t d=0;d<demands;d++)
    if((reader[d]=p->demand[d].own?(uint32_t)tasks:MESH_ABSENT)!=MESH_ABSENT)edges++,tasks++;
  if(tasks>=UINT32_MAX)status=ENOMEM;
  for(uint32_t i=0;!status && i<count;i++){
    const struct plan_demand *d=p->demand+p->of[i];
    consumer[i]=(struct plan_span){(uint32_t)tasks,declared[i].kind==MESH_PAGES?(uint32_t)d->pages:1};
    tasks+=consumer[i].count;
    edges+=d->pages;
    for(uint32_t j=d->tree;j<d->tree+d->trees;j++)
      if(p->run[j])entries+=p->run[j]*mesh_graph_climb(p->parent+p->tree[j]*n,n,d->source,declared[i].destination,
                                                       NULL,NULL);
    if(tasks>=UINT32_MAX || edges>=UINT32_MAX || entries>=UINT32_MAX)status=ENOMEM;
  }
  for(uint32_t k=0;!status && k<dependences;k++){
    const struct mesh_dependence *x=dependence+k;
    if(x->lag)continue;
    const struct plan_span a=plan_tasks(p,consumer,reader,x->from,!!(x->kinds&MESH_FROM_PRODUCED),
                                        !!(x->kinds&MESH_FROM_SOURCE_READ)),
                           b=plan_tasks(p,consumer,reader,x->to,!!(x->kinds&MESH_TO_PRODUCED),
                                        !!(x->kinds&MESH_TO_SOURCE_READ));
    if((edges+=(uint64_t)a.count*b.count)>=UINT32_MAX)status=ENOMEM;
  }
  /* the instructions (GO1): the tasks, then the communications; their dependencies: each edge's route entries and one
     more */
  const uint64_t instructions=tasks+p->comms,dependencies=edges+entries;
  if(instructions>=UINT32_MAX)status=ENOMEM;
  double *w=status?NULL:malloc(((size_t)instructions+1)*sizeof *w),
         *cost=status?NULL:malloc(((size_t)edges+1)*sizeof *cost),
         *rate=malloc((2*(size_t)p->pairs+1)*sizeof *rate),
         *completion=status?NULL:malloc(((size_t)instructions+1)*sizeof *completion);
  uint32_t *resource=status?NULL:malloc(((size_t)instructions+1)*sizeof *resource),
           *from=status?NULL:malloc(((size_t)edges+1)*sizeof *from),
           *to=status?NULL:malloc(((size_t)edges+1)*sizeof *to),
           *link=status?NULL:malloc(((size_t)entries+1)*sizeof *link),
           *communication=status?NULL:malloc(((size_t)entries+1)*sizeof *communication),
           *before=status?NULL:malloc(((size_t)dependencies+1)*sizeof *before),
           *after=status?NULL:malloc(((size_t)dependencies+1)*sizeof *after),
           *place=status?NULL:malloc(((size_t)instructions+1)*sizeof *place);
  size_t *route=status?NULL:malloc(((size_t)edges+1)*sizeof *route);
  if(!status && (!w || !cost || !rate || !completion || !resource || !from || !to || !link || !communication ||
                 !before || !after || !place || !route))status=ENOMEM;
  if(!status){
    /* the tasks: each on its event's node, weighing the weighting operand's weight of it (A3), its resource its node's
       device, resource index its node's map index (R5: "A resource is a cable in one direction, or a node's device") */
    for(uint32_t d=0;d<demands;d++){
      const struct plan_demand *x=p->demand+d;
      w[d]=weight[x->first].produced;
      resource[d]=x->source;
      if(reader[d]!=MESH_ABSENT){w[reader[d]]=weight[x->first].read;resource[reader[d]]=x->source;}
    }
    for(uint32_t i=0;i<count;i++)for(uint32_t q=0;q<consumer[i].count;q++){
      w[consumer[i].first+q]=weight[i].consumed;
      resource[consumer[i].first+q]=declared[i].destination;
    }
    /* the links (T2) */
    double most=0;
    for(size_t v=0;v<(size_t)n*n;v++)link_of[v]=MESH_ABSENT;
    for(uint32_t e=0;e<p->pairs;e++)if(p->capacity[e]>most)most=p->capacity[e];
    for(uint32_t e=0;e<p->pairs;e++){
      link_of[(size_t)p->from[e]*n+p->to[e]]=2*e;
      link_of[(size_t)p->to[e]*n+p->from[e]]=2*e+1;
      rate[2*e]=rate[2*e+1]=p->capacity[e]/most;
    }
    /* the edges, their costs and routes */
    size_t e=0,r=0;
    for(uint32_t i=0;i<count;i++){
      const struct plan_demand *d=p->demand+p->of[i];
      const uint32_t z=declared[i].destination;
      for(uint32_t j=d->tree;j<d->tree+d->trees;j++){
        if(!p->run[j])continue;
        const uint32_t *parent=p->parent+p->tree[j]*n;
        struct plan_path path={link_of,n,mesh_graph_climb(parent,n,d->source,z,NULL,NULL),z,path_link,path_head};
        mesh_graph_climb(parent,n,d->source,z,plan_path_step,&path);
        for(uint64_t q=0;q<p->run[j];q++,e++){
          from[e]=p->of[i];
          to[e]=consumer[i].first+(declared[i].kind==MESH_PAGES?(uint32_t)(p->lead[j]+q):0);
          cost[e]=weight[i].page;
          route[e]=r;
          for(uint32_t s=0;s<path.depth;s++,r++){
            link[r]=path_link[s];
            communication[r]=plan_communication(p,j,q,path_head[s]);
          }
        }
      }
    }
    for(uint32_t d=0;d<demands;d++)
      if(reader[d]!=MESH_ABSENT){from[e]=d;to[e]=reader[d];cost[e]=0;route[e++]=r;}
    for(uint32_t k=0;k<dependences;k++){
      const struct mesh_dependence *x=dependence+k;
      if(x->lag)continue;
      const struct plan_span a=plan_tasks(p,consumer,reader,x->from,!!(x->kinds&MESH_FROM_PRODUCED),
                                          !!(x->kinds&MESH_FROM_SOURCE_READ)),
                             b=plan_tasks(p,consumer,reader,x->to,!!(x->kinds&MESH_TO_PRODUCED),
                                          !!(x->kinds&MESH_TO_SOURCE_READ));
      for(uint32_t s=0;s<a.count;s++)for(uint32_t t=0;t<b.count;t++){
        from[e]=a.first+s;to[e]=b.first+t;cost[e]=0;route[e++]=r;
      }
    }
    route[e]=r;
  }
  if(!status){
    /* GO1: each edge of the task graph a chain of dependencies from its tail task through its route's communications,
       in route order, to its head task (a local edge, of no route, one dependency from its tail to its head) */
    size_t k=0;
    for(size_t x=0;x<edges;x++){
      uint32_t tail=from[x];
      for(size_t r=route[x];r<route[x+1];r++,k++){before[k]=tail;tail=after[k]=(uint32_t)tasks+communication[r];}
      before[k]=tail;after[k++]=to[x];
    }
    /* GO2: communication c's length the greatest c(e) / b(L) over the route entries carrying it, from 0 (each such
       quotient is > 0, every `page` weight and rate being > 0, and every communication lies on a route, plan_trees),
       its resource its route entries' link L in one direction, n + L (R5: "for a hop c(e) over its cable's
       bandwidth"; every route entry of one communication is on one link, its tree edge's: plan_path_step) */
    for(uint64_t c=0;c<p->comms;c++){w[tasks+c]=0;resource[tasks+c]=n;}
    for(size_t x=0;x<edges;x++)for(size_t r=route[x];r<route[x+1];r++){
      const double occupy=cost[x]/rate[link[r]];
      if(occupy>w[tasks+communication[r]])w[tasks+communication[r]]=occupy;
      resource[tasks+communication[r]]=n+link[r];
    }
    /* R5's program over the instructions, its rows the dependencies (mesh-graph.h mesh_graph_completion_times), then
       the global topological order by the heap, the least C̄ first as -C̄ (LP7), a tie to the lower instruction */
    status=mesh_graph_completion_times((uint32_t)instructions,w,resource,n+2*p->pairs,(size_t)dependencies,before,after,
                                       completion);
    if(!status){
      for(uint64_t j=0;j<instructions;j++)completion[j]=-completion[j];
      status=mesh_graph_heap_order((uint32_t)instructions,completion,(size_t)dependencies,before,after,place);
    }
  }
  if(!status){
    /* R5: each entry at its communication's place, a set's among them, sorted by it (2026-11-05, "The revisions of
       67ca86f8", SP: the groups' least places gone) */
    for(uint64_t j=0;j<p->entries;j++)p->entry[j].place=place[tasks+p->entry[j].communication];
    qsort(p->entry,p->entries,sizeof *p->entry,plan_entry_order);
  }
  free(reader);free(consumer);free(w);free(cost);free(rate);free(completion);
  free(resource);free(from);free(to);free(link);free(communication);free(before);free(after);free(place);free(route);
  return status;
}

/* The queues' rooms and the plan's synchronization edges, fixed at compile time by the self-timed buffer bound
   (design/aot-compilation-change-request.md, appendix item 2, "Its room is fixed at compile time, by the self-timed
   buffer bound"; S. S. Bhattacharyya, S. Sriram and E. A. Lee, Optimizing synchronization in multiprocessor DSP
   systems, IEEE Trans. Signal Processing 45(6), 1997, here BSL; design/compilation-coordination.md, "The revisions of
   3f106747", BS).  The graph is the plan's part of the program's IPC graph, built as BSL §4.1 builds one ("if actors v1,
   v2, …, vn are assigned to the same processor in that order, then Gipc would have a cycle ((v1, v2), (v2, v3), …, (vn −
   1, vn), (vn, v1)), with delay((vn, v1)) = 1"), the processors the program's nodes (`program`, its least s the
   sender):
   - its actors: the program's submission at s, σ (vertex 0); s's bridge's TO-broadcast of the plan's calls, β (vertex
     1: item 2, "Its source is the sender's TO-broadcast of the plan's calls, and its sink is the bridge's taking of them
     at the node"); and node v's part of the plan's call, x_v, for each node v of the plan, `nodes` (vertex 2 + v's rank
     in it; "The actors are each node's parts of the program's calls, in program order");
   - its edges: σ -> β, delay 0, the submissions' queue; β -> x_v, delay 0, the delivered calls' queue at v ("these
     queues"); at s, σ -> x_s, delay 0, and x_s -> σ, delay `outstanding` (s's cycle in program order: "At each node,
     A10's bound is the delay on the edge from a call's completion to the program's later submission"), or, where s is
     no node of the plan, σ -> σ of that delay; x_v -> x_v of that delay at every other node; and each transfer, a tree
     edge y -> c carrying pages of a demand (`child`), as x_y -> x_c and x_c -> x_y, each of delay u ("The IPC edges are
     the plans' transfers, each with C3.3's u copies as its delay");
   - its times: x_v's t(x_v) the weighting operand's weights of v's tasks in one execution (each production at its
     source, each source read, each consumption at a destination, a page's task each for MESH_PAGES) and of its hops'
     pages, sent or received (BSL §4.1: "t(v) includes the time taken to execute all IPC operations (sends and
     receives) that the actor v performs"; "Execution times t(x) are A3's weighting operand"), a demand's page weight
     the greatest of its transfers' (GO2's); t(σ) = t(β) = 0.
   Then: ENOMEM where a cycle has zero total delay (BSL Lemma 1, "Every cycle C in the IPC graph has a path delay of at
   least one if and only if the static schedule it is constructed from is free of deadlock"; item 2: "or where a cycle
   has no delay"); the edges Convert-to-SC-graph adds (§9, Fig. 6; mesh-graph.h mesh_graph_convert_sc) with their
   delays from DetermineDelays (Fig. 8; mesh_graph_determine_delays), so that every queue "becomes encapsulated in an
   SCC" (§9); and each queue's room its edge's bound in the converted graph, Theorem 2's "Bfb(e) = ρ_Gs(snk(e), src(e))
   + delay(e)", ρ the least total delay over walks from snk(e) to src(e) (BSL §3; mesh-graph.h mesh_graph_lags,
   Dijkstra's, as the theorem's "Dijkstra's algorithm"): `room` ρ(x_self, β), `submit_room` ρ(β, σ); a node of the
   program that is no node of the plan, whose queue no message reaches (Dest(m) is `nodes`), `room` 1.  ENOMEM where a
   queue lies on no cycle after the conversion, or a room passes its words.
   Decisions where the documents are silent (design/compilation-coordination.md, "The revisions of 3f106747"):
   - BS-2: the program's submission σ and its bridge's TO-broadcast β are two actors, so the submissions' queue between
     them, of PS-3, is an IPC edge with its own bound; t(σ) = t(β) = 0, A3 weighing neither, so σ, lowest of the least,
     is the source SCC's link, and the sink-source edge's sink is the program's start at s;
   - BS-3: the graph's part of the program's IPC graph is the plan's own, which is what its compilation holds when it
     is planned (each function is compiled at its first call); every cycle C containing e bounds the tokens on e by
     Delay(C) ("The number of tokens in any cycle of the IPC graph is always conserved", Lemma 2), so (5) over a
     subgraph's cycles bounds it over the program's;
   - BS-4: each added edge's source count is its node's program's starts of the plan (`submissions`), which trail that
     node's part's ends by the delay `outstanding` on its own edge, so in the rooms each added edge's delay is
     DetermineDelays' plus `outstanding`, the constraint the bridge's count implements (mesh-records.c counts_tell);
   - BS-5: a plan with no declared transfer has `nodes` {s} in the graph at every rank, where its `nodes` is each
     rank's own (mesh_plan), so every rank builds one graph.
   At most one added edge enters and one leaves each node's vertices: σ reaches every vertex, so σ's SCC is the one
   source SCC and Convert-to-SC-graph chains the sink SCCs alone, each link the one vertex x_v of its node, and its
   sink-source edge enters σ, its SCC's least time and lowest vertex.  So `waits` and `tells` name at most one each.
   Every rank computes the same graph, delays and edges from the same trees, u, weights and operands, the graph module's
   arithmetic IEEE-exact.  Allocations before any output; ENOMEM */
static uint32_t plan_room_tail(const void *context,size_t k){ return ((const uint32_t *)context)[k]; }
/* an edge of the IPC graph, a -> b of delay d, appended */
struct plan_room_edges { uint32_t *from,*to; uint64_t *delay; size_t edges; };
static void plan_room_edge(struct plan_room_edges *graph,uint32_t a,uint32_t b,uint64_t d){
  graph->from[graph->edges]=a;graph->to[graph->edges]=b;graph->delay[graph->edges]=d;graph->edges++;
}
struct plan_room_rows { const size_t *order; const uint32_t *to; const uint64_t *delay; };
static uint32_t plan_room_head(const void *context,size_t k){
  const struct plan_room_rows *rows=context;
  return rows->to[rows->order[k]];
}
static uint32_t plan_room_lag(const void *context,size_t k){
  const struct plan_room_rows *rows=context;
  const uint64_t delay=rows->delay[rows->order[k]];
  return delay>=UINT32_MAX?UINT32_MAX:(uint32_t)delay;
}
static int plan_rooms(struct plan *p,const struct mesh_weight *weight,uint64_t program,uint64_t outstanding,
                      uint64_t spanned){
  const uint32_t n=p->n,s=(uint32_t)__builtin_ctzll(program),u=p->copies;
  const uint64_t nodes=p->demands?spanned:UINT64_C(1)<<s;
  const uint32_t members=(uint32_t)__builtin_popcountll(nodes),V=2+members;
  uint64_t hops=0;
  for(uint32_t i=0;i<p->demands;i++)hops+=p->demand[i].forwards;
  const size_t room=3+2*(size_t)members+2*(size_t)hops;
  double *t=calloc(V,sizeof *t),*page=calloc((size_t)p->demands+1,sizeof *page);
  uint32_t *from=malloc((room+2*(size_t)V)*sizeof *from),*to=malloc((room+2*(size_t)V)*sizeof *to),
           *vertex=malloc(((size_t)n+1)*sizeof *vertex);
  uint64_t *delay=malloc((room+2*(size_t)V)*sizeof *delay);
  uint32_t *added_from=malloc(2*(size_t)V*sizeof *added_from),*added_to=malloc(2*(size_t)V*sizeof *added_to);
  uint64_t *added_delay=malloc(2*(size_t)V*sizeof *added_delay);
  size_t *first=malloc(((size_t)V+2)*sizeof *first),*order=malloc((room+2*(size_t)V+1)*sizeof *order);
  int status=!t || !page || !from || !to || !vertex || !delay || !added_from || !added_to || !added_delay || !first ||
             !order?ENOMEM:0;
  if(status)goto done;
  for(uint32_t v=0,k=2;v<n;v++)vertex[v]=nodes>>v&1?k++:UINT32_MAX;
  struct plan_room_edges graph={from,to,delay,0};
  plan_room_edge(&graph,0,1,0);
  for(uint32_t v=0;v<n;v++)if(vertex[v]!=UINT32_MAX){
    plan_room_edge(&graph,1,vertex[v],0);
    if(v!=s)plan_room_edge(&graph,vertex[v],vertex[v],outstanding);
  }
  if(vertex[s]!=UINT32_MAX){plan_room_edge(&graph,0,vertex[s],0);plan_room_edge(&graph,vertex[s],0,outstanding);}
  else plan_room_edge(&graph,0,0,outstanding);
  for(uint32_t i=0;i<p->demands;i++)for(uint32_t y=0;y<n;y++)
    for(uint64_t below=p->child[(size_t)i*n+y];below;below&=below-1){
      const uint32_t c=(uint32_t)__builtin_ctzll(below);
      plan_room_edge(&graph,vertex[y],vertex[c],u);plan_room_edge(&graph,vertex[c],vertex[y],u);
    }
  size_t edges=graph.edges;
  for(uint32_t i=0;i<p->transfers;i++){
    const uint32_t d=p->of[i];
    if(weight[i].page>page[d])page[d]=weight[i].page;
    t[vertex[p->declared[i].destination]]+=weight[i].consumed*(p->declared[i].kind==MESH_PAGES?(double)p->demand[d].pages:1.0);
  }
  for(uint32_t i=0;i<p->demands;i++){
    const struct plan_demand *d=p->demand+i;
    t[vertex[d->source]]+=weight[d->first].produced+(d->own?weight[d->first].read:0.0);
    for(uint32_t j=d->tree;j<d->tree+d->trees;j++){
      const uint32_t *parent=p->parent+(size_t)p->tree[j]*n;
      for(uint32_t c=0;c<n && p->run[j];c++)
        if(c!=d->source && parent[c]<n && parent[c]!=c){
          t[vertex[parent[c]]]+=(double)p->run[j]*page[i];
          t[vertex[c]]+=(double)p->run[j]*page[i];
        }
    }
  }
  double mean;
  uint32_t added;
  if((status=mesh_graph_cycle_mean(V,t,edges,from,to,delay,&mean)))goto done;
  if(mean==INFINITY){status=ENOMEM;goto done;}
  if((status=mesh_graph_convert_sc(V,t,edges,from,to,&added,added_from,added_to)) ||
     (status=mesh_graph_determine_delays(V,t,edges,from,to,delay,added,added_from,added_to,added_delay))){
    if(status==EDEADLK || status==ERANGE)status=ENOMEM;
    goto done;
  }
  p->waits=p->tells=MESH_ABSENT;p->wait_delay=0;
  for(uint32_t a=0;a<added;a++){
    uint32_t tail=s,head=s;
    for(uint32_t v=0;v<n;v++){if(vertex[v]==added_from[a])tail=v;if(vertex[v]==added_to[a])head=v;}
    if(head==p->self){p->waits=tail;p->wait_delay=added_delay[a];}
    if(tail==p->self)p->tells=head;
    if(__builtin_add_overflow(added_delay[a],outstanding,&delay[edges])){status=ENOMEM;goto done;}
    from[edges]=added_from[a];to[edges]=added_to[a];edges++;
  }
  mesh_graph_rows(V,edges,plan_room_tail,from,first,order);
  const struct plan_room_rows rows={order,to,delay};
  const uint32_t source[2]={1,vertex[p->self]==UINT32_MAX?1:vertex[p->self]},target[2]={0,1};
  uint32_t lags[4];
  if((status=mesh_graph_lags(V,first,plan_room_head,plan_room_lag,&rows,2,source,2,target,lags)))goto done;
  p->submit_room=lags[0*2+0];
  p->room=vertex[p->self]==UINT32_MAX?1:lags[1*2+1];
  if(p->submit_room==UINT32_MAX || p->room==UINT32_MAX || !p->submit_room || !p->room ||
     p->room>=UINT64_MAX/(4*sizeof(struct mesh_delivery)))status=ENOMEM;
done:
  free(t);free(page);free(from);free(to);free(vertex);free(delay);free(added_from);free(added_to);free(added_delay);
  free(first);free(order);
  return status;
}
/* mesh-collective.h mesh_plan.  The demands: the declared transfers grouped by source and identity, a range to each of
   its destinations, numbered by their first declared transfer (design/compilation-coordination.md, "The documents'
   revisions of edd4b583", GO5: "The demands stay numbered by their first declared transfer (T6's other half, not an
   order of the lists)").  The routing:
   every demand at once over the map's node graph (a node pair an edge where the map gives it a capacity both ways, of
   the lesser of the two: mesh-dataflow.h struct mesh_nodes `capacity`, D.2's node pairs and their capacities from the
   map, and nothing else of how the bridge carries pages), a destination its source reaches by no path refused
   (mesh-graph.h mesh_graph_reach), by the concurrent-flow solve (mesh_graph_flow, Garg and Koenemann's multiplicative
   weights, the tag's mesh_solve rounds and room), demand d a tree rooted at its source reaching its destinations, its
   bytes down from the root: each demand's trees and their shares, a tree's vertices past its source and destinations
   the relays, members all.  The pages: each demand's pages cut over its trees by their shares by earliest finish
   (mesh_graph_fill), a tree's pages one run of them.  The order: one execution's task graph scheduled (plan_schedule;
   design/plan-to-bridge-change-request.md, R3 to R5), and its instructions, the tasks and the communications, in one
   global topological order by their completion times in R5's linear program, the least first (MSCCLang §5.2, step 3;
   R5 at mesh 0e56d8da), this node's entries in it, a set's at their own places (design/compilation-coordination.md,
   "The documents' revisions of edd4b583", GO1; "The revisions of 46ae8fdf", LP; "The revisions of 67ca86f8", SP: R5,
   "keep their places in that order").  The sets: a
   demand whose every declared transfer is MESH_PAGES is a set on each of its trees (S1, S2; its `set`).  Every rank
   computes the same demands, trees, pages, template, u, schedule and order from the same declared transfers, weights,
   dependences and map, and writes its own hops of them (plan_walk, plan_schedule).
   The plan's name (design/compilation-coordination.md, N1): (group, function), `function` the program's ordinal of
   the compiled function, held in struct mesh_state `epoch`; `*plan` the plan's handle once it stands (N2).
   Then, in the order of the draft e6bd7db2:design/plan-bridge-coordination.md, §6, "Each plan in its claim", steps 1
   to 5, as the operator's ruling 1 allows it (design/compilation-coordination.md, Q5, as N3 and N5 amend it):
   1. everything computed first: the trees' communications (plan_trees), the template (plan_template), no lag-0 cycle
      (plan_acyclic), the least lags and u (plan_unroll), this node's entries of one copy counted and every room
      (ENOMEM where the receives, the entries or the rows pass their words), the state section's layout with the
      entries in it (plan_layout), the items and both colourings (plan_place), the entries filled (plan_walk) and in
      their places' order (plan_schedule, R5), the plan's nodes and sender; EEXIST where a plan standing at this
      node on the same map is already (group, function), the claims' heads walked (mesh-dataflow.h mesh_head_after and
      mesh_head_state);
   2. the claim (mesh-dataflow.h mesh_claim_take), its generation from hdr `serial` (the draft's §6, step 2: one past
      its count, never 0), its head's `start` the take's: rows, the head and every value's count x P after it (count u
      for a routed value, k for a value of lag k >= 1, 1 for every other); window pages, the window colouring's blocks
      x block; local pages, the local one's.  ENOMEM leaves every plan of the program standing;
   3. the state section started once (D3: its bytes before the entries 0, then struct mesh_state's `copies`,
      `receives`, `ranges`, `releases`, region offsets, `group`, `epoch` (the function, N1), `serial`, `hops`,
      `entries`, `nodes` and `sender`; `held`, every `refused`, `submitted`, `released`, every `taken` and `sequences`
      0 (design/compilation-coordination.md, "The documents' revisions of edd4b583", DL1; "The revisions of
      46ae8fdf", LP9: no sequence words, appendix item 2 at mesh 0e56d8da), every
      `state` 1 value's bytes 0 once (D3), every value's rows named and bound copy by copy (mesh_backing_bind), every
      value row's publication written, the entries u x n in plan_schedule's order, each with its `set` (S1, S2); no
      data page is written;
   4. the map read again: where its serial moved, or the reader failed, this claim alone closed
      (mesh_claim_close(context, generation), which the bridge releases once no port holds it, D10), `*plan`
      untouched: ESTALE, or the reader's error (N3);
   5. the head's publication `state` the section's region offset (release): the plan stands; `*plan` that offset (N2);
      hdr `plans` raised and woken (mesh.h mesh_notify: "every publication of a plan (mesh_plan) ... adds 1").  No
      earlier claim of this client is closed (N3, N5): its other plans stand, each until the program retires it.
   2026-11-04, owner C2 (design/compilation-coordination.md, N1 to N5; design/aot-compilation-change-request.md,
   appendix item 6): reversing C-epoch's computation (the epoch one past the context's count, `mine`, the epoch kept
   from a cancelled standing plan, context->epoch raised once the plan stood and left at the named epoch less 1 on a
   failure after the writes), context->state, step 4's close of every earlier claim of the client, and step 6 (every
   earlier claim of this client closed once the plan stood).  Why: compiled functions coexist, each its own plan with
   its own claim, released only when the program retires that function or itself (R. Frostig, M. J. Johnson and C.
   Leary, SysML 2018; NVIDIA, CUDA Programming Guide §4.2.2; Pathways §3, §4.4); a plan is named by the program's
   ordinal of its function, the same at every rank; struct mesh_ctx holds no plan.  What must stay: every failure but
   ESTALE and the second map read's found before anything is written; the map read again after everything is written
   and before the plan stands; `plans` raised and woken once the plan stands.
   2026-11-04, owner C (design/compilation-coordination.md, "The region" and Q5): reversing the writing order of
   design/disjoint-pages-revision-coordination.md, "Writing a plan": the generation hdr `plan` + 1; hdr `hops` 0, a
   release fence and `plan` woken before the entries; the entries in hdr's hop area, ENOSPC past `hop_capacity`; hdr
   `state` and `plan` stored with them; `hops` stored, and `plan` woken, after the second map read.  Why: any number of
   plans stand at once, each in its own claim and found by its head (design/plan-to-bridge-change-request.md, R8;
   design/aot-compilation-change-request.md, A7), and hdr's one-plan fields are vacant.  What must stay: every failure
   but ESTALE and the second map read's found before anything is written; the map read again after everything is
   written and before the plan stands; the earlier claims closed only once the new plan stands (that last gone with
   step 6, 2026-11-04, owner C2: N3, N5, above).
   2026-11-04, sub-assignment 3: reversing 2026-11-03's (sub-assignments 3 and 5's) tail: `executions` as an input and
   its per-execution items; plan_section's check of sections mesh_section_create made (values now come in through
   `values`); the memset of every declared instance's pages before `hops`; the word-by-word zeroing of the supplied
   words and landed records; the call to mesh_plan_erased and its ETIMEDOUT; the layout's fixed areas of words.  Why:
   C4; a data page holds payload only and nothing is zeroed or reset while a plan stands (C3.7); a new plan's state is
   in new pages, so nothing is erased or waited for (C3.4).  What must stay: the demands, mesh_graph_reach,
   mesh_graph_flow and mesh_graph_fill as they were, so every rank's trees and template are the same; every failure but
   ESTALE and the second map read's found before anything is written. */
/* 2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", PS, PQ and RC): `program` and
   `outstanding` (mesh-collective.h): EINVAL where `program` is 0, lacks this node or lacks a node of the plan's (its
   `nodes`, step 1), ENOMEM where `outstanding` is 0 or a queue's room passes its words, before anything is written;
   struct mesh_state `sender` the least of `program`, where it was the least of the plan's `nodes`; the state section
   holds the plan's two queues, the delivered calls' of `room` = `outstanding` + u x the hops of the plan's routes (the
   distinct tree edges of its demands' trees that carry pages, each demand's `forwards`) and the submissions' of
   `outstanding`, between the status and the entries (plan_layout), started 0 with every word before the entries (D3);
   every received range has its count words, a relay's too, so every receive entry names one.  Why:
   design/aot-compilation-change-request.md A7, "sender(m) is the program's one sender: the least of the program's
   nodes"; appendix item 2, "a queue of the plan's own.  It is laid out with the plan's pages when the plan is placed,
   and claimed and released with them (C3.2, C3.4)"; "Its room is fixed at compile time ... That is at most A10's bound
   on the plan's submissions outstanding at its sender, plus C3.3's u copies on each hop of the plan's routes ... A plan
   whose bound A10 cannot find fails with `ENOMEM` when it is planned"; item 5, "The code must name a page for round j
   by that count".  What must stay: every failure but ESTALE and the second map read's found before anything is
   written */
int mesh_plan(struct mesh_ctx *context,uint64_t group,uint64_t function,uint64_t program,uint64_t outstanding,
              const struct mesh_declared *declared,const struct mesh_weight *weights,uint32_t count,
              struct mesh_value *values,uint32_t given,const struct mesh_dependence *dependence,uint32_t dependences,
              uint64_t *plan){
  struct hdr *m=context->M;
  if(!group || !program || !plan || (count && (!declared || !weights)) || (given && !values) ||
     (dependences && !dependence))
    return EINVAL;
  /* a value: its bytes, its kind, its wire and state flags, and its lag (k >= 1 only on an execution value outside the
     window that is no state: design/disjoint-pages-change-request.md, D.3.3; design/disjoint-pages-rulings-
     coordination.md, R4; that no declared transfer names it is the declared transfers' test below) */
  for(uint32_t j=0;j<given;j++){
    const struct mesh_value *v=values+j;
    if(!v->section.bytes || v->kind>MESH_EXECUTION || v->wire>1 || v->state>1 ||
       (v->state && (v->wire || v->kind!=MESH_INVARIANT)) ||
       (v->lag && (v->kind!=MESH_EXECUTION || v->wire || v->state)))return EINVAL;
    if(plan_blocks(m,v->section.bytes)>=UINT32_MAX/m->block)return ENOMEM;
  }
  /* a declared transfer's kind (design/plan-to-bridge-change-request.md, R4) and its weights (design/aot-compilation-
     change-request.md, A3), each finite and > 0 (design/compilation-coordination.md, C-weights): every task's,
     `produced`, `consumed` and `read`, as Sinnen and Sousa's Definition 1 states ("The positive weight w(n)"), and
     every `page`, so every occupation of a link lasts > 0.
     2026-11-04, owner C2 (review W3 #6, the coordinator's ruling): withdrawn, the reason that with it each link's
     order of starts is a topological order (R5's no-cycle argument; MSCCLang, ASPLOS 2023, §5.2): the contention
     schedule starts a page on a route's links by Sinnen and Sousa's eq. (14), nonaligned (cut-through), so a page's
     start on a link can precede its finish on the link before, and the start order is not shown to be a topological
     order of the bridge's store-and-forward waits.  Open for the operator (design/compilation-coordination.md, the
     appendix's revisions, record).  The EINVAL stays.
     2026-11-04, owner C3 (design/compilation-coordination.md, "The documents' revisions of edd4b583", GO4): no list is
     ordered by the starts now; the no-cycle argument is GO4's (plan_schedule), over the instruction DAG's
     store-and-forward dependencies, and it rests on no weight.  So `page` > 0 keeps no reason from it; the EINVAL
     stays, as C-weights decided it (the open item "No cycle on a cable", where the refusal "stays").
     2026-11-05 (design/compilation-coordination.md, "The revisions of 67ca86f8", SP): GO4's argument went with its
     groups; the argument is R5's own for the places' order (plan_schedule), and it rests on no weight either.  The
     EINVAL stays, as above.
     2026-11-04, owner C: reversing >= 0 for every weight.  Why: C-weights (reviews V1 #8, V3 #5).  What must stay:
     EINVAL before anything is written.
     2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", LP): Sinnen and Sousa's Definition 1
     is no longer this file's reason, the contention schedule being removed (R5's linear program takes any length d_j
     >= 0); the EINVAL stays, as C-weights decided it, since R5 asks no change of the operand's test.  What must stay:
     as above */
  for(uint32_t i=0;i<count;i++){
    const struct mesh_weight *x=weights+i;
    if(declared[i].kind>MESH_PAGES || !(x->produced>0 && x->produced<=DBL_MAX) ||
       !(x->consumed>0 && x->consumed<=DBL_MAX) || !(x->read>0 && x->read<=DBL_MAX) ||
       !(x->page>0 && x->page<=DBL_MAX))return EINVAL;
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
  if(!outstanding || outstanding>=UINT32_MAX)return ENOMEM;
  struct mesh_nodes nodes,after;
  int status=mesh_nodes_read(context,&nodes);
  if(status)return status;
  const uint32_t n=nodes.count;
  if(nodes.self>=n)return EHOSTUNREACH;
  if(!(program>>nodes.self&1))return EINVAL;
  const uint64_t block=(uint64_t)m->block*m->pgsz;
  struct plan p={.context=context,.declared=declared,.transfers=count,.given=values,.givens=given,.n=n,
                 .self=nodes.self,.serial=nodes.serial};
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
    /* a transfer to its own source passes between no node pair, and nothing is copied */
    if(t->source==t->destination || !t->bytes){status=EINVAL;goto done;}
    if(plan_blocks(m,t->bytes)>=UINT32_MAX/m->block){status=ENOMEM;goto done;}
    key[i]=(struct plan_key){t->source,t->identity,i};
  }
  qsort(key,count,sizeof *key,plan_key_order);
  /* each declared transfer's range by its first transfer: key order holds a range's transfers together, each range's
     in declared order */
  for(uint32_t k=0;k<count;k++)
    placed[key[k].index]=!k || key[k].source!=key[k-1].source || key[k].identity!=key[k-1].identity?key[k].index:
                         placed[key[k-1].index];
  /* the demands numbered by their first declared transfer (design/compilation-coordination.md, "The documents'
     revisions of edd4b583", GO5, T6's other half): one opened at each
     transfer that is its range's first, in declared order */
  for(uint32_t i=0;i<count;i++){
    const struct mesh_declared *t=declared+i;
    if(placed[i]==i)
      p.demand[p.demands++]=(struct plan_demand){.source=t->source,.identity=t->identity,.first=i,.mine=MESH_ABSENT,
                                                 .set=1,.bytes=t->bytes,.pages=plan_blocks(m,t->bytes),
                                                 .range=MESH_ABSENT};
    p.of[i]=placed[i]==i?p.demands-1:p.of[placed[i]];
    struct plan_demand *d=p.demand+p.of[i];
    /* a set where every declared transfer of the range is MESH_PAGES, function B at every destination; any MESH_JOIN
       destination keeps the schedule's order (design/compilation-coordination.md, S1; design/aot-compilation-change-
       request.md, appendix item 5: "Function A's operands (joins) keep the schedule's order"), each of its pages at its
       own place in R5's one global order (R5: "A set's pages ... keep their places in that order") */
    if(t->kind!=MESH_PAGES)d->set=0;
    /* one range: its bytes alike, each destination once, its production and its source's read weighing alike
       (design/compilation-coordination.md, "The plan call", struct mesh_weight) */
    if(t->bytes!=d->bytes || d->leaves>>t->destination&1 || weights[i].produced!=weights[d->first].produced ||
       weights[i].read!=weights[d->first].read){status=EINVAL;goto done;}
    d->leaves|=UINT64_C(1)<<t->destination;
    /* `value`: MESH_ABSENT where this node is neither the transfer's source nor its destination; else an execution
       value in the window of the range's bytes and of lag 0 (a value of lag k >= 1 is written by a producer the
       program orders by an event, which no declared transfer is: R4), the source's one value of every transfer of the
       range, one range's alone */
    if(t->source!=p.self && t->destination!=p.self){
      if(t->value!=MESH_ABSENT){status=EINVAL;goto done;}
      continue;
    }
    if(t->value>=given){status=EINVAL;goto done;}
    const struct mesh_value *v=values+t->value;
    if(v->kind!=MESH_EXECUTION || !v->wire || v->lag || v->section.bytes!=t->bytes ||
       (t->source==p.self && d->mine!=MESH_ABSENT && d->mine!=t->value) ||
       (named[t->value]!=MESH_ABSENT && named[t->value]!=d->first)){status=EINVAL;goto done;}
    named[t->value]=d->first;
    d->mine=t->value;
  }
  /* a local execution value's positions, a value of lag 0's alone (a value of lag k >= 1 is ordered by the program's
     events, not by its positions, which are not read: R4) */
  for(uint32_t j=0;j<given;j++)
    if(named[j]==MESH_ABSENT && values[j].kind==MESH_EXECUTION && !values[j].lag && values[j].begin>values[j].end){
      status=EINVAL;goto done;
    }
  for(uint32_t i=0;i<p.demands;i++)if(p.demand[i].source==p.self)p.demand[i].release=(uint32_t)p.releases++;

  if(p.demands){
    for(uint32_t a=0;a<n;a++)for(uint32_t b=a+1;b<n;b++){
      const uint64_t c=nodes.capacity[a][b]<nodes.capacity[b][a]?nodes.capacity[a][b]:nodes.capacity[b][a];
      if(c){p.from[p.pairs]=a;p.to[p.pairs]=b;p.capacity[p.pairs]=(double)c;p.pairs++;}
    }
    const uint32_t E=p.pairs;
    /* a destination its source reaches by no Thunderbolt-5 path within the map: refused (mesh_graph_reach, once from
       each source; no breadth-first parents wanted here) */
    const struct plan_edges graph={p.from,p.to};
    uint32_t order[n];
    unsigned char reached[n];
    uint64_t reach[n],known=0;
    for(uint32_t i=0;i<p.demands;i++){
      const uint32_t s=p.demand[i].source;
      if(!(known>>s&1)){
        mesh_graph_reach(n,E,plan_end,&graph,s,order,reached,NULL);
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
    const uint32_t made=mesh_graph_flow(n,E,p.from,p.to,p.capacity,p.demands,root,up,down,plan_leaf,p.demand,
                                        MESH_PLAN_EPS,rounds,room,demand,p.parent,share);
    if(made==UINT32_MAX){status=ENOMEM;goto done;}
    p.made=made;
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
  /* 1. the trees' communications, the template, no lag-0 cycle, the least lags and u; this node's entries of one copy
     counted and its rooms; the state section's layout; the items and both colourings; the entries in their places'
     order (R5); the plan's nodes and sender; its name unused at this node: all before anything is claimed or
     written */
  if((status=plan_trees(&p)) || (status=plan_template(&p,dependence,dependences)) || (status=plan_acyclic(&p)) ||
     (status=plan_unroll(&p)))goto done;
  plan_walk(&p);
  const uint32_t u=p.copies;
  uint64_t entries,rows=1;
  /* 2026-11-04, owner C: the receives past a 32-bit count (struct mesh_state `receives`) are ENOMEM, as every other
     count past its word here, where they were ENOSPC with the list's room, which is gone (the entries are in the
     state section) */
  if(p.receives>=UINT32_MAX || __builtin_mul_overflow(p.entries,(uint64_t)u,&entries) ||
     entries>=UINT64_MAX/(2*sizeof(struct mesh_hop))){
    status=ENOMEM;goto done;
  }
  /* the plan's nodes (a set of map indices) and its sender, the least of them (design/compilation-coordination.md,
     "The region", struct mesh_state `nodes` and `sender`; Q2), the same at every rank: every source and destination of
     a declared transfer and every node a tree the fill gave pages reaches (its `on`), this node's own where none.
     2026-11-05 ("The revisions of 3f106747", BS): computed here, before the rooms, where it followed the schedule.
     What must stay: as above */
  uint64_t spanned=0;
  for(uint32_t i=0;i<p.demands;i++)spanned|=UINT64_C(1)<<p.demand[i].source|p.demand[i].leaves|p.demand[i].on;
  if(!spanned)spanned=UINT64_C(1)<<p.self;
  if(spanned&~program){status=EINVAL;goto done;}
  /* the queues' rooms and the synchronization edges, each queue's self-timed buffer bound in the plan's part of the
     program's IPC graph (plan_rooms), every rank's the same from the same trees, u, weights and operands.
     2026-11-05 (design/aot-compilation-change-request.md, appendix item 2, "Its room is fixed at compile time, by the
     self-timed buffer bound"; design/compilation-coordination.md, "The revisions of 3f106747", BS), reversing the
     abecd389 form, the delivered calls' queue's room A10's bound plus u copies on each hop of the plan's routes and
     the submissions' queue's room A10's bound.  Why: item 2's correction: "It holds only where the queue lies on a
     cycle through the sender.  A7's program sender may hold no page of a plan, and then the queue is BSL's feedforward
     edge with no bound".  What must stay: ENOMEM where no bound is found, before anything is written */
  p.outstanding=outstanding;
  if((status=plan_rooms(&p,weights,program,outstanding,spanned)))goto done;
  /* every value's rows: count x P, count u for a routed value, k for a value of lag k >= 1 and 1 for every other, after
     the claim's head (each term below 2^64 - 2^32, P below UINT32_MAX / block, so no sum wraps before the test) */
  for(uint32_t j=0;j<given && rows<UINT32_MAX;j++){
    const uint64_t copies=named[j]!=MESH_ABSENT?u:values[j].lag?values[j].lag:1;
    rows+=copies*plan_blocks(m,values[j].section.bytes);
  }
  if(rows>=UINT32_MAX){status=ENOMEM;goto done;}
  plan_layout(&p);
  if((status=plan_place(&p)))goto done;
  if(p.window*m->block>=UINT32_MAX || p.outside*m->block>=UINT32_MAX){status=ENOMEM;goto done;}
  if(!(p.entry=malloc((p.entries+1)*sizeof *p.entry))){status=ENOMEM;goto done;}
  plan_walk(&p);
  if(p.demands && (status=plan_schedule(&p,weights,dependence,dependences)))goto done;
  /* a plan standing at this node already (group, function) on the map this call plans on: EEXIST until the program
     retires it (design/compilation-coordination.md, N1: "A replan of a function on the same map is EEXIST until the
     program retires the standing one"; one whose `serial` is not that map's is cancelled, D.3.6, and ignored), every
     claim's head walked (mesh-dataflow.h mesh_head_after, mesh_head_state).
     2026-11-04, owner C2 (N1): the name's second word the function, where it was C-epoch's epoch.  What must stay:
     EEXIST where a plan of the name stands on the same map */
  for(uint32_t head=mesh_head_after(m,UINT32_MAX);head!=UINT32_MAX;head=mesh_head_after(m,head)){
    const struct mesh_state *standing=mesh_head_state(m,head);
    if(standing && standing->serial==p.serial && standing->group==group && standing->epoch==function){
      status=EEXIST;goto done;
    }
  }

  /* 2. the claim, its generation from hdr `serial`: exactly the rows and pages this plan places, whole or not at all */
  const uint64_t generation=atomic_fetch_add(&m->serial,1)+1;
  if((status=mesh_claim_take(context,generation,(uint32_t)rows,(uint32_t)(p.window*m->block),
                             (uint32_t)(p.outside*m->block),&p.claim)))goto done;

  /* 3. the state section started once, where it is placed (D3): its bytes before the entries 0, then struct
     mesh_state's fields; nothing else of it is written by the plan but the entries, and no word of it is reset while
     the plan stands */
  const uint32_t at=plan_item_page(&p,p.state,0);
  const uint64_t base=m->data_off+(uint64_t)at*m->pgsz;
  memset(mesh_at(m,at),0,(size_t)p.hops);
  struct mesh_state *state=(struct mesh_state *)mesh_at(m,at);
  state->copies=u;
  state->receives=(uint32_t)p.receives;
  state->ranges=p.consumed;
  state->releases=(uint32_t)p.releases;
  state->phases=base+p.phases;
  state->counts=base+p.counts;
  state->supplied=base+p.supplied;
  state->status=base+p.status;
  state->group=group;
  state->epoch=function;
  state->serial=p.serial;
  state->hops=base+p.hops;
  state->entries=entries;
  state->nodes=spanned;
  state->sender=(uint32_t)__builtin_ctzll(program);
  state->outstanding=(uint32_t)p.outstanding;
  state->room=p.room;
  state->submit_room=p.submit_room;
  state->waits=p.waits;
  state->tells=p.tells;
  state->wait_delay=p.wait_delay;
  state->deliveries=base+p.deliveries;
  state->submits=base+p.submits;
  /* every value's rows, from the head's next, copy c's rows first + c x P + k bound to its copy's pages (an item's
     first page, plan_item_page), named clean as mesh_section_create named them (no device, no owner, no publication:
     every field of a row's publication 0 but its range's words below); a `state` 1 value's bytes 0 once, its one copy
     (D3: a local invariant outside the window alone); a routed value's copies its demand's routed value's here
     (plan_page), a relay's value no row's; a value of lag k >= 1's copy c its item local x u + other + c (its k items
     in copy order), its rows bound to that item's pages, every row's publication 0 as every local value's (R4) */
  uint32_t row=p.claim.rows.first+1;
  for(uint32_t j=0,other=0;j<given;j++){
    struct mesh_value *v=values+j;
    const struct plan_demand *d=named[j]!=MESH_ABSENT?p.demand+p.of[named[j]]:NULL;
    const uint32_t P=(uint32_t)plan_blocks(m,v->section.bytes),copies=d?u:v->lag?v->lag:1;
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
      const uint32_t page=d?plan_page(&p,d->value,c,0):plan_item_page(&p,u*p.local+other+c,v->wire);
      mesh_backing_bind(context,row+c*P,P*m->block,page,c);
      if(v->state)memset(mesh_at(m,page),0,v->section.bytes);
    }
    if(!d)other+=copies;
    /* a received range's first row: its pages, its copy-0 group and count word, the state section; a source range's:
       its supplied word, the one target a host or GPU producer stores into */
    struct mesh_publication *publication=mesh_publication_at(m,row);
    if(d && d->source==p.self){
      publication->sends=1;
      publication->targets[0]=(struct mesh_target){.word=base+p.supplied+(uint64_t)d->release*sizeof(uint64_t),
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
  /* the entries, copy c's n at [c n, c n + n) of the section's `hops`, in plan_schedule's order (R5), each naming
     copy c's page and words (mesh.h struct mesh_hop): a receive its field's word and low bit in its range's copy-c
     group, its range's copy-c count word where this node consumes the range (0 at a relay, whose receive the bridge
     counts nothing for: review 15's finding 1) and P_g; a forwarding send its receive's word and field; a send from the
     range's source its release's supplied word; each its page's bytes, page k of a range of B bytes min(Q, B - k Q), Q
     = block x pgsz (design/compilation-coordination.md, T7); each its entry's `set`, every copy's the same (S1, S2) */
  struct mesh_hop *hop=mesh_hops(m,state);
  for(uint32_t c=0;c<u;c++)for(uint64_t j=0;j<p.entries;j++){
    const struct plan_entry *e=p.entry+j;
    const struct plan_demand *d=p.demand+e->demand;
    const uint64_t left=d->bytes-e->page*block;
    struct mesh_hop h={.page=plan_page(&p,e->value,c,e->page),.node=e->node,.direction=e->direction,
                       .field=MESH_ABSENT,.bytes=(uint32_t)(left<block?left:block),.serial=p.serial,.set=e->set};
    if(e->field==MESH_ABSENT)h.word=base+p.supplied+(uint64_t)d->release*sizeof(uint64_t);
    else{
      h.word=base+p.phases+d->group+(uint64_t)c*mesh_group_bytes(d->received)+(uint64_t)e->field/32*sizeof(uint64_t);
      h.field=2*(e->field%32);
      if(e->direction==MESH_RECEIVE){
        h.count=base+p.counts+((uint64_t)d->counted*u+c)*sizeof(uint64_t);
        h.pages=d->received;
      }
    }
    *hop++=h;
  }
  /* 4. the map read again: where it moved, or the reader failed, this claim alone is closed and `*plan` untouched
     (design/compilation-coordination.md, N3; a bridge refuses a plan whose `serial` is not `map_serial` as it loads
     it, and cancels one that stands where the serial moves: mesh.h, "Cancellation").
     2026-11-04, owner C2 (N3, N5): reversing the close of every earlier claim of the client, context->state 0 and
     context->epoch the named epoch less 1 (C-epoch).  Why: compiled functions coexist; this call's failure is its own
     claim's alone.  What must stay: ESTALE, or the reader's error, with this claim closed */
  if(!(status=mesh_nodes_read(context,&after)) && after.serial!=p.serial)status=ESTALE;
  if(status){mesh_claim_close(context,generation);goto done;}
  /* 5. the plan stands: its head's publication `state` (release), after every write above; `*plan` that offset, the
     plan's handle (N2); `plans` raised and woken.
     2026-11-04, owner C2 (N3, N5): step 6, every earlier claim of this client closed once this plan stood, is gone,
     and context->state and context->epoch are no longer set.  Why: a program holds any number of compiled functions
     at once, each released only by its retirement or the program's (design/aot-compilation-change-request.md, appendix
     item 6).  What must stay: the publication a release after every write, then `plans` raised and woken */
  __atomic_store_n(&mesh_publication_at(m,p.claim.rows.first)->state,base,__ATOMIC_RELEASE);
  *plan=base;
  mesh_notify(&m->plans);
done:
  free(key);free(placed);free(named);free(p.demand);free(p.of);
  free(root);free(up);free(down);free(demand);free(p.parent);free(share);
  free(p.tree);free(p.run);free(rate);free(latency);free(trees);free(p.members);free(p.lead);free(p.comm);
  free(p.child);free(p.value);free(p.read);free(p.here);free(p.first);free(p.head);free(p.lag);free(p.between);
  free(p.others);free(p.colour);free(p.entry);
  return status;
}
