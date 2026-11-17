#include "mesh-collective.h"
#include "mesh-graph.h"
#include <float.h>
#include <math.h>
#include <os/os_sync_wait_on_address.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#pragma STDC FP_CONTRACT OFF

_Static_assert(MESH_NODES_MOST<=64,"a demand's destinations are one 64-bit set of map indices");

/* design/disjoint-pages-change-request.md C3.7; design/aot-compilation-change-request.md, appendix item 7 */
void mesh_host_publish(struct mesh_ctx *context,uint64_t plan,uint32_t transfer,uint64_t execution){
  struct hdr *m=context->M;
  const struct mesh_publication *entry=mesh_transfer_at(m,(const struct mesh_state *)(const void *)((char *)m+plan),
                                                        transfer);
  _Atomic uint64_t *word=(_Atomic uint64_t *)(void *)((char *)m+entry->targets[0].word);
  uint64_t seen=atomic_load_explicit(word,memory_order_relaxed);
  while(seen<execution+1 &&
        !atomic_compare_exchange_weak_explicit(word,&seen,execution+1,memory_order_release,memory_order_relaxed));
}

/* design/disjoint-pages-change-request.md C3.1, C3.7 */
uint64_t mesh_host_arrived(struct mesh_ctx *context,uint64_t plan,uint32_t transfer,uint64_t execution){
  struct hdr *m=context->M;
  const struct mesh_state *state=(const struct mesh_state *)(const void *)((char *)m+plan);
  const struct mesh_publication *entry=mesh_transfer_at(m,state,transfer);
  const uint64_t copies=state->copies,copy=execution%copies,round=execution/copies,pages=entry->pages;
  const uint32_t expected=mesh_phase(round+1),behind=mesh_phase(round);
  const uint64_t count=atomic_load_explicit((_Atomic uint64_t *)((char *)m+entry->count+8*copy),memory_order_acquire);
  _Atomic uint64_t *group=(_Atomic uint64_t *)((char *)m+entry->phases+copy*mesh_group_bytes(entry->pages));
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

/* libfabric man/fi_cntr.3.md, fi_cntr_wait, fi_cntr_readerr; libfabric man/fi_endpoint.3.md, fi_cancel;
   rdma-core libibverbs/man/ibv_post_send.3, NOTES */
uint64_t mesh_host_sent(struct mesh_ctx *context,uint64_t plan,uint32_t transfer,uint64_t execution){
  struct hdr *m=context->M;
  const struct mesh_state *state=(const struct mesh_state *)(const void *)((char *)m+plan);
  const struct mesh_publication *entry=mesh_transfer_at(m,state,transfer);
  const uint64_t copies=state->copies;
  _Atomic uint64_t *counter=(_Atomic uint64_t *)((char *)m+entry->completion+16*(execution%copies));
  const uint64_t count=atomic_load_explicit(counter,memory_order_acquire),
                 error=atomic_load_explicit(counter+1,memory_order_acquire);
  if(count+error<(execution/copies+1)*entry->sends)return 0;
  return error?UINT64_MAX:1;
}

/* design/disjoint-pages-change-request.md C3.6; design/problems-of-14b9a757-change-request.md 7.3, steps 3 and 6;
   libfabric man/fi_cntr.3.md, fi_cntr_wait; mesh-call.h mesh_transfers_start, 2026-11-06 */
int mesh_host_wait(struct mesh_ctx *context,uint64_t plan,uint32_t transfer,uint64_t execution,uint32_t direction){
  struct hdr *m=context->M;
  const struct mesh_state *state=(const struct mesh_state *)(const void *)((char *)m+plan);
  const struct mesh_publication *entry=mesh_transfer_at(m,state,transfer);
  const uint64_t copy=execution%state->copies;
  if(direction==MESH_SEND && !entry->sends)return 0;
  _Atomic uint64_t *word=(_Atomic uint64_t *)((char *)m+(direction==MESH_SEND?entry->completion+16*copy:entry->count+8*copy));
  for(int expired=0;;){
    const uint64_t seen=atomic_load_explicit(word,memory_order_acquire);
    const uint64_t done=direction==MESH_SEND?mesh_host_sent(context,plan,transfer,execution):
                                             mesh_host_arrived(context,plan,transfer,execution);
    if(done)return done==UINT64_MAX?ECANCELED:0;
    if(expired)return ETIMEDOUT;
    expired=os_sync_wait_on_address_with_timeout(word,seen,sizeof *word,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                                                 OS_CLOCK_MACH_ABSOLUTE_TIME,MESH_PLATFORM_NS)<0 && errno==ETIMEDOUT &&
            atomic_load_explicit(word,memory_order_acquire)==seen;
  }
}

struct plan_loaded;
static void plan_loaded_free(struct plan_loaded *);
static void plan_flow_loaded_free(struct mesh_plan_flow *);
static void plan_step_loaded_free(struct plan_step *);

#define MESH_PLAN_EPS 0.1
/* struct mesh_plan_flow `bound`: a slot its function's calls bind to more than one root */
#define PLAN_SEVERAL (MESH_ABSENT-1)

/* `mine` this node's function value of the demand, `root` its shared program root (mesh-collective.h struct
   mesh_declared `root` - 1; MESH_ABSENT none), at most one of them */
static void plan_steps_results_free(struct plan_step *p){
  struct plan_schedule_result *s=p->schedule;
  if(s){
    free(s->reader);free(s->consumer);free(s->w);free(s->cost);free(s->rate);free(s->completion);free(s->priority);
    free(s->resource);free(s->from);free(s->to);free(s->link);free(s->communication);free(s->before);free(s->after);
    free(s->place);free(s->route);free(s);
  }
  struct plan_graph_result *g=p->graph;
  if(g){
    free(g->actor);free(g->later);free(g->lag);free(g->lagged);free(g->time);free(g->from);free(g->to);
    free(g->delay);free(g->call);free(g->node);free(g->observer);free(g->key);free(g->added_from);free(g->added_to);
    free(g->added_delay);free(g->entry);free(g->first);free(g->rows);free(g->wait_first);free(g->wait_order);
    free(g->tell_first);free(g->tell_order);free(g->task);free(g->cause);
    mesh_graph_group_free(g->room_rows);mesh_graph_lag_graph_free(g->room_graph);
    mesh_graph_lag_result_free(g->room_lags);free(g->room_query);free(g);
  }
}

struct plan_key { uint32_t source,identity,index; };
static int plan_key_order(const void *a,const void *b){
  const struct plan_key *x=a,*y=b;
  if(x->source!=y->source)return x->source<y->source?-1:1;
  if(x->identity!=y->identity)return x->identity<y->identity?-1:1;
  return (x->index>y->index)-(x->index<y->index);
}
static uint32_t plan_end(const void *context,uint32_t e,uint32_t side){
  const struct mesh_plan_flow *graph=context;
  return side?graph->to[e]:graph->from[e];
}
static int plan_leaf(const void *context,uint32_t d,uint32_t v){
  return (int)(((const struct mesh_plan_demand *)context)[d].leaves>>v&1);
}
static uint32_t plan_at(const void *context,size_t k){ return ((const uint32_t *)context)[k]; }
static void plan_tree_up(void *consumer,uint32_t at,uint32_t step){ if(step==1)*(uint32_t *)consumer=at; }
uint64_t mesh_plan_quantum(const struct hdr *m){ return (uint64_t)m->block*m->pgsz; }
uint64_t mesh_plan_quanta(uint64_t bytes,uint64_t quantum){ return (bytes+quantum-1)/quantum; }
static uint64_t plan_quantum(const struct mesh_plan_geometry *m){ return (uint64_t)m->block*m->pgsz; }
static uint64_t plan_blocks(const struct mesh_plan_geometry *m,uint64_t bytes){ return mesh_plan_quanta(bytes,plan_quantum(m)); }
/* L. J. Hendren, G. R. Gao, E. R. Altman and C. Mukerji, ACAPS Technical Memo 33, §2.2;
   design/plan-to-bridge-change-request.md R2 */
static uint32_t plan_copies(const struct mesh_plan_flow *p,uint32_t j){
  if(p->named[j] || (p->given[j].kind==MESH_EXECUTION && !p->value_lag[j]))return p->copies;
  return p->value_lag[j]?p->value_lag[j]:1;
}

static uint32_t plan_rank(uint64_t set,uint32_t v){ return (uint32_t)__builtin_popcountll(set&((UINT64_C(1)<<v)-1)); }
static uint32_t plan_produced(const struct mesh_plan_flow *p,uint32_t i){ return p->demand[i].vertex; }
static uint32_t plan_landed(const struct mesh_plan_flow *p,uint32_t i,uint32_t y,int last){
  return p->demand[i].vertex+1+2*plan_rank(p->demand[i].on,y)+(uint32_t)!!last;
}
static uint32_t plan_forwarded(const struct mesh_plan_flow *p,uint32_t i,uint32_t y,uint32_t c){
  const uint64_t *child=p->child+(size_t)i*p->n;
  uint32_t at=p->demand[i].vertex+1+2*(uint32_t)__builtin_popcountll(p->demand[i].on);
  for(uint32_t x=0;x<y;x++)at+=(uint32_t)__builtin_popcountll(child[x]);
  return at+plan_rank(child[y],c);
}
static uint32_t plan_consumed(const struct mesh_plan_flow *p,uint32_t i,uint32_t z){
  const struct mesh_plan_demand *d=p->demand+i;
  return d->vertex+1+2*(uint32_t)__builtin_popcountll(d->on)+d->forwards+plan_rank(d->leaves,z);
}
static uint32_t plan_own(const struct mesh_plan_flow *p,uint32_t i){
  const struct mesh_plan_demand *d=p->demand+i;
  return d->vertex+1+2*(uint32_t)__builtin_popcountll(d->on)+d->forwards+(uint32_t)__builtin_popcountll(d->leaves);
}
static uint32_t plan_event(const struct mesh_plan_flow *p,uint32_t i,int produced,int own){
  if(produced)return plan_produced(p,p->of[i]);
  if(own)return plan_own(p,p->of[i]);
  return plan_consumed(p,p->of[i],p->declared[i].destination);
}
static uint32_t plan_event_node(const struct mesh_declared *declared,uint32_t i,int produced,int own){
  return produced || own?declared[i].source:declared[i].destination;
}
static uint32_t plan_edge_tail(const void *context,size_t k){ return ((const struct mesh_graph_dependence *)context)[k].from; }

/* design/disjoint-pages-change-request.md C3.3 */
static int plan_template(struct mesh_plan_flow *p,const struct mesh_dependence *dependence,uint32_t dependences){
  const uint32_t n=p->n,self=p->self;
  p->child=calloc((size_t)p->demands*n+1,sizeof *p->child);
  if(!p->child)return ENOMEM;
  uint64_t vertices=0,edges=dependences,values=0,reads=0;
  for(uint32_t k=0;k<dependences;k++){
    const struct mesh_dependence *x=dependence+k;
    if(x->kinds&MESH_FROM_SOURCE_READ)p->demand[p->of[x->from]].own=1;
    if(x->kinds&MESH_TO_SOURCE_READ)p->demand[p->of[x->to]].own=1;
  }
  for(uint32_t i=0;i<p->demands;i++){
    struct mesh_plan_demand *d=p->demand+i;
    uint64_t *below=p->child+(size_t)i*n;
    for(uint32_t j=d->tree;j<d->tree+d->trees;j++){
      if(!p->run[j])continue;
      const size_t *first=p->adjacency[j]->children->first,*child=p->adjacency[j]->children->order;
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
  struct mesh_graph_dependence *edge=p->edge=malloc((p->edges+1)*sizeof *edge);
  if(!p->value || !p->read || !p->here || !edge)return ENOMEM;
  size_t e=0;
  for(uint32_t i=0;i<p->demands;i++){
    const struct mesh_plan_demand *d=p->demand+i;
    const uint64_t *below=p->child+(size_t)i*n;
    const uint32_t produced=plan_produced(p,i);
    for(uint64_t s=d->on;s;s&=s-1){
      const uint32_t y=(uint32_t)__builtin_ctzll(s);
      edge[e++]=(struct mesh_graph_dependence){produced,plan_landed(p,i,y,0),0};
      edge[e++]=(struct mesh_graph_dependence){plan_landed(p,i,y,0),plan_landed(p,i,y,1),0};
    }
    for(uint32_t y=0;y<n;y++)for(uint64_t s=below[y];s;s&=s-1){
      const uint32_t c=(uint32_t)__builtin_ctzll(s),forward=plan_forwarded(p,i,y,c);
      edge[e++]=(struct mesh_graph_dependence){y==d->source?produced:plan_landed(p,i,y,0),forward,0};
      edge[e++]=(struct mesh_graph_dependence){forward,plan_landed(p,i,c,1),0};
    }
    for(uint64_t s=d->leaves;s;s&=s-1){
      const uint32_t z=(uint32_t)__builtin_ctzll(s);
      edge[e++]=(struct mesh_graph_dependence){plan_landed(p,i,z,1),plan_consumed(p,i,z),0};
    }
    if(d->own)edge[e++]=(struct mesh_graph_dependence){produced,plan_own(p,i),0};
  }
  for(uint32_t k=0;k<dependences;k++){
    const struct mesh_dependence *x=dependence+k;
    edge[e++]=(struct mesh_graph_dependence){
      plan_event(p,x->from,!!(x->kinds&MESH_FROM_PRODUCED),!!(x->kinds&MESH_FROM_SOURCE_READ)),
      plan_event(p,x->to,!!(x->kinds&MESH_TO_PRODUCED),!!(x->kinds&MESH_TO_SOURCE_READ)),x->lag};
  }
  int status=mesh_graph_group(p->vertices,p->edges,plan_edge_tail,edge,&p->rows);
  if(status)return status;
  uint32_t at=0,source=0;
  p->values=0;p->local=0;
  for(uint32_t i=0;i<p->demands;i++){
    struct mesh_plan_demand *d=p->demand+i;
    d->value=MESH_ABSENT;
    for(uint64_t s=d->on|UINT64_C(1)<<d->source;s;s&=s-1){
      const uint32_t x=(uint32_t)__builtin_ctzll(s);
      struct mesh_plan_vertex *v=p->value+p->values;
      *v=(struct mesh_plan_vertex){.demand=i,.node=x,.write=x==d->source?plan_produced(p,i):plan_landed(p,i,x,0),.read=at,
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

/* design/disjoint-pages-change-request.md C3.2-C3.4 */

static uint32_t plan_head(const void *context,size_t k){
  const struct mesh_plan_flow *p=context;return p->edge[p->rows->order[k]].to;
}
static uint64_t plan_lag(const void *context,size_t k){
  const struct mesh_plan_flow *p=context;return p->edge[p->rows->order[k]].delay;
}
static uint32_t plan_zero_head(const void *context,size_t k){
  return plan_lag(context,k)?UINT32_MAX:plan_head(context,k);
}
static int plan_acyclic(struct mesh_plan_flow *p){
  p->acyclic_indegree=malloc(((size_t)p->vertices+1)*sizeof *p->acyclic_indegree);
  p->acyclic_ordinal=malloc(((size_t)p->vertices+1)*sizeof *p->acyclic_ordinal);
  if(!p->acyclic_indegree || !p->acyclic_ordinal)return ENOMEM;
  return mesh_graph_order(p->vertices,p->rows->first,plan_zero_head,p,p->acyclic_indegree,p->acyclic_ordinal)?EINVAL:0;
}

static void plan_name(const struct mesh_plan_flow *p,uint32_t v,const char *what,uint64_t lag){
  const struct mesh_plan_vertex *x=p->value+v;
  const struct mesh_plan_demand *d=p->demand+x->demand;
  char here[48]="";
  if(x->node==p->self){
    if(d->mine!=MESH_ABSENT)snprintf(here,sizeof here," (values[%u] here)",d->mine);
    else snprintf(here,sizeof here," (this node's relay)");
  }
  if(lag==UINT64_MAX)
    fprintf(stderr,"mesh_plan: the value of range %u from rank %u at rank %u%s: %s (no walk)\n",d->identity,d->source,
            x->node,here,what);
  else fprintf(stderr,"mesh_plan: the value of range %u from rank %u at rank %u%s: %s (least lag %llu)\n",d->identity,
               d->source,x->node,here,what,(unsigned long long)lag);
}

/* design/disjoint-pages-change-request.md C3.3; M. Lam, PLDI 1988; B. R. Rau, MICRO 1994;
   design/aot-compilation-change-request.md, appendix item 1 */
/* design/aot-compilation-change-request.md, appendix item 6 (block execution b runs the function's executions
   b every + ordinal); M. Lam, PLDI 1988, modulo variable expansion: an EXECUTION value of execution e is live from its
   writer to its last reader, all in e's block b (other functions' calls read it there, outside this plan's graph), and
   the program issues `horizon` block executions ahead of their ends (mesh_plan's operand, the native program's lag), so
   horizon x every executions are live at once, each with its own copy: e + copies is then in block b + horizon or
   later, which the program's `ended` wait opens only after block b ended. A10's `outstanding` (the calls the program
   may have outstanding at once) is the issue bound plan_graph lowers, not this lifetime. The graph's lags raise the
   copies where a read reaches a later write farther on; a read reaching none (a delivery to a node where its function
   has no section, BSL's feedforward edge) adds nothing, its reuse held at run time: the bridge serves execution t only
   below `executions`, which mesh_host_started raises after the block that read copy t - copies ended (mesh-flow.c,
   mesh-dataflow.c mesh_host_started) */
static int plan_unroll(struct mesh_plan_flow *p){
  const uint32_t values=p->values;
  p->copies=p->horizon*p->calls;
  /* M. Lam, PLDI 1988, §2.3: a transferred value its host writes `lag` executions ahead (design/aot-compilation-change-
     request.md, appendix item 2) takes the plan's copies (plan_copies), at least its lag, so that the copy of execution
     e + lag - 1 the host writes is not one the sends of execution e still read */
  for(uint32_t j=0;j<p->givens;j++)if(p->named[j] && p->value_lag[j]>p->copies)p->copies=p->value_lag[j];
  if(!values)return 0;
  const struct mesh_plan_vertex *last=p->value+values-1;
  const uint32_t sources=last->source+1+last->reads;
  size_t cells,pairs;
  if(__builtin_mul_overflow((size_t)sources,(size_t)values,&cells) || cells>=SIZE_MAX/sizeof(uint64_t) ||
     __builtin_mul_overflow((size_t)p->local,(size_t)p->local,&pairs) || pairs>=SIZE_MAX/sizeof(uint32_t))return ENOMEM;
  uint32_t *source=p->lag_source=malloc((size_t)sources*sizeof *source),
           *target=p->lag_target=malloc((size_t)values*sizeof *target);
  p->between=malloc((pairs+1)*sizeof *p->between);
  p->farthest=calloc((size_t)p->local+1,sizeof *p->farthest);
  int status=!source || !target || !p->between || !p->farthest?ENOMEM:0;
  for(uint32_t v=0;!status && v<values;v++){
    const struct mesh_plan_vertex *x=p->value+v;
    source[x->source]=x->write;
    for(uint32_t j=0;j<x->reads;j++)source[x->source+1+j]=p->read[x->read+j];
    target[v]=x->write;
  }
  if(!status)status=mesh_graph_lag_prepare(p->vertices,p->rows->first,plan_head,plan_lag,p,&p->lag_graph);
  if(!status)status=mesh_graph_lag_matrix(p->lag_graph,sources,source,values,target,&p->lags);
  const uint64_t *lags=p->lags?p->lags->value:NULL;
  for(uint32_t v=0;!status && v<values;v++){
    const struct mesh_plan_vertex *x=p->value+v;
    uint64_t most=0;
    for(uint32_t j=0;j<x->reads;j++){
      const uint64_t lag=lags[((size_t)x->source+1+j)*values+v];
      if(lag>most)most=lag;
    }
    if(most==UINT64_MAX)most=0;
    if(most>=UINT32_MAX){plan_name(p,v,"its copies exceed the 32-bit count of a plan's copies",most);status=ENOMEM;break;}
    if(most>p->copies)p->copies=(uint32_t)most;
  }
  for(uint32_t a=0;!status && a<p->local;a++)for(uint32_t b=0;b<p->local;b++){
    const struct mesh_plan_vertex *x=p->value+p->here[a];
    uint64_t most=0;
    for(uint32_t j=0;j<x->reads;j++){
      const uint64_t lag=lags[((size_t)x->source+1+j)*values+p->here[b]];
      if(lag>most)most=lag;
    }
    if(most!=UINT64_MAX && most>=UINT32_MAX){
      plan_name(p,p->here[b],"its lag from another value here exceeds the 32-bit lag table",most);status=ENOMEM;break;
    }
    const uint32_t held=most==UINT64_MAX?UINT32_MAX:(uint32_t)most;
    p->between[(size_t)a*p->local+b]=held;
    if(held>p->farthest[a])p->farthest[a]=held;
  }
  return status;
}

/* design/disjoint-pages-change-request.md C3.2, C3.7; design/aot-compilation-change-request.md,
   appendix items 2 and 5 */
static uint64_t plan_align(uint64_t at){ return (at+63)/64*64; }
static void plan_groups(struct mesh_plan_flow *p){
  uint64_t at=0;
  uint32_t relayed=p->consumed;
  for(uint32_t i=0;i<p->demands;i++){
    struct mesh_plan_demand *d=p->demand+i;
    d->counted=d->range==MESH_ABSENT?MESH_ABSENT:d->consumed!=MESH_ABSENT?d->consumed:relayed++;
    if(d->range==MESH_ABSENT)continue;
    d->group=at;
    at+=(uint64_t)p->copies*mesh_group_bytes(d->received);
  }
  p->group_bytes=at;
}
static void plan_layout(struct plan_step *p){
  uint64_t at=sizeof(struct mesh_state);
  p->phases=at;
  at=p->flow->group_bytes;
  p->counts=plan_align(p->phases+at);
  p->completions=plan_align(p->counts+(uint64_t)p->flow->ranges*p->flow->copies*sizeof(uint64_t));
  p->supplied=plan_align(p->completions+(uint64_t)p->flow->sending*p->flow->copies*2*sizeof(uint64_t));
  p->publications=plan_align(p->supplied+p->flow->releases*sizeof(uint64_t));
  p->status=plan_align(p->publications+(uint64_t)p->flow->transfers*sizeof(struct mesh_publication));
  p->deliveries=plan_align(p->status+sizeof(struct mesh_status));
  p->submits=plan_align(p->deliveries+p->room*sizeof(struct mesh_delivery));
  p->syncs=plan_align(p->submits+p->submit_room*sizeof(struct mesh_submission));
  p->rows=plan_align(p->syncs+((uint64_t)p->waits+p->tells)*sizeof(struct mesh_sync));
  p->hops=plan_align(p->rows+((uint64_t)p->flow->calls+1)*sizeof(uint32_t));
  p->times=plan_align(p->hops+p->flow->entries*p->flow->copies*sizeof(struct mesh_hop));
  p->words=p->times+(uint64_t)p->flow->transfers*sizeof(double);
}
static uint64_t plan_supplied(const struct plan_step *p,const struct mesh_plan_demand *d){
  return p->base+p->supplied+(uint64_t)d->release*sizeof(uint64_t);
}
static uint64_t plan_group(const struct plan_step *p,const struct mesh_plan_demand *d,uint32_t c){
  return p->base+p->phases+d->group+(uint64_t)c*mesh_group_bytes(d->received);
}
static uint64_t plan_count(const struct plan_step *p,const struct mesh_plan_demand *d,uint32_t c){
  return p->base+p->counts+((uint64_t)d->counted*p->flow->copies+c)*sizeof(uint64_t);
}
static uint64_t plan_completion(const struct plan_step *p,const struct mesh_plan_demand *d,uint32_t c){
  return p->base+p->completions+((uint64_t)d->sending*p->flow->copies+c)*2*sizeof(uint64_t);
}

enum { PLAN_ALONE, PLAN_LOCAL, PLAN_ROUTED };


/* ExecuTorch exir/memory_planning.py, update_tensor_lifetime */
static int plan_alone(const void *context,uint32_t a,uint32_t b){
  (void)context;(void)a;(void)b;
  return 0;
}

static void plan_item_put(struct plan_placement *x,uint32_t kind,uint32_t function,uint32_t index,uint32_t copy,
                          uint64_t blocks,uint32_t space){
  if(x->item){
    x->item[x->items]=(struct plan_item){x->elements,blocks,kind,function,index,copy,space};
    x->count[x->items]=blocks;
  }
  x->items++;x->elements+=blocks;
}
/* the table item: the publication record (mesh.h struct mesh_algo), then the plans' table and adjacency */
static uint64_t plan_table(uint32_t count,uint32_t nodes,uint64_t places){
  return sizeof(struct mesh_algo)+(uint64_t)count*sizeof(uint64_t)+((uint64_t)nodes+1+places)*sizeof(uint32_t);
}
static void plan_items(struct plan_step *plans,uint32_t count,const struct mesh_plan_geometry *m,struct plan_placement *x,uint32_t space){
  uint64_t places=0;
  for(uint32_t f=0;f<count;f++){
    struct plan_step *p=plans+f;
    const uint32_t u=p->flow->copies;
    places+=(uint64_t)__builtin_popcountll(p->flow->toward);
    if(space){
      p->set_item=(uint32_t)x->items;
      for(uint32_t s=0;s<p->flow->sets;s++)for(uint32_t c=0;c<u;c++)plan_item_put(x,PLAN_ROUTED,f,s,c,p->flow->set_blocks[s],1);
    }
    for(uint32_t j=0;j<p->function->values;j++){
      const struct mesh_value *v=p->function->value+j;
      /* a slot a use binds to a root: its storage the root's, none of its own */
      if(v->run || (p->flow->named[j]?1u:v->wire)!=space || (p->flow->bound && p->flow->bound[j]!=MESH_ABSENT))continue;
      const uint32_t kind=v->kind==MESH_EXECUTION && !p->flow->value_lag[j]?PLAN_LOCAL:PLAN_ALONE;
      p->item[j]=p->flow->uncovered[j]?(uint32_t)x->items:MESH_ABSENT;
      for(uint32_t c=0;p->flow->uncovered[j] && c<plan_copies(p->flow,j);c++)plan_item_put(x,kind,f,j,c,p->flow->uncovered[j],space);
    }
    if(!space){p->state=(uint32_t)x->items;plan_item_put(x,PLAN_ALONE,f,0,0,plan_blocks(m,p->words),0);}
  }
  /* design/torch-capture-change-request.md item 6; TensorFlow Lite 6cf18d88, tensorflow/lite/arena_planner.cc:433-436 */
  if(x->blocks && x->space==space){x->run=(uint32_t)x->items;plan_item_put(x,PLAN_ALONE,UINT32_MAX,0,0,x->blocks,space);}
  if(!space){
    x->table=(uint32_t)x->items;
    plan_item_put(x,PLAN_ALONE,UINT32_MAX,0,0,plan_blocks(m,plan_table(count,plans[0].flow->n,places)),0);
  }
}

/* design/plan-to-bridge-change-request.md R2, R8; design/disjoint-pages-change-request.md A3, C3.3, C3.4;
   C. Mukerji, M.Sc. thesis, McGill University, 1994, Algorithm 3.2.1 */
/* C. Mukerji, M.Sc. thesis, McGill University, 1994, §2.1, §3.1; L. J. Hendren, G. R. Gao, E. R. Altman and C. Mukerji,
   ACAPS Technical Memo 33, §2.2; M. Lam, PLDI 1988, §2.3; MPI 4.1 §3.7, §3.9;
   S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, (1) */
static void plan_landing(const struct plan_step *p,uint32_t j,int received,uint32_t far,uint64_t n,uint64_t c,
                         uint64_t *start,uint64_t *end){
  const uint64_t u=p->flow->copies,ring=u*n;
  uint64_t back=p->flow->start_lead,from=0,last=n-1,length;
  if(j!=MESH_ABSENT){
    last=p->function->value[j].end;
    if(!received){back=p->flow->value_lag[j]?p->flow->value_lag[j]-1:0;from=p->flow->value_lag[j]?0:p->function->value[j].begin;}
  }
  const uint64_t ahead=far-1;
  const int whole=far==UINT32_MAX || __builtin_mul_overflow(back+ahead,n,&length) ||
                  __builtin_add_overflow(length,last-from+1,&length) || length>=ring;
  *start=((c+u-back%u)%u)*n+from;
  *end=(*start+(whole?ring:length)-1)%ring;
}
static int plan_interval(const struct plan_step *plans,const struct plan_item *x,const uint64_t *operations,uint64_t *start,
                         uint64_t *end){
  if(x->kind!=PLAN_LOCAL && x->kind!=PLAN_ROUTED)return 0;
  const struct plan_step *p=plans+x->function;
  const uint64_t n=operations[x->function],c=x->copy;
  if(x->kind==PLAN_LOCAL){
    *start=c*n+p->function->value[x->index].begin;
    *end=c*n+p->function->value[x->index].end;
    return 1;
  }
  const uint32_t j=p->flow->demand[p->flow->value[p->flow->here[p->flow->member[p->flow->set_first[x->index]]]].demand].mine;
  int received=j==MESH_ABSENT;
  for(uint32_t k=p->flow->set_first[x->index];k<p->flow->set_first[x->index+1];k++)
    received|=p->flow->demand[p->flow->value[p->flow->here[p->flow->member[k]]].demand].source!=p->flow->self;
  plan_landing(p,j,received,p->flow->set_far[x->index],n,c,start,end);
  return 1;
}
/* design/torch-capture-change-request.md item 6; L. J. Hendren, G. R. Gao, E. R. Altman and C. Mukerji, ACAPS Technical
   Memo 33, §2.2; C. Mukerji, M.Sc. thesis, McGill University, 1994, §2.1, §3.1 */
static void plan_location(const struct plan_step *p,uint32_t j,uint64_t n,uint64_t c,uint64_t *start,uint64_t *end){
  const struct mesh_value *v=p->function->value+j;
  if(v->kind!=MESH_EXECUTION || p->flow->value_lag[j] || v->owned){*start=0;*end=(uint64_t)p->flow->copies*n-1;return;}
  if(!p->flow->named[j]){*start=c*n+v->begin;*end=c*n+v->end;return;}
  int received=0;
  uint32_t far=0;
  for(uint32_t i=0;i<p->flow->demands;i++){
    const struct mesh_plan_demand *d=p->flow->demand+i;
    if(d->mine!=j)continue;
    received|=d->source!=p->flow->self;
    if(p->flow->farthest[d->value]>far)far=p->flow->farthest[d->value];
  }
  plan_landing(p,j,received,far,n,c,start,end);
}
/* design/aot-compilation-change-request.md A10 and appendix item 6; design/disjoint-pages-change-request.md;
   mesh-collective.h mesh_plan_local: the program's calls (calls[j] call j's function, ordinal[j] its place among that
   function's calls, call_of[call_first[f] + o] function f's call o), the closure over them (NULL: no order between
   calls, different functions' copies always meet), its roots (roots[r], its uses use[use_first[r] .. use_first[r +
   1])), each root's `horizon` copies, and root_item[r] its first record; `grouped` each function's declared rows'
   grouping (mesh_declared_group's, the caller's, mesh_plan_program's; NULL, mesh_plan's: grouped by plan_run) */
struct plan_program { const struct mesh_plan_analysis *analysis; const struct mesh_value *roots;
                      uint32_t *root_item; struct mesh_plan_value *result; };

struct plan_arcs { const uint64_t *start,*end; const uint32_t *function,*index,*copy; const struct plan_step *plans;
                   uint32_t count; const struct plan_program *program; };
/* record r's block execution and its occurrences' count: a root's uses, a function copy's one call; 0 where it has no
   call occurrence (a lagged, transferred, invariant or owned value's, whose ring spans every execution; a root with no
   use) */
static int plan_lifetime(const struct plan_arcs *x,size_t r,uint32_t *block,size_t *n){
  const struct plan_program *g=x->program;
  if(x->function[r]==x->count){
    *block=x->copy[r];*n=g->analysis->use_first[x->index[r]+1]-g->analysis->use_first[x->index[r]];
    return *n>0 && !(g->analysis->carried && g->analysis->carried[x->index[r]]);
  }
  const struct plan_step *p=x->plans+x->function[r];
  const struct mesh_value *v=p->function->value+x->index[r];
  const int occurs=!p->flow->named[x->index[r]] && v->kind==MESH_EXECUTION && !p->result->value[x->index[r]].lag && !v->owned;
  if(occurs){*block=x->copy[r]/p->flow->calls;*n=1;}
  return occurs;
}
/* occurrence k of record r: its call and its value's compiled begin and end in that call */
static void plan_occurrence(const struct plan_arcs *x,size_t r,size_t k,uint32_t *call,uint32_t *begin,
                            uint32_t *end){
  const struct plan_program *g=x->program;
  const struct mesh_value *v;
  if(x->function[r]==x->count){
    const struct mesh_local_use *u=g->analysis->use+g->analysis->use_first[x->index[r]]+k;
    *call=u->call;v=x->plans[g->analysis->calls[u->call]].function->value+u->value;
  }else{
    const struct plan_step *p=x->plans+x->function[r];
    *call=g->analysis->call_group->order[g->analysis->call_group->first[x->function[r]]+x->copy[r]%p->flow->calls];v=p->function->value+x->index[r];
  }
  *begin=v->begin;*end=v->end;
}
/* every occurrence of record a before every occurrence of record b: in one call by compiled position, across calls by
   the closure (row b's call holds a's call) */
static int plan_before(const struct plan_arcs *x,size_t a,size_t na,size_t b,size_t nb){
  const struct mesh_graph_closure *c=x->program->analysis->closure;
  for(size_t i=0;i<na;i++)for(size_t k=0;k<nb;k++){
    uint32_t ca,ba,ea,cb,bb,eb;
    plan_occurrence(x,a,i,&ca,&ba,&ea);
    plan_occurrence(x,b,k,&cb,&bb,&eb);
    if(ca==cb?ea>=bb:!(c->reached[(size_t)cb*c->words+ca/64]>>(ca%64)&1))return 0;
  }
  return 1;
}
/* design/aot-compilation-change-request.md A10: two copies meet unless the source order separates their occurrences.
   Without the calls' closure (mesh_plan) one function's copies meet as their circular lifetimes do and other copies
   always meet; with it a copy of a lagged, transferred or invariant value meets one function's others by that ring and
   every other copy, and occurrence-bearing copies meet unless both are of one block and every occurrence of one
   precedes every occurrence of the other (plan_before: one call's compiled begin and end, distinct calls' closure) */
static int plan_intersect(const void *context,size_t a,size_t b){
  const struct plan_arcs *x=context;
  const int one=x->function[a]==x->function[b] && x->function[a]<x->count;
  uint32_t ta,tb;
  size_t na,nb;
  if(!x->program->analysis->closure)return !one || mesh_graph_arcs_overlap(x->start[a],x->end[a],x->start[b],x->end[b]);
  const int la=plan_lifetime(x,a,&ta,&na),lb=plan_lifetime(x,b,&tb,&nb);
  if(one && (!la || !lb))return mesh_graph_arcs_overlap(x->start[a],x->end[a],x->start[b],x->end[b]);
  if(!la || !lb || ta!=tb)return 1;
  return !plan_before(x,a,na,b,nb) && !plan_before(x,b,nb,a,na);
}
/* design/torch-capture-change-request.md item 6; TensorFlow Lite 6cf18d88,
   tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_size_assignment.cc:68-143 and
   tensorflow/lite/arena_planner.h:141-149; GaijinEntertainment/DagorEngine 2baf1947,
   prog/gameLibs/render/daFrameGraph/backend/resourceScheduling/heapLayout.h:19: every function's run copies and
   every run root's copies (plan_program) in one assignment */
static int plan_block(struct plan_step *plans,uint32_t count,uint64_t quantum,const uint64_t *operations,
                      struct plan_placement *x,const struct plan_program *program){
  size_t records=0;
  for(uint32_t f=0;f<count;f++)for(uint32_t j=0;j<plans[f].function->values;j++)
    if(plans[f].function->value[j].run)records+=plan_copies(plans[f].flow,j);
  for(uint32_t r=0;r<program->analysis->roots;r++)if(program->roots[r].run)records+=program->analysis->horizon;
  uint64_t *size=malloc((records+1)*sizeof *size),*start=malloc((records+1)*sizeof *start),
           *end=malloc((records+1)*sizeof *end),*offset=malloc((records+1)*sizeof *offset),total=0;
  uint32_t *function=malloc((records+1)*sizeof *function),*index=malloc((records+1)*sizeof *index),
           *copy=malloc((records+1)*sizeof *copy);
  x->allocations=records;x->size=size;x->start=start;x->end=end;x->offset=offset;
  x->function=function;x->index=index;x->copy=copy;
  x->at=malloc((records+1)*sizeof *x->at);
  int status=!size || !start || !end || !offset || !function || !index || !copy || !x->at?ENOMEM:0;
  uint32_t r=0;
  for(uint32_t f=0;!status && f<count;f++){
    struct plan_step *p=plans+f;
    const uint64_t n=operations[f],ring=(uint64_t)p->flow->copies*n;
    for(uint32_t j=0;j<p->function->values;j++){
      const struct mesh_value *v=p->function->value+j;
      if(!v->run)continue;
      if(p->flow->named[j] || v->wire)x->space=1;
      p->item[j]=r;
      for(uint32_t c=0;c<plan_copies(p->flow,j);c++,r++){
        uint64_t last;
        plan_location(p,j,n,c,start+r,&last);
        end[r]=(last+1)%ring;
        size[r]=v->section.bytes;
        function[r]=f;index[r]=j;copy[r]=c;
      }
    }
  }
  for(uint32_t k=0;!status && k<program->analysis->roots;k++){
    if(!program->roots[k].run)continue;
    if(program->analysis->carried && program->analysis->carried[k])x->space=1;
    program->root_item[k]=r;
    for(uint32_t t=0;t<program->analysis->horizon;t++,r++){
      start[r]=end[r]=0;size[r]=program->roots[k].section.bytes;
      function[r]=count;index[r]=k;copy[r]=t;
    }
  }
  const struct plan_arcs arcs={start,end,function,index,copy,plans,count,program};
  if(!status)status=mesh_graph_greedy_by_size(records,size,NULL,plan_intersect,&arcs,quantum,offset,&total);
  for(size_t q=0;!status && q<records;q++)x->at[q]=offset[q]/quantum;
  if(!status)x->blocks=mesh_plan_quanta(total,quantum);
  return status;
}
/* a function's operations: one past the last its execution values span, at least 1 */
static void plan_placement_free(struct plan_placement *x,int loaded){
  if(!x)return;
  if(!loaded){
    free(x->item);free(x->count);free(x->chain);free(x->at);
    free(x->size);free(x->start);free(x->end);free(x->offset);free(x->operations);
    free(x->function);free(x->index);free(x->copy);
  }
  free(x);
}
static uint64_t plan_operations(const struct plan_step *p){
  uint64_t operations=1;
  for(uint32_t j=0;j<p->function->values;j++){
    const struct mesh_value *v=p->function->value+j;
    if(v->kind==MESH_EXECUTION && (uint64_t)v->end+1>operations)operations=(uint64_t)v->end+1;
  }
  return operations;
}
static int plan_place(struct plan_step *plans,uint32_t count,const struct mesh_plan_geometry *m,struct plan_placement *x,
                      const struct plan_program *program){
  *x=(struct plan_placement){0};
  uint64_t *operations=malloc(((size_t)count+1)*sizeof *operations);
  x->operations=operations;
  if(!operations)return ENOMEM;
  for(uint32_t f=0;f<count;f++)operations[f]=plan_operations(plans+f);
  const int blocked=plan_block(plans,count,plan_quantum(m),operations,x,program);
  if(blocked){return blocked;}
  plan_items(plans,count,m,x,1);
  plan_items(plans,count,m,x,0);
  if(x->items>=UINT32_MAX || x->elements>=SIZE_MAX/sizeof(uint64_t)){return ENOMEM;}
  x->item=malloc(((size_t)x->items+1)*sizeof *x->item);
  x->count=malloc(((size_t)x->items+1)*sizeof *x->count);
  x->chain=malloc(((size_t)x->elements+1)*sizeof *x->chain);
  if(!x->item || !x->count || !x->chain){return ENOMEM;}
  x->items=x->elements=0;
  plan_items(plans,count,m,x,1);
  x->split=(uint32_t)x->items;
  plan_items(plans,count,m,x,0);
  x->range[1]=m->arena[1];
  x->range[0]=m->arena[0];
  const int one=x->range[0].first==x->range[1].first && x->range[0].count==x->range[1].count;
  const uint32_t first[3]={0,one?(uint32_t)x->items:x->split,(uint32_t)x->items};
  uint64_t *counts=malloc(((size_t)x->items+1)*sizeof *counts),
           *start=malloc(((size_t)x->items+1)*sizeof *start),*end=malloc(((size_t)x->items+1)*sizeof *end),
           *out=malloc(((size_t)x->elements+1)*sizeof *out);
  uint32_t *index=malloc(((size_t)x->items+1)*sizeof *index);
  int status=!counts || !start || !end || !out || !index?ENOMEM:0;
  for(uint32_t s=0;!status && s<2;s++){
    if(first[s]==first[s+1])continue;
    uint64_t blocks=0;
    uint32_t rest=0;
    for(uint32_t i=first[s];i<first[s+1];i++)
      if(!plan_interval(plans,x->item+i,operations,start,end)){index[rest]=i;counts[rest++]=x->count[i];}
    if(rest){
      status=mesh_graph_chains(rest,counts,plan_alone,NULL,out,&blocks);
      for(uint64_t r=0,o=0;!status && r<rest;r++)
        for(uint64_t k=0;k<counts[r];k++)x->chain[x->item[index[r]].element+k]=out[o++];
    }
    for(uint32_t f=0;!status && f<count;f++){
      uint32_t n=0;
      for(uint32_t i=first[s];i<first[s+1];i++)
        if(x->item[i].function==f && plan_interval(plans,x->item+i,operations,start+n,end+n)){
          index[n]=i;counts[n++]=x->count[i];
        }
      if(!n)continue;
      uint64_t colors=0;
      status=mesh_graph_fat_cover((uint64_t)plans[f].flow->copies*operations[f],n,start,end,counts,out,&colors);
      for(uint64_t r=0,o=0;!status && r<n;r++)
        for(uint64_t k=0;k<counts[r];k++)x->chain[x->item[index[r]].element+k]=blocks+out[o++];
      blocks+=colors;
    }
    if(!status && blocks>x->range[x->item[first[s]].space].count/m->block)status=ENOMEM;
    x->used[one?0:x->item[first[s]].space]=blocks;
  }
  free(counts);free(start);free(end);free(out);free(index);
  return status;
}
static uint32_t plan_page(const struct plan_placement *x,uint32_t block,uint32_t i,uint64_t k){
  return (uint32_t)(x->range[x->item[i].space].first+x->chain[x->item[i].element+k]*block);
}

/* design/plan-to-bridge-change-request.md R2; design/torch-capture-change-request.md item 6 */
static uint32_t plan_frame(const struct plan_step *p,const struct plan_placement *x,uint32_t block,uint32_t j,uint32_t c,
                           uint32_t k){
  if(p->function->value[j].run)return plan_page(x,block,x->run,x->at[p->item[j]+c]+k);
  if(p->flow->block_first[j]==MESH_ABSENT)return plan_page(x,block,p->item[j]+c,k);
  const uint32_t at=p->flow->block_first[j]+k,s=p->flow->block_set[at];
  return plan_page(x,block,s==MESH_ABSENT?p->item[j]+c:p->set_item+s*p->flow->copies+c,p->flow->block_element[at]);
}
static int plan_rows(struct plan_step *plans,uint32_t count,const struct mesh_plan_geometry *m,const struct plan_placement *x,
                     uint32_t *page,uint64_t rows,const struct plan_program *program){
  const uint32_t block=m->block;
  for(uint32_t f=0;f<count;f++){
    const struct plan_step *p=plans+f;
    for(uint32_t j=0;j<p->function->values;j++){
      const struct mesh_section v=p->result->value[j].section;
      for(uint32_t c=0;c<v.count;c++)for(uint32_t k=0;k<v.stride;k++)
        page[v.first+(size_t)c*v.stride+k]=plan_frame(p,x,block,j,c,k);
    }
  }
  for(uint32_t r=0;r<program->analysis->roots;r++){
    const struct mesh_section v=program->result[r].section;
    for(uint32_t c=0;program->roots[r].run && c<v.count;c++)for(uint32_t k=0;k<v.stride;k++)
      page[v.first+(size_t)c*v.stride+k]=plan_page(x,block,x->run,x->at[program->root_item[r]+c]+k);
  }
  unsigned char *seen=calloc((size_t)(((uint64_t)m->arena[0].first+m->arena[0].count>(uint64_t)m->arena[1].first+m->arena[1].count?(uint64_t)m->arena[0].first+m->arena[0].count:(uint64_t)m->arena[1].first+m->arena[1].count)/block)+1,1);
  if(!seen)return ENOMEM;
  for(uint64_t r=0;r<rows;r++)if(seen[page[r]/block]<2)seen[page[r]/block]++;
  for(uint32_t f=0;f<count;f++)for(uint32_t j=0;j<plans[f].function->values;j++){
    struct mesh_plan_value *v=plans[f].result->value+j;
    const uint64_t end=v->section.first+(uint64_t)v->section.count*v->section.stride;
    v->shares=0;
    for(uint64_t r=v->section.first;r<end && !v->shares;r++)v->shares=seen[page[r]/block]>1;
  }
  free(seen);
  return 0;
}

/* design/disjoint-pages-change-request.md C3.3, C3.7 */

static int plan_trees(struct mesh_plan_flow *p){
  const uint32_t n=p->n;
  p->adjacency=calloc((size_t)p->made+1,sizeof *p->adjacency);
  p->members=malloc(((size_t)p->made+1)*sizeof *p->members);
  p->lead=malloc(((size_t)p->made+1)*sizeof *p->lead);
  p->comm=malloc(((size_t)p->made+1)*sizeof *p->comm);
  if(!p->adjacency || !p->members || !p->lead || !p->comm)return ENOMEM;
  p->comms=0;
  for(uint32_t i=0;i<p->demands;i++){
    const struct mesh_plan_demand *d=p->demand+i;
    uint64_t k=0;
    for(uint32_t j=d->tree;j<d->tree+d->trees;j++){
      int status=mesh_graph_tree_create(n,d->source,p->parent+p->tree[j]*n,p->adjacency+j);
      if(status)return status;
      const size_t *first=p->adjacency[j]->children->first,*child=p->adjacency[j]->children->order;
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
static uint32_t plan_communication(const struct mesh_plan_flow *p,uint32_t j,uint64_t page,uint32_t head){
  return (uint32_t)(p->comm[j]+page*(uint64_t)__builtin_popcountll(p->members[j])+plan_rank(p->members[j],head));
}

/* design/disjoint-pages-change-request.md C3.7; design/plan-to-bridge-change-request.md R5 */
static void plan_walk(struct mesh_plan_flow *p){
  const uint32_t n=p->n,self=p->self;
  uint64_t at=0,receives=0;
  if(!p->entry)p->ranges=p->consumed=p->sending=0,p->toward=0;
  for(uint32_t i=0;i<p->demands;i++){
    struct mesh_plan_demand *d=p->demand+i;
    uint32_t field=0,sends=0;
    uint64_t k=0;
    for(uint32_t j=d->tree;j<d->tree+d->trees;j++){
      const uint64_t end=k+p->run[j];
      const uint32_t set=d->set?j+1:0;
      const struct mesh_graph_tree *tree=p->adjacency[j];
      const size_t *first=tree->children->first,*child=tree->children->order;
      uint32_t in=MESH_ABSENT;
      mesh_graph_climb(tree->parent,n,d->source,self,plan_tree_up,&in);
      if(in==MESH_ABSENT && first[self+1]==first[self]){k=end;continue;}
      if(k<end){
        if(in!=MESH_ABSENT)p->toward|=UINT64_C(1)<<in;
        for(size_t c=first[self];c<first[self+1];c++)p->toward|=UINT64_C(1)<<child[c];
      }
      for(;k<end;k++){
        uint32_t mine=MESH_ABSENT;
        if(in!=MESH_ABSENT){
          mine=field++;
          if(p->entry)p->entry[at]=(struct mesh_plan_entry){.node=in,.direction=MESH_RECEIVE,.demand=i,.field=mine,
                                                       .communication=plan_communication(p,j,k-p->lead[j],self),
                                                       .set=set,.instruction=k};
          at++;
        }
        for(size_t c=first[self];c<first[self+1];c++,at++,sends++)
          if(p->entry)p->entry[at]=(struct mesh_plan_entry){.node=(uint32_t)child[c],.direction=MESH_SEND,.demand=i,
                                                       .field=mine,
                                                       .communication=plan_communication(p,j,k-p->lead[j],
                                                                                         (uint32_t)child[c]),
                                                       .set=set,.instruction=k};
      }
    }
    if(!p->entry){
      d->received=field;
      d->range=field?p->ranges++:MESH_ABSENT;
      d->consumed=field && d->leaves>>self&1?p->consumed++:MESH_ABSENT;
      d->sends=sends;
      d->sending=sends?p->sending++:MESH_ABSENT;
      receives+=field;
    }
  }
  p->entries=at;
  if(!p->entry)p->receives=receives;
}

/* design/plan-to-bridge-change-request.md R3 to R5 */

struct plan_path { const uint32_t *link_of; uint32_t n,depth,below,*link,*head; };
static void plan_path_step(void *consumer,uint32_t at,uint32_t step){
  struct plan_path *path=consumer;
  if(step){
    path->link[path->depth-step]=path->link_of[(size_t)at*path->n+path->below];
    path->head[path->depth-step]=path->below;
  }
  path->below=at;
}
/* design/plan-to-bridge-change-request.md R4 */
static struct plan_span plan_tasks(const struct mesh_plan_flow *p,const struct plan_span *consumer,const uint32_t *reader,
                                   uint32_t i,int produced,int own){
  if(produced)return (struct plan_span){p->of[i],1};
  if(own)return (struct plan_span){reader[p->of[i]],1};
  return consumer[i];
}
/* design/plan-to-bridge-change-request.md R5 */
static int plan_entry_order(const void *a,const void *b){
  const struct plan_entry *x=a,*y=b;
  return (x->place>y->place)-(x->place<y->place);
}

/* design/plan-to-bridge-change-request.md R1, R3 to R5; design/aot-compilation-change-request.md A3, A6;
   O. Sinnen, Task Scheduling for Parallel Systems, Wiley, 2007; O. Sinnen and L. A. Sousa, IEEE TPDS 16(6), 2005,
   Definitions 1, 2 and 4; M. Cowan et al., MSCCLang, ASPLOS 2023, §4.2, §5.2;
   F. B. Schneider, ACM Comput. Surv. 22(4), 1990 */
static int plan_schedule(struct plan_step *p,const struct mesh_function *function){
  const struct mesh_weight *weight=p->flow->weights;
  const struct mesh_dependence *dependence=p->flow->dependence;
  const uint32_t n=p->flow->n,demands=p->flow->demands,count=p->flow->transfers,dependences=p->flow->dependences,placed=function->tasks;
  if(placed && (!function->task || !function->sender || !function->receiver ||
                (function->befores && !function->before)))return EINVAL;
  for(uint32_t k=0;k<function->befores;k++)
    if(function->before[k].from>=placed || function->before[k].to>=placed)return EINVAL;
  for(uint32_t i=0;placed && i<count;i++)if(function->sender[i]>=placed || function->receiver[i]>=placed)return EINVAL;
  const struct mesh_declared *declared=p->flow->declared;
  struct plan_schedule_result *kept=p->schedule=calloc(1,sizeof *kept);
  if(!kept)return ENOMEM;
  uint32_t link_of[(size_t)n*n+1],path_link[(size_t)n+1],path_head[(size_t)n+1];
  uint32_t *reader=kept->reader=malloc(((size_t)demands+1)*sizeof *reader);
  struct plan_span *consumer=kept->consumer=malloc(((size_t)count+1)*sizeof *consumer);
  if(!reader || !consumer)return ENOMEM;
  uint64_t tasks=demands,edges=0,entries=0;
  int status=0;
  for(uint32_t d=0;d<demands;d++)
    if((reader[d]=p->flow->demand[d].own?(uint32_t)tasks:MESH_ABSENT)!=MESH_ABSENT)edges++,tasks++;
  if(tasks>=UINT32_MAX)status=ENOMEM;
  for(uint32_t i=0;!status && i<count;i++){
    const struct mesh_plan_demand *d=p->flow->demand+p->flow->of[i];
    consumer[i]=(struct plan_span){(uint32_t)tasks,declared[i].kind==MESH_PAGES?(uint32_t)d->instructions:1};
    tasks+=consumer[i].count;
    edges+=d->instructions;
    for(uint32_t j=d->tree;j<d->tree+d->trees;j++)
      if(p->flow->run[j])entries+=p->flow->run[j]*mesh_graph_climb(p->flow->parent+p->flow->tree[j]*n,n,d->source,declared[i].destination,
                                                       NULL,NULL);
    if(tasks>=UINT32_MAX || edges>=UINT32_MAX || entries>=UINT32_MAX)status=ENOMEM;
  }
  for(uint32_t k=0;!status && k<dependences;k++){
    const struct mesh_dependence *x=dependence+k;
    if(x->lag)continue;
    const struct plan_span a=plan_tasks(p->flow,consumer,reader,x->from,!!(x->kinds&MESH_FROM_PRODUCED),
                                        !!(x->kinds&MESH_FROM_SOURCE_READ)),
                           b=plan_tasks(p->flow,consumer,reader,x->to,!!(x->kinds&MESH_TO_PRODUCED),
                                        !!(x->kinds&MESH_TO_SOURCE_READ));
    if((edges+=(uint64_t)a.count*b.count)>=UINT32_MAX)status=ENOMEM;
  }
  /* design/torch-source-lowering-change-request.md item 3; design/aot-compilation-change-request.md A6 */
  const uint32_t first=(uint32_t)tasks;
  tasks+=placed;
  edges+=function->befores;
  for(uint32_t i=0;placed && i<count;i++)edges+=1+(uint64_t)consumer[i].count;
  if(tasks>=UINT32_MAX || edges>=UINT32_MAX)status=ENOMEM;
  const uint64_t instructions=tasks+p->flow->comms,dependencies=edges+entries;
  if(instructions>=UINT32_MAX)status=ENOMEM;
  kept->first=first;kept->tasks=(uint32_t)tasks;kept->instructions=(uint32_t)instructions;
  kept->edges=(size_t)edges;kept->entries=(size_t)entries;kept->dependencies=(size_t)dependencies;
  double *w=kept->w=status?NULL:malloc(((size_t)instructions+1)*sizeof *w),
         *cost=kept->cost=status?NULL:malloc(((size_t)edges+1)*sizeof *cost),
         *rate=kept->rate=malloc((2*(size_t)p->flow->pairs+1)*sizeof *rate),
         *completion=kept->completion=status?NULL:malloc(((size_t)instructions+1)*sizeof *completion),
         *priority=kept->priority=status?NULL:malloc(((size_t)instructions+1)*sizeof *priority);
  uint32_t *resource=kept->resource=status?NULL:malloc(((size_t)instructions+1)*sizeof *resource),
           *from=kept->from=status?NULL:malloc(((size_t)edges+1)*sizeof *from),
           *to=kept->to=status?NULL:malloc(((size_t)edges+1)*sizeof *to),
           *link=kept->link=status?NULL:malloc(((size_t)entries+1)*sizeof *link),
           *communication=kept->communication=status?NULL:malloc(((size_t)entries+1)*sizeof *communication),
           *before=kept->before=status?NULL:malloc(((size_t)dependencies+1)*sizeof *before),
           *after=kept->after=status?NULL:malloc(((size_t)dependencies+1)*sizeof *after),
           *place=kept->place=status?NULL:malloc(((size_t)instructions+1)*sizeof *place);
  size_t *route=kept->route=status?NULL:malloc(((size_t)edges+1)*sizeof *route);
  if(!status && (!w || !cost || !rate || !completion || !priority || !resource || !from || !to || !link || !communication ||
                 !before || !after || !place || !route))status=ENOMEM;
  if(!status){
    /* design/aot-compilation-change-request.md A3; design/plan-to-bridge-change-request.md R5 */
    for(uint32_t d=0;d<demands;d++){
      const struct mesh_plan_demand *x=p->flow->demand+d;
      w[d]=placed?0:weight[x->first].produced;
      resource[d]=x->source;
      if(reader[d]!=MESH_ABSENT){w[reader[d]]=placed?0:weight[x->first].read;resource[reader[d]]=x->source;}
    }
    for(uint32_t i=0;i<count;i++)for(uint32_t q=0;q<consumer[i].count;q++){
      w[consumer[i].first+q]=placed?0:weight[i].consumed;
      resource[consumer[i].first+q]=declared[i].destination;
    }
    for(uint32_t k=0;k<placed;k++){
      w[first+k]=function->task[k].length;
      resource[first+k]=n+2*p->flow->pairs+function->task[k].resource;
    }
    double most=0;
    for(size_t v=0;v<(size_t)n*n;v++)link_of[v]=MESH_ABSENT;
    for(uint32_t e=0;e<p->flow->pairs;e++)if(p->flow->capacity[e]>most)most=p->flow->capacity[e];
    for(uint32_t e=0;e<p->flow->pairs;e++){
      link_of[(size_t)p->flow->from[e]*n+p->flow->to[e]]=2*e;
      link_of[(size_t)p->flow->to[e]*n+p->flow->from[e]]=2*e+1;
      rate[2*e]=rate[2*e+1]=p->flow->capacity[e]/most;
    }
    size_t e=0,r=0;
    for(uint32_t i=0;i<count;i++){
      const struct mesh_plan_demand *d=p->flow->demand+p->flow->of[i];
      const uint32_t z=declared[i].destination;
      for(uint32_t j=d->tree;j<d->tree+d->trees;j++){
        if(!p->flow->run[j])continue;
        const uint32_t *parent=p->flow->parent+p->flow->tree[j]*n;
        struct plan_path path={link_of,n,mesh_graph_climb(parent,n,d->source,z,NULL,NULL),z,path_link,path_head};
        mesh_graph_climb(parent,n,d->source,z,plan_path_step,&path);
        for(uint64_t q=0;q<p->flow->run[j];q++,e++){
          from[e]=p->flow->of[i];
          to[e]=consumer[i].first+(declared[i].kind==MESH_PAGES?(uint32_t)(p->flow->lead[j]+q):0);
          cost[e]=weight[i].page;
          route[e]=r;
          for(uint32_t s=0;s<path.depth;s++,r++){
            link[r]=path_link[s];
            communication[r]=plan_communication(p->flow,j,q,path_head[s]);
          }
        }
      }
    }
    for(uint32_t d=0;d<demands;d++)
      if(reader[d]!=MESH_ABSENT){from[e]=d;to[e]=reader[d];cost[e]=0;route[e++]=r;}
    for(uint32_t k=0;k<dependences;k++){
      const struct mesh_dependence *x=dependence+k;
      if(x->lag)continue;
      const struct plan_span a=plan_tasks(p->flow,consumer,reader,x->from,!!(x->kinds&MESH_FROM_PRODUCED),
                                          !!(x->kinds&MESH_FROM_SOURCE_READ)),
                             b=plan_tasks(p->flow,consumer,reader,x->to,!!(x->kinds&MESH_TO_PRODUCED),
                                          !!(x->kinds&MESH_TO_SOURCE_READ));
      for(uint32_t s=0;s<a.count;s++)for(uint32_t t=0;t<b.count;t++){
        from[e]=a.first+s;to[e]=b.first+t;cost[e]=0;route[e++]=r;
      }
    }
    for(uint32_t k=0;k<function->befores;k++){
      from[e]=first+function->before[k].from;to[e]=first+function->before[k].to;cost[e]=0;route[e++]=r;
    }
    for(uint32_t i=0;placed && i<count;i++){
      from[e]=first+function->sender[i];to[e]=p->flow->of[i];cost[e]=0;route[e++]=r;
      for(uint32_t q=0;q<consumer[i].count;q++){
        from[e]=consumer[i].first+q;to[e]=first+function->receiver[i];cost[e]=0;route[e++]=r;
      }
    }
    route[e]=r;
  }
  if(!status){
    size_t k=0;
    for(size_t x=0;x<edges;x++){
      uint32_t tail=from[x];
      for(size_t r=route[x];r<route[x+1];r++,k++){before[k]=tail;tail=after[k]=(uint32_t)tasks+communication[r];}
      before[k]=tail;after[k++]=to[x];
    }
    /* design/plan-to-bridge-change-request.md R5 */
    for(uint64_t c=0;c<p->flow->comms;c++){w[tasks+c]=0;resource[tasks+c]=n;}
    for(size_t x=0;x<edges;x++)for(size_t r=route[x];r<route[x+1];r++){
      const double occupy=cost[x]/rate[link[r]];
      if(occupy>w[tasks+communication[r]])w[tasks+communication[r]]=occupy;
      resource[tasks+communication[r]]=n+link[r];
    }
    /* design/plan-to-bridge-change-request.md R5 */
    status=mesh_graph_completion_times((uint32_t)instructions,w,resource,n+2*p->flow->pairs+function->resources,
                                       (size_t)dependencies,before,after,completion);
    for(uint32_t k=0;!status && k<placed;k++)p->result->completion[k]=completion[first+k];
    /* L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, Math. Oper. Res. 22(3), 1997, §2; C.-Y. Chen,
       arXiv 2205.02474, 2022, (1a) */
    if(!status && !(p->completed=calloc((size_t)count+1,sizeof *p->completed)))status=ENOMEM;
    for(uint32_t i=0;!status && i<count;i++)for(uint32_t q=0;q<consumer[i].count;q++)
      if(completion[consumer[i].first+q]>p->completed[i])p->completed[i]=completion[consumer[i].first+q];
    if(!status){
      for(uint64_t j=0;j<instructions;j++)priority[j]=-completion[j];
      status=mesh_graph_heap_order((uint32_t)instructions,priority,(size_t)dependencies,before,after,place);
    }
    for(uint32_t k=0;!status && k<placed;k++)p->result->order[k]=place[first+k];
  }
  if(!status){
    /* design/plan-to-bridge-change-request.md R5 */
    for(uint64_t j=0;j<p->flow->entries;j++)p->entry[j].place=place[tasks+p->entry[j].source->communication];
    qsort(p->entry,p->flow->entries,sizeof *p->entry,plan_entry_order);
  }
  return status;
}

/* design/aot-compilation-change-request.md A7, A9, A10, appendix item 2; design/disjoint-pages-change-request.md C3.3;
   S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §3, §4.1, §9, Lemma 1,
   Theorem 2, Figs. 6 and 8, (5), (7) */
struct plan_room_edges { uint32_t *from,*to; uint64_t *delay; size_t edges; struct mesh_planned_cause *cause,now; };
static void plan_room_edge(struct plan_room_edges *graph,uint32_t a,uint32_t b,uint64_t d){
  graph->from[graph->edges]=a;graph->to[graph->edges]=b;graph->delay[graph->edges]=d;
  if(graph->cause)graph->cause[graph->edges]=graph->now;
  graph->edges++;
}
static void plan_room_cause(struct plan_room_edges *graph,uint32_t kind,uint32_t pair){
  graph->now=(struct mesh_planned_cause){kind,pair};
}
static uint32_t plan_room_head(const void *context,size_t k){
  const struct plan_graph_result *g=context;
  return g->to[g->room_rows->order[k]];
}
static uint64_t plan_room_lag(const void *context,size_t k){
  const struct plan_graph_result *g=context;
  return g->delay[g->room_rows->order[k]];
}
/* S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §4.1;
   design/aot-compilation-change-request.md A3, appendix item 2 */
static int plan_times(struct mesh_plan_flow *p){
  const uint32_t n=p->n;
  double *page=calloc((size_t)p->demands+1,sizeof *page);
  if(!page)return ENOMEM;
  for(uint32_t i=0;i<p->transfers;i++){
    const uint32_t d=p->of[i];
    if(p->weights[i].page>page[d])page[d]=p->weights[i].page;
    p->time[p->declared[i].destination]+=
      p->weights[i].consumed*(p->declared[i].kind==MESH_PAGES?(double)p->demand[d].instructions:1.0);
  }
  for(uint32_t i=0;i<p->demands;i++){
    const struct mesh_plan_demand *d=p->demand+i;
    p->time[d->source]+=p->weights[d->first].produced+(d->own?p->weights[d->first].read:0.0);
    for(uint32_t j=d->tree;j<d->tree+d->trees;j++){
      if(!p->run[j])continue;
      const double hops=(double)p->run[j]*page[i];
      const size_t *first=p->adjacency[j]->children->first,*child=p->adjacency[j]->children->order;
      for(uint32_t y=0;y<n;y++)for(size_t c=first[y];c<first[y+1];c++){p->time[y]+=hops;p->time[child[c]]+=hops;}
    }
    p->forwarded+=d->forwards;
  }
  free(page);
  return 0;
}
/* design/aot-compilation-change-request.md, appendix item 2: call j's submission at participating node v (the
   program's nodes), after its own instance's work vertices */
static uint32_t plan_submission(const struct plan_step *plans,const uint32_t *calls,const uint32_t *actor,uint64_t program,
                                uint32_t s,uint32_t j,uint32_t v){
  if(v==s)return 2*j;
  return actor[j]+(uint32_t)__builtin_popcountll(plans[calls[j]].flow->nodes)+plan_rank(program&~(UINT64_C(1)<<s),v);
}
/* design/synchronization-signature-change-request.md items 1, 2 and 4: the plan's own synchronization as it is
   compiled, BSL §9's added edges as `struct mesh_sync` entries. A wait is mesh-dataflow.c mesh_transfers_start's, at a
   node's host before the call's submission: the count of its entry reaching `need` (K every + base, K the block); a
   raise is mesh-records.c counts_tell's, the bridge at the edge's source node telling its plan's `submissions`; the
   count words start at 0 (mesh_plan's state cleared). Processors: node v's host 2v and bridge 2v + 1 */
struct plan_signed { struct mesh_graph_repeated *row; size_t count,held; };
static int plan_signed_put(struct plan_signed *o,struct mesh_graph_repeated x){
  if(o->count==o->held){
    const size_t grown=o->held?2*o->held:256;
    struct mesh_graph_repeated *more=realloc(o->row,grown*sizeof *more);
    if(!more)return ENOMEM;
    o->row=more;o->held=grown;
  }
  o->row[o->count++]=x;
  return 0;
}
static uint32_t plan_tell_key(const void *context,size_t k){ return ((const uint32_t *)context)[k]; }
static int plan_signature(struct plan_step *plans,uint32_t count,const uint32_t *calls,uint32_t J,uint64_t program,uint32_t s,
                          uint32_t n,uint32_t V,const double *time,const struct mesh_graph_task *task,const uint32_t *node,
                          const uint32_t *observer,const uint64_t *lagged,const uint32_t *actor,
                          const size_t *place,
                          const size_t *row,size_t built,uint32_t added,const uint32_t *from,const uint32_t *to,
                          const uint64_t *delay,const uint64_t *added_delay,double lambda,const size_t *wait_first,
                          const size_t *wait_order,struct mesh_graph_signature **kept){
  struct mesh_graph_processor processor[2*MESH_NODES_MOST];
  for(uint32_t v=0;v<n;v++){processor[2*v]=(struct mesh_graph_processor){MESH_GRAPH_HOST,v};
                            processor[2*v+1]=(struct mesh_graph_processor){MESH_GRAPH_BRIDGE,v};}
  struct mesh_graph_dependence *dependence=malloc((built+1)*sizeof *dependence);
  uint64_t *offset=malloc(((size_t)V+1)*sizeof *offset);
  uint32_t *key=malloc(((size_t)added+1)*sizeof *key);
  size_t *first=malloc(((size_t)n*count+2)*sizeof *first),*tell=malloc(((size_t)added+1)*sizeof *tell);
  struct plan_signed o={NULL,0,0};
  struct mesh_graph_signature signature;
  int status=dependence && offset && key && first && tell && (*kept=calloc(1,sizeof **kept))?0:ENOMEM;
  uint64_t executions=1;
  /* the program graph without its emitted synchronization (BSL §4.1's IPC graph and A10's orders) is the relation the
     waits answer to; the added edges are checked against it by mesh_graph_resynchronized, never taken as their own
     dependences */
  for(size_t e=0;!status && e<built;e++)dependence[e]=(struct mesh_graph_dependence){from[e],to[e],delay[e]};
  for(uint32_t x=0;!status && x<V;x++)offset[x]=lagged[x];
  for(uint32_t a=0;!status && a<added;a++){
    key[a]=node[from[built+a]]*count+calls[observer[from[built+a]]];
    if(delay[built+a]+1>executions)executions=delay[built+a]+1;
    status=plan_signed_put(&o,(struct mesh_graph_repeated){.processor=2*node[to[built+a]],.primitive=MESH_GRAPH_INITIAL,
                                                           .vertex=MESH_GRAPH_NONE,.object=MESH_GRAPH_PLAN_COUNT+a,
                                                           .startup=MESH_GRAPH_NONE});
  }
  if(!status)mesh_graph_rows(n*count,added,plan_tell_key,key,first,tell);
  for(uint32_t v=0;!status && v<n;v++){
    if(!(program>>v&1))continue;
    for(uint32_t j=0;!status && j<J;j++){
      const struct plan_step *p=plans+calls[j];
      const uint32_t f=calls[j],i=place[j],u=plan_submission(plans,calls,actor,program,s,j,v);
      const uint64_t nf=p->flow->calls;
      const size_t *waits=wait_first+(size_t)v*J+row[f];
      for(size_t q=waits[i];!status && q<waits[i+1];q++){
        const uint32_t a=(uint32_t)wait_order[q],tail=from[built+a],every=plans[calls[observer[tail]]].flow->calls;
        status=plan_signed_put(&o,(struct mesh_graph_repeated){.processor=2*v,.primitive=MESH_GRAPH_HOST_WAIT,
                                                               .vertex=to[built+a],.object=MESH_GRAPH_PLAN_COUNT+a,
                                                               .every=every,.ordinal=(uint64_t)place[observer[tail]]+1,
                                                               .lookback=added_delay[a],.startup=MESH_GRAPH_NONE});
      }
      if(!status)status=plan_signed_put(&o,(struct mesh_graph_repeated){.processor=2*v,
                                        .primitive=MESH_GRAPH_BRIDGE_START,.vertex=u,
                                        .object=MESH_GRAPH_PLAN_SUBMISSION+f,.every=nf,.ordinal=(uint64_t)i+1,
                                        .startup=MESH_GRAPH_NONE});
      /* design/synchronization-signature-change-request.md items 1 and 4: this node's bridge reading the supplied word
         of each release it sources (plan_walk: a demand's root, `in` MESH_ABSENT, sends with field MESH_ABSENT, its
         hops' word plan_supplied, the word mesh_host_publish raises to the execution + 1), at call j's bridge vertex:
         the read's identity its call (the vertex's task label) and the demand's representative transfer `first`, its
         value K every + ordinal the execution + 1 it waits for */
      for(uint32_t q=0;!status && v==s && (p->flow->nodes>>v&1) && q<p->flow->demands;q++)
        if(p->flow->demand[q].source==s)
          status=plan_signed_put(&o,(struct mesh_graph_repeated){.processor=2*v+1,.primitive=MESH_GRAPH_BRIDGE_READ,
                                                                 .vertex=actor[j]+plan_rank(p->flow->nodes,v),
                                                                 .object=MESH_GRAPH_PLAN_SUPPLIED+p->flow->demand[q].first,
                                                                 .every=nf,.ordinal=(uint64_t)i+1,
                                                                 .startup=MESH_GRAPH_NONE});
      for(size_t q=first[(size_t)v*count+f];!status && q<first[(size_t)v*count+f+1];q++){
        const uint32_t a=(uint32_t)tell[q],tail=from[built+a];
        status=plan_signed_put(&o,(struct mesh_graph_repeated){.processor=2*v+1,.primitive=MESH_GRAPH_BRIDGE_RAISE,
                                                               .vertex=tail,.object=MESH_GRAPH_PLAN_COUNT+a,.every=nf,
                                                               .ordinal=(uint64_t)i+1,
                                                               .lookback=lagged[tail]*nf+place[observer[tail]],
                                                               .startup=u});
      }
    }
  }
  if(!status)status=mesh_graph_signature_repeated(2*n,processor,V,task,built,dependence,o.count,o.row,executions,
                                                  &signature);
  if(!status && (status=mesh_graph_resynchronized(&signature,time,built,from,to,delay,added,from+built,to+built,
                                                  added_delay,lambda,offset)))
    mesh_graph_signature_free(&signature);
  if(!status)**kept=signature;
  free(dependence);free(offset);free(key);free(first);free(tell);free(o.row);
  return status;
}
/* design/aot-compilation-change-request.md A10; A. J. Bernstein, 1966; M. C. Rinard and P. C. Diniz, 1997, §2 */
static void plan_ordered(struct plan_room_edges *graph,const struct plan_step *plans,const uint32_t *calls,
                         const uint32_t *actor,uint32_t a,uint32_t b,uint64_t d){
  const struct plan_step *p=plans+calls[a],*q=plans+calls[b];
  for(uint64_t set=p->flow->nodes&q->flow->nodes;set;set&=set-1){
    const uint32_t v=(uint32_t)__builtin_ctzll(set);
    plan_room_edge(graph,actor[a]+plan_rank(p->flow->nodes,v),actor[b]+plan_rank(q->flow->nodes,v),d);
  }
}
/* design/compilation-composition-change-request.md item 6, unit 6D: vertex x is node `self`'s part of its call (a
   bridge-keyed vertex past the senders' pairs) or its submission there (a vertex keyed by its node; the sender's
   2j) */
static int plan_here(const uint32_t *key,const uint32_t *node,uint32_t J,uint32_t n,uint32_t self,uint32_t x){
  return node[x]==self && ((x>=2*J && key[x]==n) || key[x]==self);
}
/* design/aot-compilation-change-request.md A10 and appendix item 2: `calls`, `row` and `order` the instance rows
   (storage, copies, transport), `source_calls`, `source_row`, `source_order` and source_outstanding[] the original
   source family's (the issue bound's successor and delay) */
static int plan_graph(struct plan_step *plans,const struct mesh_plan_analysis *a,struct mesh_planned *planned){
  const struct mesh_function *functions=a->function;
  const uint32_t count=a->functions,*calls=a->calls,norders=a->norders,self=a->nodes->self;
  const uint64_t program=a->program;
  const size_t *row=a->call_group->first,*order=a->call_group->order,*place=a->call_group->ordinal;
  const size_t *source_row=a->source_group->first,*source_order=a->source_group->order;
  const struct mesh_graph_relation_data *relation=a->relation?mesh_graph_relation_data(a->relation):NULL;
  const uint32_t s=(uint32_t)__builtin_ctzll(program),J=a->ncalls,n=plans[0].flow->n;
  struct plan_graph_result *kept=plans[0].graph=calloc(1,sizeof *kept);
  if(!kept)return ENOMEM;
  kept->analysis=a;
  uint64_t vertices=2*(uint64_t)J,edges=0;
  uint32_t *actor=kept->actor=malloc(((size_t)J+1)*sizeof *actor),
           *later=kept->later=malloc(((size_t)J+1)*sizeof *later);
  uint64_t *lag=kept->lag=malloc(((size_t)J+1)*sizeof *lag),*lagged=NULL;
  double *t=NULL;
  uint32_t *from=NULL,*to=NULL,*call=NULL,*node=NULL,*observer=NULL,*key=NULL,*added_from=NULL,*added_to=NULL,
           *entry=NULL;
  struct mesh_graph_task *task=NULL;
  struct mesh_planned_cause *cause=NULL;
  uint64_t *delay=NULL,*added_delay=NULL;
  size_t *first=NULL,*rows=NULL,*wait_first=NULL,*wait_order=NULL,*tell_first=NULL,*tell_order=NULL;
  int status=!actor || !later || !lag?ENOMEM:0;
  if(status)goto done;
  for(uint32_t q=0;!relation && q<norders;q++)
    if(a->causes[q]!=MESH_ORDER_FLOW && a->causes[q]!=MESH_ORDER_CONFLICT){status=EINVAL;goto done;}
  /* design/aot-compilation-change-request.md A10, "Its bound": call j's source-family successor later[j], `w` calls
     after it in its original function's calls, lag[j] blocks later */
  for(uint32_t j=0;j<J;j++){
      const size_t first=source_row[a->source_calls[j]],nf=source_row[a->source_calls[j]+1]-first,
                   i=a->source_group->ordinal[j];
      const uint64_t w=a->source_outstanding[a->source_calls[j]];
      if(!w || w>=UINT32_MAX){status=EINVAL;goto done;}
      later[j]=(uint32_t)source_order[first+(i+w)%nf];lag[j]=(i+w)/nf;
  }
  const uint64_t submitted=(uint64_t)__builtin_popcountll(program&~(UINT64_C(1)<<s));
  for(uint32_t j=0;j<J && vertices<UINT32_MAX;j++){
    const struct plan_step *p=plans+calls[j];
    const uint64_t parts=(uint64_t)__builtin_popcountll(p->flow->nodes),others=submitted;
    actor[j]=(uint32_t)vertices;
    vertices+=parts+others;
    edges+=4+3*parts+others+2*p->flow->forwarded;
  }
  for(uint32_t q=0;q<norders;q++){
    const uint32_t from=relation?relation->edge[a->ordered[q]].from:a->orders[2*q],
                   to=relation?relation->edge[a->ordered[q]].to:a->orders[2*q+1];
    if(from>=to || to>=J){status=EINVAL;goto done;}
    edges+=(uint64_t)__builtin_popcountll(plans[calls[from]].flow->nodes&plans[calls[to]].flow->nodes);
  }
  for(size_t k=0;k<a->chain_pairs->edges;k++){
    const struct mesh_graph_dependence *e=a->chain_pairs->edge+k;
    edges+=(uint64_t)__builtin_popcountll(plans[calls[e->from]].flow->nodes&plans[calls[e->to]].flow->nodes);
  }
  if(vertices>=UINT32_MAX/4 || edges>=SIZE_MAX/(8*sizeof(uint64_t)) || (uint64_t)n*J>=UINT32_MAX){
    status=ENOMEM;goto done;
  }
  const uint32_t V=(uint32_t)vertices;
  kept->vertices=V;
  const size_t room=(size_t)edges+2*(size_t)V;
  t=kept->time=calloc((size_t)V+1,sizeof *t);
  from=kept->from=malloc((room+1)*sizeof *from);
  to=kept->to=malloc((room+1)*sizeof *to);
  delay=kept->delay=malloc((room+1)*sizeof *delay);
  call=kept->call=malloc(((size_t)V+1)*sizeof *call);
  node=kept->node=malloc(((size_t)V+1)*sizeof *node);
  observer=kept->observer=malloc(((size_t)V+1)*sizeof *observer);
  lagged=kept->lagged=malloc(((size_t)V+1)*sizeof *lagged);
  key=kept->key=malloc((2*(size_t)V+1)*sizeof *key);
  added_from=kept->added_from=malloc((2*(size_t)V+1)*sizeof *added_from);
  added_to=kept->added_to=malloc((2*(size_t)V+1)*sizeof *added_to);
  added_delay=kept->added_delay=malloc((2*(size_t)V+1)*sizeof *added_delay);
  entry=kept->entry=malloc((2*(size_t)V+1)*sizeof *entry);
  first=kept->first=malloc(((size_t)V+n+2)*sizeof *first);
  rows=kept->rows=malloc((room+1)*sizeof *rows);
  wait_first=kept->wait_first=malloc(((size_t)n*J+1)*sizeof *wait_first);
  wait_order=kept->wait_order=malloc((2*(size_t)V+1)*sizeof *wait_order);
  tell_first=kept->tell_first=malloc(((size_t)count+2)*sizeof *tell_first);
  tell_order=kept->tell_order=malloc((2*(size_t)V+1)*sizeof *tell_order);
  cause=kept->cause=malloc((room+1)*sizeof *cause);
  if(!t || !from || !to || !delay || !call || !node || !observer || !lagged || !key || !added_from || !added_to ||
     !added_delay ||
     !entry || !first || !rows || !wait_first || !wait_order || !tell_first || !tell_order || !cause){
    status=ENOMEM;goto done;
  }
  /* design/aot-compilation-change-request.md A10 and appendix item 2: each vertex's observer, the call whose
     submissions count it, and the blocks later it does: a work vertex's its call's source-family successor, a
     submission's itself */
  for(uint32_t j=0;j<J;j++){
    const struct plan_step *p=plans+calls[j];
    call[2*j]=call[2*j+1]=j;
    node[2*j]=node[2*j+1]=s;
    observer[2*j]=observer[2*j+1]=j;lagged[2*j]=lagged[2*j+1]=0;
    key[2*j]=s;key[2*j+1]=n;
    uint32_t k=actor[j];
    for(uint64_t set=p->flow->nodes;set;set&=set-1,k++){
      call[k]=j;
      node[k]=(uint32_t)__builtin_ctzll(set);
      t[k]=p->flow->time[node[k]];
      observer[k]=later[j];lagged[k]=lag[j];
      key[k]=n;
    }
    for(uint64_t set=program&~(UINT64_C(1)<<s);set;set&=set-1,k++){
      call[k]=j;
      node[k]=(uint32_t)__builtin_ctzll(set);
      observer[k]=j;lagged[k]=0;
      key[k]=node[k];
    }
  }
  struct plan_room_edges graph={from,to,delay,0,cause,{MESH_PLANNED_PROGRAM,UINT32_MAX}};
  mesh_graph_rows(n+1,V,plan_at,key,first,rows);
  if(!(task=kept->task=malloc(((size_t)V+1)*sizeof *task))){status=ENOMEM;goto done;}
  for(uint32_t v=0;v<=n;v++)for(size_t q=first[v];q<first[v+1];q++){
    const uint32_t x=(uint32_t)rows[q];
    task[x]=(struct mesh_graph_task){.processor=v<n?2*v:2*node[x]+1,.place=v<n?(uint32_t)(q-first[v]):call[x],
                                     .label=call[x],.copy=0};
  }
  for(uint32_t v=0;v<n;v++)for(size_t q=first[v];q<first[v+1];q++){
    const size_t next=q+1<first[v+1]?q+1:first[v];
    plan_room_edge(&graph,(uint32_t)rows[q],(uint32_t)rows[next],next==first[v]);
  }
  for(uint32_t j=0;j<J;j++){
    const uint32_t f=calls[j],i=place[j],next=j+1==J?0:j+1;
    const struct plan_step *p=plans+f;
    const uint64_t nf=p->flow->calls,span=(i+(uint64_t)p->flow->copies)/nf;
    const uint32_t ahead=(uint32_t)order[row[f]+(i+(uint64_t)p->flow->copies)%nf];
    plan_room_cause(&graph,MESH_PLANNED_PROGRAM,UINT32_MAX);
    plan_room_edge(&graph,2*j+1,2*next+1,!next);
    plan_room_cause(&graph,MESH_PLANNED_SUBMIT,UINT32_MAX);
    plan_room_edge(&graph,2*j,2*j+1,0);
    for(uint64_t set=p->flow->nodes;set;set&=set-1){
      const uint32_t v=(uint32_t)__builtin_ctzll(set),x=actor[j]+plan_rank(p->flow->nodes,v);
      plan_room_cause(&graph,MESH_PLANNED_SUBMIT,UINT32_MAX);
      plan_room_edge(&graph,2*j+1,x,0);
      plan_room_edge(&graph,plan_submission(plans,calls,actor,program,s,j,v),x,0);
      plan_room_cause(&graph,MESH_PLANNED_OUTSTANDING,UINT32_MAX);
      plan_room_edge(&graph,x,plan_submission(plans,calls,actor,program,s,later[j],v),lag[j]);
    }
    plan_room_cause(&graph,MESH_PLANNED_OUTSTANDING,UINT32_MAX);
    if(!(p->flow->nodes>>s&1))plan_room_edge(&graph,2*j,2*later[j],lag[j]);
    for(uint32_t d=0;d<p->flow->demands;d++){
      plan_room_cause(&graph,MESH_PLANNED_TRANSFER,d);
      for(uint32_t y=0;y<p->flow->n;y++)for(uint64_t below=p->flow->child[(size_t)d*p->flow->n+y];below;below&=below-1){
        const uint32_t c=(uint32_t)__builtin_ctzll(below);
        plan_room_edge(&graph,actor[j]+plan_rank(p->flow->nodes,y),actor[ahead]+plan_rank(p->flow->nodes,c),span);
        plan_room_edge(&graph,actor[j]+plan_rank(p->flow->nodes,c),actor[ahead]+plan_rank(p->flow->nodes,y),span);
      }
    }
  }
  /* design/aot-compilation-change-request.md A10: Bernstein's order between calls, each at every node both calls'
     parts share (delay 0, one block); each commuting chain's calls one at a time in program order, its last call's
     part before the next block's first (delay 1), as BSL §4.1 closes a processor's order */
  for(uint32_t q=0;q<norders;q++){
    uint32_t kind=relation?MESH_GRAPH_GIVEN:0;
    if(relation && (status=mesh_graph_relation_edge_cause(a->relation,a->ordered[q],&kind)))goto done;
    plan_room_cause(&graph,relation?(kind==MESH_GRAPH_GIVEN?MESH_ORDER_FLOW:MESH_ORDER_CONFLICT):a->causes[q],q);
    plan_ordered(&graph,plans,calls,actor,relation?relation->edge[a->ordered[q]].from:a->orders[2*q],
                  relation?relation->edge[a->ordered[q]].to:a->orders[2*q+1],0);
  }
  plan_room_cause(&graph,MESH_PLANNED_CHAIN,UINT32_MAX);
  for(size_t k=0;k<a->chain_pairs->edges;k++){
    const struct mesh_graph_dependence *e=a->chain_pairs->edge+k;
    plan_ordered(&graph,plans,calls,actor,e->from,e->to,e->delay);
  }
  const size_t built=graph.edges;
  kept->built=built;
  /* design/compilation-composition-change-request.md item 6, unit 6D, and item 6.2 */
  size_t within=0;
  for(uint32_t j=0;j<J;j++){
    const struct mesh_plan_actors *a=planned->result[calls[j]].actors;
    within+=a?a->edges:functions[calls[j]].edges;
  }
  planned->call=malloc(((size_t)J+1)*sizeof *planned->call);
  planned->order=malloc((built+within+1)*sizeof *planned->order);
  planned->cause=malloc((built+within+1)*sizeof *planned->cause);
  if(!planned->call || !planned->order || !planned->cause){status=ENOMEM;goto done;}
  planned->calls=J;
  for(uint32_t j=0;j<J;j++){
    const struct plan_step *p=plans+calls[j];
    const struct mesh_plan_actors *a=p->result->actors;
    planned->call[j]=(struct mesh_planned_call){.lead=p->flow->start_lead,.function=calls[j],.ordinal=place[j],
                                                .every=p->flow->calls,.parts=a?a->parts:functions[calls[j]].parts};
  }
  /* the orders at this node in their cause's domain (mesh-collective.h struct mesh_planned): FLOW, CONFLICT and
     CHAIN between call ordinals, the bridge's other kinds between the signature's tasks */
  for(size_t e=0;e<built;e++)
    if(plan_here(key,node,J,n,self,from[e]) && plan_here(key,node,J,n,self,to[e])){
      const uint32_t k=cause[e].kind;
      const int calls_of=k==MESH_ORDER_FLOW || k==MESH_ORDER_CONFLICT || k==MESH_PLANNED_CHAIN;
      planned->cause[planned->orders]=cause[e];
      planned->order[planned->orders++]=(struct mesh_graph_dependence){calls_of?call[from[e]]:from[e],
                                                                       calls_of?call[to[e]]:to[e],delay[e]};
    }
  for(uint32_t j=0;j<J;j++){
    const struct mesh_function *f=functions+calls[j];
    const struct mesh_plan_actors *a=planned->result[calls[j]].actors;
    for(uint32_t e=0;e<(a?a->edges:f->edges);e++){
      planned->cause[planned->orders]=(struct mesh_planned_cause){MESH_PLANNED_PART,j};
      planned->order[planned->orders++]=(a?a->edge:f->edge)[e];
    }
  }
  double lambda;
  uint32_t added;
  /* design/compilation-composition-change-request.md item 6, unit 6A: the program graph's cycle mean, BSL Fig. 8's
     lambda, is DetermineDelays's operand */
  if((status=mesh_graph_cycle_mean(V,t,built,from,to,delay,&lambda)))goto done;
  if(lambda==INFINITY){status=ENOMEM;goto done;}
  if((status=mesh_graph_convert_sc(V,t,built,from,to,&added,added_from,added_to)) ||
     (status=mesh_graph_determine_delays(V,t,built,from,to,delay,added,added_from,added_to,lambda,added_delay))){
    if(status==EDEADLK || status==ERANGE)status=ENOMEM;
    goto done;
  }
  kept->lambda=lambda;kept->added=added;
  for(uint32_t a=0;a<added;a++){
    const uint32_t tail=added_from[a],head=added_to[a],h=call[head];
    if(__builtin_add_overflow(added_delay[a],lagged[tail],&delay[built+a])){
      status=ENOMEM;goto done;
    }
    from[built+a]=tail;to[built+a]=head;
    key[a]=node[head]*J+(uint32_t)row[calls[h]]+place[h];
  }
  mesh_graph_rows(n*J,added,plan_at,key,wait_first,wait_order);
  for(uint32_t q=0;q<added;q++)entry[wait_order[q]]=q;
  for(uint32_t a=0;a<added;a++)key[a]=node[added_from[a]]==self?calls[observer[added_from[a]]]:count;
  mesh_graph_rows(count+1,added,plan_at,key,tell_first,tell_order);
  for(uint32_t f=0;f<count;f++){
    struct plan_step *p=plans+f;
    const size_t *waits=wait_first+(size_t)self*J+row[f];
    p->waits=(uint32_t)(waits[p->flow->calls]-waits[0]);
    p->tells=(uint32_t)(tell_first[f+1]-tell_first[f]);
    p->sync=malloc(((size_t)p->waits+p->tells+1)*sizeof *p->sync);
    p->wait_row=malloc(((size_t)p->flow->calls+1)*sizeof *p->wait_row);
    if(!p->sync || !p->wait_row){status=ENOMEM;goto done;}
    for(uint32_t r=0;r<=p->flow->calls;r++)p->wait_row[r]=(uint32_t)(waits[r]-waits[0]);
    for(size_t q=row[f];q<row[f+1];q++)planned->call[order[q]].waits=(uint32_t)(waits[q-row[f]+1]-waits[q-row[f]]);
    for(uint32_t k=0;k<p->waits;k++){
      const uint32_t a=(uint32_t)wait_order[waits[0]+k],tail=added_from[a],j=observer[tail];
      const uint32_t every=plans[calls[j]].flow->calls;
      int64_t shifted;
      if(added_delay[a]>INT64_MAX || __builtin_mul_overflow((int64_t)added_delay[a],(int64_t)every,&shifted)){
        status=ENOMEM;goto done;
      }
      p->sync[k]=(struct plan_sync){.node=node[tail],.function=calls[j]+1,.every=every,
                                    .base=(int64_t)place[j]+1-shifted};
    }
    for(uint32_t k=0;k<p->tells;k++){
      const uint32_t a=(uint32_t)tell_order[tell_first[f]+k],head=added_to[a],g=calls[call[head]];
      p->sync[p->waits+k]=(struct plan_sync){.node=node[head],.function=g+1,
                                             .entry=(uint32_t)(entry[a]-wait_first[(size_t)node[head]*J+row[g]])};
    }
  }
  if((status=plan_signature(plans,count,calls,J,program,s,n,V,t,task,node,observer,lagged,actor,place,row,built,added,
                             from,to,delay,added_delay,lambda,wait_first,wait_order,&planned->signature)))goto done;
  /* S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, (5) */
  if((status=mesh_graph_group(V,built+added,plan_at,from,&kept->room_rows)) ||
     (status=mesh_graph_lag_prepare(V,kept->room_rows->first,plan_room_head,plan_room_lag,kept,&kept->room_graph)))goto done;
  kept->room_query=malloc((2*(size_t)J+1)*sizeof *kept->room_query);
  if(!kept->room_query){status=ENOMEM;goto done;}
  size_t queries=0;
  for(uint32_t j=0;j<J;j++){
    kept->room_query[queries++]=(struct mesh_graph_dependence){2*j+1,2*j,0};
    if(plans[calls[j]].flow->nodes>>self&1)
      kept->room_query[queries++]=(struct mesh_graph_dependence){actor[j]+plan_rank(plans[calls[j]].flow->nodes,self),2*j+1,0};
  }
  if((status=mesh_graph_lag_queries(kept->room_graph,queries,kept->room_query,&kept->room_lags)))goto done;
  for(uint32_t f=0;f<count;f++){plans[f].room=plans[f].flow->nodes>>self&1?0:1;plans[f].submit_room=0;}
  size_t q=0;
  for(uint32_t j=0;j<J;j++){
    struct plan_step *p=plans+calls[j];
    const int here=(int)(p->flow->nodes>>self&1);
    const uint64_t submit=kept->room_lags->value[q++],receive=here?kept->room_lags->value[q++]:0;
    if(submit==UINT64_MAX || !submit || (here && (receive==UINT64_MAX || !receive))){status=ENOMEM;goto done;}
    if(__builtin_add_overflow(p->submit_room,submit,&p->submit_room) ||
       (here && __builtin_add_overflow(p->room,receive,&p->room))){status=ENOMEM;goto done;}
  }
  for(uint32_t f=0;f<count;f++)
    if(plans[f].room>=UINT64_MAX/(4*sizeof(struct mesh_delivery)) ||
       plans[f].submit_room>=UINT64_MAX/(4*sizeof(struct mesh_submission))){status=ENOMEM;goto done;}
done:
  return status;
}

/* design/plan-to-bridge-change-request.md R1 */

static int plan_view(const struct mesh_strided *v,uint64_t *bytes,uint64_t *end){
  uint64_t b=v->element,e;
  if(!b || __builtin_add_overflow(v->offset,v->element,&e))return EINVAL;
  for(uint64_t d=0;d<v->dims;d++){
    const uint64_t size=v->pairs[2*d],stride=v->pairs[2*d+1];
    uint64_t step;
    if(!size || __builtin_mul_overflow(b,size,&b) || __builtin_mul_overflow(size-1,stride,&step) ||
       __builtin_add_overflow(e,step,&e))return EINVAL;
  }
  *bytes=b;*end=e;
  return 0;
}
static int plan_alike(const struct mesh_strided *a,const struct mesh_strided *b){
  if(a->offset!=b->offset || a->element!=b->element || a->dims!=b->dims)return 0;
  for(uint64_t k=0;k<2*(uint64_t)a->dims;k++)if(a->pairs[k]!=b->pairs[k])return 0;
  return 1;
}
/* R. Ross, N. Miller and W. D. Gropp, EuroPVM/MPI 2003; design/plan-to-bridge-change-request.md R1 */
void mesh_strided_extents(const struct mesh_strided *v,void (*visit)(void *,uint64_t,uint64_t),void *context){
  const uint64_t *pair=v->pairs;
  int64_t d=(int64_t)v->dims-1;
  uint64_t run=v->element;
  for(;d>=0 && (pair[2*d]==1 || pair[2*d+1]==run);d--)run*=pair[2*d];
  uint64_t index[(size_t)v->dims+1],at=v->offset,start=at,length=0;
  memset(index,0,sizeof index);
  for(;;){
    if(length && start+length==at)length+=run;
    else{
      if(length)visit(context,start,length);
      start=at;length=run;
    }
    int64_t e=d;
    for(;e>=0 && ++index[e]==pair[2*e];e--){at-=(pair[2*e]-1)*pair[2*e+1];index[e]=0;}
    if(e<0)break;
    at+=pair[2*e+1];
  }
  visit(context,start,length);
}
struct plan_cutting { uint64_t quantum,range,made,*cuts,*start,*count,*index; };
static void plan_cut(void *context,uint64_t offset,uint64_t length){
  struct plan_cutting *x=context;
  if(!x->cuts)x->made+=(offset+length-1)/x->quantum-offset/x->quantum+1;
  else{
    const uint32_t pieces=mesh_graph_grid_cut(offset,length,x->quantum,x->start,x->count,x->index);
    for(uint32_t k=0;k<pieces;k++)x->cuts[x->made++]=x->range+x->start[k]-offset;
  }
  x->range+=length;
}
struct plan_starting { const uint64_t *cut; uint64_t *at,range,next,instructions; };
static void plan_start(void *context,uint64_t offset,uint64_t length){
  struct plan_starting *x=context;
  for(;x->next<x->instructions && x->cut[x->next]<x->range+length;x->next++)
    x->at[x->next]=offset+x->cut[x->next]-x->range;
  x->range+=length;
}
static int plan_cut_order(const void *a,const void *b){
  const uint64_t x=*(const uint64_t *)a,y=*(const uint64_t *)b;
  return (x>y)-(x<y);
}
/* design/aot-compilation-change-request.md A7; F. B. Schneider, ACM Comput. Surv. 22(4), 1990 */
static int plan_instructions(struct mesh_plan_flow *p,const struct mesh_plan_geometry *m){
  const uint64_t quantum=plan_quantum(m);
  const uint32_t count=p->transfers;
  size_t *first=malloc(((size_t)p->demands+1)*sizeof *first),*order=malloc(((size_t)count+1)*sizeof *order);
  uint64_t *start=NULL,*pieces=NULL,*index=NULL,room=0,most=0,ends=0,at=0;
  int status=!first || !order?ENOMEM:0;
  if(!status){
    mesh_graph_rows(p->demands,count,plan_at,p->of,first,order);
    for(uint32_t i=0;i<p->demands;i++){
      const struct mesh_plan_demand *d=p->demand+i;
      const uint64_t blocks=plan_blocks(m,d->bytes);
      struct plan_cutting cutting={.quantum=quantum};
      mesh_strided_extents(&p->declared[d->first].from,plan_cut,&cutting);
      for(size_t q=first[i];q<first[i+1];q++)mesh_strided_extents(&p->declared[order[q]].to,plan_cut,&cutting);
      room+=cutting.made+blocks+1;
      if(blocks>most)most=blocks;
    }
    p->cut=malloc((room+1)*sizeof *p->cut);
    start=malloc((most+2)*sizeof *start);
    pieces=malloc((most+2)*sizeof *pieces);
    index=malloc((most+2)*sizeof *index);
    if(!p->cut || !start || !pieces || !index)status=ENOMEM;
  }
  for(uint32_t i=0;!status && i<p->demands;i++){
    struct mesh_plan_demand *d=p->demand+i;
    struct plan_cutting cutting={quantum,0,0,p->cut+at,start,pieces,index};
    mesh_strided_extents(&p->declared[d->first].from,plan_cut,&cutting);
    for(size_t q=first[i];q<first[i+1];q++){cutting.range=0;mesh_strided_extents(&p->declared[order[q]].to,plan_cut,&cutting);}
    const uint32_t own=mesh_graph_grid_cut(0,d->bytes,quantum,start,pieces,index);
    for(uint32_t k=0;k<own;k++)cutting.cuts[cutting.made++]=start[k];
    cutting.cuts[cutting.made++]=d->bytes;
    qsort(cutting.cuts,cutting.made,sizeof *cutting.cuts,plan_cut_order);
    uint64_t made=1;
    for(uint64_t k=1;k<cutting.made;k++)if(cutting.cuts[k]!=cutting.cuts[made-1])cutting.cuts[made++]=cutting.cuts[k];
    d->cut=at;d->instructions=made-1;at+=made;
    if(d->instructions>=UINT32_MAX)status=ENOMEM;
    if(d->mine!=MESH_ABSENT || d->root!=MESH_ABSENT)ends+=d->instructions;
  }
  p->positions=ends;
  if(!status && !(p->at=malloc((ends+1)*sizeof *p->at)))status=ENOMEM;
  at=0;
  for(uint32_t i=0;!status && i<p->demands;i++){
    struct mesh_plan_demand *d=p->demand+i;
    if(d->mine==MESH_ABSENT && d->root==MESH_ABSENT)continue;
    struct plan_starting starting={p->cut+d->cut,p->at+at,0,0,d->instructions};
    mesh_strided_extents(d->view,plan_start,&starting);
    d->at=at;at+=d->instructions;
  }
  free(first);free(order);free(start);free(pieces);free(index);
  return status;
}
/* MPI 4.1 §6.1.11 */
static int plan_landings(const struct mesh_plan_flow *p){
  uint64_t count=0,k=0;
  for(uint32_t i=0;i<p->demands;i++)
    if((p->demand[i].mine!=MESH_ABSENT || p->demand[i].root!=MESH_ABSENT) && p->demand[i].leaves>>p->self&1)
      count+=p->demand[i].instructions;
  struct mesh_graph_interval *interval=malloc((count+1)*sizeof *interval);
  if(!interval)return ENOMEM;
  for(uint32_t i=0;i<p->demands;i++){
    const struct mesh_plan_demand *d=p->demand+i;
    if((d->mine==MESH_ABSENT && d->root==MESH_ABSENT) || !(d->leaves>>p->self&1))continue;
    for(uint64_t j=0;j<d->instructions;j++){
      const uint64_t at=p->at[d->at+j];
      interval[k++]=(struct mesh_graph_interval){d->mine!=MESH_ABSENT?d->mine:p->givens+d->root,at,
                                                 at+p->cut[d->cut+j+1]-p->cut[d->cut+j]};
    }
  }
  const int status=mesh_graph_intersect(count,interval)?EINVAL:0;
  free(interval);
  return status;
}

static uint64_t plan_cover_bytes(const void *context,uint32_t value){
  return ((const struct mesh_plan_flow *)context)->given[value].section.bytes;
}
static uint32_t plan_relay_member(const void *context,uint32_t k){
  const struct mesh_plan_flow *p=context;return p->demand[p->relay[k]].value;
}
static uint64_t plan_relay_blocks(const void *context,uint32_t k){
  const struct mesh_plan_flow *p=context;return plan_blocks(p->geometry,p->demand[p->relay[k]].bytes);
}
static int plan_sets(struct mesh_plan_flow *p,const struct mesh_plan_geometry *m){
  const uint64_t quantum=plan_quantum(m);
  uint64_t blocks=0,touches=0,relays=0;
  if(!(p->block_first=malloc(((size_t)p->givens+1)*sizeof *p->block_first)))return ENOMEM;
  for(uint32_t j=0;j<p->givens;j++){
    p->block_first[j]=MESH_ABSENT;
    if(!p->named[j] || p->given[j].run)continue;
    p->block_first[j]=(uint32_t)blocks;
    if((blocks+=plan_blocks(m,p->given[j].section.bytes))>=UINT32_MAX)return ENOMEM;
  }
  for(uint32_t i=0;i<p->demands;i++){
    const struct mesh_plan_demand *d=p->demand+i;
    if(d->value==MESH_ABSENT)continue;
    if(d->mine==MESH_ABSENT && d->root==MESH_ABSENT)relays++;
    else if(d->mine!=MESH_ABSENT && !p->given[d->mine].run)touches+=d->instructions;
  }
  p->blocks=blocks;
  p->touch=malloc((touches+1)*sizeof *p->touch);
  p->relay=malloc((relays+1)*sizeof *p->relay);
  if(!p->touch || !p->relay)return ENOMEM;
  uint64_t t=0,r=0;
  for(uint32_t i=0;i<p->demands;i++){
    struct mesh_plan_demand *d=p->demand+i;
    d->relay=MESH_ABSENT;
    if(d->value!=MESH_ABSENT && d->mine==MESH_ABSENT && d->root==MESH_ABSENT)p->relay[r++]=i;
    if(d->mine==MESH_ABSENT || p->given[d->mine].run)continue;
    for(uint64_t k=0;k<d->instructions;k++){
      const uint64_t at=p->at[d->at+k];
      p->touch[t++]=(struct mesh_graph_touch){at,at+p->cut[d->cut+k+1]-p->cut[d->cut+k],d->mine,d->value};
    }
  }
  int status=mesh_graph_cover_sets(quantum,p->givens,(uint32_t)blocks,p->block_first,plan_cover_bytes,p,
                                    touches,p->touch,(uint32_t)relays,plan_relay_member,plan_relay_blocks,p,&p->covers);
  if(status)return status;
  p->sets=p->covers->sets;p->set_first=p->covers->first;p->member=p->covers->member;
  p->set_blocks=p->covers->set_blocks;p->block_set=p->covers->block_set;p->block_element=p->covers->block_element;
  p->uncovered=p->covers->uncovered;
  p->set_far=calloc((size_t)p->sets+1,sizeof *p->set_far);
  if(!p->set_far)return ENOMEM;
  for(uint32_t k=0;k<relays;k++)p->demand[p->relay[k]].relay=p->covers->extra_set[k];
  for(uint32_t x=0;x<p->sets;x++)
    for(uint32_t k=p->set_first[x];k<p->set_first[x+1];k++)
      if(p->farthest[p->member[k]]>p->set_far[x])p->set_far[x]=p->farthest[p->member[k]];
  return 0;
}

/* design/aot-compilation-change-request.md A4 (AllGather, CollectivePermute and AllToAll) and A9;
   design/plan-to-bridge-change-request.md R2, R4 and R5; mesh-collective.h mesh_declared_group: the one grouping of a
   function's declared rows into demands and their releases on node `self`, which plan_function consumes */
int mesh_declared_group(const struct mesh_declared *declared,const struct mesh_weight *weights,uint32_t count,
                        uint32_t nodes,uint32_t self,struct mesh_grouped *grouped){
  if(!grouped || nodes>MESH_NODES_MOST || self>=nodes ||
     (count && (!declared || !weights || !grouped->demand || !grouped->demand_of || !grouped->release_of)))
    return EINVAL;
  /* design/plan-to-bridge-change-request.md R4; design/aot-compilation-change-request.md A3 */
  for(uint32_t i=0;i<count;i++){
    const struct mesh_weight *x=weights+i;
    if(declared[i].kind>MESH_PAGES || !(x->produced>0 && x->produced<=DBL_MAX) ||
       !(x->consumed>0 && x->consumed<=DBL_MAX) || !(x->read>0 && x->read<=DBL_MAX) ||
       !(x->page>0 && x->page<=DBL_MAX))return EINVAL;
  }
  struct plan_key *key=malloc(((size_t)count+1)*sizeof *key);
  uint32_t *placed=malloc(((size_t)count+1)*sizeof *placed);
  int status=key && placed?0:ENOMEM;
  for(uint32_t i=0;!status && i<count;i++){
    const struct mesh_declared *t=declared+i;
    uint64_t from,to,end;
    if(t->source>=nodes || t->destination>=nodes)status=EHOSTUNREACH;
    else if(t->source==t->destination || plan_view(&t->from,&from,&end) || plan_view(&t->to,&to,&end) || from!=to)
      status=EINVAL;
    else key[i]=(struct plan_key){t->source,t->identity,i};
  }
  if(status){free(key);free(placed);return status;}
  qsort(key,count,sizeof *key,plan_key_order);
  for(uint32_t k=0;k<count;k++){
    const uint32_t i=key[k].index;
    placed[i]=i;
    for(uint32_t b=k;placed[i]==i && b-- && key[b].source==key[k].source && key[b].identity==key[k].identity;){
      const uint32_t j=key[b].index;
      uint32_t c=k;
      if(placed[j]!=j || !plan_alike(&declared[i].from,&declared[j].from) ||
         weights[i].produced!=weights[j].produced || weights[i].read!=weights[j].read)continue;
      while(c-->b && (placed[key[c].index]!=j || declared[key[c].index].destination!=declared[i].destination));
      if(c+1==b)placed[i]=j;
    }
  }
  struct mesh_demand *demand=grouped->demand;
  uint32_t *of=grouped->demand_of,demands=0,releases=0;
  for(uint32_t i=0;i<count;i++){
    const struct mesh_declared *t=declared+i;
    if(placed[i]==i){
      uint64_t bytes,end;
      plan_view(&t->from,&bytes,&end);
      demand[demands++]=(struct mesh_demand){.source=t->source,.identity=t->identity,.first=i,.release=MESH_ABSENT,
                                             .bytes=bytes,.pages=1};
    }
    of[i]=placed[i]==i?demands-1:of[placed[i]];
    struct mesh_demand *d=demand+of[i];
    /* design/aot-compilation-change-request.md, appendix item 5; design/plan-to-bridge-change-request.md R5 */
    if(t->kind!=MESH_PAGES)d->pages=0;
    d->leaves|=UINT64_C(1)<<t->destination;
  }
  for(uint32_t d=0;d<demands;d++)if(demand[d].source==self)demand[d].release=releases++;
  for(uint32_t i=0;i<count;i++)grouped->release_of[i]=declared[i].source==self?demand[of[i]].release:MESH_ABSENT;
  grouped->transfers=count;grouped->demands=demands;grouped->releases=releases;grouped->nodes=nodes;
  grouped->self=self;
  free(key);free(placed);
  return 0;
}

/* N. Garg and J. Koenemann, FOCS 1998; `grouped` its declared rows' grouping on this node (mesh_declared_group) */
static int plan_function(struct mesh_plan_flow *p,const struct mesh_function *function,const struct mesh_nodes *nodes,
                         uint64_t program,const struct mesh_grouped *grouped){
  const struct mesh_plan_geometry *m=p->geometry;
  const struct mesh_declared *declared=function->declared;
  const struct mesh_weight *weights=function->weights;
  const struct mesh_dependence *dependence=function->dependence;
  const struct mesh_value *values=function->value;
  const uint32_t count=function->count,given=function->values,dependences=function->dependences;
  if((count && (!declared || !weights)) || (given && !values) || (dependences && !dependence))return EINVAL;
  /* design/disjoint-pages-change-request.md D.3.3: a value of no bytes an absent, alias or constant slot, its ValueID
     kept, with no storage; a zeroed value (`state`) is never a wire: zeroed INVARIANT state, or zeroed EXECUTION
     metadata each execution copy of which is zeroed where solved (metal-microbench metal_transport.mm Arrivals) */
  for(uint32_t j=0;j<given;j++){
    const struct mesh_value *v=values+j;
    if(v->kind>MESH_EXECUTION || v->wire>1 || v->state>1 || v->run>1 ||
       (v->state && v->wire) ||
       (p->value_lag[j] && (v->kind!=MESH_EXECUTION || v->state)))return EINVAL;
    if(plan_blocks(m,v->section.bytes)>=UINT32_MAX/m->block)return ENOMEM;
  }
  const uint32_t kinds=MESH_FROM_PRODUCED|MESH_TO_PRODUCED|MESH_FROM_SOURCE_READ|MESH_TO_SOURCE_READ;
  for(uint32_t e=0;e<dependences;e++){
    const struct mesh_dependence *x=dependence+e;
    const int from_produced=!!(x->kinds&MESH_FROM_PRODUCED),from_own=!!(x->kinds&MESH_FROM_SOURCE_READ),
              to_produced=!!(x->kinds&MESH_TO_PRODUCED),to_own=!!(x->kinds&MESH_TO_SOURCE_READ);
    if(x->from>=count || x->to>=count || x->kinds&~kinds || (from_produced && from_own) || (to_produced && to_own) ||
       x->lag==UINT32_MAX || plan_event_node(declared,x->from,from_produced,from_own)!=
       plan_event_node(declared,x->to,to_produced,to_own))return EINVAL;
  }
  if(!function->outstanding || function->outstanding>=UINT32_MAX)return ENOMEM;
  const uint32_t n=nodes->count;
  /* mesh-collective.h mesh_declared_group: the grouping of these rows on this node, made once */
  if(grouped->transfers!=count)return EINVAL;
  if(grouped->nodes!=n || grouped->self!=p->self)return ESTALE;
  for(uint32_t i=0;i<count;i++){
    uint64_t bytes,end;
    plan_view(&declared[i].from,&bytes,&end);
    if(plan_blocks(m,bytes)>=UINT32_MAX/m->block)return ENOMEM;
  }
  p->declared=declared;p->transfers=count;p->given=values;p->givens=given;p->weights=weights;
  p->dependence=dependence;p->dependences=dependences;p->outstanding=function->outstanding;
  uint32_t *root=NULL,*demand=NULL;
  size_t *trees=NULL;
  double *up=NULL,*down=NULL,*share=NULL,*rate=NULL,*latency=NULL;
  int status=0;
  p->named=calloc((size_t)given+1,sizeof *p->named);
  p->demand=calloc((size_t)count+1,sizeof *p->demand);
  p->of=grouped->demand_of;
  if(!p->named || !p->demand){status=ENOMEM;goto done;}
  for(uint32_t i=0;i<grouped->demands;i++){
    const struct mesh_demand *x=grouped->demand+i;
    p->demand[i]=(struct mesh_plan_demand){.source=x->source,.identity=x->identity,.first=x->first,.mine=MESH_ABSENT,
                                      .root=MESH_ABSENT,.release=x->release,.leaves=x->leaves,.bytes=x->bytes,
                                      .set=x->pages,.range=MESH_ABSENT,.relay=MESH_ABSENT};
  }
  p->demands=grouped->demands;p->releases=grouped->releases;
  for(uint32_t i=0;i<count;i++){
    const struct mesh_declared *t=declared+i;
    struct mesh_plan_demand *d=p->demand+p->of[i];
    uint64_t bytes,end;
    /* mesh-collective.h struct mesh_declared root; physical binding: plan_sections */
    if(t->root>p->nroots){status=EINVAL;goto done;}
    /* design/plan-to-bridge-change-request.md R2; torch 2.14.0 torch/export/exported_program.py, ExportedProgram;
       Xu et al., GSPMD, arXiv 2105.04663, §4.5: a root row's value is its owner's ValueID on this node, any other
       row's names a site here */
    if(t->source!=p->self && t->destination!=p->self){
      if(!t->root && t->value!=MESH_ABSENT){status=EINVAL;goto done;}
      continue;
    }
    /* the row's root: its own, or the one a use binds its value to (plan_run `bound`: that slot's storage) */
    uint32_t r=t->root?t->root-1:MESH_ABSENT;
    if(!t->root && p->bound && t->value<given && p->bound[t->value]!=MESH_ABSENT){
      r=p->bound[t->value];
      if(r==PLAN_SEVERAL){status=EINVAL;goto done;}
    }
    const struct mesh_strided *view=t->source==p->self?&t->from:&t->to;
    plan_view(view,&bytes,&end);
    if(r!=MESH_ABSENT){
      if(end>p->roots[r].section.bytes || d->mine!=MESH_ABSENT || (d->root!=MESH_ABSENT && d->root!=r)){
        status=EINVAL;goto done;
      }
      d->root=r;d->view=view;p->carried[r]=1;
      continue;
    }
    if(t->value>=given || d->root!=MESH_ABSENT){status=EINVAL;goto done;}
    const struct mesh_value *v=values+t->value;
    if(end>v->section.bytes || v->state || v->kind!=MESH_EXECUTION || (t->destination==p->self && p->value_lag[t->value]) ||
       (t->source==p->self && d->mine!=MESH_ABSENT && d->mine!=t->value)){
      status=EINVAL;goto done;
    }
    d->mine=t->value;
    d->view=view;
  }
  for(uint32_t i=0;i<p->demands;i++)if(p->demand[i].mine!=MESH_ABSENT)p->named[p->demand[i].mine]=1;
  /* ExecuTorch exir/memory_planning.py, update_tensor_lifetime */
  for(uint32_t j=0;j<given;j++)
    if(values[j].kind==MESH_EXECUTION && !p->value_lag[j] && values[j].begin>values[j].end){
      status=EINVAL;goto done;
    }
  if((status=plan_instructions(p,m)) || (status=plan_landings(p)))goto done;

  if(p->demands){
    for(uint32_t a=0;a<n;a++)for(uint32_t b=a+1;b<n;b++){
      const uint64_t c=nodes->capacity[a][b]<nodes->capacity[b][a]?nodes->capacity[a][b]:nodes->capacity[b][a];
      if(c){p->from[p->pairs]=a;p->to[p->pairs]=b;p->capacity[p->pairs]=(double)c;p->pairs++;}
    }
    const uint32_t E=p->pairs;
    uint32_t order[n];
    unsigned char reached[n];
    uint64_t reach[n],known=0;
    for(uint32_t i=0;i<p->demands;i++){
      const uint32_t s=p->demand[i].source;
      if(!(known>>s&1)){
        mesh_graph_reach(n,E,plan_end,p,s,order,reached,NULL);
        reach[s]=0;
        for(uint32_t v=0;v<n;v++)if(reached[v])reach[s]|=UINT64_C(1)<<v;
        known|=UINT64_C(1)<<s;
      }
      if(p->demand[i].leaves&~reach[s]){status=EHOSTUNREACH;goto done;}
    }
    const uint32_t rounds=(uint32_t)(mesh_graph_log(2*E>1?(double)(2*E):2.0)/(MESH_PLAN_EPS*MESH_PLAN_EPS))+1;
    const uint64_t most=(uint64_t)rounds*p->demands;
    const uint32_t room=most>=UINT32_MAX?UINT32_MAX-1:(uint32_t)most;
    root=malloc((size_t)p->demands*sizeof *root);
    up=calloc(p->demands,sizeof *up);
    down=malloc((size_t)p->demands*sizeof *down);
    demand=malloc((size_t)room*sizeof *demand);
    p->parent=malloc((size_t)room*n*sizeof *p->parent);
    share=malloc((size_t)room*sizeof *share);
    if(!root || !up || !down || !demand || !p->parent || !share){status=ENOMEM;goto done;}
    for(uint32_t i=0;i<p->demands;i++){root[i]=p->demand[i].source;down[i]=(double)p->demand[i].bytes;}
    const uint32_t made=mesh_graph_flow(n,E,p->from,p->to,p->capacity,p->demands,root,up,down,plan_leaf,p->demand,
                                        MESH_PLAN_EPS,rounds,room,demand,p->parent,share);
    if(made==UINT32_MAX){status=ENOMEM;goto done;}
    p->made=made;
    p->tree=malloc(((size_t)made+1)*sizeof *p->tree);
    p->run=calloc((size_t)made+1,sizeof *p->run);
    rate=malloc(((size_t)made+1)*sizeof *rate);
    latency=calloc((size_t)made+1,sizeof *latency);
    trees=malloc(((size_t)p->demands+1)*sizeof *trees);
    if(!p->tree || !p->run || !rate || !latency || !trees){status=ENOMEM;goto done;}
    mesh_graph_rows(p->demands,made,plan_at,demand,trees,p->tree);
    for(uint32_t i=0;i<p->demands;i++){
      p->demand[i].tree=(uint32_t)trees[i];
      p->demand[i].trees=(uint32_t)(trees[i+1]-trees[i]);
    }
    for(uint32_t j=0;j<made;j++)rate[j]=share[p->tree[j]];
    for(uint32_t i=0;i<p->demands;i++){
      struct mesh_plan_demand *d=p->demand+i;
      mesh_graph_fill(d->trees,latency+d->tree,rate+d->tree,d->instructions,p->run+d->tree);
      uint64_t cut=0;
      for(uint32_t j=d->tree;j<d->tree+d->trees;j++)cut+=p->run[j];
      if(cut!=d->instructions){status=EHOSTUNREACH;goto done;}
    }
  }
  if((status=plan_trees(p)) || (status=plan_template(p,dependence,dependences)) || (status=plan_acyclic(p)) ||
     (status=plan_unroll(p)) || (status=plan_sets(p,m)))goto done;
  p->start_lead=p->copies-1;
  plan_walk(p);
  uint64_t entries;
  if(p->receives>=UINT32_MAX || __builtin_mul_overflow(p->entries,(uint64_t)p->copies,&entries) ||
     entries>=UINT64_MAX/(2*sizeof(struct mesh_hop))){
    status=ENOMEM;goto done;
  }
  uint64_t spanned=0;
  for(uint32_t i=0;i<p->demands;i++)spanned|=UINT64_C(1)<<p->demand[i].source|p->demand[i].leaves|p->demand[i].on;
  if(!spanned)spanned=UINT64_C(1)<<p->self;
  if(spanned&~program){status=EINVAL;goto done;}
  p->spanned=spanned;
  p->nodes=p->demands?spanned:UINT64_C(1)<<__builtin_ctzll(program);
  status=plan_times(p);
done:
  free(root);free(up);free(down);free(demand);free(share);free(rate);free(latency);free(trees);
  return status;
}

static void plan_free(struct mesh_plan_flow *p){
  free(p->named);free(p->cut);free(p->at);free(p->set_far);free(p->block_first);
  mesh_graph_cover_result_free(p->covers);free(p->touch);free(p->relay);
  for(uint32_t j=0;p->adjacency && j<p->made;j++)mesh_graph_tree_free(p->adjacency[j]);
  free(p->adjacency);
  free(p->demand);free(p->own.demand);free(p->own.demand_of);free(p->own.release_of);free(p->parent);
  free(p->tree);free(p->run);free(p->members);free(p->lead);free(p->comm);free(p->child);free(p->value);free(p->read);
  free(p->here);mesh_graph_group_free(p->rows);free(p->edge);free(p->between);free(p->farthest);free(p->entry);
  free(p->acyclic_indegree);free(p->acyclic_ordinal);free(p->lag_source);free(p->lag_target);
  mesh_graph_lag_matrix_free(p->lags);mesh_graph_lag_graph_free(p->lag_graph);
  free(p->bound);free(p->value_lag);
}

/* one arena shared: its two ranges equal (mesh_arena_range's, plan_place's `one`) */
static int plan_arena_shared(const struct mesh_range *arena){
  return arena[0].first==arena[1].first && arena[0].count==arena[1].count;
}
/* design/plan-to-bridge-change-request.md R8; design/aot-compilation-change-request.md, appendix item 6: one
   preparation at a time, its whole holder identity in hdr `preparing`, taken by one 16-byte compare-and-swap from
   empty or from the exact identity mesh_holds finds gone, else waited on (os_sync_wait_on_address_with_timeout) until
   `deadline`; held from reservation to publication or unwinding */
static int plan_prepare(struct mesh_ctx *context,uint64_t deadline){
  struct hdr *m=context->M;
  const struct mesh_identity mine={context->client,context->start};
  struct mesh_identity seen={0,0};
  for(;;){
    if(mesh_holder_swap(&m->preparing,&seen,mine))return 0;
    if(!mesh_holder_live(seen))continue;
    const uint64_t now=clock_gettime_nsec_np(CLOCK_MONOTONIC);
    if(now>=deadline)return ETIMEDOUT;
    os_sync_wait_on_address_with_timeout(&m->preparing.client,seen.client,sizeof seen.client,
                                         OS_SYNC_WAIT_ON_ADDRESS_SHARED,OS_CLOCK_MACH_ABSOLUTE_TIME,deadline-now);
    seen=(struct mesh_identity){0,0};
  }
}
static void plan_prepared(struct mesh_ctx *context){
  struct hdr *m=context->M;
  mesh_holder_store(&m->preparing,(struct mesh_identity){0,0});
  os_sync_wake_by_address_all(&m->preparing.client,sizeof m->preparing.client,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
/* every two reservations live at once */
static int plan_together(const void *context,size_t a,size_t b){
  (void)context;(void)a;(void)b;
  return 1;
}
/* design/aot-compilation-change-request.md, appendix item 6; TensorFlow Lite GreedyBySizeAssignment
   (mesh_graph_greedy_by_size): the extent of this preparation in one space, `size` of `room`, placed around every
   live publication's (`taken`, offsets from the space's first, pinned), `align` apart; ENOSPC where it ends past the
   room */
static int plan_fit(uint64_t taken,uint64_t *size,uint64_t *offset,uint64_t *pinned,uint64_t room,uint64_t align,
                    uint64_t *at){
  uint64_t total=0;
  for(uint64_t r=0;r<taken;r++)pinned[r/64]|=UINT64_C(1)<<(r%64);
  int status=mesh_graph_greedy_by_size((size_t)taken+1,size,pinned,plan_together,NULL,align,offset,&total);
  for(uint64_t r=0;r<=taken;r++)pinned[r/64]=0;
  if(!status && (offset[taken]>room || size[taken]>room-offset[taken]))status=ENOSPC;
  *at=offset[taken];
  return status;
}
/* the reservation, under plan_prepare: every live publication's (a slot open or closed: its rows and pages not
   released) rows and distinct arenas pinned, this plan's `rows` rows and used[a] blocks of each of `arenas` arenas
   (1 where one is shared) placed around them, into `row` and arena[a] (arena[1] arena[0] where shared); waited for, on
   `plans` until `deadline`, while it does not fit (ENOSPC). A slot's record is read under `preparing`: only a preparation
   reuses released pages */
static int plan_reserve(struct hdr *m,uint32_t rows,const uint64_t *used,uint32_t arenas,uint64_t deadline,
                        struct mesh_range *row,struct mesh_range *arena){
  for(;;){
    const uint64_t plans=atomic_load_explicit(&m->plans,memory_order_acquire);
    uint64_t live=0;
    for(uint32_t r=0;r<m->rows;r++)live+=mesh_slot(m,r)!=0;
    uint64_t *size=malloc((live+1)*sizeof *size),*offset=malloc((live+1)*sizeof *offset),
             *pinned=calloc(live/64+1,sizeof *pinned);
    int status=size && offset && pinned?0:ENOMEM;
    const struct mesh_algo **record=status?NULL:malloc((live+1)*sizeof *record);
    if(!status && !record)status=ENOMEM;
    uint64_t taken=0;
    for(uint32_t r=0;!status && r<m->rows && taken<live;r++)if(mesh_slot(m,r))record[taken++]=mesh_slot_record(m,r);
    uint64_t at=0;
    for(uint64_t k=0;!status && k<taken;k++){size[k]=record[k]->rows.count;offset[k]=record[k]->rows.first;}
    if(!status){size[taken]=rows;status=plan_fit(taken,size,offset,pinned,mesh_rows(m),1,&at);}
    if(!status)*row=(struct mesh_range){(uint32_t)at,rows};
    for(uint32_t a=0;!status && a<arenas;a++){
      const struct mesh_range range=mesh_arena_range(m,(int)a);
      for(uint64_t k=0;k<taken;k++){size[k]=record[k]->arena[a].count;offset[k]=record[k]->arena[a].first-range.first;}
      size[taken]=used[a]*m->block;
      status=plan_fit(taken,size,offset,pinned,range.count,m->block,&at);
      if(!status)arena[a]=(struct mesh_range){range.first+(uint32_t)at,(uint32_t)size[taken]};
    }
    if(!status && arenas==1)arena[1]=arena[0];
    free(size);free(offset);free(pinned);free(record);
    const uint64_t now=clock_gettime_nsec_np(CLOCK_MONOTONIC);
    if(status!=ENOSPC || now>=deadline)return status;
    os_sync_wait_on_address_with_timeout(&m->plans,plans,sizeof m->plans,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                                         OS_CLOCK_MACH_ABSOLUTE_TIME,deadline-now);
  }
}

/* each value row's page: page[r - row], `row` the first row of the plan's reservation */
static void plan_bind_table(struct mesh_ctx *context,const struct mesh_value *values,const struct mesh_plan_value *result,uint32_t count,
                            const uint32_t *page,uint32_t row){
  struct hdr *m=context->M;
  const uint64_t block=mesh_plan_quantum(m);
  for(uint32_t j=0;j<count;j++){
    const struct mesh_value *v=values+j;
    const struct mesh_section x=result[j].section;
    for(uint32_t c=0;c<x.count;c++)for(uint32_t k=0;k<x.stride;k++){
      const uint32_t r=row+x.first+c*x.stride+k;
      atomic_store_explicit(&mesh_page(m)[r].device,0,memory_order_relaxed);
      mesh_buffers(m)[r].constant=0;
      mesh_backing_bind(context,r,m->block,page[r-row],c);
      if(v->state){const uint64_t left=v->section.bytes-(uint64_t)k*block;memset(mesh_at(m,page[r-row]),0,(size_t)(left<block?left:block));}
    }
  }
}
static void plan_bind_values(struct mesh_ctx *context,const struct mesh_function *functions,const struct mesh_planned *planned,uint32_t count,
                             const uint32_t *page,uint32_t row){
  for(uint32_t f=0;f<count;f++)plan_bind_table(context,functions[f].value,planned->result[f].value,functions[f].values,page,row);
}
static int plan_publish(struct mesh_ctx *context,uint64_t group,uint32_t count,
  uint32_t table,const struct mesh_nodes *nodes,struct mesh_plan_binding *planned){
  struct mesh_nodes after;
  int status=mesh_nodes_read(context,&after);
  if(status)return status;
  if(mesh_view_compare((struct mesh_view_id){.id=after.serial,.sequencer=after.sequencer,.boot=after.boot},
      (struct mesh_view_id){.id=nodes->serial,.sequencer=nodes->sequencer,.boot=nodes->boot}))return ESTALE;
  struct hdr *m=context->M;
  struct mesh_algo *algo=(struct mesh_algo *)(void *)mesh_at(m,table);
  *algo=(struct mesh_algo){.plans=count,.nodes=nodes->count,.group=group,
    .table=m->data_off+(uint64_t)table*m->pgsz+sizeof *algo,.holder={context->client,context->start},
    .rows=planned->rows,.arena={planned->arena[0],planned->arena[1]}};
  algo->adjacent=algo->table+(uint64_t)count*sizeof(uint64_t);
  /* its root row maps the record's page; its slot, stored last (release), publishes it */
  const uint32_t root=planned->rows.first+planned->rows.count-1;
  mesh_backing_bind(context,root,m->block,table,0);
  planned->generation=atomic_fetch_add_explicit(&m->generation,2,memory_order_relaxed)+2;
  atomic_store_explicit(&mesh_page(m)[root].publication,planned->generation,memory_order_release);
  mesh_notify(&m->plans);
  return 0;
}

/* design/joint-plan-results.md#archives */
struct plan_bytes { size_t references; void (*release)(void *); void *owner; };
static void plan_bytes_release(void *bytes){
  struct plan_bytes *b=bytes;
  if(__atomic_fetch_sub(&b->references,1,__ATOMIC_ACQ_REL)==1){
    if(b->release)b->release(b->owner);
    free(b);
  }
}
struct plan_loaded { size_t references; struct plan_bytes *bytes; uint32_t relations; struct mesh_graph_relation **relation;
                     struct mesh_planned_call *call; struct mesh_function *function;
                     struct mesh_graph_closure *closure; struct mesh_plan_actors *actors; struct mesh_declared *declared;
                     struct mesh_grouped *grouped; void *rewritten; };
static struct plan_loaded *plan_loaded_new(uint32_t relations){
  struct plan_loaded *l=calloc(1,sizeof *l);
  if(!l)return NULL;
  l->bytes=calloc(1,sizeof *l->bytes);
  l->relation=calloc((size_t)relations+1,sizeof *l->relation);
  if(!l->bytes || !l->relation){free(l->bytes);free(l->relation);free(l);return NULL;}
  l->references=1;l->bytes->references=1;l->relations=relations;
  return l;
}
static void plan_loaded_free(struct plan_loaded *l){
  if(__atomic_fetch_sub(&l->references,1,__ATOMIC_ACQ_REL)!=1)return;
  for(uint32_t k=0;k<l->relations;k++)mesh_graph_relation_free(l->relation[k]);
  free(l->relation);free(l->call);free(l->function);free(l->closure);free(l->actors);free(l->declared);free(l->grouped);free(l->rewritten);
  plan_bytes_release(l->bytes);
  free(l);
}
static int plan_results(struct mesh_planned *out,const struct mesh_function *source,uint32_t count,uint32_t roots){
  out->results=count;out->roots=roots;
  out->result=calloc((size_t)count+1,sizeof *out->result);
  out->root=calloc((size_t)roots+1,sizeof *out->root);
  if(!out->result || !out->root)return ENOMEM;
  for(uint32_t f=0;f<count;f++){
    struct mesh_plan_function *x=out->result+f;
    x->value=calloc((size_t)source[f].values+1,sizeof *x->value);
    x->completion=calloc((size_t)source[f].tasks+1,sizeof *x->completion);
    x->order=calloc((size_t)source[f].tasks+1,sizeof *x->order);
    if(!x->value || !x->completion || !x->order)return ENOMEM;
  }
  return 0;
}
void mesh_planned_free(struct mesh_planned *planned){
  if(planned->signature)mesh_graph_signature_free(planned->signature);
  free(planned->signature);
  if(!planned->owned){
    for(uint32_t f=0;planned->result && f<planned->results;f++){
      free(planned->result[f].value);free(planned->result[f].completion);free(planned->result[f].order);
    }
    free(planned->root);
  }
  free(planned->result);
  if(planned->owned)plan_loaded_free(planned->owned);
  else{
    free(planned->call);free(planned->order);free(planned->cause);
    free(planned->first);free(planned->transfer);free(planned->extent);
  }
  mesh_plan_analysis_free(planned->analysis);
  *planned=(struct mesh_planned){0};
}

/* design/compilation-composition-change-request.md items 3, 6.4 and 7; design/plan-to-bridge-change-request.md R2 and
   R5: each function's entries (plan_walk, in plan_schedule's order) by demand (mesh_graph_rows), its RECEIVE entries
   then its SEND entries, each the logical bytes its hops carry: the value's (p->at) where the demand is this node's,
   else the demand's (p->cut) */
static uint32_t plan_entry_demand(const void *context,size_t k){
  return ((const struct plan_entry *)context)[k].source->demand;
}
/* room for n + 1 records of `width` bytes: 1 where that overflows */
static int plan_room(size_t n,size_t width,size_t *bytes){
  return __builtin_add_overflow(n,(size_t)1,bytes) || __builtin_mul_overflow(*bytes,width,bytes);
}
static int plan_extents(const struct plan_step *plans,uint32_t count,struct mesh_planned *planned){
  size_t transfers=0,extents=0,at=0,base=0,room[3];
  int status=0;
  for(uint32_t f=0;!status && f<count;f++)
    if(__builtin_add_overflow(transfers,(size_t)plans[f].flow->transfers,&transfers) ||
       __builtin_add_overflow(extents,(size_t)plans[f].flow->entries,&extents))status=ENOMEM;
  if(status || plan_room(count,sizeof *planned->first,room) || plan_room(transfers,sizeof *planned->transfer,room+1) ||
     plan_room(extents,sizeof *planned->extent,room+2))return ENOMEM;
  planned->functions=count;planned->extents=extents;
  planned->first=malloc(room[0]);planned->transfer=malloc(room[1]);planned->extent=malloc(room[2]);
  if(!planned->first || !planned->transfer || !planned->extent)return ENOMEM;
  for(uint32_t f=0;!status && f<count;f++){
    const struct plan_step *p=plans+f;
    size_t rows,orders,*row=NULL,*order=NULL,*sending=NULL;
    if(plan_room(p->flow->demands,sizeof *row,&rows) || plan_room((size_t)p->flow->entries,sizeof *order,&orders) ||
       !(row=malloc(rows)) || !(order=malloc(orders)) || !(sending=malloc(rows)))status=ENOMEM;
    else{
      mesh_graph_rows(p->flow->demands,(size_t)p->flow->entries,plan_entry_demand,p->entry,row,order);
      for(uint32_t i=0;i<p->flow->demands;i++){
        size_t received=base+row[i],sent=received;
        for(size_t q=row[i];q<row[i+1];q++)sent+=p->entry[order[q]].source->direction==MESH_RECEIVE;
        sending[i]=sent;
        for(size_t q=row[i];q<row[i+1];q++){
          const struct mesh_plan_entry *e=p->entry[order[q]].source;
          const struct mesh_plan_demand *d=p->flow->demand+i;
          const uint64_t *cut=p->flow->cut+d->cut+e->instruction;
          const uint64_t start=d->mine==MESH_ABSENT && d->root==MESH_ABSENT?cut[0]:p->flow->at[d->at+e->instruction];
          const int receive=e->direction==MESH_RECEIVE;
          planned->extent[receive?received++:sent++]=(struct mesh_graph_access){.start=start,
            .end=start+cut[1]-cut[0],.vertex=receive?e->field:0};
        }
      }
      planned->first[f]=at;
      for(uint32_t t=0;t<p->flow->transfers;t++){
        const uint32_t i=p->flow->of[t];
        planned->transfer[at++]=(struct mesh_planned_transfer){base+row[i],sending[i],base+row[i+1]};
      }
      base+=(size_t)p->flow->entries;
    }
    free(row);free(order);free(sending);
  }
  if(!status)planned->first[count]=at;
  return status;
}

/* design/torch-capture-change-request.md item 2 and its 2026-11-09 appendix: the horizon and the locations of the
   values the host writes, Bff of the host -> device edges of the calls' compiled parts in program order from their
   t(v), with each call's own dependences between its parts (mesh_graph_buffer_bound) */
struct plan_bound_source {
  const struct mesh_function *function;
  const uint32_t *calls;
  uint32_t functions,ncalls,actors;
  size_t edges;
  double *host,*work;
  struct mesh_graph_dependence *edge;
  uint32_t *of;
  void (*release)(void *);
  void *owner;
};
static void plan_bound_source_free(void *owner){
  struct plan_bound_source *s=owner;
  if(s->release)s->release(s->owner);
  free(s->host);free(s->work);free(s->edge);free(s->of);free(s);
}
static void plan_horizon_free(struct mesh_plan_analysis *a){
  mesh_graph_buffer_result_free(a->bounds);
  free(a->function_lag);
}
static int plan_bound_result(struct mesh_plan_analysis *a,struct mesh_graph_buffer_result *bounds,
                             const uint32_t *function){
  a->bounds=bounds;mesh_graph_buffer_result_retain(bounds);a->actor_function=function;a->horizon=1;
  a->function_lag=calloc((size_t)a->functions+1,sizeof *a->function_lag);
  if(!a->function_lag)return ENOMEM;
  for(size_t q=0;q<bounds->queries;q++){
    if(function[q]>=a->functions)return EINVAL;
    const uint64_t bound=bounds->bound[q];
    if(bound>=UINT32_MAX)return ENOMEM;
    if(bound>a->horizon)a->horizon=(uint32_t)bound;
    if(bound>a->function_lag[function[q]])a->function_lag[function[q]]=bound;
  }
  for(uint32_t f=0;f<a->functions;f++)
    for(uint32_t j=0;j<a->function[f].values;j++)if(a->function[f].value[j].lag)
      a->flow[f].value_lag[j]=(uint32_t)a->function_lag[f];
  return 0;
}
static int plan_horizon(struct mesh_plan_analysis *a){
  const struct mesh_function *functions=a->function;
  const uint32_t *calls=a->calls,ncalls=a->ncalls;
  uint64_t actors=0,dependences=0;
  for(uint32_t j=0;j<ncalls;j++){
    const struct mesh_function *f=functions+calls[j];
    if(f->parts && (!f->host || !f->work))return EINVAL;
    if(f->edges && !f->edge)return EINVAL;
    for(uint32_t e=0;e<f->edges;e++)if(f->edge[e].from>=2*(uint64_t)f->parts || f->edge[e].to>=2*(uint64_t)f->parts)return EINVAL;
    actors+=f->parts?f->parts:1;dependences+=f->edges;
  }
  if(actors>=UINT32_MAX/4 || dependences>=SIZE_MAX/sizeof(struct mesh_graph_dependence))return ENOMEM;
  const uint32_t A=(uint32_t)actors;
  struct plan_bound_source *source=calloc(1,sizeof *source);
  if(!source)return ENOMEM;
  source->function=functions;source->functions=a->functions;source->calls=calls;source->ncalls=ncalls;source->actors=A;
  double *host=source->host=calloc((size_t)A+1,sizeof *host),*work=source->work=calloc((size_t)A+1,sizeof *work);
  uint32_t *of=source->of=malloc(((size_t)A+1)*sizeof *of);
  struct mesh_graph_dependence *edge=source->edge=malloc(((size_t)dependences+1)*sizeof *edge);
  int bounded=host && work && of && edge?0:ENOMEM;
  uint32_t base=0;
  size_t made=0;
  for(uint32_t j=0;!bounded && j<ncalls;j++){
    const struct mesh_function *f=functions+calls[j];
    const uint32_t parts=f->parts?f->parts:1;
    for(uint32_t q=0;q<parts;q++){
      if(f->parts){host[base+q]=f->host[q];work[base+q]=f->work[q];}
      of[base+q]=calls[j];
    }
    for(uint32_t e=0;e<f->edges;e++){
      const uint32_t x=f->edge[e].from,y=f->edge[e].to;
      edge[made++]=(struct mesh_graph_dependence){x<parts?base+x:A+base+x-parts,y<parts?base+y:A+base+y-parts,
                                                  f->edge[e].delay};
    }
    base+=parts;
  }
  struct mesh_graph_buffer_result *bounds=NULL;
  source->edges=made;
  if(!bounded)bounded=mesh_graph_buffer_bounds(A,host,work,made,edge,&bounds);
  if(bounds){
    source->release=bounds->release;source->owner=bounds->owner;
    bounds->release=plan_bound_source_free;bounds->owner=source;
    a->bound_host=host;a->bound_work=work;a->bound_edge=edge;a->bound_function=of;
    bounded=plan_bound_result(a,bounds,of);
  }else plan_bound_source_free(source);
  mesh_graph_buffer_result_free(bounds);
  return bounded;
}

/* design/compilation-composition-change-request.md items 3, 6.4 and 7; design/torch-capture-change-request.md item 6;
   design/aot-compilation-change-request.md A10 and appendix item 6: one program's lowered machine functions' own
   storage, no transfer: plan_horizon's horizon and lags over its calls, plan_unroll's copies of each function's calls,
   every run value's and run root's copies placed by one plan_block in the caller's arena (offsets plan_block's quanta of
   `quantum` bytes), the calls' parts, orders and chains as mesh_plan's planned result records them */
static int plan_analysis_new(const struct mesh_function *functions,uint32_t count,const uint32_t *calls,uint32_t ncalls,
  const struct mesh_graph_relation *order,const uint32_t *ordered,uint32_t norders,const struct mesh_graph_closure *closure,
  const uint32_t *chains,const struct mesh_value *roots,uint32_t nroots,const size_t *use_first,
  const struct mesh_local_use *use,struct mesh_plan_analysis **out){
  if(!out)return EINVAL;
  *out=NULL;
  if(!functions || !count || !calls || !ncalls || (norders && (!order || !ordered)) ||
     (nroots && (!roots || !use_first || !use)) || (closure && closure->vertices!=ncalls))return EINVAL;
  for(uint32_t f=0;f<count;f++)if(functions[f].values && !functions[f].value)return EINVAL;
  for(uint32_t j=0;j<ncalls;j++){
    const uint32_t h=chains?chains[j]:j;
    if(calls[j]>=count || h>j || (chains && chains[h]!=h))return EINVAL;
  }
  for(uint32_t r=0;r<nroots;r++)for(size_t k=use_first[r];k<use_first[r+1];k++)
    if(use[k].call>=ncalls || use[k].value>=functions[calls[use[k].call]].values)return EINVAL;
  const struct mesh_graph_relation_data *d=order?mesh_graph_relation_data(order):NULL;
  for(uint32_t q=0;q<norders;q++)
    if(ordered[q]>=d->edges || d->edge[ordered[q]].from>=d->edge[ordered[q]].to || d->edge[ordered[q]].to>=ncalls)
      return EINVAL;
  struct mesh_plan_analysis *a=calloc(1,sizeof *a);
  if(!a)return ENOMEM;
  a->references=1;a->function=functions;a->functions=count;a->calls=calls;a->ncalls=ncalls;a->relation=order;
  a->ordered=ordered;a->norders=norders;a->closure=closure;a->chains=chains;a->root=roots;a->roots=nroots;
  a->use_first=use_first;a->use=use;
  a->flow=calloc((size_t)count+1,sizeof *a->flow);
  a->carried=calloc((size_t)nroots+1,1);
  int status=a->flow && a->carried?0:ENOMEM;
  if(!status)status=mesh_graph_group(count,ncalls,plan_at,calls,&a->call_group);
  if(!status)status=mesh_graph_chain_pairs(ncalls,chains,&a->chain_pairs);
  for(uint32_t f=0;!status && f<count;f++){
    struct mesh_plan_flow *p=a->flow+f;
    p->given=functions[f].value;p->givens=functions[f].values;p->calls=(uint32_t)(a->call_group->first[f+1]-a->call_group->first[f]);
    p->value_lag=calloc((size_t)p->givens+1,sizeof *p->value_lag);
    if(!p->value_lag)status=ENOMEM;
  }
  if(status)mesh_plan_analysis_free(a);else *out=a;
  return status;
}
void mesh_plan_analysis_retain(struct mesh_plan_analysis *a){
  __atomic_fetch_add(&a->references,1,__ATOMIC_RELAXED);
}
void mesh_plan_analysis_free(struct mesh_plan_analysis *a){
  if(!a || __atomic_fetch_sub(&a->references,1,__ATOMIC_ACQ_REL)!=1)return;
  if(a->owned){
    for(uint32_t f=0;a->flow && f<a->functions;f++){free(a->flow[f].demand);plan_flow_loaded_free(a->flow+f);}
    free(a->flow);free(a->call_group);free(a->source_group);free(a->chain_pairs);mesh_graph_buffer_result_free(a->bounds);plan_loaded_free(a->owned);free(a);return;
  }
  plan_horizon_free(a);
  for(uint32_t f=0;a->flow && f<a->functions;f++)plan_free(a->flow+f);
  free(a->flow);free(a->carried);
  mesh_graph_group_free(a->call_group);mesh_graph_group_free(a->source_group);mesh_graph_chain_result_free(a->chain_pairs);
  free(a->environment);free(a);
}
static void plan_steps_free(struct plan_step *steps,uint32_t count){
  for(uint32_t f=0;steps && f<count;f++){
    struct plan_step *p=steps+f;
    plan_steps_results_free(p);
    free(p->item);free(p->entry);free(p->completed);free(p->sync);free(p->wait_row);
  }
  free(steps);
}
static int plan_steps(struct mesh_plan_analysis *a,struct mesh_planned *out,struct plan_step **steps){
  const struct mesh_function *functions=a->function;
  int status=plan_results(out,functions,a->functions,a->roots);
  if(status)return status;
  out->analysis=a;mesh_plan_analysis_retain(a);
  *steps=calloc((size_t)a->functions+1,sizeof **steps);
  if(!*steps)return ENOMEM;
  for(uint32_t f=0;f<a->functions;f++){
    struct plan_step *p=*steps+f;
    p->flow=a->flow+f;p->function=functions+f;p->result=out->result+f;p->root_result=out->root;
    if(a->owned)p->result->actors=((const struct plan_loaded *)a->owned)->actors+f;
    p->item=malloc(((size_t)p->flow->givens+1)*sizeof *p->item);
    p->entry=malloc(((size_t)p->flow->entries+1)*sizeof *p->entry);
    if(!p->item || !p->entry)return ENOMEM;
    for(uint32_t j=0;j<p->flow->givens;j++)p->result->value[j].lag=p->flow->value_lag[j];
    for(uint64_t j=0;j<p->flow->entries;j++)p->entry[j]=(struct plan_entry){p->flow->entry+j,0};
  }
  return 0;
}
int mesh_plan_local_analyze(const struct mesh_function *functions,uint32_t count,const uint32_t *calls,uint32_t ncalls,
  const struct mesh_graph_relation *order,const uint32_t *ordered,uint32_t norders,const struct mesh_graph_closure *closure,
  const uint32_t *chains,const struct mesh_value *roots,uint32_t nroots,const size_t *use_first,
  const struct mesh_local_use *use,struct mesh_plan_analysis **out){
  int status=plan_analysis_new(functions,count,calls,ncalls,order,ordered,norders,closure,chains,roots,nroots,
                                use_first,use,out);
  if(status)return status;
  struct mesh_plan_analysis *a=*out;
  status=plan_horizon(a);
  for(uint32_t f=0;!status && f<count;f++){
    struct mesh_plan_flow *p=a->flow+f;
    p->horizon=a->horizon;
    p->named=calloc((size_t)p->givens+1,sizeof *p->named);
    status=p->named?plan_unroll(p):ENOMEM;
  }
  if(status){mesh_plan_analysis_free(a);*out=NULL;}
  return status;
}
int mesh_plan_local_complete(struct mesh_plan_analysis *a,uint64_t quantum,struct mesh_local *out){
  if(!out)return EINVAL;
  *out=(struct mesh_local){0};
  if(!a || !quantum)return EINVAL;
  const struct mesh_function *functions=a->function;
  const struct mesh_value *roots=a->root;
  const uint32_t count=a->functions,ncalls=a->ncalls,norders=a->norders,nroots=a->roots,horizon=a->horizon;
  const uint32_t *calls=a->calls,*chains=a->chains,*ordered=a->ordered;
  const size_t *ordinal=a->call_group->ordinal,*first=a->call_group->first;
  const struct mesh_graph_relation *order=a->relation;
  const struct mesh_graph_closure *closure=a->closure;
  const size_t *use_first=a->use_first;
  const struct mesh_local_use *use=a->use;
  const struct mesh_graph_relation_data *d=order?mesh_graph_relation_data(order):NULL;
  struct plan_step *plans=NULL;
  uint32_t *root_item=malloc(((size_t)nroots+1)*sizeof *root_item);
  uint64_t *operations=malloc(((size_t)count+1)*sizeof *operations);
  struct mesh_plan_value **held=NULL;
  struct plan_placement *x=out->placement=calloc(1,sizeof *x);
  struct mesh_graph_view *view=NULL;
  struct mesh_graph_conflict *conflict=NULL;
  size_t values=nroots,offsets=0,edges=norders+2*(size_t)ncalls,conflicts=0,room[6];
  int status=plan_steps(a,&out->planned,&plans);
  if(!status && (!operations || !root_item || !x))status=ENOMEM;
  for(uint32_t f=0;!status && f<count;f++){
    operations[f]=plan_operations(plans+f);values+=functions[f].values;
    for(uint32_t j=0;j<functions[f].values;j++)if(functions[f].value[j].run)offsets+=plan_copies(a->flow+f,j);
  }
  for(uint32_t j=0;!status && j<ncalls;j++){
    const struct mesh_plan_actors *actors=out->planned.result[calls[j]].actors;
    edges+=actors?actors->edges:functions[calls[j]].edges;
  }
  for(uint32_t r=0;r<nroots;r++)if(roots[r].run)offsets+=horizon;
  const struct plan_program program={a,roots,root_item,out->planned.root};
  if(x)x->operations=operations;else free(operations);
  if(!status)status=plan_block(plans,count,quantum,operations,x,&program);
  if(!status && (plan_room(values+1,sizeof *out->first,room) || plan_room(offsets,sizeof *out->offset,room+1) ||
                 plan_room(edges,sizeof *out->planned.order,room+2) ||
                 plan_room(edges,sizeof *out->planned.cause,room+3) || plan_room(offsets,sizeof *view,room+4) ||
                 plan_room(values,sizeof *held,room+5)))status=ENOMEM;
  if(!status){
    out->first=malloc(room[0]);out->offset=malloc(room[1]);
    out->planned.call=malloc(((size_t)ncalls+1)*sizeof *out->planned.call);
    out->planned.order=malloc(room[2]);out->planned.cause=malloc(room[3]);view=malloc(room[4]);held=malloc(room[5]);
    out->value_first=malloc(((size_t)count+1)*sizeof *out->value_first);
    if(!out->first || !out->offset || !out->planned.call || !out->planned.order || !out->planned.cause || !view ||
       !held || !out->value_first)status=ENOMEM;
  }
  /* design/torch-capture-change-request.md item 6: each run copy's assigned bytes one view of the arena, a value or
     root sharing where one of its copies meets another copy (mesh_graph_conflicts_views, distinct witnesses) */
  if(!status){
    size_t at=0,g=0;
    for(uint32_t f=0;f<count;f++){
      out->value_first[f]=g;
      for(uint32_t j=0;j<plans[f].flow->givens;j++,g++){
        const struct mesh_value *source=functions[f].value+j;
        struct mesh_plan_value *v=out->planned.result[f].value+j;
        held[g]=v;out->first[g]=at;
        v->section=source->section;v->section.count=plan_copies(a->flow+f,j);
        for(uint32_t c=0;source->run && c<v->section.count;c++,at++){
          out->offset[at]=x->at[plans[f].item[j]+c]*quantum;
          view[at]=(struct mesh_graph_view){.view={.offset=out->offset[at],.element=v->section.bytes},
                                            .vertex=(uint32_t)g,.write=1};
        }
      }
    }
    out->value_first[count]=g;
    for(uint32_t r=0;r<nroots;r++,g++){
      const struct mesh_value *source=roots+r;
      struct mesh_plan_value *v=out->planned.root+r;
      held[g]=v;out->first[g]=at;
      v->section=source->section;v->section.count=horizon;
      for(uint32_t t=0;source->run && t<horizon;t++,at++){
        out->offset[at]=x->at[root_item[r]+t]*quantum;
        view[at]=(struct mesh_graph_view){.view={.offset=out->offset[at],.element=v->section.bytes},
                                          .vertex=(uint32_t)g,.write=1};
      }
    }
    out->first[values]=at;
    status=mesh_graph_conflicts_views(offsets,view,0,NULL,NULL,&conflict,&conflicts);
    for(size_t k=0;!status && k<conflicts;k++)
      if(conflict[k].from_view!=conflict[k].to_view){
        held[view[conflict[k].from_view].vertex]->shares=1;
        held[view[conflict[k].to_view].vertex]->shares=1;
      }
  }
  if(!status){
    size_t o=0;
    out->values=values;out->offsets=offsets;out->bytes=x->blocks*quantum;out->quantum=quantum;
    out->functions=count;out->function=functions;out->roots=nroots;out->root=roots;out->calls=calls;
    out->chains=chains;out->ordered=ordered;out->ncalls=ncalls;out->norders=norders;out->relation=order;
    out->closure=closure;out->use_first=use_first;out->use=use;
    out->planned.calls=ncalls;out->planned.horizon=horizon;
    for(uint32_t j=0;j<ncalls;j++){
      const struct mesh_function *f=functions+calls[j];
      const struct mesh_plan_actors *actors=out->planned.result[calls[j]].actors;
      out->planned.call[j]=(struct mesh_planned_call){.function=calls[j],.ordinal=ordinal[j],.every=(uint32_t)(first[calls[j]+1]-first[calls[j]]),
                                                      .parts=actors?actors->parts:f->parts};
      for(uint32_t e=0;e<(actors?actors->edges:f->edges);e++,o++){
        out->planned.order[o]=(actors?actors->edge:f->edge)[e];
        out->planned.cause[o]=(struct mesh_planned_cause){MESH_PLANNED_PART,j};
      }
    }
    /* design/aot-compilation-change-request.md A10: Bernstein's order between calls, its cause the edge's own */
    for(uint32_t q=0;q<norders;q++,o++){
      const struct mesh_graph_dependence *e=d->edge+ordered[q];
      uint32_t cause;
      status=mesh_graph_relation_edge_cause(order,ordered[q],&cause);
      if(status)break;
      const uint32_t kind=cause==MESH_GRAPH_GIVEN?MESH_ORDER_FLOW:MESH_ORDER_CONFLICT;
      out->planned.order[o]=(struct mesh_graph_dependence){e->from,e->to,0};
      out->planned.cause[o]=(struct mesh_planned_cause){kind,q};
    }
    /* each chain's calls one at a time in program order, its last before the next block's first (delay 1), as
       plan_graph orders them */
    for(size_t j=0;!status && j<a->chain_pairs->edges;j++){
      out->planned.order[o]=a->chain_pairs->edge[j];
      out->planned.cause[o++]=(struct mesh_planned_cause){MESH_PLANNED_CHAIN,UINT32_MAX};
    }
    out->planned.orders=o;
  }
  plan_steps_free(plans,count);free(root_item);
  free(held);free(view);free(conflict);
  if(status)mesh_local_free(out);
  return status;
}
int mesh_plan_local(const struct mesh_function *functions,uint32_t count,const uint32_t *calls,uint32_t ncalls,
                    const struct mesh_graph_relation *order,const uint32_t *ordered,uint32_t norders,
                    const struct mesh_graph_closure *closure,const uint32_t *chains,const struct mesh_value *roots,
                    uint32_t nroots,const size_t *use_first,const struct mesh_local_use *use,uint64_t quantum,
                    struct mesh_local *out){
  if(!out)return EINVAL;
  *out=(struct mesh_local){0};
  struct mesh_plan_analysis *analysis=NULL;
  int status=mesh_plan_local_analyze(functions,count,calls,ncalls,order,ordered,norders,closure,chains,roots,nroots,
                                    use_first,use,&analysis);
  if(!status)status=mesh_plan_local_complete(analysis,quantum,out);
  mesh_plan_analysis_free(analysis);
  return status;
}
void mesh_local_free(struct mesh_local *out){
  plan_placement_free(out->placement,!!out->planned.owned);
  if(!out->planned.owned){
    free(out->first);free(out->offset);free(out->value_first);
    if(!out->planned.analysis){free(out->row);free(out->order);free(out->source_row);free(out->source_order);}
    free(out->demand_first);free(out->demand_of);free(out->release_of);
  }
  mesh_planned_free(&out->planned);
  *out=(struct mesh_local){0};
}

/* design/aot-compilation-change-request.md A7 to A10, appendix item 6;
   design/plan-to-bridge-change-request.md R3 to R5; M. Cowan et al., MSCCLang, ASPLOS 2023, §5.2;
   P. E. McKenney and J. D. Slingwine, PDCS 1998 */
/* mesh-collective.h mesh_plan_program's full result: each value's and root's copies, rows of pages from the planned rows'
   first (section.first + c * stride), first[] over function 0's values, ..., then the roots (value_first), and each
   instance's declared rows' demands (demand_first) with the release a row's sends share (release_of; MESH_ABSENT for a
   row this node does not send), and the caller's groupings those come from, borrowed (grouped; NULL without them) */
static int plan_result(const struct plan_step *plans,uint32_t count,const struct plan_program *joint,struct mesh_local *out){
  size_t values=joint->analysis->roots,offsets=0,transfers=0,room[6];
  for(uint32_t f=0;f<count;f++){
    values+=plans[f].function->values;transfers+=plans[f].flow->transfers;
    for(uint32_t j=0;j<plans[f].function->values;j++)offsets+=plans[f].result->value[j].section.count;
  }
  for(uint32_t r=0;r<joint->analysis->roots;r++)offsets+=joint->result[r].section.count;
  if(plan_room(values+1,sizeof *out->first,room) || plan_room(offsets,sizeof *out->offset,room+1) ||
     plan_room(count,sizeof *out->value_first,room+2) || plan_room(count,sizeof *out->demand_first,room+3) ||
     plan_room(transfers,sizeof *out->demand_of,room+4) || plan_room(transfers,sizeof *out->release_of,room+5))
    return ENOMEM;
  out->first=malloc(room[0]);out->offset=malloc(room[1]);out->value_first=malloc(room[2]);
  out->demand_first=malloc(room[3]);out->demand_of=malloc(room[4]);out->release_of=malloc(room[5]);
  if(!out->first || !out->offset || !out->value_first || !out->demand_first || !out->demand_of || !out->release_of)
    return ENOMEM;
  size_t g=0,at=0,t=0;
  for(uint32_t f=0;f<count;f++){
    const struct plan_step *p=plans+f;
    out->value_first[f]=g;out->demand_first[f]=t;
    for(uint32_t j=0;j<p->function->values;j++,g++){
      const struct mesh_section x=p->result->value[j].section;
      out->first[g]=at;
      for(uint32_t c=0;c<x.count;c++)out->offset[at++]=x.first+(uint64_t)c*x.stride;
    }
    for(uint32_t i=0;i<p->flow->transfers;i++,t++){
      out->demand_of[t]=p->flow->of[i];
      out->release_of[t]=p->flow->declared[i].source==p->flow->self?p->flow->demand[p->flow->of[i]].release:MESH_ABSENT;
    }
  }
  out->value_first[count]=g;out->demand_first[count]=t;
  for(uint32_t r=0;r<joint->analysis->roots;r++,g++){
    const struct mesh_section x=joint->result[r].section;
    out->first[g]=at;
    for(uint32_t c=0;c<x.count;c++)out->offset[at++]=x.first+(uint64_t)c*x.stride;
  }
  out->first[values]=at;out->values=values;out->offsets=offsets;out->grouped=joint->analysis->grouped;
  return 0;
}
/* mesh_plan and mesh_plan_program: `calls` the instances' rows, `source_calls` (NULL: the instances') the original
   source family's, with source_outstanding[] its issue bounds; `joint` the program's roots, uses and closure (no roots:
   mesh_plan's); `out` (NULL: mesh_plan) the full result's copies and demand correspondence */
struct plan_environment { struct mesh_nodes nodes; struct mesh_plan_geometry geometry; };
int mesh_plan_environment(struct mesh_ctx *context,struct mesh_nodes *nodes,struct mesh_plan_geometry *geometry){
  const struct hdr *m=context->M;
  *geometry=(struct mesh_plan_geometry){m->pgsz,m->block,mesh_rows(m),0,{mesh_arena_range(m,0),mesh_arena_range(m,1)}};
  return mesh_nodes_read(context,nodes);
}
static int plan_analyze_network(struct mesh_plan_analysis *a){
  const uint32_t count=a->functions,ncalls=a->ncalls,sources=a->sources,*calls=a->calls,*source_calls=a->source_calls;
  const struct mesh_function *functions=a->function;
  const uint64_t program=a->program;
  struct mesh_plan_flow *plans=a->flow;
  if(!a->geometry){
    for(uint32_t f=0;f<count;f++){
      plans[f].n=1;
      if(functions[f].count || functions[f].values)return EINVAL;
    }
    return 0;
  }
  int status=0;
  const uint32_t horizon=a->horizon;
  const struct mesh_nodes *nodes=a->nodes;
  const size_t *row=a->call_group->first;
  for(uint32_t f=0;f<count;f++)if(row[f]==row[f+1])return EINVAL;
  if(source_calls){
    for(uint32_t j=0;j<ncalls;j++)if(source_calls[j]>=sources)return EINVAL;
    if((status=mesh_graph_group(sources,ncalls,plan_at,source_calls,&a->source_group)))return status;
  }
  if(nodes->self>=nodes->count){status=EHOSTUNREACH;return status;}
  if(!(program>>nodes->self&1)){status=EINVAL;return status;}
  /* mesh-collective.h mesh_plan_program: the slots the calls' uses bind to program roots, once (each slot's storage
     its root's, PLAN_SEVERAL where its function's calls bind it to more than one) */
  for(uint32_t f=0;a->use && f<count;f++){
    if(!(plans[f].bound=malloc(((size_t)functions[f].values+1)*sizeof *plans[f].bound))){status=ENOMEM;return status;}
    for(uint32_t j=0;j<functions[f].values;j++)plans[f].bound[j]=MESH_ABSENT;
  }
  for(uint32_t r=0;a->use && r<a->roots;r++)for(size_t k=a->use_first[r];k<a->use_first[r+1];k++){
    uint32_t *b=plans[calls[a->use[k].call]].bound+a->use[k].value;
    *b=*b==MESH_ABSENT || *b==r?r:PLAN_SEVERAL;
  }
  /* design/aot-compilation-change-request.md A7; mesh-collective.h mesh_declared_group: each function's rows grouped
     once, the caller's grouping or, without one, made here */
  for(uint32_t f=0;f<count;f++){
    struct mesh_plan_flow *p=plans+f;
    p->geometry=a->geometry;p->n=nodes->count;p->self=nodes->self;

    p->view=(struct mesh_view_id){.id=nodes->serial,.sequencer=nodes->sequencer,.boot=nodes->boot};
    p->calls=(uint32_t)(row[f+1]-row[f]);p->horizon=horizon;
    p->roots=a->root;p->nroots=a->roots;p->carried=a->carried;
    if((uint64_t)horizon*p->calls>=UINT32_MAX){status=ENOMEM;return status;}
    /* mesh_plan's grouping (mesh_plan_program's is the caller's, required): the plan's own, the same algorithm once,
       its arrays held until plan_free (the plan's demand_of view of them) */
    if(!a->grouped){
      const uint32_t t=functions[f].count;
      p->own.demand=malloc(((size_t)t+1)*sizeof *p->own.demand);
      p->own.demand_of=malloc(((size_t)t+1)*sizeof *p->own.demand_of);
      p->own.release_of=malloc(((size_t)t+1)*sizeof *p->own.release_of);
      status=p->own.demand && p->own.demand_of && p->own.release_of?
             mesh_declared_group(functions[f].declared,functions[f].weights,t,nodes->count,nodes->self,&p->own):ENOMEM;
    }
    if(!status)status=plan_function(p,functions+f,nodes,program,a->grouped?a->grouped+f:&p->own);
    if(status)return status;
    p->entry=malloc((p->entries+1)*sizeof *p->entry);
    if(!p->entry)return ENOMEM;
    plan_walk(p);
    plan_groups(p);
  }
  return 0;
}
static void plan_source(struct mesh_local *out,const struct mesh_plan_analysis *a){
  out->functions=a->functions;out->function=a->function;out->roots=a->roots;out->root=a->root;out->calls=a->calls;
  out->chains=a->chains;out->ordered=a->ordered;out->ncalls=a->ncalls;out->norders=a->relation?a->norders:0;out->relation=a->relation;
  out->closure=a->closure;out->use_first=a->use_first;out->use=a->use;out->source_calls=a->source_calls;
  out->sources=a->sources;out->source_outstanding=a->source_outstanding;out->grouped=a->grouped;
  out->row=a->call_group->first;out->order=a->call_group->order;out->source_row=(a->source_group?a->source_group->first:NULL);out->source_order=(a->source_group?a->source_group->order:NULL);
}
void mesh_plan_compilation_retain(struct mesh_plan_compilation *c){
  __atomic_fetch_add(&c->references,1,__ATOMIC_RELAXED);
}
void mesh_plan_compilation_free(struct mesh_plan_compilation *c){
  if(!c || __atomic_fetch_sub(&c->references,1,__ATOMIC_ACQ_REL)!=1)return;
  const uint32_t count=c->result.functions;
  if(!c->owned){
    plan_steps_free(c->step,count);
    for(uint32_t f=0;c->state && f<count;f++)free(c->state[f]);
    free(c->root_item);free(c->page);free(c->adjacency);free(c->toward);free(c->place);free(c->rows_by);free(c->listed);
    free(c->state_bytes);
  }
  if(c->owned){
    for(uint32_t f=0;c->step && f<count;f++){free(c->step[f].entry);plan_step_loaded_free(c->step+f);}
    free(c->step);
  }
  free(c->state);
  mesh_local_free(&c->result);
  free(c);
}
static int plan_section(const struct mesh_plan_geometry *geometry,const struct mesh_value *value,
                        uint32_t copies,uint64_t *rows,struct mesh_section *section){
  const uint32_t pages=(uint32_t)plan_blocks(geometry,value->section.bytes);
  uint64_t blocks,end;
  if(__builtin_mul_overflow((uint64_t)copies,(uint64_t)pages,&blocks) ||
     __builtin_add_overflow(*rows,blocks,&end) || end>=geometry->rows)return ENOMEM;
  *section=(struct mesh_section){(uint32_t)*rows,pages*geometry->block,value->section.bytes,copies,pages};
  *rows=end;
  return 0;
}
static int plan_sections(struct mesh_plan_compilation *compiled){
  struct mesh_planned *planned=&compiled->result.planned;
  const struct mesh_plan_analysis *analysis=planned->analysis;
  uint64_t rows=0;
  int status=0;
  for(uint32_t f=0;!status && f<analysis->functions;f++){
    const struct plan_step *step=compiled->step+f;
    for(uint32_t j=0;!status && j<step->function->values;j++){
      const struct mesh_value *value=step->function->value+j;
      const int held=value->section.bytes && !(step->flow->bound && step->flow->bound[j]!=MESH_ABSENT);
      status=plan_section(analysis->geometry,value,held?plan_copies(step->flow,j):0,&rows,
                          &step->result->value[j].section);
    }
  }
  for(uint32_t r=0;!status && r<analysis->roots;r++){
    if(analysis->carried[r] && !analysis->root[r].run)return EINVAL;
    status=plan_section(analysis->geometry,analysis->root+r,analysis->root[r].run?analysis->horizon:0,
                        &rows,&planned->root[r].section);
  }
  planned->rows=(struct mesh_range){0,(uint32_t)rows+1};
  return status;
}
static int plan_program_placement(struct mesh_plan_compilation *compiled){
  struct mesh_planned *planned=&compiled->result.planned;
  const struct mesh_plan_analysis *analysis=planned->analysis;
  const struct mesh_plan_geometry *geometry=analysis->geometry;
  const struct plan_program program={analysis,analysis->root,compiled->root_item,planned->root};
  for(uint32_t f=0;f<analysis->functions;f++)plan_layout(compiled->step+f);
  int status=plan_place(compiled->step,analysis->functions,geometry,compiled->placement,&program);
  if(status)return status;
  for(uint32_t f=0;!status && f<analysis->functions;f++){
    struct plan_step *step=compiled->step+f;
    if(step->flow->demands || step->function->tasks)status=plan_schedule(step,step->function);
  }
  if(status)return status;
  compiled->page=malloc((size_t)planned->rows.count*sizeof *compiled->page);
  if(!compiled->page)return ENOMEM;
  status=plan_rows(compiled->step,analysis->functions,geometry,compiled->placement,compiled->page,
                   planned->rows.count-1,&program);
  if(status)return status;
  planned->used[0]=compiled->placement->used[0];planned->used[1]=compiled->placement->used[1];
  planned->arena[0]=compiled->placement->range[0];planned->arena[1]=compiled->placement->range[1];
  compiled->page[planned->rows.count-1]=plan_page(compiled->placement,geometry->block,compiled->placement->table,0);
  return 0;
}
static int plan_program_adjacency(struct mesh_plan_compilation *compiled){
  const struct mesh_plan_analysis *analysis=compiled->result.planned.analysis;
  const uint32_t count=analysis->functions,nodes=analysis->nodes->count;
  for(uint32_t f=0;f<count;f++)compiled->places+=(uint64_t)__builtin_popcountll(compiled->step[f].flow->toward);
  const size_t places=(size_t)compiled->places;
  compiled->adjacency=malloc(((size_t)nodes+1+places)*sizeof *compiled->adjacency);
  compiled->toward=malloc((places+1)*sizeof *compiled->toward);
  compiled->place=malloc((places+1)*sizeof *compiled->place);
  compiled->rows_by=malloc(((size_t)nodes+1)*sizeof *compiled->rows_by);
  compiled->listed=malloc((places+1)*sizeof *compiled->listed);
  if(!compiled->adjacency || !compiled->toward || !compiled->place || !compiled->rows_by || !compiled->listed)
    return ENOMEM;
  for(uint32_t f=0,q=0;f<count;f++)
    for(uint64_t set=compiled->step[f].flow->toward;set;set&=set-1){
      compiled->toward[q]=(uint32_t)__builtin_ctzll(set);compiled->place[q++]=f;
    }
  mesh_graph_rows(nodes,places,plan_at,compiled->toward,compiled->rows_by,compiled->listed);
  for(uint32_t i=0;i<=nodes;i++)compiled->adjacency[i]=(uint32_t)compiled->rows_by[i];
  for(size_t q=0;q<places;q++)compiled->adjacency[nodes+1+q]=compiled->place[compiled->listed[q]];
  return 0;
}
static int plan_state(struct mesh_plan_compilation *compiled,uint32_t f){
  const struct mesh_plan_analysis *analysis=compiled->result.planned.analysis;
  const struct mesh_plan_geometry *m=analysis->geometry;
  const uint64_t program=analysis->program,block=plan_quantum(m);
  struct plan_step *p=compiled->step+f;
  const uint32_t u=p->flow->copies;
  const uint64_t base=0;
  if(!(compiled->state[f]=calloc(1,(size_t)p->words)))return ENOMEM;
  compiled->state_bytes[f]=p->words;
  p->base=base;
  struct mesh_state *state=(struct mesh_state *)compiled->state[f];
  state->copies=u;
  state->receives=(uint32_t)p->flow->receives;
  state->ranges=p->flow->consumed;
  state->releases=(uint32_t)p->flow->releases;
  state->phases=base+p->phases;
  state->counts=base+p->counts;
  state->completions=base+p->completions;
  state->supplied=base+p->supplied;
  state->transfers=base+p->publications;
  state->status=base+p->status;
  state->group=0;
  state->epoch=(uint64_t)f+1;
  state->serial=p->flow->view.id;state->sequencer=p->flow->view.sequencer;state->boot=p->flow->view.boot;
  state->hops=base+p->hops;
  state->entries=p->flow->entries*u;
  state->nodes=p->flow->spanned;
  state->sender=(uint32_t)__builtin_ctzll(program);
  state->outstanding=(uint32_t)p->flow->outstanding;
  state->room=p->room;
  state->submit_room=p->submit_room;
  state->deliveries=base+p->deliveries;
  state->submits=base+p->submits;
  state->waits=p->waits;
  state->tells=p->tells;
  state->syncs=base+p->syncs;
  state->calls=p->flow->calls;
  state->rows=base+p->rows;
  state->times=base+p->times;
  if(p->completed)memcpy(((char *)compiled->state[f])+p->times,p->completed,(size_t)p->flow->transfers*sizeof *p->completed);
  struct mesh_sync *sync=(struct mesh_sync *)((char *)state+state->syncs);
  for(uint32_t k=0;k<p->waits+p->tells;k++){
    sync[k].node=p->sync[k].node;sync[k].function=p->sync[k].function;sync[k].entry=p->sync[k].entry;
    sync[k].every=p->sync[k].every;sync[k].base=p->sync[k].base;
  }
  memcpy(((char *)compiled->state[f])+p->rows,p->wait_row,((size_t)p->flow->calls+1)*sizeof *p->wait_row);
  for(uint32_t i=0;i<p->flow->transfers;i++){
    const struct mesh_declared *t=p->flow->declared+i;
    const struct mesh_plan_demand *d=p->flow->demand+p->flow->of[i];
    struct mesh_publication *entry=(struct mesh_publication *)(void *)(((char *)compiled->state[f])+p->publications+
                                                                       (size_t)i*sizeof(struct mesh_publication));
    if(t->source==p->flow->self){
      entry->targets[0]=(struct mesh_target){.word=plan_supplied(p,d),.count=1,.stride=0};
      entry->state=1;
    }else if(t->destination==p->flow->self){
      entry->pages=d->received;
      entry->phases=plan_group(p,d,0);
      entry->count=plan_count(p,d,0);
      entry->state=1;
    }
    if(d->sending!=MESH_ABSENT){
      entry->sends=d->sends;
      entry->completion=plan_completion(p,d,0);
      entry->state=1;
    }
  }
  /* design/plan-to-bridge-change-request.md R2, R5; design/aot-compilation-change-request.md A9 */
  struct mesh_hop *hop=(struct mesh_hop *)((char *)state+state->hops);
  for(uint32_t c=0;c<u;c++)for(uint64_t j=0;j<p->flow->entries;j++){
    const struct mesh_plan_entry *e=p->entry[j].source;
    const struct mesh_plan_demand *d=p->flow->demand+e->demand;
    const uint64_t *cut=p->flow->cut+d->cut+e->instruction;
    uint64_t byte=cut[0];
    uint32_t frame;
    if(d->mine==MESH_ABSENT && d->root==MESH_ABSENT)
      frame=plan_page(compiled->placement,m->block,p->set_item+d->relay*u+c,byte/block);
    else{
      /* a root's copy t is block t: this function's copy c is of block c / calls (mesh_plan_local) */
      const struct mesh_section x=d->root!=MESH_ABSENT?p->root_result[d->root].section:p->result->value[d->mine].section;
      const uint32_t copy=d->root!=MESH_ABSENT?c/p->flow->calls%x.count:c%x.count;
      byte=p->flow->at[d->at+e->instruction];
      frame=compiled->page[x.first+(size_t)copy*x.stride+byte/block];
    }
    struct mesh_hop h={.page=frame,.node=e->node,.direction=e->direction,.field=MESH_ABSENT,
                       .bytes=(uint32_t)(cut[1]-cut[0]),.serial=p->flow->view.id,.set=e->set,
                       .offset=(uint32_t)(byte%block)};
    if(e->field==MESH_ABSENT)h.word=plan_supplied(p,d);
    else{
      h.word=plan_group(p,d,c)+(uint64_t)e->field/32*sizeof(uint64_t);
      h.field=2*(e->field%32);
      if(e->direction==MESH_RECEIVE){
        h.count=plan_count(p,d,c);
        h.pages=d->received;
      }
    }
    if(e->direction==MESH_SEND){h.count=plan_completion(p,d,c);h.pages=d->sends;}
    *hop++=h;
  }
  return 0;
}
int mesh_plan_program_compile(struct mesh_plan_analysis *analysis,struct mesh_plan_compilation **out){
  *out=NULL;
  struct mesh_plan_compilation *compiled=calloc(1,sizeof *compiled);
  if(!compiled)return ENOMEM;
  compiled->references=1;
  plan_source(&compiled->result,analysis);
  struct mesh_planned *planned=&compiled->result.planned;
  int status=plan_steps(analysis,planned,&compiled->step);
  compiled->root_item=malloc(((size_t)analysis->roots+1)*sizeof *compiled->root_item);
  compiled->placement=calloc(1,sizeof *compiled->placement);
  compiled->result.placement=compiled->placement;
  compiled->state=calloc((size_t)analysis->functions+1,sizeof *compiled->state);
  compiled->state_bytes=calloc((size_t)analysis->functions+1,sizeof *compiled->state_bytes);
  if(!compiled->root_item || !compiled->placement || !compiled->state || !compiled->state_bytes)status=ENOMEM;
  planned->horizon=analysis->horizon;
  if(!status)status=plan_graph(compiled->step,analysis,planned);
  if(!status)status=plan_sections(compiled);
  if(!status)status=plan_program_placement(compiled);
  if(!status)status=plan_program_adjacency(compiled);
  for(uint32_t f=0;!status && f<analysis->functions;f++)status=plan_state(compiled,f);
  if(!status)status=plan_extents(compiled->step,analysis->functions,planned);
  const struct plan_program program={analysis,analysis->root,compiled->root_item,planned->root};
  if(!status)status=plan_result(compiled->step,analysis->functions,&program,&compiled->result);
  compiled->result.quantum=plan_quantum(analysis->geometry);
  if(status)mesh_plan_compilation_free(compiled);else *out=compiled;
  return status;
}

int mesh_plan(struct mesh_ctx *context,uint64_t group,uint64_t program,const struct mesh_function *functions,uint32_t count,
              const uint32_t *calls,uint32_t ncalls,const uint32_t *orders,const uint32_t *causes,uint32_t norders,
              const uint32_t *chains,uint64_t deadline,struct mesh_plan_binding *binding){
  *binding=(struct mesh_plan_binding){0};
  if(!group || !program || (norders && (!orders || !causes)))return EINVAL;
  struct mesh_plan_analysis *a=NULL;
  struct mesh_plan_compilation *compiled=NULL;
  int status=plan_analysis_new(functions,count,calls,ncalls,NULL,NULL,0,NULL,chains,NULL,0,NULL,NULL,&a);
  if(status)return status;
  struct plan_environment *environment=malloc(sizeof *environment);
  if(!environment){mesh_plan_analysis_free(a);return ENOMEM;}
  a->environment=environment;a->nodes=&environment->nodes;a->geometry=&environment->geometry;
  a->program=program;a->orders=orders;a->causes=causes;a->norders=norders;
  status=mesh_plan_environment(context,&environment->nodes,&environment->geometry);
  if(!status)status=plan_horizon(a);
  if(!status)status=plan_analyze_network(a);
  if(!status)status=mesh_plan_program_compile(a,&compiled);
  if(!status)status=mesh_plan_program_install(context,group,compiled,deadline,binding);
  mesh_plan_compilation_free(compiled);mesh_plan_analysis_free(a);
  return status;
}

/* design/aot-compilation-change-request.md A2, A4 and A7 to A10, appendix items 2 and 6; mesh-collective.h
   mesh_plan_program: the one joint network plan of a program's instances, its orders the edges `ordered` of the calls'
   relation (MESH_ORDER_CONFLICT where a view hazard is among an edge's causes, else MESH_ORDER_FLOW) */
static int plan_program_source(const struct mesh_nodes *nodes,const struct mesh_plan_geometry *geometry,uint64_t program,const struct mesh_function *functions,
                      uint32_t count,const struct mesh_grouped *grouped,const uint32_t *calls,uint32_t ncalls,
                      const uint32_t *source_calls,uint32_t sources,const uint64_t *source_outstanding,
                      const struct mesh_graph_relation *order,
                      const uint32_t *ordered,uint32_t norders,const struct mesh_graph_closure *closure,
                      const uint32_t *chains,const struct mesh_value *roots,uint32_t nroots,const size_t *use_first,
                      const struct mesh_local_use *use,struct mesh_plan_analysis **out){
  if(!out)return EINVAL;
  *out=NULL;
  if(!nodes || !geometry || !program || !grouped || !source_calls || !sources || !source_outstanding)return EINVAL;
  int status=plan_analysis_new(functions,count,calls,ncalls,order,ordered,norders,closure,chains,roots,nroots,
                                use_first,use,out);
  if(status)return status;
  struct mesh_plan_analysis *a=*out;
  a->nodes=nodes;a->geometry=geometry;a->program=program;a->grouped=grouped;a->source_calls=source_calls;
  a->sources=sources;a->source_outstanding=source_outstanding;
  return 0;
}
int mesh_plan_program_analyze(const struct mesh_nodes *nodes,const struct mesh_plan_geometry *geometry,uint64_t program,
                      struct mesh_graph_buffer_result *bounds,const uint32_t *bound_function,const struct mesh_function *functions,
                      uint32_t count,const struct mesh_grouped *grouped,const uint32_t *calls,uint32_t ncalls,
                      const uint32_t *source_calls,uint32_t sources,const uint64_t *source_outstanding,
                      const struct mesh_graph_relation *order,
                      const uint32_t *ordered,uint32_t norders,const struct mesh_graph_closure *closure,
                      const uint32_t *chains,const struct mesh_value *roots,uint32_t nroots,const size_t *use_first,
                      const struct mesh_local_use *use,struct mesh_plan_analysis **out){
  if(!out)return EINVAL;
  *out=NULL;
  if(!bounds || (bounds->queries && !bound_function))return EINVAL;
  int status=plan_program_source(nodes,geometry,program,functions,count,grouped,calls,ncalls,source_calls,sources,
                                 source_outstanding,order,ordered,norders,closure,chains,roots,nroots,use_first,use,out);
  if(status)return status;
  struct mesh_plan_analysis *a=*out;
  status=plan_bound_result(a,bounds,bound_function);
  if(!status)status=plan_analyze_network(a);
  if(status){mesh_plan_analysis_free(a);*out=NULL;}
  return status;
}
int mesh_plan_program(struct mesh_ctx *context,uint64_t group,uint64_t program,const struct mesh_function *functions,
                      uint32_t count,const struct mesh_grouped *grouped,const uint32_t *calls,uint32_t ncalls,
                      const uint32_t *source_calls,uint32_t sources,const uint64_t *source_outstanding,
                      const struct mesh_graph_relation *order,
                      const uint32_t *ordered,uint32_t norders,const struct mesh_graph_closure *closure,
                      const uint32_t *chains,const struct mesh_value *roots,uint32_t nroots,const size_t *use_first,
                      const struct mesh_local_use *use,uint64_t deadline,struct mesh_plan_binding *out){
  *out=(struct mesh_plan_binding){0};
  struct plan_environment *environment=malloc(sizeof *environment);
  if(!environment)return ENOMEM;
  struct mesh_plan_analysis *analysis=NULL;
  struct mesh_plan_compilation *compiled=NULL;
  int status=mesh_plan_environment(context,&environment->nodes,&environment->geometry);
  if(!status)status=plan_program_source(&environment->nodes,&environment->geometry,program,functions,count,grouped,
      calls,ncalls,source_calls,sources,source_outstanding,order,ordered,norders,closure,chains,roots,nroots,use_first,use,&analysis);
  if(analysis)analysis->environment=environment;else free(environment);
  if(!status)status=plan_horizon(analysis);
  if(!status)status=plan_analyze_network(analysis);
  if(!status)status=mesh_plan_program_compile(analysis,&compiled);
  if(!status)status=mesh_plan_program_install(context,group,compiled,deadline,out);
  mesh_plan_compilation_free(compiled);mesh_plan_analysis_free(analysis);
  return status;
}

/* design/aot-compilation-change-request.md A1, A7; design/plan-to-bridge-change-request.md R4. */
struct plan_archive {
  unsigned char *bytes;
  size_t at,size,capacity;
  int read,status;
};
static void plan_archive_bytes(struct plan_archive *a,void *data,size_t size){
  if(a->status)return;
  if(size>SIZE_MAX-a->at){a->status=EOVERFLOW;return;}
  const size_t end=a->at+size;
  if(a->read){
    if(end>a->size){a->status=EINVAL;return;}
    if(size)memcpy(data,a->bytes+a->at,size);
  }else{
    if(end>a->capacity){
      const size_t capacity=end>SIZE_MAX/2?end:end*2;
      void *more=realloc(a->bytes,capacity);
      if(!more){a->status=ENOMEM;return;}
      a->bytes=more;a->capacity=capacity;
    }
    if(size)memcpy(a->bytes+a->at,data,size);
    a->size=end;
  }
  a->at=end;
}
/* the next 8-byte offset from the archive's start: zeros written, or skipped read */
static void plan_archive_align(struct plan_archive *a){
  unsigned char pad[8]={0};
  plan_archive_bytes(a,pad,(8-a->at%8)%8);
}
/* an array the loaded result borrows: written at an 8-byte offset, and read there as a view of the archive's bytes */
static void plan_archive_put(struct plan_archive *a,const void *data,size_t size){
  plan_archive_align(a);
  plan_archive_bytes(a,(void *)(uintptr_t)data,size);
}
static const void *plan_archive_view(struct plan_archive *a,uint64_t count,size_t width){
  plan_archive_align(a);
  if(a->status)return NULL;
  if(count>(a->size-a->at)/width){a->status=EINVAL;return NULL;}
  const void *at=a->bytes+a->at;
  a->at+=(size_t)count*width;
  return at;
}
/* design/compilation-composition-change-request.md items 3, 6.4 and 7: the compiled parts a function is planned for,
   one common archive record: struct plan_saved_actors, then host[parts], work[parts] and the
   parts' relation (mesh_graph_relation_save; its edges the function's) or, a function with none, edge[edges] */
struct plan_saved_actors { uint64_t parts,edges,relation; };
static void plan_actors_save(struct plan_archive *out,const struct mesh_function *source,const struct mesh_plan_actors *actors){
  const struct mesh_graph_relation *given=actors?actors->relation:source->relation;
  const uint32_t parts=actors?actors->parts:source->parts,edges=actors?actors->edges:source->edges;
  void *relation=NULL;
  size_t bytes=0;
  const int status=given?mesh_graph_relation_save(given,&relation,&bytes):0;
  if(status){if(!out->status)out->status=status;return;}
  const struct plan_saved_actors head={parts,given?0:edges,bytes};
  plan_archive_put(out,&head,sizeof head);
  plan_archive_put(out,actors?actors->host:source->host,(size_t)parts*sizeof(double));
  plan_archive_put(out,actors?actors->work:source->work,(size_t)parts*sizeof(double));
  if(relation)plan_archive_put(out,relation,bytes);
  else plan_archive_put(out,actors?actors->edge:source->edge,(size_t)edges*sizeof(struct mesh_graph_dependence));
  free(relation);
}
/* the record into f's parts, host, work, edge, edges and relation, views of the archive's bytes; a loaded relation
   (*relation, the caller's to release) holds its own reference to them */
static int plan_actors_load(struct plan_archive *in,struct plan_bytes *bytes,struct mesh_plan_actors *f,
                            struct mesh_graph_relation **relation,struct mesh_graph_context *analysis){
  *relation=NULL;
  const struct plan_saved_actors *head=plan_archive_view(in,1,sizeof *head);
  if(!head)return in->status;
  if(head->parts>UINT32_MAX || head->edges>UINT32_MAX || (head->relation && head->edges))return EINVAL;
  const double *host=plan_archive_view(in,head->parts,sizeof *host),*work=plan_archive_view(in,head->parts,sizeof *work);
  const struct mesh_graph_dependence *edge=NULL;
  uint64_t edges=head->edges;
  if(head->relation){
    const void *archived=plan_archive_view(in,head->relation,1);
    if(!archived)return in->status;
    __atomic_fetch_add(&bytes->references,1,__ATOMIC_RELAXED);
    /* the actors' relation (a quotient of its program's dispatch relation, whose views may be access relations), in
       the loading compilation's analysis context */
    const int status=mesh_graph_relation_load(analysis,archived,(size_t)head->relation,plan_bytes_release,bytes,relation);
    if(status){__atomic_fetch_sub(&bytes->references,1,__ATOMIC_RELAXED);return status;}
    const struct mesh_graph_relation_data *d=mesh_graph_relation_data(*relation);
    edge=d->edge;edges=d->edges;
  }else edge=plan_archive_view(in,edges,sizeof *edge);
  if(in->status)return in->status;
  if(edges>UINT32_MAX)return EINVAL;
  for(uint64_t e=0;e<edges;e++)
    if(edge[e].from>=2*head->parts || edge[e].to>=2*head->parts)return EINVAL;
  f->parts=(uint32_t)head->parts;f->host=host;f->work=work;f->edge=edge;f->edges=(uint32_t)edges;f->relation=*relation;
  return 0;
}

/* the optional section after the signature: struct mesh_planned's first (functions + 1), transfer (first[functions])
   and extent (extents) */
struct plan_saved_extents { uint64_t tag,extents; };
#define PLAN_SAVED_EXTENTS UINT64_C(0x31544e545848534d)
static const size_t plan_addresses[]={offsetof(struct mesh_state,phases),offsetof(struct mesh_state,counts),
  offsetof(struct mesh_state,supplied),offsetof(struct mesh_state,status),offsetof(struct mesh_state,completions),
  offsetof(struct mesh_state,hops),offsetof(struct mesh_state,deliveries),offsetof(struct mesh_state,submits),
  offsetof(struct mesh_state,syncs),offsetof(struct mesh_state,rows),offsetof(struct mesh_state,transfers),
  offsetof(struct mesh_state,times)};
static void plan_shift_state(struct mesh_state *s,uint64_t from,uint64_t to){
  for(size_t k=0;k<sizeof plan_addresses/sizeof *plan_addresses;k++){
    uint64_t *field=(uint64_t *)(void *)((char *)s+plan_addresses[k]);*field=*field-from+to;
  }
}
static void plan_shift_word(uint64_t *word,uint64_t from,uint64_t to){if(*word)*word=*word-from+to;}
/* a saved page of arena s (s below `arenas`: 1 where the saved arena is shared), at its offset from the saved arena's
   first, placed in the caller's destination[s] */
static uint32_t plan_page_relocate(const struct mesh_plan_geometry *geometry,const struct mesh_range *destination,uint32_t arenas,
                                uint32_t page,int *status){
  for(uint32_t s=0;s<arenas;s++)if(page>=geometry->arena[s].first && page-geometry->arena[s].first<geometry->arena[s].count){
    const uint32_t relative=page-geometry->arena[s].first;
    if(relative>=destination[s].count){*status=ESTALE;return 0;}
    return destination[s].first+relative;
  }
  *status=EINVAL;return 0;
}


/* the planned signature (mesh_graph_archive) */
static void plan_signature_put(struct plan_archive *out,const struct mesh_planned *planned,uint64_t bytes){
  void *signature=malloc((size_t)bytes);
  if(!signature && !out->status)out->status=ENOMEM;
  if(!out->status){size_t made=0;out->status=mesh_graph_archive(planned->signature,signature,(size_t)bytes,&made);}
  plan_archive_put(out,signature,(size_t)bytes);free(signature);
}
/* the planned first, transfer and extent (struct plan_saved_extents), where the planned result has them */
static void plan_extents_put(struct plan_archive *out,const struct mesh_planned *planned,uint32_t count){
  if(!planned->first)return;
  struct plan_saved_extents extents={PLAN_SAVED_EXTENTS,planned->extents};
  plan_archive_bytes(out,&extents,sizeof extents);
  plan_archive_put(out,planned->first,((size_t)count+1)*sizeof *planned->first);
  plan_archive_put(out,planned->transfer,planned->first[count]*sizeof *planned->transfer);
  plan_archive_put(out,planned->extent,planned->extents*sizeof *planned->extent);
}
static void plan_analysis_put(struct plan_archive *,const struct mesh_plan_analysis *);
static int plan_analysis_get(struct plan_archive *,struct plan_loaded *,const struct mesh_function *,uint32_t,
                             struct mesh_plan_analysis **);

/* the archived signature viewed in place (8-aligned, plan_signature_put) and read under the loaded bytes' owner: the
   restored signature (every array a view of it, mesh_graph_archive_read) and its scopes keep one reference of `bytes`
   (plan_bytes_release with their last holder), adopted on success */
static int plan_signature_get(struct plan_archive *input,uint64_t length,struct plan_bytes *bytes,
                              struct mesh_graph_signature *signature){
  const void *archived=plan_archive_view(input,length,1);
  if(!archived)return input->status;
  __atomic_fetch_add(&bytes->references,1,__ATOMIC_RELAXED);
  const int status=mesh_graph_archive_read(archived,(size_t)length,plan_bytes_release,bytes,signature);
  if(status)__atomic_fetch_sub(&bytes->references,1,__ATOMIC_RELAXED);
  return status;
}
static int plan_saved_span(uint64_t offset,uint64_t count,uint64_t width,uint64_t length){
  return offset<=length && count<=(length-offset)/width;
}
/* the compiled extent section: each function's span of first its actual transfers, each slice in the extents */
static int plan_extents_load(struct plan_archive *input,const struct mesh_function *functions,uint32_t count,
                             struct mesh_planned *planned){
  struct plan_saved_extents saved={0};
  plan_archive_bytes(input,&saved,sizeof saved);
  if(input->status || saved.tag!=PLAN_SAVED_EXTENTS ||
     !plan_saved_span(input->at,(uint64_t)count+1,sizeof *planned->first,input->size))return EINVAL;
  planned->functions=count;
  planned->first=(size_t *)(uintptr_t)plan_archive_view(input,(uint64_t)count+1,sizeof *planned->first);
  if(!planned->first || planned->first[0])return EINVAL;
  for(uint32_t f=0;f<count;f++)
    if(planned->first[f+1]<planned->first[f] || planned->first[f+1]-planned->first[f]!=functions[f].count)return EINVAL;
  const size_t transfers=planned->first[count];
  planned->transfer=(struct mesh_planned_transfer *)(uintptr_t)plan_archive_view(input,transfers,
                                                                                  sizeof *planned->transfer);
  planned->extent=(struct mesh_graph_access *)(uintptr_t)plan_archive_view(input,saved.extents,sizeof *planned->extent);
  if(input->status)return input->status;
  planned->extents=(size_t)saved.extents;
  for(size_t t=0;!input->status && t<transfers;t++){
    const struct mesh_planned_transfer x=planned->transfer[t];
    if(x.received>x.sending || x.sending>x.end || x.end>planned->extents)return EINVAL;
  }
  return input->status;
}
/* a saved plan state's addresses inside its `size` bytes, and its spans inside them */
static int plan_state_valid(const struct mesh_state *s,uint64_t size,uint32_t transfers){
  for(size_t a=0;a<sizeof plan_addresses/sizeof *plan_addresses;a++)
    if(*(const uint64_t *)(const void *)((const char *)s+plan_addresses[a])>size)return 0;
  return s->copies && plan_saved_span(s->transfers,transfers,sizeof(struct mesh_publication),size) &&
         plan_saved_span(s->syncs,(uint64_t)s->waits+s->tells,sizeof(struct mesh_sync),size) &&
         plan_saved_span(s->rows,(uint64_t)s->calls+1,sizeof(uint32_t),size) &&
         plan_saved_span(s->hops,s->entries,sizeof(struct mesh_hop),size) &&
         plan_saved_span(s->times,transfers,sizeof(double),size);
}
/* a saved plan state relocated to its page at `base`: its publications' and hops' words, its hops' pages
   (plan_page_relocate), its own addresses, its name (group, epoch) and the current view */
static void plan_state_place(const struct mesh_plan_geometry *geometry,const struct mesh_range *destination,uint32_t arenas,
                             struct mesh_state *s,uint32_t transfers,uint64_t base,const struct mesh_nodes *nodes,
                             uint64_t group,uint64_t epoch,int *status){
  struct mesh_publication *publication=(void *)((char *)s+s->transfers);
  for(uint32_t t=0;t<transfers;t++){
    plan_shift_word(&publication[t].phases,0,base);plan_shift_word(&publication[t].count,0,base);
    plan_shift_word(&publication[t].completion,0,base);plan_shift_word(&publication[t].targets[0].word,0,base);
    publication[t].state=publication[t].state?base:0;
  }
  struct mesh_hop *hop=(void *)((char *)s+s->hops);
  for(uint64_t h=0;h<s->entries;h++){
    plan_shift_word(&hop[h].word,0,base);plan_shift_word(&hop[h].count,0,base);
    hop[h].page=plan_page_relocate(geometry,destination,arenas,hop[h].page,status);hop[h].serial=nodes->serial;
  }
  plan_shift_state(s,0,base);
  s->group=group;s->epoch=epoch;s->serial=nodes->serial;s->sequencer=nodes->sequencer;s->boot=nodes->boot;
}
/* design/planner-composition.md */
void mesh_plan_binding_free(struct mesh_plan_binding *binding){
  if(!binding)return;
  mesh_plan_compilation_free(binding->compilation);
  free(binding->plan);free(binding->page);
  *binding=(struct mesh_plan_binding){0};
}
int mesh_plan_program_install(struct mesh_ctx *context,uint64_t group,const struct mesh_plan_compilation *compiled,
                              uint64_t deadline,struct mesh_plan_binding *binding){
  *binding=(struct mesh_plan_binding){0};
  binding->compilation=(struct mesh_plan_compilation *)(uintptr_t)compiled;
  mesh_plan_compilation_retain(binding->compilation);
  const struct mesh_local *local=&compiled->result;
  const struct mesh_planned *planned=&local->planned;
  const struct mesh_plan_analysis *analysis=planned->analysis;
  struct hdr *m=context->M;
  const struct mesh_nodes *nodes=analysis->nodes;
  const uint32_t count=local->functions,arenas=plan_arena_shared(planned->arena)?1:2;
  const struct mesh_plan_geometry *geometry=analysis->geometry;
  binding->plan=malloc(((size_t)count+1)*sizeof *binding->plan);
  binding->page=malloc((size_t)planned->rows.count*sizeof *binding->page);
  int status=binding->plan && binding->page?0:ENOMEM,prepared=0;
  const struct mesh_range live[2]={mesh_arena_range(m,0),mesh_arena_range(m,1)};
  if(!status && (m->pgsz!=geometry->pgsz || m->block!=geometry->block ||
                 (arenas==1)!=plan_arena_shared(live)))status=ESTALE;
  struct mesh_nodes live_nodes;
  if(!status)status=mesh_nodes_read(context,&live_nodes);
  if(!status && (nodes->count!=live_nodes.count || nodes->self!=live_nodes.self ||
                 memcmp(nodes->node,live_nodes.node,(size_t)nodes->count*sizeof *nodes->node)))status=ESTALE;
  if(!status)status=plan_prepare(context,deadline);
  if(!status){prepared=1;status=plan_reserve(m,(uint32_t)planned->rows.count,planned->used,arenas,deadline,&binding->rows,binding->arena);}
  if(status)goto done;
  for(uint32_t r=0;r<planned->rows.count;r++)binding->page[r]=plan_page_relocate(geometry,binding->arena,arenas,compiled->page[r],&status);
  for(uint32_t f=0;!status && f<count;f++){
    const uint32_t original=plan_page(compiled->placement,geometry->block,compiled->step[f].state,0),
                   page=plan_page_relocate(geometry,binding->arena,arenas,original,&status);
    const uint64_t base=m->data_off+(uint64_t)page*m->pgsz;
    binding->plan[f]=base;
    struct mesh_state *state=(void *)((char *)m+base);
    memcpy(state,compiled->state[f],(size_t)compiled->state_bytes[f]);
    plan_state_place(geometry,binding->arena,arenas,state,local->function[f].count,base,&live_nodes,group,(uint64_t)f+1,&status);
  }
  if(status)goto done;
  plan_bind_values(context,local->function,planned,count,binding->page,binding->rows.first);
  plan_bind_table(context,local->root,planned->root,local->roots,binding->page,binding->rows.first);
  const uint32_t table=binding->page[planned->rows.count-1];
  uint64_t *word=(void *)(mesh_at(m,table)+sizeof(struct mesh_algo));
  memcpy(word,binding->plan,(size_t)count*sizeof *word);
  memcpy(word+count,compiled->adjacency,((size_t)nodes->count+1+compiled->places)*sizeof(uint32_t));
  status=plan_publish(context,group,count,table,&live_nodes,binding);
done:
  if(prepared)plan_prepared(context);
  if(status)mesh_plan_binding_free(binding);
  return status;
}



/* design/joint-plan-results.md#archives */
struct plan_saved_local { uint64_t magic,bytes,arena,offsets,orders,horizon,values,relation,words,uses;
                          uint32_t functions,roots,ncalls,norders,chained,closed,used,padding; uint64_t quantum; };
struct plan_saved_local_function { uint64_t outstanding; uint32_t values,tasks; };
#define PLAN_SAVED_LOCAL UINT64_C(0x384c434f4c48534d)
/* the local sections at the archive's next 8-byte offset (mesh_local_save's whole archive, mesh_plan_compilation_save's first
   sections), their header's `bytes` the sections' length */
struct plan_saved_arrays { uint32_t count,dependences,tasks,befores,resources,padding; uint64_t pairs; };
/* a function's declared rows' grouping (mesh-collective.h struct mesh_grouped, mesh_declared_group's as its
   preparation retained it), after its weights: these scalars, then demand[demands], demand_of[transfers] and
   release_of[transfers] */
struct plan_saved_grouped { uint32_t transfers,demands,releases,nodes,self,padding; };
static void plan_arrays_put(struct plan_archive *out,const struct mesh_function *x){
  uint64_t pairs=0;
  for(uint32_t i=0;i<x->count;i++)pairs+=2*((uint64_t)x->declared[i].from.dims+x->declared[i].to.dims);
  const struct plan_saved_arrays saved={x->count,x->dependences,x->tasks,x->befores,x->resources,0,pairs};
  plan_archive_put(out,&saved,sizeof saved);
  plan_archive_put(out,x->declared,(size_t)x->count*sizeof *x->declared);
  plan_archive_align(out);
  for(uint32_t i=0;i<x->count;i++){
    const struct mesh_strided *view[2]={&x->declared[i].from,&x->declared[i].to};
    for(int k=0;k<2;k++)plan_archive_bytes(out,(void *)(uintptr_t)view[k]->pairs,2*(size_t)view[k]->dims*sizeof(uint64_t));
  }
  plan_archive_put(out,x->weights,(size_t)x->count*sizeof *x->weights);
  plan_archive_put(out,x->dependence,(size_t)x->dependences*sizeof *x->dependence);
  plan_archive_put(out,x->task,(size_t)x->tasks*sizeof *x->task);
  plan_archive_put(out,x->before,(size_t)x->befores*sizeof *x->before);
  if(x->tasks){
    plan_archive_put(out,x->sender,(size_t)x->count*sizeof *x->sender);
    plan_archive_put(out,x->receiver,(size_t)x->count*sizeof *x->receiver);
  }
}
static void plan_grouped_put(struct plan_archive *out,const struct mesh_grouped *g){
  const struct plan_saved_grouped grouped={g->transfers,g->demands,g->releases,g->nodes,g->self,0};
  plan_archive_put(out,&grouped,sizeof grouped);
  plan_archive_put(out,g->demand,(size_t)g->demands*sizeof *g->demand);
  plan_archive_put(out,g->demand_of,(size_t)g->transfers*sizeof *g->demand_of);
  plan_archive_put(out,g->release_of,(size_t)g->transfers*sizeof *g->release_of);
}
static int plan_arrays_get(struct plan_archive *in,struct mesh_function *x,const uint64_t **pairs,uint64_t *paired){
  const struct plan_saved_arrays *saved=plan_archive_view(in,1,sizeof *saved);
  if(!saved)return in->status;
  x->count=saved->count;x->dependences=saved->dependences;x->tasks=saved->tasks;x->befores=saved->befores;
  x->resources=saved->resources;
  x->declared=plan_archive_view(in,saved->count,sizeof *x->declared);
  *pairs=plan_archive_view(in,saved->pairs,sizeof **pairs);*paired=saved->pairs;
  x->weights=plan_archive_view(in,saved->count,sizeof *x->weights);
  x->dependence=plan_archive_view(in,saved->dependences,sizeof *x->dependence);
  x->task=plan_archive_view(in,saved->tasks,sizeof *x->task);
  x->before=plan_archive_view(in,saved->befores,sizeof *x->before);
  if(saved->tasks){
    x->sender=plan_archive_view(in,saved->count,sizeof *x->sender);
    x->receiver=plan_archive_view(in,saved->count,sizeof *x->receiver);
  }
  return in->status;
}
static int plan_grouped_get(struct plan_archive *in,uint32_t count,struct mesh_grouped *g){
  const struct plan_saved_grouped *grouped=plan_archive_view(in,1,sizeof *grouped);
  if(!grouped)return in->status;
  *g=(struct mesh_grouped){.transfers=grouped->transfers,.demands=grouped->demands,.releases=grouped->releases,
                           .nodes=grouped->nodes,.self=grouped->self};
  g->demand=(struct mesh_demand *)(uintptr_t)plan_archive_view(in,grouped->demands,sizeof *g->demand);
  g->demand_of=(uint32_t *)(uintptr_t)plan_archive_view(in,grouped->transfers,sizeof *g->demand_of);
  g->release_of=(uint32_t *)(uintptr_t)plan_archive_view(in,grouped->transfers,sizeof *g->release_of);
  if(in->status)return in->status;
  if(grouped->transfers!=count)return EINVAL;
  for(uint32_t t=0;t<grouped->transfers;t++)
    if(g->demand_of[t]>=grouped->demands || (g->release_of[t]!=MESH_ABSENT && g->release_of[t]>=grouped->releases))
      return EINVAL;
  return 0;
}
/* design/joint-plan-results.md#archives */
#define PLAN_FLOW_ARRAYS(X) \
  X(value_lag,p->givens) X(named,p->givens) \
  X(cut,p->demands?p->demand[p->demands-1].cut+p->demand[p->demands-1].instructions+1:0) \
  X(at,p->positions) X(set_far,p->sets) X(block_first,p->givens) \
  X(tree,p->made) X(parent,(uint64_t)p->made*p->n) X(run,p->made) \
  X(members,p->made) X(lead,p->made) X(comm,p->made) X(entry,p->entries) \
  X(child,(uint64_t)p->demands*p->n) X(bound,p->givens) X(value,p->values) \
  X(read,p->values?(uint64_t)p->value[p->values-1].read+p->value[p->values-1].reads:0) \
  X(here,p->local) X(edge,p->edges) X(acyclic_indegree,p->vertices) X(acyclic_ordinal,p->vertices) \
  X(lag_source,p->values?(uint64_t)p->value[p->values-1].source+1+p->value[p->values-1].reads:0) \
  X(lag_target,p->values) X(between,(uint64_t)p->local*p->local) X(farthest,p->local)
static void plan_flow_extra_put(struct plan_archive *,const struct mesh_plan_flow *);
static int plan_flow_extra_get(struct plan_archive *,struct mesh_plan_flow *,const struct mesh_plan_flow *);
#define PLAN_ANALYSIS_ARRAYS(X) \
  X(source_calls,a->ncalls) X(source_outstanding,a->sources) \
  X(orders,2*(uint64_t)a->norders) X(causes,a->norders) X(carried,a->roots)
static void plan_flow_put(struct plan_archive *out,const struct mesh_plan_flow *p){
  plan_archive_put(out,p,sizeof *p);
  plan_archive_align(out);
  for(uint32_t i=0;i<p->demands;i++){
    const struct mesh_plan_demand *d=p->demand+i;
    const size_t at=out->at;
    plan_archive_bytes(out,(void *)(uintptr_t)d,sizeof *d);
    const uintptr_t view=d->view?(uintptr_t)d->view-(uintptr_t)p->declared+1:0;
    if(!out->status)memcpy(out->bytes+at+offsetof(struct mesh_plan_demand,view),&view,sizeof view);
  }
#define PUT(field,count) if(p->field)plan_archive_put(out,p->field,(size_t)(count)*sizeof *p->field);
  PLAN_FLOW_ARRAYS(PUT)
#undef PUT
  plan_flow_extra_put(out,p);
  if(p->own.demand_of)plan_grouped_put(out,&p->own);
}
static int plan_flow_get(struct plan_archive *in,struct mesh_plan_flow *p,const struct mesh_function *source,
                         const struct mesh_plan_analysis *a,uint32_t f){
  const struct mesh_plan_flow *saved=plan_archive_view(in,1,sizeof *saved);
  if(!saved)return in->status;
  *p=*saved;p->demand=NULL;p->rows=NULL;p->adjacency=NULL;p->lag_graph=NULL;p->lags=NULL;p->covers=NULL;
  p->geometry=a->geometry;p->declared=saved->declared?source->declared:NULL;
  p->given=saved->given?source->value:NULL;p->weights=saved->weights?source->weights:NULL;
  p->dependence=saved->dependence?source->dependence:NULL;
  p->roots=saved->roots?a->root:NULL;p->carried=saved->carried?a->carried:NULL;
  p->of=NULL;p->own=(struct mesh_grouped){0};
  const struct mesh_plan_demand *d=plan_archive_view(in,p->demands,sizeof *d);
  if(in->status)return in->status;
  p->demand=malloc(((size_t)p->demands+1)*sizeof *p->demand);
  if(!p->demand)return ENOMEM;
  for(uint32_t i=0;i<p->demands;i++){
    p->demand[i]=d[i];p->demand[i].view=NULL;
    const uintptr_t at=(uintptr_t)d[i].view;
    if(!at)continue;
    const uintptr_t offset=at-1,field=offset%sizeof(struct mesh_declared);
    if(offset/sizeof(struct mesh_declared)>=source->count ||
       (field!=offsetof(struct mesh_declared,from) && field!=offsetof(struct mesh_declared,to)))return EINVAL;
    p->demand[i].view=(const void *)((const char *)source->declared+offset);
  }
#define GET(field,count) \
  p->field=saved->field?(void *)(uintptr_t)plan_archive_view(in,(count),sizeof *p->field):NULL; \
  if(in->status)return in->status;
  PLAN_FLOW_ARRAYS(GET)
#undef GET
  int extra=plan_flow_extra_get(in,p,saved);
  if(extra)return extra;
  if(saved->own.demand_of){
    const int status=plan_grouped_get(in,source->count,&p->own);
    if(status)return status;
  }
  if(saved->of)p->of=a->grouped?a->grouped[f].demand_of:p->own.demand_of;
  return 0;
}
#define PLAN_BOUND_ARRAYS(X) \
  X(time,b->vertices) X(host,b->actors) X(work,b->actors) X(edge,b->part_edges) \
  X(from,b->edges) X(to,b->edges) X(delay,b->edges) X(query,b->queries) X(bound,b->queries)
static void plan_bounds_put(struct plan_archive *out,const struct mesh_graph_buffer_result *b){
  plan_archive_put(out,b,sizeof *b);
#define PUT(field,count) plan_archive_put(out,b->field,(size_t)(count)*sizeof *b->field);
  PLAN_BOUND_ARRAYS(PUT)
#undef PUT
}
static int plan_bounds_get(struct plan_archive *in,struct plan_bytes *owner,struct mesh_graph_buffer_result **out){
  const struct mesh_graph_buffer_result *saved=plan_archive_view(in,1,sizeof *saved);
  if(!saved)return in->status;
  struct mesh_graph_buffer_result *b=malloc(sizeof *b);
  if(!b)return ENOMEM;
  *out=b;*b=*saved;b->references=1;b->release=plan_bytes_release;b->owner=owner;
  __atomic_fetch_add(&owner->references,1,__ATOMIC_RELAXED);
#define GET(field,count) b->field=(void *)(uintptr_t)plan_archive_view(in,(count),sizeof *b->field);
  PLAN_BOUND_ARRAYS(GET)
#undef GET
  return in->status;
}
#undef PLAN_BOUND_ARRAYS
static void plan_group_put(struct plan_archive *out,const struct mesh_graph_grouping *g){
  plan_archive_put(out,g,sizeof *g);
  plan_archive_put(out,g->first,((size_t)g->keys+1)*sizeof *g->first);
  plan_archive_put(out,g->order,g->items*sizeof *g->order);
  plan_archive_put(out,g->ordinal,g->items*sizeof *g->ordinal);
}
static int plan_group_get(struct plan_archive *in,const void *context,struct mesh_graph_grouping **out){
  const struct mesh_graph_grouping *saved=plan_archive_view(in,1,sizeof *saved);
  if(!saved)return in->status;
  struct mesh_graph_grouping *g=malloc(sizeof *g);
  if(!g)return ENOMEM;
  *out=g;*g=*saved;g->key=plan_at;g->context=context;
  g->first=(void *)(uintptr_t)plan_archive_view(in,(uint64_t)g->keys+1,sizeof *g->first);
  g->order=(void *)(uintptr_t)plan_archive_view(in,g->items,sizeof *g->order);
  g->ordinal=(void *)(uintptr_t)plan_archive_view(in,g->items,sizeof *g->ordinal);
  return in->status;
}
#define PLAN_COVER_ARRAYS(X) \
  X(held,r->held_count) X(first,(uint64_t)r->sets+1) X(member,r->listed) X(set_blocks,r->sets) \
  X(block_set,r->blocks) X(block_element,r->blocks) X(extra_set,r->extras) X(uncovered,r->values)
static void plan_flow_extra_put(struct plan_archive *out,const struct mesh_plan_flow *p){
  if(p->rows)plan_group_put(out,p->rows);
  for(uint32_t j=0;p->adjacency && j<p->made;j++){
    const struct mesh_graph_tree *tree=p->adjacency[j];
    struct mesh_graph_tree saved=*tree;
    saved.parent=(void *)(uintptr_t)(tree->parent-p->parent);
    plan_archive_put(out,&saved,sizeof saved);
    plan_group_put(out,tree->children);
  }
  if(p->lag_graph)plan_archive_put(out,p->lag_graph,sizeof *p->lag_graph);
  if(p->lags){
    plan_archive_put(out,p->lags,sizeof *p->lags);
    plan_archive_put(out,p->lags->value,(size_t)p->lags->sources*p->lags->targets*sizeof *p->lags->value);
  }
  if(p->covers){
    const struct mesh_graph_cover_result *r=p->covers;
    plan_archive_put(out,r,sizeof *r);
    plan_archive_put(out,p->touch,r->touches*sizeof *p->touch);
    plan_archive_put(out,p->relay,(size_t)r->extras*sizeof *p->relay);
    for(size_t j=0;j<r->touches;j++){
      const uint64_t index=(uint64_t)(r->order[j]-r->touch);
      plan_archive_put(out,&index,sizeof index);
    }
    for(size_t j=0;j<r->covers;j++){
      struct mesh_graph_cover cover=r->cover[j];
      cover.member=(void *)(uintptr_t)(cover.member-r->held);
      plan_archive_put(out,&cover,sizeof cover);
    }
#define PUT(field,count) plan_archive_put(out,r->field,(size_t)(count)*sizeof *r->field);
    PLAN_COVER_ARRAYS(PUT)
#undef PUT
  }
}
static int plan_flow_extra_get(struct plan_archive *in,struct mesh_plan_flow *p,const struct mesh_plan_flow *saved){
  int status=0;
  if(saved->rows){
    if((status=plan_group_get(in,p->edge,&p->rows)))return status;
    p->rows->key=plan_edge_tail;
  }
  if(saved->adjacency){
    p->adjacency=calloc((size_t)p->made+1,sizeof *p->adjacency);
    if(!p->adjacency)return ENOMEM;
    for(uint32_t j=0;j<p->made;j++){
      const struct mesh_graph_tree *view=plan_archive_view(in,1,sizeof *view);
      if(!view)return in->status;
      struct mesh_graph_tree *tree=p->adjacency[j]=malloc(sizeof *tree);
      if(!tree)return ENOMEM;
      *tree=*view;tree->children=NULL;
      tree->parent=p->parent+(uintptr_t)view->parent;
      if((status=plan_group_get(in,tree,&tree->children)))return status;
      tree->children->key=mesh_graph_tree_parent;
    }
  }
  if(saved->lag_graph){
    const struct mesh_graph_lag_graph *g=plan_archive_view(in,1,sizeof *g);
    if(!g)return in->status;
    p->lag_graph=malloc(sizeof *p->lag_graph);
    if(!p->lag_graph)return ENOMEM;
    *p->lag_graph=(struct mesh_graph_lag_graph){g->vertices,p->rows->first,plan_head,plan_lag,p};
  }
  if(saved->lags){
    const struct mesh_graph_lag_matrix *m=plan_archive_view(in,1,sizeof *m);
    if(!m)return in->status;
    p->lags=malloc(sizeof *p->lags);
    if(!p->lags)return ENOMEM;
    *p->lags=*m;p->lags->graph=p->lag_graph;p->lags->source=p->lag_source;p->lags->target=p->lag_target;
    p->lags->value=(void *)(uintptr_t)plan_archive_view(in,(uint64_t)m->sources*m->targets,sizeof *m->value);
  }
  if(saved->covers){
    const struct mesh_graph_cover_result *view=plan_archive_view(in,1,sizeof *view);
    if(!view)return in->status;
    struct mesh_graph_cover_result *r=p->covers=malloc(sizeof *r);
    if(!r)return ENOMEM;
    *r=*view;r->order=NULL;r->cover=NULL;
    p->touch=(void *)(uintptr_t)plan_archive_view(in,r->touches,sizeof *p->touch);
    p->relay=(void *)(uintptr_t)plan_archive_view(in,r->extras,sizeof *p->relay);
    const uint64_t *order=plan_archive_view(in,r->touches,sizeof *order);
    const struct mesh_graph_cover *cover=plan_archive_view(in,r->covers,sizeof *cover);
#define GET(field,count) r->field=(void *)(uintptr_t)plan_archive_view(in,(count),sizeof *r->field);
    PLAN_COVER_ARRAYS(GET)
#undef GET
    if(in->status)return in->status;
    r->order=malloc((r->touches+1)*sizeof *r->order);r->cover=malloc((r->covers+1)*sizeof *r->cover);
    if(!r->order || !r->cover)return ENOMEM;
    r->block_first=p->block_first;r->bytes=plan_cover_bytes;r->context=p;r->touch=p->touch;
    r->extra_member=plan_relay_member;r->extra_blocks=plan_relay_blocks;r->extra_context=p;
    for(size_t j=0;j<r->touches;j++)r->order[j]=p->touch+order[j];
    for(size_t j=0;j<r->covers;j++){
      r->cover[j]=cover[j];r->cover[j].member=r->held+(uintptr_t)cover[j].member;
    }
    p->set_first=r->first;p->member=r->member;p->set_blocks=r->set_blocks;
    p->block_set=r->block_set;p->block_element=r->block_element;p->uncovered=r->uncovered;
  }
  return in->status;
}
static void plan_flow_loaded_free(struct mesh_plan_flow *p){
  free(p->rows);free(p->lag_graph);free(p->lags);
  for(uint32_t j=0;p->adjacency && j<p->made;j++)if(p->adjacency[j]){
    free(p->adjacency[j]->children);free(p->adjacency[j]);
  }
  free(p->adjacency);
  if(p->covers){free(p->covers->order);free(p->covers->cover);free(p->covers);}
}
#undef PLAN_COVER_ARRAYS
static void plan_analysis_put(struct plan_archive *out,const struct mesh_plan_analysis *a){
  if(!a){if(!out->status)out->status=EINVAL;return;}
  plan_archive_put(out,a,sizeof *a);
  if(a->nodes)plan_archive_put(out,a->nodes,sizeof *a->nodes);
  if(a->geometry)plan_archive_put(out,a->geometry,sizeof *a->geometry);
#define PUT(field,count) if(a->field)plan_archive_put(out,a->field,(size_t)(count)*sizeof *a->field);
  PLAN_ANALYSIS_ARRAYS(PUT)
#undef PUT
  if(a->bounds){
    plan_bounds_put(out,a->bounds);
    plan_archive_put(out,a->actor_function,(size_t)a->bounds->queries*sizeof *a->actor_function);
    plan_archive_put(out,a->function_lag,(size_t)a->functions*sizeof *a->function_lag);
  }
  if(a->call_group)plan_group_put(out,a->call_group);
  if(a->source_group)plan_group_put(out,a->source_group);
  if(a->chain_pairs){
    plan_archive_put(out,a->chain_pairs,sizeof *a->chain_pairs);
    plan_archive_put(out,a->chain_pairs->edge,a->chain_pairs->edges*sizeof *a->chain_pairs->edge);
  }
  if(a->grouped)for(uint32_t f=0;f<a->functions;f++)plan_grouped_put(out,a->grouped+f);
  for(uint32_t f=0;f<a->functions;f++)plan_flow_put(out,a->flow+f);
}
static int plan_analysis_get(struct plan_archive *in,struct plan_loaded *loaded,
                            const struct mesh_function *functions,uint32_t count,struct mesh_plan_analysis **out){
  const struct mesh_plan_analysis *saved=plan_archive_view(in,1,sizeof *saved);
  if(!saved)return in->status;
  if(saved->functions!=count)return EINVAL;
  struct mesh_plan_analysis *a=malloc(sizeof *a);
  if(!a)return ENOMEM;
  *a=*saved;a->references=1;a->owned=loaded;a->environment=NULL;a->function=functions;
  a->nodes=saved->nodes?plan_archive_view(in,1,sizeof *a->nodes):NULL;
  a->geometry=saved->geometry?plan_archive_view(in,1,sizeof *a->geometry):NULL;
  a->calls=NULL;a->chains=NULL;a->ordered=NULL;a->relation=NULL;a->closure=NULL;a->root=NULL;
  a->use_first=NULL;a->use=NULL;a->grouped=NULL;a->call_group=NULL;a->source_group=NULL;a->chain_pairs=NULL;a->bounds=NULL;
  a->flow=calloc((size_t)count+1,sizeof *a->flow);
  __atomic_fetch_add(&loaded->references,1,__ATOMIC_RELAXED);
  *out=a;
  if(!a->flow)return ENOMEM;
#define GET(field,count) \
  a->field=saved->field?(void *)(uintptr_t)plan_archive_view(in,(count),sizeof *a->field):NULL; \
  if(in->status)return in->status;
  PLAN_ANALYSIS_ARRAYS(GET)
#undef GET
  int status=0;
  if(saved->bounds){
    status=plan_bounds_get(in,loaded->bytes,&a->bounds);
    if(status)return status;
    a->actor_function=(void *)(uintptr_t)plan_archive_view(in,a->bounds->queries,sizeof *a->actor_function);
    a->function_lag=(void *)(uintptr_t)plan_archive_view(in,a->functions,sizeof *a->function_lag);
    a->bound_host=saved->bound_host?(double *)(uintptr_t)a->bounds->host:NULL;
    a->bound_work=saved->bound_work?(double *)(uintptr_t)a->bounds->work:NULL;
    a->bound_edge=saved->bound_edge?(struct mesh_graph_dependence *)(uintptr_t)a->bounds->edge:NULL;
    a->bound_function=saved->bound_function?(uint32_t *)(uintptr_t)a->actor_function:NULL;
  }
  if(saved->call_group)status=plan_group_get(in,NULL,&a->call_group);
  if(!status && saved->source_group)status=plan_group_get(in,a->source_calls,&a->source_group);
  if(status)return status;
  if(saved->chain_pairs){
    const struct mesh_graph_chain_result *chain=plan_archive_view(in,1,sizeof *chain);
    if(!chain)return in->status;
    a->chain_pairs=malloc(sizeof *a->chain_pairs);
    if(!a->chain_pairs)return ENOMEM;
    *a->chain_pairs=*chain;a->chain_pairs->chain=NULL;
    a->chain_pairs->edge=(void *)(uintptr_t)plan_archive_view(in,chain->edges,sizeof *chain->edge);
  }
  if(saved->grouped){
    loaded->grouped=calloc((size_t)count+1,sizeof *loaded->grouped);
    if(!loaded->grouped)return ENOMEM;
    a->grouped=loaded->grouped;
    for(uint32_t f=0;f<count;f++){
      const int status=plan_grouped_get(in,functions[f].count,loaded->grouped+f);
      if(status)return status;
    }
  }
  for(uint32_t f=0;f<count;f++){
    const int status=plan_flow_get(in,a->flow+f,functions+f,a,f);
    if(status)return status;
  }
  return 0;
}
#undef PLAN_FLOW_ARRAYS
#undef PLAN_ANALYSIS_ARRAYS
#define PLAN_PLACEMENT_ARRAYS(X) \
  X(item,p->items) X(count,p->items) X(chain,p->elements) X(at,p->allocations) \
  X(size,p->allocations) X(start,p->allocations) X(end,p->allocations) X(offset,p->allocations) \
  X(function,p->allocations) X(index,p->allocations) X(copy,p->allocations) X(operations,functions)
static void plan_placement_put(struct plan_archive *out,const struct plan_placement *p,uint32_t functions){
  plan_archive_put(out,p,sizeof *p);
#define PUT(field,count) plan_archive_put(out,p->field,(size_t)(count)*sizeof *p->field);
  PLAN_PLACEMENT_ARRAYS(PUT)
#undef PUT
}
static int plan_placement_get(struct plan_archive *in,uint32_t functions,struct plan_placement **out){
  const struct plan_placement *saved=plan_archive_view(in,1,sizeof *saved);
  if(!saved)return in->status;
  struct plan_placement *p=malloc(sizeof *p);
  if(!p)return ENOMEM;
  *out=p;*p=*saved;
#define GET(field,count) p->field=(void *)(uintptr_t)plan_archive_view(in,(count),sizeof *p->field);
  PLAN_PLACEMENT_ARRAYS(GET)
#undef GET
  return in->status;
}
static int plan_local_put(struct plan_archive *out,const struct mesh_local *local){
  if(!local || !local->function || !local->value_first || !local->calls ||
     (local->roots && !local->root))return EINVAL;
  void *relation=NULL;
  size_t relation_bytes=0;
  const int saved=local->relation?mesh_graph_relation_save(local->relation,&relation,&relation_bytes):0;
  if(saved)return saved;
  const struct mesh_graph_closure *c=local->closure;
  struct plan_saved_local head={.magic=PLAN_SAVED_LOCAL,.arena=local->bytes,.offsets=local->offsets,
    .orders=local->planned.orders,.horizon=local->planned.horizon,.values=local->values,.relation=relation_bytes,
    .words=c?c->words:0,.uses=local->use_first?local->use_first[local->roots]:0,.used=!!local->use_first,.functions=local->functions,.roots=local->roots,
    .ncalls=local->ncalls,.norders=local->norders,.chained=!!local->chains,.closed=!!c,.quantum=local->quantum};
  plan_archive_align(out);
  const size_t start=out->at;
  plan_archive_bytes(out,&head,sizeof head);
  for(uint32_t f=0;f<local->functions;f++){
    const struct mesh_function *x=local->function+f;
    const struct plan_saved_local_function saved_function={x->outstanding,x->values,x->tasks};
    plan_archive_put(out,&saved_function,sizeof saved_function);
    plan_archive_put(out,x->value,(size_t)x->values*sizeof *x->value);
    const struct mesh_plan_function *result=local->planned.result+f;
    plan_archive_put(out,result->value,(size_t)x->values*sizeof *result->value);
    plan_archive_put(out,result->completion,(size_t)x->tasks*sizeof *result->completion);
    plan_archive_put(out,result->order,(size_t)x->tasks*sizeof *result->order);
    plan_actors_save(out,x,local->planned.result[f].actors);
    plan_arrays_put(out,x);
  }
  plan_archive_put(out,local->root,(size_t)local->roots*sizeof *local->root);
  plan_archive_put(out,local->planned.root,(size_t)local->roots*sizeof *local->planned.root);
  plan_archive_put(out,local->value_first,((size_t)local->functions+1)*sizeof *local->value_first);
  plan_archive_put(out,local->calls,(size_t)local->ncalls*sizeof *local->calls);
  if(local->chains)plan_archive_put(out,local->chains,(size_t)local->ncalls*sizeof *local->chains);
  plan_archive_put(out,local->ordered,(size_t)local->norders*sizeof *local->ordered);
  if(relation)plan_archive_put(out,relation,relation_bytes);
  if(c)plan_archive_put(out,c->reached,(size_t)c->vertices*c->words*sizeof *c->reached);
  if(local->use_first)plan_archive_put(out,local->use_first,((size_t)local->roots+1)*sizeof *local->use_first);
  plan_archive_put(out,local->use,(size_t)head.uses*sizeof *local->use);
  plan_archive_put(out,local->planned.call,(size_t)local->ncalls*sizeof *local->planned.call);
  plan_archive_put(out,local->planned.order,local->planned.orders*sizeof *local->planned.order);
  plan_archive_put(out,local->planned.cause,local->planned.orders*sizeof *local->planned.cause);
  plan_archive_put(out,local->first,((size_t)head.values+1)*sizeof *local->first);
  plan_archive_put(out,local->offset,local->offsets*sizeof *local->offset);
  plan_analysis_put(out,local->planned.analysis);
  plan_placement_put(out,local->placement,local->functions);
  free(relation);
  if(local->planned.calls!=local->ncalls && !out->status)out->status=EINVAL;
  if(!out->status){head.bytes=out->size-start;memcpy(out->bytes+start,&head,sizeof head);}
  return out->status;
}
int mesh_local_save(const struct mesh_local *local,void **bytes,size_t *length){
  *bytes=NULL;*length=0;
  struct plan_archive out={0};
  const int status=plan_local_put(&out,local);
  if(status){free(out.bytes);return status;}
  *bytes=out.bytes;*length=out.size;return 0;
}
/* every planned order inside its cause's documented domain (mesh-collective.h struct mesh_planned): MESH_PLANNED_PART
   its pair call's part vertices, MESH_ORDER_FLOW, MESH_ORDER_CONFLICT and MESH_PLANNED_CHAIN the call ordinals,
   MESH_PLANNED_PROGRAM, _SUBMIT, _OUTSTANDING and _TRANSFER the signature's tasks (`tasks`: 0 for a local result, which
   has no signature) */
static int plan_orders_valid(const struct mesh_planned *planned,const uint32_t *calls,uint32_t ncalls,
                             const struct mesh_function *function,uint64_t tasks){
  for(size_t e=0;e<planned->orders;e++){
    const struct mesh_planned_cause c=planned->cause[e];
    uint64_t bound=0;
    if(c.kind==MESH_PLANNED_PART)bound=c.pair<ncalls?2*(uint64_t)function[calls[c.pair]].parts:0;
    else if(c.kind==MESH_ORDER_FLOW || c.kind==MESH_ORDER_CONFLICT || c.kind==MESH_PLANNED_CHAIN)bound=ncalls;
    else if(c.kind==MESH_PLANNED_PROGRAM || c.kind==MESH_PLANNED_SUBMIT || c.kind==MESH_PLANNED_OUTSTANDING ||
            c.kind==MESH_PLANNED_TRANSFER)bound=tasks;
    if(planned->order[e].from>=bound || planned->order[e].to>=bound)return EINVAL;
  }
  return 0;
}
/* the local sections from the archive's next 8-byte offset (mesh_local_load's whole archive, mesh_plan_compilation_load's first
   sections), ending `bytes` after it; the loaded owner's byte owner not yet adopted (its caller's).  The planned orders
   are validated in their domains here for a local result; a network result's (`network`) by mesh_plan_compilation_load once its
   signature is read */
static int plan_local_get(struct plan_archive *input,struct mesh_local *local,int network,
                          struct mesh_graph_context *analysis){
  *local=(struct mesh_local){0};
  plan_archive_align(input);
  const size_t start=input->at;
  struct plan_saved_local head={0};
  plan_archive_bytes(input,&head,sizeof head);
  if(input->status || head.magic!=PLAN_SAVED_LOCAL || head.bytes>input->size-start || !head.functions ||
     !head.ncalls || head.functions==UINT32_MAX || head.values==UINT64_MAX ||
     (head.closed && head.words!=((uint64_t)head.ncalls+63)/64))return EINVAL;
  struct plan_loaded *loaded=plan_loaded_new(head.functions+1);
  if(!loaded)return ENOMEM;
  local->planned.owned=loaded;
  loaded->function=calloc((size_t)head.functions+1,sizeof *loaded->function);
  loaded->actors=calloc((size_t)head.functions+1,sizeof *loaded->actors);
  local->planned.result=calloc((size_t)head.functions+1,sizeof *local->planned.result);
  local->planned.results=head.functions;local->planned.roots=head.roots;
  const uint64_t **pairs=calloc((size_t)head.functions+1,sizeof *pairs);
  uint64_t *paired=calloc((size_t)head.functions+1,sizeof *paired);
  loaded->closure=head.closed?calloc(1,sizeof *loaded->closure):NULL;
  int status=loaded->function && loaded->actors && local->planned.result && pairs && paired &&
             (!head.closed || loaded->closure)?0:ENOMEM;
  for(uint32_t f=0;!status && f<head.functions;f++){
    struct mesh_function *x=loaded->function+f;
    const struct plan_saved_local_function *saved=plan_archive_view(input,1,sizeof *saved);
    const struct mesh_value *value=saved?plan_archive_view(input,saved->values,sizeof *value):NULL;
    struct mesh_plan_function *result=local->planned.result+f;
    result->value=saved?(struct mesh_plan_value *)(uintptr_t)plan_archive_view(input,saved->values,sizeof *result->value):NULL;
    result->completion=saved?(double *)(uintptr_t)plan_archive_view(input,saved->tasks,sizeof *result->completion):NULL;
    result->order=saved?(uint32_t *)(uintptr_t)plan_archive_view(input,saved->tasks,sizeof *result->order):NULL;
    status=input->status;
    if(!status)status=plan_actors_load(input,loaded->bytes,loaded->actors+f,loaded->relation+f,analysis);
    if(!status){
      const struct mesh_plan_actors *a=loaded->actors+f;
      x->value=(struct mesh_value *)(uintptr_t)value;x->values=saved->values;x->tasks=saved->tasks;
      x->outstanding=saved->outstanding;result->actors=a;
      x->host=a->host;x->work=a->work;x->edge=a->edge;x->parts=a->parts;x->edges=a->edges;x->relation=a->relation;
      status=plan_arrays_get(input,x,pairs+f,paired+f);
      if(!status && x->tasks!=saved->tasks)status=EINVAL;
    }
  }
  size_t declared=0;
  for(uint32_t f=0;!status && f<head.functions;f++)declared+=loaded->function[f].count;
  if(!status && !(loaded->declared=malloc((declared+1)*sizeof *loaded->declared)))status=ENOMEM;
  size_t first_declared=0;
  for(uint32_t f=0;!status && f<head.functions;f++){
    struct mesh_function *x=loaded->function+f;
    struct mesh_declared *d=loaded->declared+first_declared;first_declared+=x->count;
    uint64_t used=0;
    for(uint32_t i=0;!status && i<x->count;i++){
      d[i]=x->declared[i];
      struct mesh_strided *view[2]={&d[i].from,&d[i].to};
      for(int k=0;k<2;k++){
        const uint64_t words=2*(uint64_t)view[k]->dims;
        if(words>paired[f]-used){status=EINVAL;break;}
        view[k]->pairs=pairs[f]+used;used+=words;
      }
    }
    if(!status && used!=paired[f])status=EINVAL;
    x->declared=d;
  }
  free(pairs);free(paired);
  const struct mesh_value *root=plan_archive_view(input,head.roots,sizeof *root);
  local->planned.root=(struct mesh_plan_value *)(uintptr_t)plan_archive_view(input,head.roots,sizeof *local->planned.root);
  const size_t *value_first=plan_archive_view(input,(uint64_t)head.functions+1,sizeof *value_first);
  const uint32_t *calls=plan_archive_view(input,head.ncalls,sizeof *calls);
  const uint32_t *chains=head.chained?plan_archive_view(input,head.ncalls,sizeof *chains):NULL;
  const uint32_t *ordered=plan_archive_view(input,head.norders,sizeof *ordered);
  struct mesh_graph_relation **relation=loaded->relation+head.functions;
  if(!status && head.relation){
    const void *archived=plan_archive_view(input,head.relation,1);
    status=archived?0:input->status;
    if(!status){
      __atomic_fetch_add(&loaded->bytes->references,1,__ATOMIC_RELAXED);
      status=mesh_graph_relation_load(analysis,archived,(size_t)head.relation,plan_bytes_release,loaded->bytes,relation);
      if(status)__atomic_fetch_sub(&loaded->bytes->references,1,__ATOMIC_RELAXED);
    }
  }
  const uint64_t *reached=head.closed?plan_archive_view(input,(uint64_t)head.ncalls*head.words,sizeof *reached):NULL;
  const size_t *use_first=head.used?plan_archive_view(input,(uint64_t)head.roots+1,sizeof *use_first):NULL;
  const struct mesh_local_use *use=plan_archive_view(input,head.uses,sizeof *use);
  const struct mesh_planned_call *call=plan_archive_view(input,head.ncalls,sizeof *call);
  const struct mesh_graph_dependence *order=plan_archive_view(input,head.orders,sizeof *order);
  const struct mesh_planned_cause *cause=plan_archive_view(input,head.orders,sizeof *cause);
  const size_t *first=plan_archive_view(input,(uint64_t)head.values+1,sizeof *first);
  const uint64_t *offset=plan_archive_view(input,head.offsets,sizeof *offset);
  if(!status)status=input->status;
  if(!status && (first[0] || first[head.values]!=head.offsets || value_first[0] ||
                 value_first[head.functions]>head.values || head.values-value_first[head.functions]!=head.roots ||
                 (use_first?(use_first[0] || use_first[head.roots]!=head.uses):(head.roots || head.uses))))status=EINVAL;
  for(uint64_t v=0;!status && v<head.values;v++)if(first[v+1]<first[v])status=EINVAL;
  for(uint32_t f=0;!status && f<head.functions;f++)
    if(value_first[f+1]<value_first[f] || value_first[f+1]-value_first[f]!=loaded->function[f].values)status=EINVAL;
  for(uint32_t j=0;!status && j<head.ncalls;j++){
    const uint32_t h=chains?chains[j]:j;
    if(calls[j]>=head.functions || call[j].function!=calls[j] || h>j || (chains && chains[h]!=h))status=EINVAL;
  }
  for(uint32_t r=0;!status && r<head.roots;r++){
    if(use_first[r+1]<use_first[r] || use_first[r+1]>head.uses)status=EINVAL;
    for(size_t k=use_first[r];!status && k<use_first[r+1];k++)
      if(use[k].call>=head.ncalls || use[k].value>=loaded->function[calls[use[k].call]].values)status=EINVAL;
  }
  const struct mesh_graph_relation_data *d=!status && *relation?mesh_graph_relation_data(*relation):NULL;
  for(uint32_t q=0;!status && q<head.norders;q++)
    if(!d || ordered[q]>=d->edges || d->edge[ordered[q]].to>=head.ncalls)status=EINVAL;
  if(!status && !network)
    status=plan_orders_valid(&(struct mesh_planned){.order=(struct mesh_graph_dependence *)(uintptr_t)order,
                                                    .cause=(struct mesh_planned_cause *)(uintptr_t)cause,
                                                    .orders=(size_t)head.orders},calls,head.ncalls,loaded->function,0);
  if(status){mesh_local_free(local);return status;}
  if(loaded->closure)*loaded->closure=(struct mesh_graph_closure){head.ncalls,(size_t)head.words,
                                                                  (uint64_t *)(uintptr_t)reached};
  local->bytes=head.arena;local->values=(size_t)head.values;local->offsets=(size_t)head.offsets;local->quantum=head.quantum;
  local->first=(size_t *)(uintptr_t)first;local->offset=(uint64_t *)(uintptr_t)offset;
  local->planned.call=(struct mesh_planned_call *)(uintptr_t)call;local->planned.calls=head.ncalls;
  local->planned.order=(struct mesh_graph_dependence *)(uintptr_t)order;
  local->planned.cause=(struct mesh_planned_cause *)(uintptr_t)cause;
  local->planned.orders=(size_t)head.orders;local->planned.horizon=head.horizon;
  local->functions=head.functions;local->function=loaded->function;local->roots=head.roots;
  local->root=(struct mesh_value *)(uintptr_t)root;local->value_first=(size_t *)(uintptr_t)value_first;
  local->calls=calls;local->chains=chains;local->ordered=ordered;local->ncalls=head.ncalls;local->norders=head.norders;
  local->relation=*relation;local->closure=loaded->closure;local->use_first=use_first;local->use=use;
  status=plan_analysis_get(input,loaded,local->function,local->functions,&local->planned.analysis);
  if(!status)status=plan_placement_get(input,local->functions,&local->placement);
  if(!status && (input->at!=start+head.bytes || local->planned.analysis->ncalls!=head.ncalls ||
                 local->planned.analysis->roots!=head.roots || (!local->planned.analysis->orders && local->planned.analysis->norders!=head.norders)))status=EINVAL;
  if(status){mesh_local_free(local);return status;}
  struct mesh_plan_analysis *a=local->planned.analysis;
  a->calls=calls;a->chains=chains;a->ordered=ordered;a->relation=*relation;a->closure=loaded->closure;
  a->call_group->context=calls;if(a->chain_pairs)a->chain_pairs->chain=chains;
  a->root=root;a->use_first=use_first;a->use=use;
  for(uint32_t f=0;f<a->functions;f++)a->flow[f].roots=root;
  local->grouped=a->grouped;local->source_calls=a->source_calls;local->source_outstanding=a->source_outstanding;
  local->sources=a->sources;local->row=a->call_group->first;local->order=a->call_group->order;
  local->source_row=(a->source_group?a->source_group->first:NULL);local->source_order=(a->source_group?a->source_group->order:NULL);
  return 0;
}
/* every array bounded by the archive's bytes, each a view of them, the value tables included; the Function[]
   descriptors and the closure header the loaded result's own (plan_loaded) */
int mesh_local_load(struct mesh_graph_context *analysis,const void *bytes,size_t length,void (*release)(void *),
                    void *owner,struct mesh_local *local){
  *local=(struct mesh_local){0};
  if((uintptr_t)bytes%8)return EINVAL;
  struct plan_archive input={.bytes=(void *)bytes,.size=length,.read=1};
  int status=plan_local_get(&input,local,0,analysis);
  if(!status && input.at!=input.size)status=EINVAL;
  if(status){mesh_local_free(local);return status;}
  struct plan_loaded *loaded=local->planned.owned;
  loaded->bytes->release=release;loaded->bytes->owner=owner;
  return 0;
}

#define PLAN_SCHEDULE_ARRAYS(X) \
  X(reader,p->flow->demands) X(consumer,p->flow->transfers) X(w,s->instructions) X(cost,s->edges) \
  X(rate,2*(uint64_t)p->flow->pairs) X(completion,s->instructions) X(priority,s->instructions) \
  X(resource,s->instructions) X(from,s->edges) X(to,s->edges) X(link,s->entries) X(communication,s->entries) \
  X(before,s->dependencies) X(after,s->dependencies) X(place,s->instructions) X(route,s->edges+1)
#define PLAN_GRAPH_ARRAYS(X) \
  X(actor,g->analysis->ncalls) X(later,g->analysis->ncalls) X(lag,g->analysis->ncalls) \
  X(time,g->vertices) X(call,g->vertices) X(node,g->vertices) X(observer,g->vertices) \
  X(lagged,g->vertices) X(task,g->vertices) X(from,g->built+g->added) X(to,g->built+g->added) \
  X(delay,g->built+g->added) X(cause,g->built) X(added_from,g->added) X(added_to,g->added) \
  X(added_delay,g->added) X(entry,g->added) X(key,g->vertices>g->added?g->vertices:g->added) \
  X(first,(uint64_t)g->analysis->nodes->count+2) X(rows,g->vertices) \
  X(wait_first,(uint64_t)g->analysis->nodes->count*g->analysis->ncalls+1) X(wait_order,g->added) \
  X(tell_first,(uint64_t)g->analysis->functions+2) X(tell_order,g->added)
static void plan_step_results_put(struct plan_archive *out,const struct plan_step *p){
  if(p->schedule){
    const struct plan_schedule_result *s=p->schedule;
    plan_archive_put(out,s,sizeof *s);
#define PUT(field,count) if(s->field)plan_archive_put(out,s->field,(size_t)(count)*sizeof *s->field);
    PLAN_SCHEDULE_ARRAYS(PUT)
#undef PUT
  }
  if(p->graph){
    const struct plan_graph_result *g=p->graph;
    plan_archive_put(out,g,sizeof *g);
#define PUT(field,count) if(g->field)plan_archive_put(out,g->field,(size_t)(count)*sizeof *g->field);
    PLAN_GRAPH_ARRAYS(PUT)
#undef PUT
    plan_group_put(out,g->room_rows);
    plan_archive_put(out,g->room_graph,sizeof *g->room_graph);
    plan_archive_put(out,g->room_lags,sizeof *g->room_lags);
    plan_archive_put(out,g->room_query,g->room_lags->queries*sizeof *g->room_query);
    plan_archive_put(out,g->room_lags->value,g->room_lags->queries*sizeof *g->room_lags->value);
    plan_group_put(out,g->room_lags->sources);
  }
}
static int plan_step_results_get(struct plan_archive *in,struct plan_step *p,const struct plan_step *saved,
                                 const struct mesh_plan_analysis *analysis){
  p->schedule=NULL;p->graph=NULL;
  if(saved->schedule){
    const struct plan_schedule_result *view=plan_archive_view(in,1,sizeof *view);
    if(!view)return in->status;
    struct plan_schedule_result *s=p->schedule=malloc(sizeof *s);
    if(!s)return ENOMEM;
    *s=*view;
#define GET(field,count) s->field=view->field?(void *)(uintptr_t)plan_archive_view(in,(count),sizeof *s->field):NULL;
    PLAN_SCHEDULE_ARRAYS(GET)
#undef GET
  }
  if(saved->graph){
    const struct plan_graph_result *view=plan_archive_view(in,1,sizeof *view);
    if(!view)return in->status;
    struct plan_graph_result *g=p->graph=malloc(sizeof *g);
    if(!g)return ENOMEM;
    *g=*view;g->analysis=analysis;g->room_rows=NULL;g->room_graph=NULL;g->room_lags=NULL;
#define GET(field,count) g->field=view->field?(void *)(uintptr_t)plan_archive_view(in,(count),sizeof *g->field):NULL;
    PLAN_GRAPH_ARRAYS(GET)
#undef GET
    int status=plan_group_get(in,g->from,&g->room_rows);
    if(status)return status;
    const struct mesh_graph_lag_graph *graph=plan_archive_view(in,1,sizeof *graph);
    const struct mesh_graph_lag_result *lags=plan_archive_view(in,1,sizeof *lags);
    if(!graph || !lags)return in->status;
    g->room_graph=malloc(sizeof *g->room_graph);g->room_lags=calloc(1,sizeof *g->room_lags);
    if(!g->room_graph || !g->room_lags)return ENOMEM;
    *g->room_graph=(struct mesh_graph_lag_graph){graph->vertices,g->room_rows->first,plan_room_head,plan_room_lag,g};
    *g->room_lags=*lags;g->room_lags->graph=g->room_graph;g->room_lags->sources=NULL;
    g->room_query=(void *)(uintptr_t)plan_archive_view(in,lags->queries,sizeof *g->room_query);
    g->room_lags->query=g->room_query;
    g->room_lags->value=(void *)(uintptr_t)plan_archive_view(in,lags->queries,sizeof *g->room_lags->value);
    status=plan_group_get(in,g->room_query,&g->room_lags->sources);
    if(status)return status;
    g->room_lags->sources->key=mesh_graph_query_source;
  }
  return in->status;
}
static void plan_step_loaded_free(struct plan_step *p){
  free(p->schedule);
  if(p->graph){
    struct plan_graph_result *g=p->graph;
    free(g->room_rows);free(g->room_graph);
    if(g->room_lags){free(g->room_lags->sources);free(g->room_lags);}
    free(g);
  }
}
#undef PLAN_SCHEDULE_ARRAYS
#undef PLAN_GRAPH_ARRAYS
struct plan_saved_compilation { uint64_t magic,bytes,places; struct mesh_range rows,arena[2]; uint64_t used[2],signature; };
#define PLAN_SAVED_COMPILATION UINT64_C(0x314d4f435048534d)
#define PLAN_STEP_ARRAYS(X) \
  X(item,p->flow->givens) X(completed,p->flow->transfers) X(sync,(uint64_t)p->waits+p->tells) X(wait_row,(uint64_t)p->flow->calls+1)
#define PLAN_COMPILATION_ARRAYS(X) \
  X(root_item,c->result.roots) X(page,c->result.planned.rows.count) \
  X(adjacency,(uint64_t)c->result.planned.analysis->nodes->count+1+c->places) \
  X(toward,c->places) X(place,c->places) X(rows_by,(uint64_t)c->result.planned.analysis->nodes->count+1) \
  X(listed,c->places) X(state_bytes,c->result.functions)
int mesh_plan_compilation_save(const struct mesh_plan_compilation *c,void **bytes,size_t *length){
  *bytes=NULL;*length=0;
  struct plan_archive out={0};
  const struct mesh_planned *planned=&c->result.planned;
  struct plan_saved_compilation head={.magic=PLAN_SAVED_COMPILATION,.places=c->places,.rows=planned->rows,
    .arena={planned->arena[0],planned->arena[1]},.used={planned->used[0],planned->used[1]}};
  size_t size=0;
  int status=mesh_graph_archive(planned->signature,NULL,0,&size);
  if(status!=ERANGE)return status?status:EINVAL;
  head.signature=size;
  plan_archive_put(&out,&head,sizeof head);
  status=plan_local_put(&out,&c->result);
  if(status){free(out.bytes);return status;}
  plan_signature_put(&out,planned,head.signature);
  plan_extents_put(&out,planned,c->result.functions);
#define PUT(field,count) plan_archive_put(&out,c->field,(size_t)(count)*sizeof *c->field);
  PLAN_COMPILATION_ARRAYS(PUT)
#undef PUT
  for(uint32_t f=0;f<c->result.functions;f++){
    const struct plan_step *p=c->step+f;
    plan_archive_put(&out,p,sizeof *p);
#define PUT(field,count) if(p->field)plan_archive_put(&out,p->field,(size_t)(count)*sizeof *p->field);
    PLAN_STEP_ARRAYS(PUT)
#undef PUT
    for(uint64_t j=0;j<p->flow->entries;j++){
      const struct plan_entry entry={(void *)(uintptr_t)(p->entry[j].source-p->flow->entry),p->entry[j].place};
      plan_archive_put(&out,&entry,sizeof entry);
    }
    plan_step_results_put(&out,p);
    plan_archive_put(&out,c->state[f],(size_t)c->state_bytes[f]);
  }
  if(out.status){free(out.bytes);return out.status;}
  head.bytes=out.size;memcpy(out.bytes,&head,sizeof head);
  *bytes=out.bytes;*length=out.size;return 0;
}
int mesh_plan_compilation_load(struct mesh_graph_context *graph,const void *bytes,size_t length,
                               void (*release)(void *),void *owner,struct mesh_plan_compilation **out){
  *out=NULL;
  if((uintptr_t)bytes%8)return EINVAL;
  struct plan_archive in={.bytes=(void *)bytes,.size=length,.read=1};
  const struct plan_saved_compilation *head=plan_archive_view(&in,1,sizeof *head);
  if(!head || head->magic!=PLAN_SAVED_COMPILATION || head->bytes!=length)return EINVAL;
  struct mesh_plan_compilation *c=calloc(1,sizeof *c);
  if(!c)return ENOMEM;
  c->references=1;c->owned=(void *)bytes;c->places=head->places;
  int status=plan_local_get(&in,&c->result,1,graph);
  if(status)goto done;
  struct mesh_planned *planned=&c->result.planned;
  planned->rows=head->rows;planned->arena[0]=head->arena[0];planned->arena[1]=head->arena[1];
  planned->used[0]=head->used[0];planned->used[1]=head->used[1];
  struct plan_loaded *loaded=planned->owned;
  planned->signature=calloc(1,sizeof *planned->signature);
  if(!planned->signature){status=ENOMEM;goto done;}
  if((status=plan_signature_get(&in,head->signature,loaded->bytes,planned->signature)))goto done;
  if((status=plan_orders_valid(planned,c->result.calls,c->result.ncalls,c->result.function,planned->signature->tasks)))goto done;
  if((status=plan_extents_load(&in,c->result.function,c->result.functions,planned)))goto done;
  c->placement=c->result.placement;
#define GET(field,count) c->field=(void *)(uintptr_t)plan_archive_view(&in,(count),sizeof *c->field);
  PLAN_COMPILATION_ARRAYS(GET)
#undef GET
  const uint32_t count=c->result.functions;
  c->step=calloc((size_t)count+1,sizeof *c->step);c->state=calloc((size_t)count+1,sizeof *c->state);
  if(!c->step || !c->state){status=ENOMEM;goto done;}
  for(uint32_t f=0;!in.status && f<count;f++){
    const struct plan_step *saved=plan_archive_view(&in,1,sizeof *saved);
    if(!saved)break;
    struct plan_step *p=c->step+f;
    *p=*saved;p->flow=planned->analysis->flow+f;p->function=c->result.function+f;
    p->result=planned->result+f;p->root_result=planned->root;p->entry=NULL;p->schedule=NULL;p->graph=NULL;
#define GET(field,count) p->field=saved->field?(void *)(uintptr_t)plan_archive_view(&in,(count),sizeof *p->field):NULL;
    PLAN_STEP_ARRAYS(GET)
#undef GET
    const struct plan_entry *entry=plan_archive_view(&in,p->flow->entries,sizeof *entry);
    if(in.status)break;
    p->entry=malloc(((size_t)p->flow->entries+1)*sizeof *p->entry);
    if(!p->entry){status=ENOMEM;break;}
    for(uint64_t j=0;j<p->flow->entries;j++){
      const uintptr_t at=(uintptr_t)entry[j].source;
      if(at>=p->flow->entries){status=EINVAL;break;}
      p->entry[j]=(struct plan_entry){p->flow->entry+at,entry[j].place};
    }
    if(!status)status=plan_step_results_get(&in,p,saved,planned->analysis);
    if(status)break;
    c->state[f]=(void *)(uintptr_t)plan_archive_view(&in,c->state_bytes[f],1);
    if(!in.status && !plan_state_valid(c->state[f],c->state_bytes[f],p->flow->transfers))status=EINVAL;
  }
  if(!status)status=in.status;
  if(!status && in.at!=in.size)status=EINVAL;
  if(!status){loaded->bytes->release=release;loaded->bytes->owner=owner;}
done:
  if(status)mesh_plan_compilation_free(c);else *out=c;
  return status;
}
#undef PLAN_PLACEMENT_ARRAYS
#undef PLAN_STEP_ARRAYS
#undef PLAN_COMPILATION_ARRAYS
