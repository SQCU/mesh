#include "mesh-collective.h"
#include "mesh-graph.h"
#include <float.h>
#include <CommonCrypto/CommonDigest.h>
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

#define MESH_PLAN_EPS 0.1

struct plan_demand {
  uint32_t source,identity,first;
  uint32_t mine;
  /* design/plan-to-bridge-change-request.md R2 */
  const struct mesh_strided *view;
  uint32_t tree,trees;
  uint32_t release;
  /* design/aot-compilation-change-request.md A9 */
  uint64_t leaves,bytes,instructions,cut,at;
  uint32_t relay;
  uint64_t on;
  uint32_t forwards;
  uint32_t own;
  uint32_t set;
  uint32_t vertex;
  uint32_t value;
  uint32_t range,received;
  uint32_t consumed;
  uint32_t counted;
  uint64_t group;
  uint32_t sending,sends;
};
/* design/disjoint-pages-change-request.md C3.3 */
struct plan_value { uint32_t demand,node,write,read,reads,source; };
/* design/plan-to-bridge-change-request.md R5; M. Cowan et al., MSCCLang, ASPLOS 2023, §5.2 */
struct plan_entry { uint32_t node,direction,demand,field,communication,set,place; uint64_t instruction; };
struct plan_sync { uint32_t node,function,entry,every; int64_t base; };
struct plan {
  struct mesh_ctx *context;
  const struct mesh_declared *declared;
  uint32_t transfers;
  struct mesh_value *given;
  uint32_t givens;
  /* design/plan-to-bridge-change-request.md R2; design/disjoint-pages-change-request.md C3.3 */
  unsigned char *named;
  uint32_t *uncovered;
  uint64_t *cut,*at,toward;
  uint32_t sets,set_item,*set_first,*member,*set_blocks,*set_far,*block_first,*block_set,*block_element;
  const struct mesh_weight *weights;
  const struct mesh_dependence *dependence;
  uint32_t dependences;
  struct plan_demand *demand;
  uint32_t demands,n,self,copies,start_lead;
  struct mesh_view_id view;               /* JGroups ViewId.java */
  uint32_t *of;
  uint32_t pairs,from[MESH_NODES_MOST*(MESH_NODES_MOST-1)/2+1],to[MESH_NODES_MOST*(MESH_NODES_MOST-1)/2+1];
  double capacity[MESH_NODES_MOST*(MESH_NODES_MOST-1)/2+1];
  size_t *tree;
  uint32_t *parent;
  uint64_t *run;
  uint32_t made;
  uint64_t *members,*lead,*comm,comms;
  uint64_t releases;
  struct plan_entry *entry;
  uint64_t entries,receives;
  uint32_t ranges,consumed;
  uint32_t sending;
  uint64_t *child;
  struct plan_value *value;
  uint32_t values;
  uint32_t *read;
  uint32_t *here,local;
  uint32_t vertices;
  size_t edges;
  size_t *first;
  uint32_t *head,*lag;
  uint32_t *between;
  uint32_t *farthest;
  uint64_t phases,counts,completions,supplied,publications,status,hops,times,words;
  /* L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, Math. Oper. Res. 22(3), 1997, §2 */
  double *completed;
  /* design/aot-compilation-change-request.md A10 */
  uint64_t deliveries,submits,syncs,rows,room,outstanding,submit_room,spanned,nodes,forwarded;
  uint32_t waits,tells,calls,horizon,*wait_row;
  struct plan_sync *sync;
  double time[MESH_NODES_MOST];
  uint32_t *item,state;
  uint64_t base;
};

struct plan_key { uint32_t source,identity,index; };
static int plan_key_order(const void *a,const void *b){
  const struct plan_key *x=a,*y=b;
  if(x->source!=y->source)return x->source<y->source?-1:1;
  if(x->identity!=y->identity)return x->identity<y->identity?-1:1;
  return (x->index>y->index)-(x->index<y->index);
}
struct plan_edges { const uint32_t *from,*to; };
static uint32_t plan_end(const void *context,uint32_t e,uint32_t side){
  const struct plan_edges *graph=context;
  return side?graph->to[e]:graph->from[e];
}
static int plan_leaf(const void *context,uint32_t d,uint32_t v){
  return (int)(((const struct plan_demand *)context)[d].leaves>>v&1);
}
static uint32_t plan_at(const void *context,size_t k){ return ((const uint32_t *)context)[k]; }
struct plan_tree { const uint32_t *parent; uint32_t n,root; };
static uint32_t plan_tree_parent(const void *context,size_t v){
  const struct plan_tree *tree=context;
  return v==tree->root || tree->parent[v]>=tree->n?tree->n:tree->parent[v];
}
static void plan_tree_up(void *consumer,uint32_t at,uint32_t step){ if(step==1)*(uint32_t *)consumer=at; }
static uint64_t plan_quantum(const struct hdr *m){ return (uint64_t)m->block*m->pgsz; }
static uint64_t plan_blocks(const struct hdr *m,uint64_t bytes){
  const uint64_t quantum=plan_quantum(m);
  return (bytes+quantum-1)/quantum;
}
/* L. J. Hendren, G. R. Gao, E. R. Altman and C. Mukerji, ACAPS Technical Memo 33, §2.2;
   design/plan-to-bridge-change-request.md R2 */
static uint32_t plan_copies(const struct plan *p,uint32_t j){
  if(p->named[j] || (p->given[j].kind==MESH_EXECUTION && !p->given[j].lag))return p->copies;
  return p->given[j].lag?p->given[j].lag:1;
}

static uint32_t plan_rank(uint64_t set,uint32_t v){ return (uint32_t)__builtin_popcountll(set&((UINT64_C(1)<<v)-1)); }
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
static uint32_t plan_event(const struct plan *p,uint32_t i,int produced,int own){
  if(produced)return plan_produced(p,p->of[i]);
  if(own)return plan_own(p,p->of[i]);
  return plan_consumed(p,p->of[i],p->declared[i].destination);
}
static uint32_t plan_event_node(const struct mesh_declared *declared,uint32_t i,int produced,int own){
  return produced || own?declared[i].source:declared[i].destination;
}
struct plan_edge { uint32_t tail,head,lag; };
static uint32_t plan_edge_tail(const void *context,size_t k){ return ((const struct plan_edge *)context)[k].tail; }

/* design/disjoint-pages-change-request.md C3.3 */
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

/* design/disjoint-pages-change-request.md C3.2-C3.4 */

static uint32_t plan_head(const void *context,size_t k){ return ((const struct plan *)context)->head[k]; }
static uint64_t plan_lag(const void *context,size_t k){ return ((const struct plan *)context)->lag[k]; }

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
    status=mesh_graph_order(p->vertices,first,plan_at,head,indegree,ordinal)?EINVAL:0;
  }
  free(first);free(head);free(indegree);free(ordinal);
  return status;
}

static void plan_name(const struct plan *p,uint32_t v,const char *what,uint64_t lag){
  const struct plan_value *x=p->value+v;
  const struct plan_demand *d=p->demand+x->demand;
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
static int plan_unroll(struct plan *p){
  const uint32_t values=p->values;
  p->copies=p->horizon*p->calls;
  /* M. Lam, PLDI 1988, §2.3: a transferred value its host writes `lag` executions ahead (design/aot-compilation-change-
     request.md, appendix item 2) takes the plan's copies (plan_copies), at least its lag, so that the copy of execution
     e + lag - 1 the host writes is not one the sends of execution e still read */
  for(uint32_t j=0;j<p->givens;j++)if(p->named[j] && p->given[j].lag>p->copies)p->copies=p->given[j].lag;
  if(!values)return 0;
  const struct plan_value *last=p->value+values-1;
  const uint32_t sources=last->source+1+last->reads;
  size_t cells,pairs;
  if(__builtin_mul_overflow((size_t)sources,(size_t)values,&cells) || cells>=SIZE_MAX/sizeof(uint64_t) ||
     __builtin_mul_overflow((size_t)p->local,(size_t)p->local,&pairs) || pairs>=SIZE_MAX/sizeof(uint32_t))return ENOMEM;
  uint32_t *source=malloc((size_t)sources*sizeof *source),*target=malloc((size_t)values*sizeof *target);
  uint64_t *lags=malloc((cells+1)*sizeof *lags);
  p->between=malloc((pairs+1)*sizeof *p->between);
  p->farthest=calloc((size_t)p->local+1,sizeof *p->farthest);
  int status=!source || !target || !lags || !p->between || !p->farthest?ENOMEM:0;
  for(uint32_t v=0;!status && v<values;v++){
    const struct plan_value *x=p->value+v;
    source[x->source]=x->write;
    for(uint32_t j=0;j<x->reads;j++)source[x->source+1+j]=p->read[x->read+j];
    target[v]=x->write;
  }
  if(!status)status=mesh_graph_lags(p->vertices,p->first,plan_head,plan_lag,p,sources,source,values,target,lags);
  for(uint32_t v=0;!status && v<values;v++){
    const struct plan_value *x=p->value+v;
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
    const struct plan_value *x=p->value+p->here[a];
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
  free(source);free(target);free(lags);
  return status;
}

/* design/disjoint-pages-change-request.md C3.2, C3.7; design/aot-compilation-change-request.md,
   appendix items 2 and 5 */
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
  p->completions=plan_align(p->counts+(uint64_t)p->ranges*p->copies*sizeof(uint64_t));
  p->supplied=plan_align(p->completions+(uint64_t)p->sending*p->copies*2*sizeof(uint64_t));
  p->publications=plan_align(p->supplied+p->releases*sizeof(uint64_t));
  p->status=plan_align(p->publications+(uint64_t)p->transfers*sizeof(struct mesh_publication));
  p->deliveries=plan_align(p->status+sizeof(struct mesh_status));
  p->submits=plan_align(p->deliveries+p->room*sizeof(struct mesh_delivery));
  p->syncs=plan_align(p->submits+p->submit_room*sizeof(struct mesh_submission));
  p->rows=plan_align(p->syncs+((uint64_t)p->waits+p->tells)*sizeof(struct mesh_sync));
  p->hops=plan_align(p->rows+((uint64_t)p->calls+1)*sizeof(uint32_t));
  p->times=plan_align(p->hops+p->entries*p->copies*sizeof(struct mesh_hop));
  p->words=p->times+(uint64_t)p->transfers*sizeof(double);
}
static uint64_t plan_supplied(const struct plan *p,const struct plan_demand *d){
  return p->base+p->supplied+(uint64_t)d->release*sizeof(uint64_t);
}
static uint64_t plan_group(const struct plan *p,const struct plan_demand *d,uint32_t c){
  return p->base+p->phases+d->group+(uint64_t)c*mesh_group_bytes(d->received);
}
static uint64_t plan_count(const struct plan *p,const struct plan_demand *d,uint32_t c){
  return p->base+p->counts+((uint64_t)d->counted*p->copies+c)*sizeof(uint64_t);
}
static uint64_t plan_completion(const struct plan *p,const struct plan_demand *d,uint32_t c){
  return p->base+p->completions+((uint64_t)d->sending*p->copies+c)*2*sizeof(uint64_t);
}

enum { PLAN_ALONE, PLAN_LOCAL, PLAN_ROUTED };
struct plan_item { uint64_t element,blocks; uint32_t kind,function,index,copy,space; };
struct plan_placement { struct plan_item *item; uint64_t *count,*chain,items,elements; uint32_t split,table;
                        struct mesh_range range[2]; uint64_t *at,blocks; uint32_t run,space; };

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
static uint64_t plan_table(uint32_t count,uint32_t nodes,uint64_t places){
  return (uint64_t)count*sizeof(uint64_t)+((uint64_t)nodes+1+places)*sizeof(uint32_t);
}
static void plan_items(struct plan *plans,uint32_t count,const struct hdr *m,struct plan_placement *x,uint32_t space){
  uint64_t places=0;
  for(uint32_t f=0;f<count;f++){
    struct plan *p=plans+f;
    const uint32_t u=p->copies;
    places+=(uint64_t)__builtin_popcountll(p->toward);
    if(space){
      p->set_item=(uint32_t)x->items;
      for(uint32_t s=0;s<p->sets;s++)for(uint32_t c=0;c<u;c++)plan_item_put(x,PLAN_ROUTED,f,s,c,p->set_blocks[s],1);
    }
    for(uint32_t j=0;j<p->givens;j++){
      const struct mesh_value *v=p->given+j;
      if(v->run || (p->named[j]?1u:v->wire)!=space)continue;
      const uint32_t kind=v->kind==MESH_EXECUTION && !v->lag?PLAN_LOCAL:PLAN_ALONE;
      p->item[j]=p->uncovered[j]?(uint32_t)x->items:MESH_ABSENT;
      for(uint32_t c=0;p->uncovered[j] && c<plan_copies(p,j);c++)plan_item_put(x,kind,f,j,c,p->uncovered[j],space);
    }
    if(!space){p->state=(uint32_t)x->items;plan_item_put(x,PLAN_ALONE,f,0,0,plan_blocks(m,p->words),0);}
  }
  /* design/torch-capture-change-request.md item 6; TensorFlow Lite 6cf18d88, tensorflow/lite/arena_planner.cc:433-436 */
  if(x->blocks && x->space==space){x->run=(uint32_t)x->items;plan_item_put(x,PLAN_ALONE,UINT32_MAX,0,0,x->blocks,space);}
  if(!space){
    x->table=(uint32_t)x->items;
    plan_item_put(x,PLAN_ALONE,UINT32_MAX,0,0,plan_blocks(m,plan_table(count,plans[0].n,places)),0);
  }
}

/* design/plan-to-bridge-change-request.md R2, R8; design/disjoint-pages-change-request.md A3, C3.3, C3.4;
   C. Mukerji, M.Sc. thesis, McGill University, 1994, Algorithm 3.2.1 */
/* C. Mukerji, M.Sc. thesis, McGill University, 1994, §2.1, §3.1; L. J. Hendren, G. R. Gao, E. R. Altman and C. Mukerji,
   ACAPS Technical Memo 33, §2.2; M. Lam, PLDI 1988, §2.3; MPI 4.1 §3.7, §3.9;
   S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, (1) */
static void plan_landing(const struct plan *p,uint32_t j,int received,uint32_t far,uint64_t n,uint64_t c,
                         uint64_t *start,uint64_t *end){
  const uint64_t u=p->copies,ring=u*n;
  uint64_t back=p->start_lead,from=0,last=n-1,length;
  if(j!=MESH_ABSENT){
    last=p->given[j].end;
    if(!received){back=p->given[j].lag?p->given[j].lag-1:0;from=p->given[j].lag?0:p->given[j].begin;}
  }
  const uint64_t ahead=far-1;
  const int whole=far==UINT32_MAX || __builtin_mul_overflow(back+ahead,n,&length) ||
                  __builtin_add_overflow(length,last-from+1,&length) || length>=ring;
  *start=((c+u-back%u)%u)*n+from;
  *end=(*start+(whole?ring:length)-1)%ring;
}
static int plan_interval(const struct plan *plans,const struct plan_item *x,const uint64_t *operations,uint64_t *start,
                         uint64_t *end){
  if(x->kind!=PLAN_LOCAL && x->kind!=PLAN_ROUTED)return 0;
  const struct plan *p=plans+x->function;
  const uint64_t n=operations[x->function],c=x->copy;
  if(x->kind==PLAN_LOCAL){
    *start=c*n+p->given[x->index].begin;
    *end=c*n+p->given[x->index].end;
    return 1;
  }
  const uint32_t j=p->demand[p->value[p->here[p->member[p->set_first[x->index]]]].demand].mine;
  int received=j==MESH_ABSENT;
  for(uint32_t k=p->set_first[x->index];k<p->set_first[x->index+1];k++)
    received|=p->demand[p->value[p->here[p->member[k]]].demand].source!=p->self;
  plan_landing(p,j,received,p->set_far[x->index],n,c,start,end);
  return 1;
}
/* design/torch-capture-change-request.md item 6; L. J. Hendren, G. R. Gao, E. R. Altman and C. Mukerji, ACAPS Technical
   Memo 33, §2.2; C. Mukerji, M.Sc. thesis, McGill University, 1994, §2.1, §3.1 */
static void plan_location(const struct plan *p,uint32_t j,uint64_t n,uint64_t c,uint64_t *start,uint64_t *end){
  const struct mesh_value *v=p->given+j;
  if(v->kind!=MESH_EXECUTION || v->lag){*start=0;*end=(uint64_t)p->copies*n-1;return;}
  if(!p->named[j]){*start=c*n+v->begin;*end=c*n+v->end;return;}
  int received=0;
  uint32_t far=0;
  for(uint32_t i=0;i<p->demands;i++){
    const struct plan_demand *d=p->demand+i;
    if(d->mine!=j)continue;
    received|=d->source!=p->self;
    if(p->farthest[d->value]>far)far=p->farthest[d->value];
  }
  plan_landing(p,j,received,far,n,c,start,end);
}
struct plan_arcs { const uint64_t *start,*end; const uint32_t *function; };
static int plan_intersect(const void *context,size_t a,size_t b){
  const struct plan_arcs *x=context;
  return x->function[a]!=x->function[b] || mesh_graph_arcs_overlap(x->start[a],x->end[a],x->start[b],x->end[b]);
}
/* design/torch-capture-change-request.md item 6; TensorFlow Lite 6cf18d88,
   tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_size_assignment.cc:68-143 and
   tensorflow/lite/arena_planner.h:141-149; GaijinEntertainment/DagorEngine 2baf1947,
   prog/gameLibs/render/daFrameGraph/backend/resourceScheduling/heapLayout.h:19 */
static int plan_block(struct plan *plans,uint32_t count,const struct hdr *m,const uint64_t *operations,
                      struct plan_placement *x){
  size_t records=0;
  for(uint32_t f=0;f<count;f++)for(uint32_t j=0;j<plans[f].givens;j++)
    if(plans[f].given[j].run)records+=plan_copies(plans+f,j);
  uint64_t *size=malloc((records+1)*sizeof *size),*start=malloc((records+1)*sizeof *start),
           *end=malloc((records+1)*sizeof *end),*offset=malloc((records+1)*sizeof *offset),total=0;
  uint32_t *function=malloc((records+1)*sizeof *function);
  x->at=malloc((records+1)*sizeof *x->at);
  int status=!size || !start || !end || !offset || !function || !x->at?ENOMEM:0;
  for(uint32_t f=0,r=0;!status && f<count;f++){
    struct plan *p=plans+f;
    const uint64_t n=operations[f],ring=(uint64_t)p->copies*n;
    for(uint32_t j=0;j<p->givens;j++){
      const struct mesh_value *v=p->given+j;
      if(!v->run)continue;
      if(p->named[j] || v->wire)x->space=1;
      p->item[j]=r;
      for(uint32_t c=0;c<plan_copies(p,j);c++,r++){
        uint64_t last;
        plan_location(p,j,n,c,start+r,&last);
        end[r]=(last+1)%ring;
        size[r]=v->section.bytes;
        function[r]=f;
      }
    }
  }
  const struct plan_arcs arcs={start,end,function};
  const uint64_t quantum=plan_quantum(m);
  if(!status)status=mesh_graph_greedy_by_size(records,size,plan_intersect,&arcs,quantum,offset,&total);
  for(size_t r=0;!status && r<records;r++)x->at[r]=offset[r]/quantum;
  if(!status)x->blocks=plan_blocks(m,total);
  free(size);free(start);free(end);free(offset);free(function);
  return status;
}
static int plan_place(struct plan *plans,uint32_t count,struct hdr *m,struct plan_placement *x){
  *x=(struct plan_placement){0};
  uint64_t *operations=malloc(((size_t)count+1)*sizeof *operations);
  if(!operations)return ENOMEM;
  for(uint32_t f=0;f<count;f++){
    operations[f]=1;
    for(uint32_t j=0;j<plans[f].givens;j++){
      const struct mesh_value *v=plans[f].given+j;
      if(v->kind==MESH_EXECUTION && (uint64_t)v->end+1>operations[f])operations[f]=(uint64_t)v->end+1;
    }
  }
  const int blocked=plan_block(plans,count,m,operations,x);
  if(blocked){free(operations);return blocked;}
  plan_items(plans,count,m,x,1);
  plan_items(plans,count,m,x,0);
  if(x->items>=UINT32_MAX || x->elements>=SIZE_MAX/sizeof(uint64_t)){free(operations);return ENOMEM;}
  x->item=malloc(((size_t)x->items+1)*sizeof *x->item);
  x->count=malloc(((size_t)x->items+1)*sizeof *x->count);
  x->chain=malloc(((size_t)x->elements+1)*sizeof *x->chain);
  if(!x->item || !x->count || !x->chain){free(operations);return ENOMEM;}
  x->items=x->elements=0;
  plan_items(plans,count,m,x,1);
  x->split=(uint32_t)x->items;
  plan_items(plans,count,m,x,0);
  x->range[1]=mesh_arena_range(m,1);
  x->range[0]=mesh_arena_range(m,0);
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
      status=mesh_graph_fat_cover((uint64_t)plans[f].copies*operations[f],n,start,end,counts,out,&colors);
      for(uint64_t r=0,o=0;!status && r<n;r++)
        for(uint64_t k=0;k<counts[r];k++)x->chain[x->item[index[r]].element+k]=blocks+out[o++];
      blocks+=colors;
    }
    if(!status && blocks>x->range[x->item[first[s]].space].count/m->block)status=ENOMEM;
  }
  free(operations);free(counts);free(start);free(end);free(out);free(index);
  return status;
}
static uint32_t plan_page(const struct plan_placement *x,uint32_t block,uint32_t i,uint64_t k){
  return (uint32_t)(x->range[x->item[i].space].first+x->chain[x->item[i].element+k]*block);
}

/* design/plan-to-bridge-change-request.md R2; design/torch-capture-change-request.md item 6 */
static uint32_t plan_frame(const struct plan *p,const struct plan_placement *x,uint32_t block,uint32_t j,uint32_t c,
                           uint32_t k){
  if(p->given[j].run)return plan_page(x,block,x->run,x->at[p->item[j]+c]+k);
  if(p->block_first[j]==MESH_ABSENT)return plan_page(x,block,p->item[j]+c,k);
  const uint32_t at=p->block_first[j]+k,s=p->block_set[at];
  return plan_page(x,block,s==MESH_ABSENT?p->item[j]+c:p->set_item+s*p->copies+c,p->block_element[at]);
}
static int plan_rows(struct plan *plans,uint32_t count,const struct hdr *m,const struct plan_placement *x,
                     uint32_t *page,uint64_t rows){
  const uint32_t block=m->block;
  for(uint32_t f=0;f<count;f++){
    const struct plan *p=plans+f;
    for(uint32_t j=0;j<p->givens;j++){
      const struct mesh_section v=p->given[j].section;
      for(uint32_t c=0;c<v.count;c++)for(uint32_t k=0;k<v.stride;k++)
        page[v.first+(size_t)c*v.stride+k]=plan_frame(p,x,block,j,c,k);
    }
  }
  unsigned char *seen=calloc((size_t)mesh_blocks(m)+1,1);
  if(!seen)return ENOMEM;
  for(uint64_t r=0;r<rows;r++)if(seen[page[r]/block]<2)seen[page[r]/block]++;
  for(uint32_t f=0;f<count;f++)for(uint32_t j=0;j<plans[f].givens;j++){
    struct mesh_value *v=plans[f].given+j;
    const uint64_t end=v->section.first+(uint64_t)v->section.count*v->section.stride;
    v->shares=0;
    for(uint64_t r=v->section.first;r<end && !v->shares;r++)v->shares=seen[page[r]/block]>1;
  }
  free(seen);
  return 0;
}

/* design/disjoint-pages-change-request.md C3.3, C3.7 */

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
static uint32_t plan_communication(const struct plan *p,uint32_t j,uint64_t page,uint32_t head){
  return (uint32_t)(p->comm[j]+page*(uint64_t)__builtin_popcountll(p->members[j])+plan_rank(p->members[j],head));
}

/* design/disjoint-pages-change-request.md C3.7; design/plan-to-bridge-change-request.md R5 */
static void plan_walk(struct plan *p){
  const uint32_t n=p->n,self=p->self;
  size_t first[(size_t)n+2],child[(size_t)n+1];
  uint64_t at=0,receives=0;
  if(!p->entry)p->ranges=p->consumed=p->sending=0,p->toward=0;
  for(uint32_t i=0;i<p->demands;i++){
    struct plan_demand *d=p->demand+i;
    uint32_t field=0,sends=0;
    uint64_t k=0;
    for(uint32_t j=d->tree;j<d->tree+d->trees;j++){
      const uint64_t end=k+p->run[j];
      const uint32_t set=d->set?j+1:0;
      const struct plan_tree tree={p->parent+p->tree[j]*n,n,d->source};
      uint32_t in=MESH_ABSENT;
      mesh_graph_climb(tree.parent,n,d->source,self,plan_tree_up,&in);
      mesh_graph_rows(n+1,n,plan_tree_parent,&tree,first,child);
      if(in==MESH_ABSENT && first[self+1]==first[self]){k=end;continue;}
      if(k<end){
        if(in!=MESH_ABSENT)p->toward|=UINT64_C(1)<<in;
        for(size_t c=first[self];c<first[self+1];c++)p->toward|=UINT64_C(1)<<child[c];
      }
      for(;k<end;k++){
        uint32_t mine=MESH_ABSENT;
        if(in!=MESH_ABSENT){
          mine=field++;
          if(p->entry)p->entry[at]=(struct plan_entry){.node=in,.direction=MESH_RECEIVE,.demand=i,.field=mine,
                                                       .communication=plan_communication(p,j,k-p->lead[j],self),
                                                       .set=set,.instruction=k};
          at++;
        }
        for(size_t c=first[self];c<first[self+1];c++,at++,sends++)
          if(p->entry)p->entry[at]=(struct plan_entry){.node=(uint32_t)child[c],.direction=MESH_SEND,.demand=i,
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
struct plan_span { uint32_t first,count; };
static struct plan_span plan_tasks(const struct plan *p,const struct plan_span *consumer,const uint32_t *reader,
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
static int plan_schedule(struct plan *p,const struct mesh_weight *weight,const struct mesh_dependence *dependence,
                         uint32_t dependences){
  const uint32_t n=p->n,demands=p->demands,count=p->transfers;
  const struct mesh_declared *declared=p->declared;
  uint32_t link_of[(size_t)n*n+1],path_link[(size_t)n+1],path_head[(size_t)n+1];
  uint32_t *reader=malloc(((size_t)demands+1)*sizeof *reader);
  struct plan_span *consumer=malloc(((size_t)count+1)*sizeof *consumer);
  if(!reader || !consumer){free(reader);free(consumer);return ENOMEM;}
  uint64_t tasks=demands,edges=0,entries=0;
  int status=0;
  for(uint32_t d=0;d<demands;d++)
    if((reader[d]=p->demand[d].own?(uint32_t)tasks:MESH_ABSENT)!=MESH_ABSENT)edges++,tasks++;
  if(tasks>=UINT32_MAX)status=ENOMEM;
  for(uint32_t i=0;!status && i<count;i++){
    const struct plan_demand *d=p->demand+p->of[i];
    consumer[i]=(struct plan_span){(uint32_t)tasks,declared[i].kind==MESH_PAGES?(uint32_t)d->instructions:1};
    tasks+=consumer[i].count;
    edges+=d->instructions;
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
    /* design/aot-compilation-change-request.md A3; design/plan-to-bridge-change-request.md R5 */
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
    double most=0;
    for(size_t v=0;v<(size_t)n*n;v++)link_of[v]=MESH_ABSENT;
    for(uint32_t e=0;e<p->pairs;e++)if(p->capacity[e]>most)most=p->capacity[e];
    for(uint32_t e=0;e<p->pairs;e++){
      link_of[(size_t)p->from[e]*n+p->to[e]]=2*e;
      link_of[(size_t)p->to[e]*n+p->from[e]]=2*e+1;
      rate[2*e]=rate[2*e+1]=p->capacity[e]/most;
    }
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
    size_t k=0;
    for(size_t x=0;x<edges;x++){
      uint32_t tail=from[x];
      for(size_t r=route[x];r<route[x+1];r++,k++){before[k]=tail;tail=after[k]=(uint32_t)tasks+communication[r];}
      before[k]=tail;after[k++]=to[x];
    }
    /* design/plan-to-bridge-change-request.md R5 */
    for(uint64_t c=0;c<p->comms;c++){w[tasks+c]=0;resource[tasks+c]=n;}
    for(size_t x=0;x<edges;x++)for(size_t r=route[x];r<route[x+1];r++){
      const double occupy=cost[x]/rate[link[r]];
      if(occupy>w[tasks+communication[r]])w[tasks+communication[r]]=occupy;
      resource[tasks+communication[r]]=n+link[r];
    }
    /* design/plan-to-bridge-change-request.md R5 */
    status=mesh_graph_completion_times((uint32_t)instructions,w,resource,n+2*p->pairs,(size_t)dependencies,before,after,
                                       completion);
    /* L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, Math. Oper. Res. 22(3), 1997, §2; C.-Y. Chen,
       arXiv 2205.02474, 2022, (1a) */
    if(!status && !(p->completed=calloc((size_t)count+1,sizeof *p->completed)))status=ENOMEM;
    for(uint32_t i=0;!status && i<count;i++)for(uint32_t q=0;q<consumer[i].count;q++)
      if(completion[consumer[i].first+q]>p->completed[i])p->completed[i]=completion[consumer[i].first+q];
    if(!status){
      for(uint64_t j=0;j<instructions;j++)completion[j]=-completion[j];
      status=mesh_graph_heap_order((uint32_t)instructions,completion,(size_t)dependencies,before,after,place);
    }
  }
  if(!status){
    /* design/plan-to-bridge-change-request.md R5 */
    for(uint64_t j=0;j<p->entries;j++)p->entry[j].place=place[tasks+p->entry[j].communication];
    qsort(p->entry,p->entries,sizeof *p->entry,plan_entry_order);
  }
  free(reader);free(consumer);free(w);free(cost);free(rate);free(completion);
  free(resource);free(from);free(to);free(link);free(communication);free(before);free(after);free(place);free(route);
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
struct plan_room_rows { const size_t *order; const uint32_t *to; const uint64_t *delay; };
static uint32_t plan_room_head(const void *context,size_t k){
  const struct plan_room_rows *rows=context;
  return rows->to[rows->order[k]];
}
static uint64_t plan_room_lag(const void *context,size_t k){
  const struct plan_room_rows *rows=context;
  return rows->delay[rows->order[k]];
}
/* S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §4.1;
   design/aot-compilation-change-request.md A3, appendix item 2 */
static int plan_times(struct plan *p){
  const uint32_t n=p->n;
  size_t first[(size_t)n+2],child[(size_t)n+1];
  double *page=calloc((size_t)p->demands+1,sizeof *page);
  if(!page)return ENOMEM;
  for(uint32_t i=0;i<p->transfers;i++){
    const uint32_t d=p->of[i];
    if(p->weights[i].page>page[d])page[d]=p->weights[i].page;
    p->time[p->declared[i].destination]+=
      p->weights[i].consumed*(p->declared[i].kind==MESH_PAGES?(double)p->demand[d].instructions:1.0);
  }
  for(uint32_t i=0;i<p->demands;i++){
    const struct plan_demand *d=p->demand+i;
    p->time[d->source]+=p->weights[d->first].produced+(d->own?p->weights[d->first].read:0.0);
    for(uint32_t j=d->tree;j<d->tree+d->trees;j++){
      if(!p->run[j])continue;
      const struct plan_tree tree={p->parent+p->tree[j]*n,n,d->source};
      const double hops=(double)p->run[j]*page[i];
      mesh_graph_rows(n+1,n,plan_tree_parent,&tree,first,child);
      for(uint32_t y=0;y<n;y++)for(size_t c=first[y];c<first[y+1];c++){p->time[y]+=hops;p->time[child[c]]+=hops;}
    }
    p->forwarded+=d->forwards;
  }
  free(page);
  return 0;
}
static uint32_t plan_submission(const struct plan *p,const uint32_t *actor,uint32_t s,uint32_t j,uint32_t v){
  if(v==s)return 2*j;
  return actor[j]+(uint32_t)__builtin_popcountll(p->nodes)+plan_rank(p->nodes&~(UINT64_C(1)<<s),v);
}
/* design/synchronization-signature-change-request.md items 1, 2 and 4: the plan's own synchronization as it is
   compiled, BSL §9's added edges as `struct mesh_sync` entries. A wait is mesh-dataflow.c mesh_transfers_start's, at a
   node's host before the call's submission: the count of its entry reaching `need` (K every + base, K the block); a
   raise is mesh-records.c counts_tell's, the bridge at the edge's source node telling its plan's `submissions`; the
   count words start at 0 (mesh_plan's state cleared). Processors: node v's host 2v and bridge 2v + 1 */
struct plan_signed { struct mesh_graph_operation *operation; size_t count,held; };
static int plan_signed_put(struct plan_signed *o,struct mesh_graph_operation x){
  if(o->count==o->held){
    const size_t grown=o->held?2*o->held:256;
    struct mesh_graph_operation *more=realloc(o->operation,grown*sizeof *more);
    if(!more)return ENOMEM;
    o->operation=more;o->held=grown;
  }
  o->operation[o->count++]=x;
  return 0;
}
static uint32_t plan_tell_key(const void *context,size_t k){ return ((const uint32_t *)context)[k]; }
static int plan_signature(struct plan *plans,uint32_t count,const uint32_t *calls,uint32_t J,uint64_t program,uint32_t s,
                          uint32_t n,uint32_t V,const double *time,const struct mesh_graph_task *task,const uint32_t *node,
                          const uint32_t *call,const uint32_t *lead,const uint32_t *actor,const uint32_t *place,
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
  for(uint32_t x=0;!status && x<V;x++)
    offset[x]=(place[call[x]]+(uint64_t)lead[x])/plans[calls[call[x]]].calls;
  for(uint32_t a=0;!status && a<added;a++){
    key[a]=node[from[built+a]]*count+calls[call[from[built+a]]];
    if(delay[built+a]+1>executions)executions=delay[built+a]+1;
    status=plan_signed_put(&o,(struct mesh_graph_operation){.processor=2*node[to[built+a]],.primitive=MESH_GRAPH_INITIAL,
                                                            .vertex=MESH_GRAPH_NONE,.object=MESH_GRAPH_PLAN_COUNT+a});
  }
  if(!status)mesh_graph_rows(n*count,added,plan_tell_key,key,first,tell);
  for(uint64_t K=0;!status && K<executions;K++)for(uint32_t v=0;!status && v<n;v++){
    if(!(program>>v&1))continue;
    for(uint32_t j=0;!status && j<J;j++){
      const struct plan *p=plans+calls[j];
      if(v!=s && !(p->nodes>>v&1))continue;
      const uint32_t f=calls[j],i=place[j],u=plan_submission(p,actor,s,j,v);
      const uint64_t nf=p->calls,submitted=K*nf+i;
      const size_t *waits=wait_first+(size_t)v*J+row[f];
      for(size_t q=waits[i];!status && q<waits[i+1];q++){
        const uint32_t a=(uint32_t)wait_order[q],tail=from[built+a],every=plans[calls[call[tail]]].calls;
        const __int128 need=(__int128)K*every+(__int128)((place[call[tail]]+(uint64_t)lead[tail])%every)+1-
                            (__int128)added_delay[a]*every;
        status=plan_signed_put(&o,(struct mesh_graph_operation){.processor=2*v,.primitive=MESH_GRAPH_HOST_WAIT,
                                                                .vertex=to[built+a],.object=MESH_GRAPH_PLAN_COUNT+a,
                                                                .execution=K,
                                                                .value=need>0?(uint64_t)need:0});
      }
      if(!status)status=plan_signed_put(&o,(struct mesh_graph_operation){.processor=2*v,
                                        .primitive=MESH_GRAPH_BRIDGE_START,.vertex=u,
                                        .object=MESH_GRAPH_PLAN_SUBMISSION+f,.execution=K,
                                        .value=submitted+1});
      for(size_t q=first[(size_t)v*count+f];!status && q<first[(size_t)v*count+f+1];q++){
        const uint32_t a=(uint32_t)tell[q],tail=from[built+a];
        const int ended=submitted>=lead[tail];
        status=plan_signed_put(&o,(struct mesh_graph_operation){.processor=2*v+1,.primitive=MESH_GRAPH_BRIDGE_RAISE,
                                                                .vertex=ended?tail:u,.object=MESH_GRAPH_PLAN_COUNT+a,
                                                                .execution=ended?(submitted-lead[tail])/nf:K,
                                                                .value=submitted+1});
      }
    }
  }
  if(!status)status=mesh_graph_signature(2*n,processor,V,task,built,dependence,o.count,o.operation,&signature);
  if(!status && (status=mesh_graph_resynchronized(&signature,time,built,from,to,delay,added,from+built,to+built,
                                                  added_delay,lambda,offset)))
    mesh_graph_signature_free(&signature);
  if(!status){
    struct mesh_graph_scope *scope=calloc(o.count+1,sizeof *scope);
    if(!scope)status=ENOMEM;
    else{
      for(size_t k=0;k<o.count;k++)if(o.operation[k].vertex<V &&
        (o.operation[k].primitive==MESH_GRAPH_HOST_WAIT || o.operation[k].primitive==MESH_GRAPH_BRIDGE_START))
        scope[k].once=task[o.operation[k].vertex].label+1;
      status=mesh_graph_signature_program(&signature,0,NULL,0,NULL,scope);free(scope);
    }
    if(status)mesh_graph_signature_free(&signature);
  }
  if(!status)**kept=signature;
  free(dependence);free(offset);free(key);free(first);free(tell);free(o.operation);
  return status;
}
/* design/aot-compilation-change-request.md A10; A. J. Bernstein, 1966; M. C. Rinard and P. C. Diniz, 1997, §2 */
static uint32_t plan_chain(const uint32_t *chains,uint32_t j){ return chains?chains[j]:j; }
static void plan_ordered(struct plan_room_edges *graph,const struct plan *plans,const uint32_t *calls,
                         const uint32_t *actor,uint32_t a,uint32_t b,uint64_t d){
  const struct plan *p=plans+calls[a],*q=plans+calls[b];
  for(uint64_t set=p->nodes&q->nodes;set;set&=set-1){
    const uint32_t v=(uint32_t)__builtin_ctzll(set);
    plan_room_edge(graph,actor[a]+plan_rank(p->nodes,v),actor[b]+plan_rank(q->nodes,v),d);
  }
}
/* design/compilation-composition-change-request.md item 6, unit 6D: vertex x is node `self`'s part of its call (a
   bridge-keyed vertex past the senders' pairs) or its submission there (a vertex keyed by its node; the sender's
   2j) */
static int plan_here(const uint32_t *key,const uint32_t *node,uint32_t J,uint32_t n,uint32_t self,uint32_t x){
  return node[x]==self && ((x>=2*J && key[x]==n) || key[x]==self);
}
static int plan_graph(struct plan *plans,uint32_t count,const uint32_t *calls,uint32_t ncalls,uint64_t program,
                      uint32_t self,const size_t *row,const size_t *order,const uint32_t *orders,const uint32_t *causes,
                      uint32_t norders,const uint32_t *chains,struct mesh_planned *planned){
  const uint32_t s=(uint32_t)__builtin_ctzll(program),J=ncalls,n=plans[0].n;
  uint64_t vertices=2*(uint64_t)J,edges=0;
  uint32_t *place=malloc(((size_t)J+1)*sizeof *place),*actor=malloc(((size_t)J+1)*sizeof *actor);
  double *t=NULL;
  uint32_t *from=NULL,*to=NULL,*call=NULL,*node=NULL,*lead=NULL,*key=NULL,*added_from=NULL,*added_to=NULL,*entry=NULL;
  struct mesh_graph_task *task=NULL;
  struct mesh_planned_cause *cause=NULL;
  uint64_t *delay=NULL,*added_delay=NULL;
  size_t *first=NULL,*rows=NULL,*wait_first=NULL,*wait_order=NULL,*tell_first=NULL,*tell_order=NULL;
  int status=!place || !actor?ENOMEM:0;
  if(status)goto done;
  for(uint32_t q=0;q<norders;q++)if(causes[q]!=MESH_ORDER_FLOW && causes[q]!=MESH_ORDER_CONFLICT){status=EINVAL;goto done;}
  for(uint32_t f=0;f<count;f++)for(size_t q=row[f];q<row[f+1];q++)place[order[q]]=(uint32_t)(q-row[f]);
  for(uint32_t j=0;j<J && vertices<UINT32_MAX;j++){
    const struct plan *p=plans+calls[j];
    const uint64_t parts=(uint64_t)__builtin_popcountll(p->nodes),
                   others=(uint64_t)__builtin_popcountll(p->nodes&~(UINT64_C(1)<<s));
    actor[j]=(uint32_t)vertices;
    vertices+=parts+others;
    edges+=4+3*parts+others+2*p->forwarded;
  }
  for(uint32_t q=0;q<norders;q++){
    const uint32_t a=orders[2*q],b=orders[2*q+1];
    if(a>=b || b>=J){status=EINVAL;goto done;}
    edges+=(uint64_t)__builtin_popcountll(plans[calls[a]].nodes&plans[calls[b]].nodes);
  }
  for(uint32_t j=0;j<J;j++){
    const uint32_t h=plan_chain(chains,j);
    if(h>j || plan_chain(chains,h)!=h){status=EINVAL;goto done;}
    if(h!=j)edges+=2*(uint64_t)__builtin_popcountll(plans[calls[h]].nodes|plans[calls[j]].nodes);
  }
  if(vertices>=UINT32_MAX/4 || edges>=SIZE_MAX/(8*sizeof(uint64_t)) || (uint64_t)n*J>=UINT32_MAX){
    status=ENOMEM;goto done;
  }
  const uint32_t V=(uint32_t)vertices;
  const size_t room=(size_t)edges+2*(size_t)V;
  t=calloc((size_t)V+1,sizeof *t);
  from=malloc((room+1)*sizeof *from);
  to=malloc((room+1)*sizeof *to);
  delay=malloc((room+1)*sizeof *delay);
  call=malloc(((size_t)V+1)*sizeof *call);
  node=malloc(((size_t)V+1)*sizeof *node);
  lead=malloc(((size_t)V+1)*sizeof *lead);
  key=malloc((2*(size_t)V+1)*sizeof *key);
  added_from=malloc((2*(size_t)V+1)*sizeof *added_from);
  added_to=malloc((2*(size_t)V+1)*sizeof *added_to);
  added_delay=malloc((2*(size_t)V+1)*sizeof *added_delay);
  entry=malloc((2*(size_t)V+1)*sizeof *entry);
  first=malloc(((size_t)V+n+2)*sizeof *first);
  rows=malloc((room+1)*sizeof *rows);
  wait_first=malloc(((size_t)n*J+1)*sizeof *wait_first);
  wait_order=malloc((2*(size_t)V+1)*sizeof *wait_order);
  tell_first=malloc(((size_t)count+2)*sizeof *tell_first);
  tell_order=malloc((2*(size_t)V+1)*sizeof *tell_order);
  cause=malloc((room+1)*sizeof *cause);
  if(!t || !from || !to || !delay || !call || !node || !lead || !key || !added_from || !added_to || !added_delay ||
     !entry || !first || !rows || !wait_first || !wait_order || !tell_first || !tell_order || !cause){
    status=ENOMEM;goto done;
  }
  /* design/aot-compilation-change-request.md A10 */
  for(uint32_t j=0;j<J;j++){
    const struct plan *p=plans+calls[j];
    call[2*j]=call[2*j+1]=j;
    node[2*j]=node[2*j+1]=s;
    lead[2*j]=lead[2*j+1]=0;
    key[2*j]=s;key[2*j+1]=n;
    uint32_t k=actor[j];
    for(uint64_t set=p->nodes;set;set&=set-1,k++){
      call[k]=j;
      node[k]=(uint32_t)__builtin_ctzll(set);
      t[k]=p->time[node[k]];
      lead[k]=(uint32_t)p->outstanding;
      key[k]=n;
    }
    for(uint64_t set=p->nodes&~(UINT64_C(1)<<s);set;set&=set-1,k++){
      call[k]=j;
      node[k]=(uint32_t)__builtin_ctzll(set);
      lead[k]=0;
      key[k]=node[k];
    }
  }
  struct plan_room_edges graph={from,to,delay,0,cause,{MESH_PLANNED_PROGRAM,UINT32_MAX}};
  mesh_graph_rows(n+1,V,plan_at,key,first,rows);
  if(!(task=malloc(((size_t)V+1)*sizeof *task))){status=ENOMEM;goto done;}
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
    const struct plan *p=plans+f;
    const uint64_t nf=p->calls,lag=(i+p->outstanding)/nf,span=(i+(uint64_t)p->copies)/nf;
    const uint32_t later=(uint32_t)order[row[f]+(i+p->outstanding)%nf],
                   ahead=(uint32_t)order[row[f]+(i+(uint64_t)p->copies)%nf];
    plan_room_cause(&graph,MESH_PLANNED_PROGRAM,UINT32_MAX);
    plan_room_edge(&graph,2*j+1,2*next+1,!next);
    plan_room_cause(&graph,MESH_PLANNED_SUBMIT,UINT32_MAX);
    plan_room_edge(&graph,2*j,2*j+1,0);
    for(uint64_t set=p->nodes;set;set&=set-1){
      const uint32_t v=(uint32_t)__builtin_ctzll(set),x=actor[j]+plan_rank(p->nodes,v);
      plan_room_cause(&graph,MESH_PLANNED_SUBMIT,UINT32_MAX);
      plan_room_edge(&graph,2*j+1,x,0);
      plan_room_edge(&graph,plan_submission(p,actor,s,j,v),x,0);
      plan_room_cause(&graph,MESH_PLANNED_OUTSTANDING,UINT32_MAX);
      plan_room_edge(&graph,x,plan_submission(p,actor,s,later,v),lag);
    }
    plan_room_cause(&graph,MESH_PLANNED_OUTSTANDING,UINT32_MAX);
    if(!(p->nodes>>s&1))plan_room_edge(&graph,2*j,2*later,lag);
    for(uint32_t d=0;d<p->demands;d++){
      plan_room_cause(&graph,MESH_PLANNED_TRANSFER,d);
      for(uint32_t y=0;y<p->n;y++)for(uint64_t below=p->child[(size_t)d*p->n+y];below;below&=below-1){
        const uint32_t c=(uint32_t)__builtin_ctzll(below);
        plan_room_edge(&graph,actor[j]+plan_rank(p->nodes,y),actor[ahead]+plan_rank(p->nodes,c),span);
        plan_room_edge(&graph,actor[j]+plan_rank(p->nodes,c),actor[ahead]+plan_rank(p->nodes,y),span);
      }
    }
  }
  /* design/aot-compilation-change-request.md A10: Bernstein's order between calls, each at every node both calls'
     parts share (delay 0, one block); each commuting chain's calls one at a time in program order, its last call's
     part before the next block's first (delay 1), as BSL §4.1 closes a processor's order */
  for(uint32_t q=0;q<norders;q++){
    plan_room_cause(&graph,causes[q],q);
    plan_ordered(&graph,plans,calls,actor,orders[2*q],orders[2*q+1],0);
  }
  plan_room_cause(&graph,MESH_PLANNED_CHAIN,UINT32_MAX);
  for(uint32_t j=0;j<J;j++)place[j]=j;
  for(uint32_t j=0;j<J;j++){
    const uint32_t h=plan_chain(chains,j);
    if(h==j)continue;
    plan_ordered(&graph,plans,calls,actor,place[h],j,0);
    place[h]=j;
  }
  for(uint32_t j=0;j<J;j++)if(plan_chain(chains,j)==j && place[j]!=j)plan_ordered(&graph,plans,calls,actor,place[j],j,1);
  for(uint32_t f=0;f<count;f++)for(size_t q=row[f];q<row[f+1];q++)place[order[q]]=(uint32_t)(q-row[f]);
  const size_t built=graph.edges;
  /* design/compilation-composition-change-request.md item 6, unit 6D */
  planned->call=malloc(((size_t)J+1)*sizeof *planned->call);
  planned->order=malloc((built+1)*sizeof *planned->order);
  planned->cause=malloc((built+1)*sizeof *planned->cause);
  if(!planned->call || !planned->order || !planned->cause){status=ENOMEM;goto done;}
  planned->calls=J;
  for(uint32_t j=0;j<J;j++){
    const struct plan *p=plans+calls[j];
    planned->call[j]=(struct mesh_planned_call){.lead=p->start_lead,.function=calls[j],.ordinal=place[j],
                                                .every=p->calls};
  }
  for(size_t e=0;e<built;e++)
    if(plan_here(key,node,J,n,self,from[e]) && plan_here(key,node,J,n,self,to[e])){
      planned->cause[planned->orders]=cause[e];
      planned->order[planned->orders++]=(struct mesh_graph_dependence){from[e],to[e],delay[e]};
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
  for(uint32_t a=0;a<added;a++){
    const uint32_t tail=added_from[a],head=added_to[a],j=call[tail],h=call[head];
    if(__builtin_add_overflow(added_delay[a],(place[j]+(uint64_t)lead[tail])/plans[calls[j]].calls,&delay[built+a])){
      status=ENOMEM;goto done;
    }
    from[built+a]=tail;to[built+a]=head;
    key[a]=node[head]*J+(uint32_t)row[calls[h]]+place[h];
  }
  mesh_graph_rows(n*J,added,plan_at,key,wait_first,wait_order);
  for(uint32_t q=0;q<added;q++)entry[wait_order[q]]=q;
  for(uint32_t a=0;a<added;a++)key[a]=node[added_from[a]]==self?calls[call[added_from[a]]]:count;
  mesh_graph_rows(count+1,added,plan_at,key,tell_first,tell_order);
  for(uint32_t f=0;f<count;f++){
    struct plan *p=plans+f;
    const size_t *waits=wait_first+(size_t)self*J+row[f];
    p->waits=(uint32_t)(waits[p->calls]-waits[0]);
    p->tells=(uint32_t)(tell_first[f+1]-tell_first[f]);
    p->sync=malloc(((size_t)p->waits+p->tells+1)*sizeof *p->sync);
    p->wait_row=malloc(((size_t)p->calls+1)*sizeof *p->wait_row);
    if(!p->sync || !p->wait_row){status=ENOMEM;goto done;}
    for(uint32_t r=0;r<=p->calls;r++)p->wait_row[r]=(uint32_t)(waits[r]-waits[0]);
    for(uint32_t k=0;k<p->waits;k++){
      const uint32_t a=(uint32_t)wait_order[waits[0]+k],tail=added_from[a],j=call[tail];
      const uint32_t every=plans[calls[j]].calls;
      int64_t lagged;
      if(added_delay[a]>INT64_MAX || __builtin_mul_overflow((int64_t)added_delay[a],(int64_t)every,&lagged)){
        status=ENOMEM;goto done;
      }
      p->sync[k]=(struct plan_sync){.node=node[tail],.function=calls[j]+1,.every=every,
                                    .base=(int64_t)((place[j]+(uint64_t)lead[tail])%every)+1-lagged};
    }
    for(uint32_t k=0;k<p->tells;k++){
      const uint32_t a=(uint32_t)tell_order[tell_first[f]+k],head=added_to[a],g=calls[call[head]];
      p->sync[p->waits+k]=(struct plan_sync){.node=node[head],.function=g+1,
                                             .entry=(uint32_t)(entry[a]-wait_first[(size_t)node[head]*J+row[g]])};
    }
  }
  if((status=plan_signature(plans,count,calls,J,program,s,n,V,t,task,node,call,lead,actor,place,row,built,added,from,to,
                             delay,added_delay,lambda,wait_first,wait_order,&planned->signature)))goto done;
  /* S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, (5) */
  mesh_graph_rows(V,built+added,plan_at,from,first,rows);
  const struct plan_room_rows rooms={rows,to,delay};
  for(uint32_t f=0;f<count;f++){plans[f].room=plans[f].nodes>>self&1?0:1;plans[f].submit_room=0;}
  for(uint32_t j=0;j<J;j++){
    struct plan *p=plans+calls[j];
    const int here=(int)(p->nodes>>self&1);
    const uint32_t source[2]={2*j+1,here?actor[j]+plan_rank(p->nodes,self):2*j+1},target[2]={2*j,2*j+1};
    uint64_t lags[4];
    if((status=mesh_graph_lags(V,first,plan_room_head,plan_room_lag,&rooms,2,source,2,target,lags)))goto done;
    if(lags[0]==UINT64_MAX || !lags[0] || (here && (lags[3]==UINT64_MAX || !lags[3]))){status=ENOMEM;goto done;}
    if(__builtin_add_overflow(p->submit_room,lags[0],&p->submit_room) ||
       (here && __builtin_add_overflow(p->room,lags[3],&p->room))){status=ENOMEM;goto done;}
  }
  for(uint32_t f=0;f<count;f++)
    if(plans[f].room>=UINT64_MAX/(4*sizeof(struct mesh_delivery)) ||
       plans[f].submit_room>=UINT64_MAX/(4*sizeof(struct mesh_submission))){status=ENOMEM;goto done;}
done:
  free(place);free(actor);free(t);free(from);free(to);free(delay);free(call);free(node);free(lead);free(key);
  free(added_from);free(added_to);free(added_delay);free(entry);free(first);free(rows);free(wait_first);
  free(wait_order);free(tell_first);free(tell_order);free(task);free(cause);
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
static int plan_instructions(struct plan *p,const struct hdr *m){
  const uint64_t quantum=plan_quantum(m);
  const uint32_t count=p->transfers;
  size_t *first=malloc(((size_t)p->demands+1)*sizeof *first),*order=malloc(((size_t)count+1)*sizeof *order);
  uint64_t *start=NULL,*pieces=NULL,*index=NULL,room=0,most=0,ends=0,at=0;
  int status=!first || !order?ENOMEM:0;
  if(!status){
    mesh_graph_rows(p->demands,count,plan_at,p->of,first,order);
    for(uint32_t i=0;i<p->demands;i++){
      const struct plan_demand *d=p->demand+i;
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
    struct plan_demand *d=p->demand+i;
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
    if(d->mine!=MESH_ABSENT)ends+=d->instructions;
  }
  if(!status && !(p->at=malloc((ends+1)*sizeof *p->at)))status=ENOMEM;
  at=0;
  for(uint32_t i=0;!status && i<p->demands;i++){
    struct plan_demand *d=p->demand+i;
    if(d->mine==MESH_ABSENT)continue;
    struct plan_starting starting={p->cut+d->cut,p->at+at,0,0,d->instructions};
    mesh_strided_extents(d->view,plan_start,&starting);
    d->at=at;at+=d->instructions;
  }
  free(first);free(order);free(start);free(pieces);free(index);
  return status;
}
/* MPI 4.1 §6.1.11 */
static int plan_landings(const struct plan *p){
  uint64_t count=0,k=0;
  for(uint32_t i=0;i<p->demands;i++)
    if(p->demand[i].mine!=MESH_ABSENT && p->demand[i].leaves>>p->self&1)count+=p->demand[i].instructions;
  struct mesh_graph_interval *interval=malloc((count+1)*sizeof *interval);
  if(!interval)return ENOMEM;
  for(uint32_t i=0;i<p->demands;i++){
    const struct plan_demand *d=p->demand+i;
    if(d->mine==MESH_ABSENT || !(d->leaves>>p->self&1))continue;
    for(uint64_t j=0;j<d->instructions;j++){
      const uint64_t at=p->at[d->at+j];
      interval[k++]=(struct mesh_graph_interval){d->mine,at,at+p->cut[d->cut+j+1]-p->cut[d->cut+j]};
    }
  }
  const int status=mesh_graph_intersect(count,interval)?EINVAL:0;
  free(interval);
  return status;
}

struct plan_touch { uint64_t start,end; uint32_t value,member; };
static int plan_touch_order(const void *a,const void *b){
  const struct plan_touch *x=a,*y=b;
  if(x->value!=y->value)return x->value<y->value?-1:1;
  return (x->start>y->start)-(x->start<y->start);
}
static int plan_member_order(const void *a,const void *b){
  const uint32_t x=*(const uint32_t *)a,y=*(const uint32_t *)b;
  return (x>y)-(x<y);
}
struct plan_cover { const uint32_t *member; uint32_t members,value,block; };
static int plan_same_set(const struct plan_cover *x,const struct plan_cover *y){
  return x->members==y->members && !memcmp(x->member,y->member,(size_t)x->members*sizeof *x->member);
}
static int plan_cover_order(const void *a,const void *b){
  const struct plan_cover *x=a,*y=b;
  for(uint32_t k=0;k<x->members && k<y->members;k++)
    if(x->member[k]!=y->member[k])return x->member[k]<y->member[k]?-1:1;
  if(x->members!=y->members)return x->members<y->members?-1:1;
  if(x->value!=y->value)return x->value<y->value?-1:1;
  return (x->block>y->block)-(x->block<y->block);
}
/* P. Druschel and L. L. Peterson, Fbufs, SOSP 1993; V. S. Pai, P. Druschel and W. Zwaenepoel, IO-Lite, ACM TOCS 18(1),
   2000; design/disjoint-pages-change-request.md C3.3; design/torch-capture-change-request.md item 6 */
static int plan_sets(struct plan *p,const struct hdr *m){
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
    const struct plan_demand *d=p->demand+i;
    if(d->value==MESH_ABSENT)continue;
    if(d->mine==MESH_ABSENT)relays++;
    else if(!p->given[d->mine].run)touches+=d->instructions;
  }
  struct plan_touch *touch=malloc((touches+1)*sizeof *touch);
  struct plan_cover *cover=malloc((blocks+1)*sizeof *cover);
  uint32_t *held=malloc((touches+1)*sizeof *held);
  p->block_set=malloc((blocks+1)*sizeof *p->block_set);
  p->block_element=malloc((blocks+1)*sizeof *p->block_element);
  int status=!touch || !cover || !held || !p->block_set || !p->block_element?ENOMEM:0;
  if(status)goto done;
  uint64_t t=0,covers=0,members=0;
  for(uint32_t i=0;i<p->demands;i++){
    const struct plan_demand *d=p->demand+i;
    if(d->mine==MESH_ABSENT || p->given[d->mine].run)continue;
    for(uint64_t k=0;k<d->instructions;k++){
      const uint64_t at=p->at[d->at+k];
      touch[t++]=(struct plan_touch){at,at+p->cut[d->cut+k+1]-p->cut[d->cut+k],d->mine,d->value};
    }
  }
  qsort(touch,touches,sizeof *touch,plan_touch_order);
  for(uint64_t b=0;b<blocks;b++)p->block_set[b]=MESH_ABSENT;
  for(uint64_t q=0;q<touches;){
    const uint32_t j=touch[q].value;
    const uint64_t b=touch[q].start/quantum,low=b*quantum,bytes=p->given[j].section.bytes,
                   whole=bytes-low<quantum?bytes-low:quantum,from=members;
    uint64_t reach=low,covered=0;
    for(;q<touches && touch[q].value==j && touch[q].start/quantum==b;q++){
      if(touch[q].end>reach){covered+=touch[q].end-(touch[q].start>reach?touch[q].start:reach);reach=touch[q].end;}
      held[members++]=touch[q].member;
    }
    if(covered!=whole){members=from;continue;}
    qsort(held+from,members-from,sizeof *held,plan_member_order);
    uint64_t kept=from+1;
    for(uint64_t k=from+1;k<members;k++)if(held[k]!=held[kept-1])held[kept++]=held[k];
    members=kept;
    cover[covers++]=(struct plan_cover){held+from,(uint32_t)(members-from),j,(uint32_t)b};
  }
  qsort(cover,covers,sizeof *cover,plan_cover_order);
  uint64_t sets=relays,listed=relays;
  for(uint64_t c=0;c<covers;c++)if(!c || !plan_same_set(cover+c-1,cover+c)){sets++;listed+=cover[c].members;}
  if(sets>=UINT32_MAX || listed>=UINT32_MAX){status=ENOMEM;goto done;}
  p->set_first=malloc((sets+1)*sizeof *p->set_first);
  p->member=malloc((listed+1)*sizeof *p->member);
  p->set_blocks=malloc((sets+1)*sizeof *p->set_blocks);
  p->set_far=malloc((sets+1)*sizeof *p->set_far);
  if(!p->set_first || !p->member || !p->set_blocks || !p->set_far){status=ENOMEM;goto done;}
  uint32_t s=0,at=0;
  for(uint64_t c=0;c<covers;c++){
    if(!c || !plan_same_set(cover+c-1,cover+c)){
      p->set_first[s]=at;p->set_blocks[s]=0;
      for(uint32_t k=0;k<cover[c].members;k++)p->member[at++]=cover[c].member[k];
      s++;
    }
    const uint32_t b=p->block_first[cover[c].value]+cover[c].block;
    p->block_set[b]=s-1;p->block_element[b]=p->set_blocks[s-1]++;
  }
  for(uint32_t i=0;i<p->demands;i++){
    struct plan_demand *d=p->demand+i;
    d->relay=MESH_ABSENT;
    if(d->value==MESH_ABSENT || d->mine!=MESH_ABSENT)continue;
    d->relay=s;p->set_first[s]=at;p->member[at++]=d->value;p->set_blocks[s++]=(uint32_t)plan_blocks(m,d->bytes);
  }
  p->set_first[s]=at;p->sets=s;
  for(uint32_t x=0;x<s;x++){
    p->set_far[x]=0;
    for(uint32_t k=p->set_first[x];k<p->set_first[x+1];k++)
      if(p->farthest[p->member[k]]>p->set_far[x])p->set_far[x]=p->farthest[p->member[k]];
  }
  for(uint32_t j=0;j<p->givens;j++){
    if(p->block_first[j]==MESH_ABSENT)continue;
    const uint64_t own=plan_blocks(m,p->given[j].section.bytes);
    uint32_t alone=0;
    for(uint64_t k=0;k<own;k++)
      if(p->block_set[p->block_first[j]+k]==MESH_ABSENT)p->block_element[p->block_first[j]+k]=alone++;
    p->uncovered[j]=alone;
  }
done:
  free(touch);free(cover);free(held);
  return status;
}

/* N. Garg and J. Koenemann, FOCS 1998 */
static int plan_function(struct plan *p,const struct mesh_function *function,const struct mesh_nodes *nodes,
                         uint64_t program){
  struct hdr *m=p->context->M;
  const struct mesh_declared *declared=function->declared;
  const struct mesh_weight *weights=function->weights;
  const struct mesh_dependence *dependence=function->dependence;
  struct mesh_value *values=function->value;
  const uint32_t count=function->count,given=function->values,dependences=function->dependences;
  if((count && (!declared || !weights)) || (given && !values) || (dependences && !dependence))return EINVAL;
  /* design/disjoint-pages-change-request.md D.3.3 */
  for(uint32_t j=0;j<given;j++){
    const struct mesh_value *v=values+j;
    if(!v->section.bytes || v->kind>MESH_EXECUTION || v->wire>1 || v->state>1 || v->run>1 ||
       (v->state && (v->wire || v->kind!=MESH_INVARIANT)) ||
       (v->lag && (v->kind!=MESH_EXECUTION || v->state)))return EINVAL;
    if(plan_blocks(m,v->section.bytes)>=UINT32_MAX/m->block)return ENOMEM;
  }
  /* design/plan-to-bridge-change-request.md R4; design/aot-compilation-change-request.md A3 */
  for(uint32_t i=0;i<count;i++){
    const struct mesh_weight *x=weights+i;
    if(declared[i].kind>MESH_PAGES || !(x->produced>0 && x->produced<=DBL_MAX) ||
       !(x->consumed>0 && x->consumed<=DBL_MAX) || !(x->read>0 && x->read<=DBL_MAX) ||
       !(x->page>0 && x->page<=DBL_MAX))return EINVAL;
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
  p->declared=declared;p->transfers=count;p->given=values;p->givens=given;p->weights=weights;
  p->dependence=dependence;p->dependences=dependences;p->outstanding=function->outstanding;
  struct plan_key *key=malloc(((size_t)count+1)*sizeof *key);
  uint32_t *root=NULL,*demand=NULL,*placed=malloc(((size_t)count+1)*sizeof *placed);
  size_t *trees=NULL;
  double *up=NULL,*down=NULL,*share=NULL,*rate=NULL,*latency=NULL;
  int status=0;
  p->named=calloc((size_t)given+1,sizeof *p->named);
  p->uncovered=malloc(((size_t)given+1)*sizeof *p->uncovered);
  p->item=malloc(((size_t)given+1)*sizeof *p->item);
  p->demand=calloc((size_t)count+1,sizeof *p->demand);
  p->of=malloc(((size_t)count+1)*sizeof *p->of);
  if(!key || !placed || !p->named || !p->uncovered || !p->item || !p->demand || !p->of){status=ENOMEM;goto done;}
  for(uint32_t j=0;j<given;j++)p->uncovered[j]=(uint32_t)plan_blocks(m,values[j].section.bytes);
  for(uint32_t i=0;i<count;i++){
    const struct mesh_declared *t=declared+i;
    uint64_t from,to,end;
    if(t->source>=n || t->destination>=n){status=EHOSTUNREACH;goto done;}
    if(t->source==t->destination || plan_view(&t->from,&from,&end) || plan_view(&t->to,&to,&end) || from!=to){
      status=EINVAL;goto done;
    }
    if(plan_blocks(m,from)>=UINT32_MAX/m->block){status=ENOMEM;goto done;}
    key[i]=(struct plan_key){t->source,t->identity,i};
  }
  qsort(key,count,sizeof *key,plan_key_order);
  for(uint32_t k=0;k<count;k++)
    placed[key[k].index]=!k || key[k].source!=key[k-1].source || key[k].identity!=key[k-1].identity?key[k].index:
                         placed[key[k-1].index];
  for(uint32_t i=0;i<count;i++){
    const struct mesh_declared *t=declared+i;
    uint64_t bytes,end;
    if(placed[i]==i){
      plan_view(&t->from,&bytes,&end);
      p->demand[p->demands++]=(struct plan_demand){.source=t->source,.identity=t->identity,.first=i,.mine=MESH_ABSENT,
                                                   .set=1,.bytes=bytes,.range=MESH_ABSENT,.relay=MESH_ABSENT};
    }
    p->of[i]=placed[i]==i?p->demands-1:p->of[placed[i]];
    struct plan_demand *d=p->demand+p->of[i];
    /* design/aot-compilation-change-request.md, appendix item 5; design/plan-to-bridge-change-request.md R5 */
    if(t->kind!=MESH_PAGES)d->set=0;
    if(!plan_alike(&t->from,&declared[d->first].from) || d->leaves>>t->destination&1 ||
       weights[i].produced!=weights[d->first].produced || weights[i].read!=weights[d->first].read){
      status=EINVAL;goto done;
    }
    d->leaves|=UINT64_C(1)<<t->destination;
    /* design/plan-to-bridge-change-request.md R2; torch 2.14.0 torch/export/exported_program.py, ExportedProgram;
       Xu et al., GSPMD, arXiv 2105.04663, §4.5 */
    if(t->source!=p->self && t->destination!=p->self){
      if(t->value!=MESH_ABSENT){status=EINVAL;goto done;}
      continue;
    }
    if(t->value>=given){status=EINVAL;goto done;}
    const struct mesh_value *v=values+t->value;
    const struct mesh_strided *view=t->source==p->self?&t->from:&t->to;
    plan_view(view,&bytes,&end);
    if(end>v->section.bytes || v->state || v->kind!=MESH_EXECUTION || (t->destination==p->self && v->lag) ||
       (t->source==p->self && d->mine!=MESH_ABSENT && d->mine!=t->value)){
      status=EINVAL;goto done;
    }
    d->mine=t->value;
    d->view=view;
  }
  for(uint32_t i=0;i<p->demands;i++)if(p->demand[i].mine!=MESH_ABSENT)p->named[p->demand[i].mine]=1;
  /* ExecuTorch exir/memory_planning.py, update_tensor_lifetime */
  for(uint32_t j=0;j<given;j++)
    if(values[j].kind==MESH_EXECUTION && !values[j].lag && values[j].begin>values[j].end){
      status=EINVAL;goto done;
    }
  for(uint32_t i=0;i<p->demands;i++)if(p->demand[i].source==p->self)p->demand[i].release=(uint32_t)p->releases++;
  if((status=plan_instructions(p,m)) || (status=plan_landings(p)))goto done;

  if(p->demands){
    for(uint32_t a=0;a<n;a++)for(uint32_t b=a+1;b<n;b++){
      const uint64_t c=nodes->capacity[a][b]<nodes->capacity[b][a]?nodes->capacity[a][b]:nodes->capacity[b][a];
      if(c){p->from[p->pairs]=a;p->to[p->pairs]=b;p->capacity[p->pairs]=(double)c;p->pairs++;}
    }
    const uint32_t E=p->pairs;
    const struct plan_edges graph={p->from,p->to};
    uint32_t order[n];
    unsigned char reached[n];
    uint64_t reach[n],known=0;
    for(uint32_t i=0;i<p->demands;i++){
      const uint32_t s=p->demand[i].source;
      if(!(known>>s&1)){
        mesh_graph_reach(n,E,plan_end,&graph,s,order,reached,NULL);
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
      struct plan_demand *d=p->demand+i;
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
  free(key);free(placed);free(root);free(up);free(down);free(demand);free(share);free(rate);free(latency);free(trees);
  return status;
}

static void plan_free(struct plan *p){
  free(p->named);free(p->uncovered);free(p->item);free(p->sync);free(p->cut);free(p->at);free(p->set_first);
  free(p->member);free(p->set_blocks);free(p->set_far);free(p->block_first);free(p->block_set);free(p->block_element);
  free(p->wait_row);free(p->demand);free(p->of);free(p->parent);
  free(p->tree);free(p->run);free(p->members);free(p->lead);free(p->comm);free(p->child);free(p->value);free(p->read);
  free(p->here);free(p->first);free(p->head);free(p->lag);free(p->between);free(p->farthest);free(p->entry);
  free(p->completed);
}

/* os/os_sync_wait_on_address.h, os_sync_wait_on_address_with_timeout; L. Lamport, Distributed Computing 1(2), 1986 */
static int plan_released(struct hdr *m,uint64_t deadline){
  _Atomic uint64_t *released=&mesh_algo(m)->released;
  for(;;){
    const uint64_t seen=atomic_load_explicit(released,memory_order_acquire),now=clock_gettime_nsec_np(CLOCK_MONOTONIC);
    if(!mesh_algo_unreleased(m))return 0;
    if(now>=deadline)return ETIMEDOUT;
    os_sync_wait_on_address_with_timeout(released,seen,sizeof *released,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                                         OS_CLOCK_MACH_ABSOLUTE_TIME,deadline-now);
  }
}

static void plan_bind_values(struct mesh_ctx *context,const struct mesh_function *functions,uint32_t count,const uint32_t *page){
  struct hdr *m=context->M;
  const uint64_t block=plan_quantum(m);
  for(uint32_t f=0;f<count;f++)for(uint32_t j=0;j<functions[f].values;j++){
    const struct mesh_value *v=functions[f].value+j;
    const struct mesh_section x=v->section;
    for(uint32_t c=0;c<x.count;c++)for(uint32_t k=0;k<x.stride;k++){
      const uint32_t r=x.first+c*x.stride+k;
      atomic_store_explicit(&mesh_page(m)[r].device,0,memory_order_relaxed);
      mesh_buffers(m)[r].constant=0;
      mesh_backing_bind(context,r,m->block,page[r],c);
      if(v->state){const uint64_t left=v->section.bytes-(uint64_t)k*block;memset(mesh_at(m,page[r]),0,(size_t)(left<block?left:block));}
    }
  }
}
static int plan_publish(struct mesh_ctx *context,uint64_t group,const struct mesh_function *functions,uint32_t count,
  uint32_t table,const struct mesh_nodes *nodes,struct mesh_planned *planned){
  struct mesh_nodes after;
  int status=mesh_nodes_read(context,&after);
  if(status)return status;
  if(mesh_view_compare((struct mesh_view_id){.id=after.serial,.sequencer=after.sequencer,.boot=after.boot},
      (struct mesh_view_id){.id=nodes->serial,.sequencer=nodes->sequencer,.boot=nodes->boot}))return ESTALE;
  struct hdr *m=context->M;
  struct mesh_algo *algo=mesh_algo(m);
  algo->plans=count;algo->nodes=nodes->count;algo->group=group;
  algo->table=m->data_off+(uint64_t)table*m->pgsz;
  algo->adjacent=algo->table+(uint64_t)count*sizeof(uint64_t);
  atomic_store_explicit(&m->client,context->client,memory_order_relaxed);
  atomic_store_explicit(&m->client_start,context->start,memory_order_relaxed);
  atomic_store_explicit(&algo->published,atomic_fetch_add(&m->serial,1)+1,memory_order_release);
  for(uint32_t j=0;j<planned->calls;j++)planned->call[j].plan=functions[planned->call[j].function].plan;
  mesh_notify(&m->plans);
  return 0;
}

void mesh_planned_free(struct mesh_planned *planned){
  if(planned->signature)mesh_graph_signature_free(planned->signature);
  free(planned->call);free(planned->order);free(planned->signature);free(planned->cause);
  *planned=(struct mesh_planned){0};
}

/* design/aot-compilation-change-request.md A7 to A10, appendix item 6;
   design/plan-to-bridge-change-request.md R3 to R5; M. Cowan et al., MSCCLang, ASPLOS 2023, §5.2;
   P. E. McKenney and J. D. Slingwine, PDCS 1998 */
int mesh_plan(struct mesh_ctx *context,uint64_t group,uint64_t program,struct mesh_function *functions,uint32_t count,
              const uint32_t *calls,uint32_t ncalls,const uint32_t *orders,const uint32_t *causes,uint32_t norders,
              const uint32_t *chains,uint64_t deadline,struct mesh_planned *planned){
  struct hdr *m=context->M;
  if(!planned)return EINVAL;
  *planned=(struct mesh_planned){0};
  if(!group || !program || !count || !functions || !ncalls || !calls || (norders && (!orders || !causes)))return EINVAL;
  for(uint32_t j=0;j<ncalls;j++)if(calls[j]>=count)return EINVAL;
  /* design/torch-capture-change-request.md item 2 and its 2026-11-09 appendix: the horizon and the locations of the
     values the host writes, Bff of the host -> device edges of the calls' compiled parts in program order from their
     t(v), with each call's own dependences between its parts (mesh_graph_buffer_bound) */
  uint32_t horizon=1;
  {
    uint64_t actors=0,dependences=0;
    for(uint32_t j=0;j<ncalls;j++){
      const struct mesh_function *f=functions+calls[j];
      if(f->parts && (!f->host || !f->work))return EINVAL;
      if(f->edges && !f->edge)return EINVAL;
      for(uint32_t e=0;e<f->edges;e++)if(f->edge[e].from>=2*(uint64_t)f->parts || f->edge[e].to>=2*(uint64_t)f->parts)return EINVAL;
      actors+=f->parts?f->parts:1;dependences+=f->edges;
    }
    if(actors>=UINT32_MAX/2 || dependences>=SIZE_MAX/sizeof(struct mesh_graph_dependence))return ENOMEM;
    const uint32_t A=(uint32_t)actors;
    double *host=calloc((size_t)A+1,sizeof *host),*work=calloc((size_t)A+1,sizeof *work);
    uint64_t *bff=malloc(((size_t)A+1)*sizeof *bff),*lagged=calloc((size_t)count+1,sizeof *lagged);
    uint32_t *of=malloc(((size_t)A+1)*sizeof *of);
    struct mesh_graph_dependence *edge=malloc(((size_t)dependences+1)*sizeof *edge);
    int bounded=host && work && bff && lagged && of && edge?0:ENOMEM;
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
    if(!bounded)bounded=mesh_graph_buffer_bound(A,host,work,made,edge,bff);
    for(uint32_t q=0;!bounded && q<A;q++){
      if(bff[q]>=UINT32_MAX)bounded=ENOMEM;
      else{
        if(bff[q]>horizon)horizon=(uint32_t)bff[q];
        if(bff[q]>lagged[of[q]])lagged[of[q]]=bff[q];
      }
    }
    for(uint32_t f=0;!bounded && f<count;f++)
      for(uint32_t j=0;j<functions[f].values;j++)if(functions[f].value[j].lag)functions[f].value[j].lag=(uint32_t)lagged[f];
    free(host);free(work);free(bff);free(lagged);free(of);free(edge);
    if(bounded)return bounded;
  }
  struct plan *plans=calloc(count,sizeof *plans);
  size_t *row=malloc(((size_t)count+1)*sizeof *row),*order=malloc(((size_t)ncalls+1)*sizeof *order),*rows_by=NULL,
         *listed=NULL;
  struct plan_placement placed={0};
  struct mesh_nodes nodes;
  uint32_t *page=NULL,*adjacency=NULL,*toward=NULL,*place=NULL;
  int status=!plans || !row || !order?ENOMEM:0;
  if(status)goto done;
  mesh_graph_rows(count,ncalls,plan_at,calls,row,order);
  for(uint32_t f=0;f<count;f++)if(row[f]==row[f+1]){status=EINVAL;goto done;}
  if((status=mesh_nodes_read(context,&nodes)))goto done;
  if(nodes.self>=nodes.count){status=EHOSTUNREACH;goto done;}
  if(!(program>>nodes.self&1)){status=EINVAL;goto done;}
  const uint64_t block=plan_quantum(m);
  /* design/aot-compilation-change-request.md A7 */
  for(uint32_t f=0;f<count;f++){
    struct plan *p=plans+f;
    p->context=context;p->n=nodes.count;p->self=nodes.self;
    p->view=(struct mesh_view_id){.id=nodes.serial,.sequencer=nodes.sequencer,.boot=nodes.boot};
    p->calls=(uint32_t)(row[f+1]-row[f]);p->horizon=horizon;
    if((uint64_t)horizon*p->calls>=UINT32_MAX){status=ENOMEM;goto done;}
    if((status=plan_function(p,functions+f,&nodes,program)))goto done;
  }
  if((status=plan_graph(plans,count,calls,ncalls,program,nodes.self,row,order,orders,causes,norders,chains,planned)))
    goto done;
  planned->horizon=horizon;
  uint64_t rows=0;
  for(uint32_t f=0;f<count;f++)for(uint32_t j=0;j<plans[f].givens;j++){
    struct mesh_value *v=plans[f].given+j;
    const uint32_t P=(uint32_t)plan_blocks(m,v->section.bytes),copies=plan_copies(plans+f,j);
    uint64_t blocks,end;
    if(__builtin_mul_overflow((uint64_t)copies,(uint64_t)P,&blocks) || __builtin_add_overflow(rows,blocks,&end) ||
       end>mesh_rows(m)){status=ENOMEM;goto done;}
    v->section=(struct mesh_section){(uint32_t)rows,P*m->block,v->section.bytes,copies,P};
    rows=end;
  }
  for(uint32_t f=0;f<count;f++)plan_layout(plans+f);
  if((status=plan_place(plans,count,m,&placed)))goto done;
  for(uint32_t f=0;f<count;f++){
    struct plan *p=plans+f;
    if(!(p->entry=malloc((p->entries+1)*sizeof *p->entry))){status=ENOMEM;goto done;}
    plan_walk(p);
    if(p->demands && (status=plan_schedule(p,p->weights,p->dependence,p->dependences)))goto done;
  }
  if(!(page=malloc(((size_t)rows+1)*sizeof *page))){status=ENOMEM;goto done;}
  if((status=plan_rows(plans,count,m,&placed,page,rows)))goto done;
  const uint32_t n=nodes.count;
  uint64_t places=0;
  for(uint32_t f=0;f<count;f++)places+=(uint64_t)__builtin_popcountll(plans[f].toward);
  adjacency=malloc(((size_t)n+1+places)*sizeof *adjacency);
  toward=malloc((places+1)*sizeof *toward);
  place=malloc((places+1)*sizeof *place);
  rows_by=malloc(((size_t)n+1)*sizeof *rows_by);
  listed=malloc((places+1)*sizeof *listed);
  if(!adjacency || !toward || !place || !rows_by || !listed){status=ENOMEM;goto done;}
  for(uint32_t f=0,q=0;f<count;f++)
    for(uint64_t s=plans[f].toward;s;s&=s-1){toward[q]=(uint32_t)__builtin_ctzll(s);place[q++]=f;}
  mesh_graph_rows(n,places,plan_at,toward,rows_by,listed);
  for(uint32_t i=0;i<=n;i++)adjacency[i]=(uint32_t)rows_by[i];
  for(uint64_t q=0;q<places;q++)adjacency[n+1+q]=place[listed[q]];
  if((status=plan_released(m,deadline)))goto done;

  for(uint32_t f=0;f<count;f++){
    struct plan *p=plans+f;
    const uint32_t u=p->copies,at=plan_page(&placed,m->block,p->state,0);
    const uint64_t base=m->data_off+(uint64_t)at*m->pgsz;
    p->base=base;
    memset(mesh_at(m,at),0,(size_t)p->hops);
    struct mesh_state *state=(struct mesh_state *)mesh_at(m,at);
    state->copies=u;
    state->receives=(uint32_t)p->receives;
    state->ranges=p->consumed;
    state->releases=(uint32_t)p->releases;
    state->phases=base+p->phases;
    state->counts=base+p->counts;
    state->completions=base+p->completions;
    state->supplied=base+p->supplied;
    state->transfers=base+p->publications;
    state->status=base+p->status;
    state->group=group;
    state->epoch=(uint64_t)f+1;
    state->serial=p->view.id;state->sequencer=p->view.sequencer;state->boot=p->view.boot;
    state->hops=base+p->hops;
    state->entries=p->entries*u;
    state->nodes=p->spanned;
    state->sender=(uint32_t)__builtin_ctzll(program);
    state->outstanding=(uint32_t)p->outstanding;
    state->room=p->room;
    state->submit_room=p->submit_room;
    state->deliveries=base+p->deliveries;
    state->submits=base+p->submits;
    state->waits=p->waits;
    state->tells=p->tells;
    state->syncs=base+p->syncs;
    state->calls=p->calls;
    state->rows=base+p->rows;
    state->times=base+p->times;
    if(p->completed)memcpy(mesh_at(m,at)+p->times,p->completed,(size_t)p->transfers*sizeof *p->completed);
    struct mesh_sync *sync=mesh_syncs(m,state);
    for(uint32_t k=0;k<p->waits+p->tells;k++){
      sync[k].node=p->sync[k].node;sync[k].function=p->sync[k].function;sync[k].entry=p->sync[k].entry;
      sync[k].every=p->sync[k].every;sync[k].base=p->sync[k].base;
    }
    memcpy(mesh_at(m,at)+p->rows,p->wait_row,((size_t)p->calls+1)*sizeof *p->wait_row);
    for(uint32_t i=0;i<p->transfers;i++){
      const struct mesh_declared *t=p->declared+i;
      const struct plan_demand *d=p->demand+p->of[i];
      struct mesh_publication *entry=(struct mesh_publication *)(void *)(mesh_at(m,at)+p->publications+
                                                                         (size_t)i*sizeof(struct mesh_publication));
      if(t->source==p->self){
        entry->targets[0]=(struct mesh_target){.word=plan_supplied(p,d),.count=1,.stride=0};
        entry->state=base;
      }else if(t->destination==p->self){
        entry->pages=d->received;
        entry->phases=plan_group(p,d,0);
        entry->count=plan_count(p,d,0);
        entry->state=base;
      }
      if(d->sending!=MESH_ABSENT){
        entry->sends=d->sends;
        entry->completion=plan_completion(p,d,0);
        entry->state=base;
      }
    }
    /* design/plan-to-bridge-change-request.md R2, R5; design/aot-compilation-change-request.md A9 */
    struct mesh_hop *hop=mesh_hops(m,state);
    for(uint32_t c=0;c<u;c++)for(uint64_t j=0;j<p->entries;j++){
      const struct plan_entry *e=p->entry+j;
      const struct plan_demand *d=p->demand+e->demand;
      const uint64_t *cut=p->cut+d->cut+e->instruction;
      uint64_t byte=cut[0];
      uint32_t frame;
      if(d->mine==MESH_ABSENT)frame=plan_page(&placed,m->block,p->set_item+d->relay*u+c,byte/block);
      else{
        const struct mesh_section x=p->given[d->mine].section;
        byte=p->at[d->at+e->instruction];
        frame=page[x.first+(size_t)(c%x.count)*x.stride+byte/block];
      }
      struct mesh_hop h={.page=frame,.node=e->node,.direction=e->direction,.field=MESH_ABSENT,
                         .bytes=(uint32_t)(cut[1]-cut[0]),.serial=p->view.id,.set=e->set,
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
      if(e->direction==MESH_SEND)h.count=plan_completion(p,d,c);
      *hop++=h;
    }
  }
  const uint32_t table=plan_page(&placed,m->block,placed.table,0);
  uint64_t *word=(uint64_t *)(void *)mesh_at(m,table);
  for(uint32_t f=0;f<count;f++)word[f]=plans[f].base;
  memcpy(word+count,adjacency,((size_t)n+1+places)*sizeof *adjacency);
  for(uint32_t f=0;f<count;f++)functions[f].plan=plans[f].base;
  plan_bind_values(context,functions,count,page);
  status=plan_publish(context,group,functions,count,table,&nodes,planned);

done:
  if(status)mesh_planned_free(planned);
  for(uint32_t f=0;plans && f<count;f++)plan_free(plans+f);
  free(plans);free(row);free(order);free(placed.item);free(placed.count);free(placed.chain);free(placed.at);free(page);
  free(adjacency);free(toward);free(place);free(rows_by);free(listed);
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
static void plan_hash(CC_SHA256_CTX *hash,const void *data,size_t bytes){
  const unsigned char *p=data;
  while(bytes){const CC_LONG n=(CC_LONG)(bytes>UINT32_MAX?UINT32_MAX:bytes);CC_SHA256_Update(hash,p,n);p+=n;bytes-=n;}
}
static void plan_hash_words(CC_SHA256_CTX *hash,const uint64_t *word,size_t n){plan_hash(hash,word,n*sizeof *word);}
static int plan_inputs(struct mesh_ctx *context,uint64_t program,const struct mesh_function *functions,uint32_t count,
  const uint32_t *calls,uint32_t ncalls,const uint32_t *orders,const uint32_t *causes,uint32_t norders,
  const uint32_t *chains,struct mesh_nodes *nodes,unsigned char digest[32]){
  int status=mesh_nodes_read(context,nodes);
  if(status)return status;
  CC_SHA256_CTX hash;CC_SHA256_Init(&hash);
  plan_hash_words(&hash,(uint64_t[]){context->M->pgsz,context->M->block,program,count,ncalls,norders,nodes->count,nodes->self},8);
  plan_hash(&hash,nodes->node,nodes->count*sizeof *nodes->node);
  for(uint32_t n=0;n<nodes->count;n++)plan_hash(&hash,nodes->capacity[n],nodes->count*sizeof(uint64_t));
  for(uint32_t f=0;f<count;f++){
    const struct mesh_function *x=functions+f;
    plan_hash_words(&hash,(uint64_t[]){x->count,x->values,x->dependences,x->outstanding,x->parts,x->edges},6);
    plan_hash(&hash,x->weights,(size_t)x->count*sizeof *x->weights);
    plan_hash(&hash,x->dependence,(size_t)x->dependences*sizeof *x->dependence);
    plan_hash(&hash,x->host,(size_t)x->parts*sizeof *x->host);
    plan_hash(&hash,x->work,(size_t)x->parts*sizeof *x->work);
    plan_hash(&hash,x->edge,(size_t)x->edges*sizeof *x->edge);
    for(uint32_t v=0;v<x->values;v++){
      const struct mesh_value *y=x->value+v;
      plan_hash_words(&hash,(uint64_t[]){y->section.bytes,y->kind,y->wire,y->state,y->begin,y->end,!!y->lag,y->run},8);
    }
    for(uint32_t t=0;t<x->count;t++){
      const struct mesh_declared *y=x->declared+t;
      plan_hash_words(&hash,(uint64_t[]){y->source,y->destination,y->identity,y->kind,y->value},5);
      const struct mesh_strided *view[2]={&y->from,&y->to};
      for(uint32_t k=0;k<2;k++){
        plan_hash_words(&hash,(uint64_t[]){view[k]->offset,view[k]->element,view[k]->dims},3);
        plan_hash(&hash,view[k]->pairs,2*(size_t)view[k]->dims*sizeof(uint64_t));
      }
    }
  }
  plan_hash(&hash,calls,(size_t)ncalls*sizeof *calls);
  plan_hash(&hash,orders,2*(size_t)norders*sizeof *orders);plan_hash(&hash,causes,(size_t)norders*sizeof *causes);
  for(uint32_t j=0;j<ncalls;j++)plan_hash_words(&hash,(uint64_t[]){plan_chain(chains,j)},1);
  CC_SHA256_Final(digest,&hash);
  return 0;
}
struct plan_saved {
  uint64_t magic,bytes,program,rows,table,table_bytes,signature,horizon,orders;
  uint32_t functions,calls,pgsz,block;
  struct mesh_range arena[2];
  unsigned char inputs[32];
};
struct plan_saved_function { uint64_t page,bytes; uint32_t values,transfers; };
#define PLAN_SAVED_MAGIC UINT64_C(0x314e414c5048534d)
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
static uint32_t plan_saved_page(const struct plan_saved *head,const struct hdr *m,uint32_t page,int *status){
  for(uint32_t s=0;s<2;s++)if(page>=head->arena[s].first && page-head->arena[s].first<head->arena[s].count){
    const struct mesh_range now=mesh_arena_range(m,(int)s);
    const uint32_t relative=page-head->arena[s].first;
    if(relative>=now.count){*status=ESTALE;return 0;}
    return now.first+relative;
  }
  *status=EINVAL;return 0;
}
static void plan_static_state(struct mesh_state *to,const struct mesh_state *s){
  *to=(struct mesh_state){.copies=s->copies,.receives=s->receives,.ranges=s->ranges,.releases=s->releases,
    .phases=s->phases,.counts=s->counts,.supplied=s->supplied,.status=s->status,.completions=s->completions,
    .hops=s->hops,.entries=s->entries,.nodes=s->nodes,.sender=s->sender,.outstanding=s->outstanding,.room=s->room,
    .deliveries=s->deliveries,.submits=s->submits,.waits=s->waits,.tells=s->tells,.syncs=s->syncs,.calls=s->calls,
    .rows=s->rows,.submit_room=s->submit_room,.transfers=s->transfers,.times=s->times};
}
static int plan_snapshot(struct hdr *m,const struct mesh_function *function,void **bytes,uint64_t *length){
  const uint64_t base=function->plan;
  const struct mesh_state *given=mesh_state_at(m,base);
  if(!given || given->times<base || (uint64_t)function->count>(UINT64_MAX-given->times)/sizeof(double))return EINVAL;
  const uint64_t end=given->times+(uint64_t)function->count*sizeof(double);
  if(end>m->length || end-base>SIZE_MAX)return EINVAL;
  *length=end-base;
  if(posix_memalign(bytes,64,(size_t)plan_align(*length)))return ENOMEM;
  memset(*bytes,0,(size_t)plan_align(*length));
  struct mesh_state *saved=*bytes;
  plan_static_state(saved,given);plan_shift_state(saved,base,0);
  struct mesh_publication *publication=(void *)((char *)saved+saved->transfers);
  memcpy(publication,(char *)m+given->transfers,(size_t)function->count*sizeof *publication);
  for(uint32_t t=0;t<function->count;t++){
    plan_shift_word(&publication[t].phases,base,0);plan_shift_word(&publication[t].count,base,0);
    plan_shift_word(&publication[t].completion,base,0);plan_shift_word(&publication[t].targets[0].word,base,0);
    publication[t].state=publication[t].state?1:0;
  }
  const struct mesh_sync *source=mesh_syncs(m,given);
  struct mesh_sync *sync=(void *)((char *)saved+saved->syncs);
  for(uint32_t k=0;k<given->waits+given->tells;k++)
    sync[k]=(struct mesh_sync){.node=source[k].node,.function=source[k].function,.entry=source[k].entry,
      .every=source[k].every,.base=source[k].base};
  memcpy((char *)saved+saved->rows,(char *)m+given->rows,((size_t)given->calls+1)*sizeof(uint32_t));
  struct mesh_hop *hop=(void *)((char *)saved+saved->hops);
  memcpy(hop,mesh_hops(m,given),(size_t)given->entries*sizeof *hop);
  for(uint64_t h=0;h<given->entries;h++){
    plan_shift_word(&hop[h].word,base,0);plan_shift_word(&hop[h].count,base,0);hop[h].serial=0;
  }
  memcpy((char *)saved+saved->times,(char *)m+given->times,(size_t)function->count*sizeof(double));
  return 0;
}
int mesh_plan_save(struct mesh_ctx *context,uint64_t program,const struct mesh_function *functions,uint32_t count,
  const uint32_t *calls,uint32_t ncalls,const uint32_t *orders,const uint32_t *causes,uint32_t norders,
  const uint32_t *chains,const struct mesh_planned *planned,void **bytes,size_t *length){
  *bytes=NULL;*length=0;
  struct hdr *m=context->M;
  struct mesh_nodes nodes;
  struct plan_saved head={.magic=PLAN_SAVED_MAGIC,.program=program,.functions=count,.calls=ncalls,.orders=planned->orders,
    .pgsz=m->pgsz,.block=m->block,.horizon=planned->horizon,.arena={mesh_arena_range(m,0),mesh_arena_range(m,1)}};
  int status=plan_inputs(context,program,functions,count,calls,ncalls,orders,causes,norders,chains,&nodes,head.inputs);
  if(status)return status;
  struct mesh_algo *algo=mesh_algo(m);
  if(algo->plans!=count || algo->nodes!=nodes.count || planned->calls!=ncalls)return ESTALE;
  const uint32_t *adjacency=(const void *)((const char *)m+algo->adjacent);
  head.table=(algo->table-m->data_off)/m->pgsz;
  head.table_bytes=(uint64_t)count*sizeof(uint64_t)+((uint64_t)nodes.count+1+adjacency[nodes.count])*sizeof(uint32_t);
  for(uint32_t f=0;f<count;f++){
    const struct mesh_state *state=mesh_state_at(m,functions[f].plan);
    if(!state || state->serial!=nodes.serial || state->sequencer!=nodes.sequencer || state->boot!=nodes.boot)return ESTALE;
    for(uint32_t v=0;v<functions[f].values;v++){
      const struct mesh_section x=functions[f].value[v].section;
      const uint64_t end=x.first+(uint64_t)x.count*x.stride;
      if(end>head.rows)head.rows=end;
    }
  }
  size_t signature_bytes=0;
  status=mesh_graph_archive(planned->signature,NULL,0,&signature_bytes);
  head.signature=signature_bytes;
  if(status!=ERANGE)return status?status:EINVAL;
  struct plan_archive out={0};
  plan_archive_bytes(&out,&head,sizeof head);
  for(uint32_t f=0;f<count && !out.status;f++){
    struct plan_saved_function function={.page=(functions[f].plan-m->data_off)/m->pgsz,
      .values=functions[f].values,.transfers=functions[f].count};
    void *state=NULL;
    out.status=plan_snapshot(m,functions+f,&state,&function.bytes);
    plan_archive_bytes(&out,&function,sizeof function);
    plan_archive_bytes(&out,functions[f].value,(size_t)function.values*sizeof(struct mesh_value));
    plan_archive_bytes(&out,state,(size_t)function.bytes);free(state);
  }
  for(uint64_t r=0;r<head.rows;r++){
    uint32_t page=(uint32_t)atomic_load_explicit(&mesh_page(m)[r].mapping,memory_order_relaxed);
    plan_archive_bytes(&out,&page,sizeof page);
  }
  plan_archive_bytes(&out,(void *)adjacency,(size_t)(head.table_bytes-(uint64_t)count*sizeof(uint64_t)));
  for(uint32_t c=0;c<ncalls;c++){
    struct mesh_planned_call call=planned->call[c];call.plan=0;
    plan_archive_bytes(&out,&call,sizeof call);
  }
  plan_archive_bytes(&out,planned->order,planned->orders*sizeof *planned->order);
  plan_archive_bytes(&out,planned->cause,planned->orders*sizeof *planned->cause);
  void *signature=malloc((size_t)head.signature);
  if(!signature)out.status=ENOMEM;
  if(!out.status){size_t made=0;out.status=mesh_graph_archive(planned->signature,signature,(size_t)head.signature,&made);}
  plan_archive_bytes(&out,signature,(size_t)head.signature);free(signature);
  if(out.status){free(out.bytes);return out.status;}
  head.bytes=out.size;memcpy(out.bytes,&head,sizeof head);
  *bytes=out.bytes;*length=out.size;return 0;
}
static int plan_saved_span(uint64_t offset,uint64_t count,uint64_t width,uint64_t length){
  return offset<=length && count<=(length-offset)/width;
}
int mesh_plan_load(struct mesh_ctx *context,uint64_t group,uint64_t program,struct mesh_function *functions,uint32_t count,
  const uint32_t *calls,uint32_t ncalls,const uint32_t *orders,const uint32_t *causes,uint32_t norders,
  const uint32_t *chains,uint64_t deadline,const void *bytes,size_t length,struct mesh_planned *planned){
  *planned=(struct mesh_planned){0};
  struct hdr *m=context->M;
  struct plan_archive input={.bytes=(void *)bytes,.size=length,.read=1};
  struct plan_saved head={0};
  plan_archive_bytes(&input,&head,sizeof head);
  if(input.status || head.magic!=PLAN_SAVED_MAGIC || head.bytes!=length || head.program!=program ||
     head.functions!=count || head.calls!=ncalls || head.pgsz!=m->pgsz || head.block!=m->block || !head.horizon ||
     head.rows>mesh_rows(m) || head.orders>SIZE_MAX/sizeof(struct mesh_graph_dependence) || !group)return EINVAL;
  struct mesh_nodes nodes;
  unsigned char digest[32];
  int status=plan_inputs(context,program,functions,count,calls,ncalls,orders,causes,norders,chains,&nodes,digest);
  if(status)return status;
  if(memcmp(digest,head.inputs,sizeof digest))return ESTALE;
  for(uint32_t s=0;s<2;s++){
    const struct mesh_range range=mesh_arena_range(m,(int)s);
    if((head.arena[0].first==head.arena[1].first)!=(mesh_arena_range(m,0).first==mesh_arena_range(m,1).first) ||
       range.count<head.arena[s].count)return ESTALE;
  }
  struct plan_saved_function *saved=calloc(count,sizeof *saved);
  void **state=calloc(count,sizeof *state);
  struct mesh_value **value=calloc(count,sizeof *value);
  uint32_t *page=malloc(((size_t)head.rows+1)*sizeof *page);
  const uint64_t table_prefix=(uint64_t)count*sizeof(uint64_t);
  uint32_t table=plan_saved_page(&head,m,(uint32_t)head.table,&status);
  uint64_t table_at=m->data_off+(uint64_t)table*m->pgsz;
  if(head.table>UINT32_MAX || head.table_bytes<table_prefix+((uint64_t)nodes.count+1)*sizeof(uint32_t) ||
     table_at>m->length || head.table_bytes>m->length-table_at)status=EINVAL;
  const size_t adjacency_bytes=status?0:(size_t)(head.table_bytes-table_prefix);
  uint32_t *adjacency=malloc(adjacency_bytes+1);
  void *signature=NULL;
  if(!saved || !state || !value || !page || !adjacency)status=ENOMEM;
  for(uint32_t f=0;!status && f<count;f++){
    plan_archive_bytes(&input,saved+f,sizeof *saved);
    if(input.status){status=input.status;break;}
    if(saved[f].values!=functions[f].values || saved[f].transfers!=functions[f].count ||
       saved[f].bytes<sizeof(struct mesh_state) || saved[f].bytes>input.size-input.at || saved[f].page>UINT32_MAX){status=EINVAL;break;}
    value[f]=malloc(((size_t)saved[f].values+1)*sizeof **value);
    if(!value[f] || posix_memalign(state+f,64,(size_t)plan_align(saved[f].bytes))){status=ENOMEM;break;}
    plan_archive_bytes(&input,value[f],(size_t)saved[f].values*sizeof **value);
    plan_archive_bytes(&input,state[f],(size_t)saved[f].bytes);
    if(input.status){status=input.status;break;}
    struct mesh_state *s=state[f];
    const uint64_t size=saved[f].bytes;
    for(size_t a=0;a<sizeof plan_addresses/sizeof *plan_addresses;a++)
      if(*(uint64_t *)(void *)((char *)s+plan_addresses[a])>size)status=EINVAL;
    if(!s->copies || !plan_saved_span(s->transfers,saved[f].transfers,sizeof(struct mesh_publication),size) ||
       !plan_saved_span(s->syncs,(uint64_t)s->waits+s->tells,sizeof(struct mesh_sync),size) ||
       !plan_saved_span(s->rows,(uint64_t)s->calls+1,sizeof(uint32_t),size) ||
       !plan_saved_span(s->hops,s->entries,sizeof(struct mesh_hop),size) ||
       !plan_saved_span(s->times,saved[f].transfers,sizeof(double),size))status=EINVAL;
    for(uint32_t v=0;v<saved[f].values;v++){
      const struct mesh_section x=value[f][v].section;
      if((uint64_t)x.first+(uint64_t)x.count*x.stride>head.rows)status=EINVAL;
    }
    saved[f].page=plan_saved_page(&head,m,(uint32_t)saved[f].page,&status);
    const uint64_t base=m->data_off+saved[f].page*m->pgsz;
    if(base>m->length || size>m->length-base)status=ESTALE;
  }
  if(status)goto done;
  plan_archive_bytes(&input,page,(size_t)head.rows*sizeof *page);
  plan_archive_bytes(&input,adjacency,adjacency_bytes);
  planned->calls=ncalls;planned->orders=(size_t)head.orders;planned->horizon=head.horizon;
  planned->call=malloc(((size_t)ncalls+1)*sizeof *planned->call);
  planned->order=malloc((planned->orders+1)*sizeof *planned->order);
  planned->cause=malloc((planned->orders+1)*sizeof *planned->cause);
  planned->signature=calloc(1,sizeof *planned->signature);
  if(!planned->call || !planned->order || !planned->cause || !planned->signature){status=ENOMEM;goto done;}
  plan_archive_bytes(&input,planned->call,(size_t)ncalls*sizeof *planned->call);
  plan_archive_bytes(&input,planned->order,planned->orders*sizeof *planned->order);
  plan_archive_bytes(&input,planned->cause,planned->orders*sizeof *planned->cause);
  if(head.signature>input.size-input.at){status=EINVAL;goto done;}
  signature=malloc((size_t)head.signature);
  if(!signature){status=ENOMEM;goto done;}
  plan_archive_bytes(&input,signature,(size_t)head.signature);
  if(input.status || input.at!=input.size){status=EINVAL;goto done;}
  if((status=mesh_graph_archive_read(signature,(size_t)head.signature,planned->signature)))goto done;
  if((uint64_t)nodes.count+1+adjacency[nodes.count]!=adjacency_bytes/sizeof *adjacency){status=EINVAL;goto done;}
  for(uint32_t j=0;j<ncalls;j++)if(planned->call[j].function!=calls[j]){status=ESTALE;goto done;}
  for(uint64_t r=0;r<head.rows;r++)page[r]=plan_saved_page(&head,m,page[r],&status);
  for(uint32_t f=0;!status && f<count;f++){
    struct mesh_state *s=state[f];
    const uint64_t base=m->data_off+saved[f].page*m->pgsz;
    struct mesh_publication *publication=(void *)((char *)s+s->transfers);
    for(uint32_t t=0;t<saved[f].transfers;t++){
      plan_shift_word(&publication[t].phases,0,base);plan_shift_word(&publication[t].count,0,base);
      plan_shift_word(&publication[t].completion,0,base);plan_shift_word(&publication[t].targets[0].word,0,base);
      publication[t].state=publication[t].state?base:0;
    }
    struct mesh_hop *hop=(void *)((char *)s+s->hops);
    for(uint64_t h=0;h<s->entries;h++){
      plan_shift_word(&hop[h].word,0,base);plan_shift_word(&hop[h].count,0,base);
      hop[h].page=plan_saved_page(&head,m,hop[h].page,&status);hop[h].serial=nodes.serial;
    }
    plan_shift_state(s,0,base);
    s->group=group;s->epoch=(uint64_t)f+1;s->serial=nodes.serial;s->sequencer=nodes.sequencer;s->boot=nodes.boot;
  }
  if(status || (status=plan_released(m,deadline)))goto done;
  for(uint32_t f=0;f<count;f++){
    functions[f].plan=m->data_off+saved[f].page*m->pgsz;
    memcpy((char *)m+functions[f].plan,state[f],(size_t)saved[f].bytes);
    memcpy(functions[f].value,value[f],(size_t)saved[f].values*sizeof **value);
  }
  plan_bind_values(context,functions,count,page);
  uint64_t *table_words=(void *)((char *)m+table_at);
  for(uint32_t f=0;f<count;f++)table_words[f]=functions[f].plan;
  memcpy(table_words+count,adjacency,adjacency_bytes);
  status=plan_publish(context,group,functions,count,table,&nodes,planned);
done:
  if(status)mesh_planned_free(planned);
  for(uint32_t f=0;f<count;f++){if(state)free(state[f]);if(value)free(value[f]);}
  free(saved);free(state);free(value);free(page);free(adjacency);free(signature);
  return status;
}
