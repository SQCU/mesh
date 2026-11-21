/* The switched bridges of N nodes over a timed wire, in one process, reentrant (design/switched-multipath-dma.md
   "Tests"; rdma/tests/route's environment).  One world is a fabric's nodes, each a region made as the bridge makes it
   (mesh.h mesh_layout) and served by the bridge's own functions (mesh-switched.h: staging, the draw, relays by name,
   delivery, the estimators, the refresh, the policy's refresh), and the wire between them.  Nothing here routes:
   every decision about a piece is the bridge's.  Its two callers are tests/sim-switched.c (make check: the payload
   checked) and tests/route/sim.c (the routing policy's training environment); no state is global, so any number of
   worlds run at once, a thread each.

   The wire is the hardware's: every cable its own lane each way at its share of the pair's rate and the pair's
   latency, a SEND waiting for a receive posted at the far end (the transport's credit), a cable sending while its
   reverse is busy at the fabric's duplex share, every SEND one whole slot.  Each node's bridge stages what its
   clients' programs release, forwards what lands, posts its lanes' outboxes and reposts its receives; copies cost the
   node's CPU time (copy_gbps).  Every refresh_ns each node's bridge refreshes its weights (Thompson sampling, eta 0.5;
   `fixed`: the unloaded shortest-delay weights, set once).

   The clients are collectives, each as libnccl-mesh declares one: a program toward every other rank it exchanges
   with, both ends' tables equal with directions swapped, one execution, its own key:
   - all-gather: each rank's S bytes, as pages, to every other rank (one span to several: the bridge's multicast);
   - all-reduce: page p's partial from every rank to its owner (rank p mod n), which sums it once its partials have
     landed and opens its reduced page's gate; the reduced page then goes to every other rank;
   - scatter: the root's S bytes for each other rank, to it.
   The relays declare nothing: they forward by name.  With `payload` 0 (training) the spans alias a few pages (each
   page's offset its own, so the bridge's multicast sees exactly the spans that would be equal), and nothing is summed
   or checked; with 1 (make check) every byte is real and sw_world_check verifies it.

   Every record the bridges log (mesh-switched.h struct mesh_switch_record) goes to `on_record` as it is drained, in
   memory; nothing here writes to storage. */
#ifndef SWITCHED_WORLD_H
#define SWITCHED_WORLD_H
#include "../mesh-switched.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SW_CABLES 8u
#define SW_POSTED 32u        /* receives a lane keeps posted */
#define SW_OPS 32u           /* collectives live at once (a program slot each) */
#define SW_PAGE 16384u
enum { SW_ALLGATHER, SW_ALLREDUCE, SW_SCATTER };

struct sw_link { uint32_t from,to,cables; double rate,latency; };   /* nodes 0..n-1; rate bytes a ns, all cables; latency ns */
struct sw_lane { uint32_t node,peer,cable,link,reverse; double free_at; uint32_t posted[SW_POSTED*4]; uint32_t head,tail; };
struct sw_block { uint64_t at,bytes; };
struct sw_node {
  struct hdr *m; struct mesh_switch *sw; uint64_t bump,end; double cpu;
  uint32_t lane_at[MESH_SWITCH_LANES];   /* switch lane -> world lane */
  uint32_t *held,nheld,room;
  struct sw_block *free; uint32_t nfree,freeroom;
  uint64_t junk,sink;                    /* payload 0: the aliased send pages and the receive page */
};
enum { SW_EV_DONE=1, SW_EV_LAND, SW_EV_OP, SW_EV_REFRESH };
struct sw_event { double t; uint32_t kind,lane,slot,to_slot; double posted,start; };
/* a collective: its kind, root, bytes a rank, pages, arrival and end (ns), its key and program slot, per rank its
   buffers and words */
struct sw_op {
  int kind,root,live,started; uint64_t S,P,key; uint32_t slot; double arrival,end;
  uint64_t input[MESH_SWITCH_N],output[MESH_SWITCH_N],scratch[MESH_SWITCH_N][MESH_SWITCH_N],words[MESH_SWITCH_N],words_bytes;
  uint64_t blocks[MESH_SWITCH_N][4*MESH_SWITCH_N+8][2]; uint32_t nblocks[MESH_SWITCH_N];
  uint64_t next_reduce[MESH_SWITCH_N];
};
struct sw_world {
  uint32_t nodes,ranks,nlinks,nlanes,ncompute,compute[MESH_SWITCH_N],rank_of[MESH_SWITCH_N];
  struct sw_link link[64]; double duplex,copy_gbps,refresh_ns; int fixed,payload; uint32_t draws;
  struct sw_node node[MESH_SWITCH_N];
  struct sw_lane lane[256];
  struct sw_event *heap; size_t events,room;
  struct sw_op op[SW_OPS]; uint32_t nlive;
  double now,next_refresh; uint64_t seed,keys,count[4];   /* count: events, draws, (unused), SENDs */
  void (*on_record)(void *ctx,const struct mesh_switch_record *r); void *ctx;
  void (*on_op)(void *ctx,uint32_t op,double end); void *op_ctx;
};

/* ---- memory in a node's region: a bump with a free list (first fit) ---- */
static inline uint64_t sw_alloc(struct sw_node *v,uint64_t bytes){
  bytes=(bytes+16383)/16384*16384;
  for(uint32_t i=0;i<v->nfree;i++)if(v->free[i].bytes>=bytes){
    const uint64_t at=v->free[i].at;v->free[i].at+=bytes;v->free[i].bytes-=bytes;
    if(!v->free[i].bytes)v->free[i]=v->free[--v->nfree];
    memset((char *)v->m+at,0,bytes);
    return at;
  }
  const uint64_t at=v->bump;v->bump+=bytes;
  if(v->bump>v->end){fprintf(stderr,"switched-world: node %u region full\n",v->m->node);abort();}
  memset((char *)v->m+at,0,bytes);   /* a run made new reuses the last one's pages */
  return at;
}
static inline void sw_release(struct sw_node *v,uint64_t at,uint64_t bytes){
  bytes=(bytes+16383)/16384*16384;
  if(v->nfree==v->freeroom){v->freeroom=v->freeroom?2*v->freeroom:64;v->free=realloc(v->free,v->freeroom*sizeof *v->free);if(!v->free)abort();}
  v->free[v->nfree++]=(struct sw_block){at,bytes};
}
static inline void *sw_at(struct sw_node *v,uint64_t off){ return (char *)v->m+off; }
static inline uint64_t sw_word(struct sw_world *w,uint32_t v,uint64_t off){ return atomic_load((_Atomic uint64_t *)sw_at(w->node+v,off)); }

/* ---- the world made: n nodes (route numbering 0..n-1, node numbers i + 1), the ranks (bits), the links ---- */
static inline struct sw_world *sw_world_new(uint32_t nodes,uint32_t ranks,const struct sw_link *links,uint32_t nlinks,double duplex,
                                            uint32_t slots,uint64_t region_mb,int payload){
  struct sw_world *w=calloc(1,sizeof *w);if(!w)abort();
  w->nodes=nodes;w->ranks=ranks;w->nlinks=nlinks;memcpy(w->link,links,nlinks*sizeof *links);w->duplex=duplex;w->payload=payload;
  w->copy_gbps=24;w->refresh_ns=20e3;w->draws=32;
  for(uint32_t i=0;i<nodes;i++){w->rank_of[i]=UINT32_MAX;if(ranks>>i&1){w->rank_of[i]=w->ncompute;w->compute[w->ncompute++]=i;}}
  for(uint32_t i=0;i<nodes;i++){
    struct hdr h;memset(&h,0,sizeof h);
    const uint64_t length=mesh_layout(&h,SW_PAGE,1,(uint32_t)(region_mb*64));
    struct sw_node *v=w->node+i;
    v->m=calloc(1,length);if(!v->m)abort();
    memcpy(v->m,&h,sizeof h);v->m->magic=MESH_MAGIC;v->m->node=i+1;
    v->bump=v->m->data_off;v->end=v->m->length;
    v->sw=mesh_switch_of(v->m);
    const uint64_t sb=mesh_switch_slot_bytes(v->m);
    mesh_switch_init(v->sw,slots,sb,sw_alloc(v,(uint64_t)slots*sb));
    v->junk=sw_alloc(v,SW_PAGE+64*8192);v->sink=sw_alloc(v,SW_PAGE);
    atomic_store(&v->sw->logging,1);
    uint32_t k=0;
    for(uint32_t j=0;j<nodes;j++)if(j!=i){struct mesh_node *x=mesh_node_at(v->m,k++);atomic_store(&x->node,j+1);atomic_store(&x->live,1);atomic_store(&x->generation,1);}
    uint32_t nums[MESH_SWITCH_N];struct mesh_switch_fabric_link fl[64];
    for(uint32_t j=0;j<nodes;j++)nums[j]=j+1;
    for(uint32_t l=0;l<nlinks;l++)fl[l]=(struct mesh_switch_fabric_link){links[l].from+1,links[l].to+1,links[l].cables,links[l].rate,links[l].latency};
    mesh_switch_view_set(v->sw,i+1,nums,nodes,fl,nlinks);
    v->sw->view.ranks=ranks;
    for(uint32_t l=0;l<MESH_SWITCH_LANES;l++)v->lane_at[l]=UINT32_MAX;
  }
  /* lanes: a cable each way; node v's k-th out-link its port k, cable c the port's lane c */
  uint32_t ports[MESH_SWITCH_N]={0};
  for(uint32_t l=0;l<nlinks;l++){
    const uint32_t port=ports[links[l].from]++;
    for(uint32_t c=0;c<links[l].cables && c<SW_CABLES;c++){
      struct sw_lane *x=w->lane+w->nlanes;
      *x=(struct sw_lane){.node=links[l].from,.peer=links[l].to,.cable=c,.link=l,.reverse=UINT32_MAX};
      const uint32_t sl=port*2*MESH_LINK_STREAMS+c;
      struct mesh_switch_lane *s=w->node[links[l].from].sw->lane+sl;
      atomic_store(&s->peer,links[l].to+1);atomic_store(&s->live,1);
      w->node[links[l].from].lane_at[sl]=w->nlanes;
      w->nlanes++;
    }
  }
  for(uint32_t a=0;a<w->nlanes;a++)for(uint32_t b=0;b<w->nlanes;b++)
    if(w->lane[b].node==w->lane[a].peer && w->lane[b].peer==w->lane[a].node && w->lane[b].cable==w->lane[a].cable)w->lane[a].reverse=b;
  return w;
}
static inline void sw_world_free(struct sw_world *w){
  for(uint32_t i=0;i<w->nodes;i++){free(w->node[i].m);free(w->node[i].held);free(w->node[i].free);}
  free(w->heap);free(w);
}
static inline uint32_t sw_switch_lane(const struct sw_world *w,uint32_t lane){
  const struct sw_node *v=w->node+w->lane[lane].node;
  for(uint32_t l=0;l<MESH_SWITCH_LANES;l++)if(v->lane_at[l]==lane)return l;
  abort();
}

/* ---- events ---- */
static inline void sw_push(struct sw_world *w,struct sw_event e){
  if(w->events==w->room){w->room=w->room?2*w->room:4096;w->heap=realloc(w->heap,w->room*sizeof *w->heap);if(!w->heap)abort();}
  size_t i=w->events++;
  while(i && w->heap[(i-1)/2].t>e.t){w->heap[i]=w->heap[(i-1)/2];i=(i-1)/2;}
  w->heap[i]=e;
}
static inline struct sw_event sw_pop(struct sw_world *w){
  struct sw_event top=w->heap[0],last=w->heap[--w->events];size_t i=0;
  for(;;){size_t c=2*i+1;if(c>=w->events)break;if(c+1<w->events && w->heap[c+1].t<w->heap[c].t)c++;if(w->heap[c].t>=last.t)break;w->heap[i]=w->heap[c];i=c;}
  if(w->events)w->heap[i]=last;
  return top;
}

/* ---- the run made new: every node's switch reset (its view, lanes and pool kept, its lines and logs cleared), no
   programs, the lanes free from each one's backlog (bytes queued at 0: `background` by link, NULL none) ---- */
static inline void sw_world_reset(struct sw_world *w,const double *background,uint64_t seed){
  for(uint32_t i=0;i<w->nodes;i++){
    struct sw_node *v=w->node+i;
    const uint64_t base=v->sw->slot_base,sb=v->sw->slot_bytes;const uint32_t slots=v->sw->slots;
    struct mesh_switch_view view=v->sw->view;struct mesh_switch_lane lane[MESH_SWITCH_LANES];memcpy(lane,v->sw->lane,sizeof lane);
    mesh_switch_init(v->sw,slots,sb,base);
    memcpy((char *)&v->sw->view+8,(char *)&view+8,sizeof view-8);
    for(uint32_t l=0;l<v->sw->view.links;l++){atomic_store(&v->sw->view.link[l].line,0);atomic_store(&v->sw->view.link[l].stamp,0);}
    for(uint32_t l=0;l<MESH_SWITCH_LANES;l++){atomic_store(&v->sw->lane[l].live,atomic_load(&lane[l].live));atomic_store(&v->sw->lane[l].peer,atomic_load(&lane[l].peer));}
    atomic_store(&v->sw->logging,1);
    for(uint32_t n=0;n<MESH_NODES;n++){struct mesh_node *x=mesh_node_at(v->m,n);for(uint32_t p=0;p<MESH_PROGRAMS;p++)atomic_store(&x->program[p].owner,0);}
    v->nheld=0;v->cpu=0;v->nfree=0;v->bump=v->junk+SW_PAGE+64*8192+SW_PAGE;v->bump=(v->bump+16383)/16384*16384;
  }
  for(uint32_t l=0;l<w->nlanes;l++){
    struct sw_lane *x=w->lane+l;x->head=x->tail=0;
    x->free_at=background?background[x->link]/w->link[x->link].rate:0;
  }
  memset(w->op,0,sizeof w->op);w->nlive=0;w->events=0;w->now=0;w->next_refresh=0;w->seed=seed;memset(w->count,0,sizeof w->count);
  for(uint32_t i=0;i<w->nodes;i++)mesh_switch_refresh(w->node[i].sw,NULL,w->fixed?1:w->draws,1,seed^i,(double)w->node[i].sw->slot_bytes,w->fixed);
  if(!w->fixed){w->next_refresh=w->refresh_ns;sw_push(w,(struct sw_event){w->refresh_ns,SW_EV_REFRESH,0,0,0,0,0});}
}

/* ---- the collectives ---- */
static inline uint64_t sw_owned(const struct sw_world *w,const struct sw_op *o,uint32_t r){ uint64_t c=0;for(uint64_t p=0;p<o->P;p++)if(p%w->ncompute==r)c++;return c; }
static inline uint64_t sw_page_bytes(const struct sw_op *o,uint64_t p){ return p+1<o->P?SW_PAGE:o->S-(uint64_t)SW_PAGE*(o->P-1); }
static inline uint64_t sw_op_alloc(struct sw_world *w,struct sw_op *o,uint32_t a,uint64_t bytes){
  const uint64_t at=sw_alloc(w->node+a,bytes);
  o->blocks[a][o->nblocks[a]][0]=at;o->blocks[a][o->nblocks[a]][1]=bytes;o->nblocks[a]++;
  return at;
}
/* a send span: the real page, or (payload 0) its alias, its own offset among the aliased pages (`salt`) */
static inline struct mesh_dma_span sw_send_span(struct sw_world *w,struct sw_op *o,uint32_t a,uint64_t base,uint64_t p,uint64_t salt){
  if(w->payload)return (struct mesh_dma_span){base+p*SW_PAGE,sw_page_bytes(o,p)};
  return (struct mesh_dma_span){w->node[a].junk+64*(salt%8192),sw_page_bytes(o,p)};
}
static inline struct mesh_dma_span sw_recv_span(struct sw_world *w,struct sw_op *o,uint32_t a,uint64_t base,uint64_t p){
  if(w->payload)return (struct mesh_dma_span){base+p*SW_PAGE,sw_page_bytes(o,p)};
  return (struct mesh_dma_span){w->node[a].sink,sw_page_bytes(o,p)};
}
/* op o's program at node a toward node b, slot o->slot, both directions' tables */
static inline void sw_declare(struct sw_world *w,struct sw_op *o,uint32_t a,uint32_t b){
  struct sw_node *v=w->node+a;
  const uint32_t ra=w->rank_of[a],rb=w->rank_of[b],n=w->ncompute;
  const uint64_t P=o->P,room=2*P+4;
  const uint64_t tables=sw_op_alloc(w,o,a,2*room*sizeof(struct mesh_crossing)),spans=sw_op_alloc(w,o,a,4*room*sizeof(struct mesh_dma_span));
  struct mesh_crossing *t[2]={sw_at(v,tables),(struct mesh_crossing *)sw_at(v,tables)+room};
  struct mesh_dma_span *sp=sw_at(v,spans);
  uint64_t nt[2]={0,0},pieces[2]={0,0},k=0;
  const uint64_t W=o->words[a];
  const uint64_t landed=W+8*(1+b),sent=W+8*(1+MESH_SWITCH_N+b),reduced=W+8*(1+2*MESH_SWITCH_N+b),partial=W+8*(4*MESH_SWITCH_N+b*(P+1)),gate=W;
  if(o->kind==SW_ALLGATHER || o->kind==SW_SCATTER){
    const int sends=o->kind==SW_ALLGATHER || (int)a==o->root,receives=o->kind==SW_ALLGATHER || (int)b==o->root;
    if(sends){
      t[0][0]=(struct mesh_crossing){.spans=spans+k*16,.units=P,.first=0,.done=sent,.done_value=1,.done_step=1};
      for(uint64_t p=0;p<P;p++)sp[k++]=o->kind==SW_ALLGATHER?sw_send_span(w,o,a,o->input[a],p,p):sw_send_span(w,o,a,o->input[a]+rb*P*SW_PAGE,p,rb*P+p);
      nt[0]=1;pieces[0]=P;
    }
    if(receives){
      t[1][0]=(struct mesh_crossing){.spans=spans+k*16,.units=P,.first=0,.done=landed,.done_value=1,.done_step=1};
      for(uint64_t p=0;p<P;p++)sp[k++]=sw_recv_span(w,o,a,o->scratch[a][b],p);
      nt[1]=1;pieces[1]=P;
    }
  }else{
    /* send: the partials of b's pages, then a's reduced pages (each its gate); receive: the partials of a's pages,
       then b's reduced pages; a transfer a page */
    const uint64_t na=sw_owned(w,o,ra),nb=sw_owned(w,o,rb);
    uint64_t q=0,i=0,j=0;
    for(uint64_t p=0;p<P;p++)if(p%n==rb){t[0][nt[0]]=(struct mesh_crossing){.spans=spans+k*16,.units=1,.first=pieces[0],.done=sent,.done_value=++q,.done_step=nb+na};
      sp[k++]=sw_send_span(w,o,a,o->input[a],p,p);nt[0]++;pieces[0]++;}
    for(uint64_t p=0;p<P;p++)if(p%n==ra){t[0][nt[0]]=(struct mesh_crossing){.spans=spans+k*16,.units=1,.first=pieces[0],.gate=gate,.gate_base=i,.gate_step=na,
      .done=sent,.done_value=++q,.done_step=nb+na};sp[k++]=sw_send_span(w,o,a,o->output[a],p,P+p);nt[0]++;pieces[0]++;i++;}
    for(uint64_t p=0;p<P;p++)if(p%n==ra){t[1][nt[1]]=(struct mesh_crossing){.spans=spans+k*16,.units=1,.first=pieces[1],.done=partial+(p/n)*8,.done_value=1,.done_step=1};
      sp[k++]=sw_recv_span(w,o,a,o->scratch[a][b],p);nt[1]++;pieces[1]++;}
    for(uint64_t p=0;p<P;p++)if(p%n==rb){t[1][nt[1]]=(struct mesh_crossing){.spans=spans+k*16,.units=1,.first=pieces[1],.done=reduced,.done_value=++j,.done_step=nb};
      sp[k++]=sw_recv_span(w,o,a,o->output[a],p);nt[1]++;pieces[1]++;}
  }
  struct mesh_node *x=mesh_switch_order(v->m,b+1);
  struct mesh_declared *d=x->program+o->slot;
  memset(d,0,sizeof *d);
  d->program=(struct mesh_declaration){.key=o->key,.period=1,.count=1,.table={nt[0]?tables:0,nt[1]?tables+room*sizeof(struct mesh_crossing):0},
    .transfers={nt[0],nt[1]},.pieces={pieces[0],pieces[1]}};
  for(int dd=0;dd<2;dd++)if(nt[dd]){d->progress[dd]=sw_op_alloc(w,o,a,mesh_switch_progress_bytes(t[dd],nt[dd]));mesh_switch_progress_init(v->m,d,dd,t[dd],nt[dd]);}
  atomic_store(&d->closed,1);atomic_store(&d->claimed,0);atomic_store(&d->issued,1);
  atomic_store(&d->owner,1);
}
/* a collective arriving at `t`: kind, root (a node), bytes a rank (scatter: a destination), the fraction of its pages
   already delivered (skipped); its index, or UINT32_MAX where SW_OPS are live (the caller arrives it again later) */
static inline uint32_t sw_op_start(struct sw_world *w,int kind,int root,uint64_t S,double done,double t,uint64_t payload_seed){
  uint32_t o=UINT32_MAX;
  for(uint32_t i=0;i<SW_OPS;i++)if(!w->op[i].live){o=i;break;}
  if(o==UINT32_MAX)return UINT32_MAX;
  struct sw_op *x=w->op+o;memset(x,0,sizeof *x);
  x->kind=kind;x->root=root;x->live=1;x->started=1;x->arrival=t;x->end=-1;x->slot=o;x->key=mesh_switch_mix(w->seed^++w->keys)|1;
  /* pages already delivered (a collective under way when the run starts) are not declared */
  const uint64_t all=S>SW_PAGE?(S+SW_PAGE-1)/SW_PAGE:1;
  uint64_t skip=(uint64_t)(done*(double)all);if(skip>=all)skip=all-1;
  x->P=all-skip;x->S=S>(uint64_t)SW_PAGE*skip?S-(uint64_t)SW_PAGE*skip:S;
  const uint64_t P=x->P;
  const uint32_t n=w->ncompute;
  for(uint32_t ci=0;ci<n;ci++){
    const uint32_t a=w->compute[ci];
    if(w->payload){
      x->input[a]=sw_op_alloc(w,x,a,(kind==SW_SCATTER?n:1)*P*SW_PAGE);x->output[a]=sw_op_alloc(w,x,a,P*SW_PAGE);
      for(uint32_t b=0;b<w->nodes;b++)if(w->rank_of[b]!=UINT32_MAX && b!=a)x->scratch[a][b]=sw_op_alloc(w,x,a,P*SW_PAGE);
      float *in=sw_at(w->node+a,x->input[a]);
      const uint64_t count=(kind==SW_SCATTER?n:1)*x->S/4;
      for(uint64_t i=0;i<count;i++)in[i]=(float)((i*7+a*13+payload_seed*3)%1000);
    }
    x->words_bytes=8*(4*MESH_SWITCH_N+MESH_SWITCH_N*(P+1)+8);
    x->words[a]=sw_op_alloc(w,x,a,x->words_bytes);   /* this rank's words, read by its declarations next */
    for(uint32_t b=0;b<w->nodes;b++)if(w->rank_of[b]!=UINT32_MAX && b!=a){
      if(kind==SW_SCATTER && (int)a!=root && (int)b!=root)continue;
      sw_declare(w,x,a,b);
    }
  }
  w->nlive++;
  return o;
}

/* whether op o is whole: every receiving program's last done word at its value, every owner's reduction done */
static inline int sw_op_whole(struct sw_world *w,struct sw_op *o){
  for(uint32_t ci=0;ci<w->ncompute;ci++){
    const uint32_t a=w->compute[ci];
    for(uint32_t n=0;n<MESH_NODES;n++){
      struct mesh_node *x=mesh_node_at(w->node[a].m,n);
      struct mesh_declared *d=x->program+o->slot;
      if(!atomic_load(&d->owner) || d->program.key!=o->key)continue;
      if(!mesh_switch_done(w->node[a].m,d))return 0;
    }
  }
  return 1;
}
static inline void sw_op_end(struct sw_world *w,uint32_t i){
  struct sw_op *o=w->op+i;
  for(uint32_t ci=0;ci<w->ncompute;ci++){
    const uint32_t a=w->compute[ci];
    for(uint32_t n=0;n<MESH_NODES;n++){struct mesh_declared *d=mesh_node_at(w->node[a].m,n)->program+o->slot;if(d->program.key==o->key)atomic_store(&d->owner,0);}
    for(uint32_t b=0;b<o->nblocks[a];b++)sw_release(w->node+a,o->blocks[a][b][0],o->blocks[a][b][1]);
  }
  o->live=0;w->nlive--;
}

/* ---- the bridges' passes ---- */
static inline void sw_hold(struct sw_node *v,uint32_t s){
  if(v->nheld==v->room){v->room=v->room?2*v->room:256;v->held=realloc(v->held,v->room*sizeof *v->held);if(!v->held)abort();}
  v->held[v->nheld++]=s;
}
static inline void sw_copy_cost(struct sw_world *w,struct sw_node *v,double t,double bytes){ if(v->cpu<t)v->cpu=t;v->cpu+=bytes/w->copy_gbps; }
static inline void sw_drain(struct sw_world *w,uint32_t a){
  struct mesh_switch_record r;
  while(mesh_switch_log_take(w->node[a].sw,MESH_SWITCH_LOGS-1,&r)){if(r.kind==MESH_LOG_DRAW)w->count[1]++;if(w->on_record)w->on_record(w->ctx,&r);}
}
/* an owner's reductions: page by page in order, once each partial of the page has landed */
static inline void sw_client_pass(struct sw_world *w,uint32_t a,double t){
  if(w->rank_of[a]==UINT32_MAX)return;
  struct sw_node *v=w->node+a;const uint32_t r=w->rank_of[a],n=w->ncompute;
  for(uint32_t oi=0;oi<SW_OPS;oi++){
    struct sw_op *o=w->op+oi;
    if(!o->live || o->kind!=SW_ALLREDUCE)continue;
    /* this rank's words (its gate the first of them): found from its program's table */
    struct mesh_declared *d=NULL;
    for(uint32_t k=0;k<MESH_NODES && !d;k++){struct mesh_declared *q=mesh_node_at(v->m,k)->program+o->slot;if(atomic_load(&q->owner) && q->program.key==o->key)d=q;}
    if(!d)continue;
    const struct mesh_crossing *t0=mesh_switch_table(v->m,d,0);
    const uint64_t owned=sw_owned(w,o,r);
    uint64_t gate=0;for(uint64_t k=0;k<d->program.transfers[0] && t0;k++)if(t0[k].gate){gate=t0[k].gate;break;}
    if(!owned)continue;
    const uint64_t W=gate;
    for(;;){
      const uint64_t i=o->next_reduce[a];
      if(i>=owned)break;
      const uint64_t p=r+i*n;
      int ready=1;
      for(uint32_t cj=0;cj<n && ready;cj++){const uint32_t b=w->compute[cj];if(b!=a && sw_word(w,a,W+8*(4*MESH_SWITCH_N+b*(o->P+1))+i*8)<1)ready=0;}
      if(!ready)break;
      if(w->payload){
        float *out=sw_at(v,o->output[a]+p*SW_PAGE);const float *in=sw_at(v,o->input[a]+p*SW_PAGE);
        const uint64_t count=sw_page_bytes(o,p)/4;
        for(uint64_t e=0;e<count;e++){float sum=in[e];for(uint32_t cj=0;cj<n;cj++){const uint32_t b=w->compute[cj];if(b!=a)sum+=((float *)sw_at(v,o->scratch[a][b]+p*SW_PAGE))[e];}out[e]=sum;}
      }
      sw_copy_cost(w,v,t,(double)sw_page_bytes(o,p)*n);
      o->next_reduce[a]++;
      mesh_switch_raise(v->m,W,o->next_reduce[a]);
    }
  }
}
static inline uint64_t sw_clock(const struct sw_node *v,double t){ return (uint64_t)(v->cpu>t?v->cpu:t); }
/* receives: each lane keeps SW_POSTED slots of its receiver's pool posted */
static inline void sw_repost(struct sw_world *w,struct sw_lane *x){
  struct sw_node *r=w->node+x->peer;
  while(x->tail-x->head<SW_POSTED){
    const uint32_t s=mesh_switch_take(r->sw);
    if(s==UINT32_MAX)return;
    atomic_store(&r->sw->slot[s].state,SLOT_RECEIVE);
    x->posted[x->tail++%(SW_POSTED*4)]=s;
  }
}
static inline int sw_idle_reverse(const struct sw_world *w,const struct sw_lane *x,double t){ return x->reverse==UINT32_MAX || w->lane[x->reverse].free_at<=t; }
/* node a's lanes' outboxes posted while the lane is free and its far end holds a receive */
static inline void sw_post_lanes(struct sw_world *w,uint32_t a,double t){
  struct sw_node *n=w->node+a;
  for(uint32_t l=0;l<MESH_SWITCH_LANES;l++){
    if(n->lane_at[l]==UINT32_MAX)continue;
    struct sw_lane *x=w->lane+n->lane_at[l];
    while(x->free_at<=t+1e-9){
      const uint32_t s=mesh_switch_peek(n->sw,l);
      if(s==UINT32_MAX)break;
      sw_repost(w,x);
      if(x->head==x->tail)break;   /* no credit: the SEND waits for a receive */
      mesh_switch_pop(n->sw,l);
      const uint32_t into=x->posted[x->head++%(SW_POSTED*4)];
      const double start=t>x->free_at?t:x->free_at;
      const double rate=w->link[x->link].rate/w->link[x->link].cables*(sw_idle_reverse(w,x,start)?1:w->duplex);
      const double end=start+(double)n->sw->slot_bytes/rate;
      x->free_at=end;w->count[3]++;
      sw_push(w,(struct sw_event){end,SW_EV_DONE,n->lane_at[l],s,into,n->sw->slot[s].posted?(double)n->sw->slot[s].posted:start,start});
    }
  }
}
static inline void sw_bridge_pass(struct sw_world *w,uint32_t a,double t){
  struct sw_node *v=w->node+a;
  const double page=(double)SW_PAGE;
  for(int again=1;again;){
    again=0;
    sw_client_pass(w,a,t);
    uint32_t keep=0;
    for(uint32_t i=0;i<v->nheld;i++){
      const uint32_t s=v->held[i];
      const double before=v->cpu;
      sw_copy_cost(w,v,t,page);
      if(mesh_switch_forward(v->m,v->sw,s,(uint64_t)v->cpu,MESH_SWITCH_LOGS-1)<=0){v->held[keep++]=s;v->cpu=before;}
      else again=1;
    }
    v->nheld=keep;
    for(uint32_t n=0;n<MESH_NODES;n++){
      struct mesh_node *x=mesh_node_at(v->m,n);
      if(!atomic_load(&x->node))continue;
      for(uint32_t p=0;p<MESH_PROGRAMS;p++){
        struct mesh_declared *d=x->program+p;
        if(!atomic_load(&d->owner))continue;
        uint32_t s;
        if(mesh_switch_credit(v->m,v->sw,x,d,sw_clock(v,t),&s)){
          if(mesh_switch_forward(v->m,v->sw,s,sw_clock(v,t),MESH_SWITCH_LOGS-1)<=0)sw_hold(v,s);
          again=1;
        }
        while(mesh_switch_stage(v->m,v->sw,x,d,sw_clock(v,t),&s)==1){
          sw_copy_cost(w,v,t,page);
          if(mesh_switch_forward(v->m,v->sw,s,(uint64_t)v->cpu,MESH_SWITCH_LOGS-1)<=0)sw_hold(v,s);
          again=1;
        }
      }
    }
  }
  sw_drain(w,a);
  sw_post_lanes(w,a,v->cpu>t?v->cpu:t);
}
static inline void sw_check_ops(struct sw_world *w,double t){
  for(uint32_t i=0;i<SW_OPS;i++){
    struct sw_op *o=w->op+i;
    if(!o->live || !sw_op_whole(w,o))continue;
    double end=t;
    for(uint32_t ci=0;ci<w->ncompute;ci++)if(w->node[w->compute[ci]].cpu>end)end=w->node[w->compute[ci]].cpu;
    o->end=end;
    if(w->on_op)w->on_op(w->op_ctx,i,end);
    if(!w->payload)sw_op_end(w,i);
  }
}
static inline void sw_step(struct sw_world *w,struct sw_event ev){
  w->count[0]++;
  if(ev.kind==SW_EV_REFRESH){
    for(uint32_t i=0;i<w->nodes;i++)mesh_switch_refresh(w->node[i].sw,NULL,w->draws,0.5f,mesh_switch_mix(w->seed^(uint64_t)ev.t)^i,(double)w->node[i].sw->slot_bytes,0);
    for(uint32_t i=0;i<w->nodes;i++)sw_bridge_pass(w,i,ev.t);   /* what waited for the view to change */
    int owed=w->events>0;for(uint32_t i=0;i<SW_OPS && !owed;i++)owed=w->op[i].live && w->op[i].end<0;
    if(owed)sw_push(w,(struct sw_event){ev.t+w->refresh_ns,SW_EV_REFRESH,0,0,0,0,0});
    return;
  }
  struct sw_lane *x=w->lane+ev.lane;
  if(ev.kind==SW_EV_DONE){
    struct sw_node *src=w->node+x->node,*dst=w->node+x->peer;
    memcpy(mesh_switch_slot_at(dst->m,dst->sw,ev.to_slot),mesh_switch_slot_at(src->m,src->sw,ev.slot),src->sw->slot_bytes);
    mesh_switch_sent(src->sw,sw_switch_lane(w,ev.lane),ev.slot,src->sw->slot_bytes,(uint64_t)ev.posted,(uint64_t)ev.start,(uint64_t)ev.t);
    sw_push(w,(struct sw_event){ev.t+w->link[x->link].latency,SW_EV_LAND,ev.lane,ev.slot,ev.to_slot,0,0});
    sw_bridge_pass(w,x->node,ev.t);
  }else{
    struct sw_node *dst=w->node+x->peer;
    const uint32_t back=x->reverse!=UINT32_MAX?sw_switch_lane(w,x->reverse):0;
    sw_copy_cost(w,dst,ev.t,0);
    if(mesh_switch_landed(dst->m,dst->sw,back,ev.to_slot,sw_clock(dst,ev.t),MESH_SWITCH_LOGS-1)<=0)sw_hold(dst,ev.to_slot);
    sw_repost(w,x);
    sw_bridge_pass(w,x->peer,ev.t);
    sw_post_lanes(w,x->node,ev.t);
    sw_check_ops(w,ev.t);
  }
}
/* every event before `until` (INFINITY: until nothing is left), in time order; the world's clock then `until` */
static inline void sw_run_until(struct sw_world *w,double until){
  while(w->events && w->heap[0].t<until){
    const struct sw_event e=sw_pop(w);
    w->now=e.t;
    sw_step(w,e);
  }
  if(isfinite(until) && until>w->now)w->now=until;
}
/* the bridges' passes at the world's clock (after a collective arrives, or a table is published) */
static inline void sw_pass_all(struct sw_world *w){
  for(uint32_t i=0;i<w->nodes;i++)sw_bridge_pass(w,i,w->now);
  if(!w->fixed && w->nlive){
    int refresh=0;for(size_t e=0;e<w->events;e++)refresh|=w->heap[e].kind==SW_EV_REFRESH;
    if(!refresh)sw_push(w,(struct sw_event){w->now+w->refresh_ns,SW_EV_REFRESH,0,0,0,0,0});
  }
}
/* every slot of every pool back (free, or posted as a receive) */
static inline int sw_world_slots_back(const struct sw_world *w){
  for(uint32_t i=0;i<w->nodes;i++){
    uint64_t posted=0;
    for(uint32_t l=0;l<w->nlanes;l++)if(w->lane[l].peer==i)posted+=w->lane[l].tail-w->lane[l].head;
    if(atomic_load(&w->node[i].sw->free_count)+posted!=w->node[i].sw->slots){
      fprintf(stderr,"switched-world: node %u: %llu slots free + %llu posted of %u\n",i,(unsigned long long)atomic_load(&w->node[i].sw->free_count),
              (unsigned long long)posted,w->node[i].sw->slots);
      return 0;
    }
  }
  return 1;
}
/* payload 1: op o's every received byte equal to what its program says (the sums exact in float on integers) */
static inline int sw_world_check(struct sw_world *w,uint32_t i,uint64_t payload_seed){
  struct sw_op *o=w->op+i;
  for(uint32_t ci=0;ci<w->ncompute;ci++){
    const uint32_t a=w->compute[ci];
    for(uint32_t cj=0;cj<w->ncompute;cj++){
      const uint32_t b=w->compute[cj];if(a==b)continue;
      if(o->kind==SW_ALLGATHER){
        const float *got=sw_at(w->node+a,o->scratch[a][b]);
        for(uint64_t k=0;k<o->S/4;k++)if(got[k]!=(float)((k*7+b*13+payload_seed*3)%1000)){fprintf(stderr,"all-gather %u<-%u [%llu]: %g\n",a,b,(unsigned long long)k,got[k]);return 0;}
      }
    }
    if(o->kind==SW_ALLREDUCE){
      const float *out=sw_at(w->node+a,o->output[a]);
      for(uint64_t k=0;k<o->S/4;k++){
        float want=0;for(uint32_t cj=0;cj<w->ncompute;cj++)want+=(float)((k*7+w->compute[cj]*13+payload_seed*3)%1000);
        if(out[k]!=want){fprintf(stderr,"all-reduce at %u [%llu]: %g != %g\n",a,(unsigned long long)k,out[k],want);return 0;}
      }
    }
  }
  return 1;
}
#endif
