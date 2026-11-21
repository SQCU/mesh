/* The switched bridges of N nodes over a timed wire (design/switched-multipath-dma.md "Tests"): the bridge's own
   functions (mesh-switched.h: staging, the draw, relays by name, delivery, the estimators, the refresh) on each node's
   region, in one process, on a fabric file (default tests/fabric-20261119.txt, the five-node cluster as measured).

   The wire is the hardware's: every cable its own lane each way at its share of the pair's measured rate and the pair's
   latency, a SEND waiting for a receive posted at the far end (the transport's credit), a cable sending while its
   reverse is busy at the file's duplex share, every SEND one whole slot.  Each node's bridge stages what its clients'
   programs release, forwards what lands, posts its lanes' outboxes and reposts its receives; copies cost the node's
   CPU time (COPY_GBPS).  The clients are the collectives as libnccl-mesh declares them, a program toward every other
   compute node, both ends' tables equal with directions swapped:
   - all-gather: each compute rank's S bytes, as pages, to every other rank;
   - all-reduce: page p's partial from every rank to its owner (compute rank p mod n), which sums it once its partials
     have landed and opens its reduced page's gate, which then goes to every other rank.
   The relay (the Mac mini) declares nothing: it forwards by name.

   Two routings on the same programs and bridges: `fixed`, each destination's one least-delay neighbour on the
   unloaded fabric, set once (the compiled routes' analog), and `switched`, the weights refreshed every REFRESH_US from
   the estimators (Thompson sampling, eta 0.5), read stale while refreshing.  Each row is the mean and spread of an
   execution's completion over MEASURE executions after LEARN.  Checked every execution: every byte of every receive
   equal to what its program says (the sums exact in float on integers), every unit placed exactly once (each
   transfer's bitmap), every slot back in the pool at the end.  The draws are logged in rdma/tests/route's record
   format (split.h struct route_draw) and checked: a token among its table's splits, a log-probability at most 0; with
   LOG=path they are written there (struct mesh_switch_record, raw).

   usage: sim-switched [fabric [sizes...]] */
#include "../mesh-switched.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CABLES 8
#define LEARN 6
#define MEASURE 8
#define POSTED 32            /* receives a lane keeps posted */
static double COPY_GBPS=24;  /* a node's memcpy, bytes a ns */
static double REFRESH_US=20;

/* ---- the fabric ---- */
static char names[MESH_SWITCH_N][16];
static uint32_t nodes,compute[MESH_SWITCH_N],ncompute,nlinks;
struct fl { uint32_t from,to,cables; double rate,latency; };
static struct fl fab[64];
static double duplex=1;
static uint32_t node_of(const char *s){ for(uint32_t i=0;i<nodes;i++)if(!strcmp(names[i],s))return i; fprintf(stderr,"no node %s\n",s);exit(2); }
static void load(const char *path){
  FILE *f=fopen(path,"r");if(!f){perror(path);exit(2);}
  char line[256];
  while(fgets(line,sizeof line,f)){
    char a[16],b[16],kind[16];double rate,us;unsigned n=1;
    if(line[0]=='#')continue;
    if(sscanf(line,"node %15s %15s",a,kind)==2){snprintf(names[nodes],16,"%s",a);if(!strcmp(kind,"compute"))compute[ncompute++]=nodes;nodes++;}
    else if(sscanf(line,"duplex %lf",&rate)==1)duplex=rate;
    else if(sscanf(line,"link %15s %15s %lf %lf %u",a,b,&rate,&us,&n)>=4)fab[nlinks++]=(struct fl){node_of(a),node_of(b),n,rate,us*1e3};
  }
  fclose(f);
}

/* ---- the nodes: a region each ---- */
struct lane { uint32_t node,peer,cable,link,reverse; double free_at; uint32_t posted[POSTED*4]; uint32_t head,tail; uint32_t waiting; };
static struct lane lanes[256];static uint32_t nlanes;
struct node {
  struct hdr *m; struct mesh_switch *sw; uint64_t bump; double cpu;
  uint32_t lane_at[MESH_SWITCH_LANES];   /* switch lane -> sim lane */
  uint32_t *held; uint32_t nheld,room;
};
static struct node node[MESH_SWITCH_N];
static uint64_t alloc(struct node *v,uint64_t bytes){
  const uint64_t at=v->bump;v->bump+=(bytes+16383)/16384*16384;
  assert(v->bump<=v->m->length);
  return at;
}
static void *at(struct node *v,uint64_t off){ return (char *)v->m+off; }

static void make_nodes(uint64_t region_mb){
  for(uint32_t i=0;i<nodes;i++){
    struct hdr h;memset(&h,0,sizeof h);
    const uint64_t length=mesh_layout(&h,16384,1,(uint32_t)(region_mb*64));
    struct node *v=node+i;
    v->m=calloc(1,length);assert(v->m);
    memcpy(v->m,&h,sizeof h);v->m->magic=MESH_MAGIC;v->m->node=i+1;
    v->bump=v->m->data_off;
    v->sw=mesh_switch_of(v->m);
    const uint32_t slots=2048;
    const uint64_t sb=mesh_switch_slot_bytes(v->m);
    mesh_switch_init(v->sw,slots,sb,alloc(v,(uint64_t)slots*sb));
    atomic_store(&v->sw->logging,1);
    uint32_t k=0;
    for(uint32_t j=0;j<nodes;j++)if(j!=i){struct mesh_node *x=mesh_node_at(v->m,k++);atomic_store(&x->node,j+1);atomic_store(&x->live,1);atomic_store(&x->generation,1);}
    uint32_t nums[MESH_SWITCH_N];struct mesh_switch_fabric_link fl[64];
    for(uint32_t j=0;j<nodes;j++)nums[j]=j+1;
    for(uint32_t l=0;l<nlinks;l++)fl[l]=(struct mesh_switch_fabric_link){fab[l].from+1,fab[l].to+1,fab[l].cables,fab[l].rate,fab[l].latency};
    mesh_switch_view_set(v->sw,i+1,nums,nodes,fl,nlinks);
    for(uint32_t l=0;l<MESH_SWITCH_LANES;l++)v->lane_at[l]=UINT32_MAX;
  }
  /* lanes: a cable each way; the switch's lane p 8 + 0 its port p */
  uint32_t ports[MESH_SWITCH_N]={0};
  for(uint32_t l=0;l<nlinks;l++)for(uint32_t c=0;c<fab[l].cables;c++){
    struct lane *x=lanes+nlanes;
    *x=(struct lane){.node=fab[l].from,.peer=fab[l].to,.cable=c,.link=l,.reverse=UINT32_MAX};
    const uint32_t sl=ports[fab[l].from]++*2*MESH_LINK_STREAMS;
    struct mesh_switch_lane *s=node[fab[l].from].sw->lane+sl;
    atomic_store(&s->peer,fab[l].to+1);atomic_store(&s->live,1);
    node[fab[l].from].lane_at[sl]=nlanes;
    nlanes++;
  }
  for(uint32_t a=0;a<nlanes;a++)for(uint32_t b=0;b<nlanes;b++)
    if(lanes[b].node==lanes[a].peer && lanes[b].peer==lanes[a].node && lanes[b].cable==lanes[a].cable)lanes[a].reverse=b;
}
static uint32_t switch_lane_of(uint32_t simlane){
  struct node *v=node+lanes[simlane].node;
  for(uint32_t l=0;l<MESH_SWITCH_LANES;l++)if(v->lane_at[l]==simlane)return l;
  abort();
}

/* ---- events ---- */
enum { EV_DONE=1, EV_LAND };
struct event { double t; uint32_t kind,lane,slot,to_slot; double posted,start; };
static struct event *heap;static size_t events,room;
static void push(struct event e){
  if(events==room){room=room?2*room:4096;heap=realloc(heap,room*sizeof *heap);}
  size_t i=events++;
  while(i && heap[(i-1)/2].t>e.t){heap[i]=heap[(i-1)/2];i=(i-1)/2;}
  heap[i]=e;
}
static struct event pop(void){
  struct event top=heap[0],last=heap[--events];size_t i=0;
  for(;;){size_t c=2*i+1;if(c>=events)break;if(c+1<events && heap[c+1].t<heap[c].t)c++;if(heap[c].t>=last.t)break;heap[i]=heap[c];i=c;}
  if(events)heap[i]=last;
  return top;
}

/* receives: each lane keeps POSTED slots of its receiver's pool posted */
static void repost(struct lane *x){
  struct node *r=node+x->peer;
  while(x->tail-x->head<POSTED){
    const uint32_t s=mesh_switch_take(r->sw);
    if(s==UINT32_MAX)return;
    atomic_store(&r->sw->slot[s].state,SLOT_RECEIVE);
    x->posted[x->tail++%(POSTED*4)]=s;
  }
}
static int idle_reverse(const struct lane *x,double t){ return x->reverse==UINT32_MAX || lanes[x->reverse].free_at<=t; }
/* node v's lanes' outboxes posted while the lane is free and its far end holds a receive */
static void post_lanes(uint32_t v,double t){
  struct node *n=node+v;
  for(uint32_t l=0;l<MESH_SWITCH_LANES;l++){
    if(n->lane_at[l]==UINT32_MAX)continue;
    struct lane *x=lanes+n->lane_at[l];
    while(x->free_at<=t+1e-9){
      const uint32_t s=mesh_switch_peek(n->sw,l);
      if(s==UINT32_MAX)break;
      repost(x);
      if(x->head==x->tail)break;   /* no credit: the SEND waits for a receive */
      mesh_switch_pop(n->sw,l);
      const uint32_t into=x->posted[x->head++%(POSTED*4)];
      const double start=t>x->free_at?t:x->free_at;
      const double rate=fab[x->link].rate/fab[x->link].cables*(idle_reverse(x,start)?1:duplex);
      const double end=start+(double)n->sw->slot_bytes/rate;
      x->free_at=end;
      push((struct event){end,EV_DONE,n->lane_at[l],s,into,n->sw->slot[s].posted?(double)n->sw->slot[s].posted:start,start});
    }
  }
}

/* ---- the programs ---- */
enum { ALLGATHER, ALLREDUCE };
static int kind;static uint64_t S,P,page=16384;
struct prog { uint64_t table,spans,done; struct mesh_declared *d; };
/* per node: buffers */
static uint64_t input[MESH_SWITCH_N],output[MESH_SWITCH_N],scratch[MESH_SWITCH_N][MESH_SWITCH_N];
static uint64_t landed_word[MESH_SWITCH_N][MESH_SWITCH_N],sent_word[MESH_SWITCH_N][MESH_SWITCH_N],gate_word[MESH_SWITCH_N];
static uint64_t partial_word[MESH_SWITCH_N][MESH_SWITCH_N];   /* all-reduce: words base, a page a word */
static uint64_t reduced_word[MESH_SWITCH_N][MESH_SWITCH_N];
static uint32_t rank_of[MESH_SWITCH_N];
static uint64_t owned_count(uint32_t r){ uint64_t c=0;for(uint64_t p=0;p<P;p++)if(p%ncompute==r)c++;return c; }
static uint64_t page_bytes(uint64_t p){ return p+1<P?page:S-page*(P-1); }
static void span_set(struct node *v,uint64_t spans,uint64_t k,uint64_t base,uint64_t p){
  ((struct mesh_dma_span *)at(v,spans))[k]=(struct mesh_dma_span){base+p*page,page_bytes(p)};
}
static void declare(uint32_t a,uint32_t b,uint32_t slot){
  struct node *v=node+a;
  const uint32_t ra=rank_of[a],rb=rank_of[b];
  struct mesh_crossing *t[2];uint64_t n[2]={0,0},pieces[2]={0,0};
  const uint64_t room=2*P+4;
  const uint64_t tables=alloc(v,2*room*sizeof(struct mesh_crossing)),spans=alloc(v,4*room*sizeof(struct mesh_dma_span));
  t[0]=at(v,tables);t[1]=t[0]+room;
  uint64_t k=0;
  if(kind==ALLGATHER){
    t[0][0]=(struct mesh_crossing){.spans=spans+k*16,.units=P,.first=0,.done=sent_word[a][b],.done_value=1,.done_step=1};
    for(uint64_t p=0;p<P;p++)span_set(v,spans,k++,input[a],p);
    t[1][0]=(struct mesh_crossing){.spans=spans+k*16,.units=P,.first=0,.done=landed_word[a][b],.done_value=1,.done_step=1};
    for(uint64_t p=0;p<P;p++)span_set(v,spans,k++,scratch[a][b],p);
    n[0]=n[1]=1;pieces[0]=pieces[1]=P;
  }else{
    /* send: the partials of b's pages, then a's reduced pages (each its gate); receive: the partials of a's pages,
       then b's reduced pages; a transfer a page */
    const uint64_t na=owned_count(ra),nb=owned_count(rb);
    uint64_t q=0;
    for(uint64_t p=0;p<P;p++)if(p%ncompute==rb){t[0][n[0]]=(struct mesh_crossing){.spans=spans+k*16,.units=1,.first=pieces[0],.done=sent_word[a][b],.done_value=++q,.done_step=nb+na};span_set(v,spans,k++,input[a],p);n[0]++;pieces[0]++;}
    uint64_t i=0;
    for(uint64_t p=0;p<P;p++)if(p%ncompute==ra){t[0][n[0]]=(struct mesh_crossing){.spans=spans+k*16,.units=1,.first=pieces[0],.gate=gate_word[a],.gate_base=i,.gate_step=na,
      .done=sent_word[a][b],.done_value=++q,.done_step=nb+na};span_set(v,spans,k++,output[a],p);n[0]++;pieces[0]++;i++;}
    for(uint64_t p=0;p<P;p++)if(p%ncompute==ra){t[1][n[1]]=(struct mesh_crossing){.spans=spans+k*16,.units=1,.first=pieces[1],.done=partial_word[a][b]+(p/ncompute)*8,.done_value=1,.done_step=1};
      span_set(v,spans,k++,scratch[a][b],p);n[1]++;pieces[1]++;}
    uint64_t j=0;
    for(uint64_t p=0;p<P;p++)if(p%ncompute==rb){t[1][n[1]]=(struct mesh_crossing){.spans=spans+k*16,.units=1,.first=pieces[1],.done=reduced_word[a][b],.done_value=++j,.done_step=nb};
      span_set(v,spans,k++,output[a],p);n[1]++;pieces[1]++;}
  }
  struct mesh_node *x=mesh_switch_order(v->m,b+1);
  struct mesh_declared *d=x->program+slot;
  memset(d,0,sizeof *d);
  d->program=(struct mesh_declaration){.key=0x1234+slot,.period=1,.count=1,.table={tables,tables+room*sizeof(struct mesh_crossing)},
    .transfers={n[0],n[1]},.pieces={pieces[0],pieces[1]}};
  for(int dd=0;dd<2;dd++){d->progress[dd]=alloc(v,mesh_switch_progress_bytes(t[dd],n[dd]));mesh_switch_progress_init(v->m,d,dd,t[dd],n[dd]);}
  atomic_store(&d->closed,UINT64_MAX);atomic_store(&d->claimed,0);atomic_store(&d->issued,0);
  atomic_store(&d->owner,1);
}
static uint64_t word(uint32_t v,uint64_t off){ return atomic_load((_Atomic uint64_t *)at(node+v,off)); }
static void setup(int k,uint64_t s){
  kind=k;S=s;P=(S+page-1)/page;
  for(uint32_t i=0;i<nodes;i++){
    struct node *v=node+i;
    v->bump=v->sw->slot_base+(uint64_t)v->sw->slots*v->sw->slot_bytes;v->bump=(v->bump+16383)/16384*16384;
    v->cpu=0;
    rank_of[i]=UINT32_MAX;
  }
  for(uint32_t c=0;c<ncompute;c++)rank_of[compute[c]]=c;
  for(uint32_t ci=0;ci<ncompute;ci++){
    const uint32_t a=compute[ci];struct node *v=node+a;
    input[a]=alloc(v,P*page);output[a]=alloc(v,P*page);
    const uint64_t words=alloc(v,8*(4*MESH_SWITCH_N+MESH_SWITCH_N*(P+1)+8));
    memset(at(v,words),0,8*(4*MESH_SWITCH_N+MESH_SWITCH_N*(P+1)+8));
    gate_word[a]=words;
    for(uint32_t b=0;b<nodes;b++){landed_word[a][b]=words+8*(1+b);sent_word[a][b]=words+8*(1+MESH_SWITCH_N+b);reduced_word[a][b]=words+8*(1+2*MESH_SWITCH_N+b);
      partial_word[a][b]=words+8*(4*MESH_SWITCH_N+b*(P+1));scratch[a][b]=alloc(v,P*page);}
  }
  for(uint32_t ci=0;ci<ncompute;ci++)for(uint32_t cj=0;cj<ncompute;cj++)if(ci!=cj)declare(compute[ci],compute[cj],0);
}

/* ---- the bridges' passes ---- */
static void hold(struct node *v,uint32_t s){
  if(v->nheld==v->room){v->room=v->room?2*v->room:256;v->held=realloc(v->held,v->room*sizeof *v->held);}
  v->held[v->nheld++]=s;
}
static void copy_cost(struct node *v,double t,double bytes){ if(v->cpu<t)v->cpu=t;v->cpu+=bytes/COPY_GBPS; }
static uint64_t ex;   /* the execution running */
static double reduce_done[MESH_SWITCH_N];static uint64_t next_reduce[MESH_SWITCH_N];
static void client_pass(uint32_t a,double t){
  if(kind!=ALLREDUCE || rank_of[a]==UINT32_MAX)return;
  struct node *v=node+a;const uint32_t r=rank_of[a];
  /* the owner's reduction, page by page in order, once each partial of the page has landed */
  for(;;){
    const uint64_t i=next_reduce[a];
    if(i>=owned_count(r))return;
    const uint64_t p=r+i*ncompute;
    for(uint32_t cj=0;cj<ncompute;cj++){const uint32_t b=compute[cj];if(b!=a && word(a,partial_word[a][b]+i*8)<ex+1)return;}
    float *o=at(v,output[a]+p*page);const float *in=at(v,input[a]+p*page);
    const uint64_t count=page_bytes(p)/4;
    for(uint64_t e=0;e<count;e++){float sum=in[e];for(uint32_t cj=0;cj<ncompute;cj++){const uint32_t b=compute[cj];if(b!=a)sum+=((float *)at(v,scratch[a][b]+p*page))[e];}o[e]=sum;}
    copy_cost(v,t,(double)page_bytes(p)*ncompute);
    next_reduce[a]++;
    mesh_switch_raise(v->m,gate_word[a],ex*owned_count(r)+next_reduce[a]);
    reduce_done[a]=v->cpu;
  }
}

static void bridge_pass(uint32_t a,double t){
  struct node *v=node+a;
  for(int again=1;again;){
    again=0;
    client_pass(a,t);
    /* the held slots retried */
    uint32_t keep=0;
    for(uint32_t i=0;i<v->nheld;i++){
      const uint32_t s=v->held[i];
      const double before=v->cpu;
      copy_cost(v,t,(double)page);
      if(mesh_switch_forward(v->m,v->sw,v->held[i],(uint64_t)v->cpu,MESH_SWITCH_LOGS-1)<=0){v->held[keep++]=s;v->cpu=before;}
      else again=1;
    }
    v->nheld=keep;
    /* staging */
    for(uint32_t n=0;n<MESH_NODES;n++){
      struct mesh_node *x=mesh_node_at(v->m,n);
      if(!atomic_load(&x->node))continue;
      for(uint32_t p=0;p<MESH_PROGRAMS;p++){
        struct mesh_declared *d=x->program+p;
        if(!atomic_load(&d->owner))continue;
        uint32_t s;
        if(mesh_switch_credit(v->m,v->sw,x,d,(uint64_t)(v->cpu>t?v->cpu:t),&s)){
          if(mesh_switch_forward(v->m,v->sw,s,(uint64_t)(v->cpu>t?v->cpu:t),MESH_SWITCH_LOGS-1)<=0)hold(v,s);
          again=1;
        }
        while(mesh_switch_stage(v->m,v->sw,x,d,(uint64_t)(v->cpu>t?v->cpu:t),&s)==1){
          copy_cost(v,t,(double)page);
          if(mesh_switch_forward(v->m,v->sw,s,(uint64_t)v->cpu,MESH_SWITCH_LOGS-1)<=0)hold(v,s);
          again=1;
        }
      }
    }
  }
  post_lanes(a,v->cpu>t?v->cpu:t);
}

/* ---- one execution ---- */
static void fill(uint64_t e){
  for(uint32_t ci=0;ci<ncompute;ci++){
    const uint32_t a=compute[ci];
    float *in=at(node+a,input[a]);
    for(uint64_t i=0;i<S/4;i++)in[i]=(float)((i*7+a*13+e*3)%1000);
  }
}
static void check(uint64_t e){
  for(uint32_t ci=0;ci<ncompute;ci++){
    const uint32_t a=compute[ci];
    for(uint32_t cj=0;cj<ncompute;cj++){
      const uint32_t b=compute[cj];if(a==b)continue;
      if(kind==ALLGATHER){
        const float *got=at(node+a,scratch[a][b]);
        for(uint64_t i=0;i<S/4;i++)if(got[i]!=(float)((i*7+b*13+e*3)%1000)){fprintf(stderr,"all-gather %u<-%u e %llu [%llu]: %g\n",a,b,(unsigned long long)e,(unsigned long long)i,got[i]);exit(4);}
      }
    }
    if(kind==ALLREDUCE){
      const float *o=at(node+a,output[a]);
      for(uint64_t i=0;i<S/4;i++){
        float want=0;for(uint32_t cj=0;cj<ncompute;cj++)want+=(float)((i*7+compute[cj]*13+e*3)%1000);
        if(o[i]!=want){fprintf(stderr,"all-reduce at %u e %llu [%llu]: %g != %g\n",a,(unsigned long long)e,(unsigned long long)i,o[i],want);exit(4);}
      }
    }
  }
}
static int finished(uint64_t e){
  for(uint32_t ci=0;ci<ncompute;ci++){
    const uint32_t a=compute[ci];
    for(uint32_t cj=0;cj<ncompute;cj++){
      const uint32_t b=compute[cj];if(a==b)continue;
      if(kind==ALLGATHER && word(a,landed_word[a][b])<e+1)return 0;
      if(kind==ALLREDUCE && word(a,reduced_word[a][b])<(e+1)*owned_count(rank_of[b]))return 0;
    }
    if(kind==ALLREDUCE && next_reduce[a]<owned_count(rank_of[a]))return 0;
  }
  return 1;
}
static FILE *logf_;static uint64_t draws,pieces_logged,dropped;
static void drain_logs(void){
  for(uint32_t i=0;i<nodes;i++){
    struct mesh_switch_record r;
    while(mesh_switch_log_take(node[i].sw,MESH_SWITCH_LOGS-1,&r)){
      if(r.kind==MESH_LOG_DRAW){
        draws++;
        assert(r.logp<=1e-6f);
        /* the token's every destination's lane valid and allowed (split.h split_of) */
        struct split sp=split_of(&r.draw,r.draw.token);
        assert(sp.ok);
      }else if(r.kind==MESH_LOG_PIECE)pieces_logged++;
      if(logf_)fwrite(&r,sizeof r,1,logf_);
    }
    dropped+=atomic_load(&node[i].sw->log[MESH_SWITCH_LOGS-1].dropped);
  }
}
static int fixed;
static void refresh_all(uint64_t e,double at){
  for(uint32_t i=0;i<nodes;i++)mesh_switch_refresh(node[i].sw,NULL,32,0.5f,(e<<20)^(uint64_t)(at*7)^i,(double)node[i].sw->slot_bytes,0);
}
static void step(struct event ev){
  struct lane *x=lanes+ev.lane;
  if(ev.kind==EV_DONE){
    struct node *src=node+x->node,*dst=node+x->peer;
    memcpy(mesh_switch_slot_at(dst->m,dst->sw,ev.to_slot),mesh_switch_slot_at(src->m,src->sw,ev.slot),src->sw->slot_bytes);
    mesh_switch_sent(src->sw,switch_lane_of(ev.lane),ev.slot,src->sw->slot_bytes,(uint64_t)ev.posted,(uint64_t)ev.start,(uint64_t)ev.t);
    push((struct event){ev.t+fab[x->link].latency,EV_LAND,ev.lane,ev.slot,ev.to_slot,0,0});
    bridge_pass(x->node,ev.t);
  }else{
    struct node *dst=node+x->peer;
    /* the receiver's lane over the same cable (its landings counted there) */
    const uint32_t back=x->reverse!=UINT32_MAX?switch_lane_of(x->reverse):0;
    copy_cost(dst,ev.t,0);
    if(mesh_switch_landed(dst->m,dst->sw,back,ev.to_slot,(uint64_t)(dst->cpu>ev.t?dst->cpu:ev.t),MESH_SWITCH_LOGS-1)<=0)hold(dst,ev.to_slot);
    repost(x);
    bridge_pass(x->peer,ev.t);
    post_lanes(x->node,ev.t);
  }
}
static double execute(uint64_t e){
  ex=e;fill(e);
  for(uint32_t i=0;i<nodes;i++){node[i].cpu=0;next_reduce[i]=0;}
  for(uint32_t l=0;l<nlanes;l++)lanes[l].free_at=0;
  /* each client issues its step */
  for(uint32_t ci=0;ci<ncompute;ci++){
    struct node *v=node+compute[ci];
    for(uint32_t n=0;n<MESH_NODES;n++){struct mesh_node *x=mesh_node_at(v->m,n);for(uint32_t p=0;p<MESH_PROGRAMS;p++)if(atomic_load(&x->program[p].owner))atomic_store(&x->program[p].issued,e+1);}
  }
  double next_refresh=0;
  if(!fixed){refresh_all(e,0);next_refresh=REFRESH_US*1e3;}
  for(uint32_t i=0;i<nodes;i++)bridge_pass(i,0);
  double t=0;uint64_t steps=0;
  while(!finished(e)){
    t=events?heap[0].t:next_refresh;
    /* the refreshes due: each node's weights recomputed from its estimators while pieces keep drawing from the last */
    while(!fixed && next_refresh<=t){refresh_all(e,next_refresh);next_refresh+=REFRESH_US*1e3;}
    if(!events){
      assert(!fixed);   /* fixed weights never move: nothing in flight with work owed is a routing that cannot finish */
      for(uint32_t i=0;i<nodes;i++)bridge_pass(i,t);
      assert(++steps<1000000);continue;
    }
    step(pop());
    assert(++steps<100000000);
  }
  double end=t;
  for(uint32_t ci=0;ci<ncompute;ci++)if(node[compute[ci]].cpu>end)end=node[compute[ci]].cpu;
  /* the wire drained (what is still crossing is a SEND whose every destination already holds it: none) */
  while(events)step(pop());
  check(e);
  drain_logs();
  for(uint32_t i=0;i<nodes;i++)assert(!node[i].nheld);
  return end;
}
static void reset_nodes(void){
  for(uint32_t i=0;i<nodes;i++){
    struct node *v=node+i;
    const uint64_t base=v->sw->slot_base,sb=v->sw->slot_bytes;const uint32_t slots=v->sw->slots;
    struct mesh_switch_view view=v->sw->view;struct mesh_switch_lane lane[MESH_SWITCH_LANES];memcpy(lane,v->sw->lane,sizeof lane);
    mesh_switch_init(v->sw,slots,sb,base);
    memcpy((char *)&v->sw->view+8,(char *)&view+8,sizeof view-8);
    for(uint32_t l=0;l<v->sw->view.links;l++){atomic_store(&v->sw->view.link[l].line,0);atomic_store(&v->sw->view.link[l].stamp,0);}
    for(uint32_t l=0;l<MESH_SWITCH_LANES;l++){atomic_store(&v->sw->lane[l].live,atomic_load(&lane[l].live));atomic_store(&v->sw->lane[l].peer,atomic_load(&lane[l].peer));}
    atomic_store(&v->sw->logging,1);
    for(uint32_t n=0;n<MESH_NODES;n++){struct mesh_node *x=mesh_node_at(v->m,n);for(uint32_t p=0;p<MESH_PROGRAMS;p++)atomic_store(&x->program[p].owner,0);}
    v->nheld=0;
  }
  for(uint32_t l=0;l<nlanes;l++){lanes[l].head=lanes[l].tail=0;lanes[l].free_at=0;}
}
static void leaked(void){
  for(uint32_t i=0;i<nodes;i++){
    uint64_t posted=0;
    for(uint32_t l=0;l<nlanes;l++)if(lanes[l].peer==i)posted+=lanes[l].tail-lanes[l].head;
    if(atomic_load(&node[i].sw->free_count)+posted!=node[i].sw->slots){
      fprintf(stderr,"node %s: %llu slots free + %llu posted of %u\n",names[i],(unsigned long long)atomic_load(&node[i].sw->free_count),(unsigned long long)posted,node[i].sw->slots);exit(5);
    }
  }
}
struct stats { double mean,sd; };
static struct stats run(int k,uint64_t s,int fixd){
  reset_nodes();fixed=fixd;setup(k,s);
  if(fixed)for(uint32_t i=0;i<nodes;i++)mesh_switch_refresh(node[i].sw,NULL,1,1,0,(double)node[i].sw->slot_bytes,1);
  double sum=0,sq=0;
  for(uint64_t e=0;e<LEARN+MEASURE;e++){
    const double T=execute(e);
    if(e>=LEARN){sum+=T;sq+=T*T;}
  }
  leaked();
  return (struct stats){sum/MEASURE,sqrt(fmax(0,sq/MEASURE-(sum/MEASURE)*(sum/MEASURE)))};
}
int main(int argc,char **argv){
  load(argc>1?argv[1]:"tests/fabric-20261119.txt");
  if(getenv("COPY_GBPS"))COPY_GBPS=atof(getenv("COPY_GBPS"));
  if(getenv("REFRESH_US"))REFRESH_US=atof(getenv("REFRESH_US"));
  if(getenv("LOG"))logf_=fopen(getenv("LOG"),"wb");
  make_nodes(160);
  static const double rows[]={2,8,32,128,1024};
  uint64_t sizes[16];int n=0;
  for(int i=2;i<argc && n<16;i++)sizes[n++]=strtoull(argv[i],NULL,10);
  if(!n)for(size_t i=0;i<sizeof rows/sizeof *rows;i++)sizes[n++]=(uint64_t)(rows[i]*1536*2);
  printf("%-10s %9s %20s %20s\n","","S bytes","fixed","switched");
  for(int k=0;k<2;k++)for(int i=0;i<n;i++){
    const struct stats f=run(k,sizes[i],1),w=run(k,sizes[i],0);
    printf("%-10s %9llu %10.2f+-%6.2fus %10.2f+-%6.2fus\n",k?"all-reduce":"all-gather",(unsigned long long)sizes[i],f.mean/1e3,f.sd/1e3,w.mean/1e3,w.sd/1e3);
    fflush(stdout);
  }
  printf("sim-switched: %llu draws logged (%llu dropped), %llu pieces logged; every byte placed exactly once, every slot back: passed\n",
         (unsigned long long)draws,(unsigned long long)dropped,(unsigned long long)pieces_logged);
  if(logf_)fclose(logf_);
  return 0;
}
