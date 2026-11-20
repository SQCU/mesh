/* A discrete-event run of mesh-switch.h's forwarding (design/switched-multipath-dma.md) on a fabric file (default
   tests/fabric-20261119.txt, the five-node cluster as measured).

   The fabric as the hardware is: every cable its own lane each way, a pair's measured rate shared equally by its
   cables, the pair's measured latency on each. A piece crossing a pair takes whichever of its cables frees first. A
   cable sending while its reverse direction is busy runs at the file's duplex share of its rate. A node's bridge
   forwards a piece the moment it lands.

   Two transactions, at each rank-partial size S:
   - all-gather: each compute rank's S bytes, as pages, to every other compute rank;
   - all-reduce: each page's partial from every rank to its owner (page p's owner is compute rank p mod n), and the
     reduced page from the owner to every other rank once its last partial has landed. No barrier between the two.

   Routers, on the same inputs:
   - fixed: every path at its unloaded delay (fixed shortest-delay routes);
   - sampled: each node's own lanes at their posted backlog, later hops at waits drawn from their lines (Thompson
     sampling, the lines from what each lane's sender's own sends met, carried in names);
   - trees: each piece's tree sampled from the multicast fluid optimum (mesh_switch_trees) with its swizzled quantile;

   The sampled and trees rows are the mean and spread of the completion over 20 executions after any
   learning. The floor is the larger of two bounds: the all-gather's cut bound (the most bytes a compute node needs
   over its inbound rate, or must emit over its outbound rate), and the slowest source-destination pair's shortest
   delay for one piece (latency and the piece's time on one cable, each hop; a piece is at most a page).

   Checked every run: every page reaches every destination that needs it exactly once, the run ends, and every path
   is simple.

   usage: switch [fabric [page]] */
#include "mesh-switch.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ENTRIES 4   /* view lines a name carries, 64 bytes each */
#define LEARN 40    /* executions the sampled router learns its lines over */
#define MEASURE 20  /* executions each reported row is the mean and spread of */
#define CABLES 8

static char names[MESH_SWITCH_NODES][16];
static uint32_t nodes,links,compute[MESH_SWITCH_NODES],ncompute,cables[256],reverse[256];
static struct mesh_switch_link wire[256];
static double duplex=1;

static uint32_t node_of(const char *s){
  for(uint32_t i=0;i<nodes;i++)if(!strcmp(names[i],s))return i;
  fprintf(stderr,"switch: no node %s\n",s);exit(2);
}
static void load(const char *path){
  FILE *f=fopen(path,"r");
  if(!f){perror(path);exit(2);}
  char line[256];
  while(fgets(line,sizeof line,f)){
    char a[16],b[16],kind[16];double rate,us;unsigned n=1;
    if(line[0]=='#')continue;
    if(sscanf(line,"node %15s %15s",a,kind)==2){
      assert(nodes<MESH_SWITCH_NODES);
      snprintf(names[nodes],sizeof names[nodes],"%s",a);
      if(!strcmp(kind,"compute"))compute[ncompute++]=nodes;
      nodes++;
    }else if(sscanf(line,"duplex %lf",&rate)==1)duplex=rate;
    else if(sscanf(line,"link %15s %15s %lf %lf %u",a,b,&rate,&us,&n)>=4){
      assert(links<sizeof wire/sizeof *wire && n>=1 && n<=CABLES);
      cables[links]=n;wire[links++]=(struct mesh_switch_link){node_of(a),node_of(b),rate,us*1e3};
    }
  }
  fclose(f);
  for(uint32_t l=0;l<links;l++){
    reverse[l]=UINT32_MAX;
    for(uint32_t r=0;r<links;r++)if(wire[r].from==wire[l].to && wire[r].to==wire[l].from)reverse[l]=r;
  }
}
static uint64_t all_compute(void){uint64_t a=0;for(uint32_t c=0;c<ncompute;c++)a|=UINT64_C(1)<<compute[c];return a;}

struct piece { uint32_t source,page; double bytes; uint64_t need,got; int phase; struct mesh_switch_tree route; };
struct event { double t; uint32_t node,piece,hops; uint64_t to,visited; uint8_t carried,lane[ENTRIES];uint32_t execution[ENTRIES];
               struct mesh_switch_line line[ENTRIES]; };
static struct event *heap;static size_t events,room;
static void push(struct event e){
  if(events==room){room=room?2*room:1024;heap=realloc(heap,room*sizeof *heap);assert(heap);}
  size_t i=events++;
  while(i && heap[(i-1)/2].t>e.t){heap[i]=heap[(i-1)/2];i=(i-1)/2;}
  heap[i]=e;
}
static struct event pop(void){
  struct event top=heap[0],last=heap[--events];
  size_t i=0;
  for(;;){
    size_t c=2*i+1;
    if(c>=events)break;
    if(c+1<events && heap[c+1].t<heap[c].t)c++;
    if(heap[c].t>=last.t)break;
    heap[i]=heap[c];i=c;
  }
  if(events)heap[i]=last;
  return top;
}

static struct mesh_switch_fabric fabric;
static const double unit=64;   /* ns, the lines' time unit */
enum { FIXED, SAMPLED, TREES, POLICY };
static int mode;
static uint32_t execution;
static double free_at[256][CABLES],busy[256];
static double *posted[256],*waited[256];static uint32_t sends[256],send_room[256];
static struct mesh_switch_line own[256],view[MESH_SWITCH_NODES][256];
static uint32_t own_execution,seen[MESH_SWITCH_NODES][256],cursor[MESH_SWITCH_NODES];
static struct piece *pieces;static uint32_t npieces;
/* the trees a piece is sampled from: commodity j's candidates at tree_first[j] .. tree_first[j + 1] - 1 */
static struct mesh_switch_tree trees[16384];static uint32_t tree_first[65],ncommodities,csrc[64];static uint64_t cto[64];
static double tdemand[64],tree_fluid;

#ifdef ROUTE_POLICY
/* policy: the trained routing policy (route/route.h) as this run's router. Every `policy_dt` ns it reads the fabric
   (each lane's earliest-cable wait, the duplex share, the bytes in flight, the cost-to-go, each pattern's delivered
   fraction) and route/forward.c computes every context's w from its recurrent state, carried from refresh to refresh
   and from execution to execution, as a bridge's would be; the table is active `policy_lag` ns after its refresh (the
   first at once). A piece at node v owing S draws its split from its pattern's context (v, S), masked to the lanes that
   leave its path and reach their share without coming back (route_split). Built with -DROUTE_POLICY; ROUTE_THETA names
   the parameters (minGRU's, or the GRU's with ROUTE_GRU=1). */
#include "route/route.h"
static struct route_stream policy_stream;static struct route_topology policy_topology;static struct route_params policy_params;
static float *policy_theta,*policy_w,*policy_next,*policy_tokens,*policy_scratch,policy_state[2*ROUTE_D],policy_lanes[ROUTE_L][ROUTE_IN];
static int policy_pattern[ROUTE_CONTEXTS],policy_node[ROUTE_CONTEXTS],policy_nc,policy_ctx[ROUTE_J][ROUTE_N][ROUTE_SETS];static uint32_t policy_owed[ROUTE_CONTEXTS];
static struct route_state policy_st,policy_stnext;
static double policy_dt=100e3,policy_lag=100e3,policy_due,policy_at,policy_cdf[3125],policy_us;
static uint64_t policy_refreshes,policy_draws;
static void policy_load(const char *path,int full){
  const int n=route_params_count(full);
  FILE *f=fopen(path,"rb");if(!f){perror(path);exit(2);}
  policy_theta=malloc((size_t)n*sizeof(float));assert(policy_theta && fread(policy_theta,sizeof(float),(size_t)n,f)==(size_t)n);fclose(f);
  route_params_bind(&policy_params,policy_theta,full);
  policy_w=malloc((size_t)ROUTE_CONTEXTS*ROUTE_FS*sizeof(float));policy_next=malloc((size_t)ROUTE_CONTEXTS*ROUTE_FS*sizeof(float));
  policy_tokens=malloc((size_t)ROUTE_CONTEXTS*ROUTE_IN*sizeof(float));policy_scratch=malloc(route_forward_scratch(ROUTE_CONTEXTS)*sizeof(float));
  assert(policy_w && policy_next && policy_tokens && policy_scratch);
}
static void policy_setup(int kind,double S){   /* this run's fabric and collective, as the policy reads them */
  struct route_fabric f;memset(&f,0,sizeof f);
  f.nodes=(int)nodes;f.ncompute=(int)ncompute;f.lanes=(int)links;f.duplex=duplex;
  for(uint32_t c=0;c<ncompute;c++)f.compute[c]=(int)compute[c];
  for(uint32_t l=0;l<links;l++){f.from[l]=(int)wire[l].from;f.to[l]=(int)wire[l].to;f.rate[l]=wire[l].rate;f.latency[l]=wire[l].latency;f.cables[l]=(int)cables[l];}
  route_stream_one(&f,kind?ROUTE_ALLREDUCE:ROUTE_ALLGATHER,S,(int)compute[0],&policy_stream);
  route_topology_make(&policy_stream.f,&policy_stream,&policy_topology);
  memset(policy_ctx,0xff,sizeof policy_ctx);
  policy_nc=route_contexts(&policy_stream,&policy_topology,policy_pattern,policy_node,policy_owed);assert(policy_nc>0);
  for(int c=0;c<policy_nc;c++)policy_ctx[policy_pattern[c]][policy_node[c]][policy_owed[c]]=c;
  memset(policy_state,0,sizeof policy_state);policy_refreshes=policy_draws=0;policy_us=0;
}
static int pattern_of_piece(uint32_t k){
  for(int j=0;j<policy_stream.patterns;j++)if((uint32_t)policy_stream.source[j]==pieces[k].source && policy_stream.dest[j]==(uint32_t)pieces[k].need)return j;
  abort();
}
static void policy_refresh(double t,int now){   /* the fabric at t, the forward, the table it makes (active now or after the lag) */
  struct route_state *st=&policy_stnext;memset(st,0,sizeof *st);st->tau=t;
  for(uint32_t l=0;l<links;l++){
    double w=INFINITY;int busy=0;
    for(uint32_t c=0;c<cables[l];c++){const double q=free_at[l][c]>t?free_at[l][c]-t:0;if(q<w)w=q;}
    if(reverse[l]!=UINT32_MAX)for(uint32_t c=0;c<cables[reverse[l]];c++)busy=busy||free_at[reverse[l]][c]>t;
    st->wait[l]=w;st->dup[l]=busy?duplex:1;
  }
  for(size_t e=0;e<events;e++){st->inflight[heap[e].node]+=pieces[heap[e].piece].bytes;st->posted+=pieces[heap[e].piece].bytes;}
  for(int a=0;a<ROUTE_N;a++)for(int b=0;b<ROUTE_N;b++)st->togo[a][b]=a==b?0:INFINITY;
  for(uint32_t l=0;l<links;l++){const double h=route_hop(&policy_stream,st,(int)l);if(h<st->togo[wire[l].from][wire[l].to])st->togo[wire[l].from][wire[l].to]=h;}
  for(int m=0;m<ROUTE_N;m++)for(int a=0;a<ROUTE_N;a++)for(int b=0;b<ROUTE_N;b++)if(st->togo[a][m]+st->togo[m][b]<st->togo[a][b])st->togo[a][b]=st->togo[a][m]+st->togo[m][b];
  double total[ROUTE_J]={0},got[ROUTE_J][ROUTE_N]={{0}};
  for(uint32_t k=0;k<npieces;k++){const int j=pattern_of_piece(k);total[j]++;for(int d=0;d<ROUTE_N;d++)if(pieces[k].got>>d&1)got[j][d]++;}
  for(int j=0;j<policy_stream.patterns;j++)for(int d=0;d<ROUTE_N;d++)st->recv[j][d]=total[j]>0?got[j][d]/total[j]:0;
  st->ops=1;
  for(int l=0;l<ROUTE_L;l++)route_lane_token(&policy_stream,st,&policy_topology,l,policy_lanes[l]);
  for(int c=0;c<policy_nc;c++)route_context_token(&policy_stream,st,&policy_topology,policy_pattern[c],policy_node[c],policy_owed[c],(int)(policy_refreshes%ROUTE_POINTS),policy_tokens+(size_t)c*ROUTE_IN);
  struct timespec a,b;clock_gettime(CLOCK_MONOTONIC,&a);
  route_forward(&policy_params,&policy_lanes[0][0],(int)links,policy_tokens,policy_nc,policy_state,policy_next,policy_scratch);
  clock_gettime(CLOCK_MONOTONIC,&b);policy_us+=(b.tv_sec-a.tv_sec)*1e6+(b.tv_nsec-a.tv_nsec)/1e3;
  policy_refreshes++;
  policy_at=now?t:t+policy_lag;
  if(now){float *x=policy_w;policy_w=policy_next;policy_next=x;policy_st=policy_stnext;policy_at=INFINITY;}
}
static uint32_t policy_route(double t,uint32_t from,uint32_t k,uint64_t to,uint64_t visited,uint32_t *lane,uint64_t *subset){
  if(t>=policy_at){float *x=policy_w;policy_w=policy_next;policy_next=x;policy_st=policy_stnext;policy_at=INFINITY;}   /* the newer table, now active */
  const int j=pattern_of_piece(k),c=policy_ctx[j][from][to];assert(c>=0);
  const double u=((mesh_switch_mix(((uint64_t)execution<<32)^k^((uint64_t)from<<56))>>11)+0.5)/9007199254740992.0;
  uint32_t tok=route_split(&policy_stream,&policy_st,&policy_topology,(int)from,(uint32_t)to,(uint32_t)visited,policy_w+(size_t)c*ROUTE_FS,u,policy_cdf);
  policy_draws++;
  const int deg=policy_topology.deg[from];uint64_t share[ROUTE_DEG]={0};
  for(int d=0;d<ROUTE_N;d++)if(to>>d&1){share[tok%(uint32_t)deg]|=UINT64_C(1)<<d;tok/=(uint32_t)deg;}
  uint32_t used=0;
  for(int q=0;q<deg;q++)if(share[q]){lane[used]=(uint32_t)policy_topology.out[from][q];subset[used]=share[q];used++;}
  return used;
}
#endif

static uint32_t earliest_cable(uint32_t l){
  uint32_t c=0;for(uint32_t k=1;k<cables[l];k++)if(free_at[l][k]<free_at[l][c])c=k;
  return c;
}

static void forward(double t,uint32_t from,uint32_t k,uint64_t to,uint64_t visited,uint32_t hops){
  double ahead[256];uint32_t lane[256];uint64_t subset[256];
  for(uint32_t l=0;l<links;l++){const double f=free_at[l][earliest_cable(l)];ahead[l]=mode!=FIXED && f>t?f-t:0;}
  const uint64_t key=mesh_switch_mix(((uint64_t)execution<<32)^k);
  uint32_t used=0;
  if(mode==TREES){   /* the piece's tree, sampled at its source; every hop sends on to its children in it */
    if(!hops){
      uint32_t j=0;while(j<ncommodities && !(csrc[j]==from && cto[j]==pieces[k].need))j++;
      assert(j<ncommodities);
      /* sampled from the multicast optimum's shares */
      double c=0;const double u=((key>>11)+0.5)/9007199254740992.0;uint32_t pick=tree_first[j+1]-1;
      for(uint32_t q=tree_first[j];q<tree_first[j+1];q++){c+=trees[q].share;if(u<c){pick=q;break;}}
      pieces[k].route=trees[pick];
    }
    const struct mesh_switch_tree *tr=&pieces[k].route;
    for(uint32_t v=0;v<nodes;v++)if(tr->parent[v]==from){
      uint64_t below=0;   /* the destinations in v's subtree */
      for(uint32_t d=0;d<nodes;d++)if(to>>d&1){uint32_t at=d;while(at!=v && tr->parent[at]!=UINT32_MAX)at=tr->parent[at];if(at==v)below|=UINT64_C(1)<<d;}
      if(below){lane[used]=tr->lane[v];subset[used]=below;used++;}
    }
  }
#ifdef ROUTE_POLICY
  else if(mode==POLICY)used=policy_route(t,from,k,to,visited,lane,subset);
#endif
  else used=mesh_switch_route(&fabric,from,t,to,visited,pieces[k].bytes,key,ahead,mode==SAMPLED?view[from]:NULL,unit,mode==SAMPLED,lane,subset);
  assert(used!=UINT32_MAX && used>0);
  uint64_t covered=0;
  for(uint32_t u=0;u<used;u++){
    const uint32_t l=lane[u];const struct mesh_switch_link *x=wire+l;
    assert(x->from==from && !(covered&subset[u]));covered|=subset[u];
    assert(!(visited>>x->to&1));   /* a simple path */
    const uint32_t c=earliest_cable(l);
    const double start=free_at[l][c]>t?free_at[l][c]:t;
    const int both=reverse[l]!=UINT32_MAX && c<cables[reverse[l]] && free_at[reverse[l]][c]>start;
    const double time=pieces[k].bytes/(x->rate/cables[l]*(both?duplex:1));
    free_at[l][c]=start+time;busy[l]+=time/cables[l];
    if(sends[l]==send_room[l]){
      send_room[l]=send_room[l]?2*send_room[l]:256;
      posted[l]=realloc(posted[l],send_room[l]*sizeof(double));waited[l]=realloc(waited[l],send_room[l]*sizeof(double));
    }
    posted[l][sends[l]]=t;waited[l][sends[l]]=start-t;sends[l]++;
    struct event e={start+time+x->latency,x->to,k,hops+1,subset[u],visited|UINT64_C(1)<<from,0,{0},{0},{{0,{0},{0},0}}};
    /* the name carries view lines: the sender's own lanes, then the others in turn */
    for(uint32_t m=0;m<links && e.carried<ENTRIES;m++)if(wire[m].from==from){
      e.lane[e.carried]=(uint8_t)m;e.execution[e.carried]=own_execution;e.line[e.carried++]=own[m];
    }
    for(uint32_t tries=0;tries<links && e.carried<ENTRIES;tries++){
      const uint32_t m=cursor[from]++%links;
      if(wire[m].from!=from && seen[from][m]){e.lane[e.carried]=(uint8_t)m;e.execution[e.carried]=seen[from][m];e.line[e.carried++]=view[from][m];}
    }
    push(e);
  }
  assert(covered==(to&~(UINT64_C(1)<<from)));   /* each destination in exactly one subset */
}

/* one execution; kind 0 all-gather, 1 all-reduce; S bytes a rank's partial; returns its completion */
static double execute(int kind,double S,double page){
  memset(free_at,0,sizeof free_at);memset(busy,0,sizeof busy);memset(sends,0,sizeof sends);
  const uint32_t P=(uint32_t)ceil(S/page);
  const uint64_t all=all_compute();
  npieces=0;
  pieces=realloc(pieces,(size_t)(2*ncompute*P+1)*sizeof *pieces);assert(pieces);
  uint32_t *left=calloc(P+1,sizeof *left);
  for(uint32_t c=0;c<ncompute;c++)for(uint32_t p=0;p<P;p++){
    const double b=p+1<P?page:S-page*(P-1);
    const uint32_t src=compute[c],owner=compute[p%ncompute];
    if(kind==0)pieces[npieces++]=(struct piece){src,p,b,all&~(UINT64_C(1)<<src),0,0,{{0},{0},0}};
    else if(src!=owner){pieces[npieces++]=(struct piece){src,p,b,UINT64_C(1)<<owner,0,1,{{0},{0},0}};left[p]++;}
  }
  const uint32_t first=npieces;
  if(kind==1)for(uint32_t p=0;p<P;p++){   /* the reduced pages, sent once their partials have landed */
    const uint32_t owner=compute[p%ncompute];
    pieces[npieces++]=(struct piece){owner,p,p+1<P?page:S-page*(P-1),all&~(UINT64_C(1)<<owner),0,2,{{0},{0},0}};
  }
#ifdef ROUTE_POLICY
  if(mode==POLICY){policy_refresh(0,1);policy_due=policy_dt;}
#endif
  for(uint32_t k=0;k<first;k++)forward(0,pieces[k].source,k,pieces[k].need,0,0);
  if(kind==1)for(uint32_t p=0;p<P;p++)if(!left[p])forward(0,pieces[first+p].source,first+p,pieces[first+p].need,0,0);
  double T=0;size_t steps=0;
  while(events){
#ifdef ROUTE_POLICY
    if(mode==POLICY)while(policy_due<=heap[0].t){policy_refresh(policy_due,0);policy_due+=policy_dt;}
#endif
    const struct event e=pop();
    assert(++steps<100000000 && e.hops<=nodes);
    for(uint32_t c=0;c<e.carried;c++)   /* this node's view: the newest line of each lane */
      if(e.execution[c]>seen[e.node][e.lane[c]]){view[e.node][e.lane[c]]=e.line[c];seen[e.node][e.lane[c]]=e.execution[c];}
    struct piece *x=pieces+e.piece;
    const uint64_t me=UINT64_C(1)<<e.node;
    if(e.to&me){
      assert(x->need&me && !(x->got&me));   /* exactly once */
      x->got|=me;if(e.t>T)T=e.t;
      if(x->phase==1 && !--left[x->page])forward(e.t,pieces[first+x->page].source,first+x->page,pieces[first+x->page].need,0,0);
    }
    if(e.to&~me)forward(e.t,e.node,e.piece,e.to&~me,e.visited,e.hops);
  }
  for(uint32_t k=0;k<npieces;k++)assert(pieces[k].got==pieces[k].need);
  free(left);
  /* each lane's sender: what its sends met, and its next line */
  for(uint32_t l=0;l<links;l++){
    double last=0;for(uint32_t c=0;c<cables[l];c++)if(free_at[l][c]>last)last=free_at[l][c];
    own[l]=mesh_switch_observe(posted[l],waited[l],sends[l],last,unit);
    view[wire[l].from][l]=own[l];seen[wire[l].from][l]=execution+1;
  }
  own_execution=++execution;
  return T;
}

struct stats { double mean,sd; };
static struct stats measure(int kind,double S,double page,int n){
  double sum=0,sq=0;
  for(int i=0;i<n;i++){const double T=execute(kind,S,page);sum+=T;sq+=T*T;}
  return (struct stats){sum/n,sqrt(fmax(0,sq/n-(sum/n)*(sum/n)))};
}
static void reset(int m){
  mode=m;memset(own,0,sizeof own);memset(view,0,sizeof view);memset(seen,0,sizeof seen);memset(cursor,0,sizeof cursor);
  execution=0;own_execution=0;
}

/* the commodities and the multicast optimum's trees for this transaction: all-gather a commodity a source to every
   other rank; all-reduce each partial to its owner (a commodity a pair) and each owner's reduced pages to every other
   rank. Then each commodity's shortest-path tree (the least delay to each destination) is added as a candidate, at
   no share. */
static void optimum(int kind,double S,double page){
  const uint64_t all=all_compute();
  ncommodities=0;
  for(uint32_t a=0;a<ncompute;a++){
    if(!kind){csrc[ncommodities]=compute[a];cto[ncommodities]=all&~(UINT64_C(1)<<compute[a]);tdemand[ncommodities++]=S;continue;}
    for(uint32_t b=0;b<ncompute;b++)if(a!=b){csrc[ncommodities]=compute[a];cto[ncommodities]=UINT64_C(1)<<compute[b];tdemand[ncommodities++]=S/ncompute;}
    csrc[ncommodities]=compute[a];cto[ncommodities]=all&~(UINT64_C(1)<<compute[a]);tdemand[ncommodities++]=S/ncompute;
  }
  static struct mesh_switch_tree packed[16384];uint32_t packed_first[65];
  assert(!mesh_switch_trees(&fabric,csrc,cto,tdemand,ncommodities,0.05,packed,sizeof packed/sizeof *packed,packed_first));
  double tload[256]={0};tree_fluid=0;
  for(uint32_t j=0;j<ncommodities;j++)for(uint32_t q=packed_first[j];q<packed_first[j+1];q++)
    for(uint32_t v=0;v<nodes;v++)if(packed[q].lane[v]!=UINT32_MAX)tload[packed[q].lane[v]]+=tdemand[j]*packed[q].share;
  for(uint32_t l=0;l<links;l++)if(tload[l]/wire[l].rate>tree_fluid)tree_fluid=tload[l]/wire[l].rate;
  double delay[256];for(uint32_t l=0;l<links;l++)delay[l]=wire[l].latency+page/(wire[l].rate/cables[l]);
  uint32_t n=0;
  for(uint32_t j=0;j<ncommodities;j++){
    tree_first[j]=n;
    for(uint32_t q=packed_first[j];q<packed_first[j+1];q++)trees[n++]=packed[q];
    struct mesh_switch_tree fast;   /* the unloaded shortest-path tree */
    assert(!mesh_switch_spt(&fabric,csrc[j],cto[j],delay,&fast));
    uint32_t q=tree_first[j];while(q<n && memcmp(trees[q].lane,fast.lane,nodes*sizeof *fast.lane))q++;
    if(q==n){fast.share=0;trees[n++]=fast;}
  }
  tree_first[ncommodities]=n;
}

/* the floor: the larger of the cut bound and the slowest source-destination pair's shortest delay for one piece. The
   cut bound is the most bytes a compute node must take in over its inbound rate, or put out over its outbound rate:
   - all-gather: in (n - 1) S (every other rank's partial), out S (its own, once);
   - all-reduce: in 2 (n - 1) S / n (the other ranks' partials of the pages it owns, then the other owners' reduced
     pages), out the same (its partials of the pages others own, then its own reduced pages, once). */
static double floor_of(int kind,double S,double page){
  double cut=0;
  const double need=kind?2*(ncompute-1)*S/ncompute:(ncompute-1)*S,give=kind?2*(ncompute-1)*S/ncompute:S;
  for(uint32_t v=0;v<nodes;v++){
    double in=0,out=0;
    for(uint32_t l=0;l<links;l++){if(wire[l].to==v)in+=wire[l].rate;if(wire[l].from==v)out+=wire[l].rate;}
    if(!(all_compute()>>v&1))continue;
    if(in>0 && need/in>cut)cut=need/in;
    if(out>0 && give/out>cut)cut=give/out;
  }
  /* the slowest pair's shortest delay for one piece (no larger than a page) */
  const double q=S<page?S:page;
  double dist[MESH_SWITCH_NODES][MESH_SWITCH_NODES];
  for(uint32_t a=0;a<nodes;a++)for(uint32_t b=0;b<nodes;b++)dist[a][b]=a==b?0:INFINITY;
  for(uint32_t l=0;l<links;l++){const double c=wire[l].latency+q/(wire[l].rate/cables[l]);if(c<dist[wire[l].from][wire[l].to])dist[wire[l].from][wire[l].to]=c;}
  for(uint32_t k=0;k<nodes;k++)for(uint32_t a=0;a<nodes;a++)for(uint32_t b=0;b<nodes;b++)if(dist[a][k]+dist[k][b]<dist[a][b])dist[a][b]=dist[a][k]+dist[k][b];
  double reach=0;
  for(uint32_t a=0;a<ncompute;a++)for(uint32_t b=0;b<ncompute;b++)if(a!=b && dist[compute[a]][compute[b]]>reach)reach=dist[compute[a]][compute[b]];
  const double floor=cut>reach?cut:reach;
  return floor;
}

int main(int argc,char **argv){
  load(argc>1?argv[1]:"tests/fabric-20261119.txt");
  const double page=argc>2?atof(argv[2]):16384;
  assert(!mesh_switch_fabric_make(&fabric,nodes,wire,links,page));
  static const double rows[]={2,8,32,128,1024,4096};
#ifdef ROUTE_POLICY
  const char *theta=getenv("ROUTE_THETA");
  if(theta)policy_load(theta,getenv("ROUTE_GRU")!=NULL);
#endif
  printf("%-10s %9s %9s %18s %18s %9s %9s\n","","S bytes","fixed","sampled","trees","fluid","floor");
  for(int kind=0;kind<2;kind++)for(size_t i=0;i<sizeof rows/sizeof *rows;i++){
    const double S=rows[i]*1536*2;
    reset(FIXED);const double fixed=execute(kind,S,page);
    reset(SAMPLED);for(int k=0;k<LEARN;k++)execute(kind,S,page);const struct stats sampled=measure(kind,S,page,MEASURE);
    reset(TREES);optimum(kind,S,page);const struct stats tr=measure(kind,S,page,MEASURE);
    const double floor=floor_of(kind,S,page);
    printf("%-10s %9.0f %7.2fus %8.2f+-%6.2fus %8.2f+-%6.2fus ",kind?"all-reduce":"all-gather",S,fixed/1e3,sampled.mean/1e3,sampled.sd/1e3,tr.mean/1e3,tr.sd/1e3);
    printf("%7.2fus %7.2fus",tree_fluid/1e3,floor/1e3);
#ifdef ROUTE_POLICY
    if(theta){
      reset(POLICY);policy_setup(kind,S);const struct stats pol=measure(kind,S,page,MEASURE);
      printf("   policy %8.2f+-%6.2fus (%llu refreshes at %.0f us each, %llu draws)",pol.mean/1e3,pol.sd/1e3,(unsigned long long)policy_refreshes,policy_us/(double)policy_refreshes,(unsigned long long)policy_draws);
      assert(kind || pol.mean>=floor*(1-1e-9));
    }
#endif
    printf("\n");
    assert(kind || (sampled.mean>=floor*(1-1e-9) && tr.mean>=floor*(1-1e-9)));
  }
  printf("switch: every page to every destination exactly once, every path simple: passed\n");
  free(heap);free(pieces);
  for(uint32_t l=0;l<links;l++){free(posted[l]);free(waited[l]);}
  return 0;
}
