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
     sampling, the lines from each lane's sender's GRU, carried in names);
   - trees: each piece's tree sampled from the multicast fluid optimum (mesh_switch_trees) with its swizzled quantile;
   - policy (with --descent): each piece chooses, at its source, one whole tree among candidates (the optimum's
     trees, the unloaded shortest-path tree, and the shortest-path tree under the lines' predicted delays at that
     moment) by sampling a softmax of a plain network's logits over one concatenated input vector a candidate
     (mesh_switch_choose_tree), never an argmax; every hop forwards to its children in it. The network starts
     random; its parameters move by online Jacobian descent over the triplet's two latency goals, the mean and the
     spread of T / floor, averaged over every transaction and size (gradients by evolution strategies, applied by
     Adam).

   The sampled, trees and policy rows are the mean and spread of the completion over 20 executions after any
   learning. The floor is the larger of two bounds: the all-gather's cut bound (the most bytes a compute node needs
   over its inbound rate, or must emit over its outbound rate), and the slowest source-destination pair's shortest
   delay for one piece (latency and the piece's time on one cable, each hop; a piece is at most a page).

   Checked every run: every page reaches every destination that needs it exactly once, the run ends, and every path
   is simple.

   usage: switch [--descent] [fabric [page]] */
#include "mesh-switch.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
static struct mesh_switch_net policy;static double current_S;static int current_kind;
/* each node's model state, and its published weighting a commodity (the double buffer the posting path reads) */
#define REFRESH 2500   /* ns between a node's refreshes: about the model's 0.25 MFLOP a step on the bridge's CPU */
static struct mesh_switch_net_state nstate[MESH_SWITCH_NODES];static double refresh_at[MESH_SWITCH_NODES];
static struct published { uint32_t n; struct mesh_switch_tree cand[64]; double p[64]; } published[64];
static double occupy[256][CABLES];   /* each cable's background occupancy at a transaction's start (ns) */
static int mode;
static uint32_t execution;
static double free_at[256][CABLES],busy[256];
static double *posted[256],*waited[256];static uint32_t sends[256],send_room[256];
static struct mesh_switch_line own[256],view[MESH_SWITCH_NODES][256];
static uint32_t own_execution,seen[MESH_SWITCH_NODES][256],cursor[MESH_SWITCH_NODES];
static struct mesh_switch_gru gru[256];static struct mesh_switch_gru_weights weights;
static struct piece *pieces;static uint32_t npieces;
/* the trees a piece is sampled from: commodity j's candidates at tree_first[j] .. tree_first[j + 1] - 1 */
static struct mesh_switch_tree trees[4096];static uint32_t tree_first[65],ncommodities,csrc[64];static uint64_t cto[64];
static double tdemand[64],tree_fluid;

static uint32_t earliest_cable(uint32_t l){
  uint32_t c=0;for(uint32_t k=1;k<cables[l];k++)if(free_at[l][k]<free_at[l][c])c=k;
  return c;
}

/* node u's refresh of its published weightings, one a commodity it sources: candidates the optimum's trees, the
   unloaded shortest-path tree (both from optimum()), and the shortest-path tree under its lines' predicted delays now */
static void refresh(uint32_t u,double t,const double *ahead){
  refresh_at[u]=t+REFRESH;
  const double bytes=current_S<fabric.page?current_S:fabric.page;
  for(uint32_t j=0;j<ncommodities;j++){
    if(csrc[j]!=u)continue;
    struct published *pb=published+j;uint8_t spt[64];
    pb->n=0;
    for(uint32_t q=tree_first[j];q<tree_first[j+1] && pb->n<63;q++){spt[pb->n]=trees[q].share>0?0:1;pb->cand[pb->n++]=trees[q];}
    double delay[256];
    for(uint32_t l=0;l<links;l++)delay[l]=(wire[l].from==u?ahead[l]:mesh_switch_wait(view[u]+l,unit,t,0))+bytes/wire[l].rate+wire[l].latency;
    if(!mesh_switch_spt(&fabric,u,cto[j],delay,pb->cand+pb->n)){spt[pb->n]=2;pb->n++;}
    const int reduced=current_kind && __builtin_popcountll(cto[j])>1;
    mesh_switch_tree_weights(&fabric,u,t,cto[j],bytes,current_S,current_kind,reduced,ahead,view[u],unit,pb->cand,spt,pb->n,&policy,nstate+u,pb->p);
  }
}

static void forward(double t,uint32_t from,uint32_t k,uint64_t to,uint64_t visited,uint32_t hops){
  double ahead[256];uint32_t lane[256];uint64_t subset[256];
  for(uint32_t l=0;l<links;l++){const double f=free_at[l][earliest_cable(l)];ahead[l]=mode!=FIXED && f>t?f-t:0;}
  const uint64_t key=mesh_switch_mix(((uint64_t)execution<<32)^k);
  uint32_t used=0;
  if(mode==TREES || mode==POLICY){   /* the piece's tree, chosen at its source; every hop sends on to its children in it */
    if(!hops){
      uint32_t j=0;while(j<ncommodities && !(csrc[j]==from && cto[j]==pieces[k].need))j++;
      assert(j<ncommodities);
      if(mode==TREES){   /* sampled from the multicast optimum's shares */
        double c=0;const double u=((key>>11)+0.5)/9007199254740992.0;uint32_t pick=tree_first[j+1]-1;
        for(uint32_t q=tree_first[j];q<tree_first[j+1];q++){c+=trees[q].share;if(u<c){pick=q;break;}}
        pieces[k].route=trees[pick];
      }else{   /* the node's published weighting for this commodity, refreshed off the posting path */
        if(t>=refresh_at[from])refresh(from,t,ahead);
        const struct published *pb=published+j;
        pieces[k].route=pb->cand[mesh_switch_sample(pb->p,pb->n,key)];
      }
    }
    const struct mesh_switch_tree *tr=&pieces[k].route;
    for(uint32_t v=0;v<nodes;v++)if(tr->parent[v]==from){
      uint64_t below=0;   /* the destinations in v's subtree */
      for(uint32_t d=0;d<nodes;d++)if(to>>d&1){uint32_t at=d;while(at!=v && tr->parent[at]!=UINT32_MAX)at=tr->parent[at];if(at==v)below|=UINT64_C(1)<<d;}
      if(below){lane[used]=tr->lane[v];subset[used]=below;used++;}
    }
  }else used=mesh_switch_route(&fabric,from,t,to,visited,pieces[k].bytes,key,ahead,mode==SAMPLED?view[from]:NULL,unit,mode==SAMPLED,lane,subset);
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
  current_S=S;current_kind=kind;
  memcpy(free_at,occupy,sizeof free_at);memset(busy,0,sizeof busy);memset(sends,0,sizeof sends);
  for(uint32_t v=0;v<nodes;v++)refresh_at[v]=0;
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
  for(uint32_t k=0;k<first;k++)forward(0,pieces[k].source,k,pieces[k].need,0,0);
  if(kind==1)for(uint32_t p=0;p<P;p++)if(!left[p])forward(0,pieces[first+p].source,first+p,pieces[first+p].need,0,0);
  double T=0;size_t steps=0;
  while(events){
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
    const struct mesh_switch_line met=mesh_switch_observe(posted[l],waited[l],sends[l],last,unit);
    mesh_switch_gru_step(gru+l,&weights,&met,unit,own+l);
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
  memset(gru,0,sizeof gru);execution=0;own_execution=0;memset(nstate,0,sizeof nstate);
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
  static struct mesh_switch_tree packed[4096];uint32_t packed_first[65];
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

/* one scenario's learned state, saved and restored by copying so both sides of a perturbation start alike */
struct world {
  struct mesh_switch_line own[256],view[MESH_SWITCH_NODES][256];uint32_t own_execution,execution,seen[MESH_SWITCH_NODES][256],cursor[MESH_SWITCH_NODES];
  struct mesh_switch_gru gru[256];struct mesh_switch_net_state nstate[MESH_SWITCH_NODES];
};
#define SAVE(w) do{memcpy((w)->own,own,sizeof own);memcpy((w)->view,view,sizeof view);(w)->own_execution=own_execution;(w)->execution=execution;\
  memcpy((w)->seen,seen,sizeof seen);memcpy((w)->cursor,cursor,sizeof cursor);memcpy((w)->gru,gru,sizeof gru);memcpy((w)->nstate,nstate,sizeof nstate);}while(0)
#define LOAD(w) do{memcpy(own,(w)->own,sizeof own);memcpy(view,(w)->view,sizeof view);own_execution=(w)->own_execution;execution=(w)->execution;\
  memcpy(seen,(w)->seen,sizeof seen);memcpy(cursor,(w)->cursor,sizeof cursor);memcpy(gru,(w)->gru,sizeof gru);memcpy(nstate,(w)->nstate,sizeof nstate);}while(0)

/* A random scenario, the domain the policy is trained over: 3 to 6 nodes, up to 2 of them relays only, connected (a
   random spanning tree, then each other pair joined with probability 0.35 while both have fewer than 4 neighbours);
   each joined pair 1 to 4 cables, each direction its own rate a cable (4 to 12 GB/s) and latency (2 to 13 us); a
   duplex share of 0.8 to 1; each lane, with probability 0.3, its cables busy with background traffic for up to
   50 us at the start; a transaction all-gather or all-reduce, its size a rank log-uniform from 6 KB to 3 MB. */
static double U(uint64_t *seed){*seed=mesh_switch_mix(*seed);return ((*seed>>11)+0.5)/9007199254740992.0;}
static void randomize(uint64_t seed,int *kind,double *S){
  nodes=3+(uint32_t)(U(&seed)*4);
  uint32_t relays=(uint32_t)(U(&seed)*3);if(relays>nodes-2)relays=nodes-2;
  ncompute=0;
  for(uint32_t v=0;v<nodes;v++){snprintf(names[v],sizeof names[v],"n%u",v);if(v<nodes-relays)compute[ncompute++]=v;}
  int joined[MESH_SWITCH_NODES][MESH_SWITCH_NODES]={{0}};uint32_t degree[MESH_SWITCH_NODES]={0};
  for(uint32_t v=1;v<nodes;v++){uint32_t u=(uint32_t)(U(&seed)*v);joined[u][v]=joined[v][u]=1;degree[u]++;degree[v]++;}
  for(uint32_t u=0;u<nodes;u++)for(uint32_t v=u+1;v<nodes;v++)
    if(!joined[u][v] && degree[u]<4 && degree[v]<4 && U(&seed)<0.35){joined[u][v]=joined[v][u]=1;degree[u]++;degree[v]++;}
  links=0;
  for(uint32_t u=0;u<nodes;u++)for(uint32_t v=u+1;v<nodes;v++)if(joined[u][v]){
    const uint32_t c=1+(uint32_t)(U(&seed)*4);
    for(int w=0;w<2;w++){
      cables[links]=c;
      wire[links]=(struct mesh_switch_link){w?v:u,w?u:v,c*(4+8*U(&seed)),(2+11*U(&seed))*1e3};
      links++;
    }
  }
  duplex=0.8+0.2*U(&seed);
  for(uint32_t l=0;l<links;l++){
    reverse[l]=UINT32_MAX;
    for(uint32_t r=0;r<links;r++)if(wire[r].from==wire[l].to && wire[r].to==wire[l].from)reverse[l]=r;
    const int busy_lane=U(&seed)<0.3;
    for(uint32_t c=0;c<CABLES;c++)occupy[l][c]=busy_lane && c<cables[l]?50e3*U(&seed):0;
  }
  *kind=U(&seed)<0.5;
  *S=round(exp(log(6144)+U(&seed)*(log(3145728)-log(6144))));
}

/* One scenario, for one perturbation: the lines warmed up over 3 executions at the base parameters, then 2 executions
   each at base + sigma eps and base - sigma eps from the same state. Its returns: for each side, the mean of T / floor
   and its spread (half the two executions' difference, over the floor). */
static void scenario(uint64_t seed,const float *base,const float *plus,const float *minus,double out[4],double page){
  int kind;double S;
  randomize(seed,&kind,&S);
  assert(!mesh_switch_fabric_make(&fabric,nodes,wire,links,page));
  const double floor=floor_of(kind,S,page);
  reset(POLICY);optimum(kind,S,page);
  memcpy(&policy,base,sizeof policy);
  for(int k=0;k<3;k++)execute(kind,S,page);
  static struct world w;SAVE(&w);
  for(int side=0;side<2;side++){
    memcpy(&policy,side?minus:plus,sizeof policy);LOAD(&w);
    const double T1=execute(kind,S,page),T2=execute(kind,S,page);
    out[2*side]=(T1+T2)/2/floor;out[2*side+1]=fabs(T1-T2)/2/floor;
  }
}

/* the perturbation of pair m at step k: a Gaussian vector from its seed, the same on every worker */
static void perturbation(uint64_t k,uint64_t m,float *eps){
  uint64_t r=mesh_switch_mix(0xe5e5ULL^(k*1000003ULL+m));
  for(size_t i=0;i<MESH_NET_PARAMS;i++){r=mesh_switch_mix(r);eps[i]=(float)mesh_switch_draw(r);}
}

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
static int full(int fd,void *p,size_t n,int out){
  char *c=p;
  while(n){const ssize_t k=out?write(fd,c,n):read(fd,c,n);if(k<=0)return -1;c+=k;n-=(size_t)k;}
  return 0;
}

/* Distributed training (--train): rank 0 listens on the root's port and the other ranks connect to it. Each step rank 0
   sends every worker the parameters; each rank runs the scenarios of the pairs m with m mod ranks its rank (one
   scenario a pair, its seed from the step and the pair) and returns their results; rank 0 then takes each goal's
   gradient by antithetic evolution strategies (centered ranks of the returns, times each pair's perturbation, over
   2 M sigma), the two combined at their min-norm point (MGDA for two), and steps by Adam. It writes the parameters to
   `out` every 10 steps and logs the mean returns. Workers run at the lowest priority, beside the bridges. */
struct result { uint32_t m,pad; double v[4]; };
static int rank_compare_values[1];
static int by_value(const void *a,const void *b){const double x=**(const double *const *)a,y=**(const double *const *)b;return (x>y)-(x<y);}
static void centered_ranks(double *v,size_t n){
  const double **p=malloc(n*sizeof *p);double *r=malloc(n*sizeof *r);
  for(size_t i=0;i<n;i++)p[i]=v+i;
  qsort(p,n,sizeof *p,by_value);
  for(size_t i=0;i<n;i++)r[p[i]-v]=(double)i/(n-1)-0.5;
  memcpy(v,r,n*sizeof *v);free(p);free(r);(void)rank_compare_values;
}
static int train(int rank,int ranks,const char *root,int steps,int pairs,const char *out,const char *resume,double page){
  const size_t P=MESH_NET_PARAMS;const double sigma=0.02,rate=0.01,beta1=0.9,beta2=0.999;
  float *theta=malloc(P*sizeof *theta),*eps=malloc(P*sizeof *eps),*plus=malloc(P*sizeof *plus),*minus=malloc(P*sizeof *minus);
  double *m1=calloc(P,sizeof *m1),*m2=calloc(P,sizeof *m2),*g1=malloc(P*sizeof *g1),*g2=malloc(P*sizeof *g2);
  struct result *res=malloc((size_t)pairs*sizeof *res);
  char host[64];int port;
  if(sscanf(root,"%63[^:]:%d",host,&port)!=2){fprintf(stderr,"switch: --root host:port\n");return 2;}
  int *peer=NULL,fd=-1;
  if(rank==0){
    if(resume){FILE *f=fopen(resume,"rb");if(!f || fread(theta,sizeof *theta,P,f)!=P){fprintf(stderr,"switch: %s\n",resume);return 2;}fclose(f);}
    else mesh_switch_net_init((struct mesh_switch_net *)theta,20261119);
    const int ls=socket(AF_INET,SOCK_STREAM,0),one=1;
    setsockopt(ls,SOL_SOCKET,SO_REUSEADDR,&one,sizeof one);
    struct sockaddr_in a={0};a.sin_family=AF_INET;a.sin_port=htons((uint16_t)port);a.sin_addr.s_addr=INADDR_ANY;
    if(bind(ls,(struct sockaddr *)&a,sizeof a) || listen(ls,256)){perror("switch: listen");return 2;}
    peer=calloc((size_t)ranks,sizeof *peer);
    for(int k=1;k<ranks;k++){
      const int c=accept(ls,NULL,NULL);int r;
      if(c<0 || full(c,&r,sizeof r,0) || r<1 || r>=ranks){perror("switch: accept");return 2;}
      setsockopt(c,IPPROTO_TCP,TCP_NODELAY,&one,sizeof one);peer[r]=c;
    }
    close(ls);
    fprintf(stderr,"switch: %d ranks joined, %zu parameters, %d pairs a step\n",ranks,P,pairs);
  }else{
    struct sockaddr_in a={0};a.sin_family=AF_INET;a.sin_port=htons((uint16_t)port);inet_pton(AF_INET,host,&a.sin_addr);
    for(int tries=0;;tries++){
      fd=socket(AF_INET,SOCK_STREAM,0);
      if(!connect(fd,(struct sockaddr *)&a,sizeof a))break;
      close(fd);if(tries>600){perror("switch: connect");return 2;}
      usleep(100000);
    }
    const int one=1;setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof one);
    if(full(fd,&rank,sizeof rank,1))return 2;
  }
  for(int k=0;k<steps;k++){
    if(rank==0){for(int r=1;r<ranks;r++)if(full(peer[r],theta,P*sizeof *theta,1))return 2;}
    else if(full(fd,theta,P*sizeof *theta,0))return 0;   /* the root is done */
    uint32_t mine=0;
    for(int m=rank;m<pairs;m+=ranks){
      perturbation((uint64_t)k,(uint64_t)m,eps);
      for(size_t i=0;i<P;i++){plus[i]=theta[i]+(float)sigma*eps[i];minus[i]=theta[i]-(float)sigma*eps[i];}
      res[mine].m=(uint32_t)m;scenario(mesh_switch_mix(0x5ce7ULL^((uint64_t)k*1000003ULL+(uint64_t)m)),theta,plus,minus,res[mine].v,page);mine++;
    }
    if(rank){if(full(fd,&mine,sizeof mine,1) || full(fd,res,mine*sizeof *res,1))return 2;continue;}
    uint32_t got=mine;
    for(int r=1;r<ranks;r++){uint32_t n;if(full(peer[r],&n,sizeof n,0) || full(peer[r],res+got,n*sizeof *res,0))return 2;got+=n;}
    /* the gradients: centered ranks of each goal's 2 M returns, then sum over pairs of (rank+ - rank-) eps / (2 M sigma) */
    double *a=malloc(2*(size_t)got*sizeof *a),*b=malloc(2*(size_t)got*sizeof *b),mean=0,spread=0;
    for(uint32_t i=0;i<got;i++){a[2*i]=res[i].v[0];a[2*i+1]=res[i].v[2];b[2*i]=res[i].v[1];b[2*i+1]=res[i].v[3];mean+=(res[i].v[0]+res[i].v[2])/(2*got);spread+=(res[i].v[1]+res[i].v[3])/(2*got);}
    centered_ranks(a,2*got);centered_ranks(b,2*got);
    for(size_t i=0;i<P;i++)g1[i]=g2[i]=0;
    for(uint32_t i=0;i<got;i++){
      perturbation((uint64_t)k,res[i].m,eps);
      const double da=(a[2*i]-a[2*i+1])/(2*got*sigma),db=(b[2*i]-b[2*i+1])/(2*got*sigma);
      for(size_t q=0;q<P;q++){g1[q]+=da*eps[q];g2[q]+=db*eps[q];}
    }
    free(a);free(b);
    double g11=0,g22=0,g12=0;
    for(size_t q=0;q<P;q++){g11+=g1[q]*g1[q];g22+=g2[q]*g2[q];g12+=g1[q]*g2[q];}
    const double den=g11-2*g12+g22;
    double lambda=den>0?(g22-g12)/den:0.5;
    if(lambda<0)lambda=0;
    if(lambda>1)lambda=1;
    for(size_t q=0;q<P;q++){
      const double d=lambda*g1[q]+(1-lambda)*g2[q];
      m1[q]=beta1*m1[q]+(1-beta1)*d;m2[q]=beta2*m2[q]+(1-beta2)*d*d;
      theta[q]-=(float)(rate*(m1[q]/(1-pow(beta1,k+1)))/(sqrt(m2[q]/(1-pow(beta2,k+1)))+1e-8));
    }
    printf("step %d: T / floor %.4f, spread %.4f, lambda %.2f, %u scenarios\n",k,mean,spread,lambda,got);fflush(stdout);
    if(out && (k%10==9 || k+1==steps)){
      char tmp[512];snprintf(tmp,sizeof tmp,"%s.tmp",out);
      FILE *f=fopen(tmp,"wb");if(f){fwrite(theta,sizeof *theta,P,f);fclose(f);rename(tmp,out);}
    }
  }
  if(rank==0)for(int r=1;r<ranks;r++)close(peer[r]);
  return 0;
}

int main(int argc,char **argv){
  const char *evaluate=NULL,*root=NULL,*out=NULL,*resume=NULL;
  int trainer=0,rank=0,ranks=1,steps=100,pairs=64;
  while(argc>1 && !strncmp(argv[1],"--",2)){
    const char *a=argv[1];
    if(!strncmp(a,"--eval=",7))evaluate=a+7;
    else if(!strcmp(a,"--train"))trainer=1;
    else if(!strncmp(a,"--rank=",7))rank=atoi(a+7);
    else if(!strncmp(a,"--ranks=",8))ranks=atoi(a+8);
    else if(!strncmp(a,"--root=",7))root=a+7;
    else if(!strncmp(a,"--steps=",8))steps=atoi(a+8);
    else if(!strncmp(a,"--pairs=",8))pairs=atoi(a+8);
    else if(!strncmp(a,"--out=",6))out=a+6;
    else if(!strncmp(a,"--resume=",9))resume=a+9;
    else{fprintf(stderr,"switch: %s?\n",a);return 2;}
    argc--;argv++;
  }
  mesh_switch_gru_weights_make(&weights,20261119);
  const double page=argc>2?atof(argv[2]):16384;
  if(trainer)return train(rank,ranks,root?root:"127.0.0.1:47000",steps,pairs,out,resume,page);
  load(argc>1?argv[1]:"tests/fabric-20261119.txt");
  assert(!mesh_switch_fabric_make(&fabric,nodes,wire,links,page));
  if(evaluate){FILE *f=fopen(evaluate,"rb");if(!f || fread(&policy,sizeof(float),MESH_NET_PARAMS,f)!=MESH_NET_PARAMS){fprintf(stderr,"switch: %s\n",evaluate);return 2;}fclose(f);}
  static const double rows[]={2,8,32,128,1024,4096};
  printf("%-10s %9s %9s %18s %18s %20s %9s %9s\n","","S bytes","fixed","sampled","trees","policy","fluid","floor");
  for(int kind=0;kind<2;kind++)for(size_t i=0;i<sizeof rows/sizeof *rows;i++){
    const double S=rows[i]*1536*2;
    reset(FIXED);const double fixed=execute(kind,S,page);
    reset(SAMPLED);for(int k=0;k<LEARN;k++)execute(kind,S,page);const struct stats sampled=measure(kind,S,page,MEASURE);
    reset(TREES);optimum(kind,S,page);const struct stats tr=measure(kind,S,page,MEASURE);
    struct stats ds={0,0};
    if(evaluate){reset(POLICY);optimum(kind,S,page);for(int k=0;k<LEARN;k++)execute(kind,S,page);ds=measure(kind,S,page,MEASURE);}
    const double floor=floor_of(kind,S,page);
    printf("%-10s %9.0f %7.2fus %8.2f+-%6.2fus %8.2f+-%6.2fus ",kind?"all-reduce":"all-gather",S,fixed/1e3,sampled.mean/1e3,sampled.sd/1e3,tr.mean/1e3,tr.sd/1e3);
    if(evaluate)printf("%10.2f+-%6.2fus ",ds.mean/1e3,ds.sd/1e3);else printf("%20s ","-");
    printf("%7.2fus %7.2fus\n",tree_fluid/1e3,floor/1e3);
    assert(kind || (sampled.mean>=floor*(1-1e-9) && tr.mean>=floor*(1-1e-9)));
  }
  printf("switch: every page to every destination exactly once, every path simple: passed\n");
  free(heap);free(pieces);
  for(uint32_t l=0;l<links;l++){free(posted[l]);free(waited[l]);}
  return 0;
}
