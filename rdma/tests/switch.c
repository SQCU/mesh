/* A discrete-event run of mesh-switch.h's forwarding rule (design/switched-multipath-dma.md) on a fabric file (default
   tests/fabric-20261119.txt, the five-node cluster as measured). Each directed link is a lane, FIFO at its measured rate
   with its measured latency. A node's bridge forwards a piece the moment it lands.

   Two transactions, at each rank-partial size S:
   - all-gather: each compute rank's S bytes, as pages, to every other compute rank;
   - all-reduce: each page's partial from every rank to its owner (page p's owner is compute rank p mod n), and the
     reduced page from the owner to every other rank once its last partial has landed. No barrier between the two.

   Each lane's sender turns what its sends met in an execution into the lane's next line (mesh_switch_gru_step).
   Every SEND's name carries ENTRIES of the sender's view lines, its own lanes and the others in turn, each with the
   execution it is from, and a node keeps the newest it has seen. The transaction runs 40 times; the mean and spread
   are of the last 20. Four ways, on the same inputs:
   - fixed: every backlog and wait taken as none (fixed shortest-delay routes);
   - own: each node's own lanes at their posted backlog, later hops idle;
   - mean: own backlog, and every later hop at its line's predicted wait;
   - sampled: own backlog, and every later hop at a wait drawn from its line (Thompson sampling, the draw a swizzle of
     the piece, the node and the lane).

   Checked every run: every page reaches every destination that needs it exactly once, the run ends, and every path
   is simple. For the all-gather, a lower bound: the most inbound bytes a compute node needs over its inbound rate, or
   the most bytes one must emit over its outbound rate.

   usage: switch [fabric [page]] */
#include "mesh-switch.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ENTRIES 4   /* view lines a name carries, 64 bytes each */
#define RUNS 40

static char names[MESH_SWITCH_NODES][16];
static uint32_t nodes,links,compute[MESH_SWITCH_NODES],ncompute;
static struct mesh_switch_link link[256];

static uint32_t node_of(const char *s){
  for(uint32_t i=0;i<nodes;i++)if(!strcmp(names[i],s))return i;
  fprintf(stderr,"switch: no node %s\n",s);exit(2);
}
static void load(const char *path){
  FILE *f=fopen(path,"r");
  if(!f){perror(path);exit(2);}
  char line[256];
  while(fgets(line,sizeof line,f)){
    char a[16],b[16],kind[16];double rate,us;
    if(line[0]=='#')continue;
    if(sscanf(line,"node %15s %15s",a,kind)==2){
      assert(nodes<MESH_SWITCH_NODES);
      snprintf(names[nodes],sizeof names[nodes],"%s",a);
      if(!strcmp(kind,"compute"))compute[ncompute++]=nodes;
      nodes++;
    }else if(sscanf(line,"link %15s %15s %lf %lf",a,b,&rate,&us)==4){
      assert(links<sizeof link/sizeof *link);
      link[links++]=(struct mesh_switch_link){node_of(a),node_of(b),rate,us*1e3};
    }
  }
  fclose(f);
}

struct piece { uint32_t source,page; double bytes; uint64_t need,got; int phase; uint32_t route[MESH_SWITCH_NODES]; };
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
static int mode;   /* 0 fixed, 1 own, 2 mean, 3 sampled, 4 the unicast fluid optimum's paths, 5 the multicast fluid optimum's trees */
static struct mesh_switch_tree trees[4096];static uint32_t tree_first[65],ncommodities;static uint32_t csrc[64];static uint64_t cto[64];
static double fluid,tree_fluid;
static uint64_t all_compute(void);static struct mesh_switch_path paths[4096];static uint32_t path_first[65],commodity[MESH_SWITCH_NODES][MESH_SWITCH_NODES];
static uint32_t execution;
static double free_at[256],busy[256];
static double *posted[256],*waited[256];static uint32_t sends[256],send_room[256];
static struct mesh_switch_line own[256],view[MESH_SWITCH_NODES][256];
static uint32_t own_execution,seen[MESH_SWITCH_NODES][256],cursor[MESH_SWITCH_NODES];
static struct mesh_switch_gru gru[256];static struct mesh_switch_gru_weights weights;
static struct piece *pieces;static uint32_t npieces,relayed,forwarded,maxhops;

static void send(double t,uint32_t from,uint32_t k,uint64_t to,uint64_t visited,uint32_t hops){
  double ahead[256];uint32_t lane[256];uint64_t subset[256];
  for(uint32_t l=0;l<links;l++)ahead[l]=mode && free_at[l]>t?free_at[l]-t:0;
  const uint64_t key=mesh_switch_mix(((uint64_t)execution<<32)^k);
  uint32_t used;
  if(mode==4){   /* each destination's path, sampled at the source with one quantile a piece; every hop follows it */
    if(!hops)for(uint32_t d=0;d<nodes;d++)if(to>>d&1 && d!=from){
      const uint32_t j=commodity[from][d];
      pieces[k].route[d]=path_first[j]+mesh_switch_pick(paths+path_first[j],path_first[j+1]-path_first[j],((key>>11)+0.5)/9007199254740992.0);
    }
    used=0;
    for(uint32_t d=0;d<nodes;d++)if(to>>d&1 && d!=from){
      const struct mesh_switch_path *q=paths+pieces[k].route[d];
      uint32_t i=0;while(i<q->hops && link[q->lane[i]].from!=from)i++;
      assert(i<q->hops);
      uint32_t u=0;while(u<used && lane[u]!=q->lane[i])u++;
      if(u==used){lane[used]=q->lane[i];subset[used]=0;used++;}
      subset[u]|=UINT64_C(1)<<d;
    }
  }else if(mode==5){   /* the piece's tree, sampled at its source; every hop sends on to its children in the tree */
    if(!hops){
      uint32_t j=0;while(j<ncommodities && !(csrc[j]==from && cto[j]==pieces[k].need))j++;
      assert(j<ncommodities);
      double c=0,u=((key>>11)+0.5)/9007199254740992.0;uint32_t pick=tree_first[j+1]-1;
      for(uint32_t q=tree_first[j];q<tree_first[j+1];q++){c+=trees[q].share;if(u<c){pick=q;break;}}
      pieces[k].route[0]=pick;
    }
    const struct mesh_switch_tree *tr=trees+pieces[k].route[0];
    used=0;
    for(uint32_t v=0;v<nodes;v++)if(tr->parent[v]==from){
      uint64_t below=0;   /* the destinations in v's subtree */
      for(uint32_t d=0;d<nodes;d++)if(to>>d&1){uint32_t at=d;while(at!=v && tr->parent[at]!=UINT32_MAX)at=tr->parent[at];if(at==v)below|=UINT64_C(1)<<d;}
      if(below){lane[used]=tr->lane[v];subset[used]=below;used++;}
    }
  }else used=mesh_switch_route(&fabric,from,t,to,visited,pieces[k].bytes,key,ahead,mode>=2?view[from]:NULL,unit,mode==3,lane,subset);
  assert(used!=UINT32_MAX && used>0);
  uint64_t covered=0;
  for(uint32_t u=0;u<used;u++){
    const uint32_t l=lane[u];const struct mesh_switch_link *x=link+l;
    assert(x->from==from && !(covered&subset[u]));covered|=subset[u];
    assert(!(visited>>x->to&1));   /* a simple path */
    const double start=free_at[l]>t?free_at[l]:t,time=pieces[k].bytes/x->rate;
    free_at[l]=start+time;busy[l]+=time;forwarded++;
    if(sends[l]==send_room[l]){
      send_room[l]=send_room[l]?2*send_room[l]:256;
      posted[l]=realloc(posted[l],send_room[l]*sizeof(double));waited[l]=realloc(waited[l],send_room[l]*sizeof(double));
    }
    posted[l][sends[l]]=t;waited[l][sends[l]]=start-t;sends[l]++;
    struct event e={start+time+x->latency,x->to,k,hops+1,subset[u],visited|UINT64_C(1)<<from,0,{0},{0},{{0,{0},{0},0}}};
    /* the name carries view lines: the sender's own lanes, then the others in turn */
    for(uint32_t m=0;m<links && e.carried<ENTRIES;m++)if(link[m].from==from){
      e.lane[e.carried]=(uint8_t)m;e.execution[e.carried]=own_execution;e.line[e.carried++]=own[m];
    }
    for(uint32_t tries=0;tries<links && e.carried<ENTRIES;tries++){
      const uint32_t m=cursor[from]++%links;
      if(link[m].from!=from && seen[from][m]){e.lane[e.carried]=(uint8_t)m;e.execution[e.carried]=seen[from][m];e.line[e.carried++]=view[from][m];}
    }
    push(e);
  }
  assert(covered==(to&~(UINT64_C(1)<<from)));   /* each destination in exactly one subset */
}

/* one execution; kind 0 all-gather, 1 all-reduce; S bytes a rank's partial; returns its completion */
static double execute(int kind,double S,double page){
  memset(free_at,0,sizeof free_at);memset(busy,0,sizeof busy);memset(sends,0,sizeof sends);
  const uint32_t P=(uint32_t)ceil(S/page);
  uint64_t all=0;for(uint32_t c=0;c<ncompute;c++)all|=UINT64_C(1)<<compute[c];
  npieces=0;relayed=forwarded=maxhops=0;
  pieces=realloc(pieces,(size_t)(2*ncompute*P+1)*sizeof *pieces);assert(pieces);
  uint32_t *left=calloc(P+1,sizeof *left);
  for(uint32_t c=0;c<ncompute;c++)for(uint32_t p=0;p<P;p++){
    const double b=p+1<P?page:S-page*(P-1);
    const uint32_t src=compute[c],owner=compute[p%ncompute];
    if(kind==0)pieces[npieces++]=(struct piece){src,p,b,all&~(UINT64_C(1)<<src),0,0,{0}};
    else if(src!=owner){pieces[npieces++]=(struct piece){src,p,b,UINT64_C(1)<<owner,0,1,{0}};left[p]++;}
  }
  const uint32_t first=npieces;
  if(kind==1)for(uint32_t p=0;p<P;p++){   /* the reduced pages, sent once their partials have landed */
    const uint32_t owner=compute[p%ncompute];
    pieces[npieces++]=(struct piece){owner,p,p+1<P?page:S-page*(P-1),all&~(UINT64_C(1)<<owner),0,2,{0}};
  }
  for(uint32_t k=0;k<first;k++)send(0,pieces[k].source,k,pieces[k].need,0,0);
  if(kind==1)for(uint32_t p=0;p<P;p++)if(!left[p])send(0,pieces[first+p].source,first+p,pieces[first+p].need,0,0);
  double T=0;size_t steps=0;
  while(events){
    const struct event e=pop();
    assert(++steps<100000000);
    assert(e.hops<=nodes);if(e.hops>maxhops)maxhops=e.hops;
    for(uint32_t c=0;c<e.carried;c++)   /* this node's view: the newest line of each lane */
      if(e.execution[c]>seen[e.node][e.lane[c]]){view[e.node][e.lane[c]]=e.line[c];seen[e.node][e.lane[c]]=e.execution[c];}
    struct piece *x=pieces+e.piece;
    const uint64_t me=UINT64_C(1)<<e.node;
    if(e.to&me){
      assert(x->need&me && !(x->got&me));   /* exactly once */
      x->got|=me;if(e.t>T)T=e.t;
      if(x->phase==1 && !--left[x->page])send(e.t,pieces[first+x->page].source,first+x->page,pieces[first+x->page].need,0,0);
    }else relayed++;
    if(e.to&~me)send(e.t,e.node,e.piece,e.to&~me,e.visited,e.hops);
  }
  for(uint32_t k=0;k<npieces;k++)assert(pieces[k].got==pieces[k].need);
  free(left);
  /* each lane's sender: what its sends met, and its next line */
  for(uint32_t l=0;l<links;l++){
    const struct mesh_switch_line met=mesh_switch_observe(posted[l],waited[l],sends[l],free_at[l],unit);
    mesh_switch_gru_step(gru+l,&weights,&met,unit,own+l);
  }
  own_execution=++execution;
  return T;
}

struct result { double T,mean,sd,bound,relayed_share,busy[256]; uint32_t hops,pieces; };
static struct result run(int kind,double S,double page,int m){
  if(m==4 || m==5){   /* the fluid optimum for this transaction's demands: all-gather S a pair, all-reduce 2 S / n a pair */
    uint32_t src[64],dst[64],n=0;double demand[64];
    for(uint32_t a=0;a<ncompute;a++)for(uint32_t b=0;b<ncompute;b++)if(a!=b){
      src[n]=compute[a];dst[n]=compute[b];demand[n]=kind?2*S/ncompute:S;commodity[compute[a]][compute[b]]=n;n++;
    }
    assert(!mesh_switch_flow(&fabric,src,dst,demand,n,0.05,paths,sizeof paths/sizeof *paths,path_first));
    double load[256]={0};fluid=0;
    /* and the multicast optimum: all-gather a commodity a source to every other rank; all-reduce each partial to its
       owner (a commodity a pair) and each owner's reduced pages to every other rank */
    ncommodities=0;double tdemand[64];
    for(uint32_t a=0;a<ncompute;a++){
      if(!kind){csrc[ncommodities]=compute[a];cto[ncommodities]=all_compute()&~(UINT64_C(1)<<compute[a]);tdemand[ncommodities++]=S;continue;}
      for(uint32_t b=0;b<ncompute;b++)if(a!=b){csrc[ncommodities]=compute[a];cto[ncommodities]=UINT64_C(1)<<compute[b];tdemand[ncommodities++]=S/ncompute;}
      csrc[ncommodities]=compute[a];cto[ncommodities]=all_compute()&~(UINT64_C(1)<<compute[a]);tdemand[ncommodities++]=S/ncompute;
    }
    assert(!mesh_switch_trees(&fabric,csrc,cto,tdemand,ncommodities,0.05,trees,sizeof trees/sizeof *trees,tree_first));
    double tload[256]={0};tree_fluid=0;
    for(uint32_t j=0;j<ncommodities;j++)for(uint32_t q=tree_first[j];q<tree_first[j+1];q++)
      for(uint32_t v=0;v<nodes;v++)if(trees[q].lane[v]!=UINT32_MAX)tload[trees[q].lane[v]]+=tdemand[j]*trees[q].share;
    for(uint32_t l=0;l<links;l++)if(tload[l]/link[l].rate>tree_fluid)tree_fluid=tload[l]/link[l].rate;   /* the optimum's fluid makespan: its most loaded lane's load over the lane's rate */
    for(uint32_t j=0;j<n;j++)for(uint32_t k=path_first[j];k<path_first[j+1];k++)
      for(uint32_t h=0;h<paths[k].hops;h++)load[paths[k].lane[h]]+=demand[j]*paths[k].share;
    for(uint32_t l=0;l<links;l++)if(load[l]/link[l].rate>fluid)fluid=load[l]/link[l].rate;
  }
  mode=m;memset(own,0,sizeof own);memset(view,0,sizeof view);memset(seen,0,sizeof seen);memset(cursor,0,sizeof cursor);
  memset(gru,0,sizeof gru);execution=0;own_execution=0;
  const int runs=m==2 || m==3?RUNS:1;
  double sum=0,sq=0,T=0;int n=0;
  for(int i=0;i<runs;i++){T=execute(kind,S,page);if(i>=runs/2 || runs==1){sum+=T;sq+=T*T;n++;}}
  struct result r={T,sum/n,0,0,forwarded?(double)relayed/forwarded:0,{0},maxhops,npieces};
  r.sd=sqrt(fmax(0,sq/n-r.mean*r.mean));
  for(uint32_t l=0;l<links;l++)r.busy[l]=T>0?busy[l]/T:0;
  if(kind==0){
    uint64_t all=0;for(uint32_t c=0;c<ncompute;c++)all|=UINT64_C(1)<<compute[c];
    for(uint32_t v=0;v<nodes;v++){
      double in=0,out=0;
      for(uint32_t l=0;l<links;l++){if(link[l].to==v)in+=link[l].rate;if(link[l].from==v)out+=link[l].rate;}
      if(!(all>>v&1))continue;
      if(in>0 && S*(ncompute-1)/in>r.bound)r.bound=S*(ncompute-1)/in;
      if(out>0 && S/out>r.bound)r.bound=S/out;
    }
  }
  return r;
}

static uint64_t all_compute(void){uint64_t a=0;for(uint32_t c=0;c<ncompute;c++)a|=UINT64_C(1)<<compute[c];return a;}
int main(int argc,char **argv){
  load(argc>1?argv[1]:"tests/fabric-20261119.txt");
  const double page=argc>2?atof(argv[2]):16384;
  assert(!mesh_switch_fabric_make(&fabric,nodes,link,links,page));
  mesh_switch_gru_weights_make(&weights,20261119);
  static const double rows[]={2,8,32,128,1024,4096};
  printf("%-10s %9s %7s %10s %10s %20s %20s %10s %10s %10s %8s\n","","S bytes","pieces","fixed","own","mean, mean+-sd","sampled, mean+-sd","optimum","fluid","bound","relayed");
  struct result last={0};
  for(int kind=0;kind<2;kind++)for(size_t i=0;i<sizeof rows/sizeof *rows;i++){
    const double S=rows[i]*1536*2;
    const struct result f=run(kind,S,page,0),o=run(kind,S,page,1),d=run(kind,S,page,2),g=run(kind,S,page,3),q=run(kind,S,page,4),t=run(kind,S,page,5);
    printf("%-10s %9.0f  trees: sampled %8.2fus, fluid %8.2fus\n","",S,t.T/1e3,tree_fluid/1e3);
    printf("%-10s %9.0f %7u %8.2fus %8.2fus %11.2f+-%6.2fus %11.2f+-%6.2fus %8.2fus %8.2fus %8.2fus %7.1f%%\n",kind?"all-reduce":"all-gather",S,g.pieces,
           f.T/1e3,o.T/1e3,d.mean/1e3,d.sd/1e3,g.mean/1e3,g.sd/1e3,q.T/1e3,fluid/1e3,g.bound/1e3,100*g.relayed_share);
    if(kind==0)assert(g.T>=g.bound*(1-1e-9) && d.T>=d.bound*(1-1e-9));
    if(kind==0 && i+1==sizeof rows/sizeof *rows)last=g;
  }
  printf("lane utilisation, all-gather at %.0f rows, sampled:",rows[sizeof rows/sizeof *rows-1]);
  for(uint32_t l=0;l<links;l++)printf(" %s>%s %.0f%%",names[link[l].from],names[link[l].to],100*last.busy[l]);
  printf("\nswitch: every page to every destination exactly once, every path simple: passed\n");
  free(heap);free(pieces);
  for(uint32_t l=0;l<links;l++){free(posted[l]);free(waited[l]);}
  return 0;
}
