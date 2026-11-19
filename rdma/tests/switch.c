/* A discrete-event run of mesh-switch.h's forwarding rule (design/switched-multipath-dma.md) on a fabric file (default
   tests/fabric-20261119.txt, the five-node cluster as measured). Each directed link is a lane, FIFO at its measured rate
   with its measured latency. A node's bridge forwards a piece the moment it lands.

   Two transactions, at each rank-partial size S:
   - all-gather: each compute rank's S bytes, as pages, to every other compute rank;
   - all-reduce: each page's partial from every rank to its owner (page p's owner is compute rank p mod n), and the
     reduced page from the owner to every other rank once its last partial has landed. No barrier between the two.
   A piece's path never revisits a node (its name's visited set).

   Checked at every size: every page reaches every destination that needs it exactly once, the run ends, and no piece
   takes more hops than there are nodes. Reported:
   - the completion time with fixed shortest-delay routes (no backlog read), with each node's own backlog, and
     diachronic: own backlog plus every lane's 8-byte expected-wait profile (32 bytes) carried in the names, 2 entries a name,, learned over 12 executions
     (mean and spread from the third);
   - for the all-gather, a lower bound: the most inbound bytes a node needs over its inbound rate, and the most bytes a
     node must emit over its outbound rate;
   - the share of pieces forwarded by a node that wasn't their destination, and the most hops;
   - per-lane utilisation at the largest size.

   usage: switch [fabric [page]] */
#include "mesh-switch.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

struct piece { uint32_t source,page; double bytes; uint64_t need,got; int phase; };
struct event { double t; uint32_t node,piece,hops; uint64_t to,visited; uint8_t carried,lane[8],generation[8]; struct mesh_switch_envelope envelope[8];
                uint8_t chooser[8],chose[8]; double chosen_at[8]; };   /* the hops that chose its path */
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

struct result { double T,bound,relayed_share,busy[256],mean,sd; uint32_t hops; };
static struct mesh_switch_fabric fabric;
static double free_at[256],busy[256];
static int mode;   /* 0 fixed shortest-delay routes, 1 own backlog only, 2 diachronic: own backlog and lines, 3 and coupled splitting */
static struct mesh_switch_split *split[MESH_SWITCH_NODES];static double slack=0.5;
#define ENTRIES 4   /* view entries a name carries: a 64-byte line each */
static const double unit=64;   /* ns, the envelopes' time unit */
static struct mesh_switch_envelope own[256],view[MESH_SWITCH_NODES][256];
static struct mesh_switch_gru gru[256];static struct mesh_switch_gru_weights weights;static int predictor;
static uint8_t own_generation,seen[MESH_SWITCH_NODES][256];static uint32_t cursor[MESH_SWITCH_NODES];
static double *begin[256],*finish[256];   /* each lane's sends this execution: posting time, wait */static uint32_t spans[256],span_room[256];
static struct piece *pieces;static uint32_t npieces,relayed,forwarded,maxhops;

static void send(double t,uint32_t from,uint32_t k,uint64_t to,uint64_t visited,uint32_t hops,const struct event *in){
  double ahead[256];uint32_t lane[256];uint64_t subset[256];
  for(uint32_t l=0;l<links;l++)ahead[l]=mode && free_at[l]>t?free_at[l]-t:0;
  const uint32_t used=mode==3?mesh_switch_route_split(&fabric,from,t,to,visited,pieces[k].bytes,ahead,view[from],unit,split[from],slack,lane,subset)
                              :mesh_switch_route(&fabric,from,t,to,visited,pieces[k].bytes,ahead,mode==2?view[from]:NULL,unit,lane,subset);
  assert(used!=UINT32_MAX && used>0);
  uint64_t covered=0;
  for(uint32_t u=0;u<used;u++){
    const uint32_t l=lane[u];const struct mesh_switch_link *x=link+l;
    assert(x->from==from && !(covered&subset[u]));covered|=subset[u];
    const double start=free_at[l]>t?free_at[l]:t,time=pieces[k].bytes/x->rate;
    free_at[l]=start+time;busy[l]+=time;forwarded++;
    if(spans[l]==span_room[l]){span_room[l]=span_room[l]?2*span_room[l]:256;begin[l]=realloc(begin[l],span_room[l]*sizeof(double));finish[l]=realloc(finish[l],span_room[l]*sizeof(double));}
    begin[l][spans[l]]=t;finish[l][spans[l]]=start-t;spans[l]++;   /* posted at t, waited start - t */
    assert(!(visited>>x->to&1));   /* a simple path */
    struct event e={start+time+x->latency,x->to,k,hops+1,subset[u],visited|UINT64_C(1)<<from,0,{0},{0},{{0,{0}}},{0},{0},{0}};
    if(in){memcpy(e.chooser,in->chooser,sizeof e.chooser);memcpy(e.chose,in->chose,sizeof e.chose);memcpy(e.chosen_at,in->chosen_at,sizeof e.chosen_at);}
    e.chooser[hops]=(uint8_t)from;e.chose[hops]=(uint8_t)l;e.chosen_at[hops]=t;   /* this hop's choice, for attribution */
    /* the name carries ENTRIES entries of the sender's view, its own lanes and others in turn */
    for(uint32_t m=0;m<links && e.carried<ENTRIES;m++)if(link[m].from==from){e.lane[e.carried]=(uint8_t)m;e.generation[e.carried]=own_generation;e.envelope[e.carried++]=own[m];}
    for(uint32_t tries=0;tries<links && e.carried<ENTRIES;tries++){
      const uint32_t m=cursor[from]++%links;
      if(link[m].from!=from && seen[from][m]){e.lane[e.carried]=(uint8_t)m;e.generation[e.carried]=seen[from][m];e.envelope[e.carried++]=view[from][m];}
    }
    push(e);
  }
  assert(covered==(to&~(UINT64_C(1)<<from)));   /* each destination in exactly one subset */
}

/* one execution; kind 0 all-gather, 1 all-reduce; S bytes a rank's partial; returns its completion */
static double execute(int kind,double S,double page){
  memset(free_at,0,sizeof free_at);memset(busy,0,sizeof busy);memset(spans,0,sizeof spans);
  const uint32_t P=(uint32_t)ceil(S/page);
  uint64_t all=0;for(uint32_t c=0;c<ncompute;c++)all|=UINT64_C(1)<<compute[c];
  npieces=0;relayed=forwarded=maxhops=0;
  pieces=realloc(pieces,(size_t)(2*ncompute*P+1)*sizeof *pieces);assert(pieces);
  uint32_t *left=calloc(P+1,sizeof *left);
  for(uint32_t c=0;c<ncompute;c++)for(uint32_t p=0;p<P;p++){
    const double b=p+1<P?page:S-page*(P-1);
    const uint32_t src=compute[c],owner=compute[p%ncompute];
    if(kind==0)pieces[npieces++]=(struct piece){src,p,b,all&~(UINT64_C(1)<<src),0,0};
    else if(src!=owner){pieces[npieces++]=(struct piece){src,p,b,UINT64_C(1)<<owner,0,1};left[p]++;}
  }
  const uint32_t first=npieces;
  if(kind==1)for(uint32_t p=0;p<P;p++){   /* the reduced pages, sent once their partials have landed */
    const uint32_t owner=compute[p%ncompute];
    pieces[npieces++]=(struct piece){owner,p,p+1<P?page:S-page*(P-1),all&~(UINT64_C(1)<<owner),0,2};
  }
  for(uint32_t k=0;k<first;k++)send(0,pieces[k].source,k,pieces[k].need,0,0,NULL);
  if(kind==1)for(uint32_t p=0;p<P;p++)if(!left[p])send(0,pieces[first+p].source,first+p,pieces[first+p].need,0,0,NULL);
  double T=0;size_t steps=0;
  while(events){
    const struct event e=pop();
    assert(++steps<100000000);
    assert(e.hops<=nodes);if(e.hops>maxhops)maxhops=e.hops;
    for(uint32_t c=0;c<e.carried;c++)   /* this node's view, refreshed where the entry is newer */
      if((uint8_t)(e.generation[c]-seen[e.node][e.lane[c]])<128 && e.generation[c]){view[e.node][e.lane[c]]=e.envelope[c];seen[e.node][e.lane[c]]=e.generation[c];}
    struct piece *x=pieces+e.piece;
    const uint64_t me=UINT64_C(1)<<e.node;
    if(e.to&me){
      assert(x->need&me && !(x->got&me));   /* exactly once */
      x->got|=me;if(e.t>T)T=e.t;
      if(mode==3)for(uint32_t h=0;h<e.hops;h++)mesh_switch_split_attribute(split[e.chooser[h]],e.node,e.chose[h],e.t-e.chosen_at[h]);
      if(x->phase==1 && !--left[x->page])send(e.t,pieces[first+x->page].source,first+x->page,pieces[first+x->page].need,0,0,NULL);
    }else relayed++;
    if(e.to&~me)send(e.t,e.node,e.piece,e.to&~me,e.visited,e.hops,&e);
  }
  for(uint32_t k=0;k<npieces;k++)assert(pieces[k].got==pieces[k].need);
  free(left);
  for(uint32_t l=0;l<links;l++){   /* each sender's own lanes: what they met, and (with the predictor) what they'll meet */
    const struct mesh_switch_envelope met=mesh_switch_envelope_fit(begin[l],finish[l],spans[l],free_at[l],unit);
    if(predictor)mesh_switch_gru_step(gru+l,&weights,&met,unit,own+l);else own[l]=met;
  }
  own_generation++;if(!own_generation)own_generation=1;
  if(mode==3)for(uint32_t v=0;v<nodes;v++)mesh_switch_split_learn(&fabric,v,split[v]);   /* the attributions, carried back by the next execution */
  return T;
}

/* `runs` executions in turn, the envelopes learned across them; the mean and spread from the third on */
static struct result run(int kind,double S,double page,int m,int runs){
  mode=m;for(uint32_t v=0;v<nodes;v++){if(!split[v])split[v]=malloc(sizeof *split[v]);memset(split[v],0,sizeof *split[v]);}
  memset(own,0,sizeof own);memset(gru,0,sizeof gru);memset(view,0,sizeof view);memset(seen,0,sizeof seen);memset(cursor,0,sizeof cursor);own_generation=1;
  double sum=0,sq=0,T=0;int n=0;
  for(int i=0;i<runs;i++){T=execute(kind,S,page);if(i>=runs/2 || runs<3){sum+=T;sq+=T*T;n++;}}
  struct result r={T,0,forwarded?(double)relayed/forwarded:0,{0},sum/n,0,maxhops};
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

int main(int argc,char **argv){
  load(argc>1?argv[1]:"tests/fabric-20261119.txt");
  const double page=argc>2?atof(argv[2]):16384;
  assert(!mesh_switch_fabric_make(&fabric,nodes,link,links,page));
  mesh_switch_gru_weights_make(&weights,20261119);
  static const double rows[]={2,8,32,128,1024,4096};
  printf("%-10s %9s %7s %10s %10s %20s %20s %20s %10s\n","","S bytes","pieces","fixed","own","last met, mean+-sd","GRU, mean+-sd","GRU+split, mean+-sd","bound");
  struct result last={0};
  for(int kind=0;kind<2;kind++)for(size_t i=0;i<sizeof rows/sizeof *rows;i++){
    const double S=rows[i]*1536*2;
    const struct result f=run(kind,S,page,0,1),o=run(kind,S,page,1,1);
    predictor=0;const struct result d=run(kind,S,page,2,40);
    predictor=1;const struct result g=run(kind,S,page,2,40);
    const struct result c=run(kind,S,page,3,40);
    printf("%-10s %9.0f %7u %8.2fus %8.2fus %11.2f+-%6.2fus %11.2f+-%6.2fus %11.2f+-%6.2fus %8.2fus\n",kind?"all-reduce":"all-gather",S,npieces,
           f.T/1e3,o.T/1e3,d.mean/1e3,d.sd/1e3,g.mean/1e3,g.sd/1e3,c.mean/1e3,c.sd/1e3,g.bound/1e3);
    if(kind==0)assert(c.T>=c.bound*(1-1e-9));
    if(kind==0)assert(g.T>=g.bound*(1-1e-9));
    if(kind==0 && i+1==sizeof rows/sizeof *rows)last=c;
  }
  printf("lane utilisation, all-gather at %.0f rows, GRU+split:",rows[sizeof rows/sizeof *rows-1]);
  for(uint32_t l=0;l<links;l++)printf(" %s>%s %.0f%%",names[link[l].from],names[link[l].to],100*last.busy[l]);
  printf("\nswitch: every page to every destination exactly once, no cycle: passed\n");
  free(heap);free(pieces);
  for(uint32_t l=0;l<links;l++){free(begin[l]);free(finish[l]);}
  return 0;
}
