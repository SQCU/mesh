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
   - the completion time, switched (each lane's posted backlog read when routing);
   - the same rule with every backlog read as 0, i.e. fixed shortest-delay routes;
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
struct event { double t; uint32_t node,piece,hops; uint64_t to,visited; };
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

struct result { double T,bound,relayed_share,busy[256]; uint32_t hops; };
static struct mesh_switch_fabric fabric;
static double free_at[256],busy[256];
static int adaptive;
static struct piece *pieces;static uint32_t npieces,relayed,forwarded,maxhops;

static void send(double t,uint32_t from,uint32_t k,uint64_t to,uint64_t visited,uint32_t hops){
  double ahead[256];uint32_t lane[256];uint64_t subset[256];
  for(uint32_t l=0;l<links;l++)ahead[l]=adaptive && free_at[l]>t?free_at[l]-t:0;
  const uint32_t used=mesh_switch_route(&fabric,from,to,visited,pieces[k].bytes,ahead,lane,subset);
  assert(used!=UINT32_MAX && used>0);
  uint64_t covered=0;
  for(uint32_t u=0;u<used;u++){
    const struct mesh_switch_link *x=link+lane[u];
    assert(x->from==from && !(covered&subset[u]));covered|=subset[u];
    const double start=free_at[lane[u]]>t?free_at[lane[u]]:t,time=pieces[k].bytes/x->rate;
    free_at[lane[u]]=start+time;busy[lane[u]]+=time;forwarded++;
    assert(!(visited>>x->to&1));   /* a simple path */
    push((struct event){start+time+x->latency,x->to,k,hops+1,subset[u],visited|UINT64_C(1)<<from});
  }
  assert(covered==(to&~(UINT64_C(1)<<from)));   /* each destination in exactly one subset */
}

/* kind 0 all-gather, 1 all-reduce; S bytes a rank's partial; returns the completion */
static struct result run(int kind,double S,double page,int adapt){
  adaptive=adapt;memset(free_at,0,sizeof free_at);memset(busy,0,sizeof busy);
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
  for(uint32_t k=0;k<first;k++)send(0,pieces[k].source,k,pieces[k].need,0,0);
  if(kind==1)for(uint32_t p=0;p<P;p++)if(!left[p])send(0,pieces[first+p].source,first+p,pieces[first+p].need,0,0);
  double T=0;size_t steps=0;
  while(events){
    const struct event e=pop();
    assert(++steps<100000000);
    assert(e.hops<=nodes);if(e.hops>maxhops)maxhops=e.hops;
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
  struct result r={T,0,forwarded?(double)relayed/forwarded:0,{0},maxhops};
  for(uint32_t l=0;l<links;l++)r.busy[l]=T>0?busy[l]/T:0;
  if(kind==0){
    for(uint32_t v=0;v<nodes;v++){
      double in=0,out=0;
      for(uint32_t l=0;l<links;l++){if(link[l].to==v)in+=link[l].rate;if(link[l].from==v)out+=link[l].rate;}
      const int c=(all>>v&1)!=0;
      if(c && in>0 && S*(ncompute-1)/in>r.bound)r.bound=S*(ncompute-1)/in;
      if(c && out>0 && S/out>r.bound)r.bound=S/out;
    }
  }
  return r;
}

int main(int argc,char **argv){
  load(argc>1?argv[1]:"tests/fabric-20261119.txt");
  const double page=argc>2?atof(argv[2]):16384;
  assert(!mesh_switch_fabric_make(&fabric,nodes,link,links,page));
  static const double rows[]={2,8,32,128,1024,4096};
  printf("%-10s %9s %7s %12s %12s %10s %8s %5s\n","","S bytes","pieces","T switched","T fixed","bound","relayed","hops");
  struct result last={0};
  for(int kind=0;kind<2;kind++)for(size_t i=0;i<sizeof rows/sizeof *rows;i++){
    const double S=rows[i]*1536*2;
    const struct result a=run(kind,S,page,1),b=run(kind,S,page,0);
    printf("%-10s %9.0f %7u %10.2fus %10.2fus %8.2fus %7.1f%% %5u\n",kind?"all-reduce":"all-gather",S,npieces,a.T/1e3,b.T/1e3,
           a.bound/1e3,100*a.relayed_share,a.hops);
    if(kind==0)assert(a.T>=a.bound*(1-1e-9));
    if(kind==0 && i+1==sizeof rows/sizeof *rows)last=a;
  }
  printf("lane utilisation, all-gather at %.0f rows:",rows[sizeof rows/sizeof *rows-1]);
  for(uint32_t l=0;l<links;l++)printf(" %s>%s %.0f%%",names[link[l].from],names[link[l].to],100*last.busy[l]);
  printf("\nswitch: every page to every destination exactly once, no cycle: passed\n");
  free(heap);free(pieces);
  return 0;
}
