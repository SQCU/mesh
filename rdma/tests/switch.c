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
   - descent (with --descent): each piece's tree sampled from a softmax over candidate trees (the optimum's, and each
     commodity's shortest-path tree), its logits moved by online Jacobian descent over two subgoals, the mean and the
     spread of the completion. Each iteration estimates both subgoals' gradients by the score function over 8
     executions and steps along the min-norm point of their convex hull (MGDA), a direction that worsens neither.

   The sampled, trees and descent rows are the mean and spread of the completion over 20 executions after any
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
static struct mesh_switch_link link[256];
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
      assert(links<sizeof link/sizeof *link && n>=1 && n<=CABLES);
      cables[links]=n;link[links++]=(struct mesh_switch_link){node_of(a),node_of(b),rate,us*1e3};
    }
  }
  fclose(f);
  for(uint32_t l=0;l<links;l++){
    reverse[l]=UINT32_MAX;
    for(uint32_t r=0;r<links;r++)if(link[r].from==link[l].to && link[r].to==link[l].from)reverse[l]=r;
  }
}
static uint64_t all_compute(void){uint64_t a=0;for(uint32_t c=0;c<ncompute;c++)a|=UINT64_C(1)<<compute[c];return a;}

struct piece { uint32_t source,page,tree,commodity; double bytes; uint64_t need,got; int phase; };
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
static struct mesh_switch_policy policy;static double score[MESH_SWITCH_FEATURES],current_S;
#define PREFER_COMMODITIES 24
#define PREFER_NODES 8
#define PREFER_LANES 32
static float prefer[PREFER_COMMODITIES][PREFER_NODES][PREFER_NODES*PREFER_LANES];
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

static void send(double t,uint32_t from,uint32_t k,uint64_t to,uint64_t visited,uint32_t hops){
  double ahead[256];uint32_t lane[256];uint64_t subset[256];
  for(uint32_t l=0;l<links;l++){const double f=free_at[l][earliest_cable(l)];ahead[l]=(mode==SAMPLED || mode==POLICY) && f>t?f-t:0;}
  const uint64_t key=mesh_switch_mix(((uint64_t)execution<<32)^k);
  uint32_t used=0;
  if(mode==TREES){   /* the piece's tree, sampled at its source; every hop sends on to its children in the tree */
    if(!hops){
      uint32_t j=0;while(j<ncommodities && !(csrc[j]==from && cto[j]==pieces[k].need))j++;
      assert(j<ncommodities);
      double c=0;const double u=((key>>11)+0.5)/9007199254740992.0;uint32_t pick=tree_first[j+1]-1;
      for(uint32_t q=tree_first[j];q<tree_first[j+1];q++){c+=trees[q].share;if(u<c){pick=q;break;}}
      pieces[k].tree=pick;
    }
    const struct mesh_switch_tree *tr=trees+pieces[k].tree;
    for(uint32_t v=0;v<nodes;v++)if(tr->parent[v]==from){
      uint64_t below=0;   /* the destinations in v's subtree */
      for(uint32_t d=0;d<nodes;d++)if(to>>d&1){uint32_t at=d;while(at!=v && tr->parent[at]!=UINT32_MAX)at=tr->parent[at];if(at==v)below|=UINT64_C(1)<<d;}
      if(below){lane[used]=tr->lane[v];subset[used]=below;used++;}
    }
  }else if(mode==POLICY){
    if(!hops){uint32_t j=0;while(j<ncommodities && !(csrc[j]==from && cto[j]==pieces[k].need))j++;assert(j<ncommodities);pieces[k].commodity=j;}
    used=mesh_switch_route_policy(&fabric,from,t,to,visited,pieces[k].bytes,current_S,key,ahead,view[from],unit,
                                  prefer[pieces[k].commodity][from],&policy,score,lane,subset);
  }else used=mesh_switch_route(&fabric,from,t,to,visited,pieces[k].bytes,key,ahead,mode==SAMPLED?view[from]:NULL,unit,mode==SAMPLED,lane,subset);
  assert(used!=UINT32_MAX && used>0);
  uint64_t covered=0;
  for(uint32_t u=0;u<used;u++){
    const uint32_t l=lane[u];const struct mesh_switch_link *x=link+l;
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
  current_S=S;
  memset(free_at,0,sizeof free_at);memset(busy,0,sizeof busy);memset(sends,0,sizeof sends);
  const uint32_t P=(uint32_t)ceil(S/page);
  const uint64_t all=all_compute();
  npieces=0;
  pieces=realloc(pieces,(size_t)(2*ncompute*P+1)*sizeof *pieces);assert(pieces);
  uint32_t *left=calloc(P+1,sizeof *left);
  for(uint32_t c=0;c<ncompute;c++)for(uint32_t p=0;p<P;p++){
    const double b=p+1<P?page:S-page*(P-1);
    const uint32_t src=compute[c],owner=compute[p%ncompute];
    if(kind==0)pieces[npieces++]=(struct piece){src,p,0,0,b,all&~(UINT64_C(1)<<src),0,0};
    else if(src!=owner){pieces[npieces++]=(struct piece){src,p,0,0,b,UINT64_C(1)<<owner,0,1};left[p]++;}
  }
  const uint32_t first=npieces;
  if(kind==1)for(uint32_t p=0;p<P;p++){   /* the reduced pages, sent once their partials have landed */
    const uint32_t owner=compute[p%ncompute];
    pieces[npieces++]=(struct piece){owner,p,0,0,p+1<P?page:S-page*(P-1),all&~(UINT64_C(1)<<owner),0,2};
  }
  for(uint32_t k=0;k<first;k++)send(0,pieces[k].source,k,pieces[k].need,0,0);
  if(kind==1)for(uint32_t p=0;p<P;p++)if(!left[p])send(0,pieces[first+p].source,first+p,pieces[first+p].need,0,0);
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
      if(x->phase==1 && !--left[x->page])send(e.t,pieces[first+x->page].source,first+x->page,pieces[first+x->page].need,0,0);
    }
    if(e.to&~me)send(e.t,e.node,e.piece,e.to&~me,e.visited,e.hops);
  }
  for(uint32_t k=0;k<npieces;k++)assert(pieces[k].got==pieces[k].need);
  free(left);
  /* each lane's sender: what its sends met, and its next line */
  for(uint32_t l=0;l<links;l++){
    double last=0;for(uint32_t c=0;c<cables[l];c++)if(free_at[l][c]>last)last=free_at[l][c];
    const struct mesh_switch_line met=mesh_switch_observe(posted[l],waited[l],sends[l],last,unit);
    mesh_switch_gru_step(gru+l,&weights,&met,unit,own+l);
    view[link[l].from][l]=own[l];seen[link[l].from][l]=execution+1;
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
  memset(gru,0,sizeof gru);execution=0;own_execution=0;
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
  for(uint32_t l=0;l<links;l++)if(tload[l]/link[l].rate>tree_fluid)tree_fluid=tload[l]/link[l].rate;
  double delay[256];for(uint32_t l=0;l<links;l++)delay[l]=link[l].latency+page/(link[l].rate/cables[l]);
  uint32_t n=0;
  for(uint32_t j=0;j<ncommodities;j++){
    tree_first[j]=n;
    for(uint32_t q=packed_first[j];q<packed_first[j+1];q++)trees[n++]=packed[q];
    /* the shortest-path tree: each destination by its least-delay path from the source, the paths' union pruned to
       the nodes that lead to a destination */
    struct mesh_switch_tree fast;double at[MESH_SWITCH_NODES];uint32_t by[MESH_SWITCH_NODES];uint64_t done=0,keep=0;
    for(uint32_t v=0;v<nodes;v++){at[v]=INFINITY;by[v]=UINT32_MAX;}
    at[csrc[j]]=0;
    for(;;){
      uint32_t u=MESH_SWITCH_NODES;
      for(uint32_t v=0;v<nodes;v++)if(!(done>>v&1) && isfinite(at[v]) && (u==MESH_SWITCH_NODES || at[v]<at[u]))u=v;
      if(u==MESH_SWITCH_NODES)break;
      done|=UINT64_C(1)<<u;
      for(uint32_t l=0;l<links;l++)if(link[l].from==u && at[u]+delay[l]<at[link[l].to]){at[link[l].to]=at[u]+delay[l];by[link[l].to]=l;}
    }
    for(uint32_t dd=0;dd<nodes;dd++)if(cto[j]>>dd&1)for(uint32_t v=dd;v!=csrc[j];v=link[by[v]].from)keep|=UINT64_C(1)<<v;
    for(uint32_t v=0;v<nodes;v++){
      const int in=keep>>v&1;
      fast.lane[v]=in?by[v]:UINT32_MAX;fast.parent[v]=in?link[by[v]].from:UINT32_MAX;
    }
    uint32_t q=tree_first[j];while(q<n && memcmp(trees[q].lane,fast.lane,nodes*sizeof *fast.lane))q++;
    if(q==n){fast.share=0;trees[n++]=fast;}
  }
  tree_first[ncommodities]=n;
  assert(ncommodities<=PREFER_COMMODITIES);
  /* each commodity's optimum preference: the share of its optimum trees that send destination d through lane l from
     node u */
  for(uint32_t j=0;j<ncommodities;j++){
    for(uint32_t u=0;u<nodes;u++)for(size_t i=0;i<(size_t)nodes*links;i++)prefer[j][u][i]=0;
    for(uint32_t q=packed_first[j];q<packed_first[j+1];q++)for(uint32_t d=0;d<nodes;d++)if(cto[j]>>d&1)
      for(uint32_t v=d;packed[q].parent[v]!=UINT32_MAX;v=packed[q].parent[v])
        prefer[j][packed[q].parent[v]][(size_t)d*links+packed[q].lane[v]]+=(float)packed[q].share;
  }
}

/* One (transaction, size)'s learned state: the lines, their predictors and views, and its optimum and preferences.
   Saved and restored by copying, so the policy can be trained over every subgoal at once. */
struct world {
  struct mesh_switch_line own[256],view[MESH_SWITCH_NODES][256];uint32_t own_execution,execution,seen[MESH_SWITCH_NODES][256],cursor[MESH_SWITCH_NODES];
  struct mesh_switch_gru gru[256];struct mesh_switch_tree trees[4096];uint32_t tree_first[65],ncommodities,csrc[64];uint64_t cto[64];
  double tdemand[64],tree_fluid;float prefer[PREFER_COMMODITIES][PREFER_NODES][PREFER_NODES*PREFER_LANES];
};
#define SAVE(w) do{memcpy((w)->own,own,sizeof own);memcpy((w)->view,view,sizeof view);(w)->own_execution=own_execution;(w)->execution=execution;\
  memcpy((w)->seen,seen,sizeof seen);memcpy((w)->cursor,cursor,sizeof cursor);memcpy((w)->gru,gru,sizeof gru);memcpy((w)->trees,trees,sizeof trees);\
  memcpy((w)->tree_first,tree_first,sizeof tree_first);(w)->ncommodities=ncommodities;memcpy((w)->csrc,csrc,sizeof csrc);memcpy((w)->cto,cto,sizeof cto);\
  memcpy((w)->tdemand,tdemand,sizeof tdemand);(w)->tree_fluid=tree_fluid;memcpy((w)->prefer,prefer,sizeof prefer);}while(0)
#define LOAD(w) do{memcpy(own,(w)->own,sizeof own);memcpy(view,(w)->view,sizeof view);own_execution=(w)->own_execution;execution=(w)->execution;\
  memcpy(seen,(w)->seen,sizeof seen);memcpy(cursor,(w)->cursor,sizeof cursor);memcpy(gru,(w)->gru,sizeof gru);memcpy(trees,(w)->trees,sizeof trees);\
  memcpy(tree_first,(w)->tree_first,sizeof tree_first);ncommodities=(w)->ncommodities;memcpy(csrc,(w)->csrc,sizeof csrc);memcpy(cto,(w)->cto,sizeof cto);\
  memcpy(tdemand,(w)->tdemand,sizeof tdemand);tree_fluid=(w)->tree_fluid;memcpy(prefer,(w)->prefer,sizeof prefer);}while(0)

/* the min-norm point of the convex hull of n gradients in R^F (Frank-Wolfe on the simplex) */
static void min_norm(double g[][MESH_SWITCH_FEATURES],int n,double *out){
  double lambda[64];for(int i=0;i<n;i++)lambda[i]=1.0/n;
  for(int it=0;it<200;it++){
    double v[MESH_SWITCH_FEATURES]={0};
    for(int i=0;i<n;i++)for(int f=0;f<MESH_SWITCH_FEATURES;f++)v[f]+=lambda[i]*g[i][f];
    int best=0;double low=INFINITY;
    for(int i=0;i<n;i++){double dot=0;for(int f=0;f<MESH_SWITCH_FEATURES;f++)dot+=g[i][f]*v[f];if(dot<low){low=dot;best=i;}}
    double d[MESH_SWITCH_FEATURES],dd=0,dv=0;
    for(int f=0;f<MESH_SWITCH_FEATURES;f++){d[f]=g[best][f]-v[f];dd+=d[f]*d[f];dv+=d[f]*v[f];}
    if(dd<=0)break;
    double step=-dv/dd;if(step<=0)break;if(step>1)step=1;
    for(int i=0;i<n;i++)lambda[i]*=1-step;lambda[best]+=step;
  }
  for(int f=0;f<MESH_SWITCH_FEATURES;f++){out[f]=0;for(int i=0;i<n;i++)out[f]+=lambda[i]*g[i][f];}
}

/* The policy, trained by online Jacobian descent over every (transaction, size)'s mean and spread at once. Each
   subgoal's gradient in the policy's weights comes from simultaneous perturbation (SPSA): the weights at +c and -c
   along a random sign vector, each side run K executions on the same draws (the same execution numbers). The
   policy has few weights, so this estimate doesn't grow noisier with the number of decisions an execution makes (the
   score function's does). Each subgoal is relative to its value at the start and its gradient scaled to unit length;
   the step is a (Spall's schedule) times the min-norm point of their convex hull, short where the subgoals conflict. */
static void train(const double *rows,int sizes,double page,int iterations,struct world **slot,struct stats *start){
  enum { K=4 };
  const int combos=2*sizes;
  for(int k=0;k<iterations;k++){
    const double a=2.0/pow(k+1,0.602),c=1.0/pow(k+1,0.101);
    double delta[MESH_SWITCH_FEATURES];
    uint64_t r=mesh_switch_mix(0x51a5ULL+(uint64_t)k);
    for(int f=0;f<MESH_SWITCH_FEATURES;f++){r=mesh_switch_mix(r);delta[f]=r&1?1:-1;}
    const struct mesh_switch_policy base=policy;
    double g[64][MESH_SWITCH_FEATURES];int n=0;
    for(int q=0;q<combos;q++){
      const int kind=q/sizes;const double S=rows[q%sizes]*1536*2;
      struct stats side[2];
      for(int sgn=0;sgn<2;sgn++){
        LOAD(slot[q]);mode=POLICY;
        for(int f=0;f<MESH_SWITCH_FEATURES;f++)policy.w[f]=base.w[f]+(sgn?-c:c)*delta[f];
        side[sgn]=measure(kind,S,page,K);
      }
      policy=base;
      LOAD(slot[q]);execution+=K;SAVE(slot[q]);   /* the next iteration's draws are new ones */
      const double m0=start[q].mean,v0=fmax(start[q].sd,start[q].mean*1e-2);
      for(int f=0;f<MESH_SWITCH_FEATURES;f++){
        g[n][f]=(side[0].mean-side[1].mean)/m0/(2*c)*delta[f];
        g[n+1][f]=(side[0].sd-side[1].sd)/v0/(2*c)*delta[f];
      }
      for(int i=n;i<n+2;i++){   /* each subgoal's gradient at unit length, so every subgoal counts alike */
        double l2=0;for(int f=0;f<MESH_SWITCH_FEATURES;f++)l2+=g[i][f]*g[i][f];
        l2=sqrt(l2);if(l2>0)for(int f=0;f<MESH_SWITCH_FEATURES;f++)g[i][f]/=l2;
      }
      n+=2;
    }
    double d[MESH_SWITCH_FEATURES];
    min_norm(g,n,d);
    for(int f=0;f<MESH_SWITCH_FEATURES;f++)policy.w[f]-=a*d[f];
  }
}

int main(int argc,char **argv){
  int descent=0,iterations=150;
  if(argc>1 && !strncmp(argv[1],"--descent",9)){descent=1;if(argv[1][9]=='=')iterations=atoi(argv[1]+10);argc--;argv++;}
  load(argc>1?argv[1]:"tests/fabric-20261119.txt");
  const double page=argc>2?atof(argv[2]):16384;
  assert(!mesh_switch_fabric_make(&fabric,nodes,link,links,page));
  assert(nodes<=PREFER_NODES && links<=PREFER_LANES);
  mesh_switch_gru_weights_make(&weights,20261119);
  static const double rows[]={2,8,32,128,1024,4096};
  enum { SIZES=6 };
  struct stats policy_row[2*SIZES]={{0,0}};
  if(descent){   /* the policy: each (transaction, size) warmed up, then trained on all at once */
    struct world *slot[2*SIZES];struct stats start[2*SIZES];
    memset(&policy,0,sizeof policy);policy.w[2]=-8;   /* it starts as the sampled router: the drawn arrival */
    for(int c=0;c<2*SIZES;c++){
      const int kind=c/SIZES;const double S=rows[c%SIZES]*1536*2;
      slot[c]=calloc(1,sizeof *slot[c]);assert(slot[c]);
      reset(POLICY);optimum(kind,S,page);
      for(int k=0;k<LEARN;k++)execute(kind,S,page);
      start[c]=measure(kind,S,page,8);SAVE(slot[c]);
    }
    train(rows,SIZES,page,iterations,slot,start);
    for(int c=0;c<2*SIZES;c++){LOAD(slot[c]);mode=POLICY;policy_row[c]=measure(c/SIZES,rows[c%SIZES]*1536*2,page,MEASURE);free(slot[c]);}
    printf("policy weights:");for(int f=0;f<MESH_SWITCH_FEATURES;f++)printf(" %.2f",policy.w[f]);printf("\n");
  }
  printf("%-10s %9s %9s %18s %18s %20s %9s %9s\n","","S bytes","fixed","sampled","trees","policy","fluid","floor");
  for(int kind=0;kind<2;kind++)for(size_t i=0;i<sizeof rows/sizeof *rows;i++){
    const double S=rows[i]*1536*2;
    reset(FIXED);const double fixed=execute(kind,S,page);
    reset(SAMPLED);for(int k=0;k<LEARN;k++)execute(kind,S,page);const struct stats sampled=measure(kind,S,page,MEASURE);
    reset(TREES);optimum(kind,S,page);const struct stats tr=measure(kind,S,page,MEASURE);
    const struct stats ds=policy_row[kind*SIZES+i];
    double cut=0;
    if(kind==0)for(uint32_t v=0;v<nodes;v++){
      double in=0,out=0;
      for(uint32_t l=0;l<links;l++){if(link[l].to==v)in+=link[l].rate;if(link[l].from==v)out+=link[l].rate;}
      if(!(all_compute()>>v&1))continue;
      if(in>0 && S*(ncompute-1)/in>cut)cut=S*(ncompute-1)/in;
      if(out>0 && S/out>cut)cut=S/out;
    }
    /* the slowest pair's shortest delay for one piece (no larger than a page) */
    const double q=S<page?S:page;
    double dist[MESH_SWITCH_NODES][MESH_SWITCH_NODES];
    for(uint32_t a=0;a<nodes;a++)for(uint32_t b=0;b<nodes;b++)dist[a][b]=a==b?0:INFINITY;
    for(uint32_t l=0;l<links;l++){const double c=link[l].latency+q/(link[l].rate/cables[l]);if(c<dist[link[l].from][link[l].to])dist[link[l].from][link[l].to]=c;}
    for(uint32_t k=0;k<nodes;k++)for(uint32_t a=0;a<nodes;a++)for(uint32_t b=0;b<nodes;b++)if(dist[a][k]+dist[k][b]<dist[a][b])dist[a][b]=dist[a][k]+dist[k][b];
    double reach=0;
    for(uint32_t a=0;a<ncompute;a++)for(uint32_t b=0;b<ncompute;b++)if(a!=b && dist[compute[a]][compute[b]]>reach)reach=dist[compute[a]][compute[b]];
    const double floor=cut>reach?cut:reach;
    printf("%-10s %9.0f %7.2fus %8.2f+-%6.2fus %8.2f+-%6.2fus ",kind?"all-reduce":"all-gather",S,fixed/1e3,sampled.mean/1e3,sampled.sd/1e3,tr.mean/1e3,tr.sd/1e3);
    if(descent)printf("%10.2f+-%6.2fus ",ds.mean/1e3,ds.sd/1e3);else printf("%20s ","-");
    printf("%7.2fus %7.2fus\n",tree_fluid/1e3,floor/1e3);
    assert(kind || (sampled.mean>=floor*(1-1e-9) && tr.mean>=floor*(1-1e-9)));
  }
  printf("switch: every page to every destination exactly once, every path simple: passed\n");
  free(heap);free(pieces);
  for(uint32_t l=0;l<links;l++){free(posted[l]);free(waited[l]);}
  return 0;
}
