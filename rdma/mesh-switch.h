/* Switched multipath DMA: the bridge's per-piece forwarding rule (design/switched-multipath-dma.md).

   A fabric is the live directed links between nodes, each with its measured rate (bytes a ns) and per-hop latency
   (ns). Each lane has a line: its sender's prediction, through the step, of the wait a piece meets there, with that
   prediction's uncertainty. Lines are carried in the names of pieces crossing, in both directions.

   A node holding a piece routes it by Thompson sampling over paths. Every lane's wait is drawn from its line (mean plus
   uncertainty times a draw), the draw a swizzle of the piece's name, the node and the lane, so it is deterministic and
   reproducible but differs from piece to piece. Then one earliest-arrival search per first lane, over nodes the piece
   hasn't visited: the first lane at its exact posted backlog, every later hop at its drawn wait at the time the piece
   would reach it. Each destination takes the first lane of its earliest drawn path. A lane's draw is the same for all
   of the piece's destinations, so destinations whose paths agree share one SEND.

   With uncertain lines, the draws spread pieces over paths in proportion to each path's chance of being best. Nodes
   sharing one estimate therefore make a distribution of traffic, not a jump of all traffic to one path. With certain
   lines, every draw agrees.

   Every hop adds a node to the visited set, so a piece's path is simple: at most as many hops as nodes, and no cycle.
   A lane is taken only on a path that reaches the destination avoiding the visited set, so no piece ends at a dead
   end. Every destination is in exactly one subset at every hop, so each is reached exactly once. These all hold by
   construction, and nothing here checks them at run time. Pure: no I/O, no clock, no global. */
#ifndef MESH_SWITCH_H
#define MESH_SWITCH_H
#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define MESH_SWITCH_NODES 64

struct mesh_switch_link { uint32_t from,to; double rate,latency; };
struct mesh_switch_fabric {
  uint32_t nodes,links;
  const struct mesh_switch_link *link;
  double page;   /* bytes a page: the store-and-forward unit */
};

/* A lane's line, one 64-byte line (less than a line costs a line's read on the weakest unit): its sender's prediction
   of the wait a piece meets there (from posting to its SEND starting) and that prediction's uncertainty, in
   MESH_SWITCH_BINS bins of `bin` units each from the transaction's start. A unit is the caller's `unit` ns. Past the
   last bin, no wait. */
#define MESH_SWITCH_BINS 15
struct mesh_switch_line { uint16_t bin,wait[MESH_SWITCH_BINS],spread[MESH_SWITCH_BINS],reserved; };
_Static_assert(sizeof(struct mesh_switch_line)==64,"a lane's line is one 64-byte line");

/* A piece's name, in its landing slot directly beside the page frame (one contiguous range a request): its
   transaction, execution and page, its bytes, the destinations still to be served through the node it lands on, and
   the nodes it has visited (the one it lands on not yet among them). The view lines the SEND carries follow it. */
struct mesh_switch_name { uint64_t transaction; uint32_t execution,page; uint64_t to,visited; uint32_t bytes,hops; };

/* 0, or EINVAL (too many nodes, a link off the nodes or of no rate) */
static inline int mesh_switch_fabric_make(struct mesh_switch_fabric *f,uint32_t nodes,const struct mesh_switch_link *link,
                                          uint32_t links,double page){
  if(nodes>MESH_SWITCH_NODES || !(page>0))return EINVAL;
  for(uint32_t l=0;l<links;l++){
    const struct mesh_switch_link *x=link+l;
    if(x->from>=nodes || x->to>=nodes || x->from==x->to || !(x->rate>0) || !(x->latency>=0))return EINVAL;
  }
  f->nodes=nodes;f->links=links;f->link=link;f->page=page;
  return 0;
}

/* a line's bin at time t (ns from the transaction's start), or -1 past it */
static inline int mesh_switch_bin(const struct mesh_switch_line *v,double unit,double t){
  if(!v || !v->bin || !(unit>0) || t<0)return -1;
  const uint32_t b=(uint32_t)(t/(v->bin*unit));
  return b<MESH_SWITCH_BINS?(int)b:-1;
}

/* the swizzle: a standard normal draw from a key (splitmix64, then Box-Muller) */
static inline uint64_t mesh_switch_mix(uint64_t x){
  x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);
}
static inline double mesh_switch_draw(uint64_t key){
  const uint64_t a=mesh_switch_mix(key),b=mesh_switch_mix(a);
  const double u=((a>>11)+0.5)/9007199254740992.0,v=((b>>11)+0.5)/9007199254740992.0;
  return sqrt(-2*log(u))*cos(6.283185307179586*v);
}

/* lane l's wait at time t, its line's mean plus its uncertainty times the draw z, at least none */
static inline double mesh_switch_wait(const struct mesh_switch_line *v,double unit,double t,double z){
  const int b=mesh_switch_bin(v,unit,t);
  if(b<0)return 0;
  const double w=(v->wait[b]+v->spread[b]*z)*unit;
  return w>0?w:0;
}

/* a lane's observed waits from one execution's sends on it, in posting order (each one's posting time and wait, ns):
   each bin the mean wait of the sends posted in it, the bins spanning the execution to `last_end` */
static inline struct mesh_switch_line mesh_switch_observe(const double *posted,const double *wait,uint32_t n,double last_end,
                                                          double unit){
  struct mesh_switch_line v={0,{0},{0},0};
  if(!n || !(unit>0) || !(last_end>0))return v;
  double width=ceil(last_end/MESH_SWITCH_BINS/unit);
  if(width<1)width=1;
  if(width>65535)width=65535;
  v.bin=(uint16_t)width;
  double sum[MESH_SWITCH_BINS]={0};uint32_t count[MESH_SWITCH_BINS]={0};
  for(uint32_t i=0;i<n;i++){
    uint32_t b=(uint32_t)(posted[i]/(width*unit));
    if(b>=MESH_SWITCH_BINS)b=MESH_SWITCH_BINS-1;
    sum[b]+=wait[i];count[b]++;
  }
  for(uint32_t b=0;b<MESH_SWITCH_BINS;b++){
    const double w=count[b]?sum[b]/count[b]/unit:0;
    v.wait[b]=(uint16_t)(w<65535?w:65535);
  }
  return v;
}

/* the earliest arrival at every node of a piece of `bytes` leaving node `from` at `t0`, over nodes outside `visited`,
   every hop at its drawn wait (z[l]) in `view` and its send time */
static inline void mesh_switch_search(const struct mesh_switch_fabric *f,uint32_t from,double t0,uint64_t visited,double bytes,
                                      const struct mesh_switch_line *view,double unit,const double *z,double *at){
  for(uint32_t v=0;v<f->nodes;v++)at[v]=INFINITY;
  at[from]=t0;
  uint64_t done=visited&~(UINT64_C(1)<<from);
  for(;;){
    uint32_t u=MESH_SWITCH_NODES;
    for(uint32_t v=0;v<f->nodes;v++)if(!(done>>v&1) && isfinite(at[v]) && (u==MESH_SWITCH_NODES || at[v]<at[u]))u=v;
    if(u==MESH_SWITCH_NODES)return;
    done|=UINT64_C(1)<<u;
    for(uint32_t l=0;l<f->links;l++){
      const struct mesh_switch_link *x=f->link+l;
      if(x->from!=u || done>>x->to&1)continue;
      const double t=at[u]+(view?mesh_switch_wait(view+l,unit,at[u],z[l]):0)+bytes/x->rate+x->latency;
      if(t<at[x->to])at[x->to]=t;
    }
  }
}

/* Node `self`, at `now` (ns from the transaction's start), sends on a piece of `bytes` named `key` (its transaction,
   execution and page, mixed) for the destinations `to`. `visited` holds the nodes it has been at, self included.
   `ahead[l]` is how long outgoing link l stays busy with what is already posted on it (ns from now; only self's links
   are read). `view` is this node's lines (NULL: later hops idle), and `sample` 0 draws every wait at its mean. Writes
   the lanes used and each one's subset, and returns how many. UINT32_MAX if a destination cannot be reached avoiding
   `visited`. */
static inline uint32_t mesh_switch_route(const struct mesh_switch_fabric *f,uint32_t self,double now,uint64_t to,uint64_t visited,
                                         double bytes,uint64_t key,const double *ahead,const struct mesh_switch_line *view,
                                         double unit,int sample,uint32_t *lane,uint64_t *subset){
  double z[256],via[MESH_SWITCH_NODES];
  uint32_t first[MESH_SWITCH_NODES];double best[MESH_SWITCH_NODES];
  for(uint32_t l=0;l<f->links && l<256;l++)z[l]=sample?mesh_switch_draw(mesh_switch_mix(key^((uint64_t)self<<40)^((uint64_t)l<<52))):0;
  visited|=UINT64_C(1)<<self;
  for(uint32_t v=0;v<f->nodes;v++){first[v]=UINT32_MAX;best[v]=INFINITY;}
  for(uint32_t l=0;l<f->links;l++){
    const struct mesh_switch_link *x=f->link+l;
    if(x->from!=self || visited>>x->to&1)continue;
    mesh_switch_search(f,x->to,now+ahead[l]+bytes/x->rate+x->latency,visited,bytes,view,unit,z,via);
    for(uint32_t d=0;d<f->nodes;d++)if(via[d]<best[d]){best[d]=via[d];first[d]=l;}
  }
  uint32_t used=0;
  for(uint32_t d=0;d<f->nodes;d++){
    if(!(to>>d&1) || visited>>d&1)continue;
    if(first[d]==UINT32_MAX)return UINT32_MAX;
    uint32_t k=0;while(k<used && lane[k]!=first[d])k++;
    if(k==used){lane[used]=first[d];subset[used]=0;used++;}
    subset[k]|=UINT64_C(1)<<d;
  }
  return used;
}

/* The multicast fluid optimum, as a distribution over trees. Commodity j moves demand[j] bytes from src[j] to every
   node of the set to[j], and any node on the way may copy (relays included). The same multiplicative weights as
   mesh_switch_flow, its oracle the cheapest arborescence from src[j] reaching every node of to[j] under the lanes'
   lengths. (Garg and Koenemann's max-concurrent-flow: every lane has a length, starting at delta / rate; each
   commodity in turn goes along its cheapest tree, in amounts no larger than the tree's narrowest rate, each lane on it
   lengthened by 1 + eps * flow / rate, until the lengths' rate-weighted sum reaches 1. The flows are within 1 + O(eps)
   of the largest common fraction of every demand the fabric can carry at once.) The oracle here is exact, by
   enumerating every node's parent, which bounds the fabric to
   MESH_SWITCH_TREE_NODES nodes. A larger fabric needs an approximate Steiner oracle in its place. Commodity j's trees
   are written at first[j] .. first[j + 1] - 1, each as its parents (UINT32_MAX for a node not in it, and the source)
   and the lane into each node, with its share of the commodity's flow. 0, ENOSPC or EINVAL. */
#define MESH_SWITCH_TREE_NODES 10
struct mesh_switch_tree { uint32_t parent[MESH_SWITCH_TREE_NODES],lane[MESH_SWITCH_TREE_NODES]; double share; };

static inline int mesh_switch_cheapest_tree(const struct mesh_switch_fabric *f,uint32_t s,uint64_t to,const double *length,
                                            struct mesh_switch_tree *best){
  const uint32_t n=f->nodes;
  uint32_t choice[MESH_SWITCH_TREE_NODES]={0},options[MESH_SWITCH_TREE_NODES][MESH_SWITCH_TREE_NODES+1],count[MESH_SWITCH_TREE_NODES];
  for(uint32_t v=0;v<n;v++){   /* each node's options: out of the tree (UINT32_MAX), or one of its inbound lanes */
    count[v]=0;options[v][count[v]++]=UINT32_MAX;
    if(v==s)continue;
    for(uint32_t l=0;l<f->links;l++)if(f->link[l].to==v && count[v]<=MESH_SWITCH_TREE_NODES)options[v][count[v]++]=l;
  }
  double cheapest=INFINITY;
  for(;;){
    double cost=0;int ok=1;
    for(uint32_t v=0;v<n && ok;v++){
      const uint32_t l=options[v][choice[v]];
      if(to>>v&1 && v!=s && l==UINT32_MAX)ok=0;
      if(l!=UINT32_MAX)cost+=length[l];
    }
    for(uint32_t v=0;v<n && ok;v++){   /* every node in the tree reaches the source */
      uint32_t at=v,steps=0;
      while(at!=s && options[at][choice[at]]!=UINT32_MAX && steps<=n){at=f->link[options[at][choice[at]]].from;steps++;}
      if(options[v][choice[v]]!=UINT32_MAX && at!=s)ok=0;
    }
    if(ok && cost<cheapest){
      cheapest=cost;
      for(uint32_t v=0;v<n;v++){const uint32_t l=options[v][choice[v]];best->lane[v]=l;best->parent[v]=l==UINT32_MAX?UINT32_MAX:f->link[l].from;}
    }
    uint32_t v=0;
    while(v<n && ++choice[v]==count[v]){choice[v]=0;v++;}
    if(v==n)break;
  }
  return isfinite(cheapest)?0:EINVAL;
}

static inline int mesh_switch_trees(const struct mesh_switch_fabric *f,const uint32_t *src,const uint64_t *to,const double *demand,
                                    uint32_t commodities,double eps,struct mesh_switch_tree *tree,uint32_t room,uint32_t *first){
  if(f->nodes>MESH_SWITCH_TREE_NODES || f->links>256 || commodities>=64)return EINVAL;
  double length[256],total[64]={0},want=0,have=0,scaled[64];
  const double m=f->links,delta=(1+eps)*pow((1+eps)*m,-1/eps);
  for(uint32_t l=0;l<f->links;l++){length[l]=delta/f->link[l].rate;have+=f->link[l].rate;}
  for(uint32_t j=0;j<commodities;j++)want+=demand[j];
  if(!(want>0))return EINVAL;
  for(uint32_t j=0;j<commodities;j++)scaled[j]=demand[j]*have/want;
  static struct mesh_switch_tree found[64][64];static double flow[64][64];uint32_t count[64]={0};
  for(uint32_t j=0;j<commodities;j++)for(uint32_t k=0;k<64;k++)flow[j][k]=0;
  for(double D=delta*m;D<1;){
    for(uint32_t j=0;j<commodities && D<1;j++){
      double left=scaled[j];
      while(left>0 && D<1){
        struct mesh_switch_tree t;
        if(mesh_switch_cheapest_tree(f,src[j],to[j],length,&t))return EINVAL;
        double narrow=INFINITY;
        for(uint32_t v=0;v<f->nodes;v++)if(t.lane[v]!=UINT32_MAX && f->link[t.lane[v]].rate<narrow)narrow=f->link[t.lane[v]].rate;
        const double g=left<narrow?left:narrow;
        uint32_t k=0;
        while(k<count[j] && memcmp(found[j][k].lane,t.lane,f->nodes*sizeof *t.lane))k++;
        if(k==count[j]){if(k==64)return ENOSPC;found[j][k]=t;count[j]++;}
        flow[j][k]+=g;total[j]+=g;left-=g;
        for(uint32_t v=0;v<f->nodes;v++)if(t.lane[v]!=UINT32_MAX){
          const uint32_t l=t.lane[v];const double old=length[l];
          length[l]*=1+eps*g/f->link[l].rate;D+=f->link[l].rate*(length[l]-old);
        }
      }
    }
  }
  uint32_t n=0;
  for(uint32_t j=0;j<commodities;j++){
    if(!count[j])return EINVAL;
    first[j]=n;
    for(uint32_t k=0;k<count[j];k++){if(n==room)return ENOSPC;tree[n]=found[j][k];tree[n].share=flow[j][k]/total[j];n++;}
  }
  first[commodities]=n;
  return 0;
}

/* A learned hop-level policy over the whole state a node has when it sends a piece on. For each destination still to
   be served through this node, each candidate first lane l (to a node the piece hasn't visited, on a path to the
   destination) has features:
   0 its arrival under the lines' predicted waits, past the best lane's, over the best lane's time to go;
   1 the same unloaded (latency and send time only);
   2 the same under waits drawn from the lines (the Thompson draw);
   3 this node's posted backlog on l, in the piece's send times on it (as log(1 + that));
   4 the multicast optimum's preference: the log share of the piece's commodity's optimum trees that send the
     destination through l from this node (`prefer`, NULL as none), as 1 + log(share + 1e-3) / log(1e3), so a softmax
     on it alone samples in proportion to the optimum;
   5 whether this piece already takes l for an earlier destination (one SEND carries both);
   6 .. 11 each of 0 .. 5 times the transaction's size, log(1 + bytes / page) / 8.
   The policy's logits are w . phi, a softmax over the candidates, and the choice is the candidate whose interval holds
   a swizzled quantile of the piece's name, the node and the destination. Where `score` is given, the gradient of the
   choice's log-probability (phi of the chosen less the policy's mean phi) is added to it, for a learner. */
#define MESH_SWITCH_FEATURES 12
struct mesh_switch_policy { double w[MESH_SWITCH_FEATURES]; };

static inline uint32_t mesh_switch_route_policy(const struct mesh_switch_fabric *f,uint32_t self,double now,uint64_t to,uint64_t visited,
                                                double bytes,double total,uint64_t key,const double *ahead,
                                                const struct mesh_switch_line *view,double unit,const float *prefer,
                                                const struct mesh_switch_policy *policy,double *score,uint32_t *lane,uint64_t *subset){
  static const double zero[256]={0};
  double z[256],mean[256][MESH_SWITCH_NODES],idle[256][MESH_SWITCH_NODES],drawn[256][MESH_SWITCH_NODES],when[MESH_SWITCH_NODES];
  (void)when;
  for(uint32_t l=0;l<f->links && l<256;l++)z[l]=mesh_switch_draw(mesh_switch_mix(key^((uint64_t)self<<40)^((uint64_t)l<<52)));
  visited|=UINT64_C(1)<<self;
  const double size=log1p(total/f->page)/8;
  for(uint32_t l=0;l<f->links;l++){
    const struct mesh_switch_link *x=f->link+l;
    for(uint32_t v=0;v<f->nodes;v++)mean[l][v]=idle[l][v]=drawn[l][v]=INFINITY;
    if(x->from!=self || visited>>x->to&1)continue;
    const double send=bytes/x->rate,t0=now+ahead[l]+send+x->latency,u0=now+send+x->latency;
    mesh_switch_search(f,x->to,t0,visited,bytes,view,unit,zero,mean[l]);
    mesh_switch_search(f,x->to,u0,visited,bytes,NULL,unit,zero,idle[l]);
    mesh_switch_search(f,x->to,t0,visited,bytes,view,unit,z,drawn[l]);
  }
  uint32_t used=0;
  for(uint32_t d=0;d<f->nodes;d++){
    if(!(to>>d&1) || visited>>d&1)continue;
    double bm=INFINITY,bi=INFINITY,bd=INFINITY;
    for(uint32_t l=0;l<f->links;l++){if(mean[l][d]<bm)bm=mean[l][d];if(idle[l][d]<bi)bi=idle[l][d];if(drawn[l][d]<bd)bd=drawn[l][d];}
    if(!isfinite(bm))return UINT32_MAX;
    uint32_t cand[256],n=0;double phi[256][MESH_SWITCH_FEATURES],logit[256],most=-INFINITY;
    for(uint32_t l=0;l<f->links;l++){
      if(!isfinite(mean[l][d]))continue;
      double *p=phi[n];
      p[0]=(mean[l][d]-bm)/(bm-now+1000);p[1]=(idle[l][d]-bi)/(bi-now+1000);p[2]=(drawn[l][d]-bd)/(bd-now+1000);
      p[3]=log1p(ahead[l]/(bytes/f->link[l].rate));p[4]=prefer?1+log(prefer[(size_t)d*f->links+l]+1e-3)/log(1e3):0;
      p[5]=0;for(uint32_t k=0;k<used;k++)if(lane[k]==l)p[5]=1;
      for(uint32_t i=0;i<6;i++)p[6+i]=p[i]*size;
      logit[n]=0;for(uint32_t i=0;i<MESH_SWITCH_FEATURES;i++)logit[n]+=policy->w[i]*p[i];
      if(logit[n]>most)most=logit[n];
      cand[n++]=l;
    }
    double pi[256],sum=0;
    for(uint32_t c=0;c<n;c++){pi[c]=exp(logit[c]-most);sum+=pi[c];}
    const uint64_t r=mesh_switch_mix(key^((uint64_t)self<<40)^((uint64_t)d<<48)^0x5bd1e995ULL);
    const double u=((r>>11)+0.5)/9007199254740992.0;
    uint32_t pick=n-1;double c=0;
    for(uint32_t k=0;k<n;k++){pi[k]/=sum;c+=pi[k];if(u<c){pick=k;break;}}
    if(score)for(uint32_t i=0;i<MESH_SWITCH_FEATURES;i++){
      double e=0;for(uint32_t k=0;k<n;k++)e+=pi[k]*phi[k][i];
      score[i]+=phi[pick][i]-e;
    }
    const uint32_t l=cand[pick];
    uint32_t k=0;while(k<used && lane[k]!=l)k++;
    if(k==used){lane[used]=l;subset[used]=0;used++;}
    subset[k]|=UINT64_C(1)<<d;
  }
  return used;
}

/* One lane's predictor at its sender: a GRU over each execution's observed waits (as log(1 + wait / 1 us) a bin),
   hidden 64, its weights fixed and seeded; an expansion of its state to 192 rectified features; and a linear readout
   of them, learned online by normalised LMS, that predicts the next execution's waits as a residual on this one's.
   Before it has learned anything, it predicts the last execution again. The uncertainty it states for each bin is its
   last prediction's error there. All of it stays at the sender; only its prediction crosses, as the lane's line. */
#define MESH_GRU_H 64
#define MESH_GRU_X MESH_SWITCH_BINS
#define MESH_GRU_F 192
struct mesh_switch_gru_weights { float z[MESH_GRU_H][MESH_GRU_H+MESH_GRU_X],r[MESH_GRU_H][MESH_GRU_H+MESH_GRU_X],
                                 c[MESH_GRU_H][MESH_GRU_H+MESH_GRU_X],up[MESH_GRU_F][MESH_GRU_H]; };
struct mesh_switch_gru { float h[MESH_GRU_H],x[MESH_GRU_X],f[MESH_GRU_F],out[MESH_GRU_X][MESH_GRU_F];double said[MESH_GRU_X];int seen; };

static inline void mesh_switch_gru_weights_make(struct mesh_switch_gru_weights *w,uint64_t seed){
  float *p=&w->z[0][0];const size_t n=sizeof *w/sizeof *p;
  for(size_t i=0;i<n;i++){
    seed=seed*6364136223846793005ULL+1442695040888963407ULL;
    p[i]=(float)(((seed>>40)&0xffffff)/16777216.0-0.5)*0.5f;
  }
}
static inline float mesh_switch_sigmoid(float v){ return 1/(1+expf(-v)); }
/* one execution's observed waits (the observed line's bins, ns) in, the lane's next line out */
static inline void mesh_switch_gru_step(struct mesh_switch_gru *g,const struct mesh_switch_gru_weights *w,
                                        const struct mesh_switch_line *seen,double unit,struct mesh_switch_line *line){
  float x[MESH_GRU_X],in[MESH_GRU_H+MESH_GRU_X],z[MESH_GRU_H],r[MESH_GRU_H];
  double met[MESH_GRU_X];
  for(uint32_t b=0;b<MESH_GRU_X;b++){met[b]=seen->wait[b]*unit;x[b]=log1pf((float)(met[b]/1000));}
  /* learn: the last prediction's error against what this execution met */
  if(g->seen){
    float norm=1e-3f;for(uint32_t k=0;k<MESH_GRU_F;k++)norm+=g->f[k]*g->f[k];
    for(uint32_t b=0;b<MESH_GRU_X;b++){
      float y=g->x[b];for(uint32_t k=0;k<MESH_GRU_F;k++)y+=g->out[b][k]*g->f[k];
      const float e=x[b]-y;
      for(uint32_t k=0;k<MESH_GRU_F;k++)g->out[b][k]+=0.5f*e*g->f[k]/norm;
    }
  }
  /* the uncertainty: the last prediction's error, a bin (none before a first prediction) */
  double spread[MESH_GRU_X];
  for(uint32_t b=0;b<MESH_GRU_X;b++)spread[b]=g->seen?fabs(met[b]-g->said[b]):0;
  /* the recurrent step */
  for(uint32_t i=0;i<MESH_GRU_H;i++)in[i]=g->h[i];
  for(uint32_t b=0;b<MESH_GRU_X;b++)in[MESH_GRU_H+b]=x[b];
  for(uint32_t i=0;i<MESH_GRU_H;i++){
    float a=0,c=0;for(uint32_t j=0;j<MESH_GRU_H+MESH_GRU_X;j++){a+=w->z[i][j]*in[j];c+=w->r[i][j]*in[j];}
    z[i]=mesh_switch_sigmoid(a);r[i]=mesh_switch_sigmoid(c);
  }
  float h[MESH_GRU_H];
  for(uint32_t i=0;i<MESH_GRU_H;i++){
    float a=0;
    for(uint32_t j=0;j<MESH_GRU_H;j++)a+=w->c[i][j]*r[j]*g->h[j];
    for(uint32_t b=0;b<MESH_GRU_X;b++)a+=w->c[i][MESH_GRU_H+b]*x[b];
    h[i]=(1-z[i])*g->h[i]+z[i]*tanhf(a);
  }
  for(uint32_t i=0;i<MESH_GRU_H;i++)g->h[i]=h[i];
  for(uint32_t k=0;k<MESH_GRU_F;k++){float a=0;for(uint32_t i=0;i<MESH_GRU_H;i++)a+=w->up[k][i]*g->h[i];g->f[k]=a>0?a:0;}
  for(uint32_t b=0;b<MESH_GRU_X;b++)g->x[b]=x[b];
  g->seen=1;
  /* the line: the prediction, and its uncertainty */
  *line=(struct mesh_switch_line){seen->bin,{0},{0},0};
  for(uint32_t b=0;b<MESH_GRU_X;b++){
    float y=x[b];for(uint32_t k=0;k<MESH_GRU_F;k++)y+=g->out[b][k]*g->f[k];
    const double ns=expm1(y>0?y:0)*1000;
    g->said[b]=ns;
    const double wu=ns/unit,su=spread[b]/unit;
    line->wait[b]=(uint16_t)(wu<65535?wu:65535);line->spread[b]=(uint16_t)(su<65535?su:65535);
  }
}
#endif
