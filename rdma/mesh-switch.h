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
