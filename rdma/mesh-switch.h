/* Switched multipath DMA: the bridge's per-piece forwarding rule (design/switched-multipath-dma.md).

   A fabric is the live directed links between nodes, each with its measured rate (bytes a ns) and per-hop latency
   (ns). A node holding a piece sends it on for each of its destinations, on the first lane of the destination's
   earliest-arrival path among nodes the piece hasn't visited. The search is diachronic: its own lanes cost their
   posted backlog, and every later hop costs its expected wait at the time the piece would reach it, learned over
   the transaction's earlier executions and carried in the names of pieces crossing (struct mesh_switch_envelope). Destinations sharing a first lane share one SEND,
   whose name carries their subset and the visited set.

   Every hop adds a node to the visited set, so a piece's path is simple: at most as many hops as nodes, and no cycle.
   A lane is taken only on a path that reaches the destination avoiding the visited set, so no piece ends at a dead
   end. Every destination is in exactly one subset at every hop, so each is reached exactly once. These all hold
   by construction, and nothing here checks them at run time. Pure: no I/O, no clock, no global. */
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
  double page;                                          /* bytes a page: the store-and-forward unit */
  double dist[MESH_SWITCH_NODES][MESH_SWITCH_NODES];   /* shortest one-page delay, ns; INFINITY unreachable */
};
/* a lane's line: its sender's predicted wait through the step (mesh_switch_wait below), one 64-byte line */
#define MESH_SWITCH_BINS 31
struct mesh_switch_envelope { uint16_t bin,wait[MESH_SWITCH_BINS]; };
_Static_assert(sizeof(struct mesh_switch_envelope)==64,"a lane's line is one 64-byte line");
/* A piece's name, in its landing slot directly beside the page frame (one contiguous range a request): its
   transaction, execution and page, its bytes, and the destinations still to be served through the node it lands on, and the nodes it has visited, the one it
   lands on not yet among them, and each lane it has crossed with that lane's envelope. */
struct mesh_switch_name { uint64_t transaction; uint32_t execution,page; uint64_t to,visited; uint32_t bytes,hops;
                          uint8_t lane[8]; struct mesh_switch_envelope envelope[8]; };   /* the lanes crossed, and theirs */

/* the fabric's distances from its links; 0, or EINVAL (too many nodes, a link off the nodes or of no rate) */
static inline int mesh_switch_fabric_make(struct mesh_switch_fabric *f,uint32_t nodes,const struct mesh_switch_link *link,
                                          uint32_t links,double page){
  if(nodes>MESH_SWITCH_NODES || !(page>0))return EINVAL;
  f->nodes=nodes;f->links=links;f->link=link;f->page=page;
  for(uint32_t i=0;i<nodes;i++)for(uint32_t j=0;j<nodes;j++)f->dist[i][j]=i==j?0:INFINITY;
  for(uint32_t l=0;l<links;l++){
    const struct mesh_switch_link *x=link+l;
    if(x->from>=nodes || x->to>=nodes || x->from==x->to || !(x->rate>0) || !(x->latency>=0))return EINVAL;
    const double c=x->latency+page/x->rate;
    if(c<f->dist[x->from][x->to])f->dist[x->from][x->to]=c;
  }
  for(uint32_t k=0;k<nodes;k++)for(uint32_t i=0;i<nodes;i++)for(uint32_t j=0;j<nodes;j++)
    if(f->dist[i][k]+f->dist[k][j]<f->dist[i][j])f->dist[i][j]=f->dist[i][k]+f->dist[k][j];
  return 0;
}

/* A lane's congestion in expectation through one execution of a transaction is the wait a piece meets there: the
   time from posting it to its SEND starting, as the step goes on. A lane's line (struct mesh_switch_envelope, one
   64-byte line: less than a line costs a line's read on the weakest unit) holds its sender's prediction of that wait in
   MESH_SWITCH_BINS bins of `bin` units each, from the transaction's start; a unit is `unit` ns. Past the last bin
   there's no wait. What makes the prediction is the sender's own (struct mesh_switch_gru below): its inputs are the
   waits its own sends met, which it knows exactly. A piece's name carries entries of the sender's view (a lane-id byte, a generation byte and
   the profile each), so every node's view of every lane is refreshed by the traffic already crossing, in both
   directions: no message of its own, and no extra crossing. A residual against the last expectation is the same size,
   as signed deltas. */
static inline double mesh_switch_wait(const struct mesh_switch_envelope *v,double unit,double t){
  if(!v || !v->bin || !(unit>0) || t<0)return 0;
  const uint32_t b=(uint32_t)(t/(v->bin*unit));
  return b<MESH_SWITCH_BINS?v->wait[b]*unit:0;
}

/* a lane's observed waits from one execution's sends on it, in posting order: each one's posting time and wait (ns); each
   bin the mean wait of the sends posted in it */
static inline struct mesh_switch_envelope mesh_switch_envelope_fit(const double *posted,const double *wait,uint32_t n,
                                                                   double last_end,double unit){
  struct mesh_switch_envelope v={0,{0}};
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

/* Node `self`, at `now` (ns from the transaction's start), sends on a piece of `bytes` for the destinations `to`.
   `visited` holds the nodes it has been at, self included. `ahead[l]` is how long outgoing link l stays busy with what
   is already posted on it (ns from now; only self's links are read).

   One time-dependent earliest-arrival search from self, over nodes outside `visited`:
   - leaving self on lane l: the SEND starts once the lane's posted backlog ends, and takes bytes / rate;
   - leaving a later node u at time t on lane l: the piece first meets the lane's expected wait at t (its envelope in
     this node's view), then takes bytes / rate.
   Each hop adds its latency. Every destination goes on the first lane of its earliest path, and destinations sharing a
   first lane share one SEND.

   Writes the lanes used and each one's subset, and returns how many. UINT32_MAX if a destination cannot be reached
   avoiding `visited`. `view[l]` is this node's latest envelope of lane l (`unit` ns its time unit); with `view` NULL,
   later hops count as idle. */
static inline uint32_t mesh_switch_route(const struct mesh_switch_fabric *f,uint32_t self,double now,uint64_t to,uint64_t visited,
                                         double bytes,const double *ahead,const struct mesh_switch_envelope *view,double unit,
                                         uint32_t *lane,uint64_t *subset){
  double at[MESH_SWITCH_NODES];uint32_t first[MESH_SWITCH_NODES];
  visited|=UINT64_C(1)<<self;
  for(uint32_t v=0;v<f->nodes;v++){at[v]=INFINITY;first[v]=UINT32_MAX;}
  at[self]=now;
  uint64_t done=visited&~(UINT64_C(1)<<self);
  for(;;){
    uint32_t u=MESH_SWITCH_NODES;
    for(uint32_t v=0;v<f->nodes;v++)if(!(done>>v&1) && isfinite(at[v]) && (u==MESH_SWITCH_NODES || at[v]<at[u]))u=v;
    if(u==MESH_SWITCH_NODES)break;
    done|=UINT64_C(1)<<u;
    for(uint32_t l=0;l<f->links;l++){
      const struct mesh_switch_link *x=f->link+l;
      if(x->from!=u || done>>x->to&1)continue;
      double end;
      if(u==self)end=now+ahead[l]+bytes/x->rate;
      else end=at[u]+mesh_switch_wait(view?view+l:NULL,unit,at[u])+bytes/x->rate;
      const double t=end+x->latency;
      if(t<at[x->to]){at[x->to]=t;first[x->to]=u==self?l:first[u];}
    }
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
   Before it has learned anything, it predicts the last execution again. All of it stays at the sender; only its
   prediction crosses, as the lane's line. */
#define MESH_GRU_H 64
#define MESH_GRU_X MESH_SWITCH_BINS
#define MESH_GRU_F 192
struct mesh_switch_gru_weights { float z[MESH_GRU_H][MESH_GRU_H+MESH_GRU_X],r[MESH_GRU_H][MESH_GRU_H+MESH_GRU_X],
                                 c[MESH_GRU_H][MESH_GRU_H+MESH_GRU_X],up[MESH_GRU_F][MESH_GRU_H]; };
struct mesh_switch_gru { float h[MESH_GRU_H],x[MESH_GRU_X],f[MESH_GRU_F],out[MESH_GRU_X][MESH_GRU_F]; int seen; };

static inline void mesh_switch_gru_weights_make(struct mesh_switch_gru_weights *w,uint64_t seed){
  float *p=&w->z[0][0];const size_t n=sizeof *w/sizeof *p;
  for(size_t i=0;i<n;i++){
    seed=seed*6364136223846793005ULL+1442695040888963407ULL;
    p[i]=(float)(((seed>>40)&0xffffff)/16777216.0-0.5)*0.5f;
  }
}
static inline float mesh_switch_sigmoid(float v){ return 1/(1+expf(-v)); }
/* one execution's observed waits (ns a bin, the line's bins) in, the predicted next (the lane's line) out */
static inline void mesh_switch_gru_step(struct mesh_switch_gru *g,const struct mesh_switch_gru_weights *w,
                                        const struct mesh_switch_envelope *seen,double unit,struct mesh_switch_envelope *predict){
  float x[MESH_GRU_X],in[MESH_GRU_H+MESH_GRU_X],z[MESH_GRU_H],r[MESH_GRU_H];
  for(uint32_t b=0;b<MESH_GRU_X;b++)x[b]=log1pf((float)(seen->wait[b]*unit/1000));
  /* learn: the last prediction's error against what this execution met */
  if(g->seen){
    float norm=1e-3f;for(uint32_t k=0;k<MESH_GRU_F;k++)norm+=g->f[k]*g->f[k];
    for(uint32_t b=0;b<MESH_GRU_X;b++){
      float y=g->x[b];for(uint32_t k=0;k<MESH_GRU_F;k++)y+=g->out[b][k]*g->f[k];
      const float e=x[b]-y;
      for(uint32_t k=0;k<MESH_GRU_F;k++)g->out[b][k]+=0.5f*e*g->f[k]/norm;
    }
  }
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
  /* predict */
  predict->bin=seen->bin;
  for(uint32_t b=0;b<MESH_GRU_X;b++){
    float y=x[b];for(uint32_t k=0;k<MESH_GRU_F;k++)y+=g->out[b][k]*g->f[k];
    double ns=expm1(y>0?y:0)*1000/unit;
    predict->wait[b]=(uint16_t)(ns<65535?ns:65535);
  }
}
#endif
