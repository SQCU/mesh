#ifndef MESH_SWITCHED_H
#define MESH_SWITCHED_H
#include "mesh.h"
#include <errno.h>
#include <math.h>
#include <string.h>

/* design/SPEC-flat-page-dma-bridge.md */
#define MESH_SWITCH_MAGIC 0x5357434Bu
#define MESH_SWITCH_LANES (MESH_PORTS*2u*MESH_LINK_STREAMS)
#define MESH_SWITCH_N 8u
#define MESH_SWITCH_LINKS 64u
#define MESH_SWITCH_FLIGHT 4096u
#define MESH_SWITCH_LOGS (MESH_PORTS+1u)
#define MESH_SWITCH_PARENT MESH_PORTS
#define MESH_SWITCH_CLASSES 3u
#define MESH_CLS_INLINE 0u
#define MESH_CLS_SCAT   1u
#define MESH_CLS_BULK8  2u

_Static_assert(MESH_SWITCH_N<=64 && MESH_SWITCH_N<=MESH_NODES,"a node a bit");
_Static_assert(!(MESH_SWITCH_FLIGHT&(MESH_SWITCH_FLIGHT-1)) && MESH_SWITCH_FLIGHT<=(UINT64_C(1)<<48),"flight size");

struct mesh_switch_link { uint32_t from,to,cables,ends; double rate,latency; };
struct mesh_switch_fabric_link { uint32_t from,to,cables; double rate,latency; };

struct mesh_switch_view {
  uint32_t self,nodes,links,ranks;
  uint32_t node[MESH_SWITCH_N];
  uint32_t slot[MESH_SWITCH_N];
  double scale;
  struct mesh_switch_link link[MESH_SWITCH_LINKS];
  uint8_t reach[1u<<MESH_SWITCH_N][MESH_SWITCH_N];
};

struct mesh_switch_cable { _Alignas(64) _Atomic uint64_t queued; _Atomic float rate,page; };

struct mesh_switch_entry { uint64_t at; uint32_t page,unused; };
_Static_assert(sizeof(struct mesh_switch_entry)==16,"mesh_switch_entry");

struct mesh_switch_lane {
  _Alignas(64) _Atomic uint64_t posted,completed;
  _Atomic uint64_t rx_posted,rx_done;
  _Atomic uint64_t sends,bytes,service,wait,landed,estimate;
  _Atomic uint32_t rx_desc,rx_rem;
  struct mesh_switch_entry flight[MESH_SWITCH_FLIGHT];
  uint32_t rx_ring[MESH_SWITCH_FLIGHT];
};

struct mesh_switch_counters {
  _Alignas(64) _Atomic uint64_t passes,pass_ns[2],part_ns[5];
  _Atomic uint64_t taken,landed,set,forwards,groups;
  _Atomic uint64_t posts,post_wait_ns,post_call_ns,sent,send_ns;
  _Atomic uint64_t starved,refreshes;
  _Atomic uint64_t unreachable,lost;
};

enum child_state { CHILD_STARTED, CHILD_DEVICE_OPEN, CHILD_LANES_MADE, CHILD_SERVING, CHILD_LEAVING };

_Static_assert(2*MESH_LINK_STREAMS<=8,"a port's plan a byte of its line");
struct mesh_switch {
  uint32_t magic,padding;
  _Alignas(64) _Atomic uint64_t port[MESH_PORTS];
  _Alignas(64) _Atomic uint64_t port_q[MESH_PORTS][MESH_SWITCH_CLASSES];
  _Alignas(64) _Atomic uint64_t tx_head[MESH_PORTS][MESH_SWITCH_CLASSES];
  _Alignas(64) _Atomic uint64_t view_version;
  struct mesh_switch_view view[2];
  struct mesh_switch_cable cable[MESH_PORTS];
  struct mesh_switch_lane lane[MESH_SWITCH_LANES];
  struct mesh_switch_counters counters[MESH_SWITCH_LOGS];
};
_Static_assert(sizeof(struct mesh_switch)<=MESH_SWITCH_BYTES,"switch fits in MESH_SWITCH_BYTES");
_Static_assert(sizeof(((struct mesh_switch *)0)->port)==64,"port lines one cache line");

static inline uint32_t mesh_port_state(uint64_t line){ return (uint32_t)(line&0xff); }
static inline uint32_t mesh_port_plan(uint64_t line){ return (uint32_t)(line>>8&0xff); }
static inline uint32_t mesh_port_peer(uint64_t line){ return (uint32_t)(line>>32); }

static inline int mesh_port_move(struct mesh_switch *sw,uint32_t p,uint32_t from,uint32_t state,uint32_t peer,uint32_t plan){
  _Atomic uint64_t *word=&sw->port[p%MESH_PORTS];
  uint64_t was=atomic_load(word);
  for(;;){
    if(!(from>>mesh_port_state(was)&1))return 0;
    const uint64_t next=state==CHILD_SERVING?(uint64_t)state|(uint64_t)(plan&0xff)<<8|(uint64_t)peer<<32:(was&~UINT64_C(0xff))|state;
    if(atomic_compare_exchange_weak(word,&was,next))return 1;
  }
}

static inline uint32_t mesh_switch_port_of(uint32_t l){ return l/(2*MESH_LINK_STREAMS)%MESH_PORTS; }
static inline struct mesh_switch_cable *mesh_switch_cable_of(struct mesh_switch *sw,uint32_t l){ return sw->cable+mesh_switch_port_of(l); }
static inline struct mesh_switch *mesh_switch_of(struct hdr *m){ return (struct mesh_switch *)((char *)m+m->switch_off); }
static inline uint32_t mesh_switch_published(const _Atomic uint64_t *version){ return (uint32_t)(atomic_load_explicit(version,memory_order_acquire)&1); }

/* Three wire QP classes per port over the flat 4096 B page space:
   j = 0,1: 3-frame (12 KiB) 1-WR inline lane for <= 2 pages (e.g. 6 KiB);
   j = 2,3: 1-frame (4 KiB) scattered zero-copy lane;
   j = 4..7: 8-frame (32 KiB) bulk zero-copy contiguous lane. */
static inline uint32_t mesh_switch_lane_frames(uint32_t j){
  const uint32_t k=j%(2u*MESH_LINK_STREAMS);
  return k<2u?3u:(k<4u?1u:8u);
}
static inline uint64_t mesh_switch_lane_bytes(uint32_t l){
  return (uint64_t)mesh_switch_lane_frames(l)*MESH_PAGE_BYTES;
}
static inline uint32_t mesh_switch_class_frames(uint32_t cls){
  return cls==MESH_CLS_INLINE?3u:(cls==MESH_CLS_BULK8?8u:1u);
}
static inline uint32_t mesh_switch_class_mask(uint32_t cls){
  return cls==MESH_CLS_INLINE?0x03u:(cls==MESH_CLS_SCAT?0x0Cu:0xF0u);
}

static inline uint64_t mesh_switch_pack(float a,float b){ uint32_t x,y;memcpy(&x,&a,4);memcpy(&y,&b,4);return (uint64_t)x|(uint64_t)y<<32; }
static inline void mesh_switch_unpack(uint64_t v,float *a,float *b){ const uint32_t x=(uint32_t)v,y=(uint32_t)(v>>32);memcpy(a,&x,4);memcpy(b,&y,4); }

static inline void mesh_switch_average(_Atomic float *term,double sample){
  const float was=atomic_load_explicit(term,memory_order_relaxed);
  atomic_store_explicit(term,(float)(was>0?was+(sample-was)/8:sample),memory_order_relaxed);
}

#define MESH_SWITCH_COUNT(sw,ring,field,v) do{ const uint32_t r_=(ring); if(r_<MESH_SWITCH_LOGS){ _Atomic uint64_t *c_=&(sw)->counters[r_].field; \
  atomic_store_explicit(c_,atomic_load_explicit(c_,memory_order_relaxed)+(uint64_t)(v),memory_order_relaxed);} }while(0)

static inline void mesh_switch_init(struct mesh_switch *sw){
  memset(sw,0,sizeof *sw);
  sw->magic=MESH_SWITCH_MAGIC;
  atomic_thread_fence(memory_order_release);
}

static inline uint32_t mesh_switch_index(const struct mesh_switch_view *v,uint32_t node){
  for(uint32_t i=0;i<v->nodes;i++)if(v->node[i]==node)return i;
  return UINT32_MAX;
}

static inline struct mesh_switch_link *mesh_switch_link_of(struct mesh_switch_view *v,uint32_t from,uint32_t to){
  for(uint32_t l=0;l<v->links;l++)if(v->link[l].from==from && v->link[l].to==to)return v->link+l;
  return NULL;
}

static inline void mesh_switch_view_set(struct mesh_switch *sw,uint32_t self,const uint32_t *nodes,const uint32_t *slots,uint32_t n,
                                        const struct mesh_switch_fabric_link *links,uint32_t count){
  const uint64_t version=atomic_load_explicit(&sw->view_version,memory_order_relaxed);
  struct mesh_switch_view *v=sw->view+((version+1)&1);
  uint32_t sorted[MESH_SWITCH_N],slot[MESH_SWITCH_N],k=0;
  for(uint32_t i=0;i<n && k<MESH_SWITCH_N;i++){
    uint32_t j=k;while(j && sorted[j-1]>nodes[i]){sorted[j]=sorted[j-1];slot[j]=slot[j-1];j--;}
    sorted[j]=nodes[i];slot[j]=slots[i];k++;
  }
  v->self=self;v->nodes=k;v->ranks=0;v->scale=1.0;
  memcpy(v->node,sorted,k*sizeof *sorted);memcpy(v->slot,slot,k*sizeof *slot);
  uint32_t made=0;
  for(uint32_t l=0;l<count && made<MESH_SWITCH_LINKS;l++){
    struct mesh_switch_link *x=v->link+made++;
    x->from=links[l].from;x->to=links[l].to;x->cables=links[l].cables?links[l].cables:1;x->rate=links[l].rate;x->latency=links[l].latency;
    x->ends=(mesh_switch_index(v,x->from)&255)|(mesh_switch_index(v,x->to)&255)<<8;
  }
  v->links=made;
  uint32_t next[MESH_SWITCH_N]={0};
  for(uint32_t l=0;l<made;l++){const uint32_t a=v->link[l].ends&255,b=v->link[l].ends>>8&255;if(a<k && b<k)next[a]|=1u<<b;}
  memset(v->reach,0,sizeof v->reach);
  for(uint32_t avoid=0;avoid<(1u<<k);avoid++)
    for(uint32_t from=0;from<k;from++){
      uint32_t seen=1u<<from,frontier=seen;
      while(frontier){
        uint32_t reached=0;
        for(uint32_t a=0;a<k;a++)if(frontier>>a&1)reached|=next[a];
        reached&=~avoid&~seen;seen|=reached;frontier=reached;
      }
      v->reach[avoid][from]=(uint8_t)seen;
    }
  atomic_store_explicit(&sw->view_version,version+1,memory_order_release);
}

static inline void mesh_switch_sent(struct mesh_switch *sw,uint32_t lane,uint64_t bytes,uint64_t posted,uint64_t start,uint64_t done){
  struct mesh_switch_lane *x=sw->lane+lane;
  atomic_fetch_sub_explicit(&mesh_switch_cable_of(sw,lane)->queued,bytes,memory_order_relaxed);
  atomic_fetch_add_explicit(&x->sends,1,memory_order_relaxed);atomic_fetch_add_explicit(&x->bytes,bytes,memory_order_relaxed);
  const uint64_t service=done>start?done-start:1,wait=start>posted?start-posted:0;
  atomic_fetch_add_explicit(&x->service,service,memory_order_relaxed);atomic_fetch_add_explicit(&x->wait,wait,memory_order_relaxed);
  float rate,waited;mesh_switch_unpack(atomic_load_explicit(&x->estimate,memory_order_relaxed),&rate,&waited);
  const float r=(float)bytes/(float)service;
  rate=rate>0?rate+(r-rate)/8:r;waited+=((float)wait-waited)/8;
  atomic_store_explicit(&x->estimate,mesh_switch_pack(rate,waited),memory_order_relaxed);
}

static inline void mesh_switch_cable_sent(struct mesh_switch *sw,uint32_t p,uint64_t bytes,uint64_t first,uint64_t last,uint64_t done,
                                          int alone,uint64_t posted){
  struct mesh_switch_cable *x=sw->cable+p%MESH_PORTS;
  if(bytes && last && first<last && done>last)mesh_switch_average(&x->rate,(double)bytes/(double)(done-last));
  const float rate=atomic_load_explicit(&x->rate,memory_order_relaxed);
  if(alone && done>posted && rate>0){
    const float was=atomic_load_explicit(&x->page,memory_order_relaxed);
    const double from=was>0?was:(double)MESH_PAGE_BYTES/rate;
    atomic_store_explicit(&x->page,(float)(from+((double)(done-posted)-from)/8),memory_order_relaxed);
  }
}

static __thread uint32_t mesh_switch_home=UINT32_MAX;

static inline void mesh_switch_ports_read(struct mesh_switch *sw,uint64_t *line){
  for(uint32_t p=0;p<MESH_PORTS;p++)line[p]=atomic_load_explicit(&sw->port[p],memory_order_acquire);
}

struct mesh_switch_terms { double backlog,rate,own; };
static inline void mesh_switch_terms_read(struct mesh_switch *sw,const struct mesh_switch_view *v,const uint64_t *line,struct mesh_switch_terms *t){
  double alone[MESH_PORTS];
  for(uint32_t p=0;p<MESH_PORTS;p++){
    if(mesh_port_state(line[p])!=CHILD_SERVING){t[p]=(struct mesh_switch_terms){INFINITY,1.0,INFINITY};alone[p]=INFINITY;continue;}
    const struct mesh_switch_cable *x=sw->cable+p;
    const struct mesh_switch_link *k=mesh_switch_link_of((struct mesh_switch_view *)v,v->self,mesh_port_peer(line[p]));
    const double nominal=k && k->rate>0?k->rate/k->cables:0,latency=k?k->latency:6000.0,measured=atomic_load_explicit(&x->rate,memory_order_relaxed);
    const double rate=measured>0?measured:nominal>0?nominal:1.0;
    const double page=atomic_load_explicit(&x->page,memory_order_relaxed);
    t[p].rate=rate;
    t[p].backlog=(double)atomic_load_explicit(&x->queued,memory_order_relaxed)/rate;
    alone[p]=page>0?page:INFINITY;
    t[p].own=measured>0?(double)MESH_PAGE_BYTES/measured:(double)MESH_PAGE_BYTES/rate+latency;
  }
  for(uint32_t p=0;p<MESH_PORTS;p++){
    if(mesh_port_state(line[p])!=CHILD_SERVING)continue;
    double least=INFINITY;
    for(uint32_t q=0;q<MESH_PORTS;q++)
      if(mesh_port_state(line[q])==CHILD_SERVING && mesh_port_peer(line[q])==mesh_port_peer(line[p]) && alone[q]<least)least=alone[q];
    if(least<INFINITY)t[p].own=least;
  }
}

struct mesh_switch_togo {
  float to[MESH_SWITCH_N][MESH_SWITCH_N],made[MESH_SWITCH_N][MESH_SWITCH_N];
  uint64_t view;
  uint32_t step;
};

static inline int mesh_switch_togo_step(struct mesh_switch *sw,struct mesh_switch_togo *g){
  const uint64_t version=atomic_load_explicit(&sw->view_version,memory_order_acquire);
  const struct mesh_switch_view *v=sw->view+(version&1);
  const uint32_t nodes_read=__atomic_load_n(&v->nodes,__ATOMIC_RELAXED),n=nodes_read<MESH_SWITCH_N?nodes_read:MESH_SWITCH_N;
  const uint32_t self=mesh_switch_index(v,v->self);
  if(g->view!=version+1){g->view=version+1;g->step=0;}
  const uint32_t k=g->step;
  float (*d)[MESH_SWITCH_N]=g->made;
  if(!k){
    for(uint32_t i=0;i<MESH_SWITCH_N;i++)for(uint32_t j=0;j<MESH_SWITCH_N;j++)d[i][j]=i==j?0:INFINITY;
    for(uint32_t l=0;l<v->links && l<MESH_SWITCH_LINKS;l++){
      const struct mesh_switch_link *x=v->link+l;
      const uint32_t a=x->ends&255,b=x->ends>>8&255;
      if(a>=n || b>=n || a==self || b==self)continue;
      const double rate=x->rate>0?x->rate/x->cables:1.0;
      const double cost=(double)MESH_PAGE_BYTES/rate+x->latency;
      if(cost<d[a][b])d[a][b]=(float)cost;
    }
  }else if(k-1!=self){
    const uint32_t via=k-1;
    for(uint32_t i=0;i<n;i++){
      const float first=d[i][via];
      if(!(first<INFINITY))continue;
      for(uint32_t j=0;j<n;j++){const float through=first+d[via][j];if(through<d[i][j])d[i][j]=through;}
    }
  }
  if(++g->step<n+1)return 0;
  for(uint32_t i=0;i<MESH_SWITCH_N;i++)for(uint32_t j=0;j<MESH_SWITCH_N;j++)
    g->to[i][j]=g->made[i][j]<INFINITY?g->made[i][j]:1e12f;
  g->step=0;
  return 1;
}

static inline int mesh_switch_reaches(const struct mesh_switch_view *v,uint32_t from,uint64_t avoid,uint32_t want){
  if(from==want)return 1;
  if(from>=MESH_SWITCH_N || want>=MESH_SWITCH_N)return 0;
  return v->reach[avoid&((1u<<MESH_SWITCH_N)-1)][from]>>want&1;
}

/* Earliest-arrival next-hop port for destination `dst_node`, avoiding {self, origin_node, from_node}. */
static inline uint32_t mesh_switch_route_dest(const struct mesh_switch_view *v,const struct mesh_switch_togo *togo,
                                              const uint64_t *line,const struct mesh_switch_terms *t,
                                              uint32_t dst_node,uint32_t origin_node,uint32_t from_node){
  const uint32_t nodes_read=__atomic_load_n(&v->nodes,__ATOMIC_RELAXED),nodes=nodes_read<MESH_SWITCH_N?nodes_read:MESH_SWITCH_N;
  const uint32_t self=mesh_switch_index(v,v->self),d=mesh_switch_index(v,dst_node);
  if(d>=nodes)return UINT32_MAX;
  uint64_t avoid=self<64?(UINT64_C(1)<<self):0;
  const uint32_t oi=mesh_switch_index(v,origin_node),fi=mesh_switch_index(v,from_node);
  if(oi<nodes && oi!=d)avoid|=UINT64_C(1)<<oi;
  if(fi<nodes && fi!=d)avoid|=UINT64_C(1)<<fi;
  double best=INFINITY;uint32_t best_p=UINT32_MAX;
  for(uint32_t p=0;p<MESH_PORTS;p++){
    if(mesh_port_state(line[p])!=CHILD_SERVING || !mesh_port_plan(line[p]))continue;
    const uint32_t nb=mesh_switch_index(v,mesh_port_peer(line[p]));
    if(nb>=nodes || (avoid>>nb&1))continue;
    if(nb!=d && !mesh_switch_reaches(v,nb,avoid,d))continue;
    const double cost=t[p].backlog+t[p].own+(nb==d?0.0:(double)togo->to[nb][d]);
    const int home=(p==mesh_switch_home),best_home=(best_p==mesh_switch_home);
    if(cost<best || (cost==best && best_p!=UINT32_MAX && (home>best_home || (home==best_home && t[p].backlog<t[best_p].backlog)))){
      best=cost;best_p=p;
    }
  }
  return best_p;
}

/* Count data WRs a DESC page requires on the wire for class `cls`. */
static inline uint32_t mesh_desc_data_wrs(const struct mesh_wire_desc *d,uint32_t cls){
  if(cls==MESH_CLS_INLINE)return 0u;
  const uint32_t n=d->count<MESH_WIRE_BATCH?d->count:MESH_WIRE_BATCH;
  uint32_t wrs=0;
  if(cls==MESH_CLS_BULK8){
    for(uint32_t i=0;i+8u<=n;i+=8u)if(!(d->e[i].flags&MESH_WIRE_SAME_DATA))wrs++;
    return wrs;
  }
  for(uint32_t i=0;i<n;i++)if(!(d->e[i].flags&MESH_WIRE_SAME_DATA))wrs++;
  return wrs;
}

/* Release a DESC page and every unique src_page it still holds (for dead-child / drop cleanup). */
static inline void mesh_desc_drop(struct hdr *m,struct mesh_switch *sw,uint32_t dpage,uint32_t cls,uint32_t target_port,uint32_t ring,int unqueue){
  if(dpage>=mesh_flat_pages(m))return;
  const struct mesh_wire_desc *d=mesh_desc_at(m,dpage);
  const uint32_t n=d->count<MESH_WIRE_BATCH?d->count:MESH_WIRE_BATCH;
  const uint32_t f=mesh_switch_class_frames(cls);
  uint32_t pages_dropped=f;
  if(cls==MESH_CLS_BULK8){
    for(uint32_t i=0;i+8u<=n;i+=8u){
      if(!(d->e[i].flags&MESH_WIRE_SAME_DATA)){
        mesh_page_release_run(m,d->e[i].src_page,8u);
        pages_dropped+=8u;
      }
    }
  }else if(cls==MESH_CLS_SCAT){
    for(uint32_t i=0;i<n;i++){
      if(!(d->e[i].flags&MESH_WIRE_SAME_DATA)){
        mesh_page_release(m,d->e[i].src_page);
        pages_dropped++;
      }
    }
  }
  mesh_page_release_run(m,dpage,f);
  if(unqueue && target_port<MESH_PORTS)
    atomic_fetch_sub_explicit(&sw->cable[target_port].queued,(uint64_t)pages_dropped*MESH_PAGE_BYTES,memory_order_relaxed);
  MESH_SWITCH_COUNT(sw,ring,lost,pages_dropped);
}

static inline void mesh_switch_port_drop(struct hdr *m,struct mesh_switch *sw,uint32_t port,uint32_t ring){
  for(uint32_t cls=0;cls<MESH_SWITCH_CLASSES;cls++){
    uint32_t lists[2]={(uint32_t)atomic_exchange_explicit(&sw->tx_head[port][cls],0,memory_order_seq_cst),
                       mesh_llist_del_all(m,&sw->port_q[port][cls])};
    for(int k=0;k<2;k++){
      uint32_t cur=lists[k];
      while(cur){
        const uint32_t d=cur-1u;
        if(d>=mesh_flat_pages(m))break;
        cur=atomic_load_explicit(&mesh_desc_at(m,d)->next,memory_order_relaxed);
        mesh_desc_drop(m,sw,d,cls,port,ring,1);
      }
    }
  }
}

static inline void mesh_switch_flight_drop(struct hdr *m,struct mesh_switch *sw,uint32_t l,uint32_t ring){
  struct mesh_switch_lane *x=sw->lane+l%MESH_SWITCH_LANES;
  const uint32_t f=mesh_switch_lane_frames(l);
  const uint64_t posted=atomic_load_explicit(&x->posted,memory_order_acquire);
  for(uint64_t at=atomic_load_explicit(&x->completed,memory_order_relaxed);at<posted;at++){
    atomic_fetch_sub_explicit(&mesh_switch_cable_of(sw,l)->queued,(uint64_t)f*MESH_PAGE_BYTES,memory_order_relaxed);
    mesh_page_release_run(m,x->flight[at%MESH_SWITCH_FLIGHT].page,f);
    MESH_SWITCH_COUNT(sw,ring,lost,f);
  }
  atomic_store_explicit(&x->completed,posted,memory_order_relaxed);
}

static inline uint32_t mesh_switch_receives_drop(struct hdr *m,struct mesh_switch *sw,uint32_t first,uint32_t count){
  uint32_t n=0;
  for(uint32_t l=first;l<first+count && l<MESH_SWITCH_LANES;l++){
    struct mesh_switch_lane *x=sw->lane+l;
    const uint32_t f=mesh_switch_lane_frames(l);
    const uint64_t posted=atomic_load_explicit(&x->rx_posted,memory_order_acquire);
    for(uint64_t at=atomic_load_explicit(&x->rx_done,memory_order_relaxed);at<posted;at++){
      mesh_page_release_run(m,x->rx_ring[at%MESH_SWITCH_FLIGHT],f);
      n+=f;
    }
    atomic_store_explicit(&x->rx_done,posted,memory_order_relaxed);
    if(atomic_exchange_explicit(&x->rx_rem,0,memory_order_acq_rel)){
      const uint32_t dp=atomic_exchange_explicit(&x->rx_desc,MESH_PAGE_NONE,memory_order_acq_rel);
      if(dp!=MESH_PAGE_NONE){mesh_page_release(m,dp);n++;}
    }
  }
  return n;
}
#endif
