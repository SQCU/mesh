#ifndef MESH_SWITCHED_H
#define MESH_SWITCHED_H
#include "mesh.h"
#include "route/split.h"
#include <errno.h>
#include <math.h>
#include <string.h>

/* design/SPEC-flat-page-dma-bridge.md */
#define MESH_SWITCH_MAGIC 0x5357434Du
#define MESH_SWITCH_LANES (MESH_PORTS*2u*MESH_LINK_STREAMS)
#define MESH_SWITCH_N 8u
#define MESH_SWITCH_LINKS 64u
#define MESH_SWITCH_FLIGHT 4096u
#define MESH_SWITCH_LOGS (MESH_PORTS+1u)
#define MESH_SWITCH_PARENT MESH_PORTS
#define MESH_SWITCH_CLASSES 2u
#define MESH_CLS_INLINE 0u
#define MESH_CLS_BULK8  1u

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
struct mesh_switch_bp { _Atomic uint64_t at_ns; _Atomic uint32_t wait_ns; uint32_t pad; };

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
  struct mesh_switch_bp bp[MESH_SWITCH_N][MESH_SWITCH_N];
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

/* Two wire QP classes per stream pair over the flat 4096 B page space:
   even j (0,2,4,6): 2-frame (8 KiB) 1-WR inline lane (raw slot 0 + slot 1 + 272 B tail);
   odd  j (1,3,5,7): 8-frame (32 KiB) bulk zero-copy contiguous lane (8 raw 4 KiB pages/WR). */
static inline uint32_t mesh_switch_lane_frames(uint32_t j){ return (j&1u)?8u:2u; }
static inline uint64_t mesh_switch_lane_bytes(uint32_t l){ return (uint64_t)mesh_switch_lane_frames(l)*MESH_PAGE_BYTES; }
static inline uint32_t mesh_switch_class_frames(uint32_t cls){ return cls==MESH_CLS_BULK8?8u:2u; }
static inline uint32_t mesh_switch_class_mask(uint32_t cls){ return cls==MESH_CLS_BULK8?0xAAu:0x55u; }

static inline _Atomic uint32_t *mesh_class_next(struct hdr *m,uint32_t cls,uint32_t q){
  return cls==MESH_CLS_INLINE?&mesh_tail_at(m,q)->next:&mesh_desc_at(m,q)->next;
}
static inline void mesh_llist_add_cls(struct hdr *m,_Atomic uint64_t *head,uint32_t cls,uint32_t q){
  _Atomic uint32_t *nxt=mesh_class_next(m,cls,q);
  uint64_t first=atomic_load_explicit(head,memory_order_relaxed);
  do {
    atomic_store_explicit(nxt,(uint32_t)first,memory_order_relaxed);
  } while(!atomic_compare_exchange_weak_explicit(head,&first,(uint64_t)(q+1u),memory_order_seq_cst,memory_order_relaxed));
}
static inline uint32_t mesh_llist_reverse_cls(struct hdr *m,uint32_t cls,uint32_t head){
  uint32_t rev=0;
  const uint32_t pages=mesh_flat_pages(m);
  while(head){
    const uint32_t q=head-1u;
    if(q>=pages)break;
    _Atomic uint32_t *nxt=mesh_class_next(m,cls,q);
    const uint32_t next=atomic_load_explicit(nxt,memory_order_relaxed);
    atomic_store_explicit(nxt,rev,memory_order_relaxed);
    rev=head;
    head=next;
  }
  return rev;
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
  uint8_t hops[MESH_SWITCH_N][MESH_SWITCH_N];
  uint64_t view,updated_at;
  uint32_t step;
};

/* Exact drain-physics estimate of async backpressure wait_ns(a -> b) at wall clock `now_ns`. */
static inline double mesh_switch_bp_wait(const struct mesh_switch *sw,uint32_t a,uint32_t b,uint64_t now_ns){
  if(a>=MESH_SWITCH_N || b>=MESH_SWITCH_N)return 0.0;
  const uint64_t at=atomic_load_explicit(&sw->bp[a][b].at_ns,memory_order_acquire);
  const uint32_t w=atomic_load_explicit(&sw->bp[a][b].wait_ns,memory_order_relaxed);
  if(!at || !w)return 0.0;
  if(now_ns<=at)return (double)w;
  const uint64_t dt=now_ns-at;
  return dt<(uint64_t)w?(double)(w-(uint32_t)dt):0.0;
}

/* Compute self's per-neighbor aggregate egress queueing time (sum queued / sum rate across parallel cables)
   and pack self's edges + fresh gossiped non-self edges into an async `mesh_bp_frame`. */
static inline uint32_t mesh_switch_bp_pack(struct mesh_switch *sw,const struct mesh_switch_view *v,const uint64_t *line,
                                           uint64_t now_ns,struct mesh_bp_frame *out){
  const uint32_t nodes_read=__atomic_load_n(&v->nodes,__ATOMIC_RELAXED),n=nodes_read<MESH_SWITCH_N?nodes_read:MESH_SWITCH_N;
  const uint32_t self=mesh_switch_index(v,v->self);
  uint64_t qsum[MESH_SWITCH_N]={0};
  double rsum[MESH_SWITCH_N]={0};
  uint32_t has[MESH_SWITCH_N]={0};
  for(uint32_t p=0;p<MESH_PORTS;p++){
    if(mesh_port_state(line[p])!=CHILD_SERVING || !mesh_port_plan(line[p]))continue;
    const uint32_t peer=mesh_port_peer(line[p]);
    const uint32_t nb=mesh_switch_index(v,peer);
    if(nb>=n)continue;
    const struct mesh_switch_cable *x=sw->cable+p;
    const struct mesh_switch_link *k=mesh_switch_link_of((struct mesh_switch_view *)v,v->self,peer);
    const double nominal=k && k->rate>0?k->rate/k->cables:0.0,measured=atomic_load_explicit(&x->rate,memory_order_relaxed);
    const double rate=measured>0?measured:nominal>0?nominal:8.5;
    qsum[nb]+=atomic_load_explicit(&x->queued,memory_order_relaxed);
    rsum[nb]+=rate;
    has[nb]=1;
  }
  uint32_t cnt=0,max_w=0;
  for(uint32_t nb=0;nb<n;nb++){
    if(!has[nb] || rsum[nb]<=0.0)continue;
    const double wd=(double)qsum[nb]/rsum[nb];
    const uint32_t w=wd>1e9?1000000000u:(uint32_t)wd;
    if(w>max_w)max_w=w;
    if(self<MESH_SWITCH_N){
      atomic_store_explicit(&sw->bp[self][nb].wait_ns,w,memory_order_relaxed);
      atomic_store_explicit(&sw->bp[self][nb].at_ns,now_ns,memory_order_release);
    }
    if(out && cnt<MESH_BP_MAX)out->e[cnt++]=(struct mesh_bp_edge){v->self,v->node[nb],w};
  }
  if(out){
    for(uint32_t a=0;a<n && cnt<MESH_BP_MAX;a++){
      if(a==self)continue;
      for(uint32_t b=0;b<n && cnt<MESH_BP_MAX;b++){
        if(a==b)continue;
        const double ew=mesh_switch_bp_wait(sw,a,b,now_ns);
        if(ew>=1000.0)out->e[cnt++]=(struct mesh_bp_edge){v->node[a],v->node[b],(uint32_t)ew};
      }
    }
    out->count=cnt;
    out->magic=MESH_BP_MAGIC;
  }
  return max_w;
}

static inline void mesh_switch_bp_ingest(struct mesh_switch *sw,const struct mesh_switch_view *v,uint64_t now_ns,const struct mesh_bp_frame *in){
  if(!in || in->magic!=MESH_BP_MAGIC)return;
  const uint32_t cnt=in->count<MESH_BP_MAX?in->count:MESH_BP_MAX;
  const uint32_t nodes_read=__atomic_load_n(&v->nodes,__ATOMIC_RELAXED),n=nodes_read<MESH_SWITCH_N?nodes_read:MESH_SWITCH_N;
  for(uint32_t i=0;i<cnt;i++){
    if(in->e[i].from==v->self)continue;
    const uint32_t a=mesh_switch_index(v,in->e[i].from),b=mesh_switch_index(v,in->e[i].to);
    if(a>=n || b>=n || a==b)continue;
    atomic_store_explicit(&sw->bp[a][b].wait_ns,in->e[i].wait_ns,memory_order_relaxed);
    atomic_store_explicit(&sw->bp[a][b].at_ns,now_ns,memory_order_release);
  }
}

/* Fast all-pairs dynamic cost-to-go with live draining async backpressure + aggregate parallel-cable rate.
   N <= 6 => <= 216 float ops (~30 ns), refreshed every 4 us on active paths. */
static inline void mesh_switch_togo_refresh(struct mesh_switch *sw,const struct mesh_switch_view *v,uint64_t now_ns,struct mesh_switch_togo *g){
  const uint64_t version=atomic_load_explicit(&sw->view_version,memory_order_acquire);
  if(g->view==version+1 && now_ns>g->updated_at && now_ns-g->updated_at<4000u)return;
  const uint32_t nodes_read=__atomic_load_n(&v->nodes,__ATOMIC_RELAXED),n=nodes_read<MESH_SWITCH_N?nodes_read:MESH_SWITCH_N;
  const uint32_t self=mesh_switch_index(v,v->self);
  float d[MESH_SWITCH_N][MESH_SWITCH_N];
  uint8_t h[MESH_SWITCH_N][MESH_SWITCH_N];
  for(uint32_t i=0;i<MESH_SWITCH_N;i++)for(uint32_t j=0;j<MESH_SWITCH_N;j++){
    d[i][j]=i==j?0.0f:INFINITY;
    h[i][j]=i==j?0u:255u;
  }
  for(uint32_t l=0;l<v->links && l<MESH_SWITCH_LINKS;l++){
    const struct mesh_switch_link *x=v->link+l;
    const uint32_t a=x->ends&255,b=x->ends>>8&255;
    if(a>=n || b>=n || a==self || b==self)continue;
    const double rate=x->rate>0?x->rate:8.5;
    const double wait=mesh_switch_bp_wait(sw,a,b,now_ns);
    const double cost=wait+(double)(8u*MESH_PAGE_BYTES)/rate+x->latency;
    if(cost<d[a][b]){d[a][b]=(float)cost;h[a][b]=1u;}
  }
  for(uint32_t via=0;via<n;via++){
    if(via==self)continue;
    for(uint32_t i=0;i<n;i++){
      const float first=d[i][via];
      if(!(first<INFINITY))continue;
      for(uint32_t j=0;j<n;j++){
        const float through=first+d[via][j];
        if(through<d[i][j])d[i][j]=through;
        if(h[i][via]<255u && h[via][j]<255u && (uint32_t)h[i][via]+h[via][j]<h[i][j])
          h[i][j]=(uint8_t)(h[i][via]+h[via][j]);
      }
    }
  }
  for(uint32_t i=0;i<MESH_SWITCH_N;i++)for(uint32_t j=0;j<MESH_SWITCH_N;j++){
    g->to[i][j]=d[i][j]<INFINITY?d[i][j]:1e12f;
    g->hops[i][j]=h[i][j];
  }
  g->view=version+1;
  g->updated_at=now_ns;
}

static inline int mesh_switch_togo_step(struct mesh_switch *sw,struct mesh_switch_togo *g){
  const uint64_t version=atomic_load_explicit(&sw->view_version,memory_order_acquire);
  mesh_switch_togo_refresh(sw,sw->view+(version&1),g->updated_at+4000u,g);
  return 1;
}

static inline int mesh_switch_reaches(const struct mesh_switch_view *v,uint32_t from,uint64_t avoid,uint32_t want){
  if(from==want)return 1;
  if(from>=MESH_SWITCH_N || want>=MESH_SWITCH_N)return 0;
  return v->reach[avoid&((1u<<MESH_SWITCH_N)-1)][from]>>want&1;
}

/* Earliest-arrival next-hop port for destination `dst_node`, avoiding {self, origin_node, from_node}.
   On relay (origin != self), direct delivery to dst_node always wins over multi-hop detours. */
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
  const int is_relay=(origin_node!=v->self);
  double best_pref=INFINITY,best_any=INFINITY;
  uint32_t best_pref_p=UINT32_MAX,best_any_p=UINT32_MAX;
  for(uint32_t p=0;p<MESH_PORTS;p++){
    if(mesh_port_state(line[p])!=CHILD_SERVING || !mesh_port_plan(line[p]))continue;
    const uint32_t peer=mesh_port_peer(line[p]);
    const uint32_t nb=mesh_switch_index(v,peer);
    if(nb>=nodes || (avoid>>nb&1))continue;
    if(nb!=d && !mesh_switch_reaches(v,nb,avoid,d))continue;
    const double cost=t[p].backlog+t[p].own+(nb==d?0.0:(double)togo->to[nb][d]);
    const int pref=is_relay?(nb==d):1;
    const int home=(p==mesh_switch_home);
    if(pref){
      const int bh=(best_pref_p==mesh_switch_home);
      if(cost<best_pref || (cost==best_pref && best_pref_p!=UINT32_MAX && (home>bh || (home==bh && t[p].backlog<t[best_pref_p].backlog)))){
        best_pref=cost;best_pref_p=p;
      }
    }
    const int bh=(best_any_p==mesh_switch_home);
    if(cost<best_any || (cost==best_any && best_any_p!=UINT32_MAX && (home>bh || (home==bh && t[p].backlog<t[best_any_p].backlog)))){
      best_any=cost;best_any_p=p;
    }
  }
  return best_pref_p<MESH_PORTS?best_pref_p:best_any_p;
}

/* Closed-form Garg-Konemann dual-priced exact directed Steiner arborescence oracle (Dreyfus-Wagner for |S| <= 3).
   Uses live local cable backlog + async draining downstream backpressure (sw->bp) to jointly maximize fabric
   utilization, conserve cut capacity (1 cut crossing per multicast), and minimize latency on small inline packets. */
static inline void mesh_switch_route_split(const struct mesh_switch *sw,const struct mesh_switch_view *v,uint64_t now_ns,
                                           const uint64_t *line,struct mesh_switch_terms *t,
                                           const uint32_t *dsts,uint32_t nd,uint32_t origin,uint32_t from,
                                           uint64_t unit_bytes,uint32_t *out_ports){
  for(uint32_t i=0;i<nd;i++)out_ports[i]=UINT32_MAX;
  if(!nd)return;
  const uint32_t nodes_read=__atomic_load_n(&v->nodes,__ATOMIC_RELAXED),n=nodes_read<MESH_SWITCH_N?nodes_read:MESH_SWITCH_N;
  const uint32_t self=mesh_switch_index(v,v->self);
  if(self>=n)return;
  const uint32_t oi=mesh_switch_index(v,origin),fi=mesh_switch_index(v,from);
  const int is_relay=(origin!=v->self);
  const int is_small=(unit_bytes<=2u*MESH_PAGE_BYTES);
  uint32_t avoid_mid=(1u<<self);
  if(oi<n)avoid_mid|=1u<<oi;
  if(fi<n)avoid_mid|=1u<<fi;

  /* Pick min-(backlog+own) serving cable per distinct neighbor and sum aggregate neighbor rate. */
  uint32_t nb_port[MESH_SWITCH_N];
  double nb_best[MESH_SWITCH_N],nb_rsum[MESH_SWITCH_N];
  for(uint32_t i=0;i<MESH_SWITCH_N;i++){nb_port[i]=UINT32_MAX;nb_best[i]=INFINITY;nb_rsum[i]=0.0;}
  for(uint32_t p=0;p<MESH_PORTS;p++){
    if(mesh_port_state(line[p])!=CHILD_SERVING || !mesh_port_plan(line[p]))continue;
    const uint32_t ni=mesh_switch_index(v,mesh_port_peer(line[p]));
    if(ni>=n)continue;
    nb_rsum[ni]+=t[p].rate;
    const double c=t[p].backlog+t[p].own;
    const int hp=(p==mesh_switch_home),hb=(nb_port[ni]==mesh_switch_home);
    if(c<nb_best[ni] || (c==nb_best[ni] && nb_port[ni]!=UINT32_MAX && (hp>hb || (hp==hb && t[p].backlog<t[nb_port[ni]].backlog)))){
      nb_best[ni]=c;nb_port[ni]=p;
    }
  }

  /* Build all-pairs shortest dual-price paths avoiding {self, origin, from} as intermediate relays. */
  float sp[MESH_SWITCH_N][MESH_SWITCH_N];
  uint8_t nh[MESH_SWITCH_N][MESH_SWITCH_N];
  for(uint32_t i=0;i<n;i++)for(uint32_t j=0;j<n;j++){
    sp[i][j]=i==j?0.0f:1e12f;
    nh[i][j]=i==j?(uint8_t)i:255u;
  }
  for(uint32_t nb=0;nb<n;nb++){
    const uint32_t p=nb_port[nb];
    if(p>=MESH_PORTS)continue;
    const double R=nb_rsum[nb]>0.0?nb_rsum[nb]:t[p].rate;
    const double W=t[p].backlog,L=t[p].own;
    const double tx=(double)unit_bytes/R;
    const double cost=is_small?(W+L+tx):(tx*(1.0+W/12000.0)+0.15*L);
    sp[self][nb]=(float)cost;
    nh[self][nb]=(uint8_t)nb;
  }
  for(uint32_t l=0;l<v->links && l<MESH_SWITCH_LINKS;l++){
    const struct mesh_switch_link *x=v->link+l;
    const uint32_t a=x->ends&255,b=x->ends>>8&255;
    if(a>=n || b>=n || a==self || b==self || (avoid_mid>>a&1))continue;
    const double R=x->rate>0?x->rate:8.5;
    const double W=mesh_switch_bp_wait(sw,a,b,now_ns);
    const double L=x->latency>0?x->latency:1500.0;
    const double tx=(double)unit_bytes/R;
    const double cost=is_small?(W+L+tx):(tx*(1.0+W/12000.0)+0.15*L);
    if((float)cost<sp[a][b]){sp[a][b]=(float)cost;nh[a][b]=(uint8_t)b;}
  }
  for(uint32_t via=0;via<n;via++){
    if(avoid_mid>>via&1)continue;
    for(uint32_t i=0;i<n;i++){
      const float first=sp[i][via];
      if(first>=1e11f)continue;
      for(uint32_t j=0;j<n;j++){
        const float through=first+sp[via][j];
        if(through<sp[i][j]){sp[i][j]=through;nh[i][j]=nh[i][via];}
      }
    }
  }

  /* Map valid destinations; on inline small packets or direct single-dest relays, take direct/min-latency path. */
  uint32_t dt[MESH_SWITCH_N],di[MESH_SWITCH_N],m=0;
  for(uint32_t i=0;i<nd;i++){
    const uint32_t d=mesh_switch_index(v,dsts[i]);
    if(d>=n || d==self)continue;
    if(is_relay && nb_port[d]<MESH_PORTS && nd==1u){
      out_ports[i]=nb_port[d];
      t[nb_port[d]].backlog+=(double)unit_bytes/t[nb_port[d]].rate;
      return;
    }
    if(is_small || nd>3u){
      const uint32_t hop=nh[self][d];
      if(hop<n && nb_port[hop]<MESH_PORTS)out_ports[i]=nb_port[hop];
      continue;
    }
    dt[m]=d;di[m]=i;m++;
  }
  if(is_small || nd>3u){
    uint32_t used=0;
    for(uint32_t i=0;i<nd;i++){
      const uint32_t tp=out_ports[i];
      if(tp<MESH_PORTS && !(used&(1u<<tp))){used|=1u<<tp;t[tp].backlog+=(double)unit_bytes/t[tp].rate;}
    }
    return;
  }

  /* Exact Dreyfus-Wagner minimum-price directed Steiner tree for m = 1, 2, 3 destinations. */
  uint32_t chosen_nb[3]={255u,255u,255u};
  if(m==1u){
    chosen_nb[0]=nh[self][dt[0]];
  }else if(m==2u){
    float best=1e12f;uint32_t best_u=self;
    for(uint32_t u=0;u<n;u++){
      if(u!=self && (avoid_mid>>u&1))continue;
      const float c=sp[self][u]+sp[u][dt[0]]+sp[u][dt[1]];
      if(c<best){best=c;best_u=u;}
    }
    if(best_u!=self){chosen_nb[0]=chosen_nb[1]=nh[self][best_u];}
    else{chosen_nb[0]=nh[self][dt[0]];chosen_nb[1]=nh[self][dt[1]];}
  }else if(m==3u){
    const uint32_t pa[3]={1u,0u,0u},pb[3]={2u,2u,1u};
    float t2[MESH_SWITCH_N][3];
    uint8_t t2x[MESH_SWITCH_N][3];
    for(uint32_t u=0;u<n;u++){
      for(uint32_t p=0;p<3;p++){t2[u][p]=1e12f;t2x[u][p]=(uint8_t)u;}
      if(u!=self && (avoid_mid>>u&1))continue;
      for(uint32_t p=0;p<3;p++){
        const uint32_t da=dt[pa[p]],db=dt[pb[p]];
        for(uint32_t x=0;x<n;x++){
          if(x!=self && (avoid_mid>>x&1))continue;
          const float c=sp[u][x]+sp[x][da]+sp[x][db];
          if(c<t2[u][p]){t2[u][p]=c;t2x[u][p]=(uint8_t)x;}
        }
      }
    }
    float best=1e12f;uint32_t best_m=self,best_p=0;
    for(uint32_t u=0;u<n;u++){
      if(u!=self && (avoid_mid>>u&1))continue;
      for(uint32_t p=0;p<3;p++){
        const float c=sp[self][u]+sp[u][dt[p]]+t2[u][p];
        if(c<best){best=c;best_m=u;best_p=p;}
      }
    }
    if(best_m!=self){
      const uint32_t h=nh[self][best_m];
      chosen_nb[0]=chosen_nb[1]=chosen_nb[2]=h;
    }else{
      chosen_nb[best_p]=nh[self][dt[best_p]];
      const uint32_t a=pa[best_p],b=pb[best_p],x=t2x[self][best_p];
      if(x!=self){chosen_nb[a]=chosen_nb[b]=nh[self][x];}
      else{chosen_nb[a]=nh[self][dt[a]];chosen_nb[b]=nh[self][dt[b]];}
    }
  }
  uint32_t used=0;
  for(uint32_t r=0;r<m;r++){
    const uint32_t hop=chosen_nb[r];
    if(hop<n && nb_port[hop]<MESH_PORTS){
      const uint32_t tp=nb_port[hop];
      out_ports[di[r]]=tp;
      if(!(used&(1u<<tp))){used|=1u<<tp;t[tp].backlog+=(double)unit_bytes/t[tp].rate;}
    }
  }
}

/* Count data WRs a DESC page requires on the wire for class `cls`. */
static inline uint32_t mesh_desc_data_wrs(const struct mesh_wire_desc *d,uint32_t cls){
  if(cls==MESH_CLS_INLINE)return 0u;
  const uint32_t n=d->count<MESH_WIRE_BATCH?d->count:MESH_WIRE_BATCH;
  uint32_t wrs=0;
  for(uint32_t i=0;i+8u<=n;i+=8u)if(!(d->e[i].flags&MESH_WIRE_SAME_DATA))wrs++;
  return wrs;
}

/* Release a DESC page (2 frames on j0) and every unique src_page it still holds (for dead-child / drop cleanup). */
static inline void mesh_desc_drop(struct hdr *m,struct mesh_switch *sw,uint32_t dpage,uint32_t cls,uint32_t target_port,uint32_t ring,int unqueue){
  if(dpage>=mesh_flat_pages(m))return;
  uint32_t pages_dropped=2u;
  if(cls==MESH_CLS_BULK8){
    const struct mesh_wire_desc *d=mesh_desc_at(m,dpage);
    const uint32_t n=d->count<MESH_WIRE_BATCH?d->count:MESH_WIRE_BATCH;
    for(uint32_t i=0;i+8u<=n;i+=8u){
      if(!(d->e[i].flags&MESH_WIRE_SAME_DATA)){
        mesh_page_release_run(m,d->e[i].src_page,8u);
        pages_dropped+=8u;
      }
    }
  }
  mesh_page_release_run(m,dpage,2u);
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
        cur=atomic_load_explicit(mesh_class_next(m,cls,d),memory_order_relaxed);
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
