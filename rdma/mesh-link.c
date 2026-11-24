#include "mesh-bridge.h"
#include <spawn.h>
#include <netinet/tcp.h>
#include <sys/wait.h>
#include <crt_externs.h>
#include <sysexits.h>
#include <mach/mach_time.h>

struct hdr *M;
struct wire_mapping wire;
struct port ports[MESH_PORTS];
uint32_t port_count;
int events=-1,listener=-1,hello=-1,network=-1,addresses=-1;
uint64_t now,sequence;
struct mesh_ctx bridge={.fd=-1};
/* design/transport.md B9b: the window of a peer interface's termination (`exit_at`), as every port's quiet
   window stands, before which the bridge does not exit, and which a port made meanwhile takes as its own; in each
   process its main thread's alone (the parent's event loop, a child's main thread: their IOKit callbacks run there) */
static uint64_t exit_at;
/* a window's end moved to `until` where it is earlier, never earlier */
static void quiet_until(uint64_t *at,uint64_t until){ if(*at<until)*at=until; }
/* the latest of the exit's window and every port's (B9, B9b): the stop's wait, and a child's end's */
static uint64_t windows_end(void){
  uint64_t latest=exit_at;
  for(uint32_t p=0;p<port_count;p++)latest=MAX(latest,ports[p].quiet_at);
  return latest;
}
/* design/transport.md section 2: the node's start and stop, one state machine on the event loop (node_step, its one
   transition function; the loop its one owner), each state's deadline `due`:
   - ADOPTING: the region a bridge before this one left adopted once every child that bridge recorded has exited (their
     exits on the loop's queue, TAG_ADOPT: until a child's exit has begun its poller may still post, take and land
     on what the adoption gives back), at most 2 MESH_PLATFORM_NS on, a child's end's bound where its parent ended
     (child_main); one that lives past it ends this bridge, which its heartbeat opens again;
   - SETTLING: the node started (node_start), its IOKit notifications armed on the loop's queue (iokit_watch: its
     ports made from IOKit's first match of its interfaces) and no port opened until IOKit's registry is quiet (its
     busy state 0, kIOBusyInterest: no termination in its course, B9), at most MESH_PLATFORM_NS on;
   - RUNNING: ports opened, paired and served;
   - STOPPING: a stop signal taken: every child halted, every control socket closed (its peer reads the stop at the
     socket's end), every line leaving and `bridge_pid` 0; the loop goes on serving the children's sockets, the
     clients' exits and the kernel's link events (a link going off moves its port's window) until every quiet window
     has ended (B9, B9b), at most MESH_PLATFORM_NS after the stop;
   - ENDING: the bridge ended once every child, each halted at the stop, is reaped (PORT_EXITED, its exit seen
     through to its end), at most MESH_PLATFORM_NS on.
   `adopting` of the bridge before's children (`children`) live, as ADOPTING last read them; `id` this node's */
enum node_state { NODE_ADOPTING, NODE_SETTLING, NODE_RUNNING, NODE_STOPPING, NODE_ENDING };
enum node_event { NODE_STOP, NODE_EXIT, NODE_QUIET, NODE_DUE };
static struct { enum node_state state; uint32_t id,adopting; uint64_t due; struct mesh_process children[2*MESH_PORTS]; } node;
#define TAG_ADOPT (UINT64_C(1)<<57)
static int stopping(void){ return node.state>=NODE_STOPPING; }
static void node_step(enum node_event event,int64_t code);
static const struct in6_addr everyone={{{0xff,0x02,0,0,0,0,0,0,0,0,0,0,0,0,0,1}}};

static uint64_t clock_ns(void){ return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
uint64_t memory;
/* why this bridge stops: the signal it took */
static void stop_text(int64_t code,char *out,size_t room){
  if(code==SIGHUP)snprintf(out,room,"SIGHUP: its heartbeat restarts it");
  else if(code==SIGTERM)snprintf(out,room,"SIGTERM: its heartbeat's release or stop");
  else if(code>0)snprintf(out,room,"signal %lld",(long long)code);
  else snprintf(out,room,"no reason given");
}
/* the port's line (mesh-switched.h struct mesh_switch `port`) leaving where its child has made its lanes or serves
   them: from the next pass on no draw pushes onto its lanes and no child takes for it (a child whose lanes are not
   made keeps its device for a later pairing) */
static inline struct mesh_switch *switch_of(void){ return mesh_switch_of(M); }
static void port_leaving(const struct port *port){
  mesh_port_move(switch_of(),port->index,1u<<CHILD_LANES_MADE|1u<<CHILD_SERVING,CHILD_LEAVING,0,0);
}
/* design/transport.md section 14: a child's own halt, its port's line leaving from whatever state it is in (one
   compare-exchange, mesh_port_move: its poller posts nothing from its next pass and returns, and the child takes no
   command again and ends), and, where this call moved it, `reason` stored for its main thread to tell its parent
   (child_report, CHILD_BROKEN: EIO a lane's or a completion's failure, ENETDOWN a doorbell fault, ETIMEDOUT a
   termination; 0, a make or connect that failed or the child's end, tells nothing).  A line leaves once in a child's
   life, so one reason at most is ever stored.  From any context of the child: its poller (a lane's failure), its main
   thread (its IOKit callbacks, its commands, its end) and the fault handler (atomics alone).  1 where this call moved
   the line */
static _Atomic uint32_t child_reason;
/* a port child's one link (child_main) */
static struct link child_link;
static int child_halt(const struct link *l,uint32_t reason){
  if(!mesh_port_move(switch_of(),l->index,1u<<CHILD_STARTED|1u<<CHILD_DEVICE_OPEN|1u<<CHILD_LANES_MADE|1u<<CHILD_SERVING,
                     CHILD_LEAVING,0,0))return 0;
  if(reason)atomic_store(&child_reason,reason);
  return 1;
}

/* a lane's failure, found by the poller: its child halted (child_halt, EIO: nothing more is posted on any lane
   of the link, as at a failed completion, link_failed, and no draw takes any of them from the next pass on), and the
   link lost by its parent on the report */
static void lane_lost(struct link *l,uint32_t lane,int64_t code){
  if(child_halt(l,EIO))
    say("link %u (%s): lane %u failed: %s; the link toward node %u lost\n",l->index,l->name,lane,
        strerror((int)(code>0?code:EIO)),l->peer);
}

/* the lane records of port p's lane pair s (mesh.h struct mesh_lanes, the bridge's own) */
static inline struct mesh_lanes *lanes_of(const struct link *l,uint32_t s){ return mesh_lanes_at(M,l->index*MESH_LINK_STREAMS+s); }
/* the SGE of an exact byte span of the region (a page's frames: within one registration extent, the pool's carve) */
_Static_assert(MESH_DMA_EXTENT==MESH_EXTENT,"span validation matches registration geometry");
static inline struct ibv_sge dma_span(const struct link *l,struct mesh_dma_span span){
  const uint64_t at=span.offset-M->data_off;
  return (struct ibv_sge){.addr=(uintptr_t)l->device.wire+at,.length=(uint32_t)span.bytes,
    .lkey=l->device.extent_mrs[at/MESH_EXTENT]->lkey};
}
static void link_failed(struct link *,int64_t,uint32_t);
/* the lane a completion's queue pair number names: one of the link's lanes, else MESH_LINK_LANES (dropped; a lane its
   plan leaves out has the number UINT32_MAX, which no queue pair has) */
static __attribute__((always_inline)) inline uint32_t link_lane(const struct link *l,uint32_t qp_num){
  uint32_t k=0;
  while(k<MESH_LINK_LANES && l->qpn[k]!=qp_num)k++;
  return k;
}
/* design/transport.md: the link's first failed completion, on any lane, or a failed poll: its child halted
   (child_halt, EIO), so nothing is posted on it again, and the link lost by its parent on the report */
static void link_failed(struct link *l,int64_t code,uint32_t domain){
  if(child_halt(l,EIO))
    say("link %u (%s): a completion failed (code %lld, domain %u): no lane served, the link toward node %u lost\n",
        l->index,l->name,(long long)code,domain,l->peer);
}

/* design/SPEC-flat-page-dma-bridge.md section 5.3: flat-page uniform 1-frame data path */
static inline uint32_t switch_lane(uint32_t p,uint32_t j){ return p*2*MESH_LINK_STREAMS+j; }
static struct mesh_switch_togo togo;

static inline uint32_t poller_seed(const struct link *l){
  static _Thread_local uint32_t cursor;
  const uint32_t base=36864u+l->index*2048u;
  return base+(cursor++&1023u);
}

static inline void tx_append_local(struct link *l,struct mesh_switch *sw,uint32_t cls,uint32_t head){
  if(!head)return;
  uint32_t tail=head;
  const uint32_t pages=mesh_flat_pages(M);
  for(;;){
    const uint32_t q=tail-1u;
    if(q>=pages)break;
    const uint32_t nxt=atomic_load_explicit(mesh_class_next(M,cls,q),memory_order_relaxed);
    if(!nxt)break;
    tail=nxt;
  }
  if(!atomic_load_explicit(&sw->tx_head[l->index][cls],memory_order_relaxed)){
    atomic_store_explicit(&sw->tx_head[l->index][cls],head,memory_order_release);
    l->tx_tail[cls]=tail;
  }else{
    atomic_store_explicit(mesh_class_next(M,cls,l->tx_tail[cls]-1u),head,memory_order_release);
    l->tx_tail[cls]=tail;
  }
}

static inline void port_enqueue_desc(struct link *l,struct mesh_switch *sw,uint32_t target,uint32_t cls,uint32_t dp,uint32_t data_wrs){
  const uint64_t fbytes=(uint64_t)mesh_switch_class_frames(cls)*MESH_PAGE_BYTES;
  atomic_store_explicit(mesh_class_next(M,cls,dp),0,memory_order_relaxed);
  atomic_fetch_add_explicit(&sw->cable[target].queued,(uint64_t)(1u+data_wrs)*fbytes,memory_order_relaxed);
  if(target==l->index)tx_append_local(l,sw,cls,dp+1u);
  else mesh_llist_add_cls(M,&sw->port_q[target][cls],cls,dp);
}

/* Earliest-arrival routing for unicast (all cables + 2-hop overflow), and direct-first
   multicast piggybacking (SAME_DATA) for multi-destination groups. */
static void route_multicast_group(const struct mesh_switch_view *v,const uint64_t *line,struct mesh_switch_terms *terms,
                                  const uint32_t *dsts,uint32_t nd,uint32_t origin,uint32_t from,uint64_t unit_bytes,
                                  uint32_t *out_ports){
  if(nd==1u){
    const uint32_t tp=mesh_switch_route_dest(v,&togo,line,terms,dsts[0],origin,from);
    out_ports[0]=tp;
    if(tp<MESH_PORTS)terms[tp].backlog+=(double)unit_bytes/terms[tp].rate;
    return;
  }
  const uint32_t nodes_read=__atomic_load_n(&v->nodes,__ATOMIC_RELAXED),nodes=nodes_read<MESH_SWITCH_N?nodes_read:MESH_SWITCH_N;
  const uint32_t self=mesh_switch_index(v,v->self);
  uint64_t avoid=self<64?(UINT64_C(1)<<self):0;
  const uint32_t oi=mesh_switch_index(v,origin),fi=mesh_switch_index(v,from);
  uint32_t used_mask=0;
  for(uint32_t i=0;i<nd;i++)out_ports[i]=UINT32_MAX;
  /* Pass 1: route direct neighbors via earliest-arrival (allowing 2-hop overflow when congested) */
  for(uint32_t i=0;i<nd;i++){
    int is_direct=0;
    for(uint32_t p=0;p<MESH_PORTS;p++)
      if(mesh_port_state(line[p])==CHILD_SERVING && mesh_port_plan(line[p]) && mesh_port_peer(line[p])==dsts[i]){is_direct=1;break;}
    if(!is_direct)continue;
    const uint32_t best_p=mesh_switch_route_dest(v,&togo,line,terms,dsts[i],origin,from);
    out_ports[i]=best_p;
    if(best_p<MESH_PORTS && !(used_mask&(1u<<best_p))){
      used_mask|=1u<<best_p;
      terms[best_p].backlog+=(double)unit_bytes/terms[best_p].rate;
    }
  }
  /* Pass 2: multi-hop destinations piggyback for free on already-chosen ports when reachable */
  for(uint32_t i=0;i<nd;i++){
    if(out_ports[i]<MESH_PORTS)continue;
    const uint32_t d=mesh_switch_index(v,dsts[i]);
    if(d>=nodes)continue;
    uint64_t av=avoid;
    if(oi<nodes && oi!=d)av|=UINT64_C(1)<<oi;
    if(fi<nodes && fi!=d)av|=UINT64_C(1)<<fi;
    double best=INFINITY;uint32_t best_p=UINT32_MAX;
    for(uint32_t m=used_mask;m;m&=m-1){
      const uint32_t p=(uint32_t)__builtin_ctz(m);
      const uint32_t nb=mesh_switch_index(v,mesh_port_peer(line[p]));
      if(nb>=nodes || (av>>nb&1))continue;
      if(nb!=d && !mesh_switch_reaches(v,nb,av,d))continue;
      const double cost=terms[p].backlog+terms[p].own+(nb==d?0.0:(double)togo.to[nb][d]);
      if(cost<best){best=cost;best_p=p;}
    }
    if(best_p<MESH_PORTS){
      out_ports[i]=best_p;
      continue;
    }
    best_p=mesh_switch_route_dest(v,&togo,line,terms,dsts[i],origin,from);
    out_ports[i]=best_p;
    if(best_p<MESH_PORTS && !(used_mask&(1u<<best_p))){
      used_mask|=1u<<best_p;
      terms[best_p].backlog+=(double)unit_bytes/terms[best_p].rate;
    }
  }
}

#define RECEIVE_BATCH 64u
static int receives_post(struct link *l,uint32_t j,const uint32_t *page,uint32_t n){
  struct mesh_switch *sw=switch_of();
  struct mesh_switch_lane *x=sw->lane+switch_lane(l->index,j);
  struct lane_state *state=l->lane+j;
  const struct mesh_queue *queue=l->provider.queues+j;
  const uint32_t f=mesh_switch_lane_frames(j);
  const uint64_t fbytes=(uint64_t)f*MESH_PAGE_BYTES;
  const uint64_t first=atomic_load_explicit(&x->rx_posted,memory_order_relaxed);
  struct ibv_recv_wr request[RECEIVE_BATCH];struct ibv_sge span[RECEIVE_BATCH];
  for(uint32_t k=0;k<n;k++){
    const uint64_t at=first+k;
    x->rx_ring[at%MESH_SWITCH_FLIGHT]=page[k];
    span[k]=dma_span(l,(struct mesh_dma_span){mesh_page_offset(M,page[k]),fbytes});
    request[k]=(struct ibv_recv_wr){.wr_id=((uint64_t)j<<48)|(at&((UINT64_C(1)<<48)-1)),.sg_list=span+k,.num_sge=1};
    if(k)request[k-1].next=request+k;
  }
  struct ibv_recv_wr *bad=NULL;
  const int error=queue->receive(queue->pair,request,&bad);
  const uint32_t made=error?(bad>=request && bad<request+n?(uint32_t)(bad-request):0):n;
  atomic_store_explicit(&x->rx_posted,first+made,memory_order_release);
  for(uint32_t k=made;k<n;k++)mesh_page_release_run(M,page[k],f);
  state->posted+=made;
  atomic_store_explicit(&lanes_of(l,j/2)->posts[1].posted[j&1],state->posted,memory_order_relaxed);
  if(error)lane_lost(l,j,error<0?-error:error);
  return error;
}

static void switch_receives(struct link *l,uint32_t j){
  struct lane_state *state=l->lane+j;
  const uint32_t f=mesh_switch_lane_frames(j);
  const uint64_t most=MIN(state->frame_capacity/f,(uint64_t)MESH_SWITCH_FLIGHT);
  while(state->posted-state->got<most){
    const uint64_t need=most-(state->posted-state->got);
    const uint32_t want=(uint32_t)(need<RECEIVE_BATCH?need:RECEIVE_BATCH);
    uint32_t page[RECEIVE_BATCH],n=0;
    if(f==1u){
      n=mesh_page_alloc_batch(M,page,want,poller_seed(l));
    }else{
      for(;n<want;n++){
        const uint32_t base=mesh_page_alloc_run(M,f,poller_seed(l));
        if(base==MESH_PAGE_NONE)break;
        page[n]=base;
      }
    }
    if(n<want)MESH_SWITCH_COUNT(switch_of(),l->index,starved,1);
    if(!n || receives_post(l,j,page,n) || n<want)return;
  }
}

static int sends_post(struct link *l,uint32_t j,struct ibv_send_wr *request,uint32_t n,uint64_t first){
  struct mesh_switch *sw=switch_of();
  struct mesh_switch_lane *x=sw->lane+switch_lane(l->index,j);
  const struct mesh_queue *queue=l->provider.queues+j;
  const uint32_t f=mesh_switch_lane_frames(j);
  const uint64_t fbytes=(uint64_t)f*MESH_PAGE_BYTES;
  struct ibv_send_wr *bad=NULL;
  const uint64_t posting=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  const int error=queue->send(queue->pair,request,&bad);
  const uint64_t posted=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  const uint32_t made=error?(bad>=request && bad<request+n?(uint32_t)(bad-request):0):n;
  if(made<n)atomic_store_explicit(&x->posted,first+made,memory_order_relaxed);
  atomic_store_explicit(&lanes_of(l,j/2)->posts[0].posted[j&1],first+made,memory_order_relaxed);
  MESH_SWITCH_COUNT(sw,l->index,post_call_ns,posted-posting);
  for(uint32_t k=0;k<made;k++){
    struct mesh_switch_entry *e=x->flight+(first+k)%MESH_SWITCH_FLIGHT;
    MESH_SWITCH_COUNT(sw,l->index,posts,1);MESH_SWITCH_COUNT(sw,l->index,post_wait_ns,posting>e->at?posting-e->at:0);
    e->at=posted;
  }
  for(uint32_t k=made;k<n;k++){
    atomic_fetch_sub_explicit(&sw->cable[l->index].queued,fbytes,memory_order_relaxed);
    mesh_page_release_run(M,x->flight[(first+k)%MESH_SWITCH_FLIGHT].page,f);
    MESH_SWITCH_COUNT(sw,l->index,lost,f);
  }
  if(error)lane_lost(l,j,error<0?-error:error);
  return !error;
}

static void switch_sends(struct link *l,uint32_t served){
  struct mesh_switch *sw=switch_of();
  static _Thread_local struct ibv_send_wr request[MESH_WIRE_BATCH+1u];
  static _Thread_local struct ibv_sge span[MESH_WIRE_BATCH+1u];
  const uint64_t now_at=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  for(uint32_t cls=0;cls<MESH_SWITCH_CLASSES;cls++){
    if(atomic_load_explicit(&sw->port_q[l->index][cls],memory_order_relaxed)){
      const uint32_t rev=mesh_llist_reverse_cls(M,cls,mesh_llist_del_all(M,&sw->port_q[l->index][cls]));
      tx_append_local(l,sw,cls,rev);
    }
    const uint32_t cmask=served&mesh_switch_class_mask(cls);
    if(!cmask)continue;
    const uint32_t f=mesh_switch_class_frames(cls);
    const uint64_t fbytes=(uint64_t)f*MESH_PAGE_BYTES;
    for(;;){
      const uint32_t cur=(uint32_t)atomic_load_explicit(&sw->tx_head[l->index][cls],memory_order_relaxed);
      if(!cur)break;
      const uint32_t dp=cur-1u;
      const uint32_t wrs=cls==MESH_CLS_INLINE?1u:(1u+mesh_desc_data_wrs(mesh_desc_at(M,dp),cls));
      uint32_t j=UINT32_MAX;uint64_t fewest=UINT64_MAX;
      for(uint32_t k=cmask;k;k&=k-1){
        const uint32_t i=(uint32_t)__builtin_ctz(k);
        const struct mesh_switch_lane *x=sw->lane+switch_lane(l->index,i);
        const uint64_t out=atomic_load_explicit(&x->posted,memory_order_relaxed)-atomic_load_explicit(&x->completed,memory_order_relaxed);
        if(out+wrs<=l->lane[i].depth && out<fewest){fewest=out;j=i;}
      }
      if(j==UINT32_MAX)break;
      const uint32_t nxt=atomic_load_explicit(mesh_class_next(M,cls,dp),memory_order_relaxed);
      atomic_store_explicit(&sw->tx_head[l->index][cls],nxt,memory_order_release);
      if(!nxt)l->tx_tail[cls]=0;
      struct mesh_switch_lane *x=sw->lane+switch_lane(l->index,j);
      const uint64_t first=atomic_load_explicit(&x->posted,memory_order_relaxed);
      uint32_t w=0;
      x->flight[(first+w)%MESH_SWITCH_FLIGHT]=(struct mesh_switch_entry){now_at,dp,0};
      span[w]=dma_span(l,(struct mesh_dma_span){mesh_page_offset(M,dp),fbytes});
      request[w]=(struct ibv_send_wr){.wr_id=((uint64_t)j<<48)|((first+w)&((UINT64_C(1)<<48)-1)),.sg_list=span+w,.num_sge=1,
                                      .opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED};
      w++;
      if(cls==MESH_CLS_BULK8){
        const struct mesh_wire_desc *d=mesh_desc_at(M,dp);
        const uint32_t cnt=d->count<MESH_WIRE_BATCH?d->count:MESH_WIRE_BATCH;
        for(uint32_t i=0;i+8u<=cnt;i+=8u){
          if(d->e[i].flags&MESH_WIRE_SAME_DATA)continue;
          const uint32_t sp=d->e[i].src_page;
          x->flight[(first+w)%MESH_SWITCH_FLIGHT]=(struct mesh_switch_entry){now_at,sp,0};
          span[w]=dma_span(l,(struct mesh_dma_span){mesh_page_offset(M,sp),fbytes});
          request[w]=(struct ibv_send_wr){.wr_id=((uint64_t)j<<48)|((first+w)&((UINT64_C(1)<<48)-1)),.sg_list=span+w,.num_sge=1,
                                          .opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED};
          request[w-1].next=request+w;
          w++;
        }
      }
      atomic_store_explicit(&x->posted,first+w,memory_order_release);
      if(!sends_post(l,j,request,w,first))return;
    }
  }
}

static int link_send(struct link *l,uint32_t served){
  struct mesh_switch *sw=switch_of();
  struct ibv_wc done[RECEIVE_BATCH];
  const int count=l->provider.poll(l->provider.sent,RECEIVE_BATCH,done);
  if(count<0){link_failed(l,EIO,2);return 0;}
  if(!count)return 0;
  const uint64_t at=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  uint64_t outstanding=0,bytes=0,first=UINT64_MAX,alone_posted=0;int alone=0;
  for(uint32_t j=0;j<MESH_LINK_LANES;j++)if(served>>j&1){
    const struct mesh_switch_lane *x=sw->lane+switch_lane(l->index,j);
    outstanding+=atomic_load_explicit(&x->posted,memory_order_relaxed)-atomic_load_explicit(&x->completed,memory_order_relaxed);
  }
  for(int i=0;i<count;i++){
    const uint32_t j=link_lane(l,done[i].qp_num);
    if(j>=MESH_LINK_LANES || !(served>>j&1))continue;
    if(done[i].status){link_failed(l,(int64_t)done[i].status,2);return count;}
    const uint32_t f=mesh_switch_lane_frames(j);
    const uint64_t fbytes=(uint64_t)f*MESH_PAGE_BYTES;
    struct lane_state *state=l->lane+j;
    struct mesh_switch_lane *x=sw->lane+switch_lane(l->index,j);
    const struct mesh_switch_entry *e=x->flight+done[i].wr_id%MESH_SWITCH_FLIGHT;
    const uint32_t page=e->page;const uint64_t posted=e->at,completed=atomic_load_explicit(&x->completed,memory_order_relaxed)+1;
    atomic_store_explicit(&x->completed,completed,memory_order_relaxed);
    const uint64_t start=posted>state->last?posted:state->last;
    mesh_switch_sent(sw,switch_lane(l->index,j),fbytes,posted,start,at);
    bytes+=fbytes;first=MIN(first,posted);
    if(outstanding==1 && posted>=l->completed_at){alone=1;alone_posted=posted;}
    mesh_page_release_run(M,page,f);
    MESH_SWITCH_COUNT(sw,l->index,sent,1);MESH_SWITCH_COUNT(sw,l->index,send_ns,at-posted);
    state->last=at;
    struct mesh_lanes *record=lanes_of(l,j/2);
    atomic_store_explicit(&record->send_ns,atomic_load_explicit(&record->send_ns,memory_order_relaxed)+(at-posted),memory_order_relaxed);
    atomic_store_explicit(&record->sends,atomic_load_explicit(&record->sends,memory_order_relaxed)+1,memory_order_relaxed);
    atomic_store_explicit(&record->posts[0].done[j&1],completed,memory_order_relaxed);
  }
  if(!bytes)return count;
  mesh_switch_cable_sent(sw,l->index,bytes,first,l->completed_at,at,alone,alone_posted);
  l->completed_at=at;
  return count;
}

static inline void relay_bulk_flush_port(struct link *l,struct mesh_switch *sw,uint32_t tp){
  if(l->relay_bulk_dp[tp]==MESH_PAGE_NONE)return;
  mesh_desc_at(M,l->relay_bulk_dp[tp])->count=l->relay_bulk_n[tp];
  port_enqueue_desc(l,sw,tp,MESH_CLS_BULK8,l->relay_bulk_dp[tp],l->relay_bulk_wrs[tp]);
  MESH_SWITCH_COUNT(sw,l->index,groups,1);
  l->relay_bulk_dp[tp]=MESH_PAGE_NONE;l->relay_bulk_n[tp]=0;l->relay_bulk_wrs[tp]=0;
}

static int link_receive(struct link *l,uint32_t served,const uint64_t *line){
  struct mesh_switch *sw=switch_of();
  struct ibv_wc done[RECEIVE_BATCH];
  const int count=l->provider.poll(l->provider.completion,RECEIVE_BATCH,done);
  if(count<0){link_failed(l,EIO,3);return 0;}
  if(!count){
    for(uint32_t tp=0;tp<MESH_PORTS;tp++)relay_bulk_flush_port(l,sw,tp);
    return 0;
  }
  atomic_thread_fence(memory_order_acquire);
  const struct mesh_switch_view *v=NULL;
  struct mesh_switch_terms terms[MESH_PORTS];
  uint32_t relay_inl_dp[MESH_PORTS],relay_inl_n[MESH_PORTS]={0},relay_inl_src[MESH_PORTS][2];
  for(uint32_t p=0;p<MESH_PORTS;p++){
    relay_inl_dp[p]=MESH_PAGE_NONE;
    relay_inl_src[p][0]=relay_inl_src[p][1]=MESH_PAGE_NONE;
  }
  for(int i=0;i<count;i++){
    const uint32_t j=link_lane(l,done[i].qp_num);
    if(j>=MESH_LINK_LANES || !(served>>j&1))continue;
    if(done[i].status){link_failed(l,(int64_t)done[i].status,2);break;}
    const uint32_t f=mesh_switch_lane_frames(j);
    struct lane_state *state=l->lane+j;
    struct mesh_switch_lane *x=sw->lane+switch_lane(l->index,j);
    const uint32_t h=x->rx_ring[done[i].wr_id%MESH_SWITCH_FLIGHT];
    const uint64_t rx_done=atomic_load_explicit(&x->rx_done,memory_order_relaxed)+1;
    atomic_store_explicit(&x->rx_done,rx_done,memory_order_relaxed);
    state->got++;
    atomic_store_explicit(&lanes_of(l,j/2)->posts[1].done[j&1],state->got,memory_order_relaxed);
    atomic_fetch_add_explicit(&x->landed,1,memory_order_relaxed);
    MESH_SWITCH_COUNT(sw,l->index,landed,1);

    /* Class 0: 2-frame (8 KiB) 1-WR inline lane (slot 0 = h+0, slot 1 = h+1, tail at end of h+1) */
    if(f==2u){
      const struct mesh_wire_tail *tl=mesh_tail_at(M,h);
      const uint32_t cnt=tl->count<MESH_TAIL_MAX?tl->count:MESH_TAIL_MAX;
      uint32_t uses[2]={0,0};
      for(uint32_t k=0;k<cnt;k++)if(tl->e[k].src_page<2u)uses[tl->e[k].src_page]++;
      if(!uses[0])mesh_page_release(M,h); else for(uint32_t u=1u;u<uses[0];u++)mesh_page_hold(M,h);
      if(!uses[1])mesh_page_release(M,h+1u); else for(uint32_t u=1u;u<uses[1];u++)mesh_page_hold(M,h+1u);
      const uint32_t origin=tl->origin_node,from=tl->from_node;
      struct mesh_wire_entry ents[MESH_TAIL_MAX];
      memcpy(ents,tl->e,cnt*sizeof *ents);
      for(uint32_t k=0;k<cnt;k++){
        const struct mesh_wire_entry ent=ents[k];
        if(ent.src_page>=2u)continue;
        const uint32_t hp=h+ent.src_page;
        const uint32_t b=ent.bytes<=MESH_PAGE_BYTES?ent.bytes:MESH_PAGE_BYTES;
        if(ent.dst_node==M->node){
          mesh_table_land(M,ent.dst_index,hp,b);
          MESH_SWITCH_COUNT(sw,l->index,set,1);
          continue;
        }
        if(!v){v=sw->view+mesh_switch_published(&sw->view_version);mesh_switch_terms_read(sw,v,line,terms);}
        const uint32_t tp=ent.hops+1<MESH_SWITCH_N?mesh_switch_route_dest(v,&togo,line,terms,ent.dst_node,origin,from):UINT32_MAX;
        if(tp>=MESH_PORTS){mesh_page_release(M,hp);MESH_SWITCH_COUNT(sw,l->index,unreachable,1);continue;}
        if(relay_inl_dp[tp]!=MESH_PAGE_NONE){
          struct mesh_wire_tail *rt=mesh_tail_at(M,relay_inl_dp[tp]);
          const int fits=(relay_inl_src[tp][0]==hp || relay_inl_src[tp][1]==hp ||
                          relay_inl_src[tp][0]==MESH_PAGE_NONE ||
                          (relay_inl_src[tp][1]==MESH_PAGE_NONE && b<=MESH_TAIL_SLOT1_MAX));
          if(rt->origin_node!=origin || !fits || relay_inl_n[tp]>=MESH_TAIL_MAX){
            rt->count=relay_inl_n[tp];
            port_enqueue_desc(l,sw,tp,MESH_CLS_INLINE,relay_inl_dp[tp],0u);
            MESH_SWITCH_COUNT(sw,l->index,groups,1);
            relay_inl_dp[tp]=MESH_PAGE_NONE;relay_inl_n[tp]=0;
            relay_inl_src[tp][0]=relay_inl_src[tp][1]=MESH_PAGE_NONE;
          }
        }
        if(relay_inl_dp[tp]==MESH_PAGE_NONE){
          relay_inl_dp[tp]=mesh_page_alloc_run(M,2u,poller_seed(l));
          if(relay_inl_dp[tp]==MESH_PAGE_NONE){mesh_page_release(M,hp);MESH_SWITCH_COUNT(sw,l->index,lost,1);continue;}
          struct mesh_wire_tail *rt=mesh_tail_at(M,relay_inl_dp[tp]);
          rt->origin_node=origin;rt->from_node=M->node;
          relay_inl_n[tp]=0;
          relay_inl_src[tp][0]=relay_inl_src[tp][1]=MESH_PAGE_NONE;
          terms[tp].backlog+=(double)(2u*MESH_PAGE_BYTES)/terms[tp].rate;
        }
        uint32_t sl=0;
        if(relay_inl_src[tp][0]==hp)sl=0u;
        else if(relay_inl_src[tp][1]==hp)sl=1u;
        else if(relay_inl_src[tp][0]==MESH_PAGE_NONE){
          sl=0u;relay_inl_src[tp][0]=hp;
          if(b)memcpy(mesh_page_at(M,relay_inl_dp[tp]),mesh_page_at(M,hp),b);
        }else{
          sl=1u;relay_inl_src[tp][1]=hp;
          if(b)memcpy(mesh_page_at(M,relay_inl_dp[tp]+1u),mesh_page_at(M,hp),b);
        }
        mesh_page_release(M,hp);
        struct mesh_wire_tail *rt=mesh_tail_at(M,relay_inl_dp[tp]);
        rt->e[relay_inl_n[tp]++]=(struct mesh_wire_entry){.dst_node=ent.dst_node,.dst_index=ent.dst_index,.bytes=(uint16_t)b,
                                                          .flags=MESH_WIRE_INLINE,.hops=(uint8_t)(ent.hops+1u),.src_page=sl};
        MESH_SWITCH_COUNT(sw,l->index,forwards,1);
      }
      continue;
    }

    /* Class 1: 8-frame (32 KiB) bulk zero-copy lane with persistent cross-poll relay batching */
    if(f==8u){
      uint32_t rem=atomic_load_explicit(&x->rx_rem,memory_order_relaxed);
      if(!rem){
        const struct mesh_wire_desc *d=mesh_desc_at(M,h);
        const uint32_t cnt=d->count<MESH_WIRE_BATCH?d->count:MESH_WIRE_BATCH;
        mesh_page_release_run(M,h+1u,7u);
        if(cnt<8u){mesh_page_release(M,h);}
        else{
          l->rx_pos[j]=0;
          atomic_store_explicit(&x->rx_desc,h,memory_order_relaxed);
          atomic_store_explicit(&x->rx_rem,cnt,memory_order_relaxed);
        }
        continue;
      }
      const uint32_t dp=atomic_load_explicit(&x->rx_desc,memory_order_relaxed);
      const struct mesh_wire_desc *d=mesh_desc_at(M,dp);
      uint32_t pos=l->rx_pos[j];
      uint32_t groups=1u;
      while(groups*8u<rem && pos+groups*8u+8u<=MESH_WIRE_BATCH && (d->e[pos+groups*8u].flags&MESH_WIRE_SAME_DATA))groups++;
      for(uint32_t g=1u;g<groups;g++)for(uint32_t k=0;k<8u;k++)mesh_page_hold(M,h+k);
      for(uint32_t g=0;g<groups;g++){
        const struct mesh_wire_entry *ge=d->e+pos;
        pos+=8u;rem=rem>=8u?rem-8u:0u;
        if(ge[0].dst_node==M->node){
          for(uint32_t k=0;k<8u;k++){
            const uint32_t b=ge[k].bytes<=MESH_PAGE_BYTES?ge[k].bytes:MESH_PAGE_BYTES;
            mesh_table_land(M,ge[k].dst_index,h+k,b);
          }
          MESH_SWITCH_COUNT(sw,l->index,set,8);
          continue;
        }
        if(!v){v=sw->view+mesh_switch_published(&sw->view_version);mesh_switch_terms_read(sw,v,line,terms);}
        const uint32_t tp=ge[0].hops+1<MESH_SWITCH_N?mesh_switch_route_dest(v,&togo,line,terms,ge[0].dst_node,d->origin_node,d->from_node):UINT32_MAX;
        if(tp>=MESH_PORTS){mesh_page_release_run(M,h,8u);MESH_SWITCH_COUNT(sw,l->index,unreachable,8);continue;}
        if(l->relay_bulk_dp[tp]!=MESH_PAGE_NONE && mesh_desc_at(M,l->relay_bulk_dp[tp])->origin_node!=d->origin_node)
          relay_bulk_flush_port(l,sw,tp);
        if(l->relay_bulk_dp[tp]==MESH_PAGE_NONE){
          l->relay_bulk_dp[tp]=mesh_page_alloc_run(M,8u,poller_seed(l));
          if(l->relay_bulk_dp[tp]==MESH_PAGE_NONE){mesh_page_release_run(M,h,8u);MESH_SWITCH_COUNT(sw,l->index,lost,8);continue;}
          struct mesh_wire_desc *rd=mesh_desc_at(M,l->relay_bulk_dp[tp]);
          rd->origin_node=d->origin_node;rd->from_node=M->node;
          l->relay_bulk_n[tp]=0;l->relay_bulk_wrs[tp]=0;
        }
        struct mesh_wire_desc *rd=mesh_desc_at(M,l->relay_bulk_dp[tp]);
        uint8_t fl=MESH_WIRE_BULK8;
        if(l->relay_bulk_n[tp]>=8u && rd->e[l->relay_bulk_n[tp]-8u].src_page==h){
          fl|=MESH_WIRE_SAME_DATA;
          mesh_page_release_run(M,h,8u);
        }else{
          l->relay_bulk_wrs[tp]++;
          terms[tp].backlog+=(double)(8u*MESH_PAGE_BYTES)/terms[tp].rate;
        }
        for(uint32_t k=0;k<8u;k++){
          const uint32_t b=ge[k].bytes<=MESH_PAGE_BYTES?ge[k].bytes:MESH_PAGE_BYTES;
          rd->e[l->relay_bulk_n[tp]+k]=(struct mesh_wire_entry){
            .dst_node=ge[k].dst_node,.dst_index=ge[k].dst_index,.bytes=(uint16_t)b,
            .flags=(k==0?fl:0u),.hops=(uint8_t)(ge[k].hops+1u),.src_page=h+k
          };
        }
        l->relay_bulk_n[tp]+=8u;
        MESH_SWITCH_COUNT(sw,l->index,forwards,8);
        if(l->relay_bulk_n[tp]>=128u || l->relay_bulk_wrs[tp]>=16u)
          relay_bulk_flush_port(l,sw,tp);
      }
      l->rx_pos[j]=pos;
      atomic_store_explicit(&x->rx_rem,rem,memory_order_relaxed);
      if(!rem){
        atomic_store_explicit(&x->rx_desc,MESH_PAGE_NONE,memory_order_relaxed);
        mesh_page_release(M,dp);
        for(uint32_t tp=0;tp<MESH_PORTS;tp++)relay_bulk_flush_port(l,sw,tp);
      }
      continue;
    }
  }
  for(uint32_t tp=0;tp<MESH_PORTS;tp++){
    if(relay_inl_dp[tp]!=MESH_PAGE_NONE){
      struct mesh_wire_tail *rt=mesh_tail_at(M,relay_inl_dp[tp]);
      rt->count=relay_inl_n[tp];
      port_enqueue_desc(l,sw,tp,MESH_CLS_INLINE,relay_inl_dp[tp],0u);
      MESH_SWITCH_COUNT(sw,l->index,groups,1);
    }
  }
  return count;
}

static inline int is_bulk8_run(const struct mesh_wire_entry *e,uint32_t pages){
  const uint32_t s=e[0].src_page,dn=e[0].dst_node;
  if(s+8u>pages)return 0;
  if((s/(MESH_DMA_EXTENT/MESH_PAGE_BYTES))!=((s+7u)/(MESH_DMA_EXTENT/MESH_PAGE_BYTES)))return 0;
  for(uint32_t k=1u;k<8u;k++)if(e[k].dst_node!=dn || e[k].src_page!=s+k)return 0;
  return 1;
}

static uint32_t switch_takes(struct link *l,const uint64_t *line){
  if(!atomic_load_explicit(mesh_pending_head(M),memory_order_relaxed))return 0;
  uint32_t head=mesh_llist_reverse(M,mesh_llist_del_all(M,mesh_pending_head(M)));
  if(!head)return 0;
  struct mesh_switch *sw=switch_of();
  const struct mesh_switch_view *v=sw->view+mesh_switch_published(&sw->view_version);
  struct mesh_switch_terms terms[MESH_PORTS];
  mesh_switch_terms_read(sw,v,line,terms);
  uint32_t inl_dp[MESH_PORTS],inl_n[MESH_PORTS]={0},inl_src[MESH_PORTS][2];
  uint32_t bulk_dp[MESH_PORTS],bulk_n[MESH_PORTS]={0},bulk_wrs[MESH_PORTS]={0};
  for(uint32_t p=0;p<MESH_PORTS;p++){
    inl_dp[p]=bulk_dp[p]=MESH_PAGE_NONE;
    inl_src[p][0]=inl_src[p][1]=MESH_PAGE_NONE;
  }
  uint32_t taken=0;
  const uint32_t pages=mesh_flat_pages(M);
  while(head){
    const uint32_t q=head-1u;
    if(q>=pages)break;
    head=atomic_load_explicit(&mesh_desc_at(M,q)->next,memory_order_relaxed);
    struct mesh_wire_desc *d=mesh_desc_at(M,q);
    const uint32_t cnt=d->count<MESH_WIRE_BATCH?d->count:MESH_WIRE_BATCH;
    if(!cnt){mesh_page_release(M,q);continue;}
    taken+=cnt;
    MESH_SWITCH_COUNT(sw,l->index,taken,cnt);

    /* Fast path: <= 2 unique src_pages fitting in 8 KiB (slot 0 <= 4096 B, slot 1 <= 3824 B) */
    if(cnt<=MESH_TAIL_MAX){
      uint32_t up[2]={MESH_PAGE_NONE,MESH_PAGE_NONE},ub[2]={0,0},nup=0;
      int fits=1;
      for(uint32_t i=0;i<cnt;i++){
        const uint32_t sp=d->e[i].src_page,b=d->e[i].bytes<=MESH_PAGE_BYTES?d->e[i].bytes:MESH_PAGE_BYTES;
        if(nup>=1u && up[0]==sp){if(b>ub[0])ub[0]=b;continue;}
        if(nup>=2u && up[1]==sp){if(b>ub[1])ub[1]=b;continue;}
        if(nup<2u){up[nup]=sp;ub[nup]=b;nup++;}
        else{fits=0;break;}
      }
      if(fits && nup==2u && ub[1]>MESH_TAIL_SLOT1_MAX){
        if(ub[0]<=MESH_TAIL_SLOT1_MAX){
          uint32_t tp_=up[0];up[0]=up[1];up[1]=tp_;
          uint32_t tb_=ub[0];ub[0]=ub[1];ub[1]=tb_;
        }else fits=0;
      }
      if(fits && nup>0u){
        uint32_t udst[MESH_TAIL_MAX],nd=0,uport[MESH_TAIL_MAX];
        for(uint32_t i=0;i<cnt;i++){
          uint32_t fnd=0;
          for(uint32_t u=0;u<nd;u++)if(udst[u]==d->e[i].dst_node){fnd=1;break;}
          if(!fnd)udst[nd++]=d->e[i].dst_node;
        }
        route_multicast_group(v,line,terms,udst,nd,M->node,M->node,(uint64_t)2u*MESH_PAGE_BYTES,uport);
        uint32_t fdp[MESH_PORTS],fn[MESH_PORTS]={0};
        for(uint32_t p=0;p<MESH_PORTS;p++)fdp[p]=MESH_PAGE_NONE;
        int ok=1;
        for(uint32_t i=0;i<cnt;i++){
          uint32_t tp=UINT32_MAX;
          for(uint32_t u=0;u<nd;u++)if(udst[u]==d->e[i].dst_node){tp=uport[u];break;}
          if(tp>=MESH_PORTS)continue;
          if(fdp[tp]==MESH_PAGE_NONE){
            fdp[tp]=mesh_page_alloc_run(M,2u,poller_seed(l));
            if(fdp[tp]==MESH_PAGE_NONE){ok=0;break;}
            struct mesh_wire_tail *it=mesh_tail_at(M,fdp[tp]);
            it->origin_node=M->node;it->from_node=M->node;
            if(ub[0])memcpy(mesh_page_at(M,fdp[tp]),mesh_page_at(M,up[0]),ub[0]);
            if(nup==2u && ub[1])memcpy(mesh_page_at(M,fdp[tp]+1u),mesh_page_at(M,up[1]),ub[1]);
          }
          struct mesh_wire_tail *it=mesh_tail_at(M,fdp[tp]);
          const uint32_t sl=(d->e[i].src_page==up[0])?0u:1u;
          it->e[fn[tp]++]=(struct mesh_wire_entry){
            .dst_node=d->e[i].dst_node,.dst_index=d->e[i].dst_index,.bytes=d->e[i].bytes,
            .flags=MESH_WIRE_INLINE,.hops=0,.src_page=sl
          };
        }
        if(ok){
          for(uint32_t i=0;i<cnt;i++){
            uint32_t tp=UINT32_MAX;
            for(uint32_t u=0;u<nd;u++)if(udst[u]==d->e[i].dst_node){tp=uport[u];break;}
            if(tp>=MESH_PORTS)MESH_SWITCH_COUNT(sw,l->index,unreachable,1);
            else MESH_SWITCH_COUNT(sw,l->index,forwards,1);
            mesh_page_release(M,d->e[i].src_page);
          }
          for(uint32_t tp=0;tp<MESH_PORTS;tp++){
            if(fdp[tp]==MESH_PAGE_NONE)continue;
            mesh_tail_at(M,fdp[tp])->count=fn[tp];
            port_enqueue_desc(l,sw,tp,MESH_CLS_INLINE,fdp[tp],0u);
            MESH_SWITCH_COUNT(sw,l->index,groups,1);
          }
          mesh_page_release(M,q);
          continue;
        }
        for(uint32_t tp=0;tp<MESH_PORTS;tp++)if(fdp[tp]!=MESH_PAGE_NONE)mesh_page_release_run(M,fdp[tp],2u);
      }
    }

    /* Walk entries: 8-page contiguous bulk blocks -> cls 1 (32 KiB); all else -> cls 0 (12 KiB inline) */
    for(uint32_t i=0;i<cnt;){
      if(i+8u<=cnt && is_bulk8_run(d->e+i,pages)){
        const uint32_t s=d->e[i].src_page;
        uint32_t g=1u;
        while(i+(g+1u)*8u<=cnt && g<MESH_DESTS && d->e[i+g*8u].src_page==s && is_bulk8_run(d->e+i+g*8u,pages))g++;
        uint32_t gdst[MESH_DESTS],gport[MESH_DESTS];
        for(uint32_t b=0;b<g;b++)gdst[b]=d->e[i+b*8u].dst_node;
        route_multicast_group(v,line,terms,gdst,g,M->node,M->node,(uint64_t)8u*MESH_PAGE_BYTES,gport);
        for(uint32_t b=0;b<g;b++){
          const struct mesh_wire_entry *ge=d->e+i+b*8u;
          const uint32_t tp=gport[b];
          if(tp>=MESH_PORTS){mesh_page_release_run(M,s,8u);MESH_SWITCH_COUNT(sw,l->index,unreachable,8);continue;}
          if(bulk_dp[tp]==MESH_PAGE_NONE){
            bulk_dp[tp]=mesh_page_alloc_run(M,8u,poller_seed(l));
            if(bulk_dp[tp]==MESH_PAGE_NONE){mesh_page_release_run(M,s,8u);MESH_SWITCH_COUNT(sw,l->index,lost,8);continue;}
            struct mesh_wire_desc *bd=mesh_desc_at(M,bulk_dp[tp]);
            bd->origin_node=M->node;bd->from_node=M->node;
            bulk_n[tp]=0;bulk_wrs[tp]=0;
          }
          struct mesh_wire_desc *bd=mesh_desc_at(M,bulk_dp[tp]);
          uint8_t fl=MESH_WIRE_BULK8;
          if(bulk_n[tp]>=8u && bd->e[bulk_n[tp]-8u].src_page==s){
            fl|=MESH_WIRE_SAME_DATA;
            mesh_page_release_run(M,s,8u);
          }else{
            bulk_wrs[tp]++;
          }
          for(uint32_t k=0;k<8u;k++){
            bd->e[bulk_n[tp]+k]=(struct mesh_wire_entry){
              .dst_node=ge[k].dst_node,.dst_index=ge[k].dst_index,.bytes=ge[k].bytes,
              .flags=(k==0?fl:0u),.hops=0,.src_page=s+k
            };
          }
          bulk_n[tp]+=8u;
          MESH_SWITCH_COUNT(sw,l->index,forwards,8);
          if(bulk_n[tp]>=128u || bulk_wrs[tp]>=16u){
            bd->count=bulk_n[tp];
            port_enqueue_desc(l,sw,tp,MESH_CLS_BULK8,bulk_dp[tp],bulk_wrs[tp]);
            MESH_SWITCH_COUNT(sw,l->index,groups,1);
            bulk_dp[tp]=MESH_PAGE_NONE;bulk_n[tp]=0;bulk_wrs[tp]=0;
          }
        }
        i+=g*8u;
        continue;
      }
      /* Scattered 1-page group packed into 8 KiB inline frames (slot 0 <= 4096 B, slot 1 <= 3824 B) */
      const uint32_t sp=d->e[i].src_page;
      uint32_t g=1u;
      while(i+g<cnt && g<MESH_DESTS && d->e[i+g].src_page==sp)g++;
      uint32_t gdst[MESH_DESTS],gport[MESH_DESTS];
      for(uint32_t b=0;b<g;b++)gdst[b]=d->e[i+b].dst_node;
      route_multicast_group(v,line,terms,gdst,g,M->node,M->node,(uint64_t)2u*MESH_PAGE_BYTES,gport);
      for(uint32_t b=0;b<g;b++){
        const struct mesh_wire_entry ent=d->e[i+b];
        const uint32_t tp=gport[b];
        if(tp>=MESH_PORTS){mesh_page_release(M,ent.src_page);MESH_SWITCH_COUNT(sw,l->index,unreachable,1);continue;}
        const uint32_t nb=ent.bytes<=MESH_PAGE_BYTES?ent.bytes:MESH_PAGE_BYTES;
        if(inl_dp[tp]!=MESH_PAGE_NONE){
          const int fits=(inl_src[tp][0]==sp || inl_src[tp][1]==sp ||
                          inl_src[tp][0]==MESH_PAGE_NONE ||
                          (inl_src[tp][1]==MESH_PAGE_NONE && nb<=MESH_TAIL_SLOT1_MAX));
          if(!fits || inl_n[tp]>=MESH_TAIL_MAX){
            mesh_tail_at(M,inl_dp[tp])->count=inl_n[tp];
            port_enqueue_desc(l,sw,tp,MESH_CLS_INLINE,inl_dp[tp],0u);
            MESH_SWITCH_COUNT(sw,l->index,groups,1);
            inl_dp[tp]=MESH_PAGE_NONE;inl_n[tp]=0;
            inl_src[tp][0]=inl_src[tp][1]=MESH_PAGE_NONE;
          }
        }
        if(inl_dp[tp]==MESH_PAGE_NONE){
          inl_dp[tp]=mesh_page_alloc_run(M,2u,poller_seed(l));
          if(inl_dp[tp]==MESH_PAGE_NONE){mesh_page_release(M,ent.src_page);MESH_SWITCH_COUNT(sw,l->index,lost,1);continue;}
          struct mesh_wire_tail *ot=mesh_tail_at(M,inl_dp[tp]);
          ot->origin_node=M->node;ot->from_node=M->node;
          inl_n[tp]=0;
          inl_src[tp][0]=inl_src[tp][1]=MESH_PAGE_NONE;
        }
        uint32_t sl=0;
        if(inl_src[tp][0]==sp)sl=0u;
        else if(inl_src[tp][1]==sp)sl=1u;
        else if(inl_src[tp][0]==MESH_PAGE_NONE){
          sl=0u;inl_src[tp][0]=sp;
          if(nb)memcpy(mesh_page_at(M,inl_dp[tp]),mesh_page_at(M,sp),nb);
        }else{
          sl=1u;inl_src[tp][1]=sp;
          if(nb)memcpy(mesh_page_at(M,inl_dp[tp]+1u),mesh_page_at(M,sp),nb);
        }
        mesh_page_release(M,ent.src_page);
        struct mesh_wire_tail *ot=mesh_tail_at(M,inl_dp[tp]);
        ot->e[inl_n[tp]++]=(struct mesh_wire_entry){.dst_node=ent.dst_node,.dst_index=ent.dst_index,.bytes=ent.bytes,
                                                    .flags=MESH_WIRE_INLINE,.hops=0,.src_page=sl};
        MESH_SWITCH_COUNT(sw,l->index,forwards,1);
      }
      i+=g;
    }
    mesh_page_release(M,q);
  }
  for(uint32_t tp=0;tp<MESH_PORTS;tp++){
    if(bulk_dp[tp]!=MESH_PAGE_NONE){
      mesh_desc_at(M,bulk_dp[tp])->count=bulk_n[tp];
      port_enqueue_desc(l,sw,tp,MESH_CLS_BULK8,bulk_dp[tp],bulk_wrs[tp]);
      MESH_SWITCH_COUNT(sw,l->index,groups,1);
    }
    if(inl_dp[tp]!=MESH_PAGE_NONE){
      mesh_tail_at(M,inl_dp[tp])->count=inl_n[tp];
      port_enqueue_desc(l,sw,tp,MESH_CLS_INLINE,inl_dp[tp],0u);
      MESH_SWITCH_COUNT(sw,l->index,groups,1);
    }
  }
  return taken;
}

static int port_pass(struct link *l){
  static __thread uint32_t timing;
  struct mesh_switch *sw=switch_of();
  uint64_t line[MESH_PORTS];
  mesh_switch_ports_read(sw,line);
  if(mesh_port_state(line[l->index])!=CHILD_SERVING)return 0;
  const int timed=!(++timing&1023);
  uint64_t t[6]={0};
  if(timed)t[0]=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  const uint32_t served=l->plan,lanes=plan_lanes(served);
  uint32_t worked=(uint32_t)link_receive(l,served,line);
  if(timed)t[1]=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  for(uint32_t j=0;j<lanes;j++)if(served>>j&1)switch_receives(l,j);
  if(timed)t[2]=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  worked|=switch_takes(l,line);
  if(timed)t[3]=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  switch_sends(l,served);
  if(timed)t[4]=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  worked|=(uint32_t)link_send(l,served);
  if(!worked)mesh_switch_togo_step(sw,&togo);
  MESH_SWITCH_COUNT(sw,l->index,passes,1);
  if(timed){
    t[5]=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    MESH_SWITCH_COUNT(sw,l->index,pass_ns[0],t[5]-t[0]);MESH_SWITCH_COUNT(sw,l->index,pass_ns[1],1);
    for(uint32_t k=0;k<5;k++)MESH_SWITCH_COUNT(sw,l->index,part_ns[k],t[k+1]-t[k]);
  }
  struct mesh_lanes *counted=lanes_of(l,0);
  atomic_store_explicit(&counted->passes,atomic_load_explicit(&counted->passes,memory_order_relaxed)+1,memory_order_relaxed);
  return 1;
}
/* a child's queue's user events (EVFILT_USER, child_main): its poller's return, triggered once in the child's life
   (port_progress), and a provider's doorbell fault absorbed (fault_heard) */
enum { POLLER_RETURNED, FAULT_ABSORBED };
static void child_trigger(uint64_t which){
  struct kevent64_s event;EV_SET64(&event,which,EVFILT_USER,0,NOTE_TRIGGER,0,0,0,0);
  kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
}
/* design/transport.md section 14: a port's poller, the one thread that posts on its device's queue pairs and polls
   their completion queues: it passes without pause while its port's line serves (rubric M8: the queues read as fast
   as a hardware thread allows) and never parks.  Its child's main thread makes it at the end of the link's connect,
   once every verbs object of the child is made and connected and its line raised to serving (link_serve), and makes
   no verbs object after: the parent asks a child for its lanes once (a link lost after that ends the child), and the
   kernel releases the objects at the child's exit.  So no post crosses a make or a connect.  It returns at the first
   pass that finds its line leaving (port_pass) and says so to its child's main thread (POLLER_RETURNED), once, so a
   child that ends leaves no page in its hands: a pass's carries, landings and posts are each whole by its end, and
   the child exits only after that word (child_main) */
static void *port_progress(void *argument){
  struct link *l=argument;
  pthread_setname_np("mesh.rdma");
  /* fault_heard's handler on a stack of its own */
  stack_t alternate={.ss_sp=malloc(SIGSTKSZ),.ss_size=SIGSTKSZ};
  if(alternate.ss_sp)sigaltstack(&alternate,NULL);
  mesh_switch_home=l->index;   /* this thread posts its own port's lanes (mesh-switched.h mesh_switch_cables_toward) */
  while(port_pass(l)){}
  child_trigger(POLLER_RETURNED);
  return NULL;
}

/* ---- design/transport.md section 14: the parent's ports, each one state machine (mesh-bridge.h enum port_state) ----
   The events of a port's machine (port_step, its one transition function, on the event loop): its link come on, its
   interface attached, its peer interface arrived, or the port found (PORT_ON: a refused registration's retry
   cleared); an IPv6 address of its interface added or changed (PORT_ADDRESSED); its link off or its interface
   detached (PORT_OFF); a hello heard on it (PORT_HEARD: `code` the node, `data` its address); a control socket
   accepted for it (PORT_ACCEPTED: `code` the socket), its connect's end (PORT_DIALED: `code` its SO_ERROR), a message
   on its control socket (PORT_MESSAGE: `data` the message) and the socket's loss (PORT_GONE: `code` why); its peer
   interface terminating (PORT_TERMINATED); its child's answer to the command out (PORT_ANSWERED: `data` the child's
   CHILD_OPENED, CHILD_MADE or CHILD_CONNECTED, each taken only in the state that waits for it), its child's own halt
   (PORT_BROKEN: `code` its reason) and its exit, reaped (PORT_EXITED: `code` its wait status); its deadline passed
   (PORT_DUE); the bridge's stop
   (PORT_STOP) */
enum port_event { PORT_ON, PORT_ADDRESSED, PORT_OFF, PORT_HEARD, PORT_ACCEPTED, PORT_DIALED, PORT_MESSAGE, PORT_GONE,
                  PORT_TERMINATED, PORT_ANSWERED, PORT_BROKEN, PORT_EXITED, PORT_DUE, PORT_STOP };
static void port_step(struct port *port,enum port_event event,int64_t code,const void *data);
static void port_move(struct port *port,enum port_state state,uint64_t due,int64_t code);
/* design/transport.md: a message to the port's peer, queued behind what the socket has not taken yet (a node's record
   in place of its older one: port_coalesce); a send that fails, or a queue the process cannot grow, is the link's loss */
static void port_watch(struct port *port,int filter){
  struct kevent64_s event;EV_SET64(&event,(uint64_t)port->control,filter,EV_ADD|(filter==EVFILT_WRITE?EV_ONESHOT:0),0,0,port->index,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))port_step(port,PORT_GONE,errno,NULL);
}
static void port_flush(struct port *port){
  while(port->control>=0 && port->out_sent<port->out_length){
    ssize_t sent=send(port->control,port->out+port->out_sent,port->out_length-port->out_sent,0);
    if(sent<0 && (errno==EAGAIN || errno==EINTR)){
      if(!port->writing){port->writing=1;port_watch(port,EVFILT_WRITE);}
      return;
    }
    if(sent<=0){port_step(port,PORT_GONE,sent?errno:EIO,NULL);return;}
    port->out_sent+=(size_t)sent;
  }
  port->out_sent=port->out_length=0;
}
/* a RECORD whose node has a RECORD waiting wholly unsent in the port's queue takes that one's place where it is no older
   (RFC 2328 s13: a node's newest record is the one that counts), so a peer that reads nothing has one record a node in
   the queue; 1 where it did */
static int port_coalesce(struct port *port,const struct message *said,const void *payload,uint32_t bytes){
  struct mesh_record fresh,queued;
  if(said->kind!=RECORD || bytes!=sizeof fresh)return 0;
  memcpy(&fresh,payload,sizeof fresh);
  for(size_t at=0;at+sizeof(struct message)<=port->out_length;){
    struct message header;
    memcpy(&header,port->out+at,sizeof header);
    if(at>=port->out_sent && header.kind==RECORD && header.bytes==sizeof queued){
      memcpy(&queued,port->out+at+sizeof header,sizeof queued);
      if(queued.node==fresh.node){
        if(fresh.boot>queued.boot || (fresh.boot==queued.boot && fresh.sequence>=queued.sequence))
          memcpy(port->out+at+sizeof header,&fresh,sizeof fresh);
        return 1;
      }
    }
    at+=sizeof header+header.bytes;
  }
  return 0;
}
void port_say(struct port *port,struct message said,const void *payload,uint32_t bytes){
  if(port->control<0 || (said.kind!=HELLO && !port_greeted(port)) || port_coalesce(port,&said,payload,bytes))return;
  said.magic=XMAGIC;said.version=MESH_VERSION;said.node=M->node;said.boot=M->boot;said.port=port->index;said.bytes=bytes;
  said.lid=port->lid;memcpy(said.gid,port->gid,16);
  size_t length=port->out_length+sizeof said+bytes;
  if(length>port->out_room){
    unsigned char *grown=realloc(port->out,length*2);
    if(!grown){port_step(port,PORT_GONE,ENOMEM,NULL);return;}
    port->out=grown;port->out_room=length*2;
  }
  memcpy(port->out+port->out_length,&said,sizeof said);
  if(bytes)memcpy(port->out+port->out_length+sizeof said,payload,bytes);
  port->out_length=length;
  port_flush(port);
}
/* design/transport.md: this node's record, its region, memory and model with its ports and health */
char model[MESH_MODEL_BYTES];
/* design/node-addressed-transfers.md: this node's reach as its record tells it (mesh-records.c record_changed): an
   entry a cable whose child serves a stream (its port's line CHILD_SERVING), the peer node its line names and that
   cable's port bandwidth, at most `room`; their count.  mesh-dataflow.c mesh_join counts a pair's entries as its
   parallel cables (the fewer of the two ends'), so a pair of three cables is three links of the map, never one link
   at their summed bandwidth.  The entries name nodes
   and bandwidths, never ports: which cable carries a page is the switch's, as it moves (mesh-switched.h) */
uint32_t record_reach(struct mesh_reach *reach,uint32_t room){
  uint32_t count=0;
  for(uint32_t p=0;p<port_count && count<room;p++){
    const uint64_t line=atomic_load(&switch_of()->port[ports[p].index]);
    if(mesh_port_state(line)==CHILD_SERVING && mesh_port_peer(line) && plan_streams(mesh_port_plan(line)))
      reach[count++]=(struct mesh_reach){.node=mesh_port_peer(line),.padding=1,.bandwidth=ports[p].bandwidth};
  }
  return count;
}
/* a port's link as the port table tells it; a change is a new record */
static void link_set(struct port *port,uint32_t peer,uint32_t phase,int64_t code){
  struct mesh_link_info *link=&mesh_links(M)[port->index];
  int changed=atomic_load(&link->peer)!=peer || atomic_load(&link->phase)!=phase;
  atomic_store_explicit(&link->peer,peer,memory_order_relaxed);atomic_store_explicit(&link->code,code,memory_order_relaxed);
  atomic_store_explicit(&link->phase,phase,memory_order_release);
  if(changed)record_changed();
}

/* design/transport.md section 14: a link's verbs calls, made on its child's main thread on its parent's command, once
   in the child's life ("A process a port") and before its poller exists, so no queue pair of the device is posted on
   while one is made or connected (port_progress): the lanes made (link_make: the port's own address read again, then a
   UC queue pair in INIT for each lane of the link's plan, d6dceac's verbs_up, each lane's share of the completion
   queues the plan's lanes dividing them: design/HETEROGENEOUS-LINKS.md); or connected to the peer's (link_connect,
   the plan's lanes alone: RTR, then RTS, each lane's number set, every lane's progress and depth made new), from the
   peer's numbers as the command gave them.  No queue pair is moved to another state, destroyed or forgotten after it:
   the in-place RESET of 127 panicked the M5 in AppleThunderboltRDMA on its first run (2026-10-19,
   design/RDMA-KERNEL-RECOVERY.md), and the kernel releases every object at the child's exit, its one end.  Between
   its steps the child's IOKit notifications are taken (link_lapsed: a termination heard halts the child, B10, B9b),
   and a step whose child has halted is not made (ECANCELED) */
static void iokit_heard(void);
static int link_lapsed(const struct link *l){
  iokit_heard();
  return mesh_port_state(atomic_load(&switch_of()->port[l->index]))==CHILD_LEAVING?ECANCELED:0;
}
static int link_make(struct link *l){
  struct mesh_verbs *p=&l->provider;
  struct ibv_port_attr attributes;
  int error=0;
  /* design/transport.md: the port's own address (its attributes and GID) read again at the make, as its opening read
     it: a peer's restart can move the port's GID table, and LINKs naming the address of the port's opening failed every
     lane's connection at the peer with EIO (2026-10-19, after the Mini's power loss) */
  if(ibv_query_port(l->device.context,1,&attributes))return errno?errno:EIO;
  l->device.port=attributes;
  if((error=device_gid(&l->device,l->name)) || (error=link_lapsed(l)) || (error=pair_up(p,plan_lanes(l->plan))))return error;
  l->share=(struct frames){(uint64_t)MIN(p->completion->cqe,p->sent->cqe)/MAX(plan_lanes(l->plan),1u)};
  return 0;
}
static void lane_reset(struct link *l,uint32_t k);
/* design/transport.md section 14: stream s's identity, alike at both ends: FNV-1a over its first lane's two ends, each
   its port's GID and its queue pair's number, the lesser end first; never 0 */
static uint64_t stream_wire(const struct link *l,const struct child_message *heard,uint32_t s){
  const uint32_t k=2*s;
  uint8_t end[2][20];
  memcpy(end[0],l->gid,16);memcpy(end[0]+16,&l->qpn[k],4);
  memcpy(end[1],heard->gid,16);memcpy(end[1]+16,&heard->peer[k][0],4);
  const int first=memcmp(end[0],end[1],20)>0;
  uint64_t h=UINT64_C(14695981039346656037);
  for(int e=0;e<2;e++)
    for(int i=0;i<20;i++){h^=end[e^first][i];h*=UINT64_C(1099511628211);}
  return h?h:1;
}
static int link_connect(struct link *l,const struct child_message *heard){
  struct mesh_verbs *p=&l->provider;
  const uint32_t plan=l->plan;
  int error=0;
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1)error=queue_receive(p,(int)k,heard->peer[k],heard->lid,heard->gid);
  if(!error)error=link_lapsed(l);
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1)error=queue_send(p,(int)k);
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)l->qpn[k]=plan>>k&1?p->queues[k].pair->qp_num:UINT32_MAX;
  if(!error)error=link_lapsed(l);
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1)lane_reset(l,k);
  for(uint32_t s=0;s<plan_streams(plan) && !error;s++)l->wire[s]=stream_wire(l,heard,s);
  return error;
}
/* design/transport.md section 14, "A process a port": the link's lane records opened as its lanes connect, by the
   child before its poller exists: each lane's requests posted and completed each way 0 (its queue pairs are new),
   this child their poster (`child`, `child_start`), the peer node, the link's bandwidth and identity told; each lane's
   in-flight ring's positions 0 (its last child's end gave back what was in it: switch_child_free); then the port's
   inbound rings emptied of what was pushed onto them after its last child's end (mesh-switched.h
   mesh_switch_inbound_drop: pages for the port's last link, each counted `lost` in the port's counters), and the
   port's line moved from its lanes made to serving toward the peer with the link's plan (one release of them all: a
   draw that reads it serving reads them emptied).  1 where the line moved, 0 where it was not at its lanes made (a
   halt came first: it is leaving) */
static int lanes_open(struct link *l,uint32_t node,uint64_t bandwidth){
  for(uint32_t s=0;s<plan_streams(l->plan);s++){
    struct mesh_lanes *x=lanes_of(l,s);
    _Atomic uint64_t *words[]={&x->posts[0].posted[0],&x->posts[0].posted[1],&x->posts[0].done[0],&x->posts[0].done[1],
                               &x->posts[1].posted[0],&x->posts[1].posted[1],&x->posts[1].done[0],&x->posts[1].done[1]};
    for(size_t i=0;i<sizeof words/sizeof *words;i++)atomic_store_explicit(words[i],0,memory_order_relaxed);
    x->child_start=mesh_started(getpid());
    atomic_store_explicit(&x->child,(uint32_t)getpid(),memory_order_release);
    atomic_store_explicit(&x->peer,node,memory_order_relaxed);
    atomic_store_explicit(&x->bandwidth,bandwidth,memory_order_relaxed);
    atomic_store_explicit(&x->wire,l->wire[0],memory_order_relaxed);
  }
  struct mesh_switch *sw=switch_of();
  for(uint32_t j=0;j<MESH_LINK_LANES;j++){
    struct mesh_switch_lane *x=sw->lane+switch_lane(l->index,j);
    atomic_store_explicit(&x->posted,0,memory_order_relaxed);atomic_store_explicit(&x->completed,0,memory_order_relaxed);
    atomic_store_explicit(&x->rx_posted,0,memory_order_relaxed);atomic_store_explicit(&x->rx_done,0,memory_order_relaxed);
    atomic_store_explicit(&x->rx_desc,MESH_PAGE_NONE,memory_order_relaxed);atomic_store_explicit(&x->rx_rem,0,memory_order_relaxed);
    l->rx_pos[j]=0;
  }
  memset(l->tx_tail,0,sizeof l->tx_tail);
  for(uint32_t p=0;p<MESH_PORTS;p++){l->relay_bulk_dp[p]=MESH_PAGE_NONE;l->relay_bulk_n[p]=0;l->relay_bulk_wrs[p]=0;}
  mesh_switch_port_drop(M,sw,l->index,l->index);
  return mesh_port_move(sw,l->index,1u<<CHILD_LANES_MADE,CHILD_SERVING,node,l->plan);
}
/* design/transport.md section 14: a connected link served: its lane records opened and its line raised to serving
   (lanes_open), then its poller made (port_progress, at the progress threads' class), the one thread that posts on the
   device's queue pairs from here on: pthread_create orders every make, connect and lane state before its first post.
   The line is raised here, before the poller exists and before the child answers its connect, so the parent's record
   of the link (record_reach) and its node table (switch_view), each made as it takes the answer, read it serving.  A
   plan of no lane has nothing to poll: its line serves and no poller is made.  `polling` 1 from the poller's making to
   its return's event (END_POLLED), the main thread's.  0, or why the link does not serve (ECANCELED: a halt came first) */
static int polling;
static int link_serve(struct link *l,uint32_t node,uint64_t bandwidth){
  if(!lanes_open(l,node,bandwidth))return ECANCELED;
  if(!l->plan)return 0;
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_set_qos_class_np(&attributes,QOS_CLASS_USER_INTERACTIVE,0);
  pthread_t created;
  const int error=pthread_create(&created,&attributes,port_progress,l);
  pthread_attr_destroy(&attributes);
  if(!error)polling=1;
  return error;
}
/* a LINK telling the peer every lane of this end's pairing, the lanes of its plan (`count` theirs): each one's queue pair
   number and first sequence number, as its child told them (CHILD_MADE), 0 for a lane the plan leaves out */
static void link_tell(struct port *port){
  struct message said={.kind=LINK,.count=plan_lanes(port->plan)};
  for(uint32_t k=0;k<MESH_LINK_LANES;k++)if(port->plan>>k&1){said.pair[k][0]=port->qpn[k];said.pair[k][1]=(port->psn+k)&0xffffff;}
  port_say(port,said,NULL,0);
}
/* a lane's progress made new at its connect: nothing posted, its receives its queues' share, and its depth (the
   SENDs it has outstanding at most) its queues' share of its send capacity, over its frames, the in-flight ring's size
   at most: each SEND on the lane covers its frames, so a post past the depth would pass the queue's capacity */
static void lane_reset(struct link *l,uint32_t k){
  const struct frames room=frames_min(l->share,(struct frames){l->provider.queues[k].receive_capacity}),
                      sends=frames_min((struct frames){l->provider.queues[k].send_capacity},(struct frames){MAX(l->share.n,1)});
  l->lane[k]=(struct lane_state){.frame_capacity=room.n,.depth=MIN(sends.n/mesh_switch_lane_frames(k),(uint64_t)MESH_SWITCH_FLIGHT)};
}
/* ---- design/PAGE-TABLE-SPEC-20261122.md: the switch, the parent's part -----------------------------------------
   The parent carves the node's pool once and makes its switch at its start, keeps the switch's view of the fabric and
   the node table (a node the map has, live while it does) as its event loop writes the map (switch_view); each port
   child's poller makes its own cost-to-go (mesh-switched.h mesh_switch_togo_step), and the parent has no thread but
   its event loop unless a theta is loaded (switch_watch).  A port's child exited, what it had is given back here
   (switch_child_free): nothing is sent again, and each page it had is counted `lost` */
/* the node table's entry of `node`, made (generation 1) as the map first has the node; MESH_ABSENT where every entry
   has another node (the event loop's alone: switch_view) */
static uint32_t node_slot(uint32_t node){
  uint32_t n=0,vacant=MESH_ABSENT;
  for(;n<MESH_NODES && atomic_load(&mesh_node_at(M,n)->node)!=node;n++)if(vacant==MESH_ABSENT && !atomic_load(&mesh_node_at(M,n)->node))vacant=n;
  if(n==MESH_NODES && (n=vacant)==MESH_ABSENT){say("bridge: no node table entry for node %u (mesh.h MESH_NODES)\n",node);return MESH_ABSENT;}
  struct mesh_node *x=mesh_node_at(M,n);
  if(!atomic_load(&x->generation))atomic_store(&x->generation,1);
  atomic_store_explicit(&x->node,node,memory_order_release);
  return n;
}
/* the frames the switch's cables have queued and not completed (struct hdr `busy`) */
uint64_t lanes_busy(void){
  const struct mesh_switch *sw=mesh_switch_of(M);
  uint64_t busy=0;
  if(sw->magic!=MESH_SWITCH_MAGIC)return 0;
  for(uint32_t p=0;p<MESH_PORTS;p++)busy+=atomic_load(&sw->cable[p].queued)/MESH_FRAME;
  return busy;
}
uint64_t pool_bytes(void){
  return (uint64_t)mesh_flat_pages(M)*MESH_PAGE_BYTES;
}
static void switch_make(void){
  struct mesh_switch *sw=mesh_switch_of(M);
  mesh_switch_init(sw);
  say("bridge: flat-page switch made over %u pages (%llu bytes)\n",mesh_flat_pages(M),(unsigned long long)pool_bytes());
}
/* the view as the records stand: every record's node, and each node pair the records join (each end's entries toward
   the other, the fewer of the two), its cables and their bandwidth summed (bytes a ns; 0 where neither end's record
   reports one), at the hop latency the fabric's fits measure (6 us): each link's nominal terms, its cost until its
   sender's measures of it are heard (mesh-switched.h struct mesh_switch_cable, struct mesh_switch_line); every node of
   it an entry of the node table (the key its links' lines are kept under: struct mesh_switch_view `slot`), live (the
   bandwidth of this node's own cables toward it), every other entry not.  Made by the event loop each time it
   publishes the map, after the publish (mesh-records.c map_make): the map's one writer, so it reads the published
   copy and the ports as they stand, with nothing read again, and a map's change is drawable at once */
#define SWITCH_LATENCY_NS 6000.0
void switch_view(void){
  const struct mesh_map_copy *map=mesh_map_published(M);
  const uint64_t count=mesh_map_nodes(map);
  const struct mesh_record *r=map->record;
  uint32_t nodes[MESH_SWITCH_N],slots[MESH_SWITCH_N],n=0;struct mesh_switch_fabric_link links[MESH_SWITCH_LINKS];uint32_t k=0;
  for(uint64_t i=0;i<count && n<MESH_SWITCH_N;i++){
    nodes[n]=r[i].node;
    slots[n]=r[i].node==M->node?MESH_NODES:node_slot(r[i].node);
    n++;
  }
  for(uint64_t a=0;a<count;a++)for(uint64_t b=0;b<count;b++){
    if(a==b || k==MESH_SWITCH_LINKS)continue;
    uint32_t ab=0,ba=0;uint64_t bandwidth=0;
    for(uint32_t e=0;e<r[a].reached && e<MESH_STREAMS;e++)if(r[a].reach[e].node==r[b].node){ab++;bandwidth+=r[a].reach[e].bandwidth;}
    for(uint32_t e=0;e<r[b].reached && e<MESH_STREAMS;e++)if(r[b].reach[e].node==r[a].node)ba++;
    const uint32_t cables=ab<ba?ab:ba;
    if(!cables)continue;
    links[k++]=(struct mesh_switch_fabric_link){r[a].node,r[b].node,cables,(double)bandwidth/8e9*cables/ab,SWITCH_LATENCY_NS};
  }
  mesh_switch_view_set(mesh_switch_of(M),M->node,nodes,slots,n,links,k);
  uint32_t live=0;
  for(uint32_t i=0;i<n;i++){
    const uint32_t s=slots[i];
    if(s>=MESH_NODES)continue;
    struct mesh_node *x=mesh_node_at(M,s);
    uint64_t bandwidth=0;
    for(uint32_t p=0;p<port_count;p++){
      const uint64_t line=atomic_load(&mesh_switch_of(M)->port[ports[p].index]);
      if(mesh_port_state(line)==CHILD_SERVING && mesh_port_peer(line)==nodes[i])bandwidth+=ports[p].bandwidth;
    }
    atomic_store(&x->bandwidth,bandwidth);
    atomic_store_explicit(&x->live,1,memory_order_release);
    live|=1u<<s;
  }
  for(uint32_t s=0;s<MESH_NODES;s++)if(!(live>>s&1))atomic_store_explicit(&mesh_node_at(M,s)->live,0,memory_order_release);
}
/* design/PAGE-TABLE-SPEC-20261122.md section 5: a port's child exited, what it had given back: its lanes' in-flight
   rings (each SEND posted and not completed) and its port's inbound rings (each page pushed toward it and not posted)
   emptied, each page's use dropped, counted `lost`, and the pages posted as its lanes' receives given back, then its
   port's line made new (STARTED, no peer, no plan) for its next child: no process but this one writes it now.  Its
   lanes are no draw's already (port_leaving).  A child that ended between passes (every end it was asked for or chose:
   its exit waits for its poller's return, child_main) held no page outside these; one that ended inside a pass (a
   crash, or its poller still inside a provider call at its end's bound) may have held one mid-carry,
   mid-landing or between an inbound ring and an in-flight ring, which is not found and stays in use (the operator's
   O4: accepted, no record) */
static void switch_child_free(struct port *port){
  struct mesh_switch *sw=mesh_switch_of(M);
  for(uint32_t j=0;j<2*MESH_LINK_STREAMS;j++)mesh_switch_flight_drop(M,sw,switch_lane(port->index,j),MESH_SWITCH_PARENT);
  mesh_switch_port_drop(M,sw,port->index,MESH_SWITCH_PARENT);
  mesh_switch_receives_drop(M,sw,switch_lane(port->index,0),2*MESH_LINK_STREAMS);
  atomic_store(&sw->port[port->index],0);
}

/* the control socket: non-blocking, its sends queued (port_say), its HELLO stating the queue pairs the port's device
   has for a link (its budget: link_plan); its peer's loss is the kernel's to tell (its end closed, a reset, the link's
   going off) */
static void port_greet(struct port *port){
  int on=1,flags=fcntl(port->control,F_GETFL);
  if(flags<0 || fcntl(port->control,F_SETFL,flags|O_NONBLOCK) || setsockopt(port->control,SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on) ||
     setsockopt(port->control,IPPROTO_TCP,TCP_NODELAY,&on,sizeof on)){
    port_step(port,PORT_GONE,errno?errno:EIO,NULL);return;
  }
  port_watch(port,EVFILT_READ);
  port_say(port,(struct message){.kind=HELLO,.count=port->budget},NULL,0);
}
/* design/transport.md: the port a link-local peer reached, by its interface; none for a scope of 0 (no
   interface: a LAN address, or the index of an interface that no longer exists) */
static struct port *port_at(uint32_t interface){
  for(uint32_t p=0;interface && p<port_count;p++)if(ports[p].interface==interface)return ports+p;
  return NULL;
}
/* design/transport.md: the hellos heard, each on the port of its interface (PORT_HEARD: the lower node dials the
   higher's listener on the address it heard from, the higher answers so the lower hears it at once; a port not open
   is to be opened, so a port whose child's open or link failed pairs again on hearing its peer, never on a clock) */
static void hello_heard(void){
  struct message heard;struct sockaddr_in6 from;socklen_t length=sizeof from;
  while(recvfrom(hello,&heard,sizeof heard,0,(struct sockaddr *)&from,&length)==(ssize_t)sizeof heard){
    struct port *port=port_at(from.sin6_scope_id);
    length=sizeof from;
    if(port && heard.magic==XMAGIC && heard.kind==HELLO && heard.node!=M->node)port_step(port,PORT_HEARD,heard.node,&from);
  }
}
/* a control socket's messages, each a header and its `bytes`, each whole one taken: a RECORD, once the port is
   greeted, to the map (mesh-records.c record_heard; one before is let be), every other to the port's machine
   (PORT_MESSAGE); a header not the mesh's, or past 64 MiB, the socket's end and a failed read are its loss
   (PORT_GONE) */
static void control_heard(struct port *port){
  for(;;){
    size_t need=sizeof(struct message);
    if(port->in_length>=need){
      const struct message *header=(const void *)port->in;
      if(header->magic!=XMAGIC || header->bytes>(UINT32_C(64)<<20)){port_step(port,PORT_GONE,EPROTO,NULL);return;}
      need+=header->bytes;
    }
    if(need>port->in_room){
      unsigned char *grown=realloc(port->in,need);
      if(!grown){port_step(port,PORT_GONE,ENOMEM,NULL);return;}
      port->in=grown;port->in_room=need;
    }
    if(port->in_length<need){
      ssize_t got=recv(port->control,port->in+port->in_length,need-port->in_length,0);
      if(got<0 && (errno==EAGAIN || errno==EINTR))return;
      if(got<=0){port_step(port,PORT_GONE,got?errno:ENETRESET,NULL);return;}
      port->in_length+=(size_t)got;
      if(port->in_length<need || need==sizeof(struct message))continue;
    }
    struct message heard;memcpy(&heard,port->in,sizeof heard);
    port->in_length=0;
    if(heard.kind==RECORD){if(port_greeted(port))record_heard(port,port->in+sizeof heard,heard.bytes);}
    else port_step(port,PORT_MESSAGE,0,&heard);
    if(port->control<0)return;
  }
}
/* design/transport.md: a Thunderbolt RDMA interface of the node, `device` (the provider's device of the same name:
   rdma_<interface>), made a port where none is yet, MESH_PORTS at most, as IOKit first matches it (interfaces_heard,
   on the event loop): CLOSED and to be opened (PORT_ON) once the node runs and its quiet window has ended, which
   starts as the exit's window stands (`exit_at`: a termination heard before it was made, B9b).  The parent makes no
   verbs call: its child lists the devices itself as it opens its own (device_up) */
static void port_found(const char *device){
  for(uint32_t p=0;p<port_count;p++)if(!strcmp(ports[p].device,device))return;
  if(port_count==MESH_PORTS)return;
  struct port *port=ports+port_count;
  *port=(struct port){.state=PORT_CLOSED,.index=port_count,.control=-1,.child_fd=-1};
  snprintf(port->device,sizeof port->device,"%s",device);snprintf(port->name,sizeof port->name,"%s",device+5);
  /* the interface as the kernel's events name it, its name read once here into typed values (addresses_heard) */
  if(!strncmp(port->name,"en",2)){port->family=APPLE_IF_FAM_ETHERNET;port->unit=(uint32_t)strtoul(port->name+2,NULL,10);}
  port_count++;
  quiet_until(&port->quiet_at,exit_at);
  port_move(port,PORT_CLOSED,0,0);
  port_step(port,PORT_ON,0,NULL);
}
/* design/transport.md: a port's link on or its interface attached (PORT_ON), or its link off or its interface detached
   (PORT_OFF: the link lost, its lanes left as they are, design/transport.md B6 (b), and the port down until its link
   comes on or its interface attaches; at the stop, its window and its child's moved, B9) */
static void network_heard(void){
  union mesh_network_event event;
  while(recv(network,event.bytes,sizeof event,0)>=(ssize_t)(KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data))){
    const struct net_event_data *interface=(const void *)(event.bytes+KEV_MSG_HEADER_SIZE);
    const uint32_t code=event.header.event_code;
    char name[IFNAMSIZ+16];
    snprintf(name,sizeof name,"%.*s%u",(int)sizeof interface->if_name,interface->if_name,interface->if_unit);
    for(uint32_t p=0;p<port_count;p++)if(!strcmp(name,ports[p].name)){
      if(code==KEV_DL_LINK_ON || code==KEV_DL_IF_ATTACHED)port_step(ports+p,PORT_ON,0,NULL);
      else if(code==KEV_DL_LINK_OFF || code==KEV_DL_IF_DETACHED)port_step(ports+p,PORT_OFF,ENETDOWN,NULL);
    }
  }
}
/* design/transport.md: an IPv6 address of a port's interface added, changed or deleted (the kernel's KEV_INET6_SUBCLASS
   events on `addresses`: a link-local address bin/mesh-fabric-init.sh puts in place, or its duplicate address detection
   done): the hellos' group joined on the interface again (a join while it had no IPv6 address took nothing) and an
   open port's hello said at this turn (PORT_ADDRESSED), so a hello that left before the address could carry it is said
   again as the address comes, not a pace (MESH_PAUSE_NS) later.  The port is the event's interface family and unit,
   the port's own read once (struct port `family`, `unit`): no name is built or compared per event (design/rubrics.md,
   Strings are messages).  Another Ethernet-family interface of the same unit (bridge0, awdl0, anpi<N>) is taken as
   the port's, and its event says the port's hello early, as the pace would: a hello on an open port with no peer is
   never wrong */
static void addresses_heard(void){
  union { struct kern_event_msg header; unsigned char bytes[512]; } event;
  while(recv(addresses,event.bytes,sizeof event,0)>=(ssize_t)(KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data))){
    const struct net_event_data *interface=(const void *)(event.bytes+KEV_MSG_HEADER_SIZE);
    for(uint32_t p=0;p<port_count;p++)if(interface->if_family==ports[p].family && interface->if_unit==ports[p].unit){
      struct ipv6_mreq group={.ipv6mr_multiaddr=everyone,.ipv6mr_interface=ports[p].interface};
      if(ports[p].interface)setsockopt(hello,IPPROTO_IPV6,IPV6_JOIN_GROUP,&group,sizeof group);
      port_step(ports+p,PORT_ADDRESSED,0,NULL);
    }
  }
}

/* ---- design/transport.md: IOKit, on each process's own queue ---------------------------------------------------
   Each process's notifications (the parent's and a child's) are on one IONotificationPort (`iokit`) whose Mach port is
   a filter of the process's queue (EVFILT_MACHPORT, TAG_IOKIT): each message it holds is taken without waiting and
   handed to IOKit (IODispatchCalloutFromMessage), which calls its callback on the thread that takes it, the process's
   main thread (the parent's event loop, a child's main thread), so no thread hears IOKit for another and nothing is
   handed on (iokit_heard).  Armed at the parent's SETTLING (iokit_watch) and as a child starts (child_iokit) */
#define TAG_IOKIT (UINT64_C(1)<<56)
static IONotificationPortRef iokit;
/* every message the notification port holds, each taken without waiting and dispatched, until none is left */
static void iokit_heard(void){
  const mach_port_t port=IONotificationPortGetMachPort(iokit);
  union { mach_msg_header_t header; uint8_t bytes[8192]; } message;
  while(mach_msg(&message.header,MACH_RCV_MSG|MACH_RCV_TIMEOUT,0,sizeof message,port,0,MACH_PORT_NULL)==MACH_MSG_SUCCESS)
    IODispatchCalloutFromMessage(NULL,&message.header,iokit);
}
/* the name of the interface (an AppleThunderboltRDMAInterface, rdma_<port>) of a terminating or arriving peer
   interface, by mesh-fabric-init.sh's rule (the two carry the same "RID"), in `name`; 0 where none is found (the
   property removed with the terminating service), 1 where one is */
static int peer_named(io_service_t peer,io_name_t name){
  CFTypeRef rid=IORegistryEntryCreateCFProperty(peer,CFSTR("RID"),kCFAllocatorDefault,0);
  io_iterator_t interfaces=IO_OBJECT_NULL;
  int found=0;
  if(rid && IOServiceGetMatchingServices(kIOMainPortDefault,IOServiceMatching("AppleThunderboltRDMAInterface"),&interfaces)==KERN_SUCCESS){
    for(io_service_t interface;(interface=IOIteratorNext(interfaces));IOObjectRelease(interface)){
      CFTypeRef its=found?NULL:IORegistryEntryCreateCFProperty(interface,CFSTR("RID"),kCFAllocatorDefault,0);
      if(its && CFEqual(its,rid) && IORegistryEntryGetName(interface,name)==KERN_SUCCESS)found=1;
      if(its)CFRelease(its);
    }
    IOObjectRelease(interfaces);
  }
  if(rid)CFRelease(rid);
  return found;
}
/* a child's: its peer interfaces' terminations (B9b: EXIT-RECLAIM.md P2, and P3 as a window in place of
   IOServiceWaitQuiet; IOKit tells one in its first phase, kIOTerminatedNotification; the iterator drained first, which
   arms it again), on its main thread: for each terminated service the child's window (`exit_at`) moved to
   MESH_PLATFORM_NS from now (no make, connect or exit inside the termination, and the child's end waits for it:
   child_main) before any property is read; then its port, where the service is its own port's peer interface
   (peer_named), halted (child_halt, ETIMEDOUT: its poller posts nothing from its next
   pass, the child takes no command again, and its main thread tells its parent); each released.  Heard as the child's
   loop wakes and between the steps of its commands (link_lapsed), so a termination during a verbs call is heard as
   the call returns.  A child whose listener is not yet armed as the termination comes (it has made no queue pair
   yet) misses it */
static void peer_terminated(void *unused,io_iterator_t iterator){
  (void)unused;
  io_service_t services[MESH_PORTS];
  uint32_t count=0;
  for(io_service_t service;(service=IOIteratorNext(iterator));){
    quiet_until(&exit_at,clock_ns()+MESH_PLATFORM_NS);
    if(count<MESH_PORTS)services[count++]=service;
    else IOObjectRelease(service);
  }
  for(uint32_t k=0;k<count;k++){
    io_name_t name;
    if(peer_named(services[k],name) && !strcmp(name,child_link.device.name))child_halt(&child_link,ETIMEDOUT);
    IOObjectRelease(services[k]);
  }
}
/* a child's listener, armed before it opens its device: its notification port a filter of its queue, its iterator
   drained once (which arms it).  A child whose notification is not armed ends: its port's poller would hear a
   termination only from its parent's line (terminations_heard), and it has made no queue pair yet, so its parent loses
   the link as it reaps it (PORT_EXITED) and the port's next child arms its own */
static void child_iokit(void){
  io_iterator_t iterator=IO_OBJECT_NULL;
  kern_return_t error=KERN_FAILURE;
  if((iokit=IONotificationPortCreate(kIOMainPortDefault))){
    struct kevent64_s event;
    EV_SET64(&event,IONotificationPortGetMachPort(iokit),EVFILT_MACHPORT,EV_ADD,0,0,TAG_IOKIT,0,0);
    error=kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)?KERN_FAILURE:
      IOServiceAddMatchingNotification(iokit,kIOTerminatedNotification,IOServiceMatching("AppleThunderboltRDMAPeerInterface"),
                                       peer_terminated,NULL,&iterator);
  }
  if(error)die("child: its port's peer-interface terminations not heard: this child ends");
  peer_terminated(NULL,iterator);
}
/* design/transport.md (EXIT-RECLAIM.md P4): the provider's doorbell fault absorbed, once.  A SIGSEGV or SIGBUS whose pc
   lies in the provider's text (`provider`, read at the start) has one anonymous page mapped at the faulting page by
   hint, never MAP_FIXED, so nothing mapped is replaced.  Where it got that page (no mapping was there: the doorbell the
   driver unmapped in a termination), the store lands in that page, no doorbell rings, the child is halted (child_halt,
   ENETDOWN: after an absorbed doorbell store its queue pair's state is unknown, so nothing is posted on any of
   them again, and the child ends; its parent loses the link on the report, ENETDOWN, and the peer at the control
   socket's end), and its main thread is woken to tell it (FAULT_ABSORBED: its poller may be inside a provider call
   that never returns).  The thread that faults is the poller or, before the poller exists, the main thread in a
   setup call: the two never run provider code at once, so no other thread maps the page first.  Any other fault ends
   the child at once (_exit, its status 128 and the signal's number, so no crash report is written to storage).  On
   the poller's alternate stack (port_progress); mmap, munmap, kevent64, _exit and atomics alone */
static uintptr_t provider[2];
static void fault_heard(int number,siginfo_t *info,void *context){
  const uintptr_t pc=__darwin_arm_thread_state64_get_pc(((ucontext_t *)context)->uc_mcontext->__ss);
  const uintptr_t page=trunc_page((uintptr_t)info->si_addr);
  if(pc>=provider[0] && pc<provider[1]){
    void *got=mmap((void *)page,vm_page_size,PROT_READ|PROT_WRITE,MAP_ANON|MAP_PRIVATE,-1,0);
    if(got==(void *)page){child_halt(&child_link,ENETDOWN);child_trigger(FAULT_ABSORBED);return;}
    if(got!=MAP_FAILED)munmap(got,vm_page_size);
  }
  _exit(128+number);
}
/* the provider's text read and the faults' handler installed, by a child before it opens its device */
static void faults_install(void){
  for(uint32_t i=0;i<_dyld_image_count();i++){
    const char *image=_dyld_get_image_name(i);
    unsigned long size=0;
    const uint8_t *text=image && strstr(image,"/libthunderboltrdma.")?
      getsegmentdata((const void *)_dyld_get_image_header(i),"__TEXT",&size):NULL;
    if(text){provider[0]=(uintptr_t)text;provider[1]=(uintptr_t)text+size;}
  }
  struct sigaction faults={.sa_sigaction=fault_heard,.sa_flags=SA_SIGINFO|SA_ONSTACK};
  sigemptyset(&faults.sa_mask);
  if(sigaction(SIGSEGV,&faults,NULL) || sigaction(SIGBUS,&faults,NULL))die("bridge faults");
}

/* ---- design/transport.md: the parent's IOKit, on the event loop ----------------------------------------------------
   Armed at SETTLING's entry (iokit_watch): the peer interfaces' terminations, the interfaces (the ports) and the peer
   interfaces' arrivals, each callback on the loop itself */
/* SETTLING's: the service plane root's busy state's changes (kIOBusyInterest), released as the node runs */
static io_object_t busy_notification;
/* design/transport.md B9b (EXIT-RECLAIM.md P2, and P3 as a window in place of IOServiceWaitQuiet): the exit's window and
   every port's moved to `until` at least (a port made meanwhile takes `exit_at`'s: port_found) */
static void quiet_all(uint64_t until){
  quiet_until(&exit_at,until);
  for(uint32_t p=0;p<port_count;p++)quiet_until(&ports[p].quiet_at,until);
}
/* the port of a peer interface (peer_named: the port whose device is the interface of the same "RID"), NULL where none
   is found (no port of that interface yet) */
static struct port *peer_port(io_service_t peer){
  io_name_t name;
  if(!peer_named(peer,name))return NULL;
  for(uint32_t p=0;p<port_count;p++)if(!strcmp(ports[p].device,name))return ports+p;
  return NULL;
}
/* design/transport.md B9b: Thunderbolt RDMA peer interfaces terminating (the iterator drained, which arms it again):
   the exit's window and every port's moved to MESH_PLATFORM_NS from now (no port job and no exit inside the
   termination, whichever port's it is) before any property is read, and logged; then each port one maps to
   (PORT_TERMINATED, ETIMEDOUT: from its make on its line leaving, so its child's poller posts nothing more from its next
   pass, its child halted and its socket closed; before it, its socket closed; no verbs call), each released: one
   handoff from IOKit, whether or not the port has a child.  The port's child hears the same termination and halts
   itself, and its report of it (CHILD_BROKEN, ETIMEDOUT: child_heard) ends it */
static void terminations_heard(void *unused,io_iterator_t iterator){
  (void)unused;
  io_service_t services[MESH_PORTS];
  uint32_t count=0,heard=0;
  for(io_service_t service;(service=IOIteratorNext(iterator));heard++){
    quiet_all(clock_ns()+MESH_PLATFORM_NS);
    if(count<MESH_PORTS)services[count++]=service;
    else IOObjectRelease(service);
  }
  if(heard){
    const uint64_t at=clock_ns(),until=exit_at;
    say("bridge node %u: %u Thunderbolt RDMA peer interface%s terminating: no port job and no exit for %.1f s "
        "(design/transport.md)\n",node.id,heard,heard==1?"":"s",(double)(until>at?until-at:0)/1e9);
  }
  for(uint32_t k=0;k<count;k++){
    struct port *port=peer_port(services[k]);
    IOObjectRelease(services[k]);
    if(!port)continue;
    say("link %u (%s): its peer interface terminating: its link lost\n",port->index,port->name);
    port_step(port,PORT_TERMINATED,ETIMEDOUT,NULL);
  }
}
/* design/transport.md: the node's Thunderbolt RDMA interfaces (kIOFirstMatchNotification of
   AppleThunderboltRDMAInterface: each one there as it is armed, then each one that comes; the iterator drained, which
   arms it again), each one named rdma_<interface> a port (port_found), by the name peer_port finds a port by */
static void interfaces_heard(void *unused,io_iterator_t iterator){
  (void)unused;
  for(io_service_t service;(service=IOIteratorNext(iterator));IOObjectRelease(service)){
    io_name_t name;
    if(IORegistryEntryGetName(service,name)==KERN_SUCCESS && !strncmp(name,"rdma_",5))port_found(name);
  }
}
/* design/transport.md: Thunderbolt RDMA peer interfaces arrived (kIOFirstMatchNotification: a cable's far end come to
   RDMA; the iterator drained, which arms it again; after a termination IOKit told before it, so a peer interface that
   left and then came is lost, then opened): each port one maps to as at its link's coming on (PORT_ON: one not open is
   opened, one open says its hello at once), so its device is opened at the hardware's word that its far end is present and not only at the
   en interface's link, which can come on before it (bin/mesh-fabric-init.sh: link active, no RDMA peer); without it
   a port whose open met its device's peer not yet present (ENETDOWN at both ends, no hello sent) opened only at its
   link's next coming on or the next hello heard on it, which might never come.  An arrival at an interface that is no
   port yet needs nothing more: the port its interface's first match makes is to be opened */
static void arrivals_heard(void *unused,io_iterator_t iterator){
  (void)unused;
  for(io_service_t service;(service=IOIteratorNext(iterator));IOObjectRelease(service)){
    struct port *port=peer_port(service);
    if(port)port_step(port,PORT_ON,0,NULL);
  }
}
/* SETTLING's: the root's busy state changed (kIOMessageServiceBusyStateChange, its argument the state now, 0 quiet) */
static void busy_heard(void *unused,io_service_t service,uint32_t type,void *argument){
  (void)unused;(void)service;
  if(type==kIOMessageServiceBusyStateChange && !argument)node_step(NODE_QUIET,0);
}
/* SETTLING's entry: the parent's notifications armed, each iterator drained once (which arms it): the peer
   interfaces' terminations first (B9b: heard from here on), then the interfaces (the ports made), then the peer
   interfaces' arrivals; a notification that cannot be armed is logged, and the node runs without it.  Then IOKit's
   busy state, the service plane root's (the state IOKitGetBusyState and IOKitWaitQuiet read): its changes heard
   (kIOBusyInterest, busy_heard), and it read once they are: 1 where the registry is busy (SETTLING waits for its change
   to 0, or its bound), 0 where it is quiet.  Where the changes cannot be heard, the registry is waited quiet here
   instead (IOKitWaitQuiet, at most MESH_PLATFORM_NS: the operator's fallback), and 0 */
static int iokit_watch(void){
  if(!(iokit=IONotificationPortCreate(kIOMainPortDefault)))die("bridge: no IOKit notification port");
  struct kevent64_s event;
  EV_SET64(&event,IONotificationPortGetMachPort(iokit),EVFILT_MACHPORT,EV_ADD,0,0,TAG_IOKIT,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");
  io_iterator_t iterator=IO_OBJECT_NULL;
  kern_return_t error=IOServiceAddMatchingNotification(iokit,kIOTerminatedNotification,
                                                       IOServiceMatching("AppleThunderboltRDMAPeerInterface"),terminations_heard,NULL,&iterator);
  if(error)say("bridge: Thunderbolt RDMA peer interfaces' terminations not heard (%#x): a link's going off is its first sign\n",(unsigned)error);
  else terminations_heard(NULL,iterator);
  error=IOServiceAddMatchingNotification(iokit,kIOFirstMatchNotification,IOServiceMatching("AppleThunderboltRDMAInterface"),
                                         interfaces_heard,NULL,&iterator);
  if(error)say("bridge: Thunderbolt RDMA interfaces not heard (%#x): this bridge has no port\n",(unsigned)error);
  else interfaces_heard(NULL,iterator);
  error=IOServiceAddMatchingNotification(iokit,kIOFirstMatchNotification,IOServiceMatching("AppleThunderboltRDMAPeerInterface"),
                                         arrivals_heard,NULL,&iterator);
  if(error)say("bridge: Thunderbolt RDMA peer interfaces' arrivals not heard (%#x): a port whose device was not yet present at "
               "its link's coming on opens at the next hello heard on it\n",(unsigned)error);
  else arrivals_heard(NULL,iterator);
  const io_registry_entry_t root=IORegistryEntryFromPath(kIOMainPortDefault,kIOServicePlane ":/");
  error=root?IOServiceAddInterestNotification(iokit,root,kIOBusyInterest,busy_heard,NULL,&busy_notification):KERN_FAILURE;
  if(root)IOObjectRelease(root);
  uint32_t busy=1;
  if(!error)return IOKitGetBusyState(kIOMainPortDefault,&busy) || busy;
  busy_notification=IO_OBJECT_NULL;
  say("bridge node %u: IOKit's busy state not heard (%#x): its registry waited quiet before its ports open\n",node.id,(unsigned)error);
  mach_timespec_t quiet={(unsigned)(MESH_PLATFORM_NS/1000000000u),(clock_res_t)(MESH_PLATFORM_NS%1000000000u)};
  const kern_return_t waited=IOKitWaitQuiet(kIOMainPortDefault,&quiet);
  if(waited)say("bridge node %u: IOKit's registry not quiet within %.1f s (%#x): its ports opened as they stand\n",node.id,
                (double)MESH_PLATFORM_NS/1e9,(unsigned)waited);
  return 0;
}

/* design/transport.md section 14, "A process a port": the parent's side of a port's child.  The bridge is a parent
   (the region, the map and records, the clients, discovery, hellos and the control sockets) and a child a port (its
   device, its link's queue pairs and completion queues, its progress thread): mesh-flow spawned again, the same image
   (`self_path`), told the contract's version it is spawned for (mesh-bridge.h MESH_CHILD_VERSION), in a process group of its own (a job's group signalled at its end reaches no verbs owner), with
   the parent's socket at descriptor 3, the parent's stderr, and nothing else.  The child's exit is the one release of
   its port's verbs objects (the kernel's), and a link once connected and then lost pairs again with the port's next
   child */
static char self_path[PATH_MAX];
/* a message to the port's child, whole, in one send that never waits: 0, or why it was not taken.  Each end reads
   the socket on its own event loop, and only a few messages are ever unread on it (a command goes only after the last
   one's answer; halts and an end besides), far below its buffer, so a message is taken whole while the child lives.
   One not taken whole (the child exited, or the socket full) is the child's loss: its socket closed (a part sent would
   put the next message out of step), so the child reads its end and leaves as at its parent's loss (child_read), and
   its exit is reaped (PORT_EXITED).  A command so lost is that command's failure (port_step) */
static int child_send(struct port *port,struct child_message said){
  if(port->child_fd<0)return EPIPE;
  const ssize_t sent=send(port->child_fd,&said,sizeof said,MSG_DONTWAIT);
  if(sent==(ssize_t)sizeof said)return 0;
  const int error=sent<0?errno:EIO;
  say("link %u (%s): a message to its child %d not taken whole (%s): its socket closed, the child's end its own\n",
      port->index,port->name,(int)port->child,strerror(error));
  close(port->child_fd);port->child_fd=-1;
  return error;
}
/* the port's child spawned, then recorded, then run: its socket's messages watched on the loop's queue (tagged by the
   port) before the spawn, then the child spawned suspended (POSIX_SPAWN_START_SUSPENDED: it runs no instruction yet),
   its pid and start recorded in an entry of the region's processes no live child has (mesh-observe.h struct
   mesh_bridge_pages `child`), and only then resumed (SIGCONT).  So every child that runs is recorded first: a bridge
   that ends at any point of a spawn leaves no child its next launch's ADOPTING does not wait for, and a device's queue
   pairs are one process's at a time (no scan of the node's processes for one).  Nothing after the spawn can fail, so
   no child is spawned that is not run.  Its exit is heard as every child's is, on the loop's SIGCHLD (children_reaped).
   0, or the error */
static int child_spawn(struct port *port){
  int pair[2],on=1;
  struct mesh_process *children=mesh_bridge_pages(M)->child;
  uint32_t entry=0;
  while(entry<2*MESH_PORTS && mesh_alive((pid_t)children[entry].pid,children[entry].start))entry++;
  if(entry==2*MESH_PORTS){say("link %u (%s): its child not spawned: every entry of the region's processes is a live one's\n",port->index,port->name);return EBUSY;}
  if(socketpair(AF_UNIX,SOCK_STREAM,0,pair)){
    const int error=errno?errno:EIO;
    say("link %u (%s): its child not spawned: %s\n",port->index,port->name,strerror(error));
    return error;
  }
  setsockopt(pair[0],SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on);
  struct kevent64_s watch;
  EV_SET64(&watch,(uint64_t)pair[0],EVFILT_READ,EV_ADD,0,0,TAG_CHILD|port->index,0,0);
  if(kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)){
    const int error=errno?errno:EIO;
    close(pair[0]);close(pair[1]);
    say("link %u (%s): its child not spawned: its socket not watched (%s)\n",port->index,port->name,strerror(error));
    return error;
  }
  char index[16];
  snprintf(index,sizeof index,"%u",port->index);
  char version[16];
  snprintf(version,sizeof version,"%u",MESH_CHILD_VERSION);
  char *arguments[]={self_path,"--child",version,port->device,index,NULL};
  posix_spawn_file_actions_t actions;
  posix_spawnattr_t attributes;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions,0,"/dev/null",O_RDONLY,0);
  posix_spawn_file_actions_addopen(&actions,1,"/dev/null",O_WRONLY,0);
  posix_spawn_file_actions_addinherit_np(&actions,2);
  posix_spawn_file_actions_adddup2(&actions,pair[1],3);
  posix_spawnattr_init(&attributes);
  posix_spawnattr_setflags(&attributes,POSIX_SPAWN_CLOEXEC_DEFAULT|POSIX_SPAWN_SETPGROUP|POSIX_SPAWN_START_SUSPENDED);
  posix_spawnattr_setpgroup(&attributes,0);
  pid_t pid=0;
  int error=posix_spawn(&pid,self_path,&actions,&attributes,arguments,*_NSGetEnviron());
  posix_spawn_file_actions_destroy(&actions);posix_spawnattr_destroy(&attributes);
  close(pair[1]);
  if(error){
    close(pair[0]);
    say("link %u (%s): its child not spawned: %s\n",port->index,port->name,strerror(error));
    return error;
  }
  children[entry]=(struct mesh_process){(uint32_t)pid,0,mesh_started(pid)};
  port->child=pid;port->child_entry=entry;port->child_fd=pair[0];port->child_in_length=0;
  kill(pid,SIGCONT);
  say("link %u (%s): its child %d spawned\n",port->index,port->name,(int)pid);
  return 0;
}
/* the child's messages, each whole one taken, each its port's event: its open's, make's and connect's answers
   (PORT_ANSWERED) and its own halt (CHILD_BROKEN, its reason in `result`: PORT_BROKEN).  The socket's end closes it;
   the child's exit is its own event (PORT_EXITED) */
static void child_heard(struct port *port){
  while(port->child_fd>=0){
    const ssize_t got=recv(port->child_fd,port->child_in+port->child_in_length,sizeof port->child_in-port->child_in_length,MSG_DONTWAIT);
    if(got<0 && (errno==EAGAIN || errno==EINTR))return;
    if(got<=0){close(port->child_fd);port->child_fd=-1;return;}
    if((port->child_in_length+=(uint32_t)got)<sizeof port->child_in)continue;
    port->child_in_length=0;
    struct child_message heard;
    memcpy(&heard,port->child_in,sizeof heard);
    if(heard.kind==CHILD_OPENED || heard.kind==CHILD_MADE || heard.kind==CHILD_CONNECTED)port_step(port,PORT_ANSWERED,0,&heard);
    else if(heard.kind==CHILD_BROKEN)port_step(port,PORT_BROKEN,heard.result,NULL);
  }
}

/* ---- the port's machine: its moves and its one transition function ------------------------------------------------ */
/* each state's published phase (mesh-observe.h struct mesh_link_info `phase`, what mesh-stat reads) */
static const uint32_t port_phases[]={[PORT_CLOSED]=MESH_STOPPED,[PORT_OPENING]=MESH_STOPPED,[PORT_DOWN]=MESH_STOPPED,
  [PORT_OPEN]=MESH_PAIRING,[PORT_DIALING]=MESH_PAIRING,[PORT_GREETING]=MESH_PAIRING,[PORT_PAIRED]=MESH_PAIRED,
  [PORT_MAKING]=MESH_PAIRED,[PORT_MADE]=MESH_PAIRED,[PORT_CONNECTING]=MESH_PAIRED,[PORT_UP]=MESH_PAIRED,
  [PORT_EXITING]=MESH_STOPPED};
/* every transition's one move: the port in `state`, its deadline `due`, and its phase and peer published with `code`
   (link_set: a change is a new record, which may lose a port as its RECORD goes out, so a caller that goes on reads
   the state again) */
static void port_move(struct port *port,enum port_state state,uint64_t due,int64_t code){
  port->state=state;port->due=due;
  link_set(port,port_greeted(port)?port->peer:0,port_phases[state],code);
}
/* a loss's window: no open, make or connect on the port, and no exit, for MESH_PLATFORM_NS, the operator's bound on
   the termination a loss may follow (B9) */
static void port_quiet(struct port *port){ quiet_until(&port->quiet_at,now+MESH_PLATFORM_NS); }
/* the port's child halted, or its halt's window moved later (CHILD_HALT naming the port's quiet window's end as it
   stands: the child exits no sooner than that, and once its own windows have ended, end_step), so a loss that left the
   child alive and untold (its link gone off, a socket lost before the make) keeps its exit out of its window (B9) */
static void port_halt(struct port *port){
  child_send(port,(struct child_message){.kind=CHILD_HALT,.until=port->quiet_at});
}
/* a port's control socket closed and its pairing's numbers made new, its loss logged where it had dialed or paired */
static void port_parted(struct port *port,int64_t code){
  if(port->state==PORT_DIALING || port_greeted(port)){
    if(stopping())say("link %u (%s): closed toward node %u: this bridge is stopping\n",port->index,port->name,port->peer);
    else if(port->peer)say("link %u (%s): node %u lost: code %lld\n",port->index,port->name,port->peer,(long long)code);
    else say("link %u (%s): lost: code %lld\n",port->index,port->name,(long long)code);
  }
  if(port->control>=0){close(port->control);port->control=-1;}
  port->writing=0;port->peer=0;port->lanes_heard=0;port->in_length=0;port->out_length=port->out_sent=0;
}
/* a loss before the make (OPEN to PAIRED): the child kept (it has made no queue pair), the port OPEN, its window from
   now, its hello at once where it had paired, at its pace where it had not (so a dial that fails again and again is
   paced) */
static void port_unpaired(struct port *port,int64_t code){
  const uint64_t due=port->state==PORT_OPEN?port->due:port_greeted(port)?now:now+MESH_PAUSE_NS;
  port_quiet(port);port_parted(port,code);
  port_move(port,PORT_OPEN,due,code);
}
/* its link off before the make: as a loss, the device DOWN until its link comes on again (PORT_ON) */
static void port_down(struct port *port,int64_t code){
  port_quiet(port);port_parted(port,code);
  port_move(port,PORT_DOWN,0,code);
}
/* design/transport.md section 14, #loss: a loss from the make on, or a child that halted itself, ends the child: its
   window from now, its line leaving (the clients' rings its port took are then the other serving ports' shares, and its
   lanes no draw's, from the next pass on, even while its poller is stuck in a provider call), its child halted
   (port_halt: no pass of its poller begins after its line leaves, so no post on its lanes once the peer can read the
   loss, and it exits once the window this loss moved MESH_PLATFORM_NS on, and its own windows, have ended), its control
   socket closed, what its lanes carried counted `lost` as its end is freed (switch_child_free), no verbs call made
   (design/RDMA-KERNEL-RECOVERY.md "October 18"), and its exit awaited (EXITING): the kernel's release of the lanes, the
   port greeting no peer meanwhile, and paired again with the port's next child ("A process a port") */
static void port_exiting(struct port *port,int64_t code){
  port_quiet(port);
  port_leaving(port);
  port_halt(port);
  port_parted(port,code);
  port_move(port,PORT_EXITING,0,code);
}
/* a command of the lanes that failed or outlived its bound: the child ends */
static void lanes_failed(struct port *port,const char *what,int64_t code){
  say("link %u (%s): lanes toward node %u not %s: %s; paired again with the port's next child\n",port->index,port->name,
      port->peer,what,strerror((int)code));
  port_exiting(port,code);
}
/* CLOSED's and DOWN's: the port to be opened, at the earliest it may be: its quiet window's end and, DOWN, a refused
   registration's retry (B9; open_answered's backoff); never once the bridge stops */
static void port_want(struct port *port){
  if(stopping())return;
  port->due=MAX(now,MAX(port->quiet_at,port->state==PORT_DOWN?port->retry_at:0));
}
/* the hellos' group joined on the port's interface (its index read again at each open), before its child is spawned or
   asked to open, so the port hears them open or not (hello_heard, B16) and a hello wants it opened again where a spawn
   failed */
static void port_join(struct port *port){
  port->interface=if_nametoindex(port->name);
  struct ipv6_mreq group={.ipv6mr_multiaddr=everyone,.ipv6mr_interface=port->interface};
  if(port->interface)setsockopt(hello,IPPROTO_IPV6,IPV6_JOIN_GROUP,&group,sizeof group);
}
/* the device opened by its child (CHILD_OPEN: OPENING, bound MESH_PLATFORM_NS); a command the child did not take is the
   open's failure (DOWN) */
static void port_opening(struct port *port){
  const int error=child_send(port,(struct child_message){.kind=CHILD_OPEN});
  port->wanted=0;
  if(error)port_move(port,PORT_DOWN,0,error);
  else port_move(port,PORT_OPENING,now+MESH_PLATFORM_NS,0);
}
/* PAIRED's: the child asked to make the plan's lanes (CHILD_MAKE: MAKING, bound MESH_PLATFORM_NS) once the port's quiet
   window has ended (B9), else at its end */
static void port_make(struct port *port){
  if(now<port->quiet_at){port->due=port->quiet_at;return;}
  port_move(port,PORT_MAKING,now+MESH_PLATFORM_NS,0);
  const int error=child_send(port,(struct child_message){.kind=CHILD_MAKE,.lanes=port->plan});
  if(error && port->state==PORT_MAKING)lanes_failed(port,"made",error);
}
/* MADE's: the child asked to connect its lanes to the peer's (CHILD_CONNECT: the peer's lanes as its LINK told them, its
   address as its LINK did, its node and the link's bandwidth; CONNECTING, bound MESH_PLATFORM_NS) once the peer's LINK
   is in and the port's quiet window has ended, else at its end */
static void port_connect(struct port *port){
  if(!port->lanes_heard){port->due=0;return;}
  if(now<port->quiet_at){port->due=port->quiet_at;return;}
  port_move(port,PORT_CONNECTING,now+MESH_PLATFORM_NS,0);
  struct child_message said={.kind=CHILD_CONNECT,.lanes=port->plan,.node=port->peer,.lid=port->heard.lid,.bandwidth=port->bandwidth};
  memcpy(said.peer,port->lanes,sizeof said.peer);memcpy(said.gid,port->heard.gid,16);
  const int error=child_send(port,said);
  if(error && port->state==PORT_CONNECTING)lanes_failed(port,"connected",error);
}
/* GREETING's: the peer's HELLO (one of another wire version is let be: mesh.h MESH_VERSION) names the peer (the node it
   states, whatever node's hello this end dialed) and pairs the link, its plan the fewer queue pairs of the two ends'
   (design/HETEROGENEOUS-LINKS.md), the peer told none of the records yet (mesh-records.c records_forget), then sent
   this node's record (link_set) and the rest of the map's it lacks once the link is up (PORT_CONNECTED); a lane's
   frames are the same constants at both ends (mesh-switched.h mesh_switch_lane_frames) */
static void port_paired(struct port *port,const struct message *heard){
  if(heard->kind!=HELLO || heard->version!=MESH_VERSION)return;
  port->peer=heard->node;port->peer_port=heard->port;port->heard=*heard;
  records_forget(port);
  port->plan=link_plan(MIN(port->budget,heard->count));port->lanes_heard=0;
  struct mesh_link_info *link=&mesh_links(M)[port->index];
  atomic_store_explicit(&link->bandwidth,port->bandwidth,memory_order_relaxed);
  atomic_fetch_add_explicit(&link->pairings,1,memory_order_relaxed);
  say("link %u (%s): paired with node %u (queue pairs for the link: this port's %u, its %u; plan %#x, %u lanes, %u stream%s)\n",
      port->index,port->name,port->peer,port->budget,heard->count,port->plan,
      plan_lanes(port->plan),plan_streams(port->plan),plan_streams(port->plan)==1?"":"s");
  port_move(port,PORT_PAIRED,0,0);
  if(port->state==PORT_PAIRED)port_make(port);
}
/* PAIRED's, MAKING's and MADE's: the peer's LINK, every lane of its pairing and the address they were made on (its
   port's LID and GID as its make read them, which its hello's may predate: the connect's address, and an end of each
   stream's identity alike at both ends, mesh-link.c stream_wire), taken once (a LINK heard again is let be: the first
   stands), its lanes this end's plan's (both ends plan from the same two budgets: design/HETEROGENEOUS-LINKS.md); one
   naming other lanes loses the link, since a lane can be connected only to a queue pair its peer made */
static void port_linked(struct port *port,const struct message *heard){
  if(port->lanes_heard)return;
  if(heard->count!=plan_lanes(port->plan)){
    say("link %u (%s): node %u's LINK names %u lanes, this end's plan %u (plan %#x): the two ends planned apart\n",port->index,
        port->name,port->peer,heard->count,plan_lanes(port->plan),port->plan);
    if(port->state==PORT_PAIRED)port_unpaired(port,EPROTO);
    else port_exiting(port,EPROTO);
    return;
  }
  memcpy(port->lanes,heard->pair,sizeof port->lanes);
  port->heard.lid=heard->lid;memcpy(port->heard.gid,heard->gid,16);
  port->lanes_heard=1;
  if(port->state==PORT_MADE)port_connect(port);
}
/* design/transport.md: OPENING's answer: a device that failed to open opens again at the next hello heard on it or its
   link's next coming on (DOWN); one whose registration of the region it refused (it keeps the extents it took:
   device_up logged the extent and why) does so too, the k-th refusal in a row no sooner than 60 2^(k-1) s on, at most
   an hour, unless its link comes on again (PORT_ON), since each try wires more of the node's memory
   (design/rubrics.md, Failure across interfaces 4 and 6: the Mini's watchdog halted it once its memory was wired
   whole, 10-20), and one whose link came on while it was out (`wanted`) is to be opened at once; one opened says hello
   at once (OPEN), its address, bandwidth and budget as the child told them */
static void open_answered(struct port *port,const struct child_message *heard){
  if(heard->result){
    if(heard->refused){
      port->refusals++;
      const uint64_t backoff=MIN(UINT64_C(60)<<MIN(port->refusals-1,6u),UINT64_C(3600));
      port->retry_at=now+backoff*UINT64_C(1000000000);
      say("link %u (%s): the region's registration refused (%s), %u in a row: opened again at a hello heard on it from %llu s "
          "on, or at its link's coming on\n",port->index,port->name,strerror(heard->refused),port->refusals,(unsigned long long)backoff);
    }
    port_move(port,PORT_DOWN,0,heard->result);
    if(port->wanted && port->state==PORT_DOWN)port_want(port);
    return;
  }
  port->lid=heard->lid;memcpy(port->gid,heard->gid,16);port->bandwidth=heard->bandwidth;port->budget=heard->lanes;
  port->refusals=0;
  port_move(port,PORT_OPEN,now,0);
}
/* CONNECTING's answer: up, every lane served, the peer sent the map's records it lacks and then this node's own naming
   the link: the map first, so a record this end's own then brings into its map goes to no peer that has it
   (mesh-records.c map_tell, map_make) */
static void port_up(struct port *port){
  port_move(port,PORT_UP,0,0);
  say("link %u (%s): %u lanes up toward node %u (plan %#x: %u stream%s), each lane's share %u frames\n",
      port->index,port->name,plan_lanes(port->plan),port->peer,port->plan,plan_streams(port->plan),plan_streams(port->plan)==1?"":"s",
      port->share);
  map_tell(port);
  if(port->state==PORT_UP)record_changed();
}
/* design/transport.md section 14: the port's child reaped, `status` its wait status (children_reaped: its exit seen
   through to its end, the kernel's release of its device's objects; its last messages taken first), its socket
   closed and its entry among the region's processes cleared; one that ended unbidden (in any state but EXITING, while
   the bridge runs) is the link's loss (ECHILD) and is spawned again once the port's quiet window has passed, however
   often its children ended (the operator, 2026-10-23 01:03Z: "this is not a real feature. delete it and anything like
   it").  What it had given back and its line made new (switch_child_free: its queue pairs went with it, so nothing
   lands in or is read from its pages again), and the port CLOSED, to be opened by its next child */
static void port_reaped(struct port *port,int status){
  const pid_t pid=port->child;
  if(port->child_fd>=0){close(port->child_fd);port->child_fd=-1;}
  port->child=0;
  if(port->child_entry<2*MESH_PORTS && mesh_bridge_pages(M)->child[port->child_entry].pid==(uint32_t)pid)
    mesh_bridge_pages(M)->child[port->child_entry]=(struct mesh_process){0};
  int64_t code=atomic_load_explicit(&mesh_links(M)[port->index].code,memory_order_relaxed);
  if(port->state==PORT_EXITING || stopping())say("link %u (%s): its child %d ended\n",port->index,port->name,(int)pid);
  else{
    say("link %u (%s): its child %d ended unbidden (%s %d)\n",port->index,port->name,(int)pid,
        WIFSIGNALED(status)?"signal":"status",WIFSIGNALED(status)?WTERMSIG(status):WEXITSTATUS(status));
    code=ECHILD;
    port_quiet(port);port_leaving(port);port_parted(port,code);
  }
  switch_child_free(port);
  port->plan=port->share=port->psn=0;memset(port->qpn,0,sizeof port->qpn);
  port_move(port,PORT_CLOSED,0,code);
  if(port->state==PORT_CLOSED)port_want(port);
}
/* design/transport.md section 14: a port's one transition function (mesh-bridge.h enum port_state), event `event` taken
   in the port's state, on the event loop; an event a state does not wait for is let be */
static void port_step(struct port *port,enum port_event event,int64_t code,const void *data){
  const enum port_state state=port->state;
  const int open=state>=PORT_OPEN && state<=PORT_PAIRED,lanes=state>=PORT_MAKING && state<=PORT_UP;
  switch(event){
  case PORT_ON:
    port->retry_at=0;
    if(state==PORT_CLOSED || state==PORT_DOWN)port_want(port);
    else if(state==PORT_OPENING)port->wanted=1;
    else if(state==PORT_OPEN)port->due=now;
    return;
  case PORT_ADDRESSED:
    if(state==PORT_OPEN)port->due=now;
    return;
  case PORT_HEARD:{
    const struct sockaddr_in6 *from=data;
    if(state==PORT_CLOSED || state==PORT_DOWN){port_want(port);return;}
    if(state!=PORT_OPEN)return;
    if(M->node>(uint32_t)code){
      struct message said={.magic=XMAGIC,.version=MESH_VERSION,.kind=HELLO,.node=M->node,.boot=M->boot,.port=port->index};
      sendto(hello,&said,sizeof said,0,(const struct sockaddr *)from,sizeof *from);
      return;
    }
    struct sockaddr_in6 to=*from;
    to.sin6_port=htons((uint16_t)atoi(MESH_PORT));
    port->control=socket(AF_INET6,SOCK_STREAM,0);
    if(port->control<0 || fcntl(port->control,F_SETFL,O_NONBLOCK) ||
       (connect(port->control,(struct sockaddr *)&to,sizeof to) && errno!=EINPROGRESS)){port_unpaired(port,errno);return;}
    port->peer=(uint32_t)code;
    port_move(port,PORT_DIALING,0,0);
    port_watch(port,EVFILT_WRITE);
    return;
  }
  case PORT_ACCEPTED:
    if(state!=PORT_OPEN){close((int)code);return;}
    port->control=(int)code;
    port_move(port,PORT_GREETING,0,0);
    port_greet(port);
    return;
  case PORT_DIALED:
    if(state!=PORT_DIALING)return;
    if(code){port_unpaired(port,code);return;}
    port_move(port,PORT_GREETING,0,0);
    port_greet(port);
    return;
  case PORT_MESSAGE:{
    const struct message *heard=data;
    if(state==PORT_GREETING)port_paired(port,heard);
    else if(heard->kind==LINK && state>=PORT_PAIRED && state<=PORT_MADE)port_linked(port,heard);
    return;
  }
  case PORT_GONE:
  case PORT_TERMINATED:
    if(state>=PORT_DIALING && state<=PORT_PAIRED)port_unpaired(port,code);
    else if(event==PORT_TERMINATED && state==PORT_OPEN)port_unpaired(port,code);
    else if(lanes)port_exiting(port,code);
    /* an ending child's halt moved to the window the termination moved (terminations_heard: quiet_all) */
    else if(event==PORT_TERMINATED && state==PORT_EXITING)port_halt(port);
    else if(event==PORT_TERMINATED)port_move(port,state,port->due,code);
    return;
  case PORT_OFF:
    if(open)port_down(port,code);
    else if(lanes)port_exiting(port,code);
    else{
      /* no socket and no lanes: its window moved, which a want and its child's end wait out (B9) */
      port_quiet(port);
      if(state==PORT_EXITING)port_halt(port);
      else if((state==PORT_CLOSED || state==PORT_DOWN) && port->due)port_want(port);
      port_move(port,state,port->due,code);
    }
    return;
  case PORT_ANSWERED:{
    const struct child_message *heard=data;
    if(state==PORT_OPENING && heard->kind==CHILD_OPENED)open_answered(port,heard);
    else if(state==PORT_MAKING && heard->kind==CHILD_MADE){
      if(heard->result){lanes_failed(port,"made",heard->result);return;}
      memcpy(port->qpn,heard->qpn,sizeof port->qpn);port->psn=heard->psn;port->share=heard->share;
      port->lid=heard->lid;memcpy(port->gid,heard->gid,16);
      port_move(port,PORT_MADE,0,0);
      link_tell(port);
      if(port->state==PORT_MADE)port_connect(port);
    }
    else if(state==PORT_CONNECTING && heard->kind==CHILD_CONNECTED){
      if(heard->result)lanes_failed(port,"connected",heard->result);
      else port_up(port);
    }
    return;
  }
  case PORT_BROKEN:
    say("link %u (%s): its child halted itself (%s): its link lost (design/transport.md)\n",port->index,port->name,
        code==ENETDOWN?"a fault of its provider's doorbell absorbed":code==ETIMEDOUT?"its peer interface terminating":
        "a lane or a completion failed");
    if(state!=PORT_CLOSED && state!=PORT_EXITING)port_exiting(port,code>0?code:EIO);
    return;
  case PORT_EXITED:
    port_reaped(port,(int)code);
    return;
  case PORT_DUE:
    switch(state){
    case PORT_CLOSED:
      if(now<port->quiet_at){port->due=port->quiet_at;return;}
      port->due=0;
      port_join(port);
      if(!child_spawn(port))port_opening(port);
      return;
    case PORT_DOWN:
      if(now<MAX(port->quiet_at,port->retry_at)){port->due=MAX(port->quiet_at,port->retry_at);return;}
      port_join(port);
      port_opening(port);
      return;
    case PORT_OPENING:
      /* design/transport.md B12 (22891630^ mesh-flight.h's pending-operation age, per port): an open still out
         MESH_PLATFORM_NS after it was asked is concluded around, once: its phase STOPPED, its window from now, no verbs
         call; the call itself cannot be ended, and its answer, whenever it comes, is still taken (a registration slower
         than its bound completes in the same child).  Each port's calls are its own child's, a process of its own
         (section 14), so one port's stuck call stalls no other port's (a stall of every port's was 10-18's, when one
         process made every port's calls) */
      say("link %u (%s): its open out %.1f s: its port concluded around it (design/transport.md)\n",port->index,port->name,
          (double)MESH_PLATFORM_NS/1e9);
      port_quiet(port);
      port_move(port,PORT_OPENING,0,ETIMEDOUT);
      return;
    case PORT_OPEN:{
      /* design/transport.md: an open port with no peer says hello to every node on its link, every MESH_PAUSE_NS (a
         paced emission, deciding nothing: a peer that starts later is heard as it says its own) */
      port->due=now+MESH_PAUSE_NS;
      setsockopt(hello,IPPROTO_IPV6,IPV6_MULTICAST_IF,&port->interface,sizeof port->interface);
      struct sockaddr_in6 to={.sin6_len=sizeof to,.sin6_family=AF_INET6,.sin6_port=htons((uint16_t)atoi(MESH_PORT)),.sin6_addr=everyone,.sin6_scope_id=port->interface};
      struct message said={.magic=XMAGIC,.version=MESH_VERSION,.kind=HELLO,.node=M->node,.boot=M->boot,.port=port->index};
      sendto(hello,&said,sizeof said,0,(struct sockaddr *)&to,sizeof to);
      return;
    }
    case PORT_PAIRED:
      port_make(port);
      return;
    case PORT_MADE:
      port_connect(port);
      return;
    case PORT_MAKING:
    case PORT_CONNECTING:
      /* B12, as OPENING's: a make or connect concluded around ends the child, which reads its halt as its call returns */
      say("link %u (%s): its %s out %.1f s: its port concluded around it (design/transport.md)\n",port->index,port->name,
          state==PORT_MAKING?"make":"connect",(double)MESH_PLATFORM_NS/1e9);
      port_exiting(port,ETIMEDOUT);
      return;
    default:
      port->due=0;
      return;
    }
  case PORT_STOP:
    /* design/transport.md: the bridge stops: its child halted (port_halt: it exits once the port's quiet window and its
       own have ended, at once where none is open, so a link gone off or a socket lost before the make keeps its exit
       out of the window it opened, B9), its line leaving, its control socket closed (the peer reads the stop as the
       socket's end), and its exit awaited (EXITING); a port with no child CLOSED, opened no more */
    if(!port->child){port_parted(port,0);port_move(port,PORT_CLOSED,0,0);return;}
    port_halt(port);
    port_leaving(port);
    port_parted(port,0);
    port_move(port,PORT_EXITING,0,0);
    return;
  }
}

/* design/transport.md: the bridge's configuration, the canonical property list bin/mesh-user.sh writes from
   the consumer's document (rdma/bridge_configuration.py): its name and its region in bytes (`region_bytes`); NULL, or
   why it is refused */
static const char *configuration_read(const char *path,char *name,size_t room,uint64_t *region_bytes){
  FILE *file=fopen(path,"rb");
  if(!file)return strerror(errno);
  unsigned char bytes[65536];
  const size_t length=fread(bytes,1,sizeof bytes,file);
  const int whole=feof(file);
  fclose(file);
  if(!whole)return "not a configuration property list (too long)";
  CFDataRef data=CFDataCreate(NULL,bytes,(CFIndex)length);
  CFPropertyListRef list=data?CFPropertyListCreateWithData(NULL,data,kCFPropertyListImmutable,NULL,NULL):NULL;
  if(data)CFRelease(data);
  const char *refused=NULL;
  CFTypeRef named=list && CFGetTypeID(list)==CFDictionaryGetTypeID()?CFDictionaryGetValue(list,CFSTR("name")):NULL;
  CFTypeRef sized=list && CFGetTypeID(list)==CFDictionaryGetTypeID()?CFDictionaryGetValue(list,CFSTR("region_bytes")):NULL;
  int64_t value=0;
  if(!list || CFGetTypeID(list)!=CFDictionaryGetTypeID())refused="not a property list of a configuration (rdma/bridge_configuration.py writes it)";
  else if(!named || CFGetTypeID(named)!=CFStringGetTypeID() || !CFStringGetCString(named,name,(CFIndex)room,kCFStringEncodingUTF8))refused="its name is missing";
  else if(!sized || CFGetTypeID(sized)!=CFNumberGetTypeID() || !CFNumberGetValue(sized,kCFNumberSInt64Type,&value) || value<=0)
    refused="its region_bytes, a positive number of bytes, is missing";
  if(list)CFRelease(list);
  *region_bytes=(uint64_t)value;
  return refused;
}

/* design/transport.md section 14, "A process a port": a port's child, `mesh-flow --child VERSION DEVICE INDEX` as its
   parent spawned it (child_spawn), VERSION the contract's (mesh-bridge.h MESH_CHILD_VERSION): the parent's region
   mapped (its magic, version and bridge the parent's), its port the parent's port INDEX on DEVICE (its lane records,
   lanes and rings the region's), the provider's faults absorbed (fault_heard) and its peer interface's terminations
   heard (child_iokit; a child whose listener cannot be armed ends there).  Two threads: its main thread, which takes
   the parent's messages on descriptor 3 one after another, its IOKit notifications and its queue's user events on its
   queue (child_main), makes every verbs call of its open, make and connect itself and answers each at once; and, from
   its connect on, its poller (port_progress), the one thread that posts.  Its state is its port's line (mesh-switched.h
   enum child_state, one compare-exchange a move: mesh_port_move), which its main thread moves as its device opens, as
   its lanes are made and as they connect and serve, and which leaves at its end: its parent's halt (CHILD_HALT), its
   own (child_halt: a lane's or a completion's failure on its poller, a termination heard, a doorbell fault absorbed,
   each told its parent once: CHILD_BROKEN; or a make or connect that failed, which its answer tells), or its parent
   lost (the socket's end, or a message to it the socket did not take whole: child_tell).  A command that does not fit
   the state is refused: ECANCELED once the line leaves, EBUSY otherwise.  The child owns its exit (its end, one state
   machine: end_step): once its own windows have ended and the port's window its parent's halt names, or, where it
   ended without one (its own halt, its parent lost), MESH_PLATFORM_NS after its end; and in either case once its
   poller has returned from its last pass: its exit, the kernel's release of its device's objects, no verbs call
   made */
static unsigned char child_in[sizeof(struct child_message)];
static uint32_t child_got;
/* the child's end, one state machine on its main thread (end_step, its one transition function; the main thread its
   one owner):
   - END_NONE: its end not come; its parent's commands taken;
   - END_ENDING: its end come, its line leaving (child_halt: no pass of its poller begins after it): its parent's halt
     (CHILD_HALT), its own (a lane's or a completion's failure, a termination heard, a doorbell fault absorbed, each
     told its parent once: child_report), or a make or connect that failed (its answer tells its parent).  Its
     parent's socket is still read, so a later halt moves its end later;
   - END_ALONE: as END_ENDING, its parent lost (the socket's end, or a message to it the socket did not take whole):
     the socket read and written no more.
   Each ending state's data: its exit due at `due` once its windows have ended (`exit_at`, B9b) and its poller has
   returned from its last pass (`polling` 0), and at `bound` whatever its poller does (a poller inside a provider call
   may never return: what it holds then is lost, as at a crash, O4).  A halt (CHILD_HALT) is due at the end of its
   port's quiet window as its parent names it (`until`: a loss's, MESH_PLATFORM_NS past the loss, or the window a loss
   or termination before a stop left open, B9), as it comes where that has passed; its own halt, a failed make or
   connect and its parent lost MESH_PLATFORM_NS on, the floor a child whose link or parent was lost keeps before its
   release (B9, B9b).  Its bound is MESH_PLATFORM_NS past its due.  Each end moves both later, never earlier, so a loss
   heard after a stop takes the loss's time.
   Its events: its parent's halt (END_HALT, `until` the window's end it names), its own halt or a failed command
   (END_SELF), its parent lost (END_LOST), its poller's return (END_POLLED), and its exit's time come (END_DUE, at
   end_due: it exits) */
enum end_state { END_NONE, END_ENDING, END_ALONE };
enum end_event { END_HALT, END_SELF, END_LOST, END_POLLED, END_DUE };
static struct { enum end_state state; uint64_t due,bound; } ending;
static void end_step(struct link *l,enum end_event event,uint64_t until){
  switch(event){
  case END_HALT:
  case END_SELF:
  case END_LOST:{
    if(event==END_LOST){
      if(ending.state==END_ALONE)return;
      /* the parent's socket read no more: its end would end each wait at once */
      struct kevent64_s parent;EV_SET64(&parent,3,EVFILT_READ,EV_DELETE,0,0,0,0,0);
      kevent64(events,&parent,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
      ending.state=END_ALONE;
    }
    else if(ending.state==END_NONE)ending.state=END_ENDING;
    child_halt(l,0);
    const uint64_t at=clock_ns(),due=event==END_HALT?MAX(at,until):at+MESH_PLATFORM_NS;
    ending.due=MAX(ending.due,due);
    ending.bound=MAX(ending.bound,due+MESH_PLATFORM_NS);
    return;
  }
  case END_POLLED:
    polling=0;
    return;
  case END_DUE:
    exit(0);
  }
}
/* when the child's exit is due (END_DUE): never before its end; at its bound while its poller has not returned; else
   once its due has come and its windows have ended, at its bound at the latest */
static uint64_t end_due(void){
  if(ending.state==END_NONE)return UINT64_MAX;
  return polling?ending.bound:MIN(ending.bound,MAX(ending.due,exit_at));
}
/* a message to the parent, whole, in one send that never waits (child_send's rule, the other way: a child tells at
   most an answer to each command and one report); one not taken whole is its parent's loss */
static void child_tell(struct link *l,struct child_message said){
  if(ending.state!=END_ALONE && send(3,&said,sizeof said,MSG_DONTWAIT)!=(ssize_t)sizeof said)end_step(l,END_LOST,0);
}
/* the child's own halt told its parent, once (child_halt's reason, taken: CHILD_BROKEN), and its end a loss's */
static void child_report(struct link *l){
  const uint32_t reason=atomic_exchange(&child_reason,0);
  if(!reason)return;
  child_tell(l,(struct child_message){.kind=CHILD_BROKEN,.result=(int32_t)reason});
  end_step(l,END_SELF,0);
}
/* why a command does not fit the line's state, 0 where it does (its state one of `fits`, bits over enum child_state) */
static int32_t child_refused(const struct link *l,uint32_t fits){
  const uint32_t state=mesh_port_state(atomic_load(&switch_of()->port[l->index]));
  return fits>>state&1?0:state==CHILD_LEAVING?ECANCELED:EBUSY;
}
/* the device opened (CHILD_OPEN; again where it is open: its port and address read again): device_up, then its GID
   (device_gid), the line moved to DEVICE_OPEN; the answer its address, bandwidth and the queue pairs it has for a link */
static struct child_message device_open(struct link *l){
  struct child_message said={.kind=CHILD_OPENED,.result=child_refused(l,1u<<CHILD_STARTED|1u<<CHILD_DEVICE_OPEN)};
  int refused=0;
  if(!said.result)said.result=device_up(&l->device,&wire,&refused);
  if(!said.result)said.result=link_lapsed(l);
  if(!said.result)said.result=device_gid(&l->device,l->name);
  if(!said.result && !l->device.gid_matched){
    const uint8_t *g=l->device.gid.raw;
    say("link %u (%s): no GID of %s is a link-local address %s has; GID[%u] %02x%02x:%02x%02x:%02x%02x:%02x%02x:"
        "%02x%02x:%02x%02x:%02x%02x:%02x%02x used\n",l->index,l->name,l->device.name,l->name,l->device.gid_index,
        g[0],g[1],g[2],g[3],g[4],g[5],g[6],g[7],g[8],g[9],g[10],g[11],g[12],g[13],g[14],g[15]);
  }
  if(!said.result && !mesh_port_move(switch_of(),l->index,1u<<CHILD_STARTED|1u<<CHILD_DEVICE_OPEN,CHILD_DEVICE_OPEN,0,0))
    said.result=ECANCELED;
  said.refused=refused;
  if(!said.result){
    l->lid=l->device.port.lid;memcpy(l->gid,&l->device.gid,16);
    said.lid=l->lid;memcpy(said.gid,l->gid,16);said.bandwidth=port_bandwidth(&l->device.port);
    said.lanes=l->device.max_pairs;
  }
  return said;
}
/* the lanes of the plan the command names made (CHILD_MAKE, from DEVICE_OPEN: link_make), the line moved to
   LANES_MADE; the answer each lane's queue pair number, the first sequence number, each lane's share and the port's
   address.  A make that fails ends the child (END_SELF: its line leaving, its end a loss's, as its parent's answer to
   the failure asks): a queue pair it made is released at its exit, never made again or reused */
static struct child_message lanes_make(struct link *l,const struct child_message *heard){
  struct child_message said={.kind=CHILD_MADE,.result=child_refused(l,1u<<CHILD_DEVICE_OPEN)};
  if(said.result)return said;
  l->plan=heard->lanes;
  said.result=link_make(l);
  if(!said.result && !mesh_port_move(switch_of(),l->index,1u<<CHILD_DEVICE_OPEN,CHILD_LANES_MADE,0,0))said.result=ECANCELED;
  if(said.result){end_step(l,END_SELF,0);return said;}
  for(uint32_t k=0;k<MESH_LINK_LANES;k++)said.qpn[k]=l->plan>>k&1?l->provider.queues[k].pair->qp_num:UINT32_MAX;
  said.psn=l->provider.psn;said.share=(uint32_t)l->share.n;
  l->lid=l->device.port.lid;memcpy(l->gid,&l->device.gid,16);
  said.lid=l->lid;memcpy(said.gid,l->gid,16);
  return said;
}
/* the lanes connected to the peer's the command names (CHILD_CONNECT, from LANES_MADE: link_connect) and served
   (link_serve: the line serving toward the command's node, the poller made); a connect that fails ends the child, as a
   make does */
static struct child_message lanes_connect(struct link *l,const struct child_message *heard){
  struct child_message said={.kind=CHILD_CONNECTED,.result=child_refused(l,1u<<CHILD_LANES_MADE)};
  if(said.result)return said;
  l->peer=heard->node;
  said.result=link_connect(l,heard);
  if(!said.result)said.result=link_serve(l,heard->node,heard->bandwidth);
  if(said.result)end_step(l,END_SELF,0);
  return said;
}
static void child_command(struct link *l,const struct child_message *heard){
  if(heard->kind==CHILD_OPEN)child_tell(l,device_open(l));
  else if(heard->kind==CHILD_MAKE)child_tell(l,lanes_make(l,heard));
  else if(heard->kind==CHILD_CONNECT)child_tell(l,lanes_connect(l,heard));
  else if(heard->kind==CHILD_HALT)end_step(l,END_HALT,heard->until);
}
static void child_read(struct link *l){
  while(ending.state!=END_ALONE){
    const ssize_t got=recv(3,child_in+child_got,sizeof child_in-child_got,MSG_DONTWAIT);
    if(got<0 && (errno==EAGAIN || errno==EINTR))return;
    if(got<=0){end_step(l,END_LOST,0);return;}
    if((child_got+=(uint32_t)got)<sizeof child_in)continue;
    child_got=0;
    struct child_message heard;
    memcpy(&heard,child_in,sizeof heard);
    child_command(l,&heard);
  }
}
/* the child's loop: its main thread waits on its queue (the parent's socket, its IOKit notifications, its poller's
   return and a fault absorbed), with no timeout until it ends; ending, until its exit is due (end_due, read again at
   each wake: a termination heard moves its windows, its poller's return its due), and exits then (END_DUE).  It goes
   on reading its parent's socket while it ends, so a loss told after a stop moves its end later (END_HALT); after
   each wake its own halt, if it has halted itself, is told (child_report) */
static int child_main(int argc,char **argv){
  signal(SIGPIPE,SIG_IGN);
  const unsigned long index=argc>=5?strtoul(argv[4],NULL,10):MESH_PORTS;
  if(index>=MESH_PORTS){say("child of %s: port %lu is past the bridge's, or its command line short: it ends\n",argc>3?argv[3]:"?",index);return 1;}
  faults_install();
  const int fd=shm_open(MESH_NAME,O_RDWR);
  struct stat shape;
  if(fd<0 || fstat(fd,&shape))die("child: its bridge's region");
  M=mmap(NULL,(size_t)shape.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  if(M==MAP_FAILED || M->magic!=MESH_MAGIC || M->version!=MESH_LAYOUT || atomic_load(&M->bridge_pid)!=(uint64_t)getppid())
    die("child: its bridge's region is not its parent's");
  if(wire_map(&wire,M,fd))die("child: transport page aliases");
  close(fd);
  struct link *l=&child_link;
  *l=(struct link){.index=(uint32_t)index};
  l->provider.device=&l->device;
  snprintf(l->device.name,sizeof l->device.name,"%s",argv[3]);snprintf(l->name,sizeof l->name,"%s",argv[3]+5);
  struct kevent64_s event;
  if((events=kqueue())<0)die("child events");
  const uint64_t users[]={POLLER_RETURNED,FAULT_ABSORBED};
  for(size_t i=0;i<sizeof users/sizeof *users;i++){
    EV_SET64(&event,users[i],EVFILT_USER,EV_ADD|EV_CLEAR,0,0,UINT64_MAX,0,0);
    if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("child events");
  }
  EV_SET64(&event,3,EVFILT_READ,EV_ADD,0,0,UINT64_MAX,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("child events");
  child_iokit();
  for(;;){
    struct timespec rest,*wait=NULL;
    const uint64_t due=end_due(),at=clock_ns();
    if(at>=due)end_step(l,END_DUE,0);
    if(due!=UINT64_MAX){
      const uint64_t left=due-at;
      rest=(struct timespec){(time_t)(left/1000000000u),(long)(left%1000000000u)};wait=&rest;
    }
    struct kevent64_s ready[4];
    const int count=kevent64(events,NULL,0,ready,4,0,wait);
    now=clock_ns();
    for(int i=0;i<count;i++){
      if(ready[i].filter==EVFILT_READ && ready[i].ident==3)child_read(l);
      else if(ready[i].filter==EVFILT_USER && ready[i].ident==POLLER_RETURNED)end_step(l,END_POLLED,0);
      else if(ready[i].filter==EVFILT_MACHPORT)iokit_heard();
    }
    child_report(l);
  }
}

/* design/transport.md section 2, the region and the heartbeat: the node's region (MESH_NAME) opened.  The region a
   bridge before this one left (a restart, SIGHUP; a crash) is adopted where its magic, layout version (MESH_LAYOUT) and
   layout (`geometry`, mesh_layout of this configuration's region) are this bridge's: the same memory, its clients'
   slots, indices and pages standing, its pages registered again by the port children (the same
   pages: nothing is wired twice, nothing reloaded).  A region a live bridge serves is never read: a second bridge of
   the node ends at its port's bind, before region_open.  A region of another layout, or one a stop released (layout version 0), is made
   expired (its layout version 0: no client's attach takes it, mesh.h mesh_expire) and left named, and this bridge
   ends with EX_TEMPFAIL:
   its heartbeat unlinks it once no process maps it and opens a bridge again, so no second region is ever made beside
   a live client of the first (the 2026-10-20 watchdog panic, design/rubrics.md "Failure across interfaces"); a bridge
   unlinks only a shared memory object that never became a region (no magic: a bridge ended while making it).  A
   region that cannot be read is left as it is, and this bridge ends.  1 adopted, 0 made; *out its descriptor */
static int region_same(const struct hdr *a,const struct hdr *b){
  return a->pgsz==b->pgsz && a->block==b->block && a->entries==b->entries && a->streams==b->streams && a->clients==b->clients &&
         a->table_off==b->table_off && a->arena_off==b->arena_off && a->ring_off==b->ring_off && a->pool_off==b->pool_off &&
         a->bridge_off==b->bridge_off && a->stream_off==b->stream_off && a->node_off==b->node_off && a->map_off==b->map_off &&
         a->switch_off==b->switch_off && a->client_off==b->client_off && a->data_off==b->data_off && a->length==b->length;
}
static int region_open(const struct hdr *geometry,int *out){
  int fd=shm_open(MESH_NAME,O_RDWR,0);
  if(fd<0 && errno!=ENOENT){say("bridge: region %s not opened (%s): this bridge ends without touching it\n",MESH_NAME,strerror(errno));exit(1);}
  if(fd>=0){
    struct stat shape;
    if(fstat(fd,&shape)){say("bridge: region %s not read (%s): this bridge ends without touching it\n",MESH_NAME,strerror(errno));exit(1);}
    const size_t size=(size_t)shape.st_size;
    struct hdr *old=size>=sizeof(struct hdr)?mmap(NULL,size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0):NULL;
    if(old==MAP_FAILED){say("bridge: region %s not mapped (%s): this bridge ends without touching it\n",MESH_NAME,strerror(errno));exit(1);}
    if(old && old->magic==MESH_MAGIC){
      if(__atomic_load_n(&old->version,__ATOMIC_ACQUIRE)==MESH_LAYOUT && region_same(old,geometry)){M=old;*out=fd;return 1;}
      mesh_expire(old);
      say("bridge: region %s is expired (another layout, or released): this bridge ends, and its heartbeat unlinks it once no "
          "process of it lives and opens a bridge again\n",MESH_NAME);
      exit(EX_TEMPFAIL);
    }
    if(old)munmap(old,size);
    close(fd);
    shm_unlink(MESH_NAME);
  }
  if((fd=shm_open(MESH_NAME,O_CREAT|O_EXCL|O_RDWR,MESH_MODE))<0)die("shm");
  if(ftruncate(fd,(off_t)geometry->length))die("ftruncate");
  fchmod(fd,MESH_MODE);
  M=mmap(NULL,(size_t)geometry->length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  if(M==MAP_FAILED)die("mmap");
  *M=*geometry;M->version=MESH_LAYOUT;
  *out=fd;
  return 0;
}

/* the adopted region's bridge state made this bridge's, before any client reads it as live (`bridge_pid` set last) and
   before any child is spawned, once the children of the bridge before have ended (NODE_ADOPTING): every lane record
   with no peer, the links' table cleared, a client slot whose leaving a process that died left half done
   finished (mesh.h mesh_leave, which any party can repeat to the same end), then what the bridge before had given
   back at once, as a child's end gives it back (switch_child_free): each lane's in-flight ring and each port's inbound
   rings emptied (a SEND posted or a page pushed: its use dropped, counted `lost`), and every page posted as a
   receive given back (a switch of another layout is not read: what its rings had stays in use).  A
   child's device may still land bytes into a page its receives were posted into until the kernel
   releases its objects in the child's exit, which another use may have by then: that is region memory, data, never
   firmware; the firmware rule, one process driving a device's queue pairs at a time, is met by the stop's wait on
   its own children until they are reaped or MESH_PLATFORM_NS passes (NODE_ENDING), and, where a bridge ended without
   that wait, by NODE_ADOPTING, which waits for every child the bridge before recorded (each recorded before it runs:
   child_spawn); it does not find a child whose exit has begun and not ended (node_end).  What is given
   back is
   what the bridge before had, each use by the ring or receive that had it, never a recount of each page's
   uses from the table: the clients attached across the restart change words, ring cells and uses as it runs (a
   write's use counted as it pushes its cell, an invalidation's dropped as it clears a word), so a count made from one
   reading of the table and the rings would miss or double each change made between its reads.  The map's published
   copy stands, the bridge before's, until this bridge's first publish (node_start's record_changed): a bridge that
   died mid-write wrote only the copy not published.  The pool's carve and the page table stand; the switch is made
   new after (switch_make), so what was given back is said here */
static void region_adopt(void){
  for(uint32_t s=0;s<MESH_STREAMS;s++){
    struct mesh_lanes *x=mesh_lanes_at(M,s);
    atomic_store(&x->peer,0);atomic_store(&x->wire,0);
  }
  struct mesh_link_info *links=mesh_links(M);
  for(uint32_t p=0;p<MESH_PORTS;p++){
    atomic_store(&links[p].peer,0);atomic_store(&links[p].phase,MESH_UNKNOWN);atomic_store(&links[p].code,0);atomic_store(&links[p].bandwidth,0);
  }
  struct mesh_switch *sw=mesh_switch_of(M);
  uint64_t lost=0;
  uint32_t receives=0;
  if(sw->magic==MESH_SWITCH_MAGIC){
    const uint64_t before=atomic_load(&sw->counters[MESH_SWITCH_PARENT].lost);
    for(uint32_t l=0;l<MESH_SWITCH_LANES;l++)mesh_switch_flight_drop(M,sw,l,MESH_SWITCH_PARENT);
    for(uint32_t p=0;p<MESH_PORTS;p++)mesh_switch_port_drop(M,sw,p,MESH_SWITCH_PARENT);
    receives=mesh_switch_receives_drop(M,sw,0,MESH_SWITCH_LANES);
    lost=atomic_load(&sw->counters[MESH_SWITCH_PARENT].lost)-before;
  }
  say("bridge: what the bridge before had given back: %llu pages of its queues (lost) and %u of its "
      "receives\n",(unsigned long long)lost,receives);
  atomic_store(&M->busy,0);
}

/* the node state's deadline: ADOPTING's, SETTLING's and ENDING's bound, STOPPING's the earlier of every quiet window's
   end and its bound; none (UINT64_MAX) while RUNNING */
static uint64_t node_due(void){
  if(node.state==NODE_RUNNING)return UINT64_MAX;
  return node.state==NODE_STOPPING?MIN(windows_end(),node.due):node.due;
}
/* design/transport.md: when the event loop's next turn is due if no event comes first: the earliest of the times its turn
   waits for that are still ahead (the node state's deadline, node_due; while it serves, the next liveness turn,
   `liveness`, and the next health sample; while it runs, each port's state's deadline, struct port `due`), and no
   later than MESH_PAUSE_NS on (design/rubrics.md M3: the wait ends in finite time).  Every other change a turn takes
   comes as an event on the loop's queue (a socket, a signal, an exit, IOKit's port), so a turn comes at the change
   or at the time it waits for, where a fixed tick (250 ms) took each such time up to a tick late; a client slot taken
   or left is read at the next turn, whatever woke it (clients_turn) */
static inline void due_at(uint64_t *due,uint64_t at){ if(at>now && at<*due)*due=at; }
static uint64_t turn_due(uint64_t liveness){
  uint64_t due=now+MESH_PAUSE_NS;
  due_at(&due,node_due());
  if(node.state!=NODE_SETTLING && node.state!=NODE_RUNNING)return due;
  due=MIN(due,liveness);
  due_at(&due,health_due());
  if(node.state!=NODE_RUNNING)return due;
  for(uint32_t p=0;p<port_count;p++)if(ports[p].due)due_at(&due,ports[p].due);
  return due;
}

/* SETTLING ended (IOKit's registry quiet, or its bound passed: `due`): its busy notification released, the node
   RUNNING, and every message the notification port still holds taken (iokit_heard: a termination it tells moves the
   windows) before the first port opens (the loop's next turn) */
static void node_settled(int due){
  if(due)say("bridge node %u: IOKit's registry not quiet within %.1f s: its ports opened as they stand\n",node.id,
             (double)MESH_PLATFORM_NS/1e9);
  if(busy_notification){IOObjectRelease(busy_notification);busy_notification=IO_OBJECT_NULL;}
  node.state=NODE_RUNNING;
  iokit_heard();
}
/* the node started, SETTLING (an adopted region's at ADOPTING's end, one made new at once): an adopted region's bridge
   state given back (region_adopt), then the node's record, its pool and switch, its sockets read on the loop and
   `bridge_pid` set last, so no client reads it as live before; then SETTLING's IOKit (iokit_watch: B9b's terminations
   heard from here on, the ports made), RUNNING once IOKit's registry is quiet (no termination in its course, B9), at
   most MESH_PLATFORM_NS on.  The turn's clock (`now`) read first: main starts a node before its loop's first turn
   reads it, and a port made here takes its opening's time from it (port_found, port_want: a due of 0 is none) */
static void node_start(int adopted){
  now=clock_ns();
  if(adopted)region_adopt();
  M->node=node.id;M->boot=clock_gettime_nsec_np(CLOCK_REALTIME);M->idle_since=M->boot;
  mesh_bitmaps_init(M);
  switch_make();
  size_t size=sizeof memory;
  if(sysctlbyname("hw.memsize",&memory,&size,NULL,0))memory=0;
  sysctl_text("hw.model",model,sizeof model);
  health_init();
  record_changed();
  struct kevent64_s event;
  int watched[4]={listener,hello,network,addresses};
  for(int i=0;i<4 && watched[i]>=0;i++){EV_SET64(&event,(uint64_t)watched[i],EVFILT_READ,EV_ADD,0,0,UINT64_MAX,0,0);if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");}
  mesh_bridge_pages(M)->bridge_start=mesh_started(getpid());
  atomic_store(&M->bridge_pid,(uint64_t)getpid());
  __sync_synchronize();M->magic=MESH_MAGIC;
  node.state=NODE_SETTLING;node.due=clock_ns()+MESH_PLATFORM_NS;
  if(!iokit_watch())node_settled(0);
}
/* NODE_ADOPTING's read of the children of the bridge before (the region's processes, mesh-observe.h `child`): each live
   one's exit watched on the loop's queue (mesh-observe.h mesh_exits_watch, TAG_ADOPT), each exit a read again, and the
   node started once none lives; at the state's bound (`due`) one still alive ends this bridge, which its heartbeat
   opens again */
static void adopting_read(int due){
  node.adopting=mesh_exits_watch(events,node.children,node.adopting,TAG_ADOPT);
  if(!node.adopting){node_start(1);return;}
  if(!due)return;
  say("bridge: %u children of the bridge before live %.0f s on: this bridge ends, and its heartbeat opens one again\n",
      node.adopting,(double)(2*MESH_PLATFORM_NS)/1e9);
  exit(1);
}
/* design/transport.md: the stop (NODE_STOPPING), at stop signal `code`: every child halted (CHILD_HALT naming its
   port's quiet window, port_halt: its line leaving, so no pass of its poller begins after it and no post is made once
   its peer can read the stop, and its exit once that window and its own have ended and its poller has returned,
   end_step), every control socket closed (the
   kernel sends what it has after the close; the peer reads the stop as the socket's end), every lane no draw's (each
   port's PORT_STOP; what the lanes have the next bridge's adoption gives back: region_adopt), `bridge_pid` 0 (every stop
   keeps the region for the next bridge to adopt; a release is the heartbeat's, which expires the region once this
   bridge has ended: mesh-heartbeat.c region_expire), and no peer heard or dialed again (the listener's, the hellos'
   and the addresses' filters deleted).  No queue, thread or verbs call is waited on, and every object is the
   kernel's to release at the children's exits (design/RDMA-KERNEL-RECOVERY.md "October 18"), each child's no sooner
   than its own windows end (a termination it hears, B9b; a loss its parent tells it of, PORT_OFF, PORT_TERMINATED:
   B9).  The bridge exits no sooner than every port's quiet window ends and a peer interface's termination's
   (`exit_at`, B9b), the loop serving the children's sockets, the clients' exits and the kernel's link events meanwhile
   (a link going off extends its port's window and its child's, network_heard, as a termination extends them all,
   terminations_heard), each no later than MESH_PLATFORM_NS past the stop's start (`due`: every window set before the
   stop ends by then, so the wait is the one 5 s budget, well inside its heartbeat's bound of 3 MESH_PLATFORM_NS) */
static void node_stop(int64_t code){
  char why[96];stop_text(code,why,sizeof why);
  say("bridge node %u: stopping (%s)\n",node.id,why);
  node.state=NODE_STOPPING;node.due=clock_ns()+MESH_PLATFORM_NS;
  for(uint32_t p=0;p<port_count;p++)port_step(ports+p,PORT_STOP,0,NULL);
  atomic_store(&M->bridge_pid,0);
  const int unheard[]={listener,hello,addresses};
  for(size_t i=0;i<sizeof unheard/sizeof *unheard;i++)if(unheard[i]>=0){
    struct kevent64_s event;EV_SET64(&event,(uint64_t)unheard[i],EVFILT_READ,EV_DELETE,0,0,0,0,0);
    kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
  }
  const uint64_t at=clock_ns(),latest=MIN(windows_end(),node.due);
  if(latest>at)say("bridge node %u: its exit put off %.1f s, to its quiet windows' end (design/transport.md)\n",node.id,
                   (double)(latest-at)/1e9);
}
/* NODE_ENDING ended (`due`: at its bound): the bridge exits once no child of it is left, its children still alive at the
   bound counted in the log (the next bridge's ADOPTING waits for those whose exit has not begun, and nothing waits for
   one already in its exit) */
static void node_ended(int due){
  uint32_t alive=0;
  for(uint32_t p=0;p<port_count;p++)alive+=ports[p].child!=0;
  if(alive && !due)return;
  if(alive)say("bridge node %u: %u of its children alive %.1f s after they were asked to end (the next bridge waits for "
               "those not yet in their exit)\n",node.id,alive,(double)MESH_PLATFORM_NS/1e9);
  exit(0);
}
/* design/transport.md section 14, "A process a port": the end (NODE_ENDING), once the stop's windows have ended: the
   bridge ended once every child, each halted at the stop and ending by its own windows (end_step), is reaped
   (PORT_EXITED, on the loop: its exit seen through to its end, children_reaped), at most MESH_PLATFORM_NS on, so
   no child of this launch has its device's queue pairs as the next launch's children make theirs.  This end stays
   beside the next bridge's ADOPTING because it alone sees an exit through to its end: a process in its exit is no
   longer found by its pid, before the kernel releases its device's objects (design/transport.md section 2), so ADOPTING
   does not see a child of this launch whose exit has begun and not ended */
static void node_end(void){
  node.state=NODE_ENDING;node.due=clock_ns()+MESH_PLATFORM_NS;
  node_ended(0);
}
/* design/transport.md section 2: the node's one transition function (enum node_state), event `event` taken in the
   current state: a stop signal (`code` its number), an exit the state waits on (ADOPTING: a child of the bridge
   before; ENDING: a child of this one, reaped), IOKit's registry quiet (SETTLING), the state's deadline (node_due) */
static void node_step(enum node_event event,int64_t code){
  switch(node.state){
  case NODE_ADOPTING:
    if(event==NODE_EXIT || event==NODE_DUE)adopting_read(event==NODE_DUE);
    else if(event==NODE_STOP)node_stop(code);
    return;
  case NODE_SETTLING:
    if(event==NODE_QUIET || event==NODE_DUE)node_settled(event==NODE_DUE);
    else if(event==NODE_STOP)node_stop(code);
    return;
  case NODE_RUNNING:
    if(event==NODE_STOP)node_stop(code);
    return;
  case NODE_STOPPING:
    if(event==NODE_DUE)node_end();
    return;
  case NODE_ENDING:
    if(event==NODE_EXIT || event==NODE_DUE)node_ended(event==NODE_DUE);
    return;
  }
}
/* design/transport.md section 14, "A process a port": the loop's SIGCHLD, the kernel's word that a child's exit is
   complete (the kernel has released its device's objects, and the child can be reaped): each port's child that has
   exited reaped (waitpid with WNOHANG: no wait), its last messages taken first (child_heard), then its port's
   PORT_EXITED, which spawns the port's next child only now, and the node's NODE_EXIT (ENDING waits for the reaps).
   A child's exit is heard as it completes, not as it begins (NOTE_EXIT), so the loop never waits on one whose kernel
   release stalls.  One SIGCHLD may tell several exits, or a child's stop or continuation, which reaps none */
static void children_reaped(void){
  for(uint32_t p=0;p<port_count;p++){
    struct port *port=ports+p;
    int status=0;
    pid_t got=0;
    while(port->child && (got=waitpid(port->child,&status,WNOHANG))<0 && errno==EINTR){}
    if(!port->child || got!=port->child)continue;
    child_heard(port);
    port_step(port,PORT_EXITED,status,NULL);
    node_step(NODE_EXIT,0);
  }
}

/* design/transport.md: the node's resident bridge; its one operand the path of its configuration (no default); a port's
   child as its parent spawns it (child_main) */
int main(int argc,char **argv){
  if(argc>=3 && !strcmp(argv[1],"--child"))return child_main(argc,argv);
  if(argc!=2)die("usage: mesh-flow CONFIGURATION: the canonical property list bin/mesh-user.sh writes from the node's bridge configuration document (etc/bridge/README.md); the bridge has no default and does not start without one");
  char name[256];uint64_t region_bytes;
  const char *refused=configuration_read(argv[1],name,sizeof name,&region_bytes);
  if(refused){say("bridge configuration %s refused: %s\n",argv[1],refused);exit(1);}
  uint32_t room=sizeof self_path;
  char resolved[PATH_MAX];
  if(_NSGetExecutablePath(self_path,&room) || !realpath(self_path,resolved))die("the bridge's own image");
  snprintf(self_path,sizeof self_path,"%s",resolved);
  /* design/transport.md: every catchable signal whose default ends the process takes the stop, its filter
     added before its default is ignored, so none is lost (one arriving before takes its default action, before anything
     shared is made) */
  struct kevent64_s event;
  static const int stopping_signals[]={SIGINT,SIGTERM,SIGHUP,SIGQUIT,SIGUSR1,SIGUSR2,SIGALRM,SIGVTALRM,SIGPROF,SIGXCPU,SIGXFSZ};
  if((events=kqueue())<0)die("bridge events");
  for(size_t i=0;i<sizeof stopping_signals/sizeof *stopping_signals;i++){
    EV_SET64(&event,(uint64_t)stopping_signals[i],EVFILT_SIGNAL,EV_ADD,0,0,UINT64_MAX,0,0);
    if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");
    signal(stopping_signals[i],SIG_IGN);
  }
  /* design/transport.md section 14: the children's exits (children_reaped), before any child is spawned; SIGCHLD's
     disposition left the default, since one set to SIG_IGN has the kernel reap the children itself */
  EV_SET64(&event,SIGCHLD,EVFILT_SIGNAL,EV_ADD,0,0,UINT64_MAX,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");
  signal(SIGPIPE,SIG_IGN);
  uuid_t uuid;struct timespec bound={5,0};
  if(gethostuuid(uuid,&bound)){say("host identity: %s; this bridge's node is drawn at random\n",strerror(errno));arc4random_buf(uuid,sizeof uuid);}
  for(int i=0;i<16;i++)node.id=(node.id*0x01000193u)^uuid[i];
  node.id=(node.id&0x7fffffffu)?node.id&0x7fffffffu:1;
  /* design/transport.md: the node's hello port, the mesh's */
  const uint16_t service=(uint16_t)atoi(MESH_PORT);
  const uint32_t pg=(uint32_t)getpagesize();
  /* design/transport.md */
  if((uint64_t)pg+4096>16773120)die("transport chunk exceeds native request capacity");
  if(region_bytes<pg || region_bytes/pg>UINT32_MAX){say("bridge configuration %s refused: its region of %llu bytes is not between one page and 2^32 pages\n",name,(unsigned long long)region_bytes);exit(1);}
  struct hdr geometry={0};
  mesh_layout(&geometry,pg,1,(uint32_t)(region_bytes/pg));
  /* design/transport.md section 2: the bridge's port bound before its region is read, so a second bridge of the node
     (one started beside the heartbeat's, or while the last one ends) ends at its bind, never touching the region */
  int on=1;
  struct sockaddr_in6 any={.sin6_len=sizeof any,.sin6_family=AF_INET6,.sin6_port=htons(service),.sin6_addr=IN6ADDR_ANY_INIT};
  listener=socket(AF_INET6,SOCK_STREAM,0);hello=socket(AF_INET6,SOCK_DGRAM,0);network=socket(PF_SYSTEM,SOCK_RAW,SYSPROTO_EVENT);
  struct kev_request filter={KEV_VENDOR_APPLE,KEV_NETWORK_CLASS,KEV_DL_SUBCLASS};
  unsigned loop=0;
  if(listener<0 || hello<0 || network<0 || setsockopt(listener,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on) ||
     setsockopt(hello,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on) || setsockopt(hello,IPPROTO_IPV6,IPV6_MULTICAST_LOOP,&loop,sizeof loop) ||
     bind(listener,(struct sockaddr *)&any,sizeof any) || listen(listener,SOMAXCONN) || bind(hello,(struct sockaddr *)&any,sizeof any) ||
     fcntl(listener,F_SETFL,O_NONBLOCK) || fcntl(hello,F_SETFL,O_NONBLOCK) || fcntl(network,F_SETFL,O_NONBLOCK) || ioctl(network,SIOCSKEVFILT,&filter))
    die("bridge sockets");
  /* design/transport.md: the kernel's IPv6 address events (addresses_heard); a bridge that cannot hear them says a
     port's hello at its pace alone */
  struct kev_request inet6={KEV_VENDOR_APPLE,KEV_NETWORK_CLASS,KEV_INET6_SUBCLASS};
  if((addresses=socket(PF_SYSTEM,SOCK_RAW,SYSPROTO_EVENT))<0 || fcntl(addresses,F_SETFL,O_NONBLOCK) || ioctl(addresses,SIOCSKEVFILT,&inet6)){
    say("bridge: the kernel's IPv6 address events not heard (%s): a port's hello after its address comes waits its pace\n",strerror(errno));
    if(addresses>=0)close(addresses);
    addresses=-1;
  }
  int fd=-1;
  const int adopted=region_open(&geometry,&fd);
  say("bridge node %u: region %s %s\n",node.id,MESH_NAME,adopted?"adopted from the bridge before this one (its clients' indices and pages stand)":"made new");
  if(wire_map(&wire,M,fd))die("transport page aliases");
  close(fd);
  bridge.M=M;
  say("bridge node %u: configuration %s, region %llu bytes, arena %llu pages, page table %u entries\n",node.id,name,
      (unsigned long long)region_bytes,(unsigned long long)mesh_arena_pages(M),M->entries);
  /* an adopted region's node starts once the children of the bridge before have exited (NODE_ADOPTING); a region made
     new starts at once */
  if(adopted){
    memcpy(node.children,mesh_bridge_pages(M)->child,sizeof node.children);
    node.adopting=2*MESH_PORTS;node.state=NODE_ADOPTING;node.due=clock_ns()+2*MESH_PLATFORM_NS;
    adopting_read(0);
  }
  else node_start(0);
  uint64_t liveness=0;
  for(;;){
    now=clock_ns();
    if(now>=node_due())node_step(NODE_DUE,0);
    /* each port's deadline passed, while the node runs (SETTLING's ports open once it runs) */
    if(node.state==NODE_RUNNING)
      for(uint32_t p=0;p<port_count;p++)if(ports[p].due && now>=ports[p].due)port_step(ports+p,PORT_DUE,0,NULL);
    if(node.state==NODE_SETTLING || node.state==NODE_RUNNING){
      health_turn();
      if(now>=liveness){
        liveness=now+MESH_PAUSE_NS;
        record_retell();
      }
    }
    if(node.state!=NODE_ADOPTING)clients_turn();
    struct kevent64_s ready[16];
    const uint64_t due=turn_due(liveness),at=clock_ns(),rest=due>at?due-at:0;
    struct timespec tick={(time_t)(rest/1000000000u),(long)(rest%1000000000u)};
    int count=kevent64(events,NULL,0,ready,16,0,&tick);
    now=clock_ns();
    for(int i=0;i<count;i++){
      struct kevent64_s *e=ready+i;
      /* design/transport.md section 14, "A process a port": a port's child's exit complete (SIGCHLD) */
      if(e->filter==EVFILT_SIGNAL && e->ident==SIGCHLD)children_reaped();
      else if(e->filter==EVFILT_SIGNAL)node_step(NODE_STOP,(int64_t)e->ident);
      else if(e->filter==EVFILT_PROC && e->udata==TAG_ADOPT)node_step(NODE_EXIT,0);
      /* design/transport.md: the parent's IOKit, its callbacks run here */
      else if(e->filter==EVFILT_MACHPORT && e->udata==TAG_IOKIT)iokit_heard();
      /* design/transport.md section 14, "A process a port": a port's child's messages */
      else if(e->udata!=UINT64_MAX && (e->udata&TAG_CHILD)){
        struct port *port=(e->udata&0xff)<port_count?ports+(e->udata&0xff):NULL;
        if(port && e->filter==EVFILT_READ && (int)e->ident==port->child_fd)child_heard(port);
      }
      else if(e->filter==EVFILT_PROC){}
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)network)network_heard();
      /* peers heard and dialed while the node serves: from the stop on their filters are deleted (node_stop), and an
         event of theirs taken in the stop's batch is let be */
      else if(stopping()){}
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)listener){
        struct sockaddr_in6 from;socklen_t size=sizeof from;int accepted;
        while((accepted=accept(listener,(struct sockaddr *)&from,&size))>=0){
          struct port *port=port_at(from.sin6_scope_id);size=sizeof from;
          if(port)port_step(port,PORT_ACCEPTED,accepted,NULL);
          else close(accepted);
        }
      }
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)hello)hello_heard();
      else if(e->filter==EVFILT_READ && addresses>=0 && e->ident==(uint64_t)addresses)addresses_heard();
      else if(e->udata<port_count && (int)e->ident==ports[e->udata].control){
        struct port *port=ports+e->udata;
        if(e->filter==EVFILT_WRITE && port->state==PORT_DIALING){
          int error=0;socklen_t size=sizeof error;
          if(getsockopt(port->control,SOL_SOCKET,SO_ERROR,&error,&size))error=errno;
          port_step(port,PORT_DIALED,error,NULL);
        }
        else if(e->filter==EVFILT_WRITE){port->writing=0;port_flush(port);}
        else control_heard(port);
      }
    }
  }
}
