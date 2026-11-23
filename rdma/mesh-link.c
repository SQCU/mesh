#include "mesh-bridge.h"
#include <spawn.h>
#include <netinet/tcp.h>
#include <sys/wait.h>
#include <crt_externs.h>
#include <libproc.h>
#include <sysexits.h>
#include <mach/mach_time.h>

/* mesh-verbs.h: the stop flag every loop reads */
_Atomic sig_atomic_t stop;

struct hdr *M;
struct wire_mapping wire;
struct port ports[MESH_PORTS];
uint32_t port_count;
int events=-1,listener=-1,hello=-1,network=-1,addresses=-1,finding;
uint64_t now,sequence;
struct mesh_ctx bridge={.fd=-1};
static dispatch_queue_t finder;
/* design/transport.md: the devices to be read again (an interface that no port is attached or its link came on), read
   on the event loop's turn once no window stands and no call is stuck (ports_find) */
static int find_wanted;
/* design/transport.md B12: a port's job out past MESH_PLATFORM_NS, during which the devices are not read
   (ports_stalled's, each event-loop turn) */
static _Atomic int stalled;
/* the ports a provider's doorbell fault was absorbed on (fault_heard), a bit a port */
static _Atomic uint32_t faulted;
/* design/transport.md B9b: the window of a peer interface's termination (`exit_at`), as every port's quiet
   window stands, before which the bridge does not exit nor read the devices */
static _Atomic uint64_t exit_at;
/* a window's end moved to `until` where it is earlier, by any thread (the event loop's conclusions, a termination's
   queue, B9b), never earlier */
static void quiet_until(_Atomic uint64_t *at,uint64_t until){
  uint64_t was=atomic_load(at);
  while(was<until && !atomic_compare_exchange_weak(at,&was,until)){}
}
/* the latest of the exit's window and every port's (B9, B9b): the stop's wait, and the devices' read (ports_find) */
static uint64_t windows_end(void){
  uint64_t latest=atomic_load(&exit_at);
  for(uint32_t p=0;p<port_count;p++){const uint64_t quiet=atomic_load(&ports[p].quiet_at);latest=MAX(latest,quiet);}
  return latest;
}
_Atomic int found;
static char devices[MESH_PORTS][32];
static uint32_t device_count;
static const struct in6_addr everyone={{{0xff,0x02,0,0,0,0,0,0,0,0,0,0,0,0,0,1}}};

static uint64_t clock_ns(void){ return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
uint64_t memory;
static int64_t stop_reason;
static void farewell_text(int64_t code,char *out,size_t room){
  if(code==SIGHUP)snprintf(out,room,"SIGHUP: its heartbeat restarts it");
  else if(code==SIGTERM)snprintf(out,room,"SIGTERM: its heartbeat's release or stop");
  else if(code>0)snprintf(out,room,"signal %lld",(long long)code);
  else snprintf(out,room,"no reason given");
}
struct known known[MESH_KNOWN];
void wake(void){
  struct kevent64_s event;EV_SET64(&event,0,EVFILT_USER,0,NOTE_TRIGGER,0,0,0,0);
  kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
}
/* design/transport.md: a port halted from any thread (a peer interface's termination heard, a provider's fault
   absorbed; atomics alone): its pairing lost, so no job of it makes a further call (B10), then no lane of its link
   served, both sequentially consistent (link_turn's connect stores served and reads the pairing again), and its bit set
   in `which` for the event loop: the child's report to its parent (child_reports), the parent's conclusion
   (terminations_turn) */
_Static_assert(MESH_PORTS<=32,"a bit a port");
static void port_halt(struct port *port,_Atomic uint32_t *which){
  atomic_fetch_add(&port->link.pairing,1);
  atomic_store(&port->link.served,0);
  atomic_fetch_or(which,UINT32_C(1)<<port->index);
}

/* a lane's failure, found by a progress thread: served no more, and the link lost by the event loop on reading
   `broken` (link_turn) */
static void lane_lost(struct port *port,uint32_t lane,int64_t code){
  struct link *l=&port->link;
  if(atomic_fetch_and_explicit(&l->served,~(UINT32_C(1)<<lane),memory_order_acq_rel)>>lane&1)
    say("link %u (%s): lane %u failed: %s; the link toward node %u lost\n",port->index,port->name,lane,
        strerror((int)(code>0?code:EIO)),port->peer);
  atomic_fetch_or_explicit(&l->broken,UINT32_C(1)<<lane,memory_order_release);
  wake();
}

/* the lane records of port p's lane pair s (mesh.h struct mesh_lanes, the bridge's own) */
static inline struct mesh_lanes *lanes_of(const struct port *port,uint32_t s){ return mesh_lanes_at(M,port->index*MESH_LINK_STREAMS+s); }
/* the SGE of an exact byte span of the region (a page's frames: within one registration extent, the pool's carve) */
_Static_assert(MESH_DMA_EXTENT==MESH_EXTENT,"span validation matches registration geometry");
static inline struct ibv_sge dma_span(const struct port *port,struct mesh_dma_span span){
  const uint64_t at=span.offset-M->data_off;
  return (struct ibv_sge){.addr=(uintptr_t)port->device.wire+at,.length=(uint32_t)span.bytes,
    .lkey=port->device.extent_mrs[at/MESH_EXTENT]->lkey};
}
static void link_failed(struct port *,int64_t,uint32_t);
/* the lane a completion's queue pair number names: one of the link's lanes, else MESH_LINK_LANES (dropped; a lane its
   plan leaves out has the number UINT32_MAX, which no queue pair has) */
static __attribute__((always_inline)) inline uint32_t link_lane(const struct link *l,uint32_t qp_num){
  uint32_t k=0;
  while(k<MESH_LINK_LANES && l->qpn[k]!=qp_num)k++;
  return k;
}
/* design/transport.md: the link's first failed completion, on any lane, or a failed poll: no lane of the
   link served from here, so nothing is posted on it again, and the link lost by the event loop on reading `broken`
   (link_turn) */
static void link_failed(struct port *port,int64_t code,uint32_t domain){
  struct link *l=&port->link;
  const uint32_t served=atomic_exchange_explicit(&l->served,0,memory_order_acq_rel);
  if(served)say("link %u (%s): a completion failed (code %lld, domain %u): no lane served, the link toward node %u lost\n",
                port->index,port->name,(long long)code,domain,port->peer);
  atomic_fetch_or_explicit(&l->broken,served,memory_order_release);
  wake();
}

/* ---- design/PAGE-TABLE-SPEC-20261122.md section 5: the switch's data path, a port child's part (mesh-switched.h) ----
   Each data lane of the link (MESH_LANE_FIRST + j) is the switch's lane port 2 MESH_LINK_STREAMS + j.  On it the child
   keeps receives posted (one frame count for every SEND and every receive on the lane, TN3205): filled from free pages
   of the lane's class, and each one a SEND fills posted again at once, into a free page or, where the class has none,
   into the page that landed (whose landing is then lost), so a lane's posted receives do not run out while every page
   of its class is used; it posts the lane's outbox, a page a SEND from the page's own memory; and gives its SENDs'
   completions to the switch's estimators, each dropping its page's use.  What lands is set in the page table and
   forwarded by its header; what this node's clients write is taken from their rings and carried.  Nothing is retained:
   what cannot go is counted (mesh-switched.h struct mesh_switch_counters), never retried.  The child's log ring,
   counters and hand are its port's */
static inline struct mesh_switch *switch_of(void){ return mesh_switch_of(M); }
static inline uint32_t switch_lane(const struct port *port,uint32_t j){ return port->index*2*MESH_LINK_STREAMS+j; }
static inline struct mesh_switch_hand *hand_of(const struct port *port){ return switch_of()->hand+port->index; }
/* data lane j's frames: a link's first data lane short (MESH_CLASS_SHORT), every other long (MESH_CLASS_LONG); both
   ends' plans alike, so both ends' lane j alike (TN3205) */
static inline uint64_t lane_frames(uint32_t j){ return j?MESH_CLASS_LONG:MESH_CLASS_SHORT; }
/* page pd made a receive of data lane j (its state RECEIVE on the switch's lane), before it is posted: a child's end
   gives it back from then on, posted or not (mesh-switched.h mesh_switch_receives_drop) */
static inline void receive_page(const struct port *port,uint32_t j,uint32_t pd){
  struct mesh_pd *p=mesh_pd_at(M,pd);
  atomic_store_explicit(&p->lane,switch_lane(port,j),memory_order_relaxed);
  atomic_store_explicit(&p->state,MESH_PD_RECEIVE,memory_order_release);
}
/* pages page[0 .. n) (each a receive of data lane j: receive_page) posted as the lane's receives, each the lane's
   frames from its page's start, as one list a post_recv (its call's cost the list's, not each receive's); those the
   provider did not take go back to the pool, and its error loses the lane: 0, or the error */
#define RECEIVE_BATCH 16u
static int receives_post(struct port *port,uint32_t j,const uint32_t *page,uint32_t n){
  struct link *l=&port->link;
  const uint32_t lane=MESH_LANE_FIRST+j;struct lane_state *state=l->lane+lane;
  const uint64_t bytes=lane_frames(j)*MESH_FRAME;
  const struct mesh_queue *queue=l->provider.queues+lane;
  struct ibv_recv_wr request[RECEIVE_BATCH];struct ibv_sge span[RECEIVE_BATCH];
  for(uint32_t k=0;k<n;k++){
    span[k]=dma_span(port,(struct mesh_dma_span){mesh_pd_at(M,page[k])->offset,bytes});
    request[k]=(struct ibv_recv_wr){.wr_id=((uint64_t)lane<<48)|page[k],.sg_list=span+k,.num_sge=1};
    if(k)request[k-1].next=request+k;
  }
  struct ibv_recv_wr *bad=NULL;
  const int error=queue->receive(queue->pair,request,&bad);
  const uint32_t made=error?(bad>=request && bad<request+n?(uint32_t)(bad-request):0):n;
  for(uint32_t k=made;k<n;k++)mesh_pool_push(M,page[k]);
  state->posted+=made;
  atomic_store_explicit(&lanes_of(port,j/2)->posts[1].posted[j&1],state->posted,memory_order_relaxed);
  if(error)lane_lost(port,lane,error<0?-error:error);
  return error;
}
/* the lane's receives filled from its class's free stack to as many as its queues' share takes: its first fill (each
   receive a SEND fills is posted again by link_receive, so the count stays there), and a receive a completion took
   that named no receive page; a stack found empty posts nothing more this pass and is counted (`starved`: the provider
   stalls the peer's SEND meanwhile) */
static void switch_receives(struct port *port,uint32_t j){
  struct lane_state *state=port->link.lane+MESH_LANE_FIRST+j;
  const uint64_t frames=lane_frames(j),most=state->frame_capacity/frames;
  while(state->posted-state->got<most){
    uint32_t page[RECEIVE_BATCH],n=0;
    int starved=0;
    while(n<RECEIVE_BATCH && state->posted+n-state->got<most){
      const uint32_t pd=mesh_pool_pop(M,(uint32_t)frames);
      if(pd==MESH_ABSENT){starved=1;break;}
      receive_page(port,j,pd);
      page[n++]=pd;
    }
    if(starved)MESH_SWITCH_COUNT(switch_of(),port->index,starved,1);
    if(!n || receives_post(port,j,page,n) || n<RECEIVE_BATCH)return;
  }
}
/* the lane's outbox posted: a page a SEND of the lane's frames from the page's start, while the lane's SEND gate (its
   queue's own capacity: a post past it fails) takes one more, each SEND's post time stored in its outbox cell; holes (a
   pusher's death settled) passed with no SEND */
static void switch_sends(struct port *port,uint32_t j){
  struct link *l=&port->link;
  struct mesh_switch *sw=switch_of();
  const uint32_t lane=MESH_LANE_FIRST+j,at=switch_lane(port,j);struct lane_state *state=l->lane+lane;
  struct mesh_switch_lane *x=sw->lane+at;
  const struct frames each={lane_frames(j)};
  const struct mesh_queue *queue=l->provider.queues+lane;
  /* every page the outbox has, as many as the lane's gate takes, one post_send for them all (the provider's call
     costs about 0.85 us whatever it carries: a request list a call, not a call a request) */
  enum { BATCH=16 };
  for(;;){
    uint64_t head=atomic_load_explicit(&x->head,memory_order_relaxed);
    uint32_t pd;
    while(mesh_switch_cell_at(sw,at,head,&pd,NULL) && pd==MESH_ABSENT)head++;
    atomic_store_explicit(&x->head,head,memory_order_relaxed);
    struct ibv_send_wr request[BATCH];struct ibv_sge span[BATCH];uint64_t pushed[BATCH];uint32_t n=0;
    while(n<BATCH && gate_fits(l->gate+lane,each)){
      if(!mesh_switch_cell_at(sw,at,head+n,&pd,pushed+n) || pd==MESH_ABSENT)break;
      span[n]=dma_span(port,(struct mesh_dma_span){mesh_pd_at(M,pd)->offset,each.n*MESH_FRAME});
      request[n]=(struct ibv_send_wr){.wr_id=gate_post(l->gate+lane,lane,each),.sg_list=span+n,.num_sge=1,
                                      .opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED};
      if(n)request[n-1].next=request+n;
      n++;
    }
    if(!n)return;
    struct ibv_send_wr *bad=NULL;
    const uint64_t posting=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    const int error=queue->send(queue->pair,request,&bad);
    const uint64_t posted=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    /* the requests before `bad` posted (all of them without an error) */
    const uint32_t made=error?(bad>=request && bad<request+n?(uint32_t)(bad-request):0):n;
    l->gate[lane].posted.n-=(uint64_t)(n-made)*each.n;
    MESH_SWITCH_COUNT(sw,port->index,post_call_ns,posted-posting);
    for(uint32_t k=0;k<made;k++){
      MESH_SWITCH_COUNT(sw,port->index,posts,1);MESH_SWITCH_COUNT(sw,port->index,post_wait_ns,posting>pushed[k]?posting-pushed[k]:0);
      mesh_switch_cell_posted(sw,at,head+k,posted);
      __atomic_store_n(&state->sent,state->sent+1,__ATOMIC_RELEASE);l->gate[lane].requests++;
    }
    atomic_store_explicit(&x->head,head+made,memory_order_relaxed);
    atomic_store_explicit(&lanes_of(port,j/2)->posts[0].posted[j&1],state->sent,memory_order_relaxed);
    if(error){lane_lost(port,lane,error<0?-error:error);return;}
    if(n<BATCH)return;
  }
}
/* design/transport.md: the link's SEND completions (a lane no longer served drops its own; a failed one fails the
   link, link_failed): each one's outbox cell given back and its page's use dropped, through the switch's estimators
   (its post, as its cell stored it, its service from the later of its post and the lane's previous completion, its
   completion), its frames completed through the gate */
static void link_send(struct port *port,uint32_t served){
  struct link *l=&port->link;
  struct mesh_switch *sw=switch_of();
  struct ibv_wc done[16];
  const int count=l->provider.poll(l->provider.sent,16,done);
  if(count<0){link_failed(port,EIO,2);return;}
  const uint64_t at=count?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
  for(int i=0;i<count;i++){
    const uint32_t lane=link_lane(l,done[i].qp_num);
    if(lane>=MESH_LINK_LANES || !(served>>lane&1))continue;
    if(done[i].status){link_failed(port,(int64_t)done[i].status,2);return;}
    gate_complete(l->gate+lane,done[i].wr_id);
    if(lane<MESH_LANE_FIRST)continue;
    struct lane_state *state=l->lane+lane;
    const uint32_t j=lane-MESH_LANE_FIRST;
    uint64_t posted=at;
    const uint32_t pd=mesh_switch_completed(sw,switch_lane(port,j),&posted);
    const uint64_t start=posted>state->last?posted:state->last;
    mesh_switch_sent(sw,switch_lane(port,j),lane_frames(j)*MESH_FRAME,posted,start,at);
    mesh_pd_drop(M,pd);
    MESH_SWITCH_COUNT(sw,port->index,sent,1);MESH_SWITCH_COUNT(sw,port->index,send_ns,at-posted);
    state->last=at;state->done++;
    struct mesh_lanes *x=lanes_of(port,j/2);
    atomic_store_explicit(&x->send_ns,atomic_load_explicit(&x->send_ns,memory_order_relaxed)+(at-posted),memory_order_relaxed);
    atomic_store_explicit(&x->sends,atomic_load_explicit(&x->sends,memory_order_relaxed)+1,memory_order_relaxed);
    atomic_store_explicit(&x->posts[0].done[j&1],state->done,memory_order_relaxed);
  }
}
/* the link's receive completions, each data lane's receive posted again at once, so a lane's posted receives stay as
   many as its fill made them whatever the pool has (TN3205: a receive for every SEND the peer posts): into a page
   popped from the lane's class, and the page that landed landed (mesh-switched.h mesh_switch_landed: its lines taken,
   its destinations here set in the table, the rest switched on); where the class's free stack is empty, into the page
   that landed, which is then not landed, counted (`lost`).  The receives of a poll posted as one list a lane
   (receives_post) */
static void link_receive(struct port *port,uint32_t served){
  struct link *l=&port->link;
  struct mesh_switch *sw=switch_of();
  struct ibv_wc done[RECEIVE_BATCH];
  const int count=l->provider.poll(l->provider.completion,RECEIVE_BATCH,done);
  if(count<0){link_failed(port,EIO,3);return;}
  if(count)atomic_thread_fence(memory_order_acquire);
  const uint64_t at=count?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
  uint32_t again[MESH_LINK_LANES-MESH_LANE_FIRST][RECEIVE_BATCH],agains[MESH_LINK_LANES-MESH_LANE_FIRST]={0};
  for(int i=0;i<count;i++){
    const uint32_t lane=link_lane(l,done[i].qp_num);
    if(lane>=MESH_LINK_LANES || !(served>>lane&1))continue;
    uint64_t status_opcode;
    memcpy(&status_opcode,&done[i].status,sizeof status_opcode);
    if((uint32_t)status_opcode){link_failed(port,(uint32_t)status_opcode,2);return;}
    if(!((status_opcode>>32)&IBV_WC_RECV) || lane<MESH_LANE_FIRST)continue;
    struct lane_state *state=l->lane+lane;
    const uint32_t j=lane-MESH_LANE_FIRST,pd=(uint32_t)(done[i].wr_id&UINT32_MAX);
    state->got++;
    atomic_store_explicit(&lanes_of(port,j/2)->posts[1].done[j&1],state->got,memory_order_relaxed);
    if(!mesh_pd_valid(M,pd) || atomic_load_explicit(&mesh_pd_at(M,pd)->state,memory_order_relaxed)!=MESH_PD_RECEIVE)continue;
    const uint32_t fresh=mesh_pool_pop(M,(uint32_t)lane_frames(j));
    if(fresh==MESH_ABSENT){again[j][agains[j]++]=pd;MESH_SWITCH_COUNT(sw,port->index,lost,1);continue;}
    receive_page(port,j,fresh);
    again[j][agains[j]++]=fresh;
    mesh_switch_landed(M,sw,switch_lane(port,j),pd,at,port->index,hand_of(port));
  }
  for(uint32_t j=0;j<MESH_LINK_LANES-MESH_LANE_FIRST;j++)if(agains[j])receives_post(port,j,again[j],agains[j]);
}
/* the writes this node's clients put on the rings this child takes (mesh-switched.h struct mesh_switch `taker`: the
   rings its parent names its port's, switch_takers), taken and carried, at most SWITCH_BURST a pass, each of those
   rings with writes on it looked at from the slot after the one where this child began its last pass, whichever
   client the slot is now (a write is carried on its own ring use, its writer there or exited).  As the parent names
   another port for a ring this child may take from it a moment longer, and the ring's compare-exchange gives each
   write to one child */
#define SWITCH_BURST 16u
static void switch_takes(struct port *port){
  struct mesh_switch *sw=switch_of();
  struct link *l=&port->link;
  uint32_t taken=0;uint64_t at=0;
  for(uint32_t k=0;k<MESH_CLIENTS && taken<SWITCH_BURST;k++){
    const uint32_t slot=(l->take_from+k)%MESH_CLIENTS;
    if((uint32_t)atomic_load_explicit(&sw->taker[slot],memory_order_relaxed)!=port->index || !mesh_ring_pending(mesh_ring_at(M,slot)))continue;
    if(!at)at=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    taken+=mesh_switch_ring_dequeue(M,sw,slot,SWITCH_BURST-taken,at,port->index,hand_of(port));
  }
  l->take_from=(l->take_from+1)%MESH_CLIENTS;
}

/* the progress thread's tell of the guard value it stops at (`guarded`), its raiser woken (port_wait) */
static void port_tell(_Atomic uint64_t *word,uint64_t value){
  atomic_store_explicit(word,value,memory_order_release);
  os_sync_wake_by_address_all(word,sizeof *word,OS_SYNC_WAKE_BY_ADDRESS_NONE);
}
/* design/PAGE-TABLE-SPEC-20261122.md section 5: one pass of a port's progress thread over the lanes the link serves
   (`served` read once a pass: none posts on a lane its link no longer serves): the receive completions landed
   (link_receive), the receives topped up, the clients' writes taken (switch_takes), every data lane's outbox posted,
   the SEND completions (link_send), the pass counted (`passes`) */
static void port_pass(struct port *port){
  /* one pass in 1024 timed (counters pass_ns: its time summed and the passes timed; part_ns: each part's time) */
  static __thread uint32_t timing;
  const int timed=!(++timing&1023);
  uint64_t t[6]={0};
  if(timed)t[0]=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  struct link *l=&port->link;
  struct mesh_switch *sw=switch_of();
  const uint32_t served=atomic_load_explicit(&l->served,memory_order_acquire);
  const uint32_t lanes=2*plan_streams(l->plan);
  if(served)link_receive(port,served);
  if(timed)t[1]=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  for(uint32_t j=0;j<lanes;j++)if(served>>(MESH_LANE_FIRST+j)&1)switch_receives(port,j);
  if(timed)t[2]=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  if(served>>MESH_LANE_FIRST)switch_takes(port);
  if(timed)t[3]=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  for(uint32_t j=0;j<lanes;j++)if(served>>(MESH_LANE_FIRST+j)&1)switch_sends(port,j);
  if(timed)t[4]=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  if(served)link_send(port,served);
  MESH_SWITCH_COUNT(sw,port->index,passes,1);   /* the pass time is the wall time over these (the thread spins) */
  if(timed){
    t[5]=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    MESH_SWITCH_COUNT(sw,port->index,pass_ns[0],t[5]-t[0]);MESH_SWITCH_COUNT(sw,port->index,pass_ns[1],1);
    for(uint32_t k=0;k<5;k++)MESH_SWITCH_COUNT(sw,port->index,part_ns[k],t[k+1]-t[k]);
  }
  struct mesh_lanes *counted=lanes_of(port,0);
  atomic_store_explicit(&counted->passes,atomic_load_explicit(&counted->passes,memory_order_relaxed)+1,memory_order_relaxed);
}
/* design/transport.md section 14: a port's progress thread, its SENDs' and its receives': it passes without pause while
   its link serves a lane (rubric M8: the queues read as fast as a hardware thread allows), and waits on the port's
   guard word while it serves none, looking again every MESH_PAUSE_NS (the stop's `closing`) */
static void *port_progress(void *argument){
  struct port *port=argument;
  pthread_setname_np("mesh.rdma");
  /* fault_heard: this thread its port's, and its handler on a stack of its own */
  stack_t alternate={.ss_sp=malloc(SIGSTKSZ),.ss_size=SIGSTKSZ};
  if(alternate.ss_sp)sigaltstack(&alternate,NULL);
  atomic_store_explicit(&port->thread[0],pthread_self(),memory_order_release);
  mesh_switch_home=port->index;   /* this thread posts its own port's lanes (mesh-switched.h mesh_switch_lanes_among_for) */
  while(!atomic_load_explicit(&port->closing,memory_order_acquire)){
    /* while the port's guard stands, or its link serves no lane, no pass (no post on any queue pair of its device) and
       no lane read: the thread waits on the guard word (design/transport.md B13: a guarded call in the driver costs no
       core), woken by its raising and lowering, by the link's coming up and by a halt or the child's end (guard_wake),
       looking again every MESH_PAUSE_NS (the stop's `closing`), the guard told at each waking, the tell waking its
       raiser (port_tell) */
    const uint64_t guard=atomic_load_explicit(&port->guard,memory_order_acquire);
    if(guard&1 || !atomic_load_explicit(&port->link.served,memory_order_acquire)){
      /* a fault absorbed on the port (fault_heard: the handler, which may make no call but mmap, munmap and sigaction,
         halted the port) wakes its child's loop, which tells its parent at once, not at its next tick */
      if(atomic_load_explicit(&faulted,memory_order_relaxed))wake();
      if(guard&1)port_tell(&port->guarded,guard);
      /* the wait on the word that parks the thread: the guard while it stands, else the lanes served while they are
         none, so a change of that word between the read above and the wait ends the wait at once (a wait on the guard
         for a link's coming up slept MESH_PAUSE_NS where the lanes were served between the two) */
      if(guard&1)os_sync_wait_on_address_with_timeout(&port->guard,guard,sizeof port->guard,OS_SYNC_WAIT_ON_ADDRESS_NONE,
                                                      OS_CLOCK_MACH_ABSOLUTE_TIME,MESH_PAUSE_NS);
      else os_sync_wait_on_address_with_timeout(&port->link.served,0,sizeof port->link.served,OS_SYNC_WAIT_ON_ADDRESS_NONE,
                                                OS_CLOCK_MACH_ABSOLUTE_TIME,MESH_PAUSE_NS);
      continue;
    }
    port_pass(port);
  }
  return NULL;
}
/* design/transport.md B13: the progress thread parked on the port's guard looks again */
static void guard_wake(struct port *port){
  os_sync_wake_by_address_all(&port->guard,sizeof port->guard,OS_SYNC_WAKE_BY_ADDRESS_NONE);
  os_sync_wake_by_address_all(&port->link.served,sizeof port->link.served,OS_SYNC_WAKE_BY_ADDRESS_NONE);
}
/* design/transport.md: the port's progress thread started, once; 0, or pthread_create's error */
static int port_threads(struct port *port){
  if(port->threads)return 0;
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_set_qos_class_np(&attributes,QOS_CLASS_USER_INTERACTIVE,0);
  pthread_t created;
  const int error=pthread_create(&created,&attributes,port_progress,port);
  if(!error)port->threads=1;
  pthread_attr_destroy(&attributes);
  return error;
}
/* design/transport.md: the raiser of a guard parked, never spinning, until the port's progress thread, where started,
   has told (`told`: `guarded`, the guard value it stops at) `value`, woken by the thread's tell (port_tell) alone, the
   guard's waiters woken as it begins and at each waking; a call that does not return is the parent's to conclude
   around (ports_stalled) */
static void port_wait(struct port *port,_Atomic uint64_t *told,uint64_t value){
  for(uint64_t seen;port->threads && (seen=atomic_load_explicit(told,memory_order_acquire))<value;){
    guard_wake(port);
    os_sync_wait_on_address(told,seen,sizeof *told,OS_SYNC_WAIT_ON_ADDRESS_NONE);
  }
}
/* design/RDMA-KERNEL-RECOVERY.md: the guard of a port's device, on the port's queue, around a change of its verbs
   objects (queue pairs and completion queues made, connected or destroyed): raised, it returns once each started
   progress thread stops (it told the guard's value before its next pass, and passes no more while the guard stands),
   so no queue pair of the device is posted on while another is made, changed or destroyed (Apple's provider: a post on
   a long-lived queue pair across another's teardown stores to the device's shared doorbell mapping after it moved,
   tbt_post_recv); lowered, the threads parked on it woken and going on (design/transport.md B13).  It stands for every
   change once the port's threads run, the link lanes' long-lived queue pairs being posted on while the link is up.
   Raised: 1; 0 where no progress thread runs (port_wait waits on the thread's tell alone) */
static void guard_lower(struct port *port,int raised){
  if(raised>0){atomic_fetch_add_explicit(&port->guard,1,memory_order_release);guard_wake(port);}
}
static int guard_raise(struct port *port){
  if(!port->threads)return 0;
  port_wait(port,&port->guarded,atomic_fetch_add_explicit(&port->guard,1,memory_order_acq_rel)+1);
  return 1;
}

/* design/transport.md: a message to the port's peer, queued behind what the socket has not taken yet (a node's record
   in place of its older one: port_coalesce); a send that fails, or a queue the process cannot grow, is the link's loss */
static void port_watch(struct port *port,int filter){
  struct kevent64_s event;EV_SET64(&event,(uint64_t)port->control,filter,EV_ADD|(filter==EVFILT_WRITE?EV_ONESHOT:0),0,0,port->index,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))port_lost(port,errno);
}
static void port_flush(struct port *port){
  while(port->control>=0 && port->out_sent<port->out_length){
    ssize_t sent=send(port->control,port->out+port->out_sent,port->out_length-port->out_sent,0);
    if(sent<0 && (errno==EAGAIN || errno==EINTR)){
      if(!port->writing){port->writing=1;port_watch(port,EVFILT_WRITE);}
      return;
    }
    if(sent<=0){port_lost(port,sent?errno:EIO);return;}
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
  if(port->control<0 || (said.kind!=HELLO && !port->greeted) || port_coalesce(port,&said,payload,bytes))return;
  said.magic=XMAGIC;said.version=MESH_VERSION;said.node=M->node;said.boot=M->boot;said.port=port->index;said.bytes=bytes;
  said.lid=port->lid;memcpy(said.gid,port->gid,16);
  size_t length=port->out_length+sizeof said+bytes;
  if(length>port->out_room){
    unsigned char *grown=realloc(port->out,length*2);
    if(!grown){port_lost(port,ENOMEM);return;}
    port->out=grown;port->out_room=length*2;
  }
  memcpy(port->out+port->out_length,&said,sizeof said);
  if(bytes)memcpy(port->out+port->out_length+sizeof said,payload,bytes);
  port->out_length=length;
  port_flush(port);
}
/* each client slot's owner as clients_turn last read it */
uint64_t owners[MESH_CLIENTS];
/* design/transport.md: this node's record, its region, memory and model with its ports and health */
char model[MESH_MODEL_BYTES];
/* design/node-addressed-transfers.md: this node's reach as its record tells it (mesh-records.c record_changed): an
   entry a cable whose link is up with lanes, the peer node and that cable's port bandwidth, at most `room`; their
   count.  mesh-dataflow.c mesh_join counts a pair's entries as its parallel cables (the fewer of the two ends'), so a
   pair of three cables is three links of the map, never one link at their summed bandwidth.  The entries name nodes
   and bandwidths, never ports: which cable carries a page is the switch's, as it moves (mesh-switched.h) */
uint32_t record_reach(struct mesh_reach *reach,uint32_t room){
  uint32_t count=0;
  for(uint32_t p=0;p<port_count && count<room;p++)
    if(ports[p].link.state==LINK_UP && ports[p].peer && plan_streams(ports[p].link.plan))
      reach[count++]=(struct mesh_reach){.node=ports[p].peer,.padding=1,.bandwidth=ports[p].bandwidth};
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

/* design/transport.md section 14: a link job on the port's queue, guarded (no queue pair of the device is posted on
   while one is made or connected: guard_raise), its verbs calls made on hearing the peer, from the peer's numbers as the
   job was started (`numbers`, its LID and GID among them: the event loop takes the peer's LINK and hello meanwhile), and
   none once its pairing is lost (B10: read as the guard stands and between its steps), each in the port's child and
   once in its life
   (design/transport.md section 14, "A process a port"): the lanes made (LINK_JOB_MAKE: the port's own address read
   again, then a UC queue pair in INIT for each lane of the link's plan, d6dceac's verbs_up, each lane's share of the
   completion queues the plan's streams' lanes alone dividing them, lanes 0 and 1 posting nothing:
   design/HETEROGENEOUS-LINKS.md; EALREADY for a make asked
   again); or connected to the peer's (LINK_JOB_CONNECT, the plan's lanes alone: RTR, then RTS, each lane's number and
   gate set, every lane's progress made new, and the port's progress thread started).  No queue pair is moved to
   another state, destroyed or forgotten after it: the in-place RESET of 127 panicked the M5 in AppleThunderboltRDMA on its
   first run (2026-10-19, design/RDMA-KERNEL-RECOVERY.md), and the kernel releases every object at the child's exit, its
   one end */
struct link_numbers { uint32_t peer[MESH_LINK_LANES][2],pairing; uint16_t lid; uint8_t gid[16]; };
static int link_lapsed(const struct link *l,const struct link_numbers *numbers){
  return atomic_load_explicit(&l->pairing,memory_order_relaxed)!=numbers->pairing?ECANCELED:0;
}
static int link_make(struct port *port,const struct link_numbers *numbers){
  struct link *l=&port->link;
  struct mesh_verbs *p=&l->provider;
  struct ibv_port_attr attributes;
  int error=0;
  if(p->queues)return EALREADY;
  /* design/transport.md: the port's own address (its attributes and GID) read again at the make, as its opening read
     it: a peer's restart can move the port's GID table, and LINKs naming the address of the port's opening failed every
     lane's connection at the peer with EIO (2026-10-19, after the Mini's power loss) */
  if(ibv_query_port(port->device.context,1,&attributes))return errno?errno:EIO;
  port->device.port=attributes;
  if((error=device_gid(&port->device,port->name)) || (error=link_lapsed(l,numbers)) || (error=pair_up(p,l->plan)))return error;
  l->share=(struct frames){(uint64_t)MIN(p->completion->cqe,p->sent->cqe)/MAX(plan_lanes(l->plan>>MESH_LANE_FIRST),1u)};
  return 0;
}
static void lane_reset(struct port *port,uint32_t k);
/* design/transport.md section 14: stream s's identity, alike at both ends: FNV-1a over its first lane's two ends, each
   its port's GID and its queue pair's number, the lesser end first; never 0 */
static uint64_t stream_wire(const struct port *port,const struct link_numbers *numbers,uint32_t s){
  const uint32_t k=MESH_LANE_FIRST+2*s*MESH_STREAM_PAIRS;
  uint8_t end[2][20];
  memcpy(end[0],port->gid,16);memcpy(end[0]+16,&port->link.qpn[k],4);
  memcpy(end[1],numbers->gid,16);memcpy(end[1]+16,&numbers->peer[k][0],4);
  const int first=memcmp(end[0],end[1],20)>0;
  uint64_t h=UINT64_C(14695981039346656037);
  for(int e=0;e<2;e++)
    for(int i=0;i<20;i++){h^=end[e^first][i];h*=UINT64_C(1099511628211);}
  return h?h:1;
}
static int link_connect(struct port *port,const struct link_numbers *numbers){
  struct link *l=&port->link;
  struct mesh_verbs *p=&l->provider;
  const uint32_t plan=l->plan;
  int error=0;
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1)error=queue_receive(p,(int)k,numbers->peer[k],numbers->lid,numbers->gid);
  if(!error)error=link_lapsed(l,numbers);
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1)error=queue_send(p,(int)k);
  const struct frames share={MAX(l->share.n,1)};
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++){
    l->qpn[k]=plan>>k&1?p->queues[k].pair->qp_num:UINT32_MAX;
    l->gate[k]=(struct send_gate){.capacity=plan>>k&1?frames_min((struct frames){p->queues[k].send_capacity},share):(struct frames){0}};
  }
  if(!error)error=link_lapsed(l,numbers);
  for(uint32_t k=MESH_LANE_FIRST;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1)lane_reset(port,k);
  for(uint32_t s=0;s<plan_streams(plan) && !error;s++)l->wire[s]=stream_wire(port,numbers,s);
  return error;
}
/* design/transport.md section 14, "A process a port": the link's lane records opened as its lanes connect, by the
   child before any lane is served: each lane's requests posted and completed each way 0 (its queue pairs are new),
   this child their poster (`child`, `child_start`), the peer node, the link's bandwidth and identity told; then each
   data lane the switch's, live toward the peer, its outbox first emptied of what was pushed onto it after its last
   child's end (mesh-switched.h mesh_switch_lane_drop: pages for the lane's last link, each counted `lost` in the
   port's counters: its progress thread passes no lane yet) */
static void lanes_open(struct port *port,uint32_t node,uint64_t bandwidth){
  struct link *l=&port->link;
  for(uint32_t s=0;s<plan_streams(l->plan);s++){
    struct mesh_lanes *x=lanes_of(port,s);
    _Atomic uint64_t *words[]={&x->posts[0].posted[0],&x->posts[0].posted[1],&x->posts[0].done[0],&x->posts[0].done[1],
                               &x->posts[1].posted[0],&x->posts[1].posted[1],&x->posts[1].done[0],&x->posts[1].done[1]};
    for(size_t i=0;i<sizeof words/sizeof *words;i++)atomic_store_explicit(words[i],0,memory_order_relaxed);
    x->child_start=mesh_started(getpid());
    atomic_store_explicit(&x->child,(uint32_t)getpid(),memory_order_release);
    atomic_store_explicit(&x->peer,node,memory_order_relaxed);
    atomic_store_explicit(&x->bandwidth,bandwidth,memory_order_relaxed);
    atomic_store_explicit(&x->wire,l->wire[0],memory_order_relaxed);
    atomic_thread_fence(memory_order_release);
    atomic_store_explicit(&x->live,1,memory_order_release);
  }
  struct mesh_switch *sw=switch_of();
  for(uint32_t j=0;j<2*plan_streams(l->plan);j++){
    struct mesh_switch_lane *x=sw->lane+switch_lane(port,j);
    mesh_switch_lane_drop(M,sw,switch_lane(port,j),port->index);
    atomic_store_explicit(&x->peer,node,memory_order_relaxed);
    atomic_store_explicit(&x->frames,lane_frames(j),memory_order_relaxed);
    atomic_store_explicit(&x->live,1,memory_order_release);
  }
}
static void link_job(struct port *port,int job){
  struct link *l=&port->link;
  struct link_numbers numbers={.pairing=atomic_load_explicit(&l->pairing,memory_order_relaxed),.lid=port->heard.lid};
  memcpy(numbers.peer,l->peer,sizeof numbers.peer);memcpy(numbers.gid,port->heard.gid,sizeof numbers.gid);
  l->job=job;l->job_pairing=numbers.pairing;l->job_at=now;atomic_store_explicit(&l->done,0,memory_order_relaxed);
  dispatch_async(port->work,^{
    atomic_store_explicit(&port->thread[1],pthread_self(),memory_order_release);
    const int raised=guard_raise(port);
    int result=link_lapsed(l,&numbers);
    if(!result)result=job==LINK_JOB_MAKE?link_make(port,&numbers):link_connect(port,&numbers);
    if(!result && job==LINK_JOB_CONNECT)result=port_threads(port);
    guard_lower(port,raised);
    l->result=result;
    atomic_store_explicit(&port->thread[1],NULL,memory_order_release);
    atomic_store_explicit(&l->done,1,memory_order_release);
    wake();
  });
}
/* a LINK telling the peer every lane of this end's pairing, the lanes of its plan (`count` theirs): each one's queue pair
   number and first sequence number, as its child told them (CHILD_MADE), 0 for a lane the plan leaves out */
static void link_tell(struct port *port){
  const struct link *l=&port->link;
  struct message said={.kind=LINK,.count=plan_lanes(l->plan)};
  for(uint32_t k=0;k<MESH_LINK_LANES;k++)if(l->plan>>k&1){said.pair[k][0]=l->qpn[k];said.pair[k][1]=(l->psn+k)&0xffffff;}
  port_say(port,said,NULL,0);
}
/* the link of a port lost (its peer's loss, a failure, the bridge stopping): its lanes served no more and left as they
   are, no verbs call made, and, once connected, never paired again in this bridge's life (design/transport.md section
   14); a job of this pairing still on the port's queue makes no call and is dropped as it ends (B10) */
static void link_lost(struct port *port){
  struct link *l=&port->link;
  atomic_fetch_add_explicit(&l->pairing,1,memory_order_relaxed);l->state=LINK_DOWN;l->heard=0;
  atomic_store_explicit(&l->served,0,memory_order_release);
  atomic_store_explicit(&l->broken,0,memory_order_relaxed);
}
/* a lane's progress made new at its connect: nothing posted, and its receives its queues' share */
static void lane_reset(struct port *port,uint32_t k){
  struct link *l=&port->link;
  const struct frames room=frames_min(l->share,(struct frames){l->provider.queues[k].receive_capacity});
  l->lane[k]=(struct lane_state){.frame_capacity=room.n};
  atomic_thread_fence(memory_order_release);
}
/* the lane records of a port (`port` NULL: of every port) live no more, as its link is lost or the bridge stops, and
   its switch lanes with them: no draw takes them, and what their outboxes have is dropped as their child ends
   (switch_child_free) */
void lanes_cancel(const struct port *port){
  for(uint32_t p=0;p<port_count;p++){
    if(port && ports+p!=port)continue;
    for(uint32_t j=0;j<2*MESH_LINK_STREAMS;j++)atomic_store_explicit(&switch_of()->lane[switch_lane(ports+p,j)].live,0,memory_order_release);
    for(uint32_t s=0;s<MESH_LINK_STREAMS;s++)atomic_store_explicit(&lanes_of(ports+p,s)->live,0,memory_order_release);
  }
}
/* design/transport.md section 14: the link's turn, on the event loop: a job's end taken (made: LINK told the peer every
   lane; connected: up, every lane served, the stream live, the clients' rings' takers named again with its port among
   them, switch_takers); a job that failed loses the link (port_lost); while it is
   up, a lane a progress thread found failed (`broken`) loses it; then, while no port's job is out past MESH_PLATFORM_NS (B12) and once the port's last
   quiet window has ended (B9), the lanes made once the port is paired (`lanes`: its child asked for them), and connected
   once both ends' are made, each by the port's child and once in its life: a link lost once its lanes were asked for
   pairs again with the port's next child (design/transport.md section 14, "A process a port": port_lost, child_turn,
   child_reaped) */
static void child_job(struct port *port,int job);
static void switch_takers(void);
static void link_turn(struct port *port){
  struct link *l=&port->link;
  if(l->job){
    if(!atomic_load_explicit(&l->done,memory_order_acquire))return;
    const int job=l->job,result=l->result;
    l->job=0;
    if(l->job_pairing!=atomic_load_explicit(&l->pairing,memory_order_relaxed)){}
    else if(result){
      say("link %u (%s): lanes toward node %u not %s: %s; paired again with the port's next child\n",port->index,port->name,
          port->peer,job==LINK_JOB_MAKE?"made":"connected",strerror(result));
      port_lost(port,result);
      return;
    }
    else if(job==LINK_JOB_MAKE){l->state=LINK_MADE;link_tell(port);}
    else{
      l->state=LINK_UP;
      say("link %u (%s): %u lanes up toward node %u (plan %#x: %u stream%s), each lane's share %llu frames\n",
          port->index,port->name,plan_lanes(l->plan),port->peer,l->plan,plan_streams(l->plan),plan_streams(l->plan)==1?"":"s",
          (unsigned long long)l->share.n);
      record_changed();
      switch_takers();
    }
  }
  const int quiet=now<port->quiet_at;
  if(!port->greeted)return;
  if(l->state==LINK_UP){
    const uint32_t broken=atomic_exchange_explicit(&l->broken,0,memory_order_acq_rel);
    if(broken){
      say("link %u (%s): lanes %#x toward node %u failed: the link lost\n",port->index,port->name,broken,port->peer);
      port_lost(port,EIO);
      return;
    }
  }
  if(l->job || quiet)return;
  if(l->state==LINK_DOWN){
    l->lanes=1;
    child_job(port,LINK_JOB_MAKE);
  }
  else if(l->state==LINK_MADE && l->heard){l->state=LINK_CONNECTING;child_job(port,LINK_JOB_CONNECT);}
}
/* design/transport.md B12 (22891630^ mesh-flight.h's pending-operation age, per port: its dispatch's
   `job_at` in place of flight_begin's stamp, its `done` in place of flight_end's): a port whose open's or link's job
   is still out MESH_PLATFORM_NS after it began is concluded around it, once: open 0 and port_lost (ETIMEDOUT: its
   socket closed, the map's phase STOPPED), no verbs call.  The call itself cannot be ended.  While any port's job is so out the devices are not read (`stalled`: ports_find, the parent's own driver
   entry), and the port itself takes no job until the call returns (`stuck`, its job's word); each port's calls are its
   own child's, a process of its own (section 14), so one port's stuck call stalls no other port's (a stall of every
   port's was 10-18's, when one process made every port's calls).  The child is told by port_lost's CHILD_HALT (the
   pairing moved, B10) and, where its lanes were asked for, ended by child_turn's CHILD_END once no job of the port is
   out; the port stays `stuck` until the call returns, then opens again (`opening`, port_open past its quiet window,
   B9; an open's own return is port_opened's) and so takes jobs again.  Read on the event loop's turn */
static void ports_stalled(void){
  int any=0;
  for(uint32_t p=0;p<port_count;p++){
    struct port *port=ports+p;
    const struct link *l=&port->link;
    const int opening=port->job && !atomic_load_explicit(&port->done,memory_order_acquire) && now-port->job_at>=MESH_PLATFORM_NS;
    const int linking=l->job && !atomic_load_explicit(&l->done,memory_order_acquire) && now-l->job_at>=MESH_PLATFORM_NS;
    const int out=opening||linking;
    any|=out;
    if(port->stuck){
      if(!out){port->stuck=0;if(!port->job)port->opening=1;}
      continue;
    }
    if(!out)continue;
    port->stuck=1;
    say("link %u (%s): its %s out %.1f s: its port concluded around it (design/transport.md)\n",port->index,
             port->name,opening?"open's job":l->job==LINK_JOB_MAKE?"make job":"connect job",
             (double)(now-(opening?port->job_at:l->job_at))/1e9);
    port->open=0;
    port_lost(port,ETIMEDOUT);
  }
  atomic_store_explicit(&stalled,any,memory_order_relaxed);
}
/* a peer's LINK: every lane of its pairing and the address they were made on (its port's LID and GID as its make read
   them, which its hello's may predate: the connect's address, and an end of each stream's identity alike at both ends,
   stream_wire), taken once, before this end's connect (a LINK heard again is let be: the first stands), its lanes this
   end's plan's (both ends plan from the same two budgets: design/HETEROGENEOUS-LINKS.md); one naming other lanes loses
   the link, since a lane can be connected only to a queue pair its peer made (a port whose child was asked for its
   lanes greets no peer once its link is lost until its next child opens it: design/transport.md section 14, "A
   process a port") */
static void link_heard(struct port *port,const struct message *heard){
  struct link *l=&port->link;
  if(l->heard || (l->state!=LINK_DOWN && l->state!=LINK_MADE))return;
  if(heard->count!=plan_lanes(l->plan)){
    say("link %u (%s): node %u's LINK names %u lanes, this end's plan %u (plan %#x): %s\n",port->index,port->name,port->peer,heard->count,
        plan_lanes(l->plan),l->plan,port->heard.count?"the two ends planned apart":"its bridge states no budget and makes every lane, "
        "and this port has fewer: the link pairs once that node runs a bridge that states its budget");
    port_lost(port,EPROTO);
    return;
  }
  memcpy(l->peer,heard->pair,sizeof l->peer);
  port->heard.lid=heard->lid;memcpy(port->heard.gid,heard->gid,16);
  l->heard=1;
  link_turn(port);
}


/* ---- design/PAGE-TABLE-SPEC-20261122.md: the switch, the parent's part -----------------------------------------
   The parent carves the node's pool once and makes its switch at its start, keeps the switch's view of the fabric as
   the records give it, keeps the node table (a node the map has, live while it does), and runs the refresh: a thread
   of its own that recomputes the routing weights from the estimators every SWITCH_REFRESH_NS (mesh-switched.h
   mesh_switch_refresh), so no page and no child ever waits on it.  A port's child exited, what it had is given back
   here (switch_child_free): nothing is sent again, and each page it had is counted `lost` */
#define SWITCH_REFRESH_NS ((uint64_t)MESH_SWITCH_REFRESH_NS)
#define SWITCH_DRAWS MESH_SWITCH_DRAWS
#define SWITCH_ETA MESH_SWITCH_ETA
/* the node table's entry of `node`, made (generation 1) as the map first has the node; MESH_ABSENT where every entry
   has another node (the refresh thread's alone) */
static uint32_t node_slot(uint32_t node){
  uint32_t n=0,vacant=MESH_ABSENT;
  for(;n<MESH_NODES && atomic_load(&mesh_node_at(M,n)->node)!=node;n++)if(vacant==MESH_ABSENT && !atomic_load(&mesh_node_at(M,n)->node))vacant=n;
  if(n==MESH_NODES && (n=vacant)==MESH_ABSENT){say("bridge: no node table entry for node %u (mesh.h MESH_NODES)\n",node);return MESH_ABSENT;}
  struct mesh_node *x=mesh_node_at(M,n);
  if(!atomic_load(&x->generation))atomic_store(&x->generation,1);
  atomic_store_explicit(&x->node,node,memory_order_release);
  return n;
}
/* the frames the switch's lanes have queued and not completed (struct hdr `busy`) */
uint64_t lanes_busy(void){
  const struct mesh_switch *sw=mesh_switch_of(M);
  uint64_t busy=0;
  if(sw->magic!=MESH_SWITCH_MAGIC)return 0;
  for(uint32_t l=0;l<MESH_SWITCH_LANES;l++)busy+=atomic_load(&sw->lane[l].queued)/MESH_FRAME;
  return busy;
}
/* the payload bytes of the pool's pages, what this node's clients may fill (its record's `region`) */
uint64_t pool_bytes(void){
  const struct mesh_pool *pool=mesh_pool(M);
  uint64_t bytes=0;
  for(uint32_t c=0;c<MESH_CLASSES;c++)bytes+=(uint64_t)pool->cls[c].pages*MESH_PAYLOAD(pool->cls[c].frames);
  return bytes;
}
/* design/PAGE-TABLE-SPEC-20261122.md section 2: the pool carved from the arena once, in a region made new (one
   adopted keeps its carve, `pages` set): MESH_POOL_SHORT short and MESH_POOL_LONG long pages, fewer of both in
   proportion where seven eighths of the arena fit fewer (the rest the bridge's other leases: the map's records), each
   class in runs the arena leases (mesh_arena_alloc: a run within one MESH_DMA_EXTENT, so a SEND of any of its pages
   lies in one registration), each run followed by MESH_CLASS_LONG - frames frames of slack (a page sent with a long
   lane's frames reads no further than its run), a run half as long where the arena has none so long; every page
   free, the short pages' ids first */
static void pool_make(void){
  struct mesh_pool *pool=mesh_pool(M);
  if(pool->pages)return;
  const uint64_t arena=(uint64_t)mesh_arena_pages(M)*M->pgsz,budget=arena-arena/8;
  const uint64_t whole=(uint64_t)MESH_POOL_SHORT*MESH_CLASS_SHORT*MESH_FRAME+(uint64_t)MESH_POOL_LONG*MESH_CLASS_LONG*MESH_FRAME;
  const double scale=budget>=whole?1:(double)budget/(double)whole;
  const uint32_t want[MESH_CLASSES]={(uint32_t)(MESH_POOL_SHORT*scale),(uint32_t)(MESH_POOL_LONG*scale)};
  uint32_t pd=0;
  for(uint32_t c=0;c<MESH_CLASSES;c++){
    struct mesh_pool_class *k=pool->cls+c;
    const uint32_t frames=c?MESH_CLASS_LONG:MESH_CLASS_SHORT,slack=MESH_CLASS_LONG-frames;
    k->frames=frames;k->pages=0;
    atomic_store(&k->head,0);atomic_store(&k->free,0);
    uint32_t run=(uint32_t)((MESH_DMA_EXTENT/MESH_FRAME-slack)/frames);
    while(k->pages<want[c] && run){
      const uint32_t n=MIN(run,want[c]-k->pages);
      const uint64_t bytes=((uint64_t)n*frames+slack)*MESH_FRAME;
      const uint32_t first=mesh_arena_alloc(&bridge,(uint32_t)((bytes+M->pgsz-1)/M->pgsz),1);
      if(first==MESH_ABSENT){run/=2;continue;}
      const uint64_t base=M->data_off+(uint64_t)first*M->pgsz;
      for(uint32_t i=0;i<n;i++){
        struct mesh_pd *p=mesh_pd_at(M,pd+i);
        atomic_store(&p->next,0);p->offset=base+(uint64_t)i*frames*MESH_FRAME;p->cls=(uint16_t)frames;
        atomic_store(&p->uses,0);atomic_store(&p->bytes,0);atomic_store(&p->state,MESH_PD_FREE);atomic_store(&p->lane,0);
      }
      pd+=n;k->pages+=n;
    }
  }
  pool->pages=pd;
  for(uint32_t i=pd;i-->0;)mesh_pool_push(M,i);
  say("bridge: pool carved: %u short pages (of %u frames) and %u long (of %u) in an arena of %llu bytes\n",pool->cls[0].pages,
      pool->cls[0].frames,pool->cls[1].pages,pool->cls[1].frames,(unsigned long long)arena);
}
/* the switch made new over the pool (no SEND, outbox or hand of a bridge before survives its stop: region_adopt gave
   back what they had) */
static void switch_make(void){
  struct mesh_switch *sw=mesh_switch_of(M);
  mesh_switch_init(sw);
  if(getenv("MESH_SWITCH_LOG"))atomic_store(&sw->logging,1);
  const struct mesh_pool *pool=mesh_pool(M);
  say("bridge: switch made over a pool of %u pages (%u free short, %u free long)\n",pool->pages,
      atomic_load(&pool->cls[0].free),atomic_load(&pool->cls[1].free));
}
/* the view as the records stand: every record's node, and each node pair the records join (each end's entries toward
   the other, the fewer of the two), its cables and their bandwidth summed (bytes a ns; 0 where neither end's record
   reports one), at the hop latency the fabric's fits measure (6 us); every node of it an entry of the node table, live
   (the bandwidth of this node's own cables toward it), every other entry not */
#define SWITCH_LATENCY_NS 6000.0
static uint64_t switch_serial;
static void switch_view(void){
  const uint64_t before=atomic_load_explicit(&M->map,memory_order_acquire);
  if(before==switch_serial || (before&1))return;
  const uint64_t count=atomic_load_explicit(&M->map_nodes,memory_order_relaxed);
  const struct mesh_record *r=mesh_records(M);
  uint32_t nodes[MESH_SWITCH_N],n=0;struct mesh_switch_fabric_link links[MESH_SWITCH_LINKS];uint32_t k=0;
  for(uint64_t i=0;i<count && n<MESH_SWITCH_N;i++)nodes[n++]=r[i].node;
  for(uint64_t a=0;a<count;a++)for(uint64_t b=0;b<count;b++){
    if(a==b || k==MESH_SWITCH_LINKS)continue;
    uint32_t ab=0,ba=0;uint64_t bandwidth=0;
    for(uint32_t e=0;e<r[a].reached && e<MESH_STREAMS;e++)if(r[a].reach[e].node==r[b].node){ab++;bandwidth+=r[a].reach[e].bandwidth;}
    for(uint32_t e=0;e<r[b].reached && e<MESH_STREAMS;e++)if(r[b].reach[e].node==r[a].node)ba++;
    const uint32_t cables=ab<ba?ab:ba;
    if(!cables)continue;
    links[k++]=(struct mesh_switch_fabric_link){r[a].node,r[b].node,cables,(double)bandwidth/8e9*cables/ab,SWITCH_LATENCY_NS};
  }
  if(!mesh_map_whole(M,before))return;
  switch_serial=before;
  mesh_switch_view_set(mesh_switch_of(M),M->node,nodes,n,links,k);
  uint32_t live=0;
  for(uint32_t i=0;i<n;i++){
    if(nodes[i]==M->node)continue;
    const uint32_t s=node_slot(nodes[i]);
    if(s==MESH_ABSENT)continue;
    struct mesh_node *x=mesh_node_at(M,s);
    uint64_t bandwidth=0;
    for(uint32_t p=0;p<port_count;p++)if(ports[p].link.state==LINK_UP && ports[p].peer==nodes[i])bandwidth+=ports[p].bandwidth;
    atomic_store(&x->bandwidth,bandwidth);
    atomic_store_explicit(&x->live,1,memory_order_release);
    live|=1u<<s;
  }
  for(uint32_t s=0;s<MESH_NODES;s++)if(!(live>>s&1))atomic_store_explicit(&mesh_node_at(M,s)->live,0,memory_order_release);
}
/* the hands of exited children (and of a region's bridge before: its children's), as each was left, until each is
   settled */
static struct mesh_switch_left *left;
static uint32_t lefts,left_room;
/* each hand left settled as far as it can be now (mesh-switched.h mesh_switch_left_settle: what a child had won first,
   then the rest), each settled one dropped; on every turn of the event loop while any is left */
static void switch_settle(void){
  struct mesh_switch *sw=mesh_switch_of(M);
  for(int won=1;won>=0;won--)for(uint32_t i=0;i<lefts;i++)mesh_switch_left_settle(M,sw,left+i,won,MESH_SWITCH_PARENT);
  uint32_t still=0;
  for(uint32_t i=0;i<lefts;i++)if(left[i].take || left[i].push)left[still++]=left[i];
  lefts=still;
}
/* port p's hand taken as its child left it (and cleared for its next); 0, or ENOMEM where it could not be stored, which
   is said */
static int switch_left(uint32_t p){
  struct mesh_switch_hand *h=mesh_switch_of(M)->hand+p;
  const struct mesh_switch_left was={atomic_exchange(&h->take,0),atomic_exchange(&h->push,0),atomic_load(&h->page)};
  if(!was.take && !was.push)return 0;
  if(lefts==left_room){
    const uint32_t room=left_room?2*left_room:16;
    struct mesh_switch_left *grown=realloc(left,room*sizeof *grown);
    if(!grown){say("bridge: no memory to store hand %u: what it had stays in use\n",p);return ENOMEM;}
    left=grown;left_room=room;
  }
  left[lefts++]=was;
  return 0;
}
/* design/PAGE-TABLE-SPEC-20261122.md section 5: a port's child exited, what it had given back: its hand settled (a
   write it took, a push it began), then its lanes' outboxes emptied (each SEND posted and not completed and each page
   pushed and not posted dropped, counted `lost`) and the pages posted as its lanes' receives given back.  Its lanes
   are no draw's already (lanes_cancel) */
static void switch_child_free(struct port *port){
  struct mesh_switch *sw=mesh_switch_of(M);
  switch_left(port->index);
  switch_settle();
  for(uint32_t j=0;j<2*MESH_LINK_STREAMS;j++)mesh_switch_lane_drop(M,sw,switch_lane(port,j),MESH_SWITCH_PARENT);
  mesh_switch_receives_drop(M,switch_lane(port,0),2*MESH_LINK_STREAMS);
}
/* each client ring's one taker named (mesh-switched.h struct mesh_switch `taker`): ring s the (s mod n)-th, by port
   index ascending, of the n ports whose link is up with a data lane (its child serves them: child_jobs), none while n
   is 0; named again as a link comes up (link_turn), as one is lost (port_lost) and as a port's child is reaped
   (child_reaped), by the event loop.  A child a ring was named for before may take from it a moment longer: each write
   is one child's by the ring's compare-exchange, and a child's death mid-take is its hand's to settle (switch_settle) */
static void switch_takers(void){
  struct mesh_switch *sw=mesh_switch_of(M);
  uint8_t serving[MESH_PORTS];uint32_t n=0;
  for(uint32_t p=0;p<port_count;p++)
    if(ports[p].link.state==LINK_UP && ports[p].link.plan>>MESH_LANE_FIRST)serving[n++]=(uint8_t)ports[p].index;
  for(uint32_t s=0;s<MESH_CLIENTS;s++)
    atomic_store_explicit(&sw->taker[s],(uint8_t)(n?serving[s%n]:MESH_SWITCH_NO_PORT),memory_order_relaxed);
}
/* design/switched-multipath-dma.md: the trained routing policy in the bridge (rdma/route forward.c on the CPU): its
   theta read as the bridge starts from the POSIX shared memory object SWITCH_THETA (route_params_read_stream's bytes,
   which the trainer puts there: memory, never storage), where one is there; each refresh then runs the forward over
   this node's tokens (the lanes from the carried state, then each context) and publishes the contexts' w, from which
   every page here draws its split (mesh_switch_choose).  With no theta, the design's rule (the earliest arrival)
   routes.  A new theta is taken at the bridge's restart (SIGHUP to its heartbeat, the region left in place) */
#define SWITCH_THETA "/mesh-route-theta"
static struct route_params switch_theta;
static int switch_policy_on;
static float switch_policy_state[2*ROUTE_D],*switch_policy_scratch;
static int switch_policy_splits(void *unused,const struct mesh_switch_tokens *tokens,float w[MESH_SWITCH_CONTEXTS][SPLIT_FS]){
  (void)unused;
  _Static_assert(SPLIT_FS==ROUTE_FS && MESH_SWITCH_CONTEXTS>=1,"a context's w the forward's");
  const int lanes=(int)(tokens->lanes<(uint32_t)ROUTE_L?tokens->lanes:(uint32_t)ROUTE_L);
  route_forward_lanes(&switch_theta,&tokens->lane[0][0],lanes,switch_policy_state,switch_policy_scratch);
  for(uint32_t c=0;c<tokens->contexts && c<MESH_SWITCH_CONTEXTS;c++)route_forward_context(&switch_theta,tokens->context[c],switch_policy_state,w[c]);
  return 0;
}
static void switch_policy_load(void){
  const int fd=shm_open(SWITCH_THETA,O_RDONLY,0);
  if(fd<0)return;
  struct stat st;
  void *bytes=!fstat(fd,&st) && st.st_size>0?mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_SHARED,fd,0):MAP_FAILED;
  close(fd);
  if(bytes==MAP_FAILED){say("bridge: the routing policy's theta %s not mapped: the design's rule routes\n",SWITCH_THETA);return;}
  /* route_params_read_stream over the object's bytes (fmemopen: no file) */
  FILE *f=fmemopen(bytes,(size_t)st.st_size,"rb");
  const int failed=!f || route_params_read_stream(&switch_theta,f);
  if(f)fclose(f);
  munmap(bytes,(size_t)st.st_size);
  if(failed){say("bridge: the routing policy's theta %s not read (not route_params_read_stream's format): the design's rule routes\n",SWITCH_THETA);return;}
  switch_policy_scratch=calloc(route_forward_scratch(0),sizeof(float));
  if(!switch_policy_scratch){say("bridge: no memory for the routing policy's scratch: the design's rule routes\n");return;}
  switch_policy_on=1;
  say("bridge: the routing policy's theta %s in memory (%lld bytes): its tables route\n",SWITCH_THETA,(long long)st.st_size);
}
/* the refresh, on a thread of its own: the view updated, the weights recomputed, the policy's tables published */
static void *switch_watch(void *unused){
  (void)unused;
  pthread_setname_np("mesh.switch");
  struct mesh_switch *sw=mesh_switch_of(M);
  uint64_t seed=mesh_switch_mix((uint64_t)getpid()^clock_ns()),point=0;
  static struct mesh_switch_tokens tokens;
  const struct mesh_switch_policy policy={NULL,NULL,switch_policy_splits};
  while(!stop){
    switch_view();
    mesh_switch_refresh(sw,NULL,SWITCH_DRAWS,SWITCH_ETA,seed=mesh_switch_mix(seed),MESH_SWITCH_PAGE);
    if(switch_policy_on)mesh_switch_policy_refresh(sw,&policy,clock_ns(),++point,0,&tokens);
    MESH_SWITCH_COUNT(sw,MESH_SWITCH_PARENT,refreshes,1);
    struct timespec pause={0,(long)SWITCH_REFRESH_NS};
    nanosleep(&pause,NULL);
  }
  return NULL;
}

/* design/transport.md: the bridge's wire, a message a kind: a record, the peer's farewell (its "release
   everything": the link lost at once, so nothing is posted toward it again, design/transport.md) or its
   LINK; a message of any other kind is let be (AGENTS.md, "Fail toward access": unknown input, keep going) */
static void message_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  if(heard->kind==RECORD)record_heard(port,payload,heard->bytes);
  else if(heard->kind==FAREWELL){port->parted=1;port->farewell=heard->code;port_lost(port,ECONNRESET);}
  else if(heard->kind==LINK)link_heard(port,heard);
}
/* a message heard on a port's control socket: before its peer's hello, that hello alone (any other let be); the hello
   names the peer (the node it states, whatever node's hello this end dialed), and pairs the link, its plan the fewer
   queue pairs of the two ends', design/HETEROGENEOUS-LINKS.md, its lanes made (link_turn); a lane's frames are the same
   constants at both ends (lane_frames).  Then each message is taken (message_heard) */
static void port_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  if(!port->greeted){
    if(heard->kind!=HELLO)return;
    /* a bridge of another wire version is not paired with: its hello is let be (mesh.h MESH_VERSION) */
    if(heard->version!=MESH_VERSION)return;
    port->peer=heard->node;port->peer_port=heard->port;port->heard=*heard;port->greeted=1;
    const uint32_t theirs=hello_budget(heard->count);
    port->link.plan=link_plan(MIN(port->budget,theirs));
    struct mesh_link_info *link=&mesh_links(M)[port->index];
    atomic_store_explicit(&link->bandwidth,port->bandwidth,memory_order_relaxed);
    atomic_fetch_add_explicit(&link->pairings,1,memory_order_relaxed);
    say("link %u (%s): paired with node %u (queue pairs for the link: this port's %u, its %u%s; plan %#x, %u lanes, %u stream%s)\n",
        port->index,port->name,port->peer,port->budget,theirs,heard->count?"":", its bridge stating none",port->link.plan,
        plan_lanes(port->link.plan),plan_streams(port->link.plan),plan_streams(port->link.plan)==1?"":"s");
    link_set(port,port->peer,MESH_PAIRED,0);
    /* a port newly paired is sent every record the map has, this node's first */
    for(uint64_t i=0;port->greeted && i<atomic_load(&M->map_nodes);i++)if(mesh_records(M)[i].node!=M->node)
      port_say(port,(struct message){.kind=RECORD},mesh_records(M)+i,sizeof(struct mesh_record));
    if(port->greeted)link_turn(port);
    return;
  }
  message_heard(port,heard,payload);
}

/* design/transport.md section 14, #loss: a link lost has the clients' rings its port took given to the ports still up
   (switch_takers), its lanes no draw's (lanes_cancel) and served no more before
   its control socket closes (its child told, CHILD_HALT: no post on them once the peer can read the loss), what they
   carried counted `lost` as its child's end is freed (switch_child_free); no verbs call is made
   (design/RDMA-KERNEL-RECOVERY.md "October 18").  A link whose child was asked for its lanes (`lanes`) waits for that
   child's end (`recycle`: child_turn asks it once the port's quiet window has ended), whose exit the kernel's release
   of the lanes is, the port closed and greeting no peer meanwhile, and pairs again with the port's next child ("A
   process a port"); a port lost before its lanes were asked for and still open says hello at once (port_hello: the
   records go on).  Its quiet window begins (B9: no port job on it, and no exit, for MESH_PLATFORM_NS, the operator's
   bound on the termination a loss may follow) */
static void child_send(struct port *port,struct child_message said);
void port_lost(struct port *port,int64_t code){
  quiet_until(&port->quiet_at,now+MESH_PLATFORM_NS);
  link_lost(port);
  switch_takers();
  if(port->child){
    child_send(port,(struct child_message){.kind=CHILD_HALT,.pairing=atomic_load(&port->link.pairing)});
    if(port->link.lanes && !stop){port->recycle=1;port->open=0;}
  }
  if(port->control>=0){close(port->control);port->control=-1;}
  if(port->greeted)port->next=now;
  /* design/transport.md: the peer named, and a peer that said farewell told from one lost */
  if(port->greeted||port->connecting){
    char reason[96];farewell_text(port->farewell,reason,sizeof reason);
    if(stop)say("link %u (%s): closed toward node %u: this bridge is stopping\n",port->index,port->name,port->peer);
    else if(port->peer && port->parted)say("link %u (%s): node %u parted, its bridge stopping (%s): code %lld\n",port->index,port->name,
                                            port->peer,reason,(long long)code);
    else if(port->peer)say("link %u (%s): node %u lost: code %lld\n",port->index,port->name,port->peer,(long long)code);
    else say("link %u (%s): lost: code %lld\n",port->index,port->name,(long long)code);
  }
  lanes_cancel(port);
  port->greeted=port->parted=port->connecting=port->writing=0;port->farewell=0;port->peer=0;port->in_length=0;port->out_length=port->out_sent=0;
  link_set(port,0,port->open?MESH_PAIRING:MESH_STOPPED,code);
}
/* the control socket: non-blocking, its sends queued (port_say), its HELLO stating the queue pairs the port's device
   has for a link (hello_budget); its peer's loss is the kernel's to tell (its end closed, a reset, the link's going
   off) */
static void port_greet(struct port *port){
  int on=1,flags=fcntl(port->control,F_GETFL);
  if(flags<0 || fcntl(port->control,F_SETFL,flags|O_NONBLOCK) || setsockopt(port->control,SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on) ||
     setsockopt(port->control,IPPROTO_TCP,TCP_NODELAY,&on,sizeof on)){
    port_lost(port,errno?errno:EIO);return;
  }
  port_watch(port,EVFILT_READ);
  port_say(port,(struct message){.kind=HELLO,.count=1+port->budget,},NULL,0);
}
/* design/transport.md: the port a link-local peer reached, by its interface; none for a scope of 0 (no
   interface: a LAN address, or the index of an interface that no longer exists) */
static struct port *port_at(uint32_t interface){
  for(uint32_t p=0;interface && p<port_count;p++)if(ports[p].interface==interface)return ports+p;
  return NULL;
}
/* design/transport.md: a port's device opened, on its own queue (guarded), where it is to be (at the start,
   at its link's coming on or its interface's attaching, at a hello heard while it is not open, while it has no peer,
   and as the call its port was concluded around returns, ports_stalled), once its quiet window has ended and no job of
   its own is out (design/transport.md B9, B12): the hellos' group joined on its interface, so the port hears them open
   or not (hello_heard, B16); its port read; a context that fails stays as it is, and the port with it, until its child
   ends (design/transport.md section 14: nothing made is forgotten or made again in a process).  The device is its
   child's: spawned here where the port has none (child_spawn: not while its last child's end is awaited, `recycle`,
   nor before the pause a spawn that failed set, `spawn_at`), and asked to open it (CHILD_OPEN), its answer the
   job's end (child_heard) */
static int child_spawn(struct port *port);
static void port_open(struct port *port){
  if(port->job || port->stuck || !port->opening || port->control>=0 || now<port->quiet_at || port->recycle || now<port->spawn_at ||
     now<port->retry_at)return;
  if(!port->child && child_spawn(port))return;
  port->opening=0;
  port->interface=if_nametoindex(port->name);
  struct ipv6_mreq group={.ipv6mr_multiaddr=everyone,.ipv6mr_interface=port->interface};
  if(port->interface)setsockopt(hello,IPPROTO_IPV6,IPV6_JOIN_GROUP,&group,sizeof group);
  port->job=1;port->job_at=now;atomic_store_explicit(&port->done,0,memory_order_relaxed);
  child_send(port,(struct child_message){.kind=CHILD_OPEN});
}
/* the child's open of its port's device, on its port's queue, guarded; its end told the child's loop (child_main) */
static void device_open(struct port *port){
  port->job=1;atomic_store_explicit(&port->done,0,memory_order_relaxed);
  dispatch_async(port->work,^{
    const int raised=guard_raise(port);
    int refused=0,result=device_up(&port->device,&wire,&refused);
    if(!result)result=device_gid(&port->device,port->name);
    if(!result && !port->device.gid_matched){
      const uint8_t *g=port->device.gid.raw;
      say("link %u (%s): no GID of %s is a link-local address %s has; GID[%u] %02x%02x:%02x%02x:%02x%02x:%02x%02x:"
          "%02x%02x:%02x%02x:%02x%02x:%02x%02x used\n",port->index,port->name,port->device.name,port->name,port->device.gid_index,
          g[0],g[1],g[2],g[3],g[4],g[5],g[6],g[7],g[8],g[9],g[10],g[11],g[12],g[13],g[14],g[15]);
    }
    guard_lower(port,raised);
    port->result=result;port->refused=refused;
    atomic_store_explicit(&port->done,1,memory_order_release);
    wake();
  });
}
/* design/transport.md: a port's opening taken (its address and bandwidth as its child told them, CHILD_OPENED): a
   device that failed to open opens again at the next hello heard on it (hello_heard) or its link's next coming on; one
   whose registration of the region it refused (it keeps the extents it took: device_up logged the extent and why) does
   so too, the k-th refusal in a row no sooner than 60 2^(k-1) s on, at most an hour, unless its link comes on again
   (network_heard), since each try wires more of the node's memory (design/rubrics.md, Failure across interfaces 4 and
   6: the Mini's watchdog halted it once its memory was wired whole, 10-20); one opened says hello at once (port_hello) */
static void port_opened(struct port *port){
  const int result=port->result;
  port->job=0;
  if(result){
    if(port->refused){
      port->refusals++;
      const uint64_t backoff=MIN(UINT64_C(60)<<MIN(port->refusals-1,6u),UINT64_C(3600));
      port->retry_at=now+backoff*UINT64_C(1000000000);
      say("link %u (%s): the region's registration refused (%s), %u in a row: opened again at a hello heard on it from %llu s "
          "on, or at its link's coming on\n",port->index,port->name,strerror(port->refused),port->refusals,(unsigned long long)backoff);
    }
    port->open=0;if(port->control<0)link_set(port,0,MESH_STOPPED,result);
    return;
  }
  port->open=1;port->refusals=0;
  if(port->control>=0)return;
  link_set(port,0,MESH_PAIRING,0);
  port->next=now;
}
/* design/transport.md: an open port with no peer says hello to every node on its link, every MESH_PAUSE_NS (a
   paced emission, deciding nothing: a peer that starts later is heard as it says its own) */
static void port_hello(struct port *port){
  if(!port->open || port->control>=0 || port->job || now<port->next)return;
  port->next=now+MESH_PAUSE_NS;
  setsockopt(hello,IPPROTO_IPV6,IPV6_MULTICAST_IF,&port->interface,sizeof port->interface);
  struct sockaddr_in6 to={.sin6_len=sizeof to,.sin6_family=AF_INET6,.sin6_port=htons((uint16_t)atoi(MESH_PORT)),.sin6_addr=everyone,.sin6_scope_id=port->interface};
  struct message said={.magic=XMAGIC,.version=MESH_VERSION,.kind=HELLO,.node=M->node,.boot=M->boot,.port=port->index};
  sendto(hello,&said,sizeof said,0,(struct sockaddr *)&to,sizeof to);
}
/* design/transport.md: a node heard on a port: the lower node dials the higher's listener on the address
   it heard from, the higher answers its hello so the lower hears it at once; a port not open is opened (its open's job
   out aside), so a port whose link job or open failed pairs again on hearing its peer, never on a clock */
static void hello_heard(void){
  struct message heard;struct sockaddr_in6 from;socklen_t length=sizeof from;
  while(recvfrom(hello,&heard,sizeof heard,0,(struct sockaddr *)&from,&length)==(ssize_t)sizeof heard){
    struct port *port=port_at(from.sin6_scope_id);
    length=sizeof from;
    if(!port || heard.magic!=XMAGIC || heard.kind!=HELLO || heard.node==M->node || port->control>=0)continue;
    if(!port->open){if(!port->job)port->opening=1;continue;}
    if(M->node>heard.node){
      struct message said={.magic=XMAGIC,.version=MESH_VERSION,.kind=HELLO,.node=M->node,.boot=M->boot,.port=port->index};
      sendto(hello,&said,sizeof said,0,(struct sockaddr *)&from,sizeof from);
      continue;
    }
    from.sin6_port=htons((uint16_t)atoi(MESH_PORT));
    port->control=socket(AF_INET6,SOCK_STREAM,0);
    if(port->control<0 || fcntl(port->control,F_SETFL,O_NONBLOCK) ||
       (connect(port->control,(struct sockaddr *)&from,sizeof from) && errno!=EINPROGRESS)){port_lost(port,errno);continue;}
    port->peer=heard.node;port->connecting=1;
    port_watch(port,EVFILT_WRITE);
  }
}
/* a control socket's messages, each a header and its `bytes`, each whole one taken */
static void control_heard(struct port *port){
  for(;;){
    size_t need=sizeof(struct message);
    if(port->in_length>=need){
      const struct message *header=(const void *)port->in;
      if(header->magic!=XMAGIC || header->bytes>(UINT32_C(64)<<20)){port_lost(port,EPROTO);return;}
      need+=header->bytes;
    }
    if(need>port->in_room){
      unsigned char *grown=realloc(port->in,need);
      if(!grown){port_lost(port,ENOMEM);return;}
      port->in=grown;port->in_room=need;
    }
    if(port->in_length<need){
      ssize_t got=recv(port->control,port->in+port->in_length,need-port->in_length,0);
      if(got<0 && (errno==EAGAIN || errno==EINTR))return;
      if(got<=0){port_lost(port,got?errno:ENETRESET);return;}
      port->in_length+=(size_t)got;
      if(port->in_length<need || need==sizeof(struct message))continue;
    }
    struct message heard;memcpy(&heard,port->in,sizeof heard);
    port->in_length=0;
    port_heard(port,&heard,port->in+sizeof heard);
    if(port->control<0)return;
  }
}
/* design/transport.md: every RDMA device of the node a port, in the order found, read on the finder's queue
   (at the start, behind IOKit's quiet registry, and where an interface attached or its link came on that no port has,
   on the event loop's turn once the windows have ended and no call is stuck: find_wanted, design/transport.md
   B9, B9b, B12) and added on the event loop */
static void ports_find(void){
  if(finding)return;
  finding=1;
  dispatch_async(finder,^{
    struct ibv_device **list=ibv_get_device_list(NULL);
    uint32_t n=0;
    for(int i=0;list && list[i] && n<MESH_PORTS;i++)if(!strncmp(ibv_get_device_name(list[i]),"rdma_",5))
      snprintf(devices[n++],sizeof devices[0],"%s",ibv_get_device_name(list[i]));
    if(list)ibv_free_device_list(list);
    device_count=n;
    atomic_store_explicit(&found,1,memory_order_release);
    wake();
  });
}
static void ports_found(void){
  finding=0;atomic_store_explicit(&found,0,memory_order_relaxed);
  for(uint32_t i=0;i<device_count && port_count<MESH_PORTS;i++){
    uint32_t known=0;
    for(uint32_t p=0;p<port_count;p++)known|=!strcmp(ports[p].device.name,devices[i]);
    if(known)continue;
    struct port *port=ports+port_count;
    *port=(struct port){.index=port_count,.control=-1,.child_fd=-1,.opening=1};
    port->link.provider.device=&port->device;
    snprintf(port->device.name,sizeof port->device.name,"%s",devices[i]);snprintf(port->name,sizeof port->name,"%s",devices[i]+5);
    /* the interface as the kernel's events name it, its name read once here into typed values (addresses_heard) */
    if(!strncmp(port->name,"en",2)){port->family=APPLE_IF_FAM_ETHERNET;port->unit=(uint32_t)strtoul(port->name+2,NULL,10);}
    /* design/transport.md B9b: counted for a termination's queue (quiet_all, peer_port) once whole, then
       quiet until the window a termination heard meanwhile set (`exit_at`) */
    __atomic_store_n(&port_count,port_count+1,__ATOMIC_SEQ_CST);
    quiet_until(&port->quiet_at,atomic_load(&exit_at));
    link_set(port,0,MESH_STOPPED,0);
  }
}
/* design/transport.md: a port's link off or its interface detached (the link lost, its lanes left as they
   are, design/transport.md B6 (b), and the port down until its link comes on: its device opened then,
   port_open); an interface of any name that attached or whose link came on and is no port yet: the node's devices to
   be read again (a link's coming on stays a reading, for a device registered after its interface attached), the read
   put off to the windows' end (find_wanted, left set while a read is out).  At the stop (`stopping`), a port's link
   going off only extends its quiet window, which the exit waits out (B9) */
static void network_heard(int stopping){
  union mesh_network_event event;
  int unknown=0;
  while(recv(network,event.bytes,sizeof event,0)>=(ssize_t)(KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data))){
    const struct net_event_data *interface=(const void *)(event.bytes+KEV_MSG_HEADER_SIZE);
    const uint32_t code=event.header.event_code;
    char name[IFNAMSIZ+16];
    snprintf(name,sizeof name,"%.*s%u",(int)sizeof interface->if_name,interface->if_name,interface->if_unit);
    int known=0;
    for(uint32_t p=0;p<port_count;p++)if(!strcmp(name,ports[p].name)){
      known=1;
      if(code==KEV_DL_LINK_ON || code==KEV_DL_IF_ATTACHED){ports[p].opening=1;ports[p].retry_at=0;}
      else if(code!=KEV_DL_LINK_OFF && code!=KEV_DL_IF_DETACHED){}
      else if(stopping)quiet_until(&ports[p].quiet_at,now+MESH_PLATFORM_NS);
      else{ports[p].open=0;port_lost(ports+p,ENETDOWN);}
    }
    unknown|=!known && (code==KEV_DL_LINK_ON || code==KEV_DL_IF_ATTACHED);
  }
  if(unknown && !stopping)find_wanted=1;
}
/* design/transport.md: an IPv6 address of a port's interface added, changed or deleted (the kernel's KEV_INET6_SUBCLASS
   events on `addresses`: a link-local address bin/mesh-fabric-init.sh puts in place, or its duplicate address detection
   done): the hellos' group joined on the interface again (a join while it had no IPv6 address took nothing) and the
   port's hello said at this turn, so a hello that left before the address could carry it is said again as the address
   comes, not a pace (MESH_PAUSE_NS) later.  The port is the event's interface family and unit, the port's own read once
   (struct port `family`, `unit`): no name is built or compared per event (design/rubrics.md, Strings are messages).
   Another Ethernet-family interface of the same unit (bridge0, awdl0, anpi<N>) is taken as the port's, and its event
   says the port's hello early, as the pace would: a hello on an open port with no peer is never wrong */
static void addresses_heard(void){
  union { struct kern_event_msg header; unsigned char bytes[512]; } event;
  while(recv(addresses,event.bytes,sizeof event,0)>=(ssize_t)(KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data))){
    const struct net_event_data *interface=(const void *)(event.bytes+KEV_MSG_HEADER_SIZE);
    for(uint32_t p=0;p<port_count;p++)if(interface->if_family==ports[p].family && interface->if_unit==ports[p].unit){
      struct ipv6_mreq group={.ipv6mr_multiaddr=everyone,.ipv6mr_interface=ports[p].interface};
      if(ports[p].interface)setsockopt(hello,IPPROTO_IPV6,IPV6_JOIN_GROUP,&group,sizeof group);
      ports[p].next=now;
    }
  }
}

/* design/transport.md B9b (EXIT-RECLAIM.md P2, and P3 as a window in place of IOServiceWaitQuiet): a
   Thunderbolt RDMA peer interface's termination, which IOKit tells in its first phase (kIOTerminatedNotification), on
   a queue of its own: the terminations heard (`terminations`), and the ports whose peer interface is terminating
   (`terminated`, a bit a port index), which the event loop concludes (terminations_turn) */
static _Atomic uint32_t terminated,terminations;
/* design/transport.md: a Thunderbolt RDMA peer interface's arrival (kIOFirstMatchNotification: a cable's far end come
   to RDMA), on the terminations' queue, so a port's termination and a later arrival are taken in their order: the ports
   whose peer interface arrived (`arrived`, a bit a port index) and whether one arrived whose interface is no port
   (`arrived_elsewhere`), which the event loop takes (arrivals_turn) */
static _Atomic uint32_t arrived;
static _Atomic int arrived_elsewhere;
/* the exit's window and every found port's moved to `until` at least (a port found meanwhile takes `exit_at`'s as it is
   counted: ports_found) */
static void quiet_all(uint64_t until){
  quiet_until(&exit_at,until);
  const uint32_t count=__atomic_load_n(&port_count,__ATOMIC_SEQ_CST);
  for(uint32_t p=0;p<count;p++)quiet_until(&ports[p].quiet_at,until);
}
/* the port whose interface is the terminating peer interface's, by mesh-fabric-init.sh's rule (an
   AppleThunderboltRDMAInterface rdma_<port> and its peer interface carry the same "RID"); NULL where none is found
   (the property removed with the terminating service, or no port of that interface yet) */
static struct port *peer_port(io_service_t peer){
  CFTypeRef rid=IORegistryEntryCreateCFProperty(peer,CFSTR("RID"),kCFAllocatorDefault,0);
  io_iterator_t interfaces=IO_OBJECT_NULL;
  struct port *found=NULL;
  if(rid && IOServiceGetMatchingServices(kIOMainPortDefault,IOServiceMatching("AppleThunderboltRDMAInterface"),&interfaces)==KERN_SUCCESS){
    for(io_service_t interface;(interface=IOIteratorNext(interfaces));IOObjectRelease(interface)){
      CFTypeRef its=found?NULL:IORegistryEntryCreateCFProperty(interface,CFSTR("RID"),kCFAllocatorDefault,0);
      io_name_t name;
      if(its && CFEqual(its,rid) && IORegistryEntryGetName(interface,name)==KERN_SUCCESS){
        const uint32_t count=__atomic_load_n(&port_count,__ATOMIC_SEQ_CST);
        for(uint32_t p=0;p<count;p++)if(!strcmp(ports[p].device.name,name))found=ports+p;
      }
      if(its)CFRelease(its);
    }
    IOObjectRelease(interfaces);
  }
  if(rid)CFRelease(rid);
  return found;
}
/* the iterator drained first, which arms the notification again: for each terminated service the exit's window and
   every port's moved to MESH_PLATFORM_NS from now (no port job and no exit inside the termination, whichever port's it
   is) and the event loop woken, before any property is read; then the port each maps to halted (port_halt: its
   pairing lost, served no more, marked for the event loop); a service that maps to no port only moves the windows;
   each released.  Each process handles a termination once: a child halts its own port at once (its progress thread
   serves no lane from then on) and reports it to its parent (CHILD_TERMINATED, child_reports), and the parent sets
   that port's bit from the report alone (child_heard); the parent's own listener halts only a port that has no child
   (`child` 0, as it always is in a child), so a port's bit is set at most once for each termination and
   terminations_turn loses its link once; a child whose listener is not yet armed as the termination comes (it has
   made no queue pair yet) misses it, and that port's bit is not set */
static void peer_terminated(void *unused,io_iterator_t iterator){
  (void)unused;
  io_service_t services[MESH_PORTS];
  uint32_t count=0;
  for(io_service_t service;(service=IOIteratorNext(iterator));){
    quiet_all(clock_ns()+MESH_PLATFORM_NS);
    atomic_fetch_add(&terminations,1);
    wake();
    if(count<MESH_PORTS)services[count++]=service;
    else IOObjectRelease(service);
  }
  for(uint32_t k=0;k<count;k++){
    struct port *port=peer_port(services[k]);
    IOObjectRelease(services[k]);
    if(!port || __atomic_load_n(&port->child,__ATOMIC_ACQUIRE))continue;
    port_halt(port,&terminated);
    wake();
  }
}
/* the arrivals drained (which arms the notification again), each service released: the port its interface is marked
   to be opened, one whose interface is no port of this process marked for the devices' read */
static void peer_arrived(void *unused,io_iterator_t iterator){
  (void)unused;
  for(io_service_t service;(service=IOIteratorNext(iterator));IOObjectRelease(service)){
    struct port *port=peer_port(service);
    if(port)atomic_fetch_or(&arrived,UINT32_C(1)<<port->index);
    else atomic_store(&arrived_elsewhere,1);
    wake();
  }
}
/* armed at the start, before IOKit's registry is waited on and the devices are read: the notification on a serial
   queue of its own, its iterator drained once (which arms it); without it a link's going off is the first sign.  The
   parent (`arrivals`) hears the peer interfaces' arrivals on the same port and queue, its iterator drained once too:
   without them a port whose open met its device's peer not yet present (ENETDOWN at both ends, no hello sent) opened
   only at its link's next coming on or the next hello heard on it, which might never come.  A child whose
   notification is not armed ends: no other process hears its port's terminations (the parent's listener halts only
   a port with no child, peer_terminated), and it has made no queue pair yet, so its parent loses the link as it reaps
   it (child_reaped) and the port's next child arms its own */
static void terminations_watch(int arrivals){
  IONotificationPortRef notifications=IONotificationPortCreate(kIOMainPortDefault);
  io_iterator_t iterator=IO_OBJECT_NULL,came=IO_OBJECT_NULL;
  kern_return_t error=KERN_FAILURE,heard=KERN_FAILURE;
  if(notifications){
    IONotificationPortSetDispatchQueue(notifications,dispatch_queue_create("mesh.terminations",DISPATCH_QUEUE_SERIAL));
    error=IOServiceAddMatchingNotification(notifications,kIOTerminatedNotification,IOServiceMatching("AppleThunderboltRDMAPeerInterface"),
                                           peer_terminated,NULL,&iterator);
    if(arrivals)heard=IOServiceAddMatchingNotification(notifications,kIOFirstMatchNotification,
                                                      IOServiceMatching("AppleThunderboltRDMAPeerInterface"),peer_arrived,NULL,&came);
  }
  if(arrivals && heard)say("bridge: Thunderbolt RDMA peer interfaces' arrivals not heard (%#x): a port whose device was not yet "
                           "present at its link's coming on opens at the next hello heard on it\n",(unsigned)heard);
  if(arrivals && !heard)peer_arrived(NULL,came);
  if(error && !arrivals)die("child: its port's peer-interface terminations not heard: this child ends");
  if(error){say("bridge: Thunderbolt RDMA peer interfaces' terminations not heard (%#x): a link's going off is its first sign\n",(unsigned)error);return;}
  peer_terminated(NULL,iterator);
}
/* design/transport.md (EXIT-RECLAIM.md P4): the provider's doorbell fault absorbed.  A SIGSEGV or SIGBUS
   whose pc lies in the provider's text (`provider`, read at the start) on a thread that posts on a port's queue
   pairs (struct port `thread`) has one anonymous page mapped at the faulting page by hint, never MAP_FIXED, so nothing
   mapped is replaced: where it got that page (no mapping was there: the doorbell the driver unmapped in a termination),
   or the port's bit in `faulted` is set (its other thread got the page first; else the store faults again until the
   event loop takes the bit), the store lands in that page, no doorbell rings, and the port is halted for the event loop
   to lose (port_halt, terminations_turn: ENETDOWN, no verbs call).  Where the page could not be mapped and the bit is
   clear, the store is made once more (`retried`, the port's last such page: its other thread may have mapped it, its
   bit not yet set or already taken by the event loop), and only a second fault at that page is any other fault.  The
   bridge does not exit.  Any other fault: the default action restored, the store made again, the process
   ended.  On the thread's alternate stack where it has one (port_progress); mmap, munmap and sigaction alone */
static uintptr_t provider[2];
static _Atomic uintptr_t retried[MESH_PORTS];
static void fault_heard(int number,siginfo_t *info,void *context){
  const uintptr_t pc=__darwin_arm_thread_state64_get_pc(((ucontext_t *)context)->uc_mcontext->__ss);
  const uintptr_t page=trunc_page((uintptr_t)info->si_addr);
  const pthread_t self=pthread_self();
  struct port *port=NULL;
  for(uint32_t p=0,count=__atomic_load_n(&port_count,__ATOMIC_SEQ_CST);p<count;p++)
    for(int t=0;t<2;t++)if(atomic_load_explicit(&ports[p].thread[t],memory_order_acquire)==self)port=ports+p;
  if(port && pc>=provider[0] && pc<provider[1]){
    void *got=mmap((void *)page,vm_page_size,PROT_READ|PROT_WRITE,MAP_ANON|MAP_PRIVATE,-1,0);
    if(got!=MAP_FAILED && got!=(void *)page)munmap(got,vm_page_size);
    if(got==(void *)page || atomic_load(&faulted)>>port->index&1){port_halt(port,&faulted);return;}
    if(atomic_exchange(&retried[port->index],page)!=page)return;
  }
  struct sigaction fallback={.sa_handler=SIG_DFL};
  sigaction(number,&fallback,NULL);
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
/* the terminations and the absorbed faults taken on the event loop: each termination logged, and each port whose peer
   interface is terminating lost (port_lost, ETIMEDOUT: its socket closed), its lanes left as they
   are, no verbs call; each port a fault was absorbed on lost as at its link's going off (open 0, ENETDOWN:
   network_heard) */
static uint32_t terminations_told;
static void terminations_turn(void){
  const uint32_t heard=atomic_load(&terminations),lost=atomic_exchange(&terminated,0),fresh=heard-terminations_told;
  const uint32_t faults=atomic_exchange(&faulted,0);
  if(fresh){
    const uint64_t until=atomic_load(&exit_at);
    say("bridge node %u: %u Thunderbolt RDMA peer interface%s terminating: no port job and no exit for %.1f s "
        "(design/transport.md)\n",M->node,fresh,fresh==1?"":"s",(double)(until>now?until-now:0)/1e9);
    terminations_told=heard;
  }
  for(uint32_t p=0;p<port_count;p++)if(lost>>p&1){
    say("link %u (%s): its peer interface terminating: its link lost\n",ports[p].index,ports[p].name);
    port_lost(ports+p,ETIMEDOUT);
  }
  for(uint32_t p=0;p<port_count;p++)if(faults>>p&1){
    say("link %u (%s): a fault of its provider's doorbell absorbed: its link lost (design/transport.md)\n",
        ports[p].index,ports[p].name);
    ports[p].open=0;port_lost(ports+p,ENETDOWN);
  }
}
/* the arrivals taken on the event loop, after the terminations (a peer interface that left and then came is lost, then
   opened): each port whose peer interface arrived and that has no peer is to be opened as at its link's coming on
   (network_heard), so its device is opened at the hardware's word that its far end is present and not only at the en
   interface's link, which can come on before it (bin/mesh-fabric-init.sh: link active, no RDMA peer); an arrival at an
   interface that is no port has the devices read again (find_wanted) */
static void arrivals_turn(void){
  const uint32_t came=atomic_exchange(&arrived,0);
  for(uint32_t p=0;p<port_count;p++)if(came>>p&1 && ports[p].control<0){ports[p].opening=1;ports[p].retry_at=0;}
  if(atomic_exchange(&arrived_elsewhere,0))find_wanted=1;
}

/* design/transport.md section 14, "A process a port": the parent's side of a port's child.  The bridge is a parent
   (the region, the map and records, the clients, discovery, hellos and the control sockets) and a child a port (its
   device, its link's queue pairs and completion queues, its progress thread): mesh-flow spawned again, the same image
   (`self_path`), told the contract's version it is spawned for (mesh-bridge.h MESH_CHILD_VERSION), in a process group of its own (a job's group signalled at its end reaches no verbs owner), with
   the parent's socket at descriptor 3, the parent's stderr, and nothing else.  The child's exit is the one release of
   its port's verbs objects (the kernel's), and a link once connected and then lost pairs again with the port's next
   child */
static char self_path[PATH_MAX];
/* a message whole on the socket between a port's child and its parent, within MESH_PLATFORM_NS of the call (the operator,
   2026-10-23: no wait runs longer than a deadline presented when it is made, 5 s by default): a peer that exited drops
   it, and one that takes no more by the deadline ends this process at once, the wait named (`what`) */
static void message_send(int fd,const struct child_message *said,const char *what){
  const char *at=(const char *)said;
  size_t left=sizeof *said;
  const uint64_t until=clock_ns()+MESH_PLATFORM_NS;
  while(left){
    const ssize_t sent=send(fd,at,left,MSG_DONTWAIT);
    if(sent>0){at+=sent;left-=(size_t)sent;continue;}
    if(sent<0 && errno==EINTR)continue;
    if(sent==0 || errno!=EAGAIN)return;
    const uint64_t t=clock_ns();
    if(t>=until){say("%s: its peer took no message for %.0f s; this process exits\n",what,(double)MESH_PLATFORM_NS/1e9);exit(EX_SOFTWARE);}
    struct pollfd room={.fd=fd,.events=POLLOUT};
    poll(&room,1,(int)((until-t+999999)/1000000));
  }
}
/* a message to the port's child, whole (message_send); one not taken (the child exited) is dropped, its end reaped
   (child_reaped) */
static void child_send(struct port *port,struct child_message said){
  if(port->child_fd>=0)message_send(port->child_fd,&said,"bridge: child_send to a port's child");
}
static void child_reaped(struct port *port);
/* a process, not this bridge nor one of its children, that is a child of the port's device of any version (its argv
   `--child VERSION DEVICE ...`, the three words every version keeps): a child of a launch whose parent ended
   unbidden, which has the device's queue pairs until its exit (child_leave, at most 2 MESH_PLATFORM_NS); its pid and
   start (pid 0: none; one exited or in its exit is none).  A device's queue pairs are one child's at a time: a link
   takes as many as its device has for it, up to MESH_LINK_LANES (ten of a macOS 26 device's ten,
   design/HETEROGENEOUS-LINKS.md).  The list has room for one process past the system's most (kern.maxproc, read
   once), so a list that fills it was cut short; where the processes cannot be listed whole, or the memory for the
   list cannot be had, this bridge ends rather than spawn beside a child it did not see */
static struct mesh_process child_elsewhere(const struct port *port){
  static pid_t *pids;
  static int capacity;
  static char *arguments;
  static size_t room;
  if(!arguments){
    int most=0,procs=0,argmax[2]={CTL_KERN,KERN_ARGMAX},maxproc[2]={CTL_KERN,KERN_MAXPROC};
    size_t length=sizeof most,size=sizeof procs;
    if(sysctl(argmax,2,&most,&length,NULL,0) || most<=0 || sysctl(maxproc,2,&procs,&size,NULL,0) || procs<=0 ||
       !(pids=calloc((size_t)procs+1,sizeof *pids)) || !(arguments=malloc(room=(size_t)most)))
      die("bridge: no memory for the processes' list and arguments (child_elsewhere)");
    capacity=procs+1;
  }
  const int count=proc_listallpids(pids,capacity*(int)sizeof *pids);
  if(count<0 || count>=capacity)die("bridge: the processes not listed whole (child_elsewhere)");
  for(int i=0;i<count;i++){
    const pid_t pid=pids[i];
    int own=pid==getpid();
    for(uint32_t p=0;p<port_count;p++)own|=pid==ports[p].child;
    if(own)continue;
    int mib[3]={CTL_KERN,KERN_PROCARGS2,pid},argc=0;
    size_t size=room;
    if(sysctl(mib,3,arguments,&size,NULL,0) || size<sizeof argc)continue;
    memcpy(&argc,arguments,sizeof argc);
    const char *at=arguments+sizeof argc,*end=arguments+size,*argv[4]={0};
    at+=strnlen(at,(size_t)(end-at));
    while(at<end && !*at)at++;
    for(int a=0;a<argc && a<4 && at<end;a++){argv[a]=at;at+=strnlen(at,(size_t)(end-at))+1;}
    if(!(argc>=4 && argv[1] && argv[3] && !strcmp(argv[1],"--child") && !strcmp(argv[3],port->device.name)))continue;
    const uint64_t start=mesh_started(pid);
    if(start)return (struct mesh_process){.pid=(uint32_t)pid,.start=start};
  }
  return (struct mesh_process){0};
}
/* the port's child spawned (its socket's messages and its exit watched, tagged by the port); 0, or the error, the
   port's next spawn put off MESH_PAUSE_NS.  A child of its device from an earlier launch (child_elsewhere) puts off
   the spawn until its exit, which the event loop hears (mesh-observe.h mesh_exits_wait on the loop's queue, tagged by
   the port: `elsewhere` cleared) and spawns at once; the processes are read again only then, not at every turn while
   it lives.  One exited or in its exit by the time its exit is watched is none, and the spawn goes on */
static int child_spawn(struct port *port){
  int pair[2],on=1;
  if(port->elsewhere)return EBUSY;
  struct mesh_process elsewhere=child_elsewhere(port);
  if(mesh_exits_wait(events,&elsewhere,1,TAG_CHILD|port->index,0)){
    say("link %u (%s): process %u, a child of its device from an earlier launch, has its queue pairs until its exit; its "
        "own child spawned after it\n",port->index,port->name,elsewhere.pid);
    port->elsewhere=(pid_t)elsewhere.pid;
    return EBUSY;
  }
  /* design/transport.md section 2: the child's entry among the region's processes, one no live child has; where every
     one is taken, the child waits as for an earlier launch's (a process of the region is never left unrecorded) */
  struct mesh_process *children=mesh_bridge_pages(M)->child;
  uint32_t entry=0;
  while(entry<2*MESH_PORTS && mesh_alive((pid_t)children[entry].pid,children[entry].start))entry++;
  if(entry==2*MESH_PORTS){port->spawn_at=now+MESH_PAUSE_NS;return EBUSY;}
  if(socketpair(AF_UNIX,SOCK_STREAM,0,pair)){
    const int error=errno?errno:EIO;
    say("link %u (%s): its child not spawned: %s\n",port->index,port->name,strerror(error));
    port->spawn_at=now+MESH_PAUSE_NS;
    return error;
  }
  setsockopt(pair[0],SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on);
  char index[16];
  snprintf(index,sizeof index,"%u",port->index);
  char version[16];
  snprintf(version,sizeof version,"%u",MESH_CHILD_VERSION);
  char *arguments[]={self_path,"--child",version,port->device.name,index,NULL};
  posix_spawn_file_actions_t actions;
  posix_spawnattr_t attributes;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions,0,"/dev/null",O_RDONLY,0);
  posix_spawn_file_actions_addopen(&actions,1,"/dev/null",O_WRONLY,0);
  posix_spawn_file_actions_addinherit_np(&actions,2);
  posix_spawn_file_actions_adddup2(&actions,pair[1],3);
  posix_spawnattr_init(&attributes);
  posix_spawnattr_setflags(&attributes,POSIX_SPAWN_CLOEXEC_DEFAULT|POSIX_SPAWN_SETPGROUP);
  posix_spawnattr_setpgroup(&attributes,0);
  pid_t pid=0;
  const int error=posix_spawn(&pid,self_path,&actions,&attributes,arguments,*_NSGetEnviron());
  posix_spawn_file_actions_destroy(&actions);posix_spawnattr_destroy(&attributes);
  close(pair[1]);
  if(error){
    close(pair[0]);
    say("link %u (%s): its child not spawned: %s\n",port->index,port->name,strerror(error));
    port->spawn_at=now+MESH_PAUSE_NS;
    return error;
  }
  __atomic_store_n(&port->child,pid,__ATOMIC_RELEASE);   /* read by the terminations' queue (peer_terminated) */
  port->child_fd=pair[0];port->child_in_length=0;port->ending=0;
  children[entry]=(struct mesh_process){(uint32_t)pid,0,mesh_started(pid)};
  port->child_entry=entry;
  say("link %u (%s): its child %d spawned\n",port->index,port->name,(int)pid);
  struct kevent64_s watch;
  EV_SET64(&watch,(uint64_t)pair[0],EVFILT_READ,EV_ADD,0,0,TAG_CHILD|port->index,0,0);
  if(kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)){close(port->child_fd);port->child_fd=-1;}
  EV_SET64(&watch,(uint64_t)pid,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,TAG_CHILD|port->index,0,0);
  if(kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)){child_reaped(port);return ECHILD;}
  return 0;
}
/* a link job asked of the port's child: its lanes made (the plan's) or connected to the peer's
   (the peer's lanes as its LINK told them, its address as its hello did, its node and the link's bandwidth), under
   the link's pairing as it is asked; its end the child's answer (child_heard), taken by link_turn */
static void child_job(struct port *port,int job){
  struct link *l=&port->link;
  l->job=job;l->job_pairing=atomic_load_explicit(&l->pairing,memory_order_relaxed);l->job_at=now;
  atomic_store_explicit(&l->done,0,memory_order_relaxed);
  struct child_message said={.kind=job==LINK_JOB_MAKE?CHILD_MAKE:CHILD_CONNECT,.pairing=l->job_pairing,.lanes=l->plan};
  if(job==LINK_JOB_CONNECT){
    memcpy(said.peer,l->peer,sizeof said.peer);said.lid=port->heard.lid;memcpy(said.gid,port->heard.gid,16);
    said.node=port->peer;said.bandwidth=port->bandwidth;
  }
  child_send(port,said);
}
/* the child's messages, each whole one taken: an open's, a make's or a connect's end for the job out (a connect of a
   pairing since lost has its lanes cancelled again: the child may have opened them after the loss's cancel); the
   lanes a progress thread found failed, for the link's pairing (heard while its connect's end is not yet taken, read
   by link_turn in the turn that takes it); a fault absorbed, a termination heard (the one setting of the port's
   `terminated` bit for it: peer_terminated), each its port's bit for the event loop's conclusion (terminations_turn).
   The socket's end closes it; the child's exit is reaped (child_reaped) */
static void child_heard(struct port *port){
  struct link *l=&port->link;
  const uint32_t bit=UINT32_C(1)<<port->index;
  for(;;){
    const ssize_t got=recv(port->child_fd,port->child_in+port->child_in_length,sizeof port->child_in-port->child_in_length,MSG_DONTWAIT);
    if(got<0 && (errno==EAGAIN || errno==EINTR))return;
    if(got<=0){close(port->child_fd);port->child_fd=-1;return;}
    if((port->child_in_length+=(uint32_t)got)<sizeof port->child_in)continue;
    port->child_in_length=0;
    struct child_message heard;
    memcpy(&heard,port->child_in,sizeof heard);
    const int current=heard.pairing==atomic_load_explicit(&l->pairing,memory_order_relaxed);
    if(heard.kind==CHILD_OPENED && port->job){
      port->result=heard.result;port->refused=heard.refused;
      if(!heard.result){port->lid=heard.lid;memcpy(port->gid,heard.gid,16);port->bandwidth=heard.bandwidth;port->budget=heard.lanes;}
      atomic_store_explicit(&port->done,1,memory_order_release);
    }
    else if(heard.kind==CHILD_MADE || heard.kind==CHILD_CONNECTED){
      if(heard.kind==CHILD_CONNECTED && !current)lanes_cancel(port);
      if(l->job!=(heard.kind==CHILD_MADE?LINK_JOB_MAKE:LINK_JOB_CONNECT) || heard.pairing!=l->job_pairing)continue;
      l->result=heard.result;
      if(heard.kind==CHILD_MADE && !heard.result){
        memcpy(l->qpn,heard.qpn,sizeof l->qpn);l->psn=heard.psn;l->share=(struct frames){heard.share};
        port->lid=heard.lid;memcpy(port->gid,heard.gid,16);
      }
      atomic_store_explicit(&l->done,1,memory_order_release);
    }
    else if(heard.kind==CHILD_BROKEN && current)atomic_fetch_or(&l->broken,heard.broken);
    else if(heard.kind==CHILD_FAULTED)atomic_fetch_or(&faulted,bit);
    else if(heard.kind==CHILD_TERMINATED)atomic_fetch_or(&terminated,bit);
  }
}
/* a port whose link was lost once its lanes were asked for (`recycle`) asks its child to end once its quiet window and
   the exit's have ended (B9, B9b: no release inside a termination) and no job of its own is out (B12) */
static void child_turn(struct port *port){
  if(!port->recycle || !port->child || port->ending || port->job || port->link.job || now<atomic_load(&port->quiet_at) ||
     now<atomic_load(&exit_at))return;
  child_send(port,(struct child_message){.kind=CHILD_END});
  port->ending=1;
  say("link %u (%s): its child %d asked to end, its lanes released at its exit (design/transport.md section 14)\n",
      port->index,port->name,(int)port->child);
}
/* the port's child ended: its last messages taken, its exit reaped, a job of it out ended (ECHILD); one that ended
   unbidden loses the link (port_lost, ECHILD) and is spawned again once the port's quiet window has passed, at the
   events that open any port, however often its children ended (the operator, 2026-10-23 01:03Z: "this is not a real feature.
   delete it and anything like it").  Its lanes no draw's, what it had given back (switch_child_free: its queue pairs
   went with it, so nothing lands in or is read from its pages again), its link down with no lanes, the clients' rings'
   takers named again without it (switch_takers), and the port to be opened by its next child */
static void child_reaped(struct port *port){
  struct link *l=&port->link;
  const pid_t pid=port->child;
  int status=0;
  if(port->child_fd>=0)child_heard(port);
  if(port->child_fd>=0){close(port->child_fd);port->child_fd=-1;}
  while(waitpid(pid,&status,0)<0 && errno==EINTR){}
  __atomic_store_n(&port->child,0,__ATOMIC_RELEASE);
  if(port->child_entry<2*MESH_PORTS && mesh_bridge_pages(M)->child[port->child_entry].pid==(uint32_t)pid)
    mesh_bridge_pages(M)->child[port->child_entry]=(struct mesh_process){0};
  const int asked=port->ending;
  port->ending=port->recycle=0;
  if(asked)say("link %u (%s): its child %d ended\n",port->index,port->name,(int)pid);
  else{
    say("link %u (%s): its child %d ended unbidden (%s %d)\n",port->index,port->name,(int)pid,
        WIFSIGNALED(status)?"signal":"status",WIFSIGNALED(status)?WTERMSIG(status):WEXITSTATUS(status));
    port->open=0;
    port_lost(port,ECHILD);
  }
  if(port->job){port->result=ECHILD;port->refused=0;atomic_store_explicit(&port->done,1,memory_order_release);}
  if(l->job){l->result=ECHILD;atomic_store_explicit(&l->done,1,memory_order_release);}
  lanes_cancel(port);
  switch_child_free(port);
  l->state=LINK_DOWN;l->lanes=l->heard=0;l->share=(struct frames){0};l->plan=0;
  memset(l->qpn,0,sizeof l->qpn);
  atomic_store_explicit(&l->broken,0,memory_order_relaxed);
  switch_takers();
  port->opening=1;
}

/* design/transport.md: the clients' control word, watched for the event loop */
static void *control_watch(void *unused){
  (void)unused;
  while(!stop){
    uint64_t seen=atomic_load_explicit(&M->control,memory_order_acquire);
    wake();
    os_sync_wait_on_address(&M->control,seen,sizeof M->control,OS_SYNC_WAIT_ON_ADDRESS_SHARED);
  }
  return NULL;
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
   lanes and hand the region's), the provider's faults absorbed (fault_heard) and its peer interface's terminations
   heard (B9b: its lanes served no more at once, port_halt, and each told its parent, CHILD_TERMINATED; a child whose
   listener cannot be armed ends there, terminations_watch).  It takes the parent's messages on descriptor 3 one after
   another: an open, a make or a connect of its link as a job on its port's queue (link_job, device_open; one asked
   while a job is out, or while a halt of its own is not yet answered by the parent's, is refused), a halt (no lane
   served, a job out lapsed), an end, and a message of any other kind let be; and tells each job's end, naming the
   pairing it was asked for.  A connect that ends while its pairing stands opens the link's lanes (lanes_open) and
   then serves every lane, its pairing read again (port_halt's sequence).  The child ends at its parent's word, once
   its own windows have ended (at most MESH_PLATFORM_NS on), or at its parent's loss (the socket's end), its lanes
   served no more, MESH_PLATFORM_NS and its windows on (at most twice that): its exit, the kernel's release of its
   device's objects, no verbs call made */
static uint32_t child_asked;
static uint64_t child_bandwidth;
static int child_halted;
static unsigned char child_in[sizeof(struct child_message)];
static uint32_t child_got;
static void child_tell(struct child_message said){ message_send(3,&said,"child: child_tell to its parent"); }
static __attribute__((noreturn)) void child_leave(struct port *port,int asked){
  atomic_fetch_add(&port->link.pairing,1);
  atomic_store(&port->link.served,0);
  atomic_store_explicit(&port->closing,1,memory_order_release);
  guard_wake(port);
  const uint64_t start=clock_ns(),bound=start+(asked?1:2)*MESH_PLATFORM_NS;
  for(;;){
    const uint64_t at=clock_ns(),windows=MAX(atomic_load(&exit_at),atomic_load(&port->quiet_at));
    const uint64_t until=MIN(bound,MAX(asked?start:start+MESH_PLATFORM_NS,windows));
    if(at>=until)break;
    struct timespec rest={(time_t)((until-at)/1000000000u),(long)((until-at)%1000000000u)};
    nanosleep(&rest,NULL);
  }
  exit(0);
}
static void child_command(struct port *port,const struct child_message *heard){
  struct link *l=&port->link;
  if(heard->kind==CHILD_OPEN){if(!port->job)device_open(port);return;}
  if(heard->kind==CHILD_HALT){
    atomic_fetch_add(&l->pairing,1);atomic_store(&l->served,0);guard_wake(port);
    child_halted=0;
    return;
  }
  if(heard->kind==CHILD_END)child_leave(port,1);
  if(heard->kind!=CHILD_MAKE && heard->kind!=CHILD_CONNECT)return;
  const int job=heard->kind==CHILD_MAKE?LINK_JOB_MAKE:LINK_JOB_CONNECT;
  int refused=l->job?EBUSY:child_halted || atomic_load(&faulted) || atomic_load(&terminated)?ECANCELED:0;
  if(!refused && job==LINK_JOB_CONNECT){
    memcpy(l->peer,heard->peer,sizeof l->peer);port->heard.lid=heard->lid;memcpy(port->heard.gid,heard->gid,16);
    port->peer=heard->node;child_bandwidth=heard->bandwidth;
  }
  if(refused){
    child_tell((struct child_message){.kind=job==LINK_JOB_MAKE?CHILD_MADE:CHILD_CONNECTED,.pairing=heard->pairing,.result=refused});
    return;
  }
  child_asked=heard->pairing;
  if(job==LINK_JOB_MAKE)l->plan=heard->lanes;
  link_job(port,job);
}
static void child_jobs(struct port *port){
  struct link *l=&port->link;
  if(port->job && atomic_load_explicit(&port->done,memory_order_acquire)){
    port->job=0;
    struct child_message said={.kind=CHILD_OPENED,.result=port->result,.refused=port->refused};
    if(!port->result){
      port->lid=port->device.port.lid;memcpy(port->gid,&port->device.gid,16);
      said.lid=port->lid;memcpy(said.gid,port->gid,16);said.bandwidth=port_bandwidth(&port->device.port);
      said.lanes=port->device.max_pairs;
    }
    child_tell(said);
  }
  if(!l->job || !atomic_load_explicit(&l->done,memory_order_acquire))return;
  const int job=l->job;
  l->job=0;
  struct child_message said={.kind=job==LINK_JOB_MAKE?CHILD_MADE:CHILD_CONNECTED,.pairing=child_asked,.result=l->result};
  if(!said.result && l->job_pairing!=atomic_load(&l->pairing))said.result=ECANCELED;
  if(!said.result && job==LINK_JOB_MAKE){
    for(uint32_t k=0;k<MESH_LINK_LANES;k++)said.qpn[k]=l->plan>>k&1?l->provider.queues[k].pair->qp_num:UINT32_MAX;
    said.psn=l->provider.psn;said.share=(uint32_t)l->share.n;
    port->lid=port->device.port.lid;memcpy(port->gid,&port->device.gid,16);
    said.lid=port->lid;memcpy(said.gid,port->gid,16);
  }
  else if(!said.result){
    lanes_open(port,port->peer,child_bandwidth);
    atomic_store(&l->served,l->plan);
    if(atomic_load(&l->pairing)!=l->job_pairing){atomic_store(&l->served,0);said.result=ECANCELED;}
    guard_wake(port);
  }
  child_tell(said);
}
static void child_reports(struct port *port){
  const uint32_t broken=atomic_exchange_explicit(&port->link.broken,0,memory_order_acq_rel);
  if(broken)child_tell((struct child_message){.kind=CHILD_BROKEN,.pairing=child_asked,.broken=broken});
  if(atomic_exchange(&faulted,0)){child_halted=1;child_tell((struct child_message){.kind=CHILD_FAULTED});}
  if(atomic_exchange(&terminated,0)){child_halted=1;child_tell((struct child_message){.kind=CHILD_TERMINATED});}
}
static void child_read(struct port *port){
  for(;;){
    const ssize_t got=recv(3,child_in+child_got,sizeof child_in-child_got,MSG_DONTWAIT);
    if(got<0 && (errno==EAGAIN || errno==EINTR))return;
    if(got<=0)child_leave(port,0);
    if((child_got+=(uint32_t)got)<sizeof child_in)continue;
    child_got=0;
    struct child_message heard;
    memcpy(&heard,child_in,sizeof heard);
    child_command(port,&heard);
  }
}
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
  struct port *port=ports;
  *port=(struct port){.index=(uint32_t)index,.control=-1,.child_fd=-1,.work=dispatch_queue_create("mesh.port",DISPATCH_QUEUE_SERIAL)};
  port->link.provider.device=&port->device;
  snprintf(port->device.name,sizeof port->device.name,"%s",argv[3]);snprintf(port->name,sizeof port->name,"%s",argv[3]+5);
  __atomic_store_n(&port_count,1,__ATOMIC_SEQ_CST);
  struct kevent64_s event;
  if((events=kqueue())<0)die("child events");
  EV_SET64(&event,0,EVFILT_USER,EV_ADD|EV_CLEAR,0,0,UINT64_MAX,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("child events");
  EV_SET64(&event,3,EVFILT_READ,EV_ADD,0,0,UINT64_MAX,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("child events");
  terminations_watch(0);
  for(;;){
    now=clock_ns();
    child_jobs(port);
    child_reports(port);
    struct kevent64_s ready[4];
    struct timespec tick={(time_t)(MESH_PAUSE_NS/1000000000u),(long)(MESH_PAUSE_NS%1000000000u)};
    const int count=kevent64(events,NULL,0,ready,4,0,&tick);
    now=clock_ns();
    for(int i=0;i<count;i++)if(ready[i].filter==EVFILT_READ && ready[i].ident==3)child_read(port);
  }
}

/* design/transport.md section 2, the region and the heartbeat: the node's region (MESH_NAME) opened.  The region a
   bridge before this one left (a restart, SIGHUP; a crash) is adopted where its magic, layout version (MESH_LAYOUT) and
   layout (`geometry`, mesh_layout of this configuration's region) are this bridge's: the same memory, its clients'
   slots, indices and pages standing, its pages registered again by the port children (the same
   pages: nothing is wired twice, nothing reloaded).  A region a live bridge serves is never read: a second bridge of
   the node ends at its port's bind, before region_open.  A region of another layout, or one a stop released (layout version 0), is made
   expired (its layout version 0: no client's attach takes it) and left named, and this bridge ends with EX_TEMPFAIL:
   its heartbeat unlinks it once no process maps it and opens a bridge again, so no second region is ever made beside
   a live client of the first (the 2026-10-20 watchdog panic, design/rubrics.md "Failure across interfaces"); a bridge
   unlinks only a shared memory object that never became a region (no magic: a bridge ended while making it).  A
   region that cannot be read is left as it is, and this bridge ends.  1 adopted, 0 made; *out its descriptor */
static int region_same(const struct hdr *a,const struct hdr *b){
  return a->pgsz==b->pgsz && a->block==b->block && a->entries==b->entries && a->streams==b->streams && a->clients==b->clients &&
         a->table_off==b->table_off && a->arena_off==b->arena_off && a->ring_off==b->ring_off && a->pool_off==b->pool_off &&
         a->bridge_off==b->bridge_off && a->stream_off==b->stream_off && a->node_off==b->node_off &&
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
      __atomic_store_n(&old->version,0,__ATOMIC_SEQ_CST);
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
  struct mesh_bridge_pages *own=mesh_bridge_pages(M);
  own->map=MESH_ABSENT;own->map_pages=0;
  *out=fd;
  return 0;
}

/* the children of the bridge before (the region's processes, mesh-observe.h `child`) waited out, before anything they
   had is given back: until a child's exit begins (from then on mesh_started reads 0 and none of its code runs), its
   progress thread may be inside a provider call and go on to post, take and land on the lanes, hands, rings and pages
   an adoption gives back.  Their exits are waited on a queue of their own (mesh-observe.h mesh_exits_wait), for at most
   2 MESH_PLATFORM_NS, child_leave's bound on a child whose parent ended; a child that lives past it ends this bridge,
   which its heartbeat opens again */
static void region_children(void){
  struct mesh_process children[2*MESH_PORTS];
  memcpy(children,mesh_bridge_pages(M)->child,sizeof children);
  const int exits=kqueue();
  if(exits<0)die("bridge: no queue for the exits of the children of the bridge before");
  const uint32_t alive=mesh_exits_wait(exits,children,2*MESH_PORTS,0,clock_ns()+2*MESH_PLATFORM_NS);
  close(exits);
  if(alive){
    say("bridge: %u children of the bridge before live %.0f s on: this bridge ends, and its heartbeat opens one again\n",
        alive,(double)(2*MESH_PLATFORM_NS)/1e9);
    exit(1);
  }
}
/* the adopted region's bridge state made this bridge's, before any client reads it as live (`bridge_pid` set last) and
   before any child is spawned, once the children of the bridge before have ended (region_children): every lane record
   not live and with no peer, the links' table cleared, a client slot whose leaving a process that died left half done
   finished (mesh.h mesh_leave, which any party can repeat to the same end), then what the bridge before had given
   back at once, as a child's end gives it back (switch_child_free): every switch lane made not live, each hand of its
   children settled, each lane's outbox emptied (a SEND posted or a page pushed: its use dropped, counted `lost`), and
   every page posted as a receive given back (a switch of another layout is not read: what its outboxes and hands had
   stays in use).  A child's device may still land bytes into a page its receives were posted into until the kernel
   releases its objects in the child's exit, which another use may have by then: that is region memory, data, never
   firmware; the firmware rule, one process driving a device's queue pairs at a time, is met by the stop's wait on
   its own children until their exits end or MESH_PLATFORM_NS passes (main), and, where a bridge ended without that
   wait, by region_children above and by child_elsewhere, which puts off a new child's spawn on a device while an old
   child of that device is found; neither finds a child whose exit has begun and not ended (the stop's comment in
   main).  What is given back is
   what the bridge before had, each use by the hand, outbox or receive that had it, never a recount of each page's
   uses from the table: the clients attached across the restart change words, ring cells and uses as it runs (a
   write's use counted as it pushes its cell, an invalidation's dropped as it clears a word), so a count made from one
   reading of the table and the rings would miss or double each change made between its reads.  A map or health
   serial a bridge that died mid-write left odd made even (its readers wait while it is odd), and the map's records
   this bridge's to write again (map_adopt).  The pool's carve and the page table stand; the switch is made new after
   (switch_make), so what was given back is said here */
static void region_adopt(void){
  region_children();
  for(uint32_t s=0;s<MESH_STREAMS;s++){
    struct mesh_lanes *x=mesh_lanes_at(M,s);
    atomic_store(&x->live,0);atomic_store(&x->peer,0);atomic_store(&x->wire,0);
  }
  struct mesh_link_info *links=mesh_links(M);
  for(uint32_t p=0;p<MESH_PORTS;p++){
    atomic_store(&links[p].peer,0);atomic_store(&links[p].phase,MESH_UNKNOWN);atomic_store(&links[p].code,0);atomic_store(&links[p].bandwidth,0);
  }
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    const uint64_t owner=atomic_load(&((struct mesh_client *)((char *)M+M->client_off))[s].owner);
    if(owner&MESH_LEAVING)mesh_leave(M,owner);
  }
  struct mesh_switch *sw=mesh_switch_of(M);
  uint64_t lost=0;
  if(sw->magic==MESH_SWITCH_MAGIC){
    const uint64_t before=atomic_load(&sw->counters[MESH_SWITCH_PARENT].lost);
    for(uint32_t l=0;l<MESH_SWITCH_LANES;l++)atomic_store_explicit(&sw->lane[l].live,0,memory_order_release);
    for(uint32_t p=0;p<MESH_PORTS;p++)switch_left(p);
    switch_settle();
    for(uint32_t l=0;l<MESH_SWITCH_LANES;l++)mesh_switch_lane_drop(M,sw,l,MESH_SWITCH_PARENT);
    lost=atomic_load(&sw->counters[MESH_SWITCH_PARENT].lost)-before;
  }
  const uint32_t receives=mesh_switch_receives_drop(M,0,MESH_SWITCH_LANES);
  say("bridge: what the bridge before had given back: %llu pages of its hands and outboxes (lost) and %u of its "
      "receives\n",(unsigned long long)lost,receives);
  if(atomic_load(&M->map)&1)atomic_fetch_add(&M->map,1);
  if(atomic_load(&M->health)&1)atomic_fetch_add(&M->health,1);
  map_adopt();
  atomic_store(&M->busy,0);
}

/* design/transport.md: when the event loop's next turn is due if no event comes first: the earliest of the times its turn
   waits for that are still ahead (the next liveness turn, `liveness`; the next health sample;
   where the devices are to be read, the windows' end; the exit's window; each port's quiet window, spawn pause, open's
   retry, next hello, and the bound of its open's or link's job out), and no later than MESH_PAUSE_NS on (design/rubrics.md M3: the wait ends in finite
   time).  Every other change a turn takes wakes the loop where it happens (wake, `control`, a socket, a signal and an
   exit on the queue), so a turn comes at the change or at the time it waits for, where a fixed tick (250 ms) took each
   such time up to a tick late */
static inline void due_at(uint64_t *due,uint64_t at){ if(at>now && at<*due)*due=at; }
static uint64_t turn_due(uint64_t liveness){
  uint64_t due=MIN(liveness,now+MESH_PAUSE_NS);
  due_at(&due,health_due());
  if(find_wanted)due_at(&due,windows_end());
  due_at(&due,atomic_load(&exit_at));
  for(uint32_t p=0;p<port_count;p++){
    const struct port *port=ports+p;
    due_at(&due,atomic_load(&port->quiet_at));due_at(&due,port->spawn_at);
    if(port->opening)due_at(&due,port->retry_at);
    if(port->open && port->control<0)due_at(&due,port->next);
    if(port->job)due_at(&due,port->job_at+MESH_PLATFORM_NS);
    if(port->link.job)due_at(&due,port->link.job_at+MESH_PLATFORM_NS);
  }
  return due;
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
  static const int stopping[]={SIGINT,SIGTERM,SIGHUP,SIGQUIT,SIGUSR1,SIGUSR2,SIGALRM,SIGVTALRM,SIGPROF,SIGXCPU,SIGXFSZ};
  if((events=kqueue())<0)die("bridge events");
  for(size_t i=0;i<sizeof stopping/sizeof *stopping;i++){
    EV_SET64(&event,(uint64_t)stopping[i],EVFILT_SIGNAL,EV_ADD,0,0,UINT64_MAX,0,0);
    if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");
    signal(stopping[i],SIG_IGN);
  }
  signal(SIGPIPE,SIG_IGN);
  uuid_t uuid;struct timespec bound={5,0};
  if(gethostuuid(uuid,&bound)){say("host identity: %s; this bridge's node is drawn at random\n",strerror(errno));arc4random_buf(uuid,sizeof uuid);}
  uint32_t node=0;
  for(int i=0;i<16;i++)node=(node*0x01000193u)^uuid[i];
  node=(node&0x7fffffffu)?node&0x7fffffffu:1;
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
  if(adopted)region_adopt();
  /* a boot is the bridge's start on the wall clock, so a node's later boot is a newer record */
  M->node=node;M->boot=clock_gettime_nsec_np(CLOCK_REALTIME);M->idle_since=M->boot;
  say("bridge node %u: region %s %s\n",node,MESH_NAME,adopted?"adopted from the bridge before this one (its clients' indices and pages stand)":"made new");
  if(wire_map(&wire,M,fd))die("transport page aliases");
  close(fd);
  bridge.M=M;
  finder=dispatch_queue_create("mesh.ports",DISPATCH_QUEUE_SERIAL);
  /* the pool (its carve the record's region) and the switch before the record and the map's lease */
  pool_make();
  switch_make();
  /* design/transport.md: the node's memory and hardware model, which its record carries with its region */
  size_t size=sizeof memory;
  if(sysctlbyname("hw.memsize",&memory,&size,NULL,0))memory=0;
  sysctl_text("hw.model",model,sizeof model);
  health_init();
  record_changed();
  if(!atomic_load(&M->map_nodes))die("the map's lease");
  switch_policy_load();
  int watched[4]={listener,hello,network,addresses};
  for(int i=0;i<4 && watched[i]>=0;i++){EV_SET64(&event,(uint64_t)watched[i],EVFILT_READ,EV_ADD,0,0,UINT64_MAX,0,0);if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");}
  EV_SET64(&event,0,EVFILT_USER,EV_ADD|EV_CLEAR,0,0,UINT64_MAX,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");
  mesh_bridge_pages(M)->bridge_start=mesh_started(getpid());
  atomic_store(&M->bridge_pid,(uint64_t)getpid());
  __sync_synchronize();M->magic=MESH_MAGIC;
  pthread_t watcher;
  if(pthread_create(&watcher,NULL,control_watch,NULL))die("control watcher");
  /* the refresh at the progress threads' class: its SWITCH_REFRESH_NS pause near its length (at the default class
     the system's timer slack made 50 us about 460 us) */
  pthread_t refresher;
  pthread_attr_t refreshing;pthread_attr_init(&refreshing);pthread_attr_set_qos_class_np(&refreshing,QOS_CLASS_USER_INTERACTIVE,0);
  if(pthread_create(&refresher,&refreshing,switch_watch,NULL))die("the switch's refresh");
  pthread_attr_destroy(&refreshing);
  /* design/transport.md B9b, B9: the peer interfaces' terminations heard from here on; the devices first read
     once IOKit's registry is quiet (no termination in its course: XNU's waitForState, its deadline MESH_PLATFORM_NS),
     else as they stand at the bound */
  terminations_watch(1);
  mach_timespec_t quiet={(unsigned)(MESH_PLATFORM_NS/1000000000u),(clock_res_t)(MESH_PLATFORM_NS%1000000000u)};
  const kern_return_t busy=IOKitWaitQuiet(kIOMainPortDefault,&quiet);
  if(busy)say("bridge node %u: IOKit's registry not quiet within %.1f s (%#x): its devices read as they stand\n",node,
              (double)MESH_PLATFORM_NS/1e9,(unsigned)busy);
  ports_find();
  say("bridge node %u: configuration %s, region %llu bytes, arena %llu pages, page table %u entries\n",node,name,
      (unsigned long long)region_bytes,(unsigned long long)mesh_arena_pages(M),M->entries);
  uint64_t liveness=0;
  while(!stop){
    now=clock_ns();
    if(atomic_load_explicit(&found,memory_order_acquire))ports_found();
    terminations_turn();
    arrivals_turn();
    ports_stalled();
    if(find_wanted && !finding && !atomic_load_explicit(&stalled,memory_order_relaxed) && now>=windows_end()){find_wanted=0;ports_find();}
    for(uint32_t p=0;p<port_count;p++){
      struct port *port=ports+p;
      if(port->job && atomic_load_explicit(&port->done,memory_order_acquire))port_opened(port);
      port_open(port);
      port_hello(port);
      link_turn(port);
      child_turn(port);
    }
    health_turn();
    int alive=now>=liveness;
    if(alive){
      liveness=now+MESH_PAUSE_NS;
      record_retell();
    }
    clients_turn(alive);
    if(lefts)switch_settle();
    struct kevent64_s ready[16];
    const uint64_t due=turn_due(liveness),at=clock_ns(),rest=due>at?due-at:0;
    struct timespec tick={(time_t)(rest/1000000000u),(long)(rest%1000000000u)};
    int count=kevent64(events,NULL,0,ready,16,0,&tick);
    now=clock_ns();
    for(int i=0;i<count;i++){
      struct kevent64_s *e=ready+i;
      if(e->filter==EVFILT_SIGNAL){stop=1;stop_reason=(int64_t)e->ident;}
      /* design/transport.md section 14, "A process a port": a port's child's message or exit */
      else if(e->udata!=UINT64_MAX && (e->udata&TAG_CHILD)){
        struct port *port=(e->udata&0xff)<port_count?ports+(e->udata&0xff):NULL;
        if(port && e->filter==EVFILT_PROC && (pid_t)e->ident==port->child)child_reaped(port);
        else if(port && e->filter==EVFILT_PROC && (pid_t)e->ident==port->elsewhere)port->elsewhere=0;
        else if(port && e->filter==EVFILT_READ && (int)e->ident==port->child_fd)child_heard(port);
      }
      else if(e->filter==EVFILT_PROC){
        for(uint32_t s=0;s<MESH_CLIENTS;s++){
          uint64_t owner=atomic_load(&((struct mesh_client *)((char *)M+M->client_off))[s].owner);
          if(owner && (uint32_t)owner==(uint32_t)e->ident)mesh_leave(M,owner);
        }
      }
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)listener){
        struct sockaddr_in6 from;socklen_t size=sizeof from;int accepted;
        while((accepted=accept(listener,(struct sockaddr *)&from,&size))>=0){
          struct port *port=port_at(from.sin6_scope_id);size=sizeof from;
          if(!port || port->control>=0 || !port->open){close(accepted);continue;}
          port->control=accepted;
          port_greet(port);
        }
      }
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)hello)hello_heard();
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)network)network_heard(0);
      else if(e->filter==EVFILT_READ && addresses>=0 && e->ident==(uint64_t)addresses)addresses_heard();
      else if(e->udata<port_count && (int)e->ident==ports[e->udata].control){
        struct port *port=ports+e->udata;
        if(e->filter==EVFILT_WRITE && port->connecting){
          int error=0;socklen_t size=sizeof error;
          if(getsockopt(port->control,SOL_SOCKET,SO_ERROR,&error,&size))error=errno;
          if(error)port_lost(port,error);
          else {port->connecting=0;port_greet(port);}
        }
        else if(e->filter==EVFILT_WRITE){port->writing=0;port_flush(port);}
        else control_heard(port);
      }
    }
  }
  char why[96];farewell_text(stop_reason,why,sizeof why);
  say("bridge node %u: stopping (%s)\n",node,why);
  /* design/transport.md: FAREWELL on every control socket (queued as the socket takes it; the kernel sends
     what it has after the close), every lane no draw's, then the bridge ends (what its lanes have the next bridge's
     adoption gives back: region_adopt); no queue, thread or verbs call waited on, and every object the kernel's to release at the exit
     (design/RDMA-KERNEL-RECOVERY.md "October 18"), the exit no sooner than every port's quiet window ends
     (design/transport.md B9: its reclaim never inside the termination a conclusion may follow) and a peer
     interface's termination's (`exit_at`, B9b), the kernel's link events read meanwhile, so a link going off then
     extends its port's (network_heard), as a termination then extends them all (peer_terminated), each no later than
     MESH_PLATFORM_NS past the stop's start (`exit_by`: every window set before the stop ends by then, so the wait is
     the one 5 s budget, well inside its heartbeat's bound of 3 MESH_PLATFORM_NS) */
  for(uint32_t p=0;p<port_count;p++){
    child_send(ports+p,(struct child_message){.kind=CHILD_HALT});
    port_say(ports+p,(struct message){.kind=FAREWELL,.code=stop_reason},NULL,0);
    if(ports[p].control>=0){close(ports[p].control);ports[p].control=-1;}
  }
  lanes_cancel(NULL);
  /* design/transport.md section 2: every stop keeps the region for the next bridge to adopt; a release is the
     heartbeat's, which expires the region once this bridge has ended (mesh-heartbeat.c region_expire) */
  atomic_store(&M->bridge_pid,0);
  mesh_notify(&M->control);
  const uint64_t exit_by=clock_ns()+MESH_PLATFORM_NS;
  for(int told=0;;){
    now=clock_ns();
    const uint64_t windows=windows_end(),latest=MIN(windows,exit_by);
    if(latest<=now)break;
    if(!told++)say("bridge node %u: its exit put off %.1f s, to its quiet windows' end (design/transport.md)\n",node,
                   (double)(latest-now)/1e9);
    struct pollfd heard={.fd=network,.events=POLLIN};
    if(poll(&heard,1,(int)((latest-now+999999)/1000000))>0){now=clock_ns();network_heard(1);}
  }
  /* design/transport.md section 14, "A process a port": each child asked to end, and its exit waited on, at most
     MESH_PLATFORM_NS (its own windows its own: child_leave), so no child of this launch has its device's queue pairs
     as the next launch's children make theirs: each exit heard on a queue of the children's exits alone, and a child
     counted exited only once its exit is whole (mesh-observe.h mesh_exits_wait: this bridge is its parent).  This
     wait stays beside the next bridge's region_children and child_elsewhere because it alone sees an exit through to
     its end: a process in its exit is no longer found by its pid, before the kernel releases its device's objects
     (design/transport.md section 2), so neither of those sees a child of this launch whose exit has begun and not
     ended.  Children still alive at the bound are counted in the log; the next bridge waits only for those whose exit
     has not begun (region_children, which ends that bridge where one outlives its own bound), and nothing waits for
     one already in its exit */
  for(uint32_t p=0;p<port_count;p++)child_send(ports+p,(struct child_message){.kind=CHILD_END});
  struct mesh_process children[MESH_PORTS];
  uint32_t count=0;
  for(uint32_t p=0;p<port_count;p++)if(ports[p].child)children[count++]=mesh_bridge_pages(M)->child[ports[p].child_entry];
  const int exits=kqueue();
  if(exits<0)die("bridge: no queue for its children's exits");
  const uint32_t alive=mesh_exits_wait(exits,children,count,0,clock_ns()+MESH_PLATFORM_NS);
  if(alive)say("bridge node %u: %u of its children alive %.1f s after they were asked to end (the next bridge waits for "
               "those not yet in their exit)\n",node,alive,(double)MESH_PLATFORM_NS/1e9);
  return 0;
}
