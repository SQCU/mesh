#include "mesh-bridge.h"
#include <spawn.h>
#include <netinet/tcp.h>
#include <sys/wait.h>
#include <crt_externs.h>
#include <libproc.h>
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
   - ENDING: the bridge ended once every child, each halted at the stop, is reaped (child_reaped, its exit seen
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
static int child_halt(const struct port *port,uint32_t reason){
  if(!mesh_port_move(switch_of(),port->index,1u<<CHILD_STARTED|1u<<CHILD_DEVICE_OPEN|1u<<CHILD_LANES_MADE|1u<<CHILD_SERVING,
                     CHILD_LEAVING,0,0))return 0;
  if(reason)atomic_store(&child_reason,reason);
  return 1;
}

/* a lane's failure, found by the poller: its child halted (child_halt, EIO: nothing more is posted on any lane
   of the link, as at a failed completion, link_failed, and no draw takes any of them from the next pass on), and the
   link lost by its parent on the report */
static void lane_lost(struct port *port,uint32_t lane,int64_t code){
  if(child_halt(port,EIO))
    say("link %u (%s): lane %u failed: %s; the link toward node %u lost\n",port->index,port->name,lane,
        strerror((int)(code>0?code:EIO)),port->peer);
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
/* design/transport.md: the link's first failed completion, on any lane, or a failed poll: its child halted
   (child_halt, EIO), so nothing is posted on it again, and the link lost by its parent on the report */
static void link_failed(struct port *port,int64_t code,uint32_t domain){
  if(child_halt(port,EIO))
    say("link %u (%s): a completion failed (code %lld, domain %u): no lane served, the link toward node %u lost\n",
        port->index,port->name,(long long)code,domain,port->peer);
}

/* ---- design/PAGE-TABLE-SPEC-20261122.md section 5: the switch's data path, a port child's part (mesh-switched.h) ----
   Lane j of the link (its queue pair j) is the switch's lane port 2 MESH_LINK_STREAMS + j, of mesh-switched.h
   mesh_switch_lane_frames(j) frames (lane 0 short, every other long; both ends' plans alike, so both ends' lane j
   alike, TN3205).  On it the child
   keeps receives posted (one frame count for every SEND and every receive on the lane, TN3205): filled from free pages
   of the lane's class, and each one a SEND fills posted again at once, into a free page or, where the class has none,
   into the page that landed (whose landing is then lost), so a lane's posted receives do not run out while every page
   of its class is used; it posts what its port's inbound rings have, a page a SEND from the page's own memory on a
   lane of the page's class, each in the lane's in-flight ring; and gives its SENDs' completions to the switch's
   estimators, each dropping its page's use.  What lands is set in the page table and forwarded by its header; what
   this node's clients write is taken from their rings and carried.  Nothing is retained: what cannot go is counted
   (mesh-switched.h struct mesh_switch_counters), never retried.  The child's log ring, counters and row of inbound
   rings are its port's */
static inline uint32_t switch_lane(const struct port *port,uint32_t j){ return port->index*2*MESH_LINK_STREAMS+j; }
/* page pd made a receive of lane j (its state RECEIVE on the switch's lane), before it is posted: a child's end
   gives it back from then on, posted or not (mesh-switched.h mesh_switch_receives_drop) */
static inline void receive_page(const struct port *port,uint32_t j,uint32_t pd){
  struct mesh_pd *p=mesh_pd_at(M,pd);
  atomic_store_explicit(&p->lane,switch_lane(port,j),memory_order_relaxed);
  atomic_store_explicit(&p->state,MESH_PD_RECEIVE,memory_order_release);
}
/* pages page[0 .. n) (each a receive of lane j: receive_page) posted as the lane's receives, each the lane's
   frames from its page's start, as one list a post_recv (its call's cost the list's, not each receive's); those the
   provider did not take go back to the pool, and its error loses the lane: 0, or the error */
#define RECEIVE_BATCH 16u
static int receives_post(struct port *port,uint32_t j,const uint32_t *page,uint32_t n){
  struct link *l=&port->link;
  struct lane_state *state=l->lane+j;
  const uint64_t bytes=mesh_switch_lane_bytes(switch_lane(port,j));
  const struct mesh_queue *queue=l->provider.queues+j;
  struct ibv_recv_wr request[RECEIVE_BATCH];struct ibv_sge span[RECEIVE_BATCH];
  for(uint32_t k=0;k<n;k++){
    span[k]=dma_span(port,(struct mesh_dma_span){mesh_pd_at(M,page[k])->offset,bytes});
    request[k]=(struct ibv_recv_wr){.wr_id=((uint64_t)j<<48)|page[k],.sg_list=span+k,.num_sge=1};
    if(k)request[k-1].next=request+k;
  }
  struct ibv_recv_wr *bad=NULL;
  const int error=queue->receive(queue->pair,request,&bad);
  const uint32_t made=error?(bad>=request && bad<request+n?(uint32_t)(bad-request):0):n;
  for(uint32_t k=made;k<n;k++)mesh_pool_push(M,page[k]);
  state->posted+=made;
  atomic_store_explicit(&lanes_of(port,j/2)->posts[1].posted[j&1],state->posted,memory_order_relaxed);
  if(error)lane_lost(port,j,error<0?-error:error);
  return error;
}
/* the lane's receives filled from its class's free stack to as many as its queues' share takes: its first fill (each
   receive a SEND fills is posted again by link_receive, so the count stays there), and a receive a completion took
   that named no receive page; a stack found empty posts nothing more this pass and is counted (`starved`: the provider
   stalls the peer's SEND meanwhile) */
static void switch_receives(struct port *port,uint32_t j){
  struct lane_state *state=port->link.lane+j;
  const uint64_t frames=mesh_switch_lane_frames(j),most=state->frame_capacity/frames;
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
/* lane j's SENDs request[0 .. n) posted as one list (the provider's call costs about 0.85 us whatever it carries: a
   request list a call, not a call a request), their entries already in the lane's in-flight ring from `first` on and
   its posted count past them (switch_sends): each one the provider took counted, its wait from its push to the post
   counted and its entry given its post time; each one it did not take (those from `bad` on) not posted: the lane's
   posted count moved back to the last one posted, then each such page's use dropped and its frames off the cable's
   queue, counted `lost`, and the lane lost (lane_lost).  The count moves back before any of those uses is dropped, as
   link_send moves `completed` before its page's drop: a child that ends between the two leaves those pages in use
   (its parent's mesh_switch_flight_drop reads the count moved back), and never has a use dropped twice.  1, or 0
   where the post failed */
static int sends_post(struct port *port,uint32_t j,struct ibv_send_wr *request,uint32_t n,uint64_t first){
  struct mesh_switch *sw=switch_of();
  struct mesh_switch_lane *x=sw->lane+switch_lane(port,j);
  const struct mesh_queue *queue=port->link.provider.queues+j;
  struct ibv_send_wr *bad=NULL;
  const uint64_t posting=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  const int error=queue->send(queue->pair,request,&bad);
  const uint64_t posted=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  /* the requests before `bad` posted (all of them without an error) */
  const uint32_t made=error?(bad>=request && bad<request+n?(uint32_t)(bad-request):0):n;
  if(made<n)atomic_store_explicit(&x->posted,first+made,memory_order_relaxed);
  atomic_store_explicit(&lanes_of(port,j/2)->posts[0].posted[j&1],first+made,memory_order_relaxed);
  MESH_SWITCH_COUNT(sw,port->index,post_call_ns,posted-posting);
  for(uint32_t k=0;k<made;k++){
    struct mesh_switch_entry *e=x->flight+(first+k)%MESH_SWITCH_FLIGHT;
    MESH_SWITCH_COUNT(sw,port->index,posts,1);MESH_SWITCH_COUNT(sw,port->index,post_wait_ns,posting>e->at?posting-e->at:0);
    e->at=posted;
  }
  /* each drop a release (mesh.h mesh_pd_drop's compare-exchange), so the count's store above is ordered before it */
  for(uint32_t k=made;k<n;k++){
    atomic_fetch_sub_explicit(&sw->cable[port->index].queued,mesh_switch_lane_bytes(switch_lane(port,j)),memory_order_relaxed);
    mesh_pd_drop(M,x->flight[(first+k)%MESH_SWITCH_FLIGHT].pd);
    MESH_SWITCH_COUNT(sw,port->index,lost,1);
  }
  if(error)lane_lost(port,j,error<0?-error:error);
  return !error;
}
/* The SENDs of a pass, a class at a time: each page the port's inbound rings of the class have (every pushing port's,
   its own among them, from the one after where the last pass began: mesh-switched.h struct mesh_switch_inbound), in
   each ring's order, a SEND of its class's frames from the page's start on the served lane of its class (never a lane
   of more frames) with the fewest SENDs outstanding that has room for one more, its in-flight ring below its depth (the
   queue's own capacity's share: a post past the queue's capacity fails); while none has room, the rest wait for a
   later pass.  Each SEND's entry is put in its lane's in-flight ring at the lane's next position, which its wr_id
   names (lane j << 48 | the position's low 48 bits), each ring's head moved past what it gave, then each lane's posted
   count past its SENDs, then one post_send a lane (sends_post): a page is on an inbound ring or an in-flight ring
   whenever a child's end may find it, but between those two stores.  A failed post stops the pass's SENDs */
#define SEND_BATCH 16u
static void switch_sends(struct port *port,uint32_t served){
  struct link *l=&port->link;
  struct mesh_switch *sw=switch_of();
  const uint32_t from=l->send_from++%MESH_PORTS;
  for(uint32_t c=0;c<MESH_CLASSES;c++){
    const uint32_t frames=mesh_switch_class_frames(c),lanes=served&mesh_switch_class_plan(frames);
    if(!lanes)continue;
    const uint64_t bytes=(uint64_t)frames*MESH_FRAME;
    for(;;){
      struct ibv_send_wr request[MESH_LINK_LANES][SEND_BATCH];struct ibv_sge span[MESH_LINK_LANES][SEND_BATCH];
      uint64_t first[MESH_LINK_LANES]={0},out[MESH_LINK_LANES]={0};uint32_t n[MESH_LINK_LANES]={0},all=0;
      int room=1,counted=0;
      for(uint32_t q=0;q<MESH_PORTS && room;q++){
        struct mesh_switch_inbound *r=mesh_switch_inbound_of(sw,(from+q)%MESH_PORTS,port->index,c);
        uint64_t head=atomic_load_explicit(&r->head,memory_order_relaxed);
        const uint64_t tail=atomic_load_explicit(&r->tail,memory_order_acquire);
        /* the lanes' SENDs outstanding read once a pass's class, where a ring has a page for them */
        for(uint32_t k=head<tail && !counted?lanes:0;k;k&=k-1){
          const uint32_t j=(uint32_t)__builtin_ctz(k);
          const struct mesh_switch_lane *x=sw->lane+switch_lane(port,j);
          first[j]=atomic_load_explicit(&x->posted,memory_order_relaxed);
          out[j]=first[j]-atomic_load_explicit(&x->completed,memory_order_relaxed);
          counted=1;
        }
        for(;head<tail;head++){
          uint32_t j=UINT32_MAX;uint64_t fewest=UINT64_MAX;
          for(uint32_t k=lanes;k;k&=k-1){
            const uint32_t i=(uint32_t)__builtin_ctz(k);
            const uint64_t outstanding=out[i]+n[i];
            if(n[i]<SEND_BATCH && outstanding<l->lane[i].depth && outstanding<fewest){fewest=outstanding;j=i;}
          }
          if(j==UINT32_MAX){room=0;break;}
          const struct mesh_switch_entry e=r->entry[head%MESH_SWITCH_INBOUND];
          const uint64_t at=first[j]+n[j];
          sw->lane[switch_lane(port,j)].flight[at%MESH_SWITCH_FLIGHT]=e;
          span[j][n[j]]=dma_span(port,(struct mesh_dma_span){mesh_pd_at(M,e.pd)->offset,bytes});
          request[j][n[j]]=(struct ibv_send_wr){.wr_id=((uint64_t)j<<48)|(at&((UINT64_C(1)<<48)-1)),.sg_list=span[j]+n[j],.num_sge=1,
                                                .opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED};
          if(n[j])request[j][n[j]-1].next=request[j]+n[j];
          n[j]++;all++;
        }
        atomic_store_explicit(&r->head,head,memory_order_relaxed);
      }
      if(!all)break;
      for(uint32_t k=lanes;k;k&=k-1){
        const uint32_t j=(uint32_t)__builtin_ctz(k);
        if(n[j])atomic_store_explicit(&sw->lane[switch_lane(port,j)].posted,first[j]+n[j],memory_order_relaxed);
      }
      int filled=0,failed=0;
      for(uint32_t k=lanes;k;k&=k-1){
        const uint32_t j=(uint32_t)__builtin_ctz(k);
        if(!n[j])continue;
        filled|=n[j]==SEND_BATCH;
        failed|=!sends_post(port,j,request[j],n[j],first[j]);
      }
      if(failed)return;
      if(!filled)break;
    }
  }
}
/* the cost-to-go this child's poller keeps and its draws read (mesh-switched.h struct mesh_switch_togo), in its own
   memory: a step each idle pass (port_pass) */
static struct mesh_switch_togo togo;
/* design/transport.md: the link's SEND completions (one naming no lane of the plan dropped; a failed one fails the
   link, link_failed): each one's in-flight entry (at the position its wr_id names: a queue pair completes its SENDs in
   their order) given back, the lane's completed count one more, and its page's use dropped, through the switch's
   estimators (its post, as its entry stored it, its service from the later of its post and the lane's previous
   completion, its completion); then the cable's terms from the poll (mesh-switched.h
   mesh_switch_cable_sent: its bytes since the link's previous completing poll, `completed_at`, and the time of a SEND
   that had the cable to itself, the only one outstanding as the poll began).  The completions polled, 0 none */
static int link_send(struct port *port,uint32_t served){
  struct link *l=&port->link;
  struct mesh_switch *sw=switch_of();
  struct ibv_wc done[16];
  const int count=l->provider.poll(l->provider.sent,16,done);
  if(count<0){link_failed(port,EIO,2);return 0;}
  if(!count)return 0;
  const uint64_t at=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  uint64_t outstanding=0,bytes=0,first=UINT64_MAX,alone_posted=0;uint32_t alone=UINT32_MAX;
  for(uint32_t j=0;j<MESH_LINK_LANES;j++)if(served>>j&1){
    const struct mesh_switch_lane *x=sw->lane+switch_lane(port,j);
    outstanding+=atomic_load_explicit(&x->posted,memory_order_relaxed)-atomic_load_explicit(&x->completed,memory_order_relaxed);
  }
  for(int i=0;i<count;i++){
    const uint32_t j=link_lane(l,done[i].qp_num);
    if(j>=MESH_LINK_LANES || !(served>>j&1))continue;
    if(done[i].status){link_failed(port,(int64_t)done[i].status,2);return count;}
    struct lane_state *state=l->lane+j;
    struct mesh_switch_lane *x=sw->lane+switch_lane(port,j);
    const struct mesh_switch_entry *e=x->flight+done[i].wr_id%MESH_SWITCH_FLIGHT;
    const uint32_t pd=e->pd;const uint64_t posted=e->at,completed=atomic_load_explicit(&x->completed,memory_order_relaxed)+1;
    atomic_store_explicit(&x->completed,completed,memory_order_relaxed);
    const uint64_t start=posted>state->last?posted:state->last,wire=mesh_switch_lane_bytes(switch_lane(port,j));
    mesh_switch_sent(sw,switch_lane(port,j),wire,posted,start,at);
    bytes+=wire;first=MIN(first,posted);
    if(outstanding==1 && posted>=l->completed_at){alone=mesh_switch_class(mesh_switch_lane_frames(j));alone_posted=posted;}
    mesh_pd_drop(M,pd);
    MESH_SWITCH_COUNT(sw,port->index,sent,1);MESH_SWITCH_COUNT(sw,port->index,send_ns,at-posted);
    state->last=at;
    struct mesh_lanes *record=lanes_of(port,j/2);
    atomic_store_explicit(&record->send_ns,atomic_load_explicit(&record->send_ns,memory_order_relaxed)+(at-posted),memory_order_relaxed);
    atomic_store_explicit(&record->sends,atomic_load_explicit(&record->sends,memory_order_relaxed)+1,memory_order_relaxed);
    atomic_store_explicit(&record->posts[0].done[j&1],completed,memory_order_relaxed);
  }
  if(!bytes)return count;
  mesh_switch_cable_sent(sw,port->index,bytes,first,l->completed_at,at,alone,alone_posted);
  l->completed_at=at;
  return count;
}
/* the link's receive completions, each lane's receive posted again at once, so a lane's posted receives stay as
   many as its fill made them whatever the pool has (TN3205: a receive for every SEND the peer posts): into a page
   popped from the lane's class, and the page that landed landed (mesh-switched.h mesh_switch_landed: its lines taken,
   its destinations here set in the table, the rest switched on); where the class's free stack is empty, into the page
   that landed, which is then not landed, counted (`lost`).  The receives of a poll posted as one list a lane
   (receives_post).  Nothing of a completion is tested but its lane (one this pass serves) and its status (a failed one
   fails the link: link_failed): its page is the one its request named, landed as it is.  The completions polled, 0
   none */
static int link_receive(struct port *port,uint32_t served){
  struct link *l=&port->link;
  struct mesh_switch *sw=switch_of();
  struct ibv_wc done[RECEIVE_BATCH];
  const int count=l->provider.poll(l->provider.completion,RECEIVE_BATCH,done);
  if(count<0){link_failed(port,EIO,3);return 0;}
  if(count)atomic_thread_fence(memory_order_acquire);
  const uint64_t at=count?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
  uint32_t again[MESH_LINK_LANES][RECEIVE_BATCH],agains[MESH_LINK_LANES]={0};
  for(int i=0;i<count;i++){
    const uint32_t j=link_lane(l,done[i].qp_num);
    if(j>=MESH_LINK_LANES || !(served>>j&1))continue;
    if(done[i].status){link_failed(port,(int64_t)done[i].status,2);return count;}
    struct lane_state *state=l->lane+j;
    /* the page this child posted the receive into, its request's low 32 bits (receives_post), clamped into the pool's
       pages: a receive's completion on its own queue (the receive completion queue has no other kind) names it */
    const uint32_t pages=mesh_pool(M)->pages,named=(uint32_t)(done[i].wr_id&UINT32_MAX),pd=named<pages?named:pages-1;
    state->got++;
    atomic_store_explicit(&lanes_of(port,j/2)->posts[1].done[j&1],state->got,memory_order_relaxed);
    const uint32_t fresh=mesh_pool_pop(M,mesh_switch_lane_frames(j));
    if(fresh==MESH_ABSENT){again[j][agains[j]++]=pd;MESH_SWITCH_COUNT(sw,port->index,lost,1);continue;}
    receive_page(port,j,fresh);
    again[j][agains[j]++]=fresh;
    mesh_switch_landed(M,sw,switch_lane(port,j),pd,at,port->index,&togo);
  }
  for(uint32_t j=0;j<MESH_LINK_LANES;j++)if(agains[j])receives_post(port,j,again[j],agains[j]);
  return count;
}
/* the client rings this child takes: ring s the (s mod n)-th, by port index, of the n ports whose child serves (their
   lines CHILD_SERVING with a lane: `line`, every port's as the pass read it), bits over the slots; none while this
   port does not serve.  Made again only as the serving ports change (`serving`, the child's copy of them, a bit a
   port) */
static uint64_t switch_mine(const struct port *port,const uint64_t *line){
  static uint32_t serving=UINT32_MAX;
  static uint64_t mine;
  uint32_t serves=0;
  for(uint32_t p=0;p<MESH_PORTS;p++)if(mesh_port_state(line[p])==CHILD_SERVING && mesh_port_plan(line[p]))serves|=UINT32_C(1)<<p;
  if(serves==serving)return mine;
  serving=serves;mine=0;
  if(serving>>port->index&1){
    const uint32_t n=(uint32_t)__builtin_popcount(serving),r=(uint32_t)__builtin_popcount(serving&((UINT32_C(1)<<port->index)-1));
    for(uint32_t s=r;s<MESH_CLIENTS;s+=n)mine|=UINT64_C(1)<<s;
  }
  return mine;
}
/* slot `slot`'s ring, flagged and found empty by the child that takes it: its flag cleared where the slot is vacant,
   then its owner and its put read again and the ring flagged again where either moved (mesh.h mesh_ring_flag: each
   of these sequentially consistent), so a client that takes the slot and writes meanwhile is never missed */
static void ring_vacant(uint32_t slot,struct mesh_ring *r){
  _Atomic uint64_t *owner=&((struct mesh_client *)((char *)M+M->client_off)+slot)->owner;
  if(atomic_load(owner))return;
  const uint64_t bit=UINT64_C(1)<<slot;
  atomic_fetch_and(&M->attached,~bit);
  if(atomic_load(owner) || atomic_load(&r->put)>atomic_load(&r->take))atomic_fetch_or(&M->attached,bit);
}
/* the writes this node's clients put on the rings this child takes (its share, switch_mine, of the rings flagged
   attached: mesh.h mesh_ring_flag), taken and carried, at most SWITCH_BURST a pass, each of those rings looked at from
   the slot after the one where this child began its last pass, whichever client the slot is now (a write is carried
   on its own ring use, its writer there or exited); a flagged ring found empty whose slot is vacant has its flag
   cleared (ring_vacant), so the scan reads only the rings of clients that write.  As the serving ports change, the
   child whose share a ring was may take from it a moment longer, and the ring's compare-exchange gives each write to
   one child.  The writes taken */
#define SWITCH_BURST 16u
static uint32_t switch_takes(struct port *port,const uint64_t *line){
  struct mesh_switch *sw=switch_of();
  struct link *l=&port->link;
  const uint64_t mine=switch_mine(port,line)&atomic_load_explicit(&M->attached,memory_order_relaxed);
  const uint32_t from=l->take_from%MESH_CLIENTS;
  uint32_t taken=0;uint64_t at=0;
  for(uint64_t k=from?mine>>from|mine<<(MESH_CLIENTS-from):mine;k && taken<SWITCH_BURST;k&=k-1){
    const uint32_t slot=((uint32_t)__builtin_ctzll(k)+from)%MESH_CLIENTS;
    struct mesh_ring *r=mesh_ring_at(M,slot);
    if(!mesh_ring_pending(r)){ring_vacant(slot,r);continue;}
    if(!at)at=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    taken+=mesh_switch_ring_dequeue(M,sw,slot,SWITCH_BURST-taken,at,port->index,&togo);
  }
  l->take_from=(from+1)%MESH_CLIENTS;
  return taken;
}

/* design/PAGE-TABLE-SPEC-20261122.md section 5: one pass of a port's poller over the lanes of its link's plan (`served`),
   made only while its port's line is CHILD_SERVING (every port's line read once a pass: a line its parent, a halt, a
   lane's failure, a termination or a fault moved to leaving stops every post from the next pass on): the receive
   completions landed (link_receive), the receives topped up, the clients' writes taken (switch_takes, the port's share
   of the rings from the same lines), the port's inbound rings posted (switch_sends), the SEND completions (link_send),
   the pass counted (`passes`); a pass that polled no completion and took no write takes one step of its poller's
   cost-to-go (mesh-switched.h mesh_switch_togo_step), so a busy pass gains nothing.  1 after a pass, 0 where the line
   no longer serves (no pass made: the poller returns) */
static int port_pass(struct port *port){
  /* one pass in 1024 timed (counters pass_ns: its time summed and the passes timed; part_ns: each part's time) */
  static __thread uint32_t timing;
  struct link *l=&port->link;
  struct mesh_switch *sw=switch_of();
  uint64_t line[MESH_PORTS];
  mesh_switch_ports_read(sw,line);
  if(mesh_port_state(line[port->index])!=CHILD_SERVING)return 0;
  const int timed=!(++timing&1023);
  uint64_t t[6]={0};
  if(timed)t[0]=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  const uint32_t served=l->plan,lanes=plan_lanes(served);
  uint32_t worked=(uint32_t)link_receive(port,served);
  if(timed)t[1]=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  for(uint32_t j=0;j<lanes;j++)if(served>>j&1)switch_receives(port,j);
  if(timed)t[2]=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  worked|=switch_takes(port,line);
  if(timed)t[3]=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  switch_sends(port,served);
  if(timed)t[4]=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  worked|=(uint32_t)link_send(port,served);
  if(!worked)mesh_switch_togo_step(sw,&togo);
  MESH_SWITCH_COUNT(sw,port->index,passes,1);   /* the pass time is the wall time over these (the thread spins) */
  if(timed){
    t[5]=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    MESH_SWITCH_COUNT(sw,port->index,pass_ns[0],t[5]-t[0]);MESH_SWITCH_COUNT(sw,port->index,pass_ns[1],1);
    for(uint32_t k=0;k<5;k++)MESH_SWITCH_COUNT(sw,port->index,part_ns[k],t[k+1]-t[k]);
  }
  struct mesh_lanes *counted=lanes_of(port,0);
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
  struct port *port=argument;
  pthread_setname_np("mesh.rdma");
  /* fault_heard's handler on a stack of its own */
  stack_t alternate={.ss_sp=malloc(SIGSTKSZ),.ss_size=SIGSTKSZ};
  if(alternate.ss_sp)sigaltstack(&alternate,NULL);
  mesh_switch_home=port->index;   /* this thread posts its own port's lanes (mesh-switched.h mesh_switch_cables_toward) */
  while(port_pass(port)){}
  child_trigger(POLLER_RETURNED);
  return NULL;
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
static int link_lapsed(const struct port *port){
  iokit_heard();
  return mesh_port_state(atomic_load(&switch_of()->port[port->index]))==CHILD_LEAVING?ECANCELED:0;
}
static int link_make(struct port *port){
  struct link *l=&port->link;
  struct mesh_verbs *p=&l->provider;
  struct ibv_port_attr attributes;
  int error=0;
  /* design/transport.md: the port's own address (its attributes and GID) read again at the make, as its opening read
     it: a peer's restart can move the port's GID table, and LINKs naming the address of the port's opening failed every
     lane's connection at the peer with EIO (2026-10-19, after the Mini's power loss) */
  if(ibv_query_port(port->device.context,1,&attributes))return errno?errno:EIO;
  port->device.port=attributes;
  if((error=device_gid(&port->device,port->name)) || (error=link_lapsed(port)) || (error=pair_up(p,plan_lanes(l->plan))))return error;
  l->share=(struct frames){(uint64_t)MIN(p->completion->cqe,p->sent->cqe)/MAX(plan_lanes(l->plan),1u)};
  return 0;
}
static void lane_reset(struct port *port,uint32_t k);
/* design/transport.md section 14: stream s's identity, alike at both ends: FNV-1a over its first lane's two ends, each
   its port's GID and its queue pair's number, the lesser end first; never 0 */
static uint64_t stream_wire(const struct port *port,const struct child_message *heard,uint32_t s){
  const uint32_t k=2*s;
  uint8_t end[2][20];
  memcpy(end[0],port->gid,16);memcpy(end[0]+16,&port->link.qpn[k],4);
  memcpy(end[1],heard->gid,16);memcpy(end[1]+16,&heard->peer[k][0],4);
  const int first=memcmp(end[0],end[1],20)>0;
  uint64_t h=UINT64_C(14695981039346656037);
  for(int e=0;e<2;e++)
    for(int i=0;i<20;i++){h^=end[e^first][i];h*=UINT64_C(1099511628211);}
  return h?h:1;
}
static int link_connect(struct port *port,const struct child_message *heard){
  struct link *l=&port->link;
  struct mesh_verbs *p=&l->provider;
  const uint32_t plan=l->plan;
  int error=0;
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1)error=queue_receive(p,(int)k,heard->peer[k],heard->lid,heard->gid);
  if(!error)error=link_lapsed(port);
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1)error=queue_send(p,(int)k);
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)l->qpn[k]=plan>>k&1?p->queues[k].pair->qp_num:UINT32_MAX;
  if(!error)error=link_lapsed(port);
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1)lane_reset(port,k);
  for(uint32_t s=0;s<plan_streams(plan) && !error;s++)l->wire[s]=stream_wire(port,heard,s);
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
static int lanes_open(struct port *port,uint32_t node,uint64_t bandwidth){
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
  }
  struct mesh_switch *sw=switch_of();
  for(uint32_t j=0;j<MESH_LINK_LANES;j++){
    struct mesh_switch_lane *x=sw->lane+switch_lane(port,j);
    atomic_store_explicit(&x->posted,0,memory_order_relaxed);atomic_store_explicit(&x->completed,0,memory_order_relaxed);
  }
  mesh_switch_inbound_drop(M,sw,port->index,port->index);
  return mesh_port_move(sw,port->index,1u<<CHILD_LANES_MADE,CHILD_SERVING,node,l->plan);
}
/* design/transport.md section 14: a connected link served: its lane records opened and its line raised to serving
   (lanes_open), then its poller made (port_progress, at the progress threads' class), the one thread that posts on the
   device's queue pairs from here on: pthread_create orders every make, connect and lane state before its first post.
   The line is raised here, before the poller exists and before the child answers its connect, so the parent's record
   of the link (record_reach) and its node table (switch_view), each made as it takes the answer, read it serving.  A
   plan of no lane has nothing to poll: its line serves and no poller is made.  `polling` 1 from the poller's making to
   its return's event (child_main), the main thread's.  0, or why the link does not serve (ECANCELED: a halt came first) */
static int polling;
static int link_serve(struct port *port,uint32_t node,uint64_t bandwidth){
  if(!lanes_open(port,node,bandwidth))return ECANCELED;
  if(!port->link.plan)return 0;
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_set_qos_class_np(&attributes,QOS_CLASS_USER_INTERACTIVE,0);
  pthread_t created;
  const int error=pthread_create(&created,&attributes,port_progress,port);
  pthread_attr_destroy(&attributes);
  if(!error)polling=1;
  return error;
}
/* a LINK telling the peer every lane of this end's pairing, the lanes of its plan (`count` theirs): each one's queue pair
   number and first sequence number, as its child told them (CHILD_MADE), 0 for a lane the plan leaves out */
static void link_tell(struct port *port){
  const struct link *l=&port->link;
  struct message said={.kind=LINK,.count=plan_lanes(l->plan)};
  for(uint32_t k=0;k<MESH_LINK_LANES;k++)if(l->plan>>k&1){said.pair[k][0]=l->qpn[k];said.pair[k][1]=(l->psn+k)&0xffffff;}
  port_say(port,said,NULL,0);
}
/* the link of a port lost (its peer's loss, a failure, the bridge stopping), in the parent: no command of it out any
   more, so an answer of its child to a command of the pairing lost is dropped as it comes (child_heard), its lanes left
   as they are, no verbs call made, and, once connected, never paired again in this bridge's life (design/transport.md
   section 14) */
static void link_lost(struct port *port){
  struct link *l=&port->link;
  l->state=LINK_DOWN;l->heard=0;l->job=0;
}
/* a lane's progress made new at its connect: nothing posted, its receives its queues' share, and its depth (the
   SENDs it has outstanding at most) its queues' share of its send capacity, over its frames, the in-flight ring's size
   at most: each SEND on the lane covers its frames, so a post past the depth would pass the queue's capacity */
static void lane_reset(struct port *port,uint32_t k){
  struct link *l=&port->link;
  const struct frames room=frames_min(l->share,(struct frames){l->provider.queues[k].receive_capacity}),
                      sends=frames_min((struct frames){l->provider.queues[k].send_capacity},(struct frames){MAX(l->share.n,1)});
  l->lane[k]=(struct lane_state){.frame_capacity=room.n,.depth=MIN(sends.n/mesh_switch_lane_frames(k),(uint64_t)MESH_SWITCH_FLIGHT)};
}
/* the line of a port (`port` NULL: of every port) leaving, as its link is lost or the bridge stops, where its child
   has made its lanes (port_leaving): no draw takes its lanes and no child takes for it from the next pass on, even
   while its poller is stuck in a provider call, and what its rings have is dropped as its child ends
   (switch_child_free) */
void lanes_cancel(const struct port *port){
  for(uint32_t p=0;p<port_count;p++)if(!port || ports+p==port)port_leaving(ports+p);
}
/* design/transport.md section 14: the link's turn, on the event loop: a job's end taken (made: LINK told the peer every
   lane; connected: up, every lane served, the stream live, the peer sent the map's records it lacks and then this
   node's own naming the link); a job that failed loses the link (port_lost); then, while no port's job is out past
   MESH_PLATFORM_NS (B12) and once the port's last quiet window has ended (B9), the lanes made once the port is paired
   (`lanes`: its child asked for them), and connected once both ends' are made, each by the port's child and once in
   its life: a link lost once its lanes were asked for pairs again with the port's next child (design/transport.md
   section 14, "A process a port": port_lost, child_reaped) */
static void child_job(struct port *port,int job);
static void link_turn(struct port *port){
  struct link *l=&port->link;
  if(l->job){
    if(!atomic_load_explicit(&l->done,memory_order_acquire))return;
    const int job=l->job,result=l->result;
    l->job=0;
    if(result){
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
      /* the map's records the peer lacks sent as the link comes up (mesh-records.c map_tell), then this node's own
         naming the link: the map first, so a record this end's own then brings into its map goes to no peer that has
         it (map_make) */
      map_tell(port);
      record_changed();
    }
  }
  const int quiet=now<port->quiet_at;
  if(!port->greeted)return;
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
   socket closed, the map's phase STOPPED), no verbs call.  The call itself cannot be ended.  The port takes no job
   until the call returns (`stuck`, its job's word); each port's calls are its own child's, a process of its own
   (section 14), so one port's stuck call stalls no other port's (a stall of every port's was 10-18's, when one process
   made every port's calls).  Where its lanes were asked for, the child is ended by port_lost's CHILD_HALT, which it
   reads as the call returns, and the port waits for its exit (`recycle`); an open's port stays `stuck` until the call
   returns, then opens again (`opening`, port_open past its quiet window, B9; an open's own return is port_opened's)
   and so takes jobs again.  Read on the event loop's turn */
static void ports_stalled(void){
  for(uint32_t p=0;p<port_count;p++){
    struct port *port=ports+p;
    const struct link *l=&port->link;
    const int opening=port->job && !atomic_load_explicit(&port->done,memory_order_acquire) && now-port->job_at>=MESH_PLATFORM_NS;
    const int linking=l->job && !atomic_load_explicit(&l->done,memory_order_acquire) && now-l->job_at>=MESH_PLATFORM_NS;
    const int out=opening||linking;
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
    say("link %u (%s): node %u's LINK names %u lanes, this end's plan %u (plan %#x): the two ends planned apart\n",port->index,
        port->name,port->peer,heard->count,plan_lanes(l->plan),l->plan);
    port_lost(port,EPROTO);
    return;
  }
  memcpy(l->peer,heard->pair,sizeof l->peer);
  port->heard.lid=heard->lid;memcpy(port->heard.gid,heard->gid,16);
  l->heard=1;
  link_turn(port);
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
   class in runs the arena leases (mesh_arena_alloc: a run within one MESH_DMA_EXTENT, so a SEND of any of its pages,
   which covers no more than the page's frames, lies in one registration), a run half as long where the arena has none
   so long; every page free, the short pages' ids first */
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
    const uint32_t frames=c?MESH_CLASS_LONG:MESH_CLASS_SHORT;
    k->frames=frames;k->pages=0;
    atomic_store(&k->head,0);atomic_store(&k->free,0);
    uint32_t run=(uint32_t)(MESH_DMA_EXTENT/MESH_FRAME/frames);
    while(k->pages<want[c] && run){
      const uint32_t n=MIN(run,want[c]-k->pages);
      const uint64_t bytes=(uint64_t)n*frames*MESH_FRAME;
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
/* the switch made new over the pool (no SEND or ring entry of a bridge before survives its stop: region_adopt gave
   back what they had), and its stamps' clock set (design/PAGE-TABLE-SPEC-20261122.md section 5, mesh-switched.h
   mesh_switch_stamp): this node's boot (M->boot, the wall clock as the bridge started) plus its uptime since
   (CLOCK_UPTIME_RAW, the clock every process of the node reads), its offset from the uptime clock set once here.  A
   stamp orders this node's own lines, a restarted bridge's after its last one's (its boot later); no node reads it
   against its own clock (a line's age is counted on the node that keeps it: mesh_switch_line_cost) */
static void switch_make(void){
  struct mesh_switch *sw=mesh_switch_of(M);
  mesh_switch_init(sw);
  atomic_store_explicit(&sw->stamp_offset,(int64_t)M->boot-(int64_t)clock_gettime_nsec_np(CLOCK_UPTIME_RAW),memory_order_relaxed);
  if(getenv("MESH_SWITCH_LOG"))atomic_store(&sw->logging,1);
  const struct mesh_pool *pool=mesh_pool(M);
  say("bridge: switch made over a pool of %u pages (%u free short, %u free long)\n",pool->pages,
      atomic_load(&pool->cls[0].free),atomic_load(&pool->cls[1].free));
}
/* the view as the records stand: every record's node, and each node pair the records join (each end's entries toward
   the other, the fewer of the two), its cables and their bandwidth summed (bytes a ns; 0 where neither end's record
   reports one), at the hop latency the fabric's fits measure (6 us): each link's nominal terms, its cost until its
   sender's measures of it are heard (mesh-switched.h struct mesh_switch_cable, struct mesh_switch_line); every node of
   it an entry of the node table (the key its links' lines are kept under: struct mesh_switch_view `slot`), live (the
   bandwidth of this node's own cables toward it), every other entry not.  Made by the event loop each time it
   writes the map, after the write (mesh-records.c map_make): the map's one writer, so it reads the records and the
   ports as they stand, with no serial to check and nothing read again, and a map's change is drawable at once */
#define SWITCH_LATENCY_NS 6000.0
void switch_view(void){
  const uint64_t count=atomic_load_explicit(&M->map_nodes,memory_order_relaxed);
  const struct mesh_record *r=mesh_records(M);
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
   lanes are no draw's already (lanes_cancel).  A child that ended between passes (every end it was asked for or chose:
   its exit waits for its poller's return, child_main) held no page outside these; one that ended inside a pass (a
   crash, or its poller still inside a provider call at its end's bound) may have held one mid-carry,
   mid-landing or between an inbound ring and an in-flight ring, which is not found and stays in use (the operator's
   O4: accepted, no record) */
static void switch_child_free(struct port *port){
  struct mesh_switch *sw=mesh_switch_of(M);
  for(uint32_t j=0;j<2*MESH_LINK_STREAMS;j++)mesh_switch_flight_drop(M,sw,switch_lane(port,j),MESH_SWITCH_PARENT);
  mesh_switch_inbound_drop(M,sw,port->index,MESH_SWITCH_PARENT);
  mesh_switch_receives_drop(M,switch_lane(port,0),2*MESH_LINK_STREAMS);
  atomic_store(&sw->port[port->index],0);
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
/* the policy's refresh, on a thread of its own, started only where a theta is loaded (switch_policy_on, the mode
   chosen once as the bridge starts): every SWITCH_REFRESH_NS the cost-to-go made whole over the published view, as
   each port child makes its own a step a pass (mesh-switched.h mesh_switch_togo_whole), then the policy's tables
   from this node's tokens, published */
#define SWITCH_REFRESH_NS UINT64_C(50000)
static void *switch_watch(void *unused){
  (void)unused;
  pthread_setname_np("mesh.switch");
  struct mesh_switch *sw=mesh_switch_of(M);
  uint64_t point=0;
  static struct mesh_switch_tokens tokens;
  static struct mesh_switch_togo policy_togo;
  const struct mesh_switch_policy policy={NULL,switch_policy_splits};
  for(;;){
    mesh_switch_togo_whole(sw,&policy_togo);
    mesh_switch_policy_refresh(sw,&policy_togo,&policy,clock_ns(),++point,0,&tokens);
    MESH_SWITCH_COUNT(sw,MESH_SWITCH_PARENT,refreshes,1);
    struct timespec pause={0,(long)SWITCH_REFRESH_NS};
    nanosleep(&pause,NULL);
  }
}

/* design/transport.md: the bridge's wire, a message a kind: a record or the peer's LINK; a message of any other kind
   is let be (AGENTS.md, "Fail toward access": unknown input, keep going).  A peer that stops closes the socket, and
   its end is the link's loss (control_heard) */
static void message_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  if(heard->kind==RECORD)record_heard(port,payload,heard->bytes);
  else if(heard->kind==LINK)link_heard(port,heard);
}
/* a message heard on a port's control socket: before its peer's hello, that hello alone (any other let be); the hello
   names the peer (the node it states, whatever node's hello this end dialed), and pairs the link, its plan the fewer
   queue pairs of the two ends', design/HETEROGENEOUS-LINKS.md, the peer told none of the records yet
   (mesh-records.c records_forget), then sent this node's record (link_set) and the rest of the map's it lacks once
   the link is up, its lanes made (link_turn); a lane's frames are the same
   constants at both ends (mesh-switched.h mesh_switch_lane_frames).  Then each message is taken (message_heard) */
static void port_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  if(!port->greeted){
    if(heard->kind!=HELLO)return;
    /* a bridge of another wire version is not paired with: its hello is let be (mesh.h MESH_VERSION) */
    if(heard->version!=MESH_VERSION)return;
    port->peer=heard->node;port->peer_port=heard->port;port->heard=*heard;port->greeted=1;
    records_forget(port);
    port->link.plan=link_plan(MIN(port->budget,heard->count));
    struct mesh_link_info *link=&mesh_links(M)[port->index];
    atomic_store_explicit(&link->bandwidth,port->bandwidth,memory_order_relaxed);
    atomic_fetch_add_explicit(&link->pairings,1,memory_order_relaxed);
    say("link %u (%s): paired with node %u (queue pairs for the link: this port's %u, its %u; plan %#x, %u lanes, %u stream%s)\n",
        port->index,port->name,port->peer,port->budget,heard->count,port->link.plan,
        plan_lanes(port->link.plan),plan_streams(port->link.plan),plan_streams(port->link.plan)==1?"":"s");
    link_set(port,port->peer,MESH_PAIRED,0);
    if(port->greeted)link_turn(port);
    return;
  }
  message_heard(port,heard,payload);
}

/* design/transport.md section 14, #loss: a link lost has its port's line leaving where its lanes were made (the
   clients' rings its port took are then the other serving ports' shares, and its lanes no draw's: lanes_cancel), its
   lanes served no more before its control socket closes (no pass of its poller begins after the line leaves, so no
   post on them once the peer can read the loss), what they carried counted `lost` as its child's end is freed
   (switch_child_free); no verbs call is made (design/RDMA-KERNEL-RECOVERY.md "October 18").  A link whose child was
   asked for its lanes (`lanes`), or whose child halted itself (child_heard), ends that child: CHILD_HALT naming the
   loss, after which the child exits MESH_PLATFORM_NS on and once its own windows have ended (mesh-link.c child_end),
   its exit the kernel's release of the lanes; the port waits for that exit (`recycle`), closed and greeting no peer
   meanwhile, and pairs again with the port's next child ("A process a port").  So does every child once the bridge
   stops, its CHILD_HALT{0} already sent (node_stop), where it then takes the loss's later time.  A port lost before
   its lanes were asked for and still open keeps its child and says hello at once (port_hello: the records go on).
   Its quiet window begins (B9: no port job on it, and no exit, for MESH_PLATFORM_NS, the operator's bound on the
   termination a loss may follow) */
static int child_send(struct port *port,struct child_message said);
void port_lost(struct port *port,int64_t code){
  quiet_until(&port->quiet_at,now+MESH_PLATFORM_NS);
  link_lost(port);
  lanes_cancel(port);
  if(port->child && port->link.lanes && !stopping()){port->recycle=1;port->open=0;}
  if(port->child && (port->recycle || stopping()))
    child_send(port,(struct child_message){.kind=CHILD_HALT,.result=code>0?(int32_t)code:EIO});
  if(port->control>=0){close(port->control);port->control=-1;}
  if(port->greeted)port->next=now;
  /* design/transport.md: the peer named */
  if(port->greeted||port->connecting){
    if(stopping())say("link %u (%s): closed toward node %u: this bridge is stopping\n",port->index,port->name,port->peer);
    else if(port->peer)say("link %u (%s): node %u lost: code %lld\n",port->index,port->name,port->peer,(long long)code);
    else say("link %u (%s): lost: code %lld\n",port->index,port->name,(long long)code);
  }
  port->greeted=port->connecting=port->writing=0;port->peer=0;port->in_length=0;port->out_length=port->out_sent=0;
  link_set(port,0,port->open?MESH_PAIRING:MESH_STOPPED,code);
}
/* the control socket: non-blocking, its sends queued (port_say), its HELLO stating the queue pairs the port's device
   has for a link (its budget: link_plan); its peer's loss is the kernel's to tell (its end closed, a reset, the link's
   going off) */
static void port_greet(struct port *port){
  int on=1,flags=fcntl(port->control,F_GETFL);
  if(flags<0 || fcntl(port->control,F_SETFL,flags|O_NONBLOCK) || setsockopt(port->control,SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on) ||
     setsockopt(port->control,IPPROTO_TCP,TCP_NODELAY,&on,sizeof on)){
    port_lost(port,errno?errno:EIO);return;
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
/* design/transport.md: a port's device opened, on its own queue, where it is to be (at the start,
   at its link's coming on or its interface's attaching, at a hello heard while it is not open, while it has no peer,
   and as the call its port was concluded around returns, ports_stalled), once its quiet window has ended and no job of
   its own is out (design/transport.md B9, B12): the hellos' group joined on its interface, so the port hears them open
   or not (hello_heard, B16); its port read; a context that fails stays as it is, and the port with it, until its child
   ends (design/transport.md section 14: nothing made is forgotten or made again in a process).  The device is its
   child's: spawned here where the port has none (child_spawn: not while its last child's end is awaited, `recycle`,
   nor before the pause a spawn that failed set, `spawn_at`), and asked to open it (CHILD_OPEN), its answer the
   job's end (child_heard); a CHILD_OPEN the child did not take is the open's end at once (port_opened) */
static int child_spawn(struct port *port);
static void port_opened(struct port *port);
static void port_open(struct port *port){
  if(port->job || port->stuck || !port->opening || port->control>=0 || now<port->quiet_at || port->recycle || now<port->spawn_at ||
     now<port->retry_at)return;
  if(!port->child && child_spawn(port))return;
  port->opening=0;
  port->interface=if_nametoindex(port->name);
  struct ipv6_mreq group={.ipv6mr_multiaddr=everyone,.ipv6mr_interface=port->interface};
  if(port->interface)setsockopt(hello,IPPROTO_IPV6,IPV6_JOIN_GROUP,&group,sizeof group);
  port->job=1;port->job_at=now;atomic_store_explicit(&port->done,0,memory_order_relaxed);
  const int error=child_send(port,(struct child_message){.kind=CHILD_OPEN});
  if(error){port->result=error;port->refused=0;port_opened(port);}
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
/* design/transport.md: a Thunderbolt RDMA interface of the node, `device` (the provider's device of the same name:
   rdma_<interface>), made a port where none is yet, MESH_PORTS at most, as IOKit first matches it (interfaces_heard,
   on the event loop): opened once the node runs and its quiet window has ended, which starts as the exit's window
   stands (`exit_at`: a termination heard before it was made, B9b).  The parent makes no verbs call: its child lists the
   devices itself as it opens its own (device_up) */
static void port_found(const char *device){
  for(uint32_t p=0;p<port_count;p++)if(!strcmp(ports[p].device.name,device))return;
  if(port_count==MESH_PORTS)return;
  struct port *port=ports+port_count;
  *port=(struct port){.index=port_count,.control=-1,.child_fd=-1,.opening=1};
  port->link.provider.device=&port->device;
  snprintf(port->device.name,sizeof port->device.name,"%s",device);snprintf(port->name,sizeof port->name,"%s",device+5);
  /* the interface as the kernel's events name it, its name read once here into typed values (addresses_heard) */
  if(!strncmp(port->name,"en",2)){port->family=APPLE_IF_FAM_ETHERNET;port->unit=(uint32_t)strtoul(port->name+2,NULL,10);}
  port_count++;
  quiet_until(&port->quiet_at,exit_at);
  link_set(port,0,MESH_STOPPED,0);
}
/* design/transport.md: a port's link off or its interface detached (the link lost, its lanes left as they
   are, design/transport.md B6 (b), and the port down until its link comes on or its interface attaches: its device
   opened then, port_open).  At the stop (NODE_STOPPING, NODE_ENDING), a port's link going off only extends its quiet
   window, which the exit waits out (B9), and its child's (CHILD_HALT naming the loss: its exit then MESH_PLATFORM_NS
   on, as a loss's) */
static void network_heard(void){
  union mesh_network_event event;
  while(recv(network,event.bytes,sizeof event,0)>=(ssize_t)(KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data))){
    const struct net_event_data *interface=(const void *)(event.bytes+KEV_MSG_HEADER_SIZE);
    const uint32_t code=event.header.event_code;
    char name[IFNAMSIZ+16];
    snprintf(name,sizeof name,"%.*s%u",(int)sizeof interface->if_name,interface->if_name,interface->if_unit);
    for(uint32_t p=0;p<port_count;p++)if(!strcmp(name,ports[p].name)){
      if(code==KEV_DL_LINK_ON || code==KEV_DL_IF_ATTACHED){ports[p].opening=1;ports[p].retry_at=0;}
      else if(code!=KEV_DL_LINK_OFF && code!=KEV_DL_IF_DETACHED){}
      else if(stopping()){
        quiet_until(&ports[p].quiet_at,now+MESH_PLATFORM_NS);
        child_send(ports+p,(struct child_message){.kind=CHILD_HALT,.result=ENETDOWN});
      }
      else{ports[p].open=0;port_lost(ports+p,ENETDOWN);}
    }
  }
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
/* design/transport.md B9b (EXIT-RECLAIM.md P2, and P3 as a window in place of IOServiceWaitQuiet): a Thunderbolt RDMA
   peer interface's termination, which IOKit tells in its first phase (kIOTerminatedNotification): the exit's window
   and every port's moved to `until` at least (a port made meanwhile takes `exit_at`'s: port_found) */
static void quiet_all(uint64_t until){
  quiet_until(&exit_at,until);
  for(uint32_t p=0;p<port_count;p++)quiet_until(&ports[p].quiet_at,until);
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
      if(its && CFEqual(its,rid) && IORegistryEntryGetName(interface,name)==KERN_SUCCESS)
        for(uint32_t p=0;p<port_count;p++)if(!strcmp(ports[p].device.name,name))found=ports+p;
      if(its)CFRelease(its);
    }
    IOObjectRelease(interfaces);
  }
  if(rid)CFRelease(rid);
  return found;
}
/* a child's: its peer interfaces' terminations (the iterator drained first, which arms it again), on its main thread:
   for each terminated service the exit's window and its port's moved to MESH_PLATFORM_NS from now (no make, connect or
   exit inside the termination, and the child's end waits for it: child_main) before any property is read; then its
   port, where the service maps to it, halted (child_halt, ETIMEDOUT: its poller posts nothing from its next
   pass, the child takes no command again, and its main thread tells its parent); each released.  Heard as the child's
   loop wakes and between the steps of its commands (link_lapsed), so a termination during a verbs call is heard as
   the call returns.  A child whose listener is not yet armed as the termination comes (it has made no queue pair
   yet) misses it */
static void peer_terminated(void *unused,io_iterator_t iterator){
  (void)unused;
  io_service_t services[MESH_PORTS];
  uint32_t count=0;
  for(io_service_t service;(service=IOIteratorNext(iterator));){
    quiet_all(clock_ns()+MESH_PLATFORM_NS);
    if(count<MESH_PORTS)services[count++]=service;
    else IOObjectRelease(service);
  }
  for(uint32_t k=0;k<count;k++){
    struct port *port=peer_port(services[k]);
    IOObjectRelease(services[k]);
    if(port)child_halt(port,ETIMEDOUT);
  }
}
/* a child's listener, armed before it opens its device: its notification port a filter of its queue, its iterator
   drained once (which arms it).  A child whose notification is not armed ends: its port's poller would hear a
   termination only from its parent's line (terminations_heard), and it has made no queue pair yet, so its parent loses
   the link as it reaps it (child_reaped) and the port's next child arms its own */
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
    if(got==(void *)page){child_halt(ports,ENETDOWN);child_trigger(FAULT_ABSORBED);return;}
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
/* design/transport.md B9b: Thunderbolt RDMA peer interfaces terminating (the iterator drained, which arms it again):
   the exit's window and every port's moved to MESH_PLATFORM_NS from now (no port job and no exit inside the
   termination, whichever port's it is) before any property is read, and logged; then each port one maps to lost
   (port_lost, ETIMEDOUT: its line leaving, so its child's poller posts nothing more from its next pass, its child
   halted where it ends and its socket closed, no verbs call), each released: one handoff from IOKit, whether or not
   the port has a child.  The port's child hears the same termination and halts itself, and its report of it
   (CHILD_BROKEN, ETIMEDOUT: child_heard) finds the link lost already */
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
    port_lost(port,ETIMEDOUT);
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
   left and then came is lost, then opened): each port one maps to that has no peer to be opened as at its link's coming
   on (network_heard), so its device is opened at the hardware's word that its far end is present and not only at the
   en interface's link, which can come on before it (bin/mesh-fabric-init.sh: link active, no RDMA peer); without it
   a port whose open met its device's peer not yet present (ENETDOWN at both ends, no hello sent) opened only at its
   link's next coming on or the next hello heard on it, which might never come.  An arrival at an interface that is no
   port yet needs nothing more: the port its interface's first match makes is to be opened */
static void arrivals_heard(void *unused,io_iterator_t iterator){
  (void)unused;
  for(io_service_t service;(service=IOIteratorNext(iterator));IOObjectRelease(service)){
    struct port *port=peer_port(service);
    if(port && port->control<0){port->opening=1;port->retry_at=0;}
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
   its exit is reaped (child_reaped).  A command so lost is its job's failure (port_open, child_job) */
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
static void child_reaped(struct port *port);
/* a process, not this bridge nor one of its children, that is a child of the port's device of any version (its argv
   `--child VERSION DEVICE ...`, the three words every version keeps): a child of a launch whose parent ended
   unbidden, which has the device's queue pairs until its exit (child_main, at most 2 MESH_PLATFORM_NS); its pid and
   start (pid 0: none; one exited or in its exit is none).  A device's queue pairs are one child's at a time: a link
   takes as many as its device has for it, up to MESH_LINK_LANES (eight of a macOS 26 device's ten,
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
   the spawn until its exit, which the event loop hears (mesh-observe.h mesh_exits_watch on the loop's queue, tagged by
   the port: `elsewhere` cleared) and spawns at once; the processes are read again only then, not at every turn while
   it lives.  One exited or in its exit by the time its exit is watched is none, and the spawn goes on */
static int child_spawn(struct port *port){
  int pair[2],on=1;
  if(port->elsewhere)return EBUSY;
  struct mesh_process elsewhere=child_elsewhere(port);
  if(mesh_exits_watch(events,&elsewhere,1,TAG_CHILD|port->index)){
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
  port->child=pid;
  port->child_fd=pair[0];port->child_in_length=0;
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
/* a link job asked of the port's child: its lanes made (the plan's) or connected to the peer's (the peer's lanes as its
   LINK told them, its address as its hello did, its node and the link's bandwidth); its end the child's answer
   (child_heard), taken by link_turn, or, where the child did not take it (child_send), its failure, taken by
   link_turn at once, which loses the link there */
static void child_job(struct port *port,int job){
  struct link *l=&port->link;
  l->job=job;l->job_at=now;
  atomic_store_explicit(&l->done,0,memory_order_relaxed);
  struct child_message said={.kind=job==LINK_JOB_MAKE?CHILD_MAKE:CHILD_CONNECT,.lanes=l->plan};
  if(job==LINK_JOB_CONNECT){
    memcpy(said.peer,l->peer,sizeof said.peer);said.lid=port->heard.lid;memcpy(said.gid,port->heard.gid,16);
    said.node=port->peer;said.bandwidth=port->bandwidth;
  }
  const int error=child_send(port,said);
  if(error){l->result=error;atomic_store_explicit(&l->done,1,memory_order_release);link_turn(port);}
}
/* the child's messages, each whole one taken: an open's, a make's or a connect's end for the command out (one for no
   command out, its pairing lost, is dropped: link_lost); and the child's own halt (CHILD_BROKEN, its reason in
   `result`: a lane's or a completion's failure, EIO; a fault absorbed on its provider's doorbell, ENETDOWN, as at its
   link's going off; its peer interface's termination as it heard it, ETIMEDOUT), no verbs call made.  A child that
   tells it has halted itself, takes no command again and ends as at a loss, whether or not its lanes were made, so
   the link is lost (port_lost, `recycle`) and the port's next child opens the device.  The parent's own listener has
   lost the link on a termination already (terminations_heard); where it could not map it to the port, this is the
   loss.  The socket's end closes it; the child's exit is reaped (child_reaped) */
static void child_heard(struct port *port){
  struct link *l=&port->link;
  for(;;){
    const ssize_t got=recv(port->child_fd,port->child_in+port->child_in_length,sizeof port->child_in-port->child_in_length,MSG_DONTWAIT);
    if(got<0 && (errno==EAGAIN || errno==EINTR))return;
    if(got<=0){close(port->child_fd);port->child_fd=-1;return;}
    if((port->child_in_length+=(uint32_t)got)<sizeof port->child_in)continue;
    port->child_in_length=0;
    struct child_message heard;
    memcpy(&heard,port->child_in,sizeof heard);
    if(heard.kind==CHILD_OPENED && port->job){
      port->result=heard.result;port->refused=heard.refused;
      if(!heard.result){port->lid=heard.lid;memcpy(port->gid,heard.gid,16);port->bandwidth=heard.bandwidth;port->budget=heard.lanes;}
      atomic_store_explicit(&port->done,1,memory_order_release);
    }
    else if(heard.kind==CHILD_MADE || heard.kind==CHILD_CONNECTED){
      if(l->job!=(heard.kind==CHILD_MADE?LINK_JOB_MAKE:LINK_JOB_CONNECT))continue;
      l->result=heard.result;
      if(heard.kind==CHILD_MADE && !heard.result){
        memcpy(l->qpn,heard.qpn,sizeof l->qpn);l->psn=heard.psn;l->share=(struct frames){heard.share};
        port->lid=heard.lid;memcpy(port->gid,heard.gid,16);
      }
      atomic_store_explicit(&l->done,1,memory_order_release);
    }
    else if(heard.kind==CHILD_BROKEN){
      say("link %u (%s): its child halted itself (%s): its link lost (design/transport.md)\n",port->index,port->name,
          heard.result==ENETDOWN?"a fault of its provider's doorbell absorbed":heard.result==ETIMEDOUT?"its peer interface terminating":
          "a lane or a completion failed");
      port->open=0;
      if(!stopping())port->recycle=1;
      port_lost(port,heard.result>0?heard.result:EIO);
    }
  }
}
/* the port's child ended: its last messages taken, its exit reaped, a job of it out ended (ECHILD); one that ended
   unbidden loses the link (port_lost, ECHILD) and is spawned again once the port's quiet window has passed, at the
   events that open any port, however often its children ended (the operator, 2026-10-23 01:03Z: "this is not a real feature.
   delete it and anything like it").  Its lanes no draw's, what it had given back and its line made new
   (switch_child_free: its queue pairs went with it, so nothing lands in or is read from its pages again), its link
   down with no lanes, and the port to be opened by its next child */
static void child_reaped(struct port *port){
  struct link *l=&port->link;
  const pid_t pid=port->child;
  int status=0;
  if(port->child_fd>=0)child_heard(port);
  if(port->child_fd>=0){close(port->child_fd);port->child_fd=-1;}
  while(waitpid(pid,&status,0)<0 && errno==EINTR){}
  port->child=0;
  if(port->child_entry<2*MESH_PORTS && mesh_bridge_pages(M)->child[port->child_entry].pid==(uint32_t)pid)
    mesh_bridge_pages(M)->child[port->child_entry]=(struct mesh_process){0};
  const int asked=port->recycle || stopping();
  port->recycle=0;
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
  port->opening=1;
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
   own (child_halt: a lane's or a completion's failure on its poller, a termination heard, a doorbell fault absorbed, a
   make or connect that failed, each told its parent once: CHILD_BROKEN), or its parent lost (the socket's end, or a
   message to it the socket did not take whole: child_tell).  A command that does not fit the state is refused:
   ECANCELED once the line leaves, EBUSY otherwise.  The child owns its exit (child_end): once its own windows have
   ended, and, where its end is a loss (a halt naming one, its own halt, its parent lost), MESH_PLATFORM_NS after it;
   and in either case once its poller has returned from its last pass: its exit, the kernel's release of its
   device's objects, no verbs call made */
static unsigned char child_in[sizeof(struct child_message)];
static uint32_t child_got;
/* the child's end, its main thread's: none while `bound` is 0; else its
   exit due at `due` once its windows have ended and its poller has returned, and at `bound` whatever its poller does.
   A stop's (CHILD_HALT with no loss) is due as it comes, within MESH_PLATFORM_NS; a loss's (CHILD_HALT naming the
   loss, its own halt, its parent lost) MESH_PLATFORM_NS on, within twice that, the floor a child whose link or parent
   was lost keeps before its release (B9, B9b), as its parent's port window was.  A later end moves both later, never
   earlier, so a loss heard after a stop takes the loss's time.  `lost`: its parent lost, its socket read and written
   no more */
static struct { int lost; uint64_t due,bound; } ending;
static void child_end(struct port *port,int loss){
  child_halt(port,0);
  const uint64_t at=clock_ns();
  ending.due=MAX(ending.due,loss?at+MESH_PLATFORM_NS:at);
  ending.bound=MAX(ending.bound,at+(loss?2:1)*MESH_PLATFORM_NS);
}
static void parent_lost(struct port *port){
  if(ending.lost)return;
  ending.lost=1;
  /* the parent's socket read no more: its end would end each wait at once */
  struct kevent64_s parent;EV_SET64(&parent,3,EVFILT_READ,EV_DELETE,0,0,0,0,0);
  kevent64(events,&parent,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
  child_end(port,1);
}
/* a message to the parent, whole, in one send that never waits (child_send's rule, the other way: a child tells at
   most an answer to each command and one report); one not taken whole is its parent's loss */
static void child_tell(struct port *port,struct child_message said){
  if(!ending.lost && send(3,&said,sizeof said,MSG_DONTWAIT)!=(ssize_t)sizeof said)parent_lost(port);
}
/* the child's own halt told its parent, once (child_halt's reason, taken: CHILD_BROKEN), and its end a loss's */
static void child_report(struct port *port){
  const uint32_t reason=atomic_exchange(&child_reason,0);
  if(!reason)return;
  child_tell(port,(struct child_message){.kind=CHILD_BROKEN,.result=(int32_t)reason});
  child_end(port,1);
}
/* why a command does not fit the line's state, 0 where it does (its state one of `fits`, bits over enum child_state) */
static int32_t child_refused(const struct port *port,uint32_t fits){
  const uint32_t state=mesh_port_state(atomic_load(&switch_of()->port[port->index]));
  return fits>>state&1?0:state==CHILD_LEAVING?ECANCELED:EBUSY;
}
/* the device opened (CHILD_OPEN; again where it is open: its port and address read again): device_up, then its GID
   (device_gid), the line moved to DEVICE_OPEN; the answer its address, bandwidth and the queue pairs it has for a link */
static struct child_message device_open(struct port *port){
  struct child_message said={.kind=CHILD_OPENED,.result=child_refused(port,1u<<CHILD_STARTED|1u<<CHILD_DEVICE_OPEN)};
  int refused=0;
  if(!said.result)said.result=device_up(&port->device,&wire,&refused);
  if(!said.result)said.result=link_lapsed(port);
  if(!said.result)said.result=device_gid(&port->device,port->name);
  if(!said.result && !port->device.gid_matched){
    const uint8_t *g=port->device.gid.raw;
    say("link %u (%s): no GID of %s is a link-local address %s has; GID[%u] %02x%02x:%02x%02x:%02x%02x:%02x%02x:"
        "%02x%02x:%02x%02x:%02x%02x:%02x%02x used\n",port->index,port->name,port->device.name,port->name,port->device.gid_index,
        g[0],g[1],g[2],g[3],g[4],g[5],g[6],g[7],g[8],g[9],g[10],g[11],g[12],g[13],g[14],g[15]);
  }
  if(!said.result && !mesh_port_move(switch_of(),port->index,1u<<CHILD_STARTED|1u<<CHILD_DEVICE_OPEN,CHILD_DEVICE_OPEN,0,0))
    said.result=ECANCELED;
  said.refused=refused;
  if(!said.result){
    port->lid=port->device.port.lid;memcpy(port->gid,&port->device.gid,16);
    said.lid=port->lid;memcpy(said.gid,port->gid,16);said.bandwidth=port_bandwidth(&port->device.port);
    said.lanes=port->device.max_pairs;
  }
  return said;
}
/* the lanes of the plan the command names made (CHILD_MAKE, from DEVICE_OPEN: link_make), the line moved to
   LANES_MADE; the answer each lane's queue pair number, the first sequence number, each lane's share and the port's
   address.  A make that fails halts the child (child_halt): a queue pair it made is released at its exit, never made
   again or reused */
static struct child_message lanes_make(struct port *port,const struct child_message *heard){
  struct link *l=&port->link;
  struct child_message said={.kind=CHILD_MADE,.result=child_refused(port,1u<<CHILD_DEVICE_OPEN)};
  if(said.result)return said;
  l->plan=heard->lanes;
  said.result=link_make(port);
  if(!said.result && !mesh_port_move(switch_of(),port->index,1u<<CHILD_DEVICE_OPEN,CHILD_LANES_MADE,0,0))said.result=ECANCELED;
  if(said.result){child_halt(port,0);return said;}
  for(uint32_t k=0;k<MESH_LINK_LANES;k++)said.qpn[k]=l->plan>>k&1?l->provider.queues[k].pair->qp_num:UINT32_MAX;
  said.psn=l->provider.psn;said.share=(uint32_t)l->share.n;
  port->lid=port->device.port.lid;memcpy(port->gid,&port->device.gid,16);
  said.lid=port->lid;memcpy(said.gid,port->gid,16);
  return said;
}
/* the lanes connected to the peer's the command names (CHILD_CONNECT, from LANES_MADE: link_connect) and served
   (link_serve: the line serving toward the command's node, the poller made); a connect that fails halts the child */
static struct child_message lanes_connect(struct port *port,const struct child_message *heard){
  struct child_message said={.kind=CHILD_CONNECTED,.result=child_refused(port,1u<<CHILD_LANES_MADE)};
  if(said.result)return said;
  port->peer=heard->node;
  said.result=link_connect(port,heard);
  if(!said.result)said.result=link_serve(port,heard->node,heard->bandwidth);
  if(said.result)child_halt(port,0);
  return said;
}
static void child_command(struct port *port,const struct child_message *heard){
  if(heard->kind==CHILD_OPEN)child_tell(port,device_open(port));
  else if(heard->kind==CHILD_MAKE)child_tell(port,lanes_make(port,heard));
  else if(heard->kind==CHILD_CONNECT)child_tell(port,lanes_connect(port,heard));
  else if(heard->kind==CHILD_HALT)child_end(port,heard->result!=0);
}
static void child_read(struct port *port){
  while(!ending.lost){
    const ssize_t got=recv(3,child_in+child_got,sizeof child_in-child_got,MSG_DONTWAIT);
    if(got<0 && (errno==EAGAIN || errno==EINTR))return;
    if(got<=0){parent_lost(port);return;}
    if((child_got+=(uint32_t)got)<sizeof child_in)continue;
    child_got=0;
    struct child_message heard;
    memcpy(&heard,child_in,sizeof heard);
    child_command(port,&heard);
  }
}
/* the child's loop: its main thread waits on its queue (the parent's socket, its IOKit notifications, its poller's
   return and a fault absorbed), with no timeout until it ends; ending, until its exit is due: the windows read again at
   each wake (a termination heard moves them), and the exit made once its end is due, its windows have ended and its
   poller has returned from its last pass, or at the end's bound whatever its poller does (a poller inside a provider
   call may never return: what it holds then is lost, as at a crash, O4).  It goes on reading its parent's socket while
   it ends, so a loss told after a stop moves its end later (child_end) */
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
  *port=(struct port){.index=(uint32_t)index,.control=-1,.child_fd=-1};
  port->link.provider.device=&port->device;
  snprintf(port->device.name,sizeof port->device.name,"%s",argv[3]);snprintf(port->name,sizeof port->name,"%s",argv[3]+5);
  port_count=1;
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
    if(ending.bound){
      const uint64_t at=clock_ns(),until=MIN(ending.bound,MAX(ending.due,windows_end()));
      if(at>=ending.bound || (at>=until && !polling))exit(0);
      const uint64_t left=(polling?ending.bound:until)-at;
      rest=(struct timespec){(time_t)(left/1000000000u),(long)(left%1000000000u)};wait=&rest;
    }
    struct kevent64_s ready[4];
    const int count=kevent64(events,NULL,0,ready,4,0,wait);
    now=clock_ns();
    for(int i=0;i<count;i++){
      if(ready[i].filter==EVFILT_READ && ready[i].ident==3)child_read(port);
      else if(ready[i].filter==EVFILT_USER && ready[i].ident==POLLER_RETURNED)polling=0;
      else if(ready[i].filter==EVFILT_MACHPORT)iokit_heard();
    }
    child_report(port);
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
   that wait, by NODE_ADOPTING and by child_elsewhere, which puts off a new child's spawn on a device while an old
   child of that device is found; neither finds a child whose exit has begun and not ended (node_end).  What is given
   back is
   what the bridge before had, each use by the ring or receive that had it, never a recount of each page's
   uses from the table: the clients attached across the restart change words, ring cells and uses as it runs (a
   write's use counted as it pushes its cell, an invalidation's dropped as it clears a word), so a count made from one
   reading of the table and the rings would miss or double each change made between its reads.  A map or health
   serial a bridge that died mid-write left odd made even (its readers wait while it is odd), and the map's records
   this bridge's to write again (map_adopt).  The pool's carve and the page table stand; the switch is made new after
   (switch_make), so what was given back is said here */
static void region_adopt(void){
  for(uint32_t s=0;s<MESH_STREAMS;s++){
    struct mesh_lanes *x=mesh_lanes_at(M,s);
    atomic_store(&x->peer,0);atomic_store(&x->wire,0);
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
    for(uint32_t l=0;l<MESH_SWITCH_LANES;l++)mesh_switch_flight_drop(M,sw,l,MESH_SWITCH_PARENT);
    for(uint32_t p=0;p<MESH_PORTS;p++)mesh_switch_inbound_drop(M,sw,p,MESH_SWITCH_PARENT);
    lost=atomic_load(&sw->counters[MESH_SWITCH_PARENT].lost)-before;
  }
  const uint32_t receives=mesh_switch_receives_drop(M,0,MESH_SWITCH_LANES);
  say("bridge: what the bridge before had given back: %llu pages of its rings (lost) and %u of its "
      "receives\n",(unsigned long long)lost,receives);
  if(atomic_load(&M->map)&1)atomic_fetch_add(&M->map,1);
  if(atomic_load(&M->health)&1)atomic_fetch_add(&M->health,1);
  map_adopt();
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
   `liveness`, and the next health sample; while it runs, the exit's window and each port's quiet window, spawn pause,
   open's retry, next hello, and the bound of its open's or link's job out), and no
   later than MESH_PAUSE_NS on (design/rubrics.md M3: the wait ends in finite time).  Every other change a turn takes
   wakes the loop where it happens (wake, a socket, a signal and an exit on the queue), so a turn comes at the change
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
  due_at(&due,exit_at);
  for(uint32_t p=0;p<port_count;p++){
    const struct port *port=ports+p;
    due_at(&due,port->quiet_at);due_at(&due,port->spawn_at);
    if(port->opening)due_at(&due,port->retry_at);
    if(port->open && port->control<0)due_at(&due,port->next);
    if(port->job)due_at(&due,port->job_at+MESH_PLATFORM_NS);
    if(port->link.job)due_at(&due,port->link.job_at+MESH_PLATFORM_NS);
  }
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
   most MESH_PLATFORM_NS on */
static void node_start(int adopted){
  if(adopted)region_adopt();
  /* a boot is the bridge's start on the wall clock, so a node's later boot is a newer record */
  M->node=node.id;M->boot=clock_gettime_nsec_np(CLOCK_REALTIME);M->idle_since=M->boot;
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
  struct kevent64_s event;
  int watched[4]={listener,hello,network,addresses};
  for(int i=0;i<4 && watched[i]>=0;i++){EV_SET64(&event,(uint64_t)watched[i],EVFILT_READ,EV_ADD,0,0,UINT64_MAX,0,0);if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");}
  mesh_bridge_pages(M)->bridge_start=mesh_started(getpid());
  atomic_store(&M->bridge_pid,(uint64_t)getpid());
  __sync_synchronize();M->magic=MESH_MAGIC;
  /* the policy's refresh, where a theta is loaded, at the progress threads' class: its SWITCH_REFRESH_NS pause near
     its length (at the default class the system's timer slack made 50 us about 460 us) */
  if(switch_policy_on){
    pthread_t refresher;
    pthread_attr_t refreshing;pthread_attr_init(&refreshing);pthread_attr_set_qos_class_np(&refreshing,QOS_CLASS_USER_INTERACTIVE,0);
    if(pthread_create(&refresher,&refreshing,switch_watch,NULL))die("the switch's policy refresh");
    pthread_attr_destroy(&refreshing);
  }
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
/* design/transport.md: the stop (NODE_STOPPING), at stop signal `code`: every child halted (CHILD_HALT with no loss:
   its line leaving, so no pass of its poller begins after it and no post is made once its peer can read the stop, and
   its exit once its own windows have ended and its poller has returned, child_end), every control socket closed (the
   kernel sends what it has after the close; the peer reads the stop as the socket's end), every lane no draw's
   (lanes_cancel; what the lanes have the next bridge's adoption gives back: region_adopt), `bridge_pid` 0 (every stop
   keeps the region for the next bridge to adopt; a release is the heartbeat's, which expires the region once this
   bridge has ended: mesh-heartbeat.c region_expire), and no peer heard or dialed again (the listener's, the hellos'
   and the addresses' filters deleted).  No queue, thread or verbs call is waited on, and every object is the
   kernel's to release at the children's exits (design/RDMA-KERNEL-RECOVERY.md "October 18"), each child's no sooner
   than its own windows end (a termination it hears, B9b; a loss its parent tells it of, port_lost, network_heard:
   B9).  The bridge exits no sooner than every port's quiet window ends and a peer interface's termination's
   (`exit_at`, B9b), the loop serving the children's sockets, the clients' exits and the kernel's link events meanwhile
   (a link going off extends its port's window and its child's, network_heard, as a termination extends them all,
   terminations_heard), each no later than MESH_PLATFORM_NS past the stop's start (`due`: every window set before the
   stop ends by then, so the wait is the one 5 s budget, well inside its heartbeat's bound of 3 MESH_PLATFORM_NS) */
static void node_stop(int64_t code){
  char why[96];stop_text(code,why,sizeof why);
  say("bridge node %u: stopping (%s)\n",node.id,why);
  for(uint32_t p=0;p<port_count;p++){
    child_send(ports+p,(struct child_message){.kind=CHILD_HALT});
    if(ports[p].control>=0){close(ports[p].control);ports[p].control=-1;}
  }
  lanes_cancel(NULL);
  atomic_store(&M->bridge_pid,0);
  const int unheard[]={listener,hello,addresses};
  for(size_t i=0;i<sizeof unheard/sizeof *unheard;i++)if(unheard[i]>=0){
    struct kevent64_s event;EV_SET64(&event,(uint64_t)unheard[i],EVFILT_READ,EV_DELETE,0,0,0,0,0);
    kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
  }
  node.state=NODE_STOPPING;node.due=clock_ns()+MESH_PLATFORM_NS;
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
   bridge ended once every child, each halted at the stop and ending by its own windows (child_end), is reaped
   (child_reaped, on the loop: its NOTE_EXIT, then its exit seen through to its end), at most MESH_PLATFORM_NS on, so
   no child of this launch has its device's queue pairs as the next launch's children make theirs.  This end stays
   beside the next bridge's ADOPTING and child_elsewhere because it alone sees an exit through to its end: a process in
   its exit is no longer found by its pid, before the kernel releases its device's objects (design/transport.md section
   2), so neither of those sees a child of this launch whose exit has begun and not ended */
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
    if(node.state==NODE_RUNNING){
      ports_stalled();
      for(uint32_t p=0;p<port_count;p++){
        struct port *port=ports+p;
        if(port->job && atomic_load_explicit(&port->done,memory_order_acquire))port_opened(port);
        port_open(port);
        port_hello(port);
        link_turn(port);
      }
    }
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
      if(e->filter==EVFILT_SIGNAL)node_step(NODE_STOP,(int64_t)e->ident);
      else if(e->filter==EVFILT_PROC && e->udata==TAG_ADOPT)node_step(NODE_EXIT,0);
      /* design/transport.md: the parent's IOKit, its callbacks run here */
      else if(e->filter==EVFILT_MACHPORT && e->udata==TAG_IOKIT)iokit_heard();
      /* design/transport.md section 14, "A process a port": a port's child's message or exit */
      else if(e->udata!=UINT64_MAX && (e->udata&TAG_CHILD)){
        struct port *port=(e->udata&0xff)<port_count?ports+(e->udata&0xff):NULL;
        if(port && e->filter==EVFILT_PROC && (pid_t)e->ident==port->child){child_reaped(port);node_step(NODE_EXIT,0);}
        else if(port && e->filter==EVFILT_PROC && (pid_t)e->ident==port->elsewhere)port->elsewhere=0;
        else if(port && e->filter==EVFILT_READ && (int)e->ident==port->child_fd)child_heard(port);
      }
      else if(e->filter==EVFILT_PROC){
        for(uint32_t s=0;s<MESH_CLIENTS;s++){
          uint64_t owner=atomic_load(&((struct mesh_client *)((char *)M+M->client_off))[s].owner);
          if(owner && (uint32_t)owner==(uint32_t)e->ident)mesh_leave(M,owner);
        }
      }
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)network)network_heard();
      /* peers heard and dialed while the node serves: from the stop on their filters are deleted (node_stop), and an
         event of theirs taken in the stop's batch is let be */
      else if(stopping()){}
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
}
