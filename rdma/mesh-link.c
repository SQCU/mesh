#include "mesh-bridge.h"

/* mesh-verbs.h: the region's name, unlinked at exit (down), and the stop flag every loop reads */
const char *shm;
_Atomic sig_atomic_t stop;

struct hdr *M;
struct wire_mapping wire;
struct port ports[MESH_PORTS];
uint32_t port_count;
struct flow *flows;
/* design/bridge-and-ipc.md#region: the flows and awaiting programs the bridge holds, told in the region (`busy`), and
   the wall clock at which it last came to hold none (`idle_since`, 0 while it holds any) */
uint32_t flow_count,await_count;
int events=-1,listener=-1,hello=-1,network=-1,unreleased,finding;
uint64_t now,sequence;
char host[MESH_NAME_BYTES];
struct mesh_ctx bridge={.fd=-1};
static dispatch_queue_t finder;
_Atomic int found;
static char devices[MESH_PORTS][32];
static uint32_t device_count;
static const struct in6_addr everyone={{{0xff,0x02,0,0,0,0,0,0,0,0,0,0,0,0,0,1}}};

static uint64_t clock_ns(void){ return clock_gettime_nsec_np(CLOCK_MONOTONIC); }
uint64_t memory;
static int64_t stop_reason;
static void farewell_text(int64_t code,char *out,size_t room){
  if(code==SIGTERM)snprintf(out,room,"SIGTERM: its agent was unloaded or restarted");
  else if(code>0)snprintf(out,room,"signal %lld",(long long)code);
  else if(code==MESH_STOP_BINARY)snprintf(out,room,"its binary changed");
  else snprintf(out,room,"no reason given");
}
struct known known[MESH_KNOWN];
void wake(void){
  struct kevent64_s event;EV_SET64(&event,0,EVFILT_USER,0,NOTE_TRIGGER,0,0,0,0);
  kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
}

/* design/bridge-and-ipc.md#ledger: whether the configuration keeps ledgers (its `ledger`), and the bridge's ledgers
   numbered as they are made */
int ledger_kept;
/* design/bridge-and-ipc.md#keepalive: the configuration's `keepalive`, the most nanoseconds a link's send thread lets
   pass between frames on its keepalive lane while any of its lanes is leased; 0 where it states none (none is sent) */
uint64_t keepalive_ns;
#ifdef MESH_LOOP_FABRIC
/* design/bridge-dataflow.md#the-loopback-fabric (mesh 800fe6bc^ mesh-flow-loop, on the resident bridge): a loop bridge's
   configuration's `loop`: its region's name, its node, the loopback port its clients' requests come to (`service`, in
   place of MESH_PORT), the loopverbs fabric it shares with the host's other loop bridges (loopverbs.c loop_fabric), and
   its ports, each a loopverbs device and a cable: the loopback port this end listens on for it and the peer end's, the
   lower of the two dialing the higher (in place of a link's hello) */
void loop_fabric(const char *name);
struct loop_port { char device[32]; uint16_t listen,peer; int fd; };
static struct { char region[32],fabric[64]; uint32_t node,ports; uint16_t service; struct loop_port port[MESH_PORTS]; } loop_config;
#endif

/* design/bridge-and-ipc.md#lanes: a port's link pages: lane MESH_LANE_KEEPALIVE's block (its frames' source and sink),
   then the stripes' MESH_DESCRIPTORS descriptors received and as many sent; leased block-aligned for every port the
   region has, at the bridge's start beside the map's (link_reserve), else as its link first pairs (link_turn) */
static uint32_t link_pages(void){
  const uint64_t bytes=(uint64_t)M->block*M->pgsz+2*MESH_DESCRIPTORS*sizeof(struct mesh_descriptor);
  return (uint32_t)((bytes+M->pgsz-1)/M->pgsz);
}
static uint64_t link_block(const struct port *port){ return (uint64_t)port->link.page*M->pgsz; }
/* the window offset of descriptor slot `slot` of the port's stripes (received 0 to MESH_DESCRIPTORS - 1, sent the next
   MESH_DESCRIPTORS) */
static uint64_t stripe_descriptor(const struct port *port,uint32_t slot){
  return link_block(port)+(uint64_t)M->block*M->pgsz+(uint64_t)slot*sizeof(struct mesh_descriptor);
}
/* a lane's failure, found by a progress thread: served no more, and made again once the peer is next heard (the event
   loop takes `broken`: link_turn, link_remake) */
static void lane_lost(struct port *port,uint32_t lane,int64_t code){
  struct link *l=&port->link;
  if(atomic_fetch_and_explicit(&l->served,~(UINT32_C(1)<<lane),memory_order_acq_rel)>>lane&1)
    say("link %u (%s): lane %u failed: %s; made again once node %u is next heard\n",port->index,port->name,lane,
        strerror((int)(code>0?code:EIO)),port->peer);
  atomic_fetch_or_explicit(&l->broken,UINT32_C(1)<<lane,memory_order_release);
  wake();
}

/* design/algorithm-sources.md#meshresult: a data thread's failure, taken by the event loop (flow_fail) */
static void flow_lost(struct flow *flow,int64_t code,uint32_t domain){
  int64_t none=0;
  if(atomic_compare_exchange_strong(&flow->failed,&none,code?code:EIO))atomic_store(&flow->domain,domain);
  atomic_store_explicit(&flow->progressing,0,memory_order_release);
  wake();
}
/* design/bridge-and-ipc.md#progress: a landing that released a SEND cell (the index hand-off of a relay, or of a member
   that sends on what it receives) wakes the send thread that posts the cell, where that thread is parked: the fence
   orders the release before the read of its parked word, as the thread orders its parked word before its last look
   (port_progress), so one of the two sees the other */
static void send_wake(_Atomic uint64_t *ready){
  const uint32_t p=atomic_load_explicit(&((struct mesh_send *)(void *)ready)->port,memory_order_acquire);
  atomic_thread_fence(memory_order_seq_cst);
  if(p && p<=MESH_PORTS && atomic_load_explicit(&ports[p-1].parked[1],memory_order_relaxed))mesh_notify(&M->bell[p-1]);
}
/* design/bridge-and-ipc.md#ledger: the send thread's stores: a cell first seen released, its queue's event from then on */
static void ledger_seen(struct ledger *l,const struct mesh_send *cell,uint64_t cycle){
  const uint32_t q=cell->queue;
  if(!l->send_room){l->current[q]=LEDGER_NONE;return;}
  const uint32_t e=l->sends;
  l->send[e%l->send_room]=(struct ledger_send){.cell=(uintptr_t)cell,.seen=ledger_ns(),.cycle=(uint32_t)cycle,.next=LEDGER_NONE};
  if(l->last[q]!=LEDGER_NONE)l->send[l->last[q]%l->send_room].next=e;
  if(l->pending[q]==LEDGER_NONE)l->pending[q]=e;
  l->last[q]=e;l->current[q]=e;
  __atomic_store_n(&l->sends,e+1,__ATOMIC_RELEASE);
}
/* SENDs of event e posted at `at`: its first, and where `done` its last, after which its gate held `mark` frames */
static void ledger_stamp(struct ledger *l,uint32_t e,int done,uint64_t mark,uint64_t at){
  if(e==LEDGER_NONE)return;
  struct ledger_send *s=l->send+e%l->send_room;
  if(!s->first)s->first=at;
  if(done){s->posted=at;s->mark=mark&MESH_GATE_MASK;}
}
/* a flow's SEND completion on queue q, its gate's frames retired through it: every event of the queue whose last SEND it
   retires (a queue pair's completions come in the order posted) completed now */
static void ledger_retired(struct ledger *l,uint32_t q,uint64_t retired){
  uint64_t at=0;
  for(uint32_t e;q<l->queues && (e=l->pending[q])!=LEDGER_NONE;){
    struct ledger_send *s=l->send+e%l->send_room;
    if(!s->posted || ((retired-s->mark)&MESH_GATE_MASK)>=(UINT64_C(1)<<47))break;
    if(!at)at=ledger_ns();
    s->completed=at;l->pending[q]=s->next;
  }
}
/* the receive thread's store: a record of the flow landed, its word stored */
static void ledger_landed(struct flow *link,const struct prepared_receive *record){
  struct ledger *l=link->ledger;
  if(!l->landing_room)return;
  const uint32_t e=l->landings;
  l->landing[e%l->landing_room]=(struct ledger_landing){(uint32_t)(record-link->receive),record->cycle,ledger_ns()};
  __atomic_store_n(&l->landings,e+1,__ATOMIC_RELEASE);
}

/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M06 */
/* design/prepared-machine.md#M15 */
/* design/algorithm-sources.md#independent-native-queues */
/* The flow's SEND completions: each retires its queue's frames through its request (the device completes every SEND,
   signaled or not); *taken their count.  0, or an error completion's status. */
static __attribute__((always_inline)) inline int send_retire(struct flow *link,int *taken){
  struct ibv_wc done[16];
  int count=link->provider.queues[0].poll(link->provider.sent,16,done);
  *taken=count>0?count:0;
  if(count<0)return EIO;
  for(int i=0;i<count;i++){
    if(done[i].status)return (int)done[i].status;
    struct send_gate *gate=link->gates+(done[i].wr_id>>48);
    gate->retired=done[i].wr_id&MESH_GATE_MASK;gate->completions++;
    /* #ledger */
    if(link->ledger)ledger_retired(link->ledger,(uint32_t)(done[i].wr_id>>48),gate->retired);
  }
  return 0;
}

/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#independent-native-queues */
/* A running flow's SENDs, one pass of its port's send thread (design/bridge-and-ipc.md#progress): each stream's cells
   in order, a cell released by its producer posting its chain's requests while its queue pair holds their frames (the
   gate: frames posted less those its SEND completions retired; ibv_post_send accepts past the queue's frames and
   corrupts later, so the gate is the caller's), then its SEND completions taken.  Whether anything moved; a failure is
   the flow's (flow_lost). */
static int flow_send(struct flow *link){
  int moved=0,taken=0;
  struct ibv_send_wr *bad;
  struct ledger *ledger=link->ledger;
  int (*post)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **)=link->provider.queues[0].send;
  for(uint32_t s=0;s<link->stream_count;s++){
    struct send_stream *stream=link->streams+s;
    for(;;){
      if(!stream->remaining){
        struct mesh_send *cell=stream->cell;
        if(atomic_load_explicit(&cell->ready,memory_order_acquire)<=stream->cycle)break;
        stream->next=&cell->request;stream->remaining=cell->chunks;
        /* #ledger */
        if(ledger)ledger_seen(ledger,cell,stream->cycle);
      }
      struct mesh_send *cell=stream->cell;
      struct send_gate *gate=link->gates+cell->queue;
      const uint32_t unposted=stream->remaining;
      while(stream->remaining){
        struct ibv_send_wr *request=stream->next;
        uint32_t frames=(request->sg_list->length+4095)/4096;
        if(((gate->posted-gate->retired)&MESH_GATE_MASK)+frames>gate->capacity)break;
        gate->posted+=frames;
        /* posted as a copy: the cell's first request keeps the producer's argument in its wr_id (mesh_transfers_prepare),
           which a host producer reads at every publication (mesh_host_publish), and the copy's names the gate */
        struct ibv_send_wr posting=*request;
        posting.next=NULL;posting.send_flags=IBV_SEND_SIGNALED;
        posting.wr_id=((uint64_t)cell->queue<<48)|(gate->posted&MESH_GATE_MASK);
        int error=post(link->provider.queues[cell->queue].pair,&posting,&bad);
        if(error){flow_lost(link,error<0?-error:error,1);return 1;}
        stream->next=request->next;stream->remaining--;gate->requests++;moved=1;
      }
      /* #ledger */
      if(ledger && stream->remaining!=unposted)ledger_stamp(ledger,ledger->current[cell->queue],!stream->remaining,gate->posted,ledger_ns());
      if(stream->remaining)break;
      stream->cell=(struct mesh_send *)cell->successor;
      if(stream->cell==stream->first)stream->cycle++;
    }
  }
  if(link->stream_count){
    int error=send_retire(link,&taken);
    if(error){flow_lost(link,error,2);return 1;}
  }
  return moved || taken;
}

/* design/prepared-machine.md#M08 */
/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#programcopy */
/* A running flow's receive completions, one pass of its port's receive thread: each stores its completion word and
   posts the ring's next records.  Whether anything landed; a failure is the flow's (flow_lost). */
static int flow_receive(struct flow *link){
  if(!link->rings)return 0;
  struct ibv_wc done[16];
  const struct mesh_queue *queue=link->provider.queues;
  int count=queue->poll(queue->completion,16,done);
  if(count<0){flow_lost(link,EIO,3);return 1;}
  for(int i=0;i<count;i++){
    uint64_t status_opcode;
    memcpy(&status_opcode,&done[i].status,sizeof status_opcode);
    if((uint32_t)status_opcode){flow_lost(link,(uint32_t)status_opcode,2);return 1;}
    if(!((status_opcode>>32)&IBV_WC_RECV))continue;
    struct prepared_receive *record=(void *)(uintptr_t)done[i].wr_id;
    struct receive_ring *ring=link->rings+record->queue;
    ring->landed++;ring->bytes+=done[i].byte_len;
    atomic_store_explicit(record->input,record->argument+record->cycle,memory_order_release);
    if(record->release)send_wake(record->input);
    /* #ledger */
    if(link->ledger)ledger_landed(link,record);
    ring->outstanding-=record->frames;ring->completed=(int64_t)(record->invocation+(uint64_t)record->cycle*link->invocations);
    int error=ring_advance(link,ring);
    if(error){flow_lost(link,error,1);return 1;}
  }
  return count>0;
}

/* a stripe's next piece from window offset `offset`, `remaining` bytes left: at most the stripes' piece and never past a
   registered extent (MESH_REGION), so both ends cut a stripe alike and each piece is one SGE of one registration */
static uint32_t stripe_length(uint64_t offset,uint64_t remaining,uint32_t piece){
  const uint64_t room=MESH_REGION-offset%MESH_REGION,most=remaining<piece?remaining:piece;
  return (uint32_t)(most<room?most:room);
}
/* a word of this node's region a stripe stores (`word` a region offset in the window, 8-byte aligned; 0 none) */
static int stripe_word_valid(uint64_t word){ return !word || (!(word&7) && word>=M->data_off && word<=M->length-sizeof(uint64_t)); }
static void stripe_word(uint64_t word,uint64_t value){
  if(word && stripe_word_valid(word))atomic_store_explicit((_Atomic uint64_t *)((char *)M+word),value,memory_order_release);
}
/* a request failed as a value: its writer's word UINT64_MAX, its ring's failures counted and the code kept (a ring whose
   client retired meanwhile is left alone: its rows stay held while the request is in flight, clients_turn) */
void stripe_failed(uint64_t ring,uint64_t owner,uint64_t word,int64_t code){
  if(!owned(owner))return;
  stripe_word(word,UINT64_MAX);
  if(ring<M->data_off || ring>M->length-sizeof(struct mesh_requests))return;
  struct mesh_requests *q=(void *)((char *)M+ring);
  atomic_fetch_add_explicit(&q->failed,1,memory_order_relaxed);
  atomic_store_explicit(&q->code,code,memory_order_release);
}
/* the largest piece this end takes in one SEND: its port's max_msg_sz in whole frames, at most MESH_STRIPE_PIECE (mesh
   8fc27cbe's 4 MiB) and half its lane's share of the link's completion queues (so two pieces are in flight) */
static uint32_t stripes_piece(const struct port *port){
  const uint32_t most=port->device.port.max_msg_sz/4096*4096,share=(port->link.share>2?(port->link.share-1)/2:1)*4096u;
  const uint32_t piece=most && most<MESH_STRIPE_PIECE?most:MESH_STRIPE_PIECE;
  return piece<share?piece:share;
}
/* a stripe's descriptor or piece SEND completed, the send thread's (a stripe's last piece stores its writer's word) */
static void stripe_sent(struct stripes *s,uint64_t id){
  if((id&~STRIPE_SLOT)==STRIPE_DESCRIPTOR){s->descriptors_done++;s->send_frames--;return;}
  const struct stripe_piece *p=s->sent+(id&STRIPE_SLOT)%MESH_STRIPE_CHUNKS;
  s->send_frames-=p->frames;s->send_tail++;
  if(!p->last)return;
  if(owned(p->owner))stripe_word(p->word,p->value);
  atomic_fetch_add_explicit(&s->written,1,memory_order_relaxed);
  atomic_fetch_sub_explicit(&s->flight,1,memory_order_release);
}
/* design/bridge-dataflow.md#stripes: one pass of the port's send thread over its stripes (mesh 8fc27cbe peer_serve's SEND
   half): the next request of each registered ring naming the port's peer taken (compare-exchange of the ring's
   `taken`, so the threads of every port toward that node take each request once), its descriptor posted on the lane,
   then its pieces, while the lane's share of frames and the piece ring hold them; a request of a range the windows do
   not hold or a word that is no word fails as a value (EINVAL).  Whether anything moved; a failure is the lane's
   (lane_lost) */
static int stripes_send(struct port *port){
  struct stripes *s=&port->stripes;
  const struct mesh_queue *lane=port->link.provider.queues+MESH_LANE_STRIPES;
  int moved=0;
  _Atomic uint64_t *rings=mesh_rings(M);
  const uint64_t window=atomic_load_explicit(&s->window,memory_order_relaxed);
  for(uint32_t scanned=0;scanned<MESH_RINGS;scanned++){
    if(!s->out.active){
      if(s->descriptors_sent-s->descriptors_done>=MESH_DESCRIPTORS || s->send_frames+1>s->send_capacity)break;
      const uint32_t r=(s->cursor+scanned)%MESH_RINGS;
      const uint64_t at=atomic_load_explicit(rings+r,memory_order_acquire);
      if(!at || at<M->data_off || at>M->length-sizeof(struct mesh_requests))continue;
      struct mesh_requests *q=(void *)((char *)M+at);
      const uint64_t entries=q->entries;
      if(!entries || (entries&(entries-1)) || entries>(M->length-at-sizeof *q)/sizeof(struct mesh_stripe))continue;
      uint64_t k=atomic_load_explicit(&q->taken,memory_order_acquire);
      const struct mesh_stripe *e=q->request+(k&(entries-1));
      if(atomic_load_explicit(&e->ready,memory_order_acquire)!=k+1 || e->node!=port->peer)continue;
      const struct mesh_descriptor d={.offset=e->offset,.bytes=e->bytes,.word=e->word,.value=e->value};
      const uint64_t owner=q->owner;
      if(!atomic_compare_exchange_strong_explicit(&q->taken,&k,k+1,memory_order_acq_rel,memory_order_relaxed))continue;
      s->cursor=(r+1)%MESH_RINGS;moved=1;
      atomic_fetch_add_explicit(&s->flight,1,memory_order_acq_rel);
      if(d.offset>wire.length || d.bytes>wire.length-d.offset || d.offset>window || d.bytes>window-d.offset || !stripe_word_valid(d.word)){
        stripe_failed(at,owner,d.word,EINVAL);
        atomic_fetch_sub_explicit(&s->flight,1,memory_order_release);
        continue;
      }
      const uint32_t slot=(uint32_t)(s->descriptors_sent%MESH_DESCRIPTORS);
      const uint64_t offset=stripe_descriptor(port,MESH_DESCRIPTORS+slot);
      memcpy(wire.data+offset,&d,sizeof d);
      struct ibv_sge span=stripe_span(port,offset,sizeof d);
      struct ibv_send_wr posting={.wr_id=STRIPE_DESCRIPTOR|slot,.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
      const int error=lane->send(lane->pair,&posting,&bad);
      if(error){
        stripe_failed(at,owner,d.word,error<0?-error:error);
        atomic_fetch_sub_explicit(&s->flight,1,memory_order_release);
        lane_lost(port,MESH_LANE_STRIPES,error<0?-error:error);
        return 1;
      }
      s->descriptors_sent++;s->send_frames++;
      s->out=(struct stripe_cut){.d=d,.owner=owner,.ring=at,.active=1};
    }
    while(s->out.active){
      const uint64_t offset=s->out.d.offset+s->out.done;
      const uint32_t length=stripe_length(offset,s->out.d.bytes-s->out.done,s->piece),frames=length?(length+4095)/4096:1;
      if(s->send_frames+frames>s->send_capacity || s->send_head-s->send_tail>=MESH_STRIPE_CHUNKS)break;
      const int last=s->out.done+length>=s->out.d.bytes;
      const uint32_t slot=(uint32_t)(s->send_head%MESH_STRIPE_CHUNKS);
      s->sent[slot]=(struct stripe_piece){.word=last?s->out.d.word:0,.value=s->out.d.value,.owner=s->out.owner,.ring=s->out.ring,
        .frames=frames,.last=(uint32_t)last};
      struct ibv_sge span=stripe_span(port,offset,length);
      struct ibv_send_wr posting={.wr_id=STRIPE_DATA|slot,.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
      const int error=lane->send(lane->pair,&posting,&bad);
      if(error){lane_lost(port,MESH_LANE_STRIPES,error<0?-error:error);return 1;}
      s->send_frames+=frames;s->send_head++;s->out.done+=length;moved=1;
      if(last)s->out.active=0;
    }
    if(s->out.active)break;
  }
  return moved;
}
/* the next descriptor listened for, once the stripe the last named has every piece's receive posted (one at a time: a
   descriptor's receive posted ahead of its predecessor's pieces would take one of their SENDs) */
static int stripe_listen(struct port *port){
  struct stripes *s=&port->stripes;
  const struct mesh_queue *lane=port->link.provider.queues+MESH_LANE_STRIPES;
  const uint32_t slot=s->listened++%MESH_DESCRIPTORS;
  struct ibv_sge span=stripe_span(port,stripe_descriptor(port,slot),sizeof(struct mesh_descriptor));
  struct ibv_recv_wr request={.wr_id=STRIPE_DESCRIPTOR|slot,.sg_list=&span,.num_sge=1},*bad;
  const int error=lane->receive(lane->pair,&request,&bad);
  if(!error)s->receive_frames++;
  return error<0?-error:error;
}
/* a stripe's descriptor or piece landed, the receive thread's (a descriptor queued by its slot; a stripe's last piece
   stores its word here) */
static void stripe_landed(struct stripes *s,uint64_t id){
  const uint64_t slot=id&STRIPE_SLOT;
  if((id&~STRIPE_SLOT)==STRIPE_DESCRIPTOR){s->arrived[s->arrived_head++%MESH_DESCRIPTORS]=(uint32_t)slot;s->receive_frames--;return;}
  const struct stripe_piece *p=s->posted+slot%MESH_STRIPE_CHUNKS;
  s->receive_frames-=p->frames;s->receive_tail++;
  if(!p->last)return;
  stripe_word(p->word,p->value);
  atomic_fetch_add_explicit(&s->landed,1,memory_order_relaxed);
}
/* design/bridge-dataflow.md#stripes: one pass of the port's receive thread over its stripes (peer_serve's receive half):
   each landed descriptor's pieces' receives posted at its offset, while the lane's share of frames and the piece ring
   hold them, and the next descriptor listened for once they all are (credit flow control holds the peer's SENDs until
   then, TN3205).  A descriptor naming bytes past this node's window or a word that is no word is the peer's protocol
   error (EPROTO).  Whether anything moved; a failure is the lane's (lane_lost) */
static int stripes_receive(struct port *port){
  struct stripes *s=&port->stripes;
  const struct mesh_queue *lane=port->link.provider.queues+MESH_LANE_STRIPES;
  int moved=0;
  for(;;){
    if(!s->in.active){
      if(s->arrived_tail==s->arrived_head)break;
      const uint32_t slot=s->arrived[s->arrived_tail%MESH_DESCRIPTORS];
      memcpy(&s->in.d,wire.data+stripe_descriptor(port,slot),sizeof s->in.d);
      const struct mesh_descriptor *d=&s->in.d;
      if(d->offset>wire.length || d->bytes>wire.length-d->offset || !stripe_word_valid(d->word)){lane_lost(port,MESH_LANE_STRIPES,EPROTO);return 1;}
      s->in.done=0;s->in.slot=slot;s->in.active=1;
    }
    const uint64_t offset=s->in.d.offset+s->in.done;
    const uint32_t length=stripe_length(offset,s->in.d.bytes-s->in.done,s->piece),frames=length?(length+4095)/4096:1;
    if(s->receive_frames+frames>s->receive_capacity || s->receive_head-s->receive_tail>=MESH_STRIPE_CHUNKS)break;
    const int last=s->in.done+length>=s->in.d.bytes;
    const uint32_t slot=(uint32_t)(s->receive_head%MESH_STRIPE_CHUNKS);
    s->posted[slot]=(struct stripe_piece){.word=last?s->in.d.word:0,.value=s->in.d.value,.frames=frames,.last=(uint32_t)last};
    struct ibv_sge span=stripe_span(port,offset,length);
    struct ibv_recv_wr request={.wr_id=STRIPE_DATA|slot,.sg_list=&span,.num_sge=1},*bad;
    int error=lane->receive(lane->pair,&request,&bad);
    if(error){lane_lost(port,MESH_LANE_STRIPES,error<0?-error:error);return 1;}
    s->receive_frames+=frames;s->receive_head++;s->in.done+=length;moved=1;
    if(!last)continue;
    s->in.active=0;s->arrived_tail++;
    if((error=stripe_listen(port))){lane_lost(port,MESH_LANE_STRIPES,error);return 1;}
  }
  return moved;
}
/* the requests the stripes took and had not completed when their lane stopped being served (the send thread's, or a job's
   while no thread passes): each failed as a value (its writer's word UINT64_MAX, ECONNRESET) */
static void stripes_abandon(struct port *port){
  struct stripes *s=&port->stripes;
  uint64_t abandoned=0;
  if(s->out.active){stripe_failed(s->out.ring,s->out.owner,s->out.d.word,ECONNRESET);s->out.active=0;abandoned++;}
  for(;s->send_tail!=s->send_head;s->send_tail++){
    const struct stripe_piece *p=s->sent+s->send_tail%MESH_STRIPE_CHUNKS;
    if(p->last){stripe_failed(p->ring,p->owner,p->word,ECONNRESET);abandoned++;}
  }
  atomic_fetch_sub_explicit(&s->flight,abandoned,memory_order_release);
}
/* the stripes ready on a lane connected (made or made again), on the port's queue while no progress thread passes: what
   the last lane held failed as values, the piece both ends cut by (the lesser of this end's and the peer's, `peer_piece`
   as its LINK told it), the lane's share each way, every count from zero and the first descriptor listened for; 0, or
   the post's error */
static int stripes_ready(struct port *port,uint32_t peer_piece){
  struct stripes *s=&port->stripes;
  const struct link *l=&port->link;
  stripes_abandon(port);
  const uint32_t own=stripes_piece(port);
  s->piece=peer_piece && peer_piece/4096*4096<own?peer_piece/4096*4096:own;
  if(!s->piece)s->piece=4096;
  s->send_capacity=l->share>1?l->share:1;s->receive_capacity=l->share>1?l->share:1;
  s->send_head=s->send_tail=s->descriptors_sent=s->descriptors_done=s->receive_head=s->receive_tail=s->arrived_head=s->arrived_tail=0;
  s->send_frames=s->receive_frames=s->cursor=s->listened=0;
  s->out=s->in=(struct stripe_cut){0};
  return stripe_listen(port);
}

/* design/bridge-and-ipc.md#keepalive: the link's keepalive frame on lane MESH_LANE_KEEPALIVE, one frame of the link's
   block, where the configuration states a keepalive, a lane of the link is leased, none is in flight and keepalive_ns
   passed since the last (12f484f8's frame, per link); 1 where one was posted */
static int link_alive(struct port *port){
  struct link *l=&port->link;
  if(!keepalive_ns || l->alive_flight)return 0;
  int leased=0;
  for(uint32_t k=MESH_LANE_FIRST;k<MESH_LINK_LANES && !leased;k++)leased=atomic_load_explicit(&M->lanes[port->index][k],memory_order_relaxed)!=0;
  if(!leased)return 0;
  const uint64_t at=ledger_ns();
  if(at-l->alive_at<keepalive_ns)return 0;
  const struct mesh_queue *lane=l->provider.queues+MESH_LANE_KEEPALIVE;
  struct ibv_sge span=wire_span(&port->device,(uint64_t)M->block*M->pgsz,link_block(port),LINK_ALIVE_BYTES);
  struct ibv_send_wr posting={.wr_id=LINK_ALIVE_SENT,.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
  const int error=lane->send(lane->pair,&posting,&bad);
  if(error){lane_lost(port,MESH_LANE_KEEPALIVE,error<0?-error:error);return 1;}
  l->alive_flight=1;l->alive_at=at;l->alive_posted++;
  return 1;
}
/* `count` receives of one frame into the link's block posted on lane MESH_LANE_KEEPALIVE; 0, or the error */
static int link_alive_listen(struct port *port,uint32_t count){
  struct link *l=&port->link;
  const struct mesh_queue *lane=l->provider.queues+MESH_LANE_KEEPALIVE;
  struct ibv_sge span=wire_span(&port->device,(uint64_t)M->block*M->pgsz,link_block(port),4096);
  struct ibv_recv_wr request={.wr_id=LINK_ALIVE_HEARD,.sg_list=&span,.num_sge=1},*bad;
  int error=0;
  for(uint32_t k=0;k<count && !error;k++)error=lane->receive(lane->pair,&request,&bad);
  return error<0?-error:error;
}
/* the lane a link completion's request names */
static uint32_t link_lane(uint64_t id){ return (id&LINK_TAG)==LINK_ALIVE_SENT || (id&LINK_TAG)==LINK_ALIVE_HEARD?MESH_LANE_KEEPALIVE:MESH_LANE_STRIPES; }
/* design/bridge-and-ipc.md#lanes: the link's SENDs, one pass of the port's send thread: its SEND completions taken (a
   lane no longer served, or a queue pair made since, drops its own), then lane MESH_LANE_KEEPALIVE's frame and the
   stripes' requests posted while their lanes are served.  Whether anything moved */
static int link_send(struct port *port){
  struct link *l=&port->link;
  uint32_t served=atomic_load_explicit(&l->served,memory_order_acquire);
  struct ibv_wc done[16];
  const int count=l->provider.queues[0].poll(l->provider.sent,16,done);
  if(count<0){lane_lost(port,MESH_LANE_KEEPALIVE,EIO);lane_lost(port,MESH_LANE_STRIPES,EIO);return 1;}
  for(int i=0;i<count;i++){
    const uint32_t lane=link_lane(done[i].wr_id);
    if(!(served>>lane&1) || done[i].qp_num!=l->provider.queues[lane].pair->qp_num)continue;
    if(done[i].status){lane_lost(port,lane,(int64_t)done[i].status);served&=~(UINT32_C(1)<<lane);continue;}
    if(lane==MESH_LANE_KEEPALIVE)l->alive_flight=0;
    else stripe_sent(&port->stripes,done[i].wr_id);
  }
  int moved=count>0;
  if(served>>MESH_LANE_KEEPALIVE&1)moved|=link_alive(port);
  if(served>>MESH_LANE_STRIPES&1)moved|=stripes_send(port);
  return moved;
}
/* the link's receive completions, one pass of the port's receive thread: a keepalive frame heard (its receive posted
   again), a stripe's descriptor or piece landed; then the stripes' receives posted.  Whether anything moved */
static int link_receive(struct port *port){
  struct link *l=&port->link;
  uint32_t served=atomic_load_explicit(&l->served,memory_order_acquire);
  struct ibv_wc done[16];
  const int count=l->provider.queues[0].poll(l->provider.completion,16,done);
  if(count<0){lane_lost(port,MESH_LANE_KEEPALIVE,EIO);lane_lost(port,MESH_LANE_STRIPES,EIO);return 1;}
  if(count)atomic_thread_fence(memory_order_acquire);
  for(int i=0;i<count;i++){
    const uint32_t lane=link_lane(done[i].wr_id);
    if(!(served>>lane&1) || done[i].qp_num!=l->provider.queues[lane].pair->qp_num)continue;
    if(done[i].status){lane_lost(port,lane,(int64_t)done[i].status);served&=~(UINT32_C(1)<<lane);continue;}
    if(lane==MESH_LANE_KEEPALIVE){
      l->alive_heard++;
      const int error=link_alive_listen(port,1);
      if(error){lane_lost(port,lane,error);served&=~(UINT32_C(1)<<lane);}
    }
    else stripe_landed(&port->stripes,done[i].wr_id);
  }
  int moved=count>0;
  if(served>>MESH_LANE_STRIPES&1)moved|=stripes_receive(port);
  return moved;
}

/* design/bridge-dataflow.md#streams: a stream's failure, the event loop's to end (streams_turn) */
static void stream_lost(struct port *port,int64_t code){
  int64_t none=0;
  atomic_compare_exchange_strong(&port->stream.failed,&none,code?code:EIO);
  atomic_store_explicit(&port->stream.up,0,memory_order_release);
  wake();
}
/* the stream's next `n` pages from `posted`, chained as mesh-verbs-stream.c chained them: SENDs, or receives */
int stream_post(struct port *port,uint32_t n){
  struct stream *s=&port->stream;
  for(uint32_t i=0;i<n;i++){
    const uint64_t page=s->posted+i;
    s->spans[i]=stripe_span(port,stream_offset(port,page),s->page_bytes);
    if(s->sending)s->sends[i]=(struct ibv_send_wr){.wr_id=page,.next=i+1<n?s->sends+i+1:NULL,.sg_list=s->spans+i,.num_sge=1,
      .opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED};
    else s->receives[i]=(struct ibv_recv_wr){.wr_id=page,.next=i+1<n?s->receives+i+1:NULL,.sg_list=s->spans+i,.num_sge=1};
  }
  const struct mesh_queue *queue=s->provider.queues;
  struct ibv_send_wr *bad_send;struct ibv_recv_wr *bad_receive;
  const int error=s->sending?queue->send(queue->pair,s->sends,&bad_send):queue->receive(queue->pair,s->receives,&bad_receive);
  if(error)return error<0?-error:error;
  if(!s->posted)s->own.begin_ns=ledger_ns();
  s->posted+=n;s->own.posts++;
  return 0;
}
/* design/bridge-dataflow.md#streams: one pass of the stream at its end's progress thread (the send thread where it
   sends, the receive thread where it receives): its completions taken (a SEND's retiring its page; a receive's landing
   its page, in posting order and of a whole page, its count's first, tenth, ninetieth percent and last landing timed),
   then its next pages posted while fewer than `window` are in flight; the last completion finishing it */
static int stream_pass(struct port *port){
  struct stream *s=&port->stream;
  const struct mesh_queue *queue=s->provider.queues;
  const int count=queue->poll(s->sending?s->provider.sent:s->provider.completion,(int)s->window,s->completions);
  s->own.polls++;
  if(count<0){stream_lost(port,EIO);return 1;}
  if(!count)s->own.empty_polls++;
  for(int i=0;i<count;i++){
    const struct ibv_wc *done=s->completions+i;
    if(done->status){stream_lost(port,(int64_t)done->status);return 1;}
    if(!s->sending && done->byte_len!=s->page_bytes){stream_lost(port,EPROTO);return 1;}
    s->own.misordered+=done->wr_id!=s->completed+(uint64_t)i;
  }
  if(count){
    const uint64_t at=ledger_ns();
    s->completed+=(uint64_t)count;
    if(!s->own.first_ns){s->own.first_ns=at;s->own.first_count=s->completed;}
    if(!s->own.low_count && s->completed>=s->count/10){s->own.low_ns=at;s->own.low_count=s->completed;}
    if(!s->own.high_count && s->completed>=9*(uint64_t)s->count/10){s->own.high_ns=at;s->own.high_count=s->completed;}
    s->own.last_ns=at;s->own.completed=s->completed;
  }
  const uint64_t room=s->window-(s->posted-s->completed),left=s->count-s->posted,n=room<left?room:left;
  if(n){
    const int error=stream_post(port,(uint32_t)n);
    if(error){stream_lost(port,error);return 1;}
  }
  if(s->completed==s->count){atomic_store_explicit(&s->up,0,memory_order_release);atomic_store_explicit(&s->finished,1,memory_order_release);wake();}
  return count || n;
}

/* design/bridge-and-ipc.md#progress: one pass of a port's progress thread, `side` 0 its receives' and 1 its SENDs':
   every flow in the port's table it serves (while the flow is progressing) and the port's link lanes it serves.  The pass
   reads the table's epoch, serves each, and tells the epoch it began at, so a flow taken out of the table is read by
   neither thread once both tell a later epoch (port_unlist).  Whether anything moved; *any where something is the
   thread's to look for: a flow it serves, or a lane served (a stripe request may name the peer, and a keepalive frame
   or a stripe land, at any time).  Stripes in flight on a lane no longer served are failed as values here, by the
   thread that holds them */
static int port_pass(struct port *port,int side,int *any,uint64_t *told){
  const uint64_t epoch=atomic_load(&port->epoch);
  int moved=0;
  *any=0;
  const uint32_t high=atomic_load_explicit(&port->high,memory_order_acquire);
  for(uint32_t i=0;i<high;i++){
    struct flow *f=atomic_load_explicit(&port->table[i],memory_order_acquire);
    if(!f || !atomic_load_explicit(&f->progressing,memory_order_acquire))continue;
    *any=1;moved|=side?flow_send(f):flow_receive(f);
  }
  /* design/bridge-and-ipc.md#lanes */
  const uint32_t served=atomic_load_explicit(&port->link.served,memory_order_acquire);
  if(served){*any=1;moved|=side?link_send(port):link_receive(port);}
  if(side && !(served>>MESH_LANE_STRIPES&1) && (port->stripes.out.active || port->stripes.send_tail!=port->stripes.send_head))
    stripes_abandon(port);
  /* design/bridge-dataflow.md#streams: a stream, at the thread of its end's side */
  if(atomic_load_explicit(&port->stream.up,memory_order_acquire) && side==port->stream.sending){*any=1;moved|=stream_pass(port);}
  /* design/bridge-and-ipc.md#progress: told only when it moved (port_settle waits for it), so the line the two threads
     read every pass (`high`, `epoch`) is not written by either each pass */
  if(epoch!=*told){atomic_store_explicit(&port->seen[side],epoch,memory_order_release);*told=epoch;}
  return moved;
}
static void *port_progress(void *argument){
  struct port *port=((struct progress *)argument)->port;
  const int side=((struct progress *)argument)->side;
  free(argument);
  pthread_setname_np(side?"mesh.rdma.send":"mesh.rdma.receive");
  _Atomic uint64_t *bell=&M->bell[port->index];
  uint64_t quiet=0,told=atomic_load_explicit(&port->seen[side],memory_order_relaxed);
  uint32_t empty=0;
  int resting=0,any=0;
  while(!atomic_load_explicit(&port->closing,memory_order_acquire)){
    /* design/bridge-dataflow.md#stripes: while the port's guard stands, no pass (no post on any queue pair of its device)
       and no flow read, so the epoch is told as it moves (port_settle inside a guarded job returns) */
    const uint64_t guard=atomic_load_explicit(&port->guard,memory_order_acquire);
    if(guard&1){
      atomic_store_explicit(&port->guarded[side],guard,memory_order_release);
      while(atomic_load_explicit(&port->guard,memory_order_acquire)==guard && !atomic_load_explicit(&port->closing,memory_order_acquire)){
        const uint64_t epoch=atomic_load_explicit(&port->epoch,memory_order_acquire);
        if(epoch!=told){atomic_store_explicit(&port->seen[side],epoch,memory_order_release);told=epoch;}
        sched_yield();
      }
      empty=0;quiet=0;resting=0;
      continue;
    }
    const uint64_t rung=atomic_load_explicit(bell,memory_order_acquire);
    if(port_pass(port,side,&any,&told)){empty=0;quiet=0;resting=0;continue;}
    if(any && !resting){
      if(++empty<MESH_IDLE_PASSES)continue;
      empty=0;
      const uint64_t at=clock_ns();
      if(!quiet)quiet=at;
      if(at-quiet<MESH_IDLE_NS)continue;
    }
    atomic_store_explicit(&port->parked[side],1,memory_order_relaxed);
    atomic_thread_fence(memory_order_seq_cst);
    const uint64_t last=atomic_load_explicit(bell,memory_order_acquire);
    if(port_pass(port,side,&any,&told)){atomic_store_explicit(&port->parked[side],0,memory_order_relaxed);empty=0;quiet=0;resting=0;continue;}
    resting=1;
    os_sync_wait_on_address_with_timeout(bell,last,sizeof *bell,OS_SYNC_WAIT_ON_ADDRESS_SHARED,OS_CLOCK_MACH_ABSOLUTE_TIME,
      any?MESH_IDLE_POLL_NS:MESH_PAUSE_NS);
    atomic_store_explicit(&port->parked[side],0,memory_order_relaxed);
    if(atomic_load_explicit(bell,memory_order_acquire)!=rung){resting=0;quiet=0;empty=0;}
  }
  return NULL;
}
/* the port's bell rung: its parked progress threads look again */
void port_bell(struct port *port){ mesh_notify(&M->bell[port->index]); }
/* design/bridge-and-ipc.md#progress: the port's two progress threads started, once; 0, or pthread_create's error */
int port_threads(struct port *port){
  int error=0;
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_set_qos_class_np(&attributes,QOS_CLASS_USER_INTERACTIVE,0);
  while(port->threads<2 && !error){
    struct progress *progress=malloc(sizeof *progress);
    if(!progress){error=ENOMEM;break;}
    *progress=(struct progress){port,port->threads};
    atomic_store(&port->seen[port->threads],atomic_load(&port->epoch));
    if((error=pthread_create(&port->thread[port->threads],&attributes,port_progress,progress)))free(progress);
    else port->threads++;
  }
  pthread_attr_destroy(&attributes);
  return error;
}
/* design/bridge-and-ipc.md#progress: a flow in its port's table at its index, on the port's queue, its SENDs and receives
   served; 0, or the threads' error */
int port_list(struct flow *f){
  struct port *port=f->port;
  const int error=port_threads(port);
  if(error)return error;
  atomic_store_explicit(&f->progressing,1,memory_order_release);
  atomic_store_explicit(&port->table[f->slot],f,memory_order_release);
  f->listed=1;
  port_bell(port);
  return 0;
}
/* the port's table read by neither progress thread past a change made before: both tell an epoch past it */
void port_settle(struct port *port){
  const uint64_t epoch=atomic_fetch_add(&port->epoch,1)+1;
  for(int side=0;side<port->threads;side++)
    while(atomic_load_explicit(&port->seen[side],memory_order_acquire)<epoch){port_bell(port);sched_yield();}
}
/* design/bridge-dataflow.md#stripes, design/RDMA-KERNEL-RECOVERY.md "The stripe writes of October 1": the guard of a
   port's device, on the port's queue, around a change of its verbs objects (queue pairs and completion queues made,
   connected or destroyed): raised, it returns once each started progress thread holds (it told the guard's value before
   its next pass, and passes no more while the guard stands), so no queue pair of the device is posted on while another
   is made, changed or destroyed (Apple's provider: a post on a long-lived queue pair across another's teardown stores
   to the device's shared doorbell mapping after it moved, tbt_post_recv); lowered, the threads go on.  It stands for
   every change once the port's threads run, the link lanes' long-lived queue pairs being posted on while the link is up */
void port_guard(struct port *port,int raise){
  if(!raise){atomic_fetch_add_explicit(&port->guard,1,memory_order_release);port_bell(port);return;}
  const uint64_t guard=atomic_fetch_add_explicit(&port->guard,1,memory_order_acq_rel)+1;
  for(int side=0;side<port->threads;side++)
    while(atomic_load_explicit(&port->guarded[side],memory_order_acquire)!=guard){port_bell(port);sched_yield();}
}
int guard_raise(struct port *port){
  if(!port->threads)return 0;
  port_guard(port,1);
  return 1;
}
void guard_lower(struct port *port,int raised){ if(raised)port_guard(port,0); }
/* design/bridge-and-ipc.md#progress: a flow's SENDs (and an exclusive flow's receives) no longer served, on its port's
   queue, returning once neither thread is serving them (a flow on the lanes keeps taking its headers' records) */
void port_quiet(struct flow *f){
  atomic_store_explicit(&f->progressing,0,memory_order_release);
  if(f->listed)port_settle(f->port);
}
/* design/bridge-and-ipc.md#progress: a flow out of its port's table, on the port's queue, returning once neither
   progress thread can read it (a lane's records it posted before land in its rows: flow_release waits on them) */
void port_unlist(struct flow *f){
  atomic_store_explicit(&f->progressing,0,memory_order_release);
  if(!f->listed)return;
  struct flow *held=f;
  atomic_compare_exchange_strong(&f->port->table[f->slot],&held,NULL);
  port_settle(f->port);
  f->listed=0;
}
/* design/bridge-and-ipc.md#region: the queue pairs a port's device holds, told in its port table */
void port_pairs(struct port *port){
  atomic_store_explicit(&mesh_links(M)[port->index].pairs,port->device.pairs,memory_order_relaxed);
}

/* design/bridge-and-ipc.md#protocol: a message to the port's peer, queued behind what the socket has not taken yet; a
   send that fails is the link's loss */
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
void port_say(struct port *port,struct message said,const void *payload,uint32_t bytes){
  if(port->control<0 || (said.kind!=HELLO && !port->greeted))return;
  said.magic=XMAGIC;said.node=M->node;said.boot=M->boot;said.block=M->block*M->pgsz;said.port=port->index;said.bytes=bytes;
  said.lid=port->lid;memcpy(said.gid,port->gid,16);
  size_t length=port->out_length+sizeof said+bytes;
  if(length-port->out_sent>MESH_UNSENT){port_lost(port,ENOBUFS);return;}
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
/* the accepted program of execution `key` that binds `node` here (its slot's owner and destination), with a flow toward
   it that has not failed; 0 where none */
uint64_t owners[MESH_CLIENTS],accepted[MESH_CLIENTS];
/* design/bridge-and-ipc.md#records: this node's record, its window, memory and model with its ports and health */
char model[MESH_MODEL_BYTES];
/* a port's link as the port table tells it; a change is a new record */
static void link_set(struct port *port,uint32_t peer,uint32_t phase,int64_t code){
  struct mesh_link_info *link=&mesh_links(M)[port->index];
  int changed=atomic_load(&link->peer)!=peer || atomic_load(&link->phase)!=phase;
  atomic_store_explicit(&link->peer,peer,memory_order_relaxed);atomic_store_explicit(&link->code,code,memory_order_relaxed);
  atomic_store_explicit(&link->phase,phase,memory_order_release);
  if(changed)record_changed();
}

/* design/bridge-and-ipc.md#lanes: a link job on the port's queue, guarded (no queue pair of the device is posted on while
   one is made, connected or destroyed: port_guard), its verbs calls each made on hearing the peer and never on a
   failure, from the peer's numbers as the job was started (`numbers`: the event loop takes the peer's next LINK
   meanwhile): the lanes made (LINK_JOB_MAKE: those of the link's last pairing destroyed where the context answers,
   d6dceac's down_pair, or forgotten where it went with the peer, then MESH_LINK_LANES UC queue pairs in INIT, d6dceac's
   verbs_up); connected to the peer's (LINK_JOB_CONNECT, every lane; LINK_JOB_JOIN, the leader's lanes of a round once
   acknowledged: RTR, then RTS, each lane's first receives posted and the port's progress threads started); a round's
   lanes made again in INIT (LINK_JOB_REMAKE, the leader's); or made again and joined at once to the leader's
   (LINK_JOB_REJOIN, the higher node's) */
struct link_numbers { uint32_t peer[MESH_LINK_LANES][2],piece; };
static int link_make(struct port *port){
  struct link *l=&port->link;
  struct mesh_verbs *p=&l->provider;
  int error=0;
  struct ibv_port_attr attributes;
  if(p->queues && (ibv_query_port(port->device.context,1,&attributes)?(error=errno?errno:EIO):(error=pairs_down(p)))){
    say("link %u (%s): the lanes of its last pairing not destroyed (%s): the device's context forgotten and opened afresh\n",
        port->index,port->name,strerror(error));
    pair_forget(p);pair_forget(&port->stream.provider);
    down_device(&port->device);
    int refused=0;
    if((error=device_up(&port->device,&wire,&refused)) || (error=device_gid(&port->device)))return error;
  }
  if((error=pair_up(p,MESH_LINK_LANES)))return error;
  l->share=(uint32_t)MIN(p->completion->cqe,p->sent->cqe)/MESH_LINK_LANES;
  return 0;
}
static int link_connect(struct port *port,uint32_t lanes,const struct link_numbers *numbers){
  struct link *l=&port->link;
  struct mesh_verbs *p=&l->provider;
  int error=0;
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)if(lanes>>k&1)error=queue_receive(p,(int)k,numbers->peer[k],port->heard.lid,port->heard.gid);
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)if(lanes>>k&1)error=queue_send(p,(int)k);
  if(!error && lanes>>MESH_LANE_KEEPALIVE&1){
    l->alive_flight=0;l->alive_at=0;
    error=link_alive_listen(port,MIN(LINK_ALIVE_RECEIVES,l->share>1?l->share-1:1));
  }
  if(!error && lanes>>MESH_LANE_STRIPES&1)error=stripes_ready(port,numbers->piece);
  return error;
}
static int lanes_remake(struct port *port,uint32_t lanes){
  struct mesh_verbs *p=&port->link.provider;
  for(uint32_t k=0;k<MESH_LINK_LANES;k++)if(lanes>>k&1){
    struct ibv_qp *q=p->queues[k].pair;
    if(q){if(ibv_destroy_qp(q))return errno?errno:EIO;p->queues[k].pair=NULL;p->device->pairs--;}
    const int error=pair_make(p,(int)k);
    if(error)return error;
  }
  return 0;
}
static void link_job(struct port *port,int job,uint32_t lanes){
  struct link *l=&port->link;
  struct link_numbers numbers={.piece=l->peer_piece};
  memcpy(numbers.peer,l->peer,sizeof numbers.peer);
  l->job=job;l->joining=lanes;l->job_pairing=l->pairing;atomic_store_explicit(&l->done,0,memory_order_relaxed);
  dispatch_async(port->work,^{
    const int raised=guard_raise(port);
    int result;
    if(job==LINK_JOB_MAKE)result=link_make(port);
    else if(job==LINK_JOB_REMAKE)result=lanes_remake(port,lanes);
    else if(job==LINK_JOB_REJOIN){result=lanes_remake(port,lanes);if(!result)result=link_connect(port,lanes,&numbers);}
    else result=link_connect(port,lanes,&numbers);
    if(!result && job==LINK_JOB_CONNECT)result=port_threads(port);
    guard_lower(port,raised);
    l->result=result;
    port_pairs(port);
    atomic_store_explicit(&l->done,1,memory_order_release);
    wake();
  });
}
/* a LINK telling the peer lanes of this end's (`count` from `first`, with `round`, the lane's round: the leader's own,
   the higher node's the leader's it joined): each one's queue pair number and first sequence number, and the largest
   stripe piece this end takes; a LINK of no lanes, the higher node's report of lane `first` failed or dirty */
static void link_tell(struct port *port,uint32_t first,uint32_t count,uint32_t round){
  const struct link *l=&port->link;
  struct message said={.kind=LINK,.count=count,.from=first,.key=round,.code=stripes_piece(port)};
  for(uint32_t k=0;k<count;k++){
    said.pair[k][0]=l->provider.queues[first+k].pair->qp_num;said.pair[k][1]=(l->provider.psn+first+k)&0xffffff;
  }
  port_say(port,said,NULL,0);
}
/* the link's lower node leads its lanes' rounds (DESIGN.md L2) */
static int link_leads(const struct port *port){ return M->node<port->peer; }
/* the link of a port lost (its peer's loss, a failure, the bridge stopping): its lanes served no more and left as they
   are, no verbs call made (its next pairing destroys and makes them again, link_make); a job of this pairing still on
   the port's queue is dropped as it ends; the send thread fails the stripes it holds (port_pass) */
static void link_lost(struct port *port){
  struct link *l=&port->link;
  l->pairing++;l->state=LINK_DOWN;l->heard=0;l->dirty=l->round=l->acked=l->pending=0;
  atomic_store_explicit(&l->served,0,memory_order_release);
  atomic_store_explicit(&l->broken,0,memory_order_relaxed);
  atomic_store_explicit(&mesh_links(M)[port->index].lanes,0,memory_order_relaxed);
  port_bell(port);
}
/* design/bridge-and-ipc.md#lanes: the leader's lanes failed, dirty or reported, a round of each begun now that the peer
   is heard (a message on the link's control socket): made again (LINK_JOB_REMAKE), then each told in a LINK of its
   round, which the higher node makes its own again on, joins and acknowledges */
static void link_remake(struct port *port){
  struct link *l=&port->link;
  if(l->job || l->state!=LINK_UP || port->parted || port->control<0 || !link_leads(port))return;
  const uint32_t lanes=l->dirty&~l->round;
  if(lanes)link_job(port,LINK_JOB_REMAKE,lanes);
}
/* the lease words a link's clients left dirty, among `lanes` */
static uint32_t lanes_dirty(const struct port *port){
  uint32_t dirty=0;
  for(uint32_t k=MESH_LANE_FIRST;k<MESH_LINK_LANES;k++)
    if(atomic_load_explicit(&M->lanes[port->index][k],memory_order_acquire)&MESH_LANE_DIRTY)dirty|=UINT32_C(1)<<k;
  return dirty;
}
static void lanes_clean(struct port *port,uint32_t lanes){
  for(uint32_t k=MESH_LANE_FIRST;k<MESH_LINK_LANES;k++)
    if(lanes>>k&1)atomic_fetch_and_explicit(&M->lanes[port->index][k],~MESH_LANE_DIRTY,memory_order_acq_rel);
}
/* design/bridge-and-ipc.md#lanes: the link's turn, on the event loop: a job's end taken (made: LINK told the peer every
   lane; connected: up, every lane served but those already to be made again; a round's lanes made again:
   each told; joined: served again, their lease words' DIRTY cleared, and the higher node's acknowledged); a job that
   failed takes the link down and leaves the port down until its link next comes on (no retry: design/recovery.md#loss);
   the lanes the progress threads found failed and the leased lanes their clients left dirty taken (served no more: the
   leader's to make again once the peer is next heard, link_remake; the higher node's reported, once); then the lanes
   made once the port is paired, connected once both ends' are made, a round's acknowledged lanes joined, and the lanes
   the leader made again made again and joined here */
static void link_turn(struct port *port){
  struct link *l=&port->link;
  if(l->job){
    if(!atomic_load_explicit(&l->done,memory_order_acquire))return;
    const int job=l->job,result=l->result;
    const uint32_t lanes=l->joining;
    l->job=0;
    if(l->job_pairing!=l->pairing){}
    else if(result){
      say("link %u (%s): lanes toward node %u not %s: %s; the port is down until its link next comes on\n",port->index,port->name,
          port->peer,job==LINK_JOB_MAKE?"made":job==LINK_JOB_CONNECT?"connected":job==LINK_JOB_JOIN?"joined again":"made again",
          strerror(result));
      port->open=0;
      port_lost(port,result);
      return;
    }
    else if(job==LINK_JOB_MAKE){
      if(!l->counted){port->committed+=MESH_LINK_LANES;l->counted=1;}
      l->state=LINK_MADE;
      link_tell(port,0,MESH_LINK_LANES,0);
    }else if(job==LINK_JOB_CONNECT){
      l->state=LINK_UP;
      atomic_store_explicit(&mesh_links(M)[port->index].lanes,MESH_LINK_LANES,memory_order_relaxed);
      atomic_store_explicit(&l->served,((UINT32_C(1)<<MESH_LINK_LANES)-1)&~l->pending&~l->dirty,memory_order_release);
      port_bell(port);
      say("link %u (%s): %u lanes up toward node %u: stripe pieces of %u bytes, each lane's share %u frames\n",port->index,port->name,
          MESH_LINK_LANES,port->peer,port->stripes.piece,l->share);
      record_changed();
    }else if(job==LINK_JOB_REMAKE){
      for(uint32_t k=0;k<MESH_LINK_LANES;k++)if(lanes>>k&1){l->seq[k]++;link_tell(port,k,1,l->seq[k]);}
      l->round|=lanes;l->acked&=~lanes;l->dirty&=~lanes;
    }else if(job==LINK_JOB_JOIN){
      l->round&=~lanes;l->acked&=~lanes;
      lanes_clean(port,lanes);
      atomic_fetch_or_explicit(&l->served,lanes,memory_order_release);
      port_bell(port);
      say("link %u (%s): lanes %#x made again toward node %u\n",port->index,port->name,lanes,port->peer);
    }else{
      for(uint32_t k=0;k<MESH_LINK_LANES;k++)if(lanes>>k&1)link_tell(port,k,1,l->seq[k]);
      l->dirty&=~lanes;
      lanes_clean(port,lanes);
      atomic_fetch_or_explicit(&l->served,lanes&~l->pending,memory_order_release);
      port_bell(port);
      say("link %u (%s): lanes %#x made again toward node %u, as it asked\n",port->index,port->name,lanes,port->peer);
    }
  }
  if(!port->greeted)return;
  /* design/bridge-dataflow.md#stripes: the peer's window, its record's: no stripe request passes it */
  const struct mesh_record *r=record_held(port->peer);
  atomic_store_explicit(&port->stripes.window,r?r->window:0,memory_order_relaxed);
  if(l->state==LINK_UP || l->state==LINK_CONNECTING){
    const uint32_t found=(atomic_exchange_explicit(&l->broken,0,memory_order_acq_rel)|lanes_dirty(port))&~l->dirty&~l->round&~l->pending;
    if(found){
      l->dirty|=found;
      atomic_fetch_and_explicit(&l->served,~found,memory_order_acq_rel);
      if(!link_leads(port))for(uint32_t k=0;k<MESH_LINK_LANES;k++)if(found>>k&1)link_tell(port,k,0,0);
    }
  }
  if(l->job)return;
  if(l->state==LINK_DOWN){
    if(l->page==MESH_ABSENT && (l->page=mesh_arena_alloc(&bridge,l->pages,M->block))==MESH_ABSENT){
      say("link %u (%s): lanes not made: the window holds no %u pages for their blocks; the port is down until its link next comes on\n",
          port->index,port->name,l->pages);
      port->open=0;
      port_lost(port,ENOMEM);
      return;
    }
    link_job(port,LINK_JOB_MAKE,0);
  }
  else if(l->state==LINK_MADE && l->heard){l->state=LINK_CONNECTING;link_job(port,LINK_JOB_CONNECT,(UINT32_C(1)<<MESH_LINK_LANES)-1);}
  else if(l->state==LINK_UP && link_leads(port) && (l->round&l->acked))link_job(port,LINK_JOB_JOIN,l->round&l->acked);
  else if(l->state==LINK_UP && !link_leads(port) && l->pending){
    const uint32_t lanes=l->pending;
    l->pending=0;
    link_job(port,LINK_JOB_REJOIN,lanes);
  }
}
/* a peer's LINK: every lane of its pairing (heard once, before this end's connect: a second is a protocol error); or, while
   the link connects or is up, one lane's: to the leader, the higher node's acknowledgement of the lane's open round (its
   new numbers; one of an older round is dropped) or its report of the lane failed (no lanes: taken where no round of the
   lane is open or the open round's acknowledgement came, else it names the queue pair the round replaces); to the higher
   node, a round the leader began (made again and joined here, link_turn) */
static void link_heard(struct port *port,const struct message *heard){
  struct link *l=&port->link;
  if(heard->from==0 && heard->count==MESH_LINK_LANES && !l->heard && (l->state==LINK_DOWN || l->state==LINK_MADE)){
    memcpy(l->peer,heard->pair,sizeof l->peer);
    l->peer_piece=(uint32_t)heard->code;l->heard=1;
    link_turn(port);
    return;
  }
  const uint32_t k=heard->from,bit=UINT32_C(1)<<(k%32);
  if(k<MESH_LINK_LANES && heard->count<=1 && (l->state==LINK_UP || l->state==LINK_CONNECTING)){
    if(link_leads(port) && !heard->count){
      if(!(l->round&bit) || (l->acked&bit)){l->dirty|=bit;atomic_fetch_and_explicit(&l->served,~bit,memory_order_acq_rel);}
      return;
    }
    if(link_leads(port)){
      if((l->round&bit) && heard->key==l->seq[k]){
        l->peer[k][0]=heard->pair[0][0];l->peer[k][1]=heard->pair[0][1];
        if(k==MESH_LANE_STRIPES)l->peer_piece=(uint32_t)heard->code;
        l->acked|=bit;
      }
      return;
    }
    if(heard->count){
      l->peer[k][0]=heard->pair[0][0];l->peer[k][1]=heard->pair[0][1];l->seq[k]=(uint32_t)heard->key;
      if(k==MESH_LANE_STRIPES)l->peer_piece=(uint32_t)heard->code;
      l->pending|=bit;
      atomic_fetch_and_explicit(&l->served,~bit,memory_order_acq_rel);
      return;
    }
  }
  port_lost(port,EPROTO);
}

int exec_served;

/* design/bridge-and-ipc.md#protocol */
static void message_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  if(heard->kind==RECORD){record_heard(port,payload,heard->bytes);return;}
  if(heard->kind==FAREWELL){port->parted=1;port->farewell=heard->code;return;}
  if(heard->kind==READY){ready_heard(port,heard,payload);return;}
  if(heard->kind==LINK){link_heard(port,heard);return;}
  if(heard->kind==STREAM){stream_heard(port,heard,payload);return;}
  if(heard->kind>=CHANNEL_OPEN && heard->kind<=CHANNEL_CLOSE){channel_heard(port,heard,payload);return;}
  struct flow *f=flow_on(port,heard->key,heard->to,heard->from,heard->stripe);
  if(!f)return;
  const uint32_t origin=heard->failed?heard->failed:port->peer;
  if(heard->kind==PAIRS){
    if(heard->count!=(uint32_t)f->qps)flow_fail(f,EPROTO,4,FAIL,port->peer,M->node);
    else{memcpy(f->pairs,heard->pair,sizeof f->pairs);f->heard|=HEARD_PAIRS;}
  }
  else if(heard->kind==POSTED){f->heard|=HEARD_POSTED;if(f->partner)flow_tell(f->partner);}
  else if(heard->kind==END)flow_fail(f,ECONNRESET,4,0,origin,origin);
  else if(heard->kind==FAIL){
    const uint64_t owner=f->partner?0:f->owner;
    flow_fail(f,heard->code==ESTALE?ESTALE:ECANCELED,4,0,origin,origin);
    if(heard->code==ESTALE && owner)program_stale(owner,origin);
  }
  flow_step(f);
}
/* a message heard on a port's control socket: the first its peer's hello (the link paired, its lanes made: link_turn),
   then each taken (message_heard), after which the lanes failed or dirty are made again, the peer being heard
   (link_remake) */
static void port_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  if(!port->greeted){
    if(heard->kind!=HELLO || heard->block!=M->block*M->pgsz || (port->peer && heard->node!=port->peer)){port_lost(port,EPROTO);return;}
    port->peer=heard->node;port->peer_port=heard->port;port->heard=*heard;port->greeted=1;
    struct mesh_link_info *link=&mesh_links(M)[port->index];
    atomic_store_explicit(&link->bandwidth,port->bandwidth,memory_order_relaxed);
    atomic_fetch_add_explicit(&link->pairings,1,memory_order_relaxed);
    say("link %u (%s): paired with node %u\n",port->index,port->name,port->peer);
    link_set(port,port->peer,MESH_PAIRED,0);
    /* a port newly paired is sent every record the map holds, this node's first */
    for(uint64_t i=0;port->greeted && i<atomic_load(&M->map_nodes);i++)if(mesh_records(M)[i].node!=M->node)
      port_say(port,(struct message){.kind=RECORD},mesh_records(M)+i,sizeof(struct mesh_record));
    if(port->greeted)link_turn(port);
    return;
  }
  message_heard(port,heard,payload);
  if(port->greeted)link_remake(port);
}

/* design/recovery.md#detection, #loss: a link lost fails every program on it, naming its peer; the port keeps its
   device's context, protection domain and registration and its link lanes as they are, no verbs call made
   (design/RDMA-KERNEL-RECOVERY.md "October 18"), and a port that was paired says hello at once where it is open
   (port_hello: a dial or a greeting that failed keeps the hellos' pace, so nothing is tried faster than they go) */
void port_lost(struct port *port,int64_t code){
  if(port->control>=0){close(port->control);port->control=-1;}
  if(port->greeted)port->next=now;
  /* design/recovery.md#diagnosis: the peer named, and a peer that said farewell told from one lost */
  if(port->greeted||port->connecting){
    char reason[96];farewell_text(port->farewell,reason,sizeof reason);
    if(stop)say("link %u (%s): closed toward node %u: this bridge is stopping\n",port->index,port->name,port->peer);
    else if(port->peer && port->parted)say("link %u (%s): node %u parted, its bridge stopping (%s): code %lld\n",port->index,port->name,
                                            port->peer,reason,(long long)code);
    else if(port->peer)say("link %u (%s): node %u lost: code %lld\n",port->index,port->name,port->peer,(long long)code);
    else say("link %u (%s): lost: code %lld\n",port->index,port->name,(long long)code);
  }
  link_lost(port);
  for(struct flow *f=flows;f;f=f->next)if(f->port==port)flow_fail(f,port->parted?ECONNRESET:code,4,0,port->peer,M->node);
  requests_lost(port,port->parted?ECONNRESET:code);
  /* design/bridge-dataflow.md#streams: a stream on the port fails, its requester answered */
  if(port->stream.state!=STREAMS_DOWN && port->stream.state!=STREAMS_CLOSING)streams_fail(port,port->parted?ECONNRESET:code,0);
  port->greeted=port->parted=port->connecting=port->writing=0;port->farewell=0;port->peer=0;port->held=0;port->out_length=port->out_sent=0;
  link_set(port,0,port->open?MESH_PAIRING:MESH_STOPPED,code);
}
/* the control socket: non-blocking, its sends queued (port_say); its peer's loss is the kernel's to tell (its end
   closed, a reset, the link's going off), never a clock's (design/recovery.md#detection) */
static void port_greet(struct port *port){
  int on=1,flags=fcntl(port->control,F_GETFL);
  if(flags<0 || fcntl(port->control,F_SETFL,flags|O_NONBLOCK) || setsockopt(port->control,SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on)){
    port_lost(port,errno?errno:EIO);return;
  }
  port_watch(port,EVFILT_READ);
  port_say(port,(struct message){.kind=HELLO},NULL,0);
}
/* design/bridge-and-ipc.md#ports: the port a link-local peer reached, by its interface; none for a scope of 0 (no
   interface: a LAN address, or the index of an interface that is gone) */
static struct port *port_at(uint32_t interface){
  for(uint32_t p=0;interface && p<port_count;p++)if(ports[p].interface==interface)return ports+p;
  return NULL;
}
/* design/bridge-and-ipc.md#ports: a port's device opened, on its own queue (guarded), where it is to be (at the start, and
   at its link's coming on or its interface's attaching, while it has no peer): its port read; a context that fails with
   the port active is gone (the peer's power went with it): forgotten with every object made on it, no verbs call
   (design/RDMA-KERNEL-RECOVERY.md "October 18"), and the device opened afresh */
static void port_open(struct port *port){
  if(port->job || !port->opening || port->control>=0)return;
  port->opening=0;
  port->interface=if_nametoindex(port->name);
  port->job=1;atomic_store_explicit(&port->done,0,memory_order_relaxed);
  dispatch_async(port->work,^{
    const int raised=guard_raise(port);
    int refused=0,result=device_up(&port->device,&wire,&refused);
    if(result && !refused && result!=ENETDOWN && port->device.context){
      say("link %u (%s): its device's context failed (%s): forgotten and opened afresh\n",port->index,port->name,strerror(result));
      pair_forget(&port->link.provider);pair_forget(&port->stream.provider);
      down_device(&port->device);
      result=device_up(&port->device,&wire,&refused);
    }
    if(!result)result=device_gid(&port->device);
    guard_lower(port,raised);
    port->result=result;port->refused=refused;
    atomic_store_explicit(&port->done,1,memory_order_release);
    wake();
  });
}
/* design/recovery.md#registration: a port's opening taken: a device whose registration of the window it refused keeps
   the extents it took and opens again at its link's next coming on (device_up logged the extent and why); a device
   that failed otherwise, likewise; one opened says hello at once (port_hello) */
static void port_opened(struct port *port){
  const int result=port->result;
  port->job=0;
  if(result){
    if(port->refused)say("link %u (%s): the window's registration refused (%s): the port is down until its link next comes on\n",
                         port->index,port->name,strerror(port->refused));
    port->open=0;if(port->control<0)link_set(port,0,MESH_STOPPED,result);
    return;
  }
  atomic_store_explicit(&mesh_links(M)[port->index].max_pairs,port->device.max_pairs,memory_order_relaxed);
  port->open=1;port->lid=port->device.port.lid;memcpy(port->gid,&port->device.gid,16);
  port->bandwidth=port_bandwidth(&port->device.port);
  if(port->control>=0)return;
  link_set(port,0,MESH_PAIRING,0);
  port->next=now;
}
/* design/bridge-and-ipc.md#ports: an open port with no peer says hello to every node on its link, every MESH_PAUSE_NS (a
   paced emission, deciding nothing: a peer that starts later is heard as it says its own) */
static void port_hello(struct port *port){
  if(!port->open || port->control>=0 || port->job || now<port->next)return;
  port->next=now+MESH_PAUSE_NS;
#ifdef MESH_LOOP_FABRIC
  /* design/bridge-dataflow.md#the-loopback-fabric: the cable's lower end dials the higher's listener on loopback, as a
     link's lower node dials the higher's on the address it heard; the higher accepts (main, TAG_LOOP) */
  for(uint32_t k=0;k<loop_config.ports;k++)if(!strcmp(loop_config.port[k].device,port->device.name) && loop_config.port[k].listen<loop_config.port[k].peer){
    struct sockaddr_in6 to={.sin6_len=sizeof to,.sin6_family=AF_INET6,.sin6_port=htons(loop_config.port[k].peer),.sin6_addr=IN6ADDR_LOOPBACK_INIT};
    port->control=socket(AF_INET6,SOCK_STREAM,0);
    if(port->control<0 || fcntl(port->control,F_SETFL,O_NONBLOCK) || (connect(port->control,(struct sockaddr *)&to,sizeof to) && errno!=EINPROGRESS)){
      port_lost(port,errno);return;
    }
    port->peer=0;port->connecting=1;
    port_watch(port,EVFILT_WRITE);
  }
  return;
#endif
  struct ipv6_mreq group={.ipv6mr_multiaddr=everyone,.ipv6mr_interface=port->interface};
  setsockopt(hello,IPPROTO_IPV6,IPV6_JOIN_GROUP,&group,sizeof group);
  setsockopt(hello,IPPROTO_IPV6,IPV6_MULTICAST_IF,&port->interface,sizeof port->interface);
  struct sockaddr_in6 to={.sin6_len=sizeof to,.sin6_family=AF_INET6,.sin6_port=htons((uint16_t)atoi(MESH_PORT)),.sin6_addr=everyone,.sin6_scope_id=port->interface};
  struct message said={.magic=XMAGIC,.kind=HELLO,.node=M->node,.boot=M->boot,.block=M->block*M->pgsz,.port=port->index};
  sendto(hello,&said,sizeof said,0,(struct sockaddr *)&to,sizeof to);
}
/* design/bridge-and-ipc.md#ports: a node heard on a port: the lower node dials the higher's listener on the address
   it heard from, the higher answers its hello so the lower hears it at once */
static void hello_heard(void){
  struct message heard;struct sockaddr_in6 from;socklen_t length=sizeof from;
  while(recvfrom(hello,&heard,sizeof heard,0,(struct sockaddr *)&from,&length)==(ssize_t)sizeof heard){
    struct port *port=port_at(from.sin6_scope_id);
    length=sizeof from;
    if(!port || heard.magic!=XMAGIC || heard.kind!=HELLO || heard.node==M->node || port->control>=0 || !port->open)continue;
    if(M->node>heard.node){
      struct message said={.magic=XMAGIC,.kind=HELLO,.node=M->node,.boot=M->boot,.block=M->block*M->pgsz,.port=port->index};
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
/* a control socket's messages, each a header and its `bytes` */
static void control_heard(struct port *port){
  for(;;){
    size_t need=sizeof(struct message);
    if(port->held>=need){
      const struct message *header=(const void *)port->in;
      if(header->magic!=XMAGIC || header->bytes>(UINT32_C(64)<<20)){port_lost(port,EPROTO);return;}
      need+=header->bytes;
    }
    if(need>port->in_room){
      unsigned char *grown=realloc(port->in,need);
      if(!grown){port_lost(port,ENOMEM);return;}
      port->in=grown;port->in_room=need;
    }
    if(port->held<need){
      ssize_t got=recv(port->control,port->in+port->held,need-port->held,0);
      if(got<0 && (errno==EAGAIN || errno==EINTR))return;
      if(got<=0){port_lost(port,got?errno:(port->parted?ECONNRESET:ENETRESET));return;}
      port->held+=(size_t)got;
      if(port->held<need || need==sizeof(struct message))continue;
    }
    struct message heard;memcpy(&heard,port->in,sizeof heard);
    port->held=0;
    port_heard(port,&heard,port->in+sizeof heard);
    if(port->control<0)return;
  }
}
/* design/bridge-and-ipc.md#lanes: the link pages of every port the region has, leased at the bridge's start beside the
   map's, so a port found later (an interface attached once clients hold the window) has them; a port whose pages the
   window could not give takes them as its link first pairs (link_turn) */
static uint32_t link_page[MESH_PORTS];
static void link_reserve(void){
  for(uint32_t p=0;p<MESH_PORTS;p++)link_page[p]=mesh_arena_alloc(&bridge,link_pages(),M->block);
}
/* design/bridge-and-ipc.md#ports: every RDMA device of the node a port, in the order found, read on the finder's queue
   (at the start, and where an interface attached or its link came on that no port has) and added on the event loop */
static void ports_find(void){
  if(finding)return;
  finding=1;
  dispatch_async(finder,^{
    struct ibv_device **list=ibv_get_device_list(NULL);
    uint32_t n=0;
#ifdef MESH_LOOP_FABRIC
    /* design/bridge-dataflow.md#the-loopback-fabric: the configuration's devices alone, in its order */
    for(uint32_t k=0;k<loop_config.ports && n<MESH_PORTS;k++)snprintf(devices[n++],sizeof devices[0],"%s",loop_config.port[k].device);
    if(list)ibv_free_device_list(list);
    list=NULL;
#endif
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
    *port=(struct port){.index=port_count,.control=-1,.opening=1,.work=dispatch_queue_create("mesh.port",DISPATCH_QUEUE_SERIAL),.stream={.fd=-1}};
    /* design/bridge-and-ipc.md#lanes: the link's block and descriptors, the pages leased for the port's index */
    port->link.pages=link_pages();
    port->link.page=link_page[port_count];
    port->link.provider.device=&port->device;
    port->stream.provider.device=&port->device;
    snprintf(port->device.name,sizeof port->device.name,"%s",devices[i]);snprintf(port->name,sizeof port->name,"%s",devices[i]+5);
    port_count++;
    link_set(port,0,MESH_STOPPED,0);
  }
}
/* design/prepared-machine.md#M24: a port's link off or its interface detached (the link lost, the port down until its
   link comes on: its device opened then, port_open); an interface that attached or whose link came on and is no port
   yet: the node's devices read again */
static void network_heard(void){
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
      if(code==KEV_DL_LINK_OFF || code==KEV_DL_IF_DETACHED){ports[p].open=0;port_lost(ports+p,ENETDOWN);}
      else if(code==KEV_DL_LINK_ON || code==KEV_DL_IF_ATTACHED)ports[p].opening=1;
    }
    unknown|=!known && (code==KEV_DL_LINK_ON || code==KEV_DL_IF_ATTACHED);
  }
  if(unknown)ports_find();
}

/* design/prepared-machine.md#M26: the clients' control word, watched for the event loop */
static void *control_watch(void *unused){
  (void)unused;
  while(!stop){
    uint64_t seen=atomic_load_explicit(&M->control,memory_order_acquire);
    wake();
    os_sync_wait_on_address(&M->control,seen,sizeof M->control,OS_SYNC_WAIT_ON_ADDRESS_SHARED);
  }
  return NULL;
}

/* design/prepared-machine.md#M09: the bridge's configuration, the canonical property list bin/mesh-user.sh writes from
   the consumer's document (rdma/bridge_configuration.py): its name and its window in bytes, and whether it serves `exec`
   and keeps ledgers (#ledger), and its `keepalive` in nanoseconds where it states one (#keepalive); NULL, or why it is
   refused */
#ifdef MESH_LOOP_FABRIC
/* design/bridge-dataflow.md#the-loopback-fabric: a loop bridge's `loop` (a dictionary: `region`, `fabric`, `node`,
   `service` and `ports`, each port a dictionary of `device`, `listen` and `peer`); NULL, or why it is refused */
static int loop_number(CFDictionaryRef d,CFStringRef key,int64_t most,int64_t *value){
  CFTypeRef v=CFDictionaryGetValue(d,key);
  return v && CFGetTypeID(v)==CFNumberGetTypeID() && CFNumberGetValue(v,kCFNumberSInt64Type,value) && *value>0 && *value<=most;
}
static int loop_text(CFDictionaryRef d,CFStringRef key,char *out,size_t room){
  CFTypeRef v=CFDictionaryGetValue(d,key);
  return v && CFGetTypeID(v)==CFStringGetTypeID() && CFStringGetCString(v,out,(CFIndex)room,kCFStringEncodingUTF8) && out[0];
}
static const char *loop_read(CFPropertyListRef list){
  CFTypeRef l=list && CFGetTypeID(list)==CFDictionaryGetTypeID()?CFDictionaryGetValue(list,CFSTR("loop")):NULL;
  if(!l || CFGetTypeID(l)!=CFDictionaryGetTypeID())return "a loop bridge's configuration states its `loop`";
  int64_t node=0,service=0;
  if(!loop_text(l,CFSTR("region"),loop_config.region,sizeof loop_config.region) || loop_config.region[0]!='/')return "its loop's region, a shared memory name";
  if(!loop_text(l,CFSTR("fabric"),loop_config.fabric,sizeof loop_config.fabric) || loop_config.fabric[0]!='/')return "its loop's fabric, a shared memory name";
  if(!loop_number(l,CFSTR("node"),0x7fffffff,&node))return "its loop's node, a positive 31-bit number";
  if(!loop_number(l,CFSTR("service"),65535,&service))return "its loop's service, a port";
  loop_config.node=(uint32_t)node;loop_config.service=(uint16_t)service;
  CFTypeRef ports=CFDictionaryGetValue(l,CFSTR("ports"));
  if(!ports || CFGetTypeID(ports)!=CFArrayGetTypeID() || CFArrayGetCount(ports)>MESH_PORTS)return "its loop's ports, at most MESH_PORTS";
  loop_config.ports=(uint32_t)CFArrayGetCount(ports);
  for(uint32_t k=0;k<loop_config.ports;k++){
    CFTypeRef p=CFArrayGetValueAtIndex(ports,k);
    int64_t listen=0,peer=0;
    if(!p || CFGetTypeID(p)!=CFDictionaryGetTypeID() || !loop_text(p,CFSTR("device"),loop_config.port[k].device,sizeof loop_config.port[k].device) ||
       strncmp(loop_config.port[k].device,"rdma_",5) || !loop_number(p,CFSTR("listen"),65535,&listen) || !loop_number(p,CFSTR("peer"),65535,&peer) ||
       listen==peer)return "a loop port: its device (rdma_...) and its cable's two ports, apart";
    loop_config.port[k].listen=(uint16_t)listen;loop_config.port[k].peer=(uint16_t)peer;loop_config.port[k].fd=-1;
  }
  return NULL;
}
#endif
static const char *configuration_read(const char *path,char *name,size_t room,uint64_t *window){
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
  CFTypeRef sized=list && CFGetTypeID(list)==CFDictionaryGetTypeID()?CFDictionaryGetValue(list,CFSTR("window")):NULL;
  CFTypeRef executes=list && CFGetTypeID(list)==CFDictionaryGetTypeID()?CFDictionaryGetValue(list,CFSTR("exec")):NULL;
  exec_served=executes && CFGetTypeID(executes)==CFBooleanGetTypeID() && CFBooleanGetValue(executes);
  /* design/bridge-and-ipc.md#ledger */
  CFTypeRef kept=list && CFGetTypeID(list)==CFDictionaryGetTypeID()?CFDictionaryGetValue(list,CFSTR("ledger")):NULL;
  ledger_kept=kept && CFGetTypeID(kept)==CFBooleanGetTypeID() && CFBooleanGetValue(kept);
  /* design/bridge-and-ipc.md#keepalive */
  CFTypeRef alive=list && CFGetTypeID(list)==CFDictionaryGetTypeID()?CFDictionaryGetValue(list,CFSTR("keepalive")):NULL;
  int64_t value=0,interval=0;
  if(!list || CFGetTypeID(list)!=CFDictionaryGetTypeID())refused="not a property list of a configuration (rdma/bridge_configuration.py writes it)";
  else if(!named || CFGetTypeID(named)!=CFStringGetTypeID() || !CFStringGetCString(named,name,(CFIndex)room,kCFStringEncodingUTF8))refused="its name is missing";
  else if(!sized || CFGetTypeID(sized)!=CFNumberGetTypeID() || !CFNumberGetValue(sized,kCFNumberSInt64Type,&value) || value<=0)
    refused="its window, a positive number of bytes, is missing";
  else if(alive && (CFGetTypeID(alive)!=CFNumberGetTypeID() || !CFNumberGetValue(alive,kCFNumberSInt64Type,&interval) || interval<=0))
    refused="its keepalive, where stated, is not a positive number of nanoseconds";
#ifdef MESH_LOOP_FABRIC
  if(!refused)refused=loop_read(list);
#endif
  keepalive_ns=refused?0:(uint64_t)interval;
  if(list)CFRelease(list);
  *window=(uint64_t)value;
  return refused;
}

/* design/bridge-and-ipc.md#ports: the node's name, its LocalHostName (else its host name), read at the start and, on
   each liveness turn, on the finder's queue (configd answers there, never on the event loop: host_read); the event loop
   takes a change (host_taken), so a renamed node's next record carries its new name (the map's label of it; a node is
   its number) */
static void host_name(char *name){
  CFStringRef local=SCDynamicStoreCopyLocalHostName(NULL);
  if(!local || !CFStringGetCString(local,name,MESH_NAME_BYTES,kCFStringEncodingUTF8))gethostname(name,MESH_NAME_BYTES);
  if(local)CFRelease(local);
  name[MESH_NAME_BYTES-1]=0;
}
static char host_heard[MESH_NAME_BYTES];
static _Atomic int host_reading,host_read_done;
static void host_read(void){
#ifdef MESH_LOOP_FABRIC
  /* design/bridge-dataflow.md#the-loopback-fabric: a loop bridge's name is its configuration's, never the host's */
  return;
#endif
  if(atomic_load(&host_read_done) || atomic_exchange(&host_reading,1))return;
  dispatch_async(finder,^{
    char name[MESH_NAME_BYTES]={0};
    host_name(name);
    memcpy(host_heard,name,sizeof name);
    atomic_store_explicit(&host_read_done,1,memory_order_release);
    atomic_store(&host_reading,0);
    wake();
  });
}
static void host_taken(void){
  if(!atomic_load_explicit(&host_read_done,memory_order_acquire))return;
  if(strcmp(host_heard,host)){
    say("node %u: renamed %s to %s\n",M->node,host,host_heard);
    memcpy(host,host_heard,sizeof host);
    record_changed();
  }
  atomic_store_explicit(&host_read_done,0,memory_order_release);
}

/* design/bridge-and-ipc.md: the node's resident bridge; its one operand the path of its configuration (no default) */
int main(int argc,char **argv){
  if(argc!=2)die("usage: mesh-flow CONFIGURATION: the canonical property list bin/mesh-user.sh writes from the node's bridge configuration document (etc/bridge/README.md); the bridge has no default and does not start without one");
  char name[256];uint64_t window;
  const char *refused=configuration_read(argv[1],name,sizeof name,&window);
  if(refused){say("bridge configuration %s refused: %s\n",argv[1],refused);exit(1);}
  struct stat binary,current;
  if(stat(argv[0],&binary))memset(&binary,0,sizeof binary);
  uuid_t uuid;struct timespec bound={5,0};
  if(gethostuuid(uuid,&bound)){say("host identity: %s; this bridge's node is drawn at random\n",strerror(errno));arc4random_buf(uuid,sizeof uuid);}
  uint32_t node=0;
  for(int i=0;i<16;i++)node=(node*0x01000193u)^uuid[i];
  node=(node&0x7fffffffu)?node&0x7fffffffu:1;
  /* design/bridge-dataflow.md#the-loopback-fabric: a loop bridge's node, region and service its configuration's */
#ifdef MESH_LOOP_FABRIC
  node=loop_config.node;
  const char *region=loop_config.region;
  const uint16_t service=loop_config.service;
  loop_fabric(loop_config.fabric);
#else
  const char *region=MESH_NAME;
  const uint16_t service=(uint16_t)atoi(MESH_PORT);
#endif
  const uint32_t pg=(uint32_t)getpagesize();
  /* design/collective-dependency-ledger.md#d6-paired-send-and-receive-frame-counts-match */
  if((uint64_t)pg+4096>16773120)die("transport chunk exceeds native request capacity");
  if(window<pg || window/pg>UINT32_MAX){say("bridge configuration %s refused: its window of %llu bytes is not between one page and 2^32 pages\n",name,(unsigned long long)window);exit(1);}
  struct hdr geometry={0};
  uint64_t length=mesh_layout(&geometry,pg,1,(uint32_t)(window/pg));
  signal(SIGPIPE,SIG_IGN);signal(SIGINT,SIG_IGN);signal(SIGTERM,SIG_IGN);signal(SIGHUP,SIG_IGN);
  atexit(down);
  shm_unlink(region);int fd=shm_open(region,O_CREAT|O_RDWR,MESH_MODE);if(fd<0)die("shm");
  if(ftruncate(fd,(off_t)length))die("ftruncate");fchmod(fd,MESH_MODE);
  M=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  if(M==MAP_FAILED)die("mmap");shm=region;
  /* a boot is the bridge's start on the wall clock, so a node's later boot is a newer record */
  *M=geometry;M->node=node;M->version=MESH_VERSION;M->boot=clock_gettime_nsec_np(CLOCK_REALTIME);M->idle_since=M->boot;
  for(uint32_t r=0;r<mesh_rows(M);r++)atomic_store_explicit(&mesh_page(M)[r].mapping,MESH_ABSENT,memory_order_relaxed);
  if(wire_map(&wire,M,fd))die("transport page aliases");
  close(fd);
  bridge.M=M;
  host_name(host);
#ifdef MESH_LOOP_FABRIC
  snprintf(host,sizeof host,"%s",name);
#endif
  finder=dispatch_queue_create("mesh.ports",DISPATCH_QUEUE_SERIAL);
  /* design/bridge-and-ipc.md#records: the node's memory and hardware model, which its record carries with its window */
  size_t size=sizeof memory;
  if(sysctlbyname("hw.memsize",&memory,&size,NULL,0))memory=0;
  sysctl_text("hw.model",model,sizeof model);
  events=kqueue();
  health_init();
  record_changed();
  if(!atomic_load(&M->map_nodes))die("the map's lease");
  link_reserve();
  struct kevent64_s event;
  int on=1;
  struct sockaddr_in6 any={.sin6_len=sizeof any,.sin6_family=AF_INET6,.sin6_port=htons(service),.sin6_addr=IN6ADDR_ANY_INIT};
#ifdef MESH_LOOP_FABRIC
  any.sin6_addr=in6addr_loopback;
#endif
  listener=socket(AF_INET6,SOCK_STREAM,0);hello=socket(AF_INET6,SOCK_DGRAM,0);network=socket(PF_SYSTEM,SOCK_RAW,SYSPROTO_EVENT);
  struct kev_request filter={KEV_VENDOR_APPLE,KEV_NETWORK_CLASS,KEV_DL_SUBCLASS};
  unsigned loop=0;
  if(events<0 || listener<0 || hello<0 || network<0 || setsockopt(listener,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on) ||
     setsockopt(hello,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on) || setsockopt(hello,IPPROTO_IPV6,IPV6_MULTICAST_LOOP,&loop,sizeof loop) ||
     bind(listener,(struct sockaddr *)&any,sizeof any) || listen(listener,SOMAXCONN) || bind(hello,(struct sockaddr *)&any,sizeof any) ||
     fcntl(listener,F_SETFL,O_NONBLOCK) || fcntl(hello,F_SETFL,O_NONBLOCK) || fcntl(network,F_SETFL,O_NONBLOCK) || ioctl(network,SIOCSKEVFILT,&filter))
    die("bridge sockets");
  int watched[3]={listener,hello,network};
  for(int i=0;i<3;i++){EV_SET64(&event,(uint64_t)watched[i],EVFILT_READ,EV_ADD,0,0,UINT64_MAX,0,0);if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");}
  EV_SET64(&event,0,EVFILT_USER,EV_ADD|EV_CLEAR,0,0,UINT64_MAX,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");
  int signals[3]={SIGINT,SIGTERM,SIGHUP};
  for(int i=0;i<3;i++){EV_SET64(&event,(uint64_t)signals[i],EVFILT_SIGNAL,EV_ADD,0,0,UINT64_MAX,0,0);kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);}
  atomic_store(&M->bridge_pid,(uint64_t)getpid());
  __sync_synchronize();M->magic=MESH_MAGIC;
  requests_init();
  pthread_t watcher;
  if(pthread_create(&watcher,NULL,control_watch,NULL))die("control watcher");
#ifdef MESH_LOOP_FABRIC
  /* design/bridge-dataflow.md#the-loopback-fabric: each cable's listener on loopback, its events tagged by the port */
  for(uint32_t k=0;k<loop_config.ports;k++){
    struct sockaddr_in6 at={.sin6_len=sizeof at,.sin6_family=AF_INET6,.sin6_port=htons(loop_config.port[k].listen),.sin6_addr=IN6ADDR_LOOPBACK_INIT};
    int fd=socket(AF_INET6,SOCK_STREAM,0);
    if(fd<0 || setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on) || bind(fd,(struct sockaddr *)&at,sizeof at) || listen(fd,SOMAXCONN) ||
       fcntl(fd,F_SETFL,O_NONBLOCK))die("a loop cable's listener");
    loop_config.port[k].fd=fd;
    EV_SET64(&event,(uint64_t)fd,EVFILT_READ,EV_ADD,0,0,TAG_LOOP|k,0,0);
    if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");
  }
#endif
  ports_find();
  char alive[128]="";
  if(keepalive_ns)snprintf(alive,sizeof alive,", a keepalive frame every %llu ns on each link's keepalive lane (design/bridge-and-ipc.md#keepalive)",
                           (unsigned long long)keepalive_ns);
  say("bridge node %u (%s): configuration %s, window %llu bytes, arena %llu pages, rows %u%s%s\n",node,host,name,
    (unsigned long long)window,(unsigned long long)mesh_arena_pages(M),M->rows,ledger_kept?", ledgers kept (design/bridge-and-ipc.md#ledger)":"",
    alive);
  uint64_t liveness=0;
  while(!stop){
    now=clock_ns();
    if(atomic_load_explicit(&found,memory_order_acquire))ports_found();
    for(uint32_t p=0;p<port_count;p++){
      struct port *port=ports+p;
      if(port->job && atomic_load_explicit(&port->done,memory_order_acquire))port_opened(port);
      port_open(port);
      port_hello(port);
      link_turn(port);
      streams_turn(port);
    }
    flows_turn();
    stripes_requests_turn();
    requests_turn();
    health_turn();
    host_taken();
    int alive=now>=liveness;
    if(alive){
      liveness=now+MESH_PAUSE_NS;
      host_read();
      /* design/bridge-and-ipc.md#stop: a bridge whose binary changed stops at once (0fb8eb54's rule), its programs failing
         as values, and launchd starts the new one */
      if(binary.st_ino && (stat(argv[0],&current) || current.st_ino!=binary.st_ino || current.st_mtimespec.tv_sec!=binary.st_mtimespec.tv_sec)){
        stop=1;stop_reason=MESH_STOP_BINARY;
      }
    }
    clients_turn(alive);
    struct kevent64_s ready[16];
    struct timespec tick={0,250000000};
    int count=kevent64(events,NULL,0,ready,16,0,&tick);
    now=clock_ns();
    for(int i=0;i<count;i++){
      struct kevent64_s *e=ready+i;
      if(e->filter==EVFILT_SIGNAL){stop=1;stop_reason=(int64_t)e->ident;}
      else if(e->filter==EVFILT_PROC){
        child_heard((pid_t)e->ident);
        for(uint32_t s=0;s<MESH_CLIENTS;s++){
          uint64_t owner=atomic_load(&((struct mesh_client *)((char *)M+M->client_off))[s].owner);
          if(owner && (uint32_t)owner==(uint32_t)e->ident)mesh_retire(M,owner);
        }
      }
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)listener){
        struct sockaddr_in6 from;socklen_t size=sizeof from;int accepted;
        while((accepted=accept(listener,(struct sockaddr *)&from,&size))>=0){
          /* #requests: a connection from this node is a client's request */
          if(IN6_IS_ADDR_LOOPBACK(&from.sin6_addr)){size=sizeof from;asking_accepted(accepted);continue;}
          struct port *port=port_at(from.sin6_scope_id);size=sizeof from;
          if(!port || port->control>=0 || !port->open){close(accepted);continue;}
          port->control=accepted;
          port_greet(port);
        }
      }
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)hello)hello_heard();
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)network)network_heard();
#ifdef MESH_LOOP_FABRIC
      /* design/bridge-dataflow.md#the-loopback-fabric: a cable's lower end dialed: its port's control socket, as a link's
         listener takes the lower node's dial */
      else if(e->udata!=UINT64_MAX && (e->udata&TAG_LOOP)){
        const uint32_t k=(uint32_t)(e->udata&0xff);
        int accepted;
        while(k<loop_config.ports && (accepted=accept(loop_config.port[k].fd,NULL,NULL))>=0){
          struct port *port=NULL;
          for(uint32_t p=0;p<port_count;p++)if(!strcmp(ports[p].device.name,loop_config.port[k].device))port=ports+p;
          if(!port || port->control>=0 || !port->open){close(accepted);continue;}
          port->control=accepted;port->peer=0;
          port_greet(port);
        }
      }
#endif
      else if(e->udata!=UINT64_MAX && (e->udata&(TAG_CHANNEL|TAG_ASKING|TAG_SERVING)))request_event(e);
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
  /* design/bridge-and-ipc.md#stop: FAREWELL on every control socket (queued as the socket takes it; the kernel sends
     what it holds after the close), every program's words ~0 and its status the stop, stored from this thread, then the
     bridge gone; no queue, thread or verbs call waited on, and every object the kernel's to release at the exit
     (design/RDMA-KERNEL-RECOVERY.md "October 18") */
  requests_stop();
  for(uint32_t p=0;p<port_count;p++){
    port_say(ports+p,(struct message){.kind=FAREWELL,.code=stop_reason},NULL,0);
    if(ports[p].control>=0){close(ports[p].control);ports[p].control=-1;}
    atomic_store_explicit(&ports[p].closing,1,memory_order_release);
  }
  programs_stop(ESHUTDOWN);
  atomic_store(&M->bridge_pid,0);
  mesh_notify(&M->control);
  return 0;
}
