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
/* design/bridge-and-ipc.md#keepalive: the configuration's `keepalive`, the most nanoseconds an exclusive flow's send side
   lets pass between frames on its keepalive queue pair while it progresses; 0 where it states none (its flows make no
   such pair) */
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

/* design/bridge-and-ipc.md#lanes: a port's lanes' pages: lane l's header frame h (its MESH_HEADERS send headers, then
   its two receive headers) and its discard block, as arena offsets; a span of them on the port's device; and their
   bytes through the window's mapping */
static uint64_t lane_offset(const struct port *port,uint32_t lane,uint32_t frame){
  return (uint64_t)port->lanes.page*M->pgsz+((uint64_t)lane*(MESH_HEADERS+2)+frame)*LANE_FRAME;
}
uint64_t lane_discard(const struct port *port,uint32_t lane){
  const uint64_t block=(uint64_t)M->block*M->pgsz,headers=(uint64_t)MESH_LANES*(MESH_HEADERS+2)*LANE_FRAME;
  return (uint64_t)port->lanes.page*M->pgsz+(headers+block-1)/block*block+(uint64_t)lane*block;
}
static uint32_t lane_pages(void){
  const uint64_t block=(uint64_t)M->block*M->pgsz,headers=(uint64_t)MESH_LANES*(MESH_HEADERS+2)*LANE_FRAME;
  return (uint32_t)(((headers+block-1)/block*block+MESH_LANES*block+M->pgsz-1)/M->pgsz);
}
struct ibv_sge lane_span(const struct port *port,uint64_t offset,uint64_t length){
  return wire_span(&port->device,(uint64_t)M->block*M->pgsz,offset,length);
}
struct ibv_qp *lane_pair(const struct flow *link,uint32_t lane){
  return link->port->lanes.provider.queues?link->port->lanes.provider.queues[lane].pair:NULL;
}
/* a lane's failure, the lanes' (taken by the event loop: lanes_turn, lanes_reset) */
static void lanes_lost(struct port *port,int64_t code){
  int64_t none=0;
  atomic_compare_exchange_strong(&port->lanes.failed,&none,code?code:EIO);
  atomic_store_explicit(&port->lanes.up,0,memory_order_release);
  wake();
}
/* lane l's next header received, into its receive header of the parity of the headers it heard */
static int lane_listen(struct port *port,uint32_t lane){
  struct lanes *l=&port->lanes;
  struct ibv_sge span=lane_span(port,lane_offset(port,lane,MESH_HEADERS+(uint32_t)(l->received[lane]&1)),LANE_FRAME);
  struct ibv_recv_wr request={.wr_id=LANE_HEARD+lane,.sg_list=&span,.num_sge=1},*bad;
  const int error=l->provider.queues[lane].receive(l->provider.queues[lane].pair,&request,&bad);
  return error<0?-error:error;
}

/* design/bridge-and-ipc.md#lanes: a lane's SENDs, one pass of its port's send thread: of the flows on the port's lanes
   that run, each stream of this lane whose cell its producer released gives its chunks, as many as the lane's gate and
   the peer's receive room take (a cell's chunks may span headers), each run of them an entry of one header; the header
   is posted, then every entry's chunks in the header's order, so the peer, which posts the records a header names
   before it listens for the next, has each SEND's receive posted as it lands (credit flow control holds it until then,
   TN3205).  A header frame is used again once its SEND retired.  Whether anything was posted; a failure is the
   lanes' (lanes_lost). */
static int lane_send(struct port *port,uint32_t lane){
  struct lanes *l=&port->lanes;
  struct send_gate *gate=l->gate+lane;
  const uint64_t sent=l->sent[lane];
  const uint32_t h=(uint32_t)(sent%MESH_HEADERS);
  if(sent>=MESH_HEADERS && ((gate->retired-l->mark[lane][h])&MESH_GATE_MASK)>=(UINT64_C(1)<<47))return 0;
  const uint64_t flight=(gate->posted-gate->retired)&MESH_GATE_MASK;
  if(flight+2>gate->capacity)return 0;
  uint64_t room=MIN(gate->capacity-flight-1,(uint64_t)l->peer_room);
  struct lane_header *header=(void *)(wire.data+lane_offset(port,lane,h));
  struct ibv_send_wr *first[LANE_ENTRIES];
  uint32_t entries=0,tracing=0;
  /* #ledger: the entries whose flows keep ledgers, each its event and whether it ends its cell */
  struct ledger_entry traced[LANE_ENTRIES];
  const uint32_t high=atomic_load_explicit(&port->high,memory_order_acquire);
  for(uint32_t n=0;n<high && entries<LANE_ENTRIES && room;n++){
    const uint32_t i=(l->cursor[lane]+n)%high;
    struct flow *f=atomic_load_explicit(&port->table[i],memory_order_acquire);
    if(!f || f->exclusive || !atomic_load_explicit(&f->progressing,memory_order_acquire))continue;
    for(uint32_t s=0;s<f->stream_count && entries<LANE_ENTRIES && room;s++){
      struct send_stream *stream=f->streams+s;
      while(stream->lane==lane && entries<LANE_ENTRIES && room){
        if(!stream->remaining){
          struct mesh_send *cell=stream->cell;
          if(atomic_load_explicit(&cell->ready,memory_order_acquire)<=stream->cycle)break;
          stream->next=&cell->request;stream->remaining=cell->chunks;
          if(f->ledger)ledger_seen(f->ledger,cell,stream->cycle);
        }
        struct ibv_send_wr *request=stream->next;
        uint32_t chunks=0;uint64_t bytes=0;
        for(;chunks<stream->remaining && request;chunks++,request=request->next){
          const uint32_t frames=(request->sg_list->length+4095)/4096;
          if(frames>room)break;
          room-=frames;bytes+=request->sg_list->length;
        }
        if(!chunks){room=0;break;}
        header->entry[entries]=(struct lane_entry){f->peer_slot,f->peer_generation,stream->cell->queue,chunks,bytes};
        first[entries++]=stream->next;
        stream->next=request;stream->remaining-=chunks;
        if(f->ledger)traced[tracing++]=(struct ledger_entry){f->ledger,f->ledger->current[stream->cell->queue],!stream->remaining};
        if(stream->remaining)break;
        stream->cell=(struct mesh_send *)stream->cell->successor;
        if(stream->cell==stream->first)stream->cycle++;
      }
    }
  }
  if(!entries)return 0;
  l->cursor[lane]=(l->cursor[lane]+1)%(high?high:1);
  header->magic=LANE_MAGIC;header->count=entries;header->sequence=sent;
  int (*post)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **)=l->provider.queues[lane].send;
  struct ibv_qp *pair=l->provider.queues[lane].pair;
  struct ibv_sge span=lane_span(port,lane_offset(port,lane,h),LANE_FRAME);
  struct ibv_send_wr posting={.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
  gate->posted+=1;gate->requests++;
  posting.wr_id=LANE_SENT|((uint64_t)lane<<48)|(gate->posted&MESH_GATE_MASK);
  int error=post(pair,&posting,&bad);
  l->mark[lane][h]=gate->posted;l->sent[lane]++;
  for(uint32_t e=0;e<entries && !error;e++){
    struct ibv_send_wr *request=first[e];
    for(uint32_t c=0;c<header->entry[e].chunks && !error;c++,request=request->next){
      posting=*request;
      posting.next=NULL;posting.send_flags=IBV_SEND_SIGNALED;
      gate->posted+=(request->sg_list->length+4095)/4096;gate->requests++;
      posting.wr_id=LANE_SENT|((uint64_t)lane<<48)|(gate->posted&MESH_GATE_MASK);
      error=post(pair,&posting,&bad);
    }
  }
  if(error)lanes_lost(port,error<0?-error:error);
  else if(tracing){
    const uint64_t at=ledger_ns();
    for(uint32_t e=0;e<tracing;e++)ledger_stamp(traced[e].ledger,traced[e].event,(int)traced[e].done,0,at);
  }
  return 1;
}
/* the lanes' SEND completions, each retiring its lane's frames; whether any was taken */
static int lane_retire(struct port *port){
  struct lanes *l=&port->lanes;
  struct ibv_wc done[16];
  const int count=l->provider.queues[0].poll(l->provider.sent,16,done);
  if(count<0){lanes_lost(port,EIO);return 1;}
  for(int i=0;i<count;i++){
    if(done[i].status){lanes_lost(port,(int64_t)done[i].status);return 1;}
    struct send_gate *gate=l->gate+((done[i].wr_id>>48)&(MESH_LANES-1));
    gate->retired=done[i].wr_id&MESH_GATE_MASK;gate->completions++;
  }
  return count>0;
}
/* design/bridge-and-ipc.md#lanes: lane l's header heard: each entry's records posted, the next `chunks` of the named
   ring of the named flow while that flow is the one the index and generation name and takes its lanes' records, else
   as many discards of the same frames (a flow that stopped: its rows are no longer its own), then the next header
   listened for.  A header out of sequence or past the room the lane told, or an entry whose frames are not its ring's
   records', is a protocol error: the lanes' where the header is, the flow's where its records are.  1 where the lanes
   failed */
static int lane_heard(struct port *port,uint32_t lane){
  struct lanes *l=&port->lanes;
  const struct lane_header *header=(const void *)(wire.data+lane_offset(port,lane,MESH_HEADERS+(uint32_t)(l->received[lane]&1)));
  const uint64_t block=(uint64_t)M->block*M->pgsz;
  uint64_t total=0;
  int error=header->magic!=LANE_MAGIC || header->sequence!=l->received[lane] || header->count>LANE_ENTRIES?EPROTO:0;
  for(uint32_t e=0;!error && e<header->count;e++){
    const struct lane_entry *entry=header->entry+e;
    if(!entry->chunks || entry->bytes<=(uint64_t)(entry->chunks-1)*block || entry->bytes>(uint64_t)entry->chunks*block)error=EPROTO;
    else total+=(uint64_t)(entry->chunks-1)*(block/4096)+(entry->bytes-(uint64_t)(entry->chunks-1)*block+4095)/4096;
  }
  if(!error && total>l->room)error=EPROTO;
  if(error){lanes_lost(port,error);return 1;}
  int (*post)(struct ibv_qp *,struct ibv_recv_wr *,struct ibv_recv_wr **)=l->provider.queues[lane].receive;
  struct ibv_qp *pair=l->provider.queues[lane].pair;
  struct ibv_recv_wr *bad;
  for(uint32_t e=0;e<header->count && !error;e++){
    const struct lane_entry entry=header->entry[e];
    struct flow *f=entry.slot<MESH_FLOWS?atomic_load_explicit(&port->table[entry.slot],memory_order_acquire):NULL;
    struct receive_ring *ring=f && !f->exclusive && f->generation==entry.generation && entry.ring<(uint32_t)f->qps &&
      atomic_load_explicit(&f->receiving,memory_order_acquire) && f->rings && ring_lane(entry.ring,(uint32_t)f->qps/f->queues)==lane?
      f->rings+entry.ring:NULL;
    for(uint32_t c=0;ring && c<entry.chunks;c++){
      const uint64_t length=MIN(block,entry.bytes-(uint64_t)c*block);
      if(!ring->count || ring->first[(ring->posted+c)%ring->count].frames!=(length+4095)/4096){flow_lost(f,EPROTO,2);ring=NULL;}
    }
    for(uint32_t c=0;c<entry.chunks && !error;c++){
      if(ring){
        struct prepared_receive *record=ring->first+ring->posted%ring->count;
        record->cycle=(uint32_t)(ring->posted/ring->count);
        error=post(pair,&record->request,&bad);
        ring->outstanding+=record->frames;ring->posted++;
        atomic_fetch_add_explicit(&f->lane_frames,record->frames,memory_order_relaxed);
      }else{
        struct ibv_sge span=lane_span(port,lane_discard(port,lane),MIN(block,entry.bytes-(uint64_t)c*block));
        struct ibv_recv_wr request={.wr_id=LANE_DISCARD,.sg_list=&span,.num_sge=1};
        error=post(pair,&request,&bad);
      }
      if(!error)l->records++;
    }
  }
  if(!error){l->received[lane]++;error=lane_listen(port,lane);}
  if(error){lanes_lost(port,error<0?-error:error);return 1;}
  return 0;
}
/* design/bridge-and-ipc.md#lanes: the lanes' receive completions, one pass of the port's receive thread: a header's
   records posted (lane_heard), a record's landing storing its word as an exclusive flow's does; whether anything
   landed */
static int lane_receive(struct port *port){
  struct lanes *l=&port->lanes;
  struct ibv_wc done[16];
  const int count=l->provider.queues[0].poll(l->provider.completion,16,done);
  if(count<0){lanes_lost(port,EIO);return 1;}
  for(int i=0;i<count;i++){
    if(done[i].status){lanes_lost(port,(int64_t)done[i].status);return 1;}
    const uint64_t id=done[i].wr_id;
    if(id>=LANE_HEARD && id<LANE_HEARD+MESH_LANES){if(lane_heard(port,(uint32_t)(id-LANE_HEARD)))return 1;continue;}
    l->records--;
    if(id==LANE_DISCARD)continue;
    struct prepared_receive *record=(void *)(uintptr_t)id;
    struct flow *link=record->link;
    struct receive_ring *ring=link->rings+record->queue;
    ring->landed++;ring->bytes+=done[i].byte_len;
    atomic_store_explicit(record->input,record->argument+record->cycle,memory_order_release);
    if(record->release)send_wake(record->input);
    /* #ledger: stored before the frames are given back, which flow_release reads before it writes the ledger */
    if(link->ledger)ledger_landed(link,record);
    ring->outstanding-=record->frames;ring->completed=(int64_t)(record->invocation+(uint64_t)record->cycle*link->invocations);
    atomic_fetch_sub_explicit(&link->lane_frames,record->frames,memory_order_release);
  }
  return count>0;
}

/* design/bridge-and-ipc.md#progress: one pass of a port's progress thread, `side` 0 its receives' and 1 its SENDs':
   every exclusive flow in the port's table it serves (while the flow is progressing) and the port's lanes while they are
   up.  The pass reads the table's epoch, serves each, and tells the epoch it began at, so a flow taken out of the table
   is read by neither thread once both tell a later epoch (port_unlist).  Whether anything moved; *any where something is
   the thread's to look for: a flow it serves (a SEND side progressing; an exclusive flow's receives, or a lane rider
   taking its headers' records), a lane's records posted and not landed, or its SENDs in flight; nothing lands on a lane
   for a flow this end has not listed, since the peer sends only after this end's POSTED, which follows the listing */
static int port_pass(struct port *port,int side,int *any,uint64_t *told){
  const uint64_t epoch=atomic_load(&port->epoch);
  int moved=0;
  *any=0;
  const uint32_t high=atomic_load_explicit(&port->high,memory_order_acquire);
  int riders=0;
  for(uint32_t i=0;i<high;i++){
    struct flow *f=atomic_load_explicit(&port->table[i],memory_order_acquire);
    if(!f)continue;
    const int progressing=atomic_load_explicit(&f->progressing,memory_order_acquire);
    if(f->exclusive && progressing){*any=1;moved|=side?flow_send(f):flow_receive(f);}
    else if(!f->exclusive){riders=1;*any|=side?progressing:atomic_load_explicit(&f->receiving,memory_order_acquire);}
  }
  /* #lanes: served while a flow on the port rides them or their SENDs or records are in flight (nothing lands on a lane
     for a flow this end has not listed) */
  struct lanes *l=&port->lanes;
  if(atomic_load_explicit(&l->up,memory_order_acquire)){
    if(side){
      int flight=0;
      for(uint32_t lane=0;lane<MESH_LANES;lane++)flight|=((l->gate[lane].posted-l->gate[lane].retired)&MESH_GATE_MASK)!=0;
      if(riders || flight){
        for(uint32_t lane=0;lane<MESH_LANES;lane++)moved|=lane_send(port,lane);
        moved|=lane_retire(port);
        flight=0;
        for(uint32_t lane=0;lane<MESH_LANES;lane++)flight|=((l->gate[lane].posted-l->gate[lane].retired)&MESH_GATE_MASK)!=0;
      }
      *any|=flight;
    }else if(riders || l->records){moved|=lane_receive(port);*any|=l->records!=0;}
  }
  /* design/bridge-dataflow.md#stripes: the port's stripe flow while it is up, its receives always posted and any ring
     able to name its peer at any time */
  if(atomic_load_explicit(&port->stripes.up,memory_order_acquire)){*any=1;moved|=side?stripes_send(port):stripes_receive(port);}
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
/* design/bridge-and-ipc.md#progress: a flow in its port's table at its index, on the port's queue, its SENDs (and an
   exclusive flow's receives) served where `progressing`, a lane's headers filling its rings where `receiving`; 0, or
   the threads' error */
int port_list(struct flow *f,int progressing,int receiving){
  struct port *port=f->port;
  const int error=port_threads(port);
  if(error)return error;
  if(progressing)atomic_store_explicit(&f->progressing,1,memory_order_release);
  if(receiving)atomic_store_explicit(&f->receiving,1,memory_order_release);
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
   to the device's shared doorbell mapping after it moved, tbt_post_recv); lowered, the threads go on.  It stands for a
   change of the stripe flow's own (`always`) and, while the device holds a stripe flow's queue pairs (whose posts
   outlive every other's teardown), for every flow's and the lanes' too; with none, HEAD's verbs work runs unguarded as
   it did */
void port_guard(struct port *port,int raise){
  if(!raise){atomic_fetch_add_explicit(&port->guard,1,memory_order_release);port_bell(port);return;}
  const uint64_t guard=atomic_fetch_add_explicit(&port->guard,1,memory_order_acq_rel)+1;
  for(int side=0;side<port->threads;side++)
    while(atomic_load_explicit(&port->guarded[side],memory_order_acquire)!=guard){port_bell(port);sched_yield();}
}
int guard_raise(struct port *port,int always){
  if(!always && !port->stripes.provider.queues && !port->stream.provider.queues)return 0;
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
  atomic_store_explicit(&f->receiving,0,memory_order_release);
  if(!f->listed)return;
  struct flow *held=f;
  atomic_compare_exchange_strong(&f->port->table[f->slot],&held,NULL);
  port_settle(f->port);
  f->listed=0;
}
/* design/bridge-and-ipc.md#progress: the port's progress threads stopped (the bridge stopping, every flow released) */
static void port_still(struct port *port){
  atomic_store_explicit(&port->closing,1,memory_order_release);
  port_bell(port);
  while(port->threads)pthread_join(port->thread[--port->threads],NULL);
}
/* design/bridge-and-ipc.md#region: the queue pairs a port's device holds, told in its port table */
void port_pairs(struct port *port){
  atomic_store_explicit(&mesh_links(M)[port->index].pairs,port->device.pairs,memory_order_relaxed);
}

/* design/bridge-and-ipc.md#protocol: a message to the port's peer, queued behind what the socket has not taken yet; a
   send that fails is the link's loss */
static void port_watch(struct port *port,int filter){
  struct kevent64_s event;EV_SET64(&event,(uint64_t)port->control,filter,EV_ADD|(filter==EVFILT_WRITE?EV_ONESHOT:0),0,0,port->index,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))port_lost(port,errno,0);
}
static void port_flush(struct port *port){
  while(port->control>=0 && port->out_sent<port->out_length){
    ssize_t sent=send(port->control,port->out+port->out_sent,port->out_length-port->out_sent,0);
    if(sent<0 && (errno==EAGAIN || errno==EINTR)){
      if(!port->writing){port->writing=1;port_watch(port,EVFILT_WRITE);}
      return;
    }
    if(sent<=0){port_lost(port,sent?errno:EIO,0);return;}
    port->out_sent+=(size_t)sent;
  }
  port->out_sent=port->out_length=0;
}
void port_say(struct port *port,struct message said,const void *payload,uint32_t bytes){
  if(port->control<0 || (said.kind!=HELLO && !port->greeted))return;
  said.magic=XMAGIC;said.node=M->node;said.boot=M->boot;said.block=M->block*M->pgsz;said.port=port->index;said.bytes=bytes;
  said.lid=port->lid;memcpy(said.gid,port->gid,16);
  size_t length=port->out_length+sizeof said+bytes;
  if(length-port->out_sent>MESH_UNSENT){port_lost(port,ENOBUFS,0);return;}
  if(length>port->out_room){
    unsigned char *grown=realloc(port->out,length*2);
    if(!grown){port_lost(port,ENOMEM,0);return;}
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

/* design/bridge-and-ipc.md#lanes: a lanes job on the port's queue: their queue pairs and completion queues made
   (LANES_MAKING); ready to receive from the peer's and to send, each lane listening for its first header and the
   port's progress threads started (LANES_CONNECTING: the event loop alone makes them up as it takes the job, so lanes
   reset while they connected stay down); or down once neither progress thread reads them, their queue pairs destroyed
   and their generation moved (LANES_CLOSING) */
static void lanes_job(struct port *port,int job){
  struct lanes *l=&port->lanes;
  l->job=job;atomic_store_explicit(&l->done,0,memory_order_relaxed);
  dispatch_async(port->work,^{
    int result=0;
    /* design/bridge-dataflow.md#stripes: guarded where the device holds a stripe flow (port_guard) */
    const int raised=guard_raise(port,0);
    if(job==LANES_MAKING)result=pair_up(&l->provider,MESH_LANES);
    else if(job==LANES_CONNECTING){
      result=pair_receive(&l->provider,(const uint32_t (*)[2])l->peer,port->heard.lid,port->heard.gid);
      if(!result)result=pair_send(&l->provider);
      const uint32_t sends=(uint32_t)l->provider.sent->cqe/MESH_LANES;
      for(uint32_t lane=0;!result && lane<MESH_LANES;lane++){
        l->gate[lane]=(struct send_gate){.capacity=MIN(l->provider.queues[lane].send_capacity,sends)};
        l->sent[lane]=l->received[lane]=0;l->cursor[lane]=0;l->records=0;
        result=lane_listen(port,lane);
      }
      if(!result)result=port_threads(port);
    }else{
      atomic_store_explicit(&l->up,0,memory_order_release);
      port_settle(port);
      result=down_pair(&l->provider)?0:EBUSY;
      if(!result)l->generation++;
    }
    guard_lower(port,raised);
    l->result=result;
    port_pairs(port);
    atomic_store_explicit(&l->done,1,memory_order_release);
    wake();
  });
}
/* the frames of payload a lane's receive queue takes beside its header: its queue pair's and its share of the receive
   completion queue, less the header's frame */
static uint32_t lanes_room(const struct lanes *l){
  uint32_t room=(uint32_t)l->provider.completion->cqe/MESH_LANES;
  for(uint32_t lane=0;lane<MESH_LANES;lane++)room=MIN(room,l->provider.queues[lane].receive_capacity);
  return room>1?room-1:0;
}
/* the port's flows stepped (its lanes changed) */
static void port_step(struct port *port){
  for(struct flow *f=flows,*next;f;f=next){next=f->next;if(f->port==port)flow_step(f);}
}
/* design/recovery.md#loss: lanes that failed (made, connected or run) or whose peer's did are the lanes' alone: the
   flows that ride them and have posted on them fail (their records and SENDs went with them), the lanes are closed and
   made again a pause later, and the peer is told (a LANES of none and why) where the failure is this end's; the link
   stays paired, and its flows with queue pairs of their own and those still pairing run on */
static void lanes_reset(struct port *port,int64_t code,const char *what,int tell){
  struct lanes *l=&port->lanes;
  say("link %u (%s): lanes toward node %u %s: %s; made again in %.0f s, the link paired and its own queue pairs running on\n",
      port->index,port->name,port->peer,what,strerror((int)code),(double)MESH_PAUSE_NS/1e9);
  if(tell)port_say(port,(struct message){.kind=LANES,.count=0,.code=code},NULL,0);
  atomic_store_explicit(&l->up,0,memory_order_release);
  for(struct flow *f=flows;f;f=f->next)
    if(f->port==port && !f->exclusive && f->state!=FAILED && (f->state>=POSTING || f->job==JOB_RECEIVE || f->job==JOB_RUN || f->job==JOB_CHANGE))
      flow_fail(f,code,1,FAIL,M->node,M->node);
  if(l->state!=LANES_DOWN)l->state=LANES_CLOSING;
  l->heard=0;
}
/* design/bridge-and-ipc.md#lanes: the port's lanes' turn, on the event loop: a job's end taken (made: LANES told the
   peer, its queue pairs and its room; connected: up, and the port's flows stepped; closed: down, the device's queue
   pairs given back; a make or connect that failed: lanes_reset); a lane's failure is the lanes' (lanes_reset); the lanes
   made once the port pairs, or a pause after they failed, where the device has room for them (flow_make counts it
   before any flow's own; else made at the first turn it has, the cause logged once), connected once both ends made
   theirs, closed once the link went or they failed */
static void lanes_turn(struct port *port){
  struct lanes *l=&port->lanes;
  if(l->job){
    if(!atomic_load_explicit(&l->done,memory_order_acquire))return;
    const int job=l->job,result=l->result;
    l->job=0;
    if(job==LANES_CLOSING){
      if(result)say("link %u (%s): lanes teardown: %s\n",port->index,port->name,strerror(result));
      else{l->state=LANES_DOWN;if(l->committed)port->committed-=MESH_LANES;l->committed=0;}
    }else if(l->state==LANES_CLOSING){}
    else if(result){
      lanes_reset(port,result,job==LANES_MAKING?"not made":"not connected",1);
      /* design/recovery.md#loss: a device whose connections keep failing is opened again, as a new bridge would open it */
      if(job==LANES_CONNECTING && ++l->refused>=MESH_REFUSED_RETIRE){
        say("link %u (%s): lanes toward node %u not connected %u times in a row\n",port->index,
            port->name,port->peer,l->refused);
        l->refused=0;
        port_lost(port,result,0);
      }
    }
    else if(job==LANES_MAKING){
      l->state=LANES_MADE;l->room=lanes_room(l);
      struct message said={.kind=LANES,.count=MESH_LANES,.code=l->room};
      for(uint32_t lane=0;lane<MESH_LANES;lane++){said.pair[lane][0]=l->provider.queues[lane].pair->qp_num;said.pair[lane][1]=(l->provider.psn+lane)&0xffffff;}
      port_say(port,said,NULL,0);
    }else{
      l->state=LANES_UP;l->code=0;l->refused=0;
      atomic_store_explicit(&l->up,1,memory_order_release);port_bell(port);
      say("link %u (%s): lanes up toward node %u: %u queue pairs, %u frames a header\n",port->index,port->name,port->peer,MESH_LANES,
          MIN(l->room,l->peer_room));
      port_step(port);
    }
  }
  const int64_t failed=atomic_exchange(&l->failed,0);
  if(failed && port->greeted && (l->state==LANES_UP || l->state==LANES_CONNECTING))lanes_reset(port,failed,"failed",1);
  if(l->job)return;
  if(l->state==LANES_CLOSING && now>=l->retry){l->retry=now+MESH_PAUSE_NS;lanes_job(port,LANES_CLOSING);}
  else if(port->greeted && l->state==LANES_DOWN && now>=l->retry){
    if(l->page==MESH_ABSENT && (l->page=mesh_arena_alloc(&bridge,l->pages,1))==MESH_ABSENT){
      if(l->code!=ENOMEM)say("link %u (%s): lanes not made: the window holds no %u pages for them; made once it has\n",port->index,
                             port->name,l->pages);
      l->code=ENOMEM;l->retry=now+MESH_PAUSE_NS;
    }else if(port->committed+MESH_LANES>port->device.max_pairs){
      if(l->code!=ENOSPC)say("link %u (%s): lanes not made: the device holds %u of its %u usable queue pairs; made once it has room\n",
                             port->index,port->name,port->committed,port->device.max_pairs);
      l->code=ENOSPC;l->retry=now+MESH_PAUSE_NS;
    }else{port->committed+=MESH_LANES;l->committed=1;l->state=LANES_MAKING;lanes_job(port,LANES_MAKING);}
  }
  else if(port->greeted && l->state==LANES_MADE && l->heard){l->state=LANES_CONNECTING;lanes_job(port,LANES_CONNECTING);}
}
/* a peer's LANES: its lanes' queue pairs and the frames a header may name, or (count 0) its lanes failed, with why:
   lanes of this end that were connected, or connecting, to its old ones are reset (lanes_reset, untold: the peer knows),
   and lanes made and not yet connected wait for its new ones.  A peer's new lanes heard while this end's are connected
   to its old ones reset them alike, the new ones kept for the next connect */
static void lanes_heard(struct port *port,const struct message *heard){
  struct lanes *l=&port->lanes;
  const int joined=l->state==LANES_UP || l->state==LANES_CONNECTING;
  if(heard->count!=MESH_LANES){
    if(joined)lanes_reset(port,heard->code?heard->code:EPROTO,"failed at its peer",0);
    else{say("link %u (%s): node %u's lanes failed: %s\n",port->index,port->name,port->peer,strerror((int)heard->code));l->heard=0;}
    return;
  }
  if(joined)lanes_reset(port,EPROTO,"made again by its peer",0);
  memcpy(l->peer,heard->pair,sizeof l->peer);
  l->peer_room=(uint32_t)heard->code;l->heard=1;
  lanes_turn(port);
}
int exec_served;

/* design/bridge-and-ipc.md#protocol */
static void port_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  if(!port->greeted){
    if(heard->kind!=HELLO || heard->block!=M->block*M->pgsz || (port->peer && heard->node!=port->peer)){port_lost(port,EPROTO,0);return;}
    port->peer=heard->node;port->peer_port=heard->port;port->heard=*heard;port->greeted=1;
    struct mesh_link_info *link=&mesh_links(M)[port->index];
    atomic_store_explicit(&link->bandwidth,port->bandwidth,memory_order_relaxed);
    atomic_fetch_add_explicit(&link->pairings,1,memory_order_relaxed);
    say("link %u (%s): paired with node %u\n",port->index,port->name,port->peer);
    link_set(port,port->peer,MESH_PAIRED,0);
    /* a port newly paired is sent every record the map holds, this node's first */
    for(uint64_t i=0;port->greeted && i<atomic_load(&M->map_nodes);i++)if(mesh_records(M)[i].node!=M->node)
      port_say(port,(struct message){.kind=RECORD},mesh_records(M)+i,sizeof(struct mesh_record));
    return;
  }
  if(heard->kind==RECORD){record_heard(port,payload,heard->bytes);return;}
  if(heard->kind==FAREWELL){port->parted=1;port->farewell=heard->code;return;}
  if(heard->kind==READY){ready_heard(port,heard,payload);return;}
  if(heard->kind==LANES){lanes_heard(port,heard);return;}
  if(heard->kind==STRIPES){stripes_heard(port,heard);return;}
  if(heard->kind==STREAM){stream_heard(port,heard,payload);return;}
  if(heard->kind>=CHANNEL_OPEN && heard->kind<=CHANNEL_CLOSE){channel_heard(port,heard,payload);return;}
  struct flow *f=flow_on(port,heard->key,heard->to,heard->from,heard->stripe);
  if(!f)return;
  const uint32_t origin=heard->failed?heard->failed:port->peer;
  if(heard->kind==PAIRS){
    /* #lanes: an exclusive flow's PAIRS its queue pairs, a lane rider's (count 0) its index and generation */
    const int error=flow_mode(f,heard->count!=0)?EPROTO:
      f->exclusive && heard->count!=(uint32_t)f->qps && heard->count!=(uint32_t)f->qps+1?EPROTO:0;
    if(error)flow_fail(f,error,4,FAIL,port->peer,M->node);
    else{
      /* #keepalive: a count one past its data's queue pairs names the peer's keepalive pair last */
      if(f->exclusive){memcpy(f->pairs,heard->pair,sizeof f->pairs);f->peer_alive=heard->count==(uint32_t)f->qps+1;}
      else{f->peer_slot=(uint32_t)heard->code;f->peer_generation=(uint32_t)((uint64_t)heard->code>>32);}
      f->heard|=HEARD_PAIRS;
    }
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

/* design/recovery.md#detection, #loss: a link lost fails every program on it, naming its peer; the port keeps its
   device's context, protection domain and registration (design/RDMA-KERNEL-RECOVERY.md "October 18"); the next
   pairing is a pause away */
void port_lost(struct port *port,int64_t code,int retire){
  if(port->control>=0){close(port->control);port->control=-1;}
  /* design/recovery.md#diagnosis: the peer named, and a peer that said farewell told from one lost */
  if(port->greeted||port->connecting){
    const char *retired=retire?", device retired":"";
    char reason[96];farewell_text(port->farewell,reason,sizeof reason);
    if(stop)say("link %u (%s): closed toward node %u: this bridge is stopping%s\n",port->index,port->name,port->peer,retired);
    else if(port->peer && port->parted)say("link %u (%s): node %u parted, its bridge stopping (%s): code %lld%s\n",port->index,port->name,
                                            port->peer,reason,(long long)code,retired);
    else if(port->peer)say("link %u (%s): node %u lost: code %lld%s\n",port->index,port->name,port->peer,(long long)code,retired);
    else say("link %u (%s): lost: code %lld%s\n",port->index,port->name,(long long)code,retired);
  }
  /* #lanes: read by no progress thread once the stops of its flows (queued after) settle the port's table, and closed
     once those stops have run */
  atomic_store_explicit(&port->lanes.up,0,memory_order_release);
  for(struct flow *f=flows;f;f=f->next)if(f->port==port)flow_fail(f,port->parted?ECONNRESET:code,4,0,port->peer,M->node);
  requests_lost(port,port->parted?ECONNRESET:code);
  if(port->lanes.state!=LANES_DOWN)port->lanes.state=LANES_CLOSING;
  port->lanes.heard=0;port->lanes.retry=0;
  /* design/bridge-dataflow.md#stripes: the stripe flow alike, closed once the stops before it have run */
  atomic_store_explicit(&port->stripes.up,0,memory_order_release);
  if(port->stripes.state!=STRIPES_DOWN)port->stripes.state=STRIPES_CLOSING;
  port->stripes.heard=0;port->stripes.wanted=0;port->stripes.retry=0;
  /* design/bridge-dataflow.md#streams: a stream on the port fails, its requester answered */
  if(port->stream.state!=STREAMS_DOWN && port->stream.state!=STREAMS_CLOSING)streams_fail(port,port->parted?ECONNRESET:code,0);
  port->greeted=port->parted=port->connecting=port->writing=0;port->farewell=0;port->peer=0;port->held=0;port->out_length=port->out_sent=0;
  port->retiring|=retire;port->open&=!retire;port->next=now+MESH_PAUSE_NS;
  link_set(port,0,retire?MESH_STOPPED:MESH_PAIRING,code);
}
/* design/recovery.md#detection: the control socket's keepalive tells a side whose own link stays up (its peer's went
   down, its peer's host died); the socket stays non-blocking, its sends queued (port_say) */
static void port_greet(struct port *port){
  int on=1,idle=1,interval=1,count=2,flags=fcntl(port->control,F_GETFL);
  if(flags<0 || fcntl(port->control,F_SETFL,flags|O_NONBLOCK) || setsockopt(port->control,SOL_SOCKET,SO_KEEPALIVE,&on,sizeof on) ||
     setsockopt(port->control,IPPROTO_TCP,TCP_KEEPALIVE,&idle,sizeof idle) || setsockopt(port->control,IPPROTO_TCP,TCP_KEEPINTVL,&interval,sizeof interval) ||
     setsockopt(port->control,IPPROTO_TCP,TCP_KEEPCNT,&count,sizeof count) ||
     setsockopt(port->control,SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on)){port_lost(port,errno?errno:EIO,0);return;}
  port_watch(port,EVFILT_READ);
  port_say(port,(struct message){.kind=HELLO},NULL,0);
}
/* design/bridge-and-ipc.md#ports: the port a link-local peer reached, by its interface; none for a scope of 0 (no
   interface: a LAN address, or the index of an interface that is gone), nor for a port whose device is retiring, which
   its turn alone opens again (design/RDMA-KERNEL-RECOVERY.md) */
static struct port *port_at(uint32_t interface){
  for(uint32_t p=0;interface && p<port_count;p++)if(ports[p].interface==interface)return ports[p].retiring?NULL:ports+p;
  return NULL;
}
/* design/bridge-and-ipc.md#ports: a port's turn: on its own queue, its device retired where its port went away (once no
   flow and no lane holds queue pairs on it), else opened and its port read; then, while it has no peer, its hello sent to every
   node on its link */
static void port_turn(struct port *port){
  if(port->job || port->control>=0 || now<port->next || (port->retiring && (port->flows || port->lanes.state!=LANES_DOWN || port->stripes.state!=STRIPES_DOWN ||
     port->stream.state!=STREAMS_DOWN)))return;
  port->next=now+MESH_PAUSE_NS;
  port->interface=if_nametoindex(port->name);
  const int retire=port->retiring;
  port->job=1;atomic_store_explicit(&port->done,0,memory_order_relaxed);
  dispatch_async(port->work,^{
    int result=0,refused=0;
    if(retire && !down_device(&port->device))result=-(errno?errno:EBUSY);
    else if(!(result=device_up(&port->device,&wire,&refused)))result=device_gid(&port->device);
    port->result=result;port->refused=refused;
    atomic_store_explicit(&port->done,1,memory_order_release);
    wake();
  });
}
/* design/recovery.md#registration: a port's turn taken: a device whose registration of the window it refused keeps the
   extents it took and turns again a pause later, 60 s doubling to 30 min (device_up logged the extent and why); a
   device that failed otherwise, its link active, is retired */
static void port_opened(struct port *port){
  const int result=port->result;
  port->job=0;
  if(result<0){say("link %u (%s): device retirement: %s\n",port->index,port->name,strerror(-result));return;}
  if(port->refused){
    port->backoff=port->backoff?MIN(2*port->backoff,MESH_REFUSED_MOST_NS):MESH_REFUSED_NS;
    port->next=now+port->backoff;
    say("link %u (%s): the window's registration refused (%s): the port turns again in %.0f s\n",port->index,port->name,
        strerror(port->refused),(double)port->backoff/1e9);
    port->open=0;if(port->control<0)link_set(port,0,MESH_STOPPED,result);
    return;
  }
  port->retiring=result && port->device.context && result!=ENETDOWN;
  if(result){port->open=0;if(port->control<0)link_set(port,0,MESH_STOPPED,result);return;}
  port->backoff=0;
  atomic_store_explicit(&mesh_links(M)[port->index].max_pairs,port->device.max_pairs,memory_order_relaxed);
  port->open=1;port->lid=port->device.port.lid;memcpy(port->gid,&port->device.gid,16);
  port->bandwidth=port_bandwidth(&port->device.port);
  if(port->control>=0)return;
  link_set(port,0,MESH_PAIRING,0);
#ifdef MESH_LOOP_FABRIC
  /* design/bridge-dataflow.md#the-loopback-fabric: the cable's lower end dials the higher's listener on loopback, as a
     link's lower node dials the higher's on the address it heard; the higher accepts (main, TAG_LOOP) */
  for(uint32_t k=0;k<loop_config.ports;k++)if(!strcmp(loop_config.port[k].device,port->device.name) && loop_config.port[k].listen<loop_config.port[k].peer){
    struct sockaddr_in6 to={.sin6_len=sizeof to,.sin6_family=AF_INET6,.sin6_port=htons(loop_config.port[k].peer),.sin6_addr=IN6ADDR_LOOPBACK_INIT};
    port->control=socket(AF_INET6,SOCK_STREAM,0);
    if(port->control<0 || fcntl(port->control,F_SETFL,O_NONBLOCK) || (connect(port->control,(struct sockaddr *)&to,sizeof to) && errno!=EINPROGRESS)){
      port_lost(port,errno,0);return;
    }
    port->peer=0;port->connecting=1;port->next=now+MESH_PAIR_NS;
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
       (connect(port->control,(struct sockaddr *)&from,sizeof from) && errno!=EINPROGRESS)){port_lost(port,errno,0);continue;}
    port->peer=heard.node;port->connecting=1;port->next=now+MESH_PAIR_NS;
    port_watch(port,EVFILT_WRITE);
  }
}
/* a control socket's messages, each a header and its `bytes` */
static void control_heard(struct port *port){
  for(;;){
    size_t need=sizeof(struct message);
    if(port->held>=need){
      const struct message *header=(const void *)port->in;
      if(header->magic!=XMAGIC || header->bytes>(UINT32_C(64)<<20)){port_lost(port,EPROTO,0);return;}
      need+=header->bytes;
    }
    if(need>port->in_room){
      unsigned char *grown=realloc(port->in,need);
      if(!grown){port_lost(port,ENOMEM,0);return;}
      port->in=grown;port->in_room=need;
    }
    if(port->held<need){
      ssize_t got=recv(port->control,port->in+port->held,need-port->held,0);
      if(got<0 && (errno==EAGAIN || errno==EINTR))return;
      if(got<=0){port_lost(port,got?errno:(port->parted?ECONNRESET:ENETRESET),0);return;}
      port->held+=(size_t)got;
      if(port->held<need || need==sizeof(struct message))continue;
    }
    struct message heard;memcpy(&heard,port->in,sizeof heard);
    port->held=0;
    port_heard(port,&heard,port->in+sizeof heard);
    if(port->control<0)return;
  }
}
/* design/bridge-and-ipc.md#lanes: the lanes' pages of every port the region has, leased at the bridge's start beside the
   map's, so a port found later (an interface attached once clients hold the window) has them; a port whose pages the
   window could not give takes them when its lanes are made (lanes_turn) */
static uint32_t lanes_page[MESH_PORTS];
static void lanes_reserve(void){
  for(uint32_t p=0;p<MESH_PORTS;p++)lanes_page[p]=mesh_arena_alloc(&bridge,lane_pages(),1);
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
    *port=(struct port){.index=port_count,.control=-1,.work=dispatch_queue_create("mesh.port",DISPATCH_QUEUE_SERIAL),.stream={.fd=-1}};
    /* design/bridge-and-ipc.md#lanes: the lanes' headers and discard blocks, the pages leased for the port's index */
    port->lanes.pages=lane_pages();
    port->lanes.page=lanes_page[port_count];
    port->lanes.provider.device=&port->device;
    port->stripes.provider.device=&port->device;
    port->stream.provider.device=&port->device;
    snprintf(port->device.name,sizeof port->device.name,"%s",devices[i]);snprintf(port->name,sizeof port->name,"%s",devices[i]+5);
    port_count++;
    link_set(port,0,MESH_STOPPED,0);
  }
}
/* design/prepared-machine.md#M24: a port's link off or its interface detached; an interface that attached or whose link
   came on and is no port yet: the node's devices read again */
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
      if(code==KEV_DL_LINK_OFF || code==KEV_DL_IF_DETACHED)port_lost(ports+p,ENETDOWN,0);
      else if(code==KEV_DL_LINK_ON || code==KEV_DL_IF_ATTACHED)ports[p].next=now;
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

/* design/recovery.md#diagnosis: a stopping bridge's FAREWELL sent before its socket closes, waiting at most 200 ms on a
   peer that does not read */
static void farewell_flush(struct port *port){
  if(port->control<0)return;
  struct timeval bound={0,200000};
  const int flags=fcntl(port->control,F_GETFL);
  if(flags>=0)fcntl(port->control,F_SETFL,flags&~O_NONBLOCK);
  setsockopt(port->control,SOL_SOCKET,SO_SNDTIMEO,&bound,sizeof bound);
  while(port->out_sent<port->out_length){
    const ssize_t sent=send(port->control,port->out+port->out_sent,port->out_length-port->out_sent,0);
    if(sent<=0)break;
    port->out_sent+=(size_t)sent;
  }
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
  lanes_reserve();
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
  if(keepalive_ns)snprintf(alive,sizeof alive,", a keepalive frame every %llu ns on each exclusive flow's own queue pair (design/bridge-and-ipc.md#keepalive)",
                           (unsigned long long)keepalive_ns);
  say("bridge node %u (%s): configuration %s, window %llu bytes, arena %llu pages, rows %u%s%s\n",node,host,name,
    (unsigned long long)window,(unsigned long long)mesh_arena_pages(M),M->rows,ledger_kept?", ledgers kept (design/bridge-and-ipc.md#ledger)":"",
    alive);
  uint64_t liveness=0;
  int rebuilt=0;
  while(!stop){
    now=clock_ns();
    if(atomic_load_explicit(&found,memory_order_acquire))ports_found();
    for(uint32_t p=0;p<port_count;p++){
      struct port *port=ports+p;
      if(port->job && atomic_load_explicit(&port->done,memory_order_acquire))port_opened(port);
      if((port->connecting || (port->control>=0 && !port->greeted)) && now>port->next)port_lost(port,ETIMEDOUT,0);
      port_turn(port);
      lanes_turn(port);
      stripes_turn(port);
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
      /* design/bridge-and-ipc.md: a bridge whose binary changed stops between calls, once it holds no flow and no
         awaiting program (`busy` 0) */
      if(!rebuilt && binary.st_ino && (stat(argv[0],&current) || current.st_ino!=binary.st_ino || current.st_mtimespec.tv_sec!=binary.st_mtimespec.tv_sec)){
        rebuilt=1;
        atomic_store_explicit(&M->stopping,1,memory_order_release);
        mesh_notify(&M->control);
        if(flow_count || await_count)say("bridge node %u: its binary changed; it stops once no program runs through it\n",node);
      }
      if(rebuilt && !flow_count && !await_count){stop=1;stop_reason=MESH_STOP_BINARY;}
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
          port->control=accepted;port->next=now+MESH_PAIR_NS;
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
          port->control=accepted;port->peer=0;port->next=now+MESH_PAIR_NS;
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
          if(error)port_lost(port,error,0);
          else {port->connecting=0;port_greet(port);}
        }
        else if(e->filter==EVFILT_WRITE){port->writing=0;port_flush(port);}
        else control_heard(port);
      }
    }
  }
  char why[96];farewell_text(stop_reason,why,sizeof why);
  say("bridge node %u: stopping (%s)\n",node,why);
  requests_stop();
  for(uint32_t p=0;p<port_count;p++){
    port_say(ports+p,(struct message){.kind=FAREWELL,.code=stop_reason},NULL,0);
    farewell_flush(ports+p);
    ports[p].parted=1;
    port_lost(ports+p,ESHUTDOWN,0);
  }
  /* design/RDMA-KERNEL-RECOVERY.md "October 18": each flow port_lost failed is released as at any flow's end (its own
     queue pairs, under the guard); the lanes', stripes', stream's and device's objects are left to the kernel at exit */
  for(uint32_t p=0;p<port_count;p++){
    struct port *port=ports+p;
    dispatch_sync(port->work,^{
      atomic_store_explicit(&port->lanes.up,0,memory_order_release);
      port_settle(port);
    });
    port_still(port);
    dispatch_sync(port->work,^{
      if(port->stream.fd>=0){fd_answer(port->stream.fd,ESHUTDOWN);close(port->stream.fd);port->stream.fd=-1;}
    });
  }
  atomic_store(&M->bridge_pid,0);
  mesh_notify(&M->control);
  pthread_join(watcher,NULL);
  munmap(wire.data,wire.length);munmap(M,length);
  return 0;
}
