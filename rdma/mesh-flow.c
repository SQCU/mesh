#include "mesh-verbs.h"
#include "mesh-dataflow.h"
#include <sys/event.h>
#include <sys/ioctl.h>
#include <sys/kern_event.h>
#include <sys/socket.h>
#include <net/if.h>
#include <net/if_var.h>
#include <net/net_kev.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

_Static_assert(sizeof(pthread_t)==8,"M17");

_Static_assert(sizeof(struct ibv_send_wr)==128,"M06");
_Static_assert(sizeof(struct ibv_sge)==16,"M07 M09");
_Static_assert(sizeof(struct ibv_recv_wr)==32,"M08");
_Static_assert(sizeof(struct ibv_wc)==48,"M11");
_Static_assert(offsetof(struct ibv_wc,status)==8 && offsetof(struct ibv_wc,opcode)==12 &&
  sizeof(((struct ibv_wc *)0)->status)==4 && sizeof(((struct ibv_wc *)0)->opcode)==4,"M11");
_Static_assert(sizeof(struct ibv_send_wr *)==8,"M15");
/* design/prepared-machine.md#M24 */
union mesh_network_event {
  struct kern_event_msg header;
  unsigned char bytes[KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data)];
};
_Static_assert(sizeof(union mesh_network_event)==48,"M24");
/* design/prepared-machine.md#M29 */
struct prepared_send {
  _Alignas(128) struct ibv_sge span;
  struct ibv_send_wr request;
};
_Static_assert(sizeof(struct prepared_send)==256 && _Alignof(struct prepared_send)==128 &&
  offsetof(struct prepared_send,span)==0 && offsetof(struct prepared_send,request)==16 &&
  offsetof(struct prepared_send,request.send_flags)+sizeof(unsigned int)<=64,"M29");
/* design/prepared-machine.md#M08 */
/* One chunk of a receive transfer in one invocation: its native request and SGE into the chunk's
   destination range, the completion word it stores (`input`), its queue pair, the frames it holds
   posted, its invocation and ring. */
struct prepared_receive {
  _Alignas(128) struct ibv_recv_wr request;
  struct ibv_sge span;
  _Atomic uint64_t *input;
  struct ibv_qp *pair;
  uint64_t argument;
  uint32_t frames,invocation,queue,cycle,behind;
};
_Static_assert(sizeof(struct prepared_receive)==128 && _Alignof(struct prepared_receive)==128 &&
  offsetof(struct prepared_receive,span)==32 && offsetof(struct prepared_receive,input)==48,"M08 M09");
/* design/prepared-machine.md#M08 */
/* A queue pair's receive ring: its records in invocation, binding, chunk order; `posted` the next to
   post, `outstanding` the frames posted and not completed, within the queue's `capacity`; a record is
   posted once the record `behind` it in the ring's running order has landed: its transfer's and chunk's
   record `span` invocations before (the ring slots' margin), or in a cyclic program its own previous
   posting where none is; a record with none is posted at once.  A transfer's slots are its own, so a ring
   whose transfers begin late, or leave invocations between them, posts each transfer's first records. */
struct receive_ring {
  struct prepared_receive *first;
  size_t count,posted;
  int64_t completed;
  uint32_t outstanding,capacity,span;
  /* the census: records landed, their bytes */
  uint64_t landed,bytes;
};
/* A queue pair's SEND gate, the send thread's alone: the frames posted and retired (each request
   carries its queue and the frames posted through it, which its completion returns), the queue's
   capacity, and the census's requests and completions. */
#define MESH_GATE_MASK ((UINT64_C(1)<<48)-1)
struct send_gate {
  _Alignas(64) uint64_t posted,retired,requests,completions;
  uint32_t capacity;
};
/* A SEND stream's progress: its current cell, the next request of that cell's chain and the requests
   left. */
struct send_stream { struct mesh_send *cell,*first; struct ibv_send_wr *next; uint32_t remaining; uint64_t cycle; };

/* design/recovery.md#pause, #first: the pause between pairings, how long a program waits for its peer's to start, and
   how long the queue pairs of two started programs take to pair */
#define PAUSE_NS UINT64_C(3000000000)
#define PEER_NS UINT64_C(600000000000)
#define PAIR_NS UINT64_C(30000000000)
#define MESH_PAIRS 16
/* design/bridge-and-ipc.md#protocol */
enum { HELLO=1, READY, PAIRS, POSTED, END, FAIL, FAREWELL };
struct message { uint32_t magic,kind,node,count; uint64_t key,boot; int64_t code; uint32_t block; uint16_t lid,padding; uint8_t gid[16]; uint32_t pair[MESH_PAIRS][2]; };
#define XMAGIC (0x4d595048u+MESH_VERSION)
/* design/bridge-and-ipc.md#flows: a client's program on one port */
enum { WAITING, READIED, PAIRING_QUEUES, POSTING, RUNNING, FAILED };
struct port;
struct flow {
  pthread_t workers[2];
  uint32_t worker_count,publication_count,stream_count,index,slot;
  _Atomic int progressing;
  _Atomic int64_t failed;
  _Atomic uint32_t domain;
  int state,qps,cyclic;
  uint64_t client,key,deadline;
  uint32_t invocations;
  struct hdr *M; struct port *port; struct mesh_verbs provider;
  struct prepared_receive *receive;
  size_t receive_count;
  struct receive_ring *rings;
  struct send_gate *gates;
  struct send_stream *streams;
  struct prepared_send *requests;
  size_t request_count;
  struct mesh_send *publications;
  struct ibv_wc completion;
  struct mesh_cancellation *cancel;
  /* design/prepared-machine.md#M27: the program's trace, where it asked for one (mesh_transfers_trace) */
  struct mesh_trace_ring *trace[2];
};
/* design/bridge-and-ipc.md#ports: a port, its link and the programs on it */
struct pending { uint64_t key,until; uint32_t count; };
struct port {
  struct mesh_device device;
  uint32_t index,interface,peer,greeted,parted,connecting,retiring;
  char name[IFNAMSIZ];
  int control;
  uint64_t next,boot;
  struct message heard;
  size_t held;
  unsigned char buffer[sizeof(struct message)];
  struct pending pending[MESH_CLIENTS];
  struct flow *flows[MESH_CLIENTS];
};
static struct hdr *M;
static struct mesh_wire wire;
static struct port ports[MESH_PORTS];
static uint32_t port_count;
static int events=-1,listener=-1,hello=-1,network=-1;
static uint64_t now;
static const struct in6_addr everyone={{{0xff,0x02,0,0,0,0,0,0,0,0,0,0,0,0,0,1}}};

static uint64_t clock_ns(void){ return clock_gettime_nsec_np(CLOCK_MONOTONIC); }
static void wake(void){
  struct kevent64_s event;EV_SET64(&event,0,EVFILT_USER,0,NOTE_TRIGGER,0,0,0,0);
  kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
}
/* design/algorithm-sources.md#meshresult: a data thread's failure, taken by the event loop (flow_fail) */
static void flow_lost(struct flow *flow,int64_t code,uint32_t domain){
  int64_t none=0;
  if(atomic_compare_exchange_strong(&flow->failed,&none,code?code:EIO))atomic_store(&flow->domain,domain);
  atomic_store_explicit(&flow->progressing,0,memory_order_release);
  wake();
}
static struct mesh_port_info *flow_status(struct flow *flow){ return &mesh_client_at(flow->M,flow->client)->port[flow->index]; }

/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M06 */
/* design/algorithm-sources.md#programcopy */
static int link_prepare(struct flow *link){
  struct hdr *m=link->M;
  struct mesh_tx *tx=&mesh_client_at(m,link->client)->tx[link->index];
  /* design/prepared-machine.md#M12 */
  link->cancel=tx->cancel?(void *)((char *)m+tx->cancel):NULL;
  if(tx->trace){link->trace[MESH_SEND]=(void *)((char *)m+tx->trace);link->trace[MESH_RECEIVE]=(void *)(link->trace[MESH_SEND]->records+link->trace[MESH_SEND]->capacity);}
  link->publications=(void *)((char *)m+tx->cells);
  link->publication_count=tx->count*tx->slots+tx->once;
  uint32_t queues=0;
  for(uint32_t q=0;q<MESH_QUEUES;q++)for(int d=0;d<2;d++)
    if(atomic_load(mesh_order_length(m,link->client,link->index*MESH_QUEUES+q,d)))queues=q+1;
  link->qps=(int)(queues*tx->slots);
  if(link->qps>MESH_PAIRS)return E2BIG;
  size_t requests=0,records=0;
  for(uint32_t q=0;q<queues;q++)for(uint32_t d=0;d<2;d++){
    uint32_t channel=link->index*MESH_QUEUES+q;
    struct mesh_transfer *transfers=mesh_transfers(m,link->client,channel,d);
    for(uint32_t i=0;i<atomic_load(mesh_order_length(m,link->client,channel,d));i++){
      size_t chunks=mesh_row_chunks(m,transfers[i].local_row,transfers[i].bytes),
        each=(size_t)transfers[i].count*mesh_transfer_active(transfers+i,tx->invocations);
      if(d==MESH_SEND)requests+=(chunks-1)*each;
      else records+=chunks*each;
    }
  }
  /* design/prepared-machine.md#M29 */
  link->requests=requests?aligned_alloc(128,requests*sizeof *link->requests):NULL;
  if(requests&&!link->requests)return ENOMEM;
  link->request_count=requests;link->receive_count=records;
  return 0;
}

/* The ring slot a transfer's invocation t addresses: (t - begin) mod its depth, at its slot pages. */
static uint32_t transfer_row(struct hdr *m,uint32_t fallback,const struct mesh_transfer *transfer,uint32_t row,uint32_t invocation,uint32_t chunk){
  uint32_t depth=transfer->depth?transfer->depth:fallback;
  if(!depth)depth=1;
  return row+(uint32_t)((size_t)((invocation-transfer->begin)%depth)*transfer->invocation_pages/m->block)+chunk;
}

/* design/prepared-machine.md#M09: a chunk's span, its extent registered on the port's device first */
static int chunk_span(struct flow *link,uint32_t address,uint64_t remaining,struct ibv_sge *span){
  struct hdr *m=link->M;
  uint64_t offset=atomic_load_explicit(&mesh_page(m)[address].address,memory_order_relaxed)-m->data_off;
  int error=device_register(link->provider.device,offset,wire.length);
  if(!error)*span=wire_span(link->provider.device,(uint64_t)m->block*m->pgsz,offset,remaining);
  return error;
}

/* design/prepared-machine.md#M08 */
/* Posts the ring's next records while its queue has room and the record behind each has landed (its
   transfer's `span` invocations before: the ring's records land in the order posted, so `landed` counts
   them).  Called once at configuration and at each receive completion: the refill is fired by the event. */
/* design/prepared-machine.md#M30 */
/* A cyclic program's ring wraps: its posted count runs on, record posted % count in cycle posted / count,
   at running invocation invocation + cycle * N; the span is held under N and a record with no record of
   its transfer behind it waits on its own previous posting, so a record is posted again only after its
   last posting completed (completions arrive in order). */
static int ring_advance(struct flow *link,struct receive_ring *ring){
  int (*post)(struct ibv_qp *,struct ibv_recv_wr *,struct ibv_recv_wr **)=link->provider.queues[0].receive;
  while(ring->count && (link->cyclic || ring->posted<ring->count)){
    struct prepared_receive *record=ring->first+ring->posted%ring->count;
    uint64_t cycle=ring->posted/ring->count;
    if((record->behind && ring->posted>=record->behind && ring->landed<=ring->posted-record->behind) ||
       ring->outstanding+record->frames>ring->capacity)break;
    record->cycle=(uint32_t)cycle;
    struct ibv_recv_wr *bad;
    int error=post(record->pair,&record->request,&bad);
    if(error)return error<0?-error:error;
    ring->outstanding+=record->frames;ring->posted++;
  }
  return 0;
}

/* design/algorithm-sources.md#programcopy */
/* design/prepared-machine.md#M08 */
/* Every SEND cell's chain of chunk requests and its stream successor, and every receive transfer's
   chunk records in one ring per queue pair, for every invocation of each transfer's range (the SEND
   cells' and completion words' layout is mesh_transfers_prepare's).  A stream's cells run
   invocation by invocation, the publications of each in binding order, as the peer's ring posts its
   receives: TN3205's credit flow control holds a SEND until its matching receive is posted, so any
   number of records an invocation streams through a ring of the queue's capacity, refilled by the
   completions (metal-microbench docs/kernels.md Transport (git 7300d1e^)).  0, or the error. */
static int link_configure(struct flow *link){
  struct hdr *m=link->M;struct mesh_client *client=mesh_client_at(m,link->client);
  struct mesh_tx *tx=&client->tx[link->index];
  uint32_t invocations=tx->invocations?tx->invocations:1,fallback=atomic_load_explicit(&client->depth,memory_order_acquire),depth=fallback;
  if(!depth||depth>invocations)depth=invocations;
  link->cyclic=atomic_load_explicit(&client->cyclic,memory_order_acquire)!=0;link->invocations=invocations;
  int qps=link->qps;uint32_t queues=(uint32_t)qps/tx->slots;
  link->streams=calloc((size_t)qps,sizeof *link->streams);
  link->gates=aligned_alloc(64,(size_t)qps*sizeof *link->gates);
  link->rings=calloc((size_t)qps,sizeof *link->rings);
  link->receive=aligned_alloc(128,(link->receive_count+1)*sizeof *link->receive);
  struct mesh_send **last=calloc((size_t)qps,sizeof *last),**terminal=calloc((size_t)qps,sizeof *terminal);
  if(!link->streams||!link->gates||!link->rings||!link->receive||!last||!terminal){free(last);free(terminal);return ENOMEM;}
  /* a record or request is one frame at least, so a queue pair's frames also bound the completions
     it can have pending in the link's receive and SEND completion queues, which its pairs share */
  uint32_t receives=(uint32_t)link->provider.completion->cqe/(uint32_t)qps,sends=(uint32_t)link->provider.sent->cqe/(uint32_t)qps;
  for(int q=0;q<qps;q++){
    link->gates[q]=(struct send_gate){.capacity=MIN(link->provider.queues[q].send_capacity,sends)};
    link->rings[q]=(struct receive_ring){.completed=-1,.capacity=MIN(link->provider.queues[q].receive_capacity,receives),.span=UINT32_MAX};
  }
  /* design/prepared-machine.md#M29 */
  size_t next=0;int error=0;
  for(uint32_t q=0;q<queues && !error;q++){
    uint32_t channel=link->index*MESH_QUEUES+q,count=atomic_load(mesh_order_length(m,link->client,channel,MESH_SEND));
    struct mesh_transfer *out=mesh_transfers(m,link->client,channel,MESH_SEND);
    /* each cell's chain over its invocation's ring slot */
    for(uint32_t i=0;i<count && !error;i++){
      uint32_t active=mesh_transfer_active(out+i,invocations),column=active+1;
      for(uint32_t slot=0;slot<out[i].count && !error;slot++){
        uint32_t row=out[i].local_row+slot*out[i].stride,chunks=mesh_row_chunks(m,row,out[i].bytes),stream=q*tx->slots+slot;
        struct mesh_send *cells=link->publications+out[i].first+(size_t)slot*column;
        for(uint32_t u=0;u<active && !error;u++){
          struct mesh_send *cell=cells+u;
          cell->pair=(uintptr_t)link->provider.queues[stream].pair;cell->queue=stream;cell->chunks=chunks;
          uint64_t remaining=out[i].bytes;
          for(uint32_t k=0;k<chunks && !error;k++){
            struct ibv_sge span;
            if((error=chunk_span(link,transfer_row(m,fallback,out+i,row,out[i].begin+u,k),remaining,&span)))break;
            remaining-=span.length;
            struct ibv_sge *entry=k?&link->requests[next+k-1].span:&cell->span;
            struct ibv_send_wr *request=k?&link->requests[next+k-1].request:&cell->request;
            *entry=span;
            /* the first request's wr_id stays the producer's nonzero argument (mesh_transfers_prepare),
               which a host producer reads at publication (mesh_host_publish) */
            *request=(struct ibv_send_wr){.wr_id=k?0:1,.next=k+1==chunks?NULL:&link->requests[next+k].request,
              .sg_list=entry,.num_sge=1,.opcode=IBV_WR_SEND};
          }
          next+=chunks-1;
        }
        terminal[stream]=cells+active;
      }
    }
    /* each stream's cells in invocation order, its publications in binding order within one */
    for(uint32_t t=0;t<invocations && !error;t++)for(uint32_t i=0;i<count;i++){
      uint32_t active=mesh_transfer_active(out+i,invocations);
      if(t<out[i].begin || t-out[i].begin>=active)continue;
      for(uint32_t slot=0;slot<out[i].count;slot++){
        uint32_t stream=q*tx->slots+slot;
        struct mesh_send *cell=link->publications+out[i].first+(size_t)slot*(active+1)+(t-out[i].begin);
        if(last[stream])last[stream]->successor=(uintptr_t)cell;
        else link->streams[stream].cell=cell;
        last[stream]=cell;
      }
    }
  }
  /* design/prepared-machine.md#M30: a cyclic program's last cell is followed by its first */
  for(int q=0;q<qps && !error;q++)if(last[q])last[q]->successor=link->cyclic?(uintptr_t)link->streams[q].cell:(uintptr_t)terminal[q];
  for(int q=0;q<qps;q++){link->streams[q].first=link->streams[q].cell;link->streams[q].cycle=0;}
  link->stream_count=0;
  for(int q=0;q<qps;q++)if(link->streams[q].cell)link->streams[link->stream_count++]=link->streams[q];
  free(last);free(terminal);
  if(error)return error;
  /* design/prepared-machine.md#M08 */
  /* each queue pair's ring: its records, invocation by invocation, the transfers of each in binding
     order, a transfer's chunks in order */
  size_t at=0;
  for(int ring=0;ring<qps && !error;ring++){
    uint32_t q=(uint32_t)ring/tx->slots,slot=(uint32_t)ring%tx->slots,channel=link->index*MESH_QUEUES+q;
    uint32_t count=atomic_load(mesh_order_length(m,link->client,channel,MESH_RECEIVE));
    struct mesh_transfer *in=mesh_transfers(m,link->client,channel,MESH_RECEIVE);
    struct receive_ring *r=link->rings+ring;
    r->first=link->receive+at;
    size_t *base=calloc((size_t)count+1,sizeof *base),*firsts=NULL;
    for(uint32_t i=0;base&&i<count;i++)base[i+1]=base[i]+(slot<in[i].count?mesh_transfer_active(in+i,invocations):0);
    if(base)firsts=malloc((base[count]+1)*sizeof *firsts);
    if(!base||!firsts){free(base);free(firsts);return ENOMEM;}
    for(uint32_t i=0;i<count;i++){
      if(slot>=in[i].count)continue;
      uint32_t active=mesh_transfer_active(in+i,invocations),own=in[i].depth?in[i].depth:depth;
      /* receive u+span lands in slot (u+span) mod depth while the local reader may still be at u:
         the ring's span is held a slot short of the ring; a transfer whose every invocation has its
         own slot bounds nothing */
      if(own<active)r->span=MIN(r->span,own>1?own-1:1);
    }
    for(uint32_t t=0;t<invocations && !error;t++)for(uint32_t i=0;i<count && !error;i++){
      uint32_t active=mesh_transfer_active(in+i,invocations);
      if(slot>=in[i].count || t<in[i].begin || t-in[i].begin>=active)continue;
      uint32_t u=t-in[i].begin,row=in[i].local_row+slot*in[i].stride,chunks=mesh_row_chunks(m,row,in[i].bytes);
      uint64_t remaining=in[i].bytes;
      for(uint32_t k=0;k<chunks;k++){
        struct ibv_sge span;
        if((error=chunk_span(link,transfer_row(m,fallback,in+i,row,t,k),remaining,&span)))break;
        remaining-=span.length;
        /* design/prepared-machine.md#M07 */
        struct mesh_publication *delivery=mesh_publication_at(m,row+k);
        uintptr_t input=(uintptr_t)&delivery->argument;
        if(delivery->device_input)input=(uintptr_t)m+delivery->device_input+(uintptr_t)delivery->device_stride*u;
        else if(delivery->sends)input=(uintptr_t)m+delivery->targets[0].stream+sizeof(struct mesh_send)*(size_t)u;
        struct prepared_receive *record=link->receive+at++;
        if(!k)firsts[base[i]+u]=(size_t)(record-r->first);
        *record=(struct prepared_receive){.request={.wr_id=(uintptr_t)record,.sg_list=&record->span,.num_sge=1},
          .span=span,.input=(_Atomic uint64_t *)input,.pair=link->provider.queues[ring].pair,.argument=1,
          .frames=(span.length+4095)/4096,.invocation=t,.queue=(uint32_t)ring};
      }
    }
    r->count=(size_t)(link->receive+at-r->first);
    /* design/prepared-machine.md#M30 */
    if(link->cyclic)r->span=MIN(r->span,invocations>1?invocations-1:1);
    for(uint32_t i=0;i<count && !error;i++){
      uint32_t active=(uint32_t)(base[i+1]-base[i]);
      if(!active)continue;
      uint32_t chunks=mesh_row_chunks(m,in[i].local_row+slot*in[i].stride,in[i].bytes);
      for(uint32_t u=0;u<active;u++){
        size_t here=firsts[base[i]+u],behind=0;
        if(u>=r->span)behind=here-firsts[base[i]+u-r->span];
        else if(link->cyclic&&u+active>=r->span)behind=here+r->count-firsts[base[i]+u+active-r->span];
        else if(link->cyclic)behind=r->count;
        for(uint32_t k=0;k<chunks;k++)r->first[here+k].behind=(uint32_t)behind;
      }
    }
    free(base);free(firsts);
    if(!error)error=ring_advance(link,r);
  }
  if(at)link->receive[at]=link->receive[at-1];
  return error;
}

/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M06 */
/* design/prepared-machine.md#M15 */
/* design/algorithm-sources.md#independent-native-queues */
/* The link's SEND completions: each retires its queue's frames through its request (the device
   completes every SEND, signaled or not).  0, or an error completion's status. */
static __attribute__((always_inline)) inline int send_retire(struct flow *link){
  struct ibv_wc done[16];
  int count=link->provider.queues[0].poll(link->provider.sent,16,done);
  if(count<0)return EIO;
  for(int i=0;i<count;i++){
    if(done[i].status)return (int)done[i].status;
    struct send_gate *gate=link->gates+(done[i].wr_id>>48);
    gate->retired=done[i].wr_id&MESH_GATE_MASK;gate->completions++;
  }
  return 0;
}

/* design/prepared-machine.md#M27 */
static __attribute__((always_inline)) inline void trace_record(struct mesh_trace_ring *ring,uint64_t identity,uint64_t begin,uint64_t middle){
  uint64_t n=atomic_load_explicit(&ring->count,memory_order_relaxed);
  if(n>=ring->capacity)return;
  ring->records[n]=(struct mesh_trace){identity,begin,middle,clock_gettime_nsec_np(CLOCK_UPTIME_RAW)};
  atomic_store_explicit(&ring->count,n+1,memory_order_release);
}

/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#independent-native-queues */
/* Each stream's cells in order: a cell released by its producer posts its chain's requests while its
   queue pair holds their frames (the gate: frames posted less those its SEND completions retired;
   ibv_post_send accepts past the queue's frames and corrupts later, so the gate is the caller's).
   The thread takes its completions when a stream waits for its producer or its gate. */
static __attribute__((always_inline)) inline void *link_send_drain(struct flow *link,int traced){
  struct send_stream *streams=link->streams;
  uint32_t count=link->stream_count;
  struct ibv_send_wr *bad;
  int (*post)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **)=link->provider.queues[0].send;
  for(uint32_t s=0;;s=s+1==count?0:s+1){
    struct send_stream *stream=streams+s;
    uint64_t observed=0;
    if(!stream->remaining){
      struct mesh_send *cell=stream->cell;
      if(atomic_load_explicit(&cell->ready,memory_order_acquire)<=stream->cycle){
        if(!atomic_load_explicit(&link->progressing,memory_order_acquire))return NULL;
        int error=send_retire(link);
        if(error){flow_lost(link,error,2);return NULL;}
        continue;
      }
      observed=traced?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
      stream->next=&cell->request;stream->remaining=cell->chunks;
    }
    struct mesh_send *cell=stream->cell;
    struct send_gate *gate=link->gates+cell->queue;
    uint64_t posting=traced?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
    while(stream->remaining){
      struct ibv_send_wr *request=stream->next;
      uint32_t frames=(request->sg_list->length+4095)/4096;
      if(((gate->posted-gate->retired)&MESH_GATE_MASK)+frames>gate->capacity){
        if(!atomic_load_explicit(&link->progressing,memory_order_acquire))return NULL;
        int error=send_retire(link);
        if(error){flow_lost(link,error,2);return NULL;}
        break;
      }
      gate->posted+=frames;
      /* posted as a copy: the cell's first request keeps the producer's argument in its wr_id (mesh_transfers_prepare),
         which a host producer reads at every publication (mesh_host_publish), and the copy's names the gate */
      struct ibv_send_wr posting=*request;
      posting.next=NULL;posting.send_flags=IBV_SEND_SIGNALED;
      posting.wr_id=((uint64_t)cell->queue<<48)|(gate->posted&MESH_GATE_MASK);
      int error=post(link->provider.queues[cell->queue].pair,&posting,&bad);
      if(error){flow_lost(link,error<0?-error:error,1);return NULL;}
      stream->next=request->next;stream->remaining--;gate->requests++;
    }
    if(traced && observed)trace_record(link->trace[MESH_SEND],(uint64_t)((char *)cell-(char *)link->M),observed,posting);
    if(!stream->remaining){
      stream->cell=(struct mesh_send *)cell->successor;
      if(stream->cell==stream->first)stream->cycle++;
    }
  }
}
static void *link_send_progress(void *argument){
  struct flow *link=argument;
  pthread_setname_np("mesh.rdma.send");
  return link->trace[MESH_SEND]?link_send_drain(link,1):link_send_drain(link,0);
}

/* design/prepared-machine.md#M08 */
/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#programcopy */
/* The link's receive completions: each stores its completion word and posts the ring's next records. */
static __attribute__((always_inline)) inline void *link_receive_drain(struct flow *link,int traced){
  struct mesh_queue queue=link->provider.queues[0];
  struct ibv_wc *completion=&link->completion;
  for(;;){
    int count=queue.poll(queue.completion,1,completion);
    if(count<0){flow_lost(link,EIO,3);return NULL;}
    if(!count){
      if(!atomic_load_explicit(&link->progressing,memory_order_acquire))return NULL;
      continue;
    }
    uint64_t status_opcode;
    memcpy(&status_opcode,&completion->status,sizeof status_opcode);
    if((uint32_t)status_opcode){flow_lost(link,(uint32_t)status_opcode,2);return NULL;}
    if(!((status_opcode>>32)&IBV_WC_RECV))continue;
    uint64_t polled=traced?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
    struct prepared_receive *record=(void *)(uintptr_t)completion->wr_id;
    struct receive_ring *ring=link->rings+record->queue;
    ring->landed++;ring->bytes+=completion->byte_len;
    atomic_store_explicit(record->input,record->argument+record->cycle,memory_order_release);
    uint64_t published=traced?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
    ring->outstanding-=record->frames;ring->completed=(int64_t)(record->invocation+(uint64_t)record->cycle*link->invocations);
    int error=ring_advance(link,ring);
    if(error){flow_lost(link,error,1);return NULL;}
    if(traced)trace_record(link->trace[MESH_RECEIVE],(uint64_t)(record-link->receive)*sizeof *record,polled,published);
  }
}
static void *link_receive_progress(void *argument){
  struct flow *link=argument;
  pthread_setname_np("mesh.rdma.receive");
  return link->trace[MESH_RECEIVE]?link_receive_drain(link,1):link_receive_drain(link,0);
}

/* design/prepared-machine.md#M08 */
/* The link's progress threads: its receives', and its SENDs' where it publishes.  0, or pthread_create's error. */
static int link_workers(struct flow *link){
  void *(*progress[2])(void *)={link_receive_progress,link_send_progress};
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_set_qos_class_np(&attributes,QOS_CLASS_USER_INTERACTIVE,0);
  int error=0;
  atomic_store_explicit(&link->progressing,1,memory_order_release);
  for(uint32_t d=0;d<1+(link->publication_count!=0 && link->stream_count) && !error;d++){
    error=pthread_create(&link->workers[d],&attributes,progress[d],link);
    if(!error)link->worker_count++;
  }
  pthread_attr_destroy(&attributes);
  return error;
}

/* design/bridge-and-ipc.md#protocol: a message to the port's peer; a send that fails is the link's loss */
static void port_lost(struct port *port,int64_t code,int retire);
static void port_say(struct port *port,uint32_t kind,uint64_t key,uint32_t count,int64_t code,const uint32_t (*pair)[2]){
  if(port->control<0 || (kind!=HELLO && !port->greeted))return;
  struct message said={.magic=XMAGIC,.kind=kind,.node=M->node,.count=count,.key=key,.boot=M->boot,.code=code,
    .block=M->block*M->pgsz,.lid=port->device.port.lid};
  if(kind==HELLO)memcpy(said.gid,&port->device.gid,16);
  for(uint32_t q=0;pair && q<count && q<MESH_PAIRS;q++){said.pair[q][0]=pair[q][0];said.pair[q][1]=pair[q][1];}
  if(send(port->control,&said,sizeof said,0)!=(ssize_t)sizeof said)port_lost(port,errno?errno:EIO,0);
}

/* design/algorithm-sources.md#meshresult: a program on a port stopped (its peer told unless the peer stopped it),
   its threads joined, its pending words cancelled, its queue pairs released; the client's status is the failure */
static void flow_release(struct flow *flow){
  while(flow->worker_count)pthread_join(flow->workers[--flow->worker_count],NULL);
  /* the link's census: each queue pair's requests and frames sent and retired by the send completions
     it took, and its ring's records landed and their bytes */
  for(int q=0;flow->gates && flow->rings && q<flow->qps;q++)
    fprintf(stderr,"link %u census queue=%d requests=%llu frames=%llu retired=%llu send_completions=%llu records=%zu posted=%zu landed=%llu bytes=%llu\n",
      flow->index,q,(unsigned long long)flow->gates[q].requests,(unsigned long long)flow->gates[q].posted,
      (unsigned long long)flow->gates[q].retired,(unsigned long long)flow->gates[q].completions,
      flow->rings[q].count,flow->rings[q].posted,(unsigned long long)flow->rings[q].landed,(unsigned long long)flow->rings[q].bytes);
  if(down_pair(&flow->provider)){
    free(flow->receive);free(flow->streams);free(flow->gates);free(flow->rings);free(flow->requests);
    flow->receive=NULL;flow->streams=NULL;flow->gates=NULL;flow->rings=NULL;flow->requests=NULL;
  }
}
static void flow_fail(struct flow *flow,int64_t code,uint32_t domain,int tell){
  if(flow->state==FAILED)return;
  flow->state=FAILED;
  atomic_store_explicit(&flow->progressing,0,memory_order_release);
  while(flow->worker_count)pthread_join(flow->workers[--flow->worker_count],NULL);
  if(flow->cancel)mesh_cancel(flow->M,flow->cancel,flow->index);
  struct mesh_port_info *status=flow_status(flow);
  atomic_store_explicit(&status->code,code,memory_order_relaxed);atomic_store_explicit(&status->domain,domain,memory_order_relaxed);
  atomic_store_explicit(&status->phase,MESH_STOPPED,memory_order_relaxed);
  atomic_store_explicit(&status->prepared,flow->client,memory_order_release);
  fprintf(stderr,"link %u: program %016llx failed: code %lld domain %u\n",flow->index,(unsigned long long)flow->key,(long long)code,domain);
  flow_release(flow);
  if(tell)port_say(flow->port,FAIL,flow->key,0,code,NULL);
}
static void flow_end(struct flow *flow){
  int announced=flow->state!=FAILED && flow->state>WAITING;
  flow->state=FAILED;
  atomic_store_explicit(&flow->progressing,0,memory_order_release);
  flow_release(flow);
  if(announced)port_say(flow->port,END,flow->key,0,0,NULL);
  if(flow->provider.qp_count)return;
  flow->port->flows[flow->slot]=NULL;
  free(flow);
}

/* design/bridge-and-ipc.md#protocol: both programs started, each side's queue pairs made and named to the other */
static void flow_pair(struct flow *flow,uint32_t count){
  if(count!=(uint32_t)flow->qps){flow_fail(flow,EPROTO,4,1);return;}
  flow->provider.device=&flow->port->device;
  int error=pair_up(&flow->provider,flow->qps);
  if(error){flow_fail(flow,error,1,1);return;}
  uint32_t pairs[MESH_PAIRS][2];
  for(int q=0;q<flow->qps;q++){pairs[q][0]=flow->provider.queues[q].pair->qp_num;pairs[q][1]=(flow->provider.psn+(uint32_t)q)&0xffffff;}
  port_say(flow->port,PAIRS,flow->key,(uint32_t)flow->qps,0,(const uint32_t (*)[2])pairs);
  flow->state=PAIRING_QUEUES;flow->deadline=now+PAIR_NS;
}
static void flow_announce(struct flow *flow){
  struct port *port=flow->port;
  struct mesh_tx *tx=&mesh_client_at(M,flow->client)->tx[flow->index];
  if(tx->peer && tx->peer!=port->peer){flow_fail(flow,EHOSTUNREACH,4,1);return;}
  port_say(port,READY,flow->key,(uint32_t)flow->qps,0,NULL);
  flow->state=READIED;
  for(uint32_t i=0;i<MESH_CLIENTS;i++)if(port->pending[i].until && port->pending[i].key==flow->key){
    uint32_t count=port->pending[i].count;port->pending[i].until=0;
    flow_pair(flow,count);
    break;
  }
}
static void flow_create(struct port *port,uint32_t slot,uint64_t owner){
  struct flow *flow=calloc(1,sizeof *flow);
  if(!flow)return;
  *flow=(struct flow){.index=port->index,.slot=slot,.client=owner,.key=mesh_client_at(M,owner)->key,.M=M,.port=port,
    .deadline=now+PEER_NS,.provider={.device=&port->device}};
  port->flows[slot]=flow;
  struct mesh_port_info *status=flow_status(flow);
  atomic_store_explicit(&status->phase,MESH_PAIRING,memory_order_relaxed);
  int error=port->greeted?link_prepare(flow):ENETDOWN;
  if(!flow->key)error=EINVAL;
  if(error){flow_fail(flow,error,1,1);return;}
  flow_announce(flow);
}
static struct flow *flow_keyed(struct port *port,uint64_t key){
  for(uint32_t s=0;s<MESH_CLIENTS;s++)if(port->flows[s] && port->flows[s]->key==key && port->flows[s]->state!=FAILED)return port->flows[s];
  return NULL;
}

/* design/bridge-and-ipc.md#protocol */
static void port_heard(struct port *port,const struct message *heard){
  if(heard->magic!=XMAGIC){port_lost(port,EPROTO,0);return;}
  if(!port->greeted){
    if(heard->kind!=HELLO || heard->block!=M->block*M->pgsz || (port->peer && heard->node!=port->peer)){port_lost(port,EPROTO,0);return;}
    port->peer=heard->node;port->boot=heard->boot;port->heard=*heard;port->greeted=1;
    struct mesh_link_info *link=&mesh_links(M)[port->index];
    atomic_store_explicit(&link->peer,port->peer,memory_order_relaxed);
    atomic_store_explicit(&link->bandwidth,port_bandwidth(&port->device.port),memory_order_relaxed);
    atomic_fetch_add_explicit(&link->connection,1,memory_order_relaxed);
    atomic_store_explicit(&link->phase,MESH_PAIRED,memory_order_release);
    fprintf(stderr,"link %u (%s): paired with node %u\n",port->index,port->name,port->peer);
    return;
  }
  struct flow *flow=flow_keyed(port,heard->key);
  if(heard->kind==READY){
    if(flow && flow->state==READIED)flow_pair(flow,heard->count);
    else if(!flow)for(uint32_t i=0;i<MESH_CLIENTS;i++)if(!port->pending[i].until || port->pending[i].until<now){
      port->pending[i]=(struct pending){heard->key,now+PEER_NS,heard->count};
      break;
    }
  } else if(heard->kind==PAIRS && flow && flow->state==PAIRING_QUEUES){
    int error=pair_receive(&flow->provider,heard->pair,port->heard.lid,port->heard.gid);
    if(!error)error=link_configure(flow);
    if(error){flow_fail(flow,error,1,1);return;}
    port_say(port,POSTED,flow->key,0,0,NULL);
    flow->state=POSTING;
  } else if(heard->kind==POSTED && flow && flow->state==POSTING){
    int error=pair_send(&flow->provider);
    if(!error)error=link_workers(flow);
    if(error){flow_fail(flow,error,1,1);return;}
    struct mesh_port_info *status=flow_status(flow);
    atomic_store_explicit(&status->phase,MESH_PAIRED,memory_order_relaxed);
    atomic_store_explicit(&status->prepared,flow->client,memory_order_release);
    flow->state=RUNNING;
  } else if((heard->kind==END || heard->kind==FAIL) && flow){
    flow_fail(flow,heard->kind==END?ECONNRESET:ECANCELED,4,0);
  } else if(heard->kind==END || heard->kind==FAIL){
    for(uint32_t i=0;i<MESH_CLIENTS;i++)if(port->pending[i].key==heard->key)port->pending[i].until=0;
  } else if(heard->kind==FAREWELL)port->parted=1;
}

/* design/recovery.md#detection, #loss: a link lost fails every program on it; a port that went away (its link off,
   its interface detached) retires its device (design/RDMA-KERNEL-RECOVERY.md); the next pairing is a pause away */
static void port_lost(struct port *port,int64_t code,int retire){
  if(port->control>=0){close(port->control);port->control=-1;}
  if(port->greeted||port->connecting)fprintf(stderr,"link %u (%s): lost: code %lld%s\n",port->index,port->name,(long long)code,retire?", device retired":"");
  for(uint32_t s=0;s<MESH_CLIENTS;s++)if(port->flows[s]){
    if(port->flows[s]->state!=FAILED)flow_fail(port->flows[s],port->parted?ECONNRESET:code,4,0);
    else flow_release(port->flows[s]);
  }
  memset(port->pending,0,sizeof port->pending);
  port->greeted=port->parted=port->connecting=0;port->peer=0;port->held=0;
  port->retiring|=retire;port->next=now+PAUSE_NS;
  struct mesh_link_info *link=&mesh_links(M)[port->index];
  atomic_store_explicit(&link->peer,0,memory_order_relaxed);atomic_store_explicit(&link->code,code,memory_order_relaxed);
  atomic_store_explicit(&link->phase,retire?MESH_STOPPED:MESH_PAIRING,memory_order_release);
}
static void port_watch(struct port *port,int filter){
  struct kevent64_s event;EV_SET64(&event,(uint64_t)port->control,filter,EV_ADD|(filter==EVFILT_WRITE?EV_ONESHOT:0),0,0,port->index,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))port_lost(port,errno,0);
}
/* design/recovery.md#detection: the control socket's keepalive tells a side whose own link stays up (its peer's went
   down, its peer's host died); a send waits at most a second */
static void port_greet(struct port *port){
  int on=1,idle=1,interval=1,count=2,flags=fcntl(port->control,F_GETFL);
  struct timeval bound={1,0};
  if(flags<0 || fcntl(port->control,F_SETFL,flags&~O_NONBLOCK) || setsockopt(port->control,SOL_SOCKET,SO_KEEPALIVE,&on,sizeof on) ||
     setsockopt(port->control,IPPROTO_TCP,TCP_KEEPALIVE,&idle,sizeof idle) || setsockopt(port->control,IPPROTO_TCP,TCP_KEEPINTVL,&interval,sizeof interval) ||
     setsockopt(port->control,IPPROTO_TCP,TCP_KEEPCNT,&count,sizeof count) || setsockopt(port->control,SOL_SOCKET,SO_SNDTIMEO,&bound,sizeof bound) ||
     setsockopt(port->control,SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on)){port_lost(port,errno?errno:EIO,0);return;}
  port_watch(port,EVFILT_READ);
  port_say(port,HELLO,0,0,0,NULL);
}
static struct port *port_at(uint32_t interface){
  for(uint32_t p=0;p<port_count;p++)if(ports[p].interface==interface)return ports+p;
  return NULL;
}
/* design/bridge-and-ipc.md#ports: a port's turn: its device retired where its port went away, else opened, and while
   it has no peer its hello sent to every node on its link; a failed program's queue pairs released again */
static void port_turn(struct port *port){
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct flow *flow=port->flows[s];
    if(flow && flow->state==FAILED && flow->provider.qp_count)flow_release(flow);
    else if(flow && flow->state!=FAILED && flow->state!=RUNNING && now>flow->deadline)flow_fail(flow,ETIMEDOUT,4,1);
    else if(flow && flow->state!=FAILED && atomic_load(&flow->failed))flow_fail(flow,atomic_load(&flow->failed),atomic_load(&flow->domain),1);
  }
  if(port->control>=0 || now<port->next)return;
  port->next=now+PAUSE_NS;
  port->interface=if_nametoindex(port->name);
  if(port->retiring){
    for(uint32_t s=0;s<MESH_CLIENTS;s++)if(port->flows[s] && port->flows[s]->provider.qp_count)return;
    if(!down_device(&port->device)){fprintf(stderr,"link %u (%s): device retirement: %s\n",port->index,port->name,strerror(errno));return;}
    port->retiring=0;
  }
  struct mesh_link_info *link=&mesh_links(M)[port->index];
  if(device_up(&port->device,&wire,M) || ibv_query_gid(port->device.context,1,0,&port->device.gid)){
    port->retiring=port->device.context && errno!=ENETDOWN;
    atomic_store_explicit(&link->code,errno,memory_order_relaxed);atomic_store_explicit(&link->phase,MESH_STOPPED,memory_order_release);
    return;
  }
  atomic_store_explicit(&link->phase,MESH_PAIRING,memory_order_release);
  struct ipv6_mreq group={.ipv6mr_multiaddr=everyone,.ipv6mr_interface=port->interface};
  setsockopt(hello,IPPROTO_IPV6,IPV6_JOIN_GROUP,&group,sizeof group);
  setsockopt(hello,IPPROTO_IPV6,IPV6_MULTICAST_IF,&port->interface,sizeof port->interface);
  struct sockaddr_in6 to={.sin6_len=sizeof to,.sin6_family=AF_INET6,.sin6_port=htons((uint16_t)atoi(MESH_PORT)),.sin6_addr=everyone,.sin6_scope_id=port->interface};
  struct message said={.magic=XMAGIC,.kind=HELLO,.node=M->node,.boot=M->boot,.block=M->block*M->pgsz};
  sendto(hello,&said,sizeof said,0,(struct sockaddr *)&to,sizeof to);
}
/* design/bridge-and-ipc.md#ports: a node heard on a port: the lower node dials the higher's listener on the address
   it heard from, the higher answers its hello so the lower hears it at once */
static void hello_heard(void){
  struct message heard;struct sockaddr_in6 from;socklen_t length=sizeof from;
  while(recvfrom(hello,&heard,sizeof heard,0,(struct sockaddr *)&from,&length)==(ssize_t)sizeof heard){
    struct port *port=port_at(from.sin6_scope_id);
    length=sizeof from;
    if(!port || heard.magic!=XMAGIC || heard.kind!=HELLO || heard.node==M->node || port->control>=0 || !port->device.frame_capacity)continue;
    if(M->node>heard.node){
      struct message said={.magic=XMAGIC,.kind=HELLO,.node=M->node,.boot=M->boot,.block=M->block*M->pgsz};
      sendto(hello,&said,sizeof said,0,(struct sockaddr *)&from,sizeof from);
      continue;
    }
    from.sin6_port=htons((uint16_t)atoi(MESH_PORT));
    port->control=socket(AF_INET6,SOCK_STREAM,0);
    if(port->control<0 || fcntl(port->control,F_SETFL,O_NONBLOCK) ||
       (connect(port->control,(struct sockaddr *)&from,sizeof from) && errno!=EINPROGRESS)){port_lost(port,errno,0);continue;}
    port->peer=heard.node;port->connecting=1;port->next=now+PAIR_NS;
    port_watch(port,EVFILT_WRITE);
  }
}
static void control_heard(struct port *port){
  for(;;){
    ssize_t got=recv(port->control,port->buffer+port->held,sizeof port->buffer-port->held,MSG_DONTWAIT);
    if(got<0 && (errno==EAGAIN || errno==EINTR))return;
    if(got<=0){port_lost(port,got?errno:(port->parted?ECONNRESET:ENETRESET),0);return;}
    port->held+=(size_t)got;
    if(port->held<sizeof port->buffer)continue;
    port->held=0;
    struct message heard;memcpy(&heard,port->buffer,sizeof heard);
    port_heard(port,&heard);
    if(port->control<0)return;
  }
}
/* design/prepared-machine.md#M24: a port's link off or its interface detached */
static void network_heard(void){
  union mesh_network_event event;
  while(recv(network,event.bytes,sizeof event,0)>=(ssize_t)(KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data))){
    const struct net_event_data *interface=(const void *)(event.bytes+KEV_MSG_HEADER_SIZE);
    char name[IFNAMSIZ+16];
    snprintf(name,sizeof name,"%.*s%u",(int)sizeof interface->if_name,interface->if_name,interface->if_unit);
    for(uint32_t p=0;p<port_count;p++)if(!strcmp(name,ports[p].name)){
      if(event.header.event_code==KEV_DL_LINK_OFF || event.header.event_code==KEV_DL_IF_DETACHED)port_lost(ports+p,ENETDOWN,1);
      else if(event.header.event_code==KEV_DL_LINK_ON)ports[p].next=now;
    }
  }
}

/* design/bridge-and-ipc.md#clients: every client slot read: a client whose process exited is retired; a program that
   ended (stopped, its client gone) leaves its ports; a program started is taken onto each port it binds.  Rows of a
   retired client are released once no port holds a program of a client gone. */
static uint64_t owners[MESH_CLIENTS],accepted[MESH_CLIENTS];
static int unreleased;
static void clients_turn(int liveness){
  int changed=0,zombie=0;
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct mesh_client *client=(struct mesh_client *)((char *)M+M->client_off)+s;
    uint64_t owner=atomic_load(&client->owner);
    if(liveness && owner && !(owner&MESH_RETIRING) && kill((pid_t)(uint32_t)owner,0) && errno==ESRCH){mesh_retire(M,owner);owner=atomic_load(&client->owner);}
    changed|=owner!=owners[s];owners[s]=owner;
    uint64_t configured=atomic_load_explicit(&client->configured,memory_order_acquire);
    int held=0;
    for(uint32_t p=0;p<port_count;p++){
      struct flow *flow=ports[p].flows[s];
      if(flow && (flow->client!=owner || configured!=owner))flow_end(flow);
      held|=ports[p].flows[s]!=NULL;
      zombie|=ports[p].flows[s] && ports[p].flows[s]->client!=owner;
    }
    if(accepted[s] && (accepted[s]!=owner || configured!=owner) && !held)accepted[s]=0;
    if(!owner || (owner&MESH_RETIRING) || configured!=owner || accepted[s]==owner || held)continue;
    accepted[s]=owner;
    struct kevent64_s event;EV_SET64(&event,(uint32_t)owner,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,s,0,0);
    kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
    for(uint32_t p=0;p<MESH_PORTS;p++){
      if(!mesh_link_transfers(M,owner,p))continue;
      if(p<port_count)flow_create(ports+p,s,owner);
      else {
        struct mesh_port_info *status=&client->port[p];
        atomic_store(&status->code,ENODEV);atomic_store(&status->domain,4);atomic_store(&status->phase,MESH_STOPPED);
        atomic_store_explicit(&status->prepared,owner,memory_order_release);
      }
    }
  }
  unreleased|=changed;
  if(unreleased && !zombie){unreleased=0;mesh_retired_release(M);}
}

/* design/bridge-and-ipc.md#ports: every RDMA device of the node a port, those found since the bridge's start added */
static void ports_found(void){
  struct ibv_device **list=ibv_get_device_list(NULL);
  for(int i=0;list && list[i] && port_count<MESH_PORTS;i++){
    const char *name=ibv_get_device_name(list[i]);
    uint32_t known=0;
    for(uint32_t p=0;p<port_count;p++)known|=!strcmp(ports[p].device.name,name);
    if(known || strncmp(name,"rdma_",5))continue;
    struct port *port=ports+port_count;
    *port=(struct port){.index=port_count,.control=-1};
    snprintf(port->device.name,sizeof port->device.name,"%s",name);snprintf(port->name,sizeof port->name,"%s",name+5);
    snprintf(mesh_links(M)[port_count].device,sizeof mesh_links(M)[port_count].device,"%s",name);
    atomic_store(&mesh_links(M)[port_count].phase,MESH_STOPPED);
    port_count++;
  }
  if(list)ibv_free_device_list(list);
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

/* design/bridge-and-ipc.md: the node's resident bridge */
int main(int argc,char **argv){
  (void)argc;
  struct stat binary,current;
  if(stat(argv[0],&binary))memset(&binary,0,sizeof binary);
  uuid_t host;struct timespec bound={5,0};
  if(gethostuuid(host,&bound)){fprintf(stderr,"host identity: %s; this bridge's node is drawn at random\n",strerror(errno));arc4random_buf(host,sizeof host);}
  uint32_t node=0;
  for(int i=0;i<16;i++)node=(node*0x01000193u)^host[i];
  node=(node&0x7fffffffu)?node&0x7fffffffu:1;
  uint64_t ram=0;size_t rl=sizeof ram;sysctlbyname("hw.memsize",&ram,&rl,NULL,0);
  const uint32_t pg=(uint32_t)getpagesize();
  /* design/collective-dependency-ledger.md#d6-paired-send-and-receive-frame-counts-match */
  if((uint64_t)pg+4096>16773120)die("transport chunk exceeds native request capacity");
  struct hdr geometry={0};
  uint64_t length=mesh_layout(&geometry,pg,1,(uint32_t)(ram/pg));
  signal(SIGPIPE,SIG_IGN);signal(SIGINT,SIG_IGN);signal(SIGTERM,SIG_IGN);signal(SIGHUP,SIG_IGN);
  atexit(down);
  shm_unlink(MESH_NAME);int fd=shm_open(MESH_NAME,O_CREAT|O_RDWR,MESH_MODE);if(fd<0)die("shm");
  if(ftruncate(fd,(off_t)length))die("ftruncate");fchmod(fd,MESH_MODE);
  M=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  if(M==MAP_FAILED)die("mmap");shm=MESH_NAME;
  *M=geometry;M->node=node;M->version=MESH_VERSION;M->boot=((uint64_t)arc4random()<<32)|arc4random();
  for(uint32_t r=0;r<mesh_rows(M);r++)atomic_store_explicit(&mesh_page(M)[r].mapping,MESH_ABSENT,memory_order_relaxed);
  if(wire_map(&wire,M,fd))die("transport page aliases");
  close(fd);
  ports_found();
  events=kqueue();
  struct kevent64_s event;
  int on=1;
  struct sockaddr_in6 any={.sin6_len=sizeof any,.sin6_family=AF_INET6,.sin6_port=htons((uint16_t)atoi(MESH_PORT)),.sin6_addr=IN6ADDR_ANY_INIT};
  listener=socket(AF_INET6,SOCK_STREAM,0);hello=socket(AF_INET6,SOCK_DGRAM,0);network=socket(PF_SYSTEM,SOCK_RAW,SYSPROTO_EVENT);
  struct kev_request filter={KEV_VENDOR_APPLE,KEV_NETWORK_CLASS,KEV_DL_SUBCLASS};
  unsigned loop=0;
  if(events<0 || listener<0 || hello<0 || network<0 || setsockopt(listener,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on) ||
     setsockopt(hello,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on) || setsockopt(hello,IPPROTO_IPV6,IPV6_MULTICAST_LOOP,&loop,sizeof loop) ||
     bind(listener,(struct sockaddr *)&any,sizeof any) || listen(listener,MESH_PORTS) || bind(hello,(struct sockaddr *)&any,sizeof any) ||
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
  pthread_t watcher;
  if(pthread_create(&watcher,NULL,control_watch,NULL))die("control watcher");
  fprintf(stderr,"bridge node %u: %u ports, arena %llu pages, rows %u\n",node,port_count,(unsigned long long)mesh_arena_pages(M),M->rows);
  uint64_t liveness=0;
  while(!stop){
    now=clock_ns();
    for(uint32_t p=0;p<port_count;p++){
      struct port *port=ports+p;
      if((port->connecting || (port->control>=0 && !port->greeted)) && now>port->next)port_lost(port,ETIMEDOUT,0);
      port_turn(port);
    }
    int alive=now>=liveness;
    if(alive){
      liveness=now+PAUSE_NS;
      ports_found();
      if(binary.st_ino && (stat(argv[0],&current) || current.st_ino!=binary.st_ino || current.st_mtimespec.tv_sec!=binary.st_mtimespec.tv_sec))stop=1;
    }
    clients_turn(alive);
    struct kevent64_s ready[16];
    struct timespec tick={0,250000000};
    int count=kevent64(events,NULL,0,ready,16,0,&tick);
    now=clock_ns();
    for(int i=0;i<count;i++){
      struct kevent64_s *e=ready+i;
      if(e->filter==EVFILT_SIGNAL)stop=1;
      else if(e->filter==EVFILT_PROC && e->udata<MESH_CLIENTS){
        uint64_t owner=atomic_load(&((struct mesh_client *)((char *)M+M->client_off))[e->udata].owner);
        if((uint32_t)owner==(uint32_t)e->ident)mesh_retire(M,owner);
      }
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)listener){
        struct sockaddr_in6 from;socklen_t size=sizeof from;int accepted;
        while((accepted=accept(listener,(struct sockaddr *)&from,&size))>=0){
          struct port *port=port_at(from.sin6_scope_id);size=sizeof from;
          if(!port || port->control>=0 || !port->device.frame_capacity){close(accepted);continue;}
          port->control=accepted;port->next=now+PAIR_NS;
          port_greet(port);
        }
      }
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)hello)hello_heard();
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)network)network_heard();
      else if(e->udata<port_count && (int)e->ident==ports[e->udata].control){
        struct port *port=ports+e->udata;
        if(e->filter==EVFILT_WRITE){
          int error=0;socklen_t size=sizeof error;
          if(getsockopt(port->control,SOL_SOCKET,SO_ERROR,&error,&size))error=errno;
          if(error)port_lost(port,error,0);
          else {port->connecting=0;port_greet(port);}
        }
        else control_heard(port);
      }
    }
  }
  fprintf(stderr,"bridge node %u: stopping\n",node);
  for(uint32_t p=0;p<port_count;p++){
    port_say(ports+p,FAREWELL,0,0,0,NULL);
    ports[p].parted=1;
    port_lost(ports+p,ESHUTDOWN,1);
    for(uint32_t s=0;s<MESH_CLIENTS;s++)if(ports[p].flows[s]){flow_release(ports[p].flows[s]);free(ports[p].flows[s]);ports[p].flows[s]=NULL;}
    if(!down_device(&ports[p].device))fprintf(stderr,"link %u (%s): verbs teardown: %s\n",p,ports[p].name,strerror(errno));
  }
  atomic_store(&M->bridge_pid,0);
  mesh_control_notify(M);
  pthread_join(watcher,NULL);
  munmap(wire.data,wire.length);munmap(M,length);
  return 0;
}
