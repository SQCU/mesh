#include "mesh-recovery.h"
#include "mesh-call.h"
#include <pthread.h>
#include <sys/event.h>
#include <sys/ioctl.h>
#include <sys/kern_event.h>
#include <net/if.h>
#include <net/if_var.h>
#include <net/net_kev.h>

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
/* design/prepared-machine.md#M27 (mesh git e9b9a08^) */
struct mesh_trace { _Alignas(32) uint64_t identity; uint64_t begin,middle,end; };
_Static_assert(sizeof(struct mesh_trace)==32 && _Alignof(struct mesh_trace)==32,"M27");
struct mesh_link {
  pthread_t workers[2];
  uint32_t worker_count,publication_count,stream_count,index;
  pthread_t controller;
  int events,network;
  union mesh_network_event network_event;
  char *configuration;
  _Atomic int progressing;
  struct hdr *M;struct mesh_verbs provider;int qps,cyclic;uint64_t client;uint32_t invocations;
  struct prepared_receive *receive;
  size_t receive_count;
  struct receive_ring *rings;
  struct send_gate *gates;
  struct send_stream *streams;
  struct prepared_send *requests;
  size_t request_count;
  struct mesh_send *publications;
  struct ibv_wc *completion;
  struct mesh_cancellation *cancel;
  /* design/prepared-machine.md#M27 (mesh git e9b9a08^) */
  struct mesh_trace *trace[2];
  size_t traced[2],trace_capacity[2];
  int ledger;
  /* design/recovery.md: the link's loss and resumptions (mesh-recovery.h) */
  struct mesh_recovery recovery;
};
/* design/prepared-machine.md#M26 */
static _Atomic(struct hdr *) control_memory;
/* design/algorithm-sources.md#meshresult */
/* design/prepared-machine.md#M26 */
static void stop_bridge(int signal){
  int error=errno;
  onsig(signal);
  struct hdr *m=atomic_load_explicit(&control_memory,memory_order_relaxed);
  if(m)mesh_control_notify(m);
  errno=error;
}
/* design/algorithm-sources.md#meshresult */
/* design/prepared-machine.md#M12 */
static void link_stop(struct mesh_link *link){
  atomic_store_explicit(&link->progressing,0,memory_order_release);
  struct kevent64_s event;EV_SET64(&event,0,EVFILT_USER,0,NOTE_TRIGGER,0,0,0,0);
  kevent64(link->events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
}
/* design/algorithm-sources.md#programcopy */
static __attribute__((noinline)) void link_error(struct mesh_link *link,int64_t code,uint32_t domain){
  struct mesh_port_info *port=&mesh_links(link->M)[link->index].port;port->code=code;port->domain=domain;
  atomic_store_explicit(&port->phase,MESH_STOPPED,memory_order_relaxed);
  atomic_store_explicit(&port->prepared,link->client,memory_order_release);
  link_stop(link);
}
/* design/recovery.md#loss: a loss the link may survive suspends it (mesh_recovery_resume, in link_run), else it stops. */
static void link_lost(struct mesh_link *link,int64_t code,uint32_t domain){
  if(mesh_recovery_lose(&link->recovery,code,domain))link_stop(link);
  else link_error(link,code,domain);
}
/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M06 */
/* design/algorithm-sources.md#programcopy */
static int link_prepare(struct mesh_link *link){
  struct hdr *m=link->M;
  struct mesh_tx *tx=(void *)mesh_events(m,mesh_notice_queue(m,link->client,link->index));
  /* design/prepared-machine.md#M12 */
  link->cancel=tx->cancel?(void *)((char *)m+tx->cancel):NULL;
  link->publications=(void *)((char *)m+tx->cells);
  link->publication_count=tx->count*tx->slots+tx->once;
  size_t requests=0,records=0;
  link->qps=(int)(m->qps*tx->slots);
  for(uint32_t q=0;q<m->qps;q++)for(uint32_t d=0;d<2;d++){
    uint32_t channel=link->index*m->qps+q;
    struct mesh_transfer *transfers=mesh_transfers(m,link->client,channel,d);
    for(uint32_t i=0;i<atomic_load(mesh_order_length(m,link->client,channel,d));i++){
      size_t chunks=mesh_row_chunks(m,transfers[i].local_row,transfers[i].bytes),
        each=(size_t)transfers[i].count*mesh_transfer_active(transfers+i,tx->invocations);
      if(d==MESH_SEND)requests+=(chunks-1)*each;
      else records+=chunks*each;
    }
  }
  free(link->requests);
  /* design/prepared-machine.md#M29 */
  link->requests=requests?aligned_alloc(128,requests*sizeof *link->requests):NULL;
  if(requests&&!link->requests)return ENOMEM;
  link->request_count=requests;link->receive_count=records;
  /* design/prepared-machine.md#M27 (mesh git e9b9a08^) */
  const char *ledger=getenv("MESH_LEDGER");
  link->ledger=ledger && ledger[0] && !(ledger[0]=='0' && !ledger[1]);
  size_t events[2]={(size_t)link->publication_count*(tx->invocations?tx->invocations:1),records};
  for(int d=0;d<2;d++){
    free(link->trace[d]);link->trace[d]=NULL;link->traced[d]=0;
    link->trace_capacity[d]=link->ledger?events[d]:0;
    if(!link->trace_capacity[d])continue;
    link->trace[d]=aligned_alloc(32,(link->trace_capacity[d]+1)*sizeof(struct mesh_trace));
    if(!link->trace[d])return ENOMEM;
  }
  return 0;
}

/* The ring slot a transfer's invocation t addresses: (t - begin) mod its depth, at its slot pages. */
static uint32_t transfer_row(struct hdr *m,const struct mesh_transfer *transfer,uint32_t row,uint32_t invocation,uint32_t chunk){
  uint32_t depth=transfer->depth?transfer->depth:atomic_load_explicit(&m->depth,memory_order_relaxed);
  if(!depth)depth=1;
  return row+(uint32_t)((size_t)((invocation-transfer->begin)%depth)*transfer->invocation_pages/m->block)+chunk;
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
static int ring_advance(struct mesh_link *link,struct receive_ring *ring){
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
   completions (metal-microbench docs/kernels.md Transport (git 7300d1e^)). */
static int link_configure(void *state,int socket,uint64_t client){
  struct mesh_link *link=state;struct hdr *m=link->M;
  struct mesh_tx *tx=(void *)mesh_events(m,mesh_notice_queue(m,client,link->index));
  uint32_t invocations=tx->invocations?tx->invocations:1,depth=atomic_load_explicit(&m->depth,memory_order_acquire);
  if(!depth||depth>invocations)depth=invocations;
  link->cyclic=atomic_load_explicit(&m->cyclic,memory_order_acquire)!=0;link->invocations=invocations;
  int qps=link->qps;
  link->streams=calloc((size_t)qps,sizeof *link->streams);
  link->gates=aligned_alloc(64,(size_t)qps*sizeof *link->gates);
  link->rings=calloc((size_t)qps,sizeof *link->rings);
  link->receive=aligned_alloc(128,(link->receive_count+1)*sizeof *link->receive);
  struct mesh_send **last=calloc((size_t)qps,sizeof *last),**terminal=calloc((size_t)qps,sizeof *terminal);
  if(!link->streams||!link->gates||!link->rings||!link->receive||!last||!terminal){free(last);free(terminal);errno=ENOMEM;return -1;}
  /* a record or request is one frame at least, so a queue pair's frames also bound the completions
     it can have pending in the link's receive and SEND completion queues, which its pairs share */
  uint32_t receives=(uint32_t)link->provider.completion->cqe/(uint32_t)qps,sends=(uint32_t)link->provider.sent->cqe/(uint32_t)qps;
  for(int q=0;q<qps;q++){
    link->gates[q]=(struct send_gate){.capacity=MIN(link->provider.queues[q].send_capacity,sends)};
    link->rings[q]=(struct receive_ring){.completed=-1,.capacity=MIN(link->provider.queues[q].receive_capacity,receives),.span=UINT32_MAX};
  }
  /* design/prepared-machine.md#M29 */
  size_t next=0;
  for(uint32_t q=0;q<m->qps;q++){
    uint32_t channel=link->index*m->qps+q,count=atomic_load(mesh_order_length(m,client,channel,MESH_SEND));
    struct mesh_transfer *out=mesh_transfers(m,client,channel,MESH_SEND);
    /* each cell's chain over its invocation's ring slot */
    for(uint32_t i=0;i<count;i++){
      uint32_t active=mesh_transfer_active(out+i,invocations),column=active+1;
      for(uint32_t slot=0;slot<out[i].count;slot++){
        uint32_t row=out[i].local_row+slot*out[i].stride,chunks=mesh_row_chunks(m,row,out[i].bytes),stream=q*tx->slots+slot;
        struct mesh_send *cells=link->publications+out[i].first+(size_t)slot*column;
        /* design/prepared-machine.md#M27 (mesh git e9b9a08^) */
        if(link->ledger)fprintf(stderr,"{\"trace_binding\":%u,\"rank\":%u,\"direction\":0,\"binding\":%u,\"slot\":%u,\"begin\":%u,\"active\":%u,\"cells\":%llu}\n",
          link->index,m->node,out[i].binding,slot,out[i].begin,active,(unsigned long long)(uintptr_t)cells);
        for(uint32_t u=0;u<active;u++){
          struct mesh_send *cell=cells+u;
          cell->pair=(uintptr_t)link->provider.queues[stream].pair;cell->queue=stream;cell->chunks=chunks;
          uint64_t remaining=out[i].bytes;
          for(uint32_t k=0;k<chunks;k++){
            uint32_t address=transfer_row(m,out+i,row,out[i].begin+u,k);
            uint64_t offset=atomic_load_explicit(&mesh_page(m)[address].address,memory_order_relaxed)-m->data_off;
            struct ibv_sge span=wire_span(link->provider.device,offset,remaining);
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
    for(uint32_t t=0;t<invocations;t++)for(uint32_t i=0;i<count;i++){
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
  for(int q=0;q<qps;q++)if(last[q])last[q]->successor=link->cyclic?(uintptr_t)link->streams[q].cell:(uintptr_t)terminal[q];
  for(int q=0;q<qps;q++){link->streams[q].first=link->streams[q].cell;link->streams[q].cycle=0;}
  link->stream_count=0;
  for(int q=0;q<qps;q++)if(link->streams[q].cell)link->streams[link->stream_count++]=link->streams[q];
  free(last);free(terminal);
  /* design/prepared-machine.md#M08 */
  /* each queue pair's ring: its records, invocation by invocation, the transfers of each in binding
     order, a transfer's chunks in order */
  size_t at=0;
  for(int ring=0;ring<qps;ring++){
    uint32_t q=(uint32_t)ring/tx->slots,slot=(uint32_t)ring%tx->slots,channel=link->index*m->qps+q;
    uint32_t count=atomic_load(mesh_order_length(m,client,channel,MESH_RECEIVE));
    struct mesh_transfer *in=mesh_transfers(m,client,channel,MESH_RECEIVE);
    struct receive_ring *r=link->rings+ring;
    r->first=link->receive+at;
    size_t *base=calloc((size_t)count+1,sizeof *base),*firsts=NULL;
    for(uint32_t i=0;base&&i<count;i++)base[i+1]=base[i]+(slot<in[i].count?mesh_transfer_active(in+i,invocations):0);
    if(base)firsts=malloc((base[count]+1)*sizeof *firsts);
    if(!base||!firsts){free(base);free(firsts);errno=ENOMEM;return -1;}
    for(uint32_t i=0;i<count;i++){
      if(slot>=in[i].count)continue;
      uint32_t active=mesh_transfer_active(in+i,invocations),own=in[i].depth?in[i].depth:depth;
      /* receive u+span lands in slot (u+span) mod depth while the local reader may still be at u:
         the ring's span is held a slot short of the ring; a transfer whose every invocation has its
         own slot bounds nothing */
      if(own<active)r->span=MIN(r->span,own>1?own-1:1);
      /* design/prepared-machine.md#M27 (mesh git e9b9a08^) */
      if(link->ledger)fprintf(stderr,"{\"trace_binding\":%u,\"rank\":%u,\"direction\":1,\"ring\":%d,\"binding\":%u,\"slot\":%u,\"begin\":%u,\"active\":%u,\"chunks\":%u}\n",
        link->index,m->node,ring,in[i].binding,slot,in[i].begin,active,mesh_row_chunks(m,in[i].local_row+slot*in[i].stride,in[i].bytes));
    }
    for(uint32_t t=0;t<invocations;t++)for(uint32_t i=0;i<count;i++){
      uint32_t active=mesh_transfer_active(in+i,invocations);
      if(slot>=in[i].count || t<in[i].begin || t-in[i].begin>=active)continue;
      uint32_t u=t-in[i].begin,row=in[i].local_row+slot*in[i].stride,chunks=mesh_row_chunks(m,row,in[i].bytes);
      uint64_t remaining=in[i].bytes;
      for(uint32_t k=0;k<chunks;k++){
        uint32_t address=transfer_row(m,in+i,row,t,k);
        uint64_t offset=atomic_load_explicit(&mesh_page(m)[address].address,memory_order_relaxed)-m->data_off;
        struct ibv_sge span=wire_span(link->provider.device,offset,remaining);
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
    for(uint32_t i=0;i<count;i++){
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
    int error=ring_advance(link,r);
    if(error){errno=error;return -1;}
    fprintf(stderr,"receive ring=%d records=%zu posted=%zu frames=%u capacity=%u span=%u\n",
      ring,r->count,r->posted,r->outstanding,r->capacity,r->span);
  }
  if(at)link->receive[at]=link->receive[at-1];
  uint32_t posted=1,peer_posted;
  /* design/prepared-machine.md#M27 (mesh git e9b9a08^) */
  if(link->ledger)fprintf(stderr,"{\"trace_layout\":%u,\"rank\":%u,\"send_base\":%llu,\"receive_base\":%llu,\"invocations\":%u,\"send_capacity\":%zu,\"receive_capacity\":%zu}\n",
    link->index,m->node,(unsigned long long)(uintptr_t)link->publications,(unsigned long long)(uintptr_t)link->receive,
    invocations,link->trace_capacity[MESH_SEND],link->trace_capacity[MESH_RECEIVE]);
  return exchange(socket,&posted,&peer_posted,sizeof posted,sizeof peer_posted,m,client,link->provider.deadline);
}

/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M06 */
/* design/prepared-machine.md#M15 */
/* design/algorithm-sources.md#independent-native-queues */
/* The link's SEND completions: each retires its queue's frames through its request (the device
   completes every SEND, signaled or not).  0, or an error completion's status. */
static __attribute__((always_inline)) inline int send_retire(struct mesh_link *link){
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

/* Each stream's cells in order: a cell released by its producer posts its chain's requests while its
   queue pair holds their frames (the gate: frames posted less those its SEND completions retired;
   ibv_post_send accepts past the queue's frames and corrupts later, so the gate is the caller's).
   The thread takes its completions when a stream waits for its producer or its gate. */
static __attribute__((always_inline)) inline void *link_send_drain(struct mesh_link *link,int traced){
  struct send_stream *streams=link->streams;
  uint32_t count=link->stream_count;
  struct ibv_send_wr *bad;
  int (*post)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **)=link->provider.queues[0].send;
  /* design/prepared-machine.md#M27 (mesh git e9b9a08^) */
  struct mesh_trace *trace=traced?link->trace[MESH_SEND]:NULL;
  size_t capacity=traced?link->trace_capacity[MESH_SEND]:0;
  for(uint32_t s=0;;s=s+1==count?0:s+1){
    struct send_stream *stream=streams+s;
    uint64_t observed=0;
    if(!stream->remaining){
      struct mesh_send *cell=stream->cell;
      if(atomic_load_explicit(&cell->ready,memory_order_acquire)<=stream->cycle){
        if(!atomic_load_explicit(&link->progressing,memory_order_acquire))return NULL;
        int error=send_retire(link);
        if(error){link_lost(link,error,2);return NULL;}
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
        if(error){link_lost(link,error,2);return NULL;}
        break;
      }
      gate->posted+=frames;
      /* posted as a copy: the cell's first request keeps the producer's argument in its wr_id (mesh_transfers_prepare),
         which a host producer reads at every publication (mesh_host_publish), and the copy's names the gate */
      struct ibv_send_wr posting=*request;
      posting.next=NULL;posting.send_flags=IBV_SEND_SIGNALED;
      posting.wr_id=((uint64_t)cell->queue<<48)|(gate->posted&MESH_GATE_MASK);
      int error=post(link->provider.queues[cell->queue].pair,&posting,&bad);
      if(error){link_lost(link,error<0?-error:error,1);return NULL;}
      stream->next=request->next;stream->remaining--;gate->requests++;
    }
    if(traced && observed && link->traced[MESH_SEND]<capacity)
      trace[link->traced[MESH_SEND]++]=(struct mesh_trace){(uintptr_t)cell,observed,posting,clock_gettime_nsec_np(CLOCK_UPTIME_RAW)};
    if(!stream->remaining){
      stream->cell=(struct mesh_send *)cell->successor;
      if(stream->cell==stream->first)stream->cycle++;
    }
  }
}

/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#independent-native-queues */
static void *link_send_progress(void *argument){
  struct mesh_link *link=argument;
  pthread_setname_np("mesh.rdma.send");
  /* design/prepared-machine.md#M27 (mesh git e9b9a08^) */
  if(link->trace[MESH_SEND])return link_send_drain(link,1);
  return link_send_drain(link,0);
}

/* design/prepared-machine.md#M08 */
/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#programcopy */
/* The link's receive completions: each stores its completion word and posts the ring's next records. */
static __attribute__((always_inline)) inline void *link_receive_drain(struct mesh_link *link,int traced){
  struct mesh_queue queue=link->provider.queues[0];
  struct ibv_wc *completion=link->completion;
  /* design/prepared-machine.md#M27 (mesh git e9b9a08^) */
  struct mesh_trace *trace=traced?link->trace[MESH_RECEIVE]:NULL;
  size_t capacity=traced?link->trace_capacity[MESH_RECEIVE]:0;
  for(;;){
    int count=queue.poll(queue.completion,1,completion);
    if(count<0){link_lost(link,count,3);return NULL;}
    if(!count){
      if(!atomic_load_explicit(&link->progressing,memory_order_acquire))return NULL;
      continue;
    }
    uint64_t status_opcode;
    memcpy(&status_opcode,&completion->status,sizeof status_opcode);
    if((uint32_t)status_opcode){link_lost(link,(uint32_t)status_opcode,2);return NULL;}
    if(!((status_opcode>>32)&IBV_WC_RECV))continue;
    uint64_t polled=traced?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
    struct prepared_receive *record=(void *)(uintptr_t)completion->wr_id;
    struct receive_ring *ring=link->rings+record->queue;
    ring->landed++;ring->bytes+=completion->byte_len;
    atomic_store_explicit(record->input,record->argument+record->cycle,memory_order_release);
    uint64_t published=traced?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
    ring->outstanding-=record->frames;ring->completed=(int64_t)(record->invocation+(uint64_t)record->cycle*link->invocations);
    int error=ring_advance(link,ring);
    if(error){link_lost(link,error,1);return NULL;}
    /* design/prepared-machine.md#M27 (mesh git e9b9a08^) */
    if(traced && link->traced[MESH_RECEIVE]<capacity)
      trace[link->traced[MESH_RECEIVE]++]=(struct mesh_trace){(uintptr_t)record,polled,published,clock_gettime_nsec_np(CLOCK_UPTIME_RAW)};
  }
}

/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#programcopy */
static void *link_receive_progress(void *argument){
  struct mesh_link *link=argument;
  pthread_setname_np("mesh.rdma.receive");
  /* design/prepared-machine.md#M27 (mesh git e9b9a08^) */
  if(link->trace[MESH_RECEIVE])return link_receive_drain(link,1);
  return link_receive_drain(link,0);
}

/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#meshresult */
static void link_close(struct mesh_link *link,int *control){
  atomic_store_explicit(&link->progressing,0,memory_order_release);
  if(link->network>=0){close(link->network);link->network=-1;}
  /* design/recovery.md#farewell */
  if(*control>=0){mesh_recovery_farewell(*control);shutdown(*control,SHUT_RDWR);}
  while(link->worker_count)pthread_join(link->workers[--link->worker_count],NULL);
  if(link->cancel)mesh_cancel(link->M,link->cancel,link->index);
  /* design/prepared-machine.md#M27 (mesh git e9b9a08^) */
  for(int d=0;d<2;d++){
    for(size_t i=0;i<link->traced[d];i++){
      struct mesh_trace t=link->trace[d][i];
      fprintf(stderr,"{\"native_trace\":%u,\"direction\":%d,\"event\":%zu,\"identity\":%llu,\"ns\":[%llu,%llu,%llu]}\n",
        link->index,d,i,(unsigned long long)t.identity,(unsigned long long)t.begin,(unsigned long long)t.middle,(unsigned long long)t.end);
    }
    free(link->trace[d]);link->trace[d]=NULL;link->traced[d]=0;link->trace_capacity[d]=0;
  }
  /* the link's census: each queue pair's requests and frames sent and retired by the send completions
     it took, and its ring's records landed and their bytes */
  for(int q=0;link->gates && link->rings && q<link->qps;q++)
    fprintf(stderr,"link %u census queue=%d requests=%llu frames=%llu retired=%llu send_completions=%llu records=%zu posted=%zu landed=%llu bytes=%llu\n",
      link->index,q,(unsigned long long)link->gates[q].requests,(unsigned long long)link->gates[q].posted,
      (unsigned long long)link->gates[q].retired,(unsigned long long)link->gates[q].completions,
      link->rings[q].count,link->rings[q].posted,(unsigned long long)link->rings[q].landed,
      (unsigned long long)link->rings[q].bytes);
  if(*control>=0){close(*control);*control=-1;}
  while(!down_pair(&link->provider))link_error(link,errno?errno:EIO,1);
  if(link->provider.listener>=0){close(link->provider.listener);link->provider.listener=-1;}
  free(link->receive);link->receive=NULL;
  free(link->streams);link->streams=NULL;free(link->gates);link->gates=NULL;free(link->rings);link->rings=NULL;
  atomic_store_explicit(&mesh_links(link->M)[link->index].port.phase,MESH_STOPPED,memory_order_release);
}

/* design/prepared-machine.md#M08 */
/* The link's progress threads: its receives', and its SENDs' where it publishes.  0, or pthread_create's error. */
static int link_workers(struct mesh_link *link){
  void *(*progress[2])(void *)={link_receive_progress,link_send_progress};
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_set_qos_class_np(&attributes,QOS_CLASS_USER_INTERACTIVE,0);
  int error=0;
  for(uint32_t d=0;d<1+(link->publication_count!=0) && !error;d++){
    error=pthread_create(&link->workers[d],&attributes,progress[d],link);
    if(!error)link->worker_count++;
  }
  pthread_attr_destroy(&attributes);
  return error;
}

/* design/recovery.md#resumption */
/* A paired link's resumption on new queue pairs (verbs_up's configuration, link_pair): every record and request
   stays where link_configure laid it.  Each ring's records landed in the order posted, so the peer's SENDs on a queue
   pair resume at the record its ring landed last, and this link's at the peer's: each side tells the other its rings'
   landed counts (mesh_recovery_exchange), posts its rings again from their own, and moves each SEND stream back to
   the peer's count of its queue pair, mid-cell where a cell's chain had landed in part (its producer released it);
   the barrier after the posting (mesh_recovery_barrier) holds every SEND until the peer's receives are posted.  0, or
   -1 with errno. */
static int link_reconfigure(void *state,int socket,uint64_t client){
  struct mesh_link *link=state;struct hdr *m=link->M;
  int qps=link->qps;
  uint32_t receives=(uint32_t)link->provider.completion->cqe/(uint32_t)qps,sends=(uint32_t)link->provider.sent->cqe/(uint32_t)qps;
  uint64_t *mine=calloc((size_t)qps,sizeof *mine),*peer=calloc((size_t)qps,sizeof *peer);
  if(!mine||!peer){free(mine);free(peer);errno=ENOMEM;return -1;}
  for(int q=0;q<qps;q++){
    struct send_gate *gate=link->gates+q;
    gate->posted=gate->retired=0;gate->capacity=MIN(link->provider.queues[q].send_capacity,sends);
    struct receive_ring *ring=link->rings+q;
    for(size_t i=0;i<ring->count;i++)ring->first[i].pair=link->provider.queues[q].pair;
    ring->posted=ring->landed;ring->outstanding=0;
    ring->capacity=MIN(link->provider.queues[q].receive_capacity,receives);
    mine[q]=ring->landed;
  }
  if(mesh_recovery_exchange(socket,qps,mine,peer,m,client,link->provider.deadline)){int error=errno;free(mine);free(peer);errno=error;return -1;}
  for(uint32_t s=0;s<link->stream_count;s++){
    struct send_stream *stream=link->streams+s;
    struct mesh_send *cell=stream->first;
    uint64_t left=peer[cell->queue],cycle=0;
    while(left && left>=cell->chunks){
      left-=cell->chunks;cell=(struct mesh_send *)cell->successor;
      if(cell==stream->first)cycle++;
    }
    stream->cell=cell;stream->cycle=cycle;stream->remaining=0;
    if(left){
      struct ibv_send_wr *request=&cell->request;
      for(uint64_t k=0;k<left;k++)request=request->next;
      stream->next=request;stream->remaining=(uint32_t)(cell->chunks-left);
    }
  }
  fprintf(stderr,"link %u resumes:",link->index);
  for(int q=0;q<qps;q++)fprintf(stderr," queue %d landed %llu, the peer's %llu",q,(unsigned long long)mine[q],(unsigned long long)peer[q]);
  fprintf(stderr,"\n");
  free(mine);free(peer);
  for(int q=0;q<qps;q++){int error=ring_advance(link,link->rings+q);if(error){errno=error;return -1;}}
  return mesh_recovery_barrier(socket,m,client,link->provider.deadline);
}

/* design/recovery.md#suspension: the transport's side of a resumption (struct mesh_recovery_transport) */
static void link_halt(void *state){
  struct mesh_link *link=state;
  while(link->worker_count)pthread_join(link->workers[--link->worker_count],NULL);
  struct ibv_wc done[16];int count;
  while(link->provider.completion && (count=ibv_poll_cq(link->provider.completion,16,done))>0)
    for(int i=0;i<count;i++){
      if(done[i].status || !(done[i].opcode&IBV_WC_RECV))continue;
      struct prepared_receive *record=(void *)(uintptr_t)done[i].wr_id;
      struct receive_ring *ring=link->rings+record->queue;
      ring->landed++;ring->bytes+=done[i].byte_len;
      atomic_store_explicit(record->input,record->argument+record->cycle,memory_order_release);
    }
}
static int link_release(void *state,int *control){
  struct mesh_link *link=state;
  if(*control>=0){close(*control);*control=-1;}
  if(link->provider.listener>=0){close(link->provider.listener);link->provider.listener=-1;}
  return down_pair(&link->provider)?0:EIO;
}
static int link_pair(void *state,uint64_t window){
  struct mesh_link *link=state;
  link->provider.window=window;
  int f=verbs_up(&link->provider,link->M,link->qps,link_reconfigure,link,link->client);
  link->provider.window=0;
  return f;
}
static int link_abandoned(void *state){
  struct mesh_link *link=state;struct hdr *m=link->M;
  return stop || atomic_load_explicit(&m->client,memory_order_acquire)!=link->client || kill((pid_t)(uint32_t)link->client,0) ||
    (link->cancel && atomic_load_explicit(&link->cancel->requested,memory_order_acquire));
}
static int link_watch(void *state,int control){
  struct mesh_link *link=state;
  struct kevent64_s event;
  EV_SET64(&event,control,EVFILT_READ,EV_ADD|EV_CLEAR,0,0,0,0,0);
  if(kevent64(link->events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))return errno;
  /* the link's events of the outage are past: the port is active again */
  while(link->network>=0 && recv(link->network,link->network_event.bytes,sizeof link->network_event,0)>0){}
  return 0;
}
static int link_restart(void *state){
  struct mesh_link *link=state;
  atomic_store_explicit(&link->progressing,1,memory_order_release);
  return link_workers(link);
}

/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M24 */
/* design/algorithm-sources.md#programcopy */
/* design/algorithm-sources.md#meshresult */
static void *link_run(void *argument){
  struct mesh_link *link=argument;struct hdr *m=link->M;
  struct mesh_port_info *port=&mesh_links(m)[link->index].port;
  uint32_t transfers=0;
  for(uint32_t q=0;q<m->qps;q++)for(int d=0;d<2;d++)transfers+=atomic_load(mesh_order_length(m,link->client,link->index*m->qps+q,d));
  struct kevent64_s event;
  EV_SET64(&event,(uint32_t)link->client,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,0,0,0);
  int error=kevent64(link->events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)?errno:0,control=-1;
  if(error){link_error(link,error,4);if(error==ESRCH)mesh_retire(m,link->client);}
  if(!error && transfers){
    link->network=socket(PF_SYSTEM,SOCK_RAW,SYSPROTO_EVENT);
    struct kev_request filter={KEV_VENDOR_APPLE,KEV_NETWORK_CLASS,KEV_DL_SUBCLASS};
    if(link->network<0 || fcntl(link->network,F_SETFL,O_NONBLOCK)<0 ||
       ioctl(link->network,SIOCSKEVFILT,&filter))error=errno;
    if(!error){
      EV_SET64(&event,link->network,EVFILT_READ,EV_ADD,0,0,0,0,0);
      if(kevent64(link->events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))error=errno;
    }
    if(error)link_error(link,error,4);
  }
  if(!error && transfers){
    atomic_store_explicit(&port->phase,MESH_PAIRING,memory_order_release);
    /* design/recovery.md#first: a port not yet active (a link down at the call's start) is tried again, a pause
       apart, within the pairing window */
    const uint64_t window=clock_gettime_nsec_np(CLOCK_MONOTONIC)+(link->provider.window?link->provider.window:pair_window_ns());
    /* design/recovery.md#first */
    link->provider.arrival=clock_gettime_nsec_np(CLOCK_MONOTONIC)+peer_window_ns();
    for(;;){
      link->provider.window=window-clock_gettime_nsec_np(CLOCK_MONOTONIC);
      control=verbs_up(&link->provider,m,link->qps,link_configure,link,link->client);
      if(control>=0 || !mesh_recovery_first_again(errno) || stop ||
         atomic_load_explicit(&m->client,memory_order_acquire)!=link->client ||
         clock_gettime_nsec_np(CLOCK_MONOTONIC)+(uint64_t)mesh_recovery_pause_ms()*1000000ull>=window)break;
      fprintf(stderr,"link %u: port not active at pairing; again in %d ms\n",link->index,mesh_recovery_pause_ms());
      poll(NULL,0,mesh_recovery_pause_ms());
    }
    link->provider.window=0;
    link->provider.arrival=0;
    /* design/recovery.md#detection */
    if(control>=0 && (error=mesh_recovery_keepalive(control))){close(control);control=-1;errno=error;}
    if(control<0)link_error(link,errno?errno:EIO,1);
    else {
      EV_SET64(&event,control,EVFILT_READ,EV_ADD|EV_CLEAR,0,0,0,0,0);
      error=kevent64(link->events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)?errno:0;
      if(!error){
        __atomic_store_n(&mesh_links(m)[link->index].bandwidth,link->provider.bandwidth,__ATOMIC_RELAXED);

      }
      if(!error)error=link_workers(link);
      if(error)link_error(link,error,1);
      else {
        atomic_store_explicit(&port->phase,MESH_PAIRED,memory_order_relaxed);
        atomic_store_explicit(&port->prepared,link->client,memory_order_release);
      }
    }
  } else if(!transfers){
    atomic_store_explicit(&port->phase,MESH_PAIRED,memory_order_release);
    atomic_store_explicit(&link->progressing,0,memory_order_release);
  }
  for(;;){
    /* design/recovery.md#suspension */
    if(mesh_recovery_pending(&link->recovery) && !stop && atomic_load_explicit(&m->client,memory_order_acquire)==link->client){
      const struct mesh_recovery_transport transport={link,link_halt,link_release,link_pair,link_abandoned,link_watch,link_restart};
      if(!mesh_recovery_resume(&link->recovery,&transport,&port->phase,link->index,&control))continue;
      link_error(link,mesh_recovery_taken(&link->recovery),link->recovery.domain);
    }
    if(!atomic_load_explicit(&link->progressing,memory_order_acquire))link_close(link,&control);
    if(stop || atomic_load_explicit(&m->client,memory_order_acquire)!=link->client ||
       atomic_load_explicit(&m->configured,memory_order_acquire)!=link->client)break;
    int count=kevent64(link->events,NULL,0,&event,1,0,NULL);
    if(count<0){if(errno==EINTR)continue;link_error(link,errno,4);break;}
    if(event.flags&EV_ERROR)link_error(link,event.data,4);
    else if(event.filter==EVFILT_PROC){link_error(link,ECANCELED,4);mesh_retire(m,link->client);}
    else if(event.filter==EVFILT_READ && event.ident==(uint64_t)link->network){
      ssize_t length=recv(link->network,link->network_event.bytes,sizeof link->network_event,0);
      if(length<0){
        if(errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR)link_error(link,errno,4);
      } else if((size_t)length>=KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data)){
        const struct kern_event_msg *notification=&link->network_event.header;
        if(notification->event_code==KEV_DL_LINK_OFF || notification->event_code==KEV_DL_IF_DETACHED){
          const struct net_event_data *interface=(const void *)(link->network_event.bytes+KEV_MSG_HEADER_SIZE);
          char device[sizeof interface->if_name+16];
          snprintf(device,sizeof device,"rdma_%.*s%u",(int)sizeof interface->if_name,interface->if_name,interface->if_unit);
          if(!strcmp(device,link->provider.device->name)){
            fprintf(stderr,"link unavailable: %s event=%u\n",device,notification->event_code);
            link_lost(link,ENETDOWN,4);
          }
        }
      }
    } else if(event.filter==EVFILT_READ){
      /* design/recovery.md#farewell: the control socket: the peer's farewell is an orderly end, its silence or reset a loss */
      if(mesh_recovery_parted((int)event.ident))link_error(link,ECONNRESET,4);
      else link_lost(link,event.flags&EV_EOF?(event.fflags?event.fflags:ECONNRESET):EPROTO,4);
    }
  }
  link_close(link,&control);
  EV_SET64(&event,(uint32_t)link->client,EVFILT_PROC,EV_DELETE,0,0,0,0,0);
  kevent64(link->events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
  return NULL;
}

/* design/algorithm-sources.md#programcopy */
int main(int argc,char **argv){
  const char *name=MESH_NAME;int me=0,layout=0;double pct=0;
  uint64_t arena_pages=0,block_pages=0,table_rows=0,window_pages=0,orders=4096;
  uint32_t link_count=0,device_count=0,qps=getenv("MESH_QPS")?(uint32_t)atoi(getenv("MESH_QPS")):1;
  struct mesh_link *links=aligned_alloc(_Alignof(struct mesh_link),(size_t)argc*sizeof *links);
  struct mesh_device *devices=calloc((size_t)argc,sizeof *devices);
  if(!links || !devices)die("bridge configuration allocation");
  memset(links,0,(size_t)argc*sizeof *links);
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-I") && i+1<argc)me=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-M") && i+1<argc)pct=atof(argv[++i]);
    /* design/prepared-machine.md#M09 */
    /* -A addressable arena pages, -B pages per block, -R page-table rows, -W registered window
       pages, -O transfer-list entries per queue and direction.  None of the five is derived from
       another; -W defaults to the whole arena and -O to 4096. */
    else if((!strcmp(argv[i],"-A") || !strcmp(argv[i],"-B") || !strcmp(argv[i],"-R") ||
             !strcmp(argv[i],"-W") || !strcmp(argv[i],"-O")) && i+1<argc){
      char kind=argv[i][1],*end;uint64_t pages=strtoull(argv[++i],&end,10);
      if(*end || !pages || pages>INT32_MAX)die("configured page count");
      if(kind=='A')arena_pages=pages;else if(kind=='B')block_pages=pages;
      else if(kind=='R')table_rows=pages;else if(kind=='W')window_pages=pages;else orders=pages;
    }
    else if(!strcmp(argv[i],"--layout"))layout=1;
    else if(!strcmp(argv[i],"-s") && i+1<argc)name=argv[++i];
    else if(!strcmp(argv[i],"--link") && i+1<argc){
      struct mesh_link *link=&links[link_count];
      char *fields=strdup(argv[++i]);link->configuration=fields;
      if(!fields)die("link configuration allocation");
      char *device=strsep(&fields,","),*peer=strsep(&fields,","),*local=strsep(&fields,","),*remote=strsep(&fields,","),*service=strsep(&fields,",");
      if(!device || !*device || !peer || !*peer || !local || !*local || !remote || !*remote || fields)die("link requires device,peer,local-address,remote-address[,service]");
      uint32_t d=0;
      while(d<device_count && strcmp(devices[d].name,device))d++;
      if(d==device_count){devices[d].name=device;pthread_mutex_init(&devices[d].setup,NULL);device_count++;}
      link->index=link_count++;link->provider=(struct mesh_verbs){.device=&devices[d],.peer=(uint32_t)strtoul(peer,NULL,10),.local_address=local,.remote_address=remote,.service=service?service:MESH_PORT,.listener=-1};
    }
    else die("unknown bridge option");
  }
  if(me<0 || !isfinite(pct) || pct<0 || pct>100 || !arena_pages || !block_pages || arena_pages<block_pages || !qps || qps>MESH_QPS)die("bridge geometry");
  const uint32_t pg=(uint32_t)getpagesize();
  /* design/collective-dependency-ledger.md#d6-paired-send-and-receive-frame-counts-match */
  if((uint64_t)block_pages*pg+4096>16773120)die("transport chunk exceeds native request capacity");
  struct hdr geometry={0};
  uint64_t length=mesh_layout(&geometry,(struct mesh_geometry){.pgsz=pg,.block=(uint32_t)block_pages,
    .pages=(uint32_t)arena_pages,.rows=(uint32_t)table_rows,.orders=(uint32_t)orders,
    .links=link_count,.qps=qps,.wire_pages=(uint32_t)window_pages});
  uint64_t ram=0;size_t rl=sizeof ram;sysctlbyname("hw.memsize",&ram,&rl,NULL,0);
  if(pct && length>(uint64_t)(pct/100*(double)ram))die("configured graph exceeds page capacity");
  if(layout){printf("%llu\n",(unsigned long long)length);return 0;}
  atexit(down);struct sigaction sa={0};sa.sa_handler=stop_bridge;
  sigaction(SIGINT,&sa,NULL);sigaction(SIGTERM,&sa,NULL);sigaction(SIGHUP,&sa,NULL);signal(SIGPIPE,SIG_IGN);
  shm_unlink(name);int fd=shm_open(name,O_CREAT|O_RDWR,MESH_MODE);if(fd<0)die("shm");
  if(ftruncate(fd,(off_t)length))die("ftruncate");fchmod(fd,MESH_MODE);
  struct hdr *m=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  if(m==MAP_FAILED)die("mmap");shm=name;
  *m=geometry;m->node=(uint32_t)me;m->version=MESH_VERSION;
  for(uint32_t r=0;r<mesh_rows(m);r++)atomic_store_explicit(&mesh_page(m)[r].mapping,MESH_ABSENT,memory_order_relaxed);
  struct mesh_wire wire={0};
  if(wire_map(&wire,m,fd))die("transport page aliases");
  close(fd);
  for(uint32_t i=0;i<link_count;i++){
    struct mesh_link *link=&links[i];
    link->M=m;link->provider.wire=&wire;link->network=-1;
    link->events=kqueue();
    struct kevent64_s event;EV_SET64(&event,0,EVFILT_USER,EV_ADD|EV_CLEAR,0,0,0,0,0);
    if(link->events<0 || kevent64(link->events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("link control events");
    mesh_links(m)[i].peer=link->provider.peer;
    snprintf(mesh_links(m)[i].device,sizeof mesh_links(m)[i].device,"%s",link->provider.device->name);
    atomic_store(&mesh_links(m)[i].port.phase,MESH_PAIRING);
  }
  int status=0;
  /* design/prepared-machine.md#M11 */
  struct ibv_wc *completion_outputs=calloc(link_count?link_count:1,sizeof *completion_outputs);
  if(!completion_outputs)die("completion output allocation");
  for(uint32_t p=0;p<link_count;p++)links[p].completion=completion_outputs+p;
  /* design/prepared-machine.md#M26 */
  atomic_store_explicit(&control_memory,m,memory_order_relaxed);
  atomic_store(&m->bridge_pid,(uint64_t)getpid());atomic_store(&m->port.phase,MESH_PAIRING);
  __sync_synchronize();m->magic=MESH_MAGIC;
  fprintf(stderr,"bridge node %d: %u links, %u queue pairs per link, arena %llu pages, window %u pages, rows %u, orders %u\n",
    me,link_count,qps,(unsigned long long)mesh_arena_pages(m),m->wire_pages,m->rows,m->orders);
  while(!stop){
    uint64_t notification=atomic_load_explicit(&m->control,memory_order_acquire);
    mesh_retired_release(m);
    uint64_t client=atomic_load_explicit(&m->client,memory_order_acquire);
    if(!client || atomic_load_explicit(&m->configured,memory_order_acquire)!=client){
      if(!stop)os_sync_wait_on_address(&m->control,notification,sizeof m->control,OS_SYNC_WAIT_ON_ADDRESS_SHARED);
      continue;
    }
    atomic_store_explicit(&m->device_client,client,memory_order_seq_cst);
    if(atomic_load_explicit(&m->client,memory_order_seq_cst)!=client || atomic_load_explicit(&m->configured,memory_order_seq_cst)!=client){atomic_store_explicit(&m->device_client,0,memory_order_seq_cst);continue;}
    uint32_t started=0;
    for(uint32_t i=0;i<link_count;i++){
      links[i].client=client;atomic_store_explicit(&links[i].progressing,1,memory_order_release);
    }
    int preparation=0;
    for(uint32_t i=0;i<link_count && !preparation;i++){
      preparation=link_prepare(&links[i]);
      if(preparation)link_error(&links[i],preparation,1);
    }
    for(uint32_t i=0;i<link_count && !preparation;i++){
      int error=pthread_create(&links[i].controller,NULL,link_run,&links[i]);
      if(error){status=error;stop=1;break;}
      started++;
    }
    while(!stop && atomic_load_explicit(&m->client,memory_order_acquire)==client &&
          atomic_load_explicit(&m->configured,memory_order_acquire)==client){
      os_sync_wait_on_address(&m->control,notification,sizeof m->control,OS_SYNC_WAIT_ON_ADDRESS_SHARED);
      notification=atomic_load_explicit(&m->control,memory_order_acquire);
    }
    for(uint32_t i=0;i<started;i++)link_stop(&links[i]);
    for(uint32_t i=0;i<started;i++)pthread_join(links[i].controller,NULL);
    atomic_store_explicit(&m->device_client,0,memory_order_seq_cst);
  }
  for(uint32_t i=0;i<device_count;i++)if(!down_device(&devices[i])){fprintf(stderr,"verbs teardown failed: %s\n",strerror(errno));return 1;}
  atomic_store_explicit(&m->device_client,0,memory_order_seq_cst);
  mesh_retired_release(m);
  for(uint32_t i=0;i<link_count;i++){
    struct mesh_link *link=&links[i];
    atomic_store(&mesh_links(m)[i].port.phase,MESH_STOPPED);
    if(link->provider.listener>=0)close(link->provider.listener);
    close(link->events);
    free(link->requests);
    free(link->receive);
    free(link->configuration);
  }
  for(uint32_t i=0;i<device_count;i++)pthread_mutex_destroy(&devices[i].setup);
  free(completion_outputs);free(devices);free(links);munmap(wire.data,wire.length);
  atomic_store(&m->port.phase,MESH_STOPPED);
  atomic_store_explicit(&control_memory,NULL,memory_order_relaxed);
  munmap(m,length);return status;
}
