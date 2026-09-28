#include "mesh-verbs.h"
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
  uint32_t frames,invocation,queue;
};
_Static_assert(sizeof(struct prepared_receive)==128 && _Alignof(struct prepared_receive)==128 &&
  offsetof(struct prepared_receive,span)==32 && offsetof(struct prepared_receive,input)==48,"M08 M09");
/* design/prepared-machine.md#M08 */
/* A queue pair's receive ring: its records in invocation, binding, chunk order; `posted` the next to
   post, `outstanding` the frames posted and not completed, within the queue's `capacity`; a record is
   posted once its invocation is within `span` of the latest completed (the ring slots' margin). */
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
struct send_stream { struct mesh_send *cell; struct ibv_send_wr *next; uint32_t remaining,stalled; };
/* design/prepared-machine.md#M27 */
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
  struct hdr *M;struct mesh_verbs provider;int qps;uint64_t client;
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
  /* design/prepared-machine.md#M27 */
  struct mesh_trace *trace[2];
  size_t traced[2],trace_capacity[2];
  int ledger;
  /* the link's live counts in the region (mesh.h mesh_net_link): the send thread's and the receive
     thread's running totals over every program, each stored by its one writer */
  struct mesh_net_link *counts;
  uint64_t sends,send_bytes,receives,receive_bytes;
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
  /* design/prepared-machine.md#M27 */
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
/* Posts the ring's next records while its queue has room and each is within the ring's span of the
   latest completed invocation.  Called once at configuration and at each receive completion: the
   refill is fired by the event. */
static int ring_advance(struct mesh_link *link,struct receive_ring *ring){
  int (*post)(struct ibv_qp *,struct ibv_recv_wr *,struct ibv_recv_wr **)=link->provider.queues[0].receive;
  while(ring->posted<ring->count){
    struct prepared_receive *record=ring->first+ring->posted;
    if((int64_t)record->invocation>ring->completed+(int64_t)ring->span ||
       ring->outstanding+record->frames>ring->capacity){
      /* a record held back: the peer's SEND into it waits for this credit */
      atomic_fetch_add_explicit(&link->counts->receive_stalls,1,memory_order_relaxed);
      break;
    }
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
   completions (metal-microbench docs/kernels.md Transport). */
static int link_configure(void *state,int socket,uint64_t client){
  struct mesh_link *link=state;struct hdr *m=link->M;
  struct mesh_tx *tx=(void *)mesh_events(m,mesh_notice_queue(m,client,link->index));
  uint32_t invocations=tx->invocations?tx->invocations:1,depth=atomic_load_explicit(&m->depth,memory_order_acquire);
  if(!depth||depth>invocations)depth=invocations;
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
        /* design/prepared-machine.md#M27 */
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
  for(int q=0;q<qps;q++)if(last[q])last[q]->successor=(uintptr_t)terminal[q];
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
    for(uint32_t i=0;i<count;i++){
      if(slot>=in[i].count)continue;
      uint32_t active=mesh_transfer_active(in+i,invocations),own=in[i].depth?in[i].depth:depth;
      /* receive u+span lands in slot (u+span) mod depth while the local reader may still be at u:
         the ring's span is held a slot short of the ring; a transfer whose every invocation has its
         own slot bounds nothing */
      if(own<active)r->span=MIN(r->span,own>1?own-1:1);
      /* design/prepared-machine.md#M27 */
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
        *record=(struct prepared_receive){.request={.wr_id=(uintptr_t)record,.sg_list=&record->span,.num_sge=1},
          .span=span,.input=(_Atomic uint64_t *)input,.pair=link->provider.queues[ring].pair,.argument=1,
          .frames=(span.length+4095)/4096,.invocation=t,.queue=(uint32_t)ring};
      }
    }
    r->count=(size_t)(link->receive+at-r->first);
    int error=ring_advance(link,r);
    if(error){errno=error;return -1;}
    fprintf(stderr,"receive ring=%d records=%zu posted=%zu frames=%u capacity=%u span=%u\n",
      ring,r->count,r->posted,r->outstanding,r->capacity,r->span);
  }
  if(at)link->receive[at]=link->receive[at-1];
  uint32_t posted=1,peer_posted;
  /* design/prepared-machine.md#M27 */
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
  /* design/prepared-machine.md#M27 */
  struct mesh_trace *trace=traced?link->trace[MESH_SEND]:NULL;
  size_t capacity=traced?link->trace_capacity[MESH_SEND]:0;
  for(uint32_t s=0;;s=s+1==count?0:s+1){
    struct send_stream *stream=streams+s;
    uint64_t observed=0;
    if(!stream->remaining){
      struct mesh_send *cell=stream->cell;
      if(!atomic_load_explicit(&cell->ready,memory_order_acquire)){
        if(!atomic_load_explicit(&link->progressing,memory_order_acquire))return NULL;
        int error=send_retire(link);
        if(error){link_error(link,error,2);return NULL;}
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
        if(!stream->stalled){stream->stalled=1;atomic_fetch_add_explicit(&link->counts->send_stalls,1,memory_order_relaxed);}
        if(!atomic_load_explicit(&link->progressing,memory_order_acquire))return NULL;
        int error=send_retire(link);
        if(error){link_error(link,error,2);return NULL;}
        break;
      }
      struct ibv_send_wr *following=request->next;
      request->next=NULL;
      gate->posted+=frames;
      request->send_flags=IBV_SEND_SIGNALED;request->wr_id=((uint64_t)cell->queue<<48)|(gate->posted&MESH_GATE_MASK);
      int error=post((struct ibv_qp *)cell->pair,request,&bad);
      if(error){link_error(link,error<0?-error:error,1);return NULL;}
      stream->next=following;stream->remaining--;gate->requests++;stream->stalled=0;
      atomic_store_explicit(&link->counts->sends,++link->sends,memory_order_relaxed);
      atomic_store_explicit(&link->counts->send_bytes,link->send_bytes+=request->sg_list->length,memory_order_relaxed);
    }
    if(traced && observed && link->traced[MESH_SEND]<capacity)
      trace[link->traced[MESH_SEND]++]=(struct mesh_trace){(uintptr_t)cell,observed,posting,clock_gettime_nsec_np(CLOCK_UPTIME_RAW)};
    if(!stream->remaining)stream->cell=(struct mesh_send *)cell->successor;
  }
}

/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#independent-native-queues */
static void *link_send_progress(void *argument){
  struct mesh_link *link=argument;
  pthread_setname_np("mesh.rdma.send");
  /* design/prepared-machine.md#M27 */
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
  /* design/prepared-machine.md#M27 */
  struct mesh_trace *trace=traced?link->trace[MESH_RECEIVE]:NULL;
  size_t capacity=traced?link->trace_capacity[MESH_RECEIVE]:0;
  for(;;){
    int count=queue.poll(queue.completion,1,completion);
    if(count<0){link_error(link,count,3);return NULL;}
    if(!count){
      if(!atomic_load_explicit(&link->progressing,memory_order_acquire))return NULL;
      continue;
    }
    uint64_t status_opcode;
    memcpy(&status_opcode,&completion->status,sizeof status_opcode);
    if((uint32_t)status_opcode){link_error(link,(uint32_t)status_opcode,2);return NULL;}
    if(!((status_opcode>>32)&IBV_WC_RECV))continue;
    uint64_t polled=traced?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
    struct prepared_receive *record=(void *)(uintptr_t)completion->wr_id;
    struct receive_ring *ring=link->rings+record->queue;
    ring->landed++;ring->bytes+=completion->byte_len;
    atomic_store_explicit(&link->counts->receives,++link->receives,memory_order_relaxed);
    atomic_store_explicit(&link->counts->receive_bytes,link->receive_bytes+=completion->byte_len,memory_order_relaxed);
    atomic_store_explicit(record->input,record->argument,memory_order_release);
    uint64_t published=traced?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
    ring->outstanding-=record->frames;ring->completed=record->invocation;
    int error=ring_advance(link,ring);
    if(error){link_error(link,error,1);return NULL;}
    /* design/prepared-machine.md#M27 */
    if(traced && link->traced[MESH_RECEIVE]<capacity)
      trace[link->traced[MESH_RECEIVE]++]=(struct mesh_trace){(uintptr_t)record,polled,published,clock_gettime_nsec_np(CLOCK_UPTIME_RAW)};
  }
}

/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#programcopy */
static void *link_receive_progress(void *argument){
  struct mesh_link *link=argument;
  pthread_setname_np("mesh.rdma.receive");
  /* design/prepared-machine.md#M27 */
  if(link->trace[MESH_RECEIVE])return link_receive_drain(link,1);
  return link_receive_drain(link,0);
}

/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#meshresult */
static void link_close(struct mesh_link *link,int *control){
  atomic_store_explicit(&link->progressing,0,memory_order_release);
  if(link->network>=0){close(link->network);link->network=-1;}
  if(*control>=0)shutdown(*control,SHUT_RDWR);
  while(link->worker_count)pthread_join(link->workers[--link->worker_count],NULL);
  if(link->cancel)mesh_cancel(link->M,link->cancel,link->index);
  /* design/prepared-machine.md#M27 */
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
    control=verbs_up(&link->provider,m,link->qps,link_configure,link,link->client);
    if(control<0)link_error(link,errno?errno:EIO,1);
    else {
      EV_SET64(&event,control,EVFILT_READ,EV_ADD|EV_CLEAR,0,0,0,0,0);
      error=kevent64(link->events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)?errno:0;
      if(!error){
        __atomic_store_n(&mesh_links(m)[link->index].bandwidth,link->provider.bandwidth,__ATOMIC_RELAXED);

      }
      /* design/prepared-machine.md#M08 */
      void *(*progress[2])(void *)={link_receive_progress,link_send_progress};
      pthread_attr_t attributes;
      pthread_attr_init(&attributes);
      pthread_attr_set_qos_class_np(&attributes,QOS_CLASS_USER_INTERACTIVE,0);
      for(uint32_t d=0;d<1+(link->publication_count!=0) && !error;d++){
        error=pthread_create(&link->workers[d],&attributes,progress[d],link);
        if(error)break;
        link->worker_count++;
      }
      pthread_attr_destroy(&attributes);
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
            link_error(link,ENETDOWN,4);
          }
        }
      }
    } else if(event.filter==EVFILT_READ)
      link_error(link,event.flags&EV_EOF?(event.fflags?event.fflags:ECONNRESET):EPROTO,4);
  }
  link_close(link,&control);
  EV_SET64(&event,(uint32_t)link->client,EVFILT_PROC,EV_DELETE,0,0,0,0,0);
  kevent64(link->events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
  return NULL;
}

/* The communicator service: NCCL's network plugin model (ncclNet_v12_t, mesh-net.h) served over each
   link, for the bridge's lifetime and for any number of clients and communicators.  One session per
   link pairs on the link's communicator port (its service port + NET_PORT_OFFSET unless given) with one
   UC queue pair.  Control is a stream of fixed messages on the session's socket: connect, accept,
   ready-to-send (RTS), credit and close.  Data is SEND into a posted RECV only (TB5 RDMA has no
   one-sided write), and the receiver drives it: a sender's isend announces its message (RTS: its size
   and where its registration regions fall); the receiver matches announcements with its irecvs in
   order, cuts the message every chunk_frames frames and wherever either end's registration region
   ends, and for each chunk posts its RECV first and only then grants it (CREDIT: the chunk's offset and
   length); the sender SENDs exactly the granted chunks in the order granted.  So a SEND never meets a
   queue without its RECV, and the queue pair's RECVs meet its SENDs one to one whatever communicators
   they belong to, as the prepared program's receives are posted before its peer's first SEND
   (link_configure).  Each session is primed the same way: one RECV posted between RTR and RTS, filled
   by the peer's first SEND.  A receive lands only in memory registered before its queue pair was set up
   (a receive into a registration made after RTR completes with a local protection error, one made
   before the queue pair with success), so communicator memory is the region's registered window, used
   in place, and the discard buffer is the device's (mesh-verbs.h device_up). */
#define NET_PORT_OFFSET 1000
#define NET_CHUNK_FRAMES (MESH_DISCARD/4096)
#define NET_PRIME 64
#define NET_SENDS 8192
#define NET_RECEIVES 8192
#define NET_OUTPUT 32768
#define NET_NONE UINT32_MAX
#define NET_DISCARD (UINT32_MAX-1)
enum { NET_CONNECT=1, NET_ACCEPT, NET_RTS, NET_CREDIT, NET_CLOSE };
/* One control message.  CONNECT: `key` the listen it names, `from` the connecting comm.  ACCEPT: `from`
   the receiving comm made for `to`.  RTS: send request `sequence` of `from`, `size` its bytes,
   extent/phase where its registration regions end.  CREDIT: a chunk of that request, `offset` into it
   and `size` long, whose RECV is posted (size 0: the empty message received; `error`: refused).
   CLOSE: `from` announces and grants nothing more on this connection; flags 1 asks for a CLOSE back. */
struct net_message { uint32_t kind,from,from_generation,to,to_generation,flags; int32_t tag,error; uint64_t key,sequence,size,extent,phase,offset; };
_Static_assert(sizeof(struct net_message)==80,"net_message");
/* A comm's bridge side, its link's session thread's alone: requests taken; a receive comm's requests
   matched with announcements and the announcements waiting in arrival order; each request's phase (a
   send: 1 announced, 2 granted; a receive: 1 waiting, 2 matched) and a send's bytes sent; what is on
   the wire (`outstanding`) and the close handshake. */
struct net_announce { uint64_t sequence,size,extent,phase; };
struct net_comm {
  uint32_t generation,outstanding,live,connect_sent,close_sent,peer_closed;
  uint64_t taken,matched,announce_head,announce_tail;
  struct net_announce announced[MESH_NET_REQUESTS];
  uint64_t sent[MESH_NET_REQUESTS];
  uint8_t phase[MESH_NET_REQUESTS];
};
static struct net_comm net_comms[MESH_NET_COMMS];
/* A granted chunk to SEND (comm NONE: a grant nobody holds, filled from the discard buffer), a matched
   message to receive (the ends' cut parameters e1/p1, e2/p2; the sender's comm and request to grant),
   and a posted RECV. */
struct net_send { uint32_t comm,generation,slot,mr; uint64_t offset,length; };
struct net_transfer { uint32_t comm,generation,slot,mr,peer,peer_generation; uint64_t sequence,size,offset,cursor,landed,e1,p1,e2,p2; };
struct net_chunk { uint32_t transfer,frames; uint64_t length; struct ibv_sge span; };
struct net_session {
  struct hdr *M;uint32_t index;struct mesh_verbs provider;struct mesh_net_link *counts;
  char service[16];pthread_t thread;int started,control,failed;
  uint64_t chunk,send_posted,send_retired;uint32_t send_capacity,receive_capacity,receive_outstanding,chunk_slots;
  struct net_send *sends;uint32_t send_head,send_post,send_tail;
  struct net_transfer *receives;uint32_t receive_head,receive_post,receive_tail;
  struct net_chunk *receive_chunks;uint32_t receive_chunk_head,receive_chunk_tail;
  struct net_message *output;uint32_t output_head,output_tail;size_t output_partial;
  unsigned char input[sizeof(struct net_message)*64];size_t input_bytes;
  struct { uint32_t from,generation; uint64_t key; } pending[MESH_NET_COMMS];uint32_t pending_count;
  int send_blocked,receive_blocked;
  uint64_t bell,scanned,reaped,strays;
};

static uint64_t net_now(void){return clock_gettime_nsec_np(CLOCK_MONOTONIC);}
static void net_nap(uint64_t ns){for(uint64_t end=net_now()+ns;!stop && net_now()<end;)poll(NULL,0,10);}
/* The next cut after `at`: a multiple of the chunk, or where either end's registration region ends
   (extent e, the message's start at phase p within its region). */
static uint64_t net_cut(uint64_t at,uint64_t size,uint64_t chunk,uint64_t e1,uint64_t p1,uint64_t e2,uint64_t p2){
  uint64_t next=(at/chunk+1)*chunk;
  if(e1 && at+e1-(p1+at)%e1<next)next=at+e1-(p1+at)%e1;
  if(e2 && at+e2-(p2+at)%e2<next)next=at+e2-(p2+at)%e2;
  return next<size?next:size;
}
/* Where the window's registered regions end around `offset` (the device's region facts). */
static void net_geometry(struct net_session *s,uint64_t offset,uint64_t *extent,uint64_t *phase){
  *extent=s->provider.device->extent;*phase=offset%s->provider.device->extent;
}
static struct ibv_sge net_span(struct net_session *s,uint32_t mr,uint64_t offset,uint64_t length){
  struct mesh_device *device=s->provider.device;
  if(mr==NET_DISCARD)return (struct ibv_sge){.addr=(uintptr_t)device->discard,.length=(uint32_t)length,.lkey=device->discard_region->lkey};
  return (struct ibv_sge){.addr=(uintptr_t)device->wire+offset,.length=(uint32_t)length,.lkey=device->regions[offset/device->extent]->lkey};
}
static void net_emit(struct net_session *s,struct net_message message){
  if(s->output_tail-s->output_head>=NET_OUTPUT){s->failed=ENOBUFS;return;}
  s->output[s->output_tail++%NET_OUTPUT]=message;
}
static struct mesh_net_comm *net_comm_at(struct net_session *s,uint32_t index,uint32_t generation){
  if(index>=MESH_NET_COMMS)return NULL;
  struct mesh_net_comm *comm=mesh_net_comms(s->M)+index;
  uint32_t state=atomic_load_explicit(&comm->state,memory_order_acquire);
  if(state==MESH_NET_FREE || state==MESH_NET_CLAIMED || comm->link!=s->index || comm->generation!=generation)return NULL;
  return comm;
}
/* The bridge's side of a comm, begun afresh for each generation of its slot. */
static struct net_comm *net_state(struct mesh_net_comm *comm,uint32_t index){
  struct net_comm *state=net_comms+index;
  if(!state->live || state->generation!=comm->generation){
    memset(state,0,sizeof *state);state->generation=comm->generation;state->live=1;
  }
  return state;
}
static void net_finish(struct mesh_net_request *request,int32_t error,uint64_t transferred){
  atomic_store_explicit(&request->error,error,memory_order_relaxed);
  atomic_store_explicit(&request->transferred,transferred,memory_order_relaxed);
  atomic_store_explicit(&request->state,error?MESH_NET_ERROR:MESH_NET_DONE,memory_order_release);
}
/* A request on the wire ends: the comm's hold released, its counts, its state. */
static void net_end(struct net_session *s,struct mesh_net_comm *comm,struct net_comm *state,uint32_t slot,int32_t error,uint64_t bytes){
  struct mesh_net_request *request=comm->requests+slot;
  state->phase[slot]=0;state->outstanding--;
  if(!error){
    atomic_fetch_add_explicit(&comm->bytes,bytes,memory_order_relaxed);
    atomic_fetch_add_explicit(&comm->completions,1,memory_order_relaxed);
    if(request->op==MESH_NET_ISEND){
      atomic_fetch_add_explicit(&s->counts->net_sends,1,memory_order_relaxed);
      atomic_fetch_add_explicit(&s->counts->net_send_bytes,bytes,memory_order_relaxed);
    } else {
      atomic_fetch_add_explicit(&s->counts->net_receives,1,memory_order_relaxed);
      atomic_fetch_add_explicit(&s->counts->net_receive_bytes,bytes,memory_order_relaxed);
    }
  }
  net_finish(request,error,bytes);
}
static void net_free(struct net_session *s,uint32_t index){
  struct mesh_net_comm *comm=mesh_net_comms(s->M)+index;
  net_comms[index].live=0;
  atomic_store_explicit(&comm->state,MESH_NET_FREE,memory_order_release);
}
static void net_refuse(struct net_session *s,const struct net_message *rts,int32_t error){
  net_emit(s,(struct net_message){.kind=NET_CREDIT,.to=rts->from,.to_generation=rts->from_generation,.from=rts->to,.from_generation=rts->to_generation,
    .sequence=rts->sequence,.error=error});
}

/* ---- requests ---- */
static int net_request_valid(struct net_session *s,struct mesh_net_request *request){
  uint64_t wire=mesh_wire_bytes(s->M);
  return !request->size || (request->mr==MESH_NET_WINDOW && s->provider.device->region_count && request->offset<=wire && request->size<=wire-request->offset);
}
/* A receive comm's irecvs in order, each with the next announced send: refused past its capacity,
   done at once when empty, else a message to receive, whose chunks net_post grants as it posts them.
   With the sender closed, an irecv no announcement will reach fails. */
static void net_match(struct net_session *s,uint32_t index){
  struct mesh_net_comm *comm=mesh_net_comms(s->M)+index;struct net_comm *state=net_comms+index;
  while(state->matched<state->taken && !s->failed){
    uint32_t slot=(uint32_t)(state->matched%MESH_NET_REQUESTS);
    struct mesh_net_request *request=comm->requests+slot;
    if(atomic_load_explicit(&request->state,memory_order_acquire)!=MESH_NET_ACTIVE || request->op!=MESH_NET_IRECV || state->phase[slot]!=1){
      state->matched++;continue;
    }
    if(state->announce_head==state->announce_tail){
      if(!state->peer_closed)break;
      net_finish(request,ECONNRESET,0);state->phase[slot]=0;state->matched++;continue;
    }
    if(s->receive_tail-s->receive_head>=NET_RECEIVES)break;
    struct net_announce a=state->announced[state->announce_head++%MESH_NET_REQUESTS];
    struct net_message credit={.kind=NET_CREDIT,.to=comm->peer,.to_generation=comm->peer_generation,.from=index,.from_generation=comm->generation,.sequence=a.sequence};
    state->matched++;
    if(a.size>request->size){credit.error=EMSGSIZE;net_emit(s,credit);net_finish(request,EMSGSIZE,0);state->phase[slot]=0;continue;}
    if(!a.size){
      net_emit(s,credit);net_finish(request,0,0);state->phase[slot]=0;
      atomic_fetch_add_explicit(&comm->completions,1,memory_order_relaxed);
      continue;
    }
    uint64_t extent,phase;
    net_geometry(s,request->offset,&extent,&phase);
    s->receives[s->receive_tail++%NET_RECEIVES]=(struct net_transfer){.comm=index,.generation=comm->generation,.slot=slot,.mr=request->mr,
      .peer=comm->peer,.peer_generation=comm->peer_generation,.sequence=a.sequence,.size=a.size,.offset=request->offset,
      .e1=extent,.p1=phase,.e2=a.extent,.p2=a.phase};
    state->phase[slot]=2;state->outstanding++;
  }
}
/* The client's new requests on a connected comm, in its order: an isend is announced at once and is on
   the wire until its last granted chunk is sent; an irecv waits for its announcement; a flush
   completes once every earlier request of its comm has. */
static void net_take(struct net_session *s,uint32_t index){
  struct mesh_net_comm *comm=mesh_net_comms(s->M)+index;struct net_comm *state=net_comms+index;
  uint32_t kind=atomic_load_explicit(&comm->state,memory_order_acquire);
  uint64_t posted=atomic_load_explicit(&comm->posted,memory_order_acquire);
  while(state->taken<posted && !s->failed){
    uint32_t slot=(uint32_t)(state->taken%MESH_NET_REQUESTS);
    struct mesh_net_request *request=comm->requests+slot;
    if(atomic_load_explicit(&request->state,memory_order_acquire)!=MESH_NET_POSTED)break;
    state->taken++;
    atomic_store_explicit(&request->state,MESH_NET_ACTIVE,memory_order_relaxed);
    state->phase[slot]=0;state->sent[slot]=0;
    int ordered=request->sequence==state->taken-1,valid=ordered && net_request_valid(s,request);
    if(request->op==MESH_NET_IFLUSH && kind==MESH_NET_RECV && ordered)continue;
    if(!valid || (request->op==MESH_NET_ISEND)!=(kind==MESH_NET_SEND) || (request->op!=MESH_NET_ISEND && request->op!=MESH_NET_IRECV)){
      net_finish(request,EINVAL,0);continue;
    }
    if(state->peer_closed && request->op==MESH_NET_ISEND){net_finish(request,ECONNRESET,0);continue;}
    if(request->op==MESH_NET_IRECV){state->phase[slot]=1;continue;}
    uint64_t extent=0,phase=0;
    if(request->size)net_geometry(s,request->offset,&extent,&phase);
    net_emit(s,(struct net_message){.kind=NET_RTS,.to=comm->peer,.to_generation=comm->peer_generation,.from=index,.from_generation=comm->generation,
      .sequence=request->sequence,.size=request->size,.extent=extent,.phase=phase,.tag=request->tag});
    state->phase[slot]=1;state->outstanding++;
  }
  if(kind!=MESH_NET_RECV)return;
  net_match(s,index);
  for(uint32_t slot=0;slot<MESH_NET_REQUESTS;slot++){
    struct mesh_net_request *flush=comm->requests+slot;
    if(flush->op!=MESH_NET_IFLUSH || atomic_load_explicit(&flush->state,memory_order_acquire)!=MESH_NET_ACTIVE)continue;
    int earlier=0;
    for(uint32_t other=0;other<MESH_NET_REQUESTS && !earlier;other++){
      struct mesh_net_request *request=comm->requests+other;uint32_t at=atomic_load_explicit(&request->state,memory_order_acquire);
      earlier=(at==MESH_NET_POSTED || at==MESH_NET_ACTIVE) && request->sequence<flush->sequence;
    }
    if(!earlier){atomic_thread_fence(memory_order_seq_cst);net_finish(flush,0,0);}
  }
}

/* ---- connections ---- */
static uint32_t net_listening(struct net_session *s,uint64_t key){
  struct mesh_net_comm *comms=mesh_net_comms(s->M);
  for(uint32_t i=0;i<MESH_NET_COMMS;i++)
    if(atomic_load_explicit(&comms[i].state,memory_order_acquire)==MESH_NET_LISTEN && comms[i].link==s->index && comms[i].key==key)return i;
  return NET_NONE;
}
/* A connect meets its listen: a receive comm for the listen's client, ACCEPTABLE until the client's
   accept takes it, and the sender's comm told which comm receives it. */
static int net_accept(struct net_session *s,uint32_t listen,uint32_t from,uint32_t generation){
  struct mesh_net_comm *comms=mesh_net_comms(s->M),*origin=comms+listen;
  for(uint32_t i=0;i<MESH_NET_COMMS;i++){
    uint32_t vacant=MESH_NET_FREE;
    if(!atomic_compare_exchange_strong_explicit(&comms[i].state,&vacant,MESH_NET_CLAIMED,memory_order_acq_rel,memory_order_relaxed))continue;
    struct mesh_net_comm *comm=comms+i;
    comm->generation++;comm->link=s->index;comm->listen=listen;comm->listen_generation=origin->generation;
    comm->peer=from;comm->peer_generation=generation;comm->owner=origin->owner;comm->key=origin->key;
    atomic_store_explicit(&comm->error,0,memory_order_relaxed);
    atomic_store_explicit(&comm->posted,0,memory_order_relaxed);atomic_store_explicit(&comm->bytes,0,memory_order_relaxed);
    atomic_store_explicit(&comm->completions,0,memory_order_relaxed);atomic_store_explicit(&comm->credit_waits,0,memory_order_relaxed);
    for(uint32_t r=0;r<MESH_NET_REQUESTS;r++)atomic_store_explicit(&comm->requests[r].state,MESH_NET_IDLE,memory_order_relaxed);
    net_state(comm,i);
    atomic_store_explicit(&comm->state,MESH_NET_ACCEPTABLE,memory_order_release);
    net_emit(s,(struct net_message){.kind=NET_ACCEPT,.to=from,.to_generation=generation,.from=i,.from_generation=comm->generation});
    return 1;
  }
  return 0;
}
static void net_connected(struct net_session *s,const struct net_message *message){
  uint32_t listen=net_listening(s,message->key);
  if(listen!=NET_NONE && net_accept(s,listen,message->from,message->from_generation))return;
  if(listen==NET_NONE && s->pending_count<MESH_NET_COMMS){
    s->pending[s->pending_count].from=message->from;s->pending[s->pending_count].generation=message->from_generation;
    s->pending[s->pending_count++].key=message->key;
    return;
  }
  net_emit(s,(struct net_message){.kind=NET_CLOSE,.to=message->from,.to_generation=message->from_generation,.from=NET_NONE,.error=ENOSPC});
}
/* CLOSE: the peer announces and grants nothing more on this connection.  `to` NONE withdraws a
   connect: its pending entry, or the comm it made.  A stale CLOSE asking for one is answered, so its
   sender can vacate.  What the peer already announced or granted stays good. */
static void net_closed(struct net_session *s,const struct net_message *message){
  struct mesh_net_comm *comm=NULL,*comms=mesh_net_comms(s->M);
  uint32_t index=message->to;
  if(index==NET_NONE){
    for(uint32_t p=0;p<s->pending_count;p++)if(s->pending[p].from==message->from && s->pending[p].generation==message->from_generation){
      s->pending[p]=s->pending[--s->pending_count];
      net_emit(s,(struct net_message){.kind=NET_CLOSE,.to=message->from,.to_generation=message->from_generation,.from=NET_NONE});
      return;
    }
    for(index=0;index<MESH_NET_COMMS && !comm;index++){
      uint32_t at=atomic_load_explicit(&comms[index].state,memory_order_acquire);
      if(at!=MESH_NET_FREE && at!=MESH_NET_CLAIMED && at!=MESH_NET_LISTEN && at!=MESH_NET_CONNECTING && comms[index].link==s->index &&
         comms[index].peer==message->from && comms[index].peer_generation==message->from_generation)comm=comms+index;
    }
    index--;
  } else comm=net_comm_at(s,index,message->to_generation);
  if(!comm){
    if(message->flags && message->from!=NET_NONE)
      net_emit(s,(struct net_message){.kind=NET_CLOSE,.to=message->from,.to_generation=message->from_generation,.from=message->to,.from_generation=message->to_generation});
    return;
  }
  struct net_comm *state=net_state(comm,index);
  uint32_t connecting=MESH_NET_CONNECTING;
  if(atomic_compare_exchange_strong_explicit(&comm->state,&connecting,MESH_NET_FAILED,memory_order_acq_rel,memory_order_relaxed))
    atomic_store_explicit(&comm->error,message->error?message->error:ECONNREFUSED,memory_order_relaxed);
  state->peer_closed=1;
  if(atomic_load_explicit(&comm->state,memory_order_acquire)==MESH_NET_RECV)net_match(s,index);
  if(!state->close_sent){
    state->close_sent=1;
    if(message->from!=NET_NONE)
      net_emit(s,(struct net_message){.kind=NET_CLOSE,.to=message->from,.to_generation=message->from_generation,.from=index,.from_generation=comm->generation});
  }
}
/* RTS: queued for the receive comm's next irecv, or refused when that comm is gone or closing. */
static void net_announced(struct net_session *s,const struct net_message *message){
  struct mesh_net_comm *comm=net_comm_at(s,message->to,message->to_generation);
  if(!comm || atomic_load_explicit(&comm->state,memory_order_acquire)!=MESH_NET_RECV){net_refuse(s,message,ECONNRESET);return;}
  struct net_comm *state=net_state(comm,message->to);
  if(state->announce_tail-state->announce_head>=MESH_NET_REQUESTS){s->failed=EPROTO;return;}
  state->announced[state->announce_tail++%MESH_NET_REQUESTS]=(struct net_announce){message->sequence,message->size,message->extent,message->phase};
  uint64_t before=state->announce_head;
  net_match(s,message->to);
  if(state->announce_head==before){
    atomic_fetch_add_explicit(&comm->credit_waits,1,memory_order_relaxed);
    atomic_fetch_add_explicit(&s->counts->credit_waits,1,memory_order_relaxed);
  }
}
/* CREDIT: a chunk whose RECV the peer posted, SENT from its request's buffer; a grant nobody holds is
   filled from the discard buffer all the same, so the peer's RECV is not left waiting. */
static void net_granted(struct net_session *s,const struct net_message *message){
  struct mesh_net_comm *comm=net_comm_at(s,message->to,message->to_generation);
  uint32_t slot=(uint32_t)(message->sequence%MESH_NET_REQUESTS);
  struct mesh_net_request *request=comm?comm->requests+slot:NULL;
  struct net_comm *state=comm?net_state(comm,message->to):NULL;
  if(request && (atomic_load_explicit(&request->state,memory_order_acquire)!=MESH_NET_ACTIVE || request->op!=MESH_NET_ISEND ||
     request->sequence!=message->sequence || !state->phase[slot]))request=NULL;
  if(message->error || !message->size){
    if(request)net_end(s,comm,state,slot,message->error,0);
    return;
  }
  if(s->send_tail-s->send_head>=NET_SENDS){s->failed=ENOBUFS;return;}
  if(message->size>s->chunk){s->failed=EPROTO;return;}
  if(!request || message->offset>request->size || message->size>request->size-message->offset){
    if(request)net_end(s,comm,state,slot,EPROTO,0);
    s->strays++;
    s->sends[s->send_tail++%NET_SENDS]=(struct net_send){.comm=NET_NONE,.mr=NET_DISCARD,.length=message->size};
    return;
  }
  state->phase[slot]=2;
  s->sends[s->send_tail++%NET_SENDS]=(struct net_send){.comm=message->to,.generation=message->to_generation,.slot=slot,.mr=request->mr,
    .offset=request->offset+message->offset,.length=message->size};
}
static void net_receive(struct net_session *s,const struct net_message *message){
  switch(message->kind){
  case NET_CONNECT: net_connected(s,message); break;
  case NET_ACCEPT: {
    struct mesh_net_comm *comm=net_comm_at(s,message->to,message->to_generation);
    if(!comm)break;
    comm->peer=message->from;comm->peer_generation=message->from_generation;
    uint32_t connecting=MESH_NET_CONNECTING;
    atomic_compare_exchange_strong_explicit(&comm->state,&connecting,MESH_NET_SEND,memory_order_acq_rel,memory_order_relaxed);
    break;
  }
  case NET_RTS: net_announced(s,message); break;
  case NET_CREDIT: net_granted(s,message); break;
  case NET_CLOSE: net_closed(s,message); break;
  default: s->failed=EPROTO;
  }
}

/* ---- the wire ---- */
/* Granted chunks are SENT in grant order while the queue has frames; matched messages' chunks are
   posted as RECVs in match order while the queue has frames, each granted once posted. */
static int net_post(struct net_session *s){
  while(s->send_post!=s->send_tail){
    struct net_send *chunk=s->sends+s->send_post%NET_SENDS;
    uint32_t frames=(uint32_t)((chunk->length+4095)/4096);
    if(s->send_posted-s->send_retired+frames>s->send_capacity){
      if(!s->send_blocked){s->send_blocked=1;atomic_fetch_add_explicit(&s->counts->send_stalls,1,memory_order_relaxed);}
      break;
    }
    s->send_blocked=0;
    struct ibv_sge span=net_span(s,chunk->mr,chunk->offset,chunk->length);
    struct ibv_send_wr request={.wr_id=s->send_post,.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
    int error=ibv_post_send(s->provider.queues[0].pair,&request,&bad);
    if(error)return error<0?-error:error;
    s->send_post++;s->send_posted+=frames;
  }
  while(s->receive_post!=s->receive_tail){
    struct net_transfer *t=s->receives+s->receive_post%NET_RECEIVES;
    if(t->cursor>=t->size){s->receive_post++;continue;}
    uint64_t next=net_cut(t->cursor,t->size,s->chunk,t->e1,t->p1,t->e2,t->p2),length=next-t->cursor;
    uint32_t frames=(uint32_t)((length+4095)/4096);
    if(s->receive_outstanding+frames>s->receive_capacity){
      if(!s->receive_blocked){s->receive_blocked=1;atomic_fetch_add_explicit(&s->counts->receive_stalls,1,memory_order_relaxed);}
      break;
    }
    s->receive_blocked=0;
    struct ibv_sge span=net_span(s,t->mr,t->offset+t->cursor,length);
    struct ibv_recv_wr request={.wr_id=s->receive_chunk_tail,.sg_list=&span,.num_sge=1},*bad;
    int error=ibv_post_recv(s->provider.queues[0].pair,&request,&bad);
    if(error)return error<0?-error:error;
    s->receive_chunks[s->receive_chunk_tail++%s->chunk_slots]=(struct net_chunk){s->receive_post,frames,length,span};
    s->receive_outstanding+=frames;
    net_emit(s,(struct net_message){.kind=NET_CREDIT,.to=t->peer,.to_generation=t->peer_generation,.from=t->comm,.from_generation=t->generation,
      .sequence=t->sequence,.offset=t->cursor,.size=length});
    t->cursor=next;
  }
  return 0;
}
/* A failed completion, logged with the work request it names: the session ends on it. */
static int net_failed(struct net_session *s,const struct ibv_wc *done,const char *kind,uint64_t expected,struct ibv_sge span,uint32_t mr,uint32_t transfer){
  fprintf(stderr,"session link %u: %s completion %s (status %d vendor %u) wr_id %llu expected %llu byte_len %u; its SGE addr=0x%llx length=%u lkey=0x%x registration %u transfer %u\n",
    s->index,kind,ibv_wc_status_str(done->status),done->status,done->vendor_err,(unsigned long long)done->wr_id,(unsigned long long)expected,done->byte_len,
    (unsigned long long)span.addr,span.length,span.lkey,mr,transfer);
  return EIO;
}
/* Completions retire chunks in the order they were posted (one queue pair each way). */
static int net_complete(struct net_session *s,int *busy){
  struct ibv_wc done[16];
  int count=ibv_poll_cq(s->provider.sent,16,done);
  if(count<0)return EIO;
  for(int i=0;i<count;i++){
    if(done[i].status || done[i].wr_id!=s->send_head){
      struct net_send *chunk=s->sends+s->send_head%NET_SENDS;
      return net_failed(s,done+i,"SEND",s->send_head,net_span(s,chunk->mr,chunk->offset,chunk->length),chunk->mr,chunk->comm);
    }
    struct net_send chunk=s->sends[s->send_head++%NET_SENDS];
    s->send_retired+=(chunk.length+4095)/4096;
    struct mesh_net_comm *comm=chunk.comm==NET_NONE?NULL:net_comm_at(s,chunk.comm,chunk.generation);
    if(!comm)continue;
    struct net_comm *state=net_state(comm,chunk.comm);
    if(state->phase[chunk.slot]==2 && (state->sent[chunk.slot]+=chunk.length)==comm->requests[chunk.slot].size)
      net_end(s,comm,state,chunk.slot,0,comm->requests[chunk.slot].size);
  }
  *busy|=count>0;
  count=ibv_poll_cq(s->provider.completion,16,done);
  if(count<0)return EIO;
  for(int i=0;i<count;i++){
    struct net_chunk *posted=s->receive_chunks+s->receive_chunk_head%s->chunk_slots;
    if(done[i].status || done[i].wr_id!=s->receive_chunk_head || done[i].byte_len!=posted->length)
      return net_failed(s,done+i,"RECV",s->receive_chunk_head,posted->span,posted->transfer==NET_NONE?NET_DISCARD:s->receives[posted->transfer%NET_RECEIVES].mr,posted->transfer);
    struct net_chunk chunk=s->receive_chunks[s->receive_chunk_head++%s->chunk_slots];
    s->receive_outstanding-=chunk.frames;
    if(chunk.transfer==NET_NONE)continue;
    struct net_transfer *t=s->receives+chunk.transfer%NET_RECEIVES;
    if((t->landed+=chunk.length)<t->size)continue;
    struct mesh_net_comm *comm=net_comm_at(s,t->comm,t->generation);
    if(comm)net_end(s,comm,net_state(comm,t->comm),t->slot,0,t->size);
  }
  while(s->receive_head!=s->receive_post && s->receives[s->receive_head%NET_RECEIVES].landed==s->receives[s->receive_head%NET_RECEIVES].size)s->receive_head++;
  *busy|=count>0;
  return 0;
}
static int net_flush(struct net_session *s){
  while(s->output_head!=s->output_tail){
    char *message=(char *)(s->output+s->output_head%NET_OUTPUT);
    ssize_t n=write(s->control,message+s->output_partial,sizeof(struct net_message)-s->output_partial);
    if(n>0){if((s->output_partial+=(size_t)n)==sizeof(struct net_message)){s->output_partial=0;s->output_head++;}continue;}
    if(n<0 && (errno==EAGAIN || errno==EWOULDBLOCK || errno==EINTR))return 0;
    return n<0?errno:ECONNRESET;
  }
  return 0;
}
static int net_read(struct net_session *s,int *busy){
  for(;;){
    ssize_t n=read(s->control,s->input+s->input_bytes,sizeof s->input-s->input_bytes);
    if(!n)return ECONNRESET;
    if(n<0)return errno==EAGAIN || errno==EWOULDBLOCK || errno==EINTR?0:errno;
    s->input_bytes+=(size_t)n;*busy=1;
    size_t whole=s->input_bytes/sizeof(struct net_message)*sizeof(struct net_message);
    for(size_t at=0;at<whole && !s->failed;at+=sizeof(struct net_message)){
      struct net_message message;memcpy(&message,s->input+at,sizeof message);net_receive(s,&message);
    }
    memmove(s->input,s->input+whole,s->input_bytes-whole);s->input_bytes-=whole;
    if(s->failed)return s->failed;
  }
}

/* ---- the client tables ---- */
static int net_dead(uint64_t owner){pid_t pid=(pid_t)(uint32_t)owner;return !pid || (kill(pid,0) && errno==ESRCH);}
/* Every comm of this link whose client has exited is closed; its client slot is vacated, and its window
   pages once no comm of it remains (link 0's session, for the region's tables). */
static void net_reap(struct net_session *s){
  struct hdr *m=s->M;
  struct mesh_net_comm *comms=mesh_net_comms(m);
  for(uint32_t i=0;i<MESH_NET_COMMS;i++){
    uint32_t at=atomic_load_explicit(&comms[i].state,memory_order_acquire);
    if(at==MESH_NET_FREE || at==MESH_NET_CLAIMED || at==MESH_NET_CLOSING || comms[i].link!=s->index || !net_dead(comms[i].owner))continue;
    atomic_store_explicit(&comms[i].state,MESH_NET_CLOSING,memory_order_release);
  }
  struct mesh_net_memory *memory=mesh_net_memory(m);
  for(uint32_t i=0;!s->index && i<MESH_NET_MEMORY;i++){
    uint64_t owner=atomic_load_explicit(&memory[i].owner,memory_order_acquire);
    if(!owner || !net_dead(owner))continue;
    int held=0;
    for(uint32_t c=0;c<MESH_NET_COMMS && !held;c++){
      uint32_t at=atomic_load_explicit(&comms[c].state,memory_order_acquire);
      held=at!=MESH_NET_FREE && at!=MESH_NET_CLAIMED && comms[c].owner==owner;
    }
    if(held)continue;
    uint32_t first=memory[i].first,pages=memory[i].pages;
    if(atomic_compare_exchange_strong_explicit(&memory[i].owner,&owner,0,memory_order_acq_rel,memory_order_relaxed) && pages)
      mesh_bits_clear(mesh_arena_bits(m),first,pages);
  }
  for(uint32_t i=0;!s->index && i<MESH_NET_CLIENTS;i++){
    uint64_t owner=atomic_load_explicit(&mesh_net_clients(m)[i].owner,memory_order_acquire);
    if(owner && net_dead(owner))atomic_compare_exchange_strong_explicit(&mesh_net_clients(m)[i].owner,&owner,0,memory_order_acq_rel,memory_order_relaxed);
  }
}
/* A comm its client closed: requests not yet on the wire fail, announcements waiting for it are
   refused, then CLOSE; what is on the wire completes, and the slot is vacated once the peer has
   closed too and nothing of it is on the wire. */
static void net_close(struct net_session *s,uint32_t index){
  struct mesh_net_comm *comm=mesh_net_comms(s->M)+index,*comms=mesh_net_comms(s->M);
  struct net_comm *state=net_state(comm,index);
  if(!state->close_sent){
    state->close_sent=1;
    for(uint32_t slot=0;slot<MESH_NET_REQUESTS;slot++){
      struct mesh_net_request *request=comm->requests+slot;
      uint32_t at=atomic_load_explicit(&request->state,memory_order_acquire);
      if(at==MESH_NET_POSTED || (at==MESH_NET_ACTIVE && (!state->phase[slot] || (request->op==MESH_NET_IRECV && state->phase[slot]==1)))){
        net_finish(request,ECONNRESET,0);state->phase[slot]=0;
      }
    }
    while(state->announce_head!=state->announce_tail){
      struct net_announce a=state->announced[state->announce_head++%MESH_NET_REQUESTS];
      net_emit(s,(struct net_message){.kind=NET_CREDIT,.to=comm->peer,.to_generation=comm->peer_generation,.from=index,.from_generation=comm->generation,
        .sequence=a.sequence,.error=ECONNRESET});
    }
    state->taken=state->matched=atomic_load_explicit(&comm->posted,memory_order_acquire);
    if(comm->peer!=NET_NONE)
      net_emit(s,(struct net_message){.kind=NET_CLOSE,.to=comm->peer,.to_generation=comm->peer_generation,.from=index,.from_generation=comm->generation,.flags=1});
    else if(state->connect_sent)
      net_emit(s,(struct net_message){.kind=NET_CLOSE,.to=NET_NONE,.from=index,.from_generation=comm->generation,.key=comm->key,.flags=1});
    else state->peer_closed=1;
    for(uint32_t i=0;i<MESH_NET_COMMS;i++)
      if(atomic_load_explicit(&comms[i].state,memory_order_acquire)==MESH_NET_ACCEPTABLE && comms[i].link==s->index &&
         comms[i].listen==index && comms[i].listen_generation==comm->generation)
        atomic_store_explicit(&comms[i].state,MESH_NET_CLOSING,memory_order_release);
  }
  if(state->peer_closed && !state->outstanding)net_free(s,index);
}
static void net_scan(struct net_session *s){
  struct hdr *m=s->M;
  struct mesh_net_comm *comms=mesh_net_comms(m);
  for(uint32_t i=0;i<MESH_NET_COMMS && !s->failed;i++){
    struct mesh_net_comm *comm=comms+i;
    uint32_t at=atomic_load_explicit(&comm->state,memory_order_acquire);
    if(at==MESH_NET_FREE || at==MESH_NET_CLAIMED || comm->link!=s->index)continue;
    struct net_comm *state=net_state(comm,i);
    if(at==MESH_NET_CONNECTING && !state->connect_sent){
      state->connect_sent=1;
      net_emit(s,(struct net_message){.kind=NET_CONNECT,.to=NET_NONE,.from=i,.from_generation=comm->generation,.key=comm->key});
    }
    else if(at==MESH_NET_SEND || at==MESH_NET_RECV)net_take(s,i);
    else if(at==MESH_NET_CLOSING)net_close(s,i);
    else if(at==MESH_NET_LISTEN){
      for(uint32_t p=0;p<s->pending_count;){
        if(s->pending[p].key==comm->key && net_accept(s,i,s->pending[p].from,s->pending[p].generation))s->pending[p]=s->pending[--s->pending_count];
        else p++;
      }
    }
  }
}
/* A session lost: every transfer is gone with its queue pair.  Connected comms fail (their clients
   close them), unaccepted receive comms and closing ones are vacated, listens stay and connects are
   sent again on the next session. */
static void net_lost(struct net_session *s,int32_t error){
  struct mesh_net_comm *comms=mesh_net_comms(s->M);
  for(uint32_t i=0;i<MESH_NET_COMMS;i++){
    struct mesh_net_comm *comm=comms+i;
    uint32_t at=atomic_load_explicit(&comm->state,memory_order_acquire);
    if(at==MESH_NET_FREE || at==MESH_NET_CLAIMED || comm->link!=s->index)continue;
    if(at==MESH_NET_ACCEPTABLE || at==MESH_NET_CLOSING){net_free(s,i);continue;}
    for(uint32_t slot=0;slot<MESH_NET_REQUESTS;slot++){
      uint32_t r=atomic_load_explicit(&comm->requests[slot].state,memory_order_acquire);
      if(r==MESH_NET_POSTED || r==MESH_NET_ACTIVE)net_finish(comm->requests+slot,error,0);
    }
    net_comms[i].live=0;
    if(at==MESH_NET_SEND || at==MESH_NET_RECV){
      atomic_store_explicit(&comm->error,error,memory_order_relaxed);
      atomic_store_explicit(&comm->state,MESH_NET_FAILED,memory_order_release);
      at=MESH_NET_FAILED;
    }
    if(at==MESH_NET_FAILED){
      struct net_comm *state=net_state(comm,i);
      state->close_sent=state->peer_closed=1;
      state->taken=state->matched=atomic_load_explicit(&comm->posted,memory_order_acquire);
    }
  }
  s->pending_count=0;
}
static void net_reset(struct net_session *s){
  s->send_posted=s->send_retired=0;s->receive_outstanding=0;
  s->send_head=s->send_post=s->send_tail=s->receive_head=s->receive_post=s->receive_tail=0;
  s->receive_chunk_head=s->receive_chunk_tail=0;
  s->output_head=s->output_tail=0;s->output_partial=s->input_bytes=0;s->failed=0;s->send_blocked=s->receive_blocked=0;
}
/* The session's configuration, between RTR and RTS: the priming RECV posted into the device's discard
   buffer, then each end's queue capacities exchanged, so that neither end SENDs before the other's
   first RECV is posted; the chunk both cut by, at most the discard buffer. */
static int net_configure(void *argument,int socket,uint64_t client){
  struct net_session *s=argument;struct mesh_queue *queue=s->provider.queues;
  s->send_capacity=MIN(queue->send_capacity,(uint32_t)s->provider.sent->cqe);
  s->receive_capacity=MIN(queue->receive_capacity,(uint32_t)s->provider.completion->cqe);
  /* a power of two past the queue's frames, so the ring's indices wrap with their counters */
  for(s->chunk_slots=1;s->chunk_slots<=s->receive_capacity;)s->chunk_slots*=2;
  free(s->receive_chunks);s->receive_chunks=calloc(s->chunk_slots,sizeof *s->receive_chunks);
  if(!s->sends)s->sends=calloc(NET_SENDS,sizeof *s->sends);
  if(!s->receives)s->receives=calloc(NET_RECEIVES,sizeof *s->receives);
  if(!s->output)s->output=calloc(NET_OUTPUT,sizeof *s->output);
  if(!s->receive_chunks || !s->sends || !s->receives || !s->output){errno=ENOMEM;return -1;}
  struct ibv_sge span=net_span(s,NET_DISCARD,0,NET_PRIME);
  struct ibv_recv_wr prime={.wr_id=s->receive_chunk_tail,.sg_list=&span,.num_sge=1},*bad;
  int error=ibv_post_recv(queue->pair,&prime,&bad);
  if(error){errno=error<0?-error:error;return -1;}
  s->receive_chunks[s->receive_chunk_tail++%s->chunk_slots]=(struct net_chunk){NET_NONE,1,NET_PRIME,span};
  s->receive_outstanding=1;
  uint32_t mine[2]={s->send_capacity,s->receive_capacity},peer[2];
  if(exchange(socket,mine,peer,sizeof mine,sizeof peer,s->M,client,s->provider.deadline))return -1;
  uint32_t least=MIN(MIN(mine[0],mine[1]),MIN(peer[0],peer[1])),frames=MIN(NET_CHUNK_FRAMES,least/2);
  s->chunk=(uint64_t)(frames?frames:1)*4096;
  atomic_store_explicit(&s->counts->chunk_frames,frames?frames:1,memory_order_relaxed);
  atomic_store_explicit(&s->counts->wire_regions,s->provider.device->region_count,memory_order_relaxed);
  return 0;
}
/* The session's progress: completions, control messages, the client tables when a client rings (or
   every millisecond), chunks posted while their queue has frames; it spins while anything is in flight
   or was lately, and otherwise waits for a client's doorbell a short while at a time.  Its first SEND
   fills the peer's priming RECV. */
static void net_serve(struct net_session *s){
  s->sends[s->send_tail++%NET_SENDS]=(struct net_send){.comm=NET_NONE,.mr=NET_DISCARD,.length=NET_PRIME};
  uint64_t last=net_now();
  while(!stop && !s->failed){
    int busy=0,error=net_complete(s,&busy);
    if(!error)error=net_read(s,&busy);
    if(error){s->failed=error;break;}
    uint64_t now=net_now(),bell=atomic_load_explicit(&s->counts->doorbell,memory_order_acquire);
    if(bell!=s->bell || now-s->scanned>1000000){busy|=bell!=s->bell;s->bell=bell;s->scanned=now;net_scan(s);}
    if(now-s->reaped>500000000){s->reaped=now;net_reap(s);}
    if(!s->failed && (error=net_post(s)))s->failed=error;
    if(!s->failed && (error=net_flush(s)))s->failed=error;
    if(busy)last=now;
    int flight=s->send_head!=s->send_tail || s->receive_chunk_head!=s->receive_chunk_tail || s->output_head!=s->output_tail;
    if(!busy && !flight && now-last>2000000)
      os_sync_wait_on_address_with_timeout(&s->counts->doorbell,bell,sizeof bell,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
        OS_CLOCK_MACH_ABSOLUTE_TIME,now-last>1000000000?1000000:50000);
  }
}
static void *net_session_run(void *argument){
  struct net_session *s=argument;struct hdr *m=s->M;
  pthread_setname_np("mesh.net.session");
  s->control=-1;
  while(!stop){
    atomic_store_explicit(&s->counts->phase,MESH_PAIRING,memory_order_release);
    net_reset(s);
    int f=verbs_up(&s->provider,m,1,net_configure,s,MESH_NET_SESSION);
    if(f<0){
      atomic_store_explicit(&s->counts->code,errno?errno:EIO,memory_order_relaxed);
      while(!down_pair(&s->provider))poll(NULL,0,100);
      if(s->provider.listener>=0){close(s->provider.listener);s->provider.listener=-1;}
      net_nap(250000000);
      continue;
    }
    s->control=f;
    __atomic_store_n(&mesh_links(m)[s->index].bandwidth,s->provider.bandwidth,__ATOMIC_RELAXED);
    atomic_store_explicit(&s->counts->code,0,memory_order_relaxed);
    atomic_fetch_add_explicit(&s->counts->sessions,1,memory_order_relaxed);
    atomic_store_explicit(&s->counts->phase,MESH_PAIRED,memory_order_release);
    net_serve(s);
    int32_t error=stop?ECANCELED:s->failed?s->failed:EIO;
    if(!stop)fprintf(stderr,"session down: link %u: %s\n",s->index,strerror(error));
    if(s->strays)fprintf(stderr,"session link %u: %llu grants no request held, filled from the discard buffer\n",s->index,(unsigned long long)s->strays);
    atomic_store_explicit(&s->counts->code,error,memory_order_relaxed);
    atomic_store_explicit(&s->counts->phase,MESH_STOPPED,memory_order_release);
    net_lost(s,error);
    close(f);s->control=-1;
    while(!down_pair(&s->provider))poll(NULL,0,100);
    /* a lost session pairs again after a pause, not at once: failed completions, torn-down queue pairs
       and immediate re-pairing preceded the Sep 5 kernel panic (design/RDMA-KERNEL-RECOVERY.md) */
    net_nap(UINT64_C(3000000000));
  }
  free(s->receive_chunks);free(s->sends);free(s->receives);free(s->output);
  atomic_store_explicit(&s->counts->phase,MESH_STOPPED,memory_order_release);
  return NULL;
}

/* design/algorithm-sources.md#programcopy */
int main(int argc,char **argv){
  const char *name=MESH_NAME;int me=0,layout=0;double pct=0;
  uint64_t arena_pages=0,block_pages=0,table_rows=0,window_pages=0,orders=4096;
  uint32_t link_count=0,device_count=0,qps=getenv("MESH_QPS")?(uint32_t)atoi(getenv("MESH_QPS")):1;
  struct mesh_link *links=aligned_alloc(_Alignof(struct mesh_link),(size_t)argc*sizeof *links);
  struct mesh_device *devices=calloc((size_t)argc,sizeof *devices);
  struct net_session *sessions=calloc((size_t)argc,sizeof *sessions);
  if(!links || !devices || !sessions)die("bridge configuration allocation");
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
      char *device=strsep(&fields,","),*peer=strsep(&fields,","),*local=strsep(&fields,","),*remote=strsep(&fields,","),*service=strsep(&fields,","),*net=strsep(&fields,",");
      if(!device || !*device || !peer || !*peer || !local || !*local || !remote || !*remote || fields)die("link requires device,peer,local-address,remote-address[,service[,communicator-service]]");
      uint32_t d=0;
      while(d<device_count && strcmp(devices[d].name,device))d++;
      if(d==device_count){devices[d].name=device;pthread_mutex_init(&devices[d].setup,NULL);device_count++;}
      link->index=link_count++;link->provider=(struct mesh_verbs){.device=&devices[d],.peer=(uint32_t)strtoul(peer,NULL,10),.local_address=local,.remote_address=remote,.service=service?service:MESH_PORT,.listener=-1};
      /* the link's communicator session: its own port, the link's service + NET_PORT_OFFSET unless given */
      struct net_session *session=&sessions[link->index];
      if(net && *net)snprintf(session->service,sizeof session->service,"%s",net);
      else snprintf(session->service,sizeof session->service,"%ld",strtol(link->provider.service,NULL,10)+NET_PORT_OFFSET);
      session->index=link->index;session->provider=link->provider;session->provider.service=session->service;
      session->provider.kind="session";session->provider.window=UINT64_C(3000000000);session->provider.magic=0x10000;
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
    link->counts=mesh_net_links(m)+i;
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
  for(uint32_t i=0;i<link_count;i++){
    struct net_session *session=&sessions[i];
    session->M=m;session->counts=links[i].counts;session->provider.wire=&wire;
    fprintf(stderr,"session link %u: communicators on port %s\n",i,session->service);
    int error=pthread_create(&session->thread,NULL,net_session_run,session);
    if(error)fprintf(stderr,"session link %u: %s\n",i,strerror(error));
    else session->started=1;
  }
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
  for(uint32_t i=0;i<link_count;i++)if(sessions[i].started)pthread_join(sessions[i].thread,NULL);
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
  free(completion_outputs);free(devices);free(links);free(sessions);munmap(wire.data,wire.length);
  atomic_store(&m->port.phase,MESH_STOPPED);
  atomic_store_explicit(&control_memory,NULL,memory_order_relaxed);
  munmap(m,length);return status;
}
