/* The bridge: one a node, started once and left serving every process of the node at once (MESH_VERSION 120;
   operator, 2026-10-01: "the *bridge* is an application which exposes an api which lets ANY PROGRAM IN THE UNIVERSE to
   say 'HEY, I WOULD LIKE TO WRITE TO A GLOBALLY ACCESSIBLE WORKSPACE ANY PROGRAM IN THE UNIVERSE ON THE OTHER COMPUTER
   CAN READ, YAH/NAH?' then gets a 'YAH, I WROTE THAT' from the application").
     Its region is one registered window (and an arena beyond it) that any number of processes map and allocate in.
   Each link carries, for the bridge's life, a stripe service: a control queue pair and a data queue pair, paired once
   (again only when the peer's bridge starts again), serving every process's stripe requests (mesh.h mesh_requests):
   the request's bytes go as a descriptor on the control pair (the peer's bridge posts its receives at the same offset)
   and then as SENDs on the data pair; its word is stored as the RDMA driver completes them, at the writer when its last
   SEND completes and at the peer when its last RECV does.  Beside it, any number of prepared sessions (mesh.h
   mesh_session: a process's prepared program, paired with the peer's session of the same key) run d6dceac's
   per-crossing path, the source of the fastest recorded E2B decode (metal-microbench docs/measurement.md): their own
   queue pairs, receives posted ahead, two progress threads each.  TCP is used at pairing alone (its connection kept as
   the peer's liveness); every byte and word after it travels as RDMA SEND into a posted RECV.  Nothing here checks or
   refuses what a process asks (operator: every process is trusted like kernel code). */
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
struct prepared_receive {
  _Alignas(128) struct ibv_recv_wr request;
  struct ibv_sge span;
  _Alignas(32) struct ibv_recv_wr *next;
  struct ibv_qp *pair;
  _Atomic uint64_t *input;
  uint64_t argument;
};
_Static_assert(sizeof(struct prepared_receive)==128 && _Alignof(struct prepared_receive)==128 &&
  offsetof(struct prepared_receive,span)==32 && offsetof(struct prepared_receive,next)==64 &&
  offsetof(struct prepared_receive,pair)==72 && offsetof(struct prepared_receive,input)==80 &&
  offsetof(struct prepared_receive,argument)==88,"M08 M09");
/* design/prepared-machine.md#M27 */
struct mesh_trace { _Alignas(32) uint64_t identity; uint64_t begin,middle,end; };
_Static_assert(sizeof(struct mesh_trace)==32 && _Alignof(struct mesh_trace)==32,"M27");
struct mesh_peer;
/* A session's link: d6dceac's client link, now one of each prepared session's (mesh.h mesh_session). */
struct mesh_link {
  pthread_t workers[2];
  uint32_t worker_count,publication_count,cursor_count,index,session;
  pthread_t controller;
  int events,network,refill,linear;
  union mesh_network_event network_event;
  _Atomic int progressing,finished;
  struct hdr *M;struct mesh_verbs provider;int qps;uint64_t key;pid_t pid;
  struct mesh_peer *peer;
  struct prepared_receive *receive;
  struct ibv_qp *receive_pair;
  struct prepared_send *requests;
  struct mesh_send *publications,**cursors;
  struct ibv_wc *completion;
  struct mesh_cancellation *cancel;
  /* design/prepared-machine.md#M27 */
  struct mesh_trace *trace[2];
  size_t traced[2],trace_capacity[2];
  int ledger;
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
  struct mesh_port_info *port=mesh_session_port(link->M,link->session,link->index);port->code=code;port->domain=domain;
  if(atomic_load_explicit(&link->progressing,memory_order_relaxed) || atomic_load_explicit(&port->phase,memory_order_relaxed)!=MESH_STOPPED)
    fprintf(stderr,"session %u link %u error %lld domain %u (1 verbs, 2 completion status, 3 poll, 4 event)\n",link->session,link->index,(long long)code,domain);
  atomic_store_explicit(&port->phase,MESH_STOPPED,memory_order_relaxed);
  atomic_store_explicit(&port->prepared,link->key,memory_order_release);
  mesh_control_notify(link->M);
  link_stop(link);
}
/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M06 */
/* design/algorithm-sources.md#programcopy */
static int link_prepare(struct mesh_link *link){
  struct hdr *m=link->M;
  struct mesh_tx *tx=(void *)mesh_events(m,mesh_notice_queue(m,link->session,link->index));
  /* design/prepared-machine.md#M12 */
  link->cancel=tx->cancel?(void *)((char *)m+tx->cancel):NULL;
  link->publications=(void *)((char *)m+tx->cells);
  link->publication_count=tx->count*tx->slots+tx->once;
  size_t count=0;
  uint32_t incoming=0;
  link->qps=(int)(m->qps*tx->slots);
  for(uint32_t q=0;q<m->qps;q++)for(uint32_t d=0;d<2;d++){
    uint32_t channel=link->index*m->qps+q;
    struct mesh_transfer *transfers=mesh_transfers(m,link->session,channel,d);
    for(uint32_t i=0;i<atomic_load(mesh_order_length(m,link->session,channel,d));i++){
      uint32_t chunks=mesh_row_chunks(m,transfers[i].local_row,transfers[i].bytes);
      if(d==MESH_SEND)count+=(size_t)(chunks-1)*transfers[i].count*(transfers[i].stride?tx->invocations:1);
      else incoming+=chunks*transfers[i].count;
    }
  }
  free(link->requests);
  /* design/prepared-machine.md#M29 */
  link->requests=count?aligned_alloc(128,count*sizeof *link->requests):NULL;
  if(count&&!link->requests)return ENOMEM;
  link->provider.completion_entries[MESH_SEND]=link->publication_count;
  link->provider.completion_entries[MESH_RECEIVE]=incoming;
  /* design/prepared-machine.md#M27 */
  const char *ledger=getenv("MESH_LEDGER");
  link->ledger=ledger && ledger[0] && !(ledger[0]=='0' && !ledger[1]);
  for(int d=0;d<2;d++){
    free(link->trace[d]);link->trace[d]=NULL;link->traced[d]=0;
    link->trace_capacity[d]=link->ledger?(size_t)link->provider.completion_entries[d]*(tx->invocations?tx->invocations:1):0;
    if(!link->trace_capacity[d])continue;
    link->trace[d]=aligned_alloc(32,(link->trace_capacity[d]+1)*sizeof(struct mesh_trace));
    if(!link->trace[d])return ENOMEM;
  }
  return 0;
}


/* design/algorithm-sources.md#programcopy */
/* design/prepared-machine.md#M08 */
static int link_configure(void *state,int socket){
  struct mesh_link *link=state;struct hdr *m=link->M;
  const uint32_t client=link->session;
  struct mesh_tx *tx=(void *)mesh_events(m,mesh_notice_queue(m,client,link->index));
  link->cursors=calloc((size_t)link->qps,sizeof *link->cursors);
  if(!link->cursors)return -1;
  /* design/prepared-machine.md#M08 */
  uint32_t invocations=tx->invocations?tx->invocations:1;
  /* design/prepared-machine.md#M01 */
  /* Invocation t addresses slot t mod depth.  depth==invocations is the unrung machine and the
     modulus is the identity; the client sizes its operands to depth and publishes it in the header
     before the handshake, so the two agree by construction. */
  uint32_t depth=atomic_load_explicit(&mesh_sessions(m)[client].depth,memory_order_acquire);
  if(!depth||depth>invocations)depth=invocations;
  uint32_t incoming=link->provider.completion_entries[MESH_RECEIVE];
  size_t receive_count=(size_t)incoming*invocations;
  link->receive=aligned_alloc(128,(receive_count+1)*sizeof *link->receive);
  if(!link->receive)return -1;
  uint32_t frames[link->qps];memset(frames,0,sizeof frames);
  /* design/prepared-machine.md#M08 */
  /* A queue retires unsignaled SEND requests only when a later signaled one completes, and holds
     its frames until then.  A stream whose one invocation sends more frames than its queue holds
     therefore signals each publication's last request, so the queue retires within the
     invocation; every other stream signals the invocation's last alone (mesh-call.c). */
  uint32_t sent[link->qps];memset(sent,0,sizeof sent);
  for(uint32_t q=0;q<m->qps;q++){
    uint32_t channel=link->index*m->qps+q;
    uint32_t count=atomic_load(mesh_order_length(m,client,channel,MESH_SEND));
    struct mesh_transfer *out=mesh_transfers(m,client,channel,MESH_SEND);
    for(uint32_t i=0;i<count;i++)for(uint32_t slot=0;slot<out[i].count;slot++){
      uint32_t row=out[i].local_row+slot*out[i].stride,chunks=mesh_row_chunks(m,row,out[i].bytes);
      uint64_t remaining=out[i].bytes;
      for(uint32_t k=0;k<chunks;k++){
        uint64_t offset=atomic_load_explicit(&mesh_page(m)[row+k].address,memory_order_relaxed)-m->data_off;
        uint32_t bytes=wire_span(link->provider.device,offset,remaining).length;remaining-=bytes;
        sent[q*tx->slots+slot]+=(bytes+4095)/4096;
      }
    }
  }
  /* each receive frame's queue and 4096-byte frames, for a window within an invocation (below) */
  uint32_t *lane=malloc((incoming?incoming:1)*sizeof *lane),*weight=malloc((incoming?incoming:1)*sizeof *weight);
  if(!lane||!weight){free(lane);free(weight);return -1;}
  uint32_t next=0,received=0;
  struct ibv_qp *receive_pair=NULL;
  int multiple_receive_queues=0;
  for(uint32_t q=0;q<m->qps;q++){
    uint32_t channel=link->index*m->qps+q;
    uint32_t counts[2]={atomic_load(mesh_order_length(m,client,channel,MESH_SEND)),atomic_load(mesh_order_length(m,client,channel,MESH_RECEIVE))};
    struct mesh_transfer *out=mesh_transfers(m,client,channel,MESH_SEND),*in=mesh_transfers(m,client,channel,MESH_RECEIVE);
    for(uint32_t i=0;i<counts[MESH_SEND];i++)for(uint32_t slot=0;slot<out[i].count;slot++){
      uint32_t row=out[i].local_row+slot*out[i].stride;
      uint32_t chunks=mesh_row_chunks(m,row,out[i].bytes);
      uint32_t publication=slot*tx->count+out[i].first;
      /* design/prepared-machine.md#M04 */
      uint32_t stream=q*tx->slots+slot;
      struct mesh_send *cell=link->publications+(size_t)publication*((size_t)invocations+1);
      /* design/prepared-machine.md#M27 */
      if(link->ledger)fprintf(stderr,"{\"trace_binding\":%u,\"rank\":%u,\"direction\":0,\"index\":%u,\"binding\":%u,\"slot\":%u}\n",
        link->index,m->node,publication,out[i].binding,slot);
      uint32_t repetitions=out[i].stride?invocations:1;
      uint64_t continuation=cell->request.wr_id;
      unsigned flags=cell->request.send_flags|(sent[stream]>link->provider.queues[stream].send_capacity?IBV_SEND_SIGNALED:0);
      for(uint32_t t=0;t<repetitions;t++){
        cell[t].pair=(uintptr_t)link->provider.queues[stream].pair;
        uint64_t remaining=out[i].bytes;
        for(uint32_t k=0;k<chunks;k++){
          uint32_t address_row=row+(uint32_t)((size_t)(t%depth)*out[i].invocation_pages/m->block)+k;
          uint64_t offset=atomic_load_explicit(&mesh_page(m)[address_row].address,memory_order_relaxed)-m->data_off;
          struct ibv_sge span=wire_span(link->provider.device,offset,remaining);
          remaining-=span.length;
          /* design/prepared-machine.md#M29 */
          struct ibv_sge *entry=k?&link->requests[next+k-1].span:&cell[t].span;
          struct ibv_send_wr *request=k?&link->requests[next+k-1].request:&cell[t].request;
          *entry=span;
          *request=(struct ibv_send_wr){.wr_id=k?0:continuation,.next=k+1==chunks?NULL:&link->requests[next+k].request,
            .send_flags=k+1==chunks?flags:0,
            .sg_list=entry,.num_sge=1,.opcode=IBV_WR_SEND};
        }
        next+=chunks-1;
      }
      if(!link->cursors[stream])link->cursors[stream]=cell;
    }
    for(uint32_t i=0;i<counts[MESH_RECEIVE];i++){
      uint32_t chunks=mesh_row_chunks(m,in[i].local_row,in[i].bytes);
      for(uint32_t slot=0;slot<in[i].count;slot++){
        uint32_t row=in[i].local_row+slot*in[i].stride;
        uint64_t remaining=in[i].bytes;
        for(uint32_t k=0;k<chunks;k++){
          uint64_t chunk=atomic_load_explicit(&mesh_page(m)[row+k].address,memory_order_relaxed)-m->data_off;
          uint32_t bytes=wire_span(link->provider.device,chunk,remaining).length;remaining-=bytes;
          /* design/prepared-machine.md#M08 */
          /* design/prepared-machine.md#M09 */
          struct mesh_queue *queue=&link->provider.queues[q*tx->slots+slot];
          multiple_receive_queues|=receive_pair && receive_pair!=queue->pair;
          receive_pair=queue->pair;
          uint32_t frame=received++;
          /* design/prepared-machine.md#M27 */
          if(link->ledger)fprintf(stderr,"{\"trace_binding\":%u,\"rank\":%u,\"direction\":1,\"index\":%u,\"binding\":%u,\"slot\":%u,\"chunk\":%u,\"chunks\":%u}\n",
            link->index,m->node,frame,in[i].binding,slot,k,chunks);
          frames[q*tx->slots+slot]+=(bytes+4095)/4096;
          lane[frame]=q*tx->slots+slot;weight[frame]=(bytes+4095)/4096;
          struct mesh_publication *delivery=mesh_publication_at(m,row+k);
          uintptr_t input=(uintptr_t)&delivery->argument;
          uint64_t argument=1;
          if(delivery->device_input)input=(uintptr_t)m+delivery->device_input;
          else if(delivery->sends){
            struct mesh_send *cell=(void *)((char *)m+delivery->targets[0].stream);
            input=(uintptr_t)cell;argument=cell->request.wr_id;
          }
          for(uint32_t t=0;t<invocations;t++){
            uint32_t address_row=row+(uint32_t)((size_t)(t%depth)*in[i].invocation_pages/m->block)+k;
            uint64_t offset=atomic_load_explicit(&mesh_page(m)[address_row].address,memory_order_relaxed)-m->data_off;
            struct ibv_sge span=wire_span(link->provider.device,offset,bytes);
            struct prepared_receive *record=link->receive+(size_t)t*incoming+frame;
            *record=(struct prepared_receive){
              .request={.wr_id=(uintptr_t)record,.sg_list=&record->span,.num_sge=1},
              .span=span,.input=(_Atomic uint64_t *)(input+(delivery->device_input?delivery->device_stride*t:delivery->sends?sizeof(struct mesh_send)*(size_t)t:0)),
              .argument=argument,.pair=queue->pair};
          }
        }
      }
    }
  }
  /* design/prepared-machine.md#M08 */
  link->receive_pair=multiple_receive_queues?NULL:receive_pair;
  link->linear=receive_count!=0;
  for(size_t i=0;i<receive_count;i++)
    link->linear&=link->receive[i].input==link->receive[0].input+i && link->receive[i].argument==1;
  uint32_t window=invocations;
  int within=0;
  for(int q=0;q<link->qps;q++)if(frames[q]){
    uint32_t capacity=link->provider.queues[q].receive_capacity/frames[q];
    if(!capacity)within=1;
    else if(capacity<window)window=capacity;
  }
  /* design/prepared-machine.md#M01 */
  /* Receive t+window lands in slot (t+window) mod depth while the local reader is at t, so the
     window is held a slot short of the ring: the same depth-1 step margin the send side has. */
  if(depth<invocations && window>depth-1)window=depth>1?depth-1:1;
  link->refill=within||window<invocations;
  if(!within){
    for(uint32_t t=0;t<invocations;t++)for(uint32_t f=0;f<incoming;f++){
      struct prepared_receive *record=link->receive+(size_t)t*incoming+f;
      record->next=t+window<invocations?&link->receive[(size_t)(t+window)*incoming+f].request:NULL;
      if(t<window){
        struct ibv_recv_wr *bad;
        int error=link->provider.queues[0].receive(record->pair,&record->request,&bad);
        if(error){free(lane);free(weight);errno=error<0?-error:error;return -1;}
      }
    }
  }
  /* design/prepared-machine.md#M08 */
  /* An invocation whose frames exceed a queue's capacity: the window is a run of records within
     it.  Each queue preposts the longest run of its records, in invocation-then-frame order, that
     fits its capacity wherever the run starts, and each completion posts the record that run's
     length later on the same queue, so a queue always holds that many consecutive receives.  A
     peer sends a record only after this rank published a crossing that record depends on, which
     it did after completing its receives before that crossing, so the receives a peer can send
     ahead of this rank's completions are the ones between; tools/mesh/programs.py chunk bounds a
     chunk's rows so that they fit (metal-microbench docs/kernels.md Multimodal). */
  uint32_t *order=within?malloc((incoming?incoming:1)*sizeof *order):NULL;
  if(within&&!order){free(lane);free(weight);return -1;}
  for(int q=0;within&&q<link->qps;q++){
    uint32_t n=0;
    for(uint32_t f=0;f<incoming;f++)if(lane[f]==(uint32_t)q)order[n++]=f;
    if(!n)continue;
    uint32_t capacity=link->provider.queues[q].receive_capacity,run=n;
    for(uint32_t j=0;j<n&&run;j++){
      uint32_t sum=0,r=0;
      while(r<run&&sum+weight[order[(j+r)%n]]<=capacity){sum+=weight[order[(j+r)%n]];r++;}
      run=r;
    }
    if(!run){free(lane);free(weight);free(order);errno=ENOMEM;return -1;}
    for(uint32_t t=0;t<invocations;t++)for(uint32_t j=0;j<n;j++){
      size_t position=(size_t)t*n+j,later=position+run;
      struct prepared_receive *record=link->receive+(size_t)t*incoming+order[j];
      record->next=later<(size_t)invocations*n?&link->receive[(later/n)*incoming+order[later%n]].request:NULL;
      if(position<run){
        struct ibv_recv_wr *bad;
        int error=link->provider.queues[0].receive(record->pair,&record->request,&bad);
        if(error){free(lane);free(weight);free(order);errno=error<0?-error:error;return -1;}
      }
    }
    fprintf(stderr,"receive queue=%d records=%u run=%u\n",q,n,run);
  }
  free(lane);free(weight);free(order);
  if(receive_count)link->receive[receive_count]=link->receive[receive_count-1];
  fprintf(stderr,"receive window=%u invocations=%u depth=%u frames=%u refill=%d within=%d\n",within?0:window,invocations,depth,incoming,link->refill,within);
  link->cursor_count=0;
  for(int q=0;q<link->qps;q++)if(link->cursors[q]){
    link->cursors[link->cursor_count++]=link->cursors[q];
  }
  uint32_t posted=1,peer_posted;
  /* design/prepared-machine.md#M27 */
  if(link->ledger)fprintf(stderr,"{\"trace_layout\":%u,\"rank\":%u,\"send_base\":%llu,\"receive_base\":%llu,\"invocations\":%u,\"send_stride\":%u,\"send_record_bytes\":%zu,\"receive_stride\":%u,\"send_capacity\":%zu,\"receive_capacity\":%zu}\n",
    link->index,m->node,(unsigned long long)(uintptr_t)link->publications,(unsigned long long)(uintptr_t)link->receive,
    invocations,invocations+1,sizeof(struct mesh_send),incoming,link->trace_capacity[MESH_SEND],link->trace_capacity[MESH_RECEIVE]);
  if(exchange(socket,&posted,&peer_posted,sizeof posted,sizeof peer_posted,&link->provider))return -1;
  return 0;
}

/* design/prepared-machine.md#M06 */
/* A full send queue is "not yet", never a failure (docs/standards.md S1): the requests the provider took stay
   posted and the rest (from bad) is posted as the receive thread's polls of the shared completion queue retire
   earlier SENDs.  Reached only from a post that failed, so the drains' posting path is unchanged.  Returns 0 once
   posted, ECANCELED if the link stopped meanwhile, or the post's other error. */
static __attribute__((noinline,cold)) int link_send_full(struct mesh_link *link,struct ibv_qp *pair,struct ibv_send_wr *rest,int error){
  int (*post)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **)=link->provider.queues[0].send;
  struct ibv_send_wr *bad=rest;
  while(error==ENOMEM || error==-ENOMEM){
    if(!atomic_load_explicit(&link->progressing,memory_order_acquire))return ECANCELED;
    rest=bad;error=post(pair,rest,&bad);
  }
  return error<0?-error:error;
}

/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M06 */
/* design/prepared-machine.md#M15 */
/* design/algorithm-sources.md#independent-native-queues */
static __attribute__((always_inline)) inline void *link_send_drain(struct mesh_link *link,uint32_t streams,int traced){
  struct mesh_send **cursors=link->cursors,*record=cursors[0];
  struct ibv_qp *pair=streams==1?(void *)record->pair:NULL;
  uint32_t stream=0;
  struct ibv_send_wr *bad;
  int (*post)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **)=link->provider.queues[0].send;
  /* design/prepared-machine.md#M27 */
  struct mesh_trace *trace=traced?link->trace[MESH_SEND]:NULL;
  size_t capacity=traced?link->trace_capacity[MESH_SEND]:0;
  for(;;){
    uint64_t continuation=atomic_load_explicit(&record->ready,memory_order_acquire);
    if(continuation){
        uint64_t observed=traced?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
        struct mesh_send *next=(void *)((uintptr_t)record+continuation);
        uint64_t posting=traced?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
        int error=post(streams==1?pair:(struct ibv_qp *)record->pair,&record->request,&bad);
        if(traced && link->traced[MESH_SEND]<capacity){
          uint64_t sent=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
          trace[link->traced[MESH_SEND]++]=(struct mesh_trace){(uintptr_t)record,observed,posting,sent};
        }
        if(error && (error=link_send_full(link,streams==1?pair:(struct ibv_qp *)record->pair,bad,error))){
          if(error!=ECANCELED)link_error(link,error,1);
          return NULL;
        }
        if(streams>1)cursors[stream]=next;
        record=next;
    }else if(!atomic_load_explicit(&link->progressing,memory_order_acquire))return NULL;
    if(streams>1){if(++stream==streams)stream=0;record=cursors[stream];}
  }
}

/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#independent-native-queues */
static void *link_send_progress(void *argument){
  struct mesh_link *link=argument;
  pthread_setname_np("mesh.rdma.send");
  /* design/prepared-machine.md#M27 */
  if(link->trace[MESH_SEND]){
    if(link->cursor_count==1)return link_send_drain(link,1,1);
    return link_send_drain(link,link->cursor_count,1);
  }
  if(link->cursor_count==1)return link_send_drain(link,1,0);
  return link_send_drain(link,link->cursor_count,0);
}

/* design/prepared-machine.md#M08 */
/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#programcopy */
static __attribute__((always_inline)) inline void *link_receive_drain(struct mesh_link *link,int ordered,int refill,int linear,int traced){
  struct mesh_queue queue=link->provider.queues[0];
  struct ibv_wc *completion=link->completion;
  int (*post)(struct ibv_qp *,struct ibv_recv_wr *,struct ibv_recv_wr **)=queue.receive;
  struct ibv_qp *pair=link->receive_pair;
  struct prepared_receive *record=link->receive;
  _Atomic uint64_t *input=linear?record->input:NULL;
  uint64_t value=linear?1:0;
  /* design/prepared-machine.md#M27 */
  struct mesh_trace *trace=traced?link->trace[MESH_RECEIVE]:NULL;
  size_t capacity=traced?link->trace_capacity[MESH_RECEIVE]:0;
  for(;;){
    if(ordered&&!linear){input=record->input;value=record->argument;}
    for(;;){
      int count=queue.poll(queue.completion,1,completion);
      if(count<0){link_error(link,count,3);return NULL;}
      if(count){
        uint64_t status_opcode;
        memcpy(&status_opcode,&completion->status,sizeof status_opcode);
        if((uint32_t)status_opcode){link_error(link,(uint32_t)status_opcode,2);return NULL;}
        if((status_opcode>>32)&IBV_WC_RECV)break;
        continue;
      }
      if(!atomic_load_explicit(&link->progressing,memory_order_acquire))return NULL;
    }
    uint64_t polled=traced?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
    /* design/prepared-machine.md#M08 */
    if(!ordered){record=(void *)(uintptr_t)completion->wr_id;input=record->input;value=record->argument;}
    atomic_store_explicit(input,value,memory_order_release);
    uint64_t published=traced?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
    struct prepared_receive *identity=record;
    if(refill){
      struct ibv_recv_wr *next,*bad;struct ibv_qp *target=pair;
      if(ordered)next=record->next;
      else __asm__("ldp %0, %1, [%2, #64]":"=r"(next),"=r"(target):"r"(record):"memory");
      if(next){
        int error=post(target,next,&bad);
        if(error){link_error(link,error<0?-error:error,1);return NULL;}
      }
    }
    /* design/prepared-machine.md#M27 */
    if(traced && link->traced[MESH_RECEIVE]<capacity){
      uint64_t reposted=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
      trace[link->traced[MESH_RECEIVE]++]=(struct mesh_trace){(uintptr_t)identity,polled,published,reposted};
    }
    if(linear)input++;
    if(ordered)record++;
  }
}

/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#programcopy */
static __attribute__((always_inline)) inline void *link_receive_select(struct mesh_link *link,int traced){
  if(link->receive_pair){
    if(link->linear){
      if(link->refill)return link_receive_drain(link,1,1,1,traced);
      return link_receive_drain(link,1,0,1,traced);
    }
    if(link->refill)return link_receive_drain(link,1,1,0,traced);
    return link_receive_drain(link,1,0,0,traced);
  }
  if(link->refill)return link_receive_drain(link,0,1,0,traced);
  return link_receive_drain(link,0,0,0,traced);
}

/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#programcopy */
static void *link_receive_progress(void *argument){
  struct mesh_link *link=argument;
  pthread_setname_np("mesh.rdma.receive");
  /* design/prepared-machine.md#M27 */
  if(link->trace[MESH_RECEIVE])return link_receive_select(link,1);
  return link_receive_select(link,0);
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
  if(*control>=0){close(*control);*control=-1;}
  while(!down_pair(&link->provider))link_error(link,errno?errno:EIO,1);
  free(link->receive);link->receive=NULL;
  free(link->cursors);link->cursors=NULL;
  atomic_store_explicit(&mesh_session_port(link->M,link->session,link->index)->phase,MESH_STOPPED,memory_order_release);
}

/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M24 */
/* design/algorithm-sources.md#programcopy */
/* design/algorithm-sources.md#meshresult */
/* A session's link: paired with the peer's session of its key, served until its process exits, its interface goes
   down, the bridge stops, or both ends' programs are done.  A session stopped by its process (it has published all it
   will) says so on its pairing connection (one byte, then the half-close) and is served on until the peer's says the
   same, so neither end's queue pairs go while the other's last SENDs or receives are in flight; the connection's end
   with no such byte is the peer's failure, and every receive word not landed is cancelled. */
static void *link_run(void *argument){
  struct mesh_link *link=argument;struct hdr *m=link->M;
  struct mesh_port_info *port=mesh_session_port(m,link->session,link->index);
  uint32_t transfers=0;
  for(uint32_t q=0;q<m->qps;q++)for(int d=0;d<2;d++)transfers+=atomic_load(mesh_order_length(m,link->session,link->index*m->qps+q,d));
  struct kevent64_s event;
  EV_SET64(&event,(uint64_t)link->pid,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,0,0,0);
  int error=kevent64(link->events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)?errno:0,control=-1;
  if(error)link_error(link,error,4);
  if(!error && transfers){
    link->network=socket(PF_SYSTEM,SOCK_RAW,SYSPROTO_EVENT);
    struct kev_request filter={KEV_VENDOR_APPLE,KEV_NETWORK_CLASS,KEV_DL_SUBCLASS};
    if(link->network<0 || fcntl(link->network,F_SETFL,O_NONBLOCK)<0 ||
       ioctl(link->network,SIOCSKEVFILT,&filter))error=errno;
    if(error)link_error(link,error,4);
  }
  if(!error && transfers){
    atomic_store_explicit(&port->phase,MESH_PAIRING,memory_order_release);
    /* a pairing whose connection went away is made again on the next one (mesh-verbs.h pairing_retried) */
    for(;;){
      control=verbs_up(&link->provider,link->qps,link_configure,link);
      if(control>=0)break;
      int failure=errno?errno:EIO;
      while(!down_pair(&link->provider))link_error(link,errno?errno:EIO,1);
      free(link->receive);link->receive=NULL;free(link->cursors);link->cursors=NULL;
      if(!pairing_retried(failure) || !pairing_active(&link->provider)){errno=failure;break;}
      fprintf(stderr,"session %u pairing again: %s\n",link->session,strerror(failure));
    }
    if(control<0)link_error(link,errno?errno:EIO,1);
    else {
      struct kevent64_s watched[2];
      EV_SET64(&watched[0],control,EVFILT_READ,EV_ADD|EV_CLEAR,0,0,0,0,0);
      EV_SET64(&watched[1],link->network,EVFILT_READ,EV_ADD,0,0,0,0,0);
      error=kevent64(link->events,watched,2,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)?errno:0;
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
        qos_class_t classes[2]={QOS_CLASS_UNSPECIFIED,QOS_CLASS_UNSPECIFIED};int relative;
        for(uint32_t d=0;d<link->worker_count;d++)pthread_get_qos_class_np(link->workers[d],&classes[d],&relative);
        fprintf(stderr,"session %u link %u key %llu progress threads %u qos receive 0x%x send 0x%x (0x%x user-interactive)\n",
          link->session,link->index,(unsigned long long)link->key,link->worker_count,classes[0],classes[1],QOS_CLASS_USER_INTERACTIVE);
        atomic_store_explicit(&port->phase,MESH_PAIRED,memory_order_relaxed);
        atomic_store_explicit(&port->prepared,link->key,memory_order_release);
        mesh_control_notify(m);
      }
    }
  } else if(!transfers){
    atomic_store_explicit(&port->phase,MESH_PAIRED,memory_order_release);
    atomic_store_explicit(&port->prepared,link->key,memory_order_release);
    atomic_store_explicit(&link->progressing,0,memory_order_release);
    mesh_control_notify(m);
  }
  int finishing=0,peer_finished=0,peer_closed=0;
  for(;;){
    if(!atomic_load_explicit(&link->progressing,memory_order_acquire))break;
    if(!finishing && control>=0 &&
       atomic_load_explicit(&mesh_sessions(m)[link->session].request,memory_order_acquire)!=MESH_REQUEST_START){
      finishing=1;
      if(write(control,"F",1)!=1 || shutdown(control,SHUT_WR))link_error(link,errno?errno:EIO,4);
    }
    if(finishing && peer_finished && peer_closed)break;
    int count=kevent64(link->events,NULL,0,&event,1,0,NULL);
    if(count<0){if(errno==EINTR)continue;link_error(link,errno,4);break;}
    if(event.flags&EV_ERROR)link_error(link,event.data,4);
    else if(event.filter==EVFILT_PROC)link_error(link,ECANCELED,4);
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
    } else if(event.filter==EVFILT_READ && event.ident==(uint64_t)control){
      char bytes[16];ssize_t n;
      while((n=read(control,bytes,sizeof bytes))>0)for(ssize_t i=0;i<n;i++)peer_finished|=bytes[i]=='F';
      if(!n){
        peer_closed=1;
        struct kevent64_s quiet;EV_SET64(&quiet,control,EVFILT_READ,EV_DELETE,0,0,0,0,0);
        kevent64(link->events,&quiet,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
        if(!peer_finished)link_error(link,ECONNRESET,4);
      } else if(n<0 && errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR)link_error(link,errno,4);
    }
  }
  link_close(link,&control);
  EV_SET64(&event,(uint64_t)link->pid,EVFILT_PROC,EV_DELETE,0,0,0,0,0);
  kevent64(link->events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
  atomic_store_explicit(&link->finished,1,memory_order_release);
  mesh_control_notify(m);
  return NULL;
}

/* ---- a link's peer: the bridge's for its life ---- */
/* A stripe's request on the data pair in flight (a ring of them, completions FIFO a queue pair and direction): its
   frames, and for a stripe's last request the word and value its completion stores. */
struct mesh_chunk { uint64_t word,value; uint32_t frames,last; };
#define MESH_CHUNKS 1024
#define MESH_STRIPE_CHUNK ((uint64_t)4<<20)
enum { WR_DESCRIPTOR_RECEIVE=1, WR_DESCRIPTOR_SEND, WR_DATA_RECEIVE, WR_DATA_SEND };
struct mesh_peer {
  uint32_t index;
  struct hdr *M;
  struct mesh_device *device; struct mesh_wire *wire;
  const char *local_address,*remote_address,*service;
  uint32_t node;
  char *configuration;
  int listener;
  pthread_t acceptor,server;
  int acceptor_events,server_events;
  _Atomic int running;
  pthread_mutex_t lock;
  struct mesh_connection parked[64]; int nparked;
  /* the sessions' kqueues on this link, triggered when a connection is parked (their pairing then takes it) */
  int waiting[MESH_SESSIONS];
  /* the stripe service */
  struct mesh_verbs provider;
  struct mesh_descriptor *received,*sent;  /* the zone's: MESH_DESCRIPTORS each */
  uint64_t received_at,sent_at;            /* their window offsets */
  struct mesh_chunk chunks[2][MESH_CHUNKS];
  uint64_t head[2],tail[2];
  uint32_t frames[2];                      /* data frames in flight: SEND, RECV */
  uint64_t descriptors_sent,descriptors_done;
  struct mesh_descriptor pending[MESH_DESCRIPTORS];
  uint64_t pending_head,pending_tail;
};

static int peer_take(void *state,uint64_t key,struct mesh_connection *connection){
  struct mesh_peer *peer=state;
  pthread_mutex_lock(&peer->lock);
  for(int i=0;i<peer->nparked;i++)if(peer->parked[i].you.key==key){
    *connection=peer->parked[i];peer->parked[i]=peer->parked[--peer->nparked];
    pthread_mutex_unlock(&peer->lock);
    return 0;
  }
  pthread_mutex_unlock(&peer->lock);
  errno=EAGAIN;return -1;
}
static void trigger(int events){
  struct kevent64_s event;EV_SET64(&event,0,EVFILT_USER,0,NOTE_TRIGGER,0,0,0,0);
  kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
}

/* The listening node's acceptor: each dial accepted and its exchange record read, then held for the pairing of its
   key (a session's, or the stripe service's: key 0), every pairing on the link told.  Ends with the bridge. */
static void *peer_accept(void *argument){
  struct mesh_peer *peer=argument;
  pthread_setname_np("mesh.accept");
  struct kevent64_s change,event;
  EV_SET64(&change,(uint64_t)peer->listener,EVFILT_READ,EV_ADD,0,0,0,0,0);
  kevent64(peer->acceptor_events,&change,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
  while(!stop && atomic_load_explicit(&peer->running,memory_order_acquire)){
    int count=kevent64(peer->acceptor_events,NULL,0,&event,1,0,NULL);
    if(count<=0 || event.filter!=EVFILT_READ)continue;
    if(event.ident==(uint64_t)peer->listener){
      int f=accept(peer->listener,NULL,NULL);
      if(f<0)continue;
      connection_prompt(f);
      if(fcntl(f,F_SETFL,O_NONBLOCK)<0){close(f);continue;}
      EV_SET64(&change,(uint64_t)f,EVFILT_READ,EV_ADD|EV_ONESHOT,0,0,0,0,0);
      kevent64(peer->acceptor_events,&change,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
      continue;
    }
    /* a connection's exchange record: the dial sends it at once (a connection that closes first is dropped) */
    int f=(int)event.ident;
    struct mesh_connection connection={.fd=f};
    ssize_t n=recv(f,&connection.you,sizeof connection.you,MSG_PEEK);
    if(n<(ssize_t)sizeof connection.you){
      if(n>0 || (n<0 && (errno==EAGAIN || errno==EINTR))){
        EV_SET64(&change,(uint64_t)f,EVFILT_READ,EV_ADD|EV_ONESHOT,0,0,0,0,0);
        kevent64(peer->acceptor_events,&change,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
      } else close(f);
      continue;
    }
    if(recv(f,&connection.you,sizeof connection.you,0)!=(ssize_t)sizeof connection.you){close(f);continue;}
    pthread_mutex_lock(&peer->lock);
    if(peer->nparked==(int)(sizeof peer->parked/sizeof *peer->parked)){close(peer->parked[0].fd);peer->parked[0]=peer->parked[--peer->nparked];}
    peer->parked[peer->nparked++]=connection;
    pthread_mutex_unlock(&peer->lock);
    trigger(peer->server_events);
    for(uint32_t s=0;s<MESH_SESSIONS;s++)trigger(peer->waiting[s]);
  }
  return NULL;
}

/* The stripe service's two queue pairs paired: the control pair's receives (each a descriptor slot of the zone)
   posted before the peer is told this end is ready. */
static int peer_configure(void *state,int socket){
  struct mesh_peer *peer=state;
  struct mesh_queue *control=&peer->provider.queues[0];
  for(uint32_t i=0;i<MESH_DESCRIPTORS && i<control->receive_capacity;i++){
    struct ibv_sge span={.addr=(uintptr_t)peer->device->wire+peer->received_at+(uint64_t)i*sizeof(struct mesh_descriptor),
      .length=sizeof(struct mesh_descriptor),.lkey=peer->device->regions[(peer->received_at+(uint64_t)i*sizeof(struct mesh_descriptor))/peer->device->extent]->lkey};
    struct ibv_recv_wr request={.wr_id=((uint64_t)WR_DESCRIPTOR_RECEIVE<<56)|i,.sg_list=&span,.num_sge=1},*bad;
    int error=control->receive(control->pair,&request,&bad);
    if(error){errno=error<0?-error:error;return -1;}
  }
  uint32_t posted=1,peer_posted;
  return exchange(socket,&posted,&peer_posted,sizeof posted,sizeof peer_posted,&peer->provider);
}

/* The bytes [offset, offset + bytes) cut alike at both ends: requests of at most MESH_STRIPE_CHUNK, none crossing a
   registered region; the next request's length. */
static inline uint32_t stripe_piece(const struct mesh_device *device,uint64_t offset,uint64_t remaining){
  uint64_t room=device->extent-offset%device->extent,length=remaining<MESH_STRIPE_CHUNK?remaining:MESH_STRIPE_CHUNK;
  return (uint32_t)(length<room?length:room);
}
static inline int stripe_post(struct mesh_peer *peer,int direction,uint64_t offset,uint32_t length,uint64_t word,uint64_t value,int last){
  const struct mesh_device *device=peer->device;
  struct mesh_queue *queue=&peer->provider.queues[1];
  struct ibv_sge span={.addr=(uintptr_t)device->wire+offset,.length=length,.lkey=device->regions[offset/device->extent]->lkey};
  uint64_t slot=peer->head[direction]&(MESH_CHUNKS-1);
  peer->chunks[direction][slot]=(struct mesh_chunk){.word=word,.value=value,.frames=length?(length+4095)/4096:1,.last=(uint32_t)last};
  int error;
  if(direction==MESH_SEND){
    struct ibv_send_wr request={.wr_id=((uint64_t)WR_DATA_SEND<<56)|slot,.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
    error=queue->send(queue->pair,&request,&bad);
  } else {
    struct ibv_recv_wr request={.wr_id=((uint64_t)WR_DATA_RECEIVE<<56)|slot,.sg_list=&span,.num_sge=1},*bad;
    error=queue->receive(queue->pair,&request,&bad);
  }
  if(error)return error<0?-error:error;
  peer->frames[direction]+=peer->chunks[direction][slot].frames;
  peer->head[direction]++;
  return 0;
}
/* The frames a stripe takes from its first request on: whether a queue with `capacity` frames and `held` in flight has
   room for its next piece. */
static inline int stripe_room(uint32_t held,uint32_t capacity,uint32_t length){return held+(length?(length+4095)/4096:1)<=capacity;}

/* A stripe in progress at this end, sending or receiving: its descriptor and the bytes posted so far. */
struct stripe_cut { struct mesh_descriptor d; uint64_t done; int active; };

/* The stripe service of one link, until it fails or the bridge stops: every completion of its queue pairs, every
   descriptor's receives posted, every registered ring's requests for this link taken in order.  0 on the bridge's
   stop, else the failure. */
static int peer_serve(struct mesh_peer *peer,int control){
  struct hdr *m=peer->M;
  struct mesh_queue *queues=peer->provider.queues;
  int (*poll)(struct ibv_cq *,int,struct ibv_wc *)=queues[0].poll;
  struct ibv_wc done[16];
  struct stripe_cut out={0},in={0};
  uint32_t ring=0;
  uint64_t quiet=0;
  const uint32_t send_capacity=queues[1].send_capacity,receive_capacity=queues[1].receive_capacity,descriptor_capacity=
    queues[0].send_capacity<MESH_DESCRIPTORS?queues[0].send_capacity:MESH_DESCRIPTORS;
  for(;;){
    if(stop || !atomic_load_explicit(&peer->running,memory_order_acquire))return 0;
    int moved=0;
    int count=poll(peer->provider.completion,16,done);
    if(count<0)return EIO;
    if(count)atomic_thread_fence(memory_order_acquire);
    for(int i=0;i<count;i++){
      if(done[i].status){fprintf(stderr,"link %u stripe completion status %d\n",peer->index,done[i].status);return EIO;}
      const uint64_t kind=done[i].wr_id>>56,slot=done[i].wr_id&((UINT64_C(1)<<56)-1);
      if(kind==WR_DESCRIPTOR_RECEIVE){
        peer->pending[peer->pending_head++&(MESH_DESCRIPTORS-1)]=peer->received[slot];
        struct mesh_queue *queue=&queues[0];
        struct ibv_sge span={.addr=(uintptr_t)peer->device->wire+peer->received_at+slot*sizeof(struct mesh_descriptor),.length=sizeof(struct mesh_descriptor),
          .lkey=peer->device->regions[(peer->received_at+slot*sizeof(struct mesh_descriptor))/peer->device->extent]->lkey};
        struct ibv_recv_wr request={.wr_id=done[i].wr_id,.sg_list=&span,.num_sge=1},*bad;
        int error=queue->receive(queue->pair,&request,&bad);
        if(error)return error<0?-error:error;
      } else if(kind==WR_DESCRIPTOR_SEND)peer->descriptors_done++;
      else {
        const int direction=kind==WR_DATA_SEND?MESH_SEND:MESH_RECEIVE;
        struct mesh_chunk *chunk=&peer->chunks[direction][slot];
        peer->frames[direction]-=chunk->frames;
        peer->tail[direction]++;
        if(chunk->last && chunk->word)atomic_store_explicit((_Atomic uint64_t *)((char *)m+chunk->word),chunk->value,memory_order_release);
      }
      moved=1;
    }
    /* receives: each descriptor's pieces posted at its offset as the data pair's receive queue has room */
    for(;;){
      if(!in.active){
        if(peer->pending_tail==peer->pending_head)break;
        in.d=peer->pending[peer->pending_tail++&(MESH_DESCRIPTORS-1)];in.done=0;in.active=1;
      }
      uint32_t length=stripe_piece(peer->device,in.d.offset+in.done,in.d.bytes-in.done);
      if(!stripe_room(peer->frames[MESH_RECEIVE],receive_capacity,length) || peer->head[MESH_RECEIVE]-peer->tail[MESH_RECEIVE]>=MESH_CHUNKS)break;
      int last=in.done+length>=in.d.bytes;
      int error=stripe_post(peer,MESH_RECEIVE,in.d.offset+in.done,length,in.d.word,in.d.value,last);
      if(error==ENOMEM)break;
      if(error)return error;
      in.done+=length;moved=1;
      if(last)in.active=0;
    }
    /* sends: the next request of each registered ring for this link, its descriptor then its pieces */
    for(uint32_t scanned=0;scanned<MESH_RINGS;scanned++){
      if(!out.active){
        if(peer->descriptors_sent-peer->descriptors_done>=descriptor_capacity)break;
        uint32_t r=(ring+scanned)%MESH_RINGS;
        uint64_t at=atomic_load_explicit(&mesh_rings(m)[r],memory_order_acquire);
        if(!at)continue;
        struct mesh_requests *requests=(void *)((char *)m+at);
        uint64_t k=atomic_load_explicit(&requests->taken,memory_order_acquire);
        struct mesh_request *request=&requests->request[k&(requests->entries-1)];
        if(atomic_load_explicit(&request->ready,memory_order_acquire)!=k+1 || request->link!=peer->index)continue;
        out.d=(struct mesh_descriptor){.offset=request->offset,.bytes=request->bytes,.word=request->word,.value=request->value};
        out.done=0;out.active=1;
        atomic_store_explicit(&requests->taken,k+1,memory_order_release);
        ring=(r+1)%MESH_RINGS;
        uint64_t slot=peer->descriptors_sent&(MESH_DESCRIPTORS-1);
        peer->sent[slot]=out.d;
        uint64_t offset=peer->sent_at+slot*sizeof(struct mesh_descriptor);
        struct ibv_sge span={.addr=(uintptr_t)peer->device->wire+offset,.length=sizeof(struct mesh_descriptor),.lkey=peer->device->regions[offset/peer->device->extent]->lkey};
        struct ibv_send_wr request_wr={.wr_id=((uint64_t)WR_DESCRIPTOR_SEND<<56)|slot,.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
        atomic_thread_fence(memory_order_release);
        int error=queues[0].send(queues[0].pair,&request_wr,&bad);
        if(error)return error<0?-error:error;
        peer->descriptors_sent++;moved=1;
      }
      while(out.active){
        uint32_t length=stripe_piece(peer->device,out.d.offset+out.done,out.d.bytes-out.done);
        if(!stripe_room(peer->frames[MESH_SEND],send_capacity,length) || peer->head[MESH_SEND]-peer->tail[MESH_SEND]>=MESH_CHUNKS)break;
        int last=out.done+length>=out.d.bytes;
        int error=stripe_post(peer,MESH_SEND,out.d.offset+out.done,length,out.d.word,out.d.value,last);
        if(error==ENOMEM)break;
        if(error)return error;
        out.done+=length;moved=1;
        if(last)out.active=0;
      }
      if(out.active)break;
    }
    /* the bridge's stop, the peer's end (its connection), the interface's: looked at between passes with nothing to do */
    if(moved){quiet=0;continue;}
    if(++quiet&4095)continue;
    struct kevent64_s event;struct timespec none={0,0};
    int events=kevent64(peer->server_events,NULL,0,&event,1,0,&none);
    if(events>0 && event.filter==EVFILT_READ && event.ident==(uint64_t)control)return ECONNRESET;
  }
}

/* The link's stripe service: paired (key 0), served, and paired again once its peer's bridge is back, until the
   bridge stops. */
static void *peer_run(void *argument){
  struct mesh_peer *peer=argument;
  pthread_setname_np("mesh.stripes");
  _Atomic int *running=&peer->running;
  struct mesh_port_info *port=&mesh_links(peer->M)[peer->index].port;
  while(!stop && atomic_load_explicit(running,memory_order_acquire)){
    peer->provider=(struct mesh_verbs){.device=peer->device,.wire=peer->wire,.peer=mesh_links(peer->M)[peer->index].peer,
      .local_address=peer->local_address,.remote_address=peer->remote_address,.service=peer->service,.events=peer->server_events,
      .m=peer->M,.session=MESH_ABSENT,.key=0,.running=running,.take=peer_take,.acceptor=peer};
    atomic_store_explicit(&port->phase,MESH_PAIRING,memory_order_release);
    int control=verbs_up(&peer->provider,2,peer_configure,peer);
    if(control<0){
      int failure=errno?errno:EIO;
      while(!down_pair(&peer->provider))atomic_store_explicit(&port->code,errno?errno:EIO,memory_order_relaxed);
      if(stop || !atomic_load_explicit(running,memory_order_acquire))break;
      if(!pairing_retried(failure) && failure!=ECANCELED)fprintf(stderr,"link %u stripe pairing: %s\n",peer->index,strerror(failure));
      struct kevent64_s timer,event;
      EV_SET64(&timer,2,EVFILT_TIMER,EV_ADD|EV_ONESHOT,NOTE_USECONDS,1000,0,0,0);
      kevent64(peer->server_events,&timer,1,&event,1,0,NULL);
      continue;
    }
    struct kevent64_s watched;
    EV_SET64(&watched,control,EVFILT_READ,EV_ADD|EV_CLEAR,0,0,0,0,0);
    kevent64(peer->server_events,&watched,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
    __atomic_store_n(&mesh_links(peer->M)[peer->index].bandwidth,peer->provider.bandwidth,__ATOMIC_RELAXED);
    atomic_store_explicit(&port->phase,MESH_PAIRED,memory_order_release);
    fprintf(stderr,"link %u stripe service paired\n",peer->index);
    peer->head[0]=peer->head[1]=peer->tail[0]=peer->tail[1]=0;peer->frames[0]=peer->frames[1]=0;
    peer->descriptors_sent=peer->descriptors_done=0;peer->pending_head=peer->pending_tail=0;
    int failure=peer_serve(peer,control);
    atomic_store_explicit(&port->phase,MESH_STOPPED,memory_order_release);
    if(failure){atomic_store_explicit(&port->code,failure,memory_order_relaxed);fprintf(stderr,"link %u stripe service ended: %s\n",peer->index,strerror(failure));}
    EV_SET64(&watched,control,EVFILT_READ,EV_DELETE,0,0,0,0,0);
    kevent64(peer->server_events,&watched,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
    shutdown(control,SHUT_RDWR);close(control);
    while(!down_pair(&peer->provider))atomic_store_explicit(&port->code,errno?errno:EIO,memory_order_relaxed);
  }
  atomic_store_explicit(&port->phase,MESH_STOPPED,memory_order_release);
  return NULL;
}

/* ---- sessions ---- */
/* Session `s`'s links started: each prepared and paired by a thread of its own. */
static int session_start(struct mesh_link *links,struct mesh_peer *peers,uint32_t link_count,uint32_t s,struct hdr *m){
  struct mesh_session *session=&mesh_sessions(m)[s];
  uint64_t key=atomic_load_explicit(&session->key,memory_order_acquire);
  pid_t pid=(pid_t)atomic_load_explicit(&session->pid,memory_order_acquire);
  for(uint32_t i=0;i<link_count;i++){
    struct mesh_link *link=&links[(size_t)s*link_count+i];
    struct mesh_peer *peer=&peers[i];
    link->M=m;link->index=i;link->session=s;link->key=key;link->pid=pid;link->peer=peer;link->network=-1;
    atomic_store_explicit(&link->finished,0,memory_order_relaxed);
    atomic_store_explicit(&link->progressing,1,memory_order_release);
    link->provider=(struct mesh_verbs){.device=peer->device,.wire=peer->wire,.peer=mesh_links(m)[i].peer,
      .local_address=peer->local_address,.remote_address=peer->remote_address,.service=peer->service,.events=link->events,
      .m=m,.session=s,.key=key,.running=&link->progressing,.take=peer_take,.acceptor=peer};
    struct mesh_port_info *port=mesh_session_port(m,s,i);
    int error=link_prepare(link);
    if(error){link_error(link,error,1);atomic_store_explicit(&link->finished,1,memory_order_release);continue;}
    atomic_store_explicit(&port->phase,MESH_PAIRING,memory_order_release);
    error=pthread_create(&link->controller,NULL,link_run,link);
    if(error){link_error(link,error,1);atomic_store_explicit(&link->finished,1,memory_order_release);link->controller=NULL;}
  }
  return 0;
}

/* design/algorithm-sources.md#programcopy */
int main(int argc,char **argv){
  const char *name=MESH_NAME;int me=0,layout=0;double pct=0;
  uint64_t arena_pages=0,block_pages=0,table_rows=0,window_pages=0,orders=4096;
  uint32_t link_count=0,device_count=0,qps=getenv("MESH_QPS")?(uint32_t)atoi(getenv("MESH_QPS")):1;
  struct mesh_peer *peers=calloc((size_t)argc,sizeof *peers);
  struct mesh_device *devices=calloc((size_t)argc,sizeof *devices);
  if(!peers || !devices)die("bridge configuration allocation");
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
      struct mesh_peer *peer=&peers[link_count];
      char *fields=strdup(argv[++i]);peer->configuration=fields;
      if(!fields)die("link configuration allocation");
      char *device=strsep(&fields,","),*node=strsep(&fields,","),*local=strsep(&fields,","),*remote=strsep(&fields,","),*service=strsep(&fields,",");
      if(!device || !*device || !node || !*node || !local || !*local || !remote || !*remote || fields)die("link requires device,peer,local-address,remote-address[,service]");
      uint32_t d=0;
      while(d<device_count && strcmp(devices[d].name,device))d++;
      if(d==device_count){devices[d].name=device;pthread_mutex_init(&devices[d].setup,NULL);device_count++;}
      peer->index=link_count++;peer->device=&devices[d];peer->node=(uint32_t)strtoul(node,NULL,10);
      peer->local_address=local;peer->remote_address=remote;peer->service=service?service:MESH_PORT;peer->listener=-1;
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
  if(geometry.zone_pages>=geometry.wire_pages)die("the window holds no more than the bridge's zone");
  if(layout){printf("%llu\n",(unsigned long long)length);return 0;}
  atexit(down);struct sigaction sa={0};sa.sa_handler=stop_bridge;
  sigaction(SIGINT,&sa,NULL);sigaction(SIGTERM,&sa,NULL);sigaction(SIGHUP,&sa,NULL);signal(SIGPIPE,SIG_IGN);
  shm_unlink(name);int fd=shm_open(name,O_CREAT|O_RDWR,MESH_MODE);if(fd<0)die("shm");
  if(ftruncate(fd,(off_t)length))die("ftruncate");fchmod(fd,MESH_MODE);
  struct hdr *m=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  if(m==MAP_FAILED)die("mmap");shm=name;
  *m=geometry;m->node=(uint32_t)me;m->version=MESH_VERSION;
  for(uint32_t r=0;r<mesh_rows(m);r++)atomic_store_explicit(&mesh_page(m)[r].mapping,MESH_ABSENT,memory_order_relaxed);
  /* the zone's pages are the bridge's: never handed out (mesh_arena_range starts past them) */
  mesh_bits_set(mesh_arena_bits(m),0,m->zone_pages);
  struct mesh_wire wire={0};
  if(wire_map(&wire,m,fd))die("transport page aliases");
  close(fd);
  /* every session's link, with a kqueue each for the bridge's life */
  struct mesh_link *links=aligned_alloc(_Alignof(struct mesh_link),(size_t)MESH_SESSIONS*(link_count?link_count:1)*sizeof *links);
  if(!links)die("session link allocation");
  memset(links,0,(size_t)MESH_SESSIONS*(link_count?link_count:1)*sizeof *links);
  struct ibv_wc *completion_outputs=calloc((size_t)MESH_SESSIONS*(link_count?link_count:1),sizeof *completion_outputs);
  if(!completion_outputs)die("completion output allocation");
  for(uint32_t s=0;s<MESH_SESSIONS;s++)for(uint32_t i=0;i<link_count;i++){
    struct mesh_link *link=&links[(size_t)s*link_count+i];
    link->events=kqueue();link->network=-1;link->completion=completion_outputs+(size_t)s*link_count+i;
    struct kevent64_s event;EV_SET64(&event,0,EVFILT_USER,EV_ADD|EV_CLEAR,0,0,0,0,0);
    if(link->events<0 || kevent64(link->events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("session control events");
    peers[i].waiting[s]=link->events;
  }
  for(uint32_t i=0;i<link_count;i++){
    struct mesh_peer *peer=&peers[i];
    peer->M=m;peer->wire=&wire;pthread_mutex_init(&peer->lock,NULL);
    peer->acceptor_events=kqueue();peer->server_events=kqueue();
    struct kevent64_s event;EV_SET64(&event,0,EVFILT_USER,EV_ADD|EV_CLEAR,0,0,0,0,0);
    if(peer->acceptor_events<0 || peer->server_events<0 || kevent64(peer->server_events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL) ||
       kevent64(peer->acceptor_events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("link control events");
    mesh_links(m)[i].peer=peer->node;
    snprintf(mesh_links(m)[i].device,sizeof mesh_links(m)[i].device,"%s",peer->device->name);
    atomic_store(&mesh_links(m)[i].port.phase,MESH_PAIRING);
    /* design/algorithm-sources.md#programcopy */
    struct mesh_verbs listening={.local_address=peer->local_address,.service=peer->service};
    if((uint32_t)me>peer->node && listener_up(&listening,&peer->listener)){
      fprintf(stderr,"pairing listener %s port %s: %s\n",peer->local_address,peer->service,strerror(errno));die("pairing listener");
    }
    /* the zone: this link's descriptors, received then sent */
    peer->received_at=(uint64_t)i*2*MESH_DESCRIPTORS*sizeof(struct mesh_descriptor);
    peer->sent_at=peer->received_at+MESH_DESCRIPTORS*sizeof(struct mesh_descriptor);
    peer->received=(void *)(wire.data+peer->received_at);peer->sent=(void *)(wire.data+peer->sent_at);
  }
  /* design/prepared-machine.md#M09 */
  /* the window registered at the start where its device's port is up (a port that is down registers at its first
     pairing), and each link's registered region's bytes published */
  for(uint32_t i=0;i<link_count;i++){
    struct ibv_port_attr port;
    if(!device_up(peers[i].device,&wire,m,&port))
      __atomic_store_n(&mesh_links(m)[i].extent,peers[i].device->extent,__ATOMIC_RELEASE);
  }
  /* design/prepared-machine.md#M26 */
  atomic_store_explicit(&control_memory,m,memory_order_relaxed);
  atomic_store(&m->bridge_pid,(uint64_t)getpid());atomic_store(&m->port.phase,MESH_PAIRING);
  __sync_synchronize();m->magic=MESH_MAGIC;
  fprintf(stderr,"bridge node %d: %u links, %u queue pairs per session link, arena %llu pages, window %u pages (zone %u), rows %u, orders %u, %d sessions, %d rings\n",
    me,link_count,qps,(unsigned long long)mesh_arena_pages(m),m->wire_pages,m->zone_pages,m->rows,m->orders,MESH_SESSIONS,MESH_RINGS);
  int status=0;
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_set_qos_class_np(&attributes,QOS_CLASS_USER_INTERACTIVE,0);
  for(uint32_t i=0;i<link_count;i++){
    atomic_store_explicit(&peers[i].running,1,memory_order_release);
    if(peers[i].listener>=0 && pthread_create(&peers[i].acceptor,NULL,peer_accept,&peers[i]))die("acceptor");
    if(pthread_create(&peers[i].server,&attributes,peer_run,&peers[i]))die("stripe service");
  }
  pthread_attr_destroy(&attributes);
  int running[MESH_SESSIONS]={0},ended[MESH_SESSIONS]={0};
  while(!stop){
    uint64_t notification=atomic_load_explicit(&m->control,memory_order_acquire);
    for(uint32_t s=0;s<MESH_SESSIONS;s++){
      struct mesh_session *session=&mesh_sessions(m)[s];
      const uint32_t request=atomic_load_explicit(&session->request,memory_order_acquire);
      const uint64_t pid=atomic_load_explicit(&session->pid,memory_order_acquire);
      if(request!=MESH_REQUEST_START)ended[s]=0;
      if(!running[s] && !ended[s] && request==MESH_REQUEST_START && pid){
        atomic_store_explicit(&session->served,1,memory_order_release);
        session_start(links,peers,link_count,s,m);
        running[s]=1;
        continue;
      }
      /* a slot whose process is gone before it started a session is free again */
      if(!running[s] && pid && request!=MESH_REQUEST_START && kill((pid_t)pid,0) && errno==ESRCH)
        atomic_compare_exchange_strong_explicit(&session->pid,&(uint64_t){pid},0,memory_order_acq_rel,memory_order_relaxed);
      if(!running[s])continue;
      int finished=1;
      for(uint32_t i=0;i<link_count;i++)finished&=atomic_load_explicit(&links[(size_t)s*link_count+i].finished,memory_order_acquire);
      if(!finished){
        /* a stop is the links' to finish with their peers' (link_run) */
        if(request!=MESH_REQUEST_START)for(uint32_t i=0;i<link_count;i++)trigger(links[(size_t)s*link_count+i].events);
        continue;
      }
      for(uint32_t i=0;i<link_count;i++){
        struct mesh_link *link=&links[(size_t)s*link_count+i];
        if(link->controller){pthread_join(link->controller,NULL);link->controller=NULL;}
      }
      running[s]=0;ended[s]=request==MESH_REQUEST_START;
      atomic_store_explicit(&session->served,0,memory_order_release);
      /* a session whose process is gone is free again (its allocations stay: nothing owns them) */
      if(pid && kill((pid_t)pid,0) && errno==ESRCH){
        atomic_store_explicit(&session->request,MESH_REQUEST_NONE,memory_order_relaxed);
        atomic_store_explicit(&session->pid,0,memory_order_release);
      }
      mesh_control_notify(m);
    }
    if(!stop)os_sync_wait_on_address(&m->control,notification,sizeof m->control,OS_SYNC_WAIT_ON_ADDRESS_SHARED);
  }
  for(uint32_t s=0;s<MESH_SESSIONS;s++)if(running[s]){
    for(uint32_t i=0;i<link_count;i++)link_stop(&links[(size_t)s*link_count+i]);
    for(uint32_t i=0;i<link_count;i++)if(links[(size_t)s*link_count+i].controller)pthread_join(links[(size_t)s*link_count+i].controller,NULL);
    atomic_store_explicit(&mesh_sessions(m)[s].served,0,memory_order_release);
  }
  for(uint32_t i=0;i<link_count;i++){
    atomic_store_explicit(&peers[i].running,0,memory_order_release);
    trigger(peers[i].server_events);trigger(peers[i].acceptor_events);
    if(peers[i].listener>=0)shutdown(peers[i].listener,SHUT_RDWR);
  }
  for(uint32_t i=0;i<link_count;i++){
    pthread_join(peers[i].server,NULL);
    if(peers[i].listener>=0)pthread_join(peers[i].acceptor,NULL);
  }
  for(uint32_t i=0;i<device_count;i++)if(!down_device(&devices[i])){fprintf(stderr,"verbs teardown failed: %s\n",strerror(errno));return 1;}
  for(uint32_t i=0;i<link_count;i++){
    struct mesh_peer *peer=&peers[i];
    atomic_store(&mesh_links(m)[i].port.phase,MESH_STOPPED);
    for(int c=0;c<peer->nparked;c++)close(peer->parked[c].fd);
    if(peer->listener>=0)close(peer->listener);
    close(peer->acceptor_events);close(peer->server_events);
    free(peer->configuration);
  }
  for(size_t k=0;k<(size_t)MESH_SESSIONS*link_count;k++){close(links[k].events);free(links[k].requests);free(links[k].receive);}
  for(uint32_t i=0;i<device_count;i++)pthread_mutex_destroy(&devices[i].setup);
  free(completion_outputs);free(links);free(devices);free(peers);munmap(wire.data,wire.length);
  atomic_store(&m->port.phase,MESH_STOPPED);
  atomic_store_explicit(&control_memory,NULL,memory_order_relaxed);
  munmap(m,length);return status;
}
