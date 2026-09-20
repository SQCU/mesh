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
_Static_assert(sizeof(struct ibv_send_wr *)==8,"M15");
/* design/prepared-machine.md#M24 */
union mesh_network_event {
  struct kern_event_msg header;
  unsigned char bytes[KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data)];
};
_Static_assert(sizeof(union mesh_network_event)==48,"M24");
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
  offsetof(struct prepared_receive,input)==80 && offsetof(struct prepared_receive,argument)==88,"M08 M09");
#if MESH_TRACE
/* design/prepared-machine.md#M27 */
struct mesh_trace { _Alignas(32) uint64_t identity; uint64_t begin,middle,end; };
_Static_assert(sizeof(struct mesh_trace)==32 && _Alignof(struct mesh_trace)==32,"M27");
#endif
struct mesh_link {
  pthread_t workers[3];
  uint32_t worker_count,publication_count,cursor_count,index;
  pthread_t controller;
  int events,network;
  union mesh_network_event network_event;
  char *configuration;
  _Atomic int progressing;
  struct hdr *M;struct mesh_verbs provider;int qps;uint64_t client;
  struct prepared_receive *receive;
  struct ibv_send_wr *requests;
  struct ibv_sge *spans;
  struct mesh_send *publications,**cursors;
  struct ibv_wc *completion[2];
  struct mesh_cancellation *cancel;
#if MESH_TRACE
  struct mesh_trace *trace[2];
  size_t traced[2],trace_capacity[2];
#endif
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
  if(link->cancel)mesh_cancel(link->M,link->cancel);
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
  uint64_t payload=(uint64_t)m->block*m->pgsz;
  size_t count=0;
  uint32_t incoming=0;
  link->qps=(int)(m->qps*tx->slots);
  link->provider.request_capacity=1;
  for(uint32_t q=0;q<m->qps;q++)for(uint32_t d=0;d<2;d++){
    uint32_t channel=link->index*m->qps+q;
    struct mesh_transfer *transfers=mesh_transfers(m,link->client,channel,d);
    uint32_t frames=0;
    for(uint32_t i=0;i<atomic_load(mesh_order_length(m,link->client,channel,d));i++){
      uint32_t chunks=(uint32_t)((transfers[i].bytes+payload-1)/payload);
      frames+=chunks;
      if(d==MESH_SEND)count+=(size_t)chunks*transfers[i].count;
      else incoming+=chunks*transfers[i].count;
    }
    if(frames>link->provider.request_capacity)link->provider.request_capacity=frames;
  }
  free(link->requests);
  link->requests=aligned_alloc(128,((count?count:1)*144+127)&~(size_t)127);
  link->spans=(void *)(link->requests+count);
  if(!link->requests)return ENOMEM;
  link->provider.completion_entries[MESH_SEND]=link->publication_count;
  link->provider.completion_entries[MESH_RECEIVE]=incoming;
#if MESH_TRACE
  /* design/prepared-machine.md#M27 */
  for(int d=0;d<2;d++){
    link->trace_capacity[d]=(size_t)link->provider.completion_entries[d]*(tx->invocations?tx->invocations:1);
    link->traced[d]=0;
    link->trace[d]=aligned_alloc(32,(link->trace_capacity[d]+1)*sizeof(struct mesh_trace));
    if(!link->trace[d])return ENOMEM;
  }
#endif
  return 0;
}

/* design/prepared-machine.md#M06 */
/* design/prepared-machine.md#M07 */
/* design/prepared-machine.md#M08 */
/* design/prepared-machine.md#M09 */
/* design/algorithm-sources.md#programcopy */
static int link_transfer_order(const void *a,const void *b){
  const struct mesh_transfer *left=a,*right=b;
  return (left->first>right->first)-(left->first<right->first);
}

/* design/algorithm-sources.md#programcopy */
/* design/prepared-machine.md#M08 */
static int link_configure(void *state,int socket,uint64_t client){
  struct mesh_link *link=state;struct hdr *m=link->M;
  uint64_t payload=(uint64_t)m->block*m->pgsz;
  struct mesh_tx *tx=(void *)mesh_events(m,mesh_notice_queue(m,client,link->index));
  struct mesh_tx peer_tx;
  if(exchange(socket,tx,&peer_tx,sizeof *tx,sizeof peer_tx,m,client,link->provider.deadline))return -1;
  link->cursors=calloc((size_t)link->qps,sizeof *link->cursors);
  if(!link->cursors)return -1;
  struct mesh_send *last[link->qps],*repeat[link->qps];
  memset(last,0,sizeof last);memset(repeat,0,sizeof repeat);
  /* design/prepared-machine.md#M08 */
  uint32_t invocations=tx->invocations?tx->invocations:1;
  size_t receive_count=(size_t)link->provider.completion_entries[MESH_RECEIVE]*invocations;
  link->receive=aligned_alloc(128,(receive_count?receive_count:1)*sizeof *link->receive);
  if(!link->receive)return -1;
  uint32_t next=0,received=0;
  for(uint32_t q=0;q<m->qps;q++){
    uint32_t channel=link->index*m->qps+q;
    uint32_t counts[2]={atomic_load(mesh_order_length(m,client,channel,MESH_SEND)),atomic_load(mesh_order_length(m,client,channel,MESH_RECEIVE))},peer_counts[2];
    if(exchange(socket,counts,peer_counts,sizeof counts,sizeof peer_counts,m,client,link->provider.deadline))return -1;
    struct mesh_transfer *out=mesh_transfers(m,client,channel,MESH_SEND),*in=mesh_transfers(m,client,channel,MESH_RECEIVE);
    qsort(out,counts[MESH_SEND],sizeof *out,link_transfer_order);
    struct mesh_transfer *peer=calloc(peer_counts[MESH_SEND]?peer_counts[MESH_SEND]:1,sizeof *peer);
    if(!peer)return -1;
    if(exchange(socket,out,peer,counts[MESH_SEND]*sizeof *out,peer_counts[MESH_SEND]*sizeof *peer,m,client,link->provider.deadline)){free(peer);return -1;}
    for(uint32_t i=0;i<counts[MESH_SEND];i++)for(uint32_t slot=0;slot<out[i].count;slot++){
      uint32_t row=out[i].local_row+slot*out[i].stride;
      uint32_t chunks=(uint32_t)((out[i].bytes+payload-1)/payload);
      uint32_t publication=slot*tx->count+out[i].first;
      /* design/prepared-machine.md#M04 */
      uint32_t stream=q*tx->slots+slot;
      struct mesh_send *cell=link->publications+(size_t)publication*invocations;
      cell->pair=(uintptr_t)link->provider.queues[stream].pair;
      cell->request=(uintptr_t)(link->requests+next);cell->next=0;
      if(last[stream])last[stream]->next=(uintptr_t)cell;else link->cursors[stream]=cell;
      last[stream]=cell;
      if(out[i].stride&&!repeat[stream])repeat[stream]=cell;
      for(uint32_t k=0;k<chunks;k++){
        uint32_t j=next+k;
        uint32_t page=(uint32_t)atomic_load_explicit(&mesh_page(m)[row+k].mapping,memory_order_relaxed);
        struct ibv_sge span=link->provider.device->spans[page/m->block];
        uint64_t remaining=out[i].bytes-k*payload;
        span.length=(uint32_t)(remaining<payload?remaining:payload);
        link->spans[j]=span;
        link->requests[j]=(struct ibv_send_wr){.next=k+1==chunks?NULL:link->requests+j+1,.sg_list=link->spans+j,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=k+1==chunks?IBV_SEND_SIGNALED:0};
      }
      next+=chunks;
    }
    for(uint32_t i=0;i<counts[MESH_RECEIVE];i++){
      uint32_t peer_index=0;
      while(peer_index<peer_counts[MESH_SEND] && peer[peer_index].binding!=in[i].binding)peer_index++;
      if(peer_index==peer_counts[MESH_SEND]){free(peer);errno=EPROTO;return -1;}
      in[i].first=peer[peer_index].first;
    }
    qsort(in,counts[MESH_RECEIVE],sizeof *in,link_transfer_order);
    for(uint32_t i=0;i<counts[MESH_RECEIVE];i++){
      uint32_t chunks=(uint32_t)((in[i].bytes+payload-1)/payload);
      for(uint32_t slot=0;slot<in[i].count;slot++){
        uint32_t row=in[i].local_row+slot*in[i].stride;
        for(uint32_t k=0;k<chunks;k++){
          uint32_t page=(uint32_t)atomic_load_explicit(&mesh_page(m)[row+k].mapping,memory_order_relaxed);
          struct ibv_sge span=link->provider.device->spans[page/m->block];
          uint64_t remaining=in[i].bytes-k*payload;
          span.length=(uint32_t)(remaining<payload?remaining:payload);
          /* design/prepared-machine.md#M08 */
          /* design/prepared-machine.md#M09 */
          struct mesh_queue *queue=&link->provider.queues[q*tx->slots+slot];
          struct prepared_receive *records=link->receive+(size_t)received++*invocations;
          struct mesh_publication *delivery=mesh_publication_at(m,row+k);
          uintptr_t input=(uintptr_t)&delivery->argument;
          if(delivery->device_input)input=(uintptr_t)m+delivery->device_input;
          else if(delivery->sends)input=(uintptr_t)m+delivery->targets[0].stream;
          for(uint32_t t=0;t<invocations;t++){
            struct prepared_receive *record=records+t;
            *record=(struct prepared_receive){
              .request={.wr_id=(uintptr_t)record,.sg_list=&record->span,.num_sge=1},
              .span=span,.input=(_Atomic uint64_t *)(input+(delivery->device_input?sizeof(struct mesh_input_status)*(size_t)t:0)),
              .argument=delivery->device_input?(k+1==chunks):1,
              .next=&records[t+1<invocations?t+1:0].request,.pair=queue->pair};
          }
          struct ibv_recv_wr *bad;
          int error=queue->receive(queue->pair,&records->request,&bad);
          if(error){free(peer);errno=error<0?-error:error;return -1;}
        }
      }
    }
    free(peer);
  }
  link->cursor_count=0;
  for(int q=0;q<link->qps;q++)if(last[q]){
    struct mesh_send *end=last[q];
    for(uint32_t t=1;t<invocations && repeat[q];t++){
      for(struct mesh_send *source=repeat[q];;source=(void *)source->next){
        struct mesh_send *cell=source+t;
        *cell=(struct mesh_send){.pair=source->pair,.request=source->request};
        last[q]->next=(uintptr_t)cell;last[q]=cell;
        if(source==end)break;
      }
    }
    link->cursors[link->cursor_count++]=link->cursors[q];
  }
  uint32_t posted=1,peer_posted;
#if MESH_TRACE
  fprintf(stderr,"{\"trace_layout\":%u,\"rank\":%u,\"send_base\":%llu,\"receive_base\":%llu,\"invocations\":%u,\"send_capacity\":%zu,\"receive_capacity\":%zu}\n",
    link->index,m->node,(unsigned long long)(uintptr_t)link->publications,(unsigned long long)(uintptr_t)link->receive,
    invocations,link->trace_capacity[MESH_SEND],link->trace_capacity[MESH_RECEIVE]);
#endif
  return exchange(socket,&posted,&peer_posted,sizeof posted,sizeof peer_posted,m,client,link->provider.deadline);
}

/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M06 */
/* design/prepared-machine.md#M15 */
/* design/algorithm-sources.md#independent-native-queues */
static __attribute__((always_inline)) inline void *link_send_drain(struct mesh_link *link,uint32_t streams){
  struct mesh_send **cursors=link->cursors,*record=cursors[0];
  uint32_t stream=0;
  struct ibv_send_wr *bad;
  int (*post)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **)=link->provider.queues[0].send;
#if MESH_TRACE
  /* design/prepared-machine.md#M27 */
  struct mesh_trace *trace=link->trace[MESH_SEND];
#endif
  for(;;){
    if((streams==1 || record) && atomic_load_explicit(&record->ready,memory_order_acquire)){
#if MESH_TRACE
        uint64_t observed=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#endif
        struct mesh_send *next=(void *)record->next;
#if MESH_TRACE
        uint64_t posting=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#endif
        int error=post((struct ibv_qp *)record->pair,(struct ibv_send_wr *)record->request,&bad);
#if MESH_TRACE
        uint64_t posted=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        trace[link->traced[MESH_SEND]++]=(struct mesh_trace){(uintptr_t)record,observed,posting,posted};
#endif
        if(error){link_error(link,error<0?-error:error,1);return NULL;}
        if(streams==1 && !next)return NULL;
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
  if(link->cursor_count==1)return link_send_drain(link,1);
  return link_send_drain(link,link->cursor_count);
}

/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#independent-native-queues */
static void *link_send_completions(void *argument){
  struct mesh_link *link=argument;
  struct mesh_queue queue=link->provider.queues[0];
  struct ibv_wc *completion=link->completion[MESH_SEND];
  pthread_setname_np("mesh.rdma.send.cq");
  while(atomic_load_explicit(&link->progressing,memory_order_acquire)){
    int count=queue.poll[MESH_SEND](queue.completions[MESH_SEND],1,completion);
    if(count<0){link_error(link,count,3);return NULL;}
    if(!count)continue;
    if(completion->status){link_error(link,completion->status,2);return NULL;}
  }
  return NULL;
}

/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#programcopy */
static void *link_receive_progress(void *argument){
  struct mesh_link *link=argument;
  struct mesh_queue queue=link->provider.queues[0];
  struct ibv_wc *completion=link->completion[MESH_RECEIVE];
  int (*post)(struct ibv_qp *,struct ibv_recv_wr *,struct ibv_recv_wr **)=queue.receive;
  pthread_setname_np("mesh.rdma.receive");
#if MESH_TRACE
  /* design/prepared-machine.md#M27 */
  struct mesh_trace *trace=link->trace[MESH_RECEIVE];
#endif
  while(atomic_load_explicit(&link->progressing,memory_order_acquire)){
    int count=queue.poll[MESH_RECEIVE](queue.completions[MESH_RECEIVE],1,completion);
#if MESH_TRACE
    uint64_t polled=count>0?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
#endif
    if(count<0){link_error(link,count,3);return NULL;}
    if(!count)continue;
    if(completion->status){link_error(link,completion->status,2);return NULL;}
    /* design/prepared-machine.md#M08 */
    struct prepared_receive *record=(void *)(uintptr_t)completion->wr_id;
    atomic_store_explicit(record->input,record->argument,memory_order_release);
#if MESH_TRACE
    uint64_t published=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#endif
    struct ibv_recv_wr *bad;
    int error=post(record->pair,record->next,&bad);
#if MESH_TRACE
    uint64_t posted=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    trace[link->traced[MESH_RECEIVE]++]=(struct mesh_trace){(uintptr_t)record,polled,published,posted};
#endif
    if(error){link_error(link,error<0?-error:error,1);return NULL;}
  }
  return NULL;
}

/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#meshresult */
static void link_close(struct mesh_link *link,int *control){
  atomic_store_explicit(&link->progressing,0,memory_order_release);
  if(link->network>=0){close(link->network);link->network=-1;}
  if(*control>=0)shutdown(*control,SHUT_RDWR);
  while(link->worker_count)pthread_join(link->workers[--link->worker_count],NULL);
#if MESH_TRACE
  /* design/prepared-machine.md#M27 */
  for(int d=0;d<2;d++){
    for(size_t i=0;i<link->traced[d];i++){
      struct mesh_trace t=link->trace[d][i];
      fprintf(stderr,"{\"native_trace\":%u,\"direction\":%d,\"event\":%zu,\"identity\":%llu,\"ns\":[%llu,%llu,%llu]}\n",
        link->index,d,i,(unsigned long long)t.identity,(unsigned long long)t.begin,(unsigned long long)t.middle,(unsigned long long)t.end);
    }
    free(link->trace[d]);link->trace[d]=NULL;link->traced[d]=0;
  }
#endif
  if(*control>=0){close(*control);*control=-1;}
  while(!down_pair(&link->provider))link_error(link,errno?errno:EIO,1);
  if(link->provider.listener>=0){close(link->provider.listener);link->provider.listener=-1;}
  free(link->receive);link->receive=NULL;
  free(link->cursors);link->cursors=NULL;
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
      void *(*progress[3])(void *)={link_send_completions,link_receive_progress,link_send_progress};
      pthread_attr_t attributes;
      pthread_attr_init(&attributes);
      pthread_attr_set_qos_class_np(&attributes,QOS_CLASS_USER_INTERACTIVE,0);
      for(uint32_t d=0;d<2+(link->publication_count!=0) && !error;d++){
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
    if(stop || atomic_load_explicit(&m->client,memory_order_acquire)!=link->client)break;
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

/* design/algorithm-sources.md#programcopy */
int main(int argc,char **argv){
  const char *name=MESH_NAME;int me=0,layout=0;double pct=0;
  uint64_t arena_pages=0,block_pages=0;
  uint32_t link_count=0,device_count=0,qps=getenv("MESH_QPS")?(uint32_t)atoi(getenv("MESH_QPS")):1;
  struct mesh_link *links=aligned_alloc(_Alignof(struct mesh_link),(size_t)argc*sizeof *links);
  struct mesh_device *devices=calloc((size_t)argc,sizeof *devices);
  if(!links || !devices)die("bridge configuration allocation");
  memset(links,0,(size_t)argc*sizeof *links);
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-I") && i+1<argc)me=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-M") && i+1<argc)pct=atof(argv[++i]);
    else if((!strcmp(argv[i],"-A") || !strcmp(argv[i],"-B")) && i+1<argc){
      char kind=argv[i][1],*end;uint64_t pages=strtoull(argv[++i],&end,10);
      if(*end || !pages || pages>INT32_MAX)die("configured page count");
      if(kind=='A')arena_pages=pages;else block_pages=pages;
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
  uint64_t length=mesh_layout(&geometry,pg,(uint32_t)block_pages,(uint32_t)arena_pages,link_count,qps);
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
  struct ibv_wc *completion_outputs=calloc(2*(link_count?link_count:1),sizeof *completion_outputs);
  if(!completion_outputs)die("completion output allocation");
  for(uint32_t p=0;p<link_count;p++)for(uint32_t d=0;d<2;d++)links[p].completion[d]=completion_outputs+d*link_count+p;
  /* design/prepared-machine.md#M26 */
  atomic_store_explicit(&control_memory,m,memory_order_relaxed);
  atomic_store(&m->bridge_pid,(uint64_t)getpid());atomic_store(&m->port.phase,MESH_PAIRING);
  __sync_synchronize();m->magic=MESH_MAGIC;
  fprintf(stderr,"bridge node %d: %u links, %u queue pairs per link\n",me,link_count,qps);
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
    while(!stop && atomic_load_explicit(&m->client,memory_order_acquire)==client){
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
  free(completion_outputs);free(devices);free(links);munmap(wire.data,wire.length);free(wire.spans);
  atomic_store(&m->port.phase,MESH_STOPPED);
  atomic_store_explicit(&control_memory,NULL,memory_order_relaxed);
  munmap(m,length);return status;
}
