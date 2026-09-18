#include "mesh-verbs.h"
#include "mesh-dataflow.h"
#include <pthread.h>
#include <sys/event.h>

/* design/prepared-machine.md#M05 */
struct prepared_send {
  _Alignas(32) int (*post)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **);
  struct ibv_qp *pair;
  struct ibv_send_wr *request;
  struct ibv_send_wr **bad;
};
_Static_assert(sizeof(struct prepared_send)==32 && _Alignof(struct prepared_send)==32,"M05");
_Static_assert(sizeof(struct ibv_send_wr)==128,"M06");
_Static_assert(sizeof(struct ibv_sge)==16,"M07 M09");
_Static_assert(sizeof(struct ibv_recv_wr)==32,"M08");
_Static_assert(sizeof(struct ibv_wc)==48,"M11");
_Static_assert(sizeof(struct ibv_send_wr *)==8,"M15");
/* design/prepared-machine.md#M16 */
struct prepared_cursor {
  struct mesh_link *link;
  _Atomic uint64_t *first,*repeat,*end;
  struct prepared_send *records;
};
_Static_assert(sizeof(struct prepared_cursor)==40,"M16");
struct mesh_receive {
  struct ibv_recv_wr *requests;
  struct ibv_sge *spans;
  uint32_t *rows;
  size_t count;
};
struct mesh_link {
  pthread_t workers[3],*senders;
  uint32_t worker_count,sender_count,index;
  pthread_t controller;
  int events;
  char *configuration;
  _Atomic int progressing;
  struct hdr *M;struct mesh_verbs provider;int qps;uint64_t client;
  struct mesh_receive *receive;
  struct prepared_send *sends;
  struct ibv_send_wr *requests;
  struct ibv_sge *spans;
  struct ibv_send_wr **bad;
  struct prepared_cursor *cursors;
  struct mesh_event_reader *returns;
  struct ibv_wc *completion[2];
  size_t send_count;
  struct mesh_instance *instances;
  uint32_t instance_count;
};
/* design/algorithm-sources.md#meshresult */
static void link_stop(struct mesh_link *link){
  atomic_store_explicit(&link->progressing,0,memory_order_release);
  struct kevent64_s event;EV_SET64(&event,0,EVFILT_USER,0,NOTE_TRIGGER,0,0,0,0);
  kevent64(link->events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
}
/* design/algorithm-sources.md#programcopy */
static void link_error(struct mesh_link *link,int64_t code,uint32_t domain){
  struct mesh_port_info *port=&mesh_links(link->M)[link->index].port;port->code=code;port->domain=domain;
  mesh_result_conclude(&link->M->result[link->client>>63],MESH_RESULT(MESH_RESULT_LINK,link->index,code));
  atomic_store_explicit(&port->phase,MESH_STOPPED,memory_order_relaxed);
  atomic_store_explicit(&port->prepared,link->client,memory_order_release);
  link_stop(link);
  for(uint32_t i=0;i<link->instance_count;i++)mesh_result_conclude(&link->instances[i].status,MESH_RESULT(MESH_RESULT_LINK,link->index,code));
}
/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M05 */
/* design/prepared-machine.md#M16 */
/* design/algorithm-sources.md#programcopy */
static int link_prepare(struct mesh_link *link){
  struct hdr *m=link->M;
  struct mesh_tx *tx=(void *)mesh_events(m,mesh_notice_queue(m,link->client,link->index));
  size_t count=(size_t)tx->count*tx->slots+tx->once;
  free(link->sends);free(link->bad);free(link->cursors);free(link->senders);
  link->sends=aligned_alloc(32,((count?count:1)*176+31)&~(size_t)31);
  link->requests=(void *)(link->sends+count);
  link->spans=(void *)(link->requests+count);
  link->bad=calloc(link->instance_count,sizeof *link->bad);
  link->cursors=calloc(link->instance_count,sizeof *link->cursors);
  link->senders=calloc(link->instance_count,sizeof *link->senders);
  if(!link->sends || !link->bad || !link->cursors || !link->senders)return ENOMEM;
  link->send_count=count;
  link->provider.completion_entries[MESH_SEND]=(uint32_t)count;
  uint32_t incoming=0;
  for(uint32_t q=0;q<m->qps;q++){
    uint32_t channel=link->index*m->qps+q;
    struct mesh_transfer *in=mesh_transfers(m,link->client,channel,MESH_RECEIVE);
    for(uint32_t i=0;i<atomic_load(mesh_order_length(m,link->client,channel,MESH_RECEIVE));i++)
      incoming+=in[i].count*(uint32_t)((in[i].bytes+(uint64_t)m->block*m->pgsz-1)/((uint64_t)m->block*m->pgsz));
  }
  link->provider.completion_entries[MESH_RECEIVE]=incoming;
  for(uint32_t slot=0;slot<link->instance_count;slot++)
    link->cursors[slot]=(struct prepared_cursor){link,tx->cells+(slot?tx->once+(size_t)slot*tx->count:0),tx->cells+tx->once+(size_t)slot*tx->count,tx->cells+tx->once+(size_t)(slot+1)*tx->count,link->sends};
  return 0;
}

/* design/prepared-machine.md#M06 */
/* design/prepared-machine.md#M07 */
/* design/prepared-machine.md#M08 */
/* design/prepared-machine.md#M09 */
/* design/algorithm-sources.md#programcopy */
static int link_configure(void *state,int socket,uint64_t client){
  struct mesh_link *link=state;struct hdr *m=link->M;
  uint64_t payload=(uint64_t)m->block*m->pgsz;
  struct mesh_tx *tx=(void *)mesh_events(m,mesh_notice_queue(m,client,link->index));
  struct mesh_tx peer_tx;
  if(exchange(socket,tx,&peer_tx,sizeof *tx,sizeof peer_tx,m,client,link->provider.deadline))return -1;
  link->receive=calloc(link->instance_count,sizeof *link->receive);
  link->returns=calloc(m->qps,sizeof *link->returns);
  if(!link->receive || !link->returns)return -1;
  for(uint32_t q=0;q<m->qps;q++){
    int error=mesh_event_reader_init(&link->returns[q],m,mesh_notice_queue(m,client,m->links+link->index*m->qps+q));
    if(error){errno=error;return -1;}
  }
  for(uint32_t slot=0;slot<link->instance_count;slot++){
    struct mesh_receive *receive=&link->receive[slot];
    receive->requests=calloc(m->rows,sizeof *receive->requests);
    receive->spans=calloc(m->rows,sizeof *receive->spans);
    if(!receive->requests || !receive->spans)return -1;
  }
  for(uint32_t q=0;q<m->qps;q++){
    uint32_t channel=link->index*m->qps+q;
    uint32_t counts[2]={atomic_load(mesh_order_length(m,client,channel,MESH_SEND)),atomic_load(mesh_order_length(m,client,channel,MESH_RECEIVE))},peer_counts[2];
    if(exchange(socket,counts,peer_counts,sizeof counts,sizeof peer_counts,m,client,link->provider.deadline))return -1;
    struct mesh_transfer *out=mesh_transfers(m,client,channel,MESH_SEND),*in=mesh_transfers(m,client,channel,MESH_RECEIVE);
    struct mesh_transfer *peer=calloc(peer_counts[MESH_SEND]?peer_counts[MESH_SEND]:1,sizeof *peer);
    if(!peer)return -1;
    if(exchange(socket,out,peer,counts[MESH_SEND]*sizeof *out,peer_counts[MESH_SEND]*sizeof *peer,m,client,link->provider.deadline)){free(peer);return -1;}
    for(uint32_t i=0;i<counts[MESH_SEND];i++)for(uint32_t slot=0;slot<out[i].count;slot++){
      struct mesh_queue *queue=&link->provider.queues[slot];
      uint32_t row=out[i].local_row+slot*out[i].stride;
      uint32_t chunks=(uint32_t)((out[i].bytes+payload-1)/payload);
      for(uint32_t k=0;k<chunks;k++){
        uint32_t j=slot*tx->count+out[i].first+k;
        uint32_t page=(uint32_t)atomic_load_explicit(&mesh_page(m)[row+k].mapping,memory_order_relaxed);
        struct ibv_sge span=link->provider.device->spans[page/m->block];
        uint64_t remaining=out[i].bytes-k*payload;
        span.length=(uint32_t)(remaining<payload?remaining:payload);
        link->spans[j]=span;
        link->requests[j]=(struct ibv_send_wr){.wr_id=((uint64_t)(out[i].stride?slot:UINT32_MAX)<<32)|row,.sg_list=link->spans+j,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=k+1==chunks?IBV_SEND_SIGNALED:0};
        link->sends[j]=(struct prepared_send){queue->send,queue->pair,link->requests+j,link->bad+slot};
      }
    }
    for(uint32_t i=0;i<counts[MESH_RECEIVE];i++){
      uint32_t peer_index=0;
      while(peer_index<peer_counts[MESH_SEND] && peer[peer_index].binding!=in[i].binding)peer_index++;
      if(peer_index==peer_counts[MESH_SEND]){free(peer);errno=EPROTO;return -1;}
      uint32_t chunks=(uint32_t)((in[i].bytes+payload-1)/payload);
      for(uint32_t slot=0;slot<in[i].count;slot++){
        struct mesh_receive *receive=&link->receive[slot];
        size_t offset=peer[peer_index].first-(slot?peer_tx.once:0),end=offset+chunks;
        if(end>receive->count){
          uint32_t *rows=realloc(receive->rows,end*sizeof *rows);
          if(!rows){free(peer);return -1;}receive->rows=rows;receive->count=end;
        }
        uint32_t row=in[i].local_row+slot*in[i].stride;
        for(uint32_t k=0;k<chunks;k++){
          uint32_t page=(uint32_t)atomic_load_explicit(&mesh_page(m)[row+k].mapping,memory_order_relaxed);
          struct ibv_sge span=link->provider.device->spans[page/m->block];
          uint64_t remaining=in[i].bytes-k*payload;
          span.length=(uint32_t)(remaining<payload?remaining:payload);
          atomic_store_explicit(&mesh_page(m)[row+k].stamp,(uint64_t)slot+1-link->instance_count,memory_order_relaxed);
          receive->spans[row+k]=span;
          receive->requests[row+k]=(struct ibv_recv_wr){.wr_id=((uint64_t)slot<<32)|(row+k),.sg_list=receive->spans+row+k,.num_sge=1};
          receive->rows[offset+k]=row+k;
        }
        struct mesh_buffer *buffer=mesh_buffers(m)+row;
        buffer->frame=slot;buffer->completions=in[i].stride!=0;
      }
    }
    free(peer);
  }
  for(uint32_t slot=0;slot<link->instance_count;slot++){
    struct mesh_receive *receive=&link->receive[slot];
    struct mesh_queue *queue=&link->provider.queues[slot];
    for(size_t i=0;i<receive->count;i++){
      struct ibv_recv_wr *bad;
      int error=queue->receive(queue->pair,receive->requests+receive->rows[i],&bad);
      if(error){errno=error<0?-error:error;return -1;}
    }
  }
  uint32_t posted=1,peer_posted;
  return exchange(socket,&posted,&peer_posted,sizeof posted,sizeof peer_posted,m,client,link->provider.deadline);
}

/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M05 */
/* design/prepared-machine.md#M16 */
/* design/algorithm-sources.md#independent-native-queues */
static void *link_send_progress(void *argument){
  const struct prepared_cursor *prepared=argument;
  struct mesh_link *link=prepared->link;
  _Atomic uint64_t *first=prepared->repeat,*end=prepared->end,*cell=prepared->first;
  if(first==end)first=cell;
  const struct prepared_send *records=prepared->records;
  pthread_setname_np("mesh.rdma.send");
  for(;;){
    uint64_t event=atomic_load_explicit(cell,memory_order_acquire);
    if(!event){
      if(!atomic_load_explicit(&link->progressing,memory_order_acquire))return NULL;
      continue;
    }
    struct prepared_send record=records[event-1];
    int error=record.post(record.pair,record.request,record.bad);
    if(error){
      if(error==ENOMEM || error==EAGAIN || error==-ENOMEM || error==-EAGAIN)continue;
      link_error(link,error<0?-error:error,1);return NULL;
    }
    atomic_store_explicit(cell,0,memory_order_release);
    if(++cell==end)cell=first;
  }
  return NULL;
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
    mesh_buffer_release(link->M,(uint32_t)completion->wr_id);
    uint32_t slot=(uint32_t)(completion->wr_id>>32);
    mesh_instance_release(link->instances+(slot==UINT32_MAX?0:slot),slot==UINT32_MAX?link->instance_count:1);
  }
  return NULL;
}

/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#programcopy */
static void *link_receive_progress(void *argument){
  struct mesh_link *link=argument;struct hdr *m=link->M;
  struct mesh_queue queue=link->provider.queues[0];
  struct mesh_page_entry *pages=mesh_page(m);
  uint32_t generations=link->instance_count;
  struct ibv_wc *completion=link->completion[MESH_RECEIVE];
  pthread_setname_np("mesh.rdma.receive");
  while(atomic_load_explicit(&link->progressing,memory_order_acquire)){
    int count=queue.poll[MESH_RECEIVE](queue.completions[MESH_RECEIVE],1,completion);
    if(count<0){link_error(link,count,3);return NULL;}
    if(!count)continue;
    if(completion->status){link_error(link,completion->status,2);return NULL;}
    uint32_t row=(uint32_t)completion->wr_id,slot=(uint32_t)(completion->wr_id>>32);
    uint64_t stamp=atomic_load_explicit(&pages[row].stamp,memory_order_relaxed)+generations;
    atomic_store_explicit(&pages[row].stamp,stamp,memory_order_release);
    struct mesh_publication *publication=mesh_publication_at(m,row);
    mesh_publish(m,publication->targets,publication->sends,publication->uses,pages+row,stamp);
    struct mesh_queue *native=&link->provider.queues[slot];
    struct ibv_recv_wr *bad;
    int error=native->receive(native->pair,link->receive[slot].requests+row,&bad);
    if(error){link_error(link,error<0?-error:error,1);return NULL;}
    if(publication->row!=MESH_ABSENT){
      row=publication->row;
      mesh_buffer_release(m,row);
      if(!mesh_buffers(m)[row].completions)mesh_instance_release(link->instances,link->instance_count);
    }
  }
  return NULL;
}

/* design/prepared-machine.md#M02 */
/* design/algorithm-sources.md#transport-retirement */
static void *link_retire_progress(void *argument){
  struct mesh_link *link=argument;struct hdr *m=link->M;
  pthread_setname_np("mesh.rdma.retire");
  while(atomic_load_explicit(&link->progressing,memory_order_acquire))for(uint32_t q=0;q<m->qps;q++){
    uint64_t event=mesh_event_take(&link->returns[q]);
    if(event==MESH_EVENT_ABSENT)continue;
    uint32_t row=(uint32_t)event;
    struct mesh_buffer *buffer=mesh_buffers(m)+row;
    uint32_t frame=buffer->frame,count=buffer->completions;
    uint32_t invocation=(uint32_t)(atomic_load_explicit(&mesh_page(m)[row+buffer->pages/m->block-1].stamp,memory_order_relaxed)-1);
    atomic_store_explicit(&buffer->references,buffer->initial,memory_order_relaxed);
    for(uint32_t i=0;i<count;i++){
      struct mesh_instance *instance=link->instances+frame+i;
      atomic_store_explicit(&instance->invocation,invocation,memory_order_relaxed);
      mesh_instance_release(instance,1);
    }
  }
  return NULL;
}

/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#meshresult */
static void link_close(struct mesh_link *link,int *control){
  atomic_store_explicit(&link->progressing,0,memory_order_release);
  if(*control>=0)shutdown(*control,SHUT_RDWR);
  while(link->sender_count)pthread_join(link->senders[--link->sender_count],NULL);
  while(link->worker_count)pthread_join(link->workers[--link->worker_count],NULL);
  if(*control>=0){close(*control);*control=-1;}
  while(!down_pair(&link->provider))link_error(link,errno?errno:EIO,1);
  if(link->provider.listener>=0){close(link->provider.listener);link->provider.listener=-1;}
  for(uint32_t slot=0;link->receive && slot<link->instance_count;slot++){
    free(link->receive[slot].requests);free(link->receive[slot].spans);free(link->receive[slot].rows);
  }
  for(uint32_t q=0;link->returns && q<link->M->qps;q++)free(link->returns[q].inputs);
  free(link->receive);link->receive=NULL;free(link->returns);link->returns=NULL;
  atomic_store_explicit(&mesh_links(link->M)[link->index].port.phase,MESH_STOPPED,memory_order_release);
}

/* design/prepared-machine.md#M16 */
/* design/algorithm-sources.md#programcopy */
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
    atomic_store_explicit(&port->phase,MESH_PAIRING,memory_order_release);
    control=verbs_up(&link->provider,m,(int)link->instance_count,link_configure,link,link->client);
    if(control<0)link_error(link,errno?errno:EIO,1);
    else {
      EV_SET64(&event,control,EVFILT_READ,EV_ADD|EV_CLEAR,0,0,0,0,0);
      error=kevent64(link->events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)?errno:0;
      if(!error){
        __atomic_store_n(&mesh_links(m)[link->index].bandwidth,link->provider.bandwidth,__ATOMIC_RELAXED);

      }
      void *(*progress[3])(void *)={link_retire_progress,link_send_completions,link_receive_progress};
      for(uint32_t d=0;d<3 && !error;d++){
        error=pthread_create(&link->workers[d],NULL,progress[d],link);
        if(error)break;
        link->worker_count++;
      }
      for(uint32_t slot=0;slot<link->instance_count && !error;slot++){
        if(link->cursors[slot].first==link->cursors[slot].end)continue;
        error=pthread_create(link->senders+link->sender_count,NULL,link_send_progress,link->cursors+slot);
        if(!error)link->sender_count++;
      }
      if(error)link_error(link,error,1);
      else {
        atomic_store_explicit(&port->phase,MESH_PAIRED,memory_order_relaxed);
        atomic_store_explicit(&port->prepared,link->client,memory_order_release);
      }
    }
  } else {
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
    else if(event.filter==EVFILT_READ)
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
  atexit(down);struct sigaction sa={0};sa.sa_handler=onsig;
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
    link->M=m;link->qps=(int)qps;link->provider.wire=&wire;
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
  uint64_t retired=0;
  atomic_store(&m->bridge_pid,(uint64_t)getpid());atomic_store(&m->port.phase,MESH_PAIRING);
  __sync_synchronize();m->magic=MESH_MAGIC;
  fprintf(stderr,"bridge node %d: %u links, %u queue pairs per link\n",me,link_count,qps);
  while(!stop){
    uint64_t retirement=atomic_load_explicit(&m->retired,memory_order_acquire);
    if(retired!=retirement){mesh_retired_release(m);retired=retirement;}
    uint64_t client=atomic_load_explicit(&m->client,memory_order_acquire);
    if(!client || atomic_load_explicit(&m->configured,memory_order_acquire)!=client)continue;
    atomic_store_explicit(&m->device_client,client,memory_order_seq_cst);
    if(atomic_load_explicit(&m->client,memory_order_seq_cst)!=client || atomic_load_explicit(&m->configured,memory_order_seq_cst)!=client){atomic_store_explicit(&m->device_client,0,memory_order_seq_cst);continue;}
    uint32_t started=0;
    for(uint32_t i=0;i<link_count;i++){
      links[i].client=client;atomic_store_explicit(&links[i].progressing,1,memory_order_release);
      links[i].instances=mesh_instances(m,client);links[i].instance_count=m->instance_count[client>>63];
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
    while(!stop && atomic_load_explicit(&m->client,memory_order_acquire)==client){}
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
    free(link->sends);free(link->bad);free(link->cursors);free(link->senders);
    for(uint32_t slot=0;link->receive && slot<link->instance_count;slot++){
      free(link->receive[slot].requests);free(link->receive[slot].spans);free(link->receive[slot].rows);
    }
    for(uint32_t q=0;link->returns && q<qps;q++)free(link->returns[q].inputs);
    free(link->receive);free(link->returns);
    free(link->configuration);
  }
  for(uint32_t i=0;i<device_count;i++)pthread_mutex_destroy(&devices[i].setup);
  free(completion_outputs);free(devices);free(links);munmap(wire.data,wire.length);free(wire.spans);
  atomic_store(&m->port.phase,MESH_STOPPED);munmap(m,length);return status;
}
