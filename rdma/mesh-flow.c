#include "mesh-verbs.h"
#include "mesh-dataflow.h"
#include <pthread.h>
#include <sys/event.h>

/* design/algorithm-sources.md#programcopy */
struct mesh_send_edge {
  _Alignas(128) struct ibv_sge span;
  struct mesh_instance *instance; uint32_t queue,row,chunks,offset,remaining;
  struct ibv_send_wr request;
};
_Static_assert(sizeof(struct mesh_send_edge)==256 && offsetof(struct mesh_send_edge,request)+offsetof(struct ibv_send_wr,wr)<=128,"mesh_send_edge native request");
struct mesh_receive_request {_Alignas(64) struct ibv_sge span;struct ibv_recv_wr request;};
_Static_assert(sizeof(struct mesh_receive_request)==64 && _Alignof(struct mesh_receive_request)==64,"mesh_receive_request");
struct mesh_ready {uint64_t head,tail;size_t first,mask;};
struct mesh_receive_record {
  _Alignas(32) struct mesh_buffer *buffer;
  _Atomic uint32_t *entry; uint32_t row,flags,frame;
};
_Static_assert(sizeof(struct mesh_receive_record)==32 && _Alignof(struct mesh_receive_record)==32,"mesh_receive_record");
struct mesh_receive {
  uint32_t *pages,first;
  struct mesh_receive_request *requests;
  struct mesh_receive_record *records;
  struct mesh_ready ready;
  struct mesh_notice_reader returns;
};
struct mesh_worker {struct mesh_link *link;pthread_t thread;uint32_t direction;};
struct mesh_link {
  struct mesh_worker workers[2];
  uint32_t worker_count,index;
  pthread_t controller;
  int events;
  char *configuration;
  _Atomic int progressing;
  struct hdr *M;struct mesh_verbs provider;int qps;uint64_t client;
  struct mesh_receive receive[MESH_QPS];
  uint32_t *send_offsets,*send_ready;
  struct mesh_send_edge *send_edges;
  struct mesh_ready ready[MESH_QPS];
  struct mesh_notice_reader notices;
  struct mesh_instance *instances;
  uint32_t instance_count;
};
static int link_receive(struct mesh_link *link,uint32_t q);
/* design/algorithm-sources.md#programcopy */
static void link_receive_destroy(struct mesh_receive *receive){
  free(receive->requests);free(receive->records);free(receive->pages);
}
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
  atomic_store_explicit(&port->phase,MESH_STOPPED,memory_order_release);
  link_stop(link);
  for(uint32_t i=0;i<link->instance_count;i++)mesh_result_conclude(&link->instances[i].status,MESH_RESULT(MESH_RESULT_LINK,link->index,code));
}
/* design/algorithm-sources.md#programcopy */
static int link_configure(void *state,int socket,uint64_t client){
  struct mesh_link *link=state;struct hdr *m=link->M;
  link->notices=mesh_notice_reader_init(m,mesh_notice_queue(m,client,link->index));
  uint64_t payload=(uint64_t)m->block*m->pgsz;
  uint32_t bytes[2*MESH_QPS]={0};
  size_t count=0,ready_count=0,queue_counts[MESH_QPS]={0};
  memset(link->send_offsets,0,((size_t)mesh_rows(m)+1)*sizeof *link->send_offsets);
  for(uint32_t q=0;q<(uint32_t)link->qps;q++)for(uint32_t i=0;i<atomic_load(mesh_order_length(m,link->client,link->index*m->qps+q,MESH_SEND));i++){
    struct mesh_transfer transfer=mesh_transfers(m,link->client,link->index*m->qps+q,MESH_SEND)[i];
    count+=transfer.count;queue_counts[q]+=transfer.count;
    for(uint32_t value=0;value<transfer.count;value++)link->send_offsets[transfer.local_row+value*transfer.stride+1]++;
  }
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    size_t capacity=1;
    while(capacity<queue_counts[q])capacity*=2;
    link->ready[q]=(struct mesh_ready){.first=ready_count,.mask=capacity-1};
    ready_count+=capacity;
  }
  free(link->send_edges);link->send_edges=aligned_alloc(_Alignof(struct mesh_send_edge),(count?count:1)*sizeof *link->send_edges);
  free(link->send_ready);link->send_ready=calloc(ready_count,sizeof *link->send_ready);
  if(!link->send_edges || !link->send_ready)return -1;
  for(uint32_t row=0;row<mesh_rows(m);row++)link->send_offsets[row+1]+=link->send_offsets[row];
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    uint32_t counts[3]={atomic_load(mesh_order_length(m,link->client,link->index*m->qps+q,MESH_SEND)),atomic_load(mesh_order_length(m,link->client,link->index*m->qps+q,MESH_RECEIVE)),mesh_rows(m)},peer_counts[3];
    if(exchange(socket,counts,peer_counts,sizeof counts,sizeof peer_counts,m,client,link->provider.deadline))return -1;
    uint32_t sends=counts[MESH_SEND],receives=counts[MESH_RECEIVE];
    if(sends!=peer_counts[MESH_RECEIVE] || receives!=peer_counts[MESH_SEND]){
      fprintf(stderr,"transfer count mismatch queue=%u local=%u,%u peer=%u,%u\n",q,sends,receives,peer_counts[MESH_SEND],peer_counts[MESH_RECEIVE]);
      errno=EPROTO;return -1;
    }
    struct mesh_transfer *bindings=calloc(sends+receives+1,sizeof *bindings),*peer=calloc(receives?receives:1,sizeof *peer);
    if(!bindings || !peer){free(bindings);free(peer);return -1;}
    struct mesh_transfer *out=bindings,*in=bindings+sends;
    memcpy(out,mesh_transfers(m,link->client,link->index*m->qps+q,MESH_SEND),sends*sizeof *out);
    memcpy(in,mesh_transfers(m,link->client,link->index*m->qps+q,MESH_RECEIVE),receives*sizeof *in);
    for(int direction=0;direction<2;direction++){
      struct mesh_transfer *transfers=direction==MESH_SEND?out:in;
      uint64_t largest=0;
      for(uint32_t i=0;i<counts[direction];i++)if(transfers[i].bytes>largest)largest=transfers[i].bytes;
      bytes[2*q+direction]=(uint32_t)(largest<payload?largest:payload)+sizeof(struct mesh_wire_tag);
      if(counts[direction] && (bytes[2*q+direction]+4095)/4096>link->provider.capacity[q][direction]){free(bindings);free(peer);errno=EMSGSIZE;return -1;}
    }
    int error=exchange(socket,out,peer,sends*sizeof *out,receives*sizeof *peer,m,client,link->provider.deadline);
    uint32_t chunks_count=0;
    for(uint32_t i=0;i<receives && !error;i++){
      if(in[i].binding!=peer[i].binding || in[i].count!=peer[i].count || in[i].bytes!=peer[i].bytes){
        fprintf(stderr,"transfer mismatch queue=%u index=%u local=%u,%u,%llu peer=%u,%u,%llu\n",q,i,in[i].binding,in[i].count,(unsigned long long)in[i].bytes,peer[i].binding,peer[i].count,(unsigned long long)peer[i].bytes);
        errno=EPROTO;error=-1;break;
      }
      uint32_t chunks=(uint32_t)((in[i].bytes+payload-1)/payload);
      chunks_count+=in[i].count*chunks;
    }
    if(error){free(bindings);free(peer);return error;}
    struct mesh_receive *receive=&link->receive[q];
    link_receive_destroy(receive);
    size_t capacity=1;
    while(capacity<chunks_count)capacity*=2;
    *receive=(struct mesh_receive){.first=receives?in[0].pool:0,.ready={.tail=chunks_count,.mask=capacity-1},
      .returns=mesh_notice_reader_init(m,mesh_notice_queue(m,client,m->links+link->index*m->qps+q))};
    receive->pages=calloc(capacity,sizeof *receive->pages);
    receive->requests=aligned_alloc(_Alignof(struct mesh_receive_request),(chunks_count?chunks_count:1)*sizeof *receive->requests);
    receive->records=aligned_alloc(_Alignof(struct mesh_receive_record),(receives?peer_counts[2]:1)*sizeof *receive->records);
    if(!receive->pages || !receive->requests || !receive->records){free(bindings);free(peer);return -1;}
    for(uint32_t i=0;i<chunks_count;i++){
      uint32_t page=receive->first+i*m->block;receive->pages[i]=i;
      struct mesh_receive_request *request=&receive->requests[i];
      *request=(struct mesh_receive_request){.span=link->provider.device->spans[page/m->block],.request={.wr_id=page,.sg_list=&request->span,.num_sge=1}};
      request->span.length=bytes[2*q+MESH_RECEIVE];
    }
    for(uint32_t i=0;i<receives;i++){
      uint32_t chunks=(uint32_t)((in[i].bytes+payload-1)/payload);
      for(uint32_t value=0;value<in[i].count;value++){
        uint32_t row=in[i].local_row+value*in[i].stride;
        for(uint32_t chunk=0;chunk<chunks;chunk++)
          receive->records[peer[i].local_row+value*peer[i].stride+chunk]=(struct mesh_receive_record){
            .buffer=&mesh_buffers(m)[row],.entry=mesh_page(m)+row+chunk,.row=row,
            .flags=(chunk+1==chunks)|((!in[i].stride)<<1),.frame=value};
      }
    }
    for(uint32_t i=0;i<sends;i++)for(uint32_t value=0;value<out[i].count;value++){
      uint32_t row=out[i].local_row+value*out[i].stride;
      uint32_t chunks=(uint32_t)((out[i].bytes+payload-1)/payload);
      uint32_t edge=link->send_offsets[row]++;struct mesh_send_edge *source=&link->send_edges[edge];
      *source=(struct mesh_send_edge){.span={.length=bytes[2*q+MESH_SEND]},.instance=out[i].stride?&link->instances[value]:NULL,.queue=q,.row=row,.chunks=chunks,.remaining=chunks,
        .request={.wr_id=edge,.sg_list=&source->span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED}};
    }
    free(bindings);free(peer);
  }
  memmove(link->send_offsets+1,link->send_offsets,(size_t)mesh_rows(m)*sizeof *link->send_offsets);link->send_offsets[0]=0;
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    int error=link_receive(link,q);
    if(error && error!=ENOMEM && error!=EAGAIN){errno=error;return -1;}
  }
  uint32_t posted=1,peer_posted;
  return exchange(socket,&posted,&peer_posted,sizeof posted,sizeof peer_posted,m,client,link->provider.deadline);
}
/* design/algorithm-sources.md#programcopy */
static int link_receive(struct mesh_link *link,uint32_t q){
  struct hdr *m=link->M;
  struct mesh_receive *receive=&link->receive[q];
  struct mesh_ready *ready=&receive->ready;
  for(;;){
    int error=0;
    while(ready->head!=ready->tail){
      struct ibv_recv_wr *bad=NULL;
      error=ibv_post_recv(link->provider.pairs[q],&receive->requests[receive->pages[ready->head&ready->mask]].request,&bad);
      if(error)break;
      ready->head++;
    }
    if(error<0)error=-error;
    if(error && error!=ENOMEM && error!=EAGAIN)return error;
    uint32_t row=mesh_notice_take(&receive->returns);
    if(row==MESH_ABSENT)return error;
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    for(uint32_t offset=0;offset<buffer->pages;offset+=m->block){
      receive->pages[ready->tail++&ready->mask]=(atomic_load_explicit(&mesh_page(m)[row+offset/m->block],memory_order_relaxed)-receive->first)/m->block;
      atomic_store_explicit(&mesh_page(m)[row+offset/m->block],MESH_ABSENT,memory_order_relaxed);
    }
    mesh_buffer_reset(m,row);
    mesh_instance_release(&link->instances[buffer->binding]);
  }
}
/* design/algorithm-sources.md#programkernel_call */
static int link_send_ready(struct mesh_link *link,uint32_t q){
  struct mesh_ready *ready=&link->ready[q];
  while(ready->head!=ready->tail){
    uint32_t edge=link->send_ready[ready->first+(ready->head&ready->mask)];
    struct mesh_send_edge *source=&link->send_edges[edge];
    uint32_t page=atomic_load_explicit(&mesh_page(link->M)[source->row+source->offset],memory_order_acquire);
    struct mesh_wire_tag *tag=mesh_tag(link->M,page);
    struct mesh_buffer *buffer=&mesh_buffers(link->M)[source->row];
    atomic_store_explicit(&tag->value,((uint64_t)buffer->invocation<<32)|(source->row+source->offset),memory_order_relaxed);
    struct ibv_sge span=link->provider.device->spans[page/link->M->block];
    source->span.addr=span.addr;source->span.lkey=span.lkey;
    struct ibv_send_wr *bad=NULL;
    int error=ibv_post_send(link->provider.pairs[q],&source->request,&bad);
    if(error)return error<0?-error:error;
    source->offset++;
    ready->head++;
    if(source->offset!=source->chunks)link->send_ready[ready->first+(ready->tail++&ready->mask)]=edge;
  }
  return 0;
}
/* design/algorithm-sources.md#programcopy */
static int mesh_progress(struct mesh_link *link,uint32_t direction){
  struct hdr *m=link->M;struct mesh_verbs *v=&link->provider;
  struct ibv_wc *completions=v->completions+direction;
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    int count=ibv_poll_cq(v->completion_queues[2*q+direction],1,completions);
    if(count<0){link_error(link,count,3);return count;}
    int error=direction==MESH_SEND?link_send_ready(link,q):0;
    if(error && error!=ENOMEM && error!=EAGAIN){link_error(link,error,1);return error;}
    for(int i=0;i<count;i++){
      struct ibv_wc *wc=&completions[i];
      if(wc->status){link_error(link,wc->status,2);return wc->status;}
      if(direction==MESH_RECEIVE){
        uint32_t page=(uint32_t)wc->wr_id;
        struct mesh_wire_tag *tag=mesh_tag(m,page);
        struct mesh_receive *receive=&link->receive[q];
        uint64_t tag_value=atomic_load_explicit(&tag->value,memory_order_relaxed);
        uint32_t invocation=(uint32_t)(tag_value>>32);
        struct mesh_receive_record record=receive->records[(uint32_t)tag_value];
        atomic_store_explicit(record.entry,page,memory_order_relaxed);
        if(record.flags&1){
          record.buffer->invocation=invocation;
          mesh_publish(m,record.row,(uint64_t)invocation+1);
          if(record.flags&2)mesh_shared_release(link->instances,link->instance_count);
          else atomic_store_explicit(&link->instances[record.frame].invocation,invocation,memory_order_relaxed);
          mesh_buffer_release(m,record.row);
        }
      } else {
        struct mesh_send_edge *source=&link->send_edges[(uint32_t)wc->wr_id];
        if(!--source->remaining){
          source->offset=0;source->remaining=source->chunks;
          mesh_buffer_release(m,source->row);
          if(source->instance)mesh_instance_release(source->instance);
          else mesh_shared_release(link->instances,link->instance_count);
        }
      }
    }
    if(direction==MESH_RECEIVE){
      error=link_receive(link,q);
      if(error && error!=ENOMEM && error!=EAGAIN){link_error(link,error,1);return error;}
    }
  }
  return 0;
}

/* design/algorithm-sources.md#programkernel_call */
static void link_publications(struct mesh_link *link){
  uint32_t row;
  while((row=mesh_notice_take(&link->notices))!=MESH_ABSENT){
    for(uint32_t at=link->send_offsets[row];at<link->send_offsets[row+1];at++){
      uint32_t q=link->send_edges[at].queue;
      struct mesh_ready *ready=&link->ready[q];
      link->send_ready[ready->first+(ready->tail++&ready->mask)]=at;
      int error=link_send_ready(link,q);
      if(error){
        if(error!=ENOMEM && error!=EAGAIN){link_error(link,error,1);return;}
        if(mesh_progress(link,MESH_SEND))return;
      }
    }
  }
}

/* design/algorithm-sources.md#programkernel_call */
static void *link_progress(void *argument){
  struct mesh_worker *worker=argument;
  pthread_setname_np(worker->direction==MESH_SEND?"mesh.rdma.send":"mesh.rdma.receive");
  while(atomic_load_explicit(&worker->link->progressing,memory_order_acquire)){
    if(mesh_progress(worker->link,worker->direction))break;
    if(worker->direction==MESH_SEND)link_publications(worker->link);
  }
  return NULL;
}

/* design/algorithm-sources.md#programcopy */
static void *link_run(void *argument){
  struct mesh_link *link=argument;struct hdr *m=link->M;
  struct mesh_port_info *port=&mesh_links(m)[link->index].port;
  uint32_t transfers=0;
  for(uint32_t q=0;q<m->qps;q++)for(int d=0;d<2;d++)transfers+=atomic_load(mesh_order_length(m,link->client,link->index*m->qps+q,d));
  if(!transfers)return NULL;
  atomic_store_explicit(&port->phase,MESH_PAIRING,memory_order_release);
  int control=verbs_up(&link->provider,m,link->qps,link_configure,link,link->client);
  if(control<0){
    if(errno!=ECANCELED)link_error(link,errno?errno:EIO,1);
  } else {
    struct kevent64_s changes[2];
    EV_SET64(&changes[0],control,EVFILT_READ,EV_ADD|EV_CLEAR,0,0,0,0,0);
    EV_SET64(&changes[1],(uint32_t)link->client,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,0,0,0);
    int error=kevent64(link->events,changes,2,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)?errno:0;
    if(!error){
      __atomic_store_n(&mesh_links(m)[link->index].bandwidth,link->provider.bandwidth,__ATOMIC_RELAXED);
      atomic_store_explicit(&port->phase,MESH_PAIRED,memory_order_release);
    }
    for(uint32_t d=0;d<2 && !error;d++){
      link->workers[d]=(struct mesh_worker){.link=link,.direction=d};
      error=pthread_create(&link->workers[d].thread,NULL,link_progress,&link->workers[d]);
      if(error)break;
      link->worker_count++;
    }
    if(error)link_error(link,error,1);
    while(atomic_load_explicit(&link->progressing,memory_order_acquire)){
      struct kevent64_s event;
      int count=kevent64(link->events,NULL,0,&event,1,0,NULL);
      if(count<0){if(errno==EINTR)continue;link_error(link,errno,4);break;}
      if(event.flags&EV_ERROR)link_error(link,event.data,4);
      else if(event.filter==EVFILT_PROC)link_error(link,ECANCELED,4);
      else if(event.filter==EVFILT_READ)
        link_error(link,event.flags&EV_EOF?(event.fflags?event.fflags:ECONNRESET):EPROTO,4);
    }
    shutdown(control,SHUT_RDWR);
    while(link->worker_count)pthread_join(link->workers[--link->worker_count].thread,NULL);
    EV_SET64(&changes[0],(uint32_t)link->client,EVFILT_PROC,EV_DELETE,0,0,0,0,0);
    kevent64(link->events,changes,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
    close(control);
  }
  while(!down_pair(&link->provider))link_error(link,errno?errno:EIO,1);
  if(link->provider.listener>=0){close(link->provider.listener);link->provider.listener=-1;}
  atomic_store_explicit(&port->phase,MESH_STOPPED,memory_order_release);
  return NULL;
}

/* design/algorithm-sources.md#programcopy */
int main(int argc,char **argv){
  const char *name=MESH_NAME;int me=0,layout=0;double pct=0;
  uint64_t arena_pages=0,block_pages=0;
  uint32_t link_count=0,device_count=0,qps=getenv("MESH_QPS")?(uint32_t)atoi(getenv("MESH_QPS")):1;
  struct mesh_link *links=calloc((size_t)argc,sizeof *links);
  struct mesh_device *devices=calloc((size_t)argc,sizeof *devices);
  if(!links || !devices)die("bridge configuration allocation");
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
  for(uint32_t r=0;r<mesh_rows(m);r++)atomic_store_explicit(&mesh_page(m)[r],MESH_ABSENT,memory_order_relaxed);
  struct mesh_wire wire={0};
  if(wire_map(&wire,m,fd))die("transport page aliases");
  close(fd);
  for(uint32_t i=0;i<link_count;i++){
    struct mesh_link *link=&links[i];
    link->M=m;link->qps=(int)qps;link->provider.wire=&wire;
    link->send_offsets=calloc((size_t)mesh_rows(m)+1,sizeof *link->send_offsets);
    link->provider.completions=calloc(2,sizeof *link->provider.completions);
    if(!link->send_offsets || !link->provider.completions)die("link allocation");
    link->events=kqueue();
    struct kevent64_s event;EV_SET64(&event,0,EVFILT_USER,EV_ADD|EV_CLEAR,0,0,0,0,0);
    if(link->events<0 || kevent64(link->events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("link control events");
    mesh_links(m)[i].peer=link->provider.peer;
    snprintf(mesh_links(m)[i].device,sizeof mesh_links(m)[i].device,"%s",link->provider.device->name);
    atomic_store(&mesh_links(m)[i].port.phase,MESH_PAIRING);
  }
  int status=0;
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
    free(link->provider.completions);free(link->send_offsets);free(link->send_edges);free(link->send_ready);
    for(uint32_t q=0;q<qps;q++)link_receive_destroy(&link->receive[q]);
    free(link->configuration);
  }
  for(uint32_t i=0;i<device_count;i++)pthread_mutex_destroy(&devices[i].setup);
  free(devices);free(links);munmap(wire.data,wire.length);free(wire.spans);
  atomic_store(&m->port.phase,MESH_STOPPED);munmap(m,length);return status;
}
