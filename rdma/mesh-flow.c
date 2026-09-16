#include "mesh-verbs.h"
#include "mesh-dataflow.h"
#include <pthread.h>

/* design/algorithm-sources.md#programcopy */
struct mesh_send_edge {uint32_t queue,row,chunks,offset,shared;};
struct mesh_ready {uint64_t head,tail;size_t first,mask;};
struct mesh_receive_record {
  _Alignas(32) uint32_t first; uint32_t stride,count,chunk,references,flags;
};
_Static_assert(sizeof(struct mesh_receive_record)==32 && _Alignof(struct mesh_receive_record)==32,"mesh_receive_record");
struct mesh_receive {
  uint32_t *pages;
  struct mesh_receive_record *records;
  struct mesh_ready ready;
  struct mesh_notice_reader returns;
};
struct mesh_worker {struct mesh_link *link;pthread_t thread;uint32_t direction;};
struct mesh_link {
  struct mesh_worker workers[2];
  uint32_t worker_count,index;
  pthread_t controller;
  char *configuration;
  _Atomic int progressing;
  struct hdr *M;struct mesh_verbs provider;int qps;uint64_t client;
  uint32_t bytes[2*MESH_QPS];
  struct mesh_receive receive[MESH_QPS];
  uint32_t *send_offsets,*send_ready;
  struct mesh_send_edge *send_edges;
  struct mesh_ready ready[MESH_QPS];
  struct mesh_notice_reader notices;
  struct mesh_instance *instances;
  uint32_t instance_count;
};
static int link_receive(struct mesh_link *link,uint32_t q);
static int link_returns(struct mesh_link *link,uint32_t q);
/* design/algorithm-sources.md#programcopy */
static void link_receive_destroy(struct mesh_receive *receive){
  free(receive->records);free(receive->pages);
}
/* design/algorithm-sources.md#programcopy */
static void link_error(struct mesh_link *link,int64_t code,uint32_t domain){
  struct mesh_port_info *port=&mesh_links(link->M)[link->index].port;port->code=code;port->domain=domain;
  atomic_store_explicit(&link->M->result[link->client>>63],MESH_RESULT(MESH_RESULT_LINK,link->index,code),memory_order_release);
  for(uint32_t i=0;i<link->instance_count;i++)mesh_result_conclude(&link->instances[i].status,MESH_RESULT(MESH_RESULT_LINK,link->index,code));
  atomic_store_explicit(&link->progressing,0,memory_order_release);
}
/* design/algorithm-sources.md#programcopy */
static int link_post(struct mesh_link *link,uint32_t q,int direction,uint32_t row,uint32_t page){
  struct hdr *m=link->M;struct mesh_verbs *v=&link->provider;
  struct ibv_sge span=v->device->spans[page/m->block];
  span.length=link->bytes[2*q+(uint32_t)direction];
  int error;
  if(direction==MESH_SEND){
    struct ibv_send_wr request={.wr_id=row,.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad=NULL;
    error=ibv_post_send(v->pairs[q],&request,&bad);
  } else {
    struct ibv_recv_wr request={.wr_id=page,.sg_list=&span,.num_sge=1},*bad=NULL;
    error=ibv_post_recv(v->pairs[q],&request,&bad);
  }
  return error<0?-error:error;
}
/* design/algorithm-sources.md#programcopy */
static int link_configure(void *state,int socket,uint64_t client){
  struct mesh_link *link=state;struct hdr *m=link->M;
  link->notices=mesh_notice_reader_init(m,mesh_notice_queue(m,client,link->index));
  uint64_t payload=(uint64_t)m->block*m->pgsz;
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
  free(link->send_edges);link->send_edges=calloc(count?count:1,sizeof *link->send_edges);
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
      uint64_t bytes=0;
      for(uint32_t i=0;i<counts[direction];i++)if(transfers[i].bytes>bytes)bytes=transfers[i].bytes;
      link->bytes[2*q+direction]=(uint32_t)(bytes<payload?bytes:payload)+sizeof(struct mesh_wire_tag);
      if(counts[direction] && (link->bytes[2*q+direction]+4095)/4096>link->provider.capacity[q][direction]){free(bindings);free(peer);errno=EMSGSIZE;return -1;}
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
    *receive=(struct mesh_receive){.ready={.tail=chunks_count,.mask=capacity-1},
      .returns=mesh_notice_reader_init(m,mesh_notice_queue(m,client,m->links+link->index*m->qps+q))};
    receive->pages=calloc(capacity,sizeof *receive->pages);
    receive->records=aligned_alloc(_Alignof(struct mesh_receive_record),(receives?peer_counts[2]:1)*sizeof *receive->records);
    if(!receive->pages || !receive->records){free(bindings);free(peer);return -1;}
    uint32_t first=receives?in[0].pool:0;
    for(uint32_t i=0;i<chunks_count;i++)receive->pages[i]=first+i*m->block;
    for(uint32_t i=0;i<receives;i++){
      uint32_t references=(uint32_t)atomic_load_explicit(&mesh_buffers(m)[in[i].local_row].ownership,memory_order_relaxed);
      uint32_t chunks=(uint32_t)((in[i].bytes+payload-1)/payload);
      for(uint32_t value=0;value<in[i].count;value++){
        uint32_t row=in[i].local_row+value*in[i].stride;
        atomic_fetch_sub_explicit(&mesh_buffers(m)[row].ownership,references,memory_order_relaxed);
        for(uint32_t chunk=0;chunk<chunks;chunk++)
          receive->records[peer[i].local_row+value*peer[i].stride+chunk]=(struct mesh_receive_record){
            .first=in[i].local_row,.stride=in[i].stride,.count=in[i].count,.chunk=chunk,.references=references,
            .flags=(!chunk)|((chunk+1==chunks)<<1)|((!in[i].stride)<<2)};
      }
    }
    for(uint32_t i=0;i<sends;i++)for(uint32_t value=0;value<out[i].count;value++){
      uint32_t row=out[i].local_row+value*out[i].stride;
      uint32_t chunks=(uint32_t)((out[i].bytes+payload-1)/payload);
      link->send_edges[link->send_offsets[row]++]=(struct mesh_send_edge){.queue=q,.row=row,.chunks=chunks,.shared=!out[i].stride};
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
  struct mesh_receive *receive=&link->receive[q];
  struct mesh_ready *ready=&receive->ready;
  while(ready->head!=ready->tail){
    int error=link_post(link,q,MESH_RECEIVE,0,receive->pages[ready->head&ready->mask]);
    if(error)return error;
    ready->head++;
  }
  return 0;
}
/* design/algorithm-sources.md#programkernel_call */
static int link_send_ready(struct mesh_link *link,uint32_t q){
  struct mesh_ready *ready=&link->ready[q];
  while(ready->head!=ready->tail){
    uint32_t edge=link->send_ready[ready->first+(ready->head&ready->mask)];
    struct mesh_send_edge *source=&link->send_edges[edge];
    uint32_t page=atomic_load_explicit(&mesh_buffer_pages(link->M,source->row)[source->offset],memory_order_acquire);
    struct mesh_wire_tag *tag=mesh_tag(link->M,page);
    struct mesh_buffer *buffer=&mesh_buffers(link->M)[source->row];
    atomic_store_explicit(&tag->value,((uint64_t)buffer->invocation<<32)|(source->row+source->offset),memory_order_relaxed);
    uint32_t end=source->offset+1;
    int error=link_post(link,q,MESH_SEND,end==source->chunks?edge:MESH_ABSENT,page);
    if(error)return error;
    source->offset=end;
    ready->head++;
    if(end!=source->chunks)link->send_ready[ready->first+(ready->tail++&ready->mask)]=edge;
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
    int error=direction==MESH_RECEIVE?link_returns(link,q):link_send_ready(link,q);
    if(error && error!=ENOMEM && error!=EAGAIN){link_error(link,error,1);return error;}
    for(int i=0;i<count;i++){
      struct ibv_wc *wc=&completions[i];
      if(wc->status){link_error(link,wc->status,2);return wc->status;}
      if(direction==MESH_RECEIVE){
        uint32_t page=(uint32_t)wc->wr_id;
        struct mesh_wire_tag *tag=mesh_tag(m,page);
        struct mesh_receive *receive=&link->receive[q];
        uint64_t tag_value=atomic_load_explicit(&tag->value,memory_order_relaxed);
        uint32_t frame=(uint32_t)(tag_value>>32);
        struct mesh_receive_record record=receive->records[(uint32_t)tag_value];
        uint32_t row=record.first+(frame%record.count)*record.stride;
        if(record.flags&1){
          atomic_fetch_add_explicit(&mesh_buffers(m)[row].ownership,record.references,memory_order_relaxed);
          atomic_store_explicit(&mesh_presence(m)[row],0,memory_order_relaxed);
          mesh_buffers(m)[row].invocation=frame;
        }
        atomic_store_explicit(&mesh_buffer_pages(m,row)[record.chunk],page,memory_order_relaxed);
        if(record.flags&2){
          mesh_publish(m,row,(uint64_t)frame+1);
          if(record.flags&4)mesh_shared_release(link->instances,link->instance_count);
        }
      } else {
        if((uint32_t)wc->wr_id!=MESH_ABSENT){
          struct mesh_send_edge *source=&link->send_edges[(uint32_t)wc->wr_id];
          uint32_t invocation=mesh_buffers(m)[source->row].invocation;
          source->offset=0;
          mesh_buffer_release(m,source->row);
          if(source->shared)mesh_shared_release(link->instances,link->instance_count);
          else mesh_instance_release(&link->instances[invocation%link->instance_count]);
        }
      }
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
      if(mesh_progress(link,MESH_SEND))return;
    }
  }
}

/* design/algorithm-sources.md#programcopy */
static int link_returns(struct mesh_link *link,uint32_t q){
  struct hdr *m=link->M;
  struct mesh_receive *receive=&link->receive[q];
  struct mesh_ready *ready=&receive->ready;
  uint32_t row=mesh_notice_take(&receive->returns);
  if(row!=MESH_ABSENT){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    for(uint32_t offset=0;offset<buffer->pages;offset+=m->block){
      receive->pages[ready->tail++&ready->mask]=atomic_load_explicit(&mesh_buffer_pages(m,row)[offset/m->block],memory_order_relaxed);
      atomic_store_explicit(&mesh_buffer_pages(m,row)[offset/m->block],MESH_ABSENT,memory_order_relaxed);
    }
    mesh_instance_release(&link->instances[buffer->invocation%link->instance_count]);
  }
  return link_receive(link,q);
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
  int setup=verbs_up(&link->provider,m,link->qps,link_configure,link,link->client);
  if(setup<0){
    if(errno!=ECANCELED)link_error(link,errno?errno:EIO,1);
  } else {
    int error=0;
    for(uint32_t d=0;d<2;d++){
      link->workers[d]=(struct mesh_worker){.link=link,.direction=d};
      error=pthread_create(&link->workers[d].thread,NULL,link_progress,&link->workers[d]);
      if(error)break;
      link->worker_count++;
    }
    if(error)link_error(link,error,1);
    else {
      __atomic_store_n(&mesh_links(m)[link->index].bandwidth,link->provider.bandwidth,__ATOMIC_RELAXED);
      atomic_store_explicit(&port->phase,MESH_PAIRED,memory_order_release);
    }
    while(link->worker_count)pthread_join(link->workers[--link->worker_count].thread,NULL);
  }
  while(!down_pair(&link->provider))link_error(link,errno?errno:EIO,1);
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
    for(uint32_t i=0;i<started;i++)atomic_store_explicit(&links[i].progressing,0,memory_order_release);
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
    free(link->provider.completions);free(link->send_offsets);free(link->send_edges);free(link->send_ready);
    for(uint32_t q=0;q<qps;q++)link_receive_destroy(&link->receive[q]);
    free(link->configuration);
  }
  for(uint32_t i=0;i<device_count;i++)pthread_mutex_destroy(&devices[i].setup);
  free(devices);free(links);munmap(wire.data,wire.length);free(wire.spans);
  atomic_store(&m->port.phase,MESH_STOPPED);munmap(m,length);return status;
}
