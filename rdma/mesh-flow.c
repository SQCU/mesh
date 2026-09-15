#include "mesh-verbs.h"
#include "mesh-dataflow.h"
#include <pthread.h>

/* design/algorithm-sources.md#programcopy */
struct mesh_send_edge {uint32_t queue,row,pages,offset;};
struct mesh_target {uint32_t row,publish;};
struct mesh_ready {uint32_t head,tail;};
struct mesh_queue {uint32_t pending,next,capacity,bytes;};
struct mesh_receive {uint32_t first,*next,length,failed;struct mesh_target *targets;};
struct mesh_worker {struct mesh_link *link;pthread_t thread;uint32_t direction;};
struct mesh_link {
  struct mesh_worker workers[2];
  uint32_t worker_count,index;
  pthread_t controller;
  char *configuration;
  _Atomic int progressing;
  struct hdr *M;struct mesh_verbs provider;int qps;uint64_t client;
  struct mesh_queue queues[2*MESH_QPS];
  struct mesh_receive receive[MESH_QPS];
  uint32_t *send_offsets,*send_ready;
  struct mesh_send_edge *send_edges;
  struct mesh_ready ready[MESH_QPS];
};
static int link_receive(struct mesh_link *link,uint32_t q);
/* design/algorithm-sources.md#programcopy */
static struct mesh_queue *link_queue(struct mesh_link *link,uint32_t q,int direction){return &link->queues[2*q+(uint32_t)direction];}
/* design/algorithm-sources.md#programcopy */
static void link_error(struct mesh_link *link,int64_t code,uint32_t domain){struct mesh_port_info *port=&mesh_links(link->M)[link->index].port;port->code=code;port->domain=domain;}
/* design/algorithm-sources.md#programcopy */
static int link_post(struct mesh_link *link,uint32_t q,int direction,uint32_t row,uint32_t page){
  struct hdr *m=link->M;struct mesh_verbs *v=&link->provider;
  struct mesh_queue *queue=link_queue(link,q,direction);
  struct ibv_sge span=v->device->spans[page/m->block];
  span.length=queue->bytes;
  int error;
  if(direction==MESH_SEND){
    struct ibv_send_wr request={.wr_id=row,.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad=NULL;
    error=ibv_post_send(v->pairs[q],&request,&bad);
  } else {
    struct ibv_recv_wr request={.wr_id=page,.sg_list=&span,.num_sge=1},*bad=NULL;
    error=ibv_post_recv(v->pairs[q],&request,&bad);
  }
  if(error){link_error(link,error,1);return error;}
  queue->pending++;
  return 0;
}
/* design/algorithm-sources.md#programcopy */
static int link_configure(void *state,int socket,uint64_t client){
  struct mesh_link *link=state;struct hdr *m=link->M;
  memset(link->queues,0,sizeof link->queues);
  uint64_t payload=(uint64_t)m->block*m->pgsz;
  size_t count=0,at=0;
  memset(link->send_offsets,0,((size_t)mesh_rows(m)+1)*sizeof *link->send_offsets);
  for(uint32_t q=0;q<(uint32_t)link->qps;q++)for(uint32_t i=0;i<atomic_load(mesh_order_length(m,link->client,link->index*m->qps+q,MESH_SEND));i++){
    struct mesh_transfer transfer=mesh_transfers(m,link->client,link->index*m->qps+q,MESH_SEND)[i];
    count+=transfer.count;
    for(uint32_t value=0;value<transfer.count;value++)link->send_offsets[transfer.local_row+value*transfer.stride+1]++;
  }
  free(link->send_edges);link->send_edges=calloc(count?count:1,sizeof *link->send_edges);
  free(link->send_ready);link->send_ready=calloc(count?count:1,sizeof *link->send_ready);
  if(!link->send_edges || !link->send_ready)return -1;
  for(uint32_t row=0;row<mesh_rows(m);row++)link->send_offsets[row+1]+=link->send_offsets[row];
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    uint32_t counts[2]={atomic_load(mesh_order_length(m,link->client,link->index*m->qps+q,MESH_SEND)),atomic_load(mesh_order_length(m,link->client,link->index*m->qps+q,MESH_RECEIVE))},peer_counts[2];
    if(exchange(socket,counts,peer_counts,sizeof counts,sizeof peer_counts,m,client,link->provider.deadline))return -1;
    uint32_t sends=counts[MESH_SEND],receives=counts[MESH_RECEIVE];
    link->ready[q]=(struct mesh_ready){(uint32_t)at,(uint32_t)at};
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
      struct mesh_queue *queue=link_queue(link,q,direction);
      struct mesh_transfer *transfers=direction==MESH_SEND?out:in;
      uint64_t bytes=0;
      for(uint32_t i=0;i<counts[direction];i++)if(transfers[i].bytes>bytes)bytes=transfers[i].bytes;
      queue->bytes=(uint32_t)(bytes<payload?bytes:payload)+sizeof(uint32_t);
      queue->capacity=link->provider.capacity[q][direction]/((queue->bytes+4095)/4096);
      if(counts[direction] && !queue->capacity){free(bindings);free(peer);errno=EMSGSIZE;return -1;}
    }
    int error=exchange(socket,out,peer,sends*sizeof *out,receives*sizeof *peer,m,client,link->provider.deadline);
    uint32_t values=0,rows=0;
    for(uint32_t i=0;i<receives && !error;i++){
      if(in[i].binding!=peer[i].binding || in[i].count!=peer[i].count || in[i].bytes!=peer[i].bytes){
        fprintf(stderr,"transfer mismatch queue=%u index=%u local=%u,%u,%llu peer=%u,%u,%llu\n",q,i,in[i].binding,in[i].count,(unsigned long long)in[i].bytes,peer[i].binding,peer[i].count,(unsigned long long)peer[i].bytes);
        errno=EPROTO;error=-1;break;
      }
      uint32_t chunks=(uint32_t)((in[i].bytes+payload-1)/payload);
      values+=in[i].count*chunks;
      uint32_t end=peer[i].local_row+(peer[i].count-1)*peer[i].stride+(chunks-1)*peer[i].chunk_stride+1;
      if(end>rows)rows=end;
    }
    if(error){free(bindings);free(peer);return error;}
    struct mesh_receive *receive=&link->receive[q];
    free(receive->next);free(receive->targets);
    *receive=(struct mesh_receive){.length=values,.first=receives?atomic_load_explicit(&mesh_page(m)[in[0].local_row],memory_order_acquire):0};
    receive->next=calloc((size_t)rows+1,sizeof *receive->next);
    receive->targets=calloc(values?values:1,sizeof *receive->targets);
    if(!receive->next || !receive->targets){free(bindings);free(peer);return -1;}
    for(uint32_t i=0;i<receives;i++)for(uint32_t value=0;value<in[i].count;value++){
      uint32_t source=peer[i].local_row+value*peer[i].stride,chunks=(uint32_t)((in[i].bytes+payload-1)/payload);
      for(uint32_t chunk=0;chunk<chunks;chunk++)receive->next[source+chunk*peer[i].chunk_stride+1]++;
    }
    for(uint32_t row=0;row<rows;row++)receive->next[row+1]+=receive->next[row];
    for(uint32_t i=0;i<receives;i++)for(uint32_t value=0;value<in[i].count;value++){
      uint32_t first=in[i].local_row+value*in[i].stride,source=peer[i].local_row+value*peer[i].stride;
      uint32_t chunks=(uint32_t)((in[i].bytes+payload-1)/payload);
      for(uint32_t chunk=0;chunk<chunks;chunk++){
        uint32_t row=first+chunk*m->block,key=source+chunk*peer[i].chunk_stride;
        receive->targets[receive->next[key]++]=(struct mesh_target){row,chunk+1==chunks?first:MESH_ABSENT};
      }
    }
    memmove(receive->next+1,receive->next,(size_t)rows*sizeof *receive->next);receive->next[0]=0;
    for(uint32_t i=0;i<sends;i++)for(uint32_t value=0;value<out[i].count;value++){
      uint32_t row=out[i].local_row+value*out[i].stride;
      uint32_t chunks=(uint32_t)((out[i].bytes+payload-1)/payload);
      link->send_edges[link->send_offsets[row]++]=(struct mesh_send_edge){.queue=q,.row=row,.pages=chunks*m->block};
      at++;
    }
    free(bindings);free(peer);
  }
  memmove(link->send_offsets+1,link->send_offsets,(size_t)mesh_rows(m)*sizeof *link->send_offsets);link->send_offsets[0]=0;
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    int error=link_receive(link,q);
    if(error){errno=error;return -1;}
  }
  uint32_t posted=1,peer_posted;
  return exchange(socket,&posted,&peer_posted,sizeof posted,sizeof peer_posted,m,client,link->provider.deadline);
}
/* design/algorithm-sources.md#programcopy */
static int link_receive(struct mesh_link *link,uint32_t q){
  struct mesh_queue *in=link_queue(link,q,MESH_RECEIVE);
  struct mesh_receive *receive=&link->receive[q];
  while(in->pending<in->capacity && in->next<receive->length){
    int error=link_post(link,q,MESH_RECEIVE,0,receive->first+in->next*link->M->block);
    if(error)return error;
    in->next++;
  }
  return 0;
}
/* design/algorithm-sources.md#programkernel_call */
static void link_send_ready(struct mesh_link *link,uint32_t q){
  struct mesh_queue *out=link_queue(link,q,MESH_SEND);
  struct mesh_ready *ready=&link->ready[q];
  while(ready->head<ready->tail && out->pending<out->capacity){
    struct mesh_send_edge *source=&link->send_edges[link->send_ready[ready->head]];
    uint32_t page=atomic_load_explicit(&mesh_page(link->M)[source->row+source->offset],memory_order_acquire);
    uint32_t end=source->offset+link->M->block;
    if(link_post(link,q,MESH_SEND,end==source->pages?source->row:MESH_ABSENT,page))return;
    source->offset=end;
    if(end==source->pages)ready->head++;
  }
}
/* design/algorithm-sources.md#programcopy */
static void mesh_progress(struct mesh_link *link,uint32_t direction){
  struct hdr *m=link->M;struct mesh_verbs *v=&link->provider;
  struct ibv_wc *completions=v->completions+direction*QD;
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    struct mesh_queue *queue=link_queue(link,q,direction);
    int count=ibv_poll_cq(v->completion_queues[2*q+direction],QD,completions);
    if(count<0){link_error(link,count,3);continue;}
    for(int i=0;i<count;i++){
      struct ibv_wc *wc=&completions[i];
      queue->pending--;
      if(wc->status)link_error(link,wc->status,2);
      if(direction==MESH_RECEIVE){
        link_receive(link,q);
        if(!wc->status){
          uint32_t page=(uint32_t)wc->wr_id,source=*mesh_tag(m,page);
          struct mesh_receive *receive=&link->receive[q];
          struct mesh_target target=receive->targets[receive->next[source]++];
          mesh_receive_assign(m,target.row,page);
          if(target.publish!=MESH_ABSENT && !receive->failed)mesh_publish(m,target.publish);
        } else link->receive[q].failed=1;
      } else {
        if((uint32_t)wc->wr_id!=MESH_ABSENT)mesh_buffer_release(m,(uint32_t)wc->wr_id,1);
        link_send_ready(link,q);
      }
    }
  }
}

/* design/algorithm-sources.md#programkernel_call */
static void link_publications(struct mesh_link *link){
  uint32_t queue=mesh_notice_queue(link->M,link->client,link->index);
  uint32_t row=mesh_notice_take(link->M,queue);
  while(row!=MESH_ABSENT){
    uint32_t next=mesh_notice_next(link->M,queue,row);
    for(uint32_t at=link->send_offsets[row];at<link->send_offsets[row+1];at++){
      uint32_t q=link->send_edges[at].queue;
      link->send_ready[link->ready[q].tail++]=at;
      link_send_ready(link,q);
      mesh_progress(link,MESH_SEND);
    }
    row=next;
  }
}

/* design/algorithm-sources.md#programkernel_call */
static void *link_progress(void *argument){
  struct mesh_worker *worker=argument;
  pthread_setname_np(worker->direction==MESH_SEND?"mesh.rdma.send":"mesh.rdma.receive");
  while(atomic_load_explicit(&worker->link->progressing,memory_order_acquire)){
    mesh_progress(worker->link,worker->direction);
    if(worker->direction==MESH_SEND)link_publications(worker->link);
  }
  return NULL;
}
/* design/algorithm-sources.md#programkernel_call */
static void link_stop(struct mesh_link *link){
  atomic_store_explicit(&link->progressing,0,memory_order_release);
  while(link->worker_count)pthread_join(link->workers[--link->worker_count].thread,NULL);
}

struct mesh_collector { struct hdr *memory; _Atomic int running; pthread_t thread; };
/* design/algorithm-sources.md#programtensor */
static void *link_collect(void *argument){
  struct mesh_collector *collector=argument;uint32_t pending=MESH_ABSENT;
  pthread_setname_np("mesh.reclaim");
  while(atomic_load_explicit(&collector->running,memory_order_acquire))pending=mesh_collect(collector->memory,pending);
  mesh_collect(collector->memory,pending);
  return NULL;
}

/* design/algorithm-sources.md#programcopy */
static void *link_run(void *argument){
  struct mesh_link *link=argument;struct hdr *m=link->M;
  struct mesh_port_info *port=&mesh_links(m)[link->index].port;
  uint32_t transfers=0;
  for(uint32_t q=0;q<m->qps;q++)for(int d=0;d<2;d++)transfers+=atomic_load(mesh_order_length(m,link->client,link->index*m->qps+q,d));
  if(!transfers)return NULL;
  while(atomic_load_explicit(&link->progressing,memory_order_acquire)){
    int setup=verbs_up(&link->provider,m,link->qps,link_configure,link,link->client);
    if(setup<0){
      if(errno!=ECANCELED)link_error(link,errno?errno:EIO,1);
      while(!down_pair(&link->provider))link_error(link,errno?errno:EIO,1);
      continue;
    }
    int error=0;
    for(uint32_t d=0;d<2;d++){
      link->workers[d]=(struct mesh_worker){.link=link,.direction=d};
      error=pthread_create(&link->workers[d].thread,NULL,link_progress,&link->workers[d]);
      if(error)break;
      link->worker_count++;
    }
    if(error){link_error(link,error,1);link_stop(link);}
    else {
      atomic_store_explicit(&port->phase,MESH_PAIRED,memory_order_release);
      while(link->worker_count)pthread_join(link->workers[--link->worker_count].thread,NULL);
    }
    while(!down_pair(&link->provider))link_error(link,errno?errno:EIO,1);
    atomic_store_explicit(&port->phase,MESH_PAIRING,memory_order_release);
  }
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
  for(uint32_t queue=0;queue<MESH_NOTICE_BANKS*(link_count+MESH_COMPUTE_THREADS);queue++)atomic_store_explicit(&mesh_notice_heads(m)[queue],MESH_ABSENT,memory_order_relaxed);
  for(uint32_t r=0;r<mesh_rows(m);r++)atomic_store_explicit(&mesh_page(m)[r],MESH_ABSENT,memory_order_relaxed);
  struct mesh_wire wire={0};
  if(wire_map(&wire,m,fd))die("transport page aliases");
  close(fd);
  for(uint32_t i=0;i<link_count;i++){
    struct mesh_link *link=&links[i];
    link->M=m;link->qps=(int)qps;link->provider.wire=&wire;
    link->send_offsets=calloc((size_t)mesh_rows(m)+1,sizeof *link->send_offsets);
    link->provider.completions=calloc(2*QD,sizeof *link->provider.completions);
    if(!link->send_offsets || !link->provider.completions)die("link allocation");
    mesh_links(m)[i].peer=link->provider.peer;
    atomic_store(&mesh_links(m)[i].port.phase,MESH_PAIRING);
  }
  struct mesh_collector collector={.memory=m,.running=1};
  int status=pthread_create(&collector.thread,NULL,link_collect,&collector);
  if(status)die("collector thread");
  atomic_store(&m->bridge_pid,(uint64_t)getpid());atomic_store(&m->port.phase,MESH_PAIRING);
  __sync_synchronize();m->magic=MESH_MAGIC;
  fprintf(stderr,"bridge node %d: %u links, %u queue pairs per link\n",me,link_count,qps);
  while(!stop){
    uint64_t client=atomic_load_explicit(&m->client,memory_order_acquire);
    if(!client || atomic_load_explicit(&m->configured,memory_order_acquire)!=client)continue;
    atomic_store_explicit(&m->device_client,client,memory_order_seq_cst);
    if(atomic_load_explicit(&m->client,memory_order_seq_cst)!=client || atomic_load_explicit(&m->configured,memory_order_seq_cst)!=client){atomic_store_explicit(&m->device_client,0,memory_order_seq_cst);continue;}
    uint32_t started=0;
    for(uint32_t i=0;i<link_count;i++){
      links[i].client=client;atomic_store_explicit(&links[i].progressing,1,memory_order_release);
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
  atomic_store_explicit(&collector.running,0,memory_order_release);pthread_join(collector.thread,NULL);
  for(uint32_t i=0;i<link_count;i++){
    struct mesh_link *link=&links[i];
    atomic_store(&mesh_links(m)[i].port.phase,MESH_STOPPED);
    if(link->provider.listener>=0)close(link->provider.listener);
    free(link->provider.completions);free(link->send_offsets);free(link->send_edges);free(link->send_ready);
    for(uint32_t q=0;q<qps;q++){free(link->receive[q].next);free(link->receive[q].targets);}
    free(link->configuration);
  }
  for(uint32_t i=0;i<device_count;i++)pthread_mutex_destroy(&devices[i].setup);
  free(devices);free(links);munmap(wire.data,wire.length);free(wire.spans);
  atomic_store(&m->port.phase,MESH_STOPPED);munmap(m,length);return status;
}
