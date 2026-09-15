#include "mesh-verbs.h"
#include "mesh-dataflow.h"
#include <pthread.h>

/* design/algorithm-sources.md#programcopy */
struct mesh_send_edge {uint32_t queue,row,pages,offset,next,ready_next;};
struct mesh_target {uint32_t row,next,publish;};
struct mesh_ready {uint32_t head,tail;};
struct mesh_queue {uint32_t pending,next,capacity,bytes;};
struct mesh_receive {uint32_t *pages,*heads,length,failed;struct mesh_target *targets;};
struct mesh_worker {struct mesh_link *link;pthread_t thread;uint32_t direction;};
struct mesh_link {
  struct mesh_worker workers[2];
  uint32_t worker_count;
  _Atomic int progressing;
  struct hdr *M;struct mesh_verbs provider;int qps;uint64_t client;
  struct mesh_queue queues[2*MESH_QPS];
  struct mesh_receive receive[MESH_QPS];
  pthread_t collector;
  _Atomic int collecting;
  uint32_t *send_heads;
  struct mesh_send_edge *send_edges;
  struct mesh_ready ready[MESH_QPS];
};
static int link_receive(struct mesh_link *link,uint32_t q);
/* design/algorithm-sources.md#programcopy */
static struct mesh_queue *link_queue(struct mesh_link *link,uint32_t q,int direction){return &link->queues[2*q+(uint32_t)direction];}
/* design/algorithm-sources.md#programcopy */
static void link_error(struct hdr *m,int64_t code,uint32_t domain){m->port.code=code;m->port.domain=domain;}
/* design/algorithm-sources.md#programcopy */
static int link_post(struct mesh_link *link,uint32_t q,int direction,uint32_t row,uint32_t page){
  struct hdr *m=link->M;struct mesh_verbs *v=&link->provider;
  struct mesh_queue *queue=link_queue(link,q,direction);
  struct ibv_sge span=v->spans[page/m->block];
  span.length=queue->bytes;
  int error;
  if(direction==MESH_SEND){
    struct ibv_send_wr request={.wr_id=row,.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad=NULL;
    error=ibv_post_send(v->pairs[q],&request,&bad);
  } else {
    struct ibv_recv_wr request={.wr_id=page,.sg_list=&span,.num_sge=1},*bad=NULL;
    error=ibv_post_recv(v->pairs[q],&request,&bad);
  }
  if(error){link_error(m,error,1);return error;}
  queue->pending++;
  return 0;
}
/* design/algorithm-sources.md#programkernel_call */
static void link_ready(struct mesh_link *link,uint32_t q,uint32_t index){
  struct mesh_ready *ready=&link->ready[q];
  link->send_edges[index].ready_next=MESH_ABSENT;
  if(ready->tail==MESH_ABSENT)ready->head=index;
  else link->send_edges[ready->tail].ready_next=index;
  ready->tail=index;
}
/* design/algorithm-sources.md#programcopy */
static int link_configure(void *state,int socket,uint64_t client){
  struct mesh_link *link=state;struct hdr *m=link->M;
  uint64_t payload=(uint64_t)m->block*m->pgsz;
  size_t count=0,at=0;
  for(uint32_t q=0;q<(uint32_t)link->qps;q++)for(uint32_t i=0;i<atomic_load(mesh_order_length(m,q,MESH_SEND));i++){
    struct mesh_transfer transfer=mesh_transfers(m,q,MESH_SEND)[i];
    count+=transfer.count;
  }
  free(link->send_edges);link->send_edges=calloc(count?count:1,sizeof *link->send_edges);
  if(!link->send_edges)return -1;
  for(uint32_t row=0;row<mesh_rows(m);row++)link->send_heads[row]=MESH_ABSENT;
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    uint32_t counts[2]={atomic_load(mesh_order_length(m,q,MESH_SEND)),atomic_load(mesh_order_length(m,q,MESH_RECEIVE))},peer_counts[2];
    if(exchange(socket,counts,peer_counts,sizeof counts,sizeof peer_counts,m,client))return -1;
    uint32_t sends=counts[MESH_SEND],receives=counts[MESH_RECEIVE];
    link->ready[q]=(struct mesh_ready){MESH_ABSENT,MESH_ABSENT};
    if(sends!=peer_counts[MESH_RECEIVE] || receives!=peer_counts[MESH_SEND]){
      fprintf(stderr,"transfer count mismatch queue=%u local=%u,%u peer=%u,%u\n",q,sends,receives,peer_counts[MESH_SEND],peer_counts[MESH_RECEIVE]);
      errno=EPROTO;return -1;
    }
    struct mesh_transfer *bindings=calloc(sends+receives+1,sizeof *bindings),*peer=calloc(receives?receives:1,sizeof *peer);
    if(!bindings || !peer){free(bindings);free(peer);return -1;}
    struct mesh_transfer *out=bindings,*in=bindings+sends;
    memcpy(out,mesh_transfers(m,q,MESH_SEND),sends*sizeof *out);
    memcpy(in,mesh_transfers(m,q,MESH_RECEIVE),receives*sizeof *in);
    for(int direction=0;direction<2;direction++){
      struct mesh_queue *queue=link_queue(link,q,direction);
      struct mesh_transfer *transfers=direction==MESH_SEND?out:in;
      uint64_t bytes=0;
      for(uint32_t i=0;i<counts[direction];i++)if(transfers[i].bytes>bytes)bytes=transfers[i].bytes;
      queue->bytes=(uint32_t)(bytes<payload?bytes:payload)+sizeof(uint32_t);
      queue->capacity=link->provider.capacity[q][direction]/((queue->bytes+4095)/4096);
      if(counts[direction] && !queue->capacity){free(bindings);free(peer);errno=EMSGSIZE;return -1;}
    }
    int error=exchange(socket,out,peer,sends*sizeof *out,receives*sizeof *peer,m,client);
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
    free(receive->pages);free(receive->heads);free(receive->targets);
    *receive=(struct mesh_receive){.length=values};
    receive->pages=calloc(values?values:1,sizeof *receive->pages);
    receive->heads=malloc((rows?rows:1)*sizeof *receive->heads);
    receive->targets=calloc(values?values:1,sizeof *receive->targets);
    if(!receive->pages || !receive->heads || !receive->targets){free(bindings);free(peer);return -1;}
    for(uint32_t row=0;row<rows;row++)receive->heads[row]=MESH_ABSENT;
    uint32_t slot=0;
    for(uint32_t i=0;i<receives;i++)for(uint32_t value=0;value<in[i].count;value++){
      uint32_t first=in[i].local_row+value*in[i].stride,source=peer[i].local_row+value*peer[i].stride;
      uint32_t chunks=(uint32_t)((in[i].bytes+payload-1)/payload);
      for(uint32_t chunk=0;chunk<chunks;chunk++){
        uint32_t row=first+chunk*m->block,key=source+chunk*peer[i].chunk_stride;
        receive->pages[slot]=atomic_load_explicit(&mesh_page(m)[row],memory_order_acquire);
        receive->targets[slot]=(struct mesh_target){row,receive->heads[key],chunk+1==chunks?first:MESH_ABSENT};
        receive->heads[key]=slot++;
      }
    }
    for(uint32_t i=0;i<sends;i++)for(uint32_t value=0;value<out[i].count;value++){
      uint32_t row=out[i].local_row+value*out[i].stride;
      uint32_t chunks=(uint32_t)((out[i].bytes+payload-1)/payload);
      link->send_edges[at]=(struct mesh_send_edge){.queue=q,.row=row,.pages=chunks*m->block,.next=link->send_heads[row]};
      link->send_heads[row]=(uint32_t)at++;
    }
    free(bindings);free(peer);
  }
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    int error=link_receive(link,q);
    if(error){errno=error;return -1;}
  }
  uint32_t posted=1,peer_posted;
  return exchange(socket,&posted,&peer_posted,sizeof posted,sizeof peer_posted,m,client);
}
/* design/algorithm-sources.md#programcopy */
static int link_receive(struct mesh_link *link,uint32_t q){
  struct mesh_queue *in=link_queue(link,q,MESH_RECEIVE);
  struct mesh_receive *receive=&link->receive[q];
  while(in->pending<in->capacity && in->next<receive->length){
    int error=link_post(link,q,MESH_RECEIVE,0,receive->pages[in->next]);
    if(error)return error;
    in->next++;
  }
  return 0;
}
/* design/algorithm-sources.md#programkernel_call */
static void link_send_ready(struct mesh_link *link,uint32_t q){
  struct mesh_queue *out=link_queue(link,q,MESH_SEND);
  struct mesh_ready *ready=&link->ready[q];
  while(ready->head!=MESH_ABSENT && out->pending<out->capacity){
    struct mesh_send_edge *source=&link->send_edges[ready->head];
    uint32_t page=atomic_load_explicit(&mesh_page(link->M)[source->row+source->offset],memory_order_acquire);
    uint32_t end=source->offset+link->M->block;
    if(link_post(link,q,MESH_SEND,end==source->pages?source->row:MESH_ABSENT,page))return;
    source->offset=end;
    if(end==source->pages)ready->head=source->ready_next;
  }
  if(ready->head==MESH_ABSENT)ready->tail=MESH_ABSENT;
}
/* design/algorithm-sources.md#programcopy */
static void mesh_progress(struct mesh_link *link,uint32_t direction){
  struct hdr *m=link->M;struct mesh_verbs *v=&link->provider;
  struct ibv_wc *completions=v->completions+direction*QD;
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    struct mesh_queue *queue=link_queue(link,q,direction);
    int count=ibv_poll_cq(v->completion_queues[2*q+direction],QD,completions);
    if(count<0){link_error(m,count,3);continue;}
    for(int i=0;i<count;i++){
      struct ibv_wc *wc=&completions[i];
      queue->pending--;
      if(wc->status)link_error(m,wc->status,2);
      if(direction==MESH_RECEIVE){
        link_receive(link,q);
        if(!wc->status){
          uint32_t page=(uint32_t)wc->wr_id,source=*mesh_tag(m,page);
          struct mesh_receive *receive=&link->receive[q];
          struct mesh_target target=receive->targets[receive->heads[source]];
          receive->heads[source]=target.next;
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
  uint32_t queue=mesh_notice_queue(link->client,MESH_NOTICE_SEND);
  uint32_t row=mesh_notice_take(link->M,queue);
  while(row!=MESH_ABSENT){
    uint32_t next=mesh_notice_next(link->M,queue,row);
    for(uint32_t at=link->send_heads[row];at!=MESH_ABSENT;at=link->send_edges[at].next){
      struct mesh_send_edge edge=link->send_edges[at];
      link_ready(link,edge.queue,at);
      link_send_ready(link,edge.queue);
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

/* design/algorithm-sources.md#programtensor */
static void *link_collect(void *argument){
  struct mesh_link *link=argument;uint32_t pending=MESH_ABSENT;
  pthread_setname_np("mesh.reclaim");
  while(atomic_load_explicit(&link->collecting,memory_order_acquire))pending=mesh_collect(link->M,pending);
  mesh_collect(link->M,pending);
  return NULL;
}

/* design/collective-dependency-ledger.md#d14-teardown-retains-outstanding-device-storage */
static int link_down(struct mesh_link *link){
  struct hdr *M=link->M;
  link_stop(link);
  if(!down_pair()){ link_error(M,errno?errno:EIO,1); return -1; }
  atomic_store_explicit(&M->device_client,0,memory_order_seq_cst);
  memset(link->queues,0,sizeof link->queues);
  for(uint32_t q=0;q<MESH_QPS;q++)link->ready[q]=(struct mesh_ready){MESH_ABSENT,MESH_ABSENT};
  link->client=0;
  atomic_store(&M->port.phase,MESH_PAIRING);
  return 0;
}

int main(int argc,char**argv){
  const char *peer=NULL,*name=MESH_NAME; int me=0,layout=0; double pct=0;
  uint64_t arena_pages=0,block_pages=0;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-I") && i+1<argc) me=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-M") && i+1<argc) pct=atof(argv[++i]);
    else if((!strcmp(argv[i],"-A") || !strcmp(argv[i],"-B")) && i+1<argc){
      char kind=argv[i][1],*end;
      uint64_t pages=strtoull(argv[++i],&end,10);
      if(*end || !pages || pages>INT32_MAX) die("configured page count");
      if(kind=='A') arena_pages=pages; else block_pages=pages;
    }
    else if(!strcmp(argv[i],"--layout")) layout=1;
    else if(!strcmp(argv[i],"-s") && i+1<argc) name=argv[++i];
    else if(argv[i][0]=='-') die("unknown bridge option");
    else peer=argv[i];
  }
  if(me<0 || me>=65535 || !isfinite(pct) || pct<0 || pct>100 || !arena_pages || !block_pages || arena_pages<block_pages) die("bridge geometry");
  const uint32_t pg=(uint32_t)getpagesize();
  /* design/collective-dependency-ledger.md#d6-paired-send-and-receive-frame-counts-match */
  if((uint64_t)block_pages*pg+4096>16773120) die("transport chunk exceeds native request capacity");
  struct hdr geometry={0};
  uint64_t length=mesh_layout(&geometry,pg,(uint32_t)block_pages,(uint32_t)arena_pages);
  uint64_t ram=0; size_t rl=sizeof ram; sysctlbyname("hw.memsize",&ram,&rl,NULL,0);
  if(pct && length>(uint64_t)(pct/100*(double)ram)) die("configured graph exceeds page capacity");
  if(layout){ printf("%llu\n",(unsigned long long)length); return 0; }
  atexit(down); struct sigaction sa={0}; sa.sa_handler=onsig;
  sigaction(SIGINT,&sa,NULL); sigaction(SIGTERM,&sa,NULL); sigaction(SIGHUP,&sa,NULL); signal(SIGPIPE,SIG_IGN);
  shm_unlink(name); int fd=shm_open(name,O_CREAT|O_RDWR,MESH_MODE); if(fd<0) die("shm");
  if(ftruncate(fd,(off_t)length)) die("ftruncate"); fchmod(fd,MESH_MODE);
  struct hdr *M=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  if(M==MAP_FAILED) die("mmap"); shm=name;
  *M=geometry; M->node=(uint32_t)me; M->version=MESH_VERSION;
  for(uint32_t r=0;r<mesh_rows(M);r++) atomic_store_explicit(&mesh_page(M)[r],MESH_ABSENT,memory_order_relaxed);
  struct mesh_link link={.M=M}; provider=&link.provider;
  if(wire_map(M,fd))die("transport page aliases");
  close(fd);
  link.qps=getenv("MESH_QPS")?atoi(getenv("MESH_QPS")):1; if(link.qps<1) link.qps=1; if(link.qps>MESH_QPS) link.qps=MESH_QPS;
  M->qps=(uint32_t)link.qps;
  atomic_store(&M->bridge_pid,(uint64_t)getpid());
  __sync_synchronize(); M->magic=MESH_MAGIC;
  link.send_heads=calloc(mesh_rows(M),sizeof *link.send_heads);
  provider->completions=calloc(2*QD,sizeof *provider->completions);
  int status=0;
  if(!link.send_heads || !provider->completions){ status=ENOMEM; goto teardown; }
  atomic_store_explicit(&link.collecting,1,memory_order_release);
  status=pthread_create(&link.collector,NULL,link_collect,&link);
  if(status){atomic_store(&link.collecting,0);goto teardown;}
  atomic_store(&M->port.phase,MESH_PAIRING);
  fprintf(stderr,"bridge node %d: %d queue pair(s), %u frames per block\n",me,link.qps,(unsigned)(block_pages*pg/4096));

  while(!stop){
    /* design/collective-dependency-ledger.md#d14-teardown-retains-outstanding-device-storage */
    if(lsock<0 && listener_up()){ link_error(M,errno?errno:EIO,1); continue; }
    uint64_t client=atomic_load_explicit(&M->client,memory_order_acquire);
    /* design/collective-dependency-ledger.md#d14-teardown-retains-outstanding-device-storage */
    if(link.client && client!=link.client && link_down(&link)) continue;
    if(!link.client && client && atomic_load_explicit(&M->configured,memory_order_acquire)==client){
      atomic_store_explicit(&M->device_client,client,memory_order_seq_cst);
      if(atomic_load_explicit(&M->client,memory_order_seq_cst)!=client || atomic_load_explicit(&M->configured,memory_order_seq_cst)!=client){
        atomic_store_explicit(&M->device_client,0,memory_order_seq_cst);continue;
      }
      /* design/collective-dependency-ledger.md#d13-connection-metadata-is-setup-work */
      int setup=verbs_up(peer,M,me,link.qps,link_configure,&link,client);
      if(setup>0){atomic_store_explicit(&M->device_client,0,memory_order_seq_cst);continue;}
      if(setup<0){
        if(errno!=ECANCELED)link_error(M,errno?errno:EIO,1);
        link_down(&link);
        continue;
      }
      if(stop || atomic_load_explicit(&M->client,memory_order_acquire)!=client){link_down(&link);continue;}
      link.client=client;
      atomic_store_explicit(&link.progressing,1,memory_order_release);
      int error=0;
      for(uint32_t direction=0;direction<2;direction++){
        link.workers[direction]=(struct mesh_worker){.link=&link,.direction=direction};
        error=pthread_create(&link.workers[direction].thread,NULL,link_progress,&link.workers[direction]);
        if(error)break;
        link.worker_count++;
      }
      if(error){link_error(M,error,1);link_down(&link);continue;}
      atomic_store(&M->port.phase,MESH_PAIRED);
      fprintf(stderr,"bridge node %d paired for client %llu\n",me,(unsigned long long)client);
    }
  }
teardown:
  link_stop(&link);
  fprintf(stderr,"bridge node %d stopping: code=%lld domain=%u errno=%d\n",me,(long long)M->port.code,M->port.domain,errno);
  atomic_store(&M->port.phase,MESH_STOPPED);
  if(!down_verbs()){ fprintf(stderr,"verbs teardown failed: %s\n",strerror(errno)); return 1; }
  munmap(provider->wire,provider->wire_length);free(provider->spans);
  atomic_store_explicit(&M->device_client,0,memory_order_seq_cst);
  if(atomic_exchange_explicit(&link.collecting,0,memory_order_acq_rel))pthread_join(link.collector,NULL);
  if(lsock>=0) close(lsock);
  free(provider->completions);free(link.send_heads);free(link.send_edges);
  for(uint32_t q=0;q<MESH_QPS;q++){free(link.receive[q].pages);free(link.receive[q].heads);free(link.receive[q].targets);}
  munmap(M,length); return status;
}
