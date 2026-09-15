#include "mesh-verbs.h"
#include "mesh-dataflow.h"
#include "mesh-memory.h"
#include <pthread.h>

/* design/algorithm-sources.md#programcopy */
struct mesh_posted { uint32_t row,page,index,bytes,frames; };
struct mesh_send_edge {uint32_t queue,index,next;};
struct mesh_ready {uint32_t head,tail;};
struct mesh_queue { struct mesh_posted posted[QD]; uint32_t head,tail,next,frames; };
struct mesh_arrivals { uint32_t *pages,*indices,length,produced,described,joined; };
struct mesh_index_frame { uint32_t queue,count; uint32_t entries[(MESH_INDEX_BYTES-8)/sizeof(uint32_t)]; };
struct mesh_worker { struct mesh_link *link; pthread_t thread; uint32_t direction; };
struct mesh_link {
  struct mesh_worker workers[2];
  uint32_t worker_count;
  _Atomic int progressing;
  struct hdr *M; struct mesh_verbs provider; int qps,frames,budget; uint64_t client;
  struct mesh_queue *queues;
  struct mesh_arrivals arrivals[MESH_QPS];
  struct mesh_transfer *transfers[2*MESH_QPS];
  uint32_t lengths[2*MESH_QPS];
  pthread_t collector;
  _Atomic int collecting;
  uint8_t *queued;
  uint32_t *ready_next,*send_heads;
  struct mesh_send_edge *send_edges;
  struct mesh_ready ready[MESH_QPS];
};
/* design/algorithm-sources.md#programcopy */
static struct mesh_queue *link_queue(struct mesh_link *link,uint32_t q,int direction){ return &link->queues[2*q+(uint32_t)direction]; }
/* design/algorithm-sources.md#programcopy */
static struct mesh_transfer *link_transfers(struct mesh_link *link,uint32_t q,int direction){return link->transfers[2*q+(uint32_t)direction];}
/* design/algorithm-sources.md#programcopy */
static void link_error(struct hdr *M,int64_t code,uint32_t domain){
  M->port.code=code; M->port.domain=domain;
}
/* design/algorithm-sources.md#programcopy */
static size_t link_index_offset(struct mesh_link *link,int direction,uint32_t slot){
  return link->M->index_off+((size_t)direction*link->budget+slot)*MESH_INDEX_BYTES;
}
/* design/algorithm-sources.md#programcopy */
static struct mesh_index_frame *link_indices(struct mesh_link *link,int direction,uint32_t slot){
  return (struct mesh_index_frame *)((char *)link->M+link_index_offset(link,direction,slot));
}
/* design/algorithm-sources.md#programcopy */
static int link_post(struct mesh_link *link,uint32_t q,int direction,struct mesh_posted entry){
  struct hdr *M=link->M; struct mesh_verbs *v=&link->provider;
  int indices=q==(uint32_t)link->qps;
  size_t offset=indices?link_index_offset(link,direction,entry.row):M->data_off+(size_t)entry.page*M->pgsz;
  struct ibv_sge span=region_sge((char*)M,offset,indices?MESH_INDEX_BYTES:entry.bytes);
  entry.frames=span.length/4096;
  int error;
  if(direction==MESH_SEND){
    struct ibv_send_wr request={.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad=NULL;
    error=ibv_post_send(v->pairs[q],&request,&bad);
  } else {
    struct ibv_recv_wr request={.sg_list=&span,.num_sge=1},*bad=NULL;
    error=ibv_post_recv(v->pairs[q],&request,&bad);
  }
  if(error){ link_error(M,error,1); return error; }
  struct mesh_queue *queue=link_queue(link,q,direction);
  queue->posted[queue->tail++%QD]=entry;queue->frames+=entry.frames;
  return 0;
}
/* design/algorithm-sources.md#programkernel_call */
static int link_ready(struct mesh_link *link,uint32_t q,uint32_t index){
  size_t key=(size_t)q*mesh_blocks(link->M)+index;
  struct mesh_transfer *transfer=&link_transfers(link,q,MESH_SEND)[index];
  if(link->queued[key] || !mesh_bits_all(link->M,MESH_PRESENT,transfer->local_row,link->M->block))return 0;
  struct mesh_ready *ready=&link->ready[q];
  link->queued[key]=1;link->ready_next[key]=MESH_ABSENT;
  if(ready->tail==MESH_ABSENT)ready->head=index;
  else link->ready_next[(size_t)q*mesh_blocks(link->M)+ready->tail]=index;
  ready->tail=index;return 1;
}
/* design/algorithm-sources.md#programcopy */
static int link_configure(void *state,int socket,uint64_t client){
  struct mesh_link *link=state;struct hdr *m=link->M;
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    uint32_t counts[2]={atomic_load(mesh_order_length(m,q,MESH_SEND)),atomic_load(mesh_order_length(m,q,MESH_RECEIVE))},peer_counts[2];
    if(exchange(socket,counts,peer_counts,sizeof counts,sizeof peer_counts,m,client))return -1;
    uint32_t sends=counts[MESH_SEND],receives=counts[MESH_RECEIVE];
    if(sends!=peer_counts[MESH_RECEIVE] || receives!=peer_counts[MESH_SEND]){
      fprintf(stderr,"transfer count mismatch queue=%u local=%u,%u peer=%u,%u\n",q,sends,receives,peer_counts[MESH_SEND],peer_counts[MESH_RECEIVE]);
      errno=EPROTO;return -1;
    }
    struct mesh_transfer *peer_bindings=calloc(sends?sends:1,sizeof *peer_bindings);
    if(!peer_bindings)return -1;
    for(uint32_t direction=0;direction<2;direction++){
      size_t index=2*q+direction;free(link->transfers[index]);
      link->lengths[index]=counts[direction];
      link->transfers[index]=calloc(counts[direction]?counts[direction]:1,sizeof(struct mesh_transfer));
      if(!link->transfers[index]){free(peer_bindings);return -1;}
      memcpy(link->transfers[index],mesh_transfers(m,q,direction),counts[direction]*sizeof(struct mesh_transfer));
    }
    struct mesh_transfer *out=link_transfers(link,q,MESH_SEND),*in=link_transfers(link,q,MESH_RECEIVE);
    int error=exchange(socket,in,peer_bindings,receives*sizeof *in,sends*sizeof *peer_bindings,m,client);
    for(uint32_t i=0;i<sends && !error;i++){
      struct mesh_transfer target=peer_bindings[i];
      if(out[i].binding!=target.binding || out[i].offset!=target.offset || out[i].bytes!=target.bytes){
        fprintf(stderr,"transfer mismatch queue=%u index=%u local=%u,%u,%u peer=%u,%u,%u\n",q,i,out[i].binding,out[i].offset,out[i].bytes,target.binding,target.offset,target.bytes);
        errno=EPROTO;error=-1;break;
      }
    }
    free(peer_bindings);if(error)return error;
    struct mesh_arrivals *arrivals=&link->arrivals[q];
    free(arrivals->pages);free(arrivals->indices);
    *arrivals=(struct mesh_arrivals){.length=receives};
    arrivals->pages=calloc(receives?receives:1,sizeof *arrivals->pages);
    arrivals->indices=calloc(receives?receives:1,sizeof *arrivals->indices);
    if(!arrivals->pages || !arrivals->indices)return -1;
    for(uint32_t i=0;i<receives;i++)arrivals->pages[i]=atomic_load_explicit(&mesh_page(m)[in[i].local_row],memory_order_acquire);
    for(int d=0;d<2;d++)for(uint32_t i=0;i<link->lengths[2*q+d];i++){
      uint32_t frames=(uint32_t)link->frames;
      if(frames>link->provider.capacity[q][d]){
        fprintf(stderr,"transfer queue=%u direction=%d index=%u requires=%u frames available=%u\n",q,d,i,frames,link->provider.capacity[q][d]);
        errno=EMSGSIZE;return -1;
      }
    }
  }
  size_t count=0;
  for(uint32_t q=0;q<(uint32_t)link->qps;q++)count+=(size_t)link->lengths[2*q+MESH_SEND]*m->block;
  free(link->send_edges);link->send_edges=calloc(count?count:1,sizeof *link->send_edges);
  if(!link->send_edges)return -1;
  for(uint32_t row=0;row<mesh_rows(m);row++)link->send_heads[row]=MESH_ABSENT;
  size_t at=0;
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    link->ready[q]=(struct mesh_ready){MESH_ABSENT,MESH_ABSENT};
    uint32_t length=link->lengths[2*q+MESH_SEND];
    struct mesh_transfer *transfers=link_transfers(link,q,MESH_SEND);
    for(uint32_t i=0;i<length;i++){
      for(uint32_t row=transfers[i].local_row;row<transfers[i].local_row+m->block;row++){
        link->send_edges[at]=(struct mesh_send_edge){q,i,link->send_heads[row]};link->send_heads[row]=(uint32_t)at++;
      }
      link_ready(link,q,i);
    }
  }
  return 0;
}
/* design/algorithm-sources.md#programcopy */
static void link_receive(void *state,uint32_t q){
  struct mesh_link *link=state;
  struct hdr *M=link->M;struct mesh_queue *in=link_queue(link,q,MESH_RECEIVE);
  while(in->frames<link->provider.capacity[q][MESH_RECEIVE]){
    if(q==(uint32_t)link->qps){
      if(in->tail-in->head>=(uint32_t)link->budget)return;
      struct mesh_posted entry={.row=in->next%(uint32_t)link->budget};
      if(link_post(link,q,MESH_RECEIVE,entry))return;
      in->next++;
    } else {
      struct mesh_arrivals *arrivals=&link->arrivals[q];
      if(in->next==arrivals->length || (uint32_t)link->frames>link->provider.capacity[q][MESH_RECEIVE]-in->frames)return;
      uint32_t slot=in->next++;
      struct mesh_posted entry={.row=slot,.page=arrivals->pages[slot],.index=slot,.bytes=M->block*M->pgsz};
      if(link_post(link,q,MESH_RECEIVE,entry))return;
    }
  }
}
/* design/algorithm-sources.md#programcopy */
static void link_deliver(struct mesh_link *link,uint32_t q){
  struct mesh_arrivals *arrivals=&link->arrivals[q];
  while(arrivals->joined<arrivals->produced && arrivals->joined<arrivals->described){
    uint32_t slot=arrivals->joined++,page=arrivals->pages[slot];
    uint32_t row=link_transfers(link,q,MESH_RECEIVE)[arrivals->indices[slot]].local_row;
    if(page!=MESH_ABSENT)mesh_receive_complete(link->M,row,page);
  }
}
/* design/algorithm-sources.md#programkernel_call */
static void link_send_ready(struct mesh_link *link,uint32_t q){
  struct hdr *M=link->M;struct mesh_verbs *v=&link->provider;
  uint32_t iq=(uint32_t)link->qps;
  struct mesh_queue *indices=link_queue(link,iq,MESH_SEND);
  struct mesh_queue *out=link_queue(link,q,MESH_SEND);
  struct mesh_transfer *transfers=link_transfers(link,q,MESH_SEND);
  struct mesh_ready *ready=&link->ready[q];
  while(ready->head!=MESH_ABSENT && (uint32_t)link->frames<=v->capacity[q][MESH_SEND]-out->frames &&
        indices->frames<v->capacity[iq][MESH_SEND] && indices->tail-indices->head<(uint32_t)link->budget){
    uint32_t slot=indices->tail%(uint32_t)link->budget;
    struct mesh_index_frame *frame=link_indices(link,MESH_SEND,slot);
    frame->queue=q;frame->count=0;
    uint32_t available=v->capacity[q][MESH_SEND]-out->frames;
    while(ready->head!=MESH_ABSENT && frame->count<sizeof frame->entries/sizeof *frame->entries && available>=(uint32_t)link->frames){
      uint32_t index=ready->head;size_t key=(size_t)q*mesh_blocks(M)+index;
      ready->head=link->ready_next[key];
      if(ready->head==MESH_ABSENT)ready->tail=MESH_ABSENT;
      frame->entries[frame->count++]=index;
      available-=(uint32_t)link->frames;
    }
    if(link_post(link,iq,MESH_SEND,(struct mesh_posted){.row=slot})){
      for(uint32_t i=0;i<frame->count;i++){
        uint32_t index=frame->entries[i];link->queued[(size_t)q*mesh_blocks(M)+index]=0;
        link_ready(link,q,index);
      }
      break;
    }
    for(uint32_t i=0;i<frame->count;i++){
      uint32_t index=frame->entries[i];
      struct mesh_transfer t=transfers[index];
      struct mesh_posted entry={.row=t.local_row,.page=atomic_load_explicit(&mesh_page(M)[t.local_row],memory_order_acquire),.index=index,.bytes=M->block*M->pgsz};
      if(link_post(link,q,MESH_SEND,entry))return;
    }
  }
}
/* design/algorithm-sources.md#programkernel_call */
static void link_publications(struct mesh_link *link){
  uint32_t row=mesh_notice_take(link->M,MESH_NOTICE_SEND);
  while(row!=MESH_ABSENT){
    uint32_t next=mesh_notice_next(link->M,MESH_NOTICE_SEND,row);
    uint32_t queues=0;
    for(uint32_t at=link->send_heads[row];at!=MESH_ABSENT;at=link->send_edges[at].next){
      struct mesh_send_edge edge=link->send_edges[at];
      if(link_ready(link,edge.queue,edge.index))queues|=UINT32_C(1)<<edge.queue;
    }
    while(queues){uint32_t q=(uint32_t)__builtin_ctz(queues);queues&=queues-1;link_send_ready(link,q);}
    row=next;
  }
}
/* design/algorithm-sources.md#programcopy */
static void mesh_progress(struct mesh_link *link,uint32_t direction){
  struct hdr *M=link->M;struct mesh_verbs *v=&link->provider;
  uint32_t iq=(uint32_t)link->qps;
  struct ibv_wc *completions=v->completions+direction*QD;
  for(uint32_t cq=direction;cq<2*(iq+1);cq+=2){
  int count=ibv_poll_cq(v->completion_queues[cq],QD,completions);
  if(count<0){link_error(M,count,3);continue;}
  for(int i=0;i<count;i++){
    struct ibv_wc *wc=&completions[i];
    uint32_t q=cq/2;
    if(wc->status)link_error(M,wc->status,2);
    struct mesh_queue *queue=link_queue(link,q,direction);
    struct mesh_posted entry=queue->posted[queue->head++%QD];queue->frames-=entry.frames;
    if(q==iq){
      if(direction==MESH_RECEIVE && !wc->status){
        struct mesh_index_frame *frame=link_indices(link,MESH_RECEIVE,entry.row);
        uint32_t data=frame->queue;
        struct mesh_arrivals *arrivals=&link->arrivals[data];
        memcpy(arrivals->indices+arrivals->described,frame->entries,(size_t)frame->count*sizeof *frame->entries);
        arrivals->described+=frame->count;
        link_receive(link,iq);
        link_deliver(link,data);
      }
      if(direction==MESH_RECEIVE && wc->status)link_receive(link,iq);
      if(direction==MESH_SEND)for(uint32_t data=0;data<iq;data++)link_send_ready(link,data);
    } else if(direction==MESH_RECEIVE){
      struct mesh_arrivals *arrivals=&link->arrivals[q];
      if(wc->status)arrivals->pages[entry.index]=MESH_ABSENT;
      arrivals->produced++;
      link_receive(link,q);
      link_deliver(link,q);
    } else {
      mesh_buffer_release(M,entry.row,M->block);
      link_send_ready(link,q);
    }
  }
  }
  if(direction==MESH_SEND)link_publications(link);
}

/* design/algorithm-sources.md#programkernel_call */
static void *link_progress(void *argument){
  struct mesh_worker *worker=argument;
  pthread_setname_np(worker->direction==MESH_SEND?"mesh.rdma.send":"mesh.rdma.receive");
  if(worker->direction==MESH_RECEIVE)
    for(uint32_t q=0;q<=(uint32_t)worker->link->qps;q++)link_receive(worker->link,q);
  else
    for(uint32_t q=0;q<(uint32_t)worker->link->qps;q++)link_send_ready(worker->link,q);
  while(atomic_load_explicit(&worker->link->progressing,memory_order_acquire))
    mesh_progress(worker->link,worker->direction);
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

/* ledger D14: destroying the queue pairs ends their work requests; release what they occupied */
static int link_down(struct mesh_link *link){
  struct hdr *M=link->M;
  link_stop(link);
  if(!down_pair()){ link_error(M,errno?errno:EIO,1); return -1; }
  atomic_store_explicit(&M->device_client,0,memory_order_seq_cst);
  memset(link->queues,0,2*(MESH_QPS+1)*sizeof *link->queues);
  memset(link->queued,0,(size_t)MESH_QPS*mesh_blocks(M));
  for(uint32_t q=0;q<MESH_QPS;q++)link->ready[q]=(struct mesh_ready){MESH_ABSENT,MESH_ABSENT};
  link->client=0;
  atomic_store(&M->port.phase,MESH_PAIRING);
  return 0;
}

int main(int argc,char**argv){
  const char *peer=NULL,*name=MESH_NAME; int me=0,layout=0,memory_check=0; double pct=0;
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
    else if(!strcmp(argv[i],"--memory-check")) memory_check=1;
    else if(!strcmp(argv[i],"-s") && i+1<argc) name=argv[++i];
    else if(argv[i][0]=='-') die("unknown bridge option");
    else peer=argv[i];
  }
  if(me<0 || me>=65535 || !isfinite(pct) || pct<0 || pct>100 || !arena_pages || !block_pages || arena_pages<block_pages) die("bridge geometry");
  const uint32_t pg=(uint32_t)getpagesize();
  /* ledger D6: a block is one message of whole 4 KB frames, at most 16,773,120 bytes */
  if((uint64_t)block_pages*pg>16773120 || (uint64_t)block_pages*pg%4096) die("block exceeds one message");
  struct hdr geometry={0};
  uint64_t length=mesh_layout(&geometry,pg,(uint32_t)block_pages,(uint32_t)arena_pages);
  uint64_t ram=0; size_t rl=sizeof ram; sysctlbyname("hw.memsize",&ram,&rl,NULL,0);
  if(pct && length>(uint64_t)(pct/100*(double)ram)) die("configured graph exceeds page capacity");
  if(memory_check){ mesh_memory_warning(length,1); return 0; }
  if(layout){ printf("%llu\n",(unsigned long long)length); return 0; }
  atexit(down); struct sigaction sa={0}; sa.sa_handler=onsig;
  sigaction(SIGINT,&sa,NULL); sigaction(SIGTERM,&sa,NULL); sigaction(SIGHUP,&sa,NULL); signal(SIGPIPE,SIG_IGN);
  mesh_memory_warning(length,1);
  shm_unlink(name); int fd=shm_open(name,O_CREAT|O_RDWR,MESH_MODE); if(fd<0) die("shm");
  if(ftruncate(fd,(off_t)length)) die("ftruncate"); fchmod(fd,MESH_MODE);
  size_t bank=(size_t)1<<32;
  unsigned char *reserved=mmap(NULL,length+bank,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0); if(reserved==MAP_FAILED) die("reserve");
  unsigned char *aligned=(unsigned char*)(((uintptr_t)reserved+bank-1)&~(uintptr_t)(bank-1));
  if(aligned>reserved) munmap(reserved,(size_t)(aligned-reserved));
  munmap(aligned+length,(size_t)(reserved+length+bank-(aligned+length)));
  struct hdr *M=mmap(aligned,length,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_FIXED,fd,0); close(fd);
  if(M==MAP_FAILED) die("mmap"); shm=name;
  *M=geometry; M->node=(uint32_t)me; M->version=MESH_VERSION;
  for(uint32_t r=0;r<mesh_rows(M);r++) atomic_store_explicit(&mesh_page(M)[r],MESH_ABSENT,memory_order_relaxed);
  struct mesh_link link={.M=M}; provider=&link.provider;
  link.frames=(int)(block_pages*pg/4096); link.budget=4095/link.frames;
  link.qps=getenv("MESH_QPS")?atoi(getenv("MESH_QPS")):1; if(link.qps<1) link.qps=1; if(link.qps>MESH_QPS) link.qps=MESH_QPS;
  M->qps=(uint32_t)link.qps;
  atomic_store(&M->bridge_pid,(uint64_t)getpid());
  __sync_synchronize(); M->magic=MESH_MAGIC;
  link.queues=calloc(2*(MESH_QPS+1),sizeof *link.queues);
  link.queued=calloc((size_t)MESH_QPS*mesh_blocks(M),1);
  link.ready_next=calloc((size_t)MESH_QPS*mesh_blocks(M),sizeof *link.ready_next);
  link.send_heads=calloc(mesh_rows(M),sizeof *link.send_heads);
  provider->completions=calloc(2*QD,sizeof *provider->completions);
  int status=0;
  if(!link.queues || !link.queued || !link.ready_next || !link.send_heads || !provider->completions){ status=ENOMEM; goto teardown; }
  atomic_store_explicit(&link.collecting,1,memory_order_release);
  status=pthread_create(&link.collector,NULL,link_collect,&link);
  if(status){atomic_store(&link.collecting,0);goto teardown;}
  atomic_store(&M->port.phase,MESH_PAIRING);
  fprintf(stderr,"bridge node %d: %d queue pair(s), %d maximum frames per block, %d index slots per direction\n",me,link.qps,link.frames,link.budget);

  while(!stop){
    /* D14: a listener failure is recorded and retried in process, never an exit */
    if(lsock<0 && listener_up()){ link_error(M,errno?errno:EIO,1); continue; }
    uint64_t client=atomic_load_explicit(&M->client,memory_order_acquire);
    /* D14: the connection follows the attached client */
    if(link.client && client!=link.client && link_down(&link)) continue;
    if(!link.client && client && atomic_load_explicit(&M->configured,memory_order_acquire)==client){
      atomic_store_explicit(&M->device_client,client,memory_order_seq_cst);
      if(atomic_load_explicit(&M->client,memory_order_seq_cst)!=client || atomic_load_explicit(&M->configured,memory_order_seq_cst)!=client){
        atomic_store_explicit(&M->device_client,0,memory_order_seq_cst);continue;
      }
      /* D13 */
      int setup=verbs_up(peer,(char*)M,length,M->data_off,me,(uint32_t)(block_pages*pg),link.qps+1,link_configure,&link,client);
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
  atomic_store_explicit(&M->device_client,0,memory_order_seq_cst);
  if(atomic_exchange_explicit(&link.collecting,0,memory_order_acq_rel))pthread_join(link.collector,NULL);
  if(lsock>=0) close(lsock);
  free(provider->completions); free(link.queues);free(link.queued);free(link.ready_next);free(link.send_heads);free(link.send_edges);
  for(uint32_t i=0;i<2*MESH_QPS;i++)free(link.transfers[i]);
  for(uint32_t q=0;q<MESH_QPS;q++){free(link.arrivals[q].pages);free(link.arrivals[q].indices);}
  munmap(M,length); return status;
}
