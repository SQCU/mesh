#include "mesh-verbs.h"
#include "mesh-dataflow.h"
#include "mesh-memory.h"
#include <pthread.h>

/* design/algorithm-sources.md#programcopy */
struct mesh_posted { uint32_t row,page,plane,index,bytes,frames; };
struct mesh_send_edge {uint32_t queue,index,next;};
struct mesh_ready {uint32_t head,tail;};
struct mesh_hold { uint32_t rows,pages; };
struct mesh_queue { struct mesh_posted posted[QD]; uint32_t head,tail,next,frames; };
struct mesh_pending { struct mesh_transfer entries[QD]; uint32_t head,tail; };
struct mesh_index_frame { uint32_t queue,count; struct mesh_transfer entries[(MESH_INDEX_BYTES-8)/sizeof(struct mesh_transfer)]; };
struct mesh_worker { struct mesh_link *link; pthread_t thread; uint32_t direction; };
struct mesh_link {
  struct mesh_worker workers[2];
  uint32_t worker_count;
  _Atomic int progressing;
  struct hdr *M; struct mesh_verbs provider; int qps,frames,budget; uint64_t client;
  struct mesh_queue *queues;
  struct mesh_pending *pending;
  struct mesh_hold *holds;
  uint8_t *active,*queued;
  uint32_t *ready_next,*send_heads;
  struct mesh_send_edge *send_edges;
  struct mesh_ready ready[MESH_QPS];
};
/* design/algorithm-sources.md#programcopy */
static struct mesh_queue *link_queue(struct mesh_link *link,uint32_t q,int direction){ return &link->queues[2*q+(uint32_t)direction]; }
/* design/algorithm-sources.md#programcopy */
static void link_error(struct hdr *M,int64_t code,uint32_t domain){
  M->port.code=code; M->port.domain=domain; M->port.when=(uint64_t)monotime();
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
    struct ibv_send_wr request={.wr_id=entry.row,.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad=NULL;
    error=ibv_post_send(v->pairs[q],&request,&bad);
  } else {
    struct ibv_recv_wr request={.wr_id=(UINT64_C(1)<<63)|entry.row,.sg_list=&span,.num_sge=1},*bad=NULL;
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
  struct mesh_transfer *transfer=&mesh_transfers(link->M,q,MESH_SEND)[index];
  if(link->active[key] || link->queued[key] || !mesh_send_postable(link->M,transfer))return 0;
  struct mesh_ready *ready=&link->ready[q];
  link->queued[key]=1;link->ready_next[key]=MESH_ABSENT;
  if(ready->tail==MESH_ABSENT)ready->head=index;
  else link->ready_next[(size_t)q*mesh_blocks(link->M)+ready->tail]=index;
  ready->tail=index;return 1;
}
/* design/algorithm-sources.md#programcopy */
static int link_configure(void *state,int socket,double deadline){
  struct mesh_link *link=state;struct hdr *m=link->M;
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    uint32_t sends=atomic_load(mesh_order_length(m,q,MESH_SEND)),receives=atomic_load(mesh_order_length(m,q,MESH_RECEIVE)),peer_receives=0;
    if(exchange(socket,&receives,&peer_receives,sizeof receives,deadline))return -1;
    if(peer_receives!=sends){fprintf(stderr,"transfer count mismatch queue=%u sends=%u peer_receives=%u\n",q,sends,peer_receives);errno=EPROTO;return -1;}
    uint32_t count=sends>receives?sends:receives;
    struct mesh_transfer *mine=calloc(count?count:1,sizeof *mine),*peer=calloc(count?count:1,sizeof *peer);
    if(!mine || !peer){free(mine);free(peer);errno=ENOMEM;return -1;}
    memcpy(mine,mesh_transfers(m,q,MESH_RECEIVE),receives*sizeof *mine);
    int error=exchange(socket,mine,peer,count*sizeof *mine,deadline);
    struct mesh_transfer *out=mesh_transfers(m,q,MESH_SEND);
    for(uint32_t i=0;i<sends && !error;i++){
      if(out[i].binding!=peer[i].binding || out[i].offset!=peer[i].offset || out[i].bytes!=peer[i].bytes){fprintf(stderr,"send plan mismatch queue=%u index=%u local=%llu,%u,%u peer=%llu,%u,%u\n",q,i,(unsigned long long)out[i].binding,out[i].offset,out[i].bytes,(unsigned long long)peer[i].binding,peer[i].offset,peer[i].bytes);errno=EPROTO;error=-1;break;}
      out[i].peer_row=peer[i].local_row;out[i].peer_page=peer[i].local_page;out[i].peer_index=peer[i].index;
    }
    if(!error){
      memset(mine,0,count*sizeof *mine);memcpy(mine,out,sends*sizeof *mine);
      error=exchange(socket,mine,peer,count*sizeof *mine,deadline);
      struct mesh_transfer *in=mesh_transfers(m,q,MESH_RECEIVE);
      for(uint32_t i=0;i<receives && !error;i++){
        if(in[i].binding!=peer[i].binding || in[i].offset!=peer[i].offset || in[i].bytes!=peer[i].bytes){fprintf(stderr,"receive plan mismatch queue=%u index=%u local=%llu,%u,%u peer=%llu,%u,%u\n",q,i,(unsigned long long)in[i].binding,in[i].offset,in[i].bytes,(unsigned long long)peer[i].binding,peer[i].offset,peer[i].bytes);errno=EPROTO;error=-1;break;}
        in[i].peer_row=peer[i].local_row;in[i].peer_page=peer[i].local_page;in[i].peer_index=peer[i].index;
      }
    }
    free(mine);free(peer);if(error)return error;
    for(int d=0;d<2;d++)for(uint32_t i=0;i<atomic_load(mesh_order_length(m,q,d));i++){
      uint32_t frames=mesh_transfers(m,q,d)[i].bytes/4096;
      if(frames>link->provider.capacity[q][d]){
        fprintf(stderr,"transfer queue=%u direction=%d index=%u requires=%u frames available=%u\n",q,d,i,frames,link->provider.capacity[q][d]);
        errno=EMSGSIZE;return -1;
      }
    }
  }
  size_t count=0;
  for(uint32_t q=0;q<(uint32_t)link->qps;q++)count+=(size_t)atomic_load(mesh_order_length(m,q,MESH_SEND))*m->block;
  free(link->send_edges);link->send_edges=calloc(count?count:1,sizeof *link->send_edges);
  if(!link->send_edges)return -1;
  for(uint32_t row=0;row<mesh_rows(m);row++)link->send_heads[row]=MESH_ABSENT;
  size_t at=0;
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    link->ready[q]=(struct mesh_ready){MESH_ABSENT,MESH_ABSENT};
    uint32_t length=atomic_load(mesh_order_length(m,q,MESH_SEND));
    struct mesh_transfer *transfers=mesh_transfers(m,q,MESH_SEND);
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
      struct mesh_pending *pending=&link->pending[q];
      if(pending->head==pending->tail)return;
      struct mesh_transfer t=pending->entries[pending->head%QD];
      if(t.bytes/4096>link->provider.capacity[q][MESH_RECEIVE]-in->frames || !mesh_receive_postable(M,t.peer_row))return;
      struct mesh_posted entry={.row=t.peer_row,.page=t.peer_page,.index=t.peer_index,.bytes=t.bytes};
      mesh_receive_posted(M,entry.row,entry.page);
      if(link_post(link,q,MESH_RECEIVE,entry)){mesh_receive_complete(M,entry.row,entry.page,0);return;}
      pending->head++;
    }
  }
}
/* design/algorithm-sources.md#programcopy */
static void link_hold(struct mesh_link *link,uint32_t q,const struct mesh_transfer *transfer){
  struct hdr *m=link->M;
  link->active[(size_t)q*mesh_blocks(m)+transfer->index]=1;
  for(uint32_t i=0;i<m->block;i++){
    link->holds[transfer->local_row+i].rows++;
    link->holds[transfer->local_page+i].pages++;
  }
  mesh_bits_set(m,MESH_ROW_HOT,transfer->local_row,m->block);
  mesh_bits_set(m,MESH_PAGE_HOT,transfer->local_page,m->block);
}
/* design/algorithm-sources.md#programcopy */
static void link_release(struct mesh_link *link,uint32_t q,struct mesh_posted entry){
  struct hdr *m=link->M;
  for(uint32_t i=0;i<m->block;i++){
    if(!--link->holds[entry.row+i].rows)mesh_bits_clear(m,MESH_ROW_HOT,entry.row+i,1);
    if(!--link->holds[entry.page+i].pages)mesh_bits_clear(m,MESH_PAGE_HOT,entry.page+i,1);
  }
  link->active[(size_t)q*mesh_blocks(m)+entry.index]=0;
}
/* design/algorithm-sources.md#programkernel_call */
static void link_send_ready(struct mesh_link *link,uint32_t q){
  struct hdr *M=link->M;struct mesh_verbs *v=&link->provider;
  uint32_t iq=(uint32_t)link->qps;
  struct mesh_queue *indices=link_queue(link,iq,MESH_SEND);
  struct mesh_queue *out=link_queue(link,q,MESH_SEND);
  struct mesh_transfer *transfers=mesh_transfers(M,q,MESH_SEND);
  struct mesh_ready *ready=&link->ready[q];
  while(ready->head!=MESH_ABSENT && out->frames<v->capacity[q][MESH_SEND] &&
        indices->frames<v->capacity[iq][MESH_SEND] && indices->tail-indices->head<(uint32_t)link->budget){
    uint32_t slot=indices->tail%(uint32_t)link->budget;
    struct mesh_index_frame *frame=link_indices(link,MESH_SEND,slot);
    frame->queue=q;frame->count=0;
    uint32_t available=v->capacity[q][MESH_SEND]-out->frames;
    uint32_t previous=MESH_ABSENT,index=ready->head;
    while(index!=MESH_ABSENT && frame->count<sizeof frame->entries/sizeof *frame->entries){
      size_t key=(size_t)q*mesh_blocks(M)+index;
      uint32_t next=link->ready_next[key],cost=transfers[index].bytes/4096;
      if(cost>available){previous=index;index=next;continue;}
      if(previous==MESH_ABSENT)ready->head=next;
      else link->ready_next[(size_t)q*mesh_blocks(M)+previous]=next;
      if(ready->tail==index)ready->tail=previous;
      link->queued[key]=0;frame->entries[frame->count++]=transfers[index];
      available-=cost;index=next;
    }
    if(!frame->count)break;
    if(link_post(link,iq,MESH_SEND,(struct mesh_posted){.row=slot})){
      for(uint32_t i=0;i<frame->count;i++)link_ready(link,q,frame->entries[i].index);
      break;
    }
    for(uint32_t i=0;i<frame->count;i++){
      struct mesh_transfer t=frame->entries[i];
      link_hold(link,q,&t);
      struct mesh_posted entry={.row=t.local_row,.page=t.local_page,.plane=t.plane,.index=t.index,.bytes=t.bytes};
      if(link_post(link,q,MESH_SEND,entry)){link_release(link,q,entry);return;}
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
  if(direction==MESH_SEND)link_publications(link);
  struct ibv_wc *completions=v->completions+direction*QD;
  for(uint32_t cq=direction;cq<2*(iq+1);cq+=2){
  int count=ibv_poll_cq(v->completion_queues[cq],QD,completions);
  if(count<0){link_error(M,count,3);continue;}
  for(int i=0;i<count;i++){
    struct ibv_wc *wc=&completions[i];
    uint32_t q=cq/2;
    if(wc->qp_num!=v->pairs[q]->qp_num || ((wc->wr_id>>63)?MESH_RECEIVE:MESH_SEND)!=direction){link_error(M,EPROTO,4);continue;}
    uint32_t row=(uint32_t)wc->wr_id;
    if(wc->status)link_error(M,wc->status,2);
    struct mesh_queue *queue=link_queue(link,q,direction);
    if(queue->head==queue->tail || queue->posted[queue->head%QD].row!=row){link_error(M,EPROTO,4);continue;}
    struct mesh_posted entry=queue->posted[queue->head++%QD];queue->frames-=entry.frames;
    if(q==iq){
      if(direction==MESH_RECEIVE && !wc->status){
        struct mesh_index_frame *frame=link_indices(link,MESH_RECEIVE,row);
        if(frame->queue>=iq || frame->count>sizeof frame->entries/sizeof *frame->entries){link_error(M,EPROTO,4);continue;}
        struct mesh_pending *pending=&link->pending[frame->queue];
        if(frame->count>QD-(pending->tail-pending->head)){link_error(M,EOVERFLOW,4);continue;}
        for(uint32_t j=0;j<frame->count;j++){
          struct mesh_transfer t=frame->entries[j];
          pending->entries[pending->tail++%QD]=t;
        }
        link_receive(link,frame->queue);
      }
      if(direction==MESH_RECEIVE)link_receive(link,iq);
      else for(uint32_t data=0;data<iq;data++)link_send_ready(link,data);
    } else if(direction==MESH_RECEIVE){mesh_receive_complete(M,entry.row,entry.page,!wc->status);link_receive(link,q);}
    else {
      link_release(link,q,entry);mesh_send_complete(M,entry.row,entry.plane);
      link_send_ready(link,q);
    }
    if(direction==MESH_SEND)link_publications(link);
  }
  }
  if(direction==MESH_RECEIVE){
    link_receive(link,iq);
    for(uint32_t q=0;q<iq;q++)link_receive(link,q);
  } else {
    link_publications(link);
    for(uint32_t q=0;q<iq;q++)link_send_ready(link,q);
  }
}

/* design/algorithm-sources.md#programkernel_call */
static void *link_progress(void *argument){
  struct mesh_worker *worker=argument;
  pthread_setname_np(worker->direction==MESH_SEND?"mesh.rdma.send":"mesh.rdma.receive");
  if(worker->direction==MESH_RECEIVE)
    for(uint32_t q=0;q<=(uint32_t)worker->link->qps;q++)link_receive(worker->link,q);
  while(atomic_load_explicit(&worker->link->progressing,memory_order_acquire))
    mesh_progress(worker->link,worker->direction);
  return NULL;
}
/* design/algorithm-sources.md#programkernel_call */
static void link_stop(struct mesh_link *link){
  atomic_store_explicit(&link->progressing,0,memory_order_release);
  while(link->worker_count)pthread_join(link->workers[--link->worker_count].thread,NULL);
}

/* ledger D14: destroying the queue pairs ends their work requests; release what they occupied */
static int link_down(struct mesh_link *link){
  struct hdr *M=link->M;
  link_stop(link);
  if(!down_pair()){ link_error(M,errno?errno:EIO,1); return -1; }
  for(uint32_t q=0;q<MESH_QPS+1;q++) for(int d=0;d<2;d++){
    struct mesh_queue *queue=link_queue(link,q,d);
    for(;queue->head!=queue->tail;queue->head++){
      struct mesh_posted entry=queue->posted[queue->head%QD];
      if(q<(uint32_t)link->qps){
        if(d==MESH_SEND)link_release(link,q,entry);
        else {mesh_bits_clear(M,MESH_PAGE_HOT,entry.page,M->block);mesh_bits_clear(M,MESH_ROW_HOT,entry.row,M->block);}
      }
    }
    queue->head=queue->tail=queue->next=queue->frames=0;
  }
  memset(link->queued,0,(size_t)MESH_QPS*mesh_blocks(M));
  for(uint32_t q=0;q<MESH_QPS;q++)link->ready[q]=(struct mesh_ready){MESH_ABSENT,MESH_ABSENT};
  memset(link->pending,0,MESH_QPS*sizeof *link->pending);
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
  link.pending=calloc(MESH_QPS,sizeof *link.pending);
  link.holds=calloc(mesh_rows(M),sizeof *link.holds);
  link.active=calloc((size_t)MESH_QPS*mesh_blocks(M),sizeof *link.active);
  link.queued=calloc((size_t)MESH_QPS*mesh_blocks(M),1);
  link.ready_next=calloc((size_t)MESH_QPS*mesh_blocks(M),sizeof *link.ready_next);
  link.send_heads=calloc(mesh_rows(M),sizeof *link.send_heads);
  provider->completions=calloc(2*QD,sizeof *provider->completions);
  int status=0;
  if(!link.queues || !link.pending || !link.holds || !link.active || !link.queued || !link.ready_next || !link.send_heads || !provider->completions){ status=ENOMEM; goto teardown; }
  atomic_store(&M->port.phase,MESH_PAIRING);
  fprintf(stderr,"bridge node %d: %d queue pair(s), %d maximum frames per block, %d index slots per direction\n",me,link.qps,link.frames,link.budget);

  while(!stop){
    struct timespec idle={0,1000000};
    /* D14: a listener failure is recorded and retried in process, never an exit */
    if(lsock<0 && listener_up()){ link_error(M,errno?errno:EIO,1); nanosleep(&idle,NULL); continue; }
    uint64_t client=atomic_load_explicit(&M->client,memory_order_acquire);
    /* D14: the connection follows the attached client */
    if(link.client && client!=link.client && link_down(&link)) continue;
    if(!link.client && client && atomic_load_explicit(&M->configured,memory_order_acquire)==client){
      /* D13 */
      if(verbs_up(peer,(char*)M,length,M->data_off,me,(uint32_t)(block_pages*pg),link.qps+1,link_configure,&link)){
        link_error(M,errno?errno:EIO,1);
        link_down(&link);
        continue;
      }
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
      snprintf(M->port.device,sizeof M->port.device,"%s",ibv_get_device_name(provider->context->device));
      M->port.peer=(uint16_t)expected_peer;
      atomic_store(&M->port.phase,MESH_PAIRED);
      fprintf(stderr,"bridge node %d paired for client %llu\n",me,(unsigned long long)client);
    }
    nanosleep(&idle,NULL);
  }
teardown:
  link_stop(&link);
  fprintf(stderr,"bridge node %d stopping: code=%lld domain=%u errno=%d\n",me,(long long)M->port.code,M->port.domain,errno);
  atomic_store(&M->port.phase,MESH_STOPPED);
  if(!down_verbs()){ fprintf(stderr,"verbs teardown failed: %s\n",strerror(errno)); return 1; }
  if(lsock>=0) close(lsock);
  free(provider->completions); free(link.queues);free(link.pending);free(link.holds);free(link.active);free(link.queued);free(link.ready_next);free(link.send_heads);free(link.send_edges);
  munmap(M,length); return status;
}
