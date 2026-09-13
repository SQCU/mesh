#include "mesh-verbs.h"
#include "mesh-dataflow.h"

/* A work request this bridge posted whose completion it has not yet polled (ledger D3). */
struct mesh_posted { uint32_t row,page,plane; };
struct mesh_queue { struct mesh_posted posted[QD]; uint32_t head,tail,next; };
struct mesh_link {
  struct hdr *M; struct mesh_verbs provider; int qps,frames,budget; uint64_t client;
  struct mesh_queue *queues;
};
static struct mesh_queue *link_queue(struct mesh_link *link,uint32_t q,int direction){ return &link->queues[2*q+(uint32_t)direction]; }

/* ledger D12: status goes to the region's port metadata, read out of band */
static void link_error(struct hdr *M,int64_t code,uint32_t domain){
  M->port.code=code; M->port.domain=domain; M->port.when=(uint64_t)monotime();
}

/* ledger D1, D4: a literal SEND of a produced block, or a RECV posted on the consumer's own pages */
static int link_post(struct mesh_link *link,uint32_t q,int direction,struct mesh_posted entry){
  struct hdr *M=link->M; struct mesh_verbs *v=&link->provider;
  struct ibv_sge span=region_sge((char*)M,M->data_off+(size_t)entry.page*M->pgsz,M->block*M->pgsz);
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
  queue->posted[queue->tail++%QD]=entry;
  return 0;
}

/* design/collective-dependency-ledger.md#d16-fixed-connection-setup-before-numerical-execution */
static void link_receive(void *state,uint32_t q){
  struct mesh_link *link=state;
  struct hdr *M=link->M; struct mesh_verbs *v=&link->provider;
  _Atomic uint32_t *table=mesh_page(M);
    struct mesh_queue *in=link_queue(link,q,MESH_RECEIVE);
    uint32_t length=atomic_load_explicit(mesh_order_length(M,q,MESH_RECEIVE),memory_order_acquire);
    while(length && in->tail-in->head<(uint32_t)v->receive_capacity){
      uint32_t row=mesh_order(M,q,MESH_RECEIVE)[in->next%length];
      if(!mesh_receive_postable(M,row)) break;
      struct mesh_posted entry={row,atomic_load_explicit(&table[row],memory_order_acquire),0};
      mesh_receive_posted(M,entry.row,entry.page);
      if(link_post(link,q,MESH_RECEIVE,entry)){ mesh_receive_complete(M,entry.row,entry.page,0); break; }
      in->next++;
    }
}

/* ledger D2, D3, D5, D7, D8, D9 */
static void mesh_progress(struct mesh_link *link){
  struct hdr *M=link->M; struct mesh_verbs *v=&link->provider;
  _Atomic uint32_t *table=mesh_page(M);
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    link_receive(link,q);
    uint32_t length;
    /* D5, D7: send produced blocks in this queue's order; producers never wait on this */
    struct mesh_queue *out=link_queue(link,q,MESH_SEND);
    length=atomic_load_explicit(mesh_order_length(M,q,MESH_SEND),memory_order_acquire);
    while(length && out->tail-out->head<(uint32_t)v->send_capacity){
      uint32_t row=mesh_order(M,q,MESH_SEND)[out->next%length];
      struct mesh_posted entry={row,atomic_load_explicit(&table[row],memory_order_acquire),mesh_send(M)[row]};
      if(!entry.plane || !mesh_send_postable(M,entry.row,entry.page)) break;
      entry.plane--;
      mesh_bits_set(M,MESH_PAGE_HOT,entry.page,M->block);
      if(link_post(link,q,MESH_SEND,entry)){ mesh_send_complete(M,entry.row,entry.page,entry.plane); break; }
      out->next++;
    }
  }
  /* D3 */
  int count=ibv_poll_cq(v->completion_queue,2*link->budget*link->qps,v->completions);
  if(count<0){ link_error(M,count,3); return; }
  for(int i=0;i<count;i++){
    struct ibv_wc *wc=&v->completions[i];
    uint32_t q=0; while(q<(uint32_t)link->qps && v->pairs[q]->qp_num!=wc->qp_num) q++;
    int direction=(wc->wr_id>>63)?MESH_RECEIVE:MESH_SEND;
    uint32_t row=(uint32_t)wc->wr_id;
    if(wc->status) link_error(M,wc->status,2);
    if(q==(uint32_t)link->qps){ link_error(M,EPROTO,4); continue; }
    /* ledger D5: "The order of processing a Work Request is guaranteed per Work Queue according to the order
       the Work Requests were added to it." A completion is therefore the head of its queue's posted order. */
    struct mesh_queue *queue=link_queue(link,q,direction);
    if(queue->head==queue->tail || queue->posted[queue->head%QD].row!=row){ link_error(M,EPROTO,4); continue; }
    struct mesh_posted entry=queue->posted[queue->head%QD];
    queue->head++;
    if(direction==MESH_RECEIVE) mesh_receive_complete(M,entry.row,entry.page,!wc->status);
    else mesh_send_complete(M,entry.row,entry.page,entry.plane);
  }
}

/* ledger D14: destroying the queue pairs ends their work requests; release what they occupied */
static int link_down(struct mesh_link *link){
  struct hdr *M=link->M;
  if(!down_pair()){ link_error(M,errno?errno:EIO,1); return -1; }
  for(uint32_t q=0;q<MESH_QPS;q++) for(int d=0;d<2;d++){
    struct mesh_queue *queue=link_queue(link,q,d);
    for(;queue->head!=queue->tail;queue->head++){
      struct mesh_posted entry=queue->posted[queue->head%QD];
      mesh_bits_clear(M,MESH_PAGE_HOT,entry.page,M->block);
      mesh_bits_clear(M,MESH_ROW_HOT,entry.row,M->block);
    }
    queue->head=queue->tail=queue->next=0;
  }
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
  /* ledger D6: a block is one message of whole 4 KB frames, at most 16,773,120 bytes */
  if((uint64_t)block_pages*pg>16773120 || (uint64_t)block_pages*pg%4096) die("block exceeds one message");
  struct hdr geometry={0};
  uint64_t length=mesh_layout(&geometry,pg,(uint32_t)block_pages,(uint32_t)arena_pages);
  uint64_t ram=0; size_t rl=sizeof ram; sysctlbyname("hw.memsize",&ram,&rl,NULL,0);
  if(pct && length>(uint64_t)(pct/100*(double)ram)) die("configured graph exceeds page capacity");
  if(layout){ printf("%llu\n",(unsigned long long)length); return 0; }
  atexit(down); struct sigaction sa={0}; sa.sa_handler=onsig;
  sigaction(SIGINT,&sa,NULL); sigaction(SIGTERM,&sa,NULL); sigaction(SIGHUP,&sa,NULL); signal(SIGPIPE,SIG_IGN);
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
  link.queues=calloc(2*MESH_QPS,sizeof *link.queues);
  provider->completions=calloc((size_t)2*(size_t)link.budget*(size_t)link.qps,sizeof *provider->completions);
  int status=0;
  if(!link.queues || !provider->completions){ status=ENOMEM; goto teardown; }
  atomic_store(&M->port.phase,MESH_PAIRING);
  fprintf(stderr,"bridge node %d: %d queue pair(s), %d frames per block, %d blocks per direction\n",me,link.qps,link.frames,link.budget);

  while(!stop){
    struct timespec idle={0,1000000};
    /* D14: a listener failure is recorded and retried in process, never an exit */
    if(lsock<0 && listener_up()){ link_error(M,errno?errno:EIO,1); nanosleep(&idle,NULL); continue; }
    uint64_t client=atomic_load_explicit(&M->client,memory_order_acquire);
    /* D14: the connection follows the attached client */
    if(link.client && client!=link.client && link_down(&link)) continue;
    if(!link.client && client && atomic_load_explicit(&M->configured,memory_order_acquire)==client){
      /* D13 */
      if(verbs_up(peer,(char*)M,length,M->data_off,me,(uint32_t)(block_pages*pg),link.qps,link_receive,&link)){
        link_error(M,errno?errno:EIO,1);
        link_down(&link);
        continue;
      }
      link.client=client;
      snprintf(M->port.device,sizeof M->port.device,"%s",ibv_get_device_name(provider->context->device));
      M->port.peer=(uint16_t)expected_peer;
      atomic_store(&M->port.phase,MESH_PAIRED);
      fprintf(stderr,"bridge node %d paired for client %llu\n",me,(unsigned long long)client);
    }
    if(link.client) mesh_progress(&link);
    else nanosleep(&idle,NULL);
  }
teardown:
  fprintf(stderr,"bridge node %d stopping: code=%lld domain=%u errno=%d\n",me,(long long)M->port.code,M->port.domain,errno);
  atomic_store(&M->port.phase,MESH_STOPPED);
  if(!down_verbs()){ fprintf(stderr,"verbs teardown failed: %s\n",strerror(errno)); return 1; }
  if(lsock>=0) close(lsock);
  free(provider->completions); free(link.queues);
  munmap(M,length); return status;
}
