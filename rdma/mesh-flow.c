#include "mesh-verbs.h"
#include "mesh-dataflow.h"
/* design/pages-and-functions.md#what-the-page-table-is
   The bridge is the hardware thread that owns the queue pairs. It never stops for anything but a signal.
   It posts landing blocks from the free index and committed blocks from the submission index, strictly
   within the device's frame budget so a post cannot fail, and on completion writes the page table:
   arrival stores page[] and ORs PRESENT; send completion ORs the NIC's read bit for the client generation
   that committed it. A client that dies leaves nothing the bridge depends on: committed sends still go
   out, delivered blocks are returned to the free index once no client can consume them. */
struct mesh_link {
  struct hdr *M; struct mesh_verbs provider; int qps,frames,budget,retry;
  int receives[MESH_QPS],sends[MESH_QPS],receive_frames,send_frames,next_receive,next_send;
  uint64_t passes;
};

static void link_post(struct mesh_link *link,int q,int send,uint64_t id,uint32_t page){
  struct mesh_verbs *v=&link->provider;
  size_t bytes=(size_t)link->M->block*link->M->pgsz;
  int base=q*link->budget,index=base+(send?v->sending:v->receiving),slot=send?link->budget*link->qps+index:index;
  v->sges[slot][0]=region_sge((char*)link->M,link->M->data_off+(size_t)page*link->M->pgsz,(uint32_t)bytes);
  if(send){
    v->sends[index]=(struct ibv_send_wr){.wr_id=id,.sg_list=&v->sges[slot][0],.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED};
    if(v->sending) v->sends[index-1].next=&v->sends[index];
    v->sending++;
  } else {
    v->receives[index]=(struct ibv_recv_wr){.wr_id=id,.sg_list=&v->sges[slot][0],.num_sge=1};
    if(v->receiving) v->receives[index-1].next=&v->receives[index];
    v->receiving++;
  }
  link->retry=q;
}

/* Post what has been prepared on one queue pair. A failure is recorded and the batch stays prepared for the next pass. */
static void link_flush(struct mesh_link *link,int q){
  struct mesh_verbs *v=&link->provider; struct hdr *M=link->M; int base=q*link->budget;
  if(v->receiving){
    struct ibv_recv_wr *bad=NULL; int e=ibv_post_recv(v->pairs[q],&v->receives[base],&bad);
    if(e){ M->port.code=e; M->port.domain=1; M->port.when=(uint64_t)monotime(); return; }
    link->receives[q]+=v->receiving; link->receive_frames+=v->receiving*link->frames; v->receiving=0;
  }
  if(v->sending){
    struct ibv_send_wr *bad=NULL; int e=ibv_post_send(v->pairs[q],&v->sends[base],&bad);
    if(e){ M->port.code=e; M->port.domain=1; M->port.when=(uint64_t)monotime(); return; }
    link->sends[q]+=v->sending; link->send_frames+=v->sending*link->frames; v->sending=0;
  }
}

static void mesh_progress(struct mesh_link *link){
  struct hdr *M=link->M; struct mesh_verbs *v=&link->provider;
  uint64_t entry; int posted=0;
  if(v->receiving||v->sending) link_flush(link,link->retry);
  /* Landing blocks: fill the device's receive budget from the free index, spread across queue pairs. */
  while(!v->receiving && !v->sending && link->receive_frames+link->frames<=4095 && !mesh_pop(M,FREE,&entry)){
    int q=link->next_receive++%link->qps;
    link_post(link,q,0,(UINT64_C(1)<<63)|entry,(uint32_t)entry); link_flush(link,q); posted=1;
  }
  /* Committed blocks: fill the device's send budget from the submission index. */
  while(!v->receiving && !v->sending && link->send_frames+link->frames<=4095 && !mesh_pop(M,SUB,&entry)){
    int q=link->next_send++%link->qps;
    uint32_t page=mesh_submission_page(entry);
    if((uint64_t)page+M->block>mesh_rows(M)){ M->port.code=EFAULT; M->port.domain=2; atomic_fetch_add_explicit(&M->bad,1,memory_order_relaxed); continue; }
    atomic_fetch_add_explicit(&M->sending,1,memory_order_acq_rel);
    link_post(link,q,1,entry,page); link_flush(link,q); posted=1;
    atomic_fetch_add_explicit(&M->sent,1,memory_order_relaxed);
  }
  int count=ibv_poll_cq(v->completion_queue,2*link->budget*link->qps,v->completions);
  if(count<0){ M->port.code=count; M->port.domain=3; M->port.when=(uint64_t)monotime(); atomic_fetch_add_explicit(&M->bad,1,memory_order_relaxed); return; }
  for(int i=0;i<count;i++){
    struct ibv_wc *wc=&v->completions[i];
    int q=0; while(q<link->qps && v->pairs[q]->qp_num!=wc->qp_num) q++;
    if(q==link->qps) q=0;
    if(wc->status){
      M->port.code=wc->status; M->port.domain=2; M->port.when=(uint64_t)monotime(); atomic_fetch_add_explicit(&M->bad,1,memory_order_relaxed);
      fprintf(stderr,"completion error: status=%d vendor=%u opcode=%d qp=%u wr_id=%llx\n",wc->status,wc->vendor_err,wc->opcode,wc->qp_num,(unsigned long long)wc->wr_id);
    }
    if(wc->wr_id>>63){
      uint32_t page=(uint32_t)wc->wr_id;
      link->receives[q]--; link->receive_frames-=link->frames;
      const struct mesh_tag *tag=(const struct mesh_tag*)mesh_at(M,page+M->block-1);
      uint32_t row=tag->binding<MESH_BINDINGS?mesh_base(M)[tag->binding]:MESH_ABSENT;
      if(wc->status || tag->magic!=MESH_TAG || row==MESH_ABSENT || (uint64_t)row+(uint64_t)tag->index*M->block+M->block>mesh_rows(M)){
        /* Nothing can consume it: back to the free index at once. */
        if(!wc->status) atomic_fetch_add_explicit(&M->bad,1,memory_order_relaxed);
        mesh_push(M,FREE,page); continue;
      }
      row+=tag->index*M->block;
      _Atomic uint32_t *table=mesh_page(M);
      for(uint32_t k=0;k<M->block;k++) atomic_store_explicit(&table[row+k],page+k,memory_order_release);
      atomic_fetch_or_explicit(&mesh_landed(M)[(page/M->block)/64],UINT64_C(1)<<((page/M->block)%64),memory_order_acq_rel);
      mesh_bits_set(M,MESH_PRESENT,row,M->block);
      atomic_fetch_add_explicit(&M->recvd,1,memory_order_relaxed);
    } else {
      link->sends[q]--; link->send_frames-=link->frames;
      atomic_fetch_sub_explicit(&M->sending,1,memory_order_acq_rel);
      /* The read bit belongs to the client generation that committed the block; a later client's rows are not touched. */
      if(mesh_submission_generation(wc->wr_id)==(atomic_load_explicit(&M->generation,memory_order_acquire)&15))
        mesh_bits_set(M,MESH_READ+(int)mesh_submission_plane(wc->wr_id),mesh_submission_row(wc->wr_id),M->block);
    }
  }
  /* A dead client cannot return the blocks it holds; the bridge does, so the peer's committed sends can land. */
  if(!posted && !count && !(++link->passes&4095)){
    uint64_t client=atomic_load_explicit(&M->client,memory_order_acquire);
    if(client && kill((pid_t)client,0) && errno==ESRCH){
      atomic_store_explicit(&M->client,0,memory_order_release);
      atomic_fetch_add_explicit(&M->generation,1,memory_order_acq_rel);
      mesh_reclaim_landed(M);
    }
  }
}

int main(int argc,char**argv){
  const char *peer=NULL,*name=MESH_NAME; int me=0,layout=0; double pct=0;
  uint64_t arena_pages=0,receive_pages=0,block_pages=0;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-I") && i+1<argc) me=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-M") && i+1<argc) pct=atof(argv[++i]);
    else if((!strcmp(argv[i],"-A") || !strcmp(argv[i],"-R") || !strcmp(argv[i],"-B")) && i+1<argc){
      char kind=argv[i][1],*end;
      uint64_t pages=strtoull(argv[++i],&end,10);
      if(*end || !pages || pages>INT32_MAX) die("configured page count");
      if(kind=='A') arena_pages=pages; else if(kind=='R') receive_pages=pages; else block_pages=pages;
    }
    else if(!strcmp(argv[i],"--layout")) layout=1;
    else if(!strcmp(argv[i],"-s") && i+1<argc) name=argv[++i];
    else if(argv[i][0]=='-') die("unknown bridge option");
    else peer=argv[i];
  }
  if(me<0 || me>=65535 || !isfinite(pct) || pct<0 || pct>100 || !arena_pages || !receive_pages || !block_pages || receive_pages%block_pages) die("bridge geometry");
  const uint32_t pg=(uint32_t)getpagesize();
  if((uint64_t)block_pages*pg>16773120 || (uint64_t)block_pages*pg%4096) die("block exceeds one message");
  struct hdr geometry={0};
  uint64_t length=mesh_layout(&geometry,pg,(uint32_t)block_pages,(uint32_t)receive_pages,(uint32_t)arena_pages);
  uint64_t ram=0; size_t rl=sizeof ram; sysctlbyname("hw.memsize",&ram,&rl,NULL,0);
  if(pct && length>(uint64_t)(pct/100*(double)ram)) die("configured graph exceeds page capacity");
  if(mesh_rows(&geometry)/geometry.block>MESH_RING) die("configured blocks exceed the submission index");
  if(layout){ printf("%llu\n",(unsigned long long)length); return 0; }
  atexit(down); struct sigaction sa={0}; sa.sa_handler=onsig;
  sigaction(SIGINT,&sa,NULL); sigaction(SIGTERM,&sa,NULL); sigaction(SIGHUP,&sa,NULL); signal(SIGPIPE,SIG_IGN);
  shm_unlink(name); int fd=shm_open(name,O_CREAT|O_RDWR,MESH_MODE); if(fd<0) die("shm");
  if(ftruncate(fd,(off_t)length)) die("ftruncate"); fchmod(fd,MESH_MODE);
  struct hdr *M=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0); close(fd);
  if(M==MAP_FAILED) die("mmap"); shm=name;
  *M=geometry; M->node=(uint32_t)me; M->version=MESH_VERSION;
  for(uint32_t r=0;r<mesh_rows(M);r++) atomic_store_explicit(&mesh_page(M)[r],MESH_ABSENT,memory_order_relaxed);
  for(uint32_t b=0;b<MESH_BINDINGS;b++) mesh_base(M)[b]=MESH_ABSENT;
  for(uint32_t page=0;page<M->pool;page+=M->block) mesh_push(M,FREE,page);
  atomic_store(&M->bridge_pid,(uint64_t)getpid());
  __sync_synchronize(); M->magic=MESH_MAGIC;
  struct mesh_link link={.M=M}; provider=&link.provider;
  /* The 4095 work-request frames are a device budget per direction, not per queue pair. */
  link.frames=(int)(block_pages*pg/4096); link.budget=4095/link.frames;
  link.qps=getenv("MESH_QPS")?atoi(getenv("MESH_QPS")):1; if(link.qps<1) link.qps=1; if(link.qps>MESH_QPS) link.qps=MESH_QPS;
  size_t entries=(size_t)link.budget*link.qps;
  provider->completions=calloc(2*entries,sizeof *provider->completions);
  provider->sges=calloc(2*entries,sizeof *provider->sges);
  provider->receives=calloc(entries,sizeof *provider->receives);
  provider->sends=calloc(entries,sizeof *provider->sends);
  int status=0;
  if(!provider->completions || !provider->sges || !provider->receives || !provider->sends){ status=ENOMEM; goto teardown; }
  status=listener_up() || verbs_up(peer,(char*)M,length,me,(uint32_t)(block_pages*pg),NULL,link.qps);
  if(status){ M->port.code=errno?errno:EIO; M->port.domain=1; goto teardown; }
  snprintf(M->port.device,sizeof M->port.device,"%s",ibv_get_device_name(provider->context->device));
  M->port.peer=(uint16_t)expected_peer;
  atomic_store(&M->port.phase,MESH_PAIRED);
  fprintf(stderr,"bridge node %d: %d queue pair(s), %d frames per block, %d blocks per direction\n",me,link.qps,link.frames,link.budget);
  /* A committed block is sent before the bridge honours a signal; a second signal ends it regardless. */
  while(stop<2 && (!stop || atomic_load_explicit(&M->r[SUB].head,memory_order_acquire)!=atomic_load_explicit(&M->r[SUB].tail,memory_order_acquire) || atomic_load_explicit(&M->sending,memory_order_acquire))) mesh_progress(&link);
teardown:
  fprintf(stderr,"bridge node %d stopping: code=%lld domain=%u errno=%d sent=%llu recvd=%llu bad=%llu\n",me,(long long)M->port.code,M->port.domain,errno,
    (unsigned long long)atomic_load(&M->sent),(unsigned long long)atomic_load(&M->recvd),(unsigned long long)atomic_load(&M->bad));
  atomic_store(&M->port.phase,MESH_STOPPED);
  if(!down_verbs()){ fprintf(stderr,"verbs teardown failed: %s\n",strerror(errno)); return 1; }
  if(lsock>=0) close(lsock);
  free(provider->completions); free(provider->sges); free(provider->receives); free(provider->sends);
  munmap(M,length); return status;
}
