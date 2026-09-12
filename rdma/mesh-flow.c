#include "mesh-verbs.h"
#include "mesh-dataflow.h"

struct mesh_link {
  struct hdr *M; struct mesh_verbs provider; int qps,frames,budget;
  int receives[MESH_QPS],sends[MESH_QPS],receive_frames,send_frames,next_receive,next_send;
};


/* design/algorithm-sources.md#independent-verbs-progress */
static int link_post(struct mesh_link *link,int q,int send,uint64_t id,uint32_t page){
  struct mesh_verbs *v=&link->provider;
  struct ibv_sge span=region_sge((char*)link->M,link->M->data_off+(size_t)page*link->M->pgsz,link->M->block*link->M->pgsz);
  int error;
  if(send){
    struct ibv_send_wr request={.wr_id=id,.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad=NULL;
    error=ibv_post_send(v->pairs[q],&request,&bad);
    if(!error){ link->sends[q]++; link->send_frames+=link->frames; }
  } else {
    struct ibv_recv_wr request={.wr_id=id,.sg_list=&span,.num_sge=1},*bad=NULL;
    mesh_bits_set(link->M,MESH_PAGE_OWN,page,link->M->block);
    mesh_bits_set(link->M,MESH_PAGE_HOT,page,link->M->block);
    error=ibv_post_recv(v->pairs[q],&request,&bad);
    if(!error){ link->receives[q]++; link->receive_frames+=link->frames; }
    else {
      mesh_bits_clear(link->M,MESH_PAGE_HOT,page,link->M->block);
      mesh_bits_clear(link->M,MESH_PAGE_OWN,page,link->M->block);
    }
  }
  if(error){ link->M->port.code=error; link->M->port.domain=1; link->M->port.when=(uint64_t)monotime(); }
  return error;
}

/* design/algorithm-sources.md#independent-verbs-progress */
static void mesh_progress(struct mesh_link *link){
  struct hdr *M=link->M; struct mesh_verbs *v=&link->provider;
  uint64_t entry;
  mesh_reclaim_consumed(M);
  for(int n=0;n<link->budget && link->receive_frames+link->frames<=4095 && !mesh_pop(M,FREE,&entry);n++){
    int q=link->next_receive++%link->qps;
    if(link_post(link,q,0,(UINT64_C(1)<<63)|entry,(uint32_t)entry)) mesh_push(M,FREE,entry);
  }
  for(int n=0;n<link->budget && link->send_frames+link->frames<=4095 && !mesh_pop(M,SUB,&entry);n++){
    int q=link->next_send++%link->qps;
    uint32_t page=mesh_submission_page(entry);
    if(link_post(link,q,1,entry,page)) mesh_send_complete(M,entry);
    else atomic_fetch_add_explicit(&M->sent,1,memory_order_relaxed);
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
      mesh_bits_clear(M,MESH_PAGE_HOT,page,M->block);
      const struct mesh_tag *tag=(const struct mesh_tag*)mesh_at(M,page+M->block-1);
      if(wc->status || tag->magic!=MESH_TAG || tag->binding==MESH_ABSENT){
        if(!wc->status) atomic_fetch_add_explicit(&M->bad,1,memory_order_relaxed);
        fprintf(stderr,"landing rejected: page=%u status=%d bytes=%u magic=%08x binding=%u index=%u\n",page,wc->status,wc->byte_len,tag->magic,tag->binding,tag->index);
        mesh_bits_clear(M,MESH_PAGE_OWN,page,M->block);
        mesh_push(M,FREE,page); continue;
      }
      atomic_store_explicit(&mesh_landing_row(M)[page/M->block],MESH_ABSENT,memory_order_release);
      atomic_fetch_or_explicit(&mesh_landed(M)[(page/M->block)/64],UINT64_C(1)<<((page/M->block)%64),memory_order_acq_rel);
      atomic_fetch_add_explicit(&M->recvd,1,memory_order_relaxed);
    } else {
      link->sends[q]--; link->send_frames-=link->frames;

      mesh_send_complete(M,wc->wr_id);
    }
  }
  mesh_reclaim_consumed(M);
  mesh_reclaim_bindings(M);
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
  size_t bank=(size_t)1<<32;
  unsigned char *reserved=mmap(NULL,length+bank,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0); if(reserved==MAP_FAILED) die("reserve");
  unsigned char *aligned=(unsigned char*)(((uintptr_t)reserved+bank-1)&~(uintptr_t)(bank-1));
  if(aligned>reserved) munmap(reserved,(size_t)(aligned-reserved));
  munmap(aligned+length,(size_t)(reserved+length+bank-(aligned+length)));
  struct hdr *M=mmap(aligned,length,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_FIXED,fd,0); close(fd);
  if(M==MAP_FAILED) die("mmap"); shm=name;
  *M=geometry; M->node=(uint32_t)me; M->version=MESH_VERSION;
  for(uint32_t r=0;r<mesh_rows(M);r++) atomic_store_explicit(&mesh_page(M)[r],MESH_ABSENT,memory_order_relaxed);
  for(uint32_t b=0;b<MESH_BINDINGS;b++) mesh_base(M)[b]=UINT64_MAX;
  for(uint32_t page=0;page<M->pool;page+=M->block) mesh_push(M,FREE,page);
  atomic_store(&M->bridge_pid,(uint64_t)getpid());
  __sync_synchronize(); M->magic=MESH_MAGIC;
  struct mesh_link link={.M=M}; provider=&link.provider;

  link.frames=(int)(block_pages*pg/4096); link.budget=4095/link.frames;
  link.qps=getenv("MESH_QPS")?atoi(getenv("MESH_QPS")):1; if(link.qps<1) link.qps=1; if(link.qps>MESH_QPS) link.qps=MESH_QPS;
  size_t entries=(size_t)link.budget*link.qps;
  provider->completions=calloc(2*entries,sizeof *provider->completions);
  int status=0;
  if(!provider->completions){ status=ENOMEM; goto teardown; }
  status=listener_up() || verbs_up(peer,(char*)M,length,M->data_off,me,(uint32_t)(block_pages*pg),NULL,link.qps);
  if(status){ M->port.code=errno?errno:EIO; M->port.domain=1; goto teardown; }
  snprintf(M->port.device,sizeof M->port.device,"%s",ibv_get_device_name(provider->context->device));
  M->port.peer=(uint16_t)expected_peer;
  atomic_store(&M->port.phase,MESH_PAIRED);
  fprintf(stderr,"bridge node %d: %d queue pair(s), %d frames per block, %d blocks per direction\n",me,link.qps,link.frames,link.budget);

  while(!stop) mesh_progress(&link);
teardown:
  fprintf(stderr,"bridge node %d stopping: code=%lld domain=%u errno=%d sent=%llu recvd=%llu bad=%llu\n",me,(long long)M->port.code,M->port.domain,errno,
    (unsigned long long)atomic_load(&M->sent),(unsigned long long)atomic_load(&M->recvd),(unsigned long long)atomic_load(&M->bad));
  atomic_store(&M->port.phase,MESH_STOPPED);
  if(!down_verbs()){ fprintf(stderr,"verbs teardown failed: %s\n",strerror(errno)); return 1; }
  if(lsock>=0) close(lsock);
  free(provider->completions);
  munmap(M,length); return status;
}
