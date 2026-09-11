#include "mesh-verbs.h"
#include "mesh-dataflow.h"
/* design/pages-and-functions.md#what-the-page-table-is
   The bridge owns the queue pairs. It posts landing blocks from the free index and outbound blocks
   from the submission index across all queue pairs, and on completion writes the page table:
   arrival stores page[] and ORs PRESENT; send completion ORs the NIC's read bit. Nothing else. */
struct mesh_link { struct hdr *M; struct mesh_verbs provider; int qps,capacity; int receives[MESH_QPS],sends[MESH_QPS]; };

static void link_post(struct mesh_link *link,int q,int send,uint64_t id,uint32_t page){
  struct mesh_verbs *v=&link->provider;
  size_t bytes=(size_t)link->M->block*link->M->pgsz;
  int base=q*link->capacity,index=base+(send?v->sending:v->receiving);
  int slot=send?link->capacity*link->qps+index:index;
  v->sges[slot][0]=region_sge((char*)link->M,link->M->data_off+(size_t)page*link->M->pgsz,(uint32_t)bytes);
  if(send){
    v->sends[index]=(struct ibv_send_wr){.wr_id=id,.sg_list=&v->sges[slot][0],.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED};
    if(v->sending) v->sends[index-1].next=&v->sends[index];
    v->sending++; link->sends[q]++;
  } else {
    v->receives[index]=(struct ibv_recv_wr){.wr_id=id,.sg_list=&v->sges[slot][0],.num_sge=1};
    if(v->receiving) v->receives[index-1].next=&v->receives[index];
    v->receiving++; link->receives[q]++;
  }
}

static void link_flush(struct mesh_link *link,int q){
  struct mesh_verbs *v=&link->provider; int base=q*link->capacity;
  if(v->receiving){ struct ibv_recv_wr *bad=NULL; int e=ibv_post_recv(v->pairs[q],&v->receives[base],&bad); if(e){ link->M->port.code=e; link->M->port.domain=1; stop=1; } v->receiving=0; }
  if(v->sending){ struct ibv_send_wr *bad=NULL; int e=ibv_post_send(v->pairs[q],&v->sends[base],&bad); if(e){ link->M->port.code=e; link->M->port.domain=1; stop=1; } v->sending=0; }
}

static void mesh_progress(struct mesh_link *link){
  struct hdr *M=link->M; struct mesh_verbs *v=&link->provider;
  uint64_t entry;
  for(int q=0;q<link->qps;q++){
    while(link->receives[q]<v->receive_capacity && !mesh_pop(M,FREE,&entry)) link_post(link,q,0,(UINT64_C(1)<<63)|entry,(uint32_t)entry);
    while(link->sends[q]<v->send_capacity && !mesh_pop(M,SUB,&entry)){
      uint32_t row=(uint32_t)(entry>>8);
      link_post(link,q,1,entry,atomic_load_explicit(&mesh_page(M)[row],memory_order_acquire));
      atomic_fetch_add_explicit(&M->sent,1,memory_order_relaxed);
    }
    link_flush(link,q);
  }
  int count=ibv_poll_cq(v->completion_queue,(v->send_capacity+v->receive_capacity)*link->qps,v->completions);
  if(count<0){ M->port.code=count; M->port.domain=3; stop=1; return; }
  for(int i=0;i<count;i++){
    struct ibv_wc *wc=&v->completions[i];
    int q=0; while(q<link->qps && v->pairs[q]->qp_num!=wc->qp_num) q++;
    if(q==link->qps) q=0;
    if(wc->status){ M->port.code=wc->status; M->port.domain=2; M->port.when=(uint64_t)monotime(); atomic_fetch_add_explicit(&M->bad,1,memory_order_relaxed); stop=1; }
    if(wc->wr_id>>63){
      uint32_t page=(uint32_t)wc->wr_id;
      link->receives[q]--;
      const struct mesh_tag *tag=(const struct mesh_tag*)mesh_at(M,page+M->block-1);
      if(wc->status) continue;
      if(tag->magic!=MESH_TAG || tag->binding>=MESH_BINDINGS){ M->port.code=EBADMSG; M->port.domain=2; stop=1; continue; }
      uint32_t row=mesh_base(M)[tag->binding]+tag->index*M->block;
      if(row+M->block>mesh_rows(M)){ M->port.code=ERANGE; M->port.domain=2; stop=1; continue; }
      _Atomic uint32_t *table=mesh_page(M);
      for(uint32_t k=0;k<M->block;k++) atomic_store_explicit(&table[row+k],page+k,memory_order_release);
      mesh_bits_set(M,MESH_PRESENT,row,M->block);
      atomic_fetch_add_explicit(&M->recvd,1,memory_order_relaxed);
    } else {
      link->sends[q]--;
      mesh_bits_set(M,MESH_READ+(int)(wc->wr_id&7),(uint32_t)(wc->wr_id>>8),M->block);
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
  if(layout){ printf("%llu\n",(unsigned long long)length); return 0; }
  atexit(down); struct sigaction sa={0}; sa.sa_handler=onsig;
  sigaction(SIGINT,&sa,NULL); sigaction(SIGTERM,&sa,NULL); sigaction(SIGHUP,&sa,NULL); signal(SIGPIPE,SIG_IGN);
  shm_unlink(name); int fd=shm_open(name,O_CREAT|O_RDWR,MESH_MODE); if(fd<0) die("shm");
  if(ftruncate(fd,(off_t)length)) die("ftruncate"); fchmod(fd,MESH_MODE);
  struct hdr *M=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0); close(fd);
  if(M==MAP_FAILED) die("mmap"); shm=name;
  *M=geometry; M->node=(uint32_t)me; M->version=MESH_VERSION;
  for(uint32_t r=0;r<mesh_rows(M);r++) atomic_store_explicit(&mesh_page(M)[r],MESH_ABSENT,memory_order_relaxed);
  for(uint32_t page=0;page<M->pool;page+=M->block) mesh_push(M,FREE,page);
  atomic_store(&M->bridge_pid,(uint64_t)getpid());
  __sync_synchronize(); M->magic=MESH_MAGIC;
  struct mesh_link link={.M=M}; provider=&link.provider;
  /* Enough queue pairs that every landing block can be posted at once. */
  size_t capacity=QD/(block_pages*pg/4096),blocks=receive_pages/block_pages;
  int qps=(int)((blocks+capacity-1)/capacity); if(qps<1) qps=1; if(qps>MESH_QPS) qps=MESH_QPS;
  link.qps=qps; link.capacity=(int)capacity;
  provider->completions=calloc(2*capacity*qps,sizeof *provider->completions);
  provider->sges=calloc(2*capacity*qps,sizeof *provider->sges);
  provider->receives=calloc(capacity*qps,sizeof *provider->receives);
  provider->sends=calloc(capacity*qps,sizeof *provider->sends);
  int status=0;
  if(!provider->completions || !provider->sges || !provider->receives || !provider->sends){ status=ENOMEM; goto teardown; }
  uint64_t entry;
  for(size_t i=0;i<capacity && !mesh_pop(M,FREE,&entry);i++){
    provider->sges[i][0]=(struct ibv_sge){.addr=(uintptr_t)mesh_at(M,(uint32_t)entry),.length=(uint32_t)(block_pages*pg)};
    provider->receives[i]=(struct ibv_recv_wr){.wr_id=(UINT64_C(1)<<63)|entry,.sg_list=&provider->sges[i][0],.num_sge=1,.next=i+1<capacity?&provider->receives[i+1]:NULL};
    link.receives[0]++;
  }
  if(link.receives[0]) provider->receives[link.receives[0]-1].next=NULL;
  status=listener_up() || verbs_up(peer,(char*)M,length,me,(uint32_t)(block_pages*pg),link.receives[0]?provider->receives:NULL,qps);
  if(status){ M->port.code=errno?errno:EIO; M->port.domain=1; goto teardown; }
  provider->receiving=0;
  if(provider->receive_capacity<(int)capacity) link.capacity=provider->receive_capacity;
  snprintf(M->port.device,sizeof M->port.device,"%s",ibv_get_device_name(provider->context->device));
  M->port.peer=(uint16_t)expected_peer;
  atomic_store(&M->port.phase,MESH_PAIRED);
  fprintf(stderr,"bridge node %d: %d queue pairs, %d blocks each, block %llu pages\n",me,qps,link.capacity,(unsigned long long)block_pages);
  while(!stop) mesh_progress(&link);
  status=M->port.code!=0;
teardown:
  atomic_store(&M->port.phase,MESH_STOPPED);
  if(!down_verbs()){ fprintf(stderr,"verbs teardown failed: %s\n",strerror(errno)); return 1; }
  if(lsock>=0) close(lsock);
  free(provider->completions); free(provider->sges); free(provider->receives); free(provider->sends);
  munmap(M,length); return status;
}
