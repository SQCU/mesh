#include "mesh-verbs.h"

#include "mesh-dataflow.h"
enum { L_RECV=1, L_SEND };
struct mesh_link {
  struct hdr *pages;
  struct mesh_port_info *status;
  struct mesh_verbs provider;
  int receives, sends;
};
// ../design/algorithm-sources.md#transport-page-addressing
static int link_submit(struct mesh_link *link,uint32_t kind,uint32_t page,uint64_t header){
  struct mesh_verbs *v=&link->provider;
  provider=v;
  int index=kind==L_SEND?v->receive_capacity+v->sending/2:v->receiving/2;
  v->sges[index][0]=region_sge((char*)link->pages,header,sizeof(struct mesh_page_header));
  v->sges[index][1]=region_sge((char*)link->pages,link->pages->data_off+(size_t)page*link->pages->pgsz,link->pages->pgsz);
  struct mesh_send *record=(struct mesh_send*)((char*)link->pages+header);
  record->page=page; record->header.code=0; record->header.domain=0;
  if(kind==L_RECV){
    __atomic_store_n(&record->row,0,__ATOMIC_RELEASE);
    __atomic_fetch_or(mesh_hot_rx(link->pages)+page/64,UINT64_C(1)<<(page%64),__ATOMIC_RELEASE);
  }
  if(kind==L_SEND){
    for(int part=0;part<2;part++){
      struct ibv_sge *span=&v->sges[index][part];
      v->sends[v->sending]=(struct ibv_send_wr){.wr_id=header|(part?0:UINT64_C(1)<<62),
        .sg_list=span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED};
      if(v->sending) v->sends[v->sending-1].next=&v->sends[v->sending];
      v->sending++;
    }
    link->sends++;
  } else {
    for(int part=0;part<2;part++){
      struct ibv_sge *span=&v->sges[index][part];
      v->receives[v->receiving]=(struct ibv_recv_wr){.wr_id=(UINT64_C(1)<<63)|(part?0:UINT64_C(1)<<62)|page,
        .sg_list=span,.num_sge=1};
      if(v->receiving) v->receives[v->receiving-1].next=&v->receives[v->receiving];
      v->receiving++;
    }
    link->receives++;
  }
  return 0;
}
// ../design/algorithm-sources.md#transport-page-addressing
static void link_flush(struct mesh_link *link){
  struct mesh_verbs *v=&link->provider;
  if(v->receiving){
    struct ibv_recv_wr *bad=NULL;
    int error=ibv_post_recv(v->pair,v->receives,&bad);
    if(error){ link->status->code=error; link->status->domain=1; stop=1; }
    v->receiving=0;
  }
  if(v->sending){
    struct ibv_send_wr *bad=NULL;
    int error=ibv_post_send(v->pair,v->sends,&bad);
    if(error){ link->status->code=error; link->status->domain=1; stop=1; }
    v->sending=0;
  }
}
#define COUNT(field) atomic_fetch_add_explicit(&M->field,1,memory_order_relaxed)
// ../design/algorithm-sources.md#transport-page-addressing
static void mesh_progress(struct hdr *M,struct mesh_link *link,int *free_pages,int *available){
  struct mesh_verbs *v=&link->provider;
  struct desc descriptor;
  while(*available && link->receives<v->receive_capacity){
    uint32_t page=(uint32_t)free_pages[--*available];
    link_submit(link,L_RECV,page,(uint64_t)((char*)mesh_header(M,page)-(char*)M));
  }
  while(link->sends<v->send_capacity && !pop(M,SUB,&descriptor)){
    link_submit(link,L_SEND,descriptor.page,descriptor.header);
    COUNT(sent);
  }
  link_flush(link);
  int count=ibv_poll_cq(v->completion_queue,2*(v->send_capacity+v->receive_capacity),v->completions);
  if(count<0){ link->status->code=count; link->status->domain=3; stop=1; return; }
  for(int i=0;i<count;i++){
    struct ibv_wc *wc=&v->completions[i];
    int receive=(int)(wc->wr_id>>63),header_completion=(int)((wc->wr_id>>62)&1);
    uint64_t offset=receive?(uint64_t)((char*)mesh_header(M,(uint32_t)wc->wr_id)-(char*)M):wc->wr_id&~(UINT64_C(1)<<62);
    struct mesh_send *record=(struct mesh_send*)((char*)M+offset);
    if(wc->status){
      record->header.code=wc->status; record->header.domain=2;
      link->status->code=wc->status; link->status->domain=2; COUNT(bad); stop=1;
    }
    if(header_completion) continue;
    if(receive){
      uint32_t page=(uint32_t)wc->wr_id;
      __atomic_fetch_and(mesh_hot_rx(M)+page/64,~(UINT64_C(1)<<(page%64)),__ATOMIC_RELEASE);
      mesh_rows_received(M,(uint32_t)wc->wr_id); link->receives--; COUNT(recvd);
    } else {
      size_t request=mesh_hot_send_index(M,offset);
      __atomic_fetch_and(mesh_hot_tx(M)+request/64,~(UINT64_C(1)<<(request%64)),__ATOMIC_RELEASE);
      mesh_rows_sent(M,record); link->sends--;
    }
  }
}
// ../design/algorithm-sources.md#transport-page-addressing
int main(int argc,char**argv){
  const char *peer=NULL,*name=MESH_NAME; int me=0,layout=0; double pct=0;
  uint64_t arena_pages=0,receive_pages=0;
  struct mesh_link link={0};
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-I") && i+1<argc) me=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-M") && i+1<argc) pct=atof(argv[++i]);
    else if((!strcmp(argv[i],"-A") || !strcmp(argv[i],"-R")) && i+1<argc){
      int arena=!strcmp(argv[i],"-A"); char *end;
      uint64_t pages=strtoull(argv[++i],&end,10);
      if(*end || !pages || pages>INT32_MAX) die("configured page count");
      if(arena) arena_pages=pages; else receive_pages=pages;
    }
    else if(!strcmp(argv[i],"--layout")) layout=1;
    else if(!strcmp(argv[i],"-s") && i+1<argc) name=argv[++i];
    else if(argv[i][0]=='-') die("unknown bridge option");
    else peer=argv[i];
  }
  if(me<0 || me>=65535 || !isfinite(pct) || pct<0 || pct>100 || !arena_pages || !receive_pages) die("bridge geometry");
  atexit(down); struct sigaction sa={0}; sa.sa_handler=onsig;
  sigaction(SIGINT,&sa,NULL); sigaction(SIGTERM,&sa,NULL); sigaction(SIGHUP,&sa,NULL); signal(SIGPIPE,SIG_IGN);
  uint64_t ram=0; size_t rl=sizeof ram; sysctlbyname("hw.memsize",&ram,&rl,NULL,0);
  const uint32_t pg=(uint32_t)getpagesize();
  size_t h0=(mesh_hot_prefix((uint32_t)receive_pages,(uint32_t)arena_pages,pg)+pg-1)/pg*pg;
  uint64_t wanted=arena_pages+receive_pages;
  if(wanted>INT32_MAX) die("page index capacity");
  int np=(int)wanted,pool=(int)receive_pages;
  size_t d0=h0+mesh_send_storage_pages((size_t)pool,pg)*pg,span=(size_t)pg*np;
  if(pct && d0+span>(uint64_t)(pct/100*(double)ram)) die("configured graph exceeds page capacity");
  if(layout){ printf("%zu\n",d0+span); return 0; }
  shm_unlink(name); int fd=shm_open(name,O_CREAT|O_RDWR,MESH_MODE); if(fd<0) die("shm");
  if(ftruncate(fd,(off_t)(d0+span))) die("ftruncate"); fchmod(fd,MESH_MODE);
  struct hdr *M=mmap(NULL,d0+span,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0); close(fd);
  if(M==MAP_FAILED) die("mmap"); shm=name;
  M->pgsz=pg; M->pool=(uint32_t)pool; M->arena=(uint32_t)(np-pool); M->node=(uint32_t)me;
  M->version=MESH_VERSION; M->headers_off=h0; M->data_off=d0; atomic_store(&M->port_count,1);
  __sync_synchronize(); M->magic=MESH_MAGIC;
  int *free_pages=malloc((size_t)pool*sizeof *free_pages);
  if(!free_pages) die("receive indices");
  int available=pool;
  for(int i=0;i<pool;i++) free_pages[i]=i;
  struct mesh_port_info *port=mesh_ports(M);
  link.pages=M; link.status=port; provider=&link.provider;
  atomic_store(&M->bridge_pid,(uint64_t)getpid());
  size_t maximum=QD/((pg+sizeof(struct mesh_page_header)+4095)/4096);
  provider->completions=calloc(4*maximum,sizeof *provider->completions);
  provider->sges=calloc(2*maximum,sizeof *provider->sges);
  provider->receives=calloc(2*maximum,sizeof *provider->receives);
  provider->sends=calloc(2*maximum,sizeof *provider->sends);
  int status=0;
  if(!provider->completions || !provider->sges || !provider->receives || !provider->sends){ status=ENOMEM; goto teardown; }
  size_t initial=(size_t)pool<maximum?(size_t)pool:maximum;
  for(size_t i=0;i<initial;i++){
    uint32_t page=(uint32_t)free_pages[pool-1-(int)i];
    struct mesh_send *record=(struct mesh_send*)mesh_header(M,page);
    record->page=page; __atomic_store_n(&record->row,0,__ATOMIC_RELEASE);
    __atomic_fetch_or(mesh_hot_rx(M)+page/64,UINT64_C(1)<<(page%64),__ATOMIC_RELEASE);
    provider->sges[i][0]=(struct ibv_sge){.addr=(uintptr_t)&record->header,.length=sizeof record->header};
    provider->sges[i][1]=(struct ibv_sge){.addr=(uintptr_t)mesh_at(M,page),.length=pg};
    for(size_t part=0;part<2;part++){
      size_t at=2*i+part;
      provider->receives[at]=(struct ibv_recv_wr){.wr_id=(UINT64_C(1)<<63)|(part?0:UINT64_C(1)<<62)|page,
        .sg_list=&provider->sges[i][part],.num_sge=1,.next=at+1<2*initial?&provider->receives[at+1]:NULL};
    }
  }
  status=listener_up() || verbs_up(peer,(char*)M,d0+span,me,pg,sizeof(struct mesh_page_header),provider->receives);
  if(status){ port->code=errno?errno:EIO; port->domain=1; goto teardown; }
  link.receives=provider->receiving/2; available-=link.receives; provider->receiving=0;
  for(size_t i=(size_t)link.receives;i<initial;i++){
    uint32_t page=(uint32_t)free_pages[pool-1-(int)i];
    __atomic_fetch_and(mesh_hot_rx(M)+page/64,~(UINT64_C(1)<<(page%64)),__ATOMIC_RELEASE);
  }
  snprintf(port->device,sizeof port->device,"%s",ibv_get_device_name(provider->context->device));
  port->peer=(uint16_t)expected_peer;
  atomic_store(&port->phase,MESH_PAIRED);
  while(!stop){
    struct desc released;
    while(!pop(M,REL,&released)) free_pages[available++]=(int)released.page;
    mesh_progress(M,&link,free_pages,&available);
  }
  status=port->code!=0;
teardown:
  atomic_store(&port->phase,MESH_STOPPED);
  if(!down_verbs()){ fprintf(stderr,"verbs teardown failed: %s\n",strerror(errno)); return 1; }
  size_t receive_words=((size_t)M->pool+63)/64;
  size_t send_words=((size_t)M->arena*(M->pgsz/sizeof(struct mesh_send))+63)/64;
  for(size_t i=0;i<receive_words;i++) __atomic_store_n(mesh_hot_rx(M)+i,0,__ATOMIC_RELEASE);
  for(size_t i=0;i<send_words;i++) __atomic_store_n(mesh_hot_tx(M)+i,0,__ATOMIC_RELEASE);
  if(lsock>=0) close(lsock);
  free(provider->completions); free(provider->sges); free(provider->receives); free(provider->sends);
  free(free_pages); munmap(M,d0+span); return status;
}
