// mesh-flow -- see RDMA-FIRST.md
#include "mesh-mem.h"
#include "mesh.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <infiniband/verbs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/select.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>



static struct ibv_qp *g_qp; static struct ibv_pd *g_pd; static struct mesh_mem g_mem;
static struct ibv_cq *g_cq; static struct ibv_context *g_ctx;
static volatile sig_atomic_t g_stop;
static const char *g_shm=NULL;
static void on_sig(int s){ (void)s; g_stop = 1; }
static void teardown(void){
  if(g_qp){ ibv_destroy_qp(g_qp); g_qp=NULL; }
  if(g_cq){ ibv_destroy_cq(g_cq); g_cq=NULL; }
  mesh_mem_release(&g_mem);
  if(g_pd){ ibv_dealloc_pd(g_pd); g_pd=NULL; }
  if(g_ctx){ ibv_close_device(g_ctx); g_ctx=NULL; }
  if(g_shm){ shm_unlink(g_shm); g_shm=NULL; }
}
static void die(const char*m){ fprintf(stderr,"%s\n",m); exit(1); }
static double now(void){ struct timeval t; gettimeofday(&t,NULL); return t.tv_sec+t.tv_usec/1e6; }

struct qpi { uint32_t qpn,psn; uint16_t lid; uint8_t gid[16]; };
static int wait_fd(int fd,int w,int s){ fd_set f; FD_ZERO(&f); FD_SET(fd,&f);
  struct timeval tv={s,0}; return select(fd+1,w?NULL:&f,w?&f:NULL,NULL,&tv); }
static int oob(const char*host,int port,int secs){
  struct addrinfo h={.ai_family=AF_UNSPEC,.ai_socktype=SOCK_STREAM},*r; char p[16];
  struct timeval tv={secs,0}; snprintf(p,sizeof p,"%d",port);
  if(host){ if(getaddrinfo(host,p,&h,&r)) die("getaddrinfo");
    int fd=socket(r->ai_family,SOCK_STREAM,0); fcntl(fd,F_SETFL,O_NONBLOCK);
    if(connect(fd,r->ai_addr,r->ai_addrlen)&&errno!=EINPROGRESS) die("connect");
    if(wait_fd(fd,1,secs)<1) die("connect timed out");
    int e=0; socklen_t el=sizeof e; getsockopt(fd,SOL_SOCKET,SO_ERROR,&e,&el);
    if(e) die("connect refused");
    fcntl(fd,F_SETFL,0); freeaddrinfo(r);
    setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof tv);
    setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&tv,sizeof tv); return fd; }
  h.ai_flags=AI_PASSIVE; h.ai_family=AF_INET6; if(getaddrinfo(NULL,p,&h,&r)) die("getaddrinfo");
  int l=socket(r->ai_family,SOCK_STREAM,0),on=1,off=0;
  setsockopt(l,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on);
  setsockopt(l,IPPROTO_IPV6,IPV6_V6ONLY,&off,sizeof off);
  if(bind(l,r->ai_addr,r->ai_addrlen)) die("bind");
  listen(l,1); freeaddrinfo(r);
  if(wait_fd(l,0,secs)<1){ close(l); die("no peer connected in time"); }
  int fd=accept(l,NULL,NULL); close(l); if(fd<0) die("accept");
  setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof tv);
  setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&tv,sizeof tv); return fd; }


int main(int argc,char**argv){
  const char *peer=NULL;
  const int pgsz=4096, port=18519, tmo=30; int npages=240, seconds=5;
  int node_idx=0; const char *shmname="/mesh0"; double pct=0.0;
  // Only what cannot be inferred: which node this is, how much of it the mesh
  // holds, and who to reach. Everything else is derived. There is deliberately
  // no switch for the page size, the registration chunk, the send window or
  // the device: each of those can produce a run that looks healthy and moves
  // corrupt data, and an operator should not be able to select that.
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-I")) node_idx=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-M")) pct=atof(argv[++i]);
    else if(!strcmp(argv[i],"-s")) shmname=argv[++i];
    else if(!strcmp(argv[i],"-T")) seconds=atoi(argv[++i]);
    else if(argv[i][0]=='-'){ fprintf(stderr,"unknown option %s\n",argv[i]); return 2; }
    else peer=argv[i];
  }
  if(pct<=0.0) pct=25.0;

  atexit(teardown);
  struct sigaction sa={0}; sa.sa_handler=on_sig;
  sigaction(SIGINT,&sa,NULL); sigaction(SIGTERM,&sa,NULL); sigaction(SIGHUP,&sa,NULL);

  int frames = (pgsz+4095)/4096;
  if(npages < 4) die("page pool too small; lower -P");

  struct ibv_device **dl=ibv_get_device_list(NULL); if(!dl) die("no rdma devices");
  struct ibv_device *d=NULL; struct ibv_port_attr pa;
  for(int i=0;dl[i];i++){                        // take the link that is up
    struct ibv_context *c=ibv_open_device(dl[i]); if(!c) continue;
    if(!ibv_query_port(c,1,&pa) && pa.state==IBV_PORT_ACTIVE){ d=dl[i]; g_ctx=c; break; }
    ibv_close_device(c);
  }
  if(!d){ fprintf(stderr,"no RDMA port is up -- no cable, or far end not RDMA-enabled\n"); return 2; }
  fprintf(stderr,"link %s\n", ibv_get_device_name(d));
  g_pd=ibv_alloc_pd(g_ctx); if(!g_pd) die("alloc_pd");

  uint64_t node_ram=0; size_t nrl=sizeof node_ram;
  sysctlbyname("hw.memsize",&node_ram,&nrl,NULL,0);
  if(pct>0.0 && node_ram) npages=(int)((pct/100.0*(double)node_ram)/(double)pgsz);
  if(npages<4) die("page pool too small");
  uint32_t ring_cap=4096;
  size_t rb=(size_t)ring_cap*sizeof(struct mesh_desc);
  size_t hdr_end=(sizeof(struct mesh_hdr)+MESH_CL-1)/MESH_CL*MESH_CL;
  uint64_t sub_off=hdr_end, cmp_off=sub_off+rb, rel_off=cmp_off+rb;
  size_t data_off=((rel_off+rb)+65535)/65536*65536;
  size_t span=(size_t)pgsz*npages;
  size_t total=data_off+span;
  shm_unlink(shmname);
  int rfd=shm_open(shmname,O_CREAT|O_RDWR,0600); if(rfd<0) die("shm_open");
  if(ftruncate(rfd,(off_t)total)) die("ftruncate");
  void *rbase=mmap(NULL,total,PROT_READ|PROT_WRITE,MAP_SHARED,rfd,0);
  if(rbase==MAP_FAILED) die("mmap region");
  memset(rbase,0,hdr_end+3*rb);
  g_shm=shmname;
  struct mesh RG={.h=(struct mesh_hdr*)rbase,.base=(unsigned char*)rbase,.len=total,.fd=rfd};
  RG.h->pgsz=pgsz; RG.h->npages=npages; RG.h->ring_cap=ring_cap; RG.h->node=node_idx;
  RG.h->data_off=data_off; RG.h->sub_off=sub_off; RG.h->cmp_off=cmp_off;
  RG.h->rel_off=rel_off; RG.h->bytes=span; RG.h->node_ram=node_ram;
  RG.h->version=MESH_VERSION;
  __sync_synchronize(); RG.h->magic=MESH_MAGIC;
  void *mem=(char*)rbase+data_off;

  if(mesh_mem_map(&g_mem,g_pd,mem,span,(size_t)pgsz,IBV_ACCESS_LOCAL_WRITE))
    die("register region");
  size_t ppc = g_mem.chunk/(size_t)pgsz;              // pages per region
  fprintf(stderr,"region %s: %.3f GB (%.2f%% of node) in %zu regions\n",
     shmname, span/1e9, node_ram?100.0*span/(double)node_ram:0.0, g_mem.nseg);
  g_cq=ibv_create_cq(g_ctx,4096,NULL,NULL,0); if(!g_cq) die("create_cq");
  struct ibv_qp_init_attr ia={.send_cq=g_cq,.recv_cq=g_cq,.qp_type=IBV_QPT_UC,
    .cap={.max_send_wr=4095,.max_recv_wr=4095,.max_send_sge=1,.max_recv_sge=1}};
  g_qp=ibv_create_qp(g_pd,&ia); if(!g_qp) die("create_qp");
  
  struct ibv_qp_attr qa; struct ibv_qp_init_attr qi;
  if(ibv_query_qp(g_qp,&qa,IBV_QP_CAP,&qi)) die("query_qp");
  int gr_send=qi.cap.max_send_wr, gr_recv=qi.cap.max_recv_wr;
  struct ibv_qp_attr at={.qp_state=IBV_QPS_INIT,.pkey_index=0,.port_num=1,.qp_access_flags=0};
  if(ibv_modify_qp(g_qp,&at,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)) die("INIT");

  union ibv_gid gid; ibv_query_gid(g_ctx,1,0,&gid);
  uint32_t psn=lrand48()&0xffffff;
  struct qpi me={g_qp->qp_num,psn,pa.lid},you; memcpy(me.gid,&gid,16);
  int fd=oob(peer,port,tmo);
  if(peer){ if(write(fd,&me,sizeof me)!=sizeof me) die("oob w");
            if(read(fd,&you,sizeof you)!=sizeof you) die("oob r"); }
  else    { if(read(fd,&you,sizeof you)!=sizeof you) die("oob r");
            if(write(fd,&me,sizeof me)!=sizeof me) die("oob w"); }
  close(fd);
  struct ibv_qp_attr r={.qp_state=IBV_QPS_RTR,.path_mtu=IBV_MTU_4096,.rq_psn=you.psn,
    .dest_qp_num=you.qpn,.ah_attr={.dlid=you.lid,.port_num=1,.is_global=1,
    .grh={.hop_limit=1,.sgid_index=0}}};
  memcpy(&r.ah_attr.grh.dgid,you.gid,16);
  if(ibv_modify_qp(g_qp,&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN)) die("RTR");
  struct ibv_qp_attr st={.qp_state=IBV_QPS_RTS,.sq_psn=psn};
  if(ibv_modify_qp(g_qp,&st,IBV_QP_STATE|IBV_QP_SQ_PSN)) die("RTS");

  // A page's address is arithmetic, not a stored field, and its state is not
  // stored at all: the completion says what the page was doing.
  char *mem_c = (char*)mem;
  #define PAGE(i) (mem_c + (size_t)(i)*(size_t)pgsz)
  int *freelist = malloc(npages*sizeof(int)), nfree=0;
  if(!freelist) die("alloc freelist");
  uint32_t maxpay = (uint32_t)pgsz - (uint32_t)sizeof(struct wire);
  int rx_pages = npages/4;                       // cap the receive pool in bytes,
  int rx_cap = (int)(1000000000ull/(unsigned)pgsz);   // not pages: 1 GB either way
  if(rx_pages > rx_cap) rx_pages = rx_cap;
  if(rx_pages < 8) rx_pages = npages;              // tiny pool: no arena
  for(int i=0;i<rx_pages;i++) freelist[nfree++]=i;   // arena pages are the app's

  #define PG_LKEY(i) (g_mem.mr[(size_t)(i)/ppc]->lkey)
  // Every send is this. There is one way to put a page on the wire.
  #define POST_SEND(i) ({ struct ibv_sge _g={(uintptr_t)PAGE(i),(uint32_t)pgsz,PG_LKEY(i)}; \
    struct ibv_send_wr _w={.wr_id=(uint64_t)(i),.sg_list=&_g,.num_sge=1, \
      .opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*_b; \
    ibv_post_send(g_qp,&_w,&_b); })
  int posted=0, sending=0;
  int arena_start=rx_pages;
  RG.h->arena_off = data_off + (uint64_t)rx_pages*(uint64_t)pgsz;
  RG.h->arena_pages = (uint32_t)(npages - rx_pages);
  RG.h->hdr_bytes = (uint32_t)sizeof(struct wire);

  // One way to give a page back. A page the bridge does not own -- an arena
  // page, or one handed to the application -- is simply never passed here.
  #define PUT(i) do{ freelist[nfree++]=(i); }while(0)
  int rx_target = gr_recv/frames;   if(rx_target > npages/4) rx_target = npages/4;
  if(rx_target<2) rx_target=2;
  int tx_budget = gr_send/frames;   if(tx_budget > npages/4) tx_budget = npages/4;
  if(tx_budget<1) tx_budget=1;
  fprintf(stderr,"arena %u pages = %.3f GB (%.2f%% of node); pool %d; rx/tx %d/%d\n",
     RG.h->arena_pages, (double)RG.h->arena_pages*(pgsz-sizeof(struct wire))/1e9,
     node_ram?100.0*(double)RG.h->arena_pages*pgsz/(double)node_ram:0.0,
     rx_pages, rx_target, tx_budget);
  unsigned long long rx=0, tx=0, bytes_rx=0, bytes_tx=0;
  unsigned long long delivered=0, cmp_full=0, app_sent=0;
  unsigned long long dropped=0;
  double t0=now(), tel_last=t0;

  while(!g_stop && now()-t0 < seconds){

    
    while(nfree>0 && posted<rx_target){
      int i=freelist[--nfree];
      struct ibv_sge g={(uintptr_t)PAGE(i),(uint32_t)pgsz,PG_LKEY(i)};
      struct ibv_recv_wr wr={.wr_id=(uint64_t)i,.sg_list=&g,.num_sge=1},*bad;
      if(ibv_post_recv(g_qp,&wr,&bad)){ freelist[nfree++]=i; break; }
      posted++;
    }

    // APPREL -- take back pages the application has finished with
    { struct mesh_desc dd; int nrel=0;
      while(nrel++ < 256 && !mesh_pop(&RG,&RG.h->rrel,RG.h->rel_off,&dd)){
        uint32_t p=dd.page;
        if(p < (uint32_t)arena_start){ PUT(p); } } }

    // APPSUB -- send what the application handed over
    if(atomic_load_explicit(&RG.h->client_pid,memory_order_acquire)){
      struct mesh_desc dd;
      for(int ns=0; ns<256 && sending<tx_budget &&
                    !mesh_pop(&RG,&RG.h->rsub,RG.h->sub_off,&dd); ns++){
        uint32_t p=dd.page; if(p>=(uint32_t)npages) continue;
        struct wire *wh=(struct wire*)PAGE(p);
        wh->magic=WIRE_MAGIC; wh->bytes=(dd.bytes>maxpay?maxpay:dd.bytes);
        wh->src=(uint16_t)node_idx; wh->dst=dd.node; wh->hops=0;
        // A failed post must not put an arena page on the free list; it is
        // the application's page, not the pool's.
        if(POST_SEND(p)) dropped++;
        else { sending++; app_sent++;
               atomic_fetch_add_explicit(&RG.h->sent,1,memory_order_relaxed); }
      }
    }

    
    struct ibv_wc wc[32];
    int k=ibv_poll_cq(g_cq,32,wc);
    if(k<0) die("poll_cq");
    for(int j=0;j<k;j++){
      int i=(int)wc[j].wr_id;
      if(i<0 || i>=npages){ dropped++; fprintf(stderr,"BUG wr_id %d out of pool\n",i); continue; }
      if(wc[j].status!=IBV_WC_SUCCESS){
        if(!dropped++) fprintf(stderr,"wc %s on page %d\n",ibv_wc_status_str(wc[j].status),i);
        if(wc[j].opcode==IBV_WC_RECV) posted--; else sending--;
        PUT(i); continue; }
      if(wc[j].opcode==IBV_WC_RECV){
        // A received page is handled here, where it already is. There is no
        // second queue: the completion is the delivery.
        posted--; rx++; bytes_rx+=wc[j].byte_len;        struct wire *h=(struct wire*)PAGE(i);
        if(h->magic!=WIRE_MAGIC){ dropped++; PUT(i); continue; }
        uint32_t pay = h->bytes > maxpay ? maxpay : h->bytes;
        h->hops++;
        if(h->dst==(uint16_t)node_idx || h->hops>32){
          delivered++;
          if(atomic_load_explicit(&RG.h->client_pid,memory_order_acquire)){
            struct mesh_desc dd={.page=(uint32_t)i,.bytes=pay,.node=h->src};
            // Handed to the application. The bridge stops tracking it; it comes
            // back through the release ring or not at all.
            if(mesh_push(&RG,&RG.h->rcmp,RG.h->cmp_off,&dd)){ cmp_full++; PUT(i); }
            else atomic_fetch_add_explicit(&RG.h->recvd,1,memory_order_relaxed);
          } else PUT(i);
        } else {
          if(POST_SEND(i)){ dropped++; PUT(i); }
          else sending++;
        }
      } else {
        tx++; bytes_tx+=pgsz; sending--;
        // A forwarded page came from the pool and returns to it. A page the
        // application sent is the application's, and the bridge forgets it.
        if(i < arena_start) PUT(i);
      }
    }

    double t=now();
    if(t-tel_last >= 1.0){
      fprintf(stderr,"t=%.0f free=%d posted=%d sending=%d rx=%llu tx=%llu pool=%d/%d\n",
        t-t0, nfree, posted, sending, rx, tx, nfree+posted+sending, arena_start);
      tel_last=t;
    }
    if(!k) usleep(50);            // nothing completed; do not spin
  }
  double el=now()-t0;
  printf("node pages=%d pool=%d  rx=%llu tx=%llu  %.2f/%.2f Gbit/s in/out\n"
         "     delivered=%llu sent=%llu dropped=%llu cmp_full=%llu  pool %d/%d\n",
    npages, arena_start, rx, tx, bytes_rx*8/el/1e9, bytes_tx*8/el/1e9,
    delivered, app_sent, dropped, cmp_full, nfree+posted+sending, arena_start);
  return 0;
}
