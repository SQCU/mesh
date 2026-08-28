// mesh-flow -- see RDMA-FIRST.md
#include "mesh-f.h"
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


enum { FREE=0, POSTED, FILLED, SENDING };
struct page { char *addr; uint32_t lkey; uint32_t refs; uint8_t state, held; };

struct wf { double n, mean, m2; };
static void wf_add(struct wf *w, double x){
  w->n += 1; double d = x - w->mean; w->mean += d / w->n; w->m2 += d * (x - w->mean); }
static double wf_var(struct wf *w){ return w->n > 1 ? w->m2 / (w->n - 1) : 0.0; }

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
  const char *dev=NULL,*peer=NULL;
  int pgsz=4096, npages=240, port=18519, tmo=30, seconds=5, source=0, inflight=0;
  int node_idx=0, target_idx=1, spanpg=1; const char *shmname="/mesh0"; double pct=0.0; unsigned route_path=0; unsigned egress=0; (void)egress;
  double tel_hz=1.0;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-d")) dev=argv[++i];
    else if(!strcmp(argv[i],"-P")) pgsz=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-n")) npages=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-p")) port=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-t")) tmo=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-T")) seconds=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-H")) tel_hz=atof(argv[++i]);
    else if(!strcmp(argv[i],"-w")) inflight=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-I")) node_idx=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-D")) target_idx=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-R")) route_path=(unsigned)strtoul(argv[++i],NULL,0);
    else if(!strcmp(argv[i],"-S")) spanpg=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-s")) shmname=argv[++i];
    else if(!strcmp(argv[i],"-M")) pct=atof(argv[++i]);
    else if(!strcmp(argv[i],"--source")) source=1;
    else peer=argv[i];
  }
  atexit(teardown);
  struct sigaction sa={0}; sa.sa_handler=on_sig;
  sigaction(SIGINT,&sa,NULL); sigaction(SIGTERM,&sa,NULL); sigaction(SIGHUP,&sa,NULL);

  int frames = (pgsz+4095)/4096;
  if(npages < 4) die("page pool too small; lower -P");

  struct ibv_device **dl=ibv_get_device_list(NULL); if(!dl) die("no rdma devices");
  struct ibv_device *d=NULL;
  for(int i=0;dl[i];i++) if(!dev||!strcmp(ibv_get_device_name(dl[i]),dev)){ d=dl[i]; break; }
  if(!d) die("device not found");
  g_ctx=ibv_open_device(d); if(!g_ctx) die("open_device");
  struct ibv_port_attr pa; if(ibv_query_port(g_ctx,1,&pa)) die("query_port");
  if(pa.state!=IBV_PORT_ACTIVE){
    fprintf(stderr,"%s is %s -- no cable, or far end not RDMA-enabled\n",
      ibv_get_device_name(d), ibv_port_state_str(pa.state)); return 2; }
  g_pd=ibv_alloc_pd(g_ctx); if(!g_pd) die("alloc_pd");

  uint64_t node_ram=0; size_t nrl=sizeof node_ram;
  sysctlbyname("hw.memsize",&node_ram,&nrl,NULL,0);
  if(pct>0.0 && node_ram) npages=(int)((pct/100.0*(double)node_ram)/(double)pgsz);
  if(npages<4) die("page pool too small");
  uint32_t ring_cap=4096;
  size_t rb=(size_t)ring_cap*sizeof(struct mesh_desc);
  size_t hdr_end=(sizeof(struct mesh_hdr)+MESH_CL-1)/MESH_CL*MESH_CL;
  uint64_t free_off=hdr_end, sub_off=free_off+rb, cmp_off=sub_off+rb, rel_off=cmp_off+rb;
  size_t data_off=((rel_off+rb)+65535)/65536*65536;
  size_t span=(size_t)pgsz*npages;
  size_t total=data_off+span;
  shm_unlink(shmname);
  int rfd=shm_open(shmname,O_CREAT|O_RDWR,0600); if(rfd<0) die("shm_open");
  if(ftruncate(rfd,(off_t)total)) die("ftruncate");
  void *rbase=mmap(NULL,total,PROT_READ|PROT_WRITE,MAP_SHARED,rfd,0);
  if(rbase==MAP_FAILED) die("mmap region");
  memset(rbase,0,hdr_end+4*rb);
  g_shm=shmname;
  struct mesh RG={.h=(struct mesh_hdr*)rbase,.base=(unsigned char*)rbase,.len=total,.fd=rfd};
  RG.h->pgsz=(uint32_t)pgsz; RG.h->npages=(uint32_t)npages; RG.h->ring_cap=ring_cap;
  RG.h->node=(uint32_t)node_idx; RG.h->data_off=data_off; RG.h->free_off=free_off;
  RG.h->sub_off=sub_off; RG.h->cmp_off=cmp_off; RG.h->rel_off=rel_off;
  RG.h->bytes=span; RG.h->node_ram=node_ram; RG.h->version=MESH_VERSION;
  __sync_synchronize(); RG.h->magic=MESH_MAGIC;
  void *mem=(char*)rbase+data_off;
  fprintf(stderr,"region %s: %d pages x %d B = %.3f GB = %.2f%% of node\n",
     shmname,npages,pgsz,span/1e9, node_ram?100.0*span/(double)node_ram:0.0);
  if(mesh_mem_init(&g_mem,g_pd,g_ctx,(size_t)pgsz)) die("mesh_mem_init");
  struct mesh_map map; mesh_map_open(&map,mem,span,IBV_ACCESS_LOCAL_WRITE);
  while(!mesh_map_done(&map) && mesh_map_step(&g_mem,&map,64)) ;
  fprintf(stderr,"map: %.3f/%.3f GB mapped in %d MRs\n",
          map.mapped/1e9, map.len/1e9, g_mem.nseg);
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

  struct page *pg = calloc(npages,sizeof *pg);
  int *freelist = malloc(npages*sizeof(int)), nfree=0;
  int *ready    = malloc(npages*sizeof(int)), nready=0, rhead=0;
  if(!pg||!freelist||!ready) die("alloc page table");
  uint32_t maxpay = (uint32_t)pgsz - (uint32_t)sizeof(struct wire);
  for(int i=0;i<npages;i++){ pg[i].addr=(char*)mem+(size_t)i*pgsz;
    pg[i].lkey=mesh_mem_lkey(&g_mem,pg[i].addr);
    if(!pg[i].lkey) die("page outside registered memory");
    pg[i].state=FREE; freelist[nfree++]=i; }

  int posted=0, sending=0;
  #define FREE_PAGE(i) do{ if(pg[i].held){} else if(pg[i].state!=FREE){ pg[i].state=FREE; \
      if(nfree<npages) freelist[nfree++]=(i); else fprintf(stderr,"BUG freelist overflow p%d\n",(i)); } \
    else fprintf(stderr,"BUG double free p%d\n",(i)); }while(0)
  int txwin=0; int *txring=NULL;
  int rx_target = gr_recv/frames/4; if(rx_target > npages/4) rx_target = npages/4;
  if(rx_target<2) rx_target=2;
  int tx_budget = gr_send/frames/4; if(tx_budget > npages/4) tx_budget = npages/4;
  if(tx_budget<1) tx_budget=1;
  txwin = npages - rx_target - 8; if(txwin<8) txwin=8;
  txring = malloc((size_t)txwin*sizeof(int)); if(!txring) die("alloc txring");
  for(int i=0;i<txwin;i++) txring[i]=-1;
  if(tx_budget > txwin/8) tx_budget = txwin/8; if(tx_budget<1) tx_budget=1;
  enum { MISSW = 1u<<18 };
  int horizon = txwin - 4096; if(horizon < txwin/2) horizon = txwin/2;
  int rearm_gap = tx_budget + 512;
  int retire_at = horizon + 4096; if(retire_at > MISSW/2) retire_at = MISSW/2;
  fprintf(stderr,"caps: granted send=%d recv=%d frames (4KB units); pgsz=%d frames/pg=%d "
                 "-> rx_target=%d (%d frames) tx_budget=%d (%d frames) txwin=%d\n",
          gr_send,gr_recv,pgsz,frames,rx_target,rx_target*frames,tx_budget,tx_budget*frames,txwin);
  fprintf(stderr,"repair: horizon=%d rearm_gap=%d retire_at=%d mrs=%d chunk=%zu regGB=%.3f\n",horizon,rearm_gap,retire_at,g_mem.nseg,g_mem.chunk,g_mem.bytes/1e9);
  unsigned long long rx=0, tx=0, bytes_rx=0, bytes_tx=0, drops_seen=0, short_pay=0;
  unsigned long long delivered=0, meta_seen=0;
  unsigned long long d_wc=0, d_ring=0, d_magic=0, d_post=0, d_bounds=0, seq_lo=0;
  unsigned long long repaired=0, nack_sent=0, nack_rx=0, resent=0;
  unsigned long long resend_fail=0, nack_entries=0, nack_stale=0;
  uint64_t *miss = calloc(MISSW/64,sizeof(uint64_t));
  if(!miss) die("alloc miss bitmap");
  uint32_t expected=0; int have_expected=0; uint32_t retire_cur=0;
  unsigned long long gone=0, rearm=0;
  uint32_t pend[4096]; int npend=0; unsigned long long pend_drop=0;
  unsigned long long spans_done=0, span_abort=0, span_pages=0; (void)span_abort;
  unsigned long long cmp_full=0, app_sent=0, app_recv=0;
  unsigned long long missing=0; uint32_t sweep_cur=0;
  uint32_t *req_at = calloc(MISSW,sizeof(uint32_t)); if(!req_at) die("alloc req_at");
  unsigned long long lat_b[7]={0}, lat_max=0, lat_sum=0, outstanding=0, out_max=0;
  unsigned long long nlag_sum=0, nlag_max=0, nlag_n=0;
  unsigned long long burst_b[7]={0}, burst_n=0; int posted_min=1<<30;
  enum { RTQ = 1u<<14 };
  uint32_t *rtq = malloc(RTQ*sizeof(uint32_t)); if(!rtq) die("alloc rtq");
  int rt_head=0, rt_n=0; unsigned long long rt_drop=0;
  unsigned long long age_b[7]={0};
  uint32_t next_seq=0, last_seq=0; int seen_any=0; int spanpos=0;
  struct wf w_free={0}, w_ready={0}, w_send={0};
  double t0=now(), t_last=t0, v_last=nfree, tel_last=t0, quiet=t0;

  while(!g_stop && now()-t0 < seconds){
    int did=0;

    
    while(nfree>0 && posted<rx_target){
      int i=freelist[--nfree];
      struct ibv_sge g={(uintptr_t)pg[i].addr,(uint32_t)pgsz,pg[i].lkey};
      struct ibv_recv_wr wr={.wr_id=(uint64_t)i,.sg_list=&g,.num_sge=1},*bad;
      if(ibv_post_recv(g_qp,&wr,&bad)){ freelist[nfree++]=i; break; }
      pg[i].state=POSTED; posted++; did=1;
    }

    // APPREL -- take back pages the application has finished with
    { struct mesh_desc dd;
      while(!mesh_pop(&RG,&RG.h->rrel,RG.h->rel_off,&dd)){
        uint32_t p=dd.page;
        if(p<(uint32_t)npages && pg[p].held){
          pg[p].held=0; pg[p].state=FREE; freelist[nfree++]=(int)p; did=1; } } }

    // APPFREE -- lend spare pages to the application, keeping the receive
    // pool above its target so the wire never starves behind the app
    if(atomic_load_explicit(&RG.h->client_pid,memory_order_acquire)){
      while(nfree > rx_target){
        int p=freelist[nfree-1];
        struct mesh_desc dd={.page=(uint32_t)p,.bytes=0,.seq=0,.node=0,.flags=0};
        if(mesh_push(&RG,&RG.h->rfree,RG.h->free_off,&dd)) break;
        nfree--; pg[p].held=1; did=1;
      }
      // APPSUB -- send what the application handed over
      struct mesh_desc dd;
      while(sending<tx_budget && !mesh_pop(&RG,&RG.h->rsub,RG.h->sub_off,&dd)){
        uint32_t p=dd.page; if(p>=(uint32_t)npages) continue;
        struct wire *wh=(struct wire*)pg[p].addr;
        wh->magic=WIRE_MAGIC; wh->path=0; wh->stream=0; wh->seq=next_seq++;
        wh->bytes=(uint16_t)(dd.bytes>maxpay?maxpay:dd.bytes);
        wh->src=(uint16_t)node_idx; wh->dst=dd.node;
        wh->flags=F_FIRST|F_LAST; wh->hops=0;
        struct ibv_sge g={(uintptr_t)pg[p].addr,(uint32_t)pgsz,pg[p].lkey};
        struct ibv_send_wr wr={.wr_id=(uint64_t)p,.sg_list=&g,.num_sge=1,
          .opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
        pg[p].held=0;
        if(ibv_post_send(g_qp,&wr,&bad)){ pg[p].state=FREE; freelist[nfree++]=(int)p; }
        else { pg[p].state=SENDING; if(!pg[p].refs++) sending++; app_sent++;
               atomic_fetch_add_explicit(&RG.h->sent,1,memory_order_relaxed); }
        did=1;
      }
    }

    
    if(source){
      int cap = inflight && inflight<tx_budget ? inflight : tx_budget;
      while(sending<cap && rt_n>0){
        uint32_t want=rtq[rt_head]; rt_head=(rt_head+1)%RTQ; rt_n--;
        uint32_t lg=next_seq-want; nlag_sum+=lg; nlag_n++; if(lg>nlag_max) nlag_max=lg;
        int tp=txring[want % (uint32_t)txwin];
        if(tp<0){ nack_stale++; continue; }
        struct wire *th=(struct wire*)pg[tp].addr;
        if(th->seq!=want){ nack_stale++; continue; }
        struct ibv_sge g={(uintptr_t)pg[tp].addr,(uint32_t)pgsz,pg[tp].lkey};
        struct ibv_send_wr wr={.wr_id=(uint64_t)tp,.sg_list=&g,.num_sge=1,
          .opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
        if(ibv_post_send(g_qp,&wr,&bad)){ resend_fail++; break; }
        if(!pg[tp].refs++) sending++; resent++; did=1;
        outstanding++; if(outstanding>out_max) out_max=outstanding;
      }
      while(sending<cap){
        uint32_t slot=next_seq % (uint32_t)txwin;
        int old=txring[slot];
        if(old>=0){
          if(pg[old].refs) break;
          txring[slot]=-1; pg[old].held=0; FREE_PAGE(old);
        }
        if(nfree<=0) break;
        int i=freelist[--nfree];
        struct wire *h=(struct wire*)pg[i].addr;
        h->magic=WIRE_MAGIC; h->path=route_path; h->stream=0; h->seq=next_seq++;
        h->bytes=(uint16_t)(pgsz-sizeof *h); h->src=(uint16_t)node_idx;
        h->dst=(uint16_t)target_idx; h->flags=0; if(spanpos==0) h->flags|=F_FIRST;
        if(spanpos==spanpg-1) h->flags|=F_LAST;
        spanpos=(spanpos+1)%spanpg; h->hops=0;
        struct ibv_sge g={(uintptr_t)pg[i].addr,(uint32_t)pgsz,pg[i].lkey};
        struct ibv_send_wr wr={.wr_id=(uint64_t)i,.sg_list=&g,.num_sge=1,
          .opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
        if(ibv_post_send(g_qp,&wr,&bad)){ freelist[nfree++]=i; next_seq--; break; }
        pg[i].state=SENDING; pg[i].refs=1; sending++; did=1;
        outstanding++; if(outstanding>out_max) out_max=outstanding;
        txring[h->seq % (uint32_t)txwin]=i; pg[i].held=1;
      }
    }

    
    struct ibv_wc wc[32];
    int k=ibv_poll_cq(g_cq,32,wc);
    if(k<0) die("poll_cq");
    for(int j=0;j<k;j++){
      int i=(int)wc[j].wr_id;
      if(i<0 || i>=npages){ d_bounds++; fprintf(stderr,"BUG wr_id %d out of pool\n",i); continue; }
      if(wc[j].status!=IBV_WC_SUCCESS){
        fprintf(stderr,"wc %s on page %d (state %d)\n",ibv_wc_status_str(wc[j].status),i,pg[i].state);
        d_wc++;
        if(pg[i].state==SENDING) sending--;
        if(pg[i].state==POSTED) posted--;
        FREE_PAGE(i); continue; }
      if(wc[j].opcode==IBV_WC_RECV){
        posted--; if(posted<posted_min) posted_min=posted; rx++; bytes_rx+=wc[j].byte_len;
        if(nready>=npages){ d_ring++; FREE_PAGE(i); continue; }
        pg[i].state=FILLED; ready[(rhead+nready)%npages]=i; nready++; did=1;
      } else {
        tx++; bytes_tx+=pgsz; if(outstanding) outstanding--;
        if(pg[i].refs && --pg[i].refs==0){
          sending--;
          if(!pg[i].held) FREE_PAGE(i); else pg[i].state=SENDING;
        }
        did=1;
      }
    }

    
    while(nready>0){
      int i=ready[rhead]; rhead=(rhead+1)%npages; nready--;
      struct wire *h=(struct wire*)pg[i].addr;
      if(h->magic!=WIRE_MAGIC){ d_magic++; FREE_PAGE(i); continue; }
      if(h->flags & F_NACK){
        nack_rx++;
        uint32_t *rq=(uint32_t*)((char*)pg[i].addr+sizeof *h);
        uint32_t nq=h->bytes/4; if(nq>1000) nq=1000; nack_entries+=nq;
        for(uint32_t q=0;q<nq;q++){
          if(rt_n>=RTQ){ rt_drop++; continue; }
          rtq[(rt_head+rt_n)%RTQ]=rq[q]; rt_n++;
        }
        FREE_PAGE(i); continue;
      }
      if(!have_expected){ expected=h->seq; retire_cur=h->seq; sweep_cur=h->seq; have_expected=1; }
      if(h->seq==expected) expected++;
      else if((int32_t)(h->seq-expected)>0){
        { uint32_t bs=h->seq-expected; burst_n++;
          burst_b[bs<2?0:bs<4?1:bs<16?2:bs<64?3:bs<256?4:bs<1024?5:6]++; }
        for(uint32_t m=expected;m!=h->seq;m++){ miss[(m%MISSW)/64] |= 1ull<<((m%MISSW)%64); drops_seen++;
          req_at[m%MISSW]=h->seq;
          missing++; if(npend<4096) pend[npend++]=m; else pend_drop++; }
        expected=h->seq+1;
      } else {
        uint64_t *w=&miss[(h->seq%MISSW)/64]; uint64_t b=1ull<<((h->seq%MISSW)%64);
        if(*w & b){ *w &= ~b; repaired++; if(missing) missing--; if(drops_seen) drops_seen--;
          uint32_t lat = expected - req_at[h->seq%MISSW];
          if(lat>lat_max) lat_max=lat; lat_sum+=lat;
          lat_b[lat<128?0:lat<512?1:lat<2048?2:lat<8192?3:lat<32768?4:lat<131072?5:6]++;
        } else seq_lo++;
      }
      last_seq=h->seq; seen_any=1;
      for(int r=0; r<128 && (uint32_t)(expected-retire_cur) > (uint32_t)retire_at; r++){
        uint32_t z=retire_cur%MISSW; uint64_t b=1ull<<(z%64);
        if(miss[z/64] & b){ miss[z/64] &= ~b; gone++; if(missing) missing--; }
        retire_cur++; }
      uint32_t pay = h->bytes > maxpay ? maxpay : h->bytes;
      if(pay != h->bytes) short_pay++;
      if(h->flags & F_META) meta_seen++;

      h->hops++;
      if(h->dst==(uint16_t)node_idx || h->hops>32){
        delivered++; spans_done++; span_pages++;
        if(atomic_load_explicit(&RG.h->client_pid,memory_order_acquire)){
          struct mesh_desc dd={.page=(uint32_t)i,.bytes=pay,.seq=h->seq,
                               .node=h->src,.flags=h->flags};
          if(mesh_push(&RG,&RG.h->rcmp,RG.h->cmp_off,&dd)){ cmp_full++; FREE_PAGE(i); }
          else { pg[i].held=1; app_recv++;
                 atomic_fetch_add_explicit(&RG.h->recvd,1,memory_order_relaxed); }
        } else FREE_PAGE(i);
      } else {
        struct ibv_sge g={(uintptr_t)pg[i].addr,(uint32_t)pgsz,pg[i].lkey};
        struct ibv_send_wr wr={.wr_id=(uint64_t)i,.sg_list=&g,.num_sge=1,
          .opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
        if(ibv_post_send(g_qp,&wr,&bad)){ d_post++; FREE_PAGE(i); }
        else { pg[i].state=SENDING; if(!pg[i].refs++) sending++; }
      }
      did=1;
    }

    
    if(missing && npend<4032 && rearm_gap < horizon){
      uint32_t swlo = expected - (uint32_t)horizon;
      if((uint32_t)(swlo-retire_cur) > (uint32_t)(expected-retire_cur)) swlo = retire_cur;
      for(int sw=0; sw<8; sw++){
        if((uint32_t)(sweep_cur-swlo) >= (uint32_t)(expected-swlo)) sweep_cur = swlo & ~63u;
        uint32_t z=sweep_cur%MISSW; uint64_t w=miss[z/64];
        while(w){
          int b=__builtin_ctzll(w); w &= w-1;
          uint32_t sq=sweep_cur+(uint32_t)b;
          if((uint32_t)(sq-swlo) >= (uint32_t)(expected-swlo)) continue;
          if((uint32_t)(expected-req_at[sq%MISSW]) < (uint32_t)rearm_gap) continue;
          req_at[sq%MISSW]=expected; pend[npend++]=sq; rearm++;
          if(npend>=4032) break; }
        sweep_cur += 64; } }

    if(npend>0 && nfree>0){
      {
        int nq = npend>500?500:npend;
        int slot=freelist[--nfree];
        struct wire *nh=(struct wire*)pg[slot].addr;
        uint32_t *q=(uint32_t*)((char*)pg[slot].addr+sizeof *nh);
        for(int z=0;z<nq;z++) q[z]=pend[z];
        nh->magic=WIRE_MAGIC; nh->path=0; nh->stream=0; nh->seq=0;
        nh->bytes=(uint16_t)(nq*4); nh->src=(uint16_t)node_idx; nh->dst=0;
        nh->flags=F_NACK; nh->hops=0;
        struct ibv_sge g={(uintptr_t)pg[slot].addr,(uint32_t)pgsz,pg[slot].lkey};
        struct ibv_send_wr wr={.wr_id=(uint64_t)slot,.sg_list=&g,.num_sge=1,
          .opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
        if(ibv_post_send(g_qp,&wr,&bad)){ FREE_PAGE(slot); }
        else { pg[slot].state=SENDING; pg[slot].refs=1; sending++; nack_sent++; outstanding++;
               npend-=nq; for(int z=0;z<npend;z++) pend[z]=pend[z+nq]; }
      }
    }
    double t=now();
    if(t-tel_last >= 1.0/tel_hz){
      double dt=t-t_last, dv=(double)nfree - v_last;
      wf_add(&w_free,nfree); wf_add(&w_ready,nready); wf_add(&w_send,sending);
      fprintf(stderr,
        "tel t=%.1f free=%d posted=%d/%d ready=%d sending=%d dV/dt=%.1f/s dV/V=%.4f "
        "var_free=%.1f rx=%llu tx=%llu gaps=%llu clamped=%llu deliv=%llu meta=%llu sum=%d/%d out=%llu\n",
        t-t0, nfree, posted, rx_target, nready, sending, dv/(dt>0?dt:1),
        nfree? dv/nfree : 0.0, wf_var(&w_free), rx, tx, drops_seen, short_pay, delivered, meta_seen, nfree+posted+nready+sending, npages, outstanding);
      t_last=t; v_last=nfree; tel_last=t;
    }
    if(!did){ if(t-quiet>tmo){ fprintf(stderr,"idle %ds\n",tmo); break; } usleep(50); }
    else quiet=t;
  }
  for(uint32_t z=0; z<MISSW; z++) if(miss[z/64] & (1ull<<(z%64))){
    uint32_t a = expected - req_at[z];
    age_b[a<128?0:a<512?1:a<2048?2:a<8192?3:a<32768?4:a<131072?5:6]++; }
  double el=now()-t0;
  printf("%-6s pages=%d pgsz=%d  rx=%llu tx=%llu  %.2f Gbit/s in  %.2f Gbit/s out  gaps=%llu clamped=%llu\n"
         "       drops: wc=%llu ring=%llu magic=%llu post=%llu bounds=%llu reorder=%llu  ours=%llu\n"
       "       repair: lost=%llu repaired=%llu unrecovered=%llu nack_tx=%llu nack_rx=%llu resent=%llu\n"
       "       nack: entries=%llu stale=%llu resend_fail=%llu pend_drop=%llu\n"
       "       gone=%llu rearm=%llu horizon=%d gap=%d retire_at=%d\n"
       "       lat(seq): max=%llu mean=%.0f b<128=%llu <512=%llu <2k=%llu <8k=%llu <32k=%llu <128k=%llu ge=%llu\n"
       "       resid age: <128=%llu <512=%llu <2k=%llu <8k=%llu <32k=%llu <128k=%llu ge=%llu  out_max=%llu\n"
       "       nacklag: n=%llu mean=%.0f max=%llu  posted_min=%d rt_drop=%llu\n"
       "       seqs: next_seq=%u expected=%u last=%u\n"
       "       burst: n=%llu 1=%llu 2-3=%llu 4-15=%llu 16-63=%llu 64-255=%llu 256-1k=%llu ge1k=%llu\n",
    source?"source":"hop", npages, pgsz, rx, tx,
    bytes_rx*8/el/1e9, bytes_tx*8/el/1e9, drops_seen, short_pay,
    d_wc,d_ring,d_magic,d_post,d_bounds,seq_lo, d_wc+d_ring+d_magic+d_post+d_bounds,
    repaired+drops_seen, repaired, drops_seen, nack_sent, nack_rx, resent,
    nack_entries, nack_stale, resend_fail, pend_drop,
    gone, rearm, horizon, rearm_gap, retire_at,
    lat_max, repaired? (double)lat_sum/repaired : 0.0,
    lat_b[0],lat_b[1],lat_b[2],lat_b[3],lat_b[4],lat_b[5],lat_b[6],
    age_b[0],age_b[1],age_b[2],age_b[3],age_b[4],age_b[5],age_b[6], out_max,
    nlag_n, nlag_n? (double)nlag_sum/nlag_n : 0.0, nlag_max, posted_min, rt_drop,
    next_seq, expected, last_seq,
    burst_n, burst_b[0],burst_b[1],burst_b[2],burst_b[3],burst_b[4],burst_b[5],burst_b[6]);
  return 0;
}
