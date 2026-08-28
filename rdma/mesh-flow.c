// mesh-flow -- see RDMA-FIRST.md
#include "mesh-f.h"
#include <infiniband/verbs.h>
#include <netdb.h>
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
struct page { char *addr; uint32_t lkey; uint32_t refs; uint8_t state; };

struct wf { double n, mean, m2; };
static void wf_add(struct wf *w, double x){
  w->n += 1; double d = x - w->mean; w->mean += d / w->n; w->m2 += d * (x - w->mean); }
static double wf_var(struct wf *w){ return w->n > 1 ? w->m2 / (w->n - 1) : 0.0; }

static struct ibv_qp *g_qp; static struct ibv_mr *g_mr; static struct ibv_pd *g_pd;
static struct ibv_cq *g_cq; static struct ibv_context *g_ctx;
static volatile sig_atomic_t g_stop;
static void on_sig(int s){ (void)s; g_stop = 1; }
static void teardown(void){
  if(g_qp){ ibv_destroy_qp(g_qp); g_qp=NULL; }
  if(g_cq){ ibv_destroy_cq(g_cq); g_cq=NULL; }
  if(g_mr){ ibv_dereg_mr(g_mr); g_mr=NULL; }
  if(g_pd){ ibv_dealloc_pd(g_pd); g_pd=NULL; }
  if(g_ctx){ ibv_close_device(g_ctx); g_ctx=NULL; }
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
  h.ai_flags=AI_PASSIVE; if(getaddrinfo(NULL,p,&h,&r)) die("getaddrinfo");
  int l=socket(r->ai_family,SOCK_STREAM,0),on=1;
  setsockopt(l,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on);
  if(bind(l,r->ai_addr,r->ai_addrlen)) die("bind");
  listen(l,1); freeaddrinfo(r);
  if(wait_fd(l,0,secs)<1){ close(l); die("no peer connected in time"); }
  int fd=accept(l,NULL,NULL); close(l); if(fd<0) die("accept");
  setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof tv);
  setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&tv,sizeof tv); return fd; }


int main(int argc,char**argv){
  const char *dev=NULL,*peer=NULL;
  int pgsz=4096, npages=240, port=18519, tmo=30, seconds=5, source=0, inflight=0;
  int node_idx=0, target_idx=1; unsigned route_path=0; unsigned egress=0; (void)egress;
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
    else if(!strcmp(argv[i],"--source")) source=1;
    else peer=argv[i];
  }
  atexit(teardown);
  struct sigaction sa={0}; sa.sa_handler=on_sig;
  sigaction(SIGINT,&sa,NULL); sigaction(SIGTERM,&sa,NULL); sigaction(SIGHUP,&sa,NULL);

  int frames = (pgsz+4095)/4096;
  if(frames*npages > 4095){ npages = 4095/frames; }
  if((size_t)pgsz*npages > 0xfa0000){ npages = 0xfa0000/pgsz; }
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

  size_t span=(size_t)pgsz*npages;
  void *mem; if(posix_memalign(&mem,getpagesize(),span)) die("memalign");
  memset(mem,0,span);
  g_mr=ibv_reg_mr(g_pd,mem,span,IBV_ACCESS_LOCAL_WRITE); if(!g_mr) die("reg_mr");
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
  for(int i=0;i<npages;i++){ pg[i].addr=(char*)mem+(size_t)i*pgsz; pg[i].lkey=g_mr->lkey;
    pg[i].state=FREE; freelist[nfree++]=i; }

  int posted=0, sending=0;
  #define FREE_PAGE(i) do{ if(pg[i].state!=FREE){ pg[i].state=FREE; \
      if(nfree<npages) freelist[nfree++]=(i); else fprintf(stderr,"BUG freelist overflow p%d\n",(i)); } \
    else fprintf(stderr,"BUG double free p%d\n",(i)); }while(0)
  int rx_target = npages*3/4; if(rx_target<2) rx_target=2;
  if(rx_target*frames > gr_recv) rx_target = gr_recv/frames;
  int tx_budget = gr_send/frames; if(tx_budget<1) tx_budget=1;
  if(tx_budget > npages/4) tx_budget = npages/4;
  fprintf(stderr,"caps: granted send=%d recv=%d frames (4KB units); pgsz=%d frames/pg=%d "
                 "-> rx_target=%d (%d frames) tx_budget=%d (%d frames)\n",
          gr_send,gr_recv,pgsz,frames,rx_target,rx_target*frames,tx_budget,tx_budget*frames);
  unsigned long long rx=0, tx=0, bytes_rx=0, bytes_tx=0, drops_seen=0, short_pay=0;
  unsigned long long delivered=0, meta_seen=0;
  uint32_t next_seq=0, last_seq=0; int seen_any=0;
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

    
    if(source){
      int cap = inflight && inflight<tx_budget ? inflight : tx_budget;
      while(nfree>0 && sending<cap){
        int i=freelist[--nfree];
        struct wire *h=(struct wire*)pg[i].addr;
        h->magic=WIRE_MAGIC; h->path=route_path; h->stream=0; h->seq=next_seq++;
        h->bytes=(uint16_t)(pgsz-sizeof *h); h->src=(uint16_t)node_idx;
        h->dst=(uint16_t)target_idx; h->flags=F_FIRST|F_LAST; h->hops=0;
        struct ibv_sge g={(uintptr_t)pg[i].addr,(uint32_t)pgsz,pg[i].lkey};
        struct ibv_send_wr wr={.wr_id=(uint64_t)i,.sg_list=&g,.num_sge=1,
          .opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
        if(ibv_post_send(g_qp,&wr,&bad)){ freelist[nfree++]=i; next_seq--; break; }
        pg[i].state=SENDING; pg[i].refs=1; sending++; did=1;
      }
    }

    
    struct ibv_wc wc[32];
    int k=ibv_poll_cq(g_cq,32,wc);
    if(k<0) die("poll_cq");
    for(int j=0;j<k;j++){
      int i=(int)wc[j].wr_id;
      if(i<0 || i>=npages){ fprintf(stderr,"BUG wr_id %d out of pool\n",i); continue; }
      if(wc[j].status!=IBV_WC_SUCCESS){
        fprintf(stderr,"wc %s on page %d (state %d)\n",ibv_wc_status_str(wc[j].status),i,pg[i].state);
        if(pg[i].state==SENDING) sending--;
        if(pg[i].state==POSTED) posted--;
        FREE_PAGE(i); continue; }
      if(wc[j].opcode==IBV_WC_RECV){
        posted--; rx++; bytes_rx+=wc[j].byte_len;
        if(nready>=npages){ fprintf(stderr,"BUG ready ring full\n"); FREE_PAGE(i); continue; }
        pg[i].state=FILLED; ready[(rhead+nready)%npages]=i; nready++; did=1;
      } else {
        tx++; bytes_tx+=pgsz;
        if(pg[i].refs && --pg[i].refs==0){ sending--; FREE_PAGE(i); }
        did=1;
      }
    }

    
    while(nready>0){
      int i=ready[rhead]; rhead=(rhead+1)%npages; nready--;
      struct wire *h=(struct wire*)pg[i].addr;
      if(h->magic!=WIRE_MAGIC){ FREE_PAGE(i); continue; }
      if(seen_any){ uint32_t d = h->seq - last_seq; if(d>1) drops_seen += d-1; }
      last_seq=h->seq; seen_any=1;
      uint32_t pay = h->bytes > maxpay ? maxpay : h->bytes;
      if(pay != h->bytes) short_pay++;
      if(h->flags & F_META) meta_seen++;
      mesh_f((char*)pg[i].addr+sizeof *h, pay, h, node_idx);
      h->hops++;
      if(h->dst==(uint16_t)node_idx || h->hops>32){ delivered++; FREE_PAGE(i); }
      else {
        egress = h->path & HOP_MASK;
        h->path >>= HOP_BITS;
        struct ibv_sge g={(uintptr_t)pg[i].addr,(uint32_t)pgsz,pg[i].lkey};
        struct ibv_send_wr wr={.wr_id=(uint64_t)i,.sg_list=&g,.num_sge=1,
          .opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
        if(ibv_post_send(g_qp,&wr,&bad)){ FREE_PAGE(i); }
        else { pg[i].state=SENDING; pg[i].refs=1; sending++; }
      }
      did=1;
    }

    
    double t=now();
    if(t-tel_last >= 1.0/tel_hz){
      double dt=t-t_last, dv=(double)nfree - v_last;
      wf_add(&w_free,nfree); wf_add(&w_ready,nready); wf_add(&w_send,sending);
      fprintf(stderr,
        "tel t=%.1f free=%d posted=%d/%d ready=%d sending=%d dV/dt=%.1f/s dV/V=%.4f "
        "var_free=%.1f rx=%llu tx=%llu gaps=%llu clamped=%llu deliv=%llu meta=%llu sum=%d/%d\n",
        t-t0, nfree, posted, rx_target, nready, sending, dv/(dt>0?dt:1),
        nfree? dv/nfree : 0.0, wf_var(&w_free), rx, tx, drops_seen, short_pay, delivered, meta_seen, nfree+posted+nready+sending, npages);
      t_last=t; v_last=nfree; tel_last=t;
    }
    if(!did){ if(t-quiet>tmo){ fprintf(stderr,"idle %ds\n",tmo); break; } usleep(50); }
    else quiet=t;
  }
  double el=now()-t0;
  printf("%-6s pages=%d pgsz=%d  rx=%llu tx=%llu  %.2f Gbit/s in  %.2f Gbit/s out  gaps=%llu clamped=%llu\n",
    source?"source":"hop", npages, pgsz, rx, tx,
    bytes_rx*8/el/1e9, bytes_tx*8/el/1e9, drops_seen, short_pay);
  return 0;
}
