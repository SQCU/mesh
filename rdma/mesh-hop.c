// A hop. Receives records, applies f, forwards them. Nothing blocks on any one
// record: K slots are in flight, the loop is driven by completions, and a slot is
// re-armed the moment its send completes.
//
// max_sge is 1 on this hardware, so a record cannot be split across scatter-gather
// entries. Records are laid out contiguously as [hdr][payload] and forwarded with a
// single SGE at an offset; the header is read and rewritten in place. The payload is
// never copied.
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

struct hdr { uint32_t seq; uint32_t dst; uint32_t len; uint32_t hops; };

typedef void (*fn)(float*, int);
static void f_add (float*p,int n){ for(int i=0;i<n;i++) p[i]+=1.0f; }
static void f_copy(float*p,int n){ (void)p;(void)n; }

static struct ibv_qp *g_qp; static struct ibv_mr *g_mr; static struct ibv_pd *g_pd;
static struct ibv_cq *g_cq; static struct ibv_context *g_ctx;
static volatile sig_atomic_t g_stop;
static void on_sig(int s){ (void)s; g_stop=1; }
static void teardown(void){
  if(g_qp){ ibv_destroy_qp(g_qp); g_qp=NULL; }
  if(g_cq){ ibv_destroy_cq(g_cq); g_cq=NULL; }
  if(g_mr){ ibv_dereg_mr(g_mr); g_mr=NULL; }
  if(g_pd){ ibv_dealloc_pd(g_pd); g_pd=NULL; }
  if(g_ctx){ ibv_close_device(g_ctx); g_ctx=NULL; }
}
static int die(const char*m){ fprintf(stderr,"%s\n",m); exit(1); }

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
  setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&tv,sizeof tv); return fd;
}

int main(int argc,char**argv){
  const char *dev=NULL,*peer=NULL,*op="add";
  int rec=65536, depth=32, total=4000, port=18518, tmo=30;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-d")) dev=argv[++i];
    else if(!strcmp(argv[i],"-s")) rec=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-k")) depth=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-N")) total=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-p")) port=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-t")) tmo=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-o")) op=argv[++i];
    else peer=argv[i];
  }
  fn f = strcmp(op,"copy") ? f_add : f_copy;
  atexit(teardown);
  struct sigaction sa={0}; sa.sa_handler=on_sig;
  sigaction(SIGINT,&sa,NULL); sigaction(SIGTERM,&sa,NULL); sigaction(SIGHUP,&sa,NULL);

  int frames_per = (rec+4095)/4096;
  if(frames_per*depth > 4095){ depth = 4095/frames_per; if(depth<1) die("record too large"); }

  struct ibv_device **dl=ibv_get_device_list(NULL); if(!dl) die("no devices");
  struct ibv_device *d=NULL;
  for(int i=0;dl[i];i++) if(!dev||!strcmp(ibv_get_device_name(dl[i]),dev)){ d=dl[i]; break; }
  if(!d) die("device not found");
  g_ctx=ibv_open_device(d); if(!g_ctx) die("open_device");
  struct ibv_port_attr pa; if(ibv_query_port(g_ctx,1,&pa)) die("query_port");
  if(pa.state!=IBV_PORT_ACTIVE){
    fprintf(stderr,"%s is %s -- no cable, or far end not RDMA-enabled\n",
      ibv_get_device_name(d), ibv_port_state_str(pa.state)); return 2; }
  g_pd=ibv_alloc_pd(g_ctx); if(!g_pd) die("alloc_pd");

  size_t span=(size_t)rec*depth*2;
  void *mem; if(posix_memalign(&mem,getpagesize(),span)) die("memalign");
  memset(mem,0,span);
  g_mr=ibv_reg_mr(g_pd,mem,span,IBV_ACCESS_LOCAL_WRITE); if(!g_mr) die("reg_mr");
  g_cq=ibv_create_cq(g_ctx,4096,NULL,NULL,0); if(!g_cq) die("create_cq");
  struct ibv_qp_init_attr ia={.send_cq=g_cq,.recv_cq=g_cq,.qp_type=IBV_QPT_UC,
    .cap={.max_send_wr=4095,.max_recv_wr=4095,.max_send_sge=1,.max_recv_sge=1}};
  g_qp=ibv_create_qp(g_pd,&ia); if(!g_qp) die("create_qp");
  struct ibv_qp_attr at={.qp_state=IBV_QPS_INIT,.pkey_index=0,.port_num=1,.qp_access_flags=0};
  if(ibv_modify_qp(g_qp,&at,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)) die("INIT");

  union ibv_gid gid; ibv_query_gid(g_ctx,1,0,&gid);
  uint32_t psn=lrand48()&0xffffff;
  struct qpi me={g_qp->qp_num,psn,pa.lid},you; memcpy(me.gid,&gid,16);
  int fd=oob(peer,port,tmo);
  if(peer){ write(fd,&me,sizeof me); if(read(fd,&you,sizeof you)!=sizeof you) die("oob"); }
  else    { if(read(fd,&you,sizeof you)!=sizeof you) die("oob"); write(fd,&me,sizeof me); }
  close(fd);
  struct ibv_qp_attr r={.qp_state=IBV_QPS_RTR,.path_mtu=IBV_MTU_4096,.rq_psn=you.psn,
    .dest_qp_num=you.qpn,.ah_attr={.dlid=you.lid,.port_num=1,.is_global=1,
    .grh={.hop_limit=1,.sgid_index=0}}};
  memcpy(&r.ah_attr.grh.dgid,you.gid,16);
  if(ibv_modify_qp(g_qp,&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN)) die("RTR");
  struct ibv_qp_attr st={.qp_state=IBV_QPS_RTS,.sq_psn=psn};
  if(ibv_modify_qp(g_qp,&st,IBV_QP_STATE|IBV_QP_SQ_PSN)) die("RTS");

  int head = peer?1:0;
  int payload_n = (rec-(int)sizeof(struct hdr))/(int)sizeof(float);
  #define SLOT(i) ((char*)mem + (size_t)(i)*rec)
  struct ibv_recv_wr *brw; struct ibv_send_wr *bsw;
  #define POST_RECV(i) do{ struct ibv_sge g={(uintptr_t)SLOT(i),(uint32_t)rec,g_mr->lkey}; \
    struct ibv_recv_wr w={.wr_id=(uint64_t)(i),.sg_list=&g,.num_sge=1}; \
    if(ibv_post_recv(g_qp,&w,&brw)) die("post_recv"); }while(0)
  #define POST_SEND(i) do{ struct ibv_sge g={(uintptr_t)SLOT(i),(uint32_t)rec,g_mr->lkey}; \
    struct ibv_send_wr w={.wr_id=(uint64_t)(i),.sg_list=&g,.num_sge=1, \
      .opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED}; \
    if(ibv_post_send(g_qp,&w,&bsw)) die("post_send"); }while(0)

  for(int i=0;i<depth;i++) POST_RECV(i);
  if(getenv("D")) fprintf(stderr,"posted %d recvs, rec=%d frames/rec=%d span=%zu\n",depth,rec,frames_per,span);
  int sent=0,done=0;
  struct timeval t0,t1,now; gettimeofday(&t0,NULL);
  if(head) for(int i=depth;i<2*depth && sent<total;i++,sent++){
    struct hdr *h=(struct hdr*)SLOT(i); h->seq=sent; h->dst=0; h->len=rec; h->hops=0;
    POST_SEND(i);
  }
  if(getenv("D")) fprintf(stderr,"head posted %d sends\n",sent);
  struct ibv_wc wc[32];
  long long last=0;
  while(!g_stop && done<total){
    int k=ibv_poll_cq(g_cq,32,wc);
    if(k<0) die("poll_cq");
    if(k==0){
      gettimeofday(&now,NULL);
      long long el=(now.tv_sec-t0.tv_sec);
      if(el-last>tmo){ fprintf(stderr,"stalled at %d/%d\n",done,total); return 1; }
      continue;
    }
    gettimeofday(&now,NULL); last=now.tv_sec-t0.tv_sec;
    if(getenv("D")) fprintf(stderr,"poll k=%d op0=%d wr0=%llu\n",k,(int)wc[0].opcode,(unsigned long long)wc[0].wr_id);
    for(int j=0;j<k;j++){
      if(wc[j].status!=IBV_WC_SUCCESS){ fprintf(stderr,"wc: %s\n",ibv_wc_status_str(wc[j].status)); return 1; }
      int i=(int)wc[j].wr_id;
      if(wc[j].opcode==IBV_WC_RECV){
        if(head){ done++; POST_RECV(i); }
        else{
          struct hdr *h=(struct hdr*)SLOT(i);
          h->hops++;
          f((float*)(SLOT(i)+sizeof *h), payload_n);
          POST_SEND(i);
        }
      } else {
        if(head){
          if(sent<total){ struct hdr *h=(struct hdr*)SLOT(i);
            h->seq=sent++; h->dst=0; h->len=rec; h->hops=0; POST_SEND(i); }
        } else { done++; POST_RECV(i); }
      }
    }
  }
  gettimeofday(&t1,NULL);
  double sec=(t1.tv_sec-t0.tv_sec)+(t1.tv_usec-t0.tv_usec)/1e6;
  printf("%s op=%s rec=%d depth=%d frames/rec=%d done=%d  %.1f Gbit/s  %.1f us/rec\n",
    head?"head":"hop", op, rec, depth, frames_per, done,
    (double)rec*done*8/sec/1e9, sec*1e6/(done?done:1));
  return 0;
}
