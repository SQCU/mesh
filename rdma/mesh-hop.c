// One hop: receive a buffer, apply f, send it on.
//   f = copy  -> relay        f = add -> reduce
// The schedule (who talks to whom, in what order) is not this program's problem.
#include <infiniband/verbs.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <signal.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/select.h>

static struct ibv_qp *g_qp; static struct ibv_mr *g_mr; static struct ibv_pd *g_pd;
static struct ibv_cq *g_cq; static struct ibv_context *g_ctx;
static volatile sig_atomic_t g_stop;
static void on_signal(int s){ (void)s; g_stop = 1; }
static void teardown(void){
  if(g_qp){ ibv_destroy_qp(g_qp); g_qp=NULL; }
  if(g_cq){ ibv_destroy_cq(g_cq); g_cq=NULL; }
  if(g_mr){ ibv_dereg_mr(g_mr); g_mr=NULL; }
  if(g_pd){ ibv_dealloc_pd(g_pd); g_pd=NULL; }
  if(g_ctx){ ibv_close_device(g_ctx); g_ctx=NULL; }
}

typedef void (*fn)(float *dst, const float *in, int n);
static void f_add (float *d, const float *s, int n){ for(int i=0;i<n;i++) d[i]+=s[i]; }
static void f_copy(float *d, const float *s, int n){ memcpy(d,s,(size_t)n*sizeof(float)); }

struct qpi { uint32_t qpn, psn; uint16_t lid; uint8_t gid[16]; };
static int die(const char*m){ fprintf(stderr,"%s\n",m); exit(1); }

static int wait_fd(int fd,int w,int secs){
  fd_set s; FD_ZERO(&s); FD_SET(fd,&s);
  struct timeval tv={secs,0};
  return select(fd+1, w?NULL:&s, w?&s:NULL, NULL, &tv);
}
static int oob(const char*host,int port,int secs){
  struct addrinfo h={.ai_family=AF_UNSPEC,.ai_socktype=SOCK_STREAM},*r; char p[16];
  struct timeval tv={secs,0};
  snprintf(p,sizeof p,"%d",port);
  if(host){
    if(getaddrinfo(host,p,&h,&r)) die("getaddrinfo");
    int fd=socket(r->ai_family,SOCK_STREAM,0);
    fcntl(fd,F_SETFL,O_NONBLOCK);
    if(connect(fd,r->ai_addr,r->ai_addrlen) && errno!=EINPROGRESS) die("connect");
    if(wait_fd(fd,1,secs)<1) die("connect timed out");
    int e=0; socklen_t el=sizeof e; getsockopt(fd,SOL_SOCKET,SO_ERROR,&e,&el);
    if(e) die("connect refused");
    fcntl(fd,F_SETFL,0);
    freeaddrinfo(r);
    setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof tv);
    setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&tv,sizeof tv);
    return fd;
  }
  h.ai_flags=AI_PASSIVE;
  if(getaddrinfo(NULL,p,&h,&r)) die("getaddrinfo");
  int l=socket(r->ai_family,SOCK_STREAM,0),on=1;
  setsockopt(l,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on);
  if(bind(l,r->ai_addr,r->ai_addrlen)) die("bind");
  listen(l,1); freeaddrinfo(r);
  if(wait_fd(l,0,secs)<1){ close(l); die("no peer connected in time"); }
  int fd=accept(l,NULL,NULL); close(l);
  if(fd<0) die("accept");
  setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof tv);
  setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&tv,sizeof tv);
  return fd;
}

static int wait_cq(struct ibv_cq *cq, int n, int secs){
  struct ibv_wc wc; struct timeval a,b; gettimeofday(&a,NULL);
  while(n>0){
    if(g_stop){ fprintf(stderr,"interrupted\n"); return -1; }
    int k=ibv_poll_cq(cq,1,&wc);
    if(k<0){ fprintf(stderr,"poll_cq failed\n"); return -1; }
    if(k){ if(wc.status!=IBV_WC_SUCCESS){ fprintf(stderr,"wc: %s\n",ibv_wc_status_str(wc.status)); return -1; } n-=k; continue; }
    gettimeofday(&b,NULL);
    if((b.tv_sec-a.tv_sec) > secs){ fprintf(stderr,"timed out after %ds waiting on peer\n",secs); return -1; }
    usleep(200);
  }
  return 0;
}

int main(int argc,char**argv){
  const char *dev=NULL,*peer=NULL,*op="add"; int n=1<<20,port=18517,iters=1,tmo=30;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-d")) dev=argv[++i]; else if(!strcmp(argv[i],"-n")) n=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-p")) port=atoi(argv[++i]); else if(!strcmp(argv[i],"-i")) iters=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-o")) op=argv[++i];
    else if(!strcmp(argv[i],"-t")) tmo=atoi(argv[++i]); else peer=argv[i];
  }
  fn f = strcmp(op,"copy") ? f_add : f_copy;
  atexit(teardown);
  struct sigaction sa={0}; sa.sa_handler=on_signal;
  sigaction(SIGINT,&sa,NULL); sigaction(SIGTERM,&sa,NULL); sigaction(SIGHUP,&sa,NULL);

  struct ibv_device **dl=ibv_get_device_list(NULL); if(!dl) die("no rdma devices");
  struct ibv_device *d=NULL;
  for(int i=0;dl[i];i++) if(!dev||!strcmp(ibv_get_device_name(dl[i]),dev)){ d=dl[i]; break; }
  if(!d) die("device not found");
  struct ibv_context *c=ibv_open_device(d); if(!c) die("open_device"); g_ctx=c;

  // check the wire before blaming the software
  struct ibv_port_attr pa;
  if(ibv_query_port(c,1,&pa)) die("query_port");
  if(pa.state!=IBV_PORT_ACTIVE){
    fprintf(stderr,"%s is %s -- no cable, or the far end is not RDMA-enabled\n",
            ibv_get_device_name(d), ibv_port_state_str(pa.state)); return 2; }

  struct ibv_pd *pd=ibv_alloc_pd(c); if(!pd) die("alloc_pd"); g_pd=pd;
  size_t bytes=(size_t)n*sizeof(float);
  void *mem; if(posix_memalign(&mem,getpagesize(),bytes*2)) die("memalign");
  float *mine=mem, *in=(float*)((char*)mem+bytes);
  struct ibv_mr *mr=ibv_reg_mr(pd,mem,bytes*2,IBV_ACCESS_LOCAL_WRITE); if(!mr) die("reg_mr"); g_mr=mr;
  struct ibv_cq *cq=ibv_create_cq(c,16,NULL,NULL,0); if(!cq) die("create_cq"); g_cq=cq;
  struct ibv_qp_init_attr ia={.send_cq=cq,.recv_cq=cq,.qp_type=IBV_QPT_UC,
    .cap={.max_send_wr=4095,.max_recv_wr=4095,.max_send_sge=1,.max_recv_sge=1}};
  struct ibv_qp *qp=ibv_create_qp(pd,&ia); if(!qp) die("create_qp"); g_qp=qp;
  struct ibv_qp_attr at={.qp_state=IBV_QPS_INIT,.pkey_index=0,.port_num=1,.qp_access_flags=0};
  if(ibv_modify_qp(qp,&at,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)) die("INIT");

  union ibv_gid g; ibv_query_gid(c,1,0,&g);
  uint32_t psn=lrand48()&0xffffff;
  struct qpi me={qp->qp_num,psn,pa.lid},you; memcpy(me.gid,&g,16);
  int fd=oob(peer,port,tmo);
  if(peer){ write(fd,&me,sizeof me); if(read(fd,&you,sizeof you)!=sizeof you) die("oob"); }
  else    { if(read(fd,&you,sizeof you)!=sizeof you) die("oob"); write(fd,&me,sizeof me); }
  close(fd);
  struct ibv_qp_attr r={.qp_state=IBV_QPS_RTR,.path_mtu=IBV_MTU_4096,.rq_psn=you.psn,
    .dest_qp_num=you.qpn,.ah_attr={.dlid=you.lid,.port_num=1,.is_global=1,
    .grh={.hop_limit=1,.sgid_index=0}}};
  memcpy(&r.ah_attr.grh.dgid,you.gid,16);
  if(ibv_modify_qp(qp,&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN)) die("RTR");
  struct ibv_qp_attr s={.qp_state=IBV_QPS_RTS,.sq_psn=psn};
  if(ibv_modify_qp(qp,&s,IBV_QP_STATE|IBV_QP_SQ_PSN)) die("RTS");

  int head = peer ? 1 : 0;                 // the connector starts the buffer moving
  for(int i=0;i<n;i++) mine[i] = head ? 1.0f : 2.0f;
  struct ibv_sge rg={(uintptr_t)in,(uint32_t)bytes,mr->lkey};
  struct ibv_recv_wr rw={.sg_list=&rg,.num_sge=1},*brw;
  struct ibv_sge sg={(uintptr_t)mine,(uint32_t)bytes,mr->lkey};
  struct ibv_send_wr sw={.sg_list=&sg,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bsw;
  struct timeval t0,t1; gettimeofday(&t0,NULL);
  for(int it=0; it<iters && !g_stop; it++){
    if(ibv_post_recv(qp,&rw,&brw)) die("post_recv");
    if(head && ibv_post_send(qp,&sw,&bsw)) die("post_send");
    if(wait_cq(cq, head?2:1, tmo)) return 1;
    f(mine,in,n);
    if(!head){
      if(ibv_post_send(qp,&sw,&bsw)) die("post_send");
      if(wait_cq(cq,1,tmo)) return 1;
    }
  }
  gettimeofday(&t1,NULL);
  double sec=(t1.tv_sec-t0.tv_sec)+(t1.tv_usec-t0.tv_usec)/1e6;
  printf("%s  op=%s n=%d  result[0]=%.1f result[%d]=%.1f  %.1f Mbit/s  %.1f us/hop\n",
         head?"head":"hop", op, n, mine[0], n-1, mine[n-1],
         (double)bytes*8*iters/sec/1e6, sec*1e6/iters);
  return 0;
}
