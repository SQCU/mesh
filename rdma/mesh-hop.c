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

typedef void (*fn)(float *dst, const float *in, int n);
static void f_add (float *d, const float *s, int n){ for(int i=0;i<n;i++) d[i]+=s[i]; }
static void f_copy(float *d, const float *s, int n){ memcpy(d,s,(size_t)n*sizeof(float)); }

struct qpi { uint32_t qpn, psn; uint16_t lid; uint8_t gid[16]; };
static int die(const char*m){ fprintf(stderr,"%s\n",m); exit(1); }

static int oob(const char*host,int port){
  struct addrinfo h={.ai_family=AF_UNSPEC,.ai_socktype=SOCK_STREAM},*r; char p[16];
  snprintf(p,sizeof p,"%d",port);
  if(host){ if(getaddrinfo(host,p,&h,&r)) die("getaddrinfo");
    int fd=socket(r->ai_family,SOCK_STREAM,0);
    if(connect(fd,r->ai_addr,r->ai_addrlen)) die("connect"); freeaddrinfo(r); return fd; }
  h.ai_flags=AI_PASSIVE; if(getaddrinfo(NULL,p,&h,&r)) die("getaddrinfo");
  int l=socket(r->ai_family,SOCK_STREAM,0),on=1;
  setsockopt(l,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on);
  if(bind(l,r->ai_addr,r->ai_addrlen)) die("bind");
  listen(l,1); int fd=accept(l,NULL,NULL); close(l); freeaddrinfo(r); return fd;
}

int main(int argc,char**argv){
  const char *dev=NULL,*peer=NULL,*op="add"; int n=1<<20,port=18517,iters=1;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-d")) dev=argv[++i]; else if(!strcmp(argv[i],"-n")) n=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-p")) port=atoi(argv[++i]); else if(!strcmp(argv[i],"-i")) iters=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-o")) op=argv[++i]; else peer=argv[i];
  }
  fn f = strcmp(op,"copy") ? f_add : f_copy;

  struct ibv_device **dl=ibv_get_device_list(NULL); if(!dl) die("no rdma devices");
  struct ibv_device *d=NULL;
  for(int i=0;dl[i];i++) if(!dev||!strcmp(ibv_get_device_name(dl[i]),dev)){ d=dl[i]; break; }
  if(!d) die("device not found");
  struct ibv_context *c=ibv_open_device(d); if(!c) die("open_device");

  // check the wire before blaming the software
  struct ibv_port_attr pa;
  if(ibv_query_port(c,1,&pa)) die("query_port");
  if(pa.state!=IBV_PORT_ACTIVE){
    fprintf(stderr,"%s is %s -- no cable, or the far end is not RDMA-enabled\n",
            ibv_get_device_name(d), ibv_port_state_str(pa.state)); return 2; }

  struct ibv_pd *pd=ibv_alloc_pd(c); if(!pd) die("alloc_pd");
  size_t bytes=(size_t)n*sizeof(float);
  void *mem; if(posix_memalign(&mem,getpagesize(),bytes*2)) die("memalign");
  float *mine=mem, *in=(float*)((char*)mem+bytes);
  struct ibv_mr *mr=ibv_reg_mr(pd,mem,bytes*2,IBV_ACCESS_LOCAL_WRITE); if(!mr) die("reg_mr");
  struct ibv_cq *cq=ibv_create_cq(c,16,NULL,NULL,0); if(!cq) die("create_cq");
  struct ibv_qp_init_attr ia={.send_cq=cq,.recv_cq=cq,.qp_type=IBV_QPT_UC,
    .cap={.max_send_wr=4095,.max_recv_wr=4095,.max_send_sge=1,.max_recv_sge=1}};
  struct ibv_qp *qp=ibv_create_qp(pd,&ia); if(!qp) die("create_qp");
  struct ibv_qp_attr at={.qp_state=IBV_QPS_INIT,.pkey_index=0,.port_num=1,.qp_access_flags=0};
  if(ibv_modify_qp(qp,&at,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)) die("INIT");

  union ibv_gid g; ibv_query_gid(c,1,0,&g);
  uint32_t psn=lrand48()&0xffffff;
  struct qpi me={qp->qp_num,psn,pa.lid},you; memcpy(me.gid,&g,16);
  int fd=oob(peer,port);
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
  struct ibv_wc wc; struct timeval t0,t1; gettimeofday(&t0,NULL);
  for(int it=0; it<iters; it++){
    if(ibv_post_recv(qp,&rw,&brw)) die("post_recv");
    if(head){ if(ibv_post_send(qp,&sw,&bsw)) die("post_send"); }
    int got=0; while(got<(head?2:1)){ int k=ibv_poll_cq(cq,1,&wc); if(k<0) die("poll");
      if(k){ if(wc.status!=IBV_WC_SUCCESS) die(ibv_wc_status_str(wc.status)); got+=k; } }
    f(mine,in,n);                                    // <-- the hop's work
    if(!head){ if(ibv_post_send(qp,&sw,&bsw)) die("post_send");
      while(ibv_poll_cq(cq,1,&wc)<1); if(wc.status!=IBV_WC_SUCCESS) die(ibv_wc_status_str(wc.status)); }
  }
  gettimeofday(&t1,NULL);
  double sec=(t1.tv_sec-t0.tv_sec)+(t1.tv_usec-t0.tv_usec)/1e6;
  printf("%s  op=%s n=%d  result[0]=%.1f result[%d]=%.1f  %.1f Mbit/s  %.1f us/hop\n",
         head?"head":"hop", op, n, mine[0], n-1, mine[n-1],
         (double)bytes*8*iters/sec/1e6, sec*1e6/iters);
  return 0;
}
