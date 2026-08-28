// Ring all-reduce over RDMA/Thunderbolt.
// Proves the hop does work: reduce-scatter adds into the running sum before
// forwarding. No checksums, no acks, no validation passes -- causal attribution
// for a bad result belongs to whatever runs above this, not to ops spent here.
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>

struct qpinfo { uint32_t qpn, psn; uint16_t lid; uint8_t gid[16]; };

static int die(const char *m){ fprintf(stderr, "%s\n", m); exit(1); }

static struct ibv_qp *mkqp(struct ibv_pd *pd, struct ibv_cq *cq, int depth){
  struct ibv_qp_init_attr a = { .send_cq=cq, .recv_cq=cq, .qp_type=IBV_QPT_UC,
    .cap={ .max_send_wr=depth, .max_recv_wr=depth, .max_send_sge=1, .max_recv_sge=1 } };
  struct ibv_qp *q = ibv_create_qp(pd, &a);
  if(!q) die("ibv_create_qp");
  struct ibv_qp_attr i = { .qp_state=IBV_QPS_INIT, .pkey_index=0, .port_num=1, .qp_access_flags=0 };
  if(ibv_modify_qp(q,&i,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)) die("qp->INIT");
  return q;
}

static void connect_qp(struct ibv_qp *q, struct qpinfo *rem, uint32_t my_psn){
  struct ibv_qp_attr r = { .qp_state=IBV_QPS_RTR, .path_mtu=IBV_MTU_4096,
    .rq_psn=rem->psn, .dest_qp_num=rem->qpn,
    .ah_attr={ .dlid=rem->lid, .sl=0, .src_path_bits=0, .port_num=1, .is_global=1,
               .grh={ .hop_limit=1, .sgid_index=0 } } };
  memcpy(&r.ah_attr.grh.dgid, rem->gid, 16);
  if(ibv_modify_qp(q,&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN))
    die("qp->RTR");
  struct ibv_qp_attr s = { .qp_state=IBV_QPS_RTS, .sq_psn=my_psn };
  if(ibv_modify_qp(q,&s,IBV_QP_STATE|IBV_QP_SQ_PSN)) die("qp->RTS");
}

static int oob(const char *host, int port){
  int fd; struct addrinfo hints={.ai_family=AF_UNSPEC,.ai_socktype=SOCK_STREAM}, *res;
  char p[16]; snprintf(p,sizeof p,"%d",port);
  if(host){
    if(getaddrinfo(host,p,&hints,&res)) die("getaddrinfo");
    fd=socket(res->ai_family,SOCK_STREAM,0);
    if(connect(fd,res->ai_addr,res->ai_addrlen)) die("connect");
    freeaddrinfo(res); return fd;
  }
  hints.ai_flags=AI_PASSIVE;
  if(getaddrinfo(NULL,p,&hints,&res)) die("getaddrinfo");
  int l=socket(res->ai_family,SOCK_STREAM,0), on=1;
  setsockopt(l,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on);
  if(bind(l,res->ai_addr,res->ai_addrlen)) die("bind");
  listen(l,1); fd=accept(l,NULL,NULL); close(l); freeaddrinfo(res); return fd;
}

static void xchg(int fd, void *out, void *in, size_t n, int server){
  if(server){ if(read(fd,in,n)!=(ssize_t)n) die("oob read"); write(fd,out,n); }
  else      { write(fd,out,n); if(read(fd,in,n)!=(ssize_t)n) die("oob read"); }
}

static void poll_n(struct ibv_cq *cq, int n){
  struct ibv_wc wc[8];
  while(n>0){
    int c = ibv_poll_cq(cq, 8, wc);
    if(c<0) die("poll_cq");
    for(int i=0;i<c;i++) if(wc[i].status!=IBV_WC_SUCCESS){
      fprintf(stderr,"wc status %d (%s)\n", wc[i].status, ibv_wc_status_str(wc[i].status)); exit(1); }
    n-=c;
  }
}

int main(int argc,char**argv){
  const char *dev=NULL,*peer=NULL; int rank=0,nranks=2,elems=1<<20,iters=1,port=18516;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-d")) dev=argv[++i];
    else if(!strcmp(argv[i],"-r")) rank=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-n")) nranks=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-e")) elems=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-i")) iters=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-p")) port=atoi(argv[++i]);
    else peer=argv[i];
  }
  if(elems % nranks) die("elements must divide by ranks");

  struct ibv_device **dl=ibv_get_device_list(NULL); if(!dl) die("no devices");
  struct ibv_device *d=NULL;
  for(int i=0;dl[i];i++) if(!dev || !strcmp(ibv_get_device_name(dl[i]),dev)) { d=dl[i]; break; }
  if(!d) die("device not found");
  struct ibv_context *ctx=ibv_open_device(d); if(!ctx) die("open_device");
  struct ibv_pd *pd=ibv_alloc_pd(ctx); if(!pd) die("alloc_pd");

  int chunk=elems/nranks; size_t cb=(size_t)chunk*sizeof(float);
  int frames=(int)((cb+4095)/4096);
  if(frames>4095) die("chunk exceeds 4095 frames; lower -e or raise -n");

  size_t total=(size_t)elems*sizeof(float)+cb*(size_t)(nranks>1?nranks-1:1);  // vector + one scratch chunk per reduce step
  void *buf; if(posix_memalign(&buf,getpagesize(),total)) die("memalign");
  float *v=(float*)buf, *scratch=(float*)((char*)buf+(size_t)elems*sizeof(float));
  for(int i=0;i<elems;i++) v[i]=(float)(rank+1);   // rank r contributes r+1

  struct ibv_mr *mr=ibv_reg_mr(pd,buf,total,IBV_ACCESS_LOCAL_WRITE); if(!mr) die("reg_mr");
  int depth=4095;   // frames, not messages: give the queue room for the whole message
  struct ibv_cq *cq=ibv_create_cq(ctx,depth*2+2,NULL,NULL,0); if(!cq) die("create_cq");
  struct ibv_qp *qsend=mkqp(pd,cq,depth), *qrecv=mkqp(pd,cq,depth);

  struct ibv_port_attr pa; ibv_query_port(ctx,1,&pa);
  union ibv_gid mygid; ibv_query_gid(ctx,1,0,&mygid);
  struct qpinfo mine[2],theirs[2];
  uint32_t psn_s=lrand48()&0xffffff, psn_r=lrand48()&0xffffff;
  mine[0]=(struct qpinfo){qsend->qp_num,psn_s,pa.lid}; memcpy(mine[0].gid,&mygid,16);
  mine[1]=(struct qpinfo){qrecv->qp_num,psn_r,pa.lid}; memcpy(mine[1].gid,&mygid,16);

  int fd=oob(peer,port);
  xchg(fd,mine,theirs,sizeof mine,peer?0:1);
  close(fd);
  connect_qp(qsend,&theirs[1],psn_s);   // my send pairs with their recv
  connect_qp(qrecv,&theirs[0],psn_r);

  struct timeval t0,t1; gettimeofday(&t0,NULL);
  for(int it=0; it<iters; it++){
    for(int i=0;i<elems;i++) v[i]=(float)(rank+1);
    // Post every receive before any send. The hardware consumes receives in the
    // order posted, and will not process a send until a matching receive exists.
    struct ibv_recv_wr *brw;
    for(int s=0;s<nranks-1;s++){
      struct ibv_sge g={(uintptr_t)(scratch+(size_t)s*chunk),(uint32_t)cb,mr->lkey};
      struct ibv_recv_wr w={.wr_id=(uint64_t)(10+s),.sg_list=&g,.num_sge=1};
      if(ibv_post_recv(qrecv,&w,&brw)) die("post_recv rs");
    }
    for(int s=0;s<nranks-1;s++){
      int ri=((rank-s)%nranks+nranks)%nranks;
      struct ibv_sge g={(uintptr_t)(v+(size_t)ri*chunk),(uint32_t)cb,mr->lkey};
      struct ibv_recv_wr w={.wr_id=(uint64_t)(100+s),.sg_list=&g,.num_sge=1};
      if(ibv_post_recv(qrecv,&w,&brw)) die("post_recv ag");
    }
    struct ibv_send_wr *bsw;
    // reduce-scatter: take the running sum, ADD ours, forward. the hop does work.
    for(int s=0;s<nranks-1;s++){
      int si=((rank-s)%nranks+nranks)%nranks, ri=((rank-s-1)%nranks+nranks)%nranks;
      struct ibv_sge g={(uintptr_t)(v+(size_t)si*chunk),(uint32_t)cb,mr->lkey};
      struct ibv_send_wr w={.wr_id=2,.sg_list=&g,.num_sge=1,.opcode=IBV_WR_SEND,
                            .send_flags=IBV_SEND_SIGNALED};
      if(ibv_post_send(qsend,&w,&bsw)) die("post_send rs");
      poll_n(cq,2);
      float *dst=v+(size_t)ri*chunk, *src=scratch+(size_t)s*chunk;
      for(int i=0;i<chunk;i++) dst[i]+=src[i];
    }
    // allgather: the finished chunk lands straight in place, nothing touches it
    for(int s=0;s<nranks-1;s++){
      int si=((rank-s+1)%nranks+nranks)%nranks;
      struct ibv_sge g={(uintptr_t)(v+(size_t)si*chunk),(uint32_t)cb,mr->lkey};
      struct ibv_send_wr w={.wr_id=4,.sg_list=&g,.num_sge=1,.opcode=IBV_WR_SEND,
                            .send_flags=IBV_SEND_SIGNALED};
      if(ibv_post_send(qsend,&w,&bsw)) die("post_send ag");
      poll_n(cq,2);
    }
  }
  gettimeofday(&t1,NULL);
  double sec=(t1.tv_sec-t0.tv_sec)+(t1.tv_usec-t0.tv_usec)/1e6;
  double expect=(double)nranks*(nranks+1)/2.0;
  double moved=2.0*(nranks-1)/(double)nranks*(double)elems*sizeof(float)*iters;
  printf("rank %d/%d  elems=%d chunk=%d frames=%d  v[0]=%.1f v[last]=%.1f expect=%.1f  %s\n",
         rank,nranks,elems,chunk,frames,v[0],v[elems-1],expect,
         (v[0]==expect && v[elems-1]==expect)?"OK":"MISMATCH");
  printf("rank %d  %.3f s  %.1f Mbit/s bus  %.1f us/allreduce\n",
         rank,sec,moved*8/sec/1e6,sec*1e6/iters);
  return 0;
}
