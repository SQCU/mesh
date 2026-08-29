// see RDMA-FIRST.md
#include "mesh.h"
#include <infiniband/verbs.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/time.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <signal.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHUNK (1ull<<30)
#define QD 4095
static struct ibv_context *ctx; static struct ibv_pd *pd; static struct ibv_cq *cq;
static struct ibv_qp *qp; static struct ibv_mr **mr; static int nmr;
static const char *shm; static volatile sig_atomic_t stop;
static void down_pair(void){ if(qp){ibv_destroy_qp(qp);qp=0;} if(cq){ibv_destroy_cq(cq);cq=0;} }
static void down_verbs(void){ down_pair();
  while(nmr) ibv_dereg_mr(mr[--nmr]); free(mr); mr=0;
  if(pd){ibv_dealloc_pd(pd);pd=0;} if(ctx){ibv_close_device(ctx);ctx=0;} }
static void down(void){ down_verbs(); if(shm)shm_unlink(shm); }
static void die(const char*m){ fprintf(stderr,"%s\n",m); exit(1); }
static double now(void){ struct timeval t; gettimeofday(&t,NULL); return t.tv_sec+t.tv_usec/1e6; }
static void onsig(int s){ (void)s; stop=1; }

struct qpi { uint32_t qpn,psn; uint16_t lid; uint8_t gid[16]; };
static int dial(struct addrinfo *a){
  int f=socket(a->ai_family,SOCK_STREAM,0); if(f<0) return -1;
  fcntl(f,F_SETFL,O_NONBLOCK);
  if(connect(f,a->ai_addr,a->ai_addrlen) && errno!=EINPROGRESS){ close(f); return -1; }
  fd_set w; FD_ZERO(&w); FD_SET(f,&w); struct timeval tv={1,0};
  int e=0; socklen_t el=sizeof e;
  if(select(f+1,NULL,&w,NULL,&tv)<1 || getsockopt(f,SOL_SOCKET,SO_ERROR,&e,&el) || e){
    close(f); return -1; }
  fcntl(f,F_SETFL,0);
  struct timeval rt={3,0};
  setsockopt(f,SOL_SOCKET,SO_RCVTIMEO,&rt,sizeof rt);
  return f; }

static int oob(const char *peer){
  struct addrinfo hint={.ai_socktype=SOCK_STREAM},*r; int f;
  if(peer){ hint.ai_family=AF_UNSPEC;
    if(getaddrinfo(peer,MESH_PORT,&hint,&r)) return -1;
    for(int pass=0; pass<2; pass++)
      for(struct addrinfo *a=r; a; a=a->ai_next){
        if((pass==0) != (a->ai_family==AF_INET)) continue;
        if((f=dial(a))>=0){ freeaddrinfo(r); return f; } }
    freeaddrinfo(r); usleep(200000); return -1; }
  hint.ai_family=AF_INET6; hint.ai_flags=AI_PASSIVE;
  if(getaddrinfo(NULL,MESH_PORT,&hint,&r)) return -1;
  int l=socket(r->ai_family,SOCK_STREAM,0),on=1,off=0;
  setsockopt(l,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on);
  setsockopt(l,IPPROTO_IPV6,IPV6_V6ONLY,&off,sizeof off);
  if(bind(l,r->ai_addr,r->ai_addrlen)){ close(l); freeaddrinfo(r); usleep(200000); return -1; }
  listen(l,1); freeaddrinfo(r); f=accept(l,NULL,NULL); close(l); return f; }

static struct ibv_port_attr pa;
static int verbs_up(const char *peer, char *mem, size_t span, int me){
  down_verbs();
  {
  struct ibv_device **dl=ibv_get_device_list(NULL);
  for(int i=0;dl&&dl[i];i++){ ctx=ibv_open_device(dl[i]);
    if(ctx && !ibv_query_port(ctx,1,&pa) && pa.state==IBV_PORT_ACTIVE) break;
    if(ctx){ ibv_close_device(ctx); ctx=0; } }
  if(dl) ibv_free_device_list(dl);
  if(!ctx){ usleep(500000); return -1; }
  pd=ibv_alloc_pd(ctx); if(!pd){ down_verbs(); return -1; }
  mr=calloc((span+CHUNK-1)/CHUNK,sizeof *mr); if(!mr) die("alloc regions");
  for(size_t o=0;o<span;o+=CHUNK){ size_t n=span-o<CHUNK?span-o:CHUNK;
    mr[nmr]=ibv_reg_mr(pd,mem+o,n,IBV_ACCESS_LOCAL_WRITE);
    if(!mr[nmr++]){ down_verbs(); return -1; } }
  }
  ibv_query_port(ctx,1,&pa);
  { char c[96]; const char *dn=ibv_get_device_name(ctx->device);
    snprintf(c,sizeof c,"ping6 -c 2 -i 0.2 ff02::1%%%s >/dev/null 2>&1",
             strncmp(dn,"rdma_",5)?dn:dn+5);
    system(c); }
  cq=ibv_create_cq(ctx,4096,NULL,NULL,0); if(!cq) return -1;
  struct ibv_qp_init_attr qi={.send_cq=cq,.recv_cq=cq,.qp_type=IBV_QPT_UC,
    .cap={.max_send_wr=QD,.max_recv_wr=QD,.max_send_sge=1,.max_recv_sge=1}};
  qp=ibv_create_qp(pd,&qi); if(!qp) return -1;
  struct ibv_qp_attr a={.qp_state=IBV_QPS_INIT,.port_num=1};
  ibv_modify_qp(qp,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS);
  union ibv_gid gid; ibv_query_gid(ctx,1,0,&gid);
  uint32_t psn=lrand48()&0xffffff; struct qpi mine={qp->qp_num,psn,pa.lid},you;
  memcpy(mine.gid,&gid,16);
  int f=oob(peer); if(f<0){ fprintf(stderr,"oob retry\n"); return -1; }
  if(peer) write(f,&mine,sizeof mine);
  for(size_t got=0; got<sizeof you;){
    ssize_t g=read(f,(char*)&you+got,sizeof you-got);
    if(g<=0){ close(f); fprintf(stderr,"xchg retry\n"); return -1; } got+=(size_t)g; }
  if(!peer) write(f,&mine,sizeof mine);
  close(f);
  struct ibv_qp_attr r={.qp_state=IBV_QPS_RTR,.path_mtu=IBV_MTU_4096,.rq_psn=you.psn,
    .dest_qp_num=you.qpn,.ah_attr={.dlid=you.lid,.port_num=1,.is_global=1,
    .grh={.hop_limit=1,.sgid_index=0}}};
  memcpy(&r.ah_attr.grh.dgid,you.gid,16);
  int rc=ibv_modify_qp(qp,&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN);
  if(rc){
    { char c[96]; const char *dn=ibv_get_device_name(ctx->device);
      snprintf(c,sizeof c,"ping6 -c 2 -i 0.2 ff02::1%%%s >/dev/null 2>&1",
               strncmp(dn,"rdma_",5)?dn:dn+5);
      system(c); }
    rc=ibv_modify_qp(qp,&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN);
  }
  if(rc){ fprintf(stderr,"rtr retry rc %d\n",rc); usleep(300000); return -1; }
  struct ibv_qp_attr t={.qp_state=IBV_QPS_RTS,.sq_psn=psn};
  ibv_modify_qp(qp,&t,IBV_QP_STATE|IBV_QP_SQ_PSN);
  fprintf(stderr,"pair up: %s node %d\n",ibv_get_device_name(ctx->device),me);
  return 0; }

struct wf { double n, mean, m2; };
static void add(struct wf *w, double x){
  w->n+=1; double d=x-w->mean; w->mean+=d/w->n; w->m2+=d*(x-w->mean); }

int main(int argc,char**argv){
  const char *peer=NULL, *name=MESH_NAME; int me=0; double pct=25;
  for(int i=1;i<argc;i++)
    if(!strcmp(argv[i],"-I")) me=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-M")) pct=atof(argv[++i]);
    else if(!strcmp(argv[i],"-s")) name=argv[++i];
    else if(argv[i][0]=='-'){ fprintf(stderr,"unknown %s\n",argv[i]); return 2; }
    else peer=argv[i];
  atexit(down); struct sigaction sa={0}; sa.sa_handler=onsig;
  sigaction(SIGINT,&sa,NULL); sigaction(SIGTERM,&sa,NULL);

  uint64_t ram=0; size_t rl=sizeof ram; sysctlbyname("hw.memsize",&ram,&rl,NULL,0);
  const int pg=4096; int np=(int)(pct/100*(double)ram/pg); if(np<64) np=64;
  int pool = np/4 < 244140 ? np/4 : 244140;
  size_t d0=(RINGS+NRING*MESH_RING*sizeof(struct desc)+65535)/65536*65536;
  size_t span=(size_t)pg*np;
  shm_unlink(name); int rf=shm_open(name,O_CREAT|O_RDWR,MESH_MODE); if(rf<0) die("shm");
  if(ftruncate(rf,(off_t)(d0+span))) die("ftruncate"); fchmod(rf,MESH_MODE);
  void *base=mmap(NULL,d0+span,PROT_READ|PROT_WRITE,MAP_SHARED,rf,0);
  if(base==MAP_FAILED) die("mmap"); shm=name;
  struct hdr *M=base;
  M->pgsz=pg; M->pool=pool; M->arena=np-pool; M->node=me; M->version=MESH_VERSION;
  M->data_off=d0;
  char *mem=(char*)base+d0;
  __sync_synchronize(); M->magic=MESH_MAGIC;
  fprintf(stderr,"%s %.2f GB = %.1f%% of node, pool %d\n",
          name,span/1e9,100.0*span/(double)ram,pool);

  int *fl=malloc(pool*sizeof(int)); unsigned char *own=calloc(pool,1);
  if(!fl||!own) die("alloc");
  int n[NOWN]={0}; n[FREE]=pool; for(int i=0;i<pool;i++) fl[i]=i;
  int sends=0;
  struct wf w[NOWN]={{0}}; double t0=now(), tel=t0;
  #define LKEY(i) (mr[(size_t)(i)*pg/CHUNK]->lkey)
  #define POOL(i)  ((uint32_t)(i) < (uint32_t)pool)
  #define VALID(i) ((uint32_t)(i) < (uint32_t)np)
  #define BUMP(f) atomic_fetch_add_explicit(&M->f,1,memory_order_relaxed)
  #define MV(i,to) do{ n[own[i]]--; own[i]=(to); n[to]++; }while(0)
  #define GIVE(i)  do{ MV(i,FREE); fl[n[FREE]-1]=(i); }while(0)
  #define POST_RECV(i) ({ struct ibv_sge g={(uintptr_t)mesh_at(M,(uint32_t)(i)),(uint32_t)pg,LKEY(i)}; \
    struct ibv_recv_wr q={.wr_id=(i),.sg_list=&g,.num_sge=1},*bq; ibv_post_recv(qp,&q,&bq); })
  #define POST_SEND(i,len) ({ struct ibv_sge g={(uintptr_t)mesh_at(M,(uint32_t)(i)),(len),LKEY(i)}; \
    struct ibv_send_wr s={.wr_id=(i),.sg_list=&g,.num_sge=1,.opcode=IBV_WR_SEND, \
      .send_flags=IBV_SEND_SIGNALED},*bs; ibv_post_send(qp,&s,&bs); })

  while(!stop){
    while(!stop && verbs_up(peer,mem,span,me)){}
    if(stop) break;
    n[FREE]=n[RECV]=n[SEND]=n[APP]=0; int nf=0;
    for(int i=0;i<pool;i++){
      if(own[i]==APP) n[APP]++;
      else { own[i]=FREE; n[FREE]++; fl[nf++]=i; } }
    sends=0; int alive=1, dry=0; uint64_t wcs=0;
    while(!stop && alive){
    while(n[FREE] && n[RECV]<QD){ int i=fl[n[FREE]-1];
      if(POST_RECV(i)) break;
      MV(i,RECV); }
    struct desc d;
    while(!pop(M,REL,&d)) if(POOL(d.page) && own[d.page]==APP) GIVE(d.page);
    uint64_t who=atomic_load_explicit(&M->client,memory_order_acquire);
    if(who)
      while(sends<QD && !pop(M,SUB,&d)){
        uint32_t p=d.page;
        if(p<(uint32_t)pool || !VALID(p)){ BUMP(bad); continue; }
        struct wire *w2=(struct wire*)mesh_at(M,p);
        w2->src=me; w2->dst=d.node; w2->hops=0;
        uint32_t len=mesh_pay(M); if(d.bytes<len) len=d.bytes;
        if(POST_SEND(p,(uint32_t)sizeof(struct wire)+len)){ BUMP(bad); break; }
        sends++; BUMP(sent); }
    struct ibv_wc wc[32]; int k=ibv_poll_cq(cq,32,wc);
    wcs+=(uint64_t)(k>0?k:0);
    for(int j=0;j<k;j++){ int i=(int)wc[j].wr_id;
      if(!VALID(i)) die("wr_id outside the pool");
      if(!POOL(i) || own[i]!=RECV){ sends--;
        if(POOL(i)) GIVE(i);
        else { struct desc a={.page=(uint32_t)i}; push(M,ACK,&a); }
        continue; }
      if(wc[j].status!=IBV_WC_SUCCESS){ GIVE(i); continue; }
      uint32_t bl=wc[j].byte_len;
      if(bl<sizeof(struct wire)) die("runt frame");
      struct wire *h=(struct wire*)mesh_at(M,(uint32_t)i);
      if(h->dst==0xffff){ GIVE(i); continue; }
      h->hops++;
      if(h->dst!=(uint16_t)me && h->hops<=32){
        if(sends<QD && !POST_SEND(i,bl)){ sends++; MV(i,SEND); } else GIVE(i); continue; }
      struct desc c={.page=(uint32_t)i,.bytes=bl-(uint32_t)sizeof(struct wire),.node=h->src};
      if(!who || push(M,CMP,&c)) GIVE(i);
      else { MV(i,APP); BUMP(recvd); } }
    for(int s=0;s<NOWN;s++) add(&w[s],n[s]);
    double tn=now();
    if(tn-tel>=1){
      if(who && kill((pid_t)who,0) && errno==ESRCH){
        for(int i=0;i<pool;i++) if(own[i]==APP) GIVE(i);
        for(int k=0;k<NRING;k++)
          atomic_store_explicit(&M->r[k].tail,M->r[k].head,memory_order_release);
        atomic_store_explicit(&M->client,0,memory_order_release); }
      for(int s=0;s<NOWN;s++){
        atomic_store_explicit(&M->mean[s],(uint64_t)w[s].mean,memory_order_relaxed);
        atomic_store_explicit(&M->sd[s],(uint64_t)sqrt(w[s].n>1?w[s].m2/(w[s].n-1):0),memory_order_relaxed);
        w[s]=(struct wf){0}; }
      atomic_store_explicit(&M->up_ms,(uint64_t)((tn-t0)*1000),memory_order_relaxed);
      if(ibv_query_port(ctx,1,&pa) || pa.state!=IBV_PORT_ACTIVE) alive=0;
      else if(wcs==0){
        if(sends>0) dry++;
        else if(n[FREE]){ int i=fl[n[FREE]-1];
          struct wire *pr=(struct wire*)mesh_at(M,(uint32_t)i);
          pr->src=(uint16_t)me; pr->dst=0xffff; pr->hops=0;
          if(!POST_SEND(i,(uint32_t)sizeof(struct wire))){ MV(i,SEND); sends++; } }
        if(dry>=4) alive=0; }
      else dry=0;
      wcs=0;
      if(!alive) fprintf(stderr,"pair down\n");
      tel=tn; } } }
  return 0; }
