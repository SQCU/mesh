// see RDMA-FIRST.md
#include "mesh.h"
#include <infiniband/verbs.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/time.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHUNK (1ull<<30)
static struct ibv_context *ctx; static struct ibv_pd *pd; static struct ibv_cq *cq;
static struct ibv_qp *qp; static struct ibv_mr **mr; static int nmr;
static const char *shm; static volatile sig_atomic_t stop;
static void down(void){ if(qp)ibv_destroy_qp(qp); if(cq)ibv_destroy_cq(cq);
  while(nmr) ibv_dereg_mr(mr[--nmr]); free(mr); if(pd)ibv_dealloc_pd(pd);
  if(ctx)ibv_close_device(ctx); if(shm)shm_unlink(shm); }
static void die(const char*m){ fprintf(stderr,"%s\n",m); exit(1); }
static double now(void){ struct timeval t; gettimeofday(&t,NULL); return t.tv_sec+t.tv_usec/1e6; }
static void onsig(int s){ (void)s; stop=1; }

struct qpi { uint32_t qpn,psn; uint16_t lid; uint8_t gid[16]; };
static int oob(const char *peer){
  struct addrinfo hint={.ai_socktype=SOCK_STREAM},*r; int f;
  if(peer){ hint.ai_family=AF_UNSPEC; if(getaddrinfo(peer,MESH_PORT,&hint,&r)) die("addr");
    f=socket(r->ai_family,SOCK_STREAM,0);
    if(connect(f,r->ai_addr,r->ai_addrlen)) die("connect"); freeaddrinfo(r); return f; }
  hint.ai_family=AF_INET6; hint.ai_flags=AI_PASSIVE;
  if(getaddrinfo(NULL,MESH_PORT,&hint,&r)) die("addr");
  int l=socket(r->ai_family,SOCK_STREAM,0),on=1,off=0;
  setsockopt(l,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on);
  setsockopt(l,IPPROTO_IPV6,IPV6_V6ONLY,&off,sizeof off);
  if(bind(l,r->ai_addr,r->ai_addrlen)) die("bind");
  listen(l,1); freeaddrinfo(r); f=accept(l,NULL,NULL); close(l); return f; }

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

  struct ibv_device **dl=ibv_get_device_list(NULL); struct ibv_port_attr pa;
  for(int i=0;dl&&dl[i];i++){ ctx=ibv_open_device(dl[i]);
    if(ctx && !ibv_query_port(ctx,1,&pa) && pa.state==IBV_PORT_ACTIVE) break;
    if(ctx){ ibv_close_device(ctx); ctx=NULL; } }
  if(!ctx) die("no RDMA port is up");
  pd=ibv_alloc_pd(ctx); if(!pd) die("pd");

  uint64_t ram=0; size_t rl=sizeof ram; sysctlbyname("hw.memsize",&ram,&rl,NULL,0);
  const int pg=4096; int np=(int)(pct/100*(double)ram/pg); if(np<64) np=64;
  int pool = np/4 < 244140 ? np/4 : 244140;
  size_t d0=(RINGS+3*MESH_RING*sizeof(struct desc)+65535)/65536*65536;
  size_t span=(size_t)pg*np;
  shm_unlink(name); int rf=shm_open(name,O_CREAT|O_RDWR,MESH_MODE); if(rf<0) die("shm");
  if(ftruncate(rf,(off_t)(d0+span))) die("ftruncate"); fchmod(rf,MESH_MODE);
  void *base=mmap(NULL,d0+span,PROT_READ|PROT_WRITE,MAP_SHARED,rf,0);
  if(base==MAP_FAILED) die("mmap"); memset(base,0,d0); shm=name;
  struct hdr *M=base;
  M->pgsz=pg; M->pool=pool; M->arena=np-pool; M->node=me; M->version=MESH_VERSION;
  M->data_off=d0;
  char *mem=(char*)base+d0;
  mr=calloc((span+CHUNK-1)/CHUNK,sizeof *mr); if(!mr) die("alloc regions");
  for(size_t o=0;o<span;o+=CHUNK){ size_t n=span-o<CHUNK?span-o:CHUNK;
    mr[nmr]=ibv_reg_mr(pd,mem+o,n,IBV_ACCESS_LOCAL_WRITE); if(!mr[nmr++]) die("reg"); }
  __sync_synchronize(); M->magic=MESH_MAGIC;
  fprintf(stderr,"%s %.2f GB = %.1f%% of node, pool %d, %d regions\n",
          name,span/1e9,100.0*span/(double)ram,pool,nmr);

  cq=ibv_create_cq(ctx,4096,NULL,NULL,0);
  struct ibv_qp_init_attr qi={.send_cq=cq,.recv_cq=cq,.qp_type=IBV_QPT_UC,
    .cap={.max_send_wr=4095,.max_recv_wr=4095,.max_send_sge=1,.max_recv_sge=1}};
  qp=ibv_create_qp(pd,&qi); if(!qp) die("qp");
  struct ibv_qp_attr a={.qp_state=IBV_QPS_INIT,.port_num=1};
  ibv_modify_qp(qp,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS);
  union ibv_gid gid; ibv_query_gid(ctx,1,0,&gid);
  uint32_t psn=lrand48()&0xffffff; struct qpi mine={qp->qp_num,psn,pa.lid},you;
  memcpy(mine.gid,&gid,16); int f=oob(peer);
  if(peer){ write(f,&mine,sizeof mine); read(f,&you,sizeof you); }
  else    { read(f,&you,sizeof you); write(f,&mine,sizeof mine); }
  close(f);
  struct ibv_qp_attr r={.qp_state=IBV_QPS_RTR,.path_mtu=IBV_MTU_4096,.rq_psn=you.psn,
    .dest_qp_num=you.qpn,.ah_attr={.dlid=you.lid,.port_num=1,.is_global=1,
    .grh={.hop_limit=1,.sgid_index=0}}};
  memcpy(&r.ah_attr.grh.dgid,you.gid,16);
  if(ibv_modify_qp(qp,&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN))
    die("RTR");
  struct ibv_qp_attr t={.qp_state=IBV_QPS_RTS,.sq_psn=psn};
  ibv_modify_qp(qp,&t,IBV_QP_STATE|IBV_QP_SQ_PSN);

  int *fl=malloc(pool*sizeof(int)); unsigned char *own=calloc(pool,1);
  if(!fl||!own) die("alloc");
  int n[NOWN]={0}; n[FREE]=pool; for(int i=0;i<pool;i++) fl[i]=i;
  int full=0;
  struct wf w[NOWN]={{0}}; double t0=now(), tel=t0;
  #define LKEY(i) (mr[(size_t)(i)*pg/CHUNK]->lkey)
  #define POOL(i)  ((uint32_t)(i) < (uint32_t)pool)
  #define VALID(i) ((uint32_t)(i) < (uint32_t)np)
  #define BUMP(f) atomic_fetch_add_explicit(&M->f,1,memory_order_relaxed)
  #define MV(i,to) do{ n[own[i]]--; own[i]=(to); n[to]++; }while(0)
  #define GIVE(i)  do{ MV(i,FREE); fl[n[FREE]-1]=(i); }while(0)
  #define POST(i,op) ({ struct ibv_sge g={(uintptr_t)mesh_at(M,(uint32_t)(i)),(uint32_t)pg,LKEY(i)}; \
    struct ibv_send_wr s={.wr_id=(i),.sg_list=&g,.num_sge=1,.opcode=IBV_WR_SEND, \
      .send_flags=IBV_SEND_SIGNALED},*bs; struct ibv_recv_wr q={.wr_id=(i),.sg_list=&g,.num_sge=1},*bq; \
    (op) ? ibv_post_send(qp,&s,&bs) : ibv_post_recv(qp,&q,&bq); })

  while(!stop){
    while(n[FREE] && !full){ int i=fl[n[FREE]-1];
      if(POST(i,0)){ full=1; break; } MV(i,RECV); }
    struct desc d;
    while(!pop(M,REL,&d)) if(POOL(d.page) && own[d.page]==APP) GIVE(d.page);
    uint64_t who=atomic_load_explicit(&M->client,memory_order_acquire);
    if(who)
      while(!pop(M,SUB,&d)){
        uint32_t p=d.page; if(!VALID(p)){ BUMP(bad); continue; }
        struct wire *w2=(struct wire*)mesh_at(M,p);
        w2->magic=WIRE_MAGIC; w2->bytes=mesh_clamp(M,d.bytes);
        w2->src=me; w2->dst=d.node; w2->hops=0;
        if(POST(p,1)) break;
        if(POOL(p)) MV(p,SEND);
        BUMP(sent); }
    struct ibv_wc wc[32]; int k=ibv_poll_cq(cq,32,wc);
    for(int j=0;j<k;j++){ int i=(int)wc[j].wr_id;
      if(!VALID(i)) die("wr_id outside the pool");
      if(wc[j].opcode!=IBV_WC_RECV){ if(POOL(i)) GIVE(i); continue; }
      full=0; if(wc[j].status!=IBV_WC_SUCCESS){ GIVE(i); continue; }
      struct wire *h=(struct wire*)mesh_at(M,(uint32_t)i);
      if(h->magic!=WIRE_MAGIC){ GIVE(i); continue; }
      h->hops++;
      if(h->dst!=(uint16_t)me && h->hops<=32){ if(!POST(i,1)) MV(i,SEND); else GIVE(i); continue; }
      struct desc c={.page=(uint32_t)i,.bytes=mesh_clamp(M,h->bytes),.node=h->src};
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
      tel=tn; } }
  return 0; }
