
#include "mesh.h"
#include "mesh-wire.h"
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
#include <time.h>
#include <pthread.h>
#include "mesh-flight.h"

#define CHUNK (1ull<<30)
#define QD 4095
#define IDLE_POLL_US 50
static _Thread_local struct ibv_context *ctx;
static _Thread_local struct ibv_pd *pd;
static _Thread_local struct ibv_cq *cq;
static _Thread_local struct ibv_qp *qp;
static _Thread_local struct ibv_mr **mr;
static _Thread_local int nmr;
static struct ibv_sge region_sge(const char *base, size_t offset, uint32_t bytes){
  uintptr_t address=(uintptr_t)base+offset;
  return (struct ibv_sge){address,bytes,mr[address/CHUNK-(uintptr_t)base/CHUNK]->lkey}; }
static _Thread_local const char *shm; static _Atomic sig_atomic_t stop;
static _Thread_local int lsock=-1;
static _Thread_local uint64_t mynonce, peernonce;
static _Thread_local int retire_device, expected_peer=-1;
static _Thread_local const char *listen_address, *selected_device;
static int down_pair(void){
  if(qp){ if(TRACE(DESTROY_QP,qp,qp->qp_num,0,ibv_destroy_qp(qp))) return 0; qp=0; return 0; }
  if(cq){ if(TRACE(DESTROY_CQ,cq,0,0,ibv_destroy_cq(cq))) return 0; cq=0; return 0; }
  return 1; }
static int down_verbs(void){
  if(!down_pair()) return 0;
  if(nmr){ struct ibv_mr *r=mr[nmr-1];
    if(!TRACE(DEREG_MR,r,r->lkey,r->length,ibv_dereg_mr(r))) nmr--;
    return 0; }
  free(mr); mr=0;
  if(pd){ if(TRACE(DEALLOC_PD,pd,0,0,ibv_dealloc_pd(pd))) return 0; pd=0; return 0; }
  if(ctx){ if(TRACE(CLOSE_DEVICE,ctx,0,0,ibv_close_device(ctx))) return 0; ctx=0; return 0; }
  return 1; }
static void down(void){ if(shm)shm_unlink(shm); }
static void die(const char*m){ fprintf(stderr,"%s\n",m); exit(1); }
static double now(void){ struct timeval t; gettimeofday(&t,NULL); return t.tv_sec+t.tv_usec/1e6; }
static double monotime(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
static void idle_poll(int activity,double *until){
  double stamp=monotime();
  if(activity) *until=stamp+0.001;
  else if(stamp>=*until) usleep(IDLE_POLL_US);
}
static void onsig(int s){ (void)s; stop=1; }
static void bridge_account(int error){ if(error){ fprintf(stderr,"bridge acknowledgement accounting failed\n"); stop=1; } }

struct qpi { uint32_t xmagic, xsize; uint64_t nonce; uint32_t qpn,psn; uint16_t lid; uint8_t gid[16]; uint16_t node; };
#define XMAGIC 0x4d585047u
static _Thread_local uint32_t exchange_magic=XMAGIC;
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

static int listener_up(void){
  struct addrinfo hint={.ai_socktype=SOCK_STREAM,.ai_family=AF_INET6,.ai_flags=AI_PASSIVE},*r;
  if(getaddrinfo(listen_address,MESH_PORT,&hint,&r)) return -1;
  lsock=socket(r->ai_family,SOCK_STREAM,0);
  int on=1,off=0;
  setsockopt(lsock,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on);
  setsockopt(lsock,IPPROTO_IPV6,IPV6_V6ONLY,&off,sizeof off);
  int error=bind(lsock,r->ai_addr,r->ai_addrlen) || listen(lsock,4);
  freeaddrinfo(r);
  if(error){ close(lsock); lsock=-1; return -1; }
  fcntl(lsock,F_SETFL,O_NONBLOCK); return 0; }

static _Thread_local int backoff_n;
static double retry_delay(void){
  double seconds=0.2*(1u<<backoff_n);
  if(backoff_n<7) backoff_n++;
  return seconds; }

static int exchange(int f, const struct qpi *mine, struct qpi *you, double deadline){
  size_t sent=0,got=0;
  fcntl(f,F_SETFL,O_NONBLOCK);
  while(!stop && (sent<sizeof *mine || got<sizeof *you)){
    double left=deadline-monotime(); if(left<=0) return -1;
    fd_set r,w; FD_ZERO(&r); FD_ZERO(&w);
    if(got<sizeof *you) FD_SET(f,&r);
    if(sent<sizeof *mine) FD_SET(f,&w);
    struct timeval tv={.tv_sec=(int)left,.tv_usec=(int)((left-(int)left)*1e6)};
    int ready=select(f+1,&r,&w,0,&tv);
    if(ready<0 && errno==EINTR) continue;
    if(ready<=0) return -1;
    if(FD_ISSET(f,&w)){
      ssize_t n=write(f,(const char*)mine+sent,sizeof *mine-sent);
      if(n>0) sent+=(size_t)n;
      else if(!n || (errno!=EAGAIN && errno!=EINTR)) return -1; }
    if(FD_ISSET(f,&r)){
      ssize_t n=read(f,(char*)you+got,sizeof *you-got);
      if(n>0) got+=(size_t)n;
      else if(!n || (errno!=EAGAIN && errno!=EINTR)) return -1; }
  }
  return stop?-1:0; }

static int oob(const char *peer){
  int f=accept(lsock,NULL,NULL);
  if(f>=0){ struct timeval rt={10,0}; setsockopt(f,SOL_SOCKET,SO_RCVTIMEO,&rt,sizeof rt);
    return f; }
  if(!peer) return -1;
  struct addrinfo hint={.ai_socktype=SOCK_STREAM,.ai_family=AF_UNSPEC},*r;
  if(getaddrinfo(peer,MESH_PORT,&hint,&r)) return -1;
  for(int pass=0; pass<2; pass++)
    for(struct addrinfo *a=r; a; a=a->ai_next){
      if((pass==0) != (a->ai_family==AF_INET)) continue;
      if((f=dial(a))>=0){ freeaddrinfo(r);
        struct timeval rt={10,0}; setsockopt(f,SOL_SOCKET,SO_RCVTIMEO,&rt,sizeof rt);
        return f; } }
  freeaddrinfo(r); return -1; }

static _Thread_local struct ibv_port_attr pa;
static int verbs_up(const char *peer, char *mem, size_t span, int me){
  if(ctx && (TRACE(QUERY_PORT,ctx,1,0,ibv_query_port(ctx,1,&pa)) || pa.state!=IBV_PORT_ACTIVE)){
    retire_device=1; return -1; }
  int f=oob(peer); if(f<0) return -1;
  if(!ctx){
  struct ibv_device **dl=ibv_get_device_list(NULL);
  for(int i=0;dl&&dl[i];i++){
    if(selected_device && strcmp(selected_device,ibv_get_device_name(dl[i]))) continue;
    ctx=TRACE(OPEN_DEVICE,dl[i],i,0,ibv_open_device(dl[i]));
    if(ctx && !TRACE(QUERY_PORT,ctx,1,0,ibv_query_port(ctx,1,&pa)) && pa.state==IBV_PORT_ACTIVE) break;
    if(ctx){
      if(TRACE(CLOSE_DEVICE,ctx,0,0,ibv_close_device(ctx))){
        retire_device=1; ibv_free_device_list(dl); close(f); return -1; }
      ctx=0; } }
  if(dl) ibv_free_device_list(dl);
  if(!ctx){ close(f); return -1; }
  }
  if(!pd) pd=TRACE(ALLOC_PD,ctx,0,0,ibv_alloc_pd(ctx));
  if(!pd){ close(f); return -1; }
  size_t head=(uintptr_t)mem%CHUNK, regions=(head+span+CHUNK-1)/CHUNK;
  if(!mr) mr=calloc(regions,sizeof *mr);
  if(!mr){ close(f); fprintf(stderr,"alloc regions: retrying\n"); return -1; }
  while((size_t)nmr<regions){
    size_t o=nmr?(size_t)nmr*CHUNK-head:0, end=((size_t)nmr+1)*CHUNK-head;
    size_t n=(end<span?end:span)-o;
    mr[nmr]=TRACE(REG_MR,mem+o,o,n,ibv_reg_mr(pd,mem+o,n,IBV_ACCESS_LOCAL_WRITE));
    if(!mr[nmr]){ close(f); return -1; } nmr++; }
  if(TRACE(QUERY_PORT,ctx,1,0,ibv_query_port(ctx,1,&pa))){ close(f); return -1; }
  { char c[96]; const char *dn=ibv_get_device_name(ctx->device);
    snprintf(c,sizeof c,"ping6 -c 2 -i 0.2 ff02::1%%%s >/dev/null 2>&1",
             strncmp(dn,"rdma_",5)?dn:dn+5);
    system(c); }
  cq=TRACE(CREATE_CQ,ctx,4096,0,ibv_create_cq(ctx,4096,NULL,NULL,0)); if(!cq){ close(f); return -1; }
  struct ibv_qp_init_attr qi={.send_cq=cq,.recv_cq=cq,.qp_type=IBV_QPT_UC,
    .cap={.max_send_wr=QD,.max_recv_wr=QD,.max_send_sge=1,.max_recv_sge=1}};
  qp=TRACE(CREATE_QP,pd,QD,0,ibv_create_qp(pd,&qi)); if(!qp){ close(f); return -1; }
  struct ibv_qp_attr a={.qp_state=IBV_QPS_INIT,.port_num=1};
  if(TRACE(INIT,qp,qp->qp_num,0,ibv_modify_qp(qp,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS))){ close(f); return -1; }
  union ibv_gid gid; if(TRACE(QUERY_GID,ctx,1,0,ibv_query_gid(ctx,1,0,&gid))){ close(f); return -1; }
  uint32_t psn=arc4random()&0xffffff;
  struct qpi mine={.xmagic=exchange_magic,.xsize=sizeof mine,.nonce=mynonce,.qpn=qp->qp_num,.psn=psn,.lid=pa.lid,.node=(uint16_t)me},you;
  memcpy(mine.gid,&gid,16);
  if(exchange(f,&mine,&you,monotime()+10)){ close(f); fprintf(stderr,"xchg retry\n"); return -1; }
  close(f);
  if((you.xmagic!=XMAGIC && you.xmagic!=XMAGIC+1) || you.xsize!=sizeof you || (expected_peer>=0 && you.node!=expected_peer)){
    fprintf(stderr,"peer speaks a different exchange, retrying\n"); return -1; }
  if(you.nonce==mynonce){ fprintf(stderr,"self nonce, retry\n"); return -1; }
  exchange_magic=you.xmagic;
  peernonce=you.nonce;
  struct ibv_qp_attr r={.qp_state=IBV_QPS_RTR,.path_mtu=IBV_MTU_4096,.rq_psn=you.psn,
    .dest_qp_num=you.qpn,.ah_attr={.dlid=you.lid,.port_num=1,.is_global=1,
    .grh={.hop_limit=1,.sgid_index=0}}};
  memcpy(&r.ah_attr.grh.dgid,you.gid,16);
  int rc=TRACE(RTR,qp,you.qpn,you.psn,ibv_modify_qp(qp,&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN));
  if(rc){ fprintf(stderr,"rtr rc %d dlid %u dqpn %u dgid %02x%02x..%02x%02x mygid %02x%02x\n",
      rc, you.lid, you.qpn, you.gid[0],you.gid[1],you.gid[14],you.gid[15],
      mine.gid[0],mine.gid[15]); return -1; }
  struct ibv_qp_attr t={.qp_state=IBV_QPS_RTS,.sq_psn=psn};
  rc=TRACE(RTS,qp,qp->qp_num,psn,ibv_modify_qp(qp,&t,IBV_QP_STATE|IBV_QP_SQ_PSN));
  if(rc){ fprintf(stderr,"rts rc %d, retrying\n",rc); return -1; }
  fprintf(stderr,"pair up: %s node %d\n",ibv_get_device_name(ctx->device),me);
  return 0; }

struct wf { double n, mean, m2; };
static void add(struct wf *w, double x){
  w->n+=1; double d=x-w->mean; w->mean+=d/w->n; w->m2+=d*(x-w->mean); }

#include "mesh-links.h"
int main(int argc,char**argv){
  const char *peer=NULL,*name=MESH_NAME; int me=0,link_count=0; double pct=25; unsigned hop_limit=32;
  struct mesh_link *links=calloc(LINK_LIMIT,sizeof *links);
  struct mesh_route *routes=calloc(65536,sizeof *routes); if(!links||!routes) die("link metadata");
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-I") && i+1<argc) me=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-M") && i+1<argc) pct=atof(argv[++i]);
    else if(!strcmp(argv[i],"-s") && i+1<argc) name=argv[++i];
    else if(!strcmp(argv[i],"--hop-limit") && i+1<argc) hop_limit=(unsigned)strtoul(argv[++i],NULL,10);
    else if(!strcmp(argv[i],"--route") && i+1<argc){
      unsigned destination,index; char extra;
      if(sscanf(argv[++i],"%u:%u%c",&destination,&index,&extra)!=2 || destination>=65535 || index>=LINK_LIMIT) die("route descriptor");
      if(!routes[destination].mask) routes[destination].first=(uint16_t)index;
      routes[destination].mask|=(uint16_t)(1u<<index);
    }
    else if(!strcmp(argv[i],"--link") && i+1<argc){
      if(link_count==LINK_LIMIT) die("link descriptor capacity");
      char *fields=strdup(argv[++i]),*cursor=fields,*part[4];
      for(int j=0;j<4;j++) part[j]=strsep(&cursor,",");
      if(!part[0]||!part[1]||!part[2]||!part[3]||cursor) die("link requires device,peer,local-address,peer-address");
      char *end; unsigned long node=strtoul(part[1],&end,10);
      if(*end || node>=65535) die("link peer identity");
      struct mesh_link *link=&links[link_count++];
      link->device=part[0]; link->peer_node=(int)node; link->local=part[2]; link->peer=part[3];
    }
    else if(argv[i][0]=='-') fprintf(stderr,"ignoring unknown %s\n",argv[i]);
    else peer=argv[i];
  }
  if(me<0 || me>=65535 || !isfinite(pct) || pct<=0 || pct>100 || !hop_limit || hop_limit>UINT16_MAX) die("bridge geometry");
  if(!link_count){ link_count=1; links[0].peer=peer; links[0].peer_node=-1; }
  for(int i=0;i<65536;i++) if(routes[i].mask>>link_count) die("route names absent link");
  atexit(down); struct sigaction sa={0}; sa.sa_handler=onsig;
  sigaction(SIGINT,&sa,NULL); sigaction(SIGTERM,&sa,NULL); sigaction(SIGHUP,&sa,NULL); signal(SIGPIPE,SIG_IGN);
  uint64_t ram=0; size_t rl=sizeof ram; sysctlbyname("hw.memsize",&ram,&rl,NULL,0);
  const uint32_t pg=4096; uint64_t wanted=(uint64_t)(pct/100*(double)ram/pg);
  if(wanted<NOWN) wanted=NOWN;
  if(wanted>INT32_MAX) die("page index capacity");
  int np=(int)wanted,pool=np/NOWN;
  int receive_share=(pool-link_count)/link_count;
  size_t d0=(RINGS+NRING*MESH_RING*sizeof(struct desc)+LINK_LIMIT*sizeof(struct mesh_port_info)+65535)/65536*65536;
  size_t span=(size_t)pg*np;
  shm_unlink(name); int fd=shm_open(name,O_CREAT|O_RDWR,MESH_MODE); if(fd<0) die("shm");
  if(ftruncate(fd,(off_t)(d0+span))) die("ftruncate"); fchmod(fd,MESH_MODE);
  struct hdr *M=mmap(NULL,d0+span,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0); close(fd);
  if(M==MAP_FAILED) die("mmap"); shm=name;
  M->pgsz=pg; M->pool=(uint32_t)pool; M->arena=(uint32_t)(np-pool); M->node=(uint32_t)me;
  M->version=MESH_VERSION; M->data_off=d0; atomic_store(&M->port_count,(uint64_t)link_count);
  __sync_synchronize(); M->magic=MESH_MAGIC;
  int *free_pages=malloc((size_t)pool*sizeof *free_pages);
  unsigned char *owner=calloc((size_t)pool,1),*pool_link=calloc((size_t)pool,1),*submitted=calloc((size_t)(np-pool),1);
  if(!free_pages||!owner||!pool_link||!submitted) die("arena ownership metadata");
  enum { READY=NOWN };
  int counts[NOWN+1]={0}; counts[FREE]=pool;
  for(int i=0;i<pool;i++) free_pages[i]=i;
  struct mesh_port_info *ports=mesh_ports(M);
  flight_status=M; atomic_store(&M->bridge_pid,(uint64_t)getpid());
  for(int i=0;i<link_count;i++){
    links[i].node=me; links[i].name=name; links[i].memory=(char*)M+d0; links[i].span=span;
    links[i].probe_page=UINT32_MAX;
    links[i].depth=links[i].device && (!strcmp(links[i].device,"udp")||!strcmp(links[i].device,"tcp"))?LINK_QUEUE/2:QD;
    snprintf(ports[i].device,sizeof ports[i].device,"%s",links[i].device?links[i].device:"automatic");
    ports[i].peer=(uint16_t)links[i].peer_node;
    int error=pthread_create(&links[i].thread,NULL,link_worker_for(&links[i]),&links[i]);
    if(error){ errno=error; perror("link worker"); atomic_store(&links[i].stopped,1); stop=1; }
  }
  fprintf(stderr,"%s %.2f GB = %.1f%% of node, pool %d, links %d\n",name,span/1e9,100.0*span/(double)ram,pool,link_count);
  struct wf stats[NOWN]={{0}}; double began=now(),telemetry=began,awake_until=0;
  int arena_pending=0,draining=0; uint64_t submit_cursor=0;
  #define COUNT(field) atomic_fetch_add_explicit(&M->field,1,memory_order_relaxed)
  #define MOVE(page,to) do{ counts[owner[page]]--; owner[page]=(to); counts[to]++; }while(0)
  #define RELEASE(page) do{ MOVE(page,FREE); free_pages[counts[FREE]-1]=(page); }while(0)
  for(;;){
    int activity=0,live=0,stopped=0;
    uint64_t who=atomic_load_explicit(&M->client,memory_order_acquire);
    int client=who && who!=MESH_CLIENT_DRAIN && who!=MESH_CLIENT_DETACH;
    if(who==MESH_CLIENT_DETACH) draining=1;
    struct desc descriptor;
    for(int budget=0;budget<256 && !pop(M,REL,&descriptor);budget++){
      if(descriptor.page<(uint32_t)pool && owner[descriptor.page]==APP) RELEASE(descriptor.page);
      activity=1;
    }
    for(int index=0;index<link_count;index++){
      struct mesh_link *link=&links[index]; uint64_t position;
      if(atomic_exchange(&ports[index].reset_request,0)){ link->up=0; atomic_store(&link->reset,1); }
      for(int budget=0;budget<1024 && ring_select(&link->completion.cursor,&link->cursor,&position);budget++){
        struct link_event event=link->completion.entries[position%LINK_QUEUE];
        uint32_t page=event.page;
        if(event.kind==L_UP){ link->accepted=event.generation; link->up=1; link->peer_seen=monotime(); goto accepted; }
        if(event.kind==L_FAULT){ COUNT(bad); link->up=0; goto accepted; }
        if(event.kind==L_RETIRED){
          link->up=0;
          for(int i=0;i<pool;i++) if(pool_link[i]==index && (owner[i]==RECV || owner[i]==SEND)) RELEASE(i);
          for(int i=pool;i<np;i++) if(submitted[i-pool]==index+1){
            struct desc ack={.page=(uint32_t)i};
            if(!draining) bridge_account(push(M,ACK,&ack));
            submitted[i-pool]=0; arena_pending--;
          }
          link->receives=link->sends=0; link->probe_page=UINT32_MAX; goto accepted;
        }
        if((event.kind!=L_READY && event.generation!=link->accepted) || page>=(uint32_t)np){ COUNT(bad); goto accepted; }
        if(event.kind==L_SEND){
          link->sends--;
          if(page==link->probe_page) link->probe_page=UINT32_MAX;
          if(page<(uint32_t)pool){ if(owner[page]==SEND) RELEASE(page); else COUNT(bad); }
          else if(submitted[page-pool]==index+1){
            submitted[page-pool]=0; arena_pending--;
            struct desc ack={.page=page}; if(!draining) bridge_account(push(M,ACK,&ack));
          } else COUNT(bad);
          goto accepted;
        }
        if(event.kind==L_RECV && page<(uint32_t)pool && owner[page]==RECV){
          link->receives--; link->peer_seen=monotime(); MOVE(page,READY);
          event.kind=L_READY; link->completion.entries[position%LINK_QUEUE].kind=L_READY;
        }
        if(event.kind!=L_READY || page>=(uint32_t)pool || owner[page]!=READY){ COUNT(bad); goto accepted; }
        if(event.bytes<sizeof(struct wire) || event.bytes>pg){ COUNT(bad); RELEASE(page); goto accepted; }
        struct wire *wire=(struct wire*)mesh_at(M,page);
        if(wire->dst==UINT16_MAX){
          if(!wire->hops){ wire->hops=1;
            if(link_submit(link,L_SEND,page,(uint32_t)sizeof *wire)){ wire->hops=0; continue; }
            MOVE(page,SEND); pool_link[page]=(unsigned char)index; goto accepted;
          }
          RELEASE(page); goto accepted;
        }
        if(wire->hops==UINT16_MAX){ RELEASE(page); goto accepted; }
        wire->hops++;
        if(wire->dst!=(uint16_t)me){
          if(wire->hops>=hop_limit){ RELEASE(page); goto accepted; }
          int next=route_link(links,link_count,routes,wire->dst,index);
          if(next<0 || link_submit(&links[next],L_SEND,page,event.bytes)){ wire->hops--; continue; }
          MOVE(page,SEND); pool_link[page]=(unsigned char)next; COUNT(sent);
          goto accepted;
        }
        if(!client){
          size_t reply=mesh_resident_reply(wire,event.bytes,(uint16_t)me);
          if(reply){
            int next=route_link(links,link_count,routes,wire->src,-1);
            if(next<0 || link_submit(&links[next],L_SEND,page,(uint32_t)reply)){ wire->hops--; continue; }
            MOVE(page,SEND); pool_link[page]=(unsigned char)next; COUNT(sent);
            goto accepted;
          }
        }
        size_t frame_bytes=event.bytes-sizeof *wire;
        struct desc receive={.page=page,.bytes=(uint32_t)frame_bytes,.node=wire->src};
        if(push(M,CMP,&receive)){ wire->hops--; continue; }
        MOVE(page,APP); COUNT(recvd);
accepted:
        ring_erase(&link->completion.cursor,link->completion.entries,sizeof event,LINK_QUEUE,position);
        activity=1;
      }
      live+=link->up;
      stopped+=atomic_load(&link->stopped);
      if(!stop && link->up){
        if(monotime()-link->peer_seen>4){ link->up=0; atomic_store(&link->reset,1); }
        uint64_t queued=atomic_load(&link->completion.cursor.head)-atomic_load(&link->completion.cursor.tail);
        int receive_limit=receive_share<link->depth?receive_share:link->depth;
        for(int budget=0;budget<512 && counts[FREE]>link_count && link->up && link->receives<receive_limit &&
          queued+(uint64_t)link->receives<LINK_QUEUE-(uint64_t)link->depth-128;budget++){
          int page=free_pages[counts[FREE]-1];
          if(link_submit(link,L_RECV,(uint32_t)page,pg)) break;
          MOVE(page,RECV); pool_link[page]=(unsigned char)index; activity=1;
        }
        if(counts[FREE] && link->probe_page==UINT32_MAX && monotime()-link->probe_at>=1){
          int page=free_pages[counts[FREE]-1];
          struct wire *wire=(struct wire*)mesh_at(M,(uint32_t)page);
          *wire=(struct wire){(uint16_t)me,UINT16_MAX,0};
          if(!link_submit(link,L_SEND,(uint32_t)page,(uint32_t)sizeof *wire)){
            MOVE(page,SEND); pool_link[page]=(unsigned char)index; link->probe_at=monotime(); link->probe_page=(uint32_t)page;
          }
        }
      }
      atomic_store(&ports[index].phase,atomic_load(&link->phase));
      atomic_store(&ports[index].heartbeat_us,atomic_load(&link->heartbeat));
      atomic_store(&ports[index].operation,atomic_load(&link->operation));
      atomic_store(&ports[index].generation,link->accepted);
    }
    uint64_t position;
    if(client && !stop) for(int budget=0;budget<256 &&
      atomic_load(&M->r[ACK].head)-atomic_load(&M->r[ACK].tail)+(uint64_t)arena_pending<MESH_RING &&
      ring_select(&M->r[SUB],&submit_cursor,&position);budget++){
      descriptor=*slot(M,SUB,position); uint32_t page=descriptor.page;
      if(page<(uint32_t)pool || page>=(uint32_t)np || submitted[page-pool]){ COUNT(bad); goto submitted; }
      struct wire *wire=(struct wire*)mesh_at(M,page);
      *wire=(struct wire){(uint16_t)me,descriptor.node,0};
      uint32_t bytes=(uint32_t)sizeof *wire+(descriptor.bytes<mesh_pay(M)?descriptor.bytes:mesh_pay(M));
      int next=route_link(links,link_count,routes,descriptor.node,-1);
      if(next<0 || link_submit(&links[next],L_SEND,page,bytes)) continue;
      submitted[page-pool]=(unsigned char)(next+1); arena_pending++; COUNT(sent);
submitted:
      ring_erase(&M->r[SUB],slot(M,SUB,0),sizeof descriptor,MESH_RING,position);
      activity=1;
    }
    for(int i=0;i<NOWN;i++) add(&stats[i],counts[i]+(i==APP?counts[READY]:0));
    double stamp=now();
    if(stamp-telemetry>=0.25){
      flight_heartbeat(stop?MESH_STOPPING:live?MESH_PAIRED:MESH_PAIRING);
      if(client && kill((pid_t)who,0) && errno==ESRCH){
        uint64_t expected=who;
        if(atomic_compare_exchange_strong(&M->client,&expected,MESH_CLIENT_DRAIN)){
          draining=1; client=0;
          atomic_store(&M->r[SUB].tail,atomic_load(&M->r[SUB].head));
        }
      }
      for(int i=0;i<NOWN;i++){
        atomic_store(&M->mean[i],(uint64_t)stats[i].mean);
        atomic_store(&M->sd[i],(uint64_t)sqrt(stats[i].n>1?stats[i].m2/(stats[i].n-1):0)); stats[i]=(struct wf){0};
      }
      atomic_store(&M->up_ms,(uint64_t)((stamp-began)*1000)); telemetry=stamp;
    }
    if(draining && !arena_pending){
      for(int i=0;i<pool;i++) if(owner[i]==APP) RELEASE(i);
      for(int i=0;i<NRING;i++) atomic_store(&M->r[i].tail,atomic_load(&M->r[i].head));
      atomic_store_explicit(&M->client,0,memory_order_release); draining=0;
    }
    if(stop && stopped==link_count) break;
    idle_poll(activity,&awake_until);
  }
  for(int i=0;i<link_count;i++) pthread_join(links[i].thread,NULL);
  flight_heartbeat(MESH_STOPPED);
  for(int i=0;i<link_count;i++) free((void*)links[i].device);
  free(links); free(routes); free(free_pages); free(owner); free(pool_link); free(submitted);
  flight_status=NULL; munmap(M,d0+span);
  return 0;
}
