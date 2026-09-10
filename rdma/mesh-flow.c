
#include "mesh.h"
#include "mesh-dataflow.h"
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
struct mesh_verbs {
  struct ibv_context *context; struct ibv_pd *domain; struct ibv_cq *completion_queue;
  struct ibv_qp *pair; struct ibv_mr **regions;
  int region_count, send_capacity, receive_capacity;
  struct ibv_wc *completions; int completed;
  struct ibv_sge (*sges)[2];
  struct ibv_recv_wr *receives; struct ibv_send_wr *sends;
  int receiving, sending;
};
static _Thread_local struct mesh_verbs *provider;
// ../design/algorithm-sources.md#transport-page-addressing
static struct ibv_sge region_sge(const char *base, size_t offset, uint32_t bytes){
  uintptr_t address=(uintptr_t)base+offset;
  return (struct ibv_sge){address,bytes,provider->regions[address/CHUNK-(uintptr_t)base/CHUNK]->lkey}; }
static _Thread_local const char *shm; static _Atomic sig_atomic_t stop;
static _Thread_local int lsock=-1;
static _Thread_local uint64_t mynonce, peernonce;
static _Thread_local int retire_device, expected_peer=-1;
static _Thread_local const char *listen_address, *selected_device;
// ../design/algorithm-sources.md#transport-page-addressing
static int down_pair(void){
  if(provider->pair){ if(TRACE(DESTROY_QP,provider->pair,provider->pair->qp_num,0,ibv_destroy_qp(provider->pair))) return 0; provider->pair=0; return 0; }
  if(provider->completion_queue){ if(TRACE(DESTROY_CQ,provider->completion_queue,0,0,ibv_destroy_cq(provider->completion_queue))) return 0; provider->completion_queue=0; return 0; }
  return 1; }
// ../design/algorithm-sources.md#transport-page-addressing
static int down_verbs(void){
  if(!down_pair()) return 0;
  if(provider->region_count){ struct ibv_mr *r=provider->regions[provider->region_count-1];
    if(!TRACE(DEREG_MR,r,r->lkey,r->length,ibv_dereg_mr(r))) provider->region_count--;
    return 0; }
  free(provider->regions); provider->regions=0;
  if(provider->domain){ if(TRACE(DEALLOC_PD,provider->domain,0,0,ibv_dealloc_pd(provider->domain))) return 0; provider->domain=0; return 0; }
  if(provider->context){ if(TRACE(CLOSE_DEVICE,provider->context,0,0,ibv_close_device(provider->context))) return 0; provider->context=0; return 0; }
  return 1; }
// ../design/algorithm-sources.md#transport-page-addressing
static void down(void){ if(shm)shm_unlink(shm); }
// ../design/algorithm-sources.md#transport-page-addressing
static void die(const char*m){ fprintf(stderr,"%s\n",m); exit(1); }
// ../design/algorithm-sources.md#transport-page-addressing
static double now(void){ struct timeval t; gettimeofday(&t,NULL); return t.tv_sec+t.tv_usec/1e6; }
// ../design/algorithm-sources.md#transport-page-addressing
static double monotime(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
// ../design/algorithm-sources.md#transport-page-addressing
static void onsig(int s){ (void)s; stop=1; }

struct qpi { uint32_t xmagic, xsize; uint64_t nonce; uint32_t qpn,psn,pgsz,header_bytes; uint16_t lid; uint8_t gid[16]; uint16_t node; };
#define XMAGIC 0x4d585047u

// ../design/algorithm-sources.md#transport-page-addressing
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

// ../design/algorithm-sources.md#transport-page-addressing
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

// ../design/algorithm-sources.md#transport-page-addressing
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

// ../design/algorithm-sources.md#transport-page-addressing
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
// ../design/algorithm-sources.md#transport-page-addressing
static int verbs_up(const char *peer, char *mem, size_t span, int me){
  if(provider->context && (TRACE(QUERY_PORT,provider->context,1,0,ibv_query_port(provider->context,1,&pa)) || pa.state!=IBV_PORT_ACTIVE)){
    retire_device=1; return -1; }
  int f=oob(peer); if(f<0) return -1;
  if(!provider->context){
  struct ibv_device **dl=ibv_get_device_list(NULL);
  for(int i=0;dl&&dl[i];i++){
    if(selected_device && strcmp(selected_device,ibv_get_device_name(dl[i]))) continue;
    provider->context=TRACE(OPEN_DEVICE,dl[i],i,0,ibv_open_device(dl[i]));
    if(provider->context && !TRACE(QUERY_PORT,provider->context,1,0,ibv_query_port(provider->context,1,&pa)) && pa.state==IBV_PORT_ACTIVE) break;
    if(provider->context){
      if(TRACE(CLOSE_DEVICE,provider->context,0,0,ibv_close_device(provider->context))){
        retire_device=1; ibv_free_device_list(dl); close(f); return -1; }
      provider->context=0; } }
  if(dl) ibv_free_device_list(dl);
  if(!provider->context){ close(f); return -1; }
  }
  struct ibv_device_attr capabilities;
  if(ibv_query_device(provider->context,&capabilities)){ close(f); return -1; }
  if(capabilities.max_sge<2){ close(f); errno=EOPNOTSUPP; return -1; }
  if(!provider->domain) provider->domain=TRACE(ALLOC_PD,provider->context,0,0,ibv_alloc_pd(provider->context));
  if(!provider->domain){ close(f); return -1; }
  size_t head=(uintptr_t)mem%CHUNK, regions=(head+span+CHUNK-1)/CHUNK;
  if(!provider->regions) provider->regions=calloc(regions,sizeof *provider->regions);
  if(!provider->regions){ close(f); fprintf(stderr,"alloc regions: retrying\n"); return -1; }
  while((size_t)provider->region_count<regions){
    size_t o=provider->region_count?(size_t)provider->region_count*CHUNK-head:0, end=((size_t)provider->region_count+1)*CHUNK-head;
    size_t n=(end<span?end:span)-o;
    provider->regions[provider->region_count]=TRACE(REG_MR,mem+o,o,n,ibv_reg_mr(provider->domain,mem+o,n,IBV_ACCESS_LOCAL_WRITE));
    if(!provider->regions[provider->region_count]){ close(f); return -1; } provider->region_count++; }
  if(TRACE(QUERY_PORT,provider->context,1,0,ibv_query_port(provider->context,1,&pa))){ close(f); return -1; }
  { char c[96]; const char *dn=ibv_get_device_name(provider->context->device);
    snprintf(c,sizeof c,"ping6 -c 2 -i 0.2 ff02::1%%%s >/dev/null 2>&1",
             strncmp(dn,"rdma_",5)?dn:dn+5);
    system(c); }
  size_t frames=(((struct hdr*)mem)->pgsz+MESH_HEADER_BYTES+4095)/4096;
  int frame_capacity=capabilities.max_qp_wr<QD?capabilities.max_qp_wr:QD;
  int completions=2*(frame_capacity/(int)frames);
  if(!completions || capabilities.max_cqe<completions){ close(f); errno=EOPNOTSUPP; return -1; }
  provider->completion_queue=TRACE(CREATE_CQ,provider->context,completions,0,ibv_create_cq(provider->context,completions,NULL,NULL,0)); if(!provider->completion_queue){ close(f); return -1; }
  struct ibv_qp_init_attr qi={.send_cq=provider->completion_queue,.recv_cq=provider->completion_queue,.qp_type=IBV_QPT_UC,
    .cap={.max_send_wr=frame_capacity,.max_recv_wr=frame_capacity,.max_send_sge=2,.max_recv_sge=2}};
  provider->pair=TRACE(CREATE_QP,provider->domain,QD,0,ibv_create_qp(provider->domain,&qi)); if(!provider->pair){ close(f); return -1; }
  struct ibv_qp_attr queried; struct ibv_qp_init_attr actual;
  if(ibv_query_qp(provider->pair,&queried,IBV_QP_CAP,&actual)){ close(f); return -1; }
  provider->send_capacity=(int)((actual.cap.max_send_wr<(uint32_t)frame_capacity?actual.cap.max_send_wr:(uint32_t)frame_capacity)/frames);
  provider->receive_capacity=(int)((actual.cap.max_recv_wr<(uint32_t)frame_capacity?actual.cap.max_recv_wr:(uint32_t)frame_capacity)/frames);
  if(!provider->send_capacity || !provider->receive_capacity){ close(f); errno=EOPNOTSUPP; return -1; }
  struct ibv_qp_attr a={.qp_state=IBV_QPS_INIT,.port_num=1};
  if(TRACE(INIT,provider->pair,provider->pair->qp_num,0,ibv_modify_qp(provider->pair,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS))){ close(f); return -1; }
  union ibv_gid gid; if(TRACE(QUERY_GID,provider->context,1,0,ibv_query_gid(provider->context,1,0,&gid))){ close(f); return -1; }
  uint32_t psn=arc4random()&0xffffff;
  struct qpi mine={.xmagic=XMAGIC+MESH_VERSION,.xsize=sizeof mine,.nonce=mynonce,.qpn=provider->pair->qp_num,.psn=psn,.lid=pa.lid,.pgsz=((struct hdr*)mem)->pgsz,.header_bytes=MESH_HEADER_BYTES,.node=(uint16_t)me},you;
  memcpy(mine.gid,&gid,16);
  if(exchange(f,&mine,&you,monotime()+10)){ close(f); fprintf(stderr,"xchg retry\n"); return -1; }
  close(f);
  if(you.xmagic!=mine.xmagic || you.xsize!=sizeof you || you.pgsz!=mine.pgsz || you.header_bytes!=mine.header_bytes || (expected_peer>=0 && you.node!=expected_peer)){
    fprintf(stderr,"peer speaks a different exchange, retrying\n"); return -1; }
  if(you.nonce==mynonce){ fprintf(stderr,"self nonce, retry\n"); return -1; }
  peernonce=you.nonce; expected_peer=you.node;
  struct ibv_qp_attr r={.qp_state=IBV_QPS_RTR,.path_mtu=IBV_MTU_4096,.rq_psn=you.psn,
    .dest_qp_num=you.qpn,.ah_attr={.dlid=you.lid,.port_num=1,.is_global=1,
    .grh={.hop_limit=1,.sgid_index=0}}};
  memcpy(&r.ah_attr.grh.dgid,you.gid,16);
  int rc=TRACE(RTR,provider->pair,you.qpn,you.psn,ibv_modify_qp(provider->pair,&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN));
  if(rc){ fprintf(stderr,"rtr rc %d dlid %u dqpn %u dgid %02x%02x..%02x%02x mygid %02x%02x\n",
      rc, you.lid, you.qpn, you.gid[0],you.gid[1],you.gid[14],you.gid[15],
      mine.gid[0],mine.gid[15]); return -1; }
  struct ibv_qp_attr t={.qp_state=IBV_QPS_RTS,.sq_psn=psn};
  rc=TRACE(RTS,provider->pair,provider->pair->qp_num,psn,ibv_modify_qp(provider->pair,&t,IBV_QP_STATE|IBV_QP_SQ_PSN));
  if(rc){ fprintf(stderr,"rts rc %d, retrying\n",rc); return -1; }
  fprintf(stderr,"pair up: %s node %d\n",ibv_get_device_name(provider->context->device),me);
  return 0; }

#include "mesh-links.h"
// ../design/algorithm-sources.md#transport-page-addressing
int main(int argc,char**argv){
  const char *peer=NULL,*name=MESH_NAME; int me=0,link_count=0; double pct=80;
  struct mesh_link *links=calloc(LINK_LIMIT,sizeof *links);
  struct mesh_route *routes=calloc(65536,sizeof *routes);
  if(!links||!routes) die("link metadata");
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-I") && i+1<argc) me=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-M") && i+1<argc) pct=atof(argv[++i]);
    else if(!strcmp(argv[i],"-s") && i+1<argc) name=argv[++i];
    else if(!strcmp(argv[i],"--route") && i+1<argc){
      unsigned destination,index; char extra;
      if(sscanf(argv[++i],"%u:%u%c",&destination,&index,&extra)!=2 || destination>=65535 || index>=LINK_LIMIT) die("route descriptor");
      routes[destination].link=(uint16_t)index;
    } else if(!strcmp(argv[i],"--link") && i+1<argc){
      if(link_count==LINK_LIMIT) die("link descriptor capacity");
      char *fields=strdup(argv[++i]),*cursor=fields,*part[4];
      for(int j=0;j<4;j++) part[j]=strsep(&cursor,",");
      if(!part[0]||!part[1]||!part[2]||!part[3]||cursor) die("link requires device,peer,local-address,peer-address");
      char *end; unsigned long node=strtoul(part[1],&end,10);
      if(*end || node>=65535) die("link peer identity");
      struct mesh_link *link=&links[link_count];
      link->device=part[0]; link->peer_node=(int)node; link->local=part[2]; link->peer=part[3];
      routes[node].link=(uint16_t)link_count++;
    } else if(argv[i][0]=='-') die("unknown bridge option");
    else peer=argv[i];
  }
  if(me<0 || me>=65535 || !isfinite(pct) || pct<=0 || pct>100) die("bridge geometry");
  if(!link_count){ link_count=1; links[0].peer=peer; links[0].peer_node=-1; }
  for(int i=0;i<65536;i++) if(routes[i].link>=link_count) die("route names absent link");
  atexit(down); struct sigaction sa={0}; sa.sa_handler=onsig;
  sigaction(SIGINT,&sa,NULL); sigaction(SIGTERM,&sa,NULL); sigaction(SIGHUP,&sa,NULL); signal(SIGPIPE,SIG_IGN);
  uint64_t ram=0; size_t rl=sizeof ram; sysctlbyname("hw.memsize",&ram,&rl,NULL,0);
  const uint32_t pg=(uint32_t)getpagesize();
  uint64_t wanted=(uint64_t)(pct/100*(double)ram/(pg+MESH_HEADER_STRIDE));
  if(wanted<NOWN || wanted>INT32_MAX) die("page index capacity");
  int np=(int)wanted,pool=np/NOWN,receive_share=pool/link_count;
  size_t h0=(RINGS+NRING*MESH_RING*sizeof(struct desc)+LINK_LIMIT*sizeof(struct mesh_port_info)+pg-1)/pg*pg;
  size_t d0=(h0+(size_t)np*MESH_HEADER_STRIDE+pg-1)/pg*pg,span=(size_t)pg*np;
  shm_unlink(name); int fd=shm_open(name,O_CREAT|O_RDWR,MESH_MODE); if(fd<0) die("shm");
  if(ftruncate(fd,(off_t)(d0+span))) die("ftruncate"); fchmod(fd,MESH_MODE);
  struct hdr *M=mmap(NULL,d0+span,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0); close(fd);
  if(M==MAP_FAILED) die("mmap"); shm=name;
  M->pgsz=pg; M->pool=(uint32_t)pool; M->arena=(uint32_t)(np-pool); M->node=(uint32_t)me;
  M->version=MESH_VERSION; M->headers_off=h0; M->data_off=d0; atomic_store(&M->port_count,(uint64_t)link_count);
  __sync_synchronize(); M->magic=MESH_MAGIC;
  int *free_pages=malloc((size_t)pool*sizeof *free_pages);
  unsigned char *owner=calloc((size_t)pool,1),*pool_link=calloc((size_t)pool,1);
  if(!free_pages||!owner||!pool_link) die("page ownership metadata");
  int counts[NOWN]={0}; counts[FREE]=pool;
  for(int i=0;i<pool;i++) free_pages[i]=i;
  struct mesh_port_info *ports=mesh_ports(M);
  flight_status=M; atomic_store(&M->bridge_pid,(uint64_t)getpid());
  for(int i=0;i<link_count;i++){
    links[i].node=me; links[i].name=name; links[i].pages=M;
    snprintf(ports[i].device,sizeof ports[i].device,"%s",links[i].device?links[i].device:"automatic");
    ports[i].peer=(uint16_t)links[i].peer_node;
    int error=pthread_create(&links[i].thread,NULL,link_worker,&links[i]);
    if(error){ errno=error; perror("link worker"); atomic_store(&links[i].stopped,1); stop=1; }
  }
  double began=now(),telemetry=began; int arena_pending=0;
  uint64_t submit_cursor=0;
  #define COUNT(field) atomic_fetch_add_explicit(&M->field,1,memory_order_relaxed)
  #define MOVE(page,to) do{ counts[owner[page]]--; owner[page]=(to); counts[to]++; }while(0)
  #define RELEASE(page) do{ MOVE(page,FREE); free_pages[counts[FREE]-1]=(page); }while(0)
  for(;;){
    int live=0,stopped=0; struct desc descriptor;
    for(int index=0;index<link_count;index++){
      struct mesh_link *link=&links[index]; struct mesh_verbs *v=&link->provider;
      if(atomic_exchange(&ports[index].reset_request,0)) atomic_store(&link->reset,1);
      int ownership=atomic_load_explicit(&link->ownership,memory_order_acquire);
      if(ownership==LINK_ACTIVE){
        if(stop || atomic_load(&link->reset)){
          link->up=0; ownership=LINK_RELEASED;
          atomic_store_explicit(&link->ownership,ownership,memory_order_release);
        } else if(!link->faulted){
          link->up=1; ports[index].peer=(uint16_t)link->peer_node;
        }
      }
      if(link->up && v->completed<v->send_capacity+v->receive_capacity){
        int count=ibv_poll_cq(v->completion_queue,v->send_capacity+v->receive_capacity-v->completed,v->completions+v->completed);
        if(count<0){
          ports[index].when=flight_time(); ports[index].code=count; ports[index].domain=3;
          COUNT(bad); link->up=0; link->faulted=1;
        } else v->completed+=count;
      }
      int accessible=ownership==LINK_ACTIVE || ownership==LINK_RELEASED || ownership==LINK_RETIRED;
      for(int position=0;accessible && position<v->completed;){
        struct ibv_wc *wc=&v->completions[position];
        int receive=(int)(wc->wr_id>>63),post=wc->opcode==(enum ibv_wc_opcode)-1;
        uint64_t offset=receive?M->headers_off+(size_t)(uint32_t)wc->wr_id*MESH_HEADER_STRIDE:wc->wr_id;
        struct mesh_send *record=(struct mesh_send*)((char*)M+offset);
        uint32_t page=receive?(uint32_t)wc->wr_id:record->page;
        uint32_t error=post?wc->vendor_err:wc->status,domain=post?1:wc->status?2:0;
        if(error){
          struct mesh_page_header *header=(struct mesh_page_header*)((char*)M+offset);
          header->when=flight_time(); header->code=error; header->domain=domain;
          header->index=page; header->peer=(uint32_t)link->peer_node;
          if(receive) header->function=UINT32_MAX;
        }
        if(!receive){
          struct desc ack={.page=page,.header=offset,.error=error,.domain=domain};
          if(offset>=M->data_off){
            if(atomic_load(&M->r[ACK].head)-atomic_load(&M->r[ACK].tail)>=MESH_RING){ position++; continue; }
            link_release(link,offset); arena_pending--; push(M,ACK,&ack);
          } else { RELEASE(page); link_release(link,offset); }
          goto accepted;
        }
        if(error){
          struct desc failure={.page=page,.header=offset,.error=error,.domain=domain};
          if(push(M,CMP,&failure)){ position++; continue; }
          MOVE(page,APP); goto accepted;
        }
        struct mesh_page_header *header=mesh_header(M,page);
        uint32_t bytes=wc->byte_len-MESH_HEADER_BYTES;
        if(header->wire.dst!=(uint16_t)me){
          if(stop){ header->when=flight_time(); header->code=ECANCELED; header->domain=1; RELEASE(page); goto accepted; }
          int next=routes[header->wire.dst].link;
          if(link_submit(&links[next],L_SEND,page,offset)){ position++; continue; }
          MOVE(page,SEND); pool_link[page]=(unsigned char)next; COUNT(sent);
        } else {
          struct desc receive={.page=page,.header=offset,.bytes=bytes,.node=header->wire.src};
          if(push(M,CMP,&receive)){ position++; continue; }
          MOVE(page,APP); COUNT(recvd);
        }
accepted:
        if(receive) link->receives--;
        v->completions[position]=v->completions[--v->completed];
      }
      if(ownership==LINK_RETIRED && !v->completed){
        for(int i=0;i<pool;i++) if(pool_link[i]==index && owner[i]==RECV) RELEASE(i);
        while(link->pending){
          uint64_t offset=link->pending;
          struct mesh_send *record=(struct mesh_send*)((char*)M+offset);
          if(offset>=M->data_off){
            struct desc ack={.page=record->page,.header=offset,.error=ECANCELED,.domain=1};
            if(atomic_load(&M->r[ACK].head)-atomic_load(&M->r[ACK].tail)>=MESH_RING) break;
            record->header.when=flight_time(); record->header.code=ECANCELED; record->header.domain=1;
            link_release(link,offset); arena_pending--; push(M,ACK,&ack);
          } else { RELEASE(record->page); link_release(link,offset); }
        }
        if(!link->pending){
          link->receives=0; link->faulted=0;
          atomic_store_explicit(&link->ownership,LINK_ACKNOWLEDGED,memory_order_release);
        }
      }
      live+=link->up; stopped+=atomic_load(&link->stopped);
      int receive_limit=link->up?(receive_share<v->receive_capacity?receive_share:v->receive_capacity):0;
      while(!stop && counts[FREE] && link->receives<receive_limit){
        int page=free_pages[counts[FREE]-1];
        if(link_submit(link,L_RECV,(uint32_t)page,M->headers_off+(size_t)page*MESH_HEADER_STRIDE)) break;
        MOVE(page,RECV); pool_link[page]=(unsigned char)index;
      }
      atomic_store(&ports[index].phase,atomic_load(&link->phase));
      atomic_store(&ports[index].heartbeat_us,atomic_load(&link->heartbeat));
      atomic_store(&ports[index].operation,atomic_load(&link->operation));
      atomic_store(&ports[index].generation,link->generation);
    }
    uint64_t position;
    uint64_t available=atomic_load_explicit(&M->r[SUB].head,memory_order_acquire)-atomic_load_explicit(&M->r[SUB].tail,memory_order_relaxed);
    for(uint64_t budget=0;budget<available && !stop &&
      atomic_load(&M->r[ACK].head)-atomic_load(&M->r[ACK].tail)+(uint64_t)arena_pending<MESH_RING &&
      ring_select(&M->r[SUB],&submit_cursor,&position);budget++){
      descriptor=*slot(M,SUB,position); uint32_t page=descriptor.page;
      struct mesh_page_header *header=(struct mesh_page_header*)((char*)M+descriptor.header);
      header->wire=(struct wire){.src=(uint16_t)me,.dst=descriptor.node};
      int next=routes[descriptor.node].link;
      if(link_submit(&links[next],L_SEND,page,descriptor.header)) continue;
      arena_pending++; COUNT(sent);
      ring_erase(&M->r[SUB],slot(M,SUB,0),sizeof descriptor,MESH_RING,position);
    }
    size_t capacity=0;
    for(int index=0;index<link_count;index++){
      link_flush(&links[index]);
      if(links[index].up) capacity+=(size_t)links[index].provider.send_capacity+links[index].provider.receive_capacity;
    }
    uint64_t released=atomic_load_explicit(&M->r[REL].head,memory_order_acquire)-atomic_load_explicit(&M->r[REL].tail,memory_order_relaxed);
    size_t retire=capacity?capacity:1;
    for(size_t index=0;index<released && index<retire && !pop(M,REL,&descriptor);index++){
      memset(mesh_at(M,descriptor.page),0,pg);
      struct mesh_row *row=(struct mesh_row*)((char*)M+descriptor.header);
      uint64_t stamp=__atomic_load_n(&row->stamp,__ATOMIC_RELAXED);
      __atomic_store_n(&row->page,MESH_ROW_ABSENT,__ATOMIC_RELEASE);
      __atomic_store_n(&row->stamp,stamp&~MESH_ROW_WRITING,__ATOMIC_RELEASE);
      if(descriptor.page<(uint32_t)pool) RELEASE(descriptor.page);
    }
    double stamp=now();
    if(stamp-telemetry>=0.25){
      flight_heartbeat(stop?MESH_STOPPING:live?MESH_PAIRED:MESH_PAIRING);
      for(int i=0;i<NOWN;i++) atomic_store(&M->mean[i],counts[i]);
      atomic_store(&M->up_ms,(uint64_t)((stamp-began)*1000)); telemetry=stamp;
    }
    if(stop && stopped==link_count) break;
  }
  for(int i=0;i<link_count;i++) pthread_join(links[i].thread,NULL);
  flight_heartbeat(MESH_STOPPED);
  for(int i=0;i<link_count;i++) free((void*)links[i].device);
  free(links); free(routes); free(free_pages); free(owner); free(pool_link);
  flight_status=NULL; munmap(M,d0+span); return 0;
}
