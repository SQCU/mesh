#pragma once
#include "mesh.h"
#include "mesh-dataflow.h"
#include "mesh-transport.h"
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

#define QD 4095
struct mesh_verbs {
  struct ibv_context *context; struct ibv_pd *domain; struct ibv_cq *completion_queue;
  struct ibv_qp *pair; struct ibv_mr **regions;
  int region_count, send_capacity, receive_capacity;
  unsigned region_shift;
  struct ibv_wc *completions; int completed;
  struct ibv_sge (*sges)[2];
  struct ibv_recv_wr *receives; struct ibv_send_wr *sends;
  int receiving, sending;
};
static _Thread_local struct mesh_verbs *provider;
// ../design/algorithm-sources.md#transport-page-addressing
static struct ibv_sge region_sge(const char *base, size_t offset, uint32_t bytes){
  uintptr_t address=(uintptr_t)base+offset;
  return (struct ibv_sge){address,bytes,provider->regions[(address>>provider->region_shift)-((uintptr_t)base>>provider->region_shift)]->lkey}; }
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
#ifdef MESH_TRANSPORT_TIMING
static uint64_t measured_passes, measured_polls;
static double measured_pass_seconds, measured_poll_seconds;
#endif
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
  size_t sent=0,got=0,send_bytes=mine?sizeof *mine:0,receive_bytes=you?sizeof *you:0;
  fcntl(f,F_SETFL,O_NONBLOCK);
  while(!stop && (sent<send_bytes || got<receive_bytes)){
    double left=deadline-monotime(); if(left<=0) return -1;
    fd_set r,w; FD_ZERO(&r); FD_ZERO(&w);
    if(got<receive_bytes) FD_SET(f,&r);
    if(sent<send_bytes) FD_SET(f,&w);
    struct timeval tv={.tv_sec=(int)left,.tv_usec=(int)((left-(int)left)*1e6)};
    int ready=select(f+1,&r,&w,0,&tv);
    if(ready<0 && errno==EINTR) continue;
    if(ready<=0) return -1;
    if(FD_ISSET(f,&w)){
      ssize_t n=write(f,(const char*)mine+sent,send_bytes-sent);
      if(n>0) sent+=(size_t)n;
      else if(!n || (errno!=EAGAIN && errno!=EINTR)) return -1; }
    if(FD_ISSET(f,&r)){
      ssize_t n=read(f,(char*)you+got,receive_bytes-got);
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
static int verbs_up(const char *peer, char *mem, size_t span, int me, uint32_t page_bytes, uint32_t header_bytes, struct ibv_recv_wr *initial_receives){
  if(provider->context && (TRACE(QUERY_PORT,provider->context,1,0,ibv_query_port(provider->context,1,&pa)) || pa.state!=IBV_PORT_ACTIVE)){
    retire_device=1; return -1; }
  int f=oob(peer); if(f<0) return -1;
  fprintf(stderr,"pair setup node=%d connected=%.6f\n",me,monotime());
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
  if(!provider->domain) provider->domain=TRACE(ALLOC_PD,provider->context,0,0,ibv_alloc_pd(provider->context));
  if(!provider->domain){ close(f); return -1; }
  if(capabilities.max_mr<1){ close(f); errno=EOPNOTSUPP; return -1; }
  if(!provider->regions){
    provider->region_shift=30;
    while((((uintptr_t)mem+span-1)>>provider->region_shift)-((uintptr_t)mem>>provider->region_shift)+1>(size_t)capabilities.max_mr)
      provider->region_shift++;
  }
  size_t extent=(size_t)1<<provider->region_shift;
  size_t head=(uintptr_t)mem&(extent-1), regions=(head+span+extent-1)>>provider->region_shift;
  if(!provider->regions) provider->regions=calloc(regions,sizeof *provider->regions);
  if(!provider->regions){ close(f); fprintf(stderr,"alloc regions: retrying\n"); return -1; }
  while((size_t)provider->region_count<regions){
    size_t o=provider->region_count?((size_t)provider->region_count<<provider->region_shift)-head:0;
    size_t end=((size_t)provider->region_count+1)*extent-head;
    size_t n=(end<span?end:span)-o;
    provider->regions[provider->region_count]=TRACE(REG_MR,mem+o,o,n,ibv_reg_mr(provider->domain,mem+o,n,IBV_ACCESS_LOCAL_WRITE));
    if(!provider->regions[provider->region_count]){ close(f); return -1; } provider->region_count++; }
  if(TRACE(QUERY_PORT,provider->context,1,0,ibv_query_port(provider->context,1,&pa))){ close(f); return -1; }
  { char c[96]; const char *dn=ibv_get_device_name(provider->context->device);
    snprintf(c,sizeof c,"ping6 -c 2 -i 0.2 ff02::1%%%s >/dev/null 2>&1",
             strncmp(dn,"rdma_",5)?dn:dn+5);
    system(c); }
  size_t frames=(page_bytes+header_bytes+4095)/4096;
  int frame_capacity=capabilities.max_qp_wr<QD?capabilities.max_qp_wr:QD;
  int completions=4*(frame_capacity/(int)frames);
  if(!completions || capabilities.max_cqe<completions){ close(f); errno=EOPNOTSUPP; return -1; }
  provider->completion_queue=TRACE(CREATE_CQ,provider->context,completions,0,ibv_create_cq(provider->context,completions,NULL,NULL,0)); if(!provider->completion_queue){ close(f); return -1; }
  struct ibv_qp_init_attr qi={.send_cq=provider->completion_queue,.recv_cq=provider->completion_queue,.qp_type=IBV_QPT_UC,
    .cap={.max_send_wr=frame_capacity,.max_recv_wr=frame_capacity,.max_send_sge=1,.max_recv_sge=1}};
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
  struct qpi mine={.xmagic=XMAGIC+MESH_VERSION,.xsize=sizeof mine,.nonce=mynonce,.qpn=provider->pair->qp_num,.psn=psn,.lid=pa.lid,.pgsz=page_bytes,.header_bytes=header_bytes,.node=(uint16_t)me},you;
  memcpy(mine.gid,&gid,16);
  fprintf(stderr,"pair setup node=%d exchange=%.6f regions=%d qpn=%u\n",me,monotime(),provider->region_count,mine.qpn);
  double exchange_deadline=monotime()+10;
  if(exchange(f,initial_receives?NULL:&mine,&you,exchange_deadline)){ close(f); fprintf(stderr,"xchg retry\n"); return -1; }
  if(you.xmagic!=mine.xmagic || you.xsize!=sizeof you || you.pgsz!=mine.pgsz || you.header_bytes!=mine.header_bytes || (expected_peer>=0 && you.node!=expected_peer)){
    fprintf(stderr,"peer speaks a different exchange, retrying\n"); close(f); return -1; }
  if(you.nonce==mynonce){ fprintf(stderr,"self nonce, retry\n"); close(f); return -1; }
  peernonce=you.nonce; expected_peer=you.node;
  struct ibv_qp_attr r={.qp_state=IBV_QPS_RTR,.path_mtu=IBV_MTU_4096,.rq_psn=you.psn,
    .dest_qp_num=you.qpn,.ah_attr={.dlid=you.lid,.port_num=1,.is_global=1,
    .grh={.hop_limit=1,.sgid_index=0}}};
  memcpy(&r.ah_attr.grh.dgid,you.gid,16);
  int rc=TRACE(RTR,provider->pair,you.qpn,you.psn,ibv_modify_qp(provider->pair,&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN));
  if(rc){ fprintf(stderr,"rtr rc %d dlid %u dqpn %u dgid %02x%02x..%02x%02x mygid %02x%02x\n",
      rc, you.lid, you.qpn, you.gid[0],you.gid[1],you.gid[14],you.gid[15],
      mine.gid[0],mine.gid[15]); close(f); return -1; }
  struct ibv_qp_attr t={.qp_state=IBV_QPS_RTS,.sq_psn=psn};
  rc=TRACE(RTS,provider->pair,provider->pair->qp_num,psn,ibv_modify_qp(provider->pair,&t,IBV_QP_STATE|IBV_QP_SQ_PSN));
  if(rc){ fprintf(stderr,"rts rc %d, retrying\n",rc); close(f); return -1; }
  if(initial_receives){
    struct ibv_recv_wr *last=initial_receives,*unposted=NULL;
    int count=1;
    for(;;){
      for(int i=0;i<last->num_sge;i++){
        struct ibv_sge *span=&last->sg_list[i];
        span->lkey=region_sge(mem,(size_t)(span->addr-(uintptr_t)mem),span->length).lkey;
      }
      if(!last->next || count==provider->receive_capacity) break;
      last=last->next; count++;
    }
    struct ibv_recv_wr *next=last->next; last->next=NULL;
    int error=ibv_post_recv(provider->pair,initial_receives,&unposted);
    last->next=next;
    if(error){ fprintf(stderr,"initial receive post=%d\n",error); close(f); errno=error; return -1; }
  }
  if(initial_receives && exchange(f,&mine,NULL,exchange_deadline)){
    close(f); fprintf(stderr,"xchg retry\n"); return -1;
  }
  close(f);
  fprintf(stderr,"pair up: %s node %d\n",ibv_get_device_name(provider->context->device),me);
  return 0; }
