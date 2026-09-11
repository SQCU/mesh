#pragma once
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
#include <time.h>
#include <limits.h>

#define QD 4095
#define MESH_QPS 8
struct mesh_verbs {
  struct ibv_context *context; struct ibv_pd *domain; struct ibv_cq *completion_queue;
  struct ibv_qp *pair,*pairs[MESH_QPS]; int qp_count; struct ibv_mr **regions;
  int region_count, send_capacity, receive_capacity;
  size_t region_origin, region_extent;
  struct ibv_wc *completions; int completed;
  struct ibv_sge (*sges)[2];
  struct ibv_recv_wr *receives; struct ibv_send_wr *sends;
  int receiving, sending;
};
static struct mesh_verbs *provider;
/* design/algorithm-sources.md#regions-follow-blocks */
static struct ibv_sge region_sge(const char *base, size_t offset, uint32_t bytes){
  size_t index=offset<provider->region_origin?0:(provider->region_origin?1:0)+(offset-provider->region_origin)/provider->region_extent;
  return (struct ibv_sge){(uintptr_t)base+offset,bytes,provider->regions[index]->lkey}; }
static const char *shm; static _Atomic sig_atomic_t stop;
static int lsock=-1;
static int expected_peer=-1;
static const char *listen_address, *selected_device;
static int down_pair(void){
  while(provider->qp_count){ struct ibv_qp *q=provider->pairs[provider->qp_count-1]; if(q && ibv_destroy_qp(q)) return 0; provider->pairs[--provider->qp_count]=0; provider->pair=0; }
  if(provider->completion_queue){ if(ibv_destroy_cq(provider->completion_queue)) return 0; provider->completion_queue=0; }
  return 1; }
static int down_verbs(void){
  if(!down_pair()) return 0;
  while(provider->region_count){ struct ibv_mr *r=provider->regions[provider->region_count-1];
    if(ibv_dereg_mr(r)) return 0;
    provider->region_count--; }
  free(provider->regions); provider->regions=0;
  if(provider->domain){ if(ibv_dealloc_pd(provider->domain)) return 0; provider->domain=0; }
  if(provider->context){ if(ibv_close_device(provider->context)) return 0; provider->context=0; }
  return 1; }
static void down(void){ if(shm)shm_unlink(shm); }
static void die(const char*m){ fprintf(stderr,"%s\n",m); exit(1); }
static double monotime(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
static void onsig(int s){ (void)s; stop++; }

struct qpi { uint32_t xmagic, xsize; uint32_t qpn,psn,pgsz,header_bytes; uint16_t lid; uint8_t gid[16]; uint16_t node; uint32_t count,qpns[MESH_QPS],psns[MESH_QPS]; };
#define XMAGIC 0x4d585047u

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

static int oob(const char *peer){
  if(!peer){
    fd_set reads; FD_ZERO(&reads); FD_SET(lsock,&reads);
    if(select(lsock+1,&reads,NULL,NULL,NULL)<1) return -1;
  }
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

static struct ibv_port_attr pa;
static int initial_receive_post(char *mem,struct ibv_recv_wr *initial_receives,int parts){
    struct ibv_recv_wr *last=initial_receives,*unposted=NULL;
    int count=1;
    for(;;){
      for(int i=0;i<last->num_sge;i++){
        struct ibv_sge *span=&last->sg_list[i];
        span->lkey=region_sge(mem,(size_t)(span->addr-(uintptr_t)mem),span->length).lkey;
      }
      if(!last->next || count==provider->receive_capacity*parts) break;
      last=last->next; count++;
    }
    struct ibv_recv_wr *next=last->next; last->next=NULL;
    int error=ibv_post_recv(provider->pair,initial_receives,&unposted);
    last->next=next;
    provider->receiving=count;
    if(error){ fprintf(stderr,"initial receive post=%d\n",error); errno=error; return -1; }
  return 0;
}
static int verbs_up(const char *peer, char *mem, size_t span, size_t origin, int me, uint32_t message_bytes, struct ibv_recv_wr *initial_receives, int qps){
  if(qps<1 || qps>MESH_QPS){ errno=EINVAL; return -1; }
  if(provider->context && (ibv_query_port(provider->context,1,&pa) || pa.state!=IBV_PORT_ACTIVE)){
    return -1; }
  int f=oob(peer); if(f<0) return -1;
  fprintf(stderr,"pair setup node=%d connected=%.6f\n",me,monotime());
  if(!provider->context){
  struct ibv_device **dl=ibv_get_device_list(NULL);
  for(int i=0;dl&&dl[i];i++){
    if(selected_device && strcmp(selected_device,ibv_get_device_name(dl[i]))) continue;
    provider->context=ibv_open_device(dl[i]);
    if(provider->context && !ibv_query_port(provider->context,1,&pa) && pa.state==IBV_PORT_ACTIVE) break;
    if(provider->context){
      if(ibv_close_device(provider->context)){
        ibv_free_device_list(dl); close(f); return -1; }
      provider->context=0; } }
  if(dl) ibv_free_device_list(dl);
  if(!provider->context){ close(f); return -1; }
  }
  struct ibv_device_attr capabilities;
  if(ibv_query_device(provider->context,&capabilities)){ close(f); return -1; }
  if(!provider->domain) provider->domain=ibv_alloc_pd(provider->context);
  if(!provider->domain){ close(f); return -1; }
  if(capabilities.max_mr<1){ close(f); errno=EOPNOTSUPP; return -1; }
  /* design/algorithm-sources.md#regions-follow-blocks */
  if(!provider->regions){
    size_t bank=(size_t)1<<32, extent;
    if(!(message_bytes&(message_bytes-1))){ origin=0; extent=(size_t)1<<30; }
    else {
      extent=((size_t)1<<30)/message_bytes*message_bytes;
      if(((uintptr_t)mem&(bank-1)) || span>bank){ fprintf(stderr,"a %u-byte block is not a power of two: the mapping must be bank-aligned and within one 4 GiB bank\n",message_bytes); errno=EINVAL; close(f); return -1; }
    }
    while((origin?1:0)+(span-origin+extent-1)/extent>(size_t)capabilities.max_mr){ errno=ENOMEM; close(f); return -1; }
    provider->region_origin=origin; provider->region_extent=extent;
  }
  size_t regions=(origin?1:0)+(span-origin+provider->region_extent-1)/provider->region_extent;
  if(!provider->regions) provider->regions=calloc(regions,sizeof *provider->regions);
  if(!provider->regions){ close(f); fprintf(stderr,"alloc regions: failed\n"); return -1; }
  while((size_t)provider->region_count<regions){
    int data=provider->region_count>=(origin?1:0);
    size_t o=data?origin+((size_t)provider->region_count-(origin?1:0))*provider->region_extent:0;
    size_t end=data?o+provider->region_extent:origin;
    size_t n=(end<span?end:span)-o;
    provider->regions[provider->region_count]=ibv_reg_mr(provider->domain,mem+o,n,IBV_ACCESS_LOCAL_WRITE);
    if(!provider->regions[provider->region_count]){ close(f); return -1; } provider->region_count++; }
  if(ibv_query_port(provider->context,1,&pa)){ close(f); return -1; }
  size_t frames=(message_bytes+4095)/4096;
  int frame_capacity=capabilities.max_qp_wr<QD?capabilities.max_qp_wr:QD;
  int completions=4*(frame_capacity/(int)frames)*qps;
  if(!completions || capabilities.max_cqe<completions){ close(f); errno=EOPNOTSUPP; return -1; }
  provider->completion_queue=ibv_create_cq(provider->context,completions,NULL,NULL,0); if(!provider->completion_queue){ close(f); return -1; }
  struct ibv_qp_init_attr qi={.send_cq=provider->completion_queue,.recv_cq=provider->completion_queue,.qp_type=IBV_QPT_UC,
    .cap={.max_send_wr=frame_capacity,.max_recv_wr=frame_capacity,.max_send_sge=1,.max_recv_sge=1}};
  for(int q=0;q<qps;q++){ provider->pairs[q]=ibv_create_qp(provider->domain,&qi); if(!provider->pairs[q]){ close(f); return -1; } provider->qp_count=q+1; }
  provider->pair=provider->pairs[0];
  struct ibv_qp_attr queried; struct ibv_qp_init_attr actual;
  if(ibv_query_qp(provider->pair,&queried,IBV_QP_CAP,&actual)){ close(f); return -1; }
  provider->send_capacity=(int)((actual.cap.max_send_wr<(uint32_t)frame_capacity?actual.cap.max_send_wr:(uint32_t)frame_capacity)/frames);
  provider->receive_capacity=(int)((actual.cap.max_recv_wr<(uint32_t)frame_capacity?actual.cap.max_recv_wr:(uint32_t)frame_capacity)/frames);
  if(!provider->send_capacity || !provider->receive_capacity){ close(f); errno=EOPNOTSUPP; return -1; }
  struct ibv_qp_attr a={.qp_state=IBV_QPS_INIT,.port_num=1};
  for(int q=0;q<qps;q++) if(ibv_modify_qp(provider->pairs[q],&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)){ close(f); return -1; }
  union ibv_gid gid; if(ibv_query_gid(provider->context,1,0,&gid)){ close(f); return -1; }
  uint32_t psn=arc4random()&0xffffff;
  struct qpi mine={.xmagic=XMAGIC+MESH_VERSION,.xsize=sizeof mine,.qpn=provider->pair->qp_num,.psn=psn,.lid=pa.lid,.pgsz=message_bytes,.header_bytes=0,.node=(uint16_t)me,.count=(uint32_t)qps},you;
  for(int q=0;q<qps;q++){ mine.qpns[q]=provider->pairs[q]->qp_num; mine.psns[q]=(psn+(uint32_t)q)&0xffffff; }
  memcpy(mine.gid,&gid,16);
  fprintf(stderr,"pair setup node=%d exchange=%.6f regions=%d qpn=%u\n",me,monotime(),provider->region_count,mine.qpn);
  double exchange_deadline=monotime()+10;
  if(exchange(f,peer?NULL:&mine,&you,exchange_deadline)){ close(f); fprintf(stderr,"exchange failed\n"); return -1; }
  if(you.xmagic!=mine.xmagic || you.xsize!=sizeof you || you.pgsz!=mine.pgsz || you.header_bytes!=mine.header_bytes || you.count!=mine.count || (expected_peer>=0 && you.node!=expected_peer)){
    fprintf(stderr,"exchange mismatch: local=%u,%u,%u,%u,%u,%u peer=%u,%u,%u,%u,%u,%u expected_node=%d\n",mine.xmagic,mine.xsize,mine.pgsz,mine.header_bytes,mine.count,mine.node,you.xmagic,you.xsize,you.pgsz,you.header_bytes,you.count,you.node,expected_peer); close(f); return -1; }
  expected_peer=you.node;
  for(int q=0;q<qps;q++){
    struct ibv_qp_attr r={.qp_state=IBV_QPS_RTR,.path_mtu=IBV_MTU_4096,.rq_psn=you.psns[q],
      .dest_qp_num=you.qpns[q],.ah_attr={.dlid=you.lid,.port_num=1,.is_global=1,
      .grh={.hop_limit=1,.sgid_index=0}}};
    memcpy(&r.ah_attr.grh.dgid,you.gid,16);
    int rc=ibv_modify_qp(provider->pairs[q],&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN);
    if(rc){ fprintf(stderr,"rtr %d rc %d dlid %u dqpn %u\n",q,rc,you.lid,you.qpns[q]); close(f); return -1; }
    struct ibv_qp_attr t={.qp_state=IBV_QPS_RTS,.sq_psn=mine.psns[q]};
    rc=ibv_modify_qp(provider->pairs[q],&t,IBV_QP_STATE|IBV_QP_SQ_PSN);
    if(rc){ fprintf(stderr,"rts %d rc %d, failed\n",q,rc); close(f); return -1; }
  }
  if(initial_receives && initial_receive_post(mem,initial_receives,1)){ close(f); return -1; }
  if(peer && exchange(f,&mine,NULL,exchange_deadline)){ close(f); return -1; }
  close(f);
  fprintf(stderr,"pair up: %s node %d\n",ibv_get_device_name(provider->context->device),me);
  return 0; }
