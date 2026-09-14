#pragma once
#include "mesh.h"
#include <infiniband/verbs.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
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
struct mesh_verbs {
  struct ibv_context *context; struct ibv_pd *domain; struct ibv_cq *completion_queues[2*(MESH_QPS+1)];
  struct ibv_qp *pair,*pairs[MESH_QPS+1]; int qp_count; struct ibv_mr **regions;
  int region_count; uint32_t capacity[MESH_QPS+1][2];
  size_t region_origin, region_extent;
  struct ibv_wc *completions;
};
static struct mesh_verbs *provider;
/* design/algorithm-sources.md#programtensor */
static struct ibv_sge region_sge(const char *base, size_t offset, uint32_t bytes){
  size_t index=offset<provider->region_origin?0:(provider->region_origin?1:0)+(offset-provider->region_origin)/provider->region_extent;
  return (struct ibv_sge){(uintptr_t)base+offset,bytes,provider->regions[index]->lkey}; }
static const char *shm; static _Atomic sig_atomic_t stop;
static int lsock=-1;
static int expected_peer=-1;
static const char *listen_address, *selected_device;
static int down_pair(void){
  while(provider->qp_count){ struct ibv_qp *q=provider->pairs[provider->qp_count-1]; if(q && ibv_destroy_qp(q)) return 0; provider->pairs[--provider->qp_count]=0; provider->pair=0; }
  for(int i=0;i<2*(MESH_QPS+1);i++)if(provider->completion_queues[i]){
    if(ibv_destroy_cq(provider->completion_queues[i]))return 0;
    provider->completion_queues[i]=0;
  }
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

/* ledger D13: the out-of-band connection record, exchanged once per pairing */
struct qpi { uint32_t xmagic, xsize; uint32_t qpn,psn,pgsz; uint16_t lid; uint8_t gid[16]; uint16_t node; uint32_t count,qpns[MESH_QPS+1],psns[MESH_QPS+1]; };
#define XMAGIC 0x4d595048u

static int dial(struct addrinfo *a,struct hdr *m,uint64_t client){
  int f=socket(a->ai_family,SOCK_STREAM,0); if(f<0) return -1;
  fcntl(f,F_SETFL,O_NONBLOCK);
  if(connect(f,a->ai_addr,a->ai_addrlen)==0)return f;
  if(errno!=EINPROGRESS){int error=errno;close(f);errno=error;return -1;}
  struct pollfd ready={.fd=f,.events=POLLOUT};
  while(!stop && atomic_load_explicit(&m->client,memory_order_acquire)==client){
    int status=poll(&ready,1,0);
    if(status<0 && errno==EINTR)continue;
    if(status<0){int error=errno;close(f);errno=error;return -1;}
    if(!status)continue;
    int error=0;socklen_t length=sizeof error;
    if(getsockopt(f,SOL_SOCKET,SO_ERROR,&error,&length))error=errno;
    if(!error)return f;
    close(f);errno=error;return -1;
  }
  close(f);errno=ECANCELED;return -1;
}

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

static int exchange(int f,const void *mine,void *you,size_t send_bytes,size_t receive_bytes,struct hdr *m,uint64_t client){
  size_t sent=0,got=0;
  fcntl(f,F_SETFL,O_NONBLOCK);
  while(!stop && atomic_load_explicit(&m->client,memory_order_acquire)==client){
    if(sent==send_bytes && got==receive_bytes)return 0;
    if(sent<send_bytes){
      ssize_t n=write(f,(const char*)mine+sent,send_bytes-sent);
      if(n>0)sent+=(size_t)n;
      else if(!n){errno=ECONNRESET;return -1;}
      else if(errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR)return -1;
    }
    if(got<receive_bytes){
      ssize_t n=read(f,(char*)you+got,receive_bytes-got);
      if(n>0)got+=(size_t)n;
      else if(!n){errno=ECONNRESET;return -1;}
      else if(errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR)return -1;
    }
  }
  errno=ECANCELED;return -1;
}

static int oob(const char *peer,struct hdr *m,uint64_t client){
  if(!peer)return accept(lsock,NULL,NULL);
  struct addrinfo hint={.ai_socktype=SOCK_STREAM,.ai_family=AF_UNSPEC},*addresses;
  if(getaddrinfo(peer,MESH_PORT,&hint,&addresses)){errno=EHOSTUNREACH;return -1;}
  int socket=-1,error=EHOSTUNREACH;
  for(struct addrinfo *a=addresses;a && !stop;a=a->ai_next){
    socket=dial(a,m,client);error=errno;
    if(socket>=0 || error==ECANCELED)break;
  }
  freeaddrinfo(addresses);errno=error;return socket;
}

static struct ibv_port_attr pa;
/* ledger D13 (out-of-band metadata), D6 (queue pair limits), TN3205 queue-pair state transitions */
static int verbs_up(const char *peer, char *mem, size_t span, size_t origin, int me, uint32_t message_bytes, int qps, int (*configure)(void *,int,uint64_t),void *state,uint64_t client){
  if(qps<1 || qps>MESH_QPS+1){ errno=EINVAL; return -1; }
  if(provider->context && (ibv_query_port(provider->context,1,&pa) || pa.state!=IBV_PORT_ACTIVE)){
    return -1; }
  struct hdr *m=(struct hdr *)mem;
  int f=oob(peer,m,client); if(f<0) return !peer && (errno==EAGAIN || errno==EWOULDBLOCK)?1:-1;
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
  /* design/algorithm-sources.md#programtensor */
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
  origin=provider->region_origin;
  size_t regions=(origin?1:0)+(span-origin+provider->region_extent-1)/provider->region_extent;
  if(!provider->regions) provider->regions=calloc(regions,sizeof *provider->regions);
  if(!provider->regions){ close(f); fprintf(stderr,"alloc regions: failed\n"); return -1; }
  while((size_t)provider->region_count<regions){
    int data=provider->region_count>=(origin?1:0);
    size_t o=data?origin+((size_t)provider->region_count-(origin?1:0))*provider->region_extent:0;
    size_t end=data?o+provider->region_extent:origin;
    size_t n=(end<span?end:span)-o;
    /* ledger D1: "applications should only register memory as IBV_ACCESS_LOCAL_WRITE" */
    provider->regions[provider->region_count]=ibv_reg_mr(provider->domain,mem+o,n,IBV_ACCESS_LOCAL_WRITE);
    if(!provider->regions[provider->region_count]){ close(f); return -1; } provider->region_count++; }
  if(ibv_query_port(provider->context,1,&pa)){ close(f); return -1; }
  /* design/algorithm-sources.md#programcopy */
  uint32_t frame_capacity=capabilities.max_qp_wr<QD?capabilities.max_qp_wr:QD;
  if(capabilities.max_cqe<=1 || !frame_capacity){close(f);errno=EOPNOTSUPP;return -1;}
  if(frame_capacity>=(uint32_t)capabilities.max_cqe)frame_capacity=(uint32_t)capabilities.max_cqe-1;
  for(int q=0;q<qps;q++){
    for(int d=0;d<2;d++){
      provider->completion_queues[2*q+d]=ibv_create_cq(provider->context,(int)frame_capacity+1,NULL,NULL,0);
      if(!provider->completion_queues[2*q+d]){close(f);return -1;}
    }
    struct ibv_qp_init_attr qi={.send_cq=provider->completion_queues[2*q+MESH_SEND],
      .recv_cq=provider->completion_queues[2*q+MESH_RECEIVE],.qp_type=IBV_QPT_UC,
      .cap={.max_send_wr=frame_capacity,.max_recv_wr=frame_capacity,.max_send_sge=1,.max_recv_sge=1}};
    provider->pairs[q]=ibv_create_qp(provider->domain,&qi);
    if(!provider->pairs[q]){close(f);return -1;}
    provider->qp_count=q+1;
    struct ibv_qp_attr queried;struct ibv_qp_init_attr actual;
    if(ibv_query_qp(provider->pairs[q],&queried,IBV_QP_CAP,&actual)){close(f);return -1;}
    provider->capacity[q][MESH_SEND]=actual.cap.max_send_wr<frame_capacity?actual.cap.max_send_wr:frame_capacity;
    provider->capacity[q][MESH_RECEIVE]=actual.cap.max_recv_wr<frame_capacity?actual.cap.max_recv_wr:frame_capacity;
    if(!provider->capacity[q][MESH_SEND] || !provider->capacity[q][MESH_RECEIVE]){close(f);errno=EOPNOTSUPP;return -1;}
    fprintf(stderr,"pair capacity queue=%d send_frames=%u receive_frames=%u cq_entries=%d,%d\n",q,
      provider->capacity[q][MESH_SEND],provider->capacity[q][MESH_RECEIVE],
      provider->completion_queues[2*q+MESH_SEND]->cqe,provider->completion_queues[2*q+MESH_RECEIVE]->cqe);
  }
  provider->pair=provider->pairs[0];
  struct ibv_qp_attr a={.qp_state=IBV_QPS_INIT,.port_num=1};
  for(int q=0;q<qps;q++) if(ibv_modify_qp(provider->pairs[q],&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)){ close(f); return -1; }
  union ibv_gid gid; if(ibv_query_gid(provider->context,1,0,&gid)){ close(f); return -1; }
  uint32_t psn=arc4random()&0xffffff;
  struct qpi mine={.xmagic=XMAGIC+MESH_VERSION,.xsize=sizeof mine,.qpn=provider->pair->qp_num,.psn=psn,.lid=pa.lid,.pgsz=message_bytes,.node=(uint16_t)me,.count=(uint32_t)qps},you;
  for(int q=0;q<qps;q++){ mine.qpns[q]=provider->pairs[q]->qp_num; mine.psns[q]=(psn+(uint32_t)q)&0xffffff; }
  memcpy(mine.gid,&gid,16);
  fprintf(stderr,"pair setup node=%d exchange=%.6f regions=%d qpn=%u\n",me,monotime(),provider->region_count,mine.qpn);
  if(exchange(f,&mine,&you,sizeof mine,sizeof you,m,client)){ close(f); fprintf(stderr,"exchange failed\n"); return -1; }
  /* ledger D6: both ends must post messages of the same frame count; D5: the same queue-pair count */
  if(you.xmagic!=mine.xmagic || you.xsize!=sizeof you || you.pgsz!=mine.pgsz || you.count!=mine.count || (expected_peer>=0 && you.node!=expected_peer)){
    fprintf(stderr,"exchange mismatch: local=%u,%u,%u,%u,%u peer=%u,%u,%u,%u,%u expected_node=%d\n",mine.xmagic,mine.xsize,mine.pgsz,mine.count,mine.node,you.xmagic,you.xsize,you.pgsz,you.count,you.node,expected_peer); close(f);errno=EPROTO;return -1; }
  expected_peer=you.node;
  for(int q=0;q<qps;q++){
    struct ibv_qp_attr r={.qp_state=IBV_QPS_RTR,.path_mtu=IBV_MTU_4096,.rq_psn=you.psns[q],
      .dest_qp_num=you.qpns[q],.ah_attr={.dlid=you.lid,.port_num=1,.is_global=1,
      .grh={.hop_limit=1,.sgid_index=0}}};
    memcpy(&r.ah_attr.grh.dgid,you.gid,16);
    int rc=ibv_modify_qp(provider->pairs[q],&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN);
    if(rc){ fprintf(stderr,"rtr %d rc %d dlid %u dqpn %u\n",q,rc,you.lid,you.qpns[q]); close(f); return -1; }
  }
  if(configure(state,f,client)){close(f);return -1;}
  for(int q=0;q<qps;q++){
    struct ibv_qp_attr t={.qp_state=IBV_QPS_RTS,.sq_psn=mine.psns[q]};
    int rc=ibv_modify_qp(provider->pairs[q],&t,IBV_QP_STATE|IBV_QP_SQ_PSN);
    if(rc){ fprintf(stderr,"rts %d rc %d, failed\n",q,rc); close(f); return -1; }
  }
  close(f);
  fprintf(stderr,"pair up: %s node %d\n",ibv_get_device_name(provider->context->device),me);
  return 0; }
