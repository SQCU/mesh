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
#include <limits.h>
#include <pthread.h>

#define QD 4095
struct mesh_wire { char *data; size_t length,region_extent; struct ibv_sge *spans; };
struct mesh_device {
  const char *name;
  struct ibv_context *context; struct ibv_pd *domain; struct ibv_mr **regions;
  uint32_t region_count,frame_capacity;
  struct ibv_sge *spans;
  pthread_mutex_t setup;
};
struct mesh_verbs {
  struct mesh_device *device; struct mesh_wire *wire;
  struct ibv_cq *completion_queues[2*MESH_QPS];
  struct ibv_qp *pairs[MESH_QPS]; int qp_count,listener;
  uint32_t capacity[MESH_QPS][2],peer;
  const char *local_address,*remote_address,*service;
  struct ibv_wc *completions;
};
/* design/algorithm-sources.md#programtensor */
static int wire_map(struct mesh_wire *wire,struct hdr *m,int file){
  size_t bank=(size_t)1<<32,stride=(size_t)(m->block+1)*m->pgsz,blocks=mesh_blocks(m);
  wire->region_extent=((size_t)1<<30)/stride*stride;
  size_t regions=(blocks*stride+wire->region_extent-1)/wire->region_extent,length=regions*bank;
  char *reserved=mmap(NULL,length+bank,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0);
  if(reserved==MAP_FAILED)return -1;
  char *base=(char *)(((uintptr_t)reserved+bank-1)&~(uintptr_t)(bank-1));
  if(base>reserved)munmap(reserved,(size_t)(base-reserved));
  munmap(base+length,(size_t)(reserved+length+bank-(base+length)));
  wire->data=base;wire->length=length;
  wire->spans=calloc(blocks,sizeof *wire->spans);
  if(!wire->spans)return -1;
  size_t payload=(size_t)m->block*m->pgsz;
  for(size_t i=0;i<blocks;i++){
    size_t offset=i*stride;
    char *address=base+offset/wire->region_extent*bank+offset%wire->region_extent;
    if(mmap(address,m->pgsz,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_FIXED,file,(off_t)(m->tags_off+i*m->pgsz))==MAP_FAILED)return -1;
    if(mmap(address+m->pgsz,payload,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_FIXED,file,(off_t)(m->data_off+i*payload))==MAP_FAILED)return -1;
    wire->spans[i]=(struct ibv_sge){.addr=(uintptr_t)address+m->pgsz-sizeof(uint32_t),.length=(uint32_t)(payload+sizeof(uint32_t))};
  }
  return 0;
}
static const char *shm; static _Atomic sig_atomic_t stop;
/* design/algorithm-sources.md#programcopy */
static int down_pair(struct mesh_verbs *provider){
  while(provider->qp_count){ struct ibv_qp *q=provider->pairs[provider->qp_count-1]; if(q && ibv_destroy_qp(q))return 0; provider->pairs[--provider->qp_count]=NULL; }
  for(int i=0;i<2*MESH_QPS;i++)if(provider->completion_queues[i]){
    if(ibv_destroy_cq(provider->completion_queues[i]))return 0;
    provider->completion_queues[i]=NULL;
  }
  return 1;
}
/* design/algorithm-sources.md#programcopy */
static int down_device(struct mesh_device *device){
  while(device->region_count){
    if(ibv_dereg_mr(device->regions[device->region_count-1]))return 0;
    device->region_count--;
  }
  free(device->regions);device->regions=NULL;
  if(device->domain){if(ibv_dealloc_pd(device->domain))return 0;device->domain=NULL;}
  if(device->context){if(ibv_close_device(device->context))return 0;device->context=NULL;}
  free(device->spans);device->spans=NULL;
  return 1;
}
static void down(void){ if(shm)shm_unlink(shm); }
static void die(const char*m){ fprintf(stderr,"%s\n",m); exit(1); }
static void onsig(int s){ (void)s; stop++; }

/* design/collective-dependency-ledger.md#d13-connection-metadata-is-setup-work */
struct qpi { uint32_t xmagic, xsize; uint32_t pgsz; uint16_t lid; uint8_t gid[16]; uint32_t node,count,qpns[MESH_QPS],psns[MESH_QPS]; };
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

/* design/algorithm-sources.md#programcopy */
static int listener_up(struct mesh_verbs *provider){
  struct addrinfo hint={.ai_socktype=SOCK_STREAM,.ai_family=AF_UNSPEC,.ai_flags=AI_PASSIVE},*r;
  if(getaddrinfo(provider->local_address,provider->service,&hint,&r)) return -1;
  provider->listener=socket(r->ai_family,SOCK_STREAM,0);
  int on=1,off=0;
  setsockopt(provider->listener,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on);
  if(r->ai_family==AF_INET6)setsockopt(provider->listener,IPPROTO_IPV6,IPV6_V6ONLY,&off,sizeof off);
  int error=bind(provider->listener,r->ai_addr,r->ai_addrlen) || listen(provider->listener,4);
  freeaddrinfo(r);
  if(error){ close(provider->listener); provider->listener=-1; return -1; }
  fcntl(provider->listener,F_SETFL,O_NONBLOCK); return 0; }

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

/* design/algorithm-sources.md#programcopy */
static int oob(struct mesh_verbs *provider,struct hdr *m,uint64_t client){
  if(m->node>provider->peer){
    if(provider->listener<0 && listener_up(provider))return -1;
    return accept(provider->listener,NULL,NULL);
  }
  struct addrinfo hint={.ai_socktype=SOCK_STREAM,.ai_family=AF_UNSPEC},*addresses;
  if(getaddrinfo(provider->remote_address,provider->service,&hint,&addresses)){errno=EHOSTUNREACH;return -1;}
  int socket=-1,error=EHOSTUNREACH;
  for(struct addrinfo *a=addresses;a && !stop;a=a->ai_next){
    socket=dial(a,m,client);error=errno;
    if(socket>=0 || error==ECANCELED)break;
  }
  freeaddrinfo(addresses);errno=error;return socket;
}

/* design/algorithm-sources.md#programcopy */
static int device_up(struct mesh_device *device,struct mesh_wire *wire,struct hdr *m){
  int error=0;
  pthread_mutex_lock(&device->setup);
  if(device->frame_capacity)goto done;
  if(!device->context){
    struct ibv_device **list=ibv_get_device_list(NULL);
    for(int i=0;list && list[i];i++)if(!strcmp(device->name,ibv_get_device_name(list[i]))){device->context=ibv_open_device(list[i]);break;}
    if(list)ibv_free_device_list(list);
    if(!device->context){error=errno?errno:ENODEV;goto done;}
  }
  struct ibv_device_attr capabilities;
  if(ibv_query_device(device->context,&capabilities)){error=errno;goto done;}
  if(!device->domain)device->domain=ibv_alloc_pd(device->context);
  if(!device->domain){error=errno;goto done;}
  size_t bank=(size_t)1<<32,stride=(size_t)(m->block+1)*m->pgsz,span=(size_t)mesh_blocks(m)*stride;
  size_t regions=(span+wire->region_extent-1)/wire->region_extent;
  if(regions>(size_t)capabilities.max_mr){error=ENOMEM;goto done;}
  if(!device->regions)device->regions=calloc(regions,sizeof *device->regions);
  if(!device->regions){error=ENOMEM;goto done;}
  while(device->region_count<regions){
    size_t offset=(size_t)device->region_count*wire->region_extent,end=offset+wire->region_extent;
    device->regions[device->region_count]=ibv_reg_mr(device->domain,wire->data+(size_t)device->region_count*bank,(end<span?end:span)-offset,IBV_ACCESS_LOCAL_WRITE);
    if(!device->regions[device->region_count]){error=errno;goto done;}
    device->region_count++;
  }
  if(!device->spans)device->spans=calloc(mesh_blocks(m),sizeof *device->spans);
  if(!device->spans){error=ENOMEM;goto done;}
  for(size_t i=0;i<mesh_blocks(m);i++){
    device->spans[i]=wire->spans[i];
    device->spans[i].lkey=device->regions[i*stride/wire->region_extent]->lkey;
  }
  uint32_t capacity=capabilities.max_qp_wr<QD?capabilities.max_qp_wr:QD;
  if(capabilities.max_cqe<=1 || !capacity){error=EOPNOTSUPP;goto done;}
  device->frame_capacity=capacity>=(uint32_t)capabilities.max_cqe?(uint32_t)capabilities.max_cqe-1:capacity;
done:
  pthread_mutex_unlock(&device->setup);
  if(error){errno=error;return -1;}
  return 0;
}
/* design/algorithm-sources.md#programcopy */
static int verbs_up(struct mesh_verbs *provider,struct hdr *m,int qps,int (*configure)(void *,int,uint64_t),void *state,uint64_t client){
  if(device_up(provider->device,provider->wire,m))return -1;
  struct ibv_port_attr pa;
  if(ibv_query_port(provider->device->context,1,&pa) || pa.state!=IBV_PORT_ACTIVE)return -1;
  int f=oob(provider,m,client);
  if(f<0)return errno==EAGAIN || errno==EWOULDBLOCK?1:-1;
  uint32_t frame_capacity=provider->device->frame_capacity;
  for(int q=0;q<qps;q++){
    for(int d=0;d<2;d++){
      provider->completion_queues[2*q+d]=ibv_create_cq(provider->device->context,(int)frame_capacity+1,NULL,NULL,0);
      if(!provider->completion_queues[2*q+d]){close(f);return -1;}
    }
    struct ibv_qp_init_attr qi={.send_cq=provider->completion_queues[2*q+MESH_SEND],
      .recv_cq=provider->completion_queues[2*q+MESH_RECEIVE],.qp_type=IBV_QPT_UC,
      .cap={.max_send_wr=frame_capacity,.max_recv_wr=frame_capacity,.max_send_sge=1,.max_recv_sge=1}};
    provider->pairs[q]=ibv_create_qp(provider->device->domain,&qi);
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
  struct ibv_qp_attr a={.qp_state=IBV_QPS_INIT,.port_num=1};
  for(int q=0;q<qps;q++) if(ibv_modify_qp(provider->pairs[q],&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)){ close(f); return -1; }
  union ibv_gid gid; if(ibv_query_gid(provider->device->context,1,0,&gid)){ close(f); return -1; }
  uint32_t psn=arc4random()&0xffffff;
  struct qpi mine={.xmagic=XMAGIC+MESH_VERSION,.xsize=sizeof mine,.lid=pa.lid,.pgsz=m->block*m->pgsz,.node=m->node,.count=(uint32_t)qps},you;
  for(int q=0;q<qps;q++){ mine.qpns[q]=provider->pairs[q]->qp_num; mine.psns[q]=(psn+(uint32_t)q)&0xffffff; }
  memcpy(mine.gid,&gid,16);
  if(exchange(f,&mine,&you,sizeof mine,sizeof you,m,client)){ close(f); fprintf(stderr,"exchange failed\n"); return -1; }
  /* design/collective-dependency-ledger.md#d6-paired-send-and-receive-frame-counts-match */
  if(you.xmagic!=mine.xmagic || you.xsize!=sizeof you || you.pgsz!=mine.pgsz || you.count!=mine.count || you.node!=provider->peer){
    fprintf(stderr,"exchange mismatch: local=%u,%u,%u,%u,%u peer=%u,%u,%u,%u,%u expected_node=%d\n",mine.xmagic,mine.xsize,mine.pgsz,mine.count,mine.node,you.xmagic,you.xsize,you.pgsz,you.count,you.node,provider->peer); close(f);errno=EPROTO;return -1; }
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
  fprintf(stderr,"pair up: %s node %d\n",ibv_get_device_name(provider->device->context->device),m->node);
  return 0; }
