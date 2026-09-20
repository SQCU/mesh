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
#include <time.h>

#define QD 4095
struct mesh_wire { char *data; size_t length; struct ibv_sge *spans; };
struct mesh_device {
  const char *name;
  struct ibv_context *context; struct ibv_pd *domain; struct ibv_mr **regions;
  uint32_t region_count,frame_capacity;
  struct ibv_sge *spans;
  pthread_mutex_t setup;
};
/* design/algorithm-sources.md#programcopy */
struct mesh_queue {
  _Alignas(64) struct ibv_qp *pair;
  struct ibv_cq *completion;
  int (*poll)(struct ibv_cq *,int,struct ibv_wc *);
  int (*send)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **);
  int (*receive)(struct ibv_qp *,struct ibv_recv_wr *,struct ibv_recv_wr **);
  uint32_t receive_capacity,send_capacity;
};
_Static_assert(sizeof(struct mesh_queue)==64 && _Alignof(struct mesh_queue)==64 &&
  offsetof(struct mesh_queue,send_capacity)==44,"M08 native dispatch");
struct mesh_verbs {
  struct mesh_device *device; struct mesh_wire *wire;
  struct mesh_queue *queues; int qp_count,listener;
  struct ibv_cq *completion;
  uint32_t peer,completion_entries[2];
  uint64_t bandwidth;
  const char *local_address,*remote_address,*service;
  uint64_t deadline;
};
/* design/prepared-machine.md#M07 */
/* design/algorithm-sources.md#programtensor */
static int wire_map(struct mesh_wire *wire,struct hdr *m,int file){
  size_t payload=(size_t)m->block*m->pgsz,blocks=mesh_blocks(m);
  wire->length=blocks*payload;
  wire->data=mmap(NULL,wire->length,PROT_READ|PROT_WRITE,MAP_SHARED,file,(off_t)m->data_off);
  if(wire->data==MAP_FAILED){wire->data=NULL;return -1;}
  wire->spans=calloc(blocks,sizeof *wire->spans);
  if(!wire->spans)return -1;
  for(size_t i=0;i<blocks;i++)wire->spans[i]=(struct ibv_sge){.addr=(uintptr_t)wire->data+i*payload,.length=(uint32_t)payload};
  return 0;
}
static const char *shm; static _Atomic sig_atomic_t stop;
/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#programcopy */
static int down_pair(struct mesh_verbs *provider){
  while(provider->qp_count){ struct ibv_qp *q=provider->queues[provider->qp_count-1].pair; if(q && ibv_destroy_qp(q))return 0; provider->queues[--provider->qp_count].pair=NULL; }
  if(provider->completion){
    if(ibv_destroy_cq(provider->completion))return 0;
    provider->completion=NULL;
  }
  free(provider->queues);provider->queues=NULL;
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
struct qpi { uint32_t xmagic, xsize; uint32_t pgsz; uint16_t lid; uint8_t gid[16]; uint32_t node,count; };
#define XMAGIC 0x4d595048u

/* design/algorithm-sources.md#programcopy */
static int pairing_active(struct hdr *m,uint64_t client,uint64_t deadline){
  if(stop || atomic_load_explicit(&m->client,memory_order_acquire)!=client){errno=ECANCELED;return 0;}
  if(clock_gettime_nsec_np(CLOCK_MONOTONIC)>=deadline){errno=ETIMEDOUT;return 0;}
  return 1;
}

/* design/algorithm-sources.md#programcopy */
static int dial(struct addrinfo *a,struct hdr *m,uint64_t client,uint64_t deadline){
  if(!pairing_active(m,client,deadline))return -1;
  int f=socket(a->ai_family,SOCK_STREAM,0); if(f<0) return -1;
  if(fcntl(f,F_SETFL,O_NONBLOCK)<0)goto failed;
  if(connect(f,a->ai_addr,a->ai_addrlen)==0)return f;
  if(errno!=EINPROGRESS)goto failed;
  struct pollfd ready={.fd=f,.events=POLLOUT};
  while(pairing_active(m,client,deadline)){
    int status=poll(&ready,1,0);
    if(status<0 && errno==EINTR)continue;
    if(status<0)goto failed;
    if(!status)continue;
    int error=0;socklen_t length=sizeof error;
    if(getsockopt(f,SOL_SOCKET,SO_ERROR,&error,&length))error=errno;
    if(!error)return f;
    errno=error;goto failed;
  }
failed:;
  int error=errno;close(f);errno=error;return -1;
}

/* design/algorithm-sources.md#programcopy */
static int listener_up(struct mesh_verbs *provider){
  struct addrinfo hint={.ai_socktype=SOCK_STREAM,.ai_family=AF_UNSPEC,.ai_flags=AI_PASSIVE|AI_NUMERICHOST|AI_NUMERICSERV},*r;
  if(getaddrinfo(provider->local_address,provider->service,&hint,&r)){errno=EINVAL;return -1;}
  provider->listener=socket(r->ai_family,SOCK_STREAM,0);
  if(provider->listener<0){int error=errno;freeaddrinfo(r);errno=error;return -1;}
  int on=1,off=0;
  setsockopt(provider->listener,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on);
  if(r->ai_family==AF_INET6)setsockopt(provider->listener,IPPROTO_IPV6,IPV6_V6ONLY,&off,sizeof off);
  int error=fcntl(provider->listener,F_SETFL,O_NONBLOCK)<0 || bind(provider->listener,r->ai_addr,r->ai_addrlen) || listen(provider->listener,4),code=errno;
  freeaddrinfo(r);
  if(error){ close(provider->listener); provider->listener=-1;errno=code;return -1; }
  return 0; }

/* design/algorithm-sources.md#programcopy */
static int exchange(int f,const void *mine,void *you,size_t send_bytes,size_t receive_bytes,struct hdr *m,uint64_t client,uint64_t deadline){
  size_t sent=0,got=0;
  while(pairing_active(m,client,deadline)){
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
  return -1;
}

/* design/algorithm-sources.md#programcopy */
/* design/prepared-machine.md#M08 */
static int oob(struct mesh_verbs *provider,struct hdr *m,uint64_t client){
  if(!pairing_active(m,client,provider->deadline))return -1;
  if(m->node>provider->peer){
    if(provider->listener<0 && listener_up(provider))return -1;
    while(pairing_active(m,client,provider->deadline)){
      int f=accept(provider->listener,NULL,NULL);
      if(f>=0){
        if(fcntl(f,F_SETFL,O_NONBLOCK)==0)return f;
        int error=errno;close(f);errno=error;return -1;
      }
      if(errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR)return -1;
    }
    return -1;
  }
  struct addrinfo hint={.ai_socktype=SOCK_STREAM,.ai_family=AF_UNSPEC,.ai_flags=AI_NUMERICHOST|AI_NUMERICSERV},*addresses;
  if(getaddrinfo(provider->remote_address,provider->service,&hint,&addresses)){errno=EINVAL;return -1;}
  int socket=-1,error=EHOSTUNREACH;
  for(;;){
    if(!pairing_active(m,client,provider->deadline)){error=errno;break;}
    for(struct addrinfo *a=addresses;a;a=a->ai_next){
      socket=dial(a,m,client,provider->deadline);error=errno;
      if(socket>=0 || error==ECANCELED || error==ETIMEDOUT)break;
    }
    if(socket>=0 || (error!=ECONNREFUSED && error!=ENETUNREACH && error!=EHOSTUNREACH))break;
    poll(NULL,0,1);
  }
  freeaddrinfo(addresses);errno=error;return socket;
}

/* design/algorithm-sources.md#programcopy */
/* design/prepared-machine.md#M09 */
static int device_up(struct mesh_device *device,struct mesh_wire *wire,struct hdr *m,struct ibv_port_attr *port){
  int error=0;
  pthread_mutex_lock(&device->setup);
  if(!device->context){
    struct ibv_device **list=ibv_get_device_list(NULL);
    for(int i=0;list && list[i];i++)if(!strcmp(device->name,ibv_get_device_name(list[i]))){device->context=ibv_open_device(list[i]);break;}
    if(list)ibv_free_device_list(list);
    if(!device->context){error=errno?errno:ENODEV;goto done;}
  }
  if(ibv_query_port(device->context,1,port)){error=errno;goto done;}
  if(port->state!=IBV_PORT_ACTIVE){error=ENETDOWN;goto done;}
  if(device->frame_capacity)goto done;
  struct ibv_device_attr capabilities;
  if(ibv_query_device(device->context,&capabilities)){error=errno;goto done;}
  if(!device->domain)device->domain=ibv_alloc_pd(device->context);
  if(!device->domain){error=errno;goto done;}
  size_t stride=(size_t)m->block*m->pgsz,span=(size_t)mesh_blocks(m)*stride;
  /* design/prepared-machine.md#M09 */
  size_t limit=capabilities.max_mr_size<span?capabilities.max_mr_size:span;
  size_t extent=limit/stride*stride;
  size_t regions=(span+extent-1)/extent;
  if(regions>(size_t)capabilities.max_mr){error=ENOMEM;goto done;}
  if(!device->regions)device->regions=calloc(regions,sizeof *device->regions);
  if(!device->regions){error=ENOMEM;goto done;}
  while(device->region_count<regions){
    size_t offset=(size_t)device->region_count*extent,end=offset+extent;
    device->regions[device->region_count]=ibv_reg_mr(device->domain,wire->data+offset,(end<span?end:span)-offset,IBV_ACCESS_LOCAL_WRITE);
    if(!device->regions[device->region_count]){
      error=errno;fprintf(stderr,"register %s offset=%zu bytes=%zu max_mr_size=%llu max_mr=%d: %s\n",
        device->name,offset,(end<span?end:span)-offset,(unsigned long long)capabilities.max_mr_size,capabilities.max_mr,strerror(error));goto done;
    }
    device->region_count++;
  }
  if(!device->spans)device->spans=calloc(mesh_blocks(m),sizeof *device->spans);
  if(!device->spans){error=ENOMEM;goto done;}
  for(size_t i=0;i<mesh_blocks(m);i++){
    device->spans[i]=wire->spans[i];
    device->spans[i].lkey=device->regions[i*stride/extent]->lkey;
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
/* design/prepared-machine.md#M06 */
/* design/prepared-machine.md#M08 */
static int verbs_up(struct mesh_verbs *provider,struct hdr *m,int qps,int (*configure)(void *,int,uint64_t),void *state,uint64_t client){
  struct ibv_port_attr pa;
  if(device_up(provider->device,provider->wire,m,&pa))return -1;
  provider->deadline=clock_gettime_nsec_np(CLOCK_MONOTONIC)+UINT64_C(30000000000);
  int f=oob(provider,m,client);
  if(f<0)return -1;
  uint32_t frame_capacity=provider->device->frame_capacity;
  /* design/prepared-machine.md#M11 */
  provider->queues=calloc((size_t)qps,sizeof *provider->queues);
  if(!provider->queues){close(f);return -1;}
  provider->completion=ibv_create_cq(provider->device->context,
    (int)(frame_capacity+1),NULL,NULL,0);
  if(!provider->completion){close(f);return -1;}
  for(int q=0;q<qps;q++){
    struct mesh_queue *queue=&provider->queues[q];
    queue->completion=provider->completion;
    queue->poll=provider->completion->context->ops.poll_cq;
    struct ibv_qp_init_attr qi={.send_cq=queue->completion,
      .recv_cq=queue->completion,.qp_type=IBV_QPT_UC,
      .cap={.max_send_wr=frame_capacity,.max_recv_wr=frame_capacity,.max_send_sge=1,.max_recv_sge=1}};
    queue->pair=ibv_create_qp(provider->device->domain,&qi);
    if(!queue->pair){close(f);return -1;}
    queue->send=queue->pair->context->ops.post_send;queue->receive=queue->pair->context->ops.post_recv;
    provider->qp_count=q+1;
    struct ibv_qp_attr queried;struct ibv_qp_init_attr actual;
    if(ibv_query_qp(queue->pair,&queried,IBV_QP_CAP,&actual)){close(f);return -1;}
    queue->receive_capacity=actual.cap.max_recv_wr;
    queue->send_capacity=actual.cap.max_send_wr;
    fprintf(stderr,"pair capacity queue=%d send_frames=%u receive_frames=%u cq_entries=%d\n",q,
      actual.cap.max_send_wr,actual.cap.max_recv_wr,
      queue->completion->cqe);
  }
  struct ibv_qp_attr a={.qp_state=IBV_QPS_INIT,.port_num=1};
  for(int q=0;q<qps;q++) if(ibv_modify_qp(provider->queues[q].pair,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)){ close(f); return -1; }
  union ibv_gid gid; if(ibv_query_gid(provider->device->context,1,0,&gid)){ close(f); return -1; }
  uint32_t psn=arc4random()&0xffffff;
  struct qpi mine={.xmagic=XMAGIC+MESH_VERSION,.xsize=sizeof mine,.lid=pa.lid,.pgsz=m->block*m->pgsz,.node=m->node,.count=(uint32_t)qps},you;
  memcpy(mine.gid,&gid,16);
  if(exchange(f,&mine,&you,sizeof mine,sizeof you,m,client,provider->deadline)){ int error=errno;close(f);fprintf(stderr,"exchange failed\n");errno=error;return -1; }
  /* design/collective-dependency-ledger.md#d6-paired-send-and-receive-frame-counts-match */
  if(you.xmagic!=mine.xmagic || you.xsize!=sizeof you || you.pgsz!=mine.pgsz || you.count!=mine.count || you.node!=provider->peer){
    fprintf(stderr,"exchange mismatch: local=%u,%u,%u,%u,%u peer=%u,%u,%u,%u,%u expected_node=%d\n",mine.xmagic,mine.xsize,mine.pgsz,mine.count,mine.node,you.xmagic,you.xsize,you.pgsz,you.count,you.node,provider->peer); close(f);errno=EPROTO;return -1; }
  for(int q=0;q<qps;q++){
    uint32_t local[2]={provider->queues[q].pair->qp_num,(psn+(uint32_t)q)&0xffffff},remote[2];
    if(exchange(f,local,remote,sizeof local,sizeof remote,m,client,provider->deadline)){close(f);return -1;}
    struct ibv_qp_attr r={.qp_state=IBV_QPS_RTR,.path_mtu=IBV_MTU_4096,.rq_psn=remote[1],
      .dest_qp_num=remote[0],.ah_attr={.dlid=you.lid,.port_num=1,.is_global=1,
        .grh={.hop_limit=1,.sgid_index=0}}};
    memcpy(&r.ah_attr.grh.dgid,you.gid,16);
    int rc=ibv_modify_qp(provider->queues[q].pair,&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN);
    if(rc){fprintf(stderr,"rtr %d rc %d\n",q,rc);close(f);return -1;}
  }
  if(configure(state,f,client)){int error=errno;close(f);errno=error;return -1;}
  for(int q=0;q<qps;q++){
    struct ibv_qp_attr t={.qp_state=IBV_QPS_RTS,.sq_psn=(psn+(uint32_t)q)&0xffffff};
    int rc=ibv_modify_qp(provider->queues[q].pair,&t,IBV_QP_STATE|IBV_QP_SQ_PSN);
    if(rc){ fprintf(stderr,"rts %d rc %d, failed\n",q,rc); close(f); return -1; }
  }
  /* design/algorithm-sources.md#link */
  static const uint64_t speeds[256]={[1]=2500000000,[2]=5000000000,[4]=10000000000,[8]=10000000000,[16]=14000000000,[32]=25000000000,[64]=50000000000,[128]=100000000000};
  static const uint8_t widths[256]={[1]=1,[2]=4,[4]=8,[8]=12};
  provider->bandwidth=speeds[pa.active_speed]*widths[pa.active_width];
  fprintf(stderr,"pair up: %s node %d\n",ibv_get_device_name(provider->device->context->device),m->node);
  return f; }
