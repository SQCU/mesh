#pragma once
#include "mesh.h"
#include "mesh-dataflow.h"
#include <sys/event.h>
#include <infiniband/verbs.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
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
/* design/RDMA-KERNEL-RECOVERY.md#tbt_post_recv */
/* tbt_post_recv loads only the low 32 address bits of an SGE, so no registration may cross a 4 GiB
   virtual-address boundary.  The window is mapped at a bank-aligned base and cut into regions of
   MESH_REGION bytes, which divides the bank: every region lies inside one bank, however many banks the
   window spans.  1 GiB is the registration the provider takes (its advertised max_mr_size, 16.4 MB, is not
   enforced: the RCA's 32 GiB and 6 GiB windows, and the tag pre-revert-20261001's 48 and 10 GiB), so max_mr
   regions of it are the device's registrable memory; a provider that refuses one registers in regions of its
   advertised max_mr_size, the window then within one bank. */
#define MESH_BANK ((size_t)1<<32)
#define MESH_REGION ((size_t)1<<30)
struct mesh_wire { char *data; size_t length; };
struct mesh_device {
  const char *name;
  struct ibv_context *context; struct ibv_pd *domain; struct ibv_mr **regions;
  uint32_t region_count,frame_capacity;
  char *wire; size_t extent,payload;
  pthread_mutex_t setup;
};
/* design/prepared-machine.md#M09 */
/* design/prepared-machine.md#M07 */
/* The one copy of the wire-address arithmetic.  offset is a byte offset into the arena, which is a
   byte offset into the registered window because the window begins at data_off.  No per-block span
   array, no page-table mapping load, no startup loop over blocks: an arena of any size costs this
   divide and nothing else. */
static inline struct ibv_sge wire_span(const struct mesh_device *device,uint64_t offset,uint64_t remaining){
  uint64_t capacity=device->payload-offset%device->payload;
  return (struct ibv_sge){.addr=(uintptr_t)device->wire+offset,
    .length=(uint32_t)(remaining<capacity?remaining:capacity),
    .lkey=device->regions[offset/device->extent]->lkey};
}
/* design/algorithm-sources.md#programcopy */
struct mesh_queue {
  _Alignas(64) struct ibv_qp *pair;
  struct ibv_cq *completion;
  int (*poll)(struct ibv_cq *,int,struct ibv_wc *);
  int (*send)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **);
  int (*receive)(struct ibv_qp *,struct ibv_recv_wr *,struct ibv_recv_wr **);
  uint32_t receive_capacity,send_capacity;
};
_Static_assert(sizeof(struct mesh_queue)==64 && _Alignof(struct mesh_queue)==64,"mesh_queue native dispatch");
/* design/collective-dependency-ledger.md#d13-connection-metadata-is-setup-work */
struct qpi { uint32_t xmagic, xsize; uint32_t pgsz; uint16_t lid; uint8_t gid[16]; uint32_t node,count; uint64_t key; };
#define XMAGIC 0x4d595048u
/* A pairing's connection on the listening node: the peer's dial, accepted by its link's acceptor (mesh-flow.c), and
   the peer's exchange record already read from it, which names the session (its key) the connection is for. */
struct mesh_connection { int fd; struct qpi you; };
struct mesh_verbs {
  struct mesh_device *device; struct mesh_wire *wire;
  struct mesh_queue *queues; int qp_count;
  struct ibv_cq *completion;
  uint32_t peer,completion_entries[2];
  uint64_t bandwidth;
  const char *local_address,*remote_address,*service;
  int events;
  /* what is paired: session `session`'s program (MESH_ABSENT: the link's stripe service), under `key`; `running` the
     pairing's own flag (link_stop clears it); `take` hands the listening node a connection its acceptor holds for
     `key` (0, or -1 with EAGAIN: none yet, its kqueue then triggered when one comes) */
  struct hdr *m; uint32_t session; uint64_t key; _Atomic int *running;
  int (*take)(void *,uint64_t key,struct mesh_connection *); void *acceptor;
};
/* design/prepared-machine.md#M07 */
/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M09 */
/* design/RDMA-KERNEL-RECOVERY.md#tbt_post_recv */
static int wire_map(struct mesh_wire *wire,struct hdr *m,int file){
  wire->length=(size_t)m->wire_pages*m->pgsz;
  wire->data=NULL;
  char *reserved=mmap(NULL,wire->length+MESH_BANK,PROT_NONE,MAP_PRIVATE|MAP_ANON|MAP_NORESERVE,-1,0);
  if(reserved==MAP_FAILED)return -1;
  char *base=(char *)(((uintptr_t)reserved+MESH_BANK-1)&~(uintptr_t)(MESH_BANK-1));
  if(base>reserved)munmap(reserved,(size_t)(base-reserved));
  size_t tail=(size_t)((reserved+wire->length+MESH_BANK)-(base+wire->length));
  if(tail)munmap(base+wire->length,tail);
  wire->data=mmap(base,wire->length,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_FIXED,file,(off_t)m->data_off);
  if(wire->data==MAP_FAILED){wire->data=NULL;return -1;}
  fprintf(stderr,"wire window=%zu bytes base=%p bank=%llu offset_in_bank=%llu\n",wire->length,(void *)wire->data,
    (unsigned long long)((uintptr_t)wire->data>>32),(unsigned long long)((uintptr_t)wire->data&(MESH_BANK-1)));
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
  return 1;
}
static void down(void){ if(shm)shm_unlink(shm); }
static void die(const char*m){ fprintf(stderr,"%s\n",m); exit(1); }
static void onsig(int s){ (void)s; stop++; }

/* design/algorithm-sources.md#programcopy */
/* Pairing ends only on observed events (standards S2): its session's stop, the bridge stopping, the pairing's own
   stop (link_stop), or its process exiting (EVFILT_PROC on the link's kqueue); no clock ends it. */
static int pairing_active(struct mesh_verbs *provider){
  if(stop || !atomic_load_explicit(provider->running,memory_order_acquire) ||
     (provider->session!=MESH_ABSENT &&
      (atomic_load_explicit(&mesh_sessions(provider->m)[provider->session].request,memory_order_acquire)!=MESH_REQUEST_START ||
       atomic_load_explicit(&mesh_sessions(provider->m)[provider->session].key,memory_order_acquire)!=provider->key))){errno=ECANCELED;return 0;}
  return 1;
}

/* design/algorithm-sources.md#programcopy */
/* Waits on the link's kqueue for fd's filter (fd<0: the redial pacing timer, which steers how often a
   refused dial is tried and decides nothing).  The link's EVFILT_USER (link_stop) and EVFILT_PROC (the
   client's exit) wake it as well. */
static int pairing_wait(struct mesh_verbs *provider,int fd,int16_t filter){
  struct kevent64_s change,event;int changes=filter?1:0;
  if(fd>=0)EV_SET64(&change,(uint64_t)fd,filter,EV_ADD|EV_ONESHOT,0,0,0,0,0);
  else if(filter)EV_SET64(&change,1,EVFILT_TIMER,EV_ADD|EV_ONESHOT,NOTE_USECONDS,1000,0,0,0);
  for(;;){
    if(!pairing_active(provider))return -1;
    int count=kevent64(provider->events,&change,changes,&event,1,0,NULL);changes=0;
    if(count<0){if(errno==EINTR)continue;return -1;}
    if(!count)continue;
    if(event.filter==EVFILT_PROC){errno=ECANCELED;return -1;}
    if(!filter)return 0;
    if(fd>=0?event.filter==filter && event.ident==(uint64_t)fd:event.filter==EVFILT_TIMER)return 0;
  }
}

/* design/algorithm-sources.md#programcopy */
/* A pairing connection carries a few small records each way, each waited for: sent at once (TCP_NODELAY), or a small
   write waits on the peer's delayed acknowledgement (a session's pairing took 35 ms in its queue numbers' exchange). */
static void connection_prompt(int f){int on=1;setsockopt(f,IPPROTO_TCP,TCP_NODELAY,&on,sizeof on);}
static int dial(struct addrinfo *a,struct mesh_verbs *provider){
  if(!pairing_active(provider))return -1;
  int f=socket(a->ai_family,SOCK_STREAM,0); if(f<0) return -1;
  connection_prompt(f);
  if(fcntl(f,F_SETFL,O_NONBLOCK)<0)goto failed;
  if(connect(f,a->ai_addr,a->ai_addrlen)==0)return f;
  if(errno!=EINPROGRESS)goto failed;
  if(pairing_wait(provider,f,EVFILT_WRITE))goto failed;
  int status=0;socklen_t length=sizeof status;
  if(getsockopt(f,SOL_SOCKET,SO_ERROR,&status,&length))status=errno;
  if(!status)return f;
  errno=status;
failed:;
  int error=errno;close(f);errno=error;return -1;
}

/* design/algorithm-sources.md#programcopy */
/* The listener lives as long as the bridge (opened at its start, closed at its exit), read by its link's acceptor
   alone: a dial waits in its backlog until accepted, and its connection until its session takes it. */
static int listener_up(struct mesh_verbs *provider,int *listener){
  struct addrinfo hint={.ai_socktype=SOCK_STREAM,.ai_family=AF_UNSPEC,.ai_flags=AI_PASSIVE|AI_NUMERICHOST|AI_NUMERICSERV},*r;
  if(getaddrinfo(provider->local_address,provider->service,&hint,&r)){errno=EINVAL;return -1;}
  *listener=socket(r->ai_family,SOCK_STREAM,0);
  if(*listener<0){int error=errno;freeaddrinfo(r);errno=error;return -1;}
  int on=1,off=0;
  setsockopt(*listener,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on);
  if(r->ai_family==AF_INET6)setsockopt(*listener,IPPROTO_IPV6,IPV6_V6ONLY,&off,sizeof off);
  int error=fcntl(*listener,F_SETFL,O_NONBLOCK)<0 || bind(*listener,r->ai_addr,r->ai_addrlen) || listen(*listener,64),code=errno;
  freeaddrinfo(r);
  if(error){ close(*listener); *listener=-1;errno=code;return -1; }
  return 0; }

/* design/algorithm-sources.md#programcopy */
static int exchange(int f,const void *mine,void *you,size_t send_bytes,size_t receive_bytes,struct mesh_verbs *provider){
  size_t sent=0,got=0;
  while(pairing_active(provider)){
    if(sent==send_bytes && got==receive_bytes)return 0;
    if(sent<send_bytes){
      ssize_t n=write(f,(const char*)mine+sent,send_bytes-sent);
      if(n>0)sent+=(size_t)n;
      else if(!n){errno=ECONNRESET;return -1;}
      else if(errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR)return -1;
      else if(pairing_wait(provider,f,EVFILT_WRITE))return -1;
      continue;
    }
    ssize_t n=read(f,(char*)you+got,receive_bytes-got);
    if(n>0)got+=(size_t)n;
    else if(!n){errno=ECONNRESET;return -1;}
    else if(errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR)return -1;
    else if(pairing_wait(provider,f,EVFILT_READ))return -1;
  }
  return -1;
}

/* design/algorithm-sources.md#programcopy */
/* design/prepared-machine.md#M08 */
/* The pairing's connection and the peer's exchange record.  The lower node dials the peer's listener and sends its
   record first; the higher node takes the connection its acceptor holds for this key (waiting on its kqueue until
   one comes), the peer's record read already. */
static int oob(struct mesh_verbs *provider,const struct qpi *mine,struct qpi *you){
  if(!pairing_active(provider))return -1;
  if(provider->m->node>provider->peer){
    for(;;){
      struct mesh_connection taken;
      if(!provider->take(provider->acceptor,provider->key,&taken)){
        *you=taken.you;
        if(exchange(taken.fd,mine,NULL,sizeof *mine,0,provider)){int error=errno;close(taken.fd);errno=error;return -1;}
        return taken.fd;
      }
      if(errno!=EAGAIN || pairing_wait(provider,-1,0))return -1;
    }
  }
  struct addrinfo hint={.ai_socktype=SOCK_STREAM,.ai_family=AF_UNSPEC,.ai_flags=AI_NUMERICHOST|AI_NUMERICSERV},*addresses;
  if(getaddrinfo(provider->remote_address,provider->service,&hint,&addresses)){errno=EINVAL;return -1;}
  int socket=-1,error=EHOSTUNREACH;
  for(;;){
    for(struct addrinfo *a=addresses;a;a=a->ai_next){
      socket=dial(a,provider);error=errno;
      if(socket>=0 || error==ECANCELED)break;
    }
    if(socket>=0 || error==ECANCELED)break;
    if(pairing_wait(provider,-1,EVFILT_TIMER)){error=errno;break;}
  }
  freeaddrinfo(addresses);
  if(socket>=0 && exchange(socket,mine,you,sizeof *mine,sizeof *you,provider)){error=errno;close(socket);socket=-1;}
  errno=error;return socket;
}

/* A pairing attempt that failed because its connection went away (the peer's session ended before it
   paired, or a stale dial from an earlier session) is tried again on the next connection; only this
   client's leaving, the bridge's stop or a mismatch end it. */
static int pairing_retried(int error){
  return error==ECONNRESET || error==EPIPE || error==ENOTCONN || error==ECONNABORTED || error==ETIMEDOUT ||
    error==ECONNREFUSED || error==EHOSTUNREACH || error==ENETUNREACH;
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
  /* design/prepared-machine.md#M09 */
  /* Registered memory is the declared window, not the arena: what an SGE names, wired 1:1 by the provider, in
     regions of MESH_REGION bytes (a whole number of blocks, dividing the bank, so no region cut falls inside a
     block and no region crosses a 4 GiB boundary: wire_span's divide is the only addressing form).  A provider
     that refuses a region that size is registered in regions of its advertised max_mr_size instead, at most
     max_mr of them, the window then within one bank. */
  size_t stride=(size_t)m->block*m->pgsz,span=wire->length;
  size_t extent=(MESH_REGION<span?MESH_REGION:(span+stride-1))/stride*stride,regions=0;
  for(;;){
    regions=extent?(span+extent-1)/extent:0;
    if(!extent || regions>(size_t)capabilities.max_mr || (MESH_BANK%extent && span>MESH_BANK) ||
       ((uintptr_t)wire->data&(MESH_BANK-1))){
      error=ENOMEM;
      fprintf(stderr,"register %s window=%zu extent=%zu regions=%zu base=%p max_mr_size=%llu max_mr=%d\n",
        device->name,span,extent,regions,(void *)wire->data,(unsigned long long)capabilities.max_mr_size,capabilities.max_mr);
      goto done;
    }
    if(!device->regions)device->regions=calloc(regions,sizeof *device->regions);
    if(!device->regions){error=ENOMEM;goto done;}
    int refused=0;
    while(device->region_count<regions){
      size_t offset=(size_t)device->region_count*extent,end=offset+extent;
      device->regions[device->region_count]=ibv_reg_mr(device->domain,wire->data+offset,(end<span?end:span)-offset,IBV_ACCESS_LOCAL_WRITE);
      if(!device->regions[device->region_count]){
        error=errno;fprintf(stderr,"register %s offset=%zu bytes=%zu window=%zu extent=%zu max_mr_size=%llu max_mr=%d: %s\n",
          device->name,offset,(end<span?end:span)-offset,span,extent,(unsigned long long)capabilities.max_mr_size,capabilities.max_mr,strerror(error));
        refused=1;break;
      }
      device->region_count++;
    }
    if(!refused){error=0;break;}
    size_t fallback=capabilities.max_mr_size/stride*stride;
    if(!fallback || fallback>=extent)goto done;
    while(device->region_count){
      if(ibv_dereg_mr(device->regions[device->region_count-1]))goto done;
      device->region_count--;
    }
    free(device->regions);device->regions=NULL;extent=fallback;
  }
  device->wire=wire->data;device->extent=extent;device->payload=stride;
  fprintf(stderr,"register %s window=%zu bytes extent=%zu regions=%zu arena=%llu bytes\n",
    device->name,span,extent,regions,(unsigned long long)mesh_arena_pages(m)*(uint64_t)m->pgsz);
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
/* `qps` queue pairs on one completion queue of the provider's, every one paired with the peer's of the same index;
   the pairing's connection (kept open: its end is the peer's end), or -1. */
static int verbs_up(struct mesh_verbs *provider,int qps,int (*configure)(void *,int),void *state){
  struct hdr *m=provider->m;
  struct ibv_port_attr pa;
  if(device_up(provider->device,provider->wire,m,&pa))return -1;
  /* design/prepared-machine.md#M11 */
  /* the queue pairs made before the pairing's connection (d6dceac's order): the end that comes first makes its while
     the other is still on its way */
  uint32_t frame_capacity=provider->device->frame_capacity;
  provider->queues=calloc((size_t)qps,sizeof *provider->queues);
  if(!provider->queues)return -1;
  provider->completion=ibv_create_cq(provider->device->context,
    (int)(frame_capacity+1),NULL,NULL,0);
  if(!provider->completion)return -1;
  for(int q=0;q<qps;q++){
    struct mesh_queue *queue=&provider->queues[q];
    queue->completion=provider->completion;
    queue->poll=provider->completion->context->ops.poll_cq;
    struct ibv_qp_init_attr qi={.send_cq=queue->completion,
      .recv_cq=queue->completion,.qp_type=IBV_QPT_UC,
      .cap={.max_send_wr=frame_capacity,.max_recv_wr=frame_capacity,.max_send_sge=1,.max_recv_sge=1}};
    queue->pair=ibv_create_qp(provider->device->domain,&qi);
    if(!queue->pair)return -1;
    queue->send=queue->pair->context->ops.post_send;queue->receive=queue->pair->context->ops.post_recv;
    provider->qp_count=q+1;
    struct ibv_qp_attr queried;struct ibv_qp_init_attr actual;
    if(ibv_query_qp(queue->pair,&queried,IBV_QP_CAP,&actual))return -1;
    queue->receive_capacity=actual.cap.max_recv_wr;queue->send_capacity=actual.cap.max_send_wr;
    fprintf(stderr,"pair capacity queue=%d send_frames=%u receive_frames=%u cq_entries=%d\n",q,
      actual.cap.max_send_wr,actual.cap.max_recv_wr,
      queue->completion->cqe);
  }
  struct ibv_qp_attr a={.qp_state=IBV_QPS_INIT,.port_num=1};
  for(int q=0;q<qps;q++) if(ibv_modify_qp(provider->queues[q].pair,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS))return -1;
  union ibv_gid gid; if(ibv_query_gid(provider->device->context,1,0,&gid))return -1;
  struct qpi mine={.xmagic=XMAGIC+MESH_VERSION,.xsize=sizeof mine,.lid=pa.lid,.pgsz=m->block*m->pgsz,.node=m->node,.count=(uint32_t)qps,
    .key=provider->key},you;
  memcpy(mine.gid,&gid,16);
  int f=oob(provider,&mine,&you);
  if(f<0){if(errno!=ECANCELED)fprintf(stderr,"exchange failed: %s\n",strerror(errno));return -1;}
  /* design/collective-dependency-ledger.md#d6-paired-send-and-receive-frame-counts-match */
  if(you.xmagic!=mine.xmagic || you.xsize!=sizeof you || you.pgsz!=mine.pgsz || you.count!=mine.count || you.node!=provider->peer || you.key!=mine.key){
    fprintf(stderr,"exchange mismatch: local=%u,%u,%u,%u,%u,%llu peer=%u,%u,%u,%u,%u,%llu expected_node=%d\n",mine.xmagic,mine.xsize,mine.pgsz,mine.count,mine.node,
      (unsigned long long)mine.key,you.xmagic,you.xsize,you.pgsz,you.count,you.node,(unsigned long long)you.key,provider->peer);
    close(f);errno=EPROTO;return -1;
  }
  uint32_t psn=arc4random()&0xffffff;
  for(int q=0;q<qps;q++){
    uint32_t local[2]={provider->queues[q].pair->qp_num,(psn+(uint32_t)q)&0xffffff},remote[2];
    if(exchange(f,local,remote,sizeof local,sizeof remote,provider)){close(f);return -1;}
    struct ibv_qp_attr r={.qp_state=IBV_QPS_RTR,.path_mtu=IBV_MTU_4096,.rq_psn=remote[1],
      .dest_qp_num=remote[0],.ah_attr={.dlid=you.lid,.port_num=1,.is_global=1,
        .grh={.hop_limit=1,.sgid_index=0}}};
    memcpy(&r.ah_attr.grh.dgid,you.gid,16);
    int rc=ibv_modify_qp(provider->queues[q].pair,&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN);
    if(rc){fprintf(stderr,"rtr %d rc %d\n",q,rc);close(f);return -1;}
  }
  if(configure(state,f)){int error=errno;close(f);errno=error;return -1;}
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
