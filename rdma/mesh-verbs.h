#pragma once
#include "mesh.h"
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
#include <stdarg.h>
#include <os/os_sync_wait_on_address.h>
#include "mesh-disk.h"

/* A line of the bridge's log (its stderr): one write, made only as mesh-disk.h's log guard allows (past
   its cap the log moves to <path>.1; under the disk floor its lines are dropped, the first saying so). */
__attribute__((format(printf, 1, 2))) static void say(const char *format, ...) {
  char line[2048];
  va_list arguments;
  va_start(arguments, format);
  int n = vsnprintf(line, sizeof line, format, arguments);
  va_end(arguments);
  if (n < 0) return;
  if ((size_t)n >= sizeof line) n = (int)sizeof line - 1;
  if (mesh_log_room((size_t)n)) (void)!write(2, line, (size_t)n);
}

#define QD 4095
/* TN3205: a device's UC queue pairs (it reports max_qp 11, 10 usable); the sessions' of every link of the device
   are within them (mesh-flow.c) */
#define MESH_DEVICE_QPS 10
/* design/RDMA-KERNEL-RECOVERY.md#tbt_post_recv */
/* tbt_post_recv loads only the low 32 address bits of an SGE, so no registration may cross a 4 GiB
   virtual-address boundary.  The window is mapped at a bank-aligned base and cut into regions of
   MESH_REGION bytes, which divides the bank: every region lies inside one bank, however many banks
   the window spans.  1 GiB is the registration the provider takes (the RCA's 32 GiB and 6 GiB windows
   in 1 GiB regions; its advertised max_mr_size, 16.4 MB, is not enforced), so max_mr regions of it
   are the device's registrable memory. */
#define MESH_BANK ((size_t)1<<32)
#define MESH_REGION ((size_t)1<<30)
/* The device's discard buffer (mesh-flow.c communicator sessions): registered with the window, before any
   queue pair, since a receive lands only in memory registered before its queue pair was set up. */
#define MESH_DISCARD ((size_t)4<<20)
struct mesh_wire { char *data; size_t length; };
struct mesh_device {
  const char *name;
  struct ibv_context *context; struct ibv_pd *domain; struct ibv_mr **regions;
  uint32_t region_count,frame_capacity;
  char *wire; size_t extent;
  char *discard; struct ibv_mr *discard_region;
  pthread_mutex_t setup;
};
/* A queue pair and the frames it takes posted (its receives complete on its provider's `completion`, its SENDs on
   `sent`). */
struct mesh_queue { struct ibv_qp *pair; uint32_t receive_capacity,send_capacity; };
struct mesh_verbs {
  struct mesh_device *device; struct mesh_wire *wire;
  struct mesh_queue *queues; int qp_count,listener;
  struct ibv_cq *completion,*sent;
  uint32_t peer;
  uint64_t bandwidth;
  const char *local_address,*remote_address,*service;
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
  say("wire window=%zu bytes base=%p bank=%llu offset_in_bank=%llu\n",wire->length,(void *)wire->data,
    (unsigned long long)((uintptr_t)wire->data>>32),(unsigned long long)((uintptr_t)wire->data&(MESH_BANK-1)));
  return 0;
}
/* The region is removed at exit, unless kept for the next bridge (`keeping`: mesh-flow.c, a bridge that ends with
   clients attached). */
static const char *shm; static int keeping; static _Atomic sig_atomic_t stop;
/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#programcopy */
static int down_pair(struct mesh_verbs *provider){
  while(provider->qp_count){ struct ibv_qp *q=provider->queues[provider->qp_count-1].pair; if(q && ibv_destroy_qp(q))return 0; provider->queues[--provider->qp_count].pair=NULL; }
  if(provider->completion){
    if(ibv_destroy_cq(provider->completion))return 0;
    provider->completion=NULL;
  }
  if(provider->sent){
    if(ibv_destroy_cq(provider->sent))return 0;
    provider->sent=NULL;
  }
  free(provider->queues);provider->queues=NULL;
  return 1;
}
/* design/algorithm-sources.md#programcopy */
static int down_device(struct mesh_device *device){
  if(device->discard_region){if(ibv_dereg_mr(device->discard_region))return 0;device->discard_region=NULL;}
  if(device->discard){munmap(device->discard,MESH_DISCARD);device->discard=NULL;}
  while(device->region_count){
    if(ibv_dereg_mr(device->regions[device->region_count-1]))return 0;
    device->region_count--;
  }
  free(device->regions);device->regions=NULL;
  if(device->domain){if(ibv_dealloc_pd(device->domain))return 0;device->domain=NULL;}
  if(device->context){if(ibv_close_device(device->context))return 0;device->context=NULL;}
  return 1;
}
static void down(void){ if(shm && !keeping)shm_unlink(shm); }
static void die(const char*m){ say("%s\n",m); exit(1); }
static void onsig(int s){ (void)s; stop++; }

/* design/collective-dependency-ledger.md#d13-connection-metadata-is-setup-work */
struct qpi { uint32_t xmagic, xsize; uint32_t pgsz; uint16_t lid; uint8_t gid[16]; uint32_t node,count; };
#define XMAGIC 0x4d595048u

/* A link's communicator session (mesh-flow.c net_session) pairs for the bridge's lifetime.  A pairing goes on
   until it pairs or this bridge stops; never a clock.  The waits inside it poll at PAIRING_POLL_MS, a pace for
   noticing the stop. */
#define PAIRING_POLL_MS 10
/* design/algorithm-sources.md#programcopy */
static int pairing_active(void){
  if(stop){errno=ECANCELED;return 0;}
  return 1;
}

/* design/algorithm-sources.md#programcopy */
static int dial(struct addrinfo *a){
  if(!pairing_active())return -1;
  int f=socket(a->ai_family,SOCK_STREAM,0); if(f<0) return -1;
  if(fcntl(f,F_SETFL,O_NONBLOCK)<0)goto failed;
  if(connect(f,a->ai_addr,a->ai_addrlen)==0)return f;
  if(errno!=EINPROGRESS)goto failed;
  struct pollfd ready={.fd=f,.events=POLLOUT};
  while(pairing_active()){
    int status=poll(&ready,1,PAIRING_POLL_MS);
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
static int exchange(int f,const void *mine,void *you,size_t send_bytes,size_t receive_bytes){
  size_t sent=0,got=0;
  while(pairing_active()){
    if(sent==send_bytes && got==receive_bytes)return 0;
    struct pollfd ready={.fd=f,.events=(short)((sent<send_bytes?POLLOUT:0)|(got<receive_bytes?POLLIN:0))};
    if(poll(&ready,1,PAIRING_POLL_MS)<0 && errno!=EINTR)return -1;
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
/* The higher node accepts, the lower dials.  The listener lives only while its pairing waits: it closes
   the moment it accepts, so a peer dialing for its next pairing is refused (and dials again) until this
   node's next pairing listens, instead of landing in a backlog that nobody accepts and that is reset when
   the link closes. */
static int oob(struct mesh_verbs *provider,struct hdr *m){
  if(!pairing_active())return -1;
  if(m->node>provider->peer){
    if(provider->listener<0 && listener_up(provider))return -1;
    while(pairing_active()){
      struct pollfd waiting={.fd=provider->listener,.events=POLLIN};
      if(poll(&waiting,1,PAIRING_POLL_MS)<=0)continue;
      int f=accept(provider->listener,NULL,NULL);
      if(f>=0){
        close(provider->listener);provider->listener=-1;
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
    if(!pairing_active()){error=errno;break;}
    for(struct addrinfo *a=addresses;a;a=a->ai_next){
      socket=dial(a);error=errno;
      if(socket>=0 || error==ECANCELED || error==ENETDOWN)break;
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
  /* design/prepared-machine.md#M09 */
  /* Registered memory is the declared window (by default the whole arena): what an SGE names, wired
     1:1 by the provider, in regions of MESH_REGION bytes.  A provider that refuses a region that size
     is registered in regions of its advertised max_mr_size instead, at most max_mr of them. */
  size_t stride=(size_t)m->block*m->pgsz,span=wire->length;
  if(!device->extent)device->extent=(MESH_REGION<span?MESH_REGION:(span+stride-1))/stride*stride;
  for(;;){
    size_t extent=device->extent,regions=extent?(span+extent-1)/extent:0;
    if(!extent || regions>(size_t)capabilities.max_mr || (MESH_BANK%extent && span>MESH_BANK)){
      error=ENOMEM;
      say("register %s window=%zu extent=%zu regions=%zu base=%p max_mr_size=%llu max_mr=%d\n",
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
        error=errno;say("register %s offset=%zu bytes=%zu window=%zu extent=%zu max_mr_size=%llu max_mr=%d: %s\n",
          device->name,offset,(end<span?end:span)-offset,span,extent,(unsigned long long)capabilities.max_mr_size,capabilities.max_mr,strerror(error));
        refused=1;break;
      }
      device->region_count++;
    }
    if(!refused)break;
    size_t fallback=capabilities.max_mr_size/stride*stride;
    if(!fallback || fallback>=extent)goto done;
    while(device->region_count){
      if(ibv_dereg_mr(device->regions[device->region_count-1]))goto done;
      device->region_count--;
    }
    free(device->regions);device->regions=NULL;device->extent=fallback;error=0;
  }
  size_t extent=device->extent,regions=device->region_count;
  device->wire=wire->data;device->extent=extent;
  if(!device->discard){
    char *reserved=mmap(NULL,MESH_DISCARD+MESH_BANK,PROT_NONE,MAP_PRIVATE|MAP_ANON|MAP_NORESERVE,-1,0);
    if(reserved==MAP_FAILED){error=errno;goto done;}
    char *base=(char *)(((uintptr_t)reserved+MESH_BANK-1)&~(uintptr_t)(MESH_BANK-1));
    if(base>reserved)munmap(reserved,(size_t)(base-reserved));
    munmap(base+MESH_DISCARD,(size_t)(reserved+MESH_DISCARD+MESH_BANK-(base+MESH_DISCARD)));
    device->discard=mmap(base,MESH_DISCARD,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON|MAP_FIXED,-1,0);
    if(device->discard==MAP_FAILED){error=errno;device->discard=NULL;munmap(base,MESH_DISCARD);goto done;}
  }
  if(!device->discard_region && !(device->discard_region=ibv_reg_mr(device->domain,device->discard,MESH_DISCARD,IBV_ACCESS_LOCAL_WRITE))){error=errno?errno:ENOMEM;goto done;}
  say("register %s window=%zu bytes extent=%zu regions=%zu arena=%llu bytes\n",
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
static int verbs_up(struct mesh_verbs *provider,struct hdr *m,int qps,int (*configure)(void *,int),void *state){
  struct ibv_port_attr pa;
  if(device_up(provider->device,provider->wire,m,&pa))return -1;
  uint32_t frame_capacity=provider->device->frame_capacity;
  union ibv_gid gid;uint32_t psn=arc4random()&0xffffff;
  struct qpi mine={.xmagic=XMAGIC+MESH_VERSION,.xsize=sizeof mine,.lid=pa.lid,.pgsz=m->block*m->pgsz,.node=m->node,.count=(uint32_t)qps},you;
  int f;
  /* A connection reset before the peers' first exchange is a peer's abandoned attempt, not this
     pairing's failure: the pairing connects again. */
  for(;;){
    f=oob(provider,m);
    if(f<0)return -1;
    /* A peer host that dies silently (power, panic) becomes an EOF on this control socket within
       idle + interval x count seconds, the session's end (mesh-flow.c net_read); UC queue pairs report
       nothing (docs/elastic.md §1 in metal-microbench). */
    int on=1,idle=5,interval=1,count=3;
    if(setsockopt(f,SOL_SOCKET,SO_KEEPALIVE,&on,sizeof on) || setsockopt(f,IPPROTO_TCP,TCP_KEEPALIVE,&idle,sizeof idle) ||
       setsockopt(f,IPPROTO_TCP,TCP_KEEPINTVL,&interval,sizeof interval) || setsockopt(f,IPPROTO_TCP,TCP_KEEPCNT,&count,sizeof count)){
      int error=errno;close(f);errno=error;return -1;
    }
    if(!provider->queues){
      /* design/prepared-machine.md#M11 */
      provider->queues=calloc((size_t)qps,sizeof *provider->queues);
      if(!provider->queues){close(f);return -1;}
      provider->completion=ibv_create_cq(provider->device->context,
        (int)(frame_capacity+1),NULL,NULL,0);
      if(!provider->completion){close(f);return -1;}
      provider->sent=ibv_create_cq(provider->device->context,(int)(frame_capacity+1),NULL,NULL,0);
      if(!provider->sent){close(f);return -1;}
      for(int q=0;q<qps;q++){
        struct mesh_queue *queue=&provider->queues[q];
        struct ibv_qp_init_attr qi={.send_cq=provider->sent,
          .recv_cq=provider->completion,.qp_type=IBV_QPT_UC,
          .cap={.max_send_wr=frame_capacity,.max_recv_wr=frame_capacity,.max_send_sge=1,.max_recv_sge=1}};
        queue->pair=ibv_create_qp(provider->device->domain,&qi);
        if(!queue->pair){close(f);return -1;}
        provider->qp_count=q+1;
        struct ibv_qp_attr queried;struct ibv_qp_init_attr actual;
        if(ibv_query_qp(queue->pair,&queried,IBV_QP_CAP,&actual)){close(f);return -1;}
        queue->receive_capacity=actual.cap.max_recv_wr;queue->send_capacity=actual.cap.max_send_wr;
        say("session capacity queue=%d send_frames=%u receive_frames=%u cq_entries=%d\n",q,
          actual.cap.max_send_wr,actual.cap.max_recv_wr,
          provider->completion->cqe);
      }
      struct ibv_qp_attr a={.qp_state=IBV_QPS_INIT,.port_num=1};
      for(int q=0;q<qps;q++) if(ibv_modify_qp(provider->queues[q].pair,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)){ close(f); return -1; }
      if(ibv_query_gid(provider->device->context,1,0,&gid)){ close(f); return -1; }
      memcpy(mine.gid,&gid,16);
    }
    if(!exchange(f,&mine,&you,sizeof mine,sizeof you))break;
    int error=errno;close(f);
    if(error!=ECONNRESET && error!=EPIPE && error!=ENOTCONN && error!=ECONNABORTED){say("exchange failed\n");errno=error;return -1;}
    say("exchange reset: %s; connecting again\n",strerror(error));
  }
  /* design/collective-dependency-ledger.md#d6-paired-send-and-receive-frame-counts-match */
  if(you.xmagic!=mine.xmagic || you.xsize!=sizeof you || you.pgsz!=mine.pgsz || you.count!=mine.count || you.node!=provider->peer){
    say("exchange mismatch: local=%u,%u,%u,%u,%u peer=%u,%u,%u,%u,%u expected_node=%d\n",mine.xmagic,mine.xsize,mine.pgsz,mine.count,mine.node,you.xmagic,you.xsize,you.pgsz,you.count,you.node,provider->peer); close(f);errno=EPROTO;return -1; }
  for(int q=0;q<qps;q++){
    uint32_t local[2]={provider->queues[q].pair->qp_num,(psn+(uint32_t)q)&0xffffff},remote[2];
    if(exchange(f,local,remote,sizeof local,sizeof remote)){close(f);return -1;}
    struct ibv_qp_attr r={.qp_state=IBV_QPS_RTR,.path_mtu=IBV_MTU_4096,.rq_psn=remote[1],
      .dest_qp_num=remote[0],.ah_attr={.dlid=you.lid,.port_num=1,.is_global=1,
        .grh={.hop_limit=1,.sgid_index=0}}};
    memcpy(&r.ah_attr.grh.dgid,you.gid,16);
    int rc=ibv_modify_qp(provider->queues[q].pair,&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN);
    if(rc){say("rtr %d rc %d\n",q,rc);close(f);return -1;}
  }
  if(configure(state,f)){int error=errno;close(f);errno=error;return -1;}
  for(int q=0;q<qps;q++){
    struct ibv_qp_attr t={.qp_state=IBV_QPS_RTS,.sq_psn=(psn+(uint32_t)q)&0xffffff};
    int rc=ibv_modify_qp(provider->queues[q].pair,&t,IBV_QP_STATE|IBV_QP_SQ_PSN);
    if(rc){ say("rts %d rc %d, failed\n",q,rc); close(f); return -1; }
  }
  /* design/algorithm-sources.md#link */
  static const uint64_t speeds[256]={[1]=2500000000,[2]=5000000000,[4]=10000000000,[8]=10000000000,[16]=14000000000,[32]=25000000000,[64]=50000000000,[128]=100000000000};
  static const uint8_t widths[256]={[1]=1,[2]=4,[4]=8,[8]=12};
  provider->bandwidth=speeds[pa.active_speed]*widths[pa.active_width];
  say("session up: %s node %d\n",ibv_get_device_name(provider->device->context->device),m->node);
  return f; }
