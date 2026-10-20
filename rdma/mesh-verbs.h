#pragma once
#include "mesh.h"
#include <infiniband/verbs.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <signal.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

#define QD 4095
/* design/RDMA-KERNEL-RECOVERY.md#tbt_post_recv */
/* tbt_post_recv loads only the low 32 address bits of an SGE, so no registration may cross a 4 GiB
   virtual-address boundary.  The window is mapped at a bank-aligned base and registered in extents of
   MESH_REGION bytes, which divides the bank: every extent lies inside one bank, however many banks the
   window spans.  1 GiB is the registration the provider takes. */
#define MESH_BANK ((size_t)1<<32)
#define MESH_REGION ((size_t)1<<30)
struct wire_mapping { char *data; size_t length; };
/* design/transport.md: a port's device: its context, protection domain and the whole window's `extents`
   registered on it (design/transport.md), the queue pairs it holds against those it can use (`max_pairs`:
   its max_qp less one, design/transport.md's "max_qp: 11 (10 usable)") */
struct mesh_device {
  char name[32];
  struct ibv_context *context; struct ibv_pd *domain; struct ibv_mr **regions;
  uint32_t extents,frame_capacity,pairs,max_pairs;
  char *wire;
  struct ibv_port_attr port;
  union ibv_gid gid;
  uint8_t gid_index;
};
/* the port's own address (its GID) and its index in the port's GID table, the first entry the table holds: a Thunderbolt
   port's table does not always begin at index 0 (the Mini's en4 after its 2026-10-16 restart held its only entry at 1, and
   index 0's empty address failed every queue pair's connection with EIO) */
static __attribute__((unused)) int device_gid(struct mesh_device *device){
  for(int i=0;i<device->port.gid_tbl_len && i<256;i++){
    union ibv_gid g;
    if(ibv_query_gid(device->context,1,i,&g))continue;
    for(int b=0;b<16;b++)if(g.raw[b]){device->gid=g;device->gid_index=(uint8_t)i;return 0;}
  }
  return ENXIO;
}
/* design/transport.md */
/* design/transport.md */
/* The one copy of the wire-address arithmetic.  offset is a byte offset into the arena, which is a
   byte offset into the registered window because the window begins at data_off; a span never crosses
   a block, so it never crosses an extent. */
static inline struct ibv_sge wire_span(const struct mesh_device *device,uint64_t payload,uint64_t offset,uint64_t remaining){
  uint64_t capacity=payload-offset%payload;
  return (struct ibv_sge){.addr=(uintptr_t)device->wire+offset,
    .length=(uint32_t)(remaining<capacity?remaining:capacity),
    .lkey=device->regions[offset/MESH_REGION]->lkey};
}
/* design/algorithm-sources.md#programcopy */
/* A queue pair: its receives complete on `completion`, its SENDs on `sent`.  The device completes
   every SEND, IBV_SEND_SIGNALED or not (the bridge's census: send completions equal requests), so a
   SEND's completion is its queue's retirement and lands where the send thread polls it. */
struct mesh_queue {
  _Alignas(64) struct ibv_qp *pair;
  struct ibv_cq *completion,*sent;
  int (*poll)(struct ibv_cq *,int,struct ibv_wc *);
  int (*send)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **);
  int (*receive)(struct ibv_qp *,struct ibv_recv_wr *,struct ibv_recv_wr **);
  uint32_t receive_capacity,send_capacity;
};
_Static_assert(sizeof(struct mesh_queue)==64 && _Alignof(struct mesh_queue)==64,"mesh_queue native dispatch");
/* a program's queue pairs on one port and their two completion queues */
struct mesh_verbs { struct mesh_device *device; struct mesh_queue *queues; int qp_count,count; struct ibv_cq *completion,*sent; uint32_t psn; };
/* design/transport.md */
/* design/algorithm-sources.md#programtensor */
/* design/transport.md */
/* design/RDMA-KERNEL-RECOVERY.md#tbt_post_recv */
static __attribute__((unused)) int wire_map(struct wire_mapping *wire,struct hdr *m,int file){
  wire->length=(size_t)mesh_arena_pages(m)*m->pgsz;
  wire->data=NULL;
  char *reserved=mmap(NULL,wire->length+MESH_BANK,PROT_NONE,MAP_PRIVATE|MAP_ANON|MAP_NORESERVE,-1,0);
  if(reserved==MAP_FAILED)return -1;
  char *base=(char *)(((uintptr_t)reserved+MESH_BANK-1)&~(uintptr_t)(MESH_BANK-1));
  if(base>reserved)munmap(reserved,(size_t)(base-reserved));
  size_t tail=(size_t)((reserved+wire->length+MESH_BANK)-(base+wire->length));
  if(tail)munmap(base+wire->length,tail);
  wire->data=mmap(base,wire->length,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_FIXED,file,(off_t)m->data_off);
  if(wire->data==MAP_FAILED){wire->data=NULL;return -1;}
  return 0;
}
/* the region's name and the stop flag, defined once in mesh-link.c (design/transport.md) */
extern const char *shm; extern _Atomic sig_atomic_t stop;
/* design/RDMA-KERNEL-RECOVERY.md "October 18": a retired device's registrations, protection domain and context
   forgotten with no verbs call (ibv_dealloc_pd during the device's termination deadlocks Apple's driver, 09-05 and
   10-18); the kernel releases them at the process's exit */
static __attribute__((unused)) int down_device(struct mesh_device *device){
  free(device->regions);device->regions=NULL;device->extents=0;
  device->domain=NULL;
  device->context=NULL;
  device->frame_capacity=0;device->pairs=0;
  return 1;
}
static __attribute__((unused)) void down(void){ if(shm)shm_unlink(shm); }
/* design/transport.md: every line the bridge writes, stamped with the wall clock (UTC, to the millisecond) and
   the node's uptime in seconds, in one write so that no thread splits another's line */
static __attribute__((unused)) void say(const char *format,...){
  struct timespec t;clock_gettime(CLOCK_REALTIME,&t);
  struct tm utc;gmtime_r(&t.tv_sec,&utc);
  char line[2048];
  size_t n=strftime(line,sizeof line,"%Y-%m-%dT%H:%M:%S",&utc);
  n+=(size_t)snprintf(line+n,sizeof line-n,".%03ldZ %.3f ",t.tv_nsec/1000000,(double)clock_gettime_nsec_np(CLOCK_MONOTONIC)/1e9);
  va_list arguments;va_start(arguments,format);vsnprintf(line+n,sizeof line-n,format,arguments);va_end(arguments);
  fputs(line,stderr);
}
static __attribute__((unused)) void die(const char*m){ say("%s\n",m); exit(1); }

/* design/algorithm-sources.md#programcopy */
/* design/transport.md */
/* design/transport.md: the port's device opened (by its name, the device list read again) and its port
   active, its protection domain, the whole window registered on it and its capacities read once a context, before any
   queue pair exists (design/transport.md), on the port's own queue (design/transport.md).  0, or the
   errno (ENETDOWN: the port is not active; ENODEV: no such device now).  A registration the device refuses sets
   *refused to its code (EFBIG: the window's extents past the device's max_mr; else ibv_reg_mr's errno), logged with
   the extent, and keeps the device and the extents it registered, so its next turn registers the rest
   (design/transport.md). */
static __attribute__((unused)) int device_up(struct mesh_device *device,struct wire_mapping *wire,int *refused){
  *refused=0;
  if(!device->context){
    struct ibv_device **list=ibv_get_device_list(NULL);
    for(int i=0;list && list[i];i++)if(!strcmp(device->name,ibv_get_device_name(list[i]))){device->context=ibv_open_device(list[i]);break;}
    if(list)ibv_free_device_list(list);
    if(!device->context)return ENODEV;
  }
  if(ibv_query_port(device->context,1,&device->port))return errno?errno:EIO;
  if(device->port.state!=IBV_PORT_ACTIVE)return ENETDOWN;
  if(device->frame_capacity)return 0;
  struct ibv_device_attr capabilities;
  if(ibv_query_device(device->context,&capabilities))return errno?errno:EIO;
  if(!device->domain && !(device->domain=ibv_alloc_pd(device->context)))return errno?errno:EIO;
  const uint32_t extents=(uint32_t)((wire->length+MESH_REGION-1)/MESH_REGION);
  if((uint64_t)extents>(uint64_t)(capabilities.max_mr>0?capabilities.max_mr:0)){
    say("%s: the window of %zu bytes is %u extents of %zu bytes, past the device's max_mr %d: refused\n",device->name,
      wire->length,extents,MESH_REGION,capabilities.max_mr);
    *refused=EFBIG;return EFBIG;
  }
  if(!device->regions && !(device->regions=calloc(extents,sizeof *device->regions)))return ENOMEM;
  device->extents=extents;
  const uint64_t began=clock_gettime_nsec_np(CLOCK_MONOTONIC);
  for(size_t e=0,first=0;e<extents;e++,first+=MESH_REGION){
    const size_t bytes=wire->length-first<MESH_REGION?wire->length-first:MESH_REGION;
    if(!device->regions[e] && !(device->regions[e]=ibv_reg_mr(device->domain,wire->data+first,bytes,IBV_ACCESS_LOCAL_WRITE))){
      const int error=errno?errno:EIO;
      uint32_t held=0;
      for(uint32_t k=0;k<extents;k++)held+=device->regions[k]!=NULL;
      say("%s: extent %zu of %u (%zu bytes at %zu) refused: %s; %u of the window's extents registered, max_mr %d\n",
        device->name,e,extents,bytes,first,strerror(error),held,capabilities.max_mr);
      *refused=error;return error;
    }
  }
  say("%s: window of %zu bytes (%u extents) registered in %.3f s\n",device->name,wire->length,device->extents,
    (double)(clock_gettime_nsec_np(CLOCK_MONOTONIC)-began)/1e9);
  device->wire=wire->data;device->max_pairs=capabilities.max_qp>1?(uint32_t)capabilities.max_qp-1:0;
  uint32_t capacity=capabilities.max_qp_wr<QD?capabilities.max_qp_wr:QD;
  if(capabilities.max_cqe<=1 || !capacity)return EOPNOTSUPP;
  device->frame_capacity=capacity>=(uint32_t)capabilities.max_cqe?(uint32_t)capabilities.max_cqe-1:capacity;
  return 0;
}
/* design/transport.md */
/* queue pair q made on the provider's completion queues, UC, its capacities read, and moved to INIT (d6dceac verbs_up's
   attributes and mask, as every pairing since); the device's count of its queue pairs one more.  0, or the error */
static __attribute__((unused)) int pair_make(struct mesh_verbs *provider,int q){
  struct mesh_device *device=provider->device;
  const uint32_t frames=device->frame_capacity;
  struct mesh_queue *queue=&provider->queues[q];
  queue->completion=provider->completion;queue->sent=provider->sent;
  queue->poll=provider->completion->context->ops.poll_cq;
  struct ibv_qp_init_attr qi={.send_cq=queue->sent,.recv_cq=queue->completion,.qp_type=IBV_QPT_UC,
    .cap={.max_send_wr=frames,.max_recv_wr=frames,.max_send_sge=1,.max_recv_sge=1}};
  if(!(queue->pair=ibv_create_qp(device->domain,&qi)))return errno?errno:EIO;
  device->pairs++;
  queue->send=queue->pair->context->ops.post_send;queue->receive=queue->pair->context->ops.post_recv;
  struct ibv_qp_attr queried;struct ibv_qp_init_attr actual;
  if(ibv_query_qp(queue->pair,&queried,IBV_QP_CAP,&actual))return errno?errno:EIO;
  queue->receive_capacity=actual.cap.max_recv_wr;queue->send_capacity=actual.cap.max_send_wr;
  struct ibv_qp_attr a={.qp_state=IBV_QPS_INIT,.port_num=1};
  if(ibv_modify_qp(queue->pair,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS))return errno?errno:EIO;
  return 0;
}
/* `count` UC queue pairs in INIT, against the device's usable queue pairs, on the provider's two completion queues, made
   where it has none (a link's are made once a context and kept across its pairings: design/transport.md).
   0, or the error (ENOSPC: the device holds no more queue pairs). */
static __attribute__((unused)) int pair_up(struct mesh_verbs *provider,int count){
  struct mesh_device *device=provider->device;
  if(device->pairs+(uint32_t)count>device->max_pairs)return ENOSPC;
  uint32_t frames=device->frame_capacity;
  provider->queues=calloc((size_t)count,sizeof *provider->queues);
  if(!provider->queues)return ENOMEM;
  provider->count=count;provider->psn=arc4random()&0xffffff;
  if(!provider->completion && !(provider->completion=ibv_create_cq(device->context,(int)(frames+1),NULL,NULL,0)))return errno?errno:EIO;
  if(!provider->sent && !(provider->sent=ibv_create_cq(device->context,(int)(frames+1),NULL,NULL,0)))return errno?errno:EIO;
  for(int q=0;q<count;q++){
    const int error=pair_make(provider,q);
    if(provider->queues[q].pair)provider->qp_count=q+1;
    if(error)return error;
  }
  return 0;
}
/* design/algorithm-sources.md#programcopy */
/* queue pair q ready to receive from the peer's (`remote`: its queue pair number and first sequence number), the peer's
   port at `lid` and `gid`; ready to send */
static __attribute__((unused)) int queue_receive(struct mesh_verbs *provider,int q,const uint32_t *remote,uint16_t lid,const uint8_t *gid){
  struct ibv_qp_attr r={.qp_state=IBV_QPS_RTR,.path_mtu=IBV_MTU_4096,.rq_psn=remote[1],.dest_qp_num=remote[0],
    .ah_attr={.dlid=lid,.port_num=1,.is_global=1,.grh={.hop_limit=1,.sgid_index=provider->device->gid_index}}};
  memcpy(&r.ah_attr.grh.dgid,gid,16);
  if(ibv_modify_qp(provider->queues[q].pair,&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN))return errno?errno:EIO;
  return 0;
}
static __attribute__((unused)) int queue_send(struct mesh_verbs *provider,int q){
  struct ibv_qp_attr t={.qp_state=IBV_QPS_RTS,.sq_psn=(provider->psn+(uint32_t)q)&0xffffff};
  if(ibv_modify_qp(provider->queues[q].pair,&t,IBV_QP_STATE|IBV_QP_SQ_PSN))return errno?errno:EIO;
  return 0;
}
/* design/transport.md: the queue pairs of `lanes` (a bit a queue) moved to the error state, so the device
   ends their posted receives and SENDs (flushed: their completions name no holder, taken as any lane's failure or by a
   destruction's poll), nothing waiting on what it flushes (4df30806 pair_flush's modify loop, without its drain).  The
   lanes moved */
static __attribute__((unused)) uint32_t pairs_error(struct mesh_verbs *provider,uint32_t lanes){
  uint32_t moved=0;
  for(int q=0;q<provider->qp_count && q<32;q++)if((lanes>>q&1) && provider->queues[q].pair){
    struct ibv_qp_attr error={.qp_state=IBV_QPS_ERR};
    if(!ibv_modify_qp(provider->queues[q].pair,&error,IBV_QP_STATE))moved|=UINT32_C(1)<<q;
  }
  return moved;
}
/* design/transport.md section 14: a link's queue pairs, kept across its pairings, made ready for the next: each moved
   to RESET (which drops what it held) and to INIT again (pair_make's attributes), a fresh first sequence number drawn,
   and what its completion queues hold of the last pairing taken and dropped (a poll returns at once, 0 once none is).
   No queue pair is destroyed (design/transport.md section 5).  0, or the move's error */
static __attribute__((unused)) int pairs_reset(struct mesh_verbs *provider){
  for(int q=0;q<provider->qp_count;q++){
    struct ibv_qp *pair=provider->queues[q].pair;
    if(!pair)continue;
    struct ibv_qp_attr reset={.qp_state=IBV_QPS_RESET},init={.qp_state=IBV_QPS_INIT,.port_num=1};
    if(ibv_modify_qp(pair,&reset,IBV_QP_STATE) ||
       ibv_modify_qp(pair,&init,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS))return errno?errno:EIO;
  }
  provider->psn=arc4random()&0xffffff;
  struct ibv_wc stale[32];
  struct ibv_cq *queues[2]={provider->completion,provider->sent};
  for(int c=0;c<2;c++)while(queues[c] && ibv_poll_cq(queues[c],32,stale)>0){}
  return 0;
}
/* a provider whose context is gone forgotten with no verbs call: its queue pairs and completion queues are the kernel's
   to release at the process's exit (design/RDMA-KERNEL-RECOVERY.md "October 18") */
static __attribute__((unused)) void pair_forget(struct mesh_verbs *provider){
  free(provider->queues);provider->queues=NULL;provider->count=provider->qp_count=0;
  provider->completion=provider->sent=NULL;
}
/* design/algorithm-sources.md#link */
static __attribute__((unused)) uint64_t port_bandwidth(const struct ibv_port_attr *port){
  static const uint64_t speeds[256]={[1]=2500000000,[2]=5000000000,[4]=10000000000,[8]=10000000000,[16]=14000000000,[32]=25000000000,[64]=50000000000,[128]=100000000000};
  static const uint8_t widths[256]={[1]=1,[2]=4,[4]=8,[8]=12};
  return speeds[port->active_speed]*widths[port->active_width];
}
