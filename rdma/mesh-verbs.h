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
/* design/bridge-and-ipc.md#ports: a port's device: its context, protection domain and the whole window's `extents`
   registered on it (design/prepared-machine.md#M09), the queue pairs it holds against those it can use (`max_pairs`:
   its max_qp less one, RDMA-FIRST.md's "max_qp: 11 (10 usable)"), a queue pair's frames and a completion queue's
   entries at most */
struct mesh_device {
  char name[32];
  struct ibv_context *context; struct ibv_pd *domain; struct ibv_mr **regions;
  uint32_t extents,frame_capacity,pairs,max_pairs,max_cqe;
  char *wire;
  struct ibv_port_attr port;
  union ibv_gid gid;
  uint8_t gid_index;
};
/* the port's own address (its GID) and its index in the port's GID table, the first entry the table holds: a Thunderbolt
   port's table does not always begin at index 0 (the Mini's en4 after its 2026-10-16 restart held its only entry at 1, and
   index 0's empty address failed every queue pair's connection with EIO) */
static int device_gid(struct mesh_device *device){
  for(int i=0;i<device->port.gid_tbl_len && i<256;i++){
    union ibv_gid g;
    if(ibv_query_gid(device->context,1,i,&g))continue;
    for(int b=0;b<16;b++)if(g.raw[b]){device->gid=g;device->gid_index=(uint8_t)i;return 0;}
  }
  return ENXIO;
}
/* design/prepared-machine.md#M09 */
/* design/prepared-machine.md#M07 */
/* The one copy of the wire-address arithmetic.  offset is a byte offset into the arena, which is a
   byte offset into the registered window because the window begins at data_off; a span never crosses
   a block, so it never crosses an extent. */
static inline struct ibv_sge wire_span(const struct mesh_device *device,uint64_t payload,uint64_t offset,uint64_t remaining){
  uint64_t capacity=payload-offset%payload;
  return (struct ibv_sge){.addr=(uintptr_t)device->wire+offset,
    .length=(uint32_t)(remaining<capacity?remaining:capacity),
    .lkey=device->regions[offset/MESH_REGION]->lkey};
}
/* design/prepared-machine.md#M07 */
/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M09 */
/* design/RDMA-KERNEL-RECOVERY.md#tbt_post_recv */
static int wire_map(struct wire_mapping *wire,struct hdr *m,int file){
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
static const char *shm; static _Atomic sig_atomic_t stop;
/* design/RDMA-KERNEL-RECOVERY.md: registrations, protection domain, context, each kept where its release fails; a
   device's registrations go only here, at its retirement (design/prepared-machine.md#M09) */
static int down_device(struct mesh_device *device){
  for(uint32_t e=0;e<device->extents;e++){
    if(device->regions[e] && ibv_dereg_mr(device->regions[e]))return 0;
    device->regions[e]=NULL;
  }
  free(device->regions);device->regions=NULL;device->extents=0;
  if(device->domain){if(ibv_dealloc_pd(device->domain))return 0;device->domain=NULL;}
  if(device->context){if(ibv_close_device(device->context))return 0;device->context=NULL;}
  device->frame_capacity=0;
  return 1;
}
static void down(void){ if(shm)shm_unlink(shm); }
/* design/recovery.md#diagnosis: every line the bridge writes, stamped with the wall clock (UTC, to the millisecond) and
   the node's uptime in seconds, in one write so that no thread splits another's line */
static void say(const char *format,...){
  struct timespec t;clock_gettime(CLOCK_REALTIME,&t);
  struct tm utc;gmtime_r(&t.tv_sec,&utc);
  char line[2048];
  size_t n=strftime(line,sizeof line,"%Y-%m-%dT%H:%M:%S",&utc);
  n+=(size_t)snprintf(line+n,sizeof line-n,".%03ldZ %.3f ",t.tv_nsec/1000000,(double)clock_gettime_nsec_np(CLOCK_MONOTONIC)/1e9);
  va_list arguments;va_start(arguments,format);vsnprintf(line+n,sizeof line-n,format,arguments);va_end(arguments);
  fputs(line,stderr);
}
static void die(const char*m){ say("%s\n",m); exit(1); }

/* design/algorithm-sources.md#programcopy */
/* design/prepared-machine.md#M09 */
/* design/bridge-and-ipc.md#ports: the port's device opened (by its name, the device list read again) and its port
   active, its protection domain, the whole window registered on it and its capacities read once a context, before any
   queue pair exists (design/prepared-machine.md#M09), on the port's own queue (design/recovery.md Q14).  0, or the
   errno (ENETDOWN: the port is not active; ENODEV: no such device now).  A registration the device refuses sets
   *refused to its code (EFBIG: the window's extents past the device's max_mr; else ibv_reg_mr's errno), logged with
   the extent, and keeps the device and the extents it registered, so its next turn registers the rest
   (design/recovery.md#registration). */
static int device_up(struct mesh_device *device,struct wire_mapping *wire,int *refused){
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
  for(size_t e=0,first=0;e<extents && !stop;e++,first+=MESH_REGION){
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
  if(stop)return ECANCELED;
  say("%s: window of %zu bytes (%u extents) registered in %.3f s\n",device->name,wire->length,device->extents,
    (double)(clock_gettime_nsec_np(CLOCK_MONOTONIC)-began)/1e9);
  device->wire=wire->data;device->max_pairs=capabilities.max_qp>1?(uint32_t)capabilities.max_qp-1:0;
  uint32_t capacity=capabilities.max_qp_wr<QD?capabilities.max_qp_wr:QD;
  if(capabilities.max_cqe<=1 || !capacity)return EOPNOTSUPP;
  device->frame_capacity=capacity>=(uint32_t)capabilities.max_cqe?(uint32_t)capabilities.max_cqe-1:capacity;
  device->max_cqe=(uint32_t)capabilities.max_cqe;
  return 0;
}
/* design/bridge-and-ipc.md#links: a link's two completion queues, its SENDs' and its receives', which every queue pair
   of the link shares; 0, or the errno */
static int link_cqs(struct mesh_device *device,uint32_t entries,struct ibv_cq **sent,struct ibv_cq **received){
  if(!(*sent=ibv_create_cq(device->context,(int)entries,NULL,NULL,0)))return errno?errno:EIO;
  if(!(*received=ibv_create_cq(device->context,(int)entries,NULL,NULL,0))){const int error=errno?errno:EIO;ibv_destroy_cq(*sent);*sent=NULL;return error;}
  return 0;
}
/* design/bridge-and-ipc.md#links: a UC queue pair on the link's completion queues, in INIT, its capacities read; 0, or
   the errno (ENOSPC: the device holds no more) */
static int pair_make(struct mesh_device *device,struct ibv_cq *sent,struct ibv_cq *received,struct ibv_qp **made,uint32_t *sends,uint32_t *receives){
  if(device->pairs>=device->max_pairs)return ENOSPC;
  struct ibv_qp_init_attr qi={.send_cq=sent,.recv_cq=received,.qp_type=IBV_QPT_UC,
    .cap={.max_send_wr=device->frame_capacity,.max_recv_wr=device->frame_capacity,.max_send_sge=1,.max_recv_sge=1}};
  struct ibv_qp *qp=ibv_create_qp(device->domain,&qi);
  if(!qp)return errno?errno:EIO;
  device->pairs++;
  struct ibv_qp_attr queried;struct ibv_qp_init_attr actual;
  struct ibv_qp_attr a={.qp_state=IBV_QPS_INIT,.port_num=1};
  if(ibv_query_qp(qp,&queried,IBV_QP_CAP,&actual) || ibv_modify_qp(qp,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)){
    const int error=errno?errno:EIO;
    if(!ibv_destroy_qp(qp))device->pairs--;
    return error;
  }
  *sends=actual.cap.max_send_wr;*receives=actual.cap.max_recv_wr;*made=qp;
  return 0;
}
/* design/bridge-and-ipc.md#links: a queue pair ready to receive from the peer's (`remote` its number, `remote_psn` its
   first sequence number), at `lid` and `gid`, and ready to send from `psn`; 0, or the errno */
static int pair_connect(struct mesh_device *device,struct ibv_qp *qp,uint32_t remote,uint32_t remote_psn,uint32_t psn,uint16_t lid,const uint8_t *gid){
  struct ibv_qp_attr r={.qp_state=IBV_QPS_RTR,.path_mtu=IBV_MTU_4096,.rq_psn=remote_psn,.dest_qp_num=remote,
    .ah_attr={.dlid=lid,.port_num=1,.is_global=1,.grh={.hop_limit=1,.sgid_index=device->gid_index}}};
  memcpy(&r.ah_attr.grh.dgid,gid,16);
  if(ibv_modify_qp(qp,&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN))return errno?errno:EIO;
  struct ibv_qp_attr t={.qp_state=IBV_QPS_RTS,.sq_psn=psn};
  if(ibv_modify_qp(qp,&t,IBV_QP_STATE|IBV_QP_SQ_PSN))return errno?errno:EIO;
  return 0;
}
/* design/RDMA-KERNEL-RECOVERY.md "Every teardown": a queue pair moved to the error state (its posted receives and SENDs
   end; the move returns, and nothing waits on what it flushes), then destroyed; 0, or the destruction's errno, its
   handle let go either way */
static int pair_destroy(struct mesh_device *device,struct ibv_qp *qp){
  if(!qp)return 0;
  struct ibv_qp_attr error={.qp_state=IBV_QPS_ERR};
  ibv_modify_qp(qp,&error,IBV_QP_STATE);
  if(ibv_destroy_qp(qp))return errno?errno:EIO;
  device->pairs--;
  return 0;
}
/* design/algorithm-sources.md#link */
static uint64_t port_bandwidth(const struct ibv_port_attr *port){
  static const uint64_t speeds[256]={[1]=2500000000,[2]=5000000000,[4]=10000000000,[8]=10000000000,[16]=14000000000,[32]=25000000000,[64]=50000000000,[128]=100000000000};
  static const uint8_t widths[256]={[1]=1,[2]=4,[4]=8,[8]=12};
  return speeds[port->active_speed]*widths[port->active_width];
}
