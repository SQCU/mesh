#pragma once
#include "mesh.h"
#include <infiniband/verbs.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <ifaddrs.h>
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
   virtual-address boundary.  The window is mapped at a bank-aligned base and is required to fit in
   one bank, which makes every region cut interior to that bank and a straddle impossible. */
#define MESH_BANK ((size_t)1<<32)
/* design/prepared-machine.md#M09; RDMA-FIRST.md#the-ledger */
#define MESH_REGION ((size_t)1<<30)
/* device_gid's and port_bandwidth's inputs, verbatim from tag pre-return-20261103 etc/bridge/thunderbolt.h (5eabe91e) */
#define MESH_GID_ADDRESSES 16
#define MESH_GID_ENTRIES 256
/* ibv_port_attr speed/width codes, in bits/second and lanes respectively. */
#define MESH_PORT_SPEEDS { [1]=2500000000, [2]=5000000000, [4]=10000000000, [8]=10000000000, [16]=14000000000, [32]=25000000000, [64]=50000000000, [128]=100000000000 }
#define MESH_PORT_WIDTHS { [1]=1, [2]=4, [4]=8, [8]=12 }
struct mesh_wire { char *data; size_t length; };
/* A port's device. Added from tag pre-return-20261103 rdma/mesh-verbs.h struct mesh_device: the queue pairs it holds
   against those it can use (`pairs`, `max_pairs`: its max_qp less MESH_PROVIDER_RESERVED_QPS, device_up; 76be3217,
   f35a0447, the rule of bd85613e design/HETEROGENEOUS-LINKS.md), and its port's attributes and own address (`port`,
   `gid`, `gid_index`, `gid_matched`, as device_gid chooses it), which the child's open, make and connect read. `name`
   is the tag's char[32], which ports_find and child_main fill. */
struct mesh_device {
  char name[32];
  struct ibv_context *context; struct ibv_pd *domain; struct ibv_mr **regions;
  uint32_t region_count,frame_capacity,pairs,max_pairs;
  char *wire; size_t extent,payload;
  struct ibv_port_attr port;
  union ibv_gid gid;
  uint8_t gid_index,gid_matched;
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
/* `sent` and `send_capacity` from tag pre-return-20261103 rdma/mesh-verbs.h struct mesh_queue: a queue pair pair_up
   makes completes its receives on `completion`, its SENDs on `sent`. The device completes every SEND, IBV_SEND_SIGNALED
   or not (the bridge's census: send completions equal requests), so a SEND's completion is its queue's retirement and
   lands where the send thread polls it. */
struct mesh_queue {
  _Alignas(64) struct ibv_qp *pair;
  struct ibv_cq *completion,*sent;
  int (*poll)(struct ibv_cq *,int,struct ibv_wc *);
  int (*send)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **);
  int (*receive)(struct ibv_qp *,struct ibv_recv_wr *,struct ibv_recv_wr **);
  uint32_t receive_capacity,send_capacity;
};
_Static_assert(sizeof(struct mesh_queue)==64 && _Alignof(struct mesh_queue)==64,"mesh_queue native dispatch");
/* `count`, `sent`, `psn` and `poll` from tag pre-return-20261103 rdma/mesh-verbs.h struct mesh_verbs (76be3217): a
   link's queue pairs on one port (`count` entries of `queues`, by lane, an entry of a lane left out of its plan holding
   none) and their two completion queues, polled through `poll`. */
struct mesh_verbs {
  struct mesh_device *device;
  struct mesh_queue *queues; int qp_count,count;
  struct ibv_cq *completion,*sent;
  uint32_t psn;
  int (*poll)(struct ibv_cq *,int,struct ibv_wc *);
};
/* design/prepared-machine.md#M07 */
/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M09 */
/* design/RDMA-KERNEL-RECOVERY.md#tbt_post_recv */
/* __attribute__((unused)) on this, die and the functions below, as the tag's wire_map and die: every file of the
   resident bridge includes this header, and one that calls none of them builds with -Werror */
static __attribute__((unused)) int wire_map(struct mesh_wire *wire,struct hdr *m,int file){
  wire->length=(size_t)m->wire_pages*m->pgsz;
  wire->data=NULL;
  if(wire->length>MESH_BANK){errno=ENOMEM;return -1;}
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
/* tag pre-return-20261103 rdma/mesh-verbs.h:125-126, verbatim (CONTRACT: one shared stop; the --link form's static went) */
/* the stop flag, defined once in mesh-link.c (design/transport.md) */
extern _Atomic sig_atomic_t stop;
/* 2026-11-04: down_pair stays deleted, under every name: b89e6362 deleted its header so this bridge fails at compile
   instead of at run time, and its orphaned body is deleted now. It destroyed queue pairs with receives still posted (as
   a failing run leaves them), the action behind the Mini's kernel data aborts of 10-16/17, inside an unbounded retry
   (link_close's while(!down_pair(...))). Its replacement is the port child's end, mesh-child.c's lanes_flush and exit
   (tag pre-return-20261103 rdma/mesh-link.c lanes_flush and child_leave; 437e2c38, 60be5eb7, bc17811b): only the queue
   pairs with work outstanding moved to the error state, their completions taken until 0.5 s quiet or 5 s, the region's
   registrations released, then the child's exit, at which the kernel releases its queue pairs (pair_up below: made once
   in a child's life). No queue pair is destroyed while a peer may post. */
static __attribute__((unused)) void die(const char*m){ fprintf(stderr,"%s\n",m); exit(1); }

/* design/algorithm-sources.md#programcopy */
/* design/prepared-machine.md#M09 */
/* 2026-11-04, for the resident bridge's device_open (mesh-lanes.c), as tag pre-return-20261103 rdma/mesh-verbs.h
   device_up does: the port's attributes are kept in the device (`device->port`: device_gid, the make and the child's
   CHILD_OPENED read them), and a registration the device refuses sets *refused to its code (the window past the
   device's max_mr_size or max_mr, ENOMEM; else ibv_reg_mr's errno), so port_opened holds off its next try, each of which
   wires more of the node's memory. The rest is main's: registration stays the window (wire_pages), not the tag's
   whole-arena extents, and it returns -1 with errno, the tag's codes where the call set none (errno?errno:EIO). Like
   the tag's device_up it holds no lock, since a port's child opens its one device on its port's serial queue. */
static __attribute__((unused)) int device_up(struct mesh_device *device,struct mesh_wire *wire,struct hdr *m,int *refused){
  int error=0;
  *refused=0;
  if(!device->context){
    struct ibv_device **list=ibv_get_device_list(NULL);
    for(int i=0;list && list[i];i++)if(!strcmp(device->name,ibv_get_device_name(list[i]))){device->context=ibv_open_device(list[i]);break;}
    if(list)ibv_free_device_list(list);
    if(!device->context){error=errno?errno:ENODEV;goto done;}
  }
  if(ibv_query_port(device->context,1,&device->port)){error=errno?errno:EIO;goto done;}
  if(device->port.state!=IBV_PORT_ACTIVE){error=ENETDOWN;goto done;}
  if(device->frame_capacity)goto done;
  struct ibv_device_attr capabilities;
  if(ibv_query_device(device->context,&capabilities)){error=errno?errno:EIO;goto done;}
  if(!device->domain)device->domain=ibv_alloc_pd(device->context);
  if(!device->domain){error=errno?errno:EIO;goto done;}
  /* design/prepared-machine.md#M09 */
  /* Registered memory is the declared window, not the arena: it is what an SGE names, it is wired
     1:1 by the provider, and it is the only thing the MR table has to cover.  The addressable arena
     grows without it. */
  size_t stride=(size_t)m->block*m->pgsz,span=wire->length;
  /* design/prepared-machine.md#M09; RDMA-FIRST.md#the-ledger */
  const size_t most=device->extent?device->extent:MESH_REGION;
  size_t extent=(most<span?most:span)/stride*stride;
  size_t regions=extent?(span+extent-1)/extent:0;
  /* extent is a whole number of blocks and the window is one bank, so no region cut falls inside a
     block and no region crosses a 4 GiB boundary: wire_span's divide is the only addressing form. */
  if(!extent || regions>(size_t)capabilities.max_mr || span>MESH_BANK ||
     ((uintptr_t)wire->data&(MESH_BANK-1))+span>MESH_BANK){
    error=ENOMEM;*refused=error;
    fprintf(stderr,"register %s window=%zu extent=%zu regions=%zu base=%p max_mr_size=%llu max_mr=%d\n",
      device->name,span,extent,regions,(void *)wire->data,(unsigned long long)capabilities.max_mr_size,capabilities.max_mr);
    goto done;
  }
  if(!device->regions)device->regions=calloc(regions,sizeof *device->regions);
  if(!device->regions){error=ENOMEM;goto done;}
  while(device->region_count<regions){
    size_t offset=(size_t)device->region_count*extent,end=offset+extent;
    device->regions[device->region_count]=ibv_reg_mr(device->domain,wire->data+offset,(end<span?end:span)-offset,IBV_ACCESS_LOCAL_WRITE);
    if(!device->regions[device->region_count]){
      error=errno?errno:EIO;*refused=error;fprintf(stderr,"register %s offset=%zu bytes=%zu window=%zu extent=%zu max_mr_size=%llu max_mr=%d: %s\n",
        device->name,offset,(end<span?end:span)-offset,span,extent,(unsigned long long)capabilities.max_mr_size,capabilities.max_mr,strerror(error));
      /* design/prepared-machine.md#M09 */
      if(!device->region_count && extent>capabilities.max_mr_size){
        free(device->regions);device->regions=NULL;device->extent=(size_t)capabilities.max_mr_size;
      }
      goto done;
    }
    device->region_count++;
  }
  device->wire=wire->data;device->extent=extent;device->payload=stride;
  fprintf(stderr,"register %s window=%zu bytes extent=%zu regions=%zu arena=%llu bytes\n",
    device->name,span,extent,regions,(unsigned long long)mesh_arena_pages(m)*(uint64_t)m->pgsz);
  /* the queue pair budget, verbatim from tag pre-return-20261103 rdma/mesh-verbs.h device_up (f35a0447; 76be3217's
     max_qp - 1, the rule of bd85613e design/HETEROGENEOUS-LINKS.md): the queue pairs this device can use, its max_qp less
     those its provider holds itself. Both ends' smaller is the pairing's (mesh-ports.c port_heard). */
  device->max_pairs=capabilities.max_qp>MESH_PROVIDER_RESERVED_QPS?
    (uint32_t)capabilities.max_qp-MESH_PROVIDER_RESERVED_QPS:0;
  uint32_t capacity=capabilities.max_qp_wr<QD?capabilities.max_qp_wr:QD;
  if(capabilities.max_cqe<=1 || !capacity){error=EOPNOTSUPP;goto done;}
  device->frame_capacity=capacity>=(uint32_t)capabilities.max_cqe?(uint32_t)capabilities.max_cqe-1:capacity;
done:
  if(error){errno=error;return -1;}
  return 0;
}
/* device_gid: verbatim from tag pre-return-20261103 rdma/mesh-verbs.h (de96b545, 672f6f7f, 5eabe91e) */
/* the port's own address (its GID), its index in the port's GID table, and whether it is an address the port's
   interface holds (`gid_matched`), chosen from the whole table (up to 256 entries) against the interface's addresses:
   the entry equal to an IPv6 link-local address `interface` holds (getifaddrs, compared as bytes; BSD embeds the scope
   in bytes 2-3 of a link-local address it reports, which the comparison clears), the address mesh-fabric-init assigns
   and the peer at the cable's other end reaches (README "Addressing").  Where no entry is one of those (an interface not
   yet addressed), the table's link-local entry (fe80::/10), else its
   first nonzero entry, `gid_matched` 0 so its opening says so.  A Thunderbolt port's table does not always begin at
   index 0 (the Mini's en4 after its 2026-10-16 restart held its only entry at 1, and index 0's empty address failed
   every queue pair's connection with EIO), and its first entry is not always its address: the M5 Ultra Studio's macOS
   27 rdma_en3 (2026-10-21) held [0] 23ff:9f:174a:401d:8d37:9dff:16ac:4594, [1] its interface's fe80:: link-local, [2]
   its fd6d ULA, and the Mini (macOS 26) failed every lane's connection toward [0] with EIO */
static __attribute__((unused)) int device_gid(struct mesh_device *device,const char *interface){
  enum { HELD=MESH_GID_ADDRESSES, ENTRIES=MESH_GID_ENTRIES };
  struct in6_addr held[HELD];
  union ibv_gid table[ENTRIES];
  int helds=0,chosen=-1,matched=0,linklocal=-1,first=-1;
  struct ifaddrs *all=NULL;
  if(!getifaddrs(&all)){
    for(const struct ifaddrs *a=all;a && helds<HELD;a=a->ifa_next){
      if(!a->ifa_addr || a->ifa_addr->sa_family!=AF_INET6 || strcmp(a->ifa_name,interface))continue;
      struct in6_addr x=((const struct sockaddr_in6 *)(const void *)a->ifa_addr)->sin6_addr;
      if(!IN6_IS_ADDR_LINKLOCAL(&x))continue;
      x.s6_addr[2]=x.s6_addr[3]=0;
      held[helds++]=x;
    }
    freeifaddrs(all);
  }
  const int entries=device->port.gid_tbl_len<ENTRIES?device->port.gid_tbl_len:ENTRIES;
  for(int i=0;i<entries;i++){
    int nonzero=0;
    if(ibv_query_gid(device->context,1,i,table+i))continue;
    for(int b=0;b<16;b++)nonzero|=table[i].raw[b];
    if(!nonzero)continue;
    if(first<0)first=i;
    if(linklocal<0 && table[i].raw[0]==0xfe && (table[i].raw[1]&0xc0)==0x80)linklocal=i;
    for(int h=0;h<helds && chosen<0;h++)if(!memcmp(table[i].raw,held[h].s6_addr,16)){chosen=i;matched=1;}
  }
  if(chosen<0)chosen=linklocal>=0?linklocal:first;
  if(chosen<0)return ENXIO;
  device->gid=table[chosen];device->gid_index=(uint8_t)chosen;device->gid_matched=(uint8_t)matched;
  return 0;
}
/* pair_make: verbatim from tag pre-return-20261103 rdma/mesh-verbs.h (ceff26ec) */
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
/* pair_up: verbatim from tag pre-return-20261103 rdma/mesh-verbs.h (76be3217) */
/* a UC queue pair in INIT for each lane of `plan` (a bit a lane, design/HETEROGENEOUS-LINKS.md), queue q lane q, against
   the device's usable queue pairs, on the provider's two completion queues: a link's, made once in its port child's life
   (design/transport.md section 14: no queue pair is moved to another state, destroyed or forgotten; the kernel releases
   them at the child's exit).  0, or the error (ENOSPC: the device holds no more queue pairs). */
static __attribute__((unused)) int pair_up(struct mesh_verbs *provider,uint32_t plan){
  struct mesh_device *device=provider->device;
  const uint32_t count=(uint32_t)__builtin_popcount(plan),entries=plan?32-(uint32_t)__builtin_clz(plan):0;
  if(device->pairs+count>device->max_pairs)return ENOSPC;
  uint32_t frames=device->frame_capacity;
  provider->queues=calloc(entries?entries:1,sizeof *provider->queues);
  if(!provider->queues)return ENOMEM;
  provider->count=(int)entries;provider->psn=arc4random()&0xffffff;
  if(!provider->completion && !(provider->completion=ibv_create_cq(device->context,(int)(frames+1),NULL,NULL,0)))return errno?errno:EIO;
  if(!provider->sent && !(provider->sent=ibv_create_cq(device->context,(int)(frames+1),NULL,NULL,0)))return errno?errno:EIO;
  provider->poll=provider->completion->context->ops.poll_cq;
  for(uint32_t q=0;q<entries;q++){
    if(!(plan>>q&1))continue;
    const int error=pair_make(provider,(int)q);
    if(provider->queues[q].pair)provider->qp_count++;
    if(error)return error;
  }
  return 0;
}
/* queue_receive, queue_send: verbatim from tag pre-return-20261103 rdma/mesh-verbs.h (ceff26ec) */
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
/* port_bandwidth: verbatim from tag pre-return-20261103 rdma/mesh-verbs.h (5eabe91e; verbs_up's tables before it) */
/* design/algorithm-sources.md#link */
static __attribute__((unused)) uint64_t port_bandwidth(const struct ibv_port_attr *port){
  static const uint64_t speeds[UINT8_MAX+1]=MESH_PORT_SPEEDS;
  static const uint8_t widths[UINT8_MAX+1]=MESH_PORT_WIDTHS;
  return speeds[port->active_speed]*widths[port->active_width];
}
