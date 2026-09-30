#include "mesh-net.h"
#include "mesh-dataflow.h"
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* The client side of the communicator service (mesh-net.h): a comm is a slot of the region's table
   (mesh.h mesh_net_comm) this process claimed or accepted; a request is its comm's ring slot; the
   bridge's session thread of the comm's link serves both (mesh-flow.c).  Memory is the region's
   registered window: mesh_net_mem_alloc claims its pages, and regMr names the device's registered
   regions they lie in, so data lands in place and nothing here waits on the wire. */
#define NET_PENDING 256
#define NET_WAIT_NS UINT64_C(60000000000)
_Static_assert(sizeof(struct mesh_net_handle)<=MESH_NET_HANDLE_BYTES,"handle");
struct net_comm_handle;
struct net_request_handle { struct net_comm_handle *comm; uint64_t sequence; };
struct net_comm_handle {
  uint32_t index,generation,link,kind;
  uint64_t sequence;
  struct net_request_handle requests[MESH_NET_REQUESTS];
};
struct net_mhandle { uint32_t mr,generation; char *address; size_t size; uint64_t offset; };
struct net_pending { struct net_comm_handle *comm; uint64_t key; uint32_t node; int dev; };
struct net_context { uint64_t comm_id; int traffic_class; mesh_net_logger log; struct net_pending pending[NET_PENDING]; };
static struct {
  pthread_mutex_t lock;
  int inits;
  struct hdr *m; size_t length; uint64_t owner; uint32_t client;
  char **names;
  uint64_t checked;
} net = {.lock=PTHREAD_MUTEX_INITIALIZER};
static _Thread_local int net_errno;

static int net_result(int error){
  net_errno=error;
  switch(error){
  case 0: return MESH_NET_SUCCESS;
  case EINVAL: case EMSGSIZE: case ERANGE: return MESH_NET_INVALID_ARGUMENT;
  case EBUSY: case ENOTSUP: return MESH_NET_INVALID_USAGE;
  case ECONNRESET: case ECONNREFUSED: case ENETDOWN: case ECANCELED: case EPIPE: case EIO: case ENOSPC: return MESH_NET_REMOTE_ERROR;
  default: return MESH_NET_SYSTEM_ERROR;
  }
}
int mesh_net_error(void){return net_errno;}
static uint64_t net_now(void){return clock_gettime_nsec_np(CLOCK_MONOTONIC);}
static void net_ring(uint32_t link){
  struct mesh_net_link *l=mesh_net_links(net.m)+link;
  atomic_fetch_add_explicit(&l->doorbell,1,memory_order_release);
  os_sync_wake_by_address_any(&l->doorbell,sizeof l->doorbell,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
/* The bridge gone is every pending request's failure; looked at no more often than every 100 ms. */
static int net_bridge_gone(void){
  uint64_t now=net_now();
  if(now-net.checked<100000000)return 0;
  net.checked=now;
  pid_t pid=(pid_t)atomic_load_explicit(&net.m->bridge_pid,memory_order_relaxed);
  return !pid || (kill(pid,0) && errno==ESRCH);
}
static struct mesh_net_comm *net_slot(struct net_comm_handle *c){return mesh_net_comms(net.m)+c->index;}

int mesh_net_init(void **ctx,uint64_t commId,struct mesh_net_comm_config *config,mesh_net_logger logFunction,mesh_net_profiler profFunction){
  (void)profFunction;
  struct net_context *context=calloc(1,sizeof *context);
  if(!context)return net_result(ENOMEM);
  context->comm_id=commId;context->traffic_class=config?config->trafficClass:-1;context->log=logFunction;
  pthread_mutex_lock(&net.lock);
  int error=0;
  if(!net.m){
    const char *name=getenv("MESH_REGION");
    int file=shm_open(name?name:MESH_NAME,O_RDWR,MESH_MODE);
    struct stat info;
    if(file<0)error=errno;
    else if(fstat(file,&info))error=errno;
    struct hdr *m=MAP_FAILED;
    if(!error && (m=mmap(NULL,(size_t)info.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,file,0))==MAP_FAILED)error=errno;
    if(file>=0)close(file);
    if(!error && ((size_t)info.st_size<sizeof *m || m->magic!=MESH_MAGIC || m->version!=MESH_VERSION || m->length>(uint64_t)info.st_size || !m->net_off)){
      munmap(m,(size_t)info.st_size);error=EINVAL;
    }
    if(!error){
      net.m=m;net.length=(size_t)info.st_size;
      net.owner=(((atomic_fetch_add_explicit(&m->serial,1,memory_order_relaxed)+1)&UINT64_C(0x7fffffff))<<32)|(uint32_t)getpid();
      net.client=MESH_NET_CLIENTS;
      for(uint32_t i=0;i<MESH_NET_CLIENTS;i++){
        uint64_t vacant=0;
        if(atomic_compare_exchange_strong_explicit(&mesh_net_clients(m)[i].owner,&vacant,net.owner,memory_order_acq_rel,memory_order_relaxed)){net.client=i;break;}
      }
      net.names=calloc(m->links?m->links:1,sizeof *net.names);
    }
  }
  if(!error)net.inits++;
  pthread_mutex_unlock(&net.lock);
  if(error){free(context);return net_result(error);}
  *ctx=context;
  return MESH_NET_SUCCESS;
}

int mesh_net_devices(int *ndev){
  if(!net.m)return net_result(EBUSY);
  *ndev=(int)net.m->links;
  return MESH_NET_SUCCESS;
}

int mesh_net_get_properties(int dev,struct mesh_net_properties *props){
  if(!net.m)return net_result(EBUSY);
  if(dev<0 || (uint32_t)dev>=net.m->links)return net_result(EINVAL);
  struct mesh_link_info *link=mesh_links(net.m)+dev;
  pthread_mutex_lock(&net.lock);
  if(!net.names[dev]){
    char name[sizeof link->device+1];memcpy(name,link->device,sizeof link->device);name[sizeof link->device]=0;
    net.names[dev]=strdup(name);
  }
  pthread_mutex_unlock(&net.lock);
  uint64_t guid=UINT64_C(1469598103934665603);
  for(const char *c=net.names[dev];c && *c;c++)guid=(guid^(unsigned char)*c)*UINT64_C(1099511628211);
  *props=(struct mesh_net_properties){.name=net.names[dev],.pciPath=net.names[dev],.guid=guid^((uint64_t)net.m->node<<48),
    .ptrSupport=MESH_NET_PTR_HOST,.regIsGlobal=0,.forceFlush=0,
    .speed=(int)(__atomic_load_n(&link->bandwidth,__ATOMIC_RELAXED)/1000000),.port=1,.latency=0,
    .maxComms=MESH_NET_COMMS,.maxRecvs=1,.netDeviceType=0,.netDeviceVersion=0,.vProps={.ndevs=1,.devs={dev}},
    .maxP2pBytes=MESH_NET_MAX_SIZE_BYTES,.maxCollBytes=MESH_NET_MAX_SIZE_BYTES,.maxMultiRequestSize=1,.railId=(int16_t)dev,.planeId=0};
  return MESH_NET_SUCCESS;
}

/* A slot of the comm table claimed for this client on `link`, published by the caller. */
static struct net_comm_handle *net_claim(uint32_t link,int kind,uint64_t key){
  struct net_comm_handle *c=calloc(1,sizeof *c);
  if(!c){errno=ENOMEM;return NULL;}
  struct mesh_net_comm *comms=mesh_net_comms(net.m);
  for(uint32_t i=0;i<MESH_NET_COMMS;i++){
    uint32_t vacant=MESH_NET_FREE;
    if(!atomic_compare_exchange_strong_explicit(&comms[i].state,&vacant,MESH_NET_CLAIMED,memory_order_acq_rel,memory_order_relaxed))continue;
    struct mesh_net_comm *comm=comms+i;
    comm->generation++;comm->link=link;comm->listen=UINT32_MAX;comm->listen_generation=0;
    comm->peer=UINT32_MAX;comm->peer_generation=0;comm->owner=net.owner;comm->key=key;
    atomic_store_explicit(&comm->error,0,memory_order_relaxed);
    atomic_store_explicit(&comm->posted,0,memory_order_relaxed);atomic_store_explicit(&comm->bytes,0,memory_order_relaxed);
    atomic_store_explicit(&comm->completions,0,memory_order_relaxed);atomic_store_explicit(&comm->credit_waits,0,memory_order_relaxed);
    for(uint32_t r=0;r<MESH_NET_REQUESTS;r++)atomic_store_explicit(&comm->requests[r].state,MESH_NET_IDLE,memory_order_relaxed);
    *c=(struct net_comm_handle){.index=i,.generation=comm->generation,.link=link,.kind=(uint32_t)kind};
    atomic_store_explicit(&comm->state,(uint32_t)kind,memory_order_release);
    net_ring(link);
    return c;
  }
  free(c);errno=ENOSPC;return NULL;
}

int mesh_net_listen(void *ctx,int dev,void *handle,void **listenComm){
  (void)ctx;
  if(!net.m)return net_result(EBUSY);
  if(dev<0 || (uint32_t)dev>=net.m->links || !handle)return net_result(EINVAL);
  struct mesh_net_handle given;memcpy(&given,handle,sizeof given);
  uint64_t key=given.magic==MESH_NET_HANDLE_MAGIC?given.key:0;
  while(!key)key=((uint64_t)arc4random()<<32)|arc4random();
  struct net_comm_handle *c=net_claim((uint32_t)dev,MESH_NET_LISTEN,key);
  if(!c)return net_result(errno);
  memset(handle,0,MESH_NET_HANDLE_BYTES);
  struct mesh_net_handle made={MESH_NET_HANDLE_MAGIC,net.m->node,key};
  memcpy(handle,&made,sizeof made);
  *listenComm=c;
  return MESH_NET_SUCCESS;
}

/* Not blocking: the first call claims a connecting comm the bridge announces to its peer; later calls
   with the same handle return it once the peer's bridge has accepted it. */
int mesh_net_connect(void *ctx,int dev,void *handle,void **sendComm,void **sendDevComm){
  (void)sendDevComm;
  struct net_context *context=ctx;
  *sendComm=NULL;
  if(!net.m)return net_result(EBUSY);
  struct mesh_net_handle given;memcpy(&given,handle,sizeof given);
  if(dev<0 || (uint32_t)dev>=net.m->links || given.magic!=MESH_NET_HANDLE_MAGIC || !given.key || mesh_links(net.m)[dev].peer!=given.node)return net_result(EINVAL);
  struct net_pending *pending=NULL,*vacant=NULL;
  for(int i=0;i<NET_PENDING && !pending;i++){
    struct net_pending *p=context->pending+i;
    if(p->comm && p->dev==dev && p->key==given.key && p->node==given.node)pending=p;
    else if(!p->comm && !vacant)vacant=p;
  }
  if(!pending){
    if(!vacant)return net_result(ENOSPC);
    struct net_comm_handle *c=net_claim((uint32_t)dev,MESH_NET_CONNECTING,given.key);
    if(!c)return net_result(errno);
    *vacant=(struct net_pending){c,given.key,given.node,dev};
    return MESH_NET_SUCCESS;
  }
  struct mesh_net_comm *comm=net_slot(pending->comm);
  uint32_t state=atomic_load_explicit(&comm->state,memory_order_acquire);
  if(state==MESH_NET_SEND){
    pending->comm->kind=MESH_NET_SEND;*sendComm=pending->comm;pending->comm=NULL;
    return MESH_NET_SUCCESS;
  }
  if(state==MESH_NET_FAILED || net_bridge_gone()){
    int error=atomic_load_explicit(&comm->error,memory_order_relaxed);
    atomic_store_explicit(&comm->state,MESH_NET_CLOSING,memory_order_release);net_ring(pending->comm->link);
    free(pending->comm);pending->comm=NULL;
    return net_result(error?error:ECONNREFUSED);
  }
  return MESH_NET_SUCCESS;
}

int mesh_net_accept(void *listenComm,void **recvComm,void **recvDevComm){
  (void)recvDevComm;
  struct net_comm_handle *l=listenComm;
  *recvComm=NULL;
  struct mesh_net_comm *comms=mesh_net_comms(net.m);
  for(uint32_t i=0;i<MESH_NET_COMMS;i++){
    struct mesh_net_comm *comm=comms+i;
    if(atomic_load_explicit(&comm->state,memory_order_acquire)!=MESH_NET_ACCEPTABLE || comm->listen!=l->index ||
       comm->listen_generation!=l->generation || comm->link!=l->link)continue;
    uint32_t acceptable=MESH_NET_ACCEPTABLE;
    if(!atomic_compare_exchange_strong_explicit(&comm->state,&acceptable,MESH_NET_RECV,memory_order_acq_rel,memory_order_relaxed))continue;
    struct net_comm_handle *c=calloc(1,sizeof *c);
    if(!c){atomic_store_explicit(&comm->state,MESH_NET_CLOSING,memory_order_release);net_ring(l->link);return net_result(ENOMEM);}
    *c=(struct net_comm_handle){.index=i,.generation=comm->generation,.link=l->link,.kind=MESH_NET_RECV};
    net_ring(l->link);
    *recvComm=c;
    return MESH_NET_SUCCESS;
  }
  if(net_bridge_gone())return net_result(ECONNRESET);
  return MESH_NET_SUCCESS;
}

/* In place: memory in the region's registered window (mesh_net_mem_alloc's, or any section a client has
   there) is named by the device's registered regions it lies in; the bridge registers nothing new, as a
   receive lands only in memory registered before its queue pair was set up.  Other memory is refused. */
int mesh_net_reg_mr(void *comm,void *data,size_t size,int type,void **mhandle){
  (void)comm;
  if(!net.m)return net_result(EBUSY);
  if(type!=MESH_NET_PTR_HOST)return net_result(EINVAL);
  char *window=(char *)net.m+net.m->data_off;uint64_t wire=mesh_wire_bytes(net.m);
  if(size && ((char *)data<window || (char *)data>window+wire || size>(size_t)(window+wire-(char *)data)))return net_result(EFAULT);
  struct net_mhandle *h=calloc(1,sizeof *h);
  if(!h)return net_result(ENOMEM);
  *h=(struct net_mhandle){.mr=MESH_NET_WINDOW,.address=data,.size=size,.offset=size?(uint64_t)((char *)data-window):0};
  *mhandle=h;
  return MESH_NET_SUCCESS;
}
int mesh_net_reg_mr_dma_buf(void *comm,void *data,size_t size,int type,uint64_t offset,int fd,void **mhandle){
  (void)comm;(void)data;(void)size;(void)type;(void)offset;(void)fd;(void)mhandle;
  return net_result(ENOTSUP);
}
int mesh_net_dereg_mr(void *comm,void *mhandle){
  (void)comm;
  free(mhandle);
  return MESH_NET_SUCCESS;
}

/* A request in the comm's ring, or none (NULL: the ring is full; try again) as ncclNet allows. */
static int net_post(struct net_comm_handle *c,uint32_t op,void *data,size_t size,int tag,struct net_mhandle *h,uint64_t *word,void **request){
  *request=NULL;
  struct mesh_net_comm *comm=net_slot(c);
  char *window=(char *)net.m+net.m->data_off;
  if(word && ((char *)word<window || (char *)(word+1)>window+mesh_wire_bytes(net.m) || ((uintptr_t)word&7)))return net_result(EINVAL);
  uint32_t state=atomic_load_explicit(&comm->state,memory_order_acquire);
  if(state!=c->kind){
    int error=atomic_load_explicit(&comm->error,memory_order_relaxed);
    return net_result(error?error:ECONNRESET);
  }
  if(size && (!h || (char *)data<h->address || (char *)data>h->address+h->size || size>(size_t)(h->address+h->size-(char *)data)))return net_result(EINVAL);
  uint64_t sequence=c->sequence;
  struct mesh_net_request *r=comm->requests+sequence%MESH_NET_REQUESTS;
  if(atomic_load_explicit(&r->state,memory_order_acquire)!=MESH_NET_IDLE)return MESH_NET_SUCCESS;
  r->op=op;r->mr=h?h->mr:MESH_NET_WINDOW;r->mr_generation=h?h->generation:0;r->tag=tag;r->sequence=sequence;
  r->offset=size?h->offset+(uint64_t)((char *)data-h->address):0;r->size=size;
  r->completion=word?(uint64_t)((char *)word-window)+1:0;
  atomic_store_explicit(&r->transferred,0,memory_order_relaxed);atomic_store_explicit(&r->error,0,memory_order_relaxed);
  atomic_store_explicit(&r->state,MESH_NET_POSTED,memory_order_release);
  atomic_store_explicit(&comm->posted,sequence+1,memory_order_release);
  c->sequence++;
  net_ring(c->link);
  struct net_request_handle *handle=c->requests+sequence%MESH_NET_REQUESTS;
  *handle=(struct net_request_handle){c,sequence};
  *request=handle;
  return MESH_NET_SUCCESS;
}
int mesh_net_isend(void *sendComm,void *data,size_t size,int tag,void *mhandle,void *phandle,void **request){
  (void)phandle;
  return net_post(sendComm,MESH_NET_ISEND,data,size,tag,mhandle,NULL,request);
}
int mesh_net_isend_word(void *sendComm,void *data,size_t size,void *mhandle,uint64_t *word,void **request){
  return net_post(sendComm,MESH_NET_ISEND,data,size,0,mhandle,word,request);
}
int mesh_net_irecv_word(void *recvComm,void *data,size_t size,void *mhandle,uint64_t *word,void **request){
  return net_post(recvComm,MESH_NET_IRECV,data,size,0,mhandle,word,request);
}
int mesh_net_isend_held(void *sendComm,void *data,size_t size,void *mhandle,uint64_t *word,void **request){
  return net_post(sendComm,MESH_NET_HELD,data,size,0,mhandle,word,request);
}
int mesh_net_release(void *request,uint64_t *granted){
  struct net_request_handle *handle=request;
  struct mesh_net_request *r=net_slot(handle->comm)->requests+handle->sequence%MESH_NET_REQUESTS;
  if(r->sequence!=handle->sequence)return net_result(EINVAL);
  if(granted)*granted=atomic_load_explicit(&r->transferred,memory_order_acquire);
  uint32_t held=MESH_NET_HELD;
  if(atomic_compare_exchange_strong_explicit(&r->op,&held,MESH_NET_ISEND,memory_order_release,memory_order_relaxed))net_ring(handle->comm->link);
  return MESH_NET_SUCCESS;
}
int mesh_net_irecv(void *recvComm,int n,void **data,size_t *sizes,int *tags,void **mhandles,void **phandles,void **request){
  (void)phandles;
  if(n!=1){*request=NULL;return net_result(EINVAL);}
  return net_post(recvComm,MESH_NET_IRECV,data[0],sizes[0],tags?tags[0]:0,mhandles?mhandles[0]:NULL,NULL,request);
}
/* A fence: done once every earlier request of the comm is, after the bridge's full barrier. */
int mesh_net_iflush(void *recvComm,int n,void **data,int *sizes,void **mhandles,void **request){
  (void)data;(void)sizes;(void)mhandles;
  if(n>1){*request=NULL;return net_result(EINVAL);}
  return net_post(recvComm,MESH_NET_IFLUSH,NULL,0,0,NULL,NULL,request);
}
int mesh_net_test(void *request,int *done,int *sizes){
  struct net_request_handle *handle=request;
  struct mesh_net_request *r=net_slot(handle->comm)->requests+handle->sequence%MESH_NET_REQUESTS;
  *done=0;
  uint32_t state=atomic_load_explicit(&r->state,memory_order_acquire);
  if(r->sequence!=handle->sequence)return net_result(EINVAL);
  if(state==MESH_NET_DONE || state==MESH_NET_ERROR){
    int error=state==MESH_NET_ERROR?atomic_load_explicit(&r->error,memory_order_relaxed):0;
    if(sizes)*sizes=(int)atomic_load_explicit(&r->transferred,memory_order_relaxed);
    atomic_store_explicit(&r->state,MESH_NET_IDLE,memory_order_release);
    *done=1;
    return net_result(error);
  }
  if(net_bridge_gone())return net_result(ECONNRESET);
  return MESH_NET_SUCCESS;
}
static int net_close(void *handle){
  struct net_comm_handle *c=handle;
  if(!c)return MESH_NET_SUCCESS;
  struct mesh_net_comm *comm=net_slot(c);
  uint32_t state=atomic_load_explicit(&comm->state,memory_order_acquire);
  while(state!=MESH_NET_FREE && state!=MESH_NET_CLAIMED && state!=MESH_NET_CLOSING && comm->generation==c->generation &&
        !atomic_compare_exchange_weak_explicit(&comm->state,&state,MESH_NET_CLOSING,memory_order_acq_rel,memory_order_acquire));
  net_ring(c->link);
  free(c);
  return MESH_NET_SUCCESS;
}
int mesh_net_close_send(void *sendComm){return net_close(sendComm);}
int mesh_net_close_recv(void *recvComm){return net_close(recvComm);}
int mesh_net_close_listen(void *listenComm){return net_close(listenComm);}
int mesh_net_get_device_mr(void *comm,void *mhandle,void **dptr_mhandle){
  (void)comm;(void)mhandle;*dptr_mhandle=NULL;
  return net_result(ENOTSUP);
}
int mesh_net_irecv_consumed(void *recvComm,int n,void *request){(void)recvComm;(void)n;(void)request;return MESH_NET_SUCCESS;}
int mesh_net_make_vdevice(int *d,struct mesh_net_vdevice *props){
  if(!net.m || !props || props->ndevs!=1 || props->devs[0]<0 || (uint32_t)props->devs[0]>=net.m->links)return net_result(ENOTSUP);
  *d=props->devs[0];
  return MESH_NET_SUCCESS;
}
int mesh_net_set_net_attr(void *ctx,struct mesh_net_attr *netAttr){(void)ctx;(void)netAttr;return MESH_NET_SUCCESS;}
int mesh_net_finalize(void *ctx){
  struct net_context *context=ctx;
  for(int i=0;context && i<NET_PENDING;i++)if(context->pending[i].comm)net_close(context->pending[i].comm);
  free(context);
  pthread_mutex_lock(&net.lock);
  if(net.inits && !--net.inits && net.m){
    struct mesh_net_memory *memory=mesh_net_memory(net.m);
    for(uint32_t i=0;i<MESH_NET_MEMORY;i++){
      uint64_t owner=net.owner;uint32_t first=memory[i].first,pages=memory[i].pages;
      if(atomic_load_explicit(&memory[i].owner,memory_order_acquire)==owner &&
         atomic_compare_exchange_strong_explicit(&memory[i].owner,&owner,0,memory_order_acq_rel,memory_order_relaxed))
        mesh_bits_clear(mesh_arena_bits(net.m),first,pages);
    }
    if(net.client<MESH_NET_CLIENTS){
      uint64_t owner=net.owner;
      atomic_compare_exchange_strong_explicit(&mesh_net_clients(net.m)[net.client].owner,&owner,0,memory_order_acq_rel,memory_order_relaxed);
    }
    for(uint32_t i=0;i<net.m->links;i++)free(net.names[i]);
    free(net.names);net.names=NULL;
    munmap(net.m,net.length);net.m=NULL;
  }
  pthread_mutex_unlock(&net.lock);
  return MESH_NET_SUCCESS;
}

/* Pages of the registered window, whole blocks of them, recorded as this client's in the region so the
   bridge vacates them if it exits without freeing them. */
int mesh_net_mem_alloc(void **pointer,size_t size){
  if(!net.m)return net_result(EBUSY);
  if(!size)return net_result(EINVAL);
  struct hdr *m=net.m;
  uint64_t quantum=(uint64_t)m->block*m->pgsz,pages=(size+quantum-1)/quantum*m->block;
  if(pages>UINT32_MAX)return net_result(ENOMEM);
  struct mesh_net_memory *memory=mesh_net_memory(m),*entry=NULL;
  for(uint32_t i=0;i<MESH_NET_MEMORY && !entry;i++){
    uint64_t vacant=0;
    if(atomic_compare_exchange_strong_explicit(&memory[i].owner,&vacant,net.owner,memory_order_acq_rel,memory_order_relaxed))entry=memory+i;
  }
  if(!entry)return net_result(ENOSPC);
  uint32_t first=mesh_arena_claim(m,(uint32_t)pages,m->block,1);
  if(first==MESH_ABSENT){int error=errno;atomic_store_explicit(&entry->owner,0,memory_order_release);return net_result(error);}
  entry->first=first;entry->pages=(uint32_t)pages;
  *pointer=mesh_at(m,first);
  return MESH_NET_SUCCESS;
}
int mesh_net_mem_free(void *pointer){
  if(!net.m)return net_result(EBUSY);
  struct mesh_net_memory *memory=mesh_net_memory(net.m);
  for(uint32_t i=0;i<MESH_NET_MEMORY;i++){
    uint64_t owner=net.owner;
    if(atomic_load_explicit(&memory[i].owner,memory_order_acquire)!=owner || mesh_at(net.m,memory[i].first)!=(unsigned char *)pointer)continue;
    uint32_t first=memory[i].first,pages=memory[i].pages;
    if(atomic_compare_exchange_strong_explicit(&memory[i].owner,&owner,0,memory_order_acq_rel,memory_order_relaxed))
      mesh_bits_clear(mesh_arena_bits(net.m),first,pages);
    return MESH_NET_SUCCESS;
  }
  return net_result(EINVAL);
}
int mesh_net_window(void **base,size_t *bytes){
  if(!net.m)return net_result(EBUSY);
  *base=(char *)net.m+net.m->data_off;*bytes=(size_t)mesh_wire_bytes(net.m);
  return MESH_NET_SUCCESS;
}
void mesh_net_comm_counts(void *comm,uint64_t counts[4]){
  struct mesh_net_comm *slot=net_slot(comm);
  counts[0]=atomic_load_explicit(&slot->bytes,memory_order_relaxed);
  counts[1]=atomic_load_explicit(&slot->completions,memory_order_relaxed);
  counts[2]=atomic_load_explicit(&slot->credit_waits,memory_order_relaxed);
  counts[3]=atomic_load_explicit(&slot->posted,memory_order_relaxed);
}

int mesh_net_alive(void *comm){
  struct net_comm_handle *c=comm;
  return c && atomic_load_explicit(&net_slot(c)->state,memory_order_acquire)==c->kind;
}
uint64_t mesh_net_slot(void *comm){
  struct net_comm_handle *c=comm;
  return ((uint64_t)c->index<<32)|c->generation;
}
int mesh_net_vacated(uint64_t slot){
  struct mesh_net_comm *comm=mesh_net_comms(net.m)+(uint32_t)(slot>>32);
  return atomic_load_explicit(&comm->state,memory_order_acquire)==MESH_NET_FREE || comm->generation!=(uint32_t)slot;
}

const struct mesh_net_v12 mesh_net_plugin={"mesh",mesh_net_init,mesh_net_devices,mesh_net_get_properties,mesh_net_listen,
  mesh_net_connect,mesh_net_accept,mesh_net_reg_mr,mesh_net_reg_mr_dma_buf,mesh_net_dereg_mr,mesh_net_isend,mesh_net_irecv,
  mesh_net_iflush,mesh_net_test,mesh_net_close_send,mesh_net_close_recv,mesh_net_close_listen,mesh_net_get_device_mr,
  mesh_net_irecv_consumed,mesh_net_make_vdevice,mesh_net_finalize,mesh_net_set_net_attr};
