#include "mesh-verbs.h"
#include "mesh-call.h"
#include <dispatch/dispatch.h>
#include <SystemConfiguration/SystemConfiguration.h>
#include <sys/event.h>
#include <sys/ioctl.h>
#include <sys/kern_event.h>
#include <sys/socket.h>
#include <net/if.h>
#include <net/if_var.h>
#include <net/net_kev.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <stdarg.h>
#include <time.h>
#include <notify.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <IOKit/ps/IOPowerSources.h>
#include <IOKit/ps/IOPSKeys.h>

_Static_assert(sizeof(pthread_t)==8,"M17");

_Static_assert(sizeof(struct ibv_send_wr)==128,"M06");
_Static_assert(sizeof(struct ibv_sge)==16,"M07 M09");
_Static_assert(sizeof(struct ibv_recv_wr)==32,"M08");
_Static_assert(sizeof(struct ibv_wc)==48,"M11");
_Static_assert(offsetof(struct ibv_wc,status)==8 && offsetof(struct ibv_wc,opcode)==12 &&
  sizeof(((struct ibv_wc *)0)->status)==4 && sizeof(((struct ibv_wc *)0)->opcode)==4,"M11");
_Static_assert(sizeof(struct ibv_send_wr *)==8,"M15");
/* design/prepared-machine.md#M24 */
union mesh_network_event {
  struct kern_event_msg header;
  unsigned char bytes[KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data)];
};
_Static_assert(sizeof(union mesh_network_event)==48,"M24");
/* design/prepared-machine.md#M29 */
struct prepared_send {
  _Alignas(128) struct ibv_sge span;
  struct ibv_send_wr request;
};
_Static_assert(sizeof(struct prepared_send)==256 && _Alignof(struct prepared_send)==128 &&
  offsetof(struct prepared_send,span)==0 && offsetof(struct prepared_send,request)==16 &&
  offsetof(struct prepared_send,request.send_flags)+sizeof(unsigned int)<=64,"M29");
/* design/prepared-machine.md#M08 */
/* One chunk of a receive transfer in one invocation: its native request and SGE into the chunk's
   destination range, the completion word it stores (`input`; its own `sink` where no reader waits on its landing),
   its queue pair, the frames it holds posted, its invocation and ring. */
struct prepared_receive {
  _Alignas(128) struct ibv_recv_wr request;
  struct ibv_sge span;
  _Atomic uint64_t *input;
  struct ibv_qp *pair;
  uint64_t argument;
  uint32_t frames,invocation,queue,cycle,behind;
  _Atomic uint64_t sink;
};
_Static_assert(sizeof(struct prepared_receive)==128 && _Alignof(struct prepared_receive)==128 &&
  offsetof(struct prepared_receive,span)==32 && offsetof(struct prepared_receive,input)==48,"M08 M09");
/* design/prepared-machine.md#M08 */
/* A queue pair's receive ring: its records in invocation, binding, chunk order; `posted` the next to
   post, `outstanding` the frames posted and not completed, within the queue's `capacity`; a record is
   posted once the record `behind` it in the ring's running order has landed: its transfer's and chunk's
   record `span` invocations before (the ring slots' margin), or in a cyclic program its own previous
   posting where none is; a record with none is posted at once.  A transfer's slots are its own, so a ring
   whose transfers begin late, or leave invocations between them, posts each transfer's first records. */
struct receive_ring {
  struct prepared_receive *first;
  size_t count,posted;
  int64_t completed;
  uint32_t outstanding,capacity,span;
  /* the census: records landed, their bytes */
  uint64_t landed,bytes;
};
/* A queue pair's SEND gate, the send thread's alone: the frames posted and retired (each request
   carries its queue and the frames posted through it, which its completion returns), the queue's
   capacity, and the census's requests and completions. */
#define MESH_GATE_MASK ((UINT64_C(1)<<48)-1)
struct send_gate {
  _Alignas(64) uint64_t posted,retired,requests,completions;
  uint32_t capacity;
};
/* A SEND stream's progress: its current cell, the next request of that cell's chain and the requests
   left. */
struct send_stream { struct mesh_send *cell,*first; struct ibv_send_wr *next; uint32_t remaining; uint64_t cycle; };

#define MESH_PAIRS 16
/* design/bridge-and-ipc.md#protocol: a message, a flow's keyed by its execution's key and its program's two end nodes
   (`from` the sender's end, `to` the far one), `failed` the node where a failure happened, and `bytes` of what follows
   it: a READY's program, a RECORD's record */
enum { HELLO=1, READY, PAIRS, POSTED, END, FAIL, FAREWELL, RECORD };
struct message { uint32_t magic,kind,node,count; uint64_t key,boot; int64_t code; uint32_t block,port,from,to,failed,bytes;
  uint16_t lid,padding; uint8_t gid[16]; uint32_t pair[MESH_PAIRS][2]; };
#define XMAGIC (0x4d595048u+MESH_VERSION)
/* A READY's program: `hops` nodes of its route from the receiver to the far end, then the transfers the sender sends
   toward the far end, lengths[k] of them on queue queue + k x step, and its lists' slots, invocations, ring depth and
   whether it is cyclic; the message's count is its queue pairs */
struct ready { uint32_t hops,slots,invocations,depth,cyclic,queue,step,lengths[MESH_QUEUES]; };
/* design/bridge-and-ipc.md#flows: a program on one port, over its link toward the program's far end (`near` its other
   end): its lists each way, a slot's destination (its client's, or on a relay a lease of the bridge's own; a flow held
   for its client, `hold`, has none until the client's program takes it), its queues queue + k x step for k < queues,
   `slots` queue pairs each, and on a relay its partner, the route's other flow.  Its verbs work runs on its port's
   queue, one job at a time (`job`, `done`); the event loop alone changes the rest, and a side whose lists arrive while
   a job runs (`pending`) between jobs */
enum { READIED, PAIRING, POSTING, RUNNING, FAILED };
enum { HEARD_READY=1, HEARD_PAIRS=2, HEARD_POSTED=4 };
enum { JOB_PAIR=1, JOB_RECEIVE, JOB_RUN, JOB_CHANGE };
struct lease { struct mesh_ctx context; uint32_t users; };
struct side { uint64_t owner; uint32_t destination; struct lease *lease; };
struct port;
struct flow {
  struct flow *next,*partner;
  pthread_t workers[2];
  uint32_t worker_count,publication_count,stream_count,near,far,queue,step,queues,heard,configured,destination,invocations;
  _Atomic int progressing,done,stopped;
  _Atomic int64_t failed;
  _Atomic uint32_t domain;
  int state,job,result,stopping,hold,qps,cyclic;
  uint64_t key,owner,created,deadline,retry;
  struct side side[2],pending[2];
  uint32_t pairs[MESH_PAIRS][2];
  struct hdr *M; struct port *port; struct mesh_verbs provider;
  struct prepared_receive *receive;
  size_t receive_count;
  struct receive_ring *rings;
  struct send_gate *gates;
  struct send_stream *streams;
  struct prepared_send *requests;
  size_t request_count;
  struct mesh_send *publications;
  struct ibv_wc completion;
  struct mesh_cancellation *cancel;
};
/* design/bridge-and-ipc.md#ports: a port, its link, its control socket's queues each way, and its verbs work's queue */
struct port {
  struct mesh_device device;
  uint32_t index,interface,peer,peer_port,greeted,parted,connecting,writing,retiring,open,flows;
  int64_t farewell;
  char name[IFNAMSIZ];
  int control,job,result;
  _Atomic int done;
  uint64_t next,bandwidth;
  struct message heard;
  uint16_t lid;
  uint8_t gid[16];
  unsigned char *in,*out;
  size_t held,in_room,out_length,out_sent,out_room;
  dispatch_queue_t work;
};
static struct hdr *M;
static struct mesh_wire wire;
static struct port ports[MESH_PORTS];
static uint32_t port_count;
static struct flow *flows;
static int events=-1,listener=-1,hello=-1,network=-1,unreleased,finding;
static uint64_t now,sequence;
static char host[MESH_NAME_BYTES];
static struct mesh_ctx bridge={.fd=-1};
static uint32_t map_page,map_pages,map_room;
static dispatch_queue_t finder;
static _Atomic int found;
static char devices[MESH_PORTS][32];
static uint32_t device_count;
static const struct in6_addr everyone={{{0xff,0x02,0,0,0,0,0,0,0,0,0,0,0,0,0,1}}};

static uint64_t clock_ns(void){ return clock_gettime_nsec_np(CLOCK_MONOTONIC); }


/* design/recovery.md#diagnosis: why this bridge stops, the code its FAREWELL carries (a signal's number; its binary
   changed) */
#define MESH_STOP_BINARY INT64_C(-1)
static int64_t stop_reason;
static void farewell_text(int64_t code,char *out,size_t room){
  if(code==SIGTERM)snprintf(out,room,"SIGTERM: its agent was unloaded or restarted");
  else if(code>0)snprintf(out,room,"signal %lld",(long long)code);
  else if(code==MESH_STOP_BINARY)snprintf(out,room,"its binary changed");
  else snprintf(out,room,"no reason given");
}

/* design/recovery.md#diagnosis: this node's health (mesh-plan.h struct mesh_health), sampled on the event loop from
   cheap reads only: sysctl, the host's VM statistics and the notify states of thermal pressure and of the power
   sources, whose description is read only when their notify state moved */
static struct mesh_health health;
static int power_token=-1,thermal_token=-1;
static uint64_t health_sent;
static mach_port_t host_port;
#define MESH_HEALTH_NS UINT64_C(10000000000)
static void sysctl_text(const char *name,char *out,size_t room){
  size_t size=room;
  if(sysctlbyname(name,out,&size,NULL,0))out[0]=0;
  out[room-1]=0;
}
static void power_read(struct mesh_health *h){
  h->power=MESH_POWER_UNKNOWN;h->charge=-1;h->minutes=-1;
  CFTypeRef info=IOPSCopyPowerSourcesInfo();
  if(!info)return;
  CFStringRef providing=IOPSGetProvidingPowerSourceType(info);
  if(providing && CFEqual(providing,CFSTR(kIOPMACPowerKey)))h->power=MESH_POWER_AC;
  else if(providing && CFEqual(providing,CFSTR(kIOPMBatteryPowerKey)))h->power=MESH_POWER_BATTERY;
  else if(providing && CFEqual(providing,CFSTR(kIOPMUPSPowerKey)))h->power=MESH_POWER_UPS;
  CFArrayRef list=IOPSCopyPowerSourcesList(info);
  for(CFIndex i=0;list && i<CFArrayGetCount(list);i++){
    CFDictionaryRef source=IOPSGetPowerSourceDescription(info,CFArrayGetValueAtIndex(list,i));
    if(!source)continue;
    int current=-1,most=-1,empty=-1;CFNumberRef value;
    if((value=CFDictionaryGetValue(source,CFSTR(kIOPSCurrentCapacityKey))))CFNumberGetValue(value,kCFNumberIntType,&current);
    if((value=CFDictionaryGetValue(source,CFSTR(kIOPSMaxCapacityKey))))CFNumberGetValue(value,kCFNumberIntType,&most);
    if((value=CFDictionaryGetValue(source,CFSTR(kIOPSTimeToEmptyKey))))CFNumberGetValue(value,kCFNumberIntType,&empty);
    if(current>=0 && most>0){h->charge=current*100/most;h->minutes=empty;break;}
  }
  if(list)CFRelease(list);
  CFRelease(info);
}
static void health_sample(struct mesh_health *h){
  h->sampled=clock_gettime_nsec_np(CLOCK_REALTIME);
  vm_statistics64_data_t vm;mach_msg_type_number_t count=HOST_VM_INFO64_COUNT;
  if(host_statistics64(host_port,HOST_VM_INFO64,(host_info64_t)&vm,&count)==KERN_SUCCESS){
    h->free=(uint64_t)vm.free_count*vm_kernel_page_size;h->wired=(uint64_t)vm.wire_count*vm_kernel_page_size;
  }
  int level=0;size_t size=sizeof level;
  if(!sysctlbyname("kern.memorystatus_vm_pressure_level",&level,&size,NULL,0))h->pressure=(uint32_t)level;
  uint64_t state=0;
  if(thermal_token>=0 && notify_get_state(thermal_token,&state)==NOTIFY_STATUS_OK)h->thermal=(uint32_t)state;
  int changed=0;
  if(power_token>=0 && notify_check(power_token,&changed)==NOTIFY_STATUS_OK && changed)power_read(h);
}
/* the boot's facts, read once: when the node booted, its boot session, and why it last restarted (the head of
   kern.shutdownreason, the power controller's records newest first) */
static void health_init(void){
  host_port=mach_host_self();
  struct timeval booted;size_t size=sizeof booted;
  if(!sysctlbyname("kern.boottime",&booted,&size,NULL,0))health.booted=(uint64_t)booted.tv_sec*1000000000u+(uint64_t)booted.tv_usec*1000u;
  sysctl_text("kern.bootsessionuuid",health.session,sizeof health.session);
  char reason[512];sysctl_text("kern.shutdownreason",reason,sizeof reason);
  const char *newest=reason;while(*newest==' ')newest++;
  snprintf(health.shutdown,sizeof health.shutdown,"%s",newest);
  health.reset=!*newest?MESH_RESET_UNKNOWN:!strncmp(newest,"por",3)?MESH_RESET_SUPPLY:!strncmp(newest,"wdog",4)?MESH_RESET_WATCHDOG:MESH_RESET_OTHER;
  notify_register_check(kIOPSNotifyAnyPowerSource,&power_token);
  notify_register_check("com.apple.system.thermalpressurelevel",&thermal_token);
  power_read(&health);
  health_sample(&health);
}
static void health_text(const struct mesh_health *h,char *out,size_t room){
  static const char *resets[]={"unknown","supply lost","watchdog","other"};
  static const char *thermals[]={"nominal","moderate","heavy","trapping","sleeping"};
  char power[64],thermal[16],when[32];
  if(h->power==MESH_POWER_AC && h->charge>=0)snprintf(power,sizeof power,"AC, battery %d%%",h->charge);
  else if(h->power==MESH_POWER_AC)snprintf(power,sizeof power,"AC");
  else if(h->power==MESH_POWER_BATTERY || h->power==MESH_POWER_UPS)
    snprintf(power,sizeof power,"%s %d%%, %d min to empty",h->power==MESH_POWER_UPS?"UPS battery":"battery",h->charge,h->minutes);
  else snprintf(power,sizeof power,"unknown");
  if(h->thermal<5)snprintf(thermal,sizeof thermal,"%s",thermals[h->thermal]);else snprintf(thermal,sizeof thermal,"%u",h->thermal);
  time_t booted=(time_t)(h->booted/1000000000u);struct tm utc;gmtime_r(&booted,&utc);strftime(when,sizeof when,"%Y-%m-%dT%H:%M:%SZ",&utc);
  snprintf(out,room,"power %s; memory free %.2f GiB, wired %.2f GiB, pressure %s; thermal %s; booted %s (session %.36s), "
           "last restart %s (%.79s)",power,(double)h->free/1073741824.0,(double)h->wired/1073741824.0,
           h->pressure==4?"critical":h->pressure==2?"warning":h->pressure==1?"normal":"unread",thermal,when,h->session,
           h->reset<4?resets[h->reset]:"unknown",h->shutdown);
}

/* design/recovery.md#diagnosis: the nodes this bridge has heard of, when it last heard each one's record (this node's
   clock), when it lost each, and the boot session it last knew, so that a node's return names a new boot */
struct known { uint32_t node; uint64_t heard,gone; char session[40]; };
static struct known known[256];
static uint32_t known_count;
static struct known *known_at(uint32_t node){
  for(uint32_t i=0;i<known_count;i++)if(known[i].node==node)return known+i;
  if(known_count==sizeof known/sizeof *known)return NULL;
  known[known_count]=(struct known){.node=node};
  return known+known_count++;
}
static void wake(void){
  struct kevent64_s event;EV_SET64(&event,0,EVFILT_USER,0,NOTE_TRIGGER,0,0,0,0);
  kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
}
/* design/algorithm-sources.md#meshresult: a data thread's failure, taken by the event loop (flow_fail) */
static void flow_lost(struct flow *flow,int64_t code,uint32_t domain){
  int64_t none=0;
  if(atomic_compare_exchange_strong(&flow->failed,&none,code?code:EIO))atomic_store(&flow->domain,domain);
  atomic_store_explicit(&flow->progressing,0,memory_order_release);
  wake();
}
static int owned(uint64_t owner){ return owner && atomic_load(&mesh_client_at(M,owner)->owner)==owner; }
static struct mesh_tx *side_tx(struct flow *f,int d){ return &mesh_client_at(M,f->side[d].owner)->tx[f->side[d].destination]; }
static uint32_t side_channel(struct flow *f,int d,uint32_t k){ return f->side[d].destination*MESH_QUEUES+f->queue+k*f->step; }
static uint32_t side_length(struct flow *f,int d,uint32_t k){ return f->side[d].owner?atomic_load(mesh_order_length(M,f->side[d].owner,side_channel(f,d,k),d)):0; }
static struct mesh_transfer *side_transfers(struct flow *f,int d,uint32_t k){ return mesh_transfers(M,f->side[d].owner,side_channel(f,d,k),d); }
/* design/bridge-and-ipc.md#clients: a status told, and its client's start woken */
static void status_prepared(struct mesh_port_info *status,uint64_t owner){
  atomic_store_explicit(&status->prepared,owner,memory_order_release);
  os_sync_wake_by_address_all(&status->prepared,sizeof status->prepared,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
static void status_failed(struct mesh_port_info *status,uint64_t owner,int64_t code,uint32_t domain,uint32_t where){
  atomic_store_explicit(&status->code,code,memory_order_relaxed);atomic_store_explicit(&status->domain,domain,memory_order_relaxed);
  atomic_store_explicit(&status->failed,where,memory_order_relaxed);atomic_store_explicit(&status->phase,MESH_STOPPED,memory_order_relaxed);
  status_prepared(status,owner);
}
/* the program whose status and cancellation words a flow tells */
static void flow_program(struct flow *f,uint64_t owner,uint32_t destination){
  uint64_t cancel=mesh_client_at(M,owner)->tx[destination].cancel;
  f->owner=owner;f->destination=destination;f->cancel=cancel?(void *)((char *)M+cancel):NULL;
}

/* The ring slot a transfer's invocation t addresses: (t - begin) mod its depth, at its slot pages. */
static uint32_t transfer_row(struct hdr *m,uint32_t fallback,const struct mesh_transfer *transfer,uint32_t row,uint32_t invocation,uint32_t chunk){
  uint32_t depth=transfer->depth?transfer->depth:fallback;
  if(!depth)depth=1;
  return row+(uint32_t)((size_t)((invocation-transfer->begin)%depth)*transfer->invocation_pages/m->block)+chunk;
}

/* design/prepared-machine.md#M09: a chunk's span in the window its port's device registered when it came up; 0, or
   EINVAL for a page outside the window */
static int chunk_span(struct flow *link,uint32_t address,uint64_t remaining,struct ibv_sge *span){
  struct hdr *m=link->M;
  uint64_t offset=atomic_load_explicit(&mesh_page(m)[address].address,memory_order_relaxed)-m->data_off;
  if(offset>=wire.length)return EINVAL;
  *span=wire_span(link->provider.device,(uint64_t)m->block*m->pgsz,offset,remaining);
  return 0;
}

/* design/prepared-machine.md#M07, #M08: the word a landed chunk of invocation u stores: its row's completion word
   (mesh_host_inputs, the Metal transport); or, on a row the receiving program (`owner`) also publishes (a relay's rows),
   that publication's SEND cell once the record's last chunk landed, so the landing releases it to the next hop (the
   index hand-off); else the record's own (`sink`), which nothing reads */
static _Atomic uint64_t *receive_input(struct hdr *m,uint64_t owner,uint32_t row,uint32_t chunk,uint32_t chunks,uint32_t u,_Atomic uint64_t *sink){
  const struct mesh_delivery *delivery=mesh_delivery_at(m,row+chunk);
  uint32_t sends;
  const struct mesh_target *target=mesh_row_targets(m,owner,row,&sends);
  if(delivery->device_input)return (void *)((char *)m+delivery->device_input+(size_t)delivery->device_stride*u);
  if(chunk+1==chunks && sends)return (void *)((char *)m+target->cells+sizeof(struct mesh_send)*(size_t)u);
  return sink;
}

/* design/prepared-machine.md#M08 */
/* Posts the ring's next records while its queue has room and the record behind each has landed (its
   transfer's `span` invocations before: the ring's records land in the order posted, so `landed` counts
   them).  Called once at configuration and at each receive completion: the refill is fired by the event. */
/* design/prepared-machine.md#M30 */
/* A cyclic program's ring wraps: its posted count runs on, record posted % count in cycle posted / count,
   at running invocation invocation + cycle * N; the span is held under N and a record with no record of
   its transfer behind it waits on its own previous posting, so a record is posted again only after its
   last posting completed (completions arrive in order). */
static int ring_advance(struct flow *link,struct receive_ring *ring){
  int (*post)(struct ibv_qp *,struct ibv_recv_wr *,struct ibv_recv_wr **)=link->provider.queues[0].receive;
  while(ring->count && (link->cyclic || ring->posted<ring->count)){
    struct prepared_receive *record=ring->first+ring->posted%ring->count;
    uint64_t cycle=ring->posted/ring->count;
    if((record->behind && ring->posted>=record->behind && ring->landed<=ring->posted-record->behind) ||
       ring->outstanding+record->frames>ring->capacity)break;
    record->cycle=(uint32_t)cycle;
    struct ibv_recv_wr *bad;
    int error=post(record->pair,&record->request,&bad);
    if(error)return error<0?-error:error;
    ring->outstanding+=record->frames;ring->posted++;
  }
  return 0;
}

/* design/algorithm-sources.md#programcopy */
/* design/prepared-machine.md#M04, #M06, #M29 */
/* The SEND side: every SEND cell's chain of chunk requests over its invocation's ring slot and its stream successor, a
   stream a queue pair (the cells' layout is mesh_transfers_prepare's), and each queue pair's gate.  A stream's cells run
   invocation by invocation, the publications of each in binding order, as the peer's ring posts its receives: TN3205's
   credit flow control holds a SEND until its matching receive is posted (metal-microbench docs/kernels.md Transport
   (git 7300d1e^)).  0, or the error. */
static int sends_configure(struct flow *link){
  struct hdr *m=link->M;struct mesh_client *client=mesh_client_at(m,link->side[MESH_SEND].owner);
  struct mesh_tx *tx=side_tx(link,MESH_SEND);
  uint32_t invocations=tx->invocations?tx->invocations:1,fallback=atomic_load_explicit(&client->depth,memory_order_acquire);
  int cyclic=atomic_load_explicit(&client->cyclic,memory_order_acquire)!=0,qps=link->qps;
  size_t requests=0;
  for(uint32_t k=0;k<link->queues;k++){
    struct mesh_transfer *out=side_transfers(link,MESH_SEND,k);
    for(uint32_t i=0;i<side_length(link,MESH_SEND,k);i++)
      requests+=(size_t)(mesh_row_chunks(m,out[i].local_row,out[i].bytes)-1)*out[i].count*mesh_transfer_active(out+i,invocations);
  }
  link->publications=(void *)((char *)m+tx->cells);
  link->publication_count=tx->count*tx->slots+tx->once;
  /* design/prepared-machine.md#M29 */
  link->requests=aligned_alloc(128,(requests+1)*sizeof *link->requests);
  link->streams=calloc((size_t)qps,sizeof *link->streams);
  link->gates=aligned_alloc(64,(size_t)qps*sizeof *link->gates);
  struct mesh_send **last=calloc((size_t)qps,sizeof *last),**terminal=calloc((size_t)qps,sizeof *terminal);
  if(!link->requests||!link->streams||!link->gates||!last||!terminal){free(last);free(terminal);return ENOMEM;}
  link->request_count=requests;
  /* a request is one frame at least, so a queue pair's frames also bound the completions it can have pending in the
     SEND completion queue its pairs share */
  uint32_t sends=(uint32_t)link->provider.sent->cqe/(uint32_t)qps;
  for(int q=0;q<qps;q++)link->gates[q]=(struct send_gate){.capacity=MIN(link->provider.queues[q].send_capacity,sends)};
  size_t next=0;int error=0;
  for(uint32_t k=0;k<link->queues && !error;k++){
    uint32_t count=side_length(link,MESH_SEND,k);
    struct mesh_transfer *out=side_transfers(link,MESH_SEND,k);
    /* each cell's chain over its invocation's ring slot */
    for(uint32_t i=0;i<count && !error;i++){
      uint32_t active=mesh_transfer_active(out+i,invocations),column=active+1;
      for(uint32_t slot=0;slot<out[i].count && !error;slot++){
        uint32_t row=out[i].local_row+slot*out[i].stride,chunks=mesh_row_chunks(m,row,out[i].bytes),stream=k*tx->slots+slot;
        struct mesh_send *cells=link->publications+out[i].first+(size_t)slot*column;
        for(uint32_t u=0;u<active && !error;u++){
          struct mesh_send *cell=cells+u;
          cell->pair=(uintptr_t)link->provider.queues[stream].pair;cell->queue=stream;cell->chunks=chunks;
          uint64_t remaining=out[i].bytes;
          for(uint32_t c=0;c<chunks && !error;c++){
            struct ibv_sge span;
            if((error=chunk_span(link,transfer_row(m,fallback,out+i,row,out[i].begin+u,c),remaining,&span)))break;
            remaining-=span.length;
            struct ibv_sge *entry=c?&link->requests[next+c-1].span:&cell->span;
            struct ibv_send_wr *request=c?&link->requests[next+c-1].request:&cell->request;
            *entry=span;
            /* the first request's wr_id stays the producer's nonzero argument (mesh_transfers_prepare),
               which a host producer reads at publication (mesh_host_publish) */
            *request=(struct ibv_send_wr){.wr_id=c?0:1,.next=c+1==chunks?NULL:&link->requests[next+c].request,
              .sg_list=entry,.num_sge=1,.opcode=IBV_WR_SEND};
          }
          next+=chunks-1;
        }
        terminal[stream]=cells+active;
      }
    }
    /* each stream's cells in invocation order, its publications in binding order within one */
    for(uint32_t t=0;t<invocations && !error;t++)for(uint32_t i=0;i<count;i++){
      uint32_t active=mesh_transfer_active(out+i,invocations);
      if(t<out[i].begin || t-out[i].begin>=active)continue;
      for(uint32_t slot=0;slot<out[i].count;slot++){
        uint32_t stream=k*tx->slots+slot;
        struct mesh_send *cell=link->publications+out[i].first+(size_t)slot*(active+1)+(t-out[i].begin);
        if(last[stream])last[stream]->successor=(uintptr_t)cell;
        else link->streams[stream].cell=cell;
        last[stream]=cell;
      }
    }
  }
  /* design/prepared-machine.md#M30: a cyclic program's last cell is followed by its first */
  for(int q=0;q<qps && !error;q++)if(last[q])last[q]->successor=cyclic?(uintptr_t)link->streams[q].cell:(uintptr_t)terminal[q];
  for(int q=0;q<qps;q++){link->streams[q].first=link->streams[q].cell;link->streams[q].cycle=0;}
  link->stream_count=0;
  for(int q=0;q<qps;q++)if(link->streams[q].cell)link->streams[link->stream_count++]=link->streams[q];
  free(last);free(terminal);
  return error;
}

/* design/prepared-machine.md#M08 */
/* The receive side: each queue pair's ring, its records invocation by invocation, the transfers of each in binding
   order, a transfer's chunks in order, each storing its word (receive_input), and each ring's first records posted.
   0, or the error. */
static int receives_configure(struct flow *link){
  struct hdr *m=link->M;struct mesh_client *client=mesh_client_at(m,link->side[MESH_RECEIVE].owner);
  struct mesh_tx *tx=side_tx(link,MESH_RECEIVE);
  uint32_t invocations=tx->invocations?tx->invocations:1,fallback=atomic_load_explicit(&client->depth,memory_order_acquire),depth=fallback;
  if(!depth||depth>invocations)depth=invocations;
  link->cyclic=atomic_load_explicit(&client->cyclic,memory_order_acquire)!=0;link->invocations=invocations;
  int qps=link->qps;
  size_t records=0;
  for(uint32_t k=0;k<link->queues;k++){
    struct mesh_transfer *in=side_transfers(link,MESH_RECEIVE,k);
    for(uint32_t i=0;i<side_length(link,MESH_RECEIVE,k);i++)records+=mesh_transfer_frames(m,in+i,invocations);
  }
  link->receive=aligned_alloc(128,(records+1)*sizeof *link->receive);
  link->rings=calloc((size_t)qps,sizeof *link->rings);
  if(!link->receive||!link->rings)return ENOMEM;
  link->receive_count=records;
  /* a record is one frame at least, so a queue pair's frames also bound the completions it can have pending in the
     receive completion queue its pairs share */
  uint32_t receives=(uint32_t)link->provider.completion->cqe/(uint32_t)qps;
  for(int q=0;q<qps;q++)link->rings[q]=(struct receive_ring){.completed=-1,.capacity=MIN(link->provider.queues[q].receive_capacity,receives),.span=UINT32_MAX};
  size_t at=0;int error=0;
  for(int ring=0;ring<qps && !error;ring++){
    uint32_t k=(uint32_t)ring/tx->slots,slot=(uint32_t)ring%tx->slots,count=side_length(link,MESH_RECEIVE,k);
    struct mesh_transfer *in=side_transfers(link,MESH_RECEIVE,k);
    struct receive_ring *r=link->rings+ring;
    r->first=link->receive+at;
    size_t *base=calloc((size_t)count+1,sizeof *base),*firsts=NULL;
    for(uint32_t i=0;base&&i<count;i++)base[i+1]=base[i]+(slot<in[i].count?mesh_transfer_active(in+i,invocations):0);
    if(base)firsts=malloc((base[count]+1)*sizeof *firsts);
    if(!base||!firsts){free(base);free(firsts);return ENOMEM;}
    for(uint32_t i=0;i<count;i++){
      if(slot>=in[i].count)continue;
      uint32_t active=mesh_transfer_active(in+i,invocations),own=in[i].depth?in[i].depth:depth;
      /* receive u+span lands in slot (u+span) mod depth while the local reader may still be at u:
         the ring's span is held a slot short of the ring; a transfer whose every invocation has its
         own slot bounds nothing */
      if(own<active)r->span=MIN(r->span,own>1?own-1:1);
    }
    for(uint32_t t=0;t<invocations && !error;t++)for(uint32_t i=0;i<count && !error;i++){
      uint32_t active=mesh_transfer_active(in+i,invocations);
      if(slot>=in[i].count || t<in[i].begin || t-in[i].begin>=active)continue;
      uint32_t u=t-in[i].begin,row=in[i].local_row+slot*in[i].stride,chunks=mesh_row_chunks(m,row,in[i].bytes);
      uint64_t remaining=in[i].bytes;
      for(uint32_t c=0;c<chunks;c++){
        struct ibv_sge span;
        if((error=chunk_span(link,transfer_row(m,fallback,in+i,row,t,c),remaining,&span)))break;
        remaining-=span.length;
        struct prepared_receive *record=link->receive+at++;
        if(!c)firsts[base[i]+u]=(size_t)(record-r->first);
        *record=(struct prepared_receive){.request={.wr_id=(uintptr_t)record,.sg_list=&record->span,.num_sge=1},
          .span=span,.input=receive_input(m,link->side[MESH_RECEIVE].owner,row,c,chunks,u,&record->sink),.pair=link->provider.queues[ring].pair,.argument=1,
          .frames=(span.length+4095)/4096,.invocation=t,.queue=(uint32_t)ring};
      }
    }
    r->count=(size_t)(link->receive+at-r->first);
    /* design/prepared-machine.md#M30 */
    if(link->cyclic)r->span=MIN(r->span,invocations>1?invocations-1:1);
    for(uint32_t i=0;i<count && !error;i++){
      uint32_t active=(uint32_t)(base[i+1]-base[i]);
      if(!active)continue;
      uint32_t chunks=mesh_row_chunks(m,in[i].local_row+slot*in[i].stride,in[i].bytes);
      for(uint32_t u=0;u<active;u++){
        size_t here=firsts[base[i]+u],behind=0;
        if(u>=r->span)behind=here-firsts[base[i]+u-r->span];
        else if(link->cyclic&&u+active>=r->span)behind=here+r->count-firsts[base[i]+u+active-r->span];
        else if(link->cyclic)behind=r->count;
        for(uint32_t c=0;c<chunks;c++)r->first[here+c].behind=(uint32_t)behind;
      }
    }
    free(base);free(firsts);
    if(!error)error=ring_advance(link,r);
  }
  if(at)link->receive[at]=link->receive[at-1];
  return error;
}

/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M06 */
/* design/prepared-machine.md#M15 */
/* design/algorithm-sources.md#independent-native-queues */
/* The link's SEND completions: each retires its queue's frames through its request (the device
   completes every SEND, signaled or not).  0, or an error completion's status. */
static __attribute__((always_inline)) inline int send_retire(struct flow *link){
  struct ibv_wc done[16];
  int count=link->provider.queues[0].poll(link->provider.sent,16,done);
  if(count<0)return EIO;
  for(int i=0;i<count;i++){
    if(done[i].status)return (int)done[i].status;
    struct send_gate *gate=link->gates+(done[i].wr_id>>48);
    gate->retired=done[i].wr_id&MESH_GATE_MASK;gate->completions++;
  }
  return 0;
}

/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#independent-native-queues */
/* Each stream's cells in order: a cell released by its producer posts its chain's requests while its
   queue pair holds their frames (the gate: frames posted less those its SEND completions retired;
   ibv_post_send accepts past the queue's frames and corrupts later, so the gate is the caller's).
   The thread takes its completions when a stream waits for its producer or its gate. */
static void *link_send_progress(void *argument){
  struct flow *link=argument;
  pthread_setname_np("mesh.rdma.send");
  struct send_stream *streams=link->streams;
  uint32_t count=link->stream_count;
  struct ibv_send_wr *bad;
  int (*post)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **)=link->provider.queues[0].send;
  for(uint32_t s=0;;s=s+1==count?0:s+1){
    struct send_stream *stream=streams+s;
    if(!stream->remaining){
      struct mesh_send *cell=stream->cell;
      if(atomic_load_explicit(&cell->ready,memory_order_acquire)<=stream->cycle){
        if(!atomic_load_explicit(&link->progressing,memory_order_acquire))return NULL;
        int error=send_retire(link);
        if(error){flow_lost(link,error,2);return NULL;}
        continue;
      }
      stream->next=&cell->request;stream->remaining=cell->chunks;
    }
    struct mesh_send *cell=stream->cell;
    struct send_gate *gate=link->gates+cell->queue;
    while(stream->remaining){
      struct ibv_send_wr *request=stream->next;
      uint32_t frames=(request->sg_list->length+4095)/4096;
      if(((gate->posted-gate->retired)&MESH_GATE_MASK)+frames>gate->capacity){
        if(!atomic_load_explicit(&link->progressing,memory_order_acquire))return NULL;
        int error=send_retire(link);
        if(error){flow_lost(link,error,2);return NULL;}
        break;
      }
      gate->posted+=frames;
      /* posted as a copy: the cell's first request keeps the producer's argument in its wr_id (mesh_transfers_prepare),
         which a host producer reads at every publication (mesh_host_publish), and the copy's names the gate */
      struct ibv_send_wr posting=*request;
      posting.next=NULL;posting.send_flags=IBV_SEND_SIGNALED;
      posting.wr_id=((uint64_t)cell->queue<<48)|(gate->posted&MESH_GATE_MASK);
      int error=post(link->provider.queues[cell->queue].pair,&posting,&bad);
      if(error){flow_lost(link,error<0?-error:error,1);return NULL;}
      stream->next=request->next;stream->remaining--;gate->requests++;
    }
    if(!stream->remaining){
      stream->cell=(struct mesh_send *)cell->successor;
      if(stream->cell==stream->first)stream->cycle++;
    }
  }
}

/* design/prepared-machine.md#M08 */
/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#programcopy */
/* The link's receive completions: each stores its completion word and posts the ring's next records. */
static void *link_receive_progress(void *argument){
  struct flow *link=argument;
  pthread_setname_np("mesh.rdma.receive");
  struct mesh_queue queue=link->provider.queues[0];
  struct ibv_wc *completion=&link->completion;
  for(;;){
    int count=queue.poll(queue.completion,1,completion);
    if(count<0){flow_lost(link,EIO,3);return NULL;}
    if(!count){
      if(!atomic_load_explicit(&link->progressing,memory_order_acquire))return NULL;
      continue;
    }
    uint64_t status_opcode;
    memcpy(&status_opcode,&completion->status,sizeof status_opcode);
    if((uint32_t)status_opcode){flow_lost(link,(uint32_t)status_opcode,2);return NULL;}
    if(!((status_opcode>>32)&IBV_WC_RECV))continue;
    struct prepared_receive *record=(void *)(uintptr_t)completion->wr_id;
    struct receive_ring *ring=link->rings+record->queue;
    ring->landed++;ring->bytes+=completion->byte_len;
    atomic_store_explicit(record->input,record->argument+record->cycle,memory_order_release);
    ring->outstanding-=record->frames;ring->completed=(int64_t)(record->invocation+(uint64_t)record->cycle*link->invocations);
    int error=ring_advance(link,ring);
    if(error){flow_lost(link,error,1);return NULL;}
  }
}

/* design/prepared-machine.md#M08 */
/* The flow's progress threads: its receives' where it has records, its SENDs' where it publishes.  0, or
   pthread_create's error. */
static int link_workers(struct flow *link){
  void *(*progress[2])(void *)={link_receive_progress,link_send_progress};
  const int needed[2]={link->receive_count!=0,link->publication_count!=0 && link->stream_count};
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_set_qos_class_np(&attributes,QOS_CLASS_USER_INTERACTIVE,0);
  int error=0;
  atomic_store_explicit(&link->progressing,1,memory_order_release);
  for(uint32_t d=0;d<2 && !error;d++)if(needed[d]){
    error=pthread_create(&link->workers[link->worker_count],&attributes,progress[d],link);
    if(!error)link->worker_count++;
  }
  pthread_attr_destroy(&attributes);
  return error;
}

/* design/bridge-and-ipc.md#flows: a flow's verbs work, on its port's queue: its queue pairs made; ready to receive and
   its receives posted; its SENDs prepared, ready to send and its threads started; or, running, its threads stopped,
   a side whose lists arrived prepared, and its threads started again.  0, or the error */
static int flow_work(struct flow *f,int job){
  int error=0;
  if(job==JOB_PAIR)return pair_up(&f->provider,f->qps);
  if(job==JOB_CHANGE){
    atomic_store_explicit(&f->progressing,0,memory_order_release);
    while(f->worker_count)pthread_join(f->workers[--f->worker_count],NULL);
  }
  if(job==JOB_RECEIVE)error=pair_receive(&f->provider,(const uint32_t (*)[2])f->pairs,f->port->heard.lid,f->port->heard.gid);
  if(!error && job!=JOB_RUN && f->side[MESH_RECEIVE].owner && !(f->configured&1u<<MESH_RECEIVE) && !(error=receives_configure(f)))
    f->configured|=1u<<MESH_RECEIVE;
  if(!error && job!=JOB_RECEIVE && f->side[MESH_SEND].owner && !(f->configured&1u<<MESH_SEND) && !(error=sends_configure(f)))
    f->configured|=1u<<MESH_SEND;
  if(!error && job==JOB_RUN)error=pair_send(&f->provider);
  if(!error && job!=JOB_RECEIVE)error=link_workers(f);
  return error;
}
static void flow_job(struct flow *f,int job){
  f->job=job;atomic_store_explicit(&f->done,0,memory_order_relaxed);
  dispatch_async(f->port->work,^{
    f->result=flow_work(f,job);
    atomic_store_explicit(&f->done,1,memory_order_release);
    wake();
  });
}

/* design/algorithm-sources.md#meshresult: a flow's threads joined and its queue pairs destroyed, on its port's queue,
   the link's census printed: each queue pair's requests and frames sent and retired by the send completions it took,
   and its ring's records landed and their bytes.  1 once nothing is held (a failed destruction keeps its handles for
   the next turn) */
static int flow_release(struct flow *flow){
  atomic_store_explicit(&flow->progressing,0,memory_order_release);
  while(flow->worker_count)pthread_join(flow->workers[--flow->worker_count],NULL);
  for(int q=0;flow->gates && flow->rings && q<flow->qps;q++)
    say("link %u census queue=%d requests=%llu frames=%llu retired=%llu send_completions=%llu records=%zu posted=%zu landed=%llu bytes=%llu\n",
      flow->port->index,q,(unsigned long long)flow->gates[q].requests,(unsigned long long)flow->gates[q].posted,
      (unsigned long long)flow->gates[q].retired,(unsigned long long)flow->gates[q].completions,
      flow->rings[q].count,flow->rings[q].posted,(unsigned long long)flow->rings[q].landed,(unsigned long long)flow->rings[q].bytes);
  if(!down_pair(&flow->provider))return 0;
  free(flow->receive);free(flow->streams);free(flow->gates);free(flow->rings);free(flow->requests);
  flow->receive=NULL;flow->streams=NULL;flow->gates=NULL;flow->rings=NULL;flow->requests=NULL;
  return 1;
}
/* A flow stopped on its port's queue, after its job: where it failed, its program's pending words toward the far end
   cancelled (M12) once its receive thread is joined, and its status the failure naming the node `where` */
static void flow_stop(struct flow *f,int64_t code,uint32_t domain,uint32_t where){
  const uint64_t owner=f->owner;const uint32_t destination=f->destination;struct mesh_cancellation *cancel=f->cancel;
  f->stopping=1;atomic_store_explicit(&f->stopped,0,memory_order_relaxed);
  dispatch_async(f->port->work,^{
    atomic_store_explicit(&f->progressing,0,memory_order_release);
    while(f->worker_count)pthread_join(f->workers[--f->worker_count],NULL);
    if(code && owned(owner)){
      if(cancel)mesh_cancel(M,cancel,destination);
      status_failed(&mesh_client_at(M,owner)->status[destination],owner,code,domain,where);
    }
    atomic_store_explicit(&f->stopped,flow_release(f)?1:2,memory_order_release);
    wake();
  });
}

/* design/bridge-and-ipc.md#protocol: a message to the port's peer, queued behind what the socket has not taken yet; a
   send that fails is the link's loss */
static void port_lost(struct port *port,int64_t code,int retire);
static void port_watch(struct port *port,int filter){
  struct kevent64_s event;EV_SET64(&event,(uint64_t)port->control,filter,EV_ADD|(filter==EVFILT_WRITE?EV_ONESHOT:0),0,0,port->index,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))port_lost(port,errno,0);
}
static void port_flush(struct port *port){
  while(port->control>=0 && port->out_sent<port->out_length){
    ssize_t sent=send(port->control,port->out+port->out_sent,port->out_length-port->out_sent,0);
    if(sent<0 && (errno==EAGAIN || errno==EINTR)){
      if(!port->writing){port->writing=1;port_watch(port,EVFILT_WRITE);}
      return;
    }
    if(sent<=0){port_lost(port,sent?errno:EIO,0);return;}
    port->out_sent+=(size_t)sent;
  }
  port->out_sent=port->out_length=0;
}
static void port_say(struct port *port,struct message said,const void *payload,uint32_t bytes){
  if(port->control<0 || (said.kind!=HELLO && !port->greeted))return;
  said.magic=XMAGIC;said.node=M->node;said.boot=M->boot;said.block=M->block*M->pgsz;said.port=port->index;said.bytes=bytes;
  said.lid=port->lid;memcpy(said.gid,port->gid,16);
  size_t length=port->out_length+sizeof said+bytes;
  if(length>port->out_room){
    unsigned char *grown=realloc(port->out,length*2);
    if(!grown){port_lost(port,ENOMEM,0);return;}
    port->out=grown;port->out_room=length*2;
  }
  memcpy(port->out+port->out_length,&said,sizeof said);
  if(bytes)memcpy(port->out+port->out_length+sizeof said,payload,bytes);
  port->out_length=length;
  port_flush(port);
}
static void flow_say(struct flow *f,uint32_t kind,uint32_t count,int64_t code,const uint32_t (*pair)[2],uint32_t origin){
  struct message said={.kind=kind,.count=count,.key=f->key,.code=code,.from=f->near,.to=f->far,.failed=origin};
  for(uint32_t q=0;pair && q<count && q<MESH_PAIRS;q++){said.pair[q][0]=pair[q][0];said.pair[q][1]=pair[q][1];}
  port_say(f->port,said,NULL,0);
}
/* a READY: the flow's queue pairs, then its route's `hops` nodes and the transfers its SEND side sends */
static void ready_say(struct flow *f,const uint32_t *route,uint32_t hops){
  int d=f->side[MESH_SEND].owner?MESH_SEND:MESH_RECEIVE;
  struct mesh_client *client=mesh_client_at(M,f->side[d].owner);
  struct mesh_tx *tx=side_tx(f,d);
  struct ready ready={.hops=hops,.slots=tx->slots,.invocations=tx->invocations,.depth=atomic_load(&client->depth),
    .cyclic=atomic_load(&client->cyclic),.queue=f->queue,.step=f->step};
  size_t transfers=0;
  for(uint32_t k=0;k<f->queues;k++)transfers+=ready.lengths[k]=side_length(f,MESH_SEND,k);
  size_t bytes=sizeof ready+(size_t)hops*sizeof *route+transfers*sizeof(struct mesh_transfer);
  unsigned char *payload=malloc(bytes),*at=payload;
  if(!payload){port_lost(f->port,ENOMEM,0);return;}
  memcpy(at,&ready,sizeof ready);at+=sizeof ready;
  memcpy(at,route,(size_t)hops*sizeof *route);at+=(size_t)hops*sizeof *route;
  for(uint32_t k=0;k<f->queues;k++){memcpy(at,side_transfers(f,MESH_SEND,k),ready.lengths[k]*sizeof(struct mesh_transfer));at+=ready.lengths[k]*sizeof(struct mesh_transfer);}
  port_say(f->port,(struct message){.kind=READY,.count=(uint32_t)f->qps,.key=f->key,.from=f->near,.to=f->far},payload,(uint32_t)bytes);
  free(payload);
}

/* design/algorithm-sources.md#meshresult: a program on a port stopped, its peer told (`tell`, END or FAIL naming the node
   `origin` where it happened), its status the failure naming `where`; a relay's other flow fails with it, telling its
   peer, and so does every flow of the same client program on its other cables */
static void flow_fail(struct flow *f,int64_t code,uint32_t domain,uint32_t tell,uint32_t where,uint32_t origin){
  if(f->state==FAILED)return;
  f->state=FAILED;
  if(tell)flow_say(f,tell,0,code,NULL,origin);
  say("link %u: program %016llx toward node %u failed at node %u: code %lld domain %u\n",f->port->index,
    (unsigned long long)f->key,f->far,where,(long long)code,domain);
  flow_stop(f,code,domain,where);
  if(f->partner)flow_fail(f->partner,code,domain,tell?tell:(uint32_t)(code==ECONNRESET?END:FAIL),origin,origin);
  for(struct flow *g=flows;f->owner && g;g=g->next)if(g->owner==f->owner && g->destination==f->destination)flow_fail(g,code,domain,FAIL,where,origin);
}
/* a program that ended, its client gone: its peer told (END) */
static void flow_end(struct flow *f){
  if(f->state==FAILED)return;
  f->state=FAILED;
  flow_say(f,END,0,0,NULL,M->node);
  flow_stop(f,0,0,0);
}
static void lease_free(struct lease *lease){
  mesh_retire(M,lease->context.client);
  unreleased=1;free(lease);
}
static void lease_drop(struct lease *lease){ if(lease && !--lease->users)lease_free(lease); }
static struct lease *lease_take(uint64_t key){
  struct lease *lease=calloc(1,sizeof *lease);
  if(lease){lease->context.fd=-1;if(mesh_slot_take(&lease->context,M,key)){free(lease);lease=NULL;}}
  return lease;
}
static void flow_free(struct flow *f){
  for(struct flow **at=&flows;*at;at=&(*at)->next)if(*at==f){*at=f->next;break;}
  if(f->partner)f->partner->partner=NULL;
  f->port->flows--;
  for(int d=0;d<2;d++){lease_drop(f->side[d].lease);lease_drop(f->pending[d].lease);}
  free(f);
}
/* design/bridge-and-ipc.md#clients: a client program's status once every flow of it toward the node runs */
static void program_paired(struct flow *f){
  for(struct flow *g=flows;g;g=g->next)
    if(g->owner==f->owner && g->destination==f->destination && (g->state!=RUNNING || g->job))return;
  if(!owned(f->owner))return;
  struct mesh_port_info *status=&mesh_client_at(M,f->owner)->status[f->destination];
  atomic_store_explicit(&status->code,0,memory_order_relaxed);atomic_store_explicit(&status->failed,0,memory_order_relaxed);
  atomic_store_explicit(&status->phase,MESH_PAIRED,memory_order_relaxed);
  status_prepared(status,f->owner);
}

/* design/bridge-and-ipc.md#protocol: a flow's next step, its job's end or a message heard: queue pairs made once the
   peer's READY or PAIRS is heard (PAIRS sent), ready to receive once its PAIRS are and, held, once its client's program
   took it (POSTED sent), running once its POSTED is; a running relay whose side's lists arrived changes */
static void flow_step(struct flow *f){
  if(f->state==FAILED || f->job || f->stopping)return;
  for(int d=0;d<2;d++)if(f->pending[d].owner && !f->side[d].owner){f->side[d]=f->pending[d];f->pending[d]=(struct side){0};}
  if(f->state==READIED && (f->heard&(HEARD_READY|HEARD_PAIRS)))flow_job(f,JOB_PAIR);
  else if(f->state==PAIRING && (f->heard&HEARD_PAIRS) && !f->hold)flow_job(f,JOB_RECEIVE);
  else if(f->state==POSTING && (f->heard&HEARD_POSTED))flow_job(f,JOB_RUN);
  else if(f->state==RUNNING && ((f->side[MESH_SEND].owner && !(f->configured&1u<<MESH_SEND)) ||
                                (f->side[MESH_RECEIVE].owner && !(f->configured&1u<<MESH_RECEIVE))))flow_job(f,JOB_CHANGE);
}
static void flow_done(struct flow *f){
  const int job=f->job;
  f->job=0;
  if(f->state==FAILED)return;
  if(f->result){flow_fail(f,f->result,1,FAIL,M->node,M->node);return;}
  if(job==JOB_PAIR){
    uint32_t pairs[MESH_PAIRS][2];
    for(int q=0;q<f->qps;q++){pairs[q][0]=f->provider.queues[q].pair->qp_num;pairs[q][1]=(f->provider.psn+(uint32_t)q)&0xffffff;}
    flow_say(f,PAIRS,(uint32_t)f->qps,0,(const uint32_t (*)[2])pairs,M->node);
    f->state=PAIRING;
    if(!f->hold)f->deadline=now+MESH_PAIR_NS;
  }else if(job==JOB_RECEIVE){
    flow_say(f,POSTED,0,0,NULL,M->node);
    /* design/recovery.md#first: the peer's POSTED waits on its program's start where no READY told of it (a peer that
       holds posts once its client's program takes it): the peer window, and the pairing window after it */
    f->state=POSTING;f->deadline=(f->heard&HEARD_READY)?now+MESH_PAIR_NS:f->created+MESH_PEER_NS+MESH_PAIR_NS;
  }else{
    if(job==JOB_RUN)f->state=RUNNING;
    program_paired(f);
  }
  flow_step(f);
}

/* design/bridge-and-ipc.md#flows: a flow made on a port toward `far` for the execution `key` */
static struct flow *flow_make(struct port *port,uint64_t key,uint32_t near,uint32_t far,uint32_t queue,uint32_t step,uint32_t queues,uint32_t slots){
  if(!queues || !slots || queues*slots>MESH_PAIRS)return NULL;
  struct flow *f=calloc(1,sizeof *f);
  if(!f)return NULL;
  *f=(struct flow){.next=flows,.near=near,.far=far,.queue=queue,.step=step?step:1,.queues=queues,.qps=(int)(queues*slots),
    .key=key,.created=now,.deadline=now+MESH_PEER_NS,.M=M,.port=port,.provider={.device=&port->device}};
  flows=f;port->flows++;
  return f;
}
static struct flow *flow_on(struct port *port,uint64_t key,uint32_t near,uint32_t far){
  for(struct flow *f=flows;f;f=f->next)if(f->port==port && f->key==key && f->near==near && f->far==far && f->state!=FAILED)return f;
  return NULL;
}
/* design/bridge-and-ipc.md#ports: the ports paired with `node`, in the cable order both ends read alike, by the index
   the lower node gave each cable */
static uint32_t cable_rank(const struct port *port){ return M->node<port->peer?port->index:port->peer_port; }
static uint32_t cables(uint32_t node,struct port **out){
  uint32_t n=0;
  for(uint32_t p=0;p<port_count;p++)if(ports[p].greeted && ports[p].peer==node && !ports[p].retiring)out[n++]=ports+p;
  for(uint32_t i=1;i<n;i++)for(uint32_t j=i;j>0 && cable_rank(out[j])<cable_rank(out[j-1]);j--){struct port *t=out[j];out[j]=out[j-1];out[j-1]=t;}
  return n;
}

/* design/bridge-and-ipc.md#flows: a relayed program a READY carried, mirrored into a lease: rows the bridge leases for
   each transfer the sender sends, bound as receives from `from` and as SENDs toward `to` over the same rows, so a landed
   record releases the SEND that carries it on (M08).  0, or the refusal */
static int mirror(struct lease *lease,const struct ready *r,const struct mesh_transfer *t,uint32_t from,uint32_t to){
  struct mesh_ctx *c=&lease->context;
  struct mesh_client *client=mesh_client_at(M,c->client);
  int error=0;
  client->tx[mesh_destination(client,from)].node=from;
  client->tx[mesh_destination(client,to)].node=to;
  for(uint32_t k=0,i=0;k<MESH_QUEUES && !error;k++)for(uint32_t j=0;j<r->lengths[k] && !error;j++,i++){
    const struct mesh_transfer *x=t+i;
    uint32_t q=r->queue+k*r->step,depth=x->depth?x->depth:r->depth;
    uint64_t tail=(uint64_t)x->invocation_pages*M->pgsz,ring=(uint64_t)(depth?depth-1:0)*tail;
    struct mesh_section s;
    if(q>=MESH_QUEUES || !x->count)error=EPROTO;
    else if(!(error=mesh_section_create(c,ring+(x->bytes>tail?x->bytes:tail),x->count,&s))){
      s.bytes=x->bytes;if(!x->stride)s.stride=0;
      error=mesh_transfer_bind(c,from,q,MESH_RECEIVE,x->binding,s,x->invocation_pages,x->begin,x->end,x->depth);
      if(!error)error=mesh_transfer_bind(c,to,q,MESH_SEND,x->binding,s,x->invocation_pages,x->begin,x->end,x->depth);
    }
  }
  if(!error)error=mesh_transfers_prepare(c,r->slots,r->invocations,r->depth);
  if(!error && r->cyclic)error=mesh_transfers_cyclic(c);
  return error;
}
/* a READY refused as a value, naming this node */
static void ready_refuse(struct port *port,const struct message *heard,int64_t code){
  port_say(port,(struct message){.kind=FAIL,.key=heard->key,.from=heard->to,.to=heard->from,.code=code,.failed=M->node},NULL,0);
}
/* a relay's direction a READY told: its lists mirrored into a lease of its own, received by the flow it arrived on
   and sent on by its partner */
static int relay_direction(struct flow *f,const struct ready *ready,const struct mesh_transfer *list){
  struct flow *g=f->partner;
  if(ready->queue || ready->step!=1)return EPROTO;
  struct lease *lease=lease_take(f->key);
  if(!lease)return EBUSY;
  int error=mirror(lease,ready,list,f->far,g->far);
  if(error){lease_free(lease);return error;}
  struct mesh_client *client=mesh_client_at(M,lease->context.client);
  f->pending[MESH_RECEIVE]=(struct side){lease->context.client,mesh_destination(client,f->far),lease};
  g->pending[MESH_SEND]=(struct side){lease->context.client,mesh_destination(client,g->far),lease};
  lease->users=2;
  if(!f->owner)flow_program(f,f->pending[MESH_RECEIVE].owner,f->pending[MESH_RECEIVE].destination);
  if(!g->owner)flow_program(g,g->pending[MESH_SEND].owner,g->pending[MESH_SEND].destination);
  flow_step(g);
  return 0;
}
/* design/bridge-and-ipc.md#protocol: a READY.  Its flow here pairs; a program whose route goes on past this node is
   relayed toward the route's next node, over rows the bridge leases (no client of this node takes part); one that ends
   here with no program of its key started is held, its queue pairs made and no rows, for the client whose program takes
   it (#flows) */
static void ready_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  struct ready ready;
  if(heard->bytes<sizeof ready){port_lost(port,EPROTO,0);return;}
  memcpy(&ready,payload,sizeof ready);
  uint64_t transfers=0;
  for(uint32_t k=0;k<MESH_QUEUES;k++)transfers+=ready.lengths[k];
  const uint32_t queues=ready.slots?heard->count/ready.slots:0;
  if(!ready.hops || !queues || !ready.step || heard->count%ready.slots || ready.queue+(uint64_t)(queues-1)*ready.step>=MESH_QUEUES ||
     heard->bytes!=sizeof ready+(uint64_t)ready.hops*sizeof(uint32_t)+transfers*sizeof(struct mesh_transfer)){port_lost(port,EPROTO,0);return;}
  uint32_t *route=malloc((size_t)ready.hops*sizeof *route);
  struct mesh_transfer *list=malloc((transfers?transfers:1)*sizeof *list);
  if(!route || !list){free(route);free(list);ready_refuse(port,heard,ENOMEM);return;}
  memcpy(route,payload+sizeof ready,(size_t)ready.hops*sizeof *route);
  memcpy(list,payload+sizeof ready+(size_t)ready.hops*sizeof *route,transfers*sizeof *list);
  struct flow *f=flow_on(port,heard->key,heard->to,heard->from);
  if(f){
    int error=f->partner && !f->side[MESH_RECEIVE].owner && !f->pending[MESH_RECEIVE].owner?relay_direction(f,&ready,list):0;
    if(error)flow_fail(f,error,1,FAIL,M->node,M->node);
    else{f->heard|=HEARD_READY;flow_step(f);}
  }else if(route[0]!=M->node || route[ready.hops-1]!=heard->to || heard->from==M->node)ready_refuse(port,heard,EPROTO);
  else if(ready.hops==1){
    f=flow_make(port,heard->key,M->node,heard->from,ready.queue,ready.step,queues,ready.slots);
    if(!f)ready_refuse(port,heard,E2BIG);
    else{f->hold=1;f->heard=HEARD_READY;flow_step(f);}
  }else{
    struct port *next[MESH_PORTS];
    struct flow *g=NULL;
    f=cables(route[1],next)?flow_make(port,heard->key,heard->to,heard->from,0,1,queues,ready.slots):NULL;
    if(f)g=flow_make(next[0],heard->key,heard->from,heard->to,0,1,queues,ready.slots);
    if(!g){
      if(f){f->state=FAILED;flow_free(f);}
      ready_refuse(port,heard,cables(route[1],next)?E2BIG:EHOSTUNREACH);
    }else{
      f->partner=g;g->partner=f;
      int error=relay_direction(f,&ready,list);
      if(error)flow_fail(f,error,1,FAIL,M->node,M->node);
      else{
        ready_say(g,route+1,ready.hops-1);
        f->heard=HEARD_READY;flow_step(f);
      }
    }
  }
  free(route);free(list);
}

/* design/bridge-and-ipc.md#ports, #region: the map, a record a node: this node's own, rebuilt from its port table when
   that changes (its sequence one more), and every other node's newest heard; each change written between two
   notifications of the map serial (odd while it writes), the records in arena pages the bridge leases and leases again
   larger as the mesh grows; nodes no paired path reaches dropped */
static int map_grow(uint64_t records){
  if(records<=map_room)return 0;
  uint32_t pages=(uint32_t)((records*2*sizeof(struct mesh_record)+M->pgsz-1)/M->pgsz),page=mesh_arena_alloc(&bridge,pages,1);
  if(page==MESH_ABSENT)return ENOMEM;
  memcpy(mesh_at(M,page),mesh_records(M),(size_t)atomic_load(&M->map_nodes)*sizeof(struct mesh_record));
  atomic_store(&M->map_off,(uint64_t)(mesh_at(M,page)-(unsigned char *)M));
  if(map_pages)mesh_bits_clear(mesh_arena_bits(M),map_page,map_pages);
  map_page=page;map_pages=pages;map_room=(uint32_t)((uint64_t)pages*M->pgsz/sizeof(struct mesh_record));
  return 0;
}
static void map_put(const struct mesh_record *record){
  mesh_notify(&M->map);
  uint64_t n=atomic_load(&M->map_nodes),i=0;
  while(i<n && mesh_records(M)[i].node!=record->node)i++;
  const int fresh=i==n;
  const uint64_t at=i;
  if(i<n || !map_grow(n+1)){
    memcpy(mesh_records(M)+i,record,sizeof *record);
    if(i==n)atomic_store(&M->map_nodes,n+1);
  }
  /* RFC 2328 s14: a record no paired path from this node reaches leaves the map */
  n=atomic_load(&M->map_nodes);
  struct mesh_record *r=mesh_records(M);
  unsigned char *reached=calloc(n?n:1,1);
  uint32_t *order=malloc((n?n:1)*sizeof *order),head=0,tail=0;
  for(uint64_t k=0;reached && order && k<n;k++)if(r[k].node==M->node){reached[k]=1;order[tail++]=(uint32_t)k;}
  while(reached && order && head<tail){
    uint32_t a=order[head++];double beta;
    for(uint64_t b=0;b<n;b++)if(!reached[b] && mesh_joined(r+a,r+b,&beta)){reached[b]=1;order[tail++]=(uint32_t)b;}
  }
  /* design/recovery.md#diagnosis: a node the prune drops is logged with its last health first (one put now and never
     reached yet is not a loss) */
  for(uint64_t k=0;reached && order && k<n;k++)if(!reached[k] && !(fresh && k==at)){
    struct known *gone=known_at(r[k].node);char text[768];health_text(&r[k].health,text,sizeof text);
    say("node %u (%s) unreachable; its last health, heard %.1f s before: %s\n",r[k].node,r[k].name,
        gone && gone->heard?(double)(now-gone->heard)/1e9:-1.0,text);
    if(gone)gone->gone=now;
  }
  uint64_t kept=0;
  for(uint64_t k=0;reached && order && k<n;k++)if(reached[k])memmove(r+kept++,r+k,sizeof *r);
  if(reached && order)atomic_store(&M->map_nodes,kept);
  free(reached);free(order);
  mesh_notify(&M->map);
}
/* design/recovery.md#diagnosis: a record whose links, name and boot are the held one's changes only its health and
   sequence, written under the region's health serial and never the map's: a communicator's session opens again where
   the map serial moved (nccl-mesh.c session_ensure), and a health sample is no change of the mesh */
static int record_links_same(const struct mesh_record *a,const struct mesh_record *b){
  return a->node==b->node && a->ports==b->ports && a->boot==b->boot && !memcmp(a->name,b->name,sizeof a->name) &&
    !memcmp(a->port,b->port,sizeof a->port);
}
static void health_put(struct mesh_record *held,const struct mesh_record *record){
  mesh_notify(&M->health);
  held->sequence=record->sequence;held->health=record->health;
  mesh_notify(&M->health);
}
static struct mesh_record *record_held(uint32_t node){
  for(uint64_t i=0;i<atomic_load(&M->map_nodes);i++)if(mesh_records(M)[i].node==node)return mesh_records(M)+i;
  return NULL;
}
static void record_changed(void){
  struct mesh_record own={.node=M->node,.ports=port_count,.boot=M->boot,.sequence=++sequence};
  memcpy(own.name,host,sizeof own.name);
  memcpy(own.port,mesh_links(M),sizeof own.port);
  own.health=health;
  struct mesh_record *held=record_held(M->node);
  if(held && record_links_same(held,&own))health_put(held,&own);
  else map_put(&own);
  for(uint32_t p=0;p<port_count;p++)if(ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&own,sizeof own);
}
/* design/recovery.md#diagnosis: this node's health sampled each turn; a change of its power source, of its charge off
   AC, or of its memory or thermal pressure is logged and sent at once, anything else with its record at most
   MESH_HEALTH_NS later */
static void health_turn(void){
  struct mesh_health sampled=health;
  health_sample(&sampled);
  const int urgent=sampled.power!=health.power || (sampled.power!=MESH_POWER_AC && sampled.charge!=health.charge) ||
    sampled.pressure!=health.pressure || sampled.thermal!=health.thermal;
  health=sampled;
  if(urgent){char text[768];health_text(&health,text,sizeof text);say("health: %s\n",text);}
  if(urgent || now-health_sent>=MESH_HEALTH_NS){health_sent=now;record_changed();}
}
static void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes){
  struct mesh_record heard;
  if(bytes!=sizeof heard)return;
  memcpy(&heard,payload,sizeof heard);
  heard.name[sizeof heard.name-1]=0;
  heard.health.session[sizeof heard.health.session-1]=0;heard.health.shutdown[sizeof heard.health.shutdown-1]=0;
  if(!heard.node || heard.node==M->node || heard.ports>MESH_PORTS)return;
  struct mesh_record *held=record_held(heard.node);
  if(held && (heard.boot<held->boot || (heard.boot==held->boot && heard.sequence<=held->sequence)))return;
  /* design/recovery.md#diagnosis: a node that joins or comes back, and a peer's urgent change, are logged here */
  struct known *k=known_at(heard.node);
  char text[768];
  if(!held && k && (k->gone || !k->heard)){
    health_text(&heard.health,text,sizeof text);
    if(k->gone)say("node %u (%s) back after %.1f s, %s: %s\n",heard.node,heard.name,(double)(now-k->gone)/1e9,
                   strncmp(k->session,heard.health.session,sizeof k->session)?"a new boot":"the same boot",text);
    else say("node %u (%s) joined: %s\n",heard.node,heard.name,text);
  }
  if(held && (held->health.power!=heard.health.power || held->health.pressure!=heard.health.pressure ||
              held->health.thermal!=heard.health.thermal ||
              (heard.health.power!=MESH_POWER_AC && held->health.charge!=heard.health.charge))){
    health_text(&heard.health,text,sizeof text);
    say("node %u (%s) health: %s\n",heard.node,heard.name,text);
  }
  if(k){k->heard=now;k->gone=0;memcpy(k->session,heard.health.session,sizeof k->session);}
  if(held && record_links_same(held,&heard))health_put(held,&heard);
  else map_put(&heard);
  for(uint32_t p=0;p<port_count;p++)if(ports+p!=from && ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&heard,sizeof heard);
}
/* a port's link as the port table tells it; a change is a new record */
static void link_set(struct port *port,uint32_t peer,uint32_t phase,int64_t code){
  struct mesh_link_info *link=&mesh_links(M)[port->index];
  int changed=atomic_load(&link->peer)!=peer || atomic_load(&link->phase)!=phase;
  atomic_store_explicit(&link->peer,peer,memory_order_relaxed);atomic_store_explicit(&link->code,code,memory_order_relaxed);
  atomic_store_explicit(&link->phase,phase,memory_order_release);
  if(changed)record_changed();
}

/* design/bridge-and-ipc.md#protocol */
static void port_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  if(!port->greeted){
    if(heard->kind!=HELLO || heard->block!=M->block*M->pgsz || (port->peer && heard->node!=port->peer)){port_lost(port,EPROTO,0);return;}
    port->peer=heard->node;port->peer_port=heard->port;port->heard=*heard;port->greeted=1;
    struct mesh_link_info *link=&mesh_links(M)[port->index];
    atomic_store_explicit(&link->bandwidth,port->bandwidth,memory_order_relaxed);
    atomic_fetch_add_explicit(&link->pairings,1,memory_order_relaxed);
    say("link %u (%s): paired with node %u\n",port->index,port->name,port->peer);
    link_set(port,port->peer,MESH_PAIRED,0);
    /* a port newly paired is sent every record the map holds, this node's first */
    for(uint64_t i=0;port->greeted && i<atomic_load(&M->map_nodes);i++)if(mesh_records(M)[i].node!=M->node)
      port_say(port,(struct message){.kind=RECORD},mesh_records(M)+i,sizeof(struct mesh_record));
    return;
  }
  if(heard->kind==RECORD){record_heard(port,payload,heard->bytes);return;}
  if(heard->kind==FAREWELL){port->parted=1;port->farewell=heard->code;return;}
  if(heard->kind==READY){ready_heard(port,heard,payload);return;}
  struct flow *f=flow_on(port,heard->key,heard->to,heard->from);
  if(!f)return;
  const uint32_t origin=heard->failed?heard->failed:port->peer;
  if(heard->kind==PAIRS && heard->count!=(uint32_t)f->qps)flow_fail(f,EPROTO,4,FAIL,port->peer,M->node);
  else if(heard->kind==PAIRS){memcpy(f->pairs,heard->pair,sizeof f->pairs);f->heard|=HEARD_PAIRS;}
  else if(heard->kind==POSTED)f->heard|=HEARD_POSTED;
  else if(heard->kind==END)flow_fail(f,ECONNRESET,4,0,origin,origin);
  else if(heard->kind==FAIL)flow_fail(f,ECANCELED,4,0,origin,origin);
  flow_step(f);
}

/* design/recovery.md#detection, #loss: a link lost fails every program on it, naming its peer; a port that went away (its
   link off, its interface detached) retires its device (design/RDMA-KERNEL-RECOVERY.md); the next pairing is a pause
   away */
static void port_lost(struct port *port,int64_t code,int retire){
  if(port->control>=0){close(port->control);port->control=-1;}
  /* design/recovery.md#diagnosis: the peer named, and a peer that said farewell told from one lost */
  if(port->greeted||port->connecting){
    const char *retired=retire?", device retired":"";
    char reason[96];farewell_text(port->farewell,reason,sizeof reason);
    if(stop)say("link %u (%s): closed toward node %u: this bridge is stopping%s\n",port->index,port->name,port->peer,retired);
    else if(port->peer && port->parted)say("link %u (%s): node %u parted, its bridge stopping (%s): code %lld%s\n",port->index,port->name,
                                            port->peer,reason,(long long)code,retired);
    else if(port->peer)say("link %u (%s): node %u lost: code %lld%s\n",port->index,port->name,port->peer,(long long)code,retired);
    else say("link %u (%s): lost: code %lld%s\n",port->index,port->name,(long long)code,retired);
  }
  for(struct flow *f=flows;f;f=f->next)if(f->port==port)flow_fail(f,port->parted?ECONNRESET:code,4,0,port->peer,M->node);
  port->greeted=port->parted=port->connecting=port->writing=0;port->farewell=0;port->peer=0;port->held=0;port->out_length=port->out_sent=0;
  port->retiring|=retire;port->open&=!retire;port->next=now+MESH_PAUSE_NS;
  link_set(port,0,retire?MESH_STOPPED:MESH_PAIRING,code);
}
/* design/recovery.md#detection: the control socket's keepalive tells a side whose own link stays up (its peer's went
   down, its peer's host died); the socket stays non-blocking, its sends queued (port_say) */
static void port_greet(struct port *port){
  int on=1,idle=1,interval=1,count=2,flags=fcntl(port->control,F_GETFL);
  if(flags<0 || fcntl(port->control,F_SETFL,flags|O_NONBLOCK) || setsockopt(port->control,SOL_SOCKET,SO_KEEPALIVE,&on,sizeof on) ||
     setsockopt(port->control,IPPROTO_TCP,TCP_KEEPALIVE,&idle,sizeof idle) || setsockopt(port->control,IPPROTO_TCP,TCP_KEEPINTVL,&interval,sizeof interval) ||
     setsockopt(port->control,IPPROTO_TCP,TCP_KEEPCNT,&count,sizeof count) ||
     setsockopt(port->control,SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on)){port_lost(port,errno?errno:EIO,0);return;}
  port_watch(port,EVFILT_READ);
  port_say(port,(struct message){.kind=HELLO},NULL,0);
}
/* design/bridge-and-ipc.md#ports: the port a link-local peer reached, by its interface; none for a scope of 0 (no
   interface: a LAN address, or the index of an interface that is gone), nor for a port whose device is retiring, which
   its turn alone opens again (design/RDMA-KERNEL-RECOVERY.md) */
static struct port *port_at(uint32_t interface){
  for(uint32_t p=0;interface && p<port_count;p++)if(ports[p].interface==interface)return ports[p].retiring?NULL:ports+p;
  return NULL;
}
/* design/bridge-and-ipc.md#ports: a port's turn: on its own queue, its device retired where its port went away (once no
   flow holds queue pairs on it), else opened and its port read; then, while it has no peer, its hello sent to every
   node on its link */
static void port_turn(struct port *port){
  if(port->job || port->control>=0 || now<port->next || (port->retiring && port->flows))return;
  port->next=now+MESH_PAUSE_NS;
  port->interface=if_nametoindex(port->name);
  const int retire=port->retiring;
  port->job=1;atomic_store_explicit(&port->done,0,memory_order_relaxed);
  dispatch_async(port->work,^{
    int result=0;
    if(retire && !down_device(&port->device))result=-(errno?errno:EBUSY);
    else if(device_up(&port->device,&wire) || ibv_query_gid(port->device.context,1,0,&port->device.gid))result=errno?errno:EIO;
    port->result=result;
    atomic_store_explicit(&port->done,1,memory_order_release);
    wake();
  });
}
static void port_opened(struct port *port){
  const int result=port->result;
  port->job=0;
  if(result<0){say("link %u (%s): device retirement: %s\n",port->index,port->name,strerror(-result));return;}
  port->retiring=result && port->device.context && result!=ENETDOWN;
  if(result){port->open=0;if(port->control<0)link_set(port,0,MESH_STOPPED,result);return;}
  port->open=1;port->lid=port->device.port.lid;memcpy(port->gid,&port->device.gid,16);
  port->bandwidth=port_bandwidth(&port->device.port);
  if(port->control>=0)return;
  link_set(port,0,MESH_PAIRING,0);
  struct ipv6_mreq group={.ipv6mr_multiaddr=everyone,.ipv6mr_interface=port->interface};
  setsockopt(hello,IPPROTO_IPV6,IPV6_JOIN_GROUP,&group,sizeof group);
  setsockopt(hello,IPPROTO_IPV6,IPV6_MULTICAST_IF,&port->interface,sizeof port->interface);
  struct sockaddr_in6 to={.sin6_len=sizeof to,.sin6_family=AF_INET6,.sin6_port=htons((uint16_t)atoi(MESH_PORT)),.sin6_addr=everyone,.sin6_scope_id=port->interface};
  struct message said={.magic=XMAGIC,.kind=HELLO,.node=M->node,.boot=M->boot,.block=M->block*M->pgsz,.port=port->index};
  sendto(hello,&said,sizeof said,0,(struct sockaddr *)&to,sizeof to);
}
/* design/bridge-and-ipc.md#ports: a node heard on a port: the lower node dials the higher's listener on the address
   it heard from, the higher answers its hello so the lower hears it at once */
static void hello_heard(void){
  struct message heard;struct sockaddr_in6 from;socklen_t length=sizeof from;
  while(recvfrom(hello,&heard,sizeof heard,0,(struct sockaddr *)&from,&length)==(ssize_t)sizeof heard){
    struct port *port=port_at(from.sin6_scope_id);
    length=sizeof from;
    if(!port || heard.magic!=XMAGIC || heard.kind!=HELLO || heard.node==M->node || port->control>=0 || !port->open)continue;
    if(M->node>heard.node){
      struct message said={.magic=XMAGIC,.kind=HELLO,.node=M->node,.boot=M->boot,.block=M->block*M->pgsz,.port=port->index};
      sendto(hello,&said,sizeof said,0,(struct sockaddr *)&from,sizeof from);
      continue;
    }
    from.sin6_port=htons((uint16_t)atoi(MESH_PORT));
    port->control=socket(AF_INET6,SOCK_STREAM,0);
    if(port->control<0 || fcntl(port->control,F_SETFL,O_NONBLOCK) ||
       (connect(port->control,(struct sockaddr *)&from,sizeof from) && errno!=EINPROGRESS)){port_lost(port,errno,0);continue;}
    port->peer=heard.node;port->connecting=1;port->next=now+MESH_PAIR_NS;
    port_watch(port,EVFILT_WRITE);
  }
}
/* a control socket's messages, each a header and its `bytes` */
static void control_heard(struct port *port){
  for(;;){
    size_t need=sizeof(struct message);
    if(port->held>=need){
      const struct message *header=(const void *)port->in;
      if(header->magic!=XMAGIC || header->bytes>(UINT32_C(64)<<20)){port_lost(port,EPROTO,0);return;}
      need+=header->bytes;
    }
    if(need>port->in_room){
      unsigned char *grown=realloc(port->in,need);
      if(!grown){port_lost(port,ENOMEM,0);return;}
      port->in=grown;port->in_room=need;
    }
    if(port->held<need){
      ssize_t got=recv(port->control,port->in+port->held,need-port->held,0);
      if(got<0 && (errno==EAGAIN || errno==EINTR))return;
      if(got<=0){port_lost(port,got?errno:(port->parted?ECONNRESET:ENETRESET),0);return;}
      port->held+=(size_t)got;
      if(port->held<need || need==sizeof(struct message))continue;
    }
    struct message heard;memcpy(&heard,port->in,sizeof heard);
    port->held=0;
    port_heard(port,&heard,port->in+sizeof heard);
    if(port->control<0)return;
  }
}
/* design/bridge-and-ipc.md#ports: every RDMA device of the node a port, in the order found, read on the finder's queue
   (at the start, and where an interface attached or its link came on that no port has) and added on the event loop */
static void ports_find(void){
  if(finding)return;
  finding=1;
  dispatch_async(finder,^{
    struct ibv_device **list=ibv_get_device_list(NULL);
    uint32_t n=0;
    for(int i=0;list && list[i] && n<MESH_PORTS;i++)if(!strncmp(ibv_get_device_name(list[i]),"rdma_",5))
      snprintf(devices[n++],sizeof devices[0],"%s",ibv_get_device_name(list[i]));
    if(list)ibv_free_device_list(list);
    device_count=n;
    atomic_store_explicit(&found,1,memory_order_release);
    wake();
  });
}
static void ports_found(void){
  finding=0;atomic_store_explicit(&found,0,memory_order_relaxed);
  for(uint32_t i=0;i<device_count && port_count<MESH_PORTS;i++){
    uint32_t known=0;
    for(uint32_t p=0;p<port_count;p++)known|=!strcmp(ports[p].device.name,devices[i]);
    if(known)continue;
    struct port *port=ports+port_count;
    *port=(struct port){.index=port_count,.control=-1,.work=dispatch_queue_create("mesh.port",DISPATCH_QUEUE_SERIAL)};
    snprintf(port->device.name,sizeof port->device.name,"%s",devices[i]);snprintf(port->name,sizeof port->name,"%s",devices[i]+5);
    port_count++;
    link_set(port,0,MESH_STOPPED,0);
  }
}
/* design/prepared-machine.md#M24: a port's link off or its interface detached; an interface that attached or whose link
   came on and is no port yet: the node's devices read again */
static void network_heard(void){
  union mesh_network_event event;
  int unknown=0;
  while(recv(network,event.bytes,sizeof event,0)>=(ssize_t)(KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data))){
    const struct net_event_data *interface=(const void *)(event.bytes+KEV_MSG_HEADER_SIZE);
    const uint32_t code=event.header.event_code;
    char name[IFNAMSIZ+16];
    snprintf(name,sizeof name,"%.*s%u",(int)sizeof interface->if_name,interface->if_name,interface->if_unit);
    int known=0;
    for(uint32_t p=0;p<port_count;p++)if(!strcmp(name,ports[p].name)){
      known=1;
      if(code==KEV_DL_LINK_OFF || code==KEV_DL_IF_DETACHED)port_lost(ports+p,ENETDOWN,1);
      else if(code==KEV_DL_LINK_ON || code==KEV_DL_IF_ATTACHED)ports[p].next=now;
    }
    unknown|=!known && (code==KEV_DL_LINK_ON || code==KEV_DL_IF_ATTACHED);
  }
  if(unknown)ports_find();
}

/* design/bridge-and-ipc.md#clients: a started program's destination taken onto its ports: over the cables paired with
   its node, its queues spread over them in cable order (queue q on cable q mod n), else along the map's route over the
   first hop's cable, a READY on each; a flow held here for its key taken alike, its sides the client's (its one job
   before, JOB_PAIR, reads none), its READY telling a relay on the route the program's other direction, its receives
   then posted (flow_step).  Refused as a value naming the node where neither reaches it (EHOSTUNREACH) or it is this
   node (EINVAL) */
static void program_take(struct mesh_client *client,uint64_t owner,uint32_t d){
  const uint32_t node=client->tx[d].node;
  struct port *cable[MESH_PORTS];
  uint32_t *route=NULL,hops=0,n=node==M->node?0:cables(node,cable);
  if(!n && node!=M->node && (hops=mesh_route(M,M->node,node,&route))>=2)n=cables(route[1],cable)?1:0;
  if(!n){free(route);status_failed(&client->status[d],owner,node==M->node?EINVAL:EHOSTUNREACH,4,node);return;}
  const uint32_t step=hops?1:n,slots=client->tx[d].slots;
  for(uint32_t j=0;j<n && j<MESH_QUEUES;j++){
    uint32_t queues=0;
    for(uint32_t q=j;q<MESH_QUEUES;q+=step)for(int w=0;w<2;w++)if(atomic_load(mesh_order_length(M,owner,d*MESH_QUEUES+q,w)))queues=(q-j)/step+1;
    if(!queues)continue;
    struct flow *f=flow_on(cable[j],client->key,M->node,node);
    if(f && !f->hold){status_failed(&client->status[d],owner,EBUSY,4,M->node);break;}
    if(f && (f->queue!=j || f->step!=step || f->queues!=queues || (uint32_t)f->qps!=queues*slots)){flow_fail(f,EPROTO,4,FAIL,M->node,M->node);f=NULL;}
    if(!f && !(f=flow_make(cable[j],client->key,M->node,node,j,step,queues,slots))){status_failed(&client->status[d],owner,E2BIG,4,M->node);break;}
    f->side[MESH_SEND]=f->side[MESH_RECEIVE]=(struct side){owner,d,NULL};
    flow_program(f,owner,d);
    if(f->hold)f->deadline=now+MESH_PAIR_NS;
    f->hold=0;
    atomic_store_explicit(&client->status[d].phase,MESH_PAIRING,memory_order_relaxed);
    if(hops)ready_say(f,route+1,hops-1);
    else ready_say(f,&node,1);
    flow_step(f);
  }
  free(route);
}
/* design/bridge-and-ipc.md#clients: every client slot read: a slot's process watched from the turn its slot is first
   seen (its exit retires every slot it holds, the event loop's EVFILT_PROC), one already gone retired at once, and on a
   liveness turn any client whose process exited retired; a program that ended (stopped, its client gone) leaves its
   ports; a program started is taken toward each destination it binds, once no flow of the slot's earlier client
   remains.  Rows of a retired client or lease are released once no flow of one remains */
static uint64_t owners[MESH_CLIENTS],accepted[MESH_CLIENTS];
static void clients_turn(int liveness){
  const pid_t self=getpid();
  int changed=0,zombie=0;
  for(struct flow *f=flows;f;f=f->next)
    if(f->owner && (pid_t)(uint32_t)f->owner!=self && (!owned(f->owner) || atomic_load(&mesh_client_at(M,f->owner)->configured)!=f->owner))flow_end(f);
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct mesh_client *client=(struct mesh_client *)((char *)M+M->client_off)+s;
    uint64_t owner=atomic_load(&client->owner);
    if(owner && (pid_t)(uint32_t)owner==self)continue;
    if(liveness && owner && !(owner&MESH_RETIRING) && mesh_gone((pid_t)(uint32_t)owner)){mesh_retire(M,owner);owner=atomic_load(&client->owner);}
    if(owner && owner!=owners[s] && !(owner&MESH_RETIRING)){
      struct kevent64_s watch;EV_SET64(&watch,(uint32_t)owner,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,s,0,0);
      if(kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)<0 && errno==ESRCH){mesh_retire(M,owner);owner=atomic_load(&client->owner);}
    }
    changed|=owner!=owners[s];owners[s]=owner;
    if(!owner || (owner&MESH_RETIRING) || atomic_load_explicit(&client->configured,memory_order_acquire)!=owner || accepted[s]==owner)continue;
    int earlier=0;
    for(struct flow *f=flows;f;f=f->next)earlier|=f->owner && f->owner!=owner && ((f->owner>>32)&(MESH_CLIENTS-1))==s && (pid_t)(uint32_t)f->owner!=self;
    if(earlier)continue;
    accepted[s]=owner;
    for(uint32_t d=0,n=mesh_destination(client,0);d<n;d++)if(mesh_destination_transfers(M,owner,d))program_take(client,owner,d);
  }
  for(struct flow *f=flows;f;f=f->next)
    zombie|=(f->owner && !owned(f->owner)) || (f->side[0].owner && !owned(f->side[0].owner)) || (f->side[1].owner && !owned(f->side[1].owner));
  unreleased|=changed;
  if(unreleased && !zombie){unreleased=0;mesh_retired_release(M);}
}
/* design/bridge-and-ipc.md#flows: every flow's turn: its job's end taken, a stopped flow freed (or its queue pairs
   destroyed again a pause later, where the driver kept them), a data thread's failure or a window passed (its peer's
   program never started, its queue pairs never paired, a held program no client took) its failure */
static void flows_turn(void){
  for(struct flow *f=flows,*next;f;f=next){
    next=f->next;
    if(f->job && atomic_load_explicit(&f->done,memory_order_acquire))flow_done(f);
    if(f->stopping){
      const int stopped=atomic_load_explicit(&f->stopped,memory_order_acquire);
      if(stopped==1 && !f->job)flow_free(f);
      else if(stopped==2 && !f->retry)f->retry=now+MESH_PAUSE_NS;
      else if(stopped==2 && now>=f->retry){f->retry=0;flow_stop(f,0,0,0);}
      continue;
    }
    if(f->state==FAILED)continue;
    const int64_t failed=atomic_load(&f->failed);
    if(failed)flow_fail(f,failed,atomic_load(&f->domain),FAIL,M->node,M->node);
    else if(f->state!=RUNNING && now>f->deadline)flow_fail(f,ETIMEDOUT,4,FAIL,f->far,f->far);
  }
}

/* design/prepared-machine.md#M26: the clients' control word, watched for the event loop */
static void *control_watch(void *unused){
  (void)unused;
  while(!stop){
    uint64_t seen=atomic_load_explicit(&M->control,memory_order_acquire);
    wake();
    os_sync_wait_on_address(&M->control,seen,sizeof M->control,OS_SYNC_WAIT_ON_ADDRESS_SHARED);
  }
  return NULL;
}

/* design/prepared-machine.md#M09: the bridge's configuration, the canonical property list bin/mesh-user.sh writes from
   the consumer's document (rdma/bridge_configuration.py): its name and its window in bytes; NULL, or why it is refused */
static const char *configuration_read(const char *path,char *name,size_t room,uint64_t *window){
  FILE *file=fopen(path,"rb");
  if(!file)return strerror(errno);
  unsigned char bytes[65536];
  const size_t length=fread(bytes,1,sizeof bytes,file);
  const int whole=feof(file);
  fclose(file);
  if(!whole)return "not a configuration property list (too long)";
  CFDataRef data=CFDataCreate(NULL,bytes,(CFIndex)length);
  CFPropertyListRef list=data?CFPropertyListCreateWithData(NULL,data,kCFPropertyListImmutable,NULL,NULL):NULL;
  if(data)CFRelease(data);
  const char *refused=NULL;
  CFTypeRef named=list && CFGetTypeID(list)==CFDictionaryGetTypeID()?CFDictionaryGetValue(list,CFSTR("name")):NULL;
  CFTypeRef sized=list && CFGetTypeID(list)==CFDictionaryGetTypeID()?CFDictionaryGetValue(list,CFSTR("window")):NULL;
  int64_t value=0;
  if(!list || CFGetTypeID(list)!=CFDictionaryGetTypeID())refused="not a property list of a configuration (rdma/bridge_configuration.py writes it)";
  else if(!named || CFGetTypeID(named)!=CFStringGetTypeID() || !CFStringGetCString(named,name,(CFIndex)room,kCFStringEncodingUTF8))refused="its name is missing";
  else if(!sized || CFGetTypeID(sized)!=CFNumberGetTypeID() || !CFNumberGetValue(sized,kCFNumberSInt64Type,&value) || value<=0)
    refused="its window, a positive number of bytes, is missing";
  if(list)CFRelease(list);
  *window=(uint64_t)value;
  return refused;
}

/* design/recovery.md#diagnosis: a stopping bridge's FAREWELL sent before its socket closes, waiting at most 200 ms on a
   peer that does not read */
static void farewell_flush(struct port *port){
  if(port->control<0)return;
  struct timeval bound={0,200000};
  const int flags=fcntl(port->control,F_GETFL);
  if(flags>=0)fcntl(port->control,F_SETFL,flags&~O_NONBLOCK);
  setsockopt(port->control,SOL_SOCKET,SO_SNDTIMEO,&bound,sizeof bound);
  while(port->out_sent<port->out_length){
    const ssize_t sent=send(port->control,port->out+port->out_sent,port->out_length-port->out_sent,0);
    if(sent<=0)break;
    port->out_sent+=(size_t)sent;
  }
}

/* design/bridge-and-ipc.md: the node's resident bridge; its one operand the path of its configuration (no default) */
int main(int argc,char **argv){
  if(argc!=2)die("usage: mesh-flow CONFIGURATION: the canonical property list bin/mesh-user.sh writes from the node's bridge configuration document (etc/bridge/README.md); the bridge has no default and does not start without one");
  char name[256];uint64_t window;
  const char *refused=configuration_read(argv[1],name,sizeof name,&window);
  if(refused){say("bridge configuration %s refused: %s\n",argv[1],refused);exit(1);}
  struct stat binary,current;
  if(stat(argv[0],&binary))memset(&binary,0,sizeof binary);
  uuid_t uuid;struct timespec bound={5,0};
  if(gethostuuid(uuid,&bound)){say("host identity: %s; this bridge's node is drawn at random\n",strerror(errno));arc4random_buf(uuid,sizeof uuid);}
  uint32_t node=0;
  for(int i=0;i<16;i++)node=(node*0x01000193u)^uuid[i];
  node=(node&0x7fffffffu)?node&0x7fffffffu:1;
  CFStringRef local=SCDynamicStoreCopyLocalHostName(NULL);
  if(!local || !CFStringGetCString(local,host,sizeof host,kCFStringEncodingUTF8))gethostname(host,sizeof host);
  if(local)CFRelease(local);
  host[sizeof host-1]=0;
  const uint32_t pg=(uint32_t)getpagesize();
  /* design/collective-dependency-ledger.md#d6-paired-send-and-receive-frame-counts-match */
  if((uint64_t)pg+4096>16773120)die("transport chunk exceeds native request capacity");
  const uint64_t extents=(window+MESH_REGION-1)/MESH_REGION,registrable=window_registrable();
  if(extents>registrable){
    say("bridge configuration %s refused: its window of %llu bytes is %llu extents of %zu bytes, past the %llu the node's RDMA devices register (max_mr)\n",
      name,(unsigned long long)window,(unsigned long long)extents,MESH_REGION,(unsigned long long)registrable);
    exit(1);
  }
  if(window<pg || window/pg>UINT32_MAX){say("bridge configuration %s refused: its window of %llu bytes is not between one page and 2^32 pages\n",name,(unsigned long long)window);exit(1);}
  struct hdr geometry={0};
  uint64_t length=mesh_layout(&geometry,pg,1,(uint32_t)(window/pg));
  signal(SIGPIPE,SIG_IGN);signal(SIGINT,SIG_IGN);signal(SIGTERM,SIG_IGN);signal(SIGHUP,SIG_IGN);
  atexit(down);
  shm_unlink(MESH_NAME);int fd=shm_open(MESH_NAME,O_CREAT|O_RDWR,MESH_MODE);if(fd<0)die("shm");
  if(ftruncate(fd,(off_t)length))die("ftruncate");fchmod(fd,MESH_MODE);
  M=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  if(M==MAP_FAILED)die("mmap");shm=MESH_NAME;
  /* a boot is the bridge's start on the wall clock, so a node's later boot is a newer record */
  *M=geometry;M->node=node;M->version=MESH_VERSION;M->boot=clock_gettime_nsec_np(CLOCK_REALTIME);
  for(uint32_t r=0;r<mesh_rows(M);r++)atomic_store_explicit(&mesh_page(M)[r].mapping,MESH_ABSENT,memory_order_relaxed);
  if(wire_map(&wire,M,fd))die("transport page aliases");
  close(fd);
  bridge.M=M;
  finder=dispatch_queue_create("mesh.ports",DISPATCH_QUEUE_SERIAL);
  events=kqueue();
  health_init();
  record_changed();
  if(!atomic_load(&M->map_nodes))die("the map's lease");
  struct kevent64_s event;
  int on=1;
  struct sockaddr_in6 any={.sin6_len=sizeof any,.sin6_family=AF_INET6,.sin6_port=htons((uint16_t)atoi(MESH_PORT)),.sin6_addr=IN6ADDR_ANY_INIT};
  listener=socket(AF_INET6,SOCK_STREAM,0);hello=socket(AF_INET6,SOCK_DGRAM,0);network=socket(PF_SYSTEM,SOCK_RAW,SYSPROTO_EVENT);
  struct kev_request filter={KEV_VENDOR_APPLE,KEV_NETWORK_CLASS,KEV_DL_SUBCLASS};
  unsigned loop=0;
  if(events<0 || listener<0 || hello<0 || network<0 || setsockopt(listener,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on) ||
     setsockopt(hello,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on) || setsockopt(hello,IPPROTO_IPV6,IPV6_MULTICAST_LOOP,&loop,sizeof loop) ||
     bind(listener,(struct sockaddr *)&any,sizeof any) || listen(listener,MESH_PORTS) || bind(hello,(struct sockaddr *)&any,sizeof any) ||
     fcntl(listener,F_SETFL,O_NONBLOCK) || fcntl(hello,F_SETFL,O_NONBLOCK) || fcntl(network,F_SETFL,O_NONBLOCK) || ioctl(network,SIOCSKEVFILT,&filter))
    die("bridge sockets");
  int watched[3]={listener,hello,network};
  for(int i=0;i<3;i++){EV_SET64(&event,(uint64_t)watched[i],EVFILT_READ,EV_ADD,0,0,UINT64_MAX,0,0);if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");}
  EV_SET64(&event,0,EVFILT_USER,EV_ADD|EV_CLEAR,0,0,UINT64_MAX,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");
  int signals[3]={SIGINT,SIGTERM,SIGHUP};
  for(int i=0;i<3;i++){EV_SET64(&event,(uint64_t)signals[i],EVFILT_SIGNAL,EV_ADD,0,0,UINT64_MAX,0,0);kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);}
  atomic_store(&M->bridge_pid,(uint64_t)getpid());
  __sync_synchronize();M->magic=MESH_MAGIC;
  pthread_t watcher;
  if(pthread_create(&watcher,NULL,control_watch,NULL))die("control watcher");
  ports_find();
  say("bridge node %u (%s): configuration %s, window %llu bytes, arena %llu pages, rows %u\n",node,host,name,
    (unsigned long long)window,(unsigned long long)mesh_arena_pages(M),M->rows);
  uint64_t liveness=0;
  while(!stop){
    now=clock_ns();
    if(atomic_load_explicit(&found,memory_order_acquire))ports_found();
    for(uint32_t p=0;p<port_count;p++){
      struct port *port=ports+p;
      if(port->job && atomic_load_explicit(&port->done,memory_order_acquire))port_opened(port);
      if((port->connecting || (port->control>=0 && !port->greeted)) && now>port->next)port_lost(port,ETIMEDOUT,0);
      port_turn(port);
    }
    flows_turn();
    health_turn();
    int alive=now>=liveness;
    if(alive){
      liveness=now+MESH_PAUSE_NS;
      if(binary.st_ino && (stat(argv[0],&current) || current.st_ino!=binary.st_ino || current.st_mtimespec.tv_sec!=binary.st_mtimespec.tv_sec)){stop=1;stop_reason=MESH_STOP_BINARY;}
    }
    clients_turn(alive);
    struct kevent64_s ready[16];
    struct timespec tick={0,250000000};
    int count=kevent64(events,NULL,0,ready,16,0,&tick);
    now=clock_ns();
    for(int i=0;i<count;i++){
      struct kevent64_s *e=ready+i;
      if(e->filter==EVFILT_SIGNAL){stop=1;stop_reason=(int64_t)e->ident;}
      else if(e->filter==EVFILT_PROC){
        for(uint32_t s=0;s<MESH_CLIENTS;s++){
          uint64_t owner=atomic_load(&((struct mesh_client *)((char *)M+M->client_off))[s].owner);
          if(owner && (uint32_t)owner==(uint32_t)e->ident)mesh_retire(M,owner);
        }
      }
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)listener){
        struct sockaddr_in6 from;socklen_t size=sizeof from;int accepted;
        while((accepted=accept(listener,(struct sockaddr *)&from,&size))>=0){
          struct port *port=port_at(from.sin6_scope_id);size=sizeof from;
          if(!port || port->control>=0 || !port->open){close(accepted);continue;}
          port->control=accepted;port->next=now+MESH_PAIR_NS;
          port_greet(port);
        }
      }
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)hello)hello_heard();
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)network)network_heard();
      else if(e->udata<port_count && (int)e->ident==ports[e->udata].control){
        struct port *port=ports+e->udata;
        if(e->filter==EVFILT_WRITE && port->connecting){
          int error=0;socklen_t size=sizeof error;
          if(getsockopt(port->control,SOL_SOCKET,SO_ERROR,&error,&size))error=errno;
          if(error)port_lost(port,error,0);
          else {port->connecting=0;port_greet(port);}
        }
        else if(e->filter==EVFILT_WRITE){port->writing=0;port_flush(port);}
        else control_heard(port);
      }
    }
  }
  char why[96];farewell_text(stop_reason,why,sizeof why);
  say("bridge node %u: stopping (%s)\n",node,why);
  for(uint32_t p=0;p<port_count;p++){
    port_say(ports+p,(struct message){.kind=FAREWELL,.code=stop_reason},NULL,0);
    farewell_flush(ports+p);
    ports[p].parted=1;
    port_lost(ports+p,ESHUTDOWN,1);
  }
  for(uint32_t p=0;p<port_count;p++){
    struct port *port=ports+p;
    dispatch_sync(port->work,^{
      for(struct flow *f=flows;f;f=f->next)if(f->port==port)flow_release(f);
      if(!down_device(&port->device))say("link %u (%s): verbs teardown: %s\n",port->index,port->name,strerror(errno));
    });
  }
  atomic_store(&M->bridge_pid,0);
  mesh_notify(&M->control);
  pthread_join(watcher,NULL);
  munmap(wire.data,wire.length);munmap(M,length);
  return 0;
}
