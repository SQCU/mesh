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
   destination range, the completion word it stores (`input`; its own `sink` where no reader waits on its landing;
   `release` where it is a SEND cell's ready word, the index hand-off, design/bridge-and-ipc.md#progress), its queue
   pair, the frames it holds posted, its invocation and ring, and its flow (`link`: the one whose ring holds it, which a
   lane's completion names, design/bridge-and-ipc.md#lanes). */
struct flow;
struct prepared_receive {
  _Alignas(128) struct ibv_recv_wr request;
  struct ibv_sge span;
  _Atomic uint64_t *input;
  struct ibv_qp *pair;
  uint64_t argument;
  uint32_t frames,invocation,queue,cycle,behind,release;
  _Atomic uint64_t sink;
  struct flow *link;
};
_Static_assert(sizeof(struct prepared_receive)==128 && _Alignof(struct prepared_receive)==128 &&
  offsetof(struct prepared_receive,span)==32 && offsetof(struct prepared_receive,input)==48 &&
  offsetof(struct prepared_receive,link)==104,"M08 M09");
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
   left, and the lane its queue rides where its flow shares its port's lanes (design/bridge-and-ipc.md#lanes). */
struct send_stream { struct mesh_send *cell,*first; struct ibv_send_wr *next; uint32_t remaining,lane; uint64_t cycle; };

#define MESH_PAIRS 16
/* design/bridge-and-ipc.md#protocol: a message, a flow's keyed by its execution's key, its program's two end nodes
   (`from` the sender's end, `to` the far one) and its stripe (#cables: several of one program may share a port), `failed`
   the node where a failure happened, and `bytes` of what follows it: a READY's program, a RECORD's record.  A port's
   LANES tells its lanes' queue pairs (#lanes) */
enum { HELLO=1, READY, PAIRS, POSTED, END, FAIL, FAREWELL, RECORD, LANES };
struct message { uint32_t magic,kind,node,count; uint64_t key,boot; int64_t code; uint32_t block,port,from,to,failed,bytes;
  uint16_t lid,stripe; uint8_t gid[16]; uint32_t pair[MESH_PAIRS][2]; };
#define XMAGIC (0x4d595048u+MESH_VERSION)
/* A READY's program: `hops` nodes of its route from the receiver to the far end, then the transfers the sender sends
   toward the far end, lengths[k] of them on queue k, and its lists' slots, invocations, ring depth and whether it is
   cyclic, the flow's stripe of the program's invocations, `stripe` of `stripes` (design/bridge-and-ipc.md#cables, at
   most MESH_STRIPES), whether the sender's end takes queue pairs of its own for it (`exclusive`, #lanes), and the
   digest of what the sender's program is made from (mesh.h mesh_client `digest`, a relay's its lease's: #protocol);
   the message's count is its queues times its slots */
struct ready { uint32_t hops,slots,invocations,depth,cyclic,stripe,stripes,exclusive,lengths[MESH_QUEUES]; uint64_t digest; };
#define MESH_STRIPES MESH_PORTS
/* design/bridge-and-ipc.md#lanes: a port's lanes, queue pairs every flow on its link that takes none of its own shares
   (a lane a queue k mod MESH_LANES), and the flows a port holds, by their index in its table (a flow's index and the
   index's generation name it to the peer); a lane's header, one frame, names the records the SENDs after it fill, an
   entry a cell (the receiving flow by index and generation, its ring, its chunks and bytes) */
#define MESH_LANES 2
#define MESH_FLOWS 256
#define MESH_HEADERS 8
#define LANE_FRAME 4096u
#define LANE_MAGIC (0x4c414e45u+MESH_VERSION)
struct lane_entry { uint32_t slot,generation,ring,chunks; uint64_t bytes; };
struct lane_header { uint32_t magic,count; uint64_t sequence; struct lane_entry entry[]; };
#define LANE_ENTRIES ((LANE_FRAME-sizeof(struct lane_header))/sizeof(struct lane_entry))
_Static_assert(sizeof(struct lane_entry)==24 && sizeof(struct lane_header)==16 && LANE_ENTRIES>=128,"lane header");
/* a lane completion's wr_id: a discard, a header (its lane), a SEND (the tag, its lane, its gate's frames), else the
   prepared_receive it filled */
#define LANE_DISCARD UINT64_C(1)
#define LANE_HEARD UINT64_C(2)
#define LANE_SENT (UINT64_C(1)<<63)
/* design/bridge-and-ipc.md#flows: a program on one port, over its link toward the program's far end (`near` its other
   end): its lists each way, a slot's destination (its client's, or on a relay a lease of the bridge's own; a flow held
   for its client, `hold`, has none until the client's program takes it), its `queues` queues of `slots` rings each,
   the program's invocations t with t mod `stripes` = `stripe` (#cables), and on a relay its partner, the route's other
   flow.  An `exclusive` flow holds a queue pair a ring (`qps`, its own completion queues); any other rides its port's
   lanes (#lanes), its index in the port's table and its generation (`slot`), the peer's (`peer_slot`), and its records
   posted on the lanes and not landed (`lane_frames`, in the lanes' `lane_epoch`).  Its verbs work runs on its port's
   queue, one job at a time (`job`, `done`); the event loop alone changes the rest, and a side whose lists arrive while
   a job runs (`pending`) between jobs.  The port's two progress threads serve it while it is in the port's table:
   its SENDs (and an exclusive flow's receives) while `progressing`, a lane's headers fill its rings while `receiving`.
   A held flow keeps the digest its READY named (`digest`), which the program that takes it must name too (#protocol) */
enum { READIED, PAIRING, POSTING, RUNNING, FAILED };
enum { HEARD_READY=1, HEARD_PAIRS=2, HEARD_POSTED=4 };
enum { JOB_PAIR=1, JOB_RECEIVE, JOB_RUN, JOB_CHANGE };
struct lease { struct mesh_ctx context; uint32_t users; uint64_t shape; };
struct side { uint64_t owner; uint32_t destination; struct lease *lease; };
struct port;
struct flow {
  struct flow *next,*partner;
  uint32_t stream_count,near,far,queues,stripe,stripes,heard,configured,destination,invocations;
  uint32_t slot,generation,peer_slot,peer_generation,lane_epoch;
  _Atomic int progressing,receiving,done,stopped;
  _Atomic int64_t failed;
  _Atomic uint32_t domain;
  _Atomic uint64_t lane_frames;
  int state,job,result,stopping,hold,qps,cyclic,told,listed,exclusive,decided;
  uint64_t key,owner,created,deadline,retry,digest;
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
  struct mesh_cancellation *cancel;
};
/* design/bridge-and-ipc.md#progress: a progress thread with nothing moved for MESH_IDLE_NS parks on its port's bell
   (mesh.h struct hdr `bell`, which a client rings as a group of a program on the port starts, mesh-call.h
   mesh_transfers_wake), looking again every MESH_IDLE_POLL_NS while a flow is in its port's table or its lanes have
   work in flight, else every MESH_PAUSE_NS */
#define MESH_IDLE_NS UINT64_C(100000000)
#define MESH_IDLE_POLL_NS UINT64_C(1000000)
/* design/recovery.md#registration: a refused registration's first pause and its longest (doubling) */
#define MESH_REFUSED_NS UINT64_C(60000000000)
#define MESH_REFUSED_MOST_NS UINT64_C(1800000000000)
/* design/bridge-and-ipc.md#protocol: the most bytes a port holds unsent toward a peer that takes none */
#define MESH_UNSENT (UINT64_C(64)<<20)
/* design/bridge-and-ipc.md#lanes: a port's lanes: their queue pairs and completion queues, their state (LANES_DOWN,
   made and told, up, closing), the peer's queue pairs and the frames a header's records may hold (`room`), their pages
   (a lane's MESH_HEADERS send headers, two receive headers and its discard block, leased for every port the region has
   at the bridge's start: lanes_reserve), each lane's gate, the next header each way and the gate's frames once each
   send header's SEND posted (`mark`), the receives a header posted and not landed (`records`, the receive thread's), when
   lanes that failed are made again (`retry`, a pause after; design/recovery.md#loss) and why they last could not be
   made (`code`, logged once a cause), and `generation` moved at each teardown; the progress threads read the lanes
   while `up` */
enum { LANES_DOWN, LANES_MAKING, LANES_MADE, LANES_CONNECTING, LANES_UP, LANES_CLOSING };
struct lanes {
  struct mesh_verbs provider;
  int state,job,result,heard,committed;
  _Atomic int up,done;
  _Atomic int64_t failed;
  uint32_t peer[MESH_LANES][2],peer_room,room,page,pages,generation;
  struct send_gate gate[MESH_LANES];
  uint64_t sent[MESH_LANES],received[MESH_LANES],mark[MESH_LANES][MESH_HEADERS],records;
  uint32_t cursor[MESH_LANES];
  uint64_t retry;
  int64_t code;
};
/* design/bridge-and-ipc.md#ports: a port, its link, its control socket's queues each way, its verbs work's queue, a
   refused registration's pause (`backoff`), the queue pairs its flows and lanes hold or will (`committed`, of the
   device's usable queue pairs), its lanes, its table of flows (`table`, the indices reserved and their generations, the
   highest reserved plus one), and its progress threads (#progress: the epochs each has seen of `epoch`, which a flow
   leaving the table moves, and whether each is parked on the port's bell) */
struct port {
  struct mesh_device device;
  uint32_t index,interface,peer,peer_port,greeted,parted,connecting,writing,retiring,open,flows,committed;
  int64_t farewell;
  char name[IFNAMSIZ];
  int control,job,result,refused,threads;
  _Atomic int done,closing;
  uint64_t next,bandwidth,backoff;
  struct message heard;
  uint16_t lid;
  uint8_t gid[16];
  unsigned char *in,*out;
  size_t held,in_room,out_length,out_sent,out_room;
  dispatch_queue_t work;
  struct lanes lanes;
  struct flow *_Atomic table[MESH_FLOWS];
  uint64_t reserved[MESH_FLOWS/64];
  uint32_t generation[MESH_FLOWS];
  _Atomic uint32_t high;
  _Atomic uint64_t epoch,seen[2];
  _Atomic int parked[2];
  pthread_t thread[2];
};
static struct hdr *M;
static struct wire_mapping wire;
static struct port ports[MESH_PORTS];
static uint32_t port_count;
static struct flow *flows;
/* design/bridge-and-ipc.md#region: the flows and awaiting programs the bridge holds, told in the region (`busy`), and
   the wall clock at which it last came to hold none (`idle_since`, 0 while it holds any) */
static uint32_t flow_count,await_count;
static void busy_tell(void){
  const uint64_t busy=(uint64_t)flow_count+await_count;
  if(busy)atomic_store_explicit(&M->idle_since,0,memory_order_relaxed);
  else if(!atomic_load_explicit(&M->idle_since,memory_order_relaxed))atomic_store_explicit(&M->idle_since,clock_gettime_nsec_np(CLOCK_REALTIME),memory_order_relaxed);
  atomic_store_explicit(&M->busy,busy,memory_order_release);
}
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
/* design/recovery.md#diagnosis: whether a power controller's record (`length` bytes of the ring kern.shutdownreason
   holds) is of `class`, the text before its comma */
static int record_is(const char *record,size_t length,const char *class){
  const size_t n=strlen(class);
  return length>=n && !strncmp(record,class,n) && (length==n || record[n]==',');
}
/* design/recovery.md#diagnosis: why the node last restarted, from the power controller's ring (newest first): the
   reset's own records, the ring's from its newest up to the next that opens an earlier boot's (a `por,por_vddrtc` or
   `wdog` record after the first), written whole into `out` (as many as its room holds, none cut); the watchdog where
   the newest is `wdog`, a sag where any of the reset's own records is `uv` (the controller watched its rails fall: an
   unplug or a supply under lockout), the supply cut where the newest is `por`, else other (none: unknown) */
static uint32_t reset_read(const char *ring,char *out,size_t room){
  uint32_t reset=MESH_RESET_UNKNOWN,records=0,sagged=0;
  size_t used=0;
  int whole=1;
  out[0]=0;
  for(const char *at=ring;;at+=strcspn(at," ")){
    at+=strspn(at," ");
    const size_t length=strcspn(at," ");
    if(!length || (records && (record_is(at,length,"wdog") || (length>=14 && !strncmp(at,"por,por_vddrtc",14)))))break;
    if(!records++)reset=record_is(at,length,"wdog")?MESH_RESET_WATCHDOG:record_is(at,length,"por")?MESH_RESET_SUPPLY:MESH_RESET_OTHER;
    sagged|=record_is(at,length,"uv");
    if(whole && used+(used?1:0)+length<room){if(used)out[used++]=' ';memcpy(out+used,at,length);used+=length;out[used]=0;}
    else whole=0;
  }
  return sagged && reset!=MESH_RESET_WATCHDOG?MESH_RESET_SAG:reset;
}
/* the boot's facts, read once: when the node booted, its boot session, and why it last restarted (reset_read) */
static void health_init(void){
  host_port=mach_host_self();
  struct timeval booted;size_t size=sizeof booted;
  if(!sysctlbyname("kern.boottime",&booted,&size,NULL,0))health.booted=(uint64_t)booted.tv_sec*1000000000u+(uint64_t)booted.tv_usec*1000u;
  sysctl_text("kern.bootsessionuuid",health.session,sizeof health.session);
  char reason[512];sysctl_text("kern.shutdownreason",reason,sizeof reason);
  health.reset=reset_read(reason,health.shutdown,sizeof health.shutdown);
  notify_register_check(kIOPSNotifyAnyPowerSource,&power_token);
  notify_register_check("com.apple.system.thermalpressurelevel",&thermal_token);
  power_read(&health);
  health_sample(&health);
}
static void health_text(const struct mesh_health *h,char *out,size_t room){
  static const char *resets[]={"unknown","supply cut","watchdog","other","supply sagged"};
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
           h->reset<5?resets[h->reset]:"unknown",h->shutdown);
}

/* design/recovery.md#diagnosis; design/bridge-and-ipc.md#records: the nodes this bridge has heard of, when it last heard
   each one's record (this node's clock), when it lost each, the boot session it last knew, so that a node's return
   names a new boot, and the newest record it ever heard of each (its boot and sequence), held apart from the map:
   RFC 2328 s13's database, so a record the map dropped is never taken, nor sent on, again.  An entry unheard for
   MESH_RECORD_AGE_NS whose node the map no longer holds compares with nothing (RFC 2328 s14's MaxAge); a full table
   gives the longest unheard of those its place */
struct known { uint32_t node; uint64_t heard,gone,boot,sequence; char session[40]; };
#define MESH_KNOWN 256
#define MESH_RECORD_AGE_NS (UINT64_C(6)*MESH_HEALTH_NS)
static struct known known[MESH_KNOWN];
static uint32_t known_count;
static struct mesh_record *record_held(uint32_t node);
static struct known *known_at(uint32_t node){
  for(uint32_t i=0;i<known_count;i++)if(known[i].node==node)return known+i;
  struct known *at=NULL;
  if(known_count<MESH_KNOWN)at=known+known_count++;
  else for(uint32_t i=0;i<known_count;i++)if(!record_held(known[i].node) && (!at || known[i].heard<at->heard))at=known+i;
  if(at)*at=(struct known){.node=node};
  return at;
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
static uint32_t side_channel(struct flow *f,int d,uint32_t k){ return f->side[d].destination*MESH_QUEUES+k; }
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

/* design/prepared-machine.md#M07, #M08, #M12: the word a landed chunk of invocation u of the transfer's slot `index` (0
   where each slot has its own rows) stores: its slot's completion word (mesh_host_inputs, the Metal transport), where
   the row's delivery names a word among the receiving program's own toward the flow's node (`words`, its cancellation
   range, [offset, offset + 8 count): a delivery another program wrote on the row, which outlives that program, is
   none of this one's); or, on a row the receiving program (`owner`) also publishes (a relay's rows), that slot's
   publication's SEND cell once the record's last chunk landed, so the landing releases it to the next hop (the index
   hand-off, *release set); else the record's own (`sink`), which nothing reads.  NULL where the word is not in the
   window (#M09) */
static _Atomic uint64_t *receive_input(struct hdr *m,uint64_t owner,const struct mesh_cancel_range *words,uint32_t row,uint32_t index,
                                       uint32_t chunk,uint32_t chunks,uint32_t u,_Atomic uint64_t *sink,uint32_t *release){
  const struct mesh_delivery *delivery=mesh_delivery_at(m,row+chunk);
  uint64_t at=0;
  const uint64_t word=delivery->device_input+delivery->device_stride*u+sizeof(uint64_t)*(uint64_t)index*chunks;
  *release=0;
  if(delivery->device_input && words && word>=words->offset && (word-words->offset)/sizeof(uint64_t)<words->count)at=word;
  else if(chunk+1==chunks){
    uint32_t sends;
    const struct mesh_target *target=mesh_row_targets(m,owner,row,&sends);
    if(index<sends){at=target[index].cells+sizeof(struct mesh_send)*(uint64_t)u;*release=1;}
  }
  if(!at)return sink;
  return at>=m->data_off && at<=m->length-sizeof(uint64_t)?(void *)((char *)m+at):NULL;
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

/* design/bridge-and-ipc.md#cables: whether the flow carries the program's invocation t, and how many of a transfer's
   active invocations [begin, begin + active) it carries */
static int stripe_of(const struct flow *link,uint32_t t){ return t%link->stripes==link->stripe; }
static uint32_t stripe_count(const struct flow *link,uint32_t begin,uint32_t active){
  uint32_t first=(link->stripe+link->stripes-begin%link->stripes)%link->stripes;
  return first<active?(active-first+link->stripes-1)/link->stripes:0;
}

/* design/bridge-and-ipc.md#lanes: the lane a flow's ring q (queue q / slots) rides, and that lane's queue pair */
static uint32_t ring_lane(uint32_t q,uint32_t slots){ return (q/(slots?slots:1))%MESH_LANES; }
static struct ibv_qp *lane_pair(const struct flow *link,uint32_t lane);

/* design/algorithm-sources.md#programcopy */
/* design/prepared-machine.md#M04, #M06, #M29 */
/* The SEND side: every SEND cell of the flow's invocations (#cables), its chain of chunk requests over its invocation's
   ring slot and its stream successor, a stream a ring (the cells' layout is mesh_transfers_prepare's), and an exclusive
   flow's gate a queue pair (a flow on its port's lanes posts through the lanes' gates, #lanes).  A stream's cells run
   invocation by invocation, the publications of each in binding order, as the peer's ring posts its receives: TN3205's
   credit flow control holds a SEND until its matching receive is posted (metal-microbench docs/kernels.md Transport
   (git 7300d1e^)).  0, or the error (EINVAL: a transfer of more slots than the program's). */
static int sends_configure(struct flow *link){
  struct hdr *m=link->M;struct mesh_client *client=mesh_client_at(m,link->side[MESH_SEND].owner);
  struct mesh_tx *tx=side_tx(link,MESH_SEND);
  uint32_t invocations=tx->invocations?tx->invocations:1,fallback=atomic_load_explicit(&client->depth,memory_order_acquire);
  int cyclic=atomic_load_explicit(&client->cyclic,memory_order_acquire)!=0,qps=link->qps;
  size_t requests=0;
  for(uint32_t k=0;k<link->queues;k++){
    struct mesh_transfer *out=side_transfers(link,MESH_SEND,k);
    for(uint32_t i=0;i<side_length(link,MESH_SEND,k);i++){
      if(out[i].count>tx->slots || (uint64_t)k*tx->slots+out[i].count>(uint64_t)qps)return EINVAL;
      requests+=(size_t)(mesh_row_chunks(m,out[i].local_row,out[i].bytes)-1)*out[i].count*
        stripe_count(link,out[i].begin,mesh_transfer_active(out+i,invocations));
    }
  }
  link->publications=(void *)((char *)m+tx->cells);
  /* design/prepared-machine.md#M29 */
  link->requests=aligned_alloc(128,(requests+1)*sizeof *link->requests);
  link->streams=calloc((size_t)qps,sizeof *link->streams);
  link->gates=link->exclusive?aligned_alloc(64,(size_t)qps*sizeof *link->gates):NULL;
  struct mesh_send **last=calloc((size_t)qps,sizeof *last),**terminal=calloc((size_t)qps,sizeof *terminal);
  if(!link->requests||!link->streams||(link->exclusive && !link->gates)||!last||!terminal){free(last);free(terminal);return ENOMEM;}
  link->request_count=requests;
  /* a request is one frame at least, so a queue pair's frames also bound the completions it can have pending in the
     SEND completion queue its pairs share */
  if(link->exclusive){
    uint32_t sends=(uint32_t)link->provider.sent->cqe/(uint32_t)qps;
    for(int q=0;q<qps;q++)link->gates[q]=(struct send_gate){.capacity=MIN(link->provider.queues[q].send_capacity,sends)};
  }
  size_t next=0;int error=0;
  for(uint32_t k=0;k<link->queues && !error;k++){
    uint32_t count=side_length(link,MESH_SEND,k);
    struct mesh_transfer *out=side_transfers(link,MESH_SEND,k);
    /* each cell's chain over its invocation's ring slot */
    for(uint32_t i=0;i<count && !error;i++){
      uint32_t active=mesh_transfer_active(out+i,invocations),column=active+1;
      for(uint32_t slot=0;slot<out[i].count && !error;slot++){
        uint32_t row=out[i].local_row+slot*out[i].stride,chunks=mesh_row_chunks(m,row,out[i].bytes),stream=k*tx->slots+slot;
        struct ibv_qp *pair=link->exclusive?link->provider.queues[stream].pair:lane_pair(link,ring_lane(stream,tx->slots));
        struct mesh_send *cells=link->publications+out[i].first+(size_t)slot*column;
        for(uint32_t u=0;u<active && !error;u++){
          if(!stripe_of(link,out[i].begin+u))continue;
          struct mesh_send *cell=cells+u;
          cell->pair=(uintptr_t)pair;cell->queue=stream;cell->chunks=chunks;
          atomic_store_explicit(&cell->port,link->port->index+1,memory_order_release);
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
    for(uint32_t t=0;t<invocations && !error;t++)for(uint32_t i=0;i<count && stripe_of(link,t);i++){
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
  for(int q=0;q<qps;q++){
    link->streams[q].first=link->streams[q].cell;link->streams[q].cycle=0;
    link->streams[q].lane=ring_lane((uint32_t)q,tx->slots);
  }
  link->stream_count=0;
  for(int q=0;q<qps;q++)if(link->streams[q].cell)link->streams[link->stream_count++]=link->streams[q];
  free(last);free(terminal);
  return error;
}

/* design/prepared-machine.md#M08 */
/* The receive side: each ring's records invocation by invocation (the flow's invocations, #cables), the transfers of
   each in binding order, a transfer's chunks in order, each storing its word (receive_input).  An exclusive flow's
   ring is its queue pair's, its first records posted at once and each record posted once the record of its transfer
   `span` of the flow's own invocations before has landed (the ring's span over the program's invocations, divided
   among the stripes; a cyclic ring's first records wait on its last of the cycle before), so a ring's posted frames
   stay within its queue pair's.  A flow on its port's lanes posts nothing here: a lane's header posts its records as
   the peer sends them (#lanes).  0, or the error (EINVAL: a transfer of more slots than the program's, a word outside
   the window). */
static int receives_configure(struct flow *link){
  struct hdr *m=link->M;struct mesh_client *client=mesh_client_at(m,link->side[MESH_RECEIVE].owner);
  /* design/prepared-machine.md#M09, #M10: the program's targets lie in the window, as each word it stores does */
  if(client->target_count && (client->targets<m->data_off || client->targets>m->length ||
     client->target_count>(m->length-client->targets)/sizeof(struct mesh_target)))return EINVAL;
  struct mesh_tx *tx=side_tx(link,MESH_RECEIVE);
  /* design/prepared-machine.md#M12: the program's own words toward the flow's node, where it has any */
  const uint64_t range=sizeof(struct mesh_cancellation)+sizeof(struct mesh_cancel_range)*((uint64_t)link->side[MESH_RECEIVE].destination+1);
  const struct mesh_cancellation *cancel=tx->cancel>=m->data_off && tx->cancel<=m->length-range?(const void *)((char *)m+tx->cancel):NULL;
  const struct mesh_cancel_range *words=cancel?cancel->ranges+link->side[MESH_RECEIVE].destination:NULL;
  uint32_t invocations=tx->invocations?tx->invocations:1,fallback=atomic_load_explicit(&client->depth,memory_order_acquire),depth=fallback;
  if(!depth||depth>invocations)depth=invocations;
  link->cyclic=atomic_load_explicit(&client->cyclic,memory_order_acquire)!=0;link->invocations=invocations;
  int qps=link->qps;
  size_t records=0;
  for(uint32_t k=0;k<link->queues;k++){
    struct mesh_transfer *in=side_transfers(link,MESH_RECEIVE,k);
    for(uint32_t i=0;i<side_length(link,MESH_RECEIVE,k);i++){
      if(in[i].count>tx->slots || (uint64_t)k*tx->slots+in[i].count>(uint64_t)qps)return EINVAL;
      records+=(uint64_t)mesh_row_chunks(m,in[i].local_row,in[i].bytes)*in[i].count*stripe_count(link,in[i].begin,mesh_transfer_active(in+i,invocations));
    }
  }
  link->receive=aligned_alloc(128,(records+1)*sizeof *link->receive);
  link->rings=calloc((size_t)qps,sizeof *link->rings);
  if(!link->receive||!link->rings)return ENOMEM;
  link->receive_count=records;
  /* a record is one frame at least, so a queue pair's frames also bound the completions it can have pending in the
     receive completion queue its pairs share */
  const uint32_t receives=link->exclusive?(uint32_t)link->provider.completion->cqe/(uint32_t)qps:UINT32_MAX;
  for(int q=0;q<qps;q++)link->rings[q]=(struct receive_ring){.completed=-1,.span=UINT32_MAX,
    .capacity=link->exclusive?MIN(link->provider.queues[q].receive_capacity,receives):UINT32_MAX};
  size_t at=0;int error=0;
  for(int ring=0;ring<qps && !error;ring++){
    uint32_t k=(uint32_t)ring/tx->slots,slot=(uint32_t)ring%tx->slots,count=k<link->queues?side_length(link,MESH_RECEIVE,k):0;
    struct mesh_transfer *in=count?side_transfers(link,MESH_RECEIVE,k):NULL;
    struct ibv_qp *pair=link->exclusive?link->provider.queues[ring].pair:lane_pair(link,ring_lane((uint32_t)ring,tx->slots));
    struct receive_ring *r=link->rings+ring;
    r->first=link->receive+at;
    /* firsts: the ring index of each transfer's own records, own invocation by own invocation (base[i] the first's) */
    size_t *base=calloc((size_t)count+1,sizeof *base),*firsts=NULL;
    for(uint32_t i=0;base&&i<count;i++)
      base[i+1]=base[i]+(slot<in[i].count?stripe_count(link,in[i].begin,mesh_transfer_active(in+i,invocations)):0);
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
    for(uint32_t t=0;t<invocations && !error;t++)for(uint32_t i=0;i<count && !error && stripe_of(link,t);i++){
      uint32_t active=mesh_transfer_active(in+i,invocations);
      if(slot>=in[i].count || t<in[i].begin || t-in[i].begin>=active)continue;
      uint32_t u=t-in[i].begin,row=in[i].local_row+slot*in[i].stride,chunks=mesh_row_chunks(m,row,in[i].bytes);
      const uint32_t first=(link->stripe+link->stripes-in[i].begin%link->stripes)%link->stripes,mine=(u-first)/link->stripes;
      uint64_t remaining=in[i].bytes;
      for(uint32_t c=0;c<chunks;c++){
        struct ibv_sge span;
        if((error=chunk_span(link,transfer_row(m,fallback,in+i,row,t,c),remaining,&span)))break;
        remaining-=span.length;
        struct prepared_receive *record=link->receive+at++;
        if(!c)firsts[base[i]+mine]=(size_t)(record-r->first);
        uint32_t release;
        *record=(struct prepared_receive){.request={.wr_id=(uintptr_t)record,.sg_list=&record->span,.num_sge=1},
          .span=span,.input=receive_input(m,link->side[MESH_RECEIVE].owner,words,row,in[i].stride?0:slot,c,chunks,u,&record->sink,&release),
          .pair=pair,.argument=1,.frames=(span.length+4095)/4096,.invocation=t,.queue=(uint32_t)ring,.link=link};
        record->release=release;
        if(!record->input){error=EINVAL;break;}
      }
    }
    r->count=(size_t)(link->receive+at-r->first);
    /* design/prepared-machine.md#M30 */
    if(link->cyclic)r->span=MIN(r->span,invocations>1?invocations-1:1);
    /* #cables: the span in the flow's own invocations, a stripe's share of the program's (one at least) */
    const uint32_t span=r->span==UINT32_MAX?UINT32_MAX:MAX(r->span/link->stripes,1u);
    for(uint32_t i=0;i<count && !error;i++){
      uint32_t own=(uint32_t)(base[i+1]-base[i]);
      if(!own)continue;
      uint32_t chunks=mesh_row_chunks(m,in[i].local_row+slot*in[i].stride,in[i].bytes);
      for(uint32_t j=0;j<own;j++){
        size_t here=firsts[base[i]+j],behind=0;
        if(span!=UINT32_MAX && j>=span)behind=here-firsts[base[i]+j-span];
        else if(span!=UINT32_MAX && link->cyclic && j+own>=span)behind=here+r->count-firsts[base[i]+j+own-span];
        else if(span!=UINT32_MAX && link->cyclic)behind=r->count;
        for(uint32_t c=0;c<chunks;c++)r->first[here+c].behind=(uint32_t)behind;
      }
    }
    r->span=span;
    free(base);free(firsts);
    if(!error && link->exclusive)error=ring_advance(link,r);
  }
  if(at)link->receive[at]=link->receive[at-1];
  return error;
}

/* design/bridge-and-ipc.md#progress: a landing that released a SEND cell (the index hand-off of a relay, or of a member
   that sends on what it receives) wakes the send thread that posts the cell, where that thread is parked: the fence
   orders the release before the read of its parked word, as the thread orders its parked word before its last look
   (port_progress), so one of the two sees the other */
static void send_wake(_Atomic uint64_t *ready){
  const uint32_t p=atomic_load_explicit(&((struct mesh_send *)(void *)ready)->port,memory_order_acquire);
  atomic_thread_fence(memory_order_seq_cst);
  if(p && p<=MESH_PORTS && atomic_load_explicit(&ports[p-1].parked[1],memory_order_relaxed))mesh_notify(&M->bell[p-1]);
}

/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M06 */
/* design/prepared-machine.md#M15 */
/* design/algorithm-sources.md#independent-native-queues */
/* The flow's SEND completions: each retires its queue's frames through its request (the device completes every SEND,
   signaled or not); *taken their count.  0, or an error completion's status. */
static __attribute__((always_inline)) inline int send_retire(struct flow *link,int *taken){
  struct ibv_wc done[16];
  int count=link->provider.queues[0].poll(link->provider.sent,16,done);
  *taken=count>0?count:0;
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
/* A running flow's SENDs, one pass of its port's send thread (design/bridge-and-ipc.md#progress): each stream's cells
   in order, a cell released by its producer posting its chain's requests while its queue pair holds their frames (the
   gate: frames posted less those its SEND completions retired; ibv_post_send accepts past the queue's frames and
   corrupts later, so the gate is the caller's), then its SEND completions taken.  Whether anything moved; a failure is
   the flow's (flow_lost). */
static int flow_send(struct flow *link){
  int moved=0,taken=0;
  struct ibv_send_wr *bad;
  int (*post)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **)=link->provider.queues[0].send;
  for(uint32_t s=0;s<link->stream_count;s++){
    struct send_stream *stream=link->streams+s;
    for(;;){
      if(!stream->remaining){
        struct mesh_send *cell=stream->cell;
        if(atomic_load_explicit(&cell->ready,memory_order_acquire)<=stream->cycle)break;
        stream->next=&cell->request;stream->remaining=cell->chunks;
      }
      struct mesh_send *cell=stream->cell;
      struct send_gate *gate=link->gates+cell->queue;
      while(stream->remaining){
        struct ibv_send_wr *request=stream->next;
        uint32_t frames=(request->sg_list->length+4095)/4096;
        if(((gate->posted-gate->retired)&MESH_GATE_MASK)+frames>gate->capacity)break;
        gate->posted+=frames;
        /* posted as a copy: the cell's first request keeps the producer's argument in its wr_id (mesh_transfers_prepare),
           which a host producer reads at every publication (mesh_host_publish), and the copy's names the gate */
        struct ibv_send_wr posting=*request;
        posting.next=NULL;posting.send_flags=IBV_SEND_SIGNALED;
        posting.wr_id=((uint64_t)cell->queue<<48)|(gate->posted&MESH_GATE_MASK);
        int error=post(link->provider.queues[cell->queue].pair,&posting,&bad);
        if(error){flow_lost(link,error<0?-error:error,1);return 1;}
        stream->next=request->next;stream->remaining--;gate->requests++;moved=1;
      }
      if(stream->remaining)break;
      stream->cell=(struct mesh_send *)cell->successor;
      if(stream->cell==stream->first)stream->cycle++;
    }
  }
  if(link->stream_count){
    int error=send_retire(link,&taken);
    if(error){flow_lost(link,error,2);return 1;}
  }
  return moved || taken;
}

/* design/prepared-machine.md#M08 */
/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#programcopy */
/* A running flow's receive completions, one pass of its port's receive thread: each stores its completion word and
   posts the ring's next records.  Whether anything landed; a failure is the flow's (flow_lost). */
static int flow_receive(struct flow *link){
  if(!link->rings)return 0;
  struct ibv_wc done[16];
  const struct mesh_queue *queue=link->provider.queues;
  int count=queue->poll(queue->completion,16,done);
  if(count<0){flow_lost(link,EIO,3);return 1;}
  for(int i=0;i<count;i++){
    uint64_t status_opcode;
    memcpy(&status_opcode,&done[i].status,sizeof status_opcode);
    if((uint32_t)status_opcode){flow_lost(link,(uint32_t)status_opcode,2);return 1;}
    if(!((status_opcode>>32)&IBV_WC_RECV))continue;
    struct prepared_receive *record=(void *)(uintptr_t)done[i].wr_id;
    struct receive_ring *ring=link->rings+record->queue;
    ring->landed++;ring->bytes+=done[i].byte_len;
    atomic_store_explicit(record->input,record->argument+record->cycle,memory_order_release);
    if(record->release)send_wake(record->input);
    ring->outstanding-=record->frames;ring->completed=(int64_t)(record->invocation+(uint64_t)record->cycle*link->invocations);
    int error=ring_advance(link,ring);
    if(error){flow_lost(link,error,1);return 1;}
  }
  return count>0;
}

/* design/bridge-and-ipc.md#lanes: a port's lanes' pages: lane l's header frame h (its MESH_HEADERS send headers, then
   its two receive headers) and its discard block, as arena offsets; a span of them on the port's device; and their
   bytes through the window's mapping */
static uint64_t lane_offset(const struct port *port,uint32_t lane,uint32_t frame){
  return (uint64_t)port->lanes.page*M->pgsz+((uint64_t)lane*(MESH_HEADERS+2)+frame)*LANE_FRAME;
}
static uint64_t lane_discard(const struct port *port,uint32_t lane){
  const uint64_t block=(uint64_t)M->block*M->pgsz,headers=(uint64_t)MESH_LANES*(MESH_HEADERS+2)*LANE_FRAME;
  return (uint64_t)port->lanes.page*M->pgsz+(headers+block-1)/block*block+(uint64_t)lane*block;
}
static uint32_t lane_pages(void){
  const uint64_t block=(uint64_t)M->block*M->pgsz,headers=(uint64_t)MESH_LANES*(MESH_HEADERS+2)*LANE_FRAME;
  return (uint32_t)(((headers+block-1)/block*block+MESH_LANES*block+M->pgsz-1)/M->pgsz);
}
static struct ibv_sge lane_span(const struct port *port,uint64_t offset,uint64_t length){
  return wire_span(&port->device,(uint64_t)M->block*M->pgsz,offset,length);
}
static struct ibv_qp *lane_pair(const struct flow *link,uint32_t lane){
  return link->port->lanes.provider.queues?link->port->lanes.provider.queues[lane].pair:NULL;
}
/* a lane's failure, the lanes' (taken by the event loop: lanes_turn, lanes_reset) */
static void lanes_lost(struct port *port,int64_t code){
  int64_t none=0;
  atomic_compare_exchange_strong(&port->lanes.failed,&none,code?code:EIO);
  atomic_store_explicit(&port->lanes.up,0,memory_order_release);
  wake();
}
/* lane l's next header received, into its receive header of the parity of the headers it heard */
static int lane_listen(struct port *port,uint32_t lane){
  struct lanes *l=&port->lanes;
  struct ibv_sge span=lane_span(port,lane_offset(port,lane,MESH_HEADERS+(uint32_t)(l->received[lane]&1)),LANE_FRAME);
  struct ibv_recv_wr request={.wr_id=LANE_HEARD+lane,.sg_list=&span,.num_sge=1},*bad;
  const int error=l->provider.queues[lane].receive(l->provider.queues[lane].pair,&request,&bad);
  return error<0?-error:error;
}

/* design/bridge-and-ipc.md#lanes: a lane's SENDs, one pass of its port's send thread: of the flows on the port's lanes
   that run, each stream of this lane whose cell its producer released gives its chunks, as many as the lane's gate and
   the peer's receive room take (a cell's chunks may span headers), each run of them an entry of one header; the header
   is posted, then every entry's chunks in the header's order, so the peer, which posts the records a header names
   before it listens for the next, has each SEND's receive posted as it lands (credit flow control holds it until then,
   TN3205).  A header frame is used again once its SEND retired.  Whether anything was posted; a failure is the
   lanes' (lanes_lost). */
static int lane_send(struct port *port,uint32_t lane){
  struct lanes *l=&port->lanes;
  struct send_gate *gate=l->gate+lane;
  const uint64_t sent=l->sent[lane];
  const uint32_t h=(uint32_t)(sent%MESH_HEADERS);
  if(sent>=MESH_HEADERS && ((gate->retired-l->mark[lane][h])&MESH_GATE_MASK)>=(UINT64_C(1)<<47))return 0;
  const uint64_t flight=(gate->posted-gate->retired)&MESH_GATE_MASK;
  if(flight+2>gate->capacity)return 0;
  uint64_t room=MIN(gate->capacity-flight-1,(uint64_t)l->peer_room);
  struct lane_header *header=(void *)(wire.data+lane_offset(port,lane,h));
  struct ibv_send_wr *first[LANE_ENTRIES];
  uint32_t entries=0;
  const uint32_t high=atomic_load_explicit(&port->high,memory_order_acquire);
  for(uint32_t n=0;n<high && entries<LANE_ENTRIES && room;n++){
    const uint32_t i=(l->cursor[lane]+n)%high;
    struct flow *f=atomic_load_explicit(&port->table[i],memory_order_acquire);
    if(!f || f->exclusive || !atomic_load_explicit(&f->progressing,memory_order_acquire))continue;
    for(uint32_t s=0;s<f->stream_count && entries<LANE_ENTRIES && room;s++){
      struct send_stream *stream=f->streams+s;
      while(stream->lane==lane && entries<LANE_ENTRIES && room){
        if(!stream->remaining){
          struct mesh_send *cell=stream->cell;
          if(atomic_load_explicit(&cell->ready,memory_order_acquire)<=stream->cycle)break;
          stream->next=&cell->request;stream->remaining=cell->chunks;
        }
        struct ibv_send_wr *request=stream->next;
        uint32_t chunks=0;uint64_t bytes=0;
        for(;chunks<stream->remaining && request;chunks++,request=request->next){
          const uint32_t frames=(request->sg_list->length+4095)/4096;
          if(frames>room)break;
          room-=frames;bytes+=request->sg_list->length;
        }
        if(!chunks){room=0;break;}
        header->entry[entries]=(struct lane_entry){f->peer_slot,f->peer_generation,stream->cell->queue,chunks,bytes};
        first[entries++]=stream->next;
        stream->next=request;stream->remaining-=chunks;
        if(stream->remaining)break;
        stream->cell=(struct mesh_send *)stream->cell->successor;
        if(stream->cell==stream->first)stream->cycle++;
      }
    }
  }
  if(!entries)return 0;
  l->cursor[lane]=(l->cursor[lane]+1)%(high?high:1);
  header->magic=LANE_MAGIC;header->count=entries;header->sequence=sent;
  int (*post)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **)=l->provider.queues[lane].send;
  struct ibv_qp *pair=l->provider.queues[lane].pair;
  struct ibv_sge span=lane_span(port,lane_offset(port,lane,h),LANE_FRAME);
  struct ibv_send_wr posting={.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
  gate->posted+=1;gate->requests++;
  posting.wr_id=LANE_SENT|((uint64_t)lane<<48)|(gate->posted&MESH_GATE_MASK);
  int error=post(pair,&posting,&bad);
  l->mark[lane][h]=gate->posted;l->sent[lane]++;
  for(uint32_t e=0;e<entries && !error;e++){
    struct ibv_send_wr *request=first[e];
    for(uint32_t c=0;c<header->entry[e].chunks && !error;c++,request=request->next){
      posting=*request;
      posting.next=NULL;posting.send_flags=IBV_SEND_SIGNALED;
      gate->posted+=(request->sg_list->length+4095)/4096;gate->requests++;
      posting.wr_id=LANE_SENT|((uint64_t)lane<<48)|(gate->posted&MESH_GATE_MASK);
      error=post(pair,&posting,&bad);
    }
  }
  if(error)lanes_lost(port,error<0?-error:error);
  return 1;
}
/* the lanes' SEND completions, each retiring its lane's frames; whether any was taken */
static int lane_retire(struct port *port){
  struct lanes *l=&port->lanes;
  struct ibv_wc done[16];
  const int count=l->provider.queues[0].poll(l->provider.sent,16,done);
  if(count<0){lanes_lost(port,EIO);return 1;}
  for(int i=0;i<count;i++){
    if(done[i].status){lanes_lost(port,(int64_t)done[i].status);return 1;}
    struct send_gate *gate=l->gate+((done[i].wr_id>>48)&(MESH_LANES-1));
    gate->retired=done[i].wr_id&MESH_GATE_MASK;gate->completions++;
  }
  return count>0;
}
/* design/bridge-and-ipc.md#lanes: lane l's header heard: each entry's records posted, the next `chunks` of the named
   ring of the named flow while that flow is the one the index and generation name and takes its lanes' records, else
   as many discards of the same frames (a flow that stopped: its rows are no longer its own), then the next header
   listened for.  A header out of sequence or past the room the lane told, or an entry whose frames are not its ring's
   records', is a protocol error: the lanes' where the header is, the flow's where its records are.  1 where the lanes
   failed */
static int lane_heard(struct port *port,uint32_t lane){
  struct lanes *l=&port->lanes;
  const struct lane_header *header=(const void *)(wire.data+lane_offset(port,lane,MESH_HEADERS+(uint32_t)(l->received[lane]&1)));
  const uint64_t block=(uint64_t)M->block*M->pgsz;
  uint64_t total=0;
  int error=header->magic!=LANE_MAGIC || header->sequence!=l->received[lane] || header->count>LANE_ENTRIES?EPROTO:0;
  for(uint32_t e=0;!error && e<header->count;e++){
    const struct lane_entry *entry=header->entry+e;
    if(!entry->chunks || entry->bytes<=(uint64_t)(entry->chunks-1)*block || entry->bytes>(uint64_t)entry->chunks*block)error=EPROTO;
    else total+=(uint64_t)(entry->chunks-1)*(block/4096)+(entry->bytes-(uint64_t)(entry->chunks-1)*block+4095)/4096;
  }
  if(!error && total>l->room)error=EPROTO;
  if(error){lanes_lost(port,error);return 1;}
  int (*post)(struct ibv_qp *,struct ibv_recv_wr *,struct ibv_recv_wr **)=l->provider.queues[lane].receive;
  struct ibv_qp *pair=l->provider.queues[lane].pair;
  struct ibv_recv_wr *bad;
  for(uint32_t e=0;e<header->count && !error;e++){
    const struct lane_entry entry=header->entry[e];
    struct flow *f=entry.slot<MESH_FLOWS?atomic_load_explicit(&port->table[entry.slot],memory_order_acquire):NULL;
    struct receive_ring *ring=f && !f->exclusive && f->generation==entry.generation && entry.ring<(uint32_t)f->qps &&
      atomic_load_explicit(&f->receiving,memory_order_acquire) && f->rings && ring_lane(entry.ring,(uint32_t)f->qps/f->queues)==lane?
      f->rings+entry.ring:NULL;
    for(uint32_t c=0;ring && c<entry.chunks;c++){
      const uint64_t length=MIN(block,entry.bytes-(uint64_t)c*block);
      if(!ring->count || ring->first[(ring->posted+c)%ring->count].frames!=(length+4095)/4096){flow_lost(f,EPROTO,2);ring=NULL;}
    }
    for(uint32_t c=0;c<entry.chunks && !error;c++){
      if(ring){
        struct prepared_receive *record=ring->first+ring->posted%ring->count;
        record->cycle=(uint32_t)(ring->posted/ring->count);
        error=post(pair,&record->request,&bad);
        ring->outstanding+=record->frames;ring->posted++;
        atomic_fetch_add_explicit(&f->lane_frames,record->frames,memory_order_relaxed);
      }else{
        struct ibv_sge span=lane_span(port,lane_discard(port,lane),MIN(block,entry.bytes-(uint64_t)c*block));
        struct ibv_recv_wr request={.wr_id=LANE_DISCARD,.sg_list=&span,.num_sge=1};
        error=post(pair,&request,&bad);
      }
      if(!error)l->records++;
    }
  }
  if(!error){l->received[lane]++;error=lane_listen(port,lane);}
  if(error){lanes_lost(port,error<0?-error:error);return 1;}
  return 0;
}
/* design/bridge-and-ipc.md#lanes: the lanes' receive completions, one pass of the port's receive thread: a header's
   records posted (lane_heard), a record's landing storing its word as an exclusive flow's does; whether anything
   landed */
static int lane_receive(struct port *port){
  struct lanes *l=&port->lanes;
  struct ibv_wc done[16];
  const int count=l->provider.queues[0].poll(l->provider.completion,16,done);
  if(count<0){lanes_lost(port,EIO);return 1;}
  for(int i=0;i<count;i++){
    if(done[i].status){lanes_lost(port,(int64_t)done[i].status);return 1;}
    const uint64_t id=done[i].wr_id;
    if(id>=LANE_HEARD && id<LANE_HEARD+MESH_LANES){if(lane_heard(port,(uint32_t)(id-LANE_HEARD)))return 1;continue;}
    l->records--;
    if(id==LANE_DISCARD)continue;
    struct prepared_receive *record=(void *)(uintptr_t)id;
    struct flow *link=record->link;
    struct receive_ring *ring=link->rings+record->queue;
    ring->landed++;ring->bytes+=done[i].byte_len;
    atomic_store_explicit(record->input,record->argument+record->cycle,memory_order_release);
    if(record->release)send_wake(record->input);
    ring->outstanding-=record->frames;ring->completed=(int64_t)(record->invocation+(uint64_t)record->cycle*link->invocations);
    atomic_fetch_sub_explicit(&link->lane_frames,record->frames,memory_order_release);
  }
  return count>0;
}

/* design/bridge-and-ipc.md#progress: one pass of a port's progress thread, `side` 0 its receives' and 1 its SENDs':
   every exclusive flow in the port's table it serves (while the flow is progressing) and the port's lanes while they are
   up.  The pass reads the table's epoch, serves each, and tells the epoch it began at, so a flow taken out of the table
   is read by neither thread once both tell a later epoch (port_unlist).  Whether anything moved; *any where something is
   the thread's to look for: a flow it serves (a SEND side progressing; an exclusive flow's receives, or a lane rider
   taking its headers' records), a lane's records posted and not landed, or its SENDs in flight; nothing lands on a lane
   for a flow this end has not listed, since the peer sends only after this end's POSTED, which follows the listing */
static int port_pass(struct port *port,int side,int *any){
  const uint64_t epoch=atomic_load(&port->epoch);
  int moved=0;
  *any=0;
  const uint32_t high=atomic_load_explicit(&port->high,memory_order_acquire);
  for(uint32_t i=0;i<high;i++){
    struct flow *f=atomic_load_explicit(&port->table[i],memory_order_acquire);
    if(!f)continue;
    const int progressing=atomic_load_explicit(&f->progressing,memory_order_acquire);
    if(f->exclusive && progressing){*any=1;moved|=side?flow_send(f):flow_receive(f);}
    else if(!f->exclusive)*any|=side?progressing:atomic_load_explicit(&f->receiving,memory_order_acquire);
  }
  struct lanes *l=&port->lanes;
  if(atomic_load_explicit(&l->up,memory_order_acquire)){
    if(side){
      for(uint32_t lane=0;lane<MESH_LANES;lane++)moved|=lane_send(port,lane);
      moved|=lane_retire(port);
      for(uint32_t lane=0;lane<MESH_LANES;lane++)*any|=((l->gate[lane].posted-l->gate[lane].retired)&MESH_GATE_MASK)!=0;
    }else{moved|=lane_receive(port);*any|=l->records!=0;}
  }
  atomic_store_explicit(&port->seen[side],epoch,memory_order_release);
  return moved;
}
/* design/bridge-and-ipc.md#progress: a port's progress thread: passes while anything moves, and with nothing moved for
   MESH_IDLE_NS it parks on its port's bell, looking again every MESH_IDLE_POLL_NS while anything is its to look for
   (port_pass), else every MESH_PAUSE_NS, and at once when the bell rings: a client's group of a program on the port
   starting (mesh-call.h mesh_transfers_wake), a landing that released one of the port's SEND cells while its send thread
   was parked (send_wake), a flow listed or unlisted, the lanes' change, the bridge stopping.  A thread tells it is
   parked, then looks once more before it waits, so a release whose send_wake read it unparked is served by that look */
#define MESH_IDLE_PASSES 4096u
struct progress { struct port *port; int side; };
static void *port_progress(void *argument){
  struct port *port=((struct progress *)argument)->port;
  const int side=((struct progress *)argument)->side;
  free(argument);
  pthread_setname_np(side?"mesh.rdma.send":"mesh.rdma.receive");
  _Atomic uint64_t *bell=&M->bell[port->index];
  uint64_t quiet=0;
  uint32_t empty=0;
  int resting=0,any=0;
  while(!atomic_load_explicit(&port->closing,memory_order_acquire)){
    const uint64_t rung=atomic_load_explicit(bell,memory_order_acquire);
    if(port_pass(port,side,&any)){empty=0;quiet=0;resting=0;continue;}
    if(any && !resting){
      if(++empty<MESH_IDLE_PASSES)continue;
      empty=0;
      const uint64_t at=clock_ns();
      if(!quiet)quiet=at;
      if(at-quiet<MESH_IDLE_NS)continue;
    }
    atomic_store_explicit(&port->parked[side],1,memory_order_relaxed);
    atomic_thread_fence(memory_order_seq_cst);
    const uint64_t last=atomic_load_explicit(bell,memory_order_acquire);
    if(port_pass(port,side,&any)){atomic_store_explicit(&port->parked[side],0,memory_order_relaxed);empty=0;quiet=0;resting=0;continue;}
    resting=1;
    os_sync_wait_on_address_with_timeout(bell,last,sizeof *bell,OS_SYNC_WAIT_ON_ADDRESS_SHARED,OS_CLOCK_MACH_ABSOLUTE_TIME,
      any?MESH_IDLE_POLL_NS:MESH_PAUSE_NS);
    atomic_store_explicit(&port->parked[side],0,memory_order_relaxed);
    if(atomic_load_explicit(bell,memory_order_acquire)!=rung){resting=0;quiet=0;empty=0;}
  }
  return NULL;
}
/* the port's bell rung: its parked progress threads look again */
static void port_bell(struct port *port){ mesh_notify(&M->bell[port->index]); }
/* design/bridge-and-ipc.md#progress: the port's two progress threads started, once; 0, or pthread_create's error */
static int port_threads(struct port *port){
  int error=0;
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_set_qos_class_np(&attributes,QOS_CLASS_USER_INTERACTIVE,0);
  while(port->threads<2 && !error){
    struct progress *progress=malloc(sizeof *progress);
    if(!progress){error=ENOMEM;break;}
    *progress=(struct progress){port,port->threads};
    atomic_store(&port->seen[port->threads],atomic_load(&port->epoch));
    if((error=pthread_create(&port->thread[port->threads],&attributes,port_progress,progress)))free(progress);
    else port->threads++;
  }
  pthread_attr_destroy(&attributes);
  return error;
}
/* design/bridge-and-ipc.md#progress: a flow in its port's table at its index, on the port's queue, its SENDs (and an
   exclusive flow's receives) served where `progressing`, a lane's headers filling its rings where `receiving`; 0, or
   the threads' error */
static int port_list(struct flow *f,int progressing,int receiving){
  struct port *port=f->port;
  const int error=port_threads(port);
  if(error)return error;
  if(progressing)atomic_store_explicit(&f->progressing,1,memory_order_release);
  if(receiving)atomic_store_explicit(&f->receiving,1,memory_order_release);
  atomic_store_explicit(&port->table[f->slot],f,memory_order_release);
  f->listed=1;
  port_bell(port);
  return 0;
}
/* the port's table read by neither progress thread past a change made before: both tell an epoch past it */
static void port_settle(struct port *port){
  const uint64_t epoch=atomic_fetch_add(&port->epoch,1)+1;
  for(int side=0;side<port->threads;side++)
    while(atomic_load_explicit(&port->seen[side],memory_order_acquire)<epoch){port_bell(port);sched_yield();}
}
/* design/bridge-and-ipc.md#progress: a flow's SENDs (and an exclusive flow's receives) no longer served, on its port's
   queue, returning once neither thread is serving them (a flow on the lanes keeps taking its headers' records) */
static void port_quiet(struct flow *f){
  atomic_store_explicit(&f->progressing,0,memory_order_release);
  if(f->listed)port_settle(f->port);
}
/* design/bridge-and-ipc.md#progress: a flow out of its port's table, on the port's queue, returning once neither
   progress thread can read it (a lane's records it posted before land in its rows: flow_release waits on them) */
static void port_unlist(struct flow *f){
  atomic_store_explicit(&f->progressing,0,memory_order_release);
  atomic_store_explicit(&f->receiving,0,memory_order_release);
  if(!f->listed)return;
  struct flow *held=f;
  atomic_compare_exchange_strong(&f->port->table[f->slot],&held,NULL);
  port_settle(f->port);
  f->listed=0;
}
/* design/bridge-and-ipc.md#progress: the port's progress threads stopped (the bridge stopping, every flow released) */
static void port_still(struct port *port){
  atomic_store_explicit(&port->closing,1,memory_order_release);
  port_bell(port);
  while(port->threads)pthread_join(port->thread[--port->threads],NULL);
}

/* design/bridge-and-ipc.md#flows: a flow's verbs work, on its port's queue: an exclusive flow's queue pairs made; ready
   to receive, its receives prepared (an exclusive flow's posted, a lane rider's taking its headers' records); its
   SENDs prepared, an exclusive flow's queue pairs ready to send, and served by its port's progress threads; or a side
   whose lists arrived, the flow's SENDs unserved while it is prepared and served again where they were.  0, or the
   error */
static int flow_work(struct flow *f,int job){
  int error=0;
  if(job==JOB_PAIR)return f->exclusive?pair_up(&f->provider,f->qps):0;
  const int ran=atomic_load_explicit(&f->progressing,memory_order_acquire);
  if(job==JOB_CHANGE)port_quiet(f);
  if(job==JOB_RECEIVE && f->exclusive)error=pair_receive(&f->provider,(const uint32_t (*)[2])f->pairs,f->port->heard.lid,f->port->heard.gid);
  if(!error && job!=JOB_RUN && f->side[MESH_RECEIVE].owner && !(f->configured&1u<<MESH_RECEIVE) && !(error=receives_configure(f))){
    f->configured|=1u<<MESH_RECEIVE;
    f->lane_epoch=f->port->lanes.generation;
    if(!f->exclusive)error=port_list(f,0,1);
  }
  if(!error && job!=JOB_RECEIVE && f->side[MESH_SEND].owner && !(f->configured&1u<<MESH_SEND) && !(error=sends_configure(f)))
    f->configured|=1u<<MESH_SEND;
  if(!error && job==JOB_RUN && f->exclusive)error=pair_send(&f->provider);
  if(!error && (job==JOB_RUN || (job==JOB_CHANGE && ran)))error=port_list(f,1,0);
  return error;
}
/* design/bridge-and-ipc.md#region: the queue pairs a port's device holds, told in its port table */
static void port_pairs(struct port *port){
  atomic_store_explicit(&mesh_links(M)[port->index].pairs,port->device.pairs,memory_order_relaxed);
}
static void flow_job(struct flow *f,int job){
  f->job=job;atomic_store_explicit(&f->done,0,memory_order_relaxed);
  dispatch_async(f->port->work,^{
    f->result=flow_work(f,job);
    port_pairs(f->port);
    atomic_store_explicit(&f->done,1,memory_order_release);
    wake();
  });
}

/* design/algorithm-sources.md#meshresult: a flow out of its port's table and its queue pairs destroyed, on its port's
   queue, the link's census printed: each ring's SEND requests and frames posted and retired by the send completions
   its queue pair took (an exclusive flow's), and its records landed and their bytes.  A flow on its port's lanes is
   released once its records posted on the lanes have landed, or the lanes' queue pairs that held them are destroyed
   (their generation moved: lanes that failed and are not yet closed may still land a SEND in its rows, which a relay's
   lease returns to the window as it is freed).  1 once nothing is held (a failed destruction, or records still posted,
   keep it for the next turn) */
static int flow_release(struct flow *flow){
  port_unlist(flow);
  for(int q=0;flow->rings && q<flow->qps;q++)
    say("link %u census queue=%d requests=%llu frames=%llu retired=%llu send_completions=%llu records=%zu posted=%zu landed=%llu bytes=%llu%s\n",
      flow->port->index,q,flow->gates?(unsigned long long)flow->gates[q].requests:0ull,flow->gates?(unsigned long long)flow->gates[q].posted:0ull,
      flow->gates?(unsigned long long)flow->gates[q].retired:0ull,flow->gates?(unsigned long long)flow->gates[q].completions:0ull,
      flow->rings[q].count,flow->rings[q].posted,(unsigned long long)flow->rings[q].landed,(unsigned long long)flow->rings[q].bytes,
      flow->exclusive?"":" (lanes)");
  if(!flow->exclusive && atomic_load_explicit(&flow->lane_frames,memory_order_acquire) &&
     flow->port->lanes.generation==flow->lane_epoch)return 0;
  if(!down_pair(&flow->provider))return 0;
  free(flow->receive);free(flow->streams);free(flow->gates);free(flow->rings);free(flow->requests);
  flow->receive=NULL;flow->streams=NULL;flow->gates=NULL;flow->rings=NULL;flow->requests=NULL;
  return 1;
}
/* A flow stopped on its port's queue, after its job: where it failed, its program's pending words toward the far end
   cancelled (M12) once its port's progress threads no longer read it, and its status the failure naming the node
   `where` */
static void flow_stop(struct flow *f,int64_t code,uint32_t domain,uint32_t where){
  const uint64_t owner=f->owner;const uint32_t destination=f->destination;struct mesh_cancellation *cancel=f->cancel;
  f->stopping=1;atomic_store_explicit(&f->stopped,0,memory_order_relaxed);
  dispatch_async(f->port->work,^{
    port_unlist(f);
    if(code && owned(owner)){
      if(cancel)mesh_cancel(M,cancel,destination);
      status_failed(&mesh_client_at(M,owner)->status[destination],owner,code,domain,where);
    }
    atomic_store_explicit(&f->stopped,flow_release(f)?1:2,memory_order_release);
    port_pairs(f->port);
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
  if(length-port->out_sent>MESH_UNSENT){port_lost(port,ENOBUFS,0);return;}
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
  struct message said={.kind=kind,.count=count,.key=f->key,.code=code,.from=f->near,.to=f->far,.failed=origin,.stripe=(uint16_t)f->stripe};
  for(uint32_t q=0;pair && q<count && q<MESH_PAIRS;q++){said.pair[q][0]=pair[q][0];said.pair[q][1]=pair[q][1];}
  port_say(f->port,said,NULL,0);
}
/* a READY: the flow's queue pairs, then its route's `hops` nodes and the transfers its SEND side sends, with its
   program's digest */
static void ready_say(struct flow *f,const uint32_t *route,uint32_t hops){
  int d=f->side[MESH_SEND].owner?MESH_SEND:MESH_RECEIVE;
  struct mesh_client *client=mesh_client_at(M,f->side[d].owner);
  struct mesh_tx *tx=side_tx(f,d);
  struct ready ready={.hops=hops,.slots=tx->slots,.invocations=tx->invocations,.depth=atomic_load(&client->depth),
    .cyclic=atomic_load(&client->cyclic),.stripe=f->stripe,.stripes=f->stripes,.exclusive=(uint32_t)f->exclusive,
    .digest=atomic_load_explicit(&client->digest,memory_order_relaxed)};
  size_t transfers=0;
  for(uint32_t k=0;k<f->queues;k++)transfers+=ready.lengths[k]=side_length(f,MESH_SEND,k);
  size_t bytes=sizeof ready+(size_t)hops*sizeof *route+transfers*sizeof(struct mesh_transfer);
  unsigned char *payload=malloc(bytes),*at=payload;
  if(!payload){port_lost(f->port,ENOMEM,0);return;}
  memcpy(at,&ready,sizeof ready);at+=sizeof ready;
  memcpy(at,route,(size_t)hops*sizeof *route);at+=(size_t)hops*sizeof *route;
  for(uint32_t k=0;k<f->queues;k++){memcpy(at,side_transfers(f,MESH_SEND,k),ready.lengths[k]*sizeof(struct mesh_transfer));at+=ready.lengths[k]*sizeof(struct mesh_transfer);}
  port_say(f->port,(struct message){.kind=READY,.count=(uint32_t)f->qps,.key=f->key,.from=f->near,.to=f->far,.stripe=(uint16_t)f->stripe},
           payload,(uint32_t)bytes);
  free(payload);
}

/* design/bridge-and-ipc.md#routes: a client program awaiting the READYs of its routed peer, the pair's lower node, whose
   route both ends take: its `stripes` once the first came (#cables), the stripes it has taken (a bit each), and its
   `deadline`: the peer window from `created` where this end's map routes to the peer (`routed`), else the pairing
   window from when the peer was last reachable (program_await) */
struct await { struct await *next; uint64_t key,owner,created,taken,deadline; uint32_t destination,far,stripes,routed; };
static struct await *awaits;
static struct await *await_find(uint64_t key,uint32_t far){
  for(struct await *a=awaits;a;a=a->next)if(a->key==key && a->far==far)return a;
  return NULL;
}
static struct await *await_of(uint64_t owner,uint32_t d){
  for(struct await *a=awaits;a;a=a->next)if(a->owner==owner && a->destination==d)return a;
  return NULL;
}
static void await_drop(struct await *a){
  for(struct await **at=&awaits;*at;at=&(*at)->next)if(*at==a){*at=a->next;await_count--;break;}
  free(a);
  busy_tell();
}
/* an await's stripe taken: dropped once every stripe is (1 then) */
static int await_took(struct await *a,uint32_t stripe,uint32_t stripes){
  a->stripes=stripes;a->taken|=UINT64_C(1)<<stripe;
  if((uint32_t)__builtin_popcountll(a->taken)!=a->stripes)return 0;
  await_drop(a);
  return 1;
}

/* design/algorithm-sources.md#meshresult: a program on a port stopped, its peer told (`tell`, END or FAIL naming the node
   `origin` where it happened), its status the failure naming `where`; a relay's other flow fails with it, telling its
   peer, and so does every flow of the same client program on its other cables and stripes, and its await for the
   stripes that have not come */
static void flow_fail(struct flow *f,int64_t code,uint32_t domain,uint32_t tell,uint32_t where,uint32_t origin){
  if(f->state==FAILED)return;
  f->state=FAILED;
  if(tell)flow_say(f,tell,0,code,NULL,origin);
  say("link %u: program %016llx toward node %u (stripe %u of %u) failed at node %u: code %lld domain %u\n",f->port->index,
    (unsigned long long)f->key,f->far,f->stripe,f->stripes,where,(long long)code,domain);
  flow_stop(f,code,domain,where);
  if(f->partner)flow_fail(f->partner,code,domain,tell?tell:(uint32_t)(code==ECONNRESET?END:FAIL),origin,origin);
  for(struct flow *g=flows;f->owner && g;g=g->next)if(g->owner==f->owner && g->destination==f->destination)flow_fail(g,code,domain,FAIL,where,origin);
  struct await *a=f->owner?await_of(f->owner,f->destination):NULL;
  if(a)await_drop(a);
}
/* a client's program toward its destination `d` failed as a value naming the node `where`: its flows failed, telling
   their peers (their FAILs naming `origin`), its await dropped, and its status the failure */
static void program_end(uint64_t owner,uint32_t d,int64_t code,uint32_t where,uint32_t origin){
  for(struct flow *g=flows;g;g=g->next)if(g->owner==owner && g->destination==d)flow_fail(g,code,4,FAIL,where,origin);
  struct await *a=await_of(owner,d);
  if(a)await_drop(a);
  if(owned(owner))status_failed(&mesh_client_at(M,owner)->status[d],owner,code,4,where);
}
static void program_fail(uint64_t owner,uint32_t d,int64_t code,uint32_t where){ program_end(owner,d,code,where,M->node); }
/* design/bridge-and-ipc.md#protocol: a client's program of this node that runs: started (its slot's `configured` its
   owner, acquired before its digest is read) and not yet failed toward every destination it binds; and its digest
   (mesh.h mesh_client `digest`, 0 where it names none, or where the program does not run: a READY that meets a program
   that ended or failed is held for the execution's next program), and whether two named ones differ */
static int program_runs(uint64_t owner){
  if(!owned(owner) || atomic_load_explicit(&mesh_client_at(M,owner)->configured,memory_order_acquire)!=owner)return 0;
  const struct mesh_client *client=mesh_client_at(M,owner);
  for(uint32_t d=0,n=mesh_destination(client,0);d<n;d++)
    if(mesh_destination_transfers(M,owner,d) && atomic_load(&client->status[d].phase)!=MESH_STOPPED)return 1;
  return 0;
}
static uint64_t program_digest(uint64_t owner){
  return program_runs(owner)?atomic_load_explicit(&mesh_client_at(M,owner)->digest,memory_order_relaxed):0;
}
static int digest_apart(uint64_t mine,uint64_t theirs){ return mine && theirs && mine!=theirs; }
/* a client's program whose peer at `node` runs a program of its execution made from another digest: failed toward every
   destination it binds as a value (ESTALE naming that node), each flow telling its peer FAIL with ESTALE naming it too
   (that node's own told this one), so the peer's program fails alike (port_heard), and so on along the execution's
   programs: no member waits for a peer whose program differs */
static void program_stale(uint64_t owner,uint32_t node){
  if(!program_runs(owner))return;
  struct mesh_client *client=mesh_client_at(M,owner);
  say("program %016llx: node %u's program of it is made from another digest than this node's (%016llx): failed\n",
      (unsigned long long)client->key,node,(unsigned long long)atomic_load_explicit(&client->digest,memory_order_relaxed));
  for(uint32_t d=0,n=mesh_destination(client,0);d<n;d++)
    if(mesh_destination_transfers(M,owner,d))program_end(owner,d,ESTALE,node,client->tx[d].node==node?M->node:node);
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
/* a flow freed, its index in its port's table and the queue pairs it held of the device's given back */
static void flow_free(struct flow *f){
  for(struct flow **at=&flows;*at;at=&(*at)->next)if(*at==f){*at=f->next;break;}
  if(f->partner)f->partner->partner=NULL;
  f->port->flows--;
  f->port->reserved[f->slot/64]&=~(UINT64_C(1)<<(f->slot%64));
  if(f->exclusive)f->port->committed-=(uint32_t)f->qps;
  for(int d=0;d<2;d++){lease_drop(f->side[d].lease);lease_drop(f->pending[d].lease);}
  free(f);
  flow_count--;busy_tell();
}
/* design/bridge-and-ipc.md#clients: a client program's status once every flow of it toward the node runs, every stripe
   of a routed one among them (no await of it left) */
static void program_paired(struct flow *f){
  for(struct flow *g=flows;g;g=g->next)
    if(g->owner==f->owner && g->destination==f->destination && (g->state!=RUNNING || g->job))return;
  if(!owned(f->owner) || await_of(f->owner,f->destination))return;
  struct mesh_port_info *status=&mesh_client_at(M,f->owner)->status[f->destination];
  atomic_store_explicit(&status->code,0,memory_order_relaxed);atomic_store_explicit(&status->failed,0,memory_order_relaxed);
  atomic_store_explicit(&status->phase,MESH_PAIRED,memory_order_relaxed);
  status_prepared(status,f->owner);
}

/* design/bridge-and-ipc.md#protocol: POSTED told the peer once the flow's receives are posted; a relay's once its own
   receives are and its partner heard the next hop's POSTED, so a POSTED reaches a route's end only once the far end's
   program has started and every relay between has posted (end to end, design/recovery.md#first).  Tried at each of
   the flow's job ends and as its partner hears POSTED (port_heard), whichever comes last */
static void flow_tell(struct flow *f){
  if(f->told || f->job || f->state==FAILED || f->state<POSTING)return;
  if(f->partner && (!(f->partner->heard&HEARD_POSTED) || !f->side[MESH_RECEIVE].owner || !(f->configured&1u<<MESH_RECEIVE)))return;
  f->told=1;
  flow_say(f,POSTED,0,0,NULL,M->node);
}
/* design/bridge-and-ipc.md#lanes: the peer's word on whether the flow takes queue pairs of its own (its READY's
   `exclusive`, its PAIRS' count), ANDed into the flow's until the flow pairs, so both ends pair alike; 0, or EPROTO
   where a flow that paired hears otherwise */
static int flow_mode(struct flow *f,int exclusive){
  if(f->decided)return f->exclusive==!!exclusive?0:EPROTO;
  if(f->exclusive && !exclusive){f->exclusive=0;f->port->committed-=(uint32_t)f->qps;}
  return 0;
}
/* design/bridge-and-ipc.md#protocol: a flow's next step, its job's end or a message heard: paired once the peer's READY
   or PAIRS is heard (an exclusive flow's queue pairs made; PAIRS sent), ready to receive once its PAIRS are and, held,
   once its client's program took it (POSTED told, flow_tell), running once its POSTED is; a relay whose side's lists
   arrived after its receives were posted changes.  A flow on its port's lanes receives, runs and changes once the lanes
   are up (lanes that fail to be made or connected are the link's loss, which fails it: #lanes) */
static void flow_step(struct flow *f){
  if(f->state==FAILED || f->job || f->stopping)return;
  for(int d=0;d<2;d++)if(f->pending[d].owner && !f->side[d].owner){f->side[d]=f->pending[d];f->pending[d]=(struct side){0};}
  const struct lanes *l=&f->port->lanes;
  const int up=f->exclusive || l->state==LANES_UP;
  if(f->state==READIED && (f->heard&(HEARD_READY|HEARD_PAIRS))){f->decided=1;flow_job(f,JOB_PAIR);}
  else if(f->state==PAIRING && (f->heard&HEARD_PAIRS) && !f->hold && up)flow_job(f,JOB_RECEIVE);
  else if(f->state==POSTING && (f->heard&HEARD_POSTED) && up)flow_job(f,JOB_RUN);
  else if(up && (f->state==POSTING || f->state==RUNNING) && ((f->side[MESH_SEND].owner && !(f->configured&1u<<MESH_SEND) && f->state==RUNNING) ||
                                (f->side[MESH_RECEIVE].owner && !(f->configured&1u<<MESH_RECEIVE))))flow_job(f,JOB_CHANGE);
}
static void flow_done(struct flow *f){
  const int job=f->job;
  f->job=0;
  if(f->state==FAILED)return;
  if(f->result){
    if(job==JOB_PAIR && f->result==ENOSPC)
      say("link %u (%s): program %016llx toward node %u: %d queue pairs refused, the device holding %u of its max_qp %u\n",
        f->port->index,f->port->name,(unsigned long long)f->key,f->far,f->qps,f->port->device.pairs,f->port->device.max_pairs);
    flow_fail(f,f->result,1,FAIL,M->node,M->node);
    return;
  }
  if(job==JOB_PAIR){
    /* #lanes: an exclusive flow's PAIRS its queue pairs, a lane rider's none, its index and generation in its code */
    uint32_t pairs[MESH_PAIRS][2];
    for(int q=0;f->exclusive && q<f->qps;q++){pairs[q][0]=f->provider.queues[q].pair->qp_num;pairs[q][1]=(f->provider.psn+(uint32_t)q)&0xffffff;}
    flow_say(f,PAIRS,f->exclusive?(uint32_t)f->qps:0,f->exclusive?0:(int64_t)((uint64_t)f->generation<<32|f->slot),
             (const uint32_t (*)[2])pairs,M->node);
    f->state=PAIRING;
    if(!f->hold)f->deadline=now+MESH_PAIR_NS;
  }else if(job==JOB_RECEIVE){
    /* design/recovery.md#first: the peer's POSTED waits on its program's start where no READY told of it (a peer that
       holds posts once its client's program takes it), and a relay's on the far end's: the peer window, and the
       pairing window after it */
    f->state=POSTING;f->deadline=(f->heard&HEARD_READY) && !f->partner?now+MESH_PAIR_NS:f->created+MESH_PEER_NS+MESH_PAIR_NS;
  }else{
    if(job==JOB_RUN)f->state=RUNNING;
    if(f->state==RUNNING)program_paired(f);
  }
  flow_tell(f);
  flow_step(f);
}

/* design/bridge-and-ipc.md#flows: a flow made on a port toward `far` for the execution `key`, the program's invocations
   t with t mod stripes = stripe (#cables), an index of the port's table its own (MESH_FLOWS a port), exclusive where
   `exclusive` asks it, its queue pairs fit a PAIRS message, and the port's device has room for them beside those its
   flows and lanes hold or will, the lanes' own counted before they are made (#lanes: else it rides the lanes); NULL
   where it is not one or the table is full */
static struct flow *flow_make(struct port *port,uint64_t key,uint32_t near,uint32_t far,uint32_t queues,uint32_t slots,uint32_t stripe,uint32_t stripes,
                              int exclusive){
  if(!queues || !slots || queues>MESH_QUEUES || (uint64_t)queues*slots>INT32_MAX || !stripes || stripes>MESH_STRIPES || stripe>=stripes)return NULL;
  uint32_t slot=0;
  while(slot<MESH_FLOWS && port->reserved[slot/64]>>(slot%64)&1)slot++;
  if(slot==MESH_FLOWS)return NULL;
  struct flow *f=calloc(1,sizeof *f);
  if(!f)return NULL;
  const uint32_t qps=queues*slots;
  exclusive=exclusive && qps<=MESH_PAIRS && port->device.max_pairs &&
    port->committed+qps+(port->lanes.committed?0:MESH_LANES)<=port->device.max_pairs;
  *f=(struct flow){.next=flows,.near=near,.far=far,.queues=queues,.stripe=stripe,.stripes=stripes,.qps=(int)qps,.exclusive=exclusive,
    .slot=slot,.generation=++port->generation[slot],.key=key,.created=now,.deadline=now+MESH_PEER_NS,.M=M,.port=port,
    .provider={.device=&port->device}};
  port->reserved[slot/64]|=UINT64_C(1)<<(slot%64);
  if(slot>=atomic_load(&port->high))atomic_store(&port->high,slot+1);
  if(exclusive)port->committed+=qps;
  flows=f;port->flows++;flow_count++;busy_tell();
  return f;
}
static struct flow *flow_on(struct port *port,uint64_t key,uint32_t near,uint32_t far,uint32_t stripe){
  for(struct flow *f=flows;f;f=f->next)
    if(f->port==port && f->key==key && f->near==near && f->far==far && f->stripe==stripe && f->state!=FAILED)return f;
  return NULL;
}

/* design/bridge-and-ipc.md#flows, #routes: a relayed program a READY carried, mirrored into a lease: rows the bridge
   leases for each transfer the sender sends (mesh.h mesh_relay_slot a slot of it), bound as receives from `from` and as
   SENDs toward `to` over the same rows, so a landed record releases the SEND that carries it on (M08), and the
   sender's digest named, which the relay's READY onward carries (#protocol).  0, or the refusal */
static int mirror(struct lease *lease,const struct ready *r,const struct mesh_transfer *t,uint32_t from,uint32_t to){
  struct mesh_ctx *c=&lease->context;
  struct mesh_client *client=mesh_client_at(M,c->client);
  int error=0;
  client->tx[mesh_destination(client,from)].node=from;
  client->tx[mesh_destination(client,to)].node=to;
  for(uint32_t k=0,i=0;k<MESH_QUEUES && !error;k++)for(uint32_t j=0;j<r->lengths[k] && !error;j++,i++){
    const struct mesh_transfer *x=t+i;
    struct mesh_section s;
    if(!x->count)error=EPROTO;
    else if(!(error=mesh_section_create(c,mesh_relay_slot(x,r->depth,M->pgsz),x->stride?x->count:1,&s))){
      s.bytes=x->bytes;if(!x->stride){s.stride=0;s.count=x->count;}
      error=mesh_transfer_bind(c,from,k,MESH_RECEIVE,x->binding,s,x->invocation_pages,x->begin,x->end,x->depth);
      if(!error)error=mesh_transfer_bind(c,to,k,MESH_SEND,x->binding,s,x->invocation_pages,x->begin,x->end,x->depth);
    }
  }
  if(!error)error=mesh_transfers_prepare(c,r->slots,r->invocations,r->depth);
  if(!error && r->cyclic)error=mesh_transfers_cyclic(c);
  if(!error)error=mesh_transfers_digest(c,r->digest);
  return error;
}
/* a READY refused as a value, naming this node */
static void ready_refuse(struct port *port,const struct message *heard,int64_t code){
  port_say(port,(struct message){.kind=FAIL,.key=heard->key,.from=heard->to,.to=heard->from,.code=code,.failed=M->node,.stripe=heard->stripe},
           NULL,0);
}
/* a READY's program apart from its stripe and route, the lease of a relayed direction its stripes share (FNV-1a) */
static uint64_t ready_shape(const struct ready *ready,const struct mesh_transfer *list,uint64_t transfers){
  struct ready shape=*ready;
  shape.hops=shape.stripe=shape.exclusive=0;
  uint64_t x=UINT64_C(0xcbf29ce484222325);
  for(size_t i=0;i<sizeof shape;i++)x=(x^((const unsigned char *)&shape)[i])*UINT64_C(0x100000001b3);
  for(size_t i=0;i<transfers*sizeof *list;i++)x=(x^((const unsigned char *)list)[i])*UINT64_C(0x100000001b3);
  return x;
}
/* the lease a relayed direction's other stripes through this node took: a relay flow of the same execution receiving
   from the same far end, whose partner sends toward the same far end; NULL where none has */
static struct lease *relay_lease(const struct flow *f,const struct flow *g){
  for(struct flow *h=flows;h;h=h->next){
    if(h==f || h->key!=f->key || h->far!=f->far || h->stripes!=f->stripes || !h->partner || h->partner->far!=g->far || h->state==FAILED ||
       h->stopping)continue;
    struct lease *lease=h->side[MESH_RECEIVE].lease?h->side[MESH_RECEIVE].lease:h->pending[MESH_RECEIVE].lease;
    if(lease)return lease;
  }
  return NULL;
}
/* a relay's direction a READY told: received by the flow it arrived on and sent on by its partner, over a lease that
   mirrors its lists, the one its program's other stripes through this node share (each stripe's flows prepare their
   own invocations of its rows, #cables), else a new one; EPROTO where a stripe's program is not its lease's */
static int relay_direction(struct flow *f,const struct ready *ready,const struct mesh_transfer *list,uint64_t shape){
  struct flow *g=f->partner;
  struct lease *lease=relay_lease(f,g);
  if(lease && lease->shape!=shape)return EPROTO;
  if(!lease){
    if(!(lease=lease_take(f->key)))return EBUSY;
    int error=mirror(lease,ready,list,f->far,g->far);
    if(error){lease_free(lease);return error;}
    lease->shape=shape;
  }
  struct mesh_client *client=mesh_client_at(M,lease->context.client);
  f->pending[MESH_RECEIVE]=(struct side){lease->context.client,mesh_destination(client,f->far),lease};
  g->pending[MESH_SEND]=(struct side){lease->context.client,mesh_destination(client,g->far),lease};
  lease->users+=2;
  if(!f->owner)flow_program(f,f->pending[MESH_RECEIVE].owner,f->pending[MESH_RECEIVE].destination);
  if(!g->owner)flow_program(g,g->pending[MESH_SEND].owner,g->pending[MESH_SEND].destination);
  flow_step(g);
  return 0;
}

static uint32_t program_queues(uint64_t owner,uint32_t d);
/* a flow taken by its client's program toward destination `d` (its one job before, JOB_PAIR, reads no side), its port
   among the program's (mesh_client `ports`, the bells mesh_transfers_wake rings) */
static void flow_take(struct flow *f,uint64_t owner,uint32_t d){
  f->side[MESH_SEND]=f->side[MESH_RECEIVE]=(struct side){owner,d,NULL};
  flow_program(f,owner,d);
  if(f->hold)f->deadline=now+MESH_PAIR_NS;
  f->hold=0;
  struct mesh_client *client=mesh_client_at(M,owner);
  atomic_fetch_or_explicit(&client->ports,UINT32_C(1)<<f->port->index,memory_order_release);
  atomic_store_explicit(&client->status[d].phase,MESH_PAIRING,memory_order_relaxed);
}
/* the accepted program of execution `key` that binds `node` here (its slot's owner and destination), with a flow toward
   it that has not failed; 0 where none */
static uint64_t owners[MESH_CLIENTS],accepted[MESH_CLIENTS];
static int program_live(uint64_t key,uint32_t node,uint64_t *owner,uint32_t *d){
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct mesh_client *client=(struct mesh_client *)((char *)M+M->client_off)+s;
    const uint64_t o=accepted[s];
    if(!o || !owned(o) || client->key!=key)continue;
    const uint32_t at=mesh_destination(client,node);
    if(at==MESH_DESTINATIONS || client->tx[at].node!=node)continue;
    for(struct flow *g=flows;g;g=g->next)if(g->owner==o && g->destination==at && g->state!=FAILED){*owner=o;*d=at;return 1;}
  }
  return 0;
}
/* the program of execution `key` that runs here (program_runs: its slot's owner), 0 where none */
static uint64_t program_keyed(uint64_t key){
  for(uint32_t s=0;s<MESH_CLIENTS;s++)
    if(accepted[s] && program_runs(accepted[s]) && mesh_client_at(M,accepted[s])->key==key)return accepted[s];
  return 0;
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
/* design/bridge-and-ipc.md#cables: a program's pair of end nodes and its execution mixed (murmur3's finalizer), alike at
   either end and on every relay, which with a stripe picks the cable a routed hop takes */
static uint64_t pair_hash(uint64_t key,uint32_t a,uint32_t b){
  uint64_t x=key^((uint64_t)(a<b?a:b)<<32|(a<b?b:a));
  x^=x>>33;x*=UINT64_C(0xff51afd7ed558ccd);x^=x>>33;x*=UINT64_C(0xc4ceb9fe1a85ec53);x^=x>>33;
  return x;
}
/* the cable of a routed hop toward the node of `next` (its n cables) that stripe `stripe` of the program between `a`
   and `b` rides: stripe j on cable (hash + j) mod n, so a hop of c cables carries stripes/c of a program whose stripes
   c divides */
static struct port *stripe_cable(struct port **next,uint32_t n,uint64_t key,uint32_t a,uint32_t b,uint32_t stripe){
  return next[(pair_hash(key,a,b)+stripe)%n];
}

/* design/bridge-and-ipc.md#protocol, #routes: a READY.  Its flow here pairs (its `exclusive` ANDed into the flow's,
   #lanes); one heard by a relay's flow (the far end's program telling its direction) is mirrored and sent on by the
   relay's other flow toward the route's start, so every relay of the route carries both directions; a program whose
   route goes on past this node is relayed toward the route's next node, over rows the bridge leases (no client of this
   node takes part, its stripes sharing them), each stripe on the lanes of the cable of the next hop it rides
   (stripe_cable); one that ends here is taken by a program awaiting it, else held, its queue pairs made where it takes
   its own and no rows, for the client whose program takes it (#flows).  One that ends here naming another digest than
   this node's program of its key (#protocol) is refused, and that program failed toward every destination, both as
   values (ESTALE: program_stale).  One that ends on a port where this node's program of its key toward its sender has
   no flow, while that program runs on others, finds the two ends' maps apart: refused, and the program failed, both as
   values (EPROTO) */
static void ready_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  struct ready ready;
  if(heard->bytes<sizeof ready){port_lost(port,EPROTO,0);return;}
  memcpy(&ready,payload,sizeof ready);
  uint64_t transfers=0;
  for(uint32_t k=0;k<MESH_QUEUES;k++)transfers+=ready.lengths[k];
  const uint32_t queues=ready.slots?heard->count/ready.slots:0;
  if(!ready.hops || !queues || queues>MESH_QUEUES || heard->count%ready.slots || !ready.stripes || ready.stripes>MESH_STRIPES ||
     ready.stripe>=ready.stripes || ready.stripe!=heard->stripe ||
     heard->bytes!=sizeof ready+(uint64_t)ready.hops*sizeof(uint32_t)+transfers*sizeof(struct mesh_transfer)){port_lost(port,EPROTO,0);return;}
  uint32_t *route=malloc((size_t)ready.hops*sizeof *route);
  struct mesh_transfer *list=malloc((transfers?transfers:1)*sizeof *list);
  if(!route || !list){free(route);free(list);ready_refuse(port,heard,ENOMEM);return;}
  memcpy(route,payload+sizeof ready,(size_t)ready.hops*sizeof *route);
  memcpy(list,payload+sizeof ready+(size_t)ready.hops*sizeof *route,transfers*sizeof *list);
  const uint64_t shape=ready_shape(&ready,list,transfers);
  struct flow *f=flow_on(port,heard->key,heard->to,heard->from,ready.stripe);
  uint64_t owner=0;uint32_t d=0;
  if(f && (f->heard&HEARD_READY)){}
  else if(f && !f->partner && digest_apart(program_digest(f->owner),ready.digest)){
    flow_fail(f,ESTALE,4,FAIL,heard->from,M->node);
    program_stale(f->owner,heard->from);
  }
  else if(f && !f->partner && (f->stripes!=ready.stripes || f->queues!=queues || (uint32_t)f->qps!=heard->count))
    flow_fail(f,EPROTO,4,FAIL,M->node,M->node);
  else if(f){
    int error=f->partner?(!f->side[MESH_RECEIVE].owner && !f->pending[MESH_RECEIVE].owner?relay_direction(f,&ready,list,shape):0):
      flow_mode(f,(int)ready.exclusive);
    if(error)flow_fail(f,error,1,FAIL,M->node,M->node);
    else{
      if(f->partner)port_say(f->partner->port,(struct message){.kind=READY,.count=heard->count,.key=heard->key,.from=heard->from,
                                                               .to=heard->to,.stripe=heard->stripe},payload,heard->bytes);
      f->heard|=HEARD_READY;flow_step(f);
    }
  }else if(route[0]!=M->node || route[ready.hops-1]!=heard->to || heard->from==M->node)ready_refuse(port,heard,EPROTO);
  else if(ready.hops==1){
    struct await *a=await_find(heard->key,heard->from);
    const uint64_t keyed=a?a->owner:program_keyed(heard->key);
    if(digest_apart(program_digest(keyed),ready.digest)){
      ready_refuse(port,heard,ESTALE);
      program_stale(keyed,heard->from);
    }else if(a && (program_queues(a->owner,a->destination)!=queues || mesh_client_at(M,a->owner)->tx[a->destination].slots!=ready.slots ||
             (a->stripes && a->stripes!=ready.stripes) || (a->taken>>ready.stripe&1))){
      say("link %u (%s): program %016llx: node %u's READY (stripe %u of %u) is not the shape of the program awaiting it here: refused\n",
          port->index,port->name,(unsigned long long)heard->key,heard->from,ready.stripe,ready.stripes);
      ready_refuse(port,heard,EPROTO);
      program_fail(a->owner,a->destination,EPROTO,M->node);
    }else if(!a && program_live(heard->key,heard->from,&owner,&d)){
      say("link %u (%s): program %016llx: node %u's READY came on a port its program here does not use (the two ends' maps "
          "differ): refused\n",port->index,port->name,(unsigned long long)heard->key,heard->from);
      ready_refuse(port,heard,EPROTO);
      program_fail(owner,d,EPROTO,M->node);
    }else if(!(f=flow_make(port,heard->key,M->node,heard->from,queues,ready.slots,ready.stripe,ready.stripes,(int)ready.exclusive)))
      ready_refuse(port,heard,E2BIG);
    else{
      f->hold=1;f->heard=HEARD_READY;f->digest=ready.digest;
      if(a){flow_take(f,a->owner,a->destination);await_took(a,ready.stripe,ready.stripes);ready_say(f,&f->far,1);}
      flow_step(f);
    }
  }else{
    struct port *next[MESH_PORTS];
    struct flow *g=NULL;
    const uint32_t n=cables(route[1],next);
    f=n?flow_make(port,heard->key,heard->to,heard->from,queues,ready.slots,ready.stripe,ready.stripes,0):NULL;
    if(f)g=flow_make(stripe_cable(next,n,heard->key,heard->from,heard->to,ready.stripe),heard->key,heard->from,heard->to,queues,
                     ready.slots,ready.stripe,ready.stripes,0);
    if(!g){
      if(f){f->state=FAILED;flow_free(f);}
      ready_refuse(port,heard,n?E2BIG:EHOSTUNREACH);
    }else{
      f->partner=g;g->partner=f;
      int error=relay_direction(f,&ready,list,shape);
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
/* RFC 2328 s14: the map with `record` put in (its node's held record replaced, else added) and every record no paired
   path from this node reaches left out, written only where it differs from the map held (so a record that changes
   nothing, or that the prune drops at once, moves no serial) */
static void map_put(const struct mesh_record *record){
  const uint64_t n=atomic_load(&M->map_nodes);
  struct mesh_record *r=mesh_records(M);
  uint64_t at=0;
  while(at<n && r[at].node!=record->node)at++;
  const uint64_t count=at==n?n+1:n;
  struct mesh_record *next=malloc(count*sizeof *next);
  unsigned char *reached=calloc(count,1);
  uint32_t *order=malloc(count*sizeof *order),head=0,tail=0;
  if(!next || !reached || !order){free(next);free(reached);free(order);say("map: no memory to put node %u's record\n",record->node);return;}
  memcpy(next,r,(size_t)n*sizeof *r);
  memcpy(next+at,record,sizeof *record);
  for(uint64_t k=0;k<count;k++)if(next[k].node==M->node){reached[k]=1;order[tail++]=(uint32_t)k;}
  while(head<tail){
    uint32_t a=order[head++];double beta;
    for(uint64_t b=0;b<count;b++)if(!reached[b] && mesh_joined(next+a,next+b,&beta)){reached[b]=1;order[tail++]=(uint32_t)b;}
  }
  /* design/recovery.md#diagnosis: a node the prune drops is logged with its last health first (one put now and never
     reached yet is not a loss) */
  for(uint64_t k=0;k<count;k++)if(!reached[k] && !(at==n && k==at)){
    struct known *gone=known_at(next[k].node);char text[768];health_text(&next[k].health,text,sizeof text);
    say("node %u (%s) unreachable; its last health, heard %.1f s before: %s\n",next[k].node,next[k].name,
        gone && gone->heard?(double)(now-gone->heard)/1e9:-1.0,text);
    if(gone)gone->gone=now;
  }
  uint64_t kept=0;
  for(uint64_t k=0;k<count;k++)if(reached[k])memmove(next+kept++,next+k,sizeof *next);
  if(kept!=n || memcmp(next,r,(size_t)n*sizeof *r)){
    mesh_notify(&M->map);
    if(map_grow(kept))say("map: no lease for %llu records; %llu kept\n",(unsigned long long)kept,(unsigned long long)map_room);
    if(kept>map_room)kept=map_room;
    memcpy(mesh_records(M),next,(size_t)kept*sizeof *next);
    atomic_store(&M->map_nodes,kept);
    mesh_notify(&M->map);
  }
  free(next);free(reached);free(order);
}
/* design/recovery.md#diagnosis: a record whose links (each port's peer, phase and bandwidth), name, boot and facts are
   the held one's changes only its health, sequence and ports' counts (their codes, pairings and queue pairs), written
   under the region's health serial and never the map's: the map serial moves only where the mesh changes (a
   communicator's session reads the map again where it moved, and reopens only where its own routes or their links'
   costs changed: nccl-mesh.c map_digest, session_ensure), and neither a health sample nor a queue pair is a change of
   the mesh */
static int record_links_same(const struct mesh_record *a,const struct mesh_record *b){
  int same=a->node==b->node && a->ports==b->ports && a->boot==b->boot && !memcmp(a->name,b->name,sizeof a->name) &&
    a->window==b->window && a->memory==b->memory && !memcmp(a->model,b->model,sizeof a->model);
  for(uint32_t p=0;same && p<MESH_PORTS;p++)
    same=atomic_load(&a->port[p].peer)==atomic_load(&b->port[p].peer) && atomic_load(&a->port[p].phase)==atomic_load(&b->port[p].phase) &&
      atomic_load(&a->port[p].bandwidth)==atomic_load(&b->port[p].bandwidth);
  return same;
}
static void health_put(struct mesh_record *held,const struct mesh_record *record){
  mesh_notify(&M->health);
  held->sequence=record->sequence;held->health=record->health;
  memcpy(held->port,record->port,sizeof held->port);
  mesh_notify(&M->health);
}
static struct mesh_record *record_held(uint32_t node){
  for(uint64_t i=0;i<atomic_load(&M->map_nodes);i++)if(mesh_records(M)[i].node==node)return mesh_records(M)+i;
  return NULL;
}
/* design/bridge-and-ipc.md#records: this node's record, its window, memory and model with its ports and health */
static uint64_t memory;
static char model[MESH_MODEL_BYTES];
static void record_changed(void){
  struct mesh_record own={.node=M->node,.ports=port_count,.boot=M->boot,.sequence=++sequence,.window=wire.length,.memory=memory};
  memcpy(own.name,host,sizeof own.name);
  memcpy(own.model,model,sizeof own.model);
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
  heard.model[sizeof heard.model-1]=0;
  struct mesh_record *held=record_held(heard.node);
  struct known *k=known_at(heard.node);
  /* design/bridge-and-ipc.md#records, RFC 2328 s13: a record no newer than the newest heard of its node, the map holding
     it or not, is dropped and not sent on, so a record the map pruned never circulates; an entry unheard for
     MESH_RECORD_AGE_NS whose node the map no longer holds compares with nothing (s14's MaxAge) */
  const int aged=k && !held && k->heard && now-k->heard>MESH_RECORD_AGE_NS;
  if(k && k->boot && !aged && (heard.boot<k->boot || (heard.boot==k->boot && heard.sequence<=k->sequence))){
    /* the newest known of a node the map dropped while no paired path reached it: put again where a path now does,
       and sent on only then (a pruned record that no path reaches stops here) */
    if(!held && heard.boot==k->boot && heard.sequence==k->sequence){
      map_put(&heard);
      for(uint32_t p=0;record_held(heard.node) && p<port_count;p++)
        if(ports+p!=from && ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&heard,sizeof heard);
    }
    return;
  }
  if(held && (heard.boot<held->boot || (heard.boot==held->boot && heard.sequence<=held->sequence)))return;
  /* design/recovery.md#diagnosis: a node that joins or comes back, and a peer's urgent change, are logged here */
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
  if(k){k->heard=now;k->gone=0;k->boot=heard.boot;k->sequence=heard.sequence;memcpy(k->session,heard.health.session,sizeof k->session);}
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

/* design/bridge-and-ipc.md#lanes: a lanes job on the port's queue: their queue pairs and completion queues made
   (LANES_MAKING); ready to receive from the peer's and to send, each lane listening for its first header and the
   port's progress threads started (LANES_CONNECTING: the event loop alone makes them up as it takes the job, so lanes
   reset while they connected stay down); or down once neither progress thread reads them, their queue pairs destroyed
   and their generation moved (LANES_CLOSING) */
static void lanes_job(struct port *port,int job){
  struct lanes *l=&port->lanes;
  l->job=job;atomic_store_explicit(&l->done,0,memory_order_relaxed);
  dispatch_async(port->work,^{
    int result=0;
    if(job==LANES_MAKING)result=pair_up(&l->provider,MESH_LANES);
    else if(job==LANES_CONNECTING){
      result=pair_receive(&l->provider,(const uint32_t (*)[2])l->peer,port->heard.lid,port->heard.gid);
      if(!result)result=pair_send(&l->provider);
      const uint32_t sends=(uint32_t)l->provider.sent->cqe/MESH_LANES;
      for(uint32_t lane=0;!result && lane<MESH_LANES;lane++){
        l->gate[lane]=(struct send_gate){.capacity=MIN(l->provider.queues[lane].send_capacity,sends)};
        l->sent[lane]=l->received[lane]=0;l->cursor[lane]=0;l->records=0;
        result=lane_listen(port,lane);
      }
      if(!result)result=port_threads(port);
    }else{
      atomic_store_explicit(&l->up,0,memory_order_release);
      port_settle(port);
      result=down_pair(&l->provider)?0:EBUSY;
      if(!result)l->generation++;
    }
    l->result=result;
    port_pairs(port);
    atomic_store_explicit(&l->done,1,memory_order_release);
    wake();
  });
}
/* the frames of payload a lane's receive queue takes beside its header: its queue pair's and its share of the receive
   completion queue, less the header's frame */
static uint32_t lanes_room(const struct lanes *l){
  uint32_t room=(uint32_t)l->provider.completion->cqe/MESH_LANES;
  for(uint32_t lane=0;lane<MESH_LANES;lane++)room=MIN(room,l->provider.queues[lane].receive_capacity);
  return room>1?room-1:0;
}
/* the port's flows stepped (its lanes changed) */
static void port_step(struct port *port){
  for(struct flow *f=flows,*next;f;f=next){next=f->next;if(f->port==port)flow_step(f);}
}
/* design/recovery.md#loss: lanes that failed (made, connected or run) or whose peer's did are the lanes' alone: the
   flows that ride them and have posted on them fail (their records and SENDs went with them), the lanes are closed and
   made again a pause later, and the peer is told (a LANES of none and why) where the failure is this end's; the link
   stays paired, and its flows with queue pairs of their own and those still pairing run on */
static void lanes_reset(struct port *port,int64_t code,const char *what,int tell){
  struct lanes *l=&port->lanes;
  say("link %u (%s): lanes toward node %u %s: %s; made again in %.0f s, the link paired and its own queue pairs running on\n",
      port->index,port->name,port->peer,what,strerror((int)code),(double)MESH_PAUSE_NS/1e9);
  if(tell)port_say(port,(struct message){.kind=LANES,.count=0,.code=code},NULL,0);
  atomic_store_explicit(&l->up,0,memory_order_release);
  for(struct flow *f=flows;f;f=f->next)
    if(f->port==port && !f->exclusive && f->state!=FAILED && (f->state>=POSTING || f->job==JOB_RECEIVE || f->job==JOB_RUN || f->job==JOB_CHANGE))
      flow_fail(f,code,1,FAIL,M->node,M->node);
  if(l->state!=LANES_DOWN)l->state=LANES_CLOSING;
  l->heard=0;
}
/* design/bridge-and-ipc.md#lanes: the port's lanes' turn, on the event loop: a job's end taken (made: LANES told the
   peer, its queue pairs and its room; connected: up, and the port's flows stepped; closed: down, the device's queue
   pairs given back; a make or connect that failed: lanes_reset); a lane's failure is the lanes' (lanes_reset); the lanes
   made once the port pairs, or a pause after they failed, where the device has room for them (flow_make counts it
   before any flow's own; else made at the first turn it has, the cause logged once), connected once both ends made
   theirs, closed once the link went or they failed */
static void lanes_turn(struct port *port){
  struct lanes *l=&port->lanes;
  if(l->job){
    if(!atomic_load_explicit(&l->done,memory_order_acquire))return;
    const int job=l->job,result=l->result;
    l->job=0;
    if(job==LANES_CLOSING){
      if(result)say("link %u (%s): lanes teardown: %s\n",port->index,port->name,strerror(result));
      else{l->state=LANES_DOWN;if(l->committed)port->committed-=MESH_LANES;l->committed=0;}
    }else if(l->state==LANES_CLOSING){}
    else if(result)lanes_reset(port,result,job==LANES_MAKING?"not made":"not connected",1);
    else if(job==LANES_MAKING){
      l->state=LANES_MADE;l->room=lanes_room(l);
      struct message said={.kind=LANES,.count=MESH_LANES,.code=l->room};
      for(uint32_t lane=0;lane<MESH_LANES;lane++){said.pair[lane][0]=l->provider.queues[lane].pair->qp_num;said.pair[lane][1]=(l->provider.psn+lane)&0xffffff;}
      port_say(port,said,NULL,0);
    }else{
      l->state=LANES_UP;l->code=0;
      atomic_store_explicit(&l->up,1,memory_order_release);port_bell(port);
      say("link %u (%s): lanes up toward node %u: %u queue pairs, %u frames a header\n",port->index,port->name,port->peer,MESH_LANES,
          MIN(l->room,l->peer_room));
      port_step(port);
    }
  }
  const int64_t failed=atomic_exchange(&l->failed,0);
  if(failed && port->greeted && (l->state==LANES_UP || l->state==LANES_CONNECTING))lanes_reset(port,failed,"failed",1);
  if(l->job)return;
  if(l->state==LANES_CLOSING && now>=l->retry){l->retry=now+MESH_PAUSE_NS;lanes_job(port,LANES_CLOSING);}
  else if(port->greeted && l->state==LANES_DOWN && now>=l->retry){
    if(l->page==MESH_ABSENT && (l->page=mesh_arena_alloc(&bridge,l->pages,1))==MESH_ABSENT){
      if(l->code!=ENOMEM)say("link %u (%s): lanes not made: the window holds no %u pages for them; made once it has\n",port->index,
                             port->name,l->pages);
      l->code=ENOMEM;l->retry=now+MESH_PAUSE_NS;
    }else if(port->committed+MESH_LANES>port->device.max_pairs){
      if(l->code!=ENOSPC)say("link %u (%s): lanes not made: the device holds %u of its %u usable queue pairs; made once it has room\n",
                             port->index,port->name,port->committed,port->device.max_pairs);
      l->code=ENOSPC;l->retry=now+MESH_PAUSE_NS;
    }else{port->committed+=MESH_LANES;l->committed=1;l->state=LANES_MAKING;lanes_job(port,LANES_MAKING);}
  }
  else if(port->greeted && l->state==LANES_MADE && l->heard){l->state=LANES_CONNECTING;lanes_job(port,LANES_CONNECTING);}
}
/* a peer's LANES: its lanes' queue pairs and the frames a header may name, or (count 0) its lanes failed, with why:
   lanes of this end that were connected, or connecting, to its old ones are reset (lanes_reset, untold: the peer knows),
   and lanes made and not yet connected wait for its new ones.  A peer's new lanes heard while this end's are connected
   to its old ones reset them alike, the new ones kept for the next connect */
static void lanes_heard(struct port *port,const struct message *heard){
  struct lanes *l=&port->lanes;
  const int joined=l->state==LANES_UP || l->state==LANES_CONNECTING;
  if(heard->count!=MESH_LANES){
    if(joined)lanes_reset(port,heard->code?heard->code:EPROTO,"failed at its peer",0);
    else{say("link %u (%s): node %u's lanes failed: %s\n",port->index,port->name,port->peer,strerror((int)heard->code));l->heard=0;}
    return;
  }
  if(joined)lanes_reset(port,EPROTO,"made again by its peer",0);
  memcpy(l->peer,heard->pair,sizeof l->peer);
  l->peer_room=(uint32_t)heard->code;l->heard=1;
  lanes_turn(port);
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
  if(heard->kind==LANES){lanes_heard(port,heard);return;}
  struct flow *f=flow_on(port,heard->key,heard->to,heard->from,heard->stripe);
  if(!f)return;
  const uint32_t origin=heard->failed?heard->failed:port->peer;
  if(heard->kind==PAIRS){
    /* #lanes: an exclusive flow's PAIRS its queue pairs, a lane rider's (count 0) its index and generation */
    const int error=flow_mode(f,heard->count!=0)?EPROTO:f->exclusive && heard->count!=(uint32_t)f->qps?EPROTO:0;
    if(error)flow_fail(f,error,4,FAIL,port->peer,M->node);
    else{
      if(f->exclusive)memcpy(f->pairs,heard->pair,sizeof f->pairs);
      else{f->peer_slot=(uint32_t)heard->code;f->peer_generation=(uint32_t)((uint64_t)heard->code>>32);}
      f->heard|=HEARD_PAIRS;
    }
  }
  else if(heard->kind==POSTED){f->heard|=HEARD_POSTED;if(f->partner)flow_tell(f->partner);}
  else if(heard->kind==END)flow_fail(f,ECONNRESET,4,0,origin,origin);
  else if(heard->kind==FAIL){
    const uint64_t owner=f->partner?0:f->owner;
    flow_fail(f,heard->code==ESTALE?ESTALE:ECANCELED,4,0,origin,origin);
    if(heard->code==ESTALE && owner)program_stale(owner,origin);
  }
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
  /* #lanes: read by no progress thread once the stops of its flows (queued after) settle the port's table, and closed
     once those stops have run */
  atomic_store_explicit(&port->lanes.up,0,memory_order_release);
  for(struct flow *f=flows;f;f=f->next)if(f->port==port)flow_fail(f,port->parted?ECONNRESET:code,4,0,port->peer,M->node);
  if(port->lanes.state!=LANES_DOWN)port->lanes.state=LANES_CLOSING;
  port->lanes.heard=0;port->lanes.retry=0;
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
   flow and no lane holds queue pairs on it), else opened and its port read; then, while it has no peer, its hello sent to every
   node on its link */
static void port_turn(struct port *port){
  if(port->job || port->control>=0 || now<port->next || (port->retiring && (port->flows || port->lanes.state!=LANES_DOWN)))return;
  port->next=now+MESH_PAUSE_NS;
  port->interface=if_nametoindex(port->name);
  const int retire=port->retiring;
  port->job=1;atomic_store_explicit(&port->done,0,memory_order_relaxed);
  dispatch_async(port->work,^{
    int result=0,refused=0;
    if(retire && !down_device(&port->device))result=-(errno?errno:EBUSY);
    else if(!(result=device_up(&port->device,&wire,&refused)))result=device_gid(&port->device);
    port->result=result;port->refused=refused;
    atomic_store_explicit(&port->done,1,memory_order_release);
    wake();
  });
}
/* design/recovery.md#registration: a port's turn taken: a device whose registration of the window it refused keeps the
   extents it took and turns again a pause later, 60 s doubling to 30 min (device_up logged the extent and why); a
   device that failed otherwise, its link active, is retired */
static void port_opened(struct port *port){
  const int result=port->result;
  port->job=0;
  if(result<0){say("link %u (%s): device retirement: %s\n",port->index,port->name,strerror(-result));return;}
  if(port->refused){
    port->backoff=port->backoff?MIN(2*port->backoff,MESH_REFUSED_MOST_NS):MESH_REFUSED_NS;
    port->next=now+port->backoff;
    say("link %u (%s): the window's registration refused (%s): the port turns again in %.0f s\n",port->index,port->name,
        strerror(port->refused),(double)port->backoff/1e9);
    port->open=0;if(port->control<0)link_set(port,0,MESH_STOPPED,result);
    return;
  }
  port->retiring=result && port->device.context && result!=ENETDOWN;
  if(result){port->open=0;if(port->control<0)link_set(port,0,MESH_STOPPED,result);return;}
  port->backoff=0;
  atomic_store_explicit(&mesh_links(M)[port->index].max_pairs,port->device.max_pairs,memory_order_relaxed);
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
/* design/bridge-and-ipc.md#lanes: the lanes' pages of every port the region has, leased at the bridge's start beside the
   map's, so a port found later (an interface attached once clients hold the window) has them; a port whose pages the
   window could not give takes them when its lanes are made (lanes_turn) */
static uint32_t lanes_page[MESH_PORTS];
static void lanes_reserve(void){
  for(uint32_t p=0;p<MESH_PORTS;p++)lanes_page[p]=mesh_arena_alloc(&bridge,lane_pages(),1);
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
    /* design/bridge-and-ipc.md#lanes: the lanes' headers and discard blocks, the pages leased for the port's index */
    port->lanes.pages=lane_pages();
    port->lanes.page=lanes_page[port_count];
    port->lanes.provider.device=&port->device;
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

/* the queues (its highest queue with a list, plus one) and slots of a client's program toward its destination `d` */
static uint32_t program_queues(uint64_t owner,uint32_t d){
  uint32_t queues=0;
  for(uint32_t q=0;q<MESH_QUEUES;q++)for(int w=0;w<2;w++)if(atomic_load(mesh_order_length(M,owner,d*MESH_QUEUES+q,w)))queues=q+1;
  return queues;
}
/* whether a READY of execution `key` from `far` left the flow held here, on any port, for the client whose program
   takes it */
static int flow_held(const struct flow *f,uint64_t key,uint32_t far){
  return f->hold && f->key==key && f->near==M->node && f->far==far && f->state!=FAILED && !f->stopping;
}
/* design/bridge-and-ipc.md#cables: the cables a record holds paired with `peer`, and the stripes a routed program takes:
   the least common multiple of its hops' cables (a hop's the fewer of its two ends' records', the first hop's this
   node's own `first`), so each cable of every hop carries an equal share, or past MESH_STRIPES the fewest cables of a
   hop, so each stripe rides a cable of its own on every hop and the route carries its slowest hop's cables */
static uint32_t record_cables(const struct mesh_record *r,uint32_t peer){
  uint32_t n=0;
  for(uint32_t p=0;r && p<r->ports && p<MESH_PORTS;p++)
    n+=atomic_load_explicit(&r->port[p].phase,memory_order_relaxed)==MESH_PAIRED && atomic_load_explicit(&r->port[p].peer,memory_order_relaxed)==peer;
  return n;
}
static uint32_t route_stripes(const uint32_t *route,uint32_t hops,uint32_t first){
  uint32_t stripes=1,fewest=MESH_STRIPES;
  for(uint32_t i=0;i+1<hops;i++){
    uint32_t c=i?MIN(record_cables(record_held(route[i]),route[i+1]),record_cables(record_held(route[i+1]),route[i])):first;
    c=MIN(MAX(c,1u),MESH_STRIPES);
    uint32_t a=stripes,b=c;
    while(b){const uint32_t t=a%b;a=b;b=t;}
    stripes=stripes/a*c>MESH_STRIPES?MESH_STRIPES+1:stripes/a*c;
    fewest=MIN(fewest,c);
  }
  return stripes<=MESH_STRIPES?stripes:fewest;
}
/* whether this end's map routes to `node` over a cable it holds paired now */
static int route_held(uint32_t node){
  uint32_t *route=NULL;
  const uint32_t hops=mesh_route(M,M->node,node,&route);
  struct port *next[MESH_PORTS];
  const int held=hops>=2 && cables(route[1],next)>0;
  free(route);
  return held;
}
/* the newest record this node holds or held of `node` (its entry apart from the map; design/bridge-and-ipc.md#records);
   NULL where it never heard of it */
static const struct known *known_of(uint32_t node){
  for(uint32_t i=0;i<known_count;i++)if(known[i].node==node)return known+i;
  return NULL;
}
/* design/bridge-and-ipc.md#routes: the higher end of a routed pair awaits the lower's route, whatever its own map holds:
   the stripes the lower's READYs left held taken at once, the rest as they come (ready_heard), within the peer window
   where its own map routes to the lower too, else within the pairing window from when the lower was last reachable
   (its record pruned: a link of its route pairing again), and refused at once (EHOSTUNREACH, naming the lower) where it
   has been unreachable longer or was never heard of; a revoke of the client's program (its slot retired) or the lower's
   failure as a value ends it sooner.  Each take updates the await before the READY that may fail the program goes out */
static void program_await(struct mesh_client *client,uint64_t owner,uint32_t d,uint32_t node,uint32_t queues,uint32_t slots){
  struct await *a=calloc(1,sizeof *a);
  if(!a){status_failed(&client->status[d],owner,ENOMEM,4,M->node);return;}
  *a=(struct await){.next=awaits,.key=client->key,.owner=owner,.created=now,.destination=d,.far=node,.routed=1,.deadline=now+MESH_PEER_NS};
  awaits=a;await_count++;busy_tell();
  for(struct flow *h=flows;h;h=h->next){
    if(!flow_held(h,client->key,node))continue;
    if(h->queues!=queues || (uint32_t)h->qps!=queues*slots || (a->stripes && h->stripes!=a->stripes) || (a->taken>>h->stripe&1)){
      flow_fail(h,EPROTO,4,FAIL,M->node,M->node);
      program_fail(owner,d,EPROTO,M->node);
      return;
    }
    flow_take(h,owner,d);
    const int last=await_took(a,h->stripe,h->stripes);
    ready_say(h,&h->far,1);flow_step(h);
    if(last || (a=await_of(owner,d))==NULL)return;
  }
  if(a->taken || route_held(node))return;
  const struct known *k=known_of(node);
  const uint64_t since=record_held(node)?now:k && k->gone?k->gone:0;
  a->routed=0;a->deadline=since+MESH_PAIR_NS;
  if(!since || now>=a->deadline){
    say("program %016llx: node %u, the lower end of its route, %s: refused\n",(unsigned long long)client->key,node,
        since?"has been unreachable longer than the pairing window":"was never heard of");
    program_fail(owner,d,EHOSTUNREACH,node);
  }
}
/* design/bridge-and-ipc.md#clients, #cables, #routes: a started program's destination taken onto its ports.  Over the
   cables paired with its node, a flow a cable in the cable order both ends read alike, flow j carrying the program's
   invocations t with t mod n = j, each asking for queue pairs of its own (#lanes); else along a route, which the
   pair's lower node chooses on its map and both ends take: the lower's flows, one a stripe of its route's
   (route_stripes), each on the lanes of the cable of the route's first hop it rides (stripe_cable), each READY
   carrying the route; the higher's program awaits the lower's READYs (program_await), each READY of its going back
   along the relays that READY made.  Its flows' sides the client's, a READY on each, its receives then posted
   (flow_step).  Refused as a value naming the node where it is this node (EINVAL) or no route reaches it (EHOSTUNREACH,
   the lower end's), and naming this node where its table is full (E2BIG) or a flow of its key toward the node runs for
   another program (EBUSY) */
static void program_take(struct mesh_client *client,uint64_t owner,uint32_t d){
  const uint32_t node=client->tx[d].node,slots=client->tx[d].slots,queues=program_queues(owner,d);
  if(!node || node==M->node){status_failed(&client->status[d],owner,EINVAL,4,node);return;}
  struct port *cable[MESH_PORTS];
  const uint32_t n=cables(node,cable);
  atomic_store_explicit(&client->status[d].phase,MESH_PAIRING,memory_order_relaxed);
  /* #routes: the peer's routed READY held here while this end sees cables to it: the two ends' maps differ */
  for(struct flow *h=flows;n && h;h=h->next){
    uint32_t j=0;
    if(!flow_held(h,client->key,node))continue;
    while(j<n && h->port!=cable[j])j++;
    if(j<n)continue;
    say("link %u (%s): program %016llx: node %u's READY came routed while this end has cables to it (the two ends' maps "
        "differ): refused\n",h->port->index,h->port->name,(unsigned long long)client->key,node);
    flow_fail(h,EPROTO,4,FAIL,M->node,M->node);
    program_fail(owner,d,EPROTO,M->node);
    return;
  }
  for(uint32_t j=0;j<n;j++){
    struct flow *f=flow_on(cable[j],client->key,M->node,node,j);
    if(f && !f->hold){program_fail(owner,d,EBUSY,M->node);return;}
    if(f && (f->stripes!=n || f->queues!=queues || (uint32_t)f->qps!=queues*slots)){flow_fail(f,EPROTO,4,FAIL,M->node,M->node);f=NULL;}
    if(!f && !(f=flow_make(cable[j],client->key,M->node,node,queues,slots,j,n,1))){program_fail(owner,d,E2BIG,M->node);return;}
    flow_take(f,owner,d);
    ready_say(f,&node,1);
    flow_step(f);
    if(f->state==FAILED)return;
  }
  if(n)return;
  if(M->node>node){program_await(client,owner,d,node,queues,slots);return;}
  uint32_t *route=NULL;
  const uint32_t hops=mesh_route(M,M->node,node,&route);
  struct port *next[MESH_PORTS];
  const uint32_t m=hops>=2?cables(route[1],next):0;
  if(!m){free(route);status_failed(&client->status[d],owner,EHOSTUNREACH,4,node);return;}
  const uint32_t stripes=route_stripes(route,hops,m);
  for(uint32_t j=0;j<stripes;j++){
    struct flow *f=flow_make(stripe_cable(next,m,client->key,M->node,node,j),client->key,M->node,node,queues,slots,j,stripes,0);
    if(!f){free(route);program_fail(owner,d,E2BIG,M->node);return;}
    flow_take(f,owner,d);
    ready_say(f,route+1,hops-1);
    flow_step(f);
    if(f->state==FAILED)break;
  }
  free(route);
}
/* design/bridge-and-ipc.md#clients: every client slot read: a slot's process watched from the turn its slot is first
   seen (its exit retires every slot it holds, the event loop's EVFILT_PROC), one already gone retired at once, and on a
   liveness turn any client whose process exited retired; a program that ended (stopped, its client gone) leaves its
   ports; a program started is taken toward each destination it binds, once no flow of the slot's earlier client
   remains, or, where a READY of its execution held here names another digest than its own, failed with that READY's
   flow, both as values (ESTALE: program_stale).  Rows of a retired client or lease are released once no flow of one
   remains */
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
    atomic_store_explicit(&client->ports,0,memory_order_relaxed);
    uint32_t stale=0;
    for(struct flow *h=flows;h;h=h->next)
      if(h->hold && h->key==client->key && h->near==M->node && h->state!=FAILED && !h->stopping &&
         digest_apart(atomic_load_explicit(&client->digest,memory_order_relaxed),h->digest)){
        stale=h->far;flow_fail(h,ESTALE,4,FAIL,h->far,M->node);
      }
    if(stale){program_stale(owner,stale);continue;}
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
  /* #routes: a program awaiting its lower peer's READYs: dropped with its program, failed past the peer window naming
     the peer, with the stripes it took */
  for(struct await *a=awaits,*next;a;a=next){
    next=a->next;
    if(!owned(a->owner) || atomic_load(&mesh_client_at(M,a->owner)->configured)!=a->owner)await_drop(a);
    else if(now>a->deadline && !a->routed && route_held(a->far)){a->routed=1;a->deadline=a->created+MESH_PEER_NS;}
    else if(now>a->deadline)program_fail(a->owner,a->destination,a->routed?ETIMEDOUT:EHOSTUNREACH,a->far);
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

/* design/bridge-and-ipc.md#ports: the node's name, its LocalHostName (else its host name), read at the start and, on
   each liveness turn, on the finder's queue (configd answers there, never on the event loop: host_read); the event loop
   takes a change (host_taken), so a renamed node's next record carries its new name (the map's label of it; a node is
   its number) */
static void host_name(char *name){
  CFStringRef local=SCDynamicStoreCopyLocalHostName(NULL);
  if(!local || !CFStringGetCString(local,name,MESH_NAME_BYTES,kCFStringEncodingUTF8))gethostname(name,MESH_NAME_BYTES);
  if(local)CFRelease(local);
  name[MESH_NAME_BYTES-1]=0;
}
static char host_heard[MESH_NAME_BYTES];
static _Atomic int host_reading,host_read_done;
static void host_read(void){
  if(atomic_load(&host_read_done) || atomic_exchange(&host_reading,1))return;
  dispatch_async(finder,^{
    char name[MESH_NAME_BYTES]={0};
    host_name(name);
    memcpy(host_heard,name,sizeof name);
    atomic_store_explicit(&host_read_done,1,memory_order_release);
    atomic_store(&host_reading,0);
    wake();
  });
}
static void host_taken(void){
  if(!atomic_load_explicit(&host_read_done,memory_order_acquire))return;
  if(strcmp(host_heard,host)){
    say("node %u: renamed %s to %s\n",M->node,host,host_heard);
    memcpy(host,host_heard,sizeof host);
    record_changed();
  }
  atomic_store_explicit(&host_read_done,0,memory_order_release);
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
  const uint32_t pg=(uint32_t)getpagesize();
  /* design/collective-dependency-ledger.md#d6-paired-send-and-receive-frame-counts-match */
  if((uint64_t)pg+4096>16773120)die("transport chunk exceeds native request capacity");
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
  *M=geometry;M->node=node;M->version=MESH_VERSION;M->boot=clock_gettime_nsec_np(CLOCK_REALTIME);M->idle_since=M->boot;
  for(uint32_t r=0;r<mesh_rows(M);r++)atomic_store_explicit(&mesh_page(M)[r].mapping,MESH_ABSENT,memory_order_relaxed);
  if(wire_map(&wire,M,fd))die("transport page aliases");
  close(fd);
  bridge.M=M;
  host_name(host);
  finder=dispatch_queue_create("mesh.ports",DISPATCH_QUEUE_SERIAL);
  /* design/bridge-and-ipc.md#records: the node's memory and hardware model, which its record carries with its window */
  size_t size=sizeof memory;
  if(sysctlbyname("hw.memsize",&memory,&size,NULL,0))memory=0;
  sysctl_text("hw.model",model,sizeof model);
  events=kqueue();
  health_init();
  record_changed();
  if(!atomic_load(&M->map_nodes))die("the map's lease");
  lanes_reserve();
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
  int rebuilt=0;
  while(!stop){
    now=clock_ns();
    if(atomic_load_explicit(&found,memory_order_acquire))ports_found();
    for(uint32_t p=0;p<port_count;p++){
      struct port *port=ports+p;
      if(port->job && atomic_load_explicit(&port->done,memory_order_acquire))port_opened(port);
      if((port->connecting || (port->control>=0 && !port->greeted)) && now>port->next)port_lost(port,ETIMEDOUT,0);
      port_turn(port);
      lanes_turn(port);
    }
    flows_turn();
    health_turn();
    host_taken();
    int alive=now>=liveness;
    if(alive){
      liveness=now+MESH_PAUSE_NS;
      host_read();
      /* design/bridge-and-ipc.md: a bridge whose binary changed stops between calls, once it holds no flow and no
         awaiting program (`busy` 0) */
      if(!rebuilt && binary.st_ino && (stat(argv[0],&current) || current.st_ino!=binary.st_ino || current.st_mtimespec.tv_sec!=binary.st_mtimespec.tv_sec)){
        rebuilt=1;
        if(flow_count || await_count)say("bridge node %u: its binary changed; it stops once no program runs through it\n",node);
      }
      if(rebuilt && !flow_count && !await_count){stop=1;stop_reason=MESH_STOP_BINARY;}
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
      atomic_store_explicit(&port->lanes.up,0,memory_order_release);
      port_settle(port);
    });
    port_still(port);
    dispatch_sync(port->work,^{
      if(!down_pair(&port->lanes.provider))say("link %u (%s): lanes teardown: %s\n",port->index,port->name,strerror(errno));
      if(!down_device(&port->device))say("link %u (%s): verbs teardown: %s\n",port->index,port->name,strerror(errno));
    });
  }
  atomic_store(&M->bridge_pid,0);
  mesh_notify(&M->control);
  pthread_join(watcher,NULL);
  munmap(wire.data,wire.length);munmap(M,length);
  return 0;
}
