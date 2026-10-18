#include "mesh-verbs.h"
#include "mesh-call.h"
#include <dispatch/dispatch.h>
#include <SystemConfiguration/SystemConfiguration.h>
#include <sys/event.h>
#include <sys/ioctl.h>
#include <sys/kern_event.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <pwd.h>
#include <libproc.h>
#include <spawn.h>
#include <signal.h>
#include <ctype.h>
#include <limits.h>
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
/* design/prepared-machine.md#M08: one chunk of a receive transfer in one invocation: its request and SGE into the chunk's
   rows, the word it stores (`input`; its own `sink` where no reader waits on its landing), the argument it stores plus
   the cycle, its frames, its invocation, ring and cycle, the record `behind` it in its ring's running order, whether it
   releases a SEND cell (a relay's), and its name (chunk, chunks, binding and position: design/bridge-and-ipc.md#ledger) */
struct prepared_receive {
  _Alignas(128) struct ibv_recv_wr request;
  struct ibv_sge span;
  _Atomic uint64_t *input;
  uint64_t argument;
  uint32_t frames,invocation,ring,cycle,behind,release,chunk,chunks;
  _Atomic uint64_t sink;
  uint32_t binding,position;
};
_Static_assert(sizeof(struct prepared_receive)==128 && _Alignof(struct prepared_receive)==128 &&
  offsetof(struct prepared_receive,span)==32 && offsetof(struct prepared_receive,input)==48,"M08 M09");
/* design/prepared-machine.md#M08, #M30: a lease's receive ring: its records in invocation, binding, chunk order; `posted`
   the next to post, `outstanding` its frames posted and not landed, within `capacity`; a record is posted once the
   record `behind` it in the ring's running order has landed, and a cyclic ring wraps (record posted % count in cycle
   posted / count) */
struct ring {
  struct prepared_receive *first;
  size_t count,posted;
  uint64_t landed,bytes;
  uint32_t outstanding,capacity,span,depth;
};
/* design/prepared-machine.md#M04: a lease's SEND stream, the send thread's: its current cell, the next request of that
   cell's chain and the requests left, its cycle (M30), and the cells it posted whole */
struct stream { struct mesh_send *cell,*first; struct ibv_send_wr *next; uint32_t remaining; uint64_t cycle,sent; };

/* design/bridge-and-ipc.md#wire: the bridge's whole wire vocabulary.  On a link's control socket: HELLO (the link's
   bootstrap), PAIRS (queue pair numbers, TN3205's one out-of-band step), FAREWELL (a stopping bridge's reason), RECORD
   (a node's record, flooded) and the request channels (RFC 4254 s5); on a link's keepalive queue pair, KEEPALIVE.  A
   member added here is a design change that needs an operator quote in design/bridge-and-ipc.md#wire. */
enum wire { HELLO=1, PAIRS, FAREWELL, RECORD, CHANNEL_OPEN, CHANNEL_CONFIRM, CHANNEL_REFUSE, CHANNEL_DATA, CHANNEL_WINDOW,
            CHANNEL_SHUT, CHANNEL_CLOSE, KEEPALIVE };
struct message { uint32_t magic,kind,node,count; uint64_t key; int64_t code; uint32_t block,port,from,to,bytes; uint16_t lid,pad; uint8_t gid[16]; };
#define XMAGIC (0x4d595048u+MESH_VERSION)
/* a PAIRS entry: queue pair `index` of the link made for the `round`th time, its number and first sequence number, the
   code of the program its holder lost with it (0: none, or one that ended in order) and the SENDs its holder completed
   on the making before (`sent`: the landings the peer's holder takes before it lets go in its turn) */
struct pairs_entry { uint32_t index,qpn,psn,round; int64_t code; uint64_t sent; };

/* design/bridge-and-ipc.md#links: a link's queue pairs, the keepalive's (index 0) and its leases' (lease k is pair
   k + 1); a completion's wr_id names its pair, the pair's making (`round`, its low 16 bits) and its record or gate
   position, so a completion of a pair made again since is none of its holder's */
#define LINK_PAIRS (1+MESH_LEASES)
#define KEEPALIVE_SLOTS 16u
#define KEEPALIVE_FRAME 64u
#define WR_LOW ((UINT64_C(1)<<40)-1)
static uint64_t wr_id(uint32_t round,uint32_t index,uint64_t low){ return (uint64_t)(round&0xffff)<<48|(uint64_t)(index&0xff)<<40|(low&WR_LOW); }
static uint32_t wr_index(uint64_t id){ return (uint32_t)(id>>40&0xff); }
static uint32_t wr_round(uint64_t id){ return (uint32_t)(id>>48); }
struct path;
struct ledger;
/* a queue pair of a link: its making (`round`), first sequence number and capacities, the peer's of the same index as
   last told (its number, first sequence number and round), whether the two are connected, whether the peer let go of
   it in order while a program here holds it and the SENDs its holder completed (`peer_sent`, UINT64_MAX none); the
   program path and ring that hold it (set under the port's guard), its gate (the send thread's: frames posted and
   retired through its SEND completions, within `gate`), and an error a progress thread took on it, for the event loop */
struct pair {
  struct ibv_qp *qp;
  uint32_t round,psn,sends,receives,gate;
  uint32_t peer_qpn,peer_psn,peer_round;
  int heard,connected,peer_gone;
  _Atomic uint64_t peer_sent;
  struct path *path;
  uint32_t ring;
  uint64_t posted,retired,requests,completions;
  _Atomic int64_t failed;
};
/* design/bridge-and-ipc.md#programs: a started program's side toward one destination: its rings' streams and receive
   rings, each riding its lease (`pair` the pair index on port `port`, 0 none), the rings it uses, its ledger where the
   configuration keeps them, whether it closes in order (its program ended in order, or a relay's other side let go),
   and its rings whose SENDs all posted and completed once it closes (`drained`, the send thread's) */
struct program;
struct path {
  struct program *program;
  uint32_t destination,node,port,rings;
  uint32_t pair[MESH_QUEUES];
  struct ring ring[MESH_QUEUES];
  struct stream stream[MESH_QUEUES];
  struct prepared_receive *receive;
  struct prepared_send *requests;
  size_t receive_count;
  struct ledger *ledger;
  _Atomic int closing;
  _Atomic uint32_t drained;
};
enum { RUNNING=1, ENDING, FAILED };
/* a started program: its slot's owner, whether it is a relay (the bridge's own slot, held by a relay channel), its state,
   its paths, its invocations and whether it is cyclic */
struct program {
  struct program *next;
  uint64_t owner;
  int relay;
  _Atomic int state;
  uint32_t invocations,cyclic,path_count;
  struct path *paths;
};
/* design/bridge-and-ipc.md#ports: a port, its link and the link's queue pairs: its device, the node it reaches and the
   index that node gave the cable, its control socket's queues each way, its verbs work's queue (the window's
   registration), the link's two completion queues and its pairs (`made` of them, the link `count`, each its `share` of
   the completion queues), its keepalive frames' page, the paths attached, and its two progress threads (`up` while the
   link is paired; `guard` odd while the event loop changes the link's verbs objects or what holds them, each thread's
   `quiet` the guard value it holds at; `bell` the word they wait on while the link is down or the guard stands) */
struct port {
  struct mesh_device device;
  uint32_t index,interface,peer,peer_port,greeted,parted,connecting,writing,retiring,open,wanted;
  int64_t farewell;
  char name[IFNAMSIZ];
  int control,job,result,refused;
  _Atomic int done;
  uint64_t bandwidth;
  struct message heard;
  uint16_t lid;
  uint8_t gid[16];
  unsigned char *in,*out;
  size_t held,in_room,out_length,out_sent,out_room;
  dispatch_queue_t work;
  struct ibv_cq *sent,*received;
  uint32_t made,count,share,page,holding;
  struct pair pair[LINK_PAIRS];
  _Atomic int up;
  _Atomic uint64_t guard,bell,quiet[2];
  pthread_t thread[2];
  int threads;
  uint64_t keepalive_at,keepalives,keepalives_heard;
};
static struct hdr *M;
static struct wire_mapping wire;
static struct port ports[MESH_PORTS];
static uint32_t port_count;
static struct program *programs;
static int events=-1,listener=-1,hello=-1,network=-1,unreleased,finding;
static uint64_t sequence;
#define HELLO_TIMER 1
static _Atomic int pressured;
static char host[MESH_NAME_BYTES];
static struct mesh_ctx bridge={.fd=-1};
static uint32_t map_page,map_pages,map_room;
static dispatch_queue_t finder;
static _Atomic int found;
static char devices[MESH_PORTS][32];
static uint32_t device_count;
static const struct in6_addr everyone={{{0xff,0x02,0,0,0,0,0,0,0,0,0,0,0,0,0,1}}};
static uint64_t keepalive_ns,hello_ns,health_ns;
static int ledger_kept;
#ifdef MESH_LOOP_FABRIC
/* design/bridge-dataflow.md#the-loopback-fabric: a loop bridge's configuration's `loop`: its region's name, its node,
   the loopback port its clients' requests come to (`service`, in place of MESH_PORT), the loopverbs fabric it shares
   with the host's other loop bridges (loopverbs.c loop_fabric), and its ports, each a loopverbs device and a cable: the
   loopback port this end listens on for it and the peer end's, the lower of the two dialing the higher */
void loop_fabric(const char *name);
struct loop_port { char device[32]; uint16_t listen,peer; int fd; };
static struct { char region[32],fabric[64]; uint32_t node,ports; uint16_t service; struct loop_port port[MESH_PORTS]; } loop_config;
#endif
static uint64_t now_ns(void){ return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static void bell_ring(_Atomic uint64_t *word){
  atomic_fetch_add_explicit(word,1,memory_order_release);
  os_sync_wake_by_address_all(word,sizeof *word,OS_SYNC_WAKE_BY_ADDRESS_NONE);
}
static uint64_t memory;
/* design/bridge-and-ipc.md#records: the bytes the kernel can give without pressure (kern.memorystatus_level percent of
   the node's memory), read now */
static uint64_t available_now(void){
  int level=0;size_t size=sizeof level;
  return sysctlbyname("kern.memorystatus_level",&level,&size,NULL,0) || level<0?0:(uint64_t)level*(memory/100);
}


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
   cheap reads only: sysctl, the host's VM statistics and the notify state of thermal pressure; the power sources'
   description read at the start and at each of their notifications (the event loop's, power_fd) */
static struct mesh_health health;
static int power_token=-1,thermal_token=-1,power_fd=-1,thermal_fd=-1;
static mach_port_t host_port;
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
  h->available=available_now();
  int level=0;size_t size=sizeof level;
  if(!sysctlbyname("kern.memorystatus_vm_pressure_level",&level,&size,NULL,0))h->pressure=(uint32_t)level;
  uint64_t state=0;
  if(thermal_token>=0 && notify_get_state(thermal_token,&state)==NOTIFY_STATUS_OK)h->thermal=(uint32_t)state;
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
  notify_register_file_descriptor(kIOPSNotifyAnyPowerSource,&power_fd,0,&power_token);
  notify_register_file_descriptor("com.apple.system.thermalpressurelevel",&thermal_fd,0,&thermal_token);
  if(power_fd>=0)fcntl(power_fd,F_SETFL,fcntl(power_fd,F_GETFL)|O_NONBLOCK);
  if(thermal_fd>=0)fcntl(thermal_fd,F_SETFL,fcntl(thermal_fd,F_GETFL)|O_NONBLOCK);
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
  snprintf(out,room,"power %s; memory available %.2f GiB, free %.2f GiB, wired %.2f GiB, pressure %s; thermal %s; booted %s "
           "(session %.36s), last restart %s (%.79s)",power,(double)h->available/1073741824.0,(double)h->free/1073741824.0,
           (double)h->wired/1073741824.0,
           h->pressure==4?"critical":h->pressure==2?"warning":h->pressure==1?"normal":"unread",thermal,when,h->session,
           h->reset<5?resets[h->reset]:"unknown",h->shutdown);
}

/* design/recovery.md#diagnosis; design/bridge-and-ipc.md#records: the nodes this bridge has heard of, whether it heard
   each one's record and lost it since (`gone`, the map's prune), and the newest record it ever heard of each, held apart
   from the map: RFC 2328 s13's database, so a record the map dropped is never taken, nor sent on, again.  A full table
   gives its place the first entry whose node the map no longer holds.  Records of one node are ordered by their
   sequence, the greater bytes newer among equal sequences (RFC 2328 s13.1's checksum, record_newer) */
struct known { uint32_t node; int heard,gone; struct mesh_record record; };
#define MESH_KNOWN 256
static struct known known[MESH_KNOWN];
static uint32_t known_count;
static struct mesh_record *record_held(uint32_t node);
static struct known *known_at(uint32_t node){
  for(uint32_t i=0;i<known_count;i++)if(known[i].node==node)return known+i;
  struct known *at=NULL;
  if(known_count<MESH_KNOWN)at=known+known_count++;
  else for(uint32_t i=0;i<known_count && !at;i++)if(!record_held(known[i].node))at=known+i;
  if(at)*at=(struct known){.node=node};
  return at;
}
static int record_newer(const struct mesh_record *a,const struct mesh_record *b){
  return a->sequence>b->sequence || (a->sequence==b->sequence && memcmp(a,b,sizeof *a)>0);
}
static void wake(void){
  struct kevent64_s event;EV_SET64(&event,0,EVFILT_USER,0,NOTE_TRIGGER,0,0,0,0);
  kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
}

/* design/bridge-and-ipc.md#ledger (Q5P's M27 trace, mesh git 5702ea5): where the configuration keeps ledgers, each path's
   SEND cells as its send thread posts them (the first request's time, then the last's) and its records as they land,
   stamped where the data path passes them; a request names a program's key and a file its paths' events since the
   latest such request are appended to, each path's after its header */
#define MESH_LEDGER_EVENTS (UINT32_C(1)<<16)
struct ledger_send { uint64_t cell,seen,posted; uint32_t cycle,pad; };
struct ledger_landing { uint64_t record,landed; uint32_t cycle,pad; };
struct ledger_cell { uint64_t cell,bytes; uint32_t ring,invocation,position,binding,chunks,pad; };
struct ledger {
  uint32_t id,cells,cut_sends,cut_landings,current[MESH_QUEUES];
  struct ledger_cell *cell;
  struct ledger_send *send;
  struct ledger_landing *landing;
  _Atomic uint32_t sends,landings;
};
static _Atomic uint32_t ledger_ids;
static void ledger_seen(struct ledger *l,uint32_t ring,const struct mesh_send *cell,uint64_t cycle){
  const uint32_t e=atomic_load_explicit(&l->sends,memory_order_relaxed);
  l->send[e%MESH_LEDGER_EVENTS]=(struct ledger_send){.cell=(uintptr_t)cell,.seen=now_ns(),.cycle=(uint32_t)cycle};
  l->current[ring]=e;
  atomic_store_explicit(&l->sends,e+1,memory_order_release);
}
static void ledger_posted(struct ledger *l,uint32_t ring){ l->send[l->current[ring]%MESH_LEDGER_EVENTS].posted=now_ns(); }
static void ledger_landed(struct ledger *l,const struct prepared_receive *record){
  const uint32_t e=atomic_load_explicit(&l->landings,memory_order_relaxed);
  l->landing[e%MESH_LEDGER_EVENTS]=(struct ledger_landing){(uintptr_t)record,now_ns(),record->cycle,0};
  atomic_store_explicit(&l->landings,e+1,memory_order_release);
}
static int ledger_cell_order(const void *a,const void *b){
  const uint64_t x=((const struct ledger_cell *)a)->cell,y=((const struct ledger_cell *)b)->cell;
  return (x>y)-(x<y);
}

/* design/bridge-and-ipc.md#clients: a status told, and its client's start woken */
static int owned(uint64_t owner){ return owner && atomic_load(&mesh_client_at(M,owner)->owner)==owner; }
static void status_prepared(struct mesh_port_info *status,uint64_t owner){
  atomic_store_explicit(&status->prepared,owner,memory_order_release);
  os_sync_wake_by_address_all(&status->prepared,sizeof status->prepared,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
static void status_failed(struct mesh_port_info *status,uint64_t owner,int64_t code,uint32_t domain,uint32_t where){
  atomic_store_explicit(&status->code,code,memory_order_relaxed);atomic_store_explicit(&status->domain,domain,memory_order_relaxed);
  atomic_store_explicit(&status->failed,where,memory_order_relaxed);atomic_store_explicit(&status->phase,MESH_STOPPED,memory_order_relaxed);
  status_prepared(status,owner);
}

/* design/prepared-machine.md#M01: the ring slot a transfer's invocation t addresses: (t - begin) mod its depth */
static uint32_t transfer_row(uint32_t fallback,const struct mesh_transfer *transfer,uint32_t row,uint32_t invocation,uint32_t chunk){
  uint32_t depth=transfer->depth?transfer->depth:fallback;
  if(!depth)depth=1;
  return row+(uint32_t)((size_t)((invocation-transfer->begin)%depth)*transfer->invocation_pages/M->block)+chunk;
}
/* design/prepared-machine.md#M09: a chunk's span in the window the port's device registered; EINVAL outside it */
static int chunk_span(const struct port *port,uint32_t address,uint64_t remaining,struct ibv_sge *span){
  const uint64_t offset=atomic_load_explicit(&mesh_page(M)[address].address,memory_order_relaxed)-M->data_off;
  if(offset>=wire.length)return EINVAL;
  *span=wire_span(&port->device,(uint64_t)M->block*M->pgsz,offset,remaining);
  return 0;
}
/* design/prepared-machine.md#M07, #M08, #M12: the word a landed chunk of invocation u of the transfer's slot `index`
   stores: its slot's completion word where the row's delivery names one among the program's own toward the
   destination (`words`, its cancellation range); or, on a row the receiving program also publishes (a relay's rows),
   that slot's SEND cell once the record's last chunk landed, so the landing releases it to the next hop (*release
   set); else the record's own `sink`.  NULL where the word is not in the window */
static _Atomic uint64_t *receive_input(uint64_t owner,const struct mesh_cancel_range *words,uint32_t row,uint32_t index,uint32_t chunk,
                                       uint32_t chunks,uint32_t u,_Atomic uint64_t *sink,uint32_t *release){
  const struct mesh_delivery *delivery=mesh_delivery_at(M,row+chunk);
  uint64_t at=0;
  const uint64_t word=delivery->device_input+delivery->device_stride*u+sizeof(uint64_t)*(uint64_t)index*chunks;
  *release=0;
  if(delivery->device_input && words && word>=words->offset && (word-words->offset)/sizeof(uint64_t)<words->count)at=word;
  else if(chunk+1==chunks){
    uint32_t sends;
    const struct mesh_target *target=mesh_row_targets(M,owner,row,&sends);
    if(index<sends){at=target[index].cells+sizeof(struct mesh_send)*(uint64_t)u;*release=1;}
  }
  if(!at)return sink;
  return at>=M->data_off && at<=M->length-sizeof(uint64_t)?(void *)((char *)M+at):NULL;
}

/* design/prepared-machine.md#M04, #M06, #M29, #M30: a path's SEND streams, one a ring (queue k's slot s is ring
   k slots + s): every cell of the destination's transfers, its chain of chunk requests over its invocation's ring slot,
   in invocation order and binding order within one, as the peer's ring posts its receives (TN3205's credit holds a
   SEND until its receive is posted); a cyclic program's last cell followed by its first, a one-pass program's by its
   column's terminal cell, which no producer releases.  0, or EINVAL (a transfer of more slots than the program's, or
   a ring past MESH_QUEUES), ENOMEM */
static int sends_configure(struct program *prog,struct path *path,struct mesh_client *client,const struct port *port){
  struct mesh_tx *tx=&client->tx[path->destination];
  const uint32_t invocations=tx->invocations?tx->invocations:1,fallback=atomic_load_explicit(&client->depth,memory_order_acquire),
    slots=tx->slots?tx->slots:1;
  size_t requests=0,cells=0;
  for(uint32_t k=0;k<MESH_QUEUES;k++){
    const uint32_t channel=path->destination*MESH_QUEUES+k,count=atomic_load(mesh_order_length(M,prog->owner,channel,MESH_SEND));
    const struct mesh_transfer *out=mesh_transfers(M,prog->owner,channel,MESH_SEND);
    for(uint32_t i=0;i<count;i++){
      if(out[i].count>slots || (uint64_t)k*slots+out[i].count>MESH_QUEUES)return EINVAL;
      const uint32_t active=mesh_transfer_active(out+i,invocations);
      requests+=(size_t)(mesh_row_chunks(M,out[i].local_row,out[i].bytes)-1)*out[i].count*active;
      cells+=(size_t)out[i].count*active;
    }
  }
  path->requests=aligned_alloc(128,(requests+1)*sizeof *path->requests);
  if(!path->requests)return ENOMEM;
  struct ledger *ledger=path->ledger;
  if(ledger && !(ledger->cell=malloc((cells+1)*sizeof *ledger->cell)))ledger=NULL;
  struct mesh_send *publications=(void *)((char *)M+tx->cells),*last[MESH_QUEUES]={0},*terminal[MESH_QUEUES]={0};
  size_t next=0;
  int error=0;
  for(uint32_t k=0;k<MESH_QUEUES && !error;k++){
    const uint32_t channel=path->destination*MESH_QUEUES+k,count=atomic_load(mesh_order_length(M,prog->owner,channel,MESH_SEND));
    const struct mesh_transfer *out=mesh_transfers(M,prog->owner,channel,MESH_SEND);
    for(uint32_t i=0;i<count && !error;i++){
      const uint32_t active=mesh_transfer_active(out+i,invocations),column=active+1;
      for(uint32_t slot=0;slot<out[i].count && !error;slot++){
        const uint32_t row=out[i].local_row+slot*out[i].stride,chunks=mesh_row_chunks(M,row,out[i].bytes),r=k*slots+slot;
        struct mesh_send *column_cells=publications+out[i].first+(size_t)slot*column;
        for(uint32_t u=0;u<active && !error;u++){
          struct mesh_send *cell=column_cells+u;
          cell->pair=0;cell->queue=r;cell->chunks=chunks;
          uint64_t remaining=out[i].bytes;
          for(uint32_t c=0;c<chunks && !error;c++){
            struct ibv_sge span;
            if((error=chunk_span(port,transfer_row(fallback,out+i,row,out[i].begin+u,c),remaining,&span)))break;
            remaining-=span.length;
            struct ibv_sge *entry=c?&path->requests[next+c-1].span:&cell->span;
            struct ibv_send_wr *request=c?&path->requests[next+c-1].request:&cell->request;
            *entry=span;
            *request=(struct ibv_send_wr){.wr_id=c?0:1,.next=c+1==chunks?NULL:&path->requests[next+c].request,
              .sg_list=entry,.num_sge=1,.opcode=IBV_WR_SEND};
          }
          next+=chunks-1;
        }
        terminal[r]=column_cells+active;
        path->rings|=1u<<r;
      }
    }
    uint32_t position[MESH_QUEUES]={0},latest[MESH_QUEUES];
    for(uint32_t r=0;r<MESH_QUEUES;r++)latest[r]=UINT32_MAX;
    for(uint32_t t=0;t<invocations && !error;t++)for(uint32_t i=0;i<count;i++){
      const uint32_t active=mesh_transfer_active(out+i,invocations);
      if(t<out[i].begin || t-out[i].begin>=active)continue;
      for(uint32_t slot=0;slot<out[i].count;slot++){
        const uint32_t r=k*slots+slot;
        struct mesh_send *cell=publications+out[i].first+(size_t)slot*(active+1)+(t-out[i].begin);
        if(last[r])last[r]->successor=(uintptr_t)cell;
        else path->stream[r].cell=cell;
        last[r]=cell;
        if(latest[r]!=t){latest[r]=t;position[r]=0;}
        if(ledger && ledger->cells<cells)
          ledger->cell[ledger->cells++]=(struct ledger_cell){(uintptr_t)cell,out[i].bytes,r,t,position[r]++,out[i].binding,cell->chunks,0};
      }
    }
  }
  for(uint32_t r=0;r<MESH_QUEUES && !error;r++)if(last[r])last[r]->successor=prog->cyclic?(uintptr_t)path->stream[r].cell:(uintptr_t)terminal[r];
  for(uint32_t r=0;r<MESH_QUEUES;r++){path->stream[r].first=path->stream[r].cell;path->stream[r].cycle=0;}
  if(ledger && ledger->cells)qsort(ledger->cell,ledger->cells,sizeof *ledger->cell,ledger_cell_order);
  return error;
}

/* design/prepared-machine.md#M08, #M30: a path's receive rings, one a ring: each ring's records invocation by
   invocation, the transfers of each in binding order, a transfer's chunks in order, each storing its word
   (receive_input); a record posted once the record of its transfer `span` invocations before has landed (the ring's
   span held a slot short of its slots, and under its invocations in a cyclic program, whose first records wait on its
   last of the cycle before).  0, or EINVAL (a transfer of more slots than the program's, a ring past MESH_QUEUES, a word
   outside the window), ENOMEM */
static int receives_configure(struct program *prog,struct path *path,struct mesh_client *client,const struct port *port){
  if(client->target_count && (client->targets<M->data_off || client->targets>M->length ||
     client->target_count>(M->length-client->targets)/sizeof(struct mesh_target)))return EINVAL;
  const uint32_t d=path->destination;
  struct mesh_tx *tx=&client->tx[d];
  const uint64_t range=sizeof(struct mesh_cancellation)+sizeof(struct mesh_cancel_range)*((uint64_t)d+1);
  const struct mesh_cancellation *cancel=tx->cancel>=M->data_off && tx->cancel<=M->length-range?(const void *)((char *)M+tx->cancel):NULL;
  const struct mesh_cancel_range *words=cancel?cancel->ranges+d:NULL;
  const uint32_t invocations=tx->invocations?tx->invocations:1,fallback=atomic_load_explicit(&client->depth,memory_order_acquire),
    slots=tx->slots?tx->slots:1;
  uint32_t depth=fallback;
  if(!depth || depth>invocations)depth=invocations;
  size_t records=0;
  for(uint32_t k=0;k<MESH_QUEUES;k++){
    const uint32_t channel=d*MESH_QUEUES+k,count=atomic_load(mesh_order_length(M,prog->owner,channel,MESH_RECEIVE));
    const struct mesh_transfer *in=mesh_transfers(M,prog->owner,channel,MESH_RECEIVE);
    for(uint32_t i=0;i<count;i++){
      if(in[i].count>slots || (uint64_t)k*slots+in[i].count>MESH_QUEUES)return EINVAL;
      records+=(uint64_t)mesh_row_chunks(M,in[i].local_row,in[i].bytes)*in[i].count*mesh_transfer_active(in+i,invocations);
    }
  }
  path->receive=aligned_alloc(128,(records+1)*sizeof *path->receive);
  if(!path->receive)return ENOMEM;
  path->receive_count=records;
  size_t at=0;
  int error=0;
  for(uint32_t r=0;r<MESH_QUEUES && !error;r++){
    const uint32_t k=r/slots,slot=r%slots,count=k<MESH_QUEUES?atomic_load(mesh_order_length(M,prog->owner,d*MESH_QUEUES+k,MESH_RECEIVE)):0;
    const struct mesh_transfer *in=count?mesh_transfers(M,prog->owner,d*MESH_QUEUES+k,MESH_RECEIVE):NULL;
    struct ring *ring=&path->ring[r];
    *ring=(struct ring){.first=path->receive+at,.span=UINT32_MAX,.depth=invocations};
    size_t *base=calloc((size_t)count+1,sizeof *base),*firsts=NULL;
    for(uint32_t i=0;base && i<count;i++)base[i+1]=base[i]+(slot<in[i].count?mesh_transfer_active(in+i,invocations):0);
    if(base)firsts=malloc((base[count]+1)*sizeof *firsts);
    if(!base || !firsts){free(base);free(firsts);return ENOMEM;}
    for(uint32_t i=0;i<count;i++){
      if(slot>=in[i].count)continue;
      const uint32_t active=mesh_transfer_active(in+i,invocations),own=in[i].depth?in[i].depth:depth;
      ring->depth=MIN(ring->depth,own);
      if(own<active)ring->span=MIN(ring->span,own>1?own-1:1);
    }
    uint32_t position=0,latest=UINT32_MAX;
    for(uint32_t t=0;t<invocations && !error;t++)for(uint32_t i=0;i<count && !error;i++){
      const uint32_t active=mesh_transfer_active(in+i,invocations);
      if(slot>=in[i].count || t<in[i].begin || t-in[i].begin>=active)continue;
      if(latest!=t){latest=t;position=0;}
      const uint32_t u=t-in[i].begin,row=in[i].local_row+slot*in[i].stride,chunks=mesh_row_chunks(M,row,in[i].bytes);
      uint64_t remaining=in[i].bytes;
      for(uint32_t c=0;c<chunks;c++){
        struct ibv_sge span;
        if((error=chunk_span(port,transfer_row(fallback,in+i,row,t,c),remaining,&span)))break;
        remaining-=span.length;
        struct prepared_receive *record=path->receive+at++;
        if(!c)firsts[base[i]+u]=(size_t)(record-ring->first);
        uint32_t release;
        *record=(struct prepared_receive){.request={.sg_list=&record->span,.num_sge=1},.span=span,
          .input=receive_input(prog->owner,words,row,in[i].stride?0:slot,c,chunks,u,&record->sink,&release),.argument=1,
          .frames=(span.length+4095)/4096,.invocation=t,.ring=r,.chunk=c,.chunks=chunks,.binding=in[i].binding,.position=position};
        record->release=release;
        if(!record->input){error=EINVAL;break;}
      }
      position++;
    }
    ring->count=(size_t)(path->receive+at-ring->first);
    if(prog->cyclic)ring->span=MIN(ring->span,invocations>1?invocations-1:1);
    for(uint32_t i=0;i<count && !error;i++){
      const uint32_t own=(uint32_t)(base[i+1]-base[i]);
      if(!own)continue;
      const uint32_t chunks=mesh_row_chunks(M,in[i].local_row+slot*in[i].stride,in[i].bytes),span=ring->span;
      for(uint32_t j=0;j<own;j++){
        const size_t here=firsts[base[i]+j];
        size_t behind=0;
        if(span!=UINT32_MAX && j>=span)behind=here-firsts[base[i]+j-span];
        else if(span!=UINT32_MAX && prog->cyclic && j+own>=span)behind=here+ring->count-firsts[base[i]+j+own-span];
        else if(span!=UINT32_MAX && prog->cyclic)behind=ring->count;
        for(uint32_t c=0;c<chunks;c++)ring->first[here+c].behind=(uint32_t)behind;
      }
    }
    free(base);free(firsts);
    if(ring->count)path->rings|=1u<<r;
  }
  if(at)path->receive[at]=path->receive[at-1];
  return error;
}

/* design/prepared-machine.md#M08, #M30: the ring's next records posted on its lease while its capacity holds them and the
   record behind each has landed; called as the ring is attached and at each of its landings */
static int ring_advance(struct pair *pair,struct ring *ring,int cyclic){
  int (*post)(struct ibv_qp *,struct ibv_recv_wr *,struct ibv_recv_wr **)=pair->qp->context->ops.post_recv;
  while(ring->count && (cyclic || ring->posted<ring->count)){
    struct prepared_receive *record=ring->first+ring->posted%ring->count;
    if((record->behind && ring->posted>=record->behind && ring->landed<=ring->posted-record->behind) ||
       ring->outstanding+record->frames>ring->capacity)break;
    record->cycle=(uint32_t)(ring->posted/ring->count);
    struct ibv_recv_wr *bad;
    const int error=post(pair->qp,&record->request,&bad);
    if(error)return error<0?-error:error;
    ring->outstanding+=record->frames;ring->posted++;
  }
  return 0;
}

/* design/bridge-and-ipc.md#loops: a thread's error on a pair (an error completion, a post refused), the event loop's to
   take (pairs_turn) */
static void wake(void);
static void pair_failed(struct pair *pair,int64_t code){
  int64_t none=0;
  if(atomic_compare_exchange_strong(&pair->failed,&none,code?code:EIO))wake();
}

/* design/bridge-and-ipc.md#loops: one pass of a port's receive thread: the link's receive completions taken, each
   landing storing its record's word (argument plus cycle, M30) and posting its ring's next records, each keepalive
   frame's slot posted again; a completion of a pair made again since is skipped.  The completions it took */
static int receive_pass(struct port *port){
  struct ibv_wc done[32];
  const int count=port->received->context->ops.poll_cq(port->received,32,done);
  if(count<0){pair_failed(port->pair,EIO);return 0;}
  for(int i=0;i<count;i++){
    const uint64_t id=done[i].wr_id;
    const uint32_t k=wr_index(id);
    if(k>=port->made)continue;
    struct pair *pair=port->pair+k;
    if(wr_round(id)!=(pair->round&0xffff))continue;
    if(done[i].status){pair_failed(pair,(int64_t)done[i].status);continue;}
    if(!k){
      const uint64_t slot=id&WR_LOW;
      struct ibv_sge span=wire_span(&port->device,(uint64_t)M->block*M->pgsz,
        (uint64_t)port->page*M->pgsz+(KEEPALIVE_SLOTS+slot%KEEPALIVE_SLOTS)*KEEPALIVE_FRAME,KEEPALIVE_FRAME);
      struct ibv_recv_wr request={.wr_id=id,.sg_list=&span,.num_sge=1},*bad;
      port->keepalives_heard++;
      if(pair->qp->context->ops.post_recv(pair->qp,&request,&bad))pair_failed(pair,EIO);
      continue;
    }
    struct path *path=pair->path;
    if(!path)continue;
    struct ring *ring=&path->ring[pair->ring];
    struct prepared_receive *record=ring->first+(id&WR_LOW);
    ring->landed++;ring->bytes+=done[i].byte_len;ring->outstanding-=record->frames;
    atomic_store_explicit(record->input,record->argument+record->cycle,memory_order_release);
    if(path->ledger)ledger_landed(path->ledger,record);
    const int error=ring_advance(pair,ring,(int)path->program->cyclic);
    if(error)pair_failed(pair,error);
    if(ring->landed==atomic_load_explicit(&pair->peer_sent,memory_order_relaxed))wake();
  }
  return count>0?count:0;
}

/* design/bridge-and-ipc.md#loops: a stream's released cells posted on its lease, each cell's chain request by request
   while the pair's gate holds their frames (ibv_post_send accepts past the queue's frames and corrupts later, so the
   gate is the caller's); 0, or the post's error */
static int stream_post(struct pair *pair,uint32_t k,struct stream *stream,struct ledger *ledger,uint32_t ring){
  int (*post)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **)=pair->qp->context->ops.post_send;
  for(;;){
    if(!stream->remaining){
      struct mesh_send *cell=stream->cell;
      if(!cell || atomic_load_explicit(&cell->ready,memory_order_acquire)<=stream->cycle)return 0;
      stream->next=&cell->request;stream->remaining=cell->chunks;
      if(ledger)ledger_seen(ledger,ring,cell,stream->cycle);
    }
    while(stream->remaining){
      struct ibv_send_wr *request=stream->next;
      const uint32_t frames=(request->sg_list->length+4095)/4096;
      if(((pair->posted-pair->retired)&WR_LOW)+frames>pair->gate)return 0;
      pair->posted+=frames;
      struct ibv_send_wr posting=*request,*bad;
      posting.next=NULL;posting.send_flags=IBV_SEND_SIGNALED;posting.wr_id=wr_id(pair->round,k,pair->posted);
      const int error=post(pair->qp,&posting,&bad);
      if(error){pair->posted-=frames;return error<0?-error:error;}
      stream->next=request->next;stream->remaining--;pair->requests++;
    }
    if(ledger)ledger_posted(ledger,ring);
    stream->cell=(struct mesh_send *)stream->cell->successor;stream->sent++;
    if(stream->cell==stream->first)stream->cycle++;
  }
}
/* design/bridge-and-ipc.md#loops: one pass of a port's send thread: the link's SEND completions taken, each retiring its
   pair's frames; a keepalive frame where the configuration paces one and a program holds the link (the Mini's
   Thunderbolt wakes slowly after quiet, mesh 12f484f8); each held lease's stream posted; and a ring of a closing path
   told drained once every cell its program released is posted and completed */
static void send_pass(struct port *port){
  struct ibv_wc done[32];
  const int count=port->sent->context->ops.poll_cq(port->sent,32,done);
  if(count<0)pair_failed(port->pair,EIO);
  for(int i=0;i<count;i++){
    const uint64_t id=done[i].wr_id;
    const uint32_t k=wr_index(id);
    if(k>=port->made)continue;
    struct pair *pair=port->pair+k;
    if(wr_round(id)!=(pair->round&0xffff))continue;
    if(done[i].status){pair_failed(pair,(int64_t)done[i].status);continue;}
    pair->retired=id&WR_LOW;pair->completions++;
  }
  struct pair *alive=port->pair;
  if(keepalive_ns && port->holding && alive->connected && ((alive->posted-alive->retired)&WR_LOW)<KEEPALIVE_SLOTS){
    const uint64_t at=now_ns();
    if(at-port->keepalive_at>=keepalive_ns){
      const uint64_t offset=(uint64_t)port->page*M->pgsz+(alive->posted%KEEPALIVE_SLOTS)*KEEPALIVE_FRAME;
      const struct message frame={.magic=XMAGIC,.kind=KEEPALIVE,.node=M->node};
      memcpy(wire.data+offset,&frame,sizeof frame);
      struct ibv_sge span=wire_span(&port->device,(uint64_t)M->block*M->pgsz,offset,KEEPALIVE_FRAME);
      struct ibv_send_wr posting={.wr_id=wr_id(alive->round,0,alive->posted+1),.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,
        .send_flags=IBV_SEND_SIGNALED},*bad;
      if(alive->qp->context->ops.post_send(alive->qp,&posting,&bad))pair_failed(alive,EIO);
      else{alive->posted++;port->keepalives++;}
      port->keepalive_at=at;
    }
  }
  for(uint32_t k=1;k<port->count;k++){
    struct pair *pair=port->pair+k;
    struct path *path=pair->path;
    if(!path || !pair->connected || atomic_load_explicit(&pair->failed,memory_order_relaxed))continue;
    struct stream *stream=&path->stream[pair->ring];
    const int error=stream_post(pair,k,stream,path->ledger,pair->ring);
    if(error){pair_failed(pair,error);continue;}
    if(!atomic_load_explicit(&path->closing,memory_order_acquire) || (pair->posted!=pair->retired && !pair->peer_gone) ||
       (atomic_load_explicit(&path->drained,memory_order_relaxed)>>pair->ring&1))continue;
    if(stream->cell && (stream->remaining || atomic_load_explicit(&stream->cell->ready,memory_order_acquire)>stream->cycle))continue;
    atomic_fetch_or_explicit(&path->drained,1u<<pair->ring,memory_order_release);
    wake();
  }
}

/* design/bridge-and-ipc.md#loops: a port's progress thread, `side` 0 its receives' and 1 its SENDs': passes without
   pause while the link is up and no guard stands, reading `stop` every pass; while the link is down or the guard
   stands it tells the guard value it holds at and waits on the port's bell, which every change of either rings; at
   `stop` it tells it holds no more and returns */
struct progress { struct port *port; int side; };
static void *port_progress(void *argument){
  struct port *port=((struct progress *)argument)->port;
  const int side=((struct progress *)argument)->side;
  free(argument);
  pthread_setname_np(side?"mesh.rdma.send":"mesh.rdma.receive");
  while(!stop){
    const uint64_t guard=atomic_load_explicit(&port->guard,memory_order_acquire);
    if((guard&1) || !atomic_load_explicit(&port->up,memory_order_acquire)){
      const uint64_t bell=atomic_load_explicit(&port->bell,memory_order_acquire);
      atomic_store_explicit(&port->quiet[side],guard,memory_order_release);
      os_sync_wake_by_address_all(&port->quiet[side],sizeof port->quiet[side],OS_SYNC_WAKE_BY_ADDRESS_NONE);
      if(!stop && atomic_load_explicit(&port->guard,memory_order_acquire)==guard && ((guard&1) || !atomic_load_explicit(&port->up,memory_order_acquire)))
        os_sync_wait_on_address(&port->bell,bell,sizeof bell,OS_SYNC_WAIT_ON_ADDRESS_NONE);
      continue;
    }
    if(side)send_pass(port);
    else receive_pass(port);
  }
  atomic_store_explicit(&port->quiet[side],UINT64_MAX,memory_order_release);
  os_sync_wake_by_address_all(&port->quiet[side],sizeof port->quiet[side],OS_SYNC_WAKE_BY_ADDRESS_NONE);
  return NULL;
}
/* design/bridge-and-ipc.md#loops: the port's two progress threads, started once; 0, or pthread_create's error */
static int port_threads(struct port *port){
  int error=0;
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_set_qos_class_np(&attributes,QOS_CLASS_USER_INTERACTIVE,0);
  while(port->threads<2 && !error){
    struct progress *progress=malloc(sizeof *progress);
    if(!progress){error=ENOMEM;break;}
    *progress=(struct progress){port,port->threads};
    atomic_store(&port->quiet[port->threads],atomic_load(&port->guard));
    if((error=pthread_create(&port->thread[port->threads],&attributes,port_progress,progress)))free(progress);
    else port->threads++;
  }
  pthread_attr_destroy(&attributes);
  return error;
}
/* design/RDMA-KERNEL-RECOVERY.md "The stripe writes of October 1": the port's guard, raised around every change of its
   device's verbs objects (a queue pair made, connected or destroyed) and of what holds its pairs: it returns once each
   progress thread holds (it told the guard's value, or that it returned at `stop`), so no queue pair of the device is
   posted on while another changes; lowered, the threads go on */
static void guard_raise(struct port *port){
  const uint64_t guard=atomic_fetch_add_explicit(&port->guard,1,memory_order_acq_rel)+1;
  bell_ring(&port->bell);
  for(int side=0;side<port->threads;side++)
    for(uint64_t held;(held=atomic_load_explicit(&port->quiet[side],memory_order_acquire))!=guard && held!=UINT64_MAX;)
      os_sync_wait_on_address(&port->quiet[side],held,sizeof held,OS_SYNC_WAIT_ON_ADDRESS_NONE);
}
static void guard_lower(struct port *port){
  atomic_fetch_add_explicit(&port->guard,1,memory_order_release);
  bell_ring(&port->bell);
}

/* design/bridge-and-ipc.md#links: a PAIRS message to the port's peer: pair `index` (all of the link's pairs where
   `index` is LINK_PAIRS) with its number, first sequence number and round, `code`, and the SENDs completed on its
   making before */
static void port_say(struct port *port,struct message said,const void *payload,uint32_t bytes);
static void pairs_tell(struct port *port,uint32_t index,int64_t code,uint64_t sent){
  struct pairs_entry entry[LINK_PAIRS];
  uint32_t n=0;
  for(uint32_t k=0;k<port->made;k++)if((index==LINK_PAIRS || k==index) && port->pair[k].qp)
    entry[n++]=(struct pairs_entry){k,port->pair[k].qp->qp_num,port->pair[k].psn,port->pair[k].round,code,sent};
  port_say(port,(struct message){.kind=PAIRS,.count=port->made},entry,n*(uint32_t)sizeof *entry);
}
/* pair k made again (its last making destroyed: the error state, then destroy), its round one more; under the guard;
   0, or the errno */
static int pair_new(struct port *port,uint32_t k){
  struct pair *pair=port->pair+k;
  if(pair->qp){
    const int error=pair_destroy(&port->device,pair->qp);
    if(error)say("link %u (%s): pair %u's destruction refused: %s; its handle let go\n",port->index,port->name,k,strerror(error));
    pair->qp=NULL;
  }
  uint32_t sends=0,receives=0;
  const int error=pair_make(&port->device,port->sent,port->received,&pair->qp,&sends,&receives);
  if(error)return error;
  pair->round++;pair->psn=arc4random()&0xffffff;pair->sends=sends;pair->receives=receives;pair->connected=0;
  pair->posted=pair->retired=pair->requests=pair->completions=0;pair->gate=MIN(sends,port->share);
  pair->peer_gone=0;atomic_store(&pair->peer_sent,UINT64_MAX);
  atomic_store(&pair->failed,0);
  return 0;
}
/* pair k connected to the peer's pair of its index where the peer's last told making is this one's round; the
   keepalive pair's receives posted; under the guard; 0, or the errno */
static int pair_join(struct port *port,uint32_t k){
  struct pair *pair=port->pair+k;
  if(pair->connected || !pair->heard || pair->peer_round!=pair->round || !pair->qp)return 0;
  int error=pair_connect(&port->device,pair->qp,pair->peer_qpn,pair->peer_psn,pair->psn,port->heard.lid,port->heard.gid);
  for(uint32_t slot=0;!error && !k && slot<KEEPALIVE_SLOTS;slot++){
    struct ibv_sge span=wire_span(&port->device,(uint64_t)M->block*M->pgsz,(uint64_t)port->page*M->pgsz+(KEEPALIVE_SLOTS+slot)*KEEPALIVE_FRAME,
                                  KEEPALIVE_FRAME);
    struct ibv_recv_wr request={.wr_id=wr_id(pair->round,0,slot),.sg_list=&span,.num_sge=1},*bad;
    error=pair->qp->context->ops.post_recv(pair->qp,&request,&bad);
  }
  if(!error)pair->connected=1;
  return error;
}
static void port_lost(struct port *port,int64_t code,int retire);
/* design/bridge-and-ipc.md#leases: lease pair k made again, joined where the peer's newer making was told, and told the
   peer with `code`; a failure of the verbs is the link's loss */
static void lease_remake(struct port *port,uint32_t k,int64_t code){
  guard_raise(port);
  const uint64_t sent=port->pair[k].completions;
  int error=pair_new(port,k);
  if(!error)error=pair_join(port,k);
  guard_lower(port);
  if(error){port_lost(port,error,0);return;}
  pairs_tell(port,k,code,sent);
}

/* design/prepared-machine.md#M12: ring r of a path's words cancelled past its landed count (~0): each record from its
   first not landed through the program's invocations less the ring's depth (a cyclic program's readers keep within the
   depth of the landings), every one not landed in a one-pass program; landed words stay */
static void ring_cancel(struct path *path,uint32_t r){
  const struct ring *ring=&path->ring[r];
  if(!ring->count)return;
  const int cyclic=(int)path->program->cyclic;
  const uint64_t n=path->program->invocations?path->program->invocations:1,first=ring->landed,end=cyclic?first+ring->count:ring->count;
  const struct prepared_receive *head=ring->first+first%ring->count;
  const uint64_t bound=head->invocation+first/ring->count*n+(ring->depth<n?n-ring->depth:1);
  for(uint64_t k=first;k<end;k++){
    const struct prepared_receive *record=ring->first+k%ring->count;
    if(cyclic && record->invocation+k/ring->count*n>=bound)break;
    if(record->input!=&record->sink && !record->release)atomic_store_explicit(record->input,UINT64_MAX,memory_order_release);
  }
}
/* design/prepared-machine.md#M12: every word of a program toward a destination whose rings were never attached */
static void words_cancel(uint64_t owner,uint32_t d){
  const struct mesh_tx *tx=&mesh_client_at(M,owner)->tx[d];
  const uint64_t range=sizeof(struct mesh_cancellation)+sizeof(struct mesh_cancel_range)*((uint64_t)d+1);
  if(!tx->cancel || tx->cancel<M->data_off || tx->cancel>M->length-range)return;
  const struct mesh_cancellation *c=(const void *)((char *)M+tx->cancel);
  const uint64_t offset=c->ranges[d].offset,count=c->ranges[d].count;
  if(offset<M->data_off || offset>M->length || count>(M->length-offset)/sizeof(uint64_t))return;
  _Atomic uint64_t *words=(void *)((char *)M+offset);
  for(uint64_t j=0;j<count;j++){
    const uint64_t word=atomic_load_explicit(words+j,memory_order_relaxed);
    if(!word || (c->cyclic && word!=UINT64_MAX))atomic_store_explicit(words+j,UINT64_MAX,memory_order_release);
  }
}

/* design/bridge-and-ipc.md#programs: a path's leases let go (all, or pair `only` alone), under its port's guard: each
   ring's words past its landed count cancelled where `cancel`, each pair made again (while the link is up) and its new
   number told the peer with `code`, each lease word free again */
static void path_release(struct path *path,int64_t code,int cancel,uint32_t only){
  struct port *port=ports+path->port;
  uint32_t told[MESH_QUEUES],n=0;
  uint64_t sent[MESH_QUEUES];
  int error=0;
  guard_raise(port);
  for(uint32_t r=0;r<MESH_QUEUES;r++){
    const uint32_t k=path->pair[r];
    if(!k || (only && k!=only))continue;
    if(cancel)ring_cancel(path,r);
    port->pair[k].path=NULL;path->pair[r]=0;port->holding--;
    atomic_fetch_or_explicit(&path->drained,1u<<r,memory_order_release);
    uint64_t held=path->program->owner;
    atomic_compare_exchange_strong(mesh_leases(M,port->index)+k-1,&held,0);
    if(!atomic_load(&port->up) || error)continue;
    sent[n]=port->pair[k].completions;
    if(!(error=pair_new(port,k)))error=pair_join(port,k);
    if(!error)told[n++]=k;
  }
  guard_lower(port);
  if(error){port_lost(port,error,0);return;}
  for(uint32_t i=0;i<n;i++)pairs_tell(port,told[i],code,sent[i]);
}
/* design/bridge-and-ipc.md#programs: a path's rings attached to their leases on its port, under the guard: each record
   named by its pair's making and index, each ring's capacity its pair's share of the link's completion queues, its
   first records posted; 0, or the post's error */
static int path_attach(struct path *path,const struct mesh_tx *tx){
  struct port *port=ports+path->port;
  int error=0;
  guard_raise(port);
  for(uint32_t r=0;r<MESH_QUEUES && !error;r++){
    if(!(path->rings>>r&1))continue;
    const uint32_t k=tx->lease[r];
    struct pair *pair=port->pair+k;
    struct ring *ring=&path->ring[r];
    for(size_t i=0;i<ring->count;i++)ring->first[i].request.wr_id=wr_id(pair->round,k,i);
    ring->capacity=MIN(pair->receives,port->share);
    pair->path=path;pair->ring=r;path->pair[r]=k;port->holding++;
    error=ring_advance(pair,ring,(int)path->program->cyclic);
  }
  guard_lower(port);
  return error;
}

static void ledger_free(struct ledger *l){ if(l){free(l->cell);free(l->send);free(l->landing);free(l);} }
static void program_free(struct program *prog){
  for(struct program **at=&programs;*at;at=&(*at)->next)if(*at==prog){*at=prog->next;break;}
  for(uint32_t p=0;p<prog->path_count;p++){free(prog->paths[p].receive);free(prog->paths[p].requests);ledger_free(prog->paths[p].ledger);}
  free(prog->paths);free(prog);
  unreleased=1;
}
/* design/algorithm-sources.md#meshresult: a program failed as a value naming the node `where`: every path's leases let
   go, their words past their landed counts cancelled and their peers told `code` (their programs fail alike), the words
   of destinations never attached cancelled, and each destination's status the failure */
static void program_fail(struct program *prog,int64_t code,uint32_t domain,uint32_t where){
  if(atomic_load(&prog->state)==FAILED)return;
  atomic_store(&prog->state,FAILED);
  const int own=owned(prog->owner);
  say("program %016llx (slot %u%s) failed at node %u: %s (domain %u)\n",(unsigned long long)mesh_client_at(M,prog->owner)->key,
      (uint32_t)(prog->owner>>32)&(MESH_CLIENTS-1),prog->relay?", a relay":"",where,strerror((int)code),domain);
  for(uint32_t p=0;p<prog->path_count;p++){
    struct path *path=prog->paths+p;
    int attached=0;
    for(uint32_t r=0;r<MESH_QUEUES;r++)attached|=path->pair[r]!=0;
    if(attached)path_release(path,code?code:ECANCELED,own,0);
    else if(own)words_cancel(prog->owner,path->destination);
    if(own)status_failed(&mesh_client_at(M,prog->owner)->status[path->destination],prog->owner,code,domain,where);
  }
}
/* design/bridge-and-ipc.md#leases, #relays: the peer of lease pair k let go of it in order while a program here holds
   it, and every SEND it completed on it has landed here: an end's status toward that destination stopped (ECONNRESET
   naming the far node: its program there ended, which the client reads before its next group), its words kept; a
   relay's other side closes, so what landed from this side is sent on before that side lets go in its turn; and the
   pair made again and joined to the peer's */
static void peer_released(struct port *port,uint32_t k){
  struct path *path=port->pair[k].path;
  if(!path){lease_remake(port,k,0);return;}
  struct program *prog=path->program;
  if(prog->relay)for(uint32_t p=0;p<prog->path_count;p++)if(prog->paths+p!=path)atomic_store(&prog->paths[p].closing,1);
  if(!prog->relay && owned(prog->owner))
    status_failed(&mesh_client_at(M,prog->owner)->status[path->destination],prog->owner,ECONNRESET,4,path->node);
  path_release(path,0,0,k);
}

/* design/bridge-and-ipc.md#programs: a started program taken: each destination with transfers a path on the port and
   leases its slot names (mesh-call.h mesh_transfers_agree), its streams and rings compiled from its lists and attached,
   each destination's status started; else each destination's status the failure (EINVAL a list or lease the bridge
   cannot carry, ENETDOWN its port not paired, ENETRESET a lease no longer its own: the link paired again since) */
static struct program *program_take(uint64_t owner,int relay){
  struct mesh_client *client=mesh_client_at(M,owner);
  const uint32_t destinations=mesh_destination(client,0);
  struct program *prog=calloc(1,sizeof *prog);
  struct path *paths=calloc(destinations+1,sizeof *paths);
  if(!prog || !paths){
    free(prog);free(paths);
    for(uint32_t d=0;d<destinations;d++)status_failed(&client->status[d],owner,ENOMEM,1,M->node);
    return NULL;
  }
  *prog=(struct program){.next=programs,.owner=owner,.relay=relay,.paths=paths,.invocations=client->tx[0].invocations?client->tx[0].invocations:1,
    .cyclic=atomic_load_explicit(&client->cyclic,memory_order_acquire)!=0};
  atomic_store(&prog->state,RUNNING);
  programs=prog;
  int64_t failed=0;
  for(uint32_t d=0;d<destinations && !failed;d++){
    if(!mesh_destination_transfers(M,owner,d))continue;
    const struct mesh_tx *tx=&client->tx[d];
    struct path *path=paths+prog->path_count++;
    *path=(struct path){.program=prog,.destination=d,.node=tx->node,.port=tx->port?tx->port-1u:MESH_PORTS};
    if(ledger_kept && (path->ledger=calloc(1,sizeof *path->ledger))){
      path->ledger->id=atomic_fetch_add(&ledger_ids,1)+1;
      path->ledger->send=calloc(MESH_LEDGER_EVENTS,sizeof *path->ledger->send);
      path->ledger->landing=calloc(MESH_LEDGER_EVENTS,sizeof *path->ledger->landing);
      if(!path->ledger->send || !path->ledger->landing){ledger_free(path->ledger);path->ledger=NULL;}
    }
    if(!tx->node || tx->node==M->node)failed=EINVAL;
    else if(path->port>=port_count || !atomic_load(&ports[path->port].up))failed=ENETDOWN;
    if(!failed)failed=sends_configure(prog,path,client,ports+path->port);
    if(!failed)failed=receives_configure(prog,path,client,ports+path->port);
    for(uint32_t r=0;r<MESH_QUEUES && !failed;r++){
      if(!(path->rings>>r&1))continue;
      if(!tx->lease[r] || tx->lease[r]>=ports[path->port].count)failed=EINVAL;
      else if(atomic_load(mesh_leases(M,path->port)+tx->lease[r]-1)!=owner)failed=ENETRESET;
    }
  }
  for(uint32_t p=0;p<prog->path_count && !failed;p++){
    const int error=path_attach(prog->paths+p,&client->tx[prog->paths[p].destination]);
    if(error)failed=error;
  }
  if(failed){
    say("program %016llx (slot %u) not started: %s\n",(unsigned long long)client->key,(uint32_t)(owner>>32)&(MESH_CLIENTS-1),strerror((int)failed));
    program_fail(prog,failed,1,M->node);
    for(uint32_t d=0;d<destinations;d++)if(!mesh_destination_transfers(M,owner,d) || atomic_load(&client->status[d].phase)!=MESH_STOPPED)
      status_failed(&client->status[d],owner,failed,1,M->node);
    return prog;
  }
  for(uint32_t p=0;p<prog->path_count;p++){
    struct mesh_port_info *status=&client->status[prog->paths[p].destination];
    atomic_store_explicit(&status->code,0,memory_order_relaxed);atomic_store_explicit(&status->failed,0,memory_order_relaxed);
    atomic_store_explicit(&status->phase,MESH_PAIRED,memory_order_relaxed);
    status_prepared(status,owner);
  }
  return prog;
}

/* design/bridge-and-ipc.md#loops: the threads' word to the event loop taken: a lease the peer let go of in order taken
   up once every SEND its holder completed has landed here (peer_released); a pair's error is its holder's failure (a
   lease's) or the link's loss (the keepalive's, or a completion queue's); a closing path whose rings drained lets go
   of its leases, and a program ending in order, or a relay, with none left has ended */
static void pairs_turn(void){
  for(uint32_t p=0;p<port_count;p++){
    struct port *port=ports+p;
    for(uint32_t k=0;k<port->count && atomic_load(&port->up);k++){
      struct pair *pair=port->pair+k;
      if(pair->peer_gone && pair->path && pair->path->ring[pair->ring].landed>=atomic_load(&pair->peer_sent))peer_released(port,k);
      const int64_t failed=atomic_exchange(&pair->failed,0);
      if(!failed)continue;
      say("link %u (%s): pair %u failed: code %lld\n",port->index,port->name,k,(long long)failed);
      if(!k)port_lost(port,failed,0);
      else if(port->pair[k].path)program_fail(port->pair[k].path->program,failed,2,M->node);
      else lease_remake(port,k,failed);
    }
  }
  for(struct program *prog=programs,*next;prog;prog=next){
    next=prog->next;
    if(atomic_load(&prog->state)==FAILED)continue;
    int attached=0;
    for(uint32_t q=0;q<prog->path_count;q++){
      struct path *path=prog->paths+q;
      int held=0;
      for(uint32_t r=0;r<MESH_QUEUES;r++)held|=path->pair[r]!=0;
      if(held && atomic_load(&path->closing) && (atomic_load(&path->drained)&path->rings)==path->rings){path_release(path,0,0,0);held=0;}
      attached|=held;
    }
    if(attached || (!prog->relay && atomic_load(&prog->state)!=ENDING))continue;
    say("program %016llx (slot %u%s) ended in order\n",(unsigned long long)mesh_client_at(M,prog->owner)->key,
        (uint32_t)(prog->owner>>32)&(MESH_CLIENTS-1),prog->relay?", a relay":"");
    uint64_t was=prog->owner|MESH_RETIRING|MESH_ORDERLY;
    struct mesh_client *client=mesh_client_at(M,prog->owner);
    const int relay=prog->relay;
    const uint64_t owner=prog->owner;
    program_free(prog);
    if(relay)mesh_retire(M,owner);
    else atomic_compare_exchange_strong(&client->owner,&was,0);
  }
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
  uint32_t *lost=malloc(count*sizeof *lost),pruned=0;
  for(uint64_t k=0;k<count;k++)if(!reached[k] && !(at==n && k==at)){
    struct known *gone=known_at(next[k].node);char text[768];health_text(&next[k].health,text,sizeof text);
    say("node %u (%s) unreachable; its last health: %s\n",next[k].node,next[k].name,text);
    if(gone)gone->gone=1;
    if(lost)lost[pruned++]=next[k].node;
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
  free(next);free(reached);free(order);free(lost);
}
/* design/recovery.md#diagnosis: a record whose links (each port's peer, phase and bandwidth), name and facts are
   the held one's changes only its health, sequence and ports' counts (their codes, pairings and queue pairs), written
   under the region's health serial and never the map's: the map serial moves only where the mesh changes (a
   communicator's session reads the map again where it moved, and reopens only where its own routes or their links'
   costs changed: nccl-mesh.c map_digest, session_ensure), and neither a health sample nor a queue pair is a change of
   the mesh */
static int record_links_same(const struct mesh_record *a,const struct mesh_record *b){
  int same=a->node==b->node && a->ports==b->ports && !memcmp(a->name,b->name,sizeof a->name) &&
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
static char model[MESH_MODEL_BYTES];
static void record_changed(void){
  struct mesh_record own={.node=M->node,.ports=port_count,.sequence=++sequence,.window=wire.length,.memory=memory};
  memcpy(own.name,host,sizeof own.name);
  memcpy(own.model,model,sizeof own.model);
  memcpy(own.port,mesh_links(M),sizeof own.port);
  own.health=health;
  struct mesh_record *held=record_held(M->node);
  if(held && record_links_same(held,&own))health_put(held,&own);
  else map_put(&own);
  for(uint32_t p=0;p<port_count;p++)if(ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&own,sizeof own);
}
/* design/recovery.md#diagnosis: this node's health sampled at each turn of the event loop; a change of its power source,
   of its charge off AC, or of its memory or thermal pressure (each a notification the loop takes) is logged and sent
   at once, anything else with its record, and the record flooded again at the configuration's `health` interval
   (health_ns, a paced emission; none where it states none) */
/* #admission: the node's memory low (the kernel's pressure at warning or worse, or its available memory under a
   MESH_LOW_SHARE-th of the node's) logged with the largest processes this bridge's user runs, by physical footprint,
   as it begins and as its pressure changes while it lasts: what any client, a bridge client or not, holds outside the
   window */
#define MESH_LOW_SHARE 16u
#define MESH_LOW_LISTED 5
static uint32_t low_told;
static void memory_watch(void){
  if(health.pressure<2 && health.available>=memory/MESH_LOW_SHARE){low_told=0;return;}
  if(low_told==health.pressure+1)return;
  low_told=health.pressure+1;
  int count=proc_listallpids(NULL,0);
  pid_t *pids=count>0?calloc((size_t)count+16,sizeof *pids):NULL;
  struct { pid_t pid; uint64_t bytes; } top[MESH_LOW_LISTED]={{0,0}};
  count=pids?proc_listallpids(pids,(int)(((size_t)count+16)*sizeof *pids)):0;
  for(int i=0;i<count;i++){
    struct rusage_info_v0 use;
    if(proc_pid_rusage(pids[i],RUSAGE_INFO_V0,(rusage_info_t *)&use))continue;
    for(int k=0;k<MESH_LOW_LISTED;k++)if(use.ri_phys_footprint>top[k].bytes){
      memmove(top+k+1,top+k,(MESH_LOW_LISTED-1-(size_t)k)*sizeof *top);
      top[k].pid=pids[i];top[k].bytes=use.ri_phys_footprint;break;
    }
  }
  free(pids);
  char listed[512]="";size_t used=0;
  for(int k=0;k<MESH_LOW_LISTED && top[k].bytes;k++){
    char name[64]="?";proc_name(top[k].pid,name,sizeof name);
    used+=(size_t)snprintf(listed+used,used<sizeof listed?sizeof listed-used:0,"%s%s (%d) %.2f GiB",k?", ":"",name,top[k].pid,
                           (double)top[k].bytes/1073741824.0);
    if(used>=sizeof listed)break;
  }
  say("memory low: %.2f GiB available of %.2f GiB, pressure %s, the window %.2f GiB; largest: %s\n",
      (double)health.available/1073741824.0,(double)memory/1073741824.0,health.pressure==4?"critical":health.pressure==2?"warning":"normal",
      (double)((uint64_t)mesh_arena_pages(M)*M->pgsz)/1073741824.0,listed);
}
static void health_turn(int flood){
  struct mesh_health sampled=health;
  health_sample(&sampled);
  const int urgent=sampled.power!=health.power || (sampled.power!=MESH_POWER_AC && sampled.charge!=health.charge) ||
    sampled.pressure!=health.pressure || sampled.thermal!=health.thermal;
  health=sampled;
  if(urgent){char text[768];health_text(&health,text,sizeof text);say("health: %s\n",text);}
  if(urgent || flood)record_changed();
  memory_watch();
}
/* design/bridge-and-ipc.md#records: a record heard on a port (RFC 2328 s13): this node's own, from an earlier start of
   its bridge, newer than the one it holds, moves its sequence past it and sends its record again (s13.4); another node's
   no newer than the newest heard of it, the map holding it or not, is dropped and not sent on, so a record the map
   pruned never circulates, and where the one held is newer, that one is sent back on the port it came from (s13 step
   8: a node that started its bridge again hears its own records' newest and moves past it) */
static void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes){
  struct mesh_record heard;
  if(bytes!=sizeof heard)return;
  memcpy(&heard,payload,sizeof heard);
  heard.name[sizeof heard.name-1]=0;
  heard.health.session[sizeof heard.health.session-1]=0;heard.health.shutdown[sizeof heard.health.shutdown-1]=0;
  if(!heard.node || heard.ports>MESH_PORTS)return;
  heard.model[sizeof heard.model-1]=0;
  if(heard.node==M->node){
    const struct mesh_record *mine=record_held(M->node);
    if(heard.sequence>sequence || (heard.sequence==sequence && mine && record_newer(&heard,mine))){sequence=heard.sequence;record_changed();}
    return;
  }
  struct mesh_record *held=record_held(heard.node);
  struct known *k=known_at(heard.node);
  if(k && k->heard && !record_newer(&heard,&k->record)){
    if(record_newer(&k->record,&heard))port_say(from,(struct message){.kind=RECORD},&k->record,sizeof k->record);
    /* the newest known of a node the map dropped while no paired path reached it: put again where a path now does,
       and sent on only then (a pruned record that no path reaches stops here) */
    else if(!held){
      map_put(&heard);
      for(uint32_t p=0;record_held(heard.node) && p<port_count;p++)
        if(ports+p!=from && ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&heard,sizeof heard);
    }
    return;
  }
  /* design/recovery.md#diagnosis: a node that joins or comes back, and a peer's urgent change, are logged here */
  char text[768];
  if(!held && k && (k->gone || !k->heard)){
    health_text(&heard.health,text,sizeof text);
    if(k->gone)say("node %u (%s) back, %s: %s\n",heard.node,heard.name,
                   strncmp(k->record.health.session,heard.health.session,sizeof heard.health.session)?"a new boot":"the same boot",text);
    else say("node %u (%s) joined: %s\n",heard.node,heard.name,text);
  }
  if(held && (held->health.power!=heard.health.power || held->health.pressure!=heard.health.pressure ||
              held->health.thermal!=heard.health.thermal ||
              (heard.health.power!=MESH_POWER_AC && held->health.charge!=heard.health.charge))){
    health_text(&heard.health,text,sizeof text);
    say("node %u (%s) health: %s\n",heard.node,heard.name,text);
  }
  if(k){k->heard=1;k->gone=0;k->record=heard;}
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
/* design/bridge-and-ipc.md#requests: a channel (RFC 4254 s5) from a client of one node to a service of another, the
   same key (its origin's node and sequence) on every hop. Its two ends, each a socket of this node's (the client's, the
   service's or an `exec` child's), a port toward the next bridge, or the bridge's `exec` taking its command line; the
   bytes one end gives go to the other. A port end takes no more than its credit (the window its peer granted), and the bytes a port end gave
   are granted back as they leave for the other end, so a hop holds at most a window each way (RFC 4254 s5.2). A
   socket end is read only while the other end can take what is read */
#define MESH_CHANNELS 256
#define MESH_SERVICES 32
#define MESH_WINDOW (UINT32_C(128)<<10)
#define MESH_PACKET (UINT32_C(32)<<10)
#define MESH_EXEC_BYTES 4096u
#define TAG_CHANNEL (UINT64_C(1)<<62)
#define TAG_ASKING (UINT64_C(1)<<61)
#define TAG_SERVING (UINT64_C(1)<<60)
/* design/bridge-dataflow.md#the-loopback-fabric: a loop port's cable listener's event, its index */
#define TAG_LOOP (UINT64_C(1)<<59)
enum { END_SOCKET=1, END_PORT, END_EXEC, END_MEET, END_RELAY };
struct end { int kind,fd,reading,writing,ended,told; struct port *port; uint64_t credit; unsigned char *queue; size_t queued,room; };
struct channel { int used,open,exited,status,drained,reported,child; pid_t pid; uint64_t key; uint32_t origin,destination;
  char service[MESH_SERVICE_BYTES]; struct end end[2]; unsigned char *held; size_t held_bytes; };
struct asking { int fd; size_t got; struct mesh_request request; };
struct serving { int fd; char service[MESH_SERVICE_BYTES],path[MESH_SERVICE_PATH]; };
static struct channel channels[MESH_CHANNELS];
static struct asking askings[MESH_CHANNELS];
static struct serving servings[MESH_SERVICES];
static uint32_t channel_sequence;
static int exec_served;
static pid_t orphans[MESH_CHANNELS];
static void fd_watch(int fd,int filter,int on,uint64_t udata){
  struct kevent64_s event;EV_SET64(&event,(uint64_t)fd,filter,EV_ADD|(on?EV_ENABLE:EV_DISABLE),0,0,udata,0,0);
  kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
}
static void fd_reset(int fd){
  struct linger now_={1,0};
  setsockopt(fd,SOL_SOCKET,SO_LINGER,&now_,sizeof now_);
  close(fd);
}
static void fd_answer(int fd,int32_t code){
  if(send(fd,&code,sizeof code,0)!=(ssize_t)sizeof code)return;
}
static uint64_t channel_udata(const struct channel *c,int s){ return TAG_CHANNEL|(uint64_t)(c-channels)<<1|(uint64_t)s; }
static void channel_say(const struct channel *c,int s,uint32_t kind,uint64_t count,int64_t code,const void *payload,uint32_t bytes){
  if(c->end[s].kind==END_PORT)port_say(c->end[s].port,(struct message){.kind=kind,.key=c->key,.from=c->origin,.to=c->destination,
                                        .count=(uint32_t)count,.code=code},payload,bytes);
}
/* a channel ended: where `code`, each of its ends but `from` told so (a port end CLOSE, a socket end reset; a client
   still awaiting its answer answered with the code); the slot freed first, so a port lost meanwhile finds it free */
static void service_gone(struct channel *c);
static void channel_end(struct channel *c,int64_t code,int from){
  if(c->end[1].kind==END_MEET || c->end[1].kind==END_RELAY)service_gone(c);
  struct channel was=*c;
  memset(c,0,sizeof *c);
  if(code)say("channel %u.%u to node %u's %s: %s\n",was.origin,(uint32_t)was.key,was.destination,was.service,strerror((int)code));
  if(was.pid && !was.exited){
    kill(-was.pid,SIGTERM);
    for(uint32_t i=0;i<MESH_CHANNELS;i++)if(!orphans[i]){orphans[i]=was.pid;break;}
  }
  for(int s=0;s<2;s++){
    struct end *e=was.end+s;
    if(e->kind==END_SOCKET && e->fd>=0){
      if(code && !was.open && s==0 && was.origin==M->node){fd_answer(e->fd,(int32_t)code);close(e->fd);}
      else if(code)fd_reset(e->fd);
      else close(e->fd);
    }
    if(e->kind==END_PORT && code && s!=from)channel_say(&was,s,CHANNEL_CLOSE,0,code,NULL,0);
    free(e->queue);
  }
  free(was.held);
}
static struct channel *channel_new(void){
  for(uint32_t i=0;i<MESH_CHANNELS;i++)if(!channels[i].used){
    struct channel *c=channels+i;
    memset(c,0,sizeof *c);c->used=1;c->end[0].fd=c->end[1].fd=-1;
    return c;
  }
  return NULL;
}
static uint32_t cable_to(uint32_t node);
static struct port *channel_hop(uint32_t node){
  uint32_t *route=NULL;
  const uint32_t hops=mesh_route(M,M->node,node,&route);
  const uint32_t p=hops>=2?cable_to(route[1]):MESH_PORTS;
  free(route);
  return p<MESH_PORTS?ports+p:NULL;
}
static void channel_pump(struct channel *c);
static void service_heard(struct channel *c,int t);
static int ledger_cut(uint64_t key,const char *path);
static int64_t child_spawn(struct channel *c,int s);
static void child_ended(struct channel *c);
/* the bytes queued for end `t` given to it as far as it takes them, each granted back to the other end where it is a
   port; once none is left and the other end has ended, end `t` told so (a socket's write side shut, a port
   CHANNEL_SHUT) */
static void end_drain(struct channel *c,int t){
  struct end *e=c->end+t,*o=c->end+(1-t);
  size_t given=0;
  while(given<e->queued){
    size_t n=e->queued-given;
    if(e->kind==END_SOCKET){
      ssize_t sent=send(e->fd,e->queue+given,n,0);
      if(sent<0 && (errno==EAGAIN || errno==EINTR))break;
      if(sent<=0){channel_end(c,sent?errno:EPIPE,-1);return;}
      n=(size_t)sent;
    }else if(e->kind==END_EXEC){
      const unsigned char *line=memchr(e->queue+given,'\n',n);
      if(line)n=(size_t)(line-(e->queue+given))+1;
      if(c->held_bytes+n>MESH_EXEC_BYTES){channel_end(c,EMSGSIZE,-1);return;}
      unsigned char *grown=realloc(c->held,c->held_bytes+n+1);
      if(!grown){channel_end(c,ENOMEM,-1);return;}
      c->held=grown;memcpy(c->held+c->held_bytes,e->queue+given,n);c->held_bytes+=n;
      if(o->kind==END_PORT)channel_say(c,1-t,CHANNEL_WINDOW,n,0,NULL,0);
      given+=n;
      if(!line)continue;
      memmove(e->queue,e->queue+given,e->queued-given);e->queued-=given;given=0;
      const int64_t error=child_spawn(c,t);
      if(error){channel_end(c,error,-1);return;}
      continue;
    }else if(e->kind==END_MEET || e->kind==END_RELAY){
      if(c->held_bytes+n>sizeof(struct mesh_relay_request)+MESH_MEET_BYTES+64){channel_end(c,EMSGSIZE,-1);return;}
      unsigned char *grown=realloc(c->held,c->held_bytes+n);
      if(!grown){channel_end(c,ENOMEM,-1);return;}
      c->held=grown;memcpy(c->held+c->held_bytes,e->queue+given,n);c->held_bytes+=n;
      given+=n;
      if(o->kind==END_PORT)channel_say(c,1-t,CHANNEL_WINDOW,n,0,NULL,0);
      continue;
    }else{
      n=MIN(n,MIN(e->credit,MESH_PACKET));
      if(!n)break;
      channel_say(c,t,CHANNEL_DATA,0,0,e->queue+given,(uint32_t)n);
      e->credit-=n;
      if(!c->used)return;
    }
    given+=n;
    if(o->kind==END_PORT)channel_say(c,1-t,CHANNEL_WINDOW,n,0,NULL,0);
  }
  memmove(e->queue,e->queue+given,e->queued-given);e->queued-=given;
  if(e->kind==END_MEET || e->kind==END_RELAY){
    service_heard(c,t);
    if(!c->used)return;
  }
  if(e->kind==END_SOCKET){
    const int want=e->queued>0;
    if(want!=e->writing){e->writing=want;fd_watch(e->fd,EVFILT_WRITE,want,channel_udata(c,t));}
  }
  if(!e->queued && o->ended && !e->told && e->kind==END_EXEC){
    const int64_t error=child_spawn(c,t);
    if(error){channel_end(c,error,-1);return;}
  }
  if(!e->queued && o->ended && !e->told){
    e->told=1;
    if(e->kind==END_RELAY){service_gone(c);e->ended=1;end_drain(c,1-t);}
    else if(e->kind==END_SOCKET)shutdown(e->fd,SHUT_WR);
    else if(e->kind!=END_MEET)channel_say(c,t,CHANNEL_SHUT,0,0,NULL,0);
    if(!c->used)return;
  }
  if(c->end[0].told && c->end[1].told){channel_end(c,0,-1);return;}
  channel_pump(c);
}
static void end_put(struct channel *c,int t,const void *bytes,size_t n){
  struct end *e=c->end+t;
  if(e->queued+n>e->room){
    unsigned char *grown=realloc(e->queue,e->queued+n);
    if(!grown){channel_end(c,ENOMEM,-1);return;}
    e->queue=grown;e->room=e->queued+n;
  }
  memcpy(e->queue+e->queued,bytes,n);e->queued+=n;
  end_drain(c,t);
}
/* each socket end read while the channel is open, it has not ended, and the other end takes what it gives at once */
static void channel_pump(struct channel *c){
  for(int s=0;s<2;s++){
    struct end *e=c->end+s,*o=c->end+(1-s);
    if(e->kind!=END_SOCKET || e->ended || (c->pid && s==c->child && c->drained))continue;
    const int want=c->open && !o->queued && (o->kind!=END_PORT || o->credit>0);
    if(want!=e->reading){e->reading=want;fd_watch(e->fd,EVFILT_READ,want,channel_udata(c,s));}
  }
}
static void end_read(struct channel *c,int s){
  struct end *e=c->end+s,*o=c->end+(1-s);
  if(e->ended)return;
  size_t room=o->kind==END_PORT?MIN(o->credit,MESH_PACKET):MESH_PACKET;
  if(!room || o->queued){channel_pump(c);return;}
  unsigned char block[MESH_PACKET];
  ssize_t got=recv(e->fd,block,room,0);
  if(got<0 && (errno==EAGAIN || errno==EINTR))return;
  if(got<0){channel_end(c,errno,-1);return;}
  if(!got){
    e->reading=0;fd_watch(e->fd,EVFILT_READ,0,channel_udata(c,s));
    if(c->pid && s==c->child){c->drained=1;child_ended(c);return;}
    e->ended=1;end_drain(c,1-s);return;
  }
  end_put(c,1-s,block,(size_t)got);
}
/* #admission: a request for `bytes` outside the window admitted while the node's available memory, read now, holds
   them; a refusal logged */
static int64_t admitted(const struct channel *c,uint64_t bytes){
  const uint64_t room=bytes?available_now():0;
  if(!bytes || room>=bytes)return 0;
  say("admission: node %u's request to %s for %.2f GiB refused: %.2f GiB available, pressure %u\n",c->origin,c->service,
      (double)bytes/1073741824.0,(double)room/1073741824.0,health.pressure);
  return ENOMEM;
}
/* a channel's far end opened here: the service's socket connected, or the bridge's own `exec` */
static int64_t service_open(struct channel *c,int s,uint64_t bytes){
  struct end *e=c->end+s;
  const int64_t refused=admitted(c,bytes);
  if(refused)return refused;
  if(!strcmp(c->service,"meet")){e->kind=END_MEET;return 0;}
  if(!strcmp(c->service,"relay")){e->kind=END_RELAY;return 0;}
  if(!strcmp(c->service,"exec")){
    if(!exec_served)return EACCES;
    e->kind=END_EXEC;return 0;
  }
  if(!strcmp(c->service,"nodeinfo")){
    static const uint16_t served[]={8100,8099};
    int64_t error=ECONNREFUSED;
    for(size_t i=0;i<sizeof served/sizeof *served;i++){
      struct sockaddr_in6 to={.sin6_len=sizeof to,.sin6_family=AF_INET6,.sin6_port=htons(served[i]),.sin6_addr=IN6ADDR_LOOPBACK_INIT};
      int fd=socket(AF_INET6,SOCK_STREAM,0),on=1;
      if(fd<0)return errno;
      if(setsockopt(fd,SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on) || connect(fd,(struct sockaddr *)&to,sizeof to) ||
         fcntl(fd,F_SETFL,fcntl(fd,F_GETFL)|O_NONBLOCK)){error=errno;close(fd);continue;}
      e->kind=END_SOCKET;e->fd=fd;
      return 0;
    }
    return error;
  }
  for(uint32_t i=0;i<MESH_SERVICES;i++)if(servings[i].fd>=0 && !strcmp(servings[i].service,c->service)){
    struct sockaddr_un to={.sun_family=AF_UNIX};
    snprintf(to.sun_path,sizeof to.sun_path,"%s",servings[i].path);
    int fd=socket(AF_UNIX,SOCK_STREAM,0),on=1;
    if(fd<0)return errno;
    if(setsockopt(fd,SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on) || connect(fd,(struct sockaddr *)&to,sizeof to) ||
       fcntl(fd,F_SETFL,fcntl(fd,F_GETFL)|O_NONBLOCK)){const int error=errno;close(fd);return error;}
    e->kind=END_SOCKET;e->fd=fd;
    return 0;
  }
  return ENOENT;
}
/* #exec: the bridge's own service `exec`, where its configuration serves it: the request's first line a command run by
   the user's shell (-c) in its home, its process group its own and every descriptor but its standard ones closed
   (POSIX_SPAWN_CLOEXEC_DEFAULT: no device or socket of the bridge's), the rest of the request its standard input, and
   the answer its standard output and error, then its wait status (int32), once both have ended; a channel that fails
   while it runs sends its group SIGTERM */
static int64_t child_spawn(struct channel *c,int s){
  struct end *e=c->end+s;
  int pair[2],on=1;
  const struct passwd *user=getpwuid(getuid());
  if(!user)return ENOENT;
  if(socketpair(AF_UNIX,SOCK_STREAM,0,pair))return errno;
  c->held=c->held?c->held:malloc(1);
  size_t n=c->held_bytes;
  while(n && (c->held[n-1]=='\n' || c->held[n-1]=='\r'))n--;
  c->held[n]=0;
  char *shell=user->pw_shell && *user->pw_shell?user->pw_shell:"/bin/zsh",temp[PATH_MAX]="/tmp";
  confstr(_CS_DARWIN_USER_TEMP_DIR,temp,sizeof temp);
  char home[PATH_MAX+8],name[256],login[256],shelled[PATH_MAX+8],tmpdir[PATH_MAX+8];
  snprintf(home,sizeof home,"HOME=%s",user->pw_dir);snprintf(name,sizeof name,"USER=%s",user->pw_name);
  snprintf(login,sizeof login,"LOGNAME=%s",user->pw_name);snprintf(shelled,sizeof shelled,"SHELL=%s",shell);
  snprintf(tmpdir,sizeof tmpdir,"TMPDIR=%s",temp);
  char *argv[]={shell,"-c",(char *)c->held,NULL};
  char *envp[]={home,name,login,shelled,tmpdir,"PATH=/usr/bin:/bin:/usr/sbin:/sbin","LANG=en_US.UTF-8",NULL};
  posix_spawn_file_actions_t actions;posix_spawnattr_t attributes;
  sigset_t every,none;sigfillset(&every);sigemptyset(&none);
  posix_spawn_file_actions_init(&actions);posix_spawnattr_init(&attributes);
  for(int k=0;k<3;k++)posix_spawn_file_actions_adddup2(&actions,pair[1],k);
  posix_spawn_file_actions_addchdir(&actions,user->pw_dir);
  posix_spawnattr_setflags(&attributes,POSIX_SPAWN_CLOEXEC_DEFAULT|POSIX_SPAWN_SETPGROUP|POSIX_SPAWN_SETSIGDEF|POSIX_SPAWN_SETSIGMASK);
  posix_spawnattr_setpgroup(&attributes,0);posix_spawnattr_setsigdefault(&attributes,&every);posix_spawnattr_setsigmask(&attributes,&none);
  pid_t pid=0;
  const int error=posix_spawn(&pid,shell,&actions,&attributes,argv,envp);
  posix_spawn_file_actions_destroy(&actions);posix_spawnattr_destroy(&attributes);
  close(pair[1]);
  if(error){close(pair[0]);return error;}
  if(fcntl(pair[0],F_SETFL,fcntl(pair[0],F_GETFL)|O_NONBLOCK) || setsockopt(pair[0],SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on)){
    kill(-pid,SIGTERM);close(pair[0]);c->pid=pid;return errno?errno:EIO;
  }
  e->kind=END_SOCKET;e->fd=pair[0];c->pid=pid;c->child=s;
  struct kevent64_s watch;EV_SET64(&watch,(uint64_t)pid,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,UINT64_MAX,0,0);
  kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
  say("exec from node %u, pid %d: %.200s\n",c->origin,pid,c->held);
  channel_pump(c);
  return 0;
}
/* a child's output ended and its process exited: its wait status sent after its output, and its end ended */
static void child_ended(struct channel *c){
  if(!c->exited){
    int status=0;
    const pid_t done=waitpid(c->pid,&status,WNOHANG);
    if(done==c->pid){c->exited=1;c->status=status;}
  }
  if(!c->exited || !c->drained || c->reported)return;
  int32_t status=(int32_t)c->status;
  c->reported=1;c->end[c->child].ended=1;
  end_put(c,1-c->child,&status,sizeof status);
}
/* a process exited: an exec's child whose channel awaits it, or one whose channel ended, reaped */
static void child_heard(pid_t pid){
  for(uint32_t i=0;i<MESH_CHANNELS;i++){
    if(orphans[i]==pid && waitpid(pid,NULL,WNOHANG)==pid)orphans[i]=0;
    if(channels[i].used && channels[i].pid==pid && !channels[i].exited)child_ended(channels+i);
  }
}
/* a client's request read whole: a channel opened toward its node's service (here, or its CHANNEL_OPEN sent to the next hop),
   a service registered, a ledger cut, or a raw stream (design/bridge-dataflow.md#streams) */
static void asked(struct asking *a){
  struct mesh_request *r=&a->request;
  const int fd=a->fd;a->fd=-1;
  fd_watch(fd,EVFILT_READ,0,0);
  r->service[sizeof r->service-1]=0;r->path[sizeof r->path-1]=0;
  if(r->magic==MESH_REQUEST_MAGIC && r->kind==MESH_REQUEST_LEDGER){
    fd_answer(fd,r->path[0]=='/'?ledger_cut(r->bytes,r->path):EINVAL);close(fd);
    return;
  }
  if(r->magic!=MESH_REQUEST_MAGIC || !r->service[0] || (r->kind!=MESH_REQUEST_OPEN && r->kind!=MESH_REQUEST_SERVE)){fd_answer(fd,EPROTO);close(fd);return;}
  if(r->kind==MESH_REQUEST_SERVE){
    struct serving *at=NULL;
    for(uint32_t i=0;i<MESH_SERVICES;i++)if(servings[i].fd>=0 && !strcmp(servings[i].service,r->service)){close(servings[i].fd);servings[i].fd=-1;}
    for(uint32_t i=0;!at && i<MESH_SERVICES;i++)if(servings[i].fd<0)at=servings+i;
    if(r->path[0]!='/' || !at){fd_answer(fd,at?EINVAL:ENOSPC);close(fd);return;}
    at->fd=fd;memcpy(at->service,r->service,sizeof at->service);memcpy(at->path,r->path,sizeof at->path);
    fd_answer(fd,0);
    fd_watch(fd,EVFILT_READ,1,TAG_SERVING|(uint64_t)(at-servings));
    say("service %s: served at %s\n",at->service,at->path);
    return;
  }
  struct channel *c=channel_new();
  if(!c){fd_answer(fd,ENOSPC);close(fd);return;}
  c->key=(uint64_t)M->node<<32|++channel_sequence;c->origin=M->node;c->destination=r->node;
  memcpy(c->service,r->service,sizeof c->service);
  c->end[0].kind=END_SOCKET;c->end[0].fd=fd;
  if(r->node==M->node){
    const int64_t error=service_open(c,1,r->bytes);
    if(error){channel_end(c,error,-1);return;}
    c->open=1;fd_answer(fd,0);channel_pump(c);
    return;
  }
  struct port *hop=channel_hop(r->node);
  if(!hop){channel_end(c,EHOSTUNREACH,-1);return;}
  c->end[1].kind=END_PORT;c->end[1].port=hop;
  channel_say(c,1,CHANNEL_OPEN,MESH_WINDOW,(int64_t)r->bytes,c->service,(uint32_t)sizeof c->service);
}
static void asking_heard(struct asking *a){
  ssize_t got=recv(a->fd,(char *)&a->request+a->got,sizeof a->request-a->got,0);
  if(got<0 && (errno==EAGAIN || errno==EINTR))return;
  if(got<=0){close(a->fd);a->fd=-1;return;}
  a->got+=(size_t)got;
  if(a->got==sizeof a->request)asked(a);
}
static void asking_accepted(int fd){
  int on=1;
  for(uint32_t i=0;i<MESH_CHANNELS;i++)if(askings[i].fd<0){
    if(fcntl(fd,F_SETFL,fcntl(fd,F_GETFL)|O_NONBLOCK) || setsockopt(fd,SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on) ||
       setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&on,sizeof on))break;
    askings[i]=(struct asking){.fd=fd};
    fd_watch(fd,EVFILT_READ,1,TAG_ASKING|i);
    return;
  }
  close(fd);
}
/* a channel's message from a port: an CHANNEL_OPEN toward this node's service or, through it, the next hop's; then its answer
   (CHANNEL_CONFIRM or CHANNEL_REFUSE), its CHANNEL_DATA, the credit its peer granted (CHANNEL_WINDOW), its end (CHANNEL_SHUT) or its failure (CHANNEL_CLOSE) */
static void channel_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  if(heard->kind==CHANNEL_OPEN){
    struct channel *c=channel_new();
    struct message refuse={.kind=CHANNEL_REFUSE,.key=heard->key,.from=heard->from,.to=heard->to};
    if(!c){refuse.code=ENOSPC;port_say(port,refuse,NULL,0);return;}
    c->key=heard->key;c->origin=heard->from;c->destination=heard->to;
    memcpy(c->service,payload,MIN(heard->bytes,sizeof c->service));c->service[sizeof c->service-1]=0;
    c->end[0].kind=END_PORT;c->end[0].port=port;c->end[0].credit=heard->count;
    if(heard->to==M->node){
      const int64_t error=service_open(c,1,(uint64_t)heard->code);
      if(error){refuse.code=error;port_say(port,refuse,NULL,0);channel_end(c,0,-1);return;}
      c->open=1;channel_say(c,0,CHANNEL_CONFIRM,MESH_WINDOW,0,NULL,0);channel_pump(c);
      return;
    }
    struct port *hop=channel_hop(heard->to);
    if(!hop || hop==port){refuse.code=EHOSTUNREACH;port_say(port,refuse,NULL,0);channel_end(c,0,-1);return;}
    c->end[1].kind=END_PORT;c->end[1].port=hop;
    channel_say(c,1,CHANNEL_OPEN,MESH_WINDOW,heard->code,c->service,(uint32_t)sizeof c->service);
    return;
  }
  struct channel *c=NULL;int s=0;
  for(uint32_t i=0;!c && i<MESH_CHANNELS;i++)if(channels[i].used && channels[i].key==heard->key)
    for(int k=0;k<2;k++)if(channels[i].end[k].kind==END_PORT && channels[i].end[k].port==port){c=channels+i;s=k;}
  if(!c){
    if(heard->kind!=CHANNEL_CLOSE && heard->kind!=CHANNEL_REFUSE)
      port_say(port,(struct message){.kind=CHANNEL_CLOSE,.key=heard->key,.from=heard->from,.to=heard->to,.code=ECONNRESET},NULL,0);
    return;
  }
  struct end *e=c->end+s;
  const int t=1-s;
  if(heard->kind==CHANNEL_CONFIRM && !c->open){
    e->credit=heard->count;c->open=1;
    if(c->end[t].kind==END_PORT)channel_say(c,t,CHANNEL_CONFIRM,MESH_WINDOW,0,NULL,0);
    else fd_answer(c->end[t].fd,0);
    channel_pump(c);
  }
  else if(heard->kind==CHANNEL_REFUSE && !c->open){
    if(c->end[t].kind==END_PORT)channel_say(c,t,CHANNEL_REFUSE,0,heard->code,NULL,0);
    c->end[t].kind==END_PORT?channel_end(c,0,-1):channel_end(c,heard->code?heard->code:ECONNREFUSED,s);
  }
  else if(heard->kind==CHANNEL_DATA && heard->bytes)end_put(c,t,payload,heard->bytes);
  else if(heard->kind==CHANNEL_WINDOW){e->credit+=heard->count;end_drain(c,s);}
  else if(heard->kind==CHANNEL_SHUT){e->ended=1;end_drain(c,t);}
  else if(heard->kind==CHANNEL_CLOSE)channel_end(c,heard->code?heard->code:ECONNRESET,s);
}
/* a request socket's event: a client's request read, a channel's socket end read or written, a registration's
   connection closed */
static void request_event(const struct kevent64_s *e){
  if(e->udata&TAG_ASKING){
    struct asking *a=askings+(e->udata&(MESH_CHANNELS-1));
    if(a->fd==(int)e->ident)asking_heard(a);
  }else if(e->udata&TAG_SERVING){
    struct serving *v=servings+(e->udata&(MESH_SERVICES-1));
    char drop[256];
    if(v->fd!=(int)e->ident)return;
    ssize_t got=recv(v->fd,drop,sizeof drop,0);
    if(got==0 || (got<0 && errno!=EAGAIN && errno!=EINTR)){say("service %s: no longer served\n",v->service);close(v->fd);v->fd=-1;}
  }else{
    struct channel *c=channels+((e->udata&~TAG_CHANNEL)>>1);
    const int s=(int)(e->udata&1);
    if(!c->used || c->end[s].kind!=END_SOCKET || c->end[s].fd!=(int)e->ident)return;
    if(e->filter==EVFILT_WRITE)end_drain(c,s);
    else end_read(c,s);
  }
}
/* a port lost: every channel with an end on it fails, each other end told (#requests) */
static void requests_lost(struct port *port,int64_t code){
  for(uint32_t i=0;i<MESH_CHANNELS;i++)for(int s=0;s<2;s++)
    if(channels[i].used && channels[i].end[s].kind==END_PORT && channels[i].end[s].port==port){channel_end(channels+i,code?code:ENETRESET,s);break;}
}
static void requests_init(void){
  for(uint32_t i=0;i<MESH_CHANNELS;i++)askings[i].fd=-1;
  for(uint32_t i=0;i<MESH_SERVICES;i++)servings[i].fd=-1;
}
static void requests_stop(void){
  for(uint32_t i=0;i<MESH_CHANNELS;i++){
    if(channels[i].used)channel_end(channels+i,ESHUTDOWN,-1);
    if(askings[i].fd>=0){close(askings[i].fd);askings[i].fd=-1;}
  }
  for(uint32_t i=0;i<MESH_SERVICES;i++)if(servings[i].fd>=0){close(servings[i].fd);servings[i].fd=-1;}
}

/* design/bridge-and-ipc.md#ledger: a client's request for its program's ledger (MESH_REQUEST_LEDGER, the program's key):
   each path of each running program of the key, its events since its latest cut appended to `path` after its header,
   each SEND cell and record named by its ring, invocation, position, binding and cycle (rdma/ledger.py pairs them
   across the two nodes); ENOENT where no program of the key keeps a ledger, ENOTSUP where the configuration keeps none */
static int ledger_cut(uint64_t key,const char *path){
  if(!ledger_kept)return ENOTSUP;
  FILE *out=NULL;
  int found=0;
  for(struct program *prog=programs;prog;prog=prog->next){
    if(mesh_client_at(M,prog->owner)->key!=key || atomic_load(&prog->state)==FAILED)continue;
    for(uint32_t p=0;p<prog->path_count;p++){
      const struct path *at=prog->paths+p;
      struct ledger *l=at->ledger;
      if(!l)continue;
      if(!out && !(out=fopen(path,"a")))return errno;
      const uint32_t sends=atomic_load_explicit(&l->sends,memory_order_acquire),landings=atomic_load_explicit(&l->landings,memory_order_acquire);
      fprintf(out,"{\"ledger\":%u,\"key\":\"%016llx\",\"node\":%u,\"peer\":%u,\"port\":%u,\"invocations\":%u,\"cyclic\":%u,\"sends\":%u,"
              "\"landings\":%u,\"cut\":[%u,%u]}\n",l->id,(unsigned long long)key,M->node,at->node,at->port,prog->invocations,prog->cyclic,sends,
              landings,l->cut_sends,l->cut_landings);
      for(uint32_t e=sends-l->cut_sends>MESH_LEDGER_EVENTS?sends-MESH_LEDGER_EVENTS:l->cut_sends;e!=sends;e++){
        const struct ledger_send *s=l->send+e%MESH_LEDGER_EVENTS;
        const struct ledger_cell want={.cell=s->cell},*c=l->cell?bsearch(&want,l->cell,l->cells,sizeof *l->cell,ledger_cell_order):NULL;
        fprintf(out,"{\"s\":[%u,%lld,%lld,%lld,%lld,%u,%lld,%llu,%llu,%llu,0,%lld]}\n",l->id,c?(long long)c->ring:-1ll,c?(long long)c->invocation:-1ll,
                c?(long long)c->position:-1ll,c?(long long)c->binding:-1ll,s->cycle,c?(long long)c->chunks:-1ll,(unsigned long long)s->seen,
                (unsigned long long)s->seen,(unsigned long long)s->posted,c?(long long)c->bytes:-1ll);
      }
      for(uint32_t e=landings-l->cut_landings>MESH_LEDGER_EVENTS?landings-MESH_LEDGER_EVENTS:l->cut_landings;e!=landings;e++){
        const struct ledger_landing *r=l->landing+e%MESH_LEDGER_EVENTS;
        const struct prepared_receive *record=(const void *)(uintptr_t)r->record;
        fprintf(out,"{\"r\":[%u,%u,%u,%u,%u,%u,%u,%u,%llu]}\n",l->id,record->ring,record->invocation,record->position,record->binding,record->chunk,
                record->chunks,r->cycle,(unsigned long long)r->landed);
      }
      l->cut_sends=sends;l->cut_landings=landings;
      found=1;
    }
  }
  if(out && fclose(out))return errno;
  return found?0:ENOENT;
}

/* design/bridge-and-ipc.md#ports: this node's port toward neighbour `node` over the cable both ends take alike (the
   lowest index the pair's lower node gave one of its cables to the other), paired; MESH_PORTS where none is */
static uint32_t cable_to(uint32_t node){
  uint32_t best=MESH_PORTS,rank=UINT32_MAX;
  for(uint32_t p=0;p<port_count;p++){
    if(!ports[p].greeted || ports[p].peer!=node || !atomic_load(&ports[p].up))continue;
    const uint32_t at=M->node<node?ports[p].index:ports[p].peer_port;
    if(at<rank){rank=at;best=p;}
  }
  return best;
}

/* design/bridge-and-ipc.md#relays: a relay channel's programs: one a relayed direction pair (mesh-plan.h struct
   mesh_relay_entry), each the bridge's own client slot whose rows it receives into from one neighbour and sends from to
   the other (a landing releases the next hop's SEND cell, receive_input), on the leases its requester chose.  Each ends
   in order as its neighbours let go of their leases (peer_released: what landed from one side sent on, then that
   side's lease let go toward the other); a channel that closes without its requester's word that it ended in order (a
   byte past its request) fails them as an abort */
struct relay_hold { struct relay_hold *next; struct channel *channel; uint32_t count; uint64_t owner[64]; int orderly; size_t asked; };
static struct relay_hold *relays;
static struct program *program_owned(uint64_t owner){
  for(struct program *prog=programs;prog;prog=prog->next)if(prog->owner==owner)return prog;
  return NULL;
}
static void relay_drop(struct relay_hold *hold,int orderly){
  for(uint32_t i=0;i<hold->count;i++){
    struct program *prog=program_owned(hold->owner[i]);
    if(prog && orderly && atomic_load(&prog->state)!=FAILED)continue;
    if(prog){program_fail(prog,ECONNRESET,4,M->node);program_free(prog);}
    mesh_retire(M,hold->owner[i]);
  }
  for(struct relay_hold **at=&relays;*at;at=&(*at)->next)if(*at==hold){*at=hold->next;break;}
  free(hold);
}
/* the lists of one relayed direction pair mirrored into the bridge's slot `c`: each transfer from the far end through
   neighbour `a` received from `a` and sent to `b` over one section, and each the other way received from `b` and sent to
   `a`, its program prepared as its ends' are, and its leases claimed toward each neighbour, free or claimed by a
   program of the same execution not yet started (a member here that is also a relay of its route: the leases it
   reported free were planned for both); 0, or the refusal */
static int mirror(struct mesh_ctx *c,const struct mesh_relay_entry *r,const struct mesh_transfer *t){
  struct mesh_client *client=mesh_client_at(M,c->client);
  const uint32_t pa=cable_to(r->a),pb=cable_to(r->b);
  if(pa==MESH_PORTS || pb==MESH_PORTS)return EHOSTUNREACH;
  int error=0;
  for(int way=0;way<2 && !error;way++){
    const uint32_t from=way?r->b:r->a,to=way?r->a:r->b;
    for(uint32_t k=0;k<MESH_QUEUES && !error;k++)for(uint32_t j=0;j<(way?r->receives[k]:r->sends[k]) && !error;j++,t++){
      struct mesh_section s;
      if(!t->count)error=EPROTO;
      else if(!(error=mesh_section_create(c,mesh_relay_slot(t,r->depth,M->pgsz),t->stride?t->count:1,&s))){
        s.bytes=t->bytes;if(!t->stride){s.stride=0;s.count=t->count;}
        error=mesh_transfer_bind(c,from,k,MESH_RECEIVE,t->binding,s,t->invocation_pages,t->begin,t->end,t->depth);
        if(!error)error=mesh_transfer_bind(c,to,k,MESH_SEND,t->binding,s,t->invocation_pages,t->begin,t->end,t->depth);
      }
    }
  }
  if(!error)error=mesh_transfers_prepare(c,r->slots,r->invocations,r->depth);
  if(!error && r->cyclic)error=mesh_transfers_cyclic(c);
  for(int side=0;side<2 && !error;side++){
    const uint32_t node=side?r->b:r->a,p=side?pb:pa,d=mesh_destination(client,node);
    const uint8_t *lease=side?r->lease_b:r->lease_a;
    if(d==MESH_DESTINATIONS || client->tx[d].node!=node){error=EINVAL;break;}
    client->tx[d].port=(uint8_t)(p+1);
    for(uint32_t q=0;q<MESH_QUEUES && !error;q++){
      client->tx[d].lease[q]=lease[q];
      if(!lease[q])continue;
      if(lease[q]>=ports[p].count){error=EBUSY;break;}
      _Atomic uint64_t *word=mesh_leases(M,p)+lease[q]-1;
      uint64_t held=atomic_load(word);
      const struct mesh_client *holder=held && held!=MESH_LEASE_DOWN?mesh_client_at(M,held):NULL;
      if(holder && (!owned(held) || holder->key!=client->key || atomic_load(&holder->configured)))error=EBUSY;
      else if(held==MESH_LEASE_DOWN || !atomic_compare_exchange_strong(word,&held,c->client))error=EBUSY;
    }
  }
  return error;
}
/* a relay request read whole (mesh-plan.h struct mesh_relay_request and its entries): each entry's slot taken, mirrored
   and started; the requester answered 0 and the channel held, else answered the refusal with every slot of it
   retired and its leases freed */
static int relay_build(struct channel *channel,const unsigned char *bytes,size_t length){
  struct mesh_relay_request head;
  if(length<sizeof head)return EPROTO;
  memcpy(&head,bytes,sizeof head);
  if(head.count>64 || length!=sizeof head+head.bytes)return EPROTO;
  struct relay_hold *hold=calloc(1,sizeof *hold);
  if(!hold)return ENOMEM;
  hold->channel=channel;
  int error=0;
  const unsigned char *at=bytes+sizeof head,*end=at+head.bytes;
  for(uint32_t i=0;i<head.count && !error;i++){
    struct mesh_relay_entry r;
    if((size_t)(end-at)<sizeof r){error=EPROTO;break;}
    memcpy(&r,at,sizeof r);at+=sizeof r;
    uint64_t transfers=0;
    for(uint32_t k=0;k<MESH_QUEUES;k++)transfers+=(uint64_t)r.sends[k]+r.receives[k];
    if((uint64_t)(end-at)<transfers*sizeof(struct mesh_transfer)){error=EPROTO;break;}
    struct mesh_transfer *list=malloc((transfers+1)*sizeof *list);
    if(!list){error=ENOMEM;break;}
    memcpy(list,at,transfers*sizeof *list);at+=transfers*sizeof *list;
    struct mesh_ctx c={.fd=-1};
    if((error=mesh_slot_take(&c,M,head.key))){free(list);break;}
    hold->owner[hold->count++]=c.client;
    if(!(error=mirror(&c,&r,list))){
      struct mesh_client *client=mesh_client_at(M,c.client);
      atomic_store_explicit(&client->configured,c.client,memory_order_release);
      struct program *prog=program_take(c.client,1);
      if(!prog || atomic_load(&prog->state)==FAILED)error=prog?(int)atomic_load(&client->status[0].code):ENOMEM;
      if(!error)say("relay of %016llx between nodes %u and %u (slot %u)\n",(unsigned long long)head.key,r.a,r.b,
                    (uint32_t)(c.client>>32)&(MESH_CLIENTS-1));
    }
    free(list);
  }
  hold->next=relays;relays=hold;
  if(error){
    for(uint32_t i=0;i<hold->count;i++)for(uint32_t p=0;p<port_count;p++)for(uint32_t l=0;l<MESH_LEASES;l++){
      uint64_t held=hold->owner[i];
      atomic_compare_exchange_strong(mesh_leases(M,p)+l,&held,0);
    }
    relay_drop(hold,0);
  }
  return error;
}

/* design/bridge-and-ipc.md#meet: the meeting of a key's `n` members at this bridge: each member's report held until
   every rank of the key has sent one, then each answered with them all in rank order (mesh-plan.h struct mesh_meet) */
struct meeting { struct meeting *next; struct channel *channel; struct mesh_meet head; };
static struct meeting *meetings;
static void meet_drop(struct channel *c){
  for(struct meeting **at=&meetings;*at;)if((*at)->channel==c){struct meeting *gone=*at;*at=gone->next;free(gone);}else at=&(*at)->next;
}
static void end_put(struct channel *c,int t,const void *bytes,size_t n);
static void channel_end(struct channel *c,int64_t code,int from);
static void meet_heard(struct channel *c){
  struct meeting *made=calloc(1,sizeof *made);
  if(!made){channel_end(c,ENOMEM,-1);return;}
  memcpy(&made->head,c->held,sizeof made->head);
  made->channel=c;
  for(struct meeting *m=meetings;m;m=m->next)if(m->head.key==made->head.key && m->head.rank==made->head.rank && m->channel!=c){
    channel_end(m->channel,ECONNRESET,-1);
    break;
  }
  made->next=meetings;meetings=made;
  uint32_t n=0,bytes=0;
  struct meeting *ranked[MESH_MEET_RANKS]={0};
  for(struct meeting *m=meetings;m;m=m->next)if(m->head.key==made->head.key && m->head.n==made->head.n && !ranked[m->head.rank]){
    ranked[m->head.rank]=m;n++;bytes+=m->head.bytes;
  }
  if(n<made->head.n)return;
  const size_t length=sizeof(uint32_t)*(1+(size_t)n)+bytes;
  unsigned char *table=malloc(length);
  if(!table)return;
  struct channel *answer[MESH_MEET_RANKS];
  memcpy(table,&n,sizeof n);
  size_t at=sizeof(uint32_t)*(1+(size_t)n);
  for(uint32_t r=0;r<n;r++){
    memcpy(table+sizeof(uint32_t)*(1+r),&ranked[r]->head.bytes,sizeof(uint32_t));
    memcpy(table+at,ranked[r]->channel->held+sizeof(struct mesh_meet),ranked[r]->head.bytes);
    at+=ranked[r]->head.bytes;
    answer[r]=ranked[r]->channel;
  }
  for(uint32_t r=0;r<n;r++)meet_drop(answer[r]);
  for(uint32_t r=0;r<n;r++){answer[r]->end[1].ended=1;end_put(answer[r],0,table,length);}
  free(table);
}
/* a meet's or a relay's bytes so far (the service end `t` of channel `c`): a meet's report once whole; a relay's request
   once whole, answered, then a byte past it its requester's word that it ends in order */
static void service_heard(struct channel *c,int t){
  if(c->end[t].kind==END_MEET){
    if(c->reported || c->held_bytes<sizeof(struct mesh_meet))return;
    struct mesh_meet head;memcpy(&head,c->held,sizeof head);
    if(!head.n || head.n>MESH_MEET_RANKS || head.rank>=head.n || head.bytes>MESH_MEET_BYTES){channel_end(c,EPROTO,-1);return;}
    if(c->held_bytes<sizeof head+head.bytes)return;
    c->reported=1;
    meet_heard(c);
    return;
  }
  struct relay_hold *hold=NULL;
  for(struct relay_hold *h=relays;h;h=h->next)if(h->channel==c)hold=h;
  if(hold){if(c->held_bytes>hold->asked)hold->orderly=1;return;}
  if(c->reported || c->held_bytes<sizeof(struct mesh_relay_request))return;
  struct mesh_relay_request head;memcpy(&head,c->held,sizeof head);
  if(head.bytes>MESH_MEET_BYTES){channel_end(c,EPROTO,-1);return;}
  if(c->held_bytes<sizeof head+head.bytes)return;
  c->reported=1;
  int32_t code=relay_build(c,c->held,sizeof head+head.bytes);
  if(!code)for(struct relay_hold *h=relays;h;h=h->next)if(h->channel==c)h->asked=sizeof head+head.bytes;
  end_put(c,1-t,&code,sizeof code);
}
/* a meet's or relay's channel ended or its requester gone: its meeting left, its relays dropped (in order where its
   requester said so) */
static void service_gone(struct channel *c){
  meet_drop(c);
  for(struct relay_hold *h=relays;h;h=h->next)if(h->channel==c){relay_drop(h,h->orderly);break;}
}
/* design/prepared-machine.md#M09: the bridge's configuration, the canonical property list bin/mesh-user.sh writes from
   the consumer's document (rdma/bridge_configuration.py): its name and its window in bytes, and whether it serves `exec`
   and keeps ledgers (#ledger), and its `keepalive`, `hello` and `health` intervals in nanoseconds where it states them
   (#keepalive); NULL, or why it is refused */
#ifdef MESH_LOOP_FABRIC
/* design/bridge-dataflow.md#the-loopback-fabric: a loop bridge's `loop` (a dictionary: `region`, `fabric`, `node`,
   `service` and `ports`, each port a dictionary of `device`, `listen` and `peer`); NULL, or why it is refused */
static int loop_number(CFDictionaryRef d,CFStringRef key,int64_t most,int64_t *value){
  CFTypeRef v=CFDictionaryGetValue(d,key);
  return v && CFGetTypeID(v)==CFNumberGetTypeID() && CFNumberGetValue(v,kCFNumberSInt64Type,value) && *value>0 && *value<=most;
}
static int loop_text(CFDictionaryRef d,CFStringRef key,char *out,size_t room){
  CFTypeRef v=CFDictionaryGetValue(d,key);
  return v && CFGetTypeID(v)==CFStringGetTypeID() && CFStringGetCString(v,out,(CFIndex)room,kCFStringEncodingUTF8) && out[0];
}
static const char *loop_read(CFPropertyListRef list){
  CFTypeRef l=list && CFGetTypeID(list)==CFDictionaryGetTypeID()?CFDictionaryGetValue(list,CFSTR("loop")):NULL;
  if(!l || CFGetTypeID(l)!=CFDictionaryGetTypeID())return "a loop bridge's configuration states its `loop`";
  int64_t node=0,service=0;
  if(!loop_text(l,CFSTR("region"),loop_config.region,sizeof loop_config.region) || loop_config.region[0]!='/')return "its loop's region, a shared memory name";
  if(!loop_text(l,CFSTR("fabric"),loop_config.fabric,sizeof loop_config.fabric) || loop_config.fabric[0]!='/')return "its loop's fabric, a shared memory name";
  if(!loop_number(l,CFSTR("node"),0x7fffffff,&node))return "its loop's node, a positive 31-bit number";
  if(!loop_number(l,CFSTR("service"),65535,&service))return "its loop's service, a port";
  loop_config.node=(uint32_t)node;loop_config.service=(uint16_t)service;
  CFTypeRef ports=CFDictionaryGetValue(l,CFSTR("ports"));
  if(!ports || CFGetTypeID(ports)!=CFArrayGetTypeID() || CFArrayGetCount(ports)>MESH_PORTS)return "its loop's ports, at most MESH_PORTS";
  loop_config.ports=(uint32_t)CFArrayGetCount(ports);
  for(uint32_t k=0;k<loop_config.ports;k++){
    CFTypeRef p=CFArrayGetValueAtIndex(ports,k);
    int64_t listen=0,peer=0;
    if(!p || CFGetTypeID(p)!=CFDictionaryGetTypeID() || !loop_text(p,CFSTR("device"),loop_config.port[k].device,sizeof loop_config.port[k].device) ||
       strncmp(loop_config.port[k].device,"rdma_",5) || !loop_number(p,CFSTR("listen"),65535,&listen) || !loop_number(p,CFSTR("peer"),65535,&peer) ||
       listen==peer)return "a loop port: its device (rdma_...) and its cable's two ports, apart";
    loop_config.port[k].listen=(uint16_t)listen;loop_config.port[k].peer=(uint16_t)peer;loop_config.port[k].fd=-1;
  }
  return NULL;
}
#endif
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
  CFTypeRef executes=list && CFGetTypeID(list)==CFDictionaryGetTypeID()?CFDictionaryGetValue(list,CFSTR("exec")):NULL;
  exec_served=executes && CFGetTypeID(executes)==CFBooleanGetTypeID() && CFBooleanGetValue(executes);
  /* design/bridge-and-ipc.md#ledger */
  CFTypeRef kept=list && CFGetTypeID(list)==CFDictionaryGetTypeID()?CFDictionaryGetValue(list,CFSTR("ledger")):NULL;
  ledger_kept=kept && CFGetTypeID(kept)==CFBooleanGetTypeID() && CFBooleanGetValue(kept);
  /* design/bridge-and-ipc.md#keepalive */
  static const CFStringRef paced[3]={CFSTR("keepalive"),CFSTR("hello"),CFSTR("health")};
  int64_t value=0,interval[3]={0};
  if(!list || CFGetTypeID(list)!=CFDictionaryGetTypeID())refused="not a property list of a configuration (rdma/bridge_configuration.py writes it)";
  else if(!named || CFGetTypeID(named)!=CFStringGetTypeID() || !CFStringGetCString(named,name,(CFIndex)room,kCFStringEncodingUTF8))refused="its name is missing";
  else if(!sized || CFGetTypeID(sized)!=CFNumberGetTypeID() || !CFNumberGetValue(sized,kCFNumberSInt64Type,&value) || value<=0)
    refused="its window, a positive number of bytes, is missing";
  for(int k=0;k<3 && !refused;k++){
    CFTypeRef stated=CFDictionaryGetValue((CFDictionaryRef)list,paced[k]);
    if(stated && (CFGetTypeID(stated)!=CFNumberGetTypeID() || !CFNumberGetValue(stated,kCFNumberSInt64Type,interval+k) || interval[k]<=0))
      refused="its keepalive, hello or health interval, where stated, is not a positive number of nanoseconds";
  }
#ifdef MESH_LOOP_FABRIC
  if(!refused)refused=loop_read(list);
#endif
  keepalive_ns=refused?0:(uint64_t)interval[0];hello_ns=refused?0:(uint64_t)interval[1];health_ns=refused?0:(uint64_t)interval[2];
  if(list)CFRelease(list);
  *window=(uint64_t)value;
  return refused;
}

/* design/bridge-and-ipc.md#ports: the node's name, its LocalHostName (else its host name), read at the start and at each
   change of the host names configd notifies, on the finder's queue (configd answers there, never on the event loop:
   host_watch); the event loop takes a change (host_taken), so a renamed node's next record carries its new name (the
   map's label of it; a node is its number) */
static void host_name(char *name){
  CFStringRef local=SCDynamicStoreCopyLocalHostName(NULL);
  if(!local || !CFStringGetCString(local,name,MESH_NAME_BYTES,kCFStringEncodingUTF8))gethostname(name,MESH_NAME_BYTES);
  if(local)CFRelease(local);
  name[MESH_NAME_BYTES-1]=0;
}
static char host_heard[MESH_NAME_BYTES];
static _Atomic int host_read_done;
static void host_changed(SCDynamicStoreRef store,CFArrayRef keys,void *info){
  (void)store;(void)keys;(void)info;
  char name[MESH_NAME_BYTES]={0};
  host_name(name);
  memcpy(host_heard,name,sizeof name);
  atomic_store_explicit(&host_read_done,1,memory_order_release);
  wake();
}
static void host_watch(void){
#ifdef MESH_LOOP_FABRIC
  /* design/bridge-dataflow.md#the-loopback-fabric: a loop bridge's name is its configuration's, never the host's */
  return;
#endif
  SCDynamicStoreRef store=SCDynamicStoreCreate(NULL,CFSTR("io.mesh.bridge"),host_changed,NULL);
  CFStringRef key=store?SCDynamicStoreKeyCreateHostNames(NULL):NULL;
  CFArrayRef keys=key?CFArrayCreate(NULL,(const void **)&key,1,&kCFTypeArrayCallBacks):NULL;
  if(!keys || !SCDynamicStoreSetNotificationKeys(store,keys,NULL) || !SCDynamicStoreSetDispatchQueue(store,finder))
    say("node %u: host names not watched: %s\n",M->node,SCErrorString(SCError()));
  if(keys)CFRelease(keys);
  if(key)CFRelease(key);
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


/* design/bridge-and-ipc.md#wire: a message to the port's peer, queued behind what the socket has not taken yet; a send
   that fails is the link's loss */
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
  said.magic=XMAGIC;said.node=M->node;said.block=M->block*M->pgsz;said.port=port->index;said.bytes=bytes;
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

/* design/bridge-and-ipc.md#links: the link's queue pairs made once at its pairing: its two completion queues and its
   device's usable queue pairs (at most LINK_PAIRS), each its share of the completion queues' entries, every pair's
   number told the peer (PAIRS); 0, or the errno */
static int link_make(struct port *port){
  struct mesh_device *device=&port->device;
  const uint32_t n=device->max_pairs<LINK_PAIRS?device->max_pairs:LINK_PAIRS;
  if(n<2)return ENOSPC;
  uint64_t entries=(uint64_t)n*device->frame_capacity;
  if(entries>device->max_cqe)entries=device->max_cqe;
  int error=link_cqs(device,(uint32_t)entries,&port->sent,&port->received);
  if(error)return error;
  port->share=(uint32_t)(entries/n);
  guard_raise(port);
  for(uint32_t k=0;k<n && !error;k++){
    port->pair[k]=(struct pair){.round=UINT32_MAX};
    if(!(error=pair_new(port,k)))port->made=k+1;
  }
  guard_lower(port);
  if(!error)pairs_tell(port,LINK_PAIRS,0,0);
  return error;
}
/* the link's queue pairs and completion queues destroyed (the error state, then destroy), under the guard */
static void link_unmake(struct port *port){
  guard_raise(port);
  for(uint32_t k=0;k<port->made;k++){
    const int error=pair_destroy(&port->device,port->pair[k].qp);
    if(error)say("link %u (%s): pair %u's destruction refused: %s\n",port->index,port->name,k,strerror(error));
    port->pair[k]=(struct pair){.round=UINT32_MAX};
  }
  if(port->sent && ibv_destroy_cq(port->sent))say("link %u (%s): a completion queue's destruction refused\n",port->index,port->name);
  if(port->received && ibv_destroy_cq(port->received))say("link %u (%s): a completion queue's destruction refused\n",port->index,port->name);
  port->sent=port->received=NULL;port->made=port->count=port->holding=0;
  guard_lower(port);
}
/* design/bridge-and-ipc.md#links: the peer's PAIRS.  At the link's pairing, each pair joined to the peer's of its index
   (the link's count the fewer the two ends made), the leases free and the link up; after, a pair the peer made again:
   joined where this end's is of the same making; else, where a program holds this end's, its program failed as a value
   (the peer's lost its holder with `code`) or, the peer having let go in order, taken up once its holder's `sent` SENDs
   have landed here (pairs_turn); else this end's made again */
static void pairs_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  if(heard->bytes%sizeof(struct pairs_entry)){port_lost(port,EPROTO,0);return;}
  const uint32_t n=heard->bytes/(uint32_t)sizeof(struct pairs_entry);
  struct pairs_entry entry[LINK_PAIRS];
  if(n>LINK_PAIRS){port_lost(port,EPROTO,0);return;}
  memcpy(entry,payload,n*sizeof *entry);
  if(!port->count){
    const uint32_t count=port->made<heard->count?port->made:heard->count;
    for(uint32_t i=0;i<n;i++)if(entry[i].index<count){
      struct pair *pair=port->pair+entry[i].index;
      pair->peer_qpn=entry[i].qpn;pair->peer_psn=entry[i].psn;pair->peer_round=entry[i].round;pair->heard=1;
    }
    int error=count<2?ENOSPC:0;
    guard_raise(port);
    for(uint32_t k=0;k<count && !error;k++)error=pair_join(port,k);
    guard_lower(port);
    if(error){port_lost(port,error,0);return;}
    port->count=count;
    for(uint32_t l=0;l<MESH_LEASES;l++)atomic_store(mesh_leases(M,port->index)+l,l+1<count?0:MESH_LEASE_DOWN);
    atomic_store_explicit(&mesh_links(M)[port->index].leases,count-1,memory_order_relaxed);
    if((error=port_threads(port))){port_lost(port,error,0);return;}
    atomic_store_explicit(&port->up,1,memory_order_release);
    bell_ring(&port->bell);
    say("link %u (%s): node %u paired: %u queue pairs, %u leases, %u frames a pair\n",port->index,port->name,port->peer,count,count-1,
        port->share);
    return;
  }
  for(uint32_t i=0;i<n;i++){
    const uint32_t k=entry[i].index;
    if(k>=port->count)continue;
    struct pair *pair=port->pair+k;
    if(entry[i].round<pair->round)continue;
    if(entry[i].round>pair->round+1)pair->round=entry[i].round-1;
    pair->peer_qpn=entry[i].qpn;pair->peer_psn=entry[i].psn;pair->peer_round=entry[i].round;pair->heard=1;
    if(entry[i].round==pair->round){
      guard_raise(port);
      const int error=pair_join(port,k);
      guard_lower(port);
      if(error)port_lost(port,error,0);
      continue;
    }
    if(!k){port_lost(port,EPROTO,0);return;}
    if(pair->path && entry[i].code)program_fail(pair->path->program,entry[i].code,4,port->peer);
    else if(pair->path){pair->peer_gone=1;atomic_store(&pair->peer_sent,entry[i].sent);}
    else lease_remake(port,k,0);
  }
}

/* design/bridge-and-ipc.md#wire */
static void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes);
static void channel_heard(struct port *port,const struct message *heard,const unsigned char *payload);
static void port_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  if(!port->greeted){
    if(heard->kind!=HELLO || heard->block!=M->block*M->pgsz || (port->peer && heard->node!=port->peer)){port_lost(port,EPROTO,0);return;}
    port->peer=heard->node;port->peer_port=heard->port;port->heard=*heard;port->greeted=1;
    struct mesh_link_info *link=&mesh_links(M)[port->index];
    atomic_store_explicit(&link->bandwidth,port->bandwidth,memory_order_relaxed);
    atomic_store_explicit(&link->peer_port,heard->port,memory_order_relaxed);
    atomic_fetch_add_explicit(&link->pairings,1,memory_order_relaxed);
    const int error=link_make(port);
    if(error){port_lost(port,error,0);return;}
    link_set(port,port->peer,MESH_PAIRED,0);
    for(uint64_t i=0;port->greeted && i<atomic_load(&M->map_nodes);i++)if(mesh_records(M)[i].node!=M->node)
      port_say(port,(struct message){.kind=RECORD},mesh_records(M)+i,sizeof(struct mesh_record));
    return;
  }
  if(heard->kind==RECORD){record_heard(port,payload,heard->bytes);return;}
  if(heard->kind==PAIRS){pairs_heard(port,heard,payload);return;}
  /* design/recovery.md#detection: a peer's FAREWELL is its bridge's stop: the link is lost now */
  if(heard->kind==FAREWELL){port->parted=1;port->farewell=heard->code;port_lost(port,ECONNRESET,0);return;}
  if(heard->kind>=CHANNEL_OPEN && heard->kind<=CHANNEL_CLOSE){channel_heard(port,heard,payload);return;}
  port_lost(port,EPROTO,0);
}

/* design/recovery.md#detection, #loss: a link lost: the link down, every program holding a lease of it failed as a value
   naming its peer (their other leases made again, their peers told), the link's queue pairs and completion queues
   destroyed, its leases down; a port that went away retires its device (design/RDMA-KERNEL-RECOVERY.md); a link that
   had paired says HELLO again at once, one that never paired at the next event */
static void requests_lost(struct port *port,int64_t code);
static void port_lost(struct port *port,int64_t code,int retire){
  if(port->control>=0){close(port->control);port->control=-1;}
  if(port->greeted||port->connecting){
    const char *retired=retire?", device retired":"";
    char reason[96];farewell_text(port->farewell,reason,sizeof reason);
    if(stop)say("link %u (%s): closed toward node %u: this bridge is stopping%s\n",port->index,port->name,port->peer,retired);
    else if(port->peer && port->parted)say("link %u (%s): node %u parted, its bridge stopping (%s): code %lld%s\n",port->index,port->name,
                                            port->peer,reason,(long long)code,retired);
    else if(port->peer)say("link %u (%s): node %u lost: code %lld%s\n",port->index,port->name,port->peer,(long long)code,retired);
    else say("link %u (%s): lost: code %lld%s\n",port->index,port->name,(long long)code,retired);
  }
  atomic_store_explicit(&port->up,0,memory_order_release);
  for(uint32_t l=0;l<MESH_LEASES;l++)atomic_store(mesh_leases(M,port->index)+l,MESH_LEASE_DOWN);
  atomic_store_explicit(&mesh_links(M)[port->index].leases,0,memory_order_relaxed);
  for(struct program *prog=programs;prog;prog=prog->next)
    for(uint32_t p=0;p<prog->path_count;p++)
      if(prog->paths[p].port==port->index && atomic_load(&prog->state)!=FAILED){
        program_fail(prog,port->parted?ECONNRESET:code,4,port->peer);
        break;
      }
  if(port->made)link_unmake(port);
  requests_lost(port,port->parted?ECONNRESET:code);
  port->wanted|=port->greeted || retire;
  port->greeted=port->parted=port->connecting=port->writing=0;port->farewell=0;port->peer=0;port->held=0;port->out_length=port->out_sent=0;
  port->retiring|=retire;port->open&=!retire;
  link_set(port,0,retire?MESH_STOPPED:MESH_PAIRING,code);
}
/* design/bridge-and-ipc.md#ports: a control socket taken (dialed or accepted): non-blocking, its sends queued, its HELLO
   said */
static void port_greet(struct port *port){
  int on=1,flags=fcntl(port->control,F_GETFL);
  if(flags<0 || fcntl(port->control,F_SETFL,flags|O_NONBLOCK) || setsockopt(port->control,SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on)){
    port_lost(port,errno?errno:EIO,0);return;
  }
  port_watch(port,EVFILT_READ);
  port_say(port,(struct message){.kind=HELLO},NULL,0);
}
static struct port *port_at(uint32_t interface){
  for(uint32_t p=0;interface && p<port_count;p++)if(ports[p].interface==interface)return ports[p].retiring?NULL:ports+p;
  return NULL;
}
/* design/bridge-and-ipc.md#ports: a port's turn, where an event asked it: on its own queue, its device retired where its
   port went away, else opened and the window registered on it (it stops between extents at `stop`) */
static void port_turn(struct port *port){
  if(!port->wanted || port->job || port->control>=0 || port->made)return;
  port->wanted=0;
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
/* design/recovery.md#registration: a port's turn taken: a registration refused turns again at the next memory-pressure
   change or link event; a device that failed otherwise is retired and opened again; an open port with no peer says
   HELLO to every node on its link (a loop port's lower end dials the higher's listener) */
static void port_opened(struct port *port){
  const int result=port->result,retired=port->retiring;
  port->job=0;
  if(result<0){say("link %u (%s): device retirement: %s\n",port->index,port->name,strerror(-result));return;}
  if(port->refused){
    say("link %u (%s): the window's registration refused (%s): the port turns again at its next memory-pressure change or link event\n",
        port->index,port->name,strerror(port->refused));
    port->open=0;if(port->control<0)link_set(port,0,MESH_STOPPED,result);
    return;
  }
  port->retiring=result && port->device.context && result!=ENETDOWN && result!=ECANCELED;
  port->wanted|=port->retiring && !retired;
  if(result){port->open=0;if(port->control<0)link_set(port,0,MESH_STOPPED,result);return;}
  atomic_store_explicit(&mesh_links(M)[port->index].max_pairs,port->device.max_pairs,memory_order_relaxed);
  port->open=1;port->lid=port->device.port.lid;memcpy(port->gid,&port->device.gid,16);
  port->bandwidth=port_bandwidth(&port->device.port);
  if(port->control>=0)return;
  link_set(port,0,MESH_PAIRING,0);
#ifdef MESH_LOOP_FABRIC
  for(uint32_t k=0;k<loop_config.ports;k++)if(!strcmp(loop_config.port[k].device,port->device.name) && loop_config.port[k].listen<loop_config.port[k].peer){
    struct sockaddr_in6 to={.sin6_len=sizeof to,.sin6_family=AF_INET6,.sin6_port=htons(loop_config.port[k].peer),.sin6_addr=IN6ADDR_LOOPBACK_INIT};
    port->control=socket(AF_INET6,SOCK_STREAM,0);
    if(port->control<0 || fcntl(port->control,F_SETFL,O_NONBLOCK) || (connect(port->control,(struct sockaddr *)&to,sizeof to) && errno!=EINPROGRESS)){
      port_lost(port,errno,0);return;
    }
    port->peer=0;port->connecting=1;
    port_watch(port,EVFILT_WRITE);
  }
  return;
#endif
  struct ipv6_mreq group={.ipv6mr_multiaddr=everyone,.ipv6mr_interface=port->interface};
  setsockopt(hello,IPPROTO_IPV6,IPV6_JOIN_GROUP,&group,sizeof group);
  setsockopt(hello,IPPROTO_IPV6,IPV6_MULTICAST_IF,&port->interface,sizeof port->interface);
  struct sockaddr_in6 to={.sin6_len=sizeof to,.sin6_family=AF_INET6,.sin6_port=htons((uint16_t)atoi(MESH_PORT)),.sin6_addr=everyone,.sin6_scope_id=port->interface};
  struct message said={.magic=XMAGIC,.kind=HELLO,.node=M->node,.block=M->block*M->pgsz,.port=port->index};
  sendto(hello,&said,sizeof said,0,(struct sockaddr *)&to,sizeof to);
}
/* design/bridge-and-ipc.md#ports: a node heard on a port: the lower node dials the higher's listener on the address it
   heard from, the higher answers its hello so the lower hears it at once */
static void hello_heard(void){
  struct message heard;struct sockaddr_in6 from;socklen_t length=sizeof from;
  while(recvfrom(hello,&heard,sizeof heard,0,(struct sockaddr *)&from,&length)==(ssize_t)sizeof heard){
    struct port *port=port_at(from.sin6_scope_id);
    length=sizeof from;
    if(!port || heard.magic!=XMAGIC || heard.kind!=HELLO || heard.node==M->node || port->control>=0 || !port->open)continue;
    if(M->node>heard.node){
      struct message said={.magic=XMAGIC,.kind=HELLO,.node=M->node,.block=M->block*M->pgsz,.port=port->index};
      sendto(hello,&said,sizeof said,0,(struct sockaddr *)&from,sizeof from);
      continue;
    }
    from.sin6_port=htons((uint16_t)atoi(MESH_PORT));
    port->control=socket(AF_INET6,SOCK_STREAM,0);
    if(port->control<0 || fcntl(port->control,F_SETFL,O_NONBLOCK) ||
       (connect(port->control,(struct sockaddr *)&from,sizeof from) && errno!=EINPROGRESS)){port_lost(port,errno,0);continue;}
    port->peer=heard.node;port->connecting=1;
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
   (at the start, and where an interface attached or its link came on that no port has) and added on the event loop,
   each with a page of the window for its keepalive frames */
static void ports_find(void){
  if(finding)return;
  finding=1;
  dispatch_async(finder,^{
    struct ibv_device **list=ibv_get_device_list(NULL);
    uint32_t n=0;
#ifdef MESH_LOOP_FABRIC
    for(uint32_t k=0;k<loop_config.ports && n<MESH_PORTS;k++)snprintf(devices[n++],sizeof devices[0],"%s",loop_config.port[k].device);
    if(list)ibv_free_device_list(list);
    list=NULL;
#endif
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
    const uint32_t page=mesh_arena_alloc(&bridge,1,1);
    if(page==MESH_ABSENT){say("port %s: no page of the window for its keepalive frames; found again at the next event\n",devices[i]);continue;}
    struct port *port=ports+port_count;
    *port=(struct port){.index=port_count,.control=-1,.wanted=1,.page=page,.work=dispatch_queue_create("mesh.port",DISPATCH_QUEUE_SERIAL)};
    snprintf(port->device.name,sizeof port->device.name,"%s",devices[i]);snprintf(port->name,sizeof port->name,"%s",devices[i]+5);
    port_count++;
    for(uint32_t l=0;l<MESH_LEASES;l++)atomic_store(mesh_leases(M,port->index)+l,MESH_LEASE_DOWN);
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
      else if(code==KEV_DL_LINK_ON || code==KEV_DL_IF_ATTACHED)ports[p].wanted=1;
    }
    unknown|=!known && (code==KEV_DL_LINK_ON || code==KEV_DL_IF_ATTACHED);
  }
  if(unknown)ports_find();
}

/* design/bridge-and-ipc.md#clients, #programs: every client slot read: a slot's process watched from the turn its slot
   is first seen (its exit retires every slot it holds, EVFILT_PROC); a started program taken (program_take); a program
   whose slot retired in order ends once drained (pairs_turn), one aborted (its process gone, a revoke) fails as a value
   at once, telling its peers, and one failed is freed once its slot retires; a lease claimed by an owner gone and held
   by no pair is free again; rows of retired clients released once no program of one holds a lease */
static uint64_t owners[MESH_CLIENTS],accepted[MESH_CLIENTS];
static void clients_turn(void){
  const pid_t self=getpid();
  int changed=0;
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct mesh_client *client=(struct mesh_client *)((char *)M+M->client_off)+s;
    uint64_t owner=atomic_load(&client->owner);
    if(owner && (pid_t)(uint32_t)owner==self)continue;
    struct program *prog=NULL;
    for(struct program *p=programs;p;p=p->next)if(!p->relay && ((p->owner>>32)&(MESH_CLIENTS-1))==s)prog=p;
    if(prog && owner!=prog->owner){
      const uint64_t base=prog->owner;
      if(owner==(base|MESH_RETIRING|MESH_ORDERLY)){
        int running=RUNNING;
        if(atomic_compare_exchange_strong(&prog->state,&running,ENDING))
          for(uint32_t q=0;q<prog->path_count;q++)atomic_store(&prog->paths[q].closing,1);
        else if(running==FAILED){
          program_free(prog);prog=NULL;
          atomic_compare_exchange_strong(&client->owner,&owner,0);
          owner=0;
        }
      }else if(owner!=(base|MESH_RETIRING)){
        program_fail(prog,ECANCELED,4,M->node);
        program_free(prog);prog=NULL;
      }
    }
    if((owner&MESH_ORDERLY) && !prog && atomic_compare_exchange_strong(&client->owner,&owner,0))owner=0;
    if(owner && owner!=owners[s] && !(owner&MESH_RETIRING)){
      struct kevent64_s watch;EV_SET64(&watch,(uint32_t)owner,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,s,0,0);
      if(kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)<0 && errno==ESRCH){mesh_retire(M,owner);owner=atomic_load(&client->owner);}
    }
    changed|=owner!=owners[s];owners[s]=owner;
    if(!owner || (owner&MESH_RETIRING) || prog || accepted[s]==owner ||
       atomic_load_explicit(&client->configured,memory_order_acquire)!=owner)continue;
    accepted[s]=owner;
    program_take(owner,0);
  }
  for(uint32_t p=0;p<port_count;p++)for(uint32_t l=0;l+1<ports[p].count;l++){
    uint64_t held=atomic_load(mesh_leases(M,p)+l);
    if(held && held!=MESH_LEASE_DOWN && !owned(held) && !ports[p].pair[l+1].path)atomic_compare_exchange_strong(mesh_leases(M,p)+l,&held,0);
  }
  int zombie=0;
  for(struct program *prog=programs;prog;prog=prog->next)zombie|=!owned(prog->owner);
  unreleased|=changed;
  if(unreleased && !zombie){unreleased=0;mesh_retired_release(M);}
}

/* design/prepared-machine.md#M26: the clients' control word, watched for the event loop until `stop` */
static void *control_watch(void *unused){
  (void)unused;
  for(;;){
    uint64_t seen=atomic_load_explicit(&M->control,memory_order_acquire);
    if(stop)return NULL;
    wake();
    os_sync_wait_on_address(&M->control,seen,sizeof M->control,OS_SYNC_WAIT_ON_ADDRESS_SHARED);
  }
}
/* design/bridge-and-ipc.md#stop: a signal's one effect, the stop flag (every loop reads it every pass) and its reason */
static void stopped(int signal){ stop_reason=signal;stop=1; }

/* design/bridge-and-ipc.md: the node's resident bridge; its one operand the path of its configuration (no default) */
int main(int argc,char **argv){
  if(argc!=2)die("usage: mesh-flow CONFIGURATION: the canonical property list bin/mesh-user.sh writes from the node's bridge configuration document (etc/bridge/README.md); the bridge has no default and does not start without one");
  char name[256];uint64_t window;
  const char *refused=configuration_read(argv[1],name,sizeof name,&window);
  if(refused){say("bridge configuration %s refused: %s\n",argv[1],refused);exit(1);}
  /* design/bridge-and-ipc.md#stop: the bridge's binary watched (EVFILT_VNODE), so its change is an event */
  const int binary=open(argv[0],O_EVTONLY);
  uuid_t uuid;const struct timespec forever={0,0};
  if(gethostuuid(uuid,&forever)){say("host identity: %s\n",strerror(errno));exit(1);}
  uint32_t node=0;
  for(int i=0;i<16;i++)node=(node*0x01000193u)^uuid[i];
  node=(node&0x7fffffffu)?node&0x7fffffffu:1;
#ifdef MESH_LOOP_FABRIC
  node=loop_config.node;
  const char *region=loop_config.region;
  const uint16_t service=loop_config.service;
  loop_fabric(loop_config.fabric);
#else
  const char *region=MESH_NAME;
  const uint16_t service=(uint16_t)atoi(MESH_PORT);
#endif
  const uint32_t pg=(uint32_t)getpagesize();
  /* design/collective-dependency-ledger.md#d6-paired-send-and-receive-frame-counts-match */
  if((uint64_t)pg+4096>16773120)die("transport chunk exceeds native request capacity");
  if(window<pg || window/pg>UINT32_MAX){say("bridge configuration %s refused: its window of %llu bytes is not between one page and 2^32 pages\n",name,(unsigned long long)window);exit(1);}
  struct hdr geometry={0};
  const uint64_t length=mesh_layout(&geometry,pg,1,(uint32_t)(window/pg));
  struct sigaction on_signal={.sa_handler=stopped};
  sigaction(SIGINT,&on_signal,NULL);sigaction(SIGTERM,&on_signal,NULL);sigaction(SIGHUP,&on_signal,NULL);
  signal(SIGPIPE,SIG_IGN);
  atexit(down);
  shm_unlink(region);int fd=shm_open(region,O_CREAT|O_RDWR,MESH_MODE);if(fd<0)die("shm");
  if(ftruncate(fd,(off_t)length))die("ftruncate");fchmod(fd,MESH_MODE);
  M=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  if(M==MAP_FAILED)die("mmap");shm=region;
  *M=geometry;M->node=node;M->version=MESH_VERSION;
  for(uint32_t r=0;r<mesh_rows(M);r++)atomic_store_explicit(&mesh_page(M)[r].mapping,MESH_ABSENT,memory_order_relaxed);
  for(uint32_t p=0;p<MESH_PORTS;p++)for(uint32_t l=0;l<MESH_LEASES;l++)atomic_store(mesh_leases(M,p)+l,MESH_LEASE_DOWN);
  if(wire_map(&wire,M,fd))die("transport page aliases");
  close(fd);
  bridge.M=M;
  host_name(host);
#ifdef MESH_LOOP_FABRIC
  snprintf(host,sizeof host,"%s",name);
#endif
  finder=dispatch_queue_create("mesh.ports",DISPATCH_QUEUE_SERIAL);
  size_t size=sizeof memory;
  if(sysctlbyname("hw.memsize",&memory,&size,NULL,0))memory=0;
  sysctl_text("hw.model",model,sizeof model);
  events=kqueue();
  health_init();
  record_changed();
  if(!atomic_load(&M->map_nodes))die("the map's lease");
  struct kevent64_s event;
  int on=1;
  struct sockaddr_in6 any={.sin6_len=sizeof any,.sin6_family=AF_INET6,.sin6_port=htons(service),.sin6_addr=IN6ADDR_ANY_INIT};
#ifdef MESH_LOOP_FABRIC
  any.sin6_addr=in6addr_loopback;
#endif
  listener=socket(AF_INET6,SOCK_STREAM,0);hello=socket(AF_INET6,SOCK_DGRAM,0);network=socket(PF_SYSTEM,SOCK_RAW,SYSPROTO_EVENT);
  struct kev_request filter={KEV_VENDOR_APPLE,KEV_NETWORK_CLASS,KEV_DL_SUBCLASS};
  unsigned loop=0;
  if(events<0 || listener<0 || hello<0 || network<0 || setsockopt(listener,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on) ||
     setsockopt(hello,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on) || setsockopt(hello,IPPROTO_IPV6,IPV6_MULTICAST_LOOP,&loop,sizeof loop) ||
     bind(listener,(struct sockaddr *)&any,sizeof any) || listen(listener,SOMAXCONN) || bind(hello,(struct sockaddr *)&any,sizeof any) ||
     fcntl(listener,F_SETFL,O_NONBLOCK) || fcntl(hello,F_SETFL,O_NONBLOCK) || fcntl(network,F_SETFL,O_NONBLOCK) || ioctl(network,SIOCSKEVFILT,&filter))
    die("bridge sockets");
  int watched[3]={listener,hello,network};
  for(int i=0;i<3;i++){EV_SET64(&event,(uint64_t)watched[i],EVFILT_READ,EV_ADD,0,0,UINT64_MAX,0,0);if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");}
  EV_SET64(&event,0,EVFILT_USER,EV_ADD|EV_CLEAR,0,0,UINT64_MAX,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");
  int signals[3]={SIGINT,SIGTERM,SIGHUP};
  for(int i=0;i<3;i++){EV_SET64(&event,(uint64_t)signals[i],EVFILT_SIGNAL,EV_ADD,0,0,UINT64_MAX,0,0);kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);}
  if(binary>=0){
    EV_SET64(&event,(uint64_t)binary,EVFILT_VNODE,EV_ADD|EV_CLEAR,NOTE_DELETE|NOTE_WRITE|NOTE_EXTEND|NOTE_RENAME|NOTE_REVOKE,0,UINT64_MAX,0,0);
    kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
  }
  const int notified[2]={power_fd,thermal_fd};
  for(int i=0;i<2;i++)if(notified[i]>=0){
    EV_SET64(&event,(uint64_t)notified[i],EVFILT_READ,EV_ADD,0,0,UINT64_MAX,0,0);kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
  }
  const uint64_t interval[2]={hello_ns,health_ns};
  for(int i=0;i<2;i++)if(interval[i]){
    EV_SET64(&event,(uint64_t)(HELLO_TIMER+i),EVFILT_TIMER,EV_ADD,NOTE_NSECONDS,(int64_t)interval[i],UINT64_MAX,0,0);
    kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
  }
  /* design/recovery.md#registration: a memory-pressure change, the event a refused registration turns at */
  dispatch_source_t pressure=dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE,0,
    DISPATCH_MEMORYPRESSURE_NORMAL|DISPATCH_MEMORYPRESSURE_WARN|DISPATCH_MEMORYPRESSURE_CRITICAL,finder);
  if(pressure){dispatch_source_set_event_handler(pressure,^{atomic_store_explicit(&pressured,1,memory_order_release);wake();});dispatch_resume(pressure);}
  atomic_store(&M->bridge_pid,(uint64_t)getpid());
  __sync_synchronize();M->magic=MESH_MAGIC;
  requests_init();
  pthread_t watcher;
  if(pthread_create(&watcher,NULL,control_watch,NULL))die("control watcher");
#ifdef MESH_LOOP_FABRIC
  for(uint32_t k=0;k<loop_config.ports;k++){
    struct sockaddr_in6 at={.sin6_len=sizeof at,.sin6_family=AF_INET6,.sin6_port=htons(loop_config.port[k].listen),.sin6_addr=IN6ADDR_LOOPBACK_INIT};
    int cable=socket(AF_INET6,SOCK_STREAM,0);
    if(cable<0 || setsockopt(cable,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on) || bind(cable,(struct sockaddr *)&at,sizeof at) || listen(cable,SOMAXCONN) ||
       fcntl(cable,F_SETFL,O_NONBLOCK))die("a loop cable's listener");
    loop_config.port[k].fd=cable;
    EV_SET64(&event,(uint64_t)cable,EVFILT_READ,EV_ADD,0,0,TAG_LOOP|k,0,0);
    if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");
  }
#endif
  ports_find();
  host_watch();
  char started[96];
#ifdef MESH_LOOP_FABRIC
  snprintf(started,sizeof started,"%s%s",MESH_STARTED,region);
#else
  snprintf(started,sizeof started,"%s",MESH_STARTED);
#endif
  notify_post(started);
  say("bridge node %u (%s): configuration %s, window %llu bytes, arena %llu pages, rows %u; keepalive %llu ns, HELLO %llu ns, health %llu ns%s\n",
      node,host,name,(unsigned long long)window,(unsigned long long)mesh_arena_pages(M),M->rows,(unsigned long long)keepalive_ns,
      (unsigned long long)hello_ns,(unsigned long long)health_ns,ledger_kept?", ledgers kept":"");
  int flood=0;
  while(!stop){
    if(atomic_load_explicit(&found,memory_order_acquire))ports_found();
    const int moved=atomic_exchange_explicit(&pressured,0,memory_order_acquire);
    for(uint32_t p=0;p<port_count;p++){
      struct port *port=ports+p;
      if(port->job && atomic_load_explicit(&port->done,memory_order_acquire))port_opened(port);
      if(moved && port->refused){port->refused=0;port->wanted=1;}
      port_turn(port);
    }
    pairs_turn();
    health_turn(flood);
    flood=0;
    host_taken();
    clients_turn();
    struct kevent64_s ready[16];
    const int count=stop?0:kevent64(events,NULL,0,ready,16,0,NULL);
    for(int i=0;i<count && !stop;i++){
      struct kevent64_s *e=ready+i;
      if(e->filter==EVFILT_SIGNAL){stop_reason=(int64_t)e->ident;stop=1;}
      else if(e->filter==EVFILT_VNODE){stop_reason=MESH_STOP_BINARY;stop=1;}
      else if(e->filter==EVFILT_TIMER && e->ident==HELLO_TIMER){
        for(uint32_t p=0;p<port_count;p++)if(!ports[p].greeted && ports[p].control<0 && !ports[p].refused)ports[p].wanted=1;
      }
      else if(e->filter==EVFILT_TIMER)flood=1;
      else if(e->filter==EVFILT_READ && (e->ident==(uint64_t)power_fd || e->ident==(uint64_t)thermal_fd)){
        int token;
        while(read((int)e->ident,&token,sizeof token)==(ssize_t)sizeof token){}
        if(e->ident==(uint64_t)power_fd)power_read(&health);
      }
      else if(e->filter==EVFILT_PROC){
        child_heard((pid_t)e->ident);
        for(uint32_t s=0;s<MESH_CLIENTS;s++){
          const uint64_t owner=atomic_load(&((struct mesh_client *)((char *)M+M->client_off))[s].owner);
          if(owner && (uint32_t)owner==(uint32_t)e->ident)mesh_retire(M,owner);
        }
      }
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)listener){
        struct sockaddr_in6 from;socklen_t from_size=sizeof from;int accepted_fd;
        while((accepted_fd=accept(listener,(struct sockaddr *)&from,&from_size))>=0){
          if(IN6_IS_ADDR_LOOPBACK(&from.sin6_addr)){from_size=sizeof from;asking_accepted(accepted_fd);continue;}
          struct port *port=port_at(from.sin6_scope_id);from_size=sizeof from;
          if(!port || port->control>=0 || !port->open){close(accepted_fd);continue;}
          port->control=accepted_fd;
          port_greet(port);
        }
      }
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)hello)hello_heard();
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)network)network_heard();
#ifdef MESH_LOOP_FABRIC
      else if(e->udata!=UINT64_MAX && (e->udata&TAG_LOOP)){
        const uint32_t k=(uint32_t)(e->udata&0xff);
        int accepted_fd;
        while(k<loop_config.ports && (accepted_fd=accept(loop_config.port[k].fd,NULL,NULL))>=0){
          struct port *port=NULL;
          for(uint32_t p=0;p<port_count;p++)if(!strcmp(ports[p].device.name,loop_config.port[k].device))port=ports+p;
          if(!port || port->control>=0 || !port->open){close(accepted_fd);continue;}
          port->control=accepted_fd;port->peer=0;
          port_greet(port);
        }
      }
#endif
      else if(e->udata!=UINT64_MAX && (e->udata&(TAG_CHANNEL|TAG_ASKING|TAG_SERVING)))request_event(e);
      else if(e->udata<port_count && (int)e->ident==ports[e->udata].control){
        struct port *port=ports+e->udata;
        if(e->filter==EVFILT_WRITE && port->connecting){
          int error=0;socklen_t error_size=sizeof error;
          if(getsockopt(port->control,SOL_SOCKET,SO_ERROR,&error,&error_size))error=errno;
          if(error)port_lost(port,error,0);
          else {port->connecting=0;port_greet(port);}
        }
        else if(e->filter==EVFILT_WRITE){port->writing=0;port_flush(port);}
        else control_heard(port);
      }
    }
  }
  /* design/bridge-and-ipc.md#stop: `stop` read by every loop: the progress threads return at their next pass; then each
     peer told FAREWELL and each link lost (every program on it failed as a value, ESHUTDOWN, its queue pairs moved to
     the error state and destroyed), the region told the bridge gone, each device retired on its port's queue (behind a
     registration under way, which returns at `stop` between extents), the threads joined, the region unmapped */
  char why[96];farewell_text(stop_reason,why,sizeof why);
  say("bridge node %u: stopping (%s)\n",node,why);
  for(uint32_t p=0;p<port_count;p++){bell_ring(&ports[p].bell);for(int side=0;side<ports[p].threads;side++)pthread_join(ports[p].thread[side],NULL);ports[p].threads=0;}
  requests_stop();
  uint32_t pairs=0;
  for(uint32_t p=0;p<port_count;p++){
    pairs+=ports[p].made;
    port_say(ports+p,(struct message){.kind=FAREWELL,.code=stop_reason},NULL,0);
    port_lost(ports+p,ESHUTDOWN,1);
  }
  for(struct program *prog=programs,*next;prog;prog=next){next=prog->next;program_free(prog);}
  atomic_store(&M->bridge_pid,0);
  mesh_notify(&M->control);
  pthread_join(watcher,NULL);
  int retired=1;
  for(uint32_t p=0;p<port_count;p++){
    struct port *port=ports+p;
    dispatch_sync(port->work,^{ if(!down_device(&port->device))say("link %u (%s): verbs teardown: %s\n",port->index,port->name,strerror(errno)); });
    retired&=!port->device.context;
  }
  say("bridge node %u: stopped: %u queue pairs destroyed, %u ports' devices %s\n",node,pairs,port_count,retired?"closed":"not all closed");
  munmap(wire.data,wire.length);munmap(M,length);
  return 0;
}
