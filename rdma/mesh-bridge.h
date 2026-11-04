/* The bridge's own declarations, process-local: the types, the state and the functions its translation units share
   (mesh-link.c, mesh-children.c, mesh-child.c, mesh-ports.c, mesh-events.c, mesh-lanes.c, mesh-records.c and mesh-flow.c:
   the resident bridge `mesh-flow` and its per-port children, the same image).  From tag pre-return-20261103
   rdma/mesh-bridge.h, adapted to main (design/page-queue-change-request.md, "Claude's proposal").  Queue pairs and links
   belong to the bridge: no client, collective or data-processing code includes this header, and nothing declared here
   is passed to one (the change request, "Queue pairs and links belong to the bridge").  A port is one cable; its child
   holds that cable's queue pairs, as many as the pairing's budget gives (link_plan): the tag's lanes are main's queue
   pairs (mesh-verbs.h struct mesh_queue, struct mesh_verbs, struct mesh_device), and "lane" below names one.  The tag's
   streams, holdings and occurrence-table walks do not come over (struct walk, mesh_zwoggle, lanes_through, the hold
   states, the stream layout): the data path is mesh-flow.c's, over the region's page lists (mesh.h).  Region types are
   named here only by pointer, declared forward. */
#ifndef MESH_BRIDGE_H
#define MESH_BRIDGE_H
/* tag etc/bridge/thunderbolt.mk MESH_PROVIDER_RESERVED_QPS, 1 (the tag Makefile's -D, as the Makefile's here; under
   #ifndef so that identical -D stands): the queue pairs of each device the provider holds itself, so a device holds
   max_qp less this for a link (mesh-verbs.h device_up `max_pairs`; 76be3217, the rule of bd85613e
   design/HETEROGENEOUS-LINKS.md).  Defined before mesh-verbs.h, whose device_up reads it */
#ifndef MESH_PROVIDER_RESERVED_QPS
#define MESH_PROVIDER_RESERVED_QPS 1
#endif
#include "mesh-verbs.h"
/* tag rdma/mesh-bridge.h: its system headers.  Its mesh-call.h, mesh-observe.h and mesh-graph.h are left out: the client
   context is named by its tag alone (`bridge`) and the client slots' `owned` does not come over, the observe header's
   mesh_say is below (main's mesh.h holds struct mesh_link_info), and the walks' chains are named by nothing here */
#include <dispatch/dispatch.h>
#include <CoreFoundation/CoreFoundation.h>
#include <sys/event.h>
#include <sys/ioctl.h>
#include <sys/kern_event.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <signal.h>
#include <limits.h>
#include <net/if.h>
#include <net/if_var.h>
#include <net/net_kev.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdarg.h>
#include <time.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <IOKit/IOKitLib.h>
#include <poll.h>
#include <mach-o/dyld.h>
#include <mach-o/getsect.h>

/* tag etc/bridge/layout.h: the bridge's bounds.  A flush ends once no completion has come for MESH_FLUSH_QUIET_NS
   (lanes_flush); MESH_PAUSE_NS paces the bridge's looks (a hello, the child's turn); no call, message or window of the
   bridge is held past MESH_PLATFORM_NS unconcluded (message_send, child_leave, port_lost's quiet window) */
#define MESH_FLUSH_QUIET_NS UINT64_C(500000000)
#define MESH_PAUSE_NS UINT64_C(3000000000)
#define MESH_PLATFORM_NS UINT64_C(5000000000)
/* tag etc/bridge/layout.h: port_opened's hold after its device's k-th refused registration in a row, MESH_RETRY_FIRST_SECONDS
   2^min(k-1, MESH_RETRY_MAX_SHIFT) s, at most MESH_RETRY_MAX_SECONDS */
#define MESH_RETRY_FIRST_SECONDS 60u
#define MESH_RETRY_MAX_SECONDS 3600u
#define MESH_RETRY_MAX_SHIFT 6u
/* tag etc/bridge/layout.h: mesh_say's line, and its log's bound */
#define MESH_DIAGNOSTIC_BYTES 640
#define MESH_LOG_BYTES 4194304
/* tag etc/bridge/layout.h: the event loop's tag on a port's child's exit (kevent64 udata TAG_CHILD | the port's index,
   mesh-children.c child_spawn) */
#define TAG_CHILD (UINT64_C(1)<<58)
/* tag etc/bridge/layout.h: the most nodes the bridge's table of nodes heard holds (struct known, mesh-records.c) */
#define MESH_KNOWN 256
/* tag etc/bridge/layout.h: a link's streams at the tag (mesh-observe.h MESH_PORTS = MESH_STREAMS / MESH_LINK_STREAMS) */
#define MESH_LINK_STREAMS 4u
/* tag etc/bridge/thunderbolt.h MESH_FRAME: the bytes of a frame, the unit of a queue's capacity (struct frames) */
#define MESH_FRAME 4096u
/* tag etc/bridge/layout.h MESH_RECORD_ALIGNMENT: a SEND gate's alignment (struct send_gate) */
#define MESH_RECORD_ALIGNMENT 64
/* tag etc/bridge/layout.h: the most requests outstanding on a lane each way (struct lane_state) */
#define MESH_LANE_UNITS 4096u
/* tag etc/bridge/layout.h, "Completion service cadence, independent of queue capacity and link use": the most a lane
   posts in a pass before the pass reaps completions, and the completions one poll takes (lanes_flush, the data path) */
#define MESH_POST_QUANTUM 64u
#define MESH_COMPLETION_BATCH 64
/* tag etc/bridge/layout.h: the bytes of this node's model as its record carries it (`model`, sysctl hw.model) */
#define MESH_MODEL_BYTES 32
/* tag etc/bridge/layout.h XMAGIC, its value, renamed: the wire's magic, what marks a datagram or a control stream's
   header as the mesh's (struct message `magic`), the value version 131 sends, fixed.  Named MESH_WIRE_MAGIC because
   main's --link form keeps its own XMAGIC (mesh-verbs.h struct qpi) until sub-assignment 16 removes it; what must stay
   is the value */
#define MESH_WIRE_MAGIC (0x4d595048u+131u+(1u<<12))

/* tag rdma/mesh-observe.h mesh_say, verbatim */
/* design/transport.md section 14: a line of the node's bridge log, the launchd job's stderr (bin/mesh-user.sh
   StandardErrorPath): the UTC time, `who` and the message, in one write; the losses and failures of the bridge and its
   heartbeat, nothing on the data path's success.  A log that has reached MESH_LOG_BYTES is emptied first (the processes
   that share it append: O_APPEND), so it holds at most that and a line */
__attribute__((format(printf,2,3))) static inline void mesh_say(const char *who,const char *format,...){
  char line[MESH_DIAGNOSTIC_BYTES];
  struct timespec t;clock_gettime(CLOCK_REALTIME,&t);
  struct tm utc;gmtime_r(&t.tv_sec,&utc);
  int n=snprintf(line,sizeof line,"%04d-%02d-%02dT%02d:%02d:%02d.%03ldZ %s: ",utc.tm_year+1900,utc.tm_mon+1,utc.tm_mday,
                 utc.tm_hour,utc.tm_min,utc.tm_sec,t.tv_nsec/1000000,who);
  va_list arguments;va_start(arguments,format);
  if(n>0 && (size_t)n<sizeof line)n+=vsnprintf(line+n,sizeof line-(size_t)n,format,arguments);
  va_end(arguments);
  if(n<=0)return;
  if((size_t)n>=sizeof line-1)n=(int)sizeof line-2;
  line[n++]='\n';
  struct stat log;
  const int flags=fcntl(2,F_GETFL);
  if(!fstat(2,&log) && S_ISREG(log.st_mode) && flags>=0){
    if(!(flags&O_APPEND))fcntl(2,F_SETFL,flags|O_APPEND);
    if(log.st_size>=MESH_LOG_BYTES)(void)!ftruncate(2,0);
  }
  (void)!write(2,line,(size_t)n);
}
/* tag rdma/mesh-link.c:48 clock_ns, verbatim but `static inline`, as a header's: the monotonic clock, ns */
static inline uint64_t clock_ns(void){ return clock_gettime_nsec_np(CLOCK_MONOTONIC); }

/* tag rdma/mesh-bridge.h, the verbs ABI the bridge's requests are laid against; the tag named the values
   MESH_SIZEOF_IBV_* (etc/bridge/thunderbolt.h), here the literals main's mesh-flow.c states */
_Static_assert(sizeof(pthread_t)==8,"M17");
_Static_assert(sizeof(struct ibv_send_wr)==128,"M06");
_Static_assert(sizeof(struct ibv_sge)==16,"M07 M09");
_Static_assert(sizeof(struct ibv_recv_wr)==32,"M08");
_Static_assert(sizeof(struct ibv_wc)==48,"M11");
_Static_assert(offsetof(struct ibv_wc,status)==8 && offsetof(struct ibv_wc,opcode)==12 &&
  sizeof(((struct ibv_wc *)0)->status)==4 && sizeof(((struct ibv_wc *)0)->opcode)==4,"M11");
_Static_assert(sizeof(struct ibv_send_wr *)==8,"M15");

/* tag rdma/mesh-bridge.h MESH_LINK_LANES, its value (MESH_LANE_FIRST + MESH_LINK_STREAMS MESH_STREAM_LANES: 2 + 4 2,
   etc/bridge/layout.h): the most queue pairs a link makes, every one a macOS 26 device holds for it (max_qp 11 less
   MESH_PROVIDER_RESERVED_QPS; 76be3217, "every lane at ten").  The stream layout it was summed from does not come over */
#define MESH_LINK_LANES 10u
_Static_assert(MESH_LINK_LANES<32,"a bit a lane");
/* tag rdma/mesh-bridge.h link_plan (76be3217), adapted: the queue pairs a link makes, a bit a queue pair, from `budget`,
   the fewer of the queue pairs its two ends' devices hold for it (hello_budget; mesh-verbs.h device_up `max_pairs`):
   queue pairs 0 to budget - 1, every one where both hold MESH_LINK_LANES.  The plan of fewer queue pairs is a subset of
   the plan of more, as at the tag.  Differs from the tag: there lanes 0 and 1 carried nothing (lane 0 made from a budget
   of three) and the rest went by streams' lane pairs, so a budget of one made none and three made two that carry; with
   the streams gone, every queue pair of a plan carries pages by the rule both ends compute (the change request, "The rule
   both ends compute"), so a plan is the budget's first queue pairs, queue pair q lane q (mesh-verbs.h pair_up) */
static inline uint32_t link_plan(uint32_t budget){
  if(budget>=MESH_LINK_LANES)return (UINT32_C(1)<<MESH_LINK_LANES)-1;
  return (UINT32_C(1)<<budget)-1;
}
/* tag rdma/mesh-bridge.h plan_lanes, verbatim: a plan's queue pairs */
static inline uint32_t plan_lanes(uint32_t plan){ return (uint32_t)__builtin_popcount(plan); }

/* tag rdma/mesh-bridge.h union mesh_network_event, verbatim (main's mesh-flow.c holds the same union) */
/* design/transport.md */
union mesh_network_event {
  struct kern_event_msg header;
  unsigned char bytes[KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data)];
};
_Static_assert(sizeof(union mesh_network_event)==48,"M24");

/* tag rdma/mesh-bridge.h struct frames, frames_of, frames_min, verbatim */
/* TN3205 (README.md): a SEND and the receive it fills cover the same number of frames of MESH_FRAME bytes, a SEND held by
   the hardware's credit until its receive is posted; the bridge's alone */
/* a count of frames: a queue's capacity, a lane's share of its completion queues, a SEND's and the send gate's, a type
   of its own so no count of pages, units or bytes passes for one; a SEND of `bytes` covers frames_of(bytes) */
struct frames { uint64_t n; };
static inline struct frames frames_of(uint64_t bytes){ return (struct frames){(bytes+MESH_FRAME-1)/MESH_FRAME}; }
static inline struct frames frames_min(struct frames a,struct frames b){ return a.n<b.n?a:b; }
/* tag rdma/mesh-bridge.h struct send_gate, gate_holds, gate_post, gate_retire, verbatim */
/* A queue pair's SEND gate: actual frames posted and retired, its capacity,
   and requests/completions. A completion names its retained span; its byte
   length determines the frames retired, independently of logical mask cuts. */
struct send_gate {
  _Alignas(MESH_RECORD_ALIGNMENT) struct frames posted,retired;
  uint64_t requests,completions;
  struct frames capacity;
};
/* whether the gate holds `f` frames more than it has posted and not retired */
static inline int gate_holds(const struct send_gate *gate,struct frames f){
  return gate->posted.n-gate->retired.n+f.n<=gate->capacity.n;
}
static inline void gate_post(struct send_gate *gate,struct frames f){ gate->posted.n+=f.n; }
static inline void gate_retire(struct send_gate *gate,struct frames f){ gate->retired.n+=f.n;gate->completions++; }
/* tag rdma/mesh-bridge.h struct lane_state, verbatim; its comment's last sentence, on the walks' work-request ids, left
   out with the walks.  lanes_flush reads it at the child's end */
/* A lane counts requests and frame credits separately (the queue's depths): a unit, one page, consumes
   ceil(page/MESH_FRAME) frames; MESH_LANE_UNITS limits outstanding requests independently. */
struct lane_state { uint64_t sent,done,posted,got,frames_posted,frames_retired,frame_capacity; };

/* tag rdma/mesh-bridge.h, the messages on a port's control socket and struct message, verbatim */
/* design/transport.md: a message on a port's control socket (the bridge's wire: HELLO, FAREWELL, RECORD
   and LINK, nothing else), `bytes` of what follows it: a RECORD's record.  A FAREWELL's `code` is why its bridge stops;
   a port's LINK tells its link lanes' queue pairs (#lanes: `count` the lanes of its plan, each one's number and first
   sequence number in `pair`, 0 for a lane the plan leaves out).  A HELLO on a control socket states in `count` 1 + the
   queue pairs its port's device holds for a link (design/HETEROGENEOUS-LINKS.md: hello_budget; a bridge before budgets
   sends 0 there and makes MESH_LINK_LANES, so two ends that both hold that many pair as before budgets) */
enum { HELLO=1, FAREWELL, RECORD, LINK };
struct message { uint32_t magic,kind,node,count; uint64_t boot; int64_t code; uint32_t block,port,bytes;
  uint16_t lid,version; uint8_t gid[16]; uint32_t pair[MESH_LINK_LANES][2]; };
static inline uint32_t hello_budget(uint32_t count){ return count?count-1:MESH_LINK_LANES; }

/* tag rdma/mesh-bridge.h struct link and its states and jobs, verbatim but for the fields left out: each stream's
   holding (`held`, `hold`, `sending`, `landing`, `wire`: the walks and the streams' identities) */
/* design/transport.md section 14: a port's link lanes (MESH_LINK_LANES), its queue pairs made once and connected once
   in its child's life ("A process a port": `lanes` once the child was asked for them, the parent's; a link lost after it
   pairs again with the port's next child), on the two completion queues its device's context holds, each lane's share
   of them (`share` frames, a request one frame at least: no lane overruns its queues), the link block (`page`, its first
   arena page, and `pages`, its arena pages: mesh-link.c link_pages), and each lane's queue pair number (`qpn`, by which
   a completion names its lane).  Its state (LINK_DOWN; made and told, LINK_MADE; connecting; up), its
   job on the port's queue (`job`, dispatched at `job_at`: mesh-link.c ports_stalled), the pairing it belongs to
   (`pairing`, moved at its loss and at a halt, mesh-link.c port_halt, so a job of a pairing gone makes no call and is
   dropped as it ends, B10, B9b), the peer's lanes (`peer`, heard in its LINK: `heard`); the lanes the progress thread
   serves (`served`, a bit a lane, the event loop's to set and to clear, a thread's to clear where it found its lane
   failed, every lane at a failed completion, mesh-link.c link_failed, and the termination's queue's, B9b), and the
   lanes a progress thread found failed, or every lane of a link none of whose streams is live (`broken`: the event loop
   loses the link on reading it), and the code the loss is booked with where it is not a failed lane's EIO (`lost`:
   LINK_STREAMS_ENDED, no stream live, the child's to set and to tell, the parent's to book, design/rubrics.md M10).  The
   lanes its pairing makes (`plan`, a bit a lane: link_plan of the fewer queue pairs its two ends hold, set as the
   parent hears the peer's HELLO and told the child with the make; its lanes left out have no queue pair and their
   `qpn` UINT32_MAX, no queue pair's number).  Each lane's SEND gate (`gate`: the frames posted and retired; its
   capacity, set as the lane connects), each data lane's progress (`lane`, struct lane_state) */
enum { LINK_DOWN, LINK_MADE, LINK_CONNECTING, LINK_UP };
enum { LINK_JOB_MAKE=1, LINK_JOB_CONNECT };
struct link {
  struct mesh_verbs provider;
  int state,job,result,heard,lanes;
  _Atomic int done,lost;
  _Atomic uint32_t served,broken,pairing;
  uint32_t plan,peer[MESH_LINK_LANES][2],qpn[MESH_LINK_LANES],page,pages,job_pairing,psn;
  struct frames share;
  struct send_gate gate[MESH_LINK_LANES];
  struct lane_state lane[MESH_LINK_LANES];
  uint64_t job_at;
};

/* tag etc/bridge/layout.h MESH_CHILD_VERSION, verbatim */
/* design/transport.md section 14, "A process a port": the version of the contract between the bridge and its children,
   declared: their command line, `mesh-flow --child VERSION DEVICE INDEX ONES` (its first three words the same
   for every version, so a bridge finds a child of any version that holds a device: child_elsewhere), and struct
   child_message.  It changes only where that contract does; a build of the same contract is the same child, whatever
   its bytes (the operator, 2026-10-21) */
/* 3: the child's command line gained its region's name as a sixth word (mesh-children.c child_spawn, mesh-child.c
   child_main), a change of the contract this version names, by the tag's own rule above */
#define MESH_CHILD_VERSION 3u
/* tag rdma/mesh-bridge.h LINK_STREAMS_ENDED, verbatim; what a stream's end is under the page lists is mesh-flow.c's
   (streams_open, streams_cancel: the cancellation of this node's words) */
/* the code a link's loss is booked with where no stream of it is live, each holding on it having ended with units
   outstanding (mesh-link.c stream_end): the ends of its clients' holdings, not a failure of its lanes (EIO) or its
   hardware (design/rubrics.md M10: a fact named as what it is), told the parent in CHILD_BROKEN's `result` (version
   2 of the children's contract) */
#define LINK_STREAMS_ENDED ECONNABORTED
/* tag rdma/mesh-bridge.h, the children's messages and struct child_message, verbatim */
/* design/transport.md section 14, "A process a port": the messages between the bridge's parent and a port's child on
   their socket pair (a stream, AF_UNIX having no SOCK_SEQPACKET here: each a whole struct child_message).  The
   parent's: open the device (CHILD_OPEN), make the link's lanes (CHILD_MAKE, naming the link's pages, its spare page
   among them, which the parent may lease after the child's spawn), connect them to the peer's (CHILD_CONNECT: the peer's
   lanes, address, node and the link's bandwidth), serve no lane again (CHILD_HALT), end (CHILD_END).  The
   child's, each naming the pairing it was asked for: the device opened or not (CHILD_OPENED: its address and bandwidth,
   and in `lanes` the queue pairs the device holds for a link), the lanes made or not (CHILD_MAKE names the lanes of the
   plan in `lanes`; CHILD_MADE: each lane's queue pair number, the first sequence number and each lane's share),
   connected or not (CHILD_CONNECTED), and the lanes a progress thread found failed, or every lane of a link none of
   whose streams is live (CHILD_BROKEN, naming its connect's pairing, `result` the loss's code where it is not a failed
   lane: LINK_STREAMS_ENDED), a fault absorbed on the provider's doorbell (CHILD_FAULTED), its port's peer interface
   terminating (CHILD_TERMINATED, as the child heard it: B9b); 11 was a guard not held, which now ends the child
   (mesh-link.c port_wait), so a parent and child of builds either side of it read every kind alike */
enum { CHILD_OPEN=1, CHILD_MAKE, CHILD_CONNECT, CHILD_HALT, CHILD_END,
       CHILD_OPENED, CHILD_MADE, CHILD_CONNECTED, CHILD_BROKEN, CHILD_FAULTED, CHILD_TERMINATED=12 };
struct child_message {
  uint32_t kind,pairing;
  int32_t result,refused;
  uint32_t qpn[MESH_LINK_LANES],peer[MESH_LINK_LANES][2],psn,share,broken,node,page,lanes;
  uint16_t lid,padding;
  uint8_t gid[16];
  uint64_t bandwidth;
};

/* tag rdma/mesh-bridge.h struct port, verbatim */
/* design/transport.md: a port, its link, its control socket's queues each way, its verbs work's queue (its
   open's job `job`, dispatched at `job_at`; `stuck` once a job of the port outlived MESH_PLATFORM_NS, or a progress
   thread did not hold within it, and the port was concluded around it, until the call returns and each progress thread
   has told the conclusion's epoch: design/transport.md B12, mesh-link.c ports_stalled), the end of its quiet window
   (`quiet_at`, MESH_PLATFORM_NS past its last loss or a peer interface's termination heard, B9b, moved only later and by
   any thread, before which no port job is dispatched on it and the bridge does not exit: B9), whether its device is to
   be opened (`opening`: at the start, at its link's coming on, at a hello heard while it is not open, B16, and as the
   call it was concluded around returns, B12; never on a clock), its link lanes, and its progress thread (`threads` 1
   once started: the epoch it has seen of `epoch`, which a conclusion around a stuck call moves); the guard of its
   device's verbs objects (`guard`, odd while raised, the word the progress thread waits on while it stands and while
   the link serves nothing, design/transport.md B13, and the value the progress thread last held at: mesh-link.c
   guard_raise; each epoch and guard told wakes the raiser parked on it, port_tell, port_wait), and the threads that
   post on its queue pairs (`thread`: its progress thread, storing itself as it starts, and its link job's while one
   runs, by which a provider's fault names its port, mesh-link.c fault_heard); the queue pairs its device holds for a link as its child
   told them at its opening (`budget`, the parent's: design/HETEROGENEOUS-LINKS.md); the registrations its device refused
   in a row (`refusals`) and the time before which it is not opened again after one (`retry_at`: mesh-link.c port_opened);
   and its interface as the kernel's events name it (`family`, `unit`: an en<N>'s APPLE_IF_FAM_ETHERNET and N, read once
   from its name as the port is named; another port's family 0, which no event carries: mesh-link.c addresses_heard) */
struct port {
  struct mesh_device device;
  uint32_t index,interface,peer,peer_port,greeted,parted,connecting,writing,open,opening,stuck,budget,refusals,family,unit;
  int64_t farewell;
  char name[IFNAMSIZ];
  int control,job,result,refused,threads,unheard;
  _Atomic int done,closing;
  uint64_t next,bandwidth,job_at,retry_at;
  _Atomic uint64_t quiet_at;
  struct message heard;
  uint16_t lid;
  uint8_t gid[16];
  unsigned char *in,*out;
  size_t held,in_room,out_length,out_sent,out_room;
  dispatch_queue_t work;
  struct link link;
  _Atomic uint64_t epoch,seen;
  _Atomic(pthread_t) thread[2];
  _Atomic uint64_t guard,guarded;
  /* the parent's: the port's child (design/transport.md section 14, "A process a port"): its process (0 none) and
     socket, the hold a spawn that failed sets before the next (`hold_to`), whether the port's link, lost once its lanes
     were asked for, waits to end its child (`recycle`) and whether that end was asked (`ending`), the part of its
     child's next message read (`child_in`, `child_held` bytes of it), and a child of its device from an earlier launch
     last found alive (`elsewhere`: mesh-link.c child_elsewhere) */
  pid_t child,elsewhere;
  uint32_t holder;
  int child_fd,recycle,ending;
  uint32_t child_held;
  uint64_t hold_to;
  unsigned char child_in[sizeof(struct child_message)];
};

/* tag rdma/mesh-bridge.h struct known, verbatim */
/* design/transport.md; design/transport.md: the nodes this bridge has heard of, when it last
   heard each one's record (this node's clock), when it lost each, and the newest record it ever heard of each (its boot
   and sequence), held apart from the map: RFC 2328 s13's database, so a record the map dropped is never taken, nor sent
   on, again (a node's later boot is always newer); a full table gives the longest unheard of those its place */
struct known { uint32_t node; uint64_t heard,gone,boot,sequence; };

/* the state the bridge's translation units share, defined in mesh-link.c: from tag rdma/mesh-bridge.h's externs and
   tag rdma/mesh-link.c's definitions (there `stop_reason`, `exit_at`, `faulted`, `terminated` and `self_path` were
   static; `stop` is tag rdma/mesh-verbs.h's extern, the stop flag defined once in mesh-link.c), each with the tag's type
   but `wire` (main's struct mesh_wire for the tag's struct wire_mapping) and `ports` (its bound, mesh.h MESH_PORTS, given
   at its definition) */
extern struct hdr *M;
extern struct mesh_wire wire;
extern struct port ports[];
extern uint32_t port_count;
extern int events,listener,hello,network,addresses;
extern uint64_t now;
extern _Atomic sig_atomic_t stop;
extern int64_t stop_reason;
extern _Atomic uint64_t exit_at;
extern _Atomic uint32_t faulted,terminated;
extern char self_path[PATH_MAX];
extern uint32_t ones;
/* the rest of tag rdma/mesh-bridge.h's externs, defined in mesh-link.c with the tag's types: the arena pages the bridge
   holds itself (`held_pages`), this node's available memory and model as its record carries them, the bridge's own
   context (`bridge`, mesh-call.h struct mesh_ctx, named here by its tag alone), the clients' slots' owners and the
   ones not released (`owners`: its bound, the tag's MESH_CLIENTS, given at its definition; `unreleased`), the record
   sequence and the nodes heard (`known`) */
struct mesh_ctx;
extern uint64_t held_pages,memory,sequence;
extern char model[MESH_MODEL_BYTES];
extern struct mesh_ctx bridge;
extern uint64_t owners[];
extern int unreleased;
extern struct known known[MESH_KNOWN];
/* defined in mesh-ports.c, with the tag's types (tag rdma/mesh-link.c; there `find_wanted`, `finder` and `everyone`
   were static): a read of the devices out (`finding`) and taken (`found`), one wanted (`find_wanted`), the queue it
   runs on (`finder`), and the hellos' group, ff02::1 (`everyone`) */
extern int finding,find_wanted;
extern _Atomic int found;
extern dispatch_queue_t finder;
extern const struct in6_addr everyone;

/* tag rdma/mesh-link.c quiet_until, verbatim but `static inline`, as a header's */
/* a window's end moved to `until` where it is earlier, by any thread (the event loop's conclusions, a termination's
   queue, B9b), never earlier */
static inline void quiet_until(_Atomic uint64_t *at,uint64_t until){
  uint64_t was=atomic_load(at);
  while(was<until && !atomic_compare_exchange_weak(at,&was,until)){}
}
/* tag rdma/mesh-link.c page_bytes, verbatim; its comment's mention of the occurrence tables left out with them */
/* The page table's quantum: the bridge's storage geometry and every lane unit's bytes. */
static inline uint64_t page_bytes(void){ return (uint64_t)M->block*M->pgsz; }

/* the region's types the prototypes below name, by pointer alone (mesh.h) */
struct mesh_record;
struct mesh_reach;

/* the functions each translation unit calls of another's, with the tag's signatures (tag rdma/mesh-link.c and
   rdma/mesh-records.c; there most were static), by the file that defines each */

/* mesh-link.c: not in the coordinator's table; tag rdma/mesh-bridge.h's prototype, the tag's mesh-link.c its definition
   (the event loop's EVFILT_USER, triggered by a port's job, the finder and a termination's queue) */
void wake(void);

/* mesh-children.c */
void message_send(int fd,const struct child_message *said);
void child_send(struct port *port,struct child_message said);
int child_spawn(struct port *port);
void child_job(struct port *port,int job);
void child_heard(struct port *port);
void child_turn(struct port *port);
void child_reaped(struct port *port);
/* the tag's main after the windows' hold (no function at the tag): CHILD_END to every child, their exits awaited on a
   queue of their own, SIGKILL past the bound, one the kernel still holds named */
void children_end(void);

/* mesh-child.c */
int child_main(int argc,char **argv);

/* mesh-ports.c */
void ports_find(void);
void ports_found(void);
struct port *port_at(uint32_t interface);
void port_open(struct port *port);
void port_opened(struct port *port);
void port_hello(struct port *port);
void hello_heard(void);
void port_greet(struct port *port);
void control_heard(struct port *port);
void port_flush(struct port *port);
void port_say(struct port *port,struct message said,const void *payload,uint32_t bytes);
void port_lost(struct port *port,int64_t code);
void link_turn(struct port *port);
uint32_t record_reach(struct mesh_reach *reach,uint32_t room);

/* mesh-events.c */
void network_heard(int stopping);
void addresses_heard(void);
void terminations_watch(int arrivals);
void terminations_turn(void);
void arrivals_turn(void);
void faults_install(void);

/* mesh-lanes.c */
void device_open(struct port *port);
void link_job(struct port *port,int job);

/* mesh-records.c */
void map_adopt(void);
void record_changed(void);
void record_retell(void);
void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes);
struct mesh_record *record_held(uint32_t node);
void sysctl_text(const char *name,char *out,size_t room);
void health_init(void);
uint64_t health_due(void);
void health_turn(void);
void clients_turn(int liveness);

/* mesh-flow.c */
int guard_raise(struct port *port);
void guard_lower(struct port *port,int raised);
void guard_wake(struct port *port);
int port_threads(struct port *port);
void streams_open(struct port *port,uint32_t node,uint64_t bandwidth);
void streams_cancel(const struct port *port);
void streams_let_go(void);
/* tag rdma/mesh-link.c:56-65 port_halt, static there; mesh-events.c (peer_terminated) and mesh-child.c call it */
void port_halt(struct port *port,_Atomic uint32_t *which);

/* mesh-events.c: tag rdma/mesh-link.c windows_end (static there), which mesh-link.c's loop, turn_due and stop read */
uint64_t windows_end(void);
/* mesh-ports.c: tag rdma/mesh-link.c:780-802 ports_stalled and :23-25 `stalled` (static there) */
void ports_stalled(void);
extern _Atomic int stalled;
/* mesh-ports.c: a port with a peer.  The tag tested `port->control>=0`; the control socket now closes after the pairing's
   one exchange (the coordinator's instruction), so a paired port is one whose socket is open or whose peer's hello was
   heard (`greeted`, cleared by port_lost).  Every "has a peer" test of the tag's reads this one */
static inline int port_paired(const struct port *port){ return port->control>=0 || port->greeted; }
/* tag rdma/mesh-link.c:268-274 link_lane (c24265e0), verbatim: the lane a completion's queue pair number names: one of
   the link's lanes, else MESH_LINK_LANES (dropped; a lane its plan leaves out has the number UINT32_MAX, which no queue
   pair has) */
static __attribute__((always_inline)) inline uint32_t link_lane(const struct link *l,uint32_t qp_num){
  uint32_t k=0;
  while(k<MESH_LINK_LANES && l->qpn[k]!=qp_num)k++;
  return k;
}
/* tag etc/bridge/layout.h, verbatim: the kernel event socket's read (mesh-events.c), a control stream's largest message
   (mesh-ports.c), the configuration's and the region name's bytes and a request's (mesh-link.c), a health sample's pace
   (mesh-records.c) */
#ifndef MESH_KERNEL_EVENT_BYTES
#define MESH_KERNEL_EVENT_BYTES 512
#endif
#ifndef MESH_TOPOLOGY_MESSAGE_BYTES
#define MESH_TOPOLOGY_MESSAGE_BYTES (UINT32_C(64)<<20)
#endif
#ifndef MESH_CONFIGURATION_BYTES
#define MESH_CONFIGURATION_BYTES 65536
#endif
#ifndef MESH_REGION_NAME_BYTES
#define MESH_REGION_NAME_BYTES 256
#endif
#ifndef MESH_REQUEST_BYTES
#define MESH_REQUEST_BYTES 16773120u
#endif

#endif
