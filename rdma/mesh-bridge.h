/* design/transport.md: the bridge's own declarations, shared by its two translation units, mesh-link.c (the
   ports, links and lanes, the progress threads and the lanes' data path, the event loop and the stop) and mesh-records.c
   (the map's records, the node's health and the client slots): the types, the state both read (defined in mesh-link.c)
   and the functions each calls of the other's */
#ifndef MESH_BRIDGE_H
#define MESH_BRIDGE_H
#include "mesh-verbs.h"
#include "mesh-call.h"
#include "mesh-observe.h"
#include <dispatch/dispatch.h>
#include <SystemConfiguration/SystemConfiguration.h>
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

_Static_assert(sizeof(pthread_t)==8,"M17");

/* design/transport.md section 14: a link's queue pairs, made once at its one pairing, are its MESH_LINK_LANES lanes:
   lane MESH_LANE_KEEPALIVE the keepalive's, lane 1 reserved and unused, and from MESH_LANE_FIRST pairs, a pair's large
   lane (MESH_LANE_FIRST + 2 p) and its small lane (+ 2 p + 1).  The link's stream (mesh.h struct mesh_stream) is striped
   over them: its unit u of a class rides the lane of that class of pair u mod MESH_LANE_PAIRS, as that lane's
   (u / MESH_LANE_PAIRS)-th unit, alike at both ends (a lane's units land in order) */
#define MESH_LINK_LANES 10u
#define MESH_LANE_KEEPALIVE 0u
#define MESH_LANE_FIRST 2u
#define MESH_LANE_PAIRS ((MESH_LINK_LANES-MESH_LANE_FIRST)/2)
_Static_assert(MESH_LANE_FIRST+2*MESH_LANE_PAIRS==MESH_LINK_LANES,"the lanes' pairs");
_Static_assert(!(MESH_STREAM_LARGE_UNITS%MESH_LANE_PAIRS) && !(MESH_STREAM_SMALL_UNITS%MESH_LANE_PAIRS),"a stream's ring striped whole");
static inline uint32_t lane_class(uint32_t k){ return (k-MESH_LANE_FIRST)&1; }
static inline uint32_t lane_stripe(uint32_t k){ return (k-MESH_LANE_FIRST)/2; }

_Static_assert(sizeof(struct ibv_send_wr)==128,"M06");
_Static_assert(sizeof(struct ibv_sge)==16,"M07 M09");
_Static_assert(sizeof(struct ibv_recv_wr)==32,"M08");
_Static_assert(sizeof(struct ibv_wc)==48,"M11");
_Static_assert(offsetof(struct ibv_wc,status)==8 && offsetof(struct ibv_wc,opcode)==12 &&
  sizeof(((struct ibv_wc *)0)->status)==4 && sizeof(((struct ibv_wc *)0)->opcode)==4,"M11");
_Static_assert(sizeof(struct ibv_send_wr *)==8,"M15");
/* design/transport.md */
union mesh_network_event {
  struct kern_event_msg header;
  unsigned char bytes[KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data)];
};
_Static_assert(sizeof(union mesh_network_event)==48,"M24");
/* design/transport.md */
struct prepared_send {
  _Alignas(128) struct ibv_sge span;
  struct ibv_send_wr request;
};
_Static_assert(sizeof(struct prepared_send)==256 && _Alignof(struct prepared_send)==128 &&
  offsetof(struct prepared_send,span)==0 && offsetof(struct prepared_send,request)==16 &&
  offsetof(struct prepared_send,request.send_flags)+sizeof(unsigned int)<=64,"M29");
/* design/transport.md: a lane unit's receive, its request and SGE into the unit's slot of the stream's in ring */
struct prepared_receive {
  _Alignas(64) struct ibv_recv_wr request;
  struct ibv_sge span;
};
_Static_assert(sizeof(struct prepared_receive)==64,"a unit's receive");
/* A queue pair's SEND gate: the send thread's frames posted and retired (each request carries its queue
   and the frames posted through it, which its completion returns), the queue's capacity, and the census's
   requests and completions */
#define MESH_GATE_MASK ((UINT64_C(1)<<48)-1)
struct send_gate {
  _Alignas(64) uint64_t posted,retired,requests,completions;
  uint32_t capacity;
};
/* design/transport.md section 14: a data lane's progress, the bridge's alone, counted in the lane's own units (its i-th
   the stream's MESH_LANE_PAIRS i + its stripe): the units it has posted SENDs for (`sent`, the send thread's), and
   received (`posted` receives posted, `got` landed, the receive thread's, posted at most `window` units ahead: the
   lane's share of the completion queue in its frames), whether its receives were posted (`armed`, the receive
   thread's), and its requests, one a slot of its stripe each way (`sends`, `receives`, made as the link connects;
   the last of `sends` the dud's, from the bridge's all-ones block) */
struct lane_state {
  uint64_t sent,posted,got;
  uint32_t window,armed;
  struct prepared_send *sends;
  struct prepared_receive *receives;
};
/* design/transport.md: a message on a port's control socket (the bridge's wire: HELLO, FAREWELL, RECORD
   and LINK, nothing else), `bytes` of what follows it: a RECORD's record.  A FAREWELL's `code` is why its bridge stops;
   a port's LINK tells its link lanes' queue pairs (#lanes: every lane, each one's number and first sequence number in
   `pair`) */
enum { HELLO=1, FAREWELL, RECORD, LINK };
struct message { uint32_t magic,kind,node,count; uint64_t boot; int64_t code; uint32_t block,port,bytes;
  uint16_t lid,padding; uint8_t gid[16]; uint32_t pair[MESH_LINK_LANES][2]; };
/* the wire's magic: its base, MESH_VERSION and the wire's revision (1<<12, past any base and version's: every link heard
   within MESH_PLATFORM_NS or concluded, design/transport.md B1, B2, B3, which a bridge of the same version
   without them breaks), so bridges of different versions or revisions drop each other's hellos and never pair */
#define XMAGIC (0x4d595048u+MESH_VERSION+(1u<<12))
/* design/transport.md: the most bytes a port holds unsent toward a peer that takes none */
#define MESH_UNSENT (UINT64_C(64)<<20)
/* design/transport.md section 14: a port's link lanes (MESH_LINK_LANES), its queue pairs made once and connected once
   in the bridge's life (its one pairing: `connected` once its connect began; a link lost after it stays down until the
   bridge's next launch), on the two completion queues its device's context holds, each lane's share of them (`share`
   frames, a request one frame at least: no lane overruns its queues), and each lane's queue pair number (`qpn`, by which
   a completion names its lane).  Its state (LINK_DOWN; made and told, LINK_MADE; connecting; up), its job on the port's
   queue (`job`, dispatched at `job_at`: mesh-link.c ports_stalled) and a registration its make refused (`refused`), the
   pairing it belongs to (`pairing`, moved at its loss and at a halt, mesh-link.c port_halt, so a job of a pairing gone
   makes no call and is dropped as it ends, B10, B9b), the peer's lanes (`peer`, heard in its LINK: `heard`); the lanes
   the progress threads serve (`served`, a bit a lane, the event loop's to set and to clear, a thread's to clear where
   it found its lane failed, every lane at a failed completion, mesh-link.c link_failed, and the termination's queue's,
   B9b), and the lanes a progress thread found failed (`broken`: the event loop loses the link on reading it).  Each
   lane's SEND gate (`gate`: the frames posted and retired, the send thread's; its capacity, set as the lane connects),
   each data lane's progress (`lane`, struct lane_state), the arena page of the stream's rings (`rings`, leased as the
   port first pairs, MESH_ABSENT none: mesh.h struct mesh_stream), and the stream's holding, the send thread's alone: the
   owner word it serves (`held`, 0 none), the most each class's holder declared as it read (`seen`), and where a holding
   ended, the unit of each class before which every unit has gone, the unsupplied ones as duds (`final`); and the
   receive thread's first unit of each class not yet released in order (`released`, mesh-link.c stream_landed).  Lane
   MESH_LANE_KEEPALIVE's frames (design/transport.md: one whenever the link is up, every `keepalive_ns` while the
   stream is held, else every MESH_PLATFORM_NS/5; the send thread's frame in flight), and its page: the keepalive's
   block */
enum { LINK_DOWN, LINK_MADE, LINK_CONNECTING, LINK_UP };
enum { LINK_JOB_MAKE=1, LINK_JOB_CONNECT };
#define LINK_ALIVE_SENT (UINT64_C(3)<<56)
#define LINK_ALIVE_HEARD (UINT64_C(4)<<56)
#define LINK_ALIVE_BYTES 16u
#define LINK_ALIVE_RECEIVES 256u
struct link {
  struct mesh_verbs provider;
  int state,job,result,heard,refused,connected;
  _Atomic int done;
  _Atomic uint32_t served,broken,pairing;
  uint32_t peer[MESH_LINK_LANES][2],qpn[MESH_LINK_LANES],share,page,pages,rings,job_pairing;
  struct send_gate gate[MESH_LINK_LANES];
  struct lane_state lane[MESH_LINK_LANES];
  uint64_t alive_at,job_at,held,seen[2],final[2],released[2];
  _Atomic int alive_flight;
};
/* design/transport.md: a port, its link, its control socket's queues each way, its verbs work's queue (its
   open's job `job`, dispatched at `job_at`; `stuck` once a job of the port outlived MESH_PLATFORM_NS, or a progress
   thread did not hold within it, and the port was concluded around it, until the call returns and each progress thread
   has told the conclusion's epoch: design/transport.md B12, mesh-link.c ports_stalled), the end of its quiet window
   (`quiet_at`, MESH_PLATFORM_NS past its last loss or a peer interface's termination heard, B9b, moved only later and by
   any thread, before which no port job is dispatched on it and the bridge does not exit: B9), whether its device is to
   be opened (`opening`: at the start, at its link's coming on, at a hello heard while it is not open, B16, and as the
   call it was concluded around returns, B12; never on a clock), its link lanes, and its progress threads (#progress: the
   epochs each has seen of `epoch`, which a conclusion around a stuck call moves); the guard of its device's verbs
   objects (`guard`, odd while raised, the word the progress threads wait on while it stands and while the link serves
   nothing, design/transport.md B13, and the value each progress thread last held at: mesh-link.c guard_raise; each epoch
   and guard told wakes the raiser parked on it, port_tell, port_wait), and the threads that post on its queue pairs
   (`thread`: its two progress threads, each storing itself as it starts, and its link job's while one runs, by which a
   provider's fault names its port, mesh-link.c fault_heard) */
struct port {
  struct mesh_device device;
  uint32_t index,interface,peer,peer_port,greeted,parted,connecting,writing,open,opening,stuck;
  int64_t farewell;
  char name[IFNAMSIZ];
  int control,job,result,refused,threads;
  _Atomic int done,closing;
  uint64_t next,bandwidth,job_at;
  _Atomic uint64_t quiet_at;
  struct message heard;
  uint16_t lid;
  uint8_t gid[16];
  unsigned char *in,*out;
  size_t held,in_room,out_length,out_sent,out_room;
  dispatch_queue_t work;
  struct link link;
  _Atomic uint64_t epoch,seen[2];
  _Atomic(pthread_t) thread[3];
  _Atomic uint64_t guard,guarded[2];
};

/* design/transport.md section 9: why this bridge stops, the code its FAREWELL carries (a signal's number) */
/* design/transport.md: the most nanoseconds between two samples of this node's available memory, each sent
   with its record (mesh-program.c health_turn) */
#define MESH_HEALTH_NS UINT64_C(10000000000)

/* design/transport.md; design/transport.md: the nodes this bridge has heard of, when it last
   heard each one's record (this node's clock), when it lost each, and the newest record it ever heard of each (its boot
   and sequence), held apart from the map: RFC 2328 s13's database, so a record the map dropped is never taken, nor sent
   on, again (a node's later boot is always newer); a full table gives the longest unheard of those its place */
struct known { uint32_t node; uint64_t heard,gone,boot,sequence; };
#define MESH_KNOWN 256
/* design/transport.md section 14: a port's progress thread, `side` 0 its receives' and 1 its SENDs': it passes without
   pause while its link serves a lane (rubric M8: the queues read as fast as a hardware thread allows), and waits on the
   port's guard word while it serves none, looking again every MESH_PAUSE_NS (the stop's `closing`) */
struct progress { struct port *port; int side; };

/* design/transport.md: a loop port's cable listener's event, its index */
#define TAG_LOOP (UINT64_C(1)<<59)

/* the state both translation units read, defined in mesh-link.c */
extern struct hdr *M;
extern struct wire_mapping wire;
extern struct port ports[MESH_PORTS];
extern uint32_t port_count;
extern int events,listener,hello,network,unreleased,finding;
extern uint64_t now,sequence;
extern char host[MESH_NAME_BYTES];
extern struct mesh_ctx bridge;
extern _Atomic int found;
extern uint64_t memory;
extern struct known known[MESH_KNOWN];
extern uint64_t keepalive_ns;
extern uint64_t owners[MESH_CLIENTS];
extern uint32_t ones;
/* the window's pages the bridge holds itself (the map, the links' pages and rings, the duds' block): its record's
   `window` is the rest, what its clients may hold */
extern uint64_t held_pages;
extern char model[MESH_MODEL_BYTES];

/* whether a client's slot is still `owner`'s */
static inline int owned(uint64_t owner){ return owner && atomic_load(&mesh_client_at(M,owner)->owner)==owner; }

/* the functions each translation unit calls of the other's */
void sysctl_text(const char *name,char *out,size_t room);
void health_init(void);
void wake(void);
void port_say(struct port *port,struct message said,const void *payload,uint32_t bytes);
void streams_cancel(const struct port *port);
void streams_let_go(void);
void record_changed(void);
void record_retell(void);
struct mesh_record *record_held(uint32_t node);
void health_turn(void);
void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes);
void port_lost(struct port *port,int64_t code);
void clients_turn(int liveness);

#endif
