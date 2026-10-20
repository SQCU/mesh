/* design/transport.md: the bridge's own declarations, shared by its two translation units, mesh-link.c (the
   ports, links and lanes, the progress threads and the lanes' data path, the event loop and the stop) and mesh-records.c
   (the map's records, the node's health and the client slots): the types, the state both read (defined in mesh-link.c)
   and the functions each calls of the other's */
#ifndef MESH_BRIDGE_H
#define MESH_BRIDGE_H
#include "mesh-verbs.h"
#include "mesh-call.h"
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
/* design/transport.md: a lane unit's receive, its request and SGE into the unit's slot of the in ring */
struct prepared_receive {
  _Alignas(64) struct ibv_recv_wr request;
  struct ibv_sge span;
};
_Static_assert(sizeof(struct prepared_receive)==64,"a unit's receive");
/* A queue pair's SEND gate: the send thread's frames posted and retired (each request carries its queue
   and the frames posted through it, which its completion returns), the queue's capacity, and the census's
   requests and completions; and the event loop's alone, `retired` as its turn last read it and when it last
   found the gate empty or `retired` moved (`seen`, `held_at`: design/transport.md, mesh-link.c
   gates_turn). */
#define MESH_GATE_MASK ((UINT64_C(1)<<48)-1)
struct send_gate {
  _Alignas(64) uint64_t posted,retired,requests,completions;
  uint32_t capacity;
  uint64_t seen,held_at;
};
/* design/transport.md section 14: a data lane's progress, the bridge's alone: the units it has posted SENDs for
   (`sent`, the send thread's), and received (`posted` receives posted, `got` landed, the receive thread's, posted at
   most `window` units ahead: the lane's share of the completion queue in its frames), whether its receives were posted
   this pairing (`armed`, the receive thread's), and its requests, a unit's each way (`sends`, `receives`, made as the
   pairing connects, naming the pairing's queue pair; sends[units] the dud's, from the bridge's all-ones block).  Its
   pair's binding: the bound client's owner word (`owner`, 0 none: the event loop's), the declaration the event loop read
   as the binding ended (`final`), the most the send thread read while it stood (`seen`), and the duds it sent (`duds`) */
struct lane_state {
  uint64_t sent,posted,got;
  uint32_t window,armed;
  struct prepared_send *sends;
  struct prepared_receive *receives;
  _Atomic uint64_t owner,final,seen,duds;
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
/* design/transport.md: a progress thread with nothing moved for MESH_IDLE_NS parks on its port's bell
   (mesh.h struct hdr `bell`, which a client rings as it releases a lane's units, mesh-call.h mesh_lane_wake), looking
   again every MESH_IDLE_POLL_NS while its link is up, else every MESH_PAUSE_NS */
#define MESH_IDLE_NS UINT64_C(100000000)
#define MESH_IDLE_POLL_NS UINT64_C(1000000)
/* design/transport.md: the most bytes a port holds unsent toward a peer that takes none */
#define MESH_UNSENT (UINT64_C(64)<<20)
/* design/transport.md: a port's link lanes (mesh.h MESH_LINK_LANES), its queue pairs made once at each of
   its pairings, on the two completion queues its device's context holds (made once a context, kept across pairings),
   each lane's share of them (`share` frames, a request one frame at least: no lane overruns its queues), and each
   lane's queue pair number (`qpn`, by which a completion names its lane).  Its state (LINK_DOWN; made and told,
   LINK_MADE; connecting; up), its job on the port's queue (`job`, dispatched at `job_at`: mesh-link.c ports_stalled;
   LINK_JOB_FLUSH the lanes of a link concluded while connected or up moved to the error state at the end of that
   conclusion's quiet window, `flush_at`, 0 for none, and of any termination's, design/transport.md B6 (b),
   B9b) and a registration its make refused (`refused`), the pairing it belongs to (`pairing`, moved at each loss and at
   a halt, a peer interface's termination heard or a guard not held, mesh-link.c port_halt, so a job of a pairing gone
   makes no call and is dropped as it ends, B10, B9b), the peer's lanes (`peer`, heard in its LINK: `heard`); the lanes
   the progress threads serve (`served`, a bit a lane, the event loop's to set and to clear, a thread's to clear where
   it found its lane failed, every lane at a failed completion, mesh-link.c link_failed, and the termination's queue's,
   B9b), and the lanes a progress thread found failed (`broken`: the event loop concludes the link on reading it, and both
   ends pair it afresh, every queue pair destroyed and made again).  Each lane's SEND gate (`gate`: the frames posted and
   retired, the send thread's; its capacity, set as the lane connects), each data lane's progress (`lane`, struct
   lane_state) and the arena page of the data lanes' rings (`rings`, leased as the port first pairs, MESH_ABSENT none:
   mesh.h struct mesh_lane).  Lane MESH_LANE_KEEPALIVE's frames (design/transport.md,
   design/transport.md: one whenever the link is up, every `keepalive_ns` while a lane of the port is leased,
   else every MESH_PLATFORM_NS/5; the send thread's frame in flight and its census `alive_posted`, the receive thread's
   `alive_heard`, which the event loop reads: mesh-link.c port_silent), and its page: the keepalive's block */
enum { LINK_DOWN, LINK_MADE, LINK_CONNECTING, LINK_UP };
enum { LINK_JOB_MAKE=1, LINK_JOB_CONNECT, LINK_JOB_FLUSH };
#define LINK_ALIVE_SENT (UINT64_C(3)<<56)
#define LINK_ALIVE_HEARD (UINT64_C(4)<<56)
#define LINK_ALIVE_BYTES 16u
#define LINK_ALIVE_RECEIVES 256u
struct link {
  struct mesh_verbs provider;
  int state,job,result,heard,refused;
  _Atomic int done;
  _Atomic uint32_t served,broken,pairing;
  uint32_t peer[MESH_LINK_LANES][2],qpn[MESH_LINK_LANES],share,page,pages,rings,job_pairing;
  struct send_gate gate[MESH_LINK_LANES];
  struct lane_state lane[MESH_LINK_LANES];
  uint64_t alive_at,job_at,flush_at;
  _Atomic uint64_t alive_posted,alive_heard;
  _Atomic int alive_flight;
  int counted;
};
/* design/transport.md: a port, its link, its control socket's queues each way, its verbs work's queue (its
   open's job `job`, dispatched at `job_at`; `stuck` once a job of the port outlived MESH_PLATFORM_NS, or a progress thread did not hold within
   it, and the port was concluded around it, until the call returns and each progress thread has told the conclusion's
   epoch: design/transport.md B12, mesh-link.c ports_stalled), the end of its quiet window (`quiet_at`,
   MESH_PLATFORM_NS past its last conclusion or a peer interface's termination heard, B9b, moved only later and by any
   thread, before which no port job but an earlier conclusion's flush, struct link `flush_at`, is dispatched on it and
   the bridge does not exit: B9), whether its device is to be opened (`opening`: at the start, at its link's coming on,
   at a hello heard while it is not open, B16, and as the call it was concluded around returns, B12; never on a clock),
   the queue pairs its link lanes hold (`committed`, of the device's usable queue pairs), its link lanes, and its progress
   threads (#progress: the epochs each has seen of `epoch`, which a conclusion around a stuck call moves, and whether each
   is parked on the port's bell); the guard of its device's verbs objects (`guard`, odd while raised, the word the progress threads
   park on while it stands, design/transport.md B13, and the value each progress thread last held at:
   mesh-link.c guard_raise; each epoch and guard told wakes the raiser parked on it, port_tell, port_wait), the threads
   that post on its queue pairs (`thread`: its two progress threads, each storing itself as it starts, and its link
   job's while one runs, by which a provider's fault names its port, mesh-link.c fault_heard), and the event loop's reading of what its
   control socket heard (design/transport.md, mesh-link.c port_silent: when it last heard anything,
   `heard_at`, the link's keepalive frames heard as last read, `frames`, and when the last was heard, `frame_at`, 0
   until this pairing's first, and its frame in flight, `flight` its census as last read and `flight_at` when) */
struct port {
  struct mesh_device device;
  uint32_t index,interface,peer,peer_port,greeted,parted,connecting,writing,open,opening,committed,stuck;
  int64_t farewell;
  char name[IFNAMSIZ];
  int control,job,result,refused,threads;
  _Atomic int done,closing;
  uint64_t next,bandwidth,heard_at,frames,frame_at,flight,flight_at,job_at;
  _Atomic uint64_t quiet_at;
  struct message heard;
  uint16_t lid;
  uint8_t gid[16];
  unsigned char *in,*out;
  size_t held,in_room,out_length,out_sent,out_room;
  dispatch_queue_t work;
  struct link link;
  _Atomic uint64_t epoch,seen[2];
  _Atomic int parked[2];
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
/* design/transport.md: a port's progress thread: passes while anything moves, and with nothing moved for
   MESH_IDLE_NS it parks on its port's bell, looking again every MESH_IDLE_POLL_NS while anything is its to look for
   (port_pass), else every MESH_PAUSE_NS, and at once when the bell rings: a client's release of a lane's units on the
   port (mesh-call.h mesh_lane_wake), the lanes' change, the bridge stopping.  A thread tells it is parked, then looks
   once more before it waits, so a release whose ring read it unparked is served by that look, and a guard raised
   meanwhile is held (its raiser rings once and parks: mesh-link.c port_wait) */
#define MESH_IDLE_PASSES 4096u
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
void lanes_cancel(const struct port *port);
void lanes_turn(struct port *port);
void record_changed(void);
void record_retell(void);
struct mesh_record *record_held(uint32_t node);
void health_turn(void);
void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes);
void port_lost(struct port *port,int64_t code);
void clients_turn(int liveness);

#endif
