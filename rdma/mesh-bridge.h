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
#include <uuid/uuid.h>

_Static_assert(sizeof(pthread_t)==8,"M17");

/* design/transport.md section 14: a link's queue pairs, made once at its agent's one pairing, are the lanes of its plan
   (link_plan, design/HETEROGENEOUS-LINKS.md: all MESH_LINK_LANES where both ends' devices hold that many), of this layout:
   lane MESH_LANE_KEEPALIVE the keepalive's, lane 1 reserved and unused, and from MESH_LANE_FIRST pairs, a pair's large
   lane (MESH_LANE_FIRST + 2 p) and its small lane (+ 2 p + 1).  The link's MESH_LINK_STREAMS streams (mesh.h struct
   mesh_stream) take MESH_STREAM_PAIRS pairs each, stream s pairs s MESH_STREAM_PAIRS on, a queue pair carrying one ring
   (a SEND fills the oldest receive posted on its queue pair); a stream's unit u of a class rides the lane of that class
   of its pair u mod MESH_STREAM_PAIRS, as that lane's (u / MESH_STREAM_PAIRS)-th unit, alike at both ends (a lane's
   units land in order) */
#define MESH_LINK_LANES 10u
#define MESH_LANE_KEEPALIVE 0u
#define MESH_LANE_FIRST 2u
#define MESH_LANE_PAIRS ((MESH_LINK_LANES-MESH_LANE_FIRST)/2)
#define MESH_STREAM_PAIRS (MESH_LANE_PAIRS/MESH_LINK_STREAMS)
_Static_assert(MESH_LANE_FIRST+2*MESH_LANE_PAIRS==MESH_LINK_LANES,"the lanes' pairs");
_Static_assert(MESH_STREAM_PAIRS && MESH_STREAM_PAIRS*MESH_LINK_STREAMS==MESH_LANE_PAIRS,"a link's pairs, its streams' whole");
_Static_assert(!(MESH_STREAM_LARGE_UNITS%MESH_STREAM_PAIRS) && !(MESH_STREAM_SMALL_UNITS%MESH_STREAM_PAIRS),"a stream's ring striped whole");
static inline uint32_t lane_class(uint32_t k){ return (k-MESH_LANE_FIRST)&1; }
static inline uint32_t lane_stream(uint32_t k){ return (k-MESH_LANE_FIRST)/2/MESH_STREAM_PAIRS; }
static inline uint32_t lane_stripe(uint32_t k){ return (k-MESH_LANE_FIRST)/2%MESH_STREAM_PAIRS; }
/* design/HETEROGENEOUS-LINKS.md: the lanes a link makes, a bit a lane of the layout above, from `budget`, the fewer of the
   queue pairs its two ends' devices hold for it: every lane where both hold MESH_LINK_LANES (the layout of every bridge
   before budgets, lane 1 with it); else no lane 1, the keepalive's lane where they hold three or more, and as many
   streams' lane pairs as the rest holds, at most MESH_LINK_STREAMS (two queue pairs: one stream, no keepalive); none
   where they hold fewer than one stream's two.  A link's streams past its lanes' are never live.  The plan of fewer
   queue pairs is a subset of the plan of more */
static inline uint32_t link_plan(uint32_t budget){
  if(budget>=MESH_LINK_LANES)return (UINT32_C(1)<<MESH_LINK_LANES)-1;
  const uint32_t keepalive=budget>=3,pairs=(budget-keepalive)/(2*MESH_STREAM_PAIRS);
  const uint32_t streams=pairs<MESH_LINK_STREAMS?pairs:MESH_LINK_STREAMS;
  return (keepalive<<MESH_LANE_KEEPALIVE)|(((UINT32_C(1)<<(2*MESH_STREAM_PAIRS*streams))-1)<<MESH_LANE_FIRST);
}
static inline uint32_t plan_lanes(uint32_t plan){ return (uint32_t)__builtin_popcount(plan); }
static inline uint32_t plan_streams(uint32_t plan){ return (uint32_t)__builtin_popcount(plan>>MESH_LANE_FIRST)/(2*MESH_STREAM_PAIRS); }
_Static_assert(MESH_LINK_LANES<32,"a bit a lane");
/* design/HETEROGENEOUS-LINKS.md P2, make until refused: the order a link's lanes are made in, the first stream's lane
   pair, the keepalive's lane, the other streams' lane pairs, lane 1 last, so that the first c made always hold
   link_plan(c) (and the lanes of link_plan(b) are the first plan_lanes(link_plan(b)) of it, for every b): a make the
   device refuses part of has made the plan of the fewer, a subset of every larger plan, and both ends connect
   link_plan of the fewer queue pairs either made (mesh-link.c link_make, link_turn) */
static const uint8_t lane_order[MESH_LINK_LANES]={MESH_LANE_FIRST,MESH_LANE_FIRST+1,MESH_LANE_KEEPALIVE,4,5,6,7,8,9,1};
_Static_assert(MESH_LINK_LANES==10 && MESH_LANE_FIRST==2 && MESH_LANE_KEEPALIVE==0 && MESH_STREAM_PAIRS==1,
               "lane_order is the layout's: link_plan(c) within its first c");

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
/* design/transport.md section 14: a data lane's progress, the bridge's alone, counted in the lane's own units (its
   count since its link connected, which both ends of the cable keep alike: a SEND fills the oldest receive posted on its
   queue pair): the units it has posted SENDs for (`sent`, the send thread's) and those of them whose posting time it
   stored (`timed`, the send thread's: mesh-link.c link_send), and received (`posted` receives posted, `got` landed, the
   receive thread's, posted at most `window` units ahead: the lane's share of the completion queue in its frames); and
   its requests, one a slot of its stream's ring each way (`sends`, `receives`, made over the rings of the slot its
   binding names, `sends_slot`, `receives_slot`, MESH_ABSENT none: mesh-link.c lane_requests), and after them the dud's,
   from the bridge's all-ones block, and the sink's, into the link's block, where a drain absorbs what the peer pads
   (mesh-link.c lane_reset) */
struct lane_state {
  uint64_t sent,timed,posted,got;
  uint32_t window,sends_slot,receives_slot;
  struct prepared_send *sends;
  struct prepared_receive *receives;
};
/* design/transport.md section 14, "A stream over several cables": a stream slot's bridge words, after its large class's
   landed words in its port's ring pages (mesh-link.c lanes_at, stream_rings), which no client header names.
   - The binding of the slot's port's lanes of that stream, the parent's (`told`: (generation << 2) | state): frozen
     (BOUND_FROZEN: nothing posted on them), drained (BOUND_DRAIN: SENDs padded with duds to `out` and receives absorbed
     into the link's sink to `in`, each class's) or open (BOUND_OPEN: the lanes are stripe `stripe` of the stream of slot
     `slot`, their bases `out` and `in`).  The parent writes its fields only once both threads have told the binding
     before it (mesh-link.c group_advance), so a thread reads them as the binding it reads them under.
   - What the port's progress threads tell of those lanes: each class's SENDs posted (`sent`) and the binding they hold
     to (`sending`), the send thread's; each class's receives posted and landed (`posted`, `got`) and the binding they
     hold to (`receiving`), the receive thread's.  A thread tells a binding once it acts on it alone: frozen, once it
     posts no more; drained, once its lanes are at its counts.
   - Where the slot holds a group's stream (its leader's: the group's least port), the group as the parent opened it:
     its stripes (`stripes`), each one's port (`port`) and lane bases (`base_out`, `base_in`); the owner word its leader's
     send thread serves and each class's holding end (`held`, `final`); and each class's first unit not yet released,
     and not yet given its landing time, which any member's receive thread advances (`released`, `timed`) */
enum { BOUND_FROZEN, BOUND_DRAIN, BOUND_OPEN };
struct stream_lanes {
  _Alignas(128) _Atomic uint64_t told;
  uint32_t slot,stripe;
  uint64_t out[2],in[2];
  _Alignas(128) _Atomic uint64_t sent[2],sending;
  _Alignas(128) _Atomic uint64_t posted[2],got[2],receiving;
  _Alignas(128) uint32_t stripes,port[MESH_PORTS];
  uint64_t base_out[MESH_PORTS][2],base_in[MESH_PORTS][2];
  _Alignas(128) _Atomic uint64_t held,final[2];
  _Alignas(128) _Atomic uint64_t released[2],timed[2];
};
_Static_assert(!(2*MESH_STREAM_LARGE_UNITS*8%128),"a slot's bridge words on lines of their own");
/* a progress thread's hold of its port's binding of one stream (mesh-link.c bound_read): the binding and its state;
   drained, the counts (`out`, `in`); open, the bases (`out`, `in`), the group's slot (its stream, `x`, and words,
   `group`), this port's stripe of `stripes`, each stripe's port's words (`member`) and bases; the slot's own words
   (`own`), and the send thread's most each class's holder declared as it read (`seen`) */
struct bound {
  _Alignas(128) uint64_t told;
  uint32_t state,slot,stripe,stripes;
  uint64_t out[2],in[2],seen[2];
  struct mesh_stream *x;
  struct stream_lanes *own,*group,*member[MESH_PORTS];
  uint64_t base_out[MESH_PORTS][2],base_in[MESH_PORTS][2];
};
/* design/transport.md: a message on a port's control socket (the bridge's wire: HELLO, FAREWELL, RECORD
   and LINK, nothing else), `bytes` of what follows it: a RECORD's record.  A FAREWELL's `code` is why its bridge stops;
   a port's LINK tells its link lanes' queue pairs (#lanes: `count` the queue pairs its make made, the first `count` of
   lane_order, each one's number and first sequence number in `pair`, 0 for a lane not made; every lane of its plan
   where its device refused none, so a LINK is a bridge's before make until refused wherever that bridge's make would
   have succeeded: design/HETEROGENEOUS-LINKS.md P2).  A HELLO on a control socket states in `count` 1 + the queue pairs
   its port's device holds for a link (design/HETEROGENEOUS-LINKS.md: hello_budget; a bridge before budgets sends 0
   there and makes MESH_LINK_LANES, so two ends that both hold that many pair as before budgets).  Every message names
   its bridge's node and that node's host UUID (`uuid`, mesh.h struct mesh_record's), so a hello of this node's number
   from another host is told (mesh-records.c number_shared).  A STREAM (design/transport.md section 14, "A stream over
   several cables") carries a struct stream_told: on each cable of a group its bridge makes anew, once a change of the
   cables joining the two nodes, as a pairing's LINK does once a cable */
enum { HELLO=1, FAREWELL, RECORD, LINK, STREAM };
struct message { uint32_t magic,kind,node,count; uint64_t boot; int64_t code; uint32_t block,port,bytes;
  uint16_t lid,padding; uint8_t gid[16]; uint32_t pair[MESH_LINK_LANES][2]; uint8_t uuid[16]; };
/* a STREAM's payload: the stream's index, the cables its group is made over (`members` of `wire`, each one's identity
   for that stream, mesh-link.c stream_wire, alike at both ends: ascending), and the sender's lanes of that stream on
   this cable as its threads froze them, each class's SENDs and receives posted */
struct stream_told { uint32_t stream,members; uint64_t wire[MESH_PORTS],sent[2],posted[2]; };
static inline uint32_t hello_budget(uint32_t count){ return count?count-1:MESH_LINK_LANES; }
/* the wire's magic: its base, MESH_VERSION and the wire's revision (1<<12, past any base and version's: every link heard
   within MESH_PLATFORM_NS or concluded, design/transport.md B1, B2, B3, which a bridge of the same version
   without them breaks), so bridges of different versions or revisions drop each other's hellos and never pair */
#define XMAGIC (0x4d595048u+MESH_VERSION+(1u<<12))
/* design/transport.md: the most bytes a port holds unsent toward a peer that takes none */
#define MESH_UNSENT (UINT64_C(64)<<20)
/* design/transport.md section 14: a port's link lanes (MESH_LINK_LANES), its queue pairs made once and connected once
   in its agent's life ("A process a port": `lanes` once the agent was asked for them, the parent's; a link lost after it
   pairs again with the port's next agent), on the two completion queues its device's context holds, each lane's share of them (`share`
   frames, a request one frame at least: no lane overruns its queues), and each lane's queue pair number (`qpn`, by which
   a completion names its lane).  Its state (LINK_DOWN; made and told, LINK_MADE; connecting; up), its job on the port's
   queue (`job`, dispatched at `job_at`: mesh-link.c ports_stalled) and a registration its make refused (`refused`), the
   pairing it belongs to (`pairing`, moved at its loss and at a halt, mesh-link.c port_halt, so a job of a pairing gone
   makes no call and is dropped as it ends, B10, B9b), the peer's lanes (`peer`, heard in its LINK: `heard`); the lanes
   the progress threads serve (`served`, a bit a lane, the event loop's to set and to clear, a thread's to clear where
   it found its lane failed, every lane at a failed completion, mesh-link.c link_failed, and the termination's queue's,
   B9b), and the lanes a progress thread found failed (`broken`: the event loop loses the link on reading it).  The
   lanes its pairing makes (`plan`, a bit a lane: link_plan of the fewer queue pairs its two ends hold, set as the
   parent hears the peer's HELLO and told the agent with the make), those its make made (`made`, a bit a lane: the
   plan's in lane_order until the device refused one, and `refusal`, the refusal's code, 0 where it made the plan
   whole: design/HETEROGENEOUS-LINKS.md P2) and the queue pairs the peer's make made (`peer_made`, its LINK's `count`);
   from its connect, `plan` is the lanes it connects and serves, link_plan of the fewer queue pairs either end made (a
   subset of what each made), told the agent with the connect; a lane left out has no queue pair connected and its
   `qpn` UINT32_MAX, no queue pair's number (one made past the connected plan stays in INIT until the agent's exit).  Each
   lane's SEND gate (`gate`: the frames posted and retired, the send thread's; its capacity, set as the lane connects),
   each data lane's progress (`lane`, struct lane_state), the streams of the port's slots whose rings the parent holds
   (`rings`, a bit a stream: each its plan carries, leased as a pairing first needs it, mesh-link.c stream_rings; mesh.h
   struct mesh_stream), each stream's identity on this cable (`wire`, the
   parent's, made as the link connects: mesh-link.c stream_wire), and each progress thread's hold of each stream's
   binding (`bound`, the receive thread's [0] and the send thread's [1], each on lines of its own: struct bound).  Lane
   MESH_LANE_KEEPALIVE's frames (design/transport.md: one whenever the link is up, every `keepalive_ns` while a stream
   it carries is held, else every MESH_PLATFORM_NS/5; the send thread's frame in flight), and its page: the keepalive's
   block, and the sink a drain absorbs into */
enum { LINK_DOWN, LINK_MADE, LINK_CONNECTING, LINK_UP };
enum { LINK_JOB_MAKE=1, LINK_JOB_CONNECT };
#define LINK_ALIVE_SENT (UINT64_C(3)<<56)
#define LINK_ALIVE_HEARD (UINT64_C(4)<<56)
#define LINK_ALIVE_BYTES 16u
#define LINK_ALIVE_RECEIVES 256u
struct link {
  struct mesh_verbs provider;
  int state,job,result,heard,refused,lanes,refusal;
  _Atomic int done;
  _Atomic uint32_t served,broken,pairing;
  uint32_t plan,made,peer_made,peer[MESH_LINK_LANES][2],qpn[MESH_LINK_LANES],share,page,pages,rings,job_pairing,psn;
  struct send_gate gate[MESH_LINK_LANES];
  struct lane_state lane[MESH_LINK_LANES];
  uint64_t alive_at,job_at;
  uint64_t wire[MESH_LINK_STREAMS];
  _Atomic int alive_flight;
  struct bound bound[2][MESH_LINK_STREAMS];
};
/* design/transport.md section 14, "A process a port": the messages between the bridge's parent and a port's agent on
   their socket pair (a stream, AF_UNIX having no SOCK_SEQPACKET here: each a whole struct agent_message).  The
   parent's: open the device (AGENT_OPEN), make the link's lanes (AGENT_MAKE, naming the link's keepalive page, which the
   parent may lease after the agent's spawn), connect them to the peer's (AGENT_CONNECT: the peer's lanes, address, node
   and the link's bandwidth, and in `lanes` the lanes it connects), serve no lane again (AGENT_HALT), end (AGENT_END).
   The agent's, each naming the pairing it was asked for: the device opened or not (AGENT_OPENED: its address and
   bandwidth, and in `lanes` the queue pairs the device holds for a link), the lanes made or not (AGENT_MAKE names the
   lanes of the plan in `lanes`; AGENT_MADE: the lanes made in `lanes`, the plan's in lane_order until the device refused
   one, that refusal's code in `refusal` (0: none), each made lane's queue pair number and the first sequence number),
   connected or not (AGENT_CONNECTED: each lane's share of the completion queues),
   and the lanes a progress thread found failed (AGENT_BROKEN, naming its connect's pairing), a fault absorbed on the
   provider's doorbell (AGENT_FAULTED), a guard not held within MESH_PLATFORM_NS (AGENT_UNGUARDED), its port's peer
   interface terminating (AGENT_TERMINATED, as the agent heard it: B9b) */
enum { AGENT_OPEN=1, AGENT_MAKE, AGENT_CONNECT, AGENT_HALT, AGENT_END,
       AGENT_OPENED, AGENT_MADE, AGENT_CONNECTED, AGENT_BROKEN, AGENT_FAULTED, AGENT_UNGUARDED, AGENT_TERMINATED };
struct agent_message {
  uint32_t kind,pairing;
  int32_t result,refused,refusal;
  uint32_t qpn[MESH_LINK_LANES],peer[MESH_LINK_LANES][2],psn,share,broken,node,page,lanes;
  uint16_t lid,padding;
  uint8_t gid[16];
  uint64_t bandwidth;
};
/* an agent that ends unbidden within this of its spawn counts toward its port's hold, from the second in a row (the boot
   ledger's rule, a port's: mesh-link.c agent_reaped) */
#define MESH_AGENT_SETTLE_NS (UINT64_C(60)*1000000000)
#define TAG_AGENT (UINT64_C(1)<<58)
/* design/transport.md: a port, its link, its control socket's queues each way, its verbs work's queue (its
   open's job `job`, dispatched at `job_at`; `stuck` once a job of the port outlived MESH_PLATFORM_NS, or a progress
   thread did not hold within it, and the port was concluded around it, until the call returns and each progress thread
   has told the conclusion's epoch: design/transport.md B12, mesh-link.c ports_stalled; another port's stuck call holds
   none of this one's jobs), the end of its quiet window (`quiet_at`, MESH_PLATFORM_NS past its last loss, its own peer
   interface's termination heard or one told for no port, B9b, mesh-link.c peer_terminated, moved only later and by any
   thread, before which no port job is dispatched on it, its agent is not asked to end and the bridge does not exit:
   B9), whether its device is to
   be opened (`opening`: at the start, at its link's coming on, at a hello heard while it is not open, B16, and as the
   call it was concluded around returns, B12; never on a clock), its link lanes, and its progress threads (#progress: the
   epochs each has seen of `epoch`, which a conclusion around a stuck call moves); the guard of its device's verbs
   objects (`guard`, odd while raised, the word the progress threads wait on while it stands and while the link serves
   nothing, design/transport.md B13, and the value each progress thread last held at: mesh-link.c guard_raise; each epoch
   and guard told wakes the raiser parked on it, port_tell, port_wait), and the threads that post on its queue pairs
   (`thread`: its two progress threads, each storing itself as it starts, and its link job's while one runs, by which a
   provider's fault names its port, mesh-link.c fault_heard); the queue pairs its device holds for a link as its agent
   told them at its opening (`budget`, the parent's: design/HETEROGENEOUS-LINKS.md) */
struct port {
  struct mesh_device device;
  uint32_t index,interface,peer,peer_port,greeted,parted,connecting,writing,open,opening,stuck,budget;
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
  /* the parent's: the port's agent (design/transport.md section 14, "A process a port"): its process (0 none) and
     socket, when it was spawned, the agents in a row that ended unbidden within MESH_AGENT_SETTLE_NS of their spawn and
     the hold they set before the next (`hold_to`), whether the port's link, lost once its lanes were asked for, waits
     to end its agent (`recycle`) and whether that end was asked (`ending`), the part of its agent's next message
     read (`agent_in`, `agent_held` bytes of it), an agent of its device from an earlier launch last found alive
     (`elsewhere`: mesh-link.c agent_elsewhere), whether the last whole read of the devices no longer listed its device
     (`unlisted`) and whether it gives its slot to a device waiting for one (`yielding`: mesh-link.c ports_found); and
     its link's part in the groups (design/transport.md section 14, "A stream over several cables"): the streams it has
     carried in no group since it connected (`fresh`, a bit a stream), and the peer's latest STREAM of each stream on it
     (`stream_heard`), not yet taken into a group's drain (`streams_heard`, a bit a stream: mesh-link.c group_advance) */
  pid_t agent,elsewhere;
  uint32_t holder;
  int agent_fd,recycle,ending,unlisted,yielding;
  uint32_t aborts,agent_held;
  uint64_t agent_at,hold_to;
  unsigned char agent_in[sizeof(struct agent_message)];
  uint32_t fresh,streams_heard;
  struct stream_told stream_heard[MESH_LINK_STREAMS];
};

/* design/transport.md section 9: why this bridge stops, the code its FAREWELL carries (a signal's number) */
/* design/transport.md: the most nanoseconds between two samples of this node's available memory, each sent
   with its record (mesh-program.c health_turn) */
#define MESH_HEALTH_NS UINT64_C(10000000000)

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
extern int events,listener,hello,network,addresses,unreleased,finding;
extern uint64_t now,sequence;
extern struct mesh_ctx bridge;
extern _Atomic int found;
extern uint64_t memory;
/* this node's host UUID (mesh.h struct mesh_record `uuid`), read once at the start */
extern uint8_t node_uuid[16];
extern uint64_t keepalive_ns;
extern uint64_t owners[MESH_CLIENTS];
extern uint32_t ones;
/* the region's pages the bridge holds itself (the map, the links' pages and rings, the duds' block): its record's
   `region` is the rest, what its clients may hold */
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
void map_adopt(void);
struct mesh_record *record_held(uint32_t node);
uint64_t health_due(void);
void health_turn(void);
void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes);
void number_shared(uint32_t node,const uint8_t *held,const uint8_t *heard,uint8_t *told,const struct port *port,const char *what);
void port_lost(struct port *port,int64_t code);
uint32_t record_reach(struct mesh_reach *reach,uint32_t room);
void clients_turn(int liveness);

#endif
