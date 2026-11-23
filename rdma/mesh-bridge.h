/* design/transport.md: the bridge's own declarations, shared by its two translation units, mesh-link.c (the
   ports, links and lanes, the progress threads and the lanes' data path, the event loop and the stop) and mesh-records.c
   (the map's records, the node's health and the client slots): the types, the state both read (defined in mesh-link.c)
   and the functions each calls of the other's */
#ifndef MESH_BRIDGE_H
#define MESH_BRIDGE_H
#include "mesh-verbs.h"
#include "mesh-observe.h"
#include "mesh-switched.h"
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
#include <IOKit/IOMessage.h>
#include <mach-o/dyld.h>
#include <mach-o/getsect.h>

/* design/transport.md section 14: a link's queue pairs, made once at its child's one pairing, are the lanes of its plan
   (link_plan, design/HETEROGENEOUS-LINKS.md), MESH_LINK_LANES at most, two a stream: stream s's lanes 2 s and 2 s + 1
   (the region's lane records of pair s, mesh.h struct mesh_lanes).  Lane j is queue pair j toward the link's peer and
   the switch's lane j of the link's port (mesh-switched.h); a lane's SENDs land in the order they were posted */
#define MESH_LINK_LANES (2u*MESH_LINK_STREAMS)
/* TN3205 (README.md): a SEND and the receive it fills cover the same number of frames of MESH_FRAME bytes (mesh-call.h),
   a SEND stalled by the hardware until its receive is posted.  A count of frames: a queue's capacity, a lane's
   share of its completion queues, a SEND's, a type of its own so no count of pages or bytes passes
   for one; a SEND of `bytes` covers frames_of(bytes) */
struct frames { uint64_t n; };
static inline struct frames frames_of(uint64_t bytes){ return (struct frames){(bytes+MESH_FRAME-1)/MESH_FRAME}; }
static inline struct frames frames_min(struct frames a,struct frames b){ return a.n<b.n?a:b; }
/* design/HETEROGENEOUS-LINKS.md section 5: the lanes a link makes, a bit a lane of the layout above, from `budget`, the
   fewer of the queue pairs its two ends' devices have for it: a stream's two lanes for every two queue pairs of the
   budget, at most MESH_LINK_STREAMS streams, its lanes from 0; none where the budget is fewer than two.  A link's
   streams past its lanes' are never live.  The plan of fewer queue pairs is a subset of the plan of more */
static inline uint32_t link_plan(uint32_t budget){
  const uint32_t streams=budget/2<MESH_LINK_STREAMS?budget/2:MESH_LINK_STREAMS;
  return (UINT32_C(1)<<(2*streams))-1;
}
static inline uint32_t plan_lanes(uint32_t plan){ return (uint32_t)__builtin_popcount(plan); }
static inline uint32_t plan_streams(uint32_t plan){ return plan_lanes(plan)/2; }
_Static_assert(MESH_LINK_LANES<32,"a bit a lane");

_Static_assert(sizeof(struct ibv_send_wr)==128,"M06");
_Static_assert(sizeof(struct ibv_sge)==16,"M07 M09");
_Static_assert(sizeof(struct ibv_recv_wr)==32,"M08");
_Static_assert(sizeof(struct ibv_wc)==48,"M11");
_Static_assert(sizeof(struct ibv_send_wr *)==8,"M15");
/* design/transport.md */
union mesh_network_event {
  struct kern_event_msg header;
  unsigned char bytes[KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data)];
};
_Static_assert(sizeof(union mesh_network_event)==48,"M24");
/* A lane's progress in its child: receives posted (`posted`) and landed (`got`), the receives its queues take
   (`frame_capacity`, its share of them), the SENDs it may have outstanding (`depth`: its queues' share of its send
   capacity over its frames, a post past which would fail), and its previous SEND completion (`last`: its next one's
   service begins no earlier); its SENDs posted and completed, their pages and post times are its in-flight ring's
   (mesh-switched.h struct mesh_switch_lane: a queue pair completes its SENDs in their order, the ring's) */
struct lane_state { uint64_t posted,got,frame_capacity,depth,last; };
/* design/transport.md: a message on a port's control socket (the bridge's wire: HELLO, RECORD and LINK, nothing
   else), `bytes` of what follows it: a RECORD's record.  A port's LINK tells its link lanes' queue pairs (#lanes:
   `count` the lanes of its plan, each one's number and first sequence number in `pair`, 0 for a lane the plan leaves
   out).  A HELLO on a control socket states in `count` the queue pairs its port's device has for a link (its budget:
   design/HETEROGENEOUS-LINKS.md, link_plan).  `unused` is unused (a stopping bridge says nothing: its peer reads the
   stop at the control socket's end), so every field after it lies where it lay in every version before */
enum { HELLO=1, RECORD, LINK };
struct message { uint32_t magic,kind,node,count; uint64_t boot; uint32_t unused[3],port,bytes;
  uint16_t lid,version; uint8_t gid[16]; uint32_t pair[MESH_LINK_LANES][2]; };
_Static_assert(offsetof(struct message,port)==36 && offsetof(struct message,version)==46 && sizeof(struct message)==128,
               "a HELLO's fields at the offsets every version has them");
/* the wire's magic: what marks a datagram or a control stream's header as the mesh's.  It is the value version 131
   sends, fixed: no version enters it, so a bridge hears a bridge of any version (the operator, 2026-10-21: "rejection
   filter removal. categorical. all of them without sanction in rubric"; "even this fake limit too must go away").  A
   message states its sender's MESH_VERSION in `version` (0 from a bridge before it), which the greeting alone tests:
   a bridge pairs only with one of its own version (mesh-link.c port_heard).  A reader finds a message's bounds where
   its sender's struct message is its own size: 128 bytes from version 163 (MESH_LINK_LANES entries in `pair`), 144
   before.  A HELLO's fields lie at the same offsets in both, so a longer HELLO is read by its first bytes (a datagram
   cut short, mesh-link.c hello_heard; a control stream's first message, control_heard) and its pairing refused on its
   version; past that HELLO the stream is misframed, and lost (EPROTO) at the next message its sender sends.  A
   shorter datagram ends hello_heard's turn */
#define XMAGIC (0x4d595048u+131u+(1u<<12))
/* design/transport.md section 14, "A process a port": a port child's link, the child's alone (its parent's side of the
   port is struct port): its port (`index`, the parent's, by which its lane records, switch lanes, line and rings are
   the region's), its device (`device`, the provider's rdma_<interface>, and its interface's `name`) and the device's
   address (`lid`, `gid`, as its open or make read it), the peer node its connect names (`peer`); its lanes
   (MESH_LINK_LANES at most), its queue pairs made once and connected once in the child's life (`provider`, queue j lane
   j; a link lost after it pairs again with the port's next child), on the two completion queues its device's context
   has, each lane's share of them (`share` frames, a request one frame at least: no lane overruns its queues), and each
   lane's queue pair number (`qpn`, by which a completion names its lane).  The lanes its pairing makes (`plan`, a bit
   a lane: link_plan of the fewer queue pairs its two ends have, as its parent's make names them; its lanes left out
   have no queue pair and their `qpn` UINT32_MAX, no queue pair's number).  Each lane's progress (`lane`, struct
   lane_state, its depth set as the lane connects), and the time of the link's last poll that completed a SEND
   (`completed_at`, the child's clock: its cable's terms, mesh-link.c link_send).  The poller's takes from the clients'
   write rings (mesh-link.c switch_takes): the slot its next pass looks at first (`take_from`); and its posts from its
   port's inbound rings (mesh-link.c switch_sends): the pushing port whose ring its next pass reads first
   (`send_from`) */
struct link {
  struct mesh_device device;
  struct mesh_verbs provider;
  uint32_t index,peer,plan,qpn[MESH_LINK_LANES];
  char name[IFNAMSIZ];
  uint16_t lid;
  uint8_t gid[16];
  struct frames share;
  struct lane_state lane[MESH_LINK_LANES];
  uint64_t completed_at;
  uint64_t wire[MESH_LINK_STREAMS];
  uint32_t take_from,send_from;
};
/* design/transport.md section 14, "A process a port": the messages between the bridge's parent and a port's child on
   their socket pair (a stream, AF_UNIX having no SOCK_SEQPACKET here: each a whole struct child_message), each
   written by one thread at each end (the parent's event loop, the child's main thread).  The parent's, a command at
   a time, each answered in its order: open the device (CHILD_OPEN), make the link's lanes (CHILD_MAKE names the lanes
   of the plan in `lanes`), connect them to the peer's (CHILD_CONNECT: the peer's lanes, address, node and the link's
   bandwidth); and end (CHILD_HALT, the child's one end, sent at a loss from the make on, at a child's own halt and at
   the bridge's stop, and again where a loss or termination heard while the child ends moves the port's window: in
   `until` the end of its port's quiet window, the parent's struct port `quiet_at` in CLOCK_UPTIME_RAW nanoseconds,
   the clock the child reads too, so the child exits no sooner than that end and once its own windows have ended:
   mesh-link.c end_step).  The child's: the device opened or not (CHILD_OPENED: its
   address and bandwidth, and in `lanes` the queue pairs the device has for a link), the lanes made or not
   (CHILD_MADE: each lane's queue pair number, the first sequence number and each lane's share), connected or not
   (CHILD_CONNECTED); and its own halt (CHILD_BROKEN, once in its life, its reason in `result`: EIO a lane's or a
   completion's failure, ENETDOWN a fault absorbed on the provider's doorbell, ETIMEDOUT its port's peer interface
   terminating as the child heard it, B9b), after which it takes no command and ends as at a loss.  No message names a
   pairing: from its make on a child serves one pairing, and any loss after the make ends it, so no answer of a
   pairing lost can act on another pairing's queue pairs, which belong to another process */
/* design/transport.md section 14, "A process a port": the version of the contract between the bridge and its children,
   declared: their command line, `mesh-flow --child VERSION DEVICE INDEX` (its first three words the same for every
   version), and struct child_message.
   It changes only where that contract does; a build of the same contract is the same child, whatever its bytes (the
   operator, 2026-10-21).  9 (design/PAGE-TABLE-SPEC-20261122.md): a child makes a link's lanes as queue pairs from 0
   (lane j queue pair j and its port's switch lane j: mesh-bridge.h link_plan), posts its lanes' receives into free
   pages of the node's pool, lands pages into the page table and forwards them by their headers, takes its clients'
   writes from its port's share of the rings (mesh-switched.h struct mesh_switch `port`, read every pass), posts what
   its port's inbound rings have, each SEND in its lane's in-flight ring, and counts what it cannot carry and reports
   nothing (each of these the region's, of MESH_LAYOUT).  Against 8, whose
   links made queue pairs 0 and 1 for nothing and their data lanes from 2: CHILD_MAKE's plan, CHILD_MADE's `qpn`,
   CHILD_CONNECT's `peer` and CHILD_BROKEN's `broken` name lane j as queue pair j, MESH_LINK_LANES of them (8's 10).
   10: the child owns its exit.  CHILD_HALT is its one end and states why in `result` (0 the bridge stopping, else
   the loss's code), and the child exits by its own windows and that reason; CHILD_END goes.  CHILD_BROKEN, its reason
   in `result`, is its one report of its own halt, so CHILD_FAULTED and CHILD_TERMINATED go, and a child halted
   by any of them ends.  No message names a pairing (`pairing` and `broken` go), and the kinds are numbered anew.
   11: CHILD_HALT names the end of its port's quiet window in `until`, a field appended to struct child_message, in
   place of 10's reason in `result`, which a halt no longer carries.  The child exits no sooner than that end: a stop
   that follows a loss the child was not told of (its link gone off, or its control socket lost before the make)
   keeps its exit out of the window that loss opened (B9), as the parent's CHILD_END, held until the window had ended,
   did before 10; 10's stop let it exit at once.  A halt at a loss names the window the loss has just moved
   MESH_PLATFORM_NS on, 10's floor for a halt naming a loss */
#define MESH_CHILD_VERSION 11u
enum { CHILD_OPEN=1, CHILD_MAKE, CHILD_CONNECT, CHILD_HALT, CHILD_OPENED, CHILD_MADE, CHILD_CONNECTED, CHILD_BROKEN };
struct child_message {
  uint32_t kind;
  int32_t result,refused;
  uint32_t qpn[MESH_LINK_LANES],peer[MESH_LINK_LANES][2],psn,share,node,lanes;
  uint16_t lid,padding;
  uint8_t gid[16];
  uint64_t bandwidth,until;
};
#define TAG_CHILD (UINT64_C(1)<<58)
/* design/transport.md section 14: a port of the bridge's parent, one state machine on its event loop (mesh-link.c
   port_step, its one transition function; the loop its one owner), each state's one deadline `due` (0: none):
   - CLOSED: no child.  `due`, where the port is to be opened: the spawn's time, no sooner than its quiet window's end;
   - OPENING: CHILD_OPEN out to its child.  `due` its bound, MESH_PLATFORM_NS on, past which the port is concluded
     around its child's call and waits for its answer (B12); `wanted` once its link comes on meanwhile, so an open that
     failed is asked again;
   - DOWN: its child alive, its device not open (its open failed, or its link went off).  `due`, where it is to be
     opened: no sooner than its quiet window's end and a refused registration's retry;
   - OPEN: its device open, no control socket.  `due` its next hello;
   - DIALING: this node the lower, its connect to the peer's listener out;
   - GREETING: its control socket up and its HELLO said, the peer's awaited;
   - PAIRED: the peer's HELLO heard and the plan made.  `due` its quiet window's end, when its child is asked to make
     the lanes (B9);
   - MAKING: CHILD_MAKE out.  From here on any loss ends the child.  `due` its bound (B12);
   - MADE: its LINK told the peer; the peer's LINK and its quiet window's end awaited, then CHILD_CONNECT (`due` the
     window's end once the LINK is in);
   - CONNECTING: CHILD_CONNECT out.  `due` its bound (B12);
   - UP: its child serves the link;
   - EXITING: its child halted (CHILD_HALT, at a loss or the bridge's stop), its exit awaited (the loop's SIGCHLD,
     mesh-link.c children_reaped); the port's next child is spawned only after it is reaped, so a device has one
     driving process.
   A port's fields: its index (its child's, the switch's), its device (`device`, the provider's rdma_<interface>, as
   IOKit's first match named it) and interface (`name`, `interface` its index once it is first opened; `family`,
   `unit`: an en<N>'s APPLE_IF_FAM_ETHERNET and N, read once from its name; another port's family 0, which no event
   carries: mesh-link.c addresses_heard), the end of its quiet window (`quiet_at`, MESH_PLATFORM_NS past its last
   loss or a peer interface's termination heard, B9b, moved only later and only by the event loop, before which no
   command is sent for it: B9), the queue pairs its device has for a link as its child told them at its opening
   (`budget`: design/HETEROGENEOUS-LINKS.md), the registrations its device refused in a row (`refusals`) and the time
   before which it is not opened again after one (`retry_at`, cleared at its link's coming on); its address and
   bandwidth as its child told them (`lid`, `gid`, `bandwidth`); its peer (`peer`, `peer_port`, its HELLO `heard`,
   whose address its LINK replaces), the lanes its pairing makes (`plan`), its own lanes as its child made them
   (`qpn`, `psn`, `share`) and the peer's as its LINK told them (`lanes`, `lanes_heard`); its control socket and its
   queues each way (`writing`: a write's room awaited); and its child (`child`, 0 none; its entry among the region's
   processes, `child_entry`, mesh-observe.h struct mesh_bridge_pages `child`; its socket, and the part of its next
   message read) */
enum port_state { PORT_CLOSED, PORT_OPENING, PORT_DOWN, PORT_OPEN, PORT_DIALING, PORT_GREETING, PORT_PAIRED, PORT_MAKING,
                  PORT_MADE, PORT_CONNECTING, PORT_UP, PORT_EXITING };
struct port {
  enum port_state state;
  uint64_t due;
  char device[32],name[IFNAMSIZ];
  uint32_t index,interface,family,unit,budget,refusals,wanted,peer,peer_port,plan,psn,share,lanes_heard,writing;
  uint32_t qpn[MESH_LINK_LANES],lanes[MESH_LINK_LANES][2];
  uint64_t quiet_at,retry_at,bandwidth;
  uint16_t lid;
  uint8_t gid[16];
  struct message heard;
  int control;
  unsigned char *in,*out;
  size_t in_length,in_room,out_length,out_sent,out_room;
  pid_t child;
  uint32_t child_entry;
  int child_fd;
  uint32_t child_in_length;
  unsigned char child_in[sizeof(struct child_message)];
};
/* a port whose peer's HELLO it has heard (PAIRED to UP): the one a RECORD goes to (mesh-records.c) */
static inline int port_greeted(const struct port *port){ return port->state>=PORT_PAIRED && port->state<=PORT_UP; }

/* design/transport.md: the most nanoseconds between two samples of this node's available memory, each sent
   with its record (mesh-records.c health_turn) */
#define MESH_HEALTH_NS UINT64_C(10000000000)

/* the state both translation units read, defined in mesh-link.c */
extern struct hdr *M;
extern struct wire_mapping wire;
extern struct port ports[MESH_PORTS];
extern uint32_t port_count;
extern int events,listener,hello,network,addresses;
extern uint64_t now,sequence;
extern struct mesh_ctx bridge;
extern uint64_t memory;
extern uint64_t owners[MESH_CLIENTS];
extern char model[MESH_MODEL_BYTES];

/* whether a client's slot is still `owner`'s */
static inline int owned(uint64_t owner){ return owner && atomic_load(&mesh_client_at(M,owner)->owner)==owner; }

/* the functions each translation unit calls of the other's */
void sysctl_text(const char *name,char *out,size_t room);
void health_init(void);
void port_say(struct port *port,struct message said,const void *payload,uint32_t bytes);
uint64_t lanes_busy(void);
uint64_t pool_bytes(void);
void record_changed(void);
void switch_view(void);
uint32_t record_reach(struct mesh_reach *reach,uint32_t room);
void record_retell(void);
void map_tell(struct port *port);
void records_forget(const struct port *port);
uint64_t health_due(void);
void health_turn(void);
void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes);
void clients_turn(void);

#endif
