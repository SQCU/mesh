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

_Static_assert(sizeof(pthread_t)==8,"M17");

/* design/transport.md section 14: a link's queue pairs, made once at its child's one pairing, are the lanes of its plan
   (link_plan, design/HETEROGENEOUS-LINKS.md), of this layout: lanes 0 and 1, on which nothing is posted, and from
   MESH_LANE_FIRST pairs, stream s on pair s (lanes MESH_LANE_FIRST + 2 s and + 2 s + 1).  A holding's pieces go on
   its stream's cables whole or cut (spread_at), and a cable's pages, each a whole page, on its two lanes in turn
   (struct cable: on one cable the stream's page p on its lane p mod 2), posted in the holding's order at each end, sent
   from the page its row names and landing in the page the receiver's row names (design/pages-and-functions.md); a
   lane's units land in the order its SENDs were posted */
#define MESH_LINK_LANES 10u
#define MESH_LANE_FIRST 2u
#define MESH_LANE_PAIRS ((MESH_LINK_LANES-MESH_LANE_FIRST)/2)
#define MESH_STREAM_PAIRS (MESH_LANE_PAIRS/MESH_LINK_STREAMS)
_Static_assert(MESH_LANE_FIRST+2*MESH_LANE_PAIRS==MESH_LINK_LANES,"the lanes' pairs");
_Static_assert(MESH_STREAM_PAIRS && MESH_STREAM_PAIRS*MESH_LINK_STREAMS==MESH_LANE_PAIRS,"a link's pairs, its streams' whole");
_Static_assert(MESH_STREAM_PAIRS==1,"a stream one lane pair");
static inline uint32_t lane_class(uint32_t k){ return (k-MESH_LANE_FIRST)&1; }
static inline uint32_t lane_stream(uint32_t k){ return (k-MESH_LANE_FIRST)/2; }
/* TN3205 (README.md): a SEND and the receive it fills cover the same number of frames of MESH_FRAME bytes, a SEND held by
   the hardware's credit until its receive is posted; the bridge's alone */
#define MESH_FRAME 4096u
/* a count of frames: a queue's capacity, a lane's share of its completion queues, a SEND's and the send gate's, a type
   of its own so no count of pages, units or bytes passes for one; a SEND of `bytes` covers frames_of(bytes) */
struct frames { uint64_t n; };
static inline struct frames frames_of(uint64_t bytes){ return (struct frames){(bytes+MESH_FRAME-1)/MESH_FRAME}; }
static inline struct frames frames_min(struct frames a,struct frames b){ return a.n<b.n?a:b; }
/* design/HETEROGENEOUS-LINKS.md sections 5 and 10: the lanes a link makes, a bit a lane of the layout above, from
   `budget`, the fewer of the queue pairs its two ends' devices hold for it: every lane where both hold MESH_LINK_LANES;
   else no lane 1, lane 0 where they hold three or more, and as many streams' lane pairs as the rest holds, at most
   MESH_LINK_STREAMS (two queue pairs: one stream); none where they hold fewer than one stream's two.  Lanes 0 and 1
   carry nothing.  A link's streams past its lanes' are never live.  The plan of fewer queue pairs is a subset of the
   plan of more */
static inline uint32_t link_plan(uint32_t budget){
  if(budget>=MESH_LINK_LANES)return (UINT32_C(1)<<MESH_LINK_LANES)-1;
  const uint32_t first=budget>=3,pairs=(budget-first)/(2*MESH_STREAM_PAIRS);
  const uint32_t streams=pairs<MESH_LINK_STREAMS?pairs:MESH_LINK_STREAMS;
  return first|(((UINT32_C(1)<<(2*MESH_STREAM_PAIRS*streams))-1)<<MESH_LANE_FIRST);
}
static inline uint32_t plan_lanes(uint32_t plan){ return (uint32_t)__builtin_popcount(plan); }
static inline uint32_t plan_streams(uint32_t plan){ return (uint32_t)__builtin_popcount(plan>>MESH_LANE_FIRST)/(2*MESH_STREAM_PAIRS); }
_Static_assert(MESH_LINK_LANES<32,"a bit a lane");

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
/* A queue pair's SEND gate: the progress thread's frames posted and retired (each request carries its lane
   and the frames posted through it, mod 2^48, which its completion returns), the queue's capacity in frames, and the
   census's requests and completions */
#define MESH_GATE_MASK ((UINT64_C(1)<<48)-1)
struct send_gate {
  _Alignas(64) struct frames posted,retired;
  uint64_t requests,completions;
  struct frames capacity;
};
/* whether the gate holds `f` frames more than it has posted and not retired */
static inline int gate_holds(const struct send_gate *gate,struct frames f){
  return ((gate->posted.n-gate->retired.n)&MESH_GATE_MASK)+f.n<=gate->capacity.n;
}
/* `f` frames posted through the gate on lane k: the request's id */
static inline uint64_t gate_post(struct send_gate *gate,uint32_t k,struct frames f){
  gate->posted.n+=f.n;
  return ((uint64_t)k<<48)|(gate->posted.n&MESH_GATE_MASK);
}
/* a request's completion: the frames posted through it retired */
static inline void gate_retire(struct send_gate *gate,uint64_t id){ gate->retired.n=id&MESH_GATE_MASK;gate->completions++; }
/* design/transport.md section 14: a data lane's progress over its pairing, its port's progress thread's alone, in the
   lane's units: the SENDs it has posted (`sent`) and those completed (`done`); the receives posted (`posted`) and landed
   (`got`), at most `window` receives ahead (the fewer frames of the lane's share and its receive queue, over a unit's
   frames: mesh-link.c lane_reset).  A lane's units outstanding one way are at most a ring of MESH_LANE_UNITS rows (the
   window, and the SEND gate's capacity, bounded by it: mesh-link.c lane_reset, link_connect), so the ring holds every
   one's row (mesh.h struct mesh_posts) */
struct lane_state { uint64_t sent,done,posted,got; uint32_t window; };
/* design/transport.md section 14: an entry a progress thread has taken from a holding's list, its own copy, typed: its
   rows (`row`, `pages`), its `base`, its word (`word`: a send entry's naming none the holding's `ready`; NULL for a
   landing entry naming none, or one not whole), and the index of its first page over the holding (`page`); `whole` 0
   where it is not the holder's to name (rows past the page table, or its word not one of the region): its pages go as
   duds, or land in the spare page */
struct taken { struct mesh_row row; struct mesh_pages pages,base,page; const _Atomic uint64_t *word; uint32_t whole; };
/* design/transport.md section 14, "A stream over several cables": cable c's pages of a piece (an entry) of `pages`
   pages among `cables`, from spread_at(c) to spread_at(c + 1): every piece cut evenly, cable 0 taking the larger share
   and so every piece's first page (a piece of one page whole on it).  Both ends cut alike: a send entry and the landing
   entry it fills are of the same pages */
_Static_assert(MESH_PORTS<=sizeof(((struct mesh_stream *)0)->member),"a group's cables, a port each at most");
static inline uint64_t spread_at(uint64_t pages,uint32_t cables,uint32_t c){ return (c*pages+cables-1)/cables; }
/* a walk's cursor over its entries taken: the entry `at`, and one cable's pages in the entries before it (`before`) */
struct cursor { uint64_t at,before; };
/* design/transport.md section 14: a cable of a holding's stream as a walk reads it: its region stream (`x`, whose
   `posts` its process tells), its two lanes' units over the pairing as the holding began (`base`, alike at both ends:
   nothing is outstanding between holdings) and its cursor at its first page not completed (mesh-link.c walk_through).
   A cable's pages go on its lanes in turn: its page g over the pairing is unit g div 2 of its lane g mod 2 */
struct cable { struct mesh_stream *x; uint64_t base[2]; struct cursor done; };
/* design/transport.md section 14: a holding's walk of one of its lists, its port's progress thread's: a ring of
   MESH_STREAM_ENTRIES entries taken (`taken`) of which `done` have all gone (or landed), `timed` given a time, the
   list's pages so far (`pages`), the count last stored (`count`: `sent` or `landed`, in pages), its stream's cables
   (`cables` of them, fixed as the holding begins: mesh-link.c walk_begin), this end's among them (`me`) and its cursor
   at this end's next page to post (`next`) */
struct walk { struct taken *ring; uint64_t taken,done,timed; struct mesh_pages pages,count;
  uint32_t cables,me; struct cursor next; struct cable cable[MESH_PORTS]; };
/* design/transport.md section 14: a holding's state on its stream at its port's progress thread: none (`HOLD_NONE`);
   served (`HOLD_SERVE`); ended (its owner word given, or its client retired), until the next pass has taken its lanes'
   completions (`HOLD_END`); ended with units outstanding on the stream's lanes, the stream not live and nothing served
   on it again in the pairing (`HOLD_GONE`) */
enum { HOLD_NONE, HOLD_SERVE, HOLD_END, HOLD_GONE };
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
/* the wire's magic: what marks a datagram or a control stream's header as the mesh's, so a reader finds a message's
   bounds.  It is the value version 131 sends, fixed: no version enters it, so bridges of any version pair (the operator,
   2026-10-21: "rejection filter removal. categorical. all of them without sanction in rubric"; "even this fake limit
   too must go away").  A message states its sender's MESH_VERSION in `version` (0 from a bridge before it), which no
   reader tests */
#define XMAGIC (0x4d595048u+131u+(1u<<12))
/* design/transport.md section 14: a port's link lanes (MESH_LINK_LANES), its queue pairs made once and connected once
   in its child's life ("A process a port": `lanes` once the child was asked for them, the parent's; a link lost after it
   pairs again with the port's next child), on the two completion queues its device's context holds, each lane's share of them (`share`
   frames, a request one frame at least: no lane overruns its queues), the link block (`page`, its first arena page, and
   `pages`, its arena pages: mesh-link.c link_pages), and each lane's queue pair number (`qpn`, by which
   a completion names its lane).  Its state (LINK_DOWN; made and told, LINK_MADE; connecting; up), its job on the port's
   queue (`job`, dispatched at `job_at`: mesh-link.c ports_stalled), the
   pairing it belongs to (`pairing`, moved at its loss and at a halt, mesh-link.c port_halt, so a job of a pairing gone
   makes no call and is dropped as it ends, B10, B9b), the peer's lanes (`peer`, heard in its LINK: `heard`); the lanes
   the progress thread serves (`served`, a bit a lane, the event loop's to set and to clear, a thread's to clear where
   it found its lane failed, every lane at a failed completion, mesh-link.c link_failed, and the termination's queue's,
   B9b), and the lanes a progress thread found failed, or every lane of a link none of whose streams is live
   (`broken`: the event loop loses the link on reading it), and the code the loss is booked with where it is not a
   failed lane's EIO (`lost`: LINK_STREAMS_ENDED, no stream live, the child's to set and to tell, the parent's to book,
   design/rubrics.md M10).  The
   lanes its pairing makes (`plan`, a bit a lane: link_plan of the fewer queue pairs its two ends hold, set as the
   parent hears the peer's HELLO and told the child with the make; its lanes left out have no queue pair and their
   `qpn` UINT32_MAX, no queue pair's number).  Each
   lane's SEND gate (`gate`: the frames posted and retired; its capacity, set as the lane connects), each data lane's
   progress (`lane`, struct lane_state), the arena page of the streams' lists and time rings (`lists`, leased as the
   port first pairs, MESH_ABSENT none: mesh.h struct mesh_stream).  Each stream's holding at the progress thread: the
   owner it serves (`held`), its state (`hold`), its group's first stream (`first`) and that stream's holdings begun as
   this cable last served or joined it (`begun`), its own `lead` as it last read it, its send and landing walks
   (`sending`, `landing`) and their time rings (`posted_at`, `landed_at`) */
enum { LINK_DOWN, LINK_MADE, LINK_CONNECTING, LINK_UP };
enum { LINK_JOB_MAKE=1, LINK_JOB_CONNECT };
struct link {
  struct mesh_verbs provider;
  int state,job,result,heard,lanes;
  _Atomic int done,lost;
  _Atomic uint32_t served,broken,pairing;
  uint32_t plan,peer[MESH_LINK_LANES][2],qpn[MESH_LINK_LANES],page,pages,lists,job_pairing,psn;
  struct frames share;
  struct send_gate gate[MESH_LINK_LANES];
  struct lane_state lane[MESH_LINK_LANES];
  uint64_t job_at;
  uint64_t held[MESH_LINK_STREAMS],begun[MESH_LINK_STREAMS];
  uint32_t hold[MESH_LINK_STREAMS],lead[MESH_LINK_STREAMS];
  struct mesh_stream *first[MESH_LINK_STREAMS];
  struct walk sending[MESH_LINK_STREAMS];
  _Atomic uint64_t *posted_at[MESH_LINK_STREAMS];
  uint64_t wire[MESH_LINK_STREAMS];
  struct walk landing[MESH_LINK_STREAMS];
  _Atomic uint64_t *landed_at[MESH_LINK_STREAMS];
};
/* design/transport.md section 14, "A process a port": the messages between the bridge's parent and a port's child on
   their socket pair (a stream, AF_UNIX having no SOCK_SEQPACKET here: each a whole struct child_message).  The
   parent's: open the device (CHILD_OPEN), make the link's lanes (CHILD_MAKE, naming the link's pages, its spare page
   among them, which the parent may lease after the child's spawn), connect them to the peer's (CHILD_CONNECT: the peer's lanes, address, node
   and the link's bandwidth), serve no lane again (CHILD_HALT), end (CHILD_END).  The child's, each naming the pairing it
   was asked for: the device opened or not (CHILD_OPENED: its address and bandwidth, and in `lanes` the queue pairs the
   device holds for a link), the lanes made or not (CHILD_MAKE names the lanes of the plan in `lanes`; CHILD_MADE:
   each lane's queue pair number, the first sequence number and each lane's share), connected or not (CHILD_CONNECTED),
   and the lanes a progress thread found failed, or every lane of a link none of whose streams is live (CHILD_BROKEN,
   naming its connect's pairing, `result` the loss's code where it is not a failed lane: LINK_STREAMS_ENDED), a fault
   absorbed on the
   provider's doorbell (CHILD_FAULTED), a guard not held within MESH_PLATFORM_NS (CHILD_UNGUARDED), its port's peer
   interface terminating (CHILD_TERMINATED, as the child heard it: B9b) */
/* design/transport.md section 14, "A process a port": the version of the contract between the bridge and its children,
   declared: their command line, `mesh-flow --child VERSION DEVICE INDEX ONES` (its first three words the same
   for every version, so a bridge finds a child of any version that holds a device: child_elsewhere), and struct
   child_message.  It changes only where that contract does; a build of the same contract is the same child, whatever
   its bytes (the operator, 2026-10-21) */
#define MESH_CHILD_VERSION 2u
/* the code a link's loss is booked with where no stream of it is live, each holding on it having ended with units
   outstanding (mesh-link.c stream_end): the ends of its clients' holdings, not a failure of its lanes (EIO) or its
   hardware (design/rubrics.md M10: a fact named as what it is), told the parent in CHILD_BROKEN's `result` (version
   2 of the children's contract) */
#define LINK_STREAMS_ENDED ECONNABORTED
enum { CHILD_OPEN=1, CHILD_MAKE, CHILD_CONNECT, CHILD_HALT, CHILD_END,
       CHILD_OPENED, CHILD_MADE, CHILD_CONNECTED, CHILD_BROKEN, CHILD_FAULTED, CHILD_UNGUARDED, CHILD_TERMINATED };
struct child_message {
  uint32_t kind,pairing;
  int32_t result,refused;
  uint32_t qpn[MESH_LINK_LANES],peer[MESH_LINK_LANES][2],psn,share,broken,node,page,lanes;
  uint16_t lid,padding;
  uint8_t gid[16];
  uint64_t bandwidth;
};
#define TAG_CHILD (UINT64_C(1)<<58)
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
  int control,job,result,refused,threads;
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
  _Atomic uint64_t epoch,seen[1];
  _Atomic(pthread_t) thread[2];
  _Atomic uint64_t guard,guarded[1];
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
extern struct known known[MESH_KNOWN];
extern uint64_t owners[MESH_CLIENTS];
extern uint32_t ones;
/* the region's arena pages the bridge holds itself (the map, the links' pages and rings, the duds' block): its
   record's `region` is the rest, what its clients may hold */
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
uint32_t record_reach(struct mesh_reach *reach,uint32_t room);
void record_retell(void);
void map_adopt(void);
struct mesh_record *record_held(uint32_t node);
uint64_t health_due(void);
void health_turn(void);
void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes);
void port_lost(struct port *port,int64_t code);
void clients_turn(int liveness);

#endif
