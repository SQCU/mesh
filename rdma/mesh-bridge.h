/* design/bridge-and-ipc.md#files: the bridge's own declarations, shared by its two translation units, mesh-link.c (the
   ports, links and lanes, the progress threads and the data path, the event loop and the stop) and mesh-program.c
   (programs on their communicators' leased lanes, the stripes' requests, records and services): the types, the state
   both read (defined in mesh-link.c) and the functions each calls of the other's */
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
#include <IOKit/IOKitLib.h>
#include <IOKit/ps/IOPowerSources.h>
#include <IOKit/ps/IOPSKeys.h>
#include <poll.h>

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
   `release` where it is a SEND cell's ready word, the index hand-off, design/bridge-and-ipc.md#progress), its lane's
   queue pair, the frames it holds posted, its invocation and ring (its program's queue), and its flow (`link`: the one
   whose ring holds it, the lane's holder while its receives are posted, design/bridge-and-ipc.md#lanes). */
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
/* A queue pair's SEND gate: the send thread's frames posted and retired (each request carries its queue
   and the frames posted through it, which its completion returns), the queue's capacity, and the census's
   requests and completions; and the event loop's alone, `retired` as its turn last read it and when it last
   found the gate empty or `retired` moved (`seen`, `held_at`: design/recovery.md#five-seconds, mesh-link.c
   gates_turn). */
#define MESH_GATE_MASK ((UINT64_C(1)<<48)-1)
struct send_gate {
  _Alignas(64) uint64_t posted,retired,requests,completions;
  uint32_t capacity;
  uint64_t seen,held_at;
};
/* A SEND stream's progress: its current cell, the next request of that cell's chain and the requests left. */
struct send_stream { struct mesh_send *cell,*first; struct ibv_send_wr *next; uint32_t remaining; uint64_t cycle; };
/* design/bridge-and-ipc.md#ledger (Q5P's M27 trace, mesh git 5702ea5): a flow's ledger, kept where the bridge's
   configuration states `ledger` true.  Each SEND cell its send thread first saw released (`cell`, its stream's `cycle`)
   with when it was seen, its first SEND posted, its last posted and that one's completion taken (an exclusive flow's:
   `mark` its gate's frames once it posted), and each record landed (its index, its cycle) with when, every time the
   bridge's clock_ns; at most MESH_LEDGER_EVENTS of each, allocated as the flow's sides are prepared, the progress
   threads only storing into them.  `current` is each queue's event of the cell being posted, `pending` and `last` its
   oldest event whose completion is not yet taken and its newest (`next` links a queue's events).  The names of the
   cells (`cell`: queue, invocation, position among the invocation's transfers of the queue, binding, chunks, and the
   transfer's bytes, by which a reader takes the wire's time out of a crossing's) and of the
   records (`record`, by index: the same, and the chunk) are kept as the sides are prepared, for ledger_write alone */
#define MESH_LEDGER_EVENTS (UINT32_C(1)<<16)
#define MESH_LEDGER_LINES 2048u
#define LEDGER_NONE UINT32_MAX
struct ledger_send { uint64_t cell,mark,seen,first,posted,completed; uint32_t cycle,next; };
struct ledger_landing { uint32_t record,cycle; uint64_t landed; };
struct ledger_cell { uint64_t cell,bytes; uint32_t queue,invocation,position,binding,chunks; };
struct ledger_record { uint32_t position,binding,chunk,chunks; };
struct ledger;
struct ledger_entry { struct ledger *ledger; uint32_t event,done; };
struct ledger {
  uint32_t id,queues,invocations,cyclic,sends,send_room,cells,cut_sends,cut_landings,sorted;
  uint32_t *current,*pending,*last;
  struct ledger_send *send;
  struct ledger_cell *cell;
  uint64_t started;
  _Alignas(64) uint32_t landings,landing_room,records;
  struct ledger_landing *landing;
  struct ledger_record *record;
};

/* design/bridge-and-ipc.md#protocol: a message on a port's control socket (the bridge's wire: HELLO, FAREWELL, RECORD,
   LINK, and the channels' kinds), `bytes` of what follows it: a RECORD's record, a channel's data.  A port's LINK tells
   its link lanes' queue pairs (#lanes: every lane, each one's number and first sequence number in `pair`, and in `code`
   the largest stripe piece this end takes); a channel's names its key, its two end nodes (`from` its origin, `to` its
   destination), a count and a code (#requests) */
enum { HELLO=1, FAREWELL, RECORD, LINK, CHANNEL_OPEN, CHANNEL_CONFIRM, CHANNEL_REFUSE, CHANNEL_DATA, CHANNEL_WINDOW, CHANNEL_SHUT,
       CHANNEL_CLOSE };
struct message { uint32_t magic,kind,node,count; uint64_t key,boot; int64_t code; uint32_t block,port,from,to,bytes;
  uint16_t lid,padding; uint8_t gid[16]; uint32_t pair[MESH_LINK_LANES][2]; };
/* the wire's magic: its base, MESH_VERSION and the wire's revision (1<<12, past any base and version's: every link heard
   within MESH_PLATFORM_NS or concluded, design/recovery.md#five-seconds B1, B2, B3, which a bridge of the same version
   without them breaks), so bridges of different versions or revisions drop each other's hellos and never pair */
#define XMAGIC (0x4d595048u+MESH_VERSION+(1u<<12))
/* design/bridge-and-ipc.md#cables: the cables a program toward one node spans, at most the node's ports */
#define MESH_STRIPES MESH_PORTS
/* design/bridge-and-ipc.md#flows: the flows a port's table holds, by their index (a flow's index and the index's generation
   name it on the port) */
#define MESH_FLOWS 256
/* design/bridge-and-ipc.md#flows: a client program's destination on one port: its lists each way toward the far node
   (its client's slot `owner`, its destination `destination`), its `queues` queues (at most two: its communicator's lane
   pair), queue k riding the port's link lane `lane[k]` (the lease its client holds, mesh.h struct hdr `lanes`), the
   program's invocations t with t mod `stripes` = `stripe` (#cables), its index in the port's table and its generation
   (`slot`), and its lanes a bit a lane as the link's `served` holds them (`lanes`).  It holds no queue pair: its rings
   post receives on its lanes' queue pairs, its streams' SENDs go on them through the lanes' gates (struct link), and
   its lanes' completions come to it as their holder.  Its one job (`job`, `done`) runs on its port's queue: its rings
   and cells prepared, then it is listed in the port's table, where the port's two progress threads serve it while it
   is `progressing` and its lanes are served (the receive thread posts its rings' first records, `armed`).  It has
   ended (`ended`) once it failed or its program ended; its stop runs on the port's queue (`stopping`; `stopped` 1 once
   it ran, 2 where a lane it left dirty may still land a SEND in its rows, `held`: the flow, and with it a retired
   client's rows, kept until its link's lanes' `generation` moves past `lane_epoch`, or until the move a window put off
   moved every lane it holds, design/recovery.md#five-seconds B7, B6 (a)), and the event loop frees it.  Where the
   configuration keeps them, its ledger (`ledger`, #ledger) */
struct port;
struct flow {
  struct flow *next;
  uint32_t stream_count,far,queues,stripe,stripes,destination,invocations,lane[2];
  uint32_t slot,generation,lanes,held,lane_epoch;
  _Atomic int progressing,done,stopped;
  _Atomic int64_t failed;
  _Atomic uint32_t domain;
  int job,result,stopping,cyclic,listed,armed,ended,running;
  uint64_t key,owner;
  struct hdr *M; struct port *port;
  int (*post_send)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **);
  int (*post_receive)(struct ibv_qp *,struct ibv_recv_wr *,struct ibv_recv_wr **);
  struct prepared_receive *receive;
  size_t receive_count;
  struct receive_ring *rings;
  struct send_stream *streams;
  struct prepared_send *requests;
  size_t request_count;
  struct mesh_send *publications;
  struct mesh_cancellation *cancel;
  struct ledger *ledger;
};
/* design/bridge-and-ipc.md#progress: a progress thread with nothing moved for MESH_IDLE_NS parks on its port's bell
   (mesh.h struct hdr `bell`, which a client rings as a group of a program on the port starts, mesh-call.h
   mesh_transfers_wake), looking again every MESH_IDLE_POLL_NS while a flow is in its port's table or its link is up,
   else every MESH_PAUSE_NS */
#define MESH_IDLE_NS UINT64_C(100000000)
#define MESH_IDLE_POLL_NS UINT64_C(1000000)
/* design/bridge-and-ipc.md#protocol: the most bytes a port holds unsent toward a peer that takes none */
#define MESH_UNSENT (UINT64_C(64)<<20)
/* design/bridge-and-ipc.md#lanes: a port's link lanes (mesh.h MESH_LINK_LANES), its queue pairs made once at each of
   its pairings, on the two completion queues its device's context holds (made once a context, kept across pairings),
   each lane's share of them (`share` frames, a request one frame at least: no lane overruns its queues), and each
   lane's queue pair number (`qpn`, by which a completion names its lane).  Its state (LINK_DOWN; made and told,
   LINK_MADE; connecting; up), its job on the port's queue (`job`, dispatched at `job_at`: mesh-link.c ports_stalled;
   LINK_JOB_FLUSH the lanes of a link concluded while connected or up moved to the error state at the end of that
   conclusion's quiet window, `flush_at`, 0 for none, and of any termination's, design/recovery.md#five-seconds B6 (b),
   B9b; LINK_JOB_MOVE the lanes `move` of flows' stops whose move to the error state a window or a stuck call put off,
   `deferred`, a bit a lane, the port queue's to set and the event loop's to take, cleared as the link is made, B6 (a),
   B9) and a registration its make refused (`refused`), the pairing it belongs to (`pairing`, moved at each loss and at
   a halt, a peer interface's termination heard or a guard not held, mesh-link.c port_halt, so a job of a pairing gone
   makes no call and is dropped as it ends, B10, B9b), the peer's lanes (`peer`, heard in its LINK: `heard`), its
   largest stripe piece;
   the lanes the progress threads serve (`served`, a bit a lane, the event loop's to set and to clear, a thread's to
   clear where it found its lane failed, every lane at a failed completion, mesh-link.c link_failed, and the
   termination's queue's, B9b), the lanes a progress thread found failed
   (`broken`, the event loop's to take) and those found so here (`dirty`: none is made again while the link is up;
   once no lane of the link is leased here, its control socket is closed and both ends pair it afresh, every lane made
   again: link_turn), and the count of its lanes'
   destructions (`generation`, link_make's and a LINK_JOB_FLUSH's that moved every queue pair to the error state, past
   which no request of an earlier pairing lands: a flow kept for a lane it left dirty is freed then,
   design/recovery.md#five-seconds B7).  Each lane's SEND gate (`gate`:
   the frames posted and retired, the send thread's; its capacity, set as the lane connects) and the flow whose program
   rides it (`holder`: the event loop's to set as the flow is made, the port's queue's to clear as it stops; a lane's
   completions go to its holder, and none is read once it has none).  Lane MESH_LANE_KEEPALIVE's frames
   (design/bridge-and-ipc.md#keepalive, design/recovery.md#five-seconds: one whenever the link is up, every
   `keepalive_ns` while a flow of the port is listed, else every MESH_PLATFORM_NS/5; the send thread's frame in flight
   and its census `alive_posted`, the receive thread's `alive_heard`, which the event loop reads: mesh-link.c port_silent),
   and its pages: the keepalive's block and the stripes' descriptors */
enum { LINK_DOWN, LINK_MADE, LINK_CONNECTING, LINK_UP };
enum { LINK_JOB_MAKE=1, LINK_JOB_CONNECT, LINK_JOB_FLUSH, LINK_JOB_MOVE };
#define LINK_ALIVE_SENT (UINT64_C(3)<<56)
#define LINK_ALIVE_HEARD (UINT64_C(4)<<56)
#define LINK_ALIVE_BYTES 16u
#define LINK_ALIVE_RECEIVES 256u
struct link {
  struct mesh_verbs provider;
  int state,job,result,heard,refused;
  _Atomic int done;
  _Atomic uint32_t served,broken,generation,pairing,deferred;
  uint32_t peer[MESH_LINK_LANES][2],qpn[MESH_LINK_LANES],peer_piece,share,page,pages,job_pairing,dirty,move;
  struct send_gate gate[MESH_LINK_LANES];
  struct flow *_Atomic holder[MESH_LINK_LANES];
  uint64_t alive_at,job_at,flush_at;
  _Atomic uint64_t alive_posted,alive_heard;
  _Atomic int alive_flight;
  int counted;
};
/* design/bridge-dataflow.md#stripes (mesh 8fc27cbe's per-link stripe service, on the resident bridge): a port's stripe
   writes, on its link's lane MESH_LANE_STRIPES, made and made again with the link: each stripe's descriptor (where the
   peer posts its receives, the word its last one stores and the value) and then its pieces, in order on the one lane,
   the peer listening for one descriptor at a time and for the next only once the pieces of the last have their
   receives posted, so its receives are posted in the order the SENDs come (credit flow control holds a SEND until
   then, TN3205).  The piece both ends cut by (`piece`, the lesser of the two ends' largest), the frames each side may
   hold posted (its lane's share), the peer's window (its record's: no request passes it) and the census (requests
   written and landed).  The send thread's: the stripes' pieces posted and not completed (a ring: frames, and for a
   stripe's last its word, value, writer and ring), their frames, the descriptors sent and completed, the stripe being
   cut (`out`) and the ring it looks at next; the receive thread's: the descriptors landed, by their slot in arrival
   order, the pieces posted, their frames, the stripe being received (`in`).  `flight`: requests taken and not yet
   completed or failed, which hold a retired client's rows (clients_turn).  The event loop's alone: the SENDs completed
   as its turn last read them and when it last found none held or one completed (`seen`, `held_at`:
   design/recovery.md#five-seconds, mesh-link.c gates_turn).  A request is one ring entry's: the bytes at
   `offset` of this node's window to the same offset of the peer's, cut alike at both ends (stripe_length) (the one
   header-driven receive the bridge keeps, for stripes alone: design/bridge-and-ipc.md#lanes) */
#define MESH_DESCRIPTORS 256u
#define MESH_STRIPE_CHUNKS 1024u
#define MESH_STRIPE_PIECE (UINT32_C(4)<<20)
struct mesh_descriptor { _Alignas(64) uint64_t offset; uint64_t bytes,word,value,reserved[4]; };
_Static_assert(sizeof(struct mesh_descriptor)==64,"stripe descriptor");
#define STRIPE_DESCRIPTOR (UINT64_C(1)<<56)
#define STRIPE_DATA (UINT64_C(2)<<56)
#define STRIPE_SLOT ((UINT64_C(1)<<56)-1)
#define LINK_TAG (UINT64_C(0xff)<<56)
struct stripe_piece { uint64_t word,value,owner,ring; uint32_t frames,last; };
struct stripe_cut { struct mesh_descriptor d; uint64_t done,owner,ring; uint32_t slot; int active; };
struct stripes {
  uint32_t piece,listened;
  _Atomic uint64_t window,flight,written,landed;
  _Alignas(128) struct stripe_piece sent[MESH_STRIPE_CHUNKS];
  uint64_t send_head,send_tail,descriptors_sent,descriptors_done;
  uint32_t send_frames,send_capacity,cursor;
  struct stripe_cut out;
  _Alignas(128) struct stripe_piece posted[MESH_STRIPE_CHUNKS];
  uint64_t receive_head,receive_tail,arrived_head,arrived_tail;
  uint32_t receive_frames,receive_capacity,arrived[MESH_DESCRIPTORS];
  struct stripe_cut in;
  uint64_t seen,held_at;
};
/* design/bridge-and-ipc.md#ports: a port, its link, its control socket's queues each way, its verbs work's queue (its
   open's job `job`, dispatched at `job_at`; a flow stop's move to the error state, begun at `moving_at`, 0 while none
   is out; `stuck` once a job or move of the port outlived MESH_PLATFORM_NS and the port was concluded around it, until
   the call returns: design/recovery.md#five-seconds B12, mesh-link.c ports_stalled), the end of its quiet window
   (`quiet_at`,
   MESH_PLATFORM_NS past its last conclusion or a peer interface's termination heard, B9b, moved only later and by any
   thread, before which no port job but an earlier conclusion's flush, struct link
   `flush_at`, is dispatched on it and the bridge does not exit: B9), whether its device is to be opened
   (`opening`: at the start, at its link's coming on, at a hello heard while it is not open, B16, and as the call it was
   concluded around returns, B12; never on a clock), the queue pairs its link lanes hold (`committed`, of the device's
   usable queue pairs), its link lanes, its
   table of flows (`table`, the indices reserved and their generations, the highest reserved plus one), and its
   progress threads (#progress: the epochs each has seen of `epoch`, which a flow leaving the table moves, and whether
   each is parked on the port's bell); its stripe writes (design/bridge-dataflow.md#stripes), the guard of its
   device's verbs objects (`guard`, odd while raised, the word the progress threads park on while it stands,
   design/recovery.md#five-seconds B13, and the value each progress thread last held at: mesh-link.c guard_raise; each
   epoch and guard told wakes the raiser parked on it, port_tell, port_wait),
   whether this end closed its control socket so both ends pair its link afresh (`cycling`, link_turn), and the event
   loop's reading of what its control socket
   heard (design/recovery.md#five-seconds, mesh-link.c port_silent: when it last heard anything, `heard_at`, the link's
   keepalive frames heard as last read, `frames`, and when the last was heard, `frame_at`, 0 until this pairing's first,
   and its frame in flight, `flight` its census as last read and `flight_at` when) */
struct port {
  struct mesh_device device;
  uint32_t index,interface,peer,peer_port,greeted,parted,connecting,writing,open,opening,flows,committed,cycling,stuck;
  int64_t farewell;
  char name[IFNAMSIZ];
  int control,job,result,refused,threads;
  _Atomic int done,closing;
  uint64_t next,bandwidth,heard_at,frames,frame_at,flight,flight_at,job_at;
  _Atomic uint64_t quiet_at,moving_at;
  struct message heard;
  uint16_t lid;
  uint8_t gid[16];
  unsigned char *in,*out;
  size_t held,in_room,out_length,out_sent,out_room;
  dispatch_queue_t work;
  struct link link;
  struct flow *_Atomic table[MESH_FLOWS];
  uint64_t reserved[MESH_FLOWS/64];
  uint32_t generation[MESH_FLOWS];
  _Atomic uint32_t high;
  _Atomic uint64_t epoch,seen[2];
  _Atomic int parked[2];
  pthread_t thread[2];
  struct stripes stripes;
  _Atomic uint64_t guard,guarded[2];
};

/* design/recovery.md#diagnosis: why this bridge stops, the code its FAREWELL carries (a signal's number; its binary
   changed) */
#define MESH_STOP_BINARY INT64_C(-1)
#define MESH_HEALTH_NS UINT64_C(10000000000)

/* design/recovery.md#diagnosis; design/bridge-and-ipc.md#records: the nodes this bridge has heard of, when it last heard
   each one's record (this node's clock), when it lost each, the boot session it last knew, so that a node's return
   names a new boot, and the newest record it ever heard of each (its boot and sequence), held apart from the map:
   RFC 2328 s13's database, so a record the map dropped is never taken, nor sent on, again (a node's later boot is
   always newer); a full table gives the longest unheard of those its place */
struct known { uint32_t node; uint64_t heard,gone,boot,sequence; char session[40]; };
#define MESH_KNOWN 256
/* design/bridge-and-ipc.md#progress: a port's progress thread: passes while anything moves, and with nothing moved for
   MESH_IDLE_NS it parks on its port's bell, looking again every MESH_IDLE_POLL_NS while anything is its to look for
   (port_pass), else every MESH_PAUSE_NS, and at once when the bell rings: a client's group of a program on the port
   starting (mesh-call.h mesh_transfers_wake), a landing that released one of the port's SEND cells while its send thread
   was parked (send_wake), a flow listed or unlisted, the lanes' change, the bridge stopping.  A thread tells it is
   parked, then looks once more before it waits, so a release whose send_wake read it unparked is served by that look,
   and a guard raised meanwhile is held (its raiser rings once and parks: mesh-link.c port_wait) */
#define MESH_IDLE_PASSES 4096u
struct progress { struct port *port; int side; };
/* design/recovery.md#diagnosis: this node's health sampled each turn; a change of its power source, of its charge off
   AC, or of its memory or thermal pressure is logged and sent at once, anything else with its record at most
   MESH_HEALTH_NS later */
/* #admission: the node's memory low (the kernel's pressure at warning or worse, or its available memory under a
   MESH_LOW_SHARE-th of the node's) logged with the largest processes this bridge's user runs, by physical footprint,
   as it begins and every MESH_LOW_NS while it lasts: what any client, a bridge client or not, holds outside the window */
#define MESH_LOW_SHARE 16u
#define MESH_LOW_NS UINT64_C(60000000000)
#define MESH_LOW_LISTED 5

/* design/bridge-and-ipc.md#requests: a channel (RFC 4254 s5) from a client of one node to a service of another, the
   same key (its origin's node and sequence) on every hop. Its two ends, each a socket of this node's (the client's or
   the service's), a port toward the next bridge, or the bridge's own service (`hold`); the bytes one end gives go to
   the other. A port end takes no more than its credit (the window its peer granted), and the bytes a port end gave
   are granted back as they leave for the other end, so a hop holds at most a window each way (RFC 4254 s5.2). A
   socket end is read only while the other end can take what is read */
#define MESH_CHANNELS 256
#define MESH_SERVICES 32
#define MESH_WINDOW (UINT32_C(128)<<10)
#define MESH_PACKET (UINT32_C(32)<<10)
#define MESH_ASKING_NS UINT64_C(5000000000)
#define MESH_HOLD_BYTES 4096u
#define TAG_CHANNEL (UINT64_C(1)<<62)
#define TAG_ASKING (UINT64_C(1)<<61)
#define TAG_SERVING (UINT64_C(1)<<60)
/* design/bridge-dataflow.md#the-loopback-fabric: a loop port's cable listener's event, its index */
#define TAG_LOOP (UINT64_C(1)<<59)
enum { END_SOCKET=1, END_PORT, END_OWN, END_EXEC };
struct end { int kind,fd,reading,writing,ended,told; struct port *port; uint64_t credit; unsigned char *queue; size_t queued,room; };
struct channel { int used,open,exited,status,drained,reported,child; pid_t pid; uint64_t key,deadline; uint32_t origin,destination;
  char service[MESH_SERVICE_BYTES]; struct end end[2]; unsigned char *held; size_t held_bytes; };
struct asking { int fd; uint64_t deadline; size_t got; struct mesh_request request; };
struct serving { int fd; char service[MESH_SERVICE_BYTES],path[MESH_SERVICE_PATH]; };

/* the state both translation units read, defined in mesh-link.c */
extern struct hdr *M;
extern struct wire_mapping wire;
extern struct port ports[MESH_PORTS];
extern uint32_t port_count;
extern struct flow *flows;
extern uint32_t flow_count;
extern int events,listener,hello,network,unreleased,finding;
extern uint64_t now,sequence;
extern char host[MESH_NAME_BYTES];
extern struct mesh_ctx bridge;
extern _Atomic int found;
extern uint64_t memory;
extern struct known known[MESH_KNOWN];
extern int ledger_kept;
extern uint64_t keepalive_ns;
extern uint64_t owners[MESH_CLIENTS],accepted[MESH_CLIENTS];
extern char model[MESH_MODEL_BYTES];
extern int exec_served;

/* the data path's small helpers, inlined into both units: the clock the ledgers read, whether a client's slot is still
   `owner`'s, a span of the window on a port's device, and a receive ring's next records posted */
static inline uint64_t ledger_ns(void){ return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static inline int owned(uint64_t owner){ return owner && atomic_load(&mesh_client_at(M,owner)->owner)==owner; }
static inline struct ibv_sge stripe_span(const struct port *port,uint64_t offset,uint32_t length){
  const uint64_t at=offset<wire.length?offset:0;
  return (struct ibv_sge){.addr=(uintptr_t)port->device.wire+offset,.length=length,.lkey=port->device.regions[at/MESH_REGION]->lkey};
}
/* design/prepared-machine.md#M08 */
/* Posts the ring's next records while its queue has room and the record behind each has landed (its
   transfer's `span` invocations before: the ring's records land in the order posted, so `landed` counts
   them).  Called once at configuration and at each receive completion: the refill is fired by the event. */
/* design/prepared-machine.md#M30 */
/* A cyclic program's ring wraps: its posted count runs on, record posted % count in cycle posted / count,
   at running invocation invocation + cycle * N; the span is held under N and a record with no record of
   its transfer behind it waits on its own previous posting, so a record is posted again only after its
   last posting completed (completions arrive in order).  Called by the port's receive thread alone, once the flow is
   listed (port_pass), so a lane's records are posted by the thread that takes their completions. */
static inline int ring_advance(struct flow *link,struct receive_ring *ring){
  int (*post)(struct ibv_qp *,struct ibv_recv_wr *,struct ibv_recv_wr **)=link->post_receive;
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

/* the functions each translation unit calls of the other's */
void sysctl_text(const char *name,char *out,size_t room);
void health_init(void);
void wake(void);
void stripe_failed(uint64_t ring,uint64_t owner,uint64_t word,int64_t code);
int port_list(struct flow *f);
void port_say(struct port *port,struct message said,const void *payload,uint32_t bytes);
int lanes_leased(const struct port *port,uint64_t owner,const uint32_t *lane,uint32_t queues);
int lanes_take(const struct port *port,uint64_t owner,const uint32_t *lane,uint32_t queues);
uint32_t lanes_release(struct flow *f,int lost);
void flow_fail(struct flow *f,int64_t code,uint32_t domain,uint32_t where,const struct port *lost);
void programs_stop(int64_t code,const struct port *port);
void record_changed(void);
void record_retell(void);
struct mesh_record *record_held(uint32_t node);
void health_turn(void);
void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes);
void stripes_requests_turn(int64_t stopping);
void fd_answer(int fd,int32_t code);
void child_heard(pid_t pid);
void asking_accepted(int fd);
void channel_heard(struct port *port,const struct message *heard,const unsigned char *payload);
void request_event(const struct kevent64_s *e);
void requests_turn(void);
void requests_lost(struct port *port,int64_t code);
int requests_held(const struct port *port);
void requests_init(void);
void requests_stop(void);
void port_lost(struct port *port,int64_t code);
void clients_turn(int liveness);
void flows_turn(void);

#endif
