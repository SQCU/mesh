/* design/bridge-and-ipc.md#files: the bridge's own declarations, shared by its two translation units, mesh-link.c (the
   ports, links and lanes, the progress threads, the event loop and the stop) and mesh-program.c (programs and their
   flows, relays, stripes, streams, records and services): the types, the state both read (defined in mesh-link.c) and
   the functions each calls of the other's */
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

#define MESH_PAIRS 16
/* design/bridge-and-ipc.md#protocol: a message, a flow's keyed by its execution's key, its program's two end nodes
   (`from` the sender's end, `to` the far one) and its stripe (#cables: several of one program may share a port), `failed`
   the node where a failure happened, and `bytes` of what follows it: a READY's program, a RECORD's record.  A port's
   LINK tells its link lanes' queue pairs (#lanes: `count` lanes from lane `from`, each lane's number and first sequence
   number in `pair`, and in `code` the largest stripe piece this end takes) */
enum { HELLO=1, READY, PAIRS, POSTED, END, FAIL, FAREWELL, RECORD, LINK, CHANNEL_OPEN, CHANNEL_CONFIRM, CHANNEL_REFUSE, CHANNEL_DATA,
       CHANNEL_WINDOW, CHANNEL_SHUT, CHANNEL_CLOSE, STREAM };
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
_Static_assert(MESH_LINK_LANES<=MESH_PAIRS,"a LINK names every lane");
/* design/bridge-and-ipc.md#flows: the flows a port's table holds, by their index (a flow's index and the index's generation
   name it on the port) */
#define MESH_FLOWS 256
/* design/bridge-and-ipc.md#flows: a program on one port, over its link toward the program's far end (`near` its other
   end): its lists each way, a slot's destination (its client's, or on a relay a lease of the bridge's own; a flow held
   for its client, `hold`, has none until the client's program takes it), its `queues` queues of `slots` rings each,
   the program's invocations t with t mod `stripes` = `stripe` (#cables), and on a relay its partner, the route's other
   flow.  It holds a queue pair a ring (`qps`, its own completion queues), its index in the port's table and its
   generation (`slot`).  Its verbs work runs on its port's queue, one job at a time (`job`, `done`); the event loop
   alone changes the rest, and a side whose lists arrive while a job runs (`pending`) between jobs.  The port's two
   progress threads serve it while it is in the port's table and `progressing`.
   A held flow keeps the digest its READY named (`digest`), which the program that takes it must name too (#protocol),
   and where the configuration keeps them its ledger (`ledger`, #ledger) */
enum { READIED, PAIRING, POSTING, RUNNING, FAILED };
enum { HEARD_READY=1, HEARD_PAIRS=2, HEARD_POSTED=4 };
enum { JOB_PAIR=1, JOB_RECEIVE, JOB_RUN, JOB_CHANGE };
struct lease { struct mesh_ctx context; uint32_t users; uint64_t shape; };
struct side { uint64_t owner; uint32_t destination; struct lease *lease; };
struct port;
struct flow {
  struct flow *next,*partner;
  uint32_t stream_count,near,far,queues,stripe,stripes,heard,configured,destination,invocations;
  uint32_t slot,generation;
  _Atomic int progressing,done,stopped;
  _Atomic int64_t failed;
  _Atomic uint32_t domain;
  int state,job,result,stopping,hold,qps,cyclic,told,listed;
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
   each lane's share of them (`share` frames, a request one frame at least: no lane overruns its queues).  Its state
   (LINK_DOWN; made and told, LINK_MADE; connecting; up), the pairing it belongs to (`pairing`, moved at each loss, so a
   job of a pairing gone is dropped as it ends), the peer's lanes (`peer`, heard in its LINK: `heard`), its largest
   stripe piece; the lanes the progress threads serve (`served`, a bit a lane, the event loop's to set and to clear, a
   thread's to clear where it found its lane failed), the lanes a progress thread found failed (`broken`, the event
   loop's to take).  A lane made again while the link is up is a round the link's lower node leads (DESIGN.md L2;
   metal-microbench output_data/structural-20261018/wp1-remake-model3.py checks it): the leader's lanes to make again
   (`dirty`: its own failed or dirty lanes, and those the higher node reported), its open rounds (`round`: made again and
   told, each lane's round in `seq`), and those whose acknowledgement came (`acked`: the higher node's new numbers in
   `peer`); the higher node's lanes reported (`dirty`) and those the leader made again (`pending`, its round in `seq`),
   each made again, joined and acknowledged at once.  `joining`: the lanes of the running job.  Lane
   MESH_LANE_KEEPALIVE's frames (design/bridge-and-ipc.md#keepalive: one every `keepalive_ns` while any lane is leased,
   its census `alive_posted` and `alive_heard`), and its pages: the keepalive's block and the stripes' descriptors */
enum { LINK_DOWN, LINK_MADE, LINK_CONNECTING, LINK_UP };
enum { LINK_JOB_MAKE=1, LINK_JOB_CONNECT, LINK_JOB_REMAKE, LINK_JOB_JOIN, LINK_JOB_REJOIN };
#define LINK_ALIVE_SENT (UINT64_C(3)<<56)
#define LINK_ALIVE_HEARD (UINT64_C(4)<<56)
#define LINK_ALIVE_BYTES 16u
#define LINK_ALIVE_RECEIVES 256u
struct link {
  struct mesh_verbs provider;
  int state,job,result,heard;
  _Atomic int done;
  _Atomic uint32_t served,broken;
  uint32_t peer[MESH_LINK_LANES][2],seq[MESH_LINK_LANES],peer_piece,share,page,pages,pairing,job_pairing,dirty,round,acked,pending,joining;
  uint64_t alive_at,alive_posted,alive_heard;
  int alive_flight,counted;
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
   completed or failed, which hold a retired client's rows (clients_turn).  A request is one ring entry's: the bytes at
   `offset` of this node's window to the same offset of the peer's, cut alike at both ends (stripe_length) (the one
   header-driven receive the bridge keeps, for stripes alone: design/bridge-and-ipc.md#lanes) */
#define MESH_DESCRIPTORS 256u
#define MESH_STRIPE_CHUNKS 1024u
#define MESH_STRIPE_PIECE (UINT32_C(4)<<20)
#define MESH_FLUSH_POLLS 65536u
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
};
/* design/bridge-dataflow.md#streams (mesh 554a3793^ rdma/mesh-verbs-stream.c, inside the bridge, the one opener of
   verbs): a raw one-way stream a client asked its bridge for (mesh-dataflow.h MESH_REQUEST_STREAM), `bytes` in pages of
   one block from a lease of the sender's window to one of the receiver's, over one UC queue pair each end made for it
   and nothing else of the bridge's (no cell, ring, record or landing word), its SENDs and receives posted by the port's
   progress threads in chains of at most `window` pages in flight (mesh-verbs-stream.c's 1,023), every word checked at
   the receiver; the STREAM message's phases (`count`): an offer (the sender's queue pair, `code` the bytes), posted
   (the receiver's queue pair, its first receives posted), the result (the receiver's numbers), 0 a failure (`code`).
   Its state (STREAMS_DOWN, filling or zeroing its lease, made, connecting, running, checking, closing), whether this
   end sends, the requester's socket (the sender's end), the lease (`page`, `pages`), the pages, the times and counts
   each end keeps (`own`) and the receiver's as its result told them (`told`) */
#define MESH_STREAM_MAGIC UINT64_C(0x6d65736870616765)
#define MESH_STREAM_WINDOW 1023u
enum { STREAM_FAILED, STREAM_OFFER, STREAM_POSTED, STREAM_RESULT };
enum { STREAMS_DOWN, STREAMS_FILLING, STREAMS_MAKING, STREAMS_MADE, STREAMS_CONNECTING, STREAMS_RUNNING, STREAMS_CHECKING, STREAMS_CLOSING };
struct stream_result { uint64_t bytes,pages,page_bytes,window,completed,mismatches,correct,misordered,first_count,low_count,high_count,
  begin_ns,first_ns,low_ns,high_ns,last_ns,posts,polls,empty_polls; };
struct stream {
  struct mesh_verbs provider;
  int state,job,result,sending,committed,fd;
  _Atomic int up,done,finished;
  _Atomic int64_t failed;
  uint32_t peer[1][2],page,pages,count,page_bytes,window;
  uint64_t key,bytes,deadline,flushed;
  struct ibv_sge *spans;
  struct ibv_send_wr *sends;
  struct ibv_recv_wr *receives;
  struct ibv_wc *completions;
  uint64_t posted,completed;
  struct stream_result own,told;
};
/* design/bridge-and-ipc.md#ports: a port, its link, its control socket's queues each way, its verbs work's queue, whether
   its device is to be opened (`opening`: at the start and at its link's coming on, never on a clock), the queue pairs its
   link lanes and its stream hold (`committed`, of the device's usable queue pairs), its link lanes, its table of flows
   (`table`, the indices reserved and their generations, the highest reserved plus one), and its progress threads
   (#progress: the epochs each has seen of `epoch`, which a flow leaving the table moves, and whether each is parked on
   the port's bell); its stripe writes (design/bridge-dataflow.md#stripes) and the guard of its device's verbs objects
   (`guard`, odd while raised, and the value each progress thread last held at: port_guard) */
struct port {
  struct mesh_device device;
  uint32_t index,interface,peer,peer_port,greeted,parted,connecting,writing,open,opening,flows,committed;
  int64_t farewell;
  char name[IFNAMSIZ];
  int control,job,result,refused,threads;
  _Atomic int done,closing;
  uint64_t next,bandwidth;
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
  struct stream stream;
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
   parked, then looks once more before it waits, so a release whose send_wake read it unparked is served by that look */
#define MESH_IDLE_PASSES 4096u
struct progress { struct port *port; int side; };

/* design/bridge-and-ipc.md#routes: a client program awaiting the READYs of its routed peer, the pair's lower node, whose
   route both ends take: its `stripes` once the first came (#cables), the stripes it has taken (a bit each), and its
   `deadline`: the peer window from `created` where this end's map routes to the peer (`routed`), else the pairing
   window from when the peer was last reachable (program_await) */
struct await { struct await *next; uint64_t key,owner,created,taken,deadline; uint32_t destination,far,stripes,routed; };
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
extern uint32_t flow_count,await_count;
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

/* the data path's small helpers, inlined into both units: the clock the ledgers and streams read, whether a client's slot
   is still `owner`'s, a span of the window on a port's device, and a receive ring's next records posted */
static inline uint64_t ledger_ns(void){ return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static inline int owned(uint64_t owner){ return owner && atomic_load(&mesh_client_at(M,owner)->owner)==owner; }
static inline struct ibv_sge stripe_span(const struct port *port,uint64_t offset,uint32_t length){
  const uint64_t at=offset<wire.length?offset:0;
  return (struct ibv_sge){.addr=(uintptr_t)port->device.wire+offset,.length=length,.lkey=port->device.regions[at/MESH_REGION]->lkey};
}
/* design/bridge-dataflow.md#streams: the window offset of a stream's page */
static inline uint64_t stream_offset(const struct port *port,uint64_t page){
  return (uint64_t)port->stream.page*M->pgsz+page*port->stream.page_bytes;
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
static inline int ring_advance(struct flow *link,struct receive_ring *ring){
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

/* the functions each translation unit calls of the other's */
void sysctl_text(const char *name,char *out,size_t room);
void health_init(void);
void wake(void);
void stripe_failed(uint64_t ring,uint64_t owner,uint64_t word,int64_t code);
int stream_post(struct port *port,uint32_t n);
void port_bell(struct port *port);
int port_threads(struct port *port);
int port_list(struct flow *f);
void port_settle(struct port *port);
void port_guard(struct port *port,int raise);
int guard_raise(struct port *port);
void guard_lower(struct port *port,int raised);
void port_quiet(struct flow *f);
void port_unlist(struct flow *f);
void port_pairs(struct port *port);
void port_say(struct port *port,struct message said,const void *payload,uint32_t bytes);
void flow_fail(struct flow *f,int64_t code,uint32_t domain,uint32_t tell,uint32_t where,uint32_t origin);
void program_stale(uint64_t owner,uint32_t node);
void programs_stop(int64_t code);
void flow_tell(struct flow *f);
void flow_step(struct flow *f);
struct flow *flow_on(struct port *port,uint64_t key,uint32_t near,uint32_t far,uint32_t stripe);
void ready_heard(struct port *port,const struct message *heard,const unsigned char *payload);
void record_changed(void);
struct mesh_record *record_held(uint32_t node);
void health_turn(void);
void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes);
void stripes_requests_turn(void);
void fd_answer(int fd,int32_t code);
void child_heard(pid_t pid);
void asking_accepted(int fd);
void channel_heard(struct port *port,const struct message *heard,const unsigned char *payload);
void request_event(const struct kevent64_s *e);
void requests_turn(void);
void requests_lost(struct port *port,int64_t code);
void requests_init(void);
void requests_stop(void);
void streams_fail(struct port *port,int64_t code,int tell);
void stream_heard(struct port *port,const struct message *heard,const unsigned char *payload);
void streams_turn(struct port *port);
void port_lost(struct port *port,int64_t code);
void clients_turn(int liveness);
void flows_turn(void);

#endif
