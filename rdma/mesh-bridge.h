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
/* A SEND stream's progress: its current cell, the next request of that cell's chain and the requests
   left, and the lane its queue rides where its flow shares its port's lanes (design/bridge-and-ipc.md#lanes). */
struct send_stream { struct mesh_send *cell,*first; struct ibv_send_wr *next; uint32_t remaining,lane; uint64_t cycle; };
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
   LANES tells its lanes' queue pairs (#lanes) */
enum { HELLO=1, READY, PAIRS, POSTED, END, FAIL, FAREWELL, RECORD, LANES, CHANNEL_OPEN, CHANNEL_CONFIRM, CHANNEL_REFUSE, CHANNEL_DATA,
       CHANNEL_WINDOW, CHANNEL_SHUT, CHANNEL_CLOSE, STRIPES, STREAM };
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
/* design/bridge-and-ipc.md#keepalive: an exclusive flow's keepalive frames, on a queue pair of their own past its
   data's: their SENDs' and receives' identities (no record's address), the bytes a frame carries, and the receives its
   receive side keeps posted */
#define FLOW_ALIVE_SENT UINT64_MAX
#define FLOW_ALIVE_HEARD UINT64_C(1)
#define FLOW_ALIVE_BYTES 16u
#define FLOW_ALIVE_RECEIVES 256u
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
  uint32_t slot,generation,peer_slot,peer_generation,lane_epoch;
  _Atomic int progressing,receiving,done,stopped;
  _Atomic int64_t failed;
  _Atomic uint32_t domain;
  _Atomic uint64_t lane_frames;
  int state,job,result,stopping,hold,qps,cyclic,told,listed,exclusive,decided;
  int alive,peer_alive,alive_flight;
  uint64_t key,owner,created,deadline,retry,digest,alive_at,alive_posted,alive_heard;
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
   while `up`; #loss: the connections that failed in a row since the device opened or the lanes last came up
   (`refused`) */
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
  uint32_t refused;
  int64_t code;
};
/* design/recovery.md#loss: lanes that failed to connect this many times in a row retire their port's device, which opens
   again (a context that refuses every connection, as after a peer's restart, is not one the retries mend) */
#define MESH_REFUSED_RETIRE 3u
/* design/bridge-dataflow.md#stripes (mesh 8fc27cbe's per-link stripe service, on the resident bridge): a port's stripe
   flow, two UC queue pairs of its own, a control pair carrying each stripe's descriptor (where the peer posts its
   receives, the word its last one stores and the value) and a data pair carrying its bytes, with their two completion
   queues; made where a client's request ring names the port's peer (`wanted`) or the peer made its own, told the peer
   (STRIPES, as LANES tells lanes) and kept while the link is paired.  Its state (STRIPES_DOWN, made and told, up,
   closing), the peer's queue pairs and its largest piece (`peer_piece`), the piece both ends cut by (`piece`, the
   lesser of the two), its pages (MESH_DESCRIPTORS descriptors received, then as many sent), why it last could not be
   made (`code`, cleared once up), when it is made or closed again (`retry`), the peer's window (its record's: no
   request passes it) and the census (requests written and landed, completions a teardown's flush drained).  The send
   thread's: the stripes' pieces posted on the data pair and not completed (a ring: frames, and for a stripe's last its
   word, value, writer and ring), their frames, the descriptors sent and completed, the stripe being cut (`out`) and the
   ring it looks at next; the receive thread's: the descriptors landed, by their slot in arrival order, the pieces
   posted, their frames, the stripe being received (`in`).  `flight`: requests taken and not yet completed or failed,
   which hold a retired client's rows (clients_turn).  A request is one ring entry's: the bytes at `offset` of this
   node's window to the same offset of the peer's, cut alike at both ends (stripe_length), a descriptor and then its
   pieces in order on one flow, so the peer's receives for them are posted in the order its SENDs come, each when its
   descriptor lands (the one header-driven receive the bridge keeps, for stripes alone: design/bridge-and-ipc.md#lanes) */
#define MESH_DESCRIPTORS 256u
#define MESH_STRIPE_CHUNKS 1024u
#define MESH_STRIPE_PIECE (UINT32_C(4)<<20)
#define MESH_FLUSH_POLLS 65536u
struct mesh_descriptor { _Alignas(64) uint64_t offset; uint64_t bytes,word,value,reserved[4]; };
_Static_assert(sizeof(struct mesh_descriptor)==64,"stripe descriptor");
#define STRIPE_DESCRIPTOR (UINT64_C(1)<<56)
#define STRIPE_DATA (UINT64_C(2)<<56)
#define STRIPE_SLOT ((UINT64_C(1)<<56)-1)
struct stripe_piece { uint64_t word,value,owner,ring; uint32_t frames,last; };
struct stripe_cut { struct mesh_descriptor d; uint64_t done,owner,ring; uint32_t slot; int active; };
enum { STRIPES_DOWN, STRIPES_MAKING, STRIPES_MADE, STRIPES_CONNECTING, STRIPES_UP, STRIPES_CLOSING };
struct stripes {
  struct mesh_verbs provider;
  int state,job,result,heard,committed,wanted;
  _Atomic int up,done;
  _Atomic int64_t failed;
  uint32_t peer[2][2],peer_piece,piece,page,pages,flushed;
  uint64_t retry;
  int64_t code;
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
/* design/bridge-and-ipc.md#ports: a port, its link, its control socket's queues each way, its verbs work's queue, a
   refused registration's pause (`backoff`), the queue pairs its flows and lanes hold or will (`committed`, of the
   device's usable queue pairs), its lanes, its table of flows (`table`, the indices reserved and their generations, the
   highest reserved plus one), and its progress threads (#progress: the epochs each has seen of `epoch`, which a flow
   leaving the table moves, and whether each is parked on the port's bell); its stripe flow (design/bridge-dataflow.md
   #stripes) and the guard of its device's verbs objects (`guard`, odd while raised, and the value each progress thread
   last held at: port_guard) */
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
   RFC 2328 s13's database, so a record the map dropped is never taken, nor sent on, again.  An entry unheard for
   MESH_RECORD_AGE_NS whose node the map no longer holds compares with nothing (RFC 2328 s14's MaxAge); a full table
   gives the longest unheard of those its place */
struct known { uint32_t node; uint64_t heard,gone,boot,sequence; char session[40]; };
#define MESH_KNOWN 256
#define MESH_RECORD_AGE_NS (UINT64_C(6)*MESH_HEALTH_NS)
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

/* the functions each translation unit calls of the other's */
uint64_t ledger_ns(void);
void sysctl_text(const char *name,char *out,size_t room);
void health_init(void);
void wake(void);
void flow_lost(struct flow *flow,int64_t code,uint32_t domain);
uint32_t ring_lane(uint32_t q,uint32_t slots);
void ledger_seen(struct ledger *l,const struct mesh_send *cell,uint64_t cycle);
void ledger_stamp(struct ledger *l,uint32_t e,int done,uint64_t mark,uint64_t at);
void ledger_landed(struct flow *link,const struct prepared_receive *record);
void send_wake(_Atomic uint64_t *ready);
int flow_send(struct flow *link);
int flow_receive(struct flow *link);
uint64_t lane_discard(const struct port *port,uint32_t lane);
struct ibv_sge lane_span(const struct port *port,uint64_t offset,uint64_t length);
struct ibv_qp *lane_pair(const struct flow *link,uint32_t lane);
int stripes_send(struct port *port);
int stripes_receive(struct port *port);
int stream_pass(struct port *port);
void port_bell(struct port *port);
int port_threads(struct port *port);
int port_list(struct flow *f,int progressing,int receiving);
void port_settle(struct port *port);
void port_guard(struct port *port,int raise);
int guard_raise(struct port *port,int always);
void guard_lower(struct port *port,int raised);
void port_quiet(struct flow *f);
void port_unlist(struct flow *f);
void port_pairs(struct port *port);
void port_say(struct port *port,struct message said,const void *payload,uint32_t bytes);
void flow_fail(struct flow *f,int64_t code,uint32_t domain,uint32_t tell,uint32_t where,uint32_t origin);
void program_stale(uint64_t owner,uint32_t node);
void flow_tell(struct flow *f);
int flow_mode(struct flow *f,int exclusive);
void flow_step(struct flow *f);
struct flow *flow_on(struct port *port,uint64_t key,uint32_t near,uint32_t far,uint32_t stripe);
void ready_heard(struct port *port,const struct message *heard,const unsigned char *payload);
void record_changed(void);
void health_turn(void);
void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes);
void stripes_turn(struct port *port);
void stripes_heard(struct port *port,const struct message *heard);
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
void port_lost(struct port *port,int64_t code,int retire);
void clients_turn(int liveness);
void flows_turn(void);

#endif
