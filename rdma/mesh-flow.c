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
  uint32_t last;
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
  uint32_t outstanding,capacity,span,depth;
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
   left, the lane its queue rides where its flow shares its port's lanes (design/bridge-and-ipc.md#lanes), its queue
   pair's index (`queue`) and the cells it posted whole (`sent`, the send thread's alone: the census). */
struct send_stream { struct mesh_send *cell,*first; struct ibv_send_wr *next; uint32_t remaining,lane,queue; uint64_t cycle,sent; };
/* design/bridge-and-ipc.md#ledger (Q5P's M27 trace, mesh git 5702ea5): a flow's ledger, kept where the bridge's
   configuration states `ledger` true.  Each SEND cell its send thread first saw released (`cell`, its stream's `cycle`)
   with when it was seen, its first SEND posted, its last posted and that one's completion taken (an exclusive flow's:
   `mark` its gate's frames once it posted), and each record landed (its index, its cycle) with when, every time the
   bridge's ledger_ns; at most MESH_LEDGER_EVENTS of each, allocated as the flow's sides are prepared, the progress
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
enum { HELLO=1, READY, PAIRS, POSTED, FAIL, FAREWELL, RECORD, LANES, CHANNEL_OPEN, CHANNEL_CONFIRM, CHANNEL_REFUSE, CHANNEL_DATA,
       CHANNEL_WINDOW, CHANNEL_SHUT, CHANNEL_CLOSE, STRIPES, STREAM };
struct message { uint32_t magic,kind,node,count; uint64_t key; int64_t code; uint32_t block,port,from,to,failed,bytes;
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
/* design/bridge-and-ipc.md#control: a flow's control frames, {kind, channel, count}, 16 bytes: ALIVE (the keepalive,
   #keepalive: nothing to apply), FAIL (the flow failed at node `channel`, the code `count`) and RELEASED (a lane rider
   at a program's end released its flow in order, every cell its program released sent; a relay's flows go once both of
   their ends said so: #routes).  An exclusive flow's ride its own queue pair past its data's (its keepalive pair, mesh
   12f484f8; a page of the window its own: CONTROL_SENDS slots of a frame its SENDs read, then the slots of CONTROL_SLOT
   bytes its CONTROL_RECEIVES receives fill), their identities tagged in the top 16 bits (no record's address reaches
   them); a lane rider's ride its lanes' headers as entries of no chunks (lane_entry: `ring` the kind and channel,
   `bytes` the count) */
enum { CONTROL_ALIVE=1, CONTROL_FAIL, CONTROL_RELEASED };
struct control_frame { uint32_t kind,channel; uint64_t count; };
_Static_assert(sizeof(struct control_frame)==16,"a control frame");
#define CONTROL_SENDS 64u
#define CONTROL_SLOT 64u
#define CONTROL_RECEIVES 192u
#define CONTROL_SENT_TAG (UINT64_C(0xffff)<<48)
#define CONTROL_HEARD_TAG (UINT64_C(0xfffe)<<48)
#define CONTROL_ID (~(UINT64_C(0xffff)<<48))
/* design/bridge-and-ipc.md#control: a flow's `ending` word, which its progress threads read once a pass: its program
   ended in order (FLOW_ENDING: every cell it released posted, then those SENDs' completions taken, `drained`), its
   program muted (FLOW_MUTED: no cell of it posted, program_quiet), its FAIL to be told on its control queue pair
   (FLOW_TELLING: the send thread posts it and, its completion taken, says so, `said`) and its release under way
   (FLOW_RELEASING: a lane rider's last record landing wakes the event loop, flow_release) */
#define FLOW_ENDING 1
#define FLOW_MUTED 2
#define FLOW_TELLING 4
#define FLOW_RELEASING 8
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
   and where the configuration keeps them its ledger (`ledger`, #ledger).  #control: its `ending` word (FLOW_ENDING,
   FLOW_MUTED, FLOW_TELLING) and what its send thread says of it (`drained`: its program's released cells all posted
   and completed; `told`: its FAIL sent), the FAIL the peer sent (`heard_fail`, the code, at node `heard_origin`), a
   relay's RELEASED heard from its end (`released`) and passed on its partner (`passed`); an exclusive flow's control
   queue pair (`alive`, the last of its pairs, and the peer's, `peer_alive`), its page (`control_page`), its frames
   posted and retired and heard, the keepalive's last posting (`alive_at`, #keepalive); a lane rider's gate positions
   past its last SENDs on each lane (`lane_mark`), which its drain waits on; the stop it takes once its FAIL is said
   (`stop_code`, `stop_domain`, `stop_where`), and the port's flows freed when its release was refused (`frees`) */
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
  int alive,peer_alive,connected,flushed,passed;
  uint64_t key,owner,digest,alive_at,alive_posted,alive_heard;
  _Atomic int ending,drained,said,released;
  _Atomic int64_t heard_fail;
  _Atomic uint32_t heard_origin;
  uint32_t control_page,control_receives;
  uint64_t control_posted,control_retired,fail_posted,lane_mark[MESH_LANES];
  int64_t fail_code,stop_code;
  uint32_t fail_origin,stop_domain,stop_where;
  uint64_t frees;
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
   mesh_transfers_wake), looking again every MESH_IDLE_POLL_NS while anything can land for it (a completion queue
   with no completion event: Apple's provider runs no completion channel, mesh-verbs.h pair_up), else waiting on the
   bell alone */
#define MESH_IDLE_NS UINT64_C(100000000)
#define MESH_IDLE_POLL_NS UINT64_C(1000000)
/* design/bridge-and-ipc.md#lanes: a port's lanes: their queue pairs and completion queues, their state (LANES_DOWN,
   made and told, up, closing), the peer's queue pairs and the frames a header's records may hold (`room`), their pages
   (a lane's MESH_HEADERS send headers, two receive headers and its discard block, leased for every port the region has
   at the bridge's start: lanes_reserve), each lane's gate, the next header each way and the gate's frames once each
   send header's SEND posted (`mark`), the receives a header posted and not landed (`records`, the receive thread's),
   why they last could not be made (`code`, logged once a cause), and `generation` moved at each teardown; the progress
   threads read the lanes while `up` */
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
  int64_t code;
};
/* design/bridge-dataflow.md#stripes (mesh 8fc27cbe's per-link stripe service, on the resident bridge): a port's stripe
   flow, two UC queue pairs of its own, a control pair carrying each stripe's descriptor (where the peer posts its
   receives, the word its last one stores and the value) and a data pair carrying its bytes, with their two completion
   queues; made where a client's request ring names the port's peer (`wanted`) or the peer made its own, told the peer
   (STRIPES, as LANES tells lanes) and kept while the link is paired.  Its state (STRIPES_DOWN, made and told, up,
   closing), the peer's queue pairs and its largest piece (`peer_piece`), the piece both ends cut by (`piece`, the
   lesser of the two), its pages (MESH_DESCRIPTORS descriptors received, then as many sent), why it last could not be
   made (`code`, cleared once up or once a request failed by it), the peer's window (its record's: no request passes it)
   and the census (requests written and landed, completions a teardown's flush drained).  The send
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
  uint64_t key,bytes,flushed;
  struct ibv_sge *spans;
  struct ibv_send_wr *sends;
  struct ibv_recv_wr *receives;
  struct ibv_wc *completions;
  uint64_t posted,completed;
  struct stream_result own,told;
};
/* design/bridge-and-ipc.md#ports: a port, its link, its control socket's queues each way, its verbs work's queue,
   whether its next turn opens its device and says HELLO (`wanted`: an event asked it, port_turn) and whether its
   window's registration was refused (`refused`: the next memory-pressure or link event turns it), the queue pairs its
   flows and lanes hold or will (`committed`, of the device's usable queue pairs), its lanes, its table of flows
   (`table`, the indices reserved and their generations, the highest reserved plus one), and its progress threads
   (#progress: the epochs each has seen of `epoch`, which a flow leaving the table moves, and whether each is parked on
   the port's bell); its stripe flow (design/bridge-dataflow.md#stripes) and the guard of its device's verbs objects
   (`guard`, odd while raised, and the value each progress thread last held at: port_guard); #control: the FAILs and
   RELEASEDs of its lane riders, as entries its lanes' send thread takes into lane 0's next header (`tells`, the event
   loop's from `tell_head`, the send thread's to `tell_tail`), and how many of its flows were freed (`frees`: a flow whose
   queue pairs' destruction was refused tries again at the next) */
#define MESH_TELLS 256u
struct port {
  struct mesh_device device;
  uint32_t index,interface,peer,peer_port,greeted,parted,connecting,writing,retiring,open,flows,committed,wanted;
  int64_t farewell;
  char name[IFNAMSIZ];
  int control,job,result,refused,threads;
  _Atomic int done,closing;
  uint64_t bandwidth,frees;
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
  struct lane_entry tells[MESH_TELLS];
  _Atomic uint64_t tell_head,tell_tail;
};
static struct hdr *M;
static struct wire_mapping wire;
static struct port ports[MESH_PORTS];
static uint32_t port_count;
static struct flow *flows;
/* design/bridge-and-ipc.md#region: the flows that carry a program (any not failed or released) and the awaiting
   programs the bridge holds, told in the region (`busy`) */
static uint32_t await_count;
static void busy_tell(void){
  uint64_t busy=await_count;
  for(const struct flow *f=flows;f;f=f->next)busy+=f->state!=FAILED;
  atomic_store_explicit(&M->busy,busy,memory_order_release);
}
static int events=-1,listener=-1,hello=-1,network=-1,unreleased,finding;
static uint64_t sequence;
/* the event loop's timers (EVFILT_TIMER idents: the configuration's HELLO and health intervals) and the memory-pressure
   change a dispatch source tells it (`pressured`) */
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

/* design/bridge-and-ipc.md#ledger: the ledger's clock, nanoseconds (CLOCK_MONOTONIC's are whole microseconds here), as
   Q5P's trace kept (mesh git 5702ea5, M27) */
static uint64_t ledger_ns(void){ return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
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
/* design/algorithm-sources.md#meshresult: a data thread's failure, taken by the event loop (flow_fail): its program's
   cells posted no more, its threads serving it on so its send thread can tell the peer (FLOW_TELLING) */
static void flow_lost(struct flow *flow,int64_t code,uint32_t domain){
  int64_t none=0;
  if(atomic_compare_exchange_strong(&flow->failed,&none,code?code:EIO))atomic_store(&flow->domain,domain);
  atomic_fetch_or_explicit(&flow->ending,FLOW_MUTED,memory_order_release);
  wake();
}
static int owned(uint64_t owner){ return owner && atomic_load(&mesh_client_at(M,owner)->owner)==owner; }
static struct mesh_tx *side_tx(struct flow *f,int d){ return &mesh_client_at(M,f->side[d].owner)->tx[f->side[d].destination]; }
static uint32_t side_channel(struct flow *f,int d,uint32_t k){ return f->side[d].destination*MESH_QUEUES+k; }
static uint32_t side_length(struct flow *f,int d,uint32_t k){ return f->side[d].owner?atomic_load(mesh_order_length(M,f->side[d].owner,side_channel(f,d,k),d)):0; }
static struct mesh_transfer *side_transfers(struct flow *f,int d,uint32_t k){ return mesh_transfers(M,f->side[d].owner,side_channel(f,d,k),d); }
/* design/bridge-and-ipc.md#clients: a status told, and its client's start woken */
static void status_prepared(struct mesh_port_info *status,uint64_t owner){
  atomic_store_explicit(&status->prepared,owner,memory_order_release);
  os_sync_wake_by_address_all(&status->prepared,sizeof status->prepared,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
static void status_failed(struct mesh_port_info *status,uint64_t owner,int64_t code,uint32_t domain,uint32_t where){
  atomic_store_explicit(&status->code,code,memory_order_relaxed);atomic_store_explicit(&status->domain,domain,memory_order_relaxed);
  atomic_store_explicit(&status->failed,where,memory_order_relaxed);atomic_store_explicit(&status->phase,MESH_STOPPED,memory_order_relaxed);
  status_prepared(status,owner);
}
/* the program whose status and cancellation words a flow tells */
static void flow_program(struct flow *f,uint64_t owner,uint32_t destination){
  uint64_t cancel=mesh_client_at(M,owner)->tx[destination].cancel;
  f->owner=owner;f->destination=destination;f->cancel=cancel?(void *)((char *)M+cancel):NULL;
}

/* The ring slot a transfer's invocation t addresses: (t - begin) mod its depth, at its slot pages. */
static uint32_t transfer_row(struct hdr *m,uint32_t fallback,const struct mesh_transfer *transfer,uint32_t row,uint32_t invocation,uint32_t chunk){
  uint32_t depth=transfer->depth?transfer->depth:fallback;
  if(!depth)depth=1;
  return row+(uint32_t)((size_t)((invocation-transfer->begin)%depth)*transfer->invocation_pages/m->block)+chunk;
}

/* design/prepared-machine.md#M09: a chunk's span in the window its port's device registered when it came up; 0, or
   EINVAL for a page outside the window */
static int chunk_span(struct flow *link,uint32_t address,uint64_t remaining,struct ibv_sge *span){
  struct hdr *m=link->M;
  uint64_t offset=atomic_load_explicit(&mesh_page(m)[address].address,memory_order_relaxed)-m->data_off;
  if(offset>=wire.length)return EINVAL;
  *span=wire_span(link->provider.device,(uint64_t)m->block*m->pgsz,offset,remaining);
  return 0;
}

/* design/prepared-machine.md#M07, #M08, #M12: the word a landed chunk of invocation u of the transfer's slot `index` (0
   where each slot has its own rows) stores: its slot's completion word (mesh_host_inputs, the Metal transport), where
   the row's delivery names a word among the receiving program's own toward the flow's node (`words`, its cancellation
   range, [offset, offset + 8 count): a delivery another program wrote on the row, which outlives that program, is
   none of this one's); or, on a row the receiving program (`owner`) also publishes (a relay's rows), that slot's
   publication's SEND cell once the record's last chunk landed, so the landing releases it to the next hop (the index
   hand-off, *release set); else the record's own (`sink`), which nothing reads.  NULL where the word is not in the
   window (#M09) */
static _Atomic uint64_t *receive_input(struct hdr *m,uint64_t owner,const struct mesh_cancel_range *words,uint32_t row,uint32_t index,
                                       uint32_t chunk,uint32_t chunks,uint32_t u,_Atomic uint64_t *sink,uint32_t *release){
  const struct mesh_delivery *delivery=mesh_delivery_at(m,row+chunk);
  uint64_t at=0;
  const uint64_t word=delivery->device_input+delivery->device_stride*u+sizeof(uint64_t)*(uint64_t)index*chunks;
  *release=0;
  if(delivery->device_input && words && word>=words->offset && (word-words->offset)/sizeof(uint64_t)<words->count)at=word;
  else if(chunk+1==chunks){
    uint32_t sends;
    const struct mesh_target *target=mesh_row_targets(m,owner,row,&sends);
    if(index<sends){at=target[index].cells+sizeof(struct mesh_send)*(uint64_t)u;*release=1;}
  }
  if(!at)return sink;
  return at>=m->data_off && at<=m->length-sizeof(uint64_t)?(void *)((char *)m+at):NULL;
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
static int ring_advance(struct flow *link,struct receive_ring *ring){
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
    link->provider.received_posted++;
    ring->outstanding+=record->frames;ring->posted++;
  }
  return 0;
}

/* design/bridge-and-ipc.md#cables: whether the flow carries the program's invocation t, and how many of a transfer's
   active invocations [begin, begin + active) it carries */
static int stripe_of(const struct flow *link,uint32_t t){ return t%link->stripes==link->stripe; }
static uint32_t stripe_count(const struct flow *link,uint32_t begin,uint32_t active){
  uint32_t first=(link->stripe+link->stripes-begin%link->stripes)%link->stripes;
  return first<active?(active-first+link->stripes-1)/link->stripes:0;
}

/* design/bridge-and-ipc.md#lanes: the lane a flow's ring q (queue q / slots) rides, and that lane's queue pair */
static uint32_t ring_lane(uint32_t q,uint32_t slots){ return (q/(slots?slots:1))%MESH_LANES; }
static struct ibv_qp *lane_pair(const struct flow *link,uint32_t lane);

/* design/bridge-and-ipc.md#ledger: whether the configuration keeps ledgers (its `ledger`), and the bridge's ledgers
   numbered as they are made */
static int ledger_kept;
static _Atomic uint32_t ledger_ids;
/* design/bridge-and-ipc.md#keepalive: the configuration's `keepalive`, the most nanoseconds an exclusive flow's send side
   lets pass between frames on its control queue pair while it progresses (0 where it states none: no keepalive frame);
   its `hello`, the interval at which a port with no peer says HELLO again (#ports: a HELLO lost on its link has no
   other event; 0, none); and its `health`, the interval at which this node's record is flooded again
   (design/recovery.md#diagnosis; 0, none): each a paced emission the configuration states */
static uint64_t keepalive_ns,hello_ns,health_ns;
#ifdef MESH_LOOP_FABRIC
/* design/bridge-dataflow.md#the-loopback-fabric (mesh 800fe6bc^ mesh-flow-loop, on the resident bridge): a loop bridge's
   configuration's `loop`: its region's name, its node, the loopback port its clients' requests come to (`service`, in
   place of MESH_PORT), the loopverbs fabric it shares with the host's other loop bridges (loopverbs.c loop_fabric), and
   its ports, each a loopverbs device and a cable: the loopback port this end listens on for it and the peer end's, the
   lower of the two dialing the higher (in place of a link's hello) */
void loop_fabric(const char *name);
struct loop_port { char device[32]; uint16_t listen,peer; int fd; };
static struct { char region[32],fabric[64]; uint32_t node,ports; uint16_t service; struct loop_port port[MESH_PORTS]; } loop_config;
#endif
/* a flow's ledger, made on its port's queue before either side is prepared or the flow listed, where the configuration
   keeps them; a flow whose ledger cannot be allocated runs without one, logged */
static void ledger_open(struct flow *f){
  if(!ledger_kept || f->ledger)return;
  struct ledger *l=calloc(1,sizeof *l);
  uint32_t *words=l?malloc(3*(size_t)f->qps*sizeof *words):NULL;
  if(!words){
    free(l);
    say("link %u: program %016llx toward node %u (stripe %u of %u): no ledger: %s\n",f->port->index,(unsigned long long)f->key,
        f->far,f->stripe,f->stripes,strerror(ENOMEM));
    return;
  }
  for(size_t i=0;i<3*(size_t)f->qps;i++)words[i]=LEDGER_NONE;
  l->id=atomic_fetch_add(&ledger_ids,1)+1;l->queues=(uint32_t)f->qps;l->started=ledger_ns();
  l->current=words;l->pending=words+f->qps;l->last=words+2*(size_t)f->qps;
  f->ledger=l;
}
/* the events a side's ledger holds: one pass of the flow's `once` cells or records, every pass of a cyclic program's
   the bound's latest (a ring: event e at e % room, its count published as each is stored, so a cut (ledger_cut) reads
   them while the flow runs) */
static uint32_t ledger_room(uint64_t once,int cyclic){ return cyclic || once>MESH_LEDGER_EVENTS?MESH_LEDGER_EVENTS:(uint32_t)once; }
/* design/bridge-and-ipc.md#ledger: taken by a cut and by a ledger's release, so a cut never reads a ledger freed */
static pthread_mutex_t ledger_lock=PTHREAD_MUTEX_INITIALIZER;
/* the send thread's stores: a cell first seen released, its queue's event from then on */
static void ledger_seen(struct ledger *l,const struct mesh_send *cell,uint64_t cycle){
  const uint32_t q=cell->queue;
  if(!l->send_room){l->current[q]=LEDGER_NONE;return;}
  const uint32_t e=l->sends;
  l->send[e%l->send_room]=(struct ledger_send){.cell=(uintptr_t)cell,.seen=ledger_ns(),.cycle=(uint32_t)cycle,.next=LEDGER_NONE};
  if(l->last[q]!=LEDGER_NONE)l->send[l->last[q]%l->send_room].next=e;
  if(l->pending[q]==LEDGER_NONE)l->pending[q]=e;
  l->last[q]=e;l->current[q]=e;
  __atomic_store_n(&l->sends,e+1,__ATOMIC_RELEASE);
}
/* SENDs of event e posted at `at`: its first, and where `done` its last, after which its gate held `mark` frames */
static void ledger_stamp(struct ledger *l,uint32_t e,int done,uint64_t mark,uint64_t at){
  if(e==LEDGER_NONE)return;
  struct ledger_send *s=l->send+e%l->send_room;
  if(!s->first)s->first=at;
  if(done){s->posted=at;s->mark=mark&MESH_GATE_MASK;}
}
/* an exclusive flow's SEND completion on queue q, its gate's frames retired through it: every event of the queue whose
   last SEND it retires (a queue pair's completions come in the order posted) completed now */
static void ledger_retired(struct ledger *l,uint32_t q,uint64_t retired){
  uint64_t at=0;
  for(uint32_t e;q<l->queues && (e=l->pending[q])!=LEDGER_NONE;){
    struct ledger_send *s=l->send+e%l->send_room;
    if(!s->posted || ((retired-s->mark)&MESH_GATE_MASK)>=(UINT64_C(1)<<47))break;
    if(!at)at=ledger_ns();
    s->completed=at;l->pending[q]=s->next;
  }
}
/* the receive thread's store: a record of the flow landed, its word stored */
static void ledger_landed(struct flow *link,const struct prepared_receive *record){
  struct ledger *l=link->ledger;
  if(!l->landing_room)return;
  const uint32_t e=l->landings;
  l->landing[e%l->landing_room]=(struct ledger_landing){(uint32_t)(record-link->receive),record->cycle,ledger_ns()};
  __atomic_store_n(&l->landings,e+1,__ATOMIC_RELEASE);
}
/* design/bridge-and-ipc.md#ledger: a flow's ledger written at its release, on its port's queue once neither progress
   thread reads the flow: its header into the bridge's log, then a line for each cell seen and each record landed, which
   name the cell or record by queue, invocation, position, binding and cycle, as the peer's ledger of the same flow
   names the record or cell (rdma/ledger.py pairs them): into the log where there are MESH_LEDGER_LINES or fewer, else
   into a file beside the log named by the program's key, the node, its peer, the stripe, the ledger and the time, which
   the header names; then freed */
static int ledger_cell_order(const void *a,const void *b){
  const uint64_t x=((const struct ledger_cell *)a)->cell,y=((const struct ledger_cell *)b)->cell;
  return (x>y)-(x<y);
}
/* the ledger's header line, `cut` the events past the latest cut */
static void ledger_header(const struct flow *f,const struct ledger *l,uint32_t sends,uint32_t landings,const char *file,char *line,size_t room){
  snprintf(line,room,"{\"ledger\":%u,\"key\":\"%016llx\",\"node\":%u,\"peer\":%u,\"port\":%u,\"near\":%u,\"far\":%u,\"stripe\":%u,"
           "\"stripes\":%u,\"exclusive\":%d,\"queues\":%u,\"qps\":%d,\"invocations\":%u,\"cyclic\":%u,\"started\":%llu,\"sends\":%u,\"send_room\":%u,"
           "\"landings\":%u,\"landing_room\":%u,\"cut\":[%u,%u],\"file\":%s%s%s}\n",l->id,(unsigned long long)f->key,M->node,f->port->peer,
           f->port->index,f->near,f->far,f->stripe,f->stripes,f->exclusive,f->queues,f->qps,l->invocations,l->cyclic,(unsigned long long)l->started,
           sends,l->send_room,landings,l->landing_room,l->cut_sends,l->cut_landings,file?"\"":"",file?file:"null",file?"\"":"");
}
/* the event lines of the ledger's cells seen and records landed from the latest cut through `sends` and `landings` (the
   ring's latest where more passed), each naming its cell or record by queue, invocation, position, binding and cycle as
   the peer's ledger of the flow names the record or cell (rdma/ledger.py pairs them), to `out`, else the log */
static void ledger_events(const struct flow *f,struct ledger *l,uint32_t sends,uint32_t landings,FILE *out){
  char line[512];
  if(l->cells && !l->sorted){qsort(l->cell,l->cells,sizeof *l->cell,ledger_cell_order);l->sorted=1;}
  for(uint32_t e=sends-l->cut_sends>l->send_room?sends-l->send_room:l->cut_sends;e!=sends;e++){
    const struct ledger_send *s=l->send+e%l->send_room;
    const struct ledger_cell key={.cell=s->cell},*c=l->cells?bsearch(&key,l->cell,l->cells,sizeof *l->cell,ledger_cell_order):NULL;
    snprintf(line,sizeof line,"{\"s\":[%u,%lld,%lld,%lld,%lld,%u,%lld,%llu,%llu,%llu,%llu,%lld]}\n",l->id,c?(long long)c->queue:-1ll,
             c?(long long)c->invocation:-1ll,c?(long long)c->position:-1ll,c?(long long)c->binding:-1ll,s->cycle,c?(long long)c->chunks:-1ll,
             (unsigned long long)s->seen,(unsigned long long)s->first,(unsigned long long)s->posted,(unsigned long long)s->completed,
             c?(long long)c->bytes:-1ll);
    if(out)fputs(line,out);else say("%s",line);
  }
  for(uint32_t e=landings-l->cut_landings>l->landing_room?landings-l->landing_room:l->cut_landings;e!=landings;e++){
    const struct ledger_landing *r=l->landing+e%l->landing_room;
    const struct prepared_receive *record=f->receive && r->record<l->records?f->receive+r->record:NULL;
    const struct ledger_record *n=record?l->record+r->record:NULL;
    snprintf(line,sizeof line,"{\"r\":[%u,%lld,%lld,%lld,%lld,%lld,%lld,%u,%llu]}\n",l->id,record?(long long)record->queue:-1ll,
             record?(long long)record->invocation:-1ll,n?(long long)n->position:-1ll,n?(long long)n->binding:-1ll,n?(long long)n->chunk:-1ll,
             n?(long long)n->chunks:-1ll,r->cycle,(unsigned long long)r->landed);
    if(out)fputs(line,out);else say("%s",line);
  }
  l->cut_sends=sends;l->cut_landings=landings;
}
/* design/bridge-and-ipc.md#ledger: a flow's ledger written at its release, on its port's queue once neither progress
   thread reads the flow: its header into the bridge's log, then its events past its latest cut: into the log where
   there are MESH_LEDGER_LINES or fewer, else into a file beside the log named by the program's key, the node, its peer,
   the stripe, the ledger and the time, which the header names; then freed */
static void ledger_write(struct flow *f){
  pthread_mutex_lock(&ledger_lock);
  struct ledger *l=f->ledger;
  f->ledger=NULL;
  pthread_mutex_unlock(&ledger_lock);
  if(!l)return;
  char path[PATH_MAX]={0},line[640];
  const char *file=NULL;
  FILE *out=NULL;
  struct stat log;
  if((uint64_t)(l->sends-l->cut_sends)+(l->landings-l->cut_landings)>MESH_LEDGER_LINES && !fstat(STDERR_FILENO,&log) && S_ISREG(log.st_mode) &&
     fcntl(STDERR_FILENO,F_GETPATH,path)!=-1){
    char *slash=strrchr(path,'/'),when[24];
    const time_t wall=time(NULL);struct tm utc;
    gmtime_r(&wall,&utc);strftime(when,sizeof when,"%Y%m%dT%H%M%SZ",&utc);
    if(slash && (size_t)snprintf(slash+1,sizeof path-(size_t)(slash+1-path),"ledger-%016llx-%u-%u-%u-%u-%s.jsonl",
                                 (unsigned long long)f->key,M->node,f->port->peer,f->stripe,l->id,when)<sizeof path-(size_t)(slash+1-path) &&
       (out=fopen(path,"w")))file=slash+1;
  }
  ledger_header(f,l,l->sends,l->landings,file,line,sizeof line);
  say("%s",line);
  if(out)fputs(line,out);
  ledger_events(f,l,l->sends,l->landings,out);
  if(out && fclose(out))say("link %u: ledger %u: %s: %s\n",f->port->index,l->id,file,strerror(errno));
  free(l->current);free(l->send);free(l->cell);free(l->landing);free(l->record);free(l);
}
/* design/bridge-and-ipc.md#ledger: a client's request for its program's ledger (MESH_REQUEST_LEDGER, the program's key):
   each running flow of the key's events since its latest cut appended to `path`, each after its header, while the flow
   runs (its progress threads store on; a cut reads the counts they published); ENOENT where no flow of the key keeps a
   ledger, ENOTSUP where the configuration keeps none */
static int ledger_cut(uint64_t key,const char *path){
  if(!ledger_kept)return ENOTSUP;
  FILE *out=NULL;
  int found=0;
  char line[640];
  pthread_mutex_lock(&ledger_lock);
  for(struct flow *f=flows;f;f=f->next){
    struct ledger *l=f->ledger;
    if(f->key!=key || !l || f->state!=RUNNING)continue;
    if(!out && !(out=fopen(path,"a"))){const int error=errno;pthread_mutex_unlock(&ledger_lock);return error;}
    const uint32_t sends=__atomic_load_n(&l->sends,__ATOMIC_ACQUIRE),landings=__atomic_load_n(&l->landings,__ATOMIC_ACQUIRE);
    ledger_header(f,l,sends,landings,NULL,line,sizeof line);
    fputs(line,out);
    ledger_events(f,l,sends,landings,out);
    found=1;
  }
  pthread_mutex_unlock(&ledger_lock);
  if(out && fclose(out))return errno;
  return found?0:ENOENT;
}

/* design/algorithm-sources.md#programcopy */
/* design/prepared-machine.md#M04, #M06, #M29 */
/* The SEND side: every SEND cell of the flow's invocations (#cables), its chain of chunk requests over its invocation's
   ring slot and its stream successor, a stream a ring (the cells' layout is mesh_transfers_prepare's), and an exclusive
   flow's gate a queue pair (a flow on its port's lanes posts through the lanes' gates, #lanes).  A stream's cells run
   invocation by invocation, the publications of each in binding order, as the peer's ring posts its receives: TN3205's
   credit flow control holds a SEND until its matching receive is posted (metal-microbench docs/kernels.md Transport
   (git 7300d1e^)).  0, or the error (EINVAL: a transfer of more slots than the program's). */
static int sends_configure(struct flow *link){
  struct hdr *m=link->M;struct mesh_client *client=mesh_client_at(m,link->side[MESH_SEND].owner);
  struct mesh_tx *tx=side_tx(link,MESH_SEND);
  uint32_t invocations=tx->invocations?tx->invocations:1,fallback=atomic_load_explicit(&client->depth,memory_order_acquire);
  int cyclic=atomic_load_explicit(&client->cyclic,memory_order_acquire)!=0,qps=link->qps;
  size_t requests=0,cells=0;
  for(uint32_t k=0;k<link->queues;k++){
    struct mesh_transfer *out=side_transfers(link,MESH_SEND,k);
    for(uint32_t i=0;i<side_length(link,MESH_SEND,k);i++){
      if(out[i].count>tx->slots || (uint64_t)k*tx->slots+out[i].count>(uint64_t)qps)return EINVAL;
      const size_t own=(size_t)out[i].count*stripe_count(link,out[i].begin,mesh_transfer_active(out+i,invocations));
      requests+=(size_t)(mesh_row_chunks(m,out[i].local_row,out[i].bytes)-1)*own;
      cells+=own;
    }
  }
  link->publications=(void *)((char *)m+tx->cells);
  /* design/prepared-machine.md#M29 */
  link->requests=aligned_alloc(128,(requests+1)*sizeof *link->requests);
  link->streams=calloc((size_t)qps,sizeof *link->streams);
  link->gates=link->exclusive?aligned_alloc(64,(size_t)qps*sizeof *link->gates):NULL;
  struct mesh_send **last=calloc((size_t)qps,sizeof *last),**terminal=calloc((size_t)qps,sizeof *terminal);
  if(!link->requests||!link->streams||(link->exclusive && !link->gates)||!last||!terminal){free(last);free(terminal);return ENOMEM;}
  link->request_count=requests;
  /* #ledger: the cells' names, each stream's position at its latest invocation (`positions`, a slot's pair), and room for
     the events of one pass, a cyclic program's of every pass up to the bound */
  struct ledger *ledger=link->ledger;
  uint32_t *positions=NULL;
  if(ledger){
    ledger->invocations=invocations;ledger->cyclic=(uint32_t)cyclic;
    ledger->cell=malloc((cells+1)*sizeof *ledger->cell);
    ledger->send=malloc(((size_t)ledger_room(cells,cyclic)+1)*sizeof *ledger->send);
    positions=malloc((2*(size_t)tx->slots+2)*sizeof *positions);
    if(!ledger->cell || !ledger->send || !positions){
      free(ledger->cell);free(ledger->send);free(positions);ledger->cell=NULL;ledger->send=NULL;positions=NULL;ledger=NULL;
      say("link %u: program %016llx toward node %u (stripe %u of %u): no ledger of its SENDs: %s\n",link->port->index,
          (unsigned long long)link->key,link->far,link->stripe,link->stripes,strerror(ENOMEM));
    }else ledger->send_room=ledger_room(cells,cyclic);
  }
  /* a request is one frame at least, so a queue pair's frames also bound the completions it can have pending in the
     SEND completion queue its pairs share, beside the control frames' (#control) */
  if(link->exclusive){
    const uint32_t cqe=(uint32_t)link->provider.sent->cqe,held=link->alive?CONTROL_SENDS:0;
    uint32_t sends=(cqe>held?cqe-held:1)/(uint32_t)qps;
    for(int q=0;q<qps;q++)link->gates[q]=(struct send_gate){.capacity=MIN(link->provider.queues[q].send_capacity,sends)};
  }
  size_t next=0;int error=0;
  for(uint32_t k=0;k<link->queues && !error;k++){
    uint32_t count=side_length(link,MESH_SEND,k);
    struct mesh_transfer *out=side_transfers(link,MESH_SEND,k);
    /* each cell's chain over its invocation's ring slot */
    for(uint32_t i=0;i<count && !error;i++){
      uint32_t active=mesh_transfer_active(out+i,invocations),column=active+1;
      for(uint32_t slot=0;slot<out[i].count && !error;slot++){
        uint32_t row=out[i].local_row+slot*out[i].stride,chunks=mesh_row_chunks(m,row,out[i].bytes),stream=k*tx->slots+slot;
        struct ibv_qp *pair=link->exclusive?link->provider.queues[stream].pair:lane_pair(link,ring_lane(stream,tx->slots));
        struct mesh_send *cells=link->publications+out[i].first+(size_t)slot*column;
        for(uint32_t u=0;u<active && !error;u++){
          if(!stripe_of(link,out[i].begin+u))continue;
          struct mesh_send *cell=cells+u;
          cell->pair=(uintptr_t)pair;cell->queue=stream;cell->chunks=chunks;
          atomic_store_explicit(&cell->port,link->port->index+1,memory_order_release);
          uint64_t remaining=out[i].bytes;
          for(uint32_t c=0;c<chunks && !error;c++){
            struct ibv_sge span;
            if((error=chunk_span(link,transfer_row(m,fallback,out+i,row,out[i].begin+u,c),remaining,&span)))break;
            remaining-=span.length;
            struct ibv_sge *entry=c?&link->requests[next+c-1].span:&cell->span;
            struct ibv_send_wr *request=c?&link->requests[next+c-1].request:&cell->request;
            *entry=span;
            /* the first request's wr_id stays the producer's nonzero argument (mesh_transfers_prepare),
               which a host producer reads at publication (mesh_host_publish) */
            *request=(struct ibv_send_wr){.wr_id=c?0:1,.next=c+1==chunks?NULL:&link->requests[next+c].request,
              .sg_list=entry,.num_sge=1,.opcode=IBV_WR_SEND};
          }
          next+=chunks-1;
        }
        terminal[stream]=cells+active;
      }
    }
    /* each stream's cells in invocation order, its publications in binding order within one */
    if(positions)memset(positions,0,(2*(size_t)tx->slots+2)*sizeof *positions);
    for(uint32_t t=0;t<invocations && !error;t++)for(uint32_t i=0;i<count && stripe_of(link,t);i++){
      uint32_t active=mesh_transfer_active(out+i,invocations);
      if(t<out[i].begin || t-out[i].begin>=active)continue;
      for(uint32_t slot=0;slot<out[i].count;slot++){
        uint32_t stream=k*tx->slots+slot;
        struct mesh_send *cell=link->publications+out[i].first+(size_t)slot*(active+1)+(t-out[i].begin);
        if(last[stream])last[stream]->successor=(uintptr_t)cell;
        else link->streams[stream].cell=cell;
        last[stream]=cell;
        /* #ledger: the cell's name, its position among the stream's cells of invocation t */
        if(positions && ledger->cells<cells){
          if(positions[2*slot]!=t+1){positions[2*slot]=t+1;positions[2*slot+1]=0;}
          ledger->cell[ledger->cells++]=(struct ledger_cell){(uintptr_t)cell,out[i].bytes,stream,t,positions[2*slot+1]++,out[i].binding,
            mesh_row_chunks(m,out[i].local_row+slot*out[i].stride,out[i].bytes)};
        }
      }
    }
  }
  /* design/prepared-machine.md#M30: a cyclic program's last cell is followed by its first */
  for(int q=0;q<qps && !error;q++)if(last[q])last[q]->successor=cyclic?(uintptr_t)link->streams[q].cell:(uintptr_t)terminal[q];
  for(int q=0;q<qps;q++){
    link->streams[q].first=link->streams[q].cell;link->streams[q].cycle=0;link->streams[q].sent=0;
    link->streams[q].lane=ring_lane((uint32_t)q,tx->slots);link->streams[q].queue=(uint32_t)q;
  }
  link->stream_count=0;
  for(int q=0;q<qps;q++)if(link->streams[q].cell)link->streams[link->stream_count++]=link->streams[q];
  free(last);free(terminal);free(positions);
  return error;
}

/* design/prepared-machine.md#M08 */
/* The receive side: each ring's records invocation by invocation (the flow's invocations, #cables), the transfers of
   each in binding order, a transfer's chunks in order, each storing its word (receive_input).  An exclusive flow's
   ring is its queue pair's, its first records posted at once and each record posted once the record of its transfer
   `span` of the flow's own invocations before has landed (the ring's span over the program's invocations, divided
   among the stripes; a cyclic ring's first records wait on its last of the cycle before), so a ring's posted frames
   stay within its queue pair's.  A flow on its port's lanes posts nothing here: a lane's header posts its records as
   the peer sends them (#lanes).  0, or the error (EINVAL: a transfer of more slots than the program's, a word outside
   the window). */
static int receives_configure(struct flow *link){
  struct hdr *m=link->M;struct mesh_client *client=mesh_client_at(m,link->side[MESH_RECEIVE].owner);
  /* design/prepared-machine.md#M09, #M10: the program's targets lie in the window, as each word it stores does */
  if(client->target_count && (client->targets<m->data_off || client->targets>m->length ||
     client->target_count>(m->length-client->targets)/sizeof(struct mesh_target)))return EINVAL;
  struct mesh_tx *tx=side_tx(link,MESH_RECEIVE);
  /* design/prepared-machine.md#M12: the program's own words toward the flow's node, where it has any */
  const uint64_t range=sizeof(struct mesh_cancellation)+sizeof(struct mesh_cancel_range)*((uint64_t)link->side[MESH_RECEIVE].destination+1);
  const struct mesh_cancellation *cancel=tx->cancel>=m->data_off && tx->cancel<=m->length-range?(const void *)((char *)m+tx->cancel):NULL;
  const struct mesh_cancel_range *words=cancel?cancel->ranges+link->side[MESH_RECEIVE].destination:NULL;
  uint32_t invocations=tx->invocations?tx->invocations:1,fallback=atomic_load_explicit(&client->depth,memory_order_acquire),depth=fallback;
  if(!depth||depth>invocations)depth=invocations;
  link->cyclic=atomic_load_explicit(&client->cyclic,memory_order_acquire)!=0;link->invocations=invocations;
  int qps=link->qps;
  size_t records=0;
  for(uint32_t k=0;k<link->queues;k++){
    struct mesh_transfer *in=side_transfers(link,MESH_RECEIVE,k);
    for(uint32_t i=0;i<side_length(link,MESH_RECEIVE,k);i++){
      if(in[i].count>tx->slots || (uint64_t)k*tx->slots+in[i].count>(uint64_t)qps)return EINVAL;
      records+=(uint64_t)mesh_row_chunks(m,in[i].local_row,in[i].bytes)*in[i].count*stripe_count(link,in[i].begin,mesh_transfer_active(in+i,invocations));
    }
  }
  link->receive=aligned_alloc(128,(records+1)*sizeof *link->receive);
  link->rings=calloc((size_t)qps,sizeof *link->rings);
  if(!link->receive||!link->rings)return ENOMEM;
  link->receive_count=records;
  /* #ledger: the records' names, and room for the landings of one pass, a cyclic program's of every pass up to the
     bound */
  struct ledger *ledger=link->ledger;
  if(ledger){
    ledger->invocations=invocations;ledger->cyclic=(uint32_t)link->cyclic;
    ledger->record=malloc((records+1)*sizeof *ledger->record);
    ledger->landing=malloc(((size_t)ledger_room(records,link->cyclic)+1)*sizeof *ledger->landing);
    if(!ledger->record || !ledger->landing){
      free(ledger->record);free(ledger->landing);ledger->record=NULL;ledger->landing=NULL;ledger=NULL;
      say("link %u: program %016llx toward node %u (stripe %u of %u): no ledger of its landings: %s\n",link->port->index,
          (unsigned long long)link->key,link->far,link->stripe,link->stripes,strerror(ENOMEM));
    }else{ledger->records=(uint32_t)records;ledger->landing_room=ledger_room(records,link->cyclic);}
  }
  /* a record is one frame at least, so a queue pair's frames also bound the completions it can have pending in the
     receive completion queue its pairs share, beside the control receives (#control) */
  const uint32_t cqe=link->exclusive?(uint32_t)link->provider.completion->cqe:0,held=link->alive?CONTROL_RECEIVES:0;
  const uint32_t receives=link->exclusive?(cqe>held?cqe-held:1)/(uint32_t)qps:UINT32_MAX;
  for(int q=0;q<qps;q++)link->rings[q]=(struct receive_ring){.completed=-1,.span=UINT32_MAX,.depth=invocations,
    .capacity=link->exclusive?MIN(link->provider.queues[q].receive_capacity,receives):UINT32_MAX};
  size_t at=0;int error=0;
  for(int ring=0;ring<qps && !error;ring++){
    uint32_t k=(uint32_t)ring/tx->slots,slot=(uint32_t)ring%tx->slots,count=k<link->queues?side_length(link,MESH_RECEIVE,k):0;
    struct mesh_transfer *in=count?side_transfers(link,MESH_RECEIVE,k):NULL;
    struct ibv_qp *pair=link->exclusive?link->provider.queues[ring].pair:lane_pair(link,ring_lane((uint32_t)ring,tx->slots));
    struct receive_ring *r=link->rings+ring;
    r->first=link->receive+at;
    /* firsts: the ring index of each transfer's own records, own invocation by own invocation (base[i] the first's) */
    size_t *base=calloc((size_t)count+1,sizeof *base),*firsts=NULL;
    for(uint32_t i=0;base&&i<count;i++)
      base[i+1]=base[i]+(slot<in[i].count?stripe_count(link,in[i].begin,mesh_transfer_active(in+i,invocations)):0);
    if(base)firsts=malloc((base[count]+1)*sizeof *firsts);
    if(!base||!firsts){free(base);free(firsts);return ENOMEM;}
    for(uint32_t i=0;i<count;i++){
      if(slot>=in[i].count)continue;
      uint32_t active=mesh_transfer_active(in+i,invocations),own=in[i].depth?in[i].depth:depth;
      /* #control: the ring's least depth, which bounds a cancel past its landed count (flow_cancel) */
      r->depth=MIN(r->depth,own);
      /* receive u+span lands in slot (u+span) mod depth while the local reader may still be at u:
         the ring's span is held a slot short of the ring; a transfer whose every invocation has its
         own slot bounds nothing */
      if(own<active)r->span=MIN(r->span,own>1?own-1:1);
    }
    uint32_t position=0,latest=UINT32_MAX;
    for(uint32_t t=0;t<invocations && !error;t++)for(uint32_t i=0;i<count && !error && stripe_of(link,t);i++){
      uint32_t active=mesh_transfer_active(in+i,invocations);
      if(slot>=in[i].count || t<in[i].begin || t-in[i].begin>=active)continue;
      if(latest!=t){latest=t;position=0;}
      uint32_t u=t-in[i].begin,row=in[i].local_row+slot*in[i].stride,chunks=mesh_row_chunks(m,row,in[i].bytes);
      const uint32_t first=(link->stripe+link->stripes-in[i].begin%link->stripes)%link->stripes,mine=(u-first)/link->stripes;
      uint64_t remaining=in[i].bytes;
      for(uint32_t c=0;c<chunks;c++){
        struct ibv_sge span;
        if((error=chunk_span(link,transfer_row(m,fallback,in+i,row,t,c),remaining,&span)))break;
        remaining-=span.length;
        struct prepared_receive *record=link->receive+at++;
        if(!c)firsts[base[i]+mine]=(size_t)(record-r->first);
        uint32_t release;
        *record=(struct prepared_receive){.request={.wr_id=(uintptr_t)record,.sg_list=&record->span,.num_sge=1},
          .span=span,.input=receive_input(m,link->side[MESH_RECEIVE].owner,words,row,in[i].stride?0:slot,c,chunks,u,&record->sink,&release),
          .pair=pair,.argument=1,.frames=(span.length+4095)/4096,.invocation=t,.queue=(uint32_t)ring,.link=link,.last=c+1==chunks};
        record->release=release;
        if(!record->input){error=EINVAL;break;}
        /* #ledger: the record's name, its transfer's position among the ring's transfers of invocation t */
        if(ledger)ledger->record[record-link->receive]=(struct ledger_record){position,in[i].binding,c,chunks};
      }
      position++;
    }
    r->count=(size_t)(link->receive+at-r->first);
    /* design/prepared-machine.md#M30 */
    if(link->cyclic)r->span=MIN(r->span,invocations>1?invocations-1:1);
    /* #cables: the span in the flow's own invocations, a stripe's share of the program's (one at least) */
    const uint32_t span=r->span==UINT32_MAX?UINT32_MAX:MAX(r->span/link->stripes,1u);
    for(uint32_t i=0;i<count && !error;i++){
      uint32_t own=(uint32_t)(base[i+1]-base[i]);
      if(!own)continue;
      uint32_t chunks=mesh_row_chunks(m,in[i].local_row+slot*in[i].stride,in[i].bytes);
      for(uint32_t j=0;j<own;j++){
        size_t here=firsts[base[i]+j],behind=0;
        if(span!=UINT32_MAX && j>=span)behind=here-firsts[base[i]+j-span];
        else if(span!=UINT32_MAX && link->cyclic && j+own>=span)behind=here+r->count-firsts[base[i]+j+own-span];
        else if(span!=UINT32_MAX && link->cyclic)behind=r->count;
        for(uint32_t c=0;c<chunks;c++)r->first[here+c].behind=(uint32_t)behind;
      }
    }
    r->span=span;
    free(base);free(firsts);
    if(!error && link->exclusive)error=ring_advance(link,r);
  }
  if(at)link->receive[at]=link->receive[at-1];
  return error;
}

/* design/bridge-and-ipc.md#progress: a landing that released a SEND cell (the index hand-off of a relay, or of a member
   that sends on what it receives) wakes the send thread that posts the cell, where that thread is parked: the fence
   orders the release before the read of its parked word, as the thread orders its parked word before its last look
   (port_progress), so one of the two sees the other */
static void send_wake(_Atomic uint64_t *ready){
  const uint32_t p=atomic_load_explicit(&((struct mesh_send *)(void *)ready)->port,memory_order_acquire);
  atomic_thread_fence(memory_order_seq_cst);
  if(p && p<=MESH_PORTS && atomic_load_explicit(&ports[p-1].parked[1],memory_order_relaxed))mesh_notify(&M->bell[p-1]);
}

/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M06 */
/* design/prepared-machine.md#M15 */
/* design/algorithm-sources.md#independent-native-queues */
/* The flow's SEND completions: each retires its queue's frames through its request (the device completes every SEND,
   signaled or not), a control frame's its control frames (#control), an error completion's as well; *taken their
   count, every one counted taken (mesh-verbs.h pair_flush).  0, or the first error completion's status. */
static uint64_t lane_discard(const struct port *port,uint32_t lane);
static struct ibv_sge lane_span(const struct port *port,uint64_t offset,uint64_t length);
static __attribute__((always_inline)) inline int send_retire(struct flow *link,int *taken){
  struct ibv_wc done[16];
  int count=link->provider.queues[0].poll(link->provider.sent,16,done);
  *taken=count>0?count:0;
  if(count<0)return EIO;
  link->provider.sent_polled+=(uint64_t)count;
  int error=0;
  for(int i=0;i<count;i++){
    if(done[i].status && !error)error=(int)done[i].status;
    /* #control */
    if((done[i].wr_id&~CONTROL_ID)==CONTROL_SENT_TAG){link->control_retired++;continue;}
    struct send_gate *gate=link->gates+(done[i].wr_id>>48);
    gate->retired=done[i].wr_id&MESH_GATE_MASK;gate->completions++;
    /* #ledger */
    if(link->ledger)ledger_retired(link->ledger,(uint32_t)(done[i].wr_id>>48),gate->retired);
  }
  return error;
}

/* design/bridge-and-ipc.md#control: the window offset of an exclusive flow's control frame slot k, of those its SENDs
   read or (`heard`) of those its receives fill, in its own page; the pages a flow's frames take */
static uint64_t control_offset(const struct flow *link,int heard,uint32_t k){
  return (uint64_t)link->control_page*M->pgsz+(heard?(uint64_t)CONTROL_SENDS*sizeof(struct control_frame)+(uint64_t)k*CONTROL_SLOT:
    (uint64_t)k*sizeof(struct control_frame));
}
static uint32_t control_pages(void){
  return (uint32_t)(((uint64_t)CONTROL_SENDS*sizeof(struct control_frame)+(uint64_t)CONTROL_RECEIVES*CONTROL_SLOT+M->pgsz-1)/M->pgsz);
}
/* a frame posted on the flow's control queue pair, from its next send slot once that slot's last SEND retired: 0,
   EAGAIN where CONTROL_SENDS are in flight, else the post's error */
static int control_post(struct flow *link,uint32_t kind,uint32_t channel,uint64_t count){
  if(link->control_posted-link->control_retired>=CONTROL_SENDS)return EAGAIN;
  const uint64_t offset=control_offset(link,0,(uint32_t)(link->control_posted%CONTROL_SENDS));
  const struct control_frame frame={kind,channel,count};
  memcpy(wire.data+offset,&frame,sizeof frame);
  struct ibv_sge span=lane_span(link->port,offset,sizeof frame);
  struct ibv_send_wr posting={.wr_id=CONTROL_SENT_TAG|(link->control_posted&CONTROL_ID),.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,
    .send_flags=IBV_SEND_SIGNALED},*bad;
  const int error=link->provider.queues[link->qps].send(link->provider.queues[link->qps].pair,&posting,&bad);
  if(error)return error<0?-error:error;
  link->control_posted++;link->provider.sent_posted++;
  return 0;
}
/* receive slot k of the flow's control queue pair posted; 0, or the error */
static int control_listen(struct flow *link,uint32_t k){
  struct ibv_sge span=lane_span(link->port,control_offset(link,1,k),CONTROL_SLOT);
  struct ibv_recv_wr request={.wr_id=CONTROL_HEARD_TAG|k,.sg_list=&span,.num_sge=1},*bad;
  const int error=link->provider.queues[link->qps].receive(link->provider.queues[link->qps].pair,&request,&bad);
  if(!error)link->provider.received_posted++;
  return error<0?-error:error;
}
/* a frame the peer sent, on the receive thread: a FAIL the flow's failure (its threads stop serving it, the event loop
   fails it, flows_turn); a relay's RELEASED its end's in-order release (the event loop passes it on and releases the
   relay once both ends said so); a keepalive nothing */
static void control_apply(struct flow *link,uint32_t kind,uint32_t channel,uint64_t count){
  if(kind==CONTROL_FAIL){
    int64_t none=0;
    if(atomic_compare_exchange_strong(&link->heard_fail,&none,(int64_t)(count?count:ECANCELED)))atomic_store(&link->heard_origin,channel);
    atomic_store_explicit(&link->progressing,0,memory_order_release);
    wake();
  }else if(kind==CONTROL_RELEASED && link->partner){
    atomic_store_explicit(&link->released,1,memory_order_release);
    wake();
  }
}
/* receive slot k's frame heard, on the receive thread: applied, the slot posted again; 0, or the post's error */
static int frame_heard(struct flow *link,uint32_t k){
  struct control_frame frame;
  memcpy(&frame,wire.data+control_offset(link,1,k),sizeof frame);
  link->alive_heard++;
  control_apply(link,frame.kind,frame.channel,frame.count);
  return control_listen(link,k);
}
/* design/bridge-and-ipc.md#control, #keepalive: an exclusive flow's control frames, one pass of its send thread: its
   FAIL while the event loop asks it told (FLOW_TELLING; `said` once its completion is taken, or at once where it cannot
   be posted); else, with the configuration's `keepalive`, an ALIVE frame once that interval passed since the last and
   none is in flight (a paced emission: the Mini's Thunderbolt wakes slowly after quiet, mesh 12f484f8).  Whether a
   frame was posted, -1 where the flow was lost */
static int flow_control(struct flow *link,int flags){
  if(flags&FLOW_TELLING){
    if(!link->fail_posted){
      const int error=control_post(link,CONTROL_FAIL,link->fail_origin,(uint64_t)link->fail_code);
      if(error==EAGAIN)return 0;
      if(error){atomic_store_explicit(&link->said,1,memory_order_release);wake();return 0;}
      link->fail_posted=link->control_posted;
      return 1;
    }
    if(!atomic_load_explicit(&link->said,memory_order_relaxed) && link->control_retired>=link->fail_posted){
      atomic_store_explicit(&link->said,1,memory_order_release);wake();
    }
    return 0;
  }
  if(!keepalive_ns || link->control_posted!=link->control_retired)return 0;
  const uint64_t at=ledger_ns();
  if(at-link->alive_at<keepalive_ns)return 0;
  const int error=control_post(link,CONTROL_ALIVE,0,0);
  if(error && error!=EAGAIN){flow_lost(link,error,1);return -1;}
  if(!error){link->alive_at=at;link->alive_posted++;}
  return !error;
}
/* design/bridge-and-ipc.md#control: whether every cell a flow's program released is posted: each stream's next cell not
   released and none of it left to post */
static int streams_idle(const struct flow *link){
  for(uint32_t s=0;s<link->stream_count;s++){
    const struct send_stream *stream=link->streams+s;
    if(stream->remaining || atomic_load_explicit(&stream->cell->ready,memory_order_acquire)>stream->cycle)return 0;
  }
  return 1;
}
/* a flow's program ended in order (FLOW_ENDING), told the event loop once it drained (`drained`, flows_turn) */
static void flow_drained(struct flow *link){
  atomic_store_explicit(&link->drained,1,memory_order_release);
  wake();
}

/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#independent-native-queues */
/* A running flow's SENDs, one pass of its port's send thread (design/bridge-and-ipc.md#progress): each stream's cells
   in order, a cell released by its producer posting its chain's requests while its queue pair holds their frames (the
   gate: frames posted less those its SEND completions retired; ibv_post_send accepts past the queue's frames and
   corrupts later, so the gate is the caller's), none of a muted program's (program_quiet); then its control frames
   (flow_control) and its SEND completions taken; and a program ended in order told drained once every cell it released
   is posted and completed, its control frames too (#control).  Whether anything moved; a failure is the flow's
   (flow_lost). */
static int flow_send(struct flow *link){
  int moved=0,taken=0;
  struct ibv_send_wr *bad;
  struct ledger *ledger=link->ledger;
  int (*post)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **)=link->provider.queues[0].send;
  const int flags=atomic_load_explicit(&link->ending,memory_order_acquire);
  for(uint32_t s=0;!(flags&FLOW_MUTED) && s<link->stream_count;s++){
    struct send_stream *stream=link->streams+s;
    for(;;){
      if(!stream->remaining){
        struct mesh_send *cell=stream->cell;
        if(atomic_load_explicit(&cell->ready,memory_order_acquire)<=stream->cycle)break;
        stream->next=&cell->request;stream->remaining=cell->chunks;
        /* #ledger */
        if(ledger)ledger_seen(ledger,cell,stream->cycle);
      }
      struct mesh_send *cell=stream->cell;
      struct send_gate *gate=link->gates+cell->queue;
      const uint32_t unposted=stream->remaining;
      while(stream->remaining){
        struct ibv_send_wr *request=stream->next;
        uint32_t frames=(request->sg_list->length+4095)/4096;
        if(((gate->posted-gate->retired)&MESH_GATE_MASK)+frames>gate->capacity)break;
        gate->posted+=frames;
        /* posted as a copy: the cell's first request keeps the producer's argument in its wr_id (mesh_transfers_prepare),
           which a host producer reads at every publication (mesh_host_publish), and the copy's names the gate */
        struct ibv_send_wr posting=*request;
        posting.next=NULL;posting.send_flags=IBV_SEND_SIGNALED;
        posting.wr_id=((uint64_t)cell->queue<<48)|(gate->posted&MESH_GATE_MASK);
        int error=post(link->provider.queues[cell->queue].pair,&posting,&bad);
        if(error){gate->posted-=frames;flow_lost(link,error<0?-error:error,1);return 1;}
        link->provider.sent_posted++;
        stream->next=request->next;stream->remaining--;gate->requests++;moved=1;
      }
      /* #ledger */
      if(ledger && stream->remaining!=unposted)ledger_stamp(ledger,ledger->current[cell->queue],!stream->remaining,gate->posted,ledger_ns());
      if(stream->remaining)break;
      stream->cell=(struct mesh_send *)cell->successor;stream->sent++;
      if(stream->cell==stream->first)stream->cycle++;
    }
  }
  /* #control */
  const int framed=link->alive && link->peer_alive?flow_control(link,flags):0;
  if(framed<0)return 1;
  if(link->stream_count || link->control_posted!=link->control_retired){
    int error=send_retire(link,&taken);
    if(error){flow_lost(link,error,2);return 1;}
  }
  if((flags&FLOW_ENDING) && !atomic_load_explicit(&link->drained,memory_order_relaxed) && link->control_posted==link->control_retired &&
     streams_idle(link)){
    int sent=1;
    for(int q=0;q<link->qps && sent;q++)sent=link->gates[q].requests==link->gates[q].completions;
    if(sent)flow_drained(link);
  }
  return moved || framed || taken;
}

/* design/prepared-machine.md#M08 */
/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#programcopy */
/* A record landed, as its port's receive thread takes it (an exclusive flow's, or a lane rider's): its completion word
   stored and a SEND it releases woken */
static __attribute__((always_inline)) inline void record_landed(struct flow *link,struct prepared_receive *record,struct receive_ring *ring,
                                                                 uint32_t bytes){
  ring->landed++;ring->bytes+=bytes;
  atomic_store_explicit(record->input,record->argument+record->cycle,memory_order_release);
  if(record->release)send_wake(record->input);
  /* #ledger */
  if(link->ledger)ledger_landed(link,record);
  ring->outstanding-=record->frames;ring->completed=(int64_t)(record->invocation+(uint64_t)record->cycle*link->invocations);
}
/* A running flow's receive completions, one pass of its port's receive thread: each record's landing (record_landed)
   posting its ring's next records, each control frame heard (frame_heard), every completion counted taken (an error's
   too: mesh-verbs.h pair_flush).  Whether anything was taken; a failure is the flow's (flow_lost). */
static int flow_receive(struct flow *link){
  if(!link->rings && !link->peer_alive)return 0;
  struct ibv_wc done[16];
  const struct mesh_queue *queue=link->provider.queues;
  int count=queue->poll(queue->completion,16,done);
  if(count<0){flow_lost(link,EIO,3);return 1;}
  link->provider.received_polled+=(uint64_t)count;
  int lost=0;
  for(int i=0;i<count && !lost;i++){
    uint64_t status_opcode;
    memcpy(&status_opcode,&done[i].status,sizeof status_opcode);
    if((uint32_t)status_opcode){flow_lost(link,(uint32_t)status_opcode,2);lost=1;continue;}
    if(!((status_opcode>>32)&IBV_WC_RECV))continue;
    /* #control: the peer's frame heard, its slot posted again */
    if((done[i].wr_id&~CONTROL_ID)==CONTROL_HEARD_TAG){
      const int error=frame_heard(link,(uint32_t)(done[i].wr_id&0xffff));
      if(error){flow_lost(link,error,1);lost=1;}
      continue;
    }
    struct prepared_receive *record=(void *)(uintptr_t)done[i].wr_id;
    struct receive_ring *ring=link->rings+record->queue;
    record_landed(link,record,ring,done[i].byte_len);
    int error=ring_advance(link,ring);
    if(error){flow_lost(link,error,1);lost=1;}
  }
  return count>0;
}

/* design/bridge-and-ipc.md#lanes: a port's lanes' pages: lane l's header frame h (its MESH_HEADERS send headers, then
   its two receive headers) and its discard block, as arena offsets; a span of them on the port's device; and their
   bytes through the window's mapping */
static uint64_t lane_offset(const struct port *port,uint32_t lane,uint32_t frame){
  return (uint64_t)port->lanes.page*M->pgsz+((uint64_t)lane*(MESH_HEADERS+2)+frame)*LANE_FRAME;
}
static uint64_t lane_discard(const struct port *port,uint32_t lane){
  const uint64_t block=(uint64_t)M->block*M->pgsz,headers=(uint64_t)MESH_LANES*(MESH_HEADERS+2)*LANE_FRAME;
  return (uint64_t)port->lanes.page*M->pgsz+(headers+block-1)/block*block+(uint64_t)lane*block;
}
static uint32_t lane_pages(void){
  const uint64_t block=(uint64_t)M->block*M->pgsz,headers=(uint64_t)MESH_LANES*(MESH_HEADERS+2)*LANE_FRAME;
  return (uint32_t)(((headers+block-1)/block*block+MESH_LANES*block+M->pgsz-1)/M->pgsz);
}
static struct ibv_sge lane_span(const struct port *port,uint64_t offset,uint64_t length){
  return wire_span(&port->device,(uint64_t)M->block*M->pgsz,offset,length);
}
static struct ibv_qp *lane_pair(const struct flow *link,uint32_t lane){
  return link->port->lanes.provider.queues?link->port->lanes.provider.queues[lane].pair:NULL;
}
/* a lane's failure, the lanes' (taken by the event loop: lanes_turn, lanes_reset) */
static void lanes_lost(struct port *port,int64_t code){
  int64_t none=0;
  atomic_compare_exchange_strong(&port->lanes.failed,&none,code?code:EIO);
  atomic_store_explicit(&port->lanes.up,0,memory_order_release);
  wake();
}
/* lane l's next header received, into its receive header of the parity of the headers it heard */
static int lane_listen(struct port *port,uint32_t lane){
  struct lanes *l=&port->lanes;
  struct ibv_sge span=lane_span(port,lane_offset(port,lane,MESH_HEADERS+(uint32_t)(l->received[lane]&1)),LANE_FRAME);
  struct ibv_recv_wr request={.wr_id=LANE_HEARD+lane,.sg_list=&span,.num_sge=1},*bad;
  const int error=l->provider.queues[lane].receive(l->provider.queues[lane].pair,&request,&bad);
  if(!error)l->provider.received_posted++;
  return error<0?-error:error;
}

/* design/bridge-and-ipc.md#lanes: a lane's SENDs, one pass of its port's send thread: of the flows on the port's lanes
   that run, each stream of this lane whose cell its producer released gives its chunks (none of a muted program's), as
   many as the lane's gate and the peer's receive room take (a cell's chunks may span headers), each run of them an
   entry of one header; lane 0's header then carries the port's control entries (#control: its riders' FAILs and
   RELEASEDs); the header is posted, then every entry's chunks in the header's order, so the peer, which posts the
   records a header names before it listens for the next, has each SEND's receive posted as it lands (credit flow
   control holds it until then, TN3205), and each flow that gave chunks marks the gate past them (`lane_mark`).  A
   header frame is used again once its SEND retired.  Whether anything was posted; a failure is the lanes'
   (lanes_lost). */
static int lane_send(struct port *port,uint32_t lane){
  struct lanes *l=&port->lanes;
  struct send_gate *gate=l->gate+lane;
  const uint64_t sent=l->sent[lane];
  const uint32_t h=(uint32_t)(sent%MESH_HEADERS);
  if(sent>=MESH_HEADERS && ((gate->retired-l->mark[lane][h])&MESH_GATE_MASK)>=(UINT64_C(1)<<47))return 0;
  const uint64_t flight=(gate->posted-gate->retired)&MESH_GATE_MASK;
  if(flight+2>gate->capacity)return 0;
  uint64_t room=MIN(gate->capacity-flight-1,(uint64_t)l->peer_room);
  struct lane_header *header=(void *)(wire.data+lane_offset(port,lane,h));
  struct ibv_send_wr *first[LANE_ENTRIES];
  struct flow *giver[LANE_ENTRIES];
  uint32_t entries=0,tracing=0;
  /* #ledger: the entries whose flows keep ledgers, each its event and whether it ends its cell */
  struct ledger_entry traced[LANE_ENTRIES];
  const uint32_t high=atomic_load_explicit(&port->high,memory_order_acquire);
  for(uint32_t n=0;n<high && entries<LANE_ENTRIES && room;n++){
    const uint32_t i=(l->cursor[lane]+n)%high;
    struct flow *f=atomic_load_explicit(&port->table[i],memory_order_acquire);
    if(!f || f->exclusive || !atomic_load_explicit(&f->progressing,memory_order_acquire))continue;
    const int flags=atomic_load_explicit(&f->ending,memory_order_acquire);
    for(uint32_t s=0;!(flags&FLOW_MUTED) && s<f->stream_count && entries<LANE_ENTRIES && room;s++){
      struct send_stream *stream=f->streams+s;
      while(stream->lane==lane && entries<LANE_ENTRIES && room){
        if(!stream->remaining){
          struct mesh_send *cell=stream->cell;
          if(atomic_load_explicit(&cell->ready,memory_order_acquire)<=stream->cycle)break;
          stream->next=&cell->request;stream->remaining=cell->chunks;
          if(f->ledger)ledger_seen(f->ledger,cell,stream->cycle);
        }
        struct ibv_send_wr *request=stream->next;
        uint32_t chunks=0;uint64_t bytes=0;
        for(;chunks<stream->remaining && request;chunks++,request=request->next){
          const uint32_t frames=(request->sg_list->length+4095)/4096;
          if(frames>room)break;
          room-=frames;bytes+=request->sg_list->length;
        }
        if(!chunks){room=0;break;}
        header->entry[entries]=(struct lane_entry){f->peer_slot,f->peer_generation,stream->cell->queue,chunks,bytes};
        giver[entries]=f;first[entries++]=stream->next;
        stream->next=request;stream->remaining-=chunks;
        if(f->ledger)traced[tracing++]=(struct ledger_entry){f->ledger,f->ledger->current[stream->cell->queue],!stream->remaining};
        if(stream->remaining)break;
        stream->cell=(struct mesh_send *)stream->cell->successor;stream->sent++;
        if(stream->cell==stream->first)stream->cycle++;
      }
    }
  }
  /* #control */
  for(uint64_t tail;!lane && entries<LANE_ENTRIES && (tail=atomic_load_explicit(&port->tell_tail,memory_order_relaxed))!=
        atomic_load_explicit(&port->tell_head,memory_order_acquire);){
    header->entry[entries]=port->tells[tail%MESH_TELLS];giver[entries]=NULL;first[entries++]=NULL;
    atomic_store_explicit(&port->tell_tail,tail+1,memory_order_release);
  }
  if(!entries)return 0;
  l->cursor[lane]=(l->cursor[lane]+1)%(high?high:1);
  header->magic=LANE_MAGIC;header->count=entries;header->sequence=sent;
  int (*post)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **)=l->provider.queues[lane].send;
  struct ibv_qp *pair=l->provider.queues[lane].pair;
  struct ibv_sge span=lane_span(port,lane_offset(port,lane,h),LANE_FRAME);
  struct ibv_send_wr posting={.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
  gate->posted+=1;gate->requests++;
  posting.wr_id=LANE_SENT|((uint64_t)lane<<48)|(gate->posted&MESH_GATE_MASK);
  int error=post(pair,&posting,&bad);
  if(!error)l->provider.sent_posted++;
  l->mark[lane][h]=gate->posted;l->sent[lane]++;
  for(uint32_t e=0;e<entries && !error;e++){
    struct ibv_send_wr *request=first[e];
    for(uint32_t c=0;c<header->entry[e].chunks && !error;c++,request=request->next){
      posting=*request;
      posting.next=NULL;posting.send_flags=IBV_SEND_SIGNALED;
      gate->posted+=(request->sg_list->length+4095)/4096;gate->requests++;
      posting.wr_id=LANE_SENT|((uint64_t)lane<<48)|(gate->posted&MESH_GATE_MASK);
      error=post(pair,&posting,&bad);
      if(!error)l->provider.sent_posted++;
    }
  }
  for(uint32_t e=0;e<entries;e++)if(giver[e])giver[e]->lane_mark[lane]=gate->posted;
  if(error)lanes_lost(port,error<0?-error:error);
  else if(tracing){
    const uint64_t at=ledger_ns();
    for(uint32_t e=0;e<tracing;e++)ledger_stamp(traced[e].ledger,traced[e].event,(int)traced[e].done,0,at);
  }
  return 1;
}
/* the lanes' SEND completions, each retiring its lane's frames (an error's too, every one counted taken: mesh-verbs.h
   pair_flush); whether any was taken */
static int lane_retire(struct port *port){
  struct lanes *l=&port->lanes;
  struct ibv_wc done[16];
  const int count=l->provider.queues[0].poll(l->provider.sent,16,done);
  if(count<0){lanes_lost(port,EIO);return 1;}
  l->provider.sent_polled+=(uint64_t)count;
  for(int i=0;i<count;i++){
    if(done[i].status)lanes_lost(port,(int64_t)done[i].status);
    struct send_gate *gate=l->gate+((done[i].wr_id>>48)&(MESH_LANES-1));
    gate->retired=done[i].wr_id&MESH_GATE_MASK;gate->completions++;
  }
  return count>0;
}
/* design/bridge-and-ipc.md#control: the lane riders of the port whose programs ended in order, each told drained once
   every cell it released is posted and the lanes' gates retired past its last SENDs (`lane_mark`) */
static void lanes_drained(struct port *port,uint32_t high){
  const struct lanes *l=&port->lanes;
  for(uint32_t i=0;i<high;i++){
    struct flow *f=atomic_load_explicit(&port->table[i],memory_order_acquire);
    if(!f || f->exclusive || !atomic_load_explicit(&f->progressing,memory_order_acquire) ||
       !(atomic_load_explicit(&f->ending,memory_order_acquire)&FLOW_ENDING) || atomic_load_explicit(&f->drained,memory_order_relaxed) ||
       !streams_idle(f))continue;
    int sent=1;
    for(uint32_t lane=0;lane<MESH_LANES && sent;lane++)sent=((l->gate[lane].retired-f->lane_mark[lane])&MESH_GATE_MASK)<(UINT64_C(1)<<47);
    if(sent)flow_drained(f);
  }
}
/* design/bridge-and-ipc.md#lanes: lane l's header heard: each entry's records posted, the next `chunks` of the named
   ring of the named flow while that flow is the one the index and generation name and takes its lanes' records, else
   as many discards of the same frames (a flow that stopped: its rows are no longer its own), then the next header
   listened for.  A header out of sequence or past the room the lane told, or an entry whose frames are not its ring's
   records', is a protocol error: the lanes' where the header is, the flow's where its records are.  1 where the lanes
   failed */
static int lane_heard(struct port *port,uint32_t lane){
  struct lanes *l=&port->lanes;
  const struct lane_header *header=(const void *)(wire.data+lane_offset(port,lane,MESH_HEADERS+(uint32_t)(l->received[lane]&1)));
  const uint64_t block=(uint64_t)M->block*M->pgsz;
  uint64_t total=0;
  int error=header->magic!=LANE_MAGIC || header->sequence!=l->received[lane] || header->count>LANE_ENTRIES?EPROTO:0;
  for(uint32_t e=0;!error && e<header->count;e++){
    const struct lane_entry *entry=header->entry+e;
    /* #control: an entry of no chunks is a control frame, its kind in `ring`'s top byte */
    if(!entry->chunks){if(!(entry->ring>>24) || (entry->ring>>24)>CONTROL_RELEASED)error=EPROTO;}
    else if(entry->bytes<=(uint64_t)(entry->chunks-1)*block || entry->bytes>(uint64_t)entry->chunks*block)error=EPROTO;
    else total+=(uint64_t)(entry->chunks-1)*(block/4096)+(entry->bytes-(uint64_t)(entry->chunks-1)*block+4095)/4096;
  }
  if(!error && total>l->room)error=EPROTO;
  if(error){lanes_lost(port,error);return 1;}
  int (*post)(struct ibv_qp *,struct ibv_recv_wr *,struct ibv_recv_wr **)=l->provider.queues[lane].receive;
  struct ibv_qp *pair=l->provider.queues[lane].pair;
  struct ibv_recv_wr *bad;
  for(uint32_t e=0;e<header->count && !error;e++){
    const struct lane_entry entry=header->entry[e];
    struct flow *f=entry.slot<MESH_FLOWS?atomic_load_explicit(&port->table[entry.slot],memory_order_acquire):NULL;
    /* #control: applied to the flow the index and generation name, else dropped (a flow gone: its FAIL is moot) */
    if(!entry.chunks){
      const uint32_t kind=entry.ring>>24;
      if(f && !f->exclusive && f->generation==entry.generation)control_apply(f,kind,(uint32_t)(entry.bytes>>32),(uint32_t)entry.bytes);
      continue;
    }
    struct receive_ring *ring=f && !f->exclusive && f->generation==entry.generation && entry.ring<(uint32_t)f->qps &&
      atomic_load_explicit(&f->receiving,memory_order_acquire) && f->rings && ring_lane(entry.ring,(uint32_t)f->qps/f->queues)==lane?
      f->rings+entry.ring:NULL;
    for(uint32_t c=0;ring && c<entry.chunks;c++){
      const uint64_t length=MIN(block,entry.bytes-(uint64_t)c*block);
      if(!ring->count || ring->first[(ring->posted+c)%ring->count].frames!=(length+4095)/4096){flow_lost(f,EPROTO,2);ring=NULL;}
    }
    for(uint32_t c=0;c<entry.chunks && !error;c++){
      if(ring){
        struct prepared_receive *record=ring->first+ring->posted%ring->count;
        record->cycle=(uint32_t)(ring->posted/ring->count);
        error=post(pair,&record->request,&bad);
        ring->outstanding+=record->frames;ring->posted++;
        atomic_fetch_add_explicit(&f->lane_frames,record->frames,memory_order_relaxed);
      }else{
        struct ibv_sge span=lane_span(port,lane_discard(port,lane),MIN(block,entry.bytes-(uint64_t)c*block));
        struct ibv_recv_wr request={.wr_id=LANE_DISCARD,.sg_list=&span,.num_sge=1};
        error=post(pair,&request,&bad);
      }
      if(!error){l->records++;l->provider.received_posted++;}
    }
  }
  if(!error){l->received[lane]++;error=lane_listen(port,lane);}
  if(error){lanes_lost(port,error<0?-error:error);return 1;}
  return 0;
}
/* design/bridge-and-ipc.md#lanes: the lanes' receive completions, one pass of the port's receive thread: a header's
   records posted (lane_heard), a record's landing storing its word as an exclusive flow's does; whether anything
   landed */
static int lane_receive(struct port *port){
  struct lanes *l=&port->lanes;
  struct ibv_wc done[16];
  const int count=l->provider.queues[0].poll(l->provider.completion,16,done);
  if(count<0){lanes_lost(port,EIO);return 1;}
  l->provider.received_polled+=(uint64_t)count;
  for(int i=0;i<count;i++){
    if(done[i].status){lanes_lost(port,(int64_t)done[i].status);return 1;}
    const uint64_t id=done[i].wr_id;
    if(id>=LANE_HEARD && id<LANE_HEARD+MESH_LANES){if(lane_heard(port,(uint32_t)(id-LANE_HEARD)))return 1;continue;}
    l->records--;
    if(id==LANE_DISCARD)continue;
    struct prepared_receive *record=(void *)(uintptr_t)id;
    struct flow *link=record->link;
    /* #ledger: stored before the frames are given back, which flow_release reads before it writes the ledger */
    record_landed(link,record,link->rings+record->queue,done[i].byte_len);
    /* #flows: a stopped rider's last record landed wakes its release (FLOW_RELEASING) */
    if(atomic_fetch_sub_explicit(&link->lane_frames,record->frames,memory_order_acq_rel)==record->frames &&
       (atomic_load_explicit(&link->ending,memory_order_acquire)&FLOW_RELEASING))wake();
  }
  return count>0;
}

/* design/bridge-dataflow.md#stripes: a stripe flow's failure, the event loop's to reset (stripes_turn) */
static void stripes_lost(struct port *port,int64_t code){
  int64_t none=0;
  atomic_compare_exchange_strong(&port->stripes.failed,&none,code?code:EIO);
  atomic_store_explicit(&port->stripes.up,0,memory_order_release);
  wake();
}
/* a stripe's next piece from window offset `offset`, `remaining` bytes left: at most the flow's piece and never past a
   registered extent (MESH_REGION), so both ends cut a stripe alike and each piece is one SGE of one registration */
static uint32_t stripe_length(uint64_t offset,uint64_t remaining,uint32_t piece){
  const uint64_t room=MESH_REGION-offset%MESH_REGION,most=remaining<piece?remaining:piece;
  return (uint32_t)(most<room?most:room);
}
static struct ibv_sge stripe_span(const struct port *port,uint64_t offset,uint32_t length){
  const uint64_t at=offset<wire.length?offset:0;
  return (struct ibv_sge){.addr=(uintptr_t)port->device.wire+offset,.length=length,.lkey=port->device.regions[at/MESH_REGION]->lkey};
}
/* the window offset of descriptor slot `slot` of the port's stripe flow (received 0 to MESH_DESCRIPTORS - 1, sent the
   next MESH_DESCRIPTORS) */
static uint64_t stripe_descriptor(const struct port *port,uint32_t slot){
  return (uint64_t)port->stripes.page*M->pgsz+(uint64_t)slot*sizeof(struct mesh_descriptor);
}
/* a word of this node's region a stripe stores (`word` a region offset in the window, 8-byte aligned; 0 none) */
static int stripe_word_valid(uint64_t word){ return !word || (!(word&7) && word>=M->data_off && word<=M->length-sizeof(uint64_t)); }
static void stripe_word(uint64_t word,uint64_t value){
  if(word && stripe_word_valid(word))atomic_store_explicit((_Atomic uint64_t *)((char *)M+word),value,memory_order_release);
}
/* a request failed as a value: its writer's word UINT64_MAX, its ring's failures counted and the code kept (a ring whose
   client retired meanwhile is left alone: its rows stay held while the request is in flight, clients_turn) */
static void stripe_failed(uint64_t ring,uint64_t owner,uint64_t word,int64_t code){
  if(!owned(owner))return;
  stripe_word(word,UINT64_MAX);
  if(ring<M->data_off || ring>M->length-sizeof(struct mesh_requests))return;
  struct mesh_requests *q=(void *)((char *)M+ring);
  atomic_fetch_add_explicit(&q->failed,1,memory_order_relaxed);
  atomic_store_explicit(&q->code,code,memory_order_release);
}
/* design/bridge-dataflow.md#stripes: one pass of the port's send thread over its stripe flow (mesh 8fc27cbe peer_serve's
   SEND half): its SEND completions taken (a stripe's last piece completing stores its writer's word), then the next
   request of each registered ring naming the port's peer taken (compare-exchange of the ring's `taken`, so the threads
   of every port toward that node take each request once), its descriptor posted on the control pair, then its pieces
   on the data pair while the pair's frames and the piece ring hold them; a request of a range the windows do not hold
   or a word that is no word fails as a value (EINVAL).  Whether anything moved; a failure is the flow's (stripes_lost) */
static int stripes_send(struct port *port){
  struct stripes *s=&port->stripes;
  const struct mesh_queue *control=s->provider.queues,*data=s->provider.queues+1;
  struct ibv_wc done[16];
  const int count=control->poll(s->provider.sent,16,done);
  if(count<0){stripes_lost(port,EIO);return 1;}
  s->provider.sent_polled+=(uint64_t)count;
  for(int i=0;i<count;i++){
    if(done[i].status){stripes_lost(port,(int64_t)done[i].status);return 1;}
    if((done[i].wr_id&~STRIPE_SLOT)==STRIPE_DESCRIPTOR){s->descriptors_done++;continue;}
    const struct stripe_piece *p=s->sent+(done[i].wr_id&STRIPE_SLOT)%MESH_STRIPE_CHUNKS;
    s->send_frames-=p->frames;s->send_tail++;
    if(!p->last)continue;
    if(owned(p->owner))stripe_word(p->word,p->value);
    atomic_fetch_add_explicit(&s->written,1,memory_order_relaxed);
    atomic_fetch_sub_explicit(&s->flight,1,memory_order_release);
  }
  int moved=count>0;
  _Atomic uint64_t *rings=mesh_rings(M);
  const uint64_t window=atomic_load_explicit(&s->window,memory_order_relaxed);
  for(uint32_t scanned=0;scanned<MESH_RINGS;scanned++){
    if(!s->out.active){
      if(s->descriptors_sent-s->descriptors_done>=MESH_DESCRIPTORS)break;
      const uint32_t r=(s->cursor+scanned)%MESH_RINGS;
      const uint64_t at=atomic_load_explicit(rings+r,memory_order_acquire);
      if(!at || at<M->data_off || at>M->length-sizeof(struct mesh_requests))continue;
      struct mesh_requests *q=(void *)((char *)M+at);
      const uint64_t entries=q->entries;
      if(!entries || (entries&(entries-1)) || entries>(M->length-at-sizeof *q)/sizeof(struct mesh_stripe))continue;
      uint64_t k=atomic_load_explicit(&q->taken,memory_order_acquire);
      const struct mesh_stripe *e=q->request+(k&(entries-1));
      if(atomic_load_explicit(&e->ready,memory_order_acquire)!=k+1 || e->node!=port->peer)continue;
      const struct mesh_descriptor d={.offset=e->offset,.bytes=e->bytes,.word=e->word,.value=e->value};
      const uint64_t owner=q->owner;
      if(!atomic_compare_exchange_strong_explicit(&q->taken,&k,k+1,memory_order_acq_rel,memory_order_relaxed))continue;
      s->cursor=(r+1)%MESH_RINGS;moved=1;
      atomic_fetch_add_explicit(&s->flight,1,memory_order_acq_rel);
      if(d.offset>wire.length || d.bytes>wire.length-d.offset || d.offset>window || d.bytes>window-d.offset || !stripe_word_valid(d.word)){
        stripe_failed(at,owner,d.word,EINVAL);
        atomic_fetch_sub_explicit(&s->flight,1,memory_order_release);
        continue;
      }
      const uint32_t slot=(uint32_t)(s->descriptors_sent%MESH_DESCRIPTORS);
      const uint64_t offset=stripe_descriptor(port,MESH_DESCRIPTORS+slot);
      memcpy(wire.data+offset,&d,sizeof d);
      struct ibv_sge span=stripe_span(port,offset,sizeof d);
      struct ibv_send_wr posting={.wr_id=STRIPE_DESCRIPTOR|slot,.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
      const int error=control->send(control->pair,&posting,&bad);
      if(!error)s->provider.sent_posted++;
      if(error){
        stripe_failed(at,owner,d.word,error<0?-error:error);
        atomic_fetch_sub_explicit(&s->flight,1,memory_order_release);
        stripes_lost(port,error<0?-error:error);
        return 1;
      }
      s->descriptors_sent++;
      s->out=(struct stripe_cut){.d=d,.owner=owner,.ring=at,.active=1};
    }
    while(s->out.active){
      const uint64_t offset=s->out.d.offset+s->out.done;
      const uint32_t length=stripe_length(offset,s->out.d.bytes-s->out.done,s->piece),frames=length?(length+4095)/4096:1;
      if(s->send_frames+frames>s->send_capacity || s->send_head-s->send_tail>=MESH_STRIPE_CHUNKS)break;
      const int last=s->out.done+length>=s->out.d.bytes;
      const uint32_t slot=(uint32_t)(s->send_head%MESH_STRIPE_CHUNKS);
      s->sent[slot]=(struct stripe_piece){.word=last?s->out.d.word:0,.value=s->out.d.value,.owner=s->out.owner,.ring=s->out.ring,
        .frames=frames,.last=(uint32_t)last};
      struct ibv_sge span=stripe_span(port,offset,length);
      struct ibv_send_wr posting={.wr_id=STRIPE_DATA|slot,.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
      const int error=data->send(data->pair,&posting,&bad);
      if(error){stripes_lost(port,error<0?-error:error);return 1;}
      s->provider.sent_posted++;
      s->send_frames+=frames;s->send_head++;s->out.done+=length;moved=1;
      if(last)s->out.active=0;
    }
    if(s->out.active)break;
  }
  return moved;
}
/* a descriptor slot's receive posted again, once the stripe it named has every piece's receive posted */
static int stripe_listen(struct port *port,uint32_t slot){
  struct stripes *s=&port->stripes;
  struct ibv_sge span=stripe_span(port,stripe_descriptor(port,slot),sizeof(struct mesh_descriptor));
  struct ibv_recv_wr request={.wr_id=STRIPE_DESCRIPTOR|slot,.sg_list=&span,.num_sge=1},*bad;
  const int error=s->provider.queues[0].receive(s->provider.queues[0].pair,&request,&bad);
  if(!error)s->provider.received_posted++;
  return error<0?-error:error;
}
/* design/bridge-dataflow.md#stripes: one pass of the port's receive thread over its stripe flow (peer_serve's receive
   half): its receive completions taken (a descriptor landed queued by its slot; a stripe's last piece landing stores
   its word here), then each landed descriptor's pieces' receives posted at its offset, in the order the descriptors
   landed, while the data pair's frames and the piece ring hold them, and its slot's receive posted again once they all
   are (so no more descriptors are in flight than slots: credit flow control holds the peer's next one until then,
   TN3205).  A descriptor naming bytes past this node's window or a word that is no word is the peer's protocol error
   (EPROTO).  Whether anything moved; a failure is the flow's (stripes_lost) */
static int stripes_receive(struct port *port){
  struct stripes *s=&port->stripes;
  const struct mesh_queue *data=s->provider.queues+1;
  struct ibv_wc done[16];
  const int count=s->provider.queues[0].poll(s->provider.completion,16,done);
  if(count<0){stripes_lost(port,EIO);return 1;}
  s->provider.received_polled+=(uint64_t)count;
  if(count)atomic_thread_fence(memory_order_acquire);
  for(int i=0;i<count;i++){
    if(done[i].status){stripes_lost(port,(int64_t)done[i].status);return 1;}
    const uint64_t slot=done[i].wr_id&STRIPE_SLOT;
    if((done[i].wr_id&~STRIPE_SLOT)==STRIPE_DESCRIPTOR){s->arrived[s->arrived_head++%MESH_DESCRIPTORS]=(uint32_t)slot;continue;}
    const struct stripe_piece *p=s->posted+slot%MESH_STRIPE_CHUNKS;
    s->receive_frames-=p->frames;s->receive_tail++;
    if(!p->last)continue;
    stripe_word(p->word,p->value);
    atomic_fetch_add_explicit(&s->landed,1,memory_order_relaxed);
  }
  int moved=count>0;
  for(;;){
    if(!s->in.active){
      if(s->arrived_tail==s->arrived_head)break;
      const uint32_t slot=s->arrived[s->arrived_tail%MESH_DESCRIPTORS];
      memcpy(&s->in.d,wire.data+stripe_descriptor(port,slot),sizeof s->in.d);
      const struct mesh_descriptor *d=&s->in.d;
      if(d->offset>wire.length || d->bytes>wire.length-d->offset || !stripe_word_valid(d->word)){stripes_lost(port,EPROTO);return 1;}
      s->in.done=0;s->in.slot=slot;s->in.active=1;
    }
    const uint64_t offset=s->in.d.offset+s->in.done;
    const uint32_t length=stripe_length(offset,s->in.d.bytes-s->in.done,s->piece),frames=length?(length+4095)/4096:1;
    if(s->receive_frames+frames>s->receive_capacity || s->receive_head-s->receive_tail>=MESH_STRIPE_CHUNKS)break;
    const int last=s->in.done+length>=s->in.d.bytes;
    const uint32_t slot=(uint32_t)(s->receive_head%MESH_STRIPE_CHUNKS);
    s->posted[slot]=(struct stripe_piece){.word=last?s->in.d.word:0,.value=s->in.d.value,.frames=frames,.last=(uint32_t)last};
    struct ibv_sge span=stripe_span(port,offset,length);
    struct ibv_recv_wr request={.wr_id=STRIPE_DATA|slot,.sg_list=&span,.num_sge=1},*bad;
    int error=data->receive(data->pair,&request,&bad);
    if(error){stripes_lost(port,error<0?-error:error);return 1;}
    s->provider.received_posted++;
    s->receive_frames+=frames;s->receive_head++;s->in.done+=length;moved=1;
    if(!last)continue;
    s->in.active=0;s->arrived_tail++;
    if((error=stripe_listen(port,s->in.slot))){stripes_lost(port,error);return 1;}
  }
  return moved;
}

/* design/bridge-dataflow.md#streams: a stream's failure, the event loop's to end (streams_turn) */
static void stream_lost(struct port *port,int64_t code){
  int64_t none=0;
  atomic_compare_exchange_strong(&port->stream.failed,&none,code?code:EIO);
  atomic_store_explicit(&port->stream.up,0,memory_order_release);
  wake();
}
/* the window offset of a stream's page */
static uint64_t stream_offset(const struct port *port,uint64_t page){
  return (uint64_t)port->stream.page*M->pgsz+page*port->stream.page_bytes;
}
/* the stream's next `n` pages from `posted`, chained as mesh-verbs-stream.c chained them: SENDs, or receives */
static int stream_post(struct port *port,uint32_t n){
  struct stream *s=&port->stream;
  for(uint32_t i=0;i<n;i++){
    const uint64_t page=s->posted+i;
    s->spans[i]=stripe_span(port,stream_offset(port,page),s->page_bytes);
    if(s->sending)s->sends[i]=(struct ibv_send_wr){.wr_id=page,.next=i+1<n?s->sends+i+1:NULL,.sg_list=s->spans+i,.num_sge=1,
      .opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED};
    else s->receives[i]=(struct ibv_recv_wr){.wr_id=page,.next=i+1<n?s->receives+i+1:NULL,.sg_list=s->spans+i,.num_sge=1};
  }
  const struct mesh_queue *queue=s->provider.queues;
  struct ibv_send_wr *bad_send;struct ibv_recv_wr *bad_receive;
  bad_send=NULL;bad_receive=NULL;
  const int error=s->sending?queue->send(queue->pair,s->sends,&bad_send):queue->receive(queue->pair,s->receives,&bad_receive);
  const uint64_t taken=!error?n:s->sending?(bad_send?(uint64_t)(bad_send-s->sends):0):(bad_receive?(uint64_t)(bad_receive-s->receives):0);
  if(s->sending)s->provider.sent_posted+=taken;else s->provider.received_posted+=taken;
  if(error)return error<0?-error:error;
  if(!s->posted)s->own.begin_ns=ledger_ns();
  s->posted+=n;s->own.posts++;
  return 0;
}
/* design/bridge-dataflow.md#streams: one pass of the stream at its end's progress thread (the send thread where it
   sends, the receive thread where it receives): its completions taken (a SEND's retiring its page; a receive's landing
   its page, in posting order and of a whole page, its count's first, tenth, ninetieth percent and last landing timed),
   then its next pages posted while fewer than `window` are in flight; the last completion finishing it */
static int stream_pass(struct port *port){
  struct stream *s=&port->stream;
  const struct mesh_queue *queue=s->provider.queues;
  const int count=queue->poll(s->sending?s->provider.sent:s->provider.completion,(int)s->window,s->completions);
  s->own.polls++;
  if(count<0){stream_lost(port,EIO);return 1;}
  if(s->sending)s->provider.sent_polled+=(uint64_t)count;else s->provider.received_polled+=(uint64_t)count;
  if(!count)s->own.empty_polls++;
  for(int i=0;i<count;i++){
    const struct ibv_wc *done=s->completions+i;
    if(done->status){stream_lost(port,(int64_t)done->status);return 1;}
    if(!s->sending && done->byte_len!=s->page_bytes){stream_lost(port,EPROTO);return 1;}
    s->own.misordered+=done->wr_id!=s->completed+(uint64_t)i;
  }
  if(count){
    const uint64_t at=ledger_ns();
    s->completed+=(uint64_t)count;
    if(!s->own.first_ns){s->own.first_ns=at;s->own.first_count=s->completed;}
    if(!s->own.low_count && s->completed>=s->count/10){s->own.low_ns=at;s->own.low_count=s->completed;}
    if(!s->own.high_count && s->completed>=9*(uint64_t)s->count/10){s->own.high_ns=at;s->own.high_count=s->completed;}
    s->own.last_ns=at;s->own.completed=s->completed;
  }
  const uint64_t room=s->window-(s->posted-s->completed),left=s->count-s->posted,n=room<left?room:left;
  if(n){
    const int error=stream_post(port,(uint32_t)n);
    if(error){stream_lost(port,error);return 1;}
  }
  if(s->completed==s->count){atomic_store_explicit(&s->up,0,memory_order_release);atomic_store_explicit(&s->finished,1,memory_order_release);wake();}
  return count || n;
}

/* design/bridge-and-ipc.md#progress: the epoch a progress thread passed told, its waiter woken (port_settle) */
static void port_told(struct port *port,int side,uint64_t epoch,uint64_t *told){
  atomic_store_explicit(&port->seen[side],epoch,memory_order_release);*told=epoch;
  os_sync_wake_by_address_all(&port->seen[side],sizeof port->seen[side],OS_SYNC_WAKE_BY_ADDRESS_NONE);
}
/* design/bridge-and-ipc.md#progress: one pass of a port's progress thread, `side` 0 its receives' and 1 its SENDs':
   every exclusive flow in the port's table it serves (while the flow is progressing) and the port's lanes while they are
   up.  The pass reads the table's epoch, serves each, and tells the epoch it began at, so a flow taken out of the table
   is read by neither thread once both tell a later epoch (port_unlist).  Whether anything moved; *any where something is
   the thread's to look for: a flow it serves (a SEND side progressing; an exclusive flow's receives, or a lane rider
   taking its headers' records), a lane's records posted and not landed, or its SENDs in flight; nothing lands on a lane
   for a flow this end has not listed, since the peer sends only after this end's POSTED, which follows the listing */
static int port_pass(struct port *port,int side,int *any,uint64_t *told){
  const uint64_t epoch=atomic_load(&port->epoch);
  int moved=0;
  *any=0;
  const uint32_t high=atomic_load_explicit(&port->high,memory_order_acquire);
  int riders=0;
  for(uint32_t i=0;i<high;i++){
    struct flow *f=atomic_load_explicit(&port->table[i],memory_order_acquire);
    if(!f)continue;
    const int progressing=atomic_load_explicit(&f->progressing,memory_order_acquire);
    if(f->exclusive && progressing){*any=1;moved|=side?flow_send(f):flow_receive(f);}
    else if(!f->exclusive){riders=1;*any|=side?progressing:atomic_load_explicit(&f->receiving,memory_order_acquire);}
  }
  /* #lanes: served while a flow on the port rides them or their SENDs or records are in flight (nothing lands on a lane
     for a flow this end has not listed) */
  struct lanes *l=&port->lanes;
  if(atomic_load_explicit(&l->up,memory_order_acquire)){
    if(side){
      /* #control: the port's FAILs and RELEASEDs queued are lane 0's to send, the riders they name gone or not */
      int flight=atomic_load_explicit(&port->tell_head,memory_order_acquire)!=atomic_load_explicit(&port->tell_tail,memory_order_relaxed);
      for(uint32_t lane=0;lane<MESH_LANES;lane++)flight|=((l->gate[lane].posted-l->gate[lane].retired)&MESH_GATE_MASK)!=0;
      if(riders || flight){
        for(uint32_t lane=0;lane<MESH_LANES;lane++)moved|=lane_send(port,lane);
        moved|=lane_retire(port);
        if(riders)lanes_drained(port,high);
        flight=0;
        for(uint32_t lane=0;lane<MESH_LANES;lane++)flight|=((l->gate[lane].posted-l->gate[lane].retired)&MESH_GATE_MASK)!=0;
      }
      *any|=flight;
    }else if(riders || l->records){moved|=lane_receive(port);*any|=l->records!=0;}
  }
  /* design/bridge-dataflow.md#stripes: the port's stripe flow while it is up, its receives always posted and any ring
     able to name its peer at any time */
  if(atomic_load_explicit(&port->stripes.up,memory_order_acquire)){*any=1;moved|=side?stripes_send(port):stripes_receive(port);}
  /* design/bridge-dataflow.md#streams: a stream, at the thread of its end's side */
  if(atomic_load_explicit(&port->stream.up,memory_order_acquire) && side==port->stream.sending){*any=1;moved|=stream_pass(port);}
  /* design/bridge-and-ipc.md#progress: told only when it moved (port_settle waits on it), so the line the two threads
     read every pass (`high`, `epoch`) is not written by either each pass */
  if(epoch!=*told)port_told(port,side,epoch,told);
  return moved;
}
/* design/bridge-and-ipc.md#progress: a port's progress thread: passes while anything moves, and with nothing moved for
   MESH_IDLE_NS it parks on its port's bell, looking again every MESH_IDLE_POLL_NS while anything is its to look for
   (port_pass: a completion queue polled, which no event of the provider's ends), else waiting on the bell alone, and at
   once when the bell rings: a client's group of a program on the port starting (mesh-call.h mesh_transfers_wake), a
   landing that released one of the port's SEND cells while its send thread was parked (send_wake), a flow listed or
   unlisted, a control entry queued, the lanes' or a stripe flow's or a stream's change, the guard, the bridge stopping.
   A thread tells it is parked, then looks once more before it waits, so a release whose send_wake read it unparked is
   served by that look */
#define MESH_IDLE_PASSES 4096u
struct progress { struct port *port; int side; };
static void *port_progress(void *argument){
  struct port *port=((struct progress *)argument)->port;
  const int side=((struct progress *)argument)->side;
  free(argument);
  pthread_setname_np(side?"mesh.rdma.send":"mesh.rdma.receive");
  _Atomic uint64_t *bell=&M->bell[port->index];
  uint64_t quiet=0,told=atomic_load_explicit(&port->seen[side],memory_order_relaxed);
  uint32_t empty=0;
  int resting=0,any=0;
  while(!atomic_load_explicit(&port->closing,memory_order_acquire)){
    /* design/bridge-dataflow.md#stripes: while the port's guard stands, no pass (no post on any queue pair of its device)
       and no flow read, so the epoch is told as it moves (port_settle inside a guarded job returns) */
    const uint64_t guard=atomic_load_explicit(&port->guard,memory_order_acquire);
    if(guard&1){
      atomic_store_explicit(&port->guarded[side],guard,memory_order_release);
      os_sync_wake_by_address_all(&port->guarded[side],sizeof port->guarded[side],OS_SYNC_WAKE_BY_ADDRESS_NONE);
      for(;;){
        const uint64_t rung=atomic_load_explicit(bell,memory_order_acquire),epoch=atomic_load_explicit(&port->epoch,memory_order_acquire);
        if(epoch!=told)port_told(port,side,epoch,&told);
        if(atomic_load_explicit(&port->guard,memory_order_acquire)!=guard || atomic_load_explicit(&port->closing,memory_order_acquire))break;
        os_sync_wait_on_address(bell,rung,sizeof *bell,OS_SYNC_WAIT_ON_ADDRESS_SHARED);
      }
      empty=0;quiet=0;resting=0;
      continue;
    }
    const uint64_t rung=atomic_load_explicit(bell,memory_order_acquire);
    if(port_pass(port,side,&any,&told)){empty=0;quiet=0;resting=0;continue;}
    if(any && !resting){
      if(++empty<MESH_IDLE_PASSES)continue;
      empty=0;
      const uint64_t at=ledger_ns();
      if(!quiet)quiet=at;
      if(at-quiet<MESH_IDLE_NS)continue;
    }
    atomic_store_explicit(&port->parked[side],1,memory_order_relaxed);
    atomic_thread_fence(memory_order_seq_cst);
    const uint64_t last=atomic_load_explicit(bell,memory_order_acquire);
    /* a guard raised or a close since the loop's head rang the bell before `last` was read: taken now, not slept past */
    if(port_pass(port,side,&any,&told) || (atomic_load_explicit(&port->guard,memory_order_acquire)&1) ||
       atomic_load_explicit(&port->closing,memory_order_acquire)){
      atomic_store_explicit(&port->parked[side],0,memory_order_relaxed);empty=0;quiet=0;resting=0;continue;
    }
    resting=1;
    if(any)os_sync_wait_on_address_with_timeout(bell,last,sizeof *bell,OS_SYNC_WAIT_ON_ADDRESS_SHARED,OS_CLOCK_MACH_ABSOLUTE_TIME,MESH_IDLE_POLL_NS);
    else os_sync_wait_on_address(bell,last,sizeof *bell,OS_SYNC_WAIT_ON_ADDRESS_SHARED);
    atomic_store_explicit(&port->parked[side],0,memory_order_relaxed);
    if(atomic_load_explicit(bell,memory_order_acquire)!=rung){resting=0;quiet=0;empty=0;}
  }
  return NULL;
}
/* the port's bell rung: its parked progress threads look again */
static void port_bell(struct port *port){ mesh_notify(&M->bell[port->index]); }
/* design/bridge-and-ipc.md#progress: the port's two progress threads started, once; 0, or pthread_create's error */
static int port_threads(struct port *port){
  int error=0;
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_set_qos_class_np(&attributes,QOS_CLASS_USER_INTERACTIVE,0);
  while(port->threads<2 && !error){
    struct progress *progress=malloc(sizeof *progress);
    if(!progress){error=ENOMEM;break;}
    *progress=(struct progress){port,port->threads};
    atomic_store(&port->seen[port->threads],atomic_load(&port->epoch));
    if((error=pthread_create(&port->thread[port->threads],&attributes,port_progress,progress)))free(progress);
    else port->threads++;
  }
  pthread_attr_destroy(&attributes);
  return error;
}
/* design/bridge-and-ipc.md#progress: a flow in its port's table at its index, on the port's queue, its SENDs (and an
   exclusive flow's receives) served where `progressing`, a lane's headers filling its rings where `receiving`; 0, or
   the threads' error */
static int port_list(struct flow *f,int progressing,int receiving){
  struct port *port=f->port;
  const int error=port_threads(port);
  if(error)return error;
  if(progressing)atomic_store_explicit(&f->progressing,1,memory_order_release);
  if(receiving)atomic_store_explicit(&f->receiving,1,memory_order_release);
  atomic_store_explicit(&port->table[f->slot],f,memory_order_release);
  f->listed=1;
  port_bell(port);
  return 0;
}
/* the port's table read by neither progress thread past a change made before: both tell an epoch past it, each woken by
   the bell and waited on at the word it tells (port_told) */
static void port_settle(struct port *port){
  const uint64_t epoch=atomic_fetch_add(&port->epoch,1)+1;
  port_bell(port);
  for(int side=0;side<port->threads;side++)
    for(uint64_t seen;(seen=atomic_load_explicit(&port->seen[side],memory_order_acquire))<epoch;)
      os_sync_wait_on_address(&port->seen[side],seen,sizeof seen,OS_SYNC_WAIT_ON_ADDRESS_NONE);
}
/* design/bridge-dataflow.md#stripes, design/RDMA-KERNEL-RECOVERY.md "The stripe writes of October 1": the guard of a
   port's device, on the port's queue, around a change of its verbs objects (queue pairs and completion queues made,
   connected or destroyed): raised, it returns once each started progress thread holds (it told the guard's value before
   its next pass, and passes no more while the guard stands), so no queue pair of the device is posted on while another
   is made, changed or destroyed (Apple's provider: a post on a long-lived queue pair across another's teardown stores
   to the device's shared doorbell mapping after it moved, tbt_post_recv); lowered, the threads go on.  It stands for a
   change of the stripe flow's own (`always`) and, while the device holds a stripe flow's queue pairs (whose posts
   outlive every other's teardown), for every flow's and the lanes' too; with none, HEAD's verbs work runs unguarded as
   it did */
static void port_guard(struct port *port,int raise){
  if(!raise){atomic_fetch_add_explicit(&port->guard,1,memory_order_release);port_bell(port);return;}
  const uint64_t guard=atomic_fetch_add_explicit(&port->guard,1,memory_order_acq_rel)+1;
  port_bell(port);
  for(int side=0;side<port->threads;side++)
    for(uint64_t held;(held=atomic_load_explicit(&port->guarded[side],memory_order_acquire))!=guard;)
      os_sync_wait_on_address(&port->guarded[side],held,sizeof held,OS_SYNC_WAIT_ON_ADDRESS_NONE);
}
static int guard_raise(struct port *port,int always){
  if(!always && !port->stripes.provider.queues && !port->stream.provider.queues)return 0;
  port_guard(port,1);
  return 1;
}
static void guard_lower(struct port *port,int raised){ if(raised)port_guard(port,0); }
/* design/bridge-and-ipc.md#progress: a flow's SENDs (and an exclusive flow's receives) no longer served, on its port's
   queue, returning once neither thread is serving them (a flow on the lanes keeps taking its headers' records) */
static void port_quiet(struct flow *f){
  atomic_store_explicit(&f->progressing,0,memory_order_release);
  if(f->listed)port_settle(f->port);
}
/* design/bridge-and-ipc.md#progress: a flow out of its port's table, on the port's queue, returning once neither
   progress thread can read it (a lane's records it posted before land in its rows: flow_release waits on them) */
static void port_unlist(struct flow *f){
  atomic_store_explicit(&f->progressing,0,memory_order_release);
  atomic_store_explicit(&f->receiving,0,memory_order_release);
  if(!f->listed)return;
  struct flow *held=f;
  atomic_compare_exchange_strong(&f->port->table[f->slot],&held,NULL);
  port_settle(f->port);
  f->listed=0;
}
/* design/bridge-and-ipc.md#progress: the port's progress threads stopped (the bridge stopping, every flow released) */
static void port_still(struct port *port){
  atomic_store_explicit(&port->closing,1,memory_order_release);
  port_bell(port);
  while(port->threads)pthread_join(port->thread[--port->threads],NULL);
}

/* design/bridge-and-ipc.md#flows: a flow's verbs work, on its port's queue: an exclusive flow's queue pairs made; ready
   to receive, its receives prepared (an exclusive flow's posted, a lane rider's taking its headers' records); its
   SENDs prepared, an exclusive flow's queue pairs ready to send, and served by its port's progress threads; or a side
   whose lists arrived, the flow's SENDs unserved while it is prepared and served again where they were.  0, or the
   error */
static int flow_work(struct flow *f,int job){
  int error=0;
  if(job==JOB_PAIR)return f->exclusive?pair_up(&f->provider,f->qps+f->alive):0;
  ledger_open(f);
  const int ran=atomic_load_explicit(&f->progressing,memory_order_acquire);
  if(job==JOB_CHANGE)port_quiet(f);
  if(job==JOB_RECEIVE && f->exclusive){
    /* #control: its queue pair connected and its receives posted where both ends made one, else left unconnected */
    f->provider.count=f->qps+(f->alive && f->peer_alive);
    error=pair_receive(&f->provider,(const uint32_t (*)[2])f->pairs,f->port->heard.lid,f->port->heard.gid);
    for(uint32_t k=0;!error && f->alive && f->peer_alive && k<CONTROL_RECEIVES;k++)error=control_listen(f,k);
  }
  if(!error && job!=JOB_RUN && f->side[MESH_RECEIVE].owner && !(f->configured&1u<<MESH_RECEIVE) && !(error=receives_configure(f))){
    f->configured|=1u<<MESH_RECEIVE;
    f->lane_epoch=f->port->lanes.generation;
    if(!f->exclusive)error=port_list(f,0,1);
  }
  if(!error && job!=JOB_RECEIVE && f->side[MESH_SEND].owner && !(f->configured&1u<<MESH_SEND) && !(error=sends_configure(f)))
    f->configured|=1u<<MESH_SEND;
  if(!error && job==JOB_RUN && f->exclusive && !(error=pair_send(&f->provider)))f->connected=1;
  if(!error && (job==JOB_RUN || (job==JOB_CHANGE && ran)))error=port_list(f,1,0);
  return error;
}
/* design/bridge-and-ipc.md#region: the queue pairs a port's device holds, told in its port table */
static void port_pairs(struct port *port){
  atomic_store_explicit(&mesh_links(M)[port->index].pairs,port->device.pairs,memory_order_relaxed);
}
static void flow_job(struct flow *f,int job){
  f->job=job;atomic_store_explicit(&f->done,0,memory_order_relaxed);
  dispatch_async(f->port->work,^{
    /* design/bridge-dataflow.md#stripes: guarded where the device holds a stripe flow (port_guard) */
    const int raised=guard_raise(f->port,0);
    f->result=flow_work(f,job);
    guard_lower(f->port,raised);
    port_pairs(f->port);
    atomic_store_explicit(&f->done,1,memory_order_release);
    wake();
  });
}

/* design/algorithm-sources.md#meshresult: a flow out of its port's table and its queue pairs destroyed, on its port's
   queue.  A flow on its port's lanes is released once its records posted on the lanes have landed (the last landing
   wakes the event loop, FLOW_RELEASING), or the lanes' queue pairs that held them are destroyed (their generation
   moved: lanes that failed and are not yet closed may still land a SEND in its rows, which a relay's lease returns to
   the window as it is freed).  An exclusive flow's queue pairs are moved to the error state and their flushed
   completions drained before they are destroyed (mesh-verbs.h pair_flush: every work request posted and not yet
   taken, each the error state flushes to one completion; the count drained, and how long, logged), all under the
   port's guard (port_guard: no queue pair of the device is posted on meanwhile, design/RDMA-KERNEL-RECOVERY.md).  Then
   the link's census (each ring's SEND requests and frames posted and retired by the send completions its queue pair
   took, an exclusive flow's; its records posted, landed and their bytes; its cells sent; its control frames) and its
   ledger (#ledger) are written.  1 once nothing is held, 0 while records are still posted on its lanes, -1 where a
   destruction was refused (its handles kept for a later release) */
static int flow_release(struct flow *flow){
  port_unlist(flow);
  if(!flow->exclusive && atomic_load_explicit(&flow->lane_frames,memory_order_acquire) &&
     flow->port->lanes.generation==flow->lane_epoch)return 0;
  const int raised=flow->provider.queues?guard_raise(flow->port,1):0;
  if(flow->provider.queues && !flow->flushed){
    const uint64_t began=ledger_ns(),outstanding=pair_outstanding(&flow->provider),drained=pair_flush(&flow->provider);
    say("link %u flushed %llu of %llu work requests in %.1f us\n",flow->port->index,(unsigned long long)drained,
        (unsigned long long)outstanding,(double)(ledger_ns()-began)/1e3);
    flow->flushed=1;
  }
  const int down=down_pair(&flow->provider);
  guard_lower(flow->port,raised);
  if(!down)return -1;
  if(!flow->rings && !flow->streams)return 1;
  for(int q=0;flow->rings && q<flow->qps;q++){
    uint64_t sent=0;
    for(uint32_t k=0;k<flow->stream_count;k++)if(flow->streams[k].queue==(uint32_t)q)sent=flow->streams[k].sent;
    say("link %u census queue=%d requests=%llu frames=%llu retired=%llu send_completions=%llu records=%zu posted=%zu landed=%llu bytes=%llu "
        "cells sent=%llu%s\n",flow->port->index,q,flow->gates?(unsigned long long)flow->gates[q].requests:0ull,
        flow->gates?(unsigned long long)flow->gates[q].posted:0ull,flow->gates?(unsigned long long)flow->gates[q].retired:0ull,
        flow->gates?(unsigned long long)flow->gates[q].completions:0ull,flow->rings[q].count,flow->rings[q].posted,
        (unsigned long long)flow->rings[q].landed,(unsigned long long)flow->rings[q].bytes,(unsigned long long)sent,flow->exclusive?"":" (lanes)");
  }
  if(flow->alive)say("link %u control frames posted=%llu heard=%llu, keepalives %llu (design/bridge-and-ipc.md#control)\n",flow->port->index,
                     (unsigned long long)flow->control_posted,(unsigned long long)flow->alive_heard,(unsigned long long)flow->alive_posted);
  ledger_write(flow);
  free(flow->receive);free(flow->streams);free(flow->gates);free(flow->rings);free(flow->requests);
  flow->receive=NULL;flow->streams=NULL;flow->gates=NULL;flow->rings=NULL;flow->requests=NULL;flow->stream_count=0;
  return 1;
}
/* design/bridge-and-ipc.md#control, design/prepared-machine.md#M12: a stopped flow's program's words cancelled, past its
   landed count alone, once its port's progress threads read it no more: each ring's records from its first not landed
   on, through the program's invocations less its ring's depth (a cyclic program's: its readers keep within the depth of
   the landings, so those records' words hold landings already read; at least the first, which every reader waits on
   before any later one), every one not landed in a one-pass program; landed words stay.  A flow whose receives were
   never prepared lands nothing: every word toward its node */
static void flow_cancel(struct flow *f){
  struct mesh_cancellation *cancel=f->cancel;
  if(!cancel)return;
  if(!f->rings){
    _Atomic uint64_t *words=(void *)((char *)M+cancel->ranges[f->destination].offset);
    for(uint64_t j=0;j<cancel->ranges[f->destination].count;j++){
      const uint64_t word=atomic_load_explicit(words+j,memory_order_relaxed);
      if(!word || (cancel->cyclic && word!=UINT64_MAX))atomic_store_explicit(words+j,UINT64_MAX,memory_order_release);
    }
    return;
  }
  const uint64_t n=f->invocations?f->invocations:1;
  for(int q=0;q<f->qps;q++){
    const struct receive_ring *r=f->rings+q;
    if(!r->count)continue;
    const uint64_t first=r->landed,end=f->cyclic?first+r->count:r->count;
    const struct prepared_receive *head=r->first+first%r->count;
    const uint64_t bound=head->invocation+first/r->count*n+(r->depth<n?n-r->depth:1);
    for(uint64_t k=first;k<end;k++){
      const struct prepared_receive *record=r->first+k%r->count;
      if(f->cyclic && record->invocation+k/r->count*n>=bound)break;
      if(record->input!=&record->sink && !record->release)atomic_store_explicit(record->input,UINT64_MAX,memory_order_release);
    }
  }
}
/* A flow stopped on its port's queue, after its job: where it failed or its peer ended it, its program's words toward
   the far end past its landed count cancelled (flow_cancel) once its port's progress threads no longer read it, and
   its status the code naming the node `where`; then released (flow_release), the event loop woken once it is, or (a
   rider's records still posted) by their last landing */
static void flow_stop(struct flow *f,int64_t code,uint32_t domain,uint32_t where){
  const uint64_t owner=f->owner;const uint32_t destination=f->destination;
  f->stopping=1;f->frees=f->port->frees;atomic_store_explicit(&f->stopped,0,memory_order_relaxed);
  atomic_fetch_or_explicit(&f->ending,FLOW_RELEASING,memory_order_release);
  dispatch_async(f->port->work,^{
    port_unlist(f);
    if(code && owned(owner)){
      flow_cancel(f);
      status_failed(&mesh_client_at(M,owner)->status[destination],owner,code,domain,where);
    }
    const int released=flow_release(f);
    atomic_store_explicit(&f->stopped,released>0?1:released?3:2,memory_order_release);
    port_pairs(f->port);
    if(released>0 || (!released && !atomic_load_explicit(&f->lane_frames,memory_order_acquire)))wake();
  });
}

/* design/bridge-and-ipc.md#protocol: a message to the port's peer, queued behind what the socket has not taken yet; a
   send that fails is the link's loss */
static void port_lost(struct port *port,int64_t code,int retire);
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
static void flow_say(struct flow *f,uint32_t kind,uint32_t count,int64_t code,const uint32_t (*pair)[2],uint32_t origin){
  struct message said={.kind=kind,.count=count,.key=f->key,.code=code,.from=f->near,.to=f->far,.failed=origin,.stripe=(uint16_t)f->stripe};
  for(uint32_t q=0;pair && q<count && q<MESH_PAIRS;q++){said.pair[q][0]=pair[q][0];said.pair[q][1]=pair[q][1];}
  port_say(f->port,said,NULL,0);
}
/* a READY: the flow's queue pairs, then its route's `hops` nodes and the transfers its SEND side sends, with its
   program's digest */
static void ready_say(struct flow *f,const uint32_t *route,uint32_t hops){
  int d=f->side[MESH_SEND].owner?MESH_SEND:MESH_RECEIVE;
  struct mesh_client *client=mesh_client_at(M,f->side[d].owner);
  struct mesh_tx *tx=side_tx(f,d);
  struct ready ready={.hops=hops,.slots=tx->slots,.invocations=tx->invocations,.depth=atomic_load(&client->depth),
    .cyclic=atomic_load(&client->cyclic),.stripe=f->stripe,.stripes=f->stripes,.exclusive=(uint32_t)f->exclusive,
    .digest=atomic_load_explicit(&client->digest,memory_order_relaxed)};
  size_t transfers=0;
  for(uint32_t k=0;k<f->queues;k++)transfers+=ready.lengths[k]=side_length(f,MESH_SEND,k);
  size_t bytes=sizeof ready+(size_t)hops*sizeof *route+transfers*sizeof(struct mesh_transfer);
  unsigned char *payload=malloc(bytes),*at=payload;
  if(!payload){port_lost(f->port,ENOMEM,0);return;}
  memcpy(at,&ready,sizeof ready);at+=sizeof ready;
  memcpy(at,route,(size_t)hops*sizeof *route);at+=(size_t)hops*sizeof *route;
  for(uint32_t k=0;k<f->queues;k++){memcpy(at,side_transfers(f,MESH_SEND,k),ready.lengths[k]*sizeof(struct mesh_transfer));at+=ready.lengths[k]*sizeof(struct mesh_transfer);}
  port_say(f->port,(struct message){.kind=READY,.count=(uint32_t)f->qps,.key=f->key,.from=f->near,.to=f->far,.stripe=(uint16_t)f->stripe},
           payload,(uint32_t)bytes);
  free(payload);
}

/* design/bridge-and-ipc.md#routes: a client program awaiting the READYs of its routed peer, the pair's lower node, whose
   route both ends take: its `stripes` once the first came (#cables) and the stripes it has taken (a bit each) */
struct await { struct await *next; uint64_t key,owner,taken; uint32_t destination,far,stripes; };
static struct await *awaits;
static struct await *await_find(uint64_t key,uint32_t far){
  for(struct await *a=awaits;a;a=a->next)if(a->key==key && a->far==far)return a;
  return NULL;
}
static struct await *await_of(uint64_t owner,uint32_t d){
  for(struct await *a=awaits;a;a=a->next)if(a->owner==owner && a->destination==d)return a;
  return NULL;
}
static void await_drop(struct await *a){
  for(struct await **at=&awaits;*at;at=&(*at)->next)if(*at==a){*at=a->next;await_count--;break;}
  free(a);
  busy_tell();
}
/* an await's stripe taken: dropped once every stripe is (1 then) */
static int await_took(struct await *a,uint32_t stripe,uint32_t stripes){
  a->stripes=stripes;a->taken|=UINT64_C(1)<<stripe;
  if((uint32_t)__builtin_popcountll(a->taken)!=a->stripes)return 0;
  await_drop(a);
  return 1;
}

/* design/bridge-and-ipc.md#control: a lane rider's control frame to its peer (FAIL, RELEASED), an entry of the port's
   queue that its lanes' send thread takes into lane 0's next header, the rider gone by then or not */
static void port_tell(struct port *port,const struct flow *f,uint32_t kind,uint64_t bytes){
  const uint64_t head=atomic_load_explicit(&port->tell_head,memory_order_relaxed);
  if(head-atomic_load_explicit(&port->tell_tail,memory_order_acquire)>=MESH_TELLS){
    say("link %u (%s): program %016llx toward node %u: its control frame %u not told, %u of the port's lane riders' queued\n",port->index,
        port->name,(unsigned long long)f->key,f->far,kind,MESH_TELLS);
    return;
  }
  port->tells[head%MESH_TELLS]=(struct lane_entry){f->peer_slot,f->peer_generation,kind<<24,0,bytes};
  atomic_store_explicit(&port->tell_head,head+1,memory_order_release);
  port_bell(port);
}
/* design/bridge-and-ipc.md#control: a running flow's peer told it failed (FAIL, the code and the node `origin`): an
   exclusive flow's on its own control queue pair by its send thread (FLOW_TELLING; the flow stops once that FAIL's
   completion is taken, or the peer's own FAIL came: flows_turn), 1 then; a lane rider's on its lanes (port_tell), 0 */
static int control_tell(struct flow *f,int64_t code,uint32_t origin){
  const int64_t told=code?code:ECANCELED;
  if(!f->exclusive){port_tell(f->port,f,CONTROL_FAIL,(uint64_t)origin<<32|(uint32_t)told);return 0;}
  if(!f->alive || !f->peer_alive || !f->listed || !atomic_load_explicit(&f->progressing,memory_order_acquire) ||
     atomic_load_explicit(&f->heard_fail,memory_order_acquire))return 0;
  f->fail_code=told;f->fail_origin=origin;
  atomic_fetch_or_explicit(&f->ending,FLOW_TELLING|FLOW_MUTED,memory_order_release);
  port_bell(f->port);
  return 1;
}
/* design/bridge-and-ipc.md#control: a client program's flows muted on the event loop, every one toward every node,
   before any word of it is cancelled (flow_stop, on their ports' queues after): no progress thread posts a cell of the
   program past this (their control frames go on until each flow tells FAIL and stops), so a GPU walk that a cancelled
   wait halted, whose later steps run on and release cells (an unbounded program reads no halt but in its waits), puts
   nothing on the wire, and its peers learn of the loss by the FAIL each flow tells them rather than from cells it
   computed without the lost one's */
static void program_quiet(uint64_t owner){
  struct port *settle[MESH_PORTS];
  uint32_t n=0;
  for(struct flow *g=flows;g;g=g->next){
    if(g->owner!=owner || g->state==FAILED)continue;
    atomic_fetch_or_explicit(&g->ending,FLOW_MUTED,memory_order_release);
    uint32_t k=0;
    while(k<n && settle[k]!=g->port)k++;
    if(k==n)settle[n++]=g->port;
  }
  for(uint32_t k=0;k<n;k++)port_settle(settle[k]);
}
/* design/algorithm-sources.md#meshresult: a program on a port stopped, its peer told (`tell` FAIL naming the node
   `origin` where it happened: a running flow's on its control channel, control_tell, an exclusive one stopping once
   that FAIL is sent; one not yet running, whose queue pairs or lanes may not reach the peer's, on the control socket),
   its status the failure naming `where`; a relay's other flow fails with it, telling its peer, and so does every flow
   of the same client program, toward every node and on its other cables and stripes, all of them quieted first
   (program_quiet), and its await for the stripes that have not come */
static void flow_fail(struct flow *f,int64_t code,uint32_t domain,uint32_t tell,uint32_t where,uint32_t origin){
  if(f->state==FAILED)return;
  if(f->owner && !f->partner && !(atomic_load_explicit(&f->ending,memory_order_relaxed)&FLOW_MUTED))program_quiet(f->owner);
  const int running=f->state==RUNNING;
  f->state=FAILED;
  say("link %u: program %016llx toward node %u (stripe %u of %u) failed at node %u: code %lld domain %u\n",f->port->index,
    (unsigned long long)f->key,f->far,f->stripe,f->stripes,where,(long long)code,domain);
  if(tell && running && control_tell(f,code,origin)){f->stop_code=code;f->stop_domain=domain;f->stop_where=where;}
  else{
    if(tell && !running)flow_say(f,tell,0,code,NULL,origin);
    flow_stop(f,code,domain,where);
  }
  if(f->partner)flow_fail(f->partner,code,domain,FAIL,origin,origin);
  for(struct flow *g=flows;f->owner && g;g=g->next)
    if(g->owner==f->owner && (g->destination==f->destination || !f->partner))flow_fail(g,code,domain,FAIL,where,origin);
  struct await *a=f->owner?await_of(f->owner,f->destination):NULL;
  if(a)await_drop(a);
}
/* a client's program toward its destination `d` failed as a value naming the node `where`: its flows failed, telling
   their peers (their FAILs naming `origin`), its await dropped, and its status the failure */
static void program_end(uint64_t owner,uint32_t d,int64_t code,uint32_t where,uint32_t origin){
  for(struct flow *g=flows;g;g=g->next)if(g->owner==owner && g->destination==d)flow_fail(g,code,4,FAIL,where,origin);
  struct await *a=await_of(owner,d);
  if(a)await_drop(a);
  if(owned(owner))status_failed(&mesh_client_at(M,owner)->status[d],owner,code,4,where);
}
static void program_fail(uint64_t owner,uint32_t d,int64_t code,uint32_t where){ program_end(owner,d,code,where,M->node); }
/* design/bridge-and-ipc.md#protocol: a client's program of this node that runs: started (its slot's `configured` its
   owner, acquired before its digest is read) and not yet failed toward every destination it binds; and its digest
   (mesh.h mesh_client `digest`, 0 where it names none, or where the program does not run: a READY that meets a program
   that ended or failed is held for the execution's next program), and whether two named ones differ */
static int program_runs(uint64_t owner){
  if(!owned(owner) || atomic_load_explicit(&mesh_client_at(M,owner)->configured,memory_order_acquire)!=owner)return 0;
  const struct mesh_client *client=mesh_client_at(M,owner);
  for(uint32_t d=0,n=mesh_destination(client,0);d<n;d++)
    if(mesh_destination_transfers(M,owner,d) && atomic_load(&client->status[d].phase)!=MESH_STOPPED)return 1;
  return 0;
}
static uint64_t program_digest(uint64_t owner){
  return program_runs(owner)?atomic_load_explicit(&mesh_client_at(M,owner)->digest,memory_order_relaxed):0;
}
static int digest_apart(uint64_t mine,uint64_t theirs){ return mine && theirs && mine!=theirs; }
/* a client's program whose peer at `node` runs a program of its execution made from another digest: failed toward every
   destination it binds as a value (ESTALE naming that node), each flow telling its peer FAIL with ESTALE naming it too
   (that node's own told this one), so the peer's program fails alike (port_heard), and so on along the execution's
   programs: no member waits for a peer whose program differs */
static void program_stale(uint64_t owner,uint32_t node){
  if(!program_runs(owner))return;
  struct mesh_client *client=mesh_client_at(M,owner);
  say("program %016llx: node %u's program of it is made from another digest than this node's (%016llx): failed\n",
      (unsigned long long)client->key,node,(unsigned long long)atomic_load_explicit(&client->digest,memory_order_relaxed));
  for(uint32_t d=0,n=mesh_destination(client,0);d<n;d++)
    if(mesh_destination_transfers(M,owner,d))program_end(owner,d,ESTALE,node,client->tx[d].node==node?M->node:node);
}
/* design/bridge-and-ipc.md#control: a program that ended (its slot retired): a running flow of an orderly end
   (mesh_detach) posts every cell its program released and takes those SENDs' completions (FLOW_ENDING, flow_send,
   lanes_drained), and the event loop then releases it, telling its peer nothing (flows_turn; a lane rider tells it
   RELEASED, for the relays of its route); an abort (its process gone, a revoke), or the end of a flow not yet running,
   fails each, its peer told FAIL (a running flow's on its control channel, one not yet running's on the control
   socket).  A retire under way (RETIRING alone) is taken at the next turn */
static void flow_end(struct flow *f){
  if(f->state==FAILED || (atomic_load_explicit(&f->ending,memory_order_relaxed)&FLOW_ENDING))return;
  const struct mesh_client *client=mesh_client_at(M,f->owner);
  const uint64_t word=atomic_load_explicit(&client->owner,memory_order_acquire);
  if(word==(f->owner|MESH_RETIRING))return;
  const int orderly=word==(f->owner|MESH_RETIRING|MESH_ORDERLY);
  if(orderly && f->state==RUNNING){
    atomic_fetch_or_explicit(&f->ending,FLOW_ENDING,memory_order_release);
    port_bell(f->port);
    return;
  }
  const int running=f->state==RUNNING;
  if(!(atomic_load_explicit(&f->ending,memory_order_relaxed)&FLOW_MUTED))program_quiet(f->owner);
  f->state=FAILED;
  if(running && control_tell(f,ECANCELED,M->node)){f->stop_code=0;f->stop_domain=0;f->stop_where=0;return;}
  if(!running)flow_say(f,FAIL,0,ECANCELED,NULL,M->node);
  flow_stop(f,0,0,0);
}
static void lease_free(struct lease *lease){
  mesh_retire(M,lease->context.client);
  unreleased=1;free(lease);
}
static void lease_drop(struct lease *lease){ if(lease && !--lease->users)lease_free(lease); }
static struct lease *lease_take(uint64_t key){
  struct lease *lease=calloc(1,sizeof *lease);
  if(lease){lease->context.fd=-1;if(mesh_slot_take(&lease->context,M,key)){free(lease);lease=NULL;}}
  return lease;
}
/* a flow freed, its index in its port's table and the queue pairs it held of the device's given back */
static void flow_free(struct flow *f){
  for(struct flow **at=&flows;*at;at=&(*at)->next)if(*at==f){*at=f->next;break;}
  if(f->partner)f->partner->partner=NULL;
  f->port->flows--;
  f->port->reserved[f->slot/64]&=~(UINT64_C(1)<<(f->slot%64));
  if(f->exclusive)f->port->committed-=(uint32_t)(f->qps+f->alive);
  for(int d=0;d<2;d++){lease_drop(f->side[d].lease);lease_drop(f->pending[d].lease);}
  if(f->control_page!=MESH_ABSENT)mesh_bits_clear(mesh_arena_bits(M),f->control_page,control_pages());
  f->port->frees++;
  free(f);
  busy_tell();
}
/* design/bridge-and-ipc.md#clients: a client program's status once every flow of it toward the node runs, every stripe
   of a routed one among them (no await of it left) */
static void program_paired(struct flow *f){
  for(struct flow *g=flows;g;g=g->next)
    if(g->owner==f->owner && g->destination==f->destination && (g->state!=RUNNING || g->job))return;
  if(!owned(f->owner) || await_of(f->owner,f->destination))return;
  struct mesh_port_info *status=&mesh_client_at(M,f->owner)->status[f->destination];
  atomic_store_explicit(&status->code,0,memory_order_relaxed);atomic_store_explicit(&status->failed,0,memory_order_relaxed);
  atomic_store_explicit(&status->phase,MESH_PAIRED,memory_order_relaxed);
  status_prepared(status,f->owner);
}

/* design/bridge-and-ipc.md#protocol: POSTED told the peer once the flow's receives are posted; a relay's once its own
   receives are and its partner heard the next hop's POSTED, so a POSTED reaches a route's end only once the far end's
   program has started and every relay between has posted (end to end, design/recovery.md#first).  Tried at each of
   the flow's job ends and as its partner hears POSTED (port_heard), whichever comes last */
static void flow_tell(struct flow *f){
  if(f->told || f->job || f->state==FAILED || f->state<POSTING)return;
  if(f->partner && (!(f->partner->heard&HEARD_POSTED) || !f->side[MESH_RECEIVE].owner || !(f->configured&1u<<MESH_RECEIVE)))return;
  f->told=1;
  flow_say(f,POSTED,0,0,NULL,M->node);
}
/* design/bridge-and-ipc.md#lanes: the peer's word on whether the flow takes queue pairs of its own (its READY's
   `exclusive`, its PAIRS' count), ANDed into the flow's until the flow pairs, so both ends pair alike; 0, or EPROTO
   where a flow that paired hears otherwise */
static int flow_mode(struct flow *f,int exclusive){
  if(f->decided)return f->exclusive==!!exclusive?0:EPROTO;
  if(f->exclusive && !exclusive){f->exclusive=0;f->port->committed-=(uint32_t)(f->qps+f->alive);f->alive=0;}
  return 0;
}
/* design/bridge-and-ipc.md#protocol: a flow's next step, its job's end or a message heard: paired once the peer's READY
   or PAIRS is heard (an exclusive flow's queue pairs made; PAIRS sent), ready to receive once its PAIRS are and, held,
   once its client's program took it (POSTED told, flow_tell), running once its POSTED is; a relay whose side's lists
   arrived after its receives were posted changes.  A flow on its port's lanes receives, runs and changes once the lanes
   are up (lanes that fail to be made or connected are the link's loss, which fails it: #lanes) */
static void flow_step(struct flow *f){
  if(f->state==FAILED || f->job || f->stopping)return;
  for(int d=0;d<2;d++)if(f->pending[d].owner && !f->side[d].owner){f->side[d]=f->pending[d];f->pending[d]=(struct side){0};}
  const struct lanes *l=&f->port->lanes;
  const int up=f->exclusive || l->state==LANES_UP;
  if(f->state==READIED && (f->heard&(HEARD_READY|HEARD_PAIRS))){f->decided=1;flow_job(f,JOB_PAIR);}
  else if(f->state==PAIRING && (f->heard&HEARD_PAIRS) && !f->hold && up)flow_job(f,JOB_RECEIVE);
  else if(f->state==POSTING && (f->heard&HEARD_POSTED) && up)flow_job(f,JOB_RUN);
  else if(up && (f->state==POSTING || f->state==RUNNING) && ((f->side[MESH_SEND].owner && !(f->configured&1u<<MESH_SEND) && f->state==RUNNING) ||
                                (f->side[MESH_RECEIVE].owner && !(f->configured&1u<<MESH_RECEIVE))))flow_job(f,JOB_CHANGE);
}
static void flow_done(struct flow *f){
  const int job=f->job;
  f->job=0;
  if(f->state==FAILED)return;
  if(f->result){
    if(job==JOB_PAIR && f->result==ENOSPC)
      say("link %u (%s): program %016llx toward node %u: %d queue pairs refused, the device holding %u of its max_qp %u\n",
        f->port->index,f->port->name,(unsigned long long)f->key,f->far,f->qps,f->port->device.pairs,f->port->device.max_pairs);
    flow_fail(f,f->result,1,FAIL,M->node,M->node);
    return;
  }
  if(job==JOB_PAIR){
    /* #lanes: an exclusive flow's PAIRS its queue pairs, a lane rider's none, its index and generation in its code */
    uint32_t pairs[MESH_PAIRS][2];
    for(int q=0;f->exclusive && q<f->qps+f->alive;q++){pairs[q][0]=f->provider.queues[q].pair->qp_num;pairs[q][1]=(f->provider.psn+(uint32_t)q)&0xffffff;}
    flow_say(f,PAIRS,f->exclusive?(uint32_t)(f->qps+f->alive):0,f->exclusive?0:(int64_t)((uint64_t)f->generation<<32|f->slot),
             (const uint32_t (*)[2])pairs,M->node);
    f->state=PAIRING;
  }else if(job==JOB_RECEIVE){
    /* design/recovery.md#first: the peer's POSTED waits on its program's start where no READY told of it (a peer that
       holds posts once its client's program takes it), and a relay's on the far end's */
    f->state=POSTING;
  }else{
    if(job==JOB_RUN)f->state=RUNNING;
    if(f->state==RUNNING)program_paired(f);
  }
  flow_tell(f);
  flow_step(f);
}

/* design/bridge-and-ipc.md#flows: a flow made on a port toward `far` for the execution `key`, the program's invocations
   t with t mod stripes = stripe (#cables), an index of the port's table its own (MESH_FLOWS a port), exclusive where
   `exclusive` asks it, its queue pairs fit a PAIRS message, and the port's device has room for them beside those its
   flows and lanes hold or will, the lanes' own counted before they are made (#lanes: else it rides the lanes); NULL
   where it is not one or the table is full */
static struct flow *flow_make(struct port *port,uint64_t key,uint32_t near,uint32_t far,uint32_t queues,uint32_t slots,uint32_t stripe,uint32_t stripes,
                              int exclusive){
  if(!queues || !slots || queues>MESH_QUEUES || (uint64_t)queues*slots>INT32_MAX || !stripes || stripes>MESH_STRIPES || stripe>=stripes)return NULL;
  uint32_t slot=0;
  while(slot<MESH_FLOWS && port->reserved[slot/64]>>(slot%64)&1)slot++;
  if(slot==MESH_FLOWS)return NULL;
  struct flow *f=calloc(1,sizeof *f);
  const uint32_t qps=queues*slots;
  if(!f)return NULL;
  /* #control: queue pairs of its own and one past them, its control pair (a PAIRS message holds them, the device has
     room for them), and a page of the window for its frames */
  exclusive=exclusive && qps<MESH_PAIRS && port->device.max_pairs &&
    port->committed+qps+1+(port->lanes.committed?0:MESH_LANES)<=port->device.max_pairs;
  const uint32_t page=exclusive?mesh_arena_alloc(&bridge,control_pages(),1):MESH_ABSENT;
  exclusive=exclusive && page!=MESH_ABSENT;
  *f=(struct flow){.next=flows,.near=near,.far=far,.queues=queues,.stripe=stripe,.stripes=stripes,.qps=(int)qps,.exclusive=exclusive,.alive=exclusive,
    .slot=slot,.generation=++port->generation[slot],.key=key,.M=M,.port=port,.provider={.device=&port->device},.control_page=page};
  const int alive=exclusive;
  port->reserved[slot/64]|=UINT64_C(1)<<(slot%64);
  if(slot>=atomic_load(&port->high))atomic_store(&port->high,slot+1);
  if(exclusive)port->committed+=qps+(uint32_t)alive;
  flows=f;port->flows++;busy_tell();
  return f;
}
static struct flow *flow_on(struct port *port,uint64_t key,uint32_t near,uint32_t far,uint32_t stripe){
  for(struct flow *f=flows;f;f=f->next)
    if(f->port==port && f->key==key && f->near==near && f->far==far && f->stripe==stripe && f->state!=FAILED)return f;
  return NULL;
}

/* design/bridge-and-ipc.md#flows, #routes: a relayed program a READY carried, mirrored into a lease: rows the bridge
   leases for each transfer the sender sends (mesh.h mesh_relay_slot a slot of it), bound as receives from `from` and as
   SENDs toward `to` over the same rows, so a landed record releases the SEND that carries it on (M08), and the
   sender's digest named, which the relay's READY onward carries (#protocol).  0, or the refusal */
static int mirror(struct lease *lease,const struct ready *r,const struct mesh_transfer *t,uint32_t from,uint32_t to){
  struct mesh_ctx *c=&lease->context;
  struct mesh_client *client=mesh_client_at(M,c->client);
  int error=0;
  client->tx[mesh_destination(client,from)].node=from;
  client->tx[mesh_destination(client,to)].node=to;
  for(uint32_t k=0,i=0;k<MESH_QUEUES && !error;k++)for(uint32_t j=0;j<r->lengths[k] && !error;j++,i++){
    const struct mesh_transfer *x=t+i;
    struct mesh_section s;
    if(!x->count)error=EPROTO;
    else if(!(error=mesh_section_create(c,mesh_relay_slot(x,r->depth,M->pgsz),x->stride?x->count:1,&s))){
      s.bytes=x->bytes;if(!x->stride){s.stride=0;s.count=x->count;}
      error=mesh_transfer_bind(c,from,k,MESH_RECEIVE,x->binding,s,x->invocation_pages,x->begin,x->end,x->depth);
      if(!error)error=mesh_transfer_bind(c,to,k,MESH_SEND,x->binding,s,x->invocation_pages,x->begin,x->end,x->depth);
    }
  }
  if(!error)error=mesh_transfers_prepare(c,r->slots,r->invocations,r->depth);
  if(!error && r->cyclic)error=mesh_transfers_cyclic(c);
  if(!error)error=mesh_transfers_digest(c,r->digest);
  return error;
}
/* a READY refused as a value, naming this node */
static void ready_refuse(struct port *port,const struct message *heard,int64_t code){
  port_say(port,(struct message){.kind=FAIL,.key=heard->key,.from=heard->to,.to=heard->from,.code=code,.failed=M->node,.stripe=heard->stripe},
           NULL,0);
}
/* a READY's program apart from its stripe and route, the lease of a relayed direction its stripes share (FNV-1a) */
static uint64_t ready_shape(const struct ready *ready,const struct mesh_transfer *list,uint64_t transfers){
  struct ready shape=*ready;
  shape.hops=shape.stripe=shape.exclusive=0;
  uint64_t x=UINT64_C(0xcbf29ce484222325);
  for(size_t i=0;i<sizeof shape;i++)x=(x^((const unsigned char *)&shape)[i])*UINT64_C(0x100000001b3);
  for(size_t i=0;i<transfers*sizeof *list;i++)x=(x^((const unsigned char *)list)[i])*UINT64_C(0x100000001b3);
  return x;
}
/* the lease a relayed direction's other stripes through this node took: a relay flow of the same execution receiving
   from the same far end, whose partner sends toward the same far end; NULL where none has */
static struct lease *relay_lease(const struct flow *f,const struct flow *g){
  for(struct flow *h=flows;h;h=h->next){
    if(h==f || h->key!=f->key || h->far!=f->far || h->stripes!=f->stripes || !h->partner || h->partner->far!=g->far || h->state==FAILED ||
       h->stopping)continue;
    struct lease *lease=h->side[MESH_RECEIVE].lease?h->side[MESH_RECEIVE].lease:h->pending[MESH_RECEIVE].lease;
    if(lease)return lease;
  }
  return NULL;
}
/* a relay's direction a READY told: received by the flow it arrived on and sent on by its partner, over a lease that
   mirrors its lists, the one its program's other stripes through this node share (each stripe's flows prepare their
   own invocations of its rows, #cables), else a new one; EPROTO where a stripe's program is not its lease's */
static int relay_direction(struct flow *f,const struct ready *ready,const struct mesh_transfer *list,uint64_t shape){
  struct flow *g=f->partner;
  struct lease *lease=relay_lease(f,g);
  if(lease && lease->shape!=shape)return EPROTO;
  if(!lease){
    if(!(lease=lease_take(f->key)))return EBUSY;
    int error=mirror(lease,ready,list,f->far,g->far);
    if(error){lease_free(lease);return error;}
    lease->shape=shape;
  }
  struct mesh_client *client=mesh_client_at(M,lease->context.client);
  f->pending[MESH_RECEIVE]=(struct side){lease->context.client,mesh_destination(client,f->far),lease};
  g->pending[MESH_SEND]=(struct side){lease->context.client,mesh_destination(client,g->far),lease};
  lease->users+=2;
  if(!f->owner)flow_program(f,f->pending[MESH_RECEIVE].owner,f->pending[MESH_RECEIVE].destination);
  if(!g->owner)flow_program(g,g->pending[MESH_SEND].owner,g->pending[MESH_SEND].destination);
  flow_step(g);
  return 0;
}

static uint32_t program_queues(uint64_t owner,uint32_t d);
/* a flow taken by its client's program toward destination `d` (its one job before, JOB_PAIR, reads no side), its port
   among the program's (mesh_client `ports`, the bells mesh_transfers_wake rings) */
static void flow_take(struct flow *f,uint64_t owner,uint32_t d){
  f->side[MESH_SEND]=f->side[MESH_RECEIVE]=(struct side){owner,d,NULL};
  flow_program(f,owner,d);
  f->hold=0;
  struct mesh_client *client=mesh_client_at(M,owner);
  atomic_fetch_or_explicit(&client->ports,UINT32_C(1)<<f->port->index,memory_order_release);
  atomic_store_explicit(&client->status[d].phase,MESH_PAIRING,memory_order_relaxed);
}
/* the accepted program of execution `key` that binds `node` here (its slot's owner and destination), with a flow toward
   it that has not failed; 0 where none */
static uint64_t owners[MESH_CLIENTS],accepted[MESH_CLIENTS];
static int program_live(uint64_t key,uint32_t node,uint64_t *owner,uint32_t *d){
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct mesh_client *client=(struct mesh_client *)((char *)M+M->client_off)+s;
    const uint64_t o=accepted[s];
    if(!o || !owned(o) || client->key!=key)continue;
    const uint32_t at=mesh_destination(client,node);
    if(at==MESH_DESTINATIONS || client->tx[at].node!=node)continue;
    for(struct flow *g=flows;g;g=g->next)if(g->owner==o && g->destination==at && g->state!=FAILED){*owner=o;*d=at;return 1;}
  }
  return 0;
}
/* the program of execution `key` that runs here (program_runs: its slot's owner), 0 where none */
static uint64_t program_keyed(uint64_t key){
  for(uint32_t s=0;s<MESH_CLIENTS;s++)
    if(accepted[s] && program_runs(accepted[s]) && mesh_client_at(M,accepted[s])->key==key)return accepted[s];
  return 0;
}
/* design/bridge-and-ipc.md#ports: the ports paired with `node`, in the cable order both ends read alike, by the index
   the lower node gave each cable */
static uint32_t cable_rank(const struct port *port){ return M->node<port->peer?port->index:port->peer_port; }
static uint32_t cables(uint32_t node,struct port **out){
  uint32_t n=0;
  for(uint32_t p=0;p<port_count;p++)if(ports[p].greeted && ports[p].peer==node && !ports[p].retiring)out[n++]=ports+p;
  for(uint32_t i=1;i<n;i++)for(uint32_t j=i;j>0 && cable_rank(out[j])<cable_rank(out[j-1]);j--){struct port *t=out[j];out[j]=out[j-1];out[j-1]=t;}
  return n;
}
/* design/bridge-and-ipc.md#cables: a program's pair of end nodes and its execution mixed (murmur3's finalizer), alike at
   either end and on every relay, which with a stripe picks the cable a routed hop takes */
static uint64_t pair_hash(uint64_t key,uint32_t a,uint32_t b){
  uint64_t x=key^((uint64_t)(a<b?a:b)<<32|(a<b?b:a));
  x^=x>>33;x*=UINT64_C(0xff51afd7ed558ccd);x^=x>>33;x*=UINT64_C(0xc4ceb9fe1a85ec53);x^=x>>33;
  return x;
}
/* the cable of a routed hop toward the node of `next` (its n cables) that stripe `stripe` of the program between `a`
   and `b` rides: stripe j on cable (hash + j) mod n, so a hop of c cables carries stripes/c of a program whose stripes
   c divides */
static struct port *stripe_cable(struct port **next,uint32_t n,uint64_t key,uint32_t a,uint32_t b,uint32_t stripe){
  return next[(pair_hash(key,a,b)+stripe)%n];
}

/* design/bridge-and-ipc.md#protocol, #routes: a READY.  One heard by a flow that heard its READY already is the peer's
   program made again: that flow fails (ECONNRESET) and the READY is taken as a new one.  Its flow here pairs (its
   `exclusive` ANDed into the flow's, #lanes); one heard by a relay's flow (the far end's program telling its direction) is mirrored and sent on by the
   relay's other flow toward the route's start, so every relay of the route carries both directions; a program whose
   route goes on past this node is relayed toward the route's next node, over rows the bridge leases (no client of this
   node takes part, its stripes sharing them), each stripe on the lanes of the cable of the next hop it rides
   (stripe_cable); one that ends here is taken by a program awaiting it, else held, its queue pairs made where it takes
   its own and no rows, for the client whose program takes it (#flows).  One that ends here naming another digest than
   this node's program of its key (#protocol) is refused, and that program failed toward every destination, both as
   values (ESTALE: program_stale).  One that ends on a port where this node's program of its key toward its sender has
   no flow, while that program runs on others, finds the two ends' maps apart: refused, and the program failed, both as
   values (EPROTO) */
static void ready_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  struct ready ready;
  if(heard->bytes<sizeof ready){port_lost(port,EPROTO,0);return;}
  memcpy(&ready,payload,sizeof ready);
  uint64_t transfers=0;
  for(uint32_t k=0;k<MESH_QUEUES;k++)transfers+=ready.lengths[k];
  const uint32_t queues=ready.slots?heard->count/ready.slots:0;
  if(!ready.hops || !queues || queues>MESH_QUEUES || heard->count%ready.slots || !ready.stripes || ready.stripes>MESH_STRIPES ||
     ready.stripe>=ready.stripes || ready.stripe!=heard->stripe ||
     heard->bytes!=sizeof ready+(uint64_t)ready.hops*sizeof(uint32_t)+transfers*sizeof(struct mesh_transfer)){port_lost(port,EPROTO,0);return;}
  uint32_t *route=malloc((size_t)ready.hops*sizeof *route);
  struct mesh_transfer *list=malloc((transfers?transfers:1)*sizeof *list);
  if(!route || !list){free(route);free(list);ready_refuse(port,heard,ENOMEM);return;}
  memcpy(route,payload+sizeof ready,(size_t)ready.hops*sizeof *route);
  memcpy(list,payload+sizeof ready+(size_t)ready.hops*sizeof *route,transfers*sizeof *list);
  const uint64_t shape=ready_shape(&ready,list,transfers);
  struct flow *f=flow_on(port,heard->key,heard->to,heard->from,ready.stripe);
  uint64_t owner=0;uint32_t d=0;
  /* the peer's program of this execution made again (a flow says READY once): its earlier one ended and released its
     flows, so this end's flow of it fails as the peer's end (ECONNRESET naming it, the words past its landed counts
     cancelled: a session on it reopens at its next group) and the READY is taken anew */
  if(f && (f->heard&HEARD_READY)){flow_fail(f,ECONNRESET,4,0,heard->from,heard->from);f=NULL;}
  if(f && !f->partner && digest_apart(program_digest(f->owner),ready.digest)){
    flow_fail(f,ESTALE,4,FAIL,heard->from,M->node);
    program_stale(f->owner,heard->from);
  }
  else if(f && !f->partner && (f->stripes!=ready.stripes || f->queues!=queues || (uint32_t)f->qps!=heard->count))
    flow_fail(f,EPROTO,4,FAIL,M->node,M->node);
  else if(f){
    int error=f->partner?(!f->side[MESH_RECEIVE].owner && !f->pending[MESH_RECEIVE].owner?relay_direction(f,&ready,list,shape):0):
      flow_mode(f,(int)ready.exclusive);
    if(error)flow_fail(f,error,1,FAIL,M->node,M->node);
    else{
      if(f->partner)port_say(f->partner->port,(struct message){.kind=READY,.count=heard->count,.key=heard->key,.from=heard->from,
                                                               .to=heard->to,.stripe=heard->stripe},payload,heard->bytes);
      f->heard|=HEARD_READY;flow_step(f);
    }
  }else if(route[0]!=M->node || route[ready.hops-1]!=heard->to || heard->from==M->node)ready_refuse(port,heard,EPROTO);
  else if(ready.hops==1){
    struct await *a=await_find(heard->key,heard->from);
    const uint64_t keyed=a?a->owner:program_keyed(heard->key);
    if(digest_apart(program_digest(keyed),ready.digest)){
      ready_refuse(port,heard,ESTALE);
      program_stale(keyed,heard->from);
    }else if(a && (program_queues(a->owner,a->destination)!=queues || mesh_client_at(M,a->owner)->tx[a->destination].slots!=ready.slots ||
             (a->stripes && a->stripes!=ready.stripes) || (a->taken>>ready.stripe&1))){
      say("link %u (%s): program %016llx: node %u's READY (stripe %u of %u) is not the shape of the program awaiting it here: refused\n",
          port->index,port->name,(unsigned long long)heard->key,heard->from,ready.stripe,ready.stripes);
      ready_refuse(port,heard,EPROTO);
      program_fail(a->owner,a->destination,EPROTO,M->node);
    }else if(!a && program_live(heard->key,heard->from,&owner,&d)){
      say("link %u (%s): program %016llx: node %u's READY came on a port its program here does not use (the two ends' maps "
          "differ): refused\n",port->index,port->name,(unsigned long long)heard->key,heard->from);
      ready_refuse(port,heard,EPROTO);
      program_fail(owner,d,EPROTO,M->node);
    }else if(!(f=flow_make(port,heard->key,M->node,heard->from,queues,ready.slots,ready.stripe,ready.stripes,(int)ready.exclusive)))
      ready_refuse(port,heard,E2BIG);
    else{
      f->hold=1;f->heard=HEARD_READY;f->digest=ready.digest;
      if(a){flow_take(f,a->owner,a->destination);await_took(a,ready.stripe,ready.stripes);ready_say(f,&f->far,1);}
      flow_step(f);
    }
  }else{
    struct port *next[MESH_PORTS];
    struct flow *g=NULL;
    const uint32_t n=cables(route[1],next);
    f=n?flow_make(port,heard->key,heard->to,heard->from,queues,ready.slots,ready.stripe,ready.stripes,0):NULL;
    if(f)g=flow_make(stripe_cable(next,n,heard->key,heard->from,heard->to,ready.stripe),heard->key,heard->from,heard->to,queues,
                     ready.slots,ready.stripe,ready.stripes,0);
    if(!g){
      if(f){f->state=FAILED;flow_free(f);}
      ready_refuse(port,heard,n?E2BIG:EHOSTUNREACH);
    }else{
      f->partner=g;g->partner=f;
      int error=relay_direction(f,&ready,list,shape);
      if(error)flow_fail(f,error,1,FAIL,M->node,M->node);
      else{
        ready_say(g,route+1,ready.hops-1);
        f->heard=HEARD_READY;flow_step(f);
      }
    }
  }
  free(route);free(list);
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
   nothing, or that the prune drops at once, moves no serial); a program awaiting a node the prune drops fails as a
   value (EHOSTUNREACH naming it: the prune is that node's loss, design/bridge-and-ipc.md#routes) */
static void awaits_lost(uint32_t node);
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
  free(next);free(reached);free(order);
  for(uint32_t k=0;k<pruned;k++)awaits_lost(lost[k]);
  free(lost);
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

/* design/bridge-and-ipc.md#lanes: a lanes job on the port's queue: their queue pairs and completion queues made
   (LANES_MAKING); ready to receive from the peer's and to send, each lane listening for its first header and the
   port's progress threads started (LANES_CONNECTING: the event loop alone makes them up as it takes the job, so lanes
   reset while they connected stay down); or down once neither progress thread reads them, their queue pairs destroyed
   and their generation moved (LANES_CLOSING) */
static void lanes_job(struct port *port,int job){
  struct lanes *l=&port->lanes;
  l->job=job;atomic_store_explicit(&l->done,0,memory_order_relaxed);
  dispatch_async(port->work,^{
    int result=0;
    /* design/bridge-dataflow.md#stripes: guarded where the device holds a stripe flow (port_guard) */
    const int raised=guard_raise(port,0);
    if(job==LANES_MAKING)result=pair_up(&l->provider,MESH_LANES);
    else if(job==LANES_CONNECTING){
      result=pair_receive(&l->provider,(const uint32_t (*)[2])l->peer,port->heard.lid,port->heard.gid);
      if(!result)result=pair_send(&l->provider);
      const uint32_t sends=(uint32_t)l->provider.sent->cqe/MESH_LANES;
      for(uint32_t lane=0;!result && lane<MESH_LANES;lane++){
        l->gate[lane]=(struct send_gate){.capacity=MIN(l->provider.queues[lane].send_capacity,sends)};
        l->sent[lane]=l->received[lane]=0;l->cursor[lane]=0;l->records=0;
        result=lane_listen(port,lane);
      }
      if(!result)result=port_threads(port);
    }else{
      atomic_store_explicit(&l->up,0,memory_order_release);
      port_settle(port);
      result=down_pair(&l->provider)?0:EBUSY;
      if(!result)l->generation++;
    }
    guard_lower(port,raised);
    l->result=result;
    port_pairs(port);
    atomic_store_explicit(&l->done,1,memory_order_release);
    wake();
  });
}
/* the frames of payload a lane's receive queue takes beside its header: its queue pair's and its share of the receive
   completion queue, less the header's frame */
static uint32_t lanes_room(const struct lanes *l){
  uint32_t room=(uint32_t)l->provider.completion->cqe/MESH_LANES;
  for(uint32_t lane=0;lane<MESH_LANES;lane++)room=MIN(room,l->provider.queues[lane].receive_capacity);
  return room>1?room-1:0;
}
/* the port's flows stepped (its lanes changed) */
static void port_step(struct port *port){
  for(struct flow *f=flows,*next;f;f=next){next=f->next;if(f->port==port)flow_step(f);}
}
/* design/recovery.md#loss: lanes that failed (made or run) or whose peer's did are the lanes' alone: the flows that
   ride them and have posted on them fail (their records and SENDs went with them), the lanes are closed and made again
   at once, and the peer is told (a LANES of none and why) where the failure is this end's; the link stays paired, and
   its flows with queue pairs of their own and those still pairing run on.  Lanes that fail to connect are the link's
   loss, its device retired and opened again (a context that refuses every connection, as after a peer's restart, is
   mended by opening it again: lanes_turn) */
static void lanes_reset(struct port *port,int64_t code,const char *what,int tell){
  struct lanes *l=&port->lanes;
  say("link %u (%s): lanes toward node %u %s: %s; made again, the link paired and its own queue pairs running on\n",
      port->index,port->name,port->peer,what,strerror((int)code));
  if(tell)port_say(port,(struct message){.kind=LANES,.count=0,.code=code},NULL,0);
  atomic_store_explicit(&l->up,0,memory_order_release);
  for(struct flow *f=flows;f;f=f->next)
    if(f->port==port && !f->exclusive && f->state!=FAILED && (f->state>=POSTING || f->job==JOB_RECEIVE || f->job==JOB_RUN || f->job==JOB_CHANGE))
      flow_fail(f,code,1,FAIL,M->node,M->node);
  if(l->state!=LANES_DOWN)l->state=LANES_CLOSING;
  l->heard=0;
}
/* design/bridge-and-ipc.md#lanes: the port's lanes' turn, on the event loop: a job's end taken (made: LANES told the
   peer, its queue pairs and its room; connected: up, and the port's flows stepped; closed: down, the device's queue
   pairs given back; a make that failed: lanes_reset; a connect that failed: the link's loss, its device retired); a
   lane's failure is the lanes' (lanes_reset); the lanes made once the port pairs or once they closed, where the window
   gives their pages and the device has room for them (flow_make counts it before any flow's own; else made at the
   first turn they have, the cause logged once), connected once both ends made theirs, closed once the link went or
   they failed */
static void lanes_turn(struct port *port){
  struct lanes *l=&port->lanes;
  if(l->job){
    if(!atomic_load_explicit(&l->done,memory_order_acquire))return;
    const int job=l->job,result=l->result;
    l->job=0;
    if(job==LANES_CLOSING){
      if(result)say("link %u (%s): lanes teardown: %s\n",port->index,port->name,strerror(result));
      else{l->state=LANES_DOWN;if(l->committed)port->committed-=MESH_LANES;l->committed=0;}
    }else if(l->state==LANES_CLOSING){}
    else if(result && job==LANES_CONNECTING){
      /* design/recovery.md#loss: lanes not connected: the device opened again, as a new bridge would open it */
      say("link %u (%s): lanes toward node %u not connected: %s; the device retired and opened again\n",port->index,port->name,
          port->peer,strerror(result));
      lanes_reset(port,result,"not connected",1);
      port_lost(port,result,1);
    }
    else if(result)lanes_reset(port,result,"not made",1);
    else if(job==LANES_MAKING){
      l->state=LANES_MADE;l->room=lanes_room(l);
      struct message said={.kind=LANES,.count=MESH_LANES,.code=l->room};
      for(uint32_t lane=0;lane<MESH_LANES;lane++){said.pair[lane][0]=l->provider.queues[lane].pair->qp_num;said.pair[lane][1]=(l->provider.psn+lane)&0xffffff;}
      port_say(port,said,NULL,0);
    }else{
      l->state=LANES_UP;l->code=0;
      atomic_store_explicit(&l->up,1,memory_order_release);port_bell(port);
      say("link %u (%s): lanes up toward node %u: %u queue pairs, %u frames a header\n",port->index,port->name,port->peer,MESH_LANES,
          MIN(l->room,l->peer_room));
      port_step(port);
    }
  }
  const int64_t failed=atomic_exchange(&l->failed,0);
  if(failed && port->greeted && (l->state==LANES_UP || l->state==LANES_CONNECTING))lanes_reset(port,failed,"failed",1);
  if(l->job)return;
  if(l->state==LANES_CLOSING)lanes_job(port,LANES_CLOSING);
  else if(port->greeted && l->state==LANES_DOWN){
    if(l->page==MESH_ABSENT && (l->page=mesh_arena_alloc(&bridge,l->pages,1))==MESH_ABSENT){
      if(l->code!=ENOMEM)say("link %u (%s): lanes not made: the window holds no %u pages for them; made once it has\n",port->index,
                             port->name,l->pages);
      l->code=ENOMEM;
    }else if(port->committed+MESH_LANES>port->device.max_pairs){
      if(l->code!=ENOSPC)say("link %u (%s): lanes not made: the device holds %u of its %u usable queue pairs; made once it has room\n",
                             port->index,port->name,port->committed,port->device.max_pairs);
      l->code=ENOSPC;
    }else{port->committed+=MESH_LANES;l->committed=1;l->state=LANES_MAKING;lanes_job(port,LANES_MAKING);}
  }
  else if(port->greeted && l->state==LANES_MADE && l->heard){l->state=LANES_CONNECTING;lanes_job(port,LANES_CONNECTING);}
}
/* a peer's LANES: its lanes' queue pairs and the frames a header may name, or (count 0) its lanes failed, with why:
   lanes of this end that were connected, or connecting, to its old ones are reset (lanes_reset, untold: the peer knows),
   and lanes made and not yet connected wait for its new ones.  A peer's new lanes heard while this end's are connected
   to its old ones reset them alike, the new ones kept for the next connect */
static void lanes_heard(struct port *port,const struct message *heard){
  struct lanes *l=&port->lanes;
  const int joined=l->state==LANES_UP || l->state==LANES_CONNECTING;
  if(heard->count!=MESH_LANES){
    if(joined)lanes_reset(port,heard->code?heard->code:EPROTO,"failed at its peer",0);
    else{say("link %u (%s): node %u's lanes failed: %s\n",port->index,port->name,port->peer,strerror((int)heard->code));l->heard=0;}
    return;
  }
  if(joined)lanes_reset(port,EPROTO,"made again by its peer",0);
  memcpy(l->peer,heard->pair,sizeof l->peer);
  l->peer_room=(uint32_t)heard->code;l->heard=1;
  lanes_turn(port);
}

/* design/bridge-dataflow.md#stripes: the largest piece this end's device takes in one SEND (its port's max_msg_sz, in
   whole frames, at most MESH_STRIPE_PIECE: mesh 8fc27cbe's 4 MiB, MESH_STRIPE_CHUNK) */
static uint32_t stripes_piece(const struct port *port){
  const uint32_t most=port->device.port.max_msg_sz/4096*4096;
  return most && most<MESH_STRIPE_PIECE?most:MESH_STRIPE_PIECE;
}
/* the stripe flow ready to run, on the port's queue once its queue pairs are connected: the piece both ends cut by, its
   queue pairs' frames less the descriptors their completion queues hold beside them, every count from zero and every
   descriptor slot's receive posted; 0, or the post's error */
static int stripes_ready(struct port *port){
  struct stripes *s=&port->stripes;
  const uint32_t own=stripes_piece(port);
  s->piece=s->peer_piece && s->peer_piece/4096*4096<own?s->peer_piece/4096*4096:own;
  if(!s->piece)s->piece=4096;
  const uint32_t sends=(uint32_t)s->provider.sent->cqe,receives=(uint32_t)s->provider.completion->cqe;
  s->send_capacity=MIN(s->provider.queues[1].send_capacity,sends>MESH_DESCRIPTORS+1?sends-MESH_DESCRIPTORS-1:1);
  s->receive_capacity=MIN(s->provider.queues[1].receive_capacity,receives>MESH_DESCRIPTORS+1?receives-MESH_DESCRIPTORS-1:1);
  s->send_head=s->send_tail=s->descriptors_sent=s->descriptors_done=s->receive_head=s->receive_tail=s->arrived_head=s->arrived_tail=0;
  s->send_frames=s->receive_frames=s->cursor=0;
  s->out=s->in=(struct stripe_cut){0};
  int error=s->provider.queues[0].receive_capacity<MESH_DESCRIPTORS || s->provider.queues[0].send_capacity<MESH_DESCRIPTORS?ENOSPC:0;
  for(uint32_t slot=0;slot<MESH_DESCRIPTORS && !error;slot++)error=stripe_listen(port,slot);
  return error;
}
/* the requests a stripe flow took and had not completed when it stopped (the threads read it no more): each failed as
   a value (its writer's word UINT64_MAX, ECONNRESET) */
static void stripes_abandon(struct port *port){
  struct stripes *s=&port->stripes;
  uint64_t abandoned=0;
  if(s->out.active){stripe_failed(s->out.ring,s->out.owner,s->out.d.word,ECONNRESET);s->out.active=0;abandoned++;}
  for(;s->send_tail!=s->send_head;s->send_tail++){
    const struct stripe_piece *p=s->sent+s->send_tail%MESH_STRIPE_CHUNKS;
    if(p->last){stripe_failed(p->ring,p->owner,p->word,ECONNRESET);abandoned++;}
  }
  atomic_fetch_sub_explicit(&s->flight,abandoned,memory_order_release);
}
/* design/bridge-dataflow.md#stripes: a stripe flow's job on its port's queue, always guarded (port_guard): its two
   queue pairs and completion queues made (STRIPES_MAKING); connected to the peer's, ready (stripes_ready) and the port's
   progress threads started (STRIPES_CONNECTING: the event loop alone makes it up as it takes the job); or, once neither
   progress thread reads it, the requests it held failed, its queue pairs moved to the error state and their flushed
   completions drained (pair_flush), then destroyed (STRIPES_CLOSING): the one owner of its teardown, and the device's
   registrations stay until the device retires (down_device, port_turn) */
static void stripes_job(struct port *port,int job){
  struct stripes *s=&port->stripes;
  s->job=job;atomic_store_explicit(&s->done,0,memory_order_relaxed);
  dispatch_async(port->work,^{
    int result=0;
    if(job==STRIPES_CLOSING){
      atomic_store_explicit(&s->up,0,memory_order_release);
      port_settle(port);
      stripes_abandon(port);
    }
    port_guard(port,1);
    if(job==STRIPES_MAKING)result=pair_up(&s->provider,2);
    else if(job==STRIPES_CONNECTING){
      result=pair_receive(&s->provider,(const uint32_t (*)[2])s->peer,port->heard.lid,port->heard.gid);
      if(!result)result=pair_send(&s->provider);
      if(!result)result=stripes_ready(port);
      if(!result)result=port_threads(port);
    }else{
      s->flushed=s->provider.queues?(uint32_t)pair_flush(&s->provider):0;
      result=down_pair(&s->provider)?0:EBUSY;
    }
    port_guard(port,0);
    s->result=result;
    port_pairs(port);
    atomic_store_explicit(&s->done,1,memory_order_release);
    wake();
  });
}
/* design/recovery.md#loss: a stripe flow that failed (made, connected or run) or whose peer's did: the requests it held
   fail as values, it is closed and made again as a request needs it (one that could not be made or connected: the next
   request toward its peer fails as a value by its code and the one after makes it again, stripes_requests_turn), and
   the peer is told (a STRIPES of none and why) where the failure is this end's; the
   link, its lanes and its flows run on */
static void stripes_reset(struct port *port,int64_t code,const char *what,int tell){
  struct stripes *s=&port->stripes;
  say("link %u (%s): stripes toward node %u %s: %s (%llu written, %llu landed)\n",port->index,port->name,port->peer,what,
      strerror((int)code),(unsigned long long)atomic_load(&s->written),(unsigned long long)atomic_load(&s->landed));
  if(tell)port_say(port,(struct message){.kind=STRIPES,.count=0,.code=code},NULL,0);
  atomic_store_explicit(&s->up,0,memory_order_release);
  if(s->state!=STRIPES_DOWN)s->state=STRIPES_CLOSING;
  s->heard=0;s->wanted=0;s->code=strcmp(what,"not made") && strcmp(what,"not connected")?0:code;
}
/* design/bridge-dataflow.md#stripes: the port's stripe flow's turn, on the event loop, as lanes_turn's: a job's end
   taken (made: STRIPES told the peer, its queue pairs and its piece; connected: up; closed: down, the device's queue
   pairs given back; a make or connect that failed: stripes_reset); its failure reset; made where it is wanted, the link
   paired and no failure's code stands (a request failed by it clears it), where the window gives its pages and the
   device has room for its two queue pairs (else `code` says why, and the next request naming the peer fails as a value
   by it: stripes_requests_turn), connected once both ends made theirs, closed once it failed or the link went */
static void stripes_turn(struct port *port){
  struct stripes *s=&port->stripes;
  if(port->greeted){
    const struct mesh_record *r=record_held(port->peer);
    atomic_store_explicit(&s->window,r?r->window:0,memory_order_relaxed);
  }
  if(s->job){
    if(!atomic_load_explicit(&s->done,memory_order_acquire))return;
    const int job=s->job,result=s->result;
    s->job=0;
    if(job==STRIPES_CLOSING){
      if(result)say("link %u (%s): stripes teardown: %s\n",port->index,port->name,strerror(result));
      else{
        say("link %u (%s): stripes closed (%u completions flushed)\n",port->index,port->name,s->flushed);
        s->state=STRIPES_DOWN;if(s->committed)port->committed-=2;s->committed=0;
      }
    }else if(s->state==STRIPES_CLOSING){}
    else if(result)stripes_reset(port,result,job==STRIPES_MAKING?"not made":"not connected",1);
    else if(job==STRIPES_MAKING){
      s->state=STRIPES_MADE;
      struct message said={.kind=STRIPES,.count=2,.code=stripes_piece(port)};
      for(uint32_t q=0;q<2;q++){said.pair[q][0]=s->provider.queues[q].pair->qp_num;said.pair[q][1]=(s->provider.psn+q)&0xffffff;}
      port_say(port,said,NULL,0);
    }else{
      s->state=STRIPES_UP;s->code=0;
      atomic_store_explicit(&s->up,1,memory_order_release);port_bell(port);
      say("link %u (%s): stripes up toward node %u: pieces of %u bytes\n",port->index,port->name,port->peer,s->piece);
    }
  }
  const int64_t failed=atomic_exchange(&s->failed,0);
  if(failed && port->greeted && (s->state==STRIPES_UP || s->state==STRIPES_CONNECTING))stripes_reset(port,failed,"failed",1);
  if(s->job)return;
  if(s->state==STRIPES_CLOSING)stripes_job(port,STRIPES_CLOSING);
  else if(port->greeted && s->state==STRIPES_DOWN && s->wanted && !s->code){
    if(!s->pages){
      const uint32_t pages=(uint32_t)((2*MESH_DESCRIPTORS*sizeof(struct mesh_descriptor)+M->pgsz-1)/M->pgsz);
      const uint32_t page=mesh_arena_alloc(&bridge,pages,pages);
      if(page!=MESH_ABSENT){s->page=page;s->pages=pages;}
    }
    if(!s->pages){
      if(s->code!=ENOMEM)say("link %u (%s): stripes not made: the window holds no pages for their descriptors\n",port->index,port->name);
      s->code=ENOMEM;
    }else if(port->committed+2>port->device.max_pairs){
      if(s->code!=ENOSPC)say("link %u (%s): stripes not made: the device holds %u of its %u usable queue pairs\n",port->index,port->name,
                             port->committed,port->device.max_pairs);
      s->code=ENOSPC;
    }else{port->committed+=2;s->committed=1;s->state=STRIPES_MAKING;stripes_job(port,STRIPES_MAKING);}
  }
  else if(port->greeted && s->state==STRIPES_MADE && s->heard){s->state=STRIPES_CONNECTING;stripes_job(port,STRIPES_CONNECTING);}
}
/* a peer's STRIPES: its stripe flow's queue pairs and its largest piece (this end makes its own where it has none), or
   (count 0) its stripe flow failed, with why: this end's, where joined to it, reset untold */
static void stripes_heard(struct port *port,const struct message *heard){
  struct stripes *s=&port->stripes;
  const int joined=s->state==STRIPES_UP || s->state==STRIPES_CONNECTING;
  if(heard->count!=2){
    if(joined)stripes_reset(port,heard->code?heard->code:EPROTO,"failed at its peer",0);
    else s->heard=0;
    return;
  }
  if(joined)stripes_reset(port,EPROTO,"made again by its peer",0);
  memcpy(s->peer,heard->pair,sizeof s->peer);
  s->peer_piece=(uint32_t)heard->code;s->heard=1;s->wanted=1;s->code=0;
  stripes_turn(port);
}
/* design/bridge-dataflow.md#stripes: every registered ring's next request read on the event loop: a ring whose client
   is gone deregistered; a request naming a node that a stripe flow of this node's carries toward is left to the send
   threads of the ports up toward it, which take it; else the ports paired with its node want their stripe flows made
   (stripes_turn), and it waits while one is being made; and it fails as a value (its writer's word UINT64_MAX, its
   ring's code) where its node is this node (EINVAL) or no cable's (EHOSTUNREACH: a stripe goes over one link, its
   node a neighbour), or where every cable's stripe flow last failed or could not be made (its code: ENOSPC, ENOMEM, a
   connection's), which clears those codes, so the next request makes them again */
static void stripes_requests_turn(void){
  _Atomic uint64_t *rings=mesh_rings(M);
  for(uint32_t r=0;r<MESH_RINGS;r++){
    uint64_t at=atomic_load_explicit(rings+r,memory_order_acquire);
    if(!at)continue;
    struct mesh_requests *q=at>=M->data_off && at<=M->length-sizeof(struct mesh_requests)?(void *)((char *)M+at):NULL;
    if(!q || !owned(q->owner) || !q->entries || (q->entries&(q->entries-1)) || q->entries>(M->length-at-sizeof *q)/sizeof(struct mesh_stripe)){
      atomic_compare_exchange_strong(rings+r,&at,0);
      continue;
    }
    uint64_t k=atomic_load_explicit(&q->taken,memory_order_acquire);
    const struct mesh_stripe *e=q->request+(k&(q->entries-1));
    if(atomic_load_explicit(&e->ready,memory_order_acquire)!=k+1)continue;
    const uint32_t node=e->node;
    struct port *cable[MESH_PORTS];
    const uint32_t n=node==M->node?0:cables(node,cable);
    int64_t code=node==M->node?EINVAL:n?0:EHOSTUNREACH;
    int up=0,failing=n>0;
    for(uint32_t j=0;j<n;j++){
      struct stripes *s=&cable[j]->stripes;
      up|=s->state==STRIPES_UP;
      if(s->state==STRIPES_DOWN && s->code){if(!code)code=s->code;}
      else failing=0;
      if(s->state!=STRIPES_UP && !s->code)s->wanted=1;
    }
    if(up || (n && !failing))continue;
    const uint64_t word=e->word,owner=q->owner;
    if(!atomic_compare_exchange_strong_explicit(&q->taken,&k,k+1,memory_order_acq_rel,memory_order_relaxed))continue;
    if(atomic_load_explicit(&q->code,memory_order_relaxed)!=code)
      say("stripes: a request of client %llx toward node %u refused: %s\n",(unsigned long long)owner,node,strerror((int)code));
    stripe_failed(at,owner,word,code);
    for(uint32_t j=0;j<n;j++)cable[j]->stripes.code=0;
  }
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
enum { END_SOCKET=1, END_PORT, END_EXEC };
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
static void channel_end(struct channel *c,int64_t code,int from){
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
static struct port *channel_hop(uint32_t node){
  uint32_t *route=NULL;
  const uint32_t hops=mesh_route(M,M->node,node,&route);
  struct port *next[MESH_PORTS];
  struct port *port=hops>=2 && cables(route[1],next)>0?next[0]:NULL;
  free(route);
  return port;
}
static void channel_pump(struct channel *c);
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
    if(e->kind==END_SOCKET)shutdown(e->fd,SHUT_WR);
    else channel_say(c,t,CHANNEL_SHUT,0,0,NULL,0);
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
static void stream_asked(int fd,uint32_t node,uint64_t bytes);
/* a client's request read whole: a channel opened toward its node's service (here, or its CHANNEL_OPEN sent to the next hop),
   a service registered, a ledger cut, or a raw stream (design/bridge-dataflow.md#streams) */
static void asked(struct asking *a){
  struct mesh_request *r=&a->request;
  const int fd=a->fd;a->fd=-1;
  fd_watch(fd,EVFILT_READ,0,0);
  r->service[sizeof r->service-1]=0;r->path[sizeof r->path-1]=0;
  if(r->magic==MESH_REQUEST_MAGIC && r->kind==MESH_REQUEST_STREAM){stream_asked(fd,r->node,r->bytes);return;}
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

/* design/bridge-dataflow.md#streams: a stream's verbs work on its port's queue, always guarded (port_guard): its queue
   pair made (STREAMS_MAKING); connected to the peer's and ready (STREAMS_CONNECTING: its chains' storage, and at the
   receiver its first `window` receives posted before it tells POSTED, so no SEND of the sender's comes before its
   receive: the 2026-09-10 benchmark's first defect, design/transport-one-gib-2026-09-10.md); or, once neither
   progress thread reads it, flushed and destroyed (STREAMS_CLOSING) */
static void streams_job(struct port *port,int job){
  struct stream *s=&port->stream;
  s->job=job;atomic_store_explicit(&s->done,0,memory_order_relaxed);
  dispatch_async(port->work,^{
    int result=0;
    if(job==STREAMS_CLOSING){atomic_store_explicit(&s->up,0,memory_order_release);port_settle(port);}
    port_guard(port,1);
    if(job==STREAMS_MAKING)result=pair_up(&s->provider,1);
    else if(job==STREAMS_CONNECTING){
      result=pair_receive(&s->provider,(const uint32_t (*)[2])s->peer,port->heard.lid,port->heard.gid);
      if(!result)result=pair_send(&s->provider);
      const uint32_t frames=(s->page_bytes+4095)/4096,cqe=(uint32_t)MIN(s->provider.sent->cqe,s->provider.completion->cqe);
      s->window=MIN(MESH_STREAM_WINDOW,MIN(s->sending?s->provider.queues[0].send_capacity/frames:s->provider.queues[0].receive_capacity/frames,
                                           cqe>1?cqe-1:1));
      if(!s->window)s->window=1;
      s->spans=calloc(s->window,sizeof *s->spans);s->sends=calloc(s->window,sizeof *s->sends);
      s->receives=calloc(s->window,sizeof *s->receives);s->completions=calloc(s->window,sizeof *s->completions);
      if(!result && (!s->spans || !s->sends || !s->receives || !s->completions))result=ENOMEM;
      s->posted=s->completed=0;
      if(!result && !s->sending)result=stream_post(port,MIN(s->window,s->count));
      if(!result)result=port_threads(port);
    }else{
      s->flushed=s->provider.queues?pair_flush(&s->provider):0;
      result=down_pair(&s->provider)?0:EBUSY;
      free(s->spans);free(s->sends);free(s->receives);free(s->completions);
      s->spans=NULL;s->sends=NULL;s->receives=NULL;s->completions=NULL;
    }
    port_guard(port,0);
    s->result=result;
    port_pairs(port);
    atomic_store_explicit(&s->done,1,memory_order_release);
    wake();
  });
}
/* a stream's lease filled with its pattern (the sender: word w of page p is p << 32 ^ w ^ MESH_STREAM_MAGIC, as
   mesh-verbs-stream.c and mesh-page-stream.c wrote it), cleared (the receiver, so a page not written fails its check),
   or checked word by word (the receiver, once every page landed); off the event loop, on a global queue */
static void streams_work(struct port *port,int job){
  struct stream *s=&port->stream;
  s->job=job;atomic_store_explicit(&s->done,0,memory_order_relaxed);
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^{
    const uint64_t words=s->page_bytes/sizeof(uint64_t);
    uint64_t mismatches=0,correct=0;
    for(uint64_t p=0;p<s->count;p++){
      uint64_t *page=(uint64_t *)(wire.data+stream_offset(port,p)),bad=0;
      if(job==STREAMS_FILLING && s->sending)for(uint64_t w=0;w<words;w++)page[w]=(p<<32)^w^MESH_STREAM_MAGIC;
      else if(job==STREAMS_FILLING)memset(page,0,s->page_bytes);
      else for(uint64_t w=0;w<words;w++)bad+=page[w]!=((p<<32)^w^MESH_STREAM_MAGIC);
      mismatches+=bad;correct+=!bad;
    }
    if(job==STREAMS_CHECKING){s->own.mismatches=mismatches;s->own.correct=correct;}
    s->result=0;
    atomic_store_explicit(&s->done,1,memory_order_release);
    wake();
  });
}
/* the stream's lease returned to the window and its state cleared, the requester (the sender's end) answered with
   `code` where it still waits */
static void streams_clear(struct port *port,int64_t code){
  struct stream *s=&port->stream;
  if(s->pages)mesh_bits_clear(mesh_arena_bits(M),s->page,s->pages);
  if(s->fd>=0){fd_answer(s->fd,(int32_t)code);close(s->fd);}
  if(s->committed)port->committed-=1;
  *s=(struct stream){.fd=-1};
}
/* design/recovery.md#loss: a stream that failed (a verbs call's error, a completion's, the peer's FAIL, the link lost):
   the peer told where the failure is this end's, the requester answered with the code, its queue
   pair closed (flushed) where it has one */
static void streams_fail(struct port *port,int64_t code,int tell){
  struct stream *s=&port->stream;
  say("link %u (%s): stream %016llx toward node %u: %s\n",port->index,port->name,(unsigned long long)s->key,port->peer,strerror((int)code));
  if(tell)port_say(port,(struct message){.kind=STREAM,.count=STREAM_FAILED,.key=s->key,.code=code},NULL,0);
  atomic_store_explicit(&s->up,0,memory_order_release);
  if(s->fd>=0){fd_answer(s->fd,(int32_t)code);close(s->fd);s->fd=-1;}
  if(s->job)s->state=STREAMS_CLOSING;
  else if(s->provider.queues || s->state>=STREAMS_MAKING){s->state=STREAMS_CLOSING;streams_job(port,STREAMS_CLOSING);}
  else streams_clear(port,0);
}
/* the stream's lease, `bytes` in whole blocks from the window; 0, or ENOMEM */
static int streams_lease(struct port *port,uint64_t bytes){
  struct stream *s=&port->stream;
  const uint64_t block=(uint64_t)M->block*M->pgsz;
  s->page_bytes=(uint32_t)block;s->count=(uint32_t)(bytes/block);s->bytes=(uint64_t)s->count*block;
  const uint64_t pages=s->bytes/M->pgsz;
  const uint32_t page=pages && pages<=UINT32_MAX?mesh_arena_alloc(&bridge,(uint32_t)pages,M->block):MESH_ABSENT;
  if(page==MESH_ABSENT)return ENOMEM;
  s->page=page;s->pages=(uint32_t)pages;
  return 0;
}
/* design/bridge-dataflow.md#streams: a client's request for a raw stream of `bytes` toward node `node` (a neighbour),
   its socket held for the answer: the first cable's port toward it, its lease filled, then its queue pair made and
   offered (streams_turn); refused at once as a value where the node has no cable (EHOSTUNREACH), that port runs a
   stream (EBUSY), the bytes are not a block at least or exceed 2^32 blocks (EINVAL), or the window has no room
   (ENOMEM) */
static void stream_asked(int fd,uint32_t node,uint64_t bytes){
  struct port *cable[MESH_PORTS];
  const uint32_t n=node==M->node?0:cables(node,cable);
  const uint64_t block=(uint64_t)M->block*M->pgsz;
  int64_t code=!n?EHOSTUNREACH:bytes<block || bytes/block>UINT32_MAX?EINVAL:cable[0]->stream.state!=STREAMS_DOWN?EBUSY:0;
  if(!code)code=streams_lease(cable[0],bytes);
  if(code){fd_answer(fd,(int32_t)code);close(fd);if(n && code==ENOMEM)streams_clear(cable[0],0);return;}
  struct stream *s=&cable[0]->stream;
  static uint32_t streams;
  s->fd=fd;s->sending=1;s->key=(uint64_t)M->node<<32|++streams;
  s->state=STREAMS_FILLING;
  say("link %u (%s): stream %016llx of %llu bytes toward node %u asked\n",cable[0]->index,cable[0]->name,(unsigned long long)s->key,
      (unsigned long long)s->bytes,node);
  streams_work(cable[0],STREAMS_FILLING);
}
/* a peer's STREAM: its offer (this end leases, clears, makes and connects, then tells POSTED), its posted receives (this
   end connects and sends), its result (the requester answered), or its failure */
static void stream_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  struct stream *s=&port->stream;
  if(heard->count==STREAM_OFFER){
    int64_t code=s->state!=STREAMS_DOWN?EBUSY:0;
    if(!code)code=streams_lease(port,(uint64_t)heard->code);
    if(code){
      if(s->state==STREAMS_DOWN)streams_clear(port,0);
      port_say(port,(struct message){.kind=STREAM,.count=STREAM_FAILED,.key=heard->key,.code=code},NULL,0);
      return;
    }
    s->fd=-1;s->sending=0;s->key=heard->key;s->peer[0][0]=heard->pair[0][0];s->peer[0][1]=heard->pair[0][1];
    s->state=STREAMS_FILLING;
    streams_work(port,STREAMS_FILLING);
    return;
  }
  if(heard->key!=s->key || s->state==STREAMS_DOWN)return;
  if(heard->count==STREAM_POSTED && s->sending && s->state==STREAMS_MADE){
    s->peer[0][0]=heard->pair[0][0];s->peer[0][1]=heard->pair[0][1];
    s->state=STREAMS_CONNECTING;streams_job(port,STREAMS_CONNECTING);
  }else if(heard->count==STREAM_RESULT && s->sending && heard->bytes==sizeof s->told){
    memcpy(&s->told,payload,sizeof s->told);
    const struct stream_result *r=&s->told,*o=&s->own;
    const double interior=r->high_ns>r->low_ns?(double)(r->high_ns-r->low_ns)/1e9:0;
    const uint64_t interior_bytes=(r->high_count-r->low_count)*r->page_bytes;
    char line[1024];
    snprintf(line,sizeof line,"{\"stream\":\"raw\",\"key\":\"%016llx\",\"node\":%u,\"peer\":%u,\"port\":%u,\"bytes\":%llu,\"page_bytes\":%u,"
      "\"pages\":%u,\"window\":%u,\"sender\":{\"window\":%u,\"posts\":%llu,\"submit_to_last_s\":%.9f,\"polls\":%llu,\"empty_polls\":%llu,"
      "\"misordered\":%llu},\"receiver\":{\"window\":%llu,\"completed\":%llu,\"mismatches\":%llu,\"correct_pages\":%llu,\"misordered\":%llu,"
      "\"first_to_last_s\":%.9f,\"first_to_last_bytes\":%llu,\"interior_s\":%.9f,\"interior_bytes\":%llu,\"interior_GBps\":%.6f,\"polls\":%llu,"
      "\"empty_polls\":%llu}}\n",(unsigned long long)s->key,M->node,port->peer,port->index,(unsigned long long)s->bytes,s->page_bytes,s->count,
      s->window,s->window,(unsigned long long)o->posts,o->last_ns>o->begin_ns?(double)(o->last_ns-o->begin_ns)/1e9:0,(unsigned long long)o->polls,
      (unsigned long long)o->empty_polls,(unsigned long long)o->misordered,(unsigned long long)r->window,(unsigned long long)r->completed,
      (unsigned long long)r->mismatches,(unsigned long long)r->correct,(unsigned long long)r->misordered,
      r->last_ns>r->first_ns?(double)(r->last_ns-r->first_ns)/1e9:0,(unsigned long long)((r->completed-r->first_count)*r->page_bytes),interior,
      (unsigned long long)interior_bytes,interior>0?(double)interior_bytes/interior/1e9:0,(unsigned long long)r->polls,
      (unsigned long long)r->empty_polls);
    say("link %u (%s): stream %s",port->index,port->name,line);
    if(s->fd>=0){fd_answer(s->fd,0);if(send(s->fd,line,strlen(line),0)<0){}close(s->fd);s->fd=-1;}
    s->state=STREAMS_CLOSING;
    if(!s->job)streams_job(port,STREAMS_CLOSING);
  }else if(heard->count==STREAM_FAILED)streams_fail(port,heard->code?heard->code:ECANCELED,0);
}
/* design/bridge-dataflow.md#streams: the port's stream's turn, on the event loop: a job's end taken (its lease filled
   or cleared: its queue pair made; made: the sender offers it, the receiver connects; connected: running, the receiver
   telling POSTED; checked: the receiver tells its result and closes; closed: its lease returned), its end at a
   progress thread taken (the receiver checks every word; the sender waits for the result), and its failure ending it
   (a verbs call's or a completion's error, the peer's, the link's loss) */
static void streams_turn(struct port *port){
  struct stream *s=&port->stream;
  if(s->state==STREAMS_DOWN)return;
  if(s->job){
    if(!atomic_load_explicit(&s->done,memory_order_acquire))return;
    const int job=s->job,result=s->result;
    s->job=0;
    if(job==STREAMS_CLOSING){
      if(result){say("link %u (%s): stream teardown: %s\n",port->index,port->name,strerror(result));return;}
      say("link %u (%s): stream %016llx closed (%llu completions flushed)\n",port->index,port->name,(unsigned long long)s->key,
          (unsigned long long)s->flushed);
      streams_clear(port,ECANCELED);
      return;
    }
    if(s->state==STREAMS_CLOSING){streams_job(port,STREAMS_CLOSING);return;}
    if(result){streams_fail(port,result,1);return;}
    if(job==STREAMS_FILLING){
      if(port->committed+1>port->device.max_pairs){streams_fail(port,ENOSPC,1);return;}
      port->committed+=1;s->committed=1;s->state=STREAMS_MAKING;streams_job(port,STREAMS_MAKING);
    }else if(job==STREAMS_MAKING){
      s->state=STREAMS_MADE;
      if(s->sending){
        struct message said={.kind=STREAM,.count=STREAM_OFFER,.key=s->key,.code=(int64_t)s->bytes};
        said.pair[0][0]=s->provider.queues[0].pair->qp_num;said.pair[0][1]=s->provider.psn&0xffffff;
        port_say(port,said,NULL,0);
      }else{s->state=STREAMS_CONNECTING;streams_job(port,STREAMS_CONNECTING);}
    }else if(job==STREAMS_CONNECTING){
      s->state=STREAMS_RUNNING;
      atomic_store_explicit(&s->up,1,memory_order_release);port_bell(port);
      if(!s->sending){
        struct message said={.kind=STREAM,.count=STREAM_POSTED,.key=s->key};
        said.pair[0][0]=s->provider.queues[0].pair->qp_num;said.pair[0][1]=s->provider.psn&0xffffff;
        port_say(port,said,NULL,0);
      }
    }else if(job==STREAMS_CHECKING){
      s->own.bytes=s->bytes;s->own.pages=s->count;s->own.page_bytes=s->page_bytes;s->own.window=s->window;
      port_say(port,(struct message){.kind=STREAM,.count=STREAM_RESULT,.key=s->key},&s->own,sizeof s->own);
      s->state=STREAMS_CLOSING;streams_job(port,STREAMS_CLOSING);
    }
    return;
  }
  const int64_t failed=atomic_exchange(&s->failed,0);
  if(failed){streams_fail(port,failed,1);return;}
  if(s->state==STREAMS_RUNNING && atomic_load_explicit(&s->finished,memory_order_acquire) && !s->sending){
    s->state=STREAMS_CHECKING;streams_work(port,STREAMS_CHECKING);
  }
}

/* design/bridge-and-ipc.md#protocol */
static void port_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  if(!port->greeted){
    if(heard->kind!=HELLO || heard->block!=M->block*M->pgsz || (port->peer && heard->node!=port->peer)){port_lost(port,EPROTO,0);return;}
    port->peer=heard->node;port->peer_port=heard->port;port->heard=*heard;port->greeted=1;
    struct mesh_link_info *link=&mesh_links(M)[port->index];
    atomic_store_explicit(&link->bandwidth,port->bandwidth,memory_order_relaxed);
    atomic_fetch_add_explicit(&link->pairings,1,memory_order_relaxed);
    say("link %u (%s): paired with node %u\n",port->index,port->name,port->peer);
    link_set(port,port->peer,MESH_PAIRED,0);
    /* a port newly paired is sent every record the map holds, this node's first */
    for(uint64_t i=0;port->greeted && i<atomic_load(&M->map_nodes);i++)if(mesh_records(M)[i].node!=M->node)
      port_say(port,(struct message){.kind=RECORD},mesh_records(M)+i,sizeof(struct mesh_record));
    return;
  }
  if(heard->kind==RECORD){record_heard(port,payload,heard->bytes);return;}
  /* design/recovery.md#detection: a peer's FAREWELL is its bridge's stop: its flows are lost now */
  if(heard->kind==FAREWELL){port->parted=1;port->farewell=heard->code;port_lost(port,ECONNRESET,0);return;}
  if(heard->kind==READY){ready_heard(port,heard,payload);return;}
  if(heard->kind==LANES){lanes_heard(port,heard);return;}
  if(heard->kind==STRIPES){stripes_heard(port,heard);return;}
  if(heard->kind==STREAM){stream_heard(port,heard,payload);return;}
  if(heard->kind>=CHANNEL_OPEN && heard->kind<=CHANNEL_CLOSE){channel_heard(port,heard,payload);return;}
  struct flow *f=flow_on(port,heard->key,heard->to,heard->from,heard->stripe);
  if(!f)return;
  const uint32_t origin=heard->failed?heard->failed:port->peer;
  if(heard->kind==PAIRS){
    /* #lanes: an exclusive flow's PAIRS its queue pairs, a lane rider's (count 0) its index and generation */
    const int error=flow_mode(f,heard->count!=0)?EPROTO:f->exclusive && heard->count!=(uint32_t)f->qps+1?EPROTO:0;
    if(error)flow_fail(f,error,4,FAIL,port->peer,M->node);
    else{
      /* #control: an exclusive flow's count is one past its data's queue pairs, its control pair last */
      if(f->exclusive){memcpy(f->pairs,heard->pair,sizeof f->pairs);f->peer_alive=1;}
      else{f->peer_slot=(uint32_t)heard->code;f->peer_generation=(uint32_t)((uint64_t)heard->code>>32);}
      f->heard|=HEARD_PAIRS;
    }
  }
  else if(heard->kind==POSTED){f->heard|=HEARD_POSTED;if(f->partner)flow_tell(f->partner);}
  else if(heard->kind==FAIL){
    const uint64_t owner=f->partner?0:f->owner;
    flow_fail(f,heard->code==ESTALE?ESTALE:ECANCELED,4,0,origin,origin);
    if(heard->code==ESTALE && owner)program_stale(owner,origin);
  }
  flow_step(f);
}

/* design/recovery.md#detection, #loss: a link lost fails every program on it, naming its peer (a flow failed and still
   telling its FAIL stops now); a port that went away (its link off, its interface detached) retires its device
   (design/RDMA-KERNEL-RECOVERY.md); a link that had paired says HELLO again at once, one that never paired at the next
   event (a HELLO heard, its link on, the configuration's HELLO interval) */
static void port_lost(struct port *port,int64_t code,int retire){
  if(port->control>=0){close(port->control);port->control=-1;}
  /* design/recovery.md#diagnosis: the peer named, and a peer that said farewell told from one lost */
  if(port->greeted||port->connecting){
    const char *retired=retire?", device retired":"";
    char reason[96];farewell_text(port->farewell,reason,sizeof reason);
    if(stop)say("link %u (%s): closed toward node %u: this bridge is stopping%s\n",port->index,port->name,port->peer,retired);
    else if(port->peer && port->parted)say("link %u (%s): node %u parted, its bridge stopping (%s): code %lld%s\n",port->index,port->name,
                                            port->peer,reason,(long long)code,retired);
    else if(port->peer)say("link %u (%s): node %u lost: code %lld%s\n",port->index,port->name,port->peer,(long long)code,retired);
    else say("link %u (%s): lost: code %lld%s\n",port->index,port->name,(long long)code,retired);
  }
  /* #lanes: read by no progress thread once the stops of its flows (queued after) settle the port's table, and closed
     once those stops have run */
  atomic_store_explicit(&port->lanes.up,0,memory_order_release);
  for(struct flow *f=flows;f;f=f->next)if(f->port==port){
    flow_fail(f,port->parted?ECONNRESET:code,4,0,port->peer,M->node);
    if(f->state==FAILED && !f->stopping && (atomic_load_explicit(&f->ending,memory_order_acquire)&FLOW_TELLING))
      flow_stop(f,f->stop_code,f->stop_domain,f->stop_where);
  }
  requests_lost(port,port->parted?ECONNRESET:code);
  if(port->lanes.state!=LANES_DOWN)port->lanes.state=LANES_CLOSING;
  port->lanes.heard=0;
  /* design/bridge-dataflow.md#stripes: the stripe flow alike, closed once the stops before it have run */
  atomic_store_explicit(&port->stripes.up,0,memory_order_release);
  if(port->stripes.state!=STRIPES_DOWN)port->stripes.state=STRIPES_CLOSING;
  port->stripes.heard=0;port->stripes.wanted=0;
  /* design/bridge-dataflow.md#streams: a stream on the port fails, its requester answered */
  if(port->stream.state!=STREAMS_DOWN && port->stream.state!=STREAMS_CLOSING)streams_fail(port,port->parted?ECONNRESET:code,0);
  port->wanted|=port->greeted || retire;port->frees++;
  port->greeted=port->parted=port->connecting=port->writing=0;port->farewell=0;port->peer=0;port->held=0;port->out_length=port->out_sent=0;
  port->retiring|=retire;port->open&=!retire;
  link_set(port,0,retire?MESH_STOPPED:MESH_PAIRING,code);
}
/* design/bridge-and-ipc.md#ports: a control socket taken (dialed or accepted): non-blocking, its sends queued
   (port_say), its HELLO said; the link's loss is an event (its link off, the socket's end of file or reset, FAREWELL) */
static void port_greet(struct port *port){
  int on=1,flags=fcntl(port->control,F_GETFL);
  if(flags<0 || fcntl(port->control,F_SETFL,flags|O_NONBLOCK) || setsockopt(port->control,SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on)){
    port_lost(port,errno?errno:EIO,0);return;
  }
  port_watch(port,EVFILT_READ);
  port_say(port,(struct message){.kind=HELLO},NULL,0);
}
/* design/bridge-and-ipc.md#ports: the port a link-local peer reached, by its interface; none for a scope of 0 (no
   interface: a LAN address, or the index of an interface that is gone), nor for a port whose device is retiring, which
   its turn alone opens again (design/RDMA-KERNEL-RECOVERY.md) */
static struct port *port_at(uint32_t interface){
  for(uint32_t p=0;interface && p<port_count;p++)if(ports[p].interface==interface)return ports[p].retiring?NULL:ports+p;
  return NULL;
}
/* design/bridge-and-ipc.md#ports: a port's turn, where an event asked it (`wanted`: its finding, its link on or its
   interface attached, a paired link lost, the configuration's HELLO interval, a memory-pressure change after its
   registration was refused): on its own queue, its device retired where its port went away (once no flow and no lane
   holds queue pairs on it), else opened and its port read; then, while it has no peer, its hello sent to every node on
   its link */
static void port_turn(struct port *port){
  if(!port->wanted || port->job || port->control>=0 || (port->retiring && (port->flows || port->lanes.state!=LANES_DOWN ||
     port->stripes.state!=STRIPES_DOWN || port->stream.state!=STREAMS_DOWN)))return;
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
/* design/recovery.md#registration: a port's turn taken: a device whose registration of the window it refused keeps the
   extents it took and turns again at the next memory-pressure change or link event (device_up logged the extent and
   why); a device that failed otherwise, its link active, is retired and opened again at once, and where that fails too,
   at the next event that asks its turn */
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
  port->retiring=result && port->device.context && result!=ENETDOWN;
  port->wanted|=port->retiring && !retired;
  if(result){port->open=0;if(port->control<0)link_set(port,0,MESH_STOPPED,result);return;}
  atomic_store_explicit(&mesh_links(M)[port->index].max_pairs,port->device.max_pairs,memory_order_relaxed);
  port->open=1;port->lid=port->device.port.lid;memcpy(port->gid,&port->device.gid,16);
  port->bandwidth=port_bandwidth(&port->device.port);
  if(port->control>=0)return;
  link_set(port,0,MESH_PAIRING,0);
#ifdef MESH_LOOP_FABRIC
  /* design/bridge-dataflow.md#the-loopback-fabric: the cable's lower end dials the higher's listener on loopback, as a
     link's lower node dials the higher's on the address it heard; the higher accepts (main, TAG_LOOP) */
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
/* design/bridge-and-ipc.md#ports: a node heard on a port: the lower node dials the higher's listener on the address
   it heard from, the higher answers its hello so the lower hears it at once */
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
/* design/bridge-and-ipc.md#lanes: the lanes' pages of every port the region has, leased at the bridge's start beside the
   map's, so a port found later (an interface attached once clients hold the window) has them; a port whose pages the
   window could not give takes them when its lanes are made (lanes_turn) */
static uint32_t lanes_page[MESH_PORTS];
static void lanes_reserve(void){
  for(uint32_t p=0;p<MESH_PORTS;p++)lanes_page[p]=mesh_arena_alloc(&bridge,lane_pages(),1);
}
/* design/bridge-and-ipc.md#ports: every RDMA device of the node a port, in the order found, read on the finder's queue
   (at the start, and where an interface attached or its link came on that no port has) and added on the event loop */
static void ports_find(void){
  if(finding)return;
  finding=1;
  dispatch_async(finder,^{
    struct ibv_device **list=ibv_get_device_list(NULL);
    uint32_t n=0;
#ifdef MESH_LOOP_FABRIC
    /* design/bridge-dataflow.md#the-loopback-fabric: the configuration's devices alone, in its order */
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
    struct port *port=ports+port_count;
    *port=(struct port){.index=port_count,.control=-1,.wanted=1,.work=dispatch_queue_create("mesh.port",DISPATCH_QUEUE_SERIAL),.stream={.fd=-1}};
    /* design/bridge-and-ipc.md#lanes: the lanes' headers and discard blocks, the pages leased for the port's index */
    port->lanes.pages=lane_pages();
    port->lanes.page=lanes_page[port_count];
    port->lanes.provider.device=&port->device;
    port->stripes.provider.device=&port->device;
    port->stream.provider.device=&port->device;
    snprintf(port->device.name,sizeof port->device.name,"%s",devices[i]);snprintf(port->name,sizeof port->name,"%s",devices[i]+5);
    port_count++;
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

/* the queues (its highest queue with a list, plus one) and slots of a client's program toward its destination `d` */
static uint32_t program_queues(uint64_t owner,uint32_t d){
  uint32_t queues=0;
  for(uint32_t q=0;q<MESH_QUEUES;q++)for(int w=0;w<2;w++)if(atomic_load(mesh_order_length(M,owner,d*MESH_QUEUES+q,w)))queues=q+1;
  return queues;
}
/* whether a READY of execution `key` from `far` left the flow held here, on any port, for the client whose program
   takes it */
static int flow_held(const struct flow *f,uint64_t key,uint32_t far){
  return f->hold && f->key==key && f->near==M->node && f->far==far && f->state!=FAILED && !f->stopping;
}
/* design/bridge-and-ipc.md#cables: the cables a record holds paired with `peer`, and the stripes a routed program takes:
   the least common multiple of its hops' cables (a hop's the fewer of its two ends' records', the first hop's this
   node's own `first`), so each cable of every hop carries an equal share, or past MESH_STRIPES the fewest cables of a
   hop, so each stripe rides a cable of its own on every hop and the route carries its slowest hop's cables */
static uint32_t record_cables(const struct mesh_record *r,uint32_t peer){
  uint32_t n=0;
  for(uint32_t p=0;r && p<r->ports && p<MESH_PORTS;p++)
    n+=atomic_load_explicit(&r->port[p].phase,memory_order_relaxed)==MESH_PAIRED && atomic_load_explicit(&r->port[p].peer,memory_order_relaxed)==peer;
  return n;
}
static uint32_t route_stripes(const uint32_t *route,uint32_t hops,uint32_t first){
  uint32_t stripes=1,fewest=MESH_STRIPES;
  for(uint32_t i=0;i+1<hops;i++){
    uint32_t c=i?MIN(record_cables(record_held(route[i]),route[i+1]),record_cables(record_held(route[i+1]),route[i])):first;
    c=MIN(MAX(c,1u),MESH_STRIPES);
    uint32_t a=stripes,b=c;
    while(b){const uint32_t t=a%b;a=b;b=t;}
    stripes=stripes/a*c>MESH_STRIPES?MESH_STRIPES+1:stripes/a*c;
    fewest=MIN(fewest,c);
  }
  return stripes<=MESH_STRIPES?stripes:fewest;
}
/* whether this end's map routes to `node` over a cable it holds paired now */
static int route_held(uint32_t node){
  uint32_t *route=NULL;
  const uint32_t hops=mesh_route(M,M->node,node,&route);
  struct port *next[MESH_PORTS];
  const int held=hops>=2 && cables(route[1],next)>0;
  free(route);
  return held;
}
/* design/bridge-and-ipc.md#routes: the higher end of a routed pair awaits the lower's route, whatever its own map holds:
   the stripes the lower's READYs left held taken at once, the rest as they come (ready_heard); refused at once
   (EHOSTUNREACH, naming the lower) where its own map holds no route to the lower, as the lower refuses where its map
   holds none to the higher; else awaited until they come, the lower's failure as a value, or a revoke of the client's
   program (its slot retired).  Each take updates the await before the READY that may fail the program goes out */
static void program_await(struct mesh_client *client,uint64_t owner,uint32_t d,uint32_t node,uint32_t queues,uint32_t slots){
  struct await *a=calloc(1,sizeof *a);
  if(!a){status_failed(&client->status[d],owner,ENOMEM,4,M->node);return;}
  *a=(struct await){.next=awaits,.key=client->key,.owner=owner,.destination=d,.far=node};
  awaits=a;await_count++;busy_tell();
  for(struct flow *h=flows;h;h=h->next){
    if(!flow_held(h,client->key,node))continue;
    if(h->queues!=queues || (uint32_t)h->qps!=queues*slots || (a->stripes && h->stripes!=a->stripes) || (a->taken>>h->stripe&1)){
      flow_fail(h,EPROTO,4,FAIL,M->node,M->node);
      program_fail(owner,d,EPROTO,M->node);
      return;
    }
    flow_take(h,owner,d);
    const int last=await_took(a,h->stripe,h->stripes);
    ready_say(h,&h->far,1);flow_step(h);
    if(last || (a=await_of(owner,d))==NULL)return;
  }
  if(a->taken || route_held(node))return;
  say("program %016llx: node %u, the lower end of its route, is reached by no route of this node's map: refused\n",
      (unsigned long long)client->key,node);
  program_fail(owner,d,EHOSTUNREACH,node);
}
/* the programs awaiting node `node`'s READYs, failed as values where the map pruned it */
static void awaits_lost(uint32_t node){
  for(struct await *a=awaits,*next;a;a=next){
    next=a->next;
    if(a->far==node)program_fail(a->owner,a->destination,EHOSTUNREACH,node);
  }
}
/* design/bridge-and-ipc.md#clients, #cables, #routes: a started program's destination taken onto its ports.  Over the
   cables paired with its node, a flow a cable in the cable order both ends read alike, flow j carrying the program's
   invocations t with t mod n = j, each asking for queue pairs of its own (#lanes); else along a route, which the
   pair's lower node chooses on its map and both ends take: the lower's flows, one a stripe of its route's
   (route_stripes), each on the lanes of the cable of the route's first hop it rides (stripe_cable), each READY
   carrying the route; the higher's program awaits the lower's READYs (program_await), each READY of its going back
   along the relays that READY made.  Its flows' sides the client's, a READY on each, its receives then posted
   (flow_step).  Refused as a value naming the node where it is this node (EINVAL) or no route reaches it (EHOSTUNREACH,
   the lower end's), and naming this node where its table is full (E2BIG) or a flow of its key toward the node runs for
   another program (EBUSY) */
static void program_take(struct mesh_client *client,uint64_t owner,uint32_t d){
  const uint32_t node=client->tx[d].node,slots=client->tx[d].slots,queues=program_queues(owner,d);
  if(!node || node==M->node){status_failed(&client->status[d],owner,EINVAL,4,node);return;}
  struct port *cable[MESH_PORTS];
  const uint32_t n=cables(node,cable);
  atomic_store_explicit(&client->status[d].phase,MESH_PAIRING,memory_order_relaxed);
  /* #routes: the peer's routed READY held here while this end sees cables to it: the two ends' maps differ */
  for(struct flow *h=flows;n && h;h=h->next){
    uint32_t j=0;
    if(!flow_held(h,client->key,node))continue;
    while(j<n && h->port!=cable[j])j++;
    if(j<n)continue;
    say("link %u (%s): program %016llx: node %u's READY came routed while this end has cables to it (the two ends' maps "
        "differ): refused\n",h->port->index,h->port->name,(unsigned long long)client->key,node);
    flow_fail(h,EPROTO,4,FAIL,M->node,M->node);
    program_fail(owner,d,EPROTO,M->node);
    return;
  }
  for(uint32_t j=0;j<n;j++){
    struct flow *f=flow_on(cable[j],client->key,M->node,node,j);
    if(f && !f->hold){program_fail(owner,d,EBUSY,M->node);return;}
    if(f && (f->stripes!=n || f->queues!=queues || (uint32_t)f->qps!=queues*slots)){flow_fail(f,EPROTO,4,FAIL,M->node,M->node);f=NULL;}
    if(!f && !(f=flow_make(cable[j],client->key,M->node,node,queues,slots,j,n,1))){program_fail(owner,d,E2BIG,M->node);return;}
    flow_take(f,owner,d);
    ready_say(f,&node,1);
    flow_step(f);
    if(f->state==FAILED)return;
  }
  if(n)return;
  if(M->node>node){program_await(client,owner,d,node,queues,slots);return;}
  uint32_t *route=NULL;
  const uint32_t hops=mesh_route(M,M->node,node,&route);
  struct port *next[MESH_PORTS];
  const uint32_t m=hops>=2?cables(route[1],next):0;
  if(!m){free(route);status_failed(&client->status[d],owner,EHOSTUNREACH,4,node);return;}
  const uint32_t stripes=route_stripes(route,hops,m);
  for(uint32_t j=0;j<stripes;j++){
    struct flow *f=flow_make(stripe_cable(next,m,client->key,M->node,node,j),client->key,M->node,node,queues,slots,j,stripes,0);
    if(!f){free(route);program_fail(owner,d,E2BIG,M->node);return;}
    flow_take(f,owner,d);
    ready_say(f,route+1,hops-1);
    flow_step(f);
    if(f->state==FAILED)break;
  }
  free(route);
}
/* design/bridge-and-ipc.md#clients: every client slot read: a slot's process watched from the turn its slot is first
   seen (its exit retires every slot it holds, the event loop's EVFILT_PROC), one already gone retired at once; a
   program that ended (stopped, its client gone) leaves its ports; a program started is taken toward each destination
   it binds, once no flow of the slot's earlier client remains, or, where a READY of its execution held here names
   another digest than its own, failed with that READY's flow, both as values (ESTALE: program_stale).  Rows of a
   retired client or lease are released once no flow of one remains */
static void clients_turn(void){
  const pid_t self=getpid();
  int changed=0,zombie=0;
  for(struct flow *f=flows;f;f=f->next)
    if(f->owner && (pid_t)(uint32_t)f->owner!=self && (!owned(f->owner) || atomic_load(&mesh_client_at(M,f->owner)->configured)!=f->owner))
      flow_end(f);
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct mesh_client *client=(struct mesh_client *)((char *)M+M->client_off)+s;
    uint64_t owner=atomic_load(&client->owner);
    if(owner && (pid_t)(uint32_t)owner==self)continue;
    /* #control: a slot retired in order, seen at the turn before (so its flows have read it, flow_end), is vacant */
    if((owner&MESH_ORDERLY) && owner==owners[s] && atomic_compare_exchange_strong(&client->owner,&owner,0))owner=0;
    if(owner && owner!=owners[s] && !(owner&MESH_RETIRING)){
      struct kevent64_s watch;EV_SET64(&watch,(uint32_t)owner,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,s,0,0);
      if(kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)<0 && errno==ESRCH){mesh_retire(M,owner);owner=atomic_load(&client->owner);}
    }
    changed|=owner!=owners[s];owners[s]=owner;
    if(!owner || (owner&MESH_RETIRING) || atomic_load_explicit(&client->configured,memory_order_acquire)!=owner || accepted[s]==owner)continue;
    int earlier=0;
    for(struct flow *f=flows;f;f=f->next)earlier|=f->owner && f->owner!=owner && ((f->owner>>32)&(MESH_CLIENTS-1))==s && (pid_t)(uint32_t)f->owner!=self;
    if(earlier)continue;
    accepted[s]=owner;
    atomic_store_explicit(&client->ports,0,memory_order_relaxed);
    uint32_t stale=0;
    for(struct flow *h=flows;h;h=h->next)
      if(h->hold && h->key==client->key && h->near==M->node && h->state!=FAILED && !h->stopping &&
         digest_apart(atomic_load_explicit(&client->digest,memory_order_relaxed),h->digest)){
        stale=h->far;flow_fail(h,ESTALE,4,FAIL,h->far,M->node);
      }
    if(stale){program_stale(owner,stale);continue;}
    for(uint32_t d=0,n=mesh_destination(client,0);d<n;d++)if(mesh_destination_transfers(M,owner,d))program_take(client,owner,d);
  }
  for(struct flow *f=flows;f;f=f->next)
    zombie|=(f->owner && !owned(f->owner)) || (f->side[0].owner && !owned(f->side[0].owner)) || (f->side[1].owner && !owned(f->side[1].owner));
  /* design/bridge-dataflow.md#stripes: a stripe in flight may name a retired client's bytes and word */
  for(uint32_t p=0;p<port_count;p++)zombie|=atomic_load_explicit(&ports[p].stripes.flight,memory_order_acquire)!=0;
  unreleased|=changed;
  if(unreleased && !zombie){unreleased=0;mesh_retired_release(M);}
}
/* design/bridge-and-ipc.md#control: a running flow whose program ended in order and drained (every cell it released
   posted and those SENDs completed: flow_send, lanes_drained) released, its peer told nothing but a lane rider's
   RELEASED (for the relays of its route), its program's words kept */
static void flow_released(struct flow *f){
  f->state=FAILED;
  if(!f->exclusive)port_tell(f->port,f,CONTROL_RELEASED,0);
  say("link %u: program %016llx toward node %u (stripe %u of %u) ended in order\n",f->port->index,(unsigned long long)f->key,f->far,
      f->stripe,f->stripes);
  flow_stop(f,0,0,0);
}
/* design/bridge-and-ipc.md#routes: a relay's flow whose end said RELEASED: passed on toward the route's other end by its
   partner, and the two flows released once each end's came (both ends' programs had every cell they waited on, and
   sent every one they released) */
static void relay_released(struct flow *f){
  struct flow *g=f->partner;
  if(!f->passed){
    f->passed=1;
    if(g && g->state!=FAILED)port_tell(g->port,g,CONTROL_RELEASED,0);
  }
  if(!g || (g->passed && f->passed)){
    say("link %u: relay of program %016llx between nodes %u and %u (stripe %u of %u) released by both its ends\n",f->port->index,
        (unsigned long long)f->key,f->far,g?g->far:0,f->stripe,f->stripes);
    f->state=FAILED;flow_stop(f,0,0,0);
    if(g && g->state!=FAILED){g->state=FAILED;flow_stop(g,0,0,0);}
  }
}
/* design/bridge-and-ipc.md#flows: every flow's turn: its job's end taken, a stopped flow freed (its release refused, it
   tries again once another flow of its port is freed or its link is lost; its records still posted, at their last
   landing), a failed one telling FAIL stopped once that is sent or its peer's own came, a data thread's failure, the
   peer's FAIL heard (#control: its failure, not told back), a program ended in order released once it drained, a
   relay's end's RELEASED heard taken */
static void flows_turn(void){
  for(struct flow *f=flows,*next;f;f=next){
    next=f->next;
    if(f->job && atomic_load_explicit(&f->done,memory_order_acquire))flow_done(f);
    if(f->stopping){
      const int stopped=atomic_load_explicit(&f->stopped,memory_order_acquire);
      if(stopped==1 && !f->job)flow_free(f);
      else if(stopped==2 || (stopped==3 && f->frees!=f->port->frees))flow_stop(f,0,0,0);
      continue;
    }
    if(f->state==FAILED){
      if((atomic_load_explicit(&f->ending,memory_order_acquire)&FLOW_TELLING) &&
         (atomic_load_explicit(&f->said,memory_order_acquire) || atomic_load_explicit(&f->heard_fail,memory_order_acquire)))
        flow_stop(f,f->stop_code,f->stop_domain,f->stop_where);
      continue;
    }
    const int64_t failed=atomic_load(&f->failed),heard=atomic_load(&f->heard_fail);
    if(heard){
      const uint32_t origin=atomic_load(&f->heard_origin);
      const uint64_t owner=f->partner?0:f->owner;
      flow_fail(f,heard==ESTALE || heard==ECONNRESET?heard:ECANCELED,4,0,origin,origin);
      if(heard==ESTALE && owner)program_stale(owner,origin);
    }
    else if(failed)flow_fail(f,failed,atomic_load(&f->domain),FAIL,M->node,M->node);
    else if(f->state==RUNNING && (atomic_load_explicit(&f->ending,memory_order_acquire)&FLOW_ENDING) &&
            atomic_load_explicit(&f->drained,memory_order_acquire))flow_released(f);
    else if(f->partner && atomic_load_explicit(&f->released,memory_order_acquire) && !f->passed)relay_released(f);
  }
  /* #routes: a program awaiting its lower peer's READYs: dropped with its program */
  for(struct await *a=awaits,*next;a;a=next){
    next=a->next;
    if(!owned(a->owner) || atomic_load(&mesh_client_at(M,a->owner)->configured)!=a->owner)await_drop(a);
  }
}

/* design/prepared-machine.md#M26: the clients' control word, watched for the event loop */
static void *control_watch(void *unused){
  (void)unused;
  while(!stop){
    uint64_t seen=atomic_load_explicit(&M->control,memory_order_acquire);
    wake();
    os_sync_wait_on_address(&M->control,seen,sizeof M->control,OS_SYNC_WAIT_ON_ADDRESS_SHARED);
  }
  return NULL;
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

/* design/bridge-and-ipc.md: the node's resident bridge; its one operand the path of its configuration (no default) */
int main(int argc,char **argv){
  if(argc!=2)die("usage: mesh-flow CONFIGURATION: the canonical property list bin/mesh-user.sh writes from the node's bridge configuration document (etc/bridge/README.md); the bridge has no default and does not start without one");
  char name[256];uint64_t window;
  const char *refused=configuration_read(argv[1],name,sizeof name,&window);
  if(refused){say("bridge configuration %s refused: %s\n",argv[1],refused);exit(1);}
  /* design/bridge-and-ipc.md: the bridge's binary watched (EVFILT_VNODE), so its change is an event (main's loop) */
  const int binary=open(argv[0],O_EVTONLY);
  uuid_t uuid;const struct timespec forever={0,0};
  if(gethostuuid(uuid,&forever)){say("host identity: %s\n",strerror(errno));exit(1);}
  uint32_t node=0;
  for(int i=0;i<16;i++)node=(node*0x01000193u)^uuid[i];
  node=(node&0x7fffffffu)?node&0x7fffffffu:1;
  /* design/bridge-dataflow.md#the-loopback-fabric: a loop bridge's node, region and service its configuration's */
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
  uint64_t length=mesh_layout(&geometry,pg,1,(uint32_t)(window/pg));
  signal(SIGPIPE,SIG_IGN);signal(SIGINT,SIG_IGN);signal(SIGTERM,SIG_IGN);signal(SIGHUP,SIG_IGN);
  atexit(down);
  shm_unlink(region);int fd=shm_open(region,O_CREAT|O_RDWR,MESH_MODE);if(fd<0)die("shm");
  if(ftruncate(fd,(off_t)length))die("ftruncate");fchmod(fd,MESH_MODE);
  M=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  if(M==MAP_FAILED)die("mmap");shm=region;
  *M=geometry;M->node=node;M->version=MESH_VERSION;
  for(uint32_t r=0;r<mesh_rows(M);r++)atomic_store_explicit(&mesh_page(M)[r].mapping,MESH_ABSENT,memory_order_relaxed);
  if(wire_map(&wire,M,fd))die("transport page aliases");
  close(fd);
  bridge.M=M;
  host_name(host);
#ifdef MESH_LOOP_FABRIC
  snprintf(host,sizeof host,"%s",name);
#endif
  finder=dispatch_queue_create("mesh.ports",DISPATCH_QUEUE_SERIAL);
  /* design/bridge-and-ipc.md#records: the node's memory and hardware model, which its record carries with its window */
  size_t size=sizeof memory;
  if(sysctlbyname("hw.memsize",&memory,&size,NULL,0))memory=0;
  sysctl_text("hw.model",model,sizeof model);
  events=kqueue();
  health_init();
  record_changed();
  if(!atomic_load(&M->map_nodes))die("the map's lease");
  lanes_reserve();
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
  /* the events the loop takes beside its sockets: the binary's change (its contents written, or the file replaced,
     renamed or removed; an attribute's change, an extended attribute the system adds, is none), the power sources' and
     thermal pressure's
     notifications, and the configuration's paced emissions (HELLO_TIMER, HEALTH_TIMER) */
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
  /* design/bridge-dataflow.md#the-loopback-fabric: each cable's listener on loopback, its events tagged by the port */
  for(uint32_t k=0;k<loop_config.ports;k++){
    struct sockaddr_in6 at={.sin6_len=sizeof at,.sin6_family=AF_INET6,.sin6_port=htons(loop_config.port[k].listen),.sin6_addr=IN6ADDR_LOOPBACK_INIT};
    int fd=socket(AF_INET6,SOCK_STREAM,0);
    if(fd<0 || setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on) || bind(fd,(struct sockaddr *)&at,sizeof at) || listen(fd,SOMAXCONN) ||
       fcntl(fd,F_SETFL,O_NONBLOCK))die("a loop cable's listener");
    loop_config.port[k].fd=fd;
    EV_SET64(&event,(uint64_t)fd,EVFILT_READ,EV_ADD,0,0,TAG_LOOP|k,0,0);
    if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");
  }
#endif
  ports_find();
  host_watch();
  /* design/bridge-and-ipc.md#region: the bridge's start told (mesh-plan.h MESH_STARTED, mesh_bridge_wait) */
  char started[96];
#ifdef MESH_LOOP_FABRIC
  snprintf(started,sizeof started,"%s%s",MESH_STARTED,region);
#else
  snprintf(started,sizeof started,"%s",MESH_STARTED);
#endif
  notify_post(started);
  char alive[256]="";
  snprintf(alive,sizeof alive,"; keepalive %llu ns, HELLO %llu ns, health %llu ns (0: none; design/bridge-and-ipc.md#keepalive)",
           (unsigned long long)keepalive_ns,(unsigned long long)hello_ns,(unsigned long long)health_ns);
  say("bridge node %u (%s): configuration %s, window %llu bytes, arena %llu pages, rows %u%s%s\n",node,host,name,
    (unsigned long long)window,(unsigned long long)mesh_arena_pages(M),M->rows,ledger_kept?", ledgers kept (design/bridge-and-ipc.md#ledger)":"",
    alive);
  int rebuilt=0,flood=0;
  while(!stop){
    if(atomic_load_explicit(&found,memory_order_acquire))ports_found();
    const int moved=atomic_exchange_explicit(&pressured,0,memory_order_acquire);
    for(uint32_t p=0;p<port_count;p++){
      struct port *port=ports+p;
      if(port->job && atomic_load_explicit(&port->done,memory_order_acquire))port_opened(port);
      if(moved && port->refused){port->refused=0;port->wanted=1;}
      port_turn(port);
      lanes_turn(port);
      stripes_turn(port);
      streams_turn(port);
    }
    flows_turn();
    stripes_requests_turn();
    health_turn(flood);
    flood=0;
    host_taken();
    clients_turn();
    busy_tell();
    /* design/bridge-and-ipc.md: a bridge whose binary changed stops between calls, once it carries no flow and no
       awaiting program (`busy` 0) */
    if(rebuilt && !atomic_load_explicit(&M->busy,memory_order_acquire)){stop=1;stop_reason=MESH_STOP_BINARY;break;}
    struct kevent64_s ready[16];
    int count=kevent64(events,NULL,0,ready,16,0,NULL);
    for(int i=0;i<count;i++){
      struct kevent64_s *e=ready+i;
      if(e->filter==EVFILT_SIGNAL){stop=1;stop_reason=(int64_t)e->ident;}
      else if(e->filter==EVFILT_VNODE){
        if(!rebuilt){
          rebuilt=1;
          atomic_store_explicit(&M->stopping,1,memory_order_release);
          os_sync_wake_by_address_all(&M->stopping,sizeof M->stopping,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
          mesh_notify(&M->control);
          say("bridge node %u: its binary changed; it stops once no program runs through it\n",node);
        }
      }
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
          uint64_t owner=atomic_load(&((struct mesh_client *)((char *)M+M->client_off))[s].owner);
          if(owner && (uint32_t)owner==(uint32_t)e->ident)mesh_retire(M,owner);
        }
      }
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)listener){
        struct sockaddr_in6 from;socklen_t size=sizeof from;int accepted;
        while((accepted=accept(listener,(struct sockaddr *)&from,&size))>=0){
          /* #requests: a connection from this node is a client's request */
          if(IN6_IS_ADDR_LOOPBACK(&from.sin6_addr)){size=sizeof from;asking_accepted(accepted);continue;}
          struct port *port=port_at(from.sin6_scope_id);size=sizeof from;
          if(!port || port->control>=0 || !port->open){close(accepted);continue;}
          port->control=accepted;
          port_greet(port);
        }
      }
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)hello)hello_heard();
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)network)network_heard();
#ifdef MESH_LOOP_FABRIC
      /* design/bridge-dataflow.md#the-loopback-fabric: a cable's lower end dialed: its port's control socket, as a link's
         listener takes the lower node's dial */
      else if(e->udata!=UINT64_MAX && (e->udata&TAG_LOOP)){
        const uint32_t k=(uint32_t)(e->udata&0xff);
        int accepted;
        while(k<loop_config.ports && (accepted=accept(loop_config.port[k].fd,NULL,NULL))>=0){
          struct port *port=NULL;
          for(uint32_t p=0;p<port_count;p++)if(!strcmp(ports[p].device.name,loop_config.port[k].device))port=ports+p;
          if(!port || port->control>=0 || !port->open){close(accepted);continue;}
          port->control=accepted;port->peer=0;
          port_greet(port);
        }
      }
#endif
      else if(e->udata!=UINT64_MAX && (e->udata&(TAG_CHANNEL|TAG_ASKING|TAG_SERVING)))request_event(e);
      else if(e->udata<port_count && (int)e->ident==ports[e->udata].control){
        struct port *port=ports+e->udata;
        if(e->filter==EVFILT_WRITE && port->connecting){
          int error=0;socklen_t size=sizeof error;
          if(getsockopt(port->control,SOL_SOCKET,SO_ERROR,&error,&size))error=errno;
          if(error)port_lost(port,error,0);
          else {port->connecting=0;port_greet(port);}
        }
        else if(e->filter==EVFILT_WRITE){port->writing=0;port_flush(port);}
        else control_heard(port);
      }
    }
  }
  char why[96];farewell_text(stop_reason,why,sizeof why);
  say("bridge node %u: stopping (%s)\n",node,why);
  requests_stop();
  for(uint32_t p=0;p<port_count;p++){
    port_say(ports+p,(struct message){.kind=FAREWELL,.code=stop_reason},NULL,0);
    ports[p].parted=1;
    port_lost(ports+p,ESHUTDOWN,1);
  }
  for(uint32_t p=0;p<port_count;p++){
    struct port *port=ports+p;
    dispatch_sync(port->work,^{
      for(struct flow *f=flows;f;f=f->next)if(f->port==port)flow_release(f);
      atomic_store_explicit(&port->lanes.up,0,memory_order_release);
      port_settle(port);
    });
    port_still(port);
    dispatch_sync(port->work,^{
      /* design/bridge-dataflow.md#stripes: no progress thread runs: the stripe flow's queue pairs flushed and destroyed */
      if(port->stripes.provider.queues){
        pair_flush(&port->stripes.provider);
        if(!down_pair(&port->stripes.provider))say("link %u (%s): stripes teardown: %s\n",port->index,port->name,strerror(errno));
      }
      if(port->stream.provider.queues){
        pair_flush(&port->stream.provider);
        if(!down_pair(&port->stream.provider))say("link %u (%s): stream teardown: %s\n",port->index,port->name,strerror(errno));
      }
      if(port->stream.fd>=0){fd_answer(port->stream.fd,ESHUTDOWN);close(port->stream.fd);port->stream.fd=-1;}
      if(!down_pair(&port->lanes.provider))say("link %u (%s): lanes teardown: %s\n",port->index,port->name,strerror(errno));
      if(!down_device(&port->device))say("link %u (%s): verbs teardown: %s\n",port->index,port->name,strerror(errno));
    });
  }
  atomic_store(&M->bridge_pid,0);
  mesh_notify(&M->control);
  pthread_join(watcher,NULL);
  munmap(wire.data,wire.length);munmap(M,length);
  return 0;
}
