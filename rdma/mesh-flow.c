#include "mesh-verbs.h"
#include "mesh-dataflow.h"
#include <pthread.h>
#include <sysexits.h>
#include <sys/event.h>
#include <sys/ioctl.h>
#include <sys/kern_event.h>
#include <net/if.h>
#include <net/if_var.h>
#include <net/net_kev.h>

/* design/prepared-machine.md#M24 */
union mesh_network_event {
  struct kern_event_msg header;
  unsigned char bytes[KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data)];
};
_Static_assert(sizeof(union mesh_network_event)==48,"M24");
/* set by SIGQUIT (crash_bridge, below): this bridge ends as a crash would */
static _Atomic int crashing;
/* SIGINT, SIGTERM, SIGHUP (and SIGUSR2's leave, SIGQUIT's crash): the bridge stops, its main thread woken from its
   wait for the stop (`stopping`, a word of this process) */
static _Atomic uint64_t stopping;
static void stop_bridge(int signal){
  int error=errno;
  onsig(signal);
  atomic_fetch_add_explicit(&stopping,1,memory_order_release);
  os_sync_wake_by_address_all(&stopping,sizeof stopping,OS_SYNC_WAKE_BY_ADDRESS_NONE);
  errno=error;
}

/* The communicator service: NCCL's network plugin model (ncclNet_v12_t, mesh-net.h) served over each
   link, for the bridge's lifetime and for any number of clients and communicators.  One session per
   link pairs on the link's communicator port (its service port + NET_PORT_OFFSET unless given) with
   NET_QUEUES UC queue pairs.  Control is a stream of fixed messages on the session's socket: connect, accept,
   ready-to-send (RTS), credit and close.  Data is SEND into a posted RECV only (TB5 RDMA has no
   one-sided write), and the receiver drives it: a sender's isend announces its message (RTS: its size
   and where its registration regions fall); the receiver matches announcements with its irecvs in
   order, cuts the message every chunk_frames frames and wherever either end's registration region
   ends, and for each chunk posts its RECV first, on the queue pair of the chunk before it while that
   queue has the frames, else on one that has, and only then grants it (CREDIT: the chunk's offset,
   length and queue pair); the sender SENDs exactly the granted chunks in the order granted, each on its
   queue pair, and a chunk on another queue pair than the SENDs still outstanding only once they have
   completed, so the wire carries the chunks in grant order.  The queue pairs' RECVs together hold more
   than one queue's frames (4095), so a message of more (tp.py's 16 MiB all-reduce: 4096) is posted and
   granted whole before its first byte.  A held isend (mesh.h
   MESH_NET_HELD) is announced before its bytes are ready, so its chunks are posted and granted ahead and
   its first byte waits for no RTS and CREDIT once its client releases it; its grants, and every grant after
   them, wait on the sender until then.  So a SEND never meets a
   queue without its RECV, and each queue pair's RECVs meet its SENDs one to one whatever communicators
   they belong to.  Each session's queue pairs are primed: one RECV posted on each between RTR and RTS, filled
   by the peer's first SEND on it.  A receive lands only in memory registered before its queue pair was set up
   (a receive into a registration made after RTR completes with a local protection error, one made
   before the queue pair with success), so communicator memory is the region's registered window, used
   in place, and the discard buffer is the device's (mesh-verbs.h device_up).
   A send request ends when its receiver says every byte of it landed (LANDED), not when its SENDs complete:
   on a UC queue pair a SEND's completion says only that the bytes left.  A session lost while the bridge
   runs (the kernel says the link's interface went off, a completion failed, the control socket ended, or
   SIGUSR1 asked) is suspended, not failed: its comms and requests stay as they are, each receive goes back
   to the bytes of it landed in order, and the next pairing, with the same peer bridge (its instance, drawn
   at its start), resumes it (net_resume): each receive asks its sender to go on from there and grants the
   rest again, each isend is announced again, connects and closes are sent again.  So a lost link makes the
   calls on it late, never failed or different; a peer that pairs as another instance fails them, as its side is
   gone.
     The bridge's side of the comms and of each session's transfers, and this node's bridge instance, live in shared
   memory named after the region (<region>.keep: net_keep), as the comm tables live in the region, not in the
   bridge's process.  A bridge stopped (SIGTERM) with clients attached leaves the region, its keep and its link
   table as they are, its sessions suspended, and exits EX_TEMPFAIL, so its supervisor starts the next (launchd:
   bin/mesh-bridge.sh; a bridge stopped with no client attached exits 0, and none is started); the next bridge started on them with the same configuration, while a client of them is
   alive, takes them as they are, registers the window before its queue pairs, pairs as the same instance and
   resumes, so a restart is to its clients and its peers a lost session: late, never failed.  Kept state whose
   clients' processes are all gone is released, by the next bridge's start (made afresh) or by --release
   (region_release), never by a clock.  The keep is written in place as the session runs (its receives, the RECVs
   posted for them as they land, each comm's side in an order a successor can complete: net_normalize), so a bridge
   that ends without its stop (a crash; SIGQUIT ends one so, for a check) leaves its successor what a stop would.  A
   peer that pairs as another instance (its machine restarted, or its region made afresh: --renew) fails the lost
   session's transfers and releases their state; each pairing's identity (mesh.h mesh_net_link.pairing) and the peer
   node's client processes (peer_clients) are its clients' membership: a peer client absent from its node's report has
   exited.  A bridge that leaves the mesh (SIGUSR2) tells its peers (LEAVE): a membership change, which each end
   marks on the link (mesh.h MESH_LEFT) before the transfers on it end failed, for the clients to plan on the nodes
   that stay; it exits 0, so no successor is started, and keeps the region for its clients where any is attached:
   a bridge started on the node again takes it, pairs, and the clients are members again. */
#define NET_PORT_OFFSET 1000
/* a session's queue pairs; every link's of a device are within the device's (mesh-verbs.h MESH_DEVICE_QPS) */
#define NET_QUEUES 2
#define NET_CHUNK_FRAMES (MESH_DISCARD/4096)
#define NET_PRIME 64
#define NET_SENDS 8192
#define NET_RECEIVES 8192
#define NET_OUTPUT 32768
#define NET_NONE UINT32_MAX
#define NET_DISCARD (UINT32_MAX-1)
#define NET_HEARTBEAT_NS UINT64_C(200000000)
enum { NET_CONNECT=1, NET_ACCEPT, NET_RTS, NET_CREDIT, NET_CLOSE, NET_HEARTBEAT, NET_LINKS, NET_BETA, NET_RESUME, NET_LANDED, NET_LEAVE,
       NET_CLIENTS, NET_VOID, NET_VOIDED };
/* One control message.  CONNECT: `key` the listen it names, `from` the connecting comm.  ACCEPT: `from`
   the receiving comm made for `to`.  RTS: send request `sequence` of `from`, `size` its bytes,
   extent/phase where its registration regions end, flags 1 a held isend (mesh.h MESH_NET_HELD: announced
   before its bytes are written, its granted chunks SENT once its client releases it), flags 2 announced again
   as a session resumes (below).  CREDIT: a chunk of
   that request, `offset` into it and `size` long, whose RECV is posted on queue pair `flags` (size 0: the
   empty message received; `error`: refused).
   CLOSE: `from` announces and grants nothing more on this connection; flags 1 asks for a CLOSE back.
   HEARTBEAT: nothing; each end sends one whenever it has sent nothing for NET_HEARTBEAT_NS: the peer's liveness,
   for whoever supervises the bridges (a peer stopped is silent, and late: it ends nothing; its session ends
   only on what is observed, the control socket's end or error, which TCP's keepalive gives a host that is gone,
   a failed completion, or the link's interface off, and resumes: below).  LINKS: node `from`'s report, sequence `sequence`, its links' up a bit a node in
   key, size, extent, phase, offset (NET_LINK_NODES: a table of at most 320 nodes); each bridge sends its own node's whenever its sessions
   pair or end, every newer one it takes on to its other peers, and every one it holds to a peer when
   their session pairs (mesh-dataflow.h: a link no report names is down).  BETA: node `from`'s estimate,
   sequence `sequence`, of link `key` -> `from`'s beta, the float's bits in `size` (-E, below), passed
   on as reports are.  LANDED: every byte of send request `sequence` of `to` has landed (the receiver's
   completions): the sender's request ends on it, not on its SENDs' completions, which on a UC queue pair say
   only that the bytes left.  RESUME, as a lost session's transfers resume (net_resume): flags 0, the receiver
   asks whether request `sequence` of `to` goes on from `offset`, the bytes of it landed in order; flags 1, the
   sender's answer, `error` if it does not (the receiver's request then fails), else the receiver grants the
   rest again.  LEAVE: the sender's bridge leaves the mesh; the session ends and nothing of it resumes.  CONNECT and
   ACCEPT carry their comm's client in `offset` (the comm's peer_owner at the other end).  CLIENTS: slot `flags` of
   the sender's node's client table holds `key` (0: none); the sender sends its whole table once a pairing begins and
   whenever it changes, before any CONNECT or ACCEPT of a client it names, and the last slot completes it (mesh.h
   peer_clients).  VOID: held isend `sequence` of `from` was never released, its comm closed first: its chunks go from
   the discard buffer, and only once its receiver has answered VOIDED (to `from`, the same sequence), so a receive whose
   bytes land before any VOID holds the bytes its sender released, and ends on their landing (the bytes reach the NIC
   only once released: no message need say so), and one that heard VOID first fails as they land. */
struct net_message { uint32_t kind,from,from_generation,to,to_generation,flags; int32_t tag,error; uint64_t key,sequence,size,extent,phase,offset; };
_Static_assert(sizeof(struct net_message)==80,"net_message");
/* A comm's bridge side, its link's session thread's alone: requests taken; a receive comm's requests
   matched with announcements and the announcements waiting in arrival order; each request's phase (a
   send: 1 announced, 2 granted; a receive: 1 waiting, 2 matched) and a send's bytes sent (and granted before
   its session was lost: again); what is on the wire (`outstanding`), the close handshake, and a receive
   comm's `seen`, past the last sequence announced to it. */
struct net_announce { uint64_t sequence,size,extent,phase; uint32_t held; };
/* An announcement consumed (by sequence, a slot each): how it ended where its LANDED, its empty CREDIT or its
   refusal can be lost with a session (`ended` 1 matched to a receive, 2 empty, 3 refused with `error`). */
struct net_outcome { uint64_t sequence; uint32_t ended; int32_t error; };
struct net_comm {
  uint32_t generation,outstanding,live,connect_sent,close_sent,peer_closed;
  uint64_t taken,matched,announce_head,announce_tail,seen;
  struct net_announce announced[MESH_NET_REQUESTS];
  struct net_outcome outcome[MESH_NET_REQUESTS];
  uint64_t sent[MESH_NET_REQUESTS],again[MESH_NET_REQUESTS];
  uint8_t phase[MESH_NET_REQUESTS],released[MESH_NET_REQUESTS];
};
/* A granted chunk to SEND on its queue pair (comm NONE: a grant nobody holds, filled from the discard
   buffer), a matched message to receive (the ends' cut parameters e1/p1, e2/p2; the sender's comm and
   request to grant; resume, waiting for its sender's RESUME answer; again, what had been posted of it before
   its session was lost), and a posted RECV (its queue pair; landed, its completion taken). */
struct net_send { uint32_t comm,generation,slot,mr,queue,voided; uint64_t sequence,offset,length; };
/* `held`: a held isend's message, which ends once landed (`released` 3), failed where its VOID came first (2) */
struct net_transfer { uint32_t comm,generation,slot,mr,peer,peer_generation,held,resume,released,padding; uint64_t sequence,size,offset,cursor,landed,e1,p1,e2,p2,again; };
/* A posted RECV: its transfer (NET_NONE: the discard buffer), frames, queue pair, whether it landed, its bytes and
   where in its transfer they go (`at`). */
struct net_chunk { uint32_t transfer,frames,queue,landed; uint64_t length,at; };
#define NET_CHUNKS 8192
/* What a link's session keeps past its bridge's process, written in place as the session runs, so that a successor
   finds it current however the last bridge ended (a crash included): the peer bridge's instance its last pairing
   exchanged (`paired`), whether that session was lost with its transfers kept for the next pairing with it
   (`suspended`), its matched messages to receive, receive_head..receive_tail (receive_post the next to post), and the
   RECVs posted for them and not yet retired, chunk_head..chunk_tail, each marked as it lands: a message's bytes landed
   in order are the least start of its chunks not landed (net_rewind). */
struct net_kept { uint64_t paired; uint32_t suspended,receive_head,receive_post,receive_tail,chunk_head,chunk_tail;
  struct net_transfer receives[NET_RECEIVES]; struct net_chunk chunks[NET_CHUNKS]; };
/* The bridge's state of the communicator service beside the region (<region>.keep), which outlives the bridge's
   process as the region does: this node's bridge instance (drawn when the keep is made: a peer pairing again with
   the instance it lost resumes the lost session's transfers, another instance's pairing fails them), each comm's
   bridge side and each link's session's kept state.  A bridge takes the keep its last one left where the region
   is this configuration's (the same geometry, node and links), else makes both afresh. */
#define NET_KEEP_MAGIC 0x4b454550u
#define NET_KEEP_VERSION 2u
/* `renew` (mesh-flow --renew): the next bridge on the region makes region, keep and link table afresh, as after the
   node restarted (a new instance; the clients attached to these are left on them). */
struct net_keep { uint32_t magic,version,links,node; uint64_t region,instance; _Atomic uint32_t renew,padding; struct net_comm comms[MESH_NET_COMMS]; struct net_kept kept[]; };
static struct net_keep *net_keep;
static struct net_comm *net_comms;
static char net_keep_name[80];
static size_t net_keep_bytes(uint32_t links){return sizeof(struct net_keep)+(size_t)links*sizeof(struct net_kept);}
/* The region's keep, made afresh (`fresh`: zeroed, a new instance), or the one its last bridge left where it is
   this region's; NULL where there is none. */
static struct net_keep *net_keep_open(const char *region,uint32_t links,uint32_t node,uint64_t length,int fresh){
  snprintf(net_keep_name,sizeof net_keep_name,"%s.keep",region);
  const size_t bytes=net_keep_bytes(links);
  if(fresh)shm_unlink(net_keep_name);
  int file=shm_open(net_keep_name,fresh?O_CREAT|O_EXCL|O_RDWR:O_RDWR,0600);
  if(file<0)return NULL;
  struct stat info;
  int ok=fresh?!ftruncate(file,(off_t)bytes):!fstat(file,&info) && (size_t)info.st_size>=bytes;
  struct net_keep *k=ok?mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,file,0):MAP_FAILED;
  close(file);
  if(k==MAP_FAILED)return NULL;
  if(fresh){
    k->version=NET_KEEP_VERSION;k->links=links;k->node=node;k->region=length;arc4random_buf(&k->instance,sizeof k->instance);
    atomic_thread_fence(memory_order_release);k->magic=NET_KEEP_MAGIC;
  } else if(k->magic!=NET_KEEP_MAGIC || k->version!=NET_KEEP_VERSION || k->links!=links || k->node!=node || k->region!=length){
    munmap(k,bytes);return NULL;
  }
  return k;
}
/* The identity of a pairing of bridge instances a and b, in either order (mesh.h mesh_net_link.pairing; never 0). */
static uint64_t net_mix(uint64_t x){
  x+=UINT64_C(0x9e3779b97f4a7c15);x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);return x^(x>>31);
}
static uint64_t net_pairing(uint64_t a,uint64_t b){const uint64_t p=net_mix(a<b?a:b)^net_mix((a<b?b:a)+1);return p?p:1;}
/* The SENDs outstanding are all on `send_queue` (send_posted less send_retired: its frames, within one queue
   pair's send_capacity); the RECVs posted and not yet taken are the keep's chunk_head..chunk_tail in posting order
   (at most receive_requests, the completion queue's entries less one, below NET_CHUNKS), each queue pair's frames
   of them within receive_capacity, the next posted on receive_queue while it has the frames. */
struct net_session {
  struct hdr *M;uint32_t index;struct mesh_verbs provider;struct mesh_net_link *counts;
  char service[16];pthread_t thread;int started,control,failed;
  uint64_t chunk,send_posted,send_retired;uint32_t send_capacity,receive_capacity,receive_outstanding;
  uint32_t send_queue,receive_queue,receive_requests,queue_frames[NET_QUEUES];
  struct net_send *sends;uint32_t send_head,send_post,send_tail;
  struct net_kept *k;
  struct net_message *output;uint32_t output_head,output_tail;size_t output_partial;
  unsigned char input[sizeof(struct net_message)*64];size_t input_bytes;
  struct { uint32_t from,generation; uint64_t key,owner; } pending[MESH_NET_COMMS];uint32_t pending_count;
  int send_blocked,receive_blocked,send_held;
  uint64_t bell,scanned,reaped,strays,heard,said;
  /* resumption: the peer bridge's instance as this pairing exchanged it (the lost session's is kept: k), the drop
     requests seen, the socket of the kernel's link events (the link's interface off: the session ends); `left`,
     the link's peer left the mesh (its LEAVE heard: `peer_left`, or this bridge leaves) and no bridge has paired on
     it since */
  uint64_t instance,drops;int events,left,peer_left;
  /* this pairing's identity (mesh.h mesh_net_link.pairing) and the node's client table as last sent to the peer */
  uint64_t pairing,clients_sent[MESH_NET_CLIENTS];int clients_known;
  /* the link reports sent to the peer: link_news when last looked, each node's sequence sent */
  uint64_t news,*sent;
  /* the estimator's (-E, estimate_observe): the receives' busy period open (1; 2: a chunk of a held isend's
     message posted in it, its sender maybe not ready, so no observation) (its start, the bytes landed
     in it), the observations waiting for their lag (each busy period's bytes and ns, and the evaluations ended
     when it closed: it is fitted once `lag` more have, as every statistic is read, mesh.h), the least-squares
     state of t = a + b x (x MB, t us) and its prior's trace, the observations since it last moved the link's
     beta, its step and the sign of its last move; estimate_news when last looked and each estimate's sequence
     sent to the peer (source x observer) */
  int period; uint64_t period_start,period_bytes;
  struct { uint64_t bytes,ns,evaluation; } lagged[64]; uint32_t lagged_head,lagged_tail;
  double theta[2],P[3],trace,step; uint32_t observed; int moved;
  uint64_t estimate_seen,*estimates_sent;
};
/* The link table beside the region (mesh-dataflow.h): the sessions write their links' up and down and
   the reports they take; link_news moves whenever a report this bridge holds does, and wakes every
   session to pass it on. */
static struct mesh_link_table *link_table;
static char link_table_region[64];
static _Atomic uint64_t link_news;
static struct net_session *net_sessions;
static uint32_t net_session_count;
/* SIGUSR1: every session ends as though its link were lost, and resumes (a supervisor's, and a check's). */
static _Atomic uint64_t net_drops;
static void net_drop(int signal){(void)signal;atomic_fetch_add_explicit(&net_drops,1,memory_order_relaxed);}
/* SIGUSR2: this bridge leaves the mesh (a membership change: every peer is told, LEAVE) and stops. */
static _Atomic int leaving;
static void leave_bridge(int signal){atomic_store(&leaving,1);stop_bridge(signal);}
/* SIGQUIT: this bridge ends as a crash would, at its threads' next loop boundary: nothing suspended, nothing told to
   its peers, nothing of the region or its keep written, the region left in place whatever is attached, and an exit
   status that is not 0 (its supervisor starts the next).  Only the device is torn down (a bridge that does not
   withdraw its registrations can wedge the RDMA provider: RDMA-RULES.md), so the successor finds exactly the keep a
   crash leaves: what the running bridge wrote in place. */
static void crash_bridge(int signal){atomic_store(&crashing,1);stop_bridge(signal);}
#define NET_LINK_NODES (5*64)
/* At exit the region's keep and link table are removed with it, unless kept for the next bridge (mesh-verbs.h
   `keeping`). */
static void link_table_down(void){
  if(keeping)return;
  if(link_table){char name[80];snprintf(name,sizeof name,"%s.links",link_table_region);shm_unlink(name);}
  if(net_keep_name[0])shm_unlink(net_keep_name);
}

static uint64_t net_now(void){return clock_gettime_nsec_np(CLOCK_MONOTONIC);}
static int net_dead(uint64_t owner);
struct net_transfer;
static int net_ended(const struct net_transfer *t);
struct net_message;
static void net_released(struct net_session *s,const struct net_message *message);
static void net_void_answered(struct net_session *s,const struct net_message *message);
static void net_links_moved(void){
  atomic_fetch_add_explicit(&link_news,1,memory_order_release);
  for(uint32_t i=0;i<net_session_count;i++){
    _Atomic uint64_t *bell=&net_sessions[i].counts->doorbell;
    atomic_fetch_add_explicit(bell,1,memory_order_release);
    os_sync_wake_by_address_any(bell,sizeof *bell,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
  }
}

/* ---- the link's rate, estimated (-E): the table's third writer ----
   Each session estimates the beta of its link into this node from its receives: a busy period opens when
   a chunk of a receive is posted and granted with none outstanding and closes when the last outstanding
   one lands (both ends are ready from its grant on, so it is wire time, not either end's lateness); a
   period of at least ESTIMATE_BYTES is one observation (x its MB, t its us) of t = a + b x, taken once the
   statistics' lag of function evaluations has ended after it (mesh.h: no call plans on a fresher one), fitted by
   recursive least squares with forgetting ESTIMATE_FORGET [Haykin 2014, Table 10.1], the covariance's
   trace capped at its prior's [Goodwin & Sin 1984] (rdma/allocate.py's Allocator), the prior the link's
   stated alpha and beta.  After ESTIMATE_MIN observations an estimate outside ESTIMATE_BAND of the
   table's beta (a relative accuracy the estimate stops at: inside it nothing moves) moves the table's
   beta toward it by the step, which halves (to 1/8 at least) each time the moves change direction (it
   oscillated); a move writes the link's beta into the table (an edit of the stated beta, the link's
   presence and alpha as stated) and passes it to every peer (BETA), each bridge applying the newest of
   each link's observer, so every table holds the observer's estimate.  design/heterogeneity.md R11: a
   robust online loop, a smaller step once it oscillates, a relative accuracy it stops at, a move only
   where it changes the modelled time by more than the band. */
#define ESTIMATE_BYTES (UINT64_C(1)<<20)
#define ESTIMATE_FORGET 0.98
#define ESTIMATE_MIN 8
#define ESTIMATE_BAND 0.1
static int estimating;
static uint32_t estimate_node;
struct estimate { uint64_t sequence; float beta; };
static struct estimate *estimates;
static pthread_mutex_t estimates_lock=PTHREAD_MUTEX_INITIALIZER;
static uint64_t estimate_sequence;
static _Atomic uint64_t estimate_news;
struct set_beta { uint32_t source,observer; float beta; };
static void estimate_edit(struct mesh_link_contents *c,const void *argument){
  const struct set_beta *b=argument;
  struct mesh_link_state *l=mesh_link_at(c,b->source,b->observer);
  if(l->stated)l->beta=b->beta;
}
/* An estimate of link source -> observer held (where newer than the one held) and written into the
   table; 1 where it was newer. */
static int estimate_hold(uint32_t source,uint32_t observer,uint64_t sequence,float beta){
  if(!link_table || source>=link_table->nodes || observer>=link_table->nodes)return 0;
  /* the table written under the same lock, so estimates are applied in their order */
  pthread_mutex_lock(&estimates_lock);
  struct estimate *e=estimates+(size_t)source*link_table->nodes+observer;
  int newer=sequence>e->sequence,moved=0;
  if(newer){e->sequence=sequence;e->beta=beta;moved=mesh_link_table_write(link_table,estimate_edit,&(struct set_beta){source,observer,beta});}
  pthread_mutex_unlock(&estimates_lock);
  if(!newer)return 0;
  if(moved)net_links_moved();
  atomic_fetch_add_explicit(&estimate_news,1,memory_order_release);
  return 1;
}
/* The table's beta of link source -> observer now (0: not stated). */
static float estimate_stated(uint32_t source,uint32_t observer){
  struct mesh_link_contents *c=mesh_link_contents_new(link_table->nodes);
  if(!c)return 0;
  mesh_link_table_read(link_table,c);
  const struct mesh_link_state *l=mesh_link_at(c,source,observer);
  float beta=l->stated?l->beta:0;
  free(c);
  return beta;
}
/* The estimator's prior from the link's stated alpha and beta, at a session's pairing. */
static void estimate_start(struct net_session *s){
  s->period=0;s->observed=0;s->moved=0;s->step=1;
  struct mesh_link_contents *c=link_table?mesh_link_contents_new(link_table->nodes):NULL;
  if(!c)return;
  mesh_link_table_read(link_table,c);
  const struct mesh_link_state *l=mesh_link_at(c,s->provider.peer,estimate_node);
  const double a=l->stated?l->alpha:20,b=l->stated?l->beta*1e3:100;
  free(c);
  s->theta[0]=a;s->theta[1]=b;s->P[0]=a*a+100;s->P[1]=0;s->P[2]=b*b/4;s->trace=s->P[0]+s->P[2];
}
static void estimate_fit(struct net_session *s,uint64_t bytes,uint64_t ns){
  const double x=(double)bytes/1e6,t=(double)ns/1e3,lambda=ESTIMATE_FORGET;
  double p00=s->P[0],p01=s->P[1],p11=s->P[2];
  const double pi0=p00+p01*x,pi1=p01+p11*x,den=lambda+pi0+pi1*x,k0=pi0/den,k1=pi1/den,e=t-s->theta[0]-s->theta[1]*x;
  s->theta[0]+=k0*e;s->theta[1]+=k1*e;
  p00=(p00-k0*pi0)/lambda;p01=(p01-k0*pi1)/lambda;p11=(p11-k1*pi1)/lambda;
  if(p00+p11>s->trace){const double f=s->trace/(p00+p11);p00*=f;p01*=f;p11*=f;}
  s->P[0]=p00;s->P[1]=p01;s->P[2]=p11;
  const double beta=s->theta[1]/1e3,stated=estimate_stated(s->provider.peer,estimate_node);
  if(++s->observed<ESTIMATE_MIN || beta<=0 || stated<=0 || fabs(beta-stated)<=ESTIMATE_BAND*stated)return;
  const int sign=beta>stated?1:-1;
  if(s->moved && sign!=s->moved)s->step=fmax(s->step/2,1.0/8);
  s->moved=sign;s->observed=0;
  pthread_mutex_lock(&estimates_lock);
  const uint64_t sequence=++estimate_sequence;
  pthread_mutex_unlock(&estimates_lock);
  const float next=(float)(stated+s->step*(beta-stated));
  say("estimate: link %u -> %u beta %.4f ns/B (fit %.4f, stated %.4f, step %.3f)\n",s->provider.peer,estimate_node,next,beta,stated,s->step);
  estimate_hold(s->provider.peer,estimate_node,sequence,next);
}
/* A busy period observed: held until the lag's evaluations have ended after it (mesh.h, the statistics' one
   read), then fitted (estimate_ripen). */
static void estimate_observe(struct net_session *s,uint64_t bytes,uint64_t ns){
  if(bytes<ESTIMATE_BYTES || !link_table)return;
  if(s->lagged_tail-s->lagged_head>=64)s->lagged_head++;
  const uint32_t at=s->lagged_tail++%64;
  s->lagged[at].bytes=bytes;s->lagged[at].ns=ns;s->lagged[at].evaluation=atomic_load_explicit(&s->M->evaluations,memory_order_acquire);
}
static void estimate_ripen(struct net_session *s){
  const uint64_t now=atomic_load_explicit(&s->M->evaluations,memory_order_acquire);
  while(s->lagged_head!=s->lagged_tail && s->lagged[s->lagged_head%64].evaluation+s->M->stats_lag<=now){
    const uint64_t bytes=s->lagged[s->lagged_head%64].bytes,ns=s->lagged[s->lagged_head%64].ns;
    s->lagged_head++;
    estimate_fit(s,bytes,ns);
  }
}
static void net_nap(uint64_t ns){for(uint64_t end=net_now()+ns;!stop && net_now()<end;)poll(NULL,0,10);}
/* The next cut after `at`: a multiple of the chunk, or where either end's registration region ends
   (extent e, the message's start at phase p within its region). */
static uint64_t net_cut(uint64_t at,uint64_t size,uint64_t chunk,uint64_t e1,uint64_t p1,uint64_t e2,uint64_t p2){
  uint64_t next=(at/chunk+1)*chunk;
  if(e1 && at+e1-(p1+at)%e1<next)next=at+e1-(p1+at)%e1;
  if(e2 && at+e2-(p2+at)%e2<next)next=at+e2-(p2+at)%e2;
  return next<size?next:size;
}
/* Where the window's registered regions end around `offset` (the device's region facts). */
static void net_geometry(struct net_session *s,uint64_t offset,uint64_t *extent,uint64_t *phase){
  *extent=s->provider.device->extent;*phase=offset%s->provider.device->extent;
}
static struct ibv_sge net_span(struct net_session *s,uint32_t mr,uint64_t offset,uint64_t length){
  struct mesh_device *device=s->provider.device;
  if(mr==NET_DISCARD)return (struct ibv_sge){.addr=(uintptr_t)device->discard,.length=(uint32_t)length,.lkey=device->discard_region->lkey};
  return (struct ibv_sge){.addr=(uintptr_t)device->wire+offset,.length=(uint32_t)length,.lkey=device->regions[offset/device->extent]->lkey};
}
static void net_emit(struct net_session *s,struct net_message message){
  if(s->output_tail-s->output_head>=NET_OUTPUT){s->failed=ENOBUFS;return;}
  s->output[s->output_tail++%NET_OUTPUT]=message;
  s->said=net_now();
  if(message.kind==NET_HEARTBEAT)atomic_fetch_add_explicit(&s->counts->heartbeats_sent,1,memory_order_relaxed);
}
static struct mesh_net_comm *net_comm_at(struct net_session *s,uint32_t index,uint32_t generation){
  if(index>=MESH_NET_COMMS)return NULL;
  struct mesh_net_comm *comm=mesh_net_comms(s->M)+index;
  uint32_t state=atomic_load_explicit(&comm->state,memory_order_acquire);
  if(state==MESH_NET_FREE || state==MESH_NET_CLAIMED || comm->link!=s->index || comm->generation!=generation)return NULL;
  return comm;
}
/* The bridge's side of a comm, begun afresh for each generation of its slot. */
static struct net_comm *net_state(struct mesh_net_comm *comm,uint32_t index){
  struct net_comm *state=net_comms+index;
  if(!state->live || state->generation!=comm->generation){
    memset(state,0,sizeof *state);state->generation=comm->generation;state->live=1;
  }
  return state;
}
/* A request's end: its completion word (a GPU kernel waits on it: mesh.h), then its state, which frees the
   slot for the client's next request. */
static struct hdr *net_region;
static void net_finish(struct mesh_net_request *request,int32_t error,uint64_t transferred){
  const uint64_t completion=request->completion;
  atomic_store_explicit(&request->error,error,memory_order_relaxed);
  atomic_store_explicit(&request->transferred,transferred,memory_order_relaxed);
  if(completion)atomic_store_explicit((_Atomic uint64_t *)((char *)net_region+net_region->data_off+completion-1),error?2:1,memory_order_release);
  atomic_store_explicit(&request->state,error?MESH_NET_ERROR:MESH_NET_DONE,memory_order_release);
}
/* A request on the wire ends: the comm's hold released, its counts, its state. */
static void net_end(struct net_session *s,struct mesh_net_comm *comm,struct net_comm *state,uint32_t slot,int32_t error,uint64_t bytes){
  struct mesh_net_request *request=comm->requests+slot;
  state->phase[slot]=0;state->outstanding--;
  if(!error){
    atomic_fetch_add_explicit(&comm->bytes,bytes,memory_order_relaxed);
    atomic_fetch_add_explicit(&comm->completions,1,memory_order_relaxed);
    if(request->op!=MESH_NET_IRECV){
      atomic_fetch_add_explicit(&s->counts->net_sends,1,memory_order_relaxed);
      atomic_fetch_add_explicit(&s->counts->net_send_bytes,bytes,memory_order_relaxed);
    } else {
      atomic_fetch_add_explicit(&s->counts->net_receives,1,memory_order_relaxed);
      atomic_fetch_add_explicit(&s->counts->net_receive_bytes,bytes,memory_order_relaxed);
    }
  }
  net_finish(request,error,bytes);
}
static void net_free(struct net_session *s,uint32_t index){
  struct mesh_net_comm *comm=mesh_net_comms(s->M)+index;
  net_comms[index].live=0;
  atomic_store_explicit(&comm->state,MESH_NET_FREE,memory_order_release);
}
/* A comm vacated with its requests not yet ended ended first (failed `error`: their completion words written). */
static void net_vacate_comm(struct net_session *s,uint32_t index,int32_t error){
  struct mesh_net_comm *comm=mesh_net_comms(s->M)+index;
  for(uint32_t slot=0;slot<MESH_NET_REQUESTS;slot++){
    const uint32_t r=atomic_load_explicit(&comm->requests[slot].state,memory_order_acquire);
    if(r==MESH_NET_POSTED || r==MESH_NET_ACTIVE)net_finish(comm->requests+slot,error,0);
  }
  net_free(s,index);
}
/* An accepted comm its client has not taken (ACCEPTABLE) moved to `to`, unless its client's accept took it first:
   whether it moved. */
static int net_unaccepted(struct mesh_net_comm *comm,uint32_t to){
  uint32_t acceptable=MESH_NET_ACCEPTABLE;
  return atomic_compare_exchange_strong_explicit(&comm->state,&acceptable,to,memory_order_acq_rel,memory_order_acquire);
}
/* This node's client table sent to the peer where it changed since last sent this pairing (or `force`): every slot,
   the last completing it (mesh.h peer_clients).  Before any CONNECT or ACCEPT naming a client, so the peer's table
   holds that client before it learns of it. */
static void net_publish_clients(struct net_session *s,int force){
  uint64_t now[MESH_NET_CLIENTS];int changed=force || !s->clients_known;
  for(uint32_t i=0;i<MESH_NET_CLIENTS;i++){
    now[i]=atomic_load_explicit(&mesh_net_clients(s->M)[i].owner,memory_order_acquire);
    changed|=now[i]!=s->clients_sent[i];
  }
  if(!changed)return;
  for(uint32_t i=0;i<MESH_NET_CLIENTS;i++)net_emit(s,(struct net_message){.kind=NET_CLIENTS,.flags=i,.key=now[i]});
  memcpy(s->clients_sent,now,sizeof now);s->clients_known=1;
}
static void net_refuse(struct net_session *s,const struct net_message *rts,int32_t error){
  net_emit(s,(struct net_message){.kind=NET_CREDIT,.to=rts->from,.to_generation=rts->from_generation,.from=rts->to,.from_generation=rts->to_generation,
    .sequence=rts->sequence,.error=error});
}

/* ---- requests ---- */
static int net_request_valid(struct net_session *s,struct mesh_net_request *request){
  uint64_t wire=mesh_wire_bytes(s->M);
  return !request->size || (request->mr==MESH_NET_WINDOW && s->provider.device->region_count && request->offset<=wire && request->size<=wire-request->offset);
}
/* A receive comm's irecvs in order, each with the next announced send: refused past its capacity,
   done at once when empty, else a message to receive, whose chunks net_post grants as it posts them.
   With the sender closed, an irecv no announcement will reach fails. */
static void net_match(struct net_session *s,uint32_t index){
  struct mesh_net_comm *comm=mesh_net_comms(s->M)+index;struct net_comm *state=net_comms+index;
  while(state->matched<state->taken && !s->failed){
    uint32_t slot=(uint32_t)(state->matched%MESH_NET_REQUESTS);
    struct mesh_net_request *request=comm->requests+slot;
    if(atomic_load_explicit(&request->state,memory_order_acquire)!=MESH_NET_ACTIVE || request->op!=MESH_NET_IRECV || state->phase[slot]!=1){
      state->matched++;continue;
    }
    if(state->announce_head==state->announce_tail){
      if(!state->peer_closed)break;
      net_finish(request,ECONNRESET,0);state->phase[slot]=0;state->matched++;continue;
    }
    if(s->k->receive_tail-s->k->receive_head>=NET_RECEIVES)break;
    struct net_announce a=state->announced[state->announce_head%MESH_NET_REQUESTS];
    struct net_message credit={.kind=NET_CREDIT,.to=comm->peer,.to_generation=comm->peer_generation,.from=index,.from_generation=comm->generation,.sequence=a.sequence};
    struct net_outcome *outcome=state->outcome+a.sequence%MESH_NET_REQUESTS;
    if(a.size>request->size || !a.size){
      *outcome=(struct net_outcome){a.sequence,a.size?3:2,a.size?EMSGSIZE:0};
      atomic_signal_fence(memory_order_seq_cst);
      state->announce_head++;state->matched++;
      if(a.size)credit.error=EMSGSIZE;
      net_emit(s,credit);net_finish(request,a.size?EMSGSIZE:0,0);state->phase[slot]=0;
      if(!a.size)atomic_fetch_add_explicit(&comm->completions,1,memory_order_relaxed);
      continue;
    }
    uint64_t extent,phase;
    net_geometry(s,request->offset,&extent,&phase);
    /* the receive kept before the announcement is consumed (a successor finds one, or both: net_normalize) */
    s->k->receives[s->k->receive_tail%NET_RECEIVES]=(struct net_transfer){.comm=index,.generation=comm->generation,.slot=slot,.mr=request->mr,
      .peer=comm->peer,.peer_generation=comm->peer_generation,.held=a.held!=0,.released=a.held==2?2:0,.sequence=a.sequence,.size=a.size,
      .offset=request->offset,.e1=extent,.p1=phase,.e2=a.extent,.p2=a.phase};
    atomic_signal_fence(memory_order_seq_cst);
    s->k->receive_tail++;
    atomic_signal_fence(memory_order_seq_cst);
    *outcome=(struct net_outcome){a.sequence,1,0};
    state->phase[slot]=2;state->outstanding++;
    atomic_signal_fence(memory_order_seq_cst);
    state->announce_head++;state->matched++;
  }
}
/* The client's new requests on a connected comm, in its order: an isend is announced at once and is on
   the wire until its receiver says every byte of it landed (LANDED); an irecv waits for its announcement; a
   flush completes once every earlier request of its comm has. */
static void net_take(struct net_session *s,uint32_t index){
  struct mesh_net_comm *comm=mesh_net_comms(s->M)+index;struct net_comm *state=net_comms+index;
  uint32_t kind=atomic_load_explicit(&comm->state,memory_order_acquire);
  uint64_t posted=atomic_load_explicit(&comm->posted,memory_order_acquire);
  /* each request's bridge side set before it is ACTIVE, and `taken` advanced last (a successor finds a request taken
     or not: net_normalize) */
  while(state->taken<posted && !s->failed){
    uint32_t slot=(uint32_t)(state->taken%MESH_NET_REQUESTS);
    struct mesh_net_request *request=comm->requests+slot;
    if(atomic_load_explicit(&request->state,memory_order_acquire)!=MESH_NET_POSTED)break;
    state->phase[slot]=0;state->sent[slot]=0;state->again[slot]=0;state->released[slot]=0;
    /* a completion word outside the window is refused, and never stored */
    const uint64_t word=request->completion,wire=mesh_wire_bytes(s->M);
    const int stray=word && (word-1>wire-8 || (word-1)%8);
    if(stray)request->completion=0;
    int ordered=request->sequence==state->taken,valid=!stray && ordered && net_request_valid(s,request);
    const uint32_t op=atomic_load_explicit(&request->op,memory_order_acquire);
    const int sending=op==MESH_NET_ISEND || op==MESH_NET_HELD;
    int32_t refused=0;
    if(op==MESH_NET_IFLUSH && kind==MESH_NET_RECV && ordered){}
    else if(!valid || sending!=(kind==MESH_NET_SEND) || (!sending && op!=MESH_NET_IRECV))refused=EINVAL;
    else if(state->peer_closed && sending)refused=ECONNRESET;
    else if(op==MESH_NET_IRECV)state->phase[slot]=1;
    else {
      uint64_t extent=0,phase=0;
      if(request->size)net_geometry(s,request->offset,&extent,&phase);
      net_emit(s,(struct net_message){.kind=NET_RTS,.to=comm->peer,.to_generation=comm->peer_generation,.from=index,.from_generation=comm->generation,
        .flags=op==MESH_NET_HELD,.sequence=request->sequence,.size=request->size,.extent=extent,.phase=phase,.tag=request->tag});
      state->phase[slot]=1;state->released[slot]=op==MESH_NET_HELD;state->outstanding++;
    }
    atomic_signal_fence(memory_order_seq_cst);
    atomic_store_explicit(&request->state,MESH_NET_ACTIVE,memory_order_release);
    if(refused)net_finish(request,refused,0);
    atomic_signal_fence(memory_order_seq_cst);
    state->taken++;
  }
  if(kind!=MESH_NET_RECV)return;
  net_match(s,index);
  for(uint32_t slot=0;slot<MESH_NET_REQUESTS;slot++){
    struct mesh_net_request *flush=comm->requests+slot;
    if(flush->op!=MESH_NET_IFLUSH || atomic_load_explicit(&flush->state,memory_order_acquire)!=MESH_NET_ACTIVE)continue;
    int earlier=0;
    for(uint32_t other=0;other<MESH_NET_REQUESTS && !earlier;other++){
      struct mesh_net_request *request=comm->requests+other;uint32_t at=atomic_load_explicit(&request->state,memory_order_acquire);
      earlier=(at==MESH_NET_POSTED || at==MESH_NET_ACTIVE) && request->sequence<flush->sequence;
    }
    if(!earlier){atomic_thread_fence(memory_order_seq_cst);net_finish(flush,0,0);}
  }
}

/* ---- connections ---- */
static uint32_t net_listening(struct net_session *s,uint64_t key){
  struct mesh_net_comm *comms=mesh_net_comms(s->M);
  for(uint32_t i=0;i<MESH_NET_COMMS;i++)
    if(atomic_load_explicit(&comms[i].state,memory_order_acquire)==MESH_NET_LISTEN && comms[i].link==s->index && comms[i].key==key)return i;
  return NET_NONE;
}
/* A connect meets its listen: a receive comm for the listen's client, ACCEPTABLE until the client's
   accept takes it, and the sender's comm told which comm receives it. */
static int net_accept(struct net_session *s,uint32_t listen,uint32_t from,uint32_t generation,uint64_t owner){
  struct mesh_net_comm *comms=mesh_net_comms(s->M),*origin=comms+listen;
  net_publish_clients(s,0);
  for(uint32_t i=0;i<MESH_NET_COMMS;i++){
    uint32_t vacant=MESH_NET_FREE;
    if(!atomic_compare_exchange_strong_explicit(&comms[i].state,&vacant,MESH_NET_CLAIMED,memory_order_acq_rel,memory_order_relaxed))continue;
    struct mesh_net_comm *comm=comms+i;
    comm->generation++;comm->link=s->index;comm->listen=listen;comm->listen_generation=origin->generation;
    comm->peer=from;comm->peer_generation=generation;comm->owner=origin->owner;comm->key=origin->key;comm->peer_owner=owner;
    atomic_store_explicit(&comm->error,0,memory_order_relaxed);
    atomic_store_explicit(&comm->posted,0,memory_order_relaxed);atomic_store_explicit(&comm->bytes,0,memory_order_relaxed);
    atomic_store_explicit(&comm->completions,0,memory_order_relaxed);atomic_store_explicit(&comm->credit_waits,0,memory_order_relaxed);
    for(uint32_t r=0;r<MESH_NET_REQUESTS;r++)atomic_store_explicit(&comm->requests[r].state,MESH_NET_IDLE,memory_order_relaxed);
    net_state(comm,i);
    atomic_store_explicit(&comm->state,MESH_NET_ACCEPTABLE,memory_order_release);
    net_emit(s,(struct net_message){.kind=NET_ACCEPT,.to=from,.to_generation=generation,.from=i,.from_generation=comm->generation,.offset=comm->owner});
    return 1;
  }
  return 0;
}
static void net_connected(struct net_session *s,const struct net_message *message){
  /* a connect sent again as its session resumed, whose accept was lost with the session: accepted again */
  struct mesh_net_comm *comms=mesh_net_comms(s->M);
  for(uint32_t i=0;i<MESH_NET_COMMS;i++){
    uint32_t at=atomic_load_explicit(&comms[i].state,memory_order_acquire);
    if((at!=MESH_NET_ACCEPTABLE && at!=MESH_NET_RECV) || comms[i].link!=s->index || comms[i].peer!=message->from ||
       comms[i].peer_generation!=message->from_generation)continue;
    net_publish_clients(s,0);
    net_emit(s,(struct net_message){.kind=NET_ACCEPT,.to=message->from,.to_generation=message->from_generation,.from=i,.from_generation=comms[i].generation,
      .offset=comms[i].owner});
    return;
  }
  uint32_t listen=net_listening(s,message->key);
  if(listen!=NET_NONE && net_accept(s,listen,message->from,message->from_generation,message->offset))return;
  if(listen==NET_NONE && s->pending_count<MESH_NET_COMMS){
    s->pending[s->pending_count].from=message->from;s->pending[s->pending_count].generation=message->from_generation;
    s->pending[s->pending_count].owner=message->offset;s->pending[s->pending_count++].key=message->key;
    return;
  }
  net_emit(s,(struct net_message){.kind=NET_CLOSE,.to=message->from,.to_generation=message->from_generation,.from=NET_NONE,.error=ENOSPC});
}
/* CLOSE: the peer announces and grants nothing more on this connection.  `to` NONE withdraws a
   connect: its pending entry, or the comm it made.  A stale CLOSE asking for one is answered, so its
   sender can vacate.  What the peer already announced or granted stays good. */
static void net_closed(struct net_session *s,const struct net_message *message){
  struct mesh_net_comm *comm=NULL,*comms=mesh_net_comms(s->M);
  uint32_t index=message->to;
  if(index==NET_NONE){
    for(uint32_t p=0;p<s->pending_count;p++)if(s->pending[p].from==message->from && s->pending[p].generation==message->from_generation){
      s->pending[p]=s->pending[--s->pending_count];
      net_emit(s,(struct net_message){.kind=NET_CLOSE,.to=message->from,.to_generation=message->from_generation,.from=NET_NONE});
      return;
    }
    for(index=0;index<MESH_NET_COMMS && !comm;index++){
      uint32_t at=atomic_load_explicit(&comms[index].state,memory_order_acquire);
      if(at!=MESH_NET_FREE && at!=MESH_NET_CLAIMED && at!=MESH_NET_LISTEN && at!=MESH_NET_CONNECTING && comms[index].link==s->index &&
         comms[index].peer==message->from && comms[index].peer_generation==message->from_generation)comm=comms+index;
    }
    index--;
  } else comm=net_comm_at(s,index,message->to_generation);
  if(!comm){
    if(message->flags && message->from!=NET_NONE)
      net_emit(s,(struct net_message){.kind=NET_CLOSE,.to=message->from,.to_generation=message->from_generation,.from=message->to,.from_generation=message->to_generation});
    return;
  }
  struct net_comm *state=net_state(comm,index);
  uint32_t connecting=MESH_NET_CONNECTING;
  if(atomic_compare_exchange_strong_explicit(&comm->state,&connecting,MESH_NET_FAILED,memory_order_acq_rel,memory_order_relaxed))
    atomic_store_explicit(&comm->error,message->error?message->error:ECONNREFUSED,memory_order_relaxed);
  /* a withdrawn connect's comm not yet taken by its client is never taken; a connected comm's client sees it closed
     (ESRCH: its peer's client had exited) */
  else if(message->to==NET_NONE)net_unaccepted(comm,MESH_NET_CLOSING);
  int32_t none=0;
  atomic_compare_exchange_strong_explicit(&comm->error,&none,message->error==ESRCH?ESRCH:ECONNRESET,memory_order_relaxed,memory_order_relaxed);
  state->peer_closed=1;
  if(atomic_load_explicit(&comm->state,memory_order_acquire)==MESH_NET_RECV)net_match(s,index);
  if(!state->close_sent){
    state->close_sent=1;
    if(message->from!=NET_NONE)
      net_emit(s,(struct net_message){.kind=NET_CLOSE,.to=message->from,.to_generation=message->from_generation,.from=index,.from_generation=comm->generation,
        .error=net_dead(comm->owner)?ESRCH:0});
  }
}
/* RTS: queued for the receive comm's next irecv, or refused when that comm is gone or closing.  A comm
   accepted but not yet taken by its client (ACCEPTABLE) queues it too: the sender's connect completes at
   this bridge's accept, so it may announce before the client's accept has taken the comm. */
static void net_announced(struct net_session *s,const struct net_message *message){
  struct mesh_net_comm *comm=net_comm_at(s,message->to,message->to_generation);
  uint32_t at=comm?atomic_load_explicit(&comm->state,memory_order_acquire):MESH_NET_FREE;
  if(at!=MESH_NET_RECV && at!=MESH_NET_ACCEPTABLE){net_refuse(s,message,ECONNRESET);return;}
  struct net_comm *state=net_state(comm,message->to);
  /* announced again as a session resumes: one this comm holds is waiting for its irecv, or in progress (its RESUME
     answers it), or ended, answered as it ended (its LANDED, empty CREDIT or refusal, lost with the session) */
  if(message->flags&2){
    for(uint64_t a=state->announce_head;a!=state->announce_tail;a++)if(state->announced[a%MESH_NET_REQUESTS].sequence==message->sequence)return;
    if(message->sequence<state->seen){
      for(uint32_t r=s->k->receive_head;r!=s->k->receive_tail;r++){
        const struct net_transfer *t=s->k->receives+r%NET_RECEIVES;
        if(t->comm==message->to && t->generation==message->to_generation && t->sequence==message->sequence && !net_ended(t))return;
      }
      const struct net_outcome *o=state->outcome+message->sequence%MESH_NET_REQUESTS;
      const int known=o->sequence==message->sequence;
      net_emit(s,(struct net_message){.kind=known && o->ended>=2?NET_CREDIT:NET_LANDED,.to=message->from,.to_generation=message->from_generation,
        .from=message->to,.from_generation=message->to_generation,.sequence=message->sequence,.size=known && o->ended>=2?0:message->size,
        .error=known && o->ended==3?o->error:0});
      return;
    }
  }
  if(state->announce_tail-state->announce_head>=MESH_NET_REQUESTS){s->failed=EPROTO;return;}
  state->announced[state->announce_tail%MESH_NET_REQUESTS]=(struct net_announce){message->sequence,message->size,message->extent,message->phase,
    message->flags&1};
  atomic_signal_fence(memory_order_seq_cst);
  state->announce_tail++;
  atomic_signal_fence(memory_order_seq_cst);
  if(message->sequence>=state->seen)state->seen=message->sequence+1;
  uint64_t before=state->announce_head;
  net_match(s,message->to);
  if(state->announce_head==before){
    atomic_fetch_add_explicit(&comm->credit_waits,1,memory_order_relaxed);
    atomic_fetch_add_explicit(&s->counts->credit_waits,1,memory_order_relaxed);
  }
}
/* CREDIT: a chunk whose RECV the peer posted, SENT from its request's buffer; a grant nobody holds is
   filled from the discard buffer all the same, so the peer's RECV is not left waiting. */
static void net_granted(struct net_session *s,const struct net_message *message){
  struct mesh_net_comm *comm=net_comm_at(s,message->to,message->to_generation);
  uint32_t slot=(uint32_t)(message->sequence%MESH_NET_REQUESTS);
  struct mesh_net_request *request=comm?comm->requests+slot:NULL;
  struct net_comm *state=comm?net_state(comm,message->to):NULL;
  const uint32_t op=request?atomic_load_explicit(&request->op,memory_order_acquire):MESH_NET_IRECV;
  if(request && (atomic_load_explicit(&request->state,memory_order_acquire)!=MESH_NET_ACTIVE || (op!=MESH_NET_ISEND && op!=MESH_NET_HELD) ||
     request->sequence!=message->sequence || !state->phase[slot]))request=NULL;
  if(message->error || !message->size){
    if(request)net_end(s,comm,state,slot,message->error,0);
    return;
  }
  if(s->send_tail-s->send_head>=NET_SENDS){s->failed=ENOBUFS;return;}
  if(message->size>s->chunk || message->flags>=NET_QUEUES){s->failed=EPROTO;return;}
  if(!request || message->offset>request->size || message->size>request->size-message->offset){
    if(request)net_end(s,comm,state,slot,EPROTO,0);
    s->strays++;
    s->sends[s->send_tail++%NET_SENDS]=(struct net_send){.comm=NET_NONE,.mr=NET_DISCARD,.queue=message->flags,.length=message->size};
    return;
  }
  state->phase[slot]=2;
  atomic_fetch_add_explicit(&request->transferred,message->size,memory_order_relaxed);  /* granted so far (mesh.h) */
  if(message->offset<state->again[slot])atomic_fetch_add_explicit(&s->counts->resends,1,memory_order_relaxed);
  s->sends[s->send_tail++%NET_SENDS]=(struct net_send){.comm=message->to,.generation=message->to_generation,.slot=slot,.mr=request->mr,
    .queue=message->flags,.sequence=message->sequence,.offset=request->offset+message->offset,.length=message->size};
}
/* Every report newer than the one last sent to this session's peer, sent (its own node's excepted). */
static void net_publish(struct net_session *s){
  for(uint32_t v=0;link_table && v<link_table->nodes && !s->failed;v++){
    uint64_t up[5]={0},sequence=v==s->provider.peer?0:mesh_link_table_row(link_table,v,up);
    if(sequence<=s->sent[v])continue;
    net_emit(s,(struct net_message){.kind=NET_LINKS,.from=v,.sequence=sequence,.key=up[0],.size=up[1],.extent=up[2],.phase=up[3],.offset=up[4]});
    s->sent[v]=sequence;
  }
}
/* Every estimate newer than the one last sent to this session's peer, sent (the peer's own excepted). */
static void net_publish_estimates(struct net_session *s){
  const uint32_t n=link_table->nodes;
  for(uint32_t source=0;source<n && !s->failed;source++)for(uint32_t observer=0;observer<n && !s->failed;observer++){
    if(observer==s->provider.peer)continue;
    pthread_mutex_lock(&estimates_lock);
    const struct estimate e=estimates[(size_t)source*n+observer];
    pthread_mutex_unlock(&estimates_lock);
    uint64_t *sent=s->estimates_sent+(size_t)source*n+observer;
    if(e.sequence<=*sent)continue;
    uint32_t bits;memcpy(&bits,&e.beta,sizeof bits);
    net_emit(s,(struct net_message){.kind=NET_BETA,.from=observer,.key=source,.sequence=e.sequence,.size=bits});
    *sent=e.sequence;
  }
}
/* A peer's estimate: the peer holds it (not sent back), and, where newer, held, written and passed on. */
static void net_estimated(struct net_session *s,const struct net_message *message){
  if(!link_table || message->from>=link_table->nodes || message->key>=link_table->nodes)return;
  const uint32_t n=link_table->nodes,bits=(uint32_t)message->size;
  float beta;memcpy(&beta,&bits,sizeof beta);
  uint64_t *sent=s->estimates_sent+(size_t)message->key*n+message->from;
  if(message->sequence>*sent)*sent=message->sequence;
  if(isfinite(beta) && beta>0 && estimate_hold((uint32_t)message->key,message->from,message->sequence,beta))
    say("estimate: link %u -> %u beta %.4f ns/B (node %u's)\n",(uint32_t)message->key,message->from,beta,message->from);
}
/* A peer's report: the peer holds it (not sent back), and, where newer, applied and passed on. */
static void net_links(struct net_session *s,const struct net_message *message){
  const uint64_t up[5]={message->key,message->size,message->extent,message->phase,message->offset};
  if(!link_table || message->from>=link_table->nodes)return;
  if(message->sequence>s->sent[message->from])s->sent[message->from]=message->sequence;
  if(link_table && mesh_link_table_report(link_table,message->from,message->sequence,up))net_links_moved();
}
/* LANDED: the send request ends, done (every byte of it landed, whatever of its SENDs' completions are still to come). */
static void net_landed(struct net_session *s,const struct net_message *message){
  struct mesh_net_comm *comm=net_comm_at(s,message->to,message->to_generation);
  if(!comm)return;
  struct net_comm *state=net_state(comm,message->to);
  const uint32_t slot=(uint32_t)(message->sequence%MESH_NET_REQUESTS);
  struct mesh_net_request *request=comm->requests+slot;
  const uint32_t op=atomic_load_explicit(&request->op,memory_order_acquire);
  if(atomic_load_explicit(&request->state,memory_order_acquire)!=MESH_NET_ACTIVE || (op!=MESH_NET_ISEND && op!=MESH_NET_HELD) ||
     request->sequence!=message->sequence || !state->phase[slot])return;
  net_end(s,comm,state,slot,0,request->size);
}
/* RESUME.  The sender (flags 0): its request goes on from the bytes landed, the receiver granting the rest
   again, or, gone (its comm closed, or ended failed), the receiver's fails.  The receiver (flags 1): the
   answer, to grant the rest (net_post) or to fail its request. */
static void net_resumed(struct net_session *s,const struct net_message *message){
  if(!message->flags){
    struct net_message answer={.kind=NET_RESUME,.flags=1,.to=message->from,.to_generation=message->from_generation,.from=message->to,
      .from_generation=message->to_generation,.sequence=message->sequence,.offset=message->offset};
    struct mesh_net_comm *comm=net_comm_at(s,message->to,message->to_generation);
    struct net_comm *state=comm?net_state(comm,message->to):NULL;
    const uint32_t slot=(uint32_t)(message->sequence%MESH_NET_REQUESTS);
    struct mesh_net_request *request=comm?comm->requests+slot:NULL;
    const uint32_t op=request?atomic_load_explicit(&request->op,memory_order_acquire):MESH_NET_IRECV;
    if(!request || atomic_load_explicit(&request->state,memory_order_acquire)!=MESH_NET_ACTIVE || (op!=MESH_NET_ISEND && op!=MESH_NET_HELD) ||
       request->sequence!=message->sequence || !state->phase[slot] || message->offset>request->size)answer.error=ECONNRESET;
    else{
      const uint64_t granted=atomic_load_explicit(&request->transferred,memory_order_relaxed);
      if(granted>state->again[slot])state->again[slot]=granted;
      state->sent[slot]=message->offset;state->phase[slot]=2;
      atomic_store_explicit(&request->transferred,message->offset,memory_order_relaxed);
    }
    net_emit(s,answer);
    return;
  }
  for(uint32_t r=s->k->receive_head;r!=s->k->receive_tail;r++){
    struct net_transfer *t=s->k->receives+r%NET_RECEIVES;
    if(!t->resume || t->comm!=message->to || t->generation!=message->to_generation || t->sequence!=message->sequence)continue;
    t->resume=0;
    if(message->error){
      struct mesh_net_comm *comm=net_comm_at(s,t->comm,t->generation);
      t->cursor=t->landed=t->size;
      if(comm)net_end(s,comm,net_state(comm,t->comm),t->slot,message->error,0);
    }
    return;
  }
}
static void net_receive(struct net_session *s,const struct net_message *message){
  switch(message->kind){
  case NET_CONNECT: net_connected(s,message); break;
  case NET_ACCEPT: {
    struct mesh_net_comm *comm=net_comm_at(s,message->to,message->to_generation);
    if(!comm)break;
    comm->peer=message->from;comm->peer_generation=message->from_generation;comm->peer_owner=message->offset;
    uint32_t connecting=MESH_NET_CONNECTING;
    atomic_compare_exchange_strong_explicit(&comm->state,&connecting,MESH_NET_SEND,memory_order_acq_rel,memory_order_relaxed);
    break;
  }
  case NET_RTS: net_announced(s,message); break;
  case NET_CREDIT: net_granted(s,message); break;
  case NET_CLOSE: net_closed(s,message); break;
  case NET_HEARTBEAT: atomic_fetch_add_explicit(&s->counts->heartbeats_heard,1,memory_order_relaxed); break;
  case NET_LINKS: net_links(s,message); break;
  case NET_BETA: if(estimates)net_estimated(s,message); break;
  case NET_LANDED: net_landed(s,message); break;
  case NET_RESUME: net_resumed(s,message); break;
  case NET_LEAVE: s->peer_left=1;s->failed=ECONNRESET; break;
  case NET_CLIENTS:
    if(message->flags>=MESH_NET_CLIENTS)break;
    atomic_store_explicit(&s->counts->peer_clients[message->flags],message->key,memory_order_release);
    if(message->flags==MESH_NET_CLIENTS-1)atomic_store_explicit(&s->counts->clients_pairing,s->pairing,memory_order_release);
    break;
  case NET_VOID: net_released(s,message); break;
  case NET_VOIDED: net_void_answered(s,message); break;
  default: s->failed=EPROTO;
  }
}

/* ---- the wire ---- */
/* A granted chunk of a held isend (mesh.h MESH_NET_HELD) waits, and every chunk granted after it (the peer's
   RECVs meet the SENDs in grant order), until its client releases it or closes its comm. */
static int net_held(struct net_session *s,const struct net_send *chunk){
  if(chunk->comm==NET_NONE)return 0;
  struct mesh_net_comm *comm=net_comm_at(s,chunk->comm,chunk->generation);
  if(!comm || atomic_load_explicit(&comm->state,memory_order_acquire)!=MESH_NET_SEND)return 0;
  return atomic_load_explicit(&comm->requests[chunk->slot].op,memory_order_acquire)==MESH_NET_HELD;
}
/* Whether a granted chunk is of a held isend voided (its comm closed before its client released it: `released` 3, its
   VOID said; 4, its receiver's VOIDED heard): SENT from the discard buffer, its receiver's RECV met, once 4. */
static int net_voided(struct net_session *s,const struct net_send *chunk,int answered){
  if(chunk->comm==NET_NONE)return 0;
  struct mesh_net_comm *comm=net_comm_at(s,chunk->comm,chunk->generation);
  return comm && comm->requests[chunk->slot].sequence==chunk->sequence && net_comms[chunk->comm].released[chunk->slot]>=(answered?4:3);
}
/* Granted chunks are SENT in grant order, each on its queue pair while it has the frames and, on another
   queue pair than the SENDs outstanding, once those have completed (a held one stops them); matched
   messages' chunks are posted as RECVs in match order on a queue pair with the frames (the last one's while
   it has them), each granted once posted, naming it. */
static int net_post(struct net_session *s){
  s->send_held=0;
  while(s->send_post!=s->send_tail){
    struct net_send *chunk=s->sends+s->send_post%NET_SENDS;
    if(net_held(s,chunk)){s->send_held=1;break;}
    if(!chunk->voided && net_voided(s,chunk,0)){chunk->voided=1;chunk->mr=NET_DISCARD;chunk->offset=0;}
    if(chunk->voided && !net_voided(s,chunk,1)){s->send_held=1;break;}
    uint32_t frames=(uint32_t)((chunk->length+4095)/4096);
    if((chunk->queue!=s->send_queue && s->send_posted!=s->send_retired) || s->send_posted-s->send_retired+frames>s->send_capacity){
      if(!s->send_blocked){s->send_blocked=1;atomic_fetch_add_explicit(&s->counts->send_stalls,1,memory_order_relaxed);}
      break;
    }
    s->send_blocked=0;s->send_queue=chunk->queue;
    struct ibv_sge span=net_span(s,chunk->mr,chunk->offset,chunk->length);
    struct ibv_send_wr request={.wr_id=s->send_post,.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
    int error=ibv_post_send(s->provider.queues[chunk->queue].pair,&request,&bad);
    if(error)return error<0?-error:error;
    s->send_post++;s->send_posted+=frames;
  }
  while(s->k->receive_post!=s->k->receive_tail){
    struct net_transfer *t=s->k->receives+s->k->receive_post%NET_RECEIVES;
    if(t->cursor>=t->size){s->k->receive_post++;continue;}
    if(t->resume)break;  /* its sender's RESUME answer first: grants stay in match order */
    uint64_t next=net_cut(t->cursor,t->size,s->chunk,t->e1,t->p1,t->e2,t->p2),length=next-t->cursor;
    uint32_t frames=(uint32_t)((length+4095)/4096),q=s->receive_queue;
    for(uint32_t tried=1;tried<NET_QUEUES && s->queue_frames[q]+frames>s->receive_capacity;tried++)q=(q+1)%NET_QUEUES;
    if(s->queue_frames[q]+frames>s->receive_capacity || s->k->chunk_tail-s->k->chunk_head>=s->receive_requests){
      if(!s->receive_blocked){s->receive_blocked=1;atomic_fetch_add_explicit(&s->counts->receive_stalls,1,memory_order_relaxed);}
      break;
    }
    s->receive_blocked=0;
    struct ibv_sge span=net_span(s,t->mr,t->offset+t->cursor,length);
    struct ibv_recv_wr request={.wr_id=s->k->chunk_tail,.sg_list=&span,.num_sge=1},*bad;
    /* the chunk recorded in the keep before its RECV is posted, counted once posted */
    s->k->chunks[s->k->chunk_tail%NET_CHUNKS]=(struct net_chunk){s->k->receive_post,frames,q,0,length,t->cursor};
    atomic_signal_fence(memory_order_seq_cst);
    int error=ibv_post_recv(s->provider.queues[q].pair,&request,&bad);
    if(error)return error<0?-error:error;
    s->k->chunk_tail++;
    atomic_signal_fence(memory_order_seq_cst);
    if(t->cursor<t->again)atomic_fetch_add_explicit(&s->counts->reposts,1,memory_order_relaxed);
    if(estimating && !s->period){s->period=1;s->period_start=net_now();s->period_bytes=0;}
    if(estimating && t->held)s->period=2;
    s->receive_outstanding+=frames;s->queue_frames[q]+=frames;s->receive_queue=q;
    net_emit(s,(struct net_message){.kind=NET_CREDIT,.to=t->peer,.to_generation=t->peer_generation,.from=t->comm,.from_generation=t->generation,
      .flags=q,.sequence=t->sequence,.offset=t->cursor,.size=length});
    t->cursor=next;
  }
  return 0;
}
/* A failed completion, logged with the work request it names: the session ends on it. */
static int net_failed(struct net_session *s,const struct ibv_wc *done,const char *kind,uint64_t expected,struct ibv_sge span,uint32_t mr,uint32_t transfer){
  say("session link %u: %s completion %s (status %d vendor %u) wr_id %llu expected %llu byte_len %u; its SGE addr=0x%llx length=%u lkey=0x%x registration %u transfer %u\n",
    s->index,kind,ibv_wc_status_str(done->status),done->status,done->vendor_err,(unsigned long long)done->wr_id,(unsigned long long)expected,done->byte_len,
    (unsigned long long)span.addr,span.length,span.lkey,mr,transfer);
  return EIO;
}
/* Whether a message to receive has ended: every byte landed and, a held isend's, its end taken (net_landing). */
static int net_ended(const struct net_transfer *t){return t->landed>=t->size && (!t->held || t->released==3);}
/* A message every byte of which has landed ends (its sender told: LANDED), a held isend's as any other (a voided one's
   bytes go only once its VOID has been answered: it fails, its VOID heard first). */
static void net_landing(struct net_session *s,struct net_transfer *t){
  if(t->released==3 || t->landed<t->size)return;
  const int voided=t->released==2;
  struct mesh_net_comm *comm=net_comm_at(s,t->comm,t->generation);
  if(comm){
    struct net_comm *state=net_state(comm,t->comm);
    if(voided)state->outcome[t->sequence%MESH_NET_REQUESTS]=(struct net_outcome){t->sequence,3,ECONNRESET};
    net_end(s,comm,state,t->slot,voided?ECONNRESET:0,voided?0:t->size);
  }
  t->released=3;
  net_emit(s,(struct net_message){.kind=voided?NET_CREDIT:NET_LANDED,.to=t->peer,.to_generation=t->peer_generation,.from=t->comm,.from_generation=t->generation,
    .sequence=t->sequence,.size=voided?0:t->size,.error=voided?ECONNRESET:0});
}
/* VOID for a held isend: its receive (or its announcement still waiting for an irecv) marked failed, and answered
   (VOIDED: its chunks may go now, from the discard buffer), again however often it is said (a session that resumed
   says it again where its answer may have been lost). */
static void net_released(struct net_session *s,const struct net_message *message){
  net_emit(s,(struct net_message){.kind=NET_VOIDED,.to=message->from,.to_generation=message->from_generation,.from=message->to,
    .from_generation=message->to_generation,.sequence=message->sequence});
  struct mesh_net_comm *comm=net_comm_at(s,message->to,message->to_generation);
  if(!comm)return;
  struct net_comm *state=net_state(comm,message->to);
  for(uint64_t a=state->announce_head;a!=state->announce_tail;a++){
    struct net_announce *announced=state->announced+a%MESH_NET_REQUESTS;
    if(announced->sequence==message->sequence){announced->held=2;return;}
  }
  for(uint32_t r=s->k->receive_head;r!=s->k->receive_tail;r++){
    struct net_transfer *t=s->k->receives+r%NET_RECEIVES;
    if(t->comm!=message->to || t->generation!=message->to_generation || t->sequence!=message->sequence || t->released)continue;
    t->released=2;
    net_landing(s,t);
    return;
  }
}
/* VOIDED: a held isend's VOID answered by its receiver (`released` 4: its chunks go). */
static void net_void_answered(struct net_session *s,const struct net_message *message){
  struct mesh_net_comm *comm=net_comm_at(s,message->to,message->to_generation);
  const uint32_t slot=(uint32_t)(message->sequence%MESH_NET_REQUESTS);
  if(comm && comm->requests[slot].sequence==message->sequence && net_comms[message->to].released[slot]==3)net_comms[message->to].released[slot]=4;
}
/* SENDs complete in the order posted (their queue pairs' one at a time); RECVs complete in each queue pair's
   order, each taken as it lands and retired in posting order.  A SEND's completion retires its frames and counts
   its bytes sent; the request ends on its receiver's LANDED.  A receive whose every byte has landed ends, and
   its sender is told (LANDED).  `lenient` (a lost session's completions still queued, net_suspend): a failed
   completion is not an error, a RECV's is taken as not landed and the SENDs' are taken no further. */
static int net_complete(struct net_session *s,int *busy,int lenient){
  struct ibv_wc done[16];
  int count=ibv_poll_cq(s->provider.sent,16,done);
  if(count<0)return EIO;
  for(int i=0;i<count;i++){
    if(done[i].status || done[i].wr_id!=s->send_head){
      if(lenient)break;
      struct net_send *chunk=s->sends+s->send_head%NET_SENDS;
      return net_failed(s,done+i,"SEND",s->send_head,net_span(s,chunk->mr,chunk->offset,chunk->length),chunk->mr,chunk->comm);
    }
    struct net_send chunk=s->sends[s->send_head++%NET_SENDS];
    s->send_retired+=(chunk.length+4095)/4096;
    struct mesh_net_comm *comm=chunk.comm==NET_NONE?NULL:net_comm_at(s,chunk.comm,chunk.generation);
    if(!comm || comm->requests[chunk.slot].sequence!=chunk.sequence)continue;
    struct net_comm *state=net_state(comm,chunk.comm);
    if(state->phase[chunk.slot]==2)state->sent[chunk.slot]+=chunk.length;
  }
  *busy|=count>0;
  count=ibv_poll_cq(s->provider.completion,16,done);
  if(count<0)return EIO;
  for(int i=0;i<count;i++){
    const uint32_t index=(uint32_t)done[i].wr_id;
    struct net_chunk *chunk=s->k->chunks+index%NET_CHUNKS;
    if(done[i].status || done[i].wr_id>UINT32_MAX || index-s->k->chunk_head>=s->k->chunk_tail-s->k->chunk_head || chunk->landed ||
       done[i].byte_len!=chunk->length){
      if(lenient)continue;
      const struct net_transfer *of=chunk->transfer==NET_NONE?NULL:s->k->receives+chunk->transfer%NET_RECEIVES;
      return net_failed(s,done+i,"RECV",s->k->chunk_head,of?net_span(s,of->mr,of->offset+chunk->at,chunk->length):net_span(s,NET_DISCARD,0,chunk->length),
                        of?of->mr:NET_DISCARD,chunk->transfer);
    }
    chunk->landed=1;
    s->receive_outstanding-=chunk->frames;s->queue_frames[chunk->queue]-=chunk->frames;
    if(chunk->transfer==NET_NONE)continue;
    s->period_bytes+=chunk->length;
    struct net_transfer *t=s->k->receives+chunk->transfer%NET_RECEIVES;
    if((t->landed+=chunk->length)<t->size)continue;
    net_landing(s,t);
  }
  while(s->k->chunk_head!=s->k->chunk_tail && s->k->chunks[s->k->chunk_head%NET_CHUNKS].landed)s->k->chunk_head++;
  while(s->k->receive_head!=s->k->receive_post && net_ended(s->k->receives+s->k->receive_head%NET_RECEIVES))s->k->receive_head++;
  if(s->period && !s->receive_outstanding){if(s->period==1)estimate_observe(s,s->period_bytes,net_now()-s->period_start);s->period=0;}
  *busy|=count>0;
  return 0;
}
static int net_flush(struct net_session *s){
  while(s->output_head!=s->output_tail){
    char *message=(char *)(s->output+s->output_head%NET_OUTPUT);
    ssize_t n=write(s->control,message+s->output_partial,sizeof(struct net_message)-s->output_partial);
    if(n>0){if((s->output_partial+=(size_t)n)==sizeof(struct net_message)){s->output_partial=0;s->output_head++;}continue;}
    if(n<0 && (errno==EAGAIN || errno==EWOULDBLOCK || errno==EINTR))return 0;
    return n<0?errno:ECONNRESET;
  }
  return 0;
}
static int net_read(struct net_session *s,int *busy){
  for(;;){
    ssize_t n=read(s->control,s->input+s->input_bytes,sizeof s->input-s->input_bytes);
    if(!n)return ECONNRESET;
    if(n<0)return errno==EAGAIN || errno==EWOULDBLOCK || errno==EINTR?0:errno;
    /* the peer heard: the longest it had been silent, and when (its liveness) */
    const uint64_t now=net_now();
    if(now-s->heard>atomic_load_explicit(&s->counts->silence_ns,memory_order_relaxed))
      atomic_store_explicit(&s->counts->silence_ns,now-s->heard,memory_order_relaxed);
    atomic_store_explicit(&s->counts->heard_ns,now,memory_order_relaxed);
    s->input_bytes+=(size_t)n;*busy=1;s->heard=now;
    size_t whole=s->input_bytes/sizeof(struct net_message)*sizeof(struct net_message);
    for(size_t at=0;at<whole && !s->failed;at+=sizeof(struct net_message)){
      struct net_message message;memcpy(&message,s->input+at,sizeof message);net_receive(s,&message);
    }
    memmove(s->input,s->input+whole,s->input_bytes-whole);s->input_bytes-=whole;
    if(s->failed)return s->failed;
  }
}

/* ---- the client tables ---- */
static int net_dead(uint64_t owner){pid_t pid=(pid_t)(uint32_t)owner;return !pid || (kill(pid,0) && errno==ESRCH);}
/* Every comm of this link whose client has exited is closed; its client slot is vacated, and its window
   pages once no comm of it remains (link 0's session, for the region's tables). */
static void net_reap(struct net_session *s){
  struct hdr *m=s->M;
  struct mesh_net_comm *comms=mesh_net_comms(m);
  for(uint32_t i=0;i<MESH_NET_COMMS;i++){
    uint32_t at=atomic_load_explicit(&comms[i].state,memory_order_acquire);
    if(at==MESH_NET_FREE || at==MESH_NET_CLAIMED || at==MESH_NET_CLOSING || comms[i].link!=s->index || !net_dead(comms[i].owner))continue;
    atomic_store_explicit(&comms[i].state,MESH_NET_CLOSING,memory_order_release);
  }
  struct mesh_net_memory *memory=mesh_net_memory(m);
  for(uint32_t i=0;!s->index && i<MESH_NET_MEMORY;i++){
    uint64_t owner=atomic_load_explicit(&memory[i].owner,memory_order_acquire);
    if(!owner || !net_dead(owner))continue;
    int held=0;
    for(uint32_t c=0;c<MESH_NET_COMMS && !held;c++){
      uint32_t at=atomic_load_explicit(&comms[c].state,memory_order_acquire);
      held=at!=MESH_NET_FREE && at!=MESH_NET_CLAIMED && comms[c].owner==owner;
    }
    if(held)continue;
    const uint32_t first=memory[i].first,pages=atomic_exchange_explicit(&memory[i].pages,0,memory_order_acq_rel);
    if(atomic_compare_exchange_strong_explicit(&memory[i].owner,&owner,0,memory_order_acq_rel,memory_order_relaxed) && pages)
      mesh_bits_clear(mesh_arena_bits(m),first,pages);
  }
  for(uint32_t i=0;!s->index && i<MESH_NET_CLIENTS;i++){
    uint64_t owner=atomic_load_explicit(&mesh_net_clients(m)[i].owner,memory_order_acquire);
    if(owner && net_dead(owner))atomic_compare_exchange_strong_explicit(&mesh_net_clients(m)[i].owner,&owner,0,memory_order_acq_rel,memory_order_relaxed);
  }
}
/* A comm its client closed: requests not yet on the wire fail, announcements waiting for it are
   refused, then CLOSE; what is on the wire completes, and the slot is vacated once the peer has
   closed too and nothing of it is on the wire. */
static void net_close(struct net_session *s,uint32_t index){
  struct mesh_net_comm *comm=mesh_net_comms(s->M)+index,*comms=mesh_net_comms(s->M);
  struct net_comm *state=net_state(comm,index);
  if(!state->close_sent){
    state->close_sent=1;
    for(uint32_t slot=0;slot<MESH_NET_REQUESTS;slot++){
      struct mesh_net_request *request=comm->requests+slot;
      uint32_t at=atomic_load_explicit(&request->state,memory_order_acquire);
      if(at==MESH_NET_POSTED || (at==MESH_NET_ACTIVE && (!state->phase[slot] || (request->op==MESH_NET_IRECV && state->phase[slot]==1)))){
        net_finish(request,ECONNRESET,0);state->phase[slot]=0;
      }
    }
    while(state->announce_head!=state->announce_tail){
      struct net_announce a=state->announced[state->announce_head%MESH_NET_REQUESTS];
      state->outcome[a.sequence%MESH_NET_REQUESTS]=(struct net_outcome){a.sequence,3,ECONNRESET};
      state->announce_head++;
      net_emit(s,(struct net_message){.kind=NET_CREDIT,.to=comm->peer,.to_generation=comm->peer_generation,.from=index,.from_generation=comm->generation,
        .sequence=a.sequence,.error=ECONNRESET});
    }
    /* a held isend its client never released: its receiver told (VOID), its chunks going from the discard buffer once
       it has answered (VOIDED) */
    for(uint32_t slot=0;slot<MESH_NET_REQUESTS;slot++)
      if(state->released[slot]==1 && atomic_load_explicit(&comm->requests[slot].state,memory_order_acquire)==MESH_NET_ACTIVE &&
         atomic_load_explicit(&comm->requests[slot].op,memory_order_acquire)==MESH_NET_HELD){
        net_emit(s,(struct net_message){.kind=NET_VOID,.to=comm->peer,.to_generation=comm->peer_generation,.from=index,.from_generation=comm->generation,
          .sequence=comm->requests[slot].sequence});
        state->released[slot]=3;
      }
    state->taken=state->matched=atomic_load_explicit(&comm->posted,memory_order_acquire);
    const int32_t why=net_dead(comm->owner)?ESRCH:0;
    if(comm->peer!=NET_NONE)
      net_emit(s,(struct net_message){.kind=NET_CLOSE,.to=comm->peer,.to_generation=comm->peer_generation,.from=index,.from_generation=comm->generation,.flags=1,
        .error=why});
    else if(state->connect_sent)
      net_emit(s,(struct net_message){.kind=NET_CLOSE,.to=NET_NONE,.from=index,.from_generation=comm->generation,.key=comm->key,.flags=1,.error=why});
    else state->peer_closed=1;
    for(uint32_t i=0;i<MESH_NET_COMMS;i++)
      if(atomic_load_explicit(&comms[i].state,memory_order_acquire)==MESH_NET_ACCEPTABLE && comms[i].link==s->index &&
         comms[i].listen==index && comms[i].listen_generation==comm->generation)
        net_unaccepted(comms+i,MESH_NET_CLOSING);
  }
  if(state->peer_closed && !state->outstanding)net_free(s,index);
}
static void net_scan(struct net_session *s){
  struct hdr *m=s->M;
  struct mesh_net_comm *comms=mesh_net_comms(m);
  for(uint32_t i=0;i<MESH_NET_COMMS && !s->failed;i++){
    struct mesh_net_comm *comm=comms+i;
    uint32_t at=atomic_load_explicit(&comm->state,memory_order_acquire);
    if(at==MESH_NET_FREE || at==MESH_NET_CLAIMED || comm->link!=s->index)continue;
    struct net_comm *state=net_state(comm,i);
    if(at==MESH_NET_CONNECTING && !state->connect_sent){
      state->connect_sent=1;
      net_publish_clients(s,0);
      net_emit(s,(struct net_message){.kind=NET_CONNECT,.to=NET_NONE,.from=i,.from_generation=comm->generation,.key=comm->key,.offset=comm->owner});
    }
    else if(at==MESH_NET_SEND || at==MESH_NET_RECV)net_take(s,i);
    else if(at==MESH_NET_CLOSING)net_close(s,i);
    else if(at==MESH_NET_LISTEN){
      for(uint32_t p=0;p<s->pending_count;){
        if(s->pending[p].key==comm->key && net_accept(s,i,s->pending[p].from,s->pending[p].generation,s->pending[p].owner))s->pending[p]=s->pending[--s->pending_count];
        else p++;
      }
    }
  }
}
/* A session lost for good (its peer paired again as another instance: what the peer held of it is gone; its
   peer left the mesh, or this bridge leaves it): every transfer is gone with its queue pair.  Connected comms fail (their clients
   close them), unaccepted receive comms and closing ones are vacated, listens stay and connects are
   sent again on the next session. */
static void net_lost(struct net_session *s,int32_t error){
  struct mesh_net_comm *comms=mesh_net_comms(s->M);
  for(uint32_t i=0;i<MESH_NET_COMMS;i++){
    struct mesh_net_comm *comm=comms+i;
    uint32_t at=atomic_load_explicit(&comm->state,memory_order_acquire);
    if(at==MESH_NET_FREE || at==MESH_NET_CLAIMED || comm->link!=s->index)continue;
    /* an unaccepted comm vacated (unless its client's accept takes it first: then failed as connected) */
    if(at==MESH_NET_ACCEPTABLE){if(net_unaccepted(comm,MESH_NET_CLAIMED)){net_vacate_comm(s,i,error);continue;}at=MESH_NET_RECV;}
    if(at==MESH_NET_CLOSING){net_vacate_comm(s,i,error);continue;}
    for(uint32_t slot=0;slot<MESH_NET_REQUESTS;slot++){
      uint32_t r=atomic_load_explicit(&comm->requests[slot].state,memory_order_acquire);
      if(r==MESH_NET_POSTED || r==MESH_NET_ACTIVE)net_finish(comm->requests+slot,error,0);
    }
    net_comms[i].live=0;
    if(at==MESH_NET_SEND || at==MESH_NET_RECV){
      /* failed unless its client closed it meanwhile (CLOSING: vacated) */
      atomic_store_explicit(&comm->error,error,memory_order_relaxed);
      if(!atomic_compare_exchange_strong_explicit(&comm->state,&at,MESH_NET_FAILED,memory_order_acq_rel,memory_order_acquire)){
        if(at==MESH_NET_CLOSING)net_vacate_comm(s,i,error);
        continue;
      }
      at=MESH_NET_FAILED;
    }
    if(at==MESH_NET_FAILED){
      struct net_comm *state=net_state(comm,i);
      state->close_sent=state->peer_closed=1;
      state->taken=state->matched=atomic_load_explicit(&comm->posted,memory_order_acquire);
    }
  }
  s->pending_count=0;
}
/* A session lost while this bridge runs (its link's interface went off, a failed completion, its control
   socket's end, a drop request): its transfers are kept for the next pairing (net_resume).  The completions
   still queued are taken (what landed), and each receive goes back to the bytes of it landed in order, its
   chunks after them to be posted and granted again; the grants its sender held go with the session. */
/* Each receive of a lost session back to the bytes of it landed in order (its least chunk start not landed, else
   its cursor), its chunks after them to be posted and granted again, and the RECVs forgotten: a session's state as
   its next pairing resumes it.  It reads only the keep, so a successor applies it to what a crashed bridge left. */
static void net_rewind(struct net_kept *k){
  for(uint32_t r=k->receive_head;r!=k->receive_tail;r++){
    struct net_transfer *t=k->receives+r%NET_RECEIVES;
    if(t->landed>=t->size)continue;
    uint64_t prefix=t->cursor;
    for(uint32_t c=k->chunk_head;c!=k->chunk_tail;c++){
      const struct net_chunk *chunk=k->chunks+c%NET_CHUNKS;
      if(chunk->transfer==r && !chunk->landed && chunk->at<prefix)prefix=chunk->at;
    }
    if(t->cursor>t->again)t->again=t->cursor;
    t->cursor=t->landed=prefix;
  }
  k->receive_post=k->receive_head;k->chunk_head=k->chunk_tail=0;
  k->suspended=1;
}
static void net_suspend(struct net_session *s){
  for(int pass=0;pass<64;pass++){int busy=0;net_complete(s,&busy,1);if(!busy)break;}
  net_rewind(s->k);s->period=0;
}
/* A session's peer left the mesh (its LEAVE), or this bridge leaves it: a membership change, nothing of the
   session resumes.  Its queue pairs are down (nothing of it lands after), the link is marked left (mesh.h
   MESH_LEFT: its clients find the peer departed once they see a request fail), then every transfer of it ends
   failed; the comms its clients close on it are vacated while it stays left (net_vacate). */
static void net_leave(struct net_session *s,int32_t error){
  s->left=1;
  atomic_store_explicit(&s->counts->phase,MESH_LEFT,memory_order_release);
  net_lost(s,error);
  s->k->suspended=0;s->k->receive_head=s->k->receive_post=s->k->receive_tail=0;
}
static void net_vacate(struct net_session *s){
  struct mesh_net_comm *comms=mesh_net_comms(s->M);
  for(uint32_t i=0;i<MESH_NET_COMMS;i++){
    uint32_t at=atomic_load_explicit(&comms[i].state,memory_order_acquire);
    if(comms[i].link!=s->index)continue;
    if(at==MESH_NET_CLOSING || (at==MESH_NET_ACCEPTABLE && net_unaccepted(comms+i,MESH_NET_CLAIMED)))net_vacate_comm(s,i,ENETDOWN);
  }
}
/* The session lost resumes on a pairing with the same peer instance.  Each receive in progress asks its
   sender to go on from the bytes of it landed (RESUME), posting and granting nothing more of it until the
   answer; each isend in progress is announced again, in its comm's order (its receiver's answer: nothing
   where it waits for its irecv or is in progress, LANDED where it landed); connects are sent again, a close
   whose answer was lost asks again.  Grants and announcements the session lost are made again so; data it
   lost is SENT again from where it landed. */
static void net_resume(struct net_session *s){
  struct mesh_net_comm *comms=mesh_net_comms(s->M);
  for(uint32_t r=s->k->receive_head;r!=s->k->receive_tail;r++){
    struct net_transfer *t=s->k->receives+r%NET_RECEIVES;
    if(t->landed>=t->size)continue;
    if(!net_comm_at(s,t->comm,t->generation)){t->cursor=t->landed=t->size;continue;}
    t->resume=1;
    net_emit(s,(struct net_message){.kind=NET_RESUME,.to=t->peer,.to_generation=t->peer_generation,.from=t->comm,.from_generation=t->generation,
      .sequence=t->sequence,.offset=t->landed});
  }
  s->pending_count=0;
  for(uint32_t i=0;i<MESH_NET_COMMS;i++){
    struct mesh_net_comm *comm=comms+i;
    uint32_t at=atomic_load_explicit(&comm->state,memory_order_acquire);
    if(at==MESH_NET_FREE || at==MESH_NET_CLAIMED || comm->link!=s->index)continue;
    struct net_comm *state=net_state(comm,i);
    if(at==MESH_NET_CONNECTING)state->connect_sent=0;
    else if(at==MESH_NET_CLOSING && state->close_sent && !state->peer_closed){
      if(comm->peer!=NET_NONE)
        net_emit(s,(struct net_message){.kind=NET_CLOSE,.to=comm->peer,.to_generation=comm->peer_generation,.from=i,.from_generation=comm->generation,.flags=1});
      else if(state->connect_sent)
        net_emit(s,(struct net_message){.kind=NET_CLOSE,.to=NET_NONE,.from=i,.from_generation=comm->generation,.key=comm->key,.flags=1});
    }
    if(at!=MESH_NET_SEND && at!=MESH_NET_CLOSING)continue;
    for(uint64_t q=state->taken>MESH_NET_REQUESTS?state->taken-MESH_NET_REQUESTS:0;q<state->taken;q++){
      const uint32_t slot=(uint32_t)(q%MESH_NET_REQUESTS);
      struct mesh_net_request *request=comm->requests+slot;
      const uint32_t op=atomic_load_explicit(&request->op,memory_order_acquire);
      if(request->sequence!=q || atomic_load_explicit(&request->state,memory_order_acquire)!=MESH_NET_ACTIVE || !state->phase[slot] ||
         (op!=MESH_NET_ISEND && op!=MESH_NET_HELD))continue;
      uint64_t extent=0,phase=0;
      if(request->size)net_geometry(s,request->offset,&extent,&phase);
      net_emit(s,(struct net_message){.kind=NET_RTS,.to=comm->peer,.to_generation=comm->peer_generation,.from=i,.from_generation=comm->generation,
        .flags=(op==MESH_NET_HELD)|2,.sequence=request->sequence,.size=request->size,.extent=extent,.phase=phase,.tag=request->tag});
      /* a held isend's void not yet answered, said again */
      if(state->released[slot]==3)
        net_emit(s,(struct net_message){.kind=NET_VOID,.to=comm->peer,.to_generation=comm->peer_generation,
          .from=i,.from_generation=comm->generation,.sequence=request->sequence});
    }
  }
  atomic_fetch_add_explicit(&s->counts->resumes,1,memory_order_relaxed);
}
static void net_reset(struct net_session *s){
  s->send_posted=s->send_retired=0;s->receive_outstanding=0;
  s->send_head=s->send_post=s->send_tail=0;
  if(!s->k->suspended)s->k->receive_head=s->k->receive_post=s->k->receive_tail=0;
  s->k->chunk_head=s->k->chunk_tail=0;s->send_queue=s->receive_queue=0;memset(s->queue_frames,0,sizeof s->queue_frames);
  s->output_head=s->output_tail=0;s->output_partial=s->input_bytes=0;s->failed=0;s->send_blocked=s->receive_blocked=0;
  s->clients_known=0;
}
/* The session's configuration, between RTR and RTS: a priming RECV posted on each queue pair into the
   device's discard buffer, then each end's queue capacities exchanged, so that neither end SENDs before
   the other's first RECVs are posted; the chunk both cut by, at most the discard buffer.  The queue pairs
   share their completion queues: the RECVs posted and not taken are at most its entries less one. */
static int net_configure(void *argument,int socket){
  struct net_session *s=argument;
  s->send_capacity=(uint32_t)s->provider.sent->cqe;s->receive_capacity=UINT32_MAX;
  for(int q=0;q<NET_QUEUES;q++){
    s->send_capacity=MIN(s->send_capacity,s->provider.queues[q].send_capacity);
    s->receive_capacity=MIN(s->receive_capacity,s->provider.queues[q].receive_capacity);
  }
  /* the RECVs posted at once: the completion queue's entries less one, within the keep's ring (NET_CHUNKS, a power
     of two, so its indices wrap with their counters) */
  s->receive_requests=MIN((uint32_t)s->provider.completion->cqe-1,NET_CHUNKS-1);
  if(!s->sends)s->sends=calloc(NET_SENDS,sizeof *s->sends);
  if(!s->output)s->output=calloc(NET_OUTPUT,sizeof *s->output);
  if(!s->sends || !s->output){errno=ENOMEM;return -1;}
  struct ibv_sge span=net_span(s,NET_DISCARD,0,NET_PRIME);
  for(uint32_t q=0;q<NET_QUEUES;q++){
    struct ibv_recv_wr prime={.wr_id=s->k->chunk_tail,.sg_list=&span,.num_sge=1},*bad;
    s->k->chunks[s->k->chunk_tail%NET_CHUNKS]=(struct net_chunk){NET_NONE,1,q,0,NET_PRIME,0};
    int error=ibv_post_recv(s->provider.queues[q].pair,&prime,&bad);
    if(error){errno=error<0?-error:error;return -1;}
    s->k->chunk_tail++;
    s->queue_frames[q]=1;s->receive_outstanding++;
  }
  /* the capacities, and each bridge's instance (a pairing with the instance lost resumes: net_resume) */
  uint32_t mine[4]={s->send_capacity,s->receive_capacity,(uint32_t)net_keep->instance,(uint32_t)(net_keep->instance>>32)},peer[4];
  if(exchange(socket,mine,peer,sizeof mine,sizeof peer))return -1;
  s->instance=peer[2]|(uint64_t)peer[3]<<32;
  uint32_t least=MIN(MIN(mine[0],mine[1]),MIN(peer[0],peer[1])),frames=MIN(NET_CHUNK_FRAMES,least/2);
  s->chunk=(uint64_t)(frames?frames:1)*4096;
  atomic_store_explicit(&s->counts->chunk_frames,frames?frames:1,memory_order_relaxed);
  atomic_store_explicit(&s->counts->wire_regions,s->provider.device->region_count,memory_order_relaxed);
  return 0;
}
/* The session's progress: completions, control messages, the client tables when a client rings (or
   every millisecond), chunks posted while their queue has frames; it spins while anything is in flight
   or was lately, and otherwise waits for a client's doorbell a short while at a time.  Its first SEND on
   each queue pair fills the peer's priming RECV there. */
/* Whether the kernel said the link's interface went off (or away) since this was last read: the session ends
   on it, an observed link down, before a SEND into the lost link could meet a RECV it was not granted to
   (UC queue pairs report nothing). */
static int net_link_off(struct net_session *s){
  union mesh_network_event event;
  int off=0;
  while(s->events>=0){
    ssize_t length=recv(s->events,event.bytes,sizeof event,0);
    if(length<0)break;
    if((size_t)length<KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data))continue;
    if(event.header.event_code!=KEV_DL_LINK_OFF && event.header.event_code!=KEV_DL_IF_DETACHED)continue;
    const struct net_event_data *interface=(const void *)(event.bytes+KEV_MSG_HEADER_SIZE);
    char device[sizeof interface->if_name+16];
    snprintf(device,sizeof device,"rdma_%.*s%u",(int)sizeof interface->if_name,interface->if_name,interface->if_unit);
    if(!strcmp(device,s->provider.device->name)){
      say("session link %u: link unavailable: %s event=%u\n",s->index,device,event.header.event_code);
      off=1;
    }
  }
  return off;
}
static void net_serve(struct net_session *s){
  for(uint32_t q=0;q<NET_QUEUES;q++)s->sends[s->send_tail++%NET_SENDS]=(struct net_send){.comm=NET_NONE,.mr=NET_DISCARD,.queue=q,.length=NET_PRIME};
  uint64_t last=net_now(),looked=last;
  s->heard=s->said=last;
  while(!stop && !s->failed){
    int busy=0,error=net_complete(s,&busy,0);
    if(!error)error=net_read(s,&busy);
    if(error){s->failed=error;break;}
    uint64_t now=net_now(),bell=atomic_load_explicit(&s->counts->doorbell,memory_order_acquire);
    const uint64_t drops=atomic_load_explicit(&net_drops,memory_order_relaxed);
    if(drops!=s->drops){s->drops=drops;say("session link %u: dropped on request\n",s->index);s->failed=ECONNABORTED;break;}
    if(now-looked>1000000){looked=now;if(net_link_off(s)){s->failed=ENETDOWN;break;}}
    if(now-s->said>NET_HEARTBEAT_NS)net_emit(s,(struct net_message){.kind=NET_HEARTBEAT});
    uint64_t news=atomic_load_explicit(&link_news,memory_order_acquire);
    if(news!=s->news){s->news=news;net_publish(s);}
    if(estimating)estimate_ripen(s);
    uint64_t estimated=atomic_load_explicit(&estimate_news,memory_order_acquire);
    if(estimates && estimated!=s->estimate_seen){s->estimate_seen=estimated;net_publish_estimates(s);}
    if(bell!=s->bell || now-s->scanned>1000000){busy|=bell!=s->bell;s->bell=bell;s->scanned=now;net_publish_clients(s,0);net_scan(s);}
    if(now-s->reaped>500000000){s->reaped=now;net_reap(s);}
    /* what the scan said (a release before its bytes) written before the SENDs are posted */
    if(!s->failed && (error=net_flush(s)))s->failed=error;
    if(!s->failed && (error=net_post(s)))s->failed=error;
    if(!s->failed && (error=net_flush(s)))s->failed=error;
    if(busy)last=now;
    int flight=s->send_head!=s->send_post || (s->send_post!=s->send_tail && !s->send_held) || s->k->chunk_head!=s->k->chunk_tail ||
      s->output_head!=s->output_tail;
    if(!busy && !flight && now-last>2000000)
      os_sync_wait_on_address_with_timeout(&s->counts->doorbell,bell,sizeof bell,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
        OS_CLOCK_MACH_ABSOLUTE_TIME,now-last>1000000000?1000000:50000);
  }
}
static void *net_session_run(void *argument){
  struct net_session *s=argument;struct hdr *m=s->M;
  pthread_setname_np("mesh.net.session");
  s->control=-1;
  /* the kernel's link events: an interface off ends the session (net_link_off) */
  s->events=socket(PF_SYSTEM,SOCK_RAW,SYSPROTO_EVENT);
  struct kev_request filter={KEV_VENDOR_APPLE,KEV_NETWORK_CLASS,KEV_DL_SUBCLASS};
  if(s->events>=0 && (fcntl(s->events,F_SETFL,O_NONBLOCK)<0 || ioctl(s->events,SIOCSKEVFILT,&filter)<0)){close(s->events);s->events=-1;}
  if(s->events<0)say("session link %u: no link events (%s): a link lost is seen only as a failed completion or the control socket's end\n",
                         s->index,strerror(errno));
  while(!stop){
    if(!s->left)atomic_store_explicit(&s->counts->phase,MESH_PAIRING,memory_order_release);
    net_reset(s);
    net_link_off(s);  /* the events before this pairing were the last session's */
    int f=verbs_up(&s->provider,m,NET_QUEUES,net_configure,s);
    if(f<0){
      atomic_store_explicit(&s->counts->code,errno?errno:EIO,memory_order_relaxed);
      while(!down_pair(&s->provider))poll(NULL,0,100);
      if(s->provider.listener>=0){close(s->provider.listener);s->provider.listener=-1;}
      if(s->left)net_vacate(s);
      net_nap(250000000);
      continue;
    }
    s->control=f;
    __atomic_store_n(&mesh_links(m)[s->index].bandwidth,s->provider.bandwidth,__ATOMIC_RELAXED);
    atomic_store_explicit(&s->counts->code,0,memory_order_relaxed);
    atomic_fetch_add_explicit(&s->counts->sessions,1,memory_order_relaxed);
    /* the pairing's identity, then this node's clients, the first the peer hears */
    s->pairing=net_pairing(net_keep->instance,s->instance);
    atomic_store_explicit(&s->counts->pairing,s->pairing,memory_order_release);
    net_publish_clients(s,1);
    /* a peer of another instance: its node's reports start again (its sequence may be below its predecessor's) */
    if(s->k->paired && s->instance!=s->k->paired && link_table)mesh_link_table_forget(link_table,s->provider.peer);
    /* the session lost resumes with the same peer instance; another's pairing means the peer's side of it is gone */
    if(s->k->suspended){
      s->k->suspended=0;
      if(s->instance==s->k->paired){say("session link %u: resumed\n",s->index);net_resume(s);}
      else{
        say("session link %u: its peer paired as another instance: the lost session's transfers fail\n",s->index);
        net_lost(s,ESTALE);s->k->receive_head=s->k->receive_post=s->k->receive_tail=0;
      }
    }
    s->k->paired=s->instance;s->drops=atomic_load_explicit(&net_drops,memory_order_relaxed);
    s->left=s->peer_left=0;
    atomic_store_explicit(&s->counts->phase,MESH_PAIRED,memory_order_release);
    /* the new peer is sent every report this bridge holds, then each newer one; every estimate too, and
       this session's estimator starts from the link's stated cost */
    if(link_table)memset(s->sent,0,link_table->nodes*sizeof *s->sent);
    s->news=atomic_load_explicit(&link_news,memory_order_acquire)-1;
    if(estimates){
      memset(s->estimates_sent,0,(size_t)link_table->nodes*link_table->nodes*sizeof *s->estimates_sent);
      s->estimate_seen=atomic_load_explicit(&estimate_news,memory_order_acquire)-1;
      if(estimating){estimate_start(s);s->lagged_head=s->lagged_tail=0;}
    }
    if(link_table && mesh_link_table_observe(link_table,s->provider.peer,1)>0)net_links_moved();
    net_serve(s);
    if(crashing){close(f);s->control=-1;while(!down_pair(&s->provider))poll(NULL,0,100);break;}
    /* a bridge that leaves says so last: nothing of the session resumes, so what is queued is dropped (but a message
       partly written, finished) and LEAVE is written whole, the socket blocking until it is or ends */
    if(leaving){
      s->output_tail=s->output_head+(s->output_partial?1:0);
      net_emit(s,(struct net_message){.kind=NET_LEAVE});
      const int flags=fcntl(f,F_GETFL);
      if(flags>=0)fcntl(f,F_SETFL,flags&~O_NONBLOCK);
      while(s->output_head!=s->output_tail && !net_flush(s)){}
    }
    int32_t error=s->peer_left?EHOSTDOWN:leaving?ENETDOWN:stop?ECANCELED:s->failed?s->failed:EIO;
    if(s->peer_left || leaving)say("session link %u: %s\n",s->index,leaving?"this bridge leaves the mesh":"its peer's bridge left the mesh");
    else if(!stop)say("session down: link %u: %s\n",s->index,strerror(error));
    if(s->strays)say("session link %u: %llu grants no request held, filled from the discard buffer\n",s->index,(unsigned long long)s->strays);
    atomic_store_explicit(&s->counts->code,error,memory_order_relaxed);
    atomic_store_explicit(&s->counts->phase,MESH_STOPPED,memory_order_release);
    if(link_table && mesh_link_table_observe(link_table,s->provider.peer,0)>0)net_links_moved();
    /* lost, or this bridge stopped: kept for the next pairing, by this bridge or the next on the region */
    if(!s->peer_left && !leaving)net_suspend(s);
    close(f);s->control=-1;
    while(!down_pair(&s->provider))poll(NULL,0,100);
    if(s->peer_left || leaving)net_leave(s,error);
    /* a lost session pairs again after a pause, not at once: failed completions, torn-down queue pairs
       and immediate re-pairing preceded the Sep 5 kernel panic (design/RDMA-KERNEL-RECOVERY.md) */
    if(!stop)net_nap(UINT64_C(3000000000));
  }
  if(s->events>=0)close(s->events);
  if(crashing){free(s->sends);free(s->output);return NULL;}
  if(leaving && !s->left)net_leave(s,ENETDOWN);
  free(s->sends);free(s->output);
  atomic_store_explicit(&s->counts->phase,s->left?MESH_LEFT:MESH_STOPPED,memory_order_release);
  return NULL;
}

/* Whether region `m` is the one this configuration makes (`g` its layout, node `me`, its links' peers and devices). */
static int region_same(struct hdr *m,const struct hdr *g,int me,const struct net_session *sessions,uint32_t count){
  if(m->magic!=MESH_MAGIC || m->version!=MESH_VERSION || m->node!=(uint32_t)me || m->pgsz!=g->pgsz || m->block!=g->block ||
     m->links!=g->links || m->wire_pages!=g->wire_pages || m->length!=g->length || m->link_off!=g->link_off || m->net_off!=g->net_off ||
     m->stats_off!=g->stats_off || m->data_off!=g->data_off)return 0;
  for(uint32_t i=0;i<count;i++)
    if(mesh_links(m)[i].peer!=sessions[i].provider.peer ||
       strncmp(mesh_links(m)[i].device,sessions[i].provider.device->name,sizeof mesh_links(m)[i].device))return 0;
  return 1;
}
/* Whether a client is attached to the region: a communicator client's process, or one owning a comm, alive. */
static int net_attached(struct hdr *m){
  for(uint32_t i=0;i<MESH_NET_CLIENTS;i++){const uint64_t owner=atomic_load(&mesh_net_clients(m)[i].owner);if(owner && !net_dead(owner))return 1;}
  for(uint32_t i=0;i<MESH_NET_COMMS;i++){
    const uint32_t at=atomic_load(&mesh_net_comms(m)[i].state);
    if(at!=MESH_NET_FREE && at!=MESH_NET_CLAIMED && !net_dead(mesh_net_comms(m)[i].owner))return 1;
  }
  return 0;
}
/* A region let go with no client of it alive (a bridge's exit that keeps nothing, --release, a start that makes it
   afresh): every request still outstanding on it ended failed and its completion word written (2), as no bridge ends
   it after.  A GPU's work outlives its process (no watchdog ends a kernel that polls), so a kernel of an exited client
   may still wait on that word in the region's pages; it ends once it reads it, and the pages are freed with it. */
static void region_end_requests(struct hdr *m){
  const uint64_t wire=mesh_wire_bytes(m);
  struct mesh_net_comm *comms=mesh_net_comms(m);
  for(uint32_t i=0;i<MESH_NET_COMMS;i++){
    const uint32_t at=atomic_load_explicit(&comms[i].state,memory_order_acquire);
    if(at==MESH_NET_FREE || at==MESH_NET_CLAIMED)continue;
    for(uint32_t slot=0;slot<MESH_NET_REQUESTS;slot++){
      struct mesh_net_request *request=comms[i].requests+slot;
      const uint32_t r=atomic_load_explicit(&request->state,memory_order_acquire);
      if(r!=MESH_NET_POSTED && r!=MESH_NET_ACTIVE)continue;
      /* a completion word outside the window is never stored (net_post) */
      const uint64_t word=request->completion;
      if(word && word-1<=wire-8 && !((word-1)%8))
        atomic_store_explicit((_Atomic uint64_t *)((char *)m+m->data_off+word-1),2,memory_order_release);
      atomic_store_explicit(&request->error,ESHUTDOWN,memory_order_relaxed);
      atomic_store_explicit(&request->state,MESH_NET_ERROR,memory_order_release);
    }
  }
}
/* --release: region `name`, kept by the last bridge on it for its clients, released where no bridge runs on it
   and no client of it is alive (net_attached: a process exists or it does not; nothing here waits or counts
   time): the region, its keep and its link table are unlinked, and their pages freed once nothing maps them.  A
   starting bridge applies the same rule (it takes a kept region only while a client of it lives). */
static int region_release(const char *name){
  int fd=shm_open(name,O_RDWR,MESH_MODE);
  if(fd<0){say("%s: no region\n",name);return 0;}
  struct stat info;struct hdr *m=MAP_FAILED;
  if(!fstat(fd,&info) && (uint64_t)info.st_size>=sizeof *m)m=mmap(NULL,(size_t)info.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  close(fd);
  const char *kept=m==MAP_FAILED || m->magic!=MESH_MAGIC || m->version!=MESH_VERSION || m->length>(uint64_t)info.st_size?"not a region of this version":
                   !net_dead(atomic_load(&m->bridge_pid))?"a bridge of it is running":net_attached(m)?"a client of it is alive":NULL;
  /* its pages in memory, which the release frees */
  uint64_t resident=0;const size_t page=(size_t)getpagesize(),pages=(size_t)info.st_size/page;char *in=m!=MAP_FAILED?malloc(pages?pages:1):NULL;
  if(in && !mincore((void *)m,pages*page,in))for(size_t i=0;i<pages;i++)resident+=(in[i]&MINCORE_INCORE)!=0;
  free(in);
  if(m!=MAP_FAILED && !kept)region_end_requests(m);
  if(m!=MAP_FAILED)munmap(m,(size_t)info.st_size);
  if(kept){say("%s kept (%llu bytes resident): %s\n",name,(unsigned long long)(resident*page),kept);return 1;}
  char other[80];
  shm_unlink(name);
  snprintf(other,sizeof other,"%s.keep",name);shm_unlink(other);
  snprintf(other,sizeof other,"%s.links",name);shm_unlink(other);
  say("%s released (%llu bytes, %llu resident): no bridge runs on it and no client of it is alive\n",name,(unsigned long long)info.st_size,
      (unsigned long long)(resident*page));
  return 0;
}

/* --renew: region `name`'s keep marked so that the next bridge on it makes region, keep and link table afresh, as after
   the node restarted (the bridge running on it goes on until it is stopped). */
static int region_renew(const char *name){
  char keep[80];snprintf(keep,sizeof keep,"%s.keep",name);
  int fd=shm_open(keep,O_RDWR,0600);
  if(fd<0){say("%s: no keep\n",name);return 1;}
  struct net_keep *k=mmap(NULL,sizeof *k,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  close(fd);
  if(k==MAP_FAILED || k->magic!=NET_KEEP_MAGIC || k->version!=NET_KEEP_VERSION){say("%s: not a keep of this version\n",name);if(k!=MAP_FAILED)munmap(k,sizeof *k);return 1;}
  atomic_store(&k->renew,1);
  say("%s: the next bridge on it makes it afresh (instance %016llx renewed)\n",name,(unsigned long long)k->instance);
  munmap(k,sizeof *k);
  return 0;
}
/* A successor's start on a kept region: every link's session suspended, its receives back to the bytes of each landed
   in order (net_rewind: the next pairing resumes it with the same peer instance, or fails it with another), and each
   comm's bridge side completed where its last bridge ended between two of its writes: a request made ACTIVE whose
   `taken` had not moved (net_take), an announcement consumed whose receive was kept (net_match), and the requests on
   the wire counted again from their phases. */
static void net_normalize(struct hdr *m,uint32_t links){
  for(uint32_t l=0;l<links;l++)net_rewind(net_keep->kept+l);
  struct mesh_net_comm *comms=mesh_net_comms(m);
  for(uint32_t i=0;i<MESH_NET_COMMS;i++){
    struct mesh_net_comm *comm=comms+i;struct net_comm *state=net_comms+i;
    const uint32_t at=atomic_load_explicit(&comm->state,memory_order_acquire);
    if(at==MESH_NET_FREE || at==MESH_NET_CLAIMED || !state->live || state->generation!=comm->generation || comm->link>=links)continue;
    const uint64_t posted=atomic_load_explicit(&comm->posted,memory_order_acquire);
    for(;state->taken<posted;state->taken++){
      const struct mesh_net_request *r=comm->requests+state->taken%MESH_NET_REQUESTS;
      if(r->sequence!=state->taken || atomic_load_explicit(&r->state,memory_order_acquire)==MESH_NET_POSTED)break;
    }
    const struct net_kept *k=net_keep->kept+comm->link;
    while(at==MESH_NET_RECV && state->announce_head!=state->announce_tail){
      const struct net_announce *a=state->announced+state->announce_head%MESH_NET_REQUESTS;
      int kept=0;
      for(uint32_t r=k->receive_head;r!=k->receive_tail && !kept;r++){
        const struct net_transfer *t=k->receives+r%NET_RECEIVES;
        kept=t->comm==i && t->generation==comm->generation && t->sequence==a->sequence;
      }
      if(!kept)break;
      state->outcome[a->sequence%MESH_NET_REQUESTS]=(struct net_outcome){a->sequence,1,0};
      state->phase[state->matched%MESH_NET_REQUESTS]=2;
      state->announce_head++;state->matched++;
    }
    state->outstanding=0;
    for(uint32_t slot=0;slot<MESH_NET_REQUESTS;slot++){
      const struct mesh_net_request *r=comm->requests+slot;
      if(atomic_load_explicit(&r->state,memory_order_acquire)!=MESH_NET_ACTIVE || !state->phase[slot])continue;
      const uint32_t op=atomic_load_explicit(&r->op,memory_order_acquire);
      state->outstanding+=op!=MESH_NET_IRECV || state->phase[slot]==2;
    }
  }
}

/* design/algorithm-sources.md#programcopy */
int main(int argc,char **argv){
  const char *name=MESH_NAME;int me=0,layout=0,release=0,renew=0;double pct=0;uint32_t table_nodes=0,lag=10;
  uint64_t arena_pages=0,block_pages=0,window_pages=0;
  uint32_t link_count=0,device_count=0;
  struct mesh_device *devices=calloc((size_t)argc,sizeof *devices);
  struct net_session *sessions=calloc((size_t)argc,sizeof *sessions);
  if(!devices || !sessions)die("bridge configuration allocation");
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-I") && i+1<argc)me=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-M") && i+1<argc)pct=atof(argv[++i]);
    /* design/prepared-machine.md#M09 */
    /* -A addressable arena pages, -B pages per block, -W registered window pages.  None of the three is derived
       from another; -W defaults to the whole arena. */
    else if((!strcmp(argv[i],"-A") || !strcmp(argv[i],"-B") || !strcmp(argv[i],"-W")) && i+1<argc){
      char kind=argv[i][1],*end;uint64_t pages=strtoull(argv[++i],&end,10);
      if(*end || !pages || pages>INT32_MAX)die("configured page count");
      if(kind=='A')arena_pages=pages;else if(kind=='B')block_pages=pages;else window_pages=pages;
    }
    /* -N the link table's nodes (the configured link map's: its node ids below it); without it, the
       largest node this bridge's configuration names (itself, its links' peers) and one */
    else if(!strcmp(argv[i],"-N") && i+1<argc){char *end;unsigned long n=strtoul(argv[++i],&end,10);if(*end || !n || n>NET_LINK_NODES)die("link table nodes (-N, at most 320)");table_nodes=(uint32_t)n;}
    /* -E: estimate the beta of each link into this node from its receives (the table's third writer) */
    else if(!strcmp(argv[i],"-E"))estimating=1;
    /* -K: the evaluations the transport's statistics lag behind their readers (mesh.h mesh_stats_read; 10) */
    else if(!strcmp(argv[i],"-K") && i+1<argc){char *end;unsigned long k=strtoul(argv[++i],&end,10);if(*end || k>MESH_STATS/2)die("statistics lag (-K)");lag=(uint32_t)k;}
    else if(!strcmp(argv[i],"--layout"))layout=1;
    else if(!strcmp(argv[i],"--release"))release=1;
    else if(!strcmp(argv[i],"--renew"))renew=1;
    else if(!strcmp(argv[i],"-s") && i+1<argc)name=argv[++i];
    else if(!strcmp(argv[i],"--link") && i+1<argc){
      char *fields=strdup(argv[++i]);
      if(!fields)die("link configuration allocation");
      char *device=strsep(&fields,","),*peer=strsep(&fields,","),*local=strsep(&fields,","),*remote=strsep(&fields,","),*service=strsep(&fields,","),*net=strsep(&fields,",");
      if(!device || !*device || !peer || !*peer || !local || !*local || !remote || !*remote || fields)die("link requires device,peer,local-address,remote-address[,service[,communicator-service]]");
      uint32_t d=0;
      while(d<device_count && strcmp(devices[d].name,device))d++;
      if(d==device_count){devices[d].name=device;pthread_mutex_init(&devices[d].setup,NULL);device_count++;}
      /* the link's communicator session: its own port, the link's service + NET_PORT_OFFSET unless given */
      struct net_session *session=&sessions[link_count];
      if(net && *net)snprintf(session->service,sizeof session->service,"%s",net);
      else snprintf(session->service,sizeof session->service,"%ld",strtol(service?service:MESH_PORT,NULL,10)+NET_PORT_OFFSET);
      session->index=link_count++;
      session->provider=(struct mesh_verbs){.device=&devices[d],.peer=(uint32_t)strtoul(peer,NULL,10),.local_address=local,.remote_address=remote,
        .service=session->service,.listener=-1};
    }
    else die("unknown bridge option");
  }
  if(release)return region_release(name);
  if(renew)return region_renew(name);
  if(me<0 || !isfinite(pct) || pct<0 || pct>100 || !arena_pages || !block_pages || arena_pages<block_pages)die("bridge geometry");
  for(uint32_t d=0;d<device_count;d++){
    uint32_t pairs=0;
    for(uint32_t i=0;i<link_count;i++)if(sessions[i].provider.device==devices+d)pairs+=NET_QUEUES;
    if(pairs>MESH_DEVICE_QPS)die("a device's links' queue pairs (each session's) past its 10");
  }
  const uint32_t pg=(uint32_t)getpagesize();
  struct hdr geometry={0};
  uint64_t length=mesh_layout(&geometry,(struct mesh_geometry){.pgsz=pg,.block=(uint32_t)block_pages,
    .pages=(uint32_t)arena_pages,.links=link_count,.wire_pages=(uint32_t)window_pages});
  uint64_t ram=0;size_t rl=sizeof ram;sysctlbyname("hw.memsize",&ram,&rl,NULL,0);
  if(pct && length>(uint64_t)(pct/100*(double)ram))die("configured graph exceeds page capacity");
  if(layout){printf("%llu\n",(unsigned long long)length);return 0;}
  atexit(down);struct sigaction sa={0};sa.sa_handler=stop_bridge;
  sigaction(SIGINT,&sa,NULL);sigaction(SIGTERM,&sa,NULL);sigaction(SIGHUP,&sa,NULL);signal(SIGPIPE,SIG_IGN);
  struct sigaction drop={0};drop.sa_handler=net_drop;drop.sa_flags=SA_RESTART;sigaction(SIGUSR1,&drop,NULL);
  struct sigaction leave={0};leave.sa_handler=leave_bridge;sigaction(SIGUSR2,&leave,NULL);
  struct sigaction crash={0};crash.sa_handler=crash_bridge;sigaction(SIGQUIT,&crash,NULL);
  if(!table_nodes){
    table_nodes=(uint32_t)me+1;
    for(uint32_t i=0;i<link_count;i++)if(sessions[i].provider.peer>=table_nodes)table_nodes=sessions[i].provider.peer+1;
  }
  for(uint32_t i=0;i<link_count;i++)if(sessions[i].provider.peer>=table_nodes)die("a link's peer past the link table's nodes (-N)");
  if((uint32_t)me>=table_nodes || table_nodes>NET_LINK_NODES)die("link table");
  /* The region its last bridge left with clients attached, where it is this configuration's and a client of it is
     still alive: taken as it is, with its keep and link table, so its clients' comms and transfers resume
     (net_keep); a live bridge's is refused.  Otherwise all three are made afresh: a kept region none of whose
     clients' processes remains is released (region_release's rule), its pages freed with its last mapping. */
  struct hdr *m=MAP_FAILED;int fd=shm_open(name,O_RDWR,MESH_MODE),kept=0;
  if(fd>=0){
    struct stat info;
    if(!fstat(fd,&info) && (uint64_t)info.st_size>=length)m=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    /* the region claimed from its last bridge, dead (one bridge a region) */
    uint64_t last=m!=MAP_FAILED && m->magic==MESH_MAGIC?atomic_load(&m->bridge_pid):0;
    if(m!=MAP_FAILED && m->magic==MESH_MAGIC && ((last && !net_dead(last)) || !atomic_compare_exchange_strong(&m->bridge_pid,&last,(uint64_t)getpid())))
      die("a bridge of this region is running");
    const int same=m!=MAP_FAILED && region_same(m,&geometry,me,sessions,link_count) && m->stats_lag==lag,attached=same && net_attached(m);
    if(same && !attached)say("bridge node %d: no client of the region its last bridge left is alive: region, keep and link table made afresh\n",me);
    if(m!=MAP_FAILED && m->magic==MESH_MAGIC && m->version==MESH_VERSION && m->length<=length && !net_attached(m))region_end_requests(m);
    if(attached && (net_keep=net_keep_open(name,link_count,(uint32_t)me,length,0)) && !mesh_link_table_open(name,0,0,&link_table)){
      kept=link_table->nodes==table_nodes && link_table->node==(uint32_t)me;
      if(!kept){mesh_link_table_close(link_table);link_table=NULL;}
      else if(atomic_load(&net_keep->renew)){
        say("bridge node %d: the region its last bridge left is renewed: region, keep and link table made afresh, as after a restart\n",me);
        mesh_link_table_close(link_table);link_table=NULL;kept=0;
      }
    }
    if(!kept){
      if(net_keep){munmap(net_keep,net_keep_bytes(link_count));net_keep=NULL;}
      if(m!=MAP_FAILED)munmap(m,length);
      m=MAP_FAILED;close(fd);fd=-1;
    }
  }
  if(!kept){
    shm_unlink(name);fd=shm_open(name,O_CREAT|O_RDWR,MESH_MODE);if(fd<0)die("shm");
    if(ftruncate(fd,(off_t)length))die("ftruncate");fchmod(fd,MESH_MODE);
    m=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if(m==MAP_FAILED)die("mmap");
    if(!(net_keep=net_keep_open(name,link_count,(uint32_t)me,length,1)))die("the region's keep");
    if(mesh_link_table_open(name,table_nodes,(uint32_t)me,&link_table))die("link table");
  }
  shm=name;keeping=kept;
  net_comms=net_keep->comms;
  snprintf(link_table_region,sizeof link_table_region,"%s",name);
  atexit(link_table_down);
  /* this bridge's reports' sequence starts past any an earlier bridge of its node sent: its start in ms, and past the
     kept table's (a clock stepped back); a peer meeting a new instance of this node takes its reports afresh */
  struct timespec started;clock_gettime(CLOCK_REALTIME,&started);
  const uint64_t base=((uint64_t)started.tv_sec*1000+(uint64_t)started.tv_nsec/1000000)<<20,held=atomic_load(&mesh_link_reported(link_table)[me]);
  atomic_store(&mesh_link_reported(link_table)[me],base>held?base:held+1);
  /* the estimates this bridge holds (every bridge applies and passes on its peers'), its own (-E) sequenced
     from its start as its reports are */
  if(!(estimates=calloc((size_t)table_nodes*table_nodes,sizeof *estimates)))die("link estimate allocation");
  estimate_node=(uint32_t)me;estimate_sequence=((uint64_t)started.tv_sec*1000+(uint64_t)started.tv_nsec/1000000)<<20;
  if(!kept){
    *m=geometry;m->node=(uint32_t)me;m->version=MESH_VERSION;m->stats_lag=lag;atomic_store(&m->instance,net_keep->instance);
  }
  struct mesh_wire wire={0};
  if(wire_map(&wire,m,fd))die("transport page aliases");
  close(fd);
  for(uint32_t i=0;i<link_count;i++){
    mesh_links(m)[i].peer=sessions[i].provider.peer;
    snprintf(mesh_links(m)[i].device,sizeof mesh_links(m)[i].device,"%s",sessions[i].provider.device->name);
  }
  net_region=m;
  if(!kept)atomic_store(&m->bridge_pid,(uint64_t)getpid());
  __sync_synchronize();m->magic=MESH_MAGIC;
  say("bridge node %d: %u links, arena %llu pages, window %u pages\n",me,link_count,(unsigned long long)mesh_arena_pages(m),m->wire_pages);
  if(kept)say("bridge node %d: the region its last bridge left taken as it is (instance %016llx)\n",me,(unsigned long long)net_keep->instance);
  /* what the last bridge left, however it ended (stopped, or crashed: its keep is what it wrote in place) */
  if(kept)net_normalize(m,link_count);
  for(uint32_t i=0;i<link_count;i++){
    sessions[i].M=m;sessions[i].counts=mesh_net_links(m)+i;sessions[i].provider.wire=&wire;sessions[i].k=net_keep->kept+i;
    struct net_kept *k=sessions[i].k;
    if(kept)say("session link %u: %u receives kept, suspended (resumed by the next pairing with instance %016llx)\n",i,k->receive_tail-k->receive_head,
                (unsigned long long)k->paired);
    if(!(sessions[i].sent=calloc(table_nodes,sizeof *sessions[i].sent)))die("link report allocation");
    if(!(sessions[i].estimates_sent=calloc((size_t)table_nodes*table_nodes,sizeof *sessions[i].estimates_sent)))die("link estimate allocation");
  }
  net_sessions=sessions;net_session_count=link_count;
  for(uint32_t i=0;i<link_count;i++){
    struct net_session *session=&sessions[i];
    say("session link %u: communicators on port %s\n",i,session->service);
    int error=pthread_create(&session->thread,NULL,net_session_run,session);
    if(error)say("session link %u: %s\n",i,strerror(error));
    else session->started=1;
  }
  /* the sessions serve until this bridge stops */
  while(!stop){
    const uint64_t seen=atomic_load_explicit(&stopping,memory_order_acquire);
    if(!stop)os_sync_wait_on_address(&stopping,seen,sizeof stopping,OS_SYNC_WAIT_ON_ADDRESS_NONE);
  }
  for(uint32_t i=0;i<link_count;i++)if(sessions[i].started)pthread_join(sessions[i].thread,NULL);
  /* a crash's end (SIGQUIT): the device torn down, nothing else, and no exit handler run (the region stays) */
  if(atomic_load(&crashing)){
    for(uint32_t i=0;i<device_count;i++)down_device(&devices[i]);
    say("bridge node %d: ended as a crash (SIGQUIT): its keep as it stood\n",me);
    _exit(EX_SOFTWARE);
  }
  /* the region, its keep and its link table kept where clients are attached, for the next bridge on it (one that left
     the mesh too: its clients rejoin when a bridge returns on the region); else removed at exit */
  keeping=net_attached(m);
  if(!keeping)region_end_requests(m);
  if(keeping)say("bridge node %d: %s with clients attached: the region kept for the next bridge\n",me,atomic_load(&leaving)?"left the mesh":"stopped");
  /* The exit status says whether a successor is wanted, to a supervisor that starts one where it is not 0
     (bin/mesh-bridge.sh: launchd's KeepAlive, SuccessfulExit false): EX_TEMPFAIL where a bridge stopped with clients
     attached (they and its peers wait for the next, which takes the region and resumes them); else 0 (no client
     waits; or it left the mesh, its successor started only by whoever brings the node back; or its device's teardown
     failed, where a successor opening the device could meet what this one still holds: RDMA-RULES.md). */
  for(uint32_t i=0;i<device_count;i++)if(!down_device(&devices[i])){say("verbs teardown failed: %s: no successor wanted\n",strerror(errno));return 0;}
  for(uint32_t i=0;i<device_count;i++)pthread_mutex_destroy(&devices[i].setup);
  free(devices);free(sessions);munmap(wire.data,wire.length);
  atomic_store(&m->bridge_pid,0);
  munmap(m,length);return keeping && !atomic_load(&leaving)?EX_TEMPFAIL:0;
}
