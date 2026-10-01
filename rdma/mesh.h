#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <string.h>
/* design/pages-and-functions.md#block-addressing */
#define MESH_MAGIC 0x4d455348u
#define MESH_NAME "/mesh0"
#define MESH_PORT "18519"
#define MESH_MODE 0666
#define MESH_VERSION 112u
#define MESH_ABSENT UINT32_MAX
/* A link's communicator session's phase (mesh_net_link); MESH_LEFT: the link's peer bridge left the mesh (its LEAVE),
   or this node's bridge did, and no bridge has paired on the link since (mesh-flow.c net_leave). */
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_STOPPED, MESH_LEFT };
struct mesh_link_info { uint32_t peer; char device[32]; uint64_t bandwidth; };
/* The region: this header, the arena's occupancy bitmap, the links, the communicator tables, the statistics ring,
   and the arena of pgsz-byte pages from data_off (its first wire_pages the registered window); `serial` numbers
   its clients. */
struct hdr {
  uint32_t magic,version,pgsz,block,node,links,wire_pages,padding;
  uint64_t arena_off,link_off,net_off,data_off,length;
  _Atomic uint64_t bridge_pid,serial;
  /* the transport's statistics, lagged (below): the function evaluations ended so far, the ring's place, its
     entries' size, how many, and the lag its readers take (the bridge's -K) */
  _Atomic uint64_t evaluations;
  uint64_t stats_off,stats_stride;
  uint32_t stats_entries,stats_lag;
  /* this node's bridge instance (its keep's): the same while a bridge takes the region its last left, another once
     the region is made afresh (a client attached to an earlier region finds its name naming another instance) */
  _Atomic uint64_t instance;
};
/* The communicator service (mesh-net.h, NCCL's network plugin ncclNet_v12_t): its tables in the region.
   A client claims a slot by CAS from MESH_NET_FREE and publishes it; the session thread of the slot's
   link serves it (mesh-flow.c net_scan).  A comm is one direction of one connection (a listen, a send or
   a receive end), its requests a ring by sequence, posted by the client and completed by the bridge.
   Communicator memory is the region's registered window, used in place: a receive lands only in memory
   registered before its queue pair was set up (mesh-flow.c net_configure), which the window is.  The tables, and
   the bridge's own state of them (<region>.keep), outlive the bridge's process: a bridge stopped with clients
   attached leaves them for the next, which resumes their transfers while a client of them is alive, and releases
   them once none is (mesh-flow.c net_keep, region_release). */
#define MESH_NET_CLIENTS 32
#define MESH_NET_COMMS 256
#define MESH_NET_REQUESTS 64
#define MESH_NET_MEMORY 256
#define MESH_NET_WINDOW UINT32_MAX
enum { MESH_NET_FREE, MESH_NET_CLAIMED, MESH_NET_LISTEN, MESH_NET_CONNECTING, MESH_NET_ACCEPTABLE, MESH_NET_SEND,
       MESH_NET_RECV, MESH_NET_CLOSING, MESH_NET_FAILED };
enum { MESH_NET_IDLE, MESH_NET_POSTED, MESH_NET_ACTIVE, MESH_NET_DONE, MESH_NET_ERROR };
/* MESH_NET_HELD: an isend announced (its receiver matches it, posts its RECVs and grants them) whose granted
   chunks are not SENT until the client stores MESH_NET_ISEND into its op (mesh_net_release): its bytes are
   not yet written.  A closing comm's held isends go as they are. */
enum { MESH_NET_ISEND, MESH_NET_IRECV, MESH_NET_IFLUSH, MESH_NET_HELD };
/* A request: `offset` into the region's registered window (mr MESH_NET_WINDOW), `size` the bytes sent or
   the receive's capacity, `transferred` the bytes moved once it is done (an active isend's: the bytes its
   receiver has granted so far); `completion` 0, or 1 + the window offset of an 8-byte word into which the
   bridge stores the request's end (1 done, 2 failed) before its state, for a GPU kernel that waits on the
   word (nccl-mesh.c). */
struct mesh_net_request {
  _Alignas(64) _Atomic uint32_t state;
  _Atomic uint32_t op;
  uint32_t mr,mr_generation;
  int32_t tag; _Atomic int32_t error;
  uint64_t sequence,offset,size;
  _Atomic uint64_t transferred;
  uint64_t completion;
};
_Static_assert(sizeof(struct mesh_net_request)==64,"mesh_net_request");
/* A comm: its link, its client (`owner`), the key a connect names, a receive end's listen, the other
   end's comm on the peer bridge and that end's client (`peer_owner`, from its CONNECT or ACCEPT); `posted` the
   client's count of requests, the rest the bridge's counts.  `error` set while the comm is connected: its peer
   closed it (its requests then fail), ESRCH where the peer's client had exited. */
struct mesh_net_comm {
  _Alignas(64) _Atomic uint32_t state;
  uint32_t generation,link,listen,listen_generation,peer,peer_generation;
  _Atomic int32_t error;
  uint64_t owner,key,peer_owner;
  _Atomic uint64_t posted,bytes,completions,credit_waits;
  struct mesh_net_request requests[MESH_NET_REQUESTS];
};
/* A client's allocation of arena pages [first, first+pages) of the registered window
   (mesh_net_mem_alloc), vacated by the bridge once its client has exited and no comm of it remains.  `pages` is
   written last when it is made and zeroed first when it is released: 0 names no range. */
struct mesh_net_memory { _Alignas(16) _Atomic uint64_t owner; uint32_t first; _Atomic uint32_t pages; };
struct mesh_net_client { _Alignas(64) _Atomic uint64_t owner; };
/* A link's live counts: chunks whose SEND waited for its queue's frames (send_stalls), RECVs held back because
   their queue's frames were full, so the peer's SENDs into them waited for credit (receive_stalls), sends
   announced before their receiver's irecv, which waited for its credit (credit_waits); the communicators'
   messages and bytes; the session's phase and pairings, and the device's registered window regions; the session's
   heartbeats sent and heard, the longest it went hearing nothing from its peer (ns), when it last heard
   (CLOCK_MONOTONIC ns: the peer's liveness), and its resumptions after a session lost (the sessions that
   resumed, the chunks SENT again, the chunks' RECVs posted again).  These are live, for the bridge and
   liveness alone; everyone else reads them lagged (mesh_stats_read).
     Membership, not statistics: `pairing`, the identity of the two bridge instances the session last paired (0:
   never; another value: either bridge is another instance, whatever of the link's transfers had been kept is gone),
   and the peer node's client processes as its bridge reported them in that pairing (`peer_clients`, a slot each;
   complete once `clients_pairing` equals `pairing`): a process of the peer node absent from a complete report has
   exited (mesh-net.h mesh_net_exited).  A live process keeps its slot, so a report being rewritten never drops it. */
struct mesh_net_link {
  _Alignas(64) _Atomic uint32_t phase,chunk_frames;
  _Atomic int64_t code;
  _Atomic uint64_t doorbell,sessions;
  _Atomic uint64_t send_stalls,receive_stalls,credit_waits;
  _Atomic uint64_t net_sends,net_send_bytes,net_receives,net_receive_bytes;
  _Atomic uint32_t wire_regions,padding;
  _Atomic uint64_t heartbeats_sent,heartbeats_heard,silence_ns,heard_ns,resumes,resends,reposts;
  _Atomic uint64_t pairing,clients_pairing,peer_clients[MESH_NET_CLIENTS];
};
static inline struct mesh_net_link *mesh_net_links(struct hdr *m){return (struct mesh_net_link *)((char *)m+m->net_off);}
static inline struct mesh_net_client *mesh_net_clients(struct hdr *m){return (struct mesh_net_client *)(mesh_net_links(m)+m->links);}
static inline struct mesh_net_memory *mesh_net_memory(struct hdr *m){return (struct mesh_net_memory *)(mesh_net_clients(m)+MESH_NET_CLIENTS);}
static inline struct mesh_net_comm *mesh_net_comms(struct hdr *m){return (struct mesh_net_comm *)(mesh_net_memory(m)+MESH_NET_MEMORY);}
static inline uint64_t mesh_net_bytes(uint32_t links){
  return (uint64_t)links*sizeof(struct mesh_net_link)+MESH_NET_CLIENTS*sizeof(struct mesh_net_client)+
    MESH_NET_MEMORY*sizeof(struct mesh_net_memory)+MESH_NET_COMMS*sizeof(struct mesh_net_comm);
}
/* The transport's statistics, lagged.  The operator: statistics "can still be available and be exposed, albeit
   they should never be exposed to the inner contents of a function evaluation ... (why not have a literally
   enforced delay on the computation of statistics ...)".  A ring of stats_entries entries indexed by function
   evaluation: the client whose evaluation n ends (a library call's part: mesh_net_stats_record) writes entry n,
   every link's counts (mesh_net_link) and its own (`client`: libnccl-mesh's) as they stand then; the one read
   takes the reader's evaluation n and gives entries of evaluations up to n - lag (mesh_stats_read), so nothing
   inside an evaluation reads anything fresher, and whatever it reads is at least `lag` evaluations old. */
#define MESH_STATS 1024
#define MESH_STATS_CLIENT 24
struct mesh_stats_link { uint64_t send_stalls,receive_stalls,credit_waits,net_sends,net_send_bytes,net_receives,net_receive_bytes,
  sessions,heartbeats_sent,heartbeats_heard,silence_ns,resumes,resends,reposts; };
struct mesh_stats_entry { _Atomic uint64_t evaluation; uint64_t ns; uint32_t links,pid; uint64_t client[MESH_STATS_CLIENT]; struct mesh_stats_link link[]; };
static inline struct mesh_stats_entry *mesh_stats_at(struct hdr *m,uint64_t evaluation){
  return (struct mesh_stats_entry *)((char *)m+m->stats_off+(size_t)(evaluation%m->stats_entries)*m->stats_stride);
}
static inline struct mesh_net_link *mesh_net_links(struct hdr *m);
/* An evaluation's end recorded (`client`: `count` counts, `pid` its process's; `ns` the time): entry n of the ring,
   n the evaluations ended with this one (not written where another writer still holds the slot); n. */
static inline uint64_t mesh_stats_record(struct hdr *m,const uint64_t *client,uint32_t count,uint32_t pid,uint64_t ns){
  const uint64_t n=atomic_fetch_add_explicit(&m->evaluations,1,memory_order_acq_rel)+1;
  struct mesh_stats_entry *e=mesh_stats_at(m,n);
  /* the slot taken from the entry it held (its writer done), else left to the writer that holds it: one writer a slot */
  uint64_t held=atomic_load_explicit(&e->evaluation,memory_order_acquire);
  if(held==UINT64_MAX || !atomic_compare_exchange_strong_explicit(&e->evaluation,&held,UINT64_MAX,memory_order_acq_rel,memory_order_relaxed))return n;
  atomic_thread_fence(memory_order_release);
  e->ns=ns;e->links=m->links;e->pid=pid;
  for(uint32_t i=0;i<MESH_STATS_CLIENT;i++)e->client[i]=i<count?client[i]:0;
  for(uint32_t l=0;l<m->links;l++){
    struct mesh_net_link *k=mesh_net_links(m)+l;
    e->link[l]=(struct mesh_stats_link){atomic_load(&k->send_stalls),atomic_load(&k->receive_stalls),atomic_load(&k->credit_waits),
      atomic_load(&k->net_sends),atomic_load(&k->net_send_bytes),atomic_load(&k->net_receives),atomic_load(&k->net_receive_bytes),
      atomic_load(&k->sessions),atomic_load(&k->heartbeats_sent),atomic_load(&k->heartbeats_heard),atomic_load(&k->silence_ns),
      atomic_load(&k->resumes),atomic_load(&k->resends),atomic_load(&k->reposts)};
  }
  atomic_store_explicit(&e->evaluation,n,memory_order_release);
  return n;
}
/* The one read of the statistics: the reader's evaluation `evaluation` (the evaluations it has seen end, at
   most the region's), and the entries of evaluations first..evaluation - lag still in the ring copied into
   `out` (capacity entries of stats_stride bytes); how many. */
static inline size_t mesh_stats_read(struct hdr *m,uint64_t evaluation,uint64_t first,void *out,size_t capacity){
  const uint64_t ended=atomic_load_explicit(&m->evaluations,memory_order_acquire);
  if(evaluation>ended)evaluation=ended;
  if(evaluation<=m->stats_lag)return 0;
  const uint64_t last=evaluation-m->stats_lag,oldest=ended>=m->stats_entries?ended-m->stats_entries+1:1;
  if(first<oldest)first=oldest;
  if(!first)first=1;
  size_t got=0;
  for(uint64_t n=first;n<=last && got<capacity;n++){
    const struct mesh_stats_entry *e=mesh_stats_at(m,n);
    struct mesh_stats_entry *to=(struct mesh_stats_entry *)((char *)out+got*m->stats_stride);
    if(atomic_load_explicit(&e->evaluation,memory_order_acquire)!=n)continue;
    memcpy((char *)to+sizeof to->evaluation,(const char *)e+sizeof e->evaluation,m->stats_stride-sizeof e->evaluation);
    atomic_thread_fence(memory_order_acquire);
    if(atomic_load_explicit(&e->evaluation,memory_order_relaxed)!=n)continue;
    atomic_store_explicit(&to->evaluation,n,memory_order_relaxed);
    got++;
  }
  return got;
}
static inline struct mesh_link_info *mesh_links(struct hdr *m){return (struct mesh_link_info *)((char *)m+m->link_off);}
/* design/prepared-machine.md#M09 */
static inline uint32_t mesh_blocks(const struct hdr *m){ return (uint32_t)((m->length-m->data_off)/((uint64_t)m->block*m->pgsz)); }
static inline uint32_t mesh_arena_pages(const struct hdr *m){ return mesh_blocks(m)*m->block; }
/* the arena occupancy bitmap, a bit an arena page */
static inline _Atomic uint64_t *mesh_arena_bits(struct hdr *m){ return (_Atomic uint64_t*)((unsigned char*)m+m->arena_off); }
/* design/prepared-machine.md#M09 */
/* the registered window is the first wire_pages of the arena; every other page is addressable and unregistered */
static inline uint64_t mesh_wire_bytes(const struct hdr *m){ return (uint64_t)m->wire_pages*m->pgsz; }
static inline unsigned char *mesh_at(struct hdr *m,uint32_t page){ return (unsigned char*)m+m->data_off+(size_t)page*m->pgsz; }

static inline uint64_t mesh_word_mask(uint32_t first,uint32_t count,uint32_t word){
  uint32_t lo=word*64,hi=lo+64,a=first>lo?first:lo,b=first+count<hi?first+count:hi;
  if(b<=a) return 0;
  uint64_t bits=b-a==64?~UINT64_C(0):((UINT64_C(1)<<(b-a))-1);
  return bits<<(a-lo);
}
static inline void mesh_bits_clear(_Atomic uint64_t *p,uint32_t first,uint32_t count){
  for(uint32_t w=first/64;count && w<=(first+count-1)/64;w++) atomic_fetch_and_explicit(&p[w],~mesh_word_mask(first,count,w),memory_order_acq_rel);
}

/* The region's layout: arena pages and the registered window are independent (an undeclared window is the whole
   arena), the tables sized by the links. */
struct mesh_geometry { uint32_t pgsz,block,pages,links,wire_pages; };
static inline uint64_t mesh_layout(struct hdr *h,struct mesh_geometry g){
  uint32_t blocks=g.pages/g.block,arena=blocks*g.block;
  if(!g.wire_pages || g.wire_pages>arena)g.wire_pages=arena;
  g.wire_pages=(g.wire_pages+g.block-1)/g.block*g.block;
  if(g.wire_pages>arena)g.wire_pages=arena;
  uint32_t pgsz=g.pgsz;
  uint64_t at=(sizeof *h+pgsz-1)/pgsz*pgsz;
  h->pgsz=pgsz; h->block=g.block; h->wire_pages=g.wire_pages; h->links=g.links;
  h->arena_off=at; at+=(((uint64_t)arena+63)/64)*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->link_off=at; at+=(uint64_t)g.links*sizeof(struct mesh_link_info); at=(at+pgsz-1)/pgsz*pgsz;
  h->net_off=at; at+=mesh_net_bytes(g.links); at=(at+pgsz-1)/pgsz*pgsz;
  h->stats_stride=(sizeof(struct mesh_stats_entry)+(uint64_t)g.links*sizeof(struct mesh_stats_link)+63)&~UINT64_C(63);
  h->stats_entries=MESH_STATS;
  h->stats_off=at; at+=(uint64_t)MESH_STATS*h->stats_stride;
  uint64_t bytes=(uint64_t)g.block*pgsz; at=(at+bytes-1)/bytes*bytes;
  h->data_off=at; at+=(uint64_t)g.pages*pgsz;
  h->length=at; return at;
}
#endif
