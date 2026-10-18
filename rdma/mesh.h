#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <infiniband/verbs.h>
#include <os/os_sync_wait_on_address.h>
#include "mesh-plan.h"
/* design/pages-and-functions.md#block-addressing */
#define MESH_MAGIC 0x4d455348u
#define MESH_NAME "/mesh0"
#define MESH_PORT "18519"
/* the mesh's unique local prefix, each node's fabric identity under it (bin/mesh-router-init.sh) */
#define MESH_ULA_PREFIX "fd6d:6573:68:"
#define MESH_MODE 0666
/* 119: each exclusive flow's control channel and a lane rider's control entries (design/bridge-and-ipc.md#control:
   LANDED, END, CLOSED and FAIL frames), a client's delivered words and its finalize word (`delivered`, `ending`), and
   the orderly retire (MESH_ORDERLY) */
#define MESH_VERSION 119u
#define MESH_ABSENT UINT32_MAX
/* design/bridge-and-ipc.md#region: the node's ports, its client slots and each client's destination nodes */
#define MESH_PORTS 8
#define MESH_CLIENTS 64
#define MESH_DESTINATIONS 64
#define MESH_QUEUES 4
#define MESH_ORDERS 4096
#define MESH_RETIRING (UINT64_C(1)<<63)
/* design/bridge-and-ipc.md#control: a slot retired in order (mesh-dataflow.c mesh_detach), held by the bridge until it
   has taken its flows' close (their released cells posted, then END), which then vacates it */
#define MESH_ORDERLY (UINT64_C(1)<<62)
/* design/recovery.md#pause, #first: the pause between a port's pairings, how long a program waits for its peer's to
   start, and how long the queue pairs of two started programs take to pair; a client's start waits on its bridge at
   most the three together */
#define MESH_PAUSE_NS UINT64_C(3000000000)
#define MESH_PEER_NS UINT64_C(600000000000)
#define MESH_PAIR_NS UINT64_C(30000000000)
/* nccl-mesh.c lease_wait: how long a host-path group waits for the caller's own GPU work to reach a point it recorded on
   a lease (ncclMeshMemUse), a command buffer the caller never commits being no event that can ever come; no transport
   completion or teardown reads it (design/recovery.md#detection) */
#define MESH_SILENT_NS UINT64_C(10000000000)
/* A prepared transfer: `count` slots of `bytes` from `local_row`, bound for the invocations [begin, end)
   that use it (end past the call's invocations: every one), invocation t addressing its ring slot
   (t - begin) mod depth (0: the client's depth) at `invocation_pages` a slot. */
struct mesh_transfer { uint32_t local_row,binding,count,stride,invocation_pages,first,begin,end,depth; uint64_t bytes; };
_Static_assert(sizeof(struct mesh_transfer)==48,"M08 prepared transfer");
static inline uint32_t mesh_transfer_active(const struct mesh_transfer *transfer,uint32_t invocations){
  uint32_t end=transfer->end<invocations?transfer->end:invocations;
  return transfer->stride?(end>transfer->begin?end-transfer->begin:0):(transfer->begin<invocations?1:0);
}
/* design/algorithm-sources.md#programtensor */
enum { MESH_ROW_OWN, MESH_ROW_HOT, MESH_FREE, MESH_PLANES };
/* design/algorithm-sources.md#programtensor; design/prepared-machine.md#M01, #M02 */
struct mesh_buffer {
  _Alignas(32) _Atomic uint64_t owner;
  _Atomic uint32_t closed;
  uint32_t pages;
};
_Static_assert(sizeof(struct mesh_buffer)==32 && _Alignof(struct mesh_buffer)==32,"mesh_buffer");
/* design/algorithm-sources.md#programtensor; design/prepared-machine.md#M10 */
struct mesh_page_entry {
  _Alignas(16) _Atomic uint64_t mapping;
  _Atomic uintptr_t address;
};
_Static_assert(sizeof(struct mesh_page_entry)==16 && _Alignof(struct mesh_page_entry)==16 &&
  offsetof(struct mesh_page_entry,address)==8,"mesh_page_entry");
/* design/collective-dependency-ledger.md#d5-receive-consumption-has-per-queue-fifo-order */
enum { MESH_SEND, MESH_RECEIVE };
/* design/prepared-machine.md#M04 */
/* One publication's SEND cell of one invocation: `ready`, released by its producer, then the chain
   of `chunks` requests from `request` the bridge posts, `successor` the stream's next cell (the
   bridge's, prepared per invocation), and `port` the index plus one of the port whose send thread
   posts it (design/bridge-and-ipc.md#progress: a relay's landing that releases it rings that port). */
struct mesh_send {
  _Alignas(128) _Atomic uint64_t ready;
  uintptr_t pair;
  _Alignas(32) struct ibv_sge span;
  struct ibv_send_wr request;
  uintptr_t successor;
  uint32_t chunks,queue;
  _Atomic uint32_t port;
};
/* design/bridge-and-ipc.md#region: a client's program to one destination node (`node`, 0 while the slot is vacant),
   which the bridge carries over its ports to that node or along a route through others: its SEND cells, its
   cancellation words and its transfer lists (MESH_QUEUES queues each way, `length` entries in each, `room` entries a
   list in the lease headed at row `lists`, which grows by doubling to MESH_ORDERS: mesh-call.c mesh_transfer_bind) */
struct mesh_tx { uint32_t count,slots,once,invocations; uint64_t cancel,cells,orders; uint32_t node; _Atomic uint32_t length[MESH_QUEUES*2];
  uint32_t room,lists; };
_Static_assert(sizeof(struct mesh_send)==256 && _Alignof(struct mesh_send)==128 &&
  offsetof(struct mesh_send,span)==32 && offsetof(struct mesh_send,request)==48 && offsetof(struct mesh_send,port)+4<=256 &&
  offsetof(struct mesh_send,request.send_flags)+sizeof(unsigned int)<=128,"M04/M29");
_Static_assert(offsetof(struct mesh_tx,cancel)==16 && offsetof(struct mesh_tx,cells)==24 && sizeof(struct mesh_tx)==88,"M04/M12");
/* design/algorithm-sources.md#index-hand-off; design/prepared-machine.md#M10: a program's publication of `row` toward
   one destination, `cells` the region offset of its column of SEND cells (invocation u's at 256 u past it) */
struct mesh_target { uint64_t cells; uint32_t row,padding; };
_Static_assert(sizeof(struct mesh_target)==16 && offsetof(struct mesh_target,row)==8,"mesh_target");
/* design/prepared-machine.md#M07, #M09: a row's delivery, the address and per-invocation stride of the completion word
   its landed chunk stores at invocation u (`device_input` plus `device_stride` u, plus slot*chunks words where every
   slot of its transfer lands on this row; 0 where its program prepared none) */
struct mesh_delivery { _Alignas(16) uint64_t device_input; uint64_t device_stride; };
_Static_assert(sizeof(struct mesh_delivery)==16 && _Alignof(struct mesh_delivery)==16,"mesh_delivery");
/* design/prepared-machine.md#M13 */
struct prepared_publication { _Alignas(16) uint64_t destination; uint64_t argument,padding[2]; };
_Static_assert(sizeof(struct prepared_publication)==32 && _Alignof(struct prepared_publication)==16 && offsetof(struct prepared_publication,argument)==8,"M13");
/* design/bridge-and-ipc.md#region: a client's program to one destination as the bridge tells it (its phase, the
   failure's code and domain, `failed` the node where it failed: the destination's, or the relay's on its route, and
   `prepared` the owner once the bridge paired or failed it); a port of the node, by the index the bridge gave it (its
   peer node while paired, its phase, MESH_UNKNOWN where the bridge has no such port, the code it last stopped with, its
   bandwidth, the pairings it made, and the queue pairs its device holds against those it can use, `pairs` of
   `max_pairs`, its max_qp less one);
   and a node's record, the newest of which every bridge floods to the mesh (its node, name and boot, the record's
   sequence in that boot, its bridge's window and the node's memory in bytes and hardware model (hw.model,
   mesh-plan.h MESH_MODEL_BYTES with its NUL), and its `ports` ports by index) */
struct mesh_port_info { _Atomic uint32_t phase,domain,failed; _Atomic int64_t code; _Atomic uint64_t prepared; };
struct mesh_link_info { _Atomic uint32_t peer,phase; _Atomic int64_t code; _Atomic uint64_t bandwidth,pairings; _Atomic uint32_t pairs,max_pairs; };
_Static_assert(sizeof(struct mesh_link_info)==40,"mesh_link_info");
struct mesh_record { uint32_t node,ports; uint64_t boot,sequence; char name[MESH_NAME_BYTES]; uint64_t window,memory;
  char model[MESH_MODEL_BYTES]; struct mesh_link_info port[MESH_PORTS]; struct mesh_health health; };
/* design/bridge-and-ipc.md#region: a client slot: its attach (`owner`, 0 vacant), its program once started
   (`configured`, the owner), the execution it belongs to (`key`, the same on every member), its ring depth, whether it
   is cyclic, its publications once prepared (`target_count` targets at region offset `targets`, after its SEND cells
   in their lease, sorted by row: mesh_row_targets), the ports its program's flows ride (`ports`, a bit a port index:
   the bells mesh_transfers_wake rings), the digest of what its program is made from (`digest`, 0 for none:
   mesh-call.h mesh_transfers_digest; design/bridge-and-ipc.md#protocol, a peer's READY naming another fails both ends'
   programs, ESTALE), and its program to each destination and that destination's status; `delivered`, its program's
   SEND cells toward each destination on each queue (channel = destination x MESH_QUEUES + queue, as mesh_transfers
   names it) that the peer's bridge reported landed (design/bridge-and-ipc.md#control: an exclusive flow's LANDED
   frames, every flow's CLOSED), a cumulative count over the program's cycles summed over its flows' stripes and slots;
   `ending`, nonzero once the client asks its bridge to close its program's flows in order and keeps its slot (nccl-mesh.c
   ncclCommFinalize).  A client writes its own slot's other words and its leased rows alone; the bridge writes `ports`,
   the statuses, `delivered`, the ports and the map */
struct mesh_client {
  _Alignas(64) _Atomic uint64_t owner;
  _Atomic uint64_t configured;
  uint64_t key;
  _Atomic uint32_t depth,cyclic;
  uint64_t targets;
  uint32_t target_count;
  _Atomic uint32_t ports;
  _Atomic uint64_t digest;
  struct mesh_tx tx[MESH_DESTINATIONS];
  struct mesh_port_info status[MESH_DESTINATIONS];
  _Atomic uint64_t delivered[MESH_DESTINATIONS*MESH_QUEUES];
  _Atomic uint32_t ending;
};
/* design/prepared-machine.md#M01, #M09; `busy`, design/bridge-and-ipc.md#region: the flows and awaiting programs the
   bridge holds (0 between calls), and since when it has held none (`idle_since`, the wall clock in ns; 0 while it holds
   any); `stopping`, design/bridge-and-ipc.md#region: nonzero once the bridge waits to stop (its binary changed) for the
   flows it holds to end, so a client holding flows past their execution (metal-microbench
   docs/placement.md#retention-and-protention-the-next-execution-planned-ahead) releases them between executions and holds
   none after; `bell`, design/bridge-and-ipc.md#progress: a port's word its parked progress threads wait on, which a
   client rings (mesh_notify) as a group of a program riding that port starts */
struct hdr {
  uint32_t magic,version,pgsz,block,rows,node,links,clients;
  uint64_t planes_off,arena_off,page_off,buffer_off,link_off,client_off,delivery_off,data_off,length,boot;
  _Atomic uint64_t bridge_pid,serial,control,map,map_off,map_nodes,health,busy,idle_since,stopping;
  _Atomic uint64_t bell[MESH_PORTS];
  uint64_t rings_off;
};
/* design/bridge-dataflow.md#stripes (mesh 8fc27cbe's request rings, on the resident bridge): the rings of stripe
   requests the node's clients registered (mesh-call.h mesh_requests_create), MESH_RINGS region offsets at `rings_off`,
   each a client's struct mesh_requests, 0 vacant.  A stripe request writes the `bytes` bytes at byte `offset` of this
   node's window (the arena, from data_off) to the same offset of the window of node `node`, a neighbour on a cable,
   with no program at the peer, and stores `value` into the word at region offset `word` (0: none) on both nodes as the
   RDMA driver completes it: the writer's once its last SEND completes, the peer's once its last receive lands; a
   request the bridge cannot carry stores UINT64_MAX into the writer's word (mesh_cancel's convention) and counts in its
   ring's `failed`, its code in `code`.  Request k of a ring is its entry k mod `entries` once that entry's `ready` is
   k + 1 (stored last, release) by whoever writes it (a thread of any process, or a GPU); `taken` is how many the bridge
   has taken (it only grows), so an entry is written again only once the request it held is taken; `head` is the next
   request's number for host writers (mesh_write takes it atomically); `owner` the client slot whose lease the ring is
   (its retirement deregisters the ring), `at` the ring's own region offset */
#define MESH_RINGS 256
struct mesh_stripe { _Alignas(64) _Atomic uint64_t ready; uint64_t offset,bytes,word,value; uint32_t node,padding; uint64_t reserved[2]; };
struct mesh_requests { _Alignas(128) _Atomic uint64_t taken; uint64_t entries,owner,at; _Atomic uint64_t failed; _Atomic int64_t code;
  _Alignas(64) _Atomic uint64_t head; uint64_t reserved[7]; struct mesh_stripe request[]; };
_Static_assert(sizeof(struct mesh_stripe)==64 && offsetof(struct mesh_requests,head)==64 && offsetof(struct mesh_requests,request)==128,
  "stripe requests");
static inline _Atomic uint64_t *mesh_rings(struct hdr *m){ return (_Atomic uint64_t *)((char *)m+m->rings_off); }
/* design/prepared-machine.md#M26 */
_Static_assert(sizeof(((struct hdr *)0)->control)==8 && offsetof(struct hdr,control)%8==0,"M26");
/* design/prepared-machine.md#M07 */
_Static_assert(sizeof(_Atomic uint64_t)==8 && _Alignof(_Atomic uint64_t)==8,"M07");
/* design/prepared-machine.md#M12 */
struct mesh_cancel_range { _Alignas(32) uint64_t offset; uint64_t count,padding[2]; };
struct mesh_cancellation { _Alignas(32) _Atomic uint32_t requested; uint32_t cyclic,padding[6]; struct mesh_cancel_range ranges[]; };
_Static_assert(sizeof(struct mesh_cancel_range)==32 && sizeof(struct mesh_cancellation)==32 && offsetof(struct mesh_cancellation,ranges)==32,"M12");
/* design/algorithm-sources.md#meshresult; design/prepared-machine.md#M12: a destination's words are cancelled by the
   bridge alone, after the last landing its flow stores, and only at positions past its landed count
   (mesh-flow.c flow_cancel, design/bridge-and-ipc.md#control) */
/* design/algorithm-sources.md#meshresult; design/prepared-machine.md#M26: a shared word (`control`, `map`) bumped and
   its waiters woken, the writes after it ordered behind it */
static inline void mesh_notify(_Atomic uint64_t *word){
  atomic_fetch_add_explicit(word,1,memory_order_release);
  atomic_thread_fence(memory_order_release);
  os_sync_wake_by_address_all(word,sizeof *word,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
/* design/bridge-and-ipc.md#region: the map, `map_nodes` records at `map_off` in arena pages the bridge leases for
   itself, and leases again larger as the mesh grows; the bridge notifies `map`, its serial, before and after each
   change (odd while it writes; its own record is in before the region opens, so a client never reads 0), and a
   reader's copy is whole where `map` read (acquired) before it was even and is unchanged after it */
static inline struct mesh_record *mesh_records(struct hdr *m){ return (struct mesh_record *)((char *)m+atomic_load_explicit(&m->map_off,memory_order_relaxed)); }
static inline int mesh_map_whole(struct hdr *m,uint64_t before){
  atomic_thread_fence(memory_order_acquire);
  return !(before&1) && atomic_load_explicit(&m->map,memory_order_relaxed)==before;
}
/* design/algorithm-sources.md#programcopy */
static inline struct mesh_link_info *mesh_links(struct hdr *m){return (struct mesh_link_info *)((char *)m+m->link_off);}
/* design/bridge-and-ipc.md#region: an owner word is its generation, slot and process: (generation << 38) | (slot << 32) | pid */
static inline struct mesh_client *mesh_client_at(struct hdr *m,uint64_t owner){return (struct mesh_client *)((char *)m+m->client_off)+((owner>>32)&(MESH_CLIENTS-1));}
/* design/bridge-and-ipc.md#clients: a client's slot for destination `node`, the one bound for it, else the first vacant
   one (slots are bound in order); MESH_DESTINATIONS where every slot is bound for another node */
static inline uint32_t mesh_destination(const struct mesh_client *client,uint32_t node){
  uint32_t d=0;
  while(d<MESH_DESTINATIONS && client->tx[d].node && client->tx[d].node!=node)d++;
  return d;
}
/* design/bridge-and-ipc.md#routes: the bytes a relay leases a slot of a transfer it mirrors (mesh-flow.c mirror), before
   the window's block rounds them: its ring's slots of `invocation_pages` pages but the last, and the last its bytes or
   its pages, whichever is more, at its own depth, else the program's `depth` */
static inline uint64_t mesh_relay_slot(const struct mesh_transfer *x,uint32_t depth,uint32_t pgsz){
  const uint32_t slots=x->depth?x->depth:depth;
  const uint64_t tail=(uint64_t)x->invocation_pages*pgsz;
  return (uint64_t)(slots?slots-1:0)*tail+(x->bytes>tail?x->bytes:tail);
}
/* design/prepared-machine.md#M07 */
static inline struct mesh_delivery *mesh_delivery_at(struct hdr *m,uint32_t row){return (struct mesh_delivery *)((char *)m+m->delivery_off)+row;}
/* design/algorithm-sources.md#index-hand-off; design/prepared-machine.md#M10: the publications of `row` in the program of
   slot `owner`, a run of its targets (their count in *count, 0 where it publishes none), found by bisection */
static inline const struct mesh_target *mesh_row_targets(struct hdr *m,uint64_t owner,uint32_t row,uint32_t *count){
  const struct mesh_client *client=mesh_client_at(m,owner);
  const uint32_t n=client->targets?client->target_count:0;
  const struct mesh_target *targets=(const struct mesh_target *)((const char *)m+client->targets);
  uint32_t low=0,high=n;
  while(low<high){uint32_t middle=low+(high-low)/2;if(targets[middle].row<row)low=middle+1;else high=middle;}
  for(high=low;high<n && targets[high].row==row;high++);
  *count=high-low;
  return targets+low;
}
static inline uint32_t mesh_rows(const struct hdr *m){ return m->rows; }
static inline uint32_t mesh_words(const struct hdr *m){ return (mesh_rows(m)+63)/64; }
/* design/prepared-machine.md#M09; design/pages-and-functions.md#what-the-page-table-is */
static inline uint32_t mesh_arena_pages(const struct hdr *m){ return (uint32_t)((m->length-m->data_off)/((uint64_t)m->block*m->pgsz))*m->block; }
static inline _Atomic uint64_t *mesh_plane(struct hdr *m,int plane){ return (_Atomic uint64_t*)((unsigned char*)m+m->planes_off)+(size_t)plane*mesh_words(m); }
/* design/prepared-machine.md#M01 */
/* the arena occupancy bitmap is indexed by arena page and sized by arena pages: it is not a row plane */
static inline _Atomic uint64_t *mesh_arena_bits(struct hdr *m){ return (_Atomic uint64_t*)((unsigned char*)m+m->arena_off); }
static inline struct mesh_page_entry *mesh_page(struct hdr *m){ return (struct mesh_page_entry*)((unsigned char*)m+m->page_off); }
/* design/algorithm-sources.md#programtensor */
static inline struct mesh_buffer *mesh_buffers(struct hdr *m){ return (struct mesh_buffer *)((char *)m+m->buffer_off); }
/* ledger D5: a client's lists of one destination, queue and direction: channel = destination x MESH_QUEUES + queue */
static inline struct mesh_transfer *mesh_transfers(struct hdr *m,uint64_t owner,uint32_t channel,int direction){
  const struct mesh_tx *tx=&mesh_client_at(m,owner)->tx[channel/MESH_QUEUES];
  return (struct mesh_transfer*)((unsigned char*)m+tx->orders)+((size_t)(channel%MESH_QUEUES)*2+(uint32_t)direction)*tx->room;
}
static inline _Atomic uint32_t *mesh_order_length(struct hdr *m,uint64_t owner,uint32_t channel,int direction){ return mesh_client_at(m,owner)->tx[channel/MESH_QUEUES].length+(channel%MESH_QUEUES)*2+(uint32_t)direction; }
/* design/bridge-and-ipc.md#control: the SEND cells of a client's program on `channel` the peer's bridge reported landed */
static inline _Atomic uint64_t *mesh_delivered(struct hdr *m,uint64_t owner,uint32_t channel){ return mesh_client_at(m,owner)->delivered+channel; }
static inline uint32_t mesh_destination_transfers(struct hdr *m,uint64_t owner,uint32_t destination){
  uint32_t transfers=0;
  for(uint32_t q=0;q<MESH_QUEUES*2;q++)transfers+=atomic_load_explicit(mesh_client_at(m,owner)->tx[destination].length+q,memory_order_acquire);
  return transfers;
}
static inline unsigned char *mesh_at(struct hdr *m,uint32_t page){ return (unsigned char*)m+m->data_off+(size_t)page*m->pgsz; }
/* design/algorithm-sources.md#programcopy; design/prepared-machine.md#M08 */
static inline uint32_t mesh_row_chunks(struct hdr *m,uint32_t row,uint64_t bytes){
  uint64_t block=(uint64_t)m->block*m->pgsz;
  uint64_t offset=atomic_load_explicit(&mesh_page(m)[row].address,memory_order_relaxed)-m->data_off;
  return (uint32_t)((offset%block+bytes+block-1)/block);
}

static inline uint64_t mesh_word_mask(uint32_t first,uint32_t count,uint32_t word){
  uint32_t lo=word*64,hi=lo+64,a=first>lo?first:lo,b=first+count<hi?first+count:hi;
  if(b<=a) return 0;
  uint64_t bits=b-a==64?~UINT64_C(0):((UINT64_C(1)<<(b-a))-1);
  return bits<<(a-lo);
}
static inline void mesh_bits_set(_Atomic uint64_t *p,uint32_t first,uint32_t count){
  for(uint32_t w=first/64;count && w<=(first+count-1)/64;w++) atomic_fetch_or_explicit(&p[w],mesh_word_mask(first,count,w),memory_order_acq_rel);
}
static inline void mesh_bits_clear(_Atomic uint64_t *p,uint32_t first,uint32_t count){
  for(uint32_t w=first/64;count && w<=(first+count-1)/64;w++) atomic_fetch_and_explicit(&p[w],~mesh_word_mask(first,count,w),memory_order_acq_rel);
}

/* design/algorithm-sources.md#programkernel_call; design/prepared-machine.md#M01 */
/* design/bridge-and-ipc.md#region; design/prepared-machine.md#M09: the node's region: `pages` arena pages of `block`
   pages a block (the bridge's window, its node's configured share of its memory), a page-table row a block (its page
   entry, buffer and delivery, 64 bytes, and a bit in each plane), the node's ports, the client slots and the stripe
   request rings' table (MESH_RINGS words); a program's
   publication targets are no part of it but a section of the program's slot (mesh_row_targets), nor are the map's
   records, a lease the bridge takes from the arena (mesh_records); the arena is registered whole on every port's device
   before any queue pair exists (mesh-verbs.h device_up) */
static inline uint64_t mesh_layout(struct hdr *h,uint32_t pgsz,uint32_t block,uint32_t pages){
  uint32_t arena=pages/block*block,rows=arena/block;
  uint64_t at=(sizeof *h+pgsz-1)/pgsz*pgsz,words=((uint64_t)rows+63)/64;
  h->pgsz=pgsz; h->block=block; h->rows=rows; h->links=MESH_PORTS; h->clients=MESH_CLIENTS;
  h->planes_off=at; at+=(uint64_t)MESH_PLANES*words*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->arena_off=at; at+=(((uint64_t)arena+63)/64)*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->page_off=at; at+=(uint64_t)rows*sizeof(struct mesh_page_entry); at=(at+pgsz-1)/pgsz*pgsz;
  h->buffer_off=at; at+=(uint64_t)rows*sizeof(struct mesh_buffer); at=(at+pgsz-1)/pgsz*pgsz;
  h->link_off=at; at+=(uint64_t)MESH_PORTS*sizeof(struct mesh_link_info); at=(at+pgsz-1)/pgsz*pgsz;
  h->client_off=at; at+=(uint64_t)MESH_CLIENTS*sizeof(struct mesh_client); at=(at+pgsz-1)/pgsz*pgsz;
  h->delivery_off=at; at+=(uint64_t)rows*sizeof(struct mesh_delivery); at=(at+pgsz-1)/pgsz*pgsz;
  h->rings_off=at; at+=(uint64_t)MESH_RINGS*sizeof(uint64_t);
  uint64_t bytes=(uint64_t)block*pgsz; at=(at+bytes-1)/bytes*bytes;
  h->data_off=at; at+=(uint64_t)arena*pgsz;
  h->length=at; return at;
}
#endif
