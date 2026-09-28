#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <infiniband/verbs.h>
#include <os/os_sync_wait_on_address.h>
/* design/pages-and-functions.md#block-addressing */
#define MESH_MAGIC 0x4d455348u
#define MESH_NAME "/mesh0"
#define MESH_PORT "18519"
#define MESH_MODE 0666
#define MESH_VERSION 108u
#define MESH_ABSENT UINT32_MAX
/* design/collective-dependency-ledger.md#d6-paired-send-and-receive-frame-counts-match */
#define MESH_QPS 8
/* A prepared transfer: `count` slots of `bytes` from `local_row`, bound for the invocations [begin, end)
   that use it (end past the call's invocations: every one), invocation t addressing its ring slot
   (t - begin) mod depth (0: the header's depth) at `invocation_pages` a slot. */
struct mesh_transfer { uint32_t local_row,binding,count,stride,invocation_pages,first,begin,end,depth; uint64_t bytes; };
_Static_assert(sizeof(struct mesh_transfer)==48,"M08 prepared transfer");
static inline uint32_t mesh_transfer_active(const struct mesh_transfer *transfer,uint32_t invocations){
  uint32_t end=transfer->end<invocations?transfer->end:invocations;
  return transfer->stride?(end>transfer->begin?end-transfer->begin:0):(transfer->begin<invocations?1:0);
}
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_STOPPED };
/* design/algorithm-sources.md#programtensor */
enum { MESH_ROW_OWN, MESH_ROW_HOT, MESH_FREE, MESH_PLANES };
/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M02 */
struct mesh_buffer {
  _Alignas(32) _Atomic uint64_t owner;
  _Atomic uint32_t closed;
  uint32_t pages,constant;
};
_Static_assert(sizeof(struct mesh_buffer)==32 && _Alignof(struct mesh_buffer)==32,"mesh_buffer");
/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M10 */
struct mesh_page_entry {
  _Alignas(16) _Atomic uint64_t mapping;
  _Atomic uintptr_t address;
  _Atomic uint64_t device;
};
_Static_assert(sizeof(struct mesh_page_entry)==32 && _Alignof(struct mesh_page_entry)==16 &&
  offsetof(struct mesh_page_entry,address)==8 && offsetof(struct mesh_page_entry,device)==16,"mesh_page_entry");
/* design/collective-dependency-ledger.md#d5-receive-consumption-has-per-queue-fifo-order */
enum { MESH_SEND, MESH_RECEIVE };
#define MESH_NOTICE_BANKS 2
/* design/prepared-machine.md#M04 */
/* One publication's SEND cell of one invocation: `ready`, released by its producer, then the chain
   of `chunks` requests from `request` the bridge posts, `successor` the stream's next cell (the
   bridge's, prepared per invocation). */
struct mesh_send {
  _Alignas(128) _Atomic uint64_t ready;
  uintptr_t pair;
  _Alignas(32) struct ibv_sge span;
  struct ibv_send_wr request;
  uintptr_t successor;
  uint32_t chunks,queue;
};
struct mesh_tx { uint32_t count,slots,once,invocations; uint64_t cancel,cells; };
_Static_assert(sizeof(struct mesh_send)==256 && _Alignof(struct mesh_send)==128 &&
  offsetof(struct mesh_send,span)==32 && offsetof(struct mesh_send,request)==48 && offsetof(struct mesh_send,queue)+4<=256 &&
  offsetof(struct mesh_send,request.send_flags)+sizeof(unsigned int)<=128,"M04/M29");
_Static_assert(offsetof(struct mesh_tx,cancel)==16 && offsetof(struct mesh_tx,cells)==24 && sizeof(struct mesh_tx)==32,"M04/M12");
struct mesh_target { uint64_t stream; uint32_t count,stride; };
_Static_assert(sizeof(struct mesh_target)==16,"mesh_target");
/* design/algorithm-sources.md#index-hand-off */
/* design/prepared-machine.md#M10 */
struct mesh_publication { _Alignas(64) uint32_t sends; _Atomic uint64_t argument; uint64_t device_input,device_stride; struct mesh_target targets[]; };
_Static_assert(sizeof(struct mesh_publication)==64 && _Alignof(struct mesh_publication)==64 && offsetof(struct mesh_publication,argument)==8 && offsetof(struct mesh_publication,device_input)==16 && offsetof(struct mesh_publication,device_stride)==24 && offsetof(struct mesh_publication,targets)==32,"mesh_publication");
/* design/prepared-machine.md#M13 */
struct prepared_publication { _Alignas(16) uint64_t destination; uint64_t argument,padding[2]; };
_Static_assert(sizeof(struct prepared_publication)==32 && _Alignof(struct prepared_publication)==16 && offsetof(struct prepared_publication,argument)==8,"M13");
struct mesh_port_info { _Atomic uint32_t phase,domain; _Atomic int64_t code; _Atomic uint64_t prepared; };
struct mesh_link_info { uint32_t peer; char device[32]; uint64_t bandwidth; struct mesh_port_info port; };
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M09 */
struct hdr {
  uint32_t magic,version,pgsz,block,rows,node,qps,links;
  uint32_t orders,wire_pages;
  _Atomic uint32_t depth;
  uint32_t padding;
  _Atomic uint64_t configured;
  uint64_t planes_off,arena_off,page_off,buffer_off,link_off,length_off,target_off,order_off,notice_off,data_off,length;
  uint64_t notice_bytes,target_stride,net_off;
  _Atomic uint64_t client,bridge_pid,device_client,serial,control;
  struct mesh_port_info port;
};
/* The communicator service (mesh-net.h, NCCL's network plugin ncclNet_v12_t): its tables in the region.
   A client claims a slot by CAS from MESH_NET_FREE and publishes it; the session thread of the slot's
   link serves it (mesh-flow.c net_scan).  A comm is one direction of one connection (a listen, a send or
   a receive end), its requests a ring by sequence, posted by the client and completed by the bridge.
   Communicator memory is the region's registered window, used in place: a receive lands only in memory
   registered before its queue pair was set up (mesh-flow.c net_configure), which the window is. */
#define MESH_NET_CLIENTS 32
#define MESH_NET_COMMS 256
#define MESH_NET_REQUESTS 64
#define MESH_NET_MEMORY 256
#define MESH_NET_WINDOW UINT32_MAX
enum { MESH_NET_FREE, MESH_NET_CLAIMED, MESH_NET_LISTEN, MESH_NET_CONNECTING, MESH_NET_ACCEPTABLE, MESH_NET_SEND,
       MESH_NET_RECV, MESH_NET_CLOSING, MESH_NET_FAILED };
enum { MESH_NET_IDLE, MESH_NET_POSTED, MESH_NET_ACTIVE, MESH_NET_DONE, MESH_NET_ERROR };
enum { MESH_NET_ISEND, MESH_NET_IRECV, MESH_NET_IFLUSH };
/* A request: `offset` into the region's registered window (mr MESH_NET_WINDOW), `size` the bytes sent or
   the receive's capacity, `transferred` the bytes moved once it is done. */
struct mesh_net_request {
  _Alignas(64) _Atomic uint32_t state;
  uint32_t op,mr,mr_generation;
  int32_t tag; _Atomic int32_t error;
  uint64_t sequence,offset,size;
  _Atomic uint64_t transferred;
};
_Static_assert(sizeof(struct mesh_net_request)==64,"mesh_net_request");
/* A comm: its link, its client (`owner`), the key a connect names, a receive end's listen, the other
   end's comm on the peer bridge; `posted` the client's count of requests, the rest the bridge's counts. */
struct mesh_net_comm {
  _Alignas(64) _Atomic uint32_t state;
  uint32_t generation,link,listen,listen_generation,peer,peer_generation;
  _Atomic int32_t error;
  uint64_t owner,key;
  _Atomic uint64_t posted,bytes,completions,credit_waits;
  struct mesh_net_request requests[MESH_NET_REQUESTS];
};
/* A client's allocation of arena pages [first, first+pages) of the registered window
   (mesh_net_mem_alloc), vacated by the bridge once its client has exited and no comm of it remains. */
struct mesh_net_memory { _Alignas(16) _Atomic uint64_t owner; uint32_t first,pages; };
struct mesh_net_client { _Alignas(64) _Atomic uint64_t owner; };
/* A link's live counts, both paths': SENDs that waited for their queue's frames (send_stalls), receives
   held back because their queue's frames or ring span were full, so the peer's SENDs into them waited
   for credit (receive_stalls), communicator sends announced before their receiver's irecv, which waited
   for its credit (credit_waits);
   the prepared program's SEND requests and receive records and bytes, the communicators' messages and
   bytes; the session's phase and pairings, and the device's registered window regions. */
struct mesh_net_link {
  _Alignas(64) _Atomic uint32_t phase,chunk_frames;
  _Atomic int64_t code;
  _Atomic uint64_t doorbell,sessions;
  _Atomic uint64_t send_stalls,receive_stalls,credit_waits;
  _Atomic uint64_t sends,send_bytes,receives,receive_bytes;
  _Atomic uint64_t net_sends,net_send_bytes,net_receives,net_receive_bytes;
  _Atomic uint32_t wire_regions,padding;
};
static inline struct mesh_net_link *mesh_net_links(struct hdr *m){return (struct mesh_net_link *)((char *)m+m->net_off);}
static inline struct mesh_net_client *mesh_net_clients(struct hdr *m){return (struct mesh_net_client *)(mesh_net_links(m)+m->links);}
static inline struct mesh_net_memory *mesh_net_memory(struct hdr *m){return (struct mesh_net_memory *)(mesh_net_clients(m)+MESH_NET_CLIENTS);}
static inline struct mesh_net_comm *mesh_net_comms(struct hdr *m){return (struct mesh_net_comm *)(mesh_net_memory(m)+MESH_NET_MEMORY);}
static inline uint64_t mesh_net_bytes(uint32_t links){
  return (uint64_t)links*sizeof(struct mesh_net_link)+MESH_NET_CLIENTS*sizeof(struct mesh_net_client)+
    MESH_NET_MEMORY*sizeof(struct mesh_net_memory)+MESH_NET_COMMS*sizeof(struct mesh_net_comm);
}
/* design/prepared-machine.md#M26 */
_Static_assert(sizeof(((struct hdr *)0)->control)==8 && offsetof(struct hdr,control)%8==0,"M26");
/* design/prepared-machine.md#M07 */
_Static_assert(sizeof(_Atomic uint64_t)==8 && _Alignof(_Atomic uint64_t)==8,"M07");
/* design/prepared-machine.md#M12 */
struct mesh_cancel_range { _Alignas(32) uint64_t offset; uint64_t count,padding[2]; };
struct mesh_cancellation { _Alignas(32) _Atomic uint32_t requested; uint32_t padding[7]; struct mesh_cancel_range ranges[]; };
_Static_assert(sizeof(struct mesh_cancel_range)==32 && sizeof(struct mesh_cancellation)==32 && offsetof(struct mesh_cancellation,ranges)==32,"M12");
/* design/algorithm-sources.md#meshresult */
/* design/prepared-machine.md#M12 */
static inline void mesh_cancel(struct hdr *m,struct mesh_cancellation *cancel,uint32_t link){
  _Atomic uint64_t *words=(void *)((char *)m+cancel->ranges[link].offset);
  for(uint64_t j=0;j<cancel->ranges[link].count;j++)
    if(!atomic_load_explicit(words+j,memory_order_relaxed)){
      atomic_store_explicit(&cancel->requested,1,memory_order_release);
      atomic_store_explicit(words+j,UINT64_MAX,memory_order_release);
    }
}
/* design/algorithm-sources.md#meshresult */
/* design/prepared-machine.md#M26 */
static inline void mesh_control_notify(struct hdr *m){
  atomic_fetch_add_explicit(&m->control,1,memory_order_release);
  os_sync_wake_by_address_all(&m->control,sizeof m->control,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
/* design/algorithm-sources.md#programtensor */
static inline uint32_t mesh_notice_queue(struct hdr *m,uint64_t owner,uint32_t queue){return (uint32_t)(owner>>63)*m->links+queue;}
/* design/algorithm-sources.md#programcopy */
static inline struct mesh_link_info *mesh_links(struct hdr *m){return (struct mesh_link_info *)((char *)m+m->link_off);}
/* design/algorithm-sources.md#index-hand-off */
static inline struct mesh_publication *mesh_publication_at(struct hdr *m,uint32_t row){return (struct mesh_publication *)((char *)m+m->target_off+(size_t)row*m->target_stride);}
static inline uint32_t mesh_rows(const struct hdr *m){ return m->rows; }
static inline uint32_t mesh_words(const struct hdr *m){ return (mesh_rows(m)+63)/64; }
/* design/prepared-machine.md#M09 */
static inline uint32_t mesh_blocks(const struct hdr *m){ return (uint32_t)((m->length-m->data_off)/((uint64_t)m->block*m->pgsz)); }
/* design/pages-and-functions.md#what-the-page-table-is */
static inline uint32_t mesh_arena_pages(const struct hdr *m){ return mesh_blocks(m)*m->block; }
static inline _Atomic uint64_t *mesh_plane(struct hdr *m,int plane){ return (_Atomic uint64_t*)((unsigned char*)m+m->planes_off)+(size_t)plane*mesh_words(m); }
/* design/prepared-machine.md#M01 */
/* the arena occupancy bitmap is indexed by arena page and sized by arena pages: it is not a row plane */
static inline _Atomic uint64_t *mesh_arena_bits(struct hdr *m){ return (_Atomic uint64_t*)((unsigned char*)m+m->arena_off); }
/* design/prepared-machine.md#M09 */
/* the registered window is the first wire_pages of the arena; every other page is addressable and unregistered */
static inline uint64_t mesh_wire_bytes(const struct hdr *m){ return (uint64_t)m->wire_pages*m->pgsz; }
static inline int mesh_wired(const struct hdr *m,uint64_t offset,uint64_t bytes){ return offset<=mesh_wire_bytes(m) && bytes<=mesh_wire_bytes(m)-offset; }
static inline struct mesh_page_entry *mesh_page(struct hdr *m){ return (struct mesh_page_entry*)((unsigned char*)m+m->page_off); }
/* design/algorithm-sources.md#programtensor */
static inline struct mesh_buffer *mesh_buffers(struct hdr *m){ return (struct mesh_buffer *)((char *)m+m->buffer_off); }
/* ledger D5 */
/* one index for the transfer list and for its length: both are (bank,queue,direction), both stride by the
   runtime queue count, so a client built with another MESH_QPS cannot disagree with the bridge about either */
static inline size_t mesh_order_index(const struct hdr *m,uint64_t owner,uint32_t queue,int direction){
  return (((size_t)(owner>>63)*m->links*m->qps)+queue)*2+(uint32_t)direction;
}
static inline struct mesh_transfer *mesh_transfers(struct hdr *m,uint64_t owner,uint32_t queue,int direction){ return (struct mesh_transfer*)((unsigned char*)m+m->order_off)+mesh_order_index(m,owner,queue,direction)*m->orders; }
static inline _Atomic uint32_t *mesh_order_length(struct hdr *m,uint64_t owner,uint32_t queue,int direction){ return (_Atomic uint32_t*)((unsigned char*)m+m->length_off)+mesh_order_index(m,owner,queue,direction); }
static inline unsigned char *mesh_at(struct hdr *m,uint32_t page){ return (unsigned char*)m+m->data_off+(size_t)page*m->pgsz; }
/* design/algorithm-sources.md#programcopy */
/* design/prepared-machine.md#M08 */
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

/* design/algorithm-sources.md#index-hand-off */
static inline void *mesh_events(struct hdr *m,uint32_t queue){
  return (char *)m+m->notice_off+(uint64_t)queue*m->notice_bytes;
}

/* design/algorithm-sources.md#programkernel_call */
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M08 */
/* Four independent dimensions, none of them derived from another: arena pages, page-table rows,
   transfer-list entries (orders) and the registered window.  rows was welded to pages because the
   arena bitmap lived in a row plane and the order table was rows-strided; both are now their own
   table, so a 512 GiB arena costs a 4 MiB bitmap and nothing else. */
struct mesh_geometry { uint32_t pgsz,block,pages,rows,orders,links,qps,wire_pages; };
static inline uint64_t mesh_layout(struct hdr *h,struct mesh_geometry g){
  uint32_t blocks=g.pages/g.block,arena=blocks*g.block;
  if(!g.rows)g.rows=arena;
  if(!g.orders)g.orders=g.rows;
  if(!g.wire_pages || g.wire_pages>arena)g.wire_pages=arena;
  g.wire_pages=(g.wire_pages+g.block-1)/g.block*g.block;
  if(g.wire_pages>arena)g.wire_pages=arena;
  uint32_t pgsz=g.pgsz;
  uint64_t at=(sizeof *h+pgsz-1)/pgsz*pgsz,words=((uint64_t)g.rows+63)/64;
  h->pgsz=pgsz; h->block=g.block; h->rows=g.rows; h->orders=g.orders; h->wire_pages=g.wire_pages;
  h->links=g.links; h->qps=g.qps;
  h->planes_off=at; at+=(uint64_t)MESH_PLANES*words*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->arena_off=at; at+=(((uint64_t)arena+63)/64)*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->page_off=at; at+=(uint64_t)g.rows*sizeof(struct mesh_page_entry); at=(at+pgsz-1)/pgsz*pgsz;
  h->buffer_off=at; at+=(uint64_t)g.rows*sizeof(struct mesh_buffer); at=(at+pgsz-1)/pgsz*pgsz;
  h->link_off=at; at+=(uint64_t)g.links*sizeof(struct mesh_link_info); at=(at+pgsz-1)/pgsz*pgsz;
  h->net_off=at; at+=mesh_net_bytes(g.links); at=(at+pgsz-1)/pgsz*pgsz;
  h->length_off=at; at+=(uint64_t)MESH_NOTICE_BANKS*2*g.links*g.qps*sizeof(uint32_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->target_stride=(offsetof(struct mesh_publication,targets)+(uint64_t)g.links*sizeof(struct mesh_target)+63)&~UINT64_C(63);
  h->target_off=at; at+=(uint64_t)g.rows*h->target_stride; at=(at+pgsz-1)/pgsz*pgsz;
  h->order_off=at; at+=(uint64_t)MESH_NOTICE_BANKS*2*g.links*g.qps*g.orders*sizeof(struct mesh_transfer); at=(at+pgsz-1)/pgsz*pgsz;
  h->notice_bytes=sizeof(struct mesh_tx);
  uint64_t bytes=(uint64_t)g.block*pgsz; at=(at+bytes-1)/bytes*bytes;
  h->notice_off=at; at+=(uint64_t)MESH_NOTICE_BANKS*g.links*h->notice_bytes; at=(at+bytes-1)/bytes*bytes;
  h->data_off=at; at+=(uint64_t)g.pages*pgsz;
  h->length=at; return at;
}
#endif
