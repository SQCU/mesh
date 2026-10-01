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
/* 104: d6dceac's prepared transfers, one client at a time.  120: one bridge a node serving every process at once
   (operator, 2026-10-01: "LITERALLY GLOBAL ACCESS TO ANY SECTION OF THE RDMA-ACCESSIBLE REGION FOR ALL POSSIBLE
   PROGRAMS"): any number of processes map the region and allocate in it, each prepared program a session of its own
   (mesh_session, paired with the peer's session of the same key), and any thread of any process writes a stripe to
   the peer's same offset through a request ring (mesh_requests).  105-119 are not reused. */
#define MESH_VERSION 120u
/* prepared sessions served at once, request rings registered at once */
#define MESH_SESSIONS 16
#define MESH_RINGS 256
#define MESH_ABSENT UINT32_MAX
/* design/collective-dependency-ledger.md#d6-paired-send-and-receive-frame-counts-match */
#define MESH_QPS 8
struct mesh_transfer { uint32_t local_row,binding,count,stride,invocation_pages,first; uint64_t bytes; };
_Static_assert(sizeof(struct mesh_transfer)==32,"M08 prepared transfer");
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_STOPPED };
/* design/algorithm-sources.md#programtensor */
enum { MESH_ROW_OWN, MESH_PLANES };
/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M02 */
struct mesh_buffer {
  _Alignas(32) uint32_t pages;
  uint32_t constant;
  uint64_t padding[3];
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
/* design/prepared-machine.md#M04 */
struct mesh_send {
  _Alignas(128) _Atomic uint64_t ready;
  uintptr_t pair;
  _Alignas(32) struct ibv_sge span;
  struct ibv_send_wr request;
};
struct mesh_tx { uint32_t count,slots,once,invocations; uint64_t cancel,cells; uint64_t padding[4]; };
_Static_assert(sizeof(struct mesh_send)==256 && _Alignof(struct mesh_send)==128 &&
  offsetof(struct mesh_send,span)==32 && offsetof(struct mesh_send,request)==48 &&
  offsetof(struct mesh_send,request.send_flags)+sizeof(unsigned int)<=128,"M04/M29");
_Static_assert(offsetof(struct mesh_tx,cancel)==16 && offsetof(struct mesh_tx,cells)==24 && sizeof(struct mesh_tx)==64,"M04/M12");
/* A stripe request (mesh.h's request rings; mesh-call.h mesh_write): the `bytes` bytes at byte `offset` of the
   registered window written to the same offset of the window of the peer on link `link`, and `value` stored into the
   word at region offset `word` (0: none) on both nodes as the RDMA driver completes it: the writer's when its last SEND
   completes, the peer's when its last RECV does.  Request k of a ring is its entry k mod entries once that entry's
   `ready` is k + 1 (stored last, release) by whoever writes it (a thread of any process, or a GPU).  `taken` is how many
   the bridge has taken (it only grows): an entry is written again only once the request it held is taken.  `head` is
   the next request's number for host writers (mesh_write takes it atomically). */
struct mesh_request { _Alignas(64) _Atomic uint64_t ready; uint64_t offset,bytes,word,value; uint32_t link,padding; uint64_t reserved[2]; };
struct mesh_requests { _Alignas(128) _Atomic uint64_t taken; uint64_t entries; _Alignas(64) _Atomic uint64_t head; uint64_t reserved[7]; struct mesh_request request[]; };
_Static_assert(sizeof(struct mesh_request)==64 && offsetof(struct mesh_requests,head)==64 && offsetof(struct mesh_requests,request)==128,"requests");
/* The bridge's descriptor of a stripe on its link's control queue pair: where the peer posts its receives, and the
   word and value its last one stores. */
struct mesh_descriptor { _Alignas(64) uint64_t offset; uint64_t bytes,word,value,reserved[4]; };
_Static_assert(sizeof(struct mesh_descriptor)==64,"descriptor");
#define MESH_DESCRIPTORS 256
/* A prepared session (mesh-call.h mesh_transfers_start): the process whose program it is (0: free; the bridge watches
   it exit), the key the peer's session pairs on, the client's request and the bridge's state, the ring depth. */
enum { MESH_REQUEST_NONE, MESH_REQUEST_START, MESH_REQUEST_STOP };
struct mesh_session { _Alignas(128) _Atomic uint64_t pid; _Atomic uint64_t key; _Atomic uint32_t request,served,depth,padding; };
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
struct mesh_link_info { uint32_t peer; char device[32]; uint64_t bandwidth,extent; struct mesh_port_info port; };
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M09 */
/* zone_pages: the window's first pages, the bridge's own (each link's descriptors), never handed out */
struct hdr {
  uint32_t magic,version,pgsz,block,rows,node,qps,links;
  uint32_t orders,wire_pages,zone_pages,padding;
  uint64_t planes_off,arena_off,page_off,buffer_off,link_off,length_off,target_off,order_off,notice_off,data_off,length;
  uint64_t notice_bytes,target_stride,session_off,port_off,ring_off;
  _Atomic uint64_t bridge_pid,control;
  struct mesh_port_info port;
};
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
static inline uint32_t mesh_notice_queue(struct hdr *m,uint32_t session,uint32_t link){return session*m->links+link;}
static inline struct mesh_session *mesh_sessions(struct hdr *m){return (struct mesh_session *)((char *)m+m->session_off);}
/* session `session`'s port on link `link`: its pairing's phase and failure (the bridge's) */
static inline struct mesh_port_info *mesh_session_port(struct hdr *m,uint32_t session,uint32_t link){
  return (struct mesh_port_info *)((char *)m+m->port_off)+(size_t)session*m->links+link;
}
static inline _Atomic uint64_t *mesh_rings(struct hdr *m){return (_Atomic uint64_t *)((char *)m+m->ring_off);}
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
/* one index for the transfer list and for its length: both are (session,queue,direction), both stride by the
   runtime queue count, so a client built with another MESH_QPS cannot disagree with the bridge about either */
static inline size_t mesh_order_index(const struct hdr *m,uint32_t session,uint32_t queue,int direction){
  return (((size_t)session*m->links*m->qps)+queue)*2+(uint32_t)direction;
}
static inline struct mesh_transfer *mesh_transfers(struct hdr *m,uint32_t session,uint32_t queue,int direction){ return (struct mesh_transfer*)((unsigned char*)m+m->order_off)+mesh_order_index(m,session,queue,direction)*m->orders; }
static inline _Atomic uint32_t *mesh_order_length(struct hdr *m,uint32_t session,uint32_t queue,int direction){ return (_Atomic uint32_t*)((unsigned char*)m+m->length_off)+mesh_order_index(m,session,queue,direction); }
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
  h->session_off=at; at+=(uint64_t)MESH_SESSIONS*sizeof(struct mesh_session); at=(at+pgsz-1)/pgsz*pgsz;
  h->port_off=at; at+=(uint64_t)MESH_SESSIONS*g.links*sizeof(struct mesh_port_info); at=(at+pgsz-1)/pgsz*pgsz;
  h->ring_off=at; at+=(uint64_t)MESH_RINGS*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->length_off=at; at+=(uint64_t)MESH_SESSIONS*2*g.links*g.qps*sizeof(uint32_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->target_stride=(offsetof(struct mesh_publication,targets)+(uint64_t)g.links*sizeof(struct mesh_target)+63)&~UINT64_C(63);
  h->target_off=at; at+=(uint64_t)g.rows*h->target_stride; at=(at+pgsz-1)/pgsz*pgsz;
  h->order_off=at; at+=(uint64_t)MESH_SESSIONS*2*g.links*g.qps*g.orders*sizeof(struct mesh_transfer); at=(at+pgsz-1)/pgsz*pgsz;
  h->notice_bytes=sizeof(struct mesh_tx);
  uint64_t bytes=(uint64_t)g.block*pgsz; at=(at+bytes-1)/bytes*bytes;
  h->notice_off=at; at+=(uint64_t)MESH_SESSIONS*g.links*h->notice_bytes; at=(at+bytes-1)/bytes*bytes;
  h->data_off=at; at+=(uint64_t)g.pages*pgsz;
  /* the bridge's zone: each link's descriptors, received and sent, in whole blocks at the window's start */
  uint64_t zone=(uint64_t)g.links*2*MESH_DESCRIPTORS*sizeof(struct mesh_descriptor);
  h->zone_pages=(uint32_t)((zone+bytes-1)/bytes*g.block);
  h->length=at; return at;
}
#endif
