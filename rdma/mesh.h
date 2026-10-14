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
#define MESH_MODE 0666
#define MESH_VERSION 107u
#define MESH_ABSENT UINT32_MAX
/* design/bridge-and-ipc.md#region */
#define MESH_PORTS 8
#define MESH_CLIENTS 64
#define MESH_QUEUES 4
#define MESH_ORDERS 4096
#define MESH_RETIRING (UINT64_C(1)<<63)
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
/* design/bridge-and-ipc.md#region: a client's program on one link: its SEND cells, its cancellation words, its
   transfer lists (MESH_QUEUES queues each way, MESH_ORDERS entries a list), its trace (0: none) and the peer node they
   were bound to */
struct mesh_tx { uint32_t count,slots,once,invocations; uint64_t cancel,cells,orders,trace; uint32_t peer,padding[3]; };
/* design/prepared-machine.md#M27: a program's trace on one link, each way: a SEND observed (its cell's region offset,
   the times its producer released it, its post began and ended) or a receive landed (its record's index x 128, the
   times it was polled, published and its next posted), on the bridge's CLOCK_UPTIME_RAW, `count` of `capacity` */
struct mesh_trace { _Alignas(32) uint64_t identity; uint64_t begin,middle,end; };
struct mesh_trace_ring { _Alignas(32) _Atomic uint64_t count; uint64_t capacity,padding[2]; struct mesh_trace records[]; };
_Static_assert(sizeof(struct mesh_send)==256 && _Alignof(struct mesh_send)==128 &&
  offsetof(struct mesh_send,span)==32 && offsetof(struct mesh_send,request)==48 && offsetof(struct mesh_send,queue)+4<=256 &&
  offsetof(struct mesh_send,request.send_flags)+sizeof(unsigned int)<=128,"M04/M29");
_Static_assert(offsetof(struct mesh_tx,cancel)==16 && offsetof(struct mesh_tx,cells)==24 && sizeof(struct mesh_tx)==64,"M04/M12");
_Static_assert(sizeof(struct mesh_trace)==32 && sizeof(struct mesh_trace_ring)==32,"M27");
struct mesh_target { uint64_t stream; uint32_t count,stride; };
_Static_assert(sizeof(struct mesh_target)==16,"mesh_target");
/* design/algorithm-sources.md#index-hand-off */
/* design/prepared-machine.md#M10 */
struct mesh_publication { _Alignas(64) uint32_t sends; _Atomic uint64_t argument; uint64_t device_input,device_stride; struct mesh_target targets[]; };
_Static_assert(sizeof(struct mesh_publication)==64 && _Alignof(struct mesh_publication)==64 && offsetof(struct mesh_publication,argument)==8 && offsetof(struct mesh_publication,device_input)==16 && offsetof(struct mesh_publication,device_stride)==24 && offsetof(struct mesh_publication,targets)==32,"mesh_publication");
/* design/prepared-machine.md#M13 */
struct prepared_publication { _Alignas(16) uint64_t destination; uint64_t argument,padding[2]; };
_Static_assert(sizeof(struct prepared_publication)==32 && _Alignof(struct prepared_publication)==16 && offsetof(struct prepared_publication,argument)==8,"M13");
/* design/bridge-and-ipc.md#region: a client's program on one link (phase, the failure's code and domain, `prepared`
   the owner once the bridge paired or failed it) and a link of the node (its peer node while paired, the pairings
   it made) */
struct mesh_port_info { _Atomic uint32_t phase,domain; _Atomic int64_t code; _Atomic uint64_t prepared; };
struct mesh_link_info { _Atomic uint32_t peer,phase; _Atomic int64_t code; _Atomic uint64_t bandwidth,connection; char device[32]; };
/* design/bridge-and-ipc.md#region: a client slot: its attach (`owner`, 0 vacant), its program once started
   (`configured`, the owner), the execution it belongs to (`key`, the same on every member), its ring depth, whether it
   is cyclic, and each link's transfer list lengths, program and status */
struct mesh_client {
  _Alignas(64) _Atomic uint64_t owner;
  _Atomic uint64_t configured;
  uint64_t key;
  _Atomic uint32_t depth,cyclic;
  _Atomic uint32_t length[MESH_PORTS*MESH_QUEUES*2];
  struct mesh_tx tx[MESH_PORTS];
  struct mesh_port_info port[MESH_PORTS];
};
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M09 */
struct hdr {
  uint32_t magic,version,pgsz,block,rows,node,links,clients,wire_pages,padding;
  uint64_t planes_off,arena_off,page_off,buffer_off,link_off,client_off,target_off,data_off,length,target_stride,boot;
  _Atomic uint64_t bridge_pid,serial,control;
};
/* design/prepared-machine.md#M26 */
_Static_assert(sizeof(((struct hdr *)0)->control)==8 && offsetof(struct hdr,control)%8==0,"M26");
/* design/prepared-machine.md#M07 */
_Static_assert(sizeof(_Atomic uint64_t)==8 && _Alignof(_Atomic uint64_t)==8,"M07");
/* design/prepared-machine.md#M12 */
struct mesh_cancel_range { _Alignas(32) uint64_t offset; uint64_t count,padding[2]; };
struct mesh_cancellation { _Alignas(32) _Atomic uint32_t requested; uint32_t cyclic,padding[6]; struct mesh_cancel_range ranges[]; };
_Static_assert(sizeof(struct mesh_cancel_range)==32 && sizeof(struct mesh_cancellation)==32 && offsetof(struct mesh_cancellation,ranges)==32,"M12");
/* design/algorithm-sources.md#meshresult */
/* design/prepared-machine.md#M12 */
/* design/prepared-machine.md#M30: a cyclic program's landed word holds 1 plus its cycle, which a reader of a later
   cycle still waits past, so every word of the link is cancelled */
static inline void mesh_cancel(struct hdr *m,struct mesh_cancellation *cancel,uint32_t link){
  _Atomic uint64_t *words=(void *)((char *)m+cancel->ranges[link].offset);
  for(uint64_t j=0;j<cancel->ranges[link].count;j++){
    const uint64_t word=atomic_load_explicit(words+j,memory_order_relaxed);
    if(!word || (cancel->cyclic && word!=UINT64_MAX)){
      atomic_store_explicit(&cancel->requested,1,memory_order_release);
      atomic_store_explicit(words+j,UINT64_MAX,memory_order_release);
    }
  }
}
/* design/algorithm-sources.md#meshresult */
/* design/prepared-machine.md#M26 */
static inline void mesh_control_notify(struct hdr *m){
  atomic_fetch_add_explicit(&m->control,1,memory_order_release);
  os_sync_wake_by_address_all(&m->control,sizeof m->control,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
/* design/algorithm-sources.md#programcopy */
static inline struct mesh_link_info *mesh_links(struct hdr *m){return (struct mesh_link_info *)((char *)m+m->link_off);}
/* design/bridge-and-ipc.md#region: an owner word is its generation, slot and process: (generation << 38) | (slot << 32) | pid */
static inline struct mesh_client *mesh_client_at(struct hdr *m,uint64_t owner){return (struct mesh_client *)((char *)m+m->client_off)+((owner>>32)&(MESH_CLIENTS-1));}
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
/* ledger D5: a client's lists of one link, queue and direction: channel = link x MESH_QUEUES + queue */
static inline struct mesh_transfer *mesh_transfers(struct hdr *m,uint64_t owner,uint32_t channel,int direction){
  return (struct mesh_transfer*)((unsigned char*)m+mesh_client_at(m,owner)->tx[channel/MESH_QUEUES].orders)+((size_t)(channel%MESH_QUEUES)*2+(uint32_t)direction)*MESH_ORDERS;
}
static inline _Atomic uint32_t *mesh_order_length(struct hdr *m,uint64_t owner,uint32_t channel,int direction){ return mesh_client_at(m,owner)->length+(size_t)channel*2+(uint32_t)direction; }
static inline uint32_t mesh_link_transfers(struct hdr *m,uint64_t owner,uint32_t link){
  uint32_t transfers=0;
  for(uint32_t q=link*MESH_QUEUES;q<(link+1)*MESH_QUEUES;q++)for(int d=0;d<2;d++)transfers+=atomic_load_explicit(mesh_order_length(m,owner,q,d),memory_order_acquire);
  return transfers;
}
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

/* design/algorithm-sources.md#programkernel_call */
/* design/prepared-machine.md#M01 */
/* design/bridge-and-ipc.md#region: the node's region, laid out from the node's memory: `pages` arena pages of `block`
   pages a block, a page-table row a block, the node's links and the client slots; every page registers on demand */
static inline uint64_t mesh_layout(struct hdr *h,uint32_t pgsz,uint32_t block,uint32_t pages){
  uint32_t arena=pages/block*block,rows=arena/block;
  uint64_t at=(sizeof *h+pgsz-1)/pgsz*pgsz,words=((uint64_t)rows+63)/64;
  h->pgsz=pgsz; h->block=block; h->rows=rows; h->wire_pages=arena; h->links=MESH_PORTS; h->clients=MESH_CLIENTS;
  h->planes_off=at; at+=(uint64_t)MESH_PLANES*words*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->arena_off=at; at+=(((uint64_t)arena+63)/64)*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->page_off=at; at+=(uint64_t)rows*sizeof(struct mesh_page_entry); at=(at+pgsz-1)/pgsz*pgsz;
  h->buffer_off=at; at+=(uint64_t)rows*sizeof(struct mesh_buffer); at=(at+pgsz-1)/pgsz*pgsz;
  h->link_off=at; at+=(uint64_t)MESH_PORTS*sizeof(struct mesh_link_info); at=(at+pgsz-1)/pgsz*pgsz;
  h->client_off=at; at+=(uint64_t)MESH_CLIENTS*sizeof(struct mesh_client); at=(at+pgsz-1)/pgsz*pgsz;
  h->target_stride=(offsetof(struct mesh_publication,targets)+(uint64_t)MESH_PORTS*sizeof(struct mesh_target)+63)&~UINT64_C(63);
  h->target_off=at; at+=(uint64_t)rows*h->target_stride;
  uint64_t bytes=(uint64_t)block*pgsz; at=(at+bytes-1)/bytes*bytes;
  h->data_off=at; at+=(uint64_t)arena*pgsz;
  h->length=at; return at;
}
#endif
