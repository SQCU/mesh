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
#define MESH_VERSION 100u
#define MESH_ABSENT UINT32_MAX
/* design/collective-dependency-ledger.md#d6-paired-send-and-receive-frame-counts-match */
#define MESH_QPS 8
struct mesh_transfer { uint32_t local_row,binding,count,stride,invocation_pages,first; uint64_t bytes; };
_Static_assert(sizeof(struct mesh_transfer)==32,"M08 prepared transfer");
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_STOPPED };
/* design/algorithm-sources.md#programtensor */
enum { MESH_ROW_OWN, MESH_ROW_HOT, MESH_PAGE_OWN, MESH_FREE, MESH_PLANES };
/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M02 */
struct mesh_buffer {
  _Alignas(32) _Atomic uint64_t owner;
  _Atomic uint32_t closed;
  uint32_t pages,channel,constant;
};
_Static_assert(sizeof(struct mesh_buffer)==32 && _Alignof(struct mesh_buffer)==32,"mesh_buffer");
struct mesh_pool { _Atomic uint64_t owner; uint32_t pages; };
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
struct mesh_send {
  _Alignas(128) _Atomic uint64_t ready;
  uintptr_t pair;
  _Alignas(32) struct ibv_sge span;
  struct ibv_send_wr request;
};
struct mesh_tx { uint32_t count,slots,once,invocations; uint64_t cancel,cells; };
_Static_assert(sizeof(struct mesh_send)==256 && _Alignof(struct mesh_send)==128 &&
  offsetof(struct mesh_send,span)==32 && offsetof(struct mesh_send,request)==48 &&
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
struct mesh_link_info { uint32_t peer; char device[32]; uint64_t bandwidth; struct mesh_port_info port; _Atomic uint32_t order_length[2*MESH_NOTICE_BANKS*MESH_QPS]; };
struct hdr {
  uint32_t magic,version,pgsz,block,rows,node,qps,links;
  _Atomic uint64_t configured;
  uint64_t planes_off,page_off,buffer_off,pool_off,link_off,target_off,order_off,notice_off,data_off,length;
  uint64_t notice_bytes,target_stride;
  _Atomic uint64_t client,bridge_pid,device_client,serial,control;
  struct mesh_port_info port;
};
/* design/prepared-machine.md#M26 */
_Static_assert(sizeof(((struct hdr *)0)->control)==8 && offsetof(struct hdr,control)%8==0,"M26");
/* design/prepared-machine.md#M07 */
struct mesh_input_status { _Alignas(128) _Atomic uint64_t value; uint64_t padding[31]; };
_Static_assert(sizeof(struct mesh_input_status)==sizeof(struct mesh_send) && _Alignof(struct mesh_input_status)==128 && offsetof(struct mesh_input_status,value)==0,"M07");
/* design/prepared-machine.md#M12 */
struct mesh_cancel_range { _Alignas(32) uint64_t offset; uint64_t count,padding[2]; };
struct mesh_cancellation { _Alignas(32) _Atomic uint32_t requested; _Atomic uint32_t abandoned; uint64_t padding[3]; struct mesh_cancel_range ranges[]; };
_Static_assert(sizeof(struct mesh_cancel_range)==32 && sizeof(struct mesh_cancellation)==32 && offsetof(struct mesh_cancellation,ranges)==32,"M12");
/* design/algorithm-sources.md#meshresult */
/* design/prepared-machine.md#M12 */
static inline void mesh_cancel(struct hdr *m,struct mesh_cancellation *cancel,uint32_t link){
  atomic_store_explicit(&cancel->requested,1,memory_order_release);
  struct mesh_input_status *words=(void *)((char *)m+cancel->ranges[link].offset);
  for(uint64_t j=0;j<cancel->ranges[link].count;j++)
    if(!atomic_load_explicit(&words[j].value,memory_order_relaxed))atomic_store_explicit(&words[j].value,UINT64_MAX,memory_order_release);
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
static inline uint32_t mesh_blocks(const struct hdr *m){ return mesh_rows(m)/m->block; }
static inline _Atomic uint64_t *mesh_plane(struct hdr *m,int plane){ return (_Atomic uint64_t*)((unsigned char*)m+m->planes_off)+(size_t)plane*mesh_words(m); }
static inline struct mesh_page_entry *mesh_page(struct hdr *m){ return (struct mesh_page_entry*)((unsigned char*)m+m->page_off); }
/* design/algorithm-sources.md#programtensor */
static inline struct mesh_buffer *mesh_buffers(struct hdr *m){ return (struct mesh_buffer *)((char *)m+m->buffer_off); }
/* design/algorithm-sources.md#programtensor */
static inline struct mesh_pool *mesh_pools(struct hdr *m){return (struct mesh_pool *)((char *)m+m->pool_off);}
/* ledger D5 */
static inline struct mesh_transfer *mesh_transfers(struct hdr *m,uint64_t owner,uint32_t queue,int direction){ return (struct mesh_transfer*)((unsigned char*)m+m->order_off)+((size_t)(owner>>63)*2*m->links*m->qps+2*queue+(uint32_t)direction)*mesh_blocks(m); }
static inline _Atomic uint32_t *mesh_order_length(struct hdr *m,uint64_t owner,uint32_t queue,int direction){ return &mesh_links(m)[queue/m->qps].order_length[(owner>>63)*2*MESH_QPS+2*(queue%m->qps)+(uint32_t)direction]; }
static inline unsigned char *mesh_at(struct hdr *m,uint32_t page){ return (unsigned char*)m+m->data_off+(size_t)page*m->pgsz; }

static inline uint64_t mesh_word_mask(uint32_t first,uint32_t count,uint32_t word){
  uint32_t lo=word*64,hi=lo+64,a=first>lo?first:lo,b=first+count<hi?first+count:hi;
  if(b<=a) return 0;
  uint64_t bits=b-a==64?~UINT64_C(0):((UINT64_C(1)<<(b-a))-1);
  return bits<<(a-lo);
}
static inline void mesh_bits_set(struct hdr *m,int plane,uint32_t first,uint32_t count){
  _Atomic uint64_t *p=mesh_plane(m,plane);
  for(uint32_t w=first/64;count && w<=(first+count-1)/64;w++) atomic_fetch_or_explicit(&p[w],mesh_word_mask(first,count,w),memory_order_acq_rel);
}
static inline void mesh_bits_clear(struct hdr *m,int plane,uint32_t first,uint32_t count){
  _Atomic uint64_t *p=mesh_plane(m,plane);
  for(uint32_t w=first/64;count && w<=(first+count-1)/64;w++) atomic_fetch_and_explicit(&p[w],~mesh_word_mask(first,count,w),memory_order_acq_rel);
}

/* design/algorithm-sources.md#index-hand-off */
static inline void *mesh_events(struct hdr *m,uint32_t queue){
  return (char *)m+m->notice_off+(uint64_t)queue*m->notice_bytes;
}

/* design/algorithm-sources.md#programkernel_call */
static inline uint64_t mesh_layout(struct hdr *h,uint32_t pgsz,uint32_t block,uint32_t rows,uint32_t links,uint32_t qps){
  uint64_t at=(sizeof *h+pgsz-1)/pgsz*pgsz,words=((uint64_t)rows+63)/64,blocks=rows/block;
  h->pgsz=pgsz; h->block=block; h->rows=rows; h->links=links; h->qps=qps;
  h->planes_off=at; at+=(uint64_t)MESH_PLANES*words*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->page_off=at; at+=(uint64_t)rows*sizeof(struct mesh_page_entry); at=(at+pgsz-1)/pgsz*pgsz;
  h->buffer_off=at; at+=(uint64_t)rows*sizeof(struct mesh_buffer); at=(at+pgsz-1)/pgsz*pgsz;
  h->pool_off=at; at+=blocks*sizeof(struct mesh_pool); at=(at+pgsz-1)/pgsz*pgsz;
  h->link_off=at; at+=(uint64_t)links*sizeof(struct mesh_link_info); at=(at+pgsz-1)/pgsz*pgsz;
  h->target_stride=(offsetof(struct mesh_publication,targets)+(uint64_t)links*sizeof(struct mesh_target)+63)&~UINT64_C(63);
  h->target_off=at; at+=(uint64_t)rows*h->target_stride; at=(at+pgsz-1)/pgsz*pgsz;
  h->order_off=at; at+=(uint64_t)MESH_NOTICE_BANKS*2*links*qps*blocks*sizeof(struct mesh_transfer); at=(at+pgsz-1)/pgsz*pgsz;
  h->notice_bytes=sizeof(struct mesh_tx);
  uint64_t bytes=(uint64_t)block*pgsz; at=(at+bytes-1)/bytes*bytes;
  h->notice_off=at; at+=(uint64_t)MESH_NOTICE_BANKS*links*h->notice_bytes; at=(at+bytes-1)/bytes*bytes;
  h->data_off=at; at+=(uint64_t)rows*pgsz;
  h->length=at; return at;
}
#endif
