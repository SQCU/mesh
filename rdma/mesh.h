#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
/* design/pages-and-functions.md#block-addressing */
#define MESH_MAGIC 0x4d455348u
#define MESH_NAME "/mesh0"
#define MESH_PORT "18519"
#define MESH_MODE 0666
#define MESH_VERSION 83u
#define MESH_ABSENT UINT32_MAX
#define MESH_EVENT_ABSENT UINT64_MAX
/* design/collective-dependency-ledger.md#d6-paired-send-and-receive-frame-counts-match */
#define MESH_QPS 8
struct mesh_transfer { uint32_t local_row,binding,count,stride,pool,first; uint64_t bytes; };
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_STOPPED };
/* design/algorithm-sources.md#programtensor */
enum { MESH_ROW_OWN, MESH_ROW_HOT, MESH_PAGE_OWN, MESH_FREE, MESH_PLANES };
/* design/algorithm-sources.md#programtensor */
struct mesh_buffer {
  _Alignas(64) _Atomic uint32_t references;
  _Atomic uint32_t closed;
  uint32_t initial,pages,channel,return_index;
  _Atomic uint64_t owner;
  uint64_t return_slot;
  uint32_t completions;
  uint32_t frame;
};
_Static_assert(sizeof(struct mesh_buffer)==64 && _Alignof(struct mesh_buffer)==64,"mesh_buffer");
struct mesh_pool { _Atomic uint64_t owner; uint32_t pages; };
/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M10 */
struct mesh_page_entry {
  _Alignas(16) _Atomic uint64_t mapping;
  _Atomic uintptr_t address;
  _Atomic uint64_t device,stamp;
};
_Static_assert(sizeof(struct mesh_page_entry)==32 && _Alignof(struct mesh_page_entry)==16 &&
  offsetof(struct mesh_page_entry,address)==8 && offsetof(struct mesh_page_entry,device)==16 &&
  offsetof(struct mesh_page_entry,stamp)==24,"mesh_page_entry");
/* design/collective-dependency-ledger.md#d5-receive-consumption-has-per-queue-fifo-order */
enum { MESH_SEND, MESH_RECEIVE };
#define MESH_COMPUTE_THREADS 8
#define MESH_NOTICE_BANKS 2
/* design/algorithm-sources.md#index-hand-off */
struct mesh_stream { _Alignas(128) uint32_t position; uint32_t mask; _Atomic uint64_t slots[]; };
_Static_assert(sizeof(struct mesh_stream)==128 && _Alignof(struct mesh_stream)==128 && offsetof(struct mesh_stream,slots)==8,"mesh_stream");
/* design/prepared-machine.md#M04 */
struct mesh_send { _Alignas(32) _Atomic uint64_t ready; uintptr_t pair,request; uint64_t reserved; };
struct mesh_tx { uint32_t count,slots,once; struct mesh_send cells[]; };
_Static_assert(sizeof(struct mesh_send)==32 && _Alignof(struct mesh_send)==32 && offsetof(struct mesh_tx,cells)==32,"M04");
struct mesh_target { uint64_t stream; uint32_t count; };
_Static_assert(sizeof(struct mesh_target)==16,"mesh_target");
/* design/algorithm-sources.md#index-hand-off */
struct mesh_publication { _Alignas(64) uint32_t sends; uint32_t uses,row; struct mesh_target targets[]; };
_Static_assert(sizeof(struct mesh_publication)==64 && _Alignof(struct mesh_publication)==64 && offsetof(struct mesh_publication,targets)==16,"mesh_publication");
struct mesh_events {
  uint32_t count;
  _Alignas(128) unsigned char streams[];
};
_Static_assert(sizeof(struct mesh_events)==128 && offsetof(struct mesh_events,streams)==128,"mesh_events header");
/* design/prepared-machine.md#M18 */
struct mesh_arrival { _Alignas(16) uint32_t call; uint32_t mask; _Atomic uint64_t stamp; };
_Static_assert(sizeof(struct mesh_arrival)==16 && _Alignof(struct mesh_arrival)==16 && offsetof(struct mesh_arrival,stamp)==8,"M18");
struct mesh_event_input { _Atomic uint64_t *slots; uint32_t position,mask; };
struct mesh_event_reader { struct mesh_event_input *inputs; uint32_t count,cursor; };
_Static_assert(sizeof(struct mesh_event_input)==16 && sizeof(struct mesh_event_reader)==16,"mesh_event_reader");
_Static_assert(sizeof(_Atomic uint64_t)==8 && __atomic_always_lock_free(8,0),"mesh_event slot");
struct mesh_port_info { _Atomic uint32_t phase,domain; _Atomic int64_t code; _Atomic uint64_t prepared; };
/* design/algorithm-sources.md#meshresult */
enum { MESH_RESULT_SUCCESS, MESH_RESULT_LINK, MESH_RESULT_FUNCTION, MESH_RESULT_BUSY };
#define MESH_RESULT(kind,id,code) ((uint64_t)(kind)<<62|(uint64_t)(id)<<32|(uint32_t)(code))
struct mesh_status { uint64_t value,completed; };
_Static_assert(sizeof(struct mesh_status)==16 && __atomic_always_lock_free(sizeof(struct mesh_status),0),"mesh_status lock-free snapshot");
/* design/prepared-machine.md#M42 */
struct mesh_instance { _Alignas(32) _Atomic(struct mesh_status) status; _Atomic uint32_t remaining,invocation; uint32_t count; };
_Static_assert(sizeof(struct mesh_instance)==32 && _Alignof(struct mesh_instance)==32,"mesh_instance");
struct mesh_link_info { uint32_t peer; char device[32]; uint64_t bandwidth; struct mesh_port_info port; _Atomic uint32_t order_length[2*MESH_NOTICE_BANKS*MESH_QPS]; };
struct hdr {
  uint32_t magic,version,pgsz,block,rows,node,qps,links;
  _Atomic uint64_t configured;
  uint64_t planes_off,page_off,buffer_off,pool_off,link_off,target_off,order_off,notice_off,instance_off,data_off,length;
  uint64_t notice_bytes,target_stride;
  uint32_t instance_count[MESH_NOTICE_BANKS];
  _Atomic(struct mesh_status) result[MESH_NOTICE_BANKS];
  _Atomic uint64_t client,bridge_pid,device_client,serial,retired;
  struct mesh_port_info port;
};
/* design/algorithm-sources.md#meshresult */
static inline struct mesh_instance *mesh_instances(struct hdr *m,uint64_t owner){return (struct mesh_instance *)((char *)m+m->instance_off)+(owner>>63)*m->rows;}
/* design/algorithm-sources.md#meshresult */
static inline void mesh_result_conclude(_Atomic(struct mesh_status) *destination,uint64_t status){
  struct mesh_status previous=atomic_load_explicit(destination,memory_order_relaxed);
  do {
    if(previous.value>>62!=MESH_RESULT_SUCCESS && previous.value>>62!=MESH_RESULT_BUSY)return;
    struct mesh_status next={status,previous.completed};
    if(status>>62==MESH_RESULT_SUCCESS)next.completed=status>>32?UINT64_MAX:(uint64_t)(uint32_t)status+1;
    if(atomic_compare_exchange_strong_explicit(destination,&previous,next,memory_order_release,memory_order_relaxed))return;
  } while(status>>62!=MESH_RESULT_SUCCESS);
}
/* design/algorithm-sources.md#meshresult */
static inline void mesh_instance_release(struct mesh_instance *instances,uint32_t count){
  for(uint32_t i=0;i<count;i++){
    struct mesh_instance *instance=&instances[i];
    if(atomic_fetch_sub_explicit(&instance->remaining,1,memory_order_acq_rel)!=1)continue;
    atomic_store_explicit(&instance->remaining,instance->count,memory_order_relaxed);
    mesh_result_conclude(&instance->status,MESH_RESULT(MESH_RESULT_SUCCESS,!instance->count,atomic_load_explicit(&instance->invocation,memory_order_relaxed)));
  }
}

/* design/algorithm-sources.md#programtensor */
static inline uint32_t mesh_notice_queue(struct hdr *m,uint64_t owner,uint32_t queue){return (uint32_t)(owner>>63)*(m->links*(m->qps+1)+2*MESH_COMPUTE_THREADS)+queue;}
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
void mesh_buffer_release(struct hdr *,uint32_t row);
/* design/algorithm-sources.md#programtensor */
static inline void mesh_buffer_reset(struct hdr *m,uint32_t row){
  struct mesh_buffer *buffer=&mesh_buffers(m)[row];
  atomic_store_explicit(&buffer->references,buffer->initial,memory_order_relaxed);
  atomic_store_explicit(&mesh_page(m)[row].stamp,0,memory_order_relaxed);
}
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
static inline struct mesh_events *mesh_events(struct hdr *m,uint32_t queue){
  return (struct mesh_events *)((char *)m+m->notice_off+(uint64_t)queue*m->notice_bytes);
}
/* design/algorithm-sources.md#index-hand-off */
static inline size_t mesh_stream_bytes(uint32_t capacity){return (offsetof(struct mesh_stream,slots)+(size_t)capacity*sizeof(uint64_t)+127)&~(size_t)127;}
/* design/algorithm-sources.md#index-hand-off */
static inline uint64_t mesh_event_bind(struct hdr *m,uint32_t queue){
  struct mesh_events *events=mesh_events(m,queue);
  struct mesh_stream *stream=(struct mesh_stream *)(events->streams+(size_t)events->count++*sizeof(struct mesh_stream));
  uint64_t offset=(uint64_t)((char *)stream-(char *)m);
  stream->position=stream->mask=0;atomic_store_explicit(stream->slots,0,memory_order_relaxed);
  return offset;
}
/* design/algorithm-sources.md#index-hand-off */
static inline void mesh_event_push(struct hdr *m,uint64_t slot,uint32_t index,uint32_t invocation){
  struct mesh_stream *stream=(struct mesh_stream *)((char *)m+slot);
  uint32_t at=stream->position++&stream->mask;
  atomic_store_explicit(stream->slots+at,((uint64_t)invocation<<32)|(index+1),memory_order_release);
}
/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M18 */
/* design/algorithm-sources.md#programkernel_call */
static inline __attribute__((always_inline)) void mesh_publish(struct hdr *m,const struct mesh_target *targets,uint32_t sends,uint32_t uses,struct mesh_page_entry *entry,uint64_t stamp){
  for(uint32_t i=0;i<sends;i++){
    struct mesh_send *cells=(void *)((char *)m+targets[i].stream);
    for(uint32_t k=0;k<targets[i].count;k++)atomic_store_explicit(&cells[k].ready,1,memory_order_release);
  }
  atomic_store_explicit(&entry->stamp,stamp,memory_order_release);
  for(uint32_t i=sends,end=sends+uses;i<end;i++){
    struct mesh_arrival *cells=(void *)((char *)m+targets[i].stream);
    for(uint32_t k=0;k<targets[i].count;k++)atomic_store_explicit(&cells[k].stamp,stamp,memory_order_release);
  }
}
/* design/algorithm-sources.md#index-hand-off */
static inline uint64_t mesh_event_take(struct mesh_event_reader *reader){
  for(uint32_t visited=0;visited<reader->count;visited++){
    struct mesh_event_input *input=&reader->inputs[reader->cursor++];
    if(reader->cursor==reader->count)reader->cursor=0;
    _Atomic uint64_t *slot=&input->slots[input->position&input->mask];
    uint64_t value=atomic_load_explicit(slot,memory_order_acquire);
    if(!(uint32_t)value)continue;
    atomic_store_explicit(slot,0,memory_order_relaxed);
    input->position++;
    return value-1;
  }
  return MESH_EVENT_ABSENT;
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
  h->target_stride=(offsetof(struct mesh_publication,targets)+((uint64_t)links+MESH_COMPUTE_THREADS)*sizeof(struct mesh_target)+63)&~UINT64_C(63);
  h->target_off=at; at+=(uint64_t)rows*h->target_stride; at=(at+pgsz-1)/pgsz*pgsz;
  h->order_off=at; at+=(uint64_t)MESH_NOTICE_BANKS*2*links*qps*blocks*sizeof(struct mesh_transfer); at=(at+pgsz-1)/pgsz*pgsz;
  h->notice_bytes=sizeof(struct mesh_events)+2*(uint64_t)rows*sizeof(struct mesh_stream);
  uint64_t bytes=(uint64_t)block*pgsz; at=(at+bytes-1)/bytes*bytes;
  h->notice_off=at; at+=(uint64_t)MESH_NOTICE_BANKS*(links*(qps+1)+2*MESH_COMPUTE_THREADS)*h->notice_bytes; at=(at+bytes-1)/bytes*bytes;
  h->instance_off=at; at+=(uint64_t)MESH_NOTICE_BANKS*rows*sizeof(struct mesh_instance); at=(at+bytes-1)/bytes*bytes;
  h->data_off=at; at+=(uint64_t)rows*pgsz;
  h->length=at; return at;
}
#endif
