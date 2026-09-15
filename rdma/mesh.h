#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
/* design/pages-and-functions.md#what-the-page-table-is
   Transport: design/collective-dependency-ledger.md, dependencies D1-D14. */
#define MESH_MAGIC 0x4d455348u
#define MESH_NAME "/mesh0"
#define MESH_PORT "18519"
#define MESH_MODE 0666
#define MESH_VERSION 39u
#define MESH_ABSENT UINT32_MAX
/* ledger D6: "A maximum of 10 unreliable connection (UC) queue pairs" */
#define MESH_QPS 8
struct mesh_transfer { uint32_t local_row,binding,bytes,count,stride; };
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_STOPPED };
/* design/algorithm-sources.md#programtensor */
enum { MESH_PRESENT, MESH_CONSTANT, MESH_ROW_OWN, MESH_ROW_HOT, MESH_PAGE_OWN, MESH_PLANES };
/* design/algorithm-sources.md#programtensor */
enum { MESH_BUFFER_PRODUCER=1, MESH_BUFFER_SEALED=2, MESH_BUFFER_QUEUED=4, MESH_BUFFER_CLOSED=8, MESH_BUFFER_RECLAIMED=16 };
#define MESH_BUFFER_FLAG(flag) ((uint64_t)(flag)<<32)
struct mesh_buffer { _Atomic uint64_t ownership; uint32_t first,pages,next; _Atomic uint32_t uses; uint64_t owner; };
/* ledger D5: one posting order per queue pair and direction */
enum { MESH_SEND, MESH_RECEIVE };
#define MESH_COMPUTE_THREADS 8
#define MESH_COMPUTE_MASK ((UINT32_C(1)<<MESH_COMPUTE_THREADS)-1)
enum { MESH_NOTICE_SEND, MESH_NOTICE_COMPUTE, MESH_NOTICE_QUEUES=MESH_NOTICE_COMPUTE+MESH_COMPUTE_THREADS };
#define MESH_NOTICE_BANKS 2
struct mesh_notice { uint32_t next; };
struct mesh_port_info { _Atomic uint32_t phase,domain; _Atomic int64_t code; };
struct hdr {
  uint32_t magic,version,pgsz,block,rows,node,qps;
  _Atomic uint64_t configured;
  uint64_t planes_off,page_off,buffer_off,backing_off,order_off,notice_off,data_off,length;
  _Atomic uint64_t client,bridge_pid,device_client,serial;
  _Atomic uint32_t reclaim_head;
  _Atomic uint32_t order_length[2*MESH_QPS];
  _Atomic uint32_t notice_head[MESH_NOTICE_BANKS*MESH_NOTICE_QUEUES];
  uint32_t send_bytes[MESH_NOTICE_BANKS][MESH_QPS];
  struct mesh_port_info port;
};
void mesh_notify(struct hdr *,uint32_t first,uint32_t count);
/* design/algorithm-sources.md#programtensor */
static inline uint32_t mesh_notice_queue(uint64_t owner,uint32_t queue){return (uint32_t)(owner>>63)*MESH_NOTICE_QUEUES+queue;}
static inline uint32_t mesh_rows(const struct hdr *m){ return m->rows; }
static inline uint32_t mesh_words(const struct hdr *m){ return (mesh_rows(m)+63)/64; }
static inline uint32_t mesh_blocks(const struct hdr *m){ return mesh_rows(m)/m->block; }
static inline _Atomic uint64_t *mesh_plane(struct hdr *m,int plane){ return (_Atomic uint64_t*)((unsigned char*)m+m->planes_off)+(size_t)plane*mesh_words(m); }
static inline _Atomic uint32_t *mesh_page(struct hdr *m){ return (_Atomic uint32_t*)((unsigned char*)m+m->page_off); }
/* design/algorithm-sources.md#programtensor */
static inline struct mesh_buffer *mesh_buffers(struct hdr *m){ return (struct mesh_buffer *)((char *)m+m->buffer_off); }
/* design/algorithm-sources.md#programtensor */
static inline uint32_t *mesh_backing(struct hdr *m){ return (uint32_t *)((char *)m+m->backing_off); }
void mesh_buffer_release(struct hdr *,uint32_t first,uint32_t count);
void mesh_buffer_produced(struct hdr *,uint32_t first,uint32_t count);
/* ledger D5 */
static inline struct mesh_transfer *mesh_transfers(struct hdr *m,uint32_t queue,int direction){ return (struct mesh_transfer*)((unsigned char*)m+m->order_off)+(size_t)(2*queue+(uint32_t)direction)*mesh_blocks(m); }
static inline _Atomic uint32_t *mesh_order_length(struct hdr *m,uint32_t queue,int direction){ return &m->order_length[2*queue+(uint32_t)direction]; }
static inline unsigned char *mesh_at(struct hdr *m,uint32_t page){ return (unsigned char*)m+m->data_off+(size_t)page*m->pgsz; }
/* design/algorithm-sources.md#programcopy */
static inline uint32_t *mesh_tag(struct hdr *m,uint32_t page,uint32_t bytes){ return (uint32_t *)(mesh_at(m,page)+bytes-sizeof(uint32_t)); }
/* design/algorithm-sources.md#programcopy */
static inline uint32_t mesh_message_bytes(uint32_t bytes){return (bytes+sizeof(uint32_t)+4095u)/4096u*4096u;}

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
static inline int mesh_bits_all(struct hdr *m,int plane,uint32_t first,uint32_t count){
  _Atomic uint64_t *p=mesh_plane(m,plane);
  for(uint32_t w=first/64;count && w<=(first+count-1)/64;w++){ uint64_t k=mesh_word_mask(first,count,w); if((atomic_load_explicit(&p[w],memory_order_acquire)&k)!=k) return 0; }
  return 1;
}
static inline int mesh_bit(struct hdr *m,int plane,uint32_t index){
  return (int)((atomic_load_explicit(&mesh_plane(m,plane)[index/64],memory_order_acquire)>>(index%64))&1);
}

/* design/algorithm-sources.md#programcopy */
static inline void mesh_receive_complete(struct hdr *m,uint32_t row,uint32_t page){
  uint32_t previous=atomic_load_explicit(&mesh_page(m)[row],memory_order_relaxed),displaced=mesh_backing(m)[page];
  for(uint32_t i=0;i<m->block;i++){
    atomic_store_explicit(&mesh_page(m)[displaced+i],previous+i,memory_order_relaxed);
    mesh_backing(m)[previous+i]=displaced;
    mesh_backing(m)[page+i]=row;
  }
  for(uint32_t i=0;i<m->block;i++)atomic_store_explicit(&mesh_page(m)[row+i],page+i,memory_order_relaxed);
  struct mesh_buffer *buffer=&mesh_buffers(m)[row];
  uint32_t queues=atomic_load_explicit(&buffer->uses,memory_order_relaxed)>>MESH_COMPUTE_THREADS;
  while(queues){
    uint32_t q=(uint32_t)__builtin_ctz(queues);queues&=queues-1;
    *mesh_tag(m,page,m->send_bytes[buffer->owner>>63][q])=row;
  }
  mesh_bits_set(m,MESH_PRESENT,row,m->block);
  mesh_notify(m,row,m->block);
  mesh_buffer_produced(m,row,m->block);
}

/* design/algorithm-sources.md#programkernel_call */
static inline struct mesh_notice *mesh_notices(struct hdr *m,uint32_t queue){
  return (struct mesh_notice *)((char *)m+m->notice_off)+(size_t)queue*mesh_rows(m);
}
/* design/algorithm-sources.md#programkernel_call */
static inline void mesh_notice_push(struct hdr *m,uint32_t queue,uint32_t index){
  struct mesh_notice *entry=&mesh_notices(m,queue)[index];
  uint32_t head=atomic_load_explicit(&m->notice_head[queue],memory_order_relaxed);
  do {entry->next=head;}
  while(!atomic_compare_exchange_weak_explicit(&m->notice_head[queue],&head,index,memory_order_release,memory_order_relaxed));
}
/* design/algorithm-sources.md#programkernel_call */
static inline uint32_t mesh_notice_take(struct hdr *m,uint32_t queue){
  return atomic_exchange_explicit(&m->notice_head[queue],MESH_ABSENT,memory_order_acquire);
}
/* design/algorithm-sources.md#programkernel_call */
static inline uint32_t mesh_notice_next(struct hdr *m,uint32_t queue,uint32_t index){
  struct mesh_notice *entry=&mesh_notices(m,queue)[index];
  uint32_t next=entry->next;
  return next;
}

static inline uint64_t mesh_layout(struct hdr *h,uint32_t pgsz,uint32_t block,uint32_t rows){
  uint64_t at=(sizeof *h+pgsz-1)/pgsz*pgsz,words=((uint64_t)rows+63)/64,blocks=rows/block;
  h->pgsz=pgsz; h->block=block; h->rows=rows;
  h->planes_off=at; at+=(uint64_t)MESH_PLANES*words*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->page_off=at; at+=(uint64_t)rows*sizeof(uint32_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->buffer_off=at; at+=(uint64_t)rows*sizeof(struct mesh_buffer); at=(at+pgsz-1)/pgsz*pgsz;
  h->backing_off=at; at+=(uint64_t)rows*sizeof(uint32_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->order_off=at; at+=(uint64_t)2*MESH_QPS*blocks*sizeof(struct mesh_transfer); at=(at+pgsz-1)/pgsz*pgsz;
  uint64_t bytes=(uint64_t)block*pgsz; at=(at+bytes-1)/bytes*bytes;
  h->notice_off=at; at+=(uint64_t)MESH_NOTICE_BANKS*MESH_NOTICE_QUEUES*rows*sizeof(struct mesh_notice); at=(at+bytes-1)/bytes*bytes;
  for(uint32_t queue=0;queue<MESH_NOTICE_BANKS*MESH_NOTICE_QUEUES;queue++)atomic_store_explicit(&h->notice_head[queue],MESH_ABSENT,memory_order_relaxed);
  atomic_store_explicit(&h->reclaim_head,MESH_ABSENT,memory_order_relaxed);
  h->data_off=at; at+=(uint64_t)rows*pgsz;
  h->length=at; return at;
}
#endif
