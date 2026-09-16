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
#define MESH_VERSION 53u
#define MESH_ABSENT UINT32_MAX
/* design/collective-dependency-ledger.md#d6-paired-send-and-receive-frame-counts-match */
#define MESH_QPS 8
struct mesh_transfer { uint32_t local_row,binding,count,stride,chunk_stride,pool; uint64_t bytes; };
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_STOPPED };
/* design/algorithm-sources.md#programtensor */
enum { MESH_ROW_OWN, MESH_ROW_HOT, MESH_PAGE_OWN, MESH_FREE, MESH_PLANES };
/* design/algorithm-sources.md#programtensor */
enum { MESH_BUFFER_CLOSED=4 };
#define MESH_BUFFER_FLAG(flag) ((uint64_t)(flag)<<32)
struct mesh_buffer { _Atomic uint64_t ownership; uint32_t first,pages; _Atomic uint32_t uses; uint32_t invocation,channel,binding; uint64_t owner; };
struct mesh_pool { _Atomic uint64_t owner; uint32_t pages; };
/* design/collective-dependency-ledger.md#d5-receive-consumption-has-per-queue-fifo-order */
enum { MESH_SEND, MESH_RECEIVE };
#define MESH_COMPUTE_THREADS 8
#define MESH_NOTICE_BANKS 2
enum { MESH_NOTICE_LEVELS=(32+5)/6 };
struct mesh_notice_reader {
  _Atomic uint64_t *words[MESH_NOTICE_LEVELS];
  uint64_t pending[MESH_NOTICE_LEVELS];
  uint32_t base[MESH_NOTICE_LEVELS],level,top;
};
struct mesh_port_info { _Atomic uint32_t phase,domain; _Atomic int64_t code; };
/* design/algorithm-sources.md#meshresult */
enum { MESH_RESULT_SUCCESS, MESH_RESULT_LINK, MESH_RESULT_FUNCTION, MESH_RESULT_BUSY };
#define MESH_RESULT(kind,id,code) ((uint64_t)(kind)<<62|(uint64_t)(id)<<32|(uint32_t)(code))
struct mesh_instance { _Atomic uint64_t status,references; };
#define MESH_TRANSFER_REFERENCE (UINT64_C(1)<<32)
struct mesh_link_info { uint32_t peer; char device[32]; uint64_t bandwidth; struct mesh_port_info port; _Atomic uint32_t order_length[2*MESH_NOTICE_BANKS*MESH_QPS]; };
struct hdr {
  uint32_t magic,version,pgsz,block,rows,node,qps,links;
  _Atomic uint64_t configured;
  uint64_t planes_off,presence_off,page_off,buffer_off,pool_off,link_off,send_off,order_off,notice_off,instance_off,tags_off,data_off,length;
  uint32_t notice_levels,notice_words,notice_offsets[MESH_NOTICE_LEVELS];
  uint32_t instance_count[MESH_NOTICE_BANKS];
  _Atomic uint64_t client,bridge_pid,device_client,serial,retired;
  struct mesh_port_info port;
};
void mesh_publish(struct hdr *,uint32_t row);
/* design/algorithm-sources.md#meshresult */
static inline struct mesh_instance *mesh_instances(struct hdr *m,uint64_t owner){return (struct mesh_instance *)((char *)m+m->instance_off)+(owner>>63)*m->rows;}
/* design/algorithm-sources.md#meshresult */
static inline void mesh_instance_conclude(struct mesh_instance *instance,uint64_t status){
  uint64_t pending=MESH_RESULT(MESH_RESULT_BUSY,0,0);
  atomic_compare_exchange_strong_explicit(&instance->status,&pending,status,memory_order_release,memory_order_relaxed);
}
/* design/algorithm-sources.md#meshresult */
static inline uint64_t mesh_instance_release(struct mesh_instance *instance,uint64_t references){
  uint64_t previous=atomic_fetch_sub_explicit(&instance->references,references,memory_order_acq_rel);
  if(previous==references)mesh_instance_conclude(instance,0);
  return previous;
}
/* design/algorithm-sources.md#programtensor */
static inline uint32_t mesh_notice_queue(struct hdr *m,uint64_t owner,uint32_t queue){return (uint32_t)(owner>>63)*(m->links*(m->qps+1)+MESH_COMPUTE_THREADS)+queue;}
/* design/algorithm-sources.md#programcopy */
static inline struct mesh_link_info *mesh_links(struct hdr *m){return (struct mesh_link_info *)((char *)m+m->link_off);}
/* design/algorithm-sources.md#programcopy */
static inline _Atomic uint64_t *mesh_send_uses(struct hdr *m,uint32_t row){return (_Atomic uint64_t *)((char *)m+m->send_off)+(size_t)row*((m->links+63)/64);}
static inline uint32_t mesh_rows(const struct hdr *m){ return m->rows; }
static inline uint32_t mesh_words(const struct hdr *m){ return (mesh_rows(m)+63)/64; }
static inline uint32_t mesh_blocks(const struct hdr *m){ return mesh_rows(m)/m->block; }
static inline _Atomic uint64_t *mesh_plane(struct hdr *m,int plane){ return (_Atomic uint64_t*)((unsigned char*)m+m->planes_off)+(size_t)plane*mesh_words(m); }
/* design/algorithm-sources.md#programkernel_call */
static inline _Atomic uint32_t *mesh_presence(struct hdr *m){return (_Atomic uint32_t *)((char *)m+m->presence_off);}
static inline _Atomic uint32_t *mesh_page(struct hdr *m){ return (_Atomic uint32_t*)((unsigned char*)m+m->page_off); }
/* design/algorithm-sources.md#programtensor */
static inline struct mesh_buffer *mesh_buffers(struct hdr *m){ return (struct mesh_buffer *)((char *)m+m->buffer_off); }
/* design/algorithm-sources.md#programtensor */
static inline struct mesh_pool *mesh_pools(struct hdr *m){return (struct mesh_pool *)((char *)m+m->pool_off);}
void mesh_buffer_release(struct hdr *,uint32_t first,uint32_t count);
/* ledger D5 */
static inline struct mesh_transfer *mesh_transfers(struct hdr *m,uint64_t owner,uint32_t queue,int direction){ return (struct mesh_transfer*)((unsigned char*)m+m->order_off)+((size_t)(owner>>63)*2*m->links*m->qps+2*queue+(uint32_t)direction)*mesh_blocks(m); }
static inline _Atomic uint32_t *mesh_order_length(struct hdr *m,uint64_t owner,uint32_t queue,int direction){ return &mesh_links(m)[queue/m->qps].order_length[(owner>>63)*2*MESH_QPS+2*(queue%m->qps)+(uint32_t)direction]; }
static inline unsigned char *mesh_at(struct hdr *m,uint32_t page){ return (unsigned char*)m+m->data_off+(size_t)page*m->pgsz; }
/* design/algorithm-sources.md#programcopy */
static inline _Atomic uint64_t *mesh_tag(struct hdr *m,uint32_t page){ return (_Atomic uint64_t *)((char *)m+m->tags_off+(size_t)(page/m->block+1)*m->pgsz-sizeof(uint64_t)); }

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

/* design/algorithm-sources.md#programkernel_call */
static inline _Atomic uint64_t *mesh_notices(struct hdr *m,uint32_t queue){
  return (_Atomic uint64_t *)((char *)m+m->notice_off)+(size_t)queue*m->notice_words;
}
/* design/algorithm-sources.md#programkernel_call */
static inline void mesh_notice_push(struct hdr *m,uint32_t queue,uint32_t index){
  _Atomic uint64_t *words=mesh_notices(m,queue);
  for(uint32_t level=0;level<m->notice_levels;level++,index>>=6)
    atomic_fetch_or_explicit(&words[m->notice_offsets[level]+(index>>6)],UINT64_C(1)<<(index&63),memory_order_release);
}
/* design/algorithm-sources.md#programkernel_call */
static inline struct mesh_notice_reader mesh_notice_reader_init(struct hdr *m,uint32_t queue){
  struct mesh_notice_reader reader={.level=m->notice_levels-1,.top=m->notice_levels-1};
  for(uint32_t level=0;level<m->notice_levels;level++)reader.words[level]=mesh_notices(m,queue)+m->notice_offsets[level];
  return reader;
}
/* design/algorithm-sources.md#programkernel_call */
static inline uint32_t mesh_notice_take(struct mesh_notice_reader *reader){
  for(;;){
    uint32_t level=reader->level;
    if(reader->pending[level]){
      uint32_t index=reader->base[level]+(uint32_t)__builtin_ctzll(reader->pending[level]);
      reader->pending[level]&=reader->pending[level]-1;
      if(!level)return index;
      reader->level=level-1;reader->base[level-1]=index*64;
      reader->pending[level-1]=atomic_exchange_explicit(&reader->words[level-1][index],0,memory_order_acquire);
    } else if(level<reader->top)reader->level=level+1;
    else {
      reader->pending[level]=atomic_exchange_explicit(reader->words[level],0,memory_order_acquire);
      if(!reader->pending[level])return MESH_ABSENT;
    }
  }
}

/* design/algorithm-sources.md#programkernel_call */
static inline uint64_t mesh_layout(struct hdr *h,uint32_t pgsz,uint32_t block,uint32_t rows,uint32_t links,uint32_t qps){
  uint64_t at=(sizeof *h+pgsz-1)/pgsz*pgsz,words=((uint64_t)rows+63)/64,blocks=rows/block;
  h->pgsz=pgsz; h->block=block; h->rows=rows; h->links=links; h->qps=qps;
  h->planes_off=at; at+=(uint64_t)MESH_PLANES*words*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->presence_off=at;at+=(uint64_t)rows*sizeof(uint32_t);at=(at+pgsz-1)/pgsz*pgsz;
  h->page_off=at; at+=(uint64_t)rows*sizeof(uint32_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->buffer_off=at; at+=(uint64_t)rows*sizeof(struct mesh_buffer); at=(at+pgsz-1)/pgsz*pgsz;
  h->pool_off=at; at+=blocks*sizeof(struct mesh_pool); at=(at+pgsz-1)/pgsz*pgsz;
  h->link_off=at; at+=(uint64_t)links*sizeof(struct mesh_link_info); at=(at+pgsz-1)/pgsz*pgsz;
  h->send_off=at; at+=(uint64_t)rows*((links+63)/64)*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->order_off=at; at+=(uint64_t)MESH_NOTICE_BANKS*2*links*qps*blocks*sizeof(struct mesh_transfer); at=(at+pgsz-1)/pgsz*pgsz;
  h->notice_levels=0;h->notice_words=0;
  for(uint64_t count=rows;;){
    count=(count+63)/64;
    h->notice_offsets[h->notice_levels++]=h->notice_words;
    h->notice_words+=(uint32_t)((count+7)&~UINT64_C(7));
    if(count<=1)break;
  }
  uint64_t bytes=(uint64_t)block*pgsz; at=(at+bytes-1)/bytes*bytes;
  h->notice_off=at; at+=(uint64_t)MESH_NOTICE_BANKS*(links*(qps+1)+MESH_COMPUTE_THREADS)*h->notice_words*sizeof(uint64_t); at=(at+bytes-1)/bytes*bytes;
  h->instance_off=at; at+=(uint64_t)MESH_NOTICE_BANKS*rows*sizeof(struct mesh_instance); at=(at+bytes-1)/bytes*bytes;
  h->tags_off=at; at+=blocks*pgsz; at=(at+bytes-1)/bytes*bytes;
  h->data_off=at; at+=(uint64_t)rows*pgsz;
  h->length=at; return at;
}
#endif
