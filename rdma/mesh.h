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
#define MESH_VERSION 20u
#define MESH_ABSENT UINT32_MAX
/* ledger D6: "A maximum of 10 unreliable connection (UC) queue pairs" */
#define MESH_QPS 8
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_STOPPED };
/* ROW_HOT: a produced send block the bridge has not completed, or a posted receive block not yet completed.
   PAGE_HOT: pages held by an outstanding work request. */
enum { MESH_PRESENT, MESH_CONSTANT, MESH_PRODUCING, MESH_ROW_OWN, MESH_ROW_HOT, MESH_PAGE_OWN, MESH_PAGE_HOT, MESH_READ, MESH_PLANES=MESH_READ+64 };
#define MESH_READERS 64
/* ledger D5: one posting order per queue pair and direction */
enum { MESH_SEND, MESH_RECEIVE };
struct mesh_port_info { char device[32]; uint16_t peer; _Atomic uint64_t phase; uint64_t when; int64_t code; uint32_t domain,reserved; };
struct hdr {
  uint32_t magic,version,pgsz,block,rows,node,qps;
  _Atomic uint32_t configured;
  uint64_t planes_off,page_off,mask_off,send_off,order_off,data_off,length;
  _Atomic uint64_t client,bridge_pid;
  _Atomic uint32_t order_length[2*MESH_QPS];
  struct mesh_port_info port;
};
static inline uint32_t mesh_rows(const struct hdr *m){ return m->rows; }
static inline uint32_t mesh_words(const struct hdr *m){ return (mesh_rows(m)+63)/64; }
static inline uint32_t mesh_blocks(const struct hdr *m){ return mesh_rows(m)/m->block; }
static inline _Atomic uint64_t *mesh_plane(struct hdr *m,int plane){ return (_Atomic uint64_t*)((unsigned char*)m+m->planes_off)+(size_t)plane*mesh_words(m); }
static inline _Atomic uint32_t *mesh_page(struct hdr *m){ return (_Atomic uint32_t*)((unsigned char*)m+m->page_off); }
static inline uint64_t *mesh_mask(struct hdr *m){ return (uint64_t*)((unsigned char*)m+m->mask_off); }
/* Nonzero at a send block's first row: its reader plane plus one. */
static inline uint8_t *mesh_send(struct hdr *m){ return (uint8_t*)m+m->send_off; }
/* ledger D5 */
static inline uint32_t *mesh_order(struct hdr *m,uint32_t queue,int direction){ return (uint32_t*)((unsigned char*)m+m->order_off)+(size_t)(2*queue+(uint32_t)direction)*mesh_blocks(m); }
static inline _Atomic uint32_t *mesh_order_length(struct hdr *m,uint32_t queue,int direction){ return &m->order_length[2*queue+(uint32_t)direction]; }
/* ledger D6: "A maximum of 4095 work requests at a time", queues sized in 4 KB frames */
static inline uint32_t mesh_window_blocks(const struct hdr *m){ return 4095u/(m->block*m->pgsz/4096u); }
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
static inline int mesh_bits_all(struct hdr *m,int plane,uint32_t first,uint32_t count){
  _Atomic uint64_t *p=mesh_plane(m,plane);
  for(uint32_t w=first/64;count && w<=(first+count-1)/64;w++){ uint64_t k=mesh_word_mask(first,count,w); if((atomic_load_explicit(&p[w],memory_order_acquire)&k)!=k) return 0; }
  return 1;
}
static inline int mesh_bit(struct hdr *m,int plane,uint32_t index){
  return (int)((atomic_load_explicit(&mesh_plane(m,plane)[index/64],memory_order_acquire)>>(index%64))&1);
}

/* ledger D9: a receive is posted on a block only after every configured reader consumed its previous value */
static inline int mesh_receive_postable(struct hdr *m,uint32_t row){
  if(mesh_bit(m,MESH_ROW_HOT,row)) return 0;
  for(uint32_t r=row;r<row+m->block;r++){
    if(!mesh_bit(m,MESH_PRESENT,r)) continue;
    uint64_t need=mesh_mask(m)[r];
    for(int p=0;need;p++,need>>=1) if((need&1) && !mesh_bit(m,MESH_READ+p,r)) return 0;
  }
  return 1;
}
/* ledger D4: the posted receive holds the consumer's own pages */
static inline void mesh_receive_posted(struct hdr *m,uint32_t row,uint32_t page){
  mesh_bits_set(m,MESH_ROW_HOT,row,m->block);
  mesh_bits_set(m,MESH_PAGE_HOT,page,m->block);
  mesh_bits_clear(m,MESH_PRESENT,row,m->block);
}
/* ledger D8: presence is the receive completion */
static inline void mesh_receive_complete(struct hdr *m,uint32_t row,uint32_t page,int landed){
  if(landed){
    for(int p=0;p<MESH_READERS;p++) mesh_bits_clear(m,MESH_READ+p,row,m->block);
    mesh_bits_set(m,MESH_PRESENT,row,m->block);
  }
  mesh_bits_clear(m,MESH_PAGE_HOT,page,m->block);
  mesh_bits_clear(m,MESH_ROW_HOT,row,m->block);
}
/* ledger D7: a produced block waits only for its queue position */
static inline int mesh_send_postable(struct hdr *m,uint32_t row,uint32_t page){
  return mesh_bit(m,MESH_ROW_HOT,row) && !mesh_bit(m,MESH_PAGE_HOT,page);
}
/* ledger D12: completion releases the block to its producer whatever its status */
static inline void mesh_send_complete(struct hdr *m,uint32_t row,uint32_t page,uint32_t plane){
  mesh_bits_set(m,MESH_READ+(int)plane,row,m->block);
  mesh_bits_clear(m,MESH_PAGE_HOT,page,m->block);
  mesh_bits_clear(m,MESH_ROW_HOT,row,m->block);
}

static inline uint64_t mesh_layout(struct hdr *h,uint32_t pgsz,uint32_t block,uint32_t rows){
  uint64_t at=(sizeof *h+pgsz-1)/pgsz*pgsz,words=((uint64_t)rows+63)/64,blocks=rows/block;
  h->pgsz=pgsz; h->block=block; h->rows=rows;
  h->planes_off=at; at+=(uint64_t)MESH_PLANES*words*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->page_off=at; at+=(uint64_t)rows*sizeof(uint32_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->mask_off=at; at+=(uint64_t)rows*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->send_off=at; at+=rows; at=(at+pgsz-1)/pgsz*pgsz;
  h->order_off=at; at+=(uint64_t)2*MESH_QPS*blocks*sizeof(uint32_t); at=(at+pgsz-1)/pgsz*pgsz;
  uint64_t bytes=(uint64_t)block*pgsz; at=(at+bytes-1)/bytes*bytes;
  h->data_off=at; at+=(uint64_t)rows*pgsz;
  h->length=at; return at;
}
#endif
