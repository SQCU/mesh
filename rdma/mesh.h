#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
/* design/pages-and-functions.md#what-the-page-table-is */
#define MESH_MAGIC 0x4d455348u
#define MESH_TAG 0x4d455347u
#define MESH_NAME "/mesh0"
#define MESH_PORT "18519"
#define MESH_MODE 0666
#define MESH_VERSION 14u
#define MESH_RING 65536
#define MESH_BINDINGS 4096
#define MESH_ABSENT UINT32_MAX
enum { SUB, FREE, NRING };
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_STOPPED };
enum { MESH_PRESENT, MESH_CONSTANT, MESH_FREED, MESH_READ, MESH_PLANES=8 };
#define MESH_READERS (MESH_PLANES-MESH_READ)
struct ring { _Alignas(128) _Atomic uint64_t head; _Alignas(128) _Atomic uint64_t tail; };
struct mesh_port_info { char device[32]; uint16_t peer; _Atomic uint64_t phase; uint64_t when; int64_t code; uint32_t domain,reserved; };
struct mesh_tag { uint32_t magic,binding,index,reserved; };
struct hdr {
  uint32_t magic,version,pgsz,block,pool,arena,node,reserved;
  uint64_t rings_off,planes_off,page_off,mask_off,send_off,base_off,landed_off,data_off,length;
  _Atomic uint64_t client,generation,bridge_pid,sent,recvd,bad,sending;
  struct mesh_port_info port;
  struct ring r[NRING];
};
static inline uint32_t mesh_rows(const struct hdr *m){ return m->pool+m->arena; }
static inline uint32_t mesh_words(const struct hdr *m){ return (mesh_rows(m)+63)/64; }
static inline _Atomic uint64_t *mesh_plane(struct hdr *m,int plane){ return (_Atomic uint64_t*)((unsigned char*)m+m->planes_off)+(size_t)plane*mesh_words(m); }
static inline _Atomic uint32_t *mesh_page(struct hdr *m){ return (_Atomic uint32_t*)((unsigned char*)m+m->page_off); }
static inline uint8_t *mesh_mask(struct hdr *m){ return (uint8_t*)m+m->mask_off; }
static inline uint8_t *mesh_send(struct hdr *m){ return (uint8_t*)m+m->send_off; }
static inline uint32_t *mesh_base(struct hdr *m){ return (uint32_t*)((unsigned char*)m+m->base_off); }
/* One bit per landing block: set by the bridge on delivery, cleared by whoever returns the block to the free index. */
static inline _Atomic uint64_t *mesh_landed(struct hdr *m){ return (_Atomic uint64_t*)((unsigned char*)m+m->landed_off); }
static inline uint32_t mesh_window_blocks(const struct hdr *m){ return 4095u/(m->block*m->pgsz/4096u); }
/* Submission entries are self-contained: the bridge needs nothing else after the push. */
static inline uint64_t mesh_submission(uint32_t page,uint32_t row,uint64_t generation,uint32_t plane){ return ((uint64_t)page<<36)|((uint64_t)row<<8)|((generation&15)<<3)|(plane&7); }
static inline uint32_t mesh_submission_page(uint64_t e){ return (uint32_t)(e>>36); }
static inline uint32_t mesh_submission_row(uint64_t e){ return (uint32_t)((e>>8)&((UINT64_C(1)<<28)-1)); }
static inline uint32_t mesh_submission_generation(uint64_t e){ return (uint32_t)((e>>3)&15); }
static inline uint32_t mesh_submission_plane(uint64_t e){ return (uint32_t)(e&7); }
static inline unsigned char *mesh_at(struct hdr *m,uint32_t page){ return (unsigned char*)m+m->data_off+(size_t)page*m->pgsz; }
static inline _Atomic uint64_t *mesh_slot(struct hdr *m,int k,uint64_t i){ return (_Atomic uint64_t*)((unsigned char*)m+m->rings_off)+(size_t)k*MESH_RING+i%MESH_RING; }
/* Multi-producer, single-consumer FIFO of nonzero words. */
static inline int mesh_push(struct hdr *m,int k,uint64_t value){
  struct ring *q=&m->r[k];
  uint64_t h=atomic_load_explicit(&q->head,memory_order_relaxed);
  do { if(h-atomic_load_explicit(&q->tail,memory_order_acquire)>=MESH_RING) return -1; }
  while(!atomic_compare_exchange_weak_explicit(&q->head,&h,h+1,memory_order_relaxed,memory_order_relaxed));
  atomic_store_explicit(mesh_slot(m,k,h),value+1,memory_order_release);
  return 0;
}
static inline int mesh_pop(struct hdr *m,int k,uint64_t *value){
  struct ring *q=&m->r[k];
  uint64_t t=atomic_load_explicit(&q->tail,memory_order_relaxed);
  _Atomic uint64_t *in=mesh_slot(m,k,t);
  uint64_t v=atomic_load_explicit(in,memory_order_acquire);
  if(!v) return -1;
  atomic_store_explicit(in,0,memory_order_relaxed);
  atomic_store_explicit(&q->tail,t+1,memory_order_release);
  *value=v-1; return 0;
}
/* Every delivered landing block not yet returned goes back to the free index. */
static inline void mesh_reclaim_landed(struct hdr *m){
  uint32_t blocks=m->pool/m->block;
  for(uint32_t w=0;w<(blocks+63)/64;w++){
    uint64_t landed=atomic_exchange_explicit(&mesh_landed(m)[w],0,memory_order_acq_rel);
    while(landed){ uint32_t b=w*64+(uint32_t)__builtin_ctzll(landed); landed&=landed-1; mesh_push(m,FREE,b*m->block); }
  }
}
/* Bit planes over logical rows: every transition is an OR, every test an AND. */
static inline uint64_t mesh_word_mask(uint32_t first,uint32_t count,uint32_t word){
  uint32_t lo=word*64,hi=lo+64,a=first>lo?first:lo,b=first+count<hi?first+count:hi;
  if(b<=a) return 0;
  uint64_t bits=b-a==64?~UINT64_C(0):((UINT64_C(1)<<(b-a))-1);
  return bits<<(a-lo);
}
static inline void mesh_bits_set(struct hdr *m,int plane,uint32_t first,uint32_t count){
  _Atomic uint64_t *p=mesh_plane(m,plane);
  for(uint32_t w=first/64;w<=(first+count-1)/64;w++) atomic_fetch_or_explicit(&p[w],mesh_word_mask(first,count,w),memory_order_acq_rel);
}
static inline void mesh_bits_clear(struct hdr *m,int plane,uint32_t first,uint32_t count){
  _Atomic uint64_t *p=mesh_plane(m,plane);
  for(uint32_t w=first/64;w<=(first+count-1)/64;w++) atomic_fetch_and_explicit(&p[w],~mesh_word_mask(first,count,w),memory_order_acq_rel);
}
static inline int mesh_bits_all(struct hdr *m,int plane,uint32_t first,uint32_t count){
  _Atomic uint64_t *p=mesh_plane(m,plane);
  for(uint32_t w=first/64;w<=(first+count-1)/64;w++){ uint64_t k=mesh_word_mask(first,count,w); if((atomic_load_explicit(&p[w],memory_order_acquire)&k)!=k) return 0; }
  return 1;
}
/* Region layout from configured geometry; returns total bytes. */
static inline uint64_t mesh_layout(struct hdr *h,uint32_t pgsz,uint32_t block,uint32_t pool,uint32_t arena){
  uint64_t at=(sizeof *h+pgsz-1)/pgsz*pgsz,rows=(uint64_t)pool+arena,words=(rows+63)/64;
  h->pgsz=pgsz; h->block=block; h->pool=pool; h->arena=arena;
  h->rings_off=at; at+=(uint64_t)NRING*MESH_RING*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->planes_off=at; at+=(uint64_t)MESH_PLANES*words*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->page_off=at; at+=rows*sizeof(uint32_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->mask_off=at; at+=rows; at=(at+pgsz-1)/pgsz*pgsz;
  h->send_off=at; at+=rows; at=(at+pgsz-1)/pgsz*pgsz;
  h->base_off=at; at+=(uint64_t)MESH_BINDINGS*sizeof(uint32_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->landed_off=at; at+=((uint64_t)pool/block+63)/64*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->data_off=at; at+=rows*pgsz;
  h->length=at; return at;
}
#endif
