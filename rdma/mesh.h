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
#define MESH_VERSION 18u
#define MESH_RING 65536
#define MESH_BINDINGS 4096
#define MESH_ABSENT UINT32_MAX
#define MESH_RESERVED (UINT32_MAX-1u)
enum { SUB, FREE, NRING };
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_STOPPED };
enum { MESH_PRESENT, MESH_CONSTANT, MESH_PRODUCING, MESH_ROW_OWN, MESH_ROW_HOT, MESH_ROW_LANDED, MESH_ROW_BOUND, MESH_PAGE_OWN, MESH_PAGE_HOT, MESH_READ, MESH_PLANES=MESH_READ+64 };
#define MESH_READERS 64
struct ring { _Alignas(128) _Atomic uint64_t head; _Alignas(128) _Atomic uint64_t tail; };
struct mesh_port_info { char device[32]; uint16_t peer; _Atomic uint64_t phase; uint64_t when; int64_t code; uint32_t domain,reserved; };
struct mesh_tag { uint32_t magic,binding,index,reserved; };
struct hdr {
  uint32_t magic,version,pgsz,block,pool,arena,node,reserved;
  uint64_t rings_off,planes_off,page_off,mask_off,send_off,base_off,landed_off,landing_row_off,changed_off,data_off,length;
  _Atomic uint64_t client,bridge_pid,sent,recvd,bad,sending,binding_next;
  struct mesh_port_info port;
  struct ring r[NRING];
};
static inline uint32_t mesh_rows(const struct hdr *m){ return m->pool+m->arena; }
static inline uint32_t mesh_words(const struct hdr *m){ return (mesh_rows(m)+63)/64; }
static inline _Atomic uint64_t *mesh_plane(struct hdr *m,int plane){ return (_Atomic uint64_t*)((unsigned char*)m+m->planes_off)+(size_t)plane*mesh_words(m); }
static inline _Atomic uint32_t *mesh_page(struct hdr *m){ return (_Atomic uint32_t*)((unsigned char*)m+m->page_off); }
static inline uint64_t *mesh_mask(struct hdr *m){ return (uint64_t*)((unsigned char*)m+m->mask_off); }
static inline uint8_t *mesh_send(struct hdr *m){ return (uint8_t*)m+m->send_off; }
/* design/algorithm-sources.md#configured-binding-identities */
static inline _Atomic uint64_t *mesh_base(struct hdr *m){ return (_Atomic uint64_t*)((unsigned char*)m+m->base_off); }

static inline _Atomic uint64_t *mesh_landed(struct hdr *m){ return (_Atomic uint64_t*)((unsigned char*)m+m->landed_off); }
/* design/algorithm-sources.md#nonblocking-table-ownership */
static inline _Atomic uint32_t *mesh_landing_row(struct hdr *m){ return (_Atomic uint32_t*)((unsigned char*)m+m->landing_row_off); }
/* design/algorithm-sources.md#nonblocking-table-ownership */
static inline _Atomic uint64_t *mesh_changed(struct hdr *m){ return (_Atomic uint64_t*)((unsigned char*)m+m->changed_off); }
static inline uint32_t mesh_window_blocks(const struct hdr *m){ return 4095u/(m->block*m->pgsz/4096u); }

static inline uint64_t mesh_submission(uint32_t page,uint32_t row,uint32_t plane){ return ((uint64_t)page<<36)|((uint64_t)row<<8)|(plane&63); }
static inline uint32_t mesh_submission_page(uint64_t e){ return (uint32_t)(e>>36); }
static inline uint32_t mesh_submission_row(uint64_t e){ return (uint32_t)((e>>8)&((UINT64_C(1)<<28)-1)); }
static inline uint32_t mesh_submission_plane(uint64_t e){ return (uint32_t)(e&63); }
static inline unsigned char *mesh_at(struct hdr *m,uint32_t page){ return (unsigned char*)m+m->data_off+(size_t)page*m->pgsz; }
static inline _Atomic uint64_t *mesh_slot(struct hdr *m,int k,uint64_t i){ return (_Atomic uint64_t*)((unsigned char*)m+m->rings_off)+(size_t)k*MESH_RING+i%MESH_RING; }

static inline void mesh_push(struct hdr *m,int k,uint64_t value){
  struct ring *q=&m->r[k];
  uint64_t h=atomic_fetch_add_explicit(&q->head,1,memory_order_relaxed);
  atomic_store_explicit(mesh_slot(m,k,h),value+1,memory_order_release);
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

/* design/algorithm-sources.md#nonblocking-table-ownership */
static inline void mesh_send_complete(struct hdr *m,uint64_t entry){
  uint32_t row=mesh_submission_row(entry),page=mesh_submission_page(entry);
  mesh_bits_set(m,MESH_READ+(int)mesh_submission_plane(entry),row,m->block);
  mesh_bits_clear(m,MESH_PAGE_HOT,page,m->block);
  mesh_bits_clear(m,MESH_ROW_HOT,row,m->block);
}
/* design/algorithm-sources.md#nonblocking-table-ownership */
static inline void mesh_reclaim_consumed(struct hdr *m){
  uint32_t blocks=m->pool/m->block;
  for(uint32_t w=0;w<(blocks+63)/64;w++){
    uint64_t landed=atomic_load_explicit(&mesh_landed(m)[w],memory_order_acquire);
    while(landed){
      uint32_t b=w*64+(uint32_t)__builtin_ctzll(landed); landed&=landed-1;
      uint32_t row=atomic_load_explicit(&mesh_landing_row(m)[b],memory_order_acquire),page=b*m->block;
      int owned=mesh_bits_all(m,MESH_PAGE_OWN,page,m->block);
      if(row==MESH_ABSENT && owned){
        const struct mesh_tag *tag=(const struct mesh_tag*)mesh_at(m,page+m->block-1);
        uint64_t entry=atomic_load_explicit(&mesh_base(m)[tag->binding%MESH_BINDINGS],memory_order_acquire);
        uint32_t base=(uint32_t)entry;
        if(base==MESH_ABSENT || base==MESH_RESERVED){
          owned=0;
        } else if((uint64_t)base+(uint64_t)tag->index*m->block+m->block>mesh_rows(m)){
          atomic_fetch_add_explicit(&m->bad,1,memory_order_relaxed);
          owned=0;
        } else {
          row=base+tag->index*m->block;
          mesh_bits_set(m,MESH_ROW_LANDED,row,m->block);
          for(int p=0;p<MESH_READERS;p++) mesh_bits_clear(m,MESH_READ+p,row,m->block);
          for(uint32_t k=0;k<m->block;k++) atomic_store_explicit(&mesh_page(m)[row+k],page+k,memory_order_release);
          atomic_store_explicit(&mesh_landing_row(m)[b],row,memory_order_release);
          mesh_bits_set(m,MESH_PRESENT,row,m->block);
          continue;
        }
      }
      int consumed=1;
      for(uint32_t r=row;owned && r<row+m->block && consumed;r++){
        uint64_t bit=UINT64_C(1)<<(r%64);
        if(!(atomic_load_explicit(&mesh_plane(m,MESH_ROW_OWN)[r/64],memory_order_acquire)&bit)) continue;
        uint64_t need=mesh_mask(m)[r];
        for(int p=0;need && consumed;p++,need>>=1)
          if((need&1) && !(atomic_load_explicit(&mesh_plane(m,MESH_READ+p)[r/64],memory_order_acquire)&bit)) consumed=0;
      }
      for(uint32_t x=page/64;x<=(page+m->block-1)/64 && consumed;x++)
        if(atomic_load_explicit(&mesh_plane(m,MESH_PAGE_HOT)[x],memory_order_acquire)&mesh_word_mask(page,m->block,x)) consumed=0;
      if(!consumed) continue;
      if(row!=MESH_ABSENT){
        mesh_bits_clear(m,MESH_PRESENT,row,m->block);
        for(uint32_t r=row;r<row+m->block;r++) atomic_store_explicit(&mesh_page(m)[r],MESH_ABSENT,memory_order_release);
        mesh_bits_clear(m,MESH_ROW_LANDED,row,m->block);
      }
      mesh_bits_clear(m,MESH_PAGE_OWN,page,m->block);
      atomic_fetch_and_explicit(&mesh_landed(m)[w],~(UINT64_C(1)<<(b%64)),memory_order_acq_rel);
      mesh_push(m,FREE,page);
    }
  }
}
/* design/algorithm-sources.md#nonblocking-table-ownership */
static inline void mesh_reclaim_bindings(struct hdr *m){
  for(uint32_t index=0;index<(mesh_words(m)+63)/64;index++){
    uint64_t changed=atomic_exchange_explicit(&mesh_changed(m)[index],0,memory_order_acq_rel);
    while(changed){
      uint32_t w=index*64+(uint32_t)__builtin_ctzll(changed); changed&=changed-1;
      uint64_t bound=atomic_load_explicit(&mesh_plane(m,MESH_ROW_BOUND)[w],memory_order_acquire);
      uint64_t owned=atomic_load_explicit(&mesh_plane(m,MESH_ROW_OWN)[w],memory_order_acquire);
      atomic_fetch_and_explicit(&mesh_plane(m,MESH_ROW_BOUND)[w],~(bound&~owned),memory_order_acq_rel);
    }
  }
}
/* design/algorithm-sources.md#nonblocking-table-ownership */
static inline void mesh_retire_rows(struct hdr *m){
  for(uint32_t w=0;w<mesh_words(m);w++){
    uint64_t owned=atomic_exchange_explicit(&mesh_plane(m,MESH_ROW_OWN)[w],0,memory_order_acq_rel);
    uint64_t bound=atomic_load_explicit(&mesh_plane(m,MESH_ROW_BOUND)[w],memory_order_acquire);
    if(bound&owned) atomic_fetch_or_explicit(&mesh_changed(m)[w/64],UINT64_C(1)<<(w%64),memory_order_release);
  }
}
static inline uint64_t mesh_layout(struct hdr *h,uint32_t pgsz,uint32_t block,uint32_t pool,uint32_t arena){
  uint64_t at=(sizeof *h+pgsz-1)/pgsz*pgsz,rows=(uint64_t)pool+arena,words=(rows+63)/64;
  h->pgsz=pgsz; h->block=block; h->pool=pool; h->arena=arena;
  h->rings_off=at; at+=(uint64_t)NRING*MESH_RING*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->planes_off=at; at+=(uint64_t)MESH_PLANES*words*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->page_off=at; at+=rows*sizeof(uint32_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->mask_off=at; at+=rows*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->send_off=at; at+=rows; at=(at+pgsz-1)/pgsz*pgsz;
  h->base_off=at; at+=(uint64_t)MESH_BINDINGS*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->landed_off=at; at+=((uint64_t)pool/block+63)/64*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->landing_row_off=at; at+=(uint64_t)pool/block*sizeof(uint32_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->changed_off=at; at+=(words+63)/64*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  uint64_t bytes=(uint64_t)block*pgsz; at=(at+bytes-1)/bytes*bytes;
  h->data_off=at; at+=rows*pgsz;
  h->length=at; return at;
}
#endif
