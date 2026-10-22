#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <os/os_sync_wait_on_address.h>
#include "mesh-plan.h"
/* design/transport.md section 2; section 15, "The region" */
#define MESH_MAGIC 0x4d455348u
#define MESH_NAME "/mesh0"
#define MESH_PORT "18519"
#define MESH_ULA_PREFIX "fd6d:6573:68:"
#define MESH_MODE 0666
#define MESH_VERSION 135u
#define MESH_LAYOUT 142u
#define MESH_LINE 128u
#define MESH_ABSENT UINT32_MAX
#define MESH_STREAMS 32
#define MESH_CLIENTS 64
#define MESH_BRIDGE_BYTES (MESH_STREAMS*32u)
#define MESH_RETIRING (UINT64_C(1)<<63)
#define MESH_STREAM_ENTRIES 4096u
#define MESH_STREAM_TIMES 16384u
#define MESH_STREAM_GIVEN (UINT64_C(1)<<63)
_Static_assert(38+25<=63,"an owner word");
_Static_assert(!(MESH_STREAM_ENTRIES&(MESH_STREAM_ENTRIES-1)) && !(MESH_STREAM_TIMES&(MESH_STREAM_TIMES-1)),"a ring a power of two");
#define MESH_PAUSE_NS UINT64_C(3000000000)
#define MESH_PLATFORM_NS UINT64_C(5000000000)
/* design/transport.md section 1, the page table; section 14, "A holding's end" */
enum { MESH_ROW_OWN, MESH_ROW_HOT, MESH_FREE, MESH_PLANES };
struct mesh_buffer {
  _Alignas(32) _Atomic uint64_t owner;
  _Atomic uint32_t closed;
  uint32_t pages;
};
_Static_assert(sizeof(struct mesh_buffer)==32 && _Alignof(struct mesh_buffer)==32,"mesh_buffer");
enum { MESH_LEASE_LIVE, MESH_LEASE_CLOSED, MESH_LEASE_RELEASED, MESH_LEASE_FREE, MESH_LEASE_STATE=3 };
static inline int mesh_lease_move(struct mesh_buffer *b,uint32_t *was,uint32_t from,uint32_t to){
  return (*was&MESH_LEASE_STATE)==from &&
         atomic_compare_exchange_strong_explicit(&b->closed,was,(*was&~(uint32_t)MESH_LEASE_STATE)|to,memory_order_acq_rel,memory_order_acquire);
}
static inline int mesh_lease_close(struct mesh_buffer *b,uint64_t owner){
  uint32_t was=atomic_load_explicit(&b->closed,memory_order_acquire);
  return (was&MESH_LEASE_STATE)==MESH_LEASE_LIVE && atomic_load_explicit(&b->owner,memory_order_acquire)==owner &&
         mesh_lease_move(b,&was,MESH_LEASE_LIVE,MESH_LEASE_CLOSED);
}
struct mesh_page_entry {
  _Alignas(16) _Atomic uint64_t mapping;
  _Atomic uintptr_t address;
};
_Static_assert(sizeof(struct mesh_page_entry)==16 && _Alignof(struct mesh_page_entry)==16 &&
  offsetof(struct mesh_page_entry,address)==8,"mesh_page_entry");
/* design/transport.md section 14, "The stream"; design/pages-and-functions.md */
struct mesh_entry { uint64_t row,pages,word,base; };
struct mesh_pages { uint64_t n; };
struct mesh_row { uint64_t r; };
static inline struct mesh_pages mesh_pages_add(struct mesh_pages a,struct mesh_pages b){ return (struct mesh_pages){a.n+b.n}; }
static inline struct mesh_pages mesh_pages_sub(struct mesh_pages a,struct mesh_pages b){ return (struct mesh_pages){a.n-b.n}; }
static inline int mesh_pages_lt(struct mesh_pages a,struct mesh_pages b){ return a.n<b.n; }
static inline struct mesh_row mesh_row_after(struct mesh_row row,struct mesh_pages i){ return (struct mesh_row){row.r+i.n}; }
static inline struct mesh_row mesh_entry_row(const struct mesh_entry *e){ return (struct mesh_row){e->row}; }
static inline struct mesh_pages mesh_entry_pages(const struct mesh_entry *e){ return (struct mesh_pages){e->pages}; }
static inline struct mesh_pages mesh_entry_base(const struct mesh_entry *e){ return (struct mesh_pages){e->base}; }
static inline struct mesh_pages mesh_pages_load(const _Atomic uint64_t *word){ return (struct mesh_pages){atomic_load_explicit(word,memory_order_acquire)}; }
static inline void mesh_pages_store(_Atomic uint64_t *word,struct mesh_pages p){ atomic_store_explicit(word,p.n,memory_order_release); }
#define MESH_PAGES_LOST ((struct mesh_pages){UINT64_MAX})
struct mesh_posts { _Alignas(MESH_LINE) _Atomic uint64_t posted[2]; _Atomic uint64_t done[2]; };
#define MESH_LANE_UNITS 4096u
_Static_assert(!(MESH_LANE_UNITS&(MESH_LANE_UNITS-1)),"a ring a power of two");
static inline _Atomic uint32_t *mesh_unit_row(void *m,uint64_t rows,int way,int lane,uint64_t u){
  return (_Atomic uint32_t *)((char *)m+rows)+((uint64_t)(2*way+lane)*MESH_LANE_UNITS+u%MESH_LANE_UNITS);
}
struct mesh_stream {
  _Alignas(MESH_LINE) _Atomic uint32_t peer;
  _Atomic uint32_t live;
  uint64_t lists,posted_at,landed_at,rows;
  _Atomic uint64_t bandwidth,wire;
  _Atomic uint32_t child;
  uint64_t child_start;
  _Alignas(MESH_LINE) _Atomic uint64_t owner;
  _Atomic uint64_t sends,lands;
  _Alignas(MESH_LINE) _Atomic uint64_t ready;
  _Alignas(MESH_LINE) _Atomic uint64_t sent;
  uint64_t sent_line[MESH_LINE/8-2];
  _Atomic uint64_t taken[2];
  _Atomic uint64_t landed;
  struct mesh_posts posts[2];
};
_Static_assert(sizeof(struct mesh_stream)==7*MESH_LINE && _Alignof(struct mesh_stream)==MESH_LINE,"mesh_stream");
_Static_assert(offsetof(struct mesh_stream,owner)==1*MESH_LINE && offsetof(struct mesh_stream,ready)==2*MESH_LINE &&
  offsetof(struct mesh_stream,sent)==3*MESH_LINE && offsetof(struct mesh_stream,taken[0])==4*MESH_LINE-8 &&
  offsetof(struct mesh_stream,taken[1])==4*MESH_LINE && offsetof(struct mesh_stream,landed)==4*MESH_LINE+8 &&
  offsetof(struct mesh_stream,posts[0])==5*MESH_LINE && offsetof(struct mesh_stream,posts[1])==6*MESH_LINE,
  "a writer's words a line of their own");
/* design/transport.md section 3, RECORD */
struct mesh_record { uint32_t node,reached; uint64_t boot,sequence,region,memory;
  char model[MESH_MODEL_BYTES]; struct mesh_reach reach[MESH_STREAMS]; struct mesh_health health; };
/* design/transport.md section 2, the client slots */
struct mesh_client {
  _Alignas(64) _Atomic uint64_t owner;
  uint64_t start,padding[6];
};
struct hdr {
  uint32_t magic,version,pgsz,block,rows,node,streams,clients;
  uint64_t planes_off,arena_off,page_off,buffer_off,bridge_off,stream_off,client_off,data_off,length,boot;
  _Atomic uint64_t bridge_pid,serial,map,map_off,map_nodes,health;
};
_Static_assert(sizeof(struct hdr)<=4096,"struct hdr");
_Static_assert(sizeof(_Atomic uint64_t)==8 && _Alignof(_Atomic uint64_t)==8,"M07");
static inline void mesh_notify(_Atomic uint64_t *word){
  atomic_fetch_add_explicit(word,1,memory_order_release);
  atomic_thread_fence(memory_order_release);
  os_sync_wake_by_address_all(word,sizeof *word,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
static inline struct mesh_record *mesh_records(struct hdr *m){ return (struct mesh_record *)((char *)m+atomic_load_explicit(&m->map_off,memory_order_relaxed)); }
static inline int mesh_map_whole(struct hdr *m,uint64_t before){
  atomic_thread_fence(memory_order_acquire);
  return !(before&1) && atomic_load_explicit(&m->map,memory_order_relaxed)==before;
}
static inline struct mesh_client *mesh_client_at(struct hdr *m,uint64_t owner){return (struct mesh_client *)((char *)m+m->client_off)+((owner>>32)&(MESH_CLIENTS-1));}
static inline struct mesh_stream *mesh_stream_at(struct hdr *m,uint32_t p){return (struct mesh_stream *)((char *)m+m->stream_off)+p;}
static inline uint32_t mesh_rows(const struct hdr *m){ return m->rows; }
static inline uint32_t mesh_words(const struct hdr *m){ return (mesh_rows(m)+63)/64; }
static inline int mesh_rows_hold(const struct hdr *m,struct mesh_row row,struct mesh_pages pages){
  return row.r<mesh_rows(m) && pages.n<=mesh_rows(m)-row.r;
}
static inline uint32_t mesh_arena_pages(const struct hdr *m){ return (uint32_t)((m->length-m->data_off)/((uint64_t)m->block*m->pgsz))*m->block; }
static inline _Atomic uint64_t *mesh_plane(struct hdr *m,int plane){ return (_Atomic uint64_t*)((unsigned char*)m+m->planes_off)+(size_t)plane*mesh_words(m); }
static inline _Atomic uint64_t *mesh_arena_bits(struct hdr *m){ return (_Atomic uint64_t*)((unsigned char*)m+m->arena_off); }
static inline struct mesh_page_entry *mesh_page(struct hdr *m){ return (struct mesh_page_entry*)((unsigned char*)m+m->page_off); }
static inline struct mesh_buffer *mesh_buffers(struct hdr *m){ return (struct mesh_buffer *)((char *)m+m->buffer_off); }
static inline unsigned char *mesh_at(struct hdr *m,uint32_t page){ return (unsigned char*)m+m->data_off+(size_t)page*m->pgsz; }
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
static inline uint64_t mesh_layout(struct hdr *h,uint32_t pgsz,uint32_t block,uint32_t pages){
  uint32_t arena=pages/block*block,rows=arena/block;
  uint64_t at=(sizeof *h+pgsz-1)/pgsz*pgsz,words=((uint64_t)rows+63)/64;
  h->pgsz=pgsz; h->block=block; h->rows=rows; h->streams=MESH_STREAMS; h->clients=MESH_CLIENTS;
  h->planes_off=at; at+=(uint64_t)MESH_PLANES*words*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->arena_off=at; at+=(((uint64_t)arena+63)/64)*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->page_off=at; at+=(uint64_t)rows*sizeof(struct mesh_page_entry); at=(at+pgsz-1)/pgsz*pgsz;
  h->buffer_off=at; at+=(uint64_t)rows*sizeof(struct mesh_buffer); at=(at+pgsz-1)/pgsz*pgsz;
  h->bridge_off=at; at+=MESH_BRIDGE_BYTES; at=(at+pgsz-1)/pgsz*pgsz;
  h->stream_off=at; at+=(uint64_t)MESH_STREAMS*sizeof(struct mesh_stream); at=(at+pgsz-1)/pgsz*pgsz;
  h->client_off=at; at+=(uint64_t)MESH_CLIENTS*sizeof(struct mesh_client);
  uint64_t bytes=(uint64_t)block*pgsz; at=(at+bytes-1)/bytes*bytes;
  h->data_off=at; at+=(uint64_t)arena*pgsz;
  h->length=at; return at;
}
#endif
