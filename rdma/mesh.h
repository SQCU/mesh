#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <string.h>
#define MESH_MAGIC 0x4d455348u
#define MESH_NAME "/mesh0"
#define MESH_PORT "18519"
#define MESH_MODE 0666
#define MESH_VERSION 12u
#define MESH_CL 128
#define MESH_RING 65536
#define MESH_OFF 16
#define RINGS ((sizeof(struct hdr)+MESH_CL-1)/MESH_CL*MESH_CL)
enum { SUB, REL, NRING };
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_STOPPED };
struct wire { uint16_t src,dst,hops; };
struct desc { uint32_t page,bytes; uint16_t node,reserved; uint32_t error,domain; uint64_t header; };
struct ring { _Alignas(MESH_CL) _Atomic uint64_t head,tail; };
struct hdr {
  uint32_t magic,version,pgsz,pool,arena,node;
  uint64_t headers_off,data_off;
  struct ring r[NRING];
  _Atomic uint64_t client,sent,recvd,bad,bridge_pid,port_count,tables;
};
struct mesh_port_info {
  char device[32]; uint16_t peer;
  _Atomic uint64_t phase;
  uint64_t when; int64_t code; uint32_t domain,reserved;
};
// ../design/algorithm-sources.md#contiguous-backing-page-views
static inline unsigned char *mesh_at(struct hdr *m,uint32_t i){ return (unsigned char*)m+m->data_off+(size_t)i*m->pgsz; }
// ../design/algorithm-sources.md#contiguous-backing-page-views
static inline unsigned char *mesh_data(struct hdr *m,uint32_t i){ return mesh_at(m,i); }
// ../design/algorithm-sources.md#contiguous-backing-page-views
static inline uint32_t mesh_pay(struct hdr *m){ return m->pgsz; }
// ../design/algorithm-sources.md#transport-page-addressing
static inline struct desc *slot(struct hdr *m,int k,uint64_t i){ return &((struct desc*)((unsigned char*)m+RINGS))[k*MESH_RING+i%MESH_RING]; }
// ../design/algorithm-sources.md#link-port-metadata
static inline struct mesh_port_info *mesh_ports(struct hdr *m){ return (struct mesh_port_info*)((unsigned char*)m+RINGS+NRING*MESH_RING*sizeof(struct desc)); }
// ../design/algorithm-sources.md#local-submission-fifo
static inline int push(struct hdr *m,int k,const struct desc *d){
  struct ring *q=&m->r[k];
  uint64_t h=atomic_load_explicit(&q->head,memory_order_relaxed);
  do {
    if(h-atomic_load_explicit(&q->tail,memory_order_acquire)>=MESH_RING) return -1;
  }while(!atomic_compare_exchange_weak_explicit(&q->head,&h,h+1,memory_order_relaxed,memory_order_relaxed));
  struct desc *out=slot(m,k,h);
  out->page=d->page; out->bytes=d->bytes; out->node=d->node;
  out->error=d->error; out->domain=d->domain;
  __atomic_store_n(&out->header,k==REL?(uint64_t)d->page+1:d->header,__ATOMIC_RELEASE);
  return 0;
}
// ../design/algorithm-sources.md#local-submission-fifo
static inline int pop(struct hdr *m,int k,struct desc *d){
  struct ring *q=&m->r[k];
  uint64_t t=atomic_load_explicit(&q->tail,memory_order_relaxed);
  struct desc *in=slot(m,k,t);
  if(!__atomic_load_n(&in->header,__ATOMIC_ACQUIRE)) return -1;
  *d=*in; __atomic_store_n(&in->header,0,__ATOMIC_RELEASE);
  atomic_store_explicit(&q->tail,t+1,memory_order_release); return 0;
}
struct mesh_rows;
struct mesh_ctx {
  struct hdr *M; unsigned char *arena; size_t len;
  struct mesh_rows **tables; size_t table_count,allocation,extent_bytes;
  uint64_t *blocks,*hot;
};
// ../design/algorithm-sources.md#configuration-storage-layout
static inline uint32_t mesh_received_pages(const struct mesh_ctx *c){ return c->M->pool; }
// ../design/algorithm-sources.md#configuration-storage-layout
static inline uint32_t mesh_arena_pages(const struct mesh_ctx *c){ return c->M->arena; }
struct mesh_ctx *mesh_context(void);
struct hdr *mesh_region(struct mesh_ctx *context);
size_t mesh_peers(struct mesh_ctx *context,uint16_t *peers,size_t capacity);
struct mesh_memory_span { const void *address; size_t bytes; };
int mesh_memory_view(const struct mesh_memory_span *spans,size_t count,void **address,size_t *bytes);
int mesh_memory_release(void *address,size_t bytes);
int mesh_attach(struct mesh_ctx *context,const char *name);
int mesh_detach(struct mesh_ctx *context);
#endif
