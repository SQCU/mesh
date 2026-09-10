
#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <string.h>
#define MESH_MAGIC   0x4d455348u
#define MESH_NAME    "/mesh0"
#define MESH_PORT    "18519"
#define MESH_MODE    0666
#define MESH_VERSION 7u
#define MESH_HEADER_BYTES 64u
#define MESH_HEADER_STRIDE 128u
#define MESH_CL      128
#define MESH_RING    65536
#define MESH_OFF     16
#define RINGS        ((sizeof(struct hdr)+MESH_CL-1)/MESH_CL*MESH_CL)
enum { FREE, RECV, SEND, APP, NOWN };
enum { SUB, CMP, REL, ACK, NRING };
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_RETIRING, MESH_STOPPING, MESH_STOPPED };
struct wire { uint16_t src, dst, hops; };
struct desc { uint32_t page, bytes; uint16_t node; uint16_t reserved; uint32_t error, domain; uint64_t header; };
struct mesh_page_header { struct wire wire; uint16_t padding; uint64_t table, stamp; uint32_t source, target; uint64_t when; int64_t code; uint32_t domain, function, index, peer; };
struct mesh_send { struct mesh_page_header header; uint32_t page, reserved; uint64_t next, previous, owner, padding[4]; };
struct ring { _Alignas(MESH_CL) _Atomic uint64_t head, tail; };
static inline int ring_select(const struct ring *q, uint64_t *cursor, uint64_t *index){
  uint64_t tail=atomic_load_explicit(&q->tail,memory_order_relaxed);
  uint64_t head=atomic_load_explicit(&q->head,memory_order_acquire);
  if(head==tail) return 0;
  if(*cursor-tail>=head-tail) *cursor=tail;
  *index=(*cursor)++; return 1;
}
static inline void ring_erase(struct ring *q, void *entries, size_t stride, size_t capacity, uint64_t index){
  uint64_t tail=atomic_load_explicit(&q->tail,memory_order_relaxed);
  if(index!=tail) memcpy((char*)entries+(index%capacity)*stride,(char*)entries+(tail%capacity)*stride,stride);
  atomic_store_explicit(&q->tail,tail+1,memory_order_release);
}
struct hdr {
  uint32_t magic, version, pgsz, pool, arena, node;
  uint64_t headers_off, data_off;
  struct ring r[NRING];
  _Alignas(MESH_CL) _Atomic uint64_t client, sent, recvd, bad, up_ms;
  _Atomic uint64_t bridge_pid, heartbeat_ms, phase, operation, operation_ms;
  _Atomic uint64_t port_count;
  _Alignas(MESH_CL) _Atomic uint64_t mean[NOWN], sd[NOWN];
};
struct mesh_port_info {
  char device[32];
  uint16_t peer;
  _Atomic uint64_t phase, heartbeat_us, operation, generation, reset_request;
  uint64_t when; int64_t code; uint32_t domain, reserved;
};
_Static_assert(offsetof(struct hdr,mean)==offsetof(struct hdr,up_ms)+MESH_CL,"diagnostics must fit the existing header padding");
static inline unsigned char *mesh_at(struct hdr *m, uint32_t i){
  return (unsigned char*)m + m->data_off + (size_t)i * m->pgsz; }
// ../design/algorithm-sources.md#contiguous-backing-page-views
static inline struct mesh_page_header *mesh_header(struct hdr *m, uint32_t i){
  return (struct mesh_page_header*)((unsigned char*)m+m->headers_off+(size_t)i*MESH_HEADER_STRIDE); }
// ../design/algorithm-sources.md#context-lifetime
static inline struct mesh_row *mesh_context_row(struct hdr *memory,uint32_t page){
  return (struct mesh_row*)((struct mesh_send*)mesh_header(memory,page))->padding;
}
static inline unsigned char *mesh_data(struct hdr *m, uint32_t i){
  return mesh_at(m,i); }
static inline uint32_t mesh_pay(struct hdr *m){
  return m->pgsz; }
static inline struct desc *slot(struct hdr *m, int k, uint64_t i){
  return &((struct desc*)((unsigned char*)m + RINGS))[k*MESH_RING + i%MESH_RING]; }
static inline struct mesh_port_info *mesh_ports(struct hdr *m){
  return (struct mesh_port_info*)((unsigned char*)m+RINGS+NRING*MESH_RING*sizeof(struct desc)); }
static inline int push(struct hdr *m, int k, const struct desc *d){
  struct ring *q=&m->r[k];
  uint64_t h=atomic_load_explicit(&q->head,memory_order_relaxed);
  if(h - atomic_load_explicit(&q->tail,memory_order_acquire) >= MESH_RING) return -1;
  *slot(m,k,h)=*d; atomic_store_explicit(&q->head,h+1,memory_order_release); return 0; }
static inline int pop(struct hdr *m, int k, struct desc *d){
  struct ring *q=&m->r[k];
  uint64_t t=atomic_load_explicit(&q->tail,memory_order_relaxed);
  if(t == atomic_load_explicit(&q->head,memory_order_acquire)) return -1;
  *d=*slot(m,k,t); atomic_store_explicit(&q->tail,t+1,memory_order_release); return 0; }
struct mesh_rows;
struct mesh_ctx {
  struct hdr *M; unsigned char *arena; size_t len;
  struct mesh_rows **tables; size_t table_count, allocation;
  int mapping_pinned;
};
// ../design/algorithm-sources.md#context-lifetime
static inline uint32_t mesh_received_pages(const struct mesh_ctx *context){ return context->M->pool; }
struct mesh_ctx *mesh_context(void);
struct hdr *mesh_region(struct mesh_ctx *context);
size_t mesh_peers(struct mesh_ctx *context,uint16_t *peers,size_t capacity);
#ifdef __APPLE__
struct mesh_memory_span { const void *address; size_t bytes; };
int mesh_memory_view(const struct mesh_memory_span *spans, size_t count,
                     void **address, size_t *bytes);
int mesh_memory_release(void *address, size_t bytes);
#endif
int mesh_attach(struct mesh_ctx *context,const char *name);
int mesh_detach(struct mesh_ctx *context);
const struct mesh_page_header *mesh_context_metadata(struct mesh_ctx *context,uint32_t page);
int mesh_context_consume(struct mesh_ctx *context,uint32_t page);
int mesh_link_reset(struct mesh_ctx *context,size_t port);
#endif
