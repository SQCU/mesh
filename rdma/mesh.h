
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
#define MESH_VERSION 5u
#define MESH_CLIENT_DRAIN UINT64_MAX
#define MESH_CLIENT_DETACH (UINT64_MAX-1)
#define MESH_CL      128
#define MESH_RING    65536
#define MESH_OFF     16
#define RINGS        ((sizeof(struct hdr)+MESH_CL-1)/MESH_CL*MESH_CL)
enum { FREE, RECV, SEND, APP, NOWN };
enum { SUB, CMP, REL, ACK, NRING };
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_RETIRING, MESH_STOPPING, MESH_STOPPED };
struct wire { uint16_t src, dst, hops; };
struct desc { uint32_t page, bytes; uint16_t node; };
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
  uint64_t data_off;
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
};
_Static_assert(offsetof(struct hdr,mean)==offsetof(struct hdr,up_ms)+MESH_CL,"diagnostics must fit the existing header padding");
static inline unsigned char *mesh_at(struct hdr *m, uint32_t i){
  return (unsigned char*)m + m->data_off + (size_t)i * m->pgsz; }
static inline unsigned char *mesh_data(struct hdr *m, uint32_t i){
  return mesh_at(m,i) + sizeof(struct wire); }
static inline uint32_t mesh_pay(struct hdr *m){
  return m->pgsz - (uint32_t)sizeof(struct wire); }
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
struct shdr { uint64_t off; uint32_t sid, k; };
enum { K_DATA, K_FIN, K_REQ, K_OK, K_CLOSE, K_CLOSED, K_OPEN, K_READY,
       K_ABORT_RX, K_ABORT_TX, K_ABORTED_RX, K_ABORTED_TX };
struct mesh_epoch { uint64_t high, low; };
struct mesh_scope { struct mesh_epoch epoch; unsigned char plan[32]; };
enum { MS_RUN, MS_DONE, MS_FAIL };
struct mesh_page_bits { uint64_t published, pending; };
static inline size_t mesh_page_words(size_t pages){ return pages/64+(pages%64!=0); }
struct mstream { char *buf; const char *src; size_t n, off, done, nb;
                 union { size_t hole; size_t retry_ns; };
                 union { unsigned char *seen; struct mesh_page_bits *work; }; uint32_t sid; int node, rx, st; uint64_t fin_after_ns;
                 size_t stride, chunk; uint32_t *pages; _Atomic size_t available; int closing, closed;
                 struct mesh_scope scope; size_t logical_offset; int agreed, error, abort_ack, retain_seen;
                 size_t scan_word; int paged, resident, parked; uint64_t open_retry_ns; uint32_t *arrivals;
                 uint64_t *changed, *runnable, changed_bit; };
static inline void mesh_stream_schedule(const struct mstream *s){
  if(s->runnable) __atomic_fetch_or(s->runnable,s->changed_bit,__ATOMIC_RELEASE);
}
static inline void mesh_stream_changed(const struct mstream *s){
  if(s->changed) __atomic_fetch_or(s->changed,s->changed_bit,__ATOMIC_RELEASE);
  mesh_stream_schedule(s);
}
static inline int mesh_stream_published(const struct mstream *s,size_t page){
  return (__atomic_load_n(&s->work[page/64].published,__ATOMIC_ACQUIRE)>>(page%64))&1;
}
typedef int (*mesh_receive_fn)(void *, const void *, size_t, int);
struct mesh_ctx { struct hdr *M; unsigned char *arena; size_t len;
                  unsigned char *busy; size_t cursor;
                  unsigned char *pending; int *pending_nodes; uint32_t *pending_bytes;
                  size_t pending_head, pending_count, pending_capacity;
                  size_t inflight;
                  uint64_t sub, ack, ino, idle; int last; char *name; int mapping_pinned;
                  uint64_t integrity_failures, stale_frames; int name_owned, detaching;
                  _Atomic int executor_attached;
                  uint32_t *stream_index; size_t stream_index_capacity;
                  struct mstream **streams; size_t stream_count, stream_cursor;
                  mesh_receive_fn receiver; void *receiver_capture; };
struct mesh_ctx *mesh_context(void);
#ifdef __APPLE__
struct mesh_memory_span { const void *address; size_t bytes; };
int mesh_memory_view(const struct mesh_memory_span *spans, size_t count,
                     void **address, size_t *bytes);
int mesh_memory_release(void *address, size_t bytes);
#endif
void mesh_receiver(struct mesh_ctx *context, mesh_receive_fn receiver, void *capture);
int    mesh_try_attach(struct mesh_ctx *c, const char *name);
int    mesh_attach(struct mesh_ctx *c, const char *name);
int    mesh_detach(struct mesh_ctx *c);
int    mesh_link_reset(struct mesh_ctx *c, size_t port);
int    mesh_turn(struct mesh_ctx *c, struct mstream **v, int k);
int    mesh_turn_window(struct mesh_ctx *c, struct mstream **v, int k, size_t window);
int    mesh_poll_streams(struct mesh_ctx *c, struct mstream **v, int k, uint64_t *now_ns);
int    mesh_progress_stream(struct mesh_ctx *c, struct mstream *s, size_t window, uint64_t now_ns);
void  *mesh_yell_view(struct mesh_ctx *c, struct mstream *s, size_t n, int node, uint32_t sid);
void  *mesh_stream_lease(struct mesh_ctx *c, struct mstream *s);
int    mesh_stream_idle(struct mesh_ctx *c, const struct mstream *s);
int    mesh_stream_push_page(struct mesh_ctx *c, struct mstream *s, size_t page, size_t window);
ptrdiff_t mesh_stream_push_pages(struct mesh_ctx *c, struct mstream *s, size_t window);
int    mesh_stream_receive(struct mesh_ctx *c, struct mstream *s, uint32_t *pages);
int    mesh_stream_reserve(struct mesh_ctx *c, size_t count);
int    mesh_stream_register(struct mesh_ctx *c, struct mstream **streams, size_t count);
int    mesh_stream_conflict(const struct mesh_ctx *c, const struct mstream *stream, struct mesh_epoch epoch);
int    mesh_lissen_view(struct mesh_ctx *c, struct mstream *s, uint32_t *pages, size_t n, uint32_t sid);
size_t mesh_release_view(struct mesh_ctx *c, struct mstream *s);
void   mesh_yell_start(struct mesh_ctx *c, struct mstream *s, const void *p, size_t n, int node, uint32_t sid);
int    mesh_lissen_start(struct mesh_ctx *c, struct mstream *s, void *p, size_t n, uint32_t sid);
int    mesh_scatter(struct mesh_ctx *c, struct mstream *ss, const void *p, size_t n, const int *nodes, int k, uint32_t sid0);
int    mesh_gather(struct mesh_ctx *c, struct mstream *ss, void *p, size_t n, int k, uint32_t sid0);
void  *mesh_open(size_t *nslots, size_t *stride, size_t *usable);
void  *mesh_try_open(size_t *nslots, size_t *stride, size_t *usable);
int    mesh_close(void);
size_t mesh_write(const void *p, size_t nbytes, int node);
size_t mesh_write_copy(const void *p, size_t stride, size_t bytes, size_t nslots, int node);
size_t mesh_queue_copy(const void *p, size_t stride, size_t bytes, size_t nslots, int node);
size_t mesh_pump(void);
size_t mesh_queued(void);
size_t mesh_inflight(void);
size_t mesh_read(void **p, int *from);
size_t mesh_readv(void *p, size_t stride, uint32_t *sizes, int *from, size_t count);
size_t mesh_yell(const void *p, size_t n, int node);
size_t mesh_lissen(void *p, size_t n);
#endif
