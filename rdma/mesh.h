
#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#define MESH_MAGIC   0x4d455348u
#define MESH_NAME    "/mesh0"
#define MESH_PORT    "18519"
#define MESH_MODE    0666
#define MESH_VERSION 5u
#define MESH_CL      128
#define MESH_RING    65536
#define MESH_OFF     16
#define RINGS        ((sizeof(struct hdr)+MESH_CL-1)/MESH_CL*MESH_CL)
enum { FREE, RECV, SEND, APP, NOWN };
enum { SUB, CMP, REL, ACK, NRING };
struct wire { uint16_t src, dst, hops; };
struct desc { uint32_t page, bytes; uint16_t node; };
struct ring { _Alignas(MESH_CL) _Atomic uint64_t head, tail; };
struct hdr {
  uint32_t magic, version, pgsz, pool, arena, node;
  uint64_t data_off;
  struct ring r[NRING];
  _Alignas(MESH_CL) _Atomic uint64_t client, sent, recvd, bad, up_ms;
  _Alignas(MESH_CL) _Atomic uint64_t mean[NOWN], sd[NOWN];
};
static inline unsigned char *mesh_at(struct hdr *m, uint32_t i){
  return (unsigned char*)m + m->data_off + (size_t)i * m->pgsz; }
static inline unsigned char *mesh_data(struct hdr *m, uint32_t i){
  return mesh_at(m,i) + sizeof(struct wire); }
static inline uint32_t mesh_pay(struct hdr *m){
  return m->pgsz - (uint32_t)sizeof(struct wire); }
static inline struct desc *slot(struct hdr *m, int k, uint64_t i){
  return &((struct desc*)((unsigned char*)m + RINGS))[k*MESH_RING + i%MESH_RING]; }
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
enum { K_DATA, K_FIN, K_REQ, K_OK };
enum { MS_RUN, MS_DONE, MS_FAIL };
struct mstream { char *buf; const char *src; size_t n, off, done, nb, hole;
                 unsigned char *seen; uint32_t sid; int node, rx, st; uint64_t fin_ack; };
struct mesh_ctx { struct hdr *M; unsigned char *arena; size_t len;
                  unsigned char *busy; size_t cursor;
                  unsigned char *pending; int *pending_nodes; uint32_t *pending_bytes;
                  size_t pending_head, pending_count, pending_capacity;
                  size_t inflight;
                  uint64_t sub, ack, ino, idle; int last; };
int    mesh_attach(struct mesh_ctx *c, const char *name);
int    mesh_turn(struct mesh_ctx *c, struct mstream **v, int k);
void   mesh_yell_start(struct mesh_ctx *c, struct mstream *s, const void *p, size_t n, int node, uint32_t sid);
int    mesh_lissen_start(struct mesh_ctx *c, struct mstream *s, void *p, size_t n, uint32_t sid);
int    mesh_scatter(struct mesh_ctx *c, struct mstream *ss, const void *p, size_t n, const int *nodes, int k, uint32_t sid0);
int    mesh_gather(struct mesh_ctx *c, struct mstream *ss, void *p, size_t n, int k, uint32_t sid0);
void  *mesh_open(size_t *nslots, size_t *stride, size_t *usable);
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
