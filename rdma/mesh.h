// see RDMA-FIRST.md
#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#define MESH_MAGIC   0x4d455348u
#define MESH_VERSION 3u
#define MESH_CL      128
#define MESH_RING    4096
#define RINGS        ((sizeof(struct hdr)+MESH_CL-1)/MESH_CL*MESH_CL)
enum { FREE, RECV, SEND, APP, NOWN };
struct wire { uint32_t magic, bytes; uint16_t src, dst, hops; };
struct desc { uint32_t page, bytes, node; };
struct ring { _Alignas(MESH_CL) _Atomic uint64_t head, tail; };
struct hdr {
  uint32_t magic, version, pgsz, pool, arena, node;
  uint64_t data_off;
  struct ring sub, cmp, rel;
  _Alignas(MESH_CL) _Atomic uint64_t client, sent, recvd, up_ms;
  _Alignas(MESH_CL) _Atomic uint64_t mean[NOWN], sd[NOWN];
};
struct mesh { struct hdr *h; unsigned char *b; };
static inline struct desc *slot(struct mesh *m, int r, uint64_t i){
  return &((struct desc*)(m->b + RINGS))[r*MESH_RING + i%MESH_RING]; }
static inline int push(struct mesh *m, struct ring *q, int r, const struct desc *d){
  uint64_t h=atomic_load_explicit(&q->head,memory_order_relaxed);
  if(h - atomic_load_explicit(&q->tail,memory_order_acquire) >= MESH_RING) return -1;
  *slot(m,r,h)=*d; atomic_store_explicit(&q->head,h+1,memory_order_release); return 0; }
static inline int pop(struct mesh *m, struct ring *q, int r, struct desc *d){
  uint64_t t=atomic_load_explicit(&q->tail,memory_order_relaxed);
  if(t == atomic_load_explicit(&q->head,memory_order_acquire)) return -1;
  *d=*slot(m,r,t); atomic_store_explicit(&q->tail,t+1,memory_order_release); return 0; }
void  *mesh_open(size_t bytes, size_t *stride, size_t *usable);
size_t mesh_write(const void *p, size_t nbytes, int node);
size_t mesh_read(void **p, int *from);
#endif
