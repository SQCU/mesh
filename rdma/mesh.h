// see RDMA-FIRST.md
#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#define MESH_MAGIC   0x4d455348u
#define WIRE_MAGIC   0x4d534831u
#define MESH_NAME    "/mesh0"
#define MESH_MODE    0666
#define MESH_VERSION 3u
#define MESH_CL      128
#define MESH_RING    4096
#define RINGS        ((sizeof(struct hdr)+MESH_CL-1)/MESH_CL*MESH_CL)
enum { FREE, RECV, SEND, APP, NOWN };
enum { SUB, CMP, REL, NRING };
struct wire { uint32_t magic, bytes; uint16_t src, dst, hops; };
struct desc { uint32_t page, bytes; uint16_t node; };
struct ring { _Alignas(MESH_CL) _Atomic uint64_t head, tail; };
struct hdr {
  uint32_t magic, version, pgsz, pool, arena, node;
  uint64_t data_off;
  struct ring r[NRING];
  _Alignas(MESH_CL) _Atomic uint64_t client, sent, recvd, up_ms;
  _Alignas(MESH_CL) _Atomic uint64_t mean[NOWN], sd[NOWN];
};
static inline unsigned char *mesh_at(struct hdr *m, uint32_t i){
  return (unsigned char*)m + m->data_off + (size_t)i * m->pgsz; }
static inline unsigned char *mesh_data(struct hdr *m, uint32_t i){
  return mesh_at(m,i) + sizeof(struct wire); }
static inline uint32_t mesh_pay(struct hdr *m){
  return m->pgsz - (uint32_t)sizeof(struct wire); }
static inline uint32_t mesh_clamp(struct hdr *m, uint32_t n){
  uint32_t p=mesh_pay(m); return n>p?p:n; }
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
void  *mesh_open(size_t bytes, size_t *stride, size_t *usable);
size_t mesh_write(const void *p, size_t nbytes, int node);
size_t mesh_read(void **p, int *from);
#endif
