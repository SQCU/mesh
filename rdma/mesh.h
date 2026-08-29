// The mesh, as an application sees it.
#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>

#define MESH_MAGIC   0x4d455348u
#define MESH_VERSION 2u
#define MESH_CL      128

struct wire { uint32_t magic, bytes; uint16_t src, dst; uint8_t hops, pad[3]; };
#define WIRE_MAGIC 0x4d534831u

struct mesh_desc { uint32_t page, bytes; uint16_t node, pad; };

struct mesh_ring { _Alignas(MESH_CL) _Atomic uint64_t head;
                   _Alignas(MESH_CL) _Atomic uint64_t tail; };

struct mesh_hdr {
  uint32_t magic, version, pgsz, npages;
  uint32_t ring_cap, node;
  uint64_t data_off, sub_off, cmp_off, rel_off;
  uint64_t bytes, node_ram, arena_off;
  uint32_t arena_pages, hdr_bytes;
  struct mesh_ring rsub;
  struct mesh_ring rcmp;
  struct mesh_ring rrel;
  _Alignas(MESH_CL) _Atomic uint64_t client_pid;
  _Alignas(MESH_CL) _Atomic uint64_t sent, recvd, lost, unrecovered;

  _Alignas(MESH_CL) _Atomic uint64_t occ_pool;

  _Alignas(MESH_CL) _Atomic uint64_t sd_free, sd_posted, sd_sending, uptime_ms;
  _Alignas(MESH_CL) _Atomic uint64_t mean_free, mean_posted, mean_sending, mean_held, sd_held;
};

struct mesh { struct mesh_hdr *h; unsigned char *base; size_t len; int fd; };

static inline struct mesh_desc *mesh_slot(const struct mesh *M,
                                          uint64_t off, uint64_t i){
  return &((struct mesh_desc*)(M->base + off))[i % M->h->ring_cap];
}
static inline int mesh_push(struct mesh *M, struct mesh_ring *r, uint64_t off,
                            const struct mesh_desc *d){
  uint64_t h = atomic_load_explicit(&r->head, memory_order_relaxed);
  uint64_t t = atomic_load_explicit(&r->tail, memory_order_acquire);
  if(h - t >= M->h->ring_cap) return -1;
  *mesh_slot(M, off, h) = *d;
  atomic_store_explicit(&r->head, h + 1, memory_order_release);
  return 0;
}
static inline int mesh_pop(struct mesh *M, struct mesh_ring *r, uint64_t off,
                           struct mesh_desc *d){
  uint64_t t = atomic_load_explicit(&r->tail, memory_order_relaxed);
  uint64_t h = atomic_load_explicit(&r->head, memory_order_acquire);
  if(t == h) return -1;
  *d = *mesh_slot(M, off, t);
  atomic_store_explicit(&r->tail, t + 1, memory_order_release);
  return 0;
}

void  *mesh_open(size_t bytes, size_t *stride, size_t *usable);

size_t mesh_write(const void *p, size_t nbytes, int node);

size_t mesh_read(void **p, int *from);
#endif
