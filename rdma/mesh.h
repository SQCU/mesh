// The mesh, as an application sees it.
//
// Everything below a page is the bridge's business: queue pairs, memory
// regions, lkeys, wire headers, residency, retransmission. None of it appears
// here and none of it should ever need to.
//
// An application attaches to a bridge that is already running, takes pages,
// fills them, sends them to a node, and gets pages back. That is the whole
// surface.
#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>

#define MESH_MAGIC   0x4d455348u
#define MESH_VERSION 2u
#define MESH_CL      128

// One page in flight. seq is assigned by the bridge and is only meaningful for
// reassembling an ordering the application itself imposed.
struct mesh_desc { uint32_t page, bytes, seq; uint16_t node, flags; };

// Single-producer single-consumer. Each ring has exactly one writer and one
// reader, so neither side ever needs a lock or a compare-and-swap.
struct mesh_ring { _Alignas(MESH_CL) _Atomic uint64_t head;
                   _Alignas(MESH_CL) _Atomic uint64_t tail; };

struct mesh_hdr {
  uint32_t magic, version, pgsz, npages;
  uint32_t ring_cap, node;
  uint64_t data_off, free_off, sub_off, cmp_off, rel_off;
  uint64_t bytes, node_ram, arena_off;
  uint32_t arena_pages, hdr_bytes;
  struct mesh_ring rfree;   // bridge -> app : pages you may use
  struct mesh_ring rsub;    // app -> bridge : send these
  struct mesh_ring rcmp;    // bridge -> app : these arrived
  struct mesh_ring rrel;    // app -> bridge : done with these
  _Alignas(MESH_CL) _Atomic uint64_t client_pid;
  _Alignas(MESH_CL) _Atomic uint64_t sent, recvd, lost, unrecovered;
};

struct mesh { struct mesh_hdr *h; unsigned char *base; size_t len; int fd; };

// ---- ring primitives, shared by the bridge and the client ----------------
static inline struct mesh_desc *mesh_slot(const struct mesh *M,
                                          uint64_t off, uint64_t i){
  return &((struct mesh_desc*)(M->base + off))[i % M->h->ring_cap];
}
static inline int mesh_push(struct mesh *M, struct mesh_ring *r, uint64_t off,
                            const struct mesh_desc *d){
  uint64_t h = atomic_load_explicit(&r->head, memory_order_relaxed);
  uint64_t t = atomic_load_explicit(&r->tail, memory_order_acquire);
  if(h - t >= M->h->ring_cap) return -1;              // full: caller retries later
  *mesh_slot(M, off, h) = *d;
  atomic_store_explicit(&r->head, h + 1, memory_order_release);
  return 0;
}
static inline int mesh_pop(struct mesh *M, struct mesh_ring *r, uint64_t off,
                           struct mesh_desc *d){
  uint64_t t = atomic_load_explicit(&r->tail, memory_order_relaxed);
  uint64_t h = atomic_load_explicit(&r->head, memory_order_acquire);
  if(t == h) return -1;                                // empty
  *d = *mesh_slot(M, off, t);
  atomic_store_explicit(&r->tail, t + 1, memory_order_release);
  return 0;
}

// ---- the application API ------------------------------------------------
//
// Three functions. Everything else in this header exists so the bridge and the
// client agree on the shape of the shared region, and no application needs to
// read it.
//
//   void *p = mesh_open(22e9, &stride, &usable);   // 22 GB of mesh memory
//   ... write into slot i at (char*)p + i*stride, up to `usable` bytes ...
//   mesh_write(p, n, 1);                           // put it on node 1
//   while((n = mesh_read(&q, &from))) { ... }      // take what arrives
//
// Memory comes in slots rather than one flat span because this hardware
// reports max_sge = 1 and has no immediate data, so every page must carry its
// own header contiguously. The gap is 24 bytes in 4096.

// Map `bytes` of mesh memory on this node. Returns the first slot, or NULL if
// the bridge is not running or is holding less than `bytes`. *stride is the
// distance between slots, *usable is how much of one is yours. Either may be
// NULL if you do not care.
void  *mesh_open(size_t bytes, size_t *stride, size_t *usable);

// Hand `nbytes` starting at `p` to `node`. Returns how many bytes were taken;
// call again with the remainder. It never truncates and never fails because
// something was too large -- bytes it did not take are still yours, where they
// already were.
size_t mesh_write(const void *p, size_t nbytes, int node);

// Take one slot that has arrived, or 0 if none has. The pointer is valid until
// the next call to mesh_read, which returns the previous slot to the mesh.
size_t mesh_read(void **p, int *from);
#endif
