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
  uint64_t bytes, node_ram;
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

// ---- the application API -------------------------------------------------
int    mesh_attach(struct mesh *M, const char *name);
void   mesh_detach(struct mesh *M);

static inline uint32_t mesh_pagesize(const struct mesh *M){ return M->h->pgsz; }
static inline uint32_t mesh_pages(const struct mesh *M){ return M->h->npages; }
static inline uint16_t mesh_node(const struct mesh *M){ return (uint16_t)M->h->node; }
// How much of this machine the mesh is holding, as a percentage.
static inline double mesh_pct_of_node(const struct mesh *M){
  return M->h->node_ram ? 100.0 * (double)M->h->bytes / (double)M->h->node_ram : 0.0;
}

// Take a page. -1 means none are free this instant, which is backpressure and
// not an error; the caller does something else and asks again.
int    mesh_acquire(struct mesh *M, uint32_t *page);
// Address of a page. Valid until it is sent or released.
static inline void *mesh_page(struct mesh *M, uint32_t page){
  return M->base + M->h->data_off + (size_t)page * M->h->pgsz;
}
// Hand a page to the mesh for delivery to `node`. Ownership passes to the
// bridge; the page comes back to the free pool on its own.
int    mesh_send(struct mesh *M, uint32_t page, uint32_t bytes, uint16_t node);
// A page that arrived, or -1 if none is waiting.
int    mesh_poll(struct mesh *M, uint32_t *page, uint32_t *bytes, uint16_t *from);
// Give a received page back.
void   mesh_release(struct mesh *M, uint32_t page);
#endif
