// Registered memory for the mesh bridge.
//
// One contiguous span, cut into equal chunks, each chunk one memory region.
// Because the chunks are equal, an address maps to its region by division:
// there is no table to search, no address that can miss, and no lookup that
// can fail. Nothing here branches on the data it is given.
//
// Setup may fail and says so. The data path may not, and does not appear here.
#ifndef MESH_MEM_H
#define MESH_MEM_H
#include <infiniband/verbs.h>
#include <stddef.h>
#include <stdint.h>

// The device advertises max_mr_size = 16.384 MB. It is not enforced, it is
// unusable at that size, and exceeding the real limit corrupts silently rather
// than failing, so the chunk is a fixed constant chosen well inside it and is
// not derived from the device or selectable by anyone. See RDMA-FIRST.md.
#define MESH_CHUNK (1ull<<30)

struct mesh_mem {
  struct ibv_pd  *pd;
  char           *base;
  size_t          len, chunk;
  struct ibv_mr **mr;
  size_t          nseg;
};

// Register [base, base+len). `granule` is the largest object that must not
// straddle a region boundary; the chunk is rounded down to a multiple of it.
// Registers all of it or none of it, and reports which.
int  mesh_mem_map(struct mesh_mem *m, struct ibv_pd *pd,
                  void *base, size_t len, size_t granule, int access);
void mesh_mem_release(struct mesh_mem *m);

// Division, not search. Defined for every address in the span.
static inline uint32_t mesh_lkey(const struct mesh_mem *m, const void *a){
  return m->mr[(size_t)((const char*)a - m->base) / m->chunk]->lkey;
}
#endif
