// Registered-memory table for the mesh bridge.
//
// A device advertises max_mr_size, "largest contiguous block that can be
// registered" (ibv_query_device(3)). It is a real limit and it is not enforced
// by ibv_reg_mr: passing a larger length returns a valid-looking MR that covers
// only part of the range, and the failure surfaces later as a protection error
// on traffic that touches the tail. See kvcache-ai/Mooncake#2017 for the same
// bug in production.
//
// So the rule this file exists to enforce, once, for every caller:
//   - chunk above the registration call, never inside it
//   - one MR per chunk, each <= max_mr_size
//   - map an address to its MR by ordered lookup, not by arithmetic
//   - never silently register or deliver less than was asked for
#ifndef MESH_MEM_H
#define MESH_MEM_H
#include <infiniband/verbs.h>
#include <stddef.h>
#include <stdint.h>

struct mesh_seg { char *base; size_t len; struct ibv_mr *mr; };

struct mesh_mem {
  struct ibv_pd *pd;
  struct mesh_seg *seg;     // ascending by base; registration order is sorted
  int nseg, cap;
  size_t max_mr;            // from the device, never hardcoded
  size_t chunk;             // max_mr floored to a whole number of granules
  size_t bytes;             // total registered
};

// Reads max_mr_size from the device. granule is the largest object that must
// never straddle an MR boundary (the page size). Returns 0, or -1 if the
// device cannot hold even one granule per MR.
int  mesh_mem_init(struct mesh_mem *m, struct ibv_pd *pd,
                   struct ibv_context *ctx, size_t granule);

// Mapping is a cursor, not a bulk call. A buffer is handed over once; the
// cursor then advances as capacity allows. There is no "too large" and no
// all-or-nothing: bytes not yet mapped are simply still in the buffer, which
// is where they already were. Nothing is ever truncated, rejected or rolled
// back, because none of those are ever needed -- the data is not going
// anywhere while it waits.
struct mesh_map { char *base; size_t len, mapped; int access; };

void   mesh_map_open(struct mesh_map *it, void *addr, size_t len, int access);
// Advance the cursor by at most `budget` chunks. Returns bytes newly mapped
// (0 means no progress was possible this turn, which is not an error).
size_t mesh_map_step(struct mesh_mem *m, struct mesh_map *it, int budget);
static inline int mesh_map_done(const struct mesh_map *it){ return it->mapped >= it->len; }

// Ordered lookup: the segment containing addr, or NULL. O(log nseg).
const struct mesh_seg *mesh_mem_find(const struct mesh_mem *m, const void *addr);

// lkey for addr, or 0 if unregistered. 0 is never a valid lkey, so a caller
// that forgets to check gets a protection error rather than silent corruption.
uint32_t mesh_mem_lkey(const struct mesh_mem *m, const void *addr);

// Deregisters every MR, not just the first.
void mesh_mem_release(struct mesh_mem *m);
#endif
