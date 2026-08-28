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

// Registers [addr, addr+len) in full. Returns 0, or -1 having registered
// nothing new. It does not clamp, and it does not partially succeed quietly.
int  mesh_mem_add(struct mesh_mem *m, void *addr, size_t len, int access);

// Ordered lookup: the segment containing addr, or NULL. O(log nseg).
const struct mesh_seg *mesh_mem_find(const struct mesh_mem *m, const void *addr);

// lkey for addr, or 0 if unregistered. 0 is never a valid lkey, so a caller
// that forgets to check gets a protection error rather than silent corruption.
uint32_t mesh_mem_lkey(const struct mesh_mem *m, const void *addr);

// Deregisters every MR, not just the first.
void mesh_mem_release(struct mesh_mem *m);
#endif
