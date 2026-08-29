// Registered memory for the mesh bridge.
#ifndef MESH_MEM_H
#define MESH_MEM_H
#include <infiniband/verbs.h>
#include <stddef.h>
#include <stdint.h>

#define MESH_CHUNK (1ull<<30)

struct mesh_mem {
  struct ibv_pd  *pd;
  char           *base;
  size_t          len, chunk;
  struct ibv_mr **mr;
  size_t          nseg;
};

int  mesh_mem_map(struct mesh_mem *m, struct ibv_pd *pd,
                  void *base, size_t len, size_t granule, int access);
void mesh_mem_release(struct mesh_mem *m);

static inline uint32_t mesh_lkey(const struct mesh_mem *m, const void *a){
  return m->mr[(size_t)((const char*)a - m->base) / m->chunk]->lkey;
}
#endif
