#include "mesh-mem.h"
#include <stdlib.h>
#include <string.h>

int mesh_mem_init(struct mesh_mem *m, struct ibv_pd *pd,
                  struct ibv_context *ctx, size_t granule){
  memset(m,0,sizeof *m);
  struct ibv_device_attr a;
  if(ibv_query_device(ctx,&a)) return -1;
  if(!granule) granule = 1;
  m->pd = pd;
  m->max_mr = (size_t)a.max_mr_size;
  m->chunk  = (m->max_mr / granule) * granule;   // no object straddles a boundary
  if(m->chunk == 0) return -1;                   // device cannot hold one granule
  return 0;
}

int mesh_mem_add(struct mesh_mem *m, void *addr, size_t len, int access){
  size_t need = (len + m->chunk - 1) / m->chunk;
  int base = m->nseg;
  if(m->nseg + (int)need > m->cap){
    int cap = m->cap ? m->cap : 8;
    while(cap < m->nseg + (int)need) cap *= 2;
    struct mesh_seg *s = realloc(m->seg, (size_t)cap * sizeof *s);
    if(!s) return -1;
    m->seg = s; m->cap = cap;
  }
  for(size_t off = 0; off < len; off += m->chunk){
    size_t n = len - off < m->chunk ? len - off : m->chunk;
    struct ibv_mr *mr = ibv_reg_mr(m->pd, (char*)addr + off, n, access);
    if(!mr){                                     // roll back this call entirely
      for(int i = base; i < m->nseg; i++) ibv_dereg_mr(m->seg[i].mr);
      m->nseg = base;
      return -1;
    }
    m->seg[m->nseg].base = (char*)addr + off;
    m->seg[m->nseg].len  = n;
    m->seg[m->nseg].mr   = mr;
    m->nseg++; m->bytes += n;
  }
  return 0;
}

const struct mesh_seg *mesh_mem_find(const struct mesh_mem *m, const void *addr){
  const char *p = (const char*)addr;
  int lo = 0, hi = m->nseg;                      // upper_bound on base
  while(lo < hi){ int mid = (lo + hi) / 2;
    if(m->seg[mid].base <= p) lo = mid + 1; else hi = mid; }
  if(lo == 0) return NULL;                       // before the first segment
  const struct mesh_seg *s = &m->seg[lo - 1];    // step back one, then verify
  return (p >= s->base && p < s->base + s->len) ? s : NULL;
}

uint32_t mesh_mem_lkey(const struct mesh_mem *m, const void *addr){
  const struct mesh_seg *s = mesh_mem_find(m, addr);
  return s ? s->mr->lkey : 0u;
}

void mesh_mem_release(struct mesh_mem *m){
  for(int i = 0; i < m->nseg; i++) if(m->seg[i].mr) ibv_dereg_mr(m->seg[i].mr);
  free(m->seg);
  m->seg = NULL; m->nseg = m->cap = 0; m->bytes = 0;
}
