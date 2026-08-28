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

void mesh_map_open(struct mesh_map *it, void *addr, size_t len, int access){
  it->base=(char*)addr; it->len=len; it->mapped=0; it->access=access;
}

size_t mesh_map_step(struct mesh_mem *m, struct mesh_map *it, int budget){
  size_t progress=0;
  for(int k=0; k<budget && it->mapped < it->len; k++){
    size_t rest = it->len - it->mapped;
    size_t n = rest < m->chunk ? rest : m->chunk;
    if(m->nseg == m->cap){
      int cap = m->cap ? m->cap*2 : 8;
      struct mesh_seg *s = realloc(m->seg,(size_t)cap*sizeof *s);
      if(!s) break;                      // no progress this turn; try again later
      m->seg=s; m->cap=cap;
    }
    struct ibv_mr *mr = ibv_reg_mr(m->pd, it->base + it->mapped, n, it->access);
    if(!mr) break;                       // capacity exhausted for now, not a failure
    m->seg[m->nseg].base = it->base + it->mapped;
    m->seg[m->nseg].len  = n;
    m->seg[m->nseg].mr   = mr;
    m->nseg++; m->bytes += n; it->mapped += n; progress += n;
  }
  return progress;
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
