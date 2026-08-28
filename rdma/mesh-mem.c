#include "mesh-mem.h"
#include <stdlib.h>
#include <string.h>

int mesh_mem_map(struct mesh_mem *m, struct ibv_pd *pd,
                 void *base, size_t len, size_t granule, int access){
  memset(m,0,sizeof *m);
  if(!granule) granule = 1;
  m->pd    = pd;
  m->base  = (char*)base;
  m->len   = len;
  m->chunk = (MESH_CHUNK / granule) * granule;      // nothing straddles a boundary
  if(!m->chunk) return -1;
  m->nseg  = (len + m->chunk - 1) / m->chunk;       // known before registering
  m->mr    = calloc(m->nseg, sizeof *m->mr);
  if(!m->mr) return -1;
  for(size_t i = 0; i < m->nseg; i++){
    size_t off = i * m->chunk;
    size_t n   = len - off < m->chunk ? len - off : m->chunk;
    m->mr[i] = ibv_reg_mr(pd, m->base + off, n, access);
    if(!m->mr[i]){ mesh_mem_release(m); return -1; }
  }
  return 0;
}

void mesh_mem_release(struct mesh_mem *m){
  for(size_t i = 0; i < m->nseg; i++) if(m->mr[i]) ibv_dereg_mr(m->mr[i]);
  free(m->mr);
  memset(m,0,sizeof *m);
}
