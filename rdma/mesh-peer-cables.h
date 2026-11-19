#ifndef MESH_PEER_CABLES_H
#define MESH_PEER_CABLES_H
#include "mesh-call.h"
static inline int mesh_peer_cables(uint32_t a,uint32_t xnode,const struct mesh_endpoint *x,uint32_t xn,
  uint32_t b,uint32_t ynode,const struct mesh_endpoint *y,uint32_t yn,struct mesh_cable *cables,uint32_t *count,int require_all){
    for (uint32_t i = 0; i < xn; i++) {
      const struct mesh_endpoint *one = x + i;
      if (one->node != ynode) continue;
      uint32_t earlier = 0;
      for (uint32_t k = 0; k < i; k++) earlier |= x[k].bundle == one->bundle;
      if (earlier) continue;
      uint64_t wire = 0, bandwidth = 0;
      int reciprocal = 0;
      for (uint32_t k = i; k < xn; k++) {
        const struct mesh_endpoint *u = x + k;
        if (u->bundle != one->bundle) continue;
        for (uint32_t j = 0; j < yn; j++) {
          const struct mesh_endpoint *v = y + j;
          if (v->node != xnode || v->wire != u->wire) continue;
          reciprocal = 1;
          if (!u->available || !v->available || (wire && wire < u->wire)) continue;
          wire = u->wire;
          bandwidth = u->bandwidth < v->bandwidth ? u->bandwidth : v->bandwidth;
        }
      }
      if (!wire) { if (require_all) return reciprocal ? EBUSY : ENOLINK; continue; }
      cables[(*count)++] = (struct mesh_cable){wire, bandwidth, {a, b}};
    }
    for (uint32_t j = 0; j < yn; j++) {
      const struct mesh_endpoint *v = y + j;
      if (v->node != xnode) continue;
      int reciprocal = 0;
      for (uint32_t i = 0; i < xn; i++) reciprocal |= x[i].node == ynode && x[i].wire == v->wire;
      if (!reciprocal && require_all) return ENOLINK;
    }
  return 0;
}
#endif
