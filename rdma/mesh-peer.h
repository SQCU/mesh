#ifndef MESH_PEER_H
#define MESH_PEER_H
#include "mesh-call.h"

/* A logical record names one match in a peer group. All offsets are bytes from
   the region base. A threshold of zero opens the record immediately. */
struct mesh_peer_transfer {
  uint64_t match,spans,units,ready,threshold,completed,value;
  uint32_t land,reserved;
};
int mesh_peer_capable(struct mesh_ctx *,uint32_t peer);
int mesh_peer_install(struct mesh_ctx *,uint32_t peer,uint64_t group,uint64_t records,
                      uint32_t count,uint64_t *installation);
int mesh_peer_wait(struct mesh_ctx *,uint64_t installation,uint64_t timeout_ns);
int mesh_peer_release(struct mesh_ctx *,uint64_t installation);
#endif
