#ifndef MESH_PEER_H
#define MESH_PEER_H
#include "mesh-call.h"

/* The peer service's groups (design/declared-programs.md): a group is a set of records, each one transfer toward
   `peer`, sending (`land` 0) or receiving (1) its `units` spans at region offset `spans` (any lengths: the service cuts
   each into pieces).  Both nodes' programs declare the same groups with the same matches and span lengths: a group is
   one declared program of one execution (its key the group), each direction's records in `match` order, so the two
   ends' transfers meet by that order alone, with nothing exchanged.  A record's release gate: the word at `ready` (0:
   none) reaching `threshold`, for the whole record.  Once every byte of it has crossed, the word at `completed` is
   raised to `value` (UINT64_MAX: lost).  `reserved` is 0.  All offsets are bytes from the region base */
struct mesh_peer_transfer {
  uint64_t match,spans,units,ready,threshold,completed,value;
  uint32_t land,reserved;
};
/* 0 where this node's bridge holds a live order toward `peer`, else ENOLINK */
int mesh_peer_capable(struct mesh_ctx *,uint32_t peer);
/* the group's `count` records at region offset `records` declared toward `peer` as one program and issued (the
   client's group before it, if any, ends where it was issued); its installation in *installation.  0, EINVAL, ENOLINK,
   ENOMEM, or mesh_declare's refusals */
int mesh_peer_install(struct mesh_ctx *,uint32_t peer,uint64_t group,uint64_t records,
                      uint32_t count,uint64_t *installation);
/* the installation's order still live (nothing is waited for: no group is agreed at run time) */
int mesh_peer_wait(struct mesh_ctx *,uint64_t installation,uint64_t timeout_ns);
/* the installation's program ended (mesh-call.h mesh_close) and its tables' lease returned (held by the bridge until the
   program is wholly done, mesh-dataflow.c mesh_pinned) */
int mesh_peer_release(struct mesh_ctx *,uint64_t installation);
#endif
