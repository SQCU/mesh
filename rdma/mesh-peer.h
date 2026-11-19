#ifndef MESH_PEER_H
#define MESH_PEER_H
#include "mesh-call.h"

/* The peer service's groups (design/node-addressed-transfers.md): a group is a set of records, each one transfer of
   the node's order toward `peer` (mesh-call.h mesh_order_append), sending (`land` 0) or receiving (1) its `units`
   spans at region offset `spans`.  Both nodes' programs declare the same groups with the same matches and span
   lengths: a group's records are appended to the order each direction in `match` order, so the two ends' transfers
   meet by that order alone, with nothing exchanged.  A record's release gate: `reserved` 0, the word at `ready` (0:
   none) reaching `threshold`; `reserved` > 0, that many mesh_peer_gate at `ready`, each to hold.  Once every byte of
   it has crossed, the word at `completed` is raised to `value` (UINT64_MAX: lost).  All offsets are bytes from the
   region base */
struct mesh_peer_gate { uint64_t word, threshold; };
struct mesh_peer_transfer {
  uint64_t match,spans,units,ready,threshold,completed,value;
  uint32_t land,reserved;
};
/* 0 where this node's bridge holds a live order toward `peer`, else ENOLINK */
int mesh_peer_capable(struct mesh_ctx *,uint32_t peer);
/* the group's `count` records at region offset `records` appended to the order toward `peer`; its installation
   (the gates it made) in *installation.  0, EINVAL, ENOLINK, ENOMEM */
int mesh_peer_install(struct mesh_ctx *,uint32_t peer,uint64_t group,uint64_t records,
                      uint32_t count,uint64_t *installation);
/* the installation's order still live (nothing is waited for: no group is agreed at run time) */
int mesh_peer_wait(struct mesh_ctx *,uint64_t installation,uint64_t timeout_ns);
/* the installation's gates returned (held by the bridge until its transfers retire, mesh-dataflow.c mesh_pinned) */
int mesh_peer_release(struct mesh_ctx *,uint64_t installation);
#endif
