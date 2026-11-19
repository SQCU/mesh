#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"
/* design/algorithm-sources.md#programtensor: a lease of `count` slots of `bytes`, its rows and arena pages claimed by
   compare-exchange (mesh-dataflow.c mesh_claim): a slot `pages` arena pages on `stride` rows from row `first`; 0, or the
   refusal as a value (ENOMEM, ENOSPC) */
struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride; };
int mesh_section_create(struct mesh_ctx *,size_t bytes,uint32_t count,struct mesh_section *);
void *mesh_section_address(struct mesh_ctx *,struct mesh_section,uint32_t index);
/* design/node-addressed-transfers.md: the nodes this node's bridge holds a live order toward, each with the bandwidth of
   the cables under it (mesh-plan.h struct mesh_reach), at most `room`; their count */
uint32_t mesh_nodes_reached(struct hdr *,struct mesh_reach *,uint32_t room);
/* design/node-addressed-transfers.md: the order toward node `node` as a client uses it: its slot and the generation
   it was opened in.  A client holds nothing by opening it; any number of clients use one order at once */
struct mesh_order { uint32_t slot,node; uint32_t generation,padding; };
/* the order toward `node`: 0, EINVAL, ENOLINK where this node's bridge holds no live order toward it */
int mesh_order_open(struct mesh_ctx *,uint32_t node,struct mesh_order *);
/* whether the order is still the generation it was opened in, and live (a restart of it loses every transfer of the
   generation: their done words ~0) */
int mesh_order_live(struct mesh_ctx *,const struct mesh_order *);
/* a transfer appended to the order (`way` 0: bytes this node sends, 1: bytes it receives), its generation the order's
   as opened: 0; EINVAL (an invalid table, gate or done word); EAGAIN where the ring holds MESH_RING transfers not yet
   retired (append it again); ENOLINK where the order is not the one opened */
int mesh_order_append(struct mesh_ctx *,const struct mesh_order *,int way,const struct mesh_crossing *);
/* the transfers outstanding in an order's direction (appended, not retired) */
uint64_t mesh_order_outstanding(struct mesh_ctx *,const struct mesh_order *,int way);
#endif
