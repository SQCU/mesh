#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"
/* design/algorithm-sources.md#programtensor: a lease of `count` slots of `bytes`, its rows and arena pages claimed by
   compare-exchange (mesh-dataflow.c mesh_claim); 0, or the refusal as a value (ENOMEM, ENOSPC) */
struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride; };
int mesh_section_create(struct mesh_ctx *,size_t bytes,uint32_t count,struct mesh_section *);
void *mesh_section_address(struct mesh_ctx *,struct mesh_section,uint32_t index);
/* design/transport.md section 14: the nodes this node's bridge has a live stream toward, each with its stream's
   bandwidth (mesh-plan.h struct mesh_reach), at most `room`; their count */
uint32_t mesh_nodes_reached(struct hdr *,struct mesh_reach *,uint32_t room);
/* design/transport.md section 14: the stream toward node `node`, held by this client as its rows are: taken while it is
   free (its owner word 0 to the client's own), each class's first unit then the stream's `next` (next[0] large,
   next[1] small): 0, EINVAL, ENOLINK where no live stream goes to the node, or EBUSY where another client holds it or
   its last holder's units are still going; let go (its owner word back to 0), after which its bridge sends every unit
   it declared, the unsupplied ones as duds */
int mesh_stream_take(struct mesh_ctx *,uint32_t node,uint64_t next[2]);
void mesh_stream_give(struct mesh_ctx *,uint32_t node);
/* design/transport.md section 14: a held stream's class (`large` 1: its large units, 0: its small), as the client maps
   it: its in and out rings, a ready word and a landed word a unit, its unit's bytes and its units; 0, EINVAL, EPERM
   where the client does not hold it, or ENOLINK.  The holder supplies unit u (its sequence in the class, from the
   class's `next`) once slot u mod units of the out ring holds the unit's bytes: ready[u mod units] := 1 + u / units,
   after a release fence; unit u from the peer has landed in slot u mod units of the in ring once landed[u mod units]
   reads at least 1 + u / units (~0: the link lost); the peer's units begin at the peer's `next`, which the
   communicator's init exchanges.  The holder declares the units of a class it will supply, up to `declared`
   (absolute) by `deadline` (uptime ns, at most MESH_PLATFORM_NS away): mesh_stream_declare; a declared unit not
   supplied by then, or by the holding's end, goes as a dud (all ones) */
struct mesh_stream_view { unsigned char *in,*out; _Atomic uint64_t *ready,*landed; uint32_t unit,units; };
int mesh_stream_view(struct mesh_ctx *,uint32_t node,int large,struct mesh_stream_view *);
int mesh_stream_declare(struct mesh_ctx *,uint32_t node,int large,uint64_t declared,uint64_t deadline);
#endif
