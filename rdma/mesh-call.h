#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"
/* design/algorithm-sources.md#programtensor: a lease of `count` slots of `bytes`, its rows and arena pages claimed by
   compare-exchange (mesh-dataflow.c mesh_claim); 0, or the refusal as a value (ENOMEM, ENOSPC) */
struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride; };
int mesh_section_create(struct mesh_ctx *,size_t bytes,uint32_t count,struct mesh_section *);
void *mesh_section_address(struct mesh_ctx *,struct mesh_section,uint32_t index);
/* design/transport.md section 14: this node's bridge's links with a live stream, one entry a link in the order of its
   ports (mesh-plan.h struct mesh_reach): the node it reaches, its live streams and their shares of its bandwidth
   together (mesh-link.c streams_open: a link's streams are the region's MESH_LINK_STREAMS from its port's index on,
   mesh-observe.h), at most `room`; their count.  Two cables toward one node are two entries */
uint32_t mesh_nodes_reached(struct hdr *,struct mesh_reach *,uint32_t room);
/* design/transport.md section 14: the streams free to take, each one's node and its `wire` (struct mesh_stream: its
   identity, alike at both its ends), at most `room`; their count.  The two ends of a communicator take the stream of one
   wire between their nodes, one free at both: its init exchanges these */
uint32_t mesh_streams_free(struct hdr *,uint32_t *node,uint64_t *wire,uint32_t room);
/* design/transport.md section 14: the live stream of identity `wire`, held by this client as its rows are: taken while
   it is free (its owner word 0 to the client's own), `*stream` its index, each class's first unit then the stream's
   `next` (next[0] large, next[1] small): 0, EINVAL, ENOLINK where no live stream has the identity, or EBUSY where
   another client holds it or its last holder's units are still going; let go by its index (its owner word back to 0),
   after which its bridge sends every unit it declared, the unsupplied ones as duds */
int mesh_stream_take(struct mesh_ctx *,uint64_t wire,uint32_t *stream,uint64_t next[2]);
void mesh_stream_give(struct mesh_ctx *,uint32_t stream);
/* design/transport.md section 14, "A process a port": whether stream `stream` is still this client's holding of identity
   `wire`: live, its owner word this client's, its identity the one taken.  A pairing lost and made again opens every
   stream of the link new (no owner): a holding of the lost pairing reads 0 here, and its holder starts no group on it */
int mesh_stream_held(struct mesh_ctx *,uint32_t stream,uint64_t wire);
/* design/transport.md section 14: a held stream's class (`large` 1: its large units, 0: its small), as the client maps
   it: its in and out rings, a ready word and a landed word a unit, its unit's bytes and its units; 0, EINVAL, EPERM
   where the client does not hold it, or ENOLINK.  The holder supplies unit u (its sequence in the class, from the
   class's `next`) once slot u mod units of the out ring holds the unit's bytes: ready[u mod units] := 1 + u / units,
   after a release fence (with mesh.h MESH_DUD: supplied as a dud, which its bridge sends as one, as every later unit of
   the holding); unit u from the peer has landed in slot u mod units of the in ring once landed[u mod units], its bits
   but MESH_DUD, reads at least 1 + u / units, and it is the peer's own unit where MESH_DUD is clear (set: it came as a
   dud, or, ~0, the link was lost; either fails the consumer's group, design/transport.md section 10); the peer's units
   begin at the peer's `next`, which the communicator's init exchanges.  The holder declares the units of a class it
   will supply, up to `declared` (absolute) and its `deadline` (uptime ns): mesh_stream_declare; a live holder's is
   UINT64_MAX, never reached, and a holder that can supply no more sets it to now; a declared unit not supplied by its
   deadline, or by the holding's end, goes as a dud (all ones, MESH_DUD_SHORT bytes short of its unit: mesh-bridge.h).  The class's two time rings, the bridge's to write (mesh.h struct mesh_stream
   `posted_at`, `landed_at`): unit u's posting or landing time (mach_absolute_time ticks) is the second word of entry u
   mod MESH_STREAM_TIMES where its first word reads 1 + u both before and after the second is read (the first two loads
   acquiring), else u has none */
struct mesh_stream_view { unsigned char *in,*out; _Atomic uint64_t *ready,*landed,*posted_at,*landed_at; uint32_t unit,units; };
int mesh_stream_view(struct mesh_ctx *,uint32_t stream,int large,struct mesh_stream_view *);
int mesh_stream_declare(struct mesh_ctx *,uint32_t stream,int large,uint64_t declared,uint64_t deadline);
#endif
