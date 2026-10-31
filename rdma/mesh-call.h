#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"
/* design/algorithm-sources.md#programtensor: a lease of `count` slots of `bytes`, its rows and arena pages claimed by
   compare-exchange (mesh-dataflow.c mesh_claim): a slot `pages` arena pages on `stride` rows from row `first`; 0, or the
   refusal as a value (ENOMEM, ENOSPC) */
struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride; };
/* Tensor storage may use any free bridge pages. A bridge descriptor table is
   read as one region interval and declares that stricter backing requirement.
   Both have the same contiguous local virtual view, prepared without a copy. */
enum mesh_backing { MESH_BACKING_PAGES, MESH_BACKING_LINEAR };
int mesh_section_create(struct mesh_ctx *,size_t bytes,uint32_t count,enum mesh_backing,struct mesh_section *);
void *mesh_section_address(struct mesh_ctx *,struct mesh_section,uint32_t index);
/* design/transport.md section 14: the nodes this node's bridge has a live stream toward, each with its stream's
   bandwidth (mesh-plan.h struct mesh_reach), at most `room`; their count */
uint32_t mesh_nodes_reached(struct hdr *,struct mesh_reach *,uint32_t room);
/* design/transport.md section 14: the streams free to take (live, owner word 0), each one's node and its `wire`
   (struct mesh_stream: its identity, alike at both its ends), at most `room`; their count.  The two ends of a
   communicator take the stream of one wire between their nodes, one free at both: its init exchanges these */
uint32_t mesh_streams_free(struct hdr *,uint32_t *node,uint64_t *wire,uint32_t room);
/* design/whole-function.md#fabric */
struct mesh_endpoint { uint64_t wire,bandwidth; uint32_t node,bundle,available,padding; };
uint32_t mesh_endpoints(struct hdr *,struct mesh_endpoint *,uint32_t room);
/* design/transport.md section 14: the live stream of identity `wire`, held by this client as its rows are: its owner
   word taken from 0 to the client's own by one compare-and-swap, `*stream` its index: 0, EINVAL, ENOLINK where no live
   stream has the identity (one whose last holding ended with units outstanding is not live), EBUSY where another
   client holds it or the bridge is still ending its last holding.  Given by its index (MESH_STREAM_GIVEN set in its owner
   word), after which its bridge posts nothing more of the holding: a lease the client returns after it is held only
   by units posted before (mesh-dataflow.c mesh_pinned) */
int mesh_stream_take(struct mesh_ctx *,uint64_t wire,uint32_t *stream);
void mesh_stream_give(struct mesh_ctx *,uint32_t stream);
/* design/transport.md section 14, "A process a port": whether stream `stream` is still this client's holding of identity
   `wire`: live, its owner word this client's, its identity the one taken.  A pairing lost and made again opens every
   stream of the link new (no owner): a holding of the lost pairing reads 0 here, and its holder starts no group on it */
int mesh_stream_held(struct mesh_ctx *,uint32_t stream,uint64_t wire);
/* design/transport.md section 14: a held stream's words as its holder uses them (mesh.h struct mesh_stream): its two
   lists (0 the send list, 1 the landing list) and their counts presented and taken, and its three contiguous unit counts; 0,
   EINVAL, EPERM where the client does not hold it, or ENOLINK */
struct mesh_stream_view {
  struct mesh_entry *list[2];
  _Atomic uint64_t *presented[2],*taken[2],*ready,*sent,*landed,*zwoggle;
};
int mesh_stream_view(struct mesh_ctx *,uint32_t stream,struct mesh_stream_view *);
/* design/transport.md section 14: `entry` presented on held stream `stream`'s send list (`land` 0) or landing list
   (`land` 1): its slot written, then the list's count one past it (a release store); 0, EINVAL for an invalid immutable span table, byte extent or publication word, EAGAIN where the list's slot is not yet taken
   (MESH_STREAM_ENTRIES presented and not taken: present it again), EPERM, or ENOLINK */
int mesh_stream_present(struct mesh_ctx *,uint32_t stream,int land,struct mesh_entry entry);
#endif
