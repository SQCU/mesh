#ifndef MESH_COLLECTIVE_H
#define MESH_COLLECTIVE_H
#include "mesh-call.h"

/* The plan call (design/page-queue-change-request.md, "What the collective decides" and "The functions": the
   collective).  A program's transfers, every collective call and pipeline hand-off of it, declared at once and planned
   together over the map's Thunderbolt-5-connected nodes, for any node count, ahead of time; its output is this node's
   page lists (mesh.h struct mesh_hop), which the bridge alone reads.

   One declared transfer: an operand range from rank `source` to rank `destination`, ranks the map's indices
   (mesh_nodes_read: a node's place in ascending node-id order).  Every rank passes the same declared transfers in the
   same order; they differ only in `section`.
   - The transfers of one `identity` from one `source` are one range, sent to each of their destinations (a multicast
     where there are several); their `section.bytes` agree, and the range's order is the least of theirs.
   - `order`: lower first (a transfer that blocks a pipeline stage).  Within a node pair the list is in that order, then
     the declared order, so the bridge's rule (mesh.h "Both ends") assigns a blocking transfer's pages first.
   - `section`: this node's pages of the range where it is the source or a destination, `section.first` MESH_ABSENT
     elsewhere; `section.bytes` the range's bytes at every rank.  The range is P = ceil(bytes / (block x pgsz)) pages
     (a hop moves a whole page) in each of the plan's `depth` slots: slot s's pages the rows first + s x pages / block
     + k, k < P (mesh_section_slice's form with `invocations` = depth, or mesh_section_create's with `count` = depth).
     Each row starts its block (a range begins at a page) and lies in the registered window.
   `executions`: the plan's executions (main's invocations); `depth`: the slots its pages ring through, 0 or past
   `executions` taken as `executions` (no ring).

   mesh_plan refuses a destination its source reaches by no path (mesh-graph.h mesh_graph_reach over the map's node
   pairs), routes every declared transfer at once (mesh_graph_flow, the concurrent-flow solve), cuts
   each range's pages over its trees by their shares (mesh_graph_fill), and writes this node's own hops in mesh.h's
   order: its sends, its receives, and as a relay its receives into relay pages it allocates in the registered window
   and the sends their landed words release.  It allocates the plan's words, a 256-byte cell each word and execution
   (outside the window, zeroed; the context's send_off and send_bytes name them, for mesh-metal.m's publication
   buffer) and records each hop page's words in its row's struct mesh_publication: a landed word as `device_input`
   with `device_stride` = hop_stride, a source page's release word as `targets[0]` = {word, 1, 1} with `sends` = 1 and
   `argument` = its position.  Each entry carries the map serial and its `forwards` (a receive's: the sends here that
   forward its page; a send's: 1 where it forwards a landed page); hdr `plan` is the plan's generation (the last
   plan's + 1, never 0).
   Returns 0; EINVAL for a malformed transfer or a page outside the registered window; EHOSTUNREACH for a source or
   destination not in the map, or with no Thunderbolt-5 path within it; ENOSPC past hdr `hop_capacity` (or a page's
   publication targets); ESTALE where the map moved while planning (plan again on the map as it is); ENOMEM; or the
   map reader's error. */
struct mesh_declared { uint32_t source,destination,identity,order; struct mesh_section section; };
int mesh_plan(struct mesh_ctx *,const struct mesh_declared *,uint32_t count,uint32_t executions,uint32_t depth);

/* A host (CPU) program's side of the plan's words (mesh_plan), as mesh-metal.m is a GPU program's: the words each hop
   page's row records (struct mesh_publication), execution t's at the word's offset + t x hdr `hop_stride`.
   mesh_host_inputs prepares nothing now: mesh_plan allocated the words and recorded them in the rows.
   mesh_host_publish releases execution t of a declared range at its source: each release word its first row records
   is stored its `argument`, with release, after the caller's writes to the range.
   mesh_host_arrived acquires execution t of a declared range at a destination: 0 until every page of it has landed,
   then 1, or UINT64_MAX once a page's word was cancelled. */
int mesh_host_inputs(struct mesh_ctx *);
void mesh_host_publish(struct mesh_ctx *,struct mesh_section,uint32_t invocation);
uint64_t mesh_host_arrived(struct mesh_ctx *,struct mesh_section,uint32_t invocation);
#endif
