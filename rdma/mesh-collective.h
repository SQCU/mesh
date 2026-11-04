#ifndef MESH_COLLECTIVE_H
#define MESH_COLLECTIVE_H
#include "mesh-call.h"

/* The plan call (design/page-queue-change-request.md, "What the collective decides" and "The functions": the
   collective; design/disjoint-pages-change-request.md, "Claude's proposal: the canonical path", Appendix A, A3, and
   Appendix B).  A program's transfers, every collective call and pipeline hand-off of it, declared at once and planned
   together over the map's Thunderbolt-5-connected nodes, for any node count, ahead of time; its output is this node's
   page lists (mesh.h struct mesh_hop), which the bridge alone reads, and the pages of every value they move.

   One declared transfer: an operand range from rank `source` to rank `destination`, ranks the map's indices
   (mesh_nodes_read: a node's place in ascending node-id order).  Every rank passes the same declared transfers and the
   same dependences in the same order; they differ only in `section`.
   - The transfers of one `identity` from one `source` are one range, sent to each of their destinations (a multicast
     where there are several); their `section.bytes` agree, and the range's order is the least of theirs.  A range is
     the operand's whole section, one declared transfer per destination: no page cut.
   - `order`: lower first (a transfer that blocks a pipeline stage).  Within a node pair the list is in that order, then
     the declared order, so the bridge's rule (mesh.h "Both ends") assigns a blocking transfer's pages first.
   - `section`: this node's pages of the range where it is the source or a destination, `section.first` MESH_ABSENT
     elsewhere; `section.bytes` the range's bytes at every rank.  At the source and each destination it is a
     mesh_section_create(bytes, executions, 1) section: `count` == executions, P = ceil(bytes / (block x pgsz)) pages an
     instance (`pages` == P x block), instance t at rows first + t x stride + k, k < P (`stride` >= P).  Its rows lie
     among the claim's section rows (mesh-dataflow.h struct mesh_ctx `rows`, from its first + 2) and are unbound, or
     bound by an earlier plan inside the program's `window`.  A section is one range's at this node, received or sent,
     never two ranges'.  The plan assigns every instance's pages and binds its rows (mesh_backing_bind), so a program
     reads its sections' addresses (mesh_section_address) after its plan, never before; and it zeroes every page it
     binds before the bridge can load the plan, so the program zeroes none of them after it (a peer's SEND held in
     credit lands as soon as this node's bridge loads the plan).
   `executions`: the plan's executions t = 0 .. executions - 1 (main's invocations), unrolled, each execution's values
   their own instances: no slots, no ring, no depth (design/disjoint-pages-change-request.md, Appendix A, A3,
   "Repeated executions are unrolled, not ringed").
   `dependences`: the program's own data edges between ranges (struct mesh_dependence; NULL where there are none).
   With the transport's own edges (a range produced before it is sent, a page landed before it is forwarded, sent
   before it lands, landed before it is consumed) they are the plan's dependence graph, and a page is reused only where
   that graph orders every read of what it held before the next write to it (Appendix A, A3, "Interference, by
   happens-before"; Appendix B: as CUDA graph memory nodes); with none, no page is reused.

   mesh_plan refuses a destination its source reaches by no path (mesh-graph.h mesh_graph_reach over the map's node
   pairs), routes every declared transfer at once (mesh_graph_flow, the concurrent-flow solve), cuts each range's pages
   over its trees by their shares (mesh_graph_fill), and assigns, once and before the first execution, the pages of
   every value instance at this node in every execution: a range's at its source and destinations, a relay's own pages
   of it elsewhere (rowless), each inside the program's `window` (the one allocator, Appendix A, A3: mesh-graph.h
   mesh_graph_happens_before and mesh_graph_left_edge).  It writes this node's hops in mesh.h's order ("Writing a
   plan"), executions x n entries, execution t's n naming its own pages and words, every execution's the same hops in
   the same order: its sends, its receives, and as a relay its receives into its own pages and the sends their
   landings release.
   The plan's words, fixed areas of the region (hdr `word_off`, `landed_off`) the next plan reuses, which mesh_plan
   zeroes where it uses them (design/disjoint-pages-coordination.md, "The plan's words"):
   - the supplied word of release r (this node's r-th range it is the source of, in the plan's demand order) in
     execution t, mesh_supplied(m, r, t): its producer stores MESH_SUPPLIED (release) after its last write to the range
     in execution t, and that one store releases the range's sends; nothing cancels it, and 0 releases nothing;
   - execution t's landed record, mesh_landed_record(m, t), hdr `landed_stride` bytes: word 0 its cancel word (0, or
     UINT64_MAX once cancelled, which three writers store: the bridge, for a receive not landed as it ends the plan,
     mesh-flow.c link_close and plan_cancel; the program's cancellation, mesh.h mesh_cancel over every execution's
     cancel word; and the GPU lane whose wait on a step reaches its bound, which stores it into the cancel word of that
     step and of every later one, records t .. executions - 1, metal_recording.m mesh_read_N), then bit b
     for the b-th receive hop of the execution in list order (word 1 + b / 64,
     bit b % 64), which its landing ORs in (release) and nothing clears while the plan stands.  A destination's pages of
     one range are one run of bits [first, first + P).  The landed list lies in its own section of the region, beside
     the page lists and never beside a page: a reader reads a step's mask, not a word a page.
   Instance 0's rows of a declared section record them (struct mesh_publication): a destination's page its bit
   (`landed`), `device_input` = hdr `landed_off` (execution 0's record) and `device_stride` = `landed_stride`, a row
   with no receive `device_input` 0; a source range's first row `targets[0]` = {region offset of mesh_supplied(m, r,
   0), 1, 1}, `sends` 1, `argument` = MESH_SUPPLIED, execution t's word at `stream` + t x MESH_WORD_STRIDE.  Each entry
   carries the map serial; hdr `plan` is the plan's generation (the last plan's + 1, never 0).  Every failure but
   ETIMEDOUT (the erase), ESTALE and the map reader's error on its second read is found before the previous plan is
   touched, and leaves it standing; with those the previous plan no longer stands (hdr `hops` 0) and no new one is
   written.
   Returns 0; EINVAL for a malformed transfer or dependence (an index past `count`, a consumed range's destination not
   the produced range's source, dependences of lag 0 ordering an execution's work in a cycle), or a held section not of
   the form above (a row outside the claim's section rows, or bound outside `window`, or rows two ranges' sections at
   this node share); EHOSTUNREACH for a source or destination not in the map, or with no Thunderbolt-5 path within it;
   ENOSPC past hdr `hop_capacity`, `word_capacity` or `landed_capacity`; ENOMEM where the allocator finds no room in
   `window`, or for the call's own memory; ESTALE where the map moved while planning (plan again on the map as it is);
   ETIMEDOUT where a port still held the previous plan past MESH_PLATFORM_NS (mesh-call.h mesh_plan_erased); or the map
   reader's error. */
struct mesh_declared { uint32_t source,destination,identity,order; struct mesh_section section; };
/* a program's data edge: the destination of declared[consumed] has read it in execution t before the source of
   declared[produced] first writes it in execution t + lag (that node is both); EINVAL where it is not */
struct mesh_dependence { uint32_t consumed,produced,lag; };
int mesh_plan(struct mesh_ctx *,const struct mesh_declared *,uint32_t count,uint32_t executions,
              const struct mesh_dependence *,uint32_t dependences);

/* A host (CPU) program's side of the plan's words (mesh_plan), as mesh-metal.m is a GPU program's.
   mesh_host_inputs prepares nothing: mesh_plan zeroes the words it uses and records them in the rows.
   mesh_host_publish releases execution t (`invocation`) of a declared range at its source: MESH_SUPPLIED stored, with
   release, at the range's supplied word of execution t (its first row's targets, `stream` + t x MESH_WORD_STRIDE),
   after the caller's writes to the range.
   mesh_host_arrived reads execution t of a declared range at a destination from execution t's landed record
   (acquire): UINT64_MAX where its cancel word is, 1 once the range's bit run [first, first + P) is whole (`first` its
   first row's `landed`), else 0; 0 where that row has no receive. */
int mesh_host_inputs(struct mesh_ctx *);
void mesh_host_publish(struct mesh_ctx *,struct mesh_section,uint32_t invocation);
uint64_t mesh_host_arrived(struct mesh_ctx *,struct mesh_section,uint32_t invocation);
#endif
