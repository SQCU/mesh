#ifndef MESH_COLLECTIVE_H
#define MESH_COLLECTIVE_H
#include "mesh-call.h"

/* The plan call (design/page-queue-change-request.md, "What the collective decides" and "The functions": the
   collective; design/disjoint-pages-change-request.md, Appendix C, C3.2-C3.7; design/disjoint-pages-revision-
   coordination.md, "The plan call", "The allocator" and "Writing a plan").  A program's transfers, every collective
   call and pipeline hand-off of it, and every value it keeps at this node, declared at once and planned together over
   the map's Thunderbolt-5-connected nodes, for any node count, ahead of time.  Its output is this node's page lists
   (mesh.h struct mesh_hop), which the bridge alone reads, the plan's state section (mesh.h struct mesh_state), and the
   rows and pages of every value, placed once by the one allocator.

   Values (struct mesh_value), this node's own: every value the program keeps here, weights, caches, intermediates,
   exchanges and GPU counters alike, placed by the plan and by nothing else (C3.2).
   - In: `section.bytes` (> 0); `kind` MESH_INVARIANT (written before the first execution and read by every one: placed
     once for the plan's life, as Lam places a loop invariant outside the kernel) or MESH_EXECUTION (written anew in
     each execution); `wire` 1 where it lies in the registered window, 0 outside it; `state` 1 where it is state (a GPU
     counter the program keeps), which the plan writes to 0 once where it places it (the contract's D3), 0 for data,
     which the plan never writes; and, for a local MESH_EXECUTION value (one no declared transfer names), `begin` <=
     `end`, its first and last positions on this node's own program order in one execution.  `state` 1 is EINVAL but
     on a local MESH_INVARIANT value outside the window (wire 0, named by no declared transfer): the plan never writes
     a routed value's pages, and state shares pages with nothing (2026-11-04, review 15's finding 3, the coordinator's
     ruling).
   - Out: `section`, every field: `first` its first row, `pages` P x block (P its blocks), `bytes` as given, `count` its
     copies (u for a routed value, 1 for every other), `stride` P.  Copy c's rows are first + c x P + k, k < P, and its
     first byte mesh_section_address(section, c).  A program binds execution t of a routed value to copy t mod count.
     Its addresses are known after the plan, never before.

   Declared transfers (struct mesh_declared): an operand range from rank `source` to rank `destination`, ranks the
   map's indices (mesh_nodes_read: a node's place in ascending node-id order).  Every rank passes the same declared
   transfers and the same dependences in the same order (every rank's events, so every rank builds the one template);
   they differ only in `value`.
   - The transfers of one `identity` from one `source` are one range, sent to each of their destinations (a multicast
     where there are several); their `bytes` agree, and the range's order is the least of theirs.  A range is the
     operand's whole value, one declared transfer per destination: no page cut.
   - `order`: lower first (a transfer that blocks a pipeline stage).  Within a node pair the list is in that order, then
     the declared order, so the bridge's rule (mesh.h "Both ends") assigns a blocking transfer's pages first.
   - `value`: the index in `values` of this node's value of the range where this node is its source or a destination,
     else MESH_ABSENT.  That value is MESH_EXECUTION and wire, its bytes the range's, and one range's alone.
   - `bytes`: the range's bytes, at every rank.
   Dependences (struct mesh_dependence): the program's own data edges at one node, event `from` before event `to`, `to`
   in execution t + lag where `from` is in t.  An event is declared[i]'s production at its source, its consumption at
   declared[i]'s destination, or the source's own read of the range it sends.  With the transport's own edges (a range
   produced before it is sent and before its source reads it, a page landed before it is forwarded, sent before it
   lands, landed before it is consumed) they are the plan's dependence template.

   mesh_plan refuses a destination its source reaches by no path (mesh-graph.h mesh_graph_reach), routes every declared
   transfer at once (mesh_graph_flow), cuts each range's pages over its trees by their shares (mesh_graph_fill), and
   unrolls by modulo variable expansion (C3.3; M. Lam, Software pipelining, PLDI 1988; B. R. Rau, Iterative modulo
   scheduling, MICRO 1994): from the template's least lags (mesh_graph_lags), L(v) for each routed value v is the
   smallest lag at which every read of it in execution t happens before its write in execution t + L, and u is the
   largest L over every routed value at every node (1 where there is none), the same at every rank.  The plan holds u
   copies; execution t runs copy t mod u, and a plan runs any number of executions (the client's start says how many:
   mesh-call.h mesh_transfers_start).  The one allocator then places, once, every item at this node: each routed
   value's u copies (a relay's value among them, which no `values` entry names and no row holds), each local value and
   the state section, by two greedy-by-size colourings (mesh_graph_left_edge), the wire items in the registered window
   and the rest outside it.  Two items share pages only where happens-before orders the last read of one before the
   next write of the other at every distance at which their executions meet.  The plan claims exactly the rows and
   pages it places (mesh-dataflow.h mesh_claim_take), and closes this client's earlier claims, which the bridge
   releases once no port holds them (C3.4); nothing is claimed at attach, and nothing waits for a release.

   The state section (mesh.h struct mesh_state, at hdr `state`), an invariant item outside the window, apart from every
   data page: the struct, then the phase list (each received range's 2-bit Gray-coded phase a page, a group a copy),
   the counts (each received range's arrivals, a word a copy), the supplied words (one a source range) and the GPU's
   status (mesh-collective.c mesh_plan gives the offsets).  Its words are set to their starting values once, where it
   is placed (the contract's D3), and none is reset while the plan stands (C3.7).  Each claimed row's publication
   (mesh.h struct mesh_publication) is written: a received range's first row names its phase groups and count words, a
   source range's first row its supplied word, every other field 0.  No data page is written by the plan.

   This node's entries: u x n (mesh.h struct mesh_hop), copy c's n at [c n, c n + n), every copy's the same hops in
   the same order, each copy's naming its own pages, phase fields and count words.  Each entry carries the map serial;
   hdr `plan` is the plan's generation (the last plan's + 1, never 0).

   Returns 0; EINVAL (a malformed transfer, value or dependence; a lag-0 cycle); EHOSTUNREACH (a source or destination
   not in the map, or with no Thunderbolt-5 path within it); ENOSPC (u x n entries past hdr `hop_capacity`); ENOMEM (no
   free run for the claim; a value with no unroll bound, or whose write does not reach its next instance at lag 1 (the
   contract's D5), named on stderr by its index and its range's identity; the call's own memory); ESTALE (the map moved
   while planning: plan again on the map as it is); or the map reader's error.  Every failure but ESTALE and the second
   map read's is found before anything is written, and leaves the previous plan standing; with those the previous plan
   no longer stands (hdr `hops` 0) and no new one is written.  There is no ETIMEDOUT: nothing waits.
   2026-11-04, sub-assignment 3: reversing 2026-11-03's call (sub-assignments 3-5), which took `executions` and unrolled
   every execution into its own instances, took each range's section from mesh_section_create, zeroed every declared
   page and every word it used, and waited for the previous plan's erase (mesh-call.h mesh_plan_erased, ETIMEDOUT).
   Why: the unroll factor is the longest lifetime, not the trip count (C1, C3.3); the allocator places every value
   (C3.2); nothing is zeroed or reset, and a new plan's state is in new pages (C3.4, C3.7).  What must stay: every rank
   the same declared transfers and dependences in the same order; every failure but ESTALE and the second map read's
   found before anything is written. */
enum { MESH_INVARIANT, MESH_EXECUTION };
/* a value at this node.  In: section.bytes (> 0), kind, wire, state (1 only on a local MESH_INVARIANT value, wire 0),
   and for a local MESH_EXECUTION value its first and last positions on this node's own program order in one
   execution, begin <= end.  Out: section, every field */
struct mesh_value { struct mesh_section section; uint32_t kind,wire,state,begin,end,padding; };
/* one declared transfer, the same at every rank but `value`: the index in `values` of this node's value of the range
   where this node is its source or a destination, else MESH_ABSENT; `bytes` the range's bytes at every rank */
struct mesh_declared { uint32_t source,destination,identity,order,value,padding; size_t bytes; };
/* a program's data edge at one node: event `from` happens before event `to`, `to` in execution t + lag where `from`
   is in t.  An event is declared[i]'s production at its source (its kinds bit set: MESH_FROM_PRODUCED for `from`,
   MESH_TO_PRODUCED for `to`) or its consumption at declared[i]'s destination (bit clear); both events at one node.
   (added 2026-11-04, the coordinator's ruling on 13's and 14's finding; design/disjoint-pages-revision-coordination.md,
   "The plan call") With its source-read bit set (MESH_FROM_SOURCE_READ for `from`, MESH_TO_SOURCE_READ for `to`), the
   event is declared[i]'s consumption at its SOURCE: the source's own read of the range it sends (a holder's combine
   reading its own partial, a broadcast source or a single owner reading what it publishes); the source's value's reads
   then include it.  An end with both its produced and its source-read bit is EINVAL */
#define MESH_FROM_PRODUCED 1u
#define MESH_TO_PRODUCED 2u
#define MESH_FROM_SOURCE_READ 4u
#define MESH_TO_SOURCE_READ 8u
struct mesh_dependence { uint32_t from,to,lag,kinds; };
int mesh_plan(struct mesh_ctx *,const struct mesh_declared *,uint32_t count,struct mesh_value *,uint32_t values,
              const struct mesh_dependence *,uint32_t dependences);

/* A host (CPU) program's side of the plan's words (mesh_plan), as mesh-metal.m is a GPU program's.
   2026-11-04, sub-assignment 3: mesh_host_inputs is removed (it did nothing: mesh_plan places and starts every word),
   and `invocation` is `execution`, a uint64_t, since a plan runs any number of executions.  What must stay: a host
   producer releases a range by one store, and a host reader reads a range's state, never a word a page. */
/* a host producer's release of execution t of a source range: its supplied word stored t + 1 (release) */
void mesh_host_publish(struct mesh_ctx *,struct mesh_section,uint64_t execution);
/* execution t of a received range: 1 where every field of its copy t mod u reads mesh_phase(t / u + 1) (acquire);
   UINT64_MAX where a field reads neither that nor mesh_phase(t / u) (ahead), or where its count has reached
   (t / u + 1) x P with a field still behind (cancelled); else 0 (not yet) */
uint64_t mesh_host_arrived(struct mesh_ctx *,struct mesh_section,uint64_t execution);
#endif
