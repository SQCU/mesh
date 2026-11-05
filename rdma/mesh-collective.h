#ifndef MESH_COLLECTIVE_H
#define MESH_COLLECTIVE_H
#include "mesh-call.h"

/* The plan call (design/page-queue-change-request.md, "What the collective decides" and "The functions": the
   collective; design/disjoint-pages-change-request.md, Appendix C, C3.2-C3.7; design/disjoint-pages-revision-
   coordination.md, "The plan call", "The allocator" and "Writing a plan"; design/plan-to-bridge-change-request.md, R3
   to R5 and R8; design/aot-compilation-change-request.md, A3, A6 and A8, and its appendix; design/compilation-
   coordination.md, "The plan call", T1 to T7 and Q5; "The appendix's revisions, 2026-11-04", U1, S1, S2 and N1 to
   N5; "The documents' revisions of edd4b583", GO1 to GO5 and DL1).  A program's transfers,
   every collective call and pipeline hand-off of it, and every value it keeps at this node, declared at once and
   planned together over the map's Thunderbolt-5-connected nodes, for any node count, ahead of time.  Its output is one
   plan, one compiled function's, named (group, function), in its own claim: this node's page lists
   (mesh.h struct mesh_hop), which the bridge alone reads, in the plan's state section (mesh.h struct mesh_state) with
   the plan's words, and the rows and pages of every value, placed once by the one allocator.

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
   - In: `lag` (design/disjoint-pages-change-request.md, D.3.3; design/disjoint-pages-rulings-coordination.md, R4; M.
     Lam, Software pipelining, PLDI 1988: modulo variable expansion, a value live across k executions gets k copies).
     `lag` 0: as before (one copy; a routed value's u).  `lag` k >= 1: the value's instance of execution t is written by
     a producer the program orders by an event, not by its own order (a host producer, D.3.3: "The program declares the
     host's lag, and the allocator gives them copies by it, as modulo variable expansion does for any value").  The
     value must be MESH_EXECUTION, wire 0, state 0, and named by no declared transfer (EINVAL otherwise); its `begin`
     and `end` are not read.  The plan gives it k copies: `section.count` k, `stride` P, execution t's instance copy
     t mod k at mesh_section_address(section, t mod k).  Each copy is an item of its own that interferes with every
     other item (its write is ordered by no edge of the template, so it shares no page with anything).  The program's
     side of the order, which the plan neither knows nor checks: copy t mod k is written for execution t only after
     execution t - k has ended, and execution t reads it only after that write.
   - Out: `section`, every field: `first` its first row, `pages` P x block (P its blocks), `bytes` as given, `count` its
     copies (u for a routed value, k for a value of `lag` k >= 1, 1 for every other), `stride` P.  Copy c's rows are
     first + c x P + i, i < P, and its first byte mesh_section_address(section, c).  A program binds execution t of a
     routed value to copy t mod count, and execution t of a value of `lag` k to copy t mod k.  Its addresses are known
     after the plan, never before.

   Declared transfers (struct mesh_declared): an operand range from rank `source` to rank `destination`, ranks the
   map's indices (mesh_nodes_read: a node's place in ascending node-id order).  Every rank passes the same declared
   transfers, weights and dependences in the same order (every rank's events, so every rank builds the one template
   and the one task graph); they differ only in `value`.
   - The transfers of one `identity` from one `source` are one range, sent to each of their destinations (a multicast
     where there are several); their `bytes` agree.  A range is the operand's whole value, one declared transfer per
     destination: no page cut.  The ranges (demands) are numbered by their first declared transfer.
   - `kind` (design/plan-to-bridge-change-request.md, R4): the kind of the transfer's consumer at its destination,
     MESH_JOIN (function A, a blocking operator over its whole operand: one task, with an edge from every page of the
     range) or MESH_PAGES (function B, a homomorphism over pages taken in any order, J. Gibbons, J. Functional
     Programming, 1996, as R4 cites it: one task a page, each with an edge from its own page); any other EINVAL.  The
     kind reaches the task graph alone: the bridge's lists carry none (R4, R5).
     2026-11-04, owner C (design/compilation-coordination.md, "The plan call", T6): `order`, the hand priority (lower
     first, the range's the least of its transfers'), is gone with plan_demand_order, `kind` in its place.  Why: R5's
     static schedule orders every node pair's sends; a hand priority is a second order.  What must stay: every rank
     the same declared transfers in the same order, so every rank numbers the demands alike.
   - `value`: the index in `values` of this node's value of the range where this node is its source or a destination,
     else MESH_ABSENT.  That value is MESH_EXECUTION and wire, its bytes the range's, and one range's alone.
   - `bytes`: the range's bytes, at every rank.
   Weights (struct mesh_weight; design/aot-compilation-change-request.md, A3: the weighting operand, the program's
   weighting function's result, which the plan takes and computes none of): weights[i] is declared[i]'s, `count` of
   them, the task graph's w and c (O. Sinnen, Task Scheduling for Parallel Systems, Wiley, 2007).  The node pairs'
   capacities are the map's (mesh_nodes_read).
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
   and the rest outside it; a value of `lag` k >= 1 is k items, its copies, outside the window.  Two items share pages
   only where happens-before orders the last read of one before the next write of the other at every distance at which
   their executions meet; a copy of a value of `lag` k >= 1 shares with none, since no edge of the template orders its
   write.  The least lags decide those tests at every larger distance (design/compilation-coordination.md, U1): every
   walk of the template into a write passes its range's production, the plan's own; and, the premise the least lags
   rest on, a property of the program's dependences and not of the plan (struct mesh_dependence below), each node's
   events of execution t precede those of t + 1 by the program's own edges, so a production reaches its next instance
   at lag 1 and a walk at lag L through it continues at every lag past L.  mesh_plan does not test the premise
   (design/aot-compilation-change-request.md, appendix item 1: ENOMEM refuses only a value with no such L).  The plan
   claims exactly the rows and pages it places (mesh-dataflow.h mesh_claim_take), and closes no other claim: its own
   is closed only when the program retires the function (mesh-dataflow.h mesh_plan_retire) or itself (mesh_detach,
   the bridge's retirement of a gone pid), or by the call's own failure after its writes (ESTALE, or the second map
   read's error), and the bridge releases it once it holds it no more (C3.4); nothing is claimed at attach, and nothing
   waits for a release.
   2026-11-04, owner C2 (design/compilation-coordination.md, N3 and N5; design/aot-compilation-change-request.md,
   appendix item 6): reversing the close of this client's earlier claims once the new plan stood.  Why: compiled
   functions coexist, each its own plan with its own claim, released only when the program retires that function or
   itself (R. Frostig, M. J. Johnson and C. Leary, SysML 2018; NVIDIA, CUDA Programming Guide §4.2.2; Pathways §3).
   What must stay: the plan claims exactly the pages it places, and nothing waits for a release.

   The task graph, its schedule and the order of the lists (design/plan-to-bridge-change-request.md, R3 to R5;
   design/compilation-coordination.md, T1 to T5, and "The documents' revisions of edd4b583", GO1 to GO5).  One
   execution's task graph G = (V, E, w, c) (Sinnen, 2007): its tasks the template's events, each on the node its event
   is at (the placement operand: the declared transfers' sources and destinations, which the program's placement
   function chose).  Each range's production at its source weighs `produced`; each declared transfer's consumption at
   its destination, one task for MESH_JOIN or one a page for MESH_PAGES, weighs `consumed` (each page's task); each
   range's source read, where a dependence names it, weighs `read`.  Its edges: page k of declared[i]'s range from the
   production to the consumption (to page k's task for MESH_PAGES), of cost `page`, over the tree path the fill gave
   page k; the production to the source read; each dependence of lag 0 (task to task at one node; an end that is a
   MESH_PAGES consumption is each of its page tasks), a local edge costing 0.  The lags stay the allocator's: the
   schedule is one execution's.  Its topology graph's links are the node graph's edges, each a pair of directed links
   of the pair's capacity over the greatest (O. Sinnen and L. A. Sousa, Communication contention in task scheduling,
   IEEE TPDS 16(6) 2005, Definition 4).  The schedule is list scheduling with each task on its node, in decreasing
   bottom level, each edge scheduled on its route's links under the contention model (Sinnen and Sousa, eq. (7),
   Algorithms 1 and 2; mesh-graph.h mesh_graph_contention_schedule), a page crossing a tree edge once (T4): R5's method
   and A6's one schedule per function, computed as before; its starts order no list (R5: "Scheduled start times do not
   order the lists"), and what reads them is the operator's to say (GO5).
   The order of the lists is R5's reference algorithm, MSCCLang's (M. Cowan et al., MSCCLang: Microsoft Collective
   Communication Language, ASPLOS 2023) §5.2, step 3: "Sort instructions into a global topological order with respect
   to their dependencies with a heap using the priority to order instructions."  The instructions are the task graph's
   tasks, then the plan's communications (a page over a tree edge; its send at the edge's upper node and its receive at
   its lower node are one instruction at one place, so both ends' lists agree, where MSCCLang §4.2 makes them two,
   joined by a communication edge: GO1).  The dependencies are each local edge of the task graph, and each remote edge
   as a chain from its tail task through its route's communications to its head task (§4.2's communication and
   processing edges: a page sent before it lands, landed before it is forwarded or consumed; GO1).  The priority is
   each instruction's bottom level (R5: "The priority is the bottom level"), Sinnen and Sousa's eq. (7) over the
   instructions, a task weighing its weight, a communication its link time c(e) / b(L), the greatest over the edges
   whose routes carry it, and every dependency 0 (GO2; mesh-graph.h mesh_graph_bottom_levels); the greatest is taken
   first, a tie to the lower instruction (GO3; mesh-graph.h mesh_graph_heap_order).  Each of this node's entries takes
   its communication's place, and every rank computes the same order from the same inputs (F. B. Schneider,
   Implementing fault-tolerant services using the state machine approach, ACM Comput. Surv., 1990).
   2026-11-04, owner C3 (GO1 to GO5; R5, "The order of each cable's list, by the reference algorithm"): reversing T6,
   each node pair's list in the order of its link's scheduled starts of its pages, ties by walk.  Why: R5 orders each
   cable's list as a subsequence of one global topological order built by the reference algorithm (the operator:
   "what algorithm is cited here, if any? no citations? use the reference algorithm instead"), and scheduled starts
   order no list; this overturns T6.  What must stay: the schedule's call (R5's method, A6); both ends' lists of a
   node pair the same hops in the same order; every copy's the same order.
   The sets (design/aot-compilation-change-request.md, appendix item 5; design/compilation-coordination.md, S1, S2 and
   GO4): a range every one of whose declared transfers is MESH_PAGES (function B at every destination, R4) is a set on
   each of its trees: every entry of a page on tree j (j the tree's index in the solve's trees grouped by demand, the
   same at every rank) has `set` j + 1; every other entry, a range with any MESH_JOIN destination among them, has `set`
   0 and is at its own place.  A set's entries on one node pair and direction take consecutive places at the place of
   the first of them in the order (R5: "They take consecutive places at the place of the set's first page in that
   order"), each among them by its own place: each copy's entries are sorted by their group's least place (an entry's
   group the entries of its nonzero `set`, `node` and `direction`, else the entry alone), then by their own place.  So
   each node pair's list at each end is a subsequence of the one global order (a set's at its first page's place), a
   set's entries are contiguous on their node pair and direction within each copy, both ends' lists are the same hops
   in the same order, and every copy's is the same.  A set's pages over a tree edge depend only on their own pages over
   the edge above, or on the production, and on nothing of each other, so a group's least place follows that of every
   group it depends on: the groups in order of their least places are a topological order of the instructions with
   each group whole (R5: "A set's pages ... share their predecessors"), whose dependencies are the bridge's
   store-and-forward waits; with every lag pointing forward, the cables' sequences cannot wait on each other in a cycle
   (MSCCLang §5.2: "all implicit dependencies introduced by thread block sequential execution cannot produce cycles";
   GO4).  The bridge spreads a node pair's sequence as its own (R5), and sees a set or a sequence and no kind (R4).
   2026-11-04, owner C3 (GO4): reversing S3, each copy's entries in the order of their group's least scheduled start,
   then their group's least walk, then their own walk.  Why: R5's one global order, a set's pages at the place of the
   set's first page in it; scheduled starts order no list.  What must stay: both ends' lists the same hops in the same
   order; a set's entries contiguous on their node pair and direction within each copy (the bridge refuses otherwise:
   design/compilation-coordination.md, S7).
   2026-11-04, owner C3 (GO4; review W3 #6): the claim owner C2 withdrew here, that the groups' order (or T6's own)
   keeps each per-pair list a topological order of the unrolled executions, rested on the schedule's starts, which are
   cut-through (Sinnen and Sousa's eq. (14), nonaligned) and not shown to order the bridge's store-and-forward waits.
   The lists' order is now the instruction DAG's, whose dependencies are those waits (GO1), and GO4 shows the claim for
   it (above), answering the open item "No cycle on a cable".

   The plan's name, and the plan in its claim (design/compilation-coordination.md, "The plan call", Q5 and N1 to N5;
   the draft e6bd7db2:design/plan-bridge-coordination.md, §6, "Each plan in its claim", as the operator's ruling 1
   allows it).  `group`: the launcher's (MESH_GROUP), the same at every rank of a run; 0 is EINVAL.  `function` (N1):
   the program's own ordinal of the compiled function, the same at every rank (every rank runs the same program and
   compiles its functions in the same order), kept by every replan of that function; held in struct mesh_state
   `epoch`.  The plan is (group, function): EEXIST where a plan standing at this node on the same map already has that
   name, until the program retires it (mesh-dataflow.h mesh_plan_retire); a plan of another map serial is cancelled
   (D.3.6) and ignored.  Any number of plans of one program stand at once, each its own claim (N1, N3).  The plan's
   state section is in its claim, and the claim's head row's publication `state` names it once the plan stands;
   `*plan` is that offset, the plan's handle (N2: the start, the GPU's transport and the retirement take it), and hdr's
   one-plan fields (`hops`, `plan`, `state`, `hop_off`, `hop_capacity`) are neither read nor written.
   2026-11-04, owner C2 (design/compilation-coordination.md, N1, N2 and N5; design/aot-compilation-change-request.md,
   appendix item 6): reversing C-epoch's name (group, epoch), epoch one past the context's count of its plans that
   stood or its cancelled standing plan's, and context->state as the context's one standing plan.  Why: compiled
   functions coexist, each named by the program's ordinal of it, and the context holds no plan (struct mesh_ctx loses
   `epoch` and `state`).  What must stay: every rank names one plan alike; EEXIST where a plan of that name stands on
   the same map.

   The state section (mesh.h struct mesh_state, at its claim's head's publication `state`), an invariant item outside
   the window, apart from every data page: the struct, then the phase list (each received range's 2-bit Gray-coded
   phase a page, a group a copy), the counts (each received range's arrivals, a word a copy), the supplied words (one a
   source range), the GPU's status, the sequence words (one a copy) and this node's entries, each part 64-aligned
   (mesh-collective.c plan_layout gives the offsets).  Its words are set to their starting values once, where it is
   placed (the contract's D3), and none is reset while the plan stands (C3.7).  The struct's words beside them:
   `group` and `epoch` the plan's name (group, function), `epoch` holding the function (N1), `serial` the map serial
   it was made on, `hops` its entries' region offset and `entries` u x n, `sequences` the sequence words' region
   offset, `held` 0 and every `refused` 0 (the ports' alone), `nodes` the plan's nodes (a set of map indices: every
   source and destination of a declared transfer and every node a tree of it reaches; this node's own where there is
   none) and `sender` the least of them (Q2), each the same at every rank; and `submitted`, `released`, every `taken`
   and every sequence word 0, the bridge's alone (Q2; design/compilation-coordination.md, "The documents' revisions of
   edd4b583", DL1: `released` and the sequence words the bridge parent's, L. Lamport's ms and B[0 .. u - 1] of his
   "Message Buffer", Concurrent reading and writing, CACM 20(11), 1977; `taken[p]` port p's, his mr).  Each claimed
   row's publication (mesh.h struct mesh_publication) is written: a received range's first row names its phase groups
   and count words, a source range's first row its supplied word, every other field 0.  No data page is written by
   the plan.
   2026-11-04, owner C3 (DL1; design/aot-compilation-change-request.md, appendix item 2): the sequence words placed
   between the status and the entries, and `taken` where `served` (the executions port p had served whole, Q4's hold)
   was.  Why: the parent publishes each released execution's sequence number in its plan's state, one word per copy,
   each word one writer, and each port merges them; Q4's hold is gone.  What must stay: every word before the sequence
   words where it was; every word started once and none reset while the plan stands.

   This node's entries: u x n (mesh.h struct mesh_hop) at the state section's `hops`, copy c's n at [c n, c n + n),
   every copy's the same hops in the same order, each copy's naming its own pages, phase fields and count words.  Each
   entry carries the map serial, its page's bytes, min(Q, B - k Q) for page k of a range of B bytes, Q = block x pgsz
   (T7), and its `set` (S1, S2), every copy's the same.
   2026-11-04, owner C (design/compilation-coordination.md, "The region" and Q5): reversing the entries in hdr's hop
   area (`hops` of them at `hop_off`, room `hop_capacity`, ENOSPC past it) and hdr `plan` as the plan's generation.
   Why: any number of plans stand at once, each its own entries in its own claim (R8, A7).  What must stay: copy c's n
   at [c n, c n + n), every copy's the same hops in the same order.

   Returns 0; EINVAL (a malformed transfer, weight, value or dependence: a kind other than MESH_JOIN and MESH_PAGES, the
   transfers of one range giving different `produced` or `read`, a weight not finite or not above 0, `group` 0, `plan`
   NULL; a lag-0 cycle); EEXIST (a plan standing at this node on the same map is already (group, function));
   EHOSTUNREACH (a source or destination not in the map, or with no Thunderbolt-5 path within it); ENOMEM (no free run
   for the claim; a value with no unroll bound, named on stderr by its index and its range's identity; the call's own
   memory); ESTALE (the map moved while planning: plan again on the map as it is); or the map reader's error.  Every
   failure but ESTALE and the second map read's is found before anything is written; with those, this plan's claim
   alone is closed.  Every failure leaves `*plan` untouched and every other plan of the program standing as it stood.
   There is no ETIMEDOUT: nothing waits.
   2026-11-04, owner C2 (design/compilation-coordination.md, U1; design/aot-compilation-change-request.md, appendix
   item 1): ENOMEM for a value whose write does not reach its next instance at lag 1 (design/disjoint-pages-revision-
   coordination.md, D5) is gone.  Why: C3.3 fails only "A value with no such L" (M. Lam, PLDI 1988; B. R. Rau, MICRO
   1994), and D5 is in neither; the least lags are exact at every larger distance by U1's reason (above), whose
   premise is the program's dependences' (struct mesh_dependence) and is not tested.  What must stay: ENOMEM, named,
   for a read of a value that reaches no later write of it.
   2026-11-04, owner C2 (N3, N5): reversing the close of every earlier claim of the client, context->state 0, on a
   failure after the writes.  Why: compiled functions coexist; a failed plan closes its own claim alone.  What must
   stay: every failure but ESTALE and the second map read's found before anything is written.
   2026-11-04, sub-assignment 3: reversing 2026-11-03's call (sub-assignments 3-5), which took `executions` and unrolled
   every execution into its own instances, took each range's section from mesh_section_create, zeroed every declared
   page and every word it used, and waited for the previous plan's erase (mesh-call.h mesh_plan_erased, ETIMEDOUT).
   Why: the unroll factor is the longest lifetime, not the trip count (C1, C3.3); the allocator places every value
   (C3.2); nothing is zeroed or reset, and a new plan's state is in new pages (C3.4, C3.7).  What must stay: every rank
   the same declared transfers and dependences in the same order; every failure but ESTALE and the second map read's
   found before anything is written. */
enum { MESH_INVARIANT, MESH_EXECUTION };
/* a value at this node.  In: section.bytes (> 0), kind, wire, state (1 only on a local MESH_INVARIANT value, wire 0),
   for a local MESH_EXECUTION value of lag 0 its first and last positions on this node's own program order in one
   execution, begin <= end, and lag: 0 for a value the program's own order writes (one copy; a routed value's u), k >= 1
   for one a producer the program orders by an event writes (a host input: MESH_EXECUTION, wire 0, state 0, named by no
   declared transfer, begin and end not read; k copies, execution t's at copy t mod k, written for t only after
   execution t - k has ended and read by t only after that write: the program's order, which the plan does not check;
   design/disjoint-pages-change-request.md, D.3.3; design/disjoint-pages-rulings-coordination.md, R4).  Out: section,
   every field */
struct mesh_value { struct mesh_section section; uint32_t kind,wire,state,begin,end,lag; };
/* plan-to-bridge R4: the kind of a declared transfer's consumer at its destination.  MESH_JOIN: function A, one task
   with an edge from every page of the range.  MESH_PAGES: function B, one task per page, each with an edge from its
   own page */
enum { MESH_JOIN, MESH_PAGES };
/* `order` (the hand priority, lower first) is gone: R5's schedule orders every cable's sends; `kind` in its place */
struct mesh_declared { uint32_t source,destination,identity,kind,value,padding; size_t bytes; };
/* aot-compilation A3: the weighting operand, one entry a declared transfer, the program's weighting function's result.
   produced: w of the range's production at its source; consumed: w of the consumption at declared[i]'s destination
   (for MESH_PAGES, of each page's task); read: w of the source's own read of the range (where a dependence names it);
   page: c of each page's edge to declared[i]'s destination.  The transfers of one range give the same produced and
   read (EINVAL otherwise).  Every weight finite and > 0 (EINVAL otherwise; design/compilation-coordination.md,
   C-weights: Sinnen and Sousa's Definition 1, "The positive weight w(n)", and every link occupation > 0, 2026-11-04,
   where it was >= 0).  The cables' capacities are the map's. */
struct mesh_weight { double produced,consumed,read,page; };
/* a program's data edge at one node: event `from` happens before event `to`, `to` in execution t + lag where `from`
   is in t.  An event is declared[i]'s production at its source (its kinds bit set: MESH_FROM_PRODUCED for `from`,
   MESH_TO_PRODUCED for `to`) or its consumption at declared[i]'s destination (bit clear); both events at one node.
   (added 2026-11-04, the coordinator's ruling on 13's and 14's finding; design/disjoint-pages-revision-coordination.md,
   "The plan call") With its source-read bit set (MESH_FROM_SOURCE_READ for `from`, MESH_TO_SOURCE_READ for `to`), the
   event is declared[i]'s consumption at its SOURCE: the source's own read of the range it sends (a holder's combine
   reading its own partial, a broadcast source or a single owner reading what it publishes); the source's value's reads
   then include it.  An end with both its produced and its source-read bit is EINVAL.
   The premise the plan's least lags rest on (design/compilation-coordination.md, U1; added 2026-11-04, reviews W1 #3
   and W3 #4, the coordinator's ruling), a property of the program's dependences and not of the plan: every event of a
   node in execution t precedes each of its events in t + 1 by the program's own edges, as both programs give them
   (tools/e2b_replay.m e2b_ir's rank chains, "r's last to its first at lag 1"; rdma/mesh_compile.py _dependences,
   "every event no lag-0 edge leaves to every event none enters (lag 1)"), so a production reaches its next instance
   at lag 1, and the least lags are C3.3's happens-before at every larger distance.  mesh_plan does not test it
   (design/aot-compilation-change-request.md, appendix item 1: ENOMEM refuses only a value with no such L) */
#define MESH_FROM_PRODUCED 1u
#define MESH_TO_PRODUCED 2u
#define MESH_FROM_SOURCE_READ 4u
#define MESH_TO_SOURCE_READ 8u
struct mesh_dependence { uint32_t from,to,lag,kinds; };
/* aot-compilation appendix item 6 (design/compilation-coordination.md, "The appendix's revisions", N1 to N5): the plan
   of one compiled function, named (group, function): `group` the launcher's (MESH_GROUP), 0 EINVAL; `function` the
   program's own ordinal of the compiled function, the same at every rank, kept by every replan of that function; held
   in struct mesh_state `epoch`.  EEXIST where a plan of that name stands here on the same map.  *plan out: the region
   offset of the plan's state section once it stands (untouched on any failure).  No earlier plan is closed */
int mesh_plan(struct mesh_ctx *,uint64_t group,uint64_t function,const struct mesh_declared *,
              const struct mesh_weight *,uint32_t count,struct mesh_value *,uint32_t values,
              const struct mesh_dependence *,uint32_t dependences,uint64_t *plan);

/* A host (CPU) program's side of the plan's words (mesh_plan), as mesh-metal.m is a GPU program's, each by its range's
   first row's publication, as before: mesh_host_publish its supplied word (`targets[0]`), mesh_host_arrived its words
   and its plan's state section (`state`).
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
