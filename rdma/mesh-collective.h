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
     which the plan never writes; and, for a MESH_EXECUTION value of `lag` 0, `begin` <= `end`, its first and last
     positions on this node's own program order in one execution.  `state` 1 is EINVAL but
     on a local MESH_INVARIANT value outside the window (wire 0, named by no declared transfer): the plan never writes
     a routed value's pages, and state shares pages with nothing (2026-11-04, review 15's finding 3, the coordinator's
     ruling).
   - In: `lag` (design/disjoint-pages-change-request.md, D.3.3; design/disjoint-pages-rulings-coordination.md, R4; M.
     Lam, Software pipelining, PLDI 1988: modulo variable expansion, a value live across k executions gets k copies).
     `lag` 0: u copies (design/problems-of-14b9a757-change-request.md, "Appendix: the stops of 36bd7cbd", 14).  `lag`
     k >= 1: the value's instance of execution t is written by a producer the program orders by an event, not by its
     own order (a host producer, D.3.3: "The program declares the host's lag, and the allocator gives them copies by it,
     as modulo variable expansion does for any value").  The
     value must be MESH_EXECUTION, wire 0, state 0, and named by no declared transfer (EINVAL otherwise); its `begin`
     is not read, nor its `end` but where a transfer names it (design/problems-of-14b9a757-change-request.md,
     "Appendix: the stops of 36bd7cbd", 14).  The plan gives it k copies: `section.count` k, `stride` P, execution
     t's instance copy t mod k at mesh_section_address(section, t mod k).  Each copy is an item of its own that
     interferes with every other item (its write is ordered by no edge of the template, so it shares no page with
     anything).  The program's side of the order, which the plan neither knows nor checks: copy t mod k is written for
     execution t only after execution t - k has ended, and execution t reads it only after that write.
   - Out: `section`, every field: `first` its first row, `pages` P x block (P its blocks), `bytes` as given, `count` its
     copies (u for a routed value and a MESH_EXECUTION value of `lag` 0, k for a value of `lag` k >= 1, 1 for every
     other), `stride` P.  Copy c's rows are first + c x P + i, i < P, and its first byte mesh_section_address(section,
     c).  A program binds execution t of a routed value or a value of `lag` 0 to copy t mod count, and execution t of a
     value of `lag` k to copy t mod k.  Its addresses are known after the plan, never before.

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
   IEEE TPDS 16(6) 2005, Definition 4), a page crossing a tree edge once (T4).
   The order of the lists is R5's reference algorithm, MSCCLang's (M. Cowan et al., MSCCLang: Microsoft Collective
   Communication Language, ASPLOS 2023) §5.2, step 3: "Sort instructions into a global topological order with respect
   to their dependencies with a heap using the priority to order instructions."  The instructions are the task graph's
   tasks, then the plan's communications (a page over a tree edge; its send at the edge's upper node and its receive at
   its lower node are one instruction at one place, so both ends' lists agree, where MSCCLang §4.2 makes them two,
   joined by a communication edge: GO1).  The dependencies are each local edge of the task graph, and each remote edge
   as a chain from its tail task through its route's communications to its head task (§4.2's communication and
   processing edges: a page sent before it lands, landed before it is forwarded or consumed; GO1).  The priority is
   each instruction's completion time C̄_j in the optimal solution of R5's one linear program (design/plan-to-bridge-
   change-request.md R5 at mesh 0e56d8da; mesh-graph.h mesh_graph_completion_times): its length a task's weight or a
   communication's c(e) / b(L), the greatest over the edges whose routes carry it (GO2), its resource its node's device
   or its link in one direction; C.-Y. Chen, arXiv 2205.02474, 2022, (14), (14a), (1b) and (1c), and L. A. Hall, A. S.
   Schulz, D. B. Shmoys and J. Wein, Math. Oper. Res. 22(3) 1997, (5) on each resource, solved by cutting planes with
   Queyranne's separation, each linear program by HiGHS; the least is taken first, a tie to the lower instruction (R5;
   mesh-graph.h mesh_graph_heap_order).  Each of this node's entries takes its communication's place, and every rank
   computes the same order from the same inputs (F. B. Schneider, Implementing fault-tolerant services using the state
   machine approach, ACM Comput. Surv., 1990).
   2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", LP): the priority the linear
   program's C̄, where it was the bottom level by eq. (7) (GO2, GO3), and the contention schedule (Sinnen and Sousa's
   Algorithms 1 and 2), whose starts ordered no list (GO5), no longer computed.  Why: R5 at mesh 0e56d8da, "It
   replaces MSCCLang's depth and reverse depth, and edge scheduling's start times, which are no longer computed for the
   order".  What must stay: both ends' lists of a node pair the same hops in the same order; every copy's the same
   order.
   2026-11-04, owner C3 (GO1 to GO5; R5, "The order of each cable's list, by the reference algorithm"): reversing T6,
   each node pair's list in the order of its link's scheduled starts of its pages, ties by walk.  Why: R5 orders each
   cable's list as a subsequence of one global topological order built by the reference algorithm (the operator:
   "what algorithm is cited here, if any? no citations? use the reference algorithm instead"), and scheduled starts
   order no list; this overturns T6.  What must stay: the schedule's call (R5's method, A6; 2026-11-05, LP: the call
   gone, R5's linear program the method now); both ends' lists of a node pair the same hops in the same order; every
   copy's the same order.
   The sets (design/aot-compilation-change-request.md, appendix item 5; design/compilation-coordination.md, S1, S2 and
   "The revisions of 67ca86f8", SP): a range every one of whose declared transfers is MESH_PAGES (function B at every destination, R4) is a set on
   each of its trees: every entry of a page on tree j (j the tree's index in the solve's trees grouped by demand, the
   same at every rank) has `set` j + 1; every other entry, a range with any MESH_JOIN destination among them, has `set`
   0.  Every entry, a set's among them, is at its own place (R5: "A set's pages ... keep their places in that order.
   The set's random permutation is drawn over those places"): each copy's entries are sorted by their place.  So each
   node pair's list at each end is a subsequence of the one global order, both ends' lists are the same hops in the
   same order, and every copy's is the same.  The global order is a topological order of the instructions, whose
   dependencies are the bridge's store-and-forward waits; with every lag pointing forward, the cables' sequences in
   that order cannot wait on each other in a cycle (R5; MSCCLang §5.2: "all implicit dependencies introduced by thread
   block sequential execution cannot produce cycles").  The bridge draws one permutation of a set's every position on
   a pair each execution, each cable carrying its own members in that order (appendix item 5; mesh-flow.c), spreads a
   node pair's sequence as its own (R5), and sees a set or a sequence and no kind (R4).
   2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", OP): one permutation of the whole set
   per execution, where each cable drew its members' own (S7′).  Item 5's argument for the permuted sequences ("each
   cable's sequence is the same renaming of them") holds where one cable carries every position of the set on each
   pair of its tree, and is not shown where the deal splits a set's positions over a pair's cables ("The revisions of
   46ae8fdf", stop 3).  What must stay: as above.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 67ca86f8", SP): reversing GO4's groups (a set's
   entries on one node pair and direction at consecutive places at its first page's place, each copy's entries sorted
   by their group's least place, then their own) and the argument that the groups in that order were a topological
   order with each group whole.  Why: R5, "(Corrected 2026-11-05: the first text moved a set's pages to consecutive
   places, which nothing requires.)"  The documents state no argument for the sequences each execution's permutation
   makes ("The revisions of 67ca86f8", stop 3).  What must stay: both ends' lists the same hops in the same order;
   every copy's the same order.
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
   source range), the GPU's status and this node's entries, each part 64-aligned
   (mesh-collective.c plan_layout gives the offsets).  Its words are set to their starting values once, where it is
   placed (the contract's D3), and none is reset while the plan stands (C3.7).  The struct's words beside them:
   `group` and `epoch` the plan's name (group, function), `epoch` holding the function (N1), `serial` the map serial
   it was made on, `hops` its entries' region offset and `entries` u x n, `sequences` 0, `held` 0 and every
   `refused` 0 (the ports' alone), `nodes` the plan's nodes (a set of map indices: every source and destination of a
   declared transfer and every node a tree of it reaches; this node's own where there is none) and `sender` the least
   of them (Q2), each the same at every rank; and `submitted`, `released` and every `taken` 0, the bridge's alone
   (Q2).  Each claimed
   row's publication (mesh.h struct mesh_publication) is written: a received range's first row names its phase groups
   and count words, a source range's first row its supplied word, every other field 0.  No data page is written by
   the plan.
   2026-11-04, owner C3 (DL1; design/aot-compilation-change-request.md, appendix item 2): the sequence words placed
   between the status and the entries, and `taken` where `served` (the executions port p had served whole, Q4's hold)
   was.  Why: the parent publishes each released execution's sequence number in its plan's state, one word per copy,
   each word one writer, and each port merges them; Q4's hold is gone.  What must stay: every word before the sequence
   words where it was; every word started once and none reset while the plan stands.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", LP9): the sequence words gone, the
   entries after the status again, `sequences` 0.  Why: design/aot-compilation-change-request.md, appendix item 2 at
   mesh 0e56d8da, "publish them to each port through the channel above, in place of the sequence words (`sequences`),
   `taken[p]` and the parent's wait while they are full (Lamport's buffer; the coordination's DL1)".  What must stay:
   every word started once and none reset while the plan stands.
   2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", PS, PQ and RC): between the status and
   the entries, the plan's delivered calls' queue (`room` entries at `deliveries`) and its program's submissions'
   queue (`outstanding` entries at `submits`); `sender` the least of `program`; `delivered`, `submissions`,
   `broadcast`, `arrived`, `announced` and `submitted` 0; the counts every received range's, a relay's too (mesh.h
   struct mesh_state's notes).  Why: design/aot-compilation-change-request.md A7 and appendix items 2 and 5.  What must
   stay: every word started once and none reset while the plan stands.

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
/* 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", points 4 to 8, and "The
   revisions of 4e78f6f4"), reversing the notes above where they state one function's plan in its own claim, its part
   of the IPC graph, greedy by size, and a range "one range's alone":
   - The plan call takes the one IR of the whole program and writes its one serialized operand (point 5: "The plan
     call's form for a whole program is A8 and A9.  A8: "The compilation must carry one IR from partitioning to
     serialization, as an operand of its own" ... With the program exported whole (point 1), there is one IR of the
     program and one serialized operand of it").  The IR is each compiled function's declared transfers, weights,
     values and dependences (struct mesh_function, R3's task graph with R4's kinds and A2's and A3's operands) and the
     program's calls of them in program order (`calls`, A10's graph).  The serialized operand at this node is, for
     each function, its plan (A9: "for each adjacent node and direction, the ordered list of page instructions ...
     the page table's bindings and the plan's state section"), with the root that names them at the layout's one
     place (mesh.h struct mesh_program).  What the bridge runs from it is A9's self-timed schedule (A9, "What the
     bridge runs: the self-timed schedule": "Each processor executes the actors assigned to it in the order specified
     at compile time.  Before firing an actor, a processor waits for the data needed by that actor to become
     available").
   - One offline allocation places every value of every function at compile time (point 5; design/plan-to-bridge-
     change-request.md R8, corrected: "One offline allocation over the whole exported program ... at compile time ...
     Nothing takes pages at run time, and no handle is looked up at run time").  Every item is one block, placed
     anywhere, by the fewest chains of the interference order (point 6; R2, corrected: "Two items may share a page
     exactly when one's last access happens before the other's first write ... the fewest pages is the fewest chains
     ... found by maximum bipartite matching (D. R. Fulkerson ...)"; mesh-graph.h mesh_graph_chains).  The rows and
     pages are numbered from the region's first row and from the first block of each space: there is no claim, no
     first fit and no ENOMEM from fragmentation.
   - A value is its ordered page-table rows (R2, corrected; mesh-section.h).  A value typed to be seen as one run
     (`run` 1: R4's function A's operand, or a torch MPS operator's operand, design/aot-compilation-change-request.md
     appendix item 7) is given its run as a mapping written when the plan is made: consecutive virtual pages over its
     pages (R2, corrected: "Such an operand's run is address translation, fixed when the plan is made"; mesh's
     `ncclMeshPageTableCompose` view, `mach_vm_remap`, design/page-maps-2026-11-02.md at 6255e92b; mesh-dataflow.h
     mesh_view_compose).  The program's mapped values at this node are one mapping (`*view`).
   - A declared transfer names the blocks of this node's value it lands in or is sent from (`first`), as NCCL's
     receives land at offsets of one result (point 7; R2, corrected: "receiving data from rank `i` at offset
     `i*sendcount`" (NCCL, ncclAllGather) ... A range is not a value of its own").
   - The program's whole IPC graph is built here, every function together, with each queue's room (5) in it (point
     4; appendix item 2: "the compilation builds the program's whole IPC graph ... and each queue's room is (5) in that
     graph"); mesh-collective.c plan_rooms' part graph goes.
   What must stay: every rank passes the same IR but its values and `value`s, and computes the same demands, trees,
   template, u, schedule and order of each function from them; every failure but ESTALE and the second map read's is
   found before anything is written. */
/* 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1;
   design/plan-to-bridge-change-request.md R1 at mesh 56ede97e, "Why the code copies" and "What the code must do
   instead"; "The revisions of 14b9a757"), reversing the notes above where a declared transfer names one run of blocks
   (`first`, `bytes`), a value's layout is fixed for a transfer, and a value's blocks are another's by a struct
   mesh_share:
   - A transfer's source and destination are each a view of a value (struct mesh_strided: offset, sizes, strides), as a
     torch tensor is one storage under an offset, sizes and strides (torch 2.14.0 c10/core/TensorImpl.h, `Storage
     storage_`, `sizes_and_strides_`).  R1: "A transfer's source and destination are each a view (offset, sizes,
     strides) of a value.  The compilation flattens each view into its list of (row, offset, bytes) blocks, adjacent
     blocks merged, as datatype processing flattens a type map.  The bridge's instructions are those blocks (aot A9:
     MSCCL-IR's steps with offset and count)"; MPI's derived datatypes, a type map read or written in place (R. Ross,
     N. Miller and W. D. Gropp, EuroPVM/MPI 2003; J. Wu, P. Wyckoff and D. K. Panda, IPDPS 2004, as R1 cites them).
   - mesh_plan flattens each view of a range into its extents of the value's bytes, in its index order, adjacent extents
     merged; the range's bytes are the concatenation of its source view's extents, and the i-th byte of the source's
     goes to the i-th byte of each destination's (MPI's matching of type signatures).  Its instructions are the common
     refinement of the source's and every destination's extents, each cut at its value's block boundaries, and of the
     range's own bytes cut at every multiple of a block's bytes (block x pgsz), so each instruction is one extent within
     one block at its source, at each destination and at a relay, as an MSCCL-IR step names one offset and one count at
     each end.  Every rank computes the same instructions from the same views (aot A7, corrected: every member of a view
     plans on the view's map; F. B. Schneider, ACM Comput. Surv. 22(4), 1990), so both ends' lists hold the same hops.
     A relay holds the range in its own bytes' order, its byte s at its block s / (block x pgsz) and offset s mod
     (block x pgsz).  The refinement's cut at the range's own block multiples is the coordinating implementer's, so a
     relay's pages are its range's bytes in blocks (the record, "Decisions").
   - A value's layout is the program's own: no layout is chosen for a transfer, no part is padded, and nothing is packed
     or unpacked (R1: "Then there is no packing and no unpacking, no layout chosen for a transfer's sake, and no share
     record.  A value's layout is the program's own, torch's strides").  Ranges at one node may share a block where
     their bytes are disjoint (PB R2, corrected 2026-11-06: "Ranges may share a block where their bytes are disjoint");
     at a source any number of ranges read the same bytes.  "no two ranges at this node share a block of it" goes.
   - A frame (a placed block) is named by the rows of the values whose blocks it holds over the program's one
     allocation, the page table's own sharing (PB R2, 2026-11-06: "A value's rows naming another value's frames is the
     page table's own sharing, with no record and no rules of its own.  The record goes"); struct mesh_share, struct
     mesh_function `share` and `shares`, and their EINVALs go.  A block a range's instructions touch is an item whose
     accesses are those of every range touching it, from the first write to the last read, ordered by C3.3's lag test
     (R-1: "A frame's item lives from the first write to the last read over every value naming it, with C3.3's lag
     test"); a block holding bytes no range here covers is its value's own item, ordered by the value's `begin` and
     `end` (ExecuTorch, exir/memory_planning.py, update_tensor_lifetime and _does_not_overlap), a range sent or
     forwarded from it holding it to the wait for its send-completion counters (MPI 4.1 §3.7: "until the send
     completes"; design/problems-of-14b9a757-change-request.md, "Sources for the third stops", 7).
   - Any value a transfer names is sent from its own pages, whatever its copies (PB R2, corrected 2026-11-06): the plan
     call's rules that a lag-k value is "named by no declared transfer" and that a transfer's value is "MESH_EXECUTION
     and wire" go.  A source is a MESH_EXECUTION value (EINVAL otherwise: design/problems-of-14b9a757-change-request.md,
     "Sources for the second stops", 8), a value of `lag` k >= 1, a host producer's (C3.1), among them, whose copies
     are then the plan's u, written for execution t into copy t mod u after execution t - k has ended, the program's
     dependences declaring that order (struct mesh_dependence's premise).  A destination is a MESH_EXECUTION value of
     `lag` 0, its landing that execution's write (EINVAL otherwise: a landing writes its bytes each execution, C3.7).
     Every value a transfer names here is placed in the registered window, whatever its `wire` ("is placed in the
     registered window when a transfer names it").
   - A range's words at this node are named by its declared transfer's index, not by a row: ranges sharing a block
     begin in one row (mesh.h struct mesh_state `transfers`; mesh_host_publish, mesh_host_arrived and mesh-metal.h).
   What must stay: every rank passes the same IR but its values and `value`s, and computes the same demands, trees,
   instructions, template, u, schedule and order of each function from them. */
enum { MESH_INVARIANT, MESH_EXECUTION };
/* a value at this node.  In: section.bytes (> 0), kind, wire, state (1 only on a local MESH_INVARIANT value, wire 0),
   for a MESH_EXECUTION value of lag 0 its first and last positions on this node's own program order in one
   execution, begin <= end, and lag: 0 for a value the program's own order writes (u copies: design/problems-of-
   14b9a757-change-request.md, "Appendix: the stops of 36bd7cbd", 14), k >= 1 for one a producer the program orders
   by an event writes (a host input: MESH_EXECUTION, wire 0, state 0, named by no declared transfer, begin not read
   and end read where a transfer names it; k copies, execution t's at copy t mod k, written for t only after execution
   t - k has ended and read by t only after that write: the program's order, which the plan does not check;
   design/disjoint-pages-change-request.md, D.3.3; design/disjoint-pages-rulings-coordination.md, R4).  Out: section,
   every field.
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", points 6 and 8; "The
   revisions of 4e78f6f4"): `run` and `view` appended (64 bytes).  In: `run` 1 where a consumer of the value at this
   node is typed to see it as one run (design/plan-to-bridge-change-request.md R4, corrected: "Only A's operand, and
   an operand whose consumer needs one strided progression, is seen as one run.  That run is a mapping written when
   the plan is made"; a torch MPS operator's operand, design/aot-compilation-change-request.md appendix item 7: "an
   operand of a torch MPS operator is seen in one strided progression of virtual addresses"), else 0.  Out: `view`,
   where `run` is 1, the address in this process of copy 0's first byte in the program's mapping (`*view` of
   mesh_plan), copy c's at view + c x section.pages x pgsz and every copy's blocks consecutive there, in row order;
   NULL where `run` is 0.  The value's rows are bound to blocks placed anywhere (mesh-section.h): no value is placed
   as a run.  What must stay: every field before `run` where it was.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G3 and K4; "The
   revisions of 55b9cbad"), reversing `view`:
   - `view` goes (`padding1` in its bytes): the plan writes no address.  The program's effector maps its `run` values'
     rows itself (mesh-dataflow.h mesh_view_compose) once the plan is written, and binds each value at its place in its
     mapping.  Why: G3, "The planner calls `mach_vm_remap` and writes virtual addresses into the IR, and reads
     `getpid()` ... Deciders see pages.  The plan writes rows, and the program's effector maps them."
   - `shares` out, where `padding` was: 1 where a page bound in one of the value's rows is bound in another row at this
     node (another value's, or another copy's of the same value: items on one chain of the one allocation, or a struct
     mesh_share), else 0; a mapping of every copy puts two such rows at two virtual addresses of one page.  Why: K4, "the
     plan knows which blocks two values share and where one hands over to the next.  The flush goes on those barriers
     alone": a program that reads its values through a mapping reads two such values at two virtual addresses of one
     page, and only a barrier between their accesses needs the alias flush (metal-microbench metal_recording.m
     MetalEncoder; MTL4CommandEncoder.h, MTL4VisibilityOptionResourceAlias: "Flushes caches to ensure that aliased
     virtual addresses are memory consistent").
   What must stay: 64 bytes, every field before `shares` where it was
   2026-11-06 ("The root cause of the copies, and the revised work list", R-1 and R-5; "The revisions of 14b9a757"):
   - A value a declared transfer names here (its source's or a destination's `value`) is placed in the registered
     window whatever its `wire`; its copies are u, a value of `lag` k >= 1 among them (the note before this struct).
     `wire` is read for every other value.
   - `shares` (out) is 1 where a frame bound in one of the value's rows is bound in another row at this node (another
     value's or another copy's, by the program's one allocation: no record makes one), as above.  R-5: "The plan knows
     each frame's virtual names through R-1's rows.  Metal 4's `MTL4VisibilityOptionResourceAlias` ... goes on the
     barrier between a write under one name and a read under another, and on no other": a value whose `shares` is 1
     hands its frames to another row's name after its last access in each execution, and another hands them to it
     before its first.
   What must stay: 64 bytes, every field where it is
   2026-11-06 (design/problems-of-14b9a757-change-request.md, "Sources for the stops", item 7): `begin` and `end` the
   value's lifetime, "the index of the first and last node referring that tensor in its inputs/outputs" (ExecuTorch,
   exir/memory_planning.py, update_tensor_lifetime), each declared transfer's end here occupying it from the position
   that starts the transfer to the one that waits for its completion (MPI 4.1 §3.7) */
struct mesh_value { struct mesh_section section; uint32_t kind,wire,state,begin,end,lag,run,shares; uint64_t padding1; };
_Static_assert(sizeof(struct mesh_value)==64 && offsetof(struct mesh_value,run)==48 &&
               offsetof(struct mesh_value,shares)==52 && offsetof(struct mesh_value,padding1)==56,
               "mesh_value: rdma/mesh.py Value");
/* plan-to-bridge R4: the kind of a declared transfer's consumer at its destination.  MESH_JOIN: function A, one task
   with an edge from every page of the range.  MESH_PAGES: function B, one task per page, each with an edge from its
   own page */
enum { MESH_JOIN, MESH_PAGES };
/* `order` (the hand priority, lower first) is gone: R5's schedule orders every cable's sends; `kind` in its place.
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 7; "The revisions
   of 4e78f6f4"): `first`, where `padding` was: the range's first block in this node's value `value` (a block one row,
   mesh-section.h), where this node is its source (the blocks it is sent from) or a destination (the blocks it lands
   in), so the range's P blocks are rows [first, first + P) of each copy of that value; 0 where `value` is
   MESH_ABSENT.  Why: point 7: "name, in each declared transfer, the pages of its destination value that it lands in,
   as NCCL's receives land at offsets of one result ... A collective's result is then the pages its parts land in, so
   no concatenation and no copy of a part remains"; R2, corrected: "A collective's receive lands in the pages of the
   value it belongs to, at its place in that value".  A range's every declared transfer from one source gives its
   source one `value` and `first` (EINVAL otherwise); the value is MESH_EXECUTION and wire, the range's blocks lie
   within its blocks, and no two ranges at this node share a block of it (EINVAL otherwise); its blocks no range
   covers here are one item of their own, which shares no block with any other (no edge of the template orders their
   write, as a value of `lag` k's copy).  "one range's alone" goes.  What must stay: 32 bytes, every other field where
   it was */
/* A view of a value's bytes (2026-11-06, "The root cause of the copies, and the revised work list", R-1: "A transfer's
   source and destination are each a view (value, offset, sizes, strides)"; PB R1 at mesh 56ede97e; "The revisions of
   14b9a757"): a torch strided view, one storage under an offset, sizes and strides (torch 2.14.0
   c10/core/TensorImpl.h `storage_offset_`, `sizes_and_strides_`), its offset and strides in bytes and `element` the
   bytes of one element.  Its bytes are, for every index (i_0 .. i_{dims-1}) with i_d < size[d], the extent
   [offset + sum_d i_d x stride[d], + element) of each copy of the value, in the order of the index with the last
   dimension fastest (C order, the order in which torch's copy_ matches two tensors' elements); `dims` 0 is the one
   extent [offset, offset + element).  `pairs`: the view's `dims` (size, stride) pairs, size[d] at pairs[2 d] and
   stride[d] at pairs[2 d + 1], read for d < dims alone (MPI 4.1 §6.1.3, MPI_TYPE_CREATE_SUBARRAY: "The ndims parameter
   specifies the number of dimensions in the full data array and gives the number of elements in array_of_sizes,
   array_of_subsizes, and array_of_starts").  `element` >= 1, every size[d] >= 1, and every extent within the value's
   bytes (struct mesh_section `bytes`): EINVAL otherwise, at the plan.  What must stay: every field where it is
   (rdma/mesh.py Strided) */
struct mesh_strided { uint64_t offset,element; uint32_t dims,padding; const uint64_t *pairs; };
_Static_assert(sizeof(struct mesh_strided)==32 && offsetof(struct mesh_strided,dims)==16 &&
               offsetof(struct mesh_strided,pairs)==24,"mesh_strided: rdma/mesh.py Strided");
/* 2026-11-06 (R-1; "The revisions of 14b9a757"), reversing `first` and `bytes`: a declared transfer's range is the bytes
   its source's value shows through `from`, landing in the bytes its destination's value shows through `to`, both every
   rank's alike (a node plans every end's instructions from them: the note before struct mesh_value).
   - `from`: the range's bytes in its source's value, the same in every declared transfer of one range (EINVAL
     otherwise); `to`: declared[i]'s bytes in its destination's value.  The two name as many bytes (EINVAL otherwise),
     the range's bytes B, and byte i of `from` lands at byte i of `to`.
   - `value`: the index in `values` of this node's value of it, the one `from` views where this node is the source, the
     one `to` views where it is declared[i]'s destination; MESH_ABSENT elsewhere.  A destination's value is
     MESH_EXECUTION of `lag` 0 (EINVAL otherwise).
   `first` and `bytes` go: a range's place is its views, and its bytes theirs.  What must stay: `source`,
   `destination`, `identity`, `kind` and `value` where they were */
struct mesh_declared { uint32_t source,destination,identity,kind,value,padding; struct mesh_strided from,to; };
_Static_assert(sizeof(struct mesh_declared)==88 && offsetof(struct mesh_declared,from)==24 &&
               offsetof(struct mesh_declared,to)==56,"mesh_declared: rdma/mesh.py Declared");
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
/* One compiled function of the program, its part of A8's IR (2026-11-05, design/compilation-coordination.md, "The
   revisions of 4e78f6f4", where mesh_plan took one function's operands as its own arguments): its `count` declared
   transfers and their weights, its `values` values at this node (`value`, each In and Out as struct mesh_value
   states, a declared transfer's `value` an index among them), its dependences, and `outstanding`, A10's bound for it,
   "the most calls of it the analysis lets be outstanding at once in the program that submits them" (A10, "Its
   output"), the delay on the edge from a call's completion to the program's later submission of its call that many
   calls of the function later (appendix item 2: "At each node, A10's bound is the delay on the edge from a call's
   completion to the program's later submission"); 0 is ENOMEM ("A plan whose bound cannot be found fails with
   `ENOMEM` when it is planned").  Out: `plan`, the region offset of its plan's state section at this node (struct
   mesh_state), the handle mesh_transfers_start, mesh-metal.h and mesh_plan_retire take, a constant of the plan.
   Function k (from 0) is named (group, k + 1) (struct mesh_state `epoch`; design/aot-compilation-change-request.md,
   appendix item 6: "name each compiled function alike at every rank: (the program's group, the function's ordinal in
   the program's order of compilation)"), the plan call's order being that order. */
/* A value's blocks that are another value's (2026-11-05, design/compilation-coordination.md, "Review of 4e78f6f4
   against the requirement lines", V2, and "Answered in the documents", a rank's own part; design/plan-to-bridge-
   change-request.md R2, corrected: "A rank's own part is shared, not copied.  A value is its page-table rows, so the
   result's rows for the rank's own part name the operand's own pages.  This is sharing by reference through the page
   table, as fbufs (P. Druschel and L. L. Peterson, Fbufs: a high-bandwidth cross-domain transfer facility, SOSP 1993)
   and IO-Lite (V. S. Pai, P. Druschel and W. Zwaenepoel, IO-Lite: a unified I/O buffering and caching system, ACM TOCS
   18(1), 2000) pass buffers without copying them.  Those pages are one item of the allocation, living until the later
   of the two values' last reads"; "The revisions of 55b9cbad").  In: values[`value`]'s blocks [`first`, `first` +
   `blocks`) of each copy c are values[`of`]'s blocks [`at`, `at` + `blocks`) of copy c: the rows of both are bound to
   the same pages, so nothing copies the part (AllGather's or AllToAll's own part: `value` the result, `of` the
   operand).  `value` != `of`, both indices of the function's values here, of the same copies (both named by a declared
   transfer here, or both of the same `lag`), `blocks` >= 1, each run within its value's blocks, `value`'s run covered
   by no declared transfer here and by no other share; EINVAL otherwise.  The shared blocks are one item of the one
   allocation, whose lifetime covers both values' accesses (mesh-collective.c's minimal form is the implementer's,
   recorded at the code).
   2026-11-05 ("The revisions of 55b9cbad", sub-agent C's decisions, accepted): EINVAL also where `of`'s run is itself
   another share's run (a share of a share: every share is tested before any is bound, so no order of them decides),
   and where the two values' `wire` differ (a block lies in one space, mesh-dataflow.h mesh_arena_range).  What must
   stay: every test before anything is written
   2026-11-06 ("The root cause of the copies, and the revised work list", R-1; design/plan-to-bridge-change-request.md
   R2 at mesh 56ede97e: "The share record of the implementation (`struct mesh_share`), and its rules, is a restricted
   form of the sharing stated above.  A value's rows naming another value's frames is the page table's own sharing,
   with no record and no rules of its own.  The record goes (R1)"; "The revisions of 14b9a757"): struct mesh_share and
   its tests go, and struct mesh_function's `share` and `shares` with it (64 bytes, as before 55b9cbad).  Two values'
   rows name one frame by the program's one allocation alone (the note before struct mesh_value).  What must stay:
   `outstanding` and `plan` as their notes state them
   design/problems-of-14b9a757-change-request.md, "Appendix: the stops of 36bd7cbd", 14: In: `lead`, the program's start
   of execution t (mesh-call.h mesh_host_started's raise of `executions` past t, its receives' post) at the start of
   execution t - `lead` in this node's program order, after every read the program makes before that point */
struct mesh_function { const struct mesh_declared *declared; const struct mesh_weight *weights; struct mesh_value *value;
                       const struct mesh_dependence *dependence; uint32_t count,values,dependences,lead;
                       uint64_t outstanding,plan; };
_Static_assert(sizeof(struct mesh_function)==64 && offsetof(struct mesh_function,count)==32 &&
               offsetof(struct mesh_function,outstanding)==48 && offsetof(struct mesh_function,plan)==56,
               "mesh_function: rdma/mesh.py Function");
/* The 116 plan call's notes on its operands, kept (2026-11-05, "The revisions of 4e78f6f4": restored where dd215cbe
   dropped them; sub-agent C's finding).  aot-compilation appendix item 6 (design/compilation-coordination.md, "The
   appendix's revisions", N1 to N5): the plan of one compiled function, named (group, function): `group` the
   launcher's (MESH_GROUP), 0 EINVAL; `function` the program's own ordinal of the compiled function, the same at every
   rank, kept by every replan of that function; held in struct mesh_state `epoch`.  EEXIST where a plan of that name stands here on the same map.  *plan out: the region
   offset of the plan's state section once it stands (untouched on any failure).  No earlier plan is closed.
   2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", PS and PQ): `program` and
   `outstanding` added after `function`.  `program`: the program's nodes, a set of map indices (bit i, map index i),
   the same at every rank and in every plan call of one program (`group`); its least is the program's one sender
   (design/aot-compilation-change-request.md A7: "sender(m) is the program's one sender: the least of the program's
   nodes, the program being its launcher's `group`"), held in struct mesh_state `sender`; EINVAL where it is 0, lacks
   this node, or lacks a node of the plan's (a relay outside the program would never hold the plan, so the barrier
   would never count it: A7, "A plan's destinations exist before its first message").  `outstanding`: A10's bound for
   this function, "the most calls of it the analysis lets be outstanding at once in the program that submits them"
   (A10, "Its output"), the room of the plan's submissions' queue and the first term of its delivered calls' queue's
   room (appendix item 2, "Its room is fixed at compile time"); 0 where A10 found none, which fails ENOMEM (item 2: "A
   plan whose bound A10 cannot find fails with `ENOMEM` when it is planned").  What must stay: every operand the same
   at every rank.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 3f106747", BS and PG): `outstanding` is the delay
   on the edge from a call's completion to the program's later submission in the program's IPC graph, and no room: each
   queue's room is its self-timed buffer bound there, with the synchronization edges Convert-to-SC-graph adds and
   DetermineDelays' delays (S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997,
   §4.3 (5), §9, Figs. 6 and 8; mesh-collective.c plan_rooms); ENOMEM also where a cycle has zero delay (BSL Lemma 1).
   `program` is the nodes the program's launcher started it at, passed by the launcher as it passes `group`
   (design/aot-compilation-change-request.md A7, sender(m): "The program's nodes are the nodes its launcher started it
   at, an operand of the plan call (`program`) that the launcher passes as it passes `group` (MESH_GROUP)"); every run
   is launched on the whole collective, so `program` holds every node of the map, every relay among them.  What must
   stay: every operand the same at every rank.
   2026-11-05 ("The revisions of 4e78f6f4"): `function` and `*plan` are struct mesh_function's (function k named (group,
   k + 1)), EEXIST is EBUSY on the program's root, and `outstanding` is each function's (struct mesh_function); the
   rest stands */
/* The plan call (2026-11-05, design/compilation-coordination.md, "The documents' corrections of 2026-11-05", points
   4 to 8; "The revisions of 4e78f6f4").  `group` the launcher's (MESH_GROUP), 0 EINVAL.  `program` the program's
   nodes, a set of map indices (the notes on `program` just above).  `functions`, `count` of them, the program's compiled
   functions, each struct mesh_function; `calls`, `ncalls` of them, the program's calls of them in program order in
   one execution of its traced block (A10, "The calls' graph": "each compiled function is represented by a computation
   node in a dataflow graph"), calls[j] the index of the j-th call's function, every function called at least once
   (EINVAL otherwise).  e2b's program is one function, called once.
   - Each function's demands, trees, template, u, entries and schedule as before, at every rank alike.
   - The program's IPC graph (appendix item 2, "The graph"): an actor for each node's part of each call ("The actors
     are each node's parts of the program's calls, in program order"), for each call its program's submission σ and
     its bridge's TO-broadcast β at the sender (the least of `program`), each sequence of the sender's in program
     order as BSL §4.1's processor cycle (delay 1 from the last call's to the first's); each call's queues (its
     function's), transfers (delay its function's u) and A10's edges (from each call's end at a node to the start
     there of its function's call `outstanding` calls later, at the sender its submission; the delay the executions
     of the block between them); each function's queues' rooms the sum of its calls' edges' bounds (5) in the graph
     Convert-to-SC-graph and DetermineDelays make of it, over the whole graph; each added edge an entry of its plans'
     synchronization edges (mesh.h struct mesh_sync).
   - One allocation of every item of every function at this node, and the program's root (mesh.h struct
     mesh_program) at the layout's place, its `table` among the items.
   0; EINVAL (as before, and the IR's malformed `calls` or `first`); EBUSY where the root's `owner` is not 0 (another
   program's operand, or this program's own earlier one, stands or is not yet released: struct mesh_program's note),
   before anything is written; EHOSTUNREACH; ENOMEM (a space's chains past its blocks; a value with no unroll bound; a
   cycle of zero delay; a queue on no cycle after the conversion; A10's bound 0; the call's own memory; a mapping
   Mach refuses); ESTALE (the map moved while planning: plan again on the map as it is); or the map reader's error.
   On ESTALE, the second map read's error, or a mapping Mach refuses (mesh_view_compose reads the rows' bindings, so
   it follows their writes: sub-agent C's finding, "The revisions of 4e78f6f4"), the root is not published and the
   mapping is released.  There is no
   ETIMEDOUT: nothing waits.  `*view` out: the program's one mapping of its `run` values at this node (mesh-dataflow.h
   mesh_view_compose: every `run` value's rows, function by function and value by value in `values`' order, each
   value's copies in order), which its holder releases with mesh_view_release once it retires the program; {NULL, 0}
   where no value is `run`, and on any failure.
   2026-11-05 (design/compilation-coordination.md, "Answered in the documents (2026-11-05, the coordinator, on this
   part's record)" and "Review of 4e78f6f4 against the requirement lines"; "The revisions of 55b9cbad"), reversing the
   notes above where they differ:
   - No mapping and no address (G3: "The planner calls `mach_vm_remap` and writes virtual addresses into the IR, and
     reads `getpid()` ... Deciders see pages.  The plan writes rows, and the program's effector maps them"): `*view`
     goes, with struct mesh_value `view`; the program maps its `run` values' rows itself once the plan returns
     (mesh-dataflow.h mesh_view_compose) and releases its mapping (mesh_view_release).  A mapping Mach refuses is the
     program's error, after a plan that stands.
   - The program's IPC graph's processors at each node (IG-5; design/aot-compilation-change-request.md appendix item 2,
     corrected: "The program-order cycle is each node's submissions, not its calls' parts ... the submissions σ_1 …
     σ_n of the block's calls are one processor's actors, in that order, each of time 0 ... Call k's part at the node,
     x_k, follows its submission (σ_k → x_k, delay 0).  A10's bound b is the edge from x_k to the submission b calls
     later ... no edge orders the parts themselves"): at each node of each call's plan, a submission actor of time 0
     for the call, the node's submissions in program order as BSL §4.1's cycle (delay 1 from the last to the first),
     each followed by the node's part (delay 0), and A10's edge from the part to the node's submission of its
     function's call `outstanding` calls later; the sender's submissions are its node's, followed by its bridge's
     TO-broadcasts as before.
   - The interference order over the u copies is design/plan-to-bridge-change-request.md R2's, as corrected for PO-1:
     "Copies (v, c) and (w, c′) are ordered where c ≤ c′, c′ − c is at least the least lag from v to w, and c′ plus
     w's largest least lag to a write at the node is at most u" (Lam's modulo variable expansion; Hendren, Gao, Altman
     and Mukerji's cyclic interval graphs).
   - A struct mesh_share's blocks are one item of the allocation, bound in both values' rows (V2; struct mesh_share).
   - The waits of each plan's synchronization edges into this node in rows by call place, each delay folded into its
     `base`, and each tell naming its sink's wait (mesh.h struct mesh_state `rows`, struct mesh_sync; K1).
   - Each value's `shares` (struct mesh_value; K4).
   - The publication rule ("Answered in the documents": "The rule is that nothing is visible before the root's
     publication: one release store after every write, in C3's writing order, as RCU publishes a structure only once
     it is initialized (McKenney and Slingwine)"), in place of "Every failure but ESTALE and the second map read's is
     found before anything is written": mesh_plan writes every word of the operand (each plan's state section, its
     synchronization edges and rows, every value's rows and bindings, each range's words, the entries, the table),
     then reads the map again, then writes struct mesh_algo `plans`, `group`, `table` and the holder (hdr `client` and
     `client_start`, from the context's `client` and `start`), then stores `published` last (release).  A failure
     before that store (ESTALE where the map moved, or the map reader's error) leaves nothing published: the pages it
     wrote belong to no published operand, and the next plan writes them again.  Every other failure is found before
     anything is written.
   - The previous operand's release waited for, bounded (B2, EB-1: "In RCU's form the writer waits for the grace period
     to end (`synchronize_rcu`), here the bridge's release of the old root.  So mesh_plan waits for it, bounded by
     MESH_PLATFORM_NS, and returns ETIMEDOUT past the bound"): after everything is computed and before anything is
     written, while struct mesh_algo `published` (acquire) is not 0 and `released` (acquire) is not `published`, a
     wait on `released` (os_sync_wait_on_address_with_timeout, region-shared, woken by the bridge parent's release:
     mesh-dataflow.h mesh_retired_release), at most MESH_PLATFORM_NS from its start, ETIMEDOUT past it with nothing
     written.  The program retires its own earlier operand first (mesh-dataflow.h mesh_retire), and the bridge retires
     a gone holder's.  The record has one writer by construction, the program's process (B2: "since a run is one
     program on the whole collective (A7, corrected), so it needs no exchange and no lock: a single-writer register (L.
     Lamport, On interprocess communication, Distributed Computing 1(2), 1986)"): no compare-and-swap.  EBUSY goes.
   0; EINVAL (as before, and a malformed struct mesh_share); EHOSTUNREACH; ENOMEM (as before, but a mapping);
   ETIMEDOUT (the previous operand not released within MESH_PLATFORM_NS); ESTALE (the map moved while planning: plan
   again on the map as it is); or the map reader's error.  What must stay: every rank passes the same IR but its values
   and `value`s, and computes the same demands, trees, template, u, schedule, order, IPC graph and allocation */
/* 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1, R-6,
   R-7 and R-8; "The revisions of 14b9a757"), reversing the notes above where they differ:
   - The declared transfers are views (struct mesh_declared `from`, `to`), flattened and refined into instructions as
     the note before struct mesh_value states; struct mesh_share and its EINVALs go; each instruction's offset in its
     page is its entry's (mesh.h struct mesh_hop `offset`).  A range's words at this node are its declared transfers'
     entries (mesh.h struct mesh_state `transfers`), not a row's publication.
   - The map planned on is the installed view's (design/aot-compilation-change-request.md A7, corrected 2026-11-06:
     "The map a view is created from is the sequencer's, and the view's first message carries it ... Every member plans
     on those records"; mesh-dataflow.h mesh_nodes_read), so the plan's `serial` is that view's identifier (mesh.h hdr
     `view`), and mesh_plan's second read is ESTALE where the installed view moved.
   - The operand's top level names, for each map index, the places of the plans with an entry toward it (mesh.h struct
     mesh_algo `adjacent`; R-6: "Each cable's processor executes its list of instructions in the compile-time order
     ... It scans no table of plans"), written before `published`.
   - `deadline` (clock_gettime_nsec_np(CLOCK_MONOTONIC) nanoseconds, absolute): the previous operand's release is
     waited for until it, ETIMEDOUT at it, nothing written (R-7: "make()'s waits share one deadline"), where the wait
     took MESH_PLATFORM_NS from its own start.  The caller's one deadline covers its own earlier wait of the same
     preparation (rdma/mesh_compile.py's retirement) and this one.
   - The release waited for is "published and not yet released" (mesh.h mesh_algo_unreleased; R-10), and nothing here
     writes `retired` or `released` (R-8: "the program writes `retired`, the bridge writes only `released`").
   0; EINVAL (as before but struct mesh_share's; a malformed view, two views of one transfer naming different bytes,
   one range's transfers with different `from`, a source's value MESH_INVARIANT (design/problems-of-14b9a757-change-
   request.md, "Sources for the second stops", 8), a destination's value not MESH_EXECUTION of `lag` 0, two of this
   node's destination instructions naming one byte of a value: MPI 4.1 §6.1.11); EHOSTUNREACH;
   ENOMEM; ETIMEDOUT (the previous operand not released by `deadline`); ESTALE (the installed view moved while planning:
   plan again); or the map reader's error.  What must stay: every rank passes the same IR but its values and `value`s,
   and computes the same demands, trees, instructions, template, u, schedule, order, IPC graph and allocation */
int mesh_plan(struct mesh_ctx *,uint64_t group,uint64_t program,struct mesh_function *functions,uint32_t count,
              const uint32_t *calls,uint32_t ncalls,uint64_t deadline);

/* A host (CPU) program's side of the plan's words (mesh_plan), as mesh-metal.m is a GPU program's, each by its range's
   first row's publication, as before: mesh_host_publish its supplied word (`targets[0]`), mesh_host_arrived its words
   and its plan's state section (`state`).
   2026-11-04, sub-assignment 3: mesh_host_inputs is removed (it did nothing: mesh_plan places and starts every word),
   and `invocation` is `execution`, a uint64_t, since a plan runs any number of executions.  What must stay: a host
   producer releases a range by one store, and a host reader reads a range's state, never a word a page. */
/* 2026-11-06 ("The root cause of the copies, and the revised work list", R-1; "The revisions of 14b9a757"), reversing
   "each by its range's first row's publication": each by its declared transfer, `plan` its function's state section
   (struct mesh_function `plan`) and `transfer` its index among the function's declared transfers, entry `transfer` of
   the plan's `transfers` (mesh.h struct mesh_state, mesh_transfer_at), which mesh_plan writes.  Why: ranges may share a
   block where their bytes are disjoint (PB R2, corrected 2026-11-06), so several ranges begin in one row and a row
   names no one range; the IR's index is the program's own name of the transfer, read by index (A9: "the bridge reads
   the serialized operand at a place its compilation fixed, and searches for nothing").  What must stay: one store
   releases a range; a range read by its words, never a word a page */
/* a host producer's release of execution t of a source range: its supplied word raised to t + 1 (release), never
   lowered, so publishes run in any order (a torch MPSEvent's notification, rdma/mesh-torch.mm) leave it at the
   greatest (2026-11-05, design/compilation-coordination.md, "The revisions of e4a7398e", TP-3, where it was stored).
   The caller names a transfer this node is the source of, which its compilation knows; no entry is tested here
   ("Compile-time, never runtime checks"; sub-agent C's finding on this note's first text, "Nothing where this node is
   not the transfer's source") */
void mesh_host_publish(struct mesh_ctx *,uint64_t plan,uint32_t transfer,uint64_t execution);
/* execution t of a received range: 1 where every field of its copy t mod u reads mesh_phase(t / u + 1) (acquire);
   UINT64_MAX where a field reads neither that nor mesh_phase(t / u) (ahead), or where its count has reached
   (t / u + 1) x P with a field still behind (cancelled); else 0 (not yet).  The caller names a transfer this node is a
   destination of, which its compilation knows; no entry is tested here (as mesh_host_publish's note) */
uint64_t mesh_host_arrived(struct mesh_ctx *,uint64_t plan,uint32_t transfer,uint64_t execution);
/* execution t of a range this node sends or forwards, where its send-completion counter of copy t mod u and its error
   count (acquire) together reach (t / u + 1) x its sends a copy: UINT64_MAX where the error count is not 0
   (cancelled), else 1; 0 before (libfabric man/fi_cntr.3.md, fi_cntr_wait and fi_cntr_readerr; man/fi_endpoint.3.md,
   fi_cancel; rdma-core libibverbs/man/ibv_post_send.3, NOTES).  The caller names a transfer whose range this node
   sends or forwards, which the plan's entry tells (mesh.h struct mesh_state `transfers`, `sends`) */
uint64_t mesh_host_sent(struct mesh_ctx *,uint64_t plan,uint32_t transfer,uint64_t execution);
/* The program's one mapping of its values (2026-11-06, "The root cause of the copies, and the revised work list", R-10:
   "the mapping's row list in one mesh-dataflow helper over the plan's rows"; W13: "the mapping's row list in two
   effectors (mesh_compile.py 2289; e2b_replay.m 1437), to be one mesh-dataflow helper"; "The revisions of 14b9a757"):
   every value of `functions` whose `run` is 1, function by function and value by value in `values`' order, each
   value's copies in order, block k of copy c its row section.first + c x section.stride + k (mesh-section.h), mapped
   by mesh_view_compose into `*view`; offset[i] the byte offset in the mapping of value i's copy 0, i counted over every
   function's values in order, copy c at offset[i] + c x section.pages x pgsz, UINT64_MAX for a value whose `run` is 0.
   Called by the program's effector once mesh_plan has written the rows; an effector, given rows and deciding nothing.
   0 or mesh_view_compose's error, nothing mapped and every offset UINT64_MAX (mesh-dataflow.c) */
int mesh_view_values(struct mesh_ctx *,const struct mesh_function *functions,uint32_t count,uint64_t *offset,
                     struct mesh_view *view);
#endif
