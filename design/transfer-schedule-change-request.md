# Change request: scheduled page transfers

This change request says how a plan decides what the bridge sends, and in what order, and what the bridge is given. It
starts from the scheduling objective. Every data structure and every algorithm below follows from that objective, and
each is one canonical method used as its source states it. Nothing here is a variant of its source, and nothing is left
for an implementer to design.

It supersedes design/disjoint-pages-change-request.md, Appendix D: D.2's model of a transfer, D.3.1 and D.5. It also
replaces design/plan-bridge-coordination.md, which is removed. The rest of that change request stands (§9).

Written by Claude on 2026-11-04, from the operator's statements in §1.

## 1. The operator's statements this answers (2026-11-04, verbatim)

> queue pairs are an abstraction which no ocllective and no data processing code can know about. the only way to use the
> dma drivers is to submit work to a queue by view, (reference), not by copy or by value.

> isntead of scheduled transfers which happen from an opaque driver consuming queues of desired transfers.

> the idea that anything needs to be ordered here at runtime seems to be conflating the idea that page transfers have
> dependency orderings (true) and per node placement (true)... but both of these relationships do not imply total
> orderings or local orderings determined aot.

> if your code is compiled aot why can't you push this one-sided before and without blocking any one-sided transactions?

> what does 'ask' even mean here semantically? what's the resource being 'asked' for?

> however, there is a memory allocation model called a 'page table' that the user keeps expecting you to use? is there
> some algorithms topic which is being forgotten here?

> uh, yes, that is the subtext and also the literal textual goal of what we're working on here with this subinterface

(of "static scheduling with costs").

> function A requires set of pages operand_a to be contiguous and complete before function A can begin to execute.
> function B requires set of pages operand_b to be complete but not contiguous before function B is complete. function B
> can work on any pages received in any order and can also release incomplete pages to consumer functions.

> the fact that one pagelist transmitted thrtough the dma driver came from a coflow vs an independent flow is *not
> something the bridge cares about*; both are page -> page sends with no logic related to this typing.

> ultimately a 'scheduling objective' here was so fundamental and central that the choice of representation and
> serialization was totally descendant from what the scheduler 'cares about' (we don't care about the scheduler being
> globally optimal btw; any approximatino measuring well in literature these properties of obvious concern is fine).

> we need a complete change request document articulating why we are choicelessly and unambiguously only using certain
> data structures and ceratin algorithms from certain citations (not 'pastiching' or 'homaging' them. we are using the
> right methods exactly here!).

## 2. The objective

The scheduler cares about one thing: **each round of the program finishes as early as it can.** A round is one execution
of the plan's template (C3.3). A consumer starts when the data it needs has landed. A function of kind A (§1) needs its
whole operand. A function of kind B needs each page as it comes.

- The measure is the round's makespan: the finish time of its last task, with communication counted on the links it
  actually uses.
- Rounds overlap by modulo variable expansion, as now (C3.3: u copies by least lags; Lam, PLDI 1988; Rau, MICRO 1994).
- Optimality is not required. The method is list scheduling, whose bounds and measured behaviour on task graphs are
  standard (R. L. Graham, Bounds on multiprocessing timing anomalies, SIAM J. Appl. Math., 1969; Y.-K. Kwok and I.
  Ahmad, Benchmarking and comparison of the task graph scheduling algorithms, JPDC, 1999).

Everything below exists to serve this objective, and in this order: what the scheduler needs to see (§3), how it decides
(§4), what it writes down (§5), and what consumes that (§6).

## 3. What the scheduler sees: the task graph

**The data structure is the task graph** of the scheduling literature: a directed acyclic graph whose vertices are tasks
with computation costs and whose edges are data with communication costs (O. Sinnen, Task Scheduling for Parallel
Systems, Wiley, 2007; Kwok and Ahmad, Static scheduling algorithms for allocating directed task graphs to
multiprocessors, ACM Comput. Surv., 1999). No other structure is used to decide anything.

It is built from what mesh_plan already takes:

- **Tasks** are the template's events at each node: a range's production at its source, its consumption at each
  destination, a relay's forward, and the source's own read (mesh-collective.h). Each task runs on a known node. The
  program fixed that by its sharding, and the scheduler doesn't move tasks.
- **A node's own order.** Consecutive tasks of one node's program are joined by edges, so the order the program's
  command queue runs them in is kept. The scheduler only orders communications.
- **Computation costs** sit on the program's dependence edges: `struct mesh_dependence` gains `uint64_t nanoseconds`,
  the program's estimate of the work between its `from` and `to`. The estimate is the roofline: the larger of the work's
  floating-point operations over the node's peak and its bytes moved over the node's memory bandwidth (S. Williams, A.
  Waterman and D. Patterson, Roofline, CACM, 2009). Data-model §1 gives the streaming rate measured on the Studios. A
  program that doesn't know the cost passes 0, and the schedule then weighs communication alone.
- **Communication costs.** A page's hop over a link costs its bytes over that link's rate, from the map (mesh-map.h). A
  page that crosses relays is a communication on each link of its route, store-and-forward at page granularity, since a
  relay forwards once its landing is in.
- **Kind A and kind B are the graph's shape, not a type.** `struct mesh_declared` gains `whole` (in place of `padding`).
  With `whole` 1, the destination's consumption is one task with an edge from every page of the range: kind A, a join.
  With `whole` 0, the consumption is one task per page, each with an edge from its own page, the declared cost split
  evenly among them: kind B, which can start on any page and pass results on. Nothing else about kinds exists anywhere.

The graph has no coflows, flows or priorities in it as types. A coflow is a join, and an independent flow is a single
edge. The scheduler's treatment of both follows from the graph (§4).

## 4. How the scheduler decides: contention-aware list scheduling

**The algorithm is list scheduling with edge scheduling under the contention model** (O. Sinnen and L. A. Sousa,
Communication contention in task scheduling, IEEE TPDS, 2005; Sinnen, 2007), with tasks taken in decreasing bottom level
(T. L. Adam, K. M. Chandy and J. R. Dickson, A comparison of list schedules for parallel processing systems, CACM, 1974;
the upward rank of HEFT, H. Topcuoglu, S. Hariri and M.-Y. Wu, IEEE TPDS, 2002).

1. **Bottom levels.** One pass in reverse topological order: a task's bottom level is its cost plus the largest, over
   its out-edges, of the edge's cost plus its head's bottom level. Costs are the node's and the link's actual ones,
   since each task's node is fixed.
2. **The list.** Repeatedly take the ready task with the largest bottom level, ties broken by task index so every rank
   decides alike.
3. **Edge scheduling.** When a task is taken, each of its incoming communications is placed on each link of its route,
   in route order, at the end of that link's sequence and no earlier than its data is available (the end technique,
   Sinnen and Sousa, 2005). The task then starts once its last input has arrived and its node's previous task has
   finished.
4. **The output.** For each link and direction, its communications in the order they were placed. That is the schedule.

What this gives the two kinds, with no rule written for either:
- **Kind A.** A join's incoming pages are placed together when the join is taken, so an operand's pages finish together,
  as a coflow schedule wants (M. Chowdhury and I. Stoica, Coflow, HotNets 2012).
- **Kind B.** Each per-page consumer is taken by its own bottom level, so its page goes when it is most urgent, and
  pages of different consumers interleave.

**Cost.** One pass for the levels, O(V + E); the list with a binary heap, O(V log V); each edge placed once on each link
of its route. All of it at compile time, once a plan, and nothing in the run's length.

**Every rank computes the same schedule.** Its inputs (the declared transfers, the dependences, the map at the plan's
serial) are the same at every rank, and every step is deterministic with fixed tie-breaks. That is the state-machine
approach: replicas apply the same deterministic function to the same inputs and decide alike without exchanging anything
(L. Lamport, Time, clocks, and the ordering of events in a distributed system, CACM, 1978; F. B. Schneider, Implementing
fault-tolerant services using the state machine approach, ACM Comput. Surv., 1990). Nothing is asked at run time: the
place and the order of every landing are already known at both ends, as data-model §5 says ("It asks no peer to choose a
place or grant permission between steps").

## 5. What the scheduler writes down: names, frames and sequences

**Names and frames: the page table.** Every page has a name, a row, and the page table binds it to a frame (T. Kilburn,
D. B. G. Edwards, M. J. Lanigan and F. H. Sumner, One-level storage system, IRE Trans. Electronic Computers, 1962; P. J.
Denning, Virtual memory, ACM Comput. Surv., 1970; data-model §6). This is the region's existing page table.
- Binding is static. Every shape is compiled, so the allocator chooses the frames at compile time, as a register
  allocator does (G. J. Chaitin, Register allocation and spilling via graph coloring, SIGPLAN 1982). This is the
  allocator C3 already builds: happens-before interference, greedy by size, u copies by modulo variable expansion.
- Every value is one run of frames, so a kind-A function reads its operand as one strided array. A kind-B function is
  indifferent to this.
- Renaming at run time, with a free list and a map table, is the method for placement decided at run time (R. M.
  Tomasulo, IBM J. R&D, 1967; K. C. Yeager, The MIPS R10000 superscalar microprocessor, IEEE Micro, 1996). No program
  decides placement at run time, so none is built here.

**The sequences.** The plan writes, for each adjacent node, direction and copy, that link's scheduled communications in
schedule order. Each is one entry of the plan's list (struct mesh_hop), and its fields are only these:
- `page`: its frame, at this end;
- `bytes`: its bytes, at most one page (data-model §3: "It is not a rule that all bytes in a page must be copied").
  `bytes` replaces the hop's `padding`;
- `word` and `field`: what releases a send (its range's supplied word, or at a relay its receive's phase), or what a
  receive's landing writes (its phase field);
- `count`: a receive's count word, where this node consumes the range;
- `node`, `direction` and `pages` as now.

An entry carries no kind, no priority and no group. Its order is its place in the schedule.

## 6. What consumes it: the bridge

The bridge is an opaque driver that consumes queues of page-to-page sends by view. For each adjacent node and direction,
it consumes the sequence of §5. Its rules:

1. **Release and landing.** A send goes once its word passes: its supplied word reads at least t + 1, or its relay's
   phase reads its round. A landing XORs its phase, adds to its count and wakes. All of this is C3's, unchanged.
2. **One naming per landing.** A received page is named in a new landing only once its last one has landed (D.3.5), u
   rounds ahead at most, in sequence order.
3. **The bridge's own lines.** The bridge spreads each sequence over the lines its devices give it on each run, as now:
   each send, in sequence order, goes to the line on which its cumulative bytes over that line's rate finish earliest,
   ties broken by cable and line index. This is the earliest-finish rule of list scheduling on machines of different
   speeds (J. W. S. Liu and C. L. Liu, Bounds on scheduling algorithms for heterogeneous computing systems, IFIP
   Congress, 1974). The count of lines is whatever both ends' devices report, the smaller of the two, on macOS 26 and
   macOS 27 alike, by the failport's rule (tag pre-return-20261103, rdma/mesh-verbs.h). The receiver runs the same
   deterministic deal, so it knows each line's order without being told (the state-machine approach of §4).
4. **No type is read.** The bridge never learns whether a sequence came from a join or from independent pages. Both are
   page-to-page sends.

Nothing else travels: no header, no request, no identity. A send that is ready waits only behind earlier sends of its
own line. When run-time timing departs from the schedule's estimates, that is where it shows, and it is the estimate's
error, not a mechanism.

## 7. Several programs at once (D.3.1, replaced)

- **Attach takes nothing.** There is no client word, no compare-and-swap and no EADDRINUSE. A client is named only in
  the heads of its claims (its pid and start), and only once it plans.
- **Each plan in its own claim.** A plan's list and its state section are placed by its allocator in its own pages. The
  region's header holds no plan. One rising word, `plans`, is raised and woken whenever any plan is written or retired,
  and the bridge finds the standing plans from the claims' heads.
- **The plan's name.** A plan is one program's collective, named (group, epoch). The group is a 64-bit value that the
  launcher passes to every rank, as NCCL's `ncclUniqueId` names a communicator. The epoch is the client's count of its
  own mesh_plan calls, the same at every rank because the call is collective.
- **Plans on one link are fair-queued.** Programs are scheduled apart, so their sequences toward one adjacent node are
  merged by deficit round robin, one page's bytes per turn, plans taken in (group, epoch) order, at both ends alike (M.
  Shreedhar and G. Varghese, Efficient fair queuing using deficit round robin, SIGCOMM 1995; A. Demers, S. Keshav and S.
  Shenker, Analysis and simulation of a fair queueing algorithm, SIGCOMM 1989). A standing plan's sequence is never
  empty, since its next send always exists whether or not it is released. So deficit round robin visits every plan in
  turn, the merge depends on position alone, and the receiver computes it too. Its cost is stated plainly: on a shared
  link, one program's send that isn't released holds the sends behind it until its release comes or its plan is
  cancelled at its deadline.
- **The set of plans on a link changes only at quiescence.** The two bridges agree on the new set by the plans' names
  over their control socket, and switch once nothing of either set is outstanding on the link (J. Kramer and J. Magee,
  The evolving philosophers problem: dynamic change management, IEEE TSE, 1990). Reaching quiescence passivates the
  link: no new sends start, and those in progress complete. A cancelled plan's sends in progress complete with their
  pages as they stand, and their landings move no word, since that plan has ended at both ends.
- **Holders and retirement.** The heartbeat's holders are the clients the live claims name. A gone client's claims are
  retired on the bridge's turn, and their pages are released once no port holds them, by deferred reclamation as now
  (C3.4). A port's load announces itself in the bridge's own memory, fences and re-reads (D10). Such an announcement has
  to live in memory reclamation can't reuse, as a hazard pointer does (M. M. Michael, Hazard pointers, IEEE TPDS, 2004).

## 8. Losses and the map

- A loss cancels, as C3.1 has it: an arrival on the count, no phase written.
- A plan stands while every cable and node it uses is in the map. A map change that removes one of them cancels the plan
  at every node (D.3.6). A change that only adds (a cable pairing late, a node joining) leaves standing plans standing,
  and the next plan uses the new map. A route stays valid while its links exist, and link-state routing recomputes
  routes for new traffic without breaking what still forwards (RFC 2328, §16). This narrows D.3.6 and answers the open
  question the D.3.2–D.3.6 unit left.

## 9. What stands, and what goes

**Stands** (design/disjoint-pages-change-request.md, as implemented through mesh 1fd0cca0 and metal-microbench
3135184e):
- the four rules of C3;
- the allocator, MVE and u (C3.2, C3.3);
- the words: Gray phases, supplied words, counts and status (C3.7);
- the masked first pass, its compaction, the indirect later pass and the queue's event waits (C3.1, D.3.4);
- one naming per landing (D.3.5);
- host inputs in copies ordered by events (D.3.3);
- the delegate's buffers placed (D.3.2).

**Goes:**
- D.5 entirely: chains (Dilworth), priorities carried in entries, placement tables, run-time maps, the GPU's placed pass
  and segment headers. The schedule's order does what chains were meant to do, and static binding needs no table beyond
  the page table.
- D.2's "page lists drained apart" as a model of order, and D.3.1 as written.
- design/plan-bridge-coordination.md.
- The plan's list in walk (demand) order: it becomes schedule order (§5).
- One client per region, the header's one plan, and a map change that only adds cancelling a plan.

## 10. The changes, by file

**mesh**
- **rdma/mesh-graph.{h,c}:**
  - `mesh_graph_bottom_levels`: §4 step 1.
  - `mesh_graph_list_schedule_contention`: §4 steps 2 to 4. Its input is the task graph, each task's node, each edge's
    route and costs. Its output is each link-direction's communications in placement order.
  - Both are graph computations, so they live here (C9).
- **rdma/mesh-collective.{h,c}:**
  - `struct mesh_declared` gains `whole`, and `struct mesh_dependence` gains `nanoseconds`;
  - the task graph is built from the template (§3);
  - the list is written in schedule order;
  - each entry's `bytes` is written.
- **rdma/mesh.h:**
  - `struct mesh_hop`'s `padding` becomes `bytes`;
  - the header's one-plan fields are made vacant;
  - `plans` is appended;
  - each plan is in its claim (§7);
  - MESH_VERSION moves.
- **rdma/mesh-dataflow.{h,c} and mesh-call.{h,c}:**
  - attach takes nothing;
  - claims carry the client;
  - `mesh_plan` takes the group;
  - `mesh_retire` closes a client's claims.
- **rdma/mesh-flow.c:**
  - each send moves `bytes`, and a landing checks them;
  - one flow per (port, standing plan);
  - the deficit-round-robin merge, and the quiescent switch (§7);
  - D.3.6 narrowed (§8).
- **rdma/mesh-records.c, mesh-link.c, mesh-stat.c, and mesh-heartbeat.c's holders read:** §7's holders and retirement.
- **rdma/mesh-ports.c:** the two bridges' agreement on a link's set of plans, by name, over the control socket.

**metal-microbench**
- **tools/prepare_e2b.py:**
  - each declared transfer's `whole`: 1 where its consumer is a delegate kernel reading the range as one run, 0 where it
    consumes per page;
  - each dependence's `nanoseconds` by the roofline, from the captured kernels between its two events.
- **tools/e2b_replay.m:**
  - passes `whole`, `nanoseconds` and the group (`MESH_GROUP`) to `mesh_plan`.
- **tools/mesh/run_pair.py:**
  - draws one 64-bit group per launch and passes it to every rank as `MESH_GROUP`.

The lifecycle files mesh-children.c, mesh-child.c and mesh-lanes.c are unchanged.

## 11. What is left to choose

Only these, each with no effect on correctness:
- **Better cost estimates** than the roofline, where a program has them. The schedule only improves with them.
- **The fair-queueing quantum** across programs: one page, fixed above. A larger quantum trades fairness for fewer
  turns.

## 12. What judges it

- C3.8's judges, and these:
  - every link-direction's sequence respects the task graph's order;
  - every rank writes the same sequences for the same inputs;
  - no entry, word or message carries a kind, a priority or an identity;
  - a second program attaches and plans while the first runs.
- The three builds are clean.
- Parity, the reuse and the timings come from runs, when the operator authorizes them.
