# Change request: what the plan gives the bridge

This change request states what the code must do after the operator's turns of 2026-11-04, since
design/disjoint-pages-change-request.md Appendix C, the last change request the operator endorsed in its writing and in
its results. Each requirement quotes the operator, then names the method the code uses: what it is, and where it comes
from in the literature. The method is used as its source states it.

It replaces design/transfer-schedule-change-request.md, which is removed, and design/disjoint-pages-change-request.md
D.2's model of a transfer, D.3.1 and D.5. Everything else in that change request stands as implemented (mesh 1fd0cca0,
metal-microbench 3135184e). How a program becomes this plan is design/aot-compilation-change-request.md.

## R1. Transfers are pages by view, from queues of desired transfers

> the only way to use the dma drivers is to submit work to a queue by view, (reference), not by copy or by value.

> scheduled transfers which happen from an opaque driver consuming queues of desired transfers.

- The code must move every page from the page that holds it to the page it lands in, with nothing copied on the way.
- **Method:** user-level, zero-copy communication through queues of descriptors that name registered memory. U-Net
  introduced it (T. von Eicken, A. Basu, V. Buch and W. Vogels, U-Net: a user-level network interface for parallel and
  distributed computing, SOSP 1995). The Virtual Interface Architecture standardized it (D. Dunning et al., The Virtual
  Interface Architecture, IEEE Micro, 1998). The region is registered once (data-model §2), and the receiver fixes where
  each page lands before it comes (data-model §3).
- **Code:** as now. The plan's lists name pages, and the bridge's records name the same pages.

## R2. Every page a program uses is placed by the allocator and named by the page table

> however, there is a memory allocation model called a 'page table' that the user keeps expecting you to use?

> do programs do nothing but what the allocator lets them, with the shapes and coarse bin packing requierments of
> allocation known aot?

- The code must give every value its pages at compile time and name each page by its row in the page table.
- **Method:**
  - The page table: names bound to frames (T. Kilburn, D. B. G. Edwards, M. J. Lanigan and F. H. Sumner, One-level
    storage system, IRE Trans. Electronic Computers, 1962; P. J. Denning, Virtual memory, ACM Comput. Surv., 1970).
  - Binding is static. The allocator is C3's: colouring greedy by size (Y. Pisarchyk and J. Lee, Efficient memory
    management for deep neural net inference, arXiv 2001.03288, 2020), with copies by modulo variable expansion (M. Lam,
    Software pipelining, PLDI 1988).
- **Code:** as now (C3, D.3.2).

## R3. An execution's transfers are a set under a dependency order

> page transfers have dependency orderings (true) and per node placement (true)... but both of these relationships do
> not imply total orderings or local orderings determined aot.

> the use of cudagraphs style 'shapes are compiled but indexes inside of shapes might not need to be'

- The code must represent one execution as its functions and its page transfers under the order their data imposes, and
  nothing more.
- **Method:** the task graph. It is a directed acyclic graph whose vertices are tasks with computation costs and whose
  edges are data with communication costs (Y.-K. Kwok and I. Ahmad, Static scheduling algorithms for allocating directed
  task graphs to multiprocessors, ACM Comput. Surv., 1999; O. Sinnen, Task Scheduling for Parallel Systems, Wiley,
  2007).
  - A function is a task at its node.
  - Each page's hop over a cable is an edge, with cost its bytes over the cable's bandwidth, from the map.
  - The program's functions are its dependences at one node (C3's `struct mesh_dependence`, from a consumption to a
    production). Each gains its cost, `nanoseconds`.
- **Code:**
  - rdma/mesh-collective.{h,c}: `struct mesh_dependence` gains `nanoseconds`; the plan builds the task graph.
  - tools/prepare_e2b.py: writes each dependence's cost by the roofline, the larger of its work's flops over the node's
    peak and its bytes over the node's memory bandwidth (S. Williams, A. Waterman and D. Patterson, Roofline: an
    insightful visual performance model for multicore architectures, CACM, 2009). It uses data-model §1's measured rate.
  - tools/e2b_replay.m: passes the costs.

## R4. A function that needs its operand whole and one that takes pages in any order are both scheduled; the bridge sees only sends

> function A requires set of pages operand_a to be contiguous and complete before function A can begin to execute.
> function B requires set of pages operand_b to be complete but not contiguous before function B is complete. function B
> can work on any pages received in any order and can also release incomplete pages to consumer functions.

> the fact that one pagelist transmitted thrtough the dma driver came from a coflow vs an independent flow is *not
> something the bridge cares about*; both are page -> page sends with no logic related to this typing.

- The code must schedule both kinds, and must not tell the bridge which is which.
- **Method:** the task graph's granularity (Sinnen, 2007).
  - Function A is one task with an edge from every page of its operand.
  - Function B is one task per page, each with an edge from its own page and its share of the cost.
  - A's contiguity is placement: each value is one run of pages (R2).
- **Code:**
  - rdma/mesh-collective.{h,c}: each dependence gains `whole`, 1 for function A and 0 for function B.
  - tools/prepare_e2b.py: writes `whole` from the consuming kernel.

## R5. The order of transfers is a static schedule with costs

> uh, yes, that is the subtext and also the literal textual goal of what we're working on here with this subinterface

(of "static scheduling with costs").

> ultimately a 'scheduling objective' here was so fundamental and central that the choice of representation and
> serialization was totally descendant from what the scheduler 'cares about' (we don't care about the scheduler being
> globally optimal btw; any approximatino measuring well in literature these properties of obvious concern is fine).

- The code must order each cable's sends by a static schedule of the task graph that minimizes the execution's makespan.
- **Method:** list scheduling (R. L. Graham, Bounds on multiprocessing timing anomalies, SIAM J. Appl. Math., 1969).
  - Tasks are taken in decreasing upward rank, HEFT's priority (H. Topcuoglu, S. Hariri and M.-Y. Wu,
    Performance-effective and low-complexity task scheduling for heterogeneous computing, IEEE TPDS, 2002).
  - Each edge is scheduled on the links of its route in a topology graph whose links are the map's cables. This is edge
    scheduling under the contention model (O. Sinnen and L. A. Sousa, Communication contention in task scheduling, IEEE
    TPDS 16(6), 2005).
  - Routes are the plan's existing ones: trees packed by Garg and Könemann's rounds (N. Garg and J. Könemann, Faster and
    simpler algorithms for multicommodity flow and other fractional packing problems, FOCS 1998; rdma/mesh-graph.h
    `mesh_graph_pack`).
  - Successive executions follow in order, each on its own copy (C3.3).
  - Every rank computes the same schedule from the same inputs, so no rank asks another anything at run time (F. B.
    Schneider, Implementing fault-tolerant services using the state machine approach, ACM Comput. Surv., 1990).
- **Code:**
  - rdma/mesh-graph.{h,c}: the upward ranks and the list schedule with edge scheduling.
  - rdma/mesh-collective.c: writes each copy's list with each cable's sends in schedule order, each with its bytes.
    `struct mesh_hop`'s `padding` becomes `bytes`.
  - rdma/mesh-flow.c: sends `bytes`.

## R6. Who sends what to whom is compiled; placement decided at run time travels ahead

> think about whether you can compile a schedule which nonblockinguinely transmits metadata about a candidate page map
> which will be used soon; if your code is compiled aot why can't you push this one-sided before and without blocking
> any one-sided transactions?

> okay sounds good lets build this.

(of "A compiled plan answers that once, at compile time. Only run-time placement is left, and it travels as a map.")

- **Method:** the metadata problem of dynamic sparse data exchange (T. Hoefler, C. Siebert and A. Lumsdaine, Scalable
  communication protocols for dynamic sparse data exchange, PPoPP 2010) is answered at compile time: every rank computes
  every transfer of the plan (R5).
  - Where a program decides placement at run time, the map is a value of the plan.
  - It is produced at a lag ahead of the transfers it names, with its own copies by modulo variable expansion (Lam,
    1988).
  - It lands one-sided before them.
- **Code:** none in this change. No program decides placement at run time, and e2b's placement is static (R2).

## R7. Nothing outside the bridge knows the transport

> queue pairs are an abstraction which no ocllective and no data processing code can know about.

> macos26 and macos27 are both required; you may not single-case or misunderstand any part of the collective's drivers
> or interfaces for any reason as any approach to any problem.

- The code must keep every transport object and count inside the bridge. The bridge must take whatever the devices
  report on each run.
- **Method:** the failport's code (tag pre-return-20261103: rdma/mesh-verbs.h, each device's `max_qp` less the
  provider's own, both ends' smaller).
- **Code:** as now (the D.3.2–D.3.6 unit's count audit).

## R8. A program holds only its plan's allocation, and several programs can run at once

> don't make any 'grabby' code which transitively or terminally 'grabs' resources it is not using

> is attaching conceptually real or do programs do nothing but what the allocator lets them

- The code must hold nothing for a program but its plan's claim. Attaching holds nothing, and there is no client word
  and no EADDRINUSE.
- **Method, pages:** each plan claims exactly its pages and releases them on retirement, by deferred reclamation (P. E.
  McKenney and J. D. Slingwine, Read-copy update, PDCS 1998; K. Fraser, Practical lock-freedom, Cambridge, 2004). This
  is C3.4, as now.
- **Method, transfers:** programs that run at once are scheduled as one composite task graph (H. Zhao and R.
  Sakellariou, Scheduling multiple DAGs onto heterogeneous systems, IPDPS 2006). The DAGs are composed into one and
  scheduled with a single list scheduler (R5).
- **Composition:** design/aot-compilation-change-request.md A6. One scheduler per island orders every program's
  computations consistently (Barham et al., Pathways, MLSys 2022, §4.4), and composes their task graphs into one
  (Zhao and Sakellariou, 2006). Which process is the island's scheduler is left there for the operator.
- **Code:**
  - rdma/mesh-dataflow.c: `mesh_attach`'s compare-and-swap and EADDRINUSE go, and attaching only maps the region.
  - The rest of R8's code follows the operator's answer.

## What judges it

- C3.8's judges hold.
- Every cable's sends are in the order of a schedule computed by R5's method, and every rank writes the same order for
  the same inputs.
- The bridge is given pages, bytes and words, and nothing else.
- The three builds are clean. Runs come only when the operator authorizes them.
