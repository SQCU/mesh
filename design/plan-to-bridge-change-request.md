# Change request: what the plan gives the bridge

This change request states what the code must do after the operator's turns of 2026-11-04, since
design/disjoint-pages-change-request.md Appendix C, the last change request the operator endorsed in its writing and in
its results. Each requirement quotes the operator, then names the method the code uses: what it is, and where it comes
from in the literature. The method is used as its source states it.

It replaces design/transfer-schedule-change-request.md, which is removed, and design/disjoint-pages-change-request.md
D.2's model of a transfer, D.3.1 and D.5. Everything else in that change request stands as implemented (mesh 1fd0cca0,
metal-microbench 3135184e). How a program becomes this plan, including the operands it consumes, is
design/aot-compilation-change-request.md.

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

## R2. Every page is placed at compile time and named by the page table

> however, there is a memory allocation model called a 'page table' that the user keeps expecting you to use?

- The code must give every value its pages at compile time and name each page by its row in the page table.
- **Method:** the page table: names bound to frames (T. Kilburn, D. B. G. Edwards, M. J. Lanigan and F. H. Sumner,
  One-level storage system, IRE Trans. Electronic Computers, 1962; P. J. Denning, Virtual memory, ACM Comput. Surv.,
  1970). Binding is static, by C3's allocator. C3.3, verbatim:
  - "For each value, L is the smallest lag at which every read of it in execution t happens before its write in
    execution t + L. The unroll factor u is the largest L."
  - "The template is unrolled into u copies, and execution t runs copy t mod u."
  - "The colouring is greedy by size (Y. Pisarchyk and J. Lee, arXiv 2001.03288, 2020)."
  - "At run time there are no allocations and no bookkeeping; every address is a constant in the plan."
- **Code:** as now (C3, D.3.2).

## R3. An execution is a task graph

> page transfers have dependency orderings (true) and per node placement (true)... but both of these relationships do
> not imply total orderings or local orderings determined aot.

> the use of cudagraphs style 'shapes are compiled but indexes inside of shapes might not need to be'

- The code must represent one execution as a task graph and nothing more.
- **Method:** the task graph G = (V, E, w, c) of the scheduling literature (O. Sinnen, Task Scheduling for Parallel
  Systems, Wiley, 2007; Y.-K. Kwok and I. Ahmad, Static scheduling algorithms for allocating directed task graphs to
  multiprocessors, ACM Comput. Surv., 1999).
  - V are tasks, each a function's execution at its node.
  - E are the data passed between tasks: here, pages.
  - w and c are the tasks' and edges' weights. They are given, not computed by this code: the weighting operand of
    design/aot-compilation-change-request.md A3.
- **Code:** rdma/mesh-collective.{h,c} builds the task graph from the compilation's IR (aot-compilation A8).

## R4. Functions that need an operand whole and functions that take pages in any order

> function A requires set of pages operand_a to be contiguous and complete before function A can begin to execute.
> function B requires set of pages operand_b to be complete but not contiguous before function B is complete. function B
> can work on any pages received in any order and can also release incomplete pages to consumer functions.

> yes, we mentioned a class of functions which are domain:Async,noncontiguous,incomplete,
> codomain:async,noncontigous,incomplete, e.g. that they can stream their outputs to consumers before they have finished
> processing all of the stuff in one buffer wrt one cycle or dependency relationship of one function; these are the
> most-tensor-parallel shardable functions that exist, basicalyl, and are of extreme and particular interest becasue
> tehy can clearly be nonblocking in both execution and transfer, increasing concurrency and also asynchrony of
> execution wrt a specific data dependency relation.

> the fact that one pagelist transmitted thrtough the dma driver came from a coflow vs an independent flow is *not
> something the bridge cares about*; both are page -> page sends with no logic related to this typing.

- The code must schedule both kinds of function, and must not tell the bridge which is which.
- **Method,** as Claude's turn that introduced the two kinds gave it:
  - Function A is a blocking operator over a dense operand: databases' blocking operator, as against a pipelined one (G.
    Graefe, Query evaluation techniques for large databases, ACM Comput. Surv., 1993). Its operand is a coflow, which
    finishes when its last flow does (M. Chowdhury and I. Stoica, Coflow: a networking abstraction for cluster
    applications, HotNets 2012). It is scheduled to minimize coflow completion time (M. Chowdhury, Y. Zhong and I.
    Stoica, Efficient coflow scheduling with Varys, SIGCOMM 2014).
  - Function B is a streaming homomorphism over pages: f(x ∪ y) = f(x) ⊕ f(y), with ⊕ associative and commutative (J.
    Gibbons, The third homomorphism theorem, J. Functional Programming, 1996). Its pages are independent flows, each
    worth having at once. Examples are online softmax (M. Milakov and N. Gimelshein, Online normalizer calculation for
    softmax, arXiv 1805.02867, 2018), FlashAttention over key and value blocks (T. Dao et al., NeurIPS 2022),
    PagedAttention over non-contiguous blocks (W. Kwon et al., SOSP 2023), and reductions' partial sums.
  - **In the task graph (R3):**
    - A is one task with an edge from every page of its operand.
    - B is one task per page, each with an edge from its own page, and its outputs are pages that leave as each is
      produced.
    - Their weights come from the weighting operand.
    - A's contiguity is placement: each value is one run of pages (R2).
- **Code:** the IR carries each consumer's kind (aot-compilation A8). The bridge's lists carry none (R5).

## R5. The order of transfers is a static schedule with costs

> uh, yes, that is the subtext and also the literal textual goal of what we're working on here with this subinterface

(of "static scheduling with costs").

> ultimately a 'scheduling objective' here was so fundamental and central that the choice of representation and
> serialization was totally descendant from what the scheduler 'cares about' (we don't care about the scheduler being
> globally optimal btw; any approximatino measuring well in literature these properties of obvious concern is fine).

- The code must order each cable's sends by a static schedule of one execution's task graph, minimizing its makespan.
- **Method:**
  - List scheduling (R. L. Graham, Bounds on multiprocessing timing anomalies, SIAM J. Appl. Math., 1969), with each
    task already on its node by the placement operand (aot-compilation A2).
  - Tasks are taken in decreasing bottom level (Sinnen, 2007), the priority of HLFET (T. L. Adam, K. M. Chandy and J. R.
    Dickson, A comparison of list schedules for parallel processing systems, CACM, 1974).
  - Each edge is scheduled on the links of its route in a topology graph whose links are the map's cables: edge
    scheduling under the contention model (O. Sinnen and L. A. Sousa, Communication contention in task scheduling, IEEE
    TPDS 16(6), 2005).
- **Routes are the plan's existing ones,** as now:
  - trees packed by Garg and Könemann's rounds (N. Garg and J. Könemann, FOCS 1998; rdma/mesh-graph.h
    `mesh_graph_pack`);
  - each tree's pages assigned by the existing fill (rdma/mesh-collective.c).
- **The order of each cable's list, by the reference algorithm.** Each cable's list is a subsequence of one global
  topological order of the instructions, built as MSCCLang §5.2 states in its step 3: "Sort instructions into a global
  topological order with respect to their dependencies with a heap using the priority to order instructions." The
  priority is the bottom level. Scheduled start times do not order the lists.
  > what algorithm is cited here, if any? no citations? use the reference algorithm instead.
  - A set's pages (design/aot-compilation-change-request.md, appendix item 5) share their predecessors. They take
    consecutive places at the place of the set's first page in that order.
- **Executions:** execution t runs copy t mod u, and every address is a constant in the plan (C3.3, R2). Each cable
  carries execution t's sends in that order, then execution t + 1's.
  - That order is a topological order of the unrolled executions, since every lag points forward.
  - So the cables' sequences cannot wait on each other in a cycle. MSCCLang states this for its thread blocks: "By
    assigning instructions to thread blocks in a topological order that respects communication and processing edges, all
    implicit dependencies introduced by thread block sequential execution cannot produce cycles so that the MSCCL-IR
    does not have deadlocks" (M. Cowan et al., MSCCLang: Microsoft Collective Communication Language, ASPLOS 2023,
    §5.2).
- **Every rank computes the same schedule** from the same inputs, so no rank asks another anything at run time (F. B.
  Schneider, Implementing fault-tolerant services using the state machine approach, ACM Comput. Surv., 1990).
- **How the bridge spreads a cable's sequence over what its devices give it is the bridge's own, as now:**
  rdma/mesh-flow.c `link_prepare`, `mesh_graph_list_schedule` in list order.
- **Code:**
  - rdma/mesh-graph.{h,c}: bottom levels, and the list schedule with edge scheduling.
  - rdma/mesh-collective.c: writes each copy's list with each cable's sends in schedule order, each with its bytes;
    `struct mesh_hop`'s `padding` becomes `bytes`.
  - rdma/mesh-flow.c: sends `bytes`.

## R6. Who sends what to whom is compiled

> okay sounds good lets build this.

(of "A compiled plan answers that once, at compile time. Only run-time placement is left, and it travels as a map.")

- **Method:** the metadata problem of dynamic sparse data exchange (T. Hoefler, C. Siebert and A. Lumsdaine, Scalable
  communication protocols for dynamic sparse data exchange, PPoPP 2010). Its run-time protocols learn who sends to whom.
  Here it is answered at compile time: every rank computes every transfer of the plan (R5).
- **Code:** none in this change. No program decides placement at run time.

## R7. Nothing outside the bridge knows the transport

> queue pairs are an abstraction which no ocllective and no data processing code can know about.

> macos26 and macos27 are both required; you may not single-case or misunderstand any part of the collective's drivers
> or interfaces for any reason as any approach to any problem.

- The code must keep every transport object and count inside the bridge, and the bridge takes whatever the devices
  report on each run.
- **Method:** the failport's code (tag pre-return-20261103: rdma/mesh-verbs.h, each device's `max_qp` less the
  provider's own, both ends' smaller).
- **Code:** as now (the D.3.2–D.3.6 unit's count audit).

## R8. A program is its function calls

> don't make any 'grabby' code which transitively or terminally 'grabs' resources it is not using

> nothing happens when a program joins or leaves because a program is a sequence of like, function calls. if there are
> no function calls happening what do you think you do? if you're a bridge, you do nothing at all since you have no
> inputs to your asynchronous runtime, don't you? if you're a collective function which interacts with a persistent
> bridge, don't you stop existing in the hsot machien because you've halted/returned?

- The code must hold nothing for a program but its plans' claims. Attaching holds nothing, and there is no client word
  and no EADDRINUSE. A bridge with no function execution to serve does nothing.
- **Method:**
  - Pages: each plan claims exactly its pages and releases them on retirement, by deferred reclamation (P. E. McKenney
    and J. D. Slingwine, Read-copy update, PDCS 1998; K. Fraser, Practical lock-freedom, PhD thesis, Cambridge, 2004).
    This is C3.4, as now.
  - Function executions of programs that run at once are ordered consistently by one scheduler for the island, a fixed
    sequencer in the bridge of the island's highest-numbered live node: design/aot-compilation-change-request.md A7
    (P. Barham et al., Pathways, MLSys 2022, §4.4; Défago, Schiper and Urbán, 2004; Garcia-Molina, 1982).
  - Placement and weights reach the plan as operands, passed in by the program from whatever function chooses them
    (aot-compilation A2, A3). Today that is e2b's existing hand partition and unit weights, and any other choosing
    function can replace them.
- **Code:** rdma/mesh-dataflow.c: `mesh_attach`'s compare-and-swap and EADDRINUSE go, and attaching only maps the
  region.

## What judges it

- C3.8's judges hold.
- Every cable's sends follow the schedule of R5's method, and every rank writes the same order for the same inputs.
- The bridge is given pages, bytes and words, and nothing else.
- The three builds are clean. Runs come only when the operator authorizes them.
