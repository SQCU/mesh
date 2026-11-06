# Index

This repo makes a set of Macs work as one machine. Each Mac runs a bridge that sends data over Thunderbolt cables by
RDMA. On top of the bridges is a library with the NCCL calls, bindings for torch and NumPy, and a few example programs.
This index lists each document that tells how the system works now, with a short note on each. The repo also holds
other projects, such as a game bot, and old dated records; they are not listed here.

## Start here

- [design/rubric-actionable.md](design/rubric-actionable.md): the short form of the rubrics. It gives the review
  procedure, what crosses each interface between mesh, collective and data code, and one line for each dimension of
  rubrics M, C and D, with that dimension's line in design/rubrics.md.
- [design/rubrics.md](design/rubrics.md): the full rubrics for mesh (M), collective (C) and data (D) code. It defines
  the three kinds of source, how a change is ranked against the form it replaces, and each dimension with its poles and
  the operator's ranked pairs and quotes. Later sections cover the inversions between kinds, the two channels, failure
  across interfaces, strings as messages and how the operator reviews.

## The bridge and transport

- [design/data-model.md](design/data-model.md): the hardware and the paths bytes take through it.
  These facts explain why writes use pages and masks, and why their work is compiled before a call.
  The later sections derive the limits on bridge code, routes, and collectives in plain prose near grades 4-6.

## Change requests

- [design/page-queue-change-request.md](design/page-queue-change-request.md): the operator's statements of 2026-11-03,
  verbatim: collectives of any node count, relays within the Thunderbolt 5 fabric, every transfer interleaved, a page
  queue per node that only the bridge consumes over every link, queue pairs kept inside the bridge, the --link form
  replaced only once its replacement covers it, the failport's process lifecycle reused verbatim. With the state of the
  code and the functions in scope.
- [design/page-queue-coordination.md](design/page-queue-coordination.md): the coordinating agent's notes for that
  change, verbatim: the contract its sub-agents worked to, and the record of each sub-assignment and review finding.
- [design/disjoint-pages-change-request.md](design/disjoint-pages-change-request.md): the operator's and Claude's
  turns of 2026-11-03, verbatim, on the page readiness word and the plan's ring: pages assigned by liveness so nothing
  live shares a page, read-after-write by masks read from a contiguous landed list, no ring and no per-page cells. With
  the literature, the state of the code and the functions in scope. Appendix C holds the review of the first
  implementation, verbatim, and Claude's consolidated revision (C3), the unit of work: masks and a later pass on counted
  completion, every value placed by the one allocator, unrolling by lifetime, no claim at attach, mbarrier-style phases,
  and nothing zeroed or reset. Appendix D holds the review of that revision, verbatim, and Claude's proposal (D.3), the
  unit of work now: any number of programs at once, each plan named by its group; the delegate's buffers placed; host
  inputs in copies ordered by events; Metal 4 submission with the queue's event waits; each received page named once
  per landing; and a plan that stands only while its map does. Its D.2 model, D.3.1 and D.5 are superseded by
  design/plan-to-bridge-change-request.md.
- [design/plan-to-bridge-change-request.md](design/plan-to-bridge-change-request.md): what the code must do after the
  operator's turns of 2026-11-04, requirement by requirement, each quoting the operator and naming its cited method:
  pages by view (U-Net, VIA); the page table, bound statically as C3.3 states; the task graph G = (V, E, w, c);
  functions that need an operand whole and streaming functions that take pages in any order (Graefe; Gibbons; coflows,
  Chowdhury and Stoica); each cable's list from MSCCLang's heap order, by the completion times of one linear program
  that holds precedence, coflows, independent flows and each cable's and device's contention (Chen 2022; Hall, Schulz,
  Shmoys and Wein 1997; Queyranne 1993), solved by HiGHS; compiled metadata (Hoefler, Siebert and Lumsdaine); nothing
  outside the bridge knowing the transport; a program as its function calls. It supersedes D.2's model, D.3.1 and D.5 of
  the disjoint-pages change request.
- [design/aot-compilation-change-request.md](design/aot-compilation-change-request.md): what the ahead-of-time
  compilation must do: whole programs captured ahead of time (`torch.export`, not a `torch.compile` backend), every
  value a list of pages placed anywhere, a typed operand's run a mapping; the placement operand (GSPMD, Alpa) and the weighting operand
  (Sinnen; Garg and Könemann), both chosen elsewhere; collectives over packed trees (Blink) lowered to page
  instructions (MSCCLang); one schedule per function; function executions ordered by one scheduler per island
  (Pathways), which is total order multicast by a fixed sequencer (Défago, Schiper and Urbán: their §3 and FIFO Order
  properties, Fig. 5 as written, §5.2.4's super-group) with one sender a program and every plan created at all its
  nodes before its first message (NCCL's communicator creation), run within each view of the island as a dynamic group
  (Chockler, Keidar and Vitenberg's view properties; the sequencer as Amoeba's membership service; Transis's view
  identifier; block and flush at a view change); the IR operand; the serialized operand in MSCCL-IR's shape, running
  only existing kernels. Its appendix gives each plan its own queue of delivered calls at each node (synchronous
  dataflow's FIFO queue on each arc), merged by sequence number, its room the self-timed buffer bound and feedforward
  queues put on cycles at compile time (Bhattacharyya, Sriram and Lee: Bfb, BBS, Convert-to-SC-graph).
- [design/problems-of-14b9a757-change-request.md](design/problems-of-14b9a757-change-request.md): the twelve problems
  the revisions of 14b9a757 created, each with its code, the published source quoted verbatim and the change. The
  sources are MPI 4.1 §§6.1.3, 6.1.11, 7.7 and 7.8, MPICH's local copies, NCCL's in-place condition, GSPMD §§4.3 and 4.5
  and its Appendix A.2, XLA's CollectivePermute, torch's `view`, `reshape` and `contiguous`, ExecuTorch's memory
  planning, Garey and Johnson's SR2, loop-invariant code motion, Lee and Ha's self-timed schedule, and JGroups' ViewId.
- [design/compilation-coordination.md](design/compilation-coordination.md): the coordinating agent's contract for
  the two change requests above (the file split, the names files share, the task graph and its contention schedule,
  plans in their claims and the island's fixed sequencer, the provisional operands, the sources read) and the record
  of the work, its reviews and what is open; then its part "The appendix's revisions, 2026-11-04" (the change
  request's last appendix, items 1-7: C3.3 without D5, delivery in order without Q4's skip and the hold that stays,
  sets of a function B operand in Threefry-drawn Fisher-Yates order, compiled functions coexisting, torch's MPS
  kernels), with its record; then its part "The documents' revisions of edd4b583" (each cable's list a subsequence of
  MSCCLang's heap-built global topological order by bottom level; delivery by Lamport's message buffer, one sequence
  word a copy, merged at each port (since replaced by design/aot-compilation-change-request.md A10 and appendix item
  2), Q4's hold gone; e2b's partial sums declared function B), with its record.
- [design/disjoint-pages-coordination.md](design/disjoint-pages-coordination.md): the coordinating agent's contract
  for that change (the file split, the region's new areas, the plan's write order, the GPU's mask read) and the record
  of each sub-assignment's commits, each review finding and where it went, and what was left open.
- [design/disjoint-pages-revision-coordination.md](design/disjoint-pages-revision-coordination.md): the coordinating
  agent's contract for that change request's C3 revision and C4 removals (the four rules, the file split, the state
  section with its phases and counts, the hop entry of u copies, the plan call over every value, claims per plan, the
  masked first pass and its later pass held by the command buffer's event wait, and the decisions where C3's text
  cannot be met as written, with their evidence) and the record of each sub-assignment's commits, each review finding
  and where it went, and what was left open.
- [design/disjoint-pages-rulings-coordination.md](design/disjoint-pages-rulings-coordination.md): the coordinating
  agent's contract for that change request's Appendix D, items D.3.2 to D.3.6 (D.3.1, and D.5 added during the work,
  outside it): the file split, the names files share, its decisions with their sources, and the record of the work,
  its reviews and what is open.
