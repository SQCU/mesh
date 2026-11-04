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
  the literature, the state of the code and the functions in scope. Appendix C holds the operator's rulings on the first
  implementation and the revision: masks and a later pass, every value placed, unrolling by lifetime, no claim at attach.
- [design/disjoint-pages-coordination.md](design/disjoint-pages-coordination.md): the coordinating agent's contract
  for that change (the file split, the region's new areas, the plan's write order, the GPU's mask read) and the record
  of each sub-assignment's commits, each review finding and where it went, and what was left open.
