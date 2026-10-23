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
- [THREAT-MODEL.md](THREAT-MODEL.md): the operator's statements on access, quoted verbatim from 2026-08-27. They cover
  nodes that stay reachable and keep working, the false negative as the main class of compromise, and the two kinds of
  version-control ref that code and configuration name: `main` or a named branch.

## The bridge and transport

- [design/data-model.md](design/data-model.md): the hardware preliminaries of the data model: the M-series
  accelerator and its caches beside sm90's, memory to NIC to memory on InfiniBand and here, the Thunderbolt RDMA
  driver's SEND and RECV, and what the chain and the LSTM cost. Section 6 gives what the bridge must have, each
  requirement tied to the section it follows from. Section 7 works out writes over many hops and many paths: the cut
  that bounds a transfer, the DMA a receiving node and a relay spend, and the sizes past which more paths win, as facts
  of the hardware rather than rules the bridge applies; then who balances the load, and the operator's position that
  the bridge does not care how big a write is.
- [design/quiet-link.md](design/quiet-link.md): an adjunct to data-model.md section 3: what a link's first SEND after
  a quiet stretch costs, where mesh runs meet it, and the transfer size above which the cost is under 1%, filed as an
  idiosyncrasy that moves no kernel on the (throughput, latency) frontier.
- [design/transport.md](design/transport.md): the transport as the code holds it: the registered region, the page
  table, links, posted receives, SENDs and readiness words, and who owns each. It explains why SEND and RECV do not
  fail and what happens on a hang, a lost cable or a stopped bridge, and gives the data path (section 14), the source's
  commentary (section 15) and the watchdog (section 16).
- [design/pages-and-functions.md](design/pages-and-functions.md): the memory model in the operator's terms: pages, the
  page table, rows, stamps, use counts and functions. It works one step through on two peers, then on any number of Mac
  minis, with freeing, link errors and waiting on data instead of on messages about data.
- [design/extents-and-liveness.md](design/extents-and-liveness.md): how the mesh knows which pages a collective program
  uses and when each page is free, point by point from the operator's statements of 2026-10-22, with the literature
  behind each. It explains why the bridge assumes no arbitrary access pattern, has no error state and waits on data
  alone.
- [etc/bridge/README.md](etc/bridge/README.md): the bridge's configuration document, its keys, the two didactic
  examples that ship with it and the E2B pair's sized configurations, and the heartbeat's signals that release,
  restart or end the bridge. It explains how each device registers the region in extents sized to its `max_mr`.
- [design/HETEROGENEOUS-LINKS.md](design/HETEROGENEOUS-LINKS.md): links between devices that hold different numbers
  of queue pairs: the failure that showed it (2026-10-21), each end's budget stated in its HELLO, and the plan of lanes
  and streams both ends make from the lesser budget. Section 10 records the keepalive of the versions before 133 and
  its deletion.

## The collective and bindings

- [GOAL.md](GOAL.md): the collective goal: collective operations of the lowest attainable overhead for any program on
  Apple GPUs, on any number of Macs in any cabling. It quotes the operator's statements that set it.
- [rdma/NCCL.md](rdma/NCCL.md): the contract of libnccl-mesh, NCCL's C API over the bridges. It covers building and
  running, communicators, the host and Metal paths, plans, persistent exchanges, memory, routes between unlinked ranks,
  failures and the planner in `mesh-plan.h`, then the NumPy binding (`mesh_mpi.py`) and what is not implemented.
- [design/heterogeneity.md](design/heterogeneity.md): work split over nodes of different speeds and moved over links
  of different costs: torch-mesh's partition operand, the parts `allocate.Balancer` moves from each rank's measured
  stretches, and the literature's requirements R1-R14 beside what the code does for each.
- [design/deliverables.md](design/deliverables.md): the streaming partial-tensor collective library: its definition,
  its invariants with the operator's statements behind them, its public surface and deliverables, and what is not part
  of the library, with the three program types of 2026-10-17.
- [design/algorithm-sources.md](design/algorithm-sources.md): the published prior art behind each part of the Python
  surface (`python/mesh`), its kernels and the transport, under the section names the source cites as anchors. It
  opens with the operator's statements on prior literature and on deleted implementations.
- [design/distributed-reduce.md](design/distributed-reduce.md): the September 2026 design of the reduce over page
  tables (reduce-scatter then all-gather, completion read from the page table, reuse proved by downstream pages), the
  prior art behind each part, and the incident record of the forms that went wrong. Received pages land in the
  receiver's own rows, and the transport exchanges no epoch or session.

## Examples and services

- [design/prepared-machine.md](design/prepared-machine.md): the prepared objects of metal-microbench's LiteRT program
  run on the mesh, each with its address, extent, constructor and use at run time. Its reference baseline is the
  unchanged litert-lm 0.17.1 GPU benchmark.

## Nodes and hardware

- [design/hardware-and-discipline.md](design/hardware-and-discipline.md): the operator's five statements of 2026-10-22
  on the hardware, each with the measured facts, literature and mechanisms behind it. They cover how Apple silicon
  differs from CUDA GPUs, why DMA is fundamental, the level of a streaming multiprocessor, trade-offs that differ from
  NCCL's, and data semantics in the transport.
- [design/ACCESS-TOPOLOGY-SPEC.md](design/ACCESS-TOPOLOGY-SPEC.md): how every node gets the same list of known
  networks (the gitignored `networks.conf`, applied by `bin/mesh-networks.sh`). It treats reachability as a graph of
  Wi-Fi, wired, tailnet, Thunderbolt and point-to-point edges, not one preferred address.
- [README.md](README.md): provisioning a fabric of Macs: the one step that takes a physical visit (`rdma_ctl enable`
  in Recovery OS), bringing up and adding a node, the Thunderbolt fabric's addressing and routing, the repository's
  layout, and how a node's installed branch is read and a new bridge installed.
- [design/claims/ACCESSIBILITY.md](design/claims/ACCESSIBILITY.md): the implementation surfaces behind the
  operator's statements on access (THREAT-MODEL.md): the installers, the network inventory, the bridge's registration
  across pairings, the bridge threads' wait while a link serves nothing, and installation that keeps loaded jobs.
