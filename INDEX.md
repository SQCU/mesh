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

## The collective and bindings

- [GOAL.md](GOAL.md): the collective goal: collective operations of the lowest attainable overhead for any program on
  Apple GPUs, on any number of Macs in any cabling. It quotes the operator's statements that set it.
- [rdma/NCCL.md](rdma/NCCL.md): the contract of libnccl-mesh, NCCL's C API over the bridges. It covers building and
  running, communicators, the host and Metal paths, plans, persistent exchanges, memory, routes between unlinked ranks,
  failures and the planner in `mesh-plan.h`, then the NumPy binding (`mesh_mpi.py`) and what is not implemented.

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
