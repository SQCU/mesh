# Measurement regressions, 2026-10-30

The hourly's bandwidth case read about 1 GB/s at 64 MiB on five nodes (hourly 20261030T1428Z, mesh 27e41a0d), an
order of magnitude under one 80 Gb/s cable. That figure was produced by the measurement, not by the library. This
records how the measurement regressed, commit by commit, and what the case's own records say instead.

## What the records say

The case submits every collective of every size in one continuous queue, then waits. Its rows' `seconds` grow case
after case in issue order (0.090 s for the first, 3.08 s for the last), nearly independent of the collective or its size,
and they move in steps of 0.045 s. Each case's own interval is the difference between consecutive completions. At
64 MiB, eight broadcasts (537 MB) completed within one step after the case before them, so at least 11.9 GB/s; eight
all-gathers (429 MB), at least 9.5 GB/s. These are lower bounds set by the 45 ms step, and they come from no other
measurement.

## The chain

1. **Continuous submission and final completion.** The earlier form was submit continuously, stamp the end, and add
   nothing to execution. Its name survived in `mesh_statistics.py` as `continuous-host-submission-and-final-completion`
   until d4916d1e.
2. **4bb01da0 (10-24): measurement changed execution.** The bandwidth benchmark issued eager calls. It drained the
   stream (`torch.mps.synchronize`) and met the other ranks through the store (`realize_inputs`) before every case.
   Its per-cable figures came from `ncclMeshCommSample`, which read a per-transfer byte account the library kept on
   its data path (`mesh-byte-account.h`) for that purpose, beside a trace observer (`nccl-mesh-trace.h`). This went
   against design/whole-function.md: observations do not change execution, and nothing is eager.
3. **86928ab3 (10-25): the accounting went, and the per-cable figure with it.** Deleting the runtime byte account and
   the traces was right. But a prepared function's bytes on each cable are fixed when it is compiled, and no
   compile-time figure replaced the runtime one.
4. **d4916d1e (10-25): the clock.** Timing became "host-clock observation of ordinary completion": a case's clock
   started at an asynchronous enqueue and stopped when an observer thread noticed its completion.
   - **Cumulative.** The cases run in one queue, so each case's time included every case still running ahead of it.
   - **Quantized.** The observer noticed completions once a pass, and a pass was a 5 ms idle wait plus the store
     round trips below. That made 45 ms steps.
   - **Narrowed.** The per-call streaming interval, the wire bytes and the numerical error left the report. What
     remained was a cumulative (us, GB/s) pair, which read as a rate.
5. **The store in the stamping loop** (rollout, from d4916d1e through 060c9bf3 and 27847856). Every completed record
   was shipped through torch's c10d TCPStore (`store.set`), and rank 0 polled `store.check` and `store.get` for every
   peer, in the same loop that stamped completions. A control-plane TCP round trip clocked the data plane's
   measurement. NCCL's model keeps TCP to the out-of-band bootstrap, once, before the communicator exists
   (backend.py's `ncclUniqueId` exchange, as ProcessGroupNCCL does), and the library has its own configuration channel
   (`ncclMeshCommConfigGather`). A measurement's records need neither during execution: each rank's records are in its
   log, and joining ranks is analysis.

## Reported wrongly on the way

- I read the cumulative rows as rates ("about 1 GB/s per rank") and took the five collectives for five ranks.
- I then said the library "is limited somewhere in its own per-step encoding" with no evidence for it.
- I wrote that two runs "measured nothing because every node was absent". The record says the collector got no
  record ("unreachable, or its job did not start"). Its one present node is the workstation, which is not a member.
  That made a gap in the observer's knowledge into a property of the machines.

## The observer

There is no observer thread. `rdma/rollout.py` stamped each completion when a polling thread noticed it, and 78895bdb
set that poll to 5 ms, so every interval was a whole number of polls (about 7.5 ms with the sleep's overrun): at 64 MiB
and 8 calls, every reported time sat on a 939 us grid. It is deleted. Each completion now carries its own end, the GPU's
end of its last part as Metal 4 reports it (`MTL4CommitFeedback` `GPUEndTime`, kept by the schedule, converted to the
statistics clock), read once the schedule's last part arrives (`MetalScheduleEnd`, woken by that arrival, never
polled). `mesh_statistics.Observations` emits each record in submission order: from its submission or the end of the
completion ahead of it, whichever is later, to its own end; the time behind earlier calls is `queued_seconds`. `put`
emits those already ended without waiting, `close` the rest. Callers: torch_mesh's bandwidth benchmark and
configuration (closed by the group's shutdown), and examples/crossing.py, engines.py and memory-read.py.

## Still owed

- **The per-cable figure, from the compiled plan.** A declared group's bytes on each channel (each a cable) are the sum
  of its compiled messages, and are known at declaration. The figure is those bytes times the calls, against the
  cable's `bits_per_second` and the case's own interval. That gives link utilization without runtime accounting.
- **Configuration exchanges.** torch_mesh's configuration-time exchanges through the store (`backend.py`'s collective
  view gather, configuration.py's capacities and proposal) duplicate `ncclMeshCommConfigGather`, and should fold into it.
  Done in 9eb3f16d: `ProcessGroupMesh.exchange_configuration` gathers over it; the store keeps the `ncclUniqueId`.

## Why collectives used about one cable's worth

The fabric is 13 TB5 cables (metal-microbench docs/facts.md, 2026-10-30): 60, 60, 60, 50 and 30 GB/s into m5ult1,
m5ult3, m5ult4, m5ult2 and the Mini. Each layer was traced on a 64 MiB all-gather and all-reduce over the five nodes.
- **Spreads.** Packed trees for every root (nccl-mesh.c comm_routes, mesh_trees_pack), a hop's capacity the sum of
  its parallel cables. Each tree edge's bytes are split over those cables by bandwidth (comm->composed,
  mesh_write_compile). The bridge runs one child process and progress thread per port.
- **Serialized.** cc1ece4b (10-28) cut torch_mesh's lowering (collective_views.lower) at every run's ends, the
  binding's own domain ends included. A rank's source run is its own segment, so a contiguous all-gather became five
  calls in one group, one per root. mesh_group_lay lays a group's calls in call order, and each cable is one FIFO
  (mesh-link.c stream_send posts in list order). Every cable carried root 0's segment, then root 1's, and so on, and
  a node received through one root's trees at a time. Before it (673f1399, 10-27), the cuts were only run ends inside
  a binding, and a contiguous all-gather was one call, whose segments the library cuts and interleaves by depth on
  every cable (mesh_program_steps_cut, mesh-routed.c dependency_order).
- **Fixed.** lower cuts only where a rank's storage breaks inside its domain. MetalTensorTraversal fragment:end:
  returns the part of a span its run holds, so an all-gather's source, its own segment, aliases the whole call.

Still serialized, next:
- **A combined SEND is published once its whole landing is consumed** (nccl-mesh.c complete's publications). A
  reduction's forward should go block by block, as its landing chunks already land (TASK_PACED).
- **Every group completes inside its own launch.** Call k + 1's entries and receives exist only after call k's last
  landing, a bubble across the fabric at every call. 014471ee (10-30) deleted the issue-then-complete form. A declared
  repeat's next call can present and open its gate before the previous call's tracks end: two generations of slots
  and guard() already protect the storage.
- **cc8a6f8e (10-17) removed chunked pipelining** with the cost model. The forms (ncclMeshCommForms) carry chunks
  again only where a configuration names them.

## The tracks pool's spinners

After the one-call all-gather ran (f8059d25, 37575833), the bandwidth case on the five nodes still read far under one
cable for the reductions and the all-to-all, and the rows said why: a cliff, not a slope. From 64 KiB to 256 KiB the
all-reduce went from 0.2 ms a call to 25 ms, the reduce-scatter to 5.5 ms and the exchange to 8.3 ms, while the
broadcast and all-gather did not move. The hourly at 27e41a0d (20261030T1428Z) had the all-reduce at 0.26 ms there, and
e576a6f1, 4530374d's parent, run on the five nodes the same afternoon, at 0.29 ms (64 MiB: 5.6 ms). 4530374d itself
stalls on the count-ordering defect f8059d25 fixed, as does every commit up to it, so the range could not be bisected
further; nothing else in it changes what the GPU runs during a call.

- **The cause.** metal_tracks sized the pool by its chunks, up to 4096 threadgroups of 256 threads. A worker that
  finds no ready task scans again at once, a system-scope fence and the loads of every open cursor, so every worker
  past what the GPU holds at once only added spinners beside the ones doing the work. The form 4530374d replaced ran one
  threadgroup a track. The pool's own premise, persistent workers, is a pool the GPU holds whole.
- **d946e1e4.** At most one worker a GPU core (IORegistry gpu-core-count). The 256 KiB all-reduce 25 ms -> 0.35 ms,
  the exchange 8.3 -> 0.18; the 64 MiB all-reduce 92 -> 27 ms.
- **4ed1a7f3.** A claim scanned every cursor, a fence each, for a 1024-element chunk. Thread 0 now fences once a pass
  and looks first at the task it ran last. 64 MiB all-reduce 27 -> 19.7 ms.
- **ace45b89.** A worker is the widest threadgroup the pipeline takes (1024 on the M5), so a chunk is a core's width.
  64 MiB all-reduce 19.7 -> 6.6 ms, reduce-scatter 22.5 -> 6.6; from 256 KiB to 4 MiB the reductions are at or under
  e576a6f1's.

## Link use

The rows are each rank's useful output over an interval the observer stamps once every 5 ms, so a 64 MiB case of 8
calls carries a third of its time in error, and they say nothing about cables. b0a71107 has each port's progress thread
publish its bytes landed with the bridge's clock (mesh-observe.h mesh_link_landed: two slots and a sequence, the writer
never waiting, nothing on the data path reading it), and `mesh-stat landed` watches them from a read-only mapping.
rdma/mesh_landed.py turns the members' watches into each node's inbound use against its ports' stated bandwidth and the
desk's computer-pounds: each member's use times its weight (a Studio 8.0 lb, the Mini 1.6; the workstation is no member).

The bandwidth case at ace45b89/b0a71107 on the five nodes (5 ms windows, the members' 33.6 lb): a peak of 17.8 to 19.5
computer-pounds, every node at 55 to 63% of its inbound and the Mini at 63 to 87%; 13.3 to 13.5 on average over the
windows over 12, the 16 and 64 MiB cases. The Mini's 30 GB/s is the bound: an all-gather gives every node the same bytes,
so with the Mini's cables full the Studios' are about half full, 17.6 computer-pounds. The 64 MiB all-gather is at 63%
of the Mini's bound (1.79 ms), the all-reduce at about 35% of its own (2.3 ms): the serializations above, a combined
SEND published once its whole landing is summed and every call completed in its own launch, are what remains.

## What a call's cables do (a 100 us watch, cd210504)

`mesh-stat landed 100` beside the bandwidth case gives each cable's bytes in 0.5 ms bins (the ports' payload peaks at
8.6 to 8.9 GB/s a cable, docs/facts.md's 8.7, not the stated 10).
- **The all-to-all (9.4 ms a call).** The Mini's three cables each ran at 8.6 GB/s for about 4 ms, then the m5ult2 cable
  alone for 4.5 ms. The library's own routes weight every simple path alike, so more of them cross m5ult2 than the
  Mini's other neighbours, and the desk landed 961 MB a call for 335 MB of useful bytes. A configured program solves its
  routes (assignment.domains); the benchmark never did. 3b976aef routes it as bind does (configuration.routes).
- **The reduce-scatter (6.4 ms, the all-gather's bytes a node in 1.9).** Each call a 2.5 ms burst (m5ult1 up to
  45 GB/s), a 2.5 ms trickle of 1 to 6 GB/s, then about 1 ms with nothing landing anywhere. No node near its cables.
  - **Whole-consumption priors.** d3869552 (10-23) made every receive and every SEND wait for the whole consumption of
    every earlier receive overlapping it, where the form before it named one earlier step. A reduce's children combine
    one whole landing after another. f1a821ab: a combine that follows a combine of the same elements in the same runs
    follows it chunk by chunk (metal_after, the task's prefix of finished chunks).
  - **The sum sent on whole.** 63a3fe3d sent a combining relay's sum on page by page; bd83c006 reverted it for a count the
    bridge could not see (a device atomic), and b6883e1f's fix never came back. 657f68f8, cd210504: each landing
    channel's count of units summed, stored system-coherent, named by the SEND as a relay names `landed`.
  - **The bubble.** Every call is its own schedule behind a queue barrier (20afa368, 8fa0b005), and its present and
    gate run only after the previous call's pool has ended: about 1 ms a call with no cable carrying anything.

## c56f21a9: every collective at 16 and 64 MiB lands nothing, chain-service ends in SIGBUS (bisected 2026-10-30)

On all five members, examples/bandwidth.py (metal-microbench tools/mesh/hourly/bandwidth.json) and
examples/chain-service.py (chain.json), each commit prepared on every node:

| Commit | chain-service | bandwidth check |
|---|---|---|
| 3f2a90b7 (last pass) | 5 × exit 0 | 45 checked, none wrong, every rank |
| c56f21a9 "Bind WRITE operands through the canonical communicator declaration" (first fail) | 4 × exit 138 (SIGBUS after placement, as the WRITE bindings prepare), 1 × exit 1 (its peers gone: "preparing exact-byte write bindings: Connection reset by peer") | all-reduce, broadcast, all-gather, reduce-scatter and the exchange wrong at 16 MiB and 64 MiB on ranks 1-4; every size to 4 MiB right |
| d217b906, 18709d4d, f21a4d24, 7e7f8726 (c56f21a9 merged) | the same | the same |

What the wrong values say: a broadcast's recipient holds its own input (rank r reads r + 1 where 1 is sent), an
all-gather's and a reduce-scatter's outputs keep their initial -1, an all-reduce is partly reduced: at those sizes the
receives are never copied out. c56f21a9's issue pipeline is the range: every issue laid with its call's `frames` as its
repeats at stride 0 (prepared, issue_layout), the binding period capped at the group's invocations, and a group's
residency and hazards declared as exact byte ranges ordered by the shared page frontier (uses_add, metal_accesses)
instead of whole buffers.

7e7f8726's message names 18709d4d's climb stop as the cause; the bisect above refutes that (d217b906 fails without it).
7e7f8726 stays: it restores mesh_tree_shape's walk for a node its own parent, as it was before 18709d4d.

The existing m5ult4 crash report locates the SIGBUS in `prepared`, inside its
inlined `send_dependencies` span walk. That function still read the frame count
from the fused recording result, which is empty during declaration. It therefore
treated the 128-frame train as one frame and paired a first-frame SEND with a
last-frame receive. The crash register holds the resulting unsigned offset:
`-127 * 16384`. The scan then reaches an unmapped page. Preparation now consumes
the declared `call.frames` and the message's explicit `frame`; no guard or wait
was added. The crash instruction/registers, strict C compilation and prepared
descriptor geometry establish this correction. Cluster execution has not yet
verified recovery.

The large-bandwidth failures have a separate first error still under review.
The saved c56f21a9 rank-0 log ends at a GPU wait deadline during the 16 MiB
all-reduce. Other ranks then report a lost and re-paired link. Their observer
threads raised that failure, but `close()` merely joined the threads, so callers
continued into output comparisons. `close()` now re-raises the observer's
original exception; polling and submission are unchanged. Those later unchanged
outputs are consequences of an aborted run, not independent evidence of a
defective kernel for each later shape. The initiating GPU wait remains unresolved.

The initiating error is located: with only a COPY's units owning SEND cuts, the 16 MiB all-reduce completes and every case checks right on all five, chain-service too (the builds below predate the frame correction above). Main at 51a6ff10, with both, passes the
same on all five.

Localized within c56f21a9 by two builds of main with one part of its mesh_fabric_refine change undone (tags
bisect-refine, bisect-owners, deleted after), each run on all five as above:

| Build | chain-service | bandwidth check |
|---|---|---|
| refinement as before c56f21a9 (COPY-only cut owners, an unrepresentable cut the rank's whole refinement refused) | 5 × exit 0 | 45 checked, none wrong |
| COPY-only cut owners, c56f21a9's fallback kept (an unrepresentable cut keeps that SEND whole) | 5 × exit 0 | 45 checked, none wrong |

The cause is c56f21a9 making every receive, REDUCE included, a cut owner (`next=j`). The fix restores COPY-only
owners and keeps the per-SEND fallback (mesh-plan.h mesh_fabric_refine says why). How a combine-owned cut loses its
data is not yet shown from source; the summed forwarding of a SEND split at a combine's boundary (send_dependencies,
the channel's count of units summed) is where to read next, before any combine is made a cut owner again.
