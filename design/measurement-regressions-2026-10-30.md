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

`rdma/rollout.py` is now a local observer. It stamps each completion as soon as it is seen (a 50 µs pause between
looks), and times it from its start or from the completion queued ahead of it, whichever is later. One queue runs one
call at a time, so a call submitted behind another starts when that one ends. The time spent behind earlier calls is
reported as `queued_seconds`. Each rank emits its own rows. There is no store, no peer and no aggregation; the hourly
already keeps every rank's rows. Callers: torch_mesh's bandwidth benchmark and configuration, and examples/crossing.py,
engines.py and memory-read.py.

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
