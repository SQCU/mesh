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
