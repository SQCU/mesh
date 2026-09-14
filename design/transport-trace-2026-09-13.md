# Fixed transfer observations

The shared arena contains one trace record for each configured transfer index,
queue and direction. Its allocation is part of `mesh_layout`, registration, and
the existing visible memory preflight. The progress loop allocates no trace
storage and copies no numerical operands. This follows the observation boundary
in [algorithm sources](algorithm-sources.md#performance-evidence).

Each record retains four atomic values:

- `ready_ns`: on send, the first admission to the ready transfer list; on
  receive, the observation immediately before the accepted receive-post call.
- `post_ns`: the host observation after the provider accepted the payload WR.
- `cq_ns`: the host observation when its work completion is processed.
- `occurrences`: the number of occurrences admitted to this record since pairing.

All timestamps use `clock_gettime_nsec_np(CLOCK_UPTIME_RAW)`, the same clock as
native CPU-function observations. They are host timestamps. In particular,
`post_ns` does not measure physical wire start and `cq_ns` does not measure the
exact hardware completion instant. Other machines have independent clock
origins; their absolute timestamps cannot be subtracted to infer link latency.

## Identity and reuse

Setup retains each receiver's explicit transfer index in the sender descriptor.
A second exchange of the completed send descriptors fills each receiver's
corresponding source row/page/index. Both endpoints are therefore available in
the trace without reconstructing identity from queue position during execution.
The receive WR keeps that explicit peer-supplied local index through completion.

The trace describes the latest occurrence, with a cumulative occurrence count;
it is not a historical event stream. Send retries preserve the first admission
time. A rejected index post does not fabricate another occurrence. A rejected
payload receive post leaves the trace unchanged; a successful receive post begins
its record, after the prior receive and its readers have released that target.
This prevents an earlier receive completion from overwriting a newer occurrence's
trace merely because the next descriptor arrived early. Send reader ownership
similarly orders repeated use of the same source transfer.

For finite programs with distinct invocation buffers, each configured descriptor
has its own record. The public `mesh_transfer_trace_count` and
`mesh_transfer_trace` functions enumerate configured records out of band and
return the retained descriptor, queue, direction, timestamps and occurrence count.
An observation during execution is a set of atomic field reads, not a transactional
multi-field snapshot; inspect completed records to compare complete intervals.
Failed WC status remains available through the existing port metadata.

Adding `peer_index` makes each transfer descriptor 40 bytes, allowing 102
descriptors in a 4096-byte index frame with its eight-byte header. Region version
25 identifies this shared layout. The existing conservative WR capacities remain
unchanged.

## Local-only setup

`mesh_realize` publishes the bridge's configured-client field only when the
realized program contains transport bindings. An all-local numerical program
therefore does not initiate network pairing. It uses the same canonical storage
and numerical implementation as its distributed counterpart, and it does not
clear another configured program's existing transport state. This uses the
already known binding count during setup; it adds no numerical readiness check.

## Validation

The bridge, runtime library, algebra library and status tool compile together.
No workload, timing comparison, fault injection, or separate harness was run for
this instrumentation change. Recording intervals enables the existing composed
program to report overlap; compilation itself does not establish overlap.
