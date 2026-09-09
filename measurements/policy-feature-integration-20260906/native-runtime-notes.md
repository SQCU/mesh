# Native feature and continuation validation

`native-runtime.json` is the final passing local run after the event-boundary
repairs, bounded receive scheduling, and matrix policy version 19's input-FFN
prenormalization. The previous 20-check result is retained as
`native-runtime-before-prenorm.json`.
The harness uses the real dedicated engine, payload, runningman cart paths and
navigation realization, with an isolated shared-memory region and Unix datagram
transport. It does not use verbs or operate the ordinary mesh bridge.

All 20 checks passed. The terminal-win policy completed 15 optimizer updates and
20 attributed actor rows, with zero unmatched executions and zero native view
faults. The match retained 1,336 native event rows. Across captured policy inputs
there were 8,562 event-neighborhood edges, including 4,617 with positive age, and
71,591 navigation-neighborhood edges. Every neighborhood parameter group changed
from its initial checkpoint, including the learned time decay and Gram metric.
The report includes five completed J measurement revisions.

The harness compares native OBSERVATION/CART/TEAM arrays with the actual assembled
policy frame before journaling, preserving all 83/18/7 columns. Native events
retain all 17 columns. Persisted replay frames are checked against the exact
captured arrays, and saved observation history is checked against captured
contexts. This establishes runtime data delivery and optimizer connectivity;
it does not establish playing strength.

Boundary injection delivers the first real EVENT frame after its snapshot
watermark. The event frame at tick 1 was consumed at tick 2. A later snapshot is
presented without state owners; its 24 event rows survive into the next policy
input as part of an exact 47-row prefix. The engine's shared state is not changed
by these input-delivery injections.

Graceful restart, forced process exit after a durable action journal append,
journal replay, terminal checkpoint restart and engine restart with durable
outcome replay all passed. Prefix comparisons cover retained observation memory,
and queued event-frame identities survive checkpoints and journal replay without
duplicating already consumed events. Realized-event reporting is bounded by each
new event batch rather than re-emitting the retained history.

The subsequently removed observation-memory suite checked twelve complete EVENT frames with
snapshot capacity two, late first arrival, retransmission deduplication,
identical payloads at distinct ticks/request IDs, pending-frame checkpointing,
journal consumption, episode boundaries and legacy inference recovery. A legacy
runstate without observation history either recovers its latest available native
frame or continues with an explicit unavailable-history report. Its incomplete
coverage marker persists through later saves until a new episode begins.

The earlier artifacts preserve failures rather than replacing their evidence:

- `native-runtime-before-default-fix.json` and the associated responder logs
  reproduce the MLX compiled-default comparator failure. The default policy's
  scalar constant output triggered the compiler failure; its tested replacement
  remains compiled.
- `native-runtime-before-monitor-fix.json` passed the runtime feature and recovery
  checks but exposed a harness check that mistook an intentionally stopped worker
  for a failure during the engine-only replay stage.
- `native-runtime-before-boundary-fix.json` passed the original 18 checks before
  adding late-event and ownerless-snapshot injection.

Long evidence-directory paths also exposed macOS Unix socket pathname truncation.
The harness now places only its transient sockets under a short `/tmp` directory.
All owned engine, responder and isolated transport processes exited at completion;
port 26510 was released. The final run used one local engine with four bots and
one responder with scale rank 16, hidden width 32 and two experts. It performed
no Mini work and added no `/mesh0` traffic while another session worked on the
distributed backend.

These reports retain historical observations. The operator subsequently removed
the test suites and verification harnesses; their assertions are not a policy
specification or a current reproduction procedure.
