# Native policy training continuation — September 6, 2026

The actual engine, full payload game code, four Havocbots and an MLX terminal-win
learner were exercised on the laptop and Mini. The adapter emits full-state rates;
the stock comparison team receives zero rates. These are controlled integration
runs, not evidence of learned competitive strength.

[The canonical contract](../../design/TRAINING-CONTINUATION.md) describes ownership,
serialization, attribution and the remaining limits. [summary.json](summary.json)
contains final counts and source/build fingerprints. Fingerprints identify evidence;
no installation or fetch is pinned to them.

## Recovery coverage

Both machines exercised:

- Actual optimizer updates and acknowledged bot applications.
- Graceful mid-round responder restart with pending episode state.
- Abrupt responder exit after a durable action record, followed by journal replay.
- An actual engine tie induced through its score-limit rule, with every transient
  EVENT frame deliberately discarded.
- Checkpointed outcome acknowledgement and terminal restart without double credit
  or silent episode truncation.
- Engine restart and replay of the earlier on-disk outcome, including when a newer
  completed round is also present in the returned ledger.
- Zero unmatched observed executions after recovery.

The [initial Mini reproduction](mini-attribution-rca.json) had 124 unmatched rows:
the telemetry window evicted action sources while transport was still delivering
them. Pending execution history now uses disk-backed retention independent of that
window. Snapshot publication waits for the preceding queued group to drain and
counts coalesced sampling opportunities. This preserves current sampling and game
execution without an expanding FIFO of obsolete states.

Detailed successful reports: [laptop](laptop-runtime.json), [Mini](mini-runtime.json).
The recorded runs preloaded [two earlier actual engine outcomes](replay-prior-outcomes.tsv)
and observed each retained identity after engine restart. The verification harness
was subsequently removed at the operator's instruction.
The `laptop/` and `mini/` directories retain engine, responder, recovery and bridge
logs plus native outcome ledgers. Large replay/checkpoint arrays and J artifacts
remain in the recorded run directories; they are not copied into this evidence set.

## Arithmetic, native execution and reporting

Each machine passed 38 focused checks for gradients, delayed attribution, durable
spilling, pending returns, optimizer identity after resume, late outcomes, snapshot
sessions, covariance factors and viewer handover. The native framing C fixture also
passed complete/partial buffer isolation, cross-session assembly, duplicates and
buffer growth. See [laptop checks](laptop-checks.log) and [Mini checks](mini-checks.log).

Each native engine fixture passed 31 checks, including independent bot views,
head-up/head-down steering, integer references, native movement, packet validity,
invalid-reference recovery, runaway recovery and a failing stock retry under
`-norunaway`. The final native finite difference was 0.003296000000005961 against
analytic 0.0032968090381473303. It differentiates an actual MLX readout parameter
through the wire and native exponential integrator. [Laptop](laptop-native.json)
and [Mini](mini-native.json) reports retain the full engine output.

[Game checks](game-checks.log) cover 39 QC/Python projection cases, six ownership
transitions, six player-driven motion cases and 3,000 unattended ticks retaining
captured cart progress and score accrual. Full engine and payload builds succeeded.

[Viewer HTTP evidence](viewer-http.json) verifies `/j`, `/policy` and both API routes
against the recovered real match. They retain optimizer/terminal measurements when
the producer becomes stale, and J contains observed geometry and application joins.
The JavaScript syntax and repository whitespace checks pass.

The whole-bot microbenchmark measured a median added 11.075 microseconds per warmed
zero-residual invocation on the laptop and 50.575 microseconds on the Mini. Those
separate, paired samples use earlier builds from this validation session, before
the final recovery correction; their exact engine fingerprints are included.
[Laptop timing](laptop-full-bot.json), [Mini timing](mini-full-bot.json). They do not
bound worst-case navigation or establish many-player throughput.

## Operational scope

All transport fixtures used unique shared-memory regions and Unix datagrams, with
no verbs device open. The abrupt exit was injected only into that socket-based
responder. Engines were stopped through their `quit` command. No RDMA bridge,
power policy, persistent learner or persistent game service was restarted.

The current evidence establishes process-interruption recovery. It does not prove
power-loss/disk-failure durability, arbitrary native side-effect containment,
sustained RDMA recovery, learned policy quality, or fleet FLOPS saturation. Full VM
state steering remains available; the fault handler is not a transaction rollback.
