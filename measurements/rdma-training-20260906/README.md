# Actual RDMA training and receive-buffer RCA

The laptop engine and Mini learners exchanged game observations and policy rates
through `/mesh0` and the resident Thunderbolt RDMA bridges. This run used the
production `Mesh` client, four teams, eight bots, two policies and dimensions
128/341/8/top-2. It used no socket transport replacement and opened no additional
verbs context.

## Successful window

[verification.json](verification.json) records:

| Measurement | Result |
| --- | ---: |
| Matrix-fusion optimizer updates | 16 |
| Terminal-win optimizer updates | 16 |
| Credited actor rows per policy | 56 |
| Matched applied player vectors | 112 |
| Unmatched applied vectors | 0 |
| Completed episodes per learner | 1 |
| Truncated episodes / observed VM faults | 0 / 0 |
| New bridge errors on either host | 0 |

Both learners resumed from the authoritative bundle after a normal TERM restart,
including pending returns and action history. A controlled engine tie reached both
learners. Bridge PIDs 83920 and 353 and both registered capacities stayed unchanged
during this successful run. During the training sample, the application owner PIDs
matched the test's engine and learner. Counter deltas were approximately 249,443
outbound laptop pages and 33,194 outbound Mini pages. These are bridge observation
window counts; they are not an exclusive bandwidth or performance benchmark.

The established port 8795 `/api/policy` followed the new run and retained both
16-update, one-completed-episode records after the test ended. `/api/j` retained
measured geometry with no reader/report errors. The old deployed viewer does not
yet expose the new `learning` field in its J response; the current source does.
The verification game and learners were stopped after the check. The resident
bridges remain available.

## Why the socket test missed the failure

`rdma/mesh.py::Mesh.read` fills a reusable 1,024-row NumPy buffer and yields views
into it. The responder called `incoming.extend(mesh.read(...))`, advancing through
every batch before parsing those views. After the first batch, earlier list
elements referred to overwritten bytes. Large STATE messages therefore could not
complete reassembly. The socket fixture allocated independent packet bytes and
did not reproduce that lifetime contract.

The responder now feeds `RuntimeFrames` directly while advancing the native
iterator. Queued asynchronous packets already own copies. The added regression
uses the actual `Mesh.read` method with controlled native batches: collecting
1,541 frames loses the complete message, while immediate parsing reproduces every
word of the 2,000-by-778 tensor. Twelve receive/recovery/attribution checks pass on
each machine. See [laptop checks](checks.log) and [Mini checks](mini-checks.log).
Status now reports per-kind pending snapshots and per-message received/expected
frame counts, rather than just an unexplained pending total.

## Interrupted attempts and attribution limits

The first attempt assembled two snapshots, then stopped advancing. Its state and
logs are retained under `initial/`. The old laptop bridge heartbeat subsequently
stopped and its release-ring tail became inconsistent with the head. Before the
second attempt started, its heartbeat was already 94 seconds old and release tail
was 49,635,904,629 against head 29,486,868.

The test operator also mistakenly launched a second harness before the first
finished, creating competing application consumers. Both were stopped with TERM.
That overlap was wrong; the earlier bridge stall and corrupt ring observation
preceded it. The bridge PID changed from 65814 to 83920 and its launch plist was
rewritten during the window. This verifier does not restart bridges; the cause and
external actor of that replacement are unestablished here. No kernel cause is
inferred from these observations.

The harness now serializes itself with a process-lifetime lock and waits for both
bridges to be paired, responsive and free of an active application before attaching.
This prevents a competing SPSC-ring consumer from demoting a working application;
it does not gate capability by machine class. Other tools still need coordinated
ownership of the shared application rings. After the successful test released
them, other application PIDs appeared in the final sample.

`initial/` and `overlap/` preserve both unsuccessful reports. The old laptop flight
tail and observed bridge replacement are evidence of an unresolved interruption,
not proof that the repaired Python aliasing bug caused that interruption. The
successful run establishes actual application training and restart over RDMA;
kernel/bridge interruption recovery and learned policy strength need separate
measurements.

These are historical measurements. The verification harness was removed at the
operator's instruction on September 6. `source-sha256.json` records the measured
implementation without changing any branch-based installation policy.
