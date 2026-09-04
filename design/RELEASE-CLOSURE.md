# Current release closure

Updated 2026-09-04. This is the current implementation/verification/deployment ledger,
not a replacement specification. Requirements remain in `GOAL.md`, `SPECIFICATION.md`,
and the operator quotes. Historical observations in `AGENDA.md` remain historical.

## 1. Preserve the achieved result; repair the availability defects

The operator's correction:

> the project runtime actually demonstrates the mesh computation being used, and in fact being used for dozens of variations of a game-playing team-up-and-take-down high teamcount strategy puzzle, even if the strategy solver was never workign to feature spec...

> basically confirming that the RDMA api actually became usable as something like an importable header which is safely used and incorporated into projects by subagents without patterns of  tampering and meddling that corrupted host or device kernel memory state or repeatedly crashed or softlocked either machine.

— Operator, 2026-09-04, review follow-up.

This is an achieved integration and operating-experience milestone, not a claim that
every solver requirement is complete or that the transport has no remaining defects.
The [August 30 execution record](../xonotic/README.md) separately reports 223 snapshots,
83 responses, 82 online updates, and both bridge bad counters remaining zero.
[R37 in the agenda](AGENDA.md) records large-team gameplay; its no-responder window is evidence
for gameplay scale, not evidence against the separate mesh-integrated runs.
The operator's dozens-of-variants observation must not be reduced to that one window.

| Review defect | Repair on `codex/availability-closure` | Verification / remaining boundary |
|---|---|---|
| Incomplete expert request could wait forever | Request waves replay after a quiet interval while incoming responses and page credits continue to drain. Session + operation IDs identify retries. | Imports and source invariant below. Loss/re-pair execution with this generation remains unmeasured. |
| Arena credits lost on failed post or discarded CQ | Failed posts return ownership immediately; successfully posted arena pages are tracked privately and returned after QP/CQ retirement. | Bridge and client library built on both Macs; public header and ABI unchanged. Live replacement not performed. |
| Allowed HUP skipped teardown; failed MR counted as registered | HUP uses the TERM/INT cleanup path, SIGPIPE becomes a reported I/O failure, and the registered-MR count advances only after success. | Two-host builds. No signals sent to live bridges. |
| Periodic install booted out working services and declared convergence early | No routine daemon bootout or ARD restart; loaded jobs and answering userspace providers are retained. Source and Python generations are staged beside working generations. Revision is written after the run; failures and retained jobs are explicit. | Shell syntax; isolated Python realization and repeat realization. Live installer run and launchd-definition handoff remain pending. |
| Local counterfactual becomes a different model after remote training | **Open measurement defect.** Local scale parameters remain frozen while the expert updates/restores its own tree. | Do not call the current local substitution a same-checkpoint speedup measurement. Both outputs may still be recorded as observations. |

### Request replay invariant

The transport remains UC and lossy. Replay lives in `xonwire.py` / the expert application,
not in the sealed verbs implementation. The 48-byte frame header and wire version stay
unchanged; two reserved words carry the requester session. A responder uses a new
operation ID for every forward, reverse, begin, and commit, including reverse-mode calls
whose corresponding forwards occurred in the opposite order.

For one serial requester and a resident worker, let `h` be that session's greatest
completed operation. The worker executes only IDs greater than `h`. Completion stores
the response and advances `h` before attempting transmission. An ID equal to `h` replays
the cached response; older IDs have already been implicitly acknowledged by subsequent
operations. Therefore loss/replay cannot add a gradient atom or optimizer commit twice.
The request sender drains while offering frames, so request backpressure does not stop
response consumption. The quiet interval schedules replay; it never cancels useful work.
Owner-requested shutdown interrupts a backpressured local datagram send.

This is not a durable exactly-once transaction across worker crashes or arbitrary
exceptions during model mutation. It assumes one in-flight operation per requester and
the updated worker/responder pair. Session-zero legacy replies remain readable, but
mixed old/new worker generations do not acquire the duplicate-suppression guarantee.
Legacy session-zero requesters retain their existing execution behavior: their reused
forward/backward IDs are not mistaken for already completed new-protocol operations.
Checkpoint I/O follows response caching, so a checkpoint-write failure cannot make an
already completed batch eligible for transport replay as a new optimizer operation.

### Page ownership invariant

Let `a` be ACK-ring occupancy and `s` the posted-send count. Before consuming an arena
submission the bridge requires `a + s < MESH_RING`. A successful post moves one slot
into `s`; a completion moves that obligation from `s` to `a`. A failed post moves the
consumed descriptor directly into `a`. Thus returning every outstanding arena send after
pair retirement fits the reserved ACK capacity; completion and retirement cannot both
return a page because completion clears its private submitted bit.

QP and CQ destruction finish before retirement returns ownership or dead-client rings
are reset. A failed destroy retains its handle and reports/retries instead of forgetting
a live device object. No page is recycled while that queue may still DMA. An ACK means
the producer may reuse its memory, not that the peer received its payload. The existing
single-client-per-region ownership contract is unchanged.

### Why the added branches do not withhold capabilities

Duplicate-request handling supplies the previous result instead of applying a destructive
second update. Teardown retries prevent premature memory reuse. Loaded-service checks
preserve the existing provider; they do not select a less capable machine class. Runtime
reuse requires the same project, lock, Python selection, and successful imports. A failed
new runtime leaves the old environment intact while the installer continues other steps.
Fetch validation protects the preceding source tree from partial extraction. Compiler
selection uses an available executable or builds one; build failure preserves the old
compiler. No new branch rejects a node because of its type, location, or workload size.

## 2. Make the implementation reconstructible

| Named branch | Contents | Publication |
|---|---|---|
| `codex/september-runtime` in mesh | Preserved pre-review September work: 1,800 changed files, including the existing comment/harness removals and runtime/geometry/policy work. | Local; not pushed. |
| `codex/availability-closure` in mesh | The above plus the review repairs, compiler reconstruction recipe, and this ledger. | Local; not pushed. |
| `codex/mesh-capacity` in the neighboring NetRadiant checkout | Names the existing compiler-capacity repair without changing its worktree. | Local; its only remote is upstream. |

`vendor/netradiant-capacity.patch` also carries that compiler repair inside the mesh
branch. `bin/mesh-q3map2-build.sh` clones the **latest named upstream branch**, applies the
repair (or recognizes it already applied), and builds in a fresh generation. Only a
successful build replaces the compiler pointer. No source fetch selects a commit hash.
See [compiler provenance and reconstruction](../vendor/PROVENANCE.md).

The new compiler was built from a fresh upstream `master` clone, not from the neighboring
checkout's old object files. Mesh's engine, QuakeC compiler, gamecode, Python source and
lock are already tracked. Stock Xonotic content, host developer tools/system libraries,
local credentials, and run/checkpoint artifacts remain external inputs; this is source
reconstruction, not a claim of a self-contained OS image or bit-identical binaries.

Root and user updaters still fetch the newest selected branch on every run. Equal
downloaded archives reuse the prior generation; changed archives extract elsewhere.
Existing generations remain available to processes using them. Generation reclamation
needs an ownership-aware policy; neither age-based deletion nor overwriting a live
environment is introduced here. Publication is still required before another node can
fetch these mesh branches from GitHub.

## 3. Reconcile closure without erasing either accomplishments or gaps

The [seventeen release obligations](../GOAL.md) are indexed below. Claimfiles describe
implementation; they are not seventeen new empirical passes. Existing game/runtime
evidence remains credited, and current-generation measurements remain separately due.

| Goal obligations | Implementation/evidence surface | Still required for current release closure |
|---|---|---|
| 1–6: rewards, Gram construction, independent interventions, formal game semantics, travel horizon, distinct actuators | [Policy](claims/POLICY.md), historical R24–R38 | Same-run reward/coordinate/intervention and realized actuator records. |
| 7, 13, 17: feasible paths, shared navigation object, stock-map/bridge fusion | [Geometry](claims/GEOMETRY.md), [reconciliation](GEOMETRY-RECONCILIATION.md), reconstructed compiler | Match the built compiler, generated geometry, engine collision/path observations, and requested composition in one artifact lineage. |
| 8, 16: identity succession and velocity cadence | [Policy](claims/POLICY.md) | Joined/departed/successor support and emitted/persisted coordinate equality. |
| 9: optimization on assigned mesh hosts | Expert, responder, curriculum; historical two-host inference/updates are already demonstrated | Current split-expert batch: input cotangent, gradient atoms, update count, checkpoint lineage, distinct host work, and resume continuity. |
| 10–11: supported operating-point search and exact leased fabric aggregate | [Mesh telemetry](claims/MESH-TELEMETRY.md); both nodes reachable in September 4 observation | Workload records with explicit missing support, retained membership, and no invented throughput. |
| 12, 14: causally independent controller scatter/gather and 256-team interfaces | [Engine scale](claims/ENGINE-SCALE.md), R36–R37 gameplay evidence | Current engine's complete transaction and team-incidence records, not only row-buffer counts. |
| 15: tensor-path DPP and measured utilization | [Policy](claims/POLICY.md), matrix execution measurements | DPP work and utilization on the same current workload interval. |

The immediate order is repairs → branch/source reconstruction → ledger reconciliation.
Items 4 and 5 are intentionally not launched by this review:

4. A current distributed learning episode would measure the **complete split-expert
   learning transaction and its recovery**, not discover whether mesh computation has
   ever worked. A hardware-necessity comparison additionally needs synchronized parameter
   identity and corresponding inputs; the current frozen-local comparison does not supply it.
5. A comparative study would measure **solver behavior and causal benefit**: mirrored
   policies/interventions, realized team-up/take-down play, objective conversion,
   robustness, held-out support, and the specification's requested controls. It is not
   a substitute for, or a retroactive pass/fail judgment on, the existing integration demo.

### Verification record, September 4

Observed compiler/runtime output:

> compiler ready: /Users/mdot/dox/mesh/.build/netradiant/current/install/q3map2
>
> mesh runtime ready
>
> mesh runtime retained: /tmp/mesh-closure-build.TlHTQL/runtime/runtime-current
>
> wire, worker, responder imports complete; no Mesh instance opened

The bridge and client library built as arm64 Mach-O on the MacBook (Darwin 25.3.0) and
Mini (Darwin 25.5.0). Both report Python 3.12.14, UV 0.12.6, MLX/MLX-Metal 0.32.2,
NumPy 2.5.2, and lock digest
`a05a01dc696ac4b40f7689765e157d3605e15ecaa292a342969328f1a7a55ee9`.
Shell syntax, Python imports/AST parsing, and diff whitespace checks passed.
The C compiler reports the pre-existing partial `qpi.gid` initializer warning;
NetRadiant reports obsolete `-s` and missing `/usr/X11R6/lib` linker-search warnings.

The September 4 observer still reports both nodes reachable and no stale node. Its
sampled RDMA rates were zero and workload producer support absent. That describes the
sampled interval only; it does not revoke the historical or operator-attested runs.
No live bridge was restarted, no new verbs device was opened, no fault was injected,
and no strategy study was launched. Source/build closure and fleet rollout are distinct.
