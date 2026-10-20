# Three rubrics: mesh, collective, data

Code is reviewed by the rubric of the layer it belongs to. Each axis is scored on its own, and failing one fails the
change. The three rubrics contradict one another on purpose: code that would satisfy two of them belongs to neither, and
is split along a boundary (Boundaries, below). The reasons R1-R8 are at the end.

## The layers

| | Mesh | Collective | Data |
|---|---|---|---|
| Code | `rdma/mesh-flow` and what it publishes (mesh-link.c, mesh-verbs.h, mesh-records.c, mesh.h's region layout); the node services (keeper, installers, map, records, mesh-stat) | libnccl-mesh (nccl.h, nccl-mesh*.c/.m), the schedule compiler (mesh-collective.c, mesh-plan.h), the device functions kernels call (mesh.metal), the bindings (torch-mesh backend.py, mesh_mpi.py) | everything that computes values: metal-microbench's engine (bootstrap.swift, dense_execution.swift, drafter.swift, metal/), LiteRT's program (litert_program.m, prepare_e2b.py), a rank's operands and crossings (mesh_rank.m, rank.swift), goal_a.swift, torch programs, examples/, torch-mesh's kernels; and the configuration they load (instance tables, attention plans, tpconfig, partition.py, allocate.py's parts) |
| Lives | as long as the node | as long as a communicator, in its client's process | one call; its configuration, one model load |
| Owns | the device, the window, the page table, every queue pair and request | its schedule and its own rows | its buffers and its numbers |
| Hides | Apple's driver and the cables' hardware | the cabling and the membership, which change daily | the model, its formats and kernels, which change weekly |
| Measured by | frames: at least 1,000 normal per transport error; time a hop | latency and bandwidth per link (αβ) | tok/s at the KL bar |

## Rubric M: mesh code

- **M1. Resident, never self-ending, always killable.** Nothing a client, call, peer, link, time or fault does stops
  it, and any signal from anyone ends it, SIGKILL included. Its stop on SIGTERM is a flag, FAREWELL, exit. Nothing
  makes it, or any other process, harder to end.
- **M2. Only kernel calls that return.** After its start: post, poll and query. No destroy, deallocate, deregister,
  close, drain, or move to the error state. What it no longer uses it forgets; the kernel releases it at exit.
- **M3. Waits on no one.** Every loop polls memory or a call that returns at once, and reads its stop flag each pass.
  It never waits for a client, a peer or a reply. No channel is used 5 s without something happening on it.
- **M4. Content-blind.** It moves bytes between rows. It reads no dtype, shape, value, checksum, model, layer or
  collective. The only bytes it writes are landed words and all-ones duds.
- **M5. Algorithm-blind.** It knows no reduction, gather, tree, root or rank. It is handed positions (peer node,
  length, rows of the client's own) and chooses the queue pairs, the order across clients and the rows a dud names.
  It never derives a position.
- **M6. Sole owner; complete mediation.** It is the only process that makes a verbs call or names a queue pair,
  request, SGE, key, completion or doorbell. A client can express only its own rows (indices into its own row table):
  anything else is unrepresentable, not checked and refused.
- **M7. Everything at start.** The window is registered once and every table sized at start from the node's
  configuration document. Nothing is allocated, registered or resized for a call or a client.
- **M8. No protocol.** HELLO, LINK, FAREWELL and RECORD on the control socket. Data is a SEND into a posted RECV, frames
  matched. No request, acknowledgement, handshake, agreement, session, stamp or retry.
- **M9. A client's error stays the client's.** Its rows are released and its unsent positions go out as duds. No
  client's failure, revocation or exit concludes a link, restarts a queue pair or delays another client. The bridge's
  own errors never reach state other services use.
- **M10. Three nines.** Per port: frames posted, frames landed, duds, transport errors by cause. At least 1,000 normal
  frames per transport error; a link is lost only by a physical event.
- **M11. Hardware latency.** No hop, copy or wake past the hardware's (about 2-3 µs a hop; three times that fails).

Forbidden in mesh code, required elsewhere:
- Failing as a value; ending anything because a call failed (C5, D8). R1, R4.
- Knowing a dtype, a combine, a schedule's meaning or a model; checking payload (C8, D1). R2, R7.
- Specializing for a workload: caches, prediction, policies, features (D3). R3: every feature is shared state with the
  node's lifetime, and changing it means restarting the bridge, the one operation that has panicked the Mini.
- GPU work (D). R1, R6.
- Agreeing with its peer about anything but a pairing (C4). R5.
- Compensating for a client: watchdogs on clients, liveness detectors, state kept for a client "in case", guards
  against being killed. R8.

## Rubric C: collective code

- **C1. Reference parity.** The API and its semantics are NCCL's, call for call, and MPI's for the V variants; each
  algorithm is transcribed from its source (Rabenseifner 2004; Patarasuk & Yuan 2009; Thakur, Rabenseifner & Gropp
  2005; Trivance 2026). A deviation cites the hardware interface that forces it.
- **C2. General.** One implementation per collective, for every dtype and op the API names, any member count of at
  least 2, any counts (ragged included), any connected map. Nothing names a model, a layer, a tensor's role or a
  member's hardware.
- **C3. The schedule is a pure function of its operands.** (map, members, counts, dtype, op, the communicator's
  configured form), compiled before the call, the same at every member. No route, form or chunk is inferred, measured
  or priced at run time.
- **C4. Symmetric and silent.** Every length, order and position is known at both ends from the schedule. Nothing is
  learned from a peer at run time; readiness is the landed word alone. No member decides anything alone.
- **C5. Fails the call, within its deadline, holding nothing.** A call completes, or returns an error naming what was
  lost within its deadline (5 s at most). Nothing resumes within a call, nothing needs cleaning, and the next call runs
  on the next map with nothing to repair.
- **C6. Own rows only.** It writes positions into its rows and supplied words, and reads landed words. It names no
  device object and posts nothing.
- **C7. Allocated once.** Scratch and staging come from its rows at communicator init; nothing is allocated per call.
- **C8. Numerics declared, never checked.** Accumulation precision and reduction order are part of the schedule (fp32
  accumulation, rounded once); bit identity across members is not required; no checksum or payload validation.
- **C9. A library, not a splice.** What a kernel needs (publish, await, read) is a function in a header that data
  kernels call. It never edits, generates or patches another program's kernels.
- **C10. Bound by αβ.** Measured per link as latency plus bytes over bandwidth [Hockney 1994], within the hardware's,
  overlapped with compute where the schedule allows.

Forbidden in collective code, required elsewhere:
- Surviving a failure: retries, resumption, waiting out a peer, keeping a failed communicator for the next call (M1).
  R4: recovery inside a call is a protocol, and calls fail and are rerun by their callers.
- Device objects, posting, registering, claiming queue pairs, repairing links (M2, M6). R1, R6.
- Specialization: variants per model or layer, model math beyond the declared combine, forms chosen by price (D3).
  R2, R3: the cabling changes daily, and a collective shaped to one model or map is wrong on the next.
- Adapting on one member at run time (D4). R5: with no protocol, a decision one member makes alone desynchronizes the
  schedule.
- Knowing what an axis means (D4). R2.
- Compensating for the mesh: timers that declare a peer dead, keepalives, communicators kept to avoid a cost in M. R8.

## Rubric D: data code

- **D1. Accuracy is measured, never gated.** The whole model's KL at the unembedding against the bar (LiteRT's pair
  KL), computed after the call. Draws at temperature 1.
- **D2. Throughput.** tok/s end to end; slower than the last measured form is a regression; under 3% is equal. Kernels
  against the hardware's roofline and LiteRT's per-matrix times.
- **D3. Specialized at load.** Every choice (forms, parts, splits, row classes, kernel instances, attention plans) is
  synthesized once when the configuration loads. Nothing is derived, defaulted or guessed in a function that takes
  tensors; a missing choice throws; as few branches at run time as physically attainable.
- **D4. Shares are data; members differ.** Every partitioned axis is a ragged share from the configuration (tpconfig,
  the Balancer's parts from the calls' evidence). Every member hosts, the second computer is always used, and members
  run different work on different engines.
- **D5. One form.** One implementation of each computation (one layer program for target and drafter; one RMSNorm). A
  deleted form is restored, not redesigned.
- **D6. Collectives through the API.** Crossings are NCCL calls, or the library's device functions called from its own
  kernels. It names no link, peer, route, queue pair, bridge row or bridge object; topology is unknown to it.
- **D7. It knows what it needs.** Weights, KV and crossing buffers are sized by the program from its configuration and
  held in its own rows from load; nothing is allocated per call; no other layer computes a ceiling for it.
- **D8. The call fails; the process does not.** A failed call returns its error to its caller, who may run it again.
  Nothing outlives the call but caches keyed by content (weights, compiled programs, KV pages). No cleanup, no recovery
  inside a call.
- **D9. Waits on its inputs only.** A kernel waits only on its inputs' landed words, through the library's bounded
  await; never on a word another queue of its own writes, never on another member's control flow.
- **D10. Foreign code unmodified.** LiteRT, torch and others are called through their APIs or extended by modules and
  headers; no text edits, monkeypatches or interposition.

Forbidden in data code, required elsewhere:
- Generality across maps, dtypes and ops (C2). R3: generic code leaves the roofline.
- Knowing topology, routes, links, peers' liveness or the bridge (C3, M). R2: a recabling never touches model code,
  and a model never touches the transport.
- Holding device or link state; never failing; cleanup protocols (M1, M2). R1, R4.
- Agreeing with other members at run time (C4). R5: members differ by configuration they all load, never by
  negotiation.
- Payload checks, stamps or digests inside the program (C8, M4). R7.
- Compensating: fallback forms, defaults, clamps, retries, ceilings. R8.

## Boundaries

- **Mesh and collective:** rows the client owns, positions in them, supplied and landed words. Nothing else.
- **Collective and data:** the NCCL API (communicators, buffers, a call's result), the device-function header, and
  memory through `ncclMemAlloc`. Nothing else.
- **Mesh and data:** nothing. Data code never reads the bridge's region, watches its process or sizes itself by its
  window.

## The contradictions

| | Mesh | Collective | Data |
|---|---|---|---|
| On failure | never fails; forgets the client | fails the call within 5 s | returns the error; the caller reruns |
| Ending | resident; any signal ends it | with its process | with its call |
| Content | bytes | dtype and op | meaning |
| Topology | its own cables | the whole map, as an operand | none |
| Specialization | forbidden | forbidden | required |
| Agreement with peers | a pairing only | total, by construction | none: members differ by configuration |
| Device | sole owner | own rows | own buffers, through the API |
| Features | none | NCCL's | any that measures faster at the bar |
| Waits | on no one | on landed words, within 5 s | on its inputs, through the library |
| Allocation | everything at start | at communicator init | at model load |

## Why they contradict

- **R1. Lifetime.** The node outlives a communicator, which outlives a call. Longer-lived state may not depend on
  shorter-lived code, and shorter-lived code may not hold longer-lived state. The operator, 2026-10-14: "functions can
  fail from any one operand becoming invalidatd, servers should not fail when any operand is invalidated (ideally)".
- **R2. What changes.** Each layer hides the one thing that changes on its own schedule: Apple's driver (M), the cabling
  and membership (C), the model (D) [Parnas 1972]. A layer that knows another's secret changes when that changes, and a
  change to M is a restart of every node's bridge.
- **R3. Objectives pull apart.** Reliability grows by removing features (M), generality by removing specialization
  (C), throughput at the bar by adding specialization (D). Code that serves two objectives loses both.
- **R4. Failure.** M never fails and can always be killed; C fails the call within its deadline; D's call fails and is
  rerun. Crash-only: the way to stop is to end, and the way to recover is the next start [Candea & Fox 2003]. Recovery
  inside a call is forbidden at every layer (the operator, 2026-10-14: "within call failures cannot be resume and
  instead must return an error").
- **R5. Coordination without protocol.** M agrees with its peer only on a pairing. C agrees by construction (both ends
  compile the same schedule), so no member decides alone. D's members differ, by configuration all of them load before
  the call.
- **R6. Privilege.** Complete mediation [Anderson 1972; Saltzer & Schroeder 1975]; the four reasons of
  design/transport.md section 2.
- **R7. End to end.** Correctness is checked at the ends (D1's KL, after the call), never in the transport or the
  collective [Saltzer, Reed & Clark 1984]. A check below the ends becomes a gate (the September `K_DIGEST`).
- **R8. An error is removed by its owner, never compensated by another layer.** A mechanism added in one layer to
  survive another layer's error is a second error: it presumes the first will happen, does not remove it, fails in its
  own way, and hides the first from its owner, so the two spread together. Examples from this repository: a guard that
  made processes unkillable, against a deadlock any exit could enter; links concluded to clear receives a communicator
  had left posted; communicators kept across calls to avoid that conclusion; meets and stamps against a missing peer;
  per-rank ceilings against a window the program could size itself. The removals are at the owners: no release calls
  (M2), no receives in client rows (M6), duds (M9), each program's own sizes (D7).

## HEAD against the rubrics (first reading, 2026-10-19)

- metal-microbench mesh_rank.m reads the bridge's region header and watches the bridge's process (`MeshRankBridge`):
  D6, Boundaries.
- metal-microbench weights.swift `withinPlan` and tools/mesh/programs.py `plans` ceilings: D7, R8.
- libnccl-mesh's sessions, meets, digests and stamps: C4, C5, R8.
- Clients claim queue pairs by compare-and-swap on `hdr->lanes` and hold `struct mesh_send`: C6, M6.
- The bridge concludes a link when a communicator ends: M9, M10.
- The bridge stops itself when its binary changes: M1.
- Communicators kept from call to call: C5, R8.
- The kill guard (`bin/mesh-kill-guard.sh`, its shims and hooks; deleted 2026-10-19): M1, R8.

## References

- Anderson, J. P. (1972). Computer security technology planning study. ESD-TR-73-51.
- Candea, G. & Fox, A. (2003). Crash-only software. HotOS IX.
- Hockney, R. (1994). The communication challenge for MPP: Intel Paragon and Meiko CS-2. Parallel Computing.
- Parnas, D. L. (1972). On the criteria to be used in decomposing systems into modules. CACM 15(12).
- Saltzer, J. H. & Schroeder, M. D. (1975). The protection of information in computer systems. Proc. IEEE 63(9).
- Saltzer, J. H., Reed, D. P. & Clark, D. D. (1984). End-to-end arguments in system design. ACM TOCS 2(4).
