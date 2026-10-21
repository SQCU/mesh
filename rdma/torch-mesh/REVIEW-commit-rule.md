# Review: the commit rule (branch commit-rule, 2026-10-21)

This branch implements findings 1-7 of metal-microbench
`output_data/crossing-20261021/commit-policy-review.md`. It was written and judged by the rubric. It was not built or
run (the operator: "source is written and rubric judged rather than built or incrementally run tested").

Rulings it applies (the operator, 2026-10-21, as relayed):
- "batching is a tensor dimension ... batching is not something that should be semantically accessible to the source
  abstraction"; "good performance cannot be opt-in".
- "we are here to REMOVE fences, REMOVE guards, REMOVE waits." So the rule's job is throughput: the GPU fed with the
  fewest command buffers. It is not a guard against what a host might do.

## Kinds of the files touched

| file | kind (design/rubrics.md, Quantities table) |
|---|---|
| `torch_mesh/_stream.mm` (commit machinery, `on_stream`, `complete`, `issue`, exports) | C: the bindings |
| `torch_mesh/__init__.py` (`compiler`, `Recording`, `record`, `replay`) | C: the bindings |
| `torch_mesh/backend.py` (`node_lead`, group creation, `counted`) | C: the bindings |
| `stream.example.json` | D: configuration, a didactic record |
| `README.md`, `docs/writing-a-service.md` | documents |

No file of `nccl.h`, `nccl-mesh.c` or `nccl-mesh-metal.m` is touched, and none of `examples/`.

## The rule in brief

1. A publication is committed at once while at most `lead` of its stream's committed buffers are in flight.
2. Otherwise it joins torch's open command buffer.
3. Each buffer the library commits signals its count on a shared event as the GPU reaches its end.
4. A listener on torch's queue applies the rule again when the count reaches `committed - lead`. So the open buffer is
   committed while the GPU still holds `lead` buffers.
5. A host wait commits: the library's (`_stream.commit`, `host_wait`) and torch's own (`synchronize`, `.item()`,
   `.cpu()`).

`lead` is the node's record, read once when a group is made. A missing record fails there.

## Ranked pairs, change by change

### 1. One commit rule for every caller (findings 1, 2, 3, 5)

- **C4 Asynchrony: old ≺ new.** Old: `publishing` committed at every publication unless `batching_` was set. Only
  `compiler` set it, so eager calls, the library's own callers and user programs paid one command buffer a
  collective (40 us eager against 26 us in the region, same bytes). New: `MeshStream::rule` serves every caller. A
  publication joins the open buffer while more than `lead` buffers are in flight, and the GPU's progress commits it.
- **D3 Control flow: old ≺ new.** Old: `if (!batching_ || ...)`, a mode the caller enters. New: one path. Its only
  branch is the depth against `lead`, which is configuration.
- **D2 When choices are made: old ≺ new.** Old: `committed_ - progress_.signaledValue < 2`, a literal in the
  function. New: `lead` comes from the node's record (`backend.node_lead`), read once at group creation. A stream made
  without it throws (`stream_of`'s `TORCH_CHECK`).
- **C5 When time may decide: old ≺ new.** Old: a deferred publication was revisited only by the next publication or
  the region's end. New: the event (the count arriving) revisits it, on known buffer depth. One sampled read remains:
  the decision at publication reads `progress.signaledValue`. It decides only grouping, never correctness.
- **C2 Invariance: old ≺ new.** One implementation for eager, compiled, library and user callers, on any torch stream.
- **C1 Provenance: old ≺ new.** NCCL's calls enqueue on a stream and return; the caller never chooses when the stream
  is flushed. Old: whether work was flushed per call depended on a region the caller entered. New: the stream flushes
  by its own progress, as a CUDA stream's launch queue does.
- **D4 One implementation: old ≺ new.** `compiler` is now Inductor's backend with nothing wrapped around it.
- **M8 Hot path: open.** New costs per publication: one read of `signaledValue`, and one `notifyListener` registration
  per open buffer (not per publication: `armed`). The registration's cost is not measured.
- **D10 Pace: not yet ranked.** No reading of this rule exists. The 10-02 targets are 17.4 us eager and 12.7 us
  compiled. The regression of finding 8 is not addressed here.

### 2. The stream's state as one object a stream owns (finding 4)

- **C6 Failure's reach: old ≺ new.** Old: `batching_`, `deferred_`, `committed_`, `progress_`, `held_` and
  `streamed_` were file statics. An exception between `batch(1)` and `batch(-1)` left `batching_` raised for good,
  every later publication deferred, and the peer hanging until the wait's bound, with nothing naming the cause. New:
  no counter exists. `MeshStream` holds `committed`, `open` and `armed`, read and written on torch's queue alone (the
  host's groups in `dispatch_sync`, the listener's notifications on that queue).
- **C2 Invariance: old ≺ new.** One object per torch MPS stream (torch 2.14 has pool streams:
  `getStreamFromPool`, `setCurrentMPSStream`).
- **New ≺ old (flagged).** The listener is a second actor on torch's stream. torch 2.14's own synchronize entry points
  run off the stream's queue (read in the installed binary, below). If a listener commit and a host
  `torch.mps.synchronize()` overlap, both can end torch's encoder or commit its buffer at once. The old form had one
  actor. `MeshStream::waiting` narrows the window: once torch has committed the root, the listener does nothing. It
  does not close it. The first build must look for it.

What an exception does at each point (C6):

| where | what is left |
|---|---|
| `backend.node_lead` (no record) | raised in `ProcessGroupMesh.__init__` before the bootstrap: no communicator, no stream state |
| `stream_of` (lead not configured) | thrown before anything is encoded; no `MeshStream` is made |
| `workspace` (torch cannot allocate) | caught; returns NULL; the library fails the group as a value ("out of GPU memory") |
| inside `on_stream`'s or `complete`'s block | nothing throws: the library returns values, and torch's `synchronize(COMMIT)` is `commitAndContinue` (installed binary) |
| `checked(result)` after the block | the rule already ran on the queue; workspace is held only for a ticket that succeeded |
| `host_wait` (torch's `checkLastError` after `COMMIT_AND_WAIT`) | the buffer was committed before the wait; `open` is cleared by the next `waiting()`; `streamed` stays set, so the next host-path call waits again |
| the listener's block | nothing throws C++; an Objective-C exception there ends the process (no catch: rare failures get no machinery) |
| a compiled graph's run | nothing is entered, so nothing is left raised |

No state defers a publication for good. After `rule()`, a waiting publication is either committed or covered by an
outstanding notification at a count the GPU reaches (`armed` is set only when the count was not yet reached after the
notification was asked). One case leaves the open buffer to the next host wait: a committed buffer that fails may skip
its tail signal (as iji-arm-port's DefaultFormer `MetalProvider.swift` notes). The call has then failed already, and the
peer's bounded wait ends it as a value.

### 3. Workspace follows the group's retirement (finding 6)

- **C6 / C4: old ≺ new.** Old: `commit()` cleared `held_`, so memory lifetime rode on the commit. New:
  `MeshStream::retire` keeps a split group's workspace by ticket until `ncclMeshRetired()` passes it. A group complete
  in the stream's order returns its workspace after the encode, where later work reuses it after the group's. The
  commit never touches workspace.
- **C6: old ≺ new.** Old: `workspace` could throw out of `at::empty` through the library's C frames. New: it returns
  NULL, which the library already reports as a value (nccl-mesh.c, "out of GPU memory").
- **Robustness: old ≺ new.** The allocator's context is the `MeshStream`, which is never freed. The library calls the
  allocator only at issue (`launch_metal`, nccl-mesh.c:3160-3193) today, but the crossing-rewrite branch edits that
  region; a never-freed context cannot dangle whatever it does.

### 4. replay without per-step waits (finding 7)

- **C4 Asynchrony: old ≺ new.** Old: `torch.mps.synchronize()` before the steps and after each step. New: steps issued
  back to back, one wait where the time is read (`elapsed_time`).
- **Measurement: restated, not ranked.** Old: the last step from an idle GPU. New: the span between MPS timing events
  around the last step, which with steps back to back runs from the end of the step before it to its own end. Readings
  of the two definitions are not compared.

### 5. Commits at the library's host waits

- **C4: old ≈ new, with a commit added before existing waits.** `backend.counted` and `Recording._backend`'s store
  meeting now call `_stream.commit()` first. They are host waits, and without a commit nothing else would. No wait
  is added.

### 6. Documents

- README "Performance", "Record and replay" and "What works"; the `compiler`, `record` and `replay` docstrings; the
  `_stream.mm` comments; `docs/writing-a-service.md`. Each states the rule as the code runs it.

## Behaviours relied on

Confirmed by reading:
- `MPSCommandBuffer.rootCommandBuffer` is the buffer `commitAndContinue` commits; its `status` is the one to query
  (MPSCore `MPSCommandBuffer.h`, `rootCommandBuffer` and `commitAndContinue`).
- A listener made with `initWithDispatchQueue:` runs its blocks on that queue (`MTLEvent.h`).
- torch 2.14 (installed binary, read with otool):
  - `MPSStream::synchronize(COMMIT)` is `commitAndContinue`.
  - `MPSHooks::deviceSynchronize`, `MPSGuardImpl::synchronizeDevice` (`torch.accelerator.synchronize`) and
    `THPMPSStream_synchronize` call `MPSStream::synchronize(COMMIT_AND_WAIT)` on the default or given stream without
    its queue.
  - `synchronizeAllMPSStreams` does take each queue; only the profiler's `stopCapture` calls it.
  - `MPSHooks::recordEvent` (`torch.mps.Event.record`) passes `syncEvent=true`, which commits.
- torch 2.11 source: `.cpu()` and `.item()` copy through `MPSStream::copy_and_sync` with `COMMIT_AND_WAIT` on the
  queue (`Copy.mm`, `Scalar.mm`, `MPSStream.mm`); `elapsedTime` commits and waits on the queue (`MPSEvent.mm`).
- torch's extension build adds no `-fobjc-arc`, so `_stream.mm` is manual retain and release.
- `at::mps::dispatch_sync_with_rethrow` is exported by libtorch_cpu.

Not confirmed, for the first build:
- A notification asked for a count not yet reached when `notifyListener` returns is delivered when the count arrives.
  (The code does not rely on delivery for a count already reached: it applies the rule at once.)
- The listener's delivery is asynchronous on torch's queue and cheap enough per open buffer.
- A queue's command buffers end in commit order, so the k-th signal means buffers 1..k are done.
- In torch 2.14, which path of `mps_copy_` calls `MPSStream::synchronize` directly, off the queue.
- `MPSCommandBuffer`'s root stays below `Committed` until torch or the library commits it, including when MPSGraph
  calls `commitAndContinue` itself (handled: then `waiting()` is false).
- The race in section 2: a listener commit overlapping a host `torch.mps.synchronize()`.

## What the first build and the first crossing.py run must show

- It builds, links `dispatch_sync_with_rethrow`, and imports. Every node has `~/.cache/mesh/stream.json`
  (`{"commit_lead": 1}` reproduces the 10-02 depth), or `init_process_group` fails naming it.
- metal-microbench `crossing.py`, both ranks, eager and `COMPILE=1 COMPILER=mesh`, with the TP ratio, aggregate and
  per-stream rates logged as the rubric's Quantities ask:
  - Compiled at or under 12.7 us a crossing; eager at or under 17.4. Under one rule the two should now be close,
    since eager no longer pays a buffer a crossing.
  - Command buffers per step (Metal System Trace, or a count of `committed`) well under 141 in both.
  - The first check passes (each rank contributes rank + 1).
  - Many steps with `torch.accelerator.synchronize()` between them, under Metal API validation, to look for the race
    of section 2.
- If either reading misses its target, finding 8's regression history decides what is restored.
