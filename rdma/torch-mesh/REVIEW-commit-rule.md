# Review: the commit rule (branch commit-rule, 2026-10-21)

This branch implements findings 1-7 of metal-microbench
`output_data/crossing-20261021/commit-policy-review.md`. It was written and judged by the rubric. It was not built or
run (the operator: "source is written and rubric judged rather than built or incrementally run tested").

Rulings it applies (the operator, 2026-10-21, as relayed):
- "batching is a tensor dimension ... batching is not something that should be semantically accessible to the source
  abstraction"; "good performance cannot be opt-in".
- "we are here to REMOVE fences, REMOVE guards, REMOVE waits." So the rule's job is throughput: the GPU fed with the
  fewest command buffers while the host keeps issuing. It is not a guard against what a host might do. A host that
  blocks outside torch while its publication waits stalls its peer, which is the program's responsibility; the wait's
  bound ends the call as a value.

An earlier revision of this branch also committed the open buffer from an `MTLSharedEventListener` on torch's queue as
the GPU's count arrived. The coordinator's instruction of 2026-10-21 removed it under that ruling. With it went the
only second thread that touched torch's stream, and with that the race against torch's off-queue `synchronize`.

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

1. Each buffer the library commits signals its count on a shared event as the GPU reaches its end.
2. A publication is committed at once, with every publication waiting in torch's open buffer, while at most `lead` of
   those buffers are in flight, read from the event's value at that moment.
3. Otherwise it joins torch's open command buffer.
4. The next publication applies the rule again. A host wait commits the open buffer: the library's (`_stream.commit`,
   `host_wait`) and torch's own (`synchronize`, `.item()`, `.cpu()`).
5. Nothing else commits it. A host that blocks outside torch with a publication waiting stalls its peer until the
   peer's bounded wait ends the call.

`lead` is the node's record, read once when a group is made. A missing record fails there.

## Ranked pairs, change by change

### 1. One commit rule for every caller (findings 1, 3, 5)

- **C4 Asynchrony: old ≺ new.** Old: `publishing` committed at every publication unless `batching_` was set. Only
  `compiler` set it, so eager calls, the library's own callers and user programs paid one command buffer a
  collective (40 us eager against 26 us in the region, same bytes). New: `MeshStream::published` serves every caller.
  A publication joins the open buffer while more than `lead` buffers are in flight.
- **D3 Control flow: old ≺ new.** Old: `if (!batching_ || ...)`, a mode the caller enters. New: one path. Its only
  branch is the depth against `lead`, which is configuration.
- **D2 When choices are made: old ≺ new.** Old: `committed_ - progress_.signaledValue < 2`, a literal in the
  function. New: `lead` comes from the node's record (`backend.node_lead`), read once at group creation. A stream made
  without it throws (`stream_of`'s `TORCH_CHECK`).
- **C5 When time may decide: old ≈ new.** Both decide at the publication from the event's value, a count of buffers
  (known depth), not a timing estimate. It decides only grouping, never correctness. New: the depth is the node's data.
- **C2 Invariance: old ≺ new.** One implementation for eager, compiled, library and user callers, on any torch stream.
- **C1 Provenance: old ≺ new.** NCCL's calls enqueue on a stream and return; the caller never chooses when the stream
  is flushed. Old: whether work was flushed per call depended on a region the caller entered. New: one rule for all.
- **D4 One implementation: old ≺ new.** `compiler` is now Inductor's backend with nothing wrapped around it.
- **M8 Hot path: old ≈ new.** Per publication, one read of `signaledValue` and, where the depth allows, one commit:
  the old region's costs, now for every caller.
- **D10 Pace: not yet ranked.** No reading of this rule exists. The 10-02 targets are 17.4 us eager and 12.7 us
  compiled, and 12.7 was read under this decision (mesh 3c00d37). The regression of finding 8 is not addressed here.

Finding 2 asked for commits at the GPU's progress and at host waits. The progress-event commit is not kept, under the
throughput ruling: the next publication and host waits commit, and a host blocked elsewhere is the program's to avoid.

### 2. The stream's state as one object a stream owns (finding 4)

- **C6 Failure's reach: old ≺ new.** Old: `batching_`, `deferred_`, `committed_`, `progress_`, `held_` and
  `streamed_` were file statics. An exception between `batch(1)` and `batch(-1)` left `batching_` raised for good,
  every later publication deferred, and the peer hanging until the wait's bound, with nothing naming the cause. New:
  no counter exists. `MeshStream` holds `committed` and `open`, read and written by the host inside torch's queue.
- **C2 Invariance: old ≺ new.** One object per torch MPS stream (torch 2.14 has pool streams:
  `getStreamFromPool`, `setCurrentMPSStream`).
- **Threads: old ≈ new.** Only the host touches torch's stream, from inside torch's queue, as torch's own kernels do.
  No second actor exists, so torch's off-queue `synchronize` meets nothing of the library's.

What an exception does at each point (C6):

| where | what is left |
|---|---|
| `backend.node_lead` (no record) | raised in `ProcessGroupMesh.__init__` before the bootstrap: no communicator, no stream state |
| `stream_of` (lead not configured) | thrown before anything is encoded; no `MeshStream` is made |
| `workspace` (torch's allocator throws) | caught; returns NULL; the library fails the group as a value ("out of GPU memory") |
| inside `on_stream`'s or `complete`'s block | nothing throws: the library returns values, and torch's `synchronize(COMMIT)` is `commitAndContinue` (installed binary) |
| `checked(result)` after the block | the rule already ran; workspace is held only for a ticket that succeeded |
| `host_wait` (torch's `checkLastError` after `COMMIT_AND_WAIT`) | the buffer was committed before the wait; `open` is cleared by the next `waiting()`; `streamed` stays set, so the next host-path call waits again |
| a compiled graph's run | nothing is entered, so nothing is left raised |

No exception leaves a state that defers publications. The next publication or host wait applies the rule afresh.

### 3. Workspace follows the group's retirement (finding 6)

- **C6 / C4: old ≺ new.** Old: `commit()` cleared `held_`, so memory lifetime rode on the commit. New:
  `MeshStream::retire` keeps a split group's workspace by ticket until `ncclMeshRetired()` passes it. A group complete
  in the stream's order returns its workspace after the encode, where later work reuses it after the group's. The
  commit never touches workspace.
- **C6: old ≺ new.** Old: an exception from `at::empty` could pass through the library's C frames. New: `workspace`
  returns NULL, which the library already reports as a value (nccl-mesh.c, "out of GPU memory").
- **Robustness: old ≺ new.** The allocator's context is the `MeshStream`, which is never freed. The library calls the
  allocator only at issue (`launch_metal`, nccl-mesh.c:3160-3193) today, but the crossing-rewrite branch edits that
  region; a never-freed context cannot dangle whatever it does.

### 4. replay without per-step waits (finding 7)

- **C4 Asynchrony: old ≺ new.** Old: `torch.mps.synchronize()` before the steps and after each step. New: steps issued
  back to back, one wait where the time is read (`elapsed_time`).
- **Measurement: restated, not ranked.** Old: the last step from an idle GPU. New: the last step's interval in the
  stream, the span between MPS timing events around it. With steps back to back it runs from the end of the step
  before it to its own end: the GPU's time or the host's issue, whichever is slower. Readings of the two definitions
  are not compared.

### 5. Commits at the library's host waits

- **C4: old ≈ new, with a commit added before existing waits.** `backend.counted` and `Recording._backend`'s store
  meeting now call `_stream.commit()` first, for every stream. They are host waits, and without the commit nothing
  else would commit the open buffer. No wait is added.
- **Host path: old ≺ new.** Old: `streamed_` was one flag, and the wait was torch's default stream off its queue
  (`deviceSynchronize`). New: completion is encoded on the current stream as before, then the current stream and each
  stream a group was issued on are committed and waited for on their queues (`host_wait`), and their retired
  workspace released.

### 6. Documents

- README "Performance", "Record and replay" and "What works"; the `compiler`, `record` and `replay` docstrings; the
  `_stream.mm` comments; `docs/writing-a-service.md`. Each states the rule as the code runs it.

## The independent review (a subagent with the rubric file, 2026-10-21) and what was done

It reviewed the revision with the listener.
1. `examples/chain-service.py:74,78,88,91` still call the deleted `_stream.batch` and fail with AttributeError.
   `examples/` belongs to another agent; finding 9 deletes the calls there. Not changed here.
2. The listener could race torch's off-queue `synchronize`. Gone: the listener is removed, and only the host touches
   torch's stream.
3. `workspace` allocates on torch's queue; under memory pressure torch's allocator enters that queue again
   (`MPSAllocator.mm` `release_cached_buffers`), a deadlock, not an exception. This predates the branch and lies outside
   findings 1-7. The comment no longer claims a value in that case. A fix needs the library to report a group's
   scratch bytes before encoding, or a pool grown off the queue: an `nccl.h` change, left for its owner.
4. It found "A host wait commits, since nothing else would" wrong while the listener committed regardless. Without the
   listener it holds again, and the code and README say so.
5. `replay`'s time is a step's interval in the stream (MPS timing events take host times in listener callbacks of
   torch's own, equally late at both ends), and with steps back to back it is the slower of the GPU's time and the
   host's issue. Restated in the docstring and README.
6. The lead record had no producer, accepted `true`, and named no path for bad JSON. The loader now names the path for
   any read or parse failure and takes only an int. No probe writes the record yet (README); provisioning is to write
   it.
7. The host path waited only on the current stream. It now waits on the current one and every stream a group was
   issued on. Completion is still encoded on the current stream (`ncclMeshComplete` completes every issued group on
   the stream it is given), so Metal-path groups from two torch streams remain unsupported, as before.
8. `on_stream`'s and `complete`'s blocks now drain an autorelease pool, as `host_wait`'s does. "In flight" counts the
   library's buffers, not torch's: stated.

Checked and found sound by the review: manual retain and release of `open`; `std::exchange`; the export of
`dispatch_sync_with_rethrow`; the `ncclMeshStream` field order; no `dispatch_sync` from torch's queue and no tensor
freed on it; a throw in `host_wait` or `stream_of` leaves nothing deferring.

## Behaviours relied on

Confirmed by reading:
- `MPSCommandBuffer.rootCommandBuffer` is the buffer `commitAndContinue` commits; its `status` is the one to query
  (MPSCore `MPSCommandBuffer.h`, `rootCommandBuffer` and `commitAndContinue`).
- torch 2.14 (installed binary, read with otool):
  - `MPSStream::synchronize(COMMIT)` is `commitAndContinue`.
  - `MPSHooks::deviceSynchronize`, `MPSGuardImpl::synchronizeDevice` (`torch.accelerator.synchronize`) and
    `THPMPSStream_synchronize` call `MPSStream::synchronize(COMMIT_AND_WAIT)`, which commits torch's open buffer.
  - `MPSHooks::recordEvent` (`torch.mps.Event.record`) passes `syncEvent=true`, which commits.
- torch 2.11 source: `.cpu()` and `.item()` copy through `MPSStream::copy_and_sync` with `COMMIT_AND_WAIT` on the
  queue (`Copy.mm`, `Scalar.mm`, `MPSStream.mm`); `elapsedTime` commits and waits on the queue (`MPSEvent.mm`).
- torch's extension build adds no `-fobjc-arc`, so `_stream.mm` is manual retain and release.
- `at::mps::dispatch_sync_with_rethrow` is exported by libtorch_cpu.

Not confirmed, for the first build:
- A queue's command buffers end in commit order, so the k-th signal means buffers 1..k are done.
- `MPSCommandBuffer`'s root stays below `Committed` until torch or the library commits it, including when MPSGraph
  calls `commitAndContinue` itself (handled: then `waiting()` is false).
- In torch 2.14, which path of `mps_copy_` calls `MPSStream::synchronize` directly.

## What the first build and the first crossing.py run must show

- It builds, links `dispatch_sync_with_rethrow`, and imports. Every node has `~/.cache/mesh/stream.json`
  (`{"commit_lead": 1}` reproduces the 10-02 depth), or `init_process_group` fails naming it.
- metal-microbench `crossing.py`, both ranks, eager and `COMPILE=1 COMPILER=mesh`, with the TP ratio, aggregate and
  per-stream rates logged as the rubric's Quantities ask:
  - Compiled at or under 12.7 us a crossing; eager at or under 17.4. Under one rule the two should now be close,
    since eager no longer pays a buffer a crossing.
  - Command buffers per step (Metal System Trace, or a count of `committed`) well under 141 in both.
  - The first check passes (each rank contributes rank + 1).
- If either reading misses its target, finding 8's regression history decides what is restored.
