# Fused crossings on torch's stream

Branch `fused-stream`, merged with mesh main (49eda816). The source was written and judged against design/rubrics.md.
On this M5 it was built and installed in the branch's worktree, `mesh-advise` and `mesh-advise mutate` were run, the
torch-mesh extension was built, and the chain kernel's instances were compiled in one process. Nothing has run on two
ranks.

A fused crossing is libnccl-mesh's one-shot all-reduce on n ranks, run by the caller's kernels. The kernel that makes
a rank's part writes it into the rank's send tensor and releases it to every peer. The kernel after the call waits for
each peer's part to land in its own landing tensor, then adds them to its own part. The library runs no kernel of its
own for either step. (Until main's merge db7326b3 it was a two-rank in-place float16 sum through a slot of 61,440
bytes; the sections below on the branch keep that history.)

## The form

**The library** (rdma/). The classic form gets what the Metal 4 form has, for a fused call only.
- `ncclMeshStream.residencySet` is read on the classic form when `fused` is set (nccl.h, Metal path).
- `metal_begin` takes it, attaches the device's argument ring, and makes the ring, its progress word and the failure
  words resident through it (nccl-mesh-metal.m:189). `record4` adds each record's own buffers.
- `metal_send_deferred` and `metal_land_deferred` need the ring, not Metal 4 (:404, :410).
- `launch_metal` refuses a classic fused call with no residency set (ncclInvalidArgument, nccl-mesh.c:3237).
- The steps' text, now in nccl-mesh-kernels.h, defines `struct mesh_step_binding` (:56): the six bindings as pointers,
  in `ncclMeshFused`'s order, which a classic kernel takes through `setBytes`.
- No step's body changes. Main's steps are kept as they are: `mesh_wait_bounded`'s fence before each poll, and
  `mesh_step_await`'s barrier and fence after an observed landing. No new exported function.
- The library's AIR is byte-identical to main's, since the struct is unused by its own kernels. Only the kernels' text
  SHA-256 in vendor/libnccl-mesh.sha256 changes.

On main, a fused group encodes no dispatch on any number of ranks. The groups in flight before it are completed first,
and then each of its channels is completed in the stream's order, one after another (nccl-mesh.c `complete`,
`complete_track`). The tracks dispatch never runs for a fused group.

**torch-mesh.** `torch_mesh/fused.py`, and a section of `_stream.mm` of its own (`fused_residency`, `metal_prelude`,
`fused_exchange`, `fused_published`):

| Call | Takes | Returns |
|---|---|---|
| `fused.steps()` | nothing | the Metal text to compile a kernel after: `ncclMeshMetalPrelude`, rdma/mesh.metal and then the steps |
| `fused.exchange(send, landings, polls, group=None)` | the rank's part; a landing tensor for each other rank, in rank order, each its own size; `polls`, the landing waits' bound (`_stream.polls`, the library's); a group of n ranks | `Crossing(publish, land)`, a tuple of six addresses for each peer each. A call the library won't fuse raises with its reason; nothing slower runs. |
| `fused.encode(handle, buffers, constants, at, gx, gy, threads, peers, land=None, publish=None)` | `_stream.encode`'s arguments, the peer count, and the crossings it lands and publishes | nothing. One that publishes goes under the one commit rule. |

- The kernel's constant block is its own words padded to an even count, then each peer's landing `mesh_step_binding`,
  then each peer's publication, each address two words, low word first.
- The call runs on main's `MeshStream`: the stream's workspace allocator (a fused group asks it for nothing), its
  `streamed`, and its `published()` for a publishing kernel. That is the one commit rule, the same as for every group.
- The residency set is the device's one set, added once to each queue it meets.

## The chain (examples/)

The host makes two calls a step: `fused.exchange` of the step's part, which only writes the records (a publication
and a landing for each peer), and the kernel's encode. One more dispatch after the last step lands the last crossing.

**Landing crossing t − 1, at the start of step t's prologue.** For each peer's landing in turn:
1. Every thread of every worker calls `mesh_step_enter`, which stores the record's sequence in the ring's progress word
   (one thread). Every thread calls `mesh_step_await`. In each worker, thread 0 polls the holding's `landed` word
   (since 2026-10-22 not `sent`: below).
   Each poll is one `volatile coherent(system)` load after a fence, at most `polls`
   of them. Then the worker passes a barrier and one system-scope fence. Nothing reads a gate.
2. Each worker's threads then read their own part from `p[t − 1]` and the worker's slice of each peer's part from that
   peer's landing tensor (`mesh_step_read<half>(l.d + l.p->dst, i)`). They add them in float32, own part first and the
   peers in rank order, round to float16 once, and use z for x = e + tanh(z). On rank 0 they also write tanh(z), step
   t − 1's output, into its staging buffer.

**Publishing crossing t, at the end of stage 1.**
1. The `Summand` epilogue writes the part into `P` (`p[t]`, the send tensor), each worker its own columns. Nothing is
   copied: the bridge sends `p[t]`'s own pages.
2. `mesh_step_enter` on each publication's bindings stores its sequence (one thread).
3. After a device barrier, each worker counts its arrival (`mesh_last_arrival<true>`: a system-scope fence, then one
   atomic add by thread 0).
4. The last worker to arrive calls `mesh_step_release` on each publication: a barrier, then thread 0's one
   system-scope fence (`mesh_publish`) and one store of that holding's `ready`.
5. That worker resets the count. A run's first step resets it too, in worker 0's prologue, before any worker counts.

No worker waits for another to publish.

Since 2026-10-22 the library gates no landing. A step's receives are posted as `fused.exchange` presents them, so the
peer's part for crossing t may land while step t's kernel still reads crossing t − 1's from the same landing tensor.
The operator: "if source says to write continuously into ab uffer, the nic must write continuously into that buffer
even if this means that dat aprocessing code can race or lose parts of a stream". The program guards the pages it
names again (design/pages-and-functions.md, Freeing: rows for NFE k+1 point at other pages until NFE k's are
released). The landing wait no longer covers `sent` either: a kernel that writes a send tensor again first calls
`mesh_step_sent` on the publication that last sent from it (torch_mesh/fused.py).

**What moved in chain-service.py.** Step t − 1's output is written by step t's kernel. So a block's outputs are copied
one step after the block ends, and the pace mark moves with that copy. The host still waits for the mark two blocks
back, and the steady time counts the steps after that mark. The sum ring stays one block, so a block is still a third
of the slots `ring_bytes` holds, and the float64 check reads the same windows. Rank 0's per-block tanh dispatch is gone,
since the kernel writes tanh(z) itself. The plain all-reduce timing (`crossing_microseconds`) is kept, as the
crossing the fused step no longer pays.

## What the first two-rank run must show

- No command buffer error (no GPU fault, so the residency set covers what the kernel touches). The async error reads
  ncclSuccess, and the chain's `err` word is 0.
- The float64 check reads what main's chain reads on the same inputs: 1.37e-3 at step 0 and 1.19e-3 at step 3,584. The
  arithmetic is the same (each sum rounded to float16 once, own part first). Only tanh's last bit may differ, so
  differences in the last digits are expected.
- The steady step falls below main's 35.4 µs. The plain crossing alone (`crossing_microseconds`) stays where it was
  (17.2 µs in the last short run; about 12 µs for a bare crossing), since that path is unchanged. Each run logs
  aggregate rows/s, steps/s per stream, the host and paced times and the parts tuple, beside main's readings by the
  same harness.
- `ncclMeshTimes` gives nonzero times for the crossings' groups, so the balancer has stretches.
