# Fused crossings on torch's stream

Branch `fused-stream`, merged with mesh main (49eda816). The source was written and judged against design/rubrics.md.
On this M5 it was built and installed in the branch's worktree, `mesh-advise` and `mesh-advise mutate` were run, the
torch-mesh extension was built, and the chain kernel's instances were compiled in one process. Nothing has run on two
ranks. The ranked pairs are in [REVIEW-fused-stream.md](REVIEW-fused-stream.md).

A fused crossing is libnccl-mesh's fastest all-reduce on two ranks. The kernel that makes a rank's part publishes it
into the crossing's slot. The kernel after the call waits for the peer's part to land, then adds it to its own. The
library runs no kernel of its own for either step.

## Could torch's stream run it, as nccl.h stood at 2e2a2718?

No.

1. **The library refused it.** nccl-mesh.c:3128-3132 failed a fused call unless the stream was the Metal 4 form.
   torch's stream is the classic form: `_stream.mm:136` built its `ncclMeshStream` with no argument table, no residency
   set and no `fused`.
2. **No place for the records.** The caller's kernel reads a fused step's arguments later, by address. Only the Metal 4
   form had such a place, the device's argument ring (nccl-mesh-metal.m:315-317, 395-409), attached only by
   `metal_begin4` (:454, :463). A classic dispatch passed its arguments with `setBytes` (:568). The two deferred steps
   refused anything but Metal 4 (:643, :649).
3. **Nothing resident.** A classic encoder makes a buffer resident only if it is bound, named by `useResource`, or in a
   residency set. `metal_resident` did nothing without a set (:475).
4. **No way to pass the bindings.** `_stream.encode` takes tensors and one block of 32-bit words (`_stream.mm:545-550`).
   The bindings are six 64-bit addresses (nccl.h:186-188), and nccl.h named no record a classic kernel could take.
5. **The steps' text was ready**, compiled after rdma/mesh.metal at Metal 4.0, as metal-microbench's rank compiles it.
6. **The order.** The call comes first, since the publishing kernel needs the bindings it returns; the landing kernel
   comes after the publishing one. On torch's stream that is the queue's order.

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

On main, a group on one channel runs as plain dispatches (`complete_track`), so a fused two-rank all-reduce encodes no
dispatch at all. The tracks dispatch runs only when another channel has a group in flight, such as the chain's next
input broadcast at a block's start.

**torch-mesh.** `torch_mesh/fused.py`, and a section of `_stream.mm` of its own (lines 801-877):

| Call | Takes | Returns |
|---|---|---|
| `fused.steps()` | nothing | the Metal text to compile a kernel after: rdma/mesh.metal, then `ncclMeshMetalSteps` |
| `fused.all_reduce(tensor, polls, group=None)` | a contiguous MPS tensor, the operand in place; `polls`, the landing wait's bound (`_stream.polls`, the library's); a two-rank group | `Crossing(publish, land)`, six addresses each. A call the library won't fuse raises with its reason; nothing slower runs. |
| `fused.encode(handle, buffers, constants, at, gx, gy, threads, land=None, publish=None)` | `_stream.encode`'s arguments, and the crossings it lands and publishes | nothing. One that publishes goes under the one commit rule. |

- The kernel's constant block is its own words padded to an even count, then the landing's `mesh_step_binding`, then
  the publication's, each address two words, low word first.
- The call runs on main's `MeshStream`: the stream's workspace allocator (a fused group asks it for nothing), its
  `streamed`, and its `published()` for a publishing kernel. That is the one commit rule, the same as for every group.
- The residency set is the device's one set, added once to each queue it meets.

## The chain (examples/)

The host makes two calls a step: `fused.all_reduce` of the step's part, which only writes the two records, and the
kernel's encode. One more dispatch after the last step lands the last crossing.

**Landing crossing t − 1, at the start of step t's prologue.**
1. Every thread of every worker calls `mesh_step_enter`, which stores the record's sequence in the ring's progress word
   (one thread). Every thread calls `mesh_step_await`. In each worker, thread 0 polls the landed word. Each poll is one
   `volatile coherent(system)` load after a fence, at most `polls` of them. Then the worker passes a barrier and one
   system-scope fence. Nothing reads a gate.
2. Each worker's threads then read the worker's slice of the peer's part from the slot (`mesh_step_read<half>`) and
   their own part from `p[t − 1]` (the landing's `d + dst`). They add the two in float32 and round to float16, as the
   library's own landing does, and use z for x = e + tanh(z). On rank 0 they also write tanh(z), step t − 1's output,
   into its staging buffer.

**Publishing crossing t, at the end of stage 1.**
1. The `Summand` epilogue writes the part into `P` (`p[t]`), each worker its own columns.
2. `mesh_step_enter` stores the publication's sequence (one thread).
3. After a device barrier, each worker's threads store the worker's columns of `P` into the slot (`mesh_step_store`).
4. Each worker counts its arrival (`mesh_last_arrival<true>`: a system-scope fence, then one atomic add by thread 0).
5. The last worker to arrive fences again and calls `mesh_step_release`: a barrier, then thread 0's one system-scope
   fence (`mesh_publish`) and the slot's ready words.
6. That worker resets the count. A run's first step resets it too, in worker 0's prologue, before any worker counts.

No worker waits for another to publish. The slot ring stays safe by the host's encode order: step t's kernel consumes
crossing t − 1 before it publishes crossing t.

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
