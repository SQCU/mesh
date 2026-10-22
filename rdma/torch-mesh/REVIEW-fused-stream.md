# Review: fused crossings on torch's stream (branch fused-stream)

The whole diff against mesh main (49eda816), both merges of main included, judged by design/rubrics.md. `old ≺ new`
means the change ranks higher. Built on this M5 in the worktree: `make -C rdma all install` (every warning an error),
`mesh-advise` (0 warnings, 24 groups) and `mesh-advise mutate` (10 of 10 firing), torch-mesh `build_ext` into a scratch
directory, and the chain kernel's six instances (32 to 1,024 threads) compiled in one process
(metal-microbench output_data/chain-service-20261021/fused-stream). Nothing ran on two ranks. The design is in
[DESIGN-fused-stream.md](DESIGN-fused-stream.md).

## The files and their kinds

| File | Kind | Change |
|---|---|---|
| rdma/nccl.h, rdma/NCCL.md | C (contract) | the classic form's `residencySet` for a fused call; `mesh_step_binding`; the order on the classic form; NCCL.md's poll fence stated as main's code has it |
| rdma/nccl-mesh.c | C | `launch_metal` admits a classic fused call with a residency set and refuses one without; `metal_begin` takes the set |
| rdma/nccl-mesh-metal.m, .h | C | `metal_begin(…, residency, …)`: the argument ring and residency for a fused classic program; the deferred steps need the ring, not Metal 4 |
| rdma/nccl-mesh-kernels.h | C | one new line in the steps' text, `struct mesh_step_binding`; no step's body changes |
| rdma/nccl-mesh-host.c | C | `metal_begin`'s signature; its refusal of a caller's kernel says why |
| vendor/libnccl-mesh.sha256 | C | the kernels' text SHA-256 anew; the AIR is byte-identical to main's |
| rdma/torch-mesh/torch_mesh/_stream.mm (801-877), fused.py | C (the bindings) | `fused_steps`, `fused_all_reduce`, `fused_published`; `steps`, `all_reduce`, `encode` |
| examples/chain.metal, chain.py, chain-service.py, docs/writing-a-service.md | D | the step's crossing fused into its kernel |

No M code is touched. The bridge sees the same units, ready words and landed words.

## The merges

- A rebased branch force-pushed ≺ main merged into the branch (decf86fc, e4d17d34). The branch was already pushed, and
  nothing is force-pushed. The tree is the one a rebase would make.
- Each conflict went onto main's structure. The steps' text moved to nccl-mesh-kernels.h, and the branch's one line
  went there. The `_stream.mm` section moved onto main's `MeshStream`. NCCL.md takes main's text plus the fused bullet.
- No guard main removed comes back, and no fence main kept goes. The fused land step calls main's `mesh_step_await`
  unchanged: thread 0's polls, each after a system-scope fence (`mesh_wait_bounded`), then a barrier and one more
  system-scope fence before the payload is read. The coordinator's earlier brief to drop those fences was wrong
  (tools/poll_probe.m: 3,198 µs median late on the M4 Pro and 2,308 µs at p99 on the M5 without the fence, 0.58 µs with
  it, at most 0.13 µs a wait).
- Main's NCCL.md said "no fence between polls" against main's own `mesh_wait_bounded`. A document contradicting its
  code ≺ the document stating the fence (rdma/NCCL.md:255-260).

## C. Collective code

**C1. Provenance.** A torch-only crossing (a Metal 4 queue of torch-mesh's own, or a crossing torch-mesh writes) ≺ the
library's own call: `ncclAllReduce` on an `ncclMeshStream` whose `fused` and `residencySet` are set
(nccl-mesh.c:3236-3238, :3256). No function is exported for it. The queue residency set follows defaultformer's
`addResidencySet:` on a classic queue (OnlinePolicy.swift:1698-1708, :3063-3066).

**C2. Invariance.** A fused crossing for Metal 4 callers alone (main nccl-mesh.c refused `!metal4`) ≺ either form,
through one record path (`record4` and the device's argument ring; nccl-mesh-metal.m:189, :404, :410). Unchanged
lower pole: fusable only on two ranks, as one piece within one small slot (61,440 bytes), a float16 sum.

**C4. Asynchrony.**
- The chain's step as its kernel, then `dist.all_reduce`'s own group (a send and a landing dispatch, a host call and a
  publication of its own) ≺ the publication at the end of the kernel that makes the part, and the wait at the start of
  the kernel that reads the sum. On main a one-track group is plain dispatches (`complete_track`), so a fused call
  encodes none, and within a block nothing of the library's runs between two step kernels.
- **The publication's two forms, also ranked by M8.** One worker, the last to arrive, storing the whole part into the
  slot and releasing it ≺ each worker storing its own columns into the slot as it finishes, the last to arrive making
  the one release (examples/chain.metal:156-166). In the first, the 16 KiB copy by one threadgroup lies on the path
  after the last worker finishes, as a serial tail. In the second, the stores run in parallel, each as its worker
  ends. The path after the last worker is its own columns, its fence, one atomic add and the release. The second costs
  one system-scope fence a worker, made in parallel; the first costs none but the release's. The worker fences are the
  data function's: several threadgroups' stores made visible to the bridge before one threadgroup's release.
- Rank 0's per-block `torch.tanh` dispatch over the block's sums ≺ tanh written by the step kernel, which already holds z
  (examples/chain.metal:143).
- Open, for the library: a fused call made while another channel has a group in flight (the next input broadcast, at a
  block's start) goes the tracks path, and dispatches the track kernel over items that run no step. Once a block.

**C5. When time may decide.** Equal. Each landing's wait ends at the data or at `polls`, the library's bound
(`_stream.polls`), which is not a setting.

**C6. Failure's reach.**
- A call the library does not fuse run as its slower form ≺ failing as a value: the library refuses (ncclInvalidUsage),
  and torch-mesh raises with its message.
- A classic fused call refused with a message naming Metal 4 (main) ≺ admitted with its residency set, refused as a
  value without one (ncclInvalidArgument, nccl-mesh.c:3236-3238).
- Unchanged: a wait that reaches its bound records a failure; the host reads it after the command buffer and fails the
  group as a value, and every later step's kernel ends at its start (0fe782ec, kept).

**C7. Symmetric knowledge.** Equal. Both ranks lay the same positions from the same program; the bindings are local
addresses.

**C8. Numerical candor.** Equal. The landing adds own part and peer part in float32 and rounds once to float16, the
same arithmetic and order as the library's own landing (`op_float`).

**C9. One primitive, offered as a header.**
- Each classic caller declaring its own record of six pointers ≺ `struct mesh_step_binding`, once in the steps' text
  (nccl-mesh-kernels.h:56).
- A copy of the steps in torch-mesh ≺ `ncclMeshMetalSteps` read at import (fused.py).
- A hand-written last-arrival count in the chain kernel ≺ torch-mesh's `mesh_last_arrival` and `mesh_arrival_reset`
  (argmax.metal), the one implementation.
- A second workspace allocator for the fused call ≺ the stream's own (`workspace` with the `MeshStream` as context).
- Open: the steps' text joined from two sources, the checkout's rdma/mesh.metal and the dylib's STEPS ≺ one source.

**Strings are messages.** The group found by its name on every call ≺ the communicator's handle passed (`group.handle`).

**The commit rule.** A fused publication outside the stream's commit rule ≺ under it: `fused.encode` calls
`fused_published`, which calls `MeshStream::published()`, as every group that publishes does. One rule for all work a
peer waits on, whose job is throughput.

## D. Data code (the chain)

- **D4.** The chain's crossing as a collective call between kernels ≺ the library's steps called from its own kernel.
- **D2.** The landing's bound is `_stream.polls`, the library's, as the stamps' patience is. Nothing is a setting.
- **D8.** The rings are sized as before: the sum ring one block deep, a block a third of `ring_bytes`' slots. A larger
  sum ring for the one-step-late copy (a quarter of the slots, a block of 384 for the short run) ≺ writing tanh(z) into
  rank 0's one-block staging buffer. The second keeps the block at 512 and so keeps the float64 check's windows (steps 0
  and 3,584) comparable with main's.
- **D9.** The timing fields are kept. The pace mark moves with the copy, one step into the next block. The steady time
  counts the steps after that mark (`steps - block - 1`).
- **D3, D10.** Open: `dist.all_reduce` after a kernel still runs the slow form where the fused one is expressible.
  Making it fail there is a change to callers.
- **D10.** No two-rank measurement exists yet. The first run must log aggregate rows/s, steps/s per stream, the step,
  host and paced times and the parts tuple, beside main's readings by the same harness.

## M. Mesh code, and the inversion

- No M code changes.
- Inversions, last row: the library's dispatches holding the addresses of the stream's slots and ready words ≺ a data
  kernel holding them through the bindings. Higher in D (D10: no copy between the kernel and the wire), lower in M
  (M5). The Metal 4 form already has this inversion. The operator, 10-20: "if this is the fastest possible operation that the
  hardware allows, we might need to allow it; memory safety is not a requirement but zero reducible overhead is a
  requirement".

## Open

- The residency set never drops a buffer. A lease's buffer stays named after `ncclMemFree`, and a communicator's
  buffers after its destroy. A set that never drops ≺ one the library drops them from. The Metal 4 form shares this.
- A fused crossing leaves no evidence marks. Marks at the call and after the landing kernel would book the step's
  compute as the crossing's, since the work and the wait share one kernel. Its stretches are the bridge's times
  (`ncclMeshTimes`), which evidence.py does not read.
- `launch_metal` admits a fused float32 sum that `ncclMeshFusableRooted` calls not fusable.
- torch builds the extension for macOS 11, so the residency-set calls warn as unguarded (`-Wunguarded-availability-new`),
  as main's `MTLLanguageVersion4_0` already does at `_stream.mm:607`.
