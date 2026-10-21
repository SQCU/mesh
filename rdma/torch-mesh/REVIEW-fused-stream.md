# Review: fused crossings on torch's stream (branch fused-stream)

The diff from mesh main 2e2a2718, read against design/rubrics.md before any build. Nothing was built or run. An
independent reviewer read the same diff with the rubric file; its findings and what became of each are at the end.
The design and the interface are in [DESIGN-fused-stream.md](DESIGN-fused-stream.md). `old ≺ new` means the change
ranks higher.

## The files and their kinds

| File | Kind | Change |
|---|---|---|
| rdma/nccl.h | C | contract only: the classic form's `residencySet` for a fused call, `mesh_step_binding`, the order on the classic form, the publishing kernel encoded after the call |
| rdma/nccl-mesh.c | C | `launch_metal`: a classic fused call admitted with a residency set and refused without one; `metal_begin` takes it |
| rdma/nccl-mesh-metal.m, .h | C | `metal_begin(…, residency, …)`: the argument ring and residency for a fused classic program; `struct mesh_step_binding` in the steps' text; the deferred steps need the ring, not Metal 4 |
| rdma/nccl-mesh-host.c | C | `metal_begin`'s signature; its refusal of a caller's kernel says why |
| rdma/NCCL.md | C (its document) | one bullet: fused crossings on either form |
| rdma/torch-mesh/torch_mesh/_stream.mm (its own section, 711-779), fused.py | C (the bindings) | `fused_steps`, `fused_all_reduce`, `fused_published`; `steps`, `all_reduce`, `encode` |
| the chain sketch in the design note | D | how examples/chain.metal would call it (examples/ is branch chain-stream's) |

No M code is touched. The bridge sees the same units, ready words and landed words.

## C. Collective code

**C1. Provenance.**
- A torch-only crossing (a Metal 4 queue of torch-mesh's own, or a crossing torch-mesh writes) ≺ the library's own
  call: `ncclAllReduce` on an `ncclMeshStream` whose `fused` and `residencySet` are set (nccl-mesh.c:3129-3131,
  :3150). No function is exported for it.
- The torch side follows prior art for the residency: defaultformer adds an `MTLResidencySet` to a classic
  `MTLCommandQueue` for buffers its encoders reach without binding them (OnlinePolicy.swift:1698-1708, :3063-3066).

**C2. Invariance.**
- A fused crossing for Metal 4 callers alone (main nccl-mesh.c:3128 refused `!metal4`; nccl-mesh-metal.m:643, :649
  refused any program but Metal 4) ≺ either form through one record path: `record4` and the device's argument ring
  (nccl-mesh-metal.m:446-466, :661, :667).
- Unchanged lower pole: fusable only on two ranks, as one piece within one small slot (61,440 bytes), a float16 sum.

**C4. Asynchrony.**
- The chain on torch's stream: a step kernel, then `dist.all_reduce`'s own group (its send, landing and track
  dispatches), then the next kernel ≺ the publication in the kernel that makes the part and the wait in the kernel that
  reads the sum, the call dispatching neither step.
- A fused publication outside the stream's commit rule (2f2777d7, which followed a ruling meant for branch
  commit-rule) ≺ under it, as every publication is (`fused_published` → the rule's entry; 5fb2127a). One rule for all
  work a peer waits on, and that rule's work is throughput: the GPU fed with the fewest command buffers.
- Open, for branch crossing-rewrite: a fused group still dispatches the track kernel over items that run no step
  (nccl-mesh-metal.m `metal_tracks`: `r->tracks` nonzero, `r->ops` 0) ≺ no dispatch. Written here as 42767aa1,
  reverted in 225e535c, since that branch owns the tracks.
- In the chain sketch: workers waiting on stamps until one releases ≺ the last worker to arrive storing the part and
  releasing it (`mesh_last_arrival` at device scope), no worker waiting.

**C5. When time may decide.** Equal. Each landing's wait ends at the data or at `polls`, the caller's bound
(`fused.all_reduce`'s required argument). Nothing is decided from a timing.

**C6. Failure's reach.**
- A call the library does not fuse run as its slower form ≺ failing as a value: the library refuses it
  (ncclInvalidUsage) and torch-mesh raises with the library's message. The operator, 2026-10-21: "good performance
  cannot be opt-in ... code which can't perform well can fail instead of being run".
- A classic fused call refused with a message naming Metal 4 (main) ≺ admitted with its residency set, and refused as a
  value without one (ncclInvalidArgument, nccl-mesh.c:3129-3131).
- torch-mesh's own filters on the operand (float16, a region tensor; 7d1f61ad) ≺ none (5fb2127a): the dtype goes to
  the library, whose refusal is the value. Contiguity stays: the call names one range of bytes.

**C7. Symmetric knowledge.** Equal. Both ends lay the same positions from the same program; the bindings are local
addresses and nothing about them crosses.

**C8. Numerical candor.** Equal. The chain sketch adds its own part and the peer's in float32 and rounds once, as
`op_float` does.

**C9. One primitive, offered as a header.**
- Each classic caller declaring its own record of six pointers ≺ `struct mesh_step_binding`, defined once in the steps'
  text (nccl-mesh-metal.m:57), in `ncclMeshFused`'s order.
- A copy of the steps in torch-mesh ≺ `ncclMeshMetalSteps` read at import (fused.py:37).
- Who owns the reduced steps: branch crossing-rewrite. The operator's form of the steps (a publication: the slot
  stores, one system-scope fence, the word; a landing: volatile coherent(system) polls with no fence, a bounded count,
  then `mesh_payload`; no gate, claim, check or stop read in a kernel; no credit check on the GPU) is that branch's
  rewrite of the `mesh_step_*` bodies. This branch adds one line to the text (`mesh_step_binding`) and changes no body.
- Open: the steps' text joined from two sources, the checkout's rdma/mesh.metal found by the package's path
  (fused.py:37, as metal-microbench's rank reads it) and the dylib's STEPS ≺ one source, the library handing out the
  mesh.metal it embeds (mesh_metal.h). That changes `ncclMeshMetalSteps`' contract and the rank's prelude, so it is not
  done here.

**Strings are messages.** The group found by its name on every call (`communicator_of(group_name)`, a first draft) ≺
the communicator's handle passed (`group.handle`, fused.py:53). A per-step crossing reads no text.

## D. Data code (the chain sketch, not in the tree)

- **D4.** The chain's own crossing code ≺ the library's steps called from its kernel, and torch-mesh's
  `mesh_last_arrival` and `mesh_arrival_reset` (argmax.metal) for the last arrival.
- **D2.** A bound defaulted inside a function ≺ `polls` named by the configuration and passed to every call.
- **D3.** Open: `dist.all_reduce` after a kernel still runs the slow form where the fused one is expressible. Making it
  fail there is the callers' change (chain-stream's), not this branch's.
- **D10.** Open: the last worker re-reads `P` to store the slot ≺ an epilogue that writes `P` and the slot in one pass.
  The second needs each simdgroup matrix element's (row, column), which Metal does not document; metal-microbench's
  layers.metal re-reads the same way.
- **D10.** No measurement exists. The run must log aggregate rows/s, steps/s per stream, the step time and the parts
  tuple beside the unfused readings by the same harness (commit-policy-review.md: 51.6 µs a step in the batch region,
  94 µs committing at each publication).

## M. Mesh code, and the inversion

- No M code changes.
- Inversions, last row: the library's dispatches holding the addresses of the stream's slots and ready words ≺ a data
  kernel holding them through the bindings. Higher in D (D10: no copy between the kernel and the wire), lower in M
  (M5); the Metal 4 form's inversion already. The operator, 10-20: "if this is the fastest possible operation that the
  hardware allows, we might need to allow it; memory safety is not a requirement but zero reducible overhead is a
  requirement".

## The independent review's findings, and what became of each

1. **The residency set never drops a buffer, and a comment claimed a lease's buffer lives for the process's life.**
   `ncclMemFree` releases a lease's buffer, and `comm_free` the channels' and failure words' buffers, while the set
   still names them. The false comment is gone (5fb2127a). Open, in the library: a residency set that never drops ≺ one
   the library drops a lease's buffer from at `ncclMemFree` and a communicator's at its destroy. The Metal 4 form has
   the same exposure. Whether a residency set retains what it holds is not stated in the SDK header.
2. **One program reads two devices** (the ring from the residency set's device, the pipelines from the default
   device). Kept as the classic form already was; listed with the behaviours not confirmed.
3. **The sketch's arrival count could be left short.** Fixed: a run's first step resets it, the last arriver resets it
   with `mesh_arrival_reset`, and a short count within a run belongs to a run that has already failed.
4. **C4: fused publications left in torch's open buffer.** Fixed by restoring `fused_published` (above).
5. **D3/D4: two rules for work a peer waits on.** Fixed by the same.
6. **D9: a fused crossing leaves no evidence marks.** Not added. Marks at the call and after the landing kernel would
   book the step kernel's compute as the crossing's time, since the work and the wait share one kernel. Ranked: such
   marks ≺ no marks, with the group's times from the bridge (`ncclMeshTimes`, as metal-microbench's rank reads them).
   Open: evidence.py reads no `ncclMeshTimes`.
7. **C6/D4: the float16 and region filters.** Removed (above).
8. **C9/Strings: mesh.metal found by the package's path.** Open (above).
9. **C9: the sketch's hand-written reset.** Now `mesh_arrival_reset`.
10. **D10: the sketch's re-read of `P`.** Open (above).
11. **"The stream's commit rule commits it"** was false on this branch alone. The text now says what calls what.
12. **nccl.h said "its publishing kernel before the call"** against "encoded after the call returns". Fixed.
13. **Stale "Metal 4" comments** in nccl-mesh-metal.m (:316 the ring "the Metal 4 form's arguments", :398 the ring
    "made at its first Metal 4 program", :893 `metal_end` committing "the Metal 4 form's residency set") and the name
    `ncclMeshMetal4Records` (an exported name, kept). Open: that file's further edits are crossing-rewrite's.
14. **Untracked files cited by a commit message.** Committed with this review.

Found here, open for crossing-rewrite: `launch_metal` admits a fused float32 sum that `ncclMeshFusableRooted` calls not
fusable (its type check is in `ncclMeshFusableRooted` alone).

## Merge points

- **commit-rule** (finished) rewrote `_stream.mm`'s commit machinery. This section calls it in two places, and the
  merge changes three lines: in `fused_all_reduce`, `streamed_ = true` becomes `stream_of(stream).streamed = true`
  (after the block, as `on_stream` does); in `fused_published`, `publishing(stream)` becomes
  `stream_of(stream).published()`. The section uses no workspace of torch's (`fused_scratch`), so the allocator's new
  context does not reach it.
- **crossing-rewrite** edits `complete()`, `issue_group`, `plan_direct`, the two-rank scratch copy and `send_small` in
  nccl-mesh.c, the `tracks` kernel in nccl-mesh-metal.m, and `mesh_wait_bounded` in mesh.metal. This branch changes
  none of those lines. The regions it shares: `launch_metal`'s fused checks and its `metal_begin` call
  (nccl-mesh.c:3127-3137, :3150), the steps' leading comment and the new `mesh_step_binding` line (nccl-mesh-metal.m:20-30,
  :57), `metal_begin` (:446-466), and the deferred steps (:659-669).
