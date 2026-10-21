# Fused crossings on torch's stream

Branch `fused-stream`, from mesh main 2e2a2718. Source written and judged against design/rubrics.md, not built or run
(the operator: "source is written and rubric judged rather than built or incrementally run tested"). The ranked pairs
are in [REVIEW-fused-stream.md](REVIEW-fused-stream.md).

A fused crossing is libnccl-mesh's fastest all-reduce on two ranks. The kernel that makes a rank's part publishes it
into the crossing's slot. The kernel after the call waits for the peer's part to land, then adds it to its own. The
library runs no kernel of its own for either step. Its callers so far are metal-microbench's rank (mesh_rank.m, on
the Metal 4 form). This note is about kernels on torch's MPS stream.

## Could torch's stream run it, as nccl.h stood?

No. Line numbers are at main 2e2a2718.

1. **The library refused it.** nccl-mesh.c:3128-3132 fails a fused call whose stream is not the Metal 4 form
   (`!metal4`). torch's stream is the classic form: \_stream.mm:136 builds the `ncclMeshStream` from torch's command
   buffer and compute encoder, with `argumentTable`, `residencySet` and `fused` all NULL.
2. **No place for the records.** A fused step's arguments must outlive the call, since the caller's kernel reads them
   later, by address. Only the Metal 4 form has such a place: the device's argument ring (nccl-mesh-metal.m:315-317,
   395-409), which only `metal_begin4` attaches (:454, :463). A classic dispatch passes its arguments with `setBytes`
   (:568), so it has no record to hand out. `metal_send_deferred` and `metal_land_deferred` refused any program that
   was not Metal 4 (:643, :649).
3. **No residency.** The caller's kernel reaches the bindings by GPU address, not through `setBuffer`. A classic
   encoder makes a buffer resident only when it is bound, named by `useResource`, or in a residency set. The library's
   classic dispatches bind theirs (:566-571), and its track kernel calls `useResource` (:855). `metal_resident` does
   nothing without a residency set (:475). So nothing made the stream's rings, the failure words, the argument ring or
   its progress word resident for a torch kernel.
4. **No way to pass the bindings.** `_stream.encode` binds tensors by index and one `setBytes` block of 32-bit words
   (\_stream.mm:545-550). The six bindings are 64-bit GPU addresses (nccl.h:186-188). A classic kernel can take them only
   as bytes: a record of six pointers in its constant block. nccl.h named no such record.
5. **The steps' text was ready.** `ncclMeshMetalSteps` gives the steps (nccl.h:979-987), compiled after
   rdma/mesh.metal. `_stream.pipeline` compiles at Metal 4.0 (\_stream.mm:517), the version the rank compiles the same
   text at (metal-microbench metal_runtime.swift:481-489). Nothing new was needed here.
6. **The order.** The call must come before the publishing kernel, since that kernel needs the bindings the call
   returns ("encoded after the call returns", nccl.h:961). The landing kernel comes after the publishing one. The
   rank's replay does exactly this: the call, then the command before the mark with the publication's bindings, then
   the command after it with the landing's (metal-microbench metal_recording.m:806-809). On torch's stream the order is
   torch's queue order: dispatches in its serial compute encoder, and command buffers in commit order. In the chain one
   kernel lands crossing t − 1 and publishes crossing t, so crossing t's call goes before step t's kernel.

What was missing was not torch's. The classic form lacked a place for a fused step's record and a residency set for
the buffers the bindings name. Everything else (the steps, the records' layout, the waits' bound) was the library's.

## The form

**The library** (rdma/). The classic form gets what the Metal 4 form has, for a fused call only.
- `ncclMeshStream.residencySet` is read on the classic form when `fused` is set. `metal_begin` takes it, attaches the
  device's argument ring, and makes the ring, its progress word and the failure words resident through it
  (nccl-mesh-metal.m:446). `record4` adds each record's own buffers, as it does for Metal 4.
- `metal_send_deferred` and `metal_land_deferred` need the ring, not Metal 4 (:661, :667).
- `launch_metal` refuses a classic fused call with no residency set (ncclInvalidArgument, nccl-mesh.c:3130), and no
  longer refuses the classic form.
- The steps' text defines `struct mesh_step_binding`: the six bindings as pointers, in `ncclMeshFused`'s order
  (nccl-mesh-metal.m:57). A classic kernel takes one through `setBytes` and passes its fields to the `mesh_step_*`
  functions.
- No new exported function. `ncclMeshStream`, `ncclMeshFused` and `ncclMeshFusableRooted` keep their layouts.

**Who owns what** (the operator, 2026-10-21: "why are you conmingling data procesing function ideas with what the
bridge program must do and what the nccl-analogue collective operation api supplies?"). The fused steps are the
library's, under NCCL's contract: every rank issues the same collectives in the same order and leaves in-flight buffers
alone, and the library delivers the bytes. The kernel that calls them is the data function's. Branch `crossing-rewrite`
reduces the steps' bodies on the two-rank path: a publication is the stores into the slot, then one system-scope fence
and the ready word; a landing polls the landed word with volatile coherent(system) loads, with no fence per poll or
after it, a bounded count of polls, then reads the payload through `mesh_payload`; no gate, claim, check or stop read
in a kernel; no credit check on the GPU (the host encodes the consumption of p − LAG before the publication of p).
That branch should own the reduced `mesh_step_*` bodies in `ncclMeshMetalSteps`. This branch adds one new line to that
text, `struct mesh_step_binding`, and the classic form's record and residency code, and changes no step's body.

**Shared regions this branch touched** (crossing-rewrite edits `complete()`, `issue_group`, `plan_direct`, the two-rank
scratch copy and `send_small` in nccl-mesh.c, the `tracks` kernel in nccl-mesh-metal.m, and `mesh_wait_bounded` in
mesh.metal; none of those lines is changed here):
- nccl-mesh.c `launch_metal`: the fused-call checks (:3127-3137) and the `metal_begin` call (:3150), some 40 lines
  above the two-rank scratch copy; `ncclMeshComplete`'s `metal_begin` call (:3262).
- nccl-mesh-metal.m: the steps' leading comment (:20-30, the sentence on `mesh_step_enter`'s progress store), the
  new `mesh_step_binding` line (:57), `struct metal_program`'s comment, `metal_begin` (:446-466), `record4`'s comment,
  `metal_send_deferred` and `metal_land_deferred` (:659-669). The track change written here (42767aa1) is reverted
  (225e535c), so `metal_tracks` is main's.
- nccl-mesh-metal.h, nccl-mesh-host.c: `metal_begin`'s new argument. nccl.h: comments only. mesh.metal: untouched.

**torch-mesh.** A new module, `torch_mesh/fused.py`, and a section of its own in `_stream.mm` (lines 711-779). It
calls the commit machinery and edits none of it. A dispatch that publishes is a publication under the stream's one
commit rule, as a group that publishes is: `fused.encode` calls `_stream.fused_published`, which applies the rule's
entry on torch's queue. On main that entry is `publishing`. Branch `commit-rule` replaces it with
`MeshStream::published`, whose work is throughput (the GPU fed with the fewest command buffers). The call's own
group needs no scratch, so it passes an allocator that makes none (`fused_scratch`) rather than torch's.

**Not taken.**
- A Metal 4 queue of torch-mesh's own beside torch's. Each step would need an event wait between torch's work and
  ours (C4's lower pole).
- `useResource` from torch-mesh. It would have to name the library's buffers, which no caller can reach.
- A copy of the steps or of the binding record in torch-mesh (C9).
- No commit rule for a fused publication (2f2777d7, on a ruling meant for the commit-rule branch). It is restored
  here: one rule for all work a peer waits on.

## The interface

| Call | Takes | Returns |
|---|---|---|
| `fused.steps()` | nothing | the Metal text a kernel source is compiled after: rdma/mesh.metal, then `ncclMeshMetalSteps` (read once, at import) |
| `fused.all_reduce(tensor, polls, group=None)` | a contiguous MPS tensor, the in-place operand (its dtype passed to the library, which fuses a float16 sum); `polls`, the landing wait's bound in loads (0: `NCCL_MESH_POLLS`), from the caller's plan; a two-rank mesh group (default: the default group) | `Crossing(publish, land)`: six GPU addresses each (`ncclMeshFused`). A call the library does not fuse: RuntimeError with its reason. Nothing slower runs instead. |
| `fused.encode(handle, buffers, constants, at, gx, gy, threads, land=None, publish=None)` | `_stream.encode`'s arguments, and the Crossing whose landing and the Crossing whose publication this dispatch runs | nothing; the dispatch is on torch's stream, and one that publishes goes under the commit rule (`fused_published`) |
| `_stream.fused_published()` | nothing | nothing; the rule's entry for a publication, on torch's queue |
| `_stream.fused_steps()` | nothing | `ncclMeshMetalSteps()` |
| `_stream.fused_all_reduce(tensor, comm, polls)` | the tensor, the group's communicator handle, `polls` | `(publish, land)`, two lists of six addresses |

The kernel's constant block at `at` is its own words padded to an even count, then the landing's
`mesh_step_binding`, then the publication's. Each address is two words, low word first. A binding the dispatch is not
given is all zero, and the kernel must not touch it.

Fusable means: two ranks, the direct exchange, a float16 sum, and the whole part within one small slot. A small slot is
(256 − 1) / 17 frames of 4,096 bytes, which is 61,440 bytes (mesh-collective.c:901). So rows × width × 2 must be at
most 61,440. torch-mesh adds no check of its own: the library's refusal is the value.

The residency set is one for the process, added to torch's queue once. The library adds to it every buffer a fused
step's bindings name (the stream's rings, the failure words, the argument ring and its progress word, and each
operand's buffer), and nothing removes any of them. A region tensor (`torch_mesh.empty`) is the natural operand: its
pages are the region's, which the library never lets go while the process lives (nccl.h, Memory).

## How examples/chain.metal would call it

Branch `chain-stream` owns examples/. This says what would change; it edits nothing there.

**Source.** Compile after `fused.steps()` and torch_mesh/argmax.metal, for its `mesh_last_arrival`:
`_stream.pipeline(fused.steps() + ARGMAX + SOURCE + '\n' + line + '\n', name)`.

**Bindings.** The `Z` buffer goes, since the operand never holds the sum. Two buffers are new: `out` (tanh of the
step before's sum) and `arrivals` (one atomic word). The constant block at 15 becomes:

```metal
struct chain_words { uint first, last, phase, pad; mesh_step_binding land, publish; };
```

**Land crossing t − 1, at the start of the prologue.** Every thread of every worker calls the wait, since it holds a
barrier. The kernel reads no gate: a failed communicator is refused at the next call, on the host. `mesh_step_enter`
is called for its record's progress word alone (the argument ring's reuse), its result not read.

```metal
const bool first = words.first != 0;
if (!first) {
    constant mesh_step_binding &l = words.land;
    mesh_step_enter(*l.p, l.v, l.progress, w == 0 && tid == 0);
    mesh_step_await(*l.p, l.a, l.s, l.v, tid);
}
if (first && w == 0) mesh_arrival_reset(arrivals, 0, tid);
uint phase = words.phase;
for (uint q = tid; q < R * Sx; q += THREADS) {
    const uint i = (q / Sx) * K + w * Sx + q % Sx;
    float z = 0.0f;
    if (!first) {
        constant mesh_step_binding &l = words.land;
        z = tanh(float(((device const half *)(l.d + l.p->dst))[i]) + float(mesh_step_read<half>(*l.p, l.s, i)));
        out[i] = half(z);
    }
    X[i] = half(float(E[i]) + z);
}
if (words.last) return;
```

The sum is the rank's own part (the landing's `d + dst`, which is `p[t − 1]`) plus the peer's (`mesh_step_read` at
its `s`), added in float32 and rounded once.

**Publish crossing t, at the end of stage 1.** The `Summand` epilogue still writes the part into `P` (`p[t]`, the
operand), each worker its own columns. A publication is one threadgroup's stores and release, so the last worker to
arrive stores the whole part into the slot and releases it. No worker waits for another.

```metal
constant mesh_step_binding &b = words.publish;
threadgroup uint last;
if (mesh_last_arrival(arrivals, 0, G, tid, &last)) {
    mesh_step_enter(*b.p, b.v, b.progress, tid == 0);
    for (uint i = tid; i < R * K; i += THREADS) mesh_step_store(*b.p, b.d, i, P[i]);
    mesh_step_release(*b.p, b.a, tid);
    mesh_arrival_reset(arrivals, 0, tid);
}
```

The arrival is the data function's: a device-scope count that orders every worker's stores to `P` before the last
reads them (`mesh_last_arrival`'s default scope, as argmax.metal's blocks use it). The release is the library's: one
threadgroup's slot stores, then one system-scope fence and the ready word. The slot's element i is `P`'s (the operand's
layout). The last arriver resets the count for the next step, and a run's first step resets it in worker 0's prologue,
before any worker counts (each counts only after the stamps that follow the prologue). Within a run, a worker that
leaves at a failed stamp wait leaves the count short; that run has already failed (`err`). The host's encode order
keeps the slot ring safe: step t's kernel consumes crossing t − 1 before it publishes crossing t.

**The host.** One call and one dispatch a step. `p` must hold steps t − 1 and t at once, so `chunk` is at least 2.

```python
previous = None
for t in range(steps):
    crossing = fused.all_reduce(p[t % chunk], config['polls'])
    fused.encode(step.handle, [step.At, step.Bt, e[t % chunk], out[(t - 1) % chunk], step.X, step.Y, p[t % chunk],
                               step.stamps, step.err, step.arrivals],
                 [int(t == 0), 0, step.phase, 0], 15, step.args['G'], 1, step.args['THREADS'], land=previous, publish=crossing)
    step.phase += 2
    previous = crossing
fused.encode(step.handle, [..., out[(steps - 1) % chunk], ...], [0, 1, step.phase, 0], 15, step.args['G'], 1,
             step.args['THREADS'], land=previous)
```

The last dispatch (`last` = 1) lands crossing T − 1, writes its output and returns before stage 0. Step t's output is
written by step t + 1's kernel, so rank 0 reads `out` one step behind (`out` holds tanh z already). The configuration
names `polls`.

## What the first build and the first two-rank run must show

**Build.**
- libnccl-mesh and the host executor (`make host`) compile with `metal_begin`'s new argument in nccl-mesh.c,
  nccl-mesh-metal.m and nccl-mesh-host.c.
- The library's own kernel source (mesh.metal, the steps with `mesh_step_binding`, its kernels and the track kernel)
  compiles at its first use.
- `_stream.mm` compiles: the residency-set calls, the new section, three new `module.def` lines.
- A kernel compiled after `fused.steps()` that takes `mesh_step_binding` from its constant block compiles at Metal
  4.0. metal-microbench's prelude, which joins the same steps text, still compiles.

**Two-rank run.**
- Both ranks get nonzero addresses from `fused.all_reduce`, and `ncclMeshMetal4Records` grows by two a step.
- No command buffer reports an error (no GPU fault, so the residency set covers what the kernel touches).
  `ncclCommGetAsyncError` reads ncclSuccess, the chain's `err` word is 0, and no wait reached its bound.
- Rank 0's `check_max_difference` against the float64 reference is close to the unfused run's on the same inputs. The
  function is the same, so only last-bit differences are expected.
- Each run logs aggregate rows/s, steps/s per stream, the step and host-issue times, and the parts tuple, beside the
  unfused readings by the same harness (commit-policy-review.md: 51.6 µs a step in the batch region, 94 µs committing
  at each publication).
- `ncclMeshTimes` gives nonzero times for the crossings' groups, so the balancer has stretches.
- A part over 61,440 bytes raises RuntimeError naming ncclInvalidUsage, and nothing runs.
