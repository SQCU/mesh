# Fused crossings on torch's stream

Current implementation, 2026-10-24. The contract is [nccl.h](../nccl.h),
[the data model](../../design/data-model.md), and [whole-function dependencies](../../design/whole-function.md).
The original branch record is retained below as history; its peer-sized API, receiver gates, and timing targets
do not describe the current implementation.

## One physical write representation

The communicator retains every reciprocal physical cable as a separate channel. A logical write can be striped
across parallel cables and forwarded through other ranks. The common compiler fixes its source ranges, relay
landings, final destinations, channel order, and prefix dependencies before execution. Fusion changes which kernel
publishes or consumes those ranges; it does not select a different fabric or discard routes. A different solver
must emit this same representation. Unsupported fused forms fail instead of silently reading the source through
an ordinary collective before its producer runs.

Rank metadata and physical bindings have different extents. `sizes`, exact `counts`, and segment `starts` have
one entry per rank. Publication and landing bindings have room for the communicator's physical channel count,
obtained from `ncclMeshCommSample`. Each binding holds six GPU addresses in `mesh_step_binding` order; channels
without applicable work have null records. Kernels test the record before entering or releasing it. There is no
assumption that ranks minus one equals channels, or that a channel contains only one fragment.

| Interface | Current contract |
|---|---|
| `fused.steps()` | The loaded library's Metal prelude and step definitions, used to compile the caller's kernels. |
| `fused.writes(send, landings, pairs, group, polls=0, repeats=1, stride=0)` | Immutable source ranges and disjoint landing leases for declared source/destination pairs, lowered to physical WRITE records; returns `Crossing(publish, land)` in physical channel order. Repeated frames use declared byte displacements in the full backing leases; registration cuts are agreed once before execution. |
| `fused.write_rows(send, landings, pairs, group, row_bytes, rows, stride)` | Exact affine row views with rank-local pitches; one logical message regardless of row count, using the same prepared WRITE machinery. |
| `fused.encode(handle, buffers, constants, at, gx, gy, threads, channels, land=None, publish=None)` | Caller words padded to an even count, then `channels` landing records and `channels` publication records; publishes under the stream's commit rule. |
| `fused.encode_many(calls)` | An ordered list of the same encoding arguments, prepared before one native queue entry; every frame keeps its own constant words. Publication applies the stream's commit policy and the batch ends by submitting any remaining producer command buffer without waiting. |
| `group.prepare_writes(deliveries)` | Realizes local page views and all physical routes before execution; the returned operands execute those fixed writes. |

The former `fused.exchange`, native `fused_exchange`, and collective-specific `fused_gather` paths have been removed.
Generic data functions and explicit contiguous/affine writes share `execute_fused_writes`. Every operation reaches
`ncclMeshWritePrepare`/`ncclMeshWriteExecute`, using the same per-pair physical routes and registration-boundary cuts.

## Generic gather, sum, and persistent outputs

`partition.gathered` and `partition.summed` declare exact source-major WRITE groups: each nonempty source sends
its contribution to every other rank. Empty sources emit no dummy packet. Sum gathers the ranks' partials and folds
them in the consumer. `_fused_form` compiles the
publisher, consumer, shapes, original source strides, and physical channel extent once per form; native caches
include those strides. `gather_publish` reads a strided source directly, without a hidden `contiguous()` tensor.
`linear_publish` computes directly into its publication segment; unsupported column strides raise at realization.

The generic publisher still writes a source segment in the collective's common layout, and `gather_land` still
assembles the logical output. This path is not a claim that every existing tensor is sent from its original bytes
or that all layout copies have disappeared. `allgatherv_out`/`fused_step_out` let the consumer write a supplied
destination, removing the additional copy from an allocated result. `examples/lstm.py` uses this for persistent
hidden state `h`; `examples/chain.py` reaches the same routed writes through its tensor operations.

For a consumer with one threadgroup, its compile-time `WAIT` specialization observes the incoming prefixes inline.
For multiple threadgroups, one `gather_ready` dispatch observes all relevant physical prefixes, then performs one
threadgroup barrier and system fence. The serial MPS encoder orders the parallel `WAIT=false` consumers after it;
they use coherent payload loads without repeating the wait in every worker. These are data-availability
dependencies, not network acknowledgments. The generic operation still joins its complete operand; finer consumers
can name narrower dependencies through the common physical representation.

Source publications release only actual source fragments. Relay fragments become sendable from their incoming
landed prefixes, forwarding those same exact byte spans. A final consumer does not wait for later relay-only entries or for
completion of its own outgoing SEND merely to read available input.

## Plans, commits, and ownership

`torch_mesh.plan` declares the function's collective order and counts before repeated execution. CP and EP examples
now supply their declarations as well as TP. `configuration.placement` computes work shares before tensors are built, prepares
the fixed plan before execution, and sends whole-call observations to the shared statistics worker afterward.
The next invocation reads the worker's saved proposal. A cached form first realized lazily is not by itself
proof that every planning cost has been moved out of the measured function; a benchmark must realize and declare
its complete workload before measuring.

Generic recurring forms keep prepared WRITE handles with each of their two bound storage generations. Each
source/destination pair owns a distinct relay lease; final landings use compact source-rank segments. Preparation
exchanges registration metadata only when a generation is realized. Repeated execution uses the retained handles
and performs no metadata exchange. Zero-share forms retain fresh generations and therefore prepare their new
addresses at each batch boundary, rather than claiming a recurrence proof they do not have.

GPU producers publish through the stream's commit rule. For configuration-time migration from tensors written by
ordinary Torch GPU kernels, `_stream.publish_boundary()` commits the preceding producer command buffer without
waiting for CPU/GPU completion. The following work stays on the same serial queue. This establishes visibility;
it sends no receipt and adds no reciprocal receive gate.

Fresh disjoint WRITE and fused landing ranges can be preposted independently of the receiver's outgoing producer.
Ordinary NCCL calls with in-place aliases retain their existing receive gates; those gates are not part of the
immutable WRITE contract. A reused range needs proof that every previous reader precedes its next first write:

- Positive-contribution recurrent all-gathers use `lifetimes.prove_recurrent_gather`: each rank consumes the previous
  pieces before its next publication. The returning natural data dependencies prove two-slot reuse. This proof is
  specific to that recurrence, including relays; it does not certify unrelated feed-forward or asymmetric sends.
- A zero-count participant has no corresponding returning publication. Native realization therefore gives that
  shape fresh generations and retires old leases asynchronously instead of applying the positive-count proof.
- Other source/relay leases remain immutable through their final DMA reads. Region retirement combines completion
  of local GPU users with `ncclMeshMemReusable`, which includes registered reads not yet posted by the bridge.
  Dropping a Python tensor or completing its producer alone does not release its bytes for reuse. Destination
  readers have their own lifetime dependencies. No receipt all-gather or remote ACK establishes these lifetimes.

## Exact indexed migration

`partition.follow` uses `migration.compile_rows` at the configuration boundary to derive compact affine ranges
between the original sources and final destinations, including interior tensor axes and zero-share ranks. For
contiguous region layouts, `_follow_bytes` binds each remote range through `fused.write_rows`: source and final
landing arguments are views of the real tensors, while intermediate ranks own separate compact relay leases.
Only useful row bytes cross the links. There is no page padding, remote packing kernel, unpacking kernel, or receipt
collective in this path. A 5,003-row range remains one logical WRITE instead of 5,003 stream-ring entries.

Each descriptor holds source/destination offsets, row length, row count, and local pitches. Native preparation
agrees cuts at row boundaries and every participating registration boundary before execution. No DMA span writes
neighboring bytes outside the declared rows. Original publications are encoded before incoming readiness waits.
A retained contiguous interval with no incoming ranges aliases its original storage; other retained pieces copy
locally once. Unsupported non-contiguous layouts still use `follow`'s older whole-operand staging fallback.

The earlier padded/masked frame implementation is superseded for these contiguous layouts. Its measurements below
remain historical evidence: they describe the implementation tested at that time and do not describe current wire
payloads. Masked stores remain a possible data-function lowering where arithmetic needs them; they are no longer
required solely because a pure-copy range cuts a transport page.

## Evidence and remaining performance work

`rdma/tests/fused-physical.py` passed 176 calls per rank on both two- and three-rank cluster runs on 2026-10-24. Each
run included 32 aligned asymmetric migrations across leading and interior axes, with the staging collective
patched to raise if used. Collective cases covered changing inputs, repeats, ragged and zero shares, fp16/fp32,
strided views, persistent output destinations, and sums. Large collective cases checked positive traffic on every
observed physical channel. The three-rank run observed six cables and ingress counts 5/5/2; this is a dated
observation, not a topology constant. Six local tensor-view lowering tests also check exact pointer offsets,
distinct relay leases, aliases, zero shares, and capability rejection.

The masked extension has 120 actual local GPU checks over vector widths 1/2/4/8/16 bytes, physical binding extents
1/3/5/6, inline/separate frame readiness, and multiple frames. They check raw bits (including NaN payloads), padded
tails, row/frame boundaries, and poisoned neighboring bytes, with null transport bindings. Pure tests compare 200
random compact masks against exact byte ranges and check source-before-wait scheduling. The extended three-rank
cluster run passed 184 calls per rank, including eight guarded masked migrations with the whole-operand staging
collective patched to raise. Original bytes and poisoned neighbors remained exact, and all six cables carried
traffic. Each odd-row commodity used three frames: 229,397 useful bytes occupied 245,760 transport bytes. The tiny
three-byte case occupied one 16,384-byte page; this page-grain amplification is reported, not counted as useful
bandwidth. The final row-specialized kernels and batch-boundary flush passed the same 120 local GPU checks and
184 calls per rank on both two- and three-rank cluster runs. The recorded results are
`metal-microbench/output_data/review-20261024/interface-refactor/masked-row-specialized-two-summary.json` and
`masked-row-specialized-three-summary.json`. The two-rank odd-row commodity used four frames and 262,144 transport
bytes for the same 229,397 useful bytes; these successful runs retain the stated page-padding cost.

The exact-byte successor passed `rdma/tests/byte-writes.py` on three ranks, including unaligned three-byte writes,
odd rows, 5,003 rows, all six cables, and a 1 GiB+19-byte registered span. Five trials of 128 complete numerical
functions observed median 31.290039 microseconds for exact-byte transport and 35.569336 microseconds for the padded
control. Those are full function observations, not isolated NIC latency. The later generic gather/sum consolidation
has local declaration tests, ASan/UBSan generation-cache tests, and a successful extension build; its live regression
must be recorded separately rather than inferred from the earlier collective or byte-write runs.

These establish correctness and route use for those cases. They do not establish maximum useful bandwidth or
minimum whole-function latency. Performance claims require complete-function timings, invariant useful output
bytes, and comparison with an all-link capacity bound during the transaction's active windows. Wire traffic
includes relay/redundant bytes and is not interchangeable with useful throughput.

## Historical branch record — not the current API

The following original branch notes are preserved as historical evidence. Their API table, receive-gate lifetime
argument, per-worker waits, unrun test status, and numerical targets were the branch's state, not current guidance.

Branch `fused-stream`, merged with mesh main (49eda816). The source was written and judged against design/rubrics.md.
On this M5 it was built and installed in the branch's worktree, `mesh-advise` and `mesh-advise mutate` were run, the
torch-mesh extension was built, and the chain kernel's instances were compiled in one process. Nothing has run on two
ranks.

A fused crossing is libnccl-mesh's one-shot all-reduce on n ranks, run by the caller's kernels. The kernel that makes
a rank's part writes it into the rank's send tensor and releases it to every peer. The kernel after the call waits for
each peer's part to land in its own landing tensor, then adds them to its own part. The library runs no kernel of its
own for either step. (Until main's merge db7326b3 it was a two-rank in-place float16 sum through a slot of 61,440
bytes; the sections below on the branch keep that history.)

### The form

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

### The chain (examples/)

The host makes two calls a step: `fused.exchange` of the step's part, which only writes the records (a publication
and a landing for each peer), and the kernel's encode. One more dispatch after the last step lands the last crossing.

**Landing crossing t − 1, at the start of step t's prologue.** For each peer's landing in turn:
1. Every thread of every worker calls `mesh_step_enter`, which stores the record's sequence in the ring's progress word
   (one thread). Every thread calls `mesh_step_await`. In each worker, thread 0 polls the holding's `landed` word, and
   its `sent` past this rank's send. Each poll is one `volatile coherent(system)` load after a fence, at most `polls`
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

No worker waits for another to publish. One landing tensor serves every step: the library posts a step's receives only
once this rank's publication of that step has stored `ready`, after the kernel that read the landing before. The sum
ring stays safe by the host's encode order: step t's kernel lands crossing t − 1, its wait covering `sent` past this
rank's send, before it publishes crossing t.

**What moved in chain-service.py.** Step t − 1's output is written by step t's kernel. So a block's outputs are copied
one step after the block ends, and the pace mark moves with that copy. The host still waits for the mark two blocks
back, and the steady time counts the steps after that mark. The sum ring stays one block, so a block is still a third
of the slots `ring_bytes` holds, and the float64 check reads the same windows. Rank 0's per-block tanh dispatch is gone,
since the kernel writes tanh(z) itself. The plain all-reduce timing (`crossing_microseconds`) is kept, as the
crossing the fused step no longer pays.

### What the first two-rank run must show

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
