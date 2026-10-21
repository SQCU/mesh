# Review: the chain as a stream (branch chain-stream)

The change: findings 9 to 13 of metal-microbench `output_data/crossing-20261021/commit-policy-review.md`, in
`examples/chain.metal`, `examples/chain.py`, `examples/chain-service.py`, `examples/chain-service.json` and
`docs/writing-a-service.md`. It is judged by [design/rubrics.md](../design/rubrics.md), as the operator asked: "source
is written and rubric judged rather than built or incrementally run tested". Nothing here was built or run.

The review had three passes:
- The author wrote the pairs.
- An independent reviewer (a subagent given the rubric, the commit-policy review and the library's source, and none
  of this file) read the first commit. Its findings were fixed in the second commit, or are listed under "Open".
- The coordinator then asked for two changes, made in the fourth commit: one broadcast a block, issued a block ahead,
  and G equal to the node's full core count, with ragged shares of tiles.

## The kind of each file

Every file is data code (D): `examples/` and the configuration it loads (rubrics.md, the table under Quantities). The
doc describes D code. No mesh (M) or collective (C) file is touched, so the pairs use D's dimensions. Where the
commit-policy review cited C4 or C5 for the example (findings 10 and 11), the pair is also given under that citation,
marked as the review's.

`old` is mesh `2e2a2718`. `new` is this branch's head. Line numbers are the new files' unless marked old.

## Ranked pairs, by finding

### 9. The batch region

- **D3, control flow.** old ≺ new.
  - Old: `_stream.batch(1)` and `batch(-1)` around each chunk and around the all-reduce timing (old
    `chain-service.py:74-78`, `88-91`), a mode the caller enters. It had no `try`/`finally`, so an exception between
    the two calls left `batching_` raised for every later publication (the review's finding 4).
  - New: the same loop for every caller, with no mode (`chain-service.py:110-143`).
- **D4.** A commit policy carried by the example ≺ none: the example states only its data flow.
- **D10, pace, on today's library.** new ≺ old.
  - Outside the region every publication commits its own command buffer: 94 µs a step against 51.6 µs in the region
    (the review's table). The new loop publishes once a step (the all-reduce), and once a block for the inputs.
  - So this branch is merged after, or with, branch `commit-rule`. That is the ruling applied, not a choice: "good
    performance cannot be opt-in".

### 10. The chunk, the operand and the rings

- **D2, when choices are made.** old ≺ new.
  - Old: `chunk` (256), a host batch written into the configuration.
  - New: the rings' depths made once at load from the configuration's `ring_bytes` and the operand's size (`ring`,
    `chain-service.py:64-75`). A block is the whole stream where each ring holds it, else a third of the slots that
    fit: an input ring of two blocks and a sum ring of one.
  - A missing `ring_bytes` is a `KeyError`. Fewer than three slots is a `ValueError` naming the bytes three take.
  - The library exposes no capacity of the region (`torch_mesh.records` lists live leases; `mesh.lease` fails past
    the free pages), so the configuration states the service's share of it.
- **D2, and Strings position 5 (declared, read once).** old ≺ new.
  - Old: `rows` and `steps` restated in the configuration and compared with the file (old `:54`, `:62-63`).
  - New: the shape read once from the input's header by rank 0 and broadcast as two int64 words (`operand`, `:27-37`),
    Megatron-LM's fixed-size typed value from rank 0.
  - An empty axis, a dtype other than float16, or a layout other than C order is a `ValueError`. The last two would
    send torch's copy through a temporary or a waited staging copy, and "code which can't perform well can fail
    instead of being run".
- **C4, as the review cites it.** old ≺ new.
  - Old: a chunk's inputs broadcast synchronously after a waited host copy, and a `.cpu()` drain at the chunk's end,
    so the GPU idled at every boundary (old `:71-72`, `:81`).
  - New: before the first step of block b, rank 0 copies block b+1's inputs into its ring and broadcasts them into
    every rank's ring, both queued on the stream (`fill`, `:102-106`; called at `:117`). The wait sits just before
    block b+1's first step, its first reader (`:118`). The drain is queued on the stream too (`:126-127`).
  - Each slot's reuse is ordered by torch's one serial stream, not by a host wait. Block b+1's copy and landing fall
    in the slots of block b-1, whose kernels and staging come earlier on the stream. The reviewer traced the per-step
    form's reuses and found them ordered; the block form keeps the same order with one group a block.
- **D8, memory the program knows.** old ≺ new.
  - Old: rank 0 staged each chunk through `np.array` and `.cpu().numpy()`, two host copies.
  - New: copies straight between the memory-mapped files and the region rings, a sum ring one block deep, and the
    tanh output a buffer held from load (`O`).
- **C5, as the review cites it ("known buffer depth").** old ≺ new.
  - Old: no bound but the chunk's full drain.
  - New: the host is held to the input ring's depth by a mark a block, waited two blocks back (`Pace`, `:83-98`), so a
    block stays queued.
- **Quantities.** old ≺ new: printed beside the parts, rows/s, steps/s and the step's and crossing's times are the
  block, both rings' depths, the host's wait on the ring, and the bytes a step's two collectives move and their rate.

### 11. What the host issues a step

- **C4, waits between steps.** old ≺ new. Old: a host wait every chunk on all of the GPU's work. New: none between
  steps, and one a block, on the mark two blocks back.
- **D10, host calls a step.** new = old: two (the step's encode, `:120`, and the all-reduce, `:121`). The block's
  broadcast and its wait are two calls a block, not a step. The first commit's per-step broadcast (four calls a step,
  and a second publication a step on rank 0) ranked below both.
- **D10, the block's landing.** new ≺ a landing at its reader.
  - A call finishes every call issued before it (`nccl.h`: "a group without issue completes at once, with every group
    before it"). So block b's first all-reduce also lands block b+1's inputs, and waits for the whole block's bytes:
    8 MiB at the sample's 512 steps, once a block.
  - The fix is the fused crossing (branch `fused-stream`: the step kernel lands its input and the peer's part itself,
    `nccl.h` `ncclMeshFusableRooted`, `ncclMeshMetalSteps`), or a completion that lands a group's own ticket only.

### 12. The planner and the cores

- **D3 and D10.** old ≺ new.
  - Old: `while G > 1 and ...: G //= 2` (old `chain.py:24-26`). A share of 520 ran on one worker.
  - New: `workers` (`chain.py:28-37`) refuses only what cannot run: a width or share that is not a multiple of 8, or
    that gives fewer tiles than cores. The `ValueError` names the nearest values that would run (`_nearest`, `:23-24`).
- **D5, every member works.** old ≺ the first commit ≺ new.
  - Old: G the largest power of two the cores allow, halved until it divided the widths.
  - First commit: G the largest power of two the cores allow, and refused otherwise. A 10-core GPU still ran 8 workers,
    with 20% of the GPU idle.
  - New: G = the node's GPU cores (`streamed.node()`), one worker a core. Each worker takes ⌊n/G⌋ or ⌈n/G⌉ of a
    stage's n tiles, the first n mod G taking one more (`share`, `chain.metal:36-40`). This holds for y's columns
    over the share (`ys`) and for z's partial columns, and x's slice, over the width (`xs`; `:123-136`).
  - The operator's rule, as the coordinator quoted it: "always load both devices fully". The coordinator's brief adds
    that a grid of one threadgroup a core is co-resident, and that the Mini's watchdog reset of 10-21 came from a grid
    larger than residency. That holds while one threadgroup of `threads` fits a core. Where another program holds a
    core, a worker waits for it, and the patience bounds the wait.
- **D2.** A fallback found inside the planner ≺ a throw.
- **D2, the ragged shares.** Each worker's tile range is made in the kernel from template constants and the worker's
  index (`share<N, G>(w)`): an identity of the partition, not a guessed value. The K splits come from the larger
  share, ⌈n/G⌉ (`words`, `chain.py:49-55`), so a worker with the larger share stays at one pass where the smaller
  would. They are the same words for every worker, so every barrier stays under uniform control.
- **Correctness (the reviewer's finding 1).** The old stamp wait polled `lane < G` with one simdgroup, so for G above
  32 workers 32 and up were never waited for. Stage 0 could then read X before they wrote it. Now each lane polls
  every 32nd stamp from its own (`chain.metal:21-23`). With G = cores this decides correctness on any GPU of more than
  32 cores, an Ultra's 80 among them.

### 13. Rows at run time

- **D2 and D11.** old ≺ new.
  - Old: RT compiled into each instance, so each row count compiled and timed new instances (old `chain.metal:96`,
    `chain.py:30-32`).
  - New: the row count and the two K splits are words of the constant block (`chain.metal:122`), and the items (row
    group × column tile) are counted at run time (`chain.metal:61-65`). RG is the only row template argument.
  - One instance per (K, S, G, RG, threads, patience) serves any rows that are a multiple of 8·RG up to
    `row_capacity`. Every barrier stays under control flow uniform over the threadgroup (`ks`, `items`, `per`).
- **D2.** The splits for a row count are made once, at `Step.plan`, and looked up by the call. An unplanned row count
  is a `KeyError`. A row count the instance cannot take is a `ValueError` (`words`).
- **D1.** One kernel for a fixed row count ≺ one kernel for every row count of its row group.
- **D6.** new = old ≺ evidence from the mesh's own calls.
  - RG is still chosen by timing the step kernel alone on the rank, as finding 13's brief kept.
  - It is now named (`kernel_alone`, and the doc says what it times). It is not printed beside the collective's
    times (Quantities: one device's performance ≺ the collective's).

### Evidence

- **D9.** old ≺ new.
  - Old: a check of the first steps only, all in the first block, before any slot is reused.
  - New: two windows checked against float64 (`checked`, `:144-152`): the first steps, and the first steps of the last
    block, seeded from rank 0's own output for the step before. The second window reads slots the ring has used
    before.

### The configuration and the doc

- **D2.** `rows`, `steps` and `chunk` ≺ `row_capacity` and `ring_bytes`, each read where it is used, none defaulted.
  The sample's `ring_bytes` (24 MiB) holds three blocks of 512 steps, so its rings wrap.
- The doc keeps its plain style. It says what each wait is and why it is there, including two the code does not
  choose: the block's landing at its first all-reduce, and Metal's queue (below).

## Open: where the new form still ranks low

- **The host's bound today is Metal's queue (the reviewer's finding 2).**
  - Torch's queue takes at most 64 command buffers not yet done. On today's library each publication commits one, so
    the host is held some tens of steps ahead and `Pace` does not wait.
  - `host_microseconds` then includes time held inside torch, and is named and documented that way, not as issue cost.
  - Both change once the commit-rule branch commits only when the GPU needs a buffer.
- **D10, the block's landing** (finding 11 above): a wait for the block's bytes at its first all-reduce, once a block.
- **D4.** tanh(z) is computed twice, as in the old form: in the kernel's prologue for x, and by the drain. The change
  that raises it: the kernel writes tanh(Z) into an output ring as it builds X, and only the last step's is taken
  apart.
- **D1 and D4.** `Pace` and the rings live in the example. The library's `replay` needs the same pacing (the review's
  finding 7). One form in torch_mesh ranks higher.
- **D6.** RG chosen by a solo timing ≺ RG chosen from the chain's own first blocks: one candidate a block, timed
  between the marks and swapped by `Step.use` between blocks. That is a choice made between calls (C5), on the mesh's
  own calls.
- **C4, the wait's form.** `torch.mps.Event.synchronize` parks on a condition variable (torch `MPSEvent.mm`). A spin on
  `query()` ranks higher, though it is once a block.

## What the first two-rank run must show

On the M5 and the Mini, at 8 rows, width 1,024, parts [512, 512] and the sample's 4,096 steps. The baseline is
`2e2a2718`, all inside `_stream.batch`: 51.6 µs a step, 26.1 µs a crossing, and 10.5 to 13.0 µs of host issue.

- **Shape:** `block` 512, `input_ring` 1024 and `sum_ring` 512, so the rings wrap. `kernel.G` equals each node's GPU
  core count. `step.check()` raises nothing, so no stamp wait ran past its patience with every core in the grid.
- **Step time:** `step_microseconds` at or below 51.6 once `commit-rule` has landed.
  - On today's library, expect about the 94 µs of the unbatched form.
  - `aggregate_rows_per_second` is 8 × 10⁶ over it: 155,000 at 51.6 µs.
  - The block's landing adds about 8 MiB over the link's bandwidth once every 512 steps.
  - All cores working should take some µs off the kernel's share of the step on the M5 (10 cores, where 8 ran).
- **Crossing:** `crossing_microseconds` at or below 26.1, the all-reduce now timed with no region around it.
- **Host:**
  - `host_microseconds` near 10.5 to 13.0 µs, with two host calls a step, once the queue no longer holds the host.
    Until then it includes the time torch holds the host.
  - `paced_microseconds` above 0 once the queue no longer binds: the host is ahead and waits only on the ring.
- **Float64 check:** `check_max_difference` for windows 0 and 3584 about the size of `2e2a2718`'s first-steps reading.
  The ragged shares change which worker computes a tile, not a tile's arithmetic. The K splits now come from the
  larger share, so a column's sum may split along K differently: its last bits may differ (C8), its size should not.

## Settings removed at integration (the operator, 2026-10-21)

"if it's parametric it's chosen by an optimization process, and if it's nonparametric it's not a choice".

- **`patience` is structural.** A wait's bound exists so a GPU is never held past the platform's limit. The library
  defines that bound once: `NCCL_MESH_POLLS` loads, holding a GPU at most `MESH_PLATFORM_NS`, 5 s. `_stream` exports
  it as `polls`, and the kernel's PATIENCE is that value. The chain's stamp waits are device-scope loads, faster than
  the system-scope loads the bound was set on, so the same count holds a GPU for less.
- **`threads` is parametric.** It is now chosen by the same timing argmin as RG. The candidates are powers of two of
  simdgroups, ended by the first thread count whose pipeline the device refuses to build; no device maximum is written
  down.
- **`trials` is parametric, but it belongs to the timing, not the service.** Each candidate is timed over one block, the
  ring's own unit of steps.
- **`ring_bytes` stays.** It is the region memory the service is given, an operand, like a process's memory limit.
- **D2: settings ≺ structure and measurement.**
