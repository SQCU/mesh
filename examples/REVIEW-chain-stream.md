# Review: the chain as a stream (branch chain-stream)

The change: findings 9 to 13 of metal-microbench `output_data/crossing-20261021/commit-policy-review.md`, in
`examples/chain.metal`, `examples/chain.py`, `examples/chain-service.py`, `examples/chain-service.json` and
`docs/writing-a-service.md`. It is judged by [design/rubrics.md](../design/rubrics.md), as the operator asked: "source
is written and rubric judged rather than built or incrementally run tested". Nothing here was built or run.

Two passes were made. The author wrote the pairs below. An independent reviewer (a subagent given the rubric, the
review and the library's source, and nothing of this file) then read the first commit. Its findings were fixed in the
second commit, or are listed under "Open" with the change that would raise them.

## The kind of each file

Every file is data code (D): `examples/` and the configuration it loads (rubrics.md, the table under Quantities). The
doc describes D code. No mesh (M) or collective (C) file is touched, so the pairs use D's dimensions. Where the
commit-policy review cited C4 or C5 for the example (findings 10 and 11), the pair is also given under that citation,
marked as the review's.

`old` is mesh `2e2a2718`. `new` is this branch's head. Line numbers are the new files' unless marked old.

## Ranked pairs, by finding

### 9. The batch region

- **D3, control flow.** `_stream.batch(1)` and `batch(-1)` around each chunk and around the all-reduce timing (old
  `chain-service.py:74-78`, `88-91`), a mode the caller enters ≺ the same loop for every caller, with no mode
  (`chain-service.py:102-139`). The review's finding 4 reached the old form too: it had no `try`/`finally`, so an
  exception between the two calls left `batching_` raised for every later publication.
- **D4.** A commit policy carried by the example ≺ none: the example states only its data flow.
- **D10, pace, on today's library.** new ≺ old. Outside the region every publication commits its own command buffer:
  94 µs a step against 51.6 µs in the region (the review's table). The new loop has two publications a step on rank 0
  (the broadcast's and the all-reduce's). So this branch is to be merged after, or with, branch `commit-rule`. That is
  the ruling applied, not a choice: "good performance cannot be opt-in".

### 10. The chunk, the operand and the rings

- **D2, when choices are made.** `chunk` (256), a host batch written into the configuration ≺ the rings' depths made
  once at load from the configuration's `ring_bytes` and the operand's size (`ring`, `chain-service.py:64-75`). A block
  is the whole stream where each ring holds it, else a third of the slots that fit (an input ring of two blocks, a sum
  ring of one). A missing `ring_bytes` is a `KeyError`; fewer than three slots is a `ValueError` naming the bytes
  three take. The library exposes no capacity of the region (`torch_mesh.records` lists live leases; `mesh.lease`
  fails past the free pages), so the configuration states the service's share of it.
- **D2, and Strings position 5 (declared, read once).** `rows` and `steps` restated in the configuration and compared
  with the file (old `:54`, `:62-63`) ≺ the shape read once from the input's header by rank 0 and broadcast as two
  int64 words (`operand`, `:27-37`): Megatron-LM's fixed-size typed value from rank 0. An empty axis, a dtype other
  than float16 or a layout other than C order is a `ValueError`: the last two would send torch's copy through a
  temporary or a waited staging copy, and "code which can't perform well can fail instead of being run".
- **C4, as the review cites it.** A chunk's inputs broadcast synchronously after a waited host copy, and a `.cpu()`
  drain at the chunk's end, so the GPU idled at every boundary (old `:71-72`, `:81`) ≺ e_{t+1}'s broadcast issued
  before step t's kernel (`:115`) and waited just before step t+1's (`:116`); rank 0's block copies in and out queued
  on the stream (`:107`, `:114`, `:124-125`). Each slot's reuse is ordered by torch's one serial stream, not by a
  host wait. The reviewer traced every reuse (input copy a block ahead into the slots of two blocks back; the root's
  issued broadcast staging its operand at issue; receivers landing at completion; the all-reduce in place; the drain
  encoded before the next block writes its slots) and found them ordered.
- **D8, memory the program knows.** Rank 0 staged each chunk through `np.array` and `.cpu().numpy()`, two host copies
  ≺ copies straight between the memory-mapped files and the region rings; the sum ring one block deep, not two; the
  tanh output a buffer held from load (`O`), not made each block.
- **C5, as the review cites it ("known buffer depth").** No bound but the chunk's full drain ≺ the host held to the
  input ring's depth by a mark a block, waited two blocks back (`Pace`, `:83-98`), so a block stays queued.
- **Quantities.** old ≺ new: the block, both rings' depths, the host's wait on the ring and the bytes a step's two
  collectives move (and their rate) are printed beside the parts, rows/s, steps/s and the step's and crossing's times.

### 11. What the host issues a step

- **C4, waits between steps.** A host wait every chunk on all of the GPU's work ≺ none between steps; one wait a block,
  on the mark two blocks back.
- **D10, host calls a step.** new ≺ old: four calls a step (the broadcast's issue, its wait, the step's encode, the
  all-reduce) against two. The wait encodes nothing, since the all-reduce before it completes every group issued
  before it (`nccl.h`: "a group without issue completes at once, with every group before it"); it stays as the wait
  at the consumer. The fused crossing (branch `fused-stream`: the step kernel lands the input and the peer's part
  itself, `nccl.h` `ncclMeshFusableRooted`, `ncclMeshMetalSteps`) is the upper pole, and removes the broadcast's and
  the all-reduce's host calls.

### 12. The planner's silent slow path

- **D3 and D10.** `while G > 1 and ...: G //= 2` (old `chain.py:24-26`): a share of 520 ran on one worker ≺
  `workers` (`chain.py:23-33`): G is the largest power of two the cores allow, and a width or share not a multiple of
  8·G is a `ValueError` naming the nearest values that are.
- **D2.** A fallback found inside the planner ≺ a throw.
- **Correctness (the reviewer's finding 1).** The stamp wait polled `lane < G` with one simdgroup, so for G above 32
  (64 workers on a GPU of 64 cores or more, as an Ultra has) workers 32 and up were never waited for, and stage 0
  could read X before they wrote it ≺ each lane polls every 32nd stamp from its own (`chain.metal:21-23`). The fault
  was older than this change, but `workers` now decides which G runs.

### 13. Rows at run time

- **D2 and D11.** RT compiled into each instance, so each row count compiled and timed new instances (old
  `chain.metal:96`, `chain.py:30-32`) ≺ the row count and the two K splits as words of the constant block
  (`chain.metal:115`), the items (row group × column tile) counted at run time (`chain.metal:54-58`), RG the only row
  template argument. One instance per (K, S, G, RG, threads, patience) serves any rows that are a multiple of 8·RG up
  to `row_capacity`. Every barrier stays under control flow uniform over the threadgroup (`ks`, `items`, `per`).
- **D2.** The splits for a row count are made once, at `Step.plan` (`chain.py:97-98`), and looked up by the call. An
  unplanned row count is a `KeyError`; a row count the instance cannot take is a `ValueError` (`words`,
  `chain.py:46-52`).
- **D1.** One kernel for a fixed row count ≺ one kernel for every row count of its row group.
- **D6.** new = old ≺ evidence from the mesh's own calls. RG is still chosen by timing the step kernel alone on the
  rank, as finding 13's brief kept. It is now named (`kernel_alone`, and the doc says what it times), and it is not
  printed beside the collective's times (Quantities: one device's performance ≺ the collective's).

### Evidence

- **D9.** A check of the first steps only, all in the first block, before any slot is reused ≺ two windows checked
  against float64: the first steps, and the first steps of the last block, seeded from rank 0's own output for the
  step before (`checked`, `:143-151`). The second window reads slots the ring has used before.

### The configuration and the doc

- **D2.** `rows`, `steps` and `chunk` ≺ `row_capacity` and `ring_bytes`, each read where it is used, none defaulted.
  The sample's `ring_bytes` (24 MiB) holds three blocks of 512 steps, so its rings wrap.
- The doc keeps its plain style. It says what each wait is and why it is there, including the one the code does not
  choose (Metal's queue, below).

## Open: where the new form still ranks low

- **The host's bound today is Metal's queue (reviewer's finding 2).** Torch's queue takes at most 64 command buffers not
  yet done, and on today's library each publication, copy and mark commits one, so the host is held about 32 steps
  ahead and `Pace` never waits. `host_microseconds` then includes time held inside torch, and is named and documented
  so, not called issue cost. Rank 0's block-boundary work (making Metal buffers over 8 MiB of file pages) has only
  those 32 steps queued behind it, so cold pages may idle the GPU. Both go once the commit-rule branch commits only
  when the GPU needs a buffer.
- **D10, host calls a step.** A form worth measuring after `commit-rule` lands (the reviewer's): the broadcast of
  e_{t+1} and the all-reduce of p_t as one coalesced group (`backend.py:212-218`), one publication a step, at the
  cost of the input crossing after kernel t, not during it. A whole-block broadcast does not help: completion is in
  ticket order, so the next all-reduce would wait for the whole block.
- **D5 and D10, G.** G is the largest power of two the cores allow, so a 10-core GPU runs 8 workers and leaves 20%
  idle. The change that raises it: G equal to the cores, each worker a ragged tile range [w·T/G, (w+1)·T/G) per stage
  (run-time begin and count in `stage` and the prologue), refusing only widths not a multiple of 8. Not made here: a
  grid that waits on all of its workers must be co-resident, the old power-of-two G leaves cores for the display's
  work, and a 128-threadgroup resident probe on the Mini reset it by watchdog (metal-microbench memory, 10-21). That
  is measured before it is changed, since a node halt ranks below a failed call (rubrics, "Failure across
  interfaces" 4).
- **D4.** tanh(z) is computed twice, as in the old form: in the kernel's prologue for x, and by the drain. The change
  that raises it: the kernel writes tanh(Z) into an output ring as it builds X, and only the last step's is taken
  apart.
- **D1 and D4.** `Pace` and the rings live in the example. The library's `replay` needs the same pacing (the review's
  finding 7); one form in torch_mesh ranks higher.
- **D6.** RG chosen by a solo timing ≺ RG chosen from the chain's own first blocks, one candidate a block, timed
  between the marks and swapped by `Step.use` between blocks: a choice made between calls (C5), on the mesh's own
  calls.
- **C4, the wait's form.** `torch.mps.Event.synchronize` parks on a condition variable (torch `MPSEvent.mm`); a spin
  on `query()` ranks higher, though it is once a block.

## What the first two-rank run must show

On the M5 and the Mini, 8 rows, width 1,024, parts [512, 512], the sample's 4,096 steps, against `2e2a2718`'s 51.6 µs
a step, 26.1 µs a crossing and 10.5 to 13.0 µs of host issue (all inside `_stream.batch`):

- `block` 512, `input_ring` 1024 and `sum_ring` 512: the rings wrap.
- `step_microseconds` at or below 51.6 once `commit-rule` has landed (on today's library, expect near the 94 µs of the
  unbatched form, or above it). `aggregate_rows_per_second` is 8 × 10⁶ over it: 155,000 at 51.6 µs.
- `crossing_microseconds` at or below 26.1, the all-reduce now timed with no region around it.
- `host_microseconds` below `step_microseconds`, and `paced_microseconds` above 0 once the queue no longer binds: the
  host is ahead and waits only on the ring. Higher than 10.5 to 13.0 µs is expected, by the broadcast's issue and wait.
- `check_max_difference` for windows 0 and 3584 of the same size as `2e2a2718`'s first-steps reading. For the same RG
  the kernel's arithmetic is unchanged (the K splits come from the same formula, now as words).
