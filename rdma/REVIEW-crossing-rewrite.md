# Review: the two-rank crossing, rewritten (branch crossing-rewrite, 2026-10-21)

Static review. Nothing here was built, compiled, installed or run, and the pair was not touched. The operator:
"source is written and rubric judged rather than built or incrementally run tested". The regression this answers is
metal-microbench `output_data/crossing-20261021/regression-review.md` (causes 1 to 4). Line numbers are this
branch's.

Every file changed is collective code (kind C): libnccl-mesh (`nccl-mesh.c`, `nccl-mesh-metal.m`,
`nccl-mesh-metal.h`, `nccl-mesh-host.c`), the schedule compiler and advisor (`mesh-collective.c`, `mesh-plan.h`,
`mesh-advise.c`), the device functions (`mesh.metal`), and the contract text (`nccl.h` comments, `NCCL.md`). No
public declaration in `nccl.h` changed. The pairs below use rubric C and the Quantities, and "Failure across
interfaces" (design/rubrics.md). `A ≺ B` means A ranks below B.

## What the crossing does now

An eager two-rank all-reduce of 3 KB, in place, on an MPS tensor:

1. `launch_metal` plans the call. Its program has no scratch (below), so no copy is encoded.
2. The group lies on one channel, so `complete()` calls `complete_track()` (nccl-mesh.c:2833).
3. That encodes two dispatches into torch's encoder: `send_small` (copy the operand into the slot, one release
   fence, store the ready word), then `land_f16` (thread 0 polls the landed word, a barrier, every thread combines).

Compiled (`COMPILER=mesh`), the issue encodes `send_small` (`issue_group`, unchanged), and the completion encodes
`land_f16` alone. No track is recorded and no `tracks` dispatch runs in either case.

## Rewrite 1: a group on one track is plain dispatches

- **Where.** `one_track` (nccl-mesh.c:2808) reads each in-flight group's `lone` channel. `lone_of`
  (nccl-mesh.c:2469) sets it once, in `launch_metal` after the group is scheduled (nccl-mesh.c:3318). It is a
  function of the group's laid positions alone, which both ends lay alike. So the choice between the two forms is
  made where the group is planned, from its declared structure. Nothing at run time chooses (rubric D2/D3's point,
  read as C3 here).
- **The order.** `complete_track` walks one channel. It publishes the next position when the host's order allows:
  the consumption LAG back is already encoded, and a SEND's receive-before is done. Otherwise it consumes the next
  position, except one whose publication reads in place what it writes and is not yet published. If neither can go,
  it returns a failure value ("admit no order"). A well-formed program on one track never gets there; a degenerate
  one (a SEND waiting on a receive at or past its own first position, which the advisor's depth check names) fails
  there rather than send what was combined into.
- **The credit.** This rank publishes position p only after it consumed the peer's p − LAG. The peer published that
  only after consuming this rank's p − 2·LAG. So at most 2·LAG of this rank's positions are unread at the peer. The
  ring holds 2·LAG + 1 runs of a position's largest run (`mesh_wire_of`: 15 small units a run × 17 = 255 ≤ 256;
  30 blocks × 17 = 510 ≤ 512). Checked from source; it fits.
- **Multi-track groups keep the tracks.** Two or more channels, or two communicators, take 264ce849's path as it
  was. The picker, its two cursors, its credit, its dependencies and its expiry are unchanged, with two exceptions
  forced by the other changes (below: the consumption's `need`, and the gate and stop reads).

## Rewrite 2: no scratch copy for the two-rank direct exchange

- **The program.** `mesh_compile` (mesh-collective.c:590) sends a node's own operand from `buffer[u*n+u]`. That is
  buffer 0 (the operand) where the node sends its operand to one node only and forwards nothing; otherwise it is
  buffer 1 (the copy). On a connected map this holds only for two nodes. Its `scratch` is then 0, so `launch_metal`
  takes no scratch and encodes no `copy_plain`. Trees and rings, and any node that sends its operand to two nodes or
  more, keep the copy. The in-place read is the program's, not inferred where it runs.
- **Why it is safe.** On two nodes, a node's SEND and REDUCE of chunk j are both at round j, of the same bytes. They
  lie at the same positions of one channel. So each element is read by its publication before its consumption writes
  it, if every executor publishes a position before it consumes it:
  - one track: by the stream's order (`complete_track`);
  - tracks: a consumption whose position's publication reads in place what it writes waits for that publication
    (`sends_what_lands`, nccl-mesh.c:2155; recorded as the item's `need` at nccl-mesh.c:2912; checked by the picker
    at nccl-mesh-metal.m:269 and the host executor's `item_ready`, nccl-mesh-host.c:432);
  - the host path: `run()` consumes such a position only once it is published (nccl-mesh.c:2204).
- **The advisor knows the order too.** `mesh_group_advise` adds the same edge (mesh-collective.c:1311), and its
  write-after-read check is now piece by piece (mesh-collective.c:1380), searched only where a later receive meets
  the SEND. The old step-level check would warn on every
  in-place exchange bigger than one position. mesh-advise's mutation M4 is rewritten: the old M4 (a round-0 receive
  into the part that round's SEND sends) is now exactly the safe in-place pattern. The new M4 receives the whole
  operand, so its pieces meet the SEND's at later positions, with no order between them.
- **The fused path** needs no special case: `launch_metal` requires `buffers == 1` (nccl-mesh.c:3245) and
  `ncclMeshFusableRooted` requires `!program->scratch` (nccl-mesh.c:3472).

## Each mechanism on the two-rank path, by owner

The bridge owns its devices, queue pairs, rings, page table and the region's registration (design/transport.md).
None of the GPU crossing path is the bridge's. The library's contract is NCCL's: the caller touches no buffer of a
collective in flight and issues the same collectives in the same order on every rank; the library delivers correct
bytes. The data function owns its own buffers and their order.

| # | Mechanism (before) | Owner | Contract term that decides | Now |
|---|---|---|---|---|
| 1 | System fence before every poll (`mesh_wait_bounded`, mesh.metal) | library | correct bytes: a `volatile coherent(system)` load already reads the word where the NIC stores it | removed |
| 2 | System fence after the wait (`mesh_step_await`) | library | correct bytes: the payload is read through `volatile coherent(system)` (`mesh_payload`), after the loop that saw the word (the waiting thread's control dependency) and the barrier (the other threads) | removed |
| 3 | mesh.metal's claim "without it the M4 did not observe the NIC's store: metal-microbench d8566be" | library text | d8566be moved the fence to after the observation and recorded no observation | removed |
| 4 | Release fence before the ready words (`mesh_publish`) | library's own release | the payload stores by many threads become visible before the word the bridge and NIC act on | kept; now one a publication, not one a unit (`mesh_step_release`) |
| 5 | Scratch copy before the two-rank all-reduce (16ac3c85) | library | correct bytes: the publication stores the operand into the slot, so the slot is already the copy | removed (rewrite 2) |
| 6 | Credit: a publication waits for the position LAG back to be consumed | library's own slot ring (transport.md §14: receives are posted for the link's life, whatever the consumer read) | the host's encode order keeps 2·LAG positions unread at most; the ring holds 2·LAG + 1 runs | one track: host encode order only, no GPU check. Multi-track: the picker's run-time check kept (below) |
| 7 | Stop-word read in every publishing kernel (a8952de9) | library | NCCL's: a revoked or destroyed communicator's work is the caller's to finish or abort; the bridge posts unit u only when its ready word reads 1 + u/units, so a late kernel's older cycle cannot make it post anything | removed |
| 8 | Gate read at every kernel's entry, and gate and stop reads every 4,096 polls in a wait (9c4248e2) | library | failure as a value: the record is kept and the host reads it; the host refuses a failed communicator (`comm_failure`); a wait on a dead peer ends at its bound | removed |
| 9 | `tracks` picker for a one-track group (264ce849) | library | one track's order is the one serial order every rank builds alike | removed (rewrite 1) |
| 10 | The wait's bound | library's own wait | a GPU spin with no end halts the node (the Mini's watchdog reset, 10-21) | kept, as a count of polls that reads nothing else |
| 11 | A failed wait that loses the claim awaits the claimant's record (up to its bound) and stores it again (`mesh_step_check`) | library | it existed only so the loser's threadgroup would read the gate closed (row 8) | removed: a loser records nothing and waits for nothing |
| 12 | The counted all-to-all's `*arrived` word: 1, or 2 where its counts pass its layout | library, the call's own outcome | nccl.h `ncclMeshAlltoAllCounted`: `*arrived` is 2 where the call failed. Row 8 used to stop the end's dispatch after a failed wait, which left 0 for the host to turn into 2; now it runs | `METAL_COUNTED_ARRIVED` also reads failure word [2] once and stores 2 where a wait recorded a failure (nccl-mesh-metal.m:143; host executor nccl-mesh-host.c:224). A read that decides a reported value, not a guard: nothing is skipped |

Kept because it guards the library's own state, not numbers: `mesh_step_enter`'s store of the dispatch's sequence
into the Metal 4 argument ring's progress word. `record4` reuses a record only once its dispatch began, so a record's
addresses are never overwritten while the GPU still reads them. It is a store by one thread, with no read.

## Ranked pairs

**Rewrite 1.**
- C4 (asynchrony): a one-track group through the `tracks` interpreter (thread 0 checks the landing before the
  publication; each pass a fence, up to four coherent loads and two barriers of 1,024 threads) ≺ the publication's
  dispatch, then the consumption's, the wait one load a poll (nccl-mesh.c:2833).
- C4: the host's work for each completion (an `NSMutableSet`, the track records, `ring_take`, a `useResource` for
  every buffer) ≺ none.
- C7 (symmetric knowledge): a thread on the GPU choosing among one track's items ≺ the positions both ends lay alike
  (`mesh_group_lay`), encoded in that order.
- C3 (form as data): a form inferred at completion from run-time cursors ≺ one read from the group's schedule where it
  is planned (`lone_of`).
- C2 (invariance): two completion forms ≺ one form for every group. The new code ranks lower here. It is weighed
  against C4 and the Quantities: one track is the special case of tracks whose order is trivially shared, and both
  forms run the same per-position functions (`publish`, `consume`) and the same kernel bodies.
- C1 (provenance): equal. NCCL runs one channel's sends and receives in its program's order; 264ce849's
  per-channel progress is NCCL's per-peer form for several peers.

**Rewrite 2.**
- C4: a copy dispatch on the chain from landing to send ≺ none.
- C3: an in-place read decided in libnccl-mesh at run time (2079a74's rank count; `plan_direct`'s refusal of an
  in-place call) ≺ the program's own buffer index, which the executors run.
- C2: a two-rank special case in `launch_metal` (the fused path's copy bypass) and in `ncclMeshFusableRooted` (a
  scratch of 1 allowed for the all-reduce) ≺ one rule: the program has no scratch.
- C4, the tracks: a copy made so that a consumption may run before its position's publication ≺ an edge from that
  publication to that consumption, the program's true write-after-read, which binds only where a SEND reads in place
  what its position's receive writes.
- C8: equal. The same sums, in the same order.

**Removals (table rows 1 to 8).**
- C4: a fence before each poll ≺ a poll that is one coherent load.
- C4: a fence after each landing ≺ none, the payload read coherently after the barrier.
- C4: a gate load and a barrier at every dispatch's entry, and a stop load and a barrier in every publishing
  dispatch ≺ none.
- C4: a fence for every unit of a run ≺ one fence a publication.
- C4: a losing failed wait that waits up to its bound for the winner's record ≺ one that returns.
- C6 and "Failure across interfaces" 1 and 2 (failure is a value): a counted all-to-all whose `*arrived` reads 1 over
  counts that never landed, once nothing stops its end after a failed wait ≺ one whose end reads the failure record and
  reports 2. One coherent load at the counted call's end, off the two-rank all-reduce's path.
- Text ranked below the operator's (design/rubrics.md, "How to read them"): a source comment citing d8566be for an
  observation it does not record ≺ no such claim.
- C6 and "Failure across interfaces" 3 (failures are not contagious): **the new code ranks lower here.** With no stop
  read, a kernel of a destroyed communicator that is still queued can write a stream that `ncclCommDestroy` gave back,
  and the next holder may already have written that slot. That is another communicator's numbers, not the bridge's
  state (the ready word it writes carries an older cycle, which the send thread does not post). `ncclCommDestroy`
  waits only for command buffers the library committed, and for those at most `MESH_PLATFORM_NS`: a committed
  program whose remaining waits each run to their bound outlasts that and can write the same way. The owner is the
  library, and the fix belongs there (below).
- C1: NCCL's own kernels read an abort flag inside their spin loops (every N spins). The removed reads inside
  `mesh_step_wait` were that form, and on a landing within 4,096 polls they cost nothing. On C1 alone the old form
  ranks higher. The operator's ruling of 10-21 decides it: "we are here to REMOVE fences, REMOVE guards, REMOVE waits".
- Quantities (a channel quiet 5 s): each wait still ends within its bound. But after a failure no later dispatch skips
  its wait, so a command buffer with k waits on a dead peer holds the GPU for up to k bounds (about k × 5 s).
  "Failure across interfaces" 4 ranks a halted node below a failed transaction. Whether the GPU watchdog bounds one
  dispatch or a whole command buffer was not read; it must be, before a compiled step of 141 crossings meets a dead
  peer.

**Credit (row 6), the multi-track picker.** The coordinator asked for every GPU-side credit check to go, the
picker's included. It is kept in the multi-track picker, for this reason: a track there has two cursors that move
independently, so the host cannot place a publication after its credit's consumption in one sequence. Doing that
needs the host to encode a multi-track completion in waves (publications to the credit, then the consumptions they
make possible, then the next wave). A single sequence per track instead adds false edges that can deadlock: a
publication blocked on another track's receive would hold a consumption behind it that the other track needs. Waves
are a redesign of the multi-track path, which the operator asked to hold until the multi-rank case can be measured.

## Where this meets other branches

`fused-stream` (another agent) edits the same comment paragraphs of STEPS (`mesh_step_enter`'s) and of `struct
metal_program`, `launch_metal`'s fused checks a few lines from this branch's `buffers != 1`, and nccl.h's fused-crossing
paragraph (this branch changes two sentences there: "no scratch" and the end of "one failure recorded of them all").
No declaration in nccl.h changed here. `commit-rule`'s `_stream.mm` and `torch_mesh/__init__.py` are untouched; this
branch changes one docstring in `torch_mesh/backend.py` and one line of torch-mesh's README, neither of which main or
fused-stream touched.

## The later commits' reasons, kept

- **264ce849, per-channel progress.** Kept for every group on two or more channels, such as the counted all-to-all
  among three or more ranks it was made for. "No rank orders anything another rank must share" still holds: one
  track's order is the positions both ends lay. Two changes reach its picker. A consumption's `need` now names its
  position's publication where that publication reads in place what it writes (rewrite 2 needs it; 0 elsewhere, so
  every other group runs as before). And `track_ready` no longer returns true on a closed gate or a set stop word:
  the bodies no longer do nothing on a closed gate, so a picked item would run its wait to the bound. Now a blocked
  item waits until its track's own bound, records its failure, and the track ends.
- **16ac3c85, a program is data the library only runs.** The in-place read is the program's buffer index from
  `mesh_compile`. Trees, rings and every node that sends its operand to two nodes or more keep their copy. The
  planner's version is 7 (`MESH_PLAN_VERSION`).
- **a8952de9, bound lanes.** Holdings, declarations, duds and `nil_send` are unchanged. Only the stop read in the
  publishing kernels is gone (row 7 and the C6 pair above).
- **9c4248e2, every wait bounded.** Kept: `bound_of` makes every program's bound at most `NCCL_MESH_POLLS`, and every
  dispatch is still bound to its communicator's failure words, which a failed wait records in. Only the gate reads
  that halted later dispatches are gone.

## The transport's failure rules

- **A failure is a value.** A wait at its bound records its cause, position and channel (`mesh_step_check`,
  unchanged). The host reads them after the command buffer (`comm_failure`) and refuses the communicator's later
  groups. Unchanged.
- **Nothing waits more than 5 s on one channel.** Each wait is bounded. Chained waits after a failure: see the
  Quantities pair above.
- **A revoke.** The units its groups declared and did not supply still go as duds at once (`nil_send`), so its peers'
  groups end. Its own GPU waits now end at their landings or bounds, not at the revoke. A wait that reaches its bound
  after a revoke still names the revoke as its cause.
- **Duds** land as bytes, as before.
- **Not contagious.** See the destroy residual above.

## Follow-ups, not in this change

Held by the operator until the multi-rank case can be measured:
- within the tracks, a picker could run statically ready publications straight through;
- the tracks kernel could be built from the group's own op kinds, not a switch over every body;
- the multi-track credit as the host's order (waves), and the picker's per-pass fence (row 1's reasoning applies).

Owed by the library:
- `ncclCommDestroy` gives a stream back only once the communicator's encoded work has ended (`finished` reaching its
  `last` for the programs the library committed, however long their bounded waits take; the Metal 4 argument ring's
  progress word passing its last record, or the caller's command buffer, for work in a caller's encoder), closing the
  destroy residual.
- metal-microbench `metal/layers.metal` tests `mesh_step_enter` and `mesh_step_await`, which now always return true.
  Drop those tests there, then make both functions return nothing.

## Relied on, not confirmed by reading

- A `volatile coherent(system)` load on Apple GPUs sees the NIC's and the host's stores without a fence, and the
  payload read after the waiting thread's loop and the barrier is the landed one. Metal's documented memory model
  does not state the control-dependency ordering. The first run's correctness check (crossing.py's first all-reduce,
  "passed") and the KL of any served program test it.
- Dispatches in one classic compute encoder run one after another (Metal's default serial dispatch type), and the
  Metal 4 form's barrier after each dispatch does the same. The one-track form's order rests on this, as the issue's
  dispatches already did.
- torch registers a backend that is a `ProcessGroup` subclass as the group itself, so crossing.py finds the
  communicator through torch-mesh's `_mesh` resolver.
- The GPU watchdog's scope (one dispatch or one command buffer), for chained bounded waits after a failure.
- That no kernel outside this repository and metal-microbench relies on `mesh_step_enter` or `mesh_step_await`
  returning false.

## The independent review, and what became of each finding

A subagent read the diff against the rubric, statically (nothing compiled). It found no compile problem under
`-Wall -Wextra -Werror` and none in the Metal strings, and confirmed `one_track`, `holding`, `sends_what_lands`, the
picker with `need` 0, the host executor's mirror, the advisor's edge bound, `mesh_compile` for more than two nodes
and with contributors (buffer 0 only where n is 2), the new M4 firing the hazard check (unordered meetings at
positions t + 4 and t + 5), and M1 to M3, M5 and G1 to G5 unaffected. Its findings:
- `complete_track` could consume a position before publishing it in a degenerate program. Fixed: such a position is
  consumed only once published, else the walk fails as a value.
- The counted all-to-all's `*arrived` could read 1 after a failed wait. Fixed (row 12).
- `mesh_step_check`'s loser still waited for the claimant's record. Removed (row 11).
- `ncclCommDestroy` gives streams back after at most 5 s, so a committed program's late kernels could write them.
  Recorded as the destroy residual, owed by the library (below); not changed here.
- "A failed communicator issues no group" was too broad: groups already encoded run as encoded. Reworded in STEPS'
  text; nccl.h `NCCL_MESH_POLLS` and mesh.metal now say the bound is each wait's own.
- Stale revoke text in torch-mesh's README and `ProcessGroupMesh.revoke`'s docstring. Updated.
- The removed per-poll fence's comment cited evidence. Read: metal-microbench d8566be's message moves the fence to
  after the observation and records no observation, so the citation went; the first run tests the poll.
- NCCL.md said a consumption is one dispatch; on the large class it is two. Fixed. NCCL.md cites this file, which is
  committed with it.
- The advisor searched reachability for every SEND position even with no receive to meet. It now skips those.

## What the first build must show on the pair

The GPU path must be dispatched on the M5 and the Mini before this change is called done. 264ce849's message said
"nothing dispatched"; this one is not dispatched either.

Run metal-microbench `tools/torch_parallel/crossing.py` (branch crossing-times) on both ranks, the variables literal
on each rank, `TIMES=1`:

| run | today (10-21) | target (10-02) | must show |
|---|---|---|---|
| eager MPS, `COMPILE=0` | 25.6 µs median | 17.4 µs (p10 16.3-16.7) | at or under 17.4 (3% is noise) |
| compiled, `COMPILE=1 COMPILER=mesh` | 17.8 µs | 12.7 µs | at or under 12.7 |
| eager `DEVICE=cpu` (host path) | 18.8 µs | no 10-02 figure | "passed" on its first call; no slower than 18.8 |

- Every record's first line must say `"passed": true`. That first all-reduce is the in-place two-rank exchange this
  change rewrote, so it checks rewrite 2's order on the Metal path (and, with `DEVICE=cpu`, on the host path).
- With `TIMES=1`, the turnaround (`issued[i+1] − ended[i]`) should fall back toward 10-02's (compiled 4.5 µs on the
  M5 and 3.2 µs on the Mini; eager 5.1-5.5 µs), and `post_to_landing` should not move much (10-02: 4.2-5.8 µs).
- The review's estimate of what this removes is 4.6 to 10.4 µs eager and 3.6 to 8.4 µs compiled, which brackets
  both targets. If a run stays above its target, the turnaround says whether the rest is still on the GPU side
  (turnaround above 10-02's) or elsewhere (the host path, c41dddfa; the bridge).
- `make` with every warning an error, `mesh-advise` (0 warnings over its groups) and `mesh-advise mutate` (10 of 10
  firing) must pass before the pair run. They were not run here.
