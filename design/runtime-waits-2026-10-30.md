# Every wait, sync and guard a transaction pays at run time, 3f2a90b7 against main (2026-10-30)

Phase correction from the ordinary prepared callers: the host range/frontier
walk described below runs while the function is realized and encoded, before
`MetalScheduleEnqueue`. Enqueue consumes compiled submission records. The GPU
and bridge rows describe execution work; the host rows describe encoding work
per declared issue. They must not be read as a claim that each invocation of a
prepared function rebuilds its page dependencies. The historical measurements
below do not establish a latency cost for that preparation work. The remaining
compiler handoff is tracked in
[Compiler followups](compiler-followups-2026-10-30.md#dependency-handoff-through-command-lowering).

Found by reading the issue path, the kernels and the bridge at 3f2a90b7 and at main (b2fe8bf3), not by running and
counting. Preparation (declaration, compile_group and what it calls) is left out: it does not run per issue. A
transaction is one physical piece: a SEND entry, its posted receive, its page units (16 KiB, block 1, so 4,096 for 64
MiB) and the combine or copy that consumes it.

## Per transaction

| Layer | Per unit | Per entry | Per combine chunk (4,096 elements) | Per issue |
|---|---|---|---|---|
| Bridge (mesh-link.c; identical in both) | sender: 1 + n readiness-word loads (re-read each pass while the frame gate is closed), 1 gate test, 1 `post_send`, 1 release store no other thread reads; receiver: 1 + n gate-word loads (gated entries only), 1 `post_recv`; each side 1 completion (a 10-way lane scan the wr_id already answers, a status test, an opcode test the receive CQ makes redundant) | 1 entry copy, 1 ring slot | – | each pass: 2 `poll_cq` of 16, about 13 thread-local loads, 4 per held stream |
| GPU (kernels identical in both) | – | SEND publication: 4 system fences, 4 threadgroup barriers, about 6 atomics; COPY consumption: 6 system fences, 3 barriers, about 8 atomics, claimable only once the whole entry has landed | W + 5 system fences (one per thread of the 1,024 after the landed wait), 2 device fences, 4 threadgroup barriers, about 10 atomics, a summed publish even when the prefix did not move | each present: 2 polled waits (ring room, previous present), a publish; the pool's every worker until the last cursor ends |
| Host, 3f2a90b7 | – | – | – | 3 + copies + gates encoder barriers, unconditional (begin, after the presents, after the pool, after each direct dispatch); 3 `group_lock` takings on the torch path |
| Host, main | – | records: a published SEND 3, a landing 1 on its present (uncoalesced where ungated) + 5 per run (land) or 2 (spin) | – | encoder barriers only where declared ranges meet (typically 2 ungated); `MetalEncoderOrder` once per direct dispatch and once for the pool, 6-7 frontier passes over its records; 4 `group_lock` takings |

An all-gather transaction costs about 10 system fences, 7 threadgroup barriers and 14 atomics on the GPU whatever its
size; an all-reduce adds its combines' chunks, about 3,300 a rank at 64 MiB (the two-phase form), each W + 5 fences.

## What changed, and the 64 MiB rates

The bridge and the kernels are the same in both revisions; c56f21a9 changed only what the host does per issue:

- removed: the unconditional encoder barriers (begin, after the presents, after the pool, after each direct
  dispatch), and the per-forward `lease_offset(comm->sums)` under `attach_lock`;
- added: every access a dispatch makes is declared as a byte range and ordered against a frontier
  (metal-recording.m MetalEncoderOrder) to place the barriers. For each record of region-backed storage, on each of the
  frontier's passes, MetalPageAccesses makes a page table of the address (ncclMeshPageTableCreate: `attach_lock`, an
  allocation, and `lease_runs` over the whole lease, one iteration per 16 KiB: 4,096 for a 64 MiB operand), reads its
  backing twice into a new NSMutableData and destroys it, to translate one range to physical pages. A pool dispatch
  carries every entry's records, so the work per issue is about records x 7 x (lease bytes / 16 KiB), inside
  `group_lock`, on every issue, of facts fixed when the declaration bound the pages. The root frontier's reads are
  erased only by writes, so it also grows with issues.

Measured once each, slowest rank, on all five (us a call at 64 MiB):

| | 3f2a90b7 | main (51a6ff10) | compiled barriers, a barrier after the pool | compiled barriers, the operands declared instead |
|---|---|---|---|---|
| all-gather | 5,630 | 6,569 | 5,631 | 6,570 |
| exchange | 5,633 | 4,696 | 5,634 | 4,697 |
| all-reduce | 7,511 | 9,390 | 9,385 | 9,393 |
| reduce-scatter | 7,508 | 6,574 | 6,574 | 6,576 |

The per-record page walk above is real work on every issue, but it is not what moved these rates: removing it (the two
right columns) changes no rate. The barrier after the pool does: without it consecutive calls overlap, which the
exchange gains by and the all-gather loses by. The all-reduce's 1.9 ms is neither; it came with c56f21a9's other changes
(its transfer order for every group, the summed word a receive) and is read next.

## The fix, in order

1. Done: a group's barriers are fixed by its dispatches' kinds (nccl-mesh-metal.m metal_order: before its first present
   and every other direct dispatch, before its pool), its accesses make buffers resident and order nothing, and in place
   of a barrier after the pool it declares each call's operands once to its caller's schedule (metal_boundary: two
   ranges a call). The frontier's per-entry records, their page walks and `attach_lock` at issue are gone.
2. Next: those two ranges' physical pages are fixed at preparation; computing them there removes the last page walk.
   Whether consecutive calls overlap is a choice the schedule can make per pair of groups (the exchange gains by it, the
   all-gather loses), as design/collective-lowering-2026-10-30.md's form A says.
3. Smaller items the readings found, each self-imposed: the bridge's per-unit release store of a private counter
   (mesh-link.c:240), its 4,096 request-depth test that cannot bind (L224, 254), the lane scan (L277) the wr_id already
   answers, one `post_send`/`post_recv` per unit where a chained work request would do, completions reaped once a pass
   after all posting (L380-383); the kernels' whole-entry gate for a COPY consumption, re-checking `need` and the
   dependencies at every chunk claim, the summed publish of an unmoved prefix (nccl-mesh-kernels.h K:300-307, 371-379).
