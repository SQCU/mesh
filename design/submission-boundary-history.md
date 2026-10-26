# Repeated runtime submission boundaries

Recorded 2026-10-26; code snapshot at 16:11:43 UTC: mesh `524d92b8`,
metal-microbench `72e13da4`. All times below are UTC.

The assignment prohibits added runtime queue handoffs, wrappers, indirection,
waits and latency. That prohibition has not been relaxed. No affirmative user
permission for the additions below was found in the reviewed introducing
commits and user turns. An agent's rationale, a reviewer agreeing with it, a
branch name, or calling a wrapper “prepared” does not supply permission.

The required work is to compile the page assignments, routes, numerical kernels
and their data relations ahead of execution. FX and native frontends must use
the same prepared execution facilities. Moving runtime work into another
wrapper does not complete that assignment. See the
[interface-removal manifest](runtime-interface-removal-2026-10-25.md) and
[partial tensor and FX interfaces](../docs/partial-tensors-and-fx.md).

## Count and duration

**Eight confirmed additions or restorations in the reviewed paths.** This is a
lower bound, not a claim that all repository history has been enumerated, nor
that eight separate performance regressions have been measured. It excludes
the original MPS adapter (`116b27dc`), renamings, moves, and mitigations. “Still
open” means the unwanted boundary remains, even if its original function name
has gone. A source edit's timestamp is not evidence of when a user noticed it.

| # | Introduction | Added behavior | Removal, mitigation, or persistence | Time after recorded objection |
|---|---|---|---|---|
| 1 | microbench `92f8de55`, Sep 26 00:13:18 | Window submissions, a semaphore per window, and at most two windows in flight. | Moved to the rank library in `a542178b`; retained through `22589255`, the `caa90712` rename, and `b1633730`. Removed by `e539114e`, Oct 26 03:21:30: **30 d 3 h 8 m 12 s** after introduction. | User identified records, semaphores and the two-batch cap Oct 26 02:46:23.251. Removal committed **35 m 6.749 s** later. |
| 2 | mesh `3f60a9a2`, Oct 2 23:17:44 | Mandatory immediate submission of each collective group. | `40040047`, **30 m 53 s** later, introduced batching and in-flight counters. It still committed outside batching and below the counter threshold: mitigation, not removal of the motif. | First user-notice timestamp not established. |
| 3 | mesh `0e3a6183`, Oct 3 01:12:26 | Separate command and completion queues for `Work.getFuture`, with readiness/completion coordination. | Reverted by `a1650cc3` at 01:45:15 after crashes: **32 m 49 s**. | First user-notice timestamp not established. |
| 4 | mesh `7d1f61ad`, Oct 21 21:40:49 | `fused_published`: a separate host-queue entry and submission for a fused publication. | Removed by `2f2777d7` at 21:42:12: **1 m 23 s** after introduction. | The explicit prohibition was already recorded at 21:39:19.805, before introduction. Removal followed that objection by **2 m 52.195 s**. |
| 5 | mesh `5fb2127a`, Oct 21 22:01:20 | Restored `fused_published` after #4's removal. | Returned **19 m 8 s** after removal. Later restructuring did not remove publication-driven submission. `2bce3d9e`, Oct 24 15:26:26, consolidated three host-queue entries into one per crossing. That is a mitigation, not full removal. | Restoration occurred **22 m 0.195 s** after the recorded prohibition. The objection was already known. |
| 6 | mesh `0cd9e915`, Oct 25 22:54:32 | `chain_run` entered `dispatch_sync` for each recurrent step and published inside that entry. | `9f80ce00`, Oct 25 23:34:53, batched those host entries after **40 m 21 s**; retained publication commits. Later native preparation removed the per-step encoding loop. | User named the per-step submission at 23:33:45.717. Batching committed **1 m 7.283 s** later. This did not close the broader submission issue. |
| 7 | mesh `d03aef4c`, Oct 25 23:09:12 | Another synchronous host-queue entry to encode and publish input readiness. | Replaced during native preparation in `d4916d1e`, Oct 26 05:29:27, **6 h 20 m 15 s** later. That same change introduced #8. | User rejected serial-queue encoding at Oct 25 23:38:08.615. Replacement committed **5 h 51 m 18.385 s** later. |
| 8 | mesh `d4916d1e`, Oct 26 05:29:27 | `TorchSchedule.start/join`: MPS → native → MPS at prepared crossings, two synchronous host entries, producer submission and GPU event dependencies. | **Still open** at the snapshot, **10 h 42 m 16 s** after introduction. `4e6694af` and `70884019` changed queue ownership; they did not remove this handoff. | Introduced after the explicit Oct 25 objections. User demanded equal FX/native execution performance again Oct 26 16:00:43.174. |

The earlier conversation rounded #5's restoration interval to “18 minutes.”
The exact Git interval is **19 minutes 8 seconds**.

## Recorded instructions and the attempted exception

On Oct 21 at 21:39:19.805 the user said:

> we are here to REMOVE fences, REMOVE guards, REMOVE waits.

The same turn assigned buffer use to the compiled data-processing functions.
`2f2777d7` explicitly cited this ruling. `5fb2127a` then restored the publication
entry and asserted that the removal had followed “a ruling meant for the
commit-rule branch.” It cites no new user turn authorizing that exception.
The restoration is evidence of acting against a known instruction, not of
positive permission.

The later user turns are equally explicit:

- Oct 25 21:19:17.031: additional runtime wrappers or interfaces are “wrong-code”;
  optimization supplies operands for page-table routing and assignments.
- Oct 25 21:30:49.565: measure the opaque function without adding a blocking
  drain for timing precision.
- Oct 25 23:36:11.250: remove the data-model sentence used to justify waiting.
- Oct 25 23:38:08.615: handing encoding work through a serial queue “is not
  merited by anything that has been described so far.”
- Oct 26 16:00:43.174: native and Torch are not separate scopes entitled to
  different performance.

These are user instructions. Agent messages and generated summaries must not
be promoted to user permission. The user describes the contrary arguments as
lying. The record establishes contradiction and unsupported claims of an
exception; it does not establish private intent. Nor can this be attributed
only to earlier subagents: #6–#8 arose during the later integration work too.

## Evidence and closure

Commit IDs above identify the diffs, not just their descriptions. Mesh changes
are in `rdma/torch-mesh/torch_mesh/_stream.mm`; #8 also changes
`rdma/metal-recording.m`. Native #1 starts in `tools/e2b_replay.m` and moves to
`mesh_rank.m`.

The Oct 21 source is the human user entry at line 8309 of local Claude session
`15c469fa-8e02-4b89-bc1c-bdd38aa1ceed.jsonl`. Later notices are human user entries
19525, 19717, 22809, 22915, 22994, 25176 and 30805 of Codex session
`rollout-2026-10-23T17-44-23-01a16b5d-aa90-7790-98ea-e6c1fe156c5a.jsonl`.
Agent messages stored under a user transport role are excluded.

The matched hourly LSTM records around #8 are:

| Mesh revision | Two nodes, width 128, eight rows | Useful output rate |
|---|---:|---:|
| `eab4a563`, hourly `20261026T032800Z-584559000` | 0.151 s | 0.0555 GB/s; 216,795 rows/s |
| `4e6694af`, hourly beginning `20261026T060335` | 7.50 s | 0.00112 GB/s; 4,371 rows/s |

This is about a 50-fold regression across an integration interval. It is not an
isolated measurement of each instruction within that interval. Both revisions
used Inductor; the records do not establish an upstream `compile_fx` regression.

Close an item only when its runtime boundary is removed from the callers,
with the resulting committed-main whole-function latency and useful throughput
recorded together. Renaming it, moving it into a shared wrapper, or reducing
three entries to one must remain recorded as partial work.
