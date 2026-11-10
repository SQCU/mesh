# Memorandum: the requests written before the example

2026-11-10. The example is `~/schedule-emit`; its data flow is quoted in [data-flow.md](data-flow.md).

## The turns

The operator, 2026-11-10 (UTC), of the example, once its review was finished:

> "It made the cross-device copy an explicit node on the sending processor, before the send, so the scheduler sees it as a function like any other." behaviorally appropriate. okay. we now have a reference program which counterexamples whatever wide-spanning and comprehensive pathologies are in the 'existing code' in the mesh lib and mesh repo.

> if the data flow has the wrong directions at any point, we don't need to read any part of it to totally replace it, since it has no contents which relate to the requirements for any function which has the wrong causal direction. isn't that cool?

> for any change requests outstanding before that separate repo was made, what structural chagnes require the total reappraisal and total rewriting of the change request documents with respect to the strictly available xample?

## Which requests

The example was first committed on 2026-11-09 at 12:53 (schedule-emit e6508cc). First committed before it:

| request | first committed |
|---|---|
| page-queue | 11-03 17:33 |
| disjoint-pages | 11-03 20:49 |
| plan-to-bridge | 11-04 14:14 |
| aot-compilation | 11-04 14:23 |
| problems-of-14b9a757 | 11-06 02:50 |
| torch-capture | 11-06 14:45 |
| torch-source-lowering, items 1 to 3 | 11-07 21:59 |
| integration-finishing | 11-08 04:22 |
| synchronization-signature | 11-08 16:18 |
| compilation-composition | 11-08 17:53 |

Written after it: unequal-shards (11-09 18:56) and torch-source-lowering items 5 and 6 (11-09 19:37 and 21:02).

## What the example establishes, and what the earlier requests assume

| The example (data-flow.md) | The earlier requests |
|---|---|
| `compile_module` runs `export`, `place`, `on_devices`, `schedule`, in that order, once. | torch-capture, item 9: "every node runs its captured program by replay". integration-finishing, item 2: "A node's replayed commands in the completion-time order". compilation-composition, requirement 6, names a second solve in `MetalProgramPrepare` as what it replaces. |
| `send(value: Tensor, channel: int)` and `recv(channel: int) -> Tensor` are nodes `place` adds to the one graph. | plan-to-bridge R3 and R6: transfers are `struct mesh_declared` records, and `mesh_plan` "builds the task graph from the IR". |
| One value, `Scheduled`, is returned by `compile_module`; `lower(scheduled)`, `Program(scheduled, functions)` and `save(directory, scheduled)` each take it. | plan-to-bridge's title: "what the plan gives the bridge". The schedule is made by one function and handed to another program as its own artifact. |
| `runtime.py` imports `ir` and nothing else of the program. | torch-capture items 2 to 6 and disjoint-pages C3: waits, phases and counts that are worked out when a recorded program is prepared. |

## What follows

Each requirement of the ten requests above is restated against data-flow.md: the stage function it belongs to, the type that function takes and the type it returns. A requirement that exists only where a program is recorded and replayed, or where a schedule is made apart from the program and handed over, has no stage in that data flow and is not restated.
