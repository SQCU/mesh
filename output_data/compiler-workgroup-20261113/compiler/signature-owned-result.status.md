# Owned signature constructor: worker status

Style: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights).

Session `64ddaf6e-3a70-4b6a-9d0b-50691a84c6cd`, model `claude-opus-5-5`. Assignment:
[signature-owned-result.md](signature-owned-result.md). No staging, commits or shared builds. **READY**, held for
review.

## Change

`rdma/mesh_graph.py` only. The construction lines of `signature` move unchanged into
`build_signature(processors, tasks, dependences, operations)`, which returns the owned `Signature` that
`mesh_graph_signature` derived once. The caller releases it with `free`. `signature` now calls `build_signature`,
then runs its existing `try: forms + record` / `finally: free`. Error behavior is the same: `_check` raises before
anything is returned. No other wrapper, algorithm or schema.

## Checks

Run against the built `rdma/libmesh.dylib` with the one-off `/tmp/planner-archive-check/owned.py` (not maintained),
using the operands of two existing canonical records, `compiler/emitted/moe_three_processors.archive` and
`conv_cpu_mps.archive`, read by `from_record`.

| Check | Result |
|---|---|
| `python3 -m py_compile rdma/mesh_graph.py` | clean |
| `signature()` equals the previous body, run inline with the same operands | equal for both records |
| `signature()['record']` equals the on-disk `signature.record` bytes | equal for both |
| `build_signature` result's forms and record equal `signature()`'s | equal for both |
| `compose(build_signature(...), from_archive(archive(...)))`, then `free` of each | moe: 22 tasks, 24 operations, 12 dependences, 12 synchronized edges; conv: 22, 22, 18, 14; each composed archive roundtrip is byte-identical after the inputs are freed |
| Invalid operands (a dependence to task 5 of 1) | `OSError` errno 22 from the previous body, `signature` and `build_signature` alike |

**Limits.** These records hold no BRIDGE_START, so the check doesn't exercise compose's aliasing with a planner
signature. Composing a signature with its own archive is an ABI check, not a program composition. The executor's
consumer (composing with `from_archive(plan.signature)`) has not run.
