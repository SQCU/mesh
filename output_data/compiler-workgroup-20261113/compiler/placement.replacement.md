# Approved narrow replacement: finite shard intersections

Style: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights).
Assignment authority: [operator correction](../../../design/compiler-workgroup-2026-11-13.md#correction-to-the-first-assignments).

Root authorizes this single unit; executor grants the exact new mesh_graph.py arithmetic hunk. Do not resume the old broad placement assignment. Codex has chosen the algorithm, record and consumers below.

## Requirement and authoritative sources

Compilation composition items1–3 and unequal-shards items2,3,8,9,12: preserve typed source/destination geometry through collective expansion; do not enumerate elements to rediscover views. schedule-emit/schedule_emit/placement.py is the stage model: explicit assigned values become one Send/Receive per receiving processor. MPI vector/subarray datatype semantics supply strided domains; the algorithm here is exactly the finite congruence intersection proved below. Modular inversion is Python's existing integer `pow(x,-1,m)`, not a new Euclidean implementation (https://docs.python.org/3/library/functions.html#pow; CPython3.12 Objects/longobject.c long_invmod).

## Exact producer, result, consumer

Add only `mesh_graph.intersection(a, b)` for two Python `range` values. Return three Python ranges: `(common, left, right)`, where `common` lists the common global coordinates ascending, and `a[left[k]] == common[k] == b[right[k]]` for every k. All three have equal length. Empty inputs/disjoint congruences/disjoint bounds return three empty ranges. Positive and negative range steps retain exact local index maps. No list, set, tensor or extent-sized materialization.

For each existing source/destination shard pair in placement redistribution, compute this record ONCE. Transfer construction consumes its left range as the source slice, its right range as the destination slice/scatter, and len(common) as the count. Do not call range.index, enumerate common coordinates, reconstruct slice strides from first/last/count, or rerun an intersection to obtain another field. Ordinary contiguous redistribution uses the same records/counts as striped redistribution. Keep zero destination behavior and all existing numerical/sharding choices unchanged. The pair matrix may scale with source-shards × destination-shards, never with tensor extent.

Delete `placement.overlaps` and its extent-sized `sorted(set(range(...)) & set(range(...)))`; remove the matching per-consumer `range.index`/stride reconstruction in `reshard`. This is the complete replacement unit. Do NOT rewrite or wrap reshard/across/partitioned, move them to another module, repair unrelated placement code, or claim whole placement is now replaced.

## Algebra to implement

For nonempty ranges normalize each to its minimum coordinate and positive step p/q; take low=max(minA,minB), high=min(maxA,maxB). Let g=gcd(p,q), d=minB-minA. Intersection is empty if low>high or d%g!=0. Otherwise solve `(p/g)*j = d/g (mod q/g)` with existing modular inverse (when q/g==1, j=0). Set base=minA+p*j and period=(p/g)*q. The least admitted point is base+ceil((low-base)/period)*period, with integer arithmetic; count=max(0,1+(high-first)//period). `common=range(first, first+count*period, period)`. For each original r, its first local index is `(first-r.start)//r.step` and local index step is `period//r.step`. Those are the emitted left/right ranges, including signed step. Bounds and congruence prove every returned element belongs to both inputs; period is the least common positive step, so no common point is omitted. Use existing math.gcd/pow; no custom solver or graph algorithm.

## Files and evidence

Own only /Users/mdot/dox/mesh/rdma/mesh_graph.py new intersection+math import and compiler/mesh_compiler/placement.py exact overlap/consumer hunks. Executor confirmed no overlap. No ir.py, driver, schedule, lowerer, configuration, tests or harness files. Explanatory prose belongs in status/docs, source only documentation citations or formal algebra.

Use /Users/mdot/.venv-mesh-uv/bin/python with TORCH_DEVICE_BACKEND_AUTOLOAD=0 for private syntax/import checks. Do not run hardware/client experiments yourself or search old outputs for commands; root coordinates existing TP uneven/striped client. A compact ephemeral arithmetic check is allowed only to catch off-by-one/sign/congruence errors beyond client metrics, report it rather than adding maintained tests. No benchmarking, new evaluator, broad source inventory, staging, commit, reset or stash. Read placement.inbox.md before edit batches and report exact changed lines/checks in placement.status.md. Stop after the complete live handoff is privately checked and READY.
