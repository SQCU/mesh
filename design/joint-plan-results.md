# Joint plan outputs and ownership

`mesh_plan`, `mesh_plan_local` and `mesh_plan_program` read their original
`mesh_function` and `mesh_value` operands. They produce derived storage in
`mesh_planned.result[f]` and `mesh_planned.root[r]`. A second solve has separate
output storage and cannot overwrite the first function or plan result.

Each function result owns its derived value sections, sharing flags, native plan
address, completion times, task order and the host-write lag computed by the original buffer-bound algorithm. A value result contains the original
complete `mesh_section` output, including its byte extent and page count; it does
not replace the original source value's lifetime, kind, run or state operands.
Local plans retain the supplied function/root arrays and call relation by view.
Native placement reads source properties from those operands and geometry from
the joint result. `mesh_view_values` takes both complete results explicitly.
The original planning, scheduling and placement algorithms remain their producers.

`mesh_plan_local_analyze` and `mesh_plan_program_analyze` return a complete
`mesh_plan_analysis`, including its original source operands, flow results,
transfer groups, call correspondence and storage-copy calculation.
`mesh_plan_local_complete` and `mesh_plan_program_complete` read that result and
produce separate physical plans. The ordinary entrypoints compose those same
functions. Each completed plan retains its analysis; another completion reads
the same analysis and writes its own result. These continuations use the fixed
source function identities captured by analysis. Numerical graph transformations
supply their actual operation correspondence and complete changed demands before
the broader [joint IR composition](../../../compiler-emit/docs/joint-ir.md) can
connect them to physical planning.

## Archives

The common local archive stores the original function values, actors, declared
views and their dimension pairs, weights, dependencies, tasks, preceding edges,
senders and receivers, alongside all derived sections, sharing, completion and
order. It also stores the original roots, call relation, closure and uses, and the
joint call/order/cause/offset results. `plan_arrays_put/get` is the sole source
array serialization implementation and runs once in both local and network
archives. The common Analysis section stores each complete flow header and its
produced arrays, transfer groups and source/instance correspondence. Original
flow producers retain their position and covered-block counts alongside those
arrays. Network archives add native states, page rows, signature and extents.

The loader produces function descriptors and repoints declared-view dimension
pairs into the retained archive. Actor arrays and relations remain complete
archive-owned results. `mesh_plan_load`, which receives an existing source
function, keeps loaded actors on `result[f].actors` instead of overwriting the
source. A subsequent continuation reads those retained actors and the archived
flow timing result. Network relocation writes separately allocated derived sections and row
offsets; source values remain archive views. The original archive release callback
runs when the physical result, analysis and loaded relations release their shared
byte owner. Loaded numerical and graph arrays remain archive views; pointer-bearing
headers retain their original descriptor relocation. Network loading binds the
live context and current verified topology explicitly. `Plan.analysis` exposes
the same retained native analysis to the standalone Python caller.
Exact archive format identifiers advance with this layout; earlier archives must
be regenerated through their original producer.

`mesh_planned_free` and `mesh_local_free` release only their own output storage and
retained archive owner. They do not free or modify borrowed source operands. The
standalone [planning entry](../../../native-execution/docs/planning.md) retains
those complete Python owners alongside the canonical native result.
