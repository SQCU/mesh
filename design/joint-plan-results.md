# Joint plan outputs and ownership

The [planner composition assignment](planner-composition.md) separates analysis,
physical compilation, and live installation. Analysis reads the original
`mesh_function` and `mesh_value` operands. Compilation produces derived storage in
`mesh_planned.result[f]` and `mesh_planned.root[r]`. Each compilation owns distinct
output storage and retains its original analysis.

Each function result owns its derived value sections, sharing flags, completion
times, task order and the host-write lag computed by the original buffer-bound
algorithm. A value result contains the complete `mesh_section` output, including
its byte extent and page count. The original source value supplies its lifetime,
kind, run and state operands through the retained analysis.
Local plans retain the supplied function/root arrays and call relation by view.
Native placement reads source properties from those operands and geometry from
the joint result. `mesh_view_values` takes both complete results explicitly.
The original planning, scheduling and placement algorithms remain their producers.

`mesh_plan_local_analyze` and `mesh_plan_program_analyze` return a complete
`mesh_plan_analysis`, including its original source operands, flow results,
transfer groups, call correspondence and storage-copy calculation.
`mesh_plan_local_complete` produces the local physical result.
`mesh_plan_program_compile` consumes the analysis and an explicit table of
`mesh_plan_schedule` results. It retains those complete results and reads their
completion/order arrays by view. The ordinary entrypoints produce that table
through the same original calculation before invoking compilation.
`mesh_plan_program_compile` produces a `mesh_plan_compilation` with its complete
physical result, placement, schedule, page rows and relocatable runtime records.
Analysis receives topology and storage geometry as explicit operands.
`mesh_plan_program_install` reads the compilation and the supplied live context,
reserves live storage, binds addresses and publishes a generation. Its distinct
`mesh_plan_binding` retains the compilation and owns the reservation and address
tables. Readers combine the original section with the binding's row base and
read the binding's function addresses. Each later compilation or installation
produces another owner.

Numerical graph transformations supply their actual operation correspondence
and complete changed demands at the analysis producer. The broader
[joint IR composition](../../../compiler-emit/docs/joint-ir.md) pairs those
operands with physical planning. The graph header owns grouping, adjacency,
chain enumeration, lag queries, interval sets, and the existing optimization
algorithms. Their producers retain complete solution arrays and correspondence;
their consumers read those same arrays by view.

The analysis owns one `mesh_plan_packing` for the complete call block. Each
demand records its root, destinations and offered byte volume across all calls
of its body. Each solver tree retains its global demand, parent row and
fractional share. With nonzero link startup, `rate` records the tree's
allocated bytes/ns and `startup` its maximum root-to-leaf delay in ns. The
zero-startup fill uses the fractional shares directly. The candidate's
directed capacities use the same arc indices as the program topology.

`first[d]..first[d+1]` selects a demand's trees in `order`. A body locates its
demands with `first_demand`. Its tree adjacency points into the corresponding
global parent row. The archive stores the packing arrays once and restores the
same relationships on load.

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

`mesh_plan_compilation_save` writes the complete compiler result through the
common archive codecs. `mesh_plan_compilation_load` returns that compiler result
with its archive owner retained. The loader produces function descriptors and
points declared-view dimension pairs into the archive. Actor arrays, relations,
solutions and source values remain archive views. The installer consumes the
loaded compilation through the same interface as a fresh compilation. It writes
the runtime records and live bindings into their separate destination storage.
The original archive release function runs when the result, analysis and loaded
relations release their shared byte owner. The standalone Python interfaces
retain each complete native owner while exposing descendant views.
Exact archive format identifiers advance with this layout; earlier archives must
be regenerated through their original producer.

`mesh_planned_free` and `mesh_local_free` release only their own output storage and
retained archive owner. They do not free or modify borrowed source operands. The
standalone [planning entry](../../../native-execution/docs/planning.md) retains
those complete Python owners alongside the canonical native result.
