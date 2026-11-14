# Typed storage to planned storage

This is the next correction to the retained-program composition assignment, under
`design/compiler-workgroup-2026-11-13.md` and the operator's quoted instruction to narrow work and stop renovating
inherited bodies. The previous connection is checkpointed in metal-microbench `76d81c90`; its canonical
`make .build/e2b_replay` build passed. Source/build evidence does not establish execution coverage.

## Exact defect and connection

Codex traced the retained alias map past `metal_ir_append`. `export_stage` now places declared source aliases in
one phase storage. But the later E2B consumer allocates a separate `mesh_value` per manifest `Operand`, sets each
host address from that separate value, and binds only the surviving IR storage root. A host write to a non-root
operand can therefore reach different bytes than the compiled operation. Fix this actual consumer handoff.

Producer: the retained model's declared storage relation, exposed by its **unlowered** `metal_ir_value_info.storage`.
Retain that correspondence during export; later allocation reuse is not new alias authority. `metal_ir_storages`
supplies the compiled spans/lifetimes for allocation. Consumer: the existing E2B construction of `mesh_value`,
`mesh_declared.value`, `placed`, host `Operand.address`, and `metal_ir_value_planned` / `metal_ir_value_placed`.
Pass the same storage identity through those consumers before the single `mesh_plan` call; do not reconstruct
aliasing from names, parse emitted text, or repair addresses after allocation. Keep each logical value's layout
and offset and each transfer's event identity distinct from its backing storage.

Inspect actual `mesh_value` / `mesh_plan` semantics in `rdma/mesh-collective.h` and `.c`: `shares` and allocated
sections are planner outputs, not alias inputs. If composition of declared equivalences is required,
`mesh_graph_components` is the existing canonical algorithm (Tarjan, cited in `mesh-graph.h`); do not inline
union-find or invent a second graph library. Reuse the solved source relation; equivalence composition must not
infer aliases that the producer did not declare.

Preserve invocation generations, lifetimes, routed effects and host reads/writes. Do not globally equate two
persistent allocations merely because a later buffer allocator reuses their bytes. Do not take MAX of lag or
pick the first activation policy without establishing its meaning from the actual operands and planner. Do not
add alias-compatibility rejection filters or unconditional numerical copies to keep the old per-Operand allocator.
If the existing planner is missing an operand necessary for this exact connection, report that operand and its
producer/consumer before inventing a new execution regime; Codex will co-design the needed producer change.

## Work and evidence

Resume this same Claude session. Own only `tools/e2b_replay.m` and this worker's status record. No provider numerical
body renovation, new runtime interpreter, new agent tree, or unrelated cleanup. Read the actual calling path and
headers as sources; inherited allocation code is the removal inventory, not the replacement design.

The final source review must trace both repeated source ValueIDs and distinct source views through host writes,
transfer declarations, planner value IDs and final physical binding, including offsets. Use the existing compiler
rule for syntax evidence. Coordinate the final shared rebuild with root. The local `/mesh0` is not currently up;
do not claim an E2B execution or performance result from syntax or private probes. No maintained test suite or
parallel evaluator is assigned. Read `integration-inbox.md` before edits and before declaring source readiness.
