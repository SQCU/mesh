Mesh’s required architecture represents allocation, lifetimes, placement, dependencies, and schedules as explicit data because these are solved ahead of execution and must remain available to later transformations.
Producers publish complete results in paged storage, and consumers access those same results by view, preserving their identity and page-table references.
Compilation therefore composes functions of explicit operands and represented effects, with original objects or argument bundles passed through existing interfaces using `*args` and `**args` where appropriate.
This is a stronger functional discipline than ordinary Python programming: every agent must recognize copying or reconstructing existing results as a departure from the specification and correct its work.

Every descendant function accesses produced structures through views of the same storage, with their page-table references carried intact.
Passing a view preserves access to the complete structure while each consumer dereferences exactly what its algorithm uses.
Packing, unpacking, or selecting fields into smaller replacement structs along that chain is unnecessary because the complete structure is never passed by copy.
Throughout this integration, agents use access by view to avoid spending memory bandwidth and stack work on duplicated representations, especially across Apple Silicon’s bandwidth-limited interconnect.

# Original planned result and selected emitter

Same66, UUID 476e57ac-cf94-49e8-801a-a3ded988dd77, existing exec25826. This positive assignment supersedes earlier lower_metal/native/runtime scopes in this file.

Read the literal /Users/mdot/schedule-emit/schedule_emit/lower_inductor.py, driver.py and ir.py. The next stage consumes the complete original compiler result. Implement the original PythonWrapperCodegen._generate/run_wrapper_ir_passes and selected wrapper generation boundary, using the original MemoryPlanner/reuse and CppWrapperCpu entry/allocation/view/control/output producers.

The input is the original scheduled wrapper and its complete root/const/control graph and planning operands. Return the complete original planned result before entry and terminal emission, preserving its existing identity and views. The original emitter consumes that result and explicit configuration/supplied bindings, and owns its corresponding output state. Root inputs, constants, allocations, pools/reuse, control occurrences, outputs, seeds, borrowed/consuming ownership and alias/mutation semantics remain the original producers’ responsibilities. Distinct configured emissions share the same original numerical/planning result.

Pass the original object or operand bundle through existing interfaces; do not replace it with selected-field snapshots, restoration, copied tuple/dict schemas or text-buffer deltas. Typed original lines/results remain the semantic source correspondence. Preserve all selected CPU/MPS/root/const/control behavior through the same original emitter.

Own the exclusive file-write turn for torch-inductor-aot-lowered-result.patch and the directly relevant cpp-wrapper-binding producer patch.68 owns lower_inductor.py and its other producer patches; coordinate the actual returned object and original consuming method, and hand off shared-file ownership explicitly. No lower_metal.py, metal.py, native/runtime/archive/placement or method edits are authorized. No imports, dependency installation, workloads, new workers or source backups. Source checks and canonical isolated-index patch preflight are allowed. Codex preserves ordinary named checkpoints and judges the whole producer handoff; no supervisory manifest is assigned to the worker.
