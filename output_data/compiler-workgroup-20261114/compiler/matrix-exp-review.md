# Placed matrix exponential: source and construction checkpoint

The source stage now expands `aten.linalg_matrix_exp` after placement/device assignment and before dependence capture. It uses the one live numerical formulation in `operations.py` through upstream `make_fx`, then the existing `ir.views` owner on changed operands, generated nodes and users. Generated constants carry the selected device. CPU spectral LAPACK nodes remain unchanged; non-CPU spectral formulations are still unfinished.

The existing schedule copies generated-node→original-node correspondence into its assembled module. `ConstructedMachine.decomposed` retains original-node→emitted-operation IDs, while `operations` retains unique task/resource→operation IDs. The measurement consumer of `decomposed` is assigned but not yet complete; this checkpoint does not call that downstream handoff closed.

This joint source checkpoint also includes the reviewed live `lower_metal` imports, single `metal.SourceOperand` ABI use, mutation-return alias correction, and original generator words. Root authorized their inclusion; numerical64 owns decomposition/provenance, root593 owns the numerical primitive import and mutation handoff, executor047 owns initial generator words.

## Observed scope

SAME numerical64 used `/tmp/numerical-decompose-check/stage.py` through real export→place→on_devices→decompose. Two matrix exponentials expanded for CPU/MPS selections. Eager CPU evaluation matched Torch with relative errors 1.3e-15/2.3e-15 (float64) and 1.5e-6/2.2e-6 (float32). This checks formula expansion, not target execution.

`construct.py` then used the actual schedule and typed constructor: MPS-selected float32 retained 312 generated nodes from two originals and emitted 396 operations; CPU float64 retained 430 nodes and emitted 506 operations. Every emitted operation had a unique task association. No missing typed primitive was encountered.

That construction diagnostic set `machines=None` and overrode `metal._repository` in-process to the existing checkout, because the installed recorder did not yet export `MetalProgramRunObserved`. Thus exact driver cost setup, native planning/invocation, throughput, and non-CPU spectral execution are not observations of this checkpoint. No production source override was installed.

The compact observation is recorded in `matrix-exp-observation.txt`; original session traces are `numerical-4.jsonl` and `numerical-5.jsonl`. The paired narrow canonical libmesh build passed (`libmesh-live-owner-build.log`, exact source hashes `libmesh-live-owner-source.sha256`), and a fresh import observed Function168/Planned136/Local176 plus `mesh_arena`. Full native library compatibility remains pending the joint caller/publication source batch.
