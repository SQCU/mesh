Mesh’s required architecture represents allocation, lifetimes, placement, dependencies, and schedules as explicit data because these are solved ahead of execution and must remain available to later transformations.
Producers publish complete results in paged storage, and consumers access those same results by view, preserving their identity and page-table references.
Compilation therefore composes functions of explicit operands and represented effects, with original objects or argument bundles passed through existing interfaces using `*args` and `**args` where appropriate.
This is a stronger functional discipline than ordinary Python programming: every agent must recognize copying or reconstructing existing results as a departure from the specification and correct its work.

Every descendant function accesses produced structures through views of the same storage, with their page-table references carried intact.
Passing a view preserves access to the complete structure while each consumer dereferences exactly what its algorithm uses.
Packing, unpacking, or selecting fields into smaller replacement structs along that chain is unnecessary because the complete structure is never passed by copy.
Throughout this integration, agents use access by view to avoid spending memory bandwidth and stack work on duplicated representations, especially across Apple Silicon’s bandwidth-limited interconnect.

# Original numerical producer and whole emission

Same68, UUID 01ede8e3-b3fd-40cd-8b4c-fd37b3941ffa, existing exec56241. This positive assignment supersedes earlier consumer/runtime scopes in this file.

Read the literal /Users/mdot/schedule-emit/schedule_emit/lower_inductor.py, driver.py and ir.py: scheduled GraphModules and their explicit operands enter the original compiler, whose complete result is retained for the next transformation. Preserve the full assigned numerical domain and actual target precision.

Implement compiler/mesh_compiler/lower_inductor.py and its directly relevant producer patches around the original compile_fx_aot/codegen_and_compile, GraphLowering, Scheduler, selected CPU/MPS backends and AotCodeCompiler. The input is the scheduled body plus its actual source/device/interface operands. The result is the complete original root/const/control compilation owner, selected definitions/calls/externs, original layout/alias/access/effect/workspace/state/source results, and the original planned result supplied by66. Expose these results by view under their originating owners. The same original final emitter consumes this result with explicit configuration/bindings and returns matching whole source, ABI, build actions and dependency ownership.

For CPU templates, retain and consume the original CppTemplateKernel.def_kernel, CppTemplateBuffer.make_kernel_render and original epilogue/group/proxy results. For CoreML, retain the actual original conversion result, full feature/state/output/alias correspondence and target precision. Frontend104 owns export.aot_aliases; its actual original EP tuple remains distinct from JD.

Own lower_inductor.py and the factory, CPU/MPS, upstream-composition, C++ build and directly relevant CoreML producer patches.66 currently has the exclusive entire-file write turn for torch-inductor-aot-lowered-result.patch; read and coordinate the exact return interface, with an explicit writer handoff before editing it. Native131 is exposing the original callable collection/Output/Address return and immediate bridge; consume that complete result when published, without reconstructing its types or bindings.

Complete the module and original whole-emission connection together. Runtime, native consumers, archive, placement and numerical methods remain held. No imports, dependency installation, workload, new workers or source backups. Source checks and canonical isolated-index patch preflight are allowed. Codex supervises source direction and preserves ordinary named checkpoints; this brief does not delegate supervisory manifests.
