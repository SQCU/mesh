# Frontend composition ownership, 14 November 2026

This dated source review uses the complete composition change request, AOT A1–A10,
source-lowering change request, and `/Users/mdot/schedule-emit` at `f41c64a`. It is
evidence about current implementations, not a new specification. The inspected
Mesh source was `8cbc2ab99c2c1305bda0869ba3bfd5d8158f9b50`; frontend archive checkpoint
`4e1a2e85` is preserved. Tracked source was clean. No import, execution or saved
compilation roundtrip was performed. SAME974 turn103 returned success and HOLD.

Compiler's companion table is `compiler/whole-lowering-ownership-20261114.md` in
this workgroup record. Executor owns the native C/provider and RDMA C algorithms;
this table owns their Python consumers. Necessary changed-input projections and
serialization are distinguished from duplicate implementations or repeated solves.

| Responsibility | Original implementation/result and consumers | Alternate/discarded/rebuilt responsibility | Whole replacement/removal unit |
|---|---|---|---|
| Numerical bodies and transfer copies | Scheduled processor bodies, original upstream result, native `compile_program`, composed/constructed/instanced/placed stages | Live `native.Target.composite:632–733` reconstructs meta tensors and invokes `_copy/_placed` plus `_Builder.generic` for stage, input landing and output landing. Compiler also identifies premature compilation followed by terminal emission. | Joint compiler/native original body, copy, view and interface result through one terminal emitter. Delete all old copy-record compiler callers with operations/Builder machinery, not just its definitions. |
| Source effects and scheduling | `schedule.dependence:92`, `dependences:128`, `schedule:289`, `analyse:643`: source relation, task order, endpoint declarations, event graph/boundaries, call relation/search/closure/chains. Consumers: assemble, processor modules, declarations, native ports/functions, sole solve_program, shares stages. | No repeated identical solve established. Original `Declarations.dependences.owner` holds events, closure, graph, redundancy/boundary results and row provenance; `CallAnalysis.declarations` retains it live, but archive queried/sourced omit it. Native saves final dependence rows, not those original derivations. | Same saved analysis/source owner as produced lowered/control/provenance results. Retain original native graph results; do not rebuild from final rows. This is a retention omission, not evidence runtime re-solves. |
| Saved compilation and preparation | archive one GraphPickler/shared tensor/qualified reference owner; native saved artifacts and LocalOwner/network Published; Target.prepare and Bound | Old wrap/export/unlift/getitem/name-repair graph route deleted by4e1. Explicit tuples in archive sourced/load, Target recipe/restore and ABI wrappers are manually maintained schemas, not graph algorithms. New upstream lowered/control results still require their actual owner. | Complete one saved-compilation owner, no opaque GraphLowering reconstruction, no compilation on load. Current codec has source checks only. |
| Placement and aliases | placement forward result, original source alias base/view replay, devices changed-graph projection, ir.views; consumers assembly, typed interfaces, bindings and archive | required:605 is the accepted upstream require_strides geometry transcription with full backing/offset preservation, not a second solve. Compiler's single source tag loses many-to-many rewrite provenance. | Pair original interface/provenance result with compiler replacement. No root-plus-delta alias guessing and no removal of valid geometry without its original-result replacement. |
| Costs and sharding | shares supplied costs or Work/sample/count, stage graph and canonical LP; allocate.separators; consumers parallel/tp and placement counts | shares.speed/allocate.speed repeat the FPM expression; shares.rows/obsolete mesh_compile.rows repeat ragged projection. shares.shifted and allocate.shift have different integer/continuous inputs. Old mesh_compile stage_times/shard is another whole metric route, in a malformed unimported module. | Legacy cost/sharding removal/consolidation as a whole, preserving the active objective. No new primitive tracer or exact dynamic work requirement. Existing sample/count runtime observation remains pending. |
| Capture and public invocation | inspect Signature/BoundArguments, Torch export, original AOT alias metadata/runtime epilogue, FakeTensor converter at changed part interface; Arguments/Program prepared slot/output recipes and saved public interface | No source fake reexecution remains. runtime.literal is the specifically approved exact upstream literal-branch transcription. Rebinding new invocation arguments is not repeated compiler analysis. | No separate replacement justified. Preserve these actual consumers while the upstream body/interface producer is replaced. |
| Displaced plan/runtime | Current solve_program and Target/Bound replace former single-function Plan path | schedule.solved:156 and argument_buffers:393 have no current compiler/example/RDMA consumer; ir.Plan is the displaced record | Delete the complete unused Plan/argument-buffer responsibility with its unused imports/ABI consumers. It is dead code, not live repeated solving. |
| Legacy backend dependencies | operations imports mesh_code:23 and uses fragment:4666; other backend implementation belongs to compiler/executor | lower_coreml imports mesh_coreml/mesh_metal4 and describes the old runtime but fails AST at321. rdma/mesh_compile fails AST at7628 and has no current compiler/example importer. The three wrappers themselves parse. | Whole legacy provider/backend replacement must include the live fragment dependency; do not silently delete supported fragment behavior. |

Reviewed file scope: export, devices, placement, schedule, shares, archive,
runtime, native, bridge, rdma/mesh_graph, mesh and allocate. Driver/IR/decompose
internals are compiler-owned cross-boundaries. Legacy mesh_compile, mesh_code,
mesh_coreml and mesh_metal4 dependencies were traced, not omitted. Serving/UI is
outside this compiler flow.

`placement.gathered:366` is captured by the shape-rule registry at515 before the
distinct whole-value gather/cache function at595. It is name shadowing, not
duplicate computation. Current saved-plan relocation, physical-copy projection,
graph contraction and changed-device/view analysis are not repeated solves. No
additional live same-input planner/relation solve was established in these Python
paths. Native saved ReplayPlan/RegionForest/search fixes remain credited.

The only Python mesh_view_values callers are dead schedule.solved:170 and obsolete
mesh_compile:6582. The analogous mesh_plan callers are schedule:160 and
mesh_compile:8225; mesh_plan_save callers schedule:162 and mesh_compile:8599.
No new source work was assigned until the three supervisor tables were reconciled.
