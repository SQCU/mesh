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

## Refresh, 15 November: export/Exported/runtime/archive family

Status words below are source states, not acceptance. `19bfd157` (the whole-family WIP:
call key identity, an unconditional duplicate-leaf `detach` at capture, a storage-matched
single alias leaf, `Exported.signature`) is rejected: its detach bypassed the AOT
strategy's mutation metadata and its alias choice dropped several real matches. Rows
marked "source change, correction required" are SAME974's later uncommitted source,
unreviewed and unrun, which stays correction-required until the whole semantics are
complete.

Source snapshot: Mesh `43ca2a0d` (AOT stage retention), `19bfd157` (rejected WIP), and
SAME974's uncommitted source after it. Read: the composition change request, AOT
A1/A8/A9/A10, source-lowering 1/2, schedule-emit `export`, `ir`, `runtime` and
`archive`, and Mesh `export`, `ir.Exported`, `runtime`, `archive`,
`schedule.call_operands`, `native.Target.bindings`/`resolved`. No import or run.

| Family member | Original producer and result | Mesh projection or consumer | Finding |
|---|---|---|---|
| Input roles and state | `ExportGraphSignature.input_specs` (kind, arg, target, persistent); `state_dict`, `constants` | `export.read`: `state` (target, tensor), `state_kinds`, `tokens`, `user_inputs` by spec order, as schedule-emit `read`; the signature itself now `Exported.signature`, saved by `archive.sourced` | Projection reviewed in source; the original signature (persistent included) retained as source data. No save, load or pickle of it has run |
| Literal inputs | `ConstantArgument` input specs | `Exported.specialized`; `runtime.literal` (the upstream literal branch); archive saves it | Reviewed in source; unrun |
| Output roles | `output_specs` (kind, arg, target) | `Exported.outputs` (kind, target); a literal output's value stays in the graph's output args (`native.resolved`) | Necessary projection; no field lost to a consumer |
| Mutation destinations | USER_INPUT_MUTATION and BUFFER_MUTATION targets | `read`'s `destination` meta and its `aten.copy` for a placeholder value | Necessary graph projection for native stores |
| Gradient result | `backward_signature`, GRADIENT output specs | `read.returned`: the public `ir.TrainingResult` TreeSpec | Necessary public result projection |
| AOT alias stages | `aot_aliases` (`ViewAndMutationMeta`, roles, prototypes) per AOT stage | `Exported.aliases` (epilogue authority), `Exported.decomposed` (final program's), saved and loaded by `archive.sourced`/`load` through canonical pickle (`43ca2a0d`) | Source path reviewed; no save or load has run and the `ViewMeta` pickle is unobserved; `decomposed` has no consumer yet |
| Call epilogue | `_RuntimeForwardEpilogue` over the first stage | `Captured.captured` | Original authority kept |
| Call export key | AOT dedup by tensor identity (`runtime_wrappers.py` 1504–1525, `AOTDedupeWrapper.pre_compile` 1600ff) | `Captured.captured` key | **Source change (`19bfd157`), correction required:** the key recorded geometry and storage class only, so `f(x, x)` and `f(x, x[:])` shared one export; it now records each tensor leaf's identity class, so a duplicate call gets its own export and torch.export's own `AOTDedupeWrapper` leafifies it, or, mutated, raises (1649) |
| Outer duplicate inputs | `AOTDedupeWrapper.pre_compile` (`runtime_wrappers.py` 1600–1720): strategy 1, a non-mutated duplicate leafified (`a.detach().requires_grad_(…)`: its own tensor, the same storage), no post-compile; strategy 2, `keep_arg_mask`/`add_dupe_map` deduplication, only for mutated duplicates, which the export path bans (1650–1659) | `export.capture` fakes each leaf by `fake.from_tensor` (one fake a real tensor) and holds by `id(fake)`, so the interpreter gave every duplicate position one fake and its last placeholder held it for every call; `call_operands` roots the storage at the first outer placeholder | **Rejected in `19bfd157`:** an unconditional detach of each later duplicate leaf. **Source change, correction required:** each call's user input argument is the outer placeholder its own FX argument names (`Captured.captured` records each placeholder's origin; `capture.provenance`), while that placeholder still holds the value; a user input mutation's written base is that argument. The value keeps its identity, so a call passing one tensor twice gives torch.export's own `AOTDedupeWrapper` the duplicate (strategy 1 or its export ban). **Open:** two outer positions holding one tensor share one storage root, an input alias contract no saved record carries to the public call |
| Outlined parts' outputs | `decompose._Part.crossing`/`_fresh`: each part's changed input prototypes and their origins | `export.substitute.realized`: each part output's prototype placed on the actual input storage at its exact byte origin through one `FakeTensorConverter` memo | **Necessary relocation:** no part has an AOT product (`ViewMetaSequence`) the epilogue could take, and no canonical API takes these operands; making one would be new compiler work |
| Outlined original function | The replaced function's `Exported` (signature, `aliases`, `decomposed`, state) | `decompose._exported` (`decompose.py:425`) builds each part `Exported` without them; `export.substitute` drops the replaced function from `functions`; `graph.meta["decomposed"]` keeps Nodes only | **Source change, correction required:** `CapturedProgram.outlined` (granted) records each replaced function's original `Exported` and its parts' keys, set by `substitute`; `archive.sourced` saves each as the same Exported record with its parts' ordinals and `load` restores it |
| Result and alias relation | `torch/utils/_python_dispatch.py get_alias_info` (torchgen's schema: each argument's and result's alias set, `read_only_alias_match_indexes`, `outs_write_aliases`); each result's producer FakeTensor storage | `ir.aliasing`/`aliased`/`written`/`source`/`view`, `placement.parent`/`realize` | **Rejected in `19bfd157`:** a storage-matched single alias leaf (`leaf_of`), none for several matches. **Source change, correction required:** `aliasing` and `written` read only the first return and `source` took `args[0]`; each result now aliases every argument of its `get_alias_info` read-only matches (torch's `_correct_storage_aliasing` loops every match: each element of a list result each such Tensor argument) and writes every argument of its write alias (a list result's element k that argument's element k, `return_and_correct_aliasing`); `source` and `view` return the whole relation; `ir.based` requires a value's bases to agree on one root and offset and raises otherwise; `ir.views` walks every source edge; `placement.parent` lists every result's aliased arguments and `realize` replaces each by its realization |
| Per-result placement geometry | Each result's own `val` | `placement.transfer`/`partitioned`/`meshed` took `ir.shape(node)`, a tuple value's first result's shape; a getitem's `aligned` rule read its producer's held dimensions through that shape | **Source change, correction required:** `ir.shape(node, k)` gives result k's shape; `ir.produced` a node's one iteration space (a tuple value's where its results share one shape, as `topk`'s; none otherwise); `partitioned` and `meshed` take that as `out` and read a getitem's operand at its own result (`read_shapes`); `transfer` places a tuple producer whose results differ in shape whole on its stated placement's first processor, and each of its getitems selects its result there and reshards that result to the stated placement at its own shape. Unrun; whether every such producer's getitems reach `transfer` with that placement is unobserved |
| Public interface | `inspect.Signature`, `TreeSpec` | `Exported.interface`; `runtime.Arguments.prepare` binds, applies defaults and flattens by the retained spec | Reviewed in source; unrun |
| Static tensor metadata between calls | FakeTensor's static metadata: an attribute (`shape`, `dtype`, `device`, `ndim`) the outer `torch.fx.Tracer` records as `call_function getattr`; a method (`size()`, `dim()`, `stride()`) it records as `call_method` | `export.calls` raises on every `call_method`; `Captured.call_function` evaluates `getitem`/`getattr` only on a tuple, list or dict and raises otherwise. A non-tensor value reaching a call is already a literal downstream (the call key's `repr`, the function export's `ConstantArgument`, the metadata node erased by `capture`'s cleanup) | **Open classification:** reading a fake's static descriptor runs no numerical work and is FakeTensor's own result; a tensor-valued attribute or method (`x.T`, `x.real`, `x.sum()`) is a tensor operation and still needs a captured function. Both exclusions are this owner's (`export.py`); no current program needing a descriptor read is established, and none is changed |
| Public input constraints | `torch/_export/utils.py _check_input_constraints_for_graph`: each tensor input's rank and static dimensions against its placeholder `val`, and the literal branch | `runtime.literal` transcribed the literal branch only; a dirty `runtime.tensor` copy of the tensor branch was rejected | **Source change, correction required:** `Arguments.prepare` calls `_check_input_constraints_for_graph` itself once, over the outer program's retained user input placeholders in user input order (each `val` its compiled prototype or literal), each leaf with its key path in the retained interface TreeSpec, and the outer program's range constraints (static shapes: none); `literal` and the copy removed. `Exported.specialized` stays a retained source record no runtime path reads |
| Tokens | TOKEN input and output specs | `Exported.tokens`; `runtime` binds `torch.empty(0)`, as schedule-emit `runtime` does | Necessary projection |
| Saved source | Exported records, CallOperands, query, compiled records | `archive` one GraphPickler payload and `state.pt` | Each function's Exported fields are in the saved source record by source review (`aliases`, `decomposed` since `43ca2a0d`, `signature` now); none observed through a save and load |

### SAME974 turn-104 checkpoints

Each row: the functions the checkpoint changed, the original owner whose result it
carries, its consumers, and what in it is still authored projection, enumeration or
inference rather than the original owner's own result. `4e1a2e85`'s single GraphPickler
source owner is the reference codec for all of them. None has run.

| Checkpoint | Changed functions | Original owner and result | Consumers | Remaining authored machinery |
|---|---|---|---|---|
| `0e54e31b` composite copy, source query, closure | `native.Target.composite`/`composed`/`ported`, `Row.copies`, `Target.copying`; `archive.queried`/`declared`/`copying`; `mesh-graph.h`/`.c` `mesh_graph_closure_save`/`load`, `mesh_graph.Closures.save`/`load`; `lower_metal.InputPart.geometry` | `CopyRecord` and `InputPart` producer geometry; native typed copy (`metal_ir_copy`, then `_Builder.copy`); `Declarations.dependences.owner`; the native closure arrays | `Prepared.remap`/`bind`, native recorder and construction, archive load | Wire and result roots sized from CopyRecord extents; the transfer domain by `mesh_graph.strided`; a complex leaf's real pairs by `lower_metal._typed` |
| `92aa0f23` shared Context | `mesh.py` load rows; `schedule.SavedNetwork`/`Solved.analysis_context`, `loaded_program`, `published_program`, `solve_program`; `native.restore`/`compile_program`/`prepare`; `bridge.Published`; `metal._archive_loaded`; `mesh_graph.Context`, `Relation.load`; `archive.load` | Native `mesh_graph_context`, created once by the driver (fresh) or `native.restore` (loaded) | Every relation, local, program and archive load | None in Python: one context object passed through |
| `b4f980a2` Produced source | `archive.constructed`/`producing`; `Pickler` raw graph reducer; `Rawed`/`RawRef`/`GraphOf`; LoopBody `body_of` | `lower_inductor.Produced`, `Kernel`, `Ranges`; `GraphPickler._GraphPickleData`; `LoopBody` | Saved query, its reload and re-save | Per-call record tuples are archive-authored shapes; LoopBody field list read by name |
| `0a79ec80` CPU loop result | `archive` `proxies`, `subkernels`, `nested` LoopNest records | `codegen/cpp.py` `CppKernelProxy`, `LoopNest`, `LoopLevel`, `OuterLoopFusedKernel.inner` | Saved query | `OuterLoopFusedKernel` dispatch by class |
| `4400ab5a` nested control and model | `archive` Lifted and qualified Seeds records, Model ExportedProgram fields; `Tensored` real tensor registry in `encoded`/`decoded` | `lower_inductor.ExternCall.lifted`, `Lifted`, `Seeds`, `Model`; `ExportedProgram` operands; `state.pt` registry | Saved query | Every real tensor in a saved value goes to `state.pt` by one generic rule |
| `6cc8465c` native associations | `archive.constructed` `interfaces`, `regions`, `inlinings`, `emitted`, `upstream`, `neural`; `Record` identity index | `lower_metal.ConstructedMachine` fields, `metal.Region`, `metal.Emitted` | Saved query | An unregistered `Emitted` record passes as it is |
| `9accce16` wrapper lines and CPU accesses | `archive` line table, `wrappers`, IR node table (`Lowered`), `Stated` layouts; subkernel domain fields; `contexts` | `wrapper.py` WrapperLine dataclasses; `ir.py` `OutputSpec` state; `cpp.py` subkernel attributes | Saved query | Subkernel domain attributes listed by name per class (`CppKernel`, `CppVecKernel`, `CppTile2DKernel`); a layout's `vars()` state |
| `29d71db9` local buffers before clear | `archive` cpu range fourth operand; `context_of` without the cleared `local_buffers` | `lower_inductor.MeshKernelGroup.finalize_kernel`'s snapshot | Saved query | None |
| `73296c4d` terminal emission call | `native.Target.instanced` calling `lower_metal.emitted` | `lower_metal.emitted` (compiler66) | Native lowering and finalize | None in `native.py` |
| `eb56bbcb` Binding source | `archive.constructed` `bindings` (occurrence pairs, record and source operands) | `lower_metal.Binding` | Saved query | None |
| `43ca2a0d` AOT stages | `ir.Exported.decomposed`; `export.export`/`read`; `archive.sourced`/`load` | `aot_aliases` of each AOT stage (the alias-metadata patch) | `Captured.captured` (stage 1); `decomposed` none | `decomposed` has no consumer |
| `19bfd157` whole-family WIP, rejected | `Captured.captured` key identity; `capture` duplicate detach; `ir` alias family with `leaf_of`; `Exported.signature`; `archive` signature record | `AOTDedupeWrapper`; `get_alias_info`; `ExportGraphSignature` | Capture, placement, archive | The detach bypassed mutation metadata; `leaf_of` dropped several real matches |
| After `19bfd157`, uncommitted | `capture.provenance` and the call's recorded origins; `ir.aliased`/`written` (every match), `source`/`view`/`based`/`views` over the whole relation, `ir.shape(node, k)`/`produced`, `placement.read_shapes`/`parent`/`realize`/`transfer` (`apart`); `CapturedProgram.outlined` and its archive record; `runtime` calling `_check_input_constraints_for_graph` | Source FX provenance; torch alias pairing; each result's `val`; the replaced function's `Exported`; torch's input constraint owner | Capture, placement, archive, public call | Every path unrun; the outer input alias contract (two outer positions of one tensor, one storage root) has no saved record |

### Introduction history

- `297c3cda` introduced `shape`'s first-result choice in placement.
- `6363f45d` introduced first-return `aliasing` and the first-argument view source.
- `0d693b4f` extracted `source()`.
- `0ad0f72f` moved them unchanged into `ir.py`.

At each of these, every node's `meta["val"]` already held all of its results and every
schema held all of its returns and arguments.

The call export key was introduced in an interrupted turn (`3a2aa8c1`), withdrawn
(`817ff6a3`) and reintroduced (`941db09b`). The briefs were
`frontends/torch/call-inputs-1.md` and `call-inputs-3.md`. Item 3 of the latter
enumerated the specialization metadata as "dtype/shape/stride/offset/alias" plus
"device/requires_grad", and did not include tensor identity.
