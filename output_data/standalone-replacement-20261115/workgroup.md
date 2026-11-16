# Standalone compilation and execution workgroup — 15 November 2026

## Operator directions

> if target runtimes are meant to be separated from the compiler activity we are working on, we must force oureslves to make our interfaces match the data flow specification by *actually lowering* to the schedule emit target runtime, whether or not it is compatible with mesh's goals, so long as it has the same domain as the domain of mesh's lowerer.

The preceding assignment supplies the replacement mechanism:

> we can write 2 clean modules standalone, demonstrate they have the same properties as the specified interfaces we need, then literally categorically replace (delete. copy.) the clean modules to get rid of arbitrarily large and complex technical debt issues in a simple and wallclock-fast further approach.

Its priority for the minor callback findings is:

> extremely low priority, lower priority than e2e testing

## Assignment and composition

Every assignment begins with the complete [functional-compilation](../../design/functional-compilation-memorandum.md)
and [access-by-view](../../design/access-by-view-memorandum.md) memoranda. The
[composition request](../../design/compilation-composition-change-request.md#numerical-lowerer-boundary)
supplies the producer, operands, lowering and consumers. AOT A1 supplies the assigned known-shape program domain;
the numerical algorithms retain their specified internal conditions. Source-lowering items 5/6 supply their
numerical methods. The two targets consume the same complete source and analysis results by reference.
The [data-flow and review memorandum](#data-flow-and-review-memorandum) records the root's assignment corrections
and the evidence to carry into subsequent work on these interfaces.

| Scope | Producer, mechanism and consumer | Assigned repository |
|---|---|---|
| Compilation and emission | The complete scheduled operand enters original upstream numerical lowering. The original compilation result and explicit target/composition operands supply final emission. Original compiler objects and solved results remain available to consumers. | `/Users/mdot/compiler-emit` |
| Executable reference target | The compiler produces per-processor functions for the existing `schedule_emit.runtime.Program(scheduled, functions)` interface. Program runs them using its original argument bindings, threads and queues. | `compiler-emit`, consuming `/Users/mdot/schedule-emit` |
| Native target | The complete compilation result and prepared bindings supply the native entry. Original numerical/device/transport APIs execute the compiled order with invocation operands and storage owners retained through completion. | `/Users/mdot/native-execution` |
| Ownership review | Source and actual Metal API obligations establish the storage-owner and completion handoff. | [Independent review](metal-external-release-review.md), committed as `a280ef88` |
| Integration | Demonstrated standalone modules replace the corresponding Mesh implementations and their direct interfaces through the operator's delete/copy mechanism. | Mesh and metal-microbench |

The reference target's execution policy is expressly accepted for this demonstration. Its threads and queues
remain its own mechanism. Native page residency, native transport and performance obligations belong to the native
target. Numerical results, state, mutations, aliases, effects and dependency ordering define the shared semantic
obligation. The complete retained producer remains accessible through both target results.

An execution observation records the actual producer, lowerer, target, source revisions, program instance and
properties exercised. The domain-wide argument comes from each lowering's rules and their composition. Concrete
reference-interface gaps become producer/lowering work under that same assignment. Existing reference samples
provide instances; the assigned program domain remains AOT A1.

## Source-review observations at assignment time

These are dated source observations, rather than standing descriptions of implementation state.

At Mesh `2f78b2ef`, `compiler/mesh_compiler/driver.py` imports `native.py`, `lower_metal.py` and the upstream
lowerer. Its ordinary execution consumer is `native.Target`/`Bound`. The visible compiler caller of the retained
`rdma/mesh_metal4.py` HOSTED callback interface is `lower_coreml.py`, whose header was already deliberately disabled.
The HOSTED implementation proves that the retained route re-enters Python; this trace establishes neither current
public-path reachability nor a measured active per-operation overhead. Replacement accounts must include the
current producer/emitter/entry boundary as well as deletion of obsolete implementations.

The same snapshot also contains an unfinished producer migration: `export.called` takes `(stack, module, args, ...)`,
while `driver.compile_module` supplies `(module, args, ...)`; `CapturedProgram` carries `exported` and `outlined`,
while `driver.compile_captured` reads `captured.functions`. These are concrete source-interface mismatches.
The current broken calling path therefore supplies neither a working baseline nor the definition of the assigned
domain. The standalone work follows the complete original upstream result and the positive AOT/composition
requirements; the replacement includes their eventual direct producer/consumer connections.

The original source of the maintained Inductor patches contains a deferred `_emission_context`/`entered` state
restoration and Mesh's `CppTemplateKernel.printed` partial-render manipulation. The compilation assignment returns
complete original upstream products at their responsible interfaces and uses original finalization in its original
scope. The independent review records legitimate Metal notification lifetime duties and the separate caller's loss
of a not-drained completion result.

Low-priority sweep-marker restoration is a separate source-history change. It preserves original bodies and the
older deliberately disabled obsolete implementations. Central deferred-emission and HOSTED markers remain pending
replacement. The native/compiler agents work in their separate repositories while that mechanical restoration runs.

## Observations from actual reference execution

The isolated `compiler-emit` environment ran the existing `schedule_emit.runtime.Program` with compiled functions
from the reference lowerer. A two-CPU-processor MLP matched outputs exactly. The MPS MLP reported maximum absolute
error `5.960464477539063e-08` and relative error `1.8593851162171478e-07` through the reference's existing comparison.

The two-processor mutable-state case initially deadlocked. Its supplied p0 graph orders `send(2)` before `recv(3)`;
the generated wrapper instead performs `recv(3)` before `send(2)`, while p1 needs channel 2 to produce channel 3.
The original AOT graph retained the ordered effect-token chain. Original Torch `lowering.with_effects` then keyed
`GraphLowering.additional_star_deps` with `new_op.get_name()` (a buffer name), while `Scheduler.compute_dependencies`
looks up that table with `BaseSchedulerNode.get_name()` (the operation name). Thus the consumer missed the producer's
already-computed effect dependencies. This is a source-name handoff defect in the upstream compiler used by the run.

Changing that original producer to `new_op.get_operation_name()` restored the handoff. With caches disabled, the
same two-processor mutable-state case completed three calls with zero observed output/state error. The runtime and
scheduled source graph remained unchanged. The local patch and observations belong to `/Users/mdot/compiler-emit`;
logs are `artifacts/reference-first.log`, `reference-mutation.log`, `reference-effects.log`, `reference-mutation-fixed.log`
and `reference-mps.log`. These observations establish the exercised numerical, effect-order and state properties;
full assigned-domain coverage remains implementation work.

Standalone compiler commit `e05aa93` preserves that effect correction. Commit `fee1381` retains the complete
original `CompiledAOTI`, `GraphLowering`, wrapper-planner and source `ExportedProgram` results. Its shared-source
CPU and MPS examples also exposed a mutable scalar being folded into a literal: `_unlift_graph` already produced
`mutated_named_buffers`, but `ConstantFolder.run_node` ignored that operand. The original constant folder now routes
those mutable attributes through its existing `unknown_value` result. The same example then preserved scalar state
`1 → 2 → 3`, returned-input identity, and view storage, strides and offset on both targets.

The independent interface review distinguishes shared **source** from shared **scheduled result**. At `fee1381`,
reference execution consumes the placed/scheduled program, while AOT packaging separately consumes its original
source EP. That establishes source identity and the observed semantics, but leaves the scheduled-native
action/page/transfer/lifetime handoff unfinished. Native-execution commit `54a8342` retains the actual complete AOT
result and records CPU/MPS caller-mutation observations; its native program bridge has compilation/linkage evidence
only. These are checkpoints, not a claim that the categorical Mesh replacement has occurred.

The separate [callback-restoration receipt](callback-restoration-20261115.md) records Mesh `ef651076`, Metal
`ac2bd2ca`, restored dependency trees, exact source comparisons and the remaining central markers. The standalone
native module separately executed repeated CPU and MPS alias/state examples at `native-execution` commit `cd09ad6`.
Those native artifact-boundary examples do not yet establish scheduled transport or external page-plan consumption.

## Placement clarification

Operator, 15 November 2026:

> 'mesh' appears to have at the level of axiomatic construction no distinction which allows for 'devices' to be programmed imperatively. (in fact, it seems to exist to turn pure linear algebra and linear algebraic functions into callgraphs which are *placed by the compiler* upon devices through a separate step from lowering or parsing code).

The questioned observation concerned the reference's `torch.device` field and its former `Program.bind` conversion.
The exact demonstration used the existing `samples.MutatesInput.forward`, whose body is `x.add_(1); return x * 2`.
The input was a CPU tensor. The separate placement operand assigned numerical operations to `work: mps` and the
output to `result: cpu`. The numerical source contains no device-management operation. New transfer nodes in this
reference case derive from that compiler-supplied placement; invocation retains the original caller binding.

This case exercises compiler placement and the source mutation/alias contract. It establishes no additional
requirement for imperative device control. The reference target's tensor-copy mechanism is target-specific evidence,
not an allocation rule for Mesh. Mesh consumes the retained placement/page results: shared storage is accessed by
view, and the compiled plan supplies any required copy or transfer. A CPU/MPS/ANE target label alone does not
establish a need to copy data. Source numerical capture, placement and target lowering retain their separate roles.

## Simplification and correction of the demonstration

Operator:

> if schedule emit is stinky in this specific respect (reimplementing something that it didn't need to) fix schedule emit too as part of implementing the right transformations in the right order

Here `schedule-emit` is the executable data-flow demonstration and runtime target; the change requests supply
the assignment, and original Torch tools supply implementations to reuse. At the `018120a`/`6b92ff0` checkpoints the reference adapter contained
its own device-keyword and `aten.to.device` rewrites, while the original `move_to_device_pass` owns that metadata
interpretation. The experimental output-region assembly also repeats kind interpretation from the original unlift
wrapper. These are the duplicated responsibilities to remove together with their dependent interface mistakes.

The refactor supplies complete producer operands to the responsible original transformations in their data-flow
order. A factored shared operation has one implementation used by its original caller and the reference adapter.
Its input/result boundary carries the needed original metadata and owners. The standalone compiler has one ordinary
scheduled-AOT entry, retaining the complete scheduled result and original compilation products for actual Program
execution. Fixing the demonstration itself is part of this assignment; source history preserves the superseded
experiments. Target-specific metadata handling supplies no additional imperative device-control requirement.

### Committed correction and executed direct consumers

The resulting source is schedule-emit `2e17bbb`, compiler-emit `1476caf`, and native-execution `824e674`;
native-execution `64db544` records the final paired runs against those producer commits. The source changes
and the following observations were reviewed together on 15 November 2026.

The original Torch device operation now has one implementation shared by its original pass and the reference.
The reference supplies compiler placement to that operation before inserting input/channel transfers. Original
`Tensor.to` supplies the transfer identity decision. Original `deepcopy` retains source state and constant tensors
through its memo; the locally recursive graph copier is removed. Every processor receives the complete original
program operand, and original signature projection and unlift produce its scheduled flat interface. Archive save
writes those actual processor products; archive load binds them to the loaded source owner. The synthetic archive
signature constructor and unused duplicate signature headers are removed.

The standalone compiler now has one `compile(scheduled, package_path)` entry. It retains that same Scheduled and
the complete original per-processor AOT products, and supplies their unchanged constant maps to the original
loader's user-managed binding operation. Both consumers execute the actual reference `Program`. These executions
use compiler-emit's isolated Torch environment and its recorded patches exposing the original shared operations;
the shared installed Mesh dependency remains unchanged.

Observed direct-consumer results:

- RunningMean: two CPU processors and three calls, output/state error `(0.0, 0.0)`; the saved/loaded processor
  products also produce `(0.0, 0.0)` after AOT compilation.
- Scalar state and returned-input/view aliases: three calls on CPU and MPS, state `1 → 2 → 3`, preserved input
  identity and view storage/stride/offset. Nested public inputs preserve caller mutation and returned identity.
- Separate placement: CPU caller input, numerical work on MPS, CPU result; caller mutation and numerical results
  pass across three calls.
- Native numerical entries supplied to the same reference Program: two processors, two channels, three calls
  on CPU and MPS; original input identity, caller mutation and source-state sums `6/16/30` pass. The MPS entry uses
  original `synchronizeAllMPSStreams`, correcting its earlier choice of an unsynchronized stream operation.

The compiler's `docs/narrow-boundary.md` and native consumer's `docs/observations.md` identify the executable
examples and logs. Independent source review found no additional confirmed defect in these changed direct
interfaces. These observations cover the stated cases; they do not establish arbitrary nested-attribute archive
ownership, the full numerical domain, native page/transport composition, or the categorical Mesh replacement.
The seven pre-existing modified reference archives remain untouched.

### Producer boundary review after the reference correction

Source review of compiler-emit `1476caf` and Mesh `b4ac32ce` locates the remaining compilation handoff earlier
than the retained final AOT product. Original `GraphLowering.codegen` constructs the Scheduler, then calls
`scheduler.codegen`, then generates the wrapper. CPU and MPS backend code renders numerical accesses during
that scheduler call. Mesh's `LoweredAOTI` return at planned wrapper lines therefore follows numerical rendering;
its later `_emission_context` and `emission` closures reconstruct ambient compiler state to print again. Its
`CppTemplateKernel.printed` also repeats the original `PartialRender.finalize_all` traversal.

The next implementation unit factors preparation and emission at their original owners. Both original callers
use those same functions; the complete GraphLowering and Scheduler remain the operands. Original backend
indexing, scalar/vector preparation and memory planning supply the subsequent numerical-access and allocation
results. Composition consumes those results before their first numerical-address rendering. The template owner
supplies its actual PartialRender to its original finalizer. A first split before scheduler emission alone does
not establish complete native bindings or an emission API usable after the original compiler context exits.

Compiler-emit `d5847df` factors original preparation/emission; `3dc106d` establishes the preceding explicit
lower-result handoff. `GraphLowering.lower(*args)` calls the unchanged original `run` and returns its complete
owner. Original constant/main callers consume that result immediately. Constructor operands retain the actual
`fake_tensor_prop` result, original extern list and complete invocation-argument bundle; child graphs receive
their parent's references. `_unlift_graph` retains the actual pre-unlift AOT signature on its returned module.
The original extern producer, index consumer and serializer share the retained live list.

The operator questioned whether CPU/MPS execution and scheduler identity mattered for this first interface.
Review consequently followed the producer's actual operands, returned objects and direct consumers in source.
Independent review of `3dc106d` confirmed all five constructor routes and the unchanged lowering body and caller
scopes. Eight patches produce fifteen parseable source files matching the isolated Torch installation. This
change had no workload runs or identity profiling. It establishes the local typed handoff; exposing ordinary
standalone preparation/composition/emission and connecting native planning remain the next work.

The native source trace identifies existing algorithm owners, rather than an additional runtime to implement:

| Result | Existing producing/consuming boundary |
|---|---|
| Native numerical values and source relation | `lower_metal.construct` → `ConstructedMachine` → `native.Target` |
| Effects, actor relation and storage lifetimes | `Target.instanced`, `metal_ir_lower`, `native.storages`, `MetalRecorderActors` |
| Joint page/transport allocation | `schedule.solve_program` → original `mesh_plan_local` / `mesh_plan_program` results |
| Prepared bindings and native action program | `Target.placed` → original recorder prepare/plan/replay → `MetalRecorderProgram` |
| Compiled execution and drain | `metal_ir_native_compile` → `MetalProgramRunArguments` → `MetalProgramEnd` |

The standalone numerical products still use reference channel effects. Their native target requires the same
Scheduled/compiler owners plus explicit native topology/planning operands, connected at the channel-effect and
allocation producers before final emission. Original native APIs then consume those values and complete plans.

The active recorder translation units are `metal_capture.m` and `metal_program.m`, as the metal-microbench Makefile
declares. The deliberately disabled `metal_recording.m` is historical source. The active producer supplies
`NativeProgram.records`, `NativeObservation.counts` and the compiled image views through save/load; independent
review confirmed these handoffs. The active translation units pass `clang -std=c11 -fblocks -O2 -Wall -Wextra
-Werror -fsyntax-only -I . -I ../dox/mesh/rdma metal_capture.m metal_program.m`. This is source-compilation evidence,
distinct from executing a fresh native library. No change to the obsolete recorder remains.

### Standalone preparation checkpoint, 15 November 2026

Compiler-emit `5adafc8` exposes original Torch `aot_prepare` and `aot_emit`. Preparation returns the original
GraphLowering after the existing export, unlift, post-grad and lowering operations. Emission consumes that result
through the original moved code-generation body. The return carries the original invocation operands, compiler
owner, configuration, constant graph, provenance and autotuning state. Original provenance consumers now share
the producer's whole owner; emission borrows the original tracing context while its caller retains cleanup.

Independent source review followed original direct callers and the constant/main tables. Main additions preserve
the constant indices used by both generated bodies. The original wrappers handle their numerical device operands;
the broader shared device set alone did not establish a need for a copied earlier table. Eight recorded patches
produce nineteen parseable Torch files matching the isolated installation. These are source observations for
this checkpoint, with no numerical runs or scheduler-identity profiles.

Two direct interface edits remain at this checkpoint: preparation must return all processors' original graphs
together with the same Scheduled before emission, and the original AOTAutograd wrapper callers must retain their
completed-callable return convention. The direct AOT preparation caller consumes lowered IR. Their numerical
algorithms and configuration normalization remain the original shared implementations. Subsequent native
page/effect composition and categorical replacement remain outstanding as described above.

Compiler-emit `8c8da0f` completes those two direct edits. `prepare(scheduled)` returns `Prepared` containing the
same Scheduled and every original processor GraphLowering before terminal emission. `emit(prepared, path)`
consumes that owner, and `Compiled.prepared` retains it. Ordinary completion callers use the same original
input normalization, configuration normalization, lowering and code-generation implementations with their
original completed-result convention. Independent source review and the nineteen-file patch projection passed.

After that source review, the existing reference examples executed at `8c8da0f`. CPU and MPS each preserved the
original input/view alias assertions and scalar state `1 → 2 → 3`. The existing two-processor RunningMean case
completed three calls through six channels; `samples.measure` reported output/state error `(0.0, 0.0)`. All three
processes exited zero, and the tracked source and projected dependency files remained unchanged. Logs are
`compiler-emit/artifacts/8c8da0f-shared-cpu.log`, `8c8da0f-shared-mps.log`, and `8c8da0f-running-mean.log`.
These observations concern the implemented reference composition, following the first-interface source review;
native page/effect composition, backend-preparation handoffs and Mesh replacement remain implementation work.

### Retained backend preparation and direct FX entry

Compiler-emit `9d53f8c` (review receipt `fe3175b`) factors original `GraphLowering.prepare_cpp_wrapper` and
`emit_cpp_wrapper`. The returned original graphs now retain their wrappers and Schedulers; terminal AOT callers
consume them directly. Original complete compilation calls the same preparation and emission operations.
Sample-input extraction is an original graph method, and its original lazy-autotuning consumer calls it directly.
Wrapper construction retains the original dual-wrapper-mode decision so later shared-device additions leave
buffer creation and emission using the same selected mode. The original predicates and extraction timing were
reviewed, including constant/main graphs, shared name allocation, subgraphs and ordinary completion callers.

Schedule-emit `4283357` removes its manual `with_effects`/`getitem` parser and multiple-result restriction.
Original functionalization and numerical lowering consume the intact token/schema/result structure. The reference
lowerer supplies its already-captured FX graphs to original `torch._inductor.compile`, and original callable
construction supplies `Program`'s positional interface. The target graph retains the original module metadata;
the original receive-reuse field and layout adaptation remain the channel-boundary operands.

After independent source review, four existing cases ran against compiler `fe3175b` and reference `4283357`.
CPU/MPS alias examples preserved state `1 → 2 → 3`; the two-processor, six-channel RunningMean instance reported
`(0.0, 0.0)` output/state error through both packaged AOT and the reference's direct FX entry. All processes
exited zero; tracked source and the seven pre-existing modified reference archives were preserved. Logs are
`compiler-emit/artifacts/fe3175b-4283357-{shared-cpu,shared-mps,running-mean-aot,running-mean-fx}.log`.
Original backend access preparation, native allocation/page/effect composition and categorical replacement remain
outstanding. These source handoffs and executions establish the stated local composition, rather than native
transport or full-domain performance.

### AOT result buffers and ordinary paging: operator clarification

Operator, 15 November 2026:

> jit-dependent-types cannot be an input to any function we use in the aot compilation, optimizatino, or lowering. so those types might plausibly exist formally but are not representable as a type iwthin our parsing, lowering, IR operations, optimization, etc.
> aot_already_compiled types are basically data or metadata returned from a thing we already processed that took an IR as an input.
> aot_gonna_be_compiled types are basically typed buffers whose values we expect ot be filled with the result of some thing we're gonna process that takes an IR as an input. all of this make sense?

On the claimed extra work of a "paged representation":

> but what is the paged representation underlying mesh? is it literally just 'know the length of the thing you're using, okay, we'll put it in some scatter of pages in the page table'?

Completed results and declared result buffers are operands of the AOT computation. Their producing operations fill
the buffers in the required dependency order. A consumer can use a buffer's known interface before another AOT
operation fills its contents; a consumer of those contents follows that producer. Page mappings and byte extents
locate the data, and its existing layout supplies interpretation. These descriptions introduce neither a new type
framework nor a content-specific page format.

The earlier assignment to expose every kernel choice ahead of terminal emission overexpanded the required
data-flow change into wholesale hoisting. Original ordered compilation already produces the completed/freed/removed
facts its later operations consume. That order remains valid. The integration passes the actual results and
declared buffers through those functions; paging supplies no reason to snapshot or simulate the ordered compiler.
Current native `Address` in `metal_ir.mm` demonstrates the distinction: it builds a load from the supplied frame
table without reading that table's entries during compilation. Its byte/element geometry is available earlier;
the original allocation producer supplies the frame contents later.

Compiler-emit `8eabdf1` records the reviewed local SIMD and CPU-template extractions; `622754c` records this
correction to the next assignment. The local extractions preserve their original consumers and finalization order.
Their source projection covers twenty-four Torch files. This checkpoint has source-review evidence, with no new
numerical execution, and leaves native allocation/transport composition and categorical replacement outstanding.

### Native build before the address-interface change

On 15 November, the ordinary Makefile targets rebuilt `rdma/libmesh.dylib`,
`libmesh_metal.dylib`, Dawn/Tint, `libmetal_ir.dylib` and `libmetal_recording.dylib` successfully.
Metal-microbench's tracked source was `ac2bd2ca`; Mesh was `c1059774`. The active recorder translation units
were `metal_capture.m` and `metal_program.m`. The later address-interface edits require their own build.
This establishes compilation and linking of the restored baseline, with no program execution or transport observation.

The two commands, run from metal-microbench, were `make ../dox/mesh/rdma/libmesh.dylib .build/libmesh_metal.dylib .build/provider/dawn-build/src/dawn/native/libwebgpu_dawn.a`
and `make .build/libmetal_ir.dylib .build/libmetal_recording.dylib`. Both exited zero.
Their logs are `metal-microbench/output_data/standalone-replacement-20261115/native-dependencies.log`
(SHA256 `55530c01dd4000851675b86e9402278701c0982ae6f878d5be713d323ab0aaff`) and
`native-libraries.log` (SHA256 `e813451b63763e500edbcbb4199b183f26f88740d67210246eacf1ea46b700da`).

### Original kernel-source return: reference execution

Compiler-emit `5886015` factors `MetalKernel.codegen_kernel_source` from the original numerical renderer;
ordinary `codegen_kernel` consumes that original buffer through its existing literal wrapper. The source producer
remains shared. The native consumer and directional page bindings are still being connected.

The existing `examples/shared.py` ran with `--device cpu` and `--device mps`, using
`PYTHONPATH=.:/Users/mdot/schedule-emit .venv/bin/python` and `--directory artifacts/5886015-shared`.
Both commands exited zero and passed the existing input identity, storage alias, offset/stride and numerical
assertions over three calls, with state `1 → 2 → 3`. Nine relevant installed compiler sources retained their
pre-run SHA256 values through both runs (`artifacts/5886015-source-before.sha256`).
The logs are `compiler-emit/artifacts/5886015-shared-cpu.log`
(SHA256 `b1ebe66ad0876c09f13d8a06f3e8087f04a19eb972b5da2a8e8464f16bd7f0c7`) and
`5886015-shared-mps.log` (SHA256 `7ddb87b6c54153cc1244f739394438bdcbb9add9891ea19874d14c06b95e68ad`).
These observations exercise the reference target through the current compiler extraction; native page execution
and categorical replacement remain outstanding.

### Shared address producer: reviewed and built checkpoint

Metal-microbench `a790f1f8` factors the existing `AddressFunctions` producer for its old caller and the standalone
byte-storage caller. The original `Address` arithmetic is shared. Independent review led to stable `Kernel.bindings`
storage across later appends and directional bindings selected from the original argument access bits.
Native-execution `db4d61e` exposes the actual function, parameter and binding views while retaining their native
Program owner. The numerical compiler continues to own tensor indexing, dtype and inplace correspondence.

The ordinary `make .build/libmetal_ir.dylib .build/libmetal_recording.dylib` build exited zero after those corrections;
all four changed native files retained their pre-build SHA256 values. The log and source hashes are
`metal-microbench/output_data/standalone-replacement-20261115/native-address-build.log` and
`address-source-before.sha256`. This establishes native compilation/linking, with final numerical
access/signature/dispatch composition still outstanding. The next direct composition uses the existing module/symbol
mechanism to accumulate address functions before one print, accounting for Tint's shared support declarations.
The old `mesh_paged` single-direction accessor remains part of the implementation to be displaced.

### Single collection, original access consumers and native binding load

Metal-microbench `4ced467f` replaces the separately printed per-value collections with one original module:
address requests return its assigned function symbols, and one final print supplies its Output/Parameter views.
Native-execution `b203938` places Python source/kernel associations in ordinary GC-visible Python views while
native handles retain their native Program. This corrects the cycle exposed by the actual kernel caller.
Compiler-emit `5a6bfbd` connects MPS load, store, atomic-add and reduction-store emission to the shared address
producer; `fdb5342` consumes the original removed-buffer result and places frame parameters after the original
numerical/scalar/dynamic/error arguments. Independent source review covered this producer/access/signature group.

The normal native library targets built `4ced467f` successfully, with all four source hashes unchanged
(`metal-microbench/output_data/standalone-replacement-20261115/4ced467f-native-build.log` and
`4ced467f-source-before.sha256`). The public `native_execution.pages.library(metal, mesh)` then built and loaded
its actual extension from compiler-emit's isolated Python, exiting zero. The log is
`compiler-emit/artifacts/4ced467f-b203938-pages-load.log`
(SHA256 `9fb7413cd50c7f250178dcc678642c42f1b0755ed396028bad5b186b2f47409f`). This is module compilation/load
evidence; native numerical/page/transport execution remains unobserved.

The next direct producer supplies the original numerical compiler's domain/access/effect and launch results
to canonical native registration. The old `lower_metal._Bodies` reconstructs access context by walking and
replaying LoopBody blocks. The current work instead exposes the original interpreter/indexing results of those
blocks' existing execution, with their original range, indirect and masked-call relationships. Native registration,
allocation and invocation remain outstanding, along with categorical replacement and full-domain coverage.

### Original access results and allocation geometry: source review

Compiler-emit `399c0b6` retains each original LoopBody invocation's interpreter, arguments, result,
parent occurrence and indexing dictionaries by reference. Review found that these operands preceded
`prepare_indexing` and omitted the internal indirect-index correspondence and reduction-store guard.
Commit `a70f94f` retains those results at their original producers, together with each actual access's
mode, interpreter/node/CSE scope and original CSEVariable or DeferredLine result. The original operation
trace retains wrapping operands and results; original CSE and masked-body selections remain accessible.
Independent source review found one original interpreter run and no added numerical replay or text parser.
The eight patches project exactly onto twenty-six original Torch files and pass AST and diff checks.

These retained generator occurrences are distinct from the final emitted accesses. The removal-filtered
view consumes the original deferred-store removal result; target lowering must also consume the original
CSE and masked selections. The typed launch-result handoff and external-kernel input/output correspondence
are the next adjacent producer work. No numerical execution of this checkpoint is recorded here.

Mesh `24b01623` exposes the existing page-quantum and page-count functions as `mesh_plan_quantum` and
`mesh_plan_quanta`; their original callers share the same unchanged definitions. C syntax and diff checks
passed. The standalone declaration producer is being paired with the actual prepared buffers, layouts,
storage identities and mapped context. This checkpoint establishes the shared geometry source, with its
native build/load, complete declarations, native relation/registration/allocation/invocation and categorical
replacement still outstanding. Paging continues to supply ordinary byte addressing over existing layouts.

### Geometry build and original producer execution

Mesh `24b01623` rebuilt through the ordinary `make ../dox/mesh/rdma/libmesh.dylib` target.
Native-execution `9a29175` built through its ordinary `make` target, then its public
`native_execution.pages.library(metal, mesh)` built and loaded the actual extension from compiler-emit's
isolated Python. All commands exited zero, and the source hashes stayed fixed. The geometry log is
`metal-microbench/output_data/standalone-replacement-20261115/24b01623-geometry-build.log`
(SHA256 `f5cdf9d9337db92b35d7a3ba06f42084c0b7820766502a31fd3af31f988ef6ad`); the bridge log is
`native-execution/artifacts/9a29175-native-build.log`
(`a1183b03831ea90139009e6d5875df57ddf1bd43ea365fb0e78157a6ea6d463e`); the extension-load log is
`compiler-emit/artifacts/24b01623-9a29175-pages-load.log`
(`9fb7413cd50c7f250178dcc678642c42f1b0755ed396028bad5b186b2f47409f`). This is build/load evidence.

Compiler-emit `8f82bca` retains the original typed MPS launch expressions, argument sources and complete
argument-definition result. It also retains the actual external-kernel input prototypes and output
correspondence at the original fake call and fallback construction. Existing reference samples then ran
through `compiler_emit.compile` and the actual `schedule_emit.runtime.Program`, with original compiler
caches disabled. RunningMean on MPS completed three calls with maximum output/state errors
`(2.384185791015625e-07, 1.0454601624268727e-07)`; Indexing on MPS completed two calls with errors `(0, 0)`.

The CPU Solve sample compiled its original external calls, then failed at the reference runtime's output
reconstruction: zero user leaves against a one-leaf output specification. The reference's pre-unlift output
signature and Torch's post-token-removal module are being traced at their producing interface. This failure
remains open at this receipt; the successful MPS observations do not cover it. The combined log is
`compiler-emit/artifacts/8f82bca-producers-reference.log`
(SHA256 `0a399143089c35527bc2dcada8a3258bdef66ea0ed883b05b7f5eba88b731004`). All twenty-six patched installed
source files retained their pre-run hashes, checked after the failure and before source editing resumed;
the hash list is `artifacts/8f82bca-producers-source-before.json`.

Source review also identified the remaining direct collective/wait consumers of ProcessKernelResult,
the declaration handoffs for constant subgraphs and later workspaces, and a native access-width mismatch:
byte-storage Values require the numerical access width supplied separately to the original access-map
producer. These are assigned producer/consumer corrections. Native relation registration and complete
native execution, complete domain coverage and categorical replacement remain outstanding.

### Corrected reference output interface and AOT geometry operand

Reference `b476887` removes Exported's copied output-kind description and the runtime parser that consumed
it. Program retains its complete Scheduled owner and the original post-unlift output tuple. That tuple
defines the result prefix passed to Torch's original pytree reconstruction; the reference adapter's
additional sender-layout results follow it. Original unlift remains responsible for token removal and
mutation copyback. The seven pre-existing dirty reference archives were unchanged.

With compiler-emit `e893bc1`, the same CPU Solve sample then completed through the actual reference runtime,
reporting errors `(7.152557373046875e-07, 6.765992183700297e-07)`. The existing shared MPS example also passed
its input-identity, alias, stride, offset and numerical assertions, with state `1 → 2 → 3`. All twenty-seven
patched Torch files and four reference source files retained their hashes across both runs. Logs:
`compiler-emit/artifacts/e893bc1-b476887-solve.log`
(SHA256 `438dea8bd8fb386da509dd233c184af7aa84c13be20483c18430a43b4e28cac5`) and
`e893bc1-b476887-shared-mps.log`
(`2b40470202a0ef02ee8415ce0e81f08d0d3952adb466cc7d4319a3e34e67daa6`).

Native `d165b94` retains the original prepared buffers, external input/output prototypes, folded-constant
correspondence and declared storage views. Review paired those existing owners; later workspace and new
inline-graph producers remain assigned. Native `c5c3ba1` adds direct bindings to canonical Domain, Index
and AccessMap owners, reusing the original mesh_graph.Context lifetime mechanism. Mesh `1e58644f` and
`9ed239e8` share the original symbolic and typed-operation mappings with their previous callers. These
checkpoints have source/syntax evidence; their composed native execution is outstanding.

Root's first declaration assignment unnecessarily required a mapped runtime context to obtain an AOT
page quantum. An actual local `mesh_attach` attempt returned ENOENT: the named bridge region was absent
(`compiler-emit/artifacts/24b01623-9a29175-context.log`). The compiler entry is being corrected to consume
the page-byte quantum as an explicit AOT operand. An attached caller can obtain it through the original
geometry function; a configured compilation can supply it directly. Runtime attachment, backing allocation
and mapping lifetime belong to their later owners. This withdraws the root-imposed live-context prerequisite
and introduces no fabricated context or frame contents.

### Canonical native relation build and load

Metal `9ed03578` built through the ordinary `libmetal_ir` and `libmetal_recording` Makefile targets.
The shared upstream producer retains the completed numerical access maps independently of byte-storage
Value types, and the original page-address operation uses u64 physical-address arithmetic. Native
`b7bc820` built through its ordinary Makefile and its public `pages.library` entry compiled and loaded
the actual canonical Domain/Index/AccessMap bindings. All commands exited zero; both changed Metal
source hashes and all three native binding source hashes remained fixed during their builds.

Logs: `metal-microbench/output_data/standalone-replacement-20261115/9ed03578-native-build.log`
(SHA256 `d2a4abc62ddbe7a6d6df173890e4ee340aa8e82812912f984c625ae691eb77af`),
`native-execution/artifacts/b7bc820-native-build.log`
(`b61dee69c0243e72341bc664e8d4744e1eef51a7de662ed2d02b9edcede890b9`), and
`native-execution/artifacts/b7bc820-binding-load.log`
(`35da620be1406a08575badd0a0887996ab97d2f8e346a80872ad9c9207763893`).
This establishes build/load evidence; the compiler-to-native emission and native numerical execution
remain unobserved at this receipt.

Review of the next composed caller found that repeated indirect-index uses must retain equality through
their original computed-value identity, alongside each invocation's actual assertion selection and bounds.
That correction is assigned to the original producer and direct relation consumer. Final native collection
composition must also consume the original unprinted address module through the existing Clone correspondence
before printing. The original scalar-double MPS binder produces local float bytes for setBytes; its binding
obligation is distinct from Tensor storage declaration. These are concrete unfinished interfaces, not new
semantic requirements on the page table.

### Public compiler-to-native registration

Compiler `9ccc520` corrects the indirect relation group by retaining each original use, actual assertion
selection and bounds, and one native representation per original computed value. Symbolic indexing,
typed arithmetic operands, masks and selected assertions consume that same value representation.

The public `native_execution.declarations.emit` entry then completed on the existing RunningMean and
Indexing sample modules, through `compiler_emit.prepare` and the actual reference scheduling entry.
The configured AOT page quantum was 65536 bytes; compilation needed no attached runtime or initialized
frame contents. Native `b7bc820` used the previously built Metal `9ed03578` library. RunningMean registered
three actual calls and thirteen access maps; Indexing registered three calls and eleven maps, including
four indirect-index uses. Both retained the exact Prepared and Scheduled objects. These counts describe
the observed instances, rather than a prescribed topology or evidence of whole-domain coverage.

Both commands exited zero. All forty-four recorded compiler, original Torch, reference, shared mapping
and native source hashes remained fixed through both calls. Logs are
`compiler-emit/artifacts/9ccc520-b7bc820-native-emit.log`
(SHA256 `e10c2d8389063e65e07e318ed6756550a697b45a97b01ca73e14f59c2117630d`) and
`9ccc520-b7bc820-indexing-native-emit.log`
(`a7aa44a444fbdf85250683de01bd0668b10f02d0ba53f8256afaa521ef01a7c9`). The source manifest is
`9ccc520-b7bc820-native-emit-source-before.json`
(`b08e14538b1ae2f82dd91db79976d0c0a23f01a52dda2cb70e3597848136838e`).

This observes actual declaration, source emission, relation lowering and canonical operation registration.
Final constructed-kernel collection composition, final printing, runtime binding/allocation and native
execution remain unfinished. The next paired source work consumes the existing Clone correspondence and
original selected call/signature owners at that final emission boundary.

### Final collection producer and binding build

Metal `0a05feb5` composes the original unprinted address module through the original Clone correspondence
before appending the original protocol and numerical functions. Metal `581f13fb` corrects the adjacent
constructor's handling of signature arguments with explicit access bits of zero: the signature remains,
and page relations follow the existing access bits. Both checkpoints built through the ordinary Metal
Makefile targets. The four recorded source hashes matched at completion of the latter build.
Its log is `metal-microbench/output_data/standalone-replacement-20261115/581f13fb-native-build.log`
(SHA256 `e4a963c77a83e944078de26f38f40da4d3d5b63913b9aac76c8c6cdf9680b95d`).

Native `8489075` exposes the original device, construction, collection composition, projected Function
and Parameter operands, and terminal source/configuration interface. Its public `pages.library` built
and loaded against Metal `581f13fb`, exit zero, with all five recorded native source hashes unchanged.
The log is `native-execution/artifacts/8489075-binding-load.log`
(`83c7d247e0f8469485e851d69928ef9f8684dc9d0e5befcdb6aa692d83a729da`). The paired original compiler
preparation/final-render split is under review; no composed execution is established by this binding load.

Independent review identified a further retained-view obligation before later native lowering: readiness
and table metadata reside inline in Kernel records that InsertKernels moves, and the new collection kernel
index plus the existing composition map need the original Renumber correspondence. The original producer
must preserve these results and associations. A restriction requiring consumers to abandon the retained
views after lowering would leave this requirement unfinished. This is an ordinary object-lifetime and
index-correspondence correction, rather than an additional semantic requirement on paged storage.

### Actual final emission and ordinary reference execution

Compiler `1ac9861` separates original schedule emission from terminal wrapper printing and numerical body
preparation from final kernel rendering. Ordinary callers immediately compose those same functions.
Native callers consume the actual selected call line, construct the original native operation, compose
its complete callable collection, then render once and call the original native Configure boundary.
The original emitter resolves DeferredLine removal at its original preparation point; its complete
ValueWithLineMap result remains the final renderer's operand. Header requirements now belong to each
original kernel rather than a mutable class-wide set.

With native `8489075` and built Metal `581f13fb`, the public `declarations.emit` completed on RunningMean
and Indexing through actual construction, final printing and Metal configuration. Each instance retained
three actual call occurrences; their access-map counts remained thirteen and eleven, respectively. The
exact Prepared and Scheduled owners remained retained, and Indexing exercised four indirect uses.
The original MPS device was an explicit compilation operand, as was the 65536-byte page quantum.

After the shared original emitter change, the existing shared MPS example passed its numerical,
input-identity, alias, stride, offset and persistent-state checks, with state `1 → 2 → 3`. The existing
CPU Solve sample completed through the actual reference Program with errors
`(8.940696716308594e-08, 2.2395714356100817e-07)`. All four commands exited zero and all forty-four recorded
source hashes remained unchanged. Logs under `compiler-emit/artifacts/`:

- `1ac9861-8489075-native-emit.log`: SHA256 `7b24b1f87ea749e8c52f84cc6275b6caea28475e21c16f389cd60dac07fb96ce`.
- `1ac9861-8489075-indexing-native-emit.log`: `a57a318634a82782159a43d1b52c08ed4fc6b15fddd299b732817d7e12dc5860`.
- `1ac9861-shared-mps.log`: `04c2a0bb0e4c4680d132d9c225c38a71464b02fdb6820af4b8153071389553c0`.
- `1ac9861-solve-cpu.log`: `fb20e9c090c9e0b5233d154204c284a595b2525fa31363a34d1ff1194cb8ed6a`.

These observations cover actual final compilation/configuration and the ordinary reference executions.
Protocol helper invocation, later-pass emission, original scalar-double binding, retained views through
later lowering, native allocation/execution and complete domain coverage remain outstanding. The next
assignments consume the existing typed protocol operands in the renderer and correct the original native
result owners before adding their later lowering consumer. Categorical Mesh replacement is still pending.

### Assignment wording and duplicated composition correction

The operator asked where a “protocol renderer” appeared in the requirements. Root introduced that phrase
in assignments; the request itself specifies composition of numerical prologues/epilogues and acquisition/
publication through shared access relations (composition item 5 and its complete-replacement section;
plan-to-bridge R4). The original ModulePass already constructs the argument bindings and the acquisition,
numerical-call and release sequence.

Root's source review found that the uncommitted MPS `render_protocol_entry` draft was constructing that
sequence and selecting operand bindings again. Root stopped that implementation and redirected the paired
assignment to the original composition owner and its complete result. The original Torch emitter supplies
its numerical body and actual ABI; the shared composition producer supplies the entry and bindings for
final emission. Compiler and native agents inspected the existing IR/call/printer interfaces to make
that handoff concrete. The draft was still present at this review; compiler `b8b39df` subsequently removed
it. The new phrase confers no requirement or independent implementation responsibility. The prior
`1ac9861`/`8489075` observations cover their recorded checkpoint, rather than this later draft.

### Standard memory views across language boundaries

The operator directed review of DLPack and Python's memory/buffer-view APIs after asking why successive
functions would drop or truncate operands. Python's buffer protocol exposes existing memory together with
its extent, layout and exporting owner; `memoryview` makes that storage available directly in Python.
DLPack supplies the corresponding standard tensor-memory descriptor and lifetime handoff, including Metal
storage. The same stored result can therefore serve Python and native consumers through views; the language
boundary does not itself require payload serialization, reduced metadata records or a second inference pass.
Existing source-program owners remain retained alongside their storage views.

Sources: [Python 3.12 buffer protocol](https://docs.python.org/3.12/c-api/buffer.html),
[DLPack Python specification](https://dmlc.github.io/dlpack/latest/python_spec.html), and
[DLPack's original header](https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h).
These are memory-sharing mechanisms, rather than a replacement IR schema.

An actual observation in compiler-emit's Python 3.12/Torch 2.14 environment passed an `array('q')` through
`memoryview`, `torch.frombuffer`, a strided tensor view and `torch.from_dlpack(..., copy=False)`.
The strided consumer retained the same data pointer and stride `(2,)`; writes through the Python buffer
and tensor were visible through both views, and the consumer remained valid after the earlier Python names
were deleted. A separate actual MPS DLPack handoff retained the same data pointer, shape `(3, 2)` and strides
`(4, 2)`; its writes changed the original MPS allocation, and its owner remained live after earlier names
were deleted. Host readback was used only to observe the MPS values. These are observations of the original
APIs, not a claim that all Mesh interfaces already use them correctly.

Logs: `compiler-emit/artifacts/buffer-dlpack-view-20261115.log`
(SHA256 `8c9cc67c7f4bf2a21a5f4b6d6aac5b84b3616e7c2ec1efebd21e703aa6af3edf`) and
`dlpack-mps-view-20261115.log`
(`f5ce26b5cc93b664498d3cd1ba214d990ee6e2f53d9099ee5cb0e7b0df74e1b6`).

### Data-flow and review memorandum

Operator, 15 November 2026:

> so what's the data flow we're supposed to be doing?

> if it has any departures from the code you were just writing: why? how did the literal content of the assigned task slip away multiple times and what note-taking and review standards does this require for us?

The operator subsequently asked for the permitted data-flow steps, their order and the three independent
demonstration repositories. The earlier ordering instruction remains literal:

> then we halt all work on anything related to any consumers until weh ave made the correct data structure at the producer side. then we only correct direct functional neighors of the producer in our callgraph, and so on.

The three repositories have distinct assigned roles:

| Repository | Functionality demonstrated in isolation |
|---|---|
| `/Users/mdot/schedule-emit` | Export, placement, processor assignment and scheduling composed as ordinary functions returning a retained `Scheduled`; original numerical lowering and `Program(scheduled, functions)` make that composition executable. |
| `/Users/mdot/compiler-emit` | Original upstream numerical compilation split at its actual preparation/result/emission boundaries, exposing complete producer results for composition before final emission. |
| `/Users/mdot/native-execution` | The independent execution consumer of those complete compiled results and prepared storage/effect bindings; its assigned complete native execution remains unfinished. |

The implementation checklist follows these data dependencies:

1. **Produce the scheduled source program.** The reference's `export → place → on_devices → schedule →
   argument_buffers` composition returns `Scheduled` with the original exported program and explicit bindings
   (`schedule_emit/driver.py::compile_module`; composition item 2).
2. **Produce the complete numerical compilation result.** The original upstream preparation and schedule-emission
   functions expose their actual graphs, operations, accesses, calls, parameters and allocation results before
   terminal emission. `compiler_emit.prepare` returns the original graph owners; original schedule emission
   produces the later call and workspace results. Both retain the scheduled operand (composition numerical-lowerer boundary).
3. **Produce the joint solved operands.** The shared graph, allocation and scheduling algorithms consume the
   complete result and any newly expanded operations, retaining source correspondence and previously solved
   results. Their output supplies the dependencies, lifetimes, copies, page bindings, transfers and execution
   order for parameters, numerical operands and calls (AOT A4–A6/A8–A10; plan-to-bridge R1–R5).
4. **Compose and emit the target program.** Original target construction consumes the numerical bodies and those
   solved operands, including prologue/epilogue and transport effects. The original final emitter and target
   compiler produce the executable and its complete bindings (composition items 5–7 and complete replacement).
5. **Consume that compiled result.** The actual reference Program and independent native entry realize their
   specified target behavior from the same complete result. Invocation supplies changing inputs through views;
   the archive retains source and derived sections (composition executable-reference boundary; AOT A9).
6. **Replace the corresponding Mesh implementations.** The demonstrated standalone implementations and complete
   interfaces displace the defective implementations through the assigned categorical replacement.

The current work is the incomplete producer-to-direct-consumer handoff in steps 2–4. Original parameter and
allocation retention, relation registration and first-pass configuration have their separate recorded evidence;
they do not establish a completed joint solve or composed native program. The next work completes the original
producer's required result, then connects its immediate consumer. Runtime integration follows that completed
interface. “Make the native caller work” supplies neither an additional step nor an independent assignment.

Complete source and derived results remain accessible by view across these boundaries, including Python/native
interfaces. Page mappings and byte extents locate stored results; their existing layouts supply interpretation.

Root's assignment and review errors were concrete. Requiring a live runtime context to obtain an AOT page size
substituted an existing caller's circumstances for an explicit operand. Assigning a new “protocol renderer”
turned shorthand into a second implementation of the existing entry-construction algorithm. Reviewing metadata
retention without simultaneously tracing its direct consumer allowed a duplicate dtype map and source-copying
adapters to look like ordinary integration work. These decisions originated in the implementation/review process;
the operator's assignment already supplied the responsibility boundaries.

Source review at compiler `b8b39df`, native `f6e42c6` and Metal `f8facce2` establishes the following limited account:

- The compiler retains the actual parameter and local-allocation results. Declaration emission and access widths
  use the same parameter objects, and the duplicate entry builder has been removed.
- Metal's original module construction returns its complete composition result before original final emission.
  The new compiler caller still renders pass zero through its original numerical emitter; the named numerical-call
  connection to that composition result, workgroup placement, writer coherence and scalar binding remain open.
- `native_execution/pages.cpp::Emitted` and `Collection.source`, `metal_ir.mm::metal_ir_upstream_emitted` and
  `metal_transport.mm::EmitModulePass` still copy emitted source across adjacent interfaces. Retaining its owner
  elsewhere does not make those copies views.

These are source observations. Earlier execution receipts establish only their stated revisions and exercised
properties. The present group has source review and the agents' recorded syntax/projection checks; this memorandum
reports no new runtime observation or completed replacement.

For subsequent assignments, root carries the literal instruction and its cited mechanism alongside the concrete
producer, complete result and storage owner, direct consumer, and implementation being displaced. The existing
two four-sentence memoranda remain the assignment preface. An assignment describes the missing handoff using the
source's vocabulary; its working name supplies no extra requirement.

Review traces the changed producer and direct consumer together: which result is genuinely new, which existing
result is read by view, where its ownership and source correspondence remain accessible, and which shared function
performs each algorithm. A retained field counts toward the handoff when its consumer uses it. Source review
establishes that local composition; execution receipts establish the properties actually observed. Revisioned
findings and remaining obligations are recorded separately from verbatim requirements, with later corrections
linked to earlier findings. Commits preserve work in progress as well as completed work; their status follows the
remaining obligations. This is the root's review practice for the existing assignment, not another change request.

### Compiler-library selection before executable emission

Operator:

> "4. Compose and emit the executable" choose ahaed of time how you'll do this, and the answer better not be all original string manipulation code written in python. if there are any preexisting libraries for this kind of transpilation, we will use them with as little bespoke work as possible given our requirements

The operator clarified:

> if a fxgraph2source final lowering is strictly required for the fundamental api we're working with, we can literally do that...

Final source emission from the complete annotated graph is permitted. The existing upstream emitter supplies
that output where the target compiler API consumes source. This instruction constrains when source becomes
the result; it does not create a requirement to avoid source by adding another representation or linker.

Root selects original TorchInductor numerical emission, Tint typed entry construction and MSL emission, and
Apple Metal's private static function linkage for the current numerical/composition boundary. The original
numerical compiler emits an ordinary callable definition from its retained result. Tint constructs the entry's
typed calls, bindings and acquisition/publication through the existing composition function. Apple's
`MTL4StaticLinkingDescriptor.privateFunctionDescriptors`, supplied through
`MTL4ComputePipelineDescriptor.staticLinkingDescriptor`, links their library functions at Metal IR level.
`MTL4LibraryFunctionDescriptor` supplies the library and function identity. The installed SDK's
`MTL4LinkingDescriptor.h` expressly permits private linkage on devices without function-pointer support.
The existing `metal_lower.mm::PipelineDescriptor` is the immediate pipeline-descriptor consumer.

The producer interface is an ordinary function definition with its original parameter, allocation, access and
effect owners. MSL's kernel-entry and ordinary-function distinction requires the original numerical emitter to
emit the callable form; it is not obtained by editing an already-emitted kernel string. Actual workgroup storage
is allocated at the entry and supplied to the callee. The shared composition consumes actual selected writer
effects; static linkage does not establish system visibility by itself. A typed external function declaration
connects the definition to the existing Tint UserCall. Installed Tint currently requires a Function body, so
supporting that ordinary declaration in its original function/clone/validation/writer interfaces is an explicit
remaining library-interface change. Its physical ABI must also represent the original reference arguments,
narrow/bfloat pointer types and original error-buffer type; a pointer-shaped substitute or type-string parser
does not establish that interface. LLVM's function declaration/definition distinction supplies the terminology.

This selection uses the existing numerical emitter and compiler writer for their own final source output.
Mesh's work connects their complete operands, declarations and library objects. It does not introduce a Python
source-assembly compiler, a generated-source parser, a second entry-construction algorithm or a runtime function
dispatcher. The whole assigned numerical domain, workgroup behavior, precision, aliases and effects remain the
obligations; the linkage APIs are source evidence of a mechanism, not proof those obligations have been implemented.
Host/ANE realization remains part of the original target coverage; this selects the currently missing Metal
composition boundary, rather than replacing the whole domain with a GPU subset.

The library review also examined Torch-MLIR/IREE. Its original FX importer, borrowed in-memory MLIR module API,
SPIR-V passes and SPIRV-Cross Metal emission are real existing implementations. The checked IREE Metal target
can retain integer capabilities unsupported by the installed Tint SPIR-V reader. Substituting that importer is
therefore not an already-established complete connection to the present composition. Metal function stitching
supplies a function DAG; it does not directly represent the original entry's conditional acquisition/skip.
Neither alternative is being implemented alongside the selected static-linking path.

Sources: installed macOS SDK `Metal.framework/Headers/MTL4LinkingDescriptor.h`,
[`staticLinkingDescriptor`](https://developer.apple.com/documentation/metal/mtl4computepipelinedescriptor/staticlinkingdescriptor),
[`MTL4LibraryFunctionDescriptor`](https://developer.apple.com/documentation/metal/mtl4libraryfunctiondescriptor),
[MSL specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) §§4.4/5.1/5.3/6.10,
[LLVM function declarations](https://llvm.org/docs/LangRef.html#functions),
[IREE Metal target](https://github.com/iree-org/iree/blob/main/compiler/plugins/target/MetalSPIRV/MetalSPIRVTarget.cpp),
and [IREE in-memory MLIR interface](https://github.com/iree-org/iree/blob/main/compiler/bindings/c/iree/compiler/mlir_interop.h).

### Build of the original composition producer

Metal `f8facce2` built through the ordinary `.build/libmetal_ir.dylib` and
`.build/libmetal_recording.dylib` targets, exit zero; all eight recorded source hashes remained unchanged.
Native `f6e42c6` then built and loaded its actual `pages.library` against those libraries, exit zero, with
all five recorded native source hashes unchanged. These observations cover the retained ModulePass producer
and its existing binding, preceding implementation of the selected callable/static-link interface.

Logs: `metal-microbench/output_data/standalone-replacement-20261115/f8facce2-native-build.log`
(SHA256 `e4a963c77a83e944078de26f38f40da4d3d5b63913b9aac76c8c6cdf9680b95d`) and
`native-execution/artifacts/f6e42c6-f8facce2-binding-load.log`
(`f3de42aa093e2d4765b6b2a71c3e12158d2f3c1d69193e1b01a3a3da787491f0`).
The corresponding `*-before.json` files retain the source manifests. Native numerical/transport execution
and categorical replacement remain outstanding.

### Original callable-form producer and ordinary execution

Compiler `f39348a` adds an explicit function form to the same original MPS emitter. Its original
`codegen_kernel_source` returns the complete MetalKernel; ordinary `codegen_kernel` consumes its source.
Original workgroup allocations produce their actual reference/pointer parameters, while local allocations
stay local. Original parameters, builtin operands and the reduction threadgroup bound remain available.
The independent source review covered all twelve original allocation callers and the ordinary emission caller.
The agent's eight-patch projection reproduced all twenty-seven installed source files, with AST/diff checks.

Root then ran the existing shared MPS example through actual compiler-emit and reference Program. Its numerical,
input-identity, alias, stride, offset and persistent-state assertions passed, with states `1 → 2 → 3` and exit zero.
All forty recorded source hashes remained unchanged. Log: `compiler-emit/artifacts/f39348a-shared-mps.log`
(SHA256 `7094e1388ed10d69ea17244cb5f804ce9b890e6faf0be9d192711ab6b91ae286`);
`f39348a-shared-mps-before.json` records the sources. This observes the original kernel-form caller, not a linked
callable execution.

The remaining producer handoff is selected writer effects: preparation currently materializes the body before
construction selects those effects. The paired work retains the original store result through that selection
and supplies it to the original final emitter. Native callable declarations/linkage remain subsequent consumers.

### Selected store effects reach original final emission

Metal `fbbf5caa` extracts its original selected-writer storage comparison into `Released`, shared by
`Coherent`, `ConstructModulePass` and the upstream query. Native `f00e101` exposes the complete composition
result and its tables by reference, with read-only buffer views and the original operation owner. Its
`released(argument)` propagates an unavailable-result error before Boolean conversion. Root built the ordinary
Metal library targets and loaded the actual native extension successfully; all eight Metal and five native
source hashes remained unchanged. Logs are `metal-microbench/output_data/standalone-replacement-20261115/`
`fbbf5caa-native-build.log` and `native-execution/artifacts/f00e101-fbbf5caa-binding-load.log`, with adjacent
`*-before.json` source manifests.

Compiler `777c416` retains each original store, its operands, guard, source occurrence and removal decision.
The original deferred-line printer receives each call's composition effects before materializing the final
body. The ordinary emitter uses the same printer. Independent source review covered plain/reduction stores,
prefix/splice views, original removal/CSE timing and per-call ownership; the eight patch projections reproduce
all twenty-eight installed original source files.

Root exercised that compiler with native `f00e101` and Metal `fbbf5caa`. The existing shared MPS example passed
its numerical, identity, alias, stride, offset and persistent-state assertions (`1 → 2 → 3`). Public native
declaration/emission of the existing RunningMean sample completed three operations, retaining the identical
Prepared and Scheduled owners and exposing all three complete composition structs as eighty-byte read-only
buffer views. Its calls retained two, one and one store effects; all selections were false. This run therefore
observes the complete handoff and ordinary final emission, while selected system-coherent stores and native
invocation remain unobserved. All fifty-three recorded source hashes remained unchanged.

Logs: `compiler-emit/artifacts/777c416-shared-mps.log`
(SHA256 `9a9613b801d4ae5dd55d7fc6049030a6a53c8d0e006751861cfceb5584f2d64d`) and
`compiler-emit/artifacts/777c416-native-emit.log`
(`7942b6fb549099f7f93ade945006318b9a60198dcf6d0edf11d023a3ced60542`);
`777c416-source-before.json` retains the source manifest.

The remaining original atomic-helper interface has a concrete distinction: narrow element helpers execute
packed thirty-two-bit read-modify-write operations, while the current relation records the numerical element
width. Their full physical access result and selected system visibility require the original helper's
interface; the plain-store transform establishes neither. Callable declaration/static linkage and the full
native program remain subsequent incomplete interfaces.

The subsequent source audit located those effects in the original `c10/metal/atomic.h`: float/int use one
thirty-two-bit atomic, the packed helper owns pointer realignment and word/lane selection, bool has its own
predicate and address formula, long uses two word additions with a carry dependency, and float2 uses two
component additions. `AtomicType::type` alone therefore also fails to describe the complete operations.
Metal's existing `LoweredBody::Access` derives physical effects from actual typed atomic instructions and their
returned-value uses. The original header presently exposes neither that typed body nor a host AOT effect
result. The next producer interface must retain the helper's actual address, atomic operation and result
dependencies where they are produced, with its implementation consuming the same result. This is a source
finding about a missing interface, rather than evidence for replacing it with a numerical-dtype classifier or
another copy of the helper. The audit performed no further builds or workloads and changed no source.

### Original atomic library compilation supplies a typed body

The next audit found an original compiler interface that supersedes the preceding unavailable-body observation.
Root installed Apple's Metal Toolchain `17E188` through the existing Xcode component command, with the per-command
`DEVELOPER_DIR=/Applications/Xcode-26.4.1.app/Contents/Developer`. The compiler agent compiled the unmodified
`c10/metal/atomic.h` directly using the original Metal compiler (`-include metal_stdlib`, `-femit-all-decls`,
`-O0`, `-g`, `-emit-llvm`). This is compilation of original library input. The numerical emitter's generated
output was not the input to that analysis.

Installed LLVM `21.1.8` successfully read the resulting bitcode through `LLVMParseBitcodeInContext2`. Its original
function, block, instruction and use APIs exposed thirty-three definitions, including bool's conditional regions,
packed load/CAS loops, long's used first atomic result and unused second result, and float2's separate component
operations. Original debug metadata retains the header correspondence. The atomics are AIR calls; their target
effect contract remains a direct-consumer obligation. Successful parsing establishes access to the actual body,
rather than a completed physical-effect analysis or native execution.

Artifacts: `compiler-emit/artifacts/atomic-original-17E188/atomic.bc` (69,504 bytes,
SHA256 `9c8d228e20f8f1bc0b0259b8a2dddcd4d7719ab4e75cac1a103192548c30f34f`) and `reader.json`
(`0644d1876fa28feed3cefd545764530189dfc623d1da359459912e41be6c049a`). The JSON is an observation receipt;
the implementation being connected retains the complete original compiler result and native module owners.

The frontend agent also compiled the original scope interfaces. Existing c10 and public Metal atomic wrappers
reject a system-coherent pointer with explicit qualifier-loss diagnostics. The same compiler's original atomic
intrinsics accept that pointer and scope `3`; their emitted AIR retains it. A defaulted scope template preserves
pointer qualification through the original byte/back casts, with ordinary scope `2` and selected scope `3`.
These compilation observations justify the small original-header interface change now assigned: carry scope and
pointer qualification through the existing numerical algorithms and their leaf operations. They establish no
GPU/system-memory execution result. Evidence and exact commands are in
`metal-microbench/output_data/standalone-replacement-20261115/atomic-coherence-17E188/README.md`, with source and
artifact hashes in its `sha256.json`. Original numerical algorithms and the mounted standard library were unchanged
for those observations.

Sources in their actual roles: Apple's [component installation](https://developer.apple.com/documentation/xcode/downloading-and-installing-additional-xcode-components)
and [source-to-IR compiler interface](https://developer.apple.com/documentation/metal/building-a-shader-library-by-precompiling-source-files),
the installed compiler's options and `metal_atomic` signatures, and LLVM's original
[`BitcodeReader`](https://llvm.org/docs/doxygen/BitcodeReader_8h_source.html) owner/result API.

### Retained compiler result and atomic scope implementation

Compiler `dbe6936` adds the small native owner over LLVM's original `MemoryBuffer`, `LLVMContext` and `Module`.
Its Python interface returns original function, block, instruction, use and debug-location views, including
inlined-at locations, with the owner retained. The public `compile_header` returns that owner alongside the
complete compiler invocation result. Root built and loaded the binding successfully against the preserved
original atomic bitcode. Its read-only buffer, function, location and use views remained usable after the initial
Python reference was dropped. All three recorded source hashes stayed unchanged. Log:
`compiler-emit/artifacts/dbe6936-llvm-binding-load.log`, SHA256
`f631a08ae09fc2baca5e51416a19a2bc746d9ef083b9e59ffac2fa5c4518d22d`.

Header checkpoint `9a1d85b` carries the explicit compile-time scope and pointer qualification through the original
c10 atomic implementations. Original packed, bool, carry and vector algorithms remain single implementations;
the original direct-helper signatures forward into their scoped implementations. The leaf calls supply scope
to the original compiler intrinsics. All eleven add and eight binary specializations compiled at device and
system scope, along with the original direct-helper calling forms. The source and compilation receipt are in
`compiler-emit/artifacts/atomic-scope-17E188/`. This source received independent review.

Compiler `cba28b1` pairs that header with the existing final store printer. Canonical selected effects now select
the explicit scope and qualified pointer for atomic calls as well as the existing plain-store form. Ordinary
call text remains unchanged. The nine patch projections reproduce all twenty-nine installed original source
files; the agent's AST/diff checks and paired source review passed.

Root then exercised the public `compile_header` interface on the explicit original-helper instantiations.
Compilation and the original reader succeeded: 144 function definitions, a read-only 139,120-byte bitcode buffer,
and 2,096 source locations, with retained function/location views after dropping the initial result reference.
The existing Indexing sample also ran through the actual original compiler and `schedule_emit.runtime.Program`;
both calls agreed with the original module with maximum absolute error zero. All fifty-eight recorded source
hashes stayed unchanged. Logs: `compiler-emit/artifacts/cba28b1-atomic-public.log`
(SHA256 `7f40e57d4af4c55e8107c6985a6bcc5367478c311d3722b34b68ccc077db4de1`) and
`cba28b1-indexing-mps.log` (`55967d8b15e37ca8c2009df6ba3a4caa6308280458132f97025974309b87eed4`);
`cba28b1-source-before.json` records the sources.

These results establish the compiled-library body/owner interface, source compilation of the scope variants,
and ordinary MPS numerical execution for the stated instance. The physical AIR effect contract still has to
enter the existing relation consumer, and actual store owners still need their corresponding retained helper
compiler result. The Indexing run exercises ordinary device scope. System-memory GPU behavior, callable static
linkage, full native execution and categorical replacement remain unfinished.

### Actual store-to-compiled-function handoff

Compiler `5366ccf` adds three source-owned Clang annotations to the original atomic leaf functions. Their
constant operands come from the compiler's actual field offset, storage size, scope and order expressions.
Compiler `a94feec` then supplies the actual numerical stores to the existing MPS emitter, compiles their
default/system definitions together, and retains the complete compiler result and original LLVM Function views
before Relations. Native `7c01bb4` forwards the explicit compiler keyword bundle through `declarations.emit`.
The single original atomic-call printer serves both these definitions and final numerical source emission.
Independent source review found no further concrete producer/direct-caller defect in this group.

Root exercised that public `declarations.emit` entry with the existing Indexing sample. It produced three calls
and operations, including one atomic store with both compiled scope definitions. Every call and relation retained
the same complete compiler result; the actual store retained its own definitions and each call selected its own
variant. LLVM's original symbol lookup returned those same Function objects, each with its actual three arguments.
The original compiler exited zero and returned an 11,680-byte read-only bitcode buffer. All fifty-nine recorded
source hashes stayed unchanged. Log: `compiler-emit/artifacts/a94feec-native-indexing.log`, SHA256
`f4c87d139576d66ac87a10df38188321fc78b6e55b064c1751bfd5a992169a41`; source manifest:
`artifacts/a94feec-native-indexing/source-before.json`; bitcode SHA256
`d95ee8fe40aa7ab18cba41d118b560deb3bab6bf826163c4885450ece6d54682`.

This observation establishes the producer-to-direct-consumer ownership and identity handoff and successful source
compilation for that instance. Relations still computes physical accesses using numerical scalar width and marks
atomic adds with `write=2`; it has yet to consume the compiled body's physical effects and returned-value dependencies.
The observed final call selected ordinary device scope. This emission did not execute the native program, establish
system-memory visibility or demonstrate static linkage. Those implementation obligations remain open.

The subsequent bounded library review found the next missing interface: the annotated AIR leaf's actual call
operands and constant payload must supply LLVM's typed memory-location/effect operands. Original ScalarEvolution,
Loop, Dominator and MemorySSA analyses are available components; their presence alone does not supply that adapter.
LLVM 21's `MemoryLocation::getForArgument` handles its known intrinsics/library contracts rather than these annotations.
Its Attributor pointer analysis uses integer offset bins and rejects `ptrtoint`; Polly's existing SCEVAffinator
requires actual Scop domains, parameters, loops and assumption/invalidation results. Neither establishes an exact
replacement for the current physical relation without the missing inputs and interface. The original `opt`
invocation also rejected `air64_v28` before running analysis; the module's target and layout were preserved.
That CLI failure is not a result about direct library analysis. The next local assignment connects the annotated
leaf to actual typed effects and retained analysis owners. Guard/domain and interprocedural composition remain
explicit further obligations. Long-add's observed internal return-to-carry dependence does not itself prove that
the whole integer addition is noncommutative.

Source checks: LLVM 21.1.8 [`MemoryLocation.cpp`](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/lib/Analysis/MemoryLocation.cpp),
[`AttributorAttributes.cpp`](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/lib/Transforms/IPO/AttributorAttributes.cpp),
and Polly [`SCEVAffinator.cpp`](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/polly/lib/Support/SCEVAffinator.cpp).
Command receipt: `compiler-emit/artifacts/a94feec-atomic-existing-analyses.log`.

### Original analysis owners and physical leaf locations

Review found that the scoped wrappers had dropped the standard library's original volatile pointer qualification.
Compiler `a3d5848` restores `const volatile` on the load object and `volatile` on the compare-exchange/add objects.
The existing full instantiation caller compiles successfully; all eighteen AIR atomic calls again retain the
original volatile flag, with device/system scopes unchanged. The three parameter changes preserve the bodies
and annotations. The receipt is `compiler-emit/artifacts/atomic-volatile-17E188/receipt.json`.

Compiler `7015b2f` connects actual annotated leaf calls to LLVM MemoryLocation/ModRef values and the original
CallGraph, ScalarEvolution, Loop, Dominator and MemorySSA owners. Compare-exchange's caller `expected` operand
has its own read/write location, matching the original wrapper's unconditional copyback. Actual zero member
offsets use the original module directly by view. An actual nonzero member-offset transformation uses LLVM's
CloneModule/ValueToValueMap and GEP construction, retaining its input and derived result.

Root built and exercised the public `declarations.emit` and `compile_header` entries. Direct library analysis
succeeded on the unchanged `air64_v28-apple-macosx26.0.0` target: the full original fixture exposed 72 leaf calls,
30 caller-expected read/write locations, 62 returned-value uses, and 72 original MemorySSA and ScalarEvolution
results. Retained views survived dropping the initial Python owners. The module stayed identical by reference.
Both ordinary Indexing calls through the original compiler and reference Program had maximum absolute error
zero after the volatile correction; all sixty recorded source hashes stayed unchanged.

Logs: `compiler-emit/artifacts/7015b2f-atomic-effects.log`, SHA256
`935155d687692ad1ab93f72cf4c42793529eaa2ce4746c30527d4eb8c7859b19`, and `7015b2f-indexing-mps.log`,
`6d0751db9cd561077784b1c7c47d9f4d4b9fc844a7f9df940c7df81f2889c4c0`. The corresponding `source-before.json`
is under `artifacts/7015b2f-atomic-effects/`. These are local physical-effect results; the side table is not yet an
AA provider or a guarded Mesh access relation. Native execution, system-memory visibility, static linkage and
categorical replacement remain unfinished.

### Existing SSA and alias-analysis consumers

The frontend's single original-compiler observation at `1e87ffe` used `-O1 -fno-inline -g` on the existing helper
fixture. It removed 216 pointer-spill allocas while retaining 144 function definitions, the eighteen annotated
leaves, volatile flags, scopes, compare-exchange feedback and long carry. Sixty address-taken cells remain.
This supplies ordinary SSA through the existing compiler; its receipt is
`compiler-emit/artifacts/atomic-ssa-17E188/receipt.json`. Final executable optimization remains a separate consumer.

Compiler `2305a34` registers a narrow AAResultBase implementation with the original default AAManager before
MemorySSA. It references the retained effect records and delegates location overlap to LLVM's existing alias
query with the same query context. Its atomic-load and relaxed compare-exchange/add distinction follows LLVM
21's [atomic ModRef implementation](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/lib/Analysis/AliasAnalysis.cpp#L520).
Global memory effects, call-to-call queries, volatile flags and raw Metal scope/order operands retain their
existing meaning. Independent review covered registration, preservation/invalidation and owner lifetimes.

Root built and exercised `declarations.emit` using those explicit SSA compiler arguments, then reused the
frontend's existing complete bitcode fixture for original AA queries. Eight same-function physical queries
returned NoModRef for disjoint locations; 132 others returned ModRef, and all 72 calls retained global ModRef.
All thirty caller-expected locations remained read/write. Borrowed analysis views stayed usable after dropping
the initial owners. The AIR target and all sixty recorded source hashes remained unchanged. Log:
`compiler-emit/artifacts/2305a34-atomic-aa.log`, SHA256
`884b9b24c6f18b21d969b49facec538e2fbb797ff462256535172dff5893f54e`.
The retained result now feeds actual original alias analysis; entry-wide argument/control correspondence and
guarded affine Mesh access relations remain unfinished.

### Actual entry operands through original inlining

Compiler `ff7a3cd` retains the supplied original entry Functions, source CallGraph and selected helper callsites,
then applies LLVM's original AlwaysInliner through ModulePassManager to a derived module. Actual primitive calls
remain explicit; the same effect extraction and analysis construction follow pass invalidation. `e7df3f0` removes
a redundant local viability check so the original inliner owns that decision. The source module stays retained;
the clone map and original inlined-at metadata have their respective meanings rather than an asserted one-to-one
mapping of every optimized instruction.

Root exercised public `declarations.emit` on the existing Indexing program with float32 and packed float16 inputs.
Every annotated primitive call appeared in its actual derived entry. The float case supplied a four-byte add
effect per scope. The half case supplied four-byte load/compare-exchange locations and a separate four-byte
expected-pointer location per scope, with the original helper and entry source chains intact. Original callgraphs,
compiled buffers and entry identities remained available; borrowed effect/analysis views survived dropping the
initial result owners. The build and emission exited zero with all sixty recorded source hashes unchanged.

Log: `compiler-emit/artifacts/e7df3f0-entry-effects.log`, SHA256
`7b0e1f917528077ad4c6c7c3a3e99109fe64a5fb4b7e33006dba3a5c685113bc`; manifest:
`artifacts/e7df3f0-entry-effects/source-before.json`. This supplies entry-local physical effects and their operand
computations. Relations still uses numerical scalar width; the next connection must consume those actual address
expressions, existing domain/argument bindings and guards through the original affine machinery. Native dispatch
and complete replacement remain open.

### Existing ISL results exposed to the original affine consumer

Mesh `ad5bb883` exposes the original context, domain box, index value and predicate through borrowed-pointer
accessors in `mesh-graph.h/.c`. The original context mutex supplies the locked interval; `mesh_graph_index_take_locked`
delegates result ownership to the existing `index_made`. The complete owners and lock/release order are specified in
[graph-isl-views.md](../../design/graph-isl-views.md). The change adds 23 source lines and a separate interface note.
Root and executor reviewed the direct interface; C11 source and C++17/ISL-header checks passed.

Root built the ordinary `make -C rdma libmesh.dylib` target, exit zero, with all six recorded source hashes unchanged.
The resulting library exports all eight added functions. Log: `ad5bb883-libmesh-build.log`, SHA256
`876aff2a4c8c8f87bf13f80706a9d59aae173a2434c4cf54b9f16c49efc355cd`; manifest:
`ad5bb883-libmesh-source-before.json` in this directory. This establishes the producer interface and library build.
The original Polly explicit-binding adapter and its direct caller remain under implementation; guarded access
relations, native execution and categorical replacement retain their outstanding obligations.

Compiler `2206bed` completes the original Polly source-library checkpoint: the shared SCEVAffinator visitor has
an additional entry over the supplied domain, original SCEV/loop bindings and original analysis owners. Source
review corrected anonymous-space construction in the new entry and unchecked initial LLVM Error ownership.
The complete original static target rebuilt successfully; both entry signatures are defined. The canonical patch,
original CMake recipe, source/license identities and exact result contracts are in
`compiler-emit/docs/polly-bindings.md`; its artifact receipt records the final archive SHA256
`c9021f934754ada128a5b9191739c2f920bba4f0852b8d47b4e1f8efbf61bc36`.

Compiler `771f1a9` supplies the reader's explicit patched-Polly, static-LLVM and Mesh library operands. Its recorded
fresh-process build/load read the existing 11,680-byte bitcode successfully, with no aggregate LLVM or separate
ISL dynamic library. That reader did not invoke SCEVAffinator. Its exact pre-final-Polly-rebuild inputs are recorded
in `compiler-emit/artifacts/polly-reader-link-21.1.8/receipt.json`.

Link review found 25 Polly-required ISL symbols omitted from the existing Mesh dynamic library even though all
were present in its original static archive. Mesh `d6da69fd` force-loads that same archive into the same provider.
Root's normal library rebuild exited zero with six source hashes unchanged and all 670 required ISL symbols
exported. Log SHA256: `ee3eeee4592d6733c89e9a83b25a33659469dcad5dda67d5b65fd77654ae0ab7`;
`d6da69fd-libmesh-receipt.json` and `d6da69fd-libmesh-source-before.json` retain the observation.
The next direct caller must retain the original domain/index owners, complete PWACtx invalid sets and assumptions;
an actual bound query, valid-domain discharge, complete logical primitive accesses and their canonical placement
remain unfinished.

Mesh `7cb4bc7e` and native `5f95c6a` put the existing GraphDomain, GraphIndex and GraphAccessMap definitions verbatim
in `rdma/mesh-graph-views.h`. The native binding includes that header and keeps the existing single Python type
registration. Independent review and C++ syntax checks passed. Root built and loaded the ordinary `pages.library`
entry with eight watched source/library hashes unchanged, exit zero. Receipt:
`native-execution/artifacts/5f95c6a-graph-views/receipt.json`; build-log SHA256:
`ba2e6893e128d7aaca62fb207b0ff3e0d6f3b47e72cde045189ab6c3f2b00eb7`.

### Logical primitive accesses precede page placement

The direct-caller audit at compiler `2206bed`, native `5f95c6a` and Metal `fbbf5caa` distinguishes the original
logical index from the printed page-address expression in Torch `MetalKernel.buffer_index`. Root initially treated
the missing typed physical subscript as a prerequisite for affine access analysis. The existing producer/consumer
order supplies the correction: `metal_ir.h`'s paged-root contract keeps effects and footprints in logical storage
bytes. `UpstreamEffects` carries those relations and their storage identities; `Touched` and `TouchRelation` use
the retained placements and `mesh_graph_access_map_placed` to place them per page, copy and access direction.

The immediate numerical producer therefore supplies its expanded primitive accesses in logical storage coordinates,
with the actual source argument, origin, alignment and guard operands. The existing placement transformation
consumes that complete result. The printed page-address expression remains final source output; reproducing its
algorithm or recovering its meaning from text supplies no required prerequisite. An original buffer index and an
already-paged helper offset have different meanings, so relabeling one as the other would still be incorrect.

The producer audit identifies this complete related group at those revisions:

- `MetalStoreLine` retains its original invocation, but `MetalAtomicDefinition` does not connect its compiled
  formals to the complete logical buffer/index/value invocation. `buffer_index` retains its Address Function and
  inputs, while the composed address call and byte-to-element conversion are formatted text.
- A subword view's root-relative origin enters helper expansion before word selection. The existing scalar map
  appends Buffer.origin afterward; increasing its width alone would select the wrong packed word.
- The original Buffer/Address owner carries that origin separately from the U8 Value's zero offset. Constructed
  `upstreamPages.origin` consequently cannot replace it. Upstream U8 binding alignment is one, and argument-table
  encoder alignment says nothing about the accessed resource's alignment. Actual base/alignment operands remain
  necessary; pointer zero is not established by those fields.
- Primitive load and compare-exchange operations inside an atomic store retain that store invocation's frame
  direction and execution copy. Their read/write dependency effects are a separate property.
- Packed-word and multiword expansion need the represented alignment and page-boundary conditions when composed
  with numerical addressing. Whole-page placement alone does not prove those conditions.
- `Relations.lower_accesses` still uses numerical element width and unconditional commuting-update classification
  for `atomic_add`. Complete primitive widths, guards, private expected-pointer effects and returned-value
  dependencies remain unconsumed. The private expected location is not a global tensor-buffer access.

The existing original numerical emitter, `compile_header`, `AtomicEffects` and LLVM analyses supply the numerical
body and its expansion. Completed logical byte relations enter `mesh_graph_access_map_create` and
`metal_ir_upstream_mapped`; the existing placement chain above consumes them. `AddressFunctions`/`Address`,
`FrameBindings` and final frame projection remain the numerical-address producers. This is a source defect
manifest for their immediate handoff, not a new numerical compiler or an alternative placement algorithm.

### Complete original affine queries and the real domain-bound correction

Compiler `955abb8` binds the original Polly entry to the existing analysis, domain and index owners. Each query
retains its input SCEV/block, whole PWACtx, assumptions collection and original LLVM Error. Derived bindings take
the complete earlier query; original Mesh inputs supply both value and invalid-domain operands. Component views
retain their complete owner. Source review corrected lost invalidity on rebinding, subview lifetimes and an added
nullable-block restriction. The reader compiled, linked and loaded with the actual original affinator definitions;
`compiler-emit/artifacts/polly-affine-binding-21.1.8/receipt.json` records that build.

The first public Indexing-to-query invocation exposed an existing Mesh producer defect: `domain_box` negated
INT64_MIN in native C arithmetic before passing the value to ISL. The actual indirect-index domain became
malformed, and querying the original compiler's four-byte extent returned no fixed value. Mesh `1f3e9ffc` moves
that negation and the equivalent access-bound/placement/partition arithmetic into original ISL values. Its adjacent
partition-span comparison uses a sufficient 128-bit intermediate. Independent source review and the ordinary
libmesh build passed, with seventeen watched inputs unchanged. The observed INT64_MIN failure is distinct from
the adjacent source-reviewed cases; no separate execution coverage is asserted for those cases.

The same public invocation then exited zero: Indexing emission produced three calls; original extent and member
offset queries returned four and zero over the actual root `[0,11]` and indirect `[INT64_MIN,INT64_MAX] domain.
The invalid predicate selected the original zero offset, whole-query rebinding retained the extent, and the actual
unbound pointer produced a retained error whose value accessor raised. Component views kept the complete query
alive after initial owners were dropped. All sixty-eight watched source/library hashes stayed unchanged.

Exact command, source manifests, original bitcode and receipt:
`compiler-emit/artifacts/955abb8-1f3e9ffc-affine-indexing/`. Run-log SHA256:
`8099628eb9534723ce5d5bf36b101fc34f648ff131f95b098e04bfcc5f1b9ece`.
The two earlier failed observations retain separate receipts. This establishes the actual constant/error query,
domain and ownership handoff. Complete logical primitive-address relations, valid-domain discharge, native
execution and categorical Mesh replacement remain unfinished; the producer manifest above identifies the next
implementation group.

### Logical invocation and required alignment producer observation

Compiler `89e4ed6` and reader `0d4e1ea` connect each original atomic definition's ordered formals to its logical
Buffer/index/value operands and original/derived LLVM arguments. The native index retains its complete original
invocation, Buffer/Address owners and directional frame. Original Clang `alignof` results accompany the existing
atomic offset, extent, scope and order payloads. The reader exposes the original constants by view; AA ordering
checks continue to use the order fields rather than interpreting the appended alignment fields as orders.

The executor's single packed-float16 `samples.Indexing` invocation through public `declarations.emit` exited zero.
Its three emitted calls retained the actual Store/Buffer/Address/formal identities. Both scope definitions exposed
the original load and compare-exchange leaves with four-byte object extent/alignment; compare-exchange retained
its separate private expected location, four-byte extent/alignment and read/write effects. Scopes were 2 and 3,
and all observed order operands were zero. Borrowed views survived dropping the outer local owners and were
released normally. All sixty-eight watched source/library hashes remained unchanged. Original `llvm-dis` of the
same 21,072-byte bitcode showed all four AIR primitive calls retaining their volatile operand.

Exact command, manifests, original bitcode and receipt:
`compiler-emit/artifacts/89e4ed6-logical-producer/`. Run-log SHA256:
`89370e7c6195e8483cc781675357330e5e0ca88975199ac69a6dcc1d4a1ca008`;
receipt SHA256: `4920a67208aaf224ea118ff391c45b703fda4453512a940db4b0060842271171`.
This observation covers the logical invocation and required-alignment producer, without numerical dispatch.
Connecting these requirements to original bindings/allocation, supplying actual base/page/view-origin congruence,
and completing primitive access relations remain work in the manifest above.

The next direct handoff associates each original primitive location with its actual formal through LLVM's
underlying-object analysis and the retained formal/source correspondence. Its required alignment belongs to that
existing parameter owner. The original `Binding.alignment` → `Storages` → allocation-quantum chain consumes that
requirement; a private compare-exchange temporary belongs to its original LLVM allocation. An alignment
requirement supplies an allocation operand, while the allocator and placement result supply the actual provided
alignment. Root's source audit keeps those two facts distinct.

### Original buffer parameter consumes the complete primitive effects

Compiler `81a0bfd`/`25274be` add direct indices of complete atomic definitions and effects to the existing
numerical parameter. Original LLVM `getUnderlyingObject(pointer, 0)` identifies the object behind each primitive
MemoryLocation; the retained derived formal and source Buffer establish its correspondence to that parameter.
The original implementation is declared in installed LLVM 21.1.8 `llvm/Analysis/ValueTracking.h`, lines 434–446.
The private compare-exchange expected location remains part of its complete effect, separate from buffer formals.

The first public invocation failed because `MetalBufferIndex.parameter` still performed mutation-name resolution
through the expired ambient graph context. Compiler `778a6a2` assigns the actual parameter during original
signature preparation and removes the delayed lookup. The next invocation exposed Python wrapper identity:
the same native LLVM argument appeared as a Value wrapper through underlying-object analysis and an Argument
wrapper through the function iterator. Inspection of the same saved module through its original Clone
correspondence confirmed that LLVM's result was correct. Compiler `b60f8e1` exposes native Value address identity
and uses that comparison at the existing association. These corrections retain the original analyses and owners.

The same public packed-float16 Indexing emission then exited zero. Its actual numerical parameter retained both
complete scope definitions and their four original primitive effects; each forwarding data formal retained those
same corresponding effects. Original native identity confirmed the global objects' formal correspondence and the
private expected locations' separation. The numerical parameter alone kept the original program, source,
21,072-byte bitcode and effect views accessible after the outer local owners were dropped. All sixty-eight
watched source/library hashes remained unchanged.

Exact command, manifests and successful receipt: `compiler-emit/artifacts/b60f8e1-underlying-producer/`.
Run-log SHA256: `a8f6bdbcfdd9a53216f6e48b240b3bc6064b0c9be62e947b6c35b42df78e3052`;
receipt SHA256: `8e3a6e6ed281884a3c0525f17ca1c7ccc8b913df928c82fdec4fba3bd0831628`.
The failed `25274bea-underlying-producer` and `778a6a2-underlying-producer` observations remain separate, with
their source manifests and the latter's saved-module identity diagnosis. This completes the observed
primitive-effect-to-buffer-parameter handoff. Required alignment is available through those complete effects;
the allocation operand handoff, actual provided base/page/view-origin facts, complete primitive access relations
and native execution remain outstanding.

### Required alignment enters the original allocation operand chain

Compiler `09f9aa1` exposes the actual LLVM ConstantInt alignment views and original `getZExtValue` method.
Native `be7dc17` consumes each original numerical parameter's scalar width and retained primitive object
requirements at registration. Metal `4ef4bb5a` appends the resulting uint64 alignment to the existing upstream
argument record and takes its maximum with the original scalar alignment in `ConstructKernels`' Binding.
Mesh `38f04a15` matches the C record in the existing ctypes definition and constructor. The complete original
call/parameter/effect owners remain retained. The separate private expected location contributes to its own
LLVM allocation, rather than the global buffer argument.

The C/Python record has size 24 and alignment at offset 16. The existing archive's whole-record `Vec(u.arguments)`
serialization includes the field; its format marker advances to version 48. The later original `Storages`
transformation and allocation algorithms are unchanged. Source review follows Binding.alignment into each
storage root's maximum, then the original local-run allocation quantum. Required alignment is therefore an
explicit planning operand. Actual base/page/view-origin congruence is still a separate result to supply.

The ordinary original reader build exited zero with nine watched inputs unchanged. Original Metal library and
native binding build/load exited zero with ninety-nine inputs unchanged. Build receipt:
`compiler-emit/artifacts/4ef4bb5a-alignment-build/receipt.json`, SHA256
`090cffc326057651a9f98c73b373f972907a278755e3fdfddc9392ed40291ca6`.
The single public packed-float16 Indexing emission then exited zero with all 104 watched inputs/libraries
unchanged. It exposed the original two-byte numerical scalar and four-byte primitive object requirement through
the typed constant API, retained the complete owners, and registered/constructed/configured three original calls.
Receipt: `compiler-emit/artifacts/4ef4bb5a-be7dc17-alignment-producer/receipt.json`, SHA256
`6e0d512080086932d6d79a60e5a80b4d444d806ad1b498190e77646afed07c05`;
run-log SHA256: `941a66e743df9da59c7d2c05f7e2fe828a92e6f366d16b6935b89d3df0b3873e`.
This invocation ends before `metal_ir_lower` and `Storages`; later allocation and numerical dispatch were not
executed. The propagation beyond construction is source evidence, distinct from that observation.

The adjacent source audit identifies where the remaining provided facts belong. Original local planning aligns
allocation offsets through `mesh_plan_local` and GreedyBySize; Arena retains the actual MTLBuffer base. Network
planning retains block/page quantum, mapped-region geometry and page rows; `metal_ir_value_placed` checks relative
page-offset divisibility. External/fixed bindings retain their original buffer and byte offset. The actual GPU
base is available from those retained MTLBuffer owners, and the logical view origin remains `Buffer.origin`.
Those facts currently do not reach the analyzed formal. An aligned offset or enclosing page alone does not
establish the alignment of a different base or view.

Original ScalarEvolution supplies same-base pointer subtraction through `getMinusSCEV`/`removePointerBase`;
original known-bits analysis reads established IR facts, and the existing Polly binding retains residual
expressions, invalid domains and assumptions. A packed helper's remaining base-residue expression needs its
actual congruence operand. This audit identifies the missing provenance without changing the existing
logical-byte analysis → placement order or replacing an unknown pointer with zero. Complete primitive relations,
native allocation/execution and categorical replacement remain unfinished.

### Consumers use the produced argument correspondence

Compiler `e12e9cb` assigns each original numerical parameter's ordinal during signature construction and retains
its actual native Buffer there. Each call references that same complete signature list. Compiler `2b10922` and
native `cd479d8` consume those operands directly: the former buffer-name normalization, argument-list search,
source-to-Buffer lookup and parameter-name reselection in Relations/Register are removed. Original removal,
mutation and dynamic-scalar ordering were source-reviewed; error, frame, workgroup and builtin operands remain
distinct from the ordinary `call.sources` prefix.

The native binding rebuilt/loaded successfully. The existing packed-float16 public emission exited zero with all
104 watched inputs unchanged. Across three calls, every emitted relation's ordinal identified the same original
parameter and Buffer/source owners. Required alignment, complete atomic definitions/effects and retained
lifetimes remained observable. Receipt: `compiler-emit/artifacts/2b10922-cd479d8-ordinal-producer/receipt.json`,
SHA256 `f23d8909f0cf125912444ae849c83f084941eee16ad4f1053aa70b1db0561a31`;
run-log SHA256 `4bc032ef7688df97265d8ef29fd3966488a442eaa7e0c430da473e87a2c83f84`.

The same boundary review exposed an omitted operand: native Register does not yet supply the original optional
error parameter, which lies outside `call.sources`. `metal_ir_upstream_error_workspace` already creates the actual
ErrorMessages-typed zeroed state, and original Upstream/error-reporting code owns its effects and completion
handling. The next correction connects that produced workspace to the actual parameter ordinal and preserves it
on the Program owner. The observed Indexing call does not establish this omitted argument or runtime handling.

## Sources used in their respective roles

- `schedule_emit/driver.py::compile_module`, `ir.py::Scheduled`, `lower_inductor.py::lower` and
  `runtime.py::Program`: executable producer/lowerer/consumer reference.
- PyTorch `GraphLowering.codegen`, `PythonWrapperCodegen.run_wrapper_ir_passes`, `AotCodeCompiler.compile` and
  `CompiledAOTI`: original numerical compiler and compilation-result boundaries.
- Jones, Gomard and Sestoft, *Partial Evaluation and Automatic Program Generation*, §4.2.2: specialization of
  known operands into a residual program, as cited by composition item 4.
- Bhattacharyya, Sriram and Lee, *Optimizing Synchronization in Multiprocessor DSP Systems*, §2: compiled processor
  assignment and ordering with execution beginning when operands are available, as cited by composition item 4.
- [AOT A5–A10](../../design/aot-compilation-change-request.md),
  [plan-to-bridge R1–R8](../../design/plan-to-bridge-change-request.md), and composition items 2/6/7: retained
  allocation, dependence, schedule, transfer, archive and owner results consumed at the corresponding boundaries.
