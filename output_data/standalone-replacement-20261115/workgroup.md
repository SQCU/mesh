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
