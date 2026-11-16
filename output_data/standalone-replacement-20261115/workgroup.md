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
