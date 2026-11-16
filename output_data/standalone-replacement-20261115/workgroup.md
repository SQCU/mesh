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

The separate [callback-restoration receipt](callback-restoration-20261115.md) records Mesh `ef651076`, Metal
`ac2bd2ca`, restored dependency trees, exact source comparisons and the remaining central markers. The standalone
native module separately executed repeated CPU and MPS alias/state examples at `native-execution` commit `cd09ad6`.
Those native artifact-boundary examples do not yet establish scheduled transport or external page-plan consumption.

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
