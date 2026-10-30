# Prepared ANE pages, state and completion

Recorded 2026-10-30T17:10:40Z. This review covers the specialized ANE part of
step 9. It does not change the data model or add an execution path.

## State is an input and a result

`engines.page_state_program` consumes the effects already present in an
`ExportedProgram.graph_signature`. A changed buffer becomes an ordinary named
input. Its updated value becomes an ordinary named result. A changed user
input keeps its one existing argument. `configuration.state_pairs` maps each
exported buffer target or user-input name to its `input` and `output` feature
names. The same pairs reach the native model contract.

This changes the signature and its names, not the numerical operations. Lifted
state is removed from captured constants. Other constants keep their declared
role and position. When a state result is also a user result, the configured
state output names that one SSA value and one page operand. Callers bind that
canonical feature; the exporter does not manufacture a second result copy.
The caller's graph supplies the next use of those pages and its dependencies.

The converter receives the graph's explicit input and output scalar types.
Its configured arithmetic precision remains separate from those operand types.
Without this signature change, the installed Core ML converter creates private
`StateType` values from mutable buffers, including when given `states=[]`, and
casts non-fp16 state to fp16. Its `frontend/torch/exir_utils.py` extracts the
mutation effects; `frontend/torch/converter.py` constructs those states.

This pass requires the exported source graph. A previously compiled model with
`LiveStateList`, or a public Core ML model with state descriptions, cannot be
functionalized from its operand layout report alone. Such an artifact still
needs its original graph transformed before compilation. This is a source
contract boundary, not a claim that ANE hardware cannot calculate stateful work.

## Output pages are already bound

The direct adapter in `rdma/mesh-coreml.m` prepares IOSurfaces over retained
bridge-page roots, with explicit offsets. It maps all named inputs and outputs
to the compiled procedure's symbol indices once. Physical shape, strides,
scalar type, footprint and alignment reach the shared placement solver before
allocation. Public Core ML's advisory `outputBackings` are a different declared
backend contract and do not establish this fixed-output guarantee.

## Completion remains unfinished

`mesh_coreml_run` still calls `evaluateWithModel:options:request:qos:error:`
synchronously. `MetalHostDescription` currently treats return from its operation
as completion and publishes the output event then. Returning after an enqueue
without changing this shared contract would publish unfinished pages.

Metadata and disassembly inspection on macOS 26.3.2, build 25D2150, established:

| Entry | Observed responsibility |
|---|---|
| `_ANERequest setCompletionHandler:` (`v24@0:8@?16`) | A prepared request can retain a completion block. The driver's completion adapter calls it with a success byte and an error object. |
| `+[_ANEClient initialize]` | Creates eight submission buckets, each with a semaphore count of two, in this framework build. These are observed vendor policy, not values to embed in our algorithms. |
| `doEvaluateDirectWithModel:options:request:qos:error:` | Waits on the selected submission bucket before `processRequest`; signals it when `processRequest` returns. This is not a tensor dependency. |
| `_ANEProgramForEvaluation initWithController:intermediateBufferHandle:queueDepth:` | Creates its in-flight semaphore from the supplied model queue depth. |
| `processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:` | Its asynchronous branch takes an in-flight slot. Its driver callback invokes the request completion block, then returns that slot. This represents model request capacity. |
| `evaluateRealTimeWithModel:options:request:error:` | Calls the same direct submission method. The real-time selector does not remove either semaphore. |

The request also has shared-event operands. Supplying them selects asynchronous
processing but retains the same capacity and submission gates. None of these
observations establishes a zero-wait submission entry. The driver's lower
device-vtable entry has not been adopted without a supported ABI contract.

The necessary shared-interface change is prepared submission with an explicit
completion operand. The ANE request binds that operand during preparation and
publishes readiness only on actual completion. CPU work completes the same
operand when it returns. The affected consumers are `metal-recording.h/.m`,
`torch_mesh/_stream.mm`, `engines.Prediction` and its declaration rebuild,
`matrix_operations.swift` host records, and `neural_engine.swift`. This is one
scheduler contract, not an extra worker around a blocking provider call.

Model request slots can become declared scheduling resources using the loaded
model's queue depth. That can avoid exhaustion by our own graph. It cannot
remove the vendor semaphore instructions or guarantee that a process-global
submission bucket is uncontended. Those limits remain explicit.

## Preparation evidence

The signature pass was inspected with persistent and nonpersistent buffer
effects, changed user inputs, shared state/user results, and retained constants.
Core ML conversion produced ordinary tensor operations without `read_state` or
`coreml_update_state`. Float32 and int32 feature types were retained, including
when arithmetic precision was explicitly configured as float16.

The native inspection process only loaded framework metadata and disassembled
methods. It did not construct a model, device, request or queue, invoke a
prediction, or evaluate numerical work. The converter inspection likewise did
not invoke the graph. Submission behavior has not been measured or changed.
