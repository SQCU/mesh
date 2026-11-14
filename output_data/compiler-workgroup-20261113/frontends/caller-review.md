# Completed-entry caller and prepared storage checkpoint

2026-11-13. Codex frontend supervision of the same Claude session
`9740398d-f513-4c78-8578-f80b79fa2225`, requested and reported `claude-opus-5-5`, harness 2.1.321.
Orders and raw turns: metal-microbench `output_data/compiler-workgroup-20261113/frontends/torch/`,
`native-caller-1.md`, `native-preparation-2.md`, `native-preparation-3.md`, sessions 4 through 7.
Session 5 made no edits because the supervisor initially wrote its order under the wrong working directory;
the absolute path was corrected and the same session resumed. No alternate model or worker tree was used.

## Local contract

[Composition request, requirement 7](../../../design/compilation-composition-change-request.md) requires
compiled composition to remove channel/task interpretation while retaining actual input, storage and effect
properties. The schedule producer (`fddeb611`) supplies representative-keyed modules and exact source task/resource
correspondence. The driver supplies the corresponding completed `Scheduled.functions` entries and projected
inputs, outputs and device/machine identities.

- `runtime.Arguments` owns input/state/token binding once. `Program` consumes one prepared entry, invokes its
  completed callable once, and reconstructs the original output tree and buffer mutations. Thread queues,
  function-or-FX execution selection and per-invocation bridge callbacks are deleted.
- `examples/engines.py` and `examples/tp.py` use `compiled(compile_module(...))` and `Program`. Existing `tp.taken`
  calibration uses the same Arguments owner and the already-composed source module. It remains calibration;
  it is not an execution fallback. `Timed` and `shares.stages` consume original `node.meta['processor']` so cost
  attribution survives grouping.
- `bridge.prepare` performs entry selection and plan binding once. Without a plan it returns the explicit or
  original output entry and Target without attaching a bridge. With a plan it binds the actual local machine's
  completed entry (or explicit key) to existing `Published` storage. The one-entry destructure expresses the
  current producer postcondition; root reviewed it as ordinary binding, not an added domain restriction.
- `Published` retains the original Plan, function/held records, loaded plan ID, actual context address, starting
  invocation and solved copies. The old Carried view construction now produces both logical `views[transfer]`
  and distinct write `distinct[transfer]` from the same mapped buffers. Both preserve offset/element/stride
  metadata. Archive-owned `Target.bind(published)` consumes those prepared views and must retain Published.
  Carried/Empty callback objects and Published start/end/address execution methods are deleted, including the
  blanket end-of-call send wait.

Review corrected an initial overloaded views field: sender-distinct and receiver-logical shape are separate
properties, not alternate meanings for one field. An earlier supervisor objection to the one-time singleton
selection was withdrawn after the producer postcondition was clarified; no extra selector or target filter was
added.

## Evidence and remaining consumers

Supervisor `py_compile` and `git diff --check` pass for runtime.py, bridge.py, shares.py and both example callers.
The worker's Bash checks were denied by its harness; the supervisor performed the authorized checks directly.
Executor archive evidence already exercised the no-effect local CPU entry, archive reload, three sequential
RunningMean buffer mutations and duplicate/aliased result leaves. See `../executor/archive.status.md`; that run
used its private native loader and preceded final signature/native-effects changes. It is not current shared
runtime or performance evidence.

This checkpoint does not complete plan execution merely by creating views. At source review, Target.bind and the
actual direct native effect consumer are still executor-owned work in progress. New native ABI imports await its
coordinated library build. The old signature argument/finished scaffolding is separately assigned for removal.
External-effect calibration remains an actual measurement consumer to connect; queues were not retained for it.
Zero-sized channels omitted from the transfer plan need their ordinary empty-value lowering in the target owner.
The existing global publication/mesh_retire lifecycle still requires canonical review before claiming independent
simultaneously prepared plan programs. No new bridge service, runtime evaluator or publication lifecycle was added.
