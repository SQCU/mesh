# Production integration assignment, 2026-11-13

Style: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights).

Read the [workgroup assignment](compiler-workgroup-2026-11-13.md), including the operator's instruction, source
roles and coordination. You are its eighth Claude implementation worker, supervised by root Codex. Implement
the connections below. Do not delegate further or write a self-acceptance review; report concrete changes and
execution evidence for Codex to judge. Do not commit concurrently with other workers; report named paths ready
for the supervisor to commit. Source edits are authorized now.

## First connection: compiled operations to the signature and caller

- **Applies to:** compilation-composition-change-request.md item 6 and the "Executable artifact" handoff;
  synchronization-signature-change-request.md item 1; implementation-order-2026-11-09.md item 10.
- **Sources and roles:** the signature request's cited task/operation/dependence representation supplies the
  observation problem; `~/schedule-emit/schedule_emit/driver.py`, `ir.py` and `runtime.py` supply the explicit
  composition example. Read those actual sources and the clauses, not a historical description of their code.
- **Composes:** compilation and target lowering -> the actual program's task, dependence and operation results
  with value/channel correspondence -> signature construction and the existing production callers.
- **Property preserved:** the observed synchronization describes the completed executable artifact. Compilation
  results survive the handoff; consumers do not silently synthesize a different program from the old FX graph.
- **Replaces:** `signature.py`'s fallback to `torch_run` for a missing target record, repeated discovery of channel
  ends and task identities when those are already available upstream, and the corresponding implicit interface.
  Remove by completing producers and consumers, not by adding a refusal for missing data.
- **Write:** own `compiler/mesh_compiler/ir.py`, `signature.py` and existing `examples/` callers. Coordinate additions
  required in `driver.py`, scheduling, target lowerers and archive with their workers through root; report exact
  proposed field/function interfaces promptly. Root will route them. Ordinary data structures suffice; do not
  build a schema framework. Implement an actual complete connection, not a field awaiting a future consumer.

Native execution and archive workers are changing the runtime and lowering orchestration. Read their source as
it changes and provide their actual needs in the common representation. The scheduling worker is making machine
configuration an explicit operand; connect acquisition in the existing caller if that change needs it. Don't
reintroduce acquisition inside the transform. Preserve the old user-visible capabilities while completing the
lowering; a smaller supported domain or new fallback is not a completed connection.

Use the existing `examples/engines.py` and other applicable clients, their existing numerical measurements,
and the repository measurement instructions. Read the invocation before running it. Do not start/stop bridges,
remote jobs or user services as a prerequisite without telling the supervisor what concrete invocation needs
them. Compile/import evidence must be reported as such, never as full execution coverage. Do not create a new
test suite or claim an unchanged graph fingerprint proves anything.

## Next connection: LiteRT program to the production consumer

This remains part of your workgroup scope, coordinated with the LiteRT worker. Inspect the model/program
publication interface and the E2B calling path in `tools/prepare_e2b.py` and `tools/e2b_replay.m`. Identify exactly
which current transformations the compiler result must displace. Implement the agreed consumer when its concrete
producer interface is established; do not independently implement a second provider or translate generated
shader text. The source and algorithm roles are the composition request's "Complete replacement, all in one
commit workgroup" and its operand and displaced-body tables. Send the concrete interface needs to root early
while completing the first connection.

Record changed paths, the producer/consumer connection, old body removed, commands/results, and specific
remaining work in your final reply. Root will preserve this with session evidence; do not expand standing
requirements with implementation-state prose.
