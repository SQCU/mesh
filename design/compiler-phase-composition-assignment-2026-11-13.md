# Retained program to phase composition

Style: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights).

This replaces the integration worker's earlier broad assignment. Read the operator's correction in
[the workgroup](compiler-workgroup-2026-11-13.md#correction-to-the-first-assignments). Codex chose the interface
below. Implement it; do not repair or copy inherited numerical lowering bodies. No new agent tree.

## The connection

- **Applies to:** compilation-composition-change-request.md item 6 and "Complete replacement, all in one commit
  workgroup": retain numerical bodies and their typed operands through their actual consumers; no source-text
  interchange or per-shader borrowing.
- **Reference role:** `~/schedule-emit/schedule_emit/export.py` returns the captured value once and `driver.py`
  passes it to its consumers. Tint's `core::ir::CloneContext`, owned by `metal_clone`, supplies typed body
  composition. These are the implementation sources; old E2B code identifies calling paths to replace.
- **Producer:** `LiteRtMeshExport`, with actual declarations in metal-microbench `provider/litert_mesh.h`. It returns
  opaque `LiteRtMeshExported`; `LiteRtMeshExportedProgram`, `LiteRtMeshExportedInterface`, and
  `LiteRtMeshExportedSignature` expose the retained result. `LiteRtMeshExportedDestroy` releases it.
- **Composition:** the numerical worker is implementing
  `int metal_ir_append(dst, const src, uint32_t *values, size_t count, char *error, size_t room)`.
  The in/out map covers all source ValueIDs. Each input is a supplied destination backing or UINT32_MAX; each
  output is the corresponding destination logical ValueID. The source's scalar, shape, layout, offset,
  quantization, alias relations and body instructions remain meaningful. Destination retains the source program
  once for borrowed bytes; source remains unchanged and reusable. Read the actual header as it lands. Do not
  implement a second clone/linker or repeat provider traversal in the consumer.
- **Consumer:** the existing E2B phase construction in `tools/e2b_replay.m`.

Export each model/signature once. Read those retained interface values to bind the phase's common storage
identities before host-fill operations are constructed. Then append retained bodies at their original stage
positions, interleaved with the actual host/typed operations. Appending all numerical bodies before host writes
changes the program and is not the assignment. Repeated KV input/output entries must keep their alias relation.
Use the typed values to keep the shared shape/layout/role facts; do not recover them from source or invent a
different value domain in the consumer.

Delete `provide(YES)`/`provide(NO)` as repeated signature/model-binding traversals in the active consumer. The
old Publish path only translated numerical bodies once; do not falsely report two numerical translations removed.
This unit completes retention and composition, not replacement of every provider numerical implementation.
No edits to `Graph::Lower`, `Builtin::Lower`, Python `Lowering`/`Native`, or target arithmetic are assigned.

## Editing and evidence

Own only `tools/e2b_replay.m` and necessary calls to its existing preparation inputs. Provider and numerical
workers own the API/clone implementation. Report exact missing operands in
`output_data/compiler-workgroup-20261113/integration-status.md`; read `integration-inbox.md` before edit batches.
Treat stale notes as observations; the actual header is the interface. No new schema framework, compatibility
path, emitter callback, runtime opcode decoder, or parallel evaluator. Preserve the supported phase sequence.

Compile the existing E2B target with its canonical provider build and use its existing client for numerical
evidence. Coordinate shared build outputs with Codex. Ordinary local checks need no extra permission. Report
source/compilation evidence distinctly from an actual run; a successful object build does not establish E2B
behavior or performance. Codex reviews the connection and commits named paths.

After this connection, the supervisor will assign the next specific replacement. Do not choose an additional
scope or renovate adjacent bodies because they are nearby.
