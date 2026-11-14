# Assigned typed construction checkpoint

2026-11-13. Same supervised Opus 5.5 session `41f48493-701e-43d0-b855-aa88142fabd5`, raw turns 8–9 under
metal-microbench's compiler-workgroup/frontends/litert directory. This is a working source checkpoint, not the
completed machine-target replacement.

The old `lower_metal` emitter/probing/independent-ordering body is displaced by construction through the existing
`rdma/mesh_compile.py` numerical owner. `_instruction` now supplies the shared result/mutation/alias boundary;
`_Builder` emits the same typed C Program. Explicit source roles, storage roots/views, output metadata and
original task→operation correspondence reach the new native consumer. Generator state is an explicit operand;
the old rdma caller supplies its seed and retains the returned state. `_instantiate` consequently returns three
values `(operations, result, generator)` and its production caller was updated.

The scheduler now retains actual copied cross-target endpoints and its existing order. The already implemented
fanout construction has one owner consumed by both the Inductor reference emitter and the typed emitter. Source
version, native action identity and copy/result correspondence were not independently re-derived by the new target.

Codex review corrected a helper-name collision, mutation mistaken for a no-op alias, hidden global RNG state,
per-placeholder alias loss, complex output metadata loss, per-SEND duplicate copies, and mixed real/fake storage
identity namespaces. The latter was observed directly: one captured parameter/user storage initially became two
constructed roots; after correction both bind root 0 with STATE authority.

Evidence at this checkpoint:

- All four changed Python files parse; scoped diff checks pass.
- Construction-only calls against canonical `libmetal_ir` preserve complex add output metadata and two aliased
  input views (one root, offsets 0 and 2). No target code was executed for this check.
- The state/user alias construction check passes after correction.
- Existing CPU FFN invocation through the reference target still returns the same error 0.0003511235991 and
  signature after factoring fanout; log `typed-fanout-cpu-reference.log`.

The whole-machine and late-plan handoff remains open. Root corrected stage order to source declarations/causal
assembly → actual typed lowering/Actors/Storages → one canonical plan → specialize/finalize/native artifact.
The current per-group construction and solved-effects input are being revised accordingly; adding target table
entries alone would omit sibling local CPU/MPS/ANE work. Runtime binding must relocate a complete artifact and
bind actual user handles, not compile on first bind or introduce blanket staging copies. The old CoreML recipe
and CPU reference path remain until their complete selected-target behavior is supplied by that connected result.

Constant definition identity is a separate pending correction: explicit declared roots now accept identity 0,
but `Builder.root` still assigns operation provenance to implicitly constructed constant roots. That must not
authorize cross-Program immutable-definition reuse. The canonical C append policy remains unchanged.
