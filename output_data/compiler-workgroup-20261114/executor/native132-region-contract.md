# Native132: region, order, lifetime, replay, selection and prepared-unit results: current ownership

Claude Opus 5.5, 2026-11-15. Metal HEAD `3330ed9b`. Committed: `38b5ea61` through `f2ae6bb2` (archive v45). The
prepared-unit hunks are staged for Codex's commit, unbuilt. Strict syntax-only checks (`-Wall -Wextra -Werror`) of the
staged snapshot pass for `metal_ir.mm`, `metal_lower.mm`, `metal_transport.mm`, `metal_capture.m` and
`metal_program.m`; `metal_archive.mm` cannot be compiled here (the Dawn-native generated headers its includes need are
absent). Nothing has been built or run by me.

## Mine

| Owner (stage) | Retained result | Consumers |
|---|---|---|
| `metal_ir_repeat`/`metal_ir_switch`/`Adopt` | Parents, kinds and arms; the mixed `postorder`; `Value::selects` | `RegionForest`, `RegionSpans`, append, `Builder::Write`, archive |
| `RegionForest` (construct) | `regionSearch`, rows in the source postorder; `repeatEnclosing`/`repeatOutermost` | `ConstructKernels`, `RegionAbove`/`RegionWithin`, `SwitchArguments`, transport |
| `ConstructKernels` test insertion | Tests in the search preorder; each test's switch and branch from its parent | `Order`, gating |
| `Order` (`Artifact`) | `order`, `levels`, `orderCuts`, `orderCutKinds` | `RegionSpans`, `PrepareOrder`, `-plan:` |
| `PrepareOrder` | `orderParts`, `partRepeats`, cuts, `orderEarliest`/`orderLatest` | `Storages`, `ReplayOrder`, capture, `MetalRepeatOf`, `metal_ir_place` |
| `Storages` | `storages`, `lastUse`; a root with `Value::owned` (the provider's explicit operator-context fact, `metal_ir_value_owned`) gets storage `[0, last]` and last use `last`, its copies the planner's | capture, planner |
| `ReplayOrder` | `replayOrder`, `orderTimes` | `-plan:` |
| `metal_ir_replay_plan` | steps, span, copies, records, `stores` | `SwitchBodies`, `metal_ir_place`, native |
| `SwitchArguments` (Lower) | `switchHeld[s]` (read once from the search's preorder subtree; copied, saved); `switchArguments[s]`; an all-host program's selection values as word DATA (own root) | selector-binding block, `SwitchBodies`, `metal_ir_place` |
| Lower selector-binding block | `Kernel::Selection::arguments` (12-byte write views), `resets` (4-byte write views) | `SwitchBodies`, relation, `metal_ir_place` |
| `SwitchBodies` | Run lengths; the selection (range-table words, or a DATA value's word 0 with UINT32_MAX through `resets`); `{0,0,0}` through `arguments` | emitted selectors |
| `Artifact` selection read | Each host step of a branch reads its switch's selection value (`Whole()`) | relation, `MetalRepeatReads` |
| capture `-plan:`, native gating | Range table; arm edges kept only where the projected commands' switch or branch differs; `NATIVE_SELECT_BEGIN`/`END` around host steps and regions | generated program |
| archive | v45, sole version owner | load |

## Prepared unit (staged)

- **Producer** (133, `b3273f38`): each unit's prepare and release declarations and its entries' trailing pool parameter at slot `LITERT_MESH_UNIT_PREPARED`. The names reach the upstream records through `metal_ir_upstream_unit`, handed over by 131.
- **Value.** One CONSTANT U64 per unit (`Upstream::prepared`, shared by the unit's upstreams, remapped by append), holding 8 zero placeholder bytes and one `METAL_IR_RELOCATE_PREPARED` relocation. No live address is held or archived.
- **Bindings.**
  - The first pass binds the value at its next slot (`metal_ir_upstream_emitted`).
  - A later pass's parameter keeps the marker. `UpstreamPasses` reuses the first pass's binding where the later kernel's kept slots hold it (emitted before Lower's words), else binds at the later kernel's own next slot.
- **Owner.** The unit's one task library (`NativeBinary` `prepare`/`release`/`prepared`/`destroy`/`readied`).
  - `UpstreamTask` appends typed wrappers to the unit source, exported under the unit's identity (`<NativeSymbol>_prepare`/`_release`, ABI `void *(void)`/`void (void *)`), calling the producer's own declarations.
  - `Prepare` runs once (after `Present`, and at a loaded program's `TargetBind`) and records `readied`; a NULL object is valid.
  - `Close` releases once wherever preparation completed, before `dlclose`.
- **Borrowers.**
  - Each placed copy of the value receives the object in `MetalInitialContents`, through `metal_ir_value_prepared` (the program's own upstream→operation→kernel→library association).
  - `NativeCompiled::borrowed` retains the task libraries the whole-program object links; `metal_ir_native_free` closes its own library first.
  - Concurrent use of the one pool is serialized by `pthreadpool`'s own `execution_mutex` (`pthreadpool/src/pthreads.c` 157–174, 676–765).
- **Archive.** The upstream `unit`/`prepare`/`release`/`prepared`; each library's `prepare`/`release`.

## Open in this unit

- **XNNPACK operator-context persistence across calls: connected in source** (uncommitted; strict syntax-only checks pass for `mesh-collective.c`, `mesh-dataflow.c`, `mesh-call.c` and the Metal sources).
  - `metal_ir_value_info` carries `owned`, its storage root's `Value::owned` (struct now 128 bytes).
  - `MetalStandIns` sets each stand-in's `mesh_value.owned` from it.
  - `mesh_value` gains `owned`, taking half of the former 8-byte `padding1`; the record stays 64 bytes and is saved as is.
  - `plan_location` gives an owned value's ring the span of every execution, as a lagged one's, with its EXECUTION kind and ordinary copies. `plan_lifetime` gives it no single call occurrence, so the cross-call allocator cannot reuse its storage between executions.
  - **For frontend104's thin projection:** `rdma/mesh.py` `Value` replaces `('padding1', Q)` with `('owned', U), ('padding1', U)`; `compiler/mesh_compiler/metal.py` `ValueInfo` appends `('owned', U), ('padding', U)`; `native.py` storages passes `info.owned` into each value's `owned`.
- **Runtime unobserved:** nothing above has been run.

## Others

- **131:** Transport/Upstream binding and effect methods, environment/compiler, binding groups, sizes and physical, `NativeBuild` compile-unit scheduling, other Lower stages; COPY/FILL in a branch as kernels; publishing a branch's value once after the switch; the diagnostic port rescan in `MetalOrder`. 131 saves `Value::owned` under v45.
- **133:** the original producer, its call sites and its unit declarations.
- **135:** `metal_neural.mm`, NeuralIO/NeuralModel, the Kernel neural fields, the shared NeuralModel archive owner (`b4e1088c`, v45).
- **66/68:** the original control (`ExternCall`/`Inlined`/`WhileLoop`/`Switch`) and lifetime operands. The native inputs that consume them are `metal_ir_value` role, `metal_ir_value_view`, the effects/bindings, `metal_ir_repeat` carried and condition, and transport ports; their results are `metal_ir_storages`, `metal_ir_last_uses`, `metal_ir_order_bounds` and `metal_ir_parts`.
