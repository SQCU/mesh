# Native132: region, order, lifetime, replay, selection and prepared-unit results: current ownership

Claude Opus 5.5, 2026-11-15. The prepared-unit source is committed in Metal `669535a7`; the native ownership metadata is in `b850050b` and its canonical allocator consumer in Mesh `15e00e26`. Frontend `1d132af4` retains the same fields through Python. Native138 full canonical build and ABI comparison passed, recorded by Metal55e2dcce. Earlier strict source checks and these checkpoints are not runtime evidence. The original stream resumed its unfinished source consumers after the build.

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

## Prepared unit (checkpointed source)

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

- **XNNPACK operator-context persistence across calls: connected in source** (checkpointed above; earlier strict syntax-only checks passed for `mesh-collective.c`, `mesh-dataflow.c`, `mesh-call.c` and the Metal sources).
  - `metal_ir_value_info` carries `owned`, its storage root's `Value::owned` (struct now 128 bytes).
  - `MetalStandIns` sets each stand-in's `mesh_value.owned` from it.
  - `mesh_value` gains `owned`, taking half of the former 8-byte `padding1`; the record stays 64 bytes and is saved as is.
  - `plan_location` gives an owned value's ring the span of every execution, as a lagged one's, with its EXECUTION kind and ordinary copies. `plan_lifetime` gives it no single call occurrence, so the cross-call allocator cannot reuse its storage between executions.
  - **For frontend104's thin projection:** `rdma/mesh.py` `Value` replaces `('padding1', Q)` with `('owned', U), ('padding1', U)`; `compiler/mesh_compiler/metal.py` `ValueInfo` appends `('owned', U), ('padding', U)`; `native.py` storages passes `info.owned` into each value's `owned`.
- **Conditional GPU publication: host-step routing withdrawn** (Codex rejected it: it replaced in-body transport with host SENT/PUBLISH steps). Removed from the working tree, never committed.
  - **Device design, to implement.** A GPU writer in a switch branch keeps its in-body release (the canonical `Releasing` protocol). The switch's device selector, after its selection, makes the releases owed by the writers of every unselected branch, through that same protocol implementation. Each such release is conditioned on `selected != arm`, the same literal-store mechanism as `SwitchBodies`.
  - **Why it is correct.** An unselected writer never writes, so releasing for it before the selected branch runs completes no scope early. The selected branch's writers release in their own bodies, and the scope's last release publishes once. There is no host wait, no added step, and the writer count is unchanged (`Completions`).
  - **Retained correspondence it needs:** each switch's held GPU writers per branch and their release records (value, transfer, owner; `Kernel::sent`, `Completions`' writers), from `Obligate` and the region search (`switchHeld`).
  - **Crossing.** The selector's module must call 131's release protocol (`Releasing`/`UpstreamProtocol`) with those records bound. The exact composition point needs pairing with 131.

- **Original WhileLoop pairing with 68/66** (needs actual source fields through the supervisor).
  - The original `CppWrapperCpu.codegen_while_loop` (~2906) rebinds the carried tensor handles to the body outputs after each pass (`aoti_torch_assign_tensors_out`) and evaluates the condition through its item call. It writes no carried initialization or copy-back.
  - Native `metal_ir_repeat` takes `carried` as one root that is the same across passes. For a body output that is a different buffer from its carried input, that is a mismatch: the original rebinds, native assumes in-place.
  - Fields needed from the retained original emitter:
    - each carried input → body output pair, as handles, with its buffer identity;
    - whether the pair aliases (the same storage) or rebinds;
    - the condition's source and its item;
    - the loop's operation interval and nested children in the emitted order;
    - the child's retained emitted result, not a replayed `graph.codegen`.
  - The aliased case maps directly to `carried`. The rebinding case needs the native repeat consumer to take the pairs, the output root becoming the next pass's input binding. That is a correspondence change in my region consumer, not a copy or a new interpreter, and is designed only once the fields exist.

- **Runtime unobserved:** nothing above has been run.

## Others

- **131:** Transport/Upstream binding and effect methods, environment/compiler, binding groups, sizes and physical, `NativeBuild` compile-unit scheduling, other Lower stages; COPY/FILL in a branch as kernels; the diagnostic port rescan in `MetalOrder`. 131 saves `Value::owned` under v45.
- **133:** the original producer, its call sites and its unit declarations.
- **135:** `metal_neural.mm`, NeuralIO/NeuralModel, the Kernel neural fields, the shared NeuralModel archive owner (`b4e1088c`, v45).
- **66/68:** the original control (`ExternCall`/`Inlined`/`WhileLoop`/`Switch`) and lifetime operands. The native inputs that consume them are `metal_ir_value` role, `metal_ir_value_view`, the effects/bindings, `metal_ir_repeat` carried and condition, and transport ports; their results are `metal_ir_storages`, `metal_ir_last_uses`, `metal_ir_order_bounds` and `metal_ir_parts`.

## Current resumed completion responsibility

132 owns the exact conditional Completions/publication block together with the existing original region/result/lifetime consumer. The first post-build draft inserted host Steps for GPU branch sends and cleared the GPU send list; Codex rejected that substitution before acceptance. Finish the original device protocol's publication through retained selected-region result/completion, without a host wait or a one-shot loop. Preserve original wrapper handle rebinding and carried/zero-iteration semantics with66; do not invent unconditional init/copyback. The matched138 evidence is prior source, not evidence for this in-progress continuation.
