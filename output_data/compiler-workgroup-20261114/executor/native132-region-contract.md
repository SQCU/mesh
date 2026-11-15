# Native132: region, order, lifetime, replay, selection and prepared-unit results: current ownership

## Codex checkpoint receipt: native140 and quota hold

Current source is Metal `6a48588d`; canonical build/ABI evidence is `70345736` (`native140-build-result.json`). Build exit 0, source changes `{}`, five libraries loaded. Native Emission120 does **not** match Python Emission112; no workload or save/load execution ran. Archive v47 is built. Earlier sizes, framing, "uncommitted", "not built" and resumed-stream statements below describe their historical snapshots, not current status.

All eight Claude streams reported the account quota resetting 17 November at 11am America/Los_Angeles. Existing processes are retained idle; no retry or replacement was made. These worker notes are preserved WIP, not an acceptance verdict. `native-flow-supervisor-account.md` records Codex's current whole-consumer review. Original Body entry/allocation/caller ownership, complete final-link action consumption, full coherence/publication coverage and target numerical/interface obligations remain unfinished. Provisional supplied name/geometry keys and consumer assumptions about bounded sizes, frees or lifetime do not supersede the original producer's semantic/resource operands and canonical lifetime result.

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
| archive | v46, sole version owner (v45 built in native138; v46 adds 131's upstream `compileSource`/`compileOutput`/`compileProduct`) | load |

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
  - **After-selection device completion: in source** (uncommitted; strict syntax-only checks pass for `metal_ir.mm`, `metal_transport.mm`, `metal_lower.mm`, `metal_capture.m`).
    - **`Obligate`** (`metal_transport.mm`): a GPU kernel (KERNEL/INDIRECT) in a switch branch outside every repeat sends nothing itself. Its writes are collected per outermost switch, reached by climbing `RegionAbove`.
    - **The entry.** For each such switch, one completion entry (`Kernel::completes = o`), created once and found again by `completes`, is inserted after the switch's last member kernel. It has original order and parent membership, so it is unconditional. Its `sent` is the collected values.
      - It is constructed as an ordinary GPU kernel: `Builder::Module`, `Entry("mesh_switch_complete", 1, 1, 1)`, an empty body. It has no numerical members.
      - The ordinary `Transport` GPU path binds its protocol from `sent`, and `Wrap` composes acquire → body → `Release`, the canonical `Releasing`.
    - **Order.** `Artifact` gives the entry whole read views (`Whole()`) of the values it publishes, so the relation orders it after every arm's writer.
    - **Counts and persistence.** `Completions` counts the one entry per value. The entry is an ordinary kernel through append, clone, save and load; `completes` is read only at Lower and in `Artifact`, so it needs no archive slot.

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

## Caller tensor handles: dataflow (native in source, uncommitted; Python pending)

1. **Body/root entry (66/68).** Each root ABI slot of the deferred original entry is registered with `metal_ir_upstream_handle`, keyed by the planned Body/root entry and its occurrence, not by an upstream kernel: slot, value, AOTI dtype/device/device_index, layout. The entry borrows that slot rather than stealing it.
   - A caller-owned value (`metal_ir_value_external`) is a caller input or a caller-owned output.
   - A plan-owned value is a prepared internal handle.
   - An output is caller-owned only where its original root/output occurrence says so; body-owned and aliased outputs keep their original owner.
2. **Frontend (`native.py` `Bound.invoked`, granted).** Each operand's `MetalArgument.handle` is the borrowed `tensor_pointer_to_tensor_handle(&THPVariable_Unpack(given))` of the `given` tensor held through End. No `_cdata`, no new references, no per-call boxing. This needs a C++ owner built against torch's Python headers (a dependency to coordinate).
3. **Native (`metal_program.m`).** For each caller-owned descriptor, a `NATIVE_EXTERNAL_HANDLE` site writes `run->arguments[argument].handle` into the task's pointer slot before each call (`NativeExternalHandle`). For each plan-owned descriptor, one prepared handle per residue (`metal_ir_task_prepare_handles`), deleted once with its library owner.
4. **Saved.** Descriptors are saved with their owner record (v47). Handles are never saved.
5. **ABI.** `MetalArgument` {buffer, offset, base, handle}, 32 bytes. Its mirror `metal.py` `Argument` appends `('handle', C.c_void_p)`; that edit is held for compiler's index snapshot.

## Others

- **131:** Transport/Upstream binding and effect methods, environment/compiler, binding groups, sizes and physical, `NativeBuild` compile-unit scheduling, other Lower stages; COPY/FILL in a branch as kernels; the diagnostic port rescan in `MetalOrder`. 131 saves `Value::owned` under v45.
- **133:** the original producer, its call sites and its unit declarations.
- **135:** `metal_neural.mm`, NeuralIO/NeuralModel, the Kernel neural fields, the shared NeuralModel archive owner (`b4e1088c`, v45).
- **For 66's original CppWrapperCpu allocation hooks: native consumer facts** (existing inputs only):
  - `make_buffer_allocation`/`make_allocation`: one storage root per original buffer, `metal_ir_value(role, scalar, rank, shape, layout terms from the original stride, offset, identity)`.
    - `Storages` derives its bytes from shape and layout, so a static size is required.
    - An original size expression must reach native as its value under the AOT known-shape scope, or as its `ShapeEnv.bound_sympy` upper bound. Native has no symbolic storage size, and an unbounded expression is an explicit error, not a default.
  - `AllocationPool.codegen_create`: one storage root for the pool, of the pool's original size.
  - `AllocFromPoolLine`/`codegen_alloc_from_pool`: a value at its original offset in the pool, then `metal_ir_value_view(value, pool)`. `Storages` folds every allocation of the pool into the pool root's lifetime: the pool exists across its allocations, as in the original.
  - `codegen_reinterpret_view`, `ReuseLine` and mutation outputs: `metal_ir_value_view(view, base)`, with the view's own layout and offset.
  - `never_reuse_buffers`: their reason is the ownership input, not a flag. A received tensor is a transport port (`metal_ir_value_receives`); a caller-owned tensor is `metal_ir_value_external`/`metal_ir_value_storage`; a producer-owned prepared operand is `metal_ir_value_owned`.
  - Frees and last use are not inputs. Lifetimes come from those roots and the bindings of the operations that use them, through `Storages` and then the canonical planner (intervals and copies).
  - The entry binding (run_impl input k, constant idx, item 5's operands) binds the native values above. Native retains the resulting bindings through body, control, replay and archive.

- **66's `torch-inductor-cpp-wrapper-binding.patch` records → native consumers** (existing APIs; nothing parsed after execution):
  - **`bound_allocations` with `buffer` set** (`name`, `device`, `dtype`, `shape`, `stride`, `allocation_shape`, `is_pinned`, `graphs`): one storage root, `metal_ir_value(role, scalar(dtype), rank, allocation_shape, layout terms from stride, offset 0, identity)`.
    - Where `shape` differs from `allocation_shape`, the buffer's own value is a second `metal_ir_value` of `shape` and `stride`, joined by `metal_ir_value_view(value, root)`.
    - `device` selects the operations' placement, not the storage owner. `is_pinned` is host-visible storage, which every placed Metal copy already is (shared storage); it needs no further native field.
  - **`bound_allocations` with `buffer` None (a pool, `name` its pool):** one storage root of the pool's size.
  - **`bound_views` "pool"** (`name`, byte `offset`, `dtype`, `shape`, `stride`): `metal_ir_value(…, offset = byte offset / element size)` then `metal_ir_value_view(value, pool root)`. A byte offset that is not a multiple of the element size is an explicit error.
  - **`bound_views` "view"** (`data`, `size`, `stride`, element `offset`, `dtype`, None keeping data's): `metal_ir_value(…)` with that layout and offset, then `metal_ir_value_view(value, data's value)`. A dtype change is a view of the same storage class.
  - **`graphs` (codegened_graph_stack):** which emitted body, top level or a while/cond subgraph, the allocation is printed in. Native needs no lifetime field from it: storage lifetime is `Storages`' over the uses' bindings, and inside a registered repeat or switch over the region spans. The stack must agree with the `metal_ir_repeat`/`metal_ir_switch` registration of those bodies.
  - **Binding.** A bound allocation is a planned value. The canonical planner places it (`metal_ir_value_placed`), and the entry receives its placed copy through the ordinary bindings. No extra `input_handles` position and no runtime view-handle allocation. The rejected draft's `input_handles[inputs + position]` is not a native input.
  - **Sizes.** Native storage is static. Each record's `allocation_shape` and `stride` must reach native as values under the AOT known-shape scope, or as their `ShapeEnv.bound_sympy` upper bounds; an unbounded expression is an explicit error.

- **66/68:** the original control (`ExternCall`/`Inlined`/`WhileLoop`/`Switch`) and lifetime operands. The native inputs that consume them are `metal_ir_value` role, `metal_ir_value_view`, the effects/bindings, `metal_ir_repeat` carried and condition, and transport ports; their results are `metal_ir_storages`, `metal_ir_last_uses`, `metal_ir_order_bounds` and `metal_ir_parts`.

## Current resumed completion responsibility

132 owns the exact conditional Completions/publication block together with the existing original region/result/lifetime consumer. The first post-build draft inserted host Steps for GPU branch sends and cleared the GPU send list; Codex rejected that substitution before acceptance. Finish the original device protocol's publication through retained selected-region result/completion, without a host wait or a one-shot loop. Preserve original wrapper handle rebinding and carried/zero-iteration semantics with66; do not invent unconditional init/copyback. The matched138 evidence is prior source, not evidence for this in-progress continuation.
