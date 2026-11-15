# 68: the returned pre-terminal result (source-defined fields)

Producer 68, 2026-11-14. This is the handoff to root, 66/476, native131/134/135 and frontend104. Every field below is
written in source:
- `compiler/mesh_compiler/lower_inductor.py`;
- `compiler/patches/torch-inductor-aot-lowered-result.patch`, which is new;
- `torch-inductor-graph-lowering-factory.patch` and `torch-inductor-upstream-composition.patch`, both as at HEAD and
  both retained.

All three patches are installed in `/Users/mdot/.venv-mesh-uv` as committed at `ba03a780`: composition, then factory,
then AOT result. The supervisor applied them at fuzz 0, and every installed file has its README "after" hash and parses.
No Torch or Mesh module was imported. The manifest is `numerical68-upstream-install.json`.

Nothing here is a proposal. This revision replaces four parts of the earlier note, all rejected by root:
- the `CompiledAOTI.filename` carrier;
- the import-time `register_backend_for_device`;
- `MeshWrapper`;
- the provenance claim.

## The seam and its stage

The entry is upstream's own, under `torch.no_grad()`:

```python
compile_fx_aot(body, [placeholder meta["val"]],
               inner_compile=partial(compile_fx_inner, graph_lowering=MeshLowering),
               config_patches={"fx_wrapper": True, "aot_inductor.lowered": True,
                               "assume_unaligned_fallback_output": received,
                               "trace.provenance_tracking_level": 1})
```

`compile_fx_aot` sets `V.aot_compilation` and keeps `cpp_wrapper` False, because `fx_wrapper` is set. It then runs
`compile_fx` → `_compile_fx_main`'s AOT branch, in upstream order:
1. `run_pre_grad_passes`;
2. `aot_export_module(trace_joint=False)` under `unlift_effect_tokens`;
3. `_unlift_graph`;
4. `inference_compiler` (joint and post-grad passes);
5. `compile_fx_inner`. In `aot_mode` it has no `count_calls == 0` bailout and no FX cache.

Then `codegen_and_compile` → the factory's `graph_lowering(gm, ...)` (a `MeshLowering`) → `GraphLowering.run` →
`GraphLowering.codegen`.

**Stage.** The result is taken after `scheduler.codegen()`. By then each backend's kernel codegen and `define_kernel`
have run, so the kernel sources exist:
- MPS: `metal_entry` and `metal_source`, and the wrapper's kernel definitions;
- CPU: `defined_kernels` and `KernelDefinitionLine`.

The wrapper lines are as written. The result is taken before `wrapper_code.generate`, so there is no memory planning, no
FX conversion and no wrapper source. It is also before any module or kernel is compiled or loaded. It is not before
source generation.

From `torch-inductor-aot-lowered-result.patch`:
- **`config.aot_inductor.lowered: bool = False`** is the operand that asks for it.
- **`GraphLowering.codegen`**: with `lowered`, after `scheduler.codegen()`, it does `pop_codegened_graph` and returns
  `(self, None)`.
- **`codegen_and_compile`**: under `aot_mode` and `lowered`, `graph.codegen()` replaces `compile_to_fn`. Its
  `V.aot_compilation` return is `output_code.py LoweredAOTI(graph=graph, device_type=graph.device_type)`.
  - `LoweredAOTI` is a new `OutputCode` dataclass, the explicit lowered-result type. No emitted-file slot is used.
  - Its `post_compile` and `set_triton_bundle` load nothing.
- **`compile_fx_aot`** returns `compiled_artifacts.graph` for a `LoweredAOTI`. Its annotation adds `GraphLowering`.
- **The GraphSignature**: `_compile_fx_main` sets `unlifted_gm.meta["graph_signature"] = graph_signature`. That GM is
  `graph.module`.
- **Cond branches**: `WrapperFxCodegen.codegen_switch` writes `SwitchLine(self, node)`, then `codegen_subgraph_common`
  for each branch. That calls `subgraph.graph.codegen()`, which returns the branch's own lowered `SubgraphLowering`.
  Under `lowered`, `define_subgraph_launcher_fn` records it: `wrapper.lowered_subgraphs[name]`. The branch has its own
  `WrapperFxCodegen` and scheduler.
- **Scalar assertions**: under `fx_wrapper`, `ir.AssertScalar.codegen` writes `AssertScalarLine(wrapper, node)`, where
  upstream wrote nothing.
- **Rewrite correspondence**: `pattern_matcher._transfer_matched` runs at both rewrite seams, before
  `match.erase_nodes()`, under `effective_provenance_tracking_level() == 1`. It appends every matched node (`Match.nodes`)
  to the `from_node` of every added node, as `NodeSource(m, pass_name, REPLACE)`.
  - The two seams are `ReplacementPatternEntry.replace_with_graph` (`added_replacement_nodes`) and
    `LoweringPatternEntry.apply` (`replacement`).
  - This adds to `set_current_meta`'s CREATE and `_transfer_meta`'s output-to-output REPLACE.

### Backend selection: a compilation operand, not process-global state

Nothing is registered, and `device_codegens` and `custom_backend` are untouched.
- `graph_lowering` (the factory patch) constructs `MeshLowering`.
- `subgraph_lowering` (the composition patch) constructs every control body's `MeshSubgraph`.
- Both are `_Lowering` classes:
  - `_update_scheduler` is upstream's own, with `MeshScheduler`;
  - `placeholder` keeps each placeholder's own result, described under `inputs` below.
- `MeshScheduler.create_backend(device)` does `V.graph.add_device_info(device)` and then `SELECTED[device.type](self)`,
  where `SELECTED = {"mps": MeshMetalScheduling, "cpu": MeshCppScheduling}`.
- The wrapper is upstream `WrapperFxCodegen`.

### Deleted, together with 66

- the `made` closure;
- `TracingContext.fw_metadata`;
- `force_disable_caches`;
- the provisional `compile_fx` / `compile_to_module`;
- the tail `zip` of `graph_input_names`;
- `num_mutated_inp_runtime_indices`;
- `mesh_made`, `_entries`, the old `Extern` and `Produced.entries`;
- the `SOURCE` tag;
- the `selected=` argument;
- the owner attributes `mesh_line` and `mesh_graph`.

The old versions remain at 8cbc2ab9.

## The adaptation (`adapted`, `_copied`, `_adapt`)

These are source facts, read statically.

- **SEND and RECV are root nodes only.**
  - `placement.py insert_send_recv` is their one constructor (lines 1009/1012). It inserts them into the function's
    graph.
  - A HOP call is one node on one processor.
  - `schedule.py assemble` copies that graph's nodes and borrows each HOP's bodies unchanged from its owner (`get_attr`
    bound from `owner`).
  - So no control body holds a SEND or RECV, and a sent value never crosses a HOP's interface. Appending the sent
    values to the processor body's own output (schedule-emit `adapt`, `keep_output_stride`) changes no HOP signature.
- **Generated HOP bodies have no effects.** A static closure over `operations.py` covers every function reachable from
  the 25 functions given to `while_loop_op` / `switch_op` / `cond_op` (92 functions). None names an effectful op
  (`_linalg_check_errors`, `_print`, `call_torchbind`, `hop_print`, `invoke_leaf_function`, mesh send/recv). Every
  `_linalg_check_errors` is at a formula's top level.
- **Exported HOP bodies:** torch 2.14.0 threads an effect token through a body only for `invoke_subgraph`
  (invoke_subgraph.py 1072–1207). That body holds `with_effects(token, op, ...)` over its own token placeholder and
  returns its token. `cond`, `while_loop` and `switch` thread none.

The adaptation now applies its one rule to **every copied graph**: the body and each control body, recursively.
- `with_effects` becomes its operator, with the token passed through.
- RECV becomes `recv_strided`.
- Each body's output is written first, as its HOP signature has it, so a body's token placeholder is returned in its own
  position.
- AOT's functionalization makes the effects again on its own tokens, including invoke_subgraph's inside its body.
- The root alone appends the sent values after its own results.
- `_engine` runs over that same adapted body. Its RECV/SEND handling is root-only, per the fact above.

## `Produced` (one processor body)

| field | value |
|---|---|
| `processor`, `kind`, `device` | as before (`kind` is "cpu", "mps" or "ane") |
| `module` | `Scheduled.processor_modules[key][processor]` as scheduled |
| `body` | the adapted copy `compile_fx_aot` traced, whose graphs are the provenance roots |
| `sources` | `{(id(copy graph), copy node name): original node}` over the body and every control body |
| `sent` | the original sent values, in the order they are appended after the returned values |
| `graph` | `LoweredAOTI.graph`, the GraphLowering. None for the Neural Engine |
| `signature` | `graph.module.meta["graph_signature"]`, the AOT export's GraphSignature. Neural Engine: `ExportedProgram.graph_signature` |
| `inputs` | `Input(position, node, name, value)`. `name` is `signature.user_inputs[position]`. `value` is `graph.mesh_placeholders[graph.qualify_name(name)]`, the original `GraphLowering.placeholder` result keyed as it keys `graph_inputs` and `graph_input_names`: a TensorBox, a sympy expression, a TorchBindObject, a generator's, or **None for an admitted None input** (which `graph_inputs` does not hold). Engine: the program's placeholder |
| `outputs` | `Output(position, node, name, value, sent)`. `name` is `signature.user_outputs[position]`, `value` is `graph.graph_outputs[position]` |
| `kernels` | `{id(KernelCallLine): Kernel}`, one entry per call, over `graph.wrapper_code` and every `lowered_subgraphs` wrapper |
| `externs` | the top scheduler's `ExternCall` records, in emission order. For a lowered cond branch, use `branch.scheduler.mesh_externs` |
| `seeds` | `Seeds(buffer, uses)` |
| `model` | `Model(program, model, selected, received, handles)`, the Neural Engine's. None for Inductor bodies |

`Produced.origins(fx_nodes)` walks the nested NodeSource chains to every source, which is the many-to-many set. For an
IR node, use `produced.origins(ir_node.get_origins())`.

## Per-call records: `Kernel`

Each call's record is made at that call and never overwritten. The tuple is
`wrapper.launched_kernels[i] = (owner, launch, line, graph, scheduler, produced)`:
- MPS: the operands patch's `(owner, launch)`, completed by `MeshMetalKernel.call_kernel` after its
  `generate_kernel_call`.
- CPU: `MeshKernelGroup.call_kernel`.

The `KernelCallLine` is the call's identity.

| field | mps (`MeshMetalKernel`) | cpu (`MeshKernelGroup`) |
|---|---|---|
| `place` | "mps" | "cpu" |
| `graph` | `V.graph` at the call (GraphLowering or control body SubgraphLowering) | same |
| `scheduler` | `V.graph.scheduler` at the call: an inlined body's scheduler for **that occurrence** | same |
| `line` | its `KernelCallLine` (`line.wrapper` its wrapper) | same |
| `owner` | the backend kernel: range trees, `range_tree_nodes`, `body`, `args` (with `workspace_args`), `headers`, `mesh_indirect` | the KernelGroup: `args`, `loops_code`, `scheduled_nodes`, `mesh_proxies` |
| `launch` | `(name, values, types, grid, group, error)` | `(name, arg_defs, call_args, arg_types)` |
| `produced` | `(metal_entry, metal_source)` at the call, as `codegen_kernel` made and `define_kernel` defined it | `defined_kernels[name]`: (declared entry, final source) |
| `schedule` | the `node_schedule` received, with its DisableReduction/EnableReduction markers | `scheduled_nodes` |
| `ranges` | one `Ranges` for each second-pass SchedulerNode | `(CppKernelProxy, nodes)` for each `finalize_kernel` |
| `scheduled`, `origins` | its SchedulerNodes, and their original nodes | same |

`Ranges(node, variables, reduction, accesses)`:
- `accesses` holds `(method, buffer, index, mode, graph, fx_node)`.
  - `graph` and `fx_node` are `V.interpreter.graph` and `V.interpreter.current_node` at the access: the LoopBodyBlock,
    root or `masked_subblockN`, and its node.
  - These are the accesses `CSEProxy` passes to the kernel. A `store_cache` hit and a store to a `removed_buffers` buffer
    are absent, and an `indirect_load` is a `load`.

## Control: `ExternCall` and `Inlined` (each occurrence, at the original method boundaries)

**`ExternCall(node, graph, scheduler, lines, inlined)`** is made by `MeshScheduler.codegen_extern_call`, at its return,
for every extern scheduler node:
- `node` is the ir node as lowered:
  - `ir.WhileLoop`: `cond_subgraph`, `body_subgraph`, `carried_inputs`, `additional_inputs`, `outputs`,
    `stack_output`;
  - `ir.InvokeSubgraph`: `subgraph`, `inputs`, `outputs`;
  - `ir.Conditional`, whose branches are lowered separately: `lowered_subgraphs`;
  - FallbackKernel, MultiOutput, DynamicScalar, AssertScalar and the other externs.
- `graph` and `scheduler` are where it was called.
- `lines` is every wrapper line its codegen wrote, as the line objects themselves, in order. Upstream's control scaffolding
  is among them as upstream's own str lines.
- `inlined` is each body inlining its codegen made, in order. `while_loop` makes three: its cond before the loop, its
  body, and its cond at the loop's end.

**`Inlined(graph, scheduler, lines, externs)`** is made by `MeshSubgraph.codegen_subgraph`. That is
`GraphLowering.codegen_subgraph`, called by `codegen_subgraph_by_inlining`. Its fields:
- `graph` is the body's SubgraphLowering, identical to `node.cond_subgraph.graph` / `body_subgraph.graph` /
  `subgraph.graph`.
- `scheduler` is **this occurrence's own** Scheduler. Each inlining reschedules the body (`_update_scheduler`), so the
  cond's two occurrences have two schedulers. `Kernel.scheduler is inlined.scheduler` places a kernel in its occurrence.
- `lines` are the lines it wrote.
- `externs` are its scheduler's ExternCall records, which gives nesting directly.
- Its interface is the body's own: `graph.mesh_placeholders` / `graph.graph_input_names` (qualified) and
  `graph.graph_outputs`, bound to the ir node's typed inputs and outputs in upstream's order.

So the control hierarchy and each occurrence are explicit records. A consumer never parses the str scaffolding or infers
nesting from `EnterSubgraphLine` spans. A `while_loop`'s first cond inlining is not bracketed by `EnterSubgraphLine`, and
that is upstream as is.

## Terminal composition and emission (signatures unchanged)

`composed(kernel, made)` and `emitted(kernel, composition=None)` keep their signatures.
- The collection is `made.callables.source`, with the binary64 notice first. Native129 prints it.
- The protocol is `callables.acquire` / `later` / `release`, each of which may be None. The arguments are in
  metal_ir.h's protocol order.
  - The readiness parameters are `classes` and `members`, at `class_slot` / `member_slot`, or `later_class_slot` /
    `later_member_slot` for the later pass. These are the names in `struct metal_ir_upstream_readiness`; `spans` is
    gone.
  - This needs `metal.py`'s `UpstreamReadiness` mirror to carry the header's field names.
- **`composition=None`**: the call's own `kernel.produced`, with no replay. Metal additionally gets `_embed_headers`.
- **A composition**: the retained owner's own upstream emitter:
  - Metal: `owner.codegen_kernel(entry, composition)` under its graph;
  - CPU: `owner.codegen_group(None, composition)` through `defined_source`.

  This is the terminal composition. Both passes come from the same owner after ordinary allocation, and distinct
  first and later programs come from one solution.
- Precision, target and custom ops are each as upstream lowered them. No subset is selected and nothing numerical is
  added.

## The Neural Engine `Model` (native135/134's consumer)

`Model(program, model, selected, received, handles)`:
- `program` is the ExportedProgram (Core ATen).
- `model` is the MLModel. Its MIL program holds each operation's EXIR_DEBUG_HANDLE scope, and its control flow and state
  as converted.
- `selected` is the conversion's own operands as given: `ct.convert(program, **selected)`, which is `convert_to`,
  `compute_units`, `compute_precision`, `minimum_deployment_target` and `skip_model_load`.
  - `compute_precision` FLOAT32 is coremltools' precision selection (no fp16 rewrite). It is not evidence of a binary64
    result.
- `received` holds each (input position, RECV node).
- `handles` maps `{debug_handle: original nodes}`.

## What 66's `lower_metal.py` consumes instead of `entries`

- Walk `produced.externs` and `produced.kernels`, plus the lines of each `ExternCall` / `Inlined` as given:
  - a `KernelCallLine` → `produced.kernels[id(line)]`;
  - an extern → `ExternCall.node`, with `produced.origins(node.get_origins())`;
  - a control node → its `inlined` occurrences, or for cond, `lowered_subgraphs`.
- Replacements for the old fields:
  - `entry.scheduled.node` → `ExternCall.node`;
  - `entry.origins` → `Kernel.origins` / `produced.origins(...)`;
  - `Input.value` → the placeholder result (it may be None);
  - `Output` gains `name`.

## Not covered, stated

- **C++ GEMM templates** (`max_autotune` CPU) call `generate_kernel_call` outside `KernelGroup.call_kernel`. They make
  no record, so their `KernelCallLine` has no `Kernel`. `max_autotune` is off here.
- **Seeds**: only `graph.module`'s top-level `inductor_lookup_seed` nodes are collected, not a control body's own.
- **`Input.value` for a None input is None.** This is the original placeholder's result, so `name` and position remain
  its interface.
