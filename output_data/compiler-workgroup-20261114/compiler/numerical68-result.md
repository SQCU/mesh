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

A fourth patch, `compiler/patches/torch-inductor-cpu-kernel-results.patch`, is new and **not installed**. It changes
`cpp.py` from `74b2de26…` (composition's after) to `0eb0c1e9…`. The fuzz-0 dry run, application, reverse dry run and
parse all passed on a temporary copy of the installed file. The lowerer's CPU records below need it: the call record
reads its `KernelGroup.arg_defs`, and its `codegen_conditions` hunk keeps the scalar factory subclass's emission as
upstream's. The supervisor owns its installation.

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
| `kernels` | `{id(KernelCallLine): Kernel}`, one entry per call, over `graph.wrapper_code` and every `lowered_subgraphs` wrapper: every MPS kernel call, CPU group call and CPU template call |
| `externs` | the top scheduler's `ExternCall` records, in emission order. Every control body's are reached through them: `ExternCall.inlined` and `.lifted`, each with its own `externs`, nested |
| `seeds` | `Seeds(graph, buffer, uses)` for the root graph and every body GraphLowering made under it |
| `model` | `Model(program, model, selected, received, handles)`, the Neural Engine's. None for Inductor bodies |

`Produced.origins(fx_nodes)` walks the nested NodeSource chains to every source, which is the many-to-many set. For an
IR node, use `produced.origins(ir_node.get_origins())`.

## Per-call records: `Kernel`

Each call's record is made at that call and never overwritten. The tuple is
`wrapper.launched_kernels[i] = (owner, launch, line, graph, scheduler, produced, ...)`:
- MPS: the operands patch's `(owner, launch)`, completed by `MeshMetalKernel.call_kernel` after its
  `generate_kernel_call`.
- CPU group: `MeshKernelGroup.call_kernel`, after the original `KernelGroup.call_kernel`. `launch` is
  `(line.kernel_name, group.arg_defs, line.call_args, line.arg_types)`:
  - the call's own `KernelCallLine` fields, as its `generate_kernel_call` wrote them;
  - the definition's own declarations, kept by the new patch's `codegen_group` hunk when `flush` defined the group
    before this call.

  Nothing calls `cpp_argdefs` again.
- CPU template: `MeshCppScheduling.codegen_template`, at the return of the original `codegen_template` (cpp.py 5921:
  render, then `define_kernel`, then `CppTemplateKernel.call_kernel`). It records:
  - the one definition that codegen made, through `MeshCppScheduling.define_kernel`. This is the definition producer
    common to groups (`flush`) and templates: `(V.kernel, nodes, name)` for every definition, appended to
    `mesh_defined`. A template defines under its `CppTemplateKernel`'s handler;
  - the one `KernelCallLine` that codegen wrote;
  - a seventh element, the nodes that definition received.
  Nothing is regenerated after the call: `launch` is the line's own call arguments and types.

The `KernelCallLine` is the call's identity. `CppTemplateKernel` has no subclass; cpp_template.py 101
`make_kernel_render` makes the one used for a call.

| field | mps (`MeshMetalKernel`) | cpu group (`MeshKernelGroup`) | cpu template (`CppTemplateKernel`) |
|---|---|---|---|
| `place` | "mps" | "cpu" | "cpu" |
| `graph` | `V.graph` at the call (GraphLowering or control body SubgraphLowering) | same | same |
| `scheduler` | `V.graph.scheduler` at the call: an inlined body's scheduler for **that occurrence** | same | same |
| `line` | its `KernelCallLine` (`line.wrapper` its wrapper) | same | same |
| `owner` | the backend kernel: range trees, `range_tree_nodes`, `body`, `args` (with `workspace_args`), `headers`, `mesh_indirect` | the KernelGroup: `args`, `arg_defs`, `loops_code`, `scheduled_nodes`, `mesh_proxies` | the template kernel: `args`, `local_buffers`, `removed_buffers` |
| `launch` | `(name, values, types, grid, group, error)` | `(line.kernel_name, arg_defs, line.call_args, line.arg_types)`, `arg_defs` its `codegen_group`'s (new patch) | `(line.kernel_name, TEMPLATE_DECLARED, line.call_args, line.arg_types)` |
| `produced` | `(metal_entry, metal_source)` at the call, as `codegen_kernel` made and `define_kernel` defined it | `defined_kernels[name]`: (declared entry, final source) | same |
| `schedule` | the `node_schedule` received, with its DisableReduction/EnableReduction markers | `scheduled_nodes` | the nodes its `define_kernel` received: the template node, then its epilogue nodes |
| `ranges` | one `Ranges` for each second-pass SchedulerNode | `(CppKernelProxy, nodes, local_buffer_context, local_buffers)` for each `finalize_kernel`; each proxy's kernels hold their `CppRanges` (below) | `TEMPLATE_RANGES` |
| `scheduled`, `origins` | its SchedulerNodes, and their original nodes | same | same |

`Unretained(result, owner)` names an actual result that the record does not hold, and its original producer. It is not
iterable and holds no value of that result, so a consumer cannot read it as empty:
- `TEMPLATE_DECLARED`: the template definition's parameter declarations. `CppTemplateKernel.def_kernel`'s render hook
  (cpp_template_kernel.py 113–114) makes them from `cpp_argdefs` at render and returns them only as text in the source.
- `TEMPLATE_RANGES`: the template's loops, indices and accesses, and its epilogues'.
  - Its CppTemplate renders its own as source (`ir.CppTemplateBuffer.template`).
  - Its epilogues are rendered through `CppKernelProxy(KernelGroup())`, an upstream group of its own
    (cpp_template_kernel.py 276–278, 348–350), not the selected backend's.
  - A template's effects are not empty. Its scheduled nodes' `read_writes` are its buffer-level reads and writes.

`Ranges(node, variables, reduction, accesses)`:
- `accesses` holds `(method, buffer, index, mode, graph, fx_node)`.
  - `graph` and `fx_node` are `V.interpreter.graph` and `V.interpreter.current_node` at the access: the LoopBodyBlock,
    root or `masked_subblockN`, and its node.
  - These are the accesses `CSEProxy` passes to the kernel. A `store_cache` hit and a store to a `removed_buffers` buffer
    are absent, and an `indirect_load` is a `load`.

## CPU kernels and accesses: `CppRanges` (66's CPU producer input)

The original factory makes each CPU kernel:
- `MeshCppScheduling.kernel_proxy_cls = MeshCppKernelProxy`. This is what `codegen_node` and `codegen_outer_loop_node`
  construct.
- Its `kernel_cls`, `vec_kernel_cls` and `tile2d_kernel_cls` are `MeshCppKernel`, `MeshCppVecKernel` and
  `MeshCppTile2DKernel`. `CppKernelProxy.codegen_functions` makes each through `codegen_kernel`
  (`kernel_group.new_kernel(cls, …)`, then `run`).

Each kernel subclasses the original kernel class through one mixin, `MeshCppAccesses`, and records at the original
methods:
- `mesh_split`: its `set_ranges` result, `(vars, reduction_vars)`.
- `mesh_ranges`: one `CppRanges(node, variables, suffix, accesses)` per `set_current_node`, that is, per
  `SchedulerNode.codegen` in that kernel.
  - `variables` is `mesh_split`.
  - `suffix` is whether `run` gave the node to `write_to_suffix`. In that case the node receives `vars` and no
    reduction vars (cpp.py `fn(vars, ())`), and its code is the kernel's `reduction_suffix`.
- `accesses`: each `(method, buffer, index, mode, graph, fx_node, mask)` as that kernel's `load`, `store` or
  `store_reduction` received it.
  - The first six fields are the MPS `Ranges.accesses` fields.
  - `mask` is the kernel's `_load_mask` at the access. That is `CppVecOverrides.masked`'s mask, which its masked loads
    use. It is None outside one, and always None for a scalar kernel, whose `CppOverrides.masked` guards a lambda
    instead.
  - The access is taken at the outermost class method, before the original's own `rename_indexing`, a tile's
    `transform_indexing` or a vector load's scalar fallback. None of these reaches the mixin again: the tile's MRO is
    `MeshCppTile2DKernel, MeshCppVecKernel, MeshCppAccesses, CppTile2DKernel, CppVecKernel, CppKernel`.
  - `CppVecOverrides.load_seed`'s direct `V.kernel.load` is among the accesses.
- `mesh_indirect`: `{symbol: (var, size, check)}` from the kernel's overrides (`MeshCppOverrides`,
  `MeshCppVecOverrides`, `MeshCppTile2DOverrides`). These share one `MeshIndirect` mixin with `MeshMetalOverrides`.

The domain as the original emits it is the retained objects' own fields, not replayed:
- `proxy.loop_nest`: the LoopLevels as `tile` and `codegen_loops` left them.
- `proxy.kernels`: the original's selection (scalar; vector and tail; tile2d and its tails). Each kernel's
  `active_ranges` and `inner_itervars` are as `codegen_functions` set them.
- Each kernel's `itervars`, `ranges` and `reduction_depth`.
- A vector kernel's `tiling_factor`, `tiling_idx`, `tail_size` and `num_elems`.
- A tile kernel's `tiling_indices`, `outer_idx`, inner and outer tail sizes and `num_elems`, and
  `inner_is_tiling_idx`.

A scalar kernel that `codegen_functions` made but did not select (the vector-tail case) is not in `proxy.kernels`.

Outer-loop fusion:
- The finalized proxy's `loop_nest.kernel` is an `OuterLoopFusedKernel`, and each of its `inner` LoopNests' kernel is a
  fused proxy.
- The record's third element is `V.local_buffer_context` at that `finalize_kernel`.
  - For the fused finalize (cpp.py 5870, inside the `with LocalBufferContext(…)` at 5809) it is that
    `LocalBufferContext`.
  - For the fallback's and `codegen_node`'s finalizes (5894, 5912) it is virtualized.py's `NullHandler`.
- The fourth element is that context's `local_buffers` at the finalize, `{local name: ir.Buffer}`: a new dict holding
  the same Buffer objects `add_local_buffer` registered. These are the kernel-local buffers its accesses name and
  `codegen_loops_impl` declares.
  - It is copied at the finalize because `LocalBufferContext.__exit__` clears `local_buffers` in place
    (cpp_utils.py 384). The retained context's own `local_buffers` is empty after the fusion returns.
  - It is taken under the original's own test, `isinstance(V.local_buffer_context, LocalBufferContext)` (cpp.py
    2717). Outside a context it is `{}`: the kernel's `KernelArgs.input`/`output` are unpatched, so no name is local.
- `__exit__` keeps the context's `global_buffers`, `global_to_local` and `removed_buffers`. They are read from the
  context itself, not copied.

Template epilogues are rendered through cpp_template_kernel.py's own `CppKernelProxy(KernelGroup())`, so they have no
`CppRanges`. That stays `TEMPLATE_RANGES`.

### Source fields changed for 66 and 104 (as landed)

- `Kernel.ranges` (cpu group): each element was `(proxy, nodes)` and is now
  `(proxy, nodes, local_buffer_context, local_buffers)`.
  - `local_buffers` is `{local name: ir.Buffer}` as held at the finalize (above). It replaces reading
    `r[2].local_buffers`, which is empty after the context exits.
  - 104's `ranged["cpu"]` reads `r[0]` and `r[1]`, which are unchanged.
- `Kernel.launch` (cpu group): the same 4-tuple shape, from the new sources above. `arg_defs`, `call_args` and
  `arg_types` are tuples.
- New on each `proxy.kernels` entry: `mesh_ranges`, `mesh_indirect`, `mesh_split` and `mesh_suffix`.
  - Their classes are the `MeshCpp*Kernel` subclasses, so 104's `type(kernel)` now records those.
  - `isinstance` against the upstream classes holds.
- `MeshMetalOverrides` is now `MeshIndirect` plus `MetalOverrides`, with the same method and the same behavior.

## Control: `ExternCall`, `Inlined` and `Lifted` (each occurrence, at the original method boundaries)

**`ExternCall(node, graph, scheduler, lines, inlined, lifted)`** is made by `MeshScheduler.codegen_extern_call`, at
its return, for every extern scheduler node, whatever its operator. No operator is selected:
- `node` is the ir node as lowered, whole. A general extern holds its kernel and `op_overload`, `inputs`,
  `constant_args`, `kwargs`, layout and outputs as ExternKernel holds them. `ir.RandomSeeds` (`aten.randint.low_out`)
  is one of them. Control nodes:
  - `ir.WhileLoop`: `cond_subgraph`, `body_subgraph`, `carried_inputs`, `additional_inputs`, `outputs`,
    `stack_output`;
  - `ir.InvokeSubgraph`: `subgraph`, `inputs`, `outputs`;
  - `ir.Switch` (ir.py 11130): `selector`, `operands`, `outputs`, `branches`.
- `graph` and `scheduler` are where it was called.
- `lines` is every wrapper line its codegen wrote, as the line objects themselves, in order. Upstream's control scaffolding
  is among them as upstream's own str lines.
- `inlined` is each body inlining its codegen made, in order. `while_loop` makes three: its cond before the loop, its
  body, and its cond at the loop's end.
- `lifted` is each body its codegen lowered on its own, in order: a Switch's branches. One call's codegen makes one
  kind:
  - a Switch lifts (wrapper_fxir.py 169 `codegen_switch` → `codegen_subgraph_common`);
  - `while_loop` and `invoke_subgraph` inline (wrapper.py, in `aot_mode`).

The native whole consumer takes every `ExternCall.node` as it is. A per-operator whitelist has no producer basis, since
every extern of every graph is recorded the same way. 66's current `lower_metal.py` raises "has no native consumer yet"
for every extern but send and receive. That is the consumer gap, and this record does not narrow it.

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

**`Lifted(graph, scheduler, wrapper, externs)`** is made by `MeshSubgraph.codegen`, upstream's
`GraphLowering.codegen` that `codegen_subgraph_common` calls under the body's graph handler (wrapper.py 4861–4892). It
is appended to `self.parent.scheduler.mesh_lifted`, where `parent` is make_subgraph's: the graph whose scheduler is
inside that Switch's `codegen_extern_call`. Its fields:
- `graph` is `Switch.branches[b].graph`, which is also `lowered_subgraphs[graph.name]`.
- `scheduler` is the Scheduler of that codegen.
- `wrapper` is its own `SubgraphFxWrapperCodegen` (wrapper_fxir.py 279), with its lines, `launched_kernels` and
  `lowered_subgraphs`.
- `externs` are its scheduler's ExternCall records.

A body is lowered once per wrapper (`already_codegened_subgraphs`). A later Switch of the same branch graph, such as a
while cond's second inlining, calls the lowered body by name. Its `lifted` is empty, and its branch's `graph` is that
record's object.

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
  - a control node → its `inlined` or `lifted` occurrences, each with its own `externs`;
- Replacements for the old fields:
  - `entry.scheduled.node` → `ExternCall.node`;
  - `entry.origins` → `Kernel.origins` / `produced.origins(...)`;
  - `Input.value` → the placeholder result (it may be None);
  - `Output` gains `name`.

## Seeds in every graph

`_recursive_joint_graph_passes` runs `replace_random_passes`, which is joint_graph.py 758's call, on the body and on
every control body (compile_fx.py 589–634, `_get_subgraph_names`). So each body graph has its own `inductor_seeds`,
lowered to its own `ir.RandomSeeds` in its own GraphLowering, and its own `inductor_lookup_seed` nodes:
- `_Lowering.make_subgraph` records each body GraphLowering it makes (`mesh_subgraphs`). It is the one constructor of
  a body's: ir.py 8592 `SubgraphBuffer` and 11065/11243/11545 InvokeSubgraph/Switch/WhileLoop. `PointwiseSubgraphLowering`
  is an Interpreter that makes no buffers.
- `_graphs` is the root and that tree, depth first in the order made.
- `Seeds(graph, buffer, uses)` is one per RandomSeeds buffer of each graph. `uses` holds that graph module's lookups'
  `(index, original nodes)`. The buffer's own `ExternCall` is in that graph's scheduler records.

## SCAN in the selected CPU/MPS result (frontend104's question)

The source says it is absent:
- `ops.scan`, which is the LoopBody scan shim `bind_scan_shim` and its `combine_fn` closure (loop_body.py 596–600,
  856–863), is reached only from `ir.Scan.store_reduction` (ir.py 3034).
- `ir.Scan` is made only by `Scan.create`. Its callers are lowering.py 8001–8097 (cumsum, cumprod, logcumsumexp,
  cummax, cummin) and 9457 (`associative_scan`).
- `Scan.create` returns `[None]` without `BackendFeature.SCAN` (ir.py 3085), and each caller then falls back to aten.
  `associative_scan` raises before it on such a device (lowering.py 9408–9415).
- `has_feature` reads the registered scheduling's features (common.py 471–488):
  - `CppScheduling.backend_features` is `{INPLACE_BUFFERS, REDUCE_TO_SINGLE_ELEMENT}`;
  - `MetalScheduling` has none (`BaseScheduling.get_backend_features`, scheduler.py 10600);
  - the selected Mesh subclasses inherit these and override nothing.
- So a cumulative op on CPU/MPS arrives as a general extern, a FallbackKernel `ExternCall`. No `ir.Scan` and no scan
  shim closure is in the result.
- The scan HOP is rewritten to `while_loop` (post_grad.py 662 `decompose_scan_to_while_loop`), so it arrives as
  `WhileLoop` control.

There is no combine subgraph owner to publish for these targets.

## Not covered, stated

- **The CPU template's composed pass.** `emitted(kernel, composition)` composes a group through the patched
  `KernelGroup.codegen_group`. A template has no such hook, and composing one means its own upstream hook. The
  definition's parameters come from `CppTemplateKernel.def_kernel`'s render hook, and the body opening comes from the
  rendered template. Neither is patched yet. A template's as-produced emission, `emitted(kernel)`, is complete.
- **A template's `TEMPLATE_DECLARED` and `TEMPLATE_RANGES`**, above. Retaining them is the same `def_kernel` and
  render seam's work; they are not inferred.
  - Root placed this after the ordinary CPU kernels' accesses, which are now landed.
  - What remains:
    - the `def_kernel` hook retaining its `arg_defs`;
    - the epilogue proxies, made through the selected factory rather than cpp_template_kernel.py's
      `CppKernelProxy(KernelGroup())`;
    - the template's own body owner and a composition hook.
- **`torch-inductor-cpu-kernel-results.patch` is not installed.** Until it is:
  - `MeshKernelGroup.call_kernel` finds no `arg_defs` and raises `AttributeError`;
  - upstream's `type(self) is CppKernel` test would not select the scalar factory subclass.
- **`Input.value` for a None input is None.** This is the original placeholder's result, so `name` and position remain
  its interface.
