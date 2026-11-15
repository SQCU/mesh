# Consume the original complete compiler result

Current assignment, 2026-11-15. Existing worker66 UUID `476e57ac-cf94-49e8-801a-a3ded988dd77` owns `lower_metal.py` and `metal.py`, with exact driver/IR changes coordinated.68 owns `lower_inductor.py`/Torch patches;67 methods;104 capture/alias/placement/runtime/archive; executor native C. Preserve their source and ordinary committed history.

Original composition CR (numerical boundary, items2/3/5/6/7 and Complete replacement), AOT A1–A10, source-lowering and literal schedule-emit prescribe Scheduled → original upstream result → canonical analysis/allocation → original composed emitter → native/archive. They do not require legacy Kernel/Builder fields, a new schema or another compiler. Current dated ownership account is evidence only.

The active whole unit consumes the actual original `FxConverter.generate` typed wrapper graph and original line→produced-node/source relation exposed by68 through the explicit original AOT return.68 adds selected CPU/MPS kernel and typed while/invoke control hooks at that owner. Consume its source-written result fields, not a guessed parallel representation. Remove displaced `planned/called/extern/inline/INERT`, result-list and name-resolution interpretation together with their responsibilities. Do not replace them with another handwritten per-op Python interpreter, resubmit the whole numerical graph to a second compiler, or count explicit missing-op errors as domain coverage.

Preserve the complete original arity/pytree/None/mutation/alias/out result and exact input/output/call occurrence. Direct Tensor FallbackKernel is itself the result; `.outputs` is not its fabricated child list. Original mutation outputs and nested results need actual bindings. Runtime scalar definitions/assertions remain values, not example hints; empty UnbackedSymbolDefs produces no nodes through its original converter. Original run_wrapper_ir_passes ordering/semantics is not an independent consumer choice.

Original aliases, mutation/reuse/never-reuse, allocation/free/workspace/state/extern ownership and call/control/result uses supply canonical lifetime analysis. Free having no FX node proves only converter symbol-table behavior. Keep omitted original facts on the retained owner and pair their canonical native consumer with132. Never supply missing semantics through guessed safety copies, defaults or last-use rules. Replace repeat's extra init/copy-back storage with the actual original control/value ownership; retain source-required zero-iteration/result semantics. Canonical transport/source relations remain the execution-dependence owner, not a new construction scheduler.

Existing direct native projections and retained owners remain useful: full Scheduled devices/machines/engines, processor_modules, one compilation Context, ConstructedMachine.produced/context/interfaces, per-occurrence Binding6, Launched arguments, Emitted/Region/inlining association.104's archive consumes these exact references. Replacing wrapper handling must preserve or directly replace the same result ownership, not introduce opaque GraphLowering serialization or name matching. The actual fresh terminal caller is Target.instanced → lower_metal.emitted after ports/planned roots and before lower; saved load consumes finalized artifacts.

The single `_Bodies` owner consumes original LoopBody/InterpreterShim/sympy_interp with canonical Graph.reach, caches immutable body/graph/cones and evaluates masks per bound scope. CPU producer records retain final scalar/vector/tail/aggregate-reduction and local declarations. Native canonical ISL consumes full domains/correlation/known guards plus data-dependent flags and source identities; no private INDEX schema, whole-storage approximation or source parser.

Complete original selected extern/template call, allocation/result/effect and ABI consumers with68/native. Preserve actual compiler executable/language/options, entry linkage, parameter types and per-call launch. No indirect per-task execution regime replaces original static whole-program composition. Native135 state `{name,read,written}`, actual MLModel declared features, EP buffer/mutation identity and selected precision require real Python projection; no source-width cast or CPU substitute. State/RNG/error and method numerical conditions remain part of the same scope.

The current matched native ABI is native138 (manifest `native138-build-result4.json`, source `97492e29`; its every-field C/ctypes check covers `metal.py` SHA `b356eed5…`). It releases no runtime, import or workload. Newer native source after 138 is not that library; pair exact declarations and a matched build before imports. Source-only work may continue. No imports/workload/new probe/harness, extra agents, source backups/reset/clean/stash or worker commits. Run relevant source checks, publish actual fields and connected changes promptly, and continue through ordinary supervisor checkpoints. Public retry5 is unchanged and queued for coherent whole-family source, not per-failure admission.

## 66 result boundary: original native wrapper result → native body (2026-11-15, source-backed, installed torch 2.14.0)

Replaces the rejected converted-gm `walk` (WIP `64e88793`). No FX/line walker, CppWrapper method copy, codegen replay, AOTI container loader or per-task dylib. The typed numerical/transport/control results stay retained before emission, as now. The original native emission is selected at its own original stage, compile_fx_aot's config: `cpp_wrapper` in place of `fx_wrapper`. CppWrapper is never run after FxWrapper, and numerical codegen is never replayed. The fields below are the original objects' own (the GraphLowering, its `wrapper_code`, AotCodeCompiler's builders as the original made them), not a new record schema.

**Producer (68), once, at compile_fx.py's original AOT branch (1709–1758, `cpp_wrapper` on, `aot_mode`):**
1. Emitted body: `graph.codegen_with_cpp_wrapper()` (graph.py 2628), on the branch the original itself takes:
   - ordinary CPU and MPS: `self.codegen()` once (the `else` at 2795–2796; `has_gpu` covers only cuda and xpu, 2641);
   - the two-pass path (2653–2793: the JIT variant run for autotuning, then `_aot_output` with its config header) only for CUDA/XPU, or for CPU with a user-defined Triton kernel under AOT with `autotune_at_compile_time` off (2645–2652).

   The selected result keeps whichever branch was taken; a replay is never assumed. Its `wrapper_code.value` and `kernel_code.value`, and `graph.wrapper_code` itself (CppWrapperCpu or CppWrapperMps) with its `aoti_model_class_name`. Control bodies are emitted inside it by the original (`codegen_subgraph` 2887, `codegen_subgraph_prefix` 2732, `codegen_while_loop` 2906: handles rebound by `aoti_torch_assign_tensors_out`; `codegen_tensor_item` 1584).
2. Entry and handle ownership (cpp_wrapper_cpu.py):
   - `{aoti_model_class_name}::run_impl(input_handles, output_handles, stream, proxy_executor)` (978);
   - input handles are stolen, `steal_from_raw_handles_to_raii_handles(input_handles, len(V.graph.graph_inputs))` (1030). Position k is `V.graph.graph_inputs` key k (1052–1066); a `sympy.Expr` input is read from its one-element handle by `codegen_tensor_item`;
   - constants are `constants_->at(idx)` in `V.graph.constants` order (1078–1081), owned by ConstantHandle and never moved; `constants_info_` comes from `codegen_model_constructor` (1210);
   - outputs are written into `output_handles` and stolen by the caller.
3. Compile-only object actions of `AotCodeCompiler.compile` (codecache.py), each its CppBuilder as built (`torch-inductor-cpp-build-result.patch`: `get_command`, `get_product`, `get_source_operands`, `get_target_operand`, `get_build_directory`; its environment after `_set_gpu_runtime_env`, 2656):
   - `wrapper_builder` (3206), with its `_precompile_header` action where `aot_inductor.precompile_headers` is set;
   - `kernel_builder` (3215);
   - the consts object: `object_builder` (2965), or `run_asm_build_object` (2973).

   For `so_builder` (3476), only its link operands (`get_link_operands`), never its shared-library product.
4. Extern call table, serialized: `V.extern_kernel_nodes` (each appended at ir.py 9967; its index at cpp_wrapper_cpu.py 4110) and `graph.extern_node_serializer(...)` (compile_fx.py 1726–1734). AotCodeCompiler writes it as `wrapper_path.with_suffix(".json")` (codecache.py 3009–3014). The body's call site is `aoti_torch_proxy_executor_call_function(proxy_executor, extern_kernel_node_index, …)` (`generate_fallback_kernel_with_runtime_lookup_aot` 4082/4112; shim.h 627). Each op in `custom_ops_to_c_shims` is a direct c-shim by its declaration.
5. Per-buffer, binding and lifetime operands, retained before print (U3 and U4 below). The whole native wrapper alone is not enough: its internal allocation, view, effect and transport operands must not be opaque, and no allocation may bypass the canonical Mesh owner. These are:
   - the planned lines in `graph.wrapper_code.lines` after `run_wrapper_ir_passes`, each with its buffer, layout and reuse/free;
   - the CppWrapper allocation emission over them (`make_buffer_allocation` 2461, `make_allocation` 2472 → `aoti_torch_empty_strided` shim.h 298; pools `codegen_alloc_from_pool` 2563 → `aoti_torch__alloc_from_pool` 272), views (`codegen_reinterpret_view` 2591) and handle assignment;
   - each kernel call's and extern call's buffer operands;
   - `never_reuse_buffers`, mutations and aliases.

   They are composed with 131's and 132's canonical storage, binding and lifetime at the producer, before the source is printed. The seam is the original target-address/allocation hooks, each of which gets the original result before any target text:
   - `make_buffer_allocation` (2461) gets the original buffer, `V.graph.get_allocation_size`, shape, stride and dtype, and delegates to `make_allocation` (2472);
   - `AllocationPool.codegen_create` (memory_planning.py 458) gets each pool;
   - `AllocFromPoolLine` (memory_planning.py 618) gives `codegen_alloc_from_pool` (2563) its original pool result;
   - `codegen_reinterpret_view` (2591) gets its original view.

   The `run_impl` handle ABI alone does not complete this. At these hooks the canonical allocator, known-bound execution and scalars take the actual source operands:
   - **Sizes:** a hook's sizes are the original sympy expressions (`get_allocation_size`, graph.py 693; `get_allocation_storage_size` 714 via `compute_required_storage_length`). The canonical allocator's bound for each is the original `ShapeEnv.bound_sympy` (symbolic_shapes.py 7167) over its `var_to_range`, never a hint or example value. An expression with no finite upper is passed as it is; owning its storage is the canonical allocator's job (132's), not a default's.
   - **Scalars:** a runtime scalar at a hook or a call is the body's own C++ value (the wrapper's `cexpr`, and its `codegen_tensor_item` locals). Nothing boxes it again: its consumer gets the expression with its bound, never a CONSTANT.

   The canonical Mesh allocation/binding operand is exposed and consumed there, keeping the original alias, reuse and result ownership. Output handles are never parsed after execution, and no wrapper allocation displaces the designated allocator. 132 pairs with these results as their fields land; no planning logic is copied.

**Consumer (66, `lower_metal.construct`), per wrapper member:** items 1–5 read directly from `produced[p].graph`, its `wrapper_code` and the retained builders, with:
- run_impl input position k bound to the member's native input value (the `Produced.inputs` entry named by `graph_inputs` key k);
- constant position idx bound to its native value (`V.graph.constants` key idx);
- output position bound to the `Produced.outputs` value;
- item 5's per-buffer operands bound to their canonical native storages, views and owners.

Control and scalar values are the emitted body's own. The per-kernel typed results (relations, accesses, Work, source owners) still go to native analysis, over item 5's operands.

**Native:**
- **131:** the body consumer and final composition, saved archive included. It compiles each action as given, links the objects into the whole program with the link operands, and calls the entry with handles over native values. Extern calls go through the original AOTI owner, with no second executor or per-op dispatcher:
  - `OSSProxyExecutor` (oss_proxy_executor.h 113, implementing the original `ProxyExecutor`), constructed as the original does, `OSSProxyExecutor(json_path, device_str, custom_objs)`, over item 4's retained serialized table;
  - the body's `aoti_torch_proxy_executor_call_function` (shim.h 627), which consumes that executor and its index;
  - the executor's lifetime is the body's, retained with it.
- **132:** the control and completion of the emitted control. It gets the typed control records (ExternCall WhileLoop/Switch/InvokeSubgraph, Inlined) and their extern table positions, plus item 5's lifetime operands.

**Unresolved (named, not worked around):**
- **U1 (68):** these result fields on `Produced`. Today only the fx_wrapper `graph` is there.
- **U2 (131):** a native call that registers a body: its sources, actions, link operands, entry, input/constant/output handle bindings and extern table. No `metal_ir.h` function exists; `metal_ir_upstream_emitted` is per-pass, with typed pointer parameters.
- **U3 (131/132, necessary actual operand):** each allocation and output handle the body makes, and each handle aliasing an input, as a canonical native Value with its owner. Paired at the producer (item 5), before print.
- **U4 (131, necessary actual operand):** each per-kernel typed record's and extern call's operands as the original buffers (the planned lines' and the call operands' own), bound to the canonical storage, not a Python-bound view. Paired at the producer (item 5), before print.
