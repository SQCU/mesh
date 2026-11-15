# Dependency patches

Style: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights).

## `torch-export-alias-metadata.patch`

Mechanism: installed torch 2.14.0 (`/Users/mdot/.venv-mesh-uv`).

- `torch/_functorch/_aot_autograd/input_output_analysis.py` `create_graph_signature` is the AOT step that both
  `aot_export_module` (`run_decompositions` and `_export_forward_backward`) and
  `torch/export/_trace.py` `_aot_export_joint_with_descriptors` call with the trace's own `fx_g` and `fw_metadata`.
  Initial public `torch.export.export` uses `_export_to_aten_ir_make_fx`; its existing `run_decompositions` then
  reaches this AOT producer.
- `torch/export/experimental/__init__.py` `_copy_graph_module_and_signature`, the deepcopy inside
  `_export_forward_backward`.

### What it changes

- **`create_graph_signature`** sets `fx_g.meta["aot_aliases"] = (fw_metadata, placeholders[num_tokens:], output
  leaves[num_tokens : num_tokens + num_mutated_inp_runtime_indices + num_outputs + num_intermediate_bases])`, where
  `num_tokens = len(fw_metadata.tokens)`.
  - That is the forward metadata (ViewAndMutationMeta, with each OutputAliasInfo and ViewMetaSequence), the graph's
    own input Nodes without tokens, and its forward output leaves without tokens or appended gradients. Scalar leaves
    are kept.
  - `ExportedProgram.__init__` carries `meta` into the program it builds.
- **`_copy_graph_module_and_signature`** reads that record with `.get`, so the source program's `meta` is untouched.
  It passes the existing deepcopy memo `{id(record): record}` to its `copy.deepcopy`, so the copy's `meta` holds the
  same tuple and metadata owner rather than a copy. It then sets the copy's record with the Nodes remapped through
  the copy's own node order (`zip` of the old and new graphs' nodes), so the copied program's record names its own
  Nodes.

`compiler/mesh_compiler/export.py` `export` pops the record from the first AOT result it produces: the
`_export_forward_backward` program with a loss, before `run_decompositions`, which drops its own; otherwise the
`run_decompositions` program. `_decompose_exported_program`'s `gm.meta.update(ep.graph_module.meta)` therefore
carries no stale record. `Captured` replays the record with `_RuntimeForwardEpilogue`.

### Target

| Item | Value |
|---|---|
| Install scope of this work | `/Users/mdot/.venv-mesh-uv`, applied by the supervisor with a fuzz-0 dry run and application |
| Torch | 2.14.0, `torch.version.git_version` `08187d9e0fba026dc8217405802ab5381dc88d90` |
| `torch/_functorch/_aot_autograd/input_output_analysis.py` | before `64da2ac0ed7c7102ce5cd18849248a5a6998cd7399380d1101f02689d0018715`, after `c22dc4ab26febeddca53f1046ef9b4e8d51a1c3de48b214d58fc8682291db24f` |
| `torch/export/experimental/__init__.py` | before `01b2d7575c126f5e53a34c481a0995637752d2c23a494ad66c411e6b44cd8d3d`, after `fbafa5a99821ef2f2cef78dab657ee7d60c48a44e881c8ef0698c803865186eb` |
| `torch/export/_trace.py` | not changed by this patch: original `28f46a16edfdecd9b7f3c1cbf388e6ca7a3c53e5d0d88ade81aefe9c42d203dd` |

The context lines are the installed source as read: `create_graph_signature` lines 492-497, and
`_copy_graph_module_and_signature` lines 28-33 and 44-49.

### Apply, check, reverse

From the repository root:

```sh
SITE=/Users/mdot/.venv-mesh-uv/lib/python3.12/site-packages
shasum -a 256 $SITE/torch/_functorch/_aot_autograd/input_output_analysis.py $SITE/torch/export/experimental/__init__.py
patch --dry-run --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-export-alias-metadata.patch
patch --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-export-alias-metadata.patch
shasum -a 256 $SITE/torch/_functorch/_aot_autograd/input_output_analysis.py $SITE/torch/export/experimental/__init__.py
```

- The first `shasum` must print the "before" hashes in the table above, and the second the "after" hashes.
- To reverse: `patch -R --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-export-alias-metadata.patch`.
- This is a pure Python source, so no rebuild is needed. The patch adds no import hook or `sitecustomize`.

### Check after applying

With `TORCH_DEVICE_BACKEND_AUTOLOAD=0`:

1. The public export plus `run_decompositions` on the noncontiguous case (`x = arange(8)[::2]`, `x.add_(y)`,
   `return x[1:]`). The returned value should alias x with stride 2 and offset 2, and the real input should be
   unchanged.
2. Then `frontends/torch/observe/fake_export.py`.
3. Then the static source observation.

The strict capture observations passed: original mutation alias/stride/offset, gradient-to-function composition,
Mapped/Scanned normalization and the fixed-result least-squares source assertion/site. The final copy observation
also confirms the source record remains identical, its metadata is shared and copied input Nodes belong to the new
graph. Evidence and exact installed hashes are in
`/Users/mdot/metal-microbench/output_data/compiler-workgroup-20261114/frontends/aot-alias-review.md`.

## `torch-clone-input-fake-mode.patch`

Mechanism: installed torch 2.14.0 (`/Users/mdot/.venv-mesh-uv`), `torch/_dynamo/utils.py` `clone_input`. `scan`'s
trace (`torch/_higher_order_ops/scan.py` `trace_scan`) calls it on each additional input while the export's
`FakeTensorMode` is active.

### What it changes

- A fake input already returns unchanged (`is_fake`). The unpatched function sends a real input, such as a captured
  tensor attribute, down its address-preserving real clone path. Under the active fake mode `torch.empty` returns a
  fake result, so `result.data_ptr()` raises.
- The patch adds one case after `is_fake`: when `torch._guards.active_fake_mode()` returns a mode, the real input is
  realized by that same mode's `from_tensor`. Its tensor and storage memo and its alias and view metadata then hold.
  No data is copied and no shape is guessed.
- With no active fake mode, the function is unchanged. `detect_fake_mode` isn't used, because it also consults an
  ambient `TracingContext` when no mode is active.

`compiler/mesh_compiler/export.py` `export` then expands the captured scan through its existing `expand` step. That
file has no workaround for this.

### Target

| Item | Value |
|---|---|
| Install scope of this work | `/Users/mdot/.venv-mesh-uv`, applied 2026-11-14 by the supervisor with fuzz 0 after a successful dry run |
| Torch | 2.14.0, `torch.version.git_version` `08187d9e0fba026dc8217405802ab5381dc88d90` |
| `torch/_dynamo/utils.py` | before `e550c68c924e1c0babaf31fc629e900c1d3b6ecbcf10aa88442d39b405b7102a`; after `fb5057fe580973d03b36217f4f3bbe7f88ac9c7d5802bc6e77597e86a4ca7413` |

The hunk's context is the installed source the supervisor read from line 2571: `clone_input`'s `is_fake(x)` early
return, then a blank line, then its nested `def is_pinned_cpu_tensor(x: torch.Tensor) -> bool:` and `return (`.

### Apply, check, reverse

From the repository root:

```sh
SITE=/Users/mdot/.venv-mesh-uv/lib/python3.12/site-packages
shasum -a 256 $SITE/torch/_dynamo/utils.py
patch --dry-run --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-clone-input-fake-mode.patch
patch --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-clone-input-fake-mode.patch
shasum -a 256 $SITE/torch/_dynamo/utils.py
```

- Record the two hashes in the table above.
- To reverse: `patch -R --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-clone-input-fake-mode.patch`.
- This is a pure Python source, so no rebuild is needed. The patch adds no import hook or `sitecustomize`.

### Check after applying

Run `/Users/mdot/metal-microbench/output_data/compiler-workgroup-20261114/frontends/torch/observe/static_source_3.py`.
Its captured-attribute scan cases (`Scanned`, `Failing`) should then export and print no HOP and no `get_attr`, with
the captured `scale` as a `CONSTANT_TENSOR`. Before the patch, both fail in `clone_input`. Then run the static caller.

The supervisor's actual source observation after application passed for the fixed map and both captured-attribute
scans: all three normalize to Core ATen with no remaining HOP or `get_attr`; parameter and constant input roles and
the scans' two-output pytree are retained. Evidence is
`/Users/mdot/metal-microbench/output_data/compiler-workgroup-20261114/frontends/torch/observe/static-source-5.txt`
and its empty `.stderr`. This records capture/normalization, not native execution.

## `torch-export-user-input-gradients.patch`

Mechanism: installed torch 2.14.0 (`/Users/mdot/.venv-mesh-uv`), `torch/export/exported_program.py`,
`_decompose_and_get_gm_with_new_signature_constants`. This is the joint-loss branch that `run_decompositions` takes
after `torch.export.experimental._export_forward_backward`.

### What it changes

That branch builds one `OutputSpec` per gradient output. It looks up the gradient's source in `specs[source]`, the
original `InputSpec` of that input.

- For a parameter, the target is `spec.target`, the parameter's FQN.
- For a user input, the unpatched line sets the target to `source`. That is the traced name (for example `arg2_1`),
  which is not the program's input name.
- The patch sets it to `spec.arg.name`, the original input's name (for example `x`). The same function renames the new
  placeholders back to those original names. So `GRADIENT_TO_USER_INPUT`'s target becomes the input placeholder's name,
  and `backward_signature.gradients_to_user_inputs` names the program's own input.

`compiler/mesh_compiler/export.py` (`read`, `returned`, `capture`) keys `TrainingResult.inputs` by that target. It has
no name parser and no fallback.

### Target

| Item | Value |
|---|---|
| Install scope of this work | `/Users/mdot/.venv-mesh-uv`, applied 2026-11-14 by the supervisor with fuzz 0, after inspecting the hunk's context in the installed file |
| Torch | 2.14.0, `torch.version.git_version` `08187d9e0fba026dc8217405802ab5381dc88d90` |
| `torch/export/exported_program.py` | before `bb4908767a26f84fbdc0d84a3413d4818cf7f88e34ae450a29f537f8fa165fff`, after `bf8418070373473faddea7a535962c2e9626a9493f98d67a0c7d1c17ed56d691` |

### Apply, check, reverse

From the repository root:

```sh
SITE=/Users/mdot/.venv-mesh-uv/lib/python3.12/site-packages
shasum -a 256 $SITE/torch/export/exported_program.py
patch --dry-run --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-export-user-input-gradients.patch
patch --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-export-user-input-gradients.patch
shasum -a 256 $SITE/torch/export/exported_program.py
```

- The first `shasum` must print the "before" hash in the table below, and the second the "after" hash.
- To reverse: `patch -R --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-export-user-input-gradients.patch`.
- This is a pure Python source, so no rebuild is needed. The patch adds no import hook or `sitecustomize`.

### Check after applying

Run `/Users/mdot/metal-microbench/output_data/compiler-workgroup-20261114/frontends/torch/observe/fake_export.py`. Its joint-loss case prints a
`GRADIENT_TO_USER_INPUT` role whose target is the module's input name, and a `TrainingResult` `out_spec` keyed by it.

## `torch-inductor-alias-mutations.patch`

Mechanism: installed torch 2.14.0 (`torch.version.git_version` `08187d9e0fba026dc8217405802ab5381dc88d90`) Inductor
owners.

- `torch/_inductor/graph.py`: `GraphLowering.mark_buffer_mutated` and `GraphLowering.get_buffer`.
- `torch/_inductor/ir.py`:
  - `MutationLayoutSHOULDREMOVE.realize_into`, `MutationOutput`, `ComputedBuffer` and `ExternKernel.get_outputs`;
  - `FallbackKernel.__init__`, `get_inputs_that_alias_output`, `create` and `codegen_outputs`.
- `torch/_inductor/scheduler.py`: `Scheduler.compute_dependencies` (alias user-list merge, mutation renames,
  graph-input `OutputNode`) and `dead_node_elimination`. These consume the outputs and are unchanged.

Assignment: design/compiler-native-assignment-2026-11-13.md. Evidence (not authority): in
output_data/compiler-workgroup-20261113/executor/, `native.status.md` sections 0 and P, and
`native-applied-plain.log` and `native-applied-plain-read16.log`.

### What it changes

In Inductor, a mutation of a typed alias of a buffer becomes a mutation of the buffer and of its other typed aliases.
An example of a typed alias is the `view.dtype` fallback of a `uint8` arena input.

- `GraphLowering` retains each alias once, when `FallbackKernel` records it: `register_alias`, `alias_family`.
  - `FallbackKernel.register_alias_facts()` runs once the kernel's alias facts are complete: at the end of
    `__init__`, and at the end of the `mutates_and_returns_first_arg` branch.
  - For a `Layout` output, it registers exactly the names that the existing
    `FallbackKernel.get_inputs_that_alias_output()` returns. That method already excludes the fresh outputs of
    auto-functionalizable mutable ops.
  - A void effect kernel (`NoneLayout`) registers no alias, so its independent mutated arguments are not unioned.
- `GraphLowering.mark_buffer_mutated` also realizes the readers of the mutated buffer's alias family. With no
  registered aliases, the body is unchanged.
- `MutationLayoutSHOULDREMOVE.realize_into` gives the store's `ComputedBuffer` one `AliasMutationOutput` for each
  family member.
  - Each keeps the family member as its mutation name.
  - Each returns the physical store buffer from `get_inputs_that_alias_output()`.
  - The store keeps its typed target and codegen name.
- Mutating ExternKernels expose dependency-only `AliasMutationOutput`s in a separate `alias_mutation_outputs` list.
  - `mutation_outputs`, `codegen_outputs()` and the C ABI are unchanged.
  - The owners are `FallbackKernel` (through `register_alias_facts()`), `InplaceBernoulliFallback`,
    `InplaceCopyFallback`, `ResizeStorageBytes`, `ScatterFallback` and `IndexPutFallback`. Each one calls
    `add_alias_mutation_outputs()` after it has registered its mutation.
- `FallbackKernel.create` marks a written schema operand mutated before `process_kernel` realizes the arguments, when
  that operand has an alias family. Tensor-list and optional operands are flattened with `pytree.tree_leaves`.

The existing scheduler consumes these outputs (`compute_dependencies`, `dead_node_elimination`). The patch adds no
keepalive, self dependence or synthetic output, and it does not disable DCE.

### Target

| Item | Value |
|---|---|
| Install scope of this work | `/Users/mdot/.venv-mesh-uv`, applied 2026-11-13 by the supervisor after Codex and compiler review |
| Torch | 2.14.0, `torch.version.git_version` `08187d9e0fba026dc8217405802ab5381dc88d90` |
| `torch/_inductor/graph.py` | before `dfc3ad98319647ee53f2ed18a712326090fc0b38839d57c9a48acc48d129047f`, after `efffd6b54ed0e67ad1f41c840957780fae092708adf460ecc16914dcaaba27fe` |
| `torch/_inductor/ir.py` | before `8cba93adeeb9cf2c8d4f1ff85a8c214be98908947128904ef01cc963aca7878f`, after `5d272a6e49cd9458745c2804942fc903450a5888d15dce0e11c06ffe516e95a0` |

The patch applies to any torch installation whose two files have the "before" hashes. These are pure Python sources,
so no C++ rebuild of Torch is needed. The patch adds no import hook, install hook or `sitecustomize`. It takes effect
only when it is applied with the command below.

### Apply, check, reverse

From the repository root:

```sh
SITE=/Users/mdot/.venv-mesh-uv/lib/python3.12/site-packages
shasum -a 256 $SITE/torch/_inductor/graph.py $SITE/torch/_inductor/ir.py
patch --dry-run --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-alias-mutations.patch
patch --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-alias-mutations.patch
shasum -a 256 $SITE/torch/_inductor/graph.py $SITE/torch/_inductor/ir.py
```

- The first `shasum` must print the "before" hashes above. The second must print the "after" hashes.
- To reverse: `patch -R --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-alias-mutations.patch`.
  The hashes then return to "before".

### Check after applying

Run the existing proof in plain mode, which installs no scratch hook. It uses the repository's
`rdma/mesh_aoti.py` and `rdma/libmesh-aoti.dylib`. Run it in a fresh process. Torch's cache key hashes its own
sources, so no cache override is needed:

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /Users/mdot/.venv-mesh-uv/bin/python -I \
  output_data/compiler-workgroup-20261113/executor/native-proof-scripts/alias_mutation_proof.py plain
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /Users/mdot/.venv-mesh-uv/bin/python -I \
  output_data/compiler-workgroup-20261113/executor/native-proof-scripts/alias_mutation_proof.py plain read16
```

Each must print `True` for every check at invocations 0, 1 and 2. Then run the canonical body and client routes.

The 2026-11-13 run, after the supervisor applied the patch, printed `True` for every check in both commands. The
logs are `native-applied-plain.log` and `native-applied-plain-read16.log` in
output_data/compiler-workgroup-20261113/executor/.

### Not covered (follow-ups for review)

- Alias facts of multi-output fallbacks (`MultiOutputLayout`), and of ExternKernels other than `FallbackKernel`.

- Mutation owners other than those listed above:
  - `TemplateBuffer` `mutated_inputs`;
  - `mkldnn_ir` and `comm_lowering` kernels;
  - control flow (`Conditional`, `WhileLoop`);
  - `ExternKernelOut`/`ExternKernelMultiOut`, created from `FallbackKernel.create` for functional custom ops with an
    out variant.
- An alias registered after its target was itself registered as a root.

## `torch-lstsq-fixed-results.patch`

Mechanism: installed torch 2.14.0 (`/Users/mdot/.venv-mesh-uv`), strict export through Dynamo. Five hunks in four
files.

- **Producer:** `torch/_dynamo/variables/torch.py` adds `handle_linalg_lstsq` to `TorchInGraphFunctionVariable._get_handlers`.
  - The call is bound by `torch.fx.operator_schemas.normalize_function` (torch.linalg.lstsq's own schema). A call that
    doesn't bind keeps torch's validation on the generic path.
  - The call's own fake (`operations._lstsq_metadata`) decides the case. Where it raises `DynamicOutputShapeException`,
    the residuals' shape is the data's: gelsd or gelss with m > n > 0, unbatched with nrhs > 0 or with a nonempty batch.
    Every other call, its residuals static (an empty batch, or unbatched with nrhs 0), is the generic path's complete
    product.
  - For the data's case it makes one product node `torch.linalg.lstsq(input, b, rcond, driver=…)`, marked
    `meta["lstsq_fixed"]`, plus a getitem per result slot.
  - Slots 0, 2 and 3 carry the metadata of `mesh::linalg_lstsq_fixed`'s fake. Slot 1 (residuals) is a `TensorVariable`
    with dtype, device and class only, and its shape unresolved.
- **Unresolved shape:** `torch/_dynamo/variables/tensor.py` makes `dynamic_getattr` and `tp_getattro_impl` read
  `example_value` with `.get`, so that tracker's attributes resolve from its own facts. A data use of it recomputes the
  product's fake (`get_fake_values_from_nodes`), which raises Dynamo's "Dynamic shape operator", the existing
  requirement.
- **Mark handoff:** `torch/_dynamo/functional_export.py` `DynamoGraphTransformer.run_node` adds `lstsq_fixed` to its
  copied metadata keys (line 412), so the mark reaches strict export's graph.
- **Selection:** in `torch/export/_trace.py` `_strict_export`, after `_export_to_torch_ir`, each marked product whose
  slot-1 getitem has no users (or no longer exists) becomes `mesh::linalg_lstsq_fixed`. The slot comes from each
  getitem's own `args[1]`, and getitems 0, 2 and 3 are re-indexed to 0, 1 and 2. A product with any other user is left
  as it is.

The operator, its fake, its autograd (solution only; rank and singular values marked non-differentiable) and its
lowering are `compiler/mesh_compiler/operations.py`'s (`LSTSQ_FIXED`).

### Target

| Item | Value |
|---|---|
| Install scope | `/Users/mdot/.venv-mesh-uv`, supervisor-applied 2026-11-14 after frontend's observation hold; fuzz 0 dry-run/application and four-file AST passed |
| Torch | 2.14.0, `torch.version.git_version` `08187d9e0fba026dc8217405802ab5381dc88d90` |
| `torch/_dynamo/variables/tensor.py` | before `0ce323ae582d1268cbaf318a366ae4224e2eb148740bbd55e136190a0119e443`, after `2505c17558e511a1e8296fb0869b90e37e9bb74fa34116dce710dca606612ec1` |
| `torch/_dynamo/variables/torch.py` | before `28e9fce5a40ee7038f1ee8517d73781655849da99a02a560ae8c71ea39042995`, after `cd924d13538aef3d6f1814d160ddce610612a135e87109cc0fbd4188c5df5878` |
| `torch/_dynamo/functional_export.py` | before `646aeca84acd49def77dca8f1bc0fe0ef1fedc07641182f7c65fe8d37a9ab46e`, after `aa2e63d83e8843cb1329070a55ae9398192f66d5c0f3f65a7577434977e06bda` |
| `torch/export/_trace.py` | before `28f46a16edfdecd9b7f3c1cbf388e6ca7a3c53e5d0d88ade81aefe9c42d203dd`, after `82a87c00dfd323b23ac8f95a0f8d5673e49a25dbd03f9d0bebef14a1a4d9c6a0` (this patch alone) |

- `patch --dry-run --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-lstsq-fixed-results.patch`
  passed against the recorded before files, followed by application and AST checks. Exact command and hashes are in
  `output_data/compiler-workgroup-20261114/compiler/lstsq-fixed-install.json`.
- The frontend's `torch-export-alias-metadata.patch` no longer changes `torch/export/_trace.py`; its earlier hunk there
  was reversed. The `_trace.py` "after" hash above is for this patch alone.
- The connected ordinary capture of `compiler/lstsq-fixed-command.md` now passes with the paired alias consumer and
  strict default: the fixed three-result product, original source assertion and original draw site survive. Source
  evidence is `/Users/mdot/metal-microbench/output_data/compiler-workgroup-20261114/frontends/torch/observe/aot-alias-strict-contracts.json`.
  This is capture evidence; numerical execution follows the matching native release.

## `torch-inductor-mps-kernel-operands.patch`

Mechanism: installed torch 2.14.0 (`/Users/mdot/.venv-mesh-uv`), `torch/_inductor/codegen/mps.py` `MetalKernel`.
Installed source-only with fuzz 0; AST passed. The before/after SHA256 values are
`d9734c7abceef021a73ab5ba02862d83907b142fac0f0aca3bb902cabfa4f774` and
`0266b4cd4723ba7f7041e7cad1b4bfca577ca904c852b5d6692a1dc7d4c37ef1`.
The exact manifest is `output_data/compiler-workgroup-20261114/compiler/numerical64-mps-producer-install.json`.
This records the producer interface; module/native consumer replacement and execution remain separate work.

### What it changes

- **`codegen_kernel`** builds the Metal source in its own `IndentedBuffer` and keeps it as `self.metal_source`. The
  existing wrapper (`compile_mps_shader('''...''')`, or the C++ raw string) then encloses that same value, so the
  wrapper's text is unchanged. It keeps the source's own entry function name, its `name` argument, as
  `self.metal_entry`. That is apart from the launch name `MetalScheduling.define_kernel` assigns
  (`generated_kernel_N`, which only the wrapper header's copy of the source takes).
- **`call_kernel`** produces the launch once as values. `grid` and `group` are the active range trees' numels (a
  reduction tree's capped at `max_threadgroup_size`; the threadgroup only when the kernel reduces), and its existing
  formatting prints them. It keeps `self.launch = (name, values, types, grid, group, error)`:
  - `values` and `types` are the ordered arguments before the grid entries: output then input buffer names (less
    `removed_buffers`), the sizevars and the dynamic numels, each with its `python_argdefs` type. A dynamic numel is
    its `tree.numel` Symbol, or else the `SymbolicCallArg` that `generate_numel_expr` returns (its `inner` symbol and
    its `inner_expr`); its formatted argument is `str(arg)`, which is its `inner`, as before;
  - `error` is the error buffer's argument index, or None. It is the count of those ordered typed arguments, each one
    positional, and the formatted `error_buf_idx=` argument prints it.
  It appends the kernel to `wrapper.launched_kernels` in call order.

## `torch-inductor-upstream-composition.patch`

Mechanism: installed torch 2.14.0 (`/Users/mdot/.venv-mesh-uv`), on top of `torch-inductor-mps-kernel-operands.patch`
as installed (`mps.py` before is that patch's after). It changes three files:
- `torch/_inductor/codegen/mps.py` `MetalKernel`;
- `torch/_inductor/codegen/cpp.py` `CppScheduling.define_kernel`, `CppScheduling.defined_source` and
  `KernelGroup.codegen_group`;
- `torch/_inductor/graph.py` `GraphLowering.make_subgraph`.

These are the upstream hooks `compiler/mesh_compiler/lower_inductor.py` takes the producer's results through: its
terminal emission, the CPU final source and the control bodies' backends. The patch was prepared from the installed
source as read.

Installed. The supervisor checked the "before" hashes below, then ran a fuzz-0 dry run and application of the
committed patch (`a08d2ded…`), a fuzz-0 reverse dry run and an AST check. All passed, and the installed files have the
"after" hashes below. The manifest is `output_data/compiler-workgroup-20261114/compiler/numerical68-upstream-install.json`, step 1. It also records that the hash comparison stopped at first: the cpp.py
"after" hash recorded here earlier, `c02487be…`, predated the committed patch's `codegen_group` composition hunk and
was stale. The applied patch is the committed one, unchanged.

Separately, reversing the committed patch on a copy of the installed files gives the "before" hashes, and applying it
again gives the "after" hashes, which parse.

### What it changes

- **`MetalKernel.codegen_kernel(name, composition=None)`**: with `composition` None, its output is unchanged. Otherwise
  the same emitter composes its terminal emission from these parts, each as given:
  - `composition.source` goes after the headers.
  - Each argument's declaration, in the launch's order (`call_kernel` `values`, then the error buffer), is written
    as `composition.argument(index, declaration)`.
  - `composition.parameters` follow the arguments, and `composition.builtins` replace the thread position
    parameters.
  - `composition.enter` statements open the body. The upstream thread position names (`thread_pos` or the one
    tree's, and `group_pos`) are then bound from the `composition.grid` and `composition.group` uint3 expressions.
  - The body runs within the launch's grid: each non-reduction tree's numel, which a partial last threadgroup's
    further threads do not run.
  - `composition.exit` statements follow the body.
- **`MetalKernel.store` `atomic_add`**: the address is the argument's own subscript `&var[index]` at offset 0, as every
  other load and store takes it. The value it adds is unchanged.
- **`CppScheduling.define_kernel`** keeps each defined kernel's declared entry and final source (after its
  placeholder substitutions) as `wrapper.defined_kernels[kernel_name] = (kernel_decl_name, src_code)`.
- **`CppScheduling.defined_source(src_code, kernel_decl_name, kernel_name)`**: a new static method. It holds
  `define_kernel`'s own placeholder substitutions, moved there unchanged. `define_kernel` calls it, and so does a
  composed pass of the same group.
- **`KernelGroup.codegen_group(name=None, composition=None)`**: with `composition` None, its output is unchanged.
  Otherwise:
  - `composition.source` goes after the `cpp_prefix.h` include;
  - `composition.parameters` follow the group's `cpp_argdefs` arguments;
  - `composition.enter` statements open the body.
- **`GraphLowering.make_subgraph`** makes `self.subgraph_lowering(...)`, with the same arguments as before.
  `GraphLowering.subgraph_lowering = SubgraphLowering` follows `SubgraphLowering`, so the default is unchanged. A
  subclass sets its own, so control bodies (`Conditional`, `WhileLoop`, `InvokeSubgraph`) are scheduled by its
  backends.

### Target

| Item | Value |
|---|---|
| Torch | 2.14.0, `torch.version.git_version` `08187d9e0fba026dc8217405802ab5381dc88d90` |
| `torch/_inductor/codegen/mps.py` | before `0266b4cd4723ba7f7041e7cad1b4bfca577ca904c852b5d6692a1dc7d4c37ef1`, after `8bbd0105926cd2e738852a9cb25962ca0745526577ad8da8dd709b894c8884db` |
| `torch/_inductor/codegen/cpp.py` | before `885b73279e8686181d3706c6d6a590d07102c44ea739e8c11bb5be5ef758292f`, after `74b2de2698fcb01ced737ca10238d16f0721a6e1271314fbec65170e66451934` |
| `torch/_inductor/graph.py` | before `efffd6b54ed0e67ad1f41c840957780fae092708adf460ecc16914dcaaba27fe`, after `1b3e2fdce21f6c499b655d4e4223fec37147decc7b3d95237b7bbd542cd23591` (installed `graph.py` is now `torch-inductor-aot-lowered-result.patch`'s after) |
| Install scope of this work | `/Users/mdot/.venv-mesh-uv`, applied by the supervisor at source revision `ba03a780`: fuzz-0 dry run, application, reverse dry run and AST check passed (`output_data/compiler-workgroup-20261114/compiler/numerical68-upstream-install.json`, step 1) |

### Apply, check, reverse

From the repository root:

```sh
SITE=/Users/mdot/.venv-mesh-uv/lib/python3.12/site-packages
F="$SITE/torch/_inductor/codegen/mps.py $SITE/torch/_inductor/codegen/cpp.py $SITE/torch/_inductor/graph.py"
shasum -a 256 $F
patch --dry-run --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-upstream-composition.patch
patch --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-upstream-composition.patch
shasum -a 256 $F
```

- The first `shasum` must print the "before" hashes in the table above, and the second the "after" hashes.
- To reverse: `patch -R --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-upstream-composition.patch`.
- This is pure Python source, so no rebuild is needed.

## `torch-inductor-cpu-kernel-results.patch`

Mechanism: installed torch 2.14.0 (`/Users/mdot/.venv-mesh-uv`), `torch/_inductor/codegen/cpp.py`, on top of
`torch-inductor-upstream-composition.patch` as installed (this patch's "before" is that patch's `cpp.py` "after"). It
changes two places, `CppKernel.codegen_conditions` and `KernelGroup.codegen_group`.

These are the CPU results `compiler/mesh_compiler/lower_inductor.py` takes from the original's own producers. The first
is each kernel's own loads and stores: `CppScheduling.kernel_proxy_cls` and `CppKernelProxy.kernel_cls`,
`vec_kernel_cls` and `tile2d_kernel_cls` make the kernels, and the lowerer's `MeshCppKernelProxy` and its kernels
subclass them. The second is the group definition's parameter declarations.

Installed from source checkpoint `205e2c94`. The supervisor verified the recorded before hash, applied the committed
patch with fuzz 0 and backups disabled, and verified the after hash, reverse dry run and AST. The source-only
realization is step 4 of `output_data/compiler-workgroup-20261114/compiler/numerical68-upstream-install.json`;
it did not import Torch or Mesh or execute a workload.

### What it changes

- **`CppKernel.codegen_conditions`**: a kernel's tail condition is narrowed for a scalar kernel. That kernel was
  identified as `type(self) is CppKernel` and is now `not isinstance(self, CppVecKernel)`.
  - Its callers are `CppKernelProxy.gen_body` and `aggregate_reduction_buffers` (installed lines 4974, 5017 and 5023).
    Each calls it only on its proxy's own kernels, which are the scalar, vector and tile kernels its factory made.
  - For upstream's own classes, both tests select the same kernels, so the output is unchanged.
  - For a factory subclass of the scalar kernel, upstream's comment at `CppKernelProxy` names subclassing
    `CppKernel` as the extension point. Without this hunk, the identity test would miss that subclass and change
    the emitted tail conditions.
- **`KernelGroup.codegen_group`** keeps the definition's own parameter declarations as made:
  `self.arg_defs = tuple(arg_defs)`. These are `KernelArgs.cpp_argdefs`'s, taken before any composition's
  parameters.
  - `flush` calls `codegen_group` before `call_kernel`. The lowerer's call record (`MeshKernelGroup.call_kernel`)
    takes these as its `arg_defs`.
  - It takes `call_args` and `arg_types` from the `KernelCallLine` that the call wrote. It no longer calls
    `cpp_argdefs` a second time.
  - A composed pass (`codegen_group(name, composition)`) keeps the same group's declarations again.
  - The returned source is unchanged.

### Target

| Item | Value |
|---|---|
| Torch | 2.14.0, `torch.version.git_version` `08187d9e0fba026dc8217405802ab5381dc88d90` |
| `torch/_inductor/codegen/cpp.py` | before `74b2de2698fcb01ced737ca10238d16f0721a6e1271314fbec65170e66451934` (`torch-inductor-upstream-composition.patch`'s after), after `0eb0c1e9348b6b03db3e81d5158e45468be7d1554c6444864e3508fa26b5c7c3` |
| Install scope of this work | installed source only; numerical68-upstream-install.json step 4 |

### Apply, check, reverse

From the repository root:

```sh
SITE=/Users/mdot/.venv-mesh-uv/lib/python3.12/site-packages
F=$SITE/torch/_inductor/codegen/cpp.py
shasum -a 256 $F
patch --dry-run --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-cpu-kernel-results.patch
patch --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-cpu-kernel-results.patch
shasum -a 256 $F
```

- The first `shasum` must print the "before" hash in the table above, and the second the "after" hash.
- To reverse: `patch -R --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-cpu-kernel-results.patch`.
- This is pure Python source, so no rebuild is needed.

## `torch-inductor-graph-lowering-factory.patch`

Mechanism: installed torch 2.14.0 (`/Users/mdot/.venv-mesh-uv`), `torch/_inductor/compile_fx.py`
`_InProcessFxCompile.codegen_and_compile`. That is the original compiler's own GraphLowering construction (installed
lines 1625-1649): `compile_fx` reaches it through AOT autograd, `compile_fx_forward`, `compile_fx_inner` and
`fx_codegen_and_compile`, after the compiler's own pre-grad, functionalization, joint and post-grad passes. It is
independent of the other Inductor patches here.

### What it changes

- **`_CompileFxKwargs`** gets one more entry: `graph_lowering: Callable[..., GraphLowering]`.
- **`codegen_and_compile`** reads `graph_lowering = graph_kwargs.get("graph_lowering", GraphLowering)` in the same
  way it reads `get_decomp_fn`. It then makes the forward graph with `graph = graph_lowering(gm, ...)`, using the same
  arguments as before. With no entry, the factory is `GraphLowering` and the output is unchanged.
- Nothing else changes. The constant graph (`const_graph`), `graph.run`, the output strides and all three output forms
  (`fx_wrapper` module, `AotCodeCompiler`, `compile_to_module`) are untouched.

`compiler/mesh_compiler/lower_inductor.py` `_inductor` passes the factory as a compilation operand:
`compile_fx_aot(body, inputs, inner_compile=partial(compile_fx_inner, graph_lowering=MeshLowering), config_patches=...)`
under `torch.no_grad()`. `MeshLowering` is a `GraphLowering` subclass. Its `_update_scheduler` makes a `MeshScheduler`,
whose `create_backend` selects the Mesh backends, and its `subgraph_lowering` (`torch-inductor-upstream-composition.patch`)
gives every control body the same backends. Nothing is registered with `register_backend_for_device`, so other
compilations in the process keep upstream's backends. The GraphLowering is returned by
`torch-inductor-aot-lowered-result.patch`, not kept from inside the factory: no closure, no `TracingContext`, no
`force_disable_caches` (`compile_fx_inner` has no FX cache in `aot_mode`).

### Target

| Item | Value |
|---|---|
| Torch | 2.14.0, `torch.version.git_version` `08187d9e0fba026dc8217405802ab5381dc88d90` |
| `torch/_inductor/compile_fx.py` | before `4e5b593e018c6bd6f69062483d524b64820ab272351fbf073b7ec118a66598f8`, after `f4cc8a8d3ce142c67e8c43fb47434cd16ff1c7ca2665f92ec590754d10dd5592` (installed `compile_fx.py` is now `torch-inductor-aot-lowered-result.patch`'s after) |
| Install scope of this work | `/Users/mdot/.venv-mesh-uv`, applied by the supervisor at source revision `ba03a780` after the composition patch: fuzz-0 dry run, application and AST check passed (`output_data/compiler-workgroup-20261114/compiler/numerical68-upstream-install.json`, step 2) |

The patch was written from the installed source's context lines 842-847, 1402-1407 and 1622-1628. Installed: the
supervisor's fuzz-0 dry run and application of the committed patch (`bb0f187b…`) passed, the result has the "after"
hash, and it parses (`output_data/compiler-workgroup-20261114/compiler/numerical68-upstream-install.json`, step 2).

### Apply, check, reverse

From the repository root:

```sh
SITE=/Users/mdot/.venv-mesh-uv/lib/python3.12/site-packages
shasum -a 256 $SITE/torch/_inductor/compile_fx.py
patch --dry-run --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-graph-lowering-factory.patch
patch --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-graph-lowering-factory.patch
shasum -a 256 $SITE/torch/_inductor/compile_fx.py
```

- The first `shasum` must print the "before" hash above, and the second the "after" hash.
- To reverse: `patch -R --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-graph-lowering-factory.patch`.
- This is pure Python source, so no rebuild is needed. The patch adds no import hook or `sitecustomize`.

## `torch-inductor-aot-lowered-result.patch`

Mechanism: installed torch 2.14.0 (`/Users/mdot/.venv-mesh-uv`), on top of
`torch-inductor-mps-kernel-operands.patch` (installed), `torch-inductor-upstream-composition.patch`,
`torch-inductor-graph-lowering-factory.patch` and `torch-inductor-cpu-kernel-results.patch`, in that order.
- `graph.py` and `mps.py` before are the composition patch's after.
- `compile_fx.py` before is the factory patch's after.
- `cpp.py` before is the cpu-kernel-results patch's after.
- `cpp_template_kernel.py`, `common.py` and `lowering.py` before are upstream's.

It is the original AOT compilation's lowered result, taken at compile_fx's original `graph.aot_mode` branch
(`cpp_wrapper`) at its codegen's planned stage: after the wrapper's IR passes (its memory plan), before any line
prints. `compile_fx_aot` returns the `LoweredAOTI` itself:
- `graph`: its GraphLowering. Its scheduler codegen has run, and its own wrapper_code (CppWrapperCpu or CppWrapperMps)
  holds its planned lines and its planner's result owner (`memory_planner` or `past_planning_states`). Nothing is
  printed.
- `planned`: those planned lines (`wrapper_code.lines`).
- `resume`, run by `emit(supplied)`: the same generation resumed with the supplied operands (`wrapper_code.supplied`),
  its print and section assembly, then the branch's own emission. That emission is the original logs, the extern
  node serializer and `AotCodeCompiler.compile`. It runs under the config and virtualized handlers as they were at
  the stage.
- `compiled`: that emission's result, `AotCodeCompiler.compile`'s `AotCompiled`
  (`torch-inductor-cpp-build-result.patch`), or None before `emit`.

A const graph (`use_runtime_constant_folding`) is not split. Its emitted code is what its main graph's GraphLowering
takes, so it emits at once.

The revision after `6d56f939` replaces the fx-wrapper return (`GraphLowering.codegen` after `FxConverter.generate`).
That return is not removed by deletion alone: this patch's `wrapper_fxir.py` hunks remain, unused under `cpp_wrapper`,
and are listed as an open item. `compiler/mesh_compiler/lower_inductor.py` `_inductor` takes the producer's
result through it. Fields: `output_data/compiler-workgroup-20261114/compiler/numerical68-result.md`.

### What it changes

- **`config.aot_inductor.lowered: bool = False`**: the operand that asks for the lowered result. With it False, every
  output is unchanged.
- **`codegen/wrapper.py` `PythonWrapperCodegen`**, the planned stage at the original owner:
  - `_generate`'s original body is now `_generation`, which yields `self.lines` once, right after
    `run_wrapper_ir_passes` and before `set_writeline` and the line `codegen` loop. `_generate` starts it.
  - Under `lowered` (and not for a const graph), `_generate` returns that planned list. Otherwise it resumes at once,
    so its output is unchanged.
  - `resume_generation(supplied)` sets `self.supplied` and resumes the print and section assembly. It returns their
    result.
  - `memory_plan` keeps its original planner instance as `self.memory_planner`: its buffer groups, allocations and live
    ranges. Each `PoolMemoryPlanningLine.group` is one of its groups.
  - `memory_plan_reuse` keeps its own `past_planning_states` as `self.past_planning_states`: each scope's
    `MemoryPlanningState` in exit order, the root's last.
  - `run_wrapper_ir_passes` sets both to None first, so exactly one of them is set.
  - `CppWrapperCpu.generate` (`write_wrapper_decl`, then `super().generate`) and the GPU/MPS overrides pass through
    unchanged.
- **`GraphLowering.codegen`**: under `lowered` (not a const graph), it returns the wrapper's planned stage without
  popping its codegened graph. `resumed_codegen(supplied)` resumes the wrapper's print, pops the graph and returns the
  code. For CPU and MPS, `codegen_with_cpp_wrapper` is `return self.codegen()`.
- **`output_code.py` `LoweredAOTI(OutputCode)`**: a new dataclass with fields `graph`, `device_type`, `planned`,
  `resume` and `compiled` (None until set), and `emit(supplied)`, which sets and returns `compiled`. Its `post_compile`
  and `set_triton_bundle` do nothing, because there is no emitted callable to load.
- **`compile_fx.py`**:
  - `codegen_and_compile`'s original `graph.aot_mode` branch keeps its emission after codegen, moved unchanged into
    `aot_compiled(wrapper_code, kernel_code)`: its logs, the extern node serializer, then `AotCodeCompiler.compile`.
    Without `lowered` it runs at once on `codegen_with_cpp_wrapper()`'s result.
  - Under `lowered`:
    - `codegen_with_cpp_wrapper()` gives the planned stage.
    - The branch captures `config.get_config_copy()` and `dict(vars(virtualized.threadlocal))`.
    - Its `resumed(supplied)` runs `aot_compiled(*graph.resumed_codegen(supplied))` under `config.patch` of that copy
      and `_virtualized_as` of those handlers. `_virtualized_as` is new: it sets this thread's V handlers and restores
      them after.
  - Its `V.aot_compilation` return is
    `LoweredAOTI(graph=graph, device_type=graph.device_type, planned=planned, resume=compiled_fn)`.
  - `_compile_fx_main` keeps the AOT export's own signature: `unlifted_gm.meta["graph_signature"] = graph_signature`.
  - `compile_fx_aot` returns the `LoweredAOTI` itself. Its annotation adds `LoweredAOTI`.
- **`codegen/wrapper_fxir.py` `WrapperFxCodegen`**:
  - `lowered_subgraphs: dict` is set in `__init__`.
  - `define_subgraph_launcher_fn`, under `lowered`, keeps two things by name:
    - each lifted control body's own lowered `SubgraphLowering`, in `lowered_subgraphs`;
    - its converted graph, `subgraph_code.wrapper_code.fx_converter.gm`, in `subgms`. The converter's
      `_generate_subgm_getattrs` reads `subgms`.
  - `_generate` keeps its converter as `self.fx_converter` and then calls `generate()`.
  - `codegen_while_loop(while_loop, stack_output)` and `codegen_invoke_subgraph(invoke_subgraph)` follow
    `codegen_switch`: one `WhileLoopLine` or `InvokeSubgraphLine`, then `codegen_subgraph_common` for each body (a
    while loop's cond, then its body). Upstream's form writes `str` scaffolding.
- **`codegen/wrapper_fxir.py` `FxConverter`**:
  - New fields:
    - `line_nodes: list[tuple[WrapperLine, tuple[fx.Node, ...]]]`: each converted line in line order, with the nodes
      its own conversion created, in graph order. A free, comment or null line has `()`.
    - `definitions: dict[str, KernelDefinitionLine]`: each selected C++ kernel's definition, by its name.
  - Under `lowered`, `_generate_kernel_definition` records a `gpu=False` definition (CPU C++, MPS Metal) in
    `definitions` and returns. Triton definitions import as before. Without `lowered`, every definition takes
    upstream's path.
  - Under `lowered`, `_generate_kernel_call` sends a `triton=False` call to `_generate_selected_kernel_call`. That
    writes one node, `call_function(selected_kernel_call, args=(kernel_name, *operands), kwargs=call_kwargs)`.
    Without `lowered`, a `triton=False` call keeps upstream's `NotImplementedError("FX conversion only supports Triton
    kernels.")`. So an ordinary `fx_wrapper` `compile_graph` (`gm.forward`) never holds the intrinsic.
    - Each operand is mapped as a Triton call's: `_lookup_args`, then `_generate_sym_node`, with `sympy.floor` as
      `FloorDiv`.
    - `selected_kernel_call` is the compile-time intrinsic `SelectedKernelCall(HigherOrderOperator)`,
      `torch.ops.higher_order.selected_kernel_call(kernel_name, *operands, **call_kwargs) -> None`. It is registered
      the way the original value-less effect HOP `_higher_order_ops/print.py` `print` is, minus print's runtime
      implementation:
      - `register_fake(selected_kernel_call, skip_cache=True)` returns `None`. `_node_metadata_hook` runs the target
        under `V.fake_mode` with the Python dispatcher, so the node's `meta["val"]` is `None` and nothing executes.
      - `fallthrough(AutogradCPU)` and `fallthrough(AutogradMPS)`, the autograd keys of its fake operands (print:
        CPU and CUDA). Dispatch then reaches the Python key and `FakeTensorMode`.
      - `has_side_effect(selected_kernel_call)`. `torch._library.utils.is_impure` reads `_side_effectful_functions`
        for a HOP, so `Node.is_impure` holds and dead code elimination keeps the node.
      - No backend or `CompositeExplicitAutograd` kernel. The lowered result's consumer compiles and calls the
        kernel. Run as a module, the HOP's dispatch raises `could not find kernel for HigherOrderOperator
        selected_kernel_call`; it never returns a silent no-op.
    - Its effect is its writes to its operand buffers: on CPU `KernelArgs`' in-place and output buffers, on MPS its
      output buffers.
  - `_generate_while_loop` writes `torch.ops.higher_order.while_loop(cond, body, carried, additional,
    mutated_arg_indices=...)`, or `while_loop_stack_output(cond, body, carried, additional)`.
  - `_generate_invoke_subgraph` writes `torch.ops.higher_order.invoke_subgraph(subgraph, identifier, *inputs)`, the
    identifier being `InvokeSubgraph.identifier` (below).
  - Both control generators record the result under the ir node's name, as `_generate_switch` does.
  - `_generate_extern_kernel_multi_out` calls `_generate_extern_kernel_common` for the out variant. Each
    `out_variant_output_nodes` buffer is passed as its own `out_arg_names` keyword, and the result takes the node's name.
  - `_generate_extern_kernel_common` takes `outs=()` for this.
  - `_generate_extern_kernel_common` keeps upstream's `buffer_to_node`: a mutated input keeps its node and is not
    rebound. Its mutation results are the ir node's own (`ExternKernel.get_outputs()`: `mutation_outputs` and
    `alias_mutation_outputs`, each `MutationOutput`'s `mutation_names`, `get_defining_op()` and
    `get_mutation_buffers()`, `AliasMutationOutput.physical`). They are reached through `line_nodes` (`line.node`).
  - `_generate_assert_scalar` converts an `AssertScalarLine` to `aten._assert_scalar`.
- **`codegen/wrapper.py`**:
  - `AssertScalarLine(WrapperLine)`: `wrapper` and `node`. `codegen` writes the Python wrapper's own form, and
    `codegen_fx` is the converter's `_generate_assert_scalar`.
  - `WhileLoopLine(wrapper, node, stack_output)` and `InvokeSubgraphLine(wrapper, node)` follow `SwitchLine`.
  - `ExternKernelMultiOutLine.codegen_fx` returns `_generate_extern_kernel_multi_out`. It used to inherit
    `WrapperLine`'s, which raised.
  - `KernelCallLine` gets two fields:
    - `operands: tuple | None = None`: the call's operands as values, in call order.
    - `call_kwargs: dict = {}`: its keyword operands.
  - `generate_kernel_call(..., operands=None, call_kwargs=None)` passes both through.
- **Selected call sites**:
  - `common.py` `KernelArgs.cpp_argdefs` sets `self.cpp_call_operands`, a tuple with each call arg's own value,
    appended where that call arg is:
    - an in-place, input or output buffer's outer name;
    - each size argument's sympy expression.

    Its return is unchanged.
  - `cpp.py` `KernelGroup.call_kernel` and `cpp_template_kernel.py` `CppTemplateKernel.call_kernel` pass
    `operands=self.args.cpp_call_operands`, right after their own `cpp_argdefs()` call.
  - `mps.py` `MetalKernel.call_kernel` passes `operands=tuple(typed[0])`. Its `call_kwargs` sets `threads` (grid),
    `group_size` (group) and `error_buf_idx` (error), each only where its formatted argument is appended.
- **`lowering.py` `invoke_subgraph(subgraph_fn, identifier, *operands)`** passes its `identifier` to
  `ir.InvokeSubgraph.create(subgraph_fn, identifier, *operands)`. Upstream received the identifier and dropped it.
  `InvokeSubgraph` keeps it as `identifier: str | None` (`__init__(subgraph, identifier, operands, layout)`). The
  lowering is `create`'s one caller.
- **`ir.py` `WhileLoop.create`**: `while_loop.mutated_arg_indices`, the body's mutated input indices it computes, in
  the HOP's comma-separated form.
- **`codegen/wrapper.py` `write_provenance_debug_handle`**: it calls `self.make_comment(...)` instead of
  `self.writeline(...)`, with the same text, `"{comment} [Provenance debug handles] {kernel_name}:{debug_handle}"`.
  - So the handle is a typed `CommentLine` (wrapper.py `make_comment`, the module's own comment producer), not a bare
    `str` among the wrapper's lines.
  - Its `codegen` writes the same line, and `wrapper_fxir.py` `_generate_comment` ignores it as every other comment.
  - Callers are unchanged: `ir.py` `ExternKernel.codegen_comment`, `scheduler.py`, `codegen/cpp.py` and
    `codegen/triton.py`. `trace.provenance_tracking_level` stays as configured.
- **`ir.py` `AssertScalar.codegen`**: under `fx_wrapper`, it writes `AssertScalarLine(wrapper, self)`. Upstream wrote
  nothing ("TODO fix").
- **`pattern_matcher.py` `_transfer_matched(new_nodes, match, pass_name="")`**: under
  `effective_provenance_tracking_level() == 1`, it appends `NodeSource(m, pass_name, REPLACE)` for every
  `m in match.nodes` to each new node's `from_node`.
  - It is called before `match.erase_nodes()` in `LoweringPatternEntry.apply` and `ReplacementPatternEntry.replace_with_graph`.
  - So every rewrite keeps its many-to-many correspondence. `_transfer_meta` pairs only the old output with its
    replacement.

### Target

| Item | Value |
|---|---|
| Torch | 2.14.0, `torch.version.git_version` `08187d9e0fba026dc8217405802ab5381dc88d90` |
| `torch/_inductor/compile_fx.py` | before `f4cc8a8d3ce142c67e8c43fb47434cd16ff1c7ca2665f92ec590754d10dd5592`, after `0a1460a62a59ac0a7877ebcf3aa5327e1a36447063ff041dded7f7925db23556` (revision; installed `8a229766…`) |
| `torch/_inductor/config.py` | before `9171c2ed249e13a19a2ae22dba0db14f3c4fe11762108f87c8a4e85f96945b0a`, after `adbe6ed0fc9bf5635cd2b318c76dd6fcdd97eb9bc14c22d3cebef781bfe244a5` (installed) |
| `torch/_inductor/graph.py` | before `1b3e2fdce21f6c499b655d4e4223fec37147decc7b3d95237b7bbd542cd23591` (reversing installed `d3a42366…` gives it), after `2c83b47f9e25b956fbfea16d44702eb7b4637b47d92cc9bd641960191646d0a2` (revision) |
| `torch/_inductor/ir.py` | before `5d272a6e49cd9458745c2804942fc903450a5888d15dce0e11c06ffe516e95a0`, after `e53a6377ff2a9d7678d639dedccbbf662d946367a6dac1be0f83ab29a10ebfa7` (installed6d56f939) |
| `torch/_inductor/lowering.py` | before `c2ea8692e9948b82175cf8fc3ad9fa67e98b04c9b9e1fba5ae2ac77902d18e0c` (upstream), after `5869663abf8440cbc4d11688acce36d080904ed6b96038028d4fb4e88291a394` (installed6d56f939) |
| `torch/_inductor/output_code.py` | before `647fdca5e7ab214acdf8280885d419efc607f3a003353a46611e59ec6b3d8fe6`, after `8a468b93162b105ff963868bd6fd5a5b0a2f985d5eb0b9fc0e14caca2ed0f7d5` (revision; installed `908714f1…`) |
| `torch/_inductor/pattern_matcher.py` | before `f3adcef339d1195dceb8ced943c1354006982462d940f50842f358706a91cf9b`, after `e57491f85eda8e8a048b2307142ec9d124b7a1317004359945f3ad06386d4441` (installed) |
| `torch/_inductor/codegen/wrapper.py` | before `d52e9acc719615a327b24fe87b85f70de31282686351fff0dc8d7f80e5cdefea`, after `c504b6a9862b7cabb76c27207eb9e0c1c407b31559d181e5aebce47979c768f9` (revision; installed `d9d77265…`) |
| `torch/_inductor/codegen/wrapper_fxir.py` | before `faa4ea42b0c00d2e8e7e94abd4de819022a377dc155b47a6790174e0af33b885`, after `1b9b5e525c9822cc3fd8df0e05acbb6713e013f6c0badec4bac9e7dd4c281131` (installed6d56f939) |
| `torch/_inductor/codegen/common.py` | before `3b244262921b936fc156ce394a2ae48e068b81bc621656e523f80a3d946f4466` (upstream), after `0de7b0e1b827998418296f8669edcfeec7564efcb9ccc7bc2bccea46da66fe44` (installed) |
| `torch/_inductor/codegen/cpp.py` | before `0eb0c1e9348b6b03db3e81d5158e45468be7d1554c6444864e3508fa26b5c7c3` (`torch-inductor-cpu-kernel-results.patch`'s after), after `c4b87c6a6db3dea918370285c2c6f2df555ab2930c8ce5d4386ed005ff505ce2` (installed) |
| `torch/_inductor/codegen/cpp_template_kernel.py` | before `b02db66a50404e4aa3ba5783238b95992be1041f74666cd9824cb0ac7cc397e9` (upstream), after `960ecabf55f0533f5aeddbbae443d4eb90a070a97459eb251f52254db11ff2ce` (installed) |
| `torch/_inductor/codegen/mps.py` | before `8bbd0105926cd2e738852a9cb25962ca0745526577ad8da8dd709b894c8884db` (`torch-inductor-upstream-composition.patch`'s after), after `57ce3bc432400c7206f9909ad8ba4a0cfa60c2d848053189ca8c243bee380f2a` (installed) |
| Install scope of this work | `/Users/mdot/.venv-mesh-uv`, applied by the supervisor at source revision `ba03a780` after the composition and factory patches: fuzz-0 dry run, application and AST check passed (`output_data/compiler-workgroup-20261114/compiler/numerical68-upstream-install.json`, step 3) |

Installed: the supervisor checked the eight "before" hashes and ran a fuzz-0 dry run and application of the committed
patch (`7916cb9d…`). Both passed. Every file has its "after" hash and parses (`output_data/compiler-workgroup-20261114/compiler/numerical68-upstream-install.json`, step 3).

The supervisor installed the `write_provenance_debug_handle` delta from `82295df7` after verifying every recorded
dependency hash. Fuzz-0 dry run, application, reverse dry run and AST passed; step 5 of the same manifest records
the commands and hashes. `wrapper.py` changed from `d667391a…` to the table's `5a7a1467…`; other dependencies stayed
unchanged. The delta is one line: `self.writeline(` becomes `self.make_comment(` in `write_provenance_debug_handle`.

The converter-family revision `c415dd73` (patch `b3c1dc99…`) was installed before the current revision. The supervisor checked it with a
canonical `git apply --cached` in an isolated temporary index: reverse the old patch `b5226195`, apply the new one,
then check the reverse. It then installed it with fuzz 0, and every "after" hash and AST matched
(`output_data/compiler-workgroup-20261114/compiler/numerical68-patch-preflight.json`,
`numerical68-upstream-install.json` `updates`).

The selected-call intrinsic and invoke-identifier revision `6d56f939` (patch `a75eb0b8…`) is installed. It changes
`ir.py` and `wrapper_fxir.py` and adds the `lowering.py` handoff; the other ten installed files are unchanged.
Independent canonical preflight reversed `c415dd73` in an isolated temporary index, applied this patch, parsed all
13 Python files, and reversed it to the exact original blobs. Installation then used canonical `patch` at fuzz 0:
reverse-old dry run and application, new dry run and application, and reverse-new dry run. Every installed hash
matches the projected hash in `numerical68-patch-preflight.json`; `numerical68-upstream-install.json` records the
commands. No Torch or Mesh import, fake-dispatch observation, compilation or workload was run.

The planned-stage revision (patch `87c0520c…`, 13 files) is not installed. It was checked in an isolated temporary
index:
- the installed 13 blobs, reversed by the installed `6d56f939` patch, gave the "before" blobs;
- this revision applied to them and reversed back exactly;
- `torch-inductor-cpp-build-result.patch` and `torch-inductor-cpp-wrapper-binding.patch` applied on top;
- its `config.py`, `ir.py`, `lowering.py`, `pattern_matcher.py`, `wrapper_fxir.py`, `common.py`, `cpp.py`,
  `cpp_template_kernel.py` and `mps.py` results equal the installed blobs;
- `compile_fx.py`, `graph.py`, `output_code.py` and `wrapper.py` take the revision's hashes.

The four changed files compile, with no pyflakes finding beyond each before file's. To install, reverse the
`6d56f939` patch (`git show 6d56f939:compiler/patches/torch-inductor-aot-lowered-result.patch`), then apply this one,
then `torch-inductor-cpp-build-result.patch`.

Open items:
- **The resumption's consumer.** `LoweredAOTI.emit(supplied)` has no caller yet. `supplied` is given to the print as
  `wrapper_code.supplied`, which 66's allocation, view and pool hooks are to consume. It is passed through as given;
  this patch defines no schema for it. The canonical prepared owner that supplies it (per instance and per residue)
  is 132's.
- **Resumable paths.** Only `codegen()`'s own planned stage is resumable. The GPU two-pass tails of
  `codegen_with_cpp_wrapper` are not; Mesh's CPU and MPS do not reach them. `distributed_autotune.graph_context` is
  not re-entered on resume.
- **Global provenance state.** `torch._inductor.debug`'s provenance state is not part of the captured config or
  handlers. A body resumed after another body's compile prints with that later state.
- **`wrapper_fxir.py`'s lowered hunks are unused.** These are the selected-call intrinsic, the lowered subgraphs and
  the converted correspondence. They are to be removed with 66's consumer once the cpp_wrapper consumer replaces
  them.
- **`lower_inductor._inductor`'s selection is paired in source.** It passes no `fx_wrapper` (so `cpp_wrapper`), and
  takes `LoweredAOTI.graph` into `Produced.graph` and the `LoweredAOTI` itself into `Produced.lowered` (announced; it
  replaces `Produced.compiled`). That source cannot run until this patch and `torch-inductor-cpp-build-result.patch`
  are installed.

### Apply, check, reverse

From the repository root, after `torch-inductor-upstream-composition.patch`,
`torch-inductor-graph-lowering-factory.patch` and `torch-inductor-cpu-kernel-results.patch`:

```sh
SITE=/Users/mdot/.venv-mesh-uv/lib/python3.12/site-packages
I=$SITE/torch/_inductor
F="$I/compile_fx.py $I/config.py $I/graph.py $I/ir.py $I/lowering.py $I/output_code.py $I/pattern_matcher.py $I/codegen/wrapper.py $I/codegen/wrapper_fxir.py $I/codegen/common.py $I/codegen/cpp.py $I/codegen/cpp_template_kernel.py $I/codegen/mps.py"
shasum -a 256 $F
patch --dry-run --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-aot-lowered-result.patch
patch --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-aot-lowered-result.patch
shasum -a 256 $F
```

- The first `shasum` must print the "before" hashes above, and the second the "after" hashes.
- To reverse, reverse this patch before the cpu-kernel-results, factory and composition patches:
  `patch -R --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-aot-lowered-result.patch`.
- This is pure Python source, so no rebuild is needed.

## `torch-inductor-cpp-build-result.patch`

Mechanism: installed torch 2.14.0 (`/Users/mdot/.venv-mesh-uv`):
- `torch/_inductor/cpp_builder.py` `CppBuilder`;
- `torch/_inductor/codecache.py` `AotCodeCompiler.compile` and `_precompile_header`.

Both "before" files are upstream's, and no other patch here changes them. Its `aot_inductor.lowered` is
`torch-inductor-aot-lowered-result.patch`'s config, so apply that patch first.

It returns AOT compilation's own build actions from their original owners, at one exact stage:
- **Already run by the original:**
  - the headers' precompilation (`PrecompiledHeader`). Its products persist under `_HEADER_DIR`. Its preprocessing
    output is removed with its temporary directory. Only that output's hash is kept, as `digest`; the output itself is
    not retained, and no later action reads it.
  - the constants object's build (`AotConstants`). Its source file is removed, so its content is retained before that
    removal, as `code` and `extension`. Its object's bytes are retained as `object_bytes`, read after its build (and
    any postprocessing), since a later compile can write the same object path.
- **Not run:**
  - the wrapper's and kernel's object actions (`wrapper_builder` and `kernel_builder`, `compile_only`);
  - the shared library's link action (`so_builder`), a separate, later action.

A retained action that already ran is a record of what ran: the original's provenance. Its removed paths are not
inputs to run it again. What it produced is retained as content (`code`, `object_bytes`) or as a persisting file
(`path`); `digest` is a hash only.

It replaces this patch's rejected `CppCodeCache.load_async` revision (never installed), which exposed a per-kernel
linked Python module's builders.

### What it changes

- **`CppBuildAction`**, a new NamedTuple in `cpp_builder.py`, is one builder's action as its command assembly makes it
  and `build()` realizes it:
  - `command`: its whole argument vector. Its first word is the compiler as given (it can be a bare name or a
    wrapper command); it is not resolved here. `_run_compile_cmd`'s `subprocess.run` resolves it with the inherited
    environment, in `working_directory`.
  - `sources` and `target`: the positions of its source operands and its target operand;
  - `ldflags`, `libraries` and `library_directories`: the positions of its link inputs and their resolution. These are
    empty unless the action links.
  - `product`: `"object"`, `"precompiled_header"`, `"preprocessor_output"` or `"python_module"` (a linked shared
    library), as the builder selects its output flags;
  - `target_file` and `precompiled_header`: the file it produces and the header its `-include` names;
  - `directories` and `working_directory`: what `build()` creates (output directory, then build directory) and runs
    in (the build directory, removed afterwards);
  - `environment`: `os.environ`, which the command's subprocess inherits;
  - `relative`: under `use_relative_path`, `build()` realizes the action through `build_fbcode_re` instead.
- **`CppBuilder`**:
  - `__init__` records `_product` in its output-flags branch. It keeps its sources and output as role-tagged parts
    (`_sources_parts`, `_output_parts`), so each source operand and the target are their own part.
  - `_command_parts()` is the command's one assembly: each field in upstream's order (Windows' and POSIX's), each
    part with its role.
  - `get_command_line()` joins those parts and is byte-identical to upstream's. `build()` and `build_fbcode_re` still
    run it.
  - `get_build_action()` takes each argument's position from its own part, with no matching against argument strings.
    An operand is its part's last word; on MSVC the target is inside the output flag's argument.
  - `get_build_directory()`: `{output_dir}/{name}_CxxBuild`. `build()` now calls it.
- **`_precompile_header`** keeps its signature and result (the header path) for `CppCodeCache.load_async`. The work moves
  unchanged into the cached `_precompiled_header`, which returns **`PrecompiledHeader`**:
  - `header`;
  - `preprocessing`: the preprocessor's action, in its own temporary directory;
  - `checksum`: the `openssl sha512` command, now one retained argument list;
  - `digest`: that command's result over the preprocessing output (`preprocessor_hash`), taken before the temporary
    directory holding that output is removed. It is a hash only: the output is not retained;
  - `path`: the header file written under `_HEADER_DIR`, named by that digest;
  - `precompiling`: the precompile action, which `_worker_compile_cpp` skips if the product exists;
  - `lock`.
- **`AotCodeCompiler.compile`**:
  - Its precompile calls use `_precompiled_header` and keep `wrapper_header` and `kernel_header`.
  - Under `aot_inductor.lowered` it does not run `wrapper_builder.build()` or `kernel_builder.build()`.
  - After `so_builder` is constructed and the compile/link comments are appended to the sources, it returns
    **`AotCompiled`**:
    - `wrapper` and `kernel`: the source paths;
    - `wrapper_object` and `kernel_object`: the two object actions;
    - `wrapper_header` and `kernel_header`;
    - `consts`;
    - `link`: the `so_builder` action;
    - `extern_kernel_nodes`: the serialized extern table it was given;
    - `generated_files`;
    - `wrapper_code` and `kernel_code`: the sources' content as this compile produced it, with the compile/link
      comments it appended (`wrapper_appended`, `kernel_appended`, the same bytes it writes). `write` does not rewrite
      an existing path, so under a fixed `model_name_for_generated_files` the files at `wrapper` and `kernel` can be an
      earlier compile's; the content fields are this compile's;
    - `model_class`: `aoti_model_class_name` (`AOTInductorModel{suffix}`, with the suffix under static linkage);
    - `model_header` and `model_header_code`: under static linkage, the renamed `model.h` header's path and content
      as written (codecache 2713–2740). Both are `""` under dynamic linkage, where the wrapper inlines
      `aoti_runtime/interface.cpp` instead.
  - Its stage, exactly: after the wrapper's final print, its sources written, its headers precompiled and its
    constants object built; before any wrapper/kernel compile or link. It is the result of the aot patch's resumed
    emission (`LoweredAOTI.emit`), after that patch's planned stage.
  - `link` is the whole `so_builder` action:
    - its command over its objects (wrapper's, kernel's, constants', GPU kernels');
    - its target, its link inputs and their resolution;
    - its product, directories, working directory and environment.
- **`_compile_consts`** returns **`AotConstants`** in place of the object path:
  - `code` and `extension`: its source's content as written (`consts_code`) and its extension (`code_ext`), held in
    memory before `os.remove` of its source file. `action`'s command still names that file, as it ran;
  - `action`: `object_builder`'s whole action;
  - `assembly` and `directory`: the command `run_asm_build_object` ran instead, and where. This happens with
    `use_consts_asm_build` off and zero-size constants; otherwise `assembly` is None.
  - `postprocessed`: whether the constants' data was then written over the object's magic number (large constants
    under `use_consts_asm_build`);
  - `object`;
  - `object_bytes`: that object's content, read after its build and postprocessing, before `os.remove` of its
    source. Its path is keyed by model name or by `consts_code`'s hash, which leaves out the postprocessed data, so a
    later compile can write over it.

  `compile` uses `.object` where it used the path.
- **`run_asm_build_object`** now returns the command it ran (`shlex.split` of it, as `_run_compile_cmd` splits it). Its
  only caller is `_compile_consts`.

### Target

| Item | Value |
|---|---|
| Torch | 2.14.0, `torch.version.git_version` `08187d9e0fba026dc8217405802ab5381dc88d90` |
| `torch/_inductor/cpp_builder.py` | before `fad619f890b25f0f15a7f563ea00e36da16d800f37f3060330be2de9304db373` (upstream; installed), after `fa98d76f45751abb6cf88bd8cb1e48d6fd8a0defe16077090185b229279a963a` |
| `torch/_inductor/codecache.py` | before `1edadc1e1229adbd7c6f84cef43a04c031add6e1793be0f12b73f22798966156` (upstream; installed), after `83409729b8f8f1c29cfb6e2fdfe82871a8ffa94e118309a78e7b8d94414b8ae8` |

Open items:
- **Constants object.** It is built here by its original owner, and retained in `AotConstants`. Whether the archive
  binds constants through it or through native values (`constants_->at(idx)`) is for 104/131 to pair.
- **Reached through the uninstalled aot revision.** `torch-inductor-aot-lowered-result.patch` `87c0520c…` returns
  `LoweredAOTI` from compile_fx's original `graph.aot_mode` branch under `cpp_wrapper`, at its planned stage. Its
  `emit` is what reaches `compile` and returns `AotCompiled`. Neither patch is installed; together they are one
  dependency unit for preflight (install order in that section).
- **After-print result.** `AotCompiled` follows the wrapper's final print. Its `wrapper_code` and `kernel_code` are
  that print's source operands. The planned lines and planner owners before that print are the aot patch's
  `LoweredAOTI.planned` and `wrapper_code`'s. The supplied bindings that print takes are 66's hooks'.
- **Multiplicity (WIP).** One `emit` per `LoweredAOTI` resumes its one generation. Several configurations of one
  planned owner (`Target.instanced` rows over one function) need one final emission each. `compile` would then run
  `_compile_consts` again for each, though the constants do not depend on the configuration. Open.
- **Unit identity.** Its emitter runs with upstream's `aot_inductor.dynamic_linkage` (True), so `aoti_model_class_name`
  carries no unit suffix. The static composition's unit identity (`dynamic_linkage` False and
  `model_name_for_generated_files`) is not selected. No local ordinal is used, and no name is derived from a repr.
  `compiled_fx_graph_hash` (codecache 1772), the canonical graph/input/config key, is not computed under `aot_mode`
  (compile_fx 1031: `use_cache` False), so it is not present here. The usable unit result is to be assessed with 131.

Not installed yet. It was checked in an isolated temporary index holding the installed blobs:
- `git apply --cached` gives the "after" hashes, with no offset;
- `git apply --cached -R` gives back exactly the original blobs.

Both "after" texts compile, and neither has a new pyflakes finding.

### Apply, check, reverse

From the repository root:

```sh
SITE=/Users/mdot/.venv-mesh-uv/lib/python3.12/site-packages
F="$SITE/torch/_inductor/cpp_builder.py $SITE/torch/_inductor/codecache.py"
shasum -a 256 $F
patch --dry-run --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-cpp-build-result.patch
patch --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-cpp-build-result.patch
shasum -a 256 $F
```

- The first `shasum` must print the "before" hashes above, and the second the "after" hashes.
- To reverse: `patch -R --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-cpp-build-result.patch`.
- This is pure Python source, so no rebuild is needed.

## `coremltools-exir-correspondence.patch`

Mechanism: installed coremltools 9.0 (`/Users/mdot/.venv-mesh-uv`).

- `coremltools/converters/mil/frontend/torch/converter.py` `TorchConverter.convert` binds each ExportedProgram user
  input and each mutated buffer's state to its MIL function input, in the function's input order.
  - It strips its own rank-0 wrap there (1347–1360).
  - It reads and writes states through `input_name_to_source_buffer_name` and `output_name_to_target_buffer_name`
    (1366–1416).
  - It assembles the outputs in `self.graph.outputs` order, dropping None ones.
- `exir_utils.py` `_extract_outputs_from_exir_program` (292–317) is the extraction owner of `self.graph.outputs`. It
  appends `output_spec.arg.name` for every `USER_OUTPUT`, in signature order. For a `ConstantArgument` (torch
  graph_signature.py `_make_argument_spec` 599 and 627) that name is `""` or a node name, and neither need be a context
  entry, so `convert`'s `self.context[name]` (`TranscriptionContext.__getitem__`) raises ValueError.
- The MIL passes that rewrite a function's outputs are each that rewrite's owner:
  - `preprocess.py` `update_output_dtypes` (one new output per old, at its index);
  - `optimize_repeat_ops.py` (one identity sink per output, later restored);
  - `quantization.py` (an input-cast var replacing its occurrences);
  - `symbol_transform.py` (a new target function from a source function's outputs).
- `Block._replace_var` replaces per index. The input renames and casts keep the function's input order:
  `_sanitize_block_input_vars` and `quantization.py` 235.

### What it changes

- **`_extract_outputs_from_exir_program`** keeps each static user output's own `ConstantArgument` at its signature
  position in `user_outputs`, and the name otherwise. This is the distinction torch's
  `ExportGraphSignature.user_outputs` makes. So `self.graph.outputs` index k is user output k.
- **`convert`** resolves a `ConstantArgument` occurrence to no Var: it is the original's None constant output, dropped
  from the MIL outputs. Its value stays at its one owner, the signature's `ConstantArgument.value`.
- **`convert`** records the converter's own correspondence on its main function:
  - `exir_inputs`: each function input's `(state, torch name or buffer, unwrapped)`;
  - `exir_outputs`: each kept output occurrence's `(user output position, torch name)`, parallel to the function's
    outputs;
  - `exir_buffers`: its read and write maps, in the torch.export frontends that have them.
- **Each output-rewrite owner** carries `exir_outputs` through its own rewrite. Each new output occurrence takes its
  source occurrence's record: `update_output_dtypes`, both `optimize_repeat_ops` output rewrites, `quantization.py`'s
  output replacement, and `symbol_transform.py`, which carries all three records to the target function.

`compiler/mesh_compiler/lower_inductor.py` `_engine` reads these into `Model.inputs`, `.outputs` and `.states`.
- Each static output reaches its consumers from the torch owners. `Produced.Output` takes `ConstantArgument.value`.
  In `lower_metal.py` `neural`, `aot_aliases`' forward result at that position is the literal: export.py 149 keeps a
  non-Node leaf, and its `OutputType.non_alias` handler returns it unchanged.
- `USER_INPUT_MUTATION` still raises in the original (`exir_utils.py` 301). Mutated user inputs, and their Neural
  Engine mapping, are not covered.

### Target

| File (under `coremltools/converters/mil/`) | Before (sha256) | After (sha256) |
|---|---|---|
| `frontend/torch/converter.py` | `fb081deff06839df6bb80221fd77ee8c00422db2f44a0c9f78b9fb1418f85ed9` | `8fba601485e31d4a5440fa0d74b8737b3b262d052bfefbf12937eca99ba39f66` |
| `frontend/torch/exir_utils.py` | `59ed5998063775f4b223ead46cef0c7048a1e934eeb1d35579877706a9f56e9b` | `7ad40697304590c172493e8c6b1f55568df915bf225bc43398135a96add4533c` |
| `mil/passes/defs/preprocess.py` | `06f2b1c42906bb364861761e199c22461341fb26b0702fae7f1a78158d76cb7f` | `e00cf45fe1cadb918c7dab27518c7cc81ec37778e09bb0ec0d2d5e701c63173f` |
| `mil/passes/defs/optimize_repeat_ops.py` | `b7d0486e9bf1ac341f1cdf84691b07d180d4d1873852f7f8ef2da3fef0cc2ce6` | `21475a1a70a27540949d9631b01b969c63c7eb03f5333e500cf9c5547ef9ef17` |
| `mil/passes/defs/quantization.py` | `14e7fcd3f00423bdae42e970f483ddb8c9d82da4b904016ac0b5055ea2fa19a2` | `5d4b5425fcc054c943c8dbf401aae19becd8fa753faf4726e7b5e29fa4cba749` |
| `mil/passes/defs/symbol_transform.py` | `4d054c4e883eb3a32f5ae70b431aff45cdf58cce8c0bab697a2a86c641d5c9af` | `31b2a7c665dd01fc2661f368c9bffcac111afa05766050b475199202683796d2` |

### Apply, check, reverse

```sh
SITE=/Users/mdot/.venv-mesh-uv/lib/python3.12/site-packages
D=$SITE/coremltools/converters/mil
F="$D/frontend/torch/converter.py $D/frontend/torch/exir_utils.py $D/mil/passes/defs/preprocess.py $D/mil/passes/defs/optimize_repeat_ops.py $D/mil/passes/defs/quantization.py $D/mil/passes/defs/symbol_transform.py"
shasum -a 256 $F
patch --dry-run --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/coremltools-exir-correspondence.patch
patch --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/coremltools-exir-correspondence.patch
shasum -a 256 $F
```

- The first `shasum` must print the "before" hashes above, and the second the "after" hashes.
- To reverse: `patch -R --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/coremltools-exir-correspondence.patch`.
- This is pure Python source, so no rebuild is needed. It is not applied here; the executor owns installation.

## `torch-inductor-cpp-wrapper-binding.patch`

Mechanism: installed torch 2.14.0 (`/Users/mdot/.venv-mesh-uv`), `torch/_inductor/codegen/cpp_wrapper_cpu.py`
`CppWrapperCpu` (`CppWrapperGpu` and `CppWrapperMps` inherit these unchanged), 16 hunks, under
`aot_inductor.lowered` only:
- the entry: `_write_input_preamble`, `_write_input_unpacking`, `_write_constants_unpacking` (JIT constant),
  `_write_entry_point_postamble`;
- the allocation and view bindings: `make_allocation` (through `make_buffer_allocation` and memory_planning.py
  `AllocationPool.codegen_create`), `codegen_alloc_from_pool`, `codegen_reinterpret_view`, and `_bound_prepared` (new);
- their consumption: `make_buffer_free`, `make_free_by_names`, `codegen_exact_buffer_reuse`, `generate_return`;
- the seeds writer: `_generate_extern_kernel_out_helper`;
- the entry's timing: `generate` and `resume_generation` (new override), hunk `@@ -1470,8 +1536,26 @@`.

Ownership (supervisor, 11-15): 66 owns the binding semantics (the entry preamble, allocation, pool, view and control
outputs). 68 owns the original `generate`/`_generate` planning-to-entry-print timing, as part of
`torch-inductor-aot-lowered-result.patch`. The two timing functions above overlap that grant. They are kept as
written here; 68 owns their next final-stage integration. Both patches change separate functions of one upstream
file and apply in this order: that patch, `torch-inductor-cpp-build-result.patch`, then this one.

The supplied operand is the original `wrapper.supplied`: 68's `LoweredAOTI.emit(supplied)` resumes the planned stage
(`resume_generation`), which sets it before print. There is no parallel copy. It is to be computed over
`LoweredAOTI.planned` (the lines `run_wrapper_ir_passes` leaves) and the wrapper's planner operands (`memory_planner`
or `past_planning_states`). That consumer, in lower_metal.py (66), is not written yet. The slots it names are the
handles that 132's per-instance/per-residue prepared owner prepares. Everything else in the emitter is unchanged:
numeric kernels, control, extern calls and the proxy executor call.

### What it changes

- **Entry timing.** `generate` skips `write_wrapper_decl` for a lowered non-const graph. `resume_generation(supplied)`
  sets `self.supplied`, runs `write_wrapper_decl` and then the original resumption. So the declaration (signature,
  input preamble and unpacking, constants and input symbols) is printed with `supplied` known, still first in
  `prefix`. Nothing in `_generate` writes `prefix` before 68's planned yield, and scheduler codegen precedes
  `generate` in the original too. A const graph, which resumes at once, keeps the original order.
- **Borrowed entry.** `_write_input_preamble` records `bound_entry`:
  - `aot`: `run_impl`'s input count, or None for a const graph's `_const_run_impl`, which has no `input_handles`;
  - `jit`: `inductor_entry_impl`'s;
  - `inputs` and `constants`: their names in entry order.

  It emits `bound_handles = input_handles + {aot}` (JIT: `+ {jit}`, after its GIL release) and steals nothing.
  `_write_input_unpacking` binds each input as `ConstantHandle {name}(input_handles[idx])`, held (`bound_held`).
  A scalar input reads `input_handles[idx]`, and JIT's constant is a `ConstantHandle`. The postamble has no
  `inputs.clear()`. A wrapper whose misaligned-input copy rebinds an input to an owned clone
  (`CppWrapperGpu.codegen_inputs`, `inputs_to_check`) raises.
- **`make_allocation`** consumes `supplied[name] = (k, prepared)`; its slot is `bound_handles[k]`.
  - Prepared: `{handle} = bound_handles[k]; ConstantHandle {name}({handle});`, the original's non-owning constant
    binding. It is held and prepared (`_bound_prepared`).
  - Otherwise: the slot is its storage, viewed at offset 0 with its runtime allocation size and stride
    (`aoti_torch__alloc_from_pool`), then restrided as the original does.
  - Recorded in `bound_allocations`, in print order, with its operands as given and `graphs`, `supplied`, `handle`
    and `result`.
  - It raises in three cases: no binding supplied, a const graph, and a prepared binding within a control subgraph.
- **`codegen_alloc_from_pool`** consumes `supplied[("pool", pool, offset, dtype, shape, stride)] = (k, names)`.
  The view is the slot, `ConstantHandle(tmp)` with `tmp = bound_handles[k]`, and `names` (its `AllocFromPoolLine`
  group's) are prepared. An unsupplied, statically known view of a prepared pool raises.
- **`codegen_reinterpret_view`** at print consumes `supplied[("view", data, size, stride, offset, dtype)] = (k, names)`:
  `ConstantHandle tmp_{data}_{id}(bound_handles[k])`, prepared with `names` (a non-exact reuse's new buffer).
  Each call is recorded in `bound_views` with `stage` set to "print" or "scheduler". An unsupplied, statically known
  print-stage view of a prepared binding raises.
- **Consumption.** For a held binding, `make_buffer_free` and `make_free_by_names` emit nothing. Its exact reuse
  (`codegen_exact_buffer_reuse`) is held, and prepared if the original was. `generate_return` emits
  `output_handles[idx] = {name}.get()`.
- **`generate_return`** first raises on any supplied key the print did not consume (`bound_consumed`). It then
  records `bound_outputs {idx: "owned" | "borrowed" | earlier idx}`:
  - "owned": the caller steals the handle (a produced value, scalar or constant clone);
  - "borrowed": the handle is a held binding's, which the caller already holds;
  - earlier idx: an alias of that output position.
- **Seeds writer.** `_generate_extern_kernel_out_helper` with `supplied[("seeds", name)]` records `bound_seeds[name]`
  and emits no `randint_low_out` draw.

### Entry binding

- 131's body call is the original entry itself: `AOTInductorModel::run_impl(input_handles, output_handles, stream,
  proxy_executor)`, or the JIT `inductor_entry_impl`. The container's `AOTInductorModelContainerRun` checks
  `num_inputs`, so it is not this entry.
- `input_handles[0, aot)` are the entry's own inputs, borrowed: the caller keeps every handle it passes. After them
  come the supplied slots, `bound_handles[k]`, also borrowed. Under MetalNativeTask's late external-pointer rebinding
  per invocation, a handle stays the caller's to rebind. Nothing here assumes pointer stability or makes a new handle
  per call.
- 132's owner prepares, retains and releases each slot. A prepared slot is that exact view. Any other slot is a
  storage of `compute_required_storage_length(allocation_shape, stride, 0)` elements of `dtype` on `device`.
- Output ownership is `bound_outputs`: a "borrowed" output is a slot or input handle, which the caller must not free.

### Unfinished obligations (supervisor source review, 11-15; not a final domain)

- **Occurrence identity.** The keys `supplied[name]` and `("view", data, size, stride, offset, dtype)` cannot tell
  apart repeated occurrences of one original control or entry. The final binding must consume the original planned
  line, IR, pool and view identity plus occurrence, not recover it from names or geometry.
- **Control and early views.** `_bound_prepared`'s errors are obligations still to be met, not limits of what is
  supported:
  - a prepared binding within a control subgraph (its outputs move into the control's RAII outputs,
    `codegen_subgraph_suffix`);
  - a scheduler-stage view of a prepared binding;
  - a `while_loop` body's per-pass allocations and carried rebinding.
- **Seeds.** The branch suppresses the original writer on key presence alone. The retained state/effect owner must
  supply the corresponding data and its advance once per source call, not a metadata assertion.
- **Const graph entry operands:** required; currently a const graph's allocation raises and its emission is early.
- **Results an operator allocates itself:** a fallback's returned handle, the proxy executor's outputs, and a constant
  output's `aoti_torch_clone`. They are "owned" in `bound_outputs`.
- **CppWrapperGpu's misaligned-input copy:** it raises.
- **`CppWrapperCpuArrayRef`'s own unpacking.**
- **Multiplicity.** The supplied operand depends only on the planned result. Per-configuration and per-residue
  differences are the caller's handle arrays, so one emission per `LoweredAOTI` would serve every row. The
  per-configuration emitter result from the same retained plan is 68's.

### Target

| File | Before (sha256) | After (sha256) |
|---|---|---|
| `torch/_inductor/codegen/cpp_wrapper_cpu.py` | `d03d3032c9eb078eb918f70f0f44bb823ec6a6d0481b677e4a60ce4a68b2025a` (upstream; installed) | `d5bad4bd3d986bb30c381764442ce9e0fea4060b3cec19ae2df49dc6b9321dff` |

Patch sha256 `71ecddd88f1222a27e3bf28fb3686e4218d2f6f942270ddb3324b54f6e6a2c1c`, 544 lines.

### Combined canonical preflight (11-15)

The preflight ran in an isolated temporary git index (`GIT_INDEX_FILE`), which held the installed blobs of all 16
files the dependency unit touches:
1. `git apply --cached -R` the installed aot revision `6d56f939` (patch `a75eb0b8…`).
2. `git apply --cached` the current `torch-inductor-aot-lowered-result.patch` (`87c0520c…`), then
   `torch-inductor-cpp-build-result.patch` (`de3ee6d8…`), then this patch (`71ecddd8…`). All applied without offset.
3. All 16 results parse (`ast.parse`) and compile.
4. The seven changed files have no pyflakes finding beyond their installed text's. Those files are `compile_fx.py`,
   `graph.py`, `output_code.py`, `wrapper.py`, `cpp_builder.py`, `codecache.py` and `cpp_wrapper_cpu.py`.
5. `cpp_wrapper_cpu.py` is touched by this patch alone: its before is the installed blob, and its after is the hash
   above.
6. Reversing the three in reverse order gave back the exact pre-aot blobs. Re-applying `6d56f939` gave back the exact
   installed blobs.
7. The index was removed.

No Torch or Mesh import, compilation or workload was run. The patch was generated by `difflib` from the after-text,
and its applicability is `git apply`'s.

### Apply, check, reverse

Not installed; the executor owns installation. In install order (after the aot revision's own steps, then
`torch-inductor-cpp-build-result.patch`), from the repository root:

```sh
SITE=/Users/mdot/.venv-mesh-uv/lib/python3.12/site-packages
shasum -a 256 $SITE/torch/_inductor/codegen/cpp_wrapper_cpu.py
patch --dry-run --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-cpp-wrapper-binding.patch
patch --fuzz=0 --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-cpp-wrapper-binding.patch
shasum -a 256 $SITE/torch/_inductor/codegen/cpp_wrapper_cpu.py
```

- The first `shasum` must print the "before" hash above, and the second the "after" hash.
- To reverse:
  `patch -R --no-backup-if-mismatch -p1 -d $SITE < compiler/patches/torch-inductor-cpp-wrapper-binding.patch`.
- This is pure Python source, so no rebuild is needed. No C++ was compiled.
