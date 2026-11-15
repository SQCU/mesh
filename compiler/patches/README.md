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
(`cpp_wrapper`). `compile_fx_aot` returns the `LoweredAOTI` itself:
- `graph`: its GraphLowering after `codegen_with_cpp_wrapper`. Its scheduler codegen has run, and its own wrapper_code
  (CppWrapperCpu or CppWrapperMps) has emitted its wrapper and kernel code.
- `compiled`: `AotCodeCompiler.compile`'s `AotCompiled` (`torch-inductor-cpp-build-result.patch`). Its object, header,
  constants and link actions are held there; its wrapper's and kernel's objects are not compiled, and its shared
  library is not linked.

The revision after `6d56f939` replaces the fx-wrapper return (`GraphLowering.codegen` after `FxConverter.generate`).
That return is not removed by deletion alone: this patch's `wrapper_fxir.py` hunks remain, unused under `cpp_wrapper`,
and are listed as an open item. `compiler/mesh_compiler/lower_inductor.py` `_inductor` takes the producer's
result through it. Fields: `output_data/compiler-workgroup-20261114/compiler/numerical68-result.md`.

### What it changes

- **`config.aot_inductor.lowered: bool = False`**: the operand that asks for the lowered result. With it False, every
  output is unchanged.
- **`GraphLowering.codegen`**: unchanged. The earlier `(self, None)` return under `lowered` is gone, because
  `codegen_with_cpp_wrapper` consumes `codegen()`'s own result. `graph.py` is no longer in this patch.
- **`output_code.py` `LoweredAOTI(OutputCode)`**: a new dataclass with `graph` (the GraphLowering), `device_type` and
  `compiled` (`AotCompiled`). Its `post_compile` and `set_triton_bundle` do nothing, because there is no emitted
  callable to load.
- **`compile_fx.py`**:
  - `codegen_and_compile` takes its original `graph.aot_mode` branch unchanged: `codegen_with_cpp_wrapper`, the extern
    node serializer, then `AotCodeCompiler.compile`, which under `lowered` returns `AotCompiled`. Its
    `V.aot_compilation` return is `LoweredAOTI(graph=graph, device_type=graph.device_type, compiled=compiled_fn)`.
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
| `torch/_inductor/compile_fx.py` | before `f4cc8a8d3ce142c67e8c43fb47434cd16ff1c7ca2665f92ec590754d10dd5592`, after `3647ae6d2286dd2c277a4b919895d04a0beae7afe029dcbbf9373dde66afa3bd` (revision; installed `8a229766…`) |
| `torch/_inductor/config.py` | before `9171c2ed249e13a19a2ae22dba0db14f3c4fe11762108f87c8a4e85f96945b0a`, after `adbe6ed0fc9bf5635cd2b318c76dd6fcdd97eb9bc14c22d3cebef781bfe244a5` (installed) |
| `torch/_inductor/graph.py` | no longer patched: reversing installed `d3a42366…` returns it to `1b3e2fdce21f6c499b655d4e4223fec37147decc7b3d95237b7bbd542cd23591` |
| `torch/_inductor/ir.py` | before `5d272a6e49cd9458745c2804942fc903450a5888d15dce0e11c06ffe516e95a0`, after `e53a6377ff2a9d7678d639dedccbbf662d946367a6dac1be0f83ab29a10ebfa7` (installed6d56f939) |
| `torch/_inductor/lowering.py` | before `c2ea8692e9948b82175cf8fc3ad9fa67e98b04c9b9e1fba5ae2ac77902d18e0c` (upstream), after `5869663abf8440cbc4d11688acce36d080904ed6b96038028d4fb4e88291a394` (installed6d56f939) |
| `torch/_inductor/output_code.py` | before `647fdca5e7ab214acdf8280885d419efc607f3a003353a46611e59ec6b3d8fe6`, after `7784b80f42c7e16c24f7a587987b4f88747a0afc6e7a7bbc676181144f81e961` (revision; installed `908714f1…`) |
| `torch/_inductor/pattern_matcher.py` | before `f3adcef339d1195dceb8ced943c1354006982462d940f50842f358706a91cf9b`, after `e57491f85eda8e8a048b2307142ec9d124b7a1317004359945f3ad06386d4441` (installed) |
| `torch/_inductor/codegen/wrapper.py` | before `d52e9acc719615a327b24fe87b85f70de31282686351fff0dc8d7f80e5cdefea`, after `d9d77265fd62a5debb212113250809b37704e1e6400ef747c68a8adbaf695e52` (installed) |
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

The `cpp_wrapper` return revision (patch `415b82a4…`, 12 files) is not installed. It was checked in an isolated
temporary index:
- the installed 13 blobs, reversed by the installed `6d56f939` patch, gave the "before" blobs;
- this revision applied to them and reversed back exactly;
- its `config.py`, `ir.py`, `lowering.py`, `pattern_matcher.py`, `wrapper.py`, `wrapper_fxir.py`, `common.py`, `cpp.py`,
  `cpp_template_kernel.py` and `mps.py` results equal the installed blobs;
- `compile_fx.py` and `output_code.py` take the revision's hashes, and `graph.py` its "before".

Both changed files compile, with no pyflakes finding beyond the before file's. To install, reverse the `6d56f939`
patch (`git show 6d56f939:compiler/patches/torch-inductor-aot-lowered-result.patch`), then apply this one, then
`torch-inductor-cpp-build-result.patch`.

Open items:
- **`wrapper_fxir.py`'s lowered hunks are unused.** These are the selected-call intrinsic, the lowered subgraphs and
  the converted correspondence. They are to be removed with 66's consumer once the cpp_wrapper consumer replaces
  them.
- **`lower_inductor._inductor`'s selection is paired in source.** It passes no `fx_wrapper` (so `cpp_wrapper`), and
  takes `LoweredAOTI.graph` and `.compiled` into `Produced.graph` and the new, announced `Produced.compiled`. That
  source cannot run until this patch and `torch-inductor-cpp-build-result.patch` are installed.

### Apply, check, reverse

From the repository root, after `torch-inductor-upstream-composition.patch`,
`torch-inductor-graph-lowering-factory.patch` and `torch-inductor-cpu-kernel-results.patch`:

```sh
SITE=/Users/mdot/.venv-mesh-uv/lib/python3.12/site-packages
I=$SITE/torch/_inductor
F="$I/compile_fx.py $I/config.py $I/ir.py $I/lowering.py $I/output_code.py $I/pattern_matcher.py $I/codegen/wrapper.py $I/codegen/wrapper_fxir.py $I/codegen/common.py $I/codegen/cpp.py $I/codegen/cpp_template_kernel.py $I/codegen/mps.py"
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
    output is removed with its temporary directory, so its result is retained before that removal, as `digest`.
  - the constants object's build (`AotConstants`). Its object persists. Its source file is removed, so its content is
    retained before that removal, as `code` and `extension`.
- **Not run:**
  - the wrapper's and kernel's object actions (`wrapper_builder` and `kernel_builder`, `compile_only`);
  - the shared library's link action (`so_builder`), a separate, later action.

A retained action that already ran is a record of what ran: the original's provenance. Its removed paths are not
inputs to run it again; what it produced is retained as content (`code`, `digest`) or as a persisting file (`object`,
`path`).

It replaces this patch's rejected `CppCodeCache.load_async` revision (never installed), which exposed a per-kernel
linked Python module's builders.

### What it changes

- **`CppBuildAction`**, a new NamedTuple in `cpp_builder.py`, is one builder's action as its command assembly makes it
  and `build()` realizes it:
  - `command`: its whole argument vector, compiler executable first;
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
    directory holding that output is removed;
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
    - `generated_files`.
  - Its stage, exactly: after its sources are written, its headers precompiled and its constants object built; before
    any wrapper/kernel compile or link.
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
  - `object`.

  `compile` uses `.object` where it used the path.
- **`run_asm_build_object`** now returns the command it ran (`shlex.split` of it, as `_run_compile_cmd` splits it). Its
  only caller is `_compile_consts`.

### Target

| Item | Value |
|---|---|
| Torch | 2.14.0, `torch.version.git_version` `08187d9e0fba026dc8217405802ab5381dc88d90` |
| `torch/_inductor/cpp_builder.py` | before `fad619f890b25f0f15a7f563ea00e36da16d800f37f3060330be2de9304db373` (upstream; installed), after `0d8c8abb7594a6c4cef4365eed0d76b8be0769b2e3eceffec917f23a37c02e46` |
| `torch/_inductor/codecache.py` | before `1edadc1e1229adbd7c6f84cef43a04c031add6e1793be0f12b73f22798966156` (upstream; installed), after `8954eba36af6573dc7ae10286aba4be3dc305d2852f80aea254c7ed7e9215e5e` |

Open items:
- **Constants object.** It is built here by its original owner, and retained in `AotConstants`. Whether the archive
  binds constants through it or through native values (`constants_->at(idx)`) is for 104/131 to pair.
- **Reached through the uninstalled aot revision.** `torch-inductor-aot-lowered-result.patch` `415b82a4…` returns
  `LoweredAOTI.compiled` from compile_fx's original `graph.aot_mode` branch under `cpp_wrapper`. Neither patch is
  installed; together they are one dependency unit for preflight (install order in that section).
- **Source operands are the wrapper's final print.** `wrapper` and `kernel` are what `codegen_with_cpp_wrapper`'s
  final `generate` wrote. The typed body/allocation/binding/transport operands before that print are not returned
  here. They are 66's hooks' (`cpp_wrapper_cpu.py`, reserved), and the stage that returns or consumes them is open.
- **Unit identity.** Its emitter runs with upstream's `aot_inductor.dynamic_linkage` (True), so `aoti_model_class_name`
  carries no unit suffix. The static composition's unit identity (`dynamic_linkage` False and
  `model_name_for_generated_files` from the body's existing unit) is not selected yet; its operand is to be paired
  with 131.

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
