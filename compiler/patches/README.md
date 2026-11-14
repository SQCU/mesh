# Dependency patches

Style: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights).

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
