# Review corrections for the current owner unit

Read this before reporting completion and before any source-validation checkpoint.

Executor047 owns the one C ABI representation. lower_metal's SourceOperand ctypes structure and the inline
metal_ir_source_relation.argtypes assignment also move to metal.py. Use metal.SourceOperand at the actual
semantic operand construction site, and remove lower_metal's duplicate class/prototype setup. Executor has this
assignment; coordinate with executor/live-metal-owner.status.md. Source-kind constants that merely encode the
C enum belong with that ABI owner too.

Publish the exact numerical boundary and source now. Dependency analysis is useful only to keep obsolete
orchestration out; do not spend the entire turn developing a migration framework. No migration script is a new
maintained tool. Original malformed files stay exactly where they are and are never imported or re-enabled.

## Output identity found by frontend and executor review

One functionalized FX result can occur in both USER_INPUT_MUTATION(target=x) and USER_OUTPUT: e.g. a function
returning x.add_(...). lower_metal's mutation output currently writes into original input and records that root;
the USER_OUTPUT later reads env[leaf]'s temporary root. Preserve the explicit output role/source correspondence
so both describe the actual caller input root/view. This is a producer annotation supplied by torch graph_signature,
not inferred from runtime pointers. The same issue applies to a BUFFER_MUTATION returned as a user output.
Complete this in the actual lower_metal output recipe and paired ConstructedMachine outputs, coordinating native
owner047; no extra runtime alias detection or redundant output copies. Preserve genuinely distinct outputs/views.

The frontend now retains TrainingResult(outputs, parameters, inputs) with separate actual gradient-target maps,
not mixed key names. Do not reconstruct or reinterpret that pytree here; keep the explicit output roles.
