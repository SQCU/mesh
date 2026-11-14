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

Executor's implemented setter declares its callback as metal.RELEASE. Passing a borrowed/no-release relation
therefore uses RELEASE() (ctypes' typed null function pointer), not Python None. Import that ABI type from metal.
Use actual stage construction to check this paired call; mere successful imports do not exercise ctypes arguments.

## Actual producer observation: returned views also retain mutation identity

Root ran actual torch.export.export(...).run_decompositions() in the sanctioned Torch environment on:
`def forward(self, x, y): x.add_(y); return x[1:]`, shape (4,).
The output specs are USER_INPUT_MUTATION x, node `add`, shape(4,), offset0, and USER_OUTPUT, node `slice_2`,
shape(3,), offset1. Both meta values have the same storage. Therefore a map keyed by exact output Node alone
misses the returned view. This is a concrete producer case, not a hypothetical parser rule.

Prefer feeding the already-declared mutation destination into the existing _instruction destination operand
before numerical construction, so subsequent explicit source view operations consume the actual destination
and preserve its strides/offset. Inspect the same canonical view/destination machinery first. Do not add a
second alias parser or a runtime alias detector. A returned slice of noncontiguous input is also a view of that
actual input; substituting contiguous temporary metadata would be wrong. Complete the actual typed handoff.

Ordering is part of that handoff: introducing an actual input write only inside lower_metal, after schedule, is
too late to preserve earlier reads such as `z=x+1; x.add_(2)`. Frontend974 is assigned producer-side mutation
destination annotation before dependence/schedule, using the existing ir.views/out/view responsibilities.
Coordinate that producer/consumer directly; do not invent lowerer-local alias or hazard analysis. Consume the
retained destination in _instruction and source views, preserving the relation's real read/write witnesses.
