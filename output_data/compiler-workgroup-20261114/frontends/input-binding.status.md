# Original input binding

SAME974 sessions 69–70 are terminal/HOLD. Supervisor inspected installed Torch
2.14.0 FX create_args_for_root677–795 and _proxy_placeholder966–1035, reviewed
export.py/runtime.py, and passed AST plus exact-path diff checks.

The interface from14be9344 retains inspect.Signature and the original bound-value
TreeSpec through Exported, runtime and archive. The public reversed-kwargs
Difference/save/drop/load caller passed with both errors0, exact interface and
unchanged source/library hashes (Metal evidence fb5f004b).

FX's display names for variadics now map once to the original Parameter names at
the existing Calls trace boundary; super owns expansion/defaults. Its placeholder
order places keyword-only parameters before variadics, unlike Signature order.
PHWithMeta.ordinal records each original flat leaf; the canonical FX transfer
retains it on its Node. called assigns the same operand at placeholder creation.
capture uses those ordinals for FX interpreter inputs and retains user_inputs in
Signature-flat order. Existing archive user_inputs carries this correspondence.
The original state/get_attr and alias owners remain unchanged.

Arguments prepares live tensor/position and user-position tuples once. Invocation
uses canonical Signature.bind, apply_defaults and TreeSpec.flatten_up_to, then
strict cardinality binding. Program prepares returned indices, constant leaves
and output TreeSpec once: the non-output owner has the actual None leaf/spec.
Invocation gathers fresh native results; there is no repeated entry/OutputKind
choice, name lookup, new parser, compatibility mode or callback dispatch.

The next single observation minimally adapts that same Difference caller to
x,*rest,scale=2,**options and public archive restoration. Variadic ordering and
prepared output recipes remain source-only until that command completes.

## Session 71: defaults discharged from the bound flat graph

That observation, on source 9f24bc96, failed before any native run. Only the
outer signature had changed, to `forward(self, x, *rest, scale=2, **options)`.
GraphModule construction failed: FX `_PyTreeCodeGen` emitted
`self, x_1, scale_1 = 2, rest_1, = fx_pytree.tree_flatten_spec(...)`, which is
not a valid assignment target. Evidence: observe/input-binding-variadic-1.stderr
in the metal workgroup. Source hashes were unchanged, and there was no native
issue.

Cause: `_proxy_placeholder`'s `replace_ph` gives each flat placeholder its
parameter's default as `node.args`. `gen_var_bindings` removes annotations and
comments from the free variables, but not the default expressions. Under plain
CodeGen the same defaults can also put a default before a required argument.

Correction, in export.py only:
- `Signature.bind` and `apply_defaults` already supply every flat argument,
  keyword-only defaults included. The internal flat graph therefore has no
  omitted argument and no default to apply.
- The existing `calls` node pass sets each placeholder's `node.args = ()` once,
  before GraphModule construction.
- Unchanged: `_PyTreeCodeGen`, `process_outputs` and the TreeSpec owner; the
  ordinals and the name mapping; `called`, whose placeholders already have empty
  args.
- The defaults themselves, including tensor identity, stay with the original
  Signature in `Exported.interface`, which the archive and runtime consume.
- There is no codegen or text patch, parser, subclass, lazy wrapper or new
  rejection.

runtime.py is unchanged. The worker did not import, run or AST-check anything.
This stays source-only until the supervisor retries the same command once.
