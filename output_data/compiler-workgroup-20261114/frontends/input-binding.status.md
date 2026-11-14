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
