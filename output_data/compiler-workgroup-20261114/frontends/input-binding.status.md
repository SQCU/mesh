# Original input binding

SAME974 sessions69–72 are terminal/HOLD. Supervisor AST4/diff checks pass; actual
Torch2.14 FX placeholder producers, graph_signature._make_argument_spec593–632
and export literal checks utils487–504 were inspected.

14be9344 retains original Signature/TreeSpec through runtime/archive. The public
reversed-kwargs Difference/save/drop/load caller passed with errors0/0 and exact
interface (Metal fb5f004b).

9f24bc96 uses canonical PHWithMeta to retain each Signature-flat leaf ordinal; FX
interpreter inputs consume those ordinals, and user_inputs retains original leaf
order. Calls delegates FX variadic names to the original parameters. Runtime
prepares input positions and returned indices/constants/spec once, gathering fresh
results at invocation. 675e0287 discharges defaults from internal flat placeholders
while the original public Signature still owns defaults. This fixed the actual
PyTreeCodeGen syntax failure. The next original caller reached native binding;
executor97d69ebc now retains original logical→compiled placeholder correspondence
for both typed bindings and constant association.

Exported.specialized retains actual ConstantArgument name/value records. Captured
projects them through the original FX argument TreeSpec before replacing Call
arguments; outer user Node identity selects the public records. No scalar-value
matching, source replay or extra graph traversal is used. The archive saves these
records in the existing state payload. Arguments prepares their flat indices and
uses the exact upstream int/float/str type/value/NaN checks; mismatches raise before
native invocation. No tensor/range/SymInt checker, None policy or guard registry
was added. State and tensor alias owners remain unchanged.

The single minimally adapted variadic/default Difference caller and public archive
retry is next, followed by one changed scale on that same restored Program. The
new literal/runtime observation is pending; previous two attempts stopped before
native invocation with unchanged source/library hashes. Metal36761cf9 preserves
the exact command and failures.
