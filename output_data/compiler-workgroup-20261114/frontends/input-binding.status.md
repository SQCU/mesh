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

The same variadic/default Difference caller and public save/drop/load passed on
395f8672: errors0/0, exact restored Signature/spec, and restored scale3 raises the
expected RuntimeError before native invocation. Exit0/stderr0,3.74s; source/library
hashes stable, native one-row reload without relation solve. Metal602da941 retains
the command/result/manifests. Actual native.py was uncommitted Bound draft SHA256
f6f7d498c9243c8a608e3b5821d73fd3fe4ea00023bfe095e7048bef23629279, not merely
logical-slot base97d69ebc; later Bound source is not covered. Earlier failures are
preserved in Metal36761cf9. No further frontend target is queued.
