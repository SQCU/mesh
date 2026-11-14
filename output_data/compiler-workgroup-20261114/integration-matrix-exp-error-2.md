# Complete the functional assertion's dependency operand

SAME593, same model/session. Your Python checkpoint is committed as e51793b1.
Root ran the actual public compiler caller after frontend41 repaired the dropped
write-only destination placeholder. Read integration-matrix-exp-error-2.stdout.
The expected native RuntimeError occurred, but the first out.tolist() still saw
[[7,7],[7,7]] and the subsequent torch.isnan(out).all() was true: the GPU output
write finished after error return. This is not a pass. The prior source ordering
claim was too weak, exactly as the root review noted.

The caller's module returns aten.linalg_matrix_exp.out(a,out=out), with
a=[[0,1e6],[0,0]] and a separate output tensor initialized to 7. Torch capture
retains that output as an actual mutation target. Keep this caller and source
condition. Do not substitute an easier input-only or finite-norm case.

Inspect the actual functional assertion result/destination -> _Check -> native
construction dependence. Its dep_token presently reaches source GIVEN causes but
_asserted/_Check discard it. A copied destination may then share the assertion's
source vertex without a dependency into the check itself. We need the exact typed
dependency operand retained through that lowering, with no numerical dummy read,
extra synchronization beyond its actual producer, or global failure-behavior change.

Own operations.py's exact functional-assert/_Check handoff. Compiler64 remains
owner of numerical eig sections. Before edits to metal.py/native C or any shared
lower_metal representation, publish a precise small contract in your status so
root can pair it with executor/frontend. Inspect canonical Operation/Accesses/
SourceRelation; do not implement another graph solver, repair the malformed old
paths, or rewrite generated source. The emitted final error must follow the output
write and its completion, not merely queue it before cancelling.

Implement the Python side of the settled contract and hold if pairing is required.
No target run, shared build, commit or stage. Keep source citations and report
exactly what is missing. The intended result is the same actual output buffer
already containing NaNs when the native error reaches the caller.
