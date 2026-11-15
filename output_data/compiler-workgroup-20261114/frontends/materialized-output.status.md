# Source output realization

SAME974 sessions75–77, terminal/HOLD. Only `placement.py` changed. Supervisor
AST and `git diff --check` passed; source SHA256
`18ec833bf191184d545c5ab272c42ee80f4e647643468ea437f83e8c548c602e`.

The existing `crossing-output-1.json` records a logical 16-byte broadcast
(5,8)/(0,1) paired with an 80-byte contiguous CAT by a scalar delta. The prior
684c7b42 shortcut removed a redundant CAT only. The current change retains the
original alias parents/operands before placement, consumes canonical
`mesh_graph.postorder` and `forward`, and realizes shared source bases once from
the existing placed result and whole-value cache. Original input aliases keep
that input. Mutation outputs keep their original destination path. Other outputs
retain the original output transfer.

Required fresh-base geometry follows Torch2.14
`_inductor.ir.ExternKernel.require_strides`: compact zero-stride dimensions,
copy into the original geometry, expand. Its storage is allocated at the source's
full extent, with the source offset/strides applied by `aten.as_strided`. Existing
CLONE.out/typed copy owns numerical work. Canonical FX DCE removes unused paths
before transfers while retaining source effects and input-part endpoints.
No new native interface, archive record, or runtime decision was added.

Root and executor reviewed the source approach. SAME974/session79 reviewed the
actual Output→Slot→resolved→saved-view consumer connection without editing source.
The existing public two-call column-sharded instance then passed on Mesh122df646 /
Metal48074624 with the matched native117 libraries (executor/native117-build.json):
direct and public save/drop/load errors were [0,0], exit0/stderr0. The first result
kept shape[5,8], stride[0,1], offset0 and its 16-byte backing; the second result
kept shape[5,8], stride[8,1], offset0 and aliased caller y's 80-byte backing.
Both exactly match eager geometry. Reload retained the signature, used fresh
first-result storage and left the prior result unchanged. Two calls were restored
without relation solving; archive size140199 bytes. Before/after source and library
hashes are identical in materialized-output-before.json and -after.json. The
numerical/layout result and stdout/stderr are materialized-output-native.*.

This observes the general public layout/alias contract on one static instance,
including saved invocation. It has no mutation or nonzero-offset result; earlier
mutation/view observations remain separate evidence. It does not exercise the
unfinished conditional native execution. Session79's draft evidence expectations
were corrected by supervisor against actual OVERLAP0 source: output0 is the
broadcast and output1 aliases y. Its outside-workspace command read was denied;
the original command and actual archived continuation were reviewed directly.
No production edit was required after0b638912. Command/order and worker records
remain in the metal workgroup frontends/torch/materialized-output-* and sessions75–79.

The first observation after the original compiler/result replacement used the exact
preserved command against fixed native137 (build source 9a5efe31/graph e4b765af),
Mesh e908087a with archive eb56bbcb and emitter caller 73296c4d. It exited 1 during
import after 3.66 s: lower_inductor.py's direct CppTemplateKernel import entered
Torch's lowering/flex/template import cycle and could not import CppTemplateCaller
from the partially initialized cpp_template_kernel. Capture, native invocation
and save/load were not reached. All recorded runtime source/dependency/library
hashes stayed unchanged. materialized-output-upstream-1.{before,after,result}.json
and its stdout/stderr preserve this first outcome; the prior native117 result
above remains historical evidence, not evidence for the replacement. Compiler68
owns the original initialization correction; the same caller is retained.

The unchanged retry on 8385bd92 passed import, capture and upstream lowering,
then exited 1 during construction after 2.86 s: lower_metal.planned rejected the
original provenance-comment string for mesh.recv_strided as an unsupported
wrapper instruction. The canonical upstream make_comment/CommentLine owner
already exists; compiler68/66 owns that producer classification correction.
No numerical execution or save/load was reached. Runtime inputs remained
unchanged; materialized-output-upstream-2.* records the exact command hash,
source/library hashes and traceback. The command itself was not modified.

The unchanged third observation on f47fa9d8 passed the typed provenance-comment
handoff, then exited 1 during receive-result projection after 2.75 s.
lower_metal.called indexed node.outputs[0], but the original single-Tensor
FallbackKernel result had no multi-output child list. Compiler66 owns the
complete original result/binding correction. No numerical execution or save/load
was reached; runtime inputs remained unchanged. materialized-output-upstream-3.*
retains the actual traceback and the same command/native137 identity.
