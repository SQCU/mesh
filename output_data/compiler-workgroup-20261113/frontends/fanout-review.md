# SEND fanout: source ownership and native publication

2026-11-13. This reviews the source at `a11354fa`; the correction is in progress under the same Torch worker
`9740398d-f513-4c78-8578-f80b79fa2225`, explicitly reported as `claude-opus-5-5` on harness 2.1.321.
Applicable contracts: compilation composition 6/6.4/7, AOT A8/A9, and plan-to-bridge R2/R4/R5.

## Established failure

`lower_inductor.body` currently emits a completion wait, a `copy_` into the planned arena, and a publication for
each source SEND. The source `placement.insert_send_recv` creates several SENDs of the same tensor when it has
consumers on several processors. Their actual source node, view and collective identity are retained.

`mesh-collective.c::plan_function` does not treat each declaration as an independent physical send. Its existing
owner groups compatible declarations: same source/identity, exactly equal `from` view, matching production/read
weights, and distinct destinations. The resulting demand owns one supplied word and one completion-counter
sequence. `plan_walk` counts all native outgoing hops in that demand. Publication construction gives the
declarations the same supplied/completion operands and aggregate expected-send count.

`mesh_host_publish` releases the supplied word to execution + 1. The bridge's send loop acquires that word and
may immediately post any released demand hop. Therefore the first source SEND can expose all demand bytes before
the later duplicate `copy_`. The later SENT waits for reuse of execution minus copy-count, not for readers of the
current execution. Rewriting those bytes can race the native RDMA reads even if the values being rewritten are
equal. This is a source ordering defect; no bridge execution was needed to establish it.

Different demands can also share a Plan value and intersecting views. A single demand's completion cannot stand
for their independent readers. Conversely, equal Plan values or addresses do not establish equal source versions
or native actions. `ir.views` and the original numerical/mutation dependence relation remain authoritative.

## Exact handoff being implemented

The existing `mesh.State` and `mesh.Publication` mirrors expose the native result. `Plan.publications` retains
only source-applicable supplied/completion representatives, indexed by original transfer, obtained from those
actual records. Absent or nonlocal fields remain absent; null words are not an equivalence class. Addresses do not
enter the archive. Within one plan the completion word is assigned from `d.sending`, whose demand also owns the
expected `d.sends`, and every demand uses that plan's copy count. This establishes the counter contract for true
equal completion operands without repeating `plan_function`'s grouping policy.

The final-body transform consumes those results with the original source producer/version and exact sent views.
It removes duplicate writes and composes the new write/effect hazards through the canonical graph owner.
Independent demand completion obligations and source mutations remain distinct. It does not assume an allocated
root is fully produced or insert a whole-root/body barrier. The original per-SEND copy/publication sequence is
the deletion target, not an alternative execution mode.

The worker owns the bounded body/driver and Plan-result/archive connection; compiler and executor granted those
hunks. Raw turn 11 and its exact work order are under metal-microbench's
`output_data/compiler-workgroup-20261113/frontends/torch`. This document records the source finding and intended
connection, not acceptance of an unwritten consumer or numerical/throughput evidence.

## Connected source review

Producer checkpoint `01163a03` preserves the native representatives. The current consumer additionally retains
the RAW writer/read pairs from the original canonical source conflict call and remaps them through the existing
Node-copy correspondence. For the same exact source view, the complete RAW predecessor set is a conservative
version discriminator: an intervening conflicting write changes it. It is not claimed to be a unique reaching
definition. No second original-source conflict solve or alias-chain version inference is used.

`Plan.places` now keeps `(processor, root, version)`; source declared identity and wire value both consume that
version. Receiving assembly roots remain independent of remote source writer labels. Existing sender lifetimes
remain conservatively bounded by root production through the last local task; native transport far/start-lead
extends them in the existing planner. Compiler review confirms that these bounds contain every staging/publication
use. Distinct live versions reach the same existing allocation owner as distinct values; physical reuse is still
that owner's decision, not a shadow buffer scheme.

The body substitutes static write/completion/publication vertices, uses canonical conflicts only on the newly
introduced copy operands, and orders them with the canonical graph owner. A native completion/threshold has one
emitted SENT site. A supplied representative has one publication. Original SEND endpoints map to that site in the
signature, and original dependencies compose through contract/quotient without self-edges. Return is emitted after
all body effects. The first draft's publication flush before every RECV and compensating current-execution waits
were rejected and deleted; they would have preserved late redundant writes by imposing extra waits.

Supervisor syntax and diff checks pass. After the matching native graph/Planned rebuild, direct observations of
the existing body API established:

| Source case | Copies | SENT sites | PUBLISH sites | Observed order/correspondence |
|---|---:|---:|---:|---|
| Two identical views, same native demand | 1 | 1 | 1 | Both logical endpoints map to the same emitted publication |
| Same view, independent native demands | 1 | 2 | 2 | Both completion obligations precede the one copy |
| Two intersecting views | 2 | 2 | 2 | Both writes precede either exposing publication |
| Disjoint `[0,64)` and `[64,128)` views | 2 | 2 | 2 | Each completion/write/publication chain remains independent |
| Broadcast view, zero outer stride | 1 | 1 | 1 | Distinct dimensions and source stride rank agree |

In these constructed compiler-input cases, source multiplication precedes every blocking reuse wait, output is
last, graph lint passes and the composed signature has no zero-delay self-edge. The canonical source dependence
API on an actual in-place `add_` graph yields an empty initial SEND writer set and `{add_}` for the subsequent
SEND; the source node/view remains part of the copy identity. This demonstrates a changed version discriminator,
not a fabricated unique reaching-definition result. The first observation script incorrectly expected the pure
initial producer to appear as an explicit RAW cause; inspection showed it is an ordinary dataflow operand and
the assertion was not a compiler failure.

The existing CPU FFN client passed again on this final source (ROWS=2, K=32, N=48, LAYERS=1; maximum relative
error 0.0003511235991), recorded in `fanout-cpu-current.log`. It covers current ordinary compilation/invocation;
the focused checks cover transformed IR construction. No live bridge/fleet execution or throughput result is
asserted.

## Supervision

The first resumed turn spent about eight minutes reading/reasoning without a source edit or a concrete reported
blocker. The supervisor interrupted only that session's PID 79961, preserved its raw turn 11, and resumed the same
session/model in turn 12 with an explicit native-result-projection-first checkpoint. The first Plan field edit
then appeared. This records the observed stall and correction, not an explanation of the worker's motives.
