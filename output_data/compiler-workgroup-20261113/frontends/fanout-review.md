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

## Supervision

The first resumed turn spent about eight minutes reading/reasoning without a source edit or a concrete reported
blocker. The supervisor interrupted only that session's PID 79961, preserved its raw turn 11, and resumed the same
session/model in turn 12 with an explicit native-result-projection-first checkpoint. The first Plan field edit
then appeared. This records the observed stall and correction, not an explanation of the worker's motives.
