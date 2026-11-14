# Literal dead source and signature operands

Same integration session 59368744-da56-491a-871a-1be809c0cc93, explicit claude-opus-5-5.
Read design/compiler-uncovered-work-2026-11-14.md and its quoted current operator instructions.

Mesh b25ec14b has deliberately removed `def ` from seven old headers. Do not restore those headers,
comment/string-quote the broken code, add stubs, or make the old bodies executable by another means.
Their source structure remains reference; production implements the cited algorithm, not compatibility.

First complete this mechanical ownership move, before any broader signature edit:

1. In rdma/mesh_compile.py move the ENTIRE existing bodies with malformed headers of `_replays`, `_bound`,
   `program`, `load`, `_runner` into `reference/retired/mesh_program.py`. Retain each header's syntax error
   and body exactly; remove those same spans from the production file. Do not copy or repair their logic.
   Their malformed definitions currently start at lines 7618, 8182, 8354, 8734, 8767 (line numbers shift on removal).
   Keep shared live primitives/ctypes/constants and other workers' numerical changes. If a needed live function
   is nested in one of these old closures, report the concrete consuming name, not a reason to reanimate it.
2. Move compiler/mesh_compiler/lower_coreml.py, already malformed at `lower(scheduled):`, in full to
   `reference/retired/lower_coreml.py`. No current source imports it (root's rg inspection).
3. Move examples/tp.py `Timed` and malformed `taken` bodies into `reference/retired/tp_cost.py`, preserving
   exact source and the malformed taken header. Frontend41 owns the replacement cost consumer; leave its caller
   for that owner. Tell root exact source ranges immediately before editing tp.py, so ownership stays exclusive.
4. Add a compact reference/retired/README.md quoting the current two operator turns, naming the source checkpoint
   and these deliberately invalid files. This is dead source, not a build input or runnable alternative. No new
   import-time checking machinery is requested.

Private checks: parse the actual live files you changed, and separately report the expected SyntaxError in each
dead reference. Do not compile/import dead reference source or invent a test that turns its failure into a gate.
Write integration-dead-source.status.md then HOLD for root's review/commit. No own staging/build during shared edits.

Next unit, only after root releases the source move: fix the shared native signature node correspondence.
MetalProgramSignatureSpans in metal_recording.m currently passes context->M->node (physical node ID), whereas
the planner's processor IDs are map indices. The existing mesh_nodes_read returns both nodes.node and nodes.self.
The current Target passes Planning.here correctly. Complete the actual wrapper/caller from this original map
operand. Use existing typed mapping; do not invent another graph, repeated node scan or reinterpret physical IDs.
The old _replays/_period path is DEAD and is not to be repaired. Retain native phases as the common emitter's
result for new consumers. Publish the exact wrapper/caller change before editing, private canonical syntax
check only, root/compiler coordinate shared builds.

Do not broaden into runtime, fleet, references elsewhere on disk, solver redesign or other units. Follow source
roles already cited in synchronization-signature and composition6/7; record concrete edits/checks, not promises.
