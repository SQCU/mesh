# Fixed native entry result ownership

2026-11-14. Executor source review, SAME native session e48f4bc7. The corrected source is held and privately syntax-checked; no runtime
claim is made here. Phase construction is separately checkpointed in metal dfe9efd1 / mesh 78bb14a4.

The composed CPU module retains its original HostEntry boundaries. Transport projects those boundaries into fixed
first, compaction, later and publication sites; the later numerical body is cloned through the existing typed Clone
owner rather than constructed again. ConstructKernels assigns numerical kernel/member identities before projection,
and the completed dispatch-to-numerical table is saved directly. Installed Dawn's MSL raise owner skips
SingleEntryPoint when host_cpp is enabled, so one generation can emit every selected entry.

Two concrete ownership defects were found in the first source draft. CompileTask copied the same source, binary and
object into each entry, and LowerProgram compared object bytes to rediscover their common result. Replacing that
with an owning kernel ordinal in Emitted was also rejected: an Emitted result is reused by call specializations,
whose inserted transport/check kernels can have different ordinals. The required result is one shared compiled-output
handle across entries and instances. A final archive table index serializes that handle; it is not its live identity.
Per-entry symbols and task pointers remain separate. The same shared owner must retain the module's Counts summary,
from which the selected entry reads its counts, rather than repeat module analysis or count every sibling entry.

Local scope comes from the original fusion decision. Transform calls Internal for one admitted fused group and then
creates that group's HostEntry; ComposeCPU concatenates those groups without numerical fusion across entries. Project
must consume the retained local slice of each entry, rather than select locals anew by accesses and overwrite
Value.internal for whichever entry happens to be last. Proved's original reader-in-group condition remains the
source of that ownership fact.

A shared compile failure must not publish siblings as completed merely because their individual status slots are
zero. The actual shared output's completion governs publication. These are corrections at the existing construction,
emission, persistence and result-reader owners; no extra waits, retries or runtime dispatch loop are introduced.

SAME45 now implements the shared NativeBinary handle through Kernel/Emitted, identity-based linking and one archive
library table. Load reuses that handle for each exported entry. NativeBinary also retains the module Counts rows;
each entry reads only its selected root. HostEntry retains its original local slice, and shared compiler failure
prevents publishing any sibling as complete. Supervisor source review and diff checks pass. Private strict checks
pass for all METAL_IR_SRCS, native_bench, e2b_replay, metal_capture and metal_program.

Execution acceptance remains pending the matching canonical build and current real local client/archive observation. The malformed obsolete source remains at its
original paths; the live build names metal_capture.m and metal_program.m explicitly.

Native source checkpoint: metal e7db540c. The first actual current CPU client reached native emission and failed
because CompileTask read a HostEntry name after Generate renamed the typed function. SAME46 now retains the actual
entry names before Generate and passes them to runner emission and selected-entry counts. The generated source was
used only to diagnose the stale handoff. Private changed-unit syntax passes; matching rebuild and retry are pending.
The failure log is executor/joint-cpu-ffn.stderr; no numerical success is claimed for that attempt.
