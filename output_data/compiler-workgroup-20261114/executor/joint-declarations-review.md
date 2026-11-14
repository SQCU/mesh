# Joint declaration and archive consumer checkpoint

2026-11-14. Executor review of SAME archive047e12be's source, following phase checkpoint78bb14a4.
This is source/ABI evidence; current joint client execution awaits the matching native build.

The actual compile_program path now obtains interface slots and passes original call bindings once to
schedule.declare_program. Per-call instances retain original source-function identity separately from plan instance
identity. Intrinsic wire declarations are applied before structural lowering; all resulting Actors/Storages feed the
same solve_program result. Release and sending marks carry `(owner call, transfer)` through preparation and archive,
so an effect's emitting row cannot silently replace its protocol owner.

Mapped arena backing and MetalBufferBorrow retain the same underlying preparation owner, rather than a Target cycle.
The archive's duplicate Plan JSON and reconstructed mesh.Function recipe are removed. A restored Scheduled.plan
references the restored Target's canonical joint result. Current local save/load uses the whole native Local owner.
The WorkAxis24, WorkExtent24 and WorkTerm sizeof(mesh_graph_work)+48 bindings match the revised native header and
are registered only when the native compiler loader is used.

Supervisor AST parsing and git diff --check pass for native.py, metal.py and archive.py. The worker also checked
actual declared symbols and plain ctypes sizes without running a stub consumer. Prior stub scripts are not evidence.

Remaining producer connections are explicit. Cross-call ports need canonical full backing extent/offset and demand
release grouping, distinct from logical payload bytes and caller-owned roots. Remote outer results need an actual
typed wire-to-result copy site before Actors and the sole solve. Network solve/load is unfinished. The symbolic
metal_ir_value_planned correspondence is not yet consumed by ExternallyVisible/Realize, which currently read physical
placed state; no physical placement is inferred. Different genuine construction operands need the existing typed
source clone owner instead of specialize on an unconstructed program. These are ongoing implementation obligations,
not accepted domain restrictions. The ordinary local no-port client can exercise the connected path independently.
