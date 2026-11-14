# Exact numerical-worker continuation: retain native planner signature

Same session 64ddaf6e-3a70-4b6a-9d0b-50691a84c6cd, model explicitly claude-opus-5-5. Old assignments are complete;
this replaces them for this turn. Do not spawn workers, modify C algorithms, build shared outputs, stage or commit.

Only source path: rdma/mesh_graph.py. CR compilation-composition 3 and 6/6.4: graph owner produces its result once;
downstream consumes it. The actual existing producer is mesh_plan's mesh.Planned.signature. schedule.solved currently
frees it after taking completion/order/storage. The paired schedule worker will retain its canonical archive bytes.
Executor owns the archive/signature consumer. Implement the following THREE thin functions now, next to existing
record/from_record wrappers; do not inventory unrelated code:

- archive(made: Signature) -> immutable bytes, using existing mesh_graph_archive(size query, buffer, final call).
- from_archive(data: bytes) -> owned Signature, using mesh_graph_archive_read(data, len(data), &made).
- compose(a: Signature, b: Signature) -> owned Signature, using mesh_graph_compose(&a, &b, &made).

Bind exact ctypes prototypes from rdma/mesh-graph.h. Use the existing _check/free conventions. Successful returned
Signature owns its C arrays and caller releases it with existing free(); composition does not take input ownership.
No Python parser, solve, copied graph derivation or printer. No new object framework or wrapper dataclasses.

Reason for archive: record already preserves edge/least/alias; record_read rederives indices/counts. Archive/read
retains those derived native tables and counters too. State this accurately. Existing native compose knows canonical
processor identity and BRIDGE_START aliasing; its caller must align actual node/object coordinates. Do not add a Python
identity guessing policy or claim arbitrary signatures can be unioned.

Scope of the real plan signature is plan_signature in rdma/mesh-collective.c1068: BSL submission waits/counts_tell,
BRIDGE_START and already selected program edges. Its PLAN_COUNT+a is an added-edge ID, not a transfer index. This unit
does not generate per-transfer arrival/completion operations.

Run py_compile and focused existing-library ABI roundtrip/compose check using one actual or existing canonical
signature; no bridge/run/evaluator. Write a short status to planner-archive-bindings.status.md in this directory,
including exact files and check results. Hold source READY; supervisor reviews/commits named paths.
