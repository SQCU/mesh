# Correct the retained source alias handoff

Continue only the phase composition assignment. The previous turn finished without consuming the alias review
in integration-inbox.md; its source-ready claim was not accepted.

`export_stage` currently gives distinct manifest Operand objects distinct phase backings even when the source
declares one ValueID or one storage class. `append_stage` then overwrites repeated values[source_id]. Fix these
two connected sites. Read the actual source `metal_ir_value_info.storage` and the canonical `metal_ir_value_view`
operation, which redirects every member of the old storage class. Carry the source-declared alias correspondence
when establishing phase bindings. Keep each logical shape/layout/offset; do not substitute equality of manifest
objects for equality of source values/storage, and do not reject such aliases.

Retain the actual source-value -> phase-backing map produced during export/binding on ModelCall and pass that
same map to append. Do not reconstruct it a second time by looping over interface strings or allow last-write-win.
Its lifetime ends after append; the retained source body ownership remains as already agreed. Use existing typed
source IDs and canonical alias operation, not a new alias solver, parser or type system.

Only tools/e2b_replay.m changes. Finish with the current canonical header syntax check and explicit review of
repeated source ValueIDs and distinct source views sharing storage. No new evaluator or broader cleanup. Read
integration-inbox.md again before reporting readiness. The supervisor will run the connected build.
