# Retained planner operations and requested spans

Style: [Google developer documentation highlights](https://developers.google.com/style/highlights).
Requirements: synchronization-signature item 1; compilation-composition 3 and 6.4.

The planner previously emitted a finite proof window, then discarded the operands needed to describe another
execution span. The caller composed that window with an independently requested host span.

`mesh-collective.c:plan_signature` now retains its ordered operation operands before expansion. The graph owner's
`repeated_operations` serves both the original proof window and `mesh_graph_signature_span`. Wait thresholds,
submission values, completion lead, startup vertex and original node/call order come from that producer.
`mesh_graph_resynchronized` retains its computed added-edge annotations; one extracted projector applies them to
newly derived finite signatures. Span construction does not rerun the planner, BSL delay selection or its proof
validation. Serialized records retain both operands and annotations.

The Python consumer loads that native archive, realizes the requested span, then uses existing native composition
with the emitted host signature. All owned results share the existing cleanup path, including constructor failures.
The source review compared every retained operand with the displaced per-execution formulas. It also checked that
`PLAN_COUNT+a` remains an added-edge identity, separate from transport publication/arrival/completion identities.

Old finite records still load unchanged. Requesting expansion without the newly retained operands reports missing
analysis data (`ENODATA`); it does not repeat the finite sequence or change numerical execution. No historical
record reconstruction is claimed.

Implementation: same Claude session `64ddaf6e-3a70-4b6a-9d0b-50691a84c6cd`, explicitly verified Opus 5.5; resumes
6–8 in `compiler/sessions.json`. Codex reviewed the producer, expander, proof projection, archive and consumer.
The API interruption and temporary Python-ABI hold preserved the session and source.

Private checks passed: canonical C objects and affected Metal consumer syntax; Python compilation; a constructed
plan-shaped record's archive roundtrip and requested spans 1, 4 and 6; old canonical records retain identical
serialization. Re-expanding the retained record at its original horizon produced the same archive bytes. The
constructed record checked startup versus completed raises and retained proof application. Composition with an
unrelated existing host record checked API ownership only, not real plan/host correspondence.

These are source review and build/ABI checks, not a machine-checked preservation proof or live transport evidence.
The real `mesh_plan` path is compiled; no bridge or E2B numerical execution is claimed for this unit. Canonical
shared rebuild and fresh ABI results are appended below when complete.
