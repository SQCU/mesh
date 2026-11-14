# Network preparation and release record checkpoint

The current native Target consumes copy offsets, value sections and root sharing from its preparation owner's
complete `Solved` result. A restored network Target holds `schedule.SavedNetwork` bytes until `Published` loads the
whole result under that preparation's actual context. Independently prepared Bounds do not replace Target state.
The native signature is saved once with Target; Scheduled uses the same loaded record.

`mesh_metal_release` retains its owner call in the former padding word (32 bytes). Native Values own all declared
release records instead of overwriting one. Preparation matches `(owner, representative transfer)`, capture carries
that owner to the existing call-row resolver, and GPU completion publishes every retained record under its own
status word. The borrowed release getter replaces ValueInfo's singular release; ValueInfo is now 120 bytes. The
Python ABI and declaration constructor consume those exact fields. Artifact version 35 records release identities
and totals; prepared addresses are rebound. The same version is reserved for the separately owned Work tables.

These are source checkpoints, not network execution evidence. Native47 strict syntax passed for the live native
sources and tools; archive39/40 checked Python syntax and layouts without importing old native libraries. The
original malformed recording/module files remain unchanged. The previous local CPU, archive and mutation results
are recorded in `joint-local-review.md` and the frontend mutation observation; no new client run was made here.

The paired producer is still active: canonical early demand grouping, complete network save/load and
`Published.solution` must land before a matching build. Publication completeness also remains explicit: the old
host transport emits publication after each writer entry, while its arrival total counts GPU writers only. The
next unit must carry the actual writer completion scope through the existing arrival owner and signature, without
serializing independent writers or emitting an unconditional logical publication at every possible winning site.
The vector checkpoint alone does not resolve that defect.

Source owners: `native.py` Arena/Mapped/Target preparation and archive recipe; `archive.py` Scheduled signature
view; `metal.py` Release/ValueInfo/getter; `mesh-metal.h` release identity; native `Value`, Transport, capture and
archive consumers. Implemented by the existing e48 and 047 sessions and reviewed by Codex. Raw session records are
`native.resume47.jsonl`, `archive.resume39.jsonl` and `archive.resume40.jsonl`.
