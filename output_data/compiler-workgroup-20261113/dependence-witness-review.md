# Dependence witness to native wait — 2026-11-13

Requirement: compilation-composition change request 6/6.4. Status at this source checkpoint: graph producer
reviewed; paired native consumer and coordinated build remain OPEN. This is one connection, not separate
progress counts for its API and consumers. No unrelated module is a gate for the local claim.

## Contract

The existing exact strided conflict owner retains each meeting view pair as original input-array indices
`from_view,to_view` on its existing cause/conflict records. Sorting retains distinct endpoint pairs after the
existing cause keys. Reversed self RAW/WAR swaps the indices; GIVEN edges retain `given` and use SIZE_MAX
for both witnesses. The distinct vertex-pair relation is unchanged. No Metal side policy enters the graph owner.

The recorder consumer retains a side per input view, projects actual witnesses into each vertex pair's host-wait
requirement before compression, and consumes that property through the existing delayed-edge projection.
`mesh_graph_delayed` now optionally returns its selected original edge index; it does not repeat selection or
reconstruct an identity. MetalHostSide's repeated exact intersection is to be deleted by the paired consumer.
Supplied plan edges retain their structural endpoint roles; they are not fabricated memory witnesses.

Compiler artifact witness indices refer to the persisted ordered p.views array: Artifact builds p.records in that
order, Records saves the views, and load reconstructs the same order without a solve. The raw-record format
must be bumped in the paired metal commit. Native preparation may discard temporary views after consuming
witnesses into the retained wait classification.

## Review and evidence

Codex reviewed the three graph files and the direct API outputs. The SAME Claude numerical session
64ddaf6e-3a70-4b6a-9d0b-50691a84c6cd resumed as PID3734; init and final model usage report claude-opus-5-5.
Local raw log: compiler/numerical.resume3.stream.jsonl; exact work order: compiler/dependence-witnesses.md.
The worker is terminal/held and did not stage, commit, or rebuild shared libraries.

Private canonical graph object compilation with -Wall/-Wextra and Python syntax checks pass. Direct C calls
confirm original indices after hull sorting, three distinct same-key/hazard witnesses, GIVEN sentinels, and
self RAW1→0/WAR0→1 orientation. The delayed selector returns original indices with unchanged source/slot
results when its new output is NULL. A private ctypes check confirms Cause size32 and preserves the existing
Python cause-only diagnostic projection, including distinct supplied edge identities. These are small API checks,
not formal proof or production numerical/performance evidence.

The consumer and coordinated canonical build remain to be reviewed at this checkpoint. Endpoint-aware
transitive reduction is not implemented by this unit: retaining selected-edge witnesses does not itself prove
that a removed GPU→CPU obligation is covered by an alternate GPU-only path. Eliminating already-supplied
internal conflict analysis and local channel planning are separate subsequent units.
