# Dependence witness to native wait — 2026-11-13

Requirement: compilation-composition change request 6/6.4. Status: the graph producer, paired native consumer,
ABI propagation and coordinated build are reviewed and connected. This is one connection, not separate
progress counts for its API and consumers. No unrelated module is a gate for the local claim.

## Contract

The existing exact strided conflict owner retains each meeting view pair as original input-array indices
`from_view,to_view` on its existing cause/conflict records. Sorting retains distinct endpoint pairs after the
existing cause keys. Reversed self RAW/WAR swaps the indices; GIVEN edges retain `given` and use SIZE_MAX
for both witnesses. The distinct vertex-pair relation is unchanged. No Metal side policy enters the graph owner.

The recorder consumer retains a side per input view, projects actual witnesses into each vertex pair's host-wait
requirement before compression, and consumes that property through the existing delayed-edge projection.
`mesh_graph_delayed` now optionally returns its selected original edge index; it does not repeat selection or
reconstruct an identity. MetalHostSide's repeated exact intersection is deleted by the paired consumer.
Supplied plan edges retain their structural endpoint roles; they are not fabricated memory witnesses.

Compiler artifact witness indices refer to the persisted ordered p.views array: Artifact builds p.records in that
order, Records saves the views, and load reconstructs the same order without a solve. The raw-record format
is bumped from v28 to v29 in the paired metal commit. Native preparation may discard temporary views after consuming
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

Endpoint-aware
transitive reduction is not implemented by this unit: retaining selected-edge witnesses does not itself prove
that a removed GPU→CPU obligation is covered by an alternate GPU-only path. Eliminating already-supplied
internal conflict analysis and local channel planning are separate subsequent units.

## Connected checkpoint

Graph source: mesh e4b2a96e. Native consumer: metal-microbench f61a2ff3. The consumer's exact side label follows
each appended access; all GPU→CPU witnesses are ORed before pair compression. The existing delayed selector
returns the same pair identity to the final fence/host-wait choice. Supplied host PART endpoints map to the existing
start vertices rather than losing host/work identity through modulo. Codex's final correction preserves a
same-endpoint supplied edge when its delay is nonzero; only zero-delay reflexive edges are omitted.

Frontends supervisor strict private syntax checks pass for recorder, IR and archive. Canonical
`make .build/e2b_replay` completed exit0, rebuilding mesh, IR, recorder, provider and the actual E2B caller.
A fresh process using the rebuilt canonical libmesh passes direct witness/ABI checks, the unchanged Python
cause-only diagnostic result, and delayed edge identity with both present and NULL output. No private library
is substituted in that final check. Logs (metal-microbench, local):
`output_data/compiler-workgroup-20261113/dependence-witness-build.log` and `dependence-witness-abi.log`.

This closes this local structural connection by source review and actual compilation/API evidence; it is not a
machine-checked proof. No E2B execution, numerical equivalence, throughput or whole-repository completion is
claimed. The initial relation solve (including its existing capacity retry), endpoint-aware reduction, and eliminating
already-supplied internal conflict analysis are not replaced by this unit. Both implementation sessions remain held.
