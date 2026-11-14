# Supervised replacement work, 13 November 2026

Style: [Google developer documentation style guide](https://developers.google.com/style/highlights).

This is a dated source and execution record. It is not a specification, an enduring claim about the repositories,
or acceptance of the whole compiler. The operator's instructions and cited roles are in
[`compiler-workgroup-2026-11-13.md`](../../design/compiler-workgroup-2026-11-13.md).

## Assignment correction

The first Codex briefs permitted repair of inherited lowering bodies. This was a supervision error: behavior of
an existing lowerer was treated as a design constraint. The operator corrected it. Root paused the eight owned
Claude sessions, checkpointed the initial patches, discarded the guessed gradient mapping and inherited numerical
repair, and issued bounded producer-to-consumer replacements. The discarded source remains in git history.

Each replacement states the source and its role, actual input/result, actual consumer, displaced body and changed
files. The runnable schedule-emit code supplies the composition example. Named literature and upstream compiler
implementations supply algorithms and vocabulary. Legacy source supplies caller/coverage/removal information;
it is not authority for the replacement structure. Codex inspects the implementation and directs corrections;
Claude does not approve its own structural similarity or choose an adjacent scope on completion.

All eight session initializations reported `claude-opus-5-5`, selected explicitly through `claude-personal`'s
profile executable. Session identities are in `sessions.json`; raw continuation logs remain locally beside the
supervisor records. Corrections resume the same identities. Only exact owned processes were interrupted.

## Narrowed units and observed results

| Worker | Assigned connection | Checkpoint / limit |
|---|---|---|
| Placement | Arithmetic range intersection, including both local index maps, into reshard | Mesh `1e82f8e6`; removes index-set materialization and per-consumer map reconstruction. Private bounded comparisons and syntax passed; no TP performance claim. |
| Scheduling | Already solved dependence endpoints and transfer strides into assembled body metadata | Mesh `ca1e4785`; AOT consumes that same metadata. No second schedule or transient Scheduled field. |
| Typed composition | Source typed program, values and owners into destination through the canonical Tint clone owner | Metal `76d81c90`; source remains reusable; no second instruction visitor. |
| Native binding | Completed AOT package into installed public loader / boxed native call | Mesh `76d6c351`; actual CPU package invocation passed. This is a numerical leaf binding, not the whole CPU/MPS/ANE program. |
| Artifact/archive | Selected numerical body into AOT files and retained bindings, then save/load | Mesh `ca1e4785`; restored example regions match. SEND/RECV remain an unevaluated recipe and whole-client connection is unfinished. |
| Torch frontend | Retained transfer endpoints through the device-copy consumer | Initial checkpoint `3a2aa8c1`; the separate guessed gradient mapping was discarded in `817ff6a3`. Full frontend replacement remains open. |
| LiteRT frontend | One opaque retained exported program/interface into the composition consumer | Metal `76d81c90`; old Declare/Publish entry points removed. Inherited numerical Graph/Builtin bodies are not accepted as replaced. |
| E2B integration | Retained interface binding map into phase composition, at original stage positions | Metal `76d81c90`; repeated source IDs and storage aliases retained. Further review found allocation still used per-Operand identities; correction assigned to the same session. |

## Concrete steering

- Rejected a guessed nonexistent native API and phantom calls to old MPS/ANE classes. The actual public package
  loader was connected instead; missing target coverage remains visible.
- Rejected a serial dependence chain substituted for the scheduled graph's relation. The scheduler now passes
  the relation it solved, through the actual copied Node map.
- Rejected a temporary dataclass shim as evidence of the real driver connection. Subsequent numerical-region
  evidence used the actual Scheduled interface.
- Rejected reconstruction of the E2B binding map at append and its last-write-wins treatment of repeated source
  IDs. The map now lives on ModelCall from export through append.
- Traced the alias relation beyond the IR and found distinct external allocations/host addresses remained.
  `integration-storage-assignment.md` assigns that exact consumer correction, with no invented alias rejection
  policy or copies merely to preserve the inherited allocator.
- Corrected Codex's own overbroad restrictions on ordinary local API checks and its insertion of a Torch-dependent
  leaf build into the generic transport build. Neither restriction was derived from the user's assignment.

These are observed edits and responses to steering, not an inference of worker motives.

## Evidence and remaining obligations

The canonical metal command `make .build/e2b_replay` passed for `76d81c90`, including compiler, provider and caller
linkage. Its log is in metal-microbench `output_data/compiler-workgroup-20261113/connected-build.log`.
Prepared E2B inputs exist. The local region query `/usr/local/mesh/bin/mesh-stat --map /mesh0` reported
`{"up":false,"errno":3}` and no mesh-bridge process was found. The actual caller attaches before export/composition;
this record therefore contains no E2B numerical execution or performance result. No bridge/service was started.

`executor/archive-region-check.log` records existing Engines numerical regions through native Model and archive
save/load: projection max absolute difference 0; FP16 FFN 0.00048828125; restored outputs bit-identical and restored
signature recomposition byte-identical. Scratch channel-value plumbing supplied inputs between these regions.
It does not establish whole-program communication, mutation/alias coverage or performance.

Whole-program native effects, local-channel storage/notification bindings, completed MPS/ANE target lowering,
replacement of inherited numerical bodies, production-client connection and end-to-end coverage/performance
remain obligations. Committing a working intermediate representation does not discharge them. The next native
effect seam identified in installed Torch is AOT's direct C-shim lowering; it still needs explicit invocation and
local-channel operands from the actual scheduling producer, not a new runtime interpreter.
