# Shared compiler integration work

Started 2026-11-01, after mesh `250e58dc`. This is the current implementation
handoff for the seven remaining integration items. It records work in progress;
a listed interface is not evidence that its consumers are complete.

## Required result

CPU, Metal and ANE are ordinary implementations of declared tensor work. The
compiler can assign a complete section or selected work domains to them using
explicit types, shapes, resource capabilities and configuration operands. Index
and activity pages remain numerical inputs. They can select masked work within
a compiled domain without choosing an executor during execution.

All frontends retain source and exact page relations until transfer effects have
been composed into the numerical work. Backend binding follows that solve.
Execution consumes the prepared pages and commands. Backend choice, layout
conversion, transport and source-fold choices are operands of the same
whole-function optimization; none introduces an execution controller.

## Work and file handoffs

| Item | Required connection | Current implementation scope |
|---|---|---|
| 1 | Every collective and frontend consumes the same typed page effects before binding | Native collective effects in `nccl-mesh.c`/`nccl.h`; Python adapter in `_stream.mm`/`page_effects.py`; common source/effect pass and native adapters described below |
| 2 | Submission and DMA posting consume exact compiled dependencies and flat operands | Recorder queue-event scope, host visibility operands and bridge descriptor traversal remain open |
| 3 | Ordinary source sections choose CPU, Metal or ANE through declared capabilities | `mesh_compute.py`, `compilation.py`, `native.py`, `algebra.py`, `engines.py`, ownership handoff |
| 4 | Existing whole-function feedback consumes backend and source-fold choices too | Extend the existing assignment operands; no second optimizer or runtime wrapper |
| 5 | Indexed contractions and partial-result fusion share numerical declarations | Generalize row grouping and finish/reduction composition through existing relations |
| 6 | LiteRT ownership and complete serving requests use the shared compiler | Remove outer model-role assumptions; propagate activity into crossings; compose prompt/generation and request state |
| 7 | Ordinary demos and hourlies establish whole-function behavior and performance | After structural integration, committed-source LSTM, sparse/indexed, native/LiteRT and prefill/decode programs; latency and useful throughput together |

The current parallel work owns disjoint boundaries:

- Native collective lowering owns the typed effect ABI and its exact source and
  result coordinates. It also removes the lower-and-record compatibility entry.
- The Python binding consumes those effects and removes its standalone
  collective execution bypass. WRITE and other collective kinds have one
  occurrence-lowering contract.
- The common native source-effect compiler owns projection and source effect
  emission. Swift/LiteRT adapters retain source until this pass; Python also
  consumes this pass. It replaces the Python-only solve, rather than adding
  another implementation alongside it.
- Backend assignment owns implementation candidates and source-region/provider
  integration. It preserves the differentiated source and page identities.

These are coordination notes for overlapping sessions, not exclusive ownership
of a repository. Commit notes should identify changed shared operands and which
consumers were migrated. Recheck current source before editing the same boundary.

## Item 1: the complete handoff

The native effect record names its collective call, source or result operand,
frame-local logical byte selection, publication/observation word and threshold.
The collective compiler resolves offsets for each collective kind. Frontends
must not reconstruct a source/result role from a kernel name or address.

The shared source pass consumes retained numerical declarations: source recipe,
backend, typed argument pages, and exact dispatch-to-page relations. It produces
observation and contribution tables, publication prefixes, source-input
dependencies and terminal result words. The same source emitter specializes
Metal, CPU and ANE entry/completion recipes. Only then are pipelines, provider
requests and replay records bound.

Swift currently records already-bound pipelines. LiteRT's prepared-provider ABI
also exports bound commands. Both need the retained source contract; adding a
callback to `CommLower` alone does not close the integration. Existing manually
supplied tile protocols are not substitutes for the general relation.

## Evidence required before completion

Source inspection must show every named frontend using the shared contract,
with no lower-and-record bypass or second source-effect solver. Build the actual
libraries and applications. Then use the ordinary committed programs to verify
numerical results and whole-function latency/useful throughput across declared
shapes, backend assignments and collective configurations. Source compilation
alone does not establish the end state.
