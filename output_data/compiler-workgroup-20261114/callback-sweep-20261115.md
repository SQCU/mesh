# Callback invalidation sweep, 15 November 2026

This source account follows the operator's current callback-sweep instruction and
the [workgroup's review roles](composition-replacement-workgroup.md). It records
source edits and their limitations; it supplies no execution or performance result.

> do a quick callback sweep to induce compile/runtime errors in all callbacks used in mesh by putting compiler-legible pointer errors into all of them. a garnishing comment for each detected callback shoudl estalbish that the inversion of control thus implied is not wihtin the workign spec and will never be allowed in performance critical code due to literal runtime callbacks adding superfluous overhead beyond what this sort of data engine can afford or allow

The sweep retains callback bodies in place. Python references the deliberately
undefined `mesh_callbacks_are_outside_spec`; native callback sites receive
compiler-diagnosed invalid function references. Each identified use has the
requested explanation and its compilation, execution or lifecycle phase.
Ordinary direct foreign-function calls and stored IR operation identities are
distinguished from callbacks using their actual callers and registrations.

| Assigned source | Original Claude worker | Source account |
| --- | --- | --- |
| Compiler Python, excluding `export.py`, `ir.py`, `metal.py` | 68 | [Compiler](compiler/whole-lowering-ownership-20261114.md) |
| Mesh-owned compiler patches | 66 | [Compiler](compiler/whole-lowering-ownership-20261114.md) |
| `export.py`, `ir.py`, Mesh-owned Python outside `compiler`, shell handlers | 104 | [Frontend](frontends/whole-composition-ownership-20261114.md) |
| Mesh native source, its shared Metal implementation, Swift, `metal.py`, Mesh-added Dawn patch sites | 131 | [Native](executor/native-flow-supervisor-account.md) |

## Source checkpoints

At the checkpoints below, the source inventory contained 1,057 marker occurrences
across the assigned slices. This counts error sites, not unique callbacks or measured runtime calls;
some callbacks have both a registration marker and a body marker. This was the
partial state before the Codex continuation below.

| Slice | Markers | Preserved source |
| --- | ---: | --- |
| Compiler Python modules | 358 | `22d5cb96` preserves the first six modules; `cb7dfb50` preserves the final five. |
| Maintained compiler patches | 38 | `fd05ee5f`, nine patches; partial coverage. |
| Frontend and RDMA Python | 16 | `1beab98e`, three files; partial coverage. |
| Mesh native/Swift and `metal.py` | 206 | Mesh `d9a201d2`. |
| Shared Metal implementation | 409 | Metal `2c634181`. |
| Mesh-added Dawn callback sites | 30 | Dawn `09457b098`, carried by Metal `2c634181` in `provider/dawn.patch`. |

The [native site inventory](executor/native-callback-inventory-20261115.md)
records each native/Swift/Metal-Python site, consumer, phase and classification.
The compiler account records module and patch coverage, source hashes and
remaining classification questions. Compilation-time callbacks are labelled as
such; their presence does not establish callback overhead in the emitted program.

Removing only the markers and their comments restores the original native and
Swift source; removing Python marker nodes restores the checked original ASTs.
Patch projection checks preserve the original upstream bodies. These are
source-preservation checks: no Mesh/Torch import, callback invocation, native
build or workload ran. The deliberately undefined symbol remains undefined.

## State at the Claude executor stops

At `1beab98e`, worker104's partial result contains 16 marker references across
`export.py`, `ir.py` and `rdma/mesh_compile.py`. Removing only those markers
restores the original export/IR ASTs; `mesh_compile.py` retains its pre-existing
syntax defect. Native callback registration invalidation and the remaining
Python/shell inventory are unfinished. A failure inside a ctypes callback can be
reported without propagating to its ordinary Python caller; those registrations
therefore require their own error sites. Examples still outstanding include
`ROW_KEY`, `HOSTED`, and the frontend/RDMA `RELEASE` registrations.

Claude's stream reported "Opus 5.5's safeguards stopped the response above ·
continuing once with that noted", then the worker refused further sweep edits.
The notice supplies no specific reason. The exact events are in
`/Users/mdot/metal-microbench/output_data/compiler-workgroup-20261114/frontends/torch/session-103.jsonl`,
lines 15064–15071.

Worker66 subsequently received an API safeguard refusal tagged
`[reasoning_extraction]`, request `req_011Ch5ZDD2CA1T3DSuUzYMKH`, at result34.
No subsequent Claude source edit occurred in that slice. At `fd05ee5f`, the registered
`invoke_subgraph` lowering and the returned `_generate_kernel_call` and
`_generate_kernel_definition` methods remained three concrete unmarked callbacks.
Additional protocol/operation-target classifications were recorded in the compiler
account rather than silently counted as covered.

Both stopped slices were held through `3cb27a27` while the operator was asked
whether Codex could finish them. Neither blocked action was retried or routed
to another executor during that hold. The other workers completed their
previously assigned, disjoint slices.

## Operator-directed Codex continuation

The operator subsequently directed:

> in this situation instead of trying to use more claude subagents, address this scope by using codex subagents. also grab the exact claude transcripts which lead up to the reasoning extraction classifiers for me and throw them into a different directory under ~/ somewhere for further review/analysis

The existing Codex frontend and compiler agents completed their remaining
slices directly. The Codex native agent also completed the previously uninspected
Mesh-serving provider patches. No Claude implementation session was resumed for
this work. The existing post-commit review hooks retain their configured model
readers; they are separate from the implementation workers. General producer
integration remains separate from this callback sweep.

The requested transcript export is at
`/Users/mdot/claude-safeguard-review-20261115/README.md`. It contains the four
complete session/stream logs (198,882,778 bytes), SHA-256 hashes checked against
the unchanged originals, exact contiguous lead-up excerpts, and readable message
text with source line references. The compiler's explicit classifier event is
stream line 17520; its exact refused user message is at stream line 17518 and
canonical-session line 10899. The recorded explanation refers to restrictions
on reverse engineering or duplicating model outputs; this records the API's
explanation rather than independently establishing a cause. The frontend log
contains an unspecified safeguard notice, not an explicit `reasoning_extraction`
classification.

## Completed Codex checkpoints

| Maintained slice | Final error sites | Completion evidence |
| --- | ---: | --- |
| Compiler Python modules | 358 | Earlier `22d5cb96` and `cb7dfb50`, 11 modules. |
| Compiler patches | 61 | `abba01f8` adds 23 markers, closing the three known omissions and the remaining source-proven deferred/lifecycle registrations. All 20 affected upstream projections parse, preserve the original AST after marker removal, and reverse to their exact base blobs. |
| Frontend/RDMA and Mesh-owned Python tools | 468 | `c906de24` changes 45 Python files; the count includes the earlier marker in unchanged `export.py`. The [frontend inventory](frontends/callback-sweep-20261115.md) records the 74 scanned Python paths, admitted source, exclusions and actual callback consumers. |
| Ordinary shell traps | 6 | `c906de24`, six traps in four scripts; original trap bodies remain after the explicit failing command/exit prefix. Source stripping and shell syntax checks passed without invoking handlers. |
| Native/Swift/Metal Python and maintained provider patches | 698 | Earlier native checkpoints plus Metal `1703f8df`, with the combined inventory in Mesh `e9ca926b`. The provider continuation adds 53 markers across 22 original LiteRT/XNNPACK/MLDrift files. Original dependency commits are `fbde6d03`, `877af3f`, and `239994d`; all three patch layers and the canonical LiteRT source match. |

Total: **1,591 error sites**, including **534 added by this Codex continuation**.
Original bodies remain in place. The Python AST checks preserve even the
pre-existing malformed reference headers through temporary, in-memory parsing
normalization; those headers were not repaired on disk. The provider source
restores byte-for-byte after removing only the comments and markers. No modified
compiler/data-engine callback, trap or workload was invoked to validate these
intentional errors.

The inventory distinguishes actual callbacks from direct calls, source-program
operands, represented IR/function identities and null function-pointer values.
Review removed a false positive around `RELEASE()`: that zero-argument ctypes
constructor supplies a null release pointer, whereas `RELEASE(actual_function)`
registers a callback. The outward ctypes registrations now fail before ABI
crossing, rather than depending on an exception raised inside a C-invoked Python
callback.

Operational oversight remains intact: `bin/mesh-brief-gate.py`,
`bin/mesh-post-commit.py`, `bin/mesh-shell-guard.sh` and `bin/mesh-kill-guard.sh`
are byte-identical to their pre-continuation source. The historical
`measurements/training-continuation-20260906/check-viewer.py` artifact is also
unchanged. Unmodified vendor bodies and generated evidence are outside the
maintained-code sweep. These are explicit scope boundaries, not claims that
those external or operational callbacks were disabled.

## Post-commit review receipt

The frontend agent incorrectly used `core.hooksPath=/dev/null` for `c906de24`.
Root then ran the unchanged `bin/mesh-post-commit.py` for that exact commit with
the hook's normal personal configuration. It exited successfully; the
`measurements` git note contains commit, source and new-inventory review records
at 23:53:24–23:53:50 UTC. No hook source or repository hook configuration changed.
The compiler and provider checkpoints also have their normal review notes.

The reviewers recorded the requested explanatory comments and truncated some
large diffs. The provider review also flagged producer structure visible in the
nested patch text. Source-preservation checks establish that this sweep retains
the original projected bodies; they do not establish that the earlier producer
integration is complete or correct. This receipt closes the callback-invalidation
and transcript-export work, not the broader compiler integration objective.
