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

| Assigned source | Existing worker | Source account |
| --- | --- | --- |
| Compiler Python, excluding `export.py`, `ir.py`, `metal.py` | 68 | [Compiler](compiler/whole-lowering-ownership-20261114.md) |
| Mesh-owned compiler patches | 66 | [Compiler](compiler/whole-lowering-ownership-20261114.md) |
| `export.py`, `ir.py`, Mesh-owned Python outside `compiler`, shell handlers | 104 | [Frontend](frontends/whole-composition-ownership-20261114.md) |
| Mesh native source, its shared Metal implementation, Swift, `metal.py`, Mesh-added Dawn patch sites | 131 | [Native](executor/native-flow-supervisor-account.md) |

## Source checkpoints

The source inventory contains 1,057 marker occurrences across the assigned
slices. This counts error sites, not unique callbacks or measured runtime calls;
some callbacks have both a registration marker and a body marker. The complete
repository sweep remains unfinished.

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

## Unfinished slices and executor stops

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
No subsequent source edit occurred in that slice. The registered
`invoke_subgraph` lowering and the returned `_generate_kernel_call` and
`_generate_kernel_definition` methods remain three concrete unmarked callbacks.
Additional protocol/operation-target classifications remain in the compiler
account; they are not silently counted as covered.

Both stopped slices remain held. The pending operator question asks whether
Codex may complete their remaining source-only edits and supersedes the earlier
frontend-only question. Neither blocked action was retried, rephrased or routed
to another executor. The other workers completed their previously assigned,
disjoint slices; the native workers are idle and producer integration remains
paused.
