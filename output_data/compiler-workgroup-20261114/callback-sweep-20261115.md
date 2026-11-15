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

At `1beab98e`, worker104's partial result contains 16 marker references across
`export.py`, `ir.py` and `rdma/mesh_compile.py`. Removing only those markers
restores the original export/IR ASTs; `mesh_compile.py` retains its pre-existing
syntax defect. Native callback registration invalidation and the remaining
Python/shell inventory are unfinished.

Claude's stream reported "Opus 5.5's safeguards stopped the response above ·
continuing once with that noted", then the worker refused further sweep edits.
The notice supplies no specific reason. The exact events are in
`/Users/mdot/metal-microbench/output_data/compiler-workgroup-20261114/frontends/torch/session-103.jsonl`,
lines 15064–15071. That slice is held while the operator is asked whether Codex
may complete its remaining source-only edits. The other workers continue their
previously assigned, disjoint slices.
