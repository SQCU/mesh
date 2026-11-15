# Compiler completion workgroup, 14 November 2026

This records the implementation work under
[compiler-workgroup-2026-11-13.md](compiler-workgroup-2026-11-13.md).
The change requests and their cited algorithms remain the requirements.

## Operator instruction

> then lets handle the concrete uncovered work next. if there are 'legacy' paths delete arbitrary substrings from their function headers by line so that they are unrunnable (producing python or c or whatever compile/interpreter errors) so there is no confusion about what their role in the codebase is: dead code. you do understand that we are not here to 'support code' within mesh or metal microbench, but to 'implement the basic algorithm specified by the change request documents and not something else'? there is no room for multiple methods here, inlined code, complexity, excess, or distractions.

> the requirement to make the old code uncompileable while leaving its structure behind for reference is literal and didactic; it is easily done and strictly communicates something important.

Mesh `b25ec14b` removes `def ` from the headers of `rdma/mesh_compile.py` `_replays`, `_bound`, `program`,
`load`, `_runner`; `compiler/mesh_compiler/lower_coreml.py` `lower`; and `examples/tp.py` `taken`.
Their bodies remain intact. Actual parsing fails in all three files. This is intentional.
Do not repair these headers, execute the old paths, introduce compatibility dispatch, or count their
feature coverage as a reason to preserve their execution.

The operator corrected root's unauthorized relocation:

> i didn't say that the obsolete code should be moved, i said it should be mangled s.t. it is impossible to compile or interpret it without obvious errors

> this is the fastest way to cause all callers to be updated immediately and the instruction was very pragmetic and literal

The malformed code must stay in its original files. Root's reference/retired relocation is reversed, including
the native signature wrappers in metal_recording.m. Production imports and builds must migrate to the actual
replacement owners. Do not move the old code, repair or hide its syntax, or parse around it at import time.
Required live algorithms move by responsibility into their new owners; this is not permission to clone a whole
old module under a new name. There is one live implementation of each required algorithm.

## Recorded state, 2026-11-14

This is a dated review of the recorded work, at mesh `774f08e4` plus the documentation revision `cdd44406`,
metal-microbench `1a2c5ade`, and the uncommitted native switch work identified below. The user reactivated the goal during this documentary revision; the worker sessions are held at the recorded
checkpoints while their work orders receive this correction. The request text specifies behavior; the observations below describe
particular revisions and inputs.

### Completed portions

| Work done | Recorded result | Scope of the observation |
|---|---|---|
| Distributed compiled transfers, saved program loading and coexisting published functions | [`distributed-multipage-1.json`](../output_data/compiler-workgroup-20261114/executor/distributed-multipage-1.json), recorded by mesh `cbced5b6` | Five peers returned 0; two prepared functions; 16 channels; 18,432-byte F16 shards cross the 16-KiB page boundary. Saved calls were restored using their compiled relations. The functions coexisted and were invoked sequentially. |
| Retirement of those published functions | [`distributed-multipage-1-retirement.json`](../output_data/compiler-workgroup-20261114/executor/distributed-multipage-1-retirement.json) | Five nodes returned to zero publications and full reported capacity with the same bridge processes. |
| Final native public bindings for state and aliases | [`bound99-outer-state.json`](../output_data/compiler-workgroup-20261114/executor/bound99-outer-state.json), recorded by `a3bcca69` | Two calls over two invocations, zero recorded numerical error, updated state, retained attribute/returned aliases, stride 2 and offset 3. |
| Withdrawn received-matmul special path | Integration item 5's named implementation and test are absent from the current metal tree | The deletion is completed history. Its old item is removed from the active request. |

The graph, allocation, routing, archive and bridge mechanisms exercised here are existing work. Their new compiler
consumers use those results. The multi-page observation happened to use contiguous physical offsets; it records
neither fragmented placement nor simultaneous invocation of the two functions. These limits describe that
observation, rather than creating new acceptance requirements.

### Domain of the remaining work

The assignment covers all linear algebra and tensor programs in AOT A1's domain. Known shapes and the statically
known outer call graph make placement, dependencies, allocation, transfers and schedules compilation operands.
The cited numerical algorithms retain their internal conditions. Torch-export/FX and typed LiteRT inputs carry
different numerical bodies into those shared transformations.

The cases below locate current defects and unfinished connections; they are not an exhaustive operator list or
a restriction of supported programs. Each assignment names the properties `P` under which its transformation
composes with the next stage. Existing client instances supply observations of those properties. Generality is
reviewed in the lowering rules and their composition, rather than inferred from a model's successful execution.

### Remaining implementation and integration

| Unit | Work remaining | Basis and clauses |
|---|---|---|
| Native consumers of numerical conditional regions | Complete the compiled GPU span selection and indirect arguments, one produced span plan consumed by command emission, and joined completion; lower the same conditional bodies for CPU and ANE | [`native-switch106.status.md`](../output_data/compiler-workgroup-20261114/executor/native-switch106.status.md), currently titled native-switch112, lists unfinished source. Source scalar branches/region ownership are committed in `d4e7f120`/`a00eb204`; matched native execution is unfinished. Composition 2/4–7, capture 9, source lowering 2/6. |
| Numerical target coverage | Complete source/primitive lowering and its composition across AOT A1's domain on each assigned processor. Located unfinished cases include ANE exact word operations, binary64/complex lowering, and numerical conditions, convergence and driver semantics | [`ane-bitcast-review.json`](../output_data/compiler-workgroup-20261114/executor/ane-bitcast-review.json) records construction status 22 before invocation; [`spectral-contract-review.md`](../output_data/compiler-workgroup-20261114/compiler/spectral-contract-review.md) records source-review limits. Source lowering 5/6 and composition's software arithmetic. |
| Public output geometry | Connect and observe the committed source-base/stride/offset realization with the matching native program through the existing Crossing caller | `0b638912`; [`materialized-output.status.md`](../output_data/compiler-workgroup-20261114/frontends/materialized-output.status.md). The source edit is committed; its native consumer observation remains pending. Capture 1/7/9, collective-values 3/6, AOT A1/A9. |
| Execution-cost inputs | Complete actual-work and repeat-multiplicity coverage in the existing observation/cost path | `7d34092d` already retains supplied fixed work alongside share-dependent coefficients. [`integration-cost-filters.status.md`](../output_data/compiler-workgroup-20261114/integration-cost-filters.status.md) records that correction. AOT A3, unequal-shards 6/7/14, integration 3. |
| Signature and domain-wide integration | Carry new emitted operations and completion through the retained signature relation and the shared compiler consumers for Torch-export and typed LiteRT programs. Existing clients observe explicit properties of those transformations | Synchronization 1–6, integration 1/2/7, composition's domain-wide integration clause. Matching the native consumers and establishing lowering coverage remain outstanding. |

CPU native execution already exists. The first row concerns its conditional-region consumer, together with the
GPU and ANE consumers. The remaining work is identified by the produced value and its consumer, rather than by
inventing a separate CPU runtime or reimplementing the transport algorithms.

### Request accounting

| Request | Completed work or retained foundation | Remaining assignment |
|---|---|---|
| AOT A1–A10 | Shared graph/routing/allocation machinery; compiled calls, bindings, publication and saved results exercised above | Native conditional consumers, current output geometry, cost coverage and connected client |
| Composition | Explicit staged driver, retained program/source relationships, graph-module consolidation | Complete the native result handoffs in the units above |
| Disjoint pages | Shared allocation, multi-page transfer and retirement mechanisms exercised above | Completion/lifetime consumption for the new conditional bodies |
| Integration finishing | Existing five-node bridge use; item 5's deletion completed | Domain-wide compiled execution, costs, feedback and property-directed observations |
| Page queue — completed | Resident bridge and compiled page transfers exercised above | Completed mechanism; new compiler consumers belong to composition/integration |
| Plan to bridge | Existing physical binding, transfer program and published-function lifetime machinery | New compiled consumers use the joint result; connected-client integration |
| Collective values | Existing collective/view/lifetime/cyclic-allocation mechanisms | Source output geometry and its downstream use |
| Synchronization signature | Existing signature/observer mechanism | Association with new emitted sites and observation of the connected program |
| Torch capture | Source/public-state bindings and saved call results exercised above | Conditional-body and output-geometry consumers |
| Torch source lowering | Source numerical decompositions and scalar Padé branches committed | Native branch execution and numerical coverage identified above |
| Unequal shards | Existing counts, placement and ratio solver; supplied fixed-cost correction committed | Actual work and multiplicity inputs to the existing cost/ratio solve |

Each active entry has the specific remaining assignment shown here. Completed clauses are historical
accomplishments, not repeated assignments. A request with all its work and recorded complaints resolved moves
to completed history. The operator's program-domain correction removes the E2B-specific initiative. Page queue is completed history;
new consumers use its delivered mechanism. Domain-wide compiler coverage remains an integration obligation.

### Assignment boundaries

Use the [composition dependency map](implementation-order-2026-11-09.md) and each request's cited mechanism.
The frontend/compiler/native owners agree on the explicit values passed across their boundary; graph routines
remain in the common graph module. A unit ends with its result consumed by the actual next stage and its
superseded live implementation removed as instructed above.

The existing Opus 5.5 sessions and Codex supervisors remain the workgroup recorded by
[compiler-workgroup-2026-11-13.md](compiler-workgroup-2026-11-13.md). Their next work orders consume this corrected
account and the operator's program-domain assignment. Commits remain ordinary working checkpoints.
