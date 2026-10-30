# Ownership integration tickets

Recorded 2026-10-30 23:14 UTC. Source checkpoints: mesh `50c452ab` and
metal-microbench `8a4454f2`. This divides item 1 of
[Compiler followups](compiler-followups-2026-10-30.md) among concurrent
integrators. It records implementation work, not a new runtime interface or a
replacement for the [data model](data-model.md).

## Finish line

Ownership and partitioning must consume the retained numerical graph throughout
lowering. A requested output selection determines the necessary producer work;
that work determines the input pages. The same graph then emits that selected
work into its assigned pages. It must reduce the actual work and storage, not
compute a whole result and select from it afterward.

Source types, source identity, masks, reduction semantics, changing index values,
and consumer layout constraints cross each step explicitly. The compiler solves
their relations before execution. Execution consumes the resulting page operands
and commands. An index-producing operation returns ordinary page data consumed
by another operation; it does not select a different runtime memory system.

The existing relation algebra, page placement, numerical kernels, collective
compiler and recorder provide the mechanisms. Reuse those definitions. This
work removes repeated inference and operation-specific ownership implementations;
it does not add an executor, submission queue, allocator or runtime policy.

## Current starting point

- `native.prepare_graph` retains source values and their actual IR objects.
  `effects` exposes work/read/write records, reduction and finish domains,
  cached body nodes, masks and changing-index producers. The scheduler and
  wrapper survive physical binding.
- `compilation.normalize_source` applies existing tensor equations before
  ownership propagation. `ownership.propagate` consumes captured result types.
- `ownership.partial_expression` captures an existing partial reduction once,
  including state, its collective boundary and postprocessing. This helper is
  not yet connected to selected-domain compilation.
- `partition.lower_owned_graph` still constructs a local FX graph and has
  separate explicit pointwise and convolution ownership implementations. It
  does not yet use retained IR for arbitrary output owners and local views.
- `declarations.BoundGraph` binds ordinary numerical and collective sections,
  but its numerical sections still enter the compiler as source graphs.
- LiteRT static indexed views now share topological page-view lowering and
  ordered initializer ranges. Live index dependencies still need to reach its
  ownership and product declarations through the shared access contract.

After bridge alignment, actual immutable-view preparation retained reordered and
repeated indices and named parameter expressions with no copy commands. Encoded
gather preparation retained exact code/scale/zero selections and used the
existing decoder. O5's immutable-view connection is prepared; its live-index
connection remains open.

The checkpoint has source, build and preparation evidence, with limits recorded
in the followups document. The local bridge was subsequently installed through
`bin/mesh-user.sh`; its shared region now reports layout 155, matching source.
No numerical or whole-function performance result follows from that alignment.

## Shared handoff

These are the facts passed between tickets, using existing objects where they
already exist. They are preparation operands, not another runtime object model.

| Boundary | Producer supplies | Consumer must retain |
|---|---|---|
| Source to numerical preparation | Source value and result-field identity, captured type, explicit state operands, declarations and existing tensor equations | The original value-to-IR binding through decomposition and aliasing |
| Numerical preparation to ownership | Named storage and typed views; work, read, write, reduction and finish relations; actual masks and index-producing edges | Common coordinates, scalar units, value meaning and the original IR/body |
| Ownership to selected work | Each owner's ordered result selection; required work and reads expressed through those relations | Repetitions as uses of values, exact reductions, empty finish work, changing index operands |
| Selected work to placement/binding | The retained numerical body and selected domains; crossings; consumer layout, alignment and instance requirements | Actual selected dispatch work, existing page aliases and required conversion edges |
| Binding to execution | Prepared page operands, numerical commands and one-sided collective commands | The existing recorder and dependency graph |

A scheduling dependency is not a mathematical access relation. A typed value is
not its raw encoded dependency leaves. A reduction's name is not its full source
expression. These distinctions matter when two tickets meet.

## Claiming and handing off work

Choose a ticket and name the files and definitions you will change in the active
agent conversation. The assignments below are the current claims, not permanent
owners. Read other files freely. Arrange a handoff before editing a file another
active integrator has claimed; two tickets can share a file without two agents
editing it at once.

Send the adjacent owner the actual operand schema and consumer call site before
building against a new interface. When releasing work, report its commit or
still-uncommitted files, the consumer connected, preparation evidence, and the
remaining gap. Update the status here when the integration changes. Do not mark
a helper complete while its intended caller still uses the previous path.

Use ordinary commits with named files. Do not sweep another integrator's changes
into a commit, revert their work, or run installation from uncommitted source.
This document and agent messages are enough coordination; no lock service,
generated inventory, linter or execution-time bookkeeping is needed.

| Ticket | Current claim | Files principally edited | Depends on |
|---|---|---|---|
| O1 — Retained access and selected numerical work | `ane_constraints` | `torch_mesh/native.py`, `torch_mesh/operand.py` | Existing retained-IR checkpoint |
| O2 — Whole-graph owner propagation | Root integrator | `torch_mesh/partition.py` | O1 records; O3 expression handoff |
| O3 — Source and partial-expression sections | `sparse_reshape` | `torch_mesh/ownership.py`, `torch_mesh/compilation.py` | O1 projection; existing `declarations.sections` |
| O4 — Bind the retained graph once | Root integrator | `torch_mesh/declarations.py`, `torch_mesh/__init__.py` | O1–O3 |
| O5 — LiteRT access adapter | `output_domains` | microbench `tools/prepare_e2b.py` | Shared relation/access contract; O1 coordination |
| O6 — Joined preparation and removal of superseded paths | Root integrator with ticket owners | Connected call sites and this document | O1–O5 |

Here `torch_mesh/` means `rdma/torch-mesh/torch_mesh/` in mesh. Changes to
`rdma/mesh_relation.py`, `mesh-index-relation.[hm]`, `mesh_matrix.py` or native
headers need a named handoff because both frontends consume them.

## O1 — Retained access and selected numerical work

**Implement:** expose exact typed storage coordinates from the retained body and
value bindings. Compose output demand with writes to select work, then compose
that work with reads. Use the shared algebra and index grammar. Preserve masks
and changing-index producers as actual source operands; fold only relations
whose operands are known during preparation.

Apply the selected domains to the same numerical body and its code generation.
Keep fused scheduling groups and dependencies consistent with those domains.
Reusing a cached body is not permission to leave the old full-domain dispatch.
Explicit external numerical declarations must participate through the same
interface; missing declarations remain named integration work.

**Handoff:** O2/O3 receive selected work and typed required-read records. O4
receives the retained graph ready for physical binding and emission.

**Completion evidence:** actual preparation shows selected work in emitted
dispatches and source bindings; typed aliases, masked accesses, changing index
dependencies and empty reduction finishes retain their declared meaning. Graph
and body identities show that projection did not rebuild the numerical program.

## O2 — Whole-graph owner propagation

**Implement:** use source value/result identities to carry explicit owners and
schema-derived ownership through the retained graph. Propagate requests backward
through read/write relations. Factories, returned inputs, local views, correlated
cuts and compound outputs use the same mechanism. An operation without tensor
inputs still has its declared result domain.

Replace `_owned_pointwise` and `_owned_convolution` ownership reconstruction when
their shared consumer is connected. Retain their existing numerical equations
and product kernels. Preserve pending reductions on ordinary local operands;
their meaning cannot depend on retaining a DTensor wrapper.

**Handoff:** O3 supplies full partial expressions; O1 supplies access projection;
O4 consumes the resulting numerical work, source pages and crossings. Ownership
specifications remain explicit operands, not a new allocation heuristic.

**Completion evidence:** ordinary `Function` preparation reaches shared
projection for the former specialized callers and arbitrary declared result
selections. Input requests and produced domains follow the retained relations;
no implicit replication or whole-result computation fills a missing connection.

## O3 — Source and partial-expression sections

**Implement:** prepare numerical sections of an existing source expression once,
split at its declared collective or provider boundaries, and retain original
input, state and result edges. Reuse `declarations.sections` and the common
normalization of functional collective result edges.

Carry demand backward across those sections using O1. A reduction request
includes its declared reduction fiber. An average preserves the original axis
divisor. A masked partial preserves the mask and captured state. Postprocessing
keeps its declared shape and order. Noncommuting reductions follow source edges;
routing/planner order does not define their mathematical order.

**Handoff:** O2 consumes the expression and its input requests. O4 binds the
already-prepared numerical sections and selected collective boundaries. Do not
feed those sections into `Function` for a second capture/lowering.

**Completion evidence:** preparation retains the source expression's full
meaning while narrowing its requested pages. Captured state is consumed once
per declaration, original state identities survive, and selected collective
configuration matches the expression's contributor and normalization semantics.

## O4 — Bind the retained graph once

**Implement:** make ordinary `Function`/`BoundGraph` preparation consume O1–O3's
retained numerical sections. Share the existing compiler's functionalization,
decomposition and numerical rewrites at preparation, so this integration does
not bypass optimized kernels or create a competing compiler path. Bind physical
page operands and emit from the same retained graph.

Compose local equality, selected WRITE/REDUCE boundaries and required layout
conversion with the existing placement and collective APIs. Repeated consumer
uses share delivered values. Consumer-required contiguous storage is arranged
before execution, with explicit conversion work where placement cannot satisfy
the declared layouts directly.

**Handoff:** O6 receives an ordinary prepared `Function`, with the existing
execution and completion interface. The collective library receives ordinary
declared calls and page bindings; its routing algorithm is a separate scope.

**Completion evidence:** preparation reaches the common recorder without a
second numerical lowering, operation-time layout rebuilding, extra executor,
whole-operand substitute, or newly introduced wait boundary. Selected work and
page bindings agree at numerical/collective boundaries.

## O5 — LiteRT access adapter

**Implement:** finish immutable indexed-view preparation against the shared
algebra, then carry live index operands and their dependencies into existing
operation/product declarations. Keep ordered typed pieces and their scalar
interpretations distinct from code/scale/zero dependency leaves. Preserve source
identities through the existing topological view pass and provider lowering.

**Handoff:** use the same access facts and page relations as O1–O4. If a needed
shared relation is missing, extend its common definition with its owner instead
of adding a LiteRT-only inverse, decoder, ownership solver or executor.

**Completion evidence:** preparation composes immutable and live selections with
their real producers and consumers, including products. Existing provider/native
commands receive selected page operands and precise dependencies. Encoded values
use the shared reader; their raw storage is never silently treated as a scalar.

## O6 — Joined preparation and removal of superseded paths

**Implement:** connect the tickets through ordinary entry points and remove the
replaced ownership reconstruction and duplicate preparation paths. Reconcile
source/result identities, metadata units, reduction meaning, physical bindings
and command accesses where the frontends and collective binder meet.

**Evidence for this structural phase:** inspect actual captured graphs, relation
operands, placement results, prepared dispatch domains and emitted commands.
Build the affected libraries. Record specific remaining gaps and the source
that still uses them. Do not add repository unit tests, numerical runs or a new
measurement harness to declare this phase complete.

Completing O6 completes the connection described here, not all six followups.
Numerical behavior and performance still require the later ordinary application
runs, with whole-function latency and useful throughput reported together.

## Interfaces with the other followups

| Followup | What this work supplies | What its integrator still implements |
|---|---|---|
| 2. Empirical placement and routing | Legal owner domains, required reads, conversion edges and a prepared function | Whole-call feedback changing work-share/routing operands between calls |
| 3. CPU/GPU/ANE assignment | Explicit source work, typed operands, dependencies and consumer constraints | Capability-driven eligibility and resource assignment across those backends |
| 4. Partial-result fusion | Preserved reduction fibers, finish expressions and precise dependencies | Combining compatible work without changing its declared numerical meaning |
| 5. Provider integration | Source identities, exact accesses, layouts and common command binding | Remaining provider declarations and actual submission/resource constraints |
| 6. Applications and hourlies | The ordinary shared compiler and prepared execution interface | Native/LiteRT DNN, canonical LSTM and disaggregated prefill/decode composition, then paired latency/throughput evidence |

Transport and bridge integrators can change their own algorithms and service
implementation without editing the tensor ownership solver. The crossing is the
existing collective API with declared calls and page operands. Shared header or
layout changes require the same explicit handoff and normal committed-source
installation as other changes.
