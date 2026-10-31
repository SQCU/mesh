# Ownership integration tickets

Recorded 2026-10-30 23:14 UTC; updated 2026-10-31 02:16 UTC. The initial source
checkpoints were mesh `64477c9f` and metal-microbench `8a4454f2`; the status below
describes the implementation accompanying this update and its open connections.
This divides
item 1 of [Compiler followups](compiler-followups-2026-10-30.md) among concurrent
integrators. It records implementation work, not a new runtime interface or a
replacement for the [data model](data-model.md).

## Finish line

Frontend declarations and the placement solver supply the work assignment.
Lowering can compile that assignment forward, including deliberately replicated
work. The assignment's access relations give its input pages and output pages.
The compiler visits the assigned sections in producer order. For each section,
it derives the required input pages, binds the declared deliveries, and emits
the retained numerical work into the assigned output pages.

Demand propagation can trim internal work that has no use. It must preserve
explicit work assignments. A copy-source selection chooses which existing value
to deliver; it does not decide where that value is computed. Narrowing numerical
work changes the emitted work and storage, not just a view of a whole result.
Local read/write analysis within a section can walk its equations backward.
That does not require a backward ownership pass across the whole program.

Source types, source identity, masks, reduction semantics, changing index values,
and consumer layout constraints cross each step explicitly. The compiler solves
their relations before execution. Execution consumes the resulting page operands
and commands. An index-producing operation returns ordinary page data consumed
by another operation; it does not select a different runtime memory system.

The existing relation algebra, page placement, numerical kernels, collective
compiler and recorder provide the mechanisms. Reuse those definitions. This
work removes repeated inference and operation-specific ownership implementations;
it does not add an executor, submission queue, allocator or runtime policy.

## Starting point and status

The committed source already retains numerical IR, scalar bodies, source-value
identities, typed access records and reduction/finish domains. `prepare_source`
shares functionalization, tensor equations and numerical rewrites with the
ordinary compiler. Source identities survive those rewrites and state lifting.
LiteRT's immutable indexed views use the shared relation algebra, ordered typed
initializers and existing encoded-value reader.

`Function.declare_graph` now consumes the ownership record and binds retained
sections to selected pages. Both ordinary and owned functions enter `BoundGraph`
and the existing recorder. Selected numerical bodies use `placement.selected`
and the same native emitter; they do not re-enter source compilation. Actual
owned local-shard preparation reaches shader construction. The complete
distributed closure remains open at the connections below.

| Ticket | State at this handoff | Next connection |
|---|---|---|
| O1 | Selected work and typed live-read sections reach ordinary page/shader preparation | Connect live-read sections to ownership/importer; selector/scan declarations and pool unrolling |
| O2 | Supplied assignments compile forward through typed placement transitions | Local-value source scopes and multiple storage uses of one field |
| O3 | Partial/complete value versions splice into ordered sections; captured mask-state edges compose | Connect local index-value scopes and live reads |
| O4 | Binder consumes expanded sections and shared port relations; collective identities and assembled views share pages | Joined distributed binding, multiple storage uses and ordinary/owned preparation unification |
| O5 | Actual importer consumes shared native stages; selected results bind existing provider pages | Live-read cuts, computed-source substitution and encoded index folding |
| O6 | Open | Join ordinary entry points, remove superseded paths, then build and inspect preparation |

None of O1–O6 is closed by a helper alone. The later whole-function measurements
remain part of followup 6.

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

### Concrete preparation operands

These names describe the current source handoff. Extend the shared record and
its consumers together when a field is missing; do not reconstruct it in each
frontend.

| Existing producer | Record passed to its consumer |
|---|---|
| `partition.lower_owned_graph` | `graph`; actual local `arguments`; global captured `metadata`; `mesh`; `values` and `uses`; per-rank `requests`; `expression` and `prepared`; per-rank `projections`; `crossings`, `deliveries`, `pending`, `external`, `result` |
| `ownership.prepare_expression` | Original expression, shared algebra and aliases, ordered sections. A numerical section retains its source/result bindings, numerical IR and access effects. |
| `ownership.project_expression` | Ordered `output_selections`, required inputs, and section projections. Numerical projections contain `storage_bytes`, selected work/read/write records, separate `finish` work and unresolved declarations. |
| `declarations.boundary_projection` | Local and peer input requests, collective records and unresolved declarations. Other ranks' output requests enter explicitly through `peer_outputs`. |
| `placement.selected` | `storage`, ordered `results`, owning `pages`, and resolved consumer `uses`, passed to the retained graph's `bind`. |
| `NativeGraph.emit` | Passive numerical command declarations with typed arguments, source-byte relations, constants, slots, pointer origins and dispatch operands. Native `bind` realizes them through the existing recorder; LiteRT consumes the same declarations. |

Source fields are keyed by `(original FX node, result-field path)`. Owner records
retain the declared global type and every rank's ordered selection. A delivery
names its source field, provided owner, type, section, and destination selection
on **every** rank, including explicit empty selections. A crossing separately
names the required placement/value transition and its before-use or after-result
site. A delivery alone does not perform a pending reduction.

Deliveries also retain actual producer `sources` and the existing mesh-axis
`steps`. A final-output delivery has `section=None`. An explicit
`configuration.deliveries` operand can select copy sources independently of work
assignments. Distinct partial and completed values retain distinct identities,
even when their coordinate ranges are equal.

Global metadata describes numerical lowering; local arguments supply actual
pages. They must not replace one another. Logical result selections retain scalar
units and order. `storage_bytes` and compact storage source layouts use bytes;
`source_dtype` makes that unit explicit. Repeated result positions share values
without losing their order.

The expression's `materializations` and before/after source-identity cuts enter
`sections(..., requested_outputs=...)` before numerical fusion. This preserves
explicitly requested intermediate results. Retained preparation belongs to its
exact compiler operands in `prepared['configuration']`; changed operands replace
the cached preparation.

For selected binding, call the retained graph's `select_domains(records)`, obtain
its `consumer_uses()`, and pass those requirements to
`placement.selected(projection, algebra=..., device=..., results=..., fixed=..., requirements=..., equalities=...)`.
The algebra and device are explicit operands.
Results supply `binding`, `selection` or `relation`, and `shape`; fixed operands
also supply their actual page `operand`. Each consumer use supplies its key,
binding, relation, shape, layout requirements and read/write role. Placement
returns per-storage `{source, source_dtype, operand}` and per-use conversion
commands attached to that consumer. The same graph's `bind(actuals, **plan)`
emits the existing numerical program.
Use keys are stable consumer identities; their resolved records keep the access
role and `before`/`after` conversion commands.
Equalities pair declared uses of the same values, including local collective
deliveries. They join page placement before allocation. Ordered returned uses
keep every supplied destination, even when the same value occurs more than once.

An unresolved declaration is preparation work still to do. Resolve it before
binding an executable. Changing index contents remain page data; resolving their
declared access structure never requires guessing or freezing those contents.

## Claiming and handing off work

Choose a ticket and name the files and definitions you will change in the active
agent conversation. The contacts below supplied the current work; they are not
permanent owners or locks. At this handoff their source changes are released for
integration. Read other files freely. Arrange a handoff before editing a file
another active integrator has claimed; two tickets can share a file without two
agents editing it at once.

Send the adjacent owner the actual operand schema and consumer call site before
building against a new interface. When releasing work, report its commit or
still-uncommitted files, the consumer connected, preparation evidence, and the
remaining gap. Update the status here when the integration changes. Do not mark
a helper complete while its intended caller still uses the previous path.

Use ordinary commits with named files. Do not sweep another integrator's changes
into a commit, revert their work, or run installation from uncommitted source.
This document and agent messages are enough coordination; no lock service,
generated inventory, linter or execution-time bookkeeping is needed.

| Ticket | Handoff contact | Files principally edited | Depends on |
|---|---|---|---|
| O1 — Retained access and selected numerical work | `ane_constraints` | `torch_mesh/native.py`, `torch_mesh/operand.py`, shared index emission in `algebra.py`, `rdma/mesh_matrix.py` | Existing retained-IR checkpoint |
| O2 — Declared work and page deliveries | `sparse_reshape` | `torch_mesh/partition.py`, ownership records in `ownership.py` | O1 records; O3 expression handoff |
| O3 — Source and partial-expression sections | `sparse_reshape` | `torch_mesh/ownership.py`, `torch_mesh/compilation.py` | O1 projection; existing `declarations.sections` |
| O4 — Bind the retained graph once | Root integrator; `output_domains` for selected placement | `torch_mesh/declarations.py`, `torch_mesh/__init__.py`; `torch_mesh/placement.py` | O1–O3 |
| O5 — LiteRT access adapter | `output_domains` | microbench `tools/prepare_e2b.py` | Shared relation/access contract; O1 coordination |
| O6 — Joined preparation and removal of superseded paths | Root integrator with ticket owners | Connected call sites and this document | O1–O5 |

Here `torch_mesh/` means `rdma/torch-mesh/torch_mesh/` in mesh. Changes to
`rdma/mesh_relation.py`, `mesh-index-relation.[hm]`, `mesh_matrix.py` or native
headers need a named handoff because both frontends consume them.

O1's declarations, O4's placement and O5's source adapter can advance in parallel
once their operand fields agree. O2 and O3 share `ownership.py`; keep one editor
there or hand off named definitions. O4 owns the final `Function`/`BoundGraph`
consumer changes, so O2/O3 do not independently rewrite those callers. O6 follows
the joined interfaces. The new ownership return value and its ordinary caller
must land together as a usable source change.

The current shared-file edits are specific: O1 owns index-expression emission in
`algebra.py`; O2 removes its obsolete `owned_product`; O4's placement work changes
only `TensorView.storage_as` in O1's `operand.py`. Review those hunks with their
respective owners. A ticket is not permission to collect every dirty file.

## O1 — Retained access and selected numerical work

**Resume here:** complete missing mathematical declarations for external commands
and declared access domains for changing-index consumers. Pending records name
those gaps. Preserve the selected scalar/product implementation already prepared.
Ordinary reduction unrolling is deferred until ownership selection; pool-local
unrolling overrides still need reconciliation with that pass order.

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
dependencies and empty reduction finishes retain their declared meaning. Retain
the IR, scheduler and cached scalar graphs. Index substitution may create ordinary
`LoopBody` wrappers; empty finishes derive their declared identity body. Neither
requires another source capture or full numerical lowering.

## O2 — Declared work and page deliveries

**Resume here:** join declared local views and typed transitions through the
forward assignment compiler. `compile_assignments` visits supplied sections in
producer order; local demand projection derives their required reads without
moving or enlarging their assigned work. Explicit work assignments retain the same
meaning with and without a `DTensorSpec`; delivery choices remain separate
operands.

**Implement:** retain each supplied work domain and its source value/result
identity. Use the section's retained read/write relations to derive its input
footprint. Bind those inputs to produced pages through the supplied delivery
relations, then emit the section. A consumer's input request must not silently
enlarge or move its producer's assigned work. Deliberate replicated work remains
valid. Factories, returned inputs, local views, correlated cuts and compound
outputs use the same mechanism. An operation without tensor inputs still has
its declared result domain. Internal operations within a fused section obtain
their work from that section's equations; callers need not assign each internal
instruction.

The worktree removes `_owned_pointwise`, `_owned_convolution`, the local FX
retrace and obsolete `algebra.owned_product`. Complete their shared caller using
the retained numerical equations and product kernels. Preserve pending
reductions on ordinary local operands; their meaning cannot depend on retaining
a DTensor wrapper.

**Handoff:** O3 supplies full partial expressions; O1 supplies access projection;
O4 consumes the resulting numerical work, source pages and crossings. Ownership
specifications remain explicit operands, not a new allocation heuristic.

**Completion evidence:** ordinary `Function` preparation reaches shared
projection for the former specialized callers and arbitrary declared result
selections. Input requests and produced domains follow the retained relations;
no implicit replication or whole-result computation fills a missing connection.

## O3 — Source and partial-expression sections

**Resume here:** splice captured placement transitions into ordered per-rank
sections with explicit source/result ports. Shared mask state now follows its
actual producing node through source composition. The existing DeviceMesh rank
operand supplies ragged local types; no local trace is resized to describe a
peer. Those expressions still need the joined binding path.

**Implement:** prepare numerical sections of an existing source expression once,
split at its declared collective or provider boundaries, and retain original
input, state and result edges. Reuse `declarations.sections` and the common
normalization of functional collective result edges.

Use O1 to derive each assigned section's required inputs. A reduction request
includes its declared reduction fiber. An average preserves the original axis
divisor. A masked partial preserves the mask and captured state. Postprocessing
keeps its declared shape and order. Noncommuting reductions follow source edges;
routing/planner order does not define their mathematical order.

Each rank can have different numerical sections around a shared collective.
Keep their source order and use the existing collective declarations to connect
their pages. This does not require aligning numerical section counts or adding
a scheduler to join the ranks' source graphs.

**Handoff:** O2 consumes the expression and its input requests. O4 binds the
already-prepared numerical sections and selected collective boundaries. Do not
feed those sections into `Function` for a second capture/lowering.

**Completion evidence:** preparation retains the source expression's full
meaning while narrowing its requested pages. Captured state is consumed once
per declaration, original state identities survive, and selected collective
configuration matches the expression's contributor and normalization semantics.

## O4 — Bind the retained graph once

**Resume here:** bind O3's expanded, versioned sections through the same selected
page mechanism. `Function` consumes O2's record; `BoundGraph` keeps actual local
arguments separate from global metadata. Finish the joined distributed
preparation. Ordinary full-domain binding still has its earlier whole-family
preparation branch: normalize those declarations to the selected binding record
and remove that branch. Both already emit through the same numerical compiler
and recorder; O4 is not complete until their preparation is shared too.

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

**Resume here:** connect the shared live-read sections to the existing importer
consumer. `prepare_e2b.indexed_source` already reaches retained access projection,
selected placement and native stage emission through the actual save path.
Those results feed ordinary provider inputs with their declared scalar types.
Computed producers still need substitution through live reads; a staged
UPDATE→GATHER must not materialize the whole updated table merely to select rows.

The adapter returns a source graph and explicit operands. Placeholder metadata
`mesh_litert_operand`, source-node metadata `mesh_litert_origin`, and module
metadata `mesh_litert_outputs` retain the importer bindings. Use those edges
through shared preparation. The artifact consumer remains the existing native
stage/Metal recorder or declared provider; source capture adds no PyTorch
execution dependency to the imported program.

Immutable encoded **data** views already retain their typed reader. Evaluating
encoded integer **index values** during preparation is still open:
`constant_integers` currently folds ordinary immutable scalar views. Until that
shared reader connection exists, encoded indices retain their typed graph
operands; their storage bytes are not interpreted as ordinary integer indices.

**Implement:** with immutable indexed-view preparation completed against the
shared algebra, carry live index operands and their dependencies into existing
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

**Resume here:** after the producer/consumer handoffs are joined, inspect the
ordinary entry points against the finish line above. O6 closes the shared path;
it does not add another wrapper that hides an incomplete O1–O5 connection.

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

## Preparation evidence at this handoff

- O1 prepared unequal reduction fibers, small reductions whose axes survive
  ownership, segmented products, and empty selected fibers with their declared
  finish outputs. Selection changes the emitted work, not just the output view.
- O2/O3 retained the global SUM and matrix graphs while selecting unequal K
  contributions. Matrix output followed by ReLU retained two sections with an
  explicit unresolved partial-to-replicated transition before ReLU. That record
  identifies unfinished integration, not a runnable substitute.
- O3's collective projection retained typed gather/scatter offsets, local
  identities, untouched output-view regions and the original AVG divisor. The
  source also represents indivisible reducer elements; this handoff does not
  include preparation evidence for multi-scalar elements. Peer requests still
  need O2's graph-wide connection.
- O4 prepared compact selected products through `select_domains`,
  `consumer_uses`, `placement.selected` and native `bind`. Byte aliases, mixed
  scalar widths, repeated/strided results, empty outputs and required per-use
  layout conversions kept their declared page relations. Ordinary numerical
  sections emitted `Program` directly, without recursive `Function` compilation.
  The ordinary owned entry point prepared a two-rank local shard and retained
  both supplied destinations of a repeated result. A selected ragged gather's
  local delivery composed to identical physical pages.
- O5 prepared live indexed source expressions through shared source lowering
  and native access extraction. This did not connect the importer stage emitter.
  The portable emitter's actual commands also reached native-stage preparation:
  ordinary/mixed byte views needed no copies, and complete packed values decoded
  directly into final pages. A partial-byte demand decoded only its reached
  scalar values before the required byte selection.

A requested partial contraction retains its reduction axis until ownership is
applied. `finish.work` assigns finishing work from the source semantics: an
explicit partial bias share may run per contributor, while a noncommuting finish
follows the complete reduction. A partial K size alone determines neither choice.
Unrolling that erases the axis before placement is a pass-order error.
These observations establish preparation behavior only. No numerical or
whole-function latency/throughput result is attributed to this integration.
