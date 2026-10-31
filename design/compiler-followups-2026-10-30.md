# Compiler followups

The October 30 [operand checkpoint](operand-contract.md#integration-checkpoint-2026-10-30)
establishes the shared preparation interfaces. The next implementation work,
in the operator's order, is:

1. Complete frontend operator declarations using the shared relation algebra.
   This includes partitioned windows, indexed selections and correlated cuts.
2. Let whole-call feedback choose among legal placements and required conversion
   edges. Layout feasibility alone does not choose the fastest program.
3. Carry backend eligibility and resources as capability operands, and assign
   work across CPU, GPU and ANE through the same graph.
4. Fuse compatible partial results using their declared dependencies.
5. Complete provider access declarations and submission constraints. Actual
   vendor restrictions remain explicit until an implemented API removes them.
6. Compose ordinary native and LiteRT applications, including the non-transformer
   DNN, LSTM comparison and disaggregated prefill/decode work. Measure complete
   functions after the structural work, recording latency and useful throughput
   together. Hourlies consume the ordinary demos.

Build and preparation evidence establishes emitted structure. Numerical results
and whole-function performance need their own evidence; these followups remain
open until their implementation and relevant verification are complete.

Item 1's active work is divided in
[Ownership integration tickets](ownership-integration-tickets-2026-10-30.md),
including file claims, shared operands, dependencies and handoffs to items 2–6.

The subsequent ownership integration connects ordinary `Function` preparation
to retained selected sections. Work assignments remain operands, including
deliberate replication; backward demand selects unassigned internal work and
declares page deliveries. The binder preserves local identities as page
equalities, actual input pages, ordered result uses and supplied destinations.
Source/placement preparation has reached ordinary owned native shader binding.
Partial/complete transitions now splice their captured expressions and distinct
value versions into those forward sections. Local index-value source scopes and
multiple storage uses of one semantic field remain open.

Native scalar emission now returns passive command declarations before pipeline
realization. The native and LiteRT consumers retain the same typed byte
selections, explicit slots, constants and binding origins. Full encoded values
can decode into final pages; sub-scalar requests retain their exact byte
interpretation. The LiteRT importer now consumes those stages and their exact
per-command accesses. Selected native results feed the existing provider input
pages, including explicit scalar-type conversions. Retained live-read sections
now prepare through the same page and shader binder; their ownership/collective
and importer connection remains open. Computed-source substitution through live reads, selector/scan
declarations and pool reduction ordering remain open. These are source, build
and preparation observations, with no new
numerical or performance claim. Followups 2–6 are still outstanding.

## Window relations

`mesh_relation.Algebra.window` describes a window's source accesses. Both FX and
LiteRT supply the same geometric operands: the iteration and source shapes;
source, output-position and window-tap axes; strides; dilations; padding; and
direction. Extra equations declare batch, channel and group relationships.
Frontend adapters translate their source syntax into these operands.

For an ordinary window, each spatial axis obeys
`source + padding = output * stride + tap * dilation`. A transposed window obeys
`output + padding = source * stride + tap * dilation`. Source bounds remove
padding positions from the read relation. Transposed windows retain only valid
stride congruences. Neither form clamps a coordinate or requires a dense index
table. The numerical declaration separately supplies the missing-value meaning
and reduction operation; an absent read does not imply an absent output.

The returned object is the existing `IndexRelation`. Ownership images, required
reads and emitted indexing use its ordinary composition operations. No window
object, relation traversal or storage choice reaches execution.

This is data-kind source. On D1 and D4, separate frontend window equations
`old ≺ new`: one relation builder supplies both. On D2 and D8, runtime layout
resolution `old ≺ new`: preparation emits the same page bindings as other
operations. This comparison makes no measured-performance claim.

## Selected page crossings

`torch_mesh::select_exchange` takes a tensor, its result shape, declared source
and destination selections for every rank, and the collective group. The
selections map local positions to shared logical positions. They contain no
payload values. Source ownership is disjoint; each requested position has a
source. A destination may name one value more than once.

`Algebra.exchanges` composes the destination relation with each source's inverse.
It emits paired, ordered selections of source and destination positions. The FX
binder equates the local pairs before placement and binds remote pairs through
`prepare_writes`. All pairs use the existing page layout and WRITE compiler.
There is no local copy for an equality or whole-input transfer for a narrow
selection. Explicit consumer layout constraints still belong to placement.

The graph node retains this preparation result through fragment construction.
Lowering and binding consume it instead of deriving the exchange twice. The
only execution objects are the same prepared writes used by other collectives.

Preparation inspected forward and transposed window relations, padding and
stride congruences, and selected exchanges with overlap and repeated destination
values. These inspections constructed relations and bindings without numerical
execution. Full frontend consumers and complete applications remain in progress.

## Shared algebra preparation

Functional coordinate relations now compose with ISL's piecewise-affine
function composition. Many-valued relations retain ISL map composition. Both
return the same relation type. Keeping a function's coordinates explicit avoids
recovering them from eliminated variables after a long composition. The captured
convolution relation now emits index terms through this path.

Native expression normalization also uses one ISL function pass. The former
recursive simplifier solved the range of many subexpressions, then repeated
until its own expression syntax stopped changing. That extra procedure is
removed. Ordinary convolution now completes native page binding and matrix
shader preparation through page assembly and the existing product kernel.
Conditional index expressions compile each arm on the positions selected by
its condition. Unreachable arms add no relation work. This uses the same term
language and changes no numerical command or runtime branch.

`Algebra.cover` exposes the existing native cover operation. Its disjoint cells
retain the contributing input domains. Objective-C and Python now consume one
implementation. The selected-copy adjoint can use those cells to declare the
necessary local sums and collective reductions; reversing its WRITEs alone
would lose contributions from repeated reads.

Placement retains a function from each byte to its component representative and
displacement. New field equations join those representatives before the result
is composed back onto the bytes. Each representative moves toward a smaller
neighbor; function composition compresses the resulting paths. The original
field edges remain available to check consistency after relabeling. This
preserves the equations without expanding every pair of bytes in a component.
Larger declared fields enter first; this changes preparation work, not the
requested layouts.
The captured attention and partial-output convolution placement operands now
complete. Preparation also retains consumer footprints, non-power-of-two base
alignment and fixed caller pages when a conflicting layout needs a conversion.
Full partial-output convolution now completes page assembly, native binding and
shader preparation. Consistent offset cycles retain their displacements;
contradictory offsets and distinct-byte collisions remain infeasible. A
trillion-byte contiguous declaration stays symbolic in this same solve.

## Remaining operator connections

Shared product declarations now supply FX and LiteRT with the same shader
bindings and dispatch geometry. Window adapters retain exact work, source-read
and result relations. Output ownership is an explicit input to `owned_domains`;
the result gives each owner's work and required source pages. `Function` accepts
source-value ownership declarations and retains each value's global type and
all ranks' ordered selections through decomposition. A composed pair of
convolutions with an intervening pointwise operation completes ordinary
page/shader preparation. Correlated channel/spatial selections also compose
through that graph without reducing them to one shard axis. Empty products
declare the ordinary zero result.

Ordinary and explicit-ownership functions now use one source capture. Captured
parameters keep their original identities, global types and local page operands;
fake gradients keep their parameter identity without replacing the real
gradient. Attribute lifting preserves those records. Existing DTensor operator
schemas supply output and required-input placements. `ownership.propagate`
consumes the captured result types rather than deriving them again. Compound
results retain their separate field types; ragged placements keep their
configured shares. Partial results stay partial until a consumer requests their
reduction. Required transitions are shared by source value and target spec.
An absent operator declaration does not become replication.

This does not establish every `Function`: declared factory outputs, arbitrary
owner results and explicit local views still need the generic retained-IR
projection. Source normalization now uses the existing decomposition interpreter
and shared tensor equations before ownership propagation. Only the final results
inherit the source's ownership and gradient identities; intermediate expressions
retain their source provenance. Means use the declared global count, including
unequal shares.

Pending partial values now have a source-expression declaration. It captures the
existing reduction function once, including tensor state, masks, collective
boundaries and postprocessing. State becomes ordinary operands and mutations
become value expressions. Projecting this complete expression onto requested
output domains remains unfinished. A reduction name alone cannot replace it:
an average retains its original divisor, and a masked partial retains its mask.

Selected-copy backward declarations sum repeated contributions locally and use
the existing REDUCE compiler for their contributor sets. LiteRT MAX, average
and L2 pooling lower through window relations and ordinary reductions. Normal,
SAME-padding and depthwise product bindings reached native pipeline preparation
without conversion copies. Compatible product domains now share a batch using
typed page pieces and the same product declaration. Batched SAME-padding and
transposed products reach native pipeline preparation without conversion copies.

Dynamic selections preserve scalar payload bits through declared byte operands.
Packed code, scale and zero-point selections use that same path. Packed 2/4-bit,
immutable and sparse inputs reach native preparation with their selected
quantization operands. A scalar consumer uses the existing decoder.
Byte-piece unions now reach placement before scalar grouping is materialized.
Fresh nonadjacent bytes and explicitly bit-interpreted heterogeneous pieces
co-place without commands. Fixed external bytes with an incompatible layout
retain their positions and use the existing compiled byte copy. These are source
and preparation results, without numerical execution or whole-function
performance claims.

Static strided slices and gathers now become the existing ordered page views.
Slice masks, new and removed axes, repeated indices and batched gathers retain
their source coordinates. Normalization rebinds already-declared stages through
the shared operand relation. Explicitly empty outputs retain their type and
identity without producers, source initializers or empty provider programs.
Symbolic dimensions retain their index expressions when their placeholder shape
contains zero. Static indexed operations now join the same topological page-view
pass as reshape and transpose. Immutable index values follow earlier views and
typed pieces through the existing ordered initializer ranges. This preparation
folds declared scalar values; encoded storage dependencies are not logical index
values and remain with the existing numerical reader.

## Retained logical IR

`native.prepare_graph` lowers the source once and returns its logical IR before
physical binding. `effects` exposes the existing operation objects, typed reads
and writes, and their shared iteration and reduction variables. A broadcast
read can omit an axis without losing its relation to the operation's work.
Compound external results retain their named field types. Whole-buffer external
dependencies do not yet supply a numerical work-to-access relation.

Logical materialization requirements enter before consumers lower, so a later
consumer reads a declared intermediate instead of inlining its producer.
`graph.bind` assigns physical pages and emits commands from that same IR through
the existing `PageKernel` and preparation view emitter. The ordinary
`compile_graph` entry calls those two stages. Preparation retained a reduction's
common variables, a declared intermediate, and a strided destination without
building the logical graph again. This is data-kind source: on D2 and D4,
`old ≺ new` because one lowered graph now supplies dependencies before binding
and final command emission. Runtime execution is unchanged.

Shared matrix declarations now expose work, input reads, output writes, and a
separate finish domain. Segmented inputs retain their own reduction intervals;
an empty contraction still has its output/finish domain. Native matrix calls and
explicitly declared product kernels compose these relations with their retained
views. `Algebra.layout` calls the existing native layout parser and relation
compiler, without expanding positions. The shared `binding_layout` supplies the
same view composition to source dependencies and physical binding. Other opaque
external commands still need their mathematical access declarations.

The existing native scheduler and wrapper now prepare once before page placement
and survive through code generation. Access records retain the scheduler's
actual scalar-body nodes, enclosing masks, and changing-index producers. Source
FX values retain their original IR objects and views. Scheduling dependencies
keep their scheduling role; they are not substituted for mathematical accesses.
The shared index-term adapter uses the framework's expression interpreter and
the existing relation compiler.

Ownership still needs to consume these common IR records throughout source
lowering. These are data-kind changes: D1 and D4 rank shared operator declarations
and binding composition above repeated frontend implementations. No runtime
executor or numerical algorithm was added.

Dynamic updates expose this boundary. Unchanged base pages remain shared, and
ordinary index expressions choose changed values from update pages. Compatible
consumers can fuse those reads; a consumer requiring materialized storage gets
an explicit conversion edge. The attempted whole-base copy and the old
same-shape ownership shortcut are removed. Distributed update declarations,
including live index operands for external products, remain unfinished.

The retained-IR checkpoint completed reduction and declared-intermediate page
preparation. Earlier source capture and static-selection preparation also
completed. The subsequent shared native library build succeeded, but the latest
source/ownership and matrix-domain preparations stop at their first page
allocation: the local bridge region reported layout 154 while source required
155. The ordinary committed-source installation now supplies layout 155.
Metadata-only preparation retained source identities, partial expression state,
body masks, and changing-index producers. Subsequent immutable-index preparation
composed reordered/repeated views and named parameters with no copy commands;
encoded indexed views retained exact code/scale/zero selections through the
existing decoder. Algebra-backed binding of the newest source/ownership
connections remains in progress. These results do not establish numerical or
whole-function performance.
