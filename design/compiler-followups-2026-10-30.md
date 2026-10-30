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

The default DTensor capture and captured parameters/gradients still need to
consume that same source declaration. The explicit-ownership entry does not
yet establish this for every `Function`. Existing DTensor output sharding and
native IR read/write records are the intended common inputs for the remaining
operator declarations; there must be no absent-rule-to-replication substitution.

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
