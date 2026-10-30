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
removed. The shared native build passes; complete convolution page binding
is still under preparation review.

`Algebra.cover` exposes the existing native cover operation. Its disjoint cells
retain the contributing input domains. Objective-C and Python now consume one
implementation. The selected-copy adjoint can use those cells to declare the
necessary local sums and collective reductions; reversing its WRITEs alone
would lose contributions from repeated reads.

The captured attention placement still stalls while resolving field equations.
Its preparation operands are recorded for the next correction. The function
composition change does not establish that placement issue is resolved.
