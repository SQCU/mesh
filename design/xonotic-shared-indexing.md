# Shared Xonotic gather and concatenation

September 14, 2026. This increment migrates gather and concatenate with one- or
two-dimensional sources/indices and at most two-dimensional results to the
existing expression compiler. It adds no new compiler primitive or execution
interface. Higher-rank operations and shape reinterpretations that cannot be
represented by the existing direct views retain their current numerical source
until their shared lowering is implemented.

## Retained numerical maps

`tensor.py` supplies gather's existing mapping tuple, advanced-index shape and
adjacency flag. `tensor_metal.gather_expression` translates that same mapping
into expressions rather than generated Metal statements. Output coordinates are
`program_id * block_extent + indices`, so the coordinates remain global even when
an independently published output block has a ragged tail.

Slices preserve their start and step, including negative steps. Fixed indices
remain fixed. Advanced index operands use the existing broadcast-axis placement;
singleton dimensions use coordinate zero. Negative dynamic indices are normalized
against the corresponding logical source axis before the source load. Inserted
axes consume output dimensions but not source dimensions. Integer indices remain
integer expressions, and source loads preserve the declared value dtype.

Concatenation retains the configured axis and cumulative source lengths. Each
output coordinate selects the corresponding source and subtracts its source's
origin on that axis. Nested selects contain ordinary indexed loads; their branch
predicates are retained by the existing expression dependency analysis. A tile
spanning a concatenation boundary legitimately reads both source regions. A tile
inside one source does not require another source's missing values.

The mechanism follows the [Pallas reference and index-map model](https://docs.jax.dev/en/latest/pallas/design/design.html)
and uses mesh's already implemented conditional indexed-load and selected-reader
lowering. The caller does not construct selection buffers, dynamic reader state,
backend source, or a second numerical scheduler.

## Storage, publication and derivatives

Logical vectors use the existing 1 by N representation. Matrix views preserve
canonical backing pointers and actual strides. No gathered input staging tensor
is created; the gather output itself is the intended numerical result, stored in
its independently publishable output blocks. Concatenation likewise writes its
actual result without first assembling a whole input.

`Program.kernel_call` realizes these block outputs and bindings during setup.
The shared `.at` lowering supplies selected-page dependencies and source holds.
Changing caller syntax has not changed that runtime protocol. The default output
grid uses the existing tile_rows/tile_columns choices; it does not impose input
block alignment because indexed loads retain their source block maps.

The graph's derivative rules are unchanged. Concatenation's derivative slices
its cotangent, so covered slices now use this shared gather lowering. Gather's
reverse operation remains the existing gather_vjp implementation with explicit
block-address metadata; duplicate gradient indices still follow that operation's
existing collision semantics. This increment does not claim to have migrated
that reverse scatter, take_along_axis, or higher-rank derivatives.

## Evidence and remaining work

Python source compilation passes. Source review compares gather's translated
mapping directly with the previous gather_address routine and traces source and
output indices through the existing expression binder. No numerical workload was
run for this increment. The existing Xonotic workflow must validate actual emitted
kernels, numerical output and derivatives; syntax compilation does not establish
those results or a performance improvement.

Remaining work includes higher-rank view lowering, gather VJP scatter composition,
take_along_axis and its VJP, and compiled cumulative-sum/search operations.
Selected-load metadata and kernel launch costs remain measurable overhead; the
migration establishes region dependency structure without claiming zero cost.
The old custom implementations remain only for still-unmigrated cases, not as a
second completed public implementation style.

The retained planner command is `bin/mesh-python xonotic/planner/plan.py solve
<player-node> 2 8 --width 16 --tile-rows 8`, paired with its existing `play`
role. Its current solve graph contains matmul, expert_matmul, comparisons and
reductions, but no gather or concatenate. It checks general integration, not
this migration's numerical results. `solver.strat.measure matrix` uses the MLX
numerical path and likewise does not validate the shared expression branch.
The optional existing streaming-algebra invocation below now supplies explicit
gather/concatenation observations; planner coverage remains separate.


## Existing gold workflow extension

Run the current installed mesh package through the Xonotic Python environment:

```sh
bin/mesh-python examples/streaming-algebra.py 0 --local --backend cpu --runs 1 --xonotic
bin/mesh-python examples/streaming-algebra.py 0 --local --backend metal --runs 1 --xonotic
```

`--xonotic` imports the actual symbolic graph and kernel_calls only when requested,
so the usual NumPy-only examples do not acquire an MLX dependency. The side graph
uses `source[indices, ::-1]` followed by concatenation with an independently
produced tail. Source and index operands each have two canonical blocks. It first
publishes one index block and its selected source block plus the tail. Output
blocks zero and two must become usable while the other index/source blocks remain
absent. It then publishes the missing blocks and compares every result with a
float64 NumPy reference prepared before the timed gold computation.

The same buffers execute twice, changing which source block is selected first and
including a negative advanced index. Both generations are observed through the
existing result/readiness interface. The case emits xonotic_indexed_early and
xonotic_indexed_complete records. It runs outside the timed FFN batch, in the same
Program; it creates no second scheduler, benchmark driver or evaluator.

This extension does not evaluate derivatives. Existing derivative graph rules and
backend code remain unchanged; numerical derivative evidence is still outstanding.
Python compilation passes for this extension. The commands above are instructions
for operational validation, not a claim that this increment has already run them.

## Subsequent operational evidence

The optional case passes on CPU and Metal with installed mesh source `5843a8b`,
both locally and on rank zero of the paired gold workload. Two occurrences change
indices and selected source blocks; early gather and independent concatenated
tail outputs are exact before the withheld index/source occurrence is supplied.
The observer checks writability rather than raw PRESENT, which may remain set on
consumed storage. `xonotic_indexed_bindings` retains actual source, index,
intermediate and output row identities for trace joins. The
[sparse routing evidence](sparse-routing-progress-2026-09-13.md#shared-directory-ownership-and-larger-execution)
links the raw observations and traces. These side computations execute on rank
zero; they do not establish derivative coverage or distributed Xonotic indexing.


The reuse observer checks the withheld inputs' writable state and the dependent
output's next-occurrence readiness separately. `Ref.present` exposes the raw
presence plane; consumed storage may retain PRESENT together with READ and still
be writable for its next occurrence. Raw presence therefore does not establish
that the next generation was supplied. Each early observation records the two
withheld writable states; premature dependent output readiness has its own error.
The first operational CPU attempt completed generation zero exactly, then exposed
this observer mistake in generation one. The corrected source has not yet been
rerun in this record.

The `xonotic_indexed_bindings` setup record retains each source, index, tail,
gathered intermediate and exported output block's actual native row range,
extent, offset, shape and strides. `mesh_tensor_rows` supplies first/count from
the canonical allocation; no row identity is reconstructed from timing or buffer
addresses. The Python RowMap return type matches all fields of mesh_row_map,
including its range/member/offset pointers. These bindings allow direct joins to
native compute and selected-reader trace records.

## Shared row-gather transpose

The next migrated derivative is gather_vjp for `x[index]` on a vector and
`x[index, :]` on a matrix, with one rank-one index vector and the complete
feature-axis slice. Its mathematical output is zero plus an indexed sum of the
cotangent rows. Duplicate indices contribute repeatedly; negative indices use
the existing normalization. The [JAX gather transpose implementation](https://raw.githubusercontent.com/jax-ml/jax/main/jax/_src/lax/slicing.py)
uses the same zero-plus-scatter-add construction and derives zero's dtype from
the cotangent rather than the primal.

Xonotic now supplies a setup-initialized canonical zero base, destination indices
and cotangent values to the existing indexed_add expression. Matrix cotangents
retain their direct row/feature block layout; vector values use transposed
metadata as in forward scatter. Output feature tiles respect actual cotangent
backing boundaries. The shared lowering owns FP32 real partial accumulation,
collision reduction, final output conversion and independent publication.
There is no derivative-specific atomic kernel or device source emitter.

This derivative does not read primal source values. Its migrated binding omits
source replication and numerical source dependencies; only shape information is
needed. A separate forward gather in the same graph still has its own legitimate
source reads. Routing metadata still determines when each destination sum is
complete; a missing unrelated cotangent region does not create a whole-cotangent
prerequisite. Zero base blocks are initialized once during setup and are reusable.

Column gathers, feature-subset slices, multiple index arrays, inserted-axis
variants, take_along_axis derivatives and higher ranks remain migration work.
Their existing source implementations are preserved. The graph differentiation
rules themselves are unchanged. Python compilation passes; the existing workflow
still needs duplicate-index and delayed-cotangent numerical observations for this
new derivative path. No execution or performance result is claimed here.

## Requested-output liveness

`kernel_calls` now requires `outputs=(...)` on its existing setup call. It walks
backward from those logical Tensor roots, retaining only numerical dependencies
and stopping at supplied input bindings. This is ordinary dead-operation
elimination, consistent with [MLIR's canonicalization rules](https://mlir.llvm.org/docs/Canonicalization/#globally-applied-rules).
It changes neither the public kernel execution interface nor the runtime owner.
Unrequested graph nodes receive no operand allocation, transfer edge or reader
registration. All requested roots and their required intermediates remain in the
returned tensor mapping. Inputs are already-realized numerical boundaries.

The migrated row-gather VJP's primal operand contributes its configured shape,
not a numerical dependency. A derivative-only root therefore does not compile an
unused forward gather, register readers for its indices, or require a supplied
primal tensor. This matters for repeated invocation: a never-executed forward
consumer cannot retain an otherwise finished index occurrence. Source shapes
remain available from the logical graph metadata even when their numerical
producer is eliminated. Other operations retain their declared dependency edges;
further shape-only operand analysis is separate work.

The planner declares its `y` output explicitly. The existing Xonotic gold graph
declares its joined output; a derivative case declares its gradient roots.
Python source compilation validates the modified compiler and planner. Runtime
reuse evidence remains the responsibility of the existing gold workflow.


## Active-domain and derivative operational follow-up

Installed numerical source `dc4c80a` passes the existing CPU and Metal local
streaming-algebra workflow with the derivative and explicit output roots. Both
occurrences publish gradient rows 0, 1 and 3 before the final cotangent chunk
arrives; row 2 completes afterward. Duplicate and normalized negative indices
match the independently prepared reference. No primal operand is allocated or
supplied. Reuse succeeds after consuming the first outputs, including the index
reader lifetimes that previously would have included an unused forward graph.
These observations supersede the pending-runtime statements above for this
specific row-gather transpose and forward gather/concatenate scope.

The bounded large-directory follow-up uses example `e0ecb91` with 4097 updates,
2049-row input chunks and 17 scatter destinations on both CPU and Metal. Its
Xonotic observations again include both derivative occurrences. The shared scatter
case includes an entirely empty occurrence whose outputs complete before update
and factor inputs are supplied, followed by a populated occurrence using the same
storage. See [active segment evidence](active-segments.md) for raw archives and
the native disposition contract. These remain local indexed operations; neither
paired gold nor remote fanout establishes remote indexed-derivative coverage.
Matched throughput evidence and the remaining derivative forms are still open.

## Shared vector and row reductions

Xonotic vector sum/mean and real matrix axis-1 sum/mean now lower through the
existing shared expression `.sum()` with whole-input BlockSpecs. The mechanism
and primary Pallas/Triton references are in
[shared region reductions](algorithm-sources.md#streamed-row-reductions-in-the-shared-region-owner).
The caller declares only the formula, output dtype, placement, and row layout;
shared lowering owns independently available feature partials and their balanced
reduction. The former `nn._row_reduce` and its row-map helper have no callers
and are removed.

A matrix reduction retains physical shape `(rows, 1)` and the same row tile
computed from the requested tile and actual input backing cuts. A vector
reduction retains physical `(1, 1)` output. Logical rank and keepdims metadata
remain the existing graph's responsibility; subsequent vector consumers use the
same direct transpose/reshape metadata as before. No operand is repacked.

Real partials and the balanced statistic accumulate in FP32. Mean divides that
statistic before casting once to the declared output dtype. This intentionally
removes the old FP16 per-panel and intermediate-sum rounding; reproducing those
rounding losses is not a numerical library contract. Integer vector sums retain
integer arithmetic and the declared modular output cast. Integer means retain
a typed sum output before integer division, including I32 wrap-before-division
behavior; no floating-point conversion or caller-built partial tree is inserted.

The existing optional Xonotic streaming-algebra graph is the numerical validation
path: matrix row means can complete for one source row block while another row
block remains unpublished, and their vector total depends on both. Source
compilation passed for this migration; operational results are recorded by the
parent integration run rather than assumed from compilation.
