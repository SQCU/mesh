# Xonotic indexed-add caller migration

September 13, 2026. This increment routes rank-one scatter_add through the shared
`kernels.indexed_add` lowering. The live source sites are scale_balance in
`xonotic/solver/strat/cast_header.py` and routed_table in `paged_matrix.py`.
The numerical layer supplies base, destination indices and updates; shared mesh
lowering owns grouping, collision reduction, partial publication and completion.

## Shape and storage

Xonotic's rank-one values use a logical 1 by N matrix representation. Indexed-add
uses D by 1 and U by 1 views of those same pages. Transposition changes only
reference metadata; the resulting D by 1 output is exposed as 1 by D without
copying. Its destination blocks are independent output regions. Single-backing
reshapes use ndarray shape assignment on a view solely to derive the existing
strides, then retain those dimensions and strides in a native Ref. This operation
does not read numerical data or create a reshaped operand copy.

Scalar/vector pointwise operations and rank-one sum/mean consumers now use the
shared expression path. Integer add, subtract, multiply, negative and sum terms
are converted algebraically to unsigned 64-bit bit patterns before arithmetic,
then stored in the declared integer dtype. This avoids signed-overflow arithmetic
inside the new paths. `nn._sum` selects the shared integer expression during setup
while retaining its existing native real-valued addition and binary tree owner.
Unsupported multi-block shape reinterpretations remain in their existing numerical
source lowering; a shape transformation is not implemented by copying a tensor.

## Retained metadata for remaining operations

The immediate consumers include floor division, cumulative sums, searches,
gathers and concatenation. Their remaining custom source bindings previously
requested a single whole-tensor region and inspected its ndarray strides. A
partitioned scatter output cannot satisfy that request. The binding now retains
all block pointers, exact per-block row/column strides, the logical matrix shape
and block-grid geometry. The existing custom source's logical flat address selects
the correct block and then the scalar within it. Ragged block strides come from
each actual Ref; they are not guessed from nominal block dimensions.

Historically, the raw Metal binding included each Ref offset in its bound pointer.
That binding has since been removed; the Xonotic subproblem remains parked.
The custom descriptor therefore keeps no second data offset. Input blocks are
bound in the same order as the explicit descriptor list; the output occupies its
own final pointer slot. Clear operations address the same output scalar bytes.
No dense operand is assembled, no extra numerical scheduler is introduced, and
no input pointer is inferred by searching an allocation range.

This metadata change preserves remaining numerical implementations during their
migration. It does not complete their lowering: a custom operation still depends
on all blocks it declares, and its original algorithm may contain whole-region
work. These are outstanding migration items, not an alternative final interface.
The shared scatter, pointwise and sum paths do not use that custom binding.

The prior art is the [Pallas reference/index-map model](https://docs.jax.dev/en/latest/pallas/design/design.html)
and the [shared segmented-add mechanism](scatter-lowering.md). The retained
logical-to-physical mapping applies that same model to existing numerical code.

## Evidence and remaining work

Python source compilation passes for tensor_metal.py and nn.py. The modified
Metal address helper source compiles through MTLDevice.makeLibrary. No numerical
workload was run by this increment. The planner and actual expert-routing workflow
must provide operational evidence; compiling helper source alone does not prove
all emitted application kernels or the complete routing workflow.

Remaining work includes gather/neighbor derivatives, arbitrary rank/axis
transformations, cumulative-sum/search lowering, and their actual region dependency
sets. Shared indexed-add currently reserves one partial ownership block per
potential segment and feature panel; many tiny contributions can therefore consume
far more registered memory than their scalar byte count. Larger live-routing
performance requires addressing that physical-storage cost while preserving
independently publishable results. This caller migration makes no broad memory,
latency or throughput claim.
