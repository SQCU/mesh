# Lowering coverage and replacement ownership

Source inventory at `7d0ce4e`, September 13, 2026, for step 1 of the
[Pallas lowering plan](pallas-lowering-plan.md). This records the starting
implementation, not completion of its replacement. No workload was executed for
this inventory. A supported graph operation is distinguished below from compiled
backend coverage and from measured numerical/performance evidence.

## Canonical library contracts

[Program, Tensor, Ref and BlockSpec](../python/mesh/__init__.py) own setup,
canonical-page storage and dependencies. Tensor/Ref currently expose two numerical
axes. Tensor blocks may have ragged tails; BlockSpec clips a requested region to
the tensor edge. A region must fit one backing block: requesting a region across
blocks raises an error. Transpose and broadcast retain physical strides (including
zero broadcast strides). Copy requires matching shapes and strides and preserves
explicit sender/receiver views. The replacement owner for N-D indexing and
cross-block coverage is this same view/binding layer, not an application store.

`kernel_call` already supports several outputs, but the compiled expression binder
currently accepts one. Arbitrary Python kernels run as numerical callbacks on
NumPy views; they are not setup-traced compiled expressions. The native operation,
compiled expression and externally supplied Metal descriptors use the same call
entry point. Preserve externally supplied kernels with explicit regions while
migrating ordinary numerical bodies to setup compilation.

[Scalar expressions](../python/mesh/kernels.py) currently support arithmetic,
exp/tanh/rsqrt and last-axis sum, compiled to CPU or Metal for FP16/FP32 views.
Native operations include matmul, affine, add, multiply, exp, tanh, row sum, rsqrt
and swish. `gather` is a Python callback using direct `np.copyto` per selected row.
The shared expression representation owns the replacement for indexed accesses,
integer/boolean values, masks, scatter contributions and multiple outputs.

[NN composition](../python/mesh/nn.py) owns tiled linear, FFN, RMSNorm, embedding
and sums. Linear retains FP32 K-panel products and tree reductions and casts at
the completed output tile; FFN preserves FP32 cross-partition sums. RMSNorm
statistics are FP32. A normalization tile needs its own row statistic, not all
rows. Today the application composition still creates partial tensors and launch
trees explicitly. Generic region/reduction lowering owns their replacement.

Readiness, transfer endpoints and source-reader lifetimes remain owned by
[mesh-dataflow.c](../rdma/mesh-dataflow.c) and [mesh-flow.c](../rdma/mesh-flow.c).
Compiled dependencies must preserve independently usable outputs, fanout and
simultaneously live value instances. A duplicate scatter contribution is a
partial value; it cannot publish a destination as its completed sum.

## Xonotic graph and backend coverage

The graph language is [tensor.py](../xonotic/solver/strat/tensor.py); the current
mesh lowering is [tensor_metal.py](../xonotic/solver/strat/tensor_metal.py).
Every row in this table has replacement owner `mesh.kernels` numerical expressions
plus canonical BlockSpec/region lowering, retaining `tensor.py` as a caller until
its complete mathematical behavior has migrated.

| Family | Existing numerical contract and derivative | Current mesh lowering / remaining coverage |
| --- | --- | --- |
| Pointwise | Add/subtract/multiply/divide, negative, power, exp/expm1, log/log1p, sqrt/rsqrt, arcsinh, abs, min/max, where, logaddexp, sigmoid; trailing-axis broadcasting. Explicit VJPs for these, min/max ties split equally, abs derivative zero at zero. | Selected rank-2 float arithmetic uses shared expressions. Other operations/ranks use whole-region custom Metal. Preserve stable sigmoid/log1p/expm1/logaddexp/asinh formulas; formula names alone do not establish equivalent range behavior. |
| Integer and boolean | Comparisons, logical operations, bitwise and/or/invert, floor_divide, isfinite; comparisons produce bool. Most return no derivative. | Custom Metal only. Floor division emits ordinary `/`, which truncates negative integers rather than Python floor semantics. Bitwise invert lacks an explicit no-derivative branch. These are existing gaps, not behavior to canonize. |
| Shape operations | Reshape, transpose/permutation, broadcast, concatenate, stack, cast, stop_gradient; transpose inverse, concatenation slices and broadcast sum-to VJPs. | Reshape/stop_gradient alias the existing tensor; transpose generally materializes custom Metal output. General N-D shape views and aliases must survive without accidental copies. `assign` emits a new output reading value/target; no explicit derivative branch. Do not infer in-place assignment from the name. |
| Reductions | Sum/mean/min/max/any/all over selected axes with keepdims; sum/mean broadcast VJP, extrema split ties among equal elements, logical reductions nondifferentiable. | Rank-2 row sum/mean uses shared NN reduction; other axes/types use one whole-region Metal call with threadgroup reduction. Preserve empty-domain identities and integer reduction types. No current evidence establishes every empty/axis combination. |
| Gather / take | Integer tensors with broadcast advanced indices, adjacent/nonadjacent advanced-axis placement, fixed indices, slices (including negative steps), new axes and ellipsis; negative dynamic indices wrap by dimension. | Whole-region Metal. VJP clears FP32 output then atomic-adds each selected cotangent, so duplicates sum. Dynamic indices currently make the whole source a dependency. Replace with actual indexed region dependencies; do not assume all indices are known at setup. |
| Take-along-axis | Selected axis replaced by indices axis extent; negative dynamic indices wrap. | Whole-region Metal; VJP clear then atomic-add. Broad NumPy-style broadcast support is not established merely by the graph shape rule. |
| Scatter-add | Base output plus all matching updates, preserving duplicate contributions; VJP is identity for base and gather for updates. | Metal scans the complete flattened index vector for each output scalar and accumulates float. For nonvector bases this flat-address implementation disagrees with the graph derivative’s ordinary row indexing; it is a migration bug, not a supported flat-index contract. Replace with explicit contribution domains and FP32 partial sums, independently complete by destination region. |
| Matmul | Broadcast batch dimensions, independent transpose flags; VJP uses two contractions and sum-to for broadcast axes. | Matching rank-2 shapes use `nn.linear`; batched/other cases use whole-region custom tiled Metal. Physical strides and ragged 64x32x32 kernel tails exist, but independent mesh publication is still the whole bound output. |
| Experts | Select one expert per input row, multiply selected weight matrix; input and weight VJPs. Routing stores per-expert count and actual row indices. | Whole-region route (atomic count + indices), followed by whole-region expert kernel. Weight VJP reduces routed rows per expert; input VJP scatters back to their original rows. Preserve expert identity, empty experts, repeated selections and ragged row/feature tails. Derivative outputs default FP32. |
| Neighborhood | Weighted neighbor-value sum, optionally scaled by query/key dot divided by sqrt(width). | Whole-region Metal; each observer's forward result can be independent. VJPs exist for query, keys, values and weights, never indices. Query/key/value gradients use cleared FP32 atomic sums, including repeated edges; weight gradients write disjoint entries. With gram disabled, query/key gradients remain zero. Shared indexed contractions/scatter lowering owns all variants. |
| Ordering and creation | Arange; argpartition currently emits full argsort with stable index tie-break; Philox normal generation; constants and dimensions. | Whole-region Metal or setup constants. Preserve sorted-index usefulness and deterministic key/counter mapping. Full-sort quadratic rank counting is a performance gap, not a required algorithm. Arange/random/order are nondifferentiable. |

`Graph.vjp` traverses graph nodes in reverse, accumulates cotangents, restores
owner placement and applies `sum_to` to input shapes. It does not differentiate
indices or nonfloating inputs. Additional operations forwarded through
`tensor.__getattr__` to MLX are not automatically graph-language support: for
example, `tanh` appears in the backend fast-path selection but has no explicit
symbolic dispatch or derivative branch. Such names cannot certify migration.

### Dtypes, rank, layout and placement limits

Graph float elementwise operations promote to FP32; matmul normally preserves
left dtype; constants default to FP32/int32; expert VJP graph nodes default to
FP32. Mesh allocation supports FP16, FP32, int32/uint32/int64/uint64, uint8 and bool.
The old Xonotic Metal `TYPES` table supports FP32 and the signed/unsigned 32/64-bit
integers and bool, **not FP16 or uint8**. `source(graph)` eagerly emits custom
kernels before `kernel_calls` selects shared fast paths, so the presence of an
FP16 fast-path predicate does not establish an executable FP16 Xonotic graph.
The atomic helper unconditionally treats its destination as `atomic_float`;
FP32 gradient storage is essential to that existing mechanism.

Xonotic's Metal View has eight shape/stride axes. Logical row-major strides and
actual physical strides are separate. Numerical outputs flatten leading axes
into the canonical 2D storage shape; operand NumPy views are reshaped to recover
logical axes. Arbitrary noncontiguous reshapes are not established. Physical
strides are unsigned; negative slicing is implemented as gather address
arithmetic rather than a negative-stride view. The replacement must retain rank
and strides rather than reconstruct them from shape or storage bytes.

Placement comes from graph owner/regions. Parameters belong to root owner;
`kernel_calls` copies an operand to each distinct consuming peer and retains
replicas keyed by `(operand.index, peer)`. Transposed replicas preserve backing
orientation. Copies and custom kernels currently demand whole regions on these
paths. This is neither general multi-participant topology evidence nor proof of
correct cross-block aliases. Preserve exact value/peer identities when replacing.

## Existing operational workflows and evidence

| Workflow | What it covers | Missing baseline or adaptation required |
| --- | --- | --- |
| [streaming-algebra.py](../examples/streaming-algebra.py) | FFN → RMSNorm → summed learned embeddings → FFN → RMSNorm; CPU/Metal, FP16/FP32; distinct simultaneous instances; withheld input section; numerical references; compute/transfer traces; optional Core ML. | Extend this existing executable with indexed gather/pointwise/scatter, collisions, ragged domains and derivatives. It does not currently exercise Xonotic expert/neighborhood VJPs or repeated reuse of a fixed slot pool. |
| [streaming-overlap.py](../examples/streaming-overlap.py) | CPU paired two-contraction chain, whole/streamed comparison, configurable depth and repeated reusable slots, early-consumer observations. | The numerical function is currently a Python callback with NumPy matmul and instrumentation; its timing is not a compiled-kernel baseline. Migrate its body through the shared representation before making compiled-path claims. |
| [planner/plan.py](../xonotic/planner/plan.py) | Retained mesh application: route logits → expert up → ReLU → expert down → objective projection; play/solve peers; row-tile slots reused for a timed run; returned plans and objective changes. | Reports application throughput but no float64 residual, per-region timeline or VJP observation. Each outer row tile builds a separate graph. Use this real caller for migration evidence; do not substitute the deleted persistent-policy runtime. |
| [solver.strat.measure matrix](../xonotic/solver/strat/measure.py) | Existing MLX paged AB/ATB/ABT forward/reverse and DPP residuals against float64 references; explicit shape, sample and seed arguments. | Uses MLX `paged_matrix`, not `tensor_metal.kernel_calls`, so it is a local numerical comparator only. Its stored variance is population variance, not the sample variance required for percentage claims. No expert/neighborhood mesh baseline is provided here. |

The [FP32 accumulation report](fp32-accumulation-2026-09-13.md) retains exact
cancellation/partial-overflow results, FP16/FP32 gold errors and limits of Core ML
backing identity evidence. The [pipeline report](literature-pipeline-2026-09-13.md)
retains matched tiny-FP32 CPU pipeline observations and raw traces. Neither
establishes no regression for a different tile shape, Xonotic operation, dtype
or SoC/backend selection. Setup, numerical invocation and observation/reference
work remain separate in every replacement measurement.

The eager MLX path remains visible in [matmul.py](../xonotic/solver/strat/matmul.py)
and [paged_matrix.py](../xonotic/solver/strat/paged_matrix.py): page gathers,
scatters, projection/contraction, experts, batched products and neighborhood
forward/VJPs. Symbolic inputs redirect supported families into `tensor.py`.
These are retained numerical functions/comparators, not an alternative mesh
binding. Preserve their externally observable mathematical behavior while
removing duplicate mesh emitters; do not equate an MLX-only timing with the
canonical registered-page path.

Step 1 source ownership is recorded here. Operational baseline coverage for
indexed/derivative/ragged/alias cases remains incomplete and must accompany each
replacement. The nine-step goal remains open until those observations and the
remaining implementation contracts are satisfied.
