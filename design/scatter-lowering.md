# Shared streamed scatter-add lowering

Source review September 13, 2026, during implementation of the
[nine-step plan](pallas-lowering-plan.md). This specifies the next lowering; it is
not a claim that scatter compilation or its performance is complete.

## Algebra and existing owners

For a finite update domain U, destination map d, validity mask m and initial
operand b, the result is y[j] = b[j] + sum(u[i] for i in U if m[i] and d[i] = j).
Every valid duplicate contributes. An empty contribution set yields b[j]. Preserve
logical axes, strides, negative-index normalization and the caller's bounds policy;
do not silently substitute clipping or dropping. Distinguish the stored output
dtype from FP32 accumulation for real half/float inputs. Integer scatters need
integer arithmetic rather than conversion through float.

JAX's [scatter-add contract](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter_add.html)
makes collision and bounds semantics explicit. Its sorted/unique flags are
optimization promises, not prerequisites. Mesh must establish those facts from
retained metadata or its caller's numerical contract. FP32 reduction trees change
rounding relative to serial accumulation; require the existing numerical tolerance,
not bit identity with an arbitrary atomic arrival order.

`xonotic/solver/strat/tensor_metal.py:kernel` currently implements scatter_add by
scanning all indices for every destination: O(D*U) comparisons. Its gather_vjp and
take_along_axis_vjp use atomic updates after clearing their output. The neighborhood
VJP routes gradient contributions by source or observer and also uses atomic
updates; the custom emitter supports float32, not a general half atomic contract.
These are different physical implementations of the same indexed sum.

The replacement owner is `python/mesh/kernels.py`'s expression/lowering machinery,
invoked by `Program.kernel_call`. `tensor_metal.kernel_calls` supplies numerical
indices and expressions and stops generating these custom kernels once equivalent
lowering is present. `nn._sum` currently owns the simple static binary sum tree;
move shared reduction construction into the common lowering when reused here,
instead of copying that implementation into Xonotic or transport.

## Published implementation mechanism

Use keyed routing metadata followed by segmented reduction. Blelloch's
[Prefix Sums and Their Applications, sections 1.3 and 1.5](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
derives stable radix partitioning from scans and independent reductions from
segment boundaries. This supplies work-efficient grouping without destination by
update rescans. NVIDIA's maintained
[CUB DeviceReduce::ReduceByKey](https://raw.githubusercontent.com/NVIDIA/cccl/main/cub/cub/device/device_reduce.cuh)
is a concrete implementation reference for separate keys, input values and reduced
runs. These references specify algorithms; CUDA code is not an Apple backend.

For each independently available index chunk, retain tuples `(destination key,
original update ordinal)`. Stable radix grouping moves these tuples, not operand
rows. Segment boundaries retain `(key, begin, end)` into that ordinal vector.
Values stay in their original canonical pages and compiled numerical code reads
them by ordinal. For vector updates, group the destination row once and preserve
the feature-axis mapping; do not duplicate the same routing work per feature.
Already grouped indices use their existing boundaries directly.

A segment's value computation consumes only its selected source pages and writes
an FP32 partial under an explicit `(invocation, producer chunk, destination,
feature region)` identity. It may fuse the gathered pointwise transform into that
reduction. Its completion publishes that partial immediately through existing page
publication and transfer bindings. Distinct producers never race on a shared half
or float output. No floating atomic capability is required.

Merge the sparse per-chunk key/segment directories using ordinary sorted merging
or radix grouping of their retained descriptors. The resulting reverse directory
maps each destination to the exact partial ordinals contributing to it. This work
reads routing metadata, not update values. Allocate tuple, boundary and partial
capacity from the configured update-domain bound; actual runs use a validity/count
field as numerical data. Do not allocate a dense producer-count by destination-size
array. Worst-case partial count is bounded by valid update count. Canonical pages
remain the storage for these operands and partials; sort scratch is setup allocated.
A stable grouping costs O(w*U) work for fixed-width w-bit keys; segmented numerical
work is proportional to update values plus output size, rather than D*U.

## What makes a destination complete

Routing metadata and update payload readiness are distinct dependencies. A final
destination region needs its initial operand region, the routing descriptors for
all index chunks whose declared domain can intersect it, and the partial values
named by those descriptors. Empty segments select no value pages. A late payload
for a descriptor naming another destination does not delay this destination.
Partial producers do not depend on the merged reverse directory and can publish
while other indices are still arriving.

If an absent index chunk is allowed to name any destination, its absence cannot
prove that a particular destination has no further contributions. Completing the
final sum requires that chunk's indices, or a valid static domain restriction.
No algorithm or Pallas primitive removes this algebraic dependency. A known
per-destination contribution domain permits earlier finalization; retain it rather
than reconstructing it from arrival order. An intermediate partial is explicitly a
contribution, never a completed y[j] stamp. Nonlinear consumers use completed sums;
linear consumers can consume and propagate separate contributions when their
algebra and precision contract allow distributivity.

Fit this into the selected-reader extension in `mesh_algebra_indexed` and
`mesh-dataflow.c`: routing selectors are canonical numerical inputs, each selects
from setup-bound candidate page maps, and existing stamps determine execution.
Selection storage must survive until all dependents resolve and consume it;
selected source holds survive until completion. Unselected candidates retire only
after selection is known. The compiler must retain both original ordinals and
physical candidate identities. No host graph rebuilding, progress counter,
acknowledgement protocol or application scheduler is needed.

## Concrete implementation order and evidence

1. Add a typed indexed-add store/reduction descriptor to the existing expression
   representation, including its finite contribution domain and accumulator dtype.
   Retain rank/axes and masks; lower all output stores independently.
2. Add compiled stable key/ordinal grouping, segment boundary production and
   segmented reduction to the existing CPU/Metal source owner. These are numerical
   kernels through kernel_call, not Python numerical callbacks. Compile and bind
   all buffers, bounded grids and functions during setup.
3. Use selected-reader bindings for segment source pages and destination partials.
   Build the reverse directory as numerical metadata. Document any needed extension
   to selector layout before implementation; the current scalar expression tree
   does not yet encode segmented variable-length reductions or indexed stores.
4. Migrate scatter_add and gather/take VJPs first. Neighborhood key/value gradients
   then produce edge contributions to the same indexed sum; observer gradients
   use their known observer domains. Delete corresponding custom atomic/clear and
   destination-scan branches only after all their supported shapes/dtypes migrate.
5. Extend the existing streaming-algebra and Xonotic operational workflows with
   duplicate destinations across producer chunks, masks, ragged tails, empty sums,
   delayed unrelated payloads, and separately delayed routing metadata. Inspect
   source dependencies and actual pointers. Compare first partial publication,
   first complete destination, numerical error, memory, launches and throughput
   against matching existing paths. Record count/mean/sample variance for every
   percentage; no new evaluator is required.

Sorting, metadata passes and launch costs are real. Grouping alone establishes
better asymptotic work than the current scatter scan, not no-overhead performance.
Setup-known routing should be realized once; changing indices use the compiled
metadata kernels. Fusion can remove an internal partial store only when no
independent reader or remote publication requires it. Keep that distinction
through the [Pallas-style fusion and storage plan](pallas-lowering-plan.md#5-fuse-work-without-erasing-observable-progress).

## First executable vector lowering

The expression spelling is:

```python
base, destination, update, valid = kernels.arguments(4)
body = kernels.expression(kernels.indexed_add(base, destination, update, mask=valid))
program.kernel_call(body, grid=output_grid,
    in_specs=(BlockSpec(None),) * 4,
    out_specs=output_spec,
    out_shape=ShapeDtypeStruct(base_shape, dtype))(*operands)
```

The first lowering supports U×1 destination/validity rows, U×F update rows and a
D×F base/output. The first three operands are references. A pointwise producer can
supply transformed updates; fusing that producer into segmented accumulation is
remaining compiler work. Explicit output regions must fit the existing base and
update feature backings. ND flattening and axis adaptation must preserve actual
logical views; this implementation does not hide a dense conversion.

Each independent routing chunk sorts keys and original update ordinals with
stable radix grouping. Chunk rows align with the source update backing, so a
partial's candidate list names that backing, rather than repeating every source
page as a potential reader. Its grouped candidate ordinals and per-segment
begin/end range are slices of the same canonical metadata output. Every segment
writes a separately publishable partial. The merged reverse directory contains
at most U partial ordinals, sorted by destination key. Each final destination
region has only a two-element range selector into that shared directory, and
its source dependency is precisely the selected partials, not every partial.
Metadata comparisons cost O(U) per fixed-width radix pass plus logarithmic
range searches; numerical value loops visit segment contributions and output
values, not every destination/update pair.

Sorting moves routing tuples, not update vectors. All grouping scratch is in the
setup-allocated canonical metadata output. Real inputs use FP32 segment partials
and final accumulation before casting to the declared output dtype. Integer
inputs use integer accumulation. Masked updates become absent keys, including
unused segment slots; empty destinations retain their base. Unmasked indices
must be valid after one negative-index normalization, matching the documented
in-bounds contract rather than inventing a clipping policy.

Physical storage is deliberately explicit: reserving up to U segment partials
with one publication block each can cost U times the canonical allocation
quantum per feature stripe, even when very few segments are nonempty. Candidate
binding vectors and indexed-reader metadata for final destination regions also
retain their worst-case candidate domains; their storage is not certified linear
in U across arbitrarily many destination regions. Compact shared candidate
bindings and lifetime/storage planning remain work. There is no dense D×U
numerical partial array or D×U copied selector array, but that alone does not
prove acceptable total memory. Empty segment slots also incur launches. This
initial executable lowering must not be presented as production-capacity or
zero-overhead evidence; storage/launch packing and measured shape choices are
remaining steps of the nine-point plan.

Python source compilation passed for this increment. No numerical workload or
bridge restart was performed by the implementing agent. Parent integration must
compile the generated CPU/Metal sources and exercise duplicate destinations,
masked rows, delayed update chunks, repeated changing indices and output
precision using the existing operational examples before asserting numerical or
progress results.

Dynamic destination validity is evaluated in the original index dtype before
conversion into the bounded U32 routing representation. Negative indices are
normalized against the destination extent; values still outside the extent become
the absent sentinel. This prevents an out-of-range 64-bit index from wrapping into
a valid destination during metadata narrowing. The validity mask is numerical
scatter semantics and does not add a tensor-wide readiness condition.
