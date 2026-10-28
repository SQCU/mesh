# Tensor operands through compilation

Established 2026-10-27. This records the shared lowering contract and remaining
integration work. It does not change the data-model document.

## Pages hold the data

The bridge supplies the pages. Inputs, outputs, temporary values, weights and
index lists use those pages. A tensor handle is a frontend view of that storage.
It does not authorize another allocator or a transport staging copy.

`ncclMeshPageTableCreate` binds existing pages during preparation. Compute can
use it without a communicator; transport uses the same table with one.
`ncclMeshPageTableBinding` supplies the local backend view of those pages. It
allocates no payload storage. `MetalTensorPages` holds the table for the compiler.
The bridge supplies its page count and page size.

The existing transport boundary uses `ncclMeshPageMask`: a page number and a byte selection
confined to that page. Selections retain logical order and repeated source
reads. The union of accessed ranges serves dependency analysis; it cannot
replace an ordered message.

## Known access structure, changing contents

A fixed shape does not make values constant. The same holds for a page index
list. A sparsity function can produce a known-width list of indices and a later
function can read it. Mixture of experts, sparse attention and indexed memory
retrieval need this property.

Compilation knows the list's type, width, storage domain, producer and consumers.
Its contents may change on each call. It remains an ordinary data operand with
read and write dependencies. The numerical body loads the needed indices and
then the selected page contents. Preparation must not read those values and
freeze them into generated source.

Constants and computed indices have the same operand type and lowering.
A producer writes index values into pages, and its consumers read those pages.
Constant folding can simplify the computation without changing that interface.
There is no separate static-selection API or dynamic-layout evaluator. The
access relation includes the selectable domain and its producer dependencies.

## One layout algebra

`rdma/mesh-tensor-layout.h` defines `MetalTensorLayout`. Native `TensorView`
and Python `operand.TensorView` use this preparation type. Logical shape is
distinct from an allocation's capacity or an encoding block. Shape, element
type, storage placement and access relations travel together through lowering.

Strides, axis maps and joint maps describe the same kind of view. Slice,
selection, permutation, reshape and gather compose those relations before
execution. Results retain no parent descriptors. Component views subdivide
typed elements for byte-preserving copies. Known maps simplify to arithmetic
and compact index expressions where possible.

These maps are compiler metadata. Their constancy says nothing about values
in the tensor, including values used as indices. Both static and dynamic
indexing need declared dependencies. A scale, rotation or reduction remains
numerical work; a layout change cannot stand in for it.

## Place values for their consumers

Each use declares its type and layout requirements. Indexed consumers can use
gathered pages. An API needing contiguous storage receives actual contiguous
storage. A view alone cannot satisfy that physical requirement.

The shared `required` operation resolves layouts without allocating or copying.
Preparation first uses compatible destinations, including placing producers
directly in the pages their consumers require. Necessary conversions become
tiled work over preassigned pages. Their reads and writes expose overlap with
independent work. Callers supply no separate packing or scheduling algorithm.

Both frontend command adapters derive accesses from the layout used for
numerical indexing. Host operations use the same dependency compiler as GPU
work. Allocation size cannot stand in for a declared use.

## Execution consumes prepared page operands

The backend records bindings, numerical work and dependencies, retaining the
pages through execution. Invocation does not construct TensorViews, match
layouts, allocate storage or generate source.

It still loads data, including changing page indices where needed. It does not
rebuild layout descriptions or walk intermediate views. The lower compiler
sees ordinary loads, numerical operations and stores with declared types and
access structure.

## Remaining integration

This contract does not declare these existing paths complete:

- Some fused consumers load order arrays whose static compositions could be
  resolved earlier. Numerical index tensors must remain data.
- Producer placement now handles compatible proper subdomains declared by
  native kernels. Conflicting requirements, already assigned storage, and API
  requirements not yet carried into that graph still need broader integration.
- Foreign providers still need all internal views and dependencies exposed.
Further integration belongs at these shared lowering boundaries. Numerical
bodies contribute their work and uses, rather than separate memory systems.

## Frontend attributes become page operands

The compiler lifts tensor attributes before splitting a function around its
collectives. Every numerical span then receives the same bound operands. This
preserves aliases and state shared across those spans.

Attributes already in bridge pages keep their storage. Foreign attributes are
imported once during preparation. Aliases of one source allocation share one
destination allocation, with their shapes, strides and relative byte positions
intact. Only referenced source ranges are copied. Foreign MPS setup work is
completed before that host copy reads its initial values; page-backed attributes
need no import wait or copy. Neither action is part of recorded invocation.

Imported contents remain ordinary data. Compiled work can read and write them
through its declared page operands. Lifting an attribute does not turn its
contents, such as indices, into constant compiler metadata.

## FX placement and numerical bindings

FX aliases now use the shared layout algebra. The separate affine rebasing
implementation has been removed. The graph keeps its logical index expressions;
Metal source emission composes them with the assigned page selections. The
command recorder receives those same selections for accesses. This works for
ordinary and gathered operands, including values read as indices. Execution
does not walk a layout object or choose an indexing implementation.

Top-k, scans, sampling, FWHT and matrix outputs use the same index emitter.
Matrix inputs carry the current MPP consumer's strided requirement in
`etc/compute/metal-apis.json`. A supplied gathered input that does not meet that
requirement gets a prepared conversion edge. That conversion alone does not
establish an optimal placement; the requirement must also reach graph placement.

`orderedSpans` retains logical traversal and repeated reads for transport.
It is distinct from the union of ranges used for dependencies. WRITE preparation
uses it for gathered operands and keeps compact row descriptions for affine
ones. Both descriptions enter the existing physical WRITE compiler.

Preparation on 2026-10-27 compiled gathered loads and stores, a placed producer
with alias and reduction consumers, and the numerical bindings listed above.
The public FX frontend also accepted a prepared gathered operand. These were
command and pipeline construction only; no numerical execution or performance
measurement was performed.

## Subview placement equations

`MetalTensorPlace` is shared preparation code in `mesh-tensor-layout`. Each use
supplies its source selection and the relative positions its consumer requires.
The solver joins those position equations, preserves each typed element's
contiguous fields and alignment, and lays out the independent groups. Its result
is an ordinary `MetalTensorLayout`, used by the same index and access emitters.
The equation graph does not survive into execution.

The FX placement pass now includes declared subviews and producer writes. It
binds the whole underlying storage relation, so unconstrained siblings and
differently typed aliases retain their data. Common scalar units avoid expanding
equal-width values into bytes. Collective destination assignments are supplied
before layout placement, as are caller-owned inputs and outputs.

Contradictory distances, incompatible alignments, or distinct values forced into
one slot mean that the demands cannot share one placement. The existing resolver
then supplies explicit conversion edges. This establishes feasibility for the
joint demands, not a minimum-cost selection among conflicting demands. Native
and foreign API declarations still need to reach this shared placement phase;
having the function in a shared header alone does not complete that integration.

Preparation on 2026-10-27 compiled a producer whose middle columns feed a
contiguous consumer while a sibling slice feeds another calculation. The
consumer recorded its numerical command with no conversion command. Wider typed
aliases retained adjacent, aligned fields. The shared library, frontend extension
and native rank/goal-a built. No numerical function was invoked.

## Collective operand binding

FX crossings, partial tensors and the bandwidth demo now use
`collective_views.lower` through `backend.prepare_fragments`. The separate
partial-tensor alias cutter and the unused schedule-level collective binder
have been removed. Each caller plans the declarations returned by binding.

The call carries its operator's `element_shape`. Scalar reductions have scalar
elements; the partials merge has whole `[m,l,O]` rows. Ordered page selections
become common logical cuts during preparation. Uneven gathers, reduce-scatters
and exchanges retain their rank counts, roles and complete exchange matrix.
An absent local contribution does not erase the declared transaction extent.

The native numerical consumer requires contiguous fields within each element,
as declared in `etc/compute/metal-apis.json`. Gaps between whole rows require
no copy. A row whose fields are scattered gets a conversion through the shared
layout compiler. Only the affected rows are materialized. Those commands use
preassigned pages and the existing native schedule. Producer placement can
still remove conversions; local binding does not establish their global
necessity or an optimal schedule.

Native collective fragments remain proportional to the common contiguous
intervals. Contiguous bindings keep their original call count. Highly scattered
selections may still produce many native records; this is a granularity limit
of the current numerical binding, not a limitation of the page representation.

On 2026-10-27, the extension built and local preparation constructed mapped
collective bindings and record-copy pipelines. Mixed-row preparation copied
only the scattered row. The inspection used no live communicator and submitted
no numerical work; distributed execution and performance remain unmeasured.

## Compressed projections

Native and Python FFNs retain each matrix's logical domain and encoded order.
Shared operand views compose the gate and up outputs with the down input.
Matching projections can use the coupled numerical kernel. Other projections
use the same product and reduction bodies with their declared views. A numerical
basis transform remains work; a permutation alone adds no copy.

The FX kernel declaration carries each binding's source view, use view, layout
requirement, and explicit output device. Preparation composes the use with the
producer's assigned storage. Several output bindings can select different parts
of one output allocation. Commands and dependency records consume those same
views.

Native `StoredMatrix` describes the bytes of existing encoded blocks with a
`TensorView`, plus their format, logical columns and first selected lane.
Stored FFNs, stacked products, experts and conversion kernels use that operand.
Resident row and column cuts compose on it without changing the stored codes.
File imports return this same operand. They retain boundary block bytes, selected
lanes and expert rows through numerical lowering. Padded storage rows do not
become logical tensor rows. The separate buffer-only importer and projection
format/offset fields have been removed. Conversion from component-byte views
to typed element views uses the same layout algebra, preserving mappings.
The shared index emitter retains grouped layouts as arithmetic when a cut starts
inside a group; it does not introduce a lookup table for that phase.

Preparation evidence on 2026-10-27: an FX FFN with distinct gate/up orders,
37 logical neurons in 64 encoded lanes, a third down-input order, and non-unit
down scales compiled to seven native command records. Its activation reads the
mapped partials and writes the down input; there is no layout-copy command.
No numerical commands were submitted. Library, frontend extension, and native
rank/goal-a builds completed. This establishes preparation coverage, not measured
throughput or numerical execution results.
