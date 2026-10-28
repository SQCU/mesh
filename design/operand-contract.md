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

- Producer placement handles native-kernel subdomains, matrix API uses and
  declared CPU/Core ML calls in FX graphs. Native normalization and convolution
  producers receive ANE input type and layout requirements before command emission. Other native
  producers still bind already assigned buffers. Conflicting requirements
  also need cost-directed placement and conversion
  choices, beyond the current placement feasibility solver.
- Foreign providers still need all internal views and dependencies exposed.
Further integration belongs at these shared lowering boundaries. Numerical
bodies contribute their work and uses, rather than separate memory systems.

CPU contraction workspace is now part of that boundary. Preparation asks BNNS
for its required byte count, assigns bridge pages and binds their address.
The host record retains those pages and declares their writes alongside its
other operands. Its sequential instructions share one workspace sized for their
largest declared need. The binding keeps instruction descriptors in private
allocation. BNNS receives a non-null workspace even when its size query returns
zero; this prevents its optional internal allocation path.

This does not split a CPU function or a set of ANE models into additional host
callbacks. The current scheduler creates a shared-event notification for each
host record. Exposing internal declarations must not introduce such boundaries
between instructions that can remain in one call.

Core ML's public `MLPredictionOptions.outputBackings` contract still permits the
provider to decline caller storage. The binding's current result-identity check
reports that condition; it does not establish an AOT guarantee. A stronger
provider output contract remains unresolved.

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
establish an optimal placement. Those requirements now also reach graph
placement through declared schema operands.

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
chosen relative positions, not a minimum-cost selection among conflicting
demands. Native normalization builders use whole-operand requirements to place
their outputs. They do not yet provide a complete graph of subview equations
for all native producers.

## API operand declarations

`operand.graph_uses` supplies native-kernel and matrix uses to placement. Matrix
schema names, input/output roles and weight permutations are declared in
`etc/compute/metal-apis.json`. Placement and numerical binding consume the same
API layout requirements. Batched products and lists of linear products retain
each operand's own view.

An API accepting strides does not require the original stride values. Placement
first solves the exact layout demands, then checks flexible uses against the
result. Compatible flexible uses keep that layout. Other uses contribute a
concrete accepted target to another equation solve. This is preparation-time
candidate construction, not a proof that every possible strided placement has
been searched. Failed joint placement leaves conversion edges in the ordinary
layout lowering.

`layout.bindings` resolves named operands for matrix, CPU and Core ML bindings.
CPU matrix layouts declare `positive_strided`, matching Accelerate's accepted
representation. Mapped or reversed selections reach that API through the same
prepared copy records used by other consumers. CPU and Core ML share host
address binding; neither interprets a view during invocation. BLAS binding uses
its actual leading-dimension requirements; other positive-stride operands use
the existing BNNS contraction.

The resolver returns `TensorView` for both retained and allocated storage.
Singleton axes can receive positive stride metadata on the same pages. Python
and Swift retain that metadata without copying values. Required allocations
also support negative fixed strides, without asking a frontend tensor handle
to represent them.

Preparation on 2026-10-27 compiled a producer shared by an exact-layout kernel
and a strided matrix consumer. The exact consumer recorded only its numerical
command, with no conversion. CPU bindings prepared mapped input/output views,
overlapping positive-stride reads, and singleton views. Singleton normalization
recorded no copy. No numerical commands were submitted.

## Host calls in the complete graph

`host.declare` records named CPU or Core ML uses in the FX graph. Its operands
have the same schema as native GPU kernels: source storage, logical use, output
slots and a list of layout requirements. A use requirement and an API binding
requirement remain separate constraints through placement. Output storage can
serve several declared use slots. The declaration reads no payload values.

`BoundGraph` partitions host calls at the same boundary as collective calls.
It places producers before binding either kind of call. Numerical spans keep
their original FakeTensor metadata and assigned destinations. Host preparation
returns the existing native host record. The graph operator, provider choice,
and layout descriptions do not survive into invocation. The complete compiler
configuration also scopes required host/collective conversion commands.

`engines.Prepared` now captures its casts, engine shares and combinations in
that graph. The separate cast prologue and GPU/host/combine lists are removed.
Concatenated shares receive their final output slices as destinations. CPU and
Core ML definitions carry explicit names, types and layout operands; Core ML
compilation and model loading remain preparation work. The same declaration
also accepts arbitrary named Core ML graph inputs and outputs, apart from the
linear/FFN convenience builders.

Preparation on 2026-10-27 compiled a GPU producer whose middle columns feed
both a contiguous host use and a matrix product. The producer's assigned pages
give the host its contiguous view directly. The ordinary engine projection
and FFN callers prepared native CPU bindings through this graph path. No
recorded numerical commands were submitted. Core ML's provider-internal
dependencies and output-backing contract remain separate unresolved work.

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

## Native numerical operands

Native normalization, GELU, stored products, fused stacks, streamed input
transforms and streamed output placement consume `TensorView`. Their old
buffer/offset/row-pitch argument combinations have been removed at these
boundaries. Both shader indexing and dependency ranges come from the supplied
view. Gathered and strided views use the same type as contiguous views. Physical
vector loads remain available where the compiled relation permits them.

This also fixes the native GELU adapter's stale pitch arguments. Its shader
already used logical indices, while the adapter still passed pitch fields that
the shader ignored. Expert gate/up slices and per-layer inputs now carry those
pitches in the same view used to emit their accesses.

`PrefillPartner.inputRequirement` exposes the model's element type, element
width and provider layouts before binding. The FFN normalization producer and
Goal A use this declaration to assign their destination pages. `TensorOperand`
carries the resulting view and type through producer emission and model binding.
`TensorView.placed` assigns an unwritten result; it emits no copy. The provider's
resolver handles inputs whose producers have already fixed their placement.

The ordinary normalizer and fused residual finish can publish several typed
destinations from one calculation. Compatible GPU and host consumers share the
same destination. Different element types get direct stores into their declared
pages; there is no intermediate half result merely to convert it for the host.
Preparation expands the stores and their accesses. Replay receives no output
list, type switch or layout interpreter. Only consumers using raw values need
the separate row-factor output. Stored and large rotated products consume their
normalized values, so an otherwise unused factor is not allocated or written.

Compressed product kernels retain their declared slab layout, including gaps
between each slab's active rows. A transformed input carries that layout as a
view. The common resolver supplies a compiled conversion if a supplied view
does not meet it. Gather and rotation producers accept indexed source views
directly. Folded gamma scales retain their existing numerical meaning; this
relation also matters when a normalizer has more than one consumer.

The native normalizer can publish its RMS factor as a float view, one element
per row. Mixed streamed/ANE preparation gives the host its normalized operand
and gives the input transform that factor view. The transform keeps its raw
input and folded gamma scales, so it applies gamma once and avoids another RMS
reduction. Gather, int8 rotation and Hadamard lowering all consume the same
factor operand. The folded-matrix path can apply a supplied factor through the
same normalizer. Large rotated FFNs already share their normalized input.

Factor reads and writes use the same view declarations as other data. Their
presence is specialized during preparation. The direct gather needs no
threadgroup staging for normalization. Hadamard still orders the butterfly
values and sign masks before their shared-memory consumers; that dependency
does not disappear with the RMS reduction.

Native FFN preparation selects the GPU share, host share and input destinations
before the preceding producer emits commands. Its fused residual finish can
write the host's normalized values, the row factors and the GPU's gathered
inputs together. The GPU gather uses the selected share's matrix. Host presence
does not disable it. Both consumers retain the producer's views; neither
reconstructs a destination from a readiness flag. An intervening convolution or
collective prevents that earlier producer from supplying the FFN input.

An input convolution consumes the normalized residual and writes the FFN's
declared destinations itself. The partner's type and layout requirements reach
this producer too. Compatible GPU and host uses share its result; different
types receive direct stores from the same calculation. The FFN retains those
views instead of rebuilding a half tensor from the network's buffer name.
No post-convolution copy is needed merely to satisfy that partner declaration.

Both convolution forms now use the common numerical helper and shared view
indexing. Inputs, taps, bias, normalization weights, residual and optional scale
carry the same layout into their shader bindings and dependency records. The
standalone producer uses the existing typed-output emitter; the normalized
finish preserves fp32 arithmetic through its final residual store. Launch width
comes from the configured pointwise width and the compiled pipeline limit.

Preparation on 2026-10-28 built native rank and Goal A and compiled convolution
producers with half, float and integer destinations, mapped inputs and strided
outputs. Both scaled and unscaled finishes compiled with mapped views. No
numerical work or performance measurement ran.

Residual normalization now consumes typed views for its input, normalization
weights, residual, destination and optional scale. Its shared numerical template
uses each view's prepared indices. Goal A uses this builder instead of its own
dispatch and byte-count declarations. Dense and vision callers use the same
interface. Destination reads within the calculation do not add an input
dependency on the old destination contents.

Scaling, residual normalization and fused finishes also share the declarations
of their control operands. Words, control tables, vectors, gains and statistic
outputs retain their views in both shader addressing and dependency records.
Control values remain runtime page data. Preparation on 2026-10-28 built rank
and Goal A and compiled typed residuals with mapped and strided operands,
including controls without a scale. No numerical commands were submitted.

Layer finishes return the gathered operand they actually produce. The next
attention product consumes that view through the shared input resolver. A finish
that emits no gather returns no such operand. Stored projections also consume
already-normalized convolution results through their ordinary product interface.
FFN activations have their own prepared pages: fitting inside the residual's
allocation does not make that still-needed residual available for reuse.

The shared gather helper takes typed operands from its caller. It no longer
assumes that every caller uses the standalone gather's buffer numbers. The
fused finish carries views for its residual, plain input, normalization outputs
and gathered destinations. Its width is a preparation operand; scalar tails
use the same lowering. Its definition and row controls live in the native
template prelude, so masked specializations can be compiled from that prelude.

Preparation on 2026-10-27 built rank and Goal A and compiled stored product,
expert, fused stack, normalization, gather, rotation and output pipelines with
strided and gathered views. Literal product pipelines also compiled. These
checks submitted no numerical work and establish no performance result.

On 2026-10-28, native builds and shader compilation also covered factor
production and consumption with strided page views, including masked tails.
The native packed-rotation caller now uses the existing specialization instead
of emitting a duplicate explicit template instantiation. No numerical work ran.

The same day's fused handoff changes built rank and Goal A and compiled from the
native instance prelude at logical widths 64 and 37. Preparation covered masked
gathers, strided normalization outputs and literal plain/partial reductions.
This was source and pipeline preparation only; no numerical commands ran.

The typed-output change (`metal-microbench` commit `d2ef9be7`) on 2026-10-28 built rank and Goal A and compiled
half, float and integer normalization destinations, including mixed destinations,
strided views and scalar tails. It also compiled the fused finish with and
without a factor output. These checks submitted no numerical work. Core ML
float64 declarations still require lowering to a provider that supports that
arithmetic; this native Metal boundary has no float64 lowering.

## Compressed projections

Native and Python FFNs retain each matrix's logical domain and encoded order.
Shared operand views compose the gate and up outputs with the down input.
Matching projections can use the coupled numerical kernel. Other projections
use the same product and reduction bodies with their declared views. A numerical
basis transform remains work; a permutation alone adds no copy.

Native `ScaledReduction` carries logical partials, scales and the result's
origin. It replaces `OrderedReduction` and its identity-order buffer. Head
consumers, fused normalization and per-layer products no longer load that
identity or scatter through it. FX finishes compose the encoded positions into
partial and scale views during preparation. Padded lanes leave the finish
domain, and the common finish kernel writes its logical output view directly.

Direct products, row gathers, Hadamard transforms, dense decoding and padded
FFN activations carry their encoding positions through preparation. The FX
kernel declaration and native library receive these positions
as compiler metadata. Both use the shared index emitter, which simplifies the
relation to arithmetic or a compact constant table. There is no order-buffer
binding or read dependency for these positions. A required encoding relation has
no implicit identity implementation; preparation supplies identity relations too.
Computed index tensors remain ordinary page operands.

The Hadamard input domain holds its source positions as preparation metadata.
Cuts compose that domain with the selected logical columns. Identity and cut
domains use the same relation emitter; neither allocates a separate source-order
buffer. Dense decoding likewise declares both axis relations, and padded
activations declare which encoded lanes hold values. Masks retain their numerical
meaning while the order-buffer loads and bindings disappear.

The native fused finish uses that same declaration for its next input gather.
Its output count is a template operand, so an absent gather adds no runtime
branch or unused encoding input. Each gather is prepared with its actual relation
and pipeline limits; bare aliases with missing relations have been removed.

Preparation on 2026-10-28 covered a complete FX compressed gather/product graph
with 29 logical output rows and 37 input columns in 64-lane storage. Its commands
have no order tensors among their inputs. Shared-library, frontend and native
builds completed, and direct products, gathers and typed fused finishes compiled.
These checks submitted no numerical work.

The same preparation covered complete dense-decoding, cut Hadamard and padded
FFN graphs, including different gate/up orders. Their runtime kernel inputs
contain no encoding-order tensors. Native builds and masked/unmasked shader
specializations completed without submitting numerical commands.

Fused consumers carry the same views as standalone consumers. Their index
expressions have a compiler namespace separate from the enclosing product's
bindings. This namespace introduces no extra buffer or runtime lookup. Per-layer
activation placement consumes prepared destination and scale views; its inverse
column table is no longer a shader operand. Head normalization and publication
also apply their declared views. Coherent output stores use the shared mapped
store helper before the existing publication operation.

Preparation on 2026-10-28 compiled standalone and fused consumers with mapped
partials, strided destinations and non-unit scale strides. An FX finish with
37 logical rows in 64 encoded lanes prepared one numerical command into a
strided destination, with no order-buffer input or conversion command. Native
rank and Goal A builds completed. No numerical commands were submitted.

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
