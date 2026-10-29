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

The 1–4 integration pass on 2026-10-28 has these boundaries:

1. **Composition.** FX preserves each product's scale operand, input basis and
   output position. Compatible products share launches, and a folded matrix
   transform is emitted once across its destination batches. Native transforms
   can write a declared slab view in one launch, including its capacity gaps;
   they do not repeat the rotation or normalization for each output batch.
   Native `StreamedProducts` retains each matrix's code storage, basis, scales
   and logical output domain. `StreamedMatrix.coalesced` can combine compatible
   products; a different code buffer, basis or slab count leaves the independent
   products intact. Cuts of an existing combined matrix preserve its slab
   layout. They do not invent additional input slabs.
   Native FFN and head consumers take the resulting `ScaledReduction` operands
   without a join copy. Independent head inputs retain their own reduction
   counts, strides and origins. Cache fill selects only key/value products and
   omits row-output stores in both fused and standalone consumers.
   Independent products still meet at one standalone consumer command, whose
   accesses cover all its inputs. Per-product/tile consumer fusion remains
   unfinished; this pass does not establish equal latency for that case.
   FX gathers, small rotations and Hadamard transforms now declare additional
   destination/scale pairs through the shared kernel interface. Preparation
   expands their stores in the existing numerical body. Each scale remains a
   tensor input, with its own view and dependencies; no scale packing or cached
   values replace it. The encoding manifest names occupied bindings and the
   device operand supplies binding capacity. Groups which fit those bindings
   share one transform. For larger rotated/Hadamard groups, that command also
   writes its unscaled float result into preassigned pages. The remaining
   destinations use the existing typed gather/scaling body to consume it. The
   basis transform occurs once, and no half-width rounding is inserted before
   those consumers apply their scales. Plain gathers need only batches of
   their ordinary stores; they have no basis transform to repeat.
2. **Constraints.** `TensorRequirement` now resolves both native streamed and
   Core ML operands. The feature declaration supplies type and layout, and the
   same resolver places storage and emits any required tiled conversion. Native
   normalization and convolution already receive the consumer requirements
   before emission; other native producers still bind assigned buffers.
   Conflicting requirements need cost-directed placement beyond feasibility.
   Hadamard widths now lower through `MetalButterflyPlan`, shared by FX and
   native preparation. Device storage limits each local factor, not the whole
   operation. The loader checks the basis's mathematical shape; the former
   small-width list, mixing cap and thread-count divisibility are removed.
   `MetalHadamardLocal` selects the existing fused schedule when its configured
   workspace fits. Other shapes compose butterfly factors and a matrix product.
   Both use the same sign, scale, order and page operands.
3. **Adapters.** Stored Q/K/V and gate/up products now use `storedProducts` for
   normalization, product selection and destination binding. The gate/up helper
   adds its declared activation. The old `qkvCommands`, `ModelFeature.resolve`,
   `TensorView.resolved` and pair-only layout resolver are removed. The forced
   `StreamedMatrix.stacked` constructor and attention's separate stack dispatcher
   are also removed. Product input transforms use `streamedCommands` for every
   row class. Normalization is an explicit use operand, not a property baked
   into stored weights. Preparation can fold a generation's constant gamma
   into that use's gather scales; already normalized inputs use the raw scales.
4. **Preparation.** Native rank and Goal A build. Complete FX graphs retain five
   product results through mapped gathers, rotation, Hadamard and product lowering.
   The folded graph has one matrix transform. Compatible groups of five now
   have one small rotation or Hadamard command, including when the scale and
   destination pages are mapped. At binding capacity, all destinations share
   that command; above capacity, preparation emits one basis transform and a
   float-input gather for its remaining consumers. Native slab shader specializations
   compile with capacity gaps. Standalone head specializations compile with
   independent value partials and mapped destinations; fused and standalone
   specializations also compile with row stores omitted. Both coupled product
   forms compile with unequal tile-row strides. This evidence involves
   no numerical execution, collective run, throughput result or latency result.

### Wide butterfly factors

`structured.expand` exposes the factors before FX partitioning. Native
`butterflyCommands` uses the same `MetalButterflyPlan` and shader. If a factor
has width `w` and stride `s`, its coordinates are
`outer * w * s + lane * s + inner`. The next factor changes another set of
coordinate bits. Local storage holds `w` floats, regardless of the full width.
Signs belong to the first factor and normalization to the last.
`Rotation` uses the full power-of-two factor of its declared width unless its
caller explicitly chooses a different basis block. The local tile setting no
longer changes that mathematical basis or inflates its dense mixing matrix.

The view compiler resolves those coordinates to assigned pages. Tile records
carry their reads and writes. Masked lanes do not add unrelated source-page
dependencies; an entirely padded tile has no input read. Intermediates are
ordinary float page operands. No descriptor interpreter or additional runtime
submission interface is involved.

Compressed bases retain their sign vector and mixing matrix. Loading no longer
expands the butterfly into a dense square matrix. The inverse also uses the
factors. The existing fused kernel's geometry lives in `metal.def`, and its
eligibility is shared by both frontends. Shapes outside that geometry compose
the factors with the existing matrix and gather operations.
The inverse supplies its encoded order as a prepared view. The shared gather
accepts explicit input, output and scale types, including int8 signs, and writes
its final one to three values through the same views. It no longer drops that
tail or needs an intermediate layout copy for a short logical output.

Preparation accepts a 32,768-wide transform as two factors with at most 16 KiB
of local workspace. Compressed `(mix, block)` shapes `(3, 32)`, `(24, 256)` and
`(3, 16384)` prepare with mapped operands and padded input cuts; their former
loader restrictions are gone. This is compilation evidence, not a numerical
or performance result.

The whole operation still needs a placement plan. Exposing factors does not
automatically distribute them across peers or place a logical result larger
than one node's memory. The native mixing command also retains a whole matrix
access declaration. Further distribution belongs in graph placement and the
shared product lowerer, using the same page operands and dependency relations.

### Mapped matrix operands

`matrix_operand.metal` loads the existing view's logical coordinates into Metal
cooperative registers. FX matrix products and native scalar-format products use
that shared loader. Affine FX operands retain direct tensor loads; the compiler
selects the load expression from the same view type. This adds no runtime layout
choice, whole-operand materialization, or separate dispatch. Matrix inputs and
weights now declare `indexed` as their API requirement.

Batch selection, segmented contraction domains, bias, result pages and tile
publication words keep their prepared mappings. The matrix dimensions are
compiled constants. Each segment masks its own contraction tail, so the final
local tile cannot consume lanes from the next segment.

The native stored-product geometry comes from `metal.def`. Scalar formats use
independent SIMD-group matrix tiles with register inputs. Encoded formats retain
their shared decoded input tiles. Both write accumulator values straight to
their output views; neither stages the accumulator in a shared output buffer.
Scalar weight decoding retains fp32 and bf16 values without a half conversion.
The shared index emitter describes the aligned contiguous span in the view's
own units. A four-byte run is enough for four byte values, but not for four
fp32 values. The decoder consumes that compiled span at its actual load width;
it does not guess layout or check it during execution.

Preparation on 2026-10-28 compiled mapped fp16, fp32 and bf16 products as one
command, including a segmented batched product with mapped publication words.
Public FX compilation prepared mapped forward and backward graphs. Native raw
and quantized product shaders compiled with typed inputs and mapped outputs.
No numerical commands were submitted. This removes a layout-copy requirement;
it does not establish collective placement or early publication for every native
product, and it provides no new throughput or latency measurement.

Further integration belongs at these shared lowering boundaries. Numerical
bodies contribute their work and uses, rather than separate memory systems.

### Head normalization storage

The shared `mesh_head` body no longer reserves arrays for exactly 512 values
and 512 weights. Width and launch size are compilation operands. Each lane
reads paired values through the existing views, reduces their squared sum and
writes the normalized, rotated result. An odd final channel remains in both
the norm and output. Normalization weights are read where they are used.

The register budget in `metal.def` decides whether pairs stay in registers
across the norm. Wider heads reread their input pages after that reduction.
This is a source specialization; execution neither chooses a storage method
nor allocates one. Shared storage holds one scalar per SIMD group and the
existing contributor word. Standalone and fused consumers use the same body.
The shared sum includes a final partial SIMD group. The compiled width is no
longer repeated as an independently bound runtime scalar.

Preparation on 2026-10-28 compiled mapped standalone heads from width 1 through
1,048,576 with 32 bytes of static threadgroup memory at 128 threads. Fused
producer/consumer shaders compiled with widths 37, 512 and 32,768. Native rank
and Goal A built. No numerical commands were submitted, and no performance
result is established. Paired accumulation changes floating-point summation
order; numerical execution remains to be measured after the structural pass.

This is still one local norm reduction per head. Collective placement must
expose its partial sum before a head can span nodes. The model loader's
256/512-only head filter also remains: the attention tile code still uses
truncated `D/4` and `D/8` dimensions. Removing that filter requires masked
channel-tail lowering, not just deleting the check.

### Row reductions and gathered consumers

`streamed_finish_norm_add` no longer stages a full row in shared memory.
Its width and launch size are preparation operands. Each lane keeps a bounded
set of values in registers; rows beyond that register budget reread their
prepared partial or plain inputs. The shared `streamed_row_value` retains
the existing scalar and vector partial-sum arithmetic. Shared storage holds
one reduction scalar per SIMD group, including a final partial group.

The finish writes the residual result to its assigned pages. Its normalized
outputs use cached values or those pages. Its gathered consumers read those
same pages through their prepared mappings. The dependency between the stores
and the gather needs a device-memory barrier; there is no layout-staging copy.
Slot controls still act on the stored result before these consumers read it.
The launch uses this operation's width, rather than the model-wide hidden
width. The bare kernel alias with an implicit width and launch is removed.
Gathered outputs use the shared slab view and scale view. The finish no longer
binds three separate destination/scale pairs or truncates work beyond them.
Slab capacity gaps and mappings stay in the declared views.

`streamed_row_gather` also reads its input pages directly. It sums squares when
normalization is requested, or consumes the supplied factor when one already
exists. Neither case stages the row. Its former workspace-width parameter is
removed from both native and FX declarations; logical input width remains an
explicit operand. Thus a storage budget cannot reject a large input row.

Preparation on 2026-10-28 compiled mapped finishes and gathers with widths from
1 to 1,048,576, including odd tails, partial SIMD groups, sparse gather indices
and multiple typed destinations. Fused gathers with 5 and 23 slabs retained
capacity gaps in their destination and scale views. At 1,024 threads the finish uses 128 bytes of
static shared memory. A gather that consumes an existing factor uses none.
Complete FX compressed-product graphs with multiple consumers also prepared
through the revised gather interface. Native rank and Goal A built. No
numerical commands or performance measurements were run.

These reductions still finish one local row in a threadgroup. Splitting that
row across nodes requires the graph to expose sums of squares of the finished
logical values, their total count, and the resulting factor as page operands.
Product partials must be summed before their values are squared. Removing
row-sized staging does not by itself supply that distributed reduction.

### Partial-result merge capacity

The shared `mesh_merge` no longer limits a row to 64 contributors or requires
its channel slices to fill complete SIMD groups. It reads weights in bounded
batches, using the register budget in `metal.def`. Each batch continues the
same accumulation over the row's parts. A final partial SIMD group uses its
active lanes; channel groups may cross SIMD boundaries.

Logical width and launch width are separate compilation operands. Native and
LiteRT preparation split wide outputs into channel tiles. Each tile merges
the same partial-result records into its own output channels. A row can span
many threadgroups without adding a runtime allocator, choosing another merge
body or copying the row to change its layout. Its last tile masks absent
channels. Native row offsets and LiteRT slice offsets use full address width.

The partial record still contains two scalars and exactly the logical number
of values. LiteRT's provider uses four-lane slices, so its input and output
bindings include physical tail padding; that padding does not enter the
partial records. Both adapters call the shared merge. The importer no longer
requires widths divisible by 128.

Preparation on 2026-10-28 compiled native merges from width 1 through 65,537,
contributor counts through 1,025, and partial SIMD launches. The ordinary
merge shaders use zero static threadgroup memory. The actual LiteRT importer
prepared tiled row export and merge stages, including odd widths and width
65,537; their shaders compiled. Existing vector, tile and int8 attention
producers also compiled. Native rank and Goal A built. No numerical execution
or throughput/latency result is established by this preparation.

This removes limits from the merge of already-produced partial results. It
does not yet make every producer width-independent: the native forward
attention kernels still truncate channel dimensions, and their fused merge
still needs shared weight storage sized for its contributor count. The
LiteRT partial-row adapter also still assumes a leading axis of one. That
provider layout must become an explicit lowering operand before this last
filter can be removed. Placing the partial numerical work across nodes remains
part of the graph's reduction lowering, rather than a property of this merge.

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
provider output contract remains unresolved. The installed Xcode 26.4.1 SDK's
`CoreML.framework/Headers/MLPredictionOptions.h` describes the backing as a
proposal and says models may not support caller storage. A manifest can describe
our requirement; it cannot make that provider promise. Before this path can claim
a fixed output-page contract, the binding must use an API/model contract which
actually guarantees the supplied storage. Internal provider work and dependencies
also remain opaque. No new runtime recovery or copy path has been added.

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

Sparse products use those same API requirements. Their independent input and
weight types reach the TensorOps body, and gradient stores use the requested
result type without separate casts. Affine stride constants come from the shared
layout declaration after placement. The generic kernel binder emits these along
with index expressions for every operand, so a sparse consumer no longer forces
its producer to retain the strides seen during tracing. Schedule construction
stays outside the numerical graph; its tables use the ordinary attribute importer.

Preparation on 2026-10-28 compiled sparse products and both gradients with mixed
types, changed input strides, mapped destinations and partial sparse tiles. Each
prepared one numerical command without a layout conversion. The public
`Function` interface also prepared the complete mixed-type forward/backward
graph. No numerical commands were submitted.

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

`MetalTensorTraversal` now supplies the ordered boundary and fragment calculation
to both FX and native crossings. It maps storage parts onto declared logical
intervals during preparation. The Python-only position and alias calculations
have been removed. No traversal object is read by numerical execution.

Native `MeshRankCross` records ordered storage-and-layout parts for its source and destination.
Ranks unite their boundaries with the existing graph configuration, then bind
each fragment to its original storage. Uneven gather and scatter counts retain
their rank domains. Fragments of one crossing stay in one transport group.
Contiguous bindings still make one call; mapped bindings add no layout copy.
The native FFN passes its prepared result view to this interface. Goal A and
LiteRT declare their existing linear operands through the same binding.

Preparation on 2026-10-28 built the shared library, frontend, native rank and
Goal A. FX collective preparation covered mapped scalars and structured rows.
Native configuration preparation merged and bound mapped reductions, uneven
gathers/scatters and broadcasts without a communicator or numerical invocation.

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

FFN preparation now carries its typed result into the layer finish. A local
provider result retains its declared output type and layout. Standalone and
fused residual consumers read that operand directly, including the shared
vector reader. GPU FFN products retain the supplied destination view too.
Combining provider partials with GPU work takes an explicit addend operand;
the destination no longer implicitly supplies that input. All three views feed
the same indexing and dependency declarations.

Native FFN preparation assigns the result's type and layout from its declared
crossing, or from its local provider when no crossing consumes it. Compatible
provider layout requirements enter that same placement. Stored and compressed
GPU products write directly into this typed destination; their output interface
no longer uses a float flag or a two-case scalar selector. The native scalar
spellings for crossing types are operands in `etc/compute/metal-apis.json`.

An expert FFN's crossing covers both dense and expert contribution planes.
Those planes are views of the same placed operand. Expert combination writes
its assigned view, including a partial final SIMD group. Branch normalization
works in place and the shared typed reduction combines the planes; the separate
half scratch and add command are removed. This also corrects the preceding
native binding, which passed only the dense plane to that crossing.

Preparation on 2026-10-28 built rank and Goal A and compiled stored, compressed
and expert products with mapped half, float, bfloat and integer destinations,
including masked output tails. No numerical commands were submitted. These
changes do not establish Core ML's output-backing guarantee, which remains
unresolved as described above.

Both convolution forms and their tap projections now consume the same typed
input operand. The compressed gather, rotation and Hadamard bodies read its
declared scalar type through their prepared indices. Normalization and the
folded matrix path retain that type too. The tap projection's separate half
conversion has been removed. Coded product inputs remain the transform's
declared half outputs; this change does not remove their numerical transform.

Stored matrix products load typed vectors from that operand. Quantized tiled
products declare their decoded input storage in the shader. Scalar tiled
products load cooperative registers directly. Results go straight to their
assigned views. Host code supplies no fixed workspace byte count.
FX matrix binding likewise carries input and weight types separately, so a
folded rotation need not convert either whole operand just to match types.
Normalization weights also carry their type and view into the shared index and
dependency declarations. They no longer use a separate buffer-offset argument
or assume linear storage.

Preparation on 2026-10-28 built rank and Goal A and compiled mapped transforms,
normalization and stored products with half, float, bfloat and integer inputs.
Complete FX gather, small- and large-row rotation, and cut Hadamard graphs
prepared with float and bfloat inputs. Mixed matrix operands prepared one
numerical command. No numerical commands were submitted.

Preparation on 2026-10-28 compiled provider-result finishes at widths 37 and 64
with mapped half, float and integer inputs, and reductions with independently
typed addends. Native rank and Goal A built. No numerical commands were submitted.

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

Python and native FFNs retain each matrix's logical domain, encoded order and
input basis. Native composition retains the product list before choosing any
physical coalescing. Shared operand views compose gate and up results
with the down input. Matching projections can use the coupled numerical kernel.
The coupled kernel receives each product's tile-row stride independently in
both frontends; unequal strides no longer exclude fusion.
Separate projections use the same product and reduction bodies with their views. A numerical
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

The native drafter uses the same logical matrix cut and collective traversal. Its
products write into final column slices, and the recipient lists those slices in
contributor order. The old encoded-band view and custom gather entry point are
removed. Python preparation uses `Matrix.cut` in place of its duplicate band
slicer. The configured partition grain remains a planner operand; it is no
longer enforced as a storage-band limit in the native share parser.
Each part retains its own storage, invocation stride and instance count through
fragment binding. No part inherits another part's buffer. Native rank and Goal A
built after this change; local collective preparation carried uneven column
parts and distinct storage offsets through the shared traversal. No numerical
commands were submitted, and no performance result is claimed here.
